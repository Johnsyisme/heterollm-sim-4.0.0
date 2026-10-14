"""HBM activation causality across command slots and execution modes."""
from contextlib import nullcontext
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig


def configuration(**changes):
    return replace(DramConfig(
        kind="HBM", banks_per_group=1, open_ns=100, close_ns=20,
        read_latency_ns=50, write_latency_ns=0, write_recovery_ns=0,
        burst_interval_ns=1, interface_bandwidth_gb_s=64,
    ), **changes)


class HbmCoreDeepFollowupTests(unittest.TestCase):
    def test_row_hit_cannot_issue_before_activation_on_another_command_slot(self):
        for details in (True, False):
            for alias in (False, True):
                with self.subTest(details=details, alias=alias):
                    command = "shared-ca" if alias else "dram:command:0"
                    timeline = ResourceTimeline(
                        lane_available={command: [0., 0.]},
                        resource_aliases={"dram:command:0": command} if alias else {},
                    )
                    core = DramCore(configuration(), timeline, capture_details=details)
                    first, hit = core.submit_many((
                        AccessRequest("first", "write", 0, 64),
                        AccessRequest("hit", "read", 64, 64),
                    ))
                    self.assertEqual(first.completion_ns, 101)
                    self.assertEqual(hit.completion_ns, 151)  # formerly 102
                    self.assertEqual(hit.counters["row_hits"], 1)
                    self.assertEqual(hit.physical_read_bytes, 64)
                    if details:
                        command_stage = next(s for s in hit.stages if s.name == "COMMAND_RESERVATION")
                        self.assertEqual(command_stage.start_ns, 100)

    def test_each_activation_replaces_gate_but_recovery_does_not_gate_row_hits(self):
        core = DramCore(configuration(write_recovery_ns=200), ResourceTimeline(
            lane_available={"dram:command:0": [0., 0.]}))
        first = core.submit(AccessRequest("row0", "write", 0, 64))
        hit = core.submit(AccessRequest("row0-hit", "read", 64, 64))
        self.assertEqual(first.completion_ns, 301)
        self.assertEqual(hit.completion_ns, 151)
        # PRE waits for the write's recovery, ACT follows PRE. A row hit
        # cannot reuse the earlier row's 100 ns activation gate.
        row1 = core.submit(AccessRequest("row1", "write", 8192, 64))
        self.assertEqual(next(s.end_ns for s in row1.stages if s.name == "ACTIVATE"), 421)
        hit1 = core.submit(AccessRequest("row1-hit", "read", 8256, 64))
        self.assertEqual(next(s.start_ns for s in hit1.stages if s.name == "COMMAND_RESERVATION"), 421)
        self.assertEqual(hit1.completion_ns, 472)

    def test_activation_state_survives_real_numeric_and_fold_paths(self):
        for mode in ("detailed", "compiled", "compiled_batch", "fold"):
            with self.subTest(mode=mode):
                core = DramCore(configuration(read_latency_ns=1000),
                                capture_details=mode == "detailed")
                context = (patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None)
                           if mode == "fold" else nullcontext())
                with context:
                    if mode == "compiled_batch":
                        results = core.submit_many((AccessRequest("a", "write", 0, 2048),
                                                    AccessRequest("b", "write", 2048, 2048)))
                        self.assertTrue(all(r.counters.get("compiled_batch_float64") for r in results))
                    else:
                        result = core.submit(AccessRequest("a", "write", 0, 4096))
                        if mode == "compiled":
                            self.assertTrue(result.counters.get("compiled_float64"))
                        if mode == "fold":
                            self.assertGreater(result.counters["accelerated_row_hit_bursts"], 0)
                bank = next(iter(core._banks.values()))
                self.assertEqual(bank.activation_ready_ns, 100)
                self.assertEqual(bank.ready_ns, 164)
                # Adding an independent command slot must not remove the
                # bank dependency, even though this new slot has no history.
                core.timeline.lane_available["dram:command:0"].append(0.)
                hit = core.submit(AccessRequest("hit", "read", 4096, 64))
                self.assertEqual(hit.completion_ns, 1101)
                self.assertEqual(sum(core.timeline.bytes_moved.values()), 4160)

    def test_numeric_packing_retains_existing_activation_after_row_hit_only_batch(self):
        core = DramCore(configuration(read_latency_ns=1000), capture_details=False)
        core.submit(AccessRequest("scalar-act", "write", 0, 64))
        rows = core.submit_many((AccessRequest("a", "write", 64, 128),
                                 AccessRequest("b", "write", 192, 128)))
        self.assertTrue(all(r.counters.get("compiled_batch_float64") for r in rows))
        self.assertEqual(next(iter(core._banks.values())).activation_ready_ns, 100)
        core.timeline.lane_available["dram:command:0"].append(0.)
        self.assertEqual(core.submit(AccessRequest("read", "read", 0, 64)).completion_ns, 1101)

    def test_independent_bank_activation_does_not_create_global_gate(self):
        config = configuration(banks_per_group=2, channels=2)
        core = DramCore(config, ResourceTimeline(lane_available={
            "dram:command:0": [0., 0.], "dram:command:1": [0., 0.],
        }))
        first = core.submit(AccessRequest("bank0", "write", 0, 64))
        independent = core.submit(AccessRequest("other-lane", "write", 64, 64))
        self.assertEqual(first.completion_ns, independent.completion_ns)
        saved = copy.deepcopy(core)
        first_hit = core.submit(AccessRequest("hit", "read", 256, 64))
        restored_hit = saved.submit(AccessRequest("hit", "read", 256, 64))
        self.assertEqual(first_hit, restored_hit)
        self.assertEqual(core._banks, saved._banks)

    def test_runtime_snapshot_restores_activation_gate_without_aliasing_it(self):
        context = PhysicalRuntimeContext(configuration(), "hbm", capture_details=False)
        core = context.runtimes["hbm"].core
        core.submit_many((AccessRequest("a", "write", 0, 64),
                          AccessRequest("b", "write", 64, 64)))
        saved = context.snapshot()
        for _ in range(2):
            context.restore(saved)
            self.assertIs(core, context.runtimes["hbm"].core)
            core.timeline.lane_available["hbm:dram:command:0"].append(0.)
            hit = core.submit(AccessRequest("hit", "read", 128, 64))
            self.assertEqual(hit.completion_ns, 151)
            core.submit(AccessRequest("new-row", "write", 8192, 64))
            self.assertGreater(next(iter(core._banks.values())).activation_ready_ns, 100)


if __name__ == "__main__":
    unittest.main()
