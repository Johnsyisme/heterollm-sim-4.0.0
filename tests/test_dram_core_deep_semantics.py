"""DRAM alias, numerical boundary and nonempty-calendar regressions."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig
import heterollm_sim.dram_core as dram_module
import test_dram_core_semantics as reference_checks


def state(core):
    return copy.deepcopy((core._banks, core._inflight, core._acceptance_ns,
                          vars(core.timeline)), {
        id(core.timeline.resource_aliases): core.timeline.resource_aliases,
    })


class DramCoreDeepSemanticsTests(unittest.TestCase):
    def test_recovery_overflow_rejects_without_changing_live_state(self):
        for details in (True, False):
            for count in (64, 4096):
                with self.subTest(details=details, bytes=count):
                    core = DramCore(DramConfig(banks_per_group=1,
                                               write_recovery_ns=1.7e308),
                                    capture_details=details)
                    core.submit(AccessRequest("history", "read", 128, 64))
                    before = state(core)
                    with self.assertRaisesRegex(ValueError, "finite timing range"):
                        core.submit(AccessRequest("overflow", "write", 0, count, 1e308))
                    self.assertEqual(state(core), before)

    def test_extreme_read_recovery_and_batch_overflow_are_also_guarded(self):
        for operation, field in (("read", "read_recovery_ns"), ("write", "write_recovery_ns")):
            core = DramCore(DramConfig(banks_per_group=1, **{field: 1.7e308}),
                            capture_details=False)
            before = state(core)
            with self.assertRaisesRegex(ValueError, "finite timing range"):
                core.submit_many((AccessRequest("overflow", operation, 0, 4096, 1e308),
                                  AccessRequest("after", operation, 8192, 64)))
            self.assertEqual(state(core), before)

    def test_guard_allows_representable_extremes_and_immutable_alias_maps(self):
        timeline = ResourceTimeline(resource_aliases=MappingProxyType({"dram:data:0": "wire"}))
        core = DramCore(DramConfig(banks_per_group=1, write_recovery_ns=1e307),
                        timeline, capture_details=False)
        result = core.submit(AccessRequest("fits", "write", 0, 64, 1e308))
        self.assertTrue(math.isfinite(result.completion_ns))
        self.assertEqual(result.completion_ns, 1.1e308)
        self.assertEqual(result.counters["resource_bytes"], {"wire": 64})
        self.assertIs(core.timeline.resource_aliases, timeline.resource_aliases)

    def test_normal_execution_does_not_copy_core_state(self):
        core = DramCore(DramConfig(), capture_details=False)
        with patch("copy.deepcopy", side_effect=AssertionError("normal execution cloned state")):
            core.submit(AccessRequest("one", "read", 0, 8192))
            core.submit_many((AccessRequest("two", "write", 8192, 4096),
                              AccessRequest("three", "read", 16384, 64)))

    def test_fallback_mapping_size_does_not_scale_with_interleave_bytes(self):
        c = DramConfig(banks_per_group=1, rows_per_bank=1,
                       row_bytes=4 * 1024 * 1024, interleave_bytes=4 * 1024 * 1024)
        request = AccessRequest("tiny_in_large_row", "write", c.row_bytes - 4097, 4096)
        reference = DramCore(c)
        expected = reference.submit(request)
        core = DramCore(c, capture_details=False)
        sizes = []
        original = dram_module._mapping_cycle
        def observed(*args):
            table = original(*args)
            sizes.append(len(table))
            return table
        with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None), \
                patch("heterollm_sim.dram_core._mapping_cycle", side_effect=observed):
            actual = core.submit(request)
        self.assertEqual(sizes, [1])
        reference_checks.DramCoreSemanticsTests().assert_matches(reference, (expected,), core, (actual,))

    def test_large_legal_bank_geometry_avoids_full_resource_table(self):
        c = DramConfig(channels=2, banks_per_group=dram_module.MAX_ACCELERATED_BANKS // 2 + 1,
                       rows_per_bank=2, row_bytes=256)
        core = DramCore(c, capture_details=False)
        self.assertFalse(core._can_accelerate())
        with patch("heterollm_sim.dram_core._mapping_cycle", side_effect=AssertionError("full geometry table")):
            result = core.submit(AccessRequest("small", "read", 0, 4096))
        self.assertEqual(result.physical_read_bytes, 4096)
        self.assertEqual(len(core._banks), 64)

    def test_slot_specific_turnaround_is_not_skipped_by_scalar_direction(self):
        timeline = ResourceTimeline(
            lane_available={"dram:data:0": [100., 200.]},
            directions={"dram:data:0": "write"},
            lane_directions={"dram:data:0": ["read", "write"]})
        config = DramConfig(banks_per_group=1, open_ns=0, close_ns=0,
                            write_latency_ns=0, burst_interval_ns=0,
                            write_recovery_ns=0, read_to_write_ns=50,
                            interface_bandwidth_gb_s=64)
        core = DramCore(config, timeline, capture_details=False)
        result = core.submit(AccessRequest("write", "write", 0, 64))
        self.assertEqual(result.completion_ns, 151)
        self.assertEqual(timeline.lane_available["dram:data:0"], [151, 200])
        self.assertEqual(timeline.busy_ns["dram:data:0"], 1)

    def test_aliases_share_canonical_wire_but_not_physical_bank_identity(self):
        bank0, bank1 = "dram:bank:0:0:0:0:0:0", "dram:bank:0:0:0:0:0:1"
        timeline = ResourceTimeline(resource_aliases={
            "dram:data:0": "wire", "dram:command:0": "command-wire",
            bank0: "array", bank1: "array"})
        timeline.transfer("wire", 1000, 0, 1, direction="read")
        core = DramCore(DramConfig(banks_per_group=2), timeline, capture_details=False)
        self.assertFalse(core._can_accelerate())
        result = core.submit(AccessRequest("read", "read", 0, 4096))
        self.assertGreaterEqual(result.completion_ns, 1000)
        self.assertEqual(result.counters["resource_bytes"], {"wire": 4096})
        self.assertIn("command-wire", result.counters["resource_busy_ns"])
        self.assertEqual(set(core._banks), {bank0, bank1})
        self.assertEqual(result.counters["row_misses"], 2)

    def test_unrelated_alias_does_not_disable_acceleration(self):
        timeline = ResourceTimeline(resource_aliases={"gpu:logical": "gpu:physical"})
        core = DramCore(DramConfig(), timeline, capture_details=False)
        self.assertTrue(core._can_accelerate())
        result = core.submit(AccessRequest("native", "read", 0, 8192))
        self.assertTrue(result.counters.get("compiled_float64"))

    def test_nonstandard_geometry_and_nonempty_history_match_all_paths(self):
        # Non-power-of-two burst/row/bank dimensions, multiple ranks/dies,
        # independent preexisting reservations and both direction changes.
        c = DramConfig(channels=3, ranks_per_channel=2, bank_groups_per_rank=2,
                       banks_per_group=3, stacks=2, dies_per_stack=2,
                       burst_bytes=48, row_bytes=768, rows_per_bank=4,
                       interleave_bytes=192, interface_bandwidth_gb_s=65.25,
                       read_to_write_ns=13., write_to_read_ns=37.,
                       read_recovery_ns=9., write_recovery_ns=15.,
                       burst_interval_ns=2.3, max_outstanding_requests=2)
        history = (AccessRequest("old_read", "read", 0, 97),
                   AccessRequest("old_write", "write", c.capacity_bytes // 2 + 3, 200),
                   AccessRequest("old_row", "read", 9217, 411))
        requests = (AccessRequest("large_write", "write", 143, 50000, 1.),
                    AccessRequest("early_read", "read", 9216, 21001, 0.),
                    AccessRequest("distant_read", "read", c.capacity_bytes // 2 + 9, 73, 7.),
                    AccessRequest("later", "write", 31, 95, 100000.))
        for shared in (False, True):
            geometry = replace(c, metadata={"shared_command_resource": "ca"} if shared else {})
            cores = [DramCore(geometry, capture_details=index == 0) for index in range(4)]
            for core in cores:
                core.submit_many(history)
                core.timeline.transfer("dram:data:2", 48, 8888, 16, direction="write")
                core.timeline.reserve("ca" if shared else "dram:command:1", 12345, 17)
                core.timeline.reserve("unrelated", 54321, 19)
            expected = cores[0].submit_many(requests)
            single = tuple(cores[1].submit(r) for r in requests)
            batch = cores[2].submit_many(requests)
            with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None):
                fallback = tuple(cores[3].submit(r) for r in requests)
            for core, result in zip(cores[1:], (single, batch, fallback)):
                reference_checks.DramCoreSemanticsTests().assert_matches(cores[0], expected, core, result)
                self.assertEqual(core.timeline.last_intervals["unrelated"], (54321, 54340))


if __name__ == "__main__":
    unittest.main()
