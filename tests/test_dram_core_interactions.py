"""DRAM shared-resource, slot-transition and chunk accounting regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest, DramConfig
import test_dram_core_semantics as reference_checks


class DramCoreInteractionTests(unittest.TestCase):
    def test_single_slot_accelerators_discard_previous_multislot_history(self):
        config = DramConfig(banks_per_group=1, open_ns=0, close_ns=0,
                            read_latency_ns=0, write_latency_ns=0,
                            burst_interval_ns=0, write_recovery_ns=0,
                            interface_bandwidth_gb_s=64,
                            write_to_read_ns=50, read_to_write_ns=40)
        for mode in ("detailed", "compiled", "compiled_batch", "fold"):
            with self.subTest(mode=mode):
                timeline = ResourceTimeline(
                    lane_available={"dram:data:0": [10., 20.]},
                    directions={"dram:data:0": "write"},
                    lane_directions={"dram:data:0": ["read", "write"],
                                     "unrelated": ["write", "read"]})
                core = DramCore(config, timeline, capture_details=mode == "detailed")
                # Contract the caller-owned calendar; single-slot direction
                # is the scalar write history, regardless of stale slot data.
                timeline.lane_available["dram:data:0"] = [20.]
                if mode == "compiled_batch":
                    rows = core.submit_many((AccessRequest("a", "write", 0, 2048),
                                             AccessRequest("b", "write", 2048, 2048)))
                    self.assertTrue(all(r.counters.get("compiled_batch_float64") for r in rows))
                elif mode == "fold":
                    with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None):
                        rows = (core.submit(AccessRequest("a", "write", 0, 4096)),)
                    self.assertGreater(rows[0].counters["accelerated_row_hit_bursts"], 0)
                else:
                    rows = (core.submit(AccessRequest("a", "write", 0, 4096)),)
                    if mode == "compiled":
                        self.assertTrue(rows[0].counters.get("compiled_float64"))
                self.assertNotIn("dram:data:0", timeline.lane_directions)
                self.assertEqual(timeline.lane_directions["unrelated"], ["write", "read"])
                self.assertEqual(sum(r.physical_write_bytes for r in rows), 4096)
                # Expanding the calendar must seed both slots from the latest
                # scalar write, not revive the pre-contraction read history.
                timeline.lane_available["dram:data:0"] = [100., 200.]
                result = core.submit(AccessRequest("expanded", "read", 0, 64))
                self.assertEqual(result.completion_ns, 151)
                self.assertEqual(timeline.lane_directions["dram:data:0"], ["read", "write"])

    def test_shared_command_data_bank_names_preserve_cost_deltas(self):
        bank0 = "dram:bank:0:0:0:0:0:0"
        shared_prefix = "dram:bank:0:0:0:0:0"
        variants = (
            {"command_resource_prefix": "wire", "data_resource_prefix": "wire"},
            {"shared_command_resource": "dram:data:0"},
            {"command_resource_prefix": shared_prefix},
            {"data_resource_prefix": shared_prefix},
            {"command_resource_prefix": shared_prefix, "data_resource_prefix": shared_prefix},
            {"shared_command_resource": bank0},
        )
        requests = (AccessRequest("write", "write", 0, 1365),
                    AccessRequest("read_overlap", "read", 63, 2049),
                    AccessRequest("write_new_row", "write", 2048, 129),
                    AccessRequest("return", "read", 0, 64))
        for metadata in variants:
            with self.subTest(metadata=metadata):
                c = DramConfig(banks_per_group=2, rows_per_bank=4, row_bytes=1024,
                               metadata=metadata, read_to_write_ns=11, write_to_read_ns=19)
                cores = [DramCore(c, capture_details=details) for details in (True, False, False)]
                before = []
                for core in cores:
                    core.submit(AccessRequest("old", "write", 4096, 128, 10))
                    core.timeline.transfer("unrelated", 99, 800, 3, direction="read")
                    before.append((dict(core.timeline.busy_ns), dict(core.timeline.bytes_moved)))
                self.assertFalse(cores[1]._can_accelerate())
                results = (cores[0].submit_many(requests),
                           tuple(cores[1].submit(r) for r in requests),
                           cores[2].submit_many(requests))
                for core, rows in zip(cores[1:], results[1:]):
                    reference_checks.DramCoreSemanticsTests().assert_matches(cores[0], results[0], core, rows)
                for core, rows, (old_busy, old_bytes) in zip(cores, results, before):
                    # Command/bank aliases add service occupancy, not payload.
                    self.assertEqual(sum(sum(r.counters["resource_bytes"].values()) for r in rows),
                                     sum(r.transfer_bytes for r in rows))
                    self.assertEqual(sum(core.timeline.bytes_moved.values()) - sum(old_bytes.values()),
                                     sum(r.transfer_bytes for r in rows))
                    self.assertAlmostEqual(sum(sum(r.counters["resource_busy_ns"].values()) for r in rows),
                                           sum(core.timeline.busy_ns.values()) - sum(old_busy.values()), delta=1e-8)
                    self.assertEqual(core.timeline.bytes_moved["unrelated"], 99)
                    self.assertEqual(sum(r.counters["row_hits"] + r.counters["row_misses"]
                                         + r.counters["row_conflicts"] for r in rows),
                                     sum(r.counters["burst_count"] for r in rows))

    def test_chunk_boundaries_preserve_physical_work_and_row_recovery(self):
        # All cut points are burst boundaries, including a partial first and
        # last burst; no chunk expands onto a neighbour's physical burst.
        for operation in ("read", "write"):
            for details in (True, False):
                with self.subTest(operation=operation, details=details):
                    c = DramConfig(channels=2, banks_per_group=2, row_bytes=256,
                                   read_to_write_ns=7, write_to_read_ns=13,
                                   read_recovery_ns=11, write_recovery_ns=23)
                    one, chunks = (DramCore(c, capture_details=details) for _ in range(2))
                    for core in (one, chunks):
                        core.submit(AccessRequest("history", "write", 8192, 128))
                    whole = one.submit(AccessRequest("whole", operation, 7, 4090))
                    rows = chunks.submit_many((AccessRequest("first", operation, 7, 57),
                                               AccessRequest("middle", operation, 64, 3968),
                                               AccessRequest("last", operation, 4032, 65)))
                    self.assertEqual(whole.transfer_bytes, 65 * 64)
                    self.assertEqual(whole.transfer_bytes, sum(r.transfer_bytes for r in rows))
                    self.assertAlmostEqual(whole.completion_ns, max(r.completion_ns for r in rows), delta=1e-8)
                    self.assertEqual(one._banks, chunks._banks)
                    self.assertEqual(one.timeline.ready_ns, chunks.timeline.ready_ns)
                    self.assertEqual(one.timeline.bytes_moved, chunks.timeline.bytes_moved)
                    for counter in ("row_hits", "row_misses", "row_conflicts", "burst_count"):
                        self.assertEqual(whole.counters[counter], sum(r.counters[counter] for r in rows))
                    # Following operations observe the same recovery gates.
                    following = AccessRequest("after", "write", 0, 64)
                    self.assertEqual(one.submit(following).completion_ns,
                                     chunks.submit(following).completion_ns)

    def test_multislot_alias_history_matches_scalar_and_batched_entrypoints(self):
        config = DramConfig(banks_per_group=2, row_bytes=512,
                            read_to_write_ns=29, write_to_read_ns=41)
        cores = []
        for details in (True, False, False):
            timeline = ResourceTimeline(
                resource_aliases={"dram:data:0": "wire", "dram:command:0": "wire"},
                lane_available={"wire": [400., 500.]},
                directions={"wire": "write"}, lane_directions={"wire": ["read", "write"]})
            cores.append(DramCore(config, timeline, capture_details=details))
        requests = (AccessRequest("read", "read", 0, 4097),
                    AccessRequest("write", "write", 1023, 130),
                    AccessRequest("return", "read", 0, 4096))
        expected = cores[0].submit_many(requests)
        for core, rows in ((cores[1], tuple(cores[1].submit(r) for r in requests)),
                           (cores[2], cores[2].submit_many(requests))):
            reference_checks.DramCoreSemanticsTests().assert_matches(cores[0], expected, core, rows)
            self.assertEqual(core.timeline.lane_directions, cores[0].timeline.lane_directions)
            self.assertEqual(core.timeline.lane_available, cores[0].timeline.lane_available)


if __name__ == "__main__":
    unittest.main()
