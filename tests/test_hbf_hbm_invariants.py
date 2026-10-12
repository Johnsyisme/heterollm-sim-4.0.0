"""Discriminating direct-core invariants for HBM/HBF timing."""
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_invariants.py"
SPEC = importlib.util.spec_from_file_location("hbf_hbm_invariants", SCRIPT)
invariants = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(invariants)


class HbfHbmInvariantTests(unittest.TestCase):
    def test_ideal_read_is_hand_calculable_and_equivalent(self):
        result = invariants.check_ideal_read_equivalence()
        self.assertTrue(result["passed"], result)
        metrics = result["metrics"]
        self.assertEqual(metrics["logical_bytes"], 4096)
        self.assertAlmostEqual(metrics["hbm_completion_ns"], 64.0, places=9)
        self.assertLessEqual(metrics["timing_relative_error"], 0.01)

    def test_aggregate_and_split_execution_match(self):
        result = invariants.check_aggregate_execution_consistency()
        self.assertTrue(result["passed"], result)
        metrics = result["metrics"]
        self.assertEqual(metrics["aggregate_logical_bytes"], metrics["split_logical_bytes"])
        self.assertEqual(metrics["aggregate_transfer_bytes"], metrics["split_transfer_bytes"])
        self.assertLessEqual(metrics["timing_relative_error"], 0.01)

    def test_long_context_uses_bounded_event_details(self):
        result = invariants.check_long_context_event_bound()
        self.assertTrue(result["passed"], result)
        metrics = result["metrics"]
        self.assertGreater(metrics["dram_burst_count"], 1000)
        self.assertGreater(metrics["nand_page_count"], 1000)
        self.assertEqual(metrics["dram_stage_count"], 0)
        self.assertEqual(metrics["nand_stage_count"], 0)
        self.assertTrue(metrics["dram_details_truncated"])
        self.assertTrue(metrics["nand_details_truncated"])


if __name__ == "__main__":
    unittest.main()
