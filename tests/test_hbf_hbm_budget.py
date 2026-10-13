"""Acceptance tests for the fixed four-slot HBM/HBF budget sweep."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_budget.py"
_SPEC = importlib.util.spec_from_file_location("hbf_hbm_budget", _SCRIPT)
_budget = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _budget
_SPEC.loader.exec_module(_budget)


class HbfHbmBudgetTests(unittest.TestCase):
    def test_architecture_sweep_keeps_flash_density_and_fixed_demand(self):
        report = _budget.run_architecture_budget_sweep()
        self.assertEqual(report["status"], "passed")
        rows = report["rows"]
        self.assertEqual([row["hbm_slots"] for row in rows], [0, 1, 2, 3, 4])
        self.assertEqual([row["aggregate_budget"]["capacity_pages"] for row in rows],
                         [128, 100, 72, 44, 16])
        self.assertEqual([row["runnable"] for row in rows], [True, True, True, True, False])
        self.assertIn("16 pages", rows[-1]["reason"])
        self.assertIsNone(rows[-1]["memory_service_duration_ns"])
        self.assertIsNone(rows[-1]["node_tps"])
        for row in rows[:-1]:
            self.assertEqual(row["logical_bytes"], (40 + 2) * 4 * _budget.PAGE)
            self.assertEqual(sum(slot["resident_pages"] for slot in row["slot_traffic"].values()), 40)
            self.assertAlmostEqual(sum(row["step_service_ns"]), row["memory_service_duration_ns"])
            self.assertGreater(sum(row["slot_traffic"]["0"]["resource_busy_ns"].values()), 0)
        self.assertEqual([slot["resident_pages"] for slot in rows[0]["slot_traffic"].values()],
                         [10, 10, 10, 10])
        self.assertFalse(report["can_infer_commercial_optimal_hbm_hbf_ratio"])
        json.dumps(report, allow_nan=False)

    def test_architecture_sweep_recomputes_runnability_for_changed_demand(self):
        report = _budget.run_architecture_budget_sweep(kv_pages=8, steps=2)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["runnable_count"], 5)
        self.assertEqual(report["capacity_failure_count"], 0)
        self.assertEqual({row["logical_bytes"] for row in report["rows"]},
                         {(4 + 8 + 2) * 2 * _budget.PAGE})

    def test_fixed_slot_budget_links_capacity_bandwidth_and_buffer(self):
        report = _budget.run_budget_sweep()
        self.assertEqual(report["total_slots"], 4)
        self.assertFalse(report["calibrated_to_silicon"])
        rows = report["rows"]
        self.assertEqual([row["hbm_slots"] for row in rows], [1, 2, 3])
        self.assertEqual([len(row["slots"]) for row in rows], [4, 4, 4])
        self.assertFalse(rows[0]["runnable"])
        self.assertIn("capacity", rows[0]["reason"])
        self.assertTrue(rows[1]["runnable"])
        self.assertTrue(rows[2]["runnable"])
        self.assertEqual(
            {row["logical_bytes"] for row in rows[1:]},
            {(12 + 2) * 4 * _budget.PAGE},
        )
        self.assertGreater(
            rows[2]["aggregate_budget"]["read_bandwidth_gb_s"],
            rows[1]["aggregate_budget"]["read_bandwidth_gb_s"],
        )
        self.assertGreater(
            rows[1]["aggregate_budget"]["write_buffer_bytes"],
            rows[2]["aggregate_budget"]["write_buffer_bytes"],
        )
        self.assertLess(
            rows[2]["memory_service_duration_ns"],
            rows[1]["memory_service_duration_ns"],
        )
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
