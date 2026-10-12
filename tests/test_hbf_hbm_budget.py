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
