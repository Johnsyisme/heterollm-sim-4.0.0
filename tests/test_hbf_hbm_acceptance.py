"""Standard-library test entry point for the HBM/HBF acceptance matrix."""
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accept_hbf_hbm.py"
SPEC = importlib.util.spec_from_file_location("accept_hbf_hbm", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class HbfHbmAcceptanceTests(unittest.TestCase):
    def test_physical_invariants_and_placement_matrix(self):
        report = acceptance.run_acceptance()
        failures = [item for item in report["checks"] + report["experiments"] if not item["passed"]]
        self.assertEqual(failures, [], json.dumps(failures, indent=2))
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["experiments"]), 4)
        self.assertFalse(report["assumptions"]["calibrated_to_silicon"])
        migration = next(item for item in report["checks"] if item["id"] == "physical_kv_migration")
        self.assertEqual(migration["metrics"]["storage"]["target_tier"], "storage")
        slo = next(item for item in report["checks"] if item["id"] == "slo_semantics")
        self.assertEqual(slo["metrics"]["loose_deadline_met"], True)
        self.assertEqual(slo["metrics"]["tight_deadline_met"], False)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
