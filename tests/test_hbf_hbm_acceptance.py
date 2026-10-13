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
        ratio = next(item for item in report["checks"] if item["id"] == "hbm_hbf_ratio")
        self.assertEqual(ratio["metrics"]["fixed_working_set_pages"], 12)
        scan = next(item for item in report["checks"] if item["id"] == "fixed_workload_scan")
        self.assertEqual(
            [row["context_tokens"] for row in scan["metrics"]["contexts"]],
            [32, 64, 128],
        )
        plan_status = {item["stage"]: item["status"] for item in report["plan_completion"]}
        self.assertEqual(plan_status["0_baseline"], "passed")
        self.assertEqual(plan_status["1_parameters_and_timing"], "passed")
        self.assertEqual(plan_status["2_simplified_hardware_execution"], "passed")
        self.assertEqual(plan_status["4_tiering_migration_and_prefetch"], "passed")
        self.assertEqual(plan_status["5_metrics_and_experiment_entry"], "passed")
        self.assertEqual(plan_status["3_capacity_lifecycle_and_write_buffer"], "passed")
        self.assertEqual(plan_status["6_acceptance_and_question_matrix"], "partial")
        self.assertFalse(report["plan_complete"])
        self.assertEqual(report["baseline_validation"]["status"], "not_available")
        self.assertIsNone(report["baseline_validation"]["relative_error"])
        self.assertEqual(len(report["validation_matrix"]), len(report["capabilities"]))
        write_buffer = next(item for item in report["checks"] if item["id"] == "finite_write_buffer")
        self.assertEqual(
            [row["capacity_bytes"] for row in write_buffer["metrics"]["buffer_scan"]],
            [4096, 8192, 16384],
        )
        self.assertEqual(write_buffer["metrics"]["buffered_semantics"]["merge_bytes"], 4096)
        self.assertEqual(write_buffer["metrics"]["buffered_semantics"]["staging_read_hits"], 1)
        self.assertEqual(write_buffer["metrics"]["buffered_semantics"]["physical_programs_on_drain"], 1)
        sessions = next(item for item in report["checks"] if item["id"] == "hot_cold_session_scan")
        self.assertEqual(
            [row["active_slots"] for row in sessions["metrics"]["rows"]],
            [1, 2],
        )
        slo = next(item for item in report["checks"] if item["id"] == "slo_semantics")
        self.assertEqual(slo["metrics"]["loose_deadline_met"], True)
        self.assertEqual(slo["metrics"]["tight_deadline_met"], False)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
