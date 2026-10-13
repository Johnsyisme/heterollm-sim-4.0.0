"""Real activity and resource cleanup in the session experiment matrix."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_session_experiments.py"
SPEC = importlib.util.spec_from_file_location("session_experiments_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SessionExperimentsTests(unittest.TestCase):
    def test_failed_checks_remove_supported_claims(self):
        with patch.object(module.SessionExperiment, "run", return_value={"checks": {"injected": False}}):
            report = module.run_session_experiments()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["capability"]["supported"], [])

    def test_all_hbf_services_active_reads_and_appends_without_swap(self):
        row = module.SessionExperiment("B_all_hbf", sessions=2, cycles=1).run()
        self.assertEqual(row["status"], "passed")
        self.assertGreater(row["active_read_bytes_by_component"]["hbf"], 0)
        self.assertGreater(row["active_append_bytes_by_component"]["hbf"], 0)
        self.assertEqual(row["migration_bytes"], 0)
        self.assertIsNone(row["llm_token_per_second"])

    def test_hbm_pause_is_retained_and_shared_prefix_is_released(self):
        row = module.SessionExperiment("A_hbm_only", sessions=4, cycles=1,
                                       prefix_reuse_rate=.5, hbm_pages=16).run()
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["retained_peak"], 4)
        self.assertEqual(row["prefix_hits"], 2)
        self.assertEqual(row["peak_bytes"]["hbm"], 12 * module.PAGE)

    def test_hybrid_restores_cold_pages_before_access(self):
        row = module.SessionExperiment("C_hot_cold", sessions=4, cycles=1,
                                       hbm_pages=4).run()
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["admitted_sessions"], 4)
        self.assertGreater(row["restore_bytes"], 0)
        self.assertGreater(row["resume_latency_p99_ns"], 0)
        self.assertEqual(row["active_read_bytes_by_component"]["hbf"], 0)

    def test_exact_offered_prefix_rate_keeps_rejection_denominator_explicit(self):
        row = module.SessionExperiment("A_hbm_only", sessions=20, cycles=1,
                                       prefix_reuse_rate=.25, hbm_pages=12).run()
        self.assertEqual(row["status"], "passed")
        self.assertEqual(row["prefix_offered_sessions"], 5)
        self.assertEqual(row["prefix_reuse_rate_offered"], .25)
        self.assertGreater(row["rejected_sessions"], 0)
        self.assertEqual(row["prefix_reuse_rate_admitted"], row["prefix_hits"] / row["admitted_sessions"])
        self.assertNotEqual(row["prefix_reuse_rate_admitted"], row["prefix_reuse_rate_offered"])


if __name__ == "__main__":
    unittest.main()
