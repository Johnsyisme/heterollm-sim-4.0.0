"""Serving pressure and rollback acceptance for the paged HBM/HBF path."""

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accept_hbf_hbm.py"
SPEC = importlib.util.spec_from_file_location("accept_hbf_hbm_serving", SCRIPT)
acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acceptance)


class ServingSwapAcceptanceTests(unittest.TestCase):
    def test_multi_page_preempt_resume_and_atomic_rollback(self):
        metrics = acceptance.check_serving_paged_swap()
        self.assertGreaterEqual(metrics["preemptions"], 1)
        self.assertGreaterEqual(metrics["swap_events"], 1)
        self.assertEqual(metrics["swap_out_pages"], 2)
        self.assertGreater(metrics["swap_transfer_time_ns"], 0.0)
        self.assertGreater(metrics["makespan_ns"], 0.0)
        self.assertGreater(metrics["swap_low_ttft_ns"], 0.0)
        self.assertGreater(metrics["swap_low_tpot_ns"], 0.0)
        self.assertEqual(metrics["kv_peak_pages"], 3)
        self.assertEqual(metrics["physical_swap_bytes"], 4 * 16_384)
        self.assertEqual(metrics["requests_finished"], 2)
        self.assertEqual(metrics["rollback_preserved_pages"], 2)
        self.assertEqual(metrics["rollback_target_used_bytes"], 0)
        self.assertEqual(metrics["restore_rollback_preserved_pages"], 2)
        self.assertEqual(metrics["restore_target_used_bytes"], 2 * 16_384)
        json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
