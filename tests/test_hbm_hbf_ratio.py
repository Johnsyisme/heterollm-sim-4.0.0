"""Acceptance coverage for the fixed HBM/HBF capacity-ratio sweep."""

import importlib.util
import json
from pathlib import Path
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accept_hbf_hbm.py"
_SPEC = importlib.util.spec_from_file_location("accept_hbf_hbm_ratio", _SCRIPT)
_acceptance = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_acceptance)


class HbmHbfRatioAcceptanceTests(unittest.TestCase):
    def test_ratio_scan_preserves_workload_and_exposes_capacity_effect(self):
        metrics = _acceptance.check_hbm_hbf_ratio()
        rows = metrics["rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["hbm_capacity_pages"] + row["hbf_capacity_pages"]) for row in rows},
            {12},
        )
        self.assertEqual({row["logical_bytes"] for row in rows}, {917504})
        self.assertLess(
            rows[-1]["memory_service_duration_ns"],
            rows[0]["memory_service_duration_ns"],
        )
        self.assertEqual(metrics["best_ratio"], "ratio_hbm_10_hbf_2")
        json.dumps(metrics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
