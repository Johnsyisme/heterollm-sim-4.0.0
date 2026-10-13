"""Trace-level hot/cold HBM/HBF session acceptance tests."""

import importlib.util
from pathlib import Path
import sys
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_sessions.py"
_SPEC = importlib.util.spec_from_file_location("hbf_hbm_sessions", _SCRIPT)
_sessions = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _sessions
_SPEC.loader.exec_module(_sessions)


class HbmHbfSessionTraceTests(unittest.TestCase):
    def test_offload_capacity_failure_is_not_silently_treated_as_success(self):
        with self.assertRaises(RuntimeError):
            _sessions.SessionTrace(
                mode="all_hbf", pause_duration_ns=0, resume_cycles=1, hbf_pages=1,
            ).run()

    def test_matrix_has_real_lifecycle_and_physical_transfer_rows(self):
        report = _sessions.run_matrix()
        rows = report["experiments"]
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {(row["mode"], row["pause_duration_ns"], row["resume_cycles"])
             for row in rows},
            {
                (mode, pause, cycles)
                for mode in ("hbm_only", "all_hbf", "hot_cold")
                for pause in (0.0, 250000.0)
                for cycles in (1, 2)
            },
        )
        required = {"create", "append", "pause", "resume_requested", "resume", "end"}
        for row in rows:
            self.assertTrue(required.issubset(set(row["event_types"])))
            self.assertEqual(row["remaining_pages"], 0)
            self.assertEqual(set(row["used_bytes_by_component"].values()), {0})
            self.assertEqual(row["released_pages"], 5)
            events = row["events"]
            self.assertEqual(
                len({event["sequence"] for event in events}), len(events)
            )

        by_mode = {
            mode: [row for row in rows if row["mode"] == mode]
            for mode in ("hbm_only", "all_hbf", "hot_cold")
        }
        self.assertTrue(all(row["migration_count"] == 0 for row in by_mode["hbm_only"]))
        for mode in ("all_hbf", "hot_cold"):
            self.assertTrue(all(row["migration_count"] > 0 for row in by_mode[mode]))
            self.assertTrue(all(row["restore_wait_ns"] > 0 for row in by_mode[mode]))
            migrated = by_mode[mode][0]
            migration_events = [
                event for event in migrated["events"]
                if event["event_type"] in {"migration_start", "migration_complete"}
            ]
            self.assertEqual(len(migration_events), 2 * migrated["migration_count"])
            self.assertTrue(all(event["session_id"] for event in migration_events))


if __name__ == "__main__":
    unittest.main()
