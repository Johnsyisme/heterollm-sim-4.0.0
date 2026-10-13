"""Physical byte and ownership checks for the controlled memory traffic study."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def report():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_traffic_study.py"
    spec = importlib.util.spec_from_file_location("traffic_study_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_traffic_study()


def test_actual_page_owners_and_cold_moves_match_controlled_hit_trace(report):
    assert report["id"] == "10_traffic_and_tiering"
    assert report["status"] == "passed"
    assert all(report["checks"].values())
    assert len(report["rows"]) == 12
    assert report["capability"]["supported"]
    for row in report["rows"]:
        assert row["node_tps"] is None
        if row["sweep"] != "hot_read_fraction":
            continue
        hits = row["requested_hot_read_count"]
        events, transfers = row["events"], row["transfers"]
        assert len(events) == 100
        assert sum(event["observed_hbm_hit"] for event in events) == hits
        assert {event["logical_page_id"] for event in events}
        for event in events:
            assert event["owner_before_read"] == ("hbm" if event["observed_hbm_hit"] else "hbf")
            assert event["owner_after_policy"] == event["owner_before_read"]
            assert event["start_ns"] <= event["read_complete_ns"] <= event["policy_complete_ns"]
        assert sum(item["source_component"] == "hbf" for item in transfers) == 100-hits
        assert sum(item["target_component"] == "hbf" for item in transfers) == 100-hits
        assert row["hbf_physical_read_bytes"] == row["hbf_physical_write_bytes"] == (100-hits)*row["page_bytes"]
        assert row["migration_bytes"] == 2*(100-hits)*row["page_bytes"]
        assert row["hbm_consumption_physical_read_bytes"] == 100*row["page_bytes"]
        assert all(row["checks"].values())
    # An invalid non-finite result must not be silently serialized as JSON NaN.
    json.dumps(report, allow_nan=False)


def test_fixed_capacity_is_applied_and_zero_hbm_is_a_real_staging_rejection(report):
    rows = [row for row in report["rows"] if row["sweep"] == "hbm_capacity_fraction"]
    assert {row["hbm_fraction"] for row in rows} == {0., .1, .25, .5}
    for row in rows:
        assert sum(row["capacity_bytes"].values()) == 100*row["page_bytes"]
        assert row["capacity_bytes"]["hbm"] == row["hbm_capacity_pages"]*row["page_bytes"]
        if row["hbm_capacity_pages"] == 0:
            assert row["status"] == "configuration_rejected"
            assert row["runnable"] is False
            assert "capacity" in row["reason"]
            assert "staging" in row["rejection_scope"]
            assert row["executed_read_count"] == 0
            assert row["memory_service_duration_ns"] is None
        else:
            assert row["status"] == "passed"
            assert row["resident_working_set_pages"] == 40
            assert row["observed_hot_read_count"] == 80
            assert row["applied_hardware"]["hbm"]["capacity_bytes"] == row["capacity_bytes"]["hbm"]
            assert all(peak <= row["capacity_bytes"][owner]
                       for owner, peak in row["peak_bytes_including_migration_copies"].items())


def test_read_write_ratios_execute_exact_bytes_with_separate_seed_window(report):
    rows = [row for row in report["rows"] if row["sweep"] == "read_write_ratio"]
    assert {row["observed_read_write_ratio"] for row in rows} == {1, 10, 100, 1000}
    for row in rows:
        page = row["page_bytes"]
        ratio = row["requested_read_write_ratio"]
        duration = row["memory_service_duration_ns"]
        assert row["initialization_physical_write_bytes"] == 8*page
        assert row["physical_read_bytes"] == row["logical_read_bytes"] == ratio*page
        assert row["physical_write_bytes"] == row["logical_write_bytes"] == page
        assert row["read_operations"] == ratio
        assert row["write_operations"] == 1
        assert row["measurement_window"]["start_ns"] == row["initialization_duration_ns"]
        assert row["measurement_window"]["end_ns"]-row["measurement_window"]["start_ns"] == duration
        assert row["effective_total_payload_gb_s"] == (ratio+1)*page/duration
        assert row["resource_bytes"] and row["resource_busy_ns"]
