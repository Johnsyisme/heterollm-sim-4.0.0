"""Physical micro results report the Core that actually serviced requests."""
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import hbf_hbm_budget as budget
import hbf_hbm_media_experiments as media
import hbf_hbm_session_experiments as sessions
import hbf_hbm_sessions as lifecycle
import hbf_hbm_traffic_study as traffic


def assert_hbm_stack(config):
    assert config["generation"] == "HBM3E"
    assert (config["stacks"], config["channels"], config["pseudo_channels_per_channel"]) == (1, 16, 2)
    assert (config["row_bytes"], config["burst_bytes"]) == (1024, 32)


def test_package_records_actual_cores_and_actual_lane_count():
    actual = []
    factory = budget._core
    def capture(profile):
        core = factory(profile)
        actual.append(core)
        return core
    with patch.object(budget, "_core", capture):
        row = budget.budget_row(hbm_slots=3)
    assert row["runnable"]
    assert budget.HBM_SLOT.parallel_units == 32
    assert [slot["parallel_units"] for slot in row["slots"]] == [32, 32, 32, 2]
    assert row["aggregate_budget"]["parallel_units"] == 98
    # The unused flash slot is configured but is not execution evidence.
    assert set(row["executed_physical_configurations"]) == {"0", "1", "2"}
    for slot, evidence in row["executed_physical_configurations"].items():
        assert evidence["physical_config"] == asdict(actual[int(slot)].config)
        assert evidence["transactions"] > 0
        assert_hbm_stack(evidence["physical_config"])
    rejected = budget.budget_row(hbm_slots=1)
    assert rejected["runnable"] is False
    assert rejected["aggregate_budget"]["parallel_units"] == 38
    assert rejected["executed_physical_configurations"] == {}


@pytest.mark.parametrize("architecture", ["A_hbm_only", "B_all_hbf", "C_hot_cold"])
def test_session_evidence_matches_used_runtime_cores(architecture):
    experiment = sessions.SessionExperiment(architecture, sessions=4, cycles=1, hbm_pages=4)
    row = experiment.run()
    assert row["status"] == "passed"
    evidence = row["executed_physical_configurations"]
    used = {owner: active for owner, active in experiment.runtime.runtimes.items() if active.clock_ns > 0}
    assert evidence and set(evidence) == set(used)
    for owner, item in evidence.items():
        assert item["physical_config"] == asdict(used[owner].core.config)
        if item["core"] == "DramCore":
            assert_hbm_stack(item["physical_config"])
    if architecture == "B_all_hbf":
        assert {item["core"] for item in evidence.values()} == {"NandCore"}


def test_allocation_only_session_does_not_claim_memory_execution():
    row = lifecycle.run_experiment("hbm_only", 1000., 1)
    assert row["physical_transfer_count"] == 0
    assert row["executed_physical_configurations"] == {}


def test_paths_and_traffic_record_only_used_live_core_configs():
    direct = media._run_path_row("direct", payload_bytes=16384)
    relay = media._run_path_row("relay", payload_bytes=16384)
    for row, hbm_count in ((direct, 1), (relay, 2)):
        records = row["executed_physical_configurations"]
        hbm = [item for item in records.values() if item["core"] == "DramCore"]
        assert len(hbm) == hbm_count
        for item in hbm:
            assert_hbm_stack(item["physical_config"])
            assert item["last_completion_ns"] > 0
            # These owner-specific prefixes are injected by the live runtime,
            # and are absent from a newly resolved catalog configuration.
            assert item["physical_config"]["metadata"]["data_resource_prefix"].endswith(":dram:data")
    row = traffic.run_hot_read_trace(hot_reads=80)
    assert row["status"] == "passed"
    assert {item["core"] for item in row["executed_physical_configurations"].values()} == {"DramCore", "NandCore"}
    nand = traffic.run_read_write_trace(1)["executed_physical_configurations"]["hbf"]
    assert nand["core"] == "NandCore"
    assert (nand["physical_config"]["channels"], nand["physical_config"]["parallel_units"]) == (4, 4)
