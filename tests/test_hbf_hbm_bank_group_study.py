"""The sensitivity experiment changes catalog input, not the backend path."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_bank_group_study import ARCHITECTURES, PROFILES, run_point, summarize
from hbf_hbm_experiments import bank_group_matrix_complete, completion_matrix


@pytest.fixture(scope="module")
def rows():
    return [run_point((profile, 16, architecture, 3))
            for profile in PROFILES for architecture in ARCHITECTURES]


def test_real_prefill_decode_and_actual_core_configuration_are_verified(rows):
    result = summarize(rows, contexts=(16,), output=3)
    assert result["status"] == "passed", result["checks"]
    assert len(result["comparisons"]) == 2
    for row in rows:
        assert row["decode_measurement"]["decode_tokens"] == 2
        assert all(row["decode_core_checks"].values())
        configs = row["executed_hbm_parameters"]
        assert configs
        for config in configs.values():
            assert config["bank_groups_per_rank"] * config["banks_per_group"] == 64
            assert (config["channels"], config["pseudo_channels_per_channel"], config["row_bytes"], config["burst_bytes"]) == (16, 2, 1024, 32)
            assert config["metadata"]["unknown_parameters"]
            assert config["control_mode"] == "jedec_hbm"
            assert config["column_same_group_ns"] >= 4. * config["command_clock_ns"]
            assert config["metadata"]["timing_validation"] == "not_a_vendor_speed_bin"
        hbm = [r for r in row["core_execution_audit"]["owners"] if r["memory_kind"] == "HBM"]
        for owner in hbm:
            assert owner["jedec_control_transactions"] == owner["transactions"] > 0


def test_missing_variant_or_mismatched_inventory_cannot_be_called_complete(rows):
    assert summarize(rows[:-1], contexts=(16,), output=3)["status"] == "failed"
    changed = deepcopy(rows)
    changed[0]["fixed_inventory"]["hbm0"]["capacity_bytes"] += 64
    report = summarize(changed, contexts=(16,), output=3)
    assert report["status"] == "failed"
    assert not report["checks"]["same_inventory"]


def test_successful_subset_does_not_claim_full_research_matrix_completion(rows):
    subset = summarize(rows, contexts=(16,), output=3)
    assert subset["status"] == "passed"
    assert not bank_group_matrix_complete([subset])
    item = next(item for item in completion_matrix([subset]) if item["groups"] == [20])
    assert item["status"] == "本次未完成"
