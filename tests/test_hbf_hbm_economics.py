"""Units, physical bounds and conditional semantics of the sensitivity models."""
import importlib.util
import math
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_economics.py"
SPEC = importlib.util.spec_from_file_location("hbf_economics_test", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_uniform_wear_counts_physical_traffic_and_extra_gc_once():
    args = dict(media_write_bytes_per_active_second=1e9, capacity_bytes=512_000_000_000,
                rated_pe_cycles=3000, active_fraction=.5)
    base = module.uniform_wear_estimate(**args)
    gc = module.uniform_wear_estimate(**args, additional_gc_factor=3)
    assert math.isclose(gc["pe_cycles_per_year"], 3*base["pe_cycles_per_year"])
    assert math.isclose(gc["uniform_wear_lifetime_years"], base["uniform_wear_lifetime_years"]/3)
    zero = module.uniform_wear_estimate(**{**args, "media_write_bytes_per_active_second": 0})
    assert zero["uniform_wear_lifetime_years"] is None
    assert zero["replacement_count_before_study_end"] == 0


def test_conditional_cost_units_and_area_failure_are_explicit():
    row = module.conditional_node_cost(hbm_packages=1, hbf_packages=3,
                                       hbf_package_area_mm2=150., active_output_tps=1000.)
    assert row["capex_usd"] == 14000.
    assert row["memory_package_area_mm2"] == 550.
    assert row["area_budget_feasible"] is False
    assert math.isclose(row["conditional_usd_per_million_tokens"],
                        row["capex_usd"]/row["conditional_output_tokens"]*1e6)
    assert module.conditional_node_cost(hbm_packages=4, hbf_packages=0, active_output_tps=0.)["conditional_usd_per_million_tokens"] is None


def test_projection_does_not_attribute_ssd_writes_to_hbf():
    row = {"id": "ssd", "placement": "F_nvme", "node_tps": 100., "gpu_count": 1, "makespan_ns": 1e9,
           "applied_hardware": {"memory_kinds": {"hbf0": "ssd"}},
           "resources": {"storage_traffic": {"owner_ids": ["hbf0.memory"], "physical_write_bytes": 4096}}}
    report = module.run_economics_sensitivity([row])
    assert report["calibrated"] is False
    assert len(report["package_cost_area_rows"]) == 30
    assert len(report["purchase_cost_amortization_rows"]) == 6
    assert len(report["serving_tps_conditional_projections"]) == 1
    assert {r["write_rate_source_id"] for r in report["uniform_wear_rows"]} == {"assumed_1GB_per_active_second"}
    assert report["write_rate_source_assessments"][0]["status"] == "non_hbf_or_mixed_storage"


def test_endurance_uses_declared_kind_and_skips_unverified_owner_names():
    base = {"node_tps": 100., "gpu_count": 1, "makespan_ns": 1e9,
            "resources": {"storage_traffic": {"owner_ids": ["hbf0.memory"], "physical_write_bytes": 4096}}}
    unknown = {**base, "id": "unknown"}
    known = {**base, "id": "known", "applied_hardware": {"memory_kinds": {"hbf0": "hbf"}}}
    report = module.run_economics_sensitivity([unknown, known])
    sources = {r["write_rate_source_id"] for r in report["uniform_wear_rows"]}
    assert sources == {"assumed_1GB_per_active_second", "known"}
    assert [r["status"] for r in report["write_rate_source_assessments"]] == ["unverified_owner_kind", "verified_hbf"]


@pytest.mark.parametrize("overrides", [
    {"gpu_count": 1.5}, {"gpu_count": 0}, {"gpu_count": True},
    {"gpu_price_usd": float("nan")}, {"node_base_price_usd": float("inf")},
    {"hbf_package_price_usd": -1}, {"active_fraction": 1.1},
    {"active_fraction": 0}, {"memory_area_budget_mm2": 0},
])
def test_invalid_node_assumptions_are_rejected(overrides):
    with pytest.raises(ValueError):
        module.conditional_node_cost(hbm_packages=2, hbf_packages=2, assumptions=overrides)


@pytest.mark.parametrize("overrides", [{"capacity_bytes": 10.5}, {"rated_pe_cycles": 0},
                                      {"media_write_bytes_per_active_second": float("nan")},
                                      {"replacement_cost_usd": -1}, {"additional_gc_factor": .5}])
def test_invalid_wear_inputs_are_rejected(overrides):
    with pytest.raises(ValueError):
        module.uniform_wear_estimate(**{**dict(media_write_bytes_per_active_second=1e9,
                                               capacity_bytes=512_000_000_000, rated_pe_cycles=3000), **overrides})
