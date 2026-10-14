"""Inspector and hardware imports use the execution configuration rules."""
from copy import deepcopy

import pytest

from heterollm_sim.config import hardware_from_dict
from heterollm_sim.hbm_geometry import make_hbm_geometry_config
from heterollm_sim.web import HttpError, physical_memory_config_payload


def reference():
    return make_hbm_geometry_config(generation="HBM4", capacity_bytes=1 << 30,
        read_bandwidth_gbps=8192., write_bandwidth_gbps=8192.,
        die_density_gbits=24, dies_per_stack=8)


def test_inspector_geometry_edit_refreshes_values_and_preserves_input():
    config = reference()
    before = deepcopy(config)
    updated = physical_memory_config_payload({"config": config,
        "overrides": {"dies_per_stack": 16}})
    assert updated["refresh_cycle_ns"] == 490.
    assert updated["bank_groups_per_rank"] == 8
    assert config == before


def test_inspector_timing_edit_is_an_experiment_not_a_standard_minimum():
    updated = physical_memory_config_payload({"config": reference(),
        "overrides": {"column_same_group_ns": 7.}})
    assert updated["column_same_group_ns"] == 7.
    assert updated["metadata"]["parameter_provenance"]["column_same_group_ns"]["status"] == "explicit_assumption"


def test_hardware_import_exposes_the_actual_canonical_configuration():
    config = reference()
    config["dies_per_stack"] = 16
    hardware = hardware_from_dict({"name": "HBM import", "components": [{"component_id": "hbm", "kind": "hbm",
        "capacity_bytes": config["capacity_bytes"], "metadata": {"physical_memory_config": config}}]})
    actual = hardware.components[0].metadata["physical_memory_config"]
    assert actual["refresh_cycle_ns"] == 490.
    assert actual["bank_groups_per_rank"] == 8


def test_inspector_rejects_invalid_configuration_as_a_client_error():
    with pytest.raises(HttpError) as error:
        physical_memory_config_payload({"config": reference(), "overrides": {"stacks": 0}})
    assert error.value.status == 422
    assert error.value.code == "invalid_physical_memory_config"
