"""Inspector and hardware imports use the execution configuration rules."""
from copy import deepcopy

import pytest

from heterollm_sim._hbm_errors import HBMConfigError
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


@pytest.mark.parametrize("field,value", [("column_same_group_ns", 3.),
    ("column_different_group_ns", 1.), ("column_different_sid_ns", 2.),
    ("precharge_to_precharge_ns", 1.), ("refresh_cycle_ns", 400.),
    ("maximum_active_ns", 0.)])
def test_inspector_rejects_protocol_timing_violations_as_client_errors(field, value):
    with pytest.raises(HttpError) as error:
        physical_memory_config_payload({"config": reference(), "overrides": {field: value}})
    assert error.value.status == 422
    assert error.value.code == "invalid_physical_memory_config"


def test_hardware_import_cannot_use_provenance_to_hide_a_protocol_timing_violation():
    config = reference()
    config["column_same_group_ns"] = 0.
    config["metadata"]["parameter_provenance"]["column_same_group_ns"]["status"] = "explicit_assumption"
    with pytest.raises(ValueError, match="column_same_group_ns"):
        hardware_from_dict({"name": "invalid HBM", "components": [{"component_id": "hbm", "kind": "hbm",
            "capacity_bytes": config["capacity_bytes"], "metadata": {"physical_memory_config": config}}]})


def test_inspector_preserves_structured_timing_diagnostics_and_both_languages():
    with pytest.raises(HttpError) as captured:
        physical_memory_config_payload({"config": reference(),
            "overrides": {"column_same_group_ns": 3.}})
    error = captured.value
    assert error.status == 422
    assert error.code == "invalid_physical_memory_config"
    assert error.message and any("\u4e00" <= char <= "\u9fff" for char in error.message)
    assert error.message_en
    assert error.details["field"] == "column_same_group_ns"
    assert error.details["value"] == 3.
    assert error.details["clock_ns"] == 1.
    assert error.details["category"] == "standard_limit"
    assert error.details["related_values"]["minimum_ns"] == 4.
    assert error.details["remedy_zh"] and error.details["remedy_en"]


def test_hardware_import_preserves_the_field_and_requirement_instead_of_flattening_the_error():
    config = reference()
    config["column_same_group_ns"] = 3.
    with pytest.raises(HBMConfigError) as captured:
        hardware_from_dict({"name": "invalid HBM", "components": [{"component_id": "hbm", "kind": "hbm",
            "capacity_bytes": config["capacity_bytes"], "metadata": {"physical_memory_config": config}}]})
    error = captured.value
    assert error.details["field"] == "column_same_group_ns"
    assert error.details["value"] == 3.
    assert error.details["related_values"]["minimum_ns"] == 4.
    assert error.details["category"] == "standard_limit"
    assert error.message_zh and error.message_en
