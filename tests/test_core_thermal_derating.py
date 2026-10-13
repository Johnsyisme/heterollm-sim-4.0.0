"""Static thermal sensitivities must reach the actual memory Core contract."""
from dataclasses import asdict, replace
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_serving_study import build_study_scenario
from heterollm_sim import thermal
from heterollm_sim.data_motion import AccessKind, resolve_service
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import (
    AccessRequest, NandThermalConfig, NandWearConfig, Operation,
    make_ddr_config, make_gddr_config, make_hbf_config, make_hbm_config,
    make_lpddr_config, make_ssd_config, parse_physical_memory_config,
)
from heterollm_sim.nand_core import NandCore
from heterollm_sim.thermal import (
    ThermalOperatingPoint, _derate_physical_memory_config,
    apply_thermal_operating_point,
)


def point(**changes):
    return ThermalOperatingPoint("memory", enabled=True,
        evidence="explicit uncalibrated test sensitivity", **changes)


@pytest.mark.parametrize("config", [
    make_hbm_config(capacity_bytes=1 << 20, interface_bandwidth_gb_s=64.,
                    read_bandwidth_gb_s=48., write_bandwidth_gb_s=32.),
    make_ddr_config(capacity_bytes=1 << 20, lane_bandwidth_gb_s=8.,
                    read_bandwidth_gb_s=6., write_bandwidth_gb_s=4.),
    make_lpddr_config(capacity_bytes=1 << 20, data_rate_mt_s=3200.),
    make_gddr_config(capacity_bytes=1 << 20, generation="GDDR7", channels=2,
                    data_width_bits=32, bandwidth_input_mode="data_rate",
                    effective_pin_data_rate_gbps=24.),
    make_gddr_config(capacity_bytes=1 << 20, generation="GDDR6", channels=2,
                    data_width_bits=32, bandwidth_input_mode="bandwidth",
                    effective_pin_data_rate_gbps=20., lane_bandwidth_gb_s=80.),
])
def test_dram_rates_remain_consistent_and_actual_core_slows(config):
    original = asdict(config)
    slowed = _derate_physical_memory_config(config,
        point(memory_bandwidth_scale=.5, latency_scale=2.))
    assert asdict(config) == original
    assert slowed.physical_interface_bandwidth_gb_s == pytest.approx(
        config.physical_interface_bandwidth_gb_s * .5)
    assert slowed.bandwidth_input_mode == config.bandwidth_input_mode
    for direction in (Operation.READ, Operation.WRITE):
        assert slowed.directional_bandwidth_gb_s(direction) == pytest.approx(
            config.directional_bandwidth_gb_s(direction) * .5)
        request = AccessRequest("transaction", direction, 0, 4096)
        before = DramCore(config).submit(request)
        after = DramCore(slowed).submit(request)
        assert after.completion_ns > before.completion_ns
        assert after.transfer_bytes == before.transfer_bytes
        assert after.counters["burst_count"] == before.counters["burst_count"]
    for name in ("open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
                 "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
                 "read_to_write_ns", "write_to_read_ns"):
        assert getattr(slowed, name) == getattr(config, name) * 2.
    changed = {name for name, value in asdict(slowed).items() if value != original[name]}
    assert changed <= {
        "data_rate_mt_s", "effective_pin_data_rate_gbps", "lane_bandwidth_gb_s",
        "interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s",
        "open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
        "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
        "read_to_write_ns", "write_to_read_ns",
    }


@pytest.mark.parametrize("factory", [make_hbf_config, make_ssd_config])
def test_nand_rates_media_timing_and_existing_feedback_are_preserved(factory):
    config = factory(capacity_bytes=8 << 20, host_bandwidth_gb_s=100.,
        host_read_bandwidth_gb_s=80., host_write_bandwidth_gb_s=20.,
        internal_bandwidth_gb_s=50., front_ns=30.,
        dynamic_thermal=NandThermalConfig(assumption="existing RC sensitivity"),
        wear_model=NandWearConfig(assumption="existing wear sensitivity"))
    original = asdict(config)
    slowed = _derate_physical_memory_config(config,
        point(memory_bandwidth_scale=.5, latency_scale=2.))
    assert asdict(config) == original
    for name in ("host_bandwidth_gb_s", "host_read_bandwidth_gb_s",
                 "host_write_bandwidth_gb_s", "internal_bandwidth_gb_s"):
        assert getattr(slowed, name) == getattr(config, name) * .5
    for name in ("page_read_ns", "page_program_ns", "block_erase_ns", "front_ns"):
        assert getattr(slowed, name) == getattr(config, name) * 2.
    assert slowed.dynamic_thermal == config.dynamic_thermal
    assert slowed.wear_model == config.wear_model
    assert slowed.capacity_bytes == config.capacity_bytes
    assert slowed.computed_capacity_bytes == config.computed_capacity_bytes
    # Isolate static scaling from optional feedback when comparing Core time.
    for operation in (Operation.READ, Operation.WRITE, Operation.ERASE):
        request = AccessRequest("transaction", operation, 0, config.page_bytes)
        before = NandCore(replace(config, dynamic_thermal=None, wear_model=None)).submit(request)
        after = NandCore(replace(slowed, dynamic_thermal=None, wear_model=None)).submit(request)
        assert after.completion_ns > before.completion_ns
        assert after.transfer_bytes == before.transfer_bytes


@pytest.mark.parametrize("memory_id", ["hbm0", "hbf0"])
@pytest.mark.parametrize("scales", [
    {"memory_bandwidth_scale": .5, "link_bandwidth_scale": .5},
    {"latency_scale": 2.},
])
def test_public_operating_point_changes_endpoint_core_not_only_preview(memory_id, scales):
    base = build_study_scenario("B_all_kv_hbf", prompt=32, batch=1, output=2)
    components = tuple(replace(c, metadata={**c.metadata, "thermal_domain_id": "memory"})
                       if c.component_id == memory_id else c for c in base.hardware.components)
    links = tuple(replace(link, metadata={**link.metadata, "thermal_domain_id": "memory"})
                  if memory_id in (link.source_component, link.target_component) else link
                  for link in base.hardware.links)
    base = replace(base, hardware=replace(base.hardware, components=components, links=links))
    before_component = base.hardware.get_component(memory_id)
    before_config = parse_physical_memory_config(before_component.metadata["physical_memory_config"])
    changed = apply_thermal_operating_point(base, point(**scales))
    after_component = changed.hardware.get_component(memory_id)
    after_config = parse_physical_memory_config(after_component.metadata["physical_memory_config"])
    assert after_config != before_config
    assert before_config == parse_physical_memory_config(
        base.hardware.get_component(memory_id).metadata["physical_memory_config"])
    assert after_component.capacity_bytes == before_component.capacity_bytes
    assert changed.hardware.get_component("hostmem0") == base.hardware.get_component("hostmem0")
    for operation in (AccessKind.READ, AccessKind.WRITE):
        before = resolve_service(before_component).price(operation, 16384, page_offset_bytes=0)
        after = resolve_service(after_component).price(operation, 16384, page_offset_bytes=0)
        assert after["service_ns"] > before["service_ns"]
        assert after["physical_bytes"] == before["physical_bytes"]


def test_disabled_operating_point_remains_identity():
    base = build_study_scenario("A_hbm_only", prompt=32, batch=1, output=2)
    assert apply_thermal_operating_point(base, ThermalOperatingPoint("unused")) is base


def test_core_only_latency_change_invalidates_off_domain_gpu_samples():
    base = build_study_scenario("A_hbm_only", prompt=32, batch=1, output=2)
    components = tuple(replace(c, metadata={**c.metadata, "thermal_domain_id": "memory"})
                       if c.component_id == "hbm0" else c for c in base.hardware.components)
    base = replace(base, hardware=replace(base.hardware, components=components))
    with patch.object(thermal, "_invalidate_kernel_samples",
                      wraps=thermal._invalidate_kernel_samples) as invalidate:
        changed = apply_thermal_operating_point(base, point(latency_scale=2.))
    before = base.hardware.get_component("hbm0").metadata
    after = changed.hardware.get_component("hbm0").metadata
    assert before["memory_service"] == after["memory_service"]
    assert before["physical_memory_config"] != after["physical_memory_config"]
    assert base.hardware.links == changed.hardware.links
    assert invalidate.call_count == len(base.component_profiles["gpu"])
    assert all("thermal memory/route changed" in call.args[1]
               for call in invalidate.call_args_list)
