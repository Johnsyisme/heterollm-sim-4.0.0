"""Catalog migration retains physical experiments and applies their scan axes."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import accept_hbf_hbm as acceptance
import hbf_hbm_budget as budget
import hbf_hbm_media_experiments as media
import hbf_hbm_sessions as sessions
from heterollm_sim.memory_types import AccessRequest, make_hbf_config, make_hbm_config


def physical_fields(config):
    fields = asdict(config)
    fields.pop("metadata")  # Catalog provenance is intentionally new.
    return fields


@pytest.mark.parametrize("source", ["acceptance", "sessions", "migration", "package"])
def test_catalog_migration_preserves_workload_controls_and_uses_standard_hbm_geometry(source):
    page = 16384
    if source == "acceptance":
        actual = (acceptance.hbm_config(), acceptance.hbf_config())
        expected = (make_hbm_config(capacity_bytes=32*1024**2,
                     interface_bandwidth_gb_s=488., burst_interval_ns=0.),
                    make_hbf_config(capacity_bytes=32*1024**2, channels=8, parallel_units=8))
    elif source == "sessions":
        components = sessions._hardware(hbm_pages=8, hbf_pages=32).components
        actual = tuple(c.metadata["physical_memory_config"] for c in components)
        expected = (make_hbm_config(capacity_bytes=8*page,
                     interface_bandwidth_gb_s=512., burst_interval_ns=0.),
                    make_hbf_config(capacity_bytes=32*page, channels=8, parallel_units=8,
                     host_bandwidth_gb_s=488., host_read_bandwidth_gb_s=488.,
                     host_write_bandwidth_gb_s=27.2))
    elif source == "migration":
        components = media._migration_hardware().components
        actual = tuple(c.metadata["physical_memory_config"] for c in components[:2])
        expected = (make_hbf_config(capacity_bytes=128*page, channels=32, parallel_units=32,
                     host_bandwidth_gb_s=128., internal_bandwidth_gb_s=128.,
                     host_read_bandwidth_gb_s=128., host_write_bandwidth_gb_s=32.),
                    make_hbm_config(capacity_bytes=128*page,
                     interface_bandwidth_gb_s=256., burst_interval_ns=0.))
    else:
        actual = (budget._core(budget.HBM_SLOT).config, budget._core(budget.HBF_SLOT).config)
        expected = (make_hbm_config(capacity_bytes=4*page, channels=4,
                     pseudo_channels_per_channel=1, interface_bandwidth_gb_s=256.,
                     read_bandwidth_gb_s=256., write_bandwidth_gb_s=256., burst_interval_ns=0.),
                    make_hbf_config(capacity_bytes=2*page, channels=2, parallel_units=2,
                     host_bandwidth_gb_s=160., host_read_bandwidth_gb_s=160.,
                     host_write_bandwidth_gb_s=20., write_buffer_bytes=page))
    preserved = ("capacity_bytes", "max_outstanding_requests", "max_expanded_segments")
    for current, old in zip(actual, expected):
        if current.kind.value != "HBM":
            assert physical_fields(current) == physical_fields(old)
            assert current.metadata["parameter_basis"] == "synthetic_research_assumptions"
            continue
        assert {key: getattr(current, key) for key in preserved} == {key: getattr(old, key) for key in preserved}
        assert current.control_mode == "jedec_hbm"
        assert current.read_latency_ns / current.command_clock_ns == 32
        assert current.write_latency_ns / current.command_clock_ns == 8
        assert current.column_different_group_ns == 2 * current.command_clock_ns
        assert current.physical_interface_bandwidth_gb_s == old.physical_interface_bandwidth_gb_s
        assert all(current.directional_bandwidth_gb_s(op) == old.directional_bandwidth_gb_s(op)
                   for op in ("read", "write"))
        assert current.generation == "HBM3E"
        assert (current.stacks, current.channels, current.pseudo_channels_per_channel,
                current.lane_count, current.row_bytes, current.burst_bytes) == (1, 16, 2, 32, 1024, 32)
        assert current.metadata["geometry_status"] == "standard_organization"
        assert current.metadata["reference_profile_id"] == "hbm3e_reference"
        assert current.metadata["vendor_verified"] is False
        assert current.computed_capacity_bytes >= current.capacity_bytes
    assert all(c.metadata["physical_config_profile_id"].startswith("research_") for c in actual)


def test_program_scan_changes_executed_nand_time_after_catalog_resolution():
    outcomes = []
    for latency in (75000., 150000.):
        core = media._write_core(write_buffer_bytes=0, program_ns=latency)
        assert core.config.metadata["physical_config_profile_id"] == "research_hbf_media_write"
        row = core.submit(AccessRequest("program", "write", 0, 16384))
        assert row.physical_write_bytes == 16384
        outcomes.append(row.completion_ns)
    assert outcomes[1] - outcomes[0] == 75000.
