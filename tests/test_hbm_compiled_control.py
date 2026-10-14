"""Compiled HBM must preserve the authoritative normal-operation controller."""
from dataclasses import replace

import pytest

from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, DramConfig


def configuration(**changes):
    values = dict(kind="HBM", generation="HBM3E", channels=2,
        pseudo_channels_per_channel=2, stacks=1, dies_per_stack=4,
        bank_groups_per_rank=8, banks_per_group=4, rows_per_bank=32,
        row_bytes=1024, burst_bytes=32, data_width_bits=32, data_rate_mt_s=4000.,
        bank_geometry_scope="per_stack", stack_interface_mode="independent",
        control_mode="jedec_hbm", command_clock_ns=1., activate_command_cycles=1.5,
        precharge_command_cycles=.5, column_command_cycles=1., activated_row_msb=4,
        activate_to_read_ns=4., activate_to_write_ns=3., precharge_to_activate_ns=3.,
        minimum_active_ns=7., activate_cycle_ns=10., read_to_precharge_ns=3.,
        write_to_precharge_ns=4., read_latency_ns=8., write_latency_ns=4.,
        column_same_group_ns=4., column_different_group_ns=2.,
        column_different_sid_ns=3., precharge_to_precharge_ns=2.,
        activate_same_group_ns=5., activate_different_group_ns=2.,
        write_to_read_same_group_ns=3., write_to_read_different_group_ns=2.,
        read_to_write_command_ns=7., write_to_read_ns=1., read_to_write_ns=2.)
    values.update(changes)
    return DramConfig(**values)


def assert_same(reference, numeric, request):
    reference.timeline.metrics_snapshot()
    numeric.timeline.metrics_snapshot()
    expected = _Controller(reference, request).execute()
    actual = execute_hbm_numeric(numeric, request)
    assert actual is not None
    assert actual.completion_ns == expected.completion_ns
    assert actual.logical_bytes == expected.logical_bytes
    assert actual.transfer_bytes == expected.transfer_bytes
    assert {k: v for k, v in actual.counters.items() if k != "compiled_hbm_float64"} == expected.counters
    assert numeric._hbm_control_state == reference._hbm_control_state
    assert numeric._banks == reference._banks
    for field in ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals", "lane_available"):
        assert getattr(numeric.timeline, field) == getattr(reference.timeline, field), field


@pytest.mark.parametrize("generation,clock,refresh", [
    ("HBM2E", 1., False), ("HBM3E", 1., False),
    ("HBM3E", .7, True), ("HBM4", .625, True),
])
def test_compiled_matches_mixed_commands_and_refresh(generation, clock, refresh):
    hbm2 = generation == "HBM2E"
    c = configuration(generation=generation, command_clock_ns=clock,
        activated_row_msb=None if hbm2 else 4,
        activate_command_cycles=2. if hbm2 else 1.5,
        precharge_command_cycles=1. if hbm2 else .5,
        data_width_bits=64 if hbm2 else 32, data_rate_mt_s=(2000. if hbm2 else 4000.) / clock,
        read_latency_ns=(32. if generation == "HBM4" else 8.) * clock,
        write_latency_ns=4. * clock, refresh_policy="all_bank" if refresh else "disabled",
        refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    row_span = c.lane_count * c.bank_count * c.row_bytes
    commands = [
        ("read", 3, 30007, 0.), ("write", 65, 9000, 0.),
        ("read", row_span + 5, 45000, 5.),
        ("write", 16 * row_span + 1, 8100, 8.),
        ("read", row_span, 33001, 20000.),
    ]
    for index, (op, address, count, time) in enumerate(commands):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, time))


def test_compiled_preserves_interleaving_stacks_and_direction_budget():
    c = configuration(stacks=2, interleave_bytes=128,
                      write_bandwidth_gb_s=64., read_bandwidth_gb_s=128.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (reference, numeric):
        core.timeline.reserve("dram:row_command:0:0", 0., 100.)
        core.timeline.reserve("dram:column_command:1:1", 0., 70.)
        core.timeline.reserve("dram:data:3", 0., 200., direction="write")
    for index, op in enumerate(("read", "write", "read")):
        assert_same(reference, numeric, AccessRequest(str(index), op, index * 4111, 270011, index * 3.))


def test_numeric_declines_nonzero_faw_without_mutation():
    c = configuration(four_activate_window_ns=20.)
    core = DramCore(c, capture_details=False)
    assert execute_hbm_numeric(core, AccessRequest("a", "read", 0, 4096)) is None
    assert not core._hbm_control_state
    assert not core._banks
    assert not core.timeline.ready_ns


def test_flat_bank_group_keeps_separate_sid_histories():
    c = configuration(bank_groups_per_rank=1, banks_per_group=64,
                      column_different_sid_ns=9., refresh_policy="all_bank",
                      refresh_interval_ns=900., refresh_cycle_ns=45.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, count) in enumerate((
            ("read", 0, 163841), ("write", 513, 81921),
            ("read", 64 * 4 * 1024, 131071), ("read", 2048, 32000))):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, 2. * index))


def test_large_idle_refresh_folding_keeps_exact_state_and_accounting():
    c = configuration(refresh_policy="all_bank", refresh_interval_ns=100., refresh_cycle_ns=20.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("warm", "read", 0, 4096))
    assert_same(reference, numeric, AccessRequest("after-idle", "write", 0, 8192, 1.e9))


def test_unrelated_kernel_aliases_do_not_disable_compiled_entry():
    c = configuration()
    core = DramCore(c, capture_details=False)
    core.timeline.resource_aliases = {"unrelated:port": "owner:wire", "dram:data:0": "dram:data:0"}
    result = core.submit(AccessRequest("aliased-kernel", "read", 0, 4096))
    assert result.counters["compiled_hbm_float64"] is True
    assert result.counters["read_commands"] == 128
    blocked = DramCore(c, capture_details=False)
    blocked.timeline.resource_aliases = {"dram:data:0": "dram:data:1"}
    assert execute_hbm_numeric(blocked, AccessRequest("shared-wire", "read", 0, 4096)) is None
    assert not blocked._hbm_control_state


@pytest.mark.parametrize("refresh", [False, True])
def test_maximum_active_deadlines_match_before_reads_writes_and_idle(refresh):
    c = configuration(maximum_active_ns=50., refresh_policy="all_bank" if refresh else "disabled",
                      refresh_interval_ns=400. if refresh else 0., refresh_cycle_ns=40. if refresh else 0.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for index, (op, address, count, arrival) in enumerate((
            ("read", 0, 256, 0.), ("read", 0, 32, 49.),
            ("write", 128, 96, 95.), ("read", 32, 128, 20000.))):
        assert_same(reference, numeric, AccessRequest(str(index), op, address, count, arrival))


@pytest.mark.parametrize("generation", ["HBM3E", "HBM4"])
def test_real_builder_maximum_active_and_idle_match(generation):
    from heterollm_sim.hbm_geometry import make_hbm_geometry_config
    c = DramConfig(**make_hbm_geometry_config(generation=generation, capacity_bytes=36 * 2 ** 30,
        read_bandwidth_gbps=9600., write_bandwidth_gbps=9600., die_density_gbits=24, dies_per_stack=12))
    assert c.maximum_active_ns == 35100.
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    assert_same(reference, numeric, AccessRequest("normal", "read", 0, 32768))
    assert_same(reference, numeric, AccessRequest("idle", "write", 0, 2048, 1.e6))


def test_maximum_active_delays_activation_until_reserved_dq_slot():
    c = configuration(maximum_active_ns=50.)
    reference, numeric = DramCore(c, capture_details=False), DramCore(c, capture_details=False)
    for core in (reference, numeric):
        core.timeline.reserve("dram:data:0", 0., 1000.)
    assert_same(reference, numeric, AccessRequest("late-dq", "read", 0, 32))


def test_numeric_rejects_an_impossible_maximum_active_window():
    c = configuration(minimum_active_ns=1., maximum_active_ns=5.)
    request = AccessRequest("too-short", "read", 0, 32)
    for numeric in (False, True):
        core = DramCore(c, capture_details=False)
        with pytest.raises(ValueError, match="tRASmax activation window"):
            execute_hbm_numeric(core, request) if numeric else _Controller(core, request).execute()


def test_refresh_cannot_precharge_an_open_row_after_its_deadline():
    c = configuration(maximum_active_ns=2000., refresh_policy="all_bank",
                      refresh_interval_ns=1000., refresh_cycle_ns=40.)
    for numeric in (False, True):
        core = DramCore(c, capture_details=False)
        warm = AccessRequest("warm", "read", 0, 32)
        execute_hbm_numeric(core, warm) if numeric else _Controller(core, warm).execute()
        core.timeline.reserve("dram:row_command:0:0", 0., 3000.)
        request = AccessRequest("blocked", "read", 0, 32, 1000.)
        with pytest.raises(ValueError, match="tRASmax.*refresh PRE"):
            execute_hbm_numeric(core, request) if numeric else _Controller(core, request).execute()
