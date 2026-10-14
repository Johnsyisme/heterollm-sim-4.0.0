"""Shared-wire direction history, rollback and finite reservation boundaries."""
import copy

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_transfer import ResourceTimeline


def test_direction_history_and_lane_choice_are_slot_local():
    timeline = ResourceTimeline(lane_available={"wire": [0., 0.]})
    first = timeline.reserve("wire", 0., 100., direction="read", switch_ns=50.)
    second = timeline.reserve("wire", 0., 101., direction="write", switch_ns=50.)
    assert (first.start_ns, second.start_ns) == (0., 0.)
    # Lane 0 is idle first, but lane 1 is ready to write sooner after turnover.
    third = timeline.reserve("wire", 0., 100., direction="write", switch_ns=50.)
    assert (third.start_ns, third.end_ns) == (101., 201.)
    # The last global direction is write; lane 0 still remembers its read.
    fourth = timeline.reserve("wire", 0., 1., direction="write", switch_ns=50.)
    assert fourth.start_ns == 150.
    assert timeline.busy_ns["wire"] == 302.


def test_aliases_share_direction_calendar_and_byte_accounting():
    timeline = ResourceTimeline(resource_aliases={"read-port": "wire", "write-port": "wire"})
    timeline.transfer("read-port", 64, 0., 8., direction="read")
    result = timeline.transfer("write-port", 32, 0., 8., direction="write", switch_ns=3.)
    assert (result.resource_id, result.start_ns, result.end_ns) == ("wire", 11., 15.)
    assert timeline.available_ns("read-port") == 15.
    assert timeline.bytes_moved == {"wire": 96}
    assert timeline.busy_ns == {"wire": 12.}


def test_alias_chain_is_idempotent_and_cycles_fail_before_reservation():
    timeline = ResourceTimeline(resource_aliases={"logical": "relay", "relay": "wire"})
    result = timeline.transfer("logical", 64, 0., 8.)
    assert result.resource_id == "wire"
    assert timeline.bytes_moved == {"wire": 64}
    assert timeline.available_ns("logical") == 8.
    timeline.lane_available["wire"][0] = 100.
    assert timeline.available_ns("logical") == 100.
    invalid = ResourceTimeline(resource_aliases={"a": "b", "b": "a"})
    before = copy.deepcopy(vars(invalid))
    with pytest.raises(ValueError, match="cycle"):
        invalid.transfer("a", 1, 0., 1.)
    assert vars(invalid) == before


def test_snapshot_restores_direct_link_details_and_slot_direction_repeatedly():
    runtime = PhysicalRuntimeContext()
    timeline = runtime.timeline
    timeline.lane_available["wire"] = [0., 0.]
    timeline.metrics_snapshot(max_intervals=10)
    timeline.transfer("wire", 64, 0., 8., direction="read")
    saved = runtime.snapshot()
    before = copy.deepcopy(vars(timeline))
    for _ in range(2):
        timeline.transfer("wire", 64, 0., 8., direction="write", switch_ns=5.)
        timeline.transfer("link", 64, 0., 8.)
        runtime.restore(saved)
        assert vars(timeline) == before


def test_overflowing_reservation_rejects_without_publishing_infinity():
    timeline = ResourceTimeline()
    before = copy.deepcopy(vars(timeline))
    with pytest.raises(ValueError, match="finite timing"):
        timeline.reserve("wire", 1e308, 1e308)
    assert vars(timeline) == before
