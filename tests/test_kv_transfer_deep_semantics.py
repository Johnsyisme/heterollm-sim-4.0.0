"""Logical KV transactions, chunk observations and shared fabric aliases."""
from types import SimpleNamespace

import pytest

from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_pool import KvPoolTransferRequest
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config


def component(name, config):
    return ComponentSpec(name, name, capacity_bytes=config.capacity_bytes,
        read_bandwidth_gbps=800, write_bandwidth_gbps=80,
        metadata={"physical_memory_config": config})


def page(owner="hbf", size=49152):
    return SimpleNamespace(logical_page_id=0, owner_component=owner, bytes=size,
        allocation_generation=0, request_id="request", layer_group="kv")


def request(p, target):
    return KvPoolTransferRequest(p.logical_page_id, p.request_id, p.layer_group,
        p.owner_component, target, p.bytes, p.allocation_generation, 0, 1)


def executor(*, buffer_bytes=32768, reject=True, observe=False):
    cfg = make_hbf_config(capacity_bytes=1048576, page_bytes=16384,
        write_buffer_bytes=buffer_bytes,
        partial_page_policy="reject" if reject else "read_modify_write",
        detailed_observability=observe)
    return PhysicalKVTransferExecutor([component("hbf", cfg)])


def test_late_rejected_chunk_reverts_entire_logical_page_write():
    e = executor()
    p = page()
    e.bind_page(p)
    before = e.snapshot()
    for _ in range(2):
        with pytest.raises(ValueError, match="partial NAND page write"):
            e.write_page(p, arrival_ns=0, size_bytes=32769)
        assert e.snapshot() == before
    result = e.write_page(p, arrival_ns=0, size_bytes=32768)
    fresh = executor()
    fresh.bind_page(p)
    assert result == fresh.write_page(p, arrival_ns=0, size_bytes=32768)


@pytest.mark.parametrize("operation", ["read_page", "write_page"])
def test_failed_page_statistics_roll_back_completed_core_work(monkeypatch, operation):
    e = executor(reject=False)
    p = page()
    e.bind_page(p)
    before = e.snapshot()

    def fail(*_args):
        assert e.runtime.timeline.bytes_moved
        raise ValueError("statistics failed")

    monkeypatch.setattr(e, "_sum_counters", fail)
    with pytest.raises(ValueError, match="statistics failed"):
        getattr(e, operation)(p, size_bytes=16384)
    assert e.snapshot() == before


def test_chunked_observations_preserve_all_events_and_one_geometry():
    e = executor(buffer_bytes=4096, reject=False, observe=True)
    p = page(size=16384)
    e.bind_page(p)
    result = e.write_page(p)
    observations = result["nand_observations"]
    assert result["pages_read"] == result["pages_programmed"] == 4
    assert len(observations["events"]) == 8
    config = e._configs["hbf"]
    plane_count = (config.channels * config.targets_per_channel * config.dies_per_target
                   * config.luns_per_die * config.planes_per_lun)
    assert sum(observations["configured_planes_by_owner"].values()) == plane_count
    assert observations["physical_plane_imbalance"]["configured_planes"] == plane_count
    assert observations["read_queue_depth"]["operations"] == 4
    assert observations["program_queue_depth"]["operations"] == 4
    assert sum(row["physical_bytes"] for row in observations["events"]) == result["physical_bytes"]


def transfer_executor(p, *, target_reject=False, topology=None):
    hbm = make_hbm_config(capacity_bytes=1048576)
    hbf = make_hbf_config(capacity_bytes=1048576, page_bytes=16384,
        write_buffer_bytes=32768,
        partial_page_policy="reject" if target_reject else "read_modify_write")
    e = PhysicalKVTransferExecutor([component("hbm", hbm), component("hbf", hbf)],
                                   topology=topology)
    e.bind_page(p)
    return e


def test_async_phase_failure_preserves_source_and_interleaved_work():
    p = page("hbm", size=32769)
    e = transfer_executor(p, target_reject=True)
    plan = e.prepare_async(request(p, "hbf"))
    e.read_page(p, arrival_ns=plan.completion_ns, size_bytes=64)
    before_phase = e.snapshot()
    for _ in range(2):
        with pytest.raises(ValueError, match="partial NAND page write"):
            plan.advance()
        assert e.snapshot() == before_phase
    assert e.counters["read_bytes"] == p.bytes + 64
    assert e.counters["write_bytes"] == 0
    plan.cancel()
    assert e._bindings[p.logical_page_id].component_id == "hbm"
    assert not e.runtime.allocators[e.services["hbf"].physical_owner].allocations()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transfer_link_accounting_uses_same_alias_as_calendar(asynchronous):
    p = page("hbm", size=16384)
    hop = SimpleNamespace(resource_id="link-view", link_id="link", service_ref=None,
                          transfer_ns=lambda _size: 100.0)
    topology = SimpleNamespace(route=lambda *_args, **_kwargs: (hop,))
    e = transfer_executor(p, topology=topology)
    e.runtime.timeline.resource_aliases = {"link-view": "wire"}
    e.runtime.timeline.reserve("wire", 0, 100000)
    plan = e.prepare_async(request(p, "hbf")) if asynchronous else e(request(p, "hbf"))
    while plan.advance is not None:
        plan = plan.advance()
    plan.commit()
    assert e.runtime.timeline.bytes_moved["wire"] == p.bytes
    assert "link-view" not in e.runtime.timeline.bytes_moved
    assert "link-view" not in e.runtime.timeline.lane_available
    phases = [phase for phase in e.events[-1]["phases"] if phase["kind"] == "link"]
    assert len(phases) == 1
    assert phases[0]["resource_id"] == "wire"
    assert phases[0]["start_ns"] >= 100000
