"""Repeatable single-device HBM/HBF physical-memory acceptance experiments.

Run ``python scripts/accept_hbf_hbm.py --output artifacts/hbf_hbm.json``.
No installation or third-party dependencies are required. Results are memory
microbenchmarks, not LLM token throughput or calibrated silicon predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.ir import (
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LinkSpec,
    PortSpec,
    RequestSpec,
)
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_transfer import ResourceTimeline, summarize_batch
from heterollm_sim.memory_types import (
    AccessRequest, DramConfig, MemoryKind, NandConfig, Operation,
    make_hbf_config, make_hbm_config, make_ssd_config,
)
from heterollm_sim.nand_core import NandCore
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import BatchCost, _OnlineRuntime, compile_serving_plan

PAGE = 16_384
CAPACITY = 32 * 1024 * 1024


def _serving_paged_scenario(
    *, hbf_capacity: int, requests=(), hbm_capacity: int = 32 * PAGE
):
    """Build a tiny physical HBM+HBF serving scenario for acceptance."""

    base = build_reference_scenario()
    hbm = base.hardware.get_component("hbm0")
    hbm = replace(
        hbm,
        capacity_bytes=hbm_capacity,
        metadata={
            **hbm.metadata,
            "physical_memory_config": make_hbm_config(
                capacity_bytes=hbm_capacity,
                interface_bandwidth_gb_s=512.0,
                burst_interval_ns=0.0,
            ),
        },
    )
    hbf_physical = make_hbf_config(
        capacity_bytes=hbf_capacity,
        channels=8,
        parallel_units=8,
        host_bandwidth_gb_s=488.0,
        host_read_bandwidth_gb_s=488.0,
        host_write_bandwidth_gb_s=27.2,
    )
    hbf = ComponentSpec(
        component_id="hbf0",
        kind="hbf",
        ports=(
            PortSpec(
                port_id="pcie0",
                protocol="PCIe",
                role="endpoint",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                payload="coherent_dma",
            ),
        ),
        capacity_bytes=hbf_capacity,
        read_bandwidth_gbps=488.0 * 8.0,
        write_bandwidth_gbps=27.2 * 8.0,
        metadata={
            "physical_memory_config": hbf_physical,
            "access_mode": "remote_flash",
        },
    )
    hardware = replace(
        base.hardware,
        components=tuple(
            hbm if component.component_id == "hbm0" else component
            for component in base.hardware.components
        ) + (hbf,),
        links=base.hardware.links
        + (
            LinkSpec(
                link_id="cpu-hbf-pcie",
                source_component="cpu0",
                source_port="pcie0",
                target_component="hbf0",
                target_port="pcie0",
                protocol="PCIe",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                latency_ns=100.0,
                payload="coherent_dma",
            ),
        ),
        require_connected=False,
    )
    policy = KVCachePolicy(
        cache_component="hbm0",
        offload_component="hbf0",
        tokens_per_page=16,
        dtype="int8",
        layout_mode="paged_pool",
        pool_components=("hbm0", "hbf0"),
        preemption_mode="swap",
    )
    scheduler = replace(
        base.workload.scheduler,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        max_num_ubatch_tokens=32,
        prefill_chunk_tokens=32,
        preemption_enabled=True,
        preemption_policy="swap",
    )
    workload = replace(
        base.workload,
        requests=tuple(requests),
        request_count=len(requests),
        scheduler=scheduler,
        mtp=None,
    )
    return replace(
        base,
        hardware=hardware,
        placement=replace(
            base.placement,
            hardware_name=hardware.name,
            kv_policy=policy,
        ),
        workload=workload,
    )


def check_serving_paged_swap() -> dict[str, Any]:
    """Verify automatic multi-page swap/resume and atomic capacity rollback."""

    requests = (
        RequestSpec(
            request_id="swap-low",
            arrival_ns=0.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=0,
        ),
        RequestSpec(
            request_id="swap-high",
            arrival_ns=1.0,
            prompt_tokens=32,
            output_tokens=8,
            priority=10,
        ),
    )
    plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=16 * 1024 * 1024,
            requests=requests,
        )
    )
    runtime = _OnlineRuntime(
        plan,
        lambda *_: BatchCost(100.0),
        apply_measurement_start_residency=False,
    )
    result = runtime.run()
    states = runtime.states
    low = states["swap-low"]
    high = states["swap-high"]
    swap_out = [event for event in runtime.events if event.event_type == "kv_swap_out"]
    swap_in = [event for event in runtime.events if event.event_type == "kv_swap_in"]
    event_times = [float(event.timestamp_ns) for event in runtime.events]
    require(event_times == sorted(event_times),
            "serving events must be returned in non-decreasing timestamp order")
    low_events = [
        event for event in runtime.events if event.request_id == "swap-low"
    ]
    low_event_by_type = {}
    for event in low_events:
        low_event_by_type.setdefault(event.event_type, []).append(event)
    low_out_start = low_event_by_type.get("kv_swap_out_start", [None])[0]
    low_out_end = low_event_by_type.get("kv_swap_out", [None])[-1]
    transfer_events = [
        event for event in runtime.events
        if event.event_type in {
            "kv_pool_transfer_start", "kv_pool_transfer_complete"
        }
        and low_out_start is not None
        and low_out_end is not None
        and low_out_start.timestamp_ns <= event.timestamp_ns <= low_out_end.timestamp_ns
    ]
    for event in transfer_events:
        low_event_by_type.setdefault(event.event_type, []).append(event)
    required_low_types = (
        "kv_swap_out_start",
        "kv_pool_transfer_start",
        "kv_pool_transfer_complete",
        "kv_swap_out",
    )
    require(all(low_event_by_type.get(kind) for kind in required_low_types),
            "paged swap must expose ordered physical transfer boundaries")
    low_order = [
        low_event_by_type[kind][0].timestamp_ns
        for kind in required_low_types[:3]
    ] + [low_event_by_type["kv_swap_out"][-1].timestamp_ns]
    require(low_order == sorted(low_order),
            "swap start, page transfer, completion, and swap end are out of order")
    low_in_events = [
        event for event in runtime.events if event.request_id == "swap-low"
        and event.event_type in {"kv_swap_in_start", "kv_swap_in"}
    ]
    require(
        any(event.event_type == "kv_swap_in_start" for event in low_in_events)
        and any(event.event_type == "kv_swap_in" for event in low_in_events),
        "paged restore must expose swap-in boundaries",
    )
    low_in_start = next(
        event for event in low_in_events if event.event_type == "kv_swap_in_start"
    )
    low_in_end = next(
        event for event in reversed(low_in_events) if event.event_type == "kv_swap_in"
    )
    restore_transfer_events = [
        event for event in runtime.events
        if event.event_type in {
            "kv_pool_transfer_start", "kv_pool_transfer_complete"
        }
        and low_in_start.timestamp_ns <= event.timestamp_ns <= low_in_end.timestamp_ns
    ]
    require(
        restore_transfer_events
        and restore_transfer_events[0].event_type == "kv_pool_transfer_start"
        and restore_transfer_events[-1].event_type == "kv_pool_transfer_complete",
        "swap-in page transfers must be enclosed by restore boundaries",
    )
    require(all(state.status.value == "finished" for state in states.values()),
            "automatic paged serving swap must finish every request")
    require(runtime.preemptions >= 1 and runtime.swap_events >= 1,
            "higher-priority arrival must trigger a serving preemption and swap")
    require(swap_out and swap_in and swap_out[0].details["pages"] == 2,
            "automatic swap must transfer the complete multi-page KV working set")
    require(runtime.swap_transfer_time_ns > 0.0,
            "serving swap time must come from physical page transfer events")
    require(low.swaps == 1 and high.swaps == 0,
            "only the lower-priority request should be swapped")
    require(not runtime.ledger.paged_pool.pages() and
            all(value == 0 for value in runtime.physical_ledger.used_bytes.values()),
            "finished requests must release resident and offloaded pages")

    rollback_plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=PAGE,
            requests=(
                RequestSpec(
                    request_id="rollback",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
            ),
        )
    )
    rollback_runtime = _OnlineRuntime(
        rollback_plan,
        lambda *_: BatchCost(0.0),
        apply_measurement_start_residency=False,
    )
    rollback_request = rollback_runtime.states["rollback"]
    pool = rollback_runtime.ledger.paged_pool
    require(pool is not None and rollback_runtime.ledger.resize(rollback_request, 2),
            "rollback setup must allocate two resident pages")
    before_owners = tuple(page.owner_component for page in pool.request_pages("rollback"))
    require(not rollback_runtime.ledger.offload(rollback_request),
            "offload must fail when the HBF target cannot fit all pages")
    after_owners = tuple(page.owner_component for page in pool.request_pages("rollback"))
    require(after_owners == before_owners and rollback_request.kv_pages == 2 and
            rollback_request.swapped_pages == 0,
            "failed multi-page offload must roll back every page and ledger field")
    require(rollback_runtime.physical_ledger.used_bytes.get("hbm0") == 2 * PAGE and
            rollback_runtime.physical_ledger.used_bytes.get("hbf0") == 0,
            "failed offload must not leak target capacity or source accounting")

    restore_plan = compile_serving_plan(
        _serving_paged_scenario(
            hbf_capacity=2 * PAGE,
            hbm_capacity=2 * PAGE,
            requests=(
                RequestSpec(
                    request_id="restore-victim",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
                RequestSpec(
                    request_id="restore-owner",
                    arrival_ns=0.0,
                    prompt_tokens=32,
                    output_tokens=8,
                ),
            ),
        )
    )
    restore_runtime = _OnlineRuntime(
        restore_plan,
        lambda *_: BatchCost(1.0),
        apply_measurement_start_residency=False,
    )
    victim = restore_runtime.states["restore-victim"]
    owner = restore_runtime.states["restore-owner"]
    restore_pool = restore_runtime.ledger.paged_pool
    require(
        restore_pool is not None
        and restore_runtime.ledger.resize(victim, 2)
        and restore_runtime.ledger.offload(victim)
        and restore_runtime.ledger.resize(owner, 2),
        "restore rollback setup must occupy HBF and all HBM pages",
    )
    restore_before = tuple(
        page.owner_component
        for page in restore_pool.request_pages("restore-victim")
    )
    require(
        not restore_runtime.ledger.restore(victim),
        "restore must fail when another request fills the HBM target",
    )
    restore_after = tuple(
        page.owner_component
        for page in restore_pool.request_pages("restore-victim")
    )
    require(
        restore_after == restore_before
        and victim.kv_pages == 0
        and victim.swapped_pages == 2
        and restore_runtime.physical_ledger.used_bytes.get("hbm0") == 2 * PAGE
        and restore_runtime.physical_ledger.used_bytes.get("hbf0") == 2 * PAGE,
        "failed restore must preserve offloaded pages and both capacity ledgers",
    )
    return {
        "preemptions": runtime.preemptions,
        "swap_events": runtime.swap_events,
        "swap_out_pages": swap_out[0].details["pages"],
        "swap_transfer_time_ns": runtime.swap_transfer_time_ns,
        "makespan_ns": result.makespan_ns,
        "swap_low_ttft_ns": result.request_metrics["swap-low"].ttft_ns,
        "swap_low_tpot_ns": result.request_metrics["swap-low"].tpot_ns,
        "kv_peak_pages": result.kv_metrics.peak_used_pages,
        "physical_swap_bytes": result.kv_metrics.physical_swap_bytes,
        "requests_finished": len(result.request_states),
        "rollback_preserved_pages": len(after_owners),
        "rollback_target_used_bytes": rollback_runtime.physical_ledger.used_bytes.get("hbf0"),
        "restore_rollback_preserved_pages": len(restore_after),
        "restore_target_used_bytes": restore_runtime.physical_ledger.used_bytes.get("hbm0"),
    }


def check_slo_semantics() -> dict[str, Any]:
    """SLO thresholds classify the same fixed trace without retiming it."""

    def run(deadline_ns: float):
        scenario = _serving_paged_scenario(
            hbf_capacity=16 * 1024 * 1024,
            requests=(RequestSpec(
                request_id="slo",
                arrival_ns=0.0,
                prompt_tokens=16,
                output_tokens=4,
                deadline_ns=deadline_ns,
            ),),
        )
        runtime = _OnlineRuntime(
            compile_serving_plan(scenario),
            lambda *_: BatchCost(100.0),
            apply_measurement_start_residency=False,
        )
        return runtime.run()

    loose = run(1_000_000.0)
    tight = run(1.0)
    loose_metric = loose.request_metrics["slo"]
    tight_metric = tight.request_metrics["slo"]
    require(equal(loose.makespan_ns, tight.makespan_ns),
            "changing an SLO threshold must not retime a fixed single-request trace")
    require(loose_metric.deadline_met is True and tight_metric.deadline_met is False,
            "SLO threshold must change qualification without changing execution")
    return {
        "makespan_ns": loose.makespan_ns,
        "loose_deadline_met": loose_metric.deadline_met,
        "tight_deadline_met": tight_metric.deadline_met,
        "loose_ttft_ns": loose_metric.ttft_ns,
        "tight_ttft_ns": tight_metric.ttft_ns,
    }


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def hbm_config(**changes: Any) -> DramConfig:
    return replace(make_hbm_config(
        capacity_bytes=CAPACITY, interface_bandwidth_gb_s=488.0,
        burst_interval_ns=0.0,
    ), **changes)


def hbf_config(**changes: Any) -> NandConfig:
    return replace(make_hbf_config(
        capacity_bytes=CAPACITY, channels=8, parallel_units=8,
    ), **changes)


def access(core: Any, operation: str, *, size: int = PAGE,
           address: int = 0, arrival_ns: float = 0.0, label: str = "access"):
    return core.submit(AccessRequest(label, operation, address, size, arrival_ns))


def check_directional_bandwidth() -> dict[str, Any]:
    fast, slow = hbf_config(), hbf_config(host_write_bandwidth_gb_s=13.6)
    observations = []
    for details, pages in ((True, 1), (True, 128), (False, 128)):
        row: dict[str, Any] = {"capture_details": details, "pages": pages}
        for operation in ("read", "write"):
            values = [access(NandCore(config, capture_details=details), operation,
                             size=pages * PAGE).completion_ns for config in (fast, slow)]
            row[operation + "_fast_ns"], row[operation + "_slow_ns"] = values
            require(equal(*values) if operation == "read" else values[1] > values[0],
                    "write-bandwidth change must affect writes and leave reads unchanged")
        observations.append(row)
    require(equal(observations[1]["write_fast_ns"], observations[2]["write_fast_ns"]),
            "detailed and accelerated NAND write timing disagree")
    return {"read_gb_s": 488.0, "write_gb_s": [27.2, 13.6], "observations": observations}


def check_legacy_configs() -> dict[str, Any]:
    old = make_ssd_config(host_bandwidth_gb_s=112.0)
    explicit = replace(old, host_read_bandwidth_gb_s=112.0, host_write_bandwidth_gb_s=112.0)
    for operation in ("read", "write"):
        require(equal(access(NandCore(old), operation).completion_ns,
                      access(NandCore(explicit), operation).completion_ns),
                "legacy shared-bandwidth SSD timing changed")
    legacy_hbf = make_hbf_config(host_bandwidth_gb_s=80.0)
    require(legacy_hbf.host_bandwidth_for("read") == legacy_hbf.host_bandwidth_for("write") == 80.0,
            "explicit legacy HBF shared bandwidth must remain authoritative")
    hbm = make_hbm_config()
    require(hbm.kind is MemoryKind.HBM and access(DramCore(hbm), "read").completion_ns > 0,
            "legacy HBM configuration is not executable")
    require(NandConfig.from_mapping({"kind": "SSD"}).write_buffer_bytes == 0,
            "old serialized NAND configs must retain write-through behavior")
    return {"ssd_shared_bandwidth_gb_s": 112.0, "explicit_hbf_shared_bandwidth_gb_s": 80.0,
            "hbm_kind": hbm.kind.value, "legacy_write_buffer_bytes": 0}


def check_standard_kv_bytes() -> dict[str, Any]:
    """Validate the hand-computable BF16 GQA KV byte baseline."""

    bytes_per_element = 2  # BF16
    layers = 80
    kv_heads = 8
    head_dim = 128
    tokens = 131_072
    bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
    total_bytes = bytes_per_token * tokens
    require(bytes_per_token == 327_680,
            "standard BF16 GQA KV bytes/token baseline changed")
    require(total_bytes == 40 * 1024**3,
            "131072-token standard KV baseline must equal 40 GiB")
    return {
        "dtype": "bf16",
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "bytes_per_token": bytes_per_token,
        "tokens": tokens,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1024**3,
    }


def check_write_buffer() -> dict[str, Any]:
    config = hbf_config(write_buffer_bytes=8192)
    results = [access_core for access_core in NandCore(config).run(
        AccessRequest(str(index), "write", index * PAGE, 4096, 0.0) for index in range(3)
    )]
    batch = summarize_batch(results)
    require(results[0].write_buffer_wait_ns == results[1].write_buffer_wait_ns == 0,
            "the first two writes must fit the finite buffer")
    require(results[2].write_buffer_wait_ns > 0, "buffer exhaustion must backpressure the third write")
    require(batch.write_buffer_peak_bytes == 8192, "write buffer peak must equal its declared capacity")
    require(batch.write_buffer_bytes == 12288, "write buffer byte accounting is incorrect")
    drained = access(NandCore(config), "write", size=4096,
                     arrival_ns=batch.last_completion_ns)
    require(drained.write_buffer_wait_ns == 0, "empty buffer must admit a later write")
    try:
        access(NandCore(config), "write", size=12288)
    except ValueError:
        pass
    else:
        raise AssertionError("an individually oversized buffered write must fail explicitly")
    return {"buffer_capacity_bytes": 8192, "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            "write_buffer_peak_bytes": batch.write_buffer_peak_bytes,
            "host_payload_bytes": batch.write_buffer_bytes, "oversized_request_rejected": True}


def check_capacity_only() -> dict[str, Any]:
    # Keep geometry and bandwidth identical; change only addressable capacity.
    durations = []
    for capacity in (8 * PAGE, 64 * PAGE):
        core = NandCore(hbf_config(capacity_bytes=capacity), capture_details=False)
        durations.append(core.run(AccessRequest(str(i), "read", i * PAGE, PAGE)
                                  for i in range(8)).duration_ns)
    require(equal(*durations), "capacity alone must not accelerate an already-fitting working set")
    # An explicit ideal-link control contains neither media latency nor capacity
    # pressure, so identical bytes and bandwidth have identical service time.
    ideal = [ResourceTimeline().transfer("ideal", 8 * PAGE, 0.0, 488.0).duration_ns
             for _capacity in (8 * PAGE, 64 * PAGE)]
    require(equal(*ideal), "ideal fixed-workload bandwidth control changed")
    small = DynamicKVPool([KvPoolComponent("hbm", 2 * PAGE, kind="hbm")], page_bytes=PAGE)
    require(small.resize("request", 2), "HBM capacity should fit two pages")
    require(not small.resize("request", 3), "HBM capacity overflow must reject allocation")
    require(len(small.pages()) == 2 and small.used_bytes_by_component["hbm"] == 2 * PAGE,
            "failed capacity admission must not leak pages")
    extended = DynamicKVPool([
        KvPoolComponent("hbm", 2 * PAGE, kind="hbm"),
        KvPoolComponent("hbf", 8 * PAGE, kind="hbf", active=True,
                        read_bandwidth_gbps=488 * 8, write_bandwidth_gbps=27.2 * 8),
    ], page_bytes=PAGE)
    require(extended.resize("request", 10), "active-memory HBF must increase admissible resident pages")
    require(not extended.resize("request", 11), "combined capacity overflow must reject allocation")
    extended.release("request")
    require(sum(extended.used_bytes_by_component.values()) == 0, "release must reclaim both tiers")
    return {"fixed_working_set_bytes": 8 * PAGE, "capacities_bytes": [8 * PAGE, 64 * PAGE],
            "memory_service_duration_ns": durations, "ideal_link_duration_ns": ideal,
            "hbm_only_max_pages": 2, "hbm_plus_active_hbf_max_pages": 10,
            "capacity_overflow_rejected": True, "release_reclaims_all_bytes": True}


def check_shared_contention() -> dict[str, Any]:
    config = hbf_config()
    isolated = access(NandCore(config), "read").completion_ns
    shared = NandCore(config)
    first = access(shared, "read", label="foreground")
    second = access(shared, "read", label="background")
    require(equal(first.completion_ns, isolated), "first shared access must match isolated timing")
    require(second.completion_ns > isolated, "same physical resources must serialize conflicting reads")
    return {"isolated_read_completion_ns": isolated,
            "shared_first_completion_ns": first.completion_ns,
            "shared_second_completion_ns": second.completion_ns,
            "contention_delay_ns": second.completion_ns - isolated}


def _migration_hardware(*, relay: bool, storage: bool = False) -> HardwareSpec:
    """Build a tiny physical graph with explicit memory services and links."""

    hbm_capacity = 4 * PAGE
    hbf_capacity = 4 * PAGE
    hbm_physical = make_hbm_config(
        capacity_bytes=hbm_capacity,
        interface_bandwidth_gb_s=488.0,
        burst_interval_ns=0.0,
    )
    hbf_physical = make_hbf_config(
        capacity_bytes=hbf_capacity,
        channels=8,
        parallel_units=8,
        write_buffer_bytes=2 * PAGE,
    )
    components = [
        ComponentSpec(
            "hbm", "hbm", capacity_bytes=hbm_capacity,
            bandwidth_gbps=488.0 * 8.0,
            metadata={"physical_memory_config": hbm_physical},
        ),
        ComponentSpec(
            "hbf", "hbf", capacity_bytes=hbf_capacity,
            bandwidth_gbps=488.0 * 8.0,
            read_bandwidth_gbps=488.0 * 8.0,
            write_bandwidth_gbps=27.2 * 8.0,
            metadata={
                "physical_memory_config": hbf_physical,
                **(
                    {
                        "access_mode": "memory",
                        "write_buffer_bytes": 2 * PAGE,
                    }
                    if not storage
                    else {"access_mode": "remote_flash"}
                ),
            },
        ),
    ]
    links = []
    if relay:
        relay_capacity = 4 * PAGE
        relay_physical = make_hbm_config(
            capacity_bytes=relay_capacity,
            interface_bandwidth_gb_s=256.0,
            burst_interval_ns=0.0,
        )
        components.insert(
            1,
            ComponentSpec(
                "relay", "hbm", capacity_bytes=relay_capacity,
                bandwidth_gbps=256.0 * 8.0,
                metadata={"physical_memory_config": relay_physical},
            ),
        )
        links.extend([
            LinkSpec("hbm-relay", "hbm", "p0", "relay", "p0", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0),
            LinkSpec("relay-hbf", "relay", "p1", "hbf", "p1", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0),
        ])
    else:
        links.append(
            LinkSpec("hbm-hbf", "hbm", "p0", "hbf", "p0", "pcie",
                     bandwidth_gbps=64.0 * 8.0, latency_ns=50.0)
        )
    return HardwareSpec("kv-migration", tuple(components), tuple(links), require_connected=False)


def _run_physical_migration(*, relay: bool, storage: bool = False) -> dict[str, Any]:
    hardware = _migration_hardware(relay=relay, storage=storage)
    executor = PhysicalKVTransferExecutor(
        hardware,
        runtime=PhysicalRuntimeContext(),
        topology=TopologyRouter(hardware),
        relay_components=("relay",) if relay else (),
    )
    pool = DynamicKVPool(
        [
            KvPoolComponent("hbm", 4 * PAGE, kind="hbm"),
            KvPoolComponent(
                "hbf", 4 * PAGE, kind="hbf", active=not storage,
                read_bandwidth_gbps=488.0 * 8.0,
                write_bandwidth_gbps=27.2 * 8.0,
            ),
        ],
        page_bytes=PAGE,
    )
    pool.set_transfer_executor(executor)
    require(pool.resize("migration", 1, compatible_components=("hbm",)),
            "physical migration source allocation failed")
    executor.sync_pages(pool.pages())
    page = pool.pages()[0]
    source_address = executor.address(page)
    migrated = (
        pool.offload_page(page.logical_page_id, "hbf")
        if storage
        else pool.migrate_page(page.logical_page_id, "hbf")
    )
    require(migrated, "physical migration failed: " + str(pool.last_error))
    event = pool.events[-1]
    physical_event = executor.events[-1]
    phase_kinds = [str(phase.get("kind")) for phase in physical_event["phases"]]
    require(page.owner_component == "hbf", "migration did not update page ownership")
    # Address spaces are owner-scoped, so both allocators may legitimately
    # return base address zero.  The ownership transition plus a valid target
    # allocation is the invariant; comparing raw integers across owners would
    # be a false failure.
    require(page.owner_component == "hbf" and executor.address(page) >= 0,
            "migration must expose a valid target physical allocation")
    require("source_read" in phase_kinds and "target_write" in phase_kinds,
            "migration must include source read and target write phases")
    require("link" in phase_kinds, "migration must reserve a real fabric link")
    if relay:
        require("relay_write" in phase_kinds and "relay_read" in phase_kinds,
                "relay migration must include relay write/read phases")
    require(executor.counters["read_bytes"] >= PAGE and
            executor.counters["write_bytes"] >= PAGE,
            "physical migration counters must include media reads and writes")
    require(any(key.startswith("link.") for key in executor.runtime.timeline.bytes_moved),
            "migration must leave shared link bytes in the runtime timeline")
    return {
        "path": "relay" if relay else "direct",
        "target_tier": "storage" if storage else "active_memory",
        "logical_page_bytes": PAGE,
        "source_address": source_address,
        "target_address": executor.address(page),
        "duration_ns": event.duration_ns,
        "phase_kinds": phase_kinds,
        "read_bytes": executor.counters["read_bytes"],
        "write_bytes": executor.counters["write_bytes"],
        "link_bytes": executor.counters["link_bytes"],
        "timeline_bytes_moved": dict(executor.runtime.timeline.bytes_moved),
    }


def check_physical_kv_migration() -> dict[str, Any]:
    direct = _run_physical_migration(relay=False)
    relay = _run_physical_migration(relay=True)
    storage = _run_physical_migration(relay=False, storage=True)
    require(relay["duration_ns"] > direct["duration_ns"],
            "relay path must include additional physical work")
    require(storage["target_tier"] == "storage",
            "storage migration must exercise the explicit offload tier")
    return {"direct": direct, "relay": relay, "storage": storage}


def placement_experiment(name: str, *, kv_tier: str, weights_tier: str,
                         cold_pages: int = 0) -> dict[str, Any]:
    """Four fixed synthetic decode-like memory steps; no compute/token model."""
    cores = {"hbm": DramCore(hbm_config(), capture_details=False),
             "hbf": NandCore(hbf_config(write_buffer_bytes=2 * PAGE), capture_details=False)}
    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    clock = 0.0
    for step in range(4):
        current = []
        # A weight scan followed by an independent KV batch. The next step
        # waits for all reads and appends, without assuming compute overlap.
        weight = access(cores[weights_tier], "read", size=8 * PAGE,
                        address=16 * PAGE, arrival_ns=clock, label=f"weights-{step}")
        by_tier[weights_tier].append(weight)
        current.append(weight)
        for page in range(8):
            tier = "hbf" if page < cold_pages else kv_tier
            result = access(cores[tier], "read", address=page * PAGE,
                            arrival_ns=clock, label=f"kv-read-{step}-{page}")
            by_tier[tier].append(result)
            current.append(result)
        for page in (6, 7):
            result = access(cores[kv_tier], "write", address=page * PAGE,
                            arrival_ns=clock, label=f"kv-append-{step}-{page}")
            by_tier[kv_tier].append(result)
            current.append(result)
        clock = max(result.completion_ns for result in current)
    logical_bytes = sum(result.logical_bytes for values in by_tier.values() for result in values)
    tiers = {}
    for tier, values in by_tier.items():
        batch = summarize_batch(values)
        tiers[tier] = {"logical_read_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.READ),
                       "logical_write_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.WRITE),
                       "physical_read_bytes": batch.physical_read_bytes,
                       "physical_write_bytes": batch.physical_write_bytes,
                       "queue_wait_ns": batch.queue_wait_ns,
                       "write_buffer_wait_ns": batch.write_buffer_wait_ns,
                       "write_buffer_peak_bytes": batch.write_buffer_peak_bytes}
    return {"id": name, "kind": "fixed_traffic_memory_microbenchmark", "steps": 4,
            "kv_pages": 8, "cold_pages_on_hbf": cold_pages,
            "weights_tier": weights_tier, "kv_default_tier": kv_tier,
            "memory_service_duration_ns": clock, "logical_bytes": logical_bytes,
            "useful_memory_bandwidth_gb_s": logical_bytes / clock,
            "tiers": tiers, "migration_count": 0,
            "placement_note": "Static placement; dynamic migration is tested separately."}


def run_acceptance() -> dict[str, Any]:
    checks = []
    tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("directional_bandwidth", check_directional_bandwidth),
        ("legacy_configs", check_legacy_configs),
        ("standard_kv_bytes", check_standard_kv_bytes),
        ("finite_write_buffer", check_write_buffer),
        ("capacity_only_and_lifecycle", check_capacity_only),
        ("shared_physical_contention", check_shared_contention),
        ("physical_kv_migration", check_physical_kv_migration),
        ("serving_paged_swap", check_serving_paged_swap),
        ("slo_semantics", check_slo_semantics),
    ]
    for name, function in tasks:
        try:
            checks.append({"id": name, "passed": True, "metrics": function()})
        except Exception as error:
            checks.append({"id": name, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})
    experiments = []
    for name, options in (
        ("A_hbm_only", {"kv_tier": "hbm", "weights_tier": "hbm"}),
        ("B_all_kv_hbf", {"kv_tier": "hbf", "weights_tier": "hbm"}),
        ("C_hot_hbm_cold_hbf", {"kv_tier": "hbm", "weights_tier": "hbm", "cold_pages": 4}),
        ("D_weights_hbf_kv_hbm", {"kv_tier": "hbm", "weights_tier": "hbf"}),
    ):
        try:
            experiments.append({"passed": True, **placement_experiment(name, **options)})
        except Exception as error:
            experiments.append({"id": name, "passed": False,
                                "error": f"{type(error).__name__}: {error}"})
    passed = {item["id"]: item["passed"] for item in checks}
    capabilities = [
        {"question": "Does the declared HBF write bandwidth change physical write service time?",
         "supported": passed["directional_bandwidth"], "evidence": ["directional_bandwidth"]},
        {"question": "When does a finite HBF write buffer introduce backpressure?",
         "supported": passed["finite_write_buffer"], "evidence": ["finite_write_buffer"]},
        {"question": "Does the standard BF16 GQA KV byte baseline match the hand calculation?",
         "supported": passed["standard_kv_bytes"], "evidence": ["standard_kv_bytes"]},
        {"question": "Does HBF expand capacity without automatically speeding up a fixed fitting workload?",
         "supported": passed["capacity_only_and_lifecycle"], "evidence": ["capacity_only_and_lifecycle"]},
        {"question": "How does static KV/weight placement change fixed memory-traffic service time?",
         "supported": all(item["passed"] for item in experiments), "evidence": [item["id"] for item in experiments]},
        {"question": "Does a KV migration reserve source read, fabric transfer and target write on shared physical state?",
         "supported": passed["physical_kv_migration"], "evidence": ["physical_kv_migration"]},
        {"question": "Can serving automatically preempt, swap a multi-page KV working set to HBF, resume it, and release it?",
         "supported": passed["serving_paged_swap"], "evidence": ["serving_paged_swap"]},
        {"question": "Do SLO thresholds classify a fixed serving trace without changing its execution timeline?",
         "supported": passed["slo_semantics"], "evidence": ["slo_semantics"]},
        {"question": "Can a fixed KV prefetch distance proactively hide restore latency?",
         "supported": False,
         "reason": "V4 keeps prefetch_distance=0; proactive KV lookahead events are not modeled."},
        {"question": "What is the real LLM token/s improvement, TTFT or TPOT on a commercial HBF device?",
         "supported": False, "reason": "The serving check uses a tiny synthetic scheduler but has no full model compute or commercial-HBF calibration."},
    ]
    return {"schema_version": 2, "passed": all(item["passed"] for item in checks + experiments),
            "scope": "single-device synthetic HBM/HBF physical and serving acceptance",
            "assumptions": {"page_bytes": PAGE, "hbm_interface_gb_s": 488.0,
                            "hbf_host_read_gb_s": 488.0, "hbf_host_write_gb_s": 27.2,
                            "hbf_read_ns": 4000.0, "hbf_program_ns": 75000.0,
                            "hbf_channels_and_parallel_units": 8,
                            "calibrated_to_silicon": False},
            "checks": checks, "experiments": experiments, "capabilities": capabilities,
            "limitations": ["Serving coverage uses a tiny synthetic scheduler workload; it does not claim calibrated LLM token throughput.",
                            "V4 prefetch_distance remains zero-only; proactive KV lookahead is an explicit follow-up model.",
                            "No FTL/GC, wear, ECC, energy, packaging budget or multi-GPU model."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this file.")
    arguments = parser.parse_args()
    report = run_acceptance()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
