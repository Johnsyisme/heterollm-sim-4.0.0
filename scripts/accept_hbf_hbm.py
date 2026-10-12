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
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec
from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_transfer import ResourceTimeline, summarize_batch
from heterollm_sim.memory_types import (
    AccessRequest, DramConfig, MemoryKind, NandConfig, Operation,
    make_hbf_config, make_hbm_config, make_ssd_config,
)
from heterollm_sim.nand_core import NandCore

PAGE = 16_384
CAPACITY = 32 * 1024 * 1024


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
        ("finite_write_buffer", check_write_buffer),
        ("capacity_only_and_lifecycle", check_capacity_only),
        ("shared_physical_contention", check_shared_contention),
        ("physical_kv_migration", check_physical_kv_migration),
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
        {"question": "Does HBF expand capacity without automatically speeding up a fixed fitting workload?",
         "supported": passed["capacity_only_and_lifecycle"], "evidence": ["capacity_only_and_lifecycle"]},
        {"question": "How does static KV/weight placement change fixed memory-traffic service time?",
         "supported": all(item["passed"] for item in experiments), "evidence": [item["id"] for item in experiments]},
        {"question": "Does a KV migration reserve source read, fabric transfer and target write on shared physical state?",
         "supported": passed["physical_kv_migration"], "evidence": ["physical_kv_migration"]},
        {"question": "What is the real LLM token/s improvement, TTFT or TPOT on a commercial HBF device?",
         "supported": False, "reason": "This suite has no model compute or serving scheduler and uses uncalibrated hardware assumptions."},
    ]
    return {"schema_version": 2, "passed": all(item["passed"] for item in checks + experiments),
            "scope": "single-device synthetic physical-memory microbenchmarks",
            "assumptions": {"page_bytes": PAGE, "hbm_interface_gb_s": 488.0,
                            "hbf_host_read_gb_s": 488.0, "hbf_host_write_gb_s": 27.2,
                            "hbf_read_ns": 4000.0, "hbf_program_ns": 75000.0,
                            "hbf_channels_and_parallel_units": 8,
                            "calibrated_to_silicon": False},
            "checks": checks, "experiments": experiments, "capabilities": capabilities,
            "limitations": ["No LLM compute, token throughput or continuous-batching claims.",
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
