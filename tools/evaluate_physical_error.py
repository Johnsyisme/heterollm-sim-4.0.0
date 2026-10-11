"""Small independent physical-path error audit.

This report separates software reference equivalence from any claim about
silicon accuracy.  It compares the accelerated DRAM/NAND paths with the
existing detailed loops, then checks byte/resource conservation, queue gating,
finite-buffer metadata, and capacity rejection.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from measure_memory_precision import (  # noqa: E402
    compare_case,
    generic_dram,
    nand_config,
)
from heterollm_sim.communication import TransferPhase, plan_transfer_pipeline  # noqa: E402
from heterollm_sim.contracts import ResourceDemand  # noqa: E402
from heterollm_sim.dram_core import DramCore  # noqa: E402
from heterollm_sim.memory_transfer import ResourceTimeline  # noqa: E402
from heterollm_sim.memory_types import AccessRequest, Operation  # noqa: E402
from heterollm_sim.nand_core import NandCore  # noqa: E402


OUTPUT = ROOT / "docs" / "simulation_error_2026-10-11" / "physical_error.json"


def _equivalence_cases() -> list[dict]:
    dram = generic_dram(metadata={"shared_command_resource": "dram:command:shared"})
    dreq = [
        AccessRequest("warm", Operation.READ, 193, 64, 0.0),
        AccessRequest("large-read", Operation.READ, 193, 9217 * 64 + 7, 0.0),
        AccessRequest("mixed-write", Operation.WRITE, 193, 9217 * 64 + 7, 0.0),
        AccessRequest("followup", Operation.READ, 193, 9217 * 64 + 7, 1_000_000.0),
    ]
    nand = nand_config(metadata={"host_resource_id": "shared:host"})
    nreq = [
        AccessRequest("read", Operation.READ, 17, 5000 * 64 - 19, 0.0),
        AccessRequest("write", Operation.WRITE, 17, 5000 * 64 - 19, 0.0),
        AccessRequest("followup", Operation.READ, 17, 5000 * 64 - 19, 10_000_000.0),
    ]
    # Long serial accumulation is kept as a separate stress case because its
    # tiny timestamp drift is a floating-point observation, not a counter
    # mismatch or a hardware precision estimate.
    roundoff = nand_config(
        page_bytes=4096,
        host_granularity_bytes=4096,
        pages_per_block=256,
        blocks_per_plane=512,
        channels=4,
        host_bandwidth_gb_s=3.7,
        internal_bandwidth_gb_s=11.3,
        page_read_ns=50000.2,
        page_program_ns=200000.1,
        front_ns=7.1,
    )
    roundoff_requests = [
        AccessRequest("100000-page-write", Operation.WRITE, 0, 100000 * 4096, 0.0),
        AccessRequest("100000-page-read-followup", Operation.READ, 0, 100000 * 4096, 1e12),
    ]
    return [
        compare_case("dram_shared_command_mixed_followup", "dram", dram, dreq),
        compare_case("nand_shared_host_partial_rmw", "nand", nand, nreq),
        compare_case("nand_100000_page_write_roundoff", "nand", roundoff, roundoff_requests),
    ]


def _error_summary(row: dict) -> dict:
    return {
        "case": row["case"],
        "memory_type": row["memory_type"],
        "request_count": row["request_count"],
        "max_completion_absolute_error_ns": row["max_completion_error"]["absolute_error"],
        "max_completion_relative_error": row["max_completion_error"]["relative_error"],
        "max_duration_absolute_error_ns": row["max_duration_error"]["absolute_error"],
        "timeline_max_absolute_error": row["timeline_error"]["max_absolute_error"],
        "core_state_max_absolute_error": row["core_state_error"]["max_absolute_error"],
        "integer_counters_pass": row["integer_counters_pass"],
        "discrete_state_pass": row["discrete_state_pass"],
        "serial_numeric_vs_python_max_completion_absolute_error_ns": row["serial_numeric_vs_python_max_completion_absolute_error_ns"],
        "serial_numeric_vs_python_integer_counters_pass": row["serial_numeric_vs_python_integer_counters_pass"],
    }


def _conservation_checks() -> dict:
    # Timeline-level conservation is independent of DRAM/NAND path selection.
    timeline = ResourceTimeline()
    first = timeline.transfer("wire", 100, 0.0, 10.0, direction="read")
    second = timeline.transfer("wire", 40, 0.0, 10.0, direction="write", switch_ns=3.0)
    moved = timeline.bytes_moved.get("wire", 0)
    expected_busy = 100 / 10.0 + 40 / 10.0
    timeline_check = {
        "passed": moved == 140 and abs(timeline.busy_ns["wire"] - expected_busy) < 1e-12,
        "bytes_moved": moved,
        "expected_bytes": 140,
        "busy_ns": timeline.busy_ns["wire"],
        "expected_busy_ns": expected_busy,
        "direction_switch_start_ns": second.start_ns,
        "first_end_ns": first.end_ns,
        "switch_ns": 3.0,
    }

    dram_cfg = replace(generic_dram(), max_outstanding_requests=1)
    dram = DramCore(dram_cfg, capture_details=False)
    dr = dram.submit(AccessRequest("d", Operation.READ, 0, 5000 * 64, 0.0))
    dr2 = dram.submit(AccessRequest("d2", Operation.READ, 0, 5000 * 64, 0.0))
    dram_data = sum(value for rid, value in dram.timeline.bytes_moved.items() if rid.startswith("dram:data:"))
    dram_queue = {
        "passed": dr2.counters["queue_wait_ns"] > 0 and dram_data == dr.transfer_bytes + dr2.transfer_bytes,
        "first_queue_wait_ns": dr.counters["queue_wait_ns"],
        "second_queue_wait_ns": dr2.counters["queue_wait_ns"],
        "timeline_data_bytes": dram_data,
        "result_transfer_bytes": dr.transfer_bytes + dr2.transfer_bytes,
    }

    nand_cfg = replace(nand_config(), max_outstanding_requests=1)
    nand = NandCore(nand_cfg, capture_details=False)
    nr = nand.submit(AccessRequest("n", Operation.READ, 0, 5000 * 64, 0.0))
    nr2 = nand.submit(AccessRequest("n2", Operation.READ, 0, 5000 * 64, 0.0))
    host_bytes = sum(value for rid, value in nand.timeline.bytes_moved.items() if rid == "nand:host")
    channel_bytes = sum(value for rid, value in nand.timeline.bytes_moved.items() if rid.startswith("nand:channel:"))
    nand_queue = {
        "passed": nr2.counters["queue_wait_ns"] > 0 and host_bytes == nr.counters["host_transfer_bytes"] + nr2.counters["host_transfer_bytes"] and channel_bytes == nr.counters["internal_transfer_bytes"] + nr2.counters["internal_transfer_bytes"],
        "first_queue_wait_ns": nr.counters["queue_wait_ns"],
        "second_queue_wait_ns": nr2.counters["queue_wait_ns"],
        "timeline_host_bytes": host_bytes,
        "result_host_bytes": nr.counters["host_transfer_bytes"] + nr2.counters["host_transfer_bytes"],
        "timeline_channel_bytes": channel_bytes,
        "result_internal_bytes": nr.counters["internal_transfer_bytes"] + nr2.counters["internal_transfer_bytes"],
    }
    return {"resource_timeline": timeline_check, "dram_queue_and_bytes": dram_queue, "nand_queue_and_bytes": nand_queue}


def _pipeline_capacity_check() -> dict:
    demand = lambda resource, service, size: ResourceDemand(resource, service, bytes_moved=size)
    phases = tuple(
        (TransferPhase("send", (demand("link", 10.0, 64),), {"event_kind": "send"}),)
        for _ in range(3)
    )
    plan = plan_transfer_pipeline(
        phases,
        chunk_byte_counts=(64, 64, 64),
        buffer_capacity_bytes=128,
        max_inflight_chunks=2,
    )
    first_tasks = [task for task in plan.tasks if task.metadata.get("chunk_index") == 0]
    credit_links = [task.metadata["credit_dependency_ids"] for task in plan.tasks if task.metadata.get("chunk_index") == 2 and task.metadata.get("credit_dependency_ids")]
    resource_bytes = {}
    for task in plan.tasks:
        for demand in task.demands:
            resource_bytes[demand.resource_id] = resource_bytes.get(demand.resource_id, 0) + demand.bytes_moved
    expected_resource_bytes = {"link": 3 * 64}
    passed = (
        all(task.metadata["chunk_bytes"] <= task.metadata["buffer_capacity_bytes"] for task in plan.tasks)
        and bool(credit_links)
        and resource_bytes == expected_resource_bytes
    )
    return {
        "passed": passed,
        "chunk_count": len(plan.chunk_byte_counts),
        "buffer_capacity_bytes": plan.buffer_capacity_bytes,
        "max_inflight_chunks": plan.max_inflight_chunks,
        "serialized_upper_bound_ns": plan.serialized_upper_bound_ns,
        "ideal_lower_bound_ns": plan.ideal_lower_bound_ns,
        "chunk0_task_count": len(first_tasks),
        "chunk2_credit_dependencies": credit_links,
        "resource_bytes": resource_bytes,
        "expected_resource_bytes": expected_resource_bytes,
        "scope": "planner metadata/dependency conservation; no packet/flit hardware calibration",
    }


def _capacity_checks() -> dict:
    dram = replace(generic_dram(data_lanes=1, banks_per_group=1, bank_groups_per_rank=1, rows_per_bank=2, row_bytes=64), capacity_bytes=128)
    nand = replace(nand_config(channels=1, planes_per_lun=1, blocks_per_plane=1, pages_per_block=2, page_bytes=64, host_granularity_bytes=64), capacity_bytes=128)
    checks = []
    for name, core_cls, cfg in (("dram", DramCore, dram), ("nand", NandCore, nand)):
        core_cls(cfg, capture_details=False).submit(AccessRequest(name + "-inside", Operation.READ, 0, cfg.effective_capacity_bytes))
        try:
            # Start inside the final unit but request one complete unit past
            # the boundary; this exercises request validation rather than
            # constructor geometry validation.
            core_cls(cfg, capture_details=False).submit(AccessRequest(name + "-overflow", Operation.READ, cfg.effective_capacity_bytes - (cfg.burst_bytes // 2 if name == "dram" else cfg.page_bytes), cfg.burst_bytes if name == "dram" else cfg.page_bytes * 2))
        except ValueError as exc:
            checks.append({"memory_type": name, "capacity_bytes": cfg.effective_capacity_bytes, "in_bounds_pass": True, "overflow_rejected": True, "error": str(exc)})
        else:
            checks.append({"memory_type": name, "capacity_bytes": cfg.effective_capacity_bytes, "in_bounds_pass": True, "overflow_rejected": False})
    return {"passed": all(item["in_bounds_pass"] and item["overflow_rejected"] for item in checks), "checks": checks}


def run() -> dict:
    equivalence = _equivalence_cases()
    report = {
        "title": "Physical memory simulation error micro-evaluation",
        "date": "2026-10-11",
        "purpose": "Measure software path equivalence and conservation invariants for representative DRAM/NAND paths.",
        "scope_limit": "解析正确性仅表示当前快速路径与独立详细参考循环的一致性；不等同于硬件精度或真实芯片校准。",
        "method": {
            "dram_reference": "DramCore subclass disabling acceleration, plus serial numeric reference in precision_dram_reference.py.",
            "nand_reference": "NandCore subclass executing the detailed page loop.",
            "hardware_precision": "not measured; no silicon trace or vendor timing calibration is used here.",
        },
        "software_equivalence": [_error_summary(row) for row in equivalence],
        "conservation_and_queue": _conservation_checks(),
        "finite_buffer_pipeline": _pipeline_capacity_check(),
        "capacity_boundary": _capacity_checks(),
    }
    report["pass_summary"] = {
        "software_equivalence_pass": all(item["integer_counters_pass"] and item["discrete_state_pass"] for item in report["software_equivalence"]),
        "conservation_and_queue_pass": all(item["passed"] for item in report["conservation_and_queue"].values()),
        "finite_buffer_pipeline_pass": report["finite_buffer_pipeline"]["passed"],
        "capacity_boundary_pass": report["capacity_boundary"]["passed"],
    }
    report["pass_summary"]["all_checks_pass"] = all(report["pass_summary"].values())
    return report


def main() -> None:
    started = time.monotonic()
    report = run()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "elapsed_seconds": report["elapsed_seconds"], "pass_summary": report["pass_summary"]}, ensure_ascii=False), flush=True)
    if not report["pass_summary"]["all_checks_pass"]:
        raise SystemExit("physical error micro-evaluation failed; see report")


if __name__ == "__main__":
    main()
