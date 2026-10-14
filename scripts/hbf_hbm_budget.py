"""Fixed-package-slot HBM/HBF capacity and bandwidth-ratio sweep.

This is a deliberately small analytical acceptance model.  Four hypothetical
package slots are available, and each selected HBM/HBF slot contributes its
own capacity, directional bandwidth, parallel-unit count, and write-buffer
budget.  The slot assumptions are not a commercial package calibration.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_transfer import summarize_batch
from heterollm_sim.memory_types import AccessRequest, Operation, make_hbm_config, make_hbf_config
from heterollm_sim.nand_core import NandCore

PAGE = 16_384
TOTAL_SLOTS = 4
WEIGHT_PAGES = 4
KV_PAGES = 8
WORKING_SET_PAGES = WEIGHT_PAGES + KV_PAGES


class SlotProfile:
    """One assumed package slot, expressed in the simulator's units."""

    __slots__ = (
        "tier", "capacity_pages", "read_bandwidth_gb_s",
        "write_bandwidth_gb_s", "parallel_units", "write_buffer_bytes",
    )

    def __init__(
        self, tier: str, capacity_pages: int, read_bandwidth_gb_s: float,
        write_bandwidth_gb_s: float, parallel_units: int,
        write_buffer_bytes: int,
    ) -> None:
        self.tier = tier
        self.capacity_pages = capacity_pages
        self.read_bandwidth_gb_s = read_bandwidth_gb_s
        self.write_bandwidth_gb_s = write_bandwidth_gb_s
        self.parallel_units = parallel_units
        self.write_buffer_bytes = write_buffer_bytes

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_pages * PAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "capacity_pages": self.capacity_pages,
            "capacity_bytes": self.capacity_bytes,
            "read_bandwidth_gb_s": self.read_bandwidth_gb_s,
            "write_bandwidth_gb_s": self.write_bandwidth_gb_s,
            "parallel_units": self.parallel_units,
            "write_buffer_bytes": self.write_buffer_bytes,
        }


# The profiles are intentionally explicit.  Changing a slot count changes
# aggregate capacity, aggregate directional bandwidth, parallelism, and (for
# HBF) finite buffering together; no ratio row silently borrows the old total.
HBM_SLOT = SlotProfile(
    tier="hbm", capacity_pages=4, read_bandwidth_gb_s=256.0,
    write_bandwidth_gb_s=256.0, parallel_units=4, write_buffer_bytes=0,
)
HBF_SLOT = SlotProfile(
    tier="hbf", capacity_pages=2, read_bandwidth_gb_s=160.0,
    write_bandwidth_gb_s=20.0, parallel_units=2, write_buffer_bytes=PAGE,
)

# Normalized architecture points retain the flash capacity advantage.  These
# page counts are deliberately small, not a scaling law for real package area.
ARCHITECTURE_HBM_SLOT = SlotProfile(
    tier="hbm", capacity_pages=4, read_bandwidth_gb_s=256.0,
    write_bandwidth_gb_s=256.0, parallel_units=4, write_buffer_bytes=0,
)
ARCHITECTURE_HBF_SLOT = SlotProfile(
    tier="hbf", capacity_pages=32, read_bandwidth_gb_s=160.0,
    write_bandwidth_gb_s=20.0, parallel_units=2, write_buffer_bytes=PAGE,
)


def _core(profile: SlotProfile):
    if profile.tier == "hbm":
        return DramCore(
            make_hbm_config(
                capacity_bytes=profile.capacity_bytes,
                channels=profile.parallel_units,
                pseudo_channels_per_channel=1,
                interface_bandwidth_gb_s=profile.read_bandwidth_gb_s,
                read_bandwidth_gb_s=profile.read_bandwidth_gb_s,
                write_bandwidth_gb_s=profile.write_bandwidth_gb_s,
                burst_interval_ns=0.0,
            ),
            capture_details=False,
        )
    return NandCore(
        make_hbf_config(
            capacity_bytes=profile.capacity_bytes,
            channels=profile.parallel_units,
            parallel_units=profile.parallel_units,
            host_bandwidth_gb_s=profile.read_bandwidth_gb_s,
            host_read_bandwidth_gb_s=profile.read_bandwidth_gb_s,
            host_write_bandwidth_gb_s=profile.write_bandwidth_gb_s,
            write_buffer_bytes=profile.write_buffer_bytes,
        ),
        capture_details=False,
    )


def _access(core: Any, operation: str, address: int, arrival_ns: float, label: str):
    return core.submit(AccessRequest(label, operation, address, PAGE, arrival_ns))


def budget_row(*, hbm_slots: int) -> dict[str, Any]:
    """Execute one ratio while preserving the same 12-page traffic trace."""
    require(1 <= hbm_slots < TOTAL_SLOTS, "hbm_slots must be in [1, 3]")
    return _budget_row(
        hbm_slots=hbm_slots, hbm_slot=HBM_SLOT, hbf_slot=HBF_SLOT,
        weight_pages=WEIGHT_PAGES, kv_pages=KV_PAGES, steps=4,
        stripe_within_tier=False,
    )


def _budget_row(*, hbm_slots: int, hbm_slot: SlotProfile, hbf_slot: SlotProfile,
                weight_pages: int, kv_pages: int, steps: int,
                stripe_within_tier: bool) -> dict[str, Any]:
    """Price fixed logical traffic against independent physical package slots."""
    require(isinstance(hbm_slots, int) and 0 <= hbm_slots <= TOTAL_SLOTS,
            "hbm_slots must be an integer in [0, 4]")
    require(weight_pages > 0 and kv_pages >= 2 and steps > 0,
            "weight_pages/steps must be positive and kv_pages must be at least 2")
    require(hbm_slot.tier == "hbm" and hbf_slot.tier == "hbf", "slot tier mismatch")
    hbf_slots = TOTAL_SLOTS - hbm_slots
    working_set_pages = weight_pages + kv_pages
    profiles = [hbm_slot] * hbm_slots + [hbf_slot] * hbf_slots
    capacity_pages = sum(profile.capacity_pages for profile in profiles)
    aggregate = {
        "capacity_pages": capacity_pages,
        "capacity_bytes": sum(profile.capacity_bytes for profile in profiles),
        "read_bandwidth_gb_s": sum(profile.read_bandwidth_gb_s for profile in profiles),
        "write_bandwidth_gb_s": sum(profile.write_bandwidth_gb_s for profile in profiles),
        "parallel_units": sum(profile.parallel_units for profile in profiles),
        "write_buffer_bytes": sum(profile.write_buffer_bytes for profile in profiles),
    }
    row: dict[str, Any] = {
        "hbm_slots": hbm_slots,
        "hbf_slots": hbf_slots,
        "total_slots": TOTAL_SLOTS,
        "working_set_pages": working_set_pages,
        "workload": {"weight_pages": weight_pages, "kv_pages": kv_pages,
                      "steps": steps, "bytes_per_access": PAGE,
                      "kv_write_pages_per_step": 2,
                      "growth": "fixed resident set; the last two KV pages are updated each step"},
        "slots": [profile.to_dict() for profile in profiles],
        "aggregate_budget": aggregate,
        "assumption": "hypothetical four-slot package; not commercial calibration",
        "scope": (
            "normalized fixed-slot architecture sweep with explicit per-slot capacity and bandwidth"
            if stripe_within_tier else
            "toy fixed-slot accounting regression; HBF deliberately has less capacity per slot to exercise a capacity failure"
        ),
        "placement_policy": (
            "weight pages then KV pages; HBM-first residency, balanced striping across slots of the selected tier"
            if stripe_within_tier else "greedy HBM-first packing of weight pages followed by KV pages"
        ),
        "node_tps": None,
        "per_gpu_tps": None,
        "ttft_ns": None,
        "tpot_ns": None,
        "metric_scope": "physical memory service only; no token-serving scheduler",
    }
    if capacity_pages < working_set_pages:
        row.update({
            "runnable": False,
            "reason": (
                f"capacity {capacity_pages} pages is below fixed working set "
                f"{working_set_pages} pages"
            ),
            "logical_bytes": None,
            "memory_service_duration_ns": None,
            "useful_memory_bandwidth_gb_s": None,
        })
        return row

    cores = [(_core(profile), profile.tier, index) for index, profile in enumerate(profiles)]
    # Pack pages into slots in declaration order.  A page's address is local to
    # its slot, so every physical core sees only in-range requests.
    assignment: list[tuple[int, int]] = []
    remaining = [profile.capacity_pages for profile in profiles]
    addresses = [0] * len(profiles)
    for _page in range(working_set_pages):
        try:
            available_slots = [index for index, available in enumerate(remaining) if available]
            slot = next(iter(available_slots))
            if stripe_within_tier:
                selected_tier = profiles[slot].tier
                slot = min(
                    (index for index in available_slots if profiles[index].tier == selected_tier),
                    key=lambda index: (addresses[index], index),
                )
        except StopIteration:
            raise AssertionError("capacity preflight disagrees with page placement")
        remaining[slot] -= 1
        assignment.append((slot, addresses[slot] * PAGE))
        addresses[slot] += 1

    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    by_slot: dict[str, list[Any]] = {str(index): [] for index in range(len(cores))}
    clock = 0.0
    step_durations = []
    for step in range(steps):
        current = []
        read_completions = {}
        for index, (slot, address) in enumerate(assignment):
            core, tier, _slot_index = cores[slot]
            result = _access(core, "read", address, clock, f"budget-weight-kv-{step}-{index}")
            by_tier[tier].append(result)
            by_slot[str(slot)].append(result)
            current.append(result)
            read_completions[index] = result.completion_ns
        for page in (working_set_pages - 2, working_set_pages - 1):
            slot, address = assignment[page]
            core, tier, _slot_index = cores[slot]
            write_arrival = read_completions[page] if stripe_within_tier else clock
            result = _access(core, "write", address, write_arrival, f"budget-kv-append-{step}-{page}")
            by_tier[tier].append(result)
            by_slot[str(slot)].append(result)
            current.append(result)
        completed = max(result.completion_ns for result in current)
        step_durations.append(completed - clock)
        clock = completed

    logical_bytes = sum(result.logical_bytes for values in by_tier.values() for result in values)
    tier_metrics: dict[str, dict[str, Any]] = {}
    for tier, values in by_tier.items():
        batch = summarize_batch(values)
        tier_metrics[tier] = {
            "logical_read_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.READ),
            "logical_write_bytes": sum(x.logical_bytes for x in values if x.operation is Operation.WRITE),
            "physical_read_bytes": batch.physical_read_bytes,
            "physical_write_bytes": batch.physical_write_bytes,
            "queue_wait_ns": batch.queue_wait_ns,
            "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            "write_buffer_peak_bytes": batch.write_buffer_peak_bytes,
        }
    row.update({
        "runnable": True,
        "reason": None,
        "logical_bytes": logical_bytes,
        "memory_service_duration_ns": clock,
        "step_service_ns": step_durations,
        "useful_memory_bandwidth_gb_s": logical_bytes / clock,
        "tiers": tier_metrics,
        "slot_traffic": {
            slot: {
                "tier": profiles[int(slot)].tier,
                "resident_pages": addresses[int(slot)],
                "capacity_pages": profiles[int(slot)].capacity_pages,
                "resident_weight_pages": sum(1 for index, (assigned, _) in enumerate(assignment)
                                              if assigned == int(slot) and index < weight_pages),
                "resident_kv_pages": sum(1 for index, (assigned, _) in enumerate(assignment)
                                          if assigned == int(slot) and index >= weight_pages),
                "logical_bytes": sum(item.logical_bytes for item in values),
                "completion_ns": max((item.completion_ns for item in values), default=0.0),
                "resource_busy_ns": dict(cores[int(slot)][0].timeline.busy_ns),
                "resource_bytes": dict(cores[int(slot)][0].timeline.bytes_moved),
            }
            for slot, values in by_slot.items()
        },
    })
    return row


def run_budget_sweep() -> dict[str, Any]:
    """Return all three four-slot HBM/HBF budget points and checks."""
    rows = [budget_row(hbm_slots=value) for value in (1, 2, 3)]
    runnable = [row for row in rows if row["runnable"]]
    require(rows[0]["runnable"] is False and rows[0]["reason"],
            "the 1-HBM/3-HBF budget must expose its capacity failure")
    require(all(row["runnable"] for row in rows[1:]),
            "the 2/2 and 3/1 budgets must fit the fixed working set")
    expected_logical_bytes = (WORKING_SET_PAGES + 2) * 4 * PAGE
    require({row["logical_bytes"] for row in runnable} == {expected_logical_bytes},
            "runnable budget points must carry the same fixed traffic")
    require(runnable[-1]["memory_service_duration_ns"] < runnable[0]["memory_service_duration_ns"],
            "adding HBM slots must reduce service time for this workload")
    return {
        "schema_version": 1,
        "total_slots": TOTAL_SLOTS,
        "fixed_working_set_pages": WORKING_SET_PAGES,
        "calibrated_to_silicon": False,
        "can_infer_optimal_hbm_hbf_ratio": False,
        "limitations": [
            "Toy slot capacities intentionally invert the real HBF density advantage; these results verify accounting only.",
            "No GPU compute, cost or area model; slots are assumed interchangeable and cannot establish a real package optimum.",
            "The 3-HBM/1-HBF point fits the entire trace in HBM, leaving its HBF slot idle.",
        ],
        "rows": rows,
        "runnable_count": len(runnable),
        "capacity_failure_count": len(rows) - len(runnable),
    }


def run_architecture_budget_sweep(
    *, hbm_slot: SlotProfile = ARCHITECTURE_HBM_SLOT,
    hbf_slot: SlotProfile = ARCHITECTURE_HBF_SLOT,
    weight_pages: int = 4, kv_pages: int = 36, steps: int = 4,
) -> dict[str, Any]:
    """Scan all four-slot mixes with fixed demand and normalized HBF density.

    Slot counts are coupled to capacity, directional bandwidth, service lanes
    and finite write buffering.  The workload is a fixed-resident-state memory
    trace: it is deliberately not a full GPU or growing-context simulation.
    """
    require(hbm_slot.capacity_pages > 0 and hbf_slot.capacity_pages > 0,
            "both slot capacities must be positive")
    rows = [
        _budget_row(hbm_slots=count, hbm_slot=hbm_slot, hbf_slot=hbf_slot,
                    weight_pages=weight_pages, kv_pages=kv_pages, steps=steps,
                    stripe_within_tier=True)
        for count in range(TOTAL_SLOTS + 1)
    ]
    expected_bytes = (weight_pages + kv_pages + 2) * steps * PAGE
    for row in rows:
        row["hbm_slot_fraction"] = row["hbm_slots"] / TOTAL_SLOTS
        row["hbf_to_hbm_slot_capacity_ratio"] = hbf_slot.capacity_pages / hbm_slot.capacity_pages
    runnable = [row for row in rows if row["runnable"]]
    checks = {
        "all_five_slot_mixes_are_executed": [row["hbm_slots"] for row in rows] == [0, 1, 2, 3, 4],
        "capacity_controls_runnability": all(
            row["runnable"] == (row["aggregate_budget"]["capacity_pages"] >= weight_pages + kv_pages)
            for row in rows
        ),
        "infeasible_points_have_reasons_and_no_throughput": all(
            row["reason"] and row["memory_service_duration_ns"] is None
            and row["useful_memory_bandwidth_gb_s"] is None and row["node_tps"] is None
            for row in rows if not row["runnable"]
        ),
        "fixed_logical_work_on_all_runnable_points": bool(runnable) and all(
            row["logical_bytes"] == expected_bytes for row in runnable
        ),
        "slot_residency_respects_capacity": all(
            0 <= slot["resident_pages"] <= slot["capacity_pages"]
            for row in runnable for slot in row["slot_traffic"].values()
        ),
        "resident_weight_and_kv_conservation": all(
            sum(slot["resident_weight_pages"] for slot in row["slot_traffic"].values()) == weight_pages
            and sum(slot["resident_kv_pages"] for slot in row["slot_traffic"].values()) == kv_pages
            for row in runnable
        ),
        "coupled_capacity_bandwidth_parallelism_and_buffer": all(
            row["aggregate_budget"][key] == sum(slot[key] for slot in row["slots"])
            for row in rows for key in (
                "capacity_bytes", "read_bandwidth_gb_s", "write_bandwidth_gb_s",
                "parallel_units", "write_buffer_bytes",
            )
        ),
        "physical_cores_price_every_runnable_point": all(
            row["memory_service_duration_ns"] > 0
            and len(row["step_service_ns"]) == steps
            and abs(sum(row["step_service_ns"]) - row["memory_service_duration_ns"]) < 1e-6
            and sum(slot["logical_bytes"] for slot in row["slot_traffic"].values()) == expected_bytes
            for row in runnable
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "normalized four-slot physical-memory architecture sweep",
        "total_slots": TOTAL_SLOTS,
        "slot_profiles": {"hbm": hbm_slot.to_dict(), "hbf": hbf_slot.to_dict()},
        "fixed_working_set_pages": weight_pages + kv_pages,
        "fixed_logical_bytes": expected_bytes,
        "calibrated_to_silicon": False,
        "can_infer_commercial_optimal_hbm_hbf_ratio": False,
        "rows": rows,
        "checks": checks,
        "runnable_count": len(runnable),
        "capacity_failure_count": len(rows) - len(runnable),
        "limitations": [
            "Default capacities are normalized pages: four per HBM slot and 32 per HBF slot (8x). They are hypothetical, not commercial package capacities or an area scaling law.",
            "Every row has four assumed interchangeable slots. Capacity, peak directional bandwidth, parallel service lanes and HBF buffering change together with the selected mix.",
            "HBM-first placement is striped across slots of each tier; weights precede KV. This is one explicit policy, not an optimized placement search.",
            "The same resident set is read at every step and its final two KV pages are updated. The experiment does not grow the resident set or execute transformer operators.",
            "Independent per-slot memory interfaces are priced by production DRAM/NAND cores. No shared GPU interconnect, compute, monetary or area budget is modeled.",
            "An infeasible all-HBM point demonstrates this normalized capacity limit; it does not define an infinite speedup or establish a commercial HBM/HBF optimum.",
        ],
    }


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--architecture", action="store_true",
                        help="run the normalized 40-page, five-point architecture sweep")
    arguments = parser.parse_args()
    report = run_architecture_budget_sweep() if arguments.architecture else run_budget_sweep()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        output = arguments.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
