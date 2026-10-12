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
    hbm_slots = int(hbm_slots)
    require(1 <= hbm_slots < TOTAL_SLOTS, "hbm_slots must be in [1, 3]")
    hbf_slots = TOTAL_SLOTS - hbm_slots
    profiles = [HBM_SLOT] * hbm_slots + [HBF_SLOT] * hbf_slots
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
        "working_set_pages": WORKING_SET_PAGES,
        "workload": {"weight_pages": WEIGHT_PAGES, "kv_pages": KV_PAGES,
                      "steps": 4, "bytes_per_access": PAGE},
        "slots": [profile.to_dict() for profile in profiles],
        "aggregate_budget": aggregate,
        "assumption": "hypothetical four-slot package; not commercial calibration",
    }
    if capacity_pages < WORKING_SET_PAGES:
        row.update({
            "runnable": False,
            "reason": (
                f"capacity {capacity_pages} pages is below fixed working set "
                f"{WORKING_SET_PAGES} pages"
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
    for _page in range(WORKING_SET_PAGES):
        try:
            slot = next(index for index, available in enumerate(remaining) if available)
        except StopIteration:
            raise AssertionError("capacity preflight disagrees with page placement")
        remaining[slot] -= 1
        assignment.append((slot, addresses[slot] * PAGE))
        addresses[slot] += 1

    by_tier: dict[str, list[Any]] = {"hbm": [], "hbf": []}
    by_slot: dict[str, list[Any]] = {str(index): [] for index in range(len(cores))}
    clock = 0.0
    for step in range(4):
        current = []
        for index, (slot, address) in enumerate(assignment):
            core, tier, _slot_index = cores[slot]
            result = _access(core, "read", address, clock, f"budget-weight-kv-{step}-{index}")
            by_tier[tier].append(result)
            by_slot[str(slot)].append(result)
            current.append(result)
        for page in (WEIGHT_PAGES + 6, WEIGHT_PAGES + 7):
            slot, address = assignment[page]
            core, tier, _slot_index = cores[slot]
            result = _access(core, "write", address, clock, f"budget-kv-append-{step}-{page}")
            by_tier[tier].append(result)
            by_slot[str(slot)].append(result)
            current.append(result)
        clock = max(result.completion_ns for result in current)

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
        "useful_memory_bandwidth_gb_s": logical_bytes / clock,
        "tiers": tier_metrics,
        "slot_traffic": {
            slot: {
                "logical_bytes": sum(item.logical_bytes for item in values),
                "completion_ns": max((item.completion_ns for item in values), default=0.0),
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
        "rows": rows,
        "runnable_count": len(runnable),
        "capacity_failure_count": len(rows) - len(runnable),
    }


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_budget_sweep()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        output = arguments.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
