"""Conditional purchase-cost, package-area and uniform-wear sensitivities.

All prices, package area and endurance constants below are
analyst-selected hypothetical inputs. They are not market quotes, calibrated
device specifications or a coupled serving/FTL simulation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

YEAR_SECONDS = 365.25 * 24 * 3600
DEFAULT_ASSUMPTIONS = {
    "gpu_count": 1, "gpu_price_usd": 8000.0, "node_base_price_usd": 5000.0,
    "hbm_package_price_usd": 400.0, "hbf_package_price_usd": 200.0,
    "hbm_package_area_mm2": 100.0, "memory_area_budget_mm2": 400.0,
    "hbm_package_capacity_bytes": 36_000_000_000,
    "hbf_package_capacity_bytes": 512_000_000_000,
    "active_fraction": .6,
    "hypothetical_active_output_tps": 1000.0,
}


def _number(name: str, value: Any, *, minimum: float | None = None,
            positive: bool = False, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the supported range")
    return float(value)


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    number = _number(name, value, minimum=minimum)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def conditional_node_cost(*, hbm_packages: int, hbf_packages: int,
                          hbf_package_area_mm2: float = 100., years: float = 3.,
                          active_output_tps: float = 1000.,
                          assumptions: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Amortize declared purchase prices while holding supplied active TPS fixed."""
    unknown = set(assumptions or ()) - set(DEFAULT_ASSUMPTIONS)
    if unknown:
        raise ValueError("unknown assumptions: " + ", ".join(sorted(unknown)))
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    for name, value in a.items():
        _number(name, value, minimum=0)
    a["gpu_count"] = _integer("gpu_count", a["gpu_count"], minimum=1)
    for name in ("hbm_package_capacity_bytes", "hbf_package_capacity_bytes"):
        _integer(name, a[name], minimum=1)
    for name in ("hbm_package_area_mm2", "memory_area_budget_mm2"):
        _number(name, a[name], positive=True)
    _number("active_fraction", a["active_fraction"], positive=True, maximum=1.)
    hbm_packages = _integer("hbm_packages", hbm_packages)
    hbf_packages = _integer("hbf_packages", hbf_packages)
    _number("years", years, positive=True)
    _number("active_output_tps", active_output_tps, minimum=0.)
    _number("hbf_package_area_mm2", hbf_package_area_mm2, positive=True)
    memory_price = hbm_packages*a["hbm_package_price_usd"] + hbf_packages*a["hbf_package_price_usd"]
    capex = a["node_base_price_usd"] + a["gpu_count"]*a["gpu_price_usd"] + memory_price
    output_tokens = active_output_tps*a["active_fraction"]*years*YEAR_SECONDS
    area = hbm_packages*a["hbm_package_area_mm2"] + hbf_packages*hbf_package_area_mm2
    return {
        "hbm_packages": hbm_packages, "hbf_packages": hbf_packages,
        "gpu_count": int(a["gpu_count"]), "years": years,
        "hbf_package_area_mm2": hbf_package_area_mm2,
        "active_output_tps_held_fixed": active_output_tps,
        "active_fraction": a["active_fraction"],
        "memory_package_area_mm2": area, "memory_area_budget_mm2": a["memory_area_budget_mm2"],
        "area_budget_feasible": area <= a["memory_area_budget_mm2"],
        "hbm_capacity_bytes": hbm_packages*a["hbm_package_capacity_bytes"],
        "hbf_capacity_bytes": hbf_packages*a["hbf_package_capacity_bytes"],
        "capex_usd": capex,
        "conditional_output_tokens": output_tokens,
        "conditional_usd_per_million_tokens": capex/output_tokens*1e6 if output_tokens else None,
        "formula_scope": "Purchase cost only; TPS is held fixed, not re-simulated for these package/area assumptions; area failure makes the hardware proposal infeasible but does not erase its conditional cost arithmetic",
    }


def uniform_wear_estimate(*, media_write_bytes_per_active_second: float,
                          capacity_bytes: int, rated_pe_cycles: int,
                          active_fraction: float = .6, additional_gc_factor: float = 1.,
                          years: float = 3., replacement_cost_usd: float = 800.) -> dict[str, Any]:
    """Uniform full-capacity wear bound, deliberately separate from real FTL."""
    _number("media_write_bytes_per_active_second", media_write_bytes_per_active_second, minimum=0.)
    capacity_bytes = _integer("capacity_bytes", capacity_bytes, minimum=1)
    rated_pe_cycles = _integer("rated_pe_cycles", rated_pe_cycles, minimum=1)
    _number("active_fraction", active_fraction, positive=True, maximum=1.)
    _number("additional_gc_factor", additional_gc_factor, minimum=1.)
    _number("years", years, positive=True)
    _number("replacement_cost_usd", replacement_cost_usd, minimum=0.)
    rate = media_write_bytes_per_active_second*active_fraction*additional_gc_factor
    annual_pe = rate*YEAR_SECONDS/capacity_bytes
    lifetime = rated_pe_cycles/annual_pe if annual_pe else None
    replacements = max(0, math.ceil(years/lifetime)-1) if lifetime else 0
    return {
        "base_media_write_bytes_per_active_second": media_write_bytes_per_active_second,
        "active_fraction": active_fraction, "additional_gc_factor": additional_gc_factor,
        "capacity_bytes": capacity_bytes, "rated_pe_cycles": rated_pe_cycles,
        "study_years": years, "average_media_write_bytes_per_second": rate,
        "pe_cycles_per_year": annual_pe, "pe_cycles_consumed_in_study": annual_pe*years,
        "uniform_wear_lifetime_years": lifetime,
        "replacement_count_before_study_end": replacements,
        "replacement_cost_usd": replacements*replacement_cost_usd,
        "zero_write_semantics": "null lifetime at zero write rate means this wear mechanism is inactive, not infinite hardware lifetime",
        "scope": "uniform physical write distribution over all declared NAND bytes; ignores retention, spare area, nonuniform wear and failure modes; additional GC multiplies already-physical baseline traffic once",
    }


def run_economics_sensitivity(serving_rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(serving_rows or ())
    costs = [conditional_node_cost(hbm_packages=hbm, hbf_packages=4-hbm,
                                  hbf_package_area_mm2=area,
                                  years=years)
             for hbm in range(5)
             for area in (50., 100., 150.) for years in (3., 5.)]
    amortization = [conditional_node_cost(hbm_packages=2, hbf_packages=2, years=years,
                                       assumptions={"active_fraction": duty})
                 for duty in (.3, .6, .9) for years in (3., 5.)]
    projections = []
    write_source_assessments = []
    write_sources = [{"id": "assumed_1GB_per_active_second", "rate": 1e9,
                      "source": "analyst-selected physical media write rate, not an observed trace"}]
    # Preserve only named valid source points; do not assign an unmodeled cost
    # curve to a serving architecture or claim unchanged TPS under repackaging.
    for row in rows:
        tps_value = row.get("node_tps")
        tps = _number("source node_tps", 0. if tps_value is None else tps_value, minimum=0.)
        if tps <= 0:
            continue
        projections.append({"source_id": row.get("id"), "source_model": row.get("model"),
                            "source_placement": row.get("placement"), "source_tps": tps,
                            "source_measurement_window": row.get("measurement_window"),
                            "cost_assumption": "hypothetical 2-HBM/2-HBF node; source TPS held fixed, no physical topology equivalence claimed",
                            "cost": conditional_node_cost(hbm_packages=2, hbf_packages=2, active_output_tps=tps,
                                                          assumptions={"gpu_count": row.get("gpu_count", 1)})})
        storage = (row.get("resources") or {}).get("storage_traffic") or {}
        owners = storage.get("owner_ids") or []
        duration_value = row.get("makespan_ns")
        duration = _number("source makespan_ns", 0. if duration_value is None else duration_value, minimum=0.)
        # A legacy component ID can still say hbf0 after its physical kind is
        # changed to SSD. Match resource owners to declared component kinds,
        # never infer flash technology from an ID or placement label.
        memory_kinds = (row.get("applied_hardware") or {}).get("memory_kinds") or {}
        if not isinstance(memory_kinds, Mapping):
            raise ValueError("applied_hardware.memory_kinds must be a mapping")
        owner_kinds = []
        for owner in owners:
            matches = [key for key in memory_kinds
                       if str(owner) == key or str(owner).startswith(key + ".") or str(owner).startswith(key + ":")]
            owner_kinds.append(str(memory_kinds[max(matches, key=len)]).lower() if matches else None)
        verified_hbf = bool(owners) and all(kind == "hbf" for kind in owner_kinds)
        write_source_assessments.append({
            "source_id": row.get("id"), "storage_owner_ids": owners, "resolved_memory_kinds": owner_kinds,
            "status": ("verified_hbf" if verified_hbf else "no_storage_owner" if not owners
                       else "unverified_owner_kind" if None in owner_kinds else "non_hbf_or_mixed_storage"),
            "used_for_hbf_uniform_wear": verified_hbf and duration > 0,
        })
        if duration > 0 and verified_hbf:
            physical_write_bytes = _number("source HBF physical_write_bytes", storage.get("physical_write_bytes") or 0., minimum=0.)
            write_sources.append({"id": row.get("id"),
                                  "rate": physical_write_bytes*1e9/duration,
                                  "source": "observed operator HBF physical writes / primary trace duration; excludes separately accounted migrations"})
    wear = [{"write_rate_source_id": source["id"], "write_rate_source": source["source"],
             **uniform_wear_estimate(media_write_bytes_per_active_second=source["rate"],
                                    capacity_bytes=capacity, rated_pe_cycles=pe,
                                    additional_gc_factor=gc, years=years,
                                    replacement_cost_usd=capacity/DEFAULT_ASSUMPTIONS["hbf_package_capacity_bytes"]*DEFAULT_ASSUMPTIONS["hbf_package_price_usd"])}
            for source in write_sources for capacity in (512_000_000_000, 2_048_000_000_000)
            for pe in (1000, 3000, 10000) for gc in (1., 3., 10., 30.) for years in (3., 5.)]
    return {
        "schema_version": 2, "id": "assumed_purchase_cost_area_wear",
        "calibrated": False, "status": "conditional_sensitivity_completed",
        "assumption_origin": "Every price, area, P/E and duty-cycle input is an explicit analyst-selected hypothesis, not an externally verified device specification.",
        "assumptions": DEFAULT_ASSUMPTIONS,
        "package_cost_area_rows": costs, "purchase_cost_amortization_rows": amortization,
        "serving_tps_conditional_projections": projections,
        "write_rate_source_assessments": write_source_assessments,
        "uniform_wear_rows": wear,
        "units": {"cost_per_token": "purchase USD per million visible output tokens"},
        "limitations": [
            "These are conditional estimates, not calibrated HBF costs, guaranteed lifespan or a commercial optimum.",
            "Cost/area scans hold TPS fixed and do not re-execute serving; no benefit in TPS, goodput or session capacity follows from these rows.",
            "Cost includes stated purchase prices only. Wear replacement costs are reported separately; labor, financing, tax, downtime and service fees are excluded.",
            "Uniform-wear lifetime excludes stateful FTL, retention and nonuniform wear.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    source_rows = []
    if args.results:
        existing = json.loads(args.results.read_text(encoding="utf-8"))
        source_rows = [r for group in existing.get("groups", ()) for r in group.get("rows", ())
                       if r.get("mode") == "full_operator_serving"]
    output = run_economics_sensitivity(source_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"calibrated": False, "status": output["status"], "cost_rows": len(output["package_cost_area_rows"]),
                      "wear_rows": len(output["uniform_wear_rows"])}))
