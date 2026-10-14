"""Opt-in physical NAND wear feedback, with explicit synthetic inputs.

All delays below are produced by NandCore during execution. This study does
not rescale an existing token-throughput result or claim silicon calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from heterollm_sim.memory_types import AccessRequest, Operation
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.physical_contract import require_physical_memory_config, reconfigure_physical_config
from heterollm_sim.nand_core import NandCore, summarize_nand_observations

PAGE = 4096


def wear_assumption(*, initial_pe_cycles=2968):
    config = resolve_physical_config("research_hbf_wear_trace")
    wear = {**asdict(config.wear_model), "initial_pe_cycles": initial_pe_cycles}
    return reconfigure_physical_config(config, overrides={"wear_model": wear}).wear_model


def run_controlled_trace(*, wear_enabled=True, cycles=32,
                         idle_ns=32e6):
    """Explicit erase/program bursts and skewed reads, with explicit idle intervals.

    Erases are submitted by the trace, not inferred from bytes or hidden FTL.
    Four physical planes share two execution slots; each cycle programs the
    first page of one block per plane, then reads those pages with a 29:1:1:1
    skew. The idle gap follows completion, so modes have different wall time.
    """
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("cycles must be a positive integer")
    if isinstance(idle_ns, bool) or not isinstance(idle_ns, (int, float)) or not 0 <= idle_ns < float("inf"):
        raise ValueError("idle_ns must be finite and nonnegative")
    config = resolve_physical_config("research_hbf_wear_trace", overrides={
        "wear_model": wear_assumption() if wear_enabled else None})
    core = NandCore(config, capture_details=False)
    observations, cycle_rows = [], []
    now = 0.
    operation_count = 0
    byte_totals = {key: 0 for key in ("logical_bytes", "physical_read_bytes", "physical_write_bytes")}

    def submit(operation, page, arrival):
        nonlocal operation_count
        operation_count += 1
        row = core.submit(AccessRequest(str(operation_count), operation, page*PAGE, PAGE, arrival))
        observations.append(row.counters["nand_observations"])
        byte_totals["logical_bytes"] += row.logical_bytes
        for key in ("physical_read_bytes", "physical_write_bytes"):
            byte_totals[key] += row.counters.get(key, 0)
        return row.completion_ns

    for cycle in range(cycles):
        start = now
        erase_done = max(submit(Operation.ERASE, page, now) for page in range(4))
        program_done = max(submit(Operation.WRITE, page, erase_done) for page in range(4))
        read_done = max(submit(Operation.READ, page, program_done) for page in [0]*29+[1, 2, 3])
        cycle_rows.append({"cycle":cycle, "start_ns":start, "end_ns":read_done,
                           "following_idle_ns":idle_ns if cycle+1 < cycles else 0.})
        now = read_done + (idle_ns if cycle+1 < cycles else 0.)
    summary = summarize_nand_observations(observations)
    return {"id":f"trace-wear-{int(wear_enabled)}",
        "mode":"physical_nand_trace_microbenchmark", "node_tps":None,
        "wear_enabled":wear_enabled,
        "assumption_status":"explicit_uncalibrated_sensitivity",
        "applied_config":asdict(config), "cycles":cycles, "request_count":operation_count,
        "duration_ns":now, "cycle_timeline":cycle_rows, **byte_totals,
        "media_gb_s":(byte_totals["physical_read_bytes"]+byte_totals["physical_write_bytes"])/now,
        "observations":summary,
        "workload_scope":"four explicit block erases, four page programs, 32 page reads per cycle; 29:1:1:1 read skew; idle gaps follow actual burst completion",
        "wear_scope":"P/E equals initial authored cycles plus actual explicit block erases. Programs alone never imply erase. No FTL, ECC, data corruption or hard end-of-life failure model."}


def build_feedback_scenario(architecture, *, wear_enabled=False,
                            prompt=128, output=8):
    from hbf_hbm_serving_study import build_study_scenario
    scenario = build_study_scenario(architecture, prompt=prompt, batch=1, output=output,
                                    request_count=2, interval_ns=50000.)
    components = []
    for component in scenario.hardware.components:
        if component.component_id == "hbf0":
            metadata = {key:value for key,value in component.metadata.items() if key != "memory_service"}
            config = require_physical_memory_config(component)
            metadata["physical_memory_config"] = reconfigure_physical_config(config, overrides={
                "detailed_observability": True,
                "wear_model": wear_assumption(initial_pe_cycles=3000) if wear_enabled else None})
            component = replace(component, metadata=metadata)
        components.append(component)
    return replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)))


def run_serving_feedback():
    from hbf_hbm_experiments import serving_row
    rows = []
    # Identical requests, GPU, memory capacities, weights and placement per pair.
    # The A pair is a negative control: its active path never uses HBF.
    for architecture in ("A_hbm_only", "B_all_kv_hbf"):
        for wear_enabled in (False, True):
            row = serving_row(build_feedback_scenario(architecture, wear_enabled=wear_enabled),
                f"14-{architecture}-wear-{int(wear_enabled)}", placement=architecture)
            row["wear_assumptions"] = {
                "wear":asdict(wear_assumption(initial_pe_cycles=3000)) if wear_enabled else None,
                "mode":"online physical-stage feedback, affecting actual serving dependencies",
                "wear_scope":"pre-aged initial P/E sensitivity; serving has no implicit erases",
                "calibrated":False}
            rows.append(row)
    return rows


def run_wear_study():
    traces = [run_controlled_trace(wear_enabled=wear) for wear in (False, True)]
    rows = run_serving_feedback()
    control, wear = traces
    a0, a1, b0, b1 = rows
    checks = {
        "identical_microbenchmark_bytes":len({(r["logical_bytes"],r["physical_read_bytes"],r["physical_write_bytes"]) for r in traces})==1,
        "long_trace_at_least_one_second":all(r["duration_ns"]>=1e9 for r in traces),
        "explicit_erase_program_conservation":all(r["observations"]["actual_erase_operations"]==4*r["cycles"] and r["observations"]["actual_program_operations"]==4*r["cycles"] for r in traces),
        "wear_changes_subsequent_program_work":wear["duration_ns"]>control["duration_ns"] and all(b["pe_cycles_after"]==3000 for b in wear["observations"]["wear_blocks"].values()),
        "real_read_and_program_queues_recorded":wear["observations"]["read_queue_depth"]["peak"]>0 and wear["observations"]["program_queue_depth"]["peak"]>0,
        "four_physical_planes_distinct_from_two_slots":wear["observations"]["physical_plane_imbalance"]["observed_planes"]==4 and len({e["execution_resource"] for e in wear["observations"]["events"]})==2,
        "full_serving_complete":all(r["completed_requests"]==r["request_count"]==2 and r["visible_output_tokens"]==16 for r in rows),
        "hbm_negative_control_unchanged":a0["node_tps"]==a1["node_tps"] and a0["ttft_ns"]==a1["ttft_ns"],
        "hbf_feedback_reaches_request_metrics":b1["node_tps"]<b0["node_tps"] and b1["ttft_ns"]["p50"]>b0["ttft_ns"]["p50"] and b1["memory_metrics"]["dynamic_wear"]["wear_extra_ns"]>0,
        "serving_does_not_infer_erase_cycles":all(r["resources"]["storage_traffic"].get("nand_observations",{}).get("actual_erase_operations",0)==0 for r in rows),
    }
    return {"id":"14_wear", "title":"显式擦写磨损与真实页队列/plane反馈",
        "status":"passed" if all(checks.values()) else "failed", "checks":checks,
        "capability":{"supported":all(checks.values()), "modeled":True,"calibrated":False,
            "scope":"explicit per-block erase counts, program-latency wear assumption, eligible page queue timelines and actual mapped plane loads"},
        "rows":rows, "trace_rows":traces,
        "serving_comparisons":[{"placement":architecture,
            "baseline_id":base["id"],"feedback_id":feedback["id"],
            "feedback_over_baseline_tps":feedback["node_tps"]/base["node_tps"],
            "feedback_over_baseline_ttft_p50":feedback["ttft_ns"]["p50"]/base["ttft_ns"]["p50"]}
            for architecture,base,feedback in (("A_hbm_only",a0,a1),("B_all_kv_hbf",b0,b1))],
        "limitations":[
            "Wear-latency curves are explicit stress assumptions, not vendor specs or external calibration.",
            "Read/program queues measure eligible-to-array-start wait excluding service. Program eligibility follows data arrival at the NAND plane; host admission and buffer waits are separate.",
            "Physical-plane load follows configured address mapping. Effective execution slots may be shared and are reported separately.",
            "P/E advances only on trace-authored block erase; FTL, automatic GC, ECC, retention, cell contents and end-of-life failure remain outside this study.",
            "The serving wear pair is a pre-aged-device sensitivity. Runtime serving writes do not silently create erase cycles.",
            "Detailed observation disables bulk page acceleration and is opt-in; original configurations keep their prior execution path.",
            "Microbenchmark idle gaps follow completion. Compare paired identical byte work and actual durations; these rows have no GPU token-throughput interpretation."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/hbf_hbm/14_wear.json")
    args = parser.parse_args()
    result = run_wear_study()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":result["checks"],"out":str(args.out)}))
