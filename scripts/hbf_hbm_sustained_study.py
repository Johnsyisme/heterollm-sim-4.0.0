"""Finite-horizon arrival stability and adjacent concurrent-session boundaries.

All points run the full serving operator graph. Completion rates are measured
before draining the final queue; warmup is excluded explicitly. These bounded
traces do not prove infinite-horizon stability or production tail latency.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hbf_hbm_serving_study import build_study_scenario, run_case
from heterollm_sim.serving import RequestStatus


def summarize_sustained(serving, *, horizon_end_ns, warmup_fraction=.25,
                        slo_ttft_ns=1e6, slo_tbt_ns=1e6, batch=16):
    metrics = sorted(serving.request_metrics.values(), key=lambda item: (item.arrival_ns, item.request_id))
    if not metrics or not 0 <= warmup_fraction < 1:
        raise ValueError("sustained measurement requires requests and a warmup fraction in [0,1)")
    warmup_count = int(len(metrics) * warmup_fraction)
    start = metrics[warmup_count].arrival_ns
    duration = horizon_end_ns - start
    if duration <= 0:
        raise ValueError("measurement horizon must extend beyond warmup")
    selected = [metric for metric in metrics if start <= metric.arrival_ns < horizon_end_ns]
    def backlog(at):
        return sum(metric.arrival_ns < at and (metric.finish_ns is None or metric.finish_ns >= at)
                   for metric in metrics)
    completed = [metric for metric in metrics if metric.status is RequestStatus.FINISHED
                 and metric.finish_ns is not None and start <= metric.finish_ns < horizon_end_ns]
    token_times = {}
    for event in serving.events:
        if event.event_type == "tokens_committed" and event.request_id:
            token_times.setdefault(event.request_id, []).extend(
                [event.timestamp_ns] * int(event.details.get("visible_tokens", 0)))
    qualified = []
    for metric in selected:
        times = token_times.get(metric.request_id, ())
        max_gap = max((b-a for a, b in zip(times, times[1:])), default=0.)
        if (metric.status is RequestStatus.FINISHED and metric.ttft_ns is not None
                and metric.ttft_ns <= slo_ttft_ns and max_gap <= slo_tbt_ns
                and metric.deadline_met is not False):
            qualified.append(metric)
    beginning, ending = backlog(start), backlog(horizon_end_ns)
    growth = ending - beginning
    terminal = all(metric.status is RequestStatus.FINISHED for metric in metrics)
    # The conservation identity must not lose unfinished requests by counting
    # completions during the later drain as if they occurred under offered load.
    conserved = len(selected) - len(completed) == growth
    measured = {"warmup_fraction": warmup_fraction, "warmup_requests": warmup_count,
        "window": {"start_ns": start, "end_ns": horizon_end_ns, "duration_ns": duration,
                   "end_inclusive": False, "includes_final_drain": False},
        "offered_requests": len(selected), "completed_during_offered_window": len(completed),
        "offered_rps": len(selected) * 1e9 / duration,
        "completed_rps": len(completed) * 1e9 / duration,
        "backlog_at_start": beginning, "backlog_at_end": ending,
        "backlog_growth_requests": growth, "backlog_growth_rps": growth * 1e9 / duration,
        "backlog_conservation": conserved,
        "qualified_requests": len(qualified), "qualification_rate": len(qualified) / len(selected),
        "all_requests_eventually_complete": terminal,
        "drain_completion_ns": max((metric.finish_ns or 0 for metric in metrics), default=0),
    }
    return classify_sustained(measured, total_requests=len(metrics), batch=batch)


def classify_sustained(measured, *, total_requests, batch=16):
    """Fixed batch-quantized acceptance screen, preserving raw observations."""
    measured = dict(measured)
    completion_fraction = measured["completed_during_offered_window"] / measured["offered_requests"]
    reasons = []
    if not measured["all_requests_eventually_complete"]:
        reasons.append("unfinished_or_rejected_requests")
    if not measured["backlog_conservation"]:
        reasons.append("backlog_accounting_mismatch")
    if measured["qualification_rate"] < 1.:
        reasons.append("post_warmup_latency_SLO_failure")
    if measured["backlog_growth_requests"] > batch:
        reasons.append("backlog_growth_exceeds_one_batch")
    if measured["backlog_at_end"] > 2 * batch:
        reasons.append("terminal_backlog_exceeds_two_batches")
    if total_requests >= 512 and completion_fraction < .95:
        reasons.append("completed_offered_fraction_below_95pct")
    measured.update(total_requests=total_requests, backlog_growth_tolerance_requests=batch,
        terminal_backlog_limit_requests=2 * batch, completed_offered_fraction=completion_fraction,
        batch_phase_completion_fraction_bound=batch / measured["offered_requests"],
        finite_horizon_sustainable=not reasons, failure_reasons=reasons,
        criterion="all post-warmup arrivals meet TTFT and every TBT; backlog growth <=one batch, end backlog <=two batches; >=512 requests require completed/offered>=95%; completions exclude drain")
    return measured


def run_arrival_case(architecture, rate_rps, count, *, progress=True):
    from hbf_hbm_experiments import serving_row
    interval = 1e9 / rate_rps
    identifier = f"sustained/{architecture}/rps-{rate_rps:g}/n-{count}"
    if progress:
        print("[sustained] " + identifier, flush=True)
    captured = []
    scenario = build_study_scenario(architecture, prompt=32, output=4, batch=16,
        request_count=count, interval_ns=interval, slo_ttft_ns=1e6, slo_tbt_ns=1e6)
    row = serving_row(scenario, identifier, placement=architecture, result_observer=captured.append)
    row["parameters"] = {"prompt": 32, "output": 4, "batch": 16,
        "rate_rps": rate_rps, "request_count": count, "interval_ns": interval,
        "slo_ttft_ns": 1e6, "slo_tbt_ns": 1e6}
    row["sustained"] = summarize_sustained(captured[0].serving,
        horizon_end_ns=count * interval)
    return row


def adjacent_integer_boundary(evaluate, low=16, high=64):
    """Local binary bracket, explicitly without a global monotonicity proof."""
    if not evaluate(low) or evaluate(high):
        raise ValueError("concurrency search requires a passing low and failing high")
    while high - low > 1:
        middle = (low + high) // 2
        if evaluate(middle):
            low = middle
        else:
            high = middle
    return low, high


def run_sustained_study(*, progress=True, reuse_rows=None):
    rows, boundaries, concurrent_boundaries = [], [], []
    arrival_cache = {}
    concurrency_reuse = {}
    # Explicit caller-provided observations from the same study contract can
    # be reclassified when measurement rules change; no timing is fabricated.
    for original in reuse_rows or ():
        row = copy.deepcopy(original)
        parameters = row.get("parameters", {})
        if ("sustained" in row and parameters.get("prompt") == 32
                and parameters.get("output") == 4 and parameters.get("batch") == 16
                and parameters.get("slo_ttft_ns") == 1e6 and parameters.get("slo_tbt_ns") == 1e6):
            row["sustained"] = classify_sustained(row["sustained"], total_requests=row["request_count"])
            arrival_cache[(row["placement"], float(parameters["rate_rps"]), parameters["request_count"])] = row
            rows.append(row)
        elif (row.get("id", "").startswith("sustained-concurrency/")
                and parameters.get("prompt") == 2048 and parameters.get("output") == 8
                and parameters.get("slo_ttft_ns") == 5e6 and parameters.get("slo_tbt_ns") == 1e6):
            concurrency_reuse[(row["placement"], parameters["batch"])] = row
            rows.append(row)
    def arrival(arch, rate, count):
        key = (arch, float(rate), count)
        if key not in arrival_cache:
            row = run_arrival_case(arch, rate, count, progress=progress)
            arrival_cache[key] = row
            rows.append(row)
        return arrival_cache[key]["sustained"]["finite_horizon_sustainable"]
    horizons = (128, 256, 512)
    for architecture in ("A_hbm_only", "B_all_kv_hbf"):
        grid = (2000., 8000., 32000., 128000.)
        evaluations = [(rate, arrival(architecture, rate, 256)) for rate in grid]
        passing = [rate for rate, passed in evaluations if passed]
        failing = [rate for rate, passed in evaluations if not passed]
        for _ in range(4):
            if passing:
                break
            rate = min(rate for rate, _ in evaluations) / 4
            passed = arrival(architecture, rate, 256)
            evaluations.append((rate, passed))
            (passing if passed else failing).append(rate)
        if not passing:
            raise RuntimeError("no feasible low arrival point in bounded search")
        low = max(passing)
        for _ in range(4):
            above = [rate for rate in failing if rate > low]
            if above:
                break
            rate = max(rate for rate, _ in evaluations) * 4
            passed = arrival(architecture, rate, 256)
            evaluations.append((rate, passed))
            (passing if passed else failing).append(rate)
            if passed:
                low = rate
        above = [rate for rate in failing if rate > low]
        if not above:
            raise RuntimeError("no overloaded high arrival point in bounded search")
        high = min(above)
        for _ in range(4):
            middle = (low + high) / 2
            if arrival(architecture, middle, 256):
                low = middle
            else:
                high = middle
        # A 256-request point is insufficient: both boundaries must be
        # exercised on short and long horizons too. Back off/expand explicitly
        # if a transient makes the nominal bracket disagree across horizons.
        for _ in range(6):
            passes = [arrival(architecture, low, count) for count in horizons]
            ending_backlogs = [arrival_cache[(architecture, low, count)]["sustained"]["backlog_at_end"] for count in horizons]
            horizon_backlog_drift = max(ending_backlogs) - min(ending_backlogs)
            if all(passes) and horizon_backlog_drift <= 16:
                break
            low *= .9
        for _ in range(6):
            failures = [not arrival(architecture, high, count) for count in horizons]
            if all(failures):
                break
            high *= 1.1
        valid = all(passes) and all(failures) and horizon_backlog_drift <= 16
        boundaries.append({"placement": architecture, "passing_rate_rps": low,
            "failing_rate_rps": high, "horizons_requests": list(horizons),
            "passing_point_passes_by_horizon": passes, "failing_point_fails_by_horizon": failures,
            "passing_backlog_at_end_by_horizon": ending_backlogs,
            "cross_horizon_backlog_drift_requests": horizon_backlog_drift,
            "cross_horizon_backlog_drift_limit_requests": 16,
            "failing_point_failure_reasons_by_horizon": [arrival_cache[(architecture, high, count)]["sustained"]["failure_reasons"] for count in horizons],
            "bracket_valid": valid,
            "highest_tested_joint_feasible_rate_rps": max((rate for arch, rate, _count in arrival_cache
                if arch == architecture and all((arch, rate, count) in arrival_cache
                    and arrival_cache[(arch, rate, count)]["sustained"]["finite_horizon_sustainable"] for count in horizons)
                and max(arrival_cache[(arch, rate, count)]["sustained"]["backlog_at_end"] for count in horizons)
                    - min(arrival_cache[(arch, rate, count)]["sustained"]["backlog_at_end"] for count in horizons) <= 16), default=None),
            "scope": "bounded periodic synthetic arrival trace; local binary-search bracket; no global monotonicity or infinite-horizon proof"})
        if not valid:
            raise RuntimeError("could not validate sustained pass/fail bracket over all horizons")
        concurrent_cache = {}
        def evaluate_concurrency(concurrency):
            if concurrency not in concurrent_cache:
                reused = concurrency_reuse.get((architecture, concurrency))
                if reused is not None:
                    concurrent_cache[concurrency] = reused
                    return reused["concurrency_boundary_pass"]
                identifier = f"sustained-concurrency/{architecture}/{concurrency}"
                if progress:
                    print("[sustained] " + identifier, flush=True)
                row = run_case(architecture, row_id=identifier, prompt=2048,
                    batch=concurrency, output=8, slo_ttft_ns=5e6, slo_tbt_ns=1e6)
                passed = (bool(row.get("executed_serving")) and row.get("request_count", 0) > 0
                    and row.get("completed_requests") == row.get("request_count") == row.get("qualified_requests"))
                row["concurrency_boundary_pass"] = passed
                concurrent_cache[concurrency] = row
                rows.append(row)
            return concurrent_cache[concurrency]["concurrency_boundary_pass"]
        low_batch, high_batch = adjacent_integer_boundary(evaluate_concurrency)
        concurrent_boundaries.append({"placement": architecture,
            "passing_sessions_per_gpu": low_batch, "failing_sessions_per_gpu": high_batch,
            "tested_sessions": sorted(concurrent_cache), "gpu_count": 1,
            "prompt_tokens": 2048, "output_tokens": 8, "slo_ttft_ns": 5e6, "slo_tbt_ns": 1e6,
            "scope": "finite simultaneous independent requests; local adjacent binary-search bracket, not multi-turn or infinite-session capacity"})
    checks = {
        "all_full_operator_runs_complete": all(row["completed_requests"] == row["request_count"] for row in rows),
        "all_arrival_windows_conserve_backlog": all(row["sustained"]["backlog_conservation"] for row in rows if "sustained" in row),
        "both_architectures_have_three_horizon_brackets": len(boundaries) == 2 and all(item["bracket_valid"] for item in boundaries),
        "adjacent_concurrency_boundaries": len(concurrent_boundaries) == 2 and all(
            item["failing_sessions_per_gpu"] == item["passing_sessions_per_gpu"] + 1 for item in concurrent_boundaries),
    }
    passed = all(checks.values())
    return {"id": "16_sustained", "status": "passed" if passed else "failed", "rows": rows,
        "checks": checks, "arrival_boundaries": boundaries, "concurrency_boundaries": concurrent_boundaries,
        "capability": {"supported": ["full_operator_arrival_backlog_measurement", "warmup_exclusion",
            "three_horizon_finite_sustainability_bracket", "adjacent_concurrency_local_boundary"] if passed else [],
            "unsupported": ["infinite-horizon stability proof", "empirical production tails", "global monotonicity proof"]},
        "measurement_criteria": {"warmup_fraction": .25, "batch_quantum_requests": 16,
            "max_within_window_backlog_growth_requests": 16, "max_end_backlog_requests": 32,
            "minimum_completed_offered_fraction_at_512_requests": .95,
            "max_backlog_end_range_across_horizons_requests": 16,
            "all_post_warmup_arrivals_meet_ttft_and_every_tbt": True,
            "scope": "fixed finite-window acceptance screen; batch phase uncertainty explicitly bounded, not a production maximum claim"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hbf_hbm/16_sustained.json")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    previous = json.loads(args.output.read_text(encoding="utf-8"))["rows"] if args.reuse_existing and args.output.exists() else None
    result = run_sustained_study(reuse_rows=previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(result["status"], len(result["rows"]), args.output, flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
