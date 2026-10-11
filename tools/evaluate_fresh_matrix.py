"""Build a fresh, strict report from a completed 20-pair simulation matrix.

This script only reads saved submissions, results and native timing records.  It
does not start jobs and does not reuse old comparison rows as measurements.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from analyze_cuda_graph_validation import compare
from render_cuda_graph_validation_report import graph_speedup, native_metrics
from analyze_frontend_validation import requests


METRICS = ("engine_ttft_ns", "engine_tpot_ns", "engine_e2e_ns")
LABELS = {"engine_ttft_ns": "TTFT", "engine_tpot_ns": "TPOT", "engine_e2e_ns": "E2E"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percentile(values, q=0.90):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    return values[max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))]


def stats(errors):
    if not errors:
        return {"n": 0, "mape_percent": None, "signed_mean_percent": None,
                "p50_ape_percent": None, "p90_ape_percent": None,
                "max_ape_percent": None, "mad_ape_percent": None}
    absolute = [abs(float(x)) for x in errors]
    med = statistics.median(absolute)
    return {"n": len(errors), "mape_percent": statistics.fmean(absolute),
            "signed_mean_percent": statistics.fmean(errors),
            "p50_ape_percent": percentile(absolute, .50),
            "p90_ape_percent": percentile(absolute, .90),
            "max_ape_percent": max(absolute),
            "mad_ape_percent": statistics.median(abs(x - med) for x in absolute)}


def metric_row(sim, native):
    delta = float(sim) - float(native)
    signed = delta / float(native) * 100.0
    return {"simulation_ns": sim, "native_median_ns": native, "signed_error_ns": delta,
            "absolute_error_ns": abs(delta), "signed_error_percent": signed,
            "absolute_error_percent": abs(signed)}


def native_sample_parts(sample):
    prompt = sample.get("server_prompt_ms")
    predicted = sample.get("server_predicted_ms")
    if prompt is None or predicted is None:
        timings = sample.get("server_timings", {})
        prompt, predicted = timings.get("prompt_ms"), timings.get("predicted_ms")
    if prompt is None:
        prompt = sample["engine_ttft_ns"] / 1e6
    if predicted is None:
        predicted = sample["engine_e2e_ns"] / 1e6 - sample["engine_ttft_ns"] / 1e6
    return float(prompt) * 1e6, float(predicted) * 1e6


def find_manifest(base: Path):
    for name in ("fresh_manifest.json", "manifest.json", "job_manifest.json", "ui_jobs.json", "ui_runs.json"):
        path = base / name
        if path.is_file():
            return path, read(path)
    return None, None


def manifest_job_id(manifest, slug, mode):
    if manifest is None:
        return None
    rows = (manifest.get("jobs", manifest.get("entries",
           manifest.get("runs", manifest))) if isinstance(manifest, dict) else manifest)
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("case_id", row.get("slug", "")))
        row_mode = str(row.get("graph_mode", row.get("mode", "")))
        prefix = str(row.get("prefix", ""))
        if not key and prefix.startswith("ui_"):
            key = prefix.removeprefix("ui_").removesuffix(f"_graph_{row_mode}")
        if key == slug and row_mode == mode:
            return row.get("job_id", row.get("id"))
    return None


def load_cases(base: Path):
    path = base / "cases.json"
    if path.is_file():
        data = read(path)
        cases = data.get("cases", data) if isinstance(data, dict) else data
        return [(row["case_id"], row.get("name", row["case_id"])) for row in cases]
    slugs = sorted({p.name.removeprefix("ui_").removesuffix("_graph_off_submission.json")
                    for p in base.glob("ui_*_graph_off_submission.json")})
    return [(slug, slug) for slug in slugs]


def compare_pair(base, slug, mode):
    row = compare(base, slug, mode)
    if row.get("status") != "compared":
        return row, None, None
    result = read(base / f"ui_{slug}_graph_{mode}_result.json")
    native = read(base / f"native_{slug}_graph_{mode}.json")
    realized = {r.get("request_id"): r for r in requests(result.get("report", result))}
    measured = realized.get("04_measured0")
    if not measured:
        row["status"] = "configuration_mismatch"
        row.setdefault("validation_errors", []).append("missing measured request 04_measured0")
        return row, None, native
    return row, measured, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-commit", required=True)
    args = parser.parse_args()
    base, out = args.matrix, args.output
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(base)
    if len(cases) != 10:
        raise SystemExit(f"expected 10 cases (20 pairs), found {len(cases)}")
    manifest_path, manifest = find_manifest(base)
    rows, modes = [], {}
    for slug, name in cases:
        modes[slug] = {}
        for mode in ("off", "on"):
            row, measured, native = compare_pair(base, slug, mode)
            created = read(base / f"ui_{slug}_graph_{mode}_job_created.json") if (base / f"ui_{slug}_graph_{mode}_job_created.json").is_file() else {}
            result = read(base / f"ui_{slug}_graph_{mode}_result.json") if (base / f"ui_{slug}_graph_{mode}_result.json").is_file() else {}
            ids = [created.get("job_id"), result.get("job_id"), manifest_job_id(manifest, slug, mode)]
            if any(x is None for x in ids) or len(set(ids)) != 1:
                row["status"] = "configuration_mismatch"
                row.setdefault("validation_errors", []).append("job_created/manifest/result job IDs are not identical")
            item = {"case_id": slug, "name": name, "graph_mode": mode, "status": row.get("status"),
                    "validation_errors": row.get("validation_errors", []), "job_id": row.get("job_id")}
            timing_keys = ("created_at", "started_at", "finished_at", "started_local", "finished_local",
                           "submitted_at", "completed_at", "duration_seconds", "wall_seconds")
            item["simulation_job_timing"] = {key: result[key] for key in timing_keys if key in result}
            if measured is not None and native is not None and row.get("status") == "compared":
                nstats = native_metrics(base / f"native_{slug}_graph_{mode}.json", mode)
                item["metrics"] = {key: metric_row(measured[key], nstats[key]["median_ns"]) for key in METRICS}
                item["native_mad_ns"] = {key: nstats[key]["mad_ns"] for key in METRICS}
                item["simulation"] = {"ttft_ns": measured["engine_ttft_ns"], "e2e_ns": measured["engine_e2e_ns"],
                                       "decode_ns": measured["engine_e2e_ns"] - measured["engine_ttft_ns"]}
                item["native_samples"] = native.get("samples", [])
            rows.append(item)
            modes[slug][mode] = item
    completed = sum(row["status"] == "compared" for row in rows)
    if completed != 20:
        raise SystemExit(f"strict gate: only {completed}/20 pairs passed existing pairing checks")

    aggregate = {}
    for group in ("off", "on", "all"):
        selected = [r for r in rows if group == "all" or r["graph_mode"] == group]
        aggregate[group] = {LABELS[key]: stats([r["metrics"][key]["signed_error_percent"] for r in selected]) for key in METRICS}
    ranked = sorted(rows, key=lambda r: r["metrics"]["engine_e2e_ns"]["absolute_error_ns"], reverse=True)
    ranked_ape = sorted(rows, key=lambda r: r["metrics"]["engine_e2e_ns"]["absolute_error_percent"], reverse=True)
    gains = []
    for slug, name in cases:
        pair = {m: {"comparison": modes[slug][m], "native": native_metrics(base / f"native_{slug}_graph_{m}.json", m)} for m in ("off", "on")}
        gain = graph_speedup(pair)
        gain["case_id"], gain["name"] = slug, name
        gains.append(gain)

    decomposition = []
    for slug, name in cases:
        for mode in ("off", "on"):
            item = modes[slug][mode]
            samples = item["native_samples"]
            sim = item["simulation"]
            residuals = {"ttft_ns": [], "decode_ns": [], "e2e_ns": []}
            native_parts = {"ttft_ns": [], "decode_ns": [], "e2e_ns": []}
            for sample in samples:
                prompt, decode = native_sample_parts(sample)
                native_parts["ttft_ns"].append(prompt)
                native_parts["decode_ns"].append(decode)
                native_parts["e2e_ns"].append(prompt + decode)
                residuals["ttft_ns"].append(sim["ttft_ns"] - prompt)
                residuals["decode_ns"].append(sim["decode_ns"] - decode)
                residuals["e2e_ns"].append(sim["e2e_ns"] - prompt - decode)
            means = {k: statistics.fmean(v) for k, v in residuals.items()}
            decomposition.append({"case_id": slug, "graph_mode": mode, "mean_residual_ns": means,
                                  "native_component_medians_ns": {k: statistics.median(v) for k, v in native_parts.items()},
                                  "simulation_components_ns": sim,
                                  "additivity_check_ns": means["ttft_ns"] + means["decode_ns"] - means["e2e_ns"],
                                  "note": "median(sum) != sum(medians); means are used for exact linear additivity"})

    report = {"schema": "heterollm.fresh-simulation-matrix-report/v1", "simulator_commit": args.simulator_commit,
              "matrix_directory": str(base), "completed_pairs": completed, "expected_pairs": 20,
              "native_reference": "reused 2026-10-08 native records; no native/profiler retest",
              "simulation_jobs": "new matrix results; job IDs checked against created record and manifest",
              "software_micro_evaluation": "not hardware measurement; kernel/DRAM are not separately attributable",
              "calibration": "no model modification and no native-latency calibration",
              "metrics": aggregate, "rows": rows, "e2e_absolute_rank": [r["case_id"] + ":" + r["graph_mode"] for r in ranked],
              "e2e_ape_rank": [r["case_id"] + ":" + r["graph_mode"] for r in ranked_ape],
              "graph_gain": gains, "decomposition": decomposition,
              "ranking_scope": "20 scenarios equally weighted; not production traffic weighting",
              "limitations": ["TTFT/decode decomposition uses native per-sample prompt_ms and predicted_ms medians via mean residuals.",
                              "GPU service intervals may overlap; they are not additive E2E time.",
                              "Fresh diagnostic retest is optional follow-up, not required for this report."],
              "manifest_file": str(manifest_path) if manifest_path else None}
    (out / "fresh_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 本轮新仿真误差报告", "", f"通过配对：{completed}/20；仿真 commit：`{args.simulator_commit}`。", "",
             "native 时间复用 2026-10-08 记录；本轮仅使用新 simulation job 结果。未进行 native/profiler 补测，未修改模型或用 native 时延校准。软件微评估不代表硬件真值，不能据此归因 kernel/DRAM。", "",
             "## 指标汇总", "", "|组|指标|MAPE|signed 均值|p50 APE|p90 APE|max APE|MAD APE|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group, vals in aggregate.items():
        for metric, v in vals.items():
            lines.append(f"|{group}|{metric}|{v['mape_percent']:.3f}%|{v['signed_mean_percent']:+.3f}%|{v['p50_ape_percent']:.3f}%|{v['p90_ape_percent']:.3f}%|{v['max_ape_percent']:.3f}%|{v['mad_ape_percent']:.3f}%|")
    lines += ["", "## 20 组明细（按 E2E 绝对误差排序）", "", "|case|Graph|E2E 仿真 ns|native 中位数 ns|signed %|绝对误差 ns|APE|", "|---|---|---:|---:|---:|---:|---:|"]
    for r in ranked:
        m = r["metrics"]["engine_e2e_ns"]
        lines.append(f"|{r['case_id']}|{r['graph_mode']}|{m['simulation_ns']:.3f}|{m['native_median_ns']:.3f}|{m['signed_error_percent']:+.3f}%|{m['absolute_error_ns']:.3f}|{m['absolute_error_percent']:.3f}%|")
    lines += ["", "## Graph 收益误差", "", "|case|native E2E 降幅|仿真 E2E 降幅|预测误差百分点|", "|---|---:|---:|---:|"]
    for g in gains:
        lines.append(f"|{g['case_id']}|{g['native_e2e_reduction_percent']:.3f}%|{g['simulation_e2e_reduction_percent']:.3f}%|{g['signed_reduction_error_percentage_points']:+.3f}pp|")
    lines += ["", "## 解释边界", "", "分解使用每个 native 样本的 prompt_ms 与 predicted_ms（缺失时使用 engine_e2e−engine_ttft），仿真 decode=E2E−TTFT。median(sum) 不等于 sum(medians)，因此平均残差才用于线性可加检查。GPU 服务区间可能重叠，不能简单相加为 E2E；20 组排名等权，不代表生产流量。深入补测与 profiler 诊断属于可选后续。"]
    (out / "fresh_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "fresh_report.json")
    print(out / "fresh_report.md")


if __name__ == "__main__":
    main()
