#!/usr/bin/env python3
"""Compare targeted native reruns with the old native records and fresh simulation."""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RERUN = Path(r"F:\codex_project\37_LLMsim_native_rerun_2026-10-11")
OLD = ROOT / "docs" / "cost_logic_native_validation_2026-10-10"
FRESH = ROOT / "docs" / "simulation_error_2026-10-11" / "fresh_report.json"
OUT_DIR = ROOT / "docs" / "simulation_error_2026-10-11" / "native_rerun"

CASES = [
    ("qwen2_5_0_5b_q4_k_m", "qwen2_5_0_5b_q4_k_m"),
    ("qwen3_0_6b_f16", "qwen3_0_6b_f16"),
]
METRICS = {
    "TTFT": "engine_ttft_ns",
    "TPOT": "engine_tpot_ns",
    "E2E": "engine_e2e_ns",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def samples(doc, key):
    return [float(x[key]) for x in doc["samples"]]


def median_mad_cv(values):
    med = statistics.median(values)
    mad = statistics.median([abs(x - med) for x in values])
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean if mean else math.nan
    return med, mad, cv


def old_path(case, mode):
    return OLD / f"native_{case}_graph_{mode}.json"


def new_path(case, mode):
    expanded = RERUN / f"{case}_graph_{mode}_9formal.json"
    if expanded.exists():
        return expanded
    return RERUN / f"{case}_graph_{mode}.json"


def fresh_row(fresh, case, mode):
    for row in fresh["rows"]:
        if row["case_id"] == case and row["graph_mode"] == mode:
            return row
    raise KeyError((case, mode))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fresh = load(FRESH)
    result = {
        "schema": "heterollm.targeted-native-rerun/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rerun_directory": str(RERUN),
        "simulation_commit": fresh["simulator_commit"],
        "report_commit": "e9e6008",
        "order": [
            "qwen2_5_0_5b_q4_k_m_graph_off",
            "qwen3_0_6b_f16_graph_off",
            "qwen2_5_0_5b_q4_k_m_graph_on",
            "qwen3_0_6b_f16_graph_on",
        ],
        "configuration": {
            "prompt_tokens": 512,
            "output_tokens": 128,
            "context": 768,
            "batch": 512,
            "ubatch": 512,
            "threads": 16,
            "gpu_layers": -1,
            "parallel": 1,
            "flash_attn": "off",
            "warmups": 2,
            "formal_repetitions": 5,
            "server": r"F:\codex_project\_runtime_sources\llama.cpp\build-native-5080-sm120\bin\llama-server.exe",
        },
        "rows": [],
    }
    need_expand = []
    for case, _ in CASES:
        for mode in ("off", "on"):
            old = load(old_path(case, mode))
            new = load(new_path(case, mode))
            if old.get("status") != "completed" or new.get("status") != "completed":
                raise RuntimeError(f"incomplete record: {case} {mode}")
            row = {"case_id": case, "graph_mode": mode, "formal_repetitions": len(new["samples"]), "metrics": {}, "native_files": {
                "old": str(old_path(case, mode)), "rerun": str(new_path(case, mode))}}
            for label, key in METRICS.items():
                ov = samples(old, key)
                nv = samples(new, key)
                om, omad, ocv = median_mad_cv(ov)
                nm, nmad, ncv = median_mad_cv(nv)
                threshold = max(3 * omad / om if om else math.inf, 0.05)
                drift = (nm - om) / om if om else math.nan
                sim = float(fresh_row(fresh, case, mode)["metrics"][key]["simulation_ns"])
                row["metrics"][label] = {
                    "old_median_ns": om, "old_mad_ns": omad, "old_cv_percent": ocv * 100,
                    "rerun_median_ns": nm, "rerun_mad_ns": nmad, "rerun_cv_percent": ncv * 100,
                    "median_shift_percent": drift * 100,
                    "drift_threshold_percent": threshold * 100,
                    "threshold_exceeded": abs(drift) > threshold or ncv > 0.05,
                    "simulation_ns": sim,
                    "rerun_simulation_signed_error_percent": (sim - nm) / nm * 100,
                    "old_simulation_signed_error_percent": (sim - om) / om * 100,
                    "rerun_samples_ns": nv,
                    "old_samples_ns": ov,
                }
                if row["metrics"][label]["threshold_exceeded"]:
                    need_expand.append((case, mode, label))
            result["rows"].append(row)
    # Graph gain is an E2E comparison, useful for checking the primary residual mechanism.
    for case, _ in CASES:
        off = next(r for r in result["rows"] if r["case_id"] == case and r["graph_mode"] == "off")
        on = next(r for r in result["rows"] if r["case_id"] == case and r["graph_mode"] == "on")
        result.setdefault("graph_e2e_reduction", []).append({
            "case_id": case,
            "old_percent": (1 - off["metrics"]["E2E"]["old_median_ns"] / off["metrics"]["E2E"]["old_median_ns"] if False else (1 - on["metrics"]["E2E"]["old_median_ns"] / off["metrics"]["E2E"]["old_median_ns"]) * 100),
            "rerun_percent": (1 - on["metrics"]["E2E"]["rerun_median_ns"] / off["metrics"]["E2E"]["rerun_median_ns"]) * 100,
        })
    result["expand_to_9_formal_required"] = bool(need_expand)
    result["expand_candidates"] = [list(x) for x in need_expand]
    result["interpretation"] = {
        "native_boundary": "llama.cpp engine TTFT/TPOT/E2E; not a hardware DRAM truth",
        "backend_granularity": "native records do not contain kernel/op durations; profiler trace is required for backend-level attribution",
        "decision_rule": "expand if absolute median shift exceeds max(3*old MAD/old median, 5%) or rerun CV exceeds 5%",
    }
    out = OUT_DIR / "targeted_native_rerun.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "expand": result["expand_to_9_formal_required"], "candidates": result["expand_candidates"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
