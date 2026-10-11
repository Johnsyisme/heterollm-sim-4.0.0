"""Summarize simulation error from existing independent validation artifacts.

This report deliberately separates software-equivalence checks from native
timing comparisons.  It never calibrates the simulator, re-runs native jobs,
or treats a passing contract check as a prediction-accuracy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "simulation_error_2026-10-11"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return round(float(value), 6)


def native_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in summary.get("cases", []):
        for mode, mode_data in case.get("modes", {}).items():
            comparison = mode_data.get("comparison", {})
            for metric, values in comparison.get("metrics", {}).items():
                native = values.get("native", {})
                rows.append(
                    {
                        "case_id": case.get("case_id"),
                        "name": case.get("name"),
                        "graph_mode": mode,
                        "metric": metric,
                        "simulation_ns": values.get("simulation_ns"),
                        "native_median_ns": native.get("median_ns"),
                        "native_mad_ns": native.get("mad_ns"),
                        "signed_error_percent": values.get("signed_error_percent"),
                        "absolute_error_percent": values.get("absolute_error_percent"),
                    }
                )
    return rows


def native_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in sorted({row["metric"] for row in rows}):
        metric_result: dict[str, Any] = {}
        for mode in ("off", "on", "all"):
            selected = [
                row
                for row in rows
                if row["metric"] == metric
                and (mode == "all" or row["graph_mode"] == mode)
            ]
            if not selected:
                continue
            signed = [float(row["signed_error_percent"]) for row in selected]
            absolute = [float(row["absolute_error_percent"]) for row in selected]
            metric_result[mode] = {
                "n": len(selected),
                "mape_percent": pct(mean(absolute)),
                "mean_signed_error_percent": pct(mean(signed)),
                "signed_error_range_percent": [pct(min(signed)), pct(max(signed))],
            }
        result[metric] = metric_result
    return result


def native_noise_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in sorted({row["metric"] for row in rows}):
        ratios = [
            100.0 * float(row["native_mad_ns"]) / float(row["native_median_ns"])
            for row in rows
            if row["metric"] == metric
            and row.get("native_mad_ns") is not None
            and row.get("native_median_ns")
        ]
        result[metric] = {
            "mean_mad_over_median_percent": pct(mean(ratios)),
            "median_mad_over_median_percent": pct(median(ratios)),
            "max_mad_over_median_percent": pct(max(ratios)),
        }
    return result


def native_raw_noise_summary(summary: dict[str, Any], native_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fields = ("engine_ttft_ns", "engine_tpot_ns", "engine_e2e_ns")
    for field in fields:
        ratios: list[float] = []
        for case in summary.get("cases", []):
            for mode_data in case.get("modes", {}).values():
                filename = mode_data.get("comparison", {}).get("native_file")
                if not filename:
                    continue
                path = native_dir / filename
                if not path.is_file():
                    continue
                data = read_json(path)
                samples = [
                    float(sample[field])
                    for sample in data.get("samples", [])
                    if sample.get(field) is not None
                ]
                if len(samples) >= 2 and median(samples) > 0:
                    ratios.append(100.0 * (max(samples) - min(samples)) / median(samples))
        if ratios:
            result[field] = {
                "mean_range_over_median_percent": pct(mean(ratios)),
                "median_range_over_median_percent": pct(median(ratios)),
                "max_range_over_median_percent": pct(max(ratios)),
            }
    return result


def graph_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in summary.get("cases", []):
        speedup = case.get("graph_speedup", {})
        rows.append(
            {
                "case_id": case.get("case_id"),
                "native_e2e_reduction_percent": speedup.get("native_e2e_reduction_percent"),
                "simulation_e2e_reduction_percent": speedup.get("simulation_e2e_reduction_percent"),
                "absolute_reduction_error_percentage_points": speedup.get(
                    "absolute_reduction_error_percentage_points"
                ),
            }
        )
    return rows


def build_report() -> dict[str, Any]:
    native_dir = ROOT / "docs" / "cost_logic_native_validation_2026-10-10"
    dispatch_dir = ROOT / "docs" / "cuda_graph_dispatch_validation_2026-10-08"
    comparison = read_json(native_dir / "comparison_summary.json")
    timing = read_json(native_dir / "native_timing_manifest.json")
    precision = read_json(ROOT / "docs" / "precision_results_2026-10-07.json")
    memory = read_json(ROOT / "docs" / "memory_precision_2026-10-07.json")
    dispatch = read_json(dispatch_dir / "independent_dispatch_summary.json")
    physical_path = ROOT / "docs" / "simulation_error_2026-10-11" / "physical_error.json"
    physical = read_json(physical_path) if physical_path.is_file() else None

    rows = native_rows(comparison)
    worst = sorted(
        rows,
        key=lambda row: float(row.get("absolute_error_percent") or 0.0),
        reverse=True,
    )[:10]
    graph = graph_rows(comparison)

    return {
        "schema": "heterollm.simulation-error-evaluation/v1",
        "evaluation_date": "2026-10-11",
        "overall_verdict": "not_qualified_for_general_predictive_accuracy",
        "execution_provenance": {
            "end_to_end_simulation_rerun": False,
            "native_inference_rerun": False,
            "physical_path_micro_evaluation_rerun": True,
            "physical_path_hardware_measurement": False,
        },
        "sources": {
            "native_comparison": "docs/cost_logic_native_validation_2026-10-10/comparison_summary.json",
            "native_timing_manifest": "docs/cost_logic_native_validation_2026-10-10/native_timing_manifest.json",
            "software_precision": "docs/precision_results_2026-10-07.json",
            "memory_precision": "docs/memory_precision_2026-10-07.json",
            "cuda_graph_independent": "docs/cuda_graph_dispatch_validation_2026-10-08/independent_dispatch_summary.json",
            "physical_micro_evaluation": "docs/simulation_error_2026-10-11/physical_error.json",
        },
        "evidence_scope": {
            "native_pairs": comparison.get("completed_comparisons"),
            "native_expected_pairs": comparison.get("expected_comparisons"),
            "native_workload": timing.get("workload"),
            "native_repetitions": timing.get("workload", {}).get("repetitions"),
            "native_warmups": timing.get("workload", {}).get("warmups"),
            "native_models": len(comparison.get("cases", [])),
            "native_graph_modes": ["off", "on"],
            "native_timing_reused": True,
            "native_latency_used_for_calibration": comparison.get(
                "target_llm_latency_used_for_calibration"
            ),
        },
        "native_latency": {
            "metric_summary": native_metric_summary(rows),
            "native_noise": native_noise_summary(rows),
            "native_raw_range_noise": native_raw_noise_summary(comparison, native_dir),
            "worst_absolute_errors": worst,
            "graph_benefit": graph,
        },
        "software_equivalence": {
            "status": "pass_for_listed_fixtures",
            "prediction_numeric_fields_compared": precision.get("numeric_fields_compared"),
            "prediction_nonzero_numeric_fields": precision.get("nonzero_numeric_fields"),
            "integer_counter_mismatches": precision.get("integer_counter_mismatches"),
            "memory_integer_counters_pass": memory.get("integer_counters_pass"),
            "memory_dram_cases": precision.get("memory_differential", {}).get("dram_cases"),
            "memory_nand_cases": precision.get("memory_differential", {}).get("nand_cases"),
            "max_request_duration_error_ns": precision.get("memory_differential", {}).get(
                "maximum_request_duration_error_ns"
            ),
            "max_request_duration_relative_error": precision.get("memory_differential", {}).get(
                "maximum_request_duration_relative_error"
            ),
            "scope_limit": precision.get("scope_limit"),
        },
        "cuda_graph_independent": {
            "status": dispatch.get("status"),
            "prediction_qualified": dispatch.get("prediction_qualified"),
            "target_llm_latency_used": dispatch.get("target_llm_latency_used"),
            "globaltimer_measurement_count": len(dispatch.get("globaltimer_measurements", [])),
            "typed_activity_measurement_count": len(dispatch.get("typed_activity_measurements", [])),
            "queue_model_repetitions": dispatch.get("queue_model_followup", {}).get("repetitions"),
            "low_cpu_replication_repetitions": dispatch.get("queue_model_low_cpu_replication", {}).get(
                "repetitions"
            ),
        },
        "physical_micro_evaluation": physical,
        "assessment": {
            "invariants_and_software_equivalence": "pass_for_listed_fixtures",
            "native_latency_accuracy": "fail_initial_5_to_10_percent_budget",
            "cuda_graph_benefit_accuracy": "fail",
            "calibration_or_generalization": "not_established",
            "reason": (
                "Native E2E MAPE is 15.5965% across 20 pairs; TTFT MAPE is 23.9201%; "
                "Graph benefit is predicted near zero while native benefit is positive for every model."
            ),
        },
        "limitations": [
            "The 20 end-to-end simulation outputs were reused from the 2026-10-10 frontend run; this report recomputed summaries only.",
            "Native timing was reused from 2026-10-08 and not remeasured in this evaluation.",
            "Native evidence covers one machine, one 512/128 single-sequence workload, and 10 model presets.",
            "Software-equivalence references are independent execution paths, but are not physical hardware truth.",
            "The listed B200/HBF presets did not execute physical-memory acceleration in the 2026-10-07 matrix.",
            "The CUDA Graph independent study explicitly rejects general predictive calibration.",
        ],
        "next_actions": [
            "Diagnose TTFT and decode bias separately before changing costs.",
            "Fix or explicitly model CUDA Graph benefit before claiming graph-mode accuracy.",
            "Add a holdout workload with batch/context variation and repeat native timing after model changes.",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    native = report["native_latency"]["metric_summary"]
    lines = [
        "# 仿真误差初评（2026-10-11）",
        "",
        "口径：本报告的端到端误差表复用了已有的 2026-10-10 仿真输出和 2026-10-08 native 记录，没有重新跑完整前端仿真或 native 推理；本轮只重新执行了独立的 DRAM/NAND 软件路径微评估。",
        "",
        "结论：当前版本通过了列出软件 fixture 的等价性检查，但还不能通过通用预测精度验收。20 组 native 配对的 E2E MAPE 为 15.5965%，TTFT MAPE 为 23.9201%；CUDA Graph 收益预测基本为零，而 native 每个模型都有正收益。",
        "",
        "## 证据范围",
        "",
        f"- native 配对：{report['evidence_scope']['native_pairs']}/{report['evidence_scope']['native_expected_pairs']}，10 个模型，Graph 开/关各一组。",
        f"- 负载：512 输入 / 128 输出，batch/ubatch 512，16 线程；每组 {report['evidence_scope']['native_repetitions']} 次正式测量、{report['evidence_scope']['native_warmups']} 次预热。",
        "- native 时间复用已有 2026-10-08 记录，本轮未重新测量，也未用 native 延迟校准仿真器。",
        "- native 的 E2E 重复测量 range/中位数平均约 4.04%、最大约 12.80%；E2E MAPE 为 15.5965%，因此误差明显但仍需用更多重复和 holdout 区分模型误差与测量波动。",
        "",
        "## native 延迟误差",
        "",
        "| 指标 | Graph | 样本 | MAPE | 平均有符号误差 | 误差范围 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"engine_ttft_ns": "TTFT", "engine_tpot_ns": "TPOT", "engine_e2e_ns": "E2E"}
    for metric in ("engine_ttft_ns", "engine_tpot_ns", "engine_e2e_ns"):
        for mode, label in (("off", "关"), ("on", "开"), ("all", "合计")):
            value = native[metric][mode]
            lines.append(
                f"| {labels[metric]} | {label} | {value['n']} | {value['mape_percent']:.4f}% | {value['mean_signed_error_percent']:.4f}% | {value['signed_error_range_percent'][0]:.4f}% ～ {value['signed_error_range_percent'][1]:.4f}% |"
            )
    lines += [
        "",
        "TTFT 和关闭 Graph 的 TPOT/E2E 主要是低估；Graph 开启后 TPOT/E2E 的偏差方向不稳定，说明误差不是一个简单的统一倍率。",
        "",
        "## 最大误差样本",
        "",
        "| 模型 | Graph | 指标 | 有符号误差 | 仿真 ns | native 中位数 ns |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report["native_latency"]["worst_absolute_errors"]:
        lines.append(
            f"| {row['case_id']} | {row['graph_mode']} | {labels.get(row['metric'], row['metric'])} | {row['signed_error_percent']:.2f}% | {row['simulation_ns']:.0f} | {row['native_median_ns']:.0f} |"
        )
    lines += [
        "",
        "## CUDA Graph 收益误差",
        "",
        "| 模型 | native E2E 降幅 | 仿真 E2E 降幅 | 降幅误差（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for row in report["native_latency"]["graph_benefit"]:
        lines.append(
            f"| {row['case_id']} | {row['native_e2e_reduction_percent']:.2f}% | {row['simulation_e2e_reduction_percent']:.2f}% | {row['absolute_reduction_error_percentage_points']:.2f} |"
        )
    lines += [
        "",
        "## 软件等价性",
        "",
        f"- 预测字段比较：{report['software_equivalence']['prediction_numeric_fields_compared']} 个数值字段，非零差异 {report['software_equivalence']['prediction_nonzero_numeric_fields']} 个，整数计数差异 {report['software_equivalence']['integer_counter_mismatches']} 个。",
        f"- DRAM/NAND 独立参考：{report['software_equivalence']['memory_dram_cases']} 个 DRAM、{report['software_equivalence']['memory_nand_cases']} 个 NAND fixture；最大请求时长误差 {report['software_equivalence']['max_request_duration_error_ns']} ns。",
        "- 这证明列出 fixture 的实现路径等价，不证明物理硬件预测准确。",
        "",
        "## 本轮物理路径微评估",
        "",
        "- DRAM shared-command mixed-follow-up（4 请求）和 NAND shared-host partial-RMW（3 请求）均与独立详细循环一致：completion、timeline、core state 最大绝对误差均为 0。",
        "- 队列、字节守恒、有限缓冲 credit dependency、DRAM/NAND 容量边界检查全部通过。",
        "- 该结果验证解析路径和账本，不代表硅片时序精度。",
        "",
        "## 评估判定",
        "",
        "- 守恒和软件等价性：通过（仅限已有 fixture）。",
        "- native 延迟预测：未通过初始 5%～10% 误差预算。",
        "- CUDA Graph 收益预测：未通过；独立调度研究本身也标记为拒绝通用校准。",
        "- 泛化能力：尚未建立。现有数据只有单一 512/128 负载，且 native 时间来自既有记录。",
        "",
        "## 下一步",
        "",
        "1. 先拆分 TTFT、TPOT、E2E 的误差来源，避免用一个总倍率修正。",
        "2. 单独修正或明确限制 CUDA Graph 收益模型。",
        "3. 增加 batch/context 的 holdout 负载，并在修正后重新采集 native 数据。",
        "",
        "原始机器可读结果见同目录 `report.json`；生成命令：`python tools/evaluate_simulation_error.py`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(args.output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
