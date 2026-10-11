"""Run balanced ABBA/BAAB observer pairs and small payload holdouts.

This is a measurement protocol helper.  It writes immutable raw blocks and a
diagnostic summary; it never fits a model or promotes a qualification flag.
The observer-off arm disables both host timestamping and per-enqueue CUDA
error checks, so the comparison measures the probe perturbation rather than a
partially disabled observer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import statistics
import sys

from run_cuda_dispatch_queue_microbench import build_probe, execute_observed, summarize


SCHEMA = "heterollm.cuda-queue-observer-pair/v1"
PAIR_ORDER = {"ABBA": (False, True, True, False), "BAAB": (True, False, False, True)}


def _stats(values: list[float]) -> dict[str, float | int]:
    center = statistics.median(values)
    return {
        "median_ns": center,
        "repeat_mad_ns": statistics.median(abs(value - center) for value in values),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
        "repeat_count": len(values),
    }


def _case(case_id: str, nodes: int, body: int, payload: int, blocks: int = 1, threads: int = 32) -> dict:
    return {
        "case_id": case_id,
        "node_count": nodes,
        "blocks": blocks,
        "threads": threads,
        "extra_parameter_bytes": payload,
        "requested_bodies_ns": [body] * nodes,
    }


def _command(executable: Path, case: dict, observed: bool, repetitions: int) -> list[str]:
    return [
        str(executable),
        "--nodes", str(case["node_count"]),
        "--blocks", str(case["blocks"]),
        "--threads", str(case["threads"]),
        "--payload-bytes", str(case["extra_parameter_bytes"]),
        "--bodies-ns", ",".join(map(str, case["requested_bodies_ns"])),
        "--observe-host", str(int(observed)),
        "--check-errors", str(int(observed)),
        "--repetitions", str(repetitions),
    ]


def _run_block(executable: Path, output_dir: Path, case: dict, order: str,
               block_index: int, arm_index: int, observed: bool,
               repetitions: int) -> dict:
    result, environment = execute_observed(_command(executable, case, observed, repetitions))
    if result.returncode:
        raise RuntimeError(f"queue probe failed for {case['case_id']} {order} arm {arm_index}: {result.stderr}")
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"queue probe returned invalid JSON for {case['case_id']}: {result.stdout[:500]}") from error
    declaration = {**case, "observe_host": observed, "check_errors": observed}
    compact = summarize(raw, declaration)
    raw["measurement_environment"] = environment
    raw_name = f"{case['case_id']}_block_{block_index:02d}_{order}_arm_{arm_index}_{'on' if observed else 'off'}.json.gz"
    raw_path = output_dir / "raw" / raw_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt", encoding="utf-8", compresslevel=9) as stream:
        json.dump(raw, stream)
    values = [float(sample["device_event_ns"]) for sample in raw["samples"]]
    return {
        "block_index": block_index,
        "order": order,
        "arm_index": arm_index,
        "observer": "on" if observed else "off",
        "check_errors": observed,
        "raw_source": str(raw_path.relative_to(output_dir)),
        "_identity": {key: raw[key] for key in ("hardware_id", "architecture", "driver_version", "runtime_version")},
        "device_event": _stats(values),
        "queue_summary": compact,
    }


def _pair_effect(block: list[dict]) -> dict:
    off_values = [entry["device_event"]["median_ns"] for entry in block if entry["observer"] == "off"]
    on_values = [entry["device_event"]["median_ns"] for entry in block if entry["observer"] == "on"]
    off_median = statistics.median(off_values)
    on_median = statistics.median(on_values)
    return {
        "observer_off_block_medians_ns": off_values,
        "observer_on_block_medians_ns": on_values,
        "observer_off_median_ns": off_median,
        "observer_on_median_ns": on_median,
        "on_minus_off_ns": on_median - off_median,
        "absolute_fraction_of_off": abs(on_median - off_median) / off_median if off_median else None,
    }


def run_protocol(args: argparse.Namespace) -> Path:
    if args.balanced_blocks < 2 or args.balanced_blocks % 2:
        raise ValueError("--balanced-blocks must be a positive even number")
    if args.repetitions < 3 or args.holdout_repetitions < 3:
        raise ValueError("at least three repetitions are required")
    output_dir = args.output.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty measurement directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _case("p0_zero_body_payload_0", 16, 0, 0),
        _case("p0_body_1000_payload_0", 16, 1000, 0),
        _case("p0_zero_body_payload_256", 16, 0, 256),
    ]
    holdouts = [
        _case(f"p2_nodes_{nodes}_body_{body}_payload_{payload}", nodes, body, payload)
        for nodes in (33, 127)
        for body in (704, 1376)
        for payload in (160, 544)
    ]
    seed = args.seed
    rng = random.Random(seed)
    block_orders = ["ABBA"] * (args.balanced_blocks // 2) + ["BAAB"] * (args.balanced_blocks // 2)
    rng.shuffle(block_orders)
    started = datetime.now(timezone.utc).isoformat()
    with __import__("tempfile").TemporaryDirectory(prefix="heterollm-queue-paired-") as build_dir:
        executable = build_probe(Path(build_dir), args.nvcc, args.vsdevcmd)
        identity = None
        paired_results = []
        for case in cases:
            blocks = []
            for block_index, order in enumerate(block_orders):
                entries = []
                for arm_index, observed in enumerate(PAIR_ORDER[order]):
                    entry = _run_block(executable, output_dir, case, order, block_index, arm_index, observed, args.repetitions)
                    if identity is None:
                        identity = entry["_identity"]
                    elif identity != entry["_identity"]:
                        raise ValueError("queue probe identity changed during paired measurement")
                    entry.pop("_identity")
                    entries.append(entry)
                blocks.append({"block_index": block_index, "order": order, "arms": entries,
                               "observer_effect": _pair_effect(entries)})
            paired_results.append({"case": case, "blocks": blocks})
        holdout_results = []
        for case in holdouts:
            entry = _run_block(executable, output_dir, case, "observer_off", 0, 0, False, args.holdout_repetitions)
            if identity != entry["_identity"]:
                raise ValueError("queue probe identity changed during payload holdout measurement")
            entry.pop("_identity")
            holdout_results.append({"case": case, "measurement": entry})
    summary = {
        **(identity or {}),
        "schema": SCHEMA,
        "source_kind": "independent_synthetic_queue_observer_protocol",
        "target_llm_latency_used": False,
        "prediction_qualified": False,
        "qualification_result": "not_qualified",
        "qualification_reason": "Observer pairs and payload holdouts are diagnostic evidence; desktop isolation and P0 perturbation gates remain external requirements.",
        "seed": seed,
        "balanced_blocks": args.balanced_blocks,
        "block_order": block_orders,
        "repetitions": args.repetitions,
        "holdout_repetitions": args.holdout_repetitions,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "paired_observer_cases": paired_results,
        "payload_holdout_cases": holdout_results,
    }
    summary_path = output_dir / "paired_measurement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balanced-blocks", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--holdout-repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8173)
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = parser.parse_args()
    if args.nvcc is None:
        args.nvcc = os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe"
    print(f"saved independent paired measurement to {run_protocol(args)}")


if __name__ == "__main__":
    main()
