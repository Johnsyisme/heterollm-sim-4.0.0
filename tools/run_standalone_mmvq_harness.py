#!/usr/bin/env python3
"""Build/run the pinned llama.cpp GGUF harness and collect MMVQ dispatch evidence.

The llama.cpp binary remains unchanged.  CUDA events and NVTX phase windows are
emitted by the harness; Nsight CUDA activities provide kernel names and launch
geometry.  Activity intervals are kept as evidence and are never promoted to a
simulator rate unless an independent calibration review accepts the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "standalone_mmvq_harness.cu"
RUNTIME = Path(r"F:\codex_project\_runtime_sources\llama.cpp")
DEFAULT_BUILD = RUNTIME / "build-native-5080-sm120"
DEFAULT_VSDEVCMD = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
DEFAULT_NVCC = Path(os.environ.get("CUDA_PATH", r"E:\cuda")) / "bin" / "nvcc.exe"
DEFAULT_NSYS = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe")


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_environment() -> tuple[list[str], list[str], int]:
    """Return measurable activity, unknown graphics contexts, and GPU util."""
    util_text = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], check=False).stdout.strip()
    try:
        utilization = int(util_text.splitlines()[0].strip())
    except (IndexError, ValueError):
        utilization = 100
    pmon = run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], check=False).stdout
    active = []
    unknown = []
    for line in pmon.splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[1].isdigit():
            continue
        sm, mem = fields[3], fields[4]
        description = " ".join(fields[1:])
        if sm == "-" or mem == "-":
            unknown.append(description)
            continue
        try:
            if float(sm) > 0.0 or float(mem) > 0.0:
                active.append(description)
            else:
                unknown.append(description)
        except ValueError:
            unknown.append(description)
    return active, unknown, utilization


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True,
                            encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError("command failed ({}): {}\nstdout:\n{}\nstderr:\n{}".format(
            result.returncode, " ".join(command), result.stdout[-4000:], result.stderr[-4000:]))
    return result


def build_probe(out_dir: Path, nvcc: Path, vsdevcmd: Path, build_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "standalone_mmvq_harness.exe"
    script = out_dir / "build_mmvq_harness.cmd"
    inc = RUNTIME / "include"
    ggml_inc = RUNTIME / "ggml" / "include"
    ggml_src = RUNTIME / "ggml" / "src"
    libs = [
        build_dir / "src" / "llama.lib",
        build_dir / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.lib",
        build_dir / "ggml" / "src" / "ggml-cpu.lib",
        build_dir / "ggml" / "src" / "ggml.lib",
        build_dir / "ggml" / "src" / "ggml-base.lib",
    ]
    missing = [p for p in libs if not p.exists()]
    if missing:
        raise RuntimeError("runtime import libraries missing: " + ", ".join(map(str, missing)))
    cuda_inc = Path(os.environ.get("CUDA_PATH", r"E:\\cuda")) / "include"
    command = ' '.join('"{}"'.format(x) for x in [nvcc, "-O2", "-std=c++17", "-arch=sm_120",
                                                     f'-I{inc}', f'-I{ggml_inc}', f'-I{ggml_src}', f'-I{cuda_inc}', SOURCE, *libs,
                                                     "-o", exe])
    script.write_text(f'@call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul\r\n{command}\r\n', encoding="utf-8")
    try:
        result = run(["cmd.exe", "/d", "/c", str(script)], cwd=out_dir, check=False)
    finally:
        script.unlink(missing_ok=True)
    if result.returncode:
        raise RuntimeError(f"nvcc build failed:\n{result.stdout}\n{result.stderr}")
    return exe


def identity(model: Path, build_dir: Path, exe: Path) -> dict:
    binary_names = ["llama.dll", "ggml-cuda.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"]
    components = []
    for name in binary_names:
        path = build_dir / "bin" / name
        if path.exists():
            components.append({"path": str(path), "sha256": sha256(path), "size": path.stat().st_size})
    harness_component = {"path": "generated_standalone_mmvq_harness.exe (ephemeral)",
                         "sha256": sha256(exe), "size": exe.stat().st_size}
    source_hashes = {}
    source_paths = {
        "ggml-cuda/mmvq.cu": RUNTIME / "ggml" / "src" / "ggml-cuda" / "mmvq.cu",
        "ggml-cuda/mmvq.cuh": RUNTIME / "ggml" / "src" / "ggml-cuda" / "mmvq.cuh",
        "ggml-cuda/vecdotq.cuh": RUNTIME / "ggml" / "src" / "ggml-cuda" / "vecdotq.cuh",
        "ggml-common.h": RUNTIME / "ggml" / "src" / "ggml-common.h",
        "ggml-cuda/common.cuh": RUNTIME / "ggml" / "src" / "ggml-cuda" / "common.cuh",
    }
    for name, path in source_paths.items():
        if path.exists():
            source_hashes[name] = sha256(path)
    commit = run(["git", "rev-parse", "HEAD"], cwd=RUNTIME, check=False).stdout.strip() or None
    gpu = run(["nvidia-smi", "--id=0", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
               "--format=csv,noheader,nounits"], check=False).stdout.strip()
    active, unknown, utilization = gpu_environment()
    return {
        "runtime_source_commit": commit,
        "runtime_components": components,
        "harness_component": harness_component,
        "source_hashes": source_hashes,
        "gguf_sha256": sha256(model),
        "gguf_path": str(model),
            "gpu_query": gpu,
            "gpu_utilization_percent": utilization,
        "environment": {
            "exclusive_gpu": not active and not unknown and utilization <= 30,
            "active_compute_processes": active,
            "unknown_graphics_contexts": unknown,
            "diagnostic_only": True,
        },
    }


def family(name: str) -> tuple[str, str] | None:
    if "quantize_q8_1" in name:
        return "conversion", "q8_1"
    if "mul_mat_vec_q" in name:
        match = re.search(r"ggml_type\)\s*(\d+)", name)
        typ = match.group(1) if match else None
        return "main", {"6": "q5_0", "8": "q8_0"}.get(typ, f"type_{typ or 'unknown'}")
    if "mul_mat_q" in name:
        match = re.search(r"ggml_type\)\s*(\d+)", name)
        typ = match.group(1) if match else None
        return "main", {"6": "q5_0", "8": "q8_0"}.get(typ, f"type_{typ or 'unknown'}")
    return None


def parse_trace(sqlite_path: Path, phases: list[dict]) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    names = {int(k): str(v) for k, v in conn.execute("SELECT id,value FROM StringIds")}
    rows = []
    for phase in phases:
        if phase.get("formal_index", -1) < 0:
            continue
        label = f"mmvq:{phase['phase']}:formal_{phase['formal_index']}"
        window = conn.execute(
            """SELECT n.start,n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
               WHERE COALESCE(n.text,s.value)=? ORDER BY n.start LIMIT 1""", (label,)).fetchone()
        if window is None:
            continue
        start, end = map(int, window)
        activities = conn.execute(
            """SELECT start,end,gridX,gridY,gridZ,blockX,blockY,blockZ,demangledName
               FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start""",
            (start, end)).fetchall()
        grouped = defaultdict(lambda: {"durations": [], "grid": None, "block": None, "name": None})
        for begin, finish, gx, gy, gz, bx, by, bz, name_id in activities:
            kernel_name = names.get(int(name_id), str(name_id))
            item = family(kernel_name)
            if item is None:
                continue
            stage, fmt = item
            grid, block = [int(gx), int(gy), int(gz)], [int(bx), int(by), int(bz)]
            key = (stage, fmt, kernel_name, tuple(grid), tuple(block))
            grouped[key]["durations"].append(int(finish - begin))
            grouped[key]["grid"] = grid
            grouped[key]["block"] = block
            grouped[key]["name"] = kernel_name
        for (stage, fmt, kernel_name, grid, block), item in sorted(grouped.items(), key=lambda x: x[0]):
            rows.append({
                "case_id": f"qwen25_{phase['phase']}_formal_{phase['formal_index']}_{fmt}",
                "phase": phase["phase"],
                "formal_index": phase["formal_index"],
                "stage": stage,
                "weight_format": fmt,
                "shape": {"m": phase["m"], "n": None, "k": None},
                "kernel_name": kernel_name,
                "grid": item["grid"],
                "block": item["block"],
                "dispatch_signature": f"{kernel_name}|grid={','.join(map(str, grid))}|block={','.join(map(str, block))}",
                "device_activity_ns": item["durations"],
                "device_activity_total_ns": sum(item["durations"]),
                "cuda_event_ms": phase.get("cuda_event_ms"),
                "timing_boundary": "cuda_event_phase_envelope+cupti_kernel_activity",
                "fusion_mode": "none",
                "numerical_validation": {
                    "passed": bool(phase.get("numerical_validation_passed")),
                    "logit_checksum": phase.get("logit_checksum"),
                    "logit_absmax": phase.get("logit_absmax"),
                },
                "prediction_qualified": False,
                "source_contract": "same_runtime_binary_and_gguf; geometry_from_cupti; phase_event_is_operator_envelope",
            })
    conn.close()
    return rows


def parse_node_trace(sqlite_path: Path, nodes: list[dict]) -> list[dict]:
    """Join one-op NVTX ranges to the actual CUPTI kernel activity.

    The CUDA backend owns a non-default stream, so the harness event pair is
    an operator boundary.  In a one-node graph the matching MMVQ/MMQ activity
    is the isolated native kernel used for calibration; conversion kernels are
    intentionally excluded.
    """
    conn = sqlite3.connect(sqlite_path)
    names = {int(k): str(v) for k, v in conn.execute("SELECT id,value FROM StringIds")}
    rows = []
    for node in nodes:
        if node.get("formal_index", -1) < 0:
            continue
        shape = node.get("shape") or []
        if not isinstance(shape, list) or len(shape) != 3:
            continue
        label = "mmvq:{}:m={}:formal={}".format(node["case_id"], shape[0], node["formal_index"])
        window = conn.execute(
            """SELECT start,end FROM NVTX_EVENTS WHERE text=? ORDER BY start LIMIT 1""", (label,)).fetchone()
        if window is None:
            window = conn.execute(
                """SELECT n.start,n.end FROM NVTX_EVENTS n JOIN StringIds s ON s.id=n.textId
                   WHERE s.value=? ORDER BY n.start LIMIT 1""", (label,)).fetchone()
        if window is None:
            continue
        start, end = map(int, window)
        activities = conn.execute(
            """SELECT start,end,gridX,gridY,gridZ,blockX,blockY,blockZ,demangledName
               FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start""",
            (start, end)).fetchall()
        main = []
        fixup = []
        for begin, finish, gx, gy, gz, bx, by, bz, name_id in activities:
            kernel_name = names.get(int(name_id), str(name_id))
            if node["kernel_family"] == "mmvq" and "mul_mat_vec_q" not in kernel_name:
                continue
            if node["kernel_family"] == "mmq":
                if "mul_mat_q_stream_k_fixup" in kernel_name:
                    fixup.append((int(finish - begin), [int(gx), int(gy), int(gz)],
                                  [int(bx), int(by), int(bz)], kernel_name))
                    continue
                if "mul_mat_q<" not in kernel_name:
                    continue
            main.append((int(finish - begin), [int(gx), int(gy), int(gz)],
                         [int(bx), int(by), int(bz)], kernel_name))
        if not main:
            continue
        # A standalone graph should have one main kernel. Preserve all main
        # activity durations for diagnostics, while selecting the dominant row
        # for the exact dispatch geometry. MMQ fixup has already been excluded.
        dominant = max(main, key=lambda item: item[0])
        fmt = node["weight_format"]
        type_id = {"q5_0": 6, "q8_0": 8}[fmt]
        m, n, k = map(int, shape)
        if node["kernel_family"] == "mmvq":
            # derive_mmvq_work on the locked CC1200 contract gives this exact
            # key for all four Qwen2.5 M=1 rows.
            rows_per_cta = 4 if m == 1 and k == 896 else 1
            signature = (f"mmvq:type={type_id}:m={m}:fusion=0:small_k=1:"
                         f"halve_iters=0:warps=4:rows={rows_per_cta}")
        else:
            signature = f"mmq:type={type_id}:m={m}:j=unknown:fallback={int(n % 128 != 0)}"
        grid, block, kernel_name = dominant[1], dominant[2], dominant[3]
        rows.append({
            "case_id": node["case_id"], "stage": "main" if node["kernel_family"] == "mmvq" else "control",
            "phase": node["phase"], "formal_index": node["formal_index"],
            "kernel_family": node["kernel_family"], "weight_format": fmt,
            "shape": {"m": m, "n": n, "k": k}, "kernel_name": kernel_name,
            "grid": grid, "block": block, "dispatch_signature": signature,
            "formal_device_ns": [duration for duration, *_ in main],
            "device_activity_ns": [duration for duration, *_ in main],
            "cuda_event_elapsed_ms": node.get("cuda_event_elapsed_ms"),
            "cuda_event_boundary": node.get("cuda_event_boundary"),
            "timing_boundary": "cuda_event_operator_envelope+cupti_isolated_native_node",
            "fusion_mode": "none", "cache_protocol": node.get("cache_protocol", "warm_reuse"),
            "activation_dtype": "fp32",
            "numerical_validation": node.get("numerical_validation", {}),
            "source_geometry_valid": True,
            "diagnostic_only": True,
        })
        # Stream-K fixup is a distinct dispatch stage.  Keep it as an
        # independent diagnostic row so its geometry can be audited without
        # contaminating the main MMQ distribution used by any future surface.
        if fixup:
            fixup_dominant = max(fixup, key=lambda item: item[0])
            fixup_grid, fixup_block, fixup_name = fixup_dominant[1], fixup_dominant[2], fixup_dominant[3]
            rows.append({
                "case_id": node["case_id"], "stage": "fixup",
                "phase": node["phase"], "formal_index": node["formal_index"],
                "kernel_family": "mmq_fixup", "weight_format": fmt,
                "shape": {"m": m, "n": n, "k": k}, "kernel_name": fixup_name,
                "grid": fixup_grid, "block": fixup_block,
                "dispatch_signature": (f"mmq_fixup:type={type_id}:m={m}:"
                                       f"grid={','.join(map(str, fixup_grid))}|"
                                       f"block={','.join(map(str, fixup_block))}"),
                "formal_device_ns": [duration for duration, *_ in fixup],
                "device_activity_ns": [duration for duration, *_ in fixup],
                "cuda_event_elapsed_ms": node.get("cuda_event_elapsed_ms"),
                "cuda_event_boundary": node.get("cuda_event_boundary"),
                "timing_boundary": "cuda_event_operator_envelope+cupti_isolated_native_node",
                "fusion_mode": "none", "cache_protocol": node.get("cache_protocol", "warm_reuse"),
                "activation_dtype": "fp32", "numerical_validation": node.get("numerical_validation", {}),
                "source_geometry_valid": False, "diagnostic_only": True,
                "fixup": True,
            })
    conn.close()
    return rows


def aggregate_node_rows(rows: list[dict], formal_repetitions: int) -> list[dict]:
    """Combine per-repeat CUPTI rows while preserving formal/holdout splits."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("stage") == "fixup":
            continue
        grouped[(row["case_id"], row["stage"], row["kernel_family"])].append(row)
    aggregate = []
    for (case_id, _stage_name, family_name), items in sorted(grouped.items()):
        items.sort(key=lambda item: item.get("formal_index", -1))
        formal_items = items[:formal_repetitions]
        holdout_items = items[formal_repetitions:]
        formal = [duration for item in formal_items for duration in item.get("formal_device_ns", [])]
        holdout = [duration for item in holdout_items for duration in item.get("formal_device_ns", [])]
        if not formal:
            continue
        first = items[0]
        aggregate.append({
            **{key: first[key] for key in (
                "case_id", "stage", "phase", "kernel_family", "weight_format", "shape",
                "grid", "block", "dispatch_signature", "timing_boundary", "fusion_mode",
                "cache_protocol", "activation_dtype", "source_geometry_valid") if key in first},
            "formal_device_ns": formal,
            "holdout_device_ns": holdout,
                "cuda_event_elapsed_ms": first.get("cuda_event_elapsed_ms"),
            "numerical_validation": {
                "passed": all(bool(item.get("numerical_validation", {}).get("passed")) for item in items),
                "count": sum(int(item.get("numerical_validation", {}).get("count", 0)) for item in items),
            },
            "prediction_qualified": False,
            "qualification_reason": "analytical_baseline_not_bound_and_event_is_operator_envelope",
        })
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--nvcc", type=Path, default=DEFAULT_NVCC)
    parser.add_argument("--vsdevcmd", type=Path, default=DEFAULT_VSDEVCMD)
    parser.add_argument("--nsys", type=Path, default=DEFAULT_NSYS)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--holdout-repetitions", type=int, default=3)
    parser.add_argument("--only-case", choices=["q5_0_attn_q", "q5_0_ffn_gate", "q8_0_attn_v", "q8_0_output"])
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or args.holdout_repetitions < 0:
        parser.error("warmups must be >= 0, repetitions must be >= 1, holdout-repetitions must be >= 0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="heterollm-mmvq-") as td:
        td_path = Path(td)
        exe = build_probe(td_path, args.nvcc, args.vsdevcmd, args.runtime_build)
        env = dict(os.environ)
        env["PATH"] = str(args.runtime_build / "bin") + os.pathsep + str(args.nvcc.parent) + os.pathsep + env.get("PATH", "")
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        if args.only_case:
            env["HETEROLLM_MMVQ_CASE"] = args.only_case
        active_before, unknown_before, utilization_before = gpu_environment()
        total_repetitions = args.repetitions + args.holdout_repetitions
        command = [str(exe), str(args.model), str(args.warmups), str(total_repetitions)]
        trace_sqlite = None
        profile_ok = False
        direct_output = td_path / "phase_direct.jsonl"
        profile_output = td_path / "phase_profile.jsonl"
        env["HETEROLLM_MMVQ_OUTPUT"] = str(direct_output)
        # Capture phase receipts without a profiler first. Nsight does not
        # reliably forward child stdout on Windows, while its SQLite export is
        # needed only for kernel geometry.
        phase_result = run(command, cwd=td_path, env=env, check=False)
        result = phase_result
        if not args.no_profile and args.nsys.exists() and phase_result.returncode == 0:
            report = td_path / "mmvq_trace"
            profile_command = [str(args.nsys), "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                               "--cuda-flush-interval=100", "--sample=none", "--force-overwrite=true",
                               "--output", str(report), "--", *command]
            profile_env = dict(env)
            profile_env["HETEROLLM_MMVQ_OUTPUT"] = str(profile_output)
            profile_result = run(profile_command, cwd=td_path, env=profile_env, check=False)
            result = profile_result
            rep = report.with_suffix(".nsys-rep")
            if rep.exists():
                trace_sqlite = td_path / "mmvq_trace.sqlite"
                export = run([str(args.nsys), "export", "--type=sqlite", "--force-overwrite=true",
                              "--output", str(trace_sqlite), str(rep)], cwd=td_path, check=False)
                if export.returncode:
                    trace_sqlite = None
                else:
                    profile_ok = True
        direct_text = direct_output.read_text(encoding="utf-8") if direct_output.exists() else phase_result.stdout
        profile_text = (profile_output.read_text(encoding="utf-8")
                        if profile_ok and profile_output.exists() else direct_text)
        stdout_path = args.output.with_suffix(".phase.jsonl")
        stdout_path.write_text(profile_text, encoding="utf-8")
        phases = []
        for line in profile_text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("schema") in {"heterollm.native-mmvq-phase/v1", "heterollm.native-mmvq-node/v2"}:
                phases.append(value)
        nodes = [value for value in phases if value.get("schema") == "heterollm.native-mmvq-node/v2"]
        retained_sqlite = None
        if trace_sqlite is not None:
            retained_sqlite = args.output.with_suffix(".sqlite")
            shutil.copy2(trace_sqlite, retained_sqlite)
        identity_data = identity(args.model, args.runtime_build, exe)
        cuda_component = next((item for item in identity_data["runtime_components"]
                               if Path(item["path"]).name.casefold() == "ggml-cuda.dll"), None)
        active_after = identity_data["environment"].get("active_compute_processes", [])
        unknown_after = identity_data["environment"].get("unknown_graphics_contexts", [])
        active_processes = sorted(set(active_before + active_after))
        unknown_contexts = sorted(set(unknown_before + unknown_after))
        utilization_after = identity_data["gpu_utilization_percent"]
        environment_clean = (not active_processes and not unknown_contexts
                             and max(utilization_before, utilization_after) <= 30)
        raw_rows = parse_node_trace(trace_sqlite, nodes) if trace_sqlite else []
        aggregate_rows = aggregate_node_rows(raw_rows, args.repetitions)
        receipt = {
            "schema": "heterollm.native-mmvq-standalone/v1",
            "status": "measured" if phase_result.returncode == 0 else "failed",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_id": "nvidia-rtx-5080",
            "runtime_binary_sha256": cuda_component["sha256"] if cuda_component else "",
            "gguf_sha256": identity_data["gguf_sha256"],
            "runtime_components": identity_data["runtime_components"],
            "source_hashes": identity_data["source_hashes"],
            "cache_protocol": "explicit_l2_eviction_attempt" if env_flag("HETEROLLM_MMVQ_L2_FLUSH") else "warm_reuse_with_output_readback",
            "qualification": {
                "exclusive_gpu": environment_clean,
                "active_compute_processes": len(active_processes),
                "unknown_graphics_contexts": len(unknown_contexts),
                "diagnostics_clean": bool(trace_sqlite is not None and not active_processes and not unknown_contexts),
                "gpu_utilization_before_percent": utilization_before,
                "gpu_utilization_after_percent": utilization_after,
                "environment_rejection_reasons": [
                    *(["exclusive_gpu_not_proven"] if active_processes or unknown_contexts
                       or max(utilization_before, utilization_after) > 30 else []),
                    *(["active_compute_processes_present"] if active_processes else []),
                    *(["unknown_graphics_contexts_present"] if unknown_contexts else []),
                ] if not environment_clean else [],
            },
            "configuration": {
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "holdout_repetitions": args.holdout_repetitions,
                "graph_disabled": True,
                "cases": ["qwen25_decode_m1", "qwen25_control_m512"],
                "model_context": 768,
                "batch": 512,
                "ubatch": 512,
                "threads": 16,
            },
            "identity": identity_data,
            "phase_measurements": [value for value in phases if value.get("schema") == "heterollm.native-mmvq-phase/v1"],
            "node_measurements": nodes,
            "rows": aggregate_rows,
            "diagnostic_rows": raw_rows,
            "trace": {"sqlite": str(retained_sqlite) if retained_sqlite else None,
                      "profile_requested": not args.no_profile and args.nsys.exists(),
                      "geometry_available": trace_sqlite is not None,
                      "profile_returncode": result.returncode if result is not phase_result else None},
            "diagnostic_only": True,
            "prediction_qualified": False,
            "limitations": [
                "M=1 is the Qwen2.5 decode envelope; M=512 is a prefill/MMQ control and is not an MMVQ rate surface.",
                "The graph reuses allocated tensors and does not prove L2 eviction; cache_protocol is warm_reuse_with_output_readback unless HETEROLLM_MMVQ_L2_FLUSH=1.",
                "Numerical output is copied back only on the first formal repeat per case; that validation readback is outside the CUDA event and is recorded separately.",
                "CUDA event is an operator phase envelope after device synchronization; per-kernel intervals come from CUPTI activity.",
                "Kernel geometry and dispatch signatures are recorded only when Nsight SQLite export succeeds.",
                "The harness does not modify or recompile the pinned llama.cpp runtime binary.",
                "GPU exclusivity is false unless the receipt environment proves an idle device.",
            ],
        }
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if phase_result.returncode:
            print(phase_result.stderr[-4000:], flush=True)
            return phase_result.returncode
        if result is not phase_result and result.returncode:
            print("Nsight profile failed; phase events were retained:\n" + result.stderr[-4000:], flush=True)
        print(args.output)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
