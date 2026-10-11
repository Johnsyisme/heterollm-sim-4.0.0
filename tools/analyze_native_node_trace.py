#!/usr/bin/env python3
"""Extract formal-request CUDA node/kernel evidence from Nsight Systems SQLite."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


def formal_window(conn):
    row = conn.execute(
        """SELECT n.start, n.end FROM NVTX_EVENTS n
           JOIN StringIds s ON s.id=n.textId
           WHERE s.value='formal_1' AND n.end IS NOT NULL
           ORDER BY n.start DESC LIMIT 1"""
    ).fetchone()
    if not row:
        raise RuntimeError("formal_1 NVTX range not found")
    return int(row[0]), int(row[1])


def kernel_family(name: str) -> str:
    if "rms_norm" in name:
        return "rms_norm"
    if "quantize_q8_1" in name:
        return "quantize_q8_1"
    if "mul_mat_vec_q" in name:
        m = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_vec_q:type_{m.group(1)}" if m else "mul_mat_vec_q"
    if "mul_mat_q_stream" in name:
        m = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_q_stream:type_{m.group(1)}" if m else "mul_mat_q_stream"
    if "mul_mat_q<" in name:
        m = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_q:type_{m.group(1)}" if m else "mul_mat_q"
    if "mul_mat_vec_f" in name:
        return "mul_mat_vec_f"
    if "mul_mat_f" in name:
        return "mul_mat_f"
    if "soft_max" in name:
        return "softmax"
    if "rope_neox" in name:
        return "rope"
    if "k_set_rows" in name:
        return "set_rows"
    if "cpy_" in name:
        return "copy"
    if "k_bin_bcast" in name:
        return "elementwise_bcast"
    return name.split("<", 1)[0].strip()


def load_ownership(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for phase_key, dispatches in data.items():
        ordinal = {}
        for d in dispatches:
            owner = {
                "dispatch_index": d.get("dispatch_index"),
                "first_ggml_index": d.get("first_ggml_index"),
                "last_ggml_index": d.get("last_ggml_index"),
                "source": [],
            }
            for source in d.get("source_nodes", []):
                output = source.get("output", {})
                owner["source"].append({
                    "name": output.get("name"),
                    "op": output.get("op"),
                    "type": output.get("type"),
                    "ne": output.get("ne"),
                })
            start = d.get("first_node_ordinal", 0)
            for n in range(start, start + d.get("node_count", 0)):
                ordinal[n] = owner
        result[phase_key] = ordinal
    return result


def extract(sqlite_path: Path, mode: str, ownership_path: Path | None):
    conn = sqlite3.connect(sqlite_path)
    start, end = formal_window(conn)
    names = {int(k): v for k, v in conn.execute("SELECT id,value FROM StringIds")}
    kernels = conn.execute(
        """SELECT start,end,graphNodeId,gridX,gridY,gridZ,blockX,blockY,blockZ,demangledName
           FROM CUPTI_ACTIVITY_KIND_KERNEL
           WHERE start>=? AND end<=? ORDER BY start""", (start, end)
    ).fetchall()
    families = defaultdict(lambda: {"kernel_count": 0, "duration_ns": 0, "names": set()})
    for s, e, node_id, gx, gy, gz, bx, by, bz, name_id in kernels:
        name = names.get(int(name_id), str(name_id))
        f = families[kernel_family(name)]
        f["kernel_count"] += 1
        f["duration_ns"] += int(e - s)
        f["names"].add(name)
    fam_out = []
    for family, value in sorted(families.items(), key=lambda kv: kv[1]["duration_ns"], reverse=True):
        fam_out.append({
            "family": family,
            "kernel_count": value["kernel_count"],
            "duration_ns": value["duration_ns"],
            "names": sorted(value["names"])[:4],
        })
    result = {
        "mode": mode,
        "sqlite": str(sqlite_path),
        "formal_window_ns": [start, end],
        "kernel_activity_count": len(kernels),
        "kernel_duration_sum_ns": sum(int(e - s) for s, e, *_ in kernels),
        "kernel_families": fam_out,
    }
    if mode == "on":
        events = conn.execute(
            """SELECT start,end,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS
               WHERE start>=? AND end<=? ORDER BY start""", (start, end)
        ).fetchall()
        node_first = {}
        for s, e, node_id in events:
            # Nsight stores graph-node handles in different typed ID domains
            # for graph events and kernel activities. The low 32-bit ordinal
            # is stable within this capture and is the only safe join key.
            node_first.setdefault(int(node_id) & 0xFFFFFFFF, len(node_first))
        kernels_by_node = defaultdict(lambda: {
            "count": 0, "duration_ns": 0, "families": set(), "grid": None,
        })
        for s, e, node_id, gx, gy, gz, bx, by, bz, name_id in kernels:
            if node_id is None:
                continue
            x = kernels_by_node[int(node_id) & 0xFFFFFFFF]
            x["count"] += 1
            x["duration_ns"] += int(e - s)
            x["families"].add(kernel_family(names.get(int(name_id), str(name_id))))
            x["grid"] = [gx, gy, gz, bx, by, bz]
        owners = load_ownership(ownership_path) if ownership_path else {}
        decode_owners = owners.get("measured:0:decode:1", {})
        node_rows = []
        for node_id, ordinal in sorted(node_first.items(), key=lambda kv: kv[1]):
            x = kernels_by_node.get(node_id, {
                "count": 0, "duration_ns": 0, "families": set(), "grid": None,
            })
            node_rows.append({
                "ordinal": ordinal,
                "graph_node_id_low32": node_id,
                "kernel_count": x["count"],
                "duration_ns": x["duration_ns"],
                "families": sorted(x["families"]),
                "grid_block": x["grid"],
                "owner": decode_owners.get(ordinal),
            })
        result["graph_node_event_count"] = len(events)
        result["unique_graph_nodes"] = len(node_first)
        result["nodes"] = node_rows
        grouped = defaultdict(lambda: {"node_count": 0, "kernel_count": 0, "duration_ns": 0})
        for row in node_rows:
            owner = row["owner"]
            key = "unmatched"
            if owner and owner["source"]:
                key = ";".join(sorted({
                    (s.get("name") or s.get("op") or "unknown") for s in owner["source"]
                }))
            g = grouped[key]
            g["node_count"] += 1
            g["kernel_count"] += row["kernel_count"]
            g["duration_ns"] += row["duration_ns"]
        result["owner_groups"] = sorted(
            ({"owner": k, **v} for k, v in grouped.items()),
            key=lambda x: x["duration_ns"], reverse=True,
        )
    conn.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--on", type=Path, required=True)
    ap.add_argument("--ownership", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {
        "schema": "heterollm.native-node-evidence/v1",
        "off": extract(args.off, "off", None),
        "on": extract(args.on, "on", args.ownership),
        "limitations": [
            "off has no CUDA graph node IDs; kernel families are phase-window aggregates only.",
            "on graphNodeId is joined to source structure by first-seen ordinal and geometry, not by persistent pointer.",
            "kernel activity duration overlaps and includes profiler-visible GPU activity; it is not E2E and is not a DRAM path measurement.",
            "source ownership kernel_predictions are unvalidated analytical predictions, so they cannot calibrate residuals.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
