#!/usr/bin/env python3
"""Summarize repeated native node traces without promoting timings to a model.

The Nsight request ranges are the only aggregation boundary used here.  CUDA
graph node ordinals are joined to a prior capture only by ordinal, family and
grid/block geometry; that join is structural evidence, never a persistent
pointer identity or a timing calibration.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import median


def family(name: str) -> str:
    if "rms_norm" in name:
        return "rms_norm"
    if "quantize_q8_1" in name:
        return "quantize_q8_1"
    if "mul_mat_vec_q" in name:
        match = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_vec_q:type_{match.group(1)}" if match else "mul_mat_vec_q"
    if "mul_mat_q_stream" in name:
        match = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_q_stream:type_{match.group(1)}" if match else "mul_mat_q_stream"
    if "mul_mat_q<" in name:
        match = re.search(r"ggml_type\)\s*(\d+)", name)
        return f"mul_mat_q:type_{match.group(1)}" if match else "mul_mat_q"
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


def robust_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"repeat_count": 0}
    center = median(values)
    return {
        "repeat_count": len(values),
        "median": center,
        "mad": median(abs(value - center) for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def prior_structure(path: Path | None) -> dict[tuple[int, tuple[int, ...], tuple[str, ...]], dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in data.get("on", {}).get("nodes", []):
        key = (int(row["ordinal"]), tuple(row.get("grid_block") or ()), tuple(sorted(row.get("families") or ())))
        owner = row.get("owner")
        if owner:
            result[key] = owner
    return result


def windows(conn: sqlite3.Connection) -> list[tuple[str, int, int]]:
    rows = conn.execute(
        """SELECT s.value,n.start,n.end FROM NVTX_EVENTS n
           JOIN StringIds s ON s.id=n.textId
           WHERE s.value LIKE 'formal_%' AND n.end IS NOT NULL
           ORDER BY n.start"""
    ).fetchall()
    return [(str(label), int(start), int(end)) for label, start, end in rows]


def summarize(path: Path, mode: str, prior: dict) -> dict:
    conn = sqlite3.connect(path)
    names = {int(key): value for key, value in conn.execute("SELECT id,value FROM StringIds")}
    formal = []
    for label, start, end in windows(conn):
        kernels = conn.execute(
            """SELECT start,end,graphNodeId,gridX,gridY,gridZ,blockX,blockY,blockZ,demangledName
               FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start""",
            (start, end),
        ).fetchall()
        by_family = defaultdict(lambda: {"count": 0, "duration_ns": 0})
        by_node = defaultdict(lambda: {"count": 0, "duration_ns": 0, "families": set(), "grid_block": None})
        for begin, finish, node_id, gx, gy, gz, bx, by, bz, name_id in kernels:
            name = names.get(int(name_id), str(name_id))
            fam = family(name)
            item = by_family[fam]
            item["count"] += 1
            item["duration_ns"] += int(finish - begin)
            if mode == "on" and node_id is not None:
                ordinal = int(node_id) & 0xFFFFFFFF
                node = by_node[ordinal]
                node["count"] += 1
                node["duration_ns"] += int(finish - begin)
                node["families"].add(fam)
                node["grid_block"] = [gx, gy, gz, bx, by, bz]
        row = {
            "label": label,
            "window_ns": [start, end],
            "kernel_activity_count": len(kernels),
            "kernel_duration_sum_ns": sum(int(finish - begin) for begin, finish, *_ in kernels),
            "kernel_families": {
                fam: {"count": value["count"], "duration_ns": value["duration_ns"]}
                for fam, value in sorted(by_family.items())
            },
        }
        if mode == "on":
            graph_events = conn.execute(
                "SELECT start,end,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS WHERE start>=? AND end<=? ORDER BY start",
                (start, end),
            ).fetchall()
            ordinal_order = {}
            for _, _, node_id in graph_events:
                ordinal_order.setdefault(int(node_id) & 0xFFFFFFFF, len(ordinal_order))
            nodes = []
            for node_id, ordinal in sorted(ordinal_order.items(), key=lambda item: item[1]):
                item = by_node.get(node_id, {"count": 0, "duration_ns": 0, "families": set(), "grid_block": None})
                key = (ordinal, tuple(item["grid_block"] or ()), tuple(sorted(item["families"])))
                owner = prior.get(key)
                nodes.append({
                    "ordinal": ordinal,
                    "graph_node_id_low32": node_id,
                    "kernel_count": item["count"],
                    "duration_ns": item["duration_ns"],
                    "families": sorted(item["families"]),
                    "grid_block": item["grid_block"],
                    "structural_owner": owner,
                    "ownership_evidence": "cross_capture_ordinal_geometry" if owner else None,
                    "timing_qualified": False,
                    "physical_ledger_qualified": False,
                })
            row.update({
                "graph_node_event_count": len(graph_events),
                "unique_graph_nodes": len(ordinal_order),
                "kernel_associated_graph_nodes": sum(node["kernel_count"] > 0 for node in nodes),
                "nodes": nodes,
            })
        formal.append(row)
    conn.close()
    family_summary = {}
    families = sorted({fam for request in formal for fam in request["kernel_families"]})
    for fam in families:
        family_summary[fam] = {
            "kernel_count": robust_stats([request["kernel_families"].get(fam, {}).get("count", 0) for request in formal]),
            "duration_ns": robust_stats([request["kernel_families"].get(fam, {}).get("duration_ns", 0) for request in formal]),
        }
    result = {
        "mode": mode,
        "sqlite": str(path),
        "formal_request_count": len(formal),
        "formal_requests": formal,
        "family_summary": family_summary,
    }
    if mode == "on":
        result["node_summary"] = {
            "unique_node_count": robust_stats([request["unique_graph_nodes"] for request in formal]),
            "graph_event_count": robust_stats([request["graph_node_event_count"] for request in formal]),
            "kernel_associated_node_count": robust_stats([request["kernel_associated_graph_nodes"] for request in formal]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = prior_structure(args.prior)
    result = {
        "schema": "heterollm.native-node-repeat-evidence/v1",
        "diagnostic_only": True,
        "target_llm_latency_used_for_calibration": False,
        "off": summarize(args.off, "off", {}),
        "on": summarize(args.on, "on", prior),
        "limitations": [
            "Graph-off has no node identity; only formal-window kernel-family aggregates are reported.",
            "Graph-on node ownership uses low32 ordinal plus grid/block/family geometry; it is capture-structural and not a persistent pointer join.",
            "Kernel activity duration is not E2E and does not expose DRAM/HBM traffic, cache misses, occupancy, power, CPU scheduling or allocator counters.",
            "No family duration is promoted to a simulator rate surface or graph-device replay envelope.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
