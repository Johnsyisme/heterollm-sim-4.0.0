
"""Coverage tests for source geometry sharing without timing fitting.

These tests intentionally operate on one GEMM call at a time.  A multi-segment
projection is safe to share only after lowering splits it into its physical
q/k/v or gate/up calls; the test never merges segment timing or installs a rate.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim.cost_models import GemmWorkload
from heterollm_sim.mmvq_work import (
    MMVQSourceContract,
    SOURCE_SHA256,
    UnsupportedMMVQ,
    derive_mmvq_work,
)
from heterollm_sim.mmq_work import MMQ_SOURCE_REVISION
from heterollm_sim import planner


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "gguf_preset_native_validation_2026-10-08" / "base"


def _contract() -> MMVQSourceContract:
    return MMVQSourceContract(
        compute_capability=1200,
        highest_compiled_arch=1200,
        warp_size=32,
        source_hashes=dict(SOURCE_SHA256),
        runtime_binary_sha256="a" * 64,
        ordinary_contiguous_2d=True,
        force_cublas=False,
        mmvq_dispatch_enabled=True,
    )


def _scenario():
    refs = [
        {"path": f"C:/runtime/{path}", "sha256": digest}
        for path, digest in SOURCE_SHA256.items()
    ]
    return SimpleNamespace(
        workload=SimpleNamespace(
            metadata={
                "llama_cpp_f32_q8_1_mmvq": True,
                "llama_cpp_f32_hidden_storage": True,
                "llama_cpp_gpu_native_invocations": {
                    "applied": True,
                    "source_refs": refs,
                },
            }
        )
    )


def _target():
    return SimpleNamespace(
        metadata={
            "llama_cpp_mmvq_source_contract": {
                "compute_capability": 1200,
                "highest_compiled_arch": 1200,
                "warp_size": 32,
                "source_hashes": dict(SOURCE_SHA256),
                "runtime_binary_sha256": "a" * 64,
                "ordinary_contiguous_2d": True,
                "force_cublas": False,
                "mmvq_dispatch_enabled": True,
            }
        }
    )


def _workload(fmt: str, *, m: int = 1, k: int = 896, n: int = 128) -> GemmWorkload:
    return GemmWorkload(
        m=m,
        k=k,
        n=n,
        activation_bits=8,
        weight_bits=8,
        output_bits=32,
        accumulator_bits=32,
        packed_weight_formats=(fmt,),
        activation_storage_bytes=4 * m * k,
        output_storage_bytes=4 * m * n,
    )


def _segment_metadata(fmt: str, k: int, n: int, name: str):
    return {
        "projection_segment_count": 1,
        "projection_segments": [{
            "local_k": k,
            "local_n": n,
            "format": fmt,
            "physical_tensor_name": name,
        }],
    }


def _walk_projection_descriptors(value):
    if isinstance(value, dict):
        descriptor = value.get("weight_projection_descriptors")
        if isinstance(descriptor, dict):
            yield descriptor
        for child in value.values():
            yield from _walk_projection_descriptors(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_projection_descriptors(child)


def test_old_preset_inventory_contains_qwen25_and_qwen3_q5_q8_candidates():
    """The prepared 20-case input has real Q5/Q8 physical candidates.

    Graph off/on doubles the ten source presets, so this test checks inventory
    rather than running the matrix.  It also makes the intended coverage visible
    when a preset descriptor changes.
    """
    found = {}
    for path in sorted(BASE.glob("scenario_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = set()
        for descriptor in _walk_projection_descriptors(data):
            for projection, payload in descriptor.get("projections", {}).items():
                for segment in payload.get("segments", []):
                    if str(segment.get("format", "")).upper() in {"Q5_0", "Q8_0"}:
                        rows.add((projection, segment["format"], segment["k"], segment["n"], segment["segment_id"]))
        if rows:
            found[path.name] = rows

    assert any("qwen2_5_0_5b" in name for name in found)
    assert any("qwen3_1_7b" in name for name in found)
    assert all(k % 32 == 0 for rows in found.values() for _, _, k, _, _ in rows)
    # The graph off/on pair must share the same physical descriptors.
    assert len(found) == 2


def test_qwen_q5_q8_segments_share_the_same_source_geometry_gate(monkeypatch):
    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    scenario, target = _scenario(), _target()
    candidates = (("Q5_0", 896, 896), ("Q5_0", 896, 128), ("Q8_0", 896, 128),
                  ("Q8_0", 2048, 2048), ("Q8_0", 2048, 1024))
    for fmt, k, n in candidates:
        workload = _workload(fmt, k=k, n=n)
        metadata = _segment_metadata(fmt, k, n, f"blk.weight.{fmt}.{k}x{n}")
        assert planner._declared_mmvq_source_limit(
            scenario, workload, target, metadata,
            model_weight_read=True, rhs_is_activation=False,
        ) == 8
        work = derive_mmvq_work(m=1, k=k, n=n, weight_format=fmt, contract=_contract())
        assert work.k == k and work.n == n and work.weight_format == fmt


def test_qkv_multisegment_is_split_then_shared_not_merged(monkeypatch):
    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    scenario, target = _scenario(), _target()
    segments = (("Q5_0", 896, 896, "q"), ("Q5_0", 896, 128, "k"), ("Q8_0", 896, 128, "v"))
    for fmt, k, n, suffix in segments:
        assert planner._declared_mmvq_source_limit(
            scenario, _workload(fmt, k=k, n=n), target,
            _segment_metadata(fmt, k, n, f"blk.0.attn_{suffix}.weight"),
            model_weight_read=True, rhs_is_activation=False,
        ) == 8

    # The current aggregate lowering is intentionally fail-closed.  A future
    # broadening should preserve this boundary by splitting before this gate.
    aggregate = {
        "projection_segment_count": 3,
        "projection_segments": [
            {"local_k": k, "local_n": n, "format": fmt, "physical_tensor_name": f"blk.0.attn_{s}.weight"}
            for fmt, k, n, s in segments
        ],
    }
    assert planner._declared_mmvq_source_limit(
        scenario, _workload("Q5_0", k=896, n=1152), target, aggregate,
        model_weight_read=True, rhs_is_activation=False,
    ) == 0


def test_q_format_policy_is_explicit_and_m512_falls_back(monkeypatch):
    contract = _contract()
    # The source geometry table is reusable for ordinary K-format calls only
    # behind the explicit capability.  The safe Q4_K domain is M<=5.
    for m in (1, 5):
        work = derive_mmvq_work(m=m, k=1024, n=1024, weight_format="Q4_K",
                                contract=contract, allow_k_formats=True)
        assert work.m == m and work.weight_format == "Q4_K"
    with pytest.raises(UnsupportedMMVQ):
        derive_mmvq_work(m=6, k=1024, n=1024, weight_format="Q4_K",
                          contract=contract, allow_k_formats=True)
    with pytest.raises(UnsupportedMMVQ):
        derive_mmvq_work(m=1, k=1024, n=1024, weight_format="Q4_K",
                          contract=contract, allow_k_formats=False)

    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    scenario, target = _scenario(), _target()
    metadata = _segment_metadata("Q8_0", 896, 128, "blk.weight.q8")
    # M=512 is outside the Q8 source dispatch limit and must remain MMQ.
    assert planner._declared_mmvq_source_limit(
        scenario, _workload("Q8_0", m=512, k=896, n=128), target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    ) == 0


def test_blackwell_source_rule_reuses_geometry_without_receipt(monkeypatch):
    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    scenario = SimpleNamespace(
        llama_cpp_config=SimpleNamespace(policy="llama_cpp"),
        model=SimpleNamespace(metadata={}),
        workload=SimpleNamespace(metadata={
            "llama_cpp_kernel_model_preset": "blackwell_analytical_v1",
            "llama_cpp_tensor_storage_contract": {
                "source_revision": MMQ_SOURCE_REVISION,
            },
        }),
    )
    target = SimpleNamespace(kind="gpu", component_id="gpu0", metadata={
        "component_preset_id": "nvidia-rtx-5080",
    })
    metadata = _segment_metadata("Q4_K", 1024, 1024, "blk.0.attn_q.weight")
    workload = replace(_workload("Q4_K", k=1024, n=1024),
                       weight_storage_bytes=1024 * 4 * 144)
    assert planner._declared_mmvq_source_limit(
        scenario, workload, target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    ) == 5
    work, audit = planner._declared_mmvq_work(
        scenario, workload, target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    )
    assert work.__class__.__name__ == "MMVQGeometry"
    assert audit["coverage_kind"] == "source_rule_geometry"
    assert audit["runtime_binary_sha256"] is None


def test_shared_geometry_accepts_explicit_architecture_without_product_name():
    """The source rule is reusable across GPU presets when CC facts are explicit."""
    scenario = SimpleNamespace(
        llama_cpp_config=SimpleNamespace(policy="llama_cpp"),
        model=SimpleNamespace(metadata={}),
        workload=SimpleNamespace(metadata={
            "llama_cpp_kernel_model_preset": "generic_analytical_v1",
            "llama_cpp_f32_hidden_storage": False,
            "llama_cpp_tensor_storage_contract": {
                "source_revision": MMQ_SOURCE_REVISION,
            },
        }),
    )
    target = SimpleNamespace(kind="gpu", component_id="gpu0", metadata={
        "component_preset_id": "local-blackwell-board",
        "compute_capability": 1200,
        "highest_compiled_arch": 1200,
        "warp_size": 32,
    })
    metadata = {
        **_segment_metadata("Q8_0", 896, 128, "blk.0.attn_q.weight"),
        "runtime_input_storage_bits": 32,
        "runtime_output_storage_bits": 32,
    }
    workload = replace(_workload("Q8_0", k=896, n=128),
                       weight_storage_bytes=128 * (896 // 32) * 34)
    geometry, audit = planner._declared_mmvq_work(
        scenario, workload, target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    )
    assert geometry is not None
    assert geometry.__class__.__name__ == "MMVQGeometry"
    assert geometry.compute_capability == 1200
    assert audit["coverage_kind"] == "source_rule_geometry"


def test_multi_segment_descriptor_selects_one_physical_child():
    """A complete q/k/v descriptor never becomes one merged MMVQ weight."""
    scenario = SimpleNamespace(
        llama_cpp_config=SimpleNamespace(policy="llama_cpp"),
        model=SimpleNamespace(metadata={}),
        workload=SimpleNamespace(metadata={
            "llama_cpp_kernel_model_preset": "generic_analytical_v1",
            "llama_cpp_tensor_storage_contract": {
                "source_revision": MMQ_SOURCE_REVISION,
            },
        }),
    )
    target = SimpleNamespace(kind="gpu", component_id="gpu0", metadata={
        "component_preset_id": "local-board",
        "compute_capability": 1200,
        "highest_compiled_arch": 1200,
        "warp_size": 32,
    })
    segments = [
        {"local_k": 896, "local_n": 896, "format": "Q8_0", "physical_tensor_name": "q"},
        {"local_k": 896, "local_n": 128, "format": "Q8_0", "physical_tensor_name": "k"},
        {"local_k": 896, "local_n": 128, "format": "Q8_0", "physical_tensor_name": "v"},
    ]
    metadata = {
        "projection_segment_count": 3,
        "projection_segments": segments,
        "runtime_input_storage_bits": 32,
        "runtime_output_storage_bits": 32,
    }
    workload = replace(_workload("Q8_0", k=896, n=896),
                       weight_storage_bytes=896 * (896 // 32) * 34)
    # q child is unique; the two 128-wide children are deliberately rejected
    # without an explicit child index because matching by shape is ambiguous.
    assert planner._declared_mmvq_source_limit(
        scenario, workload, target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    ) == 8
    assert planner._declared_mmvq_source_limit(
        scenario, _workload("Q8_0", k=896, n=128), target, metadata,
        model_weight_read=True, rhs_is_activation=False,
    ) == 0
