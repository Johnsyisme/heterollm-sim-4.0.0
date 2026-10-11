from types import SimpleNamespace

from heterollm_sim.cost_models import GemmWorkload
from heterollm_sim.mmvq_work import SOURCE_SHA256
from heterollm_sim import planner


def _scenario():
    return SimpleNamespace(
        workload=SimpleNamespace(metadata={
            "llama_cpp_f32_q8_1_mmvq": True,
            "llama_cpp_f32_hidden_storage": True,
            "llama_cpp_gpu_native_invocations": {
                "applied": True,
                "source_refs": [
                    {"path": f"C:/runtime/{path}", "sha256": digest}
                    for path, digest in SOURCE_SHA256.items()
                ],
            },
        })
    )


def _target():
    return SimpleNamespace(metadata={
        "llama_cpp_mmvq_source_contract": {
            "compute_capability": 1200,
            "highest_compiled_arch": 1200,
            "warp_size": 32,
            "source_hashes": dict(SOURCE_SHA256),
            "runtime_binary_sha256": "a" * 64,
            "ordinary_contiguous_2d": True,
            "force_cublas": False,
            "mmvq_dispatch_enabled": True,
        },
    })


def _workload(fmt="q8_0"):
    return GemmWorkload(
        m=1, k=896, n=128,
        activation_bits=8, weight_bits=8, output_bits=32,
        accumulator_bits=32, packed_weight_formats=(fmt,),
        activation_storage_bytes=4 * 896,
        weight_storage_bytes=128 * 28 * (34 if fmt == "q8_0" else 22),
        output_storage_bytes=4 * 128,
    )


def _metadata():
    return {
        "projection_segment_count": 1,
        "projection_segments": [{
            "local_k": 896, "local_n": 128, "format": "Q8_0",
            "physical_tensor_name": "blk.0.attn_v.weight",
        }],
    }


def test_exact_mmvq_geometry_is_independent_of_mmq_pricing(monkeypatch):
    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    assert planner._declared_mmvq_source_limit(
        _scenario(), _workload(), _target(), _metadata(),
        model_weight_read=True, rhs_is_activation=False,
    ) == 8


def test_geometry_gate_falls_back_for_non_applicable_or_uncovered_calls(monkeypatch):
    monkeypatch.setattr(planner, "_f32_hidden_storage_enabled", lambda scenario: True)
    scenario = _scenario()
    target = _target()
    assert planner._declared_mmvq_source_limit(
        scenario, _workload(), target, _metadata(),
        model_weight_read=False, rhs_is_activation=False,
    ) == 0
    assert planner._declared_mmvq_source_limit(
        scenario, _workload("q4_k"), target, _metadata(),
        model_weight_read=True, rhs_is_activation=False,
    ) == 0
    bad_metadata = {**_metadata(), "projection_segment_count": 2}
    assert planner._declared_mmvq_source_limit(
        scenario, _workload(), target, bad_metadata,
        model_weight_read=True, rhs_is_activation=False,
    ) == 0
    scenario.workload.metadata["llama_cpp_gpu_native_invocations"]["applied"] = False
    assert planner._declared_mmvq_source_limit(
        scenario, _workload(), target, _metadata(),
        model_weight_read=True, rhs_is_activation=False,
    ) == 0
