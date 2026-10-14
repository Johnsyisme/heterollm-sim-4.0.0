"""Small cross-contract cost regressions; no native calibration claims."""
from dataclasses import replace

import pytest

from heterollm_sim.cost_models import (
    CacheHierarchyProfile, CacheLevelProfile, FusedAttentionWorkload,
    GPUProfile, HBMProfile, TensorCoreProfile, GemmWorkload, estimate_gpu_fused_attention,
    estimate_gpu_gemm,
)
from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile, attention_mma_work
from heterollm_sim.mmq_work import derive_mmq_work
from heterollm_sim.mmvq_work import MMVQSourceContract, SOURCE_SHA256, derive_mmvq_work
from heterollm_sim.mmvq_issue_bound import MMVQIssueContract, source_issue_contract


def device(*, native=True, tensor_core=None):
    capability = KernelCapability("attention", ("fp16",), "fp16", "prefill", "tensor",
        "analytical declaration", internal_dtype="fp16", operator="attention")
    return GPUProfile(tensor_core or TensorCoreProfile(1, 1, 1.),
        CacheHierarchyProfile((CacheLevelProfile("cache", 1 << 20, 64, 0, 1000.),)),
        scalar_lanes_per_sm=32, scalar_ops_per_cycle=1, reduction_ops_per_cycle_per_sm=1,
        special_function_units_per_sm=1, special_function_ops_per_cycle=1,
        kernel_model=KernelModelProfile("device", "runtime", "architecture", kernels=(capability,)) if native else None)


@pytest.mark.parametrize("head_dim", [16, 24, 32])
@pytest.mark.parametrize("shape", [(16, 16), (1, 17), (17, 33)])
def test_native_attention_counts_independent_head_mma_padding(head_dim, shape):
    work = FusedAttentionWorkload(*shape, hidden_size=8 * head_dim, score_heads=8,
        kv_hidden_size=2 * head_dim, execution_phase="prefill")
    native = estimate_gpu_fused_attention(device(), HBMProfile(1000.), work)
    basic = estimate_gpu_fused_attention(device(native=False), HBMProfile(1000.), work)
    demand = next(d for d in native.phases[-1].demands if d.resource_id == "gpu.tensor_core")
    expected = basic.metadata["issued_tensor_operations"]
    assert demand.work_units == expected
    assert native.metadata["kernel_model"]["issued_dot_operations"] == expected
    assert native.metadata["kernel_model"]["read_bytes"] == work.read_bytes
    assert native.metadata["kernel_model"]["write_bytes"] == work.write_bytes


def test_attention_causal_tiles_cannot_pack_mma_tails_across_heads_or_key_tiles():
    work = FusedAttentionWorkload(5, 9, 48, score_heads=2, kv_hidden_size=24,
        execution_phase="prefill", query_tile_tokens=3, key_tile_tokens=5,
        causal_query_positions=(0, 1, 2, 7, 8))
    gpu = device()
    gpu = replace(gpu, kernel_model=replace(gpu.kernel_model, kernels=(
        replace(gpu.kernel_model.kernels[0], attention_query_tile=3, attention_kv_tile=5),)))
    result = estimate_gpu_fused_attention(gpu, HBMProfile(1000.), work)
    count = result.metadata["kernel_model"]["attention_mma_work"]
    # First Q tile executes one K tile; second executes two. Each step needs
    # two QK and two PV MMA per head for the 24-wide head, even on short tails.
    assert count["qk_mma_count"] == count["pv_mma_count"] == 3 * 2 * 2
    assert count["issued_tensor_operations"] == 24 * 8192


def test_causal_decode_uses_native_one_query_tile_not_workload_prefill_default():
    gpu = device()
    gpu = replace(gpu, kernel_model=replace(gpu.kernel_model, kernels=(
        replace(gpu.kernel_model.kernels[0], phase="decode", attention_query_tile=1,
                attention_kv_tile=64),)))
    work = FusedAttentionWorkload(1, 64, 64, score_heads=1, execution_phase="decode",
        causal_query_positions=(63,))
    result = estimate_gpu_fused_attention(gpu, HBMProfile(1000.), work)
    matching = estimate_gpu_fused_attention(gpu, HBMProfile(1000.),
        replace(work, query_tile_tokens=1))
    audit = result.metadata["kernel_model"]
    assert audit["attention_tile_work"]["executed_pairs"] == 64
    assert audit["attention_tile_work"]["tile_source"] == "kernel_descriptor"
    # Tensor instructions still pad the single query to the physical MMA M.
    assert audit["issued_dot_operations"] == 262144
    assert result.phases == matching.phases


def test_attention_stream_tiles_use_distinct_mma_output_and_reduction_widths():
    tensor = TensorCoreProfile(1, 1, 1., mma_m=8, mma_n=16, mma_k=4)
    work = FusedAttentionWorkload(17, 65, 256, score_heads=4,
        query_tile_tokens=16, key_tile_tokens=64)
    count = attention_mma_work(work, tensor, round_attention_tiles=True)
    assert count["qk_mma_count"] == 4 * 2 * 2 * 2 * 4 * 16
    assert count["pv_mma_count"] == 4 * 2 * 2 * 4 * 2 * 16
    assert count["issued_tensor_operations"] == (
        count["qk_mma_count"] + count["pv_mma_count"]) * tensor.operations_per_mma


def source_gemm(kind):
    if kind == "mmq":
        source = derive_mmq_work(m=17, k=256, n=129, weight_format="Q4_K", sm_count=1,
                                 shared_memory_per_block=65536)
        activation_bytes = source.consumer_unique_bytes
    else:
        source = derive_mmvq_work(m=1, k=256, n=128, weight_format="Q4_K",
            contract=MMVQSourceContract(1200, 1200, 32, SOURCE_SHA256, "1" * 64, True, False, True),
            allow_k_formats=True)
        activation_bytes = source.consumer_q8_1_unique_bytes
    return GemmWorkload(source.m, source.k, source.n, activation_bits=32, weight_bits=4,
        output_bits=32, activation_dtype="fp32", packed_weight_formats=("q4_k",),
        execution_phase="prefill" if kind == "mmq" else "decode",
        weight_storage_bytes=source.logical_weight_bytes, activation_storage_bytes=activation_bytes,
        **{kind + "_work": source})


@pytest.mark.parametrize("kind", ["mmq", "mmvq"])
@pytest.mark.parametrize("field", ["epilogue_name", "epilogue_operations",
                                   "epilogue_transcendental_operations", "epilogue_output_elements"])
def test_fixed_source_gemm_rejects_unproven_fusion(kind, field):
    work = source_gemm(kind)
    with pytest.raises(ValueError, match="source work is unfused"):
        replace(work, **{field: "activation" if field == "epilogue_name" else 1})


def test_native_profile_does_not_silently_discard_explicit_mmvq_issue_contract():
    work = source_gemm("mmvq")
    work = replace(work, mmvq_issue_contract=MMVQIssueContract.from_mapping(
        source_issue_contract(runtime_binary_sha256="1" * 64, sm_count=1)))
    basic = device(native=False)
    native = replace(basic, kernel_model=KernelModelProfile("device", "runtime", "architecture",
        kernels=(KernelCapability("mmvq", ("q4_k",), "fp32", "decode", "dp4a",
                                  "analytical declaration", output_bits=32),)))
    assert estimate_gpu_gemm(basic, HBMProfile(1000.), work).metadata["source_geometry_priced"]
    with pytest.raises(ValueError, match="separate MMVQ issue bound"):
        estimate_gpu_gemm(native, HBMProfile(1000.), work)
    unmatched = replace(native, kernel_model=replace(native.kernel_model, kernels=(
        replace(native.kernel_model.kernels[0], phase="prefill"),)))
    with pytest.raises(ValueError, match="has no GEMM dispatch"):
        estimate_gpu_gemm(unmatched, HBMProfile(1000.), work)


def test_declared_gemm_epilogue_remains_priced_without_fixed_unfused_source():
    gpu = replace(device(),
        kernel_model=KernelModelProfile("device", "runtime", "architecture", kernels=(
            KernelCapability("gemm", ("fp16",), "fp16", "prefill", "tensor",
                "analytical declaration", internal_dtype="fp16", epilogue="activation"),)))
    work = GemmWorkload(16, 16, 16, activation_bits=16, weight_bits=16, execution_phase="prefill",
        epilogue_operations=11, epilogue_transcendental_operations=7, epilogue_name="activation")
    result = estimate_gpu_gemm(gpu, HBMProfile(1000.), work)
    demands = {d.resource_id: d for d in result.phases[-1].demands}
