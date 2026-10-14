"""Parallel tensor shapes and causal boundaries remain valid after lowering."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.ir import (
    LayerSpec, LinearAttentionSpec, LinkSpec, ModelSpec, ParallelSpec,
    RankMappingSpec, build_model_graph_from_layer_specs,
)
from heterollm_sim.planner import compile_scenario, _layer_gemm


def configured_scenario(*, mixer="full_attention", moe=False, fusion=False, dtype="fp16", tp=2,
                        request_count=1, output_tokens=1):
    base = scenario(prompt=1, count=request_count, output=output_tokens)
    layer = LayerSpec("dense0", "moe" if moe else "dense", 256, 513, 4, kv_heads=2,
        dtype=dtype, quantization="w8a8" if dtype == "int8" else None,
        num_experts=2 if moe else 1, shared_expert_intermediate_size=257 if moe else 0,
        sequence_mixer=mixer,
        linear_attention=LinearAttentionSpec(2, 4, 32, 64) if mixer == "linear_attention" else None)
    model = ModelSpec(name="audit-tp", graph=build_model_graph_from_layer_specs(
        "audit-tp", (layer,), vocabulary_size=1024, max_sequence_length=256))
    hardware = base.hardware
    if tp == 2:
        gpu = next(c for c in hardware.components if c.component_id == "gpu0")
        hardware = replace(hardware,
            components=(*hardware.components, replace(gpu, component_id="gpu1", die_id="audit_gpu_die")),
            links=(*(replace(link, source_component="gpu1") if link.link_id == "gpu-hbm1" else link
                     for link in hardware.links),
                LinkSpec("audit-peer", "gpu0", "ucie0", "gpu1", "ucie0", "UCIe", "1.1", 64,
                         2048., 10., payload="streaming")))
    parallel = ParallelSpec(tp_degree=tp, rank_mapping=tuple(
        RankMappingSpec(rank, "gpu" + str(rank), rank, 0, 0, "hbm" + str(rank)) for rank in range(tp)))
    configured = replace(base, hardware=hardware, model=model,
        placement=replace(base.placement, model_name=model.name, parallel=parallel),
        fusion_policy=replace(base.fusion_policy, gemm_epilogue_activation=fusion, residual_norm=fusion),
        workload=replace(base.workload, scheduler=replace(base.workload.scheduler, mode="static")))
    return plan_runtime_placement(configured).apply(configured)


def ancestors(task, by_id):
    pending, found = list(task.dependencies), set()
    while pending:
        identity = pending.pop()
        if identity not in found:
            found.add(identity)
            pending.extend(by_id[identity].dependencies)
    return found


@pytest.mark.parametrize("mixer", ["full_attention", "linear_attention"])
@pytest.mark.parametrize("fusion", [False, True])
def test_tp_residual_adds_replicated_input_after_partial_outputs_are_reduced(mixer, fusion):
    tasks = compile_scenario(configured_scenario(mixer=mixer, fusion=fusion)).tasks
    by_id = {task.task_id: task for task in tasks}
    for role, collective_name in (("mlp", ".mlp_all_reduce"),
                                  ("attention" if mixer == "full_attention" else "linear_attention",
                                   ".attention_all_reduce" if mixer == "full_attention" else ".linear_all_reduce")):
        residuals = [task for task in tasks if task.metadata.get("phase") == "gpu_elementwise"
                     and (task.metadata.get("event_kind") == role + "_residual"
                          or role == "linear_attention" and task.metadata.get("linear_op") == "residual")]
        if fusion and role == "attention":
            # This residual is fused into the following replicated norm.
            residuals = [task for task in tasks if task.metadata.get("event_kind") == "fused_residual_norm"
                         and task.metadata.get("phase") != "kernel_launch"]
        assert len(residuals) == 2
        for residual in residuals:
            preceding = ancestors(residual, by_id)
            assert any(collective_name in by_id[identity].name for identity in preceding)
            if not (fusion and role == "attention"):
                assert residual.metadata["cost_model"]["operations"] == 256
        for collective in (task for task in tasks if collective_name in task.name):
            assert not any(residual.task_id in ancestors(collective, by_id) for residual in residuals)


def test_tp_input_and_final_norm_consume_the_full_replicated_hidden_vector():
    tasks = compile_scenario(configured_scenario()).tasks
    for event in ("input_norm_reduce", "final_norm_reduce"):
        norms = [task for task in tasks if task.metadata.get("event_kind") == event
                 and task.metadata.get("phase") == "gpu_reduction"]
        assert len(norms) == 2
        assert {task.metadata["cost_model"]["read_bytes"] for task in norms} == {256 * 2}


@pytest.mark.parametrize("moe", [False, True])
def test_padded_gate_and_up_have_one_local_element_for_each_down_input(moe):
    tasks = compile_scenario(configured_scenario(moe=moe)).tasks
    projection = "shared_expert.up_gate" if moe else "mlp.up_gate"
    up = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
          and task.metadata.get("projection_id") == projection]
    half = (257 + 1) // 2 if moe else (513 + 1) // 2
    assert len(up) == 2
    assert {task.metadata["cost_model"]["weight_bytes"] for task in up} == {256 * 2 * half * 2}


@pytest.mark.parametrize("dtype", ["fp16", "int8"])
def test_analytical_gemm_storage_matches_the_unfused_activation_consumer(dtype):
    tasks = compile_scenario(configured_scenario(dtype=dtype, tp=1)).tasks
    up = next(task for task in tasks if task.metadata.get("phase") == "gpu_gemm"
              and task.metadata.get("projection_id") == "mlp.up_gate")
    activation = next(task for task in tasks if task.metadata.get("phase") == "gpu_elementwise"
                      and task.metadata.get("event_kind") == "mlp_activation")
    assert up.metadata["cost_model"]["output_bytes"] == activation.metadata["cost_model"]["read_bytes"]


def test_source_bound_f32_storage_keeps_its_explicit_output_contract():
    layer = LayerSpec("dense0", "dense", 256, 512, 4, dtype="int8", quantization="w8a8")
    workload = _layer_gemm(layer, 1, 256, 512, name="source", f32_storage=True)
    assert workload.output_bits == workload.accumulator_bits == 32
    assert workload.activation_storage_bytes == 4 * 256


@pytest.mark.parametrize("mixer", ["full_attention", "linear_attention"])
def test_compiled_segment_reuse_preserves_parallel_dependencies_and_physical_traffic(mixer):
    configured = configured_scenario(mixer=mixer, request_count=2, output_tokens=2)
    cached = compile_scenario(configured)
    with patch("heterollm_sim.planner._task_segment_cache", return_value=None):
        fresh = compile_scenario(configured)
    def semantics(schedule):
        return [(task.task_id, task.request_id, task.dependencies, task.demands,
                 task.metadata.get("event_kind"), task.metadata.get("linear_op"),
                 task.metadata.get("memory_accesses")) for task in schedule.tasks]
    assert semantics(cached) == semantics(fresh)
