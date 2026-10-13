"""Host KV traffic must survive compaction without becoming CPU compute."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.planner import _record_compute_components
from heterollm_sim.scalable_serving import TaskExecutionRecord


def _record(category, resources, metadata=None):
    return TaskExecutionRecord("probe", 0., 100., category, (), metadata or {},
                               tuple(ResourceDemand(resource, 100.) for resource in resources))


def test_memory_namespace_is_not_a_compute_owner_and_gemm_is_preserved():
    memory = _record(TaskCategory.COMMUNICATION, ["cpu0.memory:dram:data:0"],
                     {"memory_direction": "read", "layer_id": "dense0"})
    assert _record_compute_components(memory, {"cpu0", "gpu0"}) == set()
    gemm = _record(TaskCategory.COMPUTE, ["cpu0.pipeline", "cpu0.memory:dram:data:0"],
                   {"host_gemm_offload_applied": True})
    assert _record_compute_components(gemm, {"cpu0", "gpu0"}) == {"cpu0"}
    gpu = _record(TaskCategory.COMPUTE, ["gpu0.tensor_core", "cpu0.memory:dram:data:0"])
    assert _record_compute_components(gpu, {"cpu0", "gpu0"}) == {"gpu0"}


def test_host_and_cxl_kv_complete_prefill_decode_with_real_dram_traffic():
    import hbf_hbm_serving_study as study
    import hbf_hbm_experiments as experiments

    for architecture in ("E_host_dram", "G_cxl_dram"):
        scenario = study.build_study_scenario(architecture, prompt=128, batch=1, output=8,
                                              request_count=2, interval_ns=50000,
                                              priority_preemption=True)
        row = experiments.serving_row(scenario, "host-kv-regression", placement=architecture)
        assert row["completed_requests"] == 2
        assert row["visible_output_tokens"] == 16
        assert row["phase_service_rates"]["prefill"]["useful_tokens"] == 256
        assert row["phase_service_rates"]["decode"]["useful_tokens"] > 0
        dram = row["resources"]["dram_traffic"]
        assert dram["physical_read_bytes"] > 0 and dram["physical_write_bytes"] > 0
        assert any(key.startswith("cpu0.memory:dram:") and value > 0
                   for key, value in row["resources"]["resource_utilization"].items())
