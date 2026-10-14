"""Background jobs own the configuration used by execution and lazy replay."""
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.reporting import OnlineScenarioResult
from heterollm_sim.run_jobs import RunJobManager


def test_background_run_and_replay_are_isolated_from_later_caller_edits():
    source = scenario(prompt=8, count=1, output=2)
    source.workload.metadata["job_probe"] = {"value": "submitted"}
    physical = source.hardware.get_component("hbm0").metadata["physical_memory_config"]
    original_latency = physical["read_latency_ns"]
    released = Event()
    seen = []

    def prepare(owned):
        assert owned is not source
        assert owned.workload.metadata is not source.workload.metadata
        owned.workload.metadata["normalized_by_job"] = True
        return owned

    def estimate(owned):
        assert owned.workload.metadata["normalized_by_job"] is True
        return {"recommended_retention_policy": "aggregate"}

    def execute(owned, **kwargs):
        seen.append(owned)
        return OnlineScenarioResult(owned, SimpleNamespace(), None, (), "aggregate")

    def replay(result, batch_id):
        return {"batch_id": batch_id,
                "probe": result.scenario.workload.metadata["job_probe"]["value"],
                "read_latency_ns": result.scenario.hardware.get_component("hbm0")
                    .metadata["physical_memory_config"]["read_latency_ns"]}

    manager = RunJobManager(max_workers=1)
    manager._executor.submit(released.wait)
    try:
        with patch("heterollm_sim.llama_scenario.prepare_llama_scenario", side_effect=prepare), \
             patch("heterollm_sim.run_jobs.estimate_scenario", side_effect=estimate), \
             patch("heterollm_sim.run_jobs.run_scenario", side_effect=execute), \
             patch("heterollm_sim.run_jobs.report_dict", return_value={"summary": {}}), \
             patch("heterollm_sim.run_jobs.replay_online_batch_trace", side_effect=replay), \
             patch("heterollm_sim.run_jobs.page_online_batch_trace", side_effect=lambda value, **kwargs: value):
            job_id = manager.submit(source)
            future = manager._jobs[job_id].future
            source.workload.metadata["job_probe"]["value"] = "edited while queued"
            physical["read_latency_ns"] = original_latency + 1000.
            released.set()
            future.result(timeout=10)
            assert manager.get(job_id)["status"] == "completed"
            assert seen[0].workload.metadata["job_probe"]["value"] == "submitted"
            assert "normalized_by_job" not in source.workload.metadata
            source.workload.metadata["job_probe"]["value"] = "edited after completion"
            physical["read_latency_ns"] = original_latency + 2000.
            page = manager.trace_page(job_id, "batch-1")
            assert page["probe"] == "submitted"
            assert page["read_latency_ns"] == original_latency
    finally:
        released.set()
        manager.shutdown(wait=True, cancel_pending=True)
