"""Host/device task spans may overlap; neither is a wall-time residual."""
from types import SimpleNamespace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.live_execution_metrics import LiveCohortMetrics, execution_time_breakdown


def task(name, start, duration, *, host=False, resource=None):
    return TaskSpec(name, "request", name, TaskCategory.POLICY if host else TaskCategory.COMPUTE,
        earliest_start_ns=start, demands=(ResourceDemand(resource or name, duration),),
        metadata={"orchestration_stage": "host_prefix"} if host else {})


@pytest.mark.parametrize("tasks,expected", [
    # Concurrent hosts do 20 ns of work in 10 ns; they do not erase the GPU.
    ((task("h1", 0, 10, host=True), task("h2", 0, 10, host=True), task("gpu", 0, 10)),
     (10., 10., 20., 10., 0., 10.)),
    # A suffix overlaps a device tail; both complete inside a 20 ns window.
    ((task("host", 5, 10, host=True), task("gpu", 0, 20)),
     (10., 20., 10., 10., 0., 20.)),
    # The gap is idle, not inferred device execution.
    ((task("host", 0, 3, host=True), task("gpu", 6, 4)),
     (3., 4., 3., 0., 3., 10.)),
])
def test_live_task_time_unions_preserve_parallelism_and_idle(tasks, expected):
    kernel = UnifiedEventKernel.from_closed_graph(tasks)
    live = LiveCohortMetrics()
    for _ in tasks:
        live.observe(kernel.step())
    report = live.metadata()
    keys = ("host_orchestration_ns", "device_execution_ns", "host_task_time_ns",
            "host_device_overlap_ns", "execution_idle_ns", "execution_stage_makespan_ns")
    assert tuple(report[key] for key in keys) == expected
    assert report["execution_time_semantics"] == "overlapping_task_interval_unions"
    assert expected[0] + expected[1] - expected[3] + expected[4] == expected[5]


def test_breakdown_merges_each_side_and_preserves_explicit_window():
    records = [SimpleNamespace(start_ns=a, end_ns=b, metadata={"orchestration_stage": stage})
               for a, b, stage in ((12, 18, "host_prefix"), (15, 22, "host_suffix"),
                                    (16, 20, "operator"), (19, 24, "operator"))]
    report = execution_time_breakdown(records, origin_ns=10, end_ns=26)
    assert report["host_orchestration_ns"] == 10
    assert report["host_task_time_ns"] == 13
    assert report["device_execution_ns"] == 8
    assert report["host_device_overlap_ns"] == 6
    assert report["execution_idle_ns"] == 4
    assert execution_time_breakdown((), origin_ns=5, end_ns=8)["execution_idle_ns"] == 3
