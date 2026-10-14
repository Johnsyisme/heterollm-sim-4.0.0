"""Report conservation across batches, out-of-band physical I/O and missing data."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.reporting import (
    OnlineScenarioResult, _online_report_core, _sum_batch_dram_traffic, _sum_batch_storage_traffic,
    _online_activity_samples,
)
from heterollm_sim.streaming_des import _merge_dram_traffic, _merge_storage_traffic
from test_async_kv_prefetch import runtime
from test_backend_execution_semantics import manifest


@pytest.mark.parametrize("medium, field", [("dram", "refresh_wait_ns"), ("nand", "pages_read")])
@pytest.mark.parametrize("value", [None, 3])
def test_partial_source_coverage_cannot_become_recorded_or_not_recorded(medium, field, value):
    source = {"task_count": 2, field: value,
              "metric_availability": {field: {"status": "partial"}}}
    key = "dram_traffic" if medium == "dram" else "storage_traffic"
    result = SimpleNamespace(serving=SimpleNamespace(batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={key: source}))]))
    summarize = _sum_batch_dram_traffic if medium == "dram" else _sum_batch_storage_traffic
    merge = _merge_dram_traffic if medium == "dram" else _merge_storage_traffic
    streamed = {}
    merge(streamed, source)
    for report in (summarize(result), streamed):
        assert report[field] is None
        assert report["metric_availability"][field]["status"] == "partial"


@pytest.mark.parametrize("medium, field", [("dram", "refresh_wait_ns"), ("nand", "pages_read")])
def test_missing_value_overrides_a_stale_recorded_label(medium, field):
    ledgers = [
        {"task_count": 1, "metric_availability": {field: {"status": "recorded"}}},
        {"task_count": 1, field: 0, "metric_availability": {field: {"status": "recorded"}}},
    ]
    key = "dram_traffic" if medium == "dram" else "storage_traffic"
    result = SimpleNamespace(serving=SimpleNamespace(batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={key: row})) for row in ledgers]))
    summarize = _sum_batch_dram_traffic if medium == "dram" else _sum_batch_storage_traffic
    merge = _merge_dram_traffic if medium == "dram" else _merge_storage_traffic
    streamed = {}
    for row in ledgers:
        merge(streamed, row)
    for report in (summarize(result), streamed):
        assert report[field] is None
        assert report["metric_availability"][field]["status"] == "partial"


def test_online_report_includes_executed_paged_io_once_without_inventing_batches_or_time():
    runner = runtime()
    serving = runner.run()
    empty_metrics = {**serving.runtime_kernel_metrics, "paged_kv_transfers": {}}
    original = OnlineScenarioResult(runner.plan.scenario,
        replace(serving, runtime_kernel_metrics=empty_metrics), manifest(), (), "aggregate")
    before = _online_report_core(original)
    # A separate physical executor's ledger is additive; it is not another
    # task schedule to run, nor another serving batch to extend the wall time.
    ledger = {"schema_version": "heterollm.paged-kv-physical-cost/v1",
        "resource_accounted_bytes": 2**53 + 1,
        "resource_busy_ns": {"migration-wire": 40.}, "phase_count": 3,
        "dram_traffic": {"task_count": 1, "logical_read_bytes": 64,
            "physical_bytes": 128, "resource_totals": {"dram-wire": {
                "owner": "hbm0", "bytes_moved": 128, "service_ns": 4.}}},
        "storage_traffic": {"task_count": 1, "logical_write_bytes": 64,
            "physical_write_bytes": 16384, "resource_totals": {"nand-wire": {
                "owner": "hbf0", "bytes_moved": 64, "service_ns": 8.}}}}
    changed = replace(original, serving=replace(serving, runtime_kernel_metrics={
        **empty_metrics, "paged_kv_transfers": {"cost_ledger": ledger}}))
    after = _online_report_core(changed)
    assert "total_energy_pj" not in after.summary
    assert after.summary["resource_accounted_bytes"] - before.summary["resource_accounted_bytes"] == 2**53 + 1
    assert after.resource_utilization["migration-wire"] == 40. / before.summary["makespan_ns"]
    assert after.summary["dram_traffic"]["physical_bytes"] == 128
    assert after.summary["storage_traffic"]["physical_write_bytes"] == 16384
    assert "power" not in after.summary.get("simulated_subtargets", {})
    for key in ("batch_count", "task_count", "makespan_ns", "throughput"):
        assert after.summary[key] == before.summary[key]
    assert _online_report_core(changed).summary == after.summary


@pytest.mark.parametrize("distance", [0, 17])
def test_real_paged_transfer_reports_core_and_link_cost_in_sync_and_prefetch_modes(distance):
    runner = runtime(distance=distance)
    serving = runner.run()
    ledger = serving.runtime_kernel_metrics["paged_kv_transfers"]["cost_ledger"]
    assert ledger["phase_count"] > 0
    result = OnlineScenarioResult(runner.plan.scenario, serving, manifest(), (), "aggregate")
    report = _online_report_core(result)
    assert "total_energy_pj" not in report.summary
    assert report.summary["resource_accounted_bytes"] == (
        sum(b.cost.metadata.get("resource_accounted_bytes", 0) for b in serving.batches)
        + sum(sum(p["resource_bytes"].values()) for p in ledger["phases"]))
    assert "energy_pj" not in ledger
    for key in ("dram_traffic", "storage_traffic"):
        assert report.summary[key]["physical_bytes"] >= ledger[key]["physical_bytes"] > 0
    components, links, _ = _online_activity_samples(result)
    assert any(components["hbm0"].values())
    assert any(links.values())
    # Neither detailed event retention nor repeated report generation bills I/O.
    with_no_details = replace(result, serving=replace(serving, runtime_kernel_metrics={
        **serving.runtime_kernel_metrics, "paged_kv_transfers": {
            **serving.runtime_kernel_metrics["paged_kv_transfers"], "events": ()}}))
    assert _online_report_core(with_no_details).summary == report.summary
