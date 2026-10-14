"""The 512 GiB Decode control is executed, classified and resumed separately."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts")]
import hbf_hbm_decode_study as study


def rejected_row():
    return {"id": "capacity-control", "mode": "physical_capacity_rejected",
        "placement": "A_hbm_only", "parameters": study.capacity_baseline_parameters(),
        "attempted_serving": True, "executed_serving": False, "node_tps": None,
        "reason": "unable to allocate 60129542144 bytes in capacity 549755813888",
        "fixed_inventory": {"hbm0": {"capacity_bytes": study.CAPACITY_BASELINE_HBM_BYTES}},
        "capacity_failure": {"exception_type": "AllocationError",
            "classification": "physical_allocator_no_contiguous_extent",
            "requested_allocation_bytes": 60129542144,
            "capacity_bytes": study.CAPACITY_BASELINE_HBM_BYTES,
            "failure_stage": "physical_planning_preview", "physical_owner": "hbm0.hbm_fabric",
            "occupied_bytes_at_failure": 460000000000,
            "free_bytes_at_failure": 89755813888,
            "largest_free_extent_bytes": 42000000000}}


def test_control_executes_original_parameters_and_preserves_fragmentation(monkeypatch):
    calls = []
    source = rejected_row()
    def run(architecture, **kwargs):
        calls.append((architecture, kwargs))
        return deepcopy(source)
    monkeypatch.setattr(study, "run_case", run)
    result = study.run_capacity_baseline()
    assert len(calls) == 1
    architecture, params = calls[0]
    assert architecture == "A_hbm_only"
    assert "hbm_capacity_bytes" not in params
    assert {k: v for k, v in params.items() if k != "row_id"} == study.capacity_baseline_parameters()
    assert result["status"] == "passed"
    assert result["output_tokens_per_request"] == 64
    assert result["capacity_failure"] == source["capacity_failure"]
    assert result["failed_allocation_bytes"] == 60129542144
    assert result["node_tps"] is result["decode_tps"] is None
    assert all(study.capacity_baseline_checks(result).values())


@pytest.mark.parametrize("mode", ["configuration_rejected", "full_operator_serving"])
def test_other_outcomes_are_not_relabelled_as_capacity_rejection(monkeypatch, mode):
    source = rejected_row()
    source.update(mode=mode, executed_serving=mode == "full_operator_serving")
    source.pop("capacity_failure")
    source["node_tps"] = 1. if mode == "full_operator_serving" else None
    monkeypatch.setattr(study, "run_case", lambda *args, **kwargs: deepcopy(source))
    result = study.run_capacity_baseline()
    assert result["mode"] == mode
    assert result["status"] == "failed"
    assert result["failed_allocation_bytes"] is None
    def raises(*args, **kwargs):
        raise ValueError("unclassified implementation failure")
    monkeypatch.setattr(study, "run_case", raises)
    with pytest.raises(ValueError, match="unclassified"):
        study.run_capacity_baseline()


def test_baseline_resume_requires_current_parameters_and_full_failure_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "run_case", lambda *args, **kwargs: rejected_row())
    fresh = study.run_capacity_baseline()
    path = tmp_path / study.CAPACITY_BASELINE_FILENAME
    path.write_text(json.dumps(fresh), encoding="utf-8")
    calls = []
    monkeypatch.setattr(study, "run_capacity_baseline", lambda: calls.append(True) or deepcopy(fresh))
    assert study.load_or_run_capacity_baseline(point_dir=tmp_path, resume=True) == fresh
    assert not calls
    for mutate in (
        lambda row: row.update(memory_execution_contract="historical"),
        lambda row: row["parameters"].update(output=4),
        lambda row: row.pop("capacity_failure"),
    ):
        stale = deepcopy(fresh)
        mutate(stale)
        path.write_text(json.dumps(stale), encoding="utf-8")
        result = study.load_or_run_capacity_baseline(point_dir=tmp_path, resume=True)
        assert result == fresh
    assert len(calls) == 3


def test_resume_full_matrix_adds_control_without_rerunning_sixteen_positive_points(monkeypatch, tmp_path):
    for model in study.MODELS:
        for context in study.CONTEXTS:
            for arch in study.ARCHITECTURES:
                point = (model, context, arch, 64)
                row = {"memory_execution_contract": study.EXECUTION_CONTRACT,
                    "core_execution_audit": {"executed_physical_configurations": {"hbm0": {"kind": "HBM"}}},
                    "parameters": study.point_parameters(point)}
                (tmp_path / ("-".join(map(str, point)) + ".json")).write_text(json.dumps(row), encoding="utf-8")
    class NoSubmissions:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, *args): raise AssertionError("A cached positive point was rerun")
    monkeypatch.setattr(study, "ProcessPoolExecutor", NoSubmissions)
    monkeypatch.setattr(study, "core_checks", lambda row: {"verified": True})
    monkeypatch.setattr(study, "summarize_decode", lambda rows, **kwargs: {
        "rows": rows, "checks": {"positive_count": len(rows) == 16}, "status": "passed",
        "capability": {"supported": ["real decode"]}})
    calls = []
    def run(*args, **kwargs):
        calls.append(kwargs)
        return rejected_row()
    monkeypatch.setattr(study, "run_case", run)
    result = study.run_decode_study(point_dir=tmp_path, resume=True)
    assert len(result["rows"]) == 16
    assert len(result["capacity_baseline_observations"]) == len(calls) == 1
    assert result["status"] == "passed"
    assert (tmp_path / study.CAPACITY_BASELINE_FILENAME).exists()
