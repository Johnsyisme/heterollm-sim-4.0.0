"""Finite-window accounting must distinguish offered traffic from final drain."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_sustained_study import summarize_sustained, adjacent_integer_boundary
from heterollm_sim.serving import RequestStatus


def synthetic_result(latencies):
    metrics = {}
    events = []
    for index, latency in enumerate(latencies):
        request_id = str(index)
        arrival = index * 100.
        metrics[request_id] = SimpleNamespace(request_id=request_id, arrival_ns=arrival,
            finish_ns=arrival+latency, ttft_ns=latency-10., status=RequestStatus.FINISHED,
            deadline_met=None)
        for token in (arrival+latency-10., arrival+latency):
            events.append(SimpleNamespace(event_type="tokens_committed", request_id=request_id,
                timestamp_ns=token, details={"visible_tokens": 1}))
    return SimpleNamespace(request_metrics=metrics, events=events)


def test_warmup_exclusion_and_stable_periodic_arrivals():
    measured = summarize_sustained(synthetic_result([500., 10., 20., 30., 40., 50., 60., 70.]),
                                   horizon_end_ns=800., slo_ttft_ns=100.)
    assert measured["warmup_requests"] == 2
    assert measured["offered_requests"] == 6
    # A warmup request completing later belongs in the throughput window but
    # is excluded from the steady-arrival latency sample.
    assert measured["completed_during_offered_window"] == 7
    assert measured["backlog_at_start"] == 1
    assert measured["backlog_at_end"] == 0
    assert measured["backlog_growth_requests"] == -1
    assert measured["qualification_rate"] == 1.
    assert measured["backlog_conservation"]
    assert measured["finite_horizon_sustainable"]


def test_draining_every_request_is_not_evidence_of_sustainable_arrival_rate():
    measured = summarize_sustained(synthetic_result([50., 150., 250., 350., 450., 550., 650., 750.]),
                                   horizon_end_ns=800., slo_ttft_ns=1000., batch=2)
    assert measured["all_requests_eventually_complete"]
    assert measured["qualification_rate"] == 1.
    assert measured["backlog_at_start"] == 1
    assert measured["backlog_at_end"] == 4
    assert measured["backlog_growth_requests"] == 3
    assert measured["completed_during_offered_window"] == 3
    assert measured["completed_rps"] < measured["offered_rps"]
    assert measured["drain_completion_ns"] > measured["window"]["end_ns"]
    assert not measured["finite_horizon_sustainable"]


def test_adjacent_search_runs_both_endpoint_classifications():
    observed = []
    def evaluate(value):
        observed.append(value)
        return value <= 39
    assert adjacent_integer_boundary(evaluate) == (39, 40)
    assert {16, 64, 39, 40}.issubset(observed)
    assert len(observed) <= 8


def test_concurrency_search_requires_a_real_bracket():
    with pytest.raises(ValueError, match="passing low and failing high"):
        adjacent_integer_boundary(lambda _: True)
    with pytest.raises(ValueError, match="passing low and failing high"):
        adjacent_integer_boundary(lambda _: False)
