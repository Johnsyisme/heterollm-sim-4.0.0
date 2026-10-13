"""Physical executed throughput, page traffic and bounded transport checks."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from hbf_hbm_bandwidth_study import run_read_benchmark,CHUNK_BYTES


def test_effective_bandwidth_uses_real_link_completion_and_respects_bottleneck():
    fast=run_read_benchmark(payload_bytes=16*CHUNK_BYTES)
    capped=run_read_benchmark(payload_bytes=16*CHUNK_BYTES,link_gb_s=1600.)
    for row in (fast,capped):
        observed=row["observed"]
        assert observed["link_bytes"]==observed["physical_read_bytes"]==16*CHUNK_BYTES
        assert observed["resource_bytes"]["study:nand-to-sink"]==16*CHUNK_BYTES
        assert observed["duration_ns"]==max(c["link_end_ns"] for c in row["chunk_timeline"])
        assert observed["effective_end_to_end_read_gb_s"]==16*CHUNK_BYTES/observed["duration_ns"]
        assert observed["effective_end_to_end_read_gb_s"]<row["configured"]["path_bandwidth_upper_bound_gb_s"]
    assert capped["observed"]["duration_ns"]>fast["observed"]["duration_ns"]


def test_page_size_changes_actual_mapping_page_count_without_changing_payload():
    small=run_read_benchmark(payload_bytes=4*CHUNK_BYTES,page_bytes=4096)
    large=run_read_benchmark(payload_bytes=4*CHUNK_BYTES,page_bytes=CHUNK_BYTES)
    for row in (small,large):
        assert row["observed"]["physical_read_bytes"]==4*CHUNK_BYTES
        assert row["observed"]["pages_read"]*row["configured"]["page_bytes"]==4*CHUNK_BYTES
        assert row["configured"]["nand_config"]["page_bytes"]==row["configured"]["page_bytes"]
    assert small["observed"]["pages_read"]==1536
    assert large["observed"]["pages_read"]==4


def test_finite_fifo_backpressure_is_applied_to_actual_nand_arrival():
    row=run_read_benchmark(payload_bytes=5*CHUNK_BYTES,buffer_slots=1)
    chunks=row["chunk_timeline"]
    assert row["observed"]["transport_buffer_peak_slots"]==1
    for previous,current in zip(chunks,chunks[1:]):
        assert current["arrival_ns"]==previous["link_end_ns"]
        assert current["nand_completion_ns"]>current["arrival_ns"]


@pytest.mark.parametrize("values",[{"port_gb_s":float("nan")},{"buffer_slots":True},
    {"link_latency_ns":-1},{"page_bytes":10000},{"payload_bytes":CHUNK_BYTES+1}])
def test_invalid_or_unaligned_inputs_are_rejected(values):
    with pytest.raises(ValueError):
        run_read_benchmark(**values)
