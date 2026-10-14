"""Executed sequential NAND reads and a bounded, pipelined physical link.

All speeds use actual core/link completion timestamps. No GPU TPS or hardware
calibration is inferred from this controlled transport microbenchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"scripts")]

from heterollm_sim.memory_transfer import ResourceTimeline, TransferResource
from heterollm_sim.memory_types import AccessRequest, Operation, make_hbf_config
from heterollm_sim.nand_core import NandCore

MIB = 1024**2
CHUNK_BYTES = 1_572_864
PAYLOAD_BYTES = 384*MIB
PAGE_BYTES = (4096,16384,65536,262144,1_572_864)
READ_LATENCIES_NS = (1000.,4000.,10000.,25000.)


def run_read_benchmark(*, port_gb_s=3000., link_gb_s=3000., page_bytes=16384,
                       page_read_ns=4000., payload_bytes=PAYLOAD_BYTES,
                       chunk_bytes=CHUNK_BYTES, buffer_slots=64, link_latency_ns=100.):
    for name,value in (("port_gb_s",port_gb_s),("link_gb_s",link_gb_s)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(name+" must be finite and positive")
    for name,value in (("page_read_ns",page_read_ns),("link_latency_ns",link_latency_ns)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError(name+" must be finite and nonnegative")
    for name,value in (("page_bytes",page_bytes),("payload_bytes",payload_bytes),
                       ("chunk_bytes",chunk_bytes),("buffer_slots",buffer_slots)):
        if isinstance(value,bool) or not isinstance(value,int) or value<=0:
            raise ValueError(name+" must be a positive integer")
    if chunk_bytes%page_bytes or payload_bytes%chunk_bytes:
        raise ValueError("fixed workload must have whole physical pages and whole transfer chunks")
    # Explicitly generous hypothetical geometry; parallel_units alone cannot
    # create physical units absent from the address mapping.
    config = make_hbf_config(channels=64,targets_per_channel=4,dies_per_target=2,
        luns_per_die=2,planes_per_lun=2,planes_independent=True,parallel_units=2048,
        blocks_per_plane=max(1,math.ceil(payload_bytes/(2048*1024*page_bytes))),
        pages_per_block=1024,page_bytes=page_bytes,host_granularity_bytes=1,
        capacity_bytes=payload_bytes,host_bandwidth_gb_s=port_gb_s,
        host_read_bandwidth_gb_s=port_gb_s,host_write_bandwidth_gb_s=port_gb_s,
        internal_bandwidth_gb_s=12000.,page_read_ns=page_read_ns,
        page_program_ns=75000.,front_ns=0.,max_outstanding_requests=buffer_slots,
        max_expanded_segments=1)
    timeline = ResourceTimeline()
    # True selects the actual per-page execution loop; limiting retained stage
    # detail does not change reservations. No bulk acceleration is used here.
    core = NandCore(config,timeline,capture_details=True)
    link = TransferResource("study:nand-to-sink",link_gb_s)
    chunks, completions = [], []
    totals = {key:0 for key in ("logical_bytes","host_transfer_bytes","physical_read_bytes",
                               "pages_read","internal_transfer_bytes")}
    for i,address in enumerate(range(0,payload_bytes,chunk_bytes)):
        # Reserve a finite transport slot until its link transfer completes.
        # Backpressure changes the next NAND request's real arrival time.
        arrival = completions[i-buffer_slots] if i>=buffer_slots else 0.
        result = core.submit(AccessRequest(str(i),Operation.READ,address,chunk_bytes,arrival))
        transfer = timeline.transfer(link.resource_id,chunk_bytes,
            result.completion_ns+link_latency_ns,link.bandwidth_gb_s,direction="read")
        completions.append(transfer.end_ns)
        chunks.append({"index":i,"address":address,"bytes":chunk_bytes,"arrival_ns":arrival,
            "nand_completion_ns":result.completion_ns,"link_start_ns":transfer.start_ns,
            "link_end_ns":transfer.end_ns})
        totals["logical_bytes"] += result.logical_bytes
        for key in totals:
            if key!="logical_bytes":
                totals[key] += int(result.counters[key])
    duration = max(completions)
    link_start = min(r["link_start_ns"] for r in chunks)
    link_window = duration-link_start
    changes = {}
    for row in chunks:
        changes[row["arrival_ns"]] = changes.get(row["arrival_ns"],0)+1
        changes[row["link_end_ns"]] = changes.get(row["link_end_ns"],0)-1
    population,peak = 0,0
    for _timestamp,delta in sorted(changes.items()):
        population += delta
        peak=max(peak,population)
    ceiling = min(port_gb_s,link_gb_s)
    return {"id":f"read-port{port_gb_s:g}-link{link_gb_s:g}-page{page_bytes}-lat{page_read_ns:g}",
        "mode":"physical_sequential_read_and_link_microbenchmark", "node_tps":None,
        "assumption_status":"explicit_uncalibrated_internal_sensitivity",
        "configured":{"port_gb_s":port_gb_s,"link_gb_s":link_gb_s,
            "path_bandwidth_upper_bound_gb_s":ceiling,"page_bytes":page_bytes,
            "page_read_ns":page_read_ns,"payload_bytes":payload_bytes,"chunk_bytes":chunk_bytes,
            "physical_array_units":config.array_units,"effective_execution_slots":config.parallel_units,
            "buffer_slots":buffer_slots,"transport_buffer_capacity_bytes":buffer_slots*chunk_bytes,
            "link_latency_ns":link_latency_ns,"nand_config":asdict(config)},
        "observed":{**totals,"link_bytes":timeline.bytes_moved[link.resource_id],
            "duration_ns":duration,"effective_end_to_end_read_gb_s":payload_bytes/duration,
            "effective_over_path_upper_bound":payload_bytes/duration/ceiling,
            "link_stream_duration_ns":link_window,"link_stream_effective_gb_s":payload_bytes/link_window,
            "link_busy_ns":timeline.busy_ns[link.resource_id],
            "link_busy_fraction_of_stream_window":timeline.busy_ns[link.resource_id]/link_window,
            "transport_buffer_peak_slots":peak,"transport_buffer_peak_bytes":peak*chunk_bytes,
            "resource_bytes":dict(timeline.bytes_moved),"resource_busy_ns":dict(timeline.busy_ns)},
        "chunk_timeline":chunks,
        "measurement_scope":"primary rate is total completed logical read bytes divided by time from arrival zero to final actual link completion, including startup, media latency, finite buffer backpressure and link latency. Link-stream rate starts at first link transmission and is a separate denominator.",
        "hardware_scope":"Hypothetical 2048 physical array units and 2048 effective slots, explicit 12000 GB/s per internal channel, configured host port and separate transport link. This is sufficient-parallelism sensitivity, not a claim of realizable HBF geometry, pins or area.",
        "buffer_scope":"Finite transport FIFO slots gate requests until link completion. The FIFO is capacity/backpressure state only; no additional DRAM read/write bandwidth is modeled.",
        "physical_page_scope":"page_bytes controls actual NAND mapping, full-page read latency and read bytes; chunk_bytes is independently the link transport unit. 1.5 MiB physical pages are an attachment-requested hypothetical sensitivity, not vendor NAND specifications."}


def run_bandwidth_study():
    target_rows=[]
    for target in (400.,1600.,3000.):
        row=run_read_benchmark(port_gb_s=target,link_gb_s=target)
        row.update(section="effective_bandwidth_targets",target_effective_gb_s=target)
        target_rows.append(row)
    capped=run_read_benchmark(port_gb_s=3000.,link_gb_s=1600.)
    capped.update(section="link_bottleneck_negative_control",target_effective_gb_s=3000.)
    pages=[]
    for page in PAGE_BYTES:
        row=run_read_benchmark(page_bytes=page)
        row["section"]="physical_page_fixed_payload"
        pages.append(row)
    latencies=[]
    for latency in READ_LATENCIES_NS:
        row=run_read_benchmark(page_read_ns=latency)
        row["section"]="physical_read_latency"
        latencies.append(row)
    rows=target_rows+[capped]+pages+latencies
    checks={
        "actual_read_and_link_bytes_conserved":all(r["observed"]["logical_bytes"]==r["observed"]["host_transfer_bytes"]==r["observed"]["physical_read_bytes"]==r["observed"]["link_bytes"]==PAYLOAD_BYTES for r in rows),
        "media_page_count_exact":all(r["observed"]["pages_read"]*r["configured"]["page_bytes"]==PAYLOAD_BYTES for r in rows),
        "near_target_means_at_least_90_percent_end_to_end":all(.9<=r["observed"]["effective_end_to_end_read_gb_s"]/r["target_effective_gb_s"]<=1. for r in target_rows),
        "target_ports_do_not_bypass_1600_link":capped["observed"]["effective_end_to_end_read_gb_s"]<1600. and capped["observed"]["effective_end_to_end_read_gb_s"]<target_rows[-1]["observed"]["effective_end_to_end_read_gb_s"],
        "same_payload_and_chunk_across_physical_pages":len({(r["configured"]["payload_bytes"],r["configured"]["chunk_bytes"]) for r in pages})==1,
        "page_scan_includes_hypothetical_1_5_mib":{r["configured"]["page_bytes"] for r in pages}==set(PAGE_BYTES),
        "latency_scan_1_4_10_25_us":{r["configured"]["page_read_ns"] for r in latencies}==set(READ_LATENCIES_NS),
        "latency_changes_actual_completion":latencies[-1]["observed"]["duration_ns"]>latencies[0]["observed"]["duration_ns"],
        "finite_transport_buffer_bound":all(r["observed"]["transport_buffer_peak_slots"]<=r["configured"]["buffer_slots"] for r in rows),
        "effective_rate_respects_path_ceiling":all(r["observed"]["effective_over_path_upper_bound"]<=1.+1e-12 for r in rows),
    }
    return {"id":"18_bandwidth", "title":"真实读取和链路完成时间下的有效带宽、物理页与延迟扫描",
        "status":"passed" if all(checks.values()) else "failed", "checks":checks,
        "capability":{"supported":all(checks.values()),"modeled":True,"calibrated":False,
            "scope":"real per-page NandCore execution plus finite-buffer pipelined ResourceTimeline link transfers; transport microbenchmark only"},
        "rows":rows,"summary":{"target_effective_gb_s":[400.,1600.,3000.],
            "observed_effective_gb_s":[r["observed"]["effective_end_to_end_read_gb_s"] for r in target_rows],
            "port_3000_link_1600_observed_gb_s":capped["observed"]["effective_end_to_end_read_gb_s"]},
        "coverage":{"physical_pages_bytes":list(PAGE_BYTES),"read_latency_ns":list(READ_LATENCIES_NS),
            "prior_physical_page_scan":"05 already contains hypothetical 1.5 MiB physical NAND pages; this group adds an independently fixed logical payload and transport chunk comparison."},
        "limitations":[
            "Targets use decimal GB/s: 400/1600/3000 GB/s = 0.4/1.6/3.0 TB/s. A configured port value alone is not observed throughput.",
            "This generously parallel synthetic inventory is not a feasibility, cost, area, packaging or real-silicon assertion.",
            "All points execute 384 MiB of sequential reads in 1.5 MiB transport chunks with at most 64 finite FIFO slots. There is no GPU/token serving denominator.",
            "Read-latency and physical-page scans retain the same authored geometry and port/link budget, so their actual parallelism bottlenecks remain visible.",
            "1.5 MiB physical pages are explicitly hypothetical; this variable is separate from the unchanged 1.5 MiB transport chunk.",
            "The link models serialized byte transfer and fixed propagation latency, without packet framing, retries, coherence or protocol calibration.",
            "Finite FIFO backpressure is modeled; additional staging-memory service is outside this transport benchmark."]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/hbf_hbm/18_bandwidth.json")
    args=parser.parse_args()
    result=run_bandwidth_study()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":result["status"],"checks":result["checks"],"summary":result["summary"],"out":str(args.out)}))
