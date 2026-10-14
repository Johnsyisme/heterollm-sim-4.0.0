"""Explicit measurement windows and useful-work metrics for HBF studies."""
from __future__ import annotations

import math


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def measurement_windows(serving):
    """Keep absolute event time separate from each chosen rate denominator."""
    metrics = tuple(serving.request_metrics.values())
    origin = float(serving.runtime_origin_ns)
    end = float(serving.makespan_ns)
    start = min((float(m.arrival_ns) for m in metrics), default=origin)
    return {
        "primary": "request_window",
        "request_window": {"start_ns": start, "end_ns": end,
                           "duration_ns": max(0., end-start)},
        "runtime_window": {"start_ns": origin, "end_ns": end,
                           "duration_ns": max(0., end-origin)},
        "initialization_ns": origin,
        "initialization_wait_in_request_window_ns": max(0., origin-start),
        "semantics": "primary rates span earliest request arrival through simulation end; TTFT/E2E start at each request arrival; runtime rates exclude initialization",
    }


def phase_metrics(serving, duration_ns, completed_ids):
    """Report wall-clock useful work separately from scheduled batch work."""
    rows = {}
    for phase in ("prefill", "decode"):
        batches = [b for b in serving.batches if any(i.phase == phase for i in b.items)]
        scheduled = sum(i.token_count for b in batches for i in b.items if i.phase == phase)
        busy = sum(b.end_ns-b.start_ns for b in batches)
        if phase == "prefill":
            # Recomputed prompts and rejected requests are not useful goodput.
            useful = sum(serving.request_metrics[k].prompt_tokens for k in completed_ids)
        else:
            useful = sum(int(e.details.get("visible_tokens", 0)) for e in serving.events
                         if e.event_type == "tokens_committed" and e.request_id in completed_ids
                         and e.details.get("source") != "prefill")
        rows[phase] = {
            "scheduled_tokens": scheduled, "useful_tokens": useful,
            "batch_occupied_ns": busy,
            "tokens_per_batch_busy_second": ratio(scheduled*1e9, busy),
            "wall_clock_tokens_per_second": ratio(useful*1e9, duration_ns),
            "wall_clock_denominator_ns": duration_ns,
            "semantics": ("wall rate uses completed requests' input tokens once for prefill and committed non-prefill output tokens for decode; both use the primary request window. Busy rate includes scheduled/recomputed work; mixed batches overlap and busy rates are not additive"),
        }
    return rows


def memory_metrics(report, duration_ns):
    storage = report.get("storage_traffic", {})
    totals = storage.get("resource_totals", {})
    channels = {k: float(v.get("service_ns", 0))/duration_ns
                for k, v in totals.items() if ":nand:channel:" in k and duration_ns > 0}
    lanes = {k: float(v.get("service_ns", 0)) for k, v in totals.items()
             if ":nand:parallel:" in k or ":nand:array:" in k}
    mean = sum(lanes.values())/len(lanes) if lanes else 0.
    cv = math.sqrt(sum((v-mean)**2 for v in lanes.values())/len(lanes))/mean if mean else None
    direction = {}
    for name in ("read", "write"):
        logical = storage.get(f"logical_{name}_bytes")
        physical = storage.get(f"physical_{name}_bytes")
        direction[name] = {
            "logical_bytes": logical, "media_bytes": physical,
            "effective_logical_gb_s": ratio(logical, duration_ns) if logical is not None else None,
            "media_gb_s": ratio(physical, duration_ns) if physical is not None else None,
        }
    write = direction["write"]
    page_observations = storage.get("nand_observations") or {}
    return {
        "scope": "operator NAND traffic (HBF/SSD); separately reported KV migration traffic is not implicitly included",
        "measurement_duration_ns": duration_ns,
        "directional_bandwidth": direction,
        "page_access_count": storage.get("pages_touched"),
        "page_access_payload_utilization":storage.get("page_access_payload_utilization"),
        "host_admission_peak_depths":{k:storage.get(k) for k in
            ('host_outstanding_at_admission_peak','host_read_outstanding_at_admission_peak',
             'host_write_outstanding_at_admission_peak')},
        "host_admission_scope":storage.get('queue_depth_scope'),
        "pages_read": storage.get("pages_read"),
        "pages_programmed": storage.get("pages_programmed"),
        "logical_to_media_byte_ratio": ratio(storage.get("logical_bytes", 0), storage.get("physical_bytes", 0)),
        "write_amplification": ratio(write["media_bytes"], write["logical_bytes"]) if write["media_bytes"] is not None and write["logical_bytes"] is not None else None,
        "channel_busy_fraction": channels,
        "active_service_lane_imbalance": {"coefficient_of_variation": cv,
            "max_over_mean": ratio(max(lanes.values(), default=0.), mean), "observed_lanes": len(lanes),
            "semantics": "busy-time imbalance among observed effective service lanes; not a physical-plane measurement and not all configured idle lanes"},
        "physical_plane_imbalance": page_observations.get("physical_plane_imbalance"),
        "physical_plane_loads": page_observations.get("physical_plane_loads"),
        "dynamic_wear": {key:page_observations.get(key) for key in (
            "wear_enabled", "wear_blocks", "wear_extra_ns")},
        "read_queue_depth": storage.get("nand_read_queue_depth"),
        "program_queue_depth": storage.get("nand_program_queue_depth"),
        "page_observation_coverage": {key:storage.get("metric_availability", {}).get(key)
            for key in ("nand_read_queue_depth", "nand_program_queue_depth")},
        "unavailable": {} if page_observations else {"physical_plane_imbalance": "detailed physical-plane observations not enabled; effective lanes are distinct",
                        "read_queue_depth": "page-level queue occupancy timeline not recorded",
                        "program_queue_depth": "page-level queue occupancy timeline not recorded"},
        "semantics": "GB/s equals bytes/ns; logical/media ratios include RMW and merging and must not be called physical page fill fractions",
    }


def speculative_metrics(serving):
    events = [e for e in serving.events if e.event_type == "tokens_committed"]
    drafted = sum(int(e.details.get("draft_tokens", 0)) for e in events)
    accepted = sum(int(e.details.get("accepted_draft_tokens", 0)) for e in events)
    rejected = sum(int(e.details.get("rejected_draft_tokens", 0)) for e in events)
    return {"drafted_tokens": drafted, "accepted_draft_tokens": accepted,
            "rejected_draft_tokens": rejected, "acceptance_rate": ratio(accepted, drafted),
            "rounds":[{"request_id":e.request_id,"timestamp_ns":float(e.timestamp_ns),
                       **dict(e.details)} for e in events if e.details.get("draft_tokens",0)],
            "status": "observed" if drafted else "not_exercised"}


def prefetch_overlap(serving, gpu_components):
    """Intersect copies with realized GPU operator stages, not batch envelopes."""
    def union_duration(intervals):
        total=0.; right=None
        for start,end in sorted(intervals):
            if end <= start:
                continue
            if right is None or start >= right:
                total += end-start
            elif end > right:
                total += end-right
            right=max(right if right is not None else end,end)
        return total
    stages=[]
    for b in serving.batches:
        decode=any(i.phase=='decode' for i in b.items)
        for stage in b.cost.metadata.get('execution_stage_schedule',()):
            if stage.get('component_id') not in gpu_components or not stage.get('invocation_group_id'):
                continue
            stages.append((float(stage['start_ns']),float(stage['end_ns']),decode,
                           tuple(stage.get('request_ids',b.request_ids))))
    copies=[]; all_intersections=[]; decode_intersections=[]
    for e in serving.events:
        if e.event_type!='kv_prefetch_start' or 'completion_ns' not in e.details:
            continue
        start=float(e.timestamp_ns); end=float(e.details['completion_ns'])
        intersections=[]; decode_hits=[]; premature=[]
        for left,right,decode,requests in stages:
            hit=(max(start,left),min(end,right))
            if hit[1]<=hit[0]:
                continue
            if e.request_id in requests:
                premature.append(hit)
            else:
                intersections.append(hit)
                if decode: decode_hits.append(hit)
        all_intersections.extend(intersections); decode_intersections.extend(decode_hits)
        copies.append({'request_id':e.request_id,'start_ns':start,'completion_ns':end,
                       'bytes':int(e.details.get('bytes',0)),
                       'independent_gpu_operator_overlap_ns':union_duration(intersections),
                       'independent_decode_operator_overlap_ns':union_duration(decode_hits),
                       'premature_consumer_operator_overlap_ns':union_duration(premature)})
    return {'copies':copies,'operator_overlap_ns':union_duration(all_intersections),
            'decode_operator_overlap_ns':union_duration(decode_intersections),
            'premature_consumer_overlap_ns':sum(r['premature_consumer_operator_overlap_ns'] for r in copies),
            'scope':'intersection with realized GPU operator stages of other requests; host setup and GPU frontend/controller stages excluded; operator envelopes may include memory stalls, not tensor-core busy time'}
