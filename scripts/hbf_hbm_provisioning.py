"""Re-execute full serving for a bounded, explicitly hypothetical package search."""
from __future__ import annotations

from dataclasses import replace
import math

from hbf_hbm_economics import conditional_node_cost
from hbf_hbm_serving_study import build_study_scenario
from heterollm_sim.memory_types import parse_physical_memory_config
from heterollm_sim.planner import ScenarioValidationError

HBM_CAPACITY = 36_000_000_000
HBF_CAPACITY = 512_000_000_000
BUDGETS = {"gpu_count": 1, "memory_area_mm2": 400., "active_node_power_w": 900.,
           "node_capex_usd": 15500., "total_memory_capacity_bytes": 1_600_000_000_000}


def build_package_scenario(hbm_packages, hbf_packages, architecture, *, model_id="llama3_1-70b", output=8):
    """One aggregate tier per kind; all capacity/bandwidth/concurrency scale together."""
    if not isinstance(hbm_packages, int) or not isinstance(hbf_packages, int) or min(hbm_packages, hbf_packages)<0:
        raise ValueError("package counts must be nonnegative integers")
    if not hbm_packages:
        raise ValueError("this policy requires HBM for activation/workspace and active KV")
    if not hbf_packages and architecture != "A_hbm_only":
        raise ValueError("the requested placement requires at least one HBF package")
    s = build_study_scenario(architecture, model_id=model_id, prompt=2048, batch=1, output=output,
                             hbm_capacity_bytes=HBM_CAPACITY*hbm_packages,
                             hbf_capacity_bytes=HBF_CAPACITY*max(1,hbf_packages))
    counts={"hbm0":hbm_packages,"hbf0":hbf_packages}
    components=[]
    for c in s.hardware.components:
        if c.component_id not in counts:
            components.append(c); continue
        count=counts[c.component_id]
        if not count:
            continue
        physical=parse_physical_memory_config(c.metadata["physical_memory_config"])
        if c.component_id=="hbm0":
            physical=replace(physical, channels=8*count,
                interface_bandwidth_gb_s=1600.*count,read_bandwidth_gb_s=1600.*count,
                write_bandwidth_gb_s=1600.*count,max_outstanding_requests=64*count)
        else:
            physical=replace(physical,channels=64*count,parallel_units=64*count,
                host_bandwidth_gb_s=1600.*count,host_read_bandwidth_gb_s=1600.*count,
                host_write_bandwidth_gb_s=16.*count,max_outstanding_requests=32*count)
        components.append(replace(c, bandwidth_gbps=1600.*8*count,
            read_bandwidth_gbps=1600.*8*count,
            write_bandwidth_gbps=(1600. if c.component_id=="hbm0" else 16.)*8*count,
            ports=tuple(replace(p,bandwidth_gbps=1600.*8*count) for p in c.ports),
            metadata={**{k:v for k,v in c.metadata.items() if k!='memory_service'},
                "physical_memory_config":physical,"max_outstanding_requests":(64 if c.component_id=='hbm0' else 32)*count,
                "parallel_lanes":count if c.component_id=='hbm0' else 64*count}))
    ids={c.component_id for c in components}
    links=[]; gpu_ports={}
    for link in s.hardware.links:
        if link.source_component not in ids or link.target_component not in ids:
            continue
        memory=next((x for x in (link.source_component,link.target_component) if x in counts),None)
        if memory:
            link=replace(link,bandwidth_gbps=1600.*8*counts[memory])
            gpu_ports[link.source_port if link.source_component=="gpu0" else link.target_port]=link.bandwidth_gbps
        links.append(link)
    components=[replace(c,ports=tuple(replace(p,bandwidth_gbps=gpu_ports[p.port_id]) if p.port_id in gpu_ports else p
        for p in c.ports)) if c.component_id=="gpu0" else c for c in components]
    registry={k:dict(v) for k,v in s.component_profiles.items()}
    hbm=s.hardware.get_component("hbm0")
    profile=s.resolve_component_profile("hbm0")
    registry["hbm"][hbm.cost_profile_id]=replace(profile,bandwidth_gb_s=1600.*hbm_packages,
        read_bandwidth_gb_s=1600.*hbm_packages,write_bandwidth_gb_s=1600.*hbm_packages,
        parallel_lanes=hbm_packages,max_outstanding_requests=64*hbm_packages)
    if hbf_packages:
        flash=s.hardware.get_component("hbf0")
        profile=s.resolve_component_profile("hbf0")
        registry[s.component_profile_kind(flash)][flash.cost_profile_id]=replace(profile,
            bandwidth_gb_s=1600.*hbf_packages,read_bandwidth_gb_s=1600.*hbf_packages,
            write_bandwidth_gb_s=16.*hbf_packages,parallel_lanes=64*hbf_packages,
            max_outstanding_requests=32*hbf_packages)
    return replace(s,hardware=replace(s.hardware,components=tuple(components),links=tuple(links)),component_profiles=registry)


def budget_feasibility(cost):
    return {"fixed_gpu_count":cost["gpu_count"]==BUDGETS["gpu_count"],
        "area_ceiling":cost["memory_package_area_mm2"]<=BUDGETS["memory_area_mm2"],
        "power_ceiling":cost["active_node_power_w"]<=BUDGETS["active_node_power_w"],
        "cost_ceiling":cost["capex_usd"]<=BUDGETS["node_capex_usd"],
        "capacity_ceiling":cost["hbm_capacity_bytes"]+cost["hbf_capacity_bytes"]<=BUDGETS["total_memory_capacity_bytes"]}


def run_provisioning_study(*, model_id="llama3_1-70b", progress=True):
    from hbf_hbm_experiments import serving_row, group
    rows=[]
    for hbm in (4,3,2,1,0):
        hbf=4-hbm
        architectures=("A_hbm_only",) if not hbf else ("B_all_kv_hbf","D_hbf_weights")
        for architecture in architectures:
            row_id=f"provision/{hbm}HBM-{hbf}HBF/{architecture}"
            if progress: print("[11_provisioning] "+row_id,flush=True)
            cost=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=0.)
            configuration={"hbm_packages":hbm,"hbf_packages":hbf,"model_id":model_id,
                "hbm_capacity_bytes":hbm*HBM_CAPACITY,"hbf_capacity_bytes":hbf*HBF_CAPACITY,
                "hbm_peak_read_gb_s":1600.*hbm,"hbf_peak_read_gb_s":1600.*hbf,
                "hbf_peak_write_gb_s":16.*hbf,"hbf_parallel_units":64*hbf}
            if not hbm:
                row={"id":row_id,"mode":"configuration_rejected","placement":architecture,
                    "reason":"This policy requires HBM staging/activation/workspace; this is not a proof that every zero-HBM architecture is impossible."}
            else:
                s=build_package_scenario(hbm,hbf,architecture,model_id=model_id)
                try:
                    row=serving_row(s,row_id,placement=architecture)
                except ScenarioValidationError as error:
                    row={"id":row_id,"mode":"configuration_rejected","placement":architecture,"reason":str(error)}
            if row["mode"]=="full_operator_serving":
                cost=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=row["node_tps"])
                cost["formula_scope"]="Package counts match this re-executed serving row; price/power/area remain hypothetical inputs, not measured node characteristics."
                row["conditional_5yr_cost"]=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,years=5.,active_output_tps=row["node_tps"])
                row["conditional_slo_cost"]=conditional_node_cost(hbm_packages=hbm,hbf_packages=hbf,active_output_tps=row["goodput_tps"])
                for field in ("conditional_5yr_cost","conditional_slo_cost"):
                    row[field]["formula_scope"]=cost["formula_scope"]
            row.update(package_configuration=configuration,conditional_cost=cost,budget_checks=budget_feasibility(cost))
            row["jointly_feasible"]=all(row["budget_checks"].values()) and row["mode"]=="full_operator_serving" and row.get("qualified_requests")==row.get("request_count")
            rows.append(row)
    candidates=[r for r in rows if r["jointly_feasible"] and r.get("goodput_tps",0)>0]
    selected=min(candidates,key=lambda r:r["conditional_slo_cost"]["conditional_usd_per_million_tokens"],default=None)
    full=[r for r in rows if r["mode"]=="full_operator_serving"]
    checks={"all_candidates_executed_or_explicitly_rejected":len(rows)==9 and all(r["mode"] in {"full_operator_serving","configuration_rejected"} for r in rows),
        "executed_requests_completed":bool(full) and all(r["completed_requests"]==r["request_count"] for r in full),
        "cost_counts_match_execution":all(r["conditional_cost"]["hbm_packages"]==r["package_configuration"]["hbm_packages"] and r["conditional_cost"]["hbf_packages"]==r["package_configuration"]["hbf_packages"] for r in rows),
        "all_reported_rates_finite":all(math.isfinite(r["node_tps"]) and r["node_tps"]>0 for r in full)}
    return group("11_provisioning","⑪ 预算约束下重新执行服务",rows,checks,
        ["固定4槽、GPU和SLO，重新执行70B代表工作量后进行有限候选预算筛选与条件成本比较"],
        ["容量、功率、成本是共同上限而非每点消耗完全相等；价格/面积/功耗是假设。",
         "每种内存使用一个聚合设备，容量、带宽和并行度同时随栈数改变；不是实际多栈布线模型。",
         "单请求2K/8输出、有限候选不构成全工作负载全局最优；热降额、磨损和成本尚未闭环。",
         "无可行候选时保留null，不放宽SLO或预算以制造赢家。"],
        budgets=BUDGETS,selected_candidate_id=selected["id"] if selected else None,
        selection_objective="minimum conditional dollars per million SLO-qualified tokens among actually executed feasible candidates",
        feasible_candidate_ids=[r["id"] for r in candidates])
