"""Explicit multi-GPU/shared-pool serving and bounded sustained-load searches.

The physical resources execute through the normal scheduler. No scaling of a
single-GPU TPS is used to stand in for a multi-GPU measurement.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from heterollm_sim.ir import RankMappingSpec, ParallelSpec, PortSpec, LinkSpec
from heterollm_sim.control_plane_planner import _derive_requirements
from hbf_hbm_serving_study import build_study_scenario
from hbf_hbm_experiments import serving_row, group


def build_multi_gpu(architecture, gpu_count, *, model_id='small', prompt=128, output=8,
                    batch=4, rack=False):
    if gpu_count not in (1,2,4): raise ValueError('supported explicit GPU counts are 1, 2, 4')
    s=build_study_scenario(architecture,model_id=model_id,prompt=prompt,output=output,batch=batch)
    components=list(s.hardware.components); links=list(s.hardware.links)
    gpu=s.hardware.get_component('gpu0'); hbm=s.hardware.get_component('hbm0')
    registry={k:dict(v) for k,v in s.component_profiles.items()}
    for index in range(1,gpu_count):
        gpu_id=f'gpu{index}'; mem_id=f'hbm{index}'
        gpu_copy=replace(gpu,component_id=gpu_id,metadata=deepcopy(gpu.metadata))
        mem_profile=replace(s.resolve_component_profile('hbm0'),resource_id=mem_id+'.hbm_fabric')
        profile_id='system-'+mem_id
        registry['hbm'][profile_id]=mem_profile
        mem_copy=replace(hbm,component_id=mem_id,cost_profile_id=profile_id,
            metadata={**deepcopy(hbm.metadata),'memory_service_owner':mem_id+'.hbm_fabric'})
        components.extend((gpu_copy,mem_copy))
        for link in s.hardware.links:
            if 'gpu0' not in (link.source_component,link.target_component): continue
            remap={'gpu0':gpu_id,'hbm0':mem_id}
            links.append(replace(link,link_id=f'{link.link_id}-rank{index}',
                source_component=remap.get(link.source_component,link.source_component),
                target_component=remap.get(link.target_component,link.target_component)))
    # One shared endpoint port must carry traffic from every attached rank.
    # Its existing port limit remains unchanged; adding ranks never multiplies
    # the pool's device/channel bandwidth.
    port_fanout={}
    for link in links:
        for key in ((link.source_component,link.source_port),(link.target_component,link.target_port)):
            port_fanout[key]=port_fanout.get(key,0)+1
    components=[replace(c,ports=tuple(replace(p,max_links=max(p.max_links or 1,
        port_fanout.get((c.component_id,p.port_id),1))) for p in c.ports)) for c in components]
    for left in range(gpu_count):
        for right in range(left+1,gpu_count):
            port_l=f'peer-{right}'; port_r=f'peer-{left}'
            rate=400. if rack else 7200.  # bits/ns, i.e. 50 or 900 GB/s
            latency=2000. if rack else 200.
            for idx,pid in ((left,port_l),(right,port_r)):
                components=[replace(c,ports=c.ports+(PortSpec(pid,'Ethernet' if rack else 'NVLink',
                    'peer',version='study',lanes=1,bandwidth_gbps=rate),)) if c.component_id==f'gpu{idx}' else c
                    for c in components]
            links.append(LinkSpec(f'peer-{left}-{right}',f'gpu{left}',port_l,f'gpu{right}',port_r,
                'Ethernet' if rack else 'NVLink','study',1,rate,latency))
    components=[replace(c,metadata={**c.metadata,'routing_transit':False})
                if c.is_storage else replace(c,metadata={**c.metadata,
                    'routing_transit_endpoints':['hostmem0','cpu0']})
                if c.normalized_kind=='cpu' else c for c in components]
    if rack:
        # Remote shared-pool access also pays the rack fabric; only each GPU's
        # private activation memory retains its local HBM link.
        links=[replace(l,bandwidth_gbps=400.,latency_ns=2000.)
               if {'hbf0','hostmem0'} & {l.source_component,l.target_component}
               else l for l in links]
    metadata=deepcopy(s.placement.metadata)
    if architecture == 'D_hbf_weights':
        metadata['llama_backend_memory']['hbf0']['device_ids']=[f'gpu{i}' for i in range(gpu_count)]
    options=metadata['control_plane']['policy']['options']
    options['operator_targets']={}
    # Common shared weight and KV tier is an intentional memory-pool study.
    # Local HBM remains private activation/workspace for each GPU.
    parallel=ParallelSpec(tp_degree=gpu_count,rank_mapping=tuple(
        RankMappingSpec(i,f'gpu{i}',i,0,0,f'hbm{i}') for i in range(gpu_count)),
        layer_to_stage=dict(s.placement.parallel.layer_to_stage),collective_algorithm='ring')
    s=replace(s,hardware=replace(s.hardware,components=tuple(components),links=tuple(links)),
        component_profiles=registry,placement=replace(s.placement,parallel=parallel,metadata=metadata),
        runtime_profile=replace(s.runtime_profile,gpu_controllers={
            f'gpu{i}':s.runtime_profile.gpu_controllers['gpu0'] for i in range(gpu_count)}))
    # The control-plane selector rotates representative GPUs per requirement;
    # normal GPU execution expands each selection over all explicit TP ranks.
    options['operator_targets']={r.mapping_key:f'gpu{i%gpu_count}'
        for i,r in enumerate(_derive_requirements(s)) if r.mapping_key and not r.state_tensor}
    return s


def run_multi_case(architecture,gpu_count,**parameters):
    row_id=f"system/{architecture}/{gpu_count}gpu/"+('rack' if parameters.get('rack') else 'local')
    if parameters.get('model_id','small')!='small':
        row_id+='/'+parameters['model_id']+'/'+str(parameters.get('prompt',128))
    s=build_multi_gpu(architecture,gpu_count,**parameters)
    observed=[]
    row=serving_row(s,row_id,placement=architecture,result_observer=observed.append)
    result=observed[0]
    utilization=row['resources']['resource_utilization']
    row['physical_execution']={
        'gpu_compute_resources':{k:v for k,v in utilization.items() if k.startswith('gpu') and 'tensor_core' in k},
        'peer_link_resources':{k:v for k,v in utilization.items() if 'peer-' in k},
        'operator_execution_targets':result.scenario.placement.metadata['control_plane']['decision']['operator_execution_targets'],
        'execution_schedule_modes':sorted({str(b.cost.metadata.get('execution_stage_schedule_mode')) for b in result.serving.batches}),
        'executed_batch_count':len(result.serving.batches)}
    row['system_contract']={'parallelism':'tensor parallel, explicit ranks and scheduled collectives',
        'gpu_count':gpu_count,'shared_memory_pool':True,'private_activation_hbm':True,
        'rack':parameters.get('rack',False),'gpu_peer_gb_s':50. if parameters.get('rack') else 900.,
        'scope':'all GPUs read the same fixed-capacity pool; endpoint bandwidth is shared, not multiplied by GPU count',
        'parameters':parameters}
    return row


def run_system_study():
    rows=[]
    for architecture in ('A_hbm_only','B_all_kv_hbf','D_hbf_weights','G_cxl_dram'):
        for n in (1,2,4):
            print(f'[15_system] {architecture}/{n}',flush=True)
            rows.append(run_multi_case(architecture,n))
        if architecture in ('B_all_kv_hbf','G_cxl_dram'):
            rows.append(run_multi_case(architecture,4,rack=True))
    for architecture in ('A_hbm_only','B_all_kv_hbf'):
        for n in (1,2,4):
            print(f'[15_system] llama3_1-8b/{architecture}/{n}',flush=True)
            rows.append(run_multi_case(architecture,n,model_id='llama3_1-8b',prompt=2048,output=4,batch=1))
    return group('15_system','多GPU与共享内存池',rows,
        {'all_requests_complete':all(r['completed_requests']==r['request_count'] for r in rows),
         'reported_gpu_count_matches_hardware':all(r['gpu_count']==r['system_contract']['gpu_count'] for r in rows),
         'per_gpu_rate_uses_actual_gpu_count':all(abs(r['per_gpu_tps']*r['gpu_count']-r['node_tps'])<1e-7 for r in rows),
         'every_rank_executes_compute':all(len(r['physical_execution']['gpu_compute_resources'])==r['gpu_count'] and all(v>0 for v in r['physical_execution']['gpu_compute_resources'].values()) for r in rows),
         'multiple_ranks_use_peer_links':all(any(v>0 for v in r['physical_execution']['peer_link_resources'].values()) for r in rows if r['gpu_count']>1)},
        ['显式1/2/4 GPU张量并行及固定共享内存端点的完整算子服务；小模型全对照与Llama3.1-8B的A/B对照'],
        ['所有硬件与链路是假设；共享池架构不是每GPU本地权重复制，不能当作所有TP实现的性能。'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=run_system_study()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(result['status'],len(result['rows'])); return 0 if result['status']=='passed' else 1

if __name__=='__main__': raise SystemExit(main())
