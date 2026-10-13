"""Extended model coverage without disguising unsupported attention as GQA.

Run: .venv/Scripts/python.exe -B scripts/hbf_hbm_extended_models.py
Full serving: Qwen3-Coder and an explicitly synthetic 1T GQA MoE.
Other rows: one-token structural probes with actual DRAM/NAND transactions,
analytical matrix FLOPs, and explicit omitted kernels; never LLM token/s.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore
from heterollm_sim.memory_types import AccessRequest, parse_physical_memory_config
from heterollm_sim.model_presets import EXTENDED_MODEL_SPECS
from hbf_hbm_experiments import group
from hbf_hbm_serving_study import build_study_scenario, run_case

TIB = 1024 ** 4
ARCHITECTURES = ("A_hbm_only", "B_all_kv_hbf", "D_hbf_weights")
CONTEXTS = (2048, 32768)
FULL_MODELS = ("qwen3-coder-30b-a3b", "synthetic-1t-gqa-moe")


def _matrix(name, parameters, active=None):
    active = parameters if active is None else active
    return dict(name=name, resident_parameters=int(parameters), active_parameters=int(active),
                resident_bytes=2*int(parameters), read_bytes=2*int(active), flops=2*int(active))


def _cache(name, entries, width, *, read_entries=None, write_entries=1, copies=1):
    """BF16 entries; shared K=V has copies=1, ordinary independent K/V=2."""
    read_entries = entries if read_entries is None else read_entries
    return dict(name=name, entries=int(entries), width=int(width), copies=copies,
                resident_bytes=int(entries*width*copies*2),
                read_bytes=int(read_entries*width*copies*2),
                write_bytes=int(write_entries*width*copies*2))


def structural_workload(model_id, context):
    """One decode query at position context-1; context includes the new token.

    A multiply-add is two FLOPs. Matrix weights are expanded to BF16 for an
    intentionally common precision comparison. No quantization speedup is
    inferred. Scalar activations, softmax/top-k, normalization, RoPE and mHC
    Sinkhorn arithmetic are not timed; this is a dominant-subgraph probe.
    """
    if context < 1:
        raise ValueError("context must be positive")
    s = EXTENDED_MODEL_SPECS[model_id]
    h, n, layers = s["hidden"], s["heads"], []
    for i in range(s["layers"]):
        matrices, caches, extra_flops = [], [], {}
        if s["architecture"] == "mla":
            r, q, nope, rope, v = (s[k] for k in ("kv_rank", "q_rank", "nope_dim", "rope_dim", "value_dim"))
            matrices = [_matrix("query_down",h*q), _matrix("query_up",q*n*(nope+rope)),
                _matrix("latent_and_rope",h*(r+rope)),
                _matrix("absorbed_key_factor",r*n*nope), _matrix("latent_value_factor",r*n*v),
                _matrix("output",n*v*h)]
            caches = [_cache("latent_and_shared_rope",context,r+rope)]
            # Absorbed key contraction and latent-to-value contraction use
            # the same number of multiplications as the factor matrix sizes.
            extra_flops["latent_attention_qk_av"] = 2*n*context*(2*r+rope)
            layer_type = "absorbed_mla"
        elif s["architecture"] == "compressed_shared_kv":
            d, q, o, g = (s[k] for k in ("head_dim", "q_rank", "o_rank", "o_groups"))
            ratio = s["compress_ratios"][i]
            matrices = [_matrix("query_down",h*q), _matrix("query_up",q*n*d),
                _matrix("shared_kv",h*d), _matrix("grouped_output_down",n*d*o),
                _matrix("grouped_output_up",g*o*h)]
            window = min(context,s["window"])
            caches = [_cache("shared_kv_window",window,d)]
            attended = window
            if ratio:
                overlap = 2 if ratio == 4 else 1
                entries = context//ratio
                emitted = int(context % ratio == 0)
                selected = min(entries,s["index_top_k"]) if ratio==4 else entries
                matrices += [_matrix("compress_kv",h*overlap*d), _matrix("compress_gate",h*overlap*d)]
                caches += [_cache("compressed_shared_kv",entries,d,read_entries=selected,write_entries=emitted)]
                # Persistent source projections and gates before a window
                # closes, plus CSA's previous-window overlap. Provision the
                # maximum live window; read only on an emission boundary.
                caches += [_cache("compressor_source_window",ratio,overlap*d*2,
                    read_entries=ratio if emitted else 0, write_entries=1)]
                if ratio==4:
                    caches += [_cache("compressor_previous_ca_overlap",ratio,d*2,
                        read_entries=ratio if emitted and entries>1 else 0,write_entries=ratio if emitted else 0)]
                attended += selected
                if ratio==4:
                    ih, idim = s["index_heads"], s["index_dim"]
                    matrices += [_matrix("index_query",q*ih*idim), _matrix("index_scores",h*ih),
                        _matrix("index_compress_kv",h*2*idim), _matrix("index_compress_gate",h*2*idim)]
                    caches += [_cache("compressed_index_keys",entries,idim,write_entries=emitted),
                        _cache("index_source_window",ratio,4*idim,
                            read_entries=ratio if emitted else 0,write_entries=1),
                        _cache("index_previous_ca_overlap",ratio,2*idim,
                            read_entries=ratio if emitted and entries>1 else 0,write_entries=ratio if emitted else 0)]
                    extra_flops["index_dot_products"] = 2*ih*entries*idim
                extra_flops["compression_weighted_sum"] = 2*ratio*overlap*d*emitted
                if ratio==4:
                    extra_flops["index_compression_weighted_sum"] = 2*ratio*2*s["index_dim"]*emitted
            # K and V alias storage, but QK and AV remain two computations.
            extra_flops["shared_kv_attention_qk_av"] = 4*n*attended*d
            m=s["hc_mult"]
            matrices += [_matrix("attention_mhc_linear_mixer",(m*m+2*m)*(m*h)),
                         _matrix("ffn_mhc_linear_mixer",(m*m+2*m)*(m*h))]
            layer_type = {0:"sliding_shared_kv",4:"compressed_sparse_attention",128:"heavily_compressed_attention"}[ratio]
        else:
            d, kv = s["head_dim"], s["kv_heads"]
            matrices = [_matrix("query",h*n*d), _matrix("key",h*kv*d),
                        _matrix("value",h*kv*d),_matrix("output",n*d*h)]
            full = (i+1)%s["full_attention_interval"] == 0
            # Chunk attention is not a rolling sliding window: at a new
            # chunk, context resets. End-of-chunk points retain 8192 entries.
            entries = context if full else (context-1)%s["chunk"]+1
            caches = [_cache("independent_key_value",entries,kv*d,copies=2)]
            extra_flops["gqa_attention_qk_av"] = 4*n*entries*d
            layer_type = "full_nope_gqa" if full else "chunked_rope_gqa"
        moe = (i>=s["dense_layers"] if s["architecture"]=="mla" else
               (i+1)%s["moe_step"]==0 if s["architecture"]=="chunked_full_gqa_moe" else True)
        if moe:
            unit=3*h*s["expert_ffn"]
            matrices += [_matrix("routed_experts",unit*s["experts"],unit*s["top_k"]),
                _matrix("shared_experts",unit*s["shared_experts"]),_matrix("router",h*s["experts"])]
        else:
            matrices += [_matrix("dense_mlp",3*h*s["dense_ffn"])]
        layers.append(dict(layer=i, attention=layer_type, moe=moe, matrices=matrices,
                           caches=caches, extra_flops=extra_flops))
    # Embedding lookup reads one vector, not the entire vocabulary table.
    embedding=_matrix("embedding_lookup",h*s["vocabulary"],h)
    embedding["flops"]=0
    head=_matrix("lm_head",h*s["vocabulary"])
    matrices=[embedding,head]+[m for layer in layers for m in layer["matrices"]]
    caches=[c for layer in layers for c in layer["caches"]]
    totals={"matrix_resident_parameters":sum(m["resident_parameters"] for m in matrices),
            "matrix_resident_bytes":sum(m["resident_bytes"] for m in matrices),
            "matrix_read_bytes":sum(m["read_bytes"] for m in matrices),
            "cache_resident_bytes":sum(c["resident_bytes"] for c in caches),
            "cache_read_bytes":sum(c["read_bytes"] for c in caches),
            "cache_write_bytes":sum(c["write_bytes"] for c in caches),
            "matrix_flops":sum(m["flops"] for m in matrices),
            "attention_and_compression_flops":sum(sum(l["extra_flops"].values()) for l in layers)}
    return dict(model_id=model_id,context=context,spec=dict(s),layers=layers,
                embedding=embedding,head=head,totals=totals)


def run_structural_probe(model_id, context, architecture, *, compute_tflops=100.):
    if architecture not in ARCHITECTURES or compute_tflops<=0:
        raise ValueError("invalid placement or compute throughput")
    workload=structural_workload(model_id,context)
    s=build_study_scenario("A_hbm_only",hbm_capacity_bytes=4*TIB,hbf_capacity_bytes=4*TIB)
    cores={}
    for owner in ("hbm0","hbf0"):
        config=parse_physical_memory_config(s.hardware.get_component(owner).metadata["physical_memory_config"])
        cores[owner]=(DramCore if owner=="hbm0" else NandCore)(config,capture_details=False)
    weight_owner="hbf0" if architecture=="D_hbf_weights" else "hbm0"
    cache_owner="hbf0" if architecture=="B_all_kv_hbf" else "hbm0"
    cursor={owner:0 for owner in cores}
    clock=0.
    transactions=[]
    compute_ns=0.
    def allocate(owner,size):
        address=(cursor[owner]+16383)//16384*16384
        cursor[owner]=address+size
        if cursor[owner]>cores[owner].config.capacity_bytes:
            raise ValueError("probe exceeds declared physical capacity")
        return address
    def access(owner,name,op,address,count):
        nonlocal clock
        if not count:
            return
        result=cores[owner].submit(AccessRequest(name,op,address,int(count),clock))
        transactions.append(dict(name=name,owner=owner,operation=op,address=address,
            logical_bytes=count,start_ns=clock,completion_ns=result.completion_ns,
            host_transfer_bytes=result.host_transfer_bytes,physical_read_bytes=result.physical_read_bytes,
            physical_write_bytes=result.physical_write_bytes,internal_transfer_bytes=result.internal_transfer_bytes,
            queue_wait_ns=result.queue_wait_ns))
        clock=result.completion_ns
    def execute_matrices(matrices,prefix):
        nonlocal clock,compute_ns
        for m in matrices:
            address=allocate(weight_owner,m["resident_bytes"])
            access(weight_owner,prefix+m["name"],"read",address,m["read_bytes"])
        delta=sum(m["flops"] for m in matrices)/(compute_tflops*1000.)
        clock+=delta
        compute_ns+=delta
    def compute_extra(flops):
        nonlocal clock,compute_ns
        delta=flops/(compute_tflops*1000.)
        clock+=delta
        compute_ns+=delta
    execute_matrices([workload["embedding"]],"embedding.")
    for layer in workload["layers"]:
        prefix=f"layer{layer['layer']}."
        matrices={m["name"]:m for m in layer["matrices"]}
        caches={c["name"]:(c,allocate(cache_owner,c["resident_bytes"])) for c in layer["caches"]}
        def matrix_names(names):
            execute_matrices([matrices.pop(name) for name in names if name in matrices],prefix)
        def cache_op(name,operation):
            if name not in caches:
                return
            c,address=caches[name]
            size=c[operation+"_bytes"]
            access(cache_owner,prefix+name+(".append" if operation=="write" else ".read"),operation,
                   address+c["resident_bytes"]-size,size)
        matrix_names(("attention_mhc_linear_mixer",))
        late={"latent_value_factor","output","grouped_output_down","grouped_output_up",
              "ffn_mhc_linear_mixer","router","routed_experts","shared_experts","dense_mlp"}
        matrix_names(tuple(name for name in matrices if name not in late))
        extra=dict(layer["extra_flops"])
        if layer["attention"] in {"compressed_sparse_attention","heavily_compressed_attention","sliding_shared_kv"}:
            cache_op("shared_kv_window","write")
            for branch,compressed,flop_name in (
                ("compressor","compressed_shared_kv","compression_weighted_sum"),
                ("index","compressed_index_keys","index_compression_weighted_sum")):
                # Producer data must arrive before its compressed output is
                # published; previous Ca is read before being overwritten.
                cache_op(branch+"_source_window","write")
                cache_op(branch+"_source_window","read")
                cache_op(branch+"_previous_ca_overlap","read")
                compute_extra(extra.pop(flop_name,0))
                cache_op(compressed,"write")
                cache_op(branch+"_previous_ca_overlap","write")
            cache_op("compressed_index_keys","read")
            compute_extra(extra.pop("index_dot_products",0))
            cache_op("shared_kv_window","read")
            cache_op("compressed_shared_kv","read")
        else:
            for name in caches:
                cache_op(name,"write")
                cache_op(name,"read")
        compute_extra(sum(extra.values()))
        matrix_names(("latent_value_factor","output","grouped_output_down","grouped_output_up"))
        matrix_names(("ffn_mhc_linear_mixer","router","routed_experts","shared_experts","dense_mlp"))
        if matrices:
            raise AssertionError("unexecuted structural matrix: "+str(tuple(matrices)))
    execute_matrices([workload["head"]],"head.")
    traffic={owner:{key:sum(t[key] for t in transactions if t["owner"]==owner) for key in
        ("logical_bytes","host_transfer_bytes","physical_read_bytes","physical_write_bytes","internal_transfer_bytes")}
        for owner in cores}
    return dict(id=f"{model_id}-{context}-{architecture}",mode="structural_physical_probe",
        model_id=model_id,context=context,placement=architecture,executed_serving=False,
        node_tps=None,goodput_tps=None,probe_step_ns=clock,analytical_compute_ns=compute_ns,
        physical_memory_ns=clock-compute_ns,compute_tflops_assumption=compute_tflops,
        support_boundary={"native_full_serving":False,"facts":workload["spec"]["facts"],
            "geometry_support":"partial_architecture_geometry" if model_id=="llama4-maverick" else
                "public_config_defaults_not_checkpoint_verified" if model_id=="llama4-scout" else "official_checkpoint_dimensions",
            "assumptions":list(workload["spec"]["assumptions"]),"unsupported":list(workload["spec"]["unsupported"]),
            "precision":workload["spec"]["native_precision"],"sources":list(workload["spec"]["sources"])},
        workload=workload,allocated_bytes=cursor,physical_traffic=traffic,transactions=transactions,
        measurement_scope="One query at context-1 with dependency-serialized dominant matrix/cache accesses; no prefill, scheduler, link, distributed collectives or nonlinear-kernel timing. Physical media use shared study assumptions.",
        limitations=["Not complete model latency or LLM token/s; matrix BF16 expansion differs from native mixed quantization.",
            "Resident count covers explicit matrices and cache windows; scalar norms/biases, router lookup table, mHC head and workspace are omitted.",
            "KV starts populated; prefix creation and weight loading are outside this single decode-query probe.",
            "Selected expert and CSA blocks are packed contiguous ranges; routing and gather distribution are explicit locality assumptions.",
            "MLA latent/shared-KV is fetched once for QK and AV under ideal local retention or fused reuse. This is a minimum-read traffic assumption, not a guarantee that a vendor kernel reads the cache only once.",
            "Compressor rolling-buffer maximum allocation is provisioned; state updates are a simplified store/read dataflow, not numerical tensor execution."])


def annotate_full_support(row):
    """Keep simulator graph coverage separate from native inference claims."""
    model=row["parameters"]["model_id"]
    row["support_boundary"]={"native_full_serving":True,
        "meaning":"All supported model operators execute in the simulator; this is not native numerical inference or calibrated hardware execution.",
        "geometry_source":"https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/raw/main/config.json"
            if model=="qwen3-coder-30b-a3b" else "explicit synthetic 40-layer / 512-expert top-8 GQA design",
        "synthetic_model":model=="synthetic-1t-gqa-moe",
        "inventory":"Each A/B/D has the same analytical 4 TiB HBM + 4 TiB HBF inventory; capacities do not imply a real single-stack device or equal procurement budget."}
    return row


def _full_point(args):
    model,context,architecture=args
    return annotate_full_support(run_case(architecture,row_id=f"extended-{model}-{context}-{architecture}",
                    model_id=model,prompt=context,batch=1,output=4,
                    hbm_capacity_bytes=4*TIB,hbf_capacity_bytes=4*TIB))


def run_extended_models_study(*, workers=2, sections=("full","probes")):
    rows=[]
    if "full" in sections:
        points=[(m,c,a) for m in FULL_MODELS for c in CONTEXTS for a in ARCHITECTURES]
        if workers>1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs={pool.submit(_full_point,p):p for p in points}
                for job in as_completed(jobs):
                    row=job.result(); rows.append(row)
                    print("completed full",row["id"],flush=True)
        else:
            for point in points:
                row=_full_point(point);rows.append(row)
                print("completed full",row["id"],flush=True)
    if "probes" in sections:
        for model in EXTENDED_MODEL_SPECS:
            for context in CONTEXTS:
                for architecture in ARCHITECTURES:
                    row=run_structural_probe(model,context,architecture);rows.append(row)
                    print("completed probe",row["id"],flush=True)
    return summarize_extended_models(rows,sections=sections)


def summarize_extended_models(rows, *, sections=("full","probes")):
    rows=sorted(rows,key=lambda r:r["id"])
    full=[r for r in rows if r["mode"]=="full_operator_serving"]
    probes=[r for r in rows if r["mode"]=="structural_physical_probe"]
    checks={"requested_sections_present":bool(rows)}
    if "full" in sections:
        checks.update(full_serving_12_points=len(full)==12,
            full_serving_completed=all(r["completed_requests"]==1 and r["visible_output_tokens"]==4 for r in full),
            full_placements_verified=all(all(r["placement_verification"].values()) for r in full),
            synthetic_model_over_one_trillion=all(r["model_structure"]["weight_bytes"]>=2e12 for r in full if "synthetic-1t" in r["id"]))
    if "probes" in sections:
        checks.update(structural_probe_24_points=len(probes)==24,
            no_probe_tps=all(r["node_tps"] is None and not r["executed_serving"] for r in probes),
            probes_execute_physical_memory=all(r["physical_memory_ns"]>0 and r["transactions"] for r in probes),
            probe_compute_flops_conserved=all(math.isclose(r["analytical_compute_ns"],
                (r["workload"]["totals"]["matrix_flops"]+r["workload"]["totals"]["attention_and_compression_flops"])
                /(r["compute_tflops_assumption"]*1000.),rel_tol=1e-12) for r in probes),
            probe_logical_bytes_conserved=all(sum(t["logical_bytes"] for t in r["transactions"])==
                sum(r["workload"]["totals"][k] for k in ("matrix_read_bytes","cache_read_bytes","cache_write_bytes")) for r in probes))
    return group("17_extended_models","扩展模型：完整服务与专属结构物理探针",rows,checks,
        (["Qwen3-Coder 与合成 1T GQA MoE 的 A/B/D 完整算子服务"] if "full" in sections else [])+
        (["MLA、V4 压缩共享 KV、Llama4 分块/全注意力的结构公式与物理内存执行"] if "probes" in sections else []),
        ["DeepSeek 与 Llama4 行是局部结构探针，不是原生完整服务或 LLM 吞吐。",
         "Maverick 详细几何采用明确 Scout-default 场景假设；没有绕过 gated checkpoint。",
         "统一 BF16、固定硬件和路由局部性均为分析假设；不需要或宣称真实设备校准。"],
         specification_sources={k:dict(v) for k,v in EXTENDED_MODEL_SPECS.items()},sections=list(sections),
         metric_definitions={"full_operator_serving":"完整仿真算子图的请求窗口 TPS/goodput；不是数值推理或真实芯片测量。",
            "probe_step_ns":"单 decode 查询的主要矩阵与缓存访问串行成本，不含 prefill、服务调度、链路、非线性 kernel 或分布式通信；不可取倒数称 LLM TPS。",
            "geometry_support":"Maverick=partial_architecture_geometry；Scout=public_config_defaults_not_checkpoint_verified；DeepSeek=official_checkpoint_dimensions。",
            "matrix_resident_parameters":"列出的 BF16 展开矩阵参数总量，省略标量 norm/bias、路由表、部分辅助分支及视觉参数，不是精确 checkpoint 文件大小。"})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--section",choices=("all","full","probes"),default="all")
    args=parser.parse_args()
    if args.output is None:
        suffix="" if args.section=="all" else "_"+args.section
        args.output=ROOT/"artifacts/hbf_hbm"/f"17_extended_models{suffix}.json"
    result=run_extended_models_study(workers=args.workers,sections=("full","probes") if args.section=="all" else (args.section,))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"rows":len(result["rows"]),"output":str(args.output)},ensure_ascii=False))
    return 0 if result["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
