#!/usr/bin/env python3
"""Gen5 G2 frozen P0 shared/residual causal intervention, full 840-row science.

Two independent Tesla T4 workers, non-DDP, one frozen dev shard each. Reuses the
numerically tested 22-forward/B32 reference path, NOT v10 FFT/GEMM and NOT a new
cached implementation. Phase2 FULL and unmodified-parent ZERO are checked for
EVERY batch. Does not train, update checkpoints, or access confirmatory data.

This is a separate, explicitly authorized scientific full run. Run ONLY from a
clean, exact committed repository HEAD via the ContraMamba pinned run workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre

SCHEMA_WORKER = 'GEN5_G2_FROZEN_P0_TWO_T4_FULL_SCIENCE_WORKER_V1'
SCHEMA_SUMMARY = 'GEN5_G2_FROZEN_P0_TWO_T4_FULL_SCIENCE_SUMMARY_V1'
OUTPUT_FOLDER = 'reports/reason_router_gen5_g2_functional_science_runs'
EXPECTED_BASE_HEAD = '99d1b477a84c2a93650239288439832a1d830bfe'
FULL_FORWARD_COUNT = sum(len(pre.shard_batches(w))*pre.EXPECTED_BACKBONE_FORWARDS_PER_BATCH for w in (0,1))
assert FULL_FORWARD_COUNT == 594


def output_contract(repo: Path, output_dir: Path) -> str:
    pre.need(not output_dir.exists(), 'OUTPUT_DIR_EXISTS')
    root=repo.resolve()
    path=output_dir.resolve()
    pre.need(path.is_relative_to(root), 'OUTPUT_OUTSIDE_REPO')
    rel=path.relative_to(root)
    pre.need(len(rel.parts)==3 and tuple(rel.parts[:2])==('reports',OUTPUT_FOLDER.split('/')[-1]), 'OUTPUT_SCOPE')
    run=rel.parts[-1]
    pre.need(bool(run) and all(c.islower() or c.isdigit() or c=='-' for c in run), 'RUN_SYNTAX')
    pre.need(os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT','').replace('\\','/')==rel.as_posix(),
             'RUN_ARTIFACT_ROOT_MISMATCH')
    return rel.as_posix()


def static_guard(repo:Path, head:str)->None:
    pre.static_guard(repo,head)
    pre.need(subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',
                             EXPECTED_BASE_HEAD,head]).returncode==0,'BASE_HEAD')
    path='scripts/'+Path(__file__).name
    pre.need(pre.git(repo,'hash-object',str(repo/path))==pre.git(repo,'rev-parse','HEAD:'+path),
             'FULL_SCRIPT_BLOB_DRIFT')


def mean(vals: list[float]) -> float:
    pre.need(bool(vals), 'EMPTY_MEAN')
    pre.need(all(math.isfinite(float(x)) for x in vals), 'NONFINITE_METRIC')
    return sum(float(x) for x in vals)/len(vals)


def _centered_l2(a: list[float], b: list[float]) -> float:
    pre.need(len(a)==len(b)==3,'LOGIT_VECTOR_LENGTH')
    ca=sum(a)/3; cb=sum(b)/3
    return math.sqrt(sum(((x-ca)-(y-cb))**2 for x,y in zip(a,b)))


def aggregate(worker_records: list[Mapping[str,Any]]) -> dict[str,Any]:
    """All rows, original order; no row-mean of shard/batch aggregate means."""
    pre.need(len(worker_records)==2,'WORKER_COUNT')
    w0,w1=worker_records
    pre.need(w0['worker']==0 and w1['worker']==1,'WORKER_ORDER')
    pre.need(tuple(w0['row_range'])==pre.SHARDS[0] and
             tuple(w1['row_range'])==pre.SHARDS[1],'SHARD_RANGES')
    pre.need(w0['row_ids'] and w1['row_ids'],'ROW_IDS_EMPTY')
    rows=list(w0['row_ids'])+list(w1['row_ids'])
    labels=list(w0['labels'])+list(w1['labels'])
    pre.need(len(rows)==840 and len(set(rows))==840,'POPULATION_ROW_IDENTITY')
    pre.need(len(labels)==840 and all(x in (0,1,2) for x in labels),'POPULATION_LABELS')
    results={}
    for seed in pre.SEEDS:
        zero=f'{seed}/ZERO'; full=f'{seed}/FULL'
        combined={}
        for arm in pre.ARMS:
            key=f'{seed}/{arm}'
            left=w0['cells'][key]; right=w1['cells'][key]
            logits=list(left['logits'])+list(right['logits'])
            ce=list(left['ce_per_row'])+list(right['ce_per_row'])
            pre.need(len(logits)==len(ce)==840, 'CELL_COVERAGE:'+key)
            pre.need(all(len(v)==3 and all(math.isfinite(float(x)) for x in v) for v in logits),
                     'NONFINITE_LOGITS:'+key)
            pre.need(all(math.isfinite(float(x)) and x>=0 for x in ce),'CE_NONFINITE:'+key)
            combined[arm]={'logits':logits,'ce':ce}
        ce_zero=mean(combined['ZERO']['ce'])
        ce_full=mean(combined['FULL']['ce'])
        gain_full=ce_zero-ce_full
        ref_zero=combined['ZERO']['logits']
        ref_full=combined['FULL']['logits']
        seed_out={'full_gain_vs_zero':gain_full,'arms':{}}
        for arm in pre.ARMS:
            cell=combined[arm]
            logits=cell['logits']
            ce_mean=mean(cell['ce'])
            gain=ce_zero-ce_mean
            pred=[max(range(3), key=lambda j:v[j]) for v in logits]
            zero_pred=[max(range(3), key=lambda j:v[j]) for v in ref_zero]
            full_pred=[max(range(3), key=lambda j:v[j]) for v in ref_full]
            row_accuracy=mean([float(y==l) for y,l in zip(pred,labels)])
            centered_full=mean([_centered_l2(v,r) for v,r in zip(logits,ref_full)])
            centered_zero=mean([_centered_l2(v,r) for v,r in zip(logits,ref_zero)])
            seed_out['arms'][arm]={
                'mean_ce':ce_mean,
                'gain_vs_zero_ce':gain,
                'full_gain_retained_fraction':(gain/gain_full if abs(gain_full)>1e-8 else None),
                'retention_defined':abs(gain_full)>1e-8,
                'accuracy_diagnostic':row_accuracy,
                'prediction_agreement_with_full':mean([float(a==b) for a,b in zip(pred,full_pred)]),
                'prediction_agreement_with_zero':mean([float(a==b) for a,b in zip(pred,zero_pred)]),
                'mean_centered_logit_l2_to_full':centered_full,
                'mean_centered_logit_l2_to_zero':centered_zero,
            }
        shared=combined['SHARED_ONLY']['logits']
        residual=combined['RESIDUAL_ONLY']['logits']
        # Exact downstream logit additivity diagnostic, not causal attribution.
        seed_out['nonadditivity_mean_centered_logit_l2']=mean([
            _centered_l2(
                [f-z for f,z in zip(ff,zz)],
                [s+r-2*z for s,r,z in zip(ss,rr,zz)]
            ) for ff,zz,ss,rr in zip(ref_full,ref_zero,shared,residual)
        ])
        results[str(seed)]=seed_out
    return {'row_count':len(rows),'seed_results':results,
            'scope':'FROZEN_PHASE3A_P0_DEV_ONLY','statistical_inference':'NOT_ESTABLISHED',
            'phase3b_eligibility':'BLOCKED_UNCHANGED',
            'interpretation_status':'PENDING_IMPORT_AND_INDEPENDENT_STATIC_REVIEW'}


def one_worker(args:argparse.Namespace)->None:
    repo=args.repo.resolve(strict=True); wid=args.worker
    pre.need(wid in (0,1),'WORKER')
    pre.need(os.environ.get('CUDA_VISIBLE_DEVICES')==str(wid),'GPU_MAPPING')
    pre.need(pre.git(repo,'rev-parse','HEAD')==args.expected_head,'WORKER_HEAD')
    pre.need(args.snapshot.name==pre.MODEL_REVISION and args.snapshot.is_dir(),'SNAPSHOT')
    pre.need(args.checkpoint.is_file() and pre.sha_file(args.checkpoint)==pre.PARENT_SHA,'CHECKPOINT_SHA')
    for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
        os.environ[k]='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'))
    import torch
    import torch.nn.functional as F
    import importlib.metadata as md
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts.gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb import load_real_sources
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases,parent_parameter_fingerprint,Phase2Layer22MixerWrapper,phase2_active_mask)
    from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
        installed_g2_loaded_parent,g2_companion_active_mask)
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
        validate_frozen_sources,select_frozen_coefficients)

    pre.need(sys.version_info[:3]==(3,12,13),'PYTHON_VERSION')
    pre.need(torch.__version__=='2.10.0+cu128' and torch.version.cuda=='12.8','TORCH_CUDA_VERSION')
    pre.need(md.version('transformers')=='5.0.0' and md.version('kernels')=='0.10.2' and
             md.version('einops')=='0.8.1','PACKAGE_VERSIONS')
    pre.need(torch.cuda.device_count()==1 and torch.cuda.is_available(),'VISIBLE_T4_COUNT')
    pre.need(torch.cuda.get_device_name(0)=='Tesla T4' and
             torch.cuda.get_device_capability(0)==(7,5),'T4_IDENTITY')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats(0)
    device=torch.device('cuda:0')

    t0=time.perf_counter()
    static=p3a.validate_static_artifacts()
    pre.need(p3a.row_order_sha256(static['dev_rows'])==pre.DEV_ROWS_SHA,'DEV_ROW_SHA')
    encoding=p3a.load_runtime_encoding(static,args.tokenizer_snapshot)
    pre.need(encoding['dev_encoding_sha256']==pre.DEV_ENCODING_SHA,'DEV_ENCODING_SHA')
    bundle=encoding['dev_bundle']
    pre.need(len(bundle['row_ids'])==840,'DEV_ROWS')
    verified=p2train.resolve_exact_snapshot(args.snapshot)
    p2train.authenticate_checkpoint(args.checkpoint)
    runtime,kernel_compat,_backend=p2train.validate_cuda_runtime()
    def no_download(**_kw):raise RuntimeError('BLOCKED:NETWORK_KERNEL_DOWNLOAD')
    kernel_compat._hub_snapshot_download=no_download
    kernels=kernel_compat.load_exact_fast_kernels()
    pre.need(kernels['transport_identity_status']=='EXACT_FROZEN_BINARY_SHA256_MATCH','FAST_KERNELS')
    with torch.no_grad():
        model,calls=p2train._load_parent_model(snapshot=verified,checkpoint=args.checkpoint,
            device=device,kernel_compat=kernel_compat,kernels=kernels)
        model.eval().requires_grad_(False)
        pre.need(parent_parameter_fingerprint(model)==pre.PARENT_FINGERPRINT,'PARENT_FINGERPRINT')
        pre.need(model.mamba.config.use_cache is False and len(model.mamba.layers)==24,'MODEL_STRUCTURE')
        r22,c22,_=load_frozen_owner_bases(repo)
        sources=load_real_sources(repo,r22,c22)
        for source in sources.values():source.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(sources)
    native=model.mamba.layers[22].mixer
    torch.cuda.synchronize()
    setup_sec=time.perf_counter()-t0
    print(f'G2_FULL_WORKER_SETUP_PASS worker={wid} setup_sec={setup_sec:.3f}',flush=True)

    start,end=pre.SHARDS[wid]
    labels=bundle['model_inputs']['final_labels'][start:end].tolist()
    pre.need(all(isinstance(l,int) and 0<=l<3 for l in labels),'LABELS')
    cells={f'{seed}/{arm}':{'logits':[],'ce_per_row':[]} for seed in pre.SEEDS for arm in pre.ARMS}
    diffs={f'{seed}/{arm}':{'max_logit_abs':0.0,'max_mean_ce_abs':0.0}
           for seed in pre.SEEDS for arm in ('ZERO','FULL')}
    forward_count=0
    batch_times=[]
    total_start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    for ib,(lo,hi) in enumerate(pre.shard_batches(wid),1):
        features={k:bundle['model_inputs'][k][lo:hi].to(device)
                  for k in ('input_ids','attention_mask','claim_mask','evidence_mask')}
        ys=bundle['model_inputs']['final_labels'][lo:hi].to(device)
        attention=features['attention_mask']
        rows=hi-lo
        pre.need(tuple(attention.shape)==(rows,128),'ATTENTION_SHAPE')
        torch.cuda.synchronize()
        bt=time.perf_counter()
        with torch.inference_mode():
            bare_hidden=model.mamba(input_ids=features['input_ids']).last_hidden_state
            bare=p3a._historical_forward_from_hidden(model,features,bare_hidden)['logits'].detach()
            pre.need(tuple(bare.shape)==(rows,3) and bool(torch.isfinite(bare).all()),'BARE_LOGITS')
            forward_count+=1
            for seed in pre.SEEDS:
                reference=Phase2Layer22MixerWrapper(native_mixer=native,correction=sources[seed]).eval()
                model.mamba.layers[22].mixer=reference
                try:
                    with phase2_active_mask(model,attention):
                        hidden=model.mamba(input_ids=features['input_ids']).last_hidden_state
                    full_ref=p3a._historical_forward_from_hidden(model,features,hidden)['logits'].detach()
                    forward_count+=1
                finally:
                    model.mamba.layers[22].mixer=native
                pre.need(bool(torch.isfinite(full_ref).all()),f'FULL_REFERENCE_NONFINITE:{seed}')
                for arm in pre.ARMS:
                    with installed_g2_loaded_parent(model,source_corrections=sources,
                                                    heldout_seed=seed,arm=arm):
                        with g2_companion_active_mask(model,attention):
                            hidden=model.mamba(input_ids=features['input_ids']).last_hidden_state
                        logits=p3a._historical_forward_from_hidden(model,features,hidden)['logits'].detach()
                    forward_count+=1
                    pre.need(tuple(logits.shape)==(rows,3) and bool(torch.isfinite(logits).all()),
                             f'LOGITS_NONFINITE:{seed}:{arm}')
                    ce=F.cross_entropy(logits,ys,reduction='none')
                    pre.need(bool(torch.isfinite(ce).all()),f'CE_NONFINITE:{seed}:{arm}')
                    key=f'{seed}/{arm}'
                    cells[key]['logits'].extend(logits.cpu().tolist())
                    cells[key]['ce_per_row'].extend(ce.cpu().tolist())
                    if arm in ('ZERO','FULL'):
                        reference_logits=bare if arm=='ZERO' else full_ref
                        max_diff=float((logits-reference_logits).abs().max().item())
                        ce_ref=F.cross_entropy(reference_logits,ys,reduction='mean')
                        ce_diff=abs(float(ce.mean().item())-float(ce_ref.item()))
                        pre.need(max_diff<=pre.NUMERIC_LOGITS_ATOL and ce_diff<=pre.NUMERIC_CE_ATOL,
                                 f'NUMERIC_GATE:{seed}:{arm}:{max_diff}:{ce_diff}')
                        diffs[key]['max_logit_abs']=max(diffs[key]['max_logit_abs'],max_diff)
                        diffs[key]['max_mean_ce_abs']=max(diffs[key]['max_mean_ce_abs'],ce_diff)
                pre.need(model.mamba.layers[22].mixer is native,f'MIXER_RESTORE:{seed}')
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter()-bt)
        print(f'G2_FULL_BATCH_PASS worker={wid} batch={ib}/{len(pre.shard_batches(wid))} '
              f'rows={lo}:{hi} batch_sec={batch_times[-1]:.3f} forwards={forward_count}',flush=True)
    elapsed=time.perf_counter()-total_start
    pre.need(forward_count==len(pre.shard_batches(wid))*pre.EXPECTED_BACKBONE_FORWARDS_PER_BATCH,
             'FORWARD_COUNT')
    pre.need(parent_parameter_fingerprint(model)==pre.PARENT_FINGERPRINT,'PARENT_MUTATED')
    pre.need(model.mamba.layers[22].mixer is native,'FINAL_NATIVE_MIXER')
    for key,cell in cells.items():
        pre.need(len(cell['logits'])==len(cell['ce_per_row'])==(end-start),'CELL_ROW_COUNT:'+key)
    peak=torch.cuda.max_memory_allocated(0)
    _free,total=torch.cuda.mem_get_info(0)
    pre.need(peak<0.90*total,'VRAM_HEADROOM')
    record={
        'schema':SCHEMA_WORKER,'status':'PASS_FROZEN_P0_FULL_SCIENCE_SHARD',
        'head':args.expected_head,'worker':wid,'mapped_physical_gpu':wid,
        'shard_rows':end-start,'row_range':[start,end],
        'row_ids':list(bundle['row_ids'][start:end]),
        'pair_ids':list(bundle['pair_ids'][start:end]),
        'contrast_cell_ids':list(bundle['contrast_cell_ids'][start:end]),
        'labels':labels,'cells':cells,'numeric_reference_max_diffs':diffs,
        'seeds':list(pre.SEEDS),'arms':list(pre.ARMS),
        'coefficients':{str(seed):{arm:list(select_frozen_coefficients(seed,arm))
                               for arm in pre.ARMS} for seed in pre.SEEDS},
        'expected_population_rows':840,'dev_order_sha256':pre.DEV_ROWS_SHA,
        'dev_encoding_sha256':pre.DEV_ENCODING_SHA,'parent_sha256':pre.PARENT_SHA,
        'parent_fingerprint':pre.PARENT_FINGERPRINT,'model_revision':pre.MODEL_REVISION,
        'kernel_transport_status':kernels['transport_identity_status'],
        'kernel_constructor_calls':calls,'runtime':runtime,
        'device_name':torch.cuda.get_device_name(0),'peak_cuda_allocated_bytes':int(peak),
        'gpu_total_bytes':int(total),'setup_seconds':setup_sec,'run_seconds':elapsed,
        'batch_seconds':batch_times,'backbone_forwards':forward_count,
        'reference_gate':'EVERY_BATCH_ZERO_AND_FULL_PASS',
        'numerical_logit_atol':pre.NUMERIC_LOGITS_ATOL,
        'numerical_ce_atol':pre.NUMERIC_CE_ATOL,
        'training':False,'backward':False,'optimizer':False,'checkpoint_mutation':False,
        'confirmatory_data':False,'science_interpretation':None,
    }
    out=args.output_dir/f'worker{wid}_science.json'
    pre.put_json(out,record)
    print(f'G2_FULL_WORKER_PASS worker={wid} rows={start}:{end} '
          f'forwards={forward_count} runtime_sec={elapsed:.3f} '
          f'peak_vram_gib={peak/(2**30):.3f} result_sha256={pre.sha_file(out)}',flush=True)


def run_full(args:argparse.Namespace)->None:
    pre.need(args.authorize_frozen_p0_science,'FULL_SCIENCE_AUTHORIZATION_REQUIRED')
    repo=args.repo.resolve(strict=True)
    static_guard(repo,args.expected_head)
    root=output_contract(repo,args.output_dir)
    pre.need(args.snapshot.is_dir() and args.snapshot.name==pre.MODEL_REVISION,'SNAPSHOT')
    pre.need(args.tokenizer_snapshot.is_dir() and args.tokenizer_snapshot.name==pre.MODEL_REVISION,
             'TOKENIZER_SNAPSHOT')
    pre.need(args.checkpoint.is_file() and pre.sha_file(args.checkpoint)==pre.PARENT_SHA,
             'PARENT_CHECKPOINT_SHA')
    pre.need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1','PARENT_GPU_MAPPING')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    print('G2_FULL_DESIGN=840_FROZEN_P0_DEV_3_SEEDS_6_ARMS_2_T4',flush=True)
    print(f'G2_FULL_EXPECTED_FORWARD_COUNT={FULL_FORWARD_COUNT}',flush=True)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','worker',
         '--repo',str(repo),'--expected-head',args.expected_head,
         '--snapshot',str(args.snapshot),'--tokenizer-snapshot',str(args.tokenizer_snapshot),
         '--checkpoint',str(args.checkpoint),'--output-dir',str(args.output_dir)]
    children=[]
    start=time.perf_counter()
    try:
        for wid in (0,1):
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(wid)
            env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['HF_DATASETS_OFFLINE']='1'
            with (args.output_dir/f'worker{wid}.log').open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(cmd+['--worker',str(wid)],cwd=repo,env=env,
                                      stdout=log,stderr=subprocess.STDOUT)
            children.append((wid,proc))
        for wid,proc in children:
            rc=proc.wait()
            print(f'G2_FULL_WORKER_EXIT worker={wid} rc={rc}',flush=True)
        for wid,_ in children:
            log=(args.output_dir/f'worker{wid}.log').read_text(encoding='utf-8')
            print(f'=== G2 FULL WORKER {wid} ===\n{log[-20000:]}',flush=True)
        pre.need(all(proc.returncode==0 for _,proc in children),'WORKER_FAILURE_NO_SCIENCE')
        records=[]
        for wid,_ in children:
            path=args.output_dir/f'worker{wid}_science.json'
            rec=json.loads(path.read_text(encoding='utf-8'))
            pre.need(rec['schema']==SCHEMA_WORKER and rec['status']=='PASS_FROZEN_P0_FULL_SCIENCE_SHARD',
                     'WORKER_RESULT_SCHEMA')
            pre.need(rec['worker']==wid and rec['head']==args.expected_head and
                     tuple(rec['row_range'])==pre.SHARDS[wid], 'WORKER_IDENTITY')
            pre.need(rec['dev_order_sha256']==pre.DEV_ROWS_SHA and
                     rec['dev_encoding_sha256']==pre.DEV_ENCODING_SHA and
                     rec['parent_sha256']==pre.PARENT_SHA,'WORKER_SOURCE_IDENTITY')
            records.append(rec)
        science=aggregate(records)
        wall=time.perf_counter()-start
        summary={
            'schema':SCHEMA_SUMMARY,'status':'PASS_EXECUTION_PENDING_IMPORT_AND_INDEPENDENT_SCIENCE',
            'head':args.expected_head,'artifact_root':root,
            'population':'FROZEN_PHASE3A_P0_DEV','rows':840,
            'shards':[list(x) for x in pre.SHARDS],
            'conditions':[f'{seed}/{arm}' for seed in pre.SEEDS for arm in pre.ARMS],
            'numerical_gate':'EVERY_BATCH_ZERO_AND_FULL_PASS',
            'expected_backbone_forwards':FULL_FORWARD_COUNT,
            'observed_backbone_forwards':sum(r['backbone_forwards'] for r in records),
            'wall_seconds':wall,
            'worker_results':{str(wid):{'path':f'worker{wid}_science.json',
                    'sha256':pre.sha_file(args.output_dir/f'worker{wid}_science.json')}
                    for wid in (0,1)},
            'scientific_metrics':science,
            'training':False,'optimizer':False,'checkpoint_mutation':False,
            'scientific_conclusion':'PENDING_INDEPENDENT_VALIDATION',
            'phase3b_eligibility':'BLOCKED_UNCHANGED',
        }
        pre.need(summary['observed_backbone_forwards']==FULL_FORWARD_COUNT,'TOTAL_FORWARD_COUNT')
        pre.put_json(args.output_dir/'science_summary.json',summary)
        print('GEN5_G2_TWO_T4_FROZEN_P0_FULL_SCIENCE_EXECUTION_PASS',flush=True)
        print(f'FULL_ROWS={science["row_count"]} CONDITIONS={len(summary["conditions"])}',flush=True)
        print(f'FULL_WALL_MINUTES={wall/60:.2f}',flush=True)
        print('IMPORT_VALIDITY=NOT_YET_CHECKED; SCIENTIFIC_CONCLUSION=PENDING_INDEPENDENT_VALIDATION',flush=True)
        print(f'ARTIFACT_ROOT={root}',flush=True)
    finally:
        for _,proc in children:
            if proc.poll() is None:proc.terminate()


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('full','worker'),required=True)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--snapshot',required=True,type=Path)
    p.add_argument('--tokenizer-snapshot',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--worker',type=int,choices=(0,1))
    p.add_argument('--authorize-frozen-p0-science',action='store_true')
    args=p.parse_args()
    if args.mode=='full':pre.need(args.worker is None,'WORKER_ONLY_INTERNAL')
    else:pre.need(args.worker in (0,1) and not args.authorize_frozen_p0_science,'WORKER_ARG_CONTRACT')
    return args


if __name__=='__main__':
    args=parse_args()
    if args.mode=='full':run_full(args)
    else:one_worker(args)
