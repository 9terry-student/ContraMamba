#!/usr/bin/env python3
"""Gen5 G2 frozen shared/residual science: two-T4 numerical preflight only.

Each T4 runs one authentic 32-row batch with the same 18-arm reference
path intended for the scientific run. It validates ZERO versus bare parent,
FULL versus frozen Phase2 reference, hashes all three donor correction sources,
and measures synchronized CUDA throughput/peak memory for a prospective 840-row
science run. It never extrapolates a scientific conclusion from 64 probe rows.

Preflight execution is explicit; this file has NO scientific-full-run mode.
No optimizer, training, backward, mutable checkpoints or confirmatory data.
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
from typing import Any

BRANCH = 'gen5-causal-role-state-ownership'
BASE_HEAD = '3960eff3fd4e0ea4c4e337a5a4907d9eaf7ea7bc'
SHARDS = ((0, 416), (416, 840))
BATCH_ROWS = 32
ARMS = ('ZERO','FULL','SHARED_ONLY','RESIDUAL_ONLY','SHARED_NORMMATCH','RESIDUAL_NORMMATCH')
SEEDS = (6201,6202,6203)
PARENT_SHA = '1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f'
PARENT_FINGERPRINT = 'b0b62100691686acf2597159add2a9e252a757f63f61ef52115136d9cf33a3c9'
DEV_ROWS_SHA = 'b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25'
DEV_ENCODING_SHA = 'e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51'
MODEL_REVISION = '40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37'
EXPECTED_CELLS = 18
EXPECTED_BACKBONE_FORWARDS_PER_BATCH = 1 + 3 + EXPECTED_CELLS
NUMERIC_LOGITS_ATOL = 1e-5
NUMERIC_CE_ATOL = 1e-6
SCHEMA_WORKER = 'GEN5_G2_TWO_GPU_NUMERICAL_PREFLIGHT_WORKER_V1'
SCHEMA_SUMMARY = 'GEN5_G2_TWO_GPU_NUMERICAL_PREFLIGHT_SUMMARY_V1'
OUTPUT_FOLDER = 'reports/reason_router_gen5_g2_science_preflight_runs'


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('BLOCKED:' + code)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):
            h.update(data)
    return h.hexdigest()


def put_json(path: Path, data: dict[str,Any]) -> None:
    raw = (json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
    need(not path.exists(), 'OUTPUT_COLLISION:'+str(path))
    temp = path.with_name('.'+path.name+'.tmp')
    need(not temp.exists(), 'TEMP_COLLISION:'+str(temp))
    with temp.open('xb') as f:
        f.write(raw)
    temp.rename(path)


def shard_batches(worker: int, batch_rows: int = BATCH_ROWS) -> tuple[tuple[int,int],...]:
    need(worker in (0,1),'WORKER')
    a,b=SHARDS[worker]
    return tuple((p,min(p+batch_rows,b)) for p in range(a,b,batch_rows))


def estimate_seconds(worker_times: dict[int,dict[str,float]]) -> dict[str,Any]:
    """Actually measured concurrent two-T4 32-row benchmark extrapolation.

    Lower uses row-linear last partial batch; upper bills a full batch for every
    remainder plus 30% allowance for variance/IO. No claimed measured full ETA.
    """
    times=[]
    for worker in (0,1):
        row=worker_times[worker]
        need(row['benchmark_sec']>0 and row['setup_sec']>=0, 'TIME_INPUT')
        n=SHARDS[worker][1]-SHARDS[worker][0]
        q,r=divmod(n,BATCH_ROWS)
        lower=row['setup_sec'] + row['benchmark_sec']*(q + r/BATCH_ROWS)
        upper=row['setup_sec'] + 1.30*row['benchmark_sec']*(q + (r>0))
        times.append({'worker':worker,'rows':n,'batches':q+(r>0),
                      'measured_b32_sec':row['benchmark_sec'],
                      'measured_setup_sec':row['setup_sec'],
                      'projected_lower_sec':lower,'projected_conservative_sec':upper})
    return {
        'assumption':'two concurrent independent T4s, same full reference path, B32; warm/cold variability unmeasured',
        'method':'lower: batch_time*(full_batches+tail_fraction)+setup; conservative: 1.3*batch_time*ceil(batches)+setup',
        'per_gpu':times,
        'parallel_wall_lower_sec':max(x['projected_lower_sec'] for x in times),
        'parallel_wall_conservative_sec':max(x['projected_conservative_sec'] for x in times),
        'not_a_measured_full_run':True,
    }


def output_contract(repo:Path,output_dir:Path)->str:
    need(not output_dir.exists(),'OUTPUT_DIR_EXISTS')
    resolved=output_dir.resolve()
    root=repo.resolve()
    need(resolved.is_relative_to(root),'OUTPUT_OUTSIDE_REPO')
    rel=resolved.relative_to(root)
    need(len(rel.parts)==3 and rel.parts[0]=='reports' and rel.parts[1]==OUTPUT_FOLDER.split('/')[-1], 'OUTPUT_PATH_SCOPE')
    run=rel.parts[2]
    need(bool(run) and all(c.islower() or c.isdigit() or c=='-' for c in run),'RUN_NAME_SYNTAX')
    env=os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT','').replace('\\','/')
    need(env==rel.as_posix(),'RUN_ARTIFACT_ROOT_MISMATCH')
    return rel.as_posix()


def static_guard(repo:Path,head:str)->None:
    need(git(repo,'rev-parse','HEAD')==head,'HEAD')
    need(git(repo,'branch','--show-current') in ('',BRANCH),'BRANCH')
    need(not git(repo,'status','--porcelain'),'DIRTY_WORKTREE')
    need(subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',BASE_HEAD,head]).returncode==0,'BASE')
    # The existing identity validator also verifies 10 frozen blobs and strict-load evidence.
    sys.path.insert(0,str(repo))
    from scripts.gen5_g2_real_parent_install_restore_only_d1174fc import check_sources_static
    check_sources_static(repo,head)
    rel='scripts/'+Path(__file__).name
    need(git(repo,'hash-object',str(repo/rel))==git(repo,'rev-parse','HEAD:'+rel),'SCRIPT_DISK_BLOB_DRIFT')


def _run_one_worker(args:argparse.Namespace)->None:
    """Executed only as an orchestrated subprocess on one mapped GPU."""
    repo=args.repo.resolve(strict=True)
    wid=args.worker
    need(wid in (0,1),'WORKER')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')==str(wid),'GPU_MAPPING')
    need(git(repo,'rev-parse','HEAD')==args.expected_head,'WORKER_HEAD')
    # The orchestrator ran static_guard before any output was written.
    need(args.snapshot.name==MODEL_REVISION and args.snapshot.is_dir(),'SNAPSHOT')
    need(args.checkpoint.is_file() and sha_file(args.checkpoint)==PARENT_SHA,'CHECKPOINT_SHA')
    for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
        os.environ[k]='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    sys.path.insert(0,str(repo))
    sys.path.insert(0,str(repo/'src'))
    import torch
    import torch.nn.functional as F
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts.gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb import load_real_sources
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases,parent_parameter_fingerprint, Phase2Layer22MixerWrapper,phase2_active_mask
    )
    from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
        installed_g2_loaded_parent,g2_companion_active_mask
    )
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
        validate_frozen_sources, select_frozen_coefficients
    )
    import importlib.metadata as md
    need(sys.version_info[:3]==(3,12,13),'PYTHON')
    need(torch.__version__=='2.10.0+cu128','TORCH')
    need(torch.version.cuda=='12.8','CUDA_VERSION')
    need(md.version('transformers')=='5.0.0','TRANSFORMERS')
    need(md.version('kernels')=='0.10.2','KERNELS')
    need(md.version('einops')=='0.8.1','EINOPS')
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'VISIBLE_T4_COUNT')
    need(torch.cuda.get_device_name(0)=='Tesla T4' and torch.cuda.get_device_capability(0)==(7,5),'T4_IDENTITY')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats(0)
    time0=time.perf_counter()
    static=p3a.validate_static_artifacts()
    need(len(static['dev_rows'])==840,'DEV_ROWS')
    need(p3a.row_order_sha256(static['dev_rows'])==DEV_ROWS_SHA,'DEV_ROW_SHA')
    encoded=p3a.load_runtime_encoding(static,args.tokenizer_snapshot)
    need(encoded['dev_encoding_sha256']==DEV_ENCODING_SHA,'DEV_ENCODING_SHA')
    bundle=encoded['dev_bundle']
    need(len(bundle['row_ids'])==840,'ENCODING_ROW_COUNT')
    verified_snapshot=p2train.resolve_exact_snapshot(args.snapshot)
    p2train.authenticate_checkpoint(args.checkpoint)
    runtime,kernel_compat,_backend=p2train.validate_cuda_runtime()
    def no_download(**_kw):raise RuntimeError('BLOCKED:NETWORK_KERNEL_DOWNLOAD')
    kernel_compat._hub_snapshot_download=no_download
    kernels=kernel_compat.load_exact_fast_kernels()
    need(kernels['transport_identity_status']=='EXACT_FROZEN_BINARY_SHA256_MATCH','FAST_KERNELS')
    with torch.no_grad():
        model,calls=p2train._load_parent_model(snapshot=verified_snapshot,checkpoint=args.checkpoint,
                   device=device,kernel_compat=kernel_compat,kernels=kernels)
        model.eval().requires_grad_(False)
        need(parent_parameter_fingerprint(model)==PARENT_FINGERPRINT,'PARENT_FINGERPRINT')
        need(model.mamba.config.use_cache is False and len(model.mamba.layers)==24,'MODEL_STRUCTURE')
        r22,c22,_=load_frozen_owner_bases(repo)
        sources=load_real_sources(repo,r22,c22)
        for source in sources.values():source.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(sources)
    torch.cuda.synchronize()
    setup_sec=time.perf_counter()-time0
    print(f'G2_GPU_PREFLIGHT_SETUP_PASS worker={wid} setup_sec={setup_sec:.3f}',flush=True)
    start,stop=shard_batches(wid)[0]
    need(stop-start==BATCH_ROWS,'BATCH_SIZE')
    fields=('input_ids','attention_mask','claim_mask','evidence_mask')
    features={k:bundle['model_inputs'][k][start:stop].to(device) for k in fields}
    labels=bundle['model_inputs']['final_labels'][start:stop].to(device)
    mask=features['attention_mask']
    need(tuple(mask.shape)==(BATCH_ROWS,128),'ATTENTION_MASK_SHAPE')
    need(tuple(labels.shape)==(BATCH_ROWS,),'LABELS')
    native=model.mamba.layers[22].mixer
    observed_logits={}
    ref_diffs={}
    forward_count=0
    tstart=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        # Native model is reference ZERO; same full frozen end-to-end readout.
        y=model.mamba(input_ids=features['input_ids']).last_hidden_state
        bare_logits=p3a._historical_forward_from_hidden(model,features,y)['logits'].detach()
        need(tuple(bare_logits.shape)==(BATCH_ROWS,3),'BARE_LOGIT_SHAPE')
        need(bool(torch.isfinite(bare_logits).all()),'BARE_NONFINITE')
        forward_count+=1
        for seed in SEEDS:
            reference=Phase2Layer22MixerWrapper(native_mixer=native,correction=sources[seed]).eval()
            model.mamba.layers[22].mixer=reference
            try:
                with phase2_active_mask(model,mask):
                    yref=model.mamba(input_ids=features['input_ids']).last_hidden_state
                full_ref=p3a._historical_forward_from_hidden(model,features,yref)['logits'].detach()
                forward_count+=1
            finally:
                model.mamba.layers[22].mixer=native
            need(not torch.is_grad_enabled(),'GRAD_ENABLED')
            for arm in ARMS:
                with installed_g2_loaded_parent(model,source_corrections=sources,heldout_seed=seed,arm=arm):
                    with g2_companion_active_mask(model,mask):
                        yo=model.mamba(input_ids=features['input_ids']).last_hidden_state
                    logits=p3a._historical_forward_from_hidden(model,features,yo)['logits'].detach()
                forward_count+=1
                need(bool(torch.isfinite(logits).all()),f'NONFINITE:{seed}:{arm}')
                ce=F.cross_entropy(logits,labels,reduction='mean')
                need(bool(torch.isfinite(ce)),f'CE_NONFINITE:{seed}:{arm}')
                observed_logits[f'{seed}/{arm}']={'ce_diagnostic':float(ce.item()),
                    'mean_logits':[float(x) for x in logits.mean(dim=0).cpu().tolist()]}
                if arm=='ZERO':
                    maxdiff=float((logits-bare_logits).abs().max().item())
                    cediff=abs(float(ce.item())-float(F.cross_entropy(bare_logits,labels).item()))
                    need(maxdiff<=NUMERIC_LOGITS_ATOL and cediff<=NUMERIC_CE_ATOL,
                         f'ZERO_REFERENCE_DIFF:{seed}:{maxdiff}:{cediff}')
                    ref_diffs[f'{seed}/ZERO']={'max_logit_abs':maxdiff,'ce_abs':cediff}
                if arm=='FULL':
                    maxdiff=float((logits-full_ref).abs().max().item())
                    cediff=abs(float(ce.item())-float(F.cross_entropy(full_ref,labels).item()))
                    need(maxdiff<=NUMERIC_LOGITS_ATOL and cediff<=NUMERIC_CE_ATOL,
                         f'FULL_REFERENCE_DIFF:{seed}:{maxdiff}:{cediff}')
                    ref_diffs[f'{seed}/FULL']={'max_logit_abs':maxdiff,'ce_abs':cediff}
            need(model.mamba.layers[22].mixer is native,f'RESTORE_NATIVE:{seed}')
    torch.cuda.synchronize()
    bench_sec=time.perf_counter()-tstart
    need(forward_count==EXPECTED_BACKBONE_FORWARDS_PER_BATCH,'FORWARD_COUNT')
    need(parent_parameter_fingerprint(model)==PARENT_FINGERPRINT,'PARENT_CHANGED')
    peak=torch.cuda.max_memory_allocated(0)
    free,total=torch.cuda.mem_get_info(0)
    need(peak<0.90*total,'GPU_MEMORY_HEADROOM')
    record={
        'schema':SCHEMA_WORKER,'status':'PASS_NUMERICAL_AND_TIMING_PREFLIGHT',
        'head':args.expected_head,'worker':wid,'mapped_physical_gpu':wid,
        'visible_gpu':'cuda:0','device_name':torch.cuda.get_device_name(0),
        'row_range':[start,stop],'frozen_population_rows':840,
        'full_shard':list(SHARDS[wid]),'batch_size':BATCH_ROWS,
        'input_row_ids':list(bundle['row_ids'][start:stop]),
        'dev_order_sha256':DEV_ROWS_SHA,'dev_encoding_sha256':DEV_ENCODING_SHA,
        'parent_sha256':PARENT_SHA,'parent_parameter_signature':PARENT_FINGERPRINT,
        'kernel_transport_status':kernels['transport_identity_status'],
        'kernel_constructor_calls':calls,'runtime':runtime,
        'seeds':list(SEEDS),'arms':list(ARMS),'arm_coefficients':{
          str(s):{arm:list(select_frozen_coefficients(s,arm)) for arm in ARMS} for s in SEEDS},
        'reference_differences':ref_diffs,'diagnostic_ce_only_not_science':observed_logits,
        'numerical_logit_atol':NUMERIC_LOGITS_ATOL,'numerical_ce_atol':NUMERIC_CE_ATOL,
        'benchmark_sec':bench_sec,'setup_sec':setup_sec,
        'peak_cuda_allocated_bytes':peak,'device_total_bytes':total,'device_free_after_bytes':free,
        'measured_backbone_forward_count':forward_count,
        'training':False,'backward':False,'optimizer':False,'confirmatory_data':False,
        'task_ce_used_only_for_numerical_validation':True,'scientific_conclusion':None,
        'full_run_performed':False,'all_18_arms_measured_on_one_batch':True,
    }
    out=args.output_dir/f'worker{wid}_preflight.json'
    put_json(out,record)
    print(f'G2_GPU_PREFLIGHT_WORKER_PASS worker={wid} rows={start}:{stop} '
          f'forwards={forward_count} benchmark_sec={bench_sec:.3f} '
          f'setup_sec={setup_sec:.3f} peak_vram_gib={peak/(1024**3):.3f}',flush=True)
    print(f'RESULT_SHA256={sha_file(out)}',flush=True)


def run_preflight(args:argparse.Namespace)->None:
    need(args.authorize_gpu_preflight,'GPU_PREFLIGHT_EXPLICIT_AUTHORIZATION')
    repo=args.repo.resolve(strict=True)
    static_guard(repo,args.expected_head)
    root=output_contract(repo,args.output_dir)
    need(args.snapshot.is_dir() and args.snapshot.name==MODEL_REVISION,'MODEL_SNAPSHOT')
    need(args.tokenizer_snapshot.is_dir() and args.tokenizer_snapshot.name==MODEL_REVISION,'TOKENIZER_SNAPSHOT')
    need(args.checkpoint.is_file() and sha_file(args.checkpoint)==PARENT_SHA,'PARENT_CHECKPOINT_SHA')
    # Both processes start in parallel, one bound to physical T4 0, one to T4 1.
    # No subprocess mutates tracked files. Independent worker outputs are disjoint.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    print('EXPECTED_PREFLIGHT_FORWARD_COUNTS=worker0:22,worker1:22,total:44',flush=True)
    print('REFERENCE_FULL_FORWARD_PROJECTION=worker0:286,worker1:308,total:594',flush=True)
    print('GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP',flush=True)
    cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','worker',
         '--repo',str(repo),'--expected-head',args.expected_head,
         '--snapshot',str(args.snapshot),'--tokenizer-snapshot',str(args.tokenizer_snapshot),
         '--checkpoint',str(args.checkpoint),'--output-dir',str(args.output_dir)]
    children=[]
    try:
        for wid in (0,1):
            env=os.environ.copy()
            env['CUDA_VISIBLE_DEVICES']=str(wid)
            env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['HF_DATASETS_OFFLINE']='1'
            with (args.output_dir/f'worker{wid}.log').open('x', encoding='utf-8') as log_fh:
                p=subprocess.Popen(cmd+['--worker',str(wid)],cwd=repo,env=env,
                    stdout=log_fh, stderr=subprocess.STDOUT)
            children.append((wid,p))
        for wid,p in children:
            rc=p.wait()
            print(f'WORKER_EXIT_CODE worker={wid} rc={rc}',flush=True)
        for wid,_p in children:
            print(f'=== WORKER {wid} LOG ===',flush=True)
            print((args.output_dir/f'worker{wid}.log').read_text(encoding='utf-8')[-12000:],flush=True)
        need(all(p.returncode==0 for _,p in children),'WORKER_FAILURE_NO_FULL_RUN')
        rows={}
        for wid,_p in children:
            path=args.output_dir/f'worker{wid}_preflight.json'
            rec=json.loads(path.read_text(encoding='utf-8'))
            need(rec['schema']==SCHEMA_WORKER and rec['status']=='PASS_NUMERICAL_AND_TIMING_PREFLIGHT','WORKER_REPORT')
            need(rec['worker']==wid and rec['head']==args.expected_head,'WORKER_PROVENANCE')
            need(tuple(rec['row_range'])==shard_batches(wid)[0],'PREFLIGHT_ROW_RANGE')
            need(rec['measured_backbone_forward_count']==EXPECTED_BACKBONE_FORWARDS_PER_BATCH,'WORKER_FORWARD_COUNT')
            rows[wid]=rec
        eta=estimate_seconds(rows)
        summary={'schema':SCHEMA_SUMMARY,'status':'PASS_TWO_GPU_NUMERICAL_PREFLIGHT_ESTIMATE_ONLY',
          'head':args.expected_head,'gpu_topology':'TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP',
          'full_population_size':840,'full_batch_size':32,
          'conditions':18,'total_expected_backbone_forwards':sum(len(shard_batches(w))*EXPECTED_BACKBONE_FORWARDS_PER_BATCH for w in (0,1)),
          'bench_worker_files':{str(w):{'path':f'worker{w}_preflight.json',
                         'sha256':sha_file(args.output_dir/f'worker{w}_preflight.json')} for w in (0,1)},
          'projection':eta,'full_run_executed':False,'scientific_conclusion':None}
        put_json(args.output_dir/'preflight_eta_summary.json',summary)
        print('GEN5_G2_TWO_GPU_NUMERICAL_PREFLIGHT_PASS',flush=True)
        print(f'EXPECTED_FULL_BACKBONE_FORWARDS={summary["total_expected_backbone_forwards"]}',flush=True)
        print(f'FULL_RUN_ESTIMATE_LOWER_MIN={eta["parallel_wall_lower_sec"]/60:.2f}',flush=True)
        print(f'FULL_RUN_ESTIMATE_CONSERVATIVE_MIN={eta["parallel_wall_conservative_sec"]/60:.2f}',flush=True)
        print('ETA=EXTRAPOLATED_FROM_ONE_B32_PER_GPU_NOT_MEASURED_FULL_RUN',flush=True)
        print('FULL_RUN=NOT_EXECUTED; TRAINING=NO; OPTIMIZER=NO',flush=True)
        print(f'ARTIFACT_ROOT={root}',flush=True)
    finally:
        for _,p in children:
            if p.poll() is None:p.terminate()


def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('preflight','worker'))
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--tokenizer-snapshot',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker',type=int,choices=(0,1))
    parser.add_argument('--authorize-gpu-preflight',action='store_true')
    args=parser.parse_args()
    if args.mode=='preflight':need(args.worker is None,'WORKER_ONLY_FOR_INTERNAL')
    else:need(args.worker in (0,1) and not args.authorize_gpu_preflight,'INTERNAL_WORKER_ARGS')
    return args


if __name__=='__main__':
    run_args=parse_args()
    if run_args.mode=='preflight':run_preflight(run_args)
    else:_run_one_worker(run_args)
