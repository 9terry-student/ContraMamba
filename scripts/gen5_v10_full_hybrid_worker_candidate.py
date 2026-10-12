#!/usr/bin/env python3
"""Gen5 v10 full-shard HYBRID candidate, reusing the unchanged frozen worker.

Not the frozen FFT engine. No training. Independent validation remains required.
Running needs a clean, committed branch and explicit operator GPU approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'gen5-causal-role-state-ownership'
BASE = '0dd2ba624250a3b75819d6fb34c231042d46a767'
SCHEMA = 'GEN5_V10_FULL_HYBRID_WORKER_CANDIDATE_V1'
ENGINE = 'GEN5_V10_HYBRID_96_CHUNK_GEMM_WITH_FROZEN_TAIL8_CANDIDATE'
FROZEN = {
    'scripts/audit_native_recurrent_generator_anchor_pair_localization.py': 'f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635',
    'scripts/gen5_v10_hybrid_fine_pair_candidate.py': '56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0',
    'scripts/gen5_v10_shared_dyadic_gemm_candidate.py': '9003fb9bd23b2a6680ef210a51845058622badd2',
    'scripts/gen5_v10_bounded_runtime_bridge_candidate.py': '7c134614d0e9b4f35e0af70e058a7d184c6332ca',
    'scripts/gen5_v10_single_group_cuda_runner_candidate.py': '78543c5af6f87f16500fdf66ad2e72f9faece449',
}

class FullBlocked(RuntimeError):
    pass


def need(value: bool, reason: str) -> None:
    if not value:
        raise FullBlocked('GEN5_V10_FULL_BLOCKED:' + reason)


def git(*args: str) -> str:
    r = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                       text=True, timeout=40, check=False)
    need(r.returncode == 0, 'GIT:' + args[0])
    return r.stdout.strip()


def verify_source(head: str) -> None:
    need(len(head) == 40 and all(c in '0123456789abcdef' for c in head), 'HEAD_FORMAT')
    need(git('rev-parse', 'HEAD') == head, 'HEAD_MISMATCH')
    need(git('branch', '--show-current') == BRANCH, 'BRANCH_MISMATCH')
    need(git('status', '--porcelain=v1', '--untracked-files=all') == '', 'DIRTY_WORKTREE')
    need(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', BASE, head],
                        check=False, capture_output=True).returncode == 0, 'BASE_ANCESTRY')
    for name, sha in FROZEN.items():
        actual = git('rev-parse', 'HEAD:' + name)
        if sha is not None:
            need(actual == sha, 'FROZEN_BLOB:' + name)
    own = Path(__file__).resolve()
    need(own == (ROOT / 'scripts' / Path(__file__).name).resolve(), 'SCRIPT_PATH')
    raw = own.read_bytes()
    sha = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    need(git('rev-parse', 'HEAD:scripts/' + own.name) == sha, 'UNCOMMITTED_SCRIPT_BYTES')


def install_reducer(frozen: Any, hybrid: Any, bridge: Any,
                    torch: Any, tally: dict[str, Any]) -> None:
    """Only replace frozen per-group reducer; preserve every upstream guard."""
    need(callable(frozen._component_fine_pair_stats), 'FROZEN_REDUCER')
    need(tally['calls'] == 0, 'ALREADY_STARTED')
    view = SimpleNamespace(
        p3a=frozen.p3a,
        PARENT_PAIRS=frozen.PARENT_PAIRS,
        PARENT_PAIR_NAMES=frozen.PARENT_PAIR_NAMES,
        FINE_CLASSES=frozen.FINE_CLASSES,
        FINE_PAIR_NAMES=frozen.FINE_PAIR_NAMES,
        FAST_FLOAT32_HALF_SPAN_LIMIT=frozen.FAST_FLOAT32_HALF_SPAN_LIMIT,
        FFT_CHUNK_WIDTH=frozen.FFT_CHUNK_WIDTH,
        _fine_pair_gap_interference_fft=frozen._fine_pair_gap_interference_fft,
    )
    need(frozen.FFT_CHUNK_WIDTH == 256, 'CHUNK_WIDTH')

    def candidate(*, component, log_a, survival, attention_mask,
                  role_masks, fine_masks, shape):
        k,b,seq,w = component.shape
        need((k,seq,w) == (4,128,24576) and b in (8,32), 'PRODUCTION_SHAPE')
        need(component.is_cuda and component.dtype == torch.float32, 'CUDA_FP32')
        start = time.perf_counter_ns()
        with torch.no_grad():
            coarse_rows = frozen.coarse._component_region_pair_stats(
                component=component, log_a=log_a, survival=survival,
                attention_mask=attention_mask, role_masks=role_masks, shape=shape)
            fine_gap, span, dispatch = hybrid.hybrid_fine_pair_gap(
                view, component=component, log_a=log_a, survival=survival,
                attention_mask=attention_mask, fine_masks=fine_masks,
                shape=shape, synthetic_cpu_geometry=False)
            rows = bridge._post_original_reducer(
                frozen, coarse_rows, fine_gap, span, component)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns()-start)/1e9
        need(type(rows) is list and len(rows)==4, 'REDUCER_K4')
        need(dispatch['batch_rows']==b and dispatch['total_chunks']==96,'DISPATCH_96')
        if b == 32:
            need(dispatch['gemm_chunks']+dispatch['fast_chunks']==96 and
                 dispatch['frozen_tail_calls']==0,'FULL_DISPATCH')
        else:
            need(dispatch['frozen_tail_calls']==1 and
                 dispatch['fast_chunks']==0 and dispatch['gemm_chunks']==0,'TAIL_DISPATCH')
        tally['calls']+=1
        tally['b32_calls']+=(b==32)
        tally['b8_calls']+=(b==8)
        tally['gemm_chunks']+=int(dispatch['gemm_chunks'])
        tally['fft_fast_chunks']+=int(dispatch['fast_chunks'])
        tally['frozen_tail_calls']+=int(dispatch['frozen_tail_calls'])
        tally['reducer_seconds']+=elapsed
        tally['max_reducer_seconds']=max(tally['max_reducer_seconds'],elapsed)
        tally['peak_allocated_bytes']=max(tally['peak_allocated_bytes'],
                                          int(torch.cuda.max_memory_allocated(0)))
        if tally['calls']%18==0 or b==8:
            print(f"GEN5_V10_FULL_GROUP_PROGRESS worker={tally['worker_id']} "
                  f"groups={tally['calls']}/{tally['expected_groups']} "
                  f"batch_rows={b} group_sec={elapsed:.3f}",flush=True)
        return rows
    frozen._component_fine_pair_stats = candidate


def tally_start(worker_id: int) -> dict[str, Any]:
    need(worker_id in (0,1),'WORKER_ID')
    return {'worker_id':worker_id, 'expected_groups':234 if worker_id==0 else 252,
            'calls':0, 'b32_calls':0, 'b8_calls':0, 'gemm_chunks':0,
            'fft_fast_chunks':0,'frozen_tail_calls':0, 'reducer_seconds':0.,
            'max_reducer_seconds':0., 'peak_allocated_bytes':0}


def verify_tally(tally: dict[str, Any]) -> None:
    i=tally['worker_id']
    need(tally['calls']==tally['expected_groups'],'GROUP_TOTAL')
    need(tally['b32_calls']==234 and tally['b8_calls']==(0 if i==0 else 18),
         'BATCH_GROUP_SHAPES')
    need(tally['gemm_chunks']+tally['fft_fast_chunks']==234*96,
         'FULL_CHUNK_COVERAGE')
    need(tally['frozen_tail_calls']==(0 if i==0 else 18),'TAIL_COVERAGE')


def write_json_with_sha(path: Path, obj: dict[str,Any]) -> None:
    need(not path.exists() and not path.with_suffix(path.suffix+'.sha256').exists(),
         'OUTPUT_COLLISION')
    raw=(json.dumps(obj, sort_keys=True, separators=(',',':'),allow_nan=False)+'\n').encode()
    with path.open('xb') as stream: stream.write(raw)
    with path.with_suffix(path.suffix+'.sha256').open('x',encoding='ascii') as stream:
        stream.write(hashlib.sha256(raw).hexdigest()+'\n')


def run_worker(args: argparse.Namespace) -> None:
    need(args.approval=='I_APPROVE_GEN5_V10_FULL_GPU' and
         os.environ.get('CONTRAMAMBA_GEN5_V10_FULL_GPU_APPROVED')=='YES',
         'EXPLICIT_GPU_APPROVAL_REQUIRED')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1'), 'ONE_MAPPED_T4')
    verify_source(args.expected_head)
    need(args.implementation_freeze_commit==args.expected_head, 'FREEZE_HEAD_MISMATCH')
    need(args.worker_id in (0,1), 'WORKER_ID')
    need(args.scratch_root.is_absolute() and not args.scratch_root.is_symlink(), 'SCRATCH_PATH')
    need(not args.scratch_root.resolve().is_relative_to(ROOT.resolve()), 'SCRATCH_INSIDE_REPO')
    folder=args.scratch_root / f'worker{args.worker_id}'
    need(not folder.exists(), 'WORKER_COLLISION')
    for p in (args.checkpoint,args.model_snapshot,args.tokenizer_snapshot):
        need(p.exists(), 'INPUT_MISSING:'+str(p))
    from scripts import audit_native_recurrent_generator_anchor_pair_localization as frozen
    from scripts import gen5_v10_hybrid_fine_pair_candidate as hybrid
    from scripts import gen5_v10_bounded_runtime_bridge_candidate as bridge
    from scripts import gen5_v10_stage_e2_cpu_source_byte_verifier_candidate as e2
    import torch
    need(torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_GPU_REQUIRED')
    need(torch.cuda.get_device_name(0)=='Tesla T4','T4_REQUIRED')
    free,total = torch.cuda.mem_get_info(0)
    need(free>=12*1024**3,'START_GPU_FREE_LT12GIB')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    e2.verify_checkpoint_file(args.checkpoint)
    e2.verify_snapshot_bytes(args.model_snapshot)
    need(args.tokenizer_snapshot==args.model_snapshot,'TOKENIZER_SNAPSHOT_IDENTITY')
    frozen._validate_runtime_head(args)
    frozen.validate_static_contract()
    tally=tally_start(args.worker_id)
    # This is explicitly NOT the frozen FFT engine. Intercept only payload write
    # to replace the misleading frozen engine label before it reaches disk.
    original_write=frozen._atomic_write
    def attributed_write(path: Path, raw: bytes) -> None:
        if path.name == 'worker_result.json' and path.parent==folder:
            obj=json.loads(raw)
            verify_tally(tally)
            need(obj['worker_id']==args.worker_id and
                 obj['execution_head']==args.expected_head,'PAYLOAD_IDENTITY')
            need(obj['orientation_count']==36,'ORIENTATION_COUNT')
            obj['schema_version']=SCHEMA
            obj['execution_engine']=ENGINE
            obj['v10_full_runtime']={**tally,
                'engine_type':'V10_HYBRID_CANDIDATE_NOT_FROZEN_FFT',
                'scientific_conclusion':'PENDING_INDEPENDENT_VALIDATION',
                'original_worker_source_blob':FROZEN[
                    'scripts/audit_native_recurrent_generator_anchor_pair_localization.py'],
                'source_head':args.expected_head}
            raw=frozen.canonical_json_bytes(obj)
        return original_write(path,raw)
    frozen._atomic_write=attributed_write
    install_reducer(frozen,hybrid,bridge,torch,tally)
    started=time.monotonic()
    try:
        frozen.run_worker(args)
        verify_tally(tally)
        result=folder/'worker_result.json'
        sidecar=result.with_suffix('.sha256')
        need(result.is_file() and sidecar.is_file(),'RESULT_NOT_WRITTEN')
        need(hashlib.sha256(result.read_bytes()).hexdigest()==sidecar.read_text().strip(),
             'RESULT_SHA_MISMATCH')
        row=json.loads(result.read_text())
        need(row['schema_version']==SCHEMA and row['execution_engine']==ENGINE,
             'ENGINE_ATTRIBUTION')
        print(f'GEN5_V10_FULL_WORKER_CANDIDATE_COMPLETED_NOT_SCIENCE_PASS '
              f'worker={args.worker_id} groups={tally["calls"]} '
              f'elapsed_sec={time.monotonic()-started:.3f}',flush=True)
    except BaseException as exc:
        folder.mkdir(parents=True,exist_ok=True)
        receipt={'schema_version':SCHEMA,'run_status':'FAILED_NOT_SCIENCE_PASS',
                 'head':args.expected_head,'worker':args.worker_id,'tally':tally,
                 'failure_kind':type(exc).__name__,
                 'failure_detail':str(exc)[:1200]}
        write_json_with_sha(folder/'v10_full_failure.json',receipt)
        raise
    finally:
        frozen._atomic_write=original_write


def inspect_worker(args: argparse.Namespace) -> None:
    verify_source(args.expected_head)
    need(args.scratch_root.is_absolute(), 'SCRATCH_ABSOLUTE')
    import json
    totals=[]
    for worker in (0,1):
        p=args.scratch_root/f'worker{worker}'/'worker_result.json'
        h=p.with_suffix('.sha256')
        need(p.is_file() and h.is_file(),'WORKER_MISSING:'+str(worker))
        need(hashlib.sha256(p.read_bytes()).hexdigest()==h.read_text().strip(),
             'WORKER_SHA:'+str(worker))
        row=json.loads(p.read_text())
        need(row['schema_version']==SCHEMA and row['execution_engine']==ENGINE,
             'ENGINE_MISMATCH')
        need(row['v10_full_runtime']['source_head']==args.expected_head, 'SOURCE_HEAD_MISMATCH')
        need(row['execution_head']==args.expected_head and row['worker_id']==worker,
             'WORKER_IDENTITY')
        tally=row['v10_full_runtime'];verify_tally(tally)
        need(row['orientation_count']==36 and len(row['orientation_accumulators'])==36,
             'ORIENTATIONS')
        totals.append(row)
    need(sum(r['row_count'] for r in totals)==840, 'DEV_ROWS')
    need(sum(r['valid_token_count'] for r in totals)==60094, 'ACTIVE_TOKENS')
    print('GEN5_V10_FULL_BOTH_WORKERS_ARTIFACT_INTEGRITY_PASS')
    print('GROUPS=486;DEV_ROWS=840;TOKENS=60094;SCIENTIFIC_VALIDATION=PENDING')


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--static-check',action='store_true')
    modes.add_argument('--run-worker',action='store_true')
    modes.add_argument('--inspect-worker-results',action='store_true')
    p.add_argument('--expected-head')
    p.add_argument('--implementation-freeze-commit')
    p.add_argument('--approval')
    p.add_argument('--model-snapshot',type=Path)
    p.add_argument('--tokenizer-snapshot',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--scratch-root',type=Path)
    p.add_argument('--worker-id',type=int)
    return p


def main() -> int:
    a=parser().parse_args()
    if a.static_check:
        need(os.environ.get('CUDA_VISIBLE_DEVICES','') in ('','-1'),'STATIC_GPU_OFF')
        need(not any((a.expected_head,a.implementation_freeze_commit,a.approval,a.model_snapshot,
                      a.tokenizer_snapshot,a.checkpoint,a.scratch_root,a.worker_id is not None)),
             'STATIC_EXTRA_ARGS')
        need(len(FROZEN)==5 and ENGINE!='','STATIC_CONSTANTS')
        print('GEN5_V10_FULL_HYBRID_STATIC_PASS;GPU=OFF;NOT_EXECUTED')
        return 0
    need(a.expected_head is not None,'EXPECTED_HEAD_REQUIRED')
    if a.inspect_worker_results:
        need(a.scratch_root is not None,'SCRATCH_REQUIRED')
        inspect_worker(a)
    else:
        need(a.implementation_freeze_commit is not None and a.approval is not None and
             a.scratch_root is not None and a.worker_id is not None and
             all((a.model_snapshot,a.tokenizer_snapshot,a.checkpoint)), 'ARGS_REQUIRED')
        run_worker(a)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
