#!/usr/bin/env python3
"""G2-H2 single frozen dev pair real-parent numerical preflight (NOT science).

Modes:
  --selftest    pure stdlib, no model/Torch/GPU;
  --static-plan authenticate frozen cohort/XG1/H1 NEG references, no Torch/GPU;
  --cuda-run    SEPARATE authorization required. One T4, seed6201, ZERO/FULL,
                THREE matched conditions, 4 backbone forwards; no training.

No GPU observation is authorized merely by committing or running static modes.
The H2 bridge measures mixer *output*, NOT raw recurrent-write mediation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _dir in (ROOT, ROOT / 'src'):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

BASE_HEAD = 'c1cfe063302a1c223fa8ce06e6a5eb11c7a6aad9'
BRANCH = 'gen5-causal-role-state-ownership'
H1_ROOT = (ROOT / 'reports/gen5_g2_h1_full_readonly_observation_runs/'
           'g2h1-full-observation-direct-40a279a-r1')
H1_SUMMARY_SHA = '92e418f9f63c9bd0c5b8807758b50a935bf830e7ad28641bcbd0e82ab5f29202'
RESULT_FOLDER = 'gen5_g2_h2_real_parent_preflight_runs'
SCHEMA = 'GEN5_G2_H2_FIRST_PAIR_READONLY_NUMERIC_PREFLIGHT_V1'
CONDITIONS = ('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS')
SEEDS = (6201, 6202, 6203)
RUN_SEED = 6201
ARMS = ('ZERO', 'FULL')
FEATURES = ('input_ids', 'attention_mask', 'claim_mask', 'evidence_mask')
WIDTH = 128
FORWARDS = 4
REFERENCE_ATOL = 1e-5


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('G2_H2_REAL_PREFLIGHT_BLOCKED:' + code)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def authenticate(expected_head: str, *, clean: bool, runner_blob: bool) -> None:
    need(len(expected_head) == 40 and all(c in '0123456789abcdef' for c in expected_head),
         'EXPECTED_HEAD_FORMAT')
    need(git('rev-parse', 'HEAD') == expected_head, 'HEAD_MISMATCH')
    need(git('branch', '--show-current') in ('', BRANCH), 'BRANCH_MISMATCH')
    need(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                         BASE_HEAD, expected_head], capture_output=True).returncode == 0,
         'BASE_NOT_ANCESTOR')
    if clean:
        need(not git('status', '--porcelain', '--untracked-files=all'), 'DIRTY_WORKTREE')
    if runner_blob:
        relative = 'scripts/' + Path(__file__).name
        need(git('hash-object', relative) == git('rev-parse', 'HEAD:' + relative),
             'RUNNER_NOT_IN_COMMIT_OR_BLOB_DRIFT')
    for relative in (
        'src/contramamba/gen5_shared_bypass_functional_core_diagnostic.py',
        'src/contramamba/gen5_shared_bypass_g2_loaded_parent_bridge.py',
        'src/contramamba/gen5_g2_h2_device_output_capture_bridge.py',
        'src/contramamba/gen5_g2_h2_readonly_output_adapter.py',
        'scripts/gen5_g2_h2_three_condition_cpu_metrics.py',
    ):
        need(git('hash-object', relative) == git('rev-parse', 'HEAD:' + relative),
             'FROZEN_SOURCE_BLOB_DRIFT:' + relative)


def first_pair_features(pair: Mapping[str, Any]) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    """Construct original 128-wide serialized features; never add proposed labels."""
    need(set(pair['conditions']) == set(CONDITIONS), 'CONDITION_IDENTITY')
    clen = len(pair['claim_token_ids'])
    tables: dict[str, list[list[int]]] = {key: [] for key in FEATURES}
    row_ids: list[str] = []
    positions: list[int] = []
    for name in CONDITIONS:
        item = pair['conditions'][name]
        serial = item['serialized_token_ids']
        n = len(serial)
        need(clen > 0 and n <= WIDTH and n > clen + 1,
             'FROZEN_SEQUENCE_LENGTH')
        row_ids.append(pair['pair_id'] + '__g2_h2__' + name.lower())
        positions.append(item['verb_serialized_token_index'])
        tables['input_ids'].append(list(serial) + [0] * (WIDTH - n))
        tables['attention_mask'].append([1] * n + [0] * (WIDTH - n))
        tables['claim_mask'].append([1] * clen + [0] * (WIDTH - clen))
        tables['evidence_mask'].append([0] * (clen + 1) + [1] * (n-clen-1) +
                                       [0] * (WIDTH-n))
    need(len(set(positions)) == 1 and len(set(row_ids)) == 3,
         'VERB_OR_ROW_ID_ALIGNMENT')
    for key in FEATURES:
        need(len(tables[key]) == 3 and all(len(x) == WIDTH for x in tables[key]),
             'FEATURE_WIDTH_' + key)
    a,b,c = tables['input_ids']
    need(all([i for i in range(WIDTH) if x[i] != y[i]] == [positions[0]]
             for x,y in ((a,b),(a,c),(b,c))), 'NOT_ONE_TOKEN_SWAP')
    meta = {'pair_id':pair['pair_id'], 'row_ids':row_ids,
            'conditions':list(CONDITIONS), 'verb_index':positions[0],
            'frozen_c6_row_id':pair['frozen_c6_row_id'],
            'valid_lengths':[sum(x) for x in tables['attention_mask']]}
    return tables, meta


def h1_references(first: Mapping[str, Any]) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Hash-authenticate ALL six H1 files; only select first frozen C6 row."""
    path = H1_ROOT / 'observation_summary.json'
    need(path.is_file() and digest(path) == H1_SUMMARY_SHA, 'H1_SUMMARY_SHA')
    summary = json.loads(path.read_text(encoding='utf-8'))
    need(summary['head'] == '40a279a28ac088b8728f828038127343f598fc9e'
         and summary['rows'] == 840 and summary['pairs'] == 120,
         'H1_SUMMARY_SOURCE_IDENTITY')
    need(set(summary['results']) == {f'{s}/{a}' for s in SEEDS for a in ARMS},
         'H1_SEED_ARM_IDENTITY')
    refs: dict[str, list[float]] = {}
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f'{seed}/{arm}'
            desc = summary['results'][key]
            ref_path = H1_ROOT / desc['filename']
            need(ref_path.is_file() and digest(ref_path) == desc['sha256'],
                 'H1_REFERENCE_SHA:' + key)
            value = json.loads(ref_path.read_text(encoding='utf-8'))
            need(value['seed'] == seed and value['arm'] == arm and
                 len(value['row_ids']) == len(value['pair_ids']) ==
                 len(value['cell_ids']) == len(value['logits']) == 840,
                 'H1_REFERENCE_FORMAT:' + key)
            indices = [i for i, rid in enumerate(value['row_ids'])
                       if rid == first['frozen_c6_row_id']]
            need(len(indices) == 1, 'H1_C6_ROW_NOT_UNIQUE:' + key)
            j = indices[0]
            need(value['pair_ids'][j] == first['pair_id'] and
                 value['cell_ids'][j] == 'C6_EXPLICIT_DENIAL',
                 'H1_NEG_C6_IDENTITY:' + key)
            z = value['logits'][j]
            need(len(z) == 3 and all(type(n) in (int, float) and math.isfinite(n)
                                    for n in z), 'H1_NEG_REFERENCE_NUMERIC:' + key)
            refs[key] = [float(v) for v in z]
            hashes[key] = desc['sha256']
    need(len(refs) == 6, 'H1_REFERENCE_COUNT')
    return refs, hashes


def plan(*, authenticate_source: bool = True) -> dict[str, Any]:
    from scripts.gen5_g2_h2_three_condition_cpu_metrics import (
        COHORT, validate_candidate, validate_original_source,
    )
    location = ROOT / COHORT
    records = validate_candidate(location / 'matched_verb_candidate.jsonl',
                                 location / 'static_candidate_manifest.json')
    if authenticate_source:
        validate_original_source(ROOT, records)
    need(len(records) == 120, 'COHORT_COUNT')
    pair = records[0]  # FIRST frozen dev pair, no performance selection
    tables, meta = first_pair_features(pair)
    refs, hashes = h1_references(pair)
    return {'schema':SCHEMA, 'status':'CPU_STATIC_PLAN_ONLY',
            'source_candidate_sha256':'fecdf9e1e3940a3c5e9d893df4f4a810130359c00f1ece5c1c0b1599fc6127c9',
            'pair':meta, 'cpu_features':tables, 'references':refs,
            'h1_artifact_sha256':hashes, 'seed':RUN_SEED, 'arms':list(ARMS),
            'expected_forwards':FORWARDS, 'runtime_labels': 'OMITTED',
            'phase3b':'BLOCKED_UNCHANGED', 'science_result':None}


def check_neg_against_h1(z: list[float], reference: list[float],
                         *, tol:float = REFERENCE_ATOL) -> float:
    need(len(z) == len(reference) == 3 and
         all(math.isfinite(float(x)) for x in z + reference),
         'NEG_LOGIT_FORMAT')
    worst = max(abs(float(x)-float(y)) for x,y in zip(z,reference))
    need(worst <= tol, 'NEG_REFERENCE_DRIFT:' + str(worst))
    return worst


def output_dir_contract(path: Path, expected_head: str) -> str:
    need(not path.exists(), 'OUTPUT_ALREADY_EXISTS')
    resolved = path.resolve()
    need(resolved.is_relative_to(ROOT.resolve()), 'OUTPUT_OUTSIDE_REPO')
    relative = resolved.relative_to(ROOT.resolve())
    need(len(relative.parts) == 3 and relative.parts[:2] == ('reports', RESULT_FOLDER),
         'OUTPUT_SCOPE')
    name = relative.parts[2]
    need(name.startswith('g2h2-firstpair-') and expected_head[:7] in name and
         all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name),
         'RUN_NAME_OR_COMMIT')
    need(os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT','').replace('\\','/') ==
         relative.as_posix(), 'RUN_ARTIFACT_ROOT_MISMATCH')
    return relative.as_posix()


def put_json_once(path: Path, value: Mapping[str, Any]) -> None:
    need(not path.exists(), 'OUTPUT_COLLISION')
    binary = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as fh:
        fh.write(binary)


def runtime_contract(args: argparse.Namespace) -> None:
    """GPU mode only. All gates must pass before any backbone forward."""
    need(args.authorize_g2h2_numeric_preflight is True, 'EXPLICIT_GPU_AUTH_REQUIRED')
    need(all(getattr(args, n) is not None for n in
             ('snapshot','tokenizer_snapshot','checkpoint','output_dir')),
         'RUNTIME_ARGS_MISSING')
    authenticate(args.expected_head, clean=True, runner_blob=True)
    output_relative = output_dir_contract(args.output_dir, args.expected_head)
    frozen_plan = plan()

    import torch
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    from scripts.gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb import load_real_sources
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases, parent_parameter_fingerprint,
    )
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import validate_frozen_sources
    from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
        installed_g2_loaded_parent, g2_companion_active_mask,
    )
    from contramamba.gen5_g2_h2_device_output_capture_bridge import (
        H2DeviceOutputCaptureBridge, verify_h2_runtime_transfer,
    )
    from scripts.gen5_g2_h2_three_condition_cpu_metrics import output_readout, decision

    need(sys.version_info[:3] == (3,12,13), 'PYTHON_VERSION')
    need(torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8',
         'TORCH_CUDA_VERSION')
    for key,version in (('numpy','2.0.2'), ('transformers','5.0.0'),
                        ('kernels','0.10.2'), ('einops','0.8.1'),
                        ('tokenizers','0.22.2')):
        need(md.version(key) == version, 'FROZEN_PACKAGE_VERSION:'+key)
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and
         torch.cuda.get_device_name(0)=='Tesla T4' and
         torch.cuda.get_device_capability(0)==(7,5), 'EXACT_ONE_T4_REQUIRED')
    need(args.snapshot.is_dir() and args.tokenizer_snapshot.is_dir() and
         args.snapshot.name == pre.MODEL_REVISION and
         args.tokenizer_snapshot.name == pre.MODEL_REVISION,
         'FROZEN_REVISION_DIRECTORY')
    need(args.checkpoint.is_file() and digest(args.checkpoint) == pre.PARENT_SHA,
         'FROZEN_PARENT_SHA')
    tok = args.tokenizer_snapshot / 'tokenizer.json'
    need(tok.is_file() and digest(tok) ==
         'b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf',
         'FROZEN_TOKENIZER_SHA')
    for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
        os.environ[name] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Reuse H1's historical *reference* tokenizer/encoding path (not new H2
    # candidate retokenization). The NEG row must be bitwise identical.
    static = p3a.validate_static_artifacts()
    encoding = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)
    need(encoding['dev_encoding_sha256'] == pre.DEV_ENCODING_SHA,
         'FROZEN_DEV_ENCODING')
    dev = encoding['dev_bundle']
    rowid = frozen_plan['pair']['frozen_c6_row_id']
    matches = [i for i, rid in enumerate(dev['row_ids']) if rid == rowid]
    need(len(matches) == 1 and dev['pair_ids'][matches[0]] == frozen_plan['pair']['pair_id']
         and dev['contrast_cell_ids'][matches[0]] == 'C6_EXPLICIT_DENIAL',
         'FROZEN_H1_ENCODED_C6_IDENTITY')
    cpu = {key:torch.tensor(frozen_plan['cpu_features'][key],
                            dtype=torch.long if key=='input_ids' else torch.bool)
           for key in FEATURES}
    for key in FEATURES:
        need(torch.equal(cpu[key][0], dev['model_inputs'][key][matches[0]].cpu()),
             'H2_NEG_H1_INPUT_MASK_DRIFT:'+key)
    # No model run until this point. Exact frozen parent and kernel guards.
    verified = p2train.resolve_exact_snapshot(args.snapshot)
    p2train.authenticate_checkpoint(args.checkpoint)
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    def no_network(**_kw: Any) -> None:
        raise RuntimeError('G2_H2_REAL_PREFLIGHT_BLOCKED:KERNEL_NETWORK_DOWNLOAD')
    kernel_compat._hub_snapshot_download = no_network
    kernels = kernel_compat.load_exact_fast_kernels()
    need(kernels['transport_identity_status']=='EXACT_FROZEN_BINARY_SHA256_MATCH',
         'KERNEL_BINARY_IDENTITY')
    device = torch.device('cuda:0')
    with torch.no_grad():
        model, calls = p2train._load_parent_model(
            snapshot=verified, checkpoint=args.checkpoint,
            device=device, kernel_compat=kernel_compat, kernels=kernels)
        model.eval().requires_grad_(False)
        need(parent_parameter_fingerprint(model)==pre.PARENT_FINGERPRINT,
             'PARENT_FINGERPRINT')
        need(len(model.mamba.layers)==24 and model.mamba.config.use_cache is False,
             'PARENT_ARCHITECTURE')
        r22,c22,_=load_frozen_owner_bases(ROOT)
        donors=load_real_sources(ROOT,r22,c22)
        for donor in donors.values():
            donor.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(donors)
    native = model.mamba.layers[22].mixer
    transferred={key:val.to(device=device) for key,val in cpu.items()}
    verify_h2_runtime_transfer(cpu, transferred, expected_device='cuda')
    need(not args.output_dir.exists(), 'OUTPUT_COLLISION_BEFORE_FORWARD')

    answers:dict[str,Any]={}
    total=0
    with torch.inference_mode():
        for arm in ARMS:
            with installed_g2_loaded_parent(model, source_corrections=donors,
                                            heldout_seed=RUN_SEED, arm=arm) as wrapper:
                with g2_companion_active_mask(model,transferred['attention_mask']):
                    original_hidden=model.mamba(input_ids=transferred['input_ids']).last_hidden_state
                plain_logits=p3a._historical_forward_from_hidden(
                    model,transferred,original_hidden)['logits'].detach().cpu().clone()
                total+=1
                capture=H2DeviceOutputCaptureBridge(
                    wrapper, expected_output_device='cuda', arm=arm,
                    **cpu, pair_ids=[frozen_plan['pair']['pair_id']]*3,
                    conditions=list(CONDITIONS), row_ids=frozen_plan['pair']['row_ids'],
                    verb_indices=[frozen_plan['pair']['verb_index']]*3)
                with g2_companion_active_mask(model,transferred['attention_mask']), capture:
                    watched_hidden=model.mamba(input_ids=transferred['input_ids']).last_hidden_state
                watched_logits=p3a._historical_forward_from_hidden(
                    model,transferred,watched_hidden)['logits'].detach().cpu().clone()
                total+=1
                need(torch.equal(original_hidden,watched_hidden),
                     'OBSERVER_CHANGED_HIDDEN:'+arm)
                need(torch.equal(plain_logits,watched_logits),
                     'OBSERVER_CHANGED_LOGITS:'+arm)
                need(tuple(watched_logits.shape)==(3,3) and
                     bool(torch.isfinite(watched_logits).all()),
                     'LOGIT_SHAPE_NONFINITE:'+arm)
                maximum=check_neg_against_h1(
                    watched_logits[0].tolist(),
                    frozen_plan['references'][f'{RUN_SEED}/{arm}'])
                observed=capture.record()
                need(not wrapper._forward_hooks and not native._forward_hooks,
                     'HOOKS_NOT_REMOVED:'+arm)
                need(wrapper._phase2_active_mask is None,
                     'ACTIVE_MASK_NOT_RESTORED:'+arm)
                if arm=='ZERO':
                    need(torch.count_nonzero(observed.contribution).item()==0,
                         'ZERO_CORRECTION_NONZERO')
                output_rows={c:observed.metric_row(i) for i,c in enumerate(CONDITIONS)}
                outputs={c:{'native':r['native'],'wrapped':r['wrapped']}
                         for c,r in output_rows.items()}
                metrics=output_readout(
                    outputs, length=output_rows[CONDITIONS[0]]['length'],
                    evidence_start=output_rows[CONDITIONS[0]]['evidence_start'],
                    verb_index=frozen_plan['pair']['verb_index'], arm=arm)
                z={c:watched_logits[i].tolist() for i,c in enumerate(CONDITIONS)}
                answers[arm]={'neg_h1_reference_max_abs_diff':maximum,
                              'logits':z,'contrasts':decision(z),
                              'metrics':metrics,
                              'valid_lengths':[output_rows[c]['length'] for c in CONDITIONS]}
            need(model.mamba.layers[22].mixer is native,
                 'NATIVE_MIXER_NOT_RESTORED:'+arm)
    need(total==FORWARDS, 'BACKBONE_FORWARD_COUNT')
    need(parent_parameter_fingerprint(model)==pre.PARENT_FINGERPRINT,
         'PARENT_MUTATED')
    torch.cuda.synchronize()
    evidence={'schema':SCHEMA,'status':'PASS_REAL_PARENT_NUMERICAL_PREFLIGHT_ONLY',
              'head':args.expected_head,'artifact_root':output_relative,
              'pair':frozen_plan['pair'],'seed':RUN_SEED,'arms':list(ARMS),
              'model_revision':pre.MODEL_REVISION,
              'parent_sha256':pre.PARENT_SHA,'parent_fingerprint':pre.PARENT_FINGERPRINT,
              'frozen_dev_encoding_sha256':pre.DEV_ENCODING_SHA,
              'h1_reference_file_sha256':frozen_plan['h1_artifact_sha256'],
              'h1_handoff_registration':'DIRECT_UNREGISTERED_HASH_VERIFIED',
              'frozen_cohort_sha256':frozen_plan['source_candidate_sha256'],
              'tokenizer_sha256':
              'b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf',
              'runner_sha256':digest(Path(__file__).resolve()),
              'kernel_transport_status':kernels['transport_identity_status'],
              'kernel_constructor_calls':calls,'runtime':runtime,
              'device_name':torch.cuda.get_device_name(0),
              'expected_backbone_forwards':FORWARDS,'actual_backbone_forwards':total,
              'reference_atol':REFERENCE_ATOL,'results':answers,
              'training':False,'backward':False,'optimizer':False,
              'checkpoint_mutation':False,'raw_recurrent_write':False,
              'scientific_conclusion':None,'phase3b':'BLOCKED_UNCHANGED'}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    path=args.output_dir/'g2h2_firstpair_numeric_preflight.json'
    put_json_once(path,evidence)
    print('G2_H2_FIRST_PAIR_REAL_PARENT_NUMERIC_PREFLIGHT_PASS')
    print('PAIR_ID='+frozen_plan['pair']['pair_id']+';SEED=6201;ARMS=ZERO,FULL;FORWARDS=4')
    print('REFERENCE_NEG_MAX_ZERO='+str(answers['ZERO']['neg_h1_reference_max_abs_diff']))
    print('REFERENCE_NEG_MAX_FULL='+str(answers['FULL']['neg_h1_reference_max_abs_diff']))
    print('ARTIFACT_SHA256='+digest(path))
    print('SCIENCE_RESULT=NONE;PHASE3B=BLOCKED_UNCHANGED')


def selftest() -> None:
    sample = {'pair_id':'p', 'claim_token_ids':[12,13],
              'frozen_c6_row_id':'historic-rid', 'conditions':{}}
    for c,verb in (('NEG_DENIES',31647),('POS_CONFIRMS',23849),('POS_ASSERTS',17086)):
        evidence=[31,44,verb]+list(range(100,114))
        serial=[12,13,0]+evidence
        sample['conditions'][c]={'serialized_token_ids':serial,
                                 'verb_serialized_token_index':5}
    data, meta=first_pair_features(sample)
    need(set(data)==set(FEATURES) and all(len(row)==128 for v in data.values()
                                           for row in v), 'SELFTEST_SHAPE')
    need(meta['verb_index']==5 and meta['valid_lengths']==[20,20,20],
         'SELFTEST_METADATA')
    need(check_neg_against_h1([1,2,3],[1,2,3])==0,'SELFTEST_NUMERIC')
    try:
        check_neg_against_h1([1,2,3],[1,2.002,3])
    except RuntimeError as e:
        need('NEG_REFERENCE_DRIFT' in str(e),'SELFTEST_FAIL_CLOSED')
    else:
        need(False,'SELFTEST_FAIL_OPEN')
    print('G2_H2_FIRST_PAIR_RUNNER_PURE_CPU_SELFTEST_PASS')
    print('MODEL_NOT_LOADED;TORCH_NOT_IMPORTED;GPU_NOT_USED')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--selftest',action='store_true')
    modes.add_argument('--static-plan',action='store_true')
    modes.add_argument('--cuda-run',action='store_true')
    parser.add_argument('--expected-head')
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--tokenizer-snapshot',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--authorize-g2h2-numeric-preflight',action='store_true')
    args=parser.parse_args()
    if args.selftest:
        need(not args.authorize_g2h2_numeric_preflight,'SELFTEST_GPU_AUTH')
        selftest()
    elif args.static_plan:
        need(not args.authorize_g2h2_numeric_preflight and args.expected_head is not None,
             'STATIC_PLAN_ARGS')
        authenticate(args.expected_head,clean=False,runner_blob=False)
        result=plan()
        print('G2_H2_FIRST_PAIR_CPU_STATIC_PLAN_PASS')
        print('PAIR_ID='+result['pair']['pair_id']+';H2_ROWS=3;H1_NEG_REF_COUNT=6;FORWARDS=4')
        print('FROZEN_HASHES=PASS;SOURCE_IDENTITY=PASS;FEATURE_MASKS=PASS')
        print('MODEL_NOT_LOADED;TORCH_NOT_IMPORTED;GPU_NOT_USED;SCIENCE_RESULT=NONE')
    else:
        need(args.expected_head is not None,'EXPECTED_HEAD_REQUIRED')
        runtime_contract(args)


if __name__=='__main__':
    main()
