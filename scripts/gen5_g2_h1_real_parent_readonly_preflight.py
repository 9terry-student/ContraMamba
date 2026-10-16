#!/usr/bin/env python3
"""Gen5 G2-H1 first real-parent read-only observation: one frozen seven-cell pair.

Implementation/preflight only. The CUDA mode is explicitly gated; it is NOT
training, full G2 evaluation, or H1 causal/confirmatory scientific evidence.
Historical G2/P3A implementations and frozen G2-P0 results remain unchanged.
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
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

BASE_HEAD = 'f2c931a03b9103b3a5f4a7c6ed3eb75bcbb4378e'
P0_HEAD = '75ad31e42e1baed7d99afbe9281e36276dfd1992'
P0_ROOT = Path('reports/reason_router_gen5_g2_functional_science_runs/gen5-g2-p0-full-75ad31e-r1')
P0_HASHES = (
    'd0432108dfba731bc0eaf78854faa043842b5d9cebfa6183c867b6d39fd330de',
    'dd79c803dee3b5fe96822ec9d8074e24abbf6624e35bb93988571e7d55fcabc4',
)
CELLS = ('C0_SHAM', 'C1_TITLE', 'C2_NAME', 'C3_ROLE', 'C4_PREDICATE',
         'C5_TITLE_NAME', 'C6_EXPLICIT_DENIAL')
ARMS = ('ZERO', 'FULL')
SEED = 6201
RESULT_SCHEMA = 'GEN5_G2_H1_FIRST_PAIR_REAL_PARENT_READONLY_PREFLIGHT_V1'
RESULT_FOLDER = 'gen5_g2_h1_real_parent_preflight_runs'


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError('G2_H1_REAL_PREFLIGHT_BLOCKED:' + message)


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(1 << 20), b''):
            digest.update(piece)
    return digest.hexdigest()


def select_first_complete_pair(
    pair_ids: Sequence[str], cells: Sequence[str], row_ids: Sequence[str]
) -> tuple[str, list[int]]:
    """Select first encountered pair; reject partial/duplicate cells or row IDs.

    No label/logit ranking or performance-based selection is permitted.
    """
    need(len(pair_ids) == len(cells) == len(row_ids) and len(row_ids) > 0,
         'SELECT_IDENTITY_SHAPE')
    need(all(type(p) is str and bool(p) for p in pair_ids), 'PAIR_ID_INVALID')
    need(all(type(c) is str and c in CELLS for c in cells), 'CELL_ID_INVALID')
    need(all(type(r) is str and bool(r) for r in row_ids) and
         len(set(row_ids)) == len(row_ids), 'ROW_ID_INVALID_OR_DUPLICATE')
    pair = pair_ids[0]
    indices = [i for i, p in enumerate(pair_ids) if p == pair]
    found = [cells[i] for i in indices]
    need(len(found) == len(CELLS) and set(found) == set(CELLS),
         'FIRST_PAIR_NOT_COMPLETE_SEVEN_CELLS')
    return pair, indices


def authenticate_repo(expected_head: str) -> None:
    need(len(expected_head) == 40 and all(c in '0123456789abcdef' for c in expected_head),
         'EXPECTED_HEAD_FORMAT')
    need(git('rev-parse', 'HEAD') == expected_head, 'HEAD_MISMATCH')
    need(git('branch', '--show-current') in ('', 'gen5-causal-role-state-ownership'),
         'BRANCH_MISMATCH')
    need(not git('status', '--porcelain'), 'DIRTY_WORKTREE')
    need(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                         BASE_HEAD, expected_head], check=False).returncode == 0,
         'BASE_NOT_ANCESTOR')
    relative = 'scripts/' + Path(__file__).name
    need(git('hash-object', str(ROOT / relative)) ==
         git('rev-parse', 'HEAD:' + relative), 'ENTRYPOINT_BLOB_DRIFT')
    # Historical frozen source validation, including existing G2 real-parent gate.
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    pre.static_guard(ROOT, expected_head)


def load_frozen_p0() -> tuple[dict[str, Any], list[str], list[str], list[str],
                              dict[str, list[list[float]]]]:
    base = ROOT / P0_ROOT
    summary = json.loads((base / 'science_summary.json').read_text(encoding='utf-8'))
    need(summary['head'] == P0_HEAD and summary['rows'] == 840,
         'P0_SUMMARY_IDENTITY')
    records = []
    for wid, digest in enumerate(P0_HASHES):
        path = base / f'worker{wid}_science.json'
        need(path.is_file() and sha256_file(path) == digest,
             f'P0_WORKER_SHA:{wid}')
        record = json.loads(path.read_text(encoding='utf-8'))
        need(record['head'] == P0_HEAD and record['worker'] == wid,
             f'P0_WORKER_IDENTITY:{wid}')
        need(tuple(record['row_range']) == ((0,416) if wid == 0 else (416,840)),
             f'P0_WORKER_RANGE:{wid}')
        records.append(record)
    row_ids = sum((list(r['row_ids']) for r in records), [])
    pair_ids = sum((list(r['pair_ids']) for r in records), [])
    cells = sum((list(r['contrast_cell_ids']) for r in records), [])
    need(len(row_ids) == 840 and len(set(row_ids)) == 840,
         'FROZEN_P0_ROW_IDENTITY')
    frozen: dict[str, list[list[float]]] = {}
    for arm in ARMS:
        key = f'{SEED}/{arm}'
        logits = sum((list(r['cells'][key]['logits']) for r in records), [])
        need(len(logits) == 840 and all(
            len(z) == 3 and all(math.isfinite(float(v)) for v in z) for z in logits),
            f'FROZEN_LOGIT_SHAPE:{arm}')
        frozen[arm] = logits
    return summary, row_ids, pair_ids, cells, frozen


def static_plan(expected_head: str) -> dict[str, Any]:
    authenticate_repo(expected_head)
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    _summary, rows, pairs, cells, frozen = load_frozen_p0()
    static = p3a.validate_static_artifacts()
    need(p3a.row_order_sha256(static['dev_rows']) == pre.DEV_ROWS_SHA,
         'DEV_ROW_SHA')
    need(len(static['dev_rows']) == 840, 'DEV_ROW_COUNT')
    for i, row in enumerate(static['dev_rows']):
        need(str(row['row_id']) == rows[i] and
             str(row['source_pair_id']) == pairs[i] and
             str(row['contrast_cell_id']) == cells[i],
             f'FROZEN_VS_SOURCE_ROW_IDENTITY:{i}')
    pair, indices = select_first_complete_pair(pairs, cells, rows)
    labels = [int(static['dev_rows'][i]['final_label_id']) for i in indices]
    expected_labels = {'C0_SHAM': 2, 'C6_EXPLICIT_DENIAL': 0}
    expected_labels.update({c: 1 for c in CELLS[1:-1]})
    need(all(l == expected_labels[cells[i]] for l, i in zip(labels, indices)),
         'FROZEN_LABEL_IDENTITY')
    return {
        'schema': RESULT_SCHEMA,
        'status': 'STATIC_PLAN_ONLY',
        'head': expected_head,
        'source_p0_head': P0_HEAD,
        'p0_worker_sha256': list(P0_HASHES),
        'dev_row_order_sha256': pre.DEV_ROWS_SHA,
        'dev_encoding_sha256': pre.DEV_ENCODING_SHA,
        'pair_selection': 'FIRST_PAIR_IN_FROZEN_DEV_ROW_ORDER_NO_PERFORMANCE_SELECTION',
        'pair_id': pair,
        'row_indices': indices,
        'row_ids': [rows[i] for i in indices],
        'cell_ids': [cells[i] for i in indices],
        'labels': labels,
        'heldout_seed': SEED,
        'arms': list(ARMS),
        'expected_backbone_forwards': 4,
        'frozen_logits': {arm: [frozen[arm][i] for i in indices] for arm in ARMS},
        'scope': 'ONE_PAIR_SEVEN_CELLS_READONLY_NUMERICAL_PREFLIGHT',
        'scientific_conclusion': None,
    }


def validate_output_dir(output: Path) -> str:
    need(not output.exists(), 'OUTPUT_ALREADY_EXISTS')
    path = output.resolve()
    root = ROOT.resolve()
    need(path.is_relative_to(root), 'OUTPUT_OUTSIDE_REPOSITORY')
    rel = path.relative_to(root)
    need(len(rel.parts) == 3 and rel.parts[:2] == ('reports', RESULT_FOLDER),
         'OUTPUT_SCOPE')
    run = rel.parts[-1]
    need(bool(run) and all(c.islower() or c.isdigit() or c == '-' for c in run),
         'RUN_NAME_INVALID')
    need(os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT','').replace('\\','/') ==
         rel.as_posix(), 'RUN_ARTIFACT_ROOT_MISMATCH')
    return rel.as_posix()


def cuda_preflight(args: argparse.Namespace) -> None:
    need(args.authorize_real_model_preflight, 'EXPLICIT_GPU_PREFLIGHT_FLAG_REQUIRED')
    need(args.snapshot is not None and args.tokenizer_snapshot is not None and
         args.checkpoint is not None and args.output_dir is not None,
         'RUNTIME_ASSET_ARGUMENTS_REQUIRED')
    plan = static_plan(args.expected_head)
    artifact_root = validate_output_dir(args.output_dir)
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    from scripts.gen5_g1_frozen_checkpoint_installation_cpu_7e19ecb import load_real_sources
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases, parent_parameter_fingerprint,
    )
    from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
        validate_frozen_sources,
    )
    from contramamba.gen5_shared_bypass_g2_loaded_parent_bridge import (
        installed_g2_loaded_parent, g2_companion_active_mask,
    )
    from contramamba.gen5_g2_h1_layer22_output_observer import (
        Layer22OutputObserver, final_logit_rs_contrasts,
    )
    import importlib.metadata as md
    import torch

    need(sys.version_info[:3] == (3,12,13), 'PYTHON_3_12_13_REQUIRED')
    need(torch.__version__ == '2.10.0+cu128' and torch.version.cuda == '12.8',
         'TORCH_2_10_CUDA12_8_REQUIRED')
    need(md.version('transformers') == '5.0.0' and
         md.version('kernels') == '0.10.2' and
         md.version('einops') == '0.8.1', 'PACKAGES_NOT_FROZEN')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
         torch.cuda.get_device_name(0) == 'Tesla T4' and
         torch.cuda.get_device_capability(0) == (7,5), 'ONE_VISIBLE_T4_REQUIRED')
    need(args.snapshot.is_dir() and args.snapshot.name == pre.MODEL_REVISION and
         args.tokenizer_snapshot.is_dir() and
         args.tokenizer_snapshot.name == pre.MODEL_REVISION,
         'FROZEN_SNAPSHOT_REVISIONS_REQUIRED')
    need(args.checkpoint.is_file() and
         sha256_file(args.checkpoint) == pre.PARENT_SHA, 'CHECKPOINT_SHA')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    static = p3a.validate_static_artifacts()
    encoding = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)
    need(encoding['dev_encoding_sha256'] == pre.DEV_ENCODING_SHA,
         'FROZEN_ENCODING_IDENTITY')
    bundle = encoding['dev_bundle']
    indices = plan['row_indices']
    need([bundle['row_ids'][i] for i in indices] == plan['row_ids'] and
         [bundle['pair_ids'][i] for i in indices] == [plan['pair_id']] * 7 and
         [bundle['contrast_cell_ids'][i] for i in indices] == plan['cell_ids'],
         'ENCODED_ROW_IDENTITY')
    verified = p2train.resolve_exact_snapshot(args.snapshot)
    p2train.authenticate_checkpoint(args.checkpoint)
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()
    def no_download(**_kwargs):
        raise RuntimeError('NETWORK_KERNEL_DOWNLOAD_FORBIDDEN')
    kernel_compat._hub_snapshot_download = no_download
    kernels = kernel_compat.load_exact_fast_kernels()
    need(kernels['transport_identity_status'] == 'EXACT_FROZEN_BINARY_SHA256_MATCH',
         'KERNEL_BINARY_NOT_AUTHENTICATED')
    device = torch.device('cuda:0')
    with torch.no_grad():
        model, constructor_calls = p2train._load_parent_model(
            snapshot=verified, checkpoint=args.checkpoint, device=device,
            kernel_compat=kernel_compat, kernels=kernels,
        )
        model.eval().requires_grad_(False)
        need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT,
             'PARENT_PARAMETER_FINGERPRINT')
        need(len(model.mamba.layers) == 24 and
             model.mamba.config.use_cache is False, 'MODEL_ARCHITECTURE')
        r22, c22, _ = load_frozen_owner_bases(ROOT)
        sources = load_real_sources(ROOT, r22, c22)
        for donor in sources.values():
            donor.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(sources)
    native = model.mamba.layers[22].mixer
    features = {key: bundle['model_inputs'][key][indices].to(device)
                for key in ('input_ids', 'attention_mask', 'claim_mask', 'evidence_mask')}
    attention = features['attention_mask']
    results: dict[str, Any] = {}
    total_forwards = 0
    with torch.inference_mode():
        for arm in ARMS:
            with installed_g2_loaded_parent(model, source_corrections=sources,
                                            heldout_seed=SEED, arm=arm) as wrapper:
                with g2_companion_active_mask(model, attention):
                    plain_hidden = model.mamba(input_ids=features['input_ids']).last_hidden_state
                plain_logits = p3a._historical_forward_from_hidden(
                    model, features, plain_hidden)['logits'].detach().cpu().clone()
                total_forwards += 1
                observer = Layer22OutputObserver(
                    wrapper, attention_mask=attention,
                    claim_mask=features['claim_mask'], evidence_mask=features['evidence_mask'],
                    pair_ids=[plan['pair_id']] * len(indices),
                    cell_ids=plan['cell_ids'], row_ids=plan['row_ids'],
                    allow_cuda_capture=True,
                )
                with g2_companion_active_mask(model, attention), observer:
                    hooked_hidden = model.mamba(input_ids=features['input_ids']).last_hidden_state
                hooked_logits = p3a._historical_forward_from_hidden(
                    model, features, hooked_hidden)['logits'].detach().cpu().clone()
                total_forwards += 1
                observed = observer.record()
                need(torch.equal(plain_logits, hooked_logits),
                     'OBSERVER_CHANGED_LOGITS:' + arm)
                need(torch.equal(plain_hidden, hooked_hidden),
                     'OBSERVER_CHANGED_BACKBONE_HIDDEN:' + arm)
                need(bool(torch.isfinite(hooked_logits).all()),
                     'NONFINITE_LOGITS:' + arm)
                expected_logits = torch.tensor(plan['frozen_logits'][arm],
                                               dtype=hooked_logits.dtype)
                reference_max_abs = float((hooked_logits-expected_logits).abs().max().item())
                # The original G2 numeric gate used 1e-5; never silently widen it.
                need(reference_max_abs <= pre.NUMERIC_LOGITS_ATOL,
                     'FROZEN_P0_LOGITS_MISMATCH:' + arm + ':' + str(reference_max_abs))
                pair_rs = final_logit_rs_contrasts(
                    hooked_logits, [plan['pair_id']] * 7, plan['cell_ids'])
                need(list(pair_rs) == [plan['pair_id']], 'PAIR_RS_IDENTITY')
                if arm == 'ZERO':
                    need(torch.count_nonzero(observed.contribution).item() == 0,
                         'ZERO_NONZERO_CORRECTION')
                results[arm] = {
                    'frozen_reference_max_abs_diff': reference_max_abs,
                    'observer_logit_max_abs_diff': 0.0,
                    'observer_hidden_max_abs_diff': 0.0,
                    'logits': hooked_logits.tolist(),
                    'pair_d_rs': pair_rs[plan['pair_id']],
                    'per_row': observed.per_row,
                    'pair_differences': observed.pair_differences,
                }
            need(model.mamba.layers[22].mixer is native,
                 'NATIVE_MIXER_NOT_RESTORED:' + arm)
    need(total_forwards == plan['expected_backbone_forwards'] == 4,
         'FORWARD_COUNT_NOT_FOUR')
    need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT,
         'PARENT_MUTATED_AFTER_PROBE')
    torch.cuda.synchronize()
    output = {
        **{k: v for k, v in plan.items() if k != 'frozen_logits'},
        'status': 'PASS_REAL_PARENT_READONLY_NUMERICAL_PREFLIGHT',
        'runtime': runtime,
        'kernel_transport_status': kernels['transport_identity_status'],
        'constructor_kernel_calls': constructor_calls,
        'parent_checkpoint_sha256': pre.PARENT_SHA,
        'parent_fingerprint': pre.PARENT_FINGERPRINT,
        'model_revision': pre.MODEL_REVISION,
        'actual_backbone_forwards': total_forwards,
        'device': torch.cuda.get_device_name(0),
        'results': results,
        'training': False, 'backward': False, 'checkpoint_mutation': False,
        'scientific_conclusion': None,
        'artifact_root': artifact_root,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / 'real_parent_readonly_preflight.json'
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False,
                  allow_nan=False, sort_keys=True)
        handle.write('\n')
    print('G2_H1_REAL_PARENT_READONLY_PREFLIGHT_PASS')
    print('PAIR_ID=' + plan['pair_id'])
    print('BACKBONE_FORWARDS=4')
    print('RESULT_SHA256=' + sha256_file(destination))
    print('FULL_SCIENCE=NOT_RUN; CAUSAL_ATTRIBUTION=NOT_ESTABLISHED')


def selftest() -> None:
    pairs = ('p1', 'p2', 'p1', 'p1', 'p2', 'p1', 'p1', 'p1', 'p1',
             'p2', 'p2', 'p2', 'p2', 'p2')
    cells = (CELLS[0], CELLS[0], CELLS[2], CELLS[3], CELLS[1], CELLS[4],
             CELLS[5], CELLS[6], CELLS[1], CELLS[2], CELLS[3], CELLS[4],
             CELLS[5], CELLS[6])
    ids = tuple('r' + str(i) for i in range(14))
    pair, indices = select_first_complete_pair(pairs, cells, ids)
    need(pair == 'p1' and indices == [0,2,3,5,6,7,8],
         'SELFTEST_PAIR_SELECTION')
    try:
        select_first_complete_pair(pairs, cells[:8] + ('C0_SHAM',) + cells[9:], ids)
    except RuntimeError as exc:
        need('FIRST_PAIR_NOT_COMPLETE_SEVEN_CELLS' in str(exc), 'SELFTEST_WRONG_FAILURE')
    else:
        raise RuntimeError('SELFTEST_BAD_IDENTITY_NOT_REJECTED')
    try:
        select_first_complete_pair(pairs, cells, ids[:-1] + ('r0',))
    except RuntimeError as exc:
        need('ROW_ID_INVALID_OR_DUPLICATE' in str(exc),
             'SELFTEST_DUPLICATE_ERROR')
    else:
        raise RuntimeError('SELFTEST_DUPLICATE_NOT_REJECTED')
    print('G2_H1_REAL_PARENT_STATIC_SELFTEST_PASS')
    print('GPU=NOT_USED;MODEL=NOT_LOADED;CHECKPOINT=NOT_LOADED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--selftest', action='store_true')
    group.add_argument('--static-plan', action='store_true')
    group.add_argument('--cuda-preflight', action='store_true')
    parser.add_argument('--expected-head')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--tokenizer-snapshot', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--authorize-real-model-preflight', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        need(not args.authorize_real_model_preflight, 'SELFTEST_CUDA_FLAG_FORBIDDEN')
        selftest()
    elif args.static_plan:
        need(args.expected_head is not None and
             not args.authorize_real_model_preflight, 'STATIC_PLAN_ARGUMENTS')
        plan = static_plan(args.expected_head)
        print('G2_H1_REAL_PARENT_STATIC_PLAN_PASS')
        print('PAIR_ID=' + plan['pair_id'])
        print('ROW_IDS=' + json.dumps(plan['row_ids']))
        print('ARMS=ZERO,FULL;SEED=6201;EXPECTED_FORWARDS=4')
        print('GPU=NOT_USED;MODEL=NOT_LOADED')
    else:
        need(args.expected_head is not None, 'EXPECTED_HEAD_REQUIRED')
        cuda_preflight(args)


if __name__ == '__main__':
    main()
