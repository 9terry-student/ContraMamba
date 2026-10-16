"""Gen5 G2-H1 frozen-P0 full-population read-only layer-22 observation.

Scope: 120 frozen dev pairs x 7 cells x 3 heldout seeds x ZERO/FULL.
One Tesla T4; every 32-row batch has paired unobserved/observed forwards.
No training, backward, fresh data, retokenization, coefficient search or causal
intervention. The observer captures *mixer output difference*, not raw SSM
recurrent state. Execution remains pending independent artifact validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get('CONTRAMAMBA_REPO_ROOT', '/kaggle/working/ContraMamba')).resolve()
for _directory in (ROOT, ROOT / 'src'):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

BASE_HEAD = '40a279a28ac088b8728f828038127343f598fc9e'
BRANCH = 'gen5-causal-role-state-ownership'
FOLDER = 'gen5_g2_h1_full_readonly_observation_runs'
CELLS = ('C0_SHAM', 'C1_TITLE', 'C2_NAME', 'C3_ROLE', 'C4_PREDICATE',
         'C5_TITLE_NAME', 'C6_EXPLICIT_DENIAL')
SEEDS = (6201, 6202, 6203)
ARMS = ('ZERO', 'FULL')
ROWS = 840
PAIRS = 120
BATCH = 32
EXPECTED_BATCHES = math.ceil(ROWS / BATCH)
EXPECTED_FORWARDS = EXPECTED_BATCHES * len(SEEDS) * len(ARMS) * 2
SCHEMA = 'GEN5_G2_H1_FULL_POPULATION_READONLY_OBSERVATION_V1'
REFERENCE_SCHEMA = 'GEN5_G2_H1_FULL_POPULATION_READONLY_REFERENCES_V1'


def need(ok: bool, what: str) -> None:
    if not ok:
        raise RuntimeError('G2_H1_FULL_BLOCKED:' + what)


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identity(pair_ids: list[str], cells: list[str],
                      row_ids: list[str]) -> list[str]:
    need(len(pair_ids) == len(cells) == len(row_ids) == ROWS,
         'POPULATION_LENGTH')
    need(len(set(row_ids)) == ROWS and all(row_ids), 'ROW_IDS')
    grouped: dict[str, set[str]] = defaultdict(set)
    first: list[str] = []
    for pid, cell in zip(pair_ids, cells):
        need(isinstance(pid, str) and bool(pid) and cell in CELLS,
             'PAIR_CELL_IDENTITY')
        if pid not in grouped:
            first.append(pid)
        need(cell not in grouped[pid], 'DUPLICATE_PAIR_CELL')
        grouped[pid].add(cell)
    need(len(first) == PAIRS, 'PAIR_COUNT')
    need(all(len(grouped[pid]) == 7 and grouped[pid] == set(CELLS)
             for pid in first), 'INCOMPLETE_PAIR')
    return first


def authenticate_repo(expected_head: str) -> None:
    need(len(expected_head) == 40 and
         all(c in '0123456789abcdef' for c in expected_head),
         'EXPECTED_HEAD_FORMAT')
    need(git('rev-parse', 'HEAD') == expected_head, 'HEAD_MISMATCH')
    need(git('branch', '--show-current') in ('', BRANCH), 'BRANCH')
    need(not git('status', '--porcelain'), 'DIRTY_WORKTREE')
    need(subprocess.run(['git', '-C', str(ROOT), 'merge-base',
                         '--is-ancestor', BASE_HEAD, expected_head],
                        check=False).returncode == 0, 'BASE_ANCESTRY')
    runner = Path(__file__).resolve()
    need(not runner.is_relative_to(ROOT), 'EXTERNAL_RUNNER_REQUIRED')
    pinned_sha = os.environ.get('CONTRAMAMBA_G2H1_RUNNER_SHA256', '')
    need(len(pinned_sha) == 64 and
         all(char in '0123456789abcdef' for char in pinned_sha),
         'RUNNER_SHA256_PIN_REQUIRED')
    need(sha256_file(runner) == pinned_sha, 'RUNNER_SHA256_MISMATCH')
    from scripts import gen5_g2_h1_real_parent_readonly_preflight as h1
    h1.authenticate_repo(expected_head)


def frozen_population(expected_head: str) -> dict[str, Any]:
    from scripts import gen5_g2_h1_real_parent_readonly_preflight as h1
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    # Reuses the previously authenticated first-pair plan to establish the
    # original G2 P0 worker and source identity. Not a selected science pair.
    _ = h1.static_plan(expected_head)
    summary, row_ids, pair_ids, cells, frozen_seed6201 = h1.load_frozen_p0()
    order = validate_identity(pair_ids, cells, row_ids)
    need(summary['head'] == h1.P0_HEAD and summary['rows'] == ROWS,
         'P0_SOURCE_IDENTITY')
    workers = []
    for wid, digest in enumerate(h1.P0_HASHES):
        path = ROOT / h1.P0_ROOT / f'worker{wid}_science.json'
        need(sha256_file(path) == digest, f'P0_WORKER_HASH:{wid}')
        record = json.loads(path.read_text(encoding='utf-8'))
        need(record['head'] == h1.P0_HEAD and record['worker'] == wid,
             f'P0_WORKER_IDENTITY:{wid}')
        need(tuple(record['row_range']) == pre.SHARDS[wid],
             f'P0_SHARD_RANGE:{wid}')
        workers.append(record)
    logits: dict[str, list[list[float]]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f'{seed}/{arm}'
            values = sum((list(w['cells'][key]['logits']) for w in workers), [])
            need(len(values) == ROWS and all(len(v) == 3 and
                 all(math.isfinite(float(x)) for x in v) for v in values),
                 'P0_LOGIT_FORMAT:' + key)
            logits[key] = values
    for arm in ARMS:
        need(logits[f'6201/{arm}'] == frozen_seed6201[arm],
             'PREVIOUS_PREFLIGHT_REFERENCE_DRIFT:' + arm)
    static = p3a.validate_static_artifacts()
    need(p3a.row_order_sha256(static['dev_rows']) == pre.DEV_ROWS_SHA,
         'FROZEN_DEV_ROW_ORDER')
    need(len(static['dev_rows']) == ROWS, 'DEV_COUNT')
    for i, row in enumerate(static['dev_rows']):
        need((str(row['row_id']), str(row['source_pair_id']),
              str(row['contrast_cell_id'])) ==
             (row_ids[i], pair_ids[i], cells[i]), f'DEV_ROW_IDENTITY:{i}')
    return dict(schema=REFERENCE_SCHEMA, summary=summary, row_ids=row_ids,
                pair_ids=pair_ids, cells=cells, pair_order=order,
                logits=logits)


def output_contract(directory: Path) -> str:
    need(not directory.exists(), 'OUTPUT_ALREADY_EXISTS')
    path = directory.resolve()
    need(path.is_relative_to(ROOT.resolve()), 'OUTPUT_OUTSIDE_REPO')
    relative = path.relative_to(ROOT.resolve())
    need(len(relative.parts) == 3 and
         relative.parts[:2] == ('reports', FOLDER), 'OUTPUT_SCOPE')
    run_name = relative.parts[2]
    need(bool(run_name) and all(c.islower() or c.isdigit() or c == '-'
                                for c in run_name), 'RUN_NAME_INVALID')
    need(os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT', '').replace('\\', '/') ==
         relative.as_posix(), 'RUN_ARTIFACT_ROOT_MISMATCH')
    return relative.as_posix()


def put_json(path: Path, value: Any) -> None:
    need(not path.exists(), 'OUTPUT_COLLISION:' + str(path))
    source = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False) + '\n').encode('utf-8')
    temporary = path.with_name('.' + path.name + '.tmp')
    need(not temporary.exists(), 'TEMP_COLLISION:' + str(path))
    with temporary.open('xb') as stream:
        stream.write(source)
    temporary.rename(path)


def do_run(args: argparse.Namespace) -> None:
    need(args.authorize_frozen_observation,
         'EXPLICIT_FULL_OBSERVATION_AUTHORIZATION_REQUIRED')
    need(all(x is not None for x in (args.snapshot, args.tokenizer_snapshot,
                                     args.checkpoint, args.output_dir)),
         'RUNTIME_ASSET_ARGUMENTS_REQUIRED')
    authenticate_repo(args.expected_head)
    output_relative = output_contract(args.output_dir)
    plan = frozen_population(args.expected_head)

    import importlib.metadata as md
    import torch
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train
    from scripts import gen5_g2_scientific_cuda_two_gpu_preflight_3960eff as pre
    from scripts import gen5_g2_h1_real_parent_readonly_preflight as h1
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
        G2H1PairSummaryAccumulator, Layer22OutputObserver,
        final_logit_rs_contrasts,
    )
    need(sys.version_info[:3] == (3, 12, 13), 'PYTHON_VERSION')
    need(torch.__version__ == '2.10.0+cu128' and
         torch.version.cuda == '12.8', 'TORCH_CUDA_BUILD')
    need(all(md.version(k) == v for k, v in (
        ('numpy', '2.0.2'), ('transformers', '5.0.0'),
        ('kernels', '0.10.2'), ('einops', '0.8.1'),
    )), 'FROZEN_PACKAGE_VERSIONS')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
         torch.cuda.get_device_name(0) == 'Tesla T4' and
         torch.cuda.get_device_capability(0) == (7, 5), 'EXACT_ONE_T4_REQUIRED')
    need(args.snapshot.is_dir() and
         args.snapshot.name == pre.MODEL_REVISION and
         args.tokenizer_snapshot.is_dir() and
         args.tokenizer_snapshot.name == pre.MODEL_REVISION,
         'FROZEN_SNAPSHOT_REVISIONS')
    need(args.checkpoint.is_file() and
         sha256_file(args.checkpoint) == pre.PARENT_SHA, 'PARENT_SHA')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    static = p3a.validate_static_artifacts()
    encoding = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)
    need(encoding['dev_encoding_sha256'] == pre.DEV_ENCODING_SHA,
         'FROZEN_DEV_ENCODING')
    bundle = encoding['dev_bundle']
    need(list(bundle['row_ids']) == plan['row_ids'] and
         list(bundle['pair_ids']) == plan['pair_ids'] and
         list(bundle['contrast_cell_ids']) == plan['cells'],
         'FROZEN_ENCODED_ROW_IDENTITY')
    verified_snapshot = p2train.resolve_exact_snapshot(args.snapshot)
    p2train.authenticate_checkpoint(args.checkpoint)
    runtime, kernel_compat, _backend = p2train.validate_cuda_runtime()

    def no_download(**_kwargs: Any) -> None:
        raise RuntimeError('G2_H1_FULL_BLOCKED:NETWORK_KERNEL_DOWNLOAD')

    kernel_compat._hub_snapshot_download = no_download
    kernels = kernel_compat.load_exact_fast_kernels()
    need(kernels['transport_identity_status'] ==
         'EXACT_FROZEN_BINARY_SHA256_MATCH', 'EXACT_KERNELS')
    device = torch.device('cuda:0')
    started = time.perf_counter()
    with torch.no_grad():
        model, constructor_calls = p2train._load_parent_model(
            snapshot=verified_snapshot, checkpoint=args.checkpoint,
            device=device, kernel_compat=kernel_compat, kernels=kernels,
        )
        model.eval().requires_grad_(False)
        need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT,
             'PARENT_FINGERPRINT')
        need(len(model.mamba.layers) == 24 and
             model.mamba.config.use_cache is False, 'MODEL_ARCHITECTURE')
        r22, c22, _ = load_frozen_owner_bases(ROOT)
        donors = load_real_sources(ROOT, r22, c22)
        for correction in donors.values():
            correction.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(donors)
    native = model.mamba.layers[22].mixer
    print('G2_H1_FULL_MODEL_AUTH_PASS', flush=True)
    print(f'FULL_SCOPE=rows:{ROWS};pairs:{PAIRS};seeds:{SEEDS};arms:{ARMS};batch:{BATCH}', flush=True)
    print(f'EXPECTED_BACKBONE_FORWARDS={EXPECTED_FORWARDS}', flush=True)

    # No output directory exists until every model/input/kernel identity gate
    # has passed. Partial outputs are retained on failure, never overwritten.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_forwards = 0
    files: dict[str, Any] = {}
    per_seed_rs: dict[int, dict[str, dict[str, float]]] = {}
    with torch.inference_mode():
        for seed in SEEDS:
            per_seed_rs[seed] = {}
            for arm in ARMS:
                acc = G2H1PairSummaryAccumulator()
                row_logits: list[list[float]] = []
                row_summaries: list[dict[str, Any]] = []
                ref_max_abs = 0.0
                batches = 0
                t0 = time.perf_counter()
                for lo in range(0, ROWS, BATCH):
                    hi = min(lo + BATCH, ROWS)
                    features = {key: bundle['model_inputs'][key][lo:hi].to(device)
                                for key in ('input_ids', 'attention_mask',
                                            'claim_mask', 'evidence_mask')}
                    attention = features['attention_mask']
                    with installed_g2_loaded_parent(
                            model, source_corrections=donors,
                            heldout_seed=seed, arm=arm) as wrapper:
                        with g2_companion_active_mask(model, attention):
                            plain_hidden = model.mamba(
                                input_ids=features['input_ids']).last_hidden_state
                        plain_logits = p3a._historical_forward_from_hidden(
                            model, features, plain_hidden)['logits'].detach().cpu().clone()
                        total_forwards += 1
                        observer = Layer22OutputObserver(
                            wrapper, attention_mask=attention,
                            claim_mask=features['claim_mask'],
                            evidence_mask=features['evidence_mask'],
                            pair_ids=plan['pair_ids'][lo:hi],
                            cell_ids=plan['cells'][lo:hi],
                            row_ids=plan['row_ids'][lo:hi],
                            allow_partial_pairs=True, allow_cuda_capture=True,
                        )
                        with g2_companion_active_mask(model, attention), observer:
                            hooked_hidden = model.mamba(
                                input_ids=features['input_ids']).last_hidden_state
                        hooked_logits = p3a._historical_forward_from_hidden(
                            model, features, hooked_hidden)['logits'].detach().cpu().clone()
                        total_forwards += 1
                        observed = observer.record()
                        need(torch.equal(plain_hidden, hooked_hidden),
                             f'OBSERVER_HIDDEN_CHANGED:{seed}/{arm}:{lo}')
                        need(torch.equal(plain_logits, hooked_logits),
                             f'OBSERVER_LOGITS_CHANGED:{seed}/{arm}:{lo}')
                        need(tuple(hooked_logits.shape) == (hi - lo, 3) and
                             bool(torch.isfinite(hooked_logits).all()),
                             f'LOGIT_SHAPE:{seed}/{arm}:{lo}')
                        reference = torch.tensor(
                            plan['logits'][f'{seed}/{arm}'][lo:hi],
                            dtype=hooked_logits.dtype,
                        )
                        difference = float((hooked_logits - reference).abs().max())
                        need(difference <= pre.NUMERIC_LOGITS_ATOL,
                             f'FROZEN_LOGIT_DRIFT:{seed}/{arm}:{lo}:{difference}')
                        ref_max_abs = max(ref_max_abs, difference)
                        if arm == 'ZERO':
                            need(int(torch.count_nonzero(observed.contribution)) == 0,
                                 f'ZERO_NONZERO_WRITE:{seed}:{lo}')
                        acc.add_batch(
                            observed.per_row,
                            pair_ids=plan['pair_ids'][lo:hi],
                            cell_ids=plan['cells'][lo:hi],
                            row_ids=plan['row_ids'][lo:hi],
                        )
                        row_logits.extend(hooked_logits.tolist())
                        row_summaries.extend(observed.per_row)
                    need(model.mamba.layers[22].mixer is native,
                         f'NATIVE_MIXER_NOT_RESTORED:{seed}/{arm}:{lo}')
                    batches += 1
                    print(f'G2_H1_FULL_BATCH_PASS seed={seed} arm={arm} '
                          f'batch={batches}/{EXPECTED_BATCHES} rows={lo}:{hi} '
                          f'forwards={total_forwards}', flush=True)
                need(batches == EXPECTED_BATCHES and
                     len(row_logits) == len(row_summaries) == ROWS,
                     f'ROW_OR_BATCH_COUNT:{seed}/{arm}')
                pair_stats = acc.finalize(expected_pairs=PAIRS,
                                          require_all_seven_cells=True)
                pair_rs = final_logit_rs_contrasts(
                    torch.tensor(row_logits, dtype=torch.float64),
                    plan['pair_ids'], plan['cells'],
                )
                need(set(pair_rs) == set(plan['pair_order']) == set(pair_stats),
                     f'PAIR_KEYS:{seed}/{arm}')
                key = f'{seed}/{arm}'
                per_seed_rs[seed][arm] = pair_rs
                record = {
                    'schema': SCHEMA, 'status': 'PASS_READONLY_OBSERVATION',
                    'head': args.expected_head,
                    'external_runner_sha256': sha256_file(Path(__file__).resolve()),
                    'seed': seed, 'arm': arm,
                    'population_rows': ROWS, 'population_pairs': PAIRS,
                    'row_ids': plan['row_ids'], 'pair_ids': plan['pair_ids'],
                    'cell_ids': plan['cells'], 'pair_order': plan['pair_order'],
                    'frozen_p0_head': plan['summary']['head'],
                    'frozen_worker_hashes': list(h1.P0_HASHES),
                    'frozen_dev_encoding_sha256': pre.DEV_ENCODING_SHA,
                    'parent_sha256': pre.PARENT_SHA,
                    'parent_fingerprint': pre.PARENT_FINGERPRINT,
                    'model_revision': pre.MODEL_REVISION,
                    'kernel_transport_status': kernels['transport_identity_status'],
                    'reference_max_abs_diff': ref_max_abs,
                    'reference_logit_atol': pre.NUMERIC_LOGITS_ATOL,
                    'expected_forwards': 2 * EXPECTED_BATCHES,
                    'observed_forwards': 2 * batches,
                    'pair_d_rs': pair_rs,
                    'pair_region_differences': pair_stats,
                    'per_row': row_summaries,  # CPU summaries, not CUDA tensors
                    'logits': row_logits,
                    'runtime_seconds': time.perf_counter() - t0,
                    'training': False, 'backward': False,
                    'checkpoint_mutation': False, 'intervention': False,
                    'scientific_conclusion': None,
                }
                path = args.output_dir / f'seed{seed}_{arm.lower()}_observation.json'
                put_json(path, record)
                files[key] = {'filename': path.name, 'sha256': sha256_file(path)}
                print(f'G2_H1_FULL_CONDITION_PASS seed={seed} arm={arm} '
                      f'rows={ROWS} pairs={PAIRS} sha256={files[key]["sha256"]}',
                      flush=True)
    need(total_forwards == EXPECTED_FORWARDS, 'FORWARD_COUNT')
    need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT,
         'PARENT_MUTATED')
    need(model.mamba.layers[22].mixer is native, 'NATIVE_NOT_RESTORED')
    torch.cuda.synchronize()
    contrasts: dict[str, Any] = {}
    for seed in SEEDS:
        zero, full = per_seed_rs[seed]['ZERO'], per_seed_rs[seed]['FULL']
        delta = {p: full[p] - zero[p] for p in plan['pair_order']}
        contrasts[str(seed)] = {
            'zero_pair_mean_d_rs': sum(zero.values()) / PAIRS,
            'full_pair_mean_d_rs': sum(full.values()) / PAIRS,
            'full_minus_zero_pair_mean_d_rs': sum(delta.values()) / PAIRS,
            'full_minus_zero_pair_d_rs': delta,
        }
    summary = {
        'schema': SCHEMA, 'status': 'PASS_PENDING_INDEPENDENT_IMPORT_AND_SCIENCE',
        'head': args.expected_head, 'artifact_root': output_relative,
        'external_runner_sha256': sha256_file(Path(__file__).resolve()),
        'population': 'FROZEN_PHASE3A_P0_DEV', 'rows': ROWS, 'pairs': PAIRS,
        'seeds': list(SEEDS), 'arms': list(ARMS), 'batch_rows': BATCH,
        'expected_backbone_forwards': EXPECTED_FORWARDS,
        'actual_backbone_forwards': total_forwards,
        'results': files, 'final_logit_contrasts_descriptive': contrasts,
        'frozen_dev_encoding_sha256': pre.DEV_ENCODING_SHA,
        'parent_sha256': pre.PARENT_SHA,
        'parent_fingerprint': pre.PARENT_FINGERPRINT,
        'kernel_transport_status': kernels['transport_identity_status'],
        'kernel_constructor_calls': constructor_calls,
        'runtime': runtime, 'device_name': torch.cuda.get_device_name(0),
        'wall_seconds': time.perf_counter() - started,
        'training': False, 'backward': False, 'optimizer': False,
        'checkpoint_mutation': False, 'raw_recurrent_write': False,
        'causal_attribution': 'NOT_ESTABLISHED',
        'scientific_conclusion': 'PENDING_INDEPENDENT_VALIDATION',
        'phase3b_eligibility': 'BLOCKED_UNCHANGED',
    }
    final = args.output_dir / 'observation_summary.json'
    put_json(final, summary)
    print('G2_H1_FULL_READONLY_OBSERVATION_EXECUTION_PASS', flush=True)
    print(f'ROWS={ROWS};PAIRS={PAIRS};CONDITIONS={len(files)};FORWARDS={total_forwards}',
          flush=True)
    print('SCIENTIFIC_CONCLUSION=PENDING_INDEPENDENT_VALIDATION', flush=True)
    print('SUMMARY_SHA256=' + sha256_file(final), flush=True)


def selftest() -> None:
    pairs = [f'p{i}' for i in range(PAIRS) for _ in CELLS]
    cells = list(CELLS) * PAIRS
    rows = [f'r{i}' for i in range(ROWS)]
    need(validate_identity(pairs, cells, rows) ==
         [f'p{i}' for i in range(PAIRS)], 'SELFTEST_IDENTITY')
    bad = list(cells)
    bad[6] = 'C0_SHAM'
    try:
        validate_identity(pairs, bad, rows)
    except RuntimeError as exc:
        need('DUPLICATE_PAIR_CELL' in str(exc), 'SELFTEST_DUPLICATE')
    else:
        raise RuntimeError('SELFTEST_INVALID_PAIR_ACCEPTED')
    need(EXPECTED_BATCHES == 27 and EXPECTED_FORWARDS == 324,
         'SELFTEST_FORWARD_PLAN')
    print('G2_H1_FULL_STATIC_SELFTEST_PASS')
    print(f'ROWS={ROWS};PAIRS={PAIRS};CONDITIONS=6;FORWARDS={EXPECTED_FORWARDS}')
    print('GPU=NOT_USED;MODEL=NOT_LOADED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--selftest', action='store_true')
    mode.add_argument('--static-plan', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--expected-head')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--tokenizer-snapshot', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--authorize-frozen-observation', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        need(not args.authorize_frozen_observation, 'SELFTEST_GPU_FLAG')
        selftest()
    elif args.static_plan:
        need(args.expected_head is not None and
             not args.authorize_frozen_observation, 'STATIC_PLAN_ARGS')
        authenticate_repo(args.expected_head)
        plan = frozen_population(args.expected_head)
        print('G2_H1_FULL_STATIC_PLAN_PASS')
        print(f'ROWS={len(plan["row_ids"])};PAIRS={len(plan["pair_order"])};'
              f'CONDITIONS={len(SEEDS) * len(ARMS)};FORWARDS={EXPECTED_FORWARDS}')
        print('GPU=NOT_USED;MODEL=NOT_LOADED')
    else:
        need(args.expected_head is not None, 'EXPECTED_HEAD_REQUIRED')
        do_run(args)


if __name__ == '__main__':
    main()
