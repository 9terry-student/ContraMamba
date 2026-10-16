#!/usr/bin/env python3
"""Gen5 G2-H2 frozen matched-verb *full population* read-only observation.

Only the 120 frozen dev pairs x three matched one-token conditions x seeds
6201/6202/6203 x ZERO/FULL. Exactly 144 backbone forwards: 12 complete-pair
batches per seed/arm, each with unhooked and observed passes. No optimization,
labels, retokenization, new architecture, raw recurrent-write or Phase3B.
The observations are exploratory input contrasts, NOT layer-22 mediation.
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
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT, ROOT / 'src'):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from scripts import gen5_g2_h2_first_pair_real_parent_readonly_preflight as pflight
from scripts import gen5_g2_h2_three_condition_cpu_metrics as metrics

BASE_HEAD = 'a5a244b9ba998af90397867e1a10f25d40acdfef'
BRANCH = 'gen5-causal-role-state-ownership'
SCHEMA = 'GEN5_G2_H2_FULL_MATCHED_VERB_READONLY_OBSERVATION_V1'
FOLDER = 'gen5_g2_h2_full_matched_verb_observation_runs'
PAIRS = 120
ROWS = 360
CONDITIONS = pflight.CONDITIONS
SEEDS = pflight.SEEDS
ARMS = pflight.ARMS
FEATURES = pflight.FEATURES
PAIRS_PER_BATCH = 10
BATCH_ROWS = PAIRS_PER_BATCH * 3
BATCHES = math.ceil(PAIRS / PAIRS_PER_BATCH)
EXPECTED_FORWARDS = BATCHES * len(SEEDS) * len(ARMS) * 2
REFERENCE_ATOL = pflight.REFERENCE_ATOL


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('G2_H2_FULL_BLOCKED:' + code)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def authenticate(expected_head: str, *, clean: bool, runner_blob: bool) -> None:
    need(len(expected_head) == 40 and all(x in '0123456789abcdef' for x in expected_head),
         'EXPECTED_HEAD_FORMAT')
    need(git('rev-parse', 'HEAD') == expected_head, 'HEAD_MISMATCH')
    need(git('branch', '--show-current') in ('', BRANCH), 'BRANCH_MISMATCH')
    need(subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                         BASE_HEAD, expected_head], capture_output=True).returncode == 0,
         'PREFLIGHT_HEAD_NOT_ANCESTOR')
    if clean:
        need(not git('status', '--porcelain', '--untracked-files=all'), 'DIRTY_WORKTREE')
    if runner_blob:
        relative = 'scripts/' + Path(__file__).name
        need(git('hash-object', relative) == git('rev-parse', 'HEAD:' + relative),
             'FULL_RUNNER_NOT_COMMITTED_OR_BLOB_DRIFT')
    pflight.authenticate(expected_head, clean=False, runner_blob=True)


def output_contract(path: Path, head: str) -> str:
    need(not path.exists(), 'OUTPUT_ALREADY_EXISTS')
    resolved = path.resolve()
    need(resolved.is_relative_to(ROOT.resolve()), 'OUTPUT_OUTSIDE_REPO')
    parts = resolved.relative_to(ROOT.resolve())
    need(len(parts.parts) == 3 and parts.parts[:2] == ('reports', FOLDER),
         'OUTPUT_PATH_SCOPE')
    name = parts.parts[-1]
    need(name.startswith('g2h2-full-') and head[:7] in name and
         all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in name),
         'OUTPUT_RUN_NAME_HEAD')
    need(os.environ.get('CONTRAMAMBA_RUN_ARTIFACT_ROOT', '').replace('\\', '/') ==
         parts.as_posix(), 'RUN_ARTIFACT_ROOT_MISMATCH')
    return parts.as_posix()


def write_json_once(path: Path, record: Mapping[str, Any]) -> None:
    need(not path.exists(), 'OUTPUT_COLLISION:' + path.name)
    binary = (json.dumps(record, ensure_ascii=False, sort_keys=True,
                         indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as f:
        f.write(binary)


def reference_table(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, list[float]]], dict[str, str]]:
    """Authenticate all six historical H1 files and extract every NEG/C6 pair."""
    need(len(records) == PAIRS, 'SOURCE_PAIR_COUNT')
    summary_path = pflight.H1_ROOT / 'observation_summary.json'
    need(summary_path.is_file() and sha256_file(summary_path) == pflight.H1_SUMMARY_SHA,
         'H1_SUMMARY_SHA')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    need(summary['head'] == metrics.SOURCE_HEAD and
         summary['rows'] == 840 and summary['pairs'] == PAIRS,
         'H1_SOURCE_IDENTITY')
    need(set(summary['results']) == {f'{s}/{a}' for s in SEEDS for a in ARMS},
         'H1_SEED_ARM_SET')
    expected = {r['frozen_c6_row_id']: r['pair_id'] for r in records}
    need(len(expected) == PAIRS, 'H1_C6_ROW_UNIQUENESS')
    refs: dict[str, dict[str, list[float]]] = {}
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f'{seed}/{arm}'
            entry = summary['results'][key]
            path = pflight.H1_ROOT / entry['filename']
            need(path.is_file() and sha256_file(path) == entry['sha256'],
                 'H1_FILE_SHA:' + key)
            value = json.loads(path.read_text(encoding='utf-8'))
            need(value['seed'] == seed and value['arm'] == arm and
                 value['head'] == metrics.SOURCE_HEAD and
                 all(len(value[k]) == 840 for k in
                     ('row_ids', 'pair_ids', 'cell_ids', 'logits')),
                 'H1_FILE_SCHEMA:' + key)
            hits: dict[str, list[float]] = {}
            for rid, pid, cell, z in zip(value['row_ids'], value['pair_ids'],
                                         value['cell_ids'], value['logits']):
                if rid not in expected:
                    continue
                need(rid not in hits and expected[rid] == pid and
                     cell == 'C6_EXPLICIT_DENIAL', 'H1_C6_ROW_MATCH:' + key)
                need(len(z) == 3 and all(type(x) in (int, float) and
                     math.isfinite(x) for x in z), 'H1_C6_LOGIT_FORMAT:' + key)
                hits[rid] = [float(x) for x in z]
            need(set(hits) == set(expected), 'H1_C6_COVERAGE:' + key)
            refs[key] = {pid: hits[rid] for rid, pid in expected.items()}
            hashes[key] = entry['sha256']
    return refs, hashes


def plan(*, authenticate_source: bool = True) -> dict[str, Any]:
    root = ROOT / metrics.COHORT
    records = metrics.validate_candidate(root / 'matched_verb_candidate.jsonl',
                                         root / 'static_candidate_manifest.json')
    if authenticate_source:
        metrics.validate_original_source(ROOT, records)
    need(len(records) == PAIRS, 'POPULATION_COUNT')
    expected_names = list(CONDITIONS)
    pair_order: list[str] = []
    rows: list[dict[str, Any]] = []
    cpu_features: dict[str, list[list[int]]] = {k: [] for k in FEATURES}
    row_ids: set[str] = set()
    for record in records:
        per_pair, meta = pflight.first_pair_features(record)
        need(meta['conditions'] == expected_names and
             len(set(meta['valid_lengths'])) == 1, 'PAIR_ROW_CONDITIONS')
        pair_order.append(meta['pair_id'])
        for i, condition in enumerate(CONDITIONS):
            row = {'pair_id': meta['pair_id'], 'condition': condition,
                   'row_id': meta['row_ids'][i],
                   'frozen_c6_row_id': meta['frozen_c6_row_id'],
                   'verb_index': meta['verb_index'],
                   'length': meta['valid_lengths'][i]}
            need(row['row_id'] not in row_ids, 'DUPLICATE_ROW_ID')
            row_ids.add(row['row_id'])
            rows.append(row)
        for key in FEATURES:
            cpu_features[key].extend(per_pair[key])
    need(len(set(pair_order)) == PAIRS and len(rows) == ROWS and
         len(row_ids) == ROWS, 'FULL_POPULATION_CARDINALITY')
    need(all(len(v) == ROWS for v in cpu_features.values()), 'FEATURE_COUNT')
    refs, hashes = reference_table(records)
    return {'records': records, 'pair_order': pair_order,
            'rows': rows, 'cpu_features': cpu_features,
            'references': refs, 'h1_sha256': hashes,
            'expected_forwards': EXPECTED_FORWARDS}


def batch_slices() -> list[tuple[int, int, int, int]]:
    return [(a, min(a + PAIRS_PER_BATCH, PAIRS), 3*a,
             3*min(a + PAIRS_PER_BATCH, PAIRS))
            for a in range(0, PAIRS, PAIRS_PER_BATCH)]


def merge_arm_pair_metrics(per_arm: Mapping[str, Mapping[str, Mapping[str, Any]]],
                           pair_order: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    need(set(per_arm) == set(ARMS), 'ARM_COLLECTION')
    rows = {}
    distances: dict[str, dict[str, list[float]]] = {
        r: {c: [] for c in metrics.COMPARISONS} for r in
        ('PRE', 'VERB', 'POST8', 'EVIDENCE')
    }
    for pid in pair_order:
        need(pid in per_arm['ZERO'] and pid in per_arm['FULL'], 'PAIR_ARM_MISSING')
        z, f = per_arm['ZERO'][pid], per_arm['FULL'][pid]
        contrast = metrics.arm_interactions(z['logits'], f['logits'])
        rows[pid] = contrast
        for reg in distances:
            for comp in metrics.COMPARISONS:
                value = float(f['output']['distance_rms'][reg][comp])
                need(math.isfinite(value) and value >= 0.0, 'OUTPUT_DISTANCE_FINITE')
                distances[reg][comp].append(value)
                need(z['output']['distance_rms'][reg][comp] == 0.0,
                     'ZERO_OUTPUT_DISTANCE')
    summary = metrics.summarize_pairs(rows, expected_pairs=PAIRS)
    means = {region: {comp: sum(vals) / PAIRS for comp, vals in per.items()}
             for region, per in distances.items()}
    disposition = metrics.descriptive_disposition(summary, means)
    return rows, {'logits': summary, 'mean_output_distances': means,
                  'disposition': disposition}, disposition


def run(args: argparse.Namespace) -> None:
    need(args.authorize_g2h2_full_observation is True,
         'EXPLICIT_FULL_OBSERVATION_AUTH_REQUIRED')
    need(args.expected_head is not None and all(getattr(args, x) is not None for x in
         ('snapshot', 'tokenizer_snapshot', 'checkpoint', 'output_dir')),
         'RUNTIME_ARGS')
    authenticate(args.expected_head, clean=True, runner_blob=True)
    output_rel = output_contract(args.output_dir, args.expected_head)
    frozen = plan()  # Entire 120-pair identity/hash/reference check; torch-free.

    import torch
    from scripts import train_reason_router_gen5_phase3a_contention as p3a
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2
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
    from contramamba.gen5_g2_h2_device_output_capture_bridge import (
        H2DeviceOutputCaptureBridge, verify_h2_runtime_transfer,
    )
    need(sys.version_info[:3] == (3,12,13) and
         torch.__version__ == '2.10.0+cu128' and
         torch.version.cuda == '12.8', 'PYTHON_TORCH_CUDA_VERSIONS')
    for name, version in (('numpy','2.0.2'), ('transformers','5.0.0'),
                          ('kernels','0.10.2'), ('einops','0.8.1'),
                          ('tokenizers','0.22.2')):
        need(md.version(name) == version, 'PACKAGE_VERSION:' + name)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
         torch.cuda.get_device_name(0) == 'Tesla T4' and
         torch.cuda.get_device_capability(0) == (7,5), 'EXACT_ONE_T4')
    need(args.snapshot.is_dir() and args.tokenizer_snapshot.is_dir() and
         args.snapshot.name == pre.MODEL_REVISION and
         args.tokenizer_snapshot.name == pre.MODEL_REVISION, 'MODEL_SNAPSHOT_REVISION')
    need(args.checkpoint.is_file() and sha256_file(args.checkpoint) == pre.PARENT_SHA,
         'PARENT_CHECKPOINT_SHA')
    need(sha256_file(args.tokenizer_snapshot/'tokenizer.json') == metrics.TOK_SHA,
         'TOKENIZER_SHA')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[name] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Original NEG/C6 is the frozen H1 input, including all four masks.
    static = p3a.validate_static_artifacts()
    encoding = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)
    need(encoding['dev_encoding_sha256'] == pre.DEV_ENCODING_SHA,
         'ORIGINAL_DEV_ENCODING_SHA')
    bundle = encoding['dev_bundle']
    c6_indices: dict[str, int] = {}
    for i, (rid, pid, cell) in enumerate(zip(bundle['row_ids'], bundle['pair_ids'],
                                               bundle['contrast_cell_ids'])):
        if cell == 'C6_EXPLICIT_DENIAL':
            need(rid not in c6_indices, 'FROZEN_DEV_C6_DUPLICATE')
            c6_indices[rid] = i
    need(len(c6_indices) == PAIRS, 'FROZEN_DEV_C6_COUNT')
    for i in range(0, ROWS, 3):
        meta = frozen['rows'][i]
        j = c6_indices.get(meta['frozen_c6_row_id'])
        need(j is not None and bundle['pair_ids'][j] == meta['pair_id'],
             'H2_NEG_H1_C6_SOURCE_ROW')
        for key in FEATURES:
            expected = torch.tensor(frozen['cpu_features'][key][i],
                                    dtype=torch.long if key == 'input_ids' else torch.bool)
            need(torch.equal(expected, bundle['model_inputs'][key][j].cpu()),
                 'H2_NEG_H1_FROZEN_FEATURE_DRIFT:' + key)
    verified = p2.resolve_exact_snapshot(args.snapshot)
    p2.authenticate_checkpoint(args.checkpoint)
    runtime, compat, _ = p2.validate_cuda_runtime()
    def no_network(**_kw: Any) -> None:
        raise RuntimeError('G2_H2_FULL_BLOCKED:KERNEL_NETWORK_DOWNLOAD')
    compat._hub_snapshot_download = no_network
    kernels = compat.load_exact_fast_kernels()
    need(kernels['transport_identity_status'] ==
         'EXACT_FROZEN_BINARY_SHA256_MATCH', 'KERNEL_BINARY_IDENTITY')
    device = torch.device('cuda:0')
    with torch.no_grad():
        model, constructor_calls = p2._load_parent_model(
            snapshot=verified, checkpoint=args.checkpoint,
            device=device, kernel_compat=compat, kernels=kernels)
        model.eval().requires_grad_(False)
        need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT,
             'PARENT_FINGERPRINT')
        need(len(model.mamba.layers) == 24 and model.mamba.config.use_cache is False,
             'PARENT_ARCHITECTURE')
        r22, c22, _ = load_frozen_owner_bases(ROOT)
        donors = load_real_sources(ROOT, r22, c22)
        for donor in donors.values():
            donor.to(device=device).eval().requires_grad_(False)
        validate_frozen_sources(donors)
    native = model.mamba.layers[22].mixer
    need(not args.output_dir.exists(), 'OUTPUT_COLLISION_BEFORE_FORWARD')
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    total_forwards = 0
    files: dict[str, dict[str, str]] = {}
    by_seed: dict[str, Any] = {}
    with torch.inference_mode():
        for seed in SEEDS:
            arm_rows: dict[str, dict[str, Any]] = {}
            ref_max: dict[str, float] = {}
            for arm in ARMS:
                pair_data: dict[str, Any] = {}
                max_ref = 0.0
                start_arm = time.perf_counter()
                for b, (pair_lo, pair_hi, row_lo, row_hi) in enumerate(batch_slices(), 1):
                    cpu = {key: torch.tensor(frozen['cpu_features'][key][row_lo:row_hi],
                                            dtype=torch.long if key == 'input_ids' else torch.bool)
                           for key in FEATURES}
                    transferred = {k: v.to(device=device) for k, v in cpu.items()}
                    verify_h2_runtime_transfer(cpu, transferred, expected_device='cuda')
                    metas = frozen['rows'][row_lo:row_hi]
                    with installed_g2_loaded_parent(model, source_corrections=donors,
                                                    heldout_seed=seed, arm=arm) as wrapper:
                        with g2_companion_active_mask(model, transferred['attention_mask']):
                            plain_hidden = model.mamba(input_ids=transferred['input_ids']).last_hidden_state
                        plain_logits = p3a._historical_forward_from_hidden(
                            model, transferred, plain_hidden)['logits'].detach().cpu().clone()
                        total_forwards += 1
                        capture = H2DeviceOutputCaptureBridge(
                            wrapper, expected_output_device='cuda', arm=arm,
                            **cpu,
                            pair_ids=[x['pair_id'] for x in metas],
                            conditions=[x['condition'] for x in metas],
                            row_ids=[x['row_id'] for x in metas],
                            verb_indices=[x['verb_index'] for x in metas])
                        with g2_companion_active_mask(model, transferred['attention_mask']), capture:
                            watched_hidden = model.mamba(input_ids=transferred['input_ids']).last_hidden_state
                        watched_logits = p3a._historical_forward_from_hidden(
                            model, transferred, watched_hidden)['logits'].detach().cpu().clone()
                        total_forwards += 1
                        need(torch.equal(plain_hidden, watched_hidden),
                             f'OBSERVER_CHANGED_HIDDEN:{seed}/{arm}/{b}')
                        need(torch.equal(plain_logits, watched_logits),
                             f'OBSERVER_CHANGED_LOGITS:{seed}/{arm}/{b}')
                        need(tuple(watched_logits.shape) == (row_hi-row_lo, 3) and
                             bool(torch.isfinite(watched_logits).all()),
                             f'LOGIT_SHAPE_NONFINITE:{seed}/{arm}/{b}')
                        observed = capture.record()
                        need(not wrapper._forward_hooks and not native._forward_hooks,
                             f'HOOKS_NOT_REMOVED:{seed}/{arm}/{b}')
                        need(wrapper._phase2_active_mask is None,
                             f'ACTIVE_MASK_NOT_RESTORED:{seed}/{arm}/{b}')
                        if arm == 'ZERO':
                            need(int(torch.count_nonzero(observed.contribution).item()) == 0,
                                 f'ZERO_CORRECTION_NONZERO:{seed}/{b}')
                        for pair_idx in range(pair_hi-pair_lo):
                            slot = 3 * pair_idx
                            subset = metas[slot:slot+3]
                            pid = subset[0]['pair_id']
                            need([x['condition'] for x in subset] == list(CONDITIONS) and
                                 all(x['pair_id'] == pid for x in subset),
                                 'BATCH_PAIR_JOIN')
                            logits = {cond: watched_logits[slot+i].tolist()
                                      for i, cond in enumerate(CONDITIONS)}
                            reference_diff = pflight.check_neg_against_h1(
                                logits['NEG_DENIES'], frozen['references'][f'{seed}/{arm}'][pid])
                            max_ref = max(max_ref, reference_diff)
                            output_rows = {cond: observed.metric_row(slot+i)
                                           for i, cond in enumerate(CONDITIONS)}
                            need(all(output_rows[c]['length'] == output_rows[CONDITIONS[0]]['length']
                                     and output_rows[c]['evidence_start'] ==
                                     output_rows[CONDITIONS[0]]['evidence_start']
                                     for c in CONDITIONS), 'BATCH_OUTPUT_LENGTH')
                            outputs = {cond: {'native': row['native'],
                                              'wrapped': row['wrapped']}
                                       for cond, row in output_rows.items()}
                            readout = metrics.output_readout(
                                outputs,
                                length=output_rows[CONDITIONS[0]]['length'],
                                evidence_start=output_rows[CONDITIONS[0]]['evidence_start'],
                                verb_index=subset[0]['verb_index'], arm=arm)
                            need(pid not in pair_data, 'PAIR_DUPLICATE')
                            pair_data[pid] = {'logits': logits,
                                              'h1_neg_reference_max_abs_diff': reference_diff,
                                              'output': readout,
                                              'row_ids': [x['row_id'] for x in subset]}
                    need(model.mamba.layers[22].mixer is native,
                         f'NATIVE_MIXER_NOT_RESTORED:{seed}/{arm}/{b}')
                    print(f'G2_H2_FULL_BATCH_PASS seed={seed} arm={arm} '
                          f'batch={b}/{BATCHES} pairs={pair_lo}:{pair_hi} '
                          f'forwards={total_forwards}', flush=True)
                need(len(pair_data) == PAIRS and
                     list(pair_data) == frozen['pair_order'],
                     'POPULATION_PAIR_COMPLETENESS')
                key = f'{seed}/{arm}'
                ref_max[arm] = max_ref
                record = {
                    'schema': SCHEMA, 'status': 'PASS_FULL_READONLY_OBSERVATION_PENDING_VALIDATION',
                    'head': args.expected_head,
                    'seed': seed, 'arm': arm, 'population_pairs': PAIRS,
                    'population_rows': ROWS, 'conditions': list(CONDITIONS),
                    'pair_order': frozen['pair_order'],
                    'pairs': pair_data, 'h1_neg_reference_max_abs_diff': max_ref,
                    'h1_neg_reference_atol': REFERENCE_ATOL,
                    'h1_reference_sha256': frozen['h1_sha256'][key],
                    'h1_handoff_registration': 'DIRECT_UNREGISTERED_HASH_VERIFIED',
                    'frozen_cohort_sha256': metrics.COHORT_SHA,
                    'full_runner_sha256': sha256_file(Path(__file__).resolve()),
                    'frozen_dev_encoding_sha256': pre.DEV_ENCODING_SHA,
                    'parent_sha256': pre.PARENT_SHA,
                    'parent_fingerprint': pre.PARENT_FINGERPRINT,
                    'kernel_transport_status': kernels['transport_identity_status'],
                    'expected_forwards': BATCHES*2,
                    'actual_forwards': BATCHES*2,
                    'batch_rows': BATCH_ROWS,
                    'runtime_seconds': time.perf_counter()-start_arm,
                    'training': False, 'backward': False, 'optimizer': False,
                    'checkpoint_mutation': False, 'raw_recurrent_write': False,
                    'scientific_conclusion': None, 'phase3b': 'BLOCKED_UNCHANGED',
                }
                path = args.output_dir / f'seed{seed}_{arm.lower()}_observation.json'
                write_json_once(path, record)
                files[key] = {'filename': path.name, 'sha256': sha256_file(path)}
                arm_rows[arm] = pair_data
                print(f'G2_H2_FULL_ARM_PASS seed={seed} arm={arm} '
                      f'rows={ROWS} neg_max={max_ref} artifact_sha256={files[key]["sha256"]}',
                      flush=True)
            pair_interactions, seed_summary, disposition = merge_arm_pair_metrics(
                arm_rows, frozen['pair_order'])
            by_seed[str(seed)] = {
                'summary': seed_summary,
                'pair_arm_interactions': pair_interactions,
                'h1_neg_reference_max_abs_diff': ref_max,
            }
            print(f'G2_H2_FULL_SEED_COMPLETE seed={seed} '
                  f'direction={disposition["direction"]} '
                  f'lexical={disposition["lexical_control_dominance"]} '
                  f'output={disposition["output_selectivity_compatible"]}', flush=True)
    need(total_forwards == EXPECTED_FORWARDS, 'EXPECTED_BACKBONE_FORWARD_COUNT')
    need(parent_parameter_fingerprint(model) == pre.PARENT_FINGERPRINT, 'PARENT_MUTATED')
    need(model.mamba.layers[22].mixer is native, 'MODEL_MIXER_NOT_RESTORED')
    torch.cuda.synchronize()
    summary = {
        'schema': SCHEMA, 'status': 'PASS_FULL_READONLY_OBSERVATION_PENDING_VALIDATION',
        'head': args.expected_head, 'artifact_root': output_rel,
        'pair_order': frozen['pair_order'], 'pairs': PAIRS, 'rows_per_arm': ROWS,
        'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'arms': list(ARMS),
        'expected_backbone_forwards': EXPECTED_FORWARDS,
        'actual_backbone_forwards': total_forwards,
        'batch_pairs': PAIRS_PER_BATCH, 'batch_rows': BATCH_ROWS,
        'model_revision': pre.MODEL_REVISION, 'parent_sha256': pre.PARENT_SHA,
        'parent_fingerprint': pre.PARENT_FINGERPRINT,
        'frozen_cohort_sha256': metrics.COHORT_SHA,
        'full_runner_sha256': sha256_file(Path(__file__).resolve()),
        'frozen_manifest_sha256': metrics.MANIFEST_SHA,
        'frozen_dev_encoding_sha256': pre.DEV_ENCODING_SHA,
        'tokenizer_sha256': metrics.TOK_SHA,
        'h1_reference_file_sha256': frozen['h1_sha256'],
        'h1_handoff_registration': 'DIRECT_UNREGISTERED_HASH_VERIFIED',
        'kernel_transport_status': kernels['transport_identity_status'],
        'kernel_constructor_calls': constructor_calls,
        'runtime': runtime, 'device_name': torch.cuda.get_device_name(0),
        'result_files': files, 'per_seed': by_seed,
        'all_seeds_descriptive': {
            'direction': all(by_seed[str(s)]['summary']['disposition']['direction']
                             for s in SEEDS),
            'lexical_control_dominance': all(
                by_seed[str(s)]['summary']['disposition']['lexical_control_dominance']
                for s in SEEDS),
            'output_selectivity_compatible': all(
                by_seed[str(s)]['summary']['disposition']['output_selectivity_compatible']
                for s in SEEDS),
            'observational_selectivity_compatible_only': all(
                by_seed[str(s)]['summary']['disposition']['observational_selectivity_compatible_only']
                for s in SEEDS),
            'causal_mediation': 'NOT_ESTABLISHED',
        },
        'preflight_identity': {
            'head': BASE_HEAD,
            'pair': 'xg1_fact_9012', 'seed': 6201,
            'arms': list(ARMS), 'status': 'PASS_REPORTED_BY_USER',
            'h1_neg_zero_max': 1.9073486328125e-06,
            'h1_neg_full_max': 1.1920928955078125e-07,
            'preflight_artifact_sha256':
              'bb814ef688942d06146473e2067b6e6801e68a6be13eb64256379d04904bede0',
            'preflight_collected': False,
        },
        'runtime_seconds': time.perf_counter()-started,
        'training': False, 'backward': False, 'optimizer': False,
        'checkpoint_mutation': False, 'raw_recurrent_write': False,
        'causal_mediation': 'NOT_ESTABLISHED',
        'scientific_conclusion': None,
        'phase3b': 'BLOCKED_UNCHANGED',
    }
    path = args.output_dir / 'full_observation_summary.json'
    write_json_once(path, summary)
    print('G2_H2_FULL_MATCHED_VERB_READONLY_OBSERVATION_PASS', flush=True)
    print(f'PAIRS={PAIRS};ROWS={ROWS};SEEDS=3;ARMS=2;FORWARDS={total_forwards}', flush=True)
    print('SUMMARY_SHA256=' + sha256_file(path), flush=True)
    print('SCIENCE_INTERPRETATION=PENDING_ARTIFACT_VALIDATION;PHASE3B=BLOCKED_UNCHANGED',
          flush=True)


def selftest() -> None:
    batches = batch_slices()
    need(len(batches) == 12 and batches[0] == (0,10,0,30) and
         batches[-1] == (110,120,330,360), 'SELFTEST_BATCH_PARTITION')
    need(sum(hi-lo for lo,hi,_,_ in batches) == PAIRS and
         EXPECTED_FORWARDS == 144, 'SELFTEST_FORWARD_BUDGET')
    print('G2_H2_FULL_RUNNER_PURE_CPU_SELFTEST_PASS')
    print('TORCH_NOT_IMPORTED;MODEL_NOT_LOADED;GPU_NOT_USED;SCIENCE_RESULT=NONE')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--selftest', action='store_true')
    modes.add_argument('--static-plan', action='store_true')
    modes.add_argument('--cuda-run', action='store_true')
    parser.add_argument('--expected-head')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--tokenizer-snapshot', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--authorize-g2h2-full-observation', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        need(not args.authorize_g2h2_full_observation, 'SELFTEST_CANNOT_AUTHORIZE_GPU')
        selftest()
    elif args.static_plan:
        need(not args.authorize_g2h2_full_observation and args.expected_head is not None,
             'STATIC_MODE_ARGS')
        authenticate(args.expected_head, clean=False, runner_blob=True)
        p = plan()
        print('G2_H2_FULL_CPU_STATIC_PLAN_PASS')
        print(f'PAIRS={PAIRS};ROWS={ROWS};SEEDS=3;ARMS=2;FORWARDS={p["expected_forwards"]}')
        print('ALL_120_H1_NEG_REFERENCE_HASHES=PASS;MODEL_NOT_LOADED;GPU_NOT_USED')
    else:
        run(args)


if __name__ == '__main__':
    main()
