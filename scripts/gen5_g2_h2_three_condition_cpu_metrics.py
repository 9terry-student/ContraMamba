"""G2-H2 frozen matched-verb input and synthetic metric contracts (CPU only).

Not an inference runner, tokenizer, observer hook, training/evaluation command,
or GPU permission. All proposed semantic labels are metadata, never targets.
No dependencies outside Python's standard library. H1 sources are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

SOURCE_HEAD = '40a279a28ac088b8728f828038127343f598fc9e'
INPUT_COMMIT = '9c894ee22b786ad572c92d39cd0e47df3366eb80'
COHORT = ('reports/gen5_g2_h2_static_matched_verb_candidates/'
          'g2h2-static-matched-verbs-40a279a-r1')
SOURCE = 'data/reason_router_gen5_phase3_xg1_contention_training_v1'
COHORT_SHA = 'fecdf9e1e3940a3c5e9d893df4f4a810130359c00f1ece5c1c0b1599fc6127c9'
MANIFEST_SHA = 'd186781cc55749990c168b8f9f0a98186b16e8ca581d203250275b096979fb71'
SPLIT_SHA = '87d769304f2547dc81d8698ffa256476390791e3978c8a38c86e8d01232e0c5f'
SEVEN_CELL_SHA = '9850f13872739bb8099a554242df8ea177420ae754057662b2ec324cb71b8cc1'
TOK_SHA = 'b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf'
CONDITIONS = ('NEG_DENIES', 'POS_CONFIRMS', 'POS_ASSERTS')
VERBS = {'NEG_DENIES': ('denies', 31647, 'REFUTE', 0),
         'POS_CONFIRMS': ('confirms', 23849, 'SUPPORT', 2),
         'POS_ASSERTS': ('asserts', 17086, 'SUPPORT', 2)}
COMPARISONS = {'NC': ('NEG_DENIES', 'POS_CONFIRMS'),
               'NA': ('NEG_DENIES', 'POS_ASSERTS'),
               'CA': ('POS_CONFIRMS', 'POS_ASSERTS')}
ARMS = ('ZERO', 'FULL')
SEEDS = (6201, 6202, 6203)
PREFIX_ATOL = 1e-5


class ContractError(ValueError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError('G2_H2_CPU_BLOCKED:' + code)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]


def validate_candidate(cohort_file: Path, manifest_file: Path) -> list[dict[str, Any]]:
    """Authenticate bytes and three-condition serialized IDs, without retokenizing."""
    require(cohort_file.is_file() and manifest_file.is_file(), 'CANDIDATE_FILES_MISSING')
    require(digest(cohort_file) == COHORT_SHA, 'COHORT_HASH')
    require(digest(manifest_file) == MANIFEST_SHA, 'MANIFEST_HASH')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    require(manifest['schema'] == 'G2_H2_MATCHED_VERB_STATIC_CANDIDATE_V1', 'SCHEMA')
    require(manifest['source_repo_head'] == SOURCE_HEAD, 'SOURCE_HEAD')
    require(manifest['cohort_jsonl_sha256'] == COHORT_SHA, 'MANIFEST_COHORT_SHA')
    require(manifest['source_split_sha256'] == SPLIT_SHA, 'MANIFEST_SPLIT_SHA')
    require(manifest['source_seven_cell_sha256'] == SEVEN_CELL_SHA, 'MANIFEST_SOURCE_SHA')
    require(manifest['tokenizer_sha256'] == TOK_SHA and
            manifest['tokenizers_version'] == '0.22.2', 'TOKENIZER_PROVENANCE')
    require(manifest['semantics_label_status'] == 'PROPOSED_NOT_FROZEN_ANNOTATIONS',
            'LABEL_PROVENANCE')
    require(manifest['pair_count'] == 120 and manifest['candidate_rows'] == 360,
            'MANIFEST_POPULATION')
    require(set(manifest['conditions']) == set(CONDITIONS), 'MANIFEST_CONDITIONS')
    require(manifest['model_forward_count'] == 0 and
            manifest['gpu_used'] is False and manifest['model_loaded'] is False,
            'STATIC_PROVENANCE')
    records = _jsonl(cohort_file)
    require(len(records) == 120, 'PAIR_COUNT')
    pair_ids: set[str] = set()
    h2_rows: set[str] = set()
    verb_positions: Counter[int] = Counter()
    for record in records:
        pid = record['pair_id']
        require(isinstance(pid, str) and pid not in pair_ids, 'PAIR_ID_DUPLICATE')
        pair_ids.add(pid)
        claim = record['claim_token_ids']
        require(0 < len(claim) <= 63, 'CLAIM_BUDGET')
        require(set(record['conditions']) == set(CONDITIONS), 'CONDITION_SET')
        require(record['frozen_c6_row_id'] not in h2_rows, 'H1_ROW_ID_COLLISION')
        variants = record['conditions']
        serialized_by_condition: dict[str, list[int]] = {}
        positions: set[int] = set()
        lengths: set[int] = set()
        for condition in CONDITIONS:
            row = variants[condition]
            verb, token, label, label_id = VERBS[condition]
            evidence = row['evidence_token_ids']
            serial = row['serialized_token_ids']
            require(isinstance(evidence, list) and 0 < len(evidence) <= 64,
                    'EVIDENCE_BUDGET')
            require(all(type(v) is int and v >= 0 for v in claim + evidence + serial),
                    'TOKEN_IDS')
            require(serial == claim + [0] + evidence and len(serial) <= 128,
                    'SERIALIZED_INPUT')
            require(row['evidence_len'] == len(evidence), 'EVIDENCE_LENGTH_FIELD')
            position = row['verb_serialized_token_index']
            evidence_position = row['verb_evidence_token_index']
            require(type(position) is int and
                    position == len(claim) + 1 + evidence_position,
                    'VERB_INDEX_MAPPING')
            require(0 <= evidence_position < len(evidence) and serial[position] == token,
                    'VERB_ID')
            require(position + 8 < len(serial), 'POST8_MISSING')
            require(row['expected_semantic_label_candidate'] == label and
                    row['expected_semantic_label_id_candidate'] == label_id,
                    'PROPOSED_LABEL_METADATA')
            require(row['source_kind'] == ('ORIGINAL_C6_EXACT' if condition == 'NEG_DENIES'
                                          else 'NEW_PROPOSED_STATIC_TEXT'), 'SOURCE_KIND')
            require(row['evidence'].count(' ' + verb + ' ') == 1, 'VERB_TEXT')
            serialized_by_condition[condition] = serial
            positions.add(position)
            lengths.add(len(serial))
            h2_id = pid + '__g2_h2__' + condition.lower()
            require(h2_id not in h2_rows and h2_id != record['frozen_c6_row_id'],
                    'H2_ROW_ID_COLLISION')
            h2_rows.add(h2_id)
        require(len(positions) == 1 and len(lengths) == 1, 'TOKEN_ALIGNMENT')
        pos = positions.pop()
        for a, b in COMPARISONS.values():
            first = serialized_by_condition[a]
            second = serialized_by_condition[b]
            require([i for i, (x, y) in enumerate(zip(first, second)) if x != y]
                    == [pos], 'EXACT_SINGLE_TOKEN_SUBSTITUTION')
        verb_positions[variants['NEG_DENIES']['verb_evidence_token_index']] += 1
        negative = variants['NEG_DENIES']['evidence']
        require(text_digest(negative) == record['source_c6_evidence_sha256'],
                'C6_TEXT_DIGEST')
        require(text_digest(negative.replace(
            '; the record explicitly denies that this person ', '; this person ', 1))
                == record['source_c0_evidence_sha256'], 'C0_TEXT_DIGEST')
    require(dict(sorted(verb_positions.items())) ==
            {int(k): v for k, v in manifest['verb_evidence_token_position_histogram'].items()},
            'VERB_POSITION_HISTOGRAM')
    require(len(h2_rows) == 360, 'H2_ROW_COUNT')
    return records


def validate_original_source(repo: Path, records: list[dict[str, Any]]) -> None:
    """Independently match archived candidate against exact frozen XG1 files."""
    root = repo / SOURCE
    split_file = root / 'pair_split_manifest.json'
    view_file = root / 'phase3_seven_cell_labeled_training.jsonl'
    require(split_file.is_file() and view_file.is_file(), 'SOURCE_FILES_MISSING')
    require(digest(split_file) == SPLIT_SHA and digest(view_file) == SEVEN_CELL_SHA,
            'FROZEN_SOURCE_SHA')
    split = json.loads(split_file.read_text(encoding='utf-8-sig'))
    expected_ids = split['dev_pair_ids']
    require(len(expected_ids) == 120 and [r['pair_id'] for r in records] == expected_ids,
            'FROZEN_PAIR_ORDER')
    view = _jsonl(view_file)
    require(len(view) == 4200, 'SOURCE_ROW_COUNT')
    indexed = {(r['pair_id'], r['contrast_cell_id']): r for r in view}
    require(len(indexed) == len(view), 'SOURCE_PAIR_CELL_DUPLICATE')
    for r in records:
        pid = r['pair_id']
        c0 = indexed[(pid, 'C0_SHAM')]
        c6 = indexed[(pid, 'C6_EXPLICIT_DENIAL')]
        neg = r['conditions']['NEG_DENIES']
        require(r['claim'] == c6['claim'] == c0['claim'], 'SOURCE_CLAIM')
        require(neg['evidence'] == c6['evidence'], 'SOURCE_C6_TEXT')
        require(text_digest(c0['evidence']) == r['source_c0_evidence_sha256'],
                'SOURCE_C0_TEXT')
        require(r['frozen_c6_row_id'] == c6['row_id'], 'SOURCE_C6_ROW_ID')
        require(c6['final_label'] == 'REFUTE' and c0['final_label'] == 'SUPPORT',
                'SOURCE_ORIGINAL_LABEL')


def margins(logits: Mapping[str, list[float]]) -> dict[str, float]:
    require(set(logits) == set(CONDITIONS), 'LOGIT_CONDITIONS')
    margin: dict[str, float] = {}
    for c in CONDITIONS:
        z = logits[c]
        require(isinstance(z, (list, tuple)) and len(z) == 3 and
                all(type(x) in (int, float) and math.isfinite(x) for x in z),
                'LOGIT_SHAPE_OR_NONFINITE')
        margin[c] = float(z[0]) - float(z[2])
    return margin


def decision(logits: Mapping[str, list[float]]) -> dict[str, float]:
    m = margins(logits)
    return {name: m[a] - m[b] for name, (a, b) in COMPARISONS.items()}


def arm_interactions(zero: Mapping[str, list[float]],
                     full: Mapping[str, list[float]]) -> dict[str, Any]:
    z, f = decision(zero), decision(full)
    delta = {name: f[name] - z[name] for name in COMPARISONS}
    for value in (z, f, delta):
        require(math.isclose(value['NA'], value['NC'] + value['CA'],
                             abs_tol=1e-10, rel_tol=1e-10), 'CONTRAST_IDENTITY')
    return {'ZERO': z, 'FULL': f, 'DELTA': delta}


def _matrix(value: Any, length: int, channels: int | None, label: str) -> tuple[list[list[float]], int]:
    require(isinstance(value, list) and len(value) == length, label + '_LENGTH')
    require(length > 0 and all(isinstance(row, list) and len(row) > 0 for row in value),
            label + '_SHAPE')
    n = len(value[0])
    require((channels is None or n == channels) and
            all(len(row) == n for row in value), label + '_CHANNELS')
    require(all(type(x) in (float, int) and math.isfinite(x)
                for row in value for x in row), label + '_FINITE')
    return value, n


def region_positions(length: int, evidence_start: int, verb_index: int) -> dict[str, list[int]]:
    require(0 <= evidence_start < length and
            evidence_start <= verb_index and verb_index + 8 < length,
            'REGION_INDICES')
    return {'PRE': list(range(verb_index)), 'VERB': [verb_index],
            'POST8': list(range(verb_index + 1, verb_index + 9)),
            'EVIDENCE': list(range(evidence_start, length))}


def _flat(tensor: list[list[float]], indices: list[int]) -> list[float]:
    return [float(x) for i in indices for x in tensor[i]]


def _stats(values: list[float]) -> dict[str, float]:
    require(bool(values), 'EMPTY_REGION')
    return {'mean': sum(values) / len(values),
            'mean_abs': sum(abs(v) for v in values) / len(values),
            'rms': math.sqrt(sum(v * v for v in values) / len(values))}


def output_readout(outputs: Mapping[str, Mapping[str, list[list[float]]]],
                   *, length: int, evidence_start: int, verb_index: int,
                   arm: str, prefix_tolerance: float = PREFIX_ATOL) -> dict[str, Any]:
    """Detached CPU arrays only. q = wrapped - native, not a raw recurrent write."""
    require(arm in ARMS and set(outputs) == set(CONDITIONS), 'OUTPUT_CONDITIONS_ARM')
    regions = region_positions(length, evidence_start, verb_index)
    native: dict[str, list[list[float]]] = {}
    contrib: dict[str, list[list[float]]] = {}
    channel_count = None
    for c in CONDITIONS:
        require(set(outputs[c]) == {'native', 'wrapped'}, 'OUTPUT_SOURCE_FIELDS')
        n, ch = _matrix(outputs[c]['native'], length, channel_count, c + '_NATIVE')
        w, _ = _matrix(outputs[c]['wrapped'], length, ch, c + '_WRAPPED')
        channel_count = ch
        native[c] = n
        contrib[c] = [[float(y) - float(x) for x, y in zip(nrow, wrow)]
                      for nrow, wrow in zip(n, w)]
        if arm == 'ZERO':
            require(all(x == 0.0 for row in contrib[c] for x in row),
                    'ZERO_CONTRIBUTION_NONZERO')
    deviations: dict[str, float] = {}
    for key, (a, b) in COMPARISONS.items():
        native_pre = max(abs(native[a][i][j] - native[b][i][j])
                         for i in regions['PRE'] for j in range(channel_count))
        q_pre = max(abs(contrib[a][i][j] - contrib[b][i][j])
                    for i in regions['PRE'] for j in range(channel_count))
        require(native_pre <= prefix_tolerance and q_pre <= prefix_tolerance,
                'PREFIX_NUMERICAL_GATE_' + key)
        deviations['NATIVE_PRE_' + key] = native_pre
        deviations['Q_PRE_' + key] = q_pre
    distances: dict[str, dict[str, float]] = {}
    signed: dict[str, dict[str, dict[str, float]]] = {}
    for r, indices in regions.items():
        distances[r] = {}
        signed[r] = {}
        for key, (a, b) in COMPARISONS.items():
            va = _flat(contrib[a], indices)
            vb = _flat(contrib[b], indices)
            distances[r][key] = _stats([x - y for x, y in zip(va, vb)])['rms']
            sa, sb = _stats(va), _stats(vb)
            signed[r][key] = {field: sa[field] - sb[field]
                              for field in ('mean', 'mean_abs', 'rms')}
            if arm == 'ZERO':
                require(distances[r][key] == 0.0, 'ZERO_DISTANCE_NONZERO')
    return {'distance_rms': distances, 'q_stat_differences': signed,
            'prefix_max_abs_deviations': deviations}


def quantile(data: list[float], p: float) -> float:
    require(bool(data) and 0.0 <= p <= 1.0, 'QUANTILE_ARGS')
    values = sorted(data)
    position = p * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize_pairs(rows: Mapping[str, Mapping[str, Any]],
                    *, expected_pairs: int = 120) -> dict[str, Any]:
    """Per-seed descriptive D(ZERO,FULL), delta; pair IDs are sampling units."""
    require(len(rows) == expected_pairs and len(set(rows)) == expected_pairs,
            'AGGREGATE_PAIR_COUNT')
    aggregated: dict[str, Any] = {}
    for condition in COMPARISONS:
        aggregated[condition] = {}
        for arm in ('ZERO', 'FULL', 'DELTA'):
            data = [float(v[arm][condition]) for v in rows.values()]
            require(all(math.isfinite(v) for v in data), 'AGGREGATE_FINITE')
            aggregated[condition][arm] = {
                'mean': sum(data) / len(data),
                'median': median(data),
                'negative': sum(x < 0 for x in data),
                'positive': sum(x > 0 for x in data),
                'zero': sum(x == 0 for x in data),
                'q10': quantile(data, 0.10),
                'q25': quantile(data, 0.25),
                'q75': quantile(data, 0.75),
                'q90': quantile(data, 0.90),
                'min': min(data), 'max': max(data),
            }
    return aggregated


def descriptive_disposition(decision_summary: Mapping[str, Any],
                           mean_output_distances: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Single-seed descriptive gates only; no claims of causal mediation."""
    d = {c: float(decision_summary[c]['DELTA']['mean']) for c in COMPARISONS}
    for r in ('VERB', 'POST8'):
        require(r in mean_output_distances and
                all(c in mean_output_distances[r] for c in COMPARISONS),
                'DISPOSITION_REGION')
    direction = d['NC'] < 0 and d['NA'] < 0
    lexical = abs(d['NC']) > abs(d['CA']) and abs(d['NA']) > abs(d['CA'])
    output = all(float(mean_output_distances[r]['NC']) > float(mean_output_distances[r]['CA'])
                 and float(mean_output_distances[r]['NA']) > float(mean_output_distances[r]['CA'])
                 for r in ('VERB', 'POST8'))
    require(all(math.isfinite(float(mean_output_distances[r][c])) and
                float(mean_output_distances[r][c]) >= 0
                for r in ('VERB', 'POST8') for c in COMPARISONS),
            'DISTANCE_MEAN_VALID')
    return {'direction': direction, 'lexical_control_dominance': lexical,
            'output_selectivity_compatible': output,
            'observational_selectivity_compatible_only': direction and lexical and output,
            'causal_mediation': 'NOT_ESTABLISHED', 'phase3b': 'BLOCKED_UNCHANGED'}


def selftest() -> None:
    """Analytical fixtures; not scientific results."""
    def logit(m: float) -> list[float]:
        return [m, 0., 0.]
    zero = {c: logit(v) for c, v in zip(CONDITIONS, (4., 1., 1.2))}
    full = {c: logit(v) for c, v in zip(CONDITIONS, (1., 0., 0.2))}
    answer = arm_interactions(zero, full)
    require(abs(answer['DELTA']['NC'] + 2.0) < 1e-12, 'SELFTEST_DELTA_NC')
    require(abs(answer['DELTA']['NA'] + 2.0) < 1e-12, 'SELFTEST_DELTA_NA')
    require(abs(answer['DELTA']['CA']) < 1e-12, 'SELFTEST_DELTA_CA')
    length, t, e = 18, 5, 2
    def output(arm: str) -> dict[str, dict[str, list[list[float]]]]:
        result = {}
        for c, val in zip(CONDITIONS, (-2., 1., 0.5)):
            n = [[0., 0.] for _ in range(length)]
            w = [row.copy() for row in n]
            if arm == 'FULL':
                for i in range(t, length):
                    w[i] = [val, val]
            result[c] = {'native': n, 'wrapped': w}
        return result
    fullq = output_readout(output('FULL'), length=length,
                           evidence_start=e, verb_index=t, arm='FULL')
    zeroq = output_readout(output('ZERO'), length=length,
                           evidence_start=e, verb_index=t, arm='ZERO')
    require(fullq['distance_rms']['VERB']['NC'] == 3.0 and
            fullq['distance_rms']['POST8']['NA'] == 2.5 and
            fullq['distance_rms']['PRE']['NC'] == 0.0 and
            zeroq['distance_rms']['VERB']['NC'] == 0.0, 'SELFTEST_Q')
    bad = output('FULL')
    bad['POS_CONFIRMS']['native'][0][0] = 0.01
    try:
        output_readout(bad, length=length, evidence_start=e, verb_index=t,
                       arm='FULL')
    except ContractError as exc:
        require('PREFIX_NUMERICAL_GATE' in str(exc), 'SELFTEST_PREFIX_WRONG_FAIL')
    else:
        raise ContractError('SELFTEST_PREFIX_NOT_BLOCKED')
    print('G2_H2_CPU_THREE_CONDITION_METRIC_SELFTEST_PASS')
    print('SYNTHETIC_ONLY=TRUE;MODEL_NOT_LOADED;GPU_NOT_USED')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--validate-frozen', action='store_true')
    args = p.parse_args()
    require(args.selftest or args.validate_frozen, 'NO_OPERATION_SELECTED')
    if args.selftest:
        selftest()
    if args.validate_frozen:
        require(args.repo_root is not None and args.repo_root.is_dir(), 'REPO_ROOT')
        root = args.repo_root
        records = validate_candidate(root / COHORT / 'matched_verb_candidate.jsonl',
                                     root / COHORT / 'static_candidate_manifest.json')
        validate_original_source(root, records)
        print('G2_H2_CPU_FROZEN_THREE_CONDITION_INPUT_GATE_PASS')
        print('PAIRS=120;CANDIDATES=360;CONDITIONS=3;ONE_TOKEN_SWAP=120/120')
        print('MODEL_NOT_LOADED;GPU_NOT_USED;GIT_NOT_MUTATED')


if __name__ == '__main__':
    main()
