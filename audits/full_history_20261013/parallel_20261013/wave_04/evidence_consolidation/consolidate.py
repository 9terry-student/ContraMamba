"""Deterministic evidence-only consolidation. No project modules or experiments run."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
AUDIT = ROOT / 'audits/full_history_20261013'
PAR = AUDIT / 'parallel_20261013'
W4 = PAR / 'wave_04'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s]
def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)
def write(name, value, jsonl=False):
    p = OUT / name
    assert not p.exists(), f'Refusing overwrite: {p}'
    text = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=True) + '\n' for r in value) if jsonl else json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2) + '\n'
    p.write_text(text, encoding='utf-8', newline='\n')

assert ROOT.name == 'cm-full-audit-20261013'
assert not (OUT / 'validation.json').exists(), 'Use independent readback for existing outputs; do not overwrite.'
before = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(AUDIT.rglob('*')) if p.is_file() and not p.is_relative_to(OUT)}
head = git('rev-parse', 'HEAD').decode().strip()
refs = git('show-ref')
status = git('status', '--short').decode()
assert head == 'ff806bd5bd8efc62fcf6420cf8b097a63ed70571'
checks = []
for rel, hash_key in [('git_exact_path_reconciliation/validation.json', 'output_hashes_excluding_validation'), ('external_evidence/coverage_validation.json', 'output_fingerprints_excluding_self')]:
    vp = W4 / rel
    v = load(vp)
    for name, expected in v[hash_key].items():
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        assert digest(vp.parent / name) == expected, name
        checks.append({'file': (vp.parent / name).relative_to(ROOT).as_posix(), 'sha256': expected})
    for name, expected in v.get('input_fingerprints', v.get('previous_audit_fingerprints', {})).items():
        expected = expected['sha256'] if isinstance(expected, dict) else expected
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        assert digest(p) == expected, name
        checks.append({'file': p.relative_to(ROOT).as_posix(), 'sha256': expected})
for name, expected in load(W4 / 'local_semantics/coverage_validation.json')['input_fingerprints'].items():
    assert digest(Path(name)) == expected['sha256'], name
    checks.append({'file': Path(name).relative_to(ROOT).as_posix(), 'sha256': expected['sha256']})

debt_path = PAR / 'wave_03/study_consolidation/unresolved_census_debt.jsonl'
debts = rows(debt_path)
by_id = {r['debt_id']: r for r in debts}
assert len(debts) == len(by_id) == 222
exact_path = W4 / 'git_exact_path_reconciliation/exact_path_reconciliation.jsonl'
exact = rows(exact_path)
missing = rows(W4 / 'git_exact_path_reconciliation/remaining_true_missing_originals.jsonl')
sem_path = W4 / 'local_semantics/semantic_recovery_results.jsonl'
sem = rows(sem_path)
assert len(exact) == 16 and len(missing) == 7 and len(sem) == 126
aliases = [r for r in exact if r['documentary_requirement_satisfied']]
assert len(aliases) == 9
authenticated = {}
for r in aliases:
    for candidate in r['historical_candidates']:
        identity = candidate['identity']
        if identity['path'] != r['accepted_alias_path']:
            continue
        raw = git('show', identity['blob'])
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == identity['blob']
        assert hashlib.sha256(raw).hexdigest() == identity['sha256']
        assert git('rev-parse', identity['commit_or_tree'] + ':' + identity['path']).decode().strip() == identity['blob']
        authenticated[identity['blob']] = identity
    assert any(i['path'] == r['accepted_alias_path'] for i in authenticated.values())
    for d in r['census_debt_closure']['debt_records']:
        assert d['original_debt_record'] == by_id[d['debt_id']], d['debt_id']

upstream_commit = '7c62d1ec8df3fb560d6efe2b85fa191135636f81'
sources = []
for split, blob in [('train', '0f190e115cf2ee23416e8a539c8d6ac043d7cc83'), ('dev', '40974243267f395dc583d805d10f043812419249')]:
    p = AUDIT / 'external_originals/AVeriTeC_7c62d1ec' / (split + '.json')
    raw = p.read_bytes()
    computed = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert computed == blob, split
    sources.append({'repository': 'MichSchli/AVeriTeC', 'upstream_commit': upstream_commit, 'upstream_path': 'data/' + split + '.json', 'local_path': p.relative_to(ROOT).as_posix(), 'git_blob': blob, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'json_record_count': len(json.loads(raw)), 'authentication': 'LOCAL_BYTES_MATCH_PINNED_GIT_BLOB_FROM_AUTHENTICATED_RS024_EVIDENCE', 'upstream_network_or_commit_tree_reinspection': False, 'run_or_selected_cohort_authenticated': False, 'role': 'DOCUMENTARY_SOURCE' if split == 'train' else 'FRESHNESS_COMPANION'})
train_id = 'DEBT-0d1d0a7ad7523f000d85'
alias_ids = {r['request_id'].replace('ROOT-', 'DEBT-') for r in aliases}
semantic_ids = {r['debt_id'] for r in sem if r['semantic_requirement_closed']}
assert len(semantic_ids) == 4
closed_ids = alias_ids | {train_id} | semantic_ids
assert len(closed_ids) == 14 and closed_ids <= by_id.keys()
closures = []
for r in aliases:
    id_ = r['request_id'].replace('ROOT-', 'DEBT-')
    closures.append({'debt_id': id_, 'status': 'DOCUMENTARY_PATH_REQUIREMENT_RESOLVED_IN_OVERLAY', 'scope': 'PATH_AVAILABILITY_ONLY', 'original_debt_record': by_id[id_], 'authenticated_wave04_record': r, 'evidence_locator': {'file': exact_path.relative_to(ROOT).as_posix(), 'sha256': digest(exact_path), 'jsonl_line': exact.index(r) + 1}, 'related_run_cohort_provenance_study_obligations_closed': False})
closures.append({'debt_id': train_id, 'status': 'PINNED_UPSTREAM_SOURCE_AVAILABILITY_RESOLVED_IN_OVERLAY', 'scope': 'SOURCE_BYTES_ONLY', 'original_debt_record': by_id[train_id], 'upstream_sources': sources, 'related_run_cohort_provenance_study_obligations_closed': False, 'remaining_related_debt_id': 'DEBT-92c165596a87562966e7'})
unlocated = [r for r in missing if r['request_id'].replace('ROOT-', 'DEBT-') != train_id]
assert len(unlocated) == 6
assert alias_ids | {train_id} | {r['request_id'].replace('ROOT-', 'DEBT-') for r in unlocated} == {r['request_id'].replace('ROOT-', 'DEBT-') for r in exact}
overlay = []
for r in debts:
    id_ = r['debt_id']
    status_ = 'OPEN_RETAINED'
    if id_ in alias_ids:
        status_ = 'DOCUMENTARY_PATH_REQUIREMENT_RESOLVED'
    elif id_ == train_id:
        status_ = 'PINNED_UPSTREAM_SOURCE_AVAILABILITY_RESOLVED'
    elif id_ in semantic_ids:
        status_ = 'PRIOR_WAVE04_SELECTED_SEMANTIC_REQUIREMENT_RESOLVED'
    overlay.append({'debt_id': id_, 'original_debt_record': r, 'consolidated_status': status_, 'blocks_strict_census_completeness_remaining': r['blocks_strict_census_completeness'] and id_ not in closed_ids, 'evidence_scope_only': True, 'canonical_ledger_modified': False})
assert sum(r['blocks_strict_census_completeness_remaining'] for r in overlay) == 198
external = rows(W4 / 'external_evidence/target_manifest.jsonl')
assert len(external) == 83
assert len({r['debt_id'] for r in external} - closed_ids) == 73
study = rows(PAR / 'wave_03/study_consolidation/study_identity_76.jsonl')
assert len(study) == 76
target = next(r for r in sem if r['original_source_ids'] == ['RS125'])
assert target['source_census_blocker_remaining']
next_work = {'decision': 'PRIORITIZE_RS125_LOCAL_DATA_CONSTRUCTION_CHAIN', 'ranking_basis': 'Integrator judgment: authenticated Stage31 locator narrows a bounded companion gap; explicit dataset/builder/pair obligations can be investigated locally without model execution. No claimed guaranteed closure or exhaustive ranking.', 'target_debt_id': target['debt_id'], 'original_source_ids': target['original_source_ids'], 'wave04_gap_record': target, 'steps': ['Bind exact historical 900-row auxiliary dataset and builder/code bytes to RS125 and its authority.', 'Preserve original/hardened 200-row Stage31 versions separately; identify the version required by the historical construction source.', 'Statically reconstruct label mapping, contrast-set/pair identity and train720/dev180 split provenance; preserve the zero exact-pair-reuse obligation.', 'Distinguish source-described construction from authenticated original materialization; leave missing producer/run bindings open.'], 'boundary': 'Proposed next work only; no dataset regeneration, training, evaluation or numerical science validation performed.', 'other_local_targets': ['RS062 exact upstream readout-to-corrected-manuscript documentary chain; no rerun inferred.', '116 unresolved Wave04 whole-function/control-flow and historical-authority interpretations; prioritize source-supported loss/gradient ownership and activation boundaries.', 'Five Wave04 conditional companion alternatives, including HV001270 encoder masks/gradient ownership, retain their external fallback requirements.'], 'damaged_sources_not_repaired': ['HV000202', 'HV001023', 'HV001096', 'HV001097']}
write('input_preservation_checkpoint.json', {'head': head, 'git_refs_sha256': hashlib.sha256(refs).hexdigest(), 'initial_git_status': status, 'preexisting_audit_fingerprints': before, 'declared_hash_checks': checks})
write('documentary_debt_closures.jsonl', sorted(closures, key=lambda r: r['debt_id']), True)
write('authenticated_upstream_sources.jsonl', sources, True)
write('unlocated_originals.jsonl', [{'debt_id': r['request_id'].replace('ROOT-', 'DEBT-'), 'status': 'BOUNDED_UNLOCATED_ORIGINAL_RETAINED', 'original_wave04_record': r, 'original_debt_record': by_id[r['request_id'].replace('ROOT-', 'DEBT-')], 'global_absence_or_destruction_established': False, 'missing_experiment_or_failed_experiment_inferred': False} for r in unlocated], True)
write('consolidated_debt_accounting.jsonl', overlay, True)
write('next_local_reconstruction.json', next_work)
write('user_testimony.json', {'source': 'CURRENT_USER_TASK_REQUIREMENT_4', 'statement': 'Historical Kaggle Outputs were not retained for these runs.', 'evidence_type': 'USER_TESTIMONY', 'scope': 'Runs referenced by the current bounded historical recovery task; exact run enumeration and timing not supplied by testimony.', 'independently_verified_archive_inventory': False, 'does_not_establish': ['Experiments absent or failed', 'Artifacts never produced', 'Global destruction of all originals', 'Run/cohort identity or scientific validity'], 'effect': 'Do not depend on retained historical Kaggle Outputs as an available recovery route; retain local original, command/log/manifest and provenance obligations.'})

table = '\n'.join('| ' + r['debt_id'] + ' | ' + r['authenticated_wave04_record']['accepted_alias_path'] + ' |' for r in closures if 'authenticated_wave04_record' in r)
source_table = '\n'.join('| ' + r['upstream_path'] + ' | ' + r['git_blob'] + ' | ' + r['sha256'] + ' | ' + str(r['json_record_count']) + ' |' for r in sources)
missing_table = '\n'.join('| ' + r['request_id'].replace('ROOT-', 'DEBT-') + ' | ' + r['exact_missing_path'] + ' |' for r in unlocated)
report = f'''# Wave04 historical census evidence consolidation

Decision: PARTIAL CONSOLIDATION. Nine authenticated path requirements and one pinned upstream source-byte requirement resolve in this separate findings overlay. Four earlier Wave04 semantic closures are carried forward. No full census completion, independent study promotion, replication, experiment absence or experiment failure is inferred.

Authority: current Historical Census Integrator task, repository AGENTS.md, existing full-history census and reconciled Wave04 evidence only. Writes are confined to this directory. HEAD is detached at {head}; initial status is `?? audits/`. Historical source specifications are evidence, not current execution authority. Prior ledgers and reports remain unchanged.

## Authenticated evidence and closable documentary debt

The original 16 literal requirements partition exactly into nine path corrections, one recovered pinned AVeriTeC source and six bounded unlocated originals. Documentary closure here means the selected availability obligation is resolved, with stable debt IDs and original payloads preserved. Associated execution, cohort, version-selection and study dependencies are retained. No automatic JSON/JSONL substitution is used. All accepted corrected-path candidate bytes were rechecked against Git blob SHA1, SHA256 and commit/path bindings. Multiple historical serialization/schema versions remain distinct; no run-specific input version is selected.

| Debt ID | Authenticated corrected path |
|---|---|
{table}

Pinned upstream repository: MichSchli/AVeriTeC; commit {upstream_commit}. Local preserved originals are under `audits/full_history_20261013/external_originals/AVeriTeC_7c62d1ec/`. Git-header blob SHA1 calculated from local bytes matches the identities in authenticated RS024/Wave04 evidence. The upstream network/commit tree was not re-inspected. SHA256 values below were calculated directly from the local preserved bytes; row counts describe the upstream JSON arrays, not steering cohorts.

| Upstream path | Git blob | SHA256 | Records |
|---|---|---|---:|
{source_table}

DEBT-0d1d0a7ad7523f000d85 resolves pinned train source availability. Dev is a freshness companion and adds no extra debt closure. The upstream path is not a ContraMamba-local file claim. DEBT-92c165596a87562966e7 remains open for original2800/corrected2799 cohort materialization, raw command/checkpoint/tokenizer identities and exact execution artifact chain. Recovering upstream data does not authenticate those subsets or a rerun.

## Unlocated originals and testimony

| Retained debt ID | Original literal |
|---|---|
{missing_table}

Stage35A retains the source-declared JSONL output alternative and implementation-only boundary: an original generated JSONL is required if one was produced; no regeneration is authorized. Stage163A MD/JSON are separate obligations in a coordinated recovery context, and the downstream Stage163B synthesis does not replace them. Stage196 analysis, Stage148 route-order summary and Stage146 external-generalization summary likewise retain exact original requirements. Their complete prior findings and remaining obligations are preserved in unlocated_originals.jsonl.

Current user testimony states that historical Kaggle Outputs were not retained for these runs. This is attributed testimony, not independently authenticated archive inspection. It supplies no exact run enumeration or production/failure status. Missing files do not mean missing experiments or failed experiments; bounded local non-location does not establish global absence or destruction.

## Remaining census blockers

All 222 original debt IDs and original payloads survive in consolidated_debt_accounting.jsonl. There are 14 resolved selected requirements (10 documentary/source availability plus four prior Wave04 semantics), 208 remaining debt dispositions: 197 census-blocking, three bounded historical limitations and eight deferred scientific-validation records. The snapshot chronology limitation remains strict-blocking, yielding 198 remaining strict-completeness blockers. These are debt counts, not experiment or study counts. The 83 external-assigned IDs partition into 10 resolved availability requirements and 73 retained obligations; no other external root closes.

Wave04 local semantics remain four recovered and 122 open: 116 unresolved interpretations, two companion-dependent requirements and four damaged-source unknowns. The Stage31 path correction narrows RS125's locator issue but leaves its dataset/builder/pair provenance debt open. All 76 study-identity dependencies retain their original requirements. Canonical 26-study provisional ledger and historical candidate boundaries are unchanged; zero confirmed independent studies or memberships are added.

Original run/attempt/cohort/checkpoint/tokenizer/seed bindings, branch adoption and source-authority correspondence remain source-specific. Preserve the pytest producer/options and REL-0318–REL-0321 derivation gaps, strict snapshot capture chronology, fused-streaming run/cohort provenance, original Stage112–116 command/log/manifests, Stage31B dry-run versus appended-result attribution, Stage196 P4/P5 originals, and all negative/blocked/superseded boundaries. Four damaged sources (HV000202, HV001023, HV001096, HV001097) require authenticated original/correction evidence; intended bytes are not reconstructed here. Original requirements are carried in full rather than replaced by this summary.

## Next highest-value local science-history reconstruction target

Prioritize RS125 / DEBT-0f63c887bfbf703cfb43: reconstruct the historical 900-row auxiliary dataset, builder/code identities, contrast-set/pairs and train720/dev180 split, with the exact original/hardened Stage31 probe version kept explicit and the zero exact-pair-reuse obligation retained. This prioritization is an integrator judgment: Wave04 has narrowed the locator, and the source explicitly requires no model execution or checkpoint for this data-construction record. Local companions may resolve portions; original materialization/provenance must still be established. No closure is promised.

Next alternatives are RS062's upstream readout-to-corrected-manuscript chain, the 116 unfinished whole-function/control-flow and historical-authority interpretations, and the five conditional companion alternatives (including HV001270 masks/gradient ownership). next_local_reconstruction.json preserves the exact Wave04 RS125 gap and the bounded proposed steps. None of that follow-up reconstruction is executed in this consolidation.

## Validation and activity

The saved helper uses only Python standard-library accounting and read-only Git queries. It checks input/output declared hashes, exact 222-ID accounting and original payload preservation, 16-request partition, source Git-header SHA1/SHA256, accepted alias commit/path bindings, 126 semantic target accounting, 83 external IDs and 76 study dependencies. Every pre-existing audit file is SHA256 checked before and after; Git HEAD/ref fingerprints and tracked/staged diffs are checked. validation.json records actual outcomes and output hashes. Preservation coverage is all pre-existing audit files; production artifacts were not written or executed.

Commands actually used: git status/branch/rev-parse, rg file/text discovery, PowerShell Get-Content, Python standard-library JSON/hash/schema inspections, read-only git show/rev-parse/show-ref, this consolidation helper, git diff --check and diff stats. A preliminary top-level parallel_20261013 lookup failed because the directory is nested under audits/full_history_20261013. An exploratory Python read used the default cp949 encoding and failed; all producer reads use explicit UTF-8. One orchestration syntax error occurred before any shell execution. The Python launcher reports a stale installation-location warning; successful commands still exit zero. These probe issues are distinct from evidence mismatches.

No project tests, scientific validation, training, evaluation, model forwards, dataset regeneration, Kaggle, fetch, Git mutations, commits or pushes were run. No existing audit files were edited. Git diff statistics apply only to tracked files; untracked output contents are verified by serialized readback and hashes. The next authorized action is review of this consolidation; further historical reconstruction requires its own scoped task.
'''
rp = OUT / 'report.md'
assert not rp.exists()
rp.write_text(report, encoding='utf-8', newline='\n')
assert all(digest(ROOT / p) == h for p, h in before.items())
after_paths = {p.relative_to(ROOT).as_posix() for p in AUDIT.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
assert after_paths == set(before)
assert git('rev-parse', 'HEAD').decode().strip() == head and git('show-ref') == refs
assert len(rows(OUT / 'consolidated_debt_accounting.jsonl')) == 222
for r in rows(OUT / 'consolidated_debt_accounting.jsonl'):
    assert r['original_debt_record'] == by_id[r['debt_id']]
git('diff', '--check')
assert not git('diff', '--stat').strip() and not git('diff', '--cached', '--stat').strip()
write('validation.json', {'decision': 'PASS_DETERMINISTIC_ACCOUNTING_AND_PRESERVATION_ONLY', 'head': head, 'head_and_refs_unchanged': True, 'previous_audit_files_checked': len(before), 'previous_audit_files_changed_removed_added_outside_output': [], 'original_debt_count': 222, 'resolved_documentary_requirements': 10, 'carried_prior_semantic_closures': 4, 'remaining_debt_count': 208, 'remaining_strict_completeness_blockers': 198, 'unlocated_original_count': 6, 'external_assigned_remaining': 73, 'study_identity_dependencies_retained': 76, 'authenticated_alias_blobs': authenticated, 'declared_hash_checks': checks, 'git_diff_check': 'PASSED', 'tracked_and_staged_diff_stat': 'EMPTY', 'final_git_status': git('status', '--short').decode(), 'training_evaluation_kaggle_tests': 'NOT_RUN', 'commit_push_git_mutation': 'NOT_PERFORMED', 'full_census_complete': False, 'output_hashes_excluding_validation': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}, 'validation_command': 'python -B audits/full_history_20261013/parallel_20261013/wave_04/evidence_consolidation/consolidate.py'})
print(json.dumps({'decision': 'PASS', 'debt_ids': 222, 'remaining_strict_blockers': 198, 'previous_audit_files_preserved': len(before), 'output_files': len(list(OUT.iterdir()))}))
