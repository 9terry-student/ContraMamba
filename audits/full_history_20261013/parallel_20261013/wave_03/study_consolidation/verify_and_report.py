"""Independent output readback and report, without importing the producer."""
from pathlib import Path
import collections
import hashlib
import json
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / 'audits/full_history_20261013/parallel_20261013'


def rows(p):
    with p.open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


q = rows(BASE / 'wave_02/integration_02/remaining_census_work_queue.jsonl')
qi = {r['id']: r for r in q}
wave = rows(OUT / 'wave03_result_reconciliation.jsonl')
studies = rows(OUT / 'study_identity_76.jsonl')
members = rows(OUT / 'provisional_study_memberships.jsonl')
debt = rows(OUT / 'unresolved_census_debt.jsonl')
deferred = rows(OUT / 'deferred_scientific_validation.jsonl')
v = json.loads((OUT / 'coverage_validation.json').read_text(encoding='utf-8'))
fp = json.loads((OUT / 'input_fingerprints.json').read_text(encoding='utf-8'))
assert len(qi) == len(q) == 409
for items, expected_count in [(wave, 325), (studies, 76), (deferred, 8)]:
    ids = [r['original_queue_id'] for r in items]
    assert len(ids) == len(set(ids)) == expected_count
assert set(r['original_queue_id'] for r in studies) == {r['id'] for r in q if r['category'] == 'STUDY_IDENTITY_DEPENDENCY'}
assert set(r['original_queue_id'] for r in deferred) == {r['id'] for r in q if r['category'] == 'SCIENTIFIC_NUMERICAL_VALIDATION_DEFERRED'}
assert set(r['original_queue_id'] for r in wave + studies + deferred) == set(qi)
si = {r['original_queue_id']: r for r in studies}
for r in studies:
    original = qi[r['original_queue_id']]
    assert r['original_source_ids'] == original['original_source_ids']
    assert r['all_original_source_identity_aliases'] == original['original_source_identities']
    assert r['source_identity'] in original['original_source_identities']
    assert not r['counts_as_new_experiment'] and not r['independent_replication_established']
    assert r['identity_dependency_status'] == 'OPEN'
    assert set(r['identity_dimension_review']) == {'execution', 'model', 'tokenizer', 'dataset', 'split', 'seed', 'cohort', 'checkpoint', 'endpoint', 'frozen_criteria'}
    assert r['scientific_question'] and r['study_design_and_intervention']
    assert r['original_source_evidence'] and r['exact_remaining_evidence']
assert len({(m['original_queue_id'], m['study_reference']) for m in members}) == len(members) == 63
for m in members:
    s = si[m['original_queue_id']]
    assert m['source_identity'] == s['source_identity']
    assert m['study_reference'] in s['provisional_study_references']
    assert m['original_evidence'] and m['basis']
    assert not m['counts_as_independent_study'] and not m['independent_replication_established']
for r in deferred:
    assert r['original_queue_record'] == qi[r['original_queue_id']]
    assert not r['validation_performed'] and not r['blocks_historical_census_closure']
assert len({r['deduplication_key'] for r in debt}) == len({r['debt_id'] for r in debt}) == len(debt)
owned = {i for r in debt for i in r['original_queue_ids']}
assert {r['original_queue_id'] for r in studies + deferred} <= owned
assert all(r['original_queue_id'] in owned for r in wave if r['census_closure'] != 'INTERPRETATION_RECONSTRUCTED')
assert all(r['original_queue_id'] in owned for r in wave if r['category'] == 'HISTORICAL_TRANSITION_INTERPRETATION' and not r['interpretation_complete'])
special = {r['source_record_id']: r for r in studies if r['protected_dependency']}
assert set(special) == {'RS065', 'RS069', 'RS101', 'RS124', 'RS126'}
assert all(s['identity_dependency_status'] == 'OPEN' for s in special.values())
assert sum(r['interpretation_complete'] for r in wave if r['session'].startswith('transitions')) == 181
assert sum(not r['interpretation_complete'] for r in wave if r['session'].startswith('transitions')) == 35
t1 = [r for r in wave if r['session'] == 'transitions_1']
assert len(t1) == 108 and all(r['historical_saved_disposition'] == 'UNRESOLVED' and r['status_conflict'] for r in t1)
assert sum(r['interpretation_complete'] for r in t1) == 97
assert all(not r['fully_reconstructed_scientific_content'] for r in wave if r['session'] == 'content_recovery')

# Validate all declared input hashes again. Nothing opens original PDF/image,
# tensor, checkpoint or Git objects.
preservation = {p: sha(ROOT / p) == f['sha256'] for p, f in fp['files'].items()}
assert all(preservation.values())
locator_files = {}


def check_locator(loc):
    if not isinstance(loc, dict) or 'jsonl_line' not in loc:
        return
    p = ROOT / loc['file']
    if p not in locator_files:
        locator_files[p] = rows(p)
    assert 1 <= loc['jsonl_line'] <= len(locator_files[p])


for s in studies:
    check_locator(s['original_queue_locator'])
    check_locator(s['prior_adjudication_locator'])
for w in wave:
    for loc in w['evidence_records']:
        check_locator(loc)
    check_locator(w['original_queue_locator'])
    check_locator(w['manifest_locator'])
for m in members:
    check_locator(m['prior_membership_locator'])
for d in debt:
    for loc in d['evidence_locators']:
        check_locator(loc)

git_checks = {}
for cmd in [['git', 'diff', '--check'], ['git', 'diff', '--stat'], ['git', 'diff', '--cached', '--stat'],
            ['git', 'status', '--short', '--branch'], ['git', 'rev-parse', 'HEAD']]:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    git_checks[' '.join(cmd)] = {'exit_code': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
assert git_checks['git rev-parse HEAD']['stdout'] == v['git_head'] == 'ff806bd5bd8efc62fcf6420cf8b097a63ed70571'
assert not git_checks['git diff --stat']['stdout'] and not git_checks['git diff --cached --stat']['stdout']
v['independent_saved_output_readback'] = {
    'status': 'PASS_FOR_ACCOUNTING_TRACEABILITY_AND_PRESERVATION_ONLY',
    'producer_imported_or_executed': False, 'unchanged_inputs_rechecked': len(preservation),
    'locator_files_parsed': len(locator_files), 'wave_bijection': True, 'study_bijection': True,
    'all409_partition': True, 'unique_memberships': True, 'unique_debt_keys': True,
    'ten_identity_dimensions_per_study': True, 'protected_dependencies_open': True,
    'deferred_original_payloads_exact': True, 'input_preservation_checks': preservation,
    'git_checks': git_checks}
v['commands_executed'].extend(['python audits/full_history_20261013/parallel_20261013/wave_03/study_consolidation/verify_and_report.py',
                              *git_checks])
v['tooling_notes'] = ['Two initial producer runs stopped on heterogeneous prior empty-source dependency/membership schemas (KeyError); adapter corrections succeeded. These were consolidation tooling errors, not scientific test results.',
                      'The Python interpreter emits a stale installation-location warning. Final producer and verifier exit zero.']
v['output_fingerprints_excluding_report_validation_and_scripts'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                                                 for p in OUT.glob('*.json*') if p.name != 'coverage_validation.json'}
(OUT / 'coverage_validation.json').write_text(json.dumps(v, sort_keys=True, indent=2) + '\n', encoding='utf-8')

lines = [
    '# ContraMamba — Wave 03 consolidation and Study Identity 76', '',
    '**Decision: PARTIAL. Deterministic accounting, saved-record traceability and input-preservation checks passed. The historical census is not ready for independent completeness certification.**', '',
    'Authority is the current user-directed historical census consolidation instruction, existing source evidence and AGENTS.md. Historical documents establish their own reported designs and decisions; they do not authorize current execution. Only this standalone directory was written. No previous audit, canonical ledger, scientific report, implementation, dataset, checkpoint or Git reference was edited.', '',
    '## Input integrity and exact queue coverage', '',
    f'The original Integration 02 queue contains 409 unique `id` records. Its `category` field selects exactly 325 Wave 03 targets, 76 study-identity dependencies and eight deferred scientific-validation owners. Exact IDs, aliases, original source identities, payload/line locators and dispositions are retained in the JSONL outputs. {v["input_file_count"]} consumed/declared audit input files remain SHA256-identical; all 136 Integration 02 parent fingerprints and all 19 declared Wave 03 output hashes match. No original Git authentication, source enumeration, PDF/image/tensor inspection or 156-lineage investigation was repeated.', '',
    '| Saved session | Exact assigned IDs | Final reconstructed interpretations | Remaining primary closure items |',
    '|---|---:|---:|---:|',
    '| Transition shard 0 | 108 | 84 | 24 |',
    '| Transition shard 1 | 108 | 97 | 11 |',
    '| Content recovery | 103 | 0 complete semantic closures | 103 |',
    '| Provenance and relationships | 6 | 6 documented investigations | 4 missing correspondences + 2 bounded limits |',
    '| **Unique Wave 03 targets** | **325** | | |', '',
    'Both transition manifests preserve the original queue payload. Their even/odd partitions join to all 216 original transition IDs without duplication or omission. The 103 content and six provenance IDs complete the exact 325-target bijection. All 76 study IDs receive one disposition. The 325+76+8 partition accounts for every original queue record.', '',
    '## Historical statuses and final transition candidates', '',
    'The final transition total is **181 reconstructed interpretations and 35 open items**: 166 scientific revisions and 15 editorial/procedural interpretations; 23 unresolved, six companion-dependent and six bounded representation/correspondence items remain open. These counts describe source interpretation, not executed experiments or verified numerical results.', '',
    'Shard 1 retains all 108 legacy UNRESOLVED placeholders. Its later historical-change rows cite the exact saved interpretation file hash, JSONL line and matching endpoint identities. All 108 status differences are explained by that explicit later adjudication; the legacy statuses remain in reconciliation output. Ninety later scientific-revision rows and seven editorial/procedural rows supply 97 final reconstructed candidates. Five dependent and six bounded rows remain open. No unexplained saved-status conflict was found. Shard 0 has 81 consequential-change rows: 76 complete and five partial; its interpretation ledger and 24 gap records control complete/open accounting.', '',
    'The shard-wide union component of the earlier global integration dependency is resolved. Selected ancestry still does not establish exhaustive intervening path histories, branch adoption or original scientific execution. The separate fused-streaming run/cohort dependency remains, while its transition meaning stays reconstructed.', '',
    '## Content and provenance limits', '',
    'All 103 content inputs are accounted for. Ninety-nine source roles were established and four damaged sources were bounded; **zero full semantic gaps were closed**. Exact bytes being available and AST/role inventories being present do not finish whole-function/control-flow, optional activation or historical-authority interpretation. The remaining content owners retain their original WORK/HV IDs.', '',
    'Damage remains at HV001097 (invalid UTF-8 and indentation failure), HV001096 (indentation failure), HV001023 (parser-line syntax damage) and HV000202 (C0-damaged prospective XG2 freeze and unresolved literal authority substitutions). No intended repair or run was invented.', '',
    'REL-0318–REL-0321 remain EVIDENCE_MISSING: compatible pytest rewrite evidence does not authenticate original producer/options, exact source-to-code correspondence, timestamp or a test run. Exact correspondence is withheld, not rejected. The shared snapshot has verified source lineages but unknown external capture order. The original strict capture-chronology blocker remains. Codex capture-ref actor/cause/time attribution stays unknown and nonblocking for scientific source lineage. No new attribution or chronology was inferred.', '',
    '## Exact 76 study dispositions', '',
    '| Candidate disposition | Queue records |', '|---|---:|']
for key, count in sorted(v['study_disposition_counts'].items()):
    lines.append(f'| {key} | {count} |')
lines += ['| **Total** | **76** |', '',
          'The 76 records comprise 63 prior study adjudications, ten empty-source roles and three version-owned dependencies (RS065, RS101, RS126). Original source identities join to the unchanged source inventory/mapping; saved authenticated records supply the source evidence. Each decision records the question, design/intervention, source role, original evidence, authority/implementation/execution/result context where established, missing identity dimensions and exact remaining requirements. Model, tokenizer, dataset, split, seed, cohort, checkpoint, endpoint, frozen criteria and execution are explicitly reviewed without filling absent values from neighboring runs.', '',
          'All **76 original identity dependencies remain open**. This is not an assertion that every source is unclassified or numerically false. Existing local roles and membership proposals are retained; the queue’s specific missing original run/attempt, companion, adoption or global-boundary requirements are not supplied by the saved Wave 03 adjudications. No exact target/context source triple or dependency-version join yielded a new run binding. Six original literal-path leads connect to related implementation histories; those different historical objects are explicitly bounded and do not establish adoption or membership.', '',
          'RS069 retains the reconstructed Stage112–116 synthesis and missing original commands/logs/manifests after runtime loss. RS124 preserves both its dry-run header and appended result section with unresolved original attribution. RS065 remains prospective Phase-0 state-ownership design; RS101 remains prospective artifact-only parameter-SWA analysis with unresolved version/adoption lineage. RS126 retains the C/C2/C3 distinction and prior negative tables; source chronology does not authenticate a new independent execution. None of these five dependencies was closed.', '',
          '## Membership and experiment counting', '',
          '**Zero confirmed independent studies or confirmed memberships were added.** The canonical 26-study ledger remains PROVISIONAL_SOURCE_SUPPORTED, and the original 38 candidate groups remain unchanged. This packet contains 63 provisional source/support associations across 62 source owners: five existing provisional study references and 40 candidate references (37 original candidates plus three earlier local proposals). These are references, not 45 confirmed studies. Fourteen target records have no proposed membership; RS046 retains two synthesis/support links without counting an experiment twice.', '',
          'RS005 links prospectively to STUDY-001; it does not merge STUDY-001/002 or delete the old CANDIDATE-E-005 history. RS024 remains a source-backed pre-response correction to STUDY-009. RS046’s proposed synthesis links to CANDIDATE-E-048/STUDY-012 preserve the earlier unsupported one-to-one STUDY-011 attribution as history. RS077 is tooling reproduction, RS110 is a retry on the prior population/endpoint, and the empty Stage180A retry is packaging support. No replication was inferred. Original negative, failed, blocked and superseded results remain attached to their source records; tests/guards are distinguished from observed experimental failure.', '',
          '## Remaining debt and completeness gate', '',
          f'The deduplicated debt ledger contains **{len(debt)} records**: 211 CENSUS_BLOCKING, three BOUNDED_HISTORICAL_LIMITATION and eight SCIENTIFIC_VALIDATION_DEFERRED. The strict blocking flag is retained on the bounded capture-chronology record, so {v["strict_blocking_debt_records"]} debt records block strict completeness. Classification describes the type of limitation; it does not silently clear an original blocker.', '',
          'The census-blocking records cover 103 incomplete/damaged source owners, 35 transition closures, 52 coordinated study/run evidence requests for 76 study dependencies, 16 distinct unavailable literal companion references, four exact graph correspondences and one original fused-streaming run request. Shared missing originals have one stable debt owner with all dependent WORK/RS/SU IDs and source-specific requirements; generic boilerplate or filename resemblance never establishes study equality. Local absence at a previously inspected object remains distinct from global unavailability or an authenticated replacement.', '',
          'The three bounded limits are shared snapshot capture order, Codex ref attribution and selected-ancestry versus study-wide adoption scope. The 156 lineages are not reinvestigated. The source universe (4474 inventory records), historical-version manifest (1565 records) and source-to-study mapping (152 records) are unchanged and fingerprinted. Untargeted sources retain their original owners; no new semantic closure is claimed for them.', '',
          'The original eight scientific-validation tasks remain deferred with their exact queue payloads. Numerical truth, inference, Red-Team analysis, training and evaluation were not undertaken. Scientific nonvalidation alone is not a census blocker: these eight owners explicitly have `blocks_historical_census_closure=false`. Genuine missing content and original execution identity remain separately open.', '',
          'The packet is ready for independent review of this partial consolidation, **not independent certification that the historical census is complete**. Next authorized evidence work is static completion of the recorded semantic gaps or inspection of specifically identified original run/companion/capture evidence if available. No later phase starts automatically.', '',
          '## Validation, commands and repository preservation', '',
          'The independent readback script does not import the producer. It rechecks exact 325/76/8 ID coverage, original payloads, source aliases, membership evidence and uniqueness, stable debt owners, protected dependencies, ten identity dimensions per study, and all input SHA256 values. JSONL line locators resolve in the saved input records. It verifies 181/35 transition accounting and preservation of all 108 shard-1 historical statuses.', '',
          'Commands actually executed include read-only Git status/branch/HEAD inspection; `rg --files` and filtered prior-ledger discovery; PowerShell report reads; inline Python JSON/schema/count/excerpt/hash checks; the two scripts in this directory; and `git diff --check`, `git diff --stat`, `git diff --cached --stat`, `git status --short --branch`, `git rev-parse HEAD`. No historical source code, bytecode or test suite was executed. Two initial producer runs stopped on heterogeneous empty-source schemas; adapter fixes completed successfully. Python’s stale installation-location warning is an environmental warning, separate from successful final exits.', '',
          'Final HEAD is `ff806bd5bd8efc62fcf6420cf8b097a63ed70571`, detached. `git diff --check` passed and tracked staged/unstaged diffs are empty. Pre-existing `?? audits/` remains. Untracked artifacts were verified separately by deterministic readback; tracked diff checks do not validate them. No commit or push was created. The only new files are the eight requested artifacts and the two deterministic scripts under this directory.', '',
          '## Exact study-target appendix', '',
          '| Original queue ID | Source | Disposition | Proposed references | Identity debt |', '|---|---|---|---|---|']
for s in sorted(studies, key=lambda r: r['source_record_id']):
    lines.append(f'| {s["original_queue_id"]} | {s["source_record_id"]} | {s["disposition"]} | {", ".join(s["provisional_study_references"]) or "—"} | OPEN |')
lines += ['', '## Remaining transition appendix', '',
          '| Original queue ID | Shard | Final candidate disposition | Source path |', '|---|---:|---|---|']
for w in wave:
    if w['session'].startswith('transitions') and not w['interpretation_complete']:
        lines.append(f'| {w["original_queue_id"]} | {w["session"][-1]} | {w["final_candidate_disposition"]} | {w["endpoint_identities"]["after"]["path"]} |')
lines += ['', '## Deferred scientific-validation owners', '', '| Original queue ID | Stable source IDs |', '|---|---|']
for d in deferred:
    lines.append(f'| {d["original_queue_id"]} | {", ".join(d["original_queue_record"]["original_source_ids"])} |')
(OUT / 'consolidation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'independent_readback': 'PASS', 'unchanged_inputs': len(preservation),
                  'wave03_targets': len(wave), 'study_targets': len(studies), 'debt_records': len(debt),
                  'report': str(OUT / 'consolidation_report.md'), 'research_status': 'PARTIAL'}, indent=2))
