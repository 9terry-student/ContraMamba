"""Bounded historical documentary accounting; never imports project code."""
import collections
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5]
CLOSURE = BASE / 'wave_03/census_closure'
fingerprints = {}
cache = {}
commands = []

def sha(b):
    return hashlib.sha256(b).hexdigest()

def register(p):
    p = p.resolve()
    if str(p) not in fingerprints:
        b = p.read_bytes()
        fingerprints[str(p)] = {'sha256': sha(b), 'bytes': len(b)}
    return p

def rows(p):
    p = register(p)
    if str(p) not in cache:
        cache[str(p)] = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    return cache[str(p)]

def readloc(loc):
    p = REPO / loc['file']
    register(p)
    if loc.get('file_sha256'):
        assert fingerprints[str(p.resolve())]['sha256'] == loc['file_sha256'], loc
    return rows(p)[loc['jsonl_line'] - 1]

def dump(name, records):
    (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=True, sort_keys=True) + '\n' for r in records), encoding='utf-8')

def original(blob):
    cmd = ['git', 'show', blob]
    commands.append(cmd)
    b = subprocess.check_output(cmd, cwd=REPO)
    return {'blob': blob, 'sha256': sha(b), 'text': b.decode('utf-8')}

source = rows(CLOSURE / 'source_recovery_results.jsonl')
frozen = {r['debt_id']: r for r in rows(CLOSURE / 'frozen_211_target_manifest.jsonl')}
roots = {r['root_requirement_id']: r for r in rows(CLOSURE / 'root_evidence_requirements.jsonl')}
register(CLOSURE / 'closure_validation.json')
register(CLOSURE / 'closure_report.md')
targets = [r for r in source if r['status'] == 'SOURCE_PRESENT_BUT_SEMANTICS_UNRESOLVED']
assert len(targets) == len({r['debt_id'] for r in targets}) == 126
manifest = []
results = []
for n, r in enumerate(targets, 1):
    f = frozen[r['debt_id']]
    manifest.append({'target_id': r['debt_id'], 'debt_id': r['debt_id'], 'ordinal': n,
        'original_source_ids': r['stable_original_source_ids'], 'original_queue_ids': r['original_queue_ids'],
        'original_source_identities': r['original_source_identities'], 'frozen_original_record': f,
        'closure_result': r, 'root_requirements': [roots[k] for k in r['root_requirement_ids']]})
    prior = [{'locator': loc, 'record': readloc(loc)} for loc in r['prior_evidence_locators']]
    results.append({'target_id': r['debt_id'], 'debt_id': r['debt_id'],
        'original_source_ids': r['stable_original_source_ids'], 'original_queue_ids': r['original_queue_ids'],
        'original_source_identities': r['original_source_identities'], 'root_requirement_ids': r['root_requirement_ids'],
        'precise_unresolved_requirements': r['unresolved_requirement'], 'disposition': 'UNRESOLVED',
        'semantic_requirement_closed': False, 'source_census_blocker_remaining': True,
        'study_identity_consolidation_complete': False, 'scientific_validation': 'NOT_RUN_NOT_AUTHORIZED',
        'recovered_semantic_findings': [], 'preserved_prior_evidence': prior,
        'remaining_requirements': r['unresolved_requirement'],
        'final_identity_dependencies': [roots[k]['exact_missing_evidence'] for k in r['root_requirement_ids']],
        'negative_failed_abandoned_superseded_evidence': f.get('study_context', []),
        'interpretation_scope': 'SOURCE_SPECIFIC_PARTIAL; no full-function or study-adoption conclusion from structure alone',
        'impact': {'source_census': True, 'study_identity': bool(f.get('study_context')), 'only_scientific_validation': False}})
dump('target_manifest.jsonl', manifest)
index = {sid: r for r in results for sid in r['original_source_ids']}

# Five reviewed, byte-identical local units. These do not close full trainer review.
templates = {
    'Stage31CCoverageEntailmentHead': ('75e7e7a45c1d73ff0d3dfc110767f2962ca7b255b8ec0d4d9153aa5578259d37',
        'current concatenates frame/predicate/sufficiency representations; raw_pair concatenates claim, evidence, absolute difference and product; hybrid combines both. detach_input detaches the feature tensor before the MLP. Three or four classes and three input modes are accepted. This is local readout behavior, not proof of whole-trainer gradient ownership.'),
    'install_stage31c_coverage_entailment_head': ('4c1ff82a4af5ae65bda829ffd3e50282246b447af89bbfeef915ff9974f1d544',
        'Installation is idempotent if a head already exists; dimensions come from existing classifiers. The wrapper calls the captured original forward and adds diagnostic outputs. Caller-supplied detach_input matters; this function does not establish whether any historical invocation enabled the head.'),
    'add_stage31c_coverage_entailment_outputs': ('ffebcf2b8aba4d8877eeae58d7d012d0d086b083049a9c6484c7e8bad9b0a097',
        'Missing head or required representation returns the original output. Otherwise softmax/argmax/confidence and named coverage diagnostics are added, with residual columns only for four-class logits. This function does not assign final logits; whole-trainer losses and selection still require review.'),
    'load_stage31c_coverage_entailment_jsonl': ('d50e2b5b8655f5660b0e5af545d172ca44a0012cff68d7e23781ca886d8718bb',
        'The loader rejects the exact evaluation-probe basename stage31_coverage_entailment_probe.jsonl; it checks direction label/ID agreement and train/dev split. This filename guard is not evidence of zero pair reuse or a universal anti-leakage check.'),
    'encode_stage31c_coverage_entailment_labels': ('c3dc94e24ebe8936e36162a267aa59a1e3e89198e493f064fa33f49fcdd3f7f1',
        'The encoder uses coverage_direction_id, rejects IDs outside the configured class count, and creates a long tensor on the supplied device. This auxiliary label encoding does not redefine the final three-way target.')}
contentpath = register(BASE / 'wave_03/content_recovery/recovered_scientific_source_records.jsonl')
units_count = 0
records_count = 0
with contentpath.open(encoding='utf-8') as fh:
    for line_no, raw in enumerate(fh, 1):
        rec = json.loads(raw)
        matched = [index[s] for s in rec['original_source_ids'] if s in index]
        if not matched:
            continue
        records_count += 1
        text = rec.get('lossless_original_source_text_surrogateescape', '')
        lines = text.splitlines()
        for name, (expected, meaning) in templates.items():
            found = re.search(r'^(?:class|def) ' + name + r'\b', text, re.M)
            if not found:
                continue
            tail = text[found.start():]
            nextdef = re.search(r'\n(?:def|class) ', tail)
            unit = (tail[:nextdef.start()] if nextdef else tail).rstrip()
            if sha(unit.encode('utf-8', errors='surrogateescape')) != expected:
                continue
            first = text[:found.start()].count('\n') + 1
            for result in matched:
                result['recovered_semantic_findings'].append({'kind': 'PARTIAL_LOCAL_FUNCTION_SEMANTICS',
                    'symbol': name, 'meaning': meaning, 'source_identity': rec['source_identity'],
                    'evidence_file': str(contentpath.relative_to(REPO)), 'jsonl_line': line_no,
                    'source_line_start': first, 'source_line_end': first + len(unit.splitlines()) - 1,
                    'reviewed_text_sha256': expected, 'reviewed_original_text': unit,
                    'interpretation_basis': 'Complete reviewed local unit and exact text match; no name-only or AST-only inference'})
                units_count += 1
        if records_count % 16 == 0:
            dump('semantic_recovery_results.jsonl', results)
            print('Persisted content batch', records_count, flush=True)

# Reuse source damage evidence without guessing original bytes or substitutions.
damage = rows(BASE / 'wave_03/content_recovery/source_damage_registry.jsonl')
for row in damage:
    for sid in row['original_source_ids']:
        if sid not in index:
            continue
        r = index[sid]
        r['disposition'] = 'BOUNDED_HISTORICAL_UNKNOWN'
        r['recovered_semantic_findings'].append({'kind': 'BOUNDED_SOURCE_DAMAGE', 'evidence': row,
            'meaning': 'Intended damaged bytes/placeholders remain unknown. Preserved source damage cannot be repaired by copying nearby versions without an authenticated correction. This is not recovered historical content.'})
        r['bounded_unknown'] = {'global_unavailability_proven': False, 'unknown_not_recovered': True,
            'needed': 'Authenticated pre-damage original or source-author correction binding intended bytes/substitutions to this exact version.'}

# Focused transitions retain every saved hunk and companion identity. Partial remains open.
transition_results = []
for r in results:
    for entry in r['preserved_prior_evidence']:
        rec = entry['record']
        if 'evidence' not in rec or not isinstance(rec['evidence'], dict) or 'saved_record' not in rec['evidence']:
            continue
        p = REPO / entry['locator']['file']
        ev = p.parent / rec['evidence']['saved_record']
        register(ev)
        assert fingerprints[str(ev.resolve())]['sha256'] == rec['evidence']['saved_record_sha256']
        saved = json.loads(ev.read_text(encoding='utf-8'))
        r['recovered_semantic_findings'].append({'kind': 'PRESERVED_FOCUSED_TRANSITION',
            'evidence_file': str(ev.relative_to(REPO)), 'evidence_sha256': fingerprints[str(ev.resolve())]['sha256'],
            'source_reported_prior_interpretation': rec['source_backed_interpretation'],
            'before': rec['before'], 'after': rec['after'], 'changes': saved['changes'],
            'companion_identities': rec['evidence']['companion_identities'],
            'meaning': 'Saved exact changes and prior partial interpretation retained. Full companion/authority or visual/source reconciliation is not claimed by this extraction.'})
        transition_results.append(r)
    if len(transition_results) and len(transition_results) % 8 == 0:
        dump('semantic_recovery_results.jsonl', results)

# Complete narrow PDF/source correspondence for HV000134 using existing projections.
r = index['HV000134']
ep = BASE / 'wave_03/transitions_0/evidence'
beforetex = register(ep / '038_before_companion_main.tex').read_text(encoding='utf-8')
aftertex = register(ep / '038_after_companion_main.tex').read_text(encoding='utf-8')
for suffix in ['appendix.tex', 'figure_data_manifest.json', 'README.md']:
    b = register(ep / ('038_before_companion_' + suffix))
    a = register(ep / ('038_after_companion_' + suffix))
    assert b.read_bytes() == a.read_bytes()
pdfdiff = json.loads(register(ep / '038.json').read_text(encoding='utf-8'))['changes']
assert len(pdfdiff) == 5 and pdfdiff[2]['before'] == pdfdiff[3]['after']
texchanges = [op for op in difflib.SequenceMatcher(None, beforetex.splitlines(), aftertex.splitlines(), autojunk=False).get_opcodes() if op[0] != 'equal']
assert len(texchanges) == 3
r['recovered_semantic_findings'].append({'kind': 'COMPLETE_NARROW_SOURCE_CORRESPONDENCE',
    'meaning': 'All five saved PDF text changes correspond to three main.tex source edits plus exact relocation of the selected/control paragraph. Invariance language becomes frozen-cohort descriptive rather than test-and-reject; XG2/XG4 are controlled generator contrasts, not latent semantic classes; AI disclosure removes methodology-design/result-interpretation wording and changes literature to related-work discovery. Appendix, figure manifest and README companion bytes are unchanged. No new endpoint, cohort, run or inference is established.',
    'source_diff': ''.join(difflib.unified_diff(beforetex.splitlines(True), aftertex.splitlines(True))),
    'pdf_changes_evidence': str((ep / '038.json').relative_to(REPO)),
    'source_companions': [str((ep / ('038_' + side + '_companion_main.tex')).relative_to(REPO)) for side in ['before', 'after']],
    'scope_limit': 'Historical claim/source-text correspondence, not validation of PDF rendering, numerical claims or actual AI use.'})
r.update(disposition='SEMANTICS_RECOVERED', semantic_requirement_closed=True,
    source_census_blocker_remaining=False, remaining_requirements=[],
    interpretation_scope='Selected before/after publication semantic transition fully reconciled at saved text/source level')
r['impact'] = {'source_census': False, 'study_identity': False, 'only_scientific_validation': True}
r['final_identity_dependencies'] = ['No new study identity follows from this publication correction. Underlying original study/run dependencies and separate snapshot chronology remain unchanged.']
dump('semantic_recovery_results.jsonl', results)

# Source-specific documentary revision comparison, exact known versions only.
ledgerpath = BASE.parent / 'expansion_02/study_identity_ledger.jsonl'
ledger = rows(ledgerpath)
assert len(ledger) == 26
docdefs = {
 'RS065': ('Gen5 Phase-0 design', ['b0a3b33c0be18bbb8bba29a93a65b33e469ec160','d6fac7e0cdf466c03d66e453a010363383048159'],
    'The scientific unit is a Phase-0 hypothesis/design, not an execution. Four distinct parents are preserved: Gen4 standalone, K-series, convergence, and cross-scale paper. H_G5_1 asks role integrity versus a capacity/information-matched ownership null; task score and semantic owner naming are not primary. Both known versions deny implementation/training/evaluation; their only differences are two status lines, candidate versus FROZEN_AT_25A4206. The status token is a source assertion, not a reconstructed full authority commit. Broad precursor failure, parked confident-error work, failed steering, residual criteria and non-universality remain inherited boundaries.',
    'CANDIDATE-E-065'),
 'RS101': ('Stage195-C analysis-only specification', ['e1d4e76eb41773d35670bdeacedae1286d39a6a2','0de699659f0728a82d3662910b3bc7e29fb08f40'],
    'The analysis unit is an aligned dev row within a planned frozen six-run Stage195-B matrix, with paired seed/arm summaries. Primary outlier endpoint is SWA rescues minus harms relative to epoch20; stable SUPPORT negatives are a separate boundary endpoint. Pooled rows are descriptive repeated dev positions, not independent replications. Known versions differ only by the run-namespace correction: manifest_run_id identifies seed/arm directories; trainer_run_name is single; trajectory association uses paths/hashes and need not expose a run field. Metrics, aggregation, taxonomy and nine-file schema do not change. BLOCKED/harm/support/tradeoff/no-support/mixed are rules, not observed results. A specification is not evidence that its analyzer ran.',
    'CANDIDATE-E-101'),
 'RS126': ('Stage31-C/C2/C3 diagnostic lineage', ['4ec381860d98889b48be756c071e7144e31995fe','23b7617ace807b09071e9570650189a1099595fa','374af78d4f355f9aab195f760a53957dbcff6e17','cda1626f9122feb42a561d454ed063815f855d2f'],
    'The documentary unit is a representation-access ablation proposal following failed C and C2 diagnostic readouts. Four-class C becomes default three-class C2 hard contrasts; C3 adds current/raw_pair/hybrid, uses existing FrameGate representations and denies composer integration until six safety gates pass. Old C report accuracy0.4650/macroF1 0.3878/alignment0.4450 and39/80 SUPPORT recovery are retained as historical reported failure; C3 reports C2 accuracy0.4500/macroF1 0.3535/alignment0.3650 and42/80 recovery, not a C3 success. Removed old tables are not erased outcomes. Final labels, calibration and selection remain outside diagnostic head integration. Exact chronological ordering/adoption is not inferred from blob comparison.',
    'CANDIDATE-E-126')}
for sid, (title, blobs, meaning, candidate) in docdefs.items():
    r = index[sid]
    materials = [original(b) for b in blobs]
    diffs = [{'from_blob': m['blob'], 'to_blob': materials[0]['blob'],
        'diff': ''.join(difflib.unified_diff(m['text'].splitlines(True), materials[0]['text'].splitlines(True)))} for m in materials[1:]]
    r['recovered_semantic_findings'].append({'kind': 'DOCUMENTARY_IDENTITY_AND_REVISION_SEMANTICS',
        'title': title, 'meaning': meaning, 'exact_original_versions': materials, 'version_comparisons': diffs,
        'ledger_comparison': {'evidence_file': str(ledgerpath.relative_to(REPO)), 'candidate': candidate,
            'all_26_saved_studies_considered': [{'study_id': x['study_id'], 'scope': x['scope'], 'source_record_ids': x['source_record_ids']} for x in ledger],
            'decision': 'Keep candidate as a documentary design/lineage reference; no direct source membership match among 26 ledger source lists and no source-supported execution equivalence. No canonical ledger promotion or merge.',
            'limit': 'Absence of a direct membership match does not prove globally independent execution or exhaustive historical study separation.'}})
    r.update(disposition='SEMANTICS_RECOVERED', semantic_requirement_closed=True,
        source_census_blocker_remaining=False, remaining_requirements=[],
        interpretation_scope='Documentary question, analysis unit, known version content and candidate/ledger relationship; separate original adoption/chronology/run bindings remain open')
    r['impact'] = {'source_census': False, 'study_identity': True, 'only_scientific_validation': False}
    r['final_identity_dependencies'] = ['Preserve provisional candidate; obtain original execution/attempt/adoption binding only if claiming an executed study. Separate snapshot capture chronology is not closed.',
        'Underlying parent evidence and exact implementation/result correspondence are required for any later executed-study consolidation; source design and shared code do not establish equality.']
    dump('semantic_recovery_results.jsonl', results)

# The two present documents explicitly retain companion dependencies.
for sid, meaning in {
 'RS062': 'The correction states forward-equivalent Delta_L is twice owned-gradient Delta_L and freezes three distinct scale/cohort readouts. Signs/ranks/p-values are source-reported invariant; this is arithmetic/semantic relabeling, not a new inferential family. Upstream readout implementation/artifact bindings and downstream corrected manuscript versions remain required; no new numerical calculation is performed.',
 'RS125': 'The auxiliary report describes 900 hard-contrast rows, 720/180 split, 300 per class, twelve 75-row families, and removal of OTHER_RESIDUAL. It names a builder and jsonl probe exclusion. These declarations do not bind original row/pair identities, builder hashes, or zero exact-pair reuse. Missing .json literal is not automatically replaced by .jsonl.'}.items():
    r = index[sid]
    r['disposition'] = 'REQUIRES_COMPANION_EVIDENCE'
    r['recovered_semantic_findings'].append({'kind': 'DOCUMENTARY_MEANING_WITH_COMPANION_GAP',
        'meaning': meaning, 'evidence_basis': r['preserved_prior_evidence'][0]['locator']})
    if sid == 'RS125':
        r['recovered_semantic_findings'][-1]['original_source_material'] = original(r['original_source_identities'][0]['blob'])

# Literal applicability is an exact source-owner join, not lexical similarity.
literal_records = []
targetids = {s for r in results for s in r['original_source_ids']}
for k, root in roots.items():
    probe = root.get('local_literal_probe')
    if not probe:
        continue
    literal = probe['literal']['path']
    applicable = sorted(set(root['dependent_original_source_ids']) & targetids)
    row = {'root_requirement_id': k, 'literal': literal,
        'original_literal_source_ids': root['dependent_original_source_ids'],
        'applicable_target_source_ids': applicable, 'applicable_target_ids': sorted({index[s]['target_id'] for s in applicable}),
        'applicable_to_126_population': bool(applicable),
        'original_status': probe['literal']['status'], 'original_literal_recovered': False,
        'disposition': 'REQUIRES_COMPANION_EVIDENCE' if applicable else 'OUTSIDE_ASSIGNED_POPULATION_PRESERVED',
        'prior_bounded_probe': probe, 'global_absence_established': False,
        'historical_revision_or_companion_replacement_authenticated': False,
        'needed': root['exact_missing_evidence'], 'source_evidence': str((CLOSURE / 'root_evidence_requirements.jsonl').relative_to(REPO))}
    if applicable:
        mats = [f for f in index['RS125']['recovered_semantic_findings'] if f.get('original_source_material')]
        text = mats[0]['original_source_material']['text']
        row['companion_candidate'] = {'path': 'data/stage31_coverage_entailment_probe.jsonl',
            'report_mentions': [{'line': n, 'text': s} for n,s in enumerate(text.splitlines(),1) if 'stage31_coverage_entailment_probe' in s],
            'status': 'SOURCE_NAMES_JSONL; NOT_AUTHENTICATED_RENAME_OF_JSON',
            'next_dependency': 'Original .json artifact with producer/version binding, or authenticated author correction/rename tying that literal to the exact .jsonl version and pair cohort.'}
        row['disposition'] = 'SOURCE_CONFLICT'
        row['audit_source_locator_discrepancy'] = {
            'frozen_debt_locator': literal,
            'exact_original_source_literal': 'data/stage31_coverage_entailment_probe.jsonl',
            'original_source_blob': '7e360a69c2aa924af873a1bc0d7d9e5515380768',
            'same_literal_in_prior_authenticated_excerpt': True,
            'prior_excerpt_locator': {'file': 'audits/full_history_20261013/parallel_20261013/wave_02/study_adjudication/study_identity_adjudications.jsonl', 'jsonl_line': 51, 'source_line': 47},
            'limit': 'The complete source literal is jsonl, whereas the frozen unavailable locator is json. The extraction/correction cause is not established. No rename or silent debt closure is asserted.',
            'next_dependency': 'Adjudicate the recorded locator against original source/extraction provenance, then bind the exact original jsonl probe identity and auxiliary pair cohort. The original frozen debt remains preserved.'}
    literal_records.append(row)
assert len(literal_records) == 16
dump('missing_literal_recovery.jsonl', literal_records)
# Independently read narrow changed clauses; retain unresolved whole-transition scope.
focused_meanings = {
 'HV001464': 'The test revision expands trace coverage from two recovery cells to the full 3x3 and changes the expected trace SHA mapping to HISTORICAL_TRAINING_REPORT_SHA256. Added assertions preregister correction commit341e2e59668ea9b575007ddf591424b831170577, scalar EPS32 multiplier32, parameter max-abs1e-5/relative-L2 1e-4 and operator relative-Frobenius1e-4. Endpoint torch.equal/SHA equality becomes diagnostic, while scalar/parameter/operator bounded authentication remains a gate. This is a documented authentication-contract change, not evidence that those tests or the replay passed. Exact worker/authority comparison remains required.',
 'HV001137': 'The trainer adds a default-off TemporalChannel and default-true input-detach option. Positive loss weight plus head/data/label/mask availability activates BCE on temporal_channel_logit, reusing TD/TA output when available. Gated penalty requires the preservation head and cannot coexist with an active temporal-adapter penalty. The report distinguishes diagnostic BCE from a separately enabled final-logit modifier and warns about applying an untrained head. Slot-state representation and actual detach behavior still need exact companion implementation review; metadata declarations alone are not gradient proof.',
 'HV001115': 'The changed Stage31-C clauses introduce an optional diagnostic coverage head and separate auxiliary CE batch. Architecture/head/loss/data requirements gate loading; train/dev auxiliary splits are kept distinct. The CE block also requires positive weight and no v7_no_aux_losses. The detach flag defaults false, so readout-only absence of direct final-logit assignment does not by itself imply gradient isolation. Diagnostic metrics are exported and marked unused for checkpoint selection; complete selection and companion ownership remain open.',
 'HV001117': 'The changed Stage30-D clauses distinguish mismatch-positive1 from safety-positive1, wire temporal mismatch multihead options and data into the trainer, and retain optional fusion/cap controls. Shared temporal nomenclature must not equate the two target conventions. Defaults and actual cap detach/fusion behavior require the saved companion and complete trainer context; no run activation or result is inferred.',
 'HV001142': 'The prior and changed source introduce final-decision-aware temporal constrained selection in addition to temporal diagnostic metrics. This affects selection semantics, not merely report labels. Exact thresholds, metric denominators, fallback selection and companion/stage-authority equivalence remain open; no successful historical selection is inferred.',
 'HV000113': 'The only saved PDF-text hunk deletes the explicit local-rank/non-homology subtitle. This is a presentation claim omission in this artifact, not proof that the historical study adopted coordinate identity. Companion manifest/README and visual correspondence still require reconciliation.',
 'HV000114': 'The saved figure text replaces kernel/leading-plane scalar panels with separately labeled XG2/XG4 centered-linear-CKA matrices; supported local selected/control ranks remain, while historical endpoint labels and failed370M residual criterion move to an Appendix A.4 pointer. This changes displayed evidence emphasis, not the truth or status of the failed residual criterion. Exact figure/manifest correspondence remains open.',
 'HV000107': 'The saved figure text replaces four exact causal endpoint values with Supported and an Appendix A.3 pointer, rounds the behavioral margin and association numbers, and replaces owned-readout axis wording with Local task readout. The margin-not-accuracy restriction remains. This compression/relabeling is not a new causal experiment; complete upstream/manifest correspondence remains open.',
 'HV000121': 'The saved figure text adds a0.006 tick, percentile95% pair-interval whisker wording, and changes the mixed-association subtitle. It introduces uncertainty display semantics rather than a new mean-sign result. The exact resampling artifact/manifest binding is needed to close the full figure transition; no interval is recomputed.'}
for sid, meaning in focused_meanings.items():
    r = index[sid]
    witnesses = [x for x in r['recovered_semantic_findings'] if x['kind']=='PRESERVED_FOCUSED_TRANSITION']
    assert witnesses
    r['recovered_semantic_findings'].append({'kind': 'INDEPENDENT_NARROW_CHANGED_CLAUSE_INTERPRETATION',
        'meaning': meaning, 'evidence_file': witnesses[0]['evidence_file'],
        'scope_limit': 'Partial changed-clause semantics only. Final UNRESOLVED retained for companion/authority/full-source or visual obligations.'})
dump('semantic_recovery_results.jsonl', results)
gaps = [dict(target_id=r['target_id'], disposition=r['disposition'],
    original_source_ids=r['original_source_ids'], root_requirement_ids=r['root_requirement_ids'],
    remaining_requirements=r['remaining_requirements'], final_identity_dependencies=r['final_identity_dependencies'],
    source_census_blocker_remaining=r['source_census_blocker_remaining'], impact=r['impact'],
    bounded_unknown=r.get('bounded_unknown')) for r in results if not r['semantic_requirement_closed']]
dump('remaining_semantic_gaps.jsonl', gaps)

# Independent serialized readback and immutable-input check.
readback = [json.loads(x) for x in (OUT / 'semantic_recovery_results.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(readback) == 126 and len({r['target_id'] for r in readback}) == 126
assert {r['target_id'] for r in readback} == {r['debt_id'] for r in targets}
for old, new in zip(targets, readback):
    assert old['debt_id'] == new['target_id']
    assert old['original_source_identities'] == new['original_source_identities']
    assert old['original_queue_ids'] == new['original_queue_ids']
    assert old['stable_original_source_ids'] == new['original_source_ids']
    assert old['unresolved_requirement'] == new['precise_unresolved_requirements']
    assert new['preserved_prior_evidence'] and new['recovered_semantic_findings'], new['target_id']
for p, info in fingerprints.items():
    assert sha(Path(p).read_bytes()) == info['sha256'], p
counts = dict(collections.Counter(r['disposition'] for r in readback))
validation = {'authority': 'Current user-directed Census Recovery A + AGENTS.md',
    'phase': 'STATIC_HISTORICAL_RECONSTRUCTION_ONLY', 'expected_original_targets': 126,
    'actual_original_targets': len(readback), 'id_bijection': True, 'original_identity_payload_preserved': True,
    'zero_silent_omissions': True, 'unprocessed_target_ids': [], 'disposition_counts': counts,
    'closed_target_semantic_requirements': sum(r['semantic_requirement_closed'] for r in readback),
    'previous_evidence_sufficient_whole_target_count': counts.get('PREVIOUS_EVIDENCE_SUFFICIENT', 0),
    'partial_local_function_units': units_count, 'partial_function_target_count': sum(any(f['kind']=='PARTIAL_LOCAL_FUNCTION_SEMANTICS' for f in r['recovered_semantic_findings']) for r in readback),
    'remaining_source_census_blocker_count_in_assigned_population': len(gaps),
    'all_16_literal_dependencies_accounted': True, 'applicable_missing_literals': sum(x['applicable_to_126_population'] for x in literal_records),
    'recovered_original_literals': 0, 'prior_inputs_unchanged': True, 'input_fingerprints': fingerprints,
    'applicable_literal_audit_source_discrepancies': 1,
    'all_dispositions_have_evidence': True, 'bounded_unknown_is_not_recovery': True,
    'git_commands_by_producer': commands, 'git_identity_verification_repeated': False,
    'broad_source_universe_scan_repeated': False, 'training_evaluation_gpu_tests': 'NOT_RUN',
    'scientific_validity_certified': False, 'full_census_completion_claimed': False,
    'whole_trainer_semantics_complete': False,
    'scope_limit': 'UNRESOLVED means investigated with bounded partial evidence, not completed independent whole-function review. Documentary closure does not close separate study-run/adoption or snapshot chronology debt.'}
(OUT / 'coverage_validation.json').write_text(json.dumps(validation, ensure_ascii=True, indent=2, sort_keys=True)+'\n', encoding='utf-8')
print(json.dumps({k: validation[k] for k in ['actual_original_targets','disposition_counts','partial_local_function_units','partial_function_target_count','applicable_missing_literals']}, indent=2), flush=True)
