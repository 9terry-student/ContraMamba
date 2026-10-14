"""Deterministic saved-record reconciliation. No historical code or Git objects execute.

Run from the repository root. Writes only alongside this script. All scientific
claims remain source reports. Exact source triples, not filenames, drive joins.
"""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
AUDIT = ROOT / 'audits/full_history_20261013'
BASE = AUDIT / 'parallel_20261013'
OUT = Path(__file__).resolve().parent
FP = {}
LOC = {}


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def remember(p):
    p = p.resolve()
    key = p.relative_to(ROOT).as_posix()
    if key not in FP:
        FP[key] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    return key


def load(p):
    p = Path(p)
    if not p.is_absolute():
        p = BASE / p
    key = remember(p)
    if p.suffix == '.jsonl':
        rows = []
        with p.open(encoding='utf-8') as f:
            for line, text in enumerate(f, 1):
                if text.strip():
                    row = json.loads(text)
                    rows.append(row)
                    LOC[id(row)] = {'file': key, 'jsonl_line': line}
        FP[key]['record_count'] = len(rows)
        return rows
    return json.loads(p.read_text(encoding='utf-8'))


def save(name, data):
    p = OUT / name
    if p.suffix == '.jsonl':
        with p.open('w', encoding='utf-8', newline='\n') as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')
    else:
        p.write_text(json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def triple(s):
    return tuple(s.get(k) for k in ('commit', 'path', 'blob'))


def identity_nodes(v):
    if isinstance(v, dict):
        if all(isinstance(v.get(k), str) for k in ('commit', 'path', 'blob')):
            yield v
        for value in v.values():
            yield from identity_nodes(value)
    elif isinstance(v, list):
        for value in v:
            yield from identity_nodes(value)


def evidence_ref(row, sections=None):
    d = dict(LOC[id(row)])
    if sections:
        d['sections'] = sections
    return d


def unique_index(rows, key):
    d = {r[key]: r for r in rows}
    assert len(d) == len(rows), ('duplicate key', key)
    return d


queue = load('wave_02/integration_02/remaining_census_work_queue.jsonl')
Q = unique_index(queue, 'id')
assert len(Q) == 409
categories = collections.Counter(r['category'] for r in queue)
assert categories == {'HISTORICAL_TRANSITION_INTERPRETATION': 216,
                      'SOURCE_CONTENT_RECOVERY': 103,
                      'SOURCE_RELATIONSHIP_UNCERTAINTY': 4,
                      'SNAPSHOT_CHRONOLOGY': 1,
                      'GIT_REFERENCE_PROVENANCE_LIMITATION': 1,
                      'STUDY_IDENTITY_DEPENDENCY': 76,
                      'SCIENTIFIC_NUMERICAL_VALIDATION_DEFERRED': 8}
versions = load(AUDIT / 'expansion_03/historical_source_versions.jsonl')
HV = unique_index(versions, 'id')
inventory = load(AUDIT / 'expansion_03/source_universe_manifest.jsonl')
SU = unique_index(inventory, 'id')
mapping = load(AUDIT / 'expansion_02/source_to_study_mapping.jsonl')
RS = unique_index(mapping, 'source_record_id')
ledger = load(AUDIT / 'expansion_02/study_identity_ledger.jsonl')
STUDIES = unique_index(ledger, 'study_id')
assert len(STUDIES) == 26
old_links = load('study_identity_116/provisional_study_identity_links.jsonl')
OL = unique_index(old_links, 'source_record_id')
old_evidence = load('study_identity_116/study_identity_evidence.jsonl')
OE = unique_index(old_evidence, 'source_record_id')
load('study_identity_116/target_manifest.jsonl')
load('study_identity_116/unresolved_identity_dependencies.jsonl')
prior_adjudications = load('wave_02/study_adjudication/study_identity_adjudications.jsonl')
PA = unique_index(prior_adjudications, 'work_id')
empty = load('wave_02/study_adjudication/empty_source_dispositions.jsonl')
PE = unique_index(empty, 'work_id')
prior_deps = load('wave_02/study_adjudication/unresolved_study_dependencies.jsonl')
PD = unique_index(prior_deps, 'work_id')
prior_memberships = load('wave_02/study_adjudication/candidate_study_memberships.jsonl')
load('wave_02/study_adjudication/target_manifest.jsonl')
integration_study = load('wave_02/integration_02/study_identity_reconciliation.jsonl')
IS = unique_index(integration_study, 'queue_id')
prior_revisions = load('versions_shard_1/semantic_revision_records.jsonl')
parent_fingerprints = load('wave_02/integration_02/input_fingerprints.json')
parent_checks = []
for relative, declared in parent_fingerprints['files'].items():
    p = AUDIT / relative
    key = remember(p)
    ok = FP[key]['sha256'] == declared['sha256']
    parent_checks.append({'file': key, 'matches': ok, 'expected': declared['sha256']})
    assert ok, ('Integration02 input fingerprint conflict', key)
assert len(parent_checks) == 136

# Fingerprint saved reports and read final validation records; validate declared
# output hashes instead of regenerating their authenticated Git checks.
sessions = {}
hash_checks = []
for session in ('transitions_0', 'transitions_1', 'content_recovery', 'provenance_relationships'):
    folder = BASE / 'wave_03' / session
    val = load(folder / 'coverage_validation.json')
    remember(folder / 'report.md')
    manifest = load(folder / 'target_manifest.jsonl')
    sessions[session] = {'validation': val, 'manifest': manifest}
    for field in ('output_sha256', 'output_fingerprints', 'output_fingerprints_excluding_validation_self'):
        for name, expected in val.get(field, {}).items():
            expected = expected.get('sha256') if isinstance(expected, dict) else expected
            p = folder / name
            key = remember(p)
            ok = FP[key]['sha256'] == expected
            hash_checks.append({'file': key, 'declaration': field, 'expected': expected, 'matches': ok})
            assert ok, ('saved output hash conflict', key)

population = sorted(r['id'] for r in queue if r['category'] == 'HISTORICAL_TRANSITION_INTERPRETATION')
reconciled = []
TR = {}
status_conflicts = []
for shard in (0, 1):
    session = f'transitions_{shard}'
    folder = BASE / 'wave_03' / session
    manifests = sessions[session]['manifest']
    interpretations = load(folder / 'transition_interpretations.jsonl')
    changes = load(folder / 'historical_scientific_changes.jsonl')
    gaps = load(folder / 'remaining_transition_gaps.jsonl')
    sessions[session].update(interpretations=interpretations, changes=changes, gaps=gaps)
    M = unique_index(manifests, 'original_queue_id')
    I = unique_index(interpretations, 'original_queue_id')
    C = unique_index(changes, 'original_queue_id')
    assigned = population[shard::2]
    assert set(M) == set(I) == set(assigned) and len(assigned) == 108
    assert set(C) <= set(M)
    for original_id in assigned:
        m, i, c, q = M[original_id], I[original_id], C.get(original_id), Q[original_id]
        saved_q = m.get('integrated_queue_record', m.get('original_queue_record'))
        assert saved_q == q, ('manifest payload conflict', original_id)
        assert q['original_queue_id'] == original_id and q['original_queue_ids'] == [original_id]
        for side in ('before', 'after'):
            if c:
                cs = c.get(side, c.get('source_identities', {}).get(side))
                assert triple(cs) == triple(i[side]), ('endpoint conflict', original_id, side)
        if shard == 0:
            disposition = i['disposition']
            complete = i['interpretation_complete']
            assert complete == (disposition in ('SCIENTIFIC_REVISION_RECONSTRUCTED', 'EDITORIAL_OR_PROCEDURAL_ONLY'))
            if c:
                assert c['disposition'] == disposition
            summary = i.get('source_backed_interpretation')
            conflict = None
        else:
            assert i['disposition'] == 'UNRESOLVED'
            assert m['original_source_authentication']['identity_verified']
            assert m['original_source_authentication']['git_object_hash_recomputed']
            assert all(i[side]['identity_verified'] and i[side]['git_object_hash_recomputed'] for side in ('before', 'after'))
            assert c and c['saved_disposition'] == 'UNRESOLVED'
            assert c['evidence']['jsonl_line'] == LOC[id(i)]['jsonl_line']
            assert c['evidence']['sha256'] == FP[remember(folder / 'transition_interpretations.jsonl')]['sha256']
            disposition = c['disposition']
            complete = disposition in ('SCIENTIFIC_REVISION_RECONSTRUCTED', 'EDITORIAL_OR_PROCEDURAL_ONLY')
            summary = c['revision_summary']
            conflict = {'legacy': 'UNRESOLVED', 'later_candidate': disposition,
                        'resolution': 'EXPLAINED_LATER_SOURCE_BACKED_ADJUDICATION_LEGACY_PRESERVED',
                        'basis': c['disposition_basis'], 'evidence': c['evidence']}
            status_conflicts.append({'original_queue_id': original_id, 'resolution': conflict['resolution']})
        relevant_gaps = [g for g in gaps if g.get('original_queue_id') == original_id or
                         g.get('original_queue_ids') == [original_id]]
        row = {'original_queue_id': original_id, 'category': q['category'], 'session': session,
               'original_source_ids': q['original_source_ids'],
               'original_source_identities': q['original_source_identities'],
               'original_queue_locator': evidence_ref(q), 'manifest_locator': evidence_ref(m),
               'historical_saved_disposition': i['disposition'],
               'final_candidate_disposition': disposition, 'interpretation_complete': complete,
               'census_closure': 'INTERPRETATION_RECONSTRUCTED' if complete else 'OPEN',
               'source_backed_summary': summary, 'status_conflict': conflict,
               'evidence_records': [evidence_ref(i)] + ([evidence_ref(c)] if c else []),
               'endpoint_identities': {side: i[side] for side in ('before', 'after')},
               'remaining_gap_records': [{'locator': evidence_ref(g), 'record': g} for g in relevant_gaps],
               'historical_adoption_or_execution_established': False,
               'scientific_validation': 'DEFERRED_NOT_PERFORMED'}
        TR[original_id] = row
        reconciled.append(row)
    primary_open_ids = {g.get('original_queue_id') or g.get('original_queue_ids', [None])[0]
                        for g in gaps if shard == 0 or g.get('scope') == 'COMPANION_OR_REPRESENTATION_CLOSURE'}
    assert primary_open_ids == {r['original_queue_id'] for r in reconciled if r['session'] == session and not r['interpretation_complete']}

content = load('wave_03/content_recovery/source_content_recovery.jsonl')
content_gaps = load('wave_03/content_recovery/remaining_content_gaps.jsonl')
CG = unique_index(content_gaps, 'id')
damages = load('wave_03/content_recovery/source_damage_registry.jsonl')
assert len(content) == 103 and len(CG) == 103 and len(damages) == 4
for r, m in zip(content, sessions['content_recovery']['manifest']):
    q = Q[r['id']]
    assert m['id'] == r['id'] and m['original_integrated_queue_record'] == q
    hv = HV[q['original_source_ids'][0]]
    assert triple(r['source_identity']) == triple(hv)
    assert r['git_identity']['commit_path_blob_verified'] and r['git_identity']['expanded_manifest_match']
    assert not r['fully_resolved_content_task'] and not r['census_sufficient']
    reconciled.append({'original_queue_id': r['id'], 'category': q['category'], 'session': 'content_recovery',
                       'original_source_ids': q['original_source_ids'], 'original_source_identities': q['original_source_identities'],
                       'original_queue_locator': evidence_ref(q), 'manifest_locator': evidence_ref(m),
                       'historical_saved_disposition': r['disposition'], 'final_candidate_disposition': r['disposition'],
                       'census_closure': 'OPEN_SEMANTIC_OR_DAMAGE', 'fully_reconstructed_scientific_content': False,
                       'source_identity_authentication': 'EXISTING_SAVED_RECORD_REUSED',
                       'evidence_records': [evidence_ref(r), evidence_ref(CG[r['id']])],
                       'scientific_validation': 'DEFERRED_NOT_PERFORMED'})

relations = load('wave_03/provenance_relationships/relationship_adjudications.jsonl')
PG = load('wave_03/provenance_relationships/remaining_provenance_gaps.jsonl')
snapshot = load('wave_03/provenance_relationships/snapshot_chronology_assessment.json')
attribution = load('wave_03/provenance_relationships/git_attribution_assessment.json')
RA = unique_index(relations, 'queue_id')
assert len(RA) == 4 and len(PG) == 6
for m in sessions['provenance_relationships']['manifest']:
    q = Q[m['queue_id']]
    assert m['original_queue_record'] == q
    p = RA.get(q['id'], snapshot if q['category'] == 'SNAPSHOT_CHRONOLOGY' else attribution)
    assert p['disposition'] == ('EVIDENCE_MISSING' if q['category'] == 'SOURCE_RELATIONSHIP_UNCERTAINTY' else 'BOUNDED_LIMITATION')
    if q['id'] in RA:
        assert all(x['commit_path_blob_match'] and x['git_blob_hash_verified'] for x in p['exact_source_identity']['endpoints'])
    reconciled.append({'original_queue_id': q['id'], 'category': q['category'], 'session': 'provenance_relationships',
                       'original_source_ids': q['original_source_ids'], 'original_source_identities': q['original_source_identities'],
                       'original_queue_locator': evidence_ref(q), 'manifest_locator': evidence_ref(m),
                       'historical_saved_disposition': p['disposition'], 'final_candidate_disposition': p['disposition'],
                       'census_closure': 'OPEN_EXACT_RELATIONSHIP' if q['id'] in RA else 'BOUNDED_OPEN_HISTORICAL_LIMIT',
                       'evidence_records': [evidence_ref(p)] if q['id'] in RA else [{'file': remember(BASE / 'wave_03/provenance_relationships' / m['assessment_file']), 'queue_id': q['id']}],
                       'scientific_validation': 'NOT_CERTIFIED'})

assert len(reconciled) == 325 and len({r['original_queue_id'] for r in reconciled}) == 325
expected_wave = {r['id'] for r in queue if r['category'] not in ('STUDY_IDENTITY_DEPENDENCY', 'SCIENTIFIC_NUMERICAL_VALIDATION_DEFERRED')}
assert {r['original_queue_id'] for r in reconciled} == expected_wave

# ID-based role adjudication from explicit original-source question/design/role
# evidence in prior records. These are record roles, never experiment counts.
ROLE_GROUPS = {
    'PROVISIONAL_NEW_STUDY_DESIGN': 'RS005 RS017 RS030 RS035 RS048 RS122 RS129 RS137 RS139',
    'REVISION': 'RS022 RS024 RS027 RS057 RS074 RS078 RS083 RS092 RS120 RS124 RS131 RS132 RS133 RS134 RS135 RS140 RS141',
    'RETRY_OR_TOOLING_REPRODUCTION': 'RS077 RS110',
    'PROCEDURAL_SUPPORT': 'RS044 RS062 RS091 RS125',
    'ANALYSIS_ONLY': 'RS015 RS032 RS040 RS042 RS046 RS050 RS064 RS066 RS068 RS069 RS070 RS073 RS075 RS076 RS079 RS080 RS081 RS084 RS085 RS093 RS105 RS106 RS109 RS111 RS112 RS113 RS114 RS119 RS121 RS145 RS147',
}
ROLES = {rid: role for role, ids in ROLE_GROUPS.items() for rid in ids.split()}
assert set(ROLES) == {r['target_id'] for r in prior_adjudications}
all_changes = sessions['transitions_0']['changes'] + sessions['transitions_1']['changes']
change_by_exact = collections.defaultdict(list)
change_by_path = collections.defaultdict(list)
for c in all_changes:
    for node in identity_nodes(c):
        if c not in change_by_exact[triple(node)]:
            change_by_exact[triple(node)].append(c)
    endpoint = c.get('after', c.get('source_identities', {}).get('after'))
    change_by_path[endpoint['path']].append(c)

study_targets = [r for r in queue if r['category'] == 'STUDY_IDENTITY_DEPENDENCY']
decisions = []
new_memberships = []
special = {
    'RS065': {
        'question': 'Prospective causal-role-grounded state ownership after heterogeneous Gen4 geometry and objective-dependent functional readout.',
        'design': 'Phase-0 scientific specification. Original line91 bounds scale-local causal-role recurrence; line45 denies a universal semantic state/depth. Full design/adoption correspondence across HV000203/HV000204 remains unadjudicated.',
        'role': 'PROVISIONAL_NEW_STUDY_DESIGN',
        'negative': 'Gen4 does not establish a universal semantic state or universal depth; no Gen5 execution is inferred from a frozen specification.'},
    'RS101': {
        'question': 'Artifact-only parameter-SWA question on the Stage195-B six-run matrix and epoch18/19/20 trainable parameters; complete hypothesis wording is not present in saved excerpts.',
        'design': 'Prospective artifact-only causal analysis; training seeds180/181/182, split174, exact baseline/intervention pairs and epoch1--20 export/hash requirements. No model/tokenizer/checkpoint loading or new forwards under source line8. Pooled repeated-dev rows are descriptive only; outcome hierarchy is a criterion, not an observed replication.',
        'role': 'ANALYSIS_ONLY_DESIGN',
        'negative': 'BLOCKED/replicated-harm/replicated-support labels are prospective decision rules. No outcome or independent replication is established.'},
    'RS126': {
        'question': 'Does diagnostic Coverage/Entailment improve under a representation-access ablation without changing final predictions or selection?',
        'design': 'C3 optional diagnostic head in trainer; probe forbidden for training/calibration/threshold/checkpoint selection. Prior C result and C2 three-class auxiliary design are preserved as different revisions, not independent experiments inferred from versions.',
        'role': 'REVISION',
        'negative': 'C2 did not pass safety. Original C accuracy0.4650, macroF1 0.3878, alignment0.4450 and39/80 SUPPORT recovery remain source reports even though C2 removes the table.'},
}

for q in study_targets:
    wid = q['id']
    selected_ids = [s for s in q['original_source_ids'] if s.startswith(('RS', 'SU'))]
    assert len(selected_ids) == 1
    rid = selected_ids[0]
    prior = PA.get(wid, PE.get(wid))
    source = prior['source_identity'] if prior else OE[rid]['source_identity']
    assert triple(source) in {triple(s) for s in q['original_source_identities']}
    if rid.startswith('RS'):
        mr = RS[rid]
        assert (mr['source_commit'], mr['source_path'], mr['source_blob']) == triple(source)
        assert OE[rid]['git_blob_authenticated'] and triple(OE[rid]['source_identity']) == triple(source)
        inventory_id = mr['inventory_id']
    else:
        mr = SU[rid]
        assert (mr['source_commit'], mr['path'], mr['source_blob']) == triple(source)
        inventory_id = rid
    if prior:
        auth = prior['source_authentication']
        assert auth['commit_path_blob_match'] and auth['git_blob_sha1_computed'] == source['blob']
    else:
        auth = {'source_identity': source, 'git_blob_authenticated_in_prior_record': True,
                'snapshot_alias_authentication': 'Wave03 provenance saved 159-alias checks; no new Git object check'}
    if wid in PA:
        disposition = prior['disposition']
        role = ROLES[rid]
        question = prior['scientific_question']
        design = prior['hypothesis_and_intervention_definition']
        historical_role = prior['historical_role_and_execution_distinction']
        negative = prior['negative_superseded_and_boundary_evidence']
        refs = prior['proposed_study_memberships']
        missing = prior['missing_or_unvalidated_identity_dimensions']
        evidence = prior['original_source_evidence']
        fields = prior['source_reported_identity_fields']
        associated = prior['associated_spec_run_report_artifact_context']
        unavailable = prior['unavailable_source_locators']
    elif wid in PE:
        disposition = 'SUPPORTING_OR_PROCEDURAL'
        role = 'PROCEDURAL_SUPPORT'
        question = 'Procedural source role for ' + prior['underlying_run_or_procedure']
        design = prior['empty_source_role'] + '; ' + prior['study_reconstruction_effect']
        historical_role = prior['underlying_run_or_procedure']
        negative = prior['unresolved_role_details']
        refs = [prior['proposed_supporting_study_reference']] if prior['proposed_supporting_study_reference'] else []
        missing = ['original_launcher_exit_identity', 'original_output_manifest_hashes',
                   'original_stage_input_and_source_identities', 'run_attempt_or_procedure_completeness']
        evidence = prior['role_evidence_basis']
        fields = {}
        associated = prior['associated_nonempty_sources']
        unavailable = []
    else:
        s = special[rid]
        disposition = 'PROVISIONAL_CANDIDATE_BOUNDARY_OPEN'
        role, question, design, negative = s['role'], s['question'], s['design'], s['negative']
        historical_role = OL[rid]['historical_role']
        refs = [OL[rid]['candidate_study_id']]
        missing = OL[rid]['missing_or_unvalidated_identities']
        evidence = OE[rid]['expansion_01_source_locations']
        fields = {'saved_source_reported_provenance_lines': OE[rid]['reported_provenance_lines'],
                  'saved_frozen_criterion_lines': OE[rid]['reported_frozen_criterion_lines']}
        associated, unavailable = [], []
    exact_sources = {triple(s) for s in identity_nodes({'source': source, 'associated': associated, 'evidence': evidence})}
    matches = {}
    for t in exact_sources:
        for c in change_by_exact.get(t, []):
            matches[c['original_queue_id']] = c
    dependency_version_ids = sorted({v for d in q['blocking_dependencies'] for v in d.get('version_target_ids', [])})
    dep_source_matches = []
    for vid in dependency_version_ids:
        if vid in HV:
            for c in change_by_exact.get(triple(HV[vid]), []):
                dep_source_matches.append({'version_id': vid, 'queue_id': c['original_queue_id'], 'locator': evidence_ref(c)})
    historical_revision_links = []
    for c in sorted(matches.values(), key=lambda c: c['original_queue_id']):
        historical_revision_links.append({'original_queue_id': c['original_queue_id'], 'locator': evidence_ref(c),
                                         'disposition': c['disposition'],
                                         'summary': c.get('historical_revision', c.get('revision_summary')),
                                         'relationship': 'EXACT_TARGET_OR_AUTHENTICATED_CONTEXT_SOURCE_IDENTITY',
                                         'resolves_original_execution_identity': False})
    # Only literal references in original source excerpts establish these leads.
    # A referenced pathname is not a blob/run binding or a membership proof.
    literal_links = []
    literal_evidence = evidence if isinstance(evidence, list) else []
    if rid in OE:
        literal_evidence = literal_evidence + OE[rid]['source_reported_associated_paths']
    for path, path_changes in sorted(change_by_path.items()):
        citing_lines = [e for e in literal_evidence if isinstance(e, dict) and
                        (e.get('path') == path or path in str(e.get('excerpt', e.get('text', ''))))]
        if citing_lines:
            literal_links.append({'literal_referenced_path': path, 'original_source_citations': citing_lines,
                                  'transition_records': [{'original_queue_id': c['original_queue_id'], 'locator': evidence_ref(c),
                                                          'disposition': c['disposition'],
                                                          'summary': c.get('historical_revision', c.get('revision_summary'))}
                                                         for c in path_changes],
                                  'relationship_limit': 'SOURCE_LITERAL_PATH_REFERENCE_ONLY. Different historical objects; no run, adoption, execution chronology, or study membership inferred.'})
    dimension_fields = {'execution': 'execution_identity', 'model': 'model_identity',
                        'tokenizer': 'tokenizer_identity', 'dataset': 'dataset_identity',
                        'split': 'split_and_seed_identity', 'seed': 'split_and_seed_identity',
                        'cohort': 'cohort_identity', 'checkpoint': 'checkpoint_identity',
                        'endpoint': 'endpoint_identity', 'frozen_criteria': 'frozen_criteria'}
    dimensions = {}
    for dimension, field in dimension_fields.items():
        saved = fields.get(field)
        dimensions[dimension] = {'saved_evidence': saved,
                                 'state': saved['state'] if isinstance(saved, dict) and 'state' in saved else 'UNKNOWN_OR_DECLARED_CONTEXT_ONLY_NO_ORIGINAL_RUN_BINDING',
                                 'original_run_authentication_added': False,
                                 'limit': 'Declared design/context values do not prove executed values. UNKNOWN is not absence of an executed study.'}
    # Source existence/role and source change meaning cannot fill the specific
    # original launcher/runtime/adoption and run/attempt boundary requirements.
    required = list(q['exact_missing_information'])
    if wid in PD:
        required.append(PD[wid]['exact_required_inputs'])
    required = list(dict.fromkeys(required))
    prior_locator = evidence_ref(prior) if prior else evidence_ref(OE[rid])
    row = {'original_queue_id': wid, 'original_queue_aliases': q['original_queue_ids'],
           'original_source_ids': q['original_source_ids'], 'source_record_id': rid,
           'source_identity': source, 'all_original_source_identity_aliases': q['original_source_identities'],
           'inventory_reference': {'inventory_id': inventory_id, 'historical_version_ids': dependency_version_ids},
           'original_queue_locator': evidence_ref(q), 'prior_adjudication_locator': prior_locator,
           'source_identity_authentication': {'basis': 'EXISTING_AUTHENTICATED_RECORDS_AND_EXACT_INVENTORY_JOIN', 'record': auth, 'reauthenticated_git_objects': False},
           'historical_saved_disposition': prior['disposition'] if prior else OL[rid]['disposition'],
           'disposition': disposition, 'record_role_classification': role,
           'identity_dependency_status': 'OPEN', 'scientific_question': question,
           'study_design_and_intervention': design, 'historical_role_and_execution_distinction': historical_role,
           'provisional_study_references': refs, 'confirmed_independent_study_membership': False,
           'independent_replication_established': False, 'counts_as_new_experiment': False,
           'original_source_evidence': evidence, 'associated_original_authority_implementation_execution_result_sources': associated,
           'negative_failed_blocked_abandoned_superseded_evidence': negative,
           'reported_identity_dimensions': fields, 'missing_or_unvalidated_identity_dimensions': missing,
           'identity_dimension_review': dimensions,
           'identity_dimension_limits': {
               'execution': 'Original run/attempt and adoption chain remains unestablished; prospective/no-execution roles retained.',
               'model_tokenizer_dataset_split_seed_cohort_checkpoint_endpoint_frozen_criteria': 'Source-declared evidence remains declarative. Absent dimensions in saved fields are UNKNOWN, not inferred from shared implementation or neighboring runs.',
               'model_identity': 'See source-reported evidence; no original-run authentication added.',
               'tokenizer_identity': 'See source-reported evidence; missing/unvalidated identities retained.',
               'frozen_criteria': 'Design criteria in source do not authenticate their adoption by a run.'},
           'unavailable_source_locators': unavailable, 'exact_remaining_evidence': required,
           'wave03_exact_identity_revision_matches': historical_revision_links,
           'wave03_original_literal_relationship_matches': literal_links,
           'wave03_dependency_version_matches': dep_source_matches,
           'prior_source_backed_historical_revisions': [{'locator': evidence_ref(c), 'record': c} for c in prior_revisions
                                                        if rid == 'RS126' and c.get('path') == source['path']],
           'dependency_resolution': 'SOURCE_HISTORY_CONTEXT_UPDATED_SPECIFIC_RUN_OR_BOUNDARY_DEBT_RETAINED' if matches else 'NO_EXACT_WAVE03_ADJUDICATION_SUPPLIES_MISSING_RUN_OR_BOUNDARY_EVIDENCE',
           'protected_dependency': rid in ('RS069', 'RS124', 'RS065', 'RS101', 'RS126'),
           'canonical_ledger_action': 'PRESERVE_UNCHANGED', 'scientific_validation': 'NOT_PERFORMED_NOT_AUTHORIZED'}
    decisions.append(row)
    for ref in refs:
        existing = [m for m in prior_memberships if m['work_id'] == wid and m['study_reference'] == ref]
        assert len(existing) <= 1
        if existing:
            m = existing[0]
            new_memberships.append({'original_queue_id': wid, 'source_record_id': rid, 'source_identity': source,
                                    'study_reference': ref, 'membership_status': 'PROPOSED_NOT_APPLIED_NOT_CONFIRMED',
                                    'record_role': role, 'basis': m['basis'], 'original_evidence': m.get('evidence', prior['associated_nonempty_sources'] if wid in PE else []),
                                    'prior_membership_locator': evidence_ref(m), 'membership_role': m['membership_role'],
                                    'independent_replication_established': False, 'counts_as_independent_study': False})
        else:
            assert rid in special
            new_memberships.append({'original_queue_id': wid, 'source_record_id': rid, 'source_identity': source,
                                    'study_reference': ref, 'membership_status': 'PROPOSED_NOT_APPLIED_BOUNDARY_OPEN',
                                    'record_role': role, 'basis': special[rid]['design'], 'original_evidence': evidence,
                                    'prior_membership_locator': evidence_ref(OL[rid]), 'membership_role': 'SOURCE_SPECIFIC_UNRESOLVED_BOUNDARY',
                                    'independent_replication_established': False, 'counts_as_independent_study': False})

assert len(decisions) == len({r['original_queue_id'] for r in decisions}) == 76
assert len(new_memberships) == 63 and len({(r['original_queue_id'], r['study_reference']) for r in new_memberships}) == 63
assert {r['source_record_id'] for r in decisions if r['protected_dependency']} == set(special) | {'RS069', 'RS124'}

# Debt keys identify an exact source/edge or a source-backed shared artifact
# family. Same textual boilerplate or similar filenames never merge tasks.
debt = {}


def add_debt(key, classification, kind, qids, source_ids, source_identities, missing, evidence, blocking=True):
    row = debt.setdefault(key, {'debt_id': 'DEBT-' + hashlib.sha256(key.encode()).hexdigest()[:20],
                               'deduplication_key': key, 'classification': classification,
                               'debt_kinds': [], 'original_queue_ids': [], 'stable_original_source_ids': [],
                               'original_source_identities': [], 'exact_missing_evidence': [], 'evidence_locators': [],
                               'blocks_strict_census_completeness': blocking, 'scientific_execution_authorized': False})
    assert row['classification'] == classification
    for field, values in [('debt_kinds', [kind]), ('original_queue_ids', qids), ('stable_original_source_ids', source_ids),
                          ('original_source_identities', source_identities), ('exact_missing_evidence', missing), ('evidence_locators', evidence)]:
        for value in values:
            if value not in row[field]:
                row[field].append(value)


content_keys = {}
for r in content:
    source = r['source_identity']
    key = 'source-semantics:' + source['path'] + ':' + source['blob']
    content_keys[(source['path'], source['blob'])] = key
    g = CG[r['id']]
    add_debt(key, 'CENSUS_BLOCKING', 'DAMAGED_SOURCE' if r['disposition'] == 'BOUNDED_SOURCE_DAMAGE' else 'INCOMPLETE_SOURCE_SEMANTICS',
             [r['id']], r['original_source_ids'], [source], [x['missing'] for x in g['gaps']], [evidence_ref(r), evidence_ref(g)])
for r in reconciled:
    if r['category'] != 'HISTORICAL_TRANSITION_INTERPRETATION' or r['interpretation_complete']:
        continue
    s = r['endpoint_identities']['after']
    key = content_keys.get((s['path'], s['blob']), 'transition:' + r['original_queue_id'])
    missing = []
    for g in r['remaining_gap_records']:
        v = g['record'].get('exact_missing_information', g['record'].get('needed_for_final_consolidation'))
        missing.extend(v if isinstance(v, list) else [v])
    add_debt(key, 'CENSUS_BLOCKING', 'HISTORICAL_TRANSITION_INTERPRETATION', [r['original_queue_id']],
             r['original_source_ids'], r['original_source_identities'], missing, r['evidence_records'])

# Explicit same-lineage groups retain source-specific requirements inside one
# artifact request. They do not claim equivalent cohorts or independent runs.
SHARED_STUDY_DEBT = {
    **{s: 'manual-review-stage180-original-package-and-launch-provenance' for s in ['RS093', 'SU02559', 'SU02570']},
    **{s: 'text-location-frozen-prediction-input-and-analysis-command-lineage' for s in ['RS075', 'RS076', 'RS077', 'RS078', 'RS079']},
    **{s: 'route-order-frozen-export-and-v1-v2-analysis-lineage' for s in ['RS080', 'RS081', 'RS083', 'RS084']},
    **{s: 'slot-guard-original-export-and-threshold-analysis-lineage' for s in ['RS073', 'RS074']},
    **{s: 'location-boundary-original-clean-external-export-lineage' for s in ['RS120', 'RS121']},
    **{s: 'structured-coverage-original-owner-composer-and-prediction-lineage' for s in ['RS131', 'RS132', 'RS133', 'RS134', 'RS135']},
    **{s: 'stage196-p0-r1-original-run-and-counterfactual-analysis-lineage' for s in ['RS109', 'RS110', 'RS111', 'RS112', 'RS113', 'RS114']},
    **{s: 'stage196b1-original-runtime9835cbbf-and-propagation-analysis-lineage' for s in ['RS105', 'RS106']},
    **{s: 'stage39c40-original-composer-report-and-recovery-chain' for s in ['RS145', 'RS147']},
    **{s: 'phasef-name-original-q1-q3-secondary-kinematic-lineage' for s in ['RS046', 'RS048', 'RS050']},
}
for d in decisions:
    key = 'study-evidence:' + SHARED_STUDY_DEBT.get(d['source_record_id'], d['source_record_id'])
    add_debt(key, 'CENSUS_BLOCKING', 'STUDY_IDENTITY_RUN_OR_BOUNDARY', [d['original_queue_id']],
             d['original_source_ids'], d['all_original_source_identity_aliases'], d['exact_remaining_evidence'],
             [d['prior_adjudication_locator'], d['original_queue_locator']])
    missing_companion_ids = []
    for unavailable in d['unavailable_source_locators']:
        companion_key = 'original-literal-companion:' + unavailable['commit_or_tree'] + ':' + unavailable['path']
        add_debt(companion_key, 'CENSUS_BLOCKING', 'UNAVAILABLE_ORIGINAL_LITERAL_COMPANION',
                 [d['original_queue_id']], d['original_source_ids'], [],
                 ['Authenticate/resolve the exact original literal reference ' + unavailable['path'] + ' at ' + unavailable['commit_or_tree'] + '. Recorded absence is local to the prior inspected object; global absence and a renamed substitute are not established.'],
                 [d['prior_adjudication_locator']])
        debt[companion_key]['original_literal_locator'] = unavailable
        missing_companion_ids.append(debt[companion_key]['debt_id'])
    debt[key].setdefault('dependent_source_requirements', []).append({
        'source_record_id': d['source_record_id'], 'original_queue_id': d['original_queue_id'],
        'missing_identity_dimensions': d['missing_or_unvalidated_identity_dimensions'],
        'shared_missing_companion_debt_ids': missing_companion_ids,
        'exact_remaining_evidence': d['exact_remaining_evidence']})
    debt[key]['deduplication_boundary'] = 'One coordinated original-artifact request; source-specific claims and requirements remain separate. No same-run equality is inferred.'
for g in PG:
    q = Q[g['queue_id']]
    bounded = g['disposition'] == 'BOUNDED_LIMITATION'
    add_debt('provenance:' + q['id'], 'BOUNDED_HISTORICAL_LIMITATION' if bounded else 'CENSUS_BLOCKING',
             q['category'], [q['id']], q['original_source_ids'], q['original_source_identities'],
             g['exact_missing_information'], [evidence_ref(g)], blocking=g.get('blocks_source_census_closure', False))
    debt['provenance:' + q['id']]['preserved_saved_scope_and_flags'] = g

# Nonblocking transition interpretation does not close its original run/cohort
# adoption question. The global union portion is now closed; no edge is reopened.
extras = [g for g in sessions['transitions_1']['gaps'] if g.get('scope') == 'NONBLOCKING_TRANSITION_MEANING_STUDY_CONSOLIDATION_DEPENDENCY']
assert len(extras) == 1
for g in extras:
    ids = g['original_queue_ids']
    add_debt('study-execution:' + ids[0], 'CENSUS_BLOCKING', 'ORIGINAL_FUSED_STREAMING_RUN_IDENTITY', ids,
             Q[ids[0]]['original_source_ids'], Q[ids[0]]['original_source_identities'],
             [g['needed_for_final_consolidation']], [evidence_ref(g)])
global_gap = [g for g in sessions['transitions_1']['gaps'] if g.get('scope') == 'FINAL_STUDY_CONSOLIDATION_ONLY']
assert len(global_gap) == 1
add_debt('bounded-selected-ancestry-versus-study-adoption', 'BOUNDED_HISTORICAL_LIMITATION',
         'BRANCH_ADOPTION_AND_NON_DIRECT_EDGE_SCOPE', global_gap[0]['original_queue_ids'], [], [],
         ['216-ID shard union is resolved here. Selected ancestry alone does not establish exhaustive intervening path history, study-wide branch adoption or execution/correction timing. Specific original artifacts remain in study debt; no completed transition is reopened.'],
         [evidence_ref(global_gap[0])], blocking=False)

deferred = []
for q in queue:
    if q['category'] == 'SCIENTIFIC_NUMERICAL_VALIDATION_DEFERRED':
        row = {'original_queue_id': q['id'], 'classification': 'SCIENTIFIC_VALIDATION_DEFERRED',
               'disposition': 'DEFERRED_UNCHANGED', 'original_queue_locator': evidence_ref(q),
               'original_queue_record': q, 'blocks_historical_census_closure': False,
               'validation_performed': False, 'scientific_execution_authorized': False}
        deferred.append(row)
        add_debt('scientific-validation:' + q['id'], 'SCIENTIFIC_VALIDATION_DEFERRED', q['category'],
                 [q['id']], q['original_source_ids'], q['original_source_identities'], q['exact_missing_information'],
                 [evidence_ref(q)], blocking=False)
assert len(deferred) == 8

all_covered = {r['original_queue_id'] for r in reconciled + decisions + deferred}
assert all_covered == set(Q) and len(reconciled) + len(decisions) + len(deferred) == 409
debt_rows = sorted(debt.values(), key=lambda r: (r['classification'], r['deduplication_key']))
debt_owned_ids = {s for r in debt_rows for s in r['original_queue_ids']}
must_remain = {r['original_queue_id'] for r in reconciled if not r['census_closure'].startswith('INTERPRETATION_RECONSTRUCTED')} | {r['original_queue_id'] for r in decisions + deferred}
assert must_remain <= debt_owned_ids
assert len({r['debt_id'] for r in debt_rows}) == len(debt_rows)

save('wave03_result_reconciliation.jsonl', sorted(reconciled, key=lambda r: r['original_queue_id']))
save('study_identity_76.jsonl', sorted(decisions, key=lambda r: r['original_queue_id']))
save('provisional_study_memberships.jsonl', sorted(new_memberships, key=lambda r: (r['study_reference'], r['original_queue_id'])))
save('unresolved_census_debt.jsonl', debt_rows)
save('deferred_scientific_validation.jsonl', deferred)

# Reports/context fingerprints and original canonical files are part of the
# unchanged-input contract. Original binaries are never opened by this script.
for p in [ROOT / 'AGENTS.md', BASE / 'wave_02/study_adjudication/report.md',
          BASE / 'study_identity_116/report.md', BASE / 'wave_02/integration_02/integration_report.md',
          BASE / 'wave_02/integration_02/input_fingerprints.json']:
    remember(p)
unchanged = {key: digest(ROOT / key) == rec['sha256'] for key, rec in FP.items()}
assert all(unchanged.values())
save('input_fingerprints.json', {'schema_version': 'W03_CONSOLIDATION_INPUT_FINGERPRINTS_V1',
                               'authority': 'Current user-directed historical census consolidation only',
                               'files': FP, 'declaration_checks': hash_checks,
                               'integration02_parent_fingerprint_checks': parent_checks,
                               'unchanged_after_consolidation': unchanged,
                               'git_object_authentication': 'Existing source-backed checks reused; not regenerated',
                               'binary_sources_reopened': False})

counts = collections.Counter(r['disposition'] for r in decisions)
role_counts = collections.Counter(r['record_role_classification'] for r in decisions)
refs = {r['study_reference'] for r in new_memberships}
candidate_refs = sorted(r for r in refs if r.startswith('CANDIDATE'))
study_refs = sorted(r for r in refs if r.startswith('STUDY'))
shard_counts = {s: {'targets': 108,
                    'reconstructed': sum(r['interpretation_complete'] for r in reconciled if r['session'] == s),
                    'open': sum(not r['interpretation_complete'] for r in reconciled if r['session'] == s)}
                for s in ('transitions_0', 'transitions_1')}
assert shard_counts == {'transitions_0': {'targets': 108, 'reconstructed': 84, 'open': 24},
                        'transitions_1': {'targets': 108, 'reconstructed': 97, 'open': 11}}
assert collections.Counter(r['disposition'] for r in content) == {'SCIENTIFIC_ROLE_ESTABLISHED': 99, 'BOUNDED_SOURCE_DAMAGE': 4}
assert len(candidate_refs) == 40 and len(study_refs) == 5
commands = ['git status --short', 'git branch --show-current', 'git rev-parse HEAD',
            'rg --files audits/full_history_20261013/parallel_20261013',
            'rg --files audits/full_history_20261013 (filtered for prior ledger/inventory paths)',
            'Get-Content AGENTS.md and the saved Wave03/prior study reports',
            'PowerShell inline Python saved JSON/JSONL schema, counts, excerpts, exact-path and SHA256 inspections',
            'python audits/full_history_20261013/parallel_20261013/wave_03/study_consolidation/consolidate.py']
validation = {'status': 'PASS_STRUCTURAL_RECONCILIATION_RESEARCH_CENSUS_PARTIAL',
              'authority': 'Current user task; existing evidence; AGENTS.md',
              'phase': 'HISTORICAL_CENSUS_CONSOLIDATION_ONLY',
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'input_integrity_verdict': 'SAVED_OUTPUT_HASHES_AND_CONSUMED_INPUT_PRESERVATION_PASS',
              'input_file_count': len(FP), 'declared_wave03_output_hashes_checked': len(hash_checks),
              'integration02_parent_fingerprints_checked': len(parent_checks),
              'unchanged_input_count': sum(unchanged.values()), 'queue_schema_fields': sorted(queue[0]),
              'queue_category_counts': dict(categories), 'queue_unique_ids': 409,
              'wave03_targets': 325, 'wave03_exact_unique_original_queue_id_coverage': True,
              'wave03_ids': sorted(expected_wave), 'shard_transition_counts': shard_counts,
              'transitions_total_reconstructed': 181, 'transitions_total_open': 35,
              'saved_shard1_unresolved_placeholders': 108, 'explained_historical_status_conflicts': status_conflicts,
              'unexplained_status_conflicts': [], 'content_roles_established': 99, 'content_damage_sources': 4,
              'content_full_semantic_gaps_closed': 0, 'content_tasks_remaining': 103,
              'provenance_dispositions': {'EVIDENCE_MISSING': 4, 'BOUNDED_LIMITATION': 2},
              'study_targets': 76, 'study_ids': sorted(r['id'] for r in study_targets),
              'study_disposition_counts': dict(counts), 'study_record_role_counts': dict(role_counts),
              'study_identity_dependencies_resolved': 0, 'study_identity_dependencies_open': 76,
              'study_exact_wave03_source_revision_matches': sum(len(d['wave03_exact_identity_revision_matches']) for d in decisions),
              'study_literal_path_context_leads': sum(len(d['wave03_original_literal_relationship_matches']) for d in decisions),
              'literal_path_context_is_not_run_or_membership_binding': True,
              'protected_dependencies_preserved': ['RS069', 'RS124', 'RS065', 'RS101', 'RS126'],
              'provisional_membership_rows': len(new_memberships), 'provisional_candidate_reference_count': len(candidate_refs),
              'provisional_candidate_references': candidate_refs, 'existing_provisional_study_references': study_refs,
              'confirmed_independent_studies_added': 0, 'confirmed_memberships_added': 0,
              'baseline_study_ledger_count': len(ledger), 'baseline_study_identity_statuses': dict(collections.Counter(r['identity_status'] for r in ledger)),
              'baseline_candidate_groups': len({r['candidate_study_id'] for r in old_links if r.get('candidate_study_id')}),
              'canonical_ledgers_modified': False, 'unsupported_replications_added': 0,
              'source_inventory': {'source_universe_records': len(inventory), 'historical_version_records': len(versions),
                                   'source_to_study_records': len(mapping), 'target_inventory_joins': 76,
                                   'records_deleted': 0, 'ledger_rewrites': 0,
                                   'untargeted_sources': 'Unchanged full inventory retained and fingerprinted; no claims about new semantic closure'},
              'deduplicated_debt_records': len(debt_rows),
              'debt_classification_counts': dict(collections.Counter(r['classification'] for r in debt_rows)),
              'strict_blocking_debt_records': sum(r['blocks_strict_census_completeness'] for r in debt_rows),
              'deferred_scientific_validation': 8, 'all409_queue_records_accounted': True,
              'ready_for_independent_completeness_verification': False,
              'reason_not_ready': '35 open transitions; 103 incomplete/damaged source tasks; 76 run/boundary dependencies; four unauthenticated graph correspondences; strict capture chronology unresolved.',
              'scientific_truth_declared': False, 'training_evaluation_inference_red_team': 'NOT_RUN',
              'commit_push_git_mutation': 'NOT_RUN', 'commands_executed': commands,
              'tooling_warning': 'Python reports a stale interpreter-location warning; successful executed commands exit zero.',
              'output_counts': {'wave03_result_reconciliation.jsonl': 325, 'study_identity_76.jsonl': 76,
                                'provisional_study_memberships.jsonl': len(new_memberships),
                                'unresolved_census_debt.jsonl': len(debt_rows), 'deferred_scientific_validation.jsonl': 8}}
save('coverage_validation.json', validation)
print(json.dumps({k: validation[k] for k in ['status', 'input_file_count', 'shard_transition_counts',
                                           'study_disposition_counts', 'study_record_role_counts',
                                           'provisional_candidate_reference_count', 'debt_classification_counts']}, indent=2))
