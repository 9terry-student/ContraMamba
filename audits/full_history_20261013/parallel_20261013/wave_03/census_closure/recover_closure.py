"""Census-only saved-evidence reconciliation; no historical program executes.

Deterministic outputs and checkpoints. Run from repository root with Python.
Only this directory is writable. Original bytes/ledger memberships remain intact.
"""
import collections
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
AUDIT = ROOT / 'audits/full_history_20261013'
OUT = Path(__file__).resolve().parent
SC = AUDIT / 'parallel_20261013/wave_03/study_consolidation'
COMMANDS = []
READS = {}

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(x):
    return json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(',', ':'))

def save(name, value):
    p = OUT / name
    assert p.parent == OUT
    if name.endswith('.jsonl'):
        text = ''.join(canonical(x) + '\n' for x in value)
    else:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + '\n'
    p.write_text(text, encoding='utf-8', newline='\n')

def load(p):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    k = p.relative_to(ROOT).as_posix()
    READS[k] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    if p.suffix == '.jsonl':
        return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    return json.loads(p.read_text(encoding='utf-8'))

def git(*args):
    COMMANDS.append(['git', *args])
    r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=False)
    assert r.returncode == 0, (args, r.stderr.decode('utf-8', 'replace'))
    return r.stdout.decode('utf-8', 'replace').strip()

def identity_key(s):
    return tuple(s.get(k) for k in ('commit', 'path', 'blob'))

def locator(p, line):
    return {'file': p.relative_to(ROOT).as_posix(), 'jsonl_line': line, 'file_sha256': sha(p)}

def unique(values):
    return [json.loads(x) for x in sorted(set(canonical(x) for x in values))]

baseline = load(OUT / 'input_preservation_checkpoint.json')
head = git('rev-parse', 'HEAD')
initial_status = git('status', '--short', '--branch')
assert head == 'ff806bd5bd8efc62fcf6420cf8b097a63ed70571'
declared = load(SC / 'input_fingerprints.json')
assert all(sha(ROOT / p) == v['sha256'] for p, v in declared['files'].items())
coverage = load(SC / 'coverage_validation.json')
debts = load(SC / 'unresolved_census_debt.jsonl')
blocking = sorted((x for x in debts if x['classification'] == 'CENSUS_BLOCKING'), key=lambda x:x['debt_id'])
bounded = [x for x in debts if x['classification'] == 'BOUNDED_HISTORICAL_LIMITATION']
deferred = [x for x in debts if x['classification'] == 'SCIENTIFIC_VALIDATION_DEFERRED']
assert len(blocking) == len({x['debt_id'] for x in blocking}) == 211
assert len(bounded) == 3 and len(deferred) == 8
assert sum(x['blocks_strict_census_completeness'] for x in debts) == 212
assert coverage['debt_classification_counts'] == dict(collections.Counter(x['classification'] for x in debts))
studies = load(SC / 'study_identity_76.jsonl')
assert len(studies) == len({x['original_queue_id'] for x in studies}) == 76
study_index = {x['original_queue_id']:x for x in studies}
queue_file = AUDIT / 'parallel_20261013/wave_02/integration_02/remaining_census_work_queue.jsonl'
queue = load(queue_file)
queue_index = {x['id']:(i,x) for i,x in enumerate(queue,1)}
assert len(queue_index) == 409
universe = load(AUDIT / 'expansion_03/source_universe_manifest.jsonl')
versions = load(AUDIT / 'expansion_03/historical_source_versions.jsonl')
assert len(universe) == 4474 and len(versions) == 1565
saved_cache = {}

def prior_ref(ref):
    p = ROOT / ref['file']
    if str(p) not in saved_cache:
        saved_cache[str(p)] = load(p)
    r = saved_cache[str(p)][ref['jsonl_line']-1]
    # Complete rows remain immutable at hash-bound locators, without duplicating
    # huge reversible source payloads into each derived debt record.
    keys = ['disposition','status','census_sufficient','investigation_complete',
            'scope','reason','summary','interpretation','exact_missing_information',
            'remaining_requirements','scientific_role','source_identity',
            'source_identity_authentication','source_content_status', 'damage_status',
            'revision_summary','additional_consolidation_limit','negative_evidence_scope',
            'source_reference_literals','declared_dependencies','original_dependency_work_ids',
            'dependencies','execution_adoption_limit','chronology_limit']
    return {**ref, 'file_sha256':READS[p.relative_to(ROOT).as_posix()]['sha256'],
            'record_sha256':hashlib.sha256(canonical(r).encode()).hexdigest(),
            'saved_record_fields':sorted(r),
            'verified_saved_conclusion':{k:r[k] for k in keys if k in r},
            'preservation':'ENTIRE_RECORD_RETAINED_AT_HASH_BOUND_LOCATOR; NO_PRIOR_FINDING_DELETED'}

manifest = []
for d in blocking:
    qs = []
    ss = []
    for q in d['original_queue_ids']:
        if q in queue_index:
            i,r = queue_index[q]
            qs.append({'locator':locator(queue_file,i), 'original_queue_record':r})
        if q in study_index:
            s = study_index[q]
            i = next(i for i,x in enumerate(studies,1) if x['original_queue_id']==q)
            ss.append({'locator':locator(SC/'study_identity_76.jsonl',i),
                       'source_record_id':s['source_record_id'],
                       'source_identity':s['source_identity'],
                       'historical_role':s['historical_role_and_execution_distinction'],
                       'provisional_references':s['provisional_study_references'],
                       'negative_failed_blocked_abandoned_superseded_evidence':s['negative_failed_blocked_abandoned_superseded_evidence'],
                       'exact_remaining_evidence':s['exact_remaining_evidence'],
                       'independent_execution_established':False,
                       'replication_established':False})
    manifest.append({'debt_id':d['debt_id'], 'original_debt_record':d,
                     'original_research_lineage_and_queue_payloads':qs,
                     'prior_investigations':[prior_ref(r) for r in d['evidence_locators']],
                     'study_context':ss,
                     'dependencies_on_original_debt_ids':sorted(set(z for a in d.get('dependent_source_requirements',[]) for z in a.get('shared_missing_companion_debt_ids',[]))),
                     'exact_source_identity_preserved':d['original_source_identities'],
                     'resolution_dependencies_remain_open':True})
save('frozen_211_target_manifest.jsonl',manifest)
save('batch_01_checkpoint.json',{'phase':'FROZEN', 'processed_debt_ids':[x['debt_id'] for x in blocking], 'unprocessed_ids':[], 'input_counts':{'blockers':211,'bounded':3,'deferred':8,'study':76}})

# Specific uncovered deficiency: prior literal absence was bounded to one tree.
# Join the existing inventory, then probe those exact paths in registered
# worktrees. No Git history enumeration, parsing, glob scan or rename inference.
wt_text = git('worktree','list','--porcelain')
worktrees = [Path(x[9:]) for x in wt_text.splitlines() if x.startswith('worktree ')]
local_probes = {}
for d in blocking:
    if 'original_literal_locator' not in d:
        continue
    lit = d['original_literal_locator']
    path = lit['path']
    inv = [x for x in universe if x['path']==path]
    hv = [x for x in versions if x['path']==path]
    probes = []
    for wt in worktrees:
        p = wt / path
        accessible = wt.exists()
        exists = p.is_file() if accessible else False
        entry = {'worktree':wt.as_posix(), 'worktree_accessible':accessible,
                 'exact_literal_path':path, 'present':exists}
        if exists:
            entry.update({'sha256':sha(p),'bytes':p.stat().st_size,
                          'provenance_binding':'UNBOUND_LOCAL_COPY_NOT_AUTOMATICALLY_ORIGINAL'})
        probes.append(entry)
    local_probes[d['debt_id']] = {'literal':lit, 'exact_saved_inventory_matches':inv,
                                'exact_saved_historical_version_matches':hv,
                                'exact_path_worktree_probes':probes,
                                'global_absence_established':False,
                                'renamed_substitution_authorized':False,
                                'lfs_probe':'NO_ORIGINAL_POINTER_OR_OID_FOR_THIS_LITERAL; content-addressed LFS cannot identify a missing pathname. Prior known LFS recovery reused, not repeated.'}
save('batch_02_local_probe_checkpoint.json', {'worktree_inventory':wt_text, 'literal_probes':local_probes,
                                           'processed_debt_ids':sorted(local_probes), 'unprocessed_literal_ids':[]})

# Recover RS044's missing documentary edge solely from authenticated saved
# excerpts. This closes its stated supporting-source requirement, not a study.
rs044 = next(x for x in studies if x['source_record_id']=='RS044')
support_debt = next(x for x in blocking if x['stable_original_source_ids']==['RS044'])
src = rs044['source_identity']
assert src['commit']=='ea03cdaa5394de03389d31aa1089b51e10340a57'
e = rs044['original_source_evidence']
assert all(any(a['line']==n and token in a['excerpt'] for a in e) for n,token in [(9,'final_paper_claim_disposition.md'),(11,'paper_novelty_positioning_matrix.md'),(14,'authorizes no new experiment')])
ctx = rs044['associated_original_authority_implementation_execution_result_sources']
assert {x['authentication']['source_identity']['blob'] for x in ctx} == {'04c4e8a79c443fe9dfdf6a7561ba29ef1e596a4a','20744d8673102312d0a8a8581784540706da885c'}
assert all(x['authentication']['commit_path_blob_match'] and x['authentication']['full_text_bytes_read'] and x['authentication']['source_identity']['commit']==src['commit'] for x in ctx)
documentary_closure = {
    'source_identity':src,
    'source_edges':[{'originating_lines':[8,9] if 'claim_disposition' in x['authentication']['source_identity']['path'] else [10,11],
                     'destination_identity':x['authentication']['source_identity'],
                     'saved_original_authentication':x['authentication'],
                     'selected_status_excerpts':[z for z in x['evidence'] if z['line'] in ([5,11,14,152,154] if 'claim_disposition' in x['authentication']['source_identity']['path'] else [5,16,204,206,218])]} for x in ctx],
    'finding':'The manuscript spine explicitly selects both exact same-tree documentary versions. The claim set is marked frozen; the novelty document remains a freeze candidate/ready-for-freeze. No finalized novelty adoption beyond this documentary selection is inferred.',
    'closure_definition':'Exact claim-disposition and novelty source/version links for a supporting manuscript source; no execution identity required by the original requirement.',
    'scientific_results_certified':False, 'confirmed_membership_added':False,
    'new_independent_execution_or_replication':False,
    'exclusion':'No model/checkpoint/seed/run requirement manufactured for RS044; all empirical claims remain reported claims.'}

# Uncovered companion identity at two EXACT already-named original commits.
# This is not a historical scan or repeat of source parsing. Only the missing
# observer relationship is inspected; no observer code or metric executes.
o0c_debt=next(x for x in blocking if x['stable_original_source_ids']==['HV000971'])
observer_path='scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py'
observer_before='e03b38fa0681fd04d8633a7de184559bd0473133'
observer_after='60a53b6f5d1db8d7cbecded5b94d5231adcfc520'
observer_evidence=[]
for c,expected in [(observer_before,'7ac5d0d5e93742c8d76cd3e467790a1866dbade9'),(observer_after,'49faa5eeb659ce7fbec7004e0933deaa6fb08c3b')]:
    blob=git('rev-parse',c+':'+observer_path)
    assert blob==expected
    COMMANDS.append(['git','cat-file','blob',blob])
    data=subprocess.check_output(['git','cat-file','blob',blob],cwd=ROOT)
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==blob
    lines=data.decode('utf-8-sig').splitlines()
    observer_evidence.append({'source_identity':{'commit':c,'path':observer_path,'blob':blob},
                              'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                              'git_blob_sha1_recomputed':blob,
                              'line_indexed_original_excerpts':[{'line':i,'text':x} for i,x in enumerate(lines,1) if 368<=i<=409]})
observer_diff=git('diff','--no-ext-diff','--no-textconv','--unified=6',observer_before,observer_after,'--',observer_path)
assert '\n-       for anchor in ANCHOR_ORDER:' in observer_diff
assert '\n+      for anchor in ANCHOR_ORDER:' in observer_diff
assert '\n+       for layer in layers:' in observer_diff
assert sum(x.startswith('@@') for x in observer_diff.splitlines())==1
for ev in observer_evidence:
    assert any('expected=[(p,c,a,l)' in x['text'] and 'for a in ANCHOR_ORDER for l in layers' in x['text'] for x in ev['line_indexed_original_excerpts'])
o0c_closure={'source_debt_identity':o0c_debt['original_source_identities'],
             'observer_endpoint_authentication':observer_evidence,
             'exact_diff':observer_diff,'diff_sha256':hashlib.sha256(observer_diff.encode()).hexdigest(),
             'uncovered_deficiency':'Runner changed an exact observer commit pin; prior packet had not integrated the companion source delta.',
             'historical_identity_finding':'The pinned companion changes measurements traversal from pair/comparison/layer/anchor to pair/comparison/anchor/layer. All layer pre-divergence assertions now precede measurement emission for that comparison. The unchanged coordinate validator requires pair/comparison/anchor/layer order. The diff changes loop order, not metric expressions, cohort constants, model/checkpoint definitions or frozen observer policy.',
             'closure_definition':'Integrate the exact changed observer companion for mechanism-level historical source interpretation, preserving the already reconstructed runner-pin change.',
             'limits':'Static source relationship only. No run adoption, successful execution, numerical result, replication or later artifact identity certified.'}
save('batch_02b_o0c_companion_checkpoint.json',o0c_closure)

# Three demonstrated same compiled endpoint overlaps. Union the obligations,
# rather than dropping a transition's additional semantic comparison.
graph = [x for x in blocking if x['debt_kinds']==['SOURCE_RELATIONSHIP_UNCERTAINTY']]
trans = [x for x in blocking if x['debt_kinds']==['HISTORICAL_TRANSITION_INTERPRETATION']]
owners = {x['debt_id']:x['debt_id'] for x in blocking}
duplicate_basis = {}
for t in trans:
    for g in graph:
        if set(identity_key(x) for x in t['original_source_identities']) & set(identity_key(x) for x in g['original_source_identities']):
            owners[t['debt_id']] = g['debt_id']
            duplicate_basis[t['debt_id']] = {'owner_debt_id':g['debt_id'], 'exact_shared_identity':g['original_source_identities'],
                                          'limit':'Shared original build/source correspondence only. Transition comparison and scientific-semantic interpretation remain separate obligations in this root.'}
assert len(duplicate_basis)==3
groups = collections.defaultdict(list)
for d in blocking:
    groups[owners[d['debt_id']]].append(d)
root_for = {d['debt_id']:'ROOT-'+owners[d['debt_id']][5:] for d in blocking}
PYTEST_ROOT = 'ROOT-SHARED-PYTEST-9.1.1-ORIGINAL-PRODUCER-OPTIONS'
pytest_debts = [x for x in blocking if any('pytest-9.1.1.pyc' in s['path'] for s in x['original_source_identities'])]

def classification(d):
    k = d['debt_kinds'][0]
    if d['debt_id']==support_debt['debt_id']:
        return 'RESOLVED_BY_EXISTING_EVIDENCE'
    if d['debt_id']==o0c_debt['debt_id']:
        return 'RESOLVED_BY_NEW_LOCAL_EVIDENCE'
    if d['debt_id'] in duplicate_basis:
        return 'DUPLICATE_DEPENDENCY_WITH_OPEN_ROOT'
    if k in ('INCOMPLETE_SOURCE_SEMANTICS','DAMAGED_SOURCE'):
        return 'SOURCE_PRESENT_BUT_SEMANTICS_UNRESOLVED'
    if k in ('UNAVAILABLE_ORIGINAL_LITERAL_COMPANION','SOURCE_RELATIONSHIP_UNCERTAINTY'):
        return 'EXTERNAL_EVIDENCE_REQUIRED'
    if k=='STUDY_IDENTITY_RUN_OR_BOUNDARY':
        # Documentary/version-only requirements have additional read-only work;
        # they are not relabeled missing external runs simply to shorten a list.
        ids=set(d['stable_original_source_ids'])
        if ids & {'RS062','RS065','RS101','RS125','RS126'}:
            return 'SOURCE_PRESENT_BUT_SEMANTICS_UNRESOLVED'
        return 'EXTERNAL_EVIDENCE_REQUIRED'
    if k=='ORIGINAL_FUSED_STREAMING_RUN_IDENTITY':
        return 'UNRESOLVED'
    if k=='HISTORICAL_TRANSITION_INTERPRETATION':
        s=d['original_source_identities'][0]
        if s['path'].endswith('.pyc') or 'HV000049' in d['stable_original_source_ids']:
            return 'EXTERNAL_EVIDENCE_REQUIRED'
        if 'Original full companion authority/artifact' in ' '.join(d['exact_missing_evidence']):
            return 'UNRESOLVED'
        return 'SOURCE_PRESENT_BUT_SEMANTICS_UNRESOLVED'
    return 'UNRESOLVED'

results = []
roots = []
for owner,ds in sorted(groups.items()):
    rid='ROOT-'+owner[5:]
    ids=sorted(x['debt_id'] for x in ds)
    qs=sorted(set(q for x in ds for q in x['original_queue_ids']))
    ss=sorted(set(s for x in ds for s in x['stable_original_source_ids']))
    requirements=[{'debt_id':x['debt_id'],'exact_missing_evidence':x['exact_missing_evidence'],
                   'source_specific_requirements':x.get('dependent_source_requirements',[])} for x in ds]
    refs=[prior_ref(r) for x in ds for r in x['evidence_locators']]
    deps=sorted(set(root_for[z] for x in ds for a in x.get('dependent_source_requirements',[]) for z in a.get('shared_missing_companion_debt_ids',[])))
    if any(x in pytest_debts for x in ds):
        deps.append(PYTEST_ROOT)
    states=collections.Counter(classification(x) for x in ds)
    resolved=all(classification(x).startswith('RESOLVED_') for x in ds)
    kind=ds[0]['debt_kinds'][0]
    if owner==o0c_debt['debt_id']:
        action='No further source recovery for HV000971 observer-pin/loop-order relationship. Original execution validity remains outside this source-interpretation closure.'
    elif resolved:
        action='No further recovery for RS044 documentary edge. Preserve publication-support role and provisional ledger.'
    elif kind=='UNAVAILABLE_ORIGINAL_LITERAL_COMPANION':
        action='Recover the exact original literal artifact with producer/version/cohort binding, or an authenticated historical rename/correction record; do not substitute a similarly named JSONL.'
    elif any(classification(x)=='EXTERNAL_EVIDENCE_REQUIRED' for x in ds):
        action='Recover the original provenance-bound bundle specified in source-specific obligations. Read manifests/commands/identity hashes only; do not reproduce a run or verify numerical truth.'
    else:
        action='Read only the saved original clauses, complete functions/control flow, companion records and exact before/after evidence named by this root; write a source-specific interpretation with each outstanding obligation addressed.'
    local_refs = [r for q in qs for r in next((m['study_context'] for m in manifest if q in m['original_debt_record']['original_queue_ids']),[]) if r.get('locator')]
    root={'root_requirement_id':rid,'classification':'CENSUS_RECOVERY_REQUIREMENT',
          'dependent_original_debt_ids':ids,'dependent_original_queue_ids':qs,'dependent_original_source_ids':ss,
          'exact_original_source_identities':unique(s for x in ds for s in x['original_source_identities']),
          'recovered_source_relationship':o0c_closure if owner==o0c_debt['debt_id'] else documentary_closure if owner==support_debt['debt_id'] else None,
          'exact_missing_evidence':requirements, 'already_accessible_saved_evidence':refs,
          'study_context_locators':unique(r['locator'] for r in local_refs),
          'local_literal_probe':local_probes.get(owner),
          'original_evidence_availability':'Original source bytes/identity evidence are saved; remaining execution/adoption binding is not thereby established. Literal-specific probes, if present, control the narrow absence finding.',
          'later_assertion_only':'Source-reported identity dimensions are not promoted; inspect corresponding study_identity_76 row for located versus unlocated dimensions.',
          'global_unavailability_proven':False,
          'lfs_storage':'Known original LFS recovery is retained in Expansion 04A-R; no new missing literal has a known original LFS OID. No LFS scan or fetch repeated.',
          'underlying_dependency_root_ids':sorted(set(deps)),
          'sharing_basis':duplicate_basis.get(next((x['debt_id'] for x in ds if x['debt_id'] in duplicate_basis),''), {'basis':'EXACT_ORIGINAL_DEBT_OWNER; prior coordinated requests retained with separate source obligations, not family-based same-run grouping'}),
          'establishes_study_identity':'Only source-specific provenance-bound execution manifests can establish execution identity. Documentary edges and bytecode/source correspondences improve source documentation only.',
          'independent_execution_equality_inferred':False,'replication_equality_inferred':False,
          'resolution': classification(ds[0]) if resolved else 'OPEN',
          'blocks_full_census_closure':not resolved, 'can_be_retained_as_nonblocking_bounded_limitation':False,
          'what_read_only_can_complete':action,
          'impossible_from_current_evidence':'Original run/attempt/capture or intended damaged values cannot be manufactured from aggregate reports, filenames, AST inventories or neighboring versions.',
          'exact_next_action':action}
    roots.append(root)
    for d in ds:
        st=classification(d)
        r={'debt_id':d['debt_id'],'original_queue_ids':d['original_queue_ids'],
           'stable_original_source_ids':d['stable_original_source_ids'],
           'original_source_identities':d['original_source_identities'],
           'root_requirement_ids':[rid]+([PYTEST_ROOT] if d in pytest_debts else []),
           'status':st,'census_blocking_after_recovery':not st.startswith('RESOLVED_'),
           'unresolved_requirement':[] if st.startswith('RESOLVED_') else d['exact_missing_evidence'],
           'prior_evidence_locators':[prior_ref(z) for z in d['evidence_locators']],
           'new_local_evidence':o0c_closure if d['debt_id']==o0c_debt['debt_id'] else local_probes.get(d['debt_id']),
           'new_historical_fact':o0c_closure['historical_identity_finding'] if d['debt_id']==o0c_debt['debt_id'] else 'Exact literal is absent in the saved source inventories and narrowly probed registered worktrees, where so recorded; this is not global absence or renamed equivalence.' if d['debt_id'] in local_probes else None,
           'existing_evidence_closure':documentary_closure if d['debt_id']==support_debt['debt_id'] else None,
           'duplicate_dependency_basis':duplicate_basis.get(d['debt_id']),
           'all_original_dependencies_preserved':True,'scientific_validity_certified':False,
           'new_membership_or_replication_promoted':False,
           'external_recovery_needed':st in ('EXTERNAL_EVIDENCE_REQUIRED','DUPLICATE_DEPENDENCY_WITH_OPEN_ROOT'),
           'external_absence_scope':'Missing original binding in audited packet; no claim of exhaustive disk/Git-object absence.',
           'historical_damage_bounded':d['debt_kinds']==['DAMAGED_SOURCE'],
           'unrecoverable_from_this_blob_alone':d['debt_kinds']==['DAMAGED_SOURCE'],
           'globally_historically_unrecoverable_proven':False,
           'exact_next_action':action}
        results.append(r)
    # Each completed root is checkpointed; interrupted runs retain exact IDs.
    save('source_recovery_results.jsonl',sorted(results,key=lambda x:x['debt_id']))
save('root_evidence_requirements.jsonl',roots)

roots.append({'root_requirement_id':PYTEST_ROOT,'classification':'SHARED_CENSUS_PREREQUISITE',
              'dependent_original_debt_ids':sorted(x['debt_id'] for x in pytest_debts),
              'dependent_original_queue_ids':sorted(set(q for x in pytest_debts for q in x['original_queue_ids'])),
              'dependent_original_source_ids':sorted(set(s for x in pytest_debts for s in x['stable_original_source_ids'])),
              'exact_original_source_identities':unique(s for x in pytest_debts for s in x['original_source_identities']),
              'exact_missing_evidence':'Authenticated ORIGINAL pytest 9.1.1 assertion-rewrite implementation/options and producing environment, bound to original compiled endpoints. Modern installed pytest does not establish the historical producer.',
              'sharing_basis':'Exact embedded pytest version across identified original endpoint paths; shared producer version prerequisite, never shared execution proof.',
              'resolution':'OPEN','blocks_full_census_closure':True,'global_unavailability_proven':False,
              'underlying_dependency_root_ids':[],
              'exact_next_action':'Recover preserved historical pytest wheel/source digest, options and source-hash-bound build manifest; inspect statically. Keep every endpoint-specific correspondence and transition obligation open.'})
for b in bounded:
    roots.append({'root_requirement_id':'ROOT-'+b['debt_id'][5:], 'classification':'PRESERVED_BOUNDED_LIMITATION',
                  'dependent_original_debt_ids':[b['debt_id']], 'dependent_original_queue_ids':b['original_queue_ids'],
                  'dependent_original_source_ids':b['stable_original_source_ids'],
                  'exact_original_source_identities':b['original_source_identities'],
                  'exact_missing_evidence':b['exact_missing_evidence'], 'original_debt_record':b,
                  'resolution':'BOUNDED_OPEN_NOT_RECOVERED',
                  'blocks_full_census_closure':b['blocks_strict_census_completeness'],
                  'exact_next_action':'Recover authenticated original capture/base chronology for the snapshot; retain original strict blocking flag.' if b['debt_kinds']==['SNAPSHOT_CHRONOLOGY'] else 'Retain bounded unknowns; no completed source lineage or transition investigation reopened.'})
save('root_evidence_requirements.jsonl',roots)
result_index={x['debt_id']:x for x in results}
study_groups=[]
for d in blocking:
    if d['debt_kinds']!=['STUDY_IDENTITY_RUN_OR_BOUNDARY']:
        continue
    members=[]
    for q in d['original_queue_ids']:
        s=study_index[q]
        members.append({'original_queue_id':q,'source_record_id':s['source_record_id'],
                        'original_source_identity':s['source_identity'],
                        'saved_study_record_locator':locator(SC/'study_identity_76.jsonl',next(i for i,x in enumerate(studies,1) if x['original_queue_id']==q)),
                        'documentary_membership':s['provisional_study_references'],
                        'documentary_role':s['historical_role_and_execution_distinction'],
                        'identity_requirement_status':'DOCUMENTARY_SUPPORT_REQUIREMENT_RESOLVED' if d['debt_id']==support_debt['debt_id'] else 'OPEN',
                        'independent_execution_identity':'NOT_ESTABLISHED; not required for procedural-only source classification',
                        'replication_identity':'NOT_ESTABLISHED', 'scientific_result_validity':'NOT_ASSESSED',
                        'missing_identity_dimensions':s['missing_or_unvalidated_identity_dimensions'],
                        'exact_remaining_evidence':[] if d['debt_id']==support_debt['debt_id'] else s['exact_remaining_evidence'],
                        'negative_failed_abandoned_superseded_findings_preserved':True})
    study_groups.append({'study_dependency_group_id':'STUDYDEP-'+d['debt_id'][5:],
                         'root_requirement_id':root_for[d['debt_id']],
                         'original_debt_id':d['debt_id'],'original_request_key':d['deduplication_key'],
                         'original_deduplication_boundary':d.get('deduplication_boundary'),
                         'source_specific_original_obligations':d.get('dependent_source_requirements',[]),
                         'dependent_original_queue_ids':d['original_queue_ids'],'members':members,
                         'shared_companion_root_ids':sorted(set(root_for[z] for a in d.get('dependent_source_requirements',[]) for z in a.get('shared_missing_companion_debt_ids',[]))),
                         'shared_independent_run_identity_established':False,
                         'shared_missing_request_is_shared_artifact_proof':False,
                         'grouping_boundary':'Coordinated original-artifact retrieval only. Multiple commands/attempts/datasets remain distinct unless original identity proves equality.',
                         'canonical_study_ledger_action':'NONE; original 26-study ledger and all 38 candidates preserved',
                         'membership_confirmation_added':False,'replication_confirmation_added':False})
assert len(study_groups)==52 and sum(len(g['members']) for g in study_groups)==76
save('study_identity_dependency_groups.jsonl',study_groups)
external=[]
for r in roots:
    if r['resolution'].startswith('RESOLVED_'):
        continue
    is_bound=r['classification']=='PRESERVED_BOUNDED_LIMITATION'
    dependent=[result_index[z] for z in r['dependent_original_debt_ids'] if z in result_index]
    needed=is_bound or r['classification']=='SHARED_CENSUS_PREREQUISITE' or any(x['external_recovery_needed'] or x['historical_damage_bounded'] or x['status']=='UNRESOLVED' for x in dependent)
    if needed:
        external.append({'root_requirement_id':r['root_requirement_id'],
                         'dependent_original_debt_ids':r['dependent_original_debt_ids'],
                         'dependent_original_queue_ids':r['dependent_original_queue_ids'],
                         'dependent_original_source_ids':r['dependent_original_source_ids'],
                         'exact_missing_original_artifact_or_record':r['exact_missing_evidence'],
                         'exact_original_source_identities':r['exact_original_source_identities'],
                         'already_accessible_evidence':r.get('already_accessible_saved_evidence',[]),
                         'local_probe':r.get('local_literal_probe'),
                         'external_requirement_confirmed':any(x['external_recovery_needed'] for x in dependent) or r['classification']=='SHARED_CENSUS_PREREQUISITE' or (is_bound and r['blocks_full_census_closure']),
                         'external_or_further_local_interpretation_alternative':any(x['status']=='UNRESOLVED' for x in dependent),
                         'historical_damage_intended_values_required':any(x['historical_damage_bounded'] for x in dependent),
                         'global_absence_or_destruction_proven':False,
                         'blocks_full_census_closure':r['blocks_full_census_closure'],
                         'can_retain_as_nonblocking_limitation':is_bound and not r['blocks_full_census_closure'],
                         'impossible_to_infer':'Original command/run identity, chronology, seed/cohort/checkpoint equality, or intended damaged characters without original evidence.',
                         'exact_next_action':r['exact_next_action']})
save('remaining_external_evidence.jsonl',external)
save('batch_03_checkpoint.json', {'processed_debt_ids':sorted(result_index), 'unprocessed_ids':[],
                                'study_queue_ids':sorted(study_index), 'root_group_count':len(groups),
                                'original_obligations_preserved':True})

# Final deterministic integrity verification; no scientific tests execute.
unchanged={p:sha(ROOT/p)==v['sha256'] for p,v in baseline.items()}
assert all(unchanged.values())
assert git('rev-parse','HEAD')==head
git('diff','--check')
assert not git('diff','--stat') and not git('diff','--cached','--stat')
final_status=git('status','--short','--branch')
assert final_status==initial_status
counts=dict(collections.Counter(x['status'] for x in results))
resolved_ids=sorted(x['debt_id'] for x in results if not x['census_blocking_after_recovery'])
open_ids=sorted(x['debt_id'] for x in results if x['census_blocking_after_recovery'])
study_open=[x['original_queue_id'] for x in studies if x['source_record_id']!='RS044']
census_roots=[x for x in roots if x['classification']!='PRESERVED_BOUNDED_LIMITATION']
validation={'decision':'PARTIAL_CENSUS_RECOVERY; FULL_CLOSURE_NOT_ESTABLISHED',
            'authority':'Current user-directed Full-History Census Evidence Closure + AGENTS.md',
            'phase':'CENSUS_RECOVERY_ONLY', 'git_head':head,'git_status_before':initial_status,'git_status_after':final_status,
            'original_blocker_count':211, 'accounted_unique_blocker_count':len(results),
            'exact_original_debt_ids':sorted(result_index), 'resolved_debt_ids':resolved_ids,'remaining_census_blocker_ids':open_ids,
            'status_counts':counts,'distinct_census_root_requirements_including_shared_prerequisite':len(census_roots),
            'distinct_census_closure_groups':len(groups),
            'remaining_census_root_requirements':sum(not x['resolution'].startswith('RESOLVED_') for x in census_roots),
            'root_requirement_file_rows_including_three_separate_bounded_limits':len(roots),
            'exact_endpoint_overlap_duplicate_blockers':len(duplicate_basis),
            'duplicate_dependency_debt_ids':sorted(duplicate_basis),
            'resolved_by_existing_evidence':counts.get('RESOLVED_BY_EXISTING_EVIDENCE',0),
            'resolved_by_new_local_evidence':counts.get('RESOLVED_BY_NEW_LOCAL_EVIDENCE',0),
            'missing_external_status_count':counts.get('EXTERNAL_EVIDENCE_REQUIRED',0),
            'external_needed_including_duplicate_dependents':sum(x['external_recovery_needed'] for x in results),
            'globally_historically_unrecoverable_blockers_proven':0,
            'damaged_originals_bounded_but_still_blocking':4,
            'bounded_historical_limitations_preserved':bounded,
            'bounded_strict_blocker_count':sum(x['blocks_strict_census_completeness'] for x in bounded),
            'original_strict_blocking_records':212,'remaining_strict_blocking_records':len(open_ids)+1,
            'deferred_scientific_validation_preserved':deferred,
            'deferred_original_records':load(SC/'deferred_scientific_validation.jsonl'),
            'study_original_dependency_count':76,'study_coordinated_request_groups':52,
            'study_documentary_dependency_resolved':1,'study_identity_dependencies_remaining':75,
            'remaining_study_request_groups':51,'remaining_study_queue_ids':sorted(study_open),
            'shared_original_independent_execution_count_established':0,
            'confirmed_memberships_added':0,'confirmed_replications_added':0,
            'canonical_ledger_modified':False, 'source_universe_expanded':False,
            'all_original_source_identities_and_payloads_preserved':True,
            'all_dependencies_mapped_and_unresolved_obligations_preserved':True,
            'original_study_ledger_and_provisional_candidates_preserved':True,
            'input_preservation_checks':unchanged, 'all_existing_audit_input_files_unchanged':all(unchanged.values()),
            'existing_audit_input_file_count':len(baseline),'existing_audit_input_bytes':sum(x['bytes'] for x in baseline.values()),
            'prior_declared_input_fingerprints_checked':len(declared['files']),
            'consumed_input_fingerprints':READS,
            'commands_executed_by_script':COMMANDS,
            'read_only_worktree_literal_probe_count':sum(len(x['exact_path_worktree_probes']) for x in local_probes.values()),
            'literal_probe_limits':'Exact known literal paths only; no exhaustive object/disk search. Existing source manifests reused. Local paths with no producer binding cannot authenticate original execution.',
            'no_completed_git_pdf_tensor_source_parsing_repeated':True,
            'targeted_new_original_source_inspection':'HV000971 exact observer companion at two already-named commits; one 2-addition/1-deletion loop-order diff and original coordinate-validator excerpts. No source parsing, compilation or execution.',
            'training_evaluation_gpu_scientific_red_team':'NOT_RUN', 'historical_source_or_bytecode_executed':False,
            'tests':'NOT_RUN; no implementation change or scientific execution authorized',
            'git_mutation_commit_push':False,'git_diff_check':'PASSED; tracked changes empty',
            'independent_final_census_completeness_audit_justified':False,
            'independent_partial_packet_integrity_review_justified':True,
            'unprocessed_original_debt_ids':[],
            'semantic_completion_not_claimed':'Every blocker triaged and mapped; open whole-function/transition interpretations remain open, not silently completed.',
            'warnings':['Python launcher stale installation-location warning; first schema probe used cp949 and failed decoding, corrected with explicit UTF-8. A later console probe failed cp949 output encoding, corrected with -X utf8. No artifacts were changed by either failed read.']}
save('closure_validation.json',validation)
print(json.dumps({k:validation[k] for k in ['original_blocker_count','status_counts','distinct_census_root_requirements_including_shared_prerequisite','distinct_census_closure_groups','remaining_census_root_requirements','study_identity_dependencies_remaining','external_needed_including_duplicate_dependents','existing_audit_input_file_count']},sort_keys=True))
