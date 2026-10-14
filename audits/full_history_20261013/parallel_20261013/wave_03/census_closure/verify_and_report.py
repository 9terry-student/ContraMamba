"""Independent saved-output readback, without importing the recovery producer."""
import collections
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
SC=OUT.parent/'study_consolidation'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):
    text=p.read_text(encoding='utf-8')
    if p.suffix=='.jsonl':
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return json.loads(text)

def canonical(x):
    return json.dumps(x,ensure_ascii=True,sort_keys=True,separators=(',',':'))

def index(rows,k):
    d={x[k]:x for x in rows}
    assert len(d)==len(rows),k
    return d

original=read(SC/'unresolved_census_debt.jsonl')
blockers=index([x for x in original if x['classification']=='CENSUS_BLOCKING'],'debt_id')
assert len(blockers)==211
frozen=index(read(OUT/'frozen_211_target_manifest.jsonl'),'debt_id')
results=index(read(OUT/'source_recovery_results.jsonl'),'debt_id')
roots=index(read(OUT/'root_evidence_requirements.jsonl'),'root_requirement_id')
groups=read(OUT/'study_identity_dependency_groups.jsonl')
external=index(read(OUT/'remaining_external_evidence.jsonl'),'root_requirement_id')
v=read(OUT/'closure_validation.json')
assert set(blockers)==set(frozen)==set(results)
assert len(roots)==212
for k,d in blockers.items():
    assert frozen[k]['original_debt_record']==d
    assert frozen[k]['exact_source_identity_preserved']==d['original_source_identities']
    assert results[k]['original_source_identities']==d['original_source_identities']
    assert results[k]['original_queue_ids']==d['original_queue_ids']
    for rid in results[k]['root_requirement_ids']:
        assert rid in roots and k in roots[rid]['dependent_original_debt_ids']
    if results[k]['census_blocking_after_recovery']:
        assert results[k]['unresolved_requirement']==d['exact_missing_evidence']
    else:
        assert results[k]['status'].startswith('RESOLVED_')
    assert not results[k]['new_membership_or_replication_promoted']
    assert not results[k]['scientific_validity_certified']
    for dep in frozen[k]['dependencies_on_original_debt_ids']:
        assert dep in blockers
        mapped=results[dep]['root_requirement_ids'][0]
        assert mapped in roots[results[k]['root_requirement_ids'][0]]['underlying_dependency_root_ids']
    for ref in frozen[k]['prior_investigations']:
        p=ROOT/ref['file']
        assert sha(p)==ref['file_sha256']

primary=[x for x in roots.values() if x['classification']=='CENSUS_RECOVERY_REQUIREMENT']
primary_ids=[d for x in primary for d in x['dependent_original_debt_ids']]
assert len(primary)==208 and collections.Counter(primary_ids)==collections.Counter(blockers.keys())
assert sum(len(x['dependent_original_debt_ids'])-1 for x in primary)==3
for r in primary:
    required={q for d in r['dependent_original_debt_ids'] for q in blockers[d]['original_queue_ids']}
    assert set(r['dependent_original_queue_ids'])==required
    # Exact original obligations must remain present even when they share a root.
    ix=index(r['exact_missing_evidence'],'debt_id')
    for d in r['dependent_original_debt_ids']:
        assert ix[d]['exact_missing_evidence']==blockers[d]['exact_missing_evidence']

study=read(SC/'study_identity_76.jsonl')
study_index=index(study,'original_queue_id')
member_ids=[x['original_queue_id'] for g in groups for x in g['members']]
assert len(groups)==52 and len(member_ids)==76 and set(member_ids)==set(study_index)
for g in groups:
    d=blockers[g['original_debt_id']]
    assert g['source_specific_original_obligations']==d.get('dependent_source_requirements',[])
    assert not g['shared_independent_run_identity_established']
    for m in g['members']:
        s=study_index[m['original_queue_id']]
        assert m['documentary_membership']==s['provisional_study_references']
        assert m['original_source_identity']==s['source_identity']
        if s['source_record_id']!='RS044':
            assert m['exact_remaining_evidence']==s['exact_remaining_evidence']
        assert m['replication_identity']=='NOT_ESTABLISHED'
assert v['bounded_historical_limitations_preserved']==[x for x in original if x['classification']=='BOUNDED_HISTORICAL_LIMITATION']
assert v['deferred_scientific_validation_preserved']==[x for x in original if x['classification']=='SCIENTIFIC_VALIDATION_DEFERRED']
assert len(v['deferred_original_records'])==8
assert all(not x['blocks_historical_census_closure'] for x in v['deferred_original_records'])

# Independently inspect the evidence supporting the two narrowly closed gaps.
closed=[x for x in results.values() if not x['census_blocking_after_recovery']]
assert len(closed)==2
rs=next(x for x in closed if x['stable_original_source_ids']==['RS044'])
assert rs['status']=='RESOLVED_BY_EXISTING_EVIDENCE'
edges=rs['existing_evidence_closure']['source_edges']
assert len(edges)==2
assert {x['destination_identity']['blob'] for x in edges}=={'04c4e8a79c443fe9dfdf6a7561ba29ef1e596a4a','20744d8673102312d0a8a8581784540706da885c'}
assert all(x['destination_identity']['commit']=='ea03cdaa5394de03389d31aa1089b51e10340a57' for x in edges)
assert any('FREEZE CANDIDATE' in z['excerpt'] for x in edges for z in x['selected_status_excerpts'])
o0c=next(x for x in closed if x['stable_original_source_ids']==['HV000971'])
assert o0c['status']=='RESOLVED_BY_NEW_LOCAL_EVIDENCE'
ev=o0c['new_local_evidence']
assert sha(OUT/'batch_02b_o0c_companion_checkpoint.json')
assert ev==read(OUT/'batch_02b_o0c_companion_checkpoint.json')
diff=ev['exact_diff']
assert hashlib.sha256(diff.encode()).hexdigest()==ev['diff_sha256']
assert len([x for x in diff.splitlines() if x.startswith('+') and not x.startswith('+++')])==2
assert len([x for x in diff.splitlines() if x.startswith('-') and not x.startswith('---')])==1
assert '\n+      for anchor in ANCHOR_ORDER:' in diff and '\n+       for layer in layers:' in diff
assert all(any('for a in ANCHOR_ORDER for l in layers' in z['text'] for z in x['line_indexed_original_excerpts']) for x in ev['observer_endpoint_authentication'])
assert {x['source_identity']['blob'] for x in ev['observer_endpoint_authentication']}=={'7ac5d0d5e93742c8d76cd3e467790a1866dbade9','49faa5eeb659ce7fbec7004e0933deaa6fb08c3b'}

probes=read(OUT/'batch_02_local_probe_checkpoint.json')['literal_probes']
assert len(probes)==16
assert all(len(x['exact_path_worktree_probes'])==28 for x in probes.values())
assert all(not z['present'] and z['worktree_accessible'] for x in probes.values() for z in x['exact_path_worktree_probes'])
assert all(not x['exact_saved_inventory_matches'] and not x['exact_saved_historical_version_matches'] for x in probes.values())
baseline=read(OUT/'input_preservation_checkpoint.json')
assert all(sha(ROOT/p)==x['sha256'] for p,x in baseline.items())
assert len(baseline)==669
assert v['status_counts']==dict(collections.Counter(x['status'] for x in results.values()))
assert len(v['remaining_census_blocker_ids'])==209
assert v['remaining_strict_blocking_records']==210
assert v['remaining_census_root_requirements']==207
counts=v['status_counts']
by_kind=collections.Counter(blockers[x]['debt_kinds'][0] for x in blockers)

report=[]
def add(x=''):
    report.append(x)

add('# ContraMamba — Full-history census evidence closure')
add()
add('**PARTIAL: 211/211 original census blockers reconciled; two evidence requirements closed; 209 blockers remain. Full historical census closure is not established.**')
add()
add('Authority: the current user-directed full-history census and repository AGENTS.md. Phase: CENSUS RECOVERY ONLY. Historical specifications and reports are documentary evidence, not current execution authority. Only `census_closure/` was written. Training, evaluation, GPU execution, scientific numerical validation, Red-Team analysis, Git mutation, commit and push were not performed.')
add()
add('The debt ledger contains 222 actual records: 211 classified CENSUS_BLOCKING, three bounded historical limitations, and eight deferred scientific-validation tasks. One bounded snapshot-chronology limitation retains its original strict-blocking flag: the original strict count was 212; after the two narrow closures it is 210. The 211 count denotes debt owners, not 211 independent studies or missing artifacts.')
add()
add('| Original blocker category | Records |')
add('|---|---:|')
for kind,n in sorted(by_kind.items()):
    add(f'| {kind} | {n} |')
add('| Total | 211 |')
add()
add('| Evidence-backed disposition | Records |')
add('|---|---:|')
for st,n in sorted(counts.items()):
    add(f'| {st} | {n} |')
add('| Total | 211 |')
add()
add('There are **209 distinct census evidence-requirement groups in the frozen reconciliation**, comprising 208 source-specific closure groups and one shared original pytest-producer/options prerequisite. Two groups close, leaving **207 open census requirements**. The root file separately preserves three legacy bounded limitations, so it contains 212 rows. These are retrieval/interpretation requirements, not a claim that 209 independent external artifacts or executions exist. Many requirements need several original identity dimensions; unspecified historical artifact names remain unknown rather than invented.')
add()
add('The only newly consolidated blocker overlaps are three exact compiled endpoint identities shared by graph-correspondence and historical-transition owners. Their original transition-semantic obligations are retained in the same root, and all three receive DUPLICATE_DEPENDENCY_WITH_OPEN_ROOT. The four graph owners still need four endpoint-specific source/build correspondences; a shared pytest version does not prove a common run. Generic study boilerplate, related stage names, similar filenames, and 99 versions of the same trainer do not justify merging their historical semantics.')
add()
add('The 52 original coordinated study requests account for all 76 study source owners. Their 24 fewer request bundles reflect coordination already present in the input, not 24 closures or established duplicate experiments. Ten multi-source bundles preserve each source-specific obligation. No shared independent execution was newly established. One supporting documentary dependency closes, leaving **75 source identity dependencies in 51 coordinated study requests**. The original provisional study ledger, all candidates, and every negative/failed/abandoned/superseded finding remain unchanged and hash-bound.')
add()
add('Two narrow closures are supported:')
add()
add(f'- **RS044 — {rs["debt_id"]} — existing evidence.** At `ea03cdaa5394de03389d31aa1089b51e10340a57`, manuscript spine blob `210b902247f8cbf72133ffcd80b1340c2c82e318` explicitly selects claim-disposition blob `04c4e8a79c443fe9dfdf6a7561ba29ef1e596a4a` at lines 8–9 and novelty-positioning blob `20744d8673102312d0a8a8581784540706da885c` at lines 10–11. Both exact same-tree source versions were already authenticated in the study record. The claim set is marked frozen; the novelty document remains a freeze candidate/ready-for-freeze. This closes the exact documentary selection links only. Publication-support status is retained; final novelty adoption, numerical validity and execution are not inferred.')
add(f'- **HV000971 — {o0c["debt_id"]} — new local evidence.** The saved runner-pin transition names observer commits `e03b38fa0681fd04d8633a7de184559bd0473133` and `60a53b6f5d1db8d7cbecded5b94d5231adcfc520`. Exact observer path `scripts/observe_longterm_o0c_selective_ssm_native_state_dynamics.py` resolves to blobs `7ac5d0d5e93742c8d76cd3e467790a1866dbade9` and `49faa5eeb659ce7fbec7004e0933deaa6fb08c3b`. Read-only original-blob hashing and one 2-addition/1-deletion diff establish the missing companion relationship: layer/anchor traversal becomes anchor/layer traversal, matching the unchanged pair/comparison/anchor/layer validator. Metric expressions remain identical in this diff. This closes mechanism-level source interpretation of the pin change, not evidence that an original run succeeded.')
add()
add('Local recovery feasibility was checked through reused authenticated source records, prior fingerprints, the existing 4474-source inventory and 1565-version manifest, then exact-path probes for 16 missing literals across 28 accessible registered worktrees (448 probes). All 16 are absent in those bounded inventories and paths. This does not establish absence from every historical Git object, every local disk, Kaggle, or external archives. No broad history scan, PDF inspection, tensor scan or source parsing was repeated. Known LFS recovery from Expansion 04A-R is reused; none of these missing literals supplies an original pointer/OID that could identify a payload in content-addressed LFS storage.')
add()
add('**75 blockers receive EXTERNAL_EVIDENCE_REQUIRED; 78 records depend on that evidence when the three duplicate dependents are included.** This means the preserved packet lacks the required original identity/binding, not that exhaustive global absence has been proven. Another 126 records have source bytes present with unfinished semantic/documentary interpretation. Five requirements retain UNRESOLVED because their saved evidence leaves a local companion interpretation versus missing-original recovery alternative. All five exact owners and next actions remain in the JSONL outputs; none is silently deferred to scientific validation.')
add()
add('No original census blocker is proven globally destroyed or historically unrecoverable. Four damaged originals are unrecoverable in their intended-value dimensions **from the recorded blobs alone** and remain blocking: HV000202, HV001023, HV001096 and HV001097. An authenticated original pre-damage source/correction record is needed; neighboring versions cannot supply guessed Unicode, statement boundaries, C0 notation, commit or hash substitutions. The three separately bounded legacy limitations remain open: snapshot capture/base chronology (strictly blocking), Codex reference actor/cause/time attribution (nonblocking), and selected ancestry versus study-wide adoption scope (nonblocking). They are limitations, not recovered history.')
add()
add('Exact original literal recovery requirements are listed below. All refer to the original locator at `ff806bd5bd8efc62fcf6420cf8b097a63ed70571`; no missing blob identity is fabricated. A recovery must supply that original artifact and producer/version binding, or an authenticated historical correction/rename record establishing the intended replacement.')
add()
add('| Exact missing original literal | Dependent source owners | Original queue IDs |')
add('|---|---|---|')
for k,x in sorted(probes.items(),key=lambda kv:kv[1]['literal']['path']):
    d=blockers[k]
    add(f'| `{x["literal"]["path"]}` | {", ".join(d["stable_original_source_ids"])} | {", ".join(d["original_queue_ids"])} |')
add()
add('Beyond these literals, census completion requires endpoint-bound original execution/attempt manifests and commands, code/input hashes, model/tokenizer/checkpoint identities, dataset versions, row cohorts, split/seed roles and correction/retry/adoption lineage wherever the original owner requires them. Examples that remain distinct are Stage112–116 original material after runtime loss (RS069); Stage31B dry-run versus appended result attribution (RS124); the original epoch10 checkpoint and epoch5-start/threshold history (HV000049); Stage196 original P4 and P5 artifacts, whose downstream attestations explicitly do not reconstruct their full original tables/prose; and fused-streaming original run/cohort/parity-report provenance (HV000699). Parity numerical certification is outside this task. Design-only, data-construction and manuscript sources are not forced into model execution; their exact documentary/data/identity requirements are preserved instead.')
add()
add('The exact smallest defensible requirements and dependent IDs are machine-readable in `root_evidence_requirements.jsonl`, with unresolved external or external-versus-local alternatives in `remaining_external_evidence.jsonl`. Each of the 76 source owners is separately represented in `study_identity_dependency_groups.jsonl`, separating documentary membership, execution identity, replication identity and scientific validity. Documentary candidates are never upgraded into independent replications.')
add()
add('A final independent **census-completeness certification is not justified**: 209 original blockers plus the strictly blocking bounded chronology limitation remain. Independent review of this partial packet’s accounting and two limited closures is justified. The narrowest next recovery action with shared reach is to obtain the preserved original pytest 9.1.1 rewrite producer/options and source-hash-bound build manifest for the exact endpoints listed in `ROOT-SHARED-PYTEST-9.1.1-ORIGINAL-PRODUCER-OPTIONS`. A wheel/version alone would improve documentation but would not close the four correspondences or their dependent transitions. The narrowest entirely local alternative is source-specific companion interpretation for HV001270 (encoder masks/gradient ownership) using its saved exact endpoint evidence; no execution is needed.')
add()
add('Validation independently read back the saved outputs without importing the producer. It checked exact 211-ID bijection and original payload equality, identity preservation, every root mapping, residual obligations on shared roots, all 76 original study IDs/provisional references, the two closure witnesses, separate three/eight preservation and all 669 pre-existing audit-file SHA256 values (648,332,346 bytes). All passed. No original audit input changed. No scientific test suite ran. `git diff --check` passed; staged and unstaged tracked diffs are empty. The pre-existing untracked `audits/` status remains. These tracked checks do not validate untracked reports; independent JSONL readback and hashes do.')
add()
add('Commands actually executed: read-only `git status --short`, branch/HEAD inspection, `rg --files`/targeted report search, PowerShell report reads, inline Python UTF-8 JSON/schema/count/fingerprint checks, `git worktree list --porcelain`, Git common-directory lookup, exact observer `rev-parse`/`show`/`cat-file`/path-bounded `diff` checks, the deterministic producer and this independent verifier, and `git diff --check`/`--stat`/`--cached --stat`. The JSON validation preserves exact producer command arrays. Python emitted a stale installation-location warning; two initial read probes hit Windows cp949 decoding/console encoding errors and were corrected with UTF-8. Neither failed read altered artifacts. No historical program, source code, bytecode or run command was executed.')
add()
add('The first independent-verifier attempt stopped at an audit-helper counting assertion because `Counter` received a dictionary instead of its keys. The helper was corrected without changing evidence or dispositions; the complete independent readback then passed.')
add()
add(f'HEAD remains `{v["git_head"]}` (detached). No commit was created. All 211 blockers and all 76 study owners were processed for evidence-debt reconciliation; the precise unprocessed-ID list is empty. Outstanding semantic work is explicitly open, not claimed completed. Checkpoints preserve the frozen batch, 448 literal probes, exact O0C companion evidence and the complete processed-ID set. No further historical reconstruction or scientific phase begins automatically.')
add()
add('Output files: the seven requested artifacts, two deterministic audit scripts, and five integrity/batch checkpoints, all under the authorized output directory. Existing audit outputs are preserved byte-for-byte.')
report_text='\n'.join(report)+'\n'
(OUT/'closure_report.md').write_text(report_text,encoding='utf-8',newline='\n')
v['independent_saved_output_readback']={
    'status':'PASSED','verifier':'verify_and_report.py',
    'exact_original_payload_equality':True,'exact_211_bijection':True,
    'all_root_edges_and_residual_obligations_checked':True,
    'original_76_study_ids_and_provisional_references_preserved':True,
    'two_narrow_closure_witnesses_checked':True,
    'three_bounded_and_eight_deferred_records_separately_preserved':True,
    'existing_audit_fingerprints_rechecked':669,
    'initial_verifier_failure':'Audit-helper Counter(dict) assertion; corrected to Counter(dict.keys()), evidence and dispositions unchanged; final complete readback passed.',
    'production_scientific_tests_executed':False}
outputs=['frozen_211_target_manifest.jsonl','root_evidence_requirements.jsonl','source_recovery_results.jsonl','study_identity_dependency_groups.jsonl','remaining_external_evidence.jsonl','closure_report.md']
v['final_output_fingerprints_excluding_self']={n:{'sha256':sha(OUT/n),'bytes':(OUT/n).stat().st_size} for n in outputs}
(OUT/'closure_validation.json').write_text(json.dumps(v,ensure_ascii=True,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Independent readback PASSED: 211 exact blockers, 76 study owners, two limited closures, 669 unchanged prior inputs. Full closure NOT established.')
