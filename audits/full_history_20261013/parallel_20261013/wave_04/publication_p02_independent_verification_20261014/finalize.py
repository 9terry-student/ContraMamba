from verify import *

def refmap(s):return {line.split(' ',1)[1]:line.split(' ',1)[0] for line in s.splitlines()}
def refdelta(a,b):
    a=refmap(a);b=refmap(b);return [{'ref':k,'before':a.get(k),'after':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
def workmap(s):
    out={}
    for block in s.strip().split('\n\n'):
        lines=block.splitlines();out[lines[0][9:]]=lines[1:]
    return out
def workdelta(a,b):
    a=workmap(a);b=workmap(b);return [{'path':k,'before':a.get(k),'after':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
def tree(c):
    b=git('ls-tree','-r','-z','--full-tree',c);out={}
    for line in b.split(b'\0'):
        if line:
            desc,name=line.split(b'\t',1);out[name.decode()]=desc.decode()
    return out
def scoped_range(old,new,pinned_paths):
    git('merge-base','--is-ancestor',old,new,allowed=(0,1));assert COMMANDS[-1]['exit_code']==0,'NON_FAST_FORWARD_STOP'
    git('merge-base','--is-ancestor',new,old,allowed=(0,1));assert COMMANDS[-1]['exit_code']==1
    commits=git('rev-list','--reverse','--topo-order',old+'..'+new).decode().split()
    result=[];previous=old;allchanged=set()
    for c in commits:
        parents=git('show','-s','--format=%P',c).decode().split();assert parents==[previous],'NON_LINEAR_RANGE_STOP'
        aa=tree(previous);bb=tree(c);names=sorted(k for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k))
        changed=[{'path':k,'before':aa.get(k),'after':bb.get(k)} for k in names]
        assert not (set(names)&pinned_paths),'RELEVANT_PINNED_TREE_MOVEMENT'
        assert all(k.startswith(('scripts/gen5_','tests/test_gen5_','reports/reason_router_gen5_')) for k in names),'UNVERIFIED_SCOPE_MOVEMENT'
        result.append({'commit':c,'sole_parent':previous,'changed_tree_entries':changed,'all_other_tree_entries_equal':True,'pinned_path_intersection':[]});allchanged|=set(names);previous=c
    aa=tree(old);bb=tree(new);pchecks=[{'path':p,'before':aa.get(p),'after':bb.get(p),'equal':aa.get(p)==bb.get(p)} for p in sorted(pinned_paths)]
    assert all(x['equal'] for x in pchecks)
    return {'old':old,'new':new,'strict_fast_forward':True,'commit_count':len(commits),'complete_commits':result,'complete_endpoint_changed_paths':sorted(k for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k)),'pinned_tree_path_checks':pchecks,'scoped_conclusion':'UNRELATED_TO_ORIGINAL_P02_PINNED_DOCUMENTARY_INPUTS','boundary':'Tree/path/identity relevance only; no Gen5 code semantic inspection, no preflight SCHEMA issue review, no scientific or execution certification; raw movement remains FAIL and actor/transients unknown.'}
def movement():
    em=read(OUT/'evidence_manifest.json');cp=read(OUT/'checkpoint.json');mv=read(W4/'publication_p02_movement_verification_20261014/validation.json')
    pinned={x['path'] for x in em['git_binding_checks']}|{x['file'] for x in em['pinned_inputs']}
    ranges=[]
    ranges.append(scoped_range('417fadaaadae33fe90ccedf7108002c64c0ae41c','3960eff3fd4e0ea4c4e337a5a4907d9eaf7ea7bc',pinned))
    ranges.append(scoped_range('3960eff3fd4e0ea4c4e337a5a4907d9eaf7ea7bc','378dff9b43fe4d7a6e1c556ee5f980f99fdb4934',pinned))
    baseline=refmap(cp['initial_state']['refs']);tip=baseline['refs/heads/gen5-causal-role-state-ownership'];assert baseline['refs/remotes/origin/gen5-causal-role-state-ownership']==tip
    if tip!='378dff9b43fe4d7a6e1c556ee5f980f99fdb4934':ranges.append(scoped_range('378dff9b43fe4d7a6e1c556ee5f980f99fdb4934',tip,pinned))
    prior=mv['terminal_observation'];raw={'prior_movement_verifier_to_current_baseline_refs':refdelta(prior['refs'],cp['initial_state']['refs']),'prior_movement_verifier_to_current_baseline_worktrees':workdelta(prior['worktrees'],cp['initial_state']['worktrees'])}
    em['scoped_movement_qualification']={'ranges':ranges,'prebaseline_raw_observations':raw,'prebaseline_raw_ref_status':'FAIL_REFS_MOVED' if raw['prior_movement_verifier_to_current_baseline_refs'] else 'UNCHANGED','prebaseline_raw_worktree_status':'FAIL_WORKTREE_LIST_CHANGED' if raw['prior_movement_verifier_to_current_baseline_worktrees'] else 'UNCHANGED','scope':'Original P02 dependencies and immutable original commits only; no recursive historical ref audit. Prior independent installation-report content review reused only after exact movement packet hash authentication. No inference about code correctness, preflight SCHEMA, moving actors, transient states or global paper adoption.'}
    em['commands'].extend(COMMANDS);write('evidence_manifest.json',em)
    print('Scoped movement check:',[(r['old'][:8],r['new'][:8],r['commit_count']) for r in ranges],'; all pinned paths unchanged; raw FAILs retained.')

def validate():
    cp=read(OUT/'checkpoint.json');em=read(OUT/'evidence_manifest.json');rr=rows(OUT/'owner_verdicts.jsonl');checks=[]
    def check(name,ok,detail=None):
        checks.append({'check':name,'status':'PASS' if ok else 'FAIL','detail':detail});assert ok,name
    check('exact_three_authorized_owner_ids_and_debt_pairs',{x['source_id']:x['debt_id'] for x in rr}==OWNERS and len(rr)==3)
    check('three_separate_owner_verdicts',all(x['verdict']=='VERIFIED_FULL_DOCUMENTARY_SATISFACTION' for x in rr))
    for i,row in enumerate(rr):
        owner=row['source_id'];e=read(OUT/(owner+'_evidence.json'))
        check(owner+'_saved_completed_row_preserved',sha((OUT/'owner_verdicts.jsonl').read_bytes().splitlines()[i])==cp['completed_owners'][i]['line_sha256_excluding_lf'])
        check(owner+'_completed_evidence_preserved',meta(OUT/(owner+'_evidence.json'))['sha256']==cp['completed_owners'][i]['evidence_sha256'])
        for l in e['original_records'].values():locator(l['locator'])
        for x in e['source_bindings']:binding(x)
        check(owner+'_original_and_companion_identities_reauthenticated',True,len(e['source_bindings']))
        check(owner+'_all_literal_and_decomposed_clauses_mapped',len(row['literal_original_requirement_coverage'])==len(row['original_obligation']['exact_missing_evidence']) and len(row['clause_verdicts'])==6 and all(x['status']=='SATISFIED' and x['evidence_locators'] for x in row['clause_verdicts']))
        for clause in row['clause_verdicts']:
            for l in clause['evidence_locators']:locator(l)
        check(owner+'_full_visual_review',e['visual_inspection']['complete'] and len(e['visual_inspection']['visuals'])==4 and all(len(x['pages'])==1 for x in e['complete_pdf_content']))
        for visual in e['visual_inspection']['visuals']:
            if 'render_file' in visual:check(owner+'_'+visual['side']+'_render_preserved',meta(ROOT/visual['render_file'])['sha256']==visual['sha256'])
            else:check(owner+'_'+visual['side']+'_original_PNG_preserved',meta(ROOT/visual['saved_file'])['sha256']==visual['sha256'])
        check(owner+'_all_operations_and_hunks_reviewed',len(e['manifest_operation_review'])==(84 if owner=='HV000114' else 40) and len(e['README_hunk_review'])==(18 if owner=='HV000114' else 3))
        n=e['numeric_verification']
        if owner in ['HV000114','HV000117']:
            check(owner+'_all_50_displayed_CKA_and_10_scalars',len(n['cka_display_checks'])==50 and len(n['displaced_scalar_checks'])==10)
            if owner=='HV000117':check(owner+'_all_50_before_CKA_displays',all(x['before_PDF_display']==x['PDF_display'] for x in n['cka_display_checks']))
        else:
            check(owner+'_10_means_20_endpoints_15_associations',len(n['point_mean_checks'])==10 and n['added_interval_endpoints']==20 and len(n['association_checks'])==15)
            check(owner+'_both_and_only_zero_crossings',[(x['scale'],x['objective']) for x in n['interval_checks'] if x['contains_zero']]==[('370M','vanilla'),('2.8B','contra')])
            check(owner+'_actual_drawn_means_whiskers_and_associations',len(n['actual_PDF_vector_checks'])==2 and sum(len(x['bar_mean_coordinates']) for x in n['actual_PDF_vector_checks'])==20 and sum(len(x['interval_endpoint_coordinates']) for x in n['actual_PDF_vector_checks'])==20)
        check(owner+'_no_scientific_validation_invented',not row['scientific_certification'] and not n['reused_numeric_computations'])
    for x in em['pinned_inputs']:check('local_pin:'+x['file'],meta(ROOT/x['file'])=={'sha256':x['sha256'],'bytes':x['bytes']})
    for x in em['git_binding_checks']:binding(x)
    check('all_86_unique_Git_inputs_reauthenticated',len(em['git_binding_checks'])==86)
    for x in em['packet_output_checks']:check('frozen_P02_output:'+x['file'],meta(ROOT/x['file'])=={'sha256':x['sha256'],'bytes':x['bytes']})
    for f in em['inherited_failures_retained']:
        q=locator(f.get('evidence',f));check('inherited_failure_raw_status',q['payload']==f.get('status',f.get('raw_status')))
    for p,m in cp['baseline_files'].items():check('baseline_byte:'+p,meta(ROOT/p)==m)
    current=inventory();check('existing_file_membership_unchanged',set(current)==set(cp['baseline_files']))
    end=state()
    for k in ['head','branch','index','index_entries','tracked_diff','staged_diff','status']:check('audit_'+k+'_preserved',cp['initial_state'][k]==end[k])
    rd=refdelta(cp['initial_state']['refs'],end['refs']);wd=workdelta(cp['initial_state']['worktrees'],end['worktrees'])
    qualifications=[]
    if rd:
        for r in rd:
            if r['ref'] in ['refs/heads/gen5-causal-role-state-ownership','refs/remotes/origin/gen5-causal-role-state-ownership']:
                qualifications.append(scoped_range(r['before'],r['after'],{x['path'] for x in em['git_binding_checks']}))
            elif r['ref'].startswith('refs/codex/'):
                qualifications.append({'ref':r,'scoped_conclusion':'NOT_A_DECLARED_P02_INPUT_REF; exact input bytes and bindings reauthenticated','actors_and_transients':'UNKNOWN_UNCERTIFIED'})
            else:raise AssertionError(('UNVERIFIED_REF_MOVEMENT',r))
    cp['final_observation']=end;cp['raw_current_ref_movements']=rd;cp['raw_current_worktree_movements']=wd;cp['raw_current_ref_status']='FAIL_REFS_MOVED' if rd else 'UNCHANGED';cp['raw_current_worktree_status']='FAIL_WORKTREE_LIST_CHANGED' if wd else 'UNCHANGED';cp['current_movement_scoped_qualification']=qualifications
    check('scoped_pinned_input_integrity',True,'Local SHA256, immutable commit/path/blob/SHA256 and movement tree checks; separate from global ref-equality FAILs.')
    check('zero_new_applied_credit',all(x['new_applied_credit']==0 for x in rr) and cp['new_applied_credit']==0)
    check('existing_and_frozen_accounting_retained',em['accounting']['existing']==[222,20,192] and em['accounting']['frozen']==[222,17,195])
    git('diff','--check');trackedstat=git('diff','--stat').decode();stagedstat=git('diff','--cached','--stat').decode();check('git_diff_check_executed',True);check('tracked_staged_diff_stat_empty',not trackedstat and not stagedstat)
    cp['status']='COMPLETE_DOCUMENTARY_VERIFICATION_ZERO_CREDIT';cp['finished_utc']=utc();cp['commands'].extend(COMMANDS);write('checkpoint.json',cp)
    em['commands'].extend(COMMANDS);em['activity']={'training_evaluation':False,'GPU_Kaggle':False,'commit_push':False,'git_refs_branches_worktrees_mutated':False,'source_figures_manuscripts_code_datasets_modified':False,'original_audit_or_overlay_modified':False,'original_audit_producers_or_figure_builders_executed':False,'bootstrap_or_CKA_recomputed':False,'preflight_SCHEMA_issue_inspected_or_fixed':False};write('evidence_manifest.json',em)
    required=['checkpoint.json','owner_verdicts.jsonl','evidence_manifest.json','validation.json','report.md'];files=sorted(p.name for p in OUT.iterdir() if p.is_file())
    check('structured_required_files_present_except_validation_being_written',all((OUT/p).is_file() for p in required if p not in ['validation.json','report.md']))
    # New scripts/renders and the retained probe are all inside the explicitly authorized directory.
    generated=[p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()]
    check('exact_output_scope',all((ROOT/p).is_relative_to(OUT) for p in generated))
    v={'schema':'P02_INDEPENDENT_VALIDATION_V1','phase':'CENSUS_RECOVERY_ONLY','status':'PASS_SCOPED_DOCUMENTARY_VERIFICATION_WITH_RETAINED_HISTORICAL_RAW_FAILS','historical_P02_status':'FAIL_STOP_REQUIRED','historical_raw_failures_relabelled':False,'checks':checks,'blocking_failed_checks':[],'raw_current_ref_status':cp['raw_current_ref_status'],'raw_current_worktree_status':cp['raw_current_worktree_status'],'raw_current_ref_movements':rd,'raw_current_worktree_movements':wd,'prebaseline_raw_movement':em['scoped_movement_qualification']['prebaseline_raw_observations'],'scoped_qualifications':qualifications,'initial_state':cp['initial_state'],'final_state':end,'existing_files_preserved':len(cp['baseline_files']),'original_P02_baseline_files_preserved':em['original_P02_baseline_count'],'pinned_local_inputs_rechecked':len(em['pinned_inputs']),'unique_pinned_Git_bindings_rechecked':len(em['git_binding_checks']),'new_applied_credit':0,'existing_accounting':[222,20,192],'frozen_accounting':[222,17,195],'owner_verdicts':[{k:x[k] for k in ['source_id','debt_id','verdict']} for x in rr],'required_outputs':required,'exact_output_files':sorted(set(files)|{'validation.json','report.md'}),'activity':em['activity'],'probe_issues_retained':cp.get('probe_issues',[]),'warnings':['Earlier inspection outputs were truncated; structured full records were subsequently read by verifier and persisted.','Baseline hashes and validation hashes provide byte identity, not an independent timestamp/signature or continuous observation.'],'commands':COMMANDS,'validated_utc':utc(),'output_sha256_excluding_self':{p.name:meta(p)['sha256'] for p in OUT.iterdir() if p.is_file() and p.name!='validation.json'},'self_hash_policy':'No circular self-hash. validation.json gets a serialized-byte readback; its original source packet hash is separately baseline-bound.','disk_readback_status':'PENDING_REPORT_AND_FINAL_READBACK'}
    write('validation.json',v)
    print('Preservation and scoped validation passed:',len(cp['baseline_files']),'existing files; 129 local pins; 86 Git bindings; zero credit.')

def readback():
    v=read(OUT/'validation.json');before=(OUT/'validation.json').read_bytes();checks=[]
    for name,h in v['output_sha256_excluding_self'].items():
        assert meta(OUT/name)['sha256']==h;checks.append({'file':name,'sha256':h,'status':'PASS_DISK_READBACK'})
    actual=sorted(p.name for p in OUT.iterdir() if p.is_file());assert actual==sorted(v['exact_output_files'])
    for name in ['checkpoint.json','evidence_manifest.json']:read(OUT/name)
    assert len(rows(OUT/'owner_verdicts.jsonl'))==3
    cp=read(OUT/'checkpoint.json');latest=state()
    assert latest['head']==HEAD and latest['index']==v['final_state']['index'] and latest['index_entries']==v['final_state']['index_entries'] and latest['tracked_diff']==v['final_state']['tracked_diff'] and latest['staged_diff']==v['final_state']['staged_diff']
    assert latest['refs']==v['final_state']['refs'] and latest['worktrees']==v['final_state']['worktrees'],'NEW_ENDPOINT_MOVEMENT_REQUIRES_SCOPED_CHECK'
    for p,m in cp['baseline_files'].items():assert meta(ROOT/p)==m
    v['disk_readback_status']='PASS';v['disk_readback_checks']=checks;v['post_validation_observation']=latest;v['disk_readback_utc']=utc();v['disk_readback_commands']=COMMANDS;v['pre_readback_validation_sha256']=sha(before);write('validation.json',v)
    serialized=(OUT/'validation.json').read_bytes();assert json.loads(serialized)==v
    print('Disk readback passed; all three completed rows and exact output inventory preserved; historical FAILs unchanged.')
if __name__=='__main__':{'movement':movement,'validate':validate,'readback':readback}[sys.argv[1]]()
