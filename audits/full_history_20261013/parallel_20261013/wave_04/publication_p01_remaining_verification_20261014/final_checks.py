import sys, json, subprocess, difflib, csv, io
from decimal import Decimal
sys.dont_write_bytecode=True
import verify as v

def supplement():
    data={'created_utc':v.now(),'completed_owner_records_preserved':True,'purpose':'Additional exact-decimal, chronology and narrowly relevant manuscript-reference checks; neither completed owner verdict is discarded or rewritten.','owners':{}}
    for owner in ['HV000107','HV000113']:
        e=v.read(owner+'_evidence.json'); n=v.Path(e['authenticated_saved_transition']['file']).stem
        a,b=[json.loads(v.raw(v.W3+'transitions_0/evidence/'+n+'_'+s+'_companion_figure_data_manifest.json'),parse_float=Decimal) for s in ['before','after']]
        exact=v.operations(a,b); assert [r['path'] for r in exact]==[r['path'] for r in e['json_operations']]
        assert all(a['extracted_values'][k]['value']==b['extracted_values'][k]['value'] for k in a['extracted_values'])
        t=e['original_records']['original_transition']['record']; parents=[]
        for p in e['chronology']['actual_after_parents']:
            blob=v.git('rev-parse',p+':'+t['before']['path']).decode().strip()
            parents.append({'commit':p,'publication_path':t['before']['path'],'blob':blob,'same_as_selected_before':blob==t['before']['blob']})
        o={'exact_decimal_complete_operation_paths_match':True,'exact_decimal_existing_values_unchanged':True,'actual_parent_publication_blobs':parents,'selected_before_is_direct_parent':t['before']['commit'] in e['chronology']['actual_after_parents'],'reference_comparisons':[]}
        for name in ['main.tex','appendix.tex']:
            r=[next(x for x in e['publication_reference_sources'] if x['side']==s and x['path'].endswith(name)) for s in ['before','after']]
            # Only Figure2 body/caption and Figure3 body/caption/geometry/cross-scale clauses.
            ranges=(70,95) if owner=='HV000107' and name=='main.tex' else (105,154) if owner=='HV000107' else (94,114) if name=='main.tex' else (73,123)
            ops=[]
            for tag,i,j,k,l in difflib.SequenceMatcher(None,r[0]['lines'],r[1]['lines']).get_opcodes():
                if tag!='equal' and i<ranges[1] and j>=ranges[0]-1:
                    ops.append({'op':tag,'before_start':i+1,'before_end':j,'before':r[0]['lines'][i:j],'after_start':k+1,'after_end':l,'after':r[1]['lines'][k:l]})
            o['reference_comparisons'].append({'path':r[0]['path'],'before_identity':{k:r[0][k] for k in ['commit','actual_blob','raw_sha256']},'after_identity':{k:r[1][k] for k in ['commit','actual_blob','raw_sha256']},'complete_changes_in_bounded_relevant_source_span':ops,'scope_before_lines':ranges})
        data['owners'][owner]=o
    # Additional manuscript Figure3 wording is not inferred from its PNG-only deletion.
    owner='HV000113'; e=v.read(owner+'_evidence.json'); app=next(x for x in e['publication_reference_sources'] if x['side']=='after' and x['path'].endswith('appendix.tex'))
    prefix='reports/reason_router_gen4_mamba1_iclr_pair_resampling_robustness_v1/'; commit='8423ef9d1325c007ecd31849a668131197e2214d'
    auth=[]
    for name in ['analysis_report.md','geometry_cka_bootstrap.csv','geometry_split_half_summary.csv','provenance_manifest.json']:
        path=prefix+name; ident={'commit':commit,'path':path,'blob':v.git('rev-parse',commit+':'+path).decode().strip()}; auth.append(v.authenticate(ident))
    summary=list(csv.DictReader(io.StringIO(v.git('cat-file','blob',commit+':'+prefix+'geometry_split_half_summary.csv').decode())))
    appendix='\n'.join(app['lines'][338:375]); matches=[]
    for row in summary:
        disp=format(Decimal(row['median']),'.6f'); assert disp.lstrip('0') in appendix
        matches.append({'source_row':summary.index(row)+2,'family':row['family'],'metric':row['metric'],'source_median':row['median'],'appendix_display':disp,'matches':True})
    data['owners'][owner]['additional_manuscript_context']={'interpretation':'Actual main.tex subsection95->112 changes reorganizes to only partially conserved; caption113->130 changes coordinate-insensitive geometric reorganization to only partial geometric conservation. Before106->after123 retains partial-not-invariant interpretation but newly cites joint pair resampling and stronger XG2 than XG4 ordering stability. Actual appendix339-375 adds the descriptive robustness section, whose eight table medians match the pinned source CSV. The rendered Figure3 itself introduces no new CKA endpoint or robustness interval. These are authenticated publication/source claims, not independent statistical/scientific validation.','main_reference_file':e['publication_reference_sources'][0]['path'],'appendix_identity':{k:app[k] for k in ['commit','path','actual_blob','raw_sha256']},'appendix_line_start':339,'appendix_line_end':375,'exact_appendix_lines':app['lines'][338:375],'bounded_additional_source_authentication':auth,'eight_median_display_checks':matches,'scientific_producer_executed':False,'independent_replication':False,'owner_verdict_unchanged':True}
    v.write('supplemental_checks.json',data)
    print(json.dumps({'exact_decimal_checks':'PASS','actual_parent_blobs':'AUTHENTICATED','additional_Figure3_manuscript_context':'AUTHENTICATED','median_source_checks':8,'verdict_records_rewritten':False}))

def prior_refs():
    e=v.read('evidence_manifest.json'); idir=v.W4+'publication_hv000103_closure_integration_20261014/'
    iv=json.loads(v.raw(idir+'validation.json')); prior=iv['final_preservation']; oldlines=prior['all_refs_for_each_ref']
    old={l.split('\t')[0]:l.split('\t')[1] for l in oldlines}; new={k:z[0] for k,z in v.refs_parse(v.read('baseline.json')['start_state']['refs']).items()}
    changes=[]; pinned=set(e['current_ref_movement_qualification']['all_pinned_paths'])
    raw_movements=[{'ref':k,'before':old.get(k),'after':new.get(k),'raw_status':'FAIL_REFS_MOVED'} for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)]
    e['prebaseline_global_ref_comparison']={'source':idir+'validation.json#/final_preservation','all_prior_ref_endpoints_literal':oldlines,'all_initial_ref_endpoints_literal':v.read('baseline.json')['start_state']['refs'],'raw_movements':raw_movements,'raw_status':'FAIL_REFS_MOVED' if raw_movements else 'PASS_REFS_UNCHANGED','qualification':'PENDING'}
    v.write('evidence_manifest.json',e)
    deleted=[k for k in old if k not in new]; added=[k for k in new if k not in old]
    capture=None
    if deleted or added:
        assert len(deleted)==len(added)==1 and all(k.startswith('refs/codex/turn-diffs/captures/') for k in deleted+added),'STOP: unclassified prebaseline ref creation/deletion'
        a,b=old[deleted[0]],new[added[0]]
        assert v.git('cat-file','-t',a).strip()==v.git('cat-file','-t',b).strip()==b'tree'
        def tree(t):
            z={}
            for entry in v.git('ls-tree','-r','-z',t).split(b'\0'):
                if entry:
                    meta,path=entry.split(b'\t',1);z[path.decode()]=meta.decode()
            return z
        ta,tb=tree(a),tree(b);removed=sorted(ta.keys()-tb.keys());changed=sorted(k for k in ta.keys()&tb.keys() if ta[k]!=tb[k]);newpaths=sorted(tb.keys()-ta.keys())
        assert not removed and not changed,'STOP: destructive/relevant capture payload movement'
        assert newpaths==sorted(idir+x for x in ['checkpoint.json','closure_credit_overlay.jsonl','evidence_manifest.json','report.md','validation.json'])
        copies=[]
        for path in newpaths:
            captured=v.git('cat-file','blob',tb[path].split()[2]);actual=v.raw(path)
            copies.append({'path':path,'tree_entry':tb[path],'captured_sha256':v.sha(captured),'actual_authoritative_sha256':v.sha(actual),'raw_bytes_equal':captured==actual,'raw_equality_status':'PASS' if captured==actual else 'FAIL_CAPTURE_COPY_BYTE_EQUALITY','CRLF_to_LF_diagnostic_equal':captured.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n'),'used_as_authoritative_input':False})
        assert all(x['CRLF_to_LF_diagnostic_equal'] for x in copies)
        capture={'old_ref':deleted[0],'new_ref':added[0],'old_tree':a,'new_tree':b,'old_tree_still_available':True,'new_tree_available':True,'old_entries':len(ta),'new_entries':len(tb),'removed_entries':removed,'changed_existing_entries':changed,'new_capture_copies':copies,'raw_ref_status':'FAIL_REFS_MOVED','qualification':'NONBLOCKING_AUTHENTICATED_SNAPSHOT_EXTENSION','basis':'Every prior tree entry is exactly preserved and only five integration-output snapshot copies are added. Old tree remains readable. Capture copies are not authority selectors and no required original input changes. Deleted ref is not claimed unchanged or a commit fast-forward. Actual authoritative bytes authenticate independently. LF/CRLF raw-copy differences remain diagnostic FAILs.','actor_cause_capture_time':'UNKNOWN'}
        changes.extend(raw_movements)
    for ref in sorted(old.keys()|new.keys()):
        if old.get(ref)==new.get(ref):continue
        if ref in deleted+added:continue
        a,b=old.get(ref),new.get(ref);assert a and b,'STOP: pre-baseline ref creation/deletion requires authentication'
        assert v.git('cat-file','-t',a).strip()==b'commit' and v.git('cat-file','-t',b).strip()==b'commit'
        c=subprocess.run(['git','merge-base','--is-ancestor',a,b],cwd=v.ROOT,capture_output=True); assert c.returncode==0
        paths=set();detail=[]
        for commit in v.git('rev-list',a+'..'+b).decode().splitlines():
            s=v.git('diff-tree','--root','-m','--no-commit-id','--name-status','-r','--no-renames',commit).decode();detail.append({'commit':commit,'complete_name_status':s})
            for line in s.splitlines():paths.update(line.split('\t')[1:])
        assert not paths&pinned
        assert all(x.startswith(('scripts/gen5_','reports/reason_router_gen5_','reports/gen5_','tests/test_gen5_','tests/test_reason_router_gen5_')) for x in paths),'STOP: unknown prebaseline changed-path relationship'
        changes.append({'ref':ref,'before':a,'after':b,'raw_status':'FAIL_REFS_MOVED','complete_per_commit_changes':detail,'pinned_overlap':[],'qualified_status':'AUTHENTICATED_UNRELATED_NONDESTRUCTIVE_FAST_FORWARD','all_changed_paths':sorted(paths),'actors_causes_times':'UNKNOWN'})
    e['prebaseline_global_ref_comparison']={'source':idir+'validation.json#/final_preservation','source_sha256':v.sha(v.raw(idir+'validation.json')),'all_prior_ref_endpoints_literal':oldlines,'all_initial_ref_endpoints_literal':v.read('baseline.json')['start_state']['refs'],'movements':changes,'capture_extension':capture,'raw_status':'FAIL_REFS_MOVED' if changes else 'PASS_REFS_UNCHANGED','qualification':'NONBLOCKING_AUTHENTICATED_PINNED_INPUT_INDEPENDENCE','historical_unobserved_interval_retained':True}
    e['prior_failure_records_literal']={k:iv[k] for k in ['raw_failed_checks_retained','failed_probe_retained','historical_raw_failures_retained']}
    for kind in ['publication_semantic_batch_p01_20261014','publication_hv000103_independent_verification_20261014']:
        pv=json.loads(v.raw(v.W4+kind+'/validation.json'));e['prior_failure_records_literal'][kind]={k:x for k,x in pv.items() if 'fail' in k.lower() or 'uncertified' in k.lower()}
    rows=[json.loads(l) for l in v.raw(v.W3+'study_consolidation/unresolved_census_debt.jsonl').splitlines()]
    frozen=[json.loads(l) for l in v.raw(v.W4+'census_exit_reconciliation_20261014/debt_reconciliation.jsonl').splitlines()]
    retained=[]
    for owner in ['HV001006','HV001270']:
        for i,r in enumerate(rows,1):
            if owner in r.get('stable_original_source_ids',[]):
                f=next(x for x in frozen if x['debt_id']==r['debt_id']);assert not f['documentary_obligation_fully_satisfied']
                retained.append({'source_id':owner,'original_debt_id':r['debt_id'],'original_line':i,'original_record':r,'frozen_disposition':f['status'],'remaining_strict':f['blocks_strict_census_completeness_remaining'],'new_credit':0})
    assert {x['source_id'] for x in retained}=={'HV001006','HV001270'}
    e['retained_unrelated_open_debts']=retained
    v.write('evidence_manifest.json',e)
    print(json.dumps({'prebaseline_ref_movements':len(changes),'raw_status':e['prebaseline_global_ref_comparison']['raw_status'],'qualification':'NONBLOCKING_AUTHENTICATED','unrelated_open_debts_retained':len(retained)}))

def validate():
    checks=[]
    def ck(name,condition,detail=None):
        checks.append({'check':name,'status':'PASS' if condition else 'FAIL','detail':detail})
        if not condition:
            v.write('validation.json',{'status':'BLOCKED','checks':checks});raise ValueError(name)
    base=v.read('baseline.json'); man=v.read('evidence_manifest.json');cp=v.read('checkpoint.json')
    rows=[json.loads(l) for l in (v.OUT/'owner_verdicts.jsonl').read_bytes().splitlines()]
    ck('Exactly two independently recorded specified original owners',[r['source_id'] for r in rows]==['HV000107','HV000113'] and {r['debt_id'] for r in rows}=={'DEBT-44a94a2575ad6f3415ca','DEBT-86b191b570af6722edd7'})
    loc_count=0; pinned=[];object_rechecks=[];source_rechecks=[]
    for r,done in zip(rows,cp['completed_owners']):
        ck(r['source_id']+' durable completed record preserved',v.sha((v.OUT/(r['source_id']+'_evidence.json')).read_bytes())==r['independent_evidence_sha256']==done['evidence_sha256'])
        e=v.read(r['source_id']+'_evidence.json')
        ck(r['source_id']+' independent original retrieval',e['original_records']['original_debt']['record']['debt_id']==r['debt_id'] and e['p01_original_payload_matches_independent_original'])
        ck(r['source_id']+' complete actual PNG/PDF visuals',len(e['complete_visual_inspection']['checks'])==6 and all(x['actual_visual_inspection_complete'] for x in e['complete_visual_inspection']['checks']) and not e['complete_visual_inspection']['ocr_used'] and not e['complete_visual_inspection']['extracted_text_substituted_for_visuals'])
        ck(r['source_id']+' six independently satisfied clauses',len(r['clauses'])==6 and all(x['assessment'].startswith('SATISFIED') and x['authenticated_source_locators'] for x in r['clauses']))
        for f in r['findings']:
            for l in f['evidence']:v.locator(l);loc_count+=1
        for c in e['object_checks']:
            object_rechecks.append(v.authenticate({'commit':c['commit'],'path':c['path'],'blob':c['actual_blob'],'sha256':c['raw_sha256'],'bytes':c['size']},c['saved_file']))
        for c in e['source_identity_checks']:
            source_rechecks.append(v.authenticate({'commit':c['commit'],'path':c['path'],'blob':c['actual_blob'],'sha256':c['raw_sha256'],'bytes':c['size']}))
        ck(r['source_id']+' figure-specific source values independently matched',len(e['figure_specific_value_checks'])==(317 if r['source_id']=='HV000107' else 21) and all(z['source_bound_match'] for z in e['figure_specific_value_checks']))
        ck(r['source_id']+' complete P01 comparison verified',e['p01_complete_json_operations_reproduced'] and len(e['json_operations'])==(84 if r['source_id']=='HV000107' else 40))
        ck(r['source_id']+' no scientific certification/no credit',not r['scientific_claims_certified'] and r['closure_credit_applied']==0 and not r['residual_original_documentary_requirements'])
    ck('Original before/after exact Git and saved bytes reauthenticated',len(object_rechecks)==16 and all(x['matches'] for x in object_rechecks))
    ck('All required source identities reauthenticated',len(source_rechecks)==94 and all(x['matches'] for x in source_rechecks))
    sup=v.read('supplemental_checks.json')
    for c in sup['owners']['HV000113']['additional_manuscript_context']['bounded_additional_source_authentication']:
        source_rechecks.append(v.authenticate({'commit':c['commit'],'path':c['path'],'blob':c['actual_blob'],'sha256':c['raw_sha256']}))
    ck('Exact-decimal endpoint and chronology supplementary checks',all(z['exact_decimal_complete_operation_paths_match'] and z['exact_decimal_existing_values_unchanged'] for z in sup['owners'].values()))
    for x in man['local_pinned_input_checks']:
        pinned.append({'file':x['file'],'expected_sha256':x['expected_sha256'],'actual_sha256':v.sha(v.raw(x['file'])),'matches':v.sha(v.raw(x['file']))==x['expected_sha256']})
    ck('Scoped frozen/controller/verified-input hashes freshly rechecked',len(pinned)==99 and all(x['matches'] for x in pinned))
    original=[json.loads(l) for l in v.raw(v.W3+'study_consolidation/unresolved_census_debt.jsonl').splitlines()]
    frozen=[json.loads(l) for l in v.raw(v.W4+'census_exit_reconciliation_20261014/debt_reconciliation.jsonl').splitlines()]
    ids={r['debt_id'] for r in original}; closed={r['debt_id'] for r in frozen if r['documentary_obligation_fully_satisfied']};strict={r['debt_id'] for r in frozen if r['blocks_strict_census_completeness_remaining']}
    ov=man['hv000103_accounting_baseline']['overlay_record'];new={ov['debt_id']}
    ck('Frozen exact accounting unchanged',len(original)==len(ids)==len(frozen)==222 and len(closed)==17 and len(strict)==195)
    ck('Existing single HV000103 overlay authentic and only existing credit',ov['source_id']=='HV000103' and ov['applied_closure_credit']==1 and not new&closed and new<=strict and ov['satisfied_delta']==1 and ov['strict_blocker_delta']==-1)
    ck('Current accounting remains222/18/194',len(closed|new)==18 and len(strict-new)==194 and all(r['debt_id'] in strict-new for r in rows))
    ck('No new closure credit',all(r['closure_credit_applied']==0 for r in rows) and cp['new_closure_credits']==0)
    current=v.inventory();removed=sorted(base['preexisting_files'].keys()-current.keys());added=sorted(current.keys()-base['preexisting_files'].keys());changed=sorted(k for k in current.keys()&base['preexisting_files'].keys() if current[k]!=base['preexisting_files'][k])
    ck('All preexisting files byte-preserved',not removed and not changed,{'count':len(base['preexisting_files']),'removed':removed,'changed':changed})
    ck('No new files outside exact authorized directory',not added,added)
    s=v.state();initial=base['start_state'];ck('HEAD/index/tracked/staged/detached state preserved',all(s[k]==initial[k] for k in initial if k not in ['refs','show_ref_head_dereference']))
    ck('Final observed ref state equals latest authenticated endpoint',s['refs']==man['ref_observations'][-1]['state']['refs'] and s['show_ref_head_dereference']==man['ref_observations'][-1]['state']['show_ref_head_dereference'])
    ck('All current global movements classified without deleting raw FAIL',man['current_ref_movement_qualification']['raw_status']=='FAIL_REFS_MOVED' and len(man['current_ref_movement_qualification']['movements'])==2 and not man['current_ref_movement_qualification']['blocks_adjudication'])
    ck('Prebaseline capture movements independently authenticated',man['prebaseline_global_ref_comparison']['capture_extension']['qualification']=='NONBLOCKING_AUTHENTICATED_SNAPSHOT_EXTENSION')
    ck('Historical FAILs/LFCRLF/FAIL_PROBE retained',bool(man['prior_failure_records_literal']['failed_probe_retained']) and man['historical_preservation']['capture_discrepancy_followup']['final_status']=='RETAINED_NONBLOCKING_PROVENANCE_LIMITATION')
    ck('Open HV001006/HV001270 retained',len(man['retained_unrelated_open_debts'])==2 and all(x['new_credit']==0 for x in man['retained_unrelated_open_debts']))
    wt_start=man['worktrees_first_observation'];wt_end=v.git('worktree','list','--porcelain').decode()
    membership=lambda text:[l for l in text.splitlines() if not l.startswith('HEAD ')]
    raw_wt_equal=wt_start==wt_end
    wtqualification={'raw_status':'PASS_WORKTREE_LIST_UNCHANGED' if raw_wt_equal else 'FAIL_WORKTREE_LIST_CHANGED','start_literal':wt_start,'end_literal':wt_end,'membership_equal':membership(wt_start)==membership(wt_end),'qualification':'PENDING'}
    if not raw_wt_equal:
        blocks=lambda text:{b.splitlines()[0]:b.splitlines() for b in text.strip().split('\n\n')}
        wa,wb=blocks(wt_start),blocks(wt_end);differences=[]
        for path in wa.keys()|wb.keys():
            if wa.get(path)==wb.get(path):continue
            assert path in wa and path in wb and [l for l in wa[path] if not l.startswith('HEAD ')]==[l for l in wb[path] if not l.startswith('HEAD ')]
            before=next(l[5:] for l in wa[path] if l.startswith('HEAD '));after=next(l[5:] for l in wb[path] if l.startswith('HEAD '));branch=next(l[7:] for l in wa[path] if l.startswith('branch '))
            assert any(m['ref']==branch and m['before']==before and m['after']==after and m['qualified_status']=='PASS_AUTHENTICATED_UNRELATED_NONDESTRUCTIVE_FAST_FORWARD' for m in man['current_ref_movement_qualification']['movements'])
            differences.append({'worktree':path,'branch':branch,'before_head':before,'after_head':after,'matches_authenticated_global_ref_movement':True})
        wtqualification.update(qualification='PASS_MEMBERSHIP_UNCHANGED_AUTHENTICATED_UNRELATED_HEAD_MOVEMENT',changes=differences,raw_failure_relabelled=False)
    else:wtqualification['qualification']='PASS_LIST_UNCHANGED'
    man['worktree_endpoint_qualification']=wtqualification
    ck('Worktree membership preserved with separately retained raw listing movement',wtqualification['membership_equal'] and wtqualification['qualification']!='PENDING')
    commands={}
    for name,args in [('diff_check',['diff','--check']),('tracked_diff_stat',['diff','--stat']),('staged_diff_stat',['diff','--cached','--stat'])]:
        p=subprocess.run(['git',*args],cwd=v.ROOT,capture_output=True);commands[name]={'argv':['git',*args],'returncode':p.returncode,'stdout':p.stdout.decode(),'stderr':p.stderr.decode()};ck(name,p.returncode==0 and not p.stdout and not p.stderr)
    ck('Early and sequential owner durability',cp['completed_owners'][0]['source_id']=='HV000107' and cp['completed_owners'][1]['source_id']=='HV000113' and cp['completed_owners'][0]['completed_utc']<cp['completed_owners'][1]['completed_utc'])
    man['final_scoped_input_recheck']={'checked_utc':v.now(),'local':pinned,'saved_objects':object_rechecks,'source_objects':source_rechecks,'supplement_sha256':v.sha((v.OUT/'supplemental_checks.json').read_bytes())}
    man['final_state']=s
    man['commands_actually_executed']=['PowerShell Get-Location/Get-Content/Get-Command/rg file and text discovery','Read-only git status/rev-parse/symbolic-ref/for-each-ref/show-ref/ls-files/diff/cat-file/show/merge-base/rev-list/diff-tree/ls-tree/log/worktree list','Task-local Python JSON/JSONL/SHA256/raw byte equality/Decimal/CSV/source-locator and serialized readback checks','Bundled Pillow/pypdf metadata; complete original PNG viewing; Poppler pdftoppm rendering and complete page viewing','Task-local report/evidence/checkpoint/validation writes; no prior research producer executed']
    man['current_probe_records']=[{'status':'FAIL_PROBE','description':'Initial original-debt lookup used an incorrect field; corrected to stable_original_source_ids before retrieval persisted.'},{'status':'FAIL_PROBE','description':'System Python lacked PIL; switched to existing bundled Python, no install.'},{'status':'FAIL_PROBE','description':'Positive-locator construction requested after-only chain_supported in before manifest; corrected to after-only locator before owner verdict persisted.'},{'status':'FAIL_PROBE','description':'Initial ref qualification path guard did not recognize reports/reason_router_gen5_ prefix; full added JSON read, path relationship and pinned independence authenticated before qualification.'},{'status':'FAIL_PROBE','description':'Initial prebaseline ref checker stopped on capture ref replacement; raw movements persisted and exact complete tree payload extension independently authenticated.'},{'status':'FAIL_WORKTREE_LIST_CHANGED','description':'First final-validation attempt failed literal worktree-list equality because C:/g5-state HEAD changed fda31c0... to d1174fc... on the already authenticated unrelated Gen5 branch. Exact raw FAIL retained; worktree path/branch/membership and this audit checkout are preserved. See worktree_endpoint_qualification.'},{'status':'OBSERVATION','description':'Early broad console reads truncated; replaced by bounded reads, complete durable records, full comparisons and direct actual visuals. No conclusion relies on truncation.'}]
    man['activity'].update(training=False,evaluation=False,kaggle_gpu=False,commit_push=False,fetch=False,ocr=False,scientific_certification=False)
    v.write('evidence_manifest.json',man)
    cp.update(status='COMPLETE_TWO_INDEPENDENT_VERDICTS_NO_CREDIT',unfinished_owners=[],finished_utc=v.now(),owner_verdicts_sha256=v.sha((v.OUT/'owner_verdicts.jsonl').read_bytes()),validation_scope='Documentary verification and preservation only; raw ref FAILs retained with authenticated bounded nonblocking qualifications.')
    v.write('checkpoint.json',cp)
    write_report(rows,checks,s,man)
    required=['checkpoint.json','owner_verdicts.jsonl','evidence_manifest.json','validation.json','report.md']
    allowed=sorted(required+['baseline.json','original_records.json','HV000107_evidence.json','HV000113_evidence.json','verify.py','adjudicate.py','final_checks.py','supplemental_checks.json','081_before_pdf_page1.png','081_after_pdf_page1.png','042_before_pdf_page1.png','042_after_pdf_page1.png'])
    ck('Exact output scope',sorted(p.name for p in v.OUT.iterdir() if p.name!='validation.json')==[n for n in allowed if n!='validation.json'])
    validation={'status':'PASS_DOCUMENTARY_AND_PRESERVATION_WITH_AUTHENTICATED_NONBLOCKING_RAW_REF_FAILS','phase':'CENSUS_RECOVERY_ONLY','validated_utc':v.now(),'checks':checks,'checks_passed':len(checks),'blocking_failed_checks':[],'raw_current_global_ref_status':'FAIL_REFS_MOVED','raw_prebaseline_global_ref_status':man['prebaseline_global_ref_comparison']['raw_status'],'raw_failures_relabelled':False,'source_locator_rechecks':loc_count,'preexisting_files_byte_preserved':len(base['preexisting_files']),'historical_frozen_accounting':[222,17,195],'current_accounting':[222,18,194],'new_closure_credits':0,'final_state':s,'commands':commands,'expected_output_names':allowed,'output_sha256_excluding_self':{p.name:v.sha(p.read_bytes()) for p in sorted(v.OUT.iterdir()) if p.is_file() and p.name!='validation.json'},'self_hash_policy':'No self-hash claimed. This JSON is reopened after writing; all other output hashes and current input/state fingerprints are checked.','activity':man['activity'],'current_probe_records':man['current_probe_records'],'scientific_certification':False}
    v.write('validation.json',validation)
    print(json.dumps({'validation':'PASS_WITH_RAW_REF_FAILS_RETAINED','checks':len(checks),'positive_locators_rechecked':loc_count,'preexisting_files_preserved':len(base['preexisting_files']),'new_credits':0,'accounting':[222,18,194]}))

def write_report(rows,checks,state,manifest):
    lines=['# Two-owner publication documentary verification - 2026-10-14','', '**HV000107 and HV000113 each independently receive VERIFIED_FULL_DOCUMENTARY_SATISFACTION. Zero new closure credits. Current accounting remains 222 / 18 / 194; frozen historical accounting remains 222 / 17 / 195.**','', 'Authority: current controller instruction; frozen census_exit_reconciliation_20261014; each owner\'s original Wave03 debt/root/transition; frozen publication_semantic_batch_p01_20261014; verified and integrated HV000103; repository AGENTS.md. Phase CENSUS_RECOVERY_ONLY. Expected and observed detached HEAD: `'+v.EXPECTED+'`. No scientific claim is certified.','', '1. Independently retrieved original identities','', '| Owner | Original debt | Original debt line | Root line | Transition line | Discovered saved evidence |','|---|---|---:|---:|---:|---|']
    for row in rows:
        e=v.read(row['source_id']+'_evidence.json');z=e['original_records']
        lines.append('| '+row['source_id']+' | '+row['debt_id']+' | '+str(z['original_debt']['line'])+' | '+str(z['original_root']['line'])+' | '+str(z['original_transition']['line'])+' | '+e['authenticated_saved_transition']['file'].split('/')[-1]+' |')
    lines.extend(['','Debt file: `wave_03/study_consolidation/unresolved_census_debt.jsonl`, SHA256 `'+rows[0]['original_debt_locator']['file_sha256']+'`. Root and transition are independently read, not inherited from P01. Complete raw-line/file hashes and payloads reside in original_records.json and each owner evidence file. Both roots have no additional source-specific requirements/dependency roots.','', '> Complete interpretation/reconciliation of saved before/after publication evidence and authenticated companion sources; identify all substantive claim/endpoint/provenance changes versus render-only differences. Existing recovery is not completed semantic review.','', 'Exact P01 candidate rows are publication_assessments.jsonl lines 2 and 3. All five frozen P01 files authenticate against the prior independent verifier/integration bindings; their literal candidate rows and hashes are saved. P01 claims were independently tested, not given automatic satisfaction.',''])
    for number,row in enumerate(rows,2):
        owner=row['source_id'];e=v.read(owner+'_evidence.json')
        lines.extend([str(number)+'. '+owner+' - Figure '+('2' if owner=='HV000107' else '3')+' claim, endpoint, provenance and rendering',''])
        for f in row['findings']:lines.extend(['**'+f['category']+'**: '+f['interpretation'],''])
        if owner=='HV000113':lines.extend(['**Additional authenticated manuscript context**: '+v.read('supplemental_checks.json')['owners'][owner]['additional_manuscript_context']['interpretation'],'','The exact manuscript changes, appendix339-375, four narrowly referenced source identities and eight source-median/display checks are in supplemental_checks.json. This supplements the completed review without rewriting either owner record.'])
        lines.extend(['','**Exact immutable visual identities** (raw bytes, not normalized capture copies):','','| Side/format | Commit | Git blob | Raw SHA256 | Saved file |','|---|---|---|---|---|'])
        for c in e['object_checks']:
            if c['saved_file'].endswith(('.pdf','.png')):
                lines.append('| '+('before' if '_before' in c['saved_file'] else 'after')+'/'+c['saved_file'].split('.')[-1]+' | `'+c['commit']+'` | `'+c['actual_blob']+'` | `'+c['raw_sha256']+'` | `'+c['saved_file']+'` |')
        lines.extend(['','The repository path for each immutable object and all README/manifest companion identities are in '+owner+'_evidence.json#/object_checks. Eight exact saved/Git objects authenticate per owner. Both complete original PNGs and both complete single-page PDFs were visually inspected; retained Poppler page renders bind the viewed pages to authenticated source PDF bytes. No OCR or text-only visual substitution was used.',''])
    lines.extend(['4. Independent original clause coverage','','| Clause | HV000107 | HV000113 | Positive source coverage |','|---|---|---|---|'])
    for a,b in zip(rows[0]['clauses'],rows[1]['clauses']):lines.append('| '+a['clause_id']+' '+a['literal_component']+' | '+a['assessment']+' | '+b['assessment']+' | '+', '.join(a['finding_ids'])+' / '+', '.join(b['finding_ids'])+' |')
    lines.extend(['','Every SATISFIED clause embeds authenticated file+SHA256+page/panel/JSON pointer or commit+path+blob+SHA256+source-line locators in owner_verdicts.jsonl. These are directly rechecked in final validation. Complete 84/40-operation manifest comparisons, full README hunks, 317/21 figure-specific source checks, 390/398 unchanged extracted values and exact-decimal follow-up are saved. All45/49 paired manifest source identities authenticate. Figure3\'s manifest/README pairs are byte-identical to verified HV000103\'s exact pairs; only that common interpretation is reused. Figure-specific values, references and visuals are independently examined.','','5. Two separate final verdicts and exact residuals',''])
    for r in rows:lines.extend(['- **'+r['source_id']+' / '+r['debt_id']+': '+r['verdict']+'**. Unsupported original documentary clauses: none. New credits: 0.',''])
    lines.extend(['The obligation is documentary completion. Independent science, execution identity, global adoption, editorial motive and a proof of semantic homology remain uncertified; these limits are not silently converted into invented original documentary requirements. Selected ancestor is not necessarily immediate predecessor. Actual parent publication blobs and direct-parent booleans are freshly authenticated in supplemental_checks.json.','','6. Accounting baseline and retained historical limits','','HV000103\'s existing closure_credit_overlay.jsonl remains exactly one authenticated row, debt DEBT-749ea4d7ff425c9846b4; its existing +1/-1 is the only derived accounting adjustment. All declared integration/independent-verification outputs and frozen P01 inputs match expected hashes. Frozen222/17/195 and overlay-adjusted222/18/194 are recomputed from exact original-ID sets. Both present targets remain uncredited strict IDs in the accounting view. No overlay or canonical ledger is changed.','','Retained literally and byte-preserved: FAIL_REQUIRED_REFS_UNCHANGED_CHECK; raw P01 FAIL_REFS_MOVED; HV000103 verifier raw FAIL_REFS_MOVED; earlier capture-copy LF/CRLF discrepancy including all_five_added_files_match_frozen_packet=false; historical uncertified intervals and unknown ref actors/causes/times; HV000116 original-tree owning-commit/capture chronology limitation; open HV001006 and HV001270 debts; earlier FAIL_PROBE records; historical370M failed joint residual criterion. Source absence, documentary removal and scientific falsification remain separate.','','7. Final preservation, ref policy and validation','','All '+str(len(v.read('baseline.json')['preexisting_files']))+' preexisting workspace files are byte-preserved. HEAD/index/tracked/staged/detached state is preserved. No new files exist outside this exact output directory. Tracked and staged diff statistics are empty; git diff --check actually passed. Worktree membership is preserved. The initial baseline and sequential owner append/checkpoint writes are durable. Final validation rechecks99 declared local input hashes,16 original saved/Git visual+companion identities and94 paired source identities plus four bounded manuscript-reference sources. Fresh JSON/JSONL/report/output-hash disk readback is recorded separately in validation.json.','','**Current global raw ref result is FAIL_REFS_MOVED**, retained. During this task refs/heads/gen5-causal-role-state-ownership and refs/remotes/origin/gen5-causal-role-state-ownership moved fda31c03751d9195851fd2f5a66e08c518703ed7 -> d1174fce8770c77d4e9ef62f08e0bfa1cbd863e2. Every commit\'s complete path changes authenticate: only the added Gen5 strict-load report. Full report contents were read; the change does not select or mutate pinned publication inputs. Ancestry proves non-destructive fast-forwards and all pinned inputs reauthenticate. The controller\'s unrelated-movement qualification is therefore nonblocking; the raw FAIL is never relabeled.','','A separate prebaseline comparison against the integrated HV000103 final endpoints records a capture-ref removal/addition. Old tree78ce739a697d15c8cdab0ce50bc70af8ed9d30e5 remains available; new tree4e0ab51ae66f49a08f704ac3666badb06b2ff313 preserves every old entry and adds only the five integration snapshot copies. Captured LF bytes can differ from actual final CRLF bytes; raw equality FAILs are retained, and actual authoritative hashes match independently. Capture refs are not authority selectors. This is a bounded non-destructive snapshot extension qualification, not an unchanged-ref or commit-fast-forward claim. Full endpoints, raw failures, per-commit/tree checks and uncertified observation intervals are retained in evidence_manifest.json.','','Only observable endpoint movements are authenticated. Prebaseline history, transient movement returning between observations, actor/cause and exact times remain uncertified. No fetch, ref write, branch/worktree mutation, commit or push was performed by this verifier.','','Commands actually executed: read-only Git status/identity/index/ref/object/ancestry/range-path/tree/diff/stat/check/worktree-list operations; PowerShell/rg inspection; task-local Python JSON/JSONL/SHA256/Decimal/CSV comparisons and readback; Pillow/pypdf metadata; complete PNG viewing; Poppler rendering and complete PDF-page viewing. Research tests/producers, training/evaluation, model/checkpoint loading, data regeneration, OCR, Kaggle/GPU and scientific validation were not run. Probe failures and truncated early reads are retained with corrected bounded checks; no failed evidence assertion is hidden as PASS.','','Only new files under this directory are created: the five required deliverables, baseline/original/per-owner evidence, supplemental checks, four page renders and three task-local reproducibility helpers. Full exact output names/hashes are in validation.json. Existing audits, overlays, figures, source code and checkpoint data remain byte-identical.','','8. Minimum next integration action','','A separately authorized controller integration review may consider these two exact debt IDs independently, authenticate this packet and decide whether to add one credit per eligible owner. Preserve the existing HV000103 overlay and frozen history. No integration is performed or authorized by this verification; current totals stay222/18/194.'])
    (v.OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':globals()[sys.argv[1]]()
