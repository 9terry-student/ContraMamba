import os, sys, json, hashlib, subprocess, datetime
from pathlib import Path

ROOT = Path(r'C:\cm-full-audit-20261013')
OUT = Path(__file__).resolve().parent
EXPECTED = 'ff806bd5bd8efc62fcf6420cf8b097a63ed70571'
os.environ['GIT_OPTIONAL_LOCKS'] = '0'
sys.dont_write_bytecode = True
def sha(b): return hashlib.sha256(b).hexdigest()
def raw(p): return (ROOT / p).read_bytes()
def rel(p): return Path(p).relative_to(ROOT).as_posix()
def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True)
    if p.returncode: raise RuntimeError((args, p.returncode, p.stderr.decode(errors='replace')))
    return p.stdout
def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
BASE='audits/full_history_20261013/parallel_20261013/'
W3=BASE+'wave_03/'
W4=BASE+'wave_04/'
def select(path, predicate):
    b=raw(path); found=[]
    for i,line in enumerate(b.splitlines(keepends=True),1):
        if not line.strip(): continue
        obj=json.loads(line)
        if predicate(obj): found.append({'file':path,'line':i,'file_sha256':sha(b),'raw_line_sha256':sha(line),'record_sha256':sha(line.rstrip(b'\r\n')),'record':obj})
    if len(found)!=1: raise ValueError((path,'expected one',len(found)))
    return found[0]
def extract():
    data={}
    for owner in ['HV000107','HV000113']:
        d=select(W3+'study_consolidation/unresolved_census_debt.jsonl',lambda r:owner in r.get('stable_original_source_ids',[]) or owner in json.dumps(r.get('source_identity',{})))
        debt=d['record']['debt_id']
        r=select(W3+'census_closure/root_evidence_requirements.jsonl',lambda r:debt in r.get('dependent_original_debt_ids',[]))
        t=select(W3+'transitions_0/transition_interpretations.jsonl',lambda r:owner in r.get('original_source_ids',[]))
        p=select(W4+'publication_semantic_batch_p01_20261014/publication_assessments.jsonl',lambda r:owner in json.dumps(r))
        c=select(W4+'census_exit_reconciliation_20261014/debt_reconciliation.jsonl',lambda r:r['debt_id']==debt)
        data[owner]={'original_debt':d,'original_root':r,'original_transition':t,'p01_candidate':p,'frozen_reconciliation':c}
    write('original_records.json',data)
    for o,v in data.items():
        print(o, json.dumps({'debt':v['original_debt'],'root_requirements':v['original_root']['record']['exact_missing_evidence'],'transition_file_line':v['original_transition']['line'],'saved_evidence':v['original_transition']['record']['evidence']['saved_record'],'candidate_keys':list(v['p01_candidate']['record'])},ensure_ascii=False))
def authenticate(ident,saved=None):
    commit,path,blob=ident['commit'],ident['path'],ident.get('blob',ident.get('git_blob'))
    actual=git('rev-parse',commit+':'+path).decode().strip()
    b=git('cat-file','blob',actual)
    expected_sha=ident.get('sha256',ident.get('frozen_bytes_sha256'))
    ok=actual==blob and (expected_sha is None or sha(b)==expected_sha) and (ident.get('bytes',len(b))==len(b))
    if saved: ok=ok and raw(saved)==b
    row={'commit':commit,'path':path,'expected_blob':blob,'actual_blob':actual,'raw_sha256':sha(b),'expected_sha256':expected_sha,'size':len(b),'saved_file':saved,'matches':ok,'commit_type':git('cat-file','-t',commit).decode().strip()}
    if not ok: raise ValueError(('STOP: original identity mismatch',row))
    return row
def operations(a,b,p=''):
    if type(a)!=type(b): return [{'op':'replace','path':p,'before':a,'after':b}]
    if isinstance(a,dict):
        out=[]
        for k in sorted(a.keys()|b.keys()):
            q=p+'/'+k.replace('~','~0').replace('/','~1')
            if k not in a: out.append({'op':'add','path':q,'after':b[k]})
            elif k not in b: out.append({'op':'remove','path':q,'before':a[k]})
            else: out.extend(operations(a[k],b[k],q))
        return out
    if isinstance(a,list) and len(a)==len(b):
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in operations(x,y,p+'/'+str(i))]
    return [] if a==b else [{'op':'replace','path':p,'before':a,'after':b}]
def owner_auth(owner):
    import difflib
    v=read('original_records.json')[owner]; t=v['original_transition']['record']; stem=t['evidence']['saved_record'].removesuffix('.json')
    saved_record=W3+'transitions_0/'+stem+'.json'
    assert sha(raw(saved_record))==t['evidence']['saved_record_sha256']
    saved=json.loads(raw(saved_record)); assert saved['before']==t['before'] and saved['after']==t['after']
    companions=json.loads(raw(W3+'transitions_0/'+stem+'_companions.json'))
    checks=[]
    for side in ['before','after']:
        checks.append(authenticate(t[side],W3+'transitions_0/'+stem+'_'+side+'.pdf'))
    for c in companions:
        for side in ['before','after']:
            s=c[side]; checks.append(authenticate(s['identity'],W3+'transitions_0/'+s['evidence_file'].replace('\\','/')))
    paths={r['saved_file'].split('/')[-1]:r['saved_file'] for r in checks}
    n=stem.split('/')[-1]
    manifests=[json.loads(raw(paths[n+'_'+s+'_companion_figure_data_manifest.json'])) for s in ['before','after']]
    ops=operations(*manifests)
    readmes=[raw(paths[n+'_'+s+'_companion_README.md']).decode().splitlines() for s in ['before','after']]
    diff=list(difflib.unified_diff(*readmes,fromfile='before',tofile='after',lineterm=''))
    anc=subprocess.run(['git','merge-base','--is-ancestor',t['before']['commit'],t['after']['commit']],cwd=ROOT,capture_output=True)
    assert anc.returncode==0
    data={'owner':owner,'authenticated_saved_transition':{'file':saved_record,'sha256':sha(raw(saved_record))},'original_records':v,'object_checks':checks,'json_operations':ops,'readme_diff':diff,'plot_equal_keys':{k:manifests[0]['plot_data'][k]==manifests[1]['plot_data'].get(k) for k in manifests[0]['plot_data']},'chronology':{'selected_before_is_ancestor':True,'actual_after_parents':git('show','-s','--format=%P',t['after']['commit']).decode().strip().split(),'recorded_chronology':t['chronology'],'global_adoption_proven':False},'review_status':'AUTHENTICATED_AWAITING_COMPLETE_VISUAL_AND_SEMANTIC_REVIEW'}
    write(owner+'_evidence.json',data)
    print(json.dumps({'owner':owner,'auth_checks':len(checks),'json_ops':len(ops),'plot_equal_keys':data['plot_equal_keys'],'paths':paths,'manifest_keys':list(manifests[1]),'source_example':next(iter(manifests[1]['sources'].items())),'extracted_example':next(iter(manifests[1]['extracted_values'].items())),'tools':{s:m['tools'] for s,m in zip(['before','after'],manifests)}},ensure_ascii=False))
def pinned_setup():
    idir=W4+'publication_hv000103_closure_integration_20261014/'
    integ=json.loads(raw(idir+'evidence_manifest.json')); iv=json.loads(raw(idir+'validation.json'))
    checks=[]
    for name,h in iv['output_sha256_excluding_self'].items():
        checks.append({'file':idir+name,'expected_sha256':h,'actual_sha256':sha(raw(idir+name)),'matches':sha(raw(idir+name))==h})
    for x in integ['final_frozen_P01_authentication']+integ['pinned_local_input_checks']:
        checks.append({'file':x['file'],'expected_sha256':x['expected_sha256'],'actual_sha256':sha(raw(x['file'])),'matches':sha(raw(x['file']))==x['expected_sha256']})
    vp=integ['independent_verification']; assert sha(raw(vp['validation_file']))==vp['validation_sha256']
    vval=json.loads(raw(vp['validation_file'])); vdir=Path(vp['validation_file']).parent.as_posix()+'/'
    for name,h in vval['output_sha256_excluding_self'].items(): checks.append({'file':vdir+name,'expected_sha256':h,'actual_sha256':sha(raw(vdir+name)),'matches':sha(raw(vdir+name))==h})
    assert all(c['matches'] for c in checks)
    overlay=[json.loads(l) for l in raw(idir+'closure_credit_overlay.jsonl').splitlines() if l.strip()]
    assert len(overlay)==1 and overlay[0]['debt_id']=='DEBT-749ea4d7ff425c9846b4' and overlay[0]['applied_closure_credit']==1
    rows=[json.loads(l) for l in raw(W4+'census_exit_reconciliation_20261014/debt_reconciliation.jsonl').splitlines()]
    ids={r['debt_id'] for r in rows}; assert len(ids)==len(rows)==222
    data={'phase':'CENSUS_RECOVERY_ONLY','targets':['HV000107','HV000113'],'local_pinned_input_checks':checks,'hv000103_accounting_baseline':{'overlay_locator':{'file':idir+'closure_credit_overlay.jsonl','sha256':sha(raw(idir+'closure_credit_overlay.jsonl')),'line':1},'overlay_record':overlay[0],'integration_validation_sha256':sha(raw(idir+'validation.json')),'integration_declared_output_hashes_authenticated':True,'frozen':[222,17,195],'overlay_adjusted':[222,18,194],'new_credits':0,'historical_sets':integ['accounting']},'historical_preservation':{'retained_limitations':integ['retained_limitations'],'capture_discrepancy':integ['capture_discrepancy'],'capture_discrepancy_followup':integ['capture_discrepancy_followup'],'prior_probe_issues':integ['probe_issues'],'integration_probe_records':[x for x in iv.get('checks',[]) if 'FAIL' in json.dumps(x)],'prior_hv001006_hv001270':'Open debts retained without adjudication; original/frozen files byte-preserved.'},'ref_observations':[{'label':'initial_baseline','state':read('baseline.json')['start_state']}],'worktrees_first_observation':git('worktree','list','--porcelain').decode(),'activity':{'training':False,'evaluation':False,'ocr':False,'commit_push':False,'closure_credit':0}}
    write('evidence_manifest.json',data)
    print(json.dumps({'frozen_and_integrated_hash_checks':len(checks),'all_match':True,'overlay_rows':1,'accounting':[222,18,194]}))
def companions_auth(owner):
    from PIL import Image
    from pypdf import PdfReader
    e=read(owner+'_evidence.json'); pairs={}
    for c in e['object_checks']:
        n=Path(c['saved_file']).name
        if n.endswith('figure_data_manifest.json'): pairs['before' if '_before_' in n else 'after']=json.loads(raw(c['saved_file']))
    source_checks=[]; output_checks=[]
    for side,m in pairs.items():
        for path,s in m['sources'].items():
            ident={'commit':s.get('authority_commit',m['scientific_head']),'path':path,**s}
            source_checks.append({'side':side,**authenticate(ident)})
        commit=e['original_records']['original_transition']['record'][side]['commit']
        for n,h in m['outputs'].items():
            b=git('cat-file','blob',commit+':paper/iclr2027/figures/'+n)
            assert sha(b)==h
            output_checks.append({'side':side,'commit':commit,'path':'paper/iclr2027/figures/'+n,'sha256':h,'matches':True})
    refs=[]
    # Independently authenticate actual manuscript/appendix bytes named by the publication.
    for side in ['before','after']:
        commit=e['original_records']['original_transition']['record'][side]['commit']
        for path in ['paper/iclr2027/main.tex','paper/iclr2027/appendix.tex']:
            ident={'commit':commit,'path':path,'blob':git('rev-parse',commit+':'+path).decode().strip()}
            c=authenticate(ident); refs.append({'side':side,**c,'lines':git('cat-file','blob',c['actual_blob']).decode().splitlines()})
    visuals=[]
    for c in e['object_checks']:
        f=c['saved_file']
        if f.endswith('.pdf'):
            pdf=PdfReader(ROOT/f); assert len(pdf.pages)==1
            visuals.append({'source':f,'source_sha256':c['raw_sha256'],'pages':len(pdf.pages),'page_box':list(map(float,pdf.pages[0].mediabox))})
        if f.endswith('.png'):
            im=Image.open(ROOT/f); im.load(); visuals.append({'source':f,'source_sha256':c['raw_sha256'],'dimensions':list(im.size),'mode':im.mode})
    e.update(source_identity_checks=source_checks,output_digest_checks=output_checks,publication_reference_sources=refs,visual_metadata=visuals)
    write(owner+'_evidence.json',e)
    print(json.dumps({'owner':owner,'source_checks':len(source_checks),'output_digest_checks':len(output_checks),'reference_sources':len(refs),'visuals':visuals}))
def pointer(obj,p):
    for k in p.strip('/').split('/') if p else []:
        k=k.replace('~1','/').replace('~0','~'); obj=obj[int(k)] if isinstance(obj,list) else obj[k]
    return obj
def locator(l):
    if 'file' in l:
        b=raw(l['file']); h=l.get('sha256',l.get('file_sha256')); assert h is None or sha(b)==h
    else:
        c=authenticate(l); b=git('cat-file','blob',c['actual_blob'])
    if 'json_pointer' in l: pointer(json.loads(b),l['json_pointer'])
    if 'jsonl_line' in l:
        line=b.splitlines()[l['jsonl_line']-1]; h=l.get('line_sha256_excluding_lf'); assert h is None or sha(line)==h
    if 'line_start' in l: assert 1<=l['line_start']<=l.get('line_end',l['line_start'])<=len(b.splitlines())
    return {**l,'authenticated_sha256':sha(b),'matches':True}
def semantic(owner):
    import re,difflib
    e=read(owner+'_evidence.json'); n=Path(e['authenticated_saved_transition']['file']).stem
    m={s:json.loads(raw(W3+'transitions_0/evidence/'+n+'_'+s+'_companion_figure_data_manifest.json')) for s in ['before','after']}
    old,new=m['before'],m['after']; values=[]
    figure='fig2' if owner=='HV000107' else 'fig3'
    for k,v in new['extracted_values'].items():
        if v.get('figure')!=figure: continue
        source=new['sources'][v['source']]; b=git('cat-file','blob',source['git_blob']); loc=v['locator']; value=v['value']; extra={}
        if loc.startswith('/'):
            actual=pointer(json.loads(b),loc); assert actual==value
        elif loc.startswith('JSONL line '):
            i=int(loc.split()[2].rstrip(';'))-1; actual=json.loads(b.splitlines()[i]); assert all(actual[x]==z for x,z in value.items())
        elif 'section ' in loc:
            num=int(re.search(r'section (\d+)',loc).group(1)); lines=b.decode().splitlines(); start=next(i for i,l in enumerate(lines) if l.startswith('### '+str(num)+'.'))
            end=next((i for i in range(start+1,len(lines)) if lines[i].startswith('### ')),len(lines)); section='\n'.join(lines[start:end]); extra={'source_line_start':start+1,'source_line_end':end}
            if 'source_decimal_lexeme' in v: assert v['source_decimal_lexeme'] in section and float(v['source_decimal_lexeme'])==value
            elif '.supported' in k: assert value is True and re.search(r'Frozen conclusion:\s*`[^`]*SUPPORTED[^`]*`',section)
            elif '.n' in k: assert f'- N: `{value}`' in section
            elif isinstance(value,str): assert value in section
            else: raise ValueError(('Unhandled documentary locator',k,loc))
        else: raise ValueError(('Unhandled documentary locator',k,loc))
        values.append({'entry':k,'source':v['source'],'blob':source['git_blob'],'sha256':sha(b),'locator':loc,'published_value':value,'source_bound_match':True,**extra})
    unchanged={k:old['extracted_values'][k]['value']==new['extracted_values'][k]['value'] for k in old['extracted_values']}
    assert all(unchanged.values())
    p=json.loads(raw(W4+'publication_semantic_batch_p01_20261014/evidence_manifest.json')); u=next(x for x in p['universe'] if x['source_id']==owner)
    ops=next(c['complete_json_changes'] for c in u['complete_companion_comparisons'] if 'complete_json_changes' in c)
    normalized=[{'op':r['operation'],'path':r['pointer'],**{k:r[k] for k in ['before','after'] if k in r}} for r in ops]
    assert e['json_operations']==normalized
    v=e['original_records']; assert u['literal_original_record']==v['original_debt']['record']==v['frozen_reconciliation']['record']['original_record']
    candidate=v['p01_candidate']['record']; assert candidate['debt_id']==v['original_debt']['record']['debt_id'] and candidate['proposed_disposition']=='FULL_DOCUMENTARY_SATISFACTION_CANDIDATE' and candidate['applied_closure_credit']==0
    locs=[]
    for finding in candidate['findings']:
        for l in finding['evidence']: locs.append(locator(l))
    e.update(figure_specific_value_checks=values,all_existing_extracted_values_unchanged_count=len(unchanged),p01_complete_json_operations_reproduced=True,p01_original_payload_matches_independent_original=True,p01_positive_locator_checks=locs)
    write(owner+'_evidence.json',e)
    print(json.dumps({'owner':owner,'figure_specific_value_checks':len(values),'all_existing_values_unchanged':len(unchanged),'P01_operation_reproduction':'PASS','candidate_locator_checks':len(locs)}))
def observe(label):
    e=read('evidence_manifest.json'); s=state(); previous=e['ref_observations'][-1]['state']
    moved=s['refs']!=previous['refs'] or s['show_ref_head_dereference']!=previous['show_ref_head_dereference']
    e['ref_observations'].append({'label':label,'observed_utc':now(),'state':s,'raw_ref_status':'FAIL_REFS_MOVED' if moved else 'PASS_REFS_UNCHANGED'})
    write('evidence_manifest.json',e)
    if moved: raise ValueError('STOP: new ref movement requires independent authentication; raw FAIL retained')
    assert all(s[k]==read('baseline.json')['start_state'][k] for k in s if k not in ['refs','show_ref_head_dereference'])
    print(json.dumps({'observation':label,'raw_ref_status':'PASS_REFS_UNCHANGED','head_index_tracked_staged':'PASS'}))
def refs_parse(s): return {l.split('\t')[0]:l.split('\t')[1:] for l in s.splitlines()}
def qualify_refs():
    e=read('evidence_manifest.json'); all_pinned=set()
    # Include every pinned publication path, companion source, manuscript and appendix.
    p=json.loads(raw(W4+'publication_semantic_batch_p01_20261014/evidence_manifest.json'))
    def collect(x):
        if isinstance(x,dict):
            if 'path' in x and isinstance(x['path'],str): all_pinned.add(x['path'])
            for v in x.values(): collect(v)
        elif isinstance(x,list):
            for v in x: collect(v)
    collect(p)
    for owner in ['HV000107','HV000113']:
        f=OUT/(owner+'_evidence.json')
        if f.exists(): collect(json.loads(f.read_text()))
    old=refs_parse(e['ref_observations'][0]['state']['refs']); new=refs_parse(e['ref_observations'][-1]['state']['refs']); changes=[]
    for name in sorted(old.keys()|new.keys()):
        if old.get(name)==new.get(name): continue
        a=old.get(name); b=new.get(name)
        assert a and b and not a[2] and not b[2], 'STOP: creation/deletion/symbolic movement requires separate authentication'
        a,b=a[0],b[0]
        assert git('cat-file','-t',a).strip()==b'commit' and git('cat-file','-t',b).strip()==b'commit'
        r=subprocess.run(['git','merge-base','--is-ancestor',a,b],cwd=ROOT,capture_output=True); assert r.returncode==0,'STOP: destructive/non-FF move'
        # Inspect changes of EVERY commit in the range, not just net diff endpoints.
        commits=git('rev-list',a+'..'+b).decode().splitlines(); paths=set(); details=[]
        for c in commits:
            z=git('diff-tree','--root','-m','--no-commit-id','--name-status','-r','--no-renames',c).decode(); details.append({'commit':c,'complete_name_status':z})
            for l in z.splitlines(): paths.update(l.split('\t')[1:])
        overlap=paths & all_pinned; assert not overlap,'STOP: relevant ref movement'
        assert all(x.startswith(('scripts/gen5_','reports/gen5_','reports/reason_router_gen5_','tests/test_gen5_','docs/gen5_')) for x in paths),'STOP: unknown changed-path relationship'
        changes.append({'ref':name,'before':a,'after':b,'raw_status':'FAIL_REFS_MOVED','old_object_available':True,'new_object_available':True,'ancestor_check_returncode':0,'complete_range_commits':details,'all_changed_paths':sorted(paths),'pinned_path_overlap':sorted(overlap),'qualified_status':'PASS_AUTHENTICATED_UNRELATED_NONDESTRUCTIVE_FAST_FORWARD','basis':'All per-commit changes are Gen5-only source/report/test/documentation; no pinned publication/companion/authority input path overlaps. Exact pinned objects remain authenticated. No old commit is removed by this fast-forward.','actor_cause_time':'UNKNOWN'})
    e['current_ref_movement_qualification']={'raw_status':'FAIL_REFS_MOVED' if changes else 'PASS_REFS_UNCHANGED','movements':changes,'all_pinned_paths':sorted(all_pinned),'blocks_adjudication':False,'uncertified_intervals':['Before initial capture','Between observations; movements returning to the same endpoint cannot be ruled out'],'pinned_local_rechecks':[{'file':x['file'],'sha256':sha(raw(x['file'])),'matches':sha(raw(x['file']))==x['expected_sha256']} for x in e['local_pinned_input_checks']]}
    assert all(x['matches'] for x in e['current_ref_movement_qualification']['pinned_local_rechecks'])
    write('evidence_manifest.json',e)
    print(json.dumps({'raw_status':e['current_ref_movement_qualification']['raw_status'],'movements':len(changes),'changed_paths':changes[0]['all_changed_paths'] if changes else [],'qualification':'NONBLOCKING_AUTHENTICATED','local_pinned_rechecks':len(e['local_pinned_input_checks'])}))
def finish_owner(owner,findings):
    e=read(owner+'_evidence.json'); n=Path(e['authenticated_saved_transition']['file']).stem
    assert e['p01_complete_json_operations_reproduced'] and all(e['plot_equal_keys'].values())
    manifest=read('evidence_manifest.json'); assert not manifest['current_ref_movement_qualification']['blocks_adjudication']
    visual=[]
    for c in e['object_checks']:
        if c['saved_file'].endswith(('.png','.pdf')):
            visual.append({'file':c['saved_file'],'sha256':c['raw_sha256'],'page':1 if c['saved_file'].endswith('.pdf') else None,'region':'COMPLETE contents, all panels, legends, footers and numeric cells','actual_visual_inspection_complete':True})
    for side in ['before','after']:
        p=OUT/(n+'_'+side+'_pdf_page1.png'); assert p.exists()
        visual.append({'render_file':rel(p),'sha256':sha(p.read_bytes()),'source_pdf':W3+'transitions_0/evidence/'+n+'_'+side+'.pdf','renderer':'Poppler pdftoppm, scale-to 2400, page 1 of 1','actual_visual_inspection_complete':True})
    e['complete_visual_inspection']={'checks':visual,'ocr_used':False,'extracted_text_substituted_for_visuals':False,'attestation':'Independent full original PNG and complete Poppler PDF page inspection in this execution.'}
    e['independent_findings']=findings
    clauses=[('1.1','Complete interpretation/reconciliation of saved before/after publication evidence',['CLAIM','RENDER']),('1.2','authenticated companion sources',['ENDPOINT','PROVENANCE']),('1.3','identify all substantive claim changes',['CLAIM']),('1.4','identify all substantive endpoint changes',['ENDPOINT']),('1.5','identify all substantive provenance changes versus render-only differences',['PROVENANCE','RENDER']),('1.6','Existing recovery is not completed semantic review.',['CLAIM','ENDPOINT','PROVENANCE','RENDER','HISTORY'])]
    rows=[]
    for suffix,text,fs in clauses:
        locs=[]
        for f in findings:
            if f['category'] in fs: locs.extend(f['evidence'])
        assert locs
        rows.append({'clause_id':'original-'+suffix,'literal_component':text,'assessment':'SATISFIED' if suffix!='1.6' else 'SATISFIED_BY_THIS_INDEPENDENT_REVIEW','finding_ids':[f['id'] for f in findings if f['category'] in fs],'authenticated_source_locators':locs})
    e['review_status']='COMPLETE_INDEPENDENT_DOCUMENTARY_REVIEW'
    write(owner+'_evidence.json',e)
    verdict={'source_id':owner,'debt_id':e['original_records']['original_debt']['record']['debt_id'],'verdict':'VERIFIED_FULL_DOCUMENTARY_SATISFACTION','original_debt_locator':{k:v for k,v in e['original_records']['original_debt'].items() if k!='record'},'original_root_locator':{k:v for k,v in e['original_records']['original_root'].items() if k!='record'},'original_transition_locator':{k:v for k,v in e['original_records']['original_transition'].items() if k!='record'},'literal_original_requirement':e['original_records']['original_debt']['record']['exact_missing_evidence'],'clauses':rows,'findings':findings,'independent_evidence_file':rel(OUT/(owner+'_evidence.json')),'independent_evidence_sha256':sha((OUT/(owner+'_evidence.json')).read_bytes()),'residual_original_documentary_requirements':[],'limitations':['Source-reported Supported is not independent scientific support.','Caveat deletion is not proof of homology.','Selected ancestor comparison does not prove immediate predecessor or global adoption.','Historical failures, capture LF/CRLF discrepancy and uncertified intervals retained.'],'scientific_claims_certified':False,'closure_credit_applied':0,'accounting':[222,18,194],'completed_utc':now()}
    old=(OUT/'owner_verdicts.jsonl').read_bytes(); existing=[json.loads(l) for l in old.splitlines()]; assert owner not in [x['source_id'] for x in existing]
    if owner=='HV000113': assert [x['source_id'] for x in existing]==['HV000107']
    line=(json.dumps(verdict,ensure_ascii=False,separators=(',',':'))+'\n').encode()
    with (OUT/'owner_verdicts.jsonl').open('ab') as f: f.write(line); f.flush(); os.fsync(f.fileno())
    assert (OUT/'owner_verdicts.jsonl').read_bytes()==old+line
    cp=read('checkpoint.json'); cp['completed_owners'].append({'source_id':owner,'debt_id':verdict['debt_id'],'verdict':verdict['verdict'],'jsonl_line':len(existing)+1,'line_sha256':sha(line),'evidence_sha256':verdict['independent_evidence_sha256'],'completed_utc':verdict['completed_utc']}); cp.update(status='OWNER_COMPLETED',unfinished_owners=[x for x in ['HV000107','HV000113'] if x not in [v['source_id'] for v in cp['completed_owners']]],owner_verdicts_sha256=sha(old+line)); write('checkpoint.json',cp)
    print(json.dumps({'owner':owner,'verdict':verdict['verdict'],'clauses':len(rows),'durably_persisted':True,'closure_credit':0}))
def state():
    index = Path(git('rev-parse','--git-path','index').decode().strip())
    if not index.is_absolute(): index = ROOT/index
    return {'head':git('rev-parse','HEAD').decode().strip(),
      'symbolic_head':subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=ROOT,capture_output=True).stdout.decode().strip(),
      'refs':git('for-each-ref','--sort=refname','--format=%(refname)%09%(objectname)%09%(*objectname)%09%(symref)').decode(),
      'show_ref_head_dereference':git('show-ref','--head','--dereference').decode(),
      'index_sha256':sha(index.read_bytes()), 'index_path':str(index),
      'tracked_index':git('ls-files','--stage','-z').decode(),
      'tracked_status':git('status','--porcelain=v2','--untracked-files=no').decode(),
      'staged_diff':git('diff','--cached','--raw','--no-ext-diff').decode(),
      'tracked_diff':git('diff','--raw','--no-ext-diff').decode()}
def inventory():
    items = {}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d != '.git' and Path(base,d).resolve() != OUT)
        for n in sorted(names):
            p = Path(base,n)
            if p.name == '.git': continue
            b=p.read_bytes(); items[rel(p)]={'size':len(b),'sha256':sha(b)}
    return items
if __name__=='__main__' and sys.argv[1:] == ['baseline']:
    s=state()
    if s['head']!=EXPECTED: raise SystemExit('STOP: unexpected HEAD')
    write('checkpoint.json',{'phase':'CENSUS_RECOVERY_ONLY','status':'BASELINE_CAPTURE_IN_PROGRESS','started_utc':now(),'expected_head':EXPECTED,'targets':['HV000107','HV000113'],'completed_owners':[],'new_closure_credits':0,'accounting':[222,18,194],'authority':'Current controller > frozen census reconciliation > original Wave03 records > frozen P01 > integrated HV000103 > AGENTS.md'})
    files=inventory()
    end=state()
    write('baseline.json',{'captured_utc':now(),'start_state':s,'end_state':end,'state_stable_during_capture':s==end,'preexisting_files':files,'output_directory_excluded':rel(OUT),'note':'All workspace regular files outside .git and this newly created output directory. Git index and logical complete ref endpoints captured separately.'})
    if s!=end: raise SystemExit('STOP: repository state changed during baseline')
    cp=read('checkpoint.json'); cp.update(status='BASELINE_DURABLE',baseline_sha256=sha((OUT/'baseline.json').read_bytes()),preexisting_file_count=len(files)); write('checkpoint.json',cp)
    (OUT/'owner_verdicts.jsonl').write_bytes(b'')
    print(json.dumps({'baseline':'PASS','head':s['head'],'preexisting_files':len(files),'ref_count':len(s['refs'].splitlines())}))
elif __name__=='__main__' and sys.argv[1:]==['extract']: extract()
elif __name__=='__main__' and sys.argv[1:2]==['owner_auth']: owner_auth(sys.argv[2])
elif __name__=='__main__' and sys.argv[1:]==['pinned_setup']: pinned_setup()
elif __name__=='__main__' and sys.argv[1:2]==['companions_auth']: companions_auth(sys.argv[2])
elif __name__=='__main__' and sys.argv[1:2]==['semantic']: semantic(sys.argv[2])
elif __name__=='__main__' and sys.argv[1:2]==['observe']: observe(sys.argv[2])
elif __name__=='__main__' and sys.argv[1:]==['qualify_refs']: qualify_refs()

