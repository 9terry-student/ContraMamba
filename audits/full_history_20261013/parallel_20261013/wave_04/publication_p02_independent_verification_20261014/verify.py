import csv, datetime, difflib, hashlib, io, json, os, pathlib, subprocess, sys
from pypdf import PdfReader
from PIL import Image

ROOT = pathlib.Path('C:/cm-full-audit-20261013')
OUT = pathlib.Path(__file__).resolve().parent
W4 = ROOT / 'audits/full_history_20261013/parallel_20261013/wave_04'
P02 = W4 / 'publication_semantic_batch_p02_20261014'
HEAD = 'ff806bd5bd8efc62fcf6420cf8b097a63ed70571'
OWNERS = {'HV000114':'DEBT-e90f0a74043687c39794', 'HV000117':'DEBT-58c0e634b5f5ecea13e5', 'HV000121':'DEBT-44e964f5e0e98e4986cb'}
COMMANDS = []
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def meta(p):
    p=pathlib.Path(p); b=p.read_bytes(); return {'sha256':sha(b),'bytes':len(b)}
def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in pathlib.Path(p).read_text(encoding='utf-8').splitlines()]
def write(name,j): (OUT/name).write_text(json.dumps(j,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
def git(*args, allowed=(0,)):
    argv=['git',*args]; env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0';r=subprocess.run(argv,cwd=ROOT,capture_output=True,env=env)
    COMMANDS.append({'argv':argv,'exit_code':r.returncode,'stdout_sha256':sha(r.stdout),'stderr':r.stderr.decode('utf-8','replace')})
    assert r.returncode in allowed, (argv,r.returncode,r.stderr)
    return r.stdout
def state():
    s={k:git(*a).decode('utf-8','replace') for k,a in {
      'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],
      'refs':['show-ref','--head'],'worktrees':['worktree','list','--porcelain'],
      'status':['status','--porcelain=v1','--untracked-files=normal'],
      'index_entries':['ls-files','--stage'],'tracked_diff':['diff','--binary'],
      'staged_diff':['diff','--cached','--binary']}.items()}
    s['head']=s['head'].strip();s['branch']=s['branch'].strip()
    ip=pathlib.Path(git('rev-parse','--git-path','index').decode().strip())
    if not ip.is_absolute():ip=ROOT/ip
    s['index_path']=str(ip.resolve());s['index']=meta(ip);s['captured_utc']=utc()
    assert s['head']==HEAD,'AUDIT_HEAD_MISMATCH'
    return s
def inventory():
    names=set(git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0'))-{''}
    names |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'audits').rglob('*') if p.is_file()}
    return {n:meta(ROOT/n) for n in sorted(names) if not (ROOT/n).is_relative_to(OUT) and (ROOT/n).is_file()}
def ptr(j,p):
    for k in p.strip('/').split('/') if p else []:
        k=k.replace('~1','/').replace('~0','~'); j=j[int(k)] if isinstance(j,list) else j[k]
    return j
def locator(loc):
    p=ROOT/loc['file']; b=p.read_bytes(); actual=meta(p)
    expected=loc.get('file_sha256',loc.get('sha256'))
    if expected: assert actual['sha256']==expected,('FILE_HASH',loc)
    if 'bytes' in loc:assert actual['bytes']==loc['bytes']
    result={'locator':loc,'actual':actual}
    if 'jsonl_line' in loc:
        line=b.splitlines()[loc['jsonl_line']-1]; expected=loc.get('line_sha256_excluding_lf',loc.get('line_sha256'))
        if expected:assert sha(line)==expected,('LINE_HASH',loc)
        result['line_sha256_excluding_lf']=sha(line);result['payload']=json.loads(line)
    elif 'json_pointer' in loc:result['payload']=ptr(json.loads(b),loc['json_pointer'])
    return result
def binding(x):
    assert git('cat-file','-t',x['commit']).strip()==b'commit'
    blob=git('rev-parse',x['commit']+':'+x['path']).decode().strip()
    assert blob==x['blob'],('BLOB_ID',x)
    b=git('cat-file','blob',blob)
    assert sha(b)==x['sha256'],('BLOB_BYTES',x)
    if 'bytes' in x:assert len(b)==x['bytes']
    if 'saved_file' in x:assert (ROOT/x['saved_file']).read_bytes()==b,('SAVED_BYTES',x)
    return {**{k:v for k,v in x.items() if k in ['side','role','commit','path','blob','sha256','saved_file']},'bytes':len(b),'status':'AUTHENTICATED'},b
def diff(a,b,p=''):
    out=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            q=p+'/'+k.replace('~','~0').replace('/','~1')
            if k not in a:out.append({'operation':'add','pointer':q,'after':b[k]})
            elif k not in b:out.append({'operation':'remove','pointer':q,'before':a[k]})
            else:out.extend(diff(a[k],b[k],q))
    elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        for i,(av,bv) in enumerate(zip(a,b)):out.extend(diff(av,bv,p+'/'+str(i)))
    elif a!=b:out.append({'operation':'replace','pointer':p,'before':a,'after':b})
    return out
def hunks(a,b):
    a=a.splitlines();b=b.splitlines();out=[]
    for op,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if op!='equal':out.append({'operation':op,'before_line_start':i+1,'before_line_end':j,'after_line_start':k+1,'after_line_end':l,'before':a[i:j],'after':b[k:l]})
    return out
def start():
    assert not (OUT/'checkpoint.json').exists(),'EXISTING_OUTPUT_STOP'
    s=state(); inv=inventory()
    c={'schema':'P02_INDEPENDENT_CHECKPOINT_V1','phase':'CENSUS_RECOVERY_ONLY','status':'IN_PROGRESS','authority':['current controller instruction','frozen census_exit_reconciliation_20261014','original Wave03 debt/root/transition','frozen P02 packet','P02 movement review and fixed Git diff','verified P01 overlays','AGENTS.md'], 'started_utc':utc(),'expected_head':HEAD,'authorized_owners':OWNERS,'initial_state':s,'baseline_files':inv,'completed_owners':[],'new_applied_credit':0,'existing_accounting':[222,20,192],'frozen_accounting':[222,17,195],'commands':COMMANDS}
    write('checkpoint.json',c)
    (OUT/'owner_verdicts.jsonl').write_bytes(b'')
    print('Initial baseline captured:',len(inv),'files; expected detached HEAD; zero tracked/staged diff.')
def packet():
    c=read(P02/'checkpoint.json');v=read(P02/'validation.json'); cp=read(OUT/'checkpoint.json')
    checks=[]
    for n,h in v['output_sha256_excluding_self'].items():
        m=meta(P02/n);assert m['sha256']==h;checks.append({'file':str((P02/n).relative_to(ROOT)),'declared_sha256':h,**m})
    checks.append({'file':str((P02/'validation.json').relative_to(ROOT)),**meta(P02/'validation.json'),'authentication':'baseline-bound bytes; no circular self-hash or independent signature'})
    mv=read(W4/'publication_p02_movement_verification_20261014/checkpoint.json')
    assert meta(P02/'validation.json')['sha256']==mv['original_p02_files']['validation.json']['sha256']
    for p,m in c['baseline_files'].items():assert meta(ROOT/p)['sha256']==m,('ORIGINAL_BASELINE',p)
    pins=[]
    for p,m in c['pinned_inputs'].items():
        assert meta(ROOT/p)==m,('PINNED_INPUT',p);pins.append({'file':p,**m})
    unique={ (x['commit'],x['path'],x['blob'],x['sha256']):x for x in c['git_source_checks'] }
    binds=[binding(x)[0] for x in unique.values()]
    original_rows=rows(P02/'documentary_assessments.jsonl');raw=(P02/'documentary_assessments.jsonl').read_bytes().splitlines()
    assert {r['source_id']:r['debt_id'] for r in original_rows}==OWNERS
    for x in c['completed_assessments']:assert sha(raw[x['line']-1])==x['line_sha256']
    authorities=[]
    for d in ['census_exit_reconciliation_20261014','publication_hv000103_closure_integration_20261014','publication_p01_two_id_closure_integration_20261014','publication_p02_movement_verification_20261014']:
        av=read(W4/d/'validation.json')
        for n,h in av['output_sha256_excluding_self'].items():
            assert meta(W4/d/n)['sha256']==h;authorities.append({'file':str((W4/d/n).relative_to(ROOT)),'declared_sha256':h,**meta(W4/d/n)})
    failures=[]
    for f in c['historical_failures_retained']:
        q=locator(f.get('evidence',f));assert q['payload']==f.get('status',f.get('raw_status'));failures.append(f)
    debts=rows(ROOT/'audits/full_history_20261013/parallel_20261013/wave_03/study_consolidation/unresolved_census_debt.jsonl')
    rules=read(W4/'census_exit_reconciliation_20261014/evidence_input_manifest.json')['accounting_rules']
    frozen=set(rules['fully_satisfied_ids']);credits=[]
    for d in ['publication_hv000103_closure_integration_20261014','publication_p01_two_id_closure_integration_20261014']:
        credits.extend(rows(W4/d/'closure_credit_overlay.jsonl'))
    credit_ids={x['debt_id'] for x in credits};current=frozen|credit_ids
    assert len(debts)==222 and len(frozen)==17 and len(current)==20 and len(credit_ids)==3
    assert sum(x['blocks_strict_census_completeness'] and x['debt_id'] not in frozen for x in debts)==195
    assert sum(x['blocks_strict_census_completeness'] and x['debt_id'] not in current for x in debts)==192
    assert not (set(OWNERS.values())&current)
    result={'schema':'P02_INDEPENDENT_EVIDENCE_V1','phase':'CENSUS_RECOVERY_ONLY','packet_output_checks':checks,'original_P02_status':v['status'],'original_P02_raw_ref_status':v['raw_ref_status'],'original_P02_raw_worktree_status':v['raw_worktree_status'],'original_P02_baseline_count':len(c['baseline_files']),'pinned_inputs':pins,'git_binding_checks':binds,'authority_hash_checks':authorities,'inherited_failures_retained':failures,'original_raw_movements':{'P02_refs':v['raw_ref_movements'],'P02_worktrees':v['raw_worktree_movements'],'movement_review_refs':mv['additional_observed_ref_movements'],'movement_review_worktrees':mv['additional_observed_worktree_movements']},'accounting':{'frozen':[222,17,195],'existing':[222,20,192],'frozen_satisfied_ids':sorted(frozen),'existing_satisfied_ids':sorted(current),'previous_credit_rows':credits,'new_applied_credit':0},'owners':{},'commands':COMMANDS}
    write('evidence_manifest.json',result)
    print('Packet authenticated: five files,',len(pins),'local inputs,',len(binds),'unique Git bindings;',len(failures),'historical failures retained.')
def owner(owner_id):
    r=next(x for x in rows(P02/'documentary_assessments.jsonl') if x['source_id']==owner_id)
    rm=next(x for x in rows(P02/'readiness_matrix.jsonl') if x['source_id']==owner_id)
    c=read(OUT/'checkpoint.json');assert owner_id not in [x['source_id'] for x in c['completed_owners']]
    expected=list(OWNERS);assert owner_id==expected[len(c['completed_owners'])],'OWNER_ORDER'
    locs={k:locator(rm[k]) for k in ['original_locator','root_locator','transition_locator','saved_transition_locator','frozen_locator']}
    debt=locs['original_locator']['payload'];root=locs['root_locator']['payload'];tr=locs['transition_locator']['payload']
    assert debt==r['original_obligation']==rm['literal_original_record']
    assert debt['debt_id']==OWNERS[owner_id] and debt['stable_original_source_ids']==[owner_id]
    assert root==rm['root_record'] and tr==rm['transition_record']
    assert root['dependent_original_debt_ids']==[debt['debt_id']]
    assert root['exact_missing_evidence'][0]['source_specific_requirements']==[]
    bs=[];contents={}
    for x in r['source_identity_checks']+r['source_binding_checks']+r['manuscript_reference_bindings']:
        check,b=binding(x);bs.append(check);contents[(x.get('side','manuscript'),x['path'])]=b
    before=next(x for x in r['source_identity_checks'] if x['role']=='publication' and x['side']=='before')
    after=next(x for x in r['source_identity_checks'] if x['role']=='publication' and x['side']=='after')
    anc=git('merge-base','--is-ancestor',before['commit'],after['commit'],allowed=(0,1))
    assert COMMANDS[-1]['exit_code']==0
    parents=git('show','-s','--format=%P',after['commit']).decode().split()
    parent_bindings=[{'commit':p,'blob':git('rev-parse',p+':'+after['path']).decode().strip()} for p in parents]
    assert any(x['blob']==before['blob'] for x in parent_bindings)
    get=lambda side,end: next(b for (s,p),b in contents.items() if s==side and p.endswith(end))
    a=json.loads(get('before','figure_data_manifest.json'));b=json.loads(get('after','figure_data_manifest.json'))
    ops=diff(a,b);hs=hunks(get('before','README.md').decode(),get('after','README.md').decode())
    assert ops==r['complete_companion_comparison']['operations'],'OPERATIONS_DIFFER'
    assert hs==r['complete_companion_comparison']['README_hunks'],'HUNKS_DIFFER'
    assert len(ops)==(84 if owner_id=='HV000114' else 40)
    assert len(hs)==(18 if owner_id=='HV000114' else 3)
    # Entire old-value inventory and old plot fields, not just selected assertions.
    ev_checks=[{'key':k,'before':v['value'],'after':b['extracted_values'][k]['value'],'equal':v['value']==b['extracted_values'][k]['value']} for k,v in a['extracted_values'].items()]
    assert all(x['equal'] for x in ev_checks)
    old_plots=[{'key':k,'equal':v==b['plot_data'][k]} for k,v in a['plot_data'].items()]
    assert all(x['equal'] for x in old_plots)
    pdfs=[];pngs=[]
    for x in r['source_identity_checks']:
        if x['path'].endswith('.pdf'):
            reader=PdfReader(ROOT/x['saved_file']);pdfs.append({'side':x['side'],'saved_file':x['saved_file'],'sha256':x['sha256'],'pages':[{'page':i+1,'mediabox':list(map(float,p.mediabox)),'text':p.extract_text(extraction_mode='layout')} for i,p in enumerate(reader.pages)]})
        if x['path'].endswith('.png'):
            im=Image.open(ROOT/x['saved_file']);im.load();pngs.append({'side':x['side'],'saved_file':x['saved_file'],'sha256':x['sha256'],'size':list(im.size),'mode':im.mode})
    manuscripts=[]
    for x in r['manuscript_reference_bindings']:
        bsrc=contents[('manuscript',x['path'])];manuscripts.append({'commit':x['commit'],'path':x['path'],'sha256':sha(bsrc),'numbered_text':bsrc.decode().splitlines()})
    result={'source_id':owner_id,'debt_id':debt['debt_id'],'original_records':locs,'source_bindings':bs,'chronology':{'before_commit':before['commit'],'after_commit':after['commit'],'ancestor_exit_code':0,'after_parents':parent_bindings,'immediate_parent_preserves_before_blob':True,'selected_before_commit_need_not_be_immediate_parent':True},'manifest_operations':ops,'README_hunks':hs,'all_existing_extracted_values':ev_checks,'old_plot_data_checks':old_plots,'complete_pdf_content':pdfs,'png_metadata':pngs,'manuscripts':manuscripts,'independently_recomputed':True,'semantic_review_status':'IN_PROGRESS'}
    write(owner_id+'_evidence.json',result)
    m=read(OUT/'evidence_manifest.json');m['owners'][owner_id]={'evidence_file':owner_id+'_evidence.json','evidence_sha256':meta(OUT/(owner_id+'_evidence.json'))['sha256'],'source_binding_count':len(bs)};m['commands'].extend(COMMANDS);write('evidence_manifest.json',m)
    print(owner_id+': original debt/root/transition, saved bytes, '+str(len(bs))+' bindings, '+str(len(ops))+' operations, '+str(len(hs))+' hunks authenticated and recomputed.')
if __name__=='__main__':
    {'start':start,'packet':packet,'owner':lambda:owner(sys.argv[2])}[sys.argv[1]]()
