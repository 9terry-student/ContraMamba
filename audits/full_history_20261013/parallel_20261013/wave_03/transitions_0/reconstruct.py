"""Read-only Git source reconstruction; writes only beside this helper."""
import json, pathlib, subprocess, hashlib, difflib, sys, io, marshal, types, dis
sys.path.append('C:/Users/Home1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
ROOT=pathlib.Path(__file__).resolve().parents[5]
BASE=ROOT/'audits/full_history_20261013/parallel_20261013'
OUT=pathlib.Path(__file__).resolve().parent
def read(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT)
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p,rs): p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs),encoding='utf-8')
def endpoint(c,p,b):
    typ=git('cat-file','-t',c).decode().strip()
    assert typ in ['commit','tree'],(c,typ)
    actual=git('rev-parse',c+':'+p).decode().strip(); assert actual==b,(c,p,b,actual)
    raw=git('cat-file','blob',b)
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==b
    return dict(commit=c,path=p,blob=b,object_type=typ,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),raw
def project(raw,path):
    if path.endswith('.pdf'):
        from pypdf import PdfReader
        r=PdfReader(io.BytesIO(raw)); return '\n'.join(f'PAGE {i+1}\n'+(p.extract_text() or '') for i,p in enumerate(r.pages))
    if path.endswith('.png'): return 'PNG: visual comparison required'
    if path.endswith('.pyc'):
        code=marshal.loads(raw[16:]); assert isinstance(code,types.CodeType)
        def walk(c,qual=''):
            q=qual+'/'+c.co_name
            yield q+' vars='+repr(c.co_varnames)+' names='+repr(c.co_names)+'\n'
            yield 'constants='+repr([x for x in c.co_consts if not isinstance(x,types.CodeType)])+'\n'
            yield dis.Bytecode(c).dis()+'\n'
            for s in c.co_consts:
                if isinstance(s,types.CodeType): yield from walk(s,q)
        return 'header='+raw[:16].hex()+'\n'+''.join(walk(code))
    if path.endswith('.ipynb'):
        j=json.loads(raw);return '\n'.join('CELL '+str(i)+' '+c['cell_type']+'\n'+''.join(c.get('source',[]))+'\nOUTPUTS '+json.dumps(c.get('outputs',[]),ensure_ascii=False) for i,c in enumerate(j['cells']))
    return raw.decode('utf-8-sig')
def prepare():
    qpath=BASE/'wave_02/integration_02/remaining_census_work_queue.jsonl'
    queue=read(qpath); population=sorted([r for r in queue if r['category']=='HISTORICAL_TRANSITION_INTERPRETATION'],key=lambda r:r['original_queue_id'])
    assert len(population)==216 and len({r['original_queue_id'] for r in population})==216
    assert all(r['original_queue_id']==r['id']==r['original_queue_record']['id'] for r in population)
    manifest=[dict(population_index=i,shard=0,original_queue_id=r['original_queue_id'],original_source_ids=r['original_source_ids'],original_source_identities=r['original_source_identities'],integrated_queue_record=r) for i,r in enumerate(population) if i%2==0]
    assert len(manifest)==108
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'evidence').mkdir(exist_ok=True)
    lines(OUT/'target_manifest.jsonl',manifest)
    rev=read(BASE/'wave_02/transitions_shard_1/transition_review.jsonl')
    old=[]
    for shard in [0,1]:
        for n,r in enumerate(read(BASE/f'versions_shard_{shard}/semantic_revision_records.jsonl'),1): old.append((f'versions_shard_{shard}/semantic_revision_records.jsonl',n,r))
    for k,m in enumerate(manifest):
        q=m['integrated_queue_record']; ident=q['original_source_identities'][0]; p=ident['path']
        prior=next((r for r in rev if q['id'] in r['queue_ids']),None)
        refs=[]
        if prior:
            bef=prior['before']; aft=prior['after']; refs=[dict(audit_file='wave_02/transitions_shard_1/transition_review.jsonl',transition_id=prior['transition_id'])]
        else:
            matches=[(f,n,r) for f,n,r in old if (r.get('to_id') in q['original_source_ids'] or set(r.get('selected_target_ids',[]))&set(q['original_source_ids'])) and r['to_blob']==ident['blob']]
            assert len(matches)==1,(q['id'],len(matches))
            f,n,r=matches[0];refs=[dict(audit_file=f,jsonl_line=n,record=r)]
            bef=dict(commit=r.get('from_commit',r.get('before_commit')),path=p,blob=r['from_blob'])
            aft=dict(commit=r.get('to_commit',r.get('after_commit')),path=p,blob=r['to_blob'])
        origin,raw=endpoint(ident['commit'],p,ident['blob']); before,braw=endpoint(bef['commit'],p,bef['blob']);after,araw=endpoint(aft['commit'],p,aft['blob'])
        assert origin['blob']==after['blob']
        assert subprocess.run(['git','merge-base','--is-ancestor',before['commit'],after['commit']],cwd=ROOT).returncode==0
        parents=git('show','-s','--format=%P',after['commit']).decode().split()
        parent_blobs=[]
        for par in parents:
            v=subprocess.run(['git','rev-parse',par+':'+p],cwd=ROOT,capture_output=True)
            parent_blobs.append(dict(commit=par,blob=v.stdout.decode().strip() if v.returncode==0 else None))
        bt=project(braw,p);at=project(araw,p)
        (OUT/'evidence'/f'{k:03}_before.txt').write_text(bt,encoding='utf-8')
        (OUT/'evidence'/f'{k:03}_after.txt').write_text(at,encoding='utf-8')
        if p.endswith(('.pdf','.png')):
            (OUT/'evidence'/f'{k:03}_before{pathlib.Path(p).suffix}').write_bytes(braw)
            (OUT/'evidence'/f'{k:03}_after{pathlib.Path(p).suffix}').write_bytes(araw)
        changes=[]
        bl=bt.splitlines();al=at.splitlines()
        for op,a,b,c,d in difflib.SequenceMatcher(None,bl,al,autojunk=False).get_opcodes():
            if op!='equal':changes.append(dict(operation=op,before_start=a+1,before_end_exclusive=b+1,after_start=c+1,after_end_exclusive=d+1,before=bl[a:b],after=al[c:d],before_context=bl[max(0,a-3):a]+bl[b:b+3],after_context=al[max(0,c-3):c]+al[d:d+3]))
        ev=dict(shard_ordinal=k,original_queue_id=q['id'],original_source_ids=q['original_source_ids'],original_source_identity=origin,before=before,after=after,prior_evidence=refs,prior_interpretation=prior.get('historical_interpretation') if prior else None,relationship='ANCESTOR_TO_DESCENDANT_VERIFIED',after_parents=parent_blobs,immediate_parent_has_before_blob=any(s['blob']==before['blob'] for s in parent_blobs),comparison_projection='disassembled_bytecode' if p.endswith('.pyc') else 'extracted_pdf_text' if p.endswith('.pdf') else 'image' if p.endswith('.png') else 'notebook_sources_and_outputs' if p.endswith('.ipynb') else 'source_lines',changes=changes)
        dump(OUT/'evidence'/f'{k:03}.json',ev)
    dump(OUT/'partition_validation.json',dict(queue_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),queue_rows=len(queue),population_count=len(population),selected_count=len(manifest),sorted_population_ids=[r['id'] for r in population],selected_ids=[r['original_queue_id'] for r in manifest],source_identity_checks='108 original and 216 transition endpoints authenticated',head=git('rev-parse','HEAD').decode().strip()))
def show(args):
    for k in map(int,args):
        e=json.loads((OUT/'evidence'/f'{k:03}.json').read_text(encoding='utf-8'))
        print('\nTARGET',k,e['original_queue_id'],e['after']['path'],e['after']['commit'],e['before']['commit'])
        for h in e['changes']:
            print('HUNK',h['before_start'],h['after_start']);print('\n'.join('-'+s for s in h['before']));print('\n'.join('+'+s for s in h['after']))
if __name__=='__main__':
    if sys.argv[1]=='prepare': prepare()
    elif sys.argv[1]=='show':show(sys.argv[2:])
