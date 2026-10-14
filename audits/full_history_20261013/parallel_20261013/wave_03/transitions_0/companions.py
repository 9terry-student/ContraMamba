import reconstruct as r
import json,sys,pathlib,difflib,marshal,types,dis,io,re
def save_companion(k,c,p,side):
    check=r.subprocess.run(['git','rev-parse',c+':'+p],cwd=r.ROOT,capture_output=True)
    if check.returncode:return None
    b=check.stdout.decode().strip(); e,raw=r.endpoint(c,p,b)
    suffix=pathlib.Path(p).suffix
    dest=r.OUT/'evidence'/f'{k:03}_{side}_companion_{pathlib.Path(p).name}'
    dest.write_bytes(raw)
    return dict(identity=e,evidence_file=str(dest.relative_to(r.OUT)),text_projection=r.project(raw,p) if suffix!='.png' else None)
def code_inventory(raw):
    top=marshal.loads(raw[16:]);out={}
    def walk(c,q):
        ops=list(dis.get_instructions(c));offsets={v.offset:i for i,v in enumerate(ops)}
        ins=[]
        for v in ops:
            a=v.argval
            if isinstance(a,types.CodeType):a='CODE:'+a.co_name
            elif v.opcode in dis.hasjabs or v.opcode in dis.hasjrel:a='JUMP_INDEX:'+str(offsets.get(a,a))
            elif not isinstance(a,(str,float,int,bool,type(None),list,tuple)):a=repr(a)
            ins.append([v.opname,a])
        out[q]=dict(varnames=c.co_varnames,names=c.co_names,constants=[repr(x) for x in c.co_consts if not isinstance(x,types.CodeType)],instructions=ins)
        for x in c.co_consts:
            if isinstance(x,types.CodeType):walk(x,q+'/'+x.co_name)
    walk(top,'module');return out
def prepare():
    for k in range(108):
        e=json.loads((r.OUT/'evidence'/f'{k:03}.json').read_text(encoding='utf-8'));p=e['after']['path'];cs=[]
        if p.endswith(('.png','.pdf')):
            paths=['paper/iclr2027/main.tex','paper/iclr2027/appendix.tex','paper/iclr2027/figures/figure_data_manifest.json','paper/iclr2027/figures/README.md'] if p.endswith('main.pdf') else [p[:-4]+'.pdf' if p.endswith('.png') else p[:-4]+'.png','paper/iclr2027/figures/figure_data_manifest.json','paper/iclr2027/figures/README.md']
        elif p.endswith('.pyc'):
            paths=[p.split('/__pycache__/')[0]+'/'+p.split('/__pycache__/')[1].split('.cpython-')[0]+'.py']
        else:continue
        for path in paths:
            pair={}
            for side in ['before','after']:
                v=save_companion(k,e[side]['commit'],path,side)
                if v: pair[side]=v
            if pair:
                pair['path']=path
                if pair.get('before',{}).get('text_projection') and pair.get('after',{}).get('text_projection'):
                    pair['diff']=list(difflib.unified_diff(pair['before']['text_projection'].splitlines(),pair['after']['text_projection'].splitlines(),n=2))
                cs.append(pair)
        if p.endswith('.pyc'):
            bi=code_inventory(r.git('cat-file','blob',e['before']['blob']));ai=code_inventory(r.git('cat-file','blob',e['after']['blob']))
            e['canonical_bytecode_changes']=[dict(code_path=q,before=bi.get(q),after=ai.get(q)) for q in sorted(set(bi)|set(ai)) if bi.get(q)!=ai.get(q)]
            r.dump(r.OUT/'evidence'/f'{k:03}.json',e)
        r.dump(r.OUT/'evidence'/f'{k:03}_companions.json',cs)
def show(args):
    for k in map(int,args):
        print('\nCOMPANIONS TARGET',k)
        cs=json.loads((r.OUT/'evidence'/f'{k:03}_companions.json').read_text(encoding='utf-8'))
        for c in cs:
            print('PATH',c['path'])
            if c['path'].endswith('figure_data_manifest.json'):continue
            print('\n'.join(c.get('diff',[])))
        e=json.loads((r.OUT/'evidence'/f'{k:03}.json').read_text(encoding='utf-8'))
        for c in e.get('canonical_bytecode_changes',[]):
            print('CODE',c['code_path'])
            for side in ['before','after']:
                v=c.get(side)
                print(side, {x:v[x] for x in ['varnames','names','constants']} if v else None)
            if c.get('before') and c.get('after'):
                print('\n'.join(difflib.unified_diff([repr(x) for x in c['before']['instructions']],[repr(x) for x in c['after']['instructions']],n=1)))
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:show(sys.argv[2:])
