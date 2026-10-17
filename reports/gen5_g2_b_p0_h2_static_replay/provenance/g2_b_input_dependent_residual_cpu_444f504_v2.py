#!/usr/bin/env python3
"""ContraMamba G2-B frozen P0->H2 CPU-only, pair-excluded exploratory replay.

Read-only. No torch, GPU, weights, tokenizer, new training/evaluation, or repo writes.
Fit ONLY P0 7-cell logits on other pairs; use H2 solely for prediction.
The quadratic comparator is predeclared for a single bounded diagnostic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

HEAD='444f504683c1f2284f9cbc52e59de1d4426cc8cd'
P0_HEAD='75ad31e42e1baed7d99afbe9281e36276dfd1992'
H2_HEAD='3c9fc396acb2621c001432ab5b6a2bc748a25a94'
P0_ROOT='reports/reason_router_gen5_g2_functional_science_runs/gen5-g2-p0-full-75ad31e-r1'
H2_ROOT='reports/gen5_g2_h2_full_matched_verb_observation_runs/g2h2-full-3c9fc39-r1'
P0_WORKER_SHA=('d0432108dfba731bc0eaf78854faa043842b5d9cebfa6183c867b6d39fd330de',
               'dd79c803dee3b5fe96822ec9d8074e24abbf6624e35bb93988571e7d55fcabc4')
H2_SUMMARY_SHA='6aa730522416e688d3105dbdaccd9d04695726bf78ecfebe766e68671ebe8491'
SEEDS=(6201,6202,6203)
ARMS=('ZERO','FULL')
CONDS=('NEG_DENIES','POS_CONFIRMS','POS_ASSERTS')
COMP={'NC':('NEG_DENIES','POS_CONFIRMS'), 'NA':('NEG_DENIES','POS_ASSERTS'), 'CA':('POS_CONFIRMS','POS_ASSERTS')}
CELLS=tuple('C'+str(i)+'_'+suffix for i,suffix in enumerate(('SHAM','', '', '', '', '', 'EXPLICIT_DENIAL')))
# 2D logit coordinates. Mapping cannot use condition metadata.
def coord(z):
    need(len(z)==3 and all(type(t) in (int,float) and math.isfinite(t) for t in z),'LOGIT_INVALID')
    return (float(z[0])-float(z[2]), float(z[0])-float(z[1])) # (RS,RN)

def need(v, msg):
    if not v: raise RuntimeError('G2_B_BLOCKED:'+msg)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def read(path, expected_sha=None):
    need(path.is_file(),'MISSING:'+str(path))
    if expected_sha: need(sha(path)==expected_sha,'SHA:'+str(path))
    return json.loads(path.read_text(encoding='utf-8'))

def mean(v):return sum(v)/len(v)
def rms(v):return math.sqrt(mean([x*x for x in v]))
def rmse2(pairs):
    return math.sqrt(sum((a-b)**2+(c-d)**2 for (a,c),(b,d) in pairs)/(2*len(pairs)))

def solve(a,b):
    # Unpivoted is inappropriate for correlated features. Partial-pivot Gauss-Jordan.
    n=len(b); A=[list(a[i])+[b[i]] for i in range(n)]
    for j in range(n):
        k=max(range(j,n),key=lambda i: abs(A[i][j]))
        need(abs(A[k][j])>1e-11,'SINGULAR_REGRESSION')
        A[k],A[j]=A[j],A[k]
        den=A[j][j]
        A[j]=[v/den for v in A[j]]
        for i in range(n):
            if i==j:continue
            m=A[i][j]
            A[i]=[x-m*y for x,y in zip(A[i],A[j])]
    return [A[i][-1] for i in range(n)]

def fit(xy, kind):
    need(len(xy)>=100 and kind in ('M0','M3','M4','M5'),'MODEL_KIND')
    xs=[x for x,y in xy];ys=[y for x,y in xy]
    mu=[mean([x[j] for x in xs]) for j in (0,1)]
    sd=[math.sqrt(mean([(x[j]-mu[j])**2 for x in xs])) for j in (0,1)]
    need(min(sd)>1e-9,'ZERO_VARIANCE_INPUT')
    ymu=[mean([y[j] for y in ys]) for j in (0,1)]
    ysd=[math.sqrt(mean([(y[j]-ymu[j])**2 for y in ys])) for j in (0,1)]
    need(min(ysd)>1e-10,'ZERO_VARIANCE_OUTPUT')
    def phi(x,axis):
        u=(x[0]-mu[0])/sd[0];v=(x[1]-mu[1])/sd[1]
        if kind=='M0':return [1.]
        if kind=='M3':return [1., (u,v)[axis]]
        if kind=='M4':return [1.,u,v]
        return [1.,u,v,u*u,u*v,v*v]  # M5: one fixed quadratic basis
    coeff=[]
    for axis in (0,1):
        features=[phi(x,axis) for x in xs]
        d=len(features[0]);a=[[0.]*d for _ in range(d)];b=[0.]*d
        for v,y in zip(features,ys):
            target=(y[axis]-ymu[axis])/ysd[axis]
            for i in range(d):
                b[i]+=v[i]*target
                for j in range(d):a[i][j]+=v[i]*v[j]
        for i in range(d):
            # For M5 only, fixed ridge=0.01 normalized SSE (not H2 tuned).
            if kind=='M5' and i>0:a[i][i]+=0.01*len(xs)
            elif kind!='M5' and i>0:a[i][i]+=1e-10*len(xs)
        coeff.append(solve(a,b))
    def predict(x):
        return tuple(ymu[j]+ysd[j]*sum(c*v for c,v in zip(coeff[j],phi(x,j))) for j in (0,1))
    return predict,mu,sd

def summarize(v):
    need(len(v)==120,'EXPECTED_120')
    return {'mean':mean(v),'median':st.median(v),'rmse':rms(v), 'mae':mean(list(map(abs,v))),
            'negative':sum(x<0 for x in v),'positive':sum(x>0 for x in v)}

def inspect_repo(root):
    root=root.resolve(strict=True)
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    need(head==HEAD,'HEAD_MISMATCH:'+head)
    status=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip()
    need(not status,'DIRTY_REPOSITORY')
    return root

def load(root):
    workers=[read(root/P0_ROOT/f'worker{i}_science.json',P0_WORKER_SHA[i]) for i in (0,1)]
    need([w['worker'] for w in workers]==[0,1],'P0_WORKERS')
    need(all(w['head']==P0_HEAD for w in workers),'P0_HEAD')
    need([w['row_range'] for w in workers]==[[0,416],[416,840]],'P0_SHARD_RANGE')
    rows=[]
    for w in workers:
        ids=w['pair_ids']; cells=w['contrast_cell_ids']; keys=set(w['cells'])
        need(len(ids)==len(cells)==len(w['row_ids'])==(w['row_range'][1]-w['row_range'][0]),'P0_WORKER_ROWS')
        need(all(f'{seed}/{arm}' in keys for seed in SEEDS for arm in ARMS),'P0_WORKER_CELLS')
        for i,(pid,cell) in enumerate(zip(ids,cells)):
            logits={}
            for seed in SEEDS:
                for arm in ARMS:
                    z=w['cells'][f'{seed}/{arm}']['logits'][i]
                    logits[f'{seed}/{arm}']=coord(z)
            rows.append({'pair':pid,'cell':cell,'z':logits})
    need(len(rows)==840,'P0_TOTAL_ROWS')
    pp={}
    for r in rows: pp.setdefault(r['pair'],set()).add(r['cell'])
    need(len(pp)==120 and all(len(v)==7 and 'C6_EXPLICIT_DENIAL' in v for v in pp.values()),'P0_120_SEVENCELL')
    summary=read(root/H2_ROOT/'full_observation_summary.json',H2_SUMMARY_SHA)
    need(summary['head']==H2_HEAD and summary['pairs']==120 and summary['actual_backbone_forwards']==144,'H2_SUMMARY_ID')
    expected_pairs=summary['pair_order'];need(set(expected_pairs)==set(pp),'P0_H2_PAIR_IDENTITY')
    h2={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f'{seed}/{arm}'
            e=summary['result_files'][key]
            d=read(root/H2_ROOT/e['filename'],e['sha256'])
            need(d['seed']==seed and d['arm']==arm and d['head']==H2_HEAD and
                 d['pair_order']==expected_pairs and set(d['pairs'])==set(expected_pairs),'H2_FILE_SCOPE:'+key)
            h2[key]={p:{c:coord(d['pairs'][p]['logits'][c]) for c in CONDS} for p in expected_pairs}
    for seed in SEEDS:
        for arm in ARMS:
            c6={r['pair']:r['z'][f'{seed}/{arm}'] for r in rows if r['cell']=='C6_EXPLICIT_DENIAL'}
            need(len(c6)==120,'P0_C6_COUNT')
            maxd=max(max(abs(a-b) for a,b in zip(c6[p],h2[f'{seed}/{arm}'][p]['NEG_DENIES'])) for p in expected_pairs)
            need(maxd<=2e-5,'P0_H2_NEG_DRIFT:'+str(seed)+'/'+arm)
            print(f'P0_H2_NEG_RS_RN_MAX_DIFF seed={seed} arm={arm} value={maxd:.9g}')
    return rows,h2,sorted(pp)

def pct(vals,p):
    s=sorted(vals);idx=p*(len(s)-1);i=int(idx);j=min(i+1,len(s)-1)
    return s[i]+(idx-i)*(s[j]-s[i])

def d2(a,b):return math.hypot(a[0]-b[0],a[1]-b[1])

def support(p0_train,p0_held,h2_by_cond, mu, sd):
    # standardized distance from each query to nearest P0 training ZERO row.
    scale=lambda x:tuple((x[j]-mu[j])/sd[j] for j in (0,1))
    cloud=[scale(x) for x in p0_train]
    min_distance=lambda x:min(d2(scale(x),z) for z in cloud)
    ref=[min_distance(x) for x in p0_held]
    t=pct(ref,.95) # calibration on P0 held-out, not selected using H2.
    return t,{c:[min_distance(x) for x in h2_by_cond[c]] for c in CONDS}, ref

def main(root):
    rows,h2,pairs=load(inspect_repo(root)); folds={p:i//24 for i,p in enumerate(pairs)}
    need(set(folds.values())==set(range(5)),'FOLDS')
    print('G2_B_SOURCE_HASH_SCHEMA_AND_PAIR_GATE_PASS')
    print('FOLDS=5 PAIRS_PER_FOLD=24 P0_TRAIN_PAIRS=96 H2_REFIT=NO')
    for seed in SEEDS:
        key0=f'{seed}/ZERO';keyf=f'{seed}/FULL'
        p0_out={m:[] for m in ('M0','M3','M4','M5')}
        h2_out={m:{p:{} for p in pairs} for m in p0_out}
        qdist={c:[] for c in CONDS};qcut=[];qref=[];qcover={c:0 for c in CONDS}
        for fold in range(5):
            train=[(r['z'][key0],r['z'][keyf]) for r in rows if folds[r['pair']]!=fold]
            held=[r for r in rows if folds[r['pair']]==fold]
            need(len(train)==672 and len(held)==168,'TRAIN_HELD_ROWS')
            funcs={};mu=sd=None
            for model in p0_out:
                fn,u,v=fit(train,model);funcs[model]=fn
                if model=='M3':mu,sd=u,v
                for r in held:p0_out[model].append((r['pair'],r['cell'],r['z'][keyf],fn(r['z'][key0])))
                for p in pairs:
                    if folds[p]!=fold:continue
                    for cond in CONDS:
                        x=h2[key0][p][cond]; y=h2[keyf][p][cond]
                        h2_out[model][p][cond]=(x,y,fn(x))
            ztrain=[r['z'][key0] for r in rows if folds[r['pair']]!=fold]
            zheld=[r['z'][key0] for r in held]
            condtest={c:[h2[key0][p][c] for p in pairs if folds[p]==fold] for c in CONDS}
            thr,ds,ref=support(ztrain,zheld,condtest,mu,sd)
            qcut.append(thr);qref.extend(ref)
            for c in CONDS:
                qdist[c].extend(ds[c]);qcover[c]+=sum(z<=thr for z in ds[c])
        print(f'===== SEED {seed} =====')
        print('ZERO_SPACE_P0_HELDOUT_NN95_CALIBRATED_COVERAGE '
              +' '.join(f'{c}={qcover[c]}/120 medianNN={st.median(qdist[c]):.7f} p90NN={pct(qdist[c],.9):.7f}' for c in CONDS))
        print(f'ZERO_SPACE_P0_HELDOUT_REF_NN median={st.median(qref):.7f} p95={pct(qref,.95):.7f}')
        xs={c:[h2[key0][p][c] for p in pairs] for c in CONDS}
        bounds=[(max(min(x[j] for x in xs[c]) for c in CONDS),
                 min(max(x[j] for x in xs[c]) for c in CONDS)) for j in (0,1)]
        overlap={c:sum(all(lo<=x[j]<=hi for j,(lo,hi) in enumerate(bounds)) for x in xs[c]) for c in CONDS}
        print('ZERO_SPACE_ALL_CONDITION_RS_RN_RECTANGLE_OVERLAP '
              +f'RS=[{bounds[0][0]:.7f},{bounds[0][1]:.7f}] '
              +f'RN=[{bounds[1][0]:.7f},{bounds[1][1]:.7f}] '
              +' '.join(f'{c}={overlap[c]}/120' for c in CONDS))
        # A bounding box can overlap even when the 2D point clouds do not.
        # Cross-condition k=1 distances use each condition's within-condition
        # 95th-percentile nearest-neighbor radius; descriptive overlap only.
        x0=[r['z'][key0] for r in rows]
        scales=[math.sqrt(mean([(x[j]-mean([v[j] for v in x0]))**2 for x in x0])) for j in (0,1)]
        need(min(scales)>1e-9,'SUPPORT_SCALE')
        clouds={c:[(x[0]/scales[0],x[1]/scales[1]) for x in xs[c]] for c in CONDS}
        self95={c:pct([min(d2(x,y) for j,y in enumerate(clouds[c]) if j!=i)
                      for i,x in enumerate(clouds[c])],.95) for c in CONDS}
        print('ZERO_SPACE_CONDITION_CROSS_NN_WITHIN95 '
              +' '.join(f'{name}={sum(min(d2(x,y) for y in clouds[b]) <= self95[a] for x in clouds[a])}/120'
                       for name,a,b in (('N_to_C','NEG_DENIES','POS_CONFIRMS'),
                                        ('C_to_N','POS_CONFIRMS','NEG_DENIES'),
                                        ('N_to_A','NEG_DENIES','POS_ASSERTS'),
                                        ('A_to_N','POS_ASSERTS','NEG_DENIES'),
                                        ('C_to_A','POS_CONFIRMS','POS_ASSERTS'),
                                        ('A_to_C','POS_ASSERTS','POS_CONFIRMS'))))
        for model in p0_out:
            pp=p0_out[model];need(len(pp)==840,'OOF_COUNT')
            print(f'{model}_P0_OOF_RMSE2={rmse2([(o,pr) for _,_,o,pr in pp]):.9f} '
                  +f'P0_C6_OOF_RMSE2={rmse2([(o,pr) for _,cell,o,pr in pp if cell=="C6_EXPLICIT_DENIAL"]):.9f}')
            for cond in CONDS:
                print(f'{model}_H2_{cond}_RMSE2={rmse2([(h2_out[model][p][cond][1],h2_out[model][p][cond][2]) for p in pairs]):.9f}')
            for comp,(ca,cb) in COMP.items():
                observed=[tuple(h2_out[model][p][ca][1][j]-h2_out[model][p][cb][1][j] for j in (0,1)) for p in pairs]
                pred=[tuple(h2_out[model][p][ca][2][j]-h2_out[model][p][cb][2][j] for j in (0,1)) for p in pairs]
                res=[observed[i][0]-pred[i][0] for i in range(120)]
                r=summarize(res)
                ref_rms=rms([x[0] for x in observed])
                inp_rms=rms([h2_out[model][p][ca][0][0]-h2_out[model][p][cb][0][0] for p in pairs])
                need(ref_rms>1e-10 and inp_rms>1e-10,'DEGENERATE_CONTRAST')
                print(f'{model}_{comp}_RS_RESIDUAL mean={r["mean"]:.9f} med={r["median"]:.9f} '
                      +f'MAE={r["mae"]:.9f} RMSE={r["rmse"]:.9f} '
                      +f'negative={r["negative"]}/120 '
                      +f'relZERO={r["rmse"]/inp_rms:.5f} relFULL={r["rmse"]/ref_rms:.5f} '
                      +f'contrast2D_RMSE={rmse2(list(zip(observed,pred))):.9f}')
        # The C6 row maps to H2 NEG. Contrast residual = residual_NEG - residual_POS.
        for model in ('M3','M5'):
            print(f'{model}_H2_ROW_RS_RESIDUAL_MEAN '
                  +' '.join(f'{c}={mean([h2_out[model][p][c][1][0]-h2_out[model][p][c][2][0] for p in pairs]):.9f}' for c in CONDS))
    print('G2_B_FROZEN_P0_H2_STATIC_REPLAY_PASS')
    print('EXPLORATORY_ONLY;H2_REFIT=NO;NO_GPU;NO_MODEL;NO_TRAINING;FILES_WRITTEN=0')
    print('STATE_MEDIATION=NOT_ESTABLISHED;PHASE3B=BLOCKED_UNCHANGED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    args=p.parse_args()
    main(args.repo)
