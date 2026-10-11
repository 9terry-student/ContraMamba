#!/usr/bin/env python3
"""Independent CPU test suite with signed exact recurrence oracle; no model/GPU."""
from __future__ import annotations
import copy
import math
import os
import unittest
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from scripts import gen5_v10_hybrid_fine_pair_candidate as v10
from scripts import gen5_v10_shared_dyadic_gemm_candidate as gemm

CLASSES=('A_TITLE','A_NAME','A_ROLE','A_PREDICATE','RESIDUAL')
PARENTS=('CLAIM->CLAIM','EVIDENCE->EVIDENCE','CLAIM->EVIDENCE')
PAIRS=tuple(f'{s}->{t}' for s in CLASSES for t in CLASSES)


def input_fixture(batch=32,width=8,kind='mixed',seed=47):
    torch.manual_seed(seed)
    k,s=4,128
    comp=torch.randn(k,batch,s,width,dtype=torch.float32)*.08
    # Large-span chunk and near-zero signed interference produce cancellation.
    la=torch.empty(batch,s,1,width,dtype=torch.float32)
    la.fill_(-.01)
    if kind=='large':la.fill_(-1.5)
    if kind=='mixed':la[:,:,:,4:]=-1.5
    # Pad some rows; frozen source-active includes tokens up to the last valid.
    attention=torch.ones(batch,s,dtype=torch.bool)
    for b in range(batch):
        attention[b,123+b%5:]=False
    comp*=attention[None,:,:,None].to(comp.dtype)
    survival=.4+torch.rand(batch,s,width,dtype=torch.float32)*.6
    idx=torch.arange(s)[None,:]
    split=50
    # Include CLS partitions and a deliberately unassigned EOS gap.
    cls1=torch.randint(0,5,(batch,s))
    cls2=torch.randint(0,5,(batch,s))
    fine_masks={
       'CLAIM':{cls:((cls1==i)&(idx<split)&attention) for i,cls in enumerate(CLASSES)},
       'EVIDENCE':{cls:((cls2==i)&(idx>split)&attention) for i,cls in enumerate(CLASSES)},
    }
    return dict(component=comp,log_a=la,survival=survival,attention_mask=attention,
                fine_masks=fine_masks,shape=SimpleNamespace(intermediate_size=1,state_size=width))


def exact_causal_oracle(*,component,log_a,survival,attention_mask,fine_masks,shape):
    """Independent lag-pair direct sum; not FFT/GEMM/dyadic factorization."""
    k,b,s,w=component.shape
    p=torch.cumsum(log_a.reshape(b,s,w).to(torch.float64),dim=1)
    labels=torch.full((b,s),-1,dtype=torch.int64)
    for base,side in ((0,'CLAIM'),(5,'EVIDENCE')):
        for ci,cls in enumerate(CLASSES):
            mask=fine_masks[side][cls]
            assert not bool(torch.any(mask & (labels!=-1)))
            labels[mask]=base+ci
    accum=torch.zeros(k,75*s,dtype=torch.float64)
    w64=component.to(torch.float64)
    surv=survival.to(torch.float64)
    for t in range(1,s):
        src=labels[:,:t]
        dst=labels[:,t,None]
        cc=(src>=0)&(src<5)&(dst>=0)&(dst<5)
        ee=(src>=5)&(src<10)&(dst>=5)&(dst<10)
        ce=(src>=0)&(src<5)&(dst>=5)&(dst<10)
        parent=torch.full_like(src,-1)
        parent[cc]=0;parent[ee]=1;parent[ce]=2
        valid=parent>=0
        fi=(src%5)*5+(dst%5)
        gap=t-torch.arange(t)[None,:]
        index=((parent*25+fi)*s+gap).clamp(min=0)
        decays=torch.exp(p[:,t,None,:]-p[:,:t,:])
        terms=(w64[:,:,:t,:]*w64[:,:,t,None,:]
               *surv[:,t,None,:][None,:,:,:]*decays[None,:,:,:])
        signed=2*terms.sum(-1)
        accum.scatter_add_(1,index.reshape(1,-1).expand(k,-1),
                           torch.where(valid[None,:,:],signed,0).reshape(k,-1))
    output={}
    for pi,parent in enumerate(PARENTS):
        output[parent]={}
        for ci,name in enumerate(PAIRS):
            output[parent][name]=accum[:,(pi*25+ci)*s:(pi*25+ci+1)*s]
    return output


class FrozenStub:
    p3a=SimpleNamespace(MAX_LENGTH=128)
    FFT_CHUNK_WIDTH=4  # Explicit synthetic CPU-only, production remains 256.
    FAST_FLOAT32_HALF_SPAN_LIMIT=60.0
    FINE_CLASSES=CLASSES
    PARENT_PAIR_NAMES=PARENTS
    PARENT_PAIRS=(('CLAIM','CLAIM'),('EVIDENCE','EVIDENCE'),('CLAIM','EVIDENCE'))
    FINE_PAIR_NAMES=PAIRS
    ENERGY_EPSILON=1e-30
    PAIR_GAP_ENERGY_ATOL=2e-5
    PAIR_GAP_ENERGY_RTOL=2e-5

    def __init__(self):self.calls=[]

    def _fine_pair_gap_interference_fft(self,**kw):
        self.calls.append((int(kw['component'].shape[1]),int(kw['component'].shape[-1])))
        output=exact_causal_oracle(**kw)
        span=v10.measure_half_span(kw['log_a'].reshape(kw['component'].shape[1],128,-1),kw['attention_mask'])
        return output,span


def compare(ref,obs):
    all_count=0;bad=0;max_abs=0.0
    for parent in PARENTS:
        for cell in PAIRS:
            delta=(ref[parent][cell]-obs[parent][cell]).abs()
            rel=delta/ref[parent][cell].abs().clamp_min(1e-30)
            bad+=int(((delta>2e-5)&(rel>2e-5)).sum().item())
            all_count+=delta.numel()
            max_abs=max(max_abs,float(delta.max().item()))
    return (all_count,bad,max_abs)


class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_available()

    def test_mixed_s32_switch_and_full_reference(self):
        obj=FrozenStub()
        kw=input_fixture(kind='mixed')
        out, span, routing=v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)
        reference=exact_causal_oracle(**kw)
        n,failed,diff=compare(reference,out)
        self.assertEqual(n,4*75*128)
        self.assertEqual(failed,0)
        self.assertEqual((routing['fast_chunks'],routing['gemm_chunks']), (1,1))
        self.assertEqual(obj.calls,[(32,4)])
        self.assertGreater(span,60.0)

    def test_all_large_uses_gemm_b32(self):
        obj=FrozenStub();kw=input_fixture(kind='large',seed=48)
        out,_,route=v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)
        self.assertEqual((route['fast_chunks'],route['gemm_chunks']),(0,2))
        self.assertEqual(obj.calls,[])
        self.assertEqual(compare(exact_causal_oracle(**kw),out)[1],0)

    def test_all_fast_uses_frozen_b32(self):
        obj=FrozenStub();kw=input_fixture(kind='fast',seed=49)
        out,_,route=v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)
        self.assertEqual((route['fast_chunks'],route['gemm_chunks']),(2,0))
        self.assertEqual(obj.calls,[(32,4),(32,4)])
        self.assertEqual(compare(exact_causal_oracle(**kw),out)[1],0)

    def test_tail8_keeps_frozen_whole_width(self):
        obj=FrozenStub();kw=input_fixture(batch=8,kind='mixed',seed=50)
        out,_,route=v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)
        self.assertEqual(route['mode'],'FROZEN_TAIL8')
        self.assertEqual((route['fast_chunks'],route['gemm_chunks'],route['frozen_tail_calls']),(0,0,1))
        self.assertEqual(obj.calls,[(8,8)])
        self.assertEqual(compare(exact_causal_oracle(**kw),out)[1],0)

    def test_wrong_batch_blocked(self):
        obj=FrozenStub();kw=input_fixture(batch=16)
        with self.assertRaisesRegex(RuntimeError,'K_BATCH_SEQUENCE'):
            v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)

    def test_production_width_mandatory(self):
        obj=FrozenStub();kw=input_fixture()
        with self.assertRaisesRegex(RuntimeError,'PRODUCTION_WIDTH'):
            v10.hybrid_fine_pair_gap(obj,**kw)

    def test_nonfinite_log_blocks(self):
        obj=FrozenStub();kw=input_fixture(kind='fast')
        kw['log_a'][0,0,0,0]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'HALFSPAN_NONFINITE'):
            v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)

    def test_extracted_v9_lineage(self):
        self.assertEqual(gemm.FROZEN_HEAD, v10.FROZEN_HEAD)
        self.assertEqual(gemm.SOURCE_V9_SHA256,
                         '09f93ee6879e553d737b77e85c901556f578d3c32e3b6119a9460bc0bef0c453')

    def test_overlapping_fine_masks_blocked(self):
        obj=FrozenStub();kw=input_fixture(kind='large',seed=51)
        kw['fine_masks']['EVIDENCE']['A_NAME'] |= kw['fine_masks']['EVIDENCE']['A_TITLE']
        with self.assertRaisesRegex(RuntimeError,'GEMM_FINE_MASK_OVERLAP'):
            v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)

    def test_dtype_guard(self):
        obj=FrozenStub();kw=input_fixture()
        kw['component']=kw['component'].double()
        with self.assertRaisesRegex(RuntimeError,'COMPONENT_DTYPE_SHAPE'):
            v10.hybrid_fine_pair_gap(obj,**kw,synthetic_cpu_geometry=True)

if __name__=='__main__':
    unittest.main(verbosity=2)
