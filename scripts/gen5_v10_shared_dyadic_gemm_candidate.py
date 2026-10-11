#!/usr/bin/env python3
"""Gen5 v10 extracted dyadic GEMM candidate; not an authorized audit engine.

Extracted from hash-pinned outside-repo v9 diagnostic without changing its
numerical tensor operations. Never substitutes for the frozen scientific audit.
"""
from __future__ import annotations

FROZEN_HEAD = 'e145d7d22b256c4ce5854939148a07d972bb75c8'
SOURCE_V9_SHA256 = '09f93ee6879e553d737b77e85c901556f578d3c32e3b6119a9460bc0bef0c453'


def require(ok, msg):
    if not ok:
        raise RuntimeError('V10_GEMM_CANDIDATE_BLOCKED:' + msg)


def _build_gemm_index_cache(fine, *, fine_masks, batch, seq_len, device):
    """Cache frozen semantic pair->(parent,fine,gap) scatter routing, once per batch."""
    import torch
    labels=torch.full((batch,seq_len),-1,device=device,dtype=torch.int64)
    for side_offset,side in ((0,'CLAIM'),(5,'EVIDENCE')):
        for ci,name in enumerate(fine.FINE_CLASSES):
            mask=fine_masks[side][name].to(device=device,dtype=torch.bool)
            require(tuple(mask.shape)==(batch,seq_len),'GEMM_MASK_SHAPE')
            require(not bool(torch.any((labels>=0)&mask).item()),'GEMM_FINE_MASK_OVERLAP')
            labels=torch.where(mask,side_offset+ci,labels)
    pad_len=1<<((seq_len-1).bit_length())
    if pad_len>seq_len:
        labels=torch.nn.functional.pad(labels,(0,pad_len-seq_len),value=-1)
    cache={}
    for block in (1 << n for n in range(1,pad_len.bit_length())):
        half=block//2
        nb=pad_len//block
        lab=labels.reshape(batch,nb,block)
        src=lab[:,:,:half,None]
        dst=lab[:,:,None,half:]
        claim_src=(src>=0)&(src<5)
        claim_dst=(dst>=0)&(dst<5)
        evidence_src=(src>=5)&(src<10)
        evidence_dst=(dst>=5)&(dst<10)
        parent=torch.full((batch,nb,half,half),-1,device=device,dtype=torch.int64)
        parent=torch.where(claim_src&claim_dst,0,parent)
        parent=torch.where(evidence_src&evidence_dst,1,parent)
        parent=torch.where(claim_src&evidence_dst,2,parent)
        pos_i=torch.arange(half,device=device).view(1,1,half,1)
        pos_j=torch.arange(half,device=device).view(1,1,1,half)
        gap=half+pos_j-pos_i
        valid=(parent>=0)&(gap<seq_len)
        index=(parent*25+(src%5)*5+dst%5)*seq_len+gap
        cache[block]=(
            torch.where(valid,index,0).reshape(-1),
            valid.reshape(-1),
        )
    return cache


def _dyadic_shared_gemm_chunk(fine, *, component, log_a, survival,
                               attention_mask, fine_masks, shape,
                               routing_cache=None):
    """One channel-wise GEMM per dyadic level; out-of-repo diagnostic only.

    WARNING: FP32 matmul and scatter reduction do not guarantee bitwise
    equivalence with frozen complex64 FFT correlations. Frozen replay guards
    must be run without modifying tolerances, including the relative-max
    reconstruction diagnostics. No fallback/rounding bypass is permitted.
    """
    import torch
    k,b,seq,width=component.shape
    require(seq==fine.p3a.MAX_LENGTH,'GEMM_SEQ')
    require(k>0 and width<=fine.FFT_CHUNK_WIDTH,'GEMM_CHUNK_WIDTH')
    require(tuple(log_a.shape)==(b,seq,shape.intermediate_size,shape.state_size),'GEMM_LOG_SHAPE')
    require(width==shape.intermediate_size*shape.state_size,'GEMM_WIDTH')
    require(tuple(survival.shape)==(b,seq,width),'GEMM_SURVIVAL')
    require(tuple(attention_mask.shape)==(b,seq),'GEMM_ATTENTION')
    require(component.dtype==torch.float32,'GEMM_COMPONENT_DTYPE')
    if component.is_cuda:
        require(not torch.backends.cuda.matmul.allow_tf32,'GEMM_FP32_NOT_TF32')
    active=torch.flip(torch.cumsum(torch.flip(attention_mask.to(torch.int64),dims=(1,)),dim=1),dims=(1,))>0
    require(bool(torch.all(torch.any(active,dim=1)).item()),'GEMM_ACTIVE_SEQUENCE')
    p=torch.cumsum(log_a.reshape(b,seq,width).to(torch.float64),dim=1)
    pmax=torch.amax(torch.where(active[:,:,None],p,torch.full_like(p,-torch.inf)),dim=1)
    pmin=torch.amin(torch.where(active[:,:,None],p,torch.full_like(p,torch.inf)),dim=1)
    require(bool(torch.all(torch.isfinite(pmax)&torch.isfinite(pmin)).item()),'GEMM_PREFIX_NONFINITE')
    half_span=float(torch.amax(0.5*(pmax-pmin)).item())
    require(half_span>fine.FAST_FLOAT32_HALF_SPAN_LIMIT,'GEMM_FALLBACK_ONLY')
    width_data=torch.where(active[None,:,:,None],component,torch.zeros_like(component))
    pad_len=1<<((seq-1).bit_length())
    if pad_len>seq:
        extra=pad_len-seq
        width_data=torch.nn.functional.pad(width_data,(0,0,0,extra))
        p=torch.cat((p,p[:,-1:,:].expand(b,extra,width)),dim=1)
        survival=torch.nn.functional.pad(survival,(0,0,0,extra))
    routes=routing_cache or _build_gemm_index_cache(
        fine,fine_masks=fine_masks,batch=b,seq_len=seq,device=component.device)
    result=torch.zeros((k,75*seq),dtype=torch.float64,device=component.device)
    block=2
    while block<=pad_len:
        half=block//2
        nb=pad_len//block
        pblocks=p.reshape(b,nb,block,width)
        left_p=pblocks[:,:,:half,:]
        right_p=pblocks[:,:,half:,:]
        shift=torch.amin(left_p,dim=2)
        lexp=-left_p+shift[:,:,None,:]
        rexp=right_p-shift[:,:,None,:]
        require(bool(torch.all(lexp<=0).item()),'DYADIC_CROSS_LEFT_SCALE_SIGN')
        require(bool(torch.all(rexp<=0).item()),'DYADIC_CROSS_RIGHT_SCALE_SIGN')
        xs=torch.exp(lexp).to(component.dtype)
        ys=torch.exp(rexp).to(component.dtype)
        require(bool(torch.all(torch.isfinite(xs)).item()),'DYADIC_CROSS_X_SCALE_NONFINITE')
        require(bool(torch.all(torch.isfinite(ys)).item()),'DYADIC_CROSS_Y_SCALE_NONFINITE')
        blocks=width_data.reshape(k,b,nb,block,width)
        left=blocks[:,:,:,:half,:]*xs[None,:,:,:,:]
        right=(blocks[:,:,:,half:,:]
               *survival.reshape(b,nb,block,width)[:,:,half:,:][None,:,:,:,:]
               *ys[None,:,:,:,:])
        paired=2.0*torch.matmul(left,right.transpose(-1,-2))
        index,valid=routes[block]
        flat=paired.reshape(k,-1)
        values=torch.where(valid[None,:],flat,torch.zeros_like(flat)).to(torch.float64)
        result.scatter_add_(1,index[None,:].expand(k,-1),values)
        block*=2
    expanded=result.reshape(k,3,25,seq)
    all_finite=bool(torch.all(torch.isfinite(expanded)).item())
    require(all_finite,'GEMM_ALL_FINE_FINITE')
    require(bool(torch.all(expanded[:,:,:,0]==0).item()),'GEMM_ZERO_GAP')
    output={}
    for pi,(src_side,tgt_side) in enumerate(fine.PARENT_PAIRS):
        parent=f'{src_side}->{tgt_side}'
        output[parent]={fine_name:expanded[:,pi,fi,:]
                        for fi,fine_name in enumerate(fine.FINE_PAIR_NAMES)}
    return output,half_span
