#!/usr/bin/env python3
"""Gen5 v10 hybrid fine-gap *candidate*: CPU static preflight, not scientific execution.

Frozen implementation is injected as `fine`. 32-row batches use exact per-chunk
frozen fast FFT (half-span <=60) or the extracted v9 large-span shared GEMM
candidate (>60). Tail-8 invokes the full frozen fine-gap reducer unchanged.

No input/model/checkpoint loading; no GPU entrypoint; does not change repository.
The outer frozen _component_fine_pair_stats reconstruction guard is NOT
replaced here. This candidate is NOT approved for full scientific execution.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from types import SimpleNamespace

FROZEN_HEAD = 'e145d7d22b256c4ce5854939148a07d972bb75c8'
FROZEN_SCRIPT_BLOB = 'f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635'
FROZEN_TEST_BLOB = '26c90ada9a74a2ac2716c880e58d15df5bd01704'


def require(pred: bool, reason: str) -> None:
    if not pred:
        raise RuntimeError('V10_STATIC_BLOCKED:' + reason)


def measure_half_span(log_chunk, attention_mask):
    """Match frozen float64 prefix and source-active mask criterion exactly."""
    import torch
    b, seq, cw = log_chunk.shape
    require(tuple(attention_mask.shape) == (b, seq), 'HALFSPAN_MASK_SHAPE')
    source_active = torch.flip(
        torch.cumsum(torch.flip(attention_mask.to(torch.int64), (1,)), 1), (1,)
    ) > 0
    require(bool(torch.all(torch.any(source_active, dim=1)).item()), 'HALFSPAN_EMPTY')
    p = torch.cumsum(log_chunk.to(torch.float64), dim=1)
    active = source_active[:, :, None]
    p_max = torch.amax(torch.where(active, p, torch.full_like(p, -torch.inf)), dim=1)
    p_min = torch.amin(torch.where(active, p, torch.full_like(p, torch.inf)), dim=1)
    require(bool(torch.all(torch.isfinite(p_max) & torch.isfinite(p_min)).item()), 'HALFSPAN_NONFINITE')
    observed = float(torch.amax(.5 * (p_max - p_min)).item())
    require(math.isfinite(observed), 'HALFSPAN_BAD')
    return observed


def hybrid_fine_pair_gap(
    fine, *, component, log_a, survival, attention_mask, fine_masks, shape,
    synthetic_cpu_geometry=False,
):
    """Read-only candidate returns (fine_gap, max_half_span, dispatch_record).

    All production-like shapes must be K=4, B=32 or B=8, S=128, W=24576.
    Synthetic CPU tests MAY use smaller widths with explicit synthetic flag;
    this is never an authorization to run GEMM on arbitrary production shapes.
    """
    import torch
    require(not component.is_cuda or not synthetic_cpu_geometry, 'SYNTHETIC_CUDA_FORBIDDEN')
    require(component.ndim == 4 and component.dtype == torch.float32, 'COMPONENT_DTYPE_SHAPE')
    k, b, seq, width = component.shape
    require(k == 4 and b in (8, 32) and seq == 128, 'K_BATCH_SEQUENCE')
    require(width == shape.intermediate_size * shape.state_size, 'STATE_WIDTH')
    require(synthetic_cpu_geometry or width == 24576, 'PRODUCTION_WIDTH')
    require(tuple(log_a.shape) == (b, seq, shape.intermediate_size, shape.state_size), 'LOG_A_SHAPE')
    require(tuple(survival.shape) == (b, seq, width), 'SURVIVAL_SHAPE')
    require(tuple(attention_mask.shape) == (b, seq), 'ATTENTION_SHAPE')
    require(len(fine.PARENT_PAIR_NAMES) == 3 and len(fine.FINE_PAIR_NAMES) == 25, 'CELL_GEOMETRY')
    require((fine.FFT_CHUNK_WIDTH == 256 or
             (synthetic_cpu_geometry and not component.is_cuda and fine.FFT_CHUNK_WIDTH == 4))
            and fine.FAST_FLOAT32_HALF_SPAN_LIMIT == 60.0, 'FROZEN_DISPATCH')
    if component.is_cuda:
        require(not torch.backends.cuda.matmul.allow_tf32, 'TF32_ENABLED')
    # Safest branch: tail8 uses whole frozen function; do not re-chunk it.
    if b == 8:
        with torch.no_grad():
            gap, span = fine._fine_pair_gap_interference_fft(
                component=component, log_a=log_a, survival=survival,
                attention_mask=attention_mask, fine_masks=fine_masks, shape=shape,
            )
        verify_output(fine, gap, k, seq)
        return gap, span, {'batch_rows':8, 'mode':'FROZEN_TAIL8',
                           'fast_chunks':0, 'gemm_chunks':0, 'frozen_tail_calls':1,
                           'total_chunks':math.ceil(width / fine.FFT_CHUNK_WIDTH),
                           'max_half_span':span}

    from scripts import gen5_v10_shared_dyadic_gemm_candidate as gemm
    flat_log = log_a.reshape(b, seq, width)
    output = {parent:{cell:torch.zeros((k,seq),dtype=torch.float64,device=component.device)
                      for cell in fine.FINE_PAIR_NAMES}
              for parent in fine.PARENT_PAIR_NAMES}
    routes = None
    nfast, ngemm = 0, 0
    max_span = 0.0
    # Always maintain original start:stop chunk order, with float64 accumulation.
    with torch.no_grad():
        for start in range(0, width, fine.FFT_CHUNK_WIDTH):
            stop = min(start + fine.FFT_CHUNK_WIDTH, width)
            cw = stop - start
            local_log = flat_log[:,:,start:stop].reshape(b,seq,1,cw)
            local_shape = SimpleNamespace(intermediate_size=1,state_size=cw)
            kw = dict(component=component[:,:,:,start:stop], log_a=local_log,
                      survival=survival[:,:,start:stop], attention_mask=attention_mask,
                      fine_masks=fine_masks, shape=local_shape)
            observed_span = measure_half_span(flat_log[:,:,start:stop], attention_mask)
            max_span = max(max_span, observed_span)
            if observed_span <= fine.FAST_FLOAT32_HALF_SPAN_LIMIT:
                part, frozen_span = fine._fine_pair_gap_interference_fft(**kw)
                require(observed_span == frozen_span, 'FAST_CHUNK_SPAN_DRIFT')
                nfast += 1
            else:
                if routes is None:
                    routes = gemm._build_gemm_index_cache(
                        fine, fine_masks=fine_masks, batch=b, seq_len=seq,
                        device=component.device)
                part, gemm_span = gemm._dyadic_shared_gemm_chunk(
                    fine, **kw, routing_cache=routes)
                require(observed_span == gemm_span, 'GEMM_CHUNK_SPAN_DRIFT')
                ngemm += 1
            # Mimic frozen per-chunk fine-cell update: gap 0 must remain zero.
            for parent in fine.PARENT_PAIR_NAMES:
                for cell in fine.FINE_PAIR_NAMES:
                    output[parent][cell][:,1:] += part[parent][cell][:,1:]
        require(nfast + ngemm == math.ceil(width / fine.FFT_CHUNK_WIDTH), 'CHUNK_COVERAGE')
        verify_output(fine,output,k,seq)
    return output,max_span,{'batch_rows':32, 'mode':'CHUNKWISE_HYBRID',
                            'fast_chunks':nfast,'gemm_chunks':ngemm,
                            'frozen_tail_calls':0,'total_chunks':nfast+ngemm,
                            'max_half_span':max_span}


def verify_output(fine, gap, k, seq):
    import torch
    require(set(gap)==set(fine.PARENT_PAIR_NAMES), 'PARENT_OUTPUT_KEYS')
    for parent in fine.PARENT_PAIR_NAMES:
        require(set(gap[parent])==set(fine.FINE_PAIR_NAMES), 'FINE_OUTPUT_KEYS')
        for cell in fine.FINE_PAIR_NAMES:
            v=gap[parent][cell]
            require(tuple(v.shape)==(k,seq) and v.dtype==torch.float64, 'OUTPUT_SHAPE_DTYPE')
            require(bool(torch.all(torch.isfinite(v)).item()), 'OUTPUT_NONFINITE')
            require(bool(torch.all(v[:,0] == 0).item()), 'OUTPUT_ZERO_GAP')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',action='store_true')
    a=p.parse_args()
    require(a.plan,'ONLY_PLAN_ENTRYPOINT_AUTHORIZED')
    require(os.environ.get('CUDA_VISIBLE_DEVICES','') == '', 'PLAN_REQUIRES_CUDA_DISABLED')
    print('GEN5_V10_HYBRID_CPU_STATIC_PLAN_PASS')
    print('MODE=CPU_ONLY;GPU=OFF;SCIENTIFIC_EXECUTION=NO;TRAINING=NO')
    print('FROZEN_HEAD='+FROZEN_HEAD)
    print('FROZEN_SCRIPT_BLOB='+FROZEN_SCRIPT_BLOB)
    print('K4_B32:chunk_span_le_60=frozen_fast_fft;chunk_span_gt_60=shared_gemm_candidate')
    print('K4_B8:full_width_frozen_unchanged')
    print('FROZEN_GUARDS=NOT_BYPASSED;PRODUCTION=BLOCKED')

if __name__=='__main__':
    main()
