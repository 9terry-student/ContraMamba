#!/usr/bin/env python3
"""Gen5 v10 bounded *real* CUDA experiment candidate.

Never dispatch the frozen full worker. A real execution needs a separately
approved, clean, pushed commit and explicit operator GPU consent. A successful
run does not itself establish scientific E0-E3 or source attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BRANCH = "gen5-causal-role-state-ownership"
PREDECESSOR = "f01dabd0d73014423d5c1a3c5f8d7f8c1187ae28"
SCHEMA = "GEN5_V10_TWO_LOCUS_CUDA_RUNTIME_CANDIDATE_V1"
SAMPLES = {
    "gpu0": {"worker": 0, "batch": 1, "start": 0, "stop": 32,
             "source": (6202, 6203),
             "targets": ((6201, 6203), (6202, 6201)),
             "visible_device": "0"},
    "gpu1": {"worker": 1, "batch": 1, "start": 416, "stop": 448,
             "source": (6203, 6202),
             "targets": ((6201, 6202), (6202, 6202)),
             "visible_device": "1"},
}
EXPECTED_GEOMETRY = {"K": 4, "B": 32, "S": 128, "W": 24576}


FROZEN = {
 "scripts/audit_native_recurrent_generator_anchor_pair_localization.py": "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635",
 "scripts/gen5_v10_hybrid_fine_pair_candidate.py": "56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0",
 "scripts/gen5_v10_bounded_worker_export_candidate.py": "cbfc6004c6ab177eea8e7def1cc001049ec4bc4e",
 "scripts/gen5_v10_bounded_runtime_bridge_candidate.py": "7c134614d0e9b4f35e0af70e058a7d184c6332ca",
 "scripts/gen5_v10_stage_e0_frozen_locus_adapter_candidate.py": "f16e9c47678504ddf5d2e60b294b0ef03ffb65df",
 "scripts/gen5_v10_stage_e1_bounded_cpu_dispatch_candidate.py": "5cd4e67b94dcb5c4612613f8fdb9eda600be03fd",
 "scripts/gen5_v10_stage_e2_cpu_source_byte_verifier_candidate.py": "fcb421cb363bbfd866c57e892be55665e1f8e135",
}
MIN_START_FREE_GIB = 12
MIN_PHASE_FREE_GIB = 1.5
MAX_CAPTURE_CHUNK = 4 * 1024 * 1024
HEX40 = re.compile(r"[0-9a-f]{40}\Z")

class BoundedBlocked(RuntimeError):
    pass

def need(ok: bool, why: str) -> None:
    if not ok:
        raise BoundedBlocked("GEN5_V10_BOUNDED_BLOCKED:" + why)

def _git(repo: Path, *args: str) -> str:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                           timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BoundedBlocked("GIT_COMMAND_FAILURE") from exc
    need(p.returncode == 0, "GIT_COMMAND_FAILED:" + args[0])
    return p.stdout.decode("utf-8", errors="strict").strip()

def verify_committed_execution(repo: Path, expected_head: str,
                               *, require_current_file: bool = True) -> dict[str, Any]:
    """Fail before CUDA/model imports on dirty/unpinned execution source."""
    need(type(expected_head) is str and HEX40.fullmatch(expected_head) is not None,
         "EXPECTED_HEAD_FORMAT")
    need(repo.is_dir() and not repo.is_symlink(), "REPO_MISSING")
    need(_git(repo, "rev-parse", "HEAD") == expected_head, "HEAD_DRIFT")
    need(_git(repo, "branch", "--show-current") in (BRANCH, ""), "BRANCH_DRIFT")
    need(_git(repo, "status", "--porcelain=v1", "--untracked-files=all") == "",
         "WORKTREE_NOT_CLEAN")
    _git(repo, "merge-base", "--is-ancestor", PREDECESSOR, "HEAD")
    for path, blob in FROZEN.items():
        recorded = _git(repo, "rev-parse", "HEAD:" + path)
        if blob:
            need(recorded == blob, "FROZEN_BLOB_DRIFT:" + path)
    if require_current_file:
        current = Path(__file__).resolve()
        need(current == (repo / "scripts" / Path(__file__).name).resolve(),
             "SCRIPT_OUTSIDE_REPO")
        disk = current.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(disk)).encode("ascii") + b"\x00" + disk).hexdigest()
        need(_git(repo, "rev-parse", "HEAD:scripts/" + current.name) == blob,
             "RUNNER_BYTES_DIFFER_FROM_COMMIT")
    return {"head": expected_head, "branch": BRANCH,
            "frozen_git_blobs": {k: _git(repo, "rev-parse", "HEAD:" + k) for k in FROZEN},
            "runtime_source_status": "PINNED_DISK_VS_GIT_AT_CHECKPOINT"}

def validate_locus(*, row_ids: list[str], sources: tuple[Any, ...],
                   orientations: tuple[Any, ...], spec: dict[str, Any]) -> dict[str, Any]:
    """Check exact frozen enumeration and the assigned worker/batch/source."""
    from scripts import gen5_v10_bounded_worker_export_candidate as b
    need(type(row_ids) is list and len(row_ids) == 32 and
         all(type(s) is str and s for s in row_ids) and
         len(set(row_ids)) == 32, "ROW_IDENTITY")
    need(sources == b.SOURCE_CELLS and
         orientations == b.FROZEN_ORIENTATIONS, "FROZEN_ENUMERATION_DRIFT")
    need(spec["stop"] - spec["start"] == 32 and
         spec["batch"] == 1 and spec["worker"] in (0, 1), "LOCUS_GEOMETRY")
    worker_start = 0 if spec["worker"] == 0 else 416
    need(spec["start"] == worker_start, "WORKER_FIRST_BATCH_DRIFT")
    src = spec["source"]
    need(src in sources, "SOURCE_NOT_FROZEN")
    src_rows = tuple(row for row in orientations if row[1] == src)
    need(len(src_rows) == 4 and
         tuple(row[2] for row in src_rows[:2]) == spec["targets"],
         "TARGET_PAIR_NOT_FROZEN_GROUP_ZERO")
    return {
        "worker_id": spec["worker"], "batch_index_one_based": spec["batch"],
        "row_start": spec["start"], "row_stop": spec["stop"],
        "row_ids": list(row_ids),
        "source_cell": f'{src[0]}_{src[1]}',
        "target_cells": [f'{t[0]}_{t[1]}' for t in spec["targets"]],
        "target_groups": [row[0] for row in src_rows[:2]],
        "target_group_index": 0, "expected_component_count": 4,
        "expected_batch_rows": 32,
    }

def physical_sample(torch: Any, label: str, *, min_free: int = 0,
                    samples: list[dict[str, Any]]) -> None:
    """Actual allocator/driver counters. Never infer unobserved lifetime peak."""
    torch.cuda.synchronize(torch.device("cuda:0"))
    free, total = torch.cuda.mem_get_info(torch.device("cuda:0"))
    item = {"label": label, "elapsed_ns": time.monotonic_ns(),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "max_allocated_since_reset_bytes": int(torch.cuda.max_memory_allocated(0)),
            "max_reserved_since_reset_bytes": int(torch.cuda.max_memory_reserved(0)),
            "driver_free_bytes": int(free), "driver_total_bytes": int(total)}
    samples.append(item)
    need(free >= min_free, "INSUFFICIENT_GPU_FREE_AT:" + label)

def _gpu_entry_guard(args: argparse.Namespace) -> None:
    # Not a security boundary, but prevents accidental candidate CUDA use.
    need(args.run_bounded and args.explicit_gpu_approval == "I_APPROVE_GEN5_V10_BOUNDED_GPU",
         "OPERATOR_GPU_APPROVAL_REQUIRED")
    need(os.environ.get("CONTRAMAMBA_GEN5_V10_RUN_AUTHORIZED") == "YES",
         "EXTERNAL_EXECUTION_AUTHORITY_REQUIRED")
    need(args.locus_id in SAMPLES, "LOCUS_ID_REQUIRED")
    need(os.environ.get("CUDA_VISIBLE_DEVICES") ==
         SAMPLES[args.locus_id]["visible_device"], "PHYSICAL_GPU_MAPPING")
    for field in ("expected_head", "implementation_freeze_commit"):
        need(getattr(args, field) is not None, "MISSING:" + field)
    need(args.expected_head == args.implementation_freeze_commit,
         "EXECUTION_HEAD_NOT_FREEZE")
    for field in ("model_snapshot", "tokenizer_snapshot", "checkpoint", "output_dir"):
        need(getattr(args, field) is not None, "MISSING:" + field)

def _output_dir(repo: Path, requested: Path, locus_id: str) -> Path:
    """Store each GPU outside the shared clean, commit-pinned worktree."""
    base = (repo.parent / "gen5_v10_two_sample_e2e3_artifacts").resolve()
    folder = requested.absolute()
    need(locus_id in SAMPLES and folder.name == locus_id,
         "OUTPUT_LOCUS_ID_MISMATCH")
    need(folder.parent == base and not folder.is_symlink(),
         "OUTPUT_DIR_OUTSIDE_APPROVED_ROOT")
    need(not folder.exists(), "OUTPUT_COLLISION")
    return folder

def _write_one(folder: Path, filename: str, data: dict[str, Any]) -> None:
    path = folder / filename
    raw = (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False) + "\n").encode("utf-8")
    # O_EXCL ensures no earlier artifact is silently modified.
    with path.open("xb") as stream:
        stream.write(raw)
    with (folder / (filename + ".sha256")).open("x", encoding="ascii") as stream:
        stream.write(hashlib.sha256(raw).hexdigest() + "\n")

def _capture(torch: Any, tensors: dict[str, Any], phase: str,
             *, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stream 17 actual live GPU values, no full-size GPU clones."""
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as a
    from scripts import gen5_v10_bounded_worker_export_candidate as b
    need(set(tensors) == set(a.TENSOR_KEYS), "CAPTURE_KEY_IDENTITY")
    out = {}
    for key in a.TENSOR_KEYS:
        tensor = tensors[key]
        need(tensor.device.type == "cuda" and tensor.device.index in (None, 0),
             "CAPTURE_NOT_LIVE_CUDA:" + key)
        out[key] = b.capture_value_digest(tensor, phase=phase,
                                          max_chunk_bytes=MAX_CAPTURE_CHUNK)
    physical_sample(torch, "capture_" + phase, samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    return out

def _equal_capture(pre: dict[str, Any], post: dict[str, Any], tag: str) -> None:
    need(set(pre) == set(post), "CAPTURE_KEYS:" + tag)
    for key in pre:
        for field in ("content_sha256", "byte_count", "shape", "dtype", "source_device"):
            need(pre[key][field] == post[key][field], "INPUT_MUTATED:" + tag + ":" + key)

def _record_guards(frozen: Any, accum: list[Any], old: list[dict[str, Any]]) -> dict[str, Any]:
    """Execute frozen guard semantics directly and stop on any failure."""
    from scripts import gen5_v10_bounded_runtime_bridge_candidate as c
    c.verify_projector_original_guards(
        frozen=frozen, projector_accumulators=accum, frozen_records=old)
    return {"raw_reconstruction": "CALLED_SUCCESS", "visible_raw_energy": "CALLED_SUCCESS",
            "complement_raw_energy": "CALLED_SUCCESS",
            "attribution": "IN_PROCESS_ORIGINAL_HELPER_NOT_INDEPENDENTLY_ATTESTED"}

def _prepare_actual_sources(args: argparse.Namespace, frozen: Any) -> tuple[Any, ...]:
    """Original source validator runs during loading, not via JSON claims."""
    from scripts import gen5_v10_stage_e2_cpu_source_byte_verifier_candidate as e2
    e2.verify_checkpoint_file(args.checkpoint)
    e2.verify_snapshot_bytes(args.model_snapshot)
    static, encoded, model, wrapper, strong_mask, planes = frozen.transport._runtime_inputs(args)
    e2.verify_dev_rows(static["dev_rows"])
    e2.verify_encoded_cpu_bundle(encoded["dev_bundle"], static["dev_rows"])
    return static, encoded, model, wrapper, strong_mask, planes

def _run_measured(args: argparse.Namespace, torch: Any, folder: Path,
                  source: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    from types import SimpleNamespace
    from scripts import audit_native_recurrent_generator_anchor_pair_localization as frozen
    from scripts import gen5_v10_hybrid_fine_pair_candidate as hybrid
    from scripts import gen5_v10_bounded_runtime_bridge_candidate as c
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as a
    from scripts import gen5_v10_e0e3_v3_runtime_verifier_candidate as v3
    from scripts import gen5_v10_stage_e1_bounded_cpu_dispatch_candidate as e1
    call_trace=[]
    def completed(name: str) -> None:
        index=len(call_trace)
        need(index < len(e1.ORDER) and e1.ORDER[index] == name,
             "BOUNDED_HELPER_CALL_ORDER:" + name)
        call_trace.append(name)
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1,
         "EXACT_ONE_CUDA_DEVICE")
    props = torch.cuda.get_device_properties(0)
    need(int(props.total_memory) >= 14 * 1024 ** 3, "GPU_TOTAL_LT_14GIB")
    need(not torch.backends.cuda.matmul.allow_tf32, "TF32_MUST_BE_DISABLED")
    torch.cuda.reset_peak_memory_stats(0)
    physical_sample(torch, "before_model", samples=samples,
                    min_free=MIN_START_FREE_GIB * 1024 ** 3)
    frozen._validate_runtime_head(args)
    frozen.validate_static_contract()
    static, encoded, model, wrapper, strong_mask, planes = _prepare_actual_sources(args, frozen)
    physical_sample(torch, "after_real_model", samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    model.eval()
    model.mamba.config.use_cache = False
    need(not any(p.requires_grad or p.grad is not None for p in model.parameters()),
         "TRAINING_OR_PARAM_GRADIENTS")
    # Full dev features may be staged once, but only one 32-row batch is forwarded.
    features, labels, active, targets = frozen.p3a._feature_batch_to_device(
        encoded["dev_bundle"], torch.device("cuda:0"))
    spec = SAMPLES[args.locus_id]
    start, stop = spec["start"], spec["stop"]
    worker_start, worker_stop = ((0, 416) if spec["worker"] == 0 else (416, 840))
    row_ids = list(encoded["dev_bundle"]["row_ids"][start:stop])
    cells = tuple(frozen.FULL_FACTORIAL_CELLS)
    orientations = tuple(frozen.all_orientations())
    selector = validate_locus(row_ids=row_ids, sources=cells, orientations=orientations, spec=spec)
    physical_sample(torch, "before_batch", samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    batch_features = frozen.transport.legacy._phase_b_batch_features(features, start, stop)
    completed("batch_features")
    batch_mask = batch_features["attention_mask"]
    need(tuple(batch_mask.shape) == (32,128) and batch_mask.is_cuda, "BATCH_SHAPE_DEVICE")
    role_masks = frozen.coarse._serialization_role_masks(batch_features)
    partition, anchor_provenance = frozen._build_anchor_partition(
        static=static, encoded=encoded, tokenizer_snapshot=args.tokenizer_snapshot,
        row_start=worker_start, row_stop=worker_stop)
    completed("anchor_partition")
    fine_masks = frozen._batch_anchor_masks(partition,local_start=start-worker_start,
                                            local_stop=stop-worker_start,device=batch_mask.device)
    completed("fine_masks")
    coarse_counts = frozen.coarse._pair_opportunity_counts(role_masks)
    frozen._fine_pair_opportunity_counts(fine_masks, coarse_counts)
    context = frozen.transport.legacy._phase_b_prepare_common_context(
        model=model,wrapper=wrapper,features=batch_features,
        stressor_active=active[start:stop],target_indices=targets[start:stop],
        strong_mask=strong_mask,planes=planes)
    completed("context")
    physical_sample(torch, "after_context", samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    shape=wrapper.correction.shape
    log_a=frozen.kernel._discrete_log_a(context,shape=shape).to(batch_mask.device)
    survival=frozen.pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(log_a.shape[0],log_a.shape[1],-1),
        attention_mask=batch_mask)
    raw_by_cell={}
    weights=frozen.transport._cell_weights()
    with torch.no_grad():
        for index,cell in enumerate(cells):
            need(cell in weights and cell not in raw_by_cell, "RAW_NINE_IDENTITY")
            aa,bb=weights[cell]
            raw_by_cell[cell]=frozen.transport.legacy._phase_b_raw_write(
                context["mixer_input"], batch_mask, aa, bb).detach()
            completed("raw_write")
            physical_sample(torch,"raw_"+str(index+1),samples=samples,
                            min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    need(len(raw_by_cell)==9,"RAW_NINE_COUNT")
    source_cell=spec["source"]
    source_rows=tuple(row for row in orientations if row[1]==source_cell)
    need(len(source_rows)==4 and tuple(row[2] for row in source_rows[:2]) == spec["targets"],
         "SOURCE_GROUP_DRIFT")
    _, gr, gs = frozen.transport._source_gradient_bundle(
        model=model,wrapper=wrapper,features=batch_features,
        context=context,raw_source=raw_by_cell[source_cell])
    completed("source_gradient")
    gr,gs,pinv=frozen.transport._prepare_projector_pinv(
        grad_refute=gr,grad_support=gs,attention_mask=batch_mask)
    completed("projector_pinv")
    acc=[]
    for row in source_rows[:2]:
        acc.append(frozen.transport._empty_orientation(*row))
        completed("orientation_accumulator")
    visible,complement=frozen.transport._project_visible_group(
        grad_refute=gr,grad_support=gs,pinv_cpu=pinv,
        attention_mask=batch_mask,raw_source=raw_by_cell[source_cell],
        raw_targets=[raw_by_cell[source_rows[0][2]],raw_by_cell[source_rows[1][2]]],
        accumulators=acc)
    completed("project_group")
    combined=torch.cat((visible,complement),dim=0)
    need(tuple(combined.shape)==(4,32,128,24576) and combined.is_cuda,
         "EXACT_K4_PRODUCTION_SHAPE")
    physical_sample(torch,"after_projected_k4",samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    flat=a._flat_tensors(component=combined,log_a=log_a,survival=survival,
                         attention_mask=batch_mask,role_masks=role_masks,fine_masks=fine_masks)
    before=_capture(torch,flat,"PRE_FROZEN",samples=samples)
    with torch.no_grad():
        old=frozen._component_fine_pair_stats(
            component=combined,log_a=log_a,survival=survival,
            attention_mask=batch_mask,role_masks=role_masks,
            fine_masks=fine_masks,shape=shape)
    completed("frozen_full_reducer")
    need(type(old) is list and len(old)==4,"FROZEN_K4_OUTPUT")
    physical_sample(torch,"after_frozen_fft",samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    after_frozen=_capture(torch,flat,"POST_FROZEN",samples=samples)
    _equal_capture(before,after_frozen,"FROZEN")
    view=SimpleNamespace(p3a=frozen.p3a,PARENT_PAIRS=frozen.PARENT_PAIRS,
        PARENT_PAIR_NAMES=frozen.PARENT_PAIR_NAMES,FINE_CLASSES=frozen.FINE_CLASSES,
        FINE_PAIR_NAMES=frozen.FINE_PAIR_NAMES,
        FAST_FLOAT32_HALF_SPAN_LIMIT=frozen.FAST_FLOAT32_HALF_SPAN_LIMIT,
        FFT_CHUNK_WIDTH=frozen.FFT_CHUNK_WIDTH,
        _fine_pair_gap_interference_fft=frozen._fine_pair_gap_interference_fft)
    with torch.no_grad():
        coarse_rows=frozen.coarse._component_region_pair_stats(
            component=combined,log_a=log_a,survival=survival,
            attention_mask=batch_mask,role_masks=role_masks,shape=shape)
        fine_gap,span,dispatch=hybrid.hybrid_fine_pair_gap(
            view,component=combined,log_a=log_a,survival=survival,
            attention_mask=batch_mask,fine_masks=fine_masks,shape=shape,
            synthetic_cpu_geometry=False)
        candidate=c._post_original_reducer(frozen,coarse_rows,fine_gap,span,combined)
    physical_sample(torch,"after_v10_hybrid",samples=samples,
                    min_free=int(MIN_PHASE_FREE_GIB * 1024 ** 3))
    after_v10=_capture(torch,flat,"POST_V10",samples=samples)
    _equal_capture(before,after_v10,"V10")
    guard=_record_guards(frozen,acc,old)
    need(not any(p.grad is not None for p in model.parameters()),
         "PARAMETER_GRADIENTS_ACCUMULATED")
    need(type(candidate) is list and len(candidate)==4,"V10_K4_OUTPUT")
    for engine,records in (("frozen",old),("v10",candidate)):
        for index,rec in enumerate(records):
            v3.validate_record(rec,engine+":"+str(index))
    physical_sample(torch,"after_verification",samples=samples)
    from collections import Counter
    need(tuple(call_trace) == e1.ORDER and dict(Counter(call_trace)) == e1.CALL_BUDGET,
         "BOUND_HELPER_COUNT_DRIFT")
    return {"schema_version":SCHEMA,"run_status":"BOUNDED_RUNTIME_COMPLETED_NOT_SCIENCE_PASS",
            "execution_head":args.expected_head,"selected_locus":selector,
            "checkpoint_sha256":source["checkpoint_sha256"],
            "model_snapshot":source["snapshot"],
            "dev_order_sha256":source["dev_order_sha256"],
            "dev_encoding_sha256":source["dev_encoding_sha256"],
            "bounded_original_helper_calls":dict(Counter(call_trace)),
            "v10_reducer_calls":1,
            "capture":{"pre_frozen":before,"post_frozen":after_frozen,
                       "post_v10":after_v10},"frozen_records":old,"v10_records":candidate,
            "v10_dispatch":dispatch,"raw_projector_guards":guard,
            "gpu_hardware":{"device_name":props.name,"total_bytes":int(props.total_memory),
                            "cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"]},
            "bounded_locus_id":args.locus_id,
            "gpu_memory_samples":samples,
            "GPU_EXECUTION":"BOUNDED_OBSERVED_NOT_INDEPENDENTLY_ATTESTED",
            "SOURCE_LINEAGE":"IN_PROCESS_CHECKS_NO_EXTERNAL_ATTESTATION",
            "V10_E0_PROVENANCE":"NOT_EVALUATED",
            "V10_E1_FROZEN_GUARDS":"NOT_EVALUATED",
            "V10_E2_PAYLOAD":"NOT_EVALUATED",
            "V10_E3_DIAGNOSTICS":"NOT_EVALUATED",
            "SCIENTIFIC_CONCLUSION":"PENDING_INDEPENDENT_VALIDATION"}

def run_bounded(args: argparse.Namespace) -> int:
    _gpu_entry_guard(args)
    repo=Path(__file__).resolve().parents[1]
    identity=verify_committed_execution(repo,args.expected_head)
    folder=_output_dir(repo,args.output_dir,args.locus_id)
    from scripts import gen5_v10_stage_e2_cpu_source_byte_verifier_candidate as e2
    checkpoint=e2.verify_checkpoint_file(args.checkpoint)
    snap=e2.verify_snapshot_bytes(args.model_snapshot)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder.mkdir(parents=True,exist_ok=False)
    source={"checkpoint_sha256":checkpoint["sha256"],"snapshot":str(args.model_snapshot),
            "dev_order_sha256":e2.DEV_ORDER_SHA256,"dev_encoding_sha256":e2.DEV_ENCODING_SHA256,
            "identity":identity,"snapshot_bytes":snap}
    samples=[]
    try:
        result=_run_measured(args,torch,folder,source,samples)
        _write_one(folder,"bounded_result.json",result)
        print("GEN5_V10_BOUNDED_RUNTIME_COMPLETED_REQUIRES_INDEPENDENT_VALIDATION")
        print("RESULT="+str(folder/"bounded_result.json"))
        return 0
    except BaseException as exc:
        # Memory OOM has its own receipt; never fake a completed science run.
        kind="CUDA_OOM" if isinstance(exc,torch.cuda.OutOfMemoryError) else type(exc).__name__
        failure={"schema_version":SCHEMA,"run_status":"FAILED_NOT_SCIENTIFIC_EVIDENCE",
                 "execution_head":args.expected_head,"failure_kind":kind,
                 "failure_detail":str(exc)[:500],"gpu_memory_samples":samples,
                 "E0_E3":"NOT_EVALUATED","GPU_EXECUTION":"ATTEMPTED_NOT_VALIDATED"}
        _write_one(folder,"bounded_failure.json",failure)
        print("GEN5_V10_BOUNDED_RUNTIME_FAILED_NO_SCIENTIFIC_PROMOTION",file=sys.stderr)
        raise

def inspect_predecessor_source() -> dict[str, Any]:
    """CPU only, verify exact frozen worker against its actual committed Git bytes.

    This mode runs on the E2 predecessor while the new candidate is untracked.
    No torch/model/worker import or evaluation is performed.
    """
    from scripts import gen5_v10_stage_e0_frozen_locus_adapter_candidate as e0
    need(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
         "SOURCE_INSPECTION_GPU_OFF")
    repo = Path(__file__).resolve().parents[1]
    need(_git(repo, "rev-parse", "HEAD") == PREDECESSOR,
         "CANDIDATE_PREDECESSOR_HEAD")
    need(_git(repo, "branch", "--show-current") == BRANCH,
         "SOURCE_INSPECTION_BRANCH")
    need(_git(repo, "diff", "--name-only") == "" and
         _git(repo, "diff", "--cached", "--name-only") == "",
         "SOURCE_INSPECTION_TRACKED_DRIFT")
    path = "scripts/audit_native_recurrent_generator_anchor_pair_localization.py"
    need(_git(repo, "rev-parse", "HEAD:" + path) == FROZEN[path],
         "SOURCE_INSPECTION_BLOB_DRIFT")
    for filename, git_blob in FROZEN.items():
        if git_blob:
            need(_git(repo, "rev-parse", "HEAD:" + filename) == git_blob,
                 "SOURCE_INSPECTION_PREDECESSOR_BLOB:" + filename)
    proc = subprocess.run(["git", "-C", str(repo), "show", "HEAD:" + path],
                          capture_output=True, timeout=30, check=False)
    need(proc.returncode == 0, "SOURCE_INSPECTION_READ")
    report = e0.inspect_frozen_source(proc.stdout)
    need(report["frozen_git_blob"] == FROZEN[path] and
         len(report["static_callsites"]) == 13 and
         len(report["frozen_raw_guard_literals"]) == 3,
         "SOURCE_INSPECTION_STRUCTURE")
    return {"head": PREDECESSOR, "frozen_blob": FROZEN[path],
            "callsites": len(report["static_callsites"]),
            "original_guards": len(report["frozen_raw_guard_literals"]),
            "runtime": "NOT_EXECUTED"}

def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument("--static-contract-check", action="store_true")
    m.add_argument("--inspect-predecessor-source", action="store_true")
    m.add_argument("--run-bounded", action="store_true")
    p.add_argument("--explicit-gpu-approval")
    p.add_argument("--expected-head")
    p.add_argument("--locus-id",choices=("gpu0","gpu1"))
    p.add_argument("--implementation-freeze-commit")
    p.add_argument("--model-snapshot",type=Path)
    p.add_argument("--tokenizer-snapshot",type=Path)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--output-dir",type=Path)
    return p

def main(argv: list[str] | None = None) -> int:
    args=build_parser().parse_args(argv)
    if args.static_contract_check or args.inspect_predecessor_source:
        need(not args.explicit_gpu_approval and not args.expected_head and
             not args.implementation_freeze_commit and not args.model_snapshot and
             not args.tokenizer_snapshot and not args.checkpoint and not args.output_dir and not args.locus_id,
             "STATIC_EXTRA_ARGS_FORBIDDEN")
        need(os.environ.get("CUDA_VISIBLE_DEVICES","") in ("","-1"),
             "STATIC_REQUIRES_GPU_OFF")
        need(len(SAMPLES)==2 and len(FROZEN)==7 and PREDECESSOR != "",
             "STATIC_CONSTANT_DRIFT")
        if args.inspect_predecessor_source:
            need(False, "OLD_PREDECESSOR_INSPECTION_NOT_VALID_FOR_THIS_RUNNER")
            report = inspect_predecessor_source()
            print("GEN5_V10_BOUNDED_FROZEN_SOURCE_INTEGRATION_PASS")
            print("HEAD="+report["head"]+";BLOB="+report["frozen_blob"]+";CALLSITES=13;GUARDS=3")
            print("GPU=OFF;MODEL_NOT_LOADED;RUN_NOT_EXECUTED")
            return 0
        from scripts import gen5_v10_bounded_worker_export_candidate as b
        for locus_id, spec in SAMPLES.items():
            validate_locus(row_ids=[f"synthetic-id-{i}" for i in range(32)],
                           sources=b.SOURCE_CELLS,
                           orientations=b.FROZEN_ORIENTATIONS, spec=spec)
            print("LOCUS="+locus_id+";WORKER="+str(spec["worker"])+
                  ";ROWS="+str(spec["start"])+":"+str(spec["stop"])+
                  ";SOURCE="+str(spec["source"])+
                  ";GPU="+spec["visible_device"])
        print("GEN5_V10_TWO_LOCUS_CUDA_RUNNER_STATIC_PASS")
        print("BASE="+PREDECESSOR+";SCOPE=GPU0_FIRST_BATCH_AND_GPU1_FIRST_BATCH_K4_W24576")
        print("GPU=OFF;MODEL_NOT_LOADED;EXPERIMENT_NOT_RUN;E0_E3=NOT_EVALUATED")
        return 0
    return run_bounded(args)

if __name__ == "__main__":
    raise SystemExit(main())
