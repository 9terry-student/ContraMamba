from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_pp3_necessity_fast_cuda as pp3
from scripts import reason_router_gen5_phase1b_q22_cuda_equivalence as cuda_eq
from scripts import reason_router_gen5_phase1b_r22_local_necessity_fast_cuda as necessity_cuda
from scripts import reason_router_gen5_phase1b_r22_restoration_confirmation as restoration

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
QUALIFIED_BACKEND = "FROZEN_MAMBA_SSM_KERNEL_CAPTURE_PLUS_LAYER22_STATE_REPLAY"

AUTHORITY_COMMIT = restoration.IMPLEMENTATION_AUTHORITY_COMMIT
Q22_SOURCE_BLOB = "421d30f00cf71690ed41c983ccf0540808e1de1c"
PP3_SOURCE_BLOB = "26ca67ad8603799a849c151a39728368227326df"

FROZEN_BLOBS = {
    restoration.IMPLEMENTATION_AUTHORITY_PATH:
        restoration.IMPLEMENTATION_AUTHORITY_BLOB,
    "scripts/reason_router_gen5_phase1b_q22_cuda_equivalence.py":
        Q22_SOURCE_BLOB,
    "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py":
        PP3_SOURCE_BLOB,
    "scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py":
        "50be219720b2809969b4f3caac25b2c25598a81b",
}

CONDITIONS = ("B", "RR", "RC")
WORKER_FORWARD_BUDGET = restoration.SHARD_MODEL_FORWARD_BUDGET


class RestorationCudaError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RestorationCudaError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RestorationCudaError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    restoration.authenticate_repo(expected_head)
    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB_DRIFT:{path}:{observed}")
    cuda_eq.authenticate_repo(expected_head)


def _finite_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    require(torch.is_tensor(value), f"{label}_NOT_TENSOR")
    out = value.detach().cpu().to(torch.float32).contiguous().clone()
    require(tuple(out.shape) == restoration.STATE_SHAPE, f"{label}_SHAPE")
    require(bool(torch.isfinite(out).all().item()), f"{label}_NONFINITE")
    return out


def _tensor_hash(value: torch.Tensor) -> str:
    return restoration.sha256_bytes(
        value.detach().cpu().contiguous().numpy().tobytes()
    )


def _install_upstream_hook(
    *,
    runtime_ctx: Mapping[str, Any],
    target: int,
    direction: torch.Tensor,
    orientation: int,
    plus_branch: bool,
    pp3_planes: Mapping[str, torch.Tensor],
    neutralized: bool,
):
    audit: dict[str, Any] = {}
    handle = pp3.install_hook(
        runtime_ctx["mixer17"],
        token_index=int(target),
        strong_mask=runtime_ctx["strong_mask"],
        condition="pp3_neutralized" if neutralized else "native",
        planes=pp3_planes,
        direction=direction,
        orientation=int(orientation),
        branch_sign=1 if plus_branch else -1,
        audit=audit,
    )
    return handle, audit


def replay_restoration_window(
    captured: tuple[torch.Tensor, ...],
    *,
    anchor: int,
    condition: str,
    donor_write: torch.Tensor | None,
    r22: torch.Tensor,
    c22: torch.Tensor,
    kernels: Mapping[str, Any],
) -> dict[str, Any]:
    require(condition in {"DONOR", *CONDITIONS}, f"CONDITION:{condition}")
    require(len(captured) == 8, "REPLAY_CAPTURE_WIDTH")
    u, delta, a_matrix, b_scan, c_scan, d_vector, gate, delta_bias = captured
    target = int(anchor) + cuda_eq.core.TARGET_OFFSET
    require(anchor >= 1 and anchor + 4 < u.shape[-1], "REPLAY_RANGE")

    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    _, state = kernel_scan(
        cuda_eq._slice_last(u, anchor),
        cuda_eq._slice_last(delta, anchor),
        a_matrix,
        cuda_eq._slice_last(b_scan, anchor),
        cuda_eq._slice_last(c_scan, anchor),
        d_vector,
        cuda_eq._slice_last(gate, anchor),
        delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    require(torch.is_tensor(state), "PREFIX_STATE")

    post_states: dict[int, torch.Tensor] = {}
    native_write: torch.Tensor | None = None
    actual_write: torch.Tensor | None = None
    plan: dict[str, Any] | None = None

    for token in range(anchor, anchor + 5):
        if token != target:
            _ = kernel_update(
                state, u[..., token], delta[..., token], a_matrix,
                b_scan[..., token], c_scan[..., token], d_vector,
                gate[..., token], delta_bias, dt_softplus=True,
            )
        else:
            pre = state.detach().clone()
            state_native = pre.clone()
            state_decay = pre.clone()
            _ = kernel_update(
                state_native, u[..., token], delta[..., token], a_matrix,
                b_scan[..., token], c_scan[..., token], d_vector,
                gate[..., token], delta_bias, dt_softplus=True,
            )
            _ = kernel_update(
                state_decay, torch.zeros_like(u[..., token]), delta[..., token],
                a_matrix, b_scan[..., token], c_scan[..., token], d_vector,
                gate[..., token], delta_bias, dt_softplus=True,
            )
            native_write = _finite_tensor(state_native - state_decay, "NATIVE_WRITE22")
            if condition == "DONOR":
                actual_write = native_write.clone()
                plan = {
                    "a_native": (r22.T @ native_write.to(torch.float64).reshape(-1)).contiguous(),
                    "matched_addition_norm_residual": 0.0,
                    "matched_addition_norm_tolerance": 0.0,
                    "r_add_l2": 0.0,
                    "c_add_l2": 0.0,
                }
            else:
                require(donor_write is not None, "DONOR_WRITE_REQUIRED")
                plan = restoration.restoration_vectors(
                    native_write, donor_write, r22, c22, condition
                )
                actual_write = (
                    plan["modified"]
                    .reshape(restoration.STATE_SHAPE)
                    .to(device=state_decay.device, dtype=state_decay.dtype)
                    .detach().cpu().to(torch.float32).contiguous()
                )
            state = (
                state_decay
                + actual_write.to(device=state_decay.device, dtype=state_decay.dtype)
            ).contiguous()
        post_states[token] = _finite_tensor(state, f"POST_STATE22:{token}")

    require(native_write is not None and actual_write is not None and plan is not None,
            "TARGET_WRITE_MISSING")
    pe22 = restoration.necessity.q22_path_efficiency(post_states, int(anchor))
    return {
        "native_write": native_write,
        "actual_write": actual_write,
        "plan": plan,
        "post_states": post_states,
        "pe22": float(pe22),
        "target_token_index": int(target),
    }


def run_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
    direction: torch.Tensor,
    orientation: int,
    plus_branch: bool,
    condition: str,
    donor_write: torch.Tensor | None,
    pp3_planes: Mapping[str, torch.Tensor],
    r22: torch.Tensor,
    c22: torch.Tensor,
    budget: Any,
) -> dict[str, Any]:
    target = int(anchor) + cuda_eq.core.TARGET_OFFSET
    neutralized = condition != "DONOR"
    handle, upstream_audit = _install_upstream_hook(
        runtime_ctx=runtime_ctx,
        target=target,
        direction=direction,
        orientation=orientation,
        plus_branch=plus_branch,
        pp3_planes=pp3_planes,
        neutralized=neutralized,
    )
    capture = cuda_eq.Layer22FastScanCapture(model.mamba.layers[restoration.LAYER22].mixer)
    budget.consume()
    try:
        with capture.capture():
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids.to("cuda:0"))
    finally:
        handle.remove()
    require(bool(upstream_audit), "UPSTREAM_HOOK_AUDIT")
    require(int(upstream_audit["token_index"]) == target, "UPSTREAM_TARGET_TOKEN")
    replay = replay_restoration_window(
        capture.captured or (),
        anchor=int(anchor),
        condition=condition,
        donor_write=donor_write,
        r22=r22,
        c22=c22,
        kernels=kernels,
    )
    replay["upstream_condition"] = "native" if not neutralized else "pp3_neutralized"
    replay["upstream_audit"] = {
        "condition": str(upstream_audit["condition"]),
        "token_index": int(upstream_audit["token_index"]),
        "orientation": int(upstream_audit["orientation"]),
        "branch_sign": int(upstream_audit["branch_sign"]),
        "probe_correction_l2": float(upstream_audit["probe_correction_l2"]),
        "applied_correction_max_abs_residual":
            float(upstream_audit["applied_correction_max_abs_residual"]),
    }
    return replay


def _public_audit(
    branch: Mapping[str, Any],
    *,
    donor: Mapping[str, Any],
    background: Mapping[str, Any],
    rr: Mapping[str, Any],
    rc: Mapping[str, Any],
    r22: torch.Tensor,
) -> dict[str, Any]:
    donor_w = donor["native_write"].to(torch.float64)
    b_w = background["native_write"].to(torch.float64)
    rr_w = rr["native_write"].to(torch.float64)
    rc_w = rc["native_write"].to(torch.float64)

    b_rr_native_residual = float(torch.max(torch.abs(b_w - rr_w)).item())
    b_rc_native_residual = float(torch.max(torch.abs(b_w - rc_w)).item())
    require(b_rr_native_residual <= restoration.MATCH_TOL, "RR_BACKGROUND_WRITE_MISMATCH")
    require(b_rc_native_residual <= restoration.MATCH_TOL, "RC_BACKGROUND_WRITE_MISMATCH")

    rr_plan = rr["plan"]
    rc_plan = rc["plan"]
    rr_a = rr_plan["a_native"].detach().cpu().to(torch.float64)
    rc_a = rc_plan["a_native"].detach().cpu().to(torch.float64)
    coeff_residual = float(torch.max(torch.abs(rr_a - rc_a)).item())
    require(coeff_residual <= restoration.MATCH_TOL, "DONOR_COEFFICIENT_MISMATCH")

    target = int(branch["target_token_index"])
    donor_state = donor["post_states"][target].to(torch.float64)
    b_state = background["post_states"][target].to(torch.float64)
    rr_state = rr["post_states"][target].to(torch.float64)
    rc_state = rc["post_states"][target].to(torch.float64)

    return {
        "target_token_index": target,
        "donor_write_sha256": _tensor_hash(donor["native_write"]),
        "background_write_sha256": _tensor_hash(background["native_write"]),
        "rr_actual_write_sha256": _tensor_hash(rr["actual_write"]),
        "rc_actual_write_sha256": _tensor_hash(rc["actual_write"]),
        "donor_r22_projection_l2": float(
            torch.linalg.vector_norm(r22.T @ donor_w.reshape(-1)).item()
        ),
        "rr_addition_l2": float(rr_plan["r_add_l2"]),
        "rc_addition_l2": float(rc_plan["c_add_l2"]),
        "matched_addition_norm_residual":
            max(float(rr_plan["matched_addition_norm_residual"]),
                float(rc_plan["matched_addition_norm_residual"])),
        "donor_coefficient_rr_rc_max_abs_residual": coeff_residual,
        "background_native_write_rr_max_abs_residual": b_rr_native_residual,
        "background_native_write_rc_max_abs_residual": b_rc_native_residual,
        "rr_target_post_state_change_l2":
            float(torch.linalg.vector_norm(rr_state - b_state).item()),
        "rc_target_post_state_change_l2":
            float(torch.linalg.vector_norm(rc_state - b_state).item()),
        "rr_target_post_state_to_donor_l2":
            float(torch.linalg.vector_norm(rr_state - donor_state).item()),
        "rc_target_post_state_to_donor_l2":
            float(torch.linalg.vector_norm(rc_state - donor_state).item()),
        "donor_upstream_condition": donor["upstream_condition"],
        "background_upstream_condition": background["upstream_condition"],
        "rr_upstream_condition": rr["upstream_condition"],
        "rc_upstream_condition": rc["upstream_condition"],
    }


def run_signed_coordinate(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[Any, Any],
    pp3_planes: Mapping[str, torch.Tensor],
    r22: torch.Tensor,
    c22: torch.Tensor,
    budget: Any,
) -> dict[str, Any]:
    pair = str(seed["source_pair_id"])
    anchors = restoration.necessity.phase1._anchors_for_pair(pair, events)
    cells = restoration.necessity.phase1._cells()

    result: dict[str, Any] = {"orientation": int(orientation), "branches": {}}
    for role, plus_branch in (("tp", True), ("tm", False)):
        input_ids = restoration.necessity.construction.phase2._input_row(
            encoded, row_index, pair, cells[role]
        )
        common = dict(
            model=model, runtime_ctx=runtime_ctx, kernels=kernels,
            input_ids=input_ids, anchor=int(anchors[role]), direction=direction,
            orientation=int(orientation), plus_branch=plus_branch,
            pp3_planes=pp3_planes, r22=r22, c22=c22, budget=budget,
        )
        donor = run_branch(condition="DONOR", donor_write=None, **common)
        background = run_branch(condition="B", donor_write=donor["native_write"], **common)
        rr = run_branch(condition="RR", donor_write=donor["native_write"], **common)
        rc = run_branch(condition="RC", donor_write=donor["native_write"], **common)

        require(
            donor["target_token_index"]
            == background["target_token_index"]
            == rr["target_token_index"]
            == rc["target_token_index"],
            "TARGET_TOKEN_IDENTITY",
        )
        public = _public_audit(
            {"target_token_index": donor["target_token_index"]},
            donor=donor, background=background, rr=rr, rc=rc, r22=r22,
        )
        public.update({
            "donor_pe22": float(donor["pe22"]),
            "B_pe22": float(background["pe22"]),
            "RR_pe22": float(rr["pe22"]),
            "RC_pe22": float(rc["pe22"]),
        })
        result["branches"][role] = public

    for cond in CONDITIONS:
        result[f"F22_{cond}"] = (
            float(result["branches"]["tp"][f"{cond}_pe22"])
            - float(result["branches"]["tm"][f"{cond}_pe22"])
        )
    return result


def run_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    family: str,
    basis_index: int,
    **kwargs,
) -> dict[str, Any]:
    pos = run_signed_coordinate(seed, direction, orientation=1, **kwargs)
    neg = run_signed_coordinate(seed, direction, orientation=-1, **kwargs)
    out: dict[str, Any] = {
        "direction_key": f"{family}_{basis_index}",
        "basis_family": family,
        "basis_index": int(basis_index),
        "positive_probe": pos,
        "negative_probe": neg,
    }
    for cond in CONDITIONS:
        f_plus = float(pos[f"F22_{cond}"])
        f_minus = float(neg[f"F22_{cond}"])
        j = (f_plus - f_minus) / (2.0 * restoration.EPSILON)
        require(math.isfinite(j), f"J22_NONFINITE:{cond}:{family}:{basis_index}")
        out[f"F22_{cond}_plus"] = f_plus
        out[f"F22_{cond}_minus"] = f_minus
        out[f"J22_{cond}"] = j
        out[f"J22_{cond}_squared"] = j * j
    return out


def run_pair(
    seed: Mapping[str, Any],
    *,
    q_bases: Mapping[str, torch.Tensor],
    **kwargs,
) -> dict[str, Any]:
    directions: list[dict[str, Any]] = []
    for family in ("xg2", "xg4"):
        for basis_index in range(restoration.K):
            directions.append(
                run_direction(
                    seed, q_bases[family][:, basis_index],
                    family=family, basis_index=basis_index, **kwargs
                )
            )
    require(
        [row["direction_key"] for row in directions] == list(restoration.DIRECTIONS),
        "DIRECTION_ORDER",
    )

    q: dict[str, float] = {}
    for cond in CONDITIONS:
        e2 = sum(float(row[f"J22_{cond}_squared"]) for row in directions[:restoration.K]) / restoration.K
        e4 = sum(float(row[f"J22_{cond}_squared"]) for row in directions[restoration.K:]) / restoration.K
        q[cond] = e2 - e4
        require(math.isfinite(q[cond]), f"Q22_NONFINITE:{cond}")

    ep = restoration.endpoint(q["B"], q["RR"], q["RC"])
    return {
        **dict(seed),
        "schema_version": restoration.ITEM_SCHEMA,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "qualified_backend": QUALIFIED_BACKEND,
        "direction_order": list(restoration.DIRECTIONS),
        "directions": directions,
        **ep,
        "scientific_model_forward_count_this_run": restoration.FORWARDS_PER_PAIR,
        "cpu_scientific_model_forward_count_this_run": 0,
        "row_dropped": False,
    }


def _seed(pair_index: int, pair: str, events: Mapping[Any, Any]) -> dict[str, Any]:
    anchors = restoration.necessity.phase1._anchors_for_pair(pair, events)
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": int(pair_index),
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def run_worker(args: argparse.Namespace) -> None:
    require(args.shard_id in (0, 1), "WORKER_SHARD_ID")
    authenticate_repo(args.expected_head)
    restoration.validate_necessity_prerequisite()
    restoration.validate_static_inputs()
    cuda_eq.backend.runtime_gate()
    require(torch.cuda.device_count() == 1, f"WORKER_VISIBLE_DEVICE_COUNT:{torch.cuda.device_count()}")
    require(torch.cuda.get_device_name(0) == "Tesla T4", "WORKER_DEVICE")
    require(not args.shard_output.exists(), "SHARD_OUTPUT_COLLISION")
    require(not args.shard_meta.exists(), "SHARD_META_COLLISION")

    r22, c22, basis_geometry = restoration.load_owner_bases()
    q_bases = restoration.load_q_bases()
    pp3_planes = restoration.load_pp3_planes()

    with cuda_eq.backend.parent_runtime_rebind():
        rows, encoded, events, row_index, tokenizer_provenance = restoration.load_inputs(
            args.tokenizer_snapshot
        )
        del rows
        kernels = cuda_eq.kernel_compat.load_exact_fast_kernels()
        with cuda_eq.kernel_compat.exact_transformers_kernel_loader(kernels) as constructor_calls:
            model, checkpoint_sha = cuda_eq.parent.load_representative_model_external(
                model_snapshot=args.model_snapshot,
                checkpoint_path=args.checkpoint,
            )
            require(
                checkpoint_sha == restoration.necessity.CHECKPOINT_SHA256,
                "CHECKPOINT_SHA256",
            )
            runtime_ctx = cuda_eq.transport_runtime.validate_runtime_components(model)
        counts = Counter(constructor_calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"}
            and counts["causal-conv1d"] == counts["mamba-ssm"]
            and counts["mamba-ssm"] > 0,
            f"CONSTRUCTOR_KERNEL_COUNTS:{dict(counts)}",
        )
        cuda_eq.kernel_compat.validate_transformers_kernel_bindings(kernels)
        model.to(torch.device("cuda:0"))
        model.eval()
        before_signature = cuda_eq.model_parameter_signature(model)

        budget = cuda_eq.parent.ForwardBudget(WORKER_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []
        pairs = restoration.shard_pairs(args.shard_id)
        global_pairs = restoration.expected_pairs()
        start_index = 0 if args.shard_id == 0 else restoration.SHARD_PAIR_COUNT
        for local_index, pair in enumerate(pairs):
            global_index = start_index + local_index
            require(global_pairs[global_index] == pair, "GLOBAL_PAIR_INDEX")
            item = run_pair(
                _seed(global_index, pair, events),
                q_bases=q_bases,
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                encoded=encoded,
                row_index=row_index,
                events=events,
                pp3_planes=pp3_planes,
                r22=r22,
                c22=c22,
                budget=budget,
            )
            item["shard_id"] = int(args.shard_id)
            items.append(item)
            if (local_index + 1) % 10 == 0:
                gc.collect()
                print(
                    f"GEN5_RESTORATION_GPU{args.shard_id}_PROGRESS="
                    f"{local_index + 1}/{restoration.SHARD_PAIR_COUNT}",
                    flush=True,
                )
        budget.assert_exact()
        torch.cuda.synchronize()
        after_signature = cuda_eq.model_parameter_signature(model)
        require(before_signature == after_signature, "MODEL_PARAMETER_MUTATION")

    require(
        [str(x["source_pair_id"]) for x in items] == list(restoration.shard_pairs(args.shard_id)),
        "WORKER_PAIR_ORDER",
    )
    require(
        sum(int(x["scientific_model_forward_count_this_run"]) for x in items)
        == WORKER_FORWARD_BUDGET,
        "WORKER_FORWARD_BUDGET",
    )
    args.shard_output.parent.mkdir(parents=True, exist_ok=True)
    args.shard_output.write_bytes(restoration.jsonl_bytes(items))
    args.shard_meta.write_bytes(restoration.canonical_json_bytes({
        "shard_id": int(args.shard_id),
        "pair_first": items[0]["source_pair_id"],
        "pair_last": items[-1]["source_pair_id"],
        "pair_count": len(items),
        "scientific_model_forward_count": WORKER_FORWARD_BUDGET,
        "cpu_scientific_model_forward_count": 0,
        "confirmatory_p_value_count": 0,
        "checkpoint_load_count": 1,
        "checkpoint_sha256": checkpoint_sha,
        "model_parameter_signature_before": before_signature,
        "model_parameter_signature_after": after_signature,
        "basis_geometry": basis_geometry,
        "tokenizer": tokenizer_provenance,
        "scientific_conclusion": None,
    }))
    print(f"RESULT=PASS_GEN5_PHASE1B_RESTORATION_WORKER_{args.shard_id}", flush=True)
    print("CONFIRMATORY_P_VALUE_COUNT=0", flush=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return restoration.read_jsonl(path)


def run_coordinator(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head)
    restoration.validate_necessity_prerequisite()
    restoration.validate_static_inputs()
    require(not args.output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 2, f"EXACT_TWO_GPU_REQUIRED:{torch.cuda.device_count()}")
    require(
        [torch.cuda.get_device_name(i) for i in range(2)] == ["Tesla T4", "Tesla T4"],
        "EXACT_TWO_T4_REQUIRED",
    )

    with tempfile.TemporaryDirectory(prefix="gen5-r22-restoration-shards-") as tmp:
        tmpdir = Path(tmp)
        procs = []
        for shard_id in (0, 1):
            shard_output = tmpdir / f"shard{shard_id}.jsonl"
            shard_meta = tmpdir / f"shard{shard_id}.meta.json"
            cmd = [
                sys.executable, "-u", "-m",
                "scripts.reason_router_gen5_phase1b_r22_restoration_fast_cuda_2gpu",
                "--worker",
                "--shard-id", str(shard_id),
                "--expected-head", args.expected_head,
                "--model-snapshot", str(args.model_snapshot),
                "--tokenizer-snapshot", str(args.tokenizer_snapshot),
                "--checkpoint", str(args.checkpoint),
                "--shard-output", str(shard_output),
                "--shard-meta", str(shard_meta),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(shard_id)
            procs.append((
                shard_id,
                shard_output,
                shard_meta,
                subprocess.Popen(cmd, cwd=ROOT, env=env),
            ))

        failures = []
        for shard_id, _, _, proc in procs:
            rc = proc.wait()
            if rc != 0:
                failures.append((shard_id, rc))
        require(not failures, f"WORKER_FAILURES:{failures}")

        shard_items = []
        shard_meta_rows = []
        for shard_id, shard_output, shard_meta, _ in procs:
            require(shard_output.is_file(), f"SHARD_OUTPUT_MISSING:{shard_id}")
            require(shard_meta.is_file(), f"SHARD_META_MISSING:{shard_id}")
            rows = _load_jsonl(shard_output)
            meta = json.loads(shard_meta.read_text(encoding="utf-8"))
            require(int(meta["shard_id"]) == shard_id, "SHARD_META_ID")
            require(int(meta["confirmatory_p_value_count"]) == 0, "SHARD_P_VALUE_FORBIDDEN")
            require(meta["scientific_conclusion"] is None, "SHARD_CONCLUSION_FORBIDDEN")
            require(int(meta["scientific_model_forward_count"]) == WORKER_FORWARD_BUDGET,
                    "SHARD_FORWARD_BUDGET")
            shard_items.append(rows)
            shard_meta_rows.append(meta)

        items = restoration.merge_shards(shard_items[0], shard_items[1])
        decision = restoration.confirmatory_decision(items)
        checkpoint_shas = {str(x["checkpoint_sha256"]) for x in shard_meta_rows}
        require(
            checkpoint_shas == {restoration.necessity.CHECKPOINT_SHA256},
            "CHECKPOINT_SHA_MERGE",
        )
        before = [str(x["model_parameter_signature_before"]) for x in shard_meta_rows]
        after = [str(x["model_parameter_signature_after"]) for x in shard_meta_rows]
        require(before == after, "PARAMETER_SIGNATURE_WORKERS")

        summary = {
            "schema_version": restoration.SUMMARY_SCHEMA,
            "result": restoration.RESULT_PASS,
            "execution_head": args.expected_head,
            "implementation_authority_commit": AUTHORITY_COMMIT,
            "necessity_evidence_freeze_commit": restoration.NECESSITY_EVIDENCE_FREEZE_COMMIT,
            "qualified_cuda_equivalence_artifact_freeze_commit":
                restoration.CUDA_EQ_ARTIFACT_FREEZE_COMMIT,
            "qualified_backend": QUALIFIED_BACKEND,
            "source_pair_count": restoration.PAIR_COUNT,
            "pair_id_first": "xg1_fact_8401",
            "pair_id_last": "xg1_fact_8700",
            "shard_topology": {
                "gpu0": ["xg1_fact_8401", "xg1_fact_8550"],
                "gpu1": ["xg1_fact_8551", "xg1_fact_8700"],
                "pairs_per_shard": restoration.SHARD_PAIR_COUNT,
                "worker_visible_device_count": 1,
                "physical_gpu_count": 2,
                "device_name": "Tesla T4",
            },
            "forwards_per_pair": restoration.FORWARDS_PER_PAIR,
            "scientific_model_forward_count_this_run":
                restoration.FULL_MODEL_FORWARD_BUDGET,
            "cuda_scientific_model_forward_count_this_run":
                restoration.FULL_MODEL_FORWARD_BUDGET,
            "cpu_scientific_model_forward_count_this_run": 0,
            "per_shard_forward_budget": restoration.SHARD_MODEL_FORWARD_BUDGET,
            "checkpoint_load_count_this_run": 2,
            "representative_checkpoint_sha256": restoration.necessity.CHECKPOINT_SHA256,
            "r22_basis_sha256": restoration.necessity.R22_SHA256,
            "c22_basis_sha256": restoration.necessity.C22_SHA256,
            "pp3_plus_sha256": pp3.PLANE_FILES["pp3_plus"][1],
            "pp3_minus_sha256": pp3.PLANE_FILES["pp3_minus"][1],
            "direction_order": list(restoration.DIRECTIONS),
            "epsilon": restoration.EPSILON,
            "q22_definition": "E_XG2_22-E_XG4_22",
            "conditions": ["B", "RR", "RC"],
            "decision": decision,
            "confirmatory_p_value_count": 1,
            "training_executed": False,
            "backward_executed": False,
            "task_heads_executed": False,
            "logits_read": False,
            "row_dropping_executed": False,
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "scientific_conclusion": decision["label"],
            "next_stage": (
                "GEN5_PHASE2_STATE_UPDATE_OWNERSHIP_IMPLEMENTATION_DESIGN"
                if decision["label"] == restoration.LABEL_SUPPORTED
                else "STOP_PHASE1B_MEDIATION_RESTORATION_NOT_ESTABLISHED"
            ),
        }
        restoration.write_outputs(args.output_dir, items=items, summary=summary)

    print("RESULT=" + summary["result"])
    print("PAIR_ID_FIRST=" + summary["pair_id_first"])
    print("PAIR_ID_LAST=" + summary["pair_id_last"])
    print(
        "CUDA_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["cuda_scientific_model_forward_count_this_run"])
    )
    print("CPU_SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=0")
    print("CONFIRMATORY_P_VALUE_COUNT=1")
    print("MEAN_Q_RR=" + repr(decision["mean_Q_RR"]))
    print("MEAN_S_R=" + repr(decision["mean_S_R"]))
    print("MEAN_D_SUF22=" + repr(decision["mean_D_SUF22"]))
    print(
        "P_ONE_SIDED_GREATER="
        + repr(decision["confirmatory_test"]["p_one_sided_greater"])
    )
    print("SCIENTIFIC_CONCLUSION=" + summary["scientific_conclusion"])
    print("NEXT_STAGE=" + summary["next_stage"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--shard-meta", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        require(args.shard_id in (0, 1), "SHARD_ID_REQUIRED")
        require(args.shard_output is not None and args.shard_meta is not None,
                "SHARD_PATHS_REQUIRED")
        require(args.output_dir is None, "WORKER_OUTPUT_DIR_FORBIDDEN")
    else:
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
        require(args.shard_output is None and args.shard_meta is None,
                "COORDINATOR_SHARD_PATH_FORBIDDEN")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
    else:
        run_coordinator(args)


if __name__ == "__main__":
    main()
