from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import types
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_generator_family_prevalence_kernel_compat as kernel_compat
from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import reason_router_gen5_phase1b_r22_c22_construction as construction
from scripts import reason_router_gen5_phase1b_r22_local_necessity_confirmation as necessity

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

AUTHORITY_COMMIT = "9770a2f0588353b0708b760e7fcce6083f4c7e06"
NECESSITY_IMPLEMENTATION_COMMIT = "130a3474cf83d1ba6080561afbf886614dde4786"
Q22_CORRECTION_COMMIT = "d31b9df1fddfb719d95a6d486ebb8f18b397f6ad"
R22_C22_FREEZE_COMMIT = "1d3542013934870aa9181d1bbaf565ff4724112c"

FROZEN_BLOBS = {
    "reports/reason_router_gen5_phase1b_q22_cuda_backend_equivalence_gate_authority_spec_candidate.md":
        "bb3f632f06f9ab5a9e0409662475ada0bc1670ed",
    "scripts/reason_router_gen5_phase1b_r22_local_necessity_confirmation.py":
        "50be219720b2809969b4f3caac25b2c25598a81b",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py":
        "4a8326c442b883510ba452f049c88f3163677fc5",
    "scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py":
        "b3b52c4116ca142fc1e31c4bdda57828c610d6e7",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py":
        "8c8d63cce182dfab66a632299e93cd5b25fc36af",
}

PAIR_ID = "xg1_fact_7801"
PAIR_INDEX = 0
CONDITIONS = necessity.CONDITIONS
DIRECTIONS = necessity.DIRECTIONS
EPSILON = necessity.EPSILON
K = necessity.K
LAYER22 = necessity.LAYER22
STATE_SHAPE = necessity.STATE_SHAPE
STATE_WIDTH = necessity.STATE_WIDTH

FORWARDS_PER_BACKEND = 120
TOTAL_EQUIVALENCE_MODEL_FORWARDS = 240

WRITE_ATOL = 1e-4
WRITE_RTOL = 1e-4
STATE_ATOL = 1e-4
STATE_RTOL = 1e-4
PE22_ATOL = 1e-4
F22_ATOL = 2e-4
J22_ATOL = 0.008
COEFFICIENT_ATOL = necessity.MATCH_TOL

ITEM_FILE = "q22_cuda_equivalence_items.jsonl"
SUMMARY_FILE = "q22_cuda_equivalence_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ITEM_SCHEMA = "gen5-phase1b-q22-cuda-equivalence-item-v1"
SUMMARY_SCHEMA = "gen5-phase1b-q22-cuda-equivalence-summary-v1"
MANIFEST_SCHEMA = "gen5-phase1b-q22-cuda-equivalence-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE1B_Q22_CUDA_BACKEND_EQUIVALENCE"


class EquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise EquivalenceError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EquivalenceError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    for ancestor, label in (
        (AUTHORITY_COMMIT, "AUTHORITY"),
        (NECESSITY_IMPLEMENTATION_COMMIT, "NECESSITY_IMPLEMENTATION"),
        (Q22_CORRECTION_COMMIT, "Q22_CORRECTION"),
        (R22_C22_FREEZE_COMMIT, "R22_C22_FREEZE"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )
    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB_DRIFT:{path}:{observed}")


def seed_for_pair(events: Mapping[Any, Any]) -> dict[str, Any]:
    require(construction.expected_pairs()[PAIR_INDEX] == PAIR_ID, "PAIR_BINDING")
    anchors = construction.phase1._anchors_for_pair(PAIR_ID, events)
    return {
        "family_key": "xg1",
        "source_pair_id": PAIR_ID,
        "pair_index": PAIR_INDEX,
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def model_parameter_signature(model: Any) -> str:
    h = hashlib.sha256()
    for name, parameter in model.mamba.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        tensor = parameter.detach().cpu().contiguous()
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes())
        h.update(b"\0")
    return h.hexdigest()


@contextmanager
def force_layer22_slow(mixer22: Any):
    sentinel = object()
    prior = mixer22.__dict__.get("forward", sentinel)

    def slow_forward_bound(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        return self.slow_forward(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )

    mixer22.forward = types.MethodType(slow_forward_bound, mixer22)
    try:
        yield
    finally:
        if prior is sentinel:
            del mixer22.__dict__["forward"]
        else:
            mixer22.forward = prior


@dataclass(frozen=True)
class SlowPending:
    s_prev: torch.Tensor
    g: torch.Tensor
    actual_w: torch.Tensor


class SlowLayer22EquivalenceCollector:
    def __init__(
        self,
        *,
        code: Any,
        mixer22: Any,
        anchor: int,
        condition: str,
        r22: torch.Tensor,
        c22: torch.Tensor,
    ) -> None:
        self.code = code
        self.mixer22_id = id(mixer22)
        self.anchor = int(anchor)
        self.target = self.anchor + core.TARGET_OFFSET
        self.indices = frozenset(range(self.anchor, self.anchor + 5))
        self.condition = condition
        self.r22 = r22
        self.c22 = c22
        self.pending: dict[int, SlowPending] = {}
        self.post_states: dict[int, torch.Tensor] = {}
        self.target_native_write: torch.Tensor | None = None
        self.target_actual_write: torch.Tensor | None = None
        self.target_plan: dict[str, Any] | None = None
        self.target_recurrence_residual: float | None = None
        self._prior_trace: Any = None
        self._used = False

    def _coordinate(self, frame: Any) -> int | None:
        if id(frame.f_locals.get("self")) != self.mixer22_id:
            return None
        token = frame.f_locals.get("i")
        require(type(token) is int and token >= 0, "SLOW_AMBIGUOUS_TOKEN")
        return int(token) if token in self.indices else None

    def _trace(self, frame: Any, event: str, arg: Any):
        del arg
        if frame.f_code is not self.code or event != "line":
            return self._trace

        if frame.f_lineno == measurement.RECURRENT_UPDATE_LINE:
            token = self._coordinate(frame)
            if token is None:
                return self._trace
            require(token not in self.pending and token not in self.post_states, "SLOW_DUPLICATE_PRE")
            s_prev = necessity.state_snapshot(frame.f_locals.get("ssm_state"), "SLOW_S_PREV")
            discrete_a = frame.f_locals.get("discrete_A")
            delta_b_u = frame.f_locals.get("deltaB_u")
            require(torch.is_tensor(discrete_a) and torch.is_tensor(delta_b_u), "SLOW_NATIVE_LOCALS")
            g = necessity.state_snapshot(discrete_a[:, :, token, :], "SLOW_G")
            native_w = necessity.state_snapshot(delta_b_u[:, :, token, :], "SLOW_NATIVE_W")
            actual_w = native_w

            if token == self.target:
                plan = necessity.intervention_vectors(native_w, self.r22, self.c22, self.condition)
                view = delta_b_u[:, :, token, :]
                before = delta_b_u.detach().clone()
                intended = (
                    plan["modified"]
                    .reshape(STATE_SHAPE)
                    .to(device=view.device, dtype=view.dtype)
                )
                view.copy_(intended)
                actual_w = necessity.state_snapshot(delta_b_u[:, :, token, :], "SLOW_ACTUAL_W")
                before[:, :, token, :].copy_(delta_b_u[:, :, token, :])
                require(torch.equal(before, delta_b_u), "SLOW_NON_TARGET_WRITE_CHANGED")
                self.target_native_write = native_w
                self.target_actual_write = actual_w
                self.target_plan = plan

            self.pending[token] = SlowPending(
                s_prev=s_prev,
                g=g,
                actual_w=actual_w,
            )
            return self._trace

        if frame.f_lineno == measurement.CAPTURE_LINE:
            token = self._coordinate(frame)
            if token is None:
                return self._trace
            require(token in self.pending and token not in self.post_states, "SLOW_POST_WITHOUT_PRE")
            pre = self.pending.pop(token)
            s_post = necessity.state_snapshot(frame.f_locals.get("ssm_state"), "SLOW_S_POST")
            recon = (
                pre.g.to(torch.float64) * pre.s_prev.to(torch.float64)
                + pre.actual_w.to(torch.float64)
            )
            residual = recon - s_post.to(torch.float64)
            denom = max(
                float(torch.linalg.vector_norm(recon).item()),
                float(torch.linalg.vector_norm(s_post.to(torch.float64)).item()),
                1e-12,
            )
            rel = float(torch.linalg.vector_norm(residual).item() / denom)
            require(rel <= necessity.RECURRENCE_REL_TOL, f"SLOW_RECURRENCE:{token}:{rel}")
            self.post_states[token] = s_post
            if token == self.target:
                self.target_recurrence_residual = rel
            return self._trace

        return self._trace

    @contextmanager
    def capture(self):
        require(not self._used, "SLOW_COLLECTOR_REUSE")
        self._used = True
        self._prior_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._prior_trace)
            require(not self.pending, f"SLOW_PENDING:{sorted(self.pending)}")
            require(set(self.post_states) == set(self.indices), "SLOW_POST_STATE_SET")
            require(self.target_native_write is not None, "SLOW_TARGET_NATIVE_MISSING")
            require(self.target_actual_write is not None, "SLOW_TARGET_ACTUAL_MISSING")
            require(self.target_plan is not None, "SLOW_TARGET_PLAN_MISSING")
            require(self.target_recurrence_residual is not None, "SLOW_TARGET_RECON_MISSING")


class Layer22FastScanCapture:
    def __init__(self, mixer22: Any) -> None:
        self.mixer22 = mixer22
        self.captured: tuple[torch.Tensor, ...] | None = None
        self._used = False

    @contextmanager
    def capture(self):
        import transformers.models.mamba.modeling_mamba as mm

        require(not self._used, "FAST_CAPTURE_REUSE")
        self._used = True
        original_scan = mm.selective_scan_fn
        original_cuda = self.mixer22.cuda_kernels_forward
        active = {"value": False}

        def scan_wrapper(*args, **kwargs):
            if active["value"]:
                require(self.captured is None, "FAST_LAYER22_SCAN_DUPLICATE")
                require(len(args) >= 8, "FAST_LAYER22_SCAN_ARGS")
                values = []
                for value in args[:8]:
                    require(torch.is_tensor(value), "FAST_LAYER22_SCAN_TENSOR")
                    values.append(value.detach().clone())
                self.captured = tuple(values)
            return original_scan(*args, **kwargs)

        def cuda_wrapper(
            _self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            require(not active["value"], "FAST_LAYER22_REENTRY")
            active["value"] = True
            try:
                return original_cuda(
                    hidden_states,
                    cache_params,
                    cache_position,
                    attention_mask,
                )
            finally:
                active["value"] = False

        mm.selective_scan_fn = scan_wrapper
        self.mixer22.cuda_kernels_forward = types.MethodType(cuda_wrapper, self.mixer22)
        try:
            yield self
        finally:
            mm.selective_scan_fn = original_scan
            if "cuda_kernels_forward" in self.mixer22.__dict__:
                del self.mixer22.__dict__["cuda_kernels_forward"]
            require(self.captured is not None, "FAST_LAYER22_SCAN_NOT_CAPTURED")


def _slice_last(tensor: torch.Tensor, end: int) -> torch.Tensor:
    return tensor[..., :end].contiguous()


def replay_q22_window(
    captured: tuple[torch.Tensor, ...],
    *,
    anchor: int,
    condition: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
    kernel_scan: Any,
    kernel_update: Any,
) -> dict[str, Any]:
    require(len(captured) == 8, "REPLAY_CAPTURE_WIDTH")
    u, delta, a_matrix, b_scan, c_scan, d_vector, gate, delta_bias = captured
    require(anchor >= 1, "REPLAY_ANCHOR")
    require(anchor + 4 < u.shape[-1], "REPLAY_WINDOW_RANGE")
    target = anchor + core.TARGET_OFFSET

    _, state = kernel_scan(
        _slice_last(u, anchor),
        _slice_last(delta, anchor),
        a_matrix,
        _slice_last(b_scan, anchor),
        _slice_last(c_scan, anchor),
        d_vector,
        _slice_last(gate, anchor),
        delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    require(torch.is_tensor(state), "REPLAY_PREFIX_STATE")

    post_states: dict[int, torch.Tensor] = {}
    native_write_cpu: torch.Tensor | None = None
    actual_write_cpu: torch.Tensor | None = None
    plan: dict[str, Any] | None = None

    for token in range(anchor, anchor + 5):
        if token != target:
            _ = kernel_update(
                state,
                u[..., token],
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
        else:
            pre = state.detach().clone()
            state_native = pre.clone()
            state_decay = pre.clone()

            _ = kernel_update(
                state_native,
                u[..., token],
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
            zero_u = torch.zeros_like(u[..., token])
            _ = kernel_update(
                state_decay,
                zero_u,
                delta[..., token],
                a_matrix,
                b_scan[..., token],
                c_scan[..., token],
                d_vector,
                gate[..., token],
                delta_bias,
                dt_softplus=True,
            )
            write_fast = (state_native - state_decay).detach()
            native_write_cpu = write_fast.cpu().to(torch.float32).contiguous()
            require(tuple(native_write_cpu.shape) == STATE_SHAPE, "FAST_WRITE_SHAPE")
            plan = necessity.intervention_vectors(
                native_write_cpu,
                r22,
                c22,
                condition,
            )
            modified_write = (
                plan["modified"]
                .reshape(STATE_SHAPE)
                .to(device=state_decay.device, dtype=state_decay.dtype)
            )
            state = (state_decay + modified_write).contiguous()
            actual_write_cpu = modified_write.detach().cpu().to(torch.float32).contiguous()

        post_states[token] = state.detach().cpu().to(torch.float32).contiguous().clone()

    require(native_write_cpu is not None, "FAST_NATIVE_WRITE_MISSING")
    require(actual_write_cpu is not None, "FAST_ACTUAL_WRITE_MISSING")
    require(plan is not None, "FAST_PLAN_MISSING")
    pe22 = necessity.q22_path_efficiency(post_states, anchor)
    return {
        "native_write": native_write_cpu,
        "actual_write": actual_write_cpu,
        "plan": plan,
        "post_states": post_states,
        "pe22": float(pe22),
    }


def _layer17_probe_hook(
    *,
    runtime_ctx: Mapping[str, Any],
    target: int,
    direction: torch.Tensor,
    orientation: int,
    plus_branch: bool,
):
    audit: dict[str, Any] = {}
    delta_h = (
        direction.detach().cpu().to(torch.float64)
        * (float(orientation) * 2.0 * EPSILON)
    ).contiguous()
    handle = transport_runtime.install_inproj_hook(
        runtime_ctx["mixer17"],
        token_index=target,
        strong_mask=runtime_ctx["strong_mask"],
        delta_h=delta_h,
        plus_branch=plus_branch,
        audit=audit,
    )
    return handle, audit


def run_reference_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    input_ids: torch.Tensor,
    anchor: int,
    direction: torch.Tensor,
    orientation: int,
    plus_branch: bool,
    condition: str,
    r22: torch.Tensor,
    c22: torch.Tensor,
    budget: Any,
) -> dict[str, Any]:
    target = int(anchor) + core.TARGET_OFFSET
    handle, layer17_audit = _layer17_probe_hook(
        runtime_ctx=runtime_ctx,
        target=target,
        direction=direction,
        orientation=orientation,
        plus_branch=plus_branch,
    )
    mixer22 = model.mamba.layers[LAYER22].mixer
    collector = SlowLayer22EquivalenceCollector(
        code=trace_code,
        mixer22=mixer22,
        anchor=int(anchor),
        condition=condition,
        r22=r22,
        c22=c22,
    )
    prior_trace = sys.gettrace()
    budget.consume()
    try:
        with force_layer22_slow(mixer22):
            with collector.capture():
                with torch.inference_mode():
                    _ = model.mamba(input_ids=input_ids.to("cuda:0"))
    finally:
        handle.remove()
    require(sys.gettrace() is prior_trace, "REFERENCE_TRACE_RESTORATION")
    require(bool(layer17_audit), "REFERENCE_LAYER17_AUDIT")
    require(int(layer17_audit["token_index"]) == target, "REFERENCE_LAYER17_TOKEN")
    require(collector.target_plan is not None, "REFERENCE_TARGET_PLAN")
    pe22 = necessity.q22_path_efficiency(collector.post_states, int(anchor))
    return {
        "pe22": float(pe22),
        "native_write": collector.target_native_write,
        "actual_write": collector.target_actual_write,
        "plan": collector.target_plan,
        "post_states": collector.post_states,
        "target_recurrence_relative_residual": float(collector.target_recurrence_residual),
        "layer17_probe_norm": float(layer17_audit["runtime_correction_l2"]),
        "target_token_index": int(target),
    }


def run_accelerated_branch(
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
    r22: torch.Tensor,
    c22: torch.Tensor,
    budget: Any,
) -> dict[str, Any]:
    target = int(anchor) + core.TARGET_OFFSET
    handle, layer17_audit = _layer17_probe_hook(
        runtime_ctx=runtime_ctx,
        target=target,
        direction=direction,
        orientation=orientation,
        plus_branch=plus_branch,
    )
    capture = Layer22FastScanCapture(model.mamba.layers[LAYER22].mixer)
    budget.consume()
    try:
        with capture.capture():
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids.to("cuda:0"))
    finally:
        handle.remove()
    require(bool(layer17_audit), "FAST_LAYER17_AUDIT")
    require(int(layer17_audit["token_index"]) == target, "FAST_LAYER17_TOKEN")
    replay = replay_q22_window(
        capture.captured or (),
        anchor=int(anchor),
        condition=condition,
        r22=r22,
        c22=c22,
        kernel_scan=kernels["selective_scan_fn"],
        kernel_update=kernels["selective_state_update"],
    )
    replay["layer17_probe_norm"] = float(layer17_audit["runtime_correction_l2"])
    replay["target_token_index"] = int(target)
    return replay


def _run_signed(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    orientation: int,
    branch_runner: Any,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[Any, Any],
    **kwargs,
) -> dict[str, Any]:
    pair = str(seed["source_pair_id"])
    anchors = construction.phase1._anchors_for_pair(pair, events)
    cells = construction.phase1._cells()
    branches: dict[str, Any] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        branches[role] = branch_runner(
            input_ids=construction.phase2._input_row(
                encoded,
                row_index,
                pair,
                cells[role],
            ),
            anchor=int(anchors[role]),
            direction=direction,
            orientation=orientation,
            plus_branch=plus_branch,
            condition=condition,
            **kwargs,
        )
    f22 = float(branches["tp"]["pe22"]) - float(branches["tm"]["pe22"])
    return {
        "orientation": int(orientation),
        "F22": f22,
        "_branches": branches,
    }


def _run_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    family: str,
    basis_index: int,
    condition: str,
    **kwargs,
) -> dict[str, Any]:
    positive = _run_signed(
        seed, direction, condition=condition, orientation=1, **kwargs
    )
    negative = _run_signed(
        seed, direction, condition=condition, orientation=-1, **kwargs
    )
    j22 = (float(positive["F22"]) - float(negative["F22"])) / (2.0 * EPSILON)
    return {
        "direction_key": f"{family}_{basis_index}",
        "basis_family": family,
        "basis_index": int(basis_index),
        "F22_plus": float(positive["F22"]),
        "F22_minus": float(negative["F22"]),
        "J22": float(j22),
        "J22_squared": float(j22 * j22),
        "_positive": positive,
        "_negative": negative,
    }


def run_backend_pair(
    seed: Mapping[str, Any],
    *,
    q_bases: Mapping[str, torch.Tensor],
    branch_runner: Any,
    **kwargs,
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        probes: list[dict[str, Any]] = []
        for family in ("xg2", "xg4"):
            for basis_index in range(K):
                probes.append(
                    _run_direction(
                        seed,
                        q_bases[family][:, basis_index],
                        family=family,
                        basis_index=basis_index,
                        condition=condition,
                        branch_runner=branch_runner,
                        **kwargs,
                    )
                )
        e2 = sum(float(row["J22_squared"]) for row in probes[:K]) / K
        e4 = sum(float(row["J22_squared"]) for row in probes[K:]) / K
        conditions.append(
            {
                "condition": condition,
                "direction_probes": probes,
                "E_XG2_22": float(e2),
                "E_XG4_22": float(e4),
                "Q22": float(e2 - e4),
            }
        )
    return conditions


def tensor_equivalence(
    reference: torch.Tensor,
    accelerated: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> dict[str, float]:
    ref = reference.detach().cpu().to(torch.float64).contiguous()
    acc = accelerated.detach().cpu().to(torch.float64).contiguous()
    require(tuple(ref.shape) == tuple(acc.shape), f"{label}_SHAPE")
    diff = float(torch.max(torch.abs(ref - acc)).item())
    scale = max(
        float(torch.max(torch.abs(ref)).item()),
        float(torch.max(torch.abs(acc)).item()),
        1.0,
    )
    limit = float(atol + rtol * scale)
    require(diff <= limit, f"{label}_EQUIVALENCE:{diff}:{limit}")
    return {"max_abs_diff": diff, "limit": limit, "scale": scale}


def j2_error_bound(reference_j: float) -> float:
    return 2.0 * abs(float(reference_j)) * J22_ATOL + J22_ATOL * J22_ATOL


def compare_backends(
    reference: Sequence[Mapping[str, Any]],
    accelerated: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    require(len(reference) == len(accelerated) == len(CONDITIONS), "CONDITION_COUNT")
    items: list[dict[str, Any]] = []
    maxima = {
        "write_max_abs_diff": 0.0,
        "post_state_max_abs_diff": 0.0,
        "pe22_abs_diff": 0.0,
        "f22_abs_diff": 0.0,
        "j22_abs_diff": 0.0,
        "coefficient_max_abs_diff": 0.0,
    }

    for ref_c, acc_c in zip(reference, accelerated, strict=True):
        condition = str(ref_c["condition"])
        require(condition == acc_c["condition"], "CONDITION_IDENTITY")
        ref_probes = ref_c["direction_probes"]
        acc_probes = acc_c["direction_probes"]
        direction_rows: list[dict[str, Any]] = []
        xg2_bounds: list[float] = []
        xg4_bounds: list[float] = []

        for d_index, (ref_d, acc_d) in enumerate(zip(ref_probes, acc_probes, strict=True)):
            require(ref_d["direction_key"] == acc_d["direction_key"], "DIRECTION_IDENTITY")
            jdiff = abs(float(ref_d["J22"]) - float(acc_d["J22"]))
            require(jdiff <= J22_ATOL, f"J22_EQUIVALENCE:{ref_d['direction_key']}:{jdiff}")
            maxima["j22_abs_diff"] = max(maxima["j22_abs_diff"], jdiff)
            bound = j2_error_bound(float(ref_d["J22"]))
            (xg2_bounds if d_index < K else xg4_bounds).append(bound)

            signed_rows: list[dict[str, Any]] = []
            for sign_name in ("_positive", "_negative"):
                ref_s = ref_d[sign_name]
                acc_s = acc_d[sign_name]
                fdiff = abs(float(ref_s["F22"]) - float(acc_s["F22"]))
                require(fdiff <= F22_ATOL, f"F22_EQUIVALENCE:{ref_d['direction_key']}:{sign_name}:{fdiff}")
                maxima["f22_abs_diff"] = max(maxima["f22_abs_diff"], fdiff)

                branch_rows: dict[str, Any] = {}
                for role in ("tp", "tm"):
                    rb = ref_s["_branches"][role]
                    ab = acc_s["_branches"][role]
                    require(rb["target_token_index"] == ab["target_token_index"], "TARGET_TOKEN_IDENTITY")
                    require(
                        abs(float(rb["layer17_probe_norm"]) - float(ab["layer17_probe_norm"])) <= 1e-12,
                        "LAYER17_PROBE_NORM_IDENTITY",
                    )
                    nw = tensor_equivalence(
                        rb["native_write"], ab["native_write"],
                        atol=WRITE_ATOL, rtol=WRITE_RTOL, label="NATIVE_WRITE22"
                    )
                    aw = tensor_equivalence(
                        rb["actual_write"], ab["actual_write"],
                        atol=WRITE_ATOL, rtol=WRITE_RTOL, label="ACTUAL_WRITE22"
                    )
                    maxima["write_max_abs_diff"] = max(
                        maxima["write_max_abs_diff"], nw["max_abs_diff"], aw["max_abs_diff"]
                    )

                    ref_a = rb["plan"]["a"].detach().cpu().to(torch.float64)
                    acc_a = ab["plan"]["a"].detach().cpu().to(torch.float64)
                    coeff_diff = float(torch.max(torch.abs(ref_a - acc_a)).item())
                    require(coeff_diff <= COEFFICIENT_ATOL, f"COEFFICIENT_EQUIVALENCE:{coeff_diff}")
                    maxima["coefficient_max_abs_diff"] = max(
                        maxima["coefficient_max_abs_diff"], coeff_diff
                    )

                    state_rows: list[dict[str, Any]] = []
                    ref_states = rb["post_states"]
                    acc_states = ab["post_states"]
                    require(set(ref_states) == set(acc_states), "POST_STATE_COORDINATES")
                    for token in sorted(ref_states):
                        se = tensor_equivalence(
                            ref_states[token], acc_states[token],
                            atol=STATE_ATOL, rtol=STATE_RTOL,
                            label=f"POST_STATE22_{token}",
                        )
                        maxima["post_state_max_abs_diff"] = max(
                            maxima["post_state_max_abs_diff"], se["max_abs_diff"]
                        )
                        state_rows.append(
                            {"token_index": int(token), "max_abs_diff": se["max_abs_diff"], "limit": se["limit"]}
                        )

                    pediff = abs(float(rb["pe22"]) - float(ab["pe22"]))
                    require(pediff <= PE22_ATOL, f"PE22_EQUIVALENCE:{pediff}")
                    maxima["pe22_abs_diff"] = max(maxima["pe22_abs_diff"], pediff)
                    branch_rows[role] = {
                        "target_token_index": int(rb["target_token_index"]),
                        "native_write_max_abs_diff": nw["max_abs_diff"],
                        "native_write_limit": nw["limit"],
                        "actual_write_max_abs_diff": aw["max_abs_diff"],
                        "actual_write_limit": aw["limit"],
                        "coefficient_max_abs_diff": coeff_diff,
                        "post_state_diffs": state_rows,
                        "pe22_reference": float(rb["pe22"]),
                        "pe22_accelerated": float(ab["pe22"]),
                        "pe22_abs_diff": pediff,
                    }

                signed_rows.append(
                    {
                        "orientation": 1 if sign_name == "_positive" else -1,
                        "F22_reference": float(ref_s["F22"]),
                        "F22_accelerated": float(acc_s["F22"]),
                        "F22_abs_diff": fdiff,
                        "branches": branch_rows,
                    }
                )

            direction_rows.append(
                {
                    "direction_key": ref_d["direction_key"],
                    "J22_reference": float(ref_d["J22"]),
                    "J22_accelerated": float(acc_d["J22"]),
                    "J22_abs_diff": jdiff,
                    "J22_atol": J22_ATOL,
                    "J22_squared_error_bound": bound,
                    "signed": signed_rows,
                }
            )

        b_e2 = sum(xg2_bounds) / K
        b_e4 = sum(xg4_bounds) / K
        b_q = b_e2 + b_e4
        e2diff = abs(float(ref_c["E_XG2_22"]) - float(acc_c["E_XG2_22"]))
        e4diff = abs(float(ref_c["E_XG4_22"]) - float(acc_c["E_XG4_22"]))
        qdiff = abs(float(ref_c["Q22"]) - float(acc_c["Q22"]))
        require(e2diff <= b_e2, f"E_XG2_22_EQUIVALENCE:{condition}:{e2diff}:{b_e2}")
        require(e4diff <= b_e4, f"E_XG4_22_EQUIVALENCE:{condition}:{e4diff}:{b_e4}")
        require(qdiff <= b_q, f"Q22_EQUIVALENCE:{condition}:{qdiff}:{b_q}")

        items.append(
            {
                "schema_version": ITEM_SCHEMA,
                "source_pair_id": PAIR_ID,
                "condition": condition,
                "direction_comparisons": direction_rows,
                "E_XG2_22_reference": float(ref_c["E_XG2_22"]),
                "E_XG2_22_accelerated": float(acc_c["E_XG2_22"]),
                "E_XG2_22_abs_diff": e2diff,
                "E_XG2_22_error_bound": b_e2,
                "E_XG4_22_reference": float(ref_c["E_XG4_22"]),
                "E_XG4_22_accelerated": float(acc_c["E_XG4_22"]),
                "E_XG4_22_abs_diff": e4diff,
                "E_XG4_22_error_bound": b_e4,
                "Q22_reference": float(ref_c["Q22"]),
                "Q22_accelerated": float(acc_c["Q22"]),
                "Q22_abs_diff": qdiff,
                "Q22_error_bound": b_q,
                "equivalence_pass": True,
            }
        )

    return items, maxima


def write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name + ".staging-", dir=str(output_dir.parent)) as tmp:
        staging = Path(tmp)
        primary = {
            ITEM_FILE: jsonl_bytes(items),
            SUMMARY_FILE: canonical_json_bytes(summary),
        }
        for name, raw in primary.items():
            (staging / name).write_bytes(raw)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "result": RESULT_PASS,
            "files": {
                name: {"sha256": sha256_bytes(raw), "bytes": len(raw)}
                for name, raw in sorted(primary.items())
            },
            "raw_native_vectors_persisted": False,
            "raw_post_state_vectors_persisted": False,
            "scientific_p_value_count": 0,
        }
        manifest_raw = canonical_json_bytes(manifest)
        (staging / MANIFEST_FILE).write_bytes(manifest_raw)
        checksum_raw = "".join(
            f"{sha256_bytes(raw)}  {name}\n"
            for name, raw in sorted({**primary, MANIFEST_FILE: manifest_raw}.items())
        ).encode("utf-8")
        (staging / CHECKSUM_FILE).write_bytes(checksum_raw)
        required = {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE, CHECKSUM_FILE}
        require({p.name for p in staging.iterdir() if p.is_file()} == required, "OUTPUT_FILE_SET")
        output_dir.mkdir(parents=False, exist_ok=False)
        for name in sorted(required):
            (output_dir / name).write_bytes((staging / name).read_bytes())


def run_equivalence(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    backend.runtime_gate()
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    r22, c22, basis_geometry = necessity.load_owner_bases()
    q_bases = necessity.load_q_bases()

    with backend.parent_runtime_rebind():
        rows, encoded, events, row_index, tokenizer_provenance = construction.load_inputs(tokenizer_snapshot)
        del rows
        seed = seed_for_pair(events)
        trace_code, trace_line = measurement._resolve_and_validate_runtime_binding()
        require(trace_line == measurement.CAPTURE_LINE, "TRACE_LINE_BINDING")

        kernels = kernel_compat.load_exact_fast_kernels()
        with kernel_compat.exact_transformers_kernel_loader(kernels) as constructor_calls:
            model, checkpoint_sha = parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
            require(checkpoint_sha == necessity.CHECKPOINT_SHA256, "CHECKPOINT_SHA256")
            runtime_ctx = transport_runtime.validate_runtime_components(model)

        counts = Counter(constructor_calls)
        require(set(counts) == {"causal-conv1d", "mamba-ssm"}, f"CONSTRUCTOR_KERNEL_NAMES:{dict(counts)}")
        require(
            counts["causal-conv1d"] > 0 and counts["causal-conv1d"] == counts["mamba-ssm"],
            f"CONSTRUCTOR_KERNEL_COUNTS:{dict(counts)}",
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(torch.device("cuda:0"))
        model.eval()
        require(all(p.device.type == "cuda" for p in model.mamba.parameters()), "MODEL_NOT_CUDA")
        signature_before = model_parameter_signature(model)

        reference_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        reference = run_backend_pair(
            seed,
            q_bases=q_bases,
            branch_runner=run_reference_branch,
            model=model,
            runtime_ctx=runtime_ctx,
            trace_code=trace_code,
            encoded=encoded,
            row_index=row_index,
            events=events,
            r22=r22,
            c22=c22,
            budget=reference_budget,
        )
        reference_budget.assert_exact()
        torch.cuda.synchronize()

        accelerated_budget = parent.ForwardBudget(FORWARDS_PER_BACKEND)
        accelerated = run_backend_pair(
            seed,
            q_bases=q_bases,
            branch_runner=run_accelerated_branch,
            model=model,
            runtime_ctx=runtime_ctx,
            kernels=kernels,
            encoded=encoded,
            row_index=row_index,
            events=events,
            r22=r22,
            c22=c22,
            budget=accelerated_budget,
        )
        accelerated_budget.assert_exact()
        torch.cuda.synchronize()

        signature_after = model_parameter_signature(model)
        require(signature_before == signature_after, "MODEL_PARAMETER_MUTATION")

    items, maxima = compare_backends(reference, accelerated)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "authority_commit": AUTHORITY_COMMIT,
        "necessity_implementation_freeze_commit": NECESSITY_IMPLEMENTATION_COMMIT,
        "q22_correction_commit": Q22_CORRECTION_COMMIT,
        "source_pair_id": PAIR_ID,
        "source_population_role": "prior_construction_cohort_only",
        "necessity_confirmation_population_loaded": False,
        "restoration_confirmation_population_loaded": False,
        "reference_backend": "shared_fast_upstream_plus_layer22_transformers_5_0_0_slow_forward_cuda",
        "accelerated_backend": "frozen_cuda_scan_capture_plus_layer22_state_replay",
        "reference_cuda_model_forward_count": FORWARDS_PER_BACKEND,
        "accelerated_cuda_model_forward_count": FORWARDS_PER_BACKEND,
        "total_cuda_model_forward_count": TOTAL_EQUIVALENCE_MODEL_FORWARDS,
        "cpu_model_forward_count": 0,
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "epsilon": EPSILON,
        "r22_sha256": necessity.R22_SHA256,
        "c22_sha256": necessity.C22_SHA256,
        "basis_geometry": basis_geometry,
        "checkpoint_sha256": checkpoint_sha,
        "runtime": dict(backend.EXPECTED_RUNTIME),
        "cuda_runtime": backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": backend.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(backend.EXPECTED_CAPABILITY),
        "kernels_version": backend.KERNELS_VERSION,
        "mamba_kernel_revision": backend.MAMBA_REV,
        "causal_conv1d_kernel_revision": backend.CONV_REV,
        "mamba_binary_sha256": backend.MAMBA_BINARY_SHA256,
        "causal_conv1d_binary_sha256": backend.CONV_BINARY_SHA256,
        "prospective_tolerances": {
            "write_atol": WRITE_ATOL,
            "write_rtol": WRITE_RTOL,
            "state_atol": STATE_ATOL,
            "state_rtol": STATE_RTOL,
            "pe22_atol": PE22_ATOL,
            "f22_atol": F22_ATOL,
            "j22_atol": J22_ATOL,
        },
        "observed_maxima": maxima,
        "model_parameter_signature_before": signature_before,
        "model_parameter_signature_after": signature_after,
        "tokenizer": tokenizer_provenance,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "next_stage": "FULL_NECESSITY_CUDA_EXECUTION_AUTHORITY",
    }
    write_outputs(output_dir, items=items, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen5 Phase1B Q22 CUDA backend equivalence gate on xg1_fact_7801: "
            "120 CUDA reference forwards plus 120 accelerated CUDA forwards."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_equivalence(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + summary["result"])
    print("SOURCE_PAIR_ID=" + summary["source_pair_id"])
    print("REFERENCE_CUDA_MODEL_FORWARD_COUNT=120")
    print("ACCELERATED_CUDA_MODEL_FORWARD_COUNT=120")
    print("TOTAL_CUDA_MODEL_FORWARD_COUNT=240")
    print("CPU_MODEL_FORWARD_COUNT=0")
    print("NECESSITY_CONFIRMATION_POPULATION_LOADED=False")
    print("RESTORATION_CONFIRMATION_POPULATION_LOADED=False")
    print("SCIENTIFIC_P_VALUE_COUNT=0")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
