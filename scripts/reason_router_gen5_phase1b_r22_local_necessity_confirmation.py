from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from scripts import reason_router_gen4_k_directional_alignment_transport_core as core
from scripts import reason_router_gen4_k_directional_alignment_transport_runner as parent
from scripts import reason_router_gen4_k_directional_alignment_transport_runtime as transport_runtime
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import reason_router_gen4_pp3_necessity_fast_cuda as pp3
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1 as phase1
from scripts import reason_router_gen5_phase1b_r22_c22_construction as construction

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

IMPLEMENTATION_AUTHORITY_COMMIT = "eed071f4dc93f49973da8b2313d7d1b390e0e5d1"
Q22_CORRECTION_COMMIT = "d31b9df1fddfb719d95a6d486ebb8f18b397f6ad"
R22_C22_FREEZE_COMMIT = "1d3542013934870aa9181d1bbaf565ff4724112c"
PHASE1B_DESIGN_COMMIT = "c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8"
STATIC_PREPARATION_FREEZE_COMMIT = "5c0d91959af1f502b667ed6ab815c949c1043cbf"

FROZEN_BLOBS = {
    "reports/reason_router_gen5_phase1b_r22_local_necessity_confirmation_implementation_authority_spec_candidate.md":
        "bf50fd336aa28cb527846f59cbcafe11a6604524",
    "reports/reason_router_gen5_phase1b_downstream_q22_causal_order_correction_spec_candidate.md":
        "027b4072b42482a5dc5f7d08d629a1760ed499c6",
    "reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/r22_basis.f64le":
        "0fc55904c7bd8c7e02f37aa5546c8089812279b4",
    "reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/c22_basis.f64le":
        "3daa1e87d071e9c85436a7e1e1f1cffa8716af99",
    "reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/construction_summary.json":
        "40d1bb80422c52d7b463887bd7cbbd095e54ea95",
    "reports/reason_router_gen5_phase1b_static_preparation_c8dc7a4_v1/static_preparation_summary.json":
        "4b6edf39860e6cb043137d2c109a39405f9ad6a1",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py":
        "d98b2dcd3436433c04bb56ecc57dec4240abe820",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py":
        "3677dd83950789e41417c3a1ffaf70b82d7003ad",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py":
        "8c8d63cce182dfab66a632299e93cd5b25fc36af",
    "scripts/reason_router_gen4_xg2_xg4_local_jacobian_fast_cuda.py":
        "1749e4614f0e50d9c9bf551f672497321a5a6253",
    "scripts/reason_router_gen4_family_subspace_sensitivity_fast_cuda.py":
        "03f3bf1482913bce30bfdb665ab223a67e6e4159",
    "scripts/reason_router_gen4_xg2_basis_cross_family_holdout_fast_cuda.py":
        "2d35e5ed936fd37f4ecfc063e060290304c6bf10",
}

DATA_ROOT = Path("data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1")
SOURCE_FILE = "structured_source_facts.jsonl"
ROWS_FILE = "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
ELIGIBILITY_FILE = "tokenizer_eligibility_summary.json"

STATIC_INPUT_SHA256 = {
    SOURCE_FILE: "813534ffa5753dcf84637c43e177a6f96dac45230ceb3326747a528330e5b285",
    ROWS_FILE: "cf27ab041f77302562fd4cdd3fa4f2fc24e4b3f8f7367460512476b77efb1d14",
    STRUCTURAL_FILE: "043fb36890b2663e430fc018b7a72fc1d9872b7bed15c019e2c9e8c626d98b58",
    ANCHOR_FILE: "5ed5dc57b95df7121b301db2414c886655771dc77a53e0c6cbaad210d4683f88",
    ELIGIBILITY_FILE: "752361c5367a68ea460f10dd554fd385d325d2c9434e998714366723c3004373",
}

R22_ROOT = Path("reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1")
R22_PATH = R22_ROOT / "r22_basis.f64le"
C22_PATH = R22_ROOT / "c22_basis.f64le"
R22_SHA256 = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA256 = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"

CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
NATIVE_BACKBONE_SIGNATURE_SHA256 = "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"

PAIR_FIRST = 8101
PAIR_LAST = 8400
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

LAYER22 = 22
STATE_SHAPE = (1, 1536, 16)
STATE_WIDTH = 1536 * 16
RANK = 2
BASIS_ATOL = 1e-10
RECURRENCE_REL_TOL = 5e-6
MATCH_TOL = 5e-6
EPSILON = 0.025
K = 5
CONDITIONS = ("native22", "r22_neutralized", "c22_coefficient_control")
BRANCHES = ("tp", "tm")
DIRECTIONS = tuple([f"xg2_{i}" for i in range(K)] + [f"xg4_{i}" for i in range(K)])
FORWARDS_PER_SIGNED = 2
FORWARDS_PER_DIRECTION = 4
FORWARDS_PER_CONDITION = 40
FORWARDS_PER_PAIR = 120
FULL_FORWARD_BUDGET = 36000

ITEM_FILE = "r22_local_necessity_items.jsonl"
SUMMARY_FILE = "r22_local_necessity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen5-phase1b-r22-local-necessity-item-v1"
SUMMARY_SCHEMA = "gen5-phase1b-r22-local-necessity-summary-v1"
MANIFEST_SCHEMA = "gen5-phase1b-r22-local-necessity-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE1B_R22_LOCAL_NECESSITY_CONFIRMATION"
LABEL_SUPPORTED = "GEN5_R22_LOCAL_NECESSITY_OVER_MATCHED_C22_CONTROL_SUPPORTED"
LABEL_NOT_ESTABLISHED = "GEN5_R22_LOCAL_NECESSITY_NOT_ESTABLISHED"


class NecessityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise NecessityError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NecessityError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{index}" for index in range(PAIR_FIRST, PAIR_LAST + 1))


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    for ancestor, label in (
        (IMPLEMENTATION_AUTHORITY_COMMIT, "IMPLEMENTATION_AUTHORITY"),
        (Q22_CORRECTION_COMMIT, "Q22_CORRECTION"),
        (R22_C22_FREEZE_COMMIT, "R22_C22_FREEZE"),
        (PHASE1B_DESIGN_COMMIT, "PHASE1B_DESIGN"),
        (STATIC_PREPARATION_FREEZE_COMMIT, "STATIC_PREPARATION"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )
    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB_DRIFT:{path}:{observed}")


def validate_static_inputs() -> None:
    for name, expected in STATIC_INPUT_SHA256.items():
        path = ROOT / DATA_ROOT / name
        require(path.is_file(), f"STATIC_INPUT_MISSING:{name}")
        require(sha256_file(path) == expected, f"STATIC_INPUT_SHA256:{name}")

    structural = json.loads(
        (ROOT / DATA_ROOT / STRUCTURAL_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        structural.get("result") == "PASS_GEN5_PHASE1B_XG1_STRUCTURAL_HOLDOUT",
        "STRUCTURAL_RESULT",
    )
    require(structural.get("role") == "necessity_confirmation", "STRUCTURAL_ROLE")
    require(structural.get("pair_id_first") == "xg1_fact_8101", "STRUCTURAL_FIRST")
    require(structural.get("pair_id_last") == "xg1_fact_8400", "STRUCTURAL_LAST")
    require(structural.get("source_pair_count") == PAIR_COUNT, "STRUCTURAL_PAIR_COUNT")
    require(structural.get("row_count") == ROW_COUNT, "STRUCTURAL_ROW_COUNT")
    require(structural.get("primary_p_value_count") == 1, "STRUCTURAL_P_VALUE_COUNT")
    require(
        structural.get("primary_endpoint")
        == "D_NEC22=Q_C22_CONTROL-Q_R22_NEUTRALIZED",
        "STRUCTURAL_ENDPOINT",
    )
    for key in (
        "labels_present",
        "response_fields_present",
        "endpoint_values_present",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
        "training_executed",
        "backward_executed",
        "R22_construction_allowed",
        "C22_construction_allowed",
        "construction_responses_access_allowed",
        "cohort_replacement_allowed",
        "row_filtering_allowed",
    ):
        require(structural.get(key) is False, f"STRUCTURAL_BOUNDARY:{key}")

    eligibility = json.loads(
        (ROOT / DATA_ROOT / ELIGIBILITY_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        eligibility.get("result")
        == "PASS_GEN5_PHASE1B_TOKENIZER_ANCHOR_ELIGIBILITY",
        "ELIGIBILITY_RESULT",
    )
    require(eligibility.get("role") == "necessity_confirmation", "ELIGIBILITY_ROLE")
    require(
        eligibility.get("eligible_anchor_row_count") == ROW_COUNT,
        "ELIGIBILITY_ANCHOR_COUNT",
    )
    require(
        eligibility.get("identity_name_coordinate_mismatch_count") == 0,
        "ELIGIBILITY_COORDINATE_MISMATCH",
    )


def pair_order(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    seen: set[str] = set()
    order: list[str] = []
    for row in rows:
        pair = str(row["source_pair_id"])
        if pair not in seen:
            seen.add(pair)
            order.append(pair)
    require(tuple(order) == expected_pairs(), "PAIR_ORDER")
    return tuple(order)


def load_inputs(tokenizer_snapshot: str | Path | None):
    validate_static_inputs()
    rows = adapter.validate_gen4_rows(
        read_jsonl(ROOT / DATA_ROOT / ROWS_FILE),
        require_canonical_shape=True,
    )
    pairs = pair_order(rows)
    require(len(rows) == ROW_COUNT, "ROW_COUNT")
    tokenizer, tokenizer_provenance = tokenizer_gate.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    encoded = adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = read_jsonl(ROOT / DATA_ROOT / ANCHOR_FILE)
    require(len(event_rows) == ROW_COUNT, "ANCHOR_ROW_COUNT")
    events = parent.event_lookup(event_rows)
    parent.validate_transport_event_plan(pairs, events)
    row_index = parent.build_row_index(rows)
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, events, row_index, tokenizer_provenance


def _basis_from_bytes(raw: bytes, label: str) -> torch.Tensor:
    require(len(raw) == STATE_WIDTH * RANK * 8, f"{label}_BYTES")
    values = struct.unpack(f"<{STATE_WIDTH * RANK}d", raw)
    matrix = (
        torch.tensor(values, dtype=torch.float64)
        .reshape(RANK, STATE_WIDTH)
        .T
        .contiguous()
    )
    require(tuple(matrix.shape) == (STATE_WIDTH, RANK), f"{label}_SHAPE")
    require(bool(torch.isfinite(matrix).all().item()), f"{label}_FINITE")
    return matrix


def load_owner_bases() -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    r_path = ROOT / R22_PATH
    c_path = ROOT / C22_PATH
    require(r_path.is_file() and sha256_file(r_path) == R22_SHA256, "R22_IDENTITY")
    require(c_path.is_file() and sha256_file(c_path) == C22_SHA256, "C22_IDENTITY")
    r22 = _basis_from_bytes(r_path.read_bytes(), "R22")
    c22 = _basis_from_bytes(c_path.read_bytes(), "C22")
    eye = torch.eye(RANK, dtype=torch.float64)
    r_orth = float(torch.max(torch.abs(r22.T @ r22 - eye)).item())
    c_orth = float(torch.max(torch.abs(c22.T @ c22 - eye)).item())
    cross = float(torch.max(torch.abs(r22.T @ c22)).item())
    require(r_orth <= BASIS_ATOL, f"R22_ORTHONORMALITY:{r_orth}")
    require(c_orth <= BASIS_ATOL, f"C22_ORTHONORMALITY:{c_orth}")
    require(cross <= BASIS_ATOL, f"R22_C22_CROSS:{cross}")
    return r22, c22, {
        "r22_orthonormality_max_abs": r_orth,
        "c22_orthonormality_max_abs": c_orth,
        "r22_c22_cross_max_abs": cross,
    }


def load_q_bases() -> dict[str, torch.Tensor]:
    bases = pp3.load_bases()
    require(set(bases) == {"xg2", "xg4"}, "Q_BASIS_FAMILIES")
    for family in ("xg2", "xg4"):
        require(tuple(bases[family].shape) == (core.EXPECTED_STRONG_COUNT, K), f"Q_BASIS_SHAPE:{family}")
    return bases


def intervention_vectors(
    native_w: torch.Tensor,
    r22: torch.Tensor,
    c22: torch.Tensor,
    condition: str,
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    w = native_w.detach().cpu().to(torch.float64).contiguous().reshape(-1).clone()
    require(w.numel() == STATE_WIDTH, "WRITE_WIDTH")
    require(tuple(r22.shape) == (STATE_WIDTH, RANK), "R22_SHAPE")
    require(tuple(c22.shape) == (STATE_WIDTH, RANK), "C22_SHAPE")
    require(bool(torch.isfinite(w).all().item()), "WRITE_NONFINITE")
    a = (r22.T @ w).contiguous()
    r_delta = (r22 @ a).contiguous()
    c_delta = (c22 @ a).contiguous()
    r_norm = float(torch.linalg.vector_norm(r_delta).item())
    c_norm = float(torch.linalg.vector_norm(c_delta).item())
    equality_residual = abs(r_norm - c_norm)
    equality_tol = max(1e-12, 1e-10 * max(r_norm, c_norm, 1.0))
    require(equality_residual <= equality_tol, f"COEFFICIENT_TRANSFER_NORM:{equality_residual}:{equality_tol}")

    if condition == "native22":
        modified = w.clone()
        applied = torch.zeros_like(w)
    elif condition == "r22_neutralized":
        modified = (w - r_delta).contiguous()
        applied = (-r_delta).contiguous()
    else:
        modified = (w - c_delta).contiguous()
        applied = (-c_delta).contiguous()

    return {
        "native": w,
        "modified": modified,
        "a": a,
        "r_delta": r_delta,
        "c_delta": c_delta,
        "applied": applied,
        "r_correction_l2": r_norm,
        "c_correction_l2": c_norm,
        "norm_equality_residual": equality_residual,
        "norm_equality_tolerance": equality_tol,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return sha256_bytes(raw)


def state_snapshot(value: Any, label: str) -> torch.Tensor:
    require(torch.is_tensor(value), f"{label}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(tuple(out.shape) == STATE_SHAPE, f"{label}_SHAPE:{tuple(out.shape)}")
    require(out.dtype == torch.float32, f"{label}_DTYPE:{out.dtype}")
    require(bool(torch.isfinite(out).all().item()), f"{label}_NONFINITE")
    return out


@dataclass(frozen=True)
class _PendingUpdate:
    s_prev: torch.Tensor
    g: torch.Tensor
    w_actual: torch.Tensor


class Layer22Q22Collector:
    def __init__(
        self,
        *,
        code: Any,
        update_line: int,
        readout_line: int,
        mixer22: Any,
        anchor: int,
        condition: str,
        r22: torch.Tensor,
        c22: torch.Tensor,
    ) -> None:
        require(condition in CONDITIONS, f"CONDITION:{condition}")
        require(type(anchor) is int and anchor >= 1, "ANCHOR")
        require(readout_line > update_line > 0, "TRACE_LINES")
        self.code = code
        self.update_line = update_line
        self.readout_line = readout_line
        self.mixer22_id = id(mixer22)
        self.anchor = anchor
        self.target = anchor + core.TARGET_OFFSET
        self.indices = frozenset(range(anchor, anchor + 5))
        self.condition = condition
        self.r22 = r22
        self.c22 = c22
        self._pending: dict[int, _PendingUpdate] | None = None
        self.post_states: dict[int, torch.Tensor] | None = None
        self.target_audit: dict[str, Any] | None = None
        self._prior_trace: Any = None
        self._used = False

    def _coordinate(self, frame: Any) -> int | None:
        if id(frame.f_locals.get("self")) != self.mixer22_id:
            return None
        token_index = frame.f_locals.get("i")
        require(type(token_index) is int and token_index >= 0, "AMBIGUOUS_TOKEN_INDEX")
        if token_index not in self.indices:
            return None
        return int(token_index)

    def _trace(self, frame: Any, event: str, arg: Any):
        del arg
        if frame.f_code is not self.code or event != "line":
            return self._trace

        if frame.f_lineno == self.update_line:
            token = self._coordinate(frame)
            if token is None:
                return self._trace
            require(self._pending is not None and self.post_states is not None, "TRACE_NOT_ACTIVE")
            require(token not in self._pending and token not in self.post_states, "DUPLICATE_PRE")

            s_prev = state_snapshot(frame.f_locals.get("ssm_state"), "S_PREV")
            discrete_a = frame.f_locals.get("discrete_A")
            deltab_u = frame.f_locals.get("deltaB_u")
            require(torch.is_tensor(discrete_a), "DISCRETE_A_MISSING")
            require(torch.is_tensor(deltab_u), "DELTAB_U_MISSING")
            g = state_snapshot(discrete_a[:, :, token, :], "G")
            native_w = state_snapshot(deltab_u[:, :, token, :], "W_NATIVE")

            actual_w = native_w
            if token == self.target:
                plan = intervention_vectors(native_w, self.r22, self.c22, self.condition)
                target_view = deltab_u[:, :, token, :]
                before_other = deltab_u.detach().clone()
                intended = (
                    plan["modified"]
                    .reshape(STATE_SHAPE)
                    .to(device=target_view.device, dtype=target_view.dtype)
                )
                target_view.copy_(intended)
                actual_w = state_snapshot(deltab_u[:, :, token, :], "W_ACTUAL")
                applied = (
                    actual_w.to(torch.float64).reshape(-1)
                    - native_w.to(torch.float64).reshape(-1)
                )
                applied_residual = float(
                    torch.max(torch.abs(applied - plan["applied"])).item()
                )
                require(applied_residual <= MATCH_TOL, f"WRITE_APPLIED_RESIDUAL:{applied_residual}")
                # Restore target in the reference clone before exact non-target comparison.
                before_other[:, :, token, :].copy_(deltab_u[:, :, token, :])
                require(torch.equal(before_other, deltab_u), "NON_TARGET_WRITE_CHANGED")

                modified_flat = actual_w.to(torch.float64).reshape(-1)
                r_after = float(torch.linalg.vector_norm(self.r22.T @ modified_flat).item())
                c_after = float(torch.linalg.vector_norm(self.c22.T @ modified_flat).item())
                self.target_audit = {
                    "condition": self.condition,
                    "target_token_index": int(token),
                    "native_write_sha256": tensor_sha256(native_w),
                    "modified_write_sha256": tensor_sha256(actual_w),
                    "r22_coefficients": [float(v) for v in plan["a"].tolist()],
                    "native_r22_projection_l2": float(torch.linalg.vector_norm(plan["a"]).item()),
                    "native_c22_projection_l2": float(
                        torch.linalg.vector_norm(
                            self.c22.T @ native_w.to(torch.float64).reshape(-1)
                        ).item()
                    ),
                    "post_r22_projection_l2": r_after,
                    "post_c22_projection_l2": c_after,
                    "r22_candidate_correction_l2": float(plan["r_correction_l2"]),
                    "c22_candidate_correction_l2": float(plan["c_correction_l2"]),
                    "coefficient_transfer_norm_equality_residual": float(plan["norm_equality_residual"]),
                    "coefficient_transfer_norm_equality_tolerance": float(plan["norm_equality_tolerance"]),
                    "actual_write_change_l2": float(torch.linalg.vector_norm(applied).item()),
                    "applied_correction_max_abs_residual": applied_residual,
                }

            self._pending[token] = _PendingUpdate(s_prev=s_prev, g=g, w_actual=actual_w)
            return self._trace

        if frame.f_lineno == self.readout_line:
            token = self._coordinate(frame)
            if token is None:
                return self._trace
            require(self._pending is not None and self.post_states is not None, "TRACE_NOT_ACTIVE")
            require(token in self._pending and token not in self.post_states, "POST_WITHOUT_PRE")
            pre = self._pending.pop(token)
            s_post = state_snapshot(frame.f_locals.get("ssm_state"), "S_POST")
            recon = (
                pre.g.to(torch.float64) * pre.s_prev.to(torch.float64)
                + pre.w_actual.to(torch.float64)
            )
            residual = recon - s_post.to(torch.float64)
            denom = max(
                float(torch.linalg.vector_norm(recon).item()),
                float(torch.linalg.vector_norm(s_post.to(torch.float64)).item()),
                1e-12,
            )
            rel = float(torch.linalg.vector_norm(residual).item() / denom)
            require(rel <= RECURRENCE_REL_TOL, f"RECURRENCE_RECONSTRUCTION:{token}:{rel}")
            self.post_states[token] = s_post
            if token == self.target:
                require(self.target_audit is not None, "TARGET_AUDIT_MISSING")
                self.target_audit["target_post_state_sha256"] = tensor_sha256(s_post)
                self.target_audit["target_recurrence_relative_residual"] = rel
            return self._trace

        return self._trace

    @contextmanager
    def capture(self):
        require(not self._used, "OBSERVER_REUSE")
        self._used = True
        self._pending = {}
        self.post_states = {}
        self.target_audit = None
        self._prior_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._prior_trace)
            require(not self._pending, f"PENDING_UPDATES:{sorted(self._pending)}")
            require(
                self.post_states is not None and set(self.post_states) == set(self.indices),
                f"POST_STATE_SET:{sorted(self.post_states or {})}",
            )
            require(self.target_audit is not None, "TARGET_INTERVENTION_NOT_OBSERVED")


def q22_path_efficiency(post_states: Mapping[int, torch.Tensor], anchor: int) -> float:
    require(set(post_states) == set(range(anchor, anchor + 5)), "Q22_TRAJECTORY_COORDINATES")
    first = (
        post_states[anchor]
        .detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .numpy()
        .copy()
    )
    states = [first.copy() for _ in range(anchor + 5)]
    for token in range(anchor, anchor + 5):
        states[token] = (
            post_states[token]
            .detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .numpy()
            .copy()
        )
    # Intentionally reuse the frozen functional; only the recurrent-state layer changes.
    return core.post4_path_efficiency(states, anchor)


def _sanitize_layer17_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token_index": int(audit["token_index"]),
        "plus_branch": bool(audit["plus_branch"]),
        "applied_correction_max_abs_residual": float(
            audit["applied_correction_max_abs_residual"]
        ),
        "runtime_correction_l2": float(audit["runtime_correction_l2"]),
    }


def branch_input(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    return phase1.base.prevalence_eq.parent._input_row(encoded, row_index, pair, cell) if hasattr(
        phase1.base.prevalence_eq.parent, "_input_row"
    ) else construction.phase2._input_row(encoded, row_index, pair, cell)


def capture_signed_branch(
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
    require(orientation in {-1, 1}, "ORIENTATION")
    direction64 = direction.detach().cpu().to(torch.float64).contiguous()
    require(tuple(direction64.shape) == (core.EXPECTED_STRONG_COUNT,), "DIRECTION_SHAPE")
    require(
        abs(float(torch.linalg.vector_norm(direction64).item()) - 1.0) <= 1e-10,
        "DIRECTION_NORM",
    )
    delta_h = (direction64 * (float(orientation) * 2.0 * EPSILON)).contiguous()

    target = int(anchor) + core.TARGET_OFFSET
    layer17_audit: dict[str, Any] = {}
    handle = transport_runtime.install_inproj_hook(
        runtime_ctx["mixer17"],
        token_index=target,
        strong_mask=runtime_ctx["strong_mask"],
        delta_h=delta_h,
        plus_branch=plus_branch,
        audit=layer17_audit,
    )
    collector = Layer22Q22Collector(
        code=trace_code,
        update_line=measurement.RECURRENT_UPDATE_LINE,
        readout_line=measurement.CAPTURE_LINE,
        mixer22=model.mamba.layers[LAYER22].mixer,
        anchor=int(anchor),
        condition=condition,
        r22=r22,
        c22=c22,
    )

    prior_trace = sys.gettrace()
    budget.consume()
    try:
        model.mamba.eval()
        with collector.capture():
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids)
    finally:
        handle.remove()

    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION")
    require(bool(layer17_audit), "LAYER17_PROBE_NOT_OBSERVED")
    require(int(layer17_audit["token_index"]) == target, "LAYER17_PROBE_TOKEN")
    require(collector.post_states is not None, "LAYER22_STATES_MISSING")
    require(collector.target_audit is not None, "LAYER22_AUDIT_MISSING")
    pe22 = q22_path_efficiency(collector.post_states, int(anchor))
    return {
        "pe22": float(pe22),
        "layer17_probe_audit": _sanitize_layer17_audit(layer17_audit),
        "layer22_write_audit": dict(collector.target_audit),
        "_target_post_state": collector.post_states[target].detach().cpu().to(torch.float64).contiguous(),
    }


def run_signed(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    r22: torch.Tensor,
    c22: torch.Tensor,
    budget: Any,
) -> dict[str, Any]:
    pair = str(seed["source_pair_id"])
    cells = phase1._cells()
    anchors = phase1._anchors_for_pair(pair, events)
    captured: dict[str, dict[str, Any]] = {}
    for role, plus_branch in (("tp", True), ("tm", False)):
        captured[role] = capture_signed_branch(
            model=model,
            runtime_ctx=runtime_ctx,
            trace_code=trace_code,
            input_ids=construction.phase2._input_row(
                encoded, row_index, pair, cells[role]
            ),
            anchor=int(anchors[role]),
            direction=direction,
            orientation=orientation,
            plus_branch=plus_branch,
            condition=condition,
            r22=r22,
            c22=c22,
            budget=budget,
        )

    plus_audit = captured["tp"]["layer17_probe_audit"]
    minus_audit = captured["tm"]["layer17_probe_audit"]
    require(plus_audit["plus_branch"] is True, "TP_PROBE_SIGN")
    require(minus_audit["plus_branch"] is False, "TM_PROBE_SIGN")
    require(
        abs(float(plus_audit["runtime_correction_l2"]) - 2.0 * EPSILON) <= 1e-12,
        "TP_PROBE_NORM",
    )
    require(
        abs(float(minus_audit["runtime_correction_l2"]) - 2.0 * EPSILON) <= 1e-12,
        "TM_PROBE_NORM",
    )

    f22 = float(captured["tp"]["pe22"]) - float(captured["tm"]["pe22"])
    return {
        "condition": condition,
        "orientation": int(orientation),
        "F22": f22,
        "tp_pe22": float(captured["tp"]["pe22"]),
        "tm_pe22": float(captured["tm"]["pe22"]),
        "branch_audits": {
            role: {
                "layer17_probe_audit": captured[role]["layer17_probe_audit"],
                "layer22_write_audit": captured[role]["layer22_write_audit"],
            }
            for role in BRANCHES
        },
        "_target_post_states": {
            role: captured[role]["_target_post_state"] for role in BRANCHES
        },
        "model_forward_count": FORWARDS_PER_SIGNED,
    }


def run_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    family: str,
    basis_index: int,
    **kwargs,
) -> dict[str, Any]:
    positive = run_signed(
        seed, direction, condition=condition, orientation=1, **kwargs
    )
    negative = run_signed(
        seed, direction, condition=condition, orientation=-1, **kwargs
    )
    f_plus = float(positive["F22"])
    f_minus = float(negative["F22"])
    j22 = (f_plus - f_minus) / (2.0 * EPSILON)
    require(math.isfinite(j22), f"J22_NONFINITE:{family}:{basis_index}")
    return {
        "direction_key": f"{family}_{basis_index}",
        "basis_family": family,
        "basis_index": int(basis_index),
        "F22_plus": f_plus,
        "F22_minus": f_minus,
        "J22": j22,
        "J22_squared": j22 * j22,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": FORWARDS_PER_DIRECTION,
    }


def run_condition(
    seed: Mapping[str, Any],
    *,
    condition: str,
    q_bases: Mapping[str, torch.Tensor],
    **kwargs,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for family in ("xg2", "xg4"):
        for basis_index in range(K):
            probes.append(
                run_direction(
                    seed,
                    q_bases[family][:, basis_index],
                    condition=condition,
                    family=family,
                    basis_index=basis_index,
                    **kwargs,
                )
            )
    require([row["direction_key"] for row in probes] == list(DIRECTIONS), "DIRECTION_ORDER")
    e_xg2 = sum(float(row["J22_squared"]) for row in probes[:K]) / K
    e_xg4 = sum(float(row["J22_squared"]) for row in probes[K:]) / K
    q22 = e_xg2 - e_xg4
    require(all(math.isfinite(v) for v in (e_xg2, e_xg4, q22)), "Q22_NONFINITE")
    return {
        "condition": condition,
        "direction_order": list(DIRECTIONS),
        "direction_probes": probes,
        "E_XG2_22": e_xg2,
        "E_XG4_22": e_xg4,
        "Q22": q22,
        "scientific_model_forward_count": FORWARDS_PER_CONDITION,
    }


def _coordinate_records(condition_row: Mapping[str, Any]):
    for direction in condition_row["direction_probes"]:
        for probe_name in ("positive_probe", "negative_probe"):
            probe = direction[probe_name]
            for role in BRANCHES:
                yield (
                    direction["direction_key"],
                    probe_name,
                    role,
                    probe["branch_audits"][role]["layer22_write_audit"],
                    probe["_target_post_states"][role],
                )


def _strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


def validate_condition_matching_and_diagnostics(
    native: Mapping[str, Any],
    r_row: Mapping[str, Any],
    c_row: Mapping[str, Any],
) -> dict[str, Any]:
    n_records = list(_coordinate_records(native))
    r_records = list(_coordinate_records(r_row))
    c_records = list(_coordinate_records(c_row))
    require(len(n_records) == len(r_records) == len(c_records) == 40, "COORDINATE_COUNT")

    post_r_changes: list[float] = []
    post_c_changes: list[float] = []
    r_write_reductions: list[float] = []
    c_write_reductions: list[float] = []
    max_coeff_residual = 0.0
    max_applied_residual = 0.0
    max_recurrence_residual = 0.0

    for n, r, c in zip(n_records, r_records, c_records, strict=True):
        require(n[:3] == r[:3] == c[:3], "COORDINATE_KEY")
        na, ns = n[3], n[4]
        ra, rs = r[3], r[4]
        ca, cs = c[3], c[4]
        require(
            na["native_write_sha256"]
            == ra["native_write_sha256"]
            == ca["native_write_sha256"],
            "NATIVE_WRITE_IDENTITY_ACROSS_CONDITIONS",
        )
        require(
            na["target_token_index"]
            == ra["target_token_index"]
            == ca["target_token_index"],
            "TARGET_TOKEN_IDENTITY",
        )
        for x, y in zip(
            ra["r22_coefficients"], ca["r22_coefficients"], strict=True
        ):
            require(abs(float(x) - float(y)) <= MATCH_TOL, "R22_COEFFICIENT_MATCH")

        max_coeff_residual = max(
            max_coeff_residual,
            float(ra["coefficient_transfer_norm_equality_residual"]),
            float(ca["coefficient_transfer_norm_equality_residual"]),
        )
        max_applied_residual = max(
            max_applied_residual,
            float(ra["applied_correction_max_abs_residual"]),
            float(ca["applied_correction_max_abs_residual"]),
        )
        max_recurrence_residual = max(
            max_recurrence_residual,
            float(na["target_recurrence_relative_residual"]),
            float(ra["target_recurrence_relative_residual"]),
            float(ca["target_recurrence_relative_residual"]),
        )

        post_r_changes.append(float(torch.linalg.vector_norm(rs - ns).item()))
        post_c_changes.append(float(torch.linalg.vector_norm(cs - ns).item()))
        r_write_reductions.append(
            float(na["native_r22_projection_l2"])
            - float(ra["post_r22_projection_l2"])
        )
        c_write_reductions.append(
            float(na["native_r22_projection_l2"])
            - float(ca["post_r22_projection_l2"])
        )

    return {
        "coordinate_count": 40,
        "mean_r22_write_projection_reduction": sum(r_write_reductions) / 40.0,
        "mean_c22_write_projection_reduction": sum(c_write_reductions) / 40.0,
        "mean_r22_target_post_state_change_l2": sum(post_r_changes) / 40.0,
        "mean_c22_target_post_state_change_l2": sum(post_c_changes) / 40.0,
        "r22_write_reduction_positive": (sum(r_write_reductions) / 40.0) > 0.0,
        "r22_vs_c22_post_state_change_delta_positive":
            (sum(post_r_changes) - sum(post_c_changes)) > 0.0,
        "max_coefficient_transfer_norm_equality_residual": max_coeff_residual,
        "max_applied_correction_residual": max_applied_residual,
        "max_target_recurrence_relative_residual": max_recurrence_residual,
    }


def probe_seed(index: int, pair: str, events: Mapping[Any, Any]) -> dict[str, Any]:
    require(pair == expected_pairs()[index], f"PAIR:{index}")
    anchors = phase1._anchors_for_pair(pair, events)
    return {
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": int(index),
        "target_plus_anchor": int(anchors["tp"]),
        "target_minus_anchor": int(anchors["tm"]),
        "reference_plus_anchor": int(anchors["rp"]),
        "reference_minus_anchor": int(anchors["rm"]),
    }


def endpoint(q0: float, qr: float, qc: float) -> dict[str, float]:
    a_r = q0 - qr
    a_c = q0 - qc
    d_nec22 = qc - qr
    require(abs((a_r - a_c) - d_nec22) <= 1e-12, "D_NEC22_IDENTITY")
    return {
        "Q0": q0,
        "QR": qr,
        "QC": qc,
        "A_R": a_r,
        "A_C": a_c,
        "D_NEC22": d_nec22,
    }


def run_pair(
    seed: Mapping[str, Any],
    *,
    q_bases: Mapping[str, torch.Tensor],
    **kwargs,
) -> dict[str, Any]:
    condition_rows = [
        run_condition(seed, condition=condition, q_bases=q_bases, **kwargs)
        for condition in CONDITIONS
    ]
    by = {row["condition"]: row for row in condition_rows}
    require(set(by) == set(CONDITIONS), "CONDITION_SET")
    ep = endpoint(
        float(by["native22"]["Q22"]),
        float(by["r22_neutralized"]["Q22"]),
        float(by["c22_coefficient_control"]["Q22"]),
    )
    diagnostics = validate_condition_matching_and_diagnostics(
        by["native22"],
        by["r22_neutralized"],
        by["c22_coefficient_control"],
    )
    return {
        **dict(seed),
        "schema_version": ITEM_SCHEMA,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "q22_correction_commit": Q22_CORRECTION_COMMIT,
        "r22_c22_freeze_commit": R22_C22_FREEZE_COMMIT,
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "conditions": _strip_private(condition_rows),
        **ep,
        "native_diagnostics": diagnostics,
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "row_dropped": False,
    }


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            return h
    raise NecessityError("BETACF_DID_NOT_CONVERGE")


def _regularized_beta(x: float, a: float, b: float) -> float:
    require(0.0 <= x <= 1.0 and a > 0.0 and b > 0.0, "BETA_DOMAIN")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def one_sided_one_sample_student_t(values: Sequence[float]) -> dict[str, float]:
    xs = [float(v) for v in values]
    n = len(xs)
    require(n >= 2, "TTEST_N")
    require(all(math.isfinite(v) for v in xs), "TTEST_NONFINITE")
    mean = sum(xs) / n
    variance = sum((v - mean) ** 2 for v in xs) / (n - 1)
    require(variance > 0.0 and math.isfinite(variance), "TTEST_ZERO_VARIANCE")
    se = math.sqrt(variance / n)
    t_stat = mean / se
    df = n - 1
    x = df / (df + t_stat * t_stat)
    ib = _regularized_beta(x, 0.5 * df, 0.5)
    p = 0.5 * ib if t_stat >= 0.0 else 1.0 - 0.5 * ib
    require(0.0 <= p <= 1.0 and math.isfinite(p), "TTEST_P")
    return {
        "n": int(n),
        "mean": float(mean),
        "sample_variance": float(variance),
        "t_statistic": float(t_stat),
        "degrees_of_freedom": int(df),
        "p_one_sided_greater": float(p),
    }


def confirmatory_decision(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(items) == PAIR_COUNT, "DECISION_ITEM_COUNT")
    q0 = [float(row["Q0"]) for row in items]
    ar = [float(row["A_R"]) for row in items]
    d = [float(row["D_NEC22"]) for row in items]
    require(all(row.get("row_dropped") is False for row in items), "ROW_DROPPING")
    test = one_sided_one_sample_student_t(d)
    mean_q0 = sum(q0) / PAIR_COUNT
    mean_ar = sum(ar) / PAIR_COUNT
    mean_d = sum(d) / PAIR_COUNT
    supported = (
        mean_q0 > 0.0
        and mean_ar > 0.0
        and mean_d > 0.0
        and float(test["p_one_sided_greater"]) < 0.05
    )
    return {
        "mean_Q0": mean_q0,
        "mean_A_R": mean_ar,
        "mean_D_NEC22": mean_d,
        "confirmatory_test": {
            "test": "one_sided_one_sample_student_t",
            "alternative": "greater_than_zero",
            "endpoint": "D_NEC22",
            **test,
        },
        "confirmatory_p_value_count": 1,
        "label": LABEL_SUPPORTED if supported else LABEL_NOT_ESTABLISHED,
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


def checksums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n" for name, raw in sorted(files.items())
    ).encode("utf-8")


def write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".staging-", dir=str(output_dir.parent)
    ) as tmp:
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
            "confirmatory_p_value_count": 1,
        }
        manifest_raw = canonical_json_bytes(manifest)
        (staging / MANIFEST_FILE).write_bytes(manifest_raw)
        checksum_raw = checksums_bytes({**primary, MANIFEST_FILE: manifest_raw})
        (staging / CHECKSUM_FILE).write_bytes(checksum_raw)
        required = {ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE, CHECKSUM_FILE}
        require(
            {path.name for path in staging.iterdir() if path.is_file()} == required,
            "OUTPUT_FILE_SET",
        )
        output_dir.mkdir(parents=False, exist_ok=False)
        for name in sorted(required):
            (output_dir / name).write_bytes((staging / name).read_bytes())


def run_necessity(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path | None,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_static_inputs()
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    r22, c22, basis_geometry = load_owner_bases()
    q_bases = load_q_bases()
    rows, encoded, events, row_index, tokenizer_provenance = load_inputs(tokenizer_snapshot)
    del rows

    with construction.exact_slow_runtime_contract():
        trace_code, trace_line = measurement._resolve_and_validate_runtime_binding()
        require(trace_line == measurement.CAPTURE_LINE, "TRACE_LINE_BINDING")
        model, checkpoint_sha = parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        require(checkpoint_sha == CHECKPOINT_SHA256, "CHECKPOINT_SHA256")
        require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "MODEL_NOT_CPU")
        runtime_ctx = transport_runtime.validate_runtime_components(model)
        require(len(model.mamba.layers) == core.LAYER_COUNT, "LAYER_COUNT")
        require(model.mamba.layers[LAYER22].mixer is not None, "LAYER22_MIXER")

        before_signature = model_parameter_signature(model)
        budget = parent.ForwardBudget(FULL_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []
        for index, pair in enumerate(expected_pairs()):
            seed = probe_seed(index, pair, events)
            items.append(
                run_pair(
                    seed,
                    q_bases=q_bases,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    r22=r22,
                    c22=c22,
                    budget=budget,
                )
            )
        budget.assert_exact()
        after_signature = model_parameter_signature(model)
        require(before_signature == after_signature, "MODEL_PARAMETER_MUTATION")

    decision = confirmatory_decision(items)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "implementation_authority_commit": IMPLEMENTATION_AUTHORITY_COMMIT,
        "q22_correction_commit": Q22_CORRECTION_COMMIT,
        "r22_c22_freeze_commit": R22_C22_FREEZE_COMMIT,
        "source_pair_count": PAIR_COUNT,
        "pair_id_first": "xg1_fact_8101",
        "pair_id_last": "xg1_fact_8400",
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "epsilon": EPSILON,
        "subspace_dim": K,
        "forwards_per_condition": FORWARDS_PER_CONDITION,
        "forwards_per_pair": FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run": FULL_FORWARD_BUDGET,
        "checkpoint_load_count_this_run": 1,
        "representative_checkpoint_sha256": checkpoint_sha,
        "native_backbone_signature_sha256": NATIVE_BACKBONE_SIGNATURE_SHA256,
        "target_layer": LAYER22,
        "target_object": "WRITE22=deltaB_u[:,:,target_token,:]",
        "post_state_object": "POST_STATE22=ssm_state_after_modified_recurrent_update",
        "q22_definition": "E_XG2_22-E_XG4_22",
        "q22_readout_function": "core.post4_path_efficiency",
        "q22_state_layer": 22,
        "basis_geometry": basis_geometry,
        "r22_basis_sha256": R22_SHA256,
        "c22_basis_sha256": C22_SHA256,
        "model_parameter_signature_before": before_signature,
        "model_parameter_signature_after": after_signature,
        "tokenizer": tokenizer_provenance,
        "decision": decision,
        "confirmatory_p_value_count": 1,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "cuda_executed": False,
        "row_dropping_executed": False,
        "raw_native_vectors_persisted": False,
        "raw_post_state_vectors_persisted": False,
        "restoration_executed": False,
        "ownership_implementation_executed": False,
        "scientific_conclusion": decision["label"],
        "next_stage": (
            "RESTORATION_CONFIRMATION_IMPLEMENTATION_ONLY_IF_NECESSITY_SUPPORTED"
            if decision["label"] == LABEL_SUPPORTED
            else "STOP_PHASE1B_BRIDGE_NOT_ESTABLISHED"
        ),
    }
    write_outputs(output_dir, items=items, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gen5 Phase1B R22 local necessity semantic-reference runner. "
            "Uses frozen layer17 XG2/XG4 signed probes, layer22 native-write "
            "R22/C22 intervention, and downstream layer22 post-state Q22. "
            "Exactly one confirmatory p-value is produced in an authorized full run."
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
    summary = run_necessity(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + summary["result"])
    print("PAIR_ID_FIRST=" + summary["pair_id_first"])
    print("PAIR_ID_LAST=" + summary["pair_id_last"])
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("CONFIRMATORY_P_VALUE_COUNT=1")
    print("SCIENTIFIC_CONCLUSION=" + summary["scientific_conclusion"])


if __name__ == "__main__":
    main()
