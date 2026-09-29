#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
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
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import reason_router_gen4_native_mamba_state_extraction as extraction
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import reason_router_gen4_pp3_necessity_fast_cuda as pp3
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate
from scripts import reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1 as phase1
from scripts import reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2 as phase2

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

PHASE1B_DESIGN_COMMIT = "c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8"
STATIC_PREPARATION_FREEZE_COMMIT = "5c0d91959af1f502b667ed6ab815c949c1043cbf"
CONSTRUCTION_CORRECTION_FREEZE_COMMIT = "0eb93f49cd33553ba244bb943364ea97125a2e23"

FROZEN_BLOBS = {
    "reports/reason_router_gen5_phase1b_static_preparation_c8dc7a4_v1/"
    "static_preparation_summary.json":
        "4b6edf39860e6cb043137d2c109a39405f9ad6a1",
    "reports/reason_router_gen5_phase1b_construction_coordinate_native_write_runtime_"
    "correction_spec_candidate.md":
        "b0c3faa12c06878690f654bd155f2a1c5792d183",
    "reports/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence_6afb619_r1/"
    "equivalence_report.json":
        "35d4a3b75e9a52d4ebdd5fd86eec0424a1b72991",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py":
        "3677dd83950789e41417c3a1ffaf70b82d7003ad",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py":
        "989c4a8947560dcf35e9523373d09ba085a9431a",
    "scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py":
        "4a8326c442b883510ba452f049c88f3163677fc5",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py":
        "7f51683ecbf60d63cdc0399d2b0e89f8d42dd0ab",
    "scripts/reason_router_gen4_native_mamba_state_measurement.py":
        "8c8d63cce182dfab66a632299e93cd5b25fc36af",
    "scripts/reason_router_gen4_pp3_necessity_fast_cuda.py":
        "26ca67ad8603799a849c151a39728368227326df",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py":
        "00a81ce6ec4ada4d5c0bf36418347b222543d174",
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py":
        "6c98ce022ca134e385db28851fd364dc6daff423",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py":
        "62df03e9c48014650dee69ae007b91d87d5cb7ac",
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2.py":
        "bf00bdf2c090b7d9ef3c40a4604470e292f4db9d",
}

DATA_ROOT = Path("data/reason_router_gen5_phase1b_xg1_construction_v1")
SOURCE_FILE = "structured_source_facts.jsonl"
ROWS_FILE = "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
ELIGIBILITY_FILE = "tokenizer_eligibility_summary.json"

STATIC_INPUT_SHA256 = {
    SOURCE_FILE: "3f8eac771794e1d022bee9f669f315c3ac05feeaa9c2195095b8a892b4269ea1",
    ROWS_FILE: "5a82508e54cd6097aeee5afe10d7c416357302701d558cfdc8740382168feca4",
    STRUCTURAL_FILE: "d68d812ca43a6c651d284f0aa2889f87adb7e030c599af37cb49deaa9d6d8105",
    ANCHOR_FILE: "1dbdd3e072245072f197d87443cc8cf81cbbcd2e61437c99b26e0d82815e33b4",
    ELIGIBILITY_FILE: "7e304fb1d6f2ecb9ed6fb87911623c47d0b6ae9c2eb9bf9b8f45f70ae1fcb06a",
}

CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
NATIVE_BACKBONE_SIGNATURE_SHA256 = (
    "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"
)

PAIR_FIRST = 7801
PAIR_LAST = 8100
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

CONDITIONS = ("native", "pp3_neutralized", "pp5_coefficient_control")
BRANCHES = ("tp", "tm")
FORWARDS_PER_CONDITION = 2
FORWARDS_PER_PAIR = 6
FULL_FORWARD_BUDGET = 1800

LAYER22 = 22
STATE_SHAPE = (1, 1536, 16)
STATE_WIDTH = 1536 * 16
RANK = 2
BASIS_ORTHOGONALITY_ATOL = 1e-10
RECONSTRUCTION_REL_TOL = 5e-6
RUNTIME_MATCH_TOL = transport_runtime.RUNTIME_CAST_TOL

ITEM_FILE = "construction_item_audit.jsonl"
R22_FILE = "r22_basis.f64le"
C22_FILE = "c22_basis.f64le"
SUMMARY_FILE = "construction_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen5-phase1b-r22-c22-construction-item-v1"
SUMMARY_SCHEMA = "gen5-phase1b-r22-c22-construction-summary-v1"
MANIFEST_SCHEMA = "gen5-phase1b-r22-c22-construction-manifest-v1"
RESULT_PASS = "PASS_GEN5_PHASE1B_R22_C22_CONSTRUCTION"


class ConstructionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ConstructionError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConstructionError("GIT_FAILURE:" + " ".join(args)) from exc


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
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
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
        (PHASE1B_DESIGN_COMMIT, "PHASE1B_DESIGN"),
        (STATIC_PREPARATION_FREEZE_COMMIT, "STATIC_PREPARATION"),
        (CONSTRUCTION_CORRECTION_FREEZE_COMMIT, "CONSTRUCTION_CORRECTION"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    for path, expected_blob in FROZEN_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_BLOB_DRIFT:{path}:{observed}")


def validate_historical_backend_boundary() -> dict[str, Any]:
    path = (
        ROOT
        / "reports/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence_6afb619_r1"
        / "equivalence_report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        report.get("result") == "PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "HISTORICAL_BACKEND_EQUIVALENCE_RESULT",
    )
    require(
        report.get("representative_checkpoint_sha256") == CHECKPOINT_SHA256,
        "HISTORICAL_BACKEND_CHECKPOINT",
    )
    require(float(report.get("state_atol")) == 1e-4, "HISTORICAL_STATE_ATOL")
    require(float(report.get("state_rtol")) == 1e-4, "HISTORICAL_STATE_RTOL")
    require(report.get("scientific_conclusion") is None, "HISTORICAL_SCIENTIFIC_CONCLUSION")
    return {
        "result": report["result"],
        "source_pair_id": report["source_pair_id"],
        "max_state_abs_diff": float(report["max_state_abs_diff"]),
        "state_atol": float(report["state_atol"]),
        "state_rtol": float(report["state_rtol"]),
    }


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
    require(structural.get("role") == "construction", "STRUCTURAL_ROLE")
    require(structural.get("pair_id_first") == "xg1_fact_7801", "STRUCTURAL_FIRST")
    require(structural.get("pair_id_last") == "xg1_fact_8100", "STRUCTURAL_LAST")
    require(structural.get("source_pair_count") == PAIR_COUNT, "STRUCTURAL_PAIR_COUNT")
    require(structural.get("row_count") == ROW_COUNT, "STRUCTURAL_ROW_COUNT")
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
    require(eligibility.get("role") == "construction", "ELIGIBILITY_ROLE")
    require(
        eligibility.get("eligible_anchor_row_count") == 1800,
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
    require(len(event_rows) == 1800, "ANCHOR_ROW_COUNT")

    events = parent.event_lookup(event_rows)
    parent.validate_transport_event_plan(pairs, events)
    row_index = parent.build_row_index(rows)
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, events, row_index, tokenizer_provenance


@contextmanager
def exact_slow_runtime_contract():
    original_torch = measurement.EXPECTED_VERSIONS["torch"]
    observed_torch = torch.__version__
    allowed_cuda_build = backend.EXPECTED_RUNTIME["torch"]

    if observed_torch == original_torch:
        measurement.runtime_gate()
        yield
        return

    require(
        observed_torch == allowed_cuda_build,
        f"TORCH_RUNTIME_NOT_AUTHORIZED:{observed_torch}",
    )
    with backend.parent_runtime_rebind():
        measurement.runtime_gate()
        yield


def tensor_snapshot(value: Any, label: str) -> torch.Tensor:
    require(torch.is_tensor(value), f"{label}_NOT_TENSOR")
    out = value.detach().cpu().contiguous().clone()
    require(tuple(out.shape) == STATE_SHAPE, f"{label}_SHAPE:{tuple(out.shape)}")
    require(out.dtype == torch.float32, f"{label}_DTYPE:{out.dtype}")
    require(bool(torch.isfinite(out).all().item()), f"{label}_NONFINITE")
    return out


@dataclass(frozen=True)
class NativeWriteRecord:
    w: torch.Tensor
    s_post: torch.Tensor
    reconstruction_relative_residual: float


@dataclass(frozen=True)
class _PreUpdate:
    s_prev: torch.Tensor
    g: torch.Tensor
    w: torch.Tensor


class Layer22NativeWriteCollector:
    def __init__(
        self,
        *,
        code: Any,
        update_line: int,
        readout_line: int,
        mixer22: Any,
        target_indices: Iterable[int],
    ) -> None:
        require(type(update_line) is int and update_line > 0, "UPDATE_LINE")
        require(type(readout_line) is int and readout_line > update_line, "READOUT_LINE")
        self.code = code
        self.update_line = update_line
        self.readout_line = readout_line
        self.mixer22_id = id(mixer22)
        self.target_indices = frozenset(int(v) for v in target_indices)
        require(bool(self.target_indices), "TARGET_INDICES_EMPTY")
        require(min(self.target_indices) >= 0, "TARGET_INDEX_NEGATIVE")
        self._pending: dict[int, _PreUpdate] | None = None
        self.records: dict[int, NativeWriteRecord] | None = None
        self._prior_trace: Any = None
        self._used = False

    def _coordinate(self, frame: Any) -> int | None:
        if id(frame.f_locals.get("self")) != self.mixer22_id:
            return None
        token_index = frame.f_locals.get("i")
        require(type(token_index) is int and token_index >= 0, "AMBIGUOUS_TOKEN_INDEX")
        if token_index not in self.target_indices:
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
            require(self._pending is not None and self.records is not None, "TRACE_NOT_ACTIVE")
            require(token not in self._pending and token not in self.records, "DUPLICATE_PRE")

            s_prev = tensor_snapshot(frame.f_locals.get("ssm_state"), "S_PREV")
            discrete_a = frame.f_locals.get("discrete_A")
            deltab_u = frame.f_locals.get("deltaB_u")
            require(torch.is_tensor(discrete_a), "DISCRETE_A_MISSING")
            require(torch.is_tensor(deltab_u), "DELTAB_U_MISSING")
            g = tensor_snapshot(discrete_a[:, :, token, :], "G")
            w = tensor_snapshot(deltab_u[:, :, token, :], "W")
            self._pending[token] = _PreUpdate(s_prev=s_prev, g=g, w=w)
            return self._trace

        if frame.f_lineno == self.readout_line:
            token = self._coordinate(frame)
            if token is None:
                return self._trace
            require(self._pending is not None and self.records is not None, "TRACE_NOT_ACTIVE")
            require(token in self._pending and token not in self.records, "POST_WITHOUT_PRE")
            pre = self._pending.pop(token)
            s_post = tensor_snapshot(frame.f_locals.get("ssm_state"), "S_POST")

            recon = pre.g.to(torch.float64) * pre.s_prev.to(torch.float64) + pre.w.to(torch.float64)
            residual = recon - s_post.to(torch.float64)
            denom = max(
                float(torch.linalg.vector_norm(recon).item()),
                float(torch.linalg.vector_norm(s_post.to(torch.float64)).item()),
                1e-12,
            )
            rel = float(torch.linalg.vector_norm(residual).item() / denom)
            require(rel <= RECONSTRUCTION_REL_TOL, f"RECURRENCE_RECONSTRUCTION:{rel}")
            self.records[token] = NativeWriteRecord(
                w=pre.w,
                s_post=s_post,
                reconstruction_relative_residual=rel,
            )
            return self._trace

        return self._trace

    @contextmanager
    def capture(self):
        require(not self._used, "OBSERVER_REUSE")
        self._used = True
        self._pending = {}
        self.records = {}
        self._prior_trace = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._prior_trace)
        require(self._pending == {}, "PENDING_CAPTURE_REMAINS")


def condition_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    audit: dict[str, Any],
) -> torch.Tensor:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_SHAPE",
    )
    require(0 <= token_index < output.shape[1], "TARGET_TOKEN_RANGE")

    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == core.INTERMEDIATE_SIZE, "MASK_WIDTH")
    require(int(mask_cpu.sum().item()) == pp3.DIM, "MASK_COUNT")
    mask = mask_cpu.to(output.device)

    before = output.detach().clone()
    h = before[0, token_index, : core.INTERMEDIATE_SIZE][mask].detach().cpu().to(torch.float64)
    ci = pp3.condition_correction(h, condition, planes)
    out = output.clone()
    intended = ci["d"].to(device=out.device, dtype=out.dtype)
    out[0, token_index, : core.INTERMEDIATE_SIZE][mask] += intended

    require(
        torch.equal(
            out[:, :, core.INTERMEDIATE_SIZE :],
            before[:, :, core.INTERMEDIATE_SIZE :],
        ),
        "GATE_BRANCH_CHANGED",
    )
    non = ~mask
    require(
        torch.equal(
            out[:, :, : core.INTERMEDIATE_SIZE][:, :, non],
            before[:, :, : core.INTERMEDIATE_SIZE][:, :, non],
        ),
        "NONSTRONG_CHANGED",
    )
    if token_index > 0:
        require(
            torch.equal(out[:, :token_index, :], before[:, :token_index, :]),
            "EARLIER_TOKEN_CHANGED",
        )
    if token_index + 1 < out.shape[1]:
        require(
            torch.equal(out[:, token_index + 1 :, :], before[:, token_index + 1 :, :]),
            "LATER_TOKEN_CHANGED",
        )

    applied = (
        out[0, token_index, : core.INTERMEDIATE_SIZE][mask]
        - before[0, token_index, : core.INTERMEDIATE_SIZE][mask]
    ).detach().cpu().to(torch.float64)
    residual = float(
        torch.max(torch.abs(applied - intended.detach().cpu().to(torch.float64))).item()
    )
    require(residual <= RUNTIME_MATCH_TOL, f"APPLIED_CORRECTION_RESIDUAL:{residual}")

    audit.clear()
    audit.update(
        {
            "condition": condition,
            "token_index": int(token_index),
            "native_pp3_a": float(ci["a"]),
            "native_pp3_b": float(ci["b"]),
            "condition_correction_l2": float(ci["l2"]),
            "pp3_post_condition_residual_plus": float(ci["res_plus"]),
            "pp3_post_condition_residual_minus": float(ci["res_minus"]),
            "applied_correction_max_abs_residual": residual,
        }
    )
    return out


def install_condition_hook(mixer17: Any, **kwargs):
    def hook(_module, _args, output):
        return condition_hook(output, **kwargs)

    return mixer17.in_proj.register_forward_hook(hook)


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return sha256_bytes(raw)


def vec64(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().cpu().to(torch.float64).contiguous().reshape(-1).clone()
    require(out.numel() == STATE_WIDTH, "VECTOR_WIDTH")
    require(bool(torch.isfinite(out).all().item()), "VECTOR_NONFINITE")
    return out


def capture_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    input_ids: torch.Tensor,
    anchor: int,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    budget: Any,
) -> dict[str, Any]:
    require(tuple(input_ids.shape) == (1, extraction.MAX_MODEL_SEQUENCE_LENGTH), "INPUT_SHAPE")
    require(input_ids.device.type == "cpu", "INPUT_NOT_CPU")
    target = int(anchor) + core.TARGET_OFFSET
    require(0 <= target < input_ids.shape[1], "TARGET_OUT_OF_RANGE")

    mixer17 = runtime_ctx["mixer17"]
    mixer22 = model.mamba.layers[LAYER22].mixer
    audit: dict[str, Any] = {}

    handle = install_condition_hook(
        mixer17,
        token_index=target,
        strong_mask=runtime_ctx["strong_mask"],
        condition=condition,
        planes=planes,
        audit=audit,
    )
    observer = Layer22NativeWriteCollector(
        code=trace_code,
        update_line=measurement.RECURRENT_UPDATE_LINE,
        readout_line=measurement.CAPTURE_LINE,
        mixer22=mixer22,
        target_indices=(target,),
    )

    prior_trace = sys.gettrace()
    budget.consume()
    try:
        model.mamba.eval()
        with observer.capture():
            with torch.inference_mode():
                _ = model.mamba(input_ids=input_ids)
    finally:
        handle.remove()

    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION")
    require(bool(audit), "CONDITION_HOOK_NOT_OBSERVED")
    require(observer.records is not None, "OBSERVER_RECORDS_MISSING")
    require(set(observer.records) == {target}, "OBSERVER_TARGET_SET")
    record = observer.records[target]

    return {
        "target_token_index": target,
        "condition_audit": dict(audit),
        "w": record.w,
        "s_post": record.s_post,
        "w_sha256": tensor_sha256(record.w),
        "s_post_sha256": tensor_sha256(record.s_post),
        "reconstruction_relative_residual": record.reconstruction_relative_residual,
    }


def branch_input(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    return phase2._input_row(encoded, row_index, pair, cell)


def verify_matched_condition_audits(
    pp3_branch: Mapping[str, Any],
    pp5_branch: Mapping[str, Any],
) -> None:
    a = pp3_branch["condition_audit"]
    b = pp5_branch["condition_audit"]
    for key in ("native_pp3_a", "native_pp3_b", "condition_correction_l2"):
        residual = abs(float(a[key]) - float(b[key]))
        require(residual <= RUNTIME_MATCH_TOL, f"MATCHED_CONDITION:{key}:{residual}")


def run_pair(
    index: int,
    pair: str,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    planes: Mapping[str, torch.Tensor],
    budget: Any,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    require(pair == expected_pairs()[index], f"PAIR:{index}")
    cells = phase1._cells()
    anchors = phase1._anchors_for_pair(pair, events)

    by_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        branch_records: dict[str, dict[str, Any]] = {}
        for role in BRANCHES:
            branch_records[role] = capture_branch(
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                input_ids=branch_input(encoded, row_index, pair, cells[role]),
                anchor=int(anchors[role]),
                condition=condition,
                planes=planes,
                budget=budget,
            )
        by_condition[condition] = branch_records

    for role in BRANCHES:
        verify_matched_condition_audits(
            by_condition["pp3_neutralized"][role],
            by_condition["pp5_coefficient_control"][role],
        )

    bw: dict[str, torch.Tensor] = {}
    bs: dict[str, torch.Tensor] = {}
    for condition in CONDITIONS:
        bw[condition] = (
            vec64(by_condition[condition]["tp"]["w"])
            - vec64(by_condition[condition]["tm"]["w"])
        )
        bs[condition] = (
            vec64(by_condition[condition]["tp"]["s_post"])
            - vec64(by_condition[condition]["tm"]["s_post"])
        )

    d_w = bw["pp5_coefficient_control"] - bw["pp3_neutralized"]
    d_s = bs["pp5_coefficient_control"] - bs["pp3_neutralized"]
    native = bw["native"]

    item_conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        branch_payload: dict[str, Any] = {}
        for role in BRANCHES:
            rec = by_condition[condition][role]
            branch_payload[role] = {
                "target_token_index": int(rec["target_token_index"]),
                "w_sha256": str(rec["w_sha256"]),
                "s_post_sha256": str(rec["s_post_sha256"]),
                "reconstruction_relative_residual": float(
                    rec["reconstruction_relative_residual"]
                ),
                "condition_audit": dict(rec["condition_audit"]),
            }
        item_conditions.append(
            {
                "condition": condition,
                "branches": branch_payload,
                "branch_write_contrast_l2": float(torch.linalg.vector_norm(bw[condition]).item()),
                "branch_post_state_contrast_l2": float(torch.linalg.vector_norm(bs[condition]).item()),
            }
        )

    item = {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "pair_index": index,
        "condition_order": list(CONDITIONS),
        "branch_order": list(BRANCHES),
        "conditions": item_conditions,
        "dW_l2": float(torch.linalg.vector_norm(d_w).item()),
        "dS_l2": float(torch.linalg.vector_norm(d_s).item()),
        "native_branch_write_l2": float(torch.linalg.vector_norm(native).item()),
        "raw_native_vectors_persisted": False,
        "Q_computed": False,
        "task_heads_executed": False,
        "scientific_conclusion": None,
    }
    return item, d_w.contiguous(), d_s.contiguous(), native.contiguous()


def sign_canonicalize_basis(basis: torch.Tensor) -> torch.Tensor:
    require(basis.ndim == 2 and basis.shape[1] == RANK, "BASIS_SHAPE")
    out = basis.detach().cpu().to(torch.float64).contiguous().clone()
    for column in range(RANK):
        vec = out[:, column]
        index = int(torch.argmax(torch.abs(vec)).item())
        if float(vec[index].item()) < 0.0:
            out[:, column] = -vec
    return out.contiguous()


def rank2_basis_from_centered(
    centered: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(centered.ndim == 2 and centered.shape[0] == PAIR_COUNT, f"{label}:MATRIX_SHAPE")
    require(bool(torch.isfinite(centered).all().item()), f"{label}:NONFINITE")
    centered = centered.detach().cpu().to(torch.float64).contiguous()
    _u, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    require(singular.numel() >= 3, f"{label}:SINGULAR_COUNT")
    sigma = [float(singular[index].item()) for index in range(3)]
    gap_tol = max(1e-12, 1e-10 * max(sigma[0], 1.0))
    gap = sigma[1] - sigma[2]
    require(gap > gap_tol, f"BLOCKED_{label}_RANK2_NOT_IDENTIFIABLE:{gap}:{gap_tol}")

    basis = sign_canonicalize_basis(vh[:RANK, :].T.contiguous())
    gram = basis.T @ basis
    orth_residual = float(
        torch.max(torch.abs(gram - torch.eye(RANK, dtype=torch.float64))).item()
    )
    require(
        orth_residual <= BASIS_ORTHOGONALITY_ATOL,
        f"{label}:ORTHONORMALITY:{orth_residual}",
    )
    return basis, {
        "singular_values_first3": sigma,
        "sigma2_minus_sigma3": gap,
        "gap_tolerance": gap_tol,
        "orthonormality_max_abs_residual": orth_residual,
    }


def construct_bases(
    d_w_rows: Sequence[torch.Tensor],
    native_rows: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    require(len(d_w_rows) == PAIR_COUNT, "DW_ROW_COUNT")
    require(len(native_rows) == PAIR_COUNT, "NATIVE_ROW_COUNT")
    d = torch.stack(list(d_w_rows), dim=0).to(torch.float64).contiguous()
    n = torch.stack(list(native_rows), dim=0).to(torch.float64).contiguous()
    require(tuple(d.shape) == (PAIR_COUNT, STATE_WIDTH), "DW_MATRIX_SHAPE")
    require(tuple(n.shape) == (PAIR_COUNT, STATE_WIDTH), "NATIVE_MATRIX_SHAPE")

    mu_d = d.mean(dim=0)
    d_centered = (d - mu_d).contiguous()
    r22, r_meta = rank2_basis_from_centered(d_centered, label="R22")

    mu_n = n.mean(dim=0)
    n_centered = (n - mu_n).contiguous()
    n_perp = (n_centered - (n_centered @ r22) @ r22.T).contiguous()
    r_projection_residual = float(torch.max(torch.abs(n_perp @ r22)).item())
    require(
        r_projection_residual <= BASIS_ORTHOGONALITY_ATOL,
        f"NATIVE_RESIDUAL_R22_PROJECTION:{r_projection_residual}",
    )
    c22, c_meta = rank2_basis_from_centered(n_perp, label="C22")

    cross = r22.T @ c22
    cross_residual = float(torch.max(torch.abs(cross)).item())
    require(
        cross_residual <= BASIS_ORTHOGONALITY_ATOL,
        f"R22_C22_CROSS_ORTHOGONALITY:{cross_residual}",
    )

    return r22, c22, {
        "R22": r_meta,
        "C22": c_meta,
        "R22_C22_cross_orthogonality_max_abs_residual": cross_residual,
        "native_residual_R22_projection_max_abs": r_projection_residual,
        "dW_mean_l2": float(torch.linalg.vector_norm(mu_d).item()),
        "native_branch_write_mean_l2": float(torch.linalg.vector_norm(mu_n).item()),
    }


def basis_bytes(basis: torch.Tensor) -> bytes:
    require(tuple(basis.shape) == (STATE_WIDTH, RANK), "SERIALIZE_BASIS_SHAPE")
    canonical = sign_canonicalize_basis(basis)
    raw = canonical.T.contiguous().numpy().astype("<f8", copy=False).tobytes(order="C")
    require(len(raw) == STATE_WIDTH * RANK * 8, "BASIS_BYTES")
    return raw


def checksums_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n"
        for name, raw in sorted(files.items())
    ).encode("utf-8")


def write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    r22: torch.Tensor,
    c22: torch.Tensor,
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=output_dir.name + ".staging-",
        dir=str(output_dir.parent),
    ) as tmp:
        staging = Path(tmp)
        primary = {
            R22_FILE: basis_bytes(r22),
            C22_FILE: basis_bytes(c22),
            ITEM_FILE: jsonl_bytes(items),
            SUMMARY_FILE: canonical_json_bytes(summary),
        }
        for name, raw in primary.items():
            (staging / name).write_bytes(raw)

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "result": RESULT_PASS,
            "files": {
                name: {
                    "sha256": sha256_bytes(raw),
                    "bytes": len(raw),
                }
                for name, raw in sorted(primary.items())
            },
            "raw_native_vectors_persisted": False,
            "scientific_conclusion": None,
        }
        manifest_raw = canonical_json_bytes(manifest)
        (staging / MANIFEST_FILE).write_bytes(manifest_raw)

        checksum_inputs = {**primary, MANIFEST_FILE: manifest_raw}
        checksum_raw = checksums_bytes(checksum_inputs)
        (staging / CHECKSUM_FILE).write_bytes(checksum_raw)

        required = {
            R22_FILE,
            C22_FILE,
            ITEM_FILE,
            SUMMARY_FILE,
            MANIFEST_FILE,
            CHECKSUM_FILE,
        }
        require(
            {path.name for path in staging.iterdir() if path.is_file()} == required,
            "OUTPUT_FILE_SET",
        )

        output_dir.mkdir(parents=False, exist_ok=False)
        for name in sorted(required):
            (output_dir / name).write_bytes((staging / name).read_bytes())


def run_construction(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path | None,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_static_inputs()
    backend_boundary = validate_historical_backend_boundary()
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    planes = pp3.load_planes()
    rows, encoded, events, row_index, tokenizer_provenance = load_inputs(
        tokenizer_snapshot
    )
    del rows

    with exact_slow_runtime_contract():
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

        budget = parent.ForwardBudget(FULL_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []
        d_w_rows: list[torch.Tensor] = []
        native_rows: list[torch.Tensor] = []
        d_s_norms: list[float] = []

        for index, pair in enumerate(expected_pairs()):
            item, d_w, d_s, native = run_pair(
                index,
                pair,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                encoded=encoded,
                row_index=row_index,
                events=events,
                planes=planes,
                budget=budget,
            )
            items.append(item)
            d_w_rows.append(d_w)
            native_rows.append(native)
            d_s_norms.append(float(torch.linalg.vector_norm(d_s).item()))

        budget.assert_exact()
        r22, c22, basis_meta = construct_bases(d_w_rows, native_rows)

    r22_raw = basis_bytes(r22)
    c22_raw = basis_bytes(c22)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "phase1b_design_commit": PHASE1B_DESIGN_COMMIT,
        "static_preparation_freeze_commit": STATIC_PREPARATION_FREEZE_COMMIT,
        "construction_correction_freeze_commit": CONSTRUCTION_CORRECTION_FREEZE_COMMIT,
        "source_pair_count": PAIR_COUNT,
        "pair_id_first": "xg1_fact_7801",
        "pair_id_last": "xg1_fact_8100",
        "condition_order": list(CONDITIONS),
        "branch_order": list(BRANCHES),
        "construction_probe": "NONE",
        "forwards_per_condition": FORWARDS_PER_CONDITION,
        "forwards_per_pair": FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run": FULL_FORWARD_BUDGET,
        "checkpoint_load_count_this_run": 1,
        "representative_checkpoint_sha256": checkpoint_sha,
        "native_backbone_signature_sha256": NATIVE_BACKBONE_SIGNATURE_SHA256,
        "target_layer": LAYER22,
        "native_write_source": "deltaB_u",
        "native_post_state_source": "ssm_state_after_recurrent_update",
        "state_shape": list(STATE_SHAPE),
        "state_width": STATE_WIDTH,
        "rank": RANK,
        "basis_orthogonality_atol": BASIS_ORTHOGONALITY_ATOL,
        "reconstruction_relative_tolerance": RECONSTRUCTION_REL_TOL,
        "basis_diagnostics": basis_meta,
        "mean_dS_l2": float(sum(d_s_norms) / len(d_s_norms)),
        "max_dS_l2": float(max(d_s_norms)),
        "r22_basis_sha256": sha256_bytes(r22_raw),
        "c22_basis_sha256": sha256_bytes(c22_raw),
        "r22_basis_bytes": len(r22_raw),
        "c22_basis_bytes": len(c22_raw),
        "tokenizer": tokenizer_provenance,
        "historical_backend_boundary": backend_boundary,
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_expected_version": measurement.EXPECTED_VERSIONS["transformers"],
            "device": "cpu",
        },
        "necessity_confirmation_data_loaded": False,
        "restoration_confirmation_data_loaded": False,
        "Q_computed": False,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "cuda_executed": False,
        "raw_native_vectors_persisted": False,
        "scientific_conclusion": None,
        "next_stage": "FREEZE_R22_C22_THEN_NECESSITY_CONFIRMATION_IMPLEMENTATION",
    }

    write_outputs(
        output_dir,
        items=items,
        r22=r22,
        c22=c22,
        summary=summary,
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gen5 Phase 1B R22/C22 construction only; no confirmation or inference."
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_construction(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + summary["result"])
    print("SOURCE_PAIR_COUNT=" + str(summary["source_pair_count"]))
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("CHECKPOINT_LOAD_COUNT_THIS_RUN=1")
    print("Q_COMPUTED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("RAW_NATIVE_VECTORS_PERSISTED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
