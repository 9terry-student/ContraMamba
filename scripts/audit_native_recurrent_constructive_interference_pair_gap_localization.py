#!/usr/bin/env python3
"""Exact pair-gap localization of recurrent cross-token interference.

This audit is bounded to the frozen raw_write -> recurrent_state transition.
It extends the validated recurrent-kernel decomposition by resolving the exact
cross-token interference term C into source-token pair gaps d = sigma - tau.

No downstream readout/gate/out-projection/head transport is executed. No
training, optimizer step, checkpoint mutation, projector refit, confirmatory
population, or VitaminC access is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import audit_native_recurrent_transport_kernel_decomposition as kernel  # noqa: E402

transport = kernel.transport

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_COMMIT = "6e2cbeec8c6a46bb7b3a9b11d8774bea6b30c7a8"
DESIGN_PATH = (
    "reports/"
    "native_recurrent_constructive_interference_pair_gap_localization_design_candidate.md"
)
DESIGN_BLOB = "29e71b2cacd103a4a946ed626542cdd5e6f9caaf"

VALIDATED_EVIDENCE_COMMIT = "d64a4f6f60419de529eb239516b82ae62e613b17"
VALIDATED_KERNEL_SCRIPT_PATH = (
    "scripts/audit_native_recurrent_transport_kernel_decomposition.py"
)
VALIDATED_KERNEL_SCRIPT_BLOB = "d7a5a5369eea425d40e63faf7228532581f9d66d"

VALIDATED_KERNEL_RUN = "gen5-recurrent-kernel-decomposition-09577cf-r1"
VALIDATED_KERNEL_ROOT = (
    ROOT
    / "reports/native_recurrent_transport_kernel_decomposition_runs"
    / VALIDATED_KERNEL_RUN
)
VALIDATED_KERNEL_FILES = {
    "recurrent_kernel_decomposition_summary.json":
        "4072aec0f92a66ccb1435c4deb600bd60e58fbc078d646c2fd91a9cea295ba96",
    "orientation_kernel_decomposition_metrics.jsonl":
        "939a7411c9c30ad8172a01bf1a5d568e637e45866f45865721dc0c600a5107d5",
    "shard_manifest.json":
        "b1efe53d34c401d2a4ab6de0560973f91b77f16eab4dcee5376effef3246e78c",
}
VALIDATED_KERNEL_PROVENANCE_BLOB = "22fb65a143d95b9e04ca703147539819e6ed1fc9"

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_native_recurrent_constructive_interference_pair_gap_localization.py",
        "tests/test_native_recurrent_constructive_interference_pair_gap_localization.py",
    }
)

OUTPUT_SCHEMA = "GEN5_NATIVE_RECURRENT_PAIR_GAP_LOCALIZATION_V1"
WORKER_SCHEMA = "GEN5_NATIVE_RECURRENT_PAIR_GAP_LOCALIZATION_WORKER_V1"
PROVENANCE_SCHEMA = "GEN5_NATIVE_RECURRENT_PAIR_GAP_LOCALIZATION_PROVENANCE_V1"
SHARD_SCHEMA = "GEN5_NATIVE_RECURRENT_PAIR_GAP_LOCALIZATION_SHARD_V1"

PAIR_GAP_ENERGY_RTOL = kernel.REPLAY_ENERGY_RTOL
PAIR_GAP_ENERGY_ATOL = kernel.REPLAY_ENERGY_ATOL
PAIR_GAP_LOG_ATOL = kernel.REPLAY_LOG_ATOL
RAW_COMPONENT_MATCH_RTOL = kernel.RAW_COMPONENT_MATCH_RTOL
ENERGY_EPSILON = kernel.ENERGY_EPSILON
FFT_CHUNK_WIDTH = kernel.FFT_CHUNK_WIDTH
FAST_FLOAT32_HALF_SPAN_LIMIT = 60.0

FACTOR_SEEDS = kernel.FACTOR_SEEDS
FULL_FACTORIAL_CELLS = kernel.FULL_FACTORIAL_CELLS
DEV_ROWS = kernel.DEV_ROWS
VALID_TOKEN_COUNT = kernel.VALID_TOKEN_COUNT
BATCH_ROWS = kernel.BATCH_ROWS
TARGETS_PER_TRANSPORT = kernel.TARGETS_PER_TRANSPORT
WORKER_ROW_RANGES = kernel.WORKER_ROW_RANGES


class PairGapLocalizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PairGapLocalizationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PairGapLocalizationError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def cell_name(cell: tuple[int, int]) -> str:
    return kernel.cell_name(cell)


def all_orientations() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return kernel.all_orientations()


def worker_row_range(worker_id: int) -> tuple[int, int]:
    return kernel.worker_row_range(worker_id)


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), ENERGY_EPSILON)


def authenticate_repo(
    *,
    expected_head: str | None,
    allow_implementation_worktree: bool,
) -> str:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH:{branch}")
    if expected_head is not None:
        require(head == expected_head, f"HEAD:{head}:EXPECTED:{expected_head}")

    require(
        git_rc("merge-base", "--is-ancestor", DESIGN_COMMIT, head) == 0,
        "DESIGN_NOT_ANCESTOR",
    )
    require(
        git_rc("merge-base", "--is-ancestor", VALIDATED_EVIDENCE_COMMIT, head) == 0,
        "VALIDATED_EVIDENCE_NOT_ANCESTOR",
    )
    require(
        git("rev-parse", f"HEAD:{DESIGN_PATH}") == DESIGN_BLOB,
        "DESIGN_BLOB_DRIFT",
    )
    require(
        git("rev-parse", f"HEAD:{VALIDATED_KERNEL_SCRIPT_PATH}")
        == VALIDATED_KERNEL_SCRIPT_BLOB,
        "VALIDATED_KERNEL_SCRIPT_DRIFT",
    )

    for name, digest in VALIDATED_KERNEL_FILES.items():
        path = VALIDATED_KERNEL_ROOT / name
        require(path.is_file(), f"VALIDATED_KERNEL_ARTIFACT_MISSING:{name}")
        require(
            sha256_file(path) == digest,
            f"VALIDATED_KERNEL_ARTIFACT_SHA:{name}",
        )

    provenance_path = VALIDATED_KERNEL_ROOT / "run_provenance.json"
    require(provenance_path.is_file(), "VALIDATED_KERNEL_PROVENANCE_MISSING")
    require(
        git("rev-parse", f"HEAD:{provenance_path.relative_to(ROOT).as_posix()}")
        == VALIDATED_KERNEL_PROVENANCE_BLOB,
        "VALIDATED_KERNEL_PROVENANCE_BLOB_DRIFT",
    )

    dirty = transport.status_paths()
    if allow_implementation_worktree:
        require(
            dirty <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(dirty)}",
        )
    else:
        require(not dirty, f"WORKTREE_NOT_CLEAN:{sorted(dirty)}")
    return head


def _load_validated_kernel_reference(
) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = VALIDATED_KERNEL_ROOT / "orientation_kernel_decomposition_metrics.jsonl"
    require(
        sha256_file(path) == VALIDATED_KERNEL_FILES[path.name],
        "VALIDATED_KERNEL_METRICS_SHA",
    )
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["group"]), str(row["source"]), str(row["target"]))
        require(key not in rows, f"VALIDATED_KERNEL_DUPLICATE:{key}")
        rows[key] = row
    require(len(rows) == 36, f"VALIDATED_KERNEL_REFERENCE_COUNT:{len(rows)}")
    return rows


def validate_static_contract() -> None:
    kernel.validate_static_contract()
    require(FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(len(all_orientations()) == 36, "ORIENTATION_COUNT")
    require(
        sum(1 for group, _, _ in all_orientations() if group == "PRIMARY_A") == 18,
        "PRIMARY_ORIENTATION_COUNT",
    )
    require(
        sum(1 for group, _, _ in all_orientations() if group == "CONTROL_R") == 18,
        "CONTROL_ORIENTATION_COUNT",
    )
    require(worker_row_range(0) == (0, 416), "WORKER0_RANGE")
    require(worker_row_range(1) == (416, 840), "WORKER1_RANGE")
    require(TARGETS_PER_TRANSPORT == 2, "TARGETS_PER_TRANSPORT")
    require(FFT_CHUNK_WIDTH > 0, "FFT_CHUNK_WIDTH")
    require(FAST_FLOAT32_HALF_SPAN_LIMIT > 0.0, "FAST_SPAN_LIMIT")
    _load_validated_kernel_reference()


def _backward_survival_factor(
    *,
    log_a_flat: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    require(log_a_flat.ndim == 3, "SURVIVAL_LOG_A_RANK")
    batch, seq_len, width = log_a_flat.shape
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "SURVIVAL_MASK_SHAPE",
    )
    require(
        bool(torch.all(torch.isfinite(log_a_flat)).item()),
        "SURVIVAL_LOG_A_NONFINITE",
    )
    require(
        bool(torch.all(log_a_flat <= 0.0).item()),
        "SURVIVAL_LOG_A_POSITIVE",
    )

    mask = attention_mask.to(device=log_a_flat.device, dtype=torch.float32)
    log_native = log_a_flat.to(dtype=torch.float32)
    output = torch.empty(
        (batch, seq_len, width),
        device=log_a_flat.device,
        dtype=torch.float32,
    )
    carry = torch.zeros(
        (batch, width),
        device=log_a_flat.device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for sigma in range(seq_len - 1, -1, -1):
            if sigma + 1 < seq_len:
                carry = (
                    mask[:, sigma, None]
                    + torch.exp(2.0 * log_native[:, sigma + 1]) * carry
                )
            else:
                carry = mask[:, sigma, None]
            output[:, sigma] = carry

    require(
        bool(torch.all(torch.isfinite(output)).item()),
        "SURVIVAL_NONFINITE",
    )
    require(bool(torch.all(output >= 0.0).item()), "SURVIVAL_NEGATIVE")
    return output


def _pair_gap_strict_dyadic_fft(
    *,
    w_chunk: torch.Tensor,
    p_chunk: torch.Tensor,
    survival_chunk: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Signed strict-pair interference via overflow-safe dyadic GPU FFT.

    The strict source-token triangle tau < sigma is partitioned into dyadic
    cross-half rectangles. Every ordered token pair appears in exactly one
    rectangle. Native stable recurrence gives p_sigma <= p_tau, so the
    per-rectangle shift keeps both exponential factors non-positive.
    """
    require(w_chunk.ndim == 4, "DYADIC_W_RANK")
    k_count, batch, observed_seq_len, width = w_chunk.shape
    require(observed_seq_len == seq_len, "DYADIC_SEQ_LEN")
    require(
        tuple(p_chunk.shape) == (batch, seq_len, width),
        "DYADIC_P_SHAPE",
    )
    require(
        tuple(survival_chunk.shape) == (batch, seq_len, width),
        "DYADIC_SURVIVAL_SHAPE",
    )
    require(seq_len > 0, "DYADIC_EMPTY_SEQUENCE")

    padded_len = 1 << ((seq_len - 1).bit_length())
    if padded_len > seq_len:
        pad_count = padded_len - seq_len
        w_pad = torch.cat(
            (
                w_chunk,
                torch.zeros(
                    (k_count, batch, pad_count, width),
                    device=w_chunk.device,
                    dtype=w_chunk.dtype,
                ),
            ),
            dim=2,
        )
        p_pad = torch.cat(
            (
                p_chunk,
                p_chunk[:, -1:, :].expand(batch, pad_count, width),
            ),
            dim=1,
        )
        survival_pad = torch.cat(
            (
                survival_chunk,
                torch.zeros(
                    (batch, pad_count, width),
                    device=survival_chunk.device,
                    dtype=survival_chunk.dtype,
                ),
            ),
            dim=1,
        )
    else:
        w_pad = w_chunk
        p_pad = p_chunk
        survival_pad = survival_chunk

    gap = torch.zeros(
        (k_count, seq_len),
        device=w_chunk.device,
        dtype=torch.float64,
    )

    block_len = 2
    while block_len <= padded_len:
        half = block_len // 2
        block_count = padded_len // block_len

        p_blocks = p_pad.reshape(batch, block_count, block_len, width)
        w_blocks = w_pad.reshape(
            k_count,
            batch,
            block_count,
            block_len,
            width,
        )
        survival_blocks = survival_pad.reshape(
            batch,
            block_count,
            block_len,
            width,
        )

        left_p = p_blocks[:, :, :half, :]
        right_p = p_blocks[:, :, half:, :]
        shift = torch.amin(left_p, dim=2)

        left_exponent = -left_p + shift[:, :, None, :]
        right_exponent = right_p - shift[:, :, None, :]
        require(
            bool(torch.all(left_exponent <= 0.0).item()),
            "DYADIC_LEFT_SCALE_SIGN",
        )
        require(
            bool(torch.all(right_exponent <= 0.0).item()),
            "DYADIC_RIGHT_SCALE_SIGN",
        )

        x_scale = torch.exp(left_exponent).to(dtype=w_chunk.dtype)
        y_scale = torch.exp(right_exponent).to(dtype=w_chunk.dtype)
        require(
            bool(torch.all(torch.isfinite(x_scale)).item()),
            "DYADIC_X_SCALE_NONFINITE",
        )
        require(
            bool(torch.all(torch.isfinite(y_scale)).item()),
            "DYADIC_Y_SCALE_NONFINITE",
        )

        source = w_blocks[:, :, :, :half, :] * x_scale.unsqueeze(0)
        target = (
            w_blocks[:, :, :, half:, :]
            * survival_blocks[:, :, half:, :].to(dtype=w_chunk.dtype).unsqueeze(0)
            * y_scale.unsqueeze(0)
        )

        n_fft = 1 << ((2 * half - 1).bit_length())
        source_fft = torch.fft.rfft(source, n=n_fft, dim=3)
        target_fft = torch.fft.rfft(target, n=n_fft, dim=3)
        correlation = torch.fft.irfft(
            torch.conj(source_fft) * target_fft,
            n=n_fft,
            dim=3,
        )
        correlation_sum = 2.0 * torch.sum(
            correlation,
            dim=(1, 2, 4),
            dtype=torch.float64,
        )

        negative_stop = min(half, seq_len)
        if negative_stop > 1:
            gap[:, 1:negative_stop] += correlation_sum[
                :,
                n_fft + 1 - half : n_fft + negative_stop - half,
            ]

        positive_stop = min(block_len, seq_len)
        if half < positive_stop:
            gap[:, half:positive_stop] += correlation_sum[
                :,
                : positive_stop - half,
            ]

        block_len *= 2

    return gap


def _pair_gap_interference_fft(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    shape: Any,
) -> tuple[torch.Tensor, float]:
    """Return signed C(d), with index 0 reserved and identically zero."""
    require(component.ndim == 4, "PAIR_COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(
        width == shape.intermediate_size * shape.state_size,
        "PAIR_WIDTH",
    )
    require(
        tuple(log_a.shape)
        == (batch, seq_len, shape.intermediate_size, shape.state_size),
        "PAIR_LOG_A_SHAPE",
    )
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "PAIR_MASK_SHAPE",
    )
    require(
        bool(torch.all(torch.isfinite(log_a)).item()),
        "PAIR_LOG_A_NONFINITE",
    )
    require(
        bool(torch.all(log_a <= 0.0).item()),
        "PAIR_LOG_A_POSITIVE",
    )

    w = component.reshape(k_count, batch, seq_len, width)
    log_a_flat = log_a.reshape(batch, seq_len, width)
    target_mask = attention_mask.to(device=component.device, dtype=torch.bool)
    source_active = (
        torch.flip(
            torch.cumsum(
                torch.flip(target_mask.to(dtype=torch.int64), dims=(1,)),
                dim=1,
            ),
            dims=(1,),
        )
        > 0
    )
    require(
        bool(torch.all(torch.any(source_active, dim=1)).item()),
        "PAIR_EMPTY_VALID_SEQUENCE",
    )

    require(
        tuple(survival.shape) == (batch, seq_len, width),
        "PAIR_SURVIVAL_SHAPE",
    )
    require(
        bool(torch.all(torch.isfinite(survival)).item()),
        "PAIR_SURVIVAL_NONFINITE",
    )
    require(bool(torch.all(survival >= 0.0).item()), "PAIR_SURVIVAL_NEGATIVE")
    n_fft = 1 << ((2 * seq_len - 1).bit_length())
    gap = torch.zeros(
        (k_count, seq_len),
        device=component.device,
        dtype=torch.float64,
    )
    max_half_span = 0.0

    for start in range(0, width, FFT_CHUNK_WIDTH):
        stop = min(start + FFT_CHUNK_WIDTH, width)
        p = torch.cumsum(
            log_a_flat[:, :, start:stop].to(dtype=torch.float64),
            dim=1,
        )
        active = source_active[:, :, None]
        p_max = torch.amax(
            torch.where(active, p, torch.full_like(p, -torch.inf)),
            dim=1,
        )
        p_min = torch.amin(
            torch.where(active, p, torch.full_like(p, torch.inf)),
            dim=1,
        )
        require(
            bool(torch.all(torch.isfinite(p_max) & torch.isfinite(p_min)).item()),
            "PAIR_ACTIVE_PREFIX_NONFINITE",
        )
        half_span = 0.5 * (p_max - p_min)
        chunk_max = float(torch.amax(half_span).item())
        max_half_span = max(max_half_span, chunk_max)

        w_native = w[:, :, :, start:stop].to(dtype=torch.float32)
        w_active = torch.where(
            source_active[None, :, :, None],
            w_native,
            torch.zeros_like(w_native),
        )
        survival_chunk = survival[:, :, start:stop]

        if chunk_max <= FAST_FLOAT32_HALF_SPAN_LIMIT:
            center = 0.5 * (
                p_max.to(dtype=torch.float32)
                + p_min.to(dtype=torch.float32)
            )
            p_calc = p.to(dtype=torch.float32)
            p_safe = torch.where(
                active,
                p_calc,
                center[:, None, :],
            )
            x_scale = torch.exp(-p_safe + center[:, None, :])
            y_scale = torch.exp(p_safe - center[:, None, :])
            require(
                bool(torch.all(torch.isfinite(x_scale)).item()),
                f"PAIR_X_SCALE_NONFINITE:{chunk_max}",
            )
            require(
                bool(torch.all(torch.isfinite(y_scale)).item()),
                f"PAIR_Y_SCALE_NONFINITE:{chunk_max}",
            )

            source = w_active * x_scale.unsqueeze(0)
            target = (
                w_active
                * survival_chunk.unsqueeze(0)
                * y_scale.unsqueeze(0)
            )

            source_fft = torch.fft.rfft(source, n=n_fft, dim=2)
            target_fft = torch.fft.rfft(target, n=n_fft, dim=2)
            correlation = torch.fft.irfft(
                torch.conj(source_fft) * target_fft,
                n=n_fft,
                dim=2,
            )
            gap[:, 1:] += 2.0 * torch.sum(
                correlation[:, :, 1:seq_len, :],
                dim=(1, 3),
                dtype=torch.float64,
            )
            continue

        gap += _pair_gap_strict_dyadic_fft(
            w_chunk=w_active,
            p_chunk=p,
            survival_chunk=survival_chunk,
            seq_len=seq_len,
        )

    require(
        bool(torch.all(torch.isfinite(gap)).item()),
        "PAIR_GAP_NONFINITE",
    )
    require(
        bool(torch.all(gap[:, 0] == 0.0).item()),
        "PAIR_GAP_ZERO_INDEX_NONZERO",
    )
    return gap, max_half_span


def _component_pair_gap_stats(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    shape: Any,
) -> list[dict[str, Any]]:
    require(component.ndim == 4, "COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(width == shape.state_width, "COMPONENT_STATE_WIDTH")
    require(
        shape.state_width == shape.intermediate_size * shape.state_size,
        "STATE_WIDTH_FACTORIZATION",
    )
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "COMPONENT_MASK_SHAPE",
    )

    w = component.reshape(
        k_count,
        batch,
        seq_len,
        shape.intermediate_size,
        shape.state_size,
    )
    require(
        tuple(log_a.shape)
        == (batch, seq_len, shape.intermediate_size, shape.state_size),
        "COMPONENT_LOG_A_SHAPE",
    )
    require(
        tuple(survival.shape) == (batch, seq_len, width),
        "COMPONENT_SURVIVAL_SHAPE",
    )

    state = torch.zeros(
        (k_count, batch, shape.intermediate_size, shape.state_size),
        device=component.device,
        dtype=component.dtype,
    )
    self_state = torch.zeros_like(state)
    raw = torch.zeros(k_count, device=component.device, dtype=torch.float64)
    recurrent = torch.zeros_like(raw)
    self_energy = torch.zeros_like(raw)

    with torch.no_grad():
        for token_index in range(seq_len):
            write_t = w[:, :, token_index]
            a_t = torch.exp(log_a[:, token_index]).to(dtype=component.dtype)
            state = a_t.unsqueeze(0) * state + write_t
            self_state = a_t.square().unsqueeze(0) * self_state + write_t.square()

            valid_t = attention_mask[:, token_index].to(
                device=component.device,
                dtype=component.dtype,
            ).reshape(1, batch, 1, 1)
            raw += torch.sum(
                write_t.square() * valid_t,
                dim=(1, 2, 3),
                dtype=torch.float64,
            )
            recurrent += torch.sum(
                state.square() * valid_t,
                dim=(1, 2, 3),
                dtype=torch.float64,
            )
            self_energy += torch.sum(
                self_state * valid_t,
                dim=(1, 2, 3),
                dtype=torch.float64,
            )

    pair_gap, max_half_span = _pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        shape=shape,
    )

    rows: list[dict[str, Any]] = []
    for index in range(k_count):
        c_direct = float((recurrent[index] - self_energy[index]).item())
        c_gap = float(torch.sum(pair_gap[index, 1:], dtype=torch.float64).item())
        c_abs = abs(c_gap - c_direct)
        c_rel = _relative_error(c_gap, c_direct)
        require(
            c_abs <= PAIR_GAP_ENERGY_ATOL or c_rel <= PAIR_GAP_ENERGY_RTOL,
            f"PAIR_GAP_C_RECON:{index}:{c_abs}:{c_rel}",
        )

        s_value = float(self_energy[index].item())
        e_value = float(recurrent[index].item())
        require(s_value > 0.0 and e_value > 0.0, f"PAIR_NONPOSITIVE_ENERGY:{index}")
        q_direct = e_value / s_value
        q_gap = 1.0 + c_gap / s_value
        q_abs = abs(q_gap - q_direct)
        q_rel = _relative_error(q_gap, q_direct)
        require(
            q_abs <= PAIR_GAP_ENERGY_ATOL or q_rel <= PAIR_GAP_ENERGY_RTOL,
            f"PAIR_GAP_Q_RECON:{index}:{q_abs}:{q_rel}",
        )

        rows.append(
            {
                "R": float(raw[index].item()),
                "S": s_value,
                "E": e_value,
                "C": c_direct,
                "Q": q_direct,
                "pair_gap_C": [
                    float(value)
                    for value in pair_gap[index, 1:].detach().to("cpu").tolist()
                ],
                "pair_gap_C_reconstruction_abs": c_abs,
                "pair_gap_Q_reconstruction_abs": q_abs,
                "pair_gap_fft_half_log_span_max": max_half_span,
            }
        )
    return rows


def _empty_orientation(
    group: str,
    source: tuple[int, int],
    target: tuple[int, int],
) -> dict[str, Any]:
    return {
        "group": group,
        "source": cell_name(source),
        "target": cell_name(target),
        "example_count": 0,
        "raw_reconstruction_max_abs": 0.0,
        "R_visible": 0.0,
        "R_complement": 0.0,
        "S_visible": 0.0,
        "S_complement": 0.0,
        "E_visible": 0.0,
        "E_complement": 0.0,
        "pair_gap_C_visible": [],
        "pair_gap_C_complement": [],
        "pair_gap_C_reconstruction_abs_max": 0.0,
        "pair_gap_Q_reconstruction_abs_max": 0.0,
        "pair_gap_fft_half_log_span_max": 0.0,
    }


def _accumulate_component_row(
    accumulator: dict[str, Any],
    *,
    prefix: str,
    stats: Mapping[str, Any],
) -> None:
    require(prefix in {"visible", "complement"}, f"COMPONENT_PREFIX:{prefix}")
    accumulator[f"R_{prefix}"] += float(stats["R"])
    accumulator[f"S_{prefix}"] += float(stats["S"])
    accumulator[f"E_{prefix}"] += float(stats["E"])

    key = f"pair_gap_C_{prefix}"
    observed = [float(x) for x in stats["pair_gap_C"]]
    if not accumulator[key]:
        accumulator[key] = [0.0 for _ in observed]
    require(
        len(accumulator[key]) == len(observed),
        f"PAIR_GAP_LENGTH:{prefix}",
    )
    for index, value in enumerate(observed):
        accumulator[key][index] += value

    accumulator["pair_gap_C_reconstruction_abs_max"] = max(
        float(accumulator["pair_gap_C_reconstruction_abs_max"]),
        float(stats["pair_gap_C_reconstruction_abs"]),
    )
    accumulator["pair_gap_Q_reconstruction_abs_max"] = max(
        float(accumulator["pair_gap_Q_reconstruction_abs_max"]),
        float(stats["pair_gap_Q_reconstruction_abs"]),
    )
    accumulator["pair_gap_fft_half_log_span_max"] = max(
        float(accumulator["pair_gap_fft_half_log_span_max"]),
        float(stats["pair_gap_fft_half_log_span_max"]),
    )


def _gap_band_summary(values: Sequence[float]) -> dict[str, float]:
    observed = [float(x) for x in values]

    def band(first_gap: int, last_gap: int | None) -> float:
        start = max(first_gap - 1, 0)
        stop = None if last_gap is None else last_gap
        return sum(observed[start:stop])

    return {
        "gap1": band(1, 1),
        "gap2_4": band(2, 4),
        "gap5_8": band(5, 8),
        "gap9plus": band(9, None),
        "total": sum(observed),
    }


def _replay_scalar(
    *,
    name: str,
    observed: float,
    expected: float,
    log_metric: bool = False,
) -> dict[str, Any]:
    abs_error = abs(observed - expected)
    if log_metric:
        passed = abs_error <= PAIR_GAP_LOG_ATOL
        rel_error = _relative_error(observed, expected)
    else:
        rel_error = _relative_error(observed, expected)
        passed = (
            abs_error <= PAIR_GAP_ENERGY_ATOL
            or rel_error <= PAIR_GAP_ENERGY_RTOL
        )
    require(passed, f"VALIDATED_KERNEL_REPLAY:{name}:{abs_error}:{rel_error}")
    return {
        "observed": observed,
        "expected": expected,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "pass": True,
    }


def _finalize_orientation(
    accumulator: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    require(int(accumulator["example_count"]) == DEV_ROWS, "FINAL_EXAMPLE_COUNT")
    values = {
        key: float(accumulator[key])
        for key in (
            "R_visible",
            "R_complement",
            "S_visible",
            "S_complement",
            "E_visible",
            "E_complement",
        )
    }
    require(all(value > 0.0 for value in values.values()), "NONPOSITIVE_ENERGY")

    c_visible = values["E_visible"] - values["S_visible"]
    c_complement = values["E_complement"] - values["S_complement"]
    q_visible = values["E_visible"] / values["S_visible"]
    q_complement = values["E_complement"] / values["S_complement"]
    require(q_visible > 0.0 and q_complement > 0.0, "NONPOSITIVE_Q")
    l_interference = math.log(q_complement / q_visible)

    c_gap_visible = [float(x) for x in accumulator["pair_gap_C_visible"]]
    c_gap_complement = [float(x) for x in accumulator["pair_gap_C_complement"]]
    require(
        len(c_gap_visible) == len(c_gap_complement) and len(c_gap_visible) > 0,
        "FINAL_PAIR_GAP_LENGTH",
    )

    h_gap_visible = [value / values["S_visible"] for value in c_gap_visible]
    h_gap_complement = [
        value / values["S_complement"] for value in c_gap_complement
    ]

    c_visible_gap = sum(c_gap_visible)
    c_complement_gap = sum(c_gap_complement)
    q_visible_gap = 1.0 + sum(h_gap_visible)
    q_complement_gap = 1.0 + sum(h_gap_complement)

    c_visible_abs = abs(c_visible_gap - c_visible)
    c_complement_abs = abs(c_complement_gap - c_complement)
    q_visible_abs = abs(q_visible_gap - q_visible)
    q_complement_abs = abs(q_complement_gap - q_complement)

    for name, observed, expected in (
        ("C_visible", c_visible_gap, c_visible),
        ("C_complement", c_complement_gap, c_complement),
        ("Q_visible", q_visible_gap, q_visible),
        ("Q_complement", q_complement_gap, q_complement),
    ):
        require(
            abs(observed - expected) <= PAIR_GAP_ENERGY_ATOL
            or _relative_error(observed, expected) <= PAIR_GAP_ENERGY_RTOL,
            f"FINAL_PAIR_GAP_RECON:{name}:{observed}:{expected}",
        )

    replay: dict[str, Any] = {}
    for key in (
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    ):
        replay[key] = _replay_scalar(
            name=key,
            observed=values[key],
            expected=float(reference[key]),
        )
    for key, observed in (
        ("C_visible", c_visible),
        ("C_complement", c_complement),
        ("Q_visible", q_visible),
        ("Q_complement", q_complement),
    ):
        replay[key] = _replay_scalar(
            name=key,
            observed=observed,
            expected=float(reference[key]),
        )
    replay["L_interference"] = _replay_scalar(
        name="L_interference",
        observed=l_interference,
        expected=float(reference["L_interference"]),
        log_metric=True,
    )

    return {
        "group": str(accumulator["group"]),
        "source": str(accumulator["source"]),
        "target": str(accumulator["target"]),
        "example_count": int(accumulator["example_count"]),
        "raw_reconstruction_max_abs": float(
            accumulator["raw_reconstruction_max_abs"]
        ),
        **values,
        "C_visible": c_visible,
        "C_complement": c_complement,
        "Q_visible": q_visible,
        "Q_complement": q_complement,
        "log_Q_visible": math.log(q_visible),
        "log_Q_complement": math.log(q_complement),
        "L_interference": l_interference,
        "pair_gap_index_origin": 1,
        "C_gap_visible": c_gap_visible,
        "C_gap_complement": c_gap_complement,
        "H_gap_visible": h_gap_visible,
        "H_gap_complement": h_gap_complement,
        "H_band_visible": _gap_band_summary(h_gap_visible),
        "H_band_complement": _gap_band_summary(h_gap_complement),
        "pair_gap_C_reconstruction_abs": {
            "visible": c_visible_abs,
            "complement": c_complement_abs,
        },
        "pair_gap_Q_reconstruction_abs": {
            "visible": q_visible_abs,
            "complement": q_complement_abs,
        },
        "batch_pair_gap_C_reconstruction_abs_max": float(
            accumulator["pair_gap_C_reconstruction_abs_max"]
        ),
        "batch_pair_gap_Q_reconstruction_abs_max": float(
            accumulator["pair_gap_Q_reconstruction_abs_max"]
        ),
        "pair_gap_fft_half_log_span_max": float(
            accumulator["pair_gap_fft_half_log_span_max"]
        ),
        "validated_kernel_replay": replay,
    }


def _worker_payload_path(scratch_root: Path, worker_id: int) -> Path:
    return scratch_root / f"worker{worker_id}" / "worker_result.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".tmp.",
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _worker_sha_sidecar_bytes(digest: str) -> bytes:
    require(
        len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "WORKER_SHA_DIGEST_FORMAT",
    )
    return (digest + "\n").encode("utf-8")


def _validate_runtime_head(args: argparse.Namespace) -> str:
    require(args.expected_head is not None, "EXPECTED_HEAD_REQUIRED")
    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(
        args.expected_head == args.implementation_freeze_commit,
        "EXECUTION_HEAD_MUST_EQUAL_IMPLEMENTATION_FREEZE",
    )
    return authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=False,
    )


def _expected_counts() -> dict[str, Any]:
    workers: dict[str, Any] = {}
    groups_per_source = math.ceil(4 / TARGETS_PER_TRANSPORT)
    for worker_id in (0, 1):
        start, stop = worker_row_range(worker_id)
        batches = math.ceil((stop - start) / BATCH_ROWS)
        groups = 9 * groups_per_source * batches
        workers[str(worker_id)] = {
            "row_start": start,
            "row_stop": stop,
            "row_count": stop - start,
            "batches": batches,
            "source_gradient_forwards": 9 * batches,
            "pair_gap_group_decompositions": groups,
            "orientation_component_decompositions": groups * TARGETS_PER_TRANSPORT,
            "downstream_stage_transport_calls": 0,
            "final_head_transport_calls": 0,
        }
    return {
        "batch_rows": BATCH_ROWS,
        "targets_per_group": TARGETS_PER_TRANSPORT,
        "workers": workers,
        "total_source_gradient_forwards": sum(
            row["source_gradient_forwards"] for row in workers.values()
        ),
        "total_pair_gap_group_decompositions": sum(
            row["pair_gap_group_decompositions"] for row in workers.values()
        ),
        "total_downstream_stage_transport_calls": 0,
        "total_final_head_transport_calls": 0,
    }


def run_static_contract_check(args: argparse.Namespace) -> None:
    head = authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=True,
    )
    validate_static_contract()
    print("GEN5_NATIVE_RECURRENT_PAIR_GAP_STATIC_CONTRACT_PASS")
    print(f"HEAD={head}")
    print("PRIMARY_ORIENTATIONS=18")
    print("CONTROL_ORIENTATIONS=18")
    print("TOTAL_ORIENTATIONS=36")
    print("WORKER0_ROWS=0:416")
    print("WORKER1_ROWS=416:840")
    print("VALIDATED_KERNEL_REFERENCE=AUTHENTICATED")
    print("RECURRENCE_ONLY=True")
    print("DOWNSTREAM_STAGE_TRANSPORT_CALLS=0")
    print("FINAL_HEAD_TRANSPORT_CALLS=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")


def run_preflight(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    transport.legacy._validate_two_t4s()
    transport._runtime_inputs(args)
    print("GEN5_NATIVE_RECURRENT_PAIR_GAP_CUDA_PREFLIGHT_PASS")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("EXECUTION_ENGINE=RECURRENT_PAIR_GAP_EXACT_INTERFERENCE_V1")
    print("VALIDATED_KERNEL_REFERENCE=AUTHENTICATED")
    print("RECURRENCE_ONLY=True")
    print("DOWNSTREAM_STAGE_TRANSPORT_CALLS=0")
    print("FINAL_HEAD_TRANSPORT_CALLS=0")
    print("TRAINING_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print("EXPECTED_COUNTS=" + json.dumps(_expected_counts(), sort_keys=True))


def run_worker(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    require(torch.cuda.device_count() == 1, "WORKER_VISIBLE_GPU_COUNT_MUST_BE_ONE")

    worker_id = int(args.worker_id)
    scratch_root = Path(args.scratch_root)
    worker_root = scratch_root / f"worker{worker_id}"
    require(not worker_root.exists(), f"WORKER_OUTPUT_COLLISION:{worker_root}")

    (
        _encoded,
        model,
        wrapper,
        strong_mask,
        planes,
        features,
        active,
        targets,
    ) = transport._prepare_worker_runtime(args)
    weights = transport._cell_weights()
    local_sources = tuple(FULL_FACTORIAL_CELLS)
    orientations = all_orientations()
    accumulators = {
        (group, source, target): _empty_orientation(group, source, target)
        for group, source, target in orientations
    }
    by_source = {
        source: tuple(row for row in orientations if row[1] == source)
        for source in local_sources
    }
    for source, rows in by_source.items():
        require(
            len(rows) == 4,
            f"SOURCE_ORIENTATION_COUNT:{cell_name(source)}",
        )

    row_start, row_stop = worker_row_range(worker_id)
    row_count = row_stop - row_start
    total_batches = math.ceil(row_count / BATCH_ROWS)
    valid_token_seen = 0

    for batch_index, start in enumerate(
        range(row_start, row_stop, BATCH_ROWS),
        start=1,
    ):
        stop = min(start + BATCH_ROWS, row_stop)
        batch_features = transport.legacy._phase_b_batch_features(
            features,
            start,
            stop,
        )
        batch_mask = batch_features["attention_mask"]
        valid_token_seen += int(torch.count_nonzero(batch_mask).item())

        context = transport.legacy._phase_b_prepare_common_context(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            stressor_active=active[start:stop],
            target_indices=targets[start:stop],
            strong_mask=strong_mask,
            planes=planes,
        )
        pair_log_a = kernel._discrete_log_a(
            context,
            shape=wrapper.correction.shape,
        ).to(device=batch_mask.device)
        pair_survival = _backward_survival_factor(
            log_a_flat=pair_log_a.reshape(
                pair_log_a.shape[0],
                pair_log_a.shape[1],
                -1,
            ),
            attention_mask=batch_mask,
        )

        raw_by_cell: dict[tuple[int, int], torch.Tensor] = {}
        with torch.no_grad():
            for cell in FULL_FACTORIAL_CELLS:
                a_weight, b_weight = weights[cell]
                raw_by_cell[cell] = transport.legacy._phase_b_raw_write(
                    context["mixer_input"],
                    batch_mask,
                    a_weight,
                    b_weight,
                ).detach()

        for source in local_sources:
            source_rows = by_source[source]
            raw_source = raw_by_cell[source]
            _, g_refute, g_support = transport._source_gradient_bundle(
                model=model,
                wrapper=wrapper,
                features=batch_features,
                context=context,
                raw_source=raw_source,
            )
            g_refute, g_support, pinv_cpu = transport._prepare_projector_pinv(
                grad_refute=g_refute,
                grad_support=g_support,
                attention_mask=batch_mask,
            )

            for group_start in range(
                0,
                len(source_rows),
                TARGETS_PER_TRANSPORT,
            ):
                group_rows = source_rows[
                    group_start : group_start + TARGETS_PER_TRANSPORT
                ]
                raw_targets = [raw_by_cell[row[2]] for row in group_rows]

                projector_accs = [
                    transport._empty_orientation(*row)
                    for row in group_rows
                ]
                visible, complement = transport._project_visible_group(
                    grad_refute=g_refute,
                    grad_support=g_support,
                    pinv_cpu=pinv_cpu,
                    attention_mask=batch_mask,
                    raw_source=raw_source,
                    raw_targets=raw_targets,
                    accumulators=projector_accs,
                )

                combined = torch.cat((visible, complement), dim=0)
                stats = _component_pair_gap_stats(
                    component=combined,
                    log_a=pair_log_a,
                    survival=pair_survival,
                    attention_mask=batch_mask,
                    shape=wrapper.correction.shape,
                )
                target_count = len(group_rows)

                for index, row in enumerate(group_rows):
                    accumulator = accumulators[row]
                    visible_stats = stats[index]
                    complement_stats = stats[target_count + index]
                    _accumulate_component_row(
                        accumulator,
                        prefix="visible",
                        stats=visible_stats,
                    )
                    _accumulate_component_row(
                        accumulator,
                        prefix="complement",
                        stats=complement_stats,
                    )

                    projector_acc = projector_accs[index]
                    accumulator["raw_reconstruction_max_abs"] = max(
                        float(accumulator["raw_reconstruction_max_abs"]),
                        float(projector_acc["raw_reconstruction_max_abs"]),
                    )
                    require(
                        float(projector_acc["raw_reconstruction_max_abs"])
                        <= transport.RAW_RECON_ATOL,
                        "RAW_RECONSTRUCTION_AUTH",
                    )

                    raw_visible = float(
                        projector_acc["stage"]["raw_write"]["D_visible"]
                    )
                    raw_complement = float(
                        projector_acc["stage"]["raw_write"]["D_complement"]
                    )
                    require(
                        _relative_error(
                            float(visible_stats["R"]),
                            raw_visible,
                        )
                        <= RAW_COMPONENT_MATCH_RTOL,
                        "VISIBLE_RAW_ENERGY_INTERNAL_REPLAY",
                    )
                    require(
                        _relative_error(
                            float(complement_stats["R"]),
                            raw_complement,
                        )
                        <= RAW_COMPONENT_MATCH_RTOL,
                        "COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY",
                    )
                    accumulator["example_count"] += stop - start

                del combined, visible, complement, stats, projector_accs

            del g_refute, g_support, pinv_cpu

        del raw_by_cell, pair_log_a, pair_survival
        print(
            "GEN5_RECURRENT_PAIR_GAP_PROGRESS "
            f"worker={worker_id} batch={batch_index}/{total_batches} "
            f"shard_rows={stop-row_start}/{row_count}",
            flush=True,
        )

    for accumulator in accumulators.values():
        require(
            int(accumulator["example_count"]) == row_count,
            "WORKER_ORIENTATION_EXAMPLE_COUNT",
        )
    require(
        not any(parameter.grad is not None for parameter in model.parameters()),
        "PARAMETER_GRADIENT_ACCUMULATED_FINAL",
    )

    result = {
        "schema_version": WORKER_SCHEMA,
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "worker_id": worker_id,
        "row_start": row_start,
        "row_stop": row_stop,
        "row_count": row_count,
        "orientation_count": 36,
        "orientation_accumulators": list(accumulators.values()),
        "valid_token_count": valid_token_seen,
        "execution_engine": "RECURRENT_PAIR_GAP_EXACT_INTERFERENCE_V1",
        "recurrence_only": True,
        "downstream_stage_transport_calls": 0,
        "final_head_transport_calls": 0,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }

    payload_path = _worker_payload_path(scratch_root, worker_id)
    _atomic_write(payload_path, canonical_json_bytes(result))
    digest = sha256_file(payload_path)
    _atomic_write(
        payload_path.with_suffix(".sha256"),
        _worker_sha_sidecar_bytes(digest),
    )
    print(
        "GEN5_NATIVE_RECURRENT_PAIR_GAP_WORKER_PASS "
        f"worker={worker_id} rows={row_start}:{row_stop} "
        f"orientations=36 sha256={digest}"
    )


def _read_worker(scratch_root: Path, worker_id: int) -> dict[str, Any]:
    path = _worker_payload_path(scratch_root, worker_id)
    sidecar = path.with_suffix(".sha256")
    require(path.is_file(), f"WORKER_RESULT_MISSING:{worker_id}")
    require(sidecar.is_file(), f"WORKER_SHA_MISSING:{worker_id}")
    digest = sidecar.read_text(encoding="utf-8").strip()
    require(sha256_file(path) == digest, f"WORKER_SHA_MISMATCH:{worker_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["schema_version"] == WORKER_SCHEMA, "WORKER_SCHEMA")
    require(payload["worker_id"] == worker_id, "WORKER_ID_MISMATCH")
    require(payload["orientation_count"] == 36, "WORKER_ORIENTATION_COUNT")
    return payload


def _merge_worker_accumulators(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    scalar_sum = (
        "example_count",
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    )
    scalar_max = (
        "raw_reconstruction_max_abs",
        "pair_gap_C_reconstruction_abs_max",
        "pair_gap_Q_reconstruction_abs_max",
        "pair_gap_fft_half_log_span_max",
    )
    vector_sum = (
        "pair_gap_C_visible",
        "pair_gap_C_complement",
    )

    for worker in workers:
        for row in worker["orientation_accumulators"]:
            key = (str(row["group"]), str(row["source"]), str(row["target"]))
            if key not in merged:
                merged[key] = {
                    "group": key[0],
                    "source": key[1],
                    "target": key[2],
                    **{name: 0 for name in scalar_sum},
                    **{name: 0.0 for name in scalar_max},
                    **{name: [] for name in vector_sum},
                }
            dst = merged[key]
            for name in scalar_sum:
                dst[name] += row[name]
            for name in scalar_max:
                dst[name] = max(float(dst[name]), float(row[name]))
            for name in vector_sum:
                observed = [float(x) for x in row[name]]
                if not dst[name]:
                    dst[name] = [0.0 for _ in observed]
                require(
                    len(dst[name]) == len(observed),
                    f"MERGE_VECTOR_LENGTH:{name}:{key}",
                )
                for index, value in enumerate(observed):
                    dst[name][index] += value

    require(len(merged) == 36, f"MERGED_ACCUMULATOR_COUNT:{len(merged)}")
    reference = _load_validated_kernel_reference()
    output = []
    for key in sorted(merged):
        require(key in reference, f"VALIDATED_REFERENCE_MISSING:{key}")
        output.append(_finalize_orientation(merged[key], reference[key]))
    return output


def _median(values: Sequence[float]) -> float:
    observed = sorted(float(x) for x in values)
    require(bool(observed), "EMPTY_MEDIAN")
    mid = len(observed) // 2
    if len(observed) % 2:
        return observed[mid]
    return 0.5 * (observed[mid - 1] + observed[mid])


def _grouped_ordered_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ("PRIMARY_A", "CONTROL_R"):
        subset = [row for row in rows if row["group"] == group]
        require(len(subset) == 18, f"GROUPED_ORIENTATION_COUNT:{group}")
        metrics: dict[str, Any] = {}
        for name in ("log_Q_visible", "log_Q_complement", "L_interference"):
            values = [float(row[name]) for row in subset]
            metrics[name] = {
                "count": len(values),
                "median": _median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        for component in ("visible", "complement"):
            for band in ("gap1", "gap2_4", "gap5_8", "gap9plus", "total"):
                values = [
                    float(row[f"H_band_{component}"][band])
                    for row in subset
                ]
                metrics[f"H_{component}_{band}"] = {
                    "count": len(values),
                    "median": _median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
        output[group] = metrics
    return output


def _mean(values: Sequence[float]) -> float:
    require(bool(values), "EMPTY_MEAN")
    return sum(float(x) for x in values) / len(values)


def _source_matched(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)
    require(len(by_source) == 9, "SOURCE_MATCHED_CELL_COUNT")

    output = []
    for source, source_rows in sorted(by_source.items()):
        primary = [row for row in source_rows if row["group"] == "PRIMARY_A"]
        control = [row for row in source_rows if row["group"] == "CONTROL_R"]
        require(
            len(primary) == 2 and len(control) == 2,
            f"SOURCE_MATCH_COUNTS:{source}",
        )

        scalar_metrics = {}
        for name in ("log_Q_visible", "log_Q_complement", "L_interference"):
            p = _mean([float(row[name]) for row in primary])
            c = _mean([float(row[name]) for row in control])
            scalar_metrics[name] = {
                "primary_log_mean": p,
                "control_log_mean": c,
                "primary_minus_control": p - c,
            }

        gap_metrics: dict[str, Any] = {}
        for component in ("visible", "complement"):
            p_vectors = [
                [float(x) for x in row[f"H_gap_{component}"]]
                for row in primary
            ]
            c_vectors = [
                [float(x) for x in row[f"H_gap_{component}"]]
                for row in control
            ]
            lengths = {len(vector) for vector in p_vectors + c_vectors}
            require(len(lengths) == 1, f"SOURCE_GAP_LENGTH:{source}:{component}")
            length = lengths.pop()
            delta = [
                _mean([vector[index] for vector in p_vectors])
                - _mean([vector[index] for vector in c_vectors])
                for index in range(length)
            ]
            gap_metrics[component] = {
                "primary_minus_control_H_by_gap": delta,
                "primary_minus_control_H_bands": _gap_band_summary(delta),
            }

        output.append(
            {
                "source": source,
                "metrics": scalar_metrics,
                "gap_metrics": gap_metrics,
            }
        )
    return output


def run_merge_only(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    scratch_root = Path(args.scratch_root)
    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")

    workers = [_read_worker(scratch_root, worker_id) for worker_id in (0, 1)]
    for worker in workers:
        require(
            worker["execution_head"] == args.expected_head,
            "MIXED_WORKER_HEAD",
        )
    require(
        sum(int(worker["valid_token_count"]) for worker in workers)
        == VALID_TOKEN_COUNT,
        "MERGED_VALID_TOKEN_COUNT",
    )

    rows = _merge_worker_accumulators(workers)
    require(len(rows) == 36, "MERGED_ORIENTATION_COUNT")
    require(
        all(
            all(bool(check["pass"]) for check in row["validated_kernel_replay"].values())
            for row in rows
        ),
        "VALIDATED_KERNEL_REPLAY_FAIL",
    )
    require(
        max(float(row["raw_reconstruction_max_abs"]) for row in rows)
        <= transport.RAW_RECON_ATOL,
        "MERGED_RAW_RECONSTRUCTION_FAIL",
    )

    grouped = _grouped_ordered_summary(rows)
    matched = _source_matched(rows)

    max_replay_rel = max(
        float(check["rel_error"])
        for row in rows
        for check in row["validated_kernel_replay"].values()
    )
    max_replay_abs = max(
        float(check["abs_error"])
        for row in rows
        for check in row["validated_kernel_replay"].values()
    )
    max_c_recon = max(
        max(float(x) for x in row["pair_gap_C_reconstruction_abs"].values())
        for row in rows
    )
    max_q_recon = max(
        max(float(x) for x in row["pair_gap_Q_reconstruction_abs"].values())
        for row in rows
    )

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_VALIDATED_RECURRENT_PAIR_GAP_LOCALIZATION",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_kernel_run": VALIDATED_KERNEL_RUN,
        "population": "FROZEN_PHASE3A_P0_DEV",
        "dev_rows": DEV_ROWS,
        "valid_token_count": VALID_TOKEN_COUNT,
        "primary_orientations": 18,
        "control_orientations": 18,
        "pair_gap_identity": "C=sum_d_C(d);Q=1+sum_d_H(d)",
        "gap_index_origin": 1,
        "descriptive_gap_bands": {
            "gap1": [1, 1],
            "gap2_4": [2, 4],
            "gap5_8": [5, 8],
            "gap9plus": [9, None],
        },
        "grouped_ordered_orientation_summary": grouped,
        "source_cell_matched_primary_minus_control": matched,
        "validated_kernel_replay": {
            "pass": True,
            "energy_rtol": PAIR_GAP_ENERGY_RTOL,
            "energy_atol": PAIR_GAP_ENERGY_ATOL,
            "log_atol": PAIR_GAP_LOG_ATOL,
            "max_rel_error": max_replay_rel,
            "max_abs_error": max_replay_abs,
        },
        "pair_gap_reconstruction": {
            "pass": True,
            "max_C_abs_error": max_c_recon,
            "max_Q_abs_error": max_q_recon,
        },
        "mechanism_classification_thresholds": None,
        "scientific_p_value_count": 0,
        "recurrence_only": True,
        "downstream_stage_transport_calls": 0,
        "final_head_transport_calls": 0,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }

    shard_manifest = {
        "schema_version": SHARD_SCHEMA,
        "execution_head": args.expected_head,
        "workers": {
            str(worker["worker_id"]): {
                "row_start": worker["row_start"],
                "row_stop": worker["row_stop"],
                "row_count": worker["row_count"],
                "orientation_count": worker["orientation_count"],
                "valid_token_count": worker["valid_token_count"],
                "worker_result_sha256": sha256_file(
                    _worker_payload_path(
                        scratch_root,
                        int(worker["worker_id"]),
                    )
                ),
            }
            for worker in workers
        },
        "total_orientations": 36,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = output_root / "pair_gap_localization_summary.json"
    rows_path = output_root / "orientation_pair_gap_metrics.jsonl"
    shard_path = output_root / "shard_manifest.json"
    provenance_path = output_root / "run_provenance.json"

    summary_path.write_bytes(canonical_json_bytes(summary))
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(row) for row in rows)
    )
    shard_path.write_bytes(canonical_json_bytes(shard_manifest))

    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "design_blob": DESIGN_BLOB,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_kernel_script_blob": VALIDATED_KERNEL_SCRIPT_BLOB,
        "validated_kernel_artifact_sha256": VALIDATED_KERNEL_FILES,
        "validated_kernel_provenance_blob": VALIDATED_KERNEL_PROVENANCE_BLOB,
        "summary_sha256": sha256_file(summary_path),
        "orientation_metrics_sha256": sha256_file(rows_path),
        "shard_manifest_sha256": sha256_file(shard_path),
        "worker_result_sha256": {
            str(worker_id): sha256_file(
                _worker_payload_path(scratch_root, worker_id)
            )
            for worker_id in (0, 1)
        },
        "projector_backend": transport.PROJECTOR_BACKEND,
        "projector_pinv_rtol": transport.PROJECTOR_PINV_RTOL,
        "pair_gap_energy_rtol": PAIR_GAP_ENERGY_RTOL,
        "pair_gap_energy_atol": PAIR_GAP_ENERGY_ATOL,
        "pair_gap_log_atol": PAIR_GAP_LOG_ATOL,
        "fft_chunk_width": FFT_CHUNK_WIDTH,
        "fast_float32_half_span_limit": FAST_FLOAT32_HALF_SPAN_LIMIT,
        "worker_row_ranges": {
            str(worker_id): list(worker_row_range(worker_id))
            for worker_id in (0, 1)
        },
        "dev_order_sha256": transport.DEV_ORDER_SHA256,
        "dev_encoding_sha256": transport.DEV_ENCODING_SHA256,
        "parent_checkpoint_sha256": transport.PARENT_CHECKPOINT_SHA256,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    print("GEN5_NATIVE_RECURRENT_PAIR_GAP_MERGE_PASS")
    print("ORIENTATIONS=36")
    print("VALIDATED_KERNEL_REPLAY=PASS")
    print("PAIR_GAP_RECONSTRUCTION=PASS")
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={rows_path}")
    print(f"SHARD_MANIFEST={shard_path}")
    print(f"PROVENANCE={provenance_path}")


def _spawn_worker(
    args: argparse.Namespace,
    worker_id: int,
) -> subprocess.Popen:
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-worker",
        "--expected-head",
        str(args.expected_head),
        "--implementation-freeze-commit",
        str(args.implementation_freeze_commit),
        "--model-snapshot",
        str(args.model_snapshot),
        "--tokenizer-snapshot",
        str(args.tokenizer_snapshot),
        "--checkpoint",
        str(args.checkpoint),
        "--scratch-root",
        str(args.scratch_root),
        "--worker-id",
        str(worker_id),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    return subprocess.Popen(command, cwd=ROOT, env=env)


def run_full(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    transport.legacy._validate_two_t4s()
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    scratch_root = Path(args.scratch_root)
    output_root = Path(args.output_root)
    require(not scratch_root.exists(), f"SCRATCH_COLLISION:{scratch_root}")
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    scratch_root.mkdir(parents=True, exist_ok=False)

    workers = [_spawn_worker(args, worker_id) for worker_id in (0, 1)]
    return_codes = [process.wait() for process in workers]
    require(return_codes == [0, 0], f"WORKER_FAILURE:{return_codes}")
    run_merge_only(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-contract-check", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run-worker", action="store_true")
    modes.add_argument("--merge-only", action="store_true")
    modes.add_argument("--run", action="store_true")

    parser.add_argument("--expected-head")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_contract_check:
        require(args.worker_id is None, "STATIC_WORKER_ID_FORBIDDEN")
        require(args.scratch_root is None, "STATIC_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_FORBIDDEN")
        return

    require(args.expected_head is not None, "EXPECTED_HEAD_REQUIRED")
    require(
        args.implementation_freeze_commit is not None,
        "IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(args.model_snapshot is not None, "MODEL_SNAPSHOT_REQUIRED")
    require(args.tokenizer_snapshot is not None, "TOKENIZER_SNAPSHOT_REQUIRED")
    require(args.checkpoint is not None, "PARENT_CHECKPOINT_REQUIRED")

    if args.preflight:
        require(args.worker_id is None, "PREFLIGHT_WORKER_FORBIDDEN")
        require(args.scratch_root is None, "PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_FORBIDDEN")
    elif args.run_worker:
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.output_root is None, "WORKER_OUTPUT_FORBIDDEN")
    elif args.merge_only:
        require(args.worker_id is None, "MERGE_WORKER_FORBIDDEN")
        require(args.scratch_root is not None, "MERGE_SCRATCH_REQUIRED")
        require(args.output_root is not None, "MERGE_OUTPUT_REQUIRED")
    elif args.run:
        require(args.worker_id is None, "RUN_WORKER_ID_FORBIDDEN")
        require(args.scratch_root is not None, "RUN_SCRATCH_REQUIRED")
        require(args.output_root is not None, "RUN_OUTPUT_REQUIRED")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    if args.static_contract_check:
        run_static_contract_check(args)
    elif args.preflight:
        run_preflight(args)
    elif args.run_worker:
        run_worker(args)
    elif args.merge_only:
        run_merge_only(args)
    elif args.run:
        run_full(args)
    else:
        raise PairGapLocalizationError("UNREACHABLE_MODE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
