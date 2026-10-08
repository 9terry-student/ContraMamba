#!/usr/bin/env python3
"""Exact recurrence-kernel decomposition for validated Gen5 native transport.

This audit is bounded to the raw-write -> recurrent-state transition.  It
reuses the frozen source-local two-margin projector from the validated
transport implementation and decomposes recurrent selective retention into:

    log selective retention
      = log self-propagation-gain ratio
      + log cross-token-interference ratio.

No downstream readout/gate/out-projection/head transport is executed by this
audit beyond the source-local task-gradient computation required to construct
the already-frozen projector.
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

from scripts import audit_native_downstream_functional_reconvergence_raw_write_transport as transport  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_COMMIT = "b31a829e40a661b641afdaba05ef722a5d059658"
DESIGN_PATH = "reports/native_recurrent_transport_kernel_decomposition_design_candidate.md"
DESIGN_BLOB = "99cd4b5d5c61e9df6e80f7d739108b592b3528ca"

VALIDATED_EVIDENCE_COMMIT = "6d36e49931389c21e9299b84dfd880151f71b8d3"
TRANSPORT_SCRIPT_PATH = "scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py"
TRANSPORT_SCRIPT_BLOB = "984319fecb5ed2c103b92da0ac868ac7746770d8"

VALIDATED_RUN = "gen5-native-reconvergence-raw-write-transport-3f73ec3-r1"
VALIDATED_ROOT = (
    ROOT
    / "reports/native_downstream_functional_reconvergence_raw_write_transport_runs"
    / VALIDATED_RUN
)
VALIDATED_FILES = {
    "transport_summary.json": "6c5d4438c8cb0dfbabaf00e7893d8c41897da90d354b5826faec5ba82f3cc464",
    "pair_stage_transport_metrics.jsonl": "fb7abbd92fb82033fc310d5f9fae98e42345e51c5159716e432e5a50d46045de",
    "shard_manifest.json": "cebd15304cf2c47df5bbec5ba58fc6c9426c4a3ca4d8626bbe269d6096ee59c4",
    "run_provenance.json": "40a40d22cd5b9b04b8b965af488fb9bf340084fd14a2278eb9a5162fd4af16a4",
}

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_native_recurrent_transport_kernel_decomposition.py",
        "tests/test_native_recurrent_transport_kernel_decomposition.py",
    }
)

OUTPUT_SCHEMA = "GEN5_NATIVE_RECURRENT_KERNEL_DECOMPOSITION_V1"
WORKER_SCHEMA = "GEN5_NATIVE_RECURRENT_KERNEL_DECOMPOSITION_WORKER_V1"
PROVENANCE_SCHEMA = "GEN5_NATIVE_RECURRENT_KERNEL_DECOMPOSITION_PROVENANCE_V1"
SHARD_SCHEMA = "GEN5_NATIVE_RECURRENT_KERNEL_DECOMPOSITION_SHARD_V1"

ENERGY_EPSILON = 1.0e-30
REPLAY_ENERGY_RTOL = 2.0e-5
REPLAY_ENERGY_ATOL = 2.0e-5
REPLAY_LOG_ATOL = 5.0e-5
LAG0_REPLAY_RTOL = 5.0e-4
RAW_COMPONENT_MATCH_RTOL = 2.0e-6
LOG_IDENTITY_ATOL = 5.0e-10
FFT_CHUNK_WIDTH = 256

FACTOR_SEEDS = transport.FACTOR_SEEDS
FULL_FACTORIAL_CELLS = transport.FULL_FACTORIAL_CELLS
DEV_ROWS = transport.DEV_ROWS
VALID_TOKEN_COUNT = transport.VALID_TOKEN_COUNT
BATCH_ROWS = transport.BATCH_ROWS
TARGETS_PER_TRANSPORT = transport.TARGETS_PER_TRANSPORT
WORKER_ROW_RANGES = transport.WORKER_ROW_RANGES


class KernelDecompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise KernelDecompositionError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise KernelDecompositionError("GIT_FAILURE:" + " ".join(args)) from exc


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
    return transport.cell_name(cell)


def all_orientations() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return transport.all_orientations()


def worker_row_range(worker_id: int) -> tuple[int, int]:
    return transport.worker_row_range(worker_id)


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
        git("rev-parse", f"HEAD:{TRANSPORT_SCRIPT_PATH}") == TRANSPORT_SCRIPT_BLOB,
        "TRANSPORT_IMPLEMENTATION_BLOB_DRIFT",
    )

    for name, digest in VALIDATED_FILES.items():
        path = VALIDATED_ROOT / name
        require(path.is_file(), f"VALIDATED_ARTIFACT_MISSING:{name}")
        require(sha256_file(path) == digest, f"VALIDATED_ARTIFACT_SHA:{name}")

    dirty = transport.status_paths()
    if allow_implementation_worktree:
        require(
            dirty <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(dirty)}",
        )
    else:
        require(not dirty, f"WORKTREE_NOT_CLEAN:{sorted(dirty)}")
    return head


def _load_validated_reference() -> dict[tuple[str, str, str], dict[str, float]]:
    path = VALIDATED_ROOT / "pair_stage_transport_metrics.jsonl"
    require(sha256_file(path) == VALIDATED_FILES[path.name], "VALIDATED_METRICS_SHA")
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stage = str(row["stage"])
        if stage not in {"raw_write", "recurrent_state"}:
            continue
        key = (str(row["group"]), str(row["source"]), str(row["target"]))
        dst = grouped.setdefault(key, {})
        if stage == "raw_write":
            dst["R_visible"] = float(row["D_visible"])
            dst["R_complement"] = float(row["D_complement"])
        else:
            dst["E_visible"] = float(row["D_visible"])
            dst["E_complement"] = float(row["D_complement"])
            dst["LOG_SELECTIVE_RETENTION"] = float(row["LOG_SELECTIVE_RETENTION"])
    require(len(grouped) == 36, f"VALIDATED_REFERENCE_COUNT:{len(grouped)}")
    required = {
        "R_visible",
        "R_complement",
        "E_visible",
        "E_complement",
        "LOG_SELECTIVE_RETENTION",
    }
    for key, row in grouped.items():
        require(set(row) == required, f"VALIDATED_REFERENCE_FIELDS:{key}:{sorted(row)}")
    return grouped


def validate_static_contract() -> None:
    transport.validate_static_contract()
    require(FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(len(all_orientations()) == 36, "ORIENTATION_COUNT")
    require(
        sum(1 for g, _, _ in all_orientations() if g == "PRIMARY_A") == 18,
        "PRIMARY_ORIENTATION_COUNT",
    )
    require(
        sum(1 for g, _, _ in all_orientations() if g == "CONTROL_R") == 18,
        "CONTROL_ORIENTATION_COUNT",
    )
    require(worker_row_range(0) == (0, 416), "WORKER0_RANGE")
    require(worker_row_range(1) == (416, 840), "WORKER1_RANGE")
    require(TARGETS_PER_TRANSPORT == 2, "TARGETS_PER_TRANSPORT")
    require(FFT_CHUNK_WIDTH > 0, "FFT_CHUNK_WIDTH")
    _load_validated_reference()


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
        "J1_visible_numer": 0.0,
        "J1_visible_denom": 0.0,
        "J1_complement_numer": 0.0,
        "J1_complement_denom": 0.0,
        "lag_visible": [],
        "lag_complement": [],
        "fft_half_log_span_max": 0.0,
    }


def _discrete_log_a(
    context: Mapping[str, torch.Tensor],
    *,
    shape: Any,
) -> torch.Tensor:
    a_cont = context["a_continuous"]
    dt = context["discrete_time_step"]
    require(a_cont.ndim == 2, "A_CONTINUOUS_RANK")
    require(dt.ndim == 3, "DISCRETE_TIME_STEP_RANK")
    require(
        tuple(a_cont.shape) == (shape.intermediate_size, shape.state_size),
        "A_CONTINUOUS_SHAPE",
    )
    require(dt.shape[1] == shape.intermediate_size, "DT_INTERMEDIATE_SIZE")
    return (
        a_cont[None, :, None, :].float()
        * dt[:, :, :, None].float()
    ).permute(0, 2, 1, 3).contiguous()


def _lag_strict_energy_dyadic_fft(
    *,
    w_chunk: torch.Tensor,
    p_chunk: torch.Tensor,
    target_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Strict-causal lag energy via overflow-safe batched GPU FFT rectangles.

    The strict triangle tau < t is partitioned dyadically. Each ordered pair
    belongs to exactly one cross-half rectangle, so pair accounting is not
    duplicated. Because every source precedes every target in a rectangle,
    the native stable recurrence gives p_target <= p_source. A per-rectangle
    shift therefore keeps both exponential factors <= 1 without changing
    their product.
    """
    require(w_chunk.ndim == 4, "DYADIC_W_RANK")
    k_count, batch, observed_seq_len, width = w_chunk.shape
    require(observed_seq_len == seq_len, "DYADIC_SEQ_LEN")
    require(
        tuple(p_chunk.shape) == (batch, seq_len, width),
        "DYADIC_P_SHAPE",
    )
    require(
        tuple(target_mask.shape) == (batch, seq_len),
        "DYADIC_MASK_SHAPE",
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
        mask_pad = torch.cat(
            (
                target_mask,
                torch.zeros(
                    (batch, pad_count),
                    device=target_mask.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
    else:
        w_pad = w_chunk
        p_pad = p_chunk
        mask_pad = target_mask

    strict_lag = torch.zeros(
        (k_count, seq_len),
        device=w_chunk.device,
        dtype=torch.float64,
    )

    block_len = 2
    while block_len <= padded_len:
        half = block_len // 2
        block_count = padded_len // block_len

        p_blocks = p_pad.reshape(
            batch,
            block_count,
            block_len,
            width,
        )
        w_blocks = w_pad.reshape(
            k_count,
            batch,
            block_count,
            block_len,
            width,
        )
        mask_blocks = mask_pad.reshape(
            batch,
            block_count,
            block_len,
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

        source = (
            w_blocks[:, :, :, :half, :].square()
            * x_scale.unsqueeze(0)
        )
        target = torch.where(
            mask_blocks[:, :, half:, None],
            y_scale,
            torch.zeros_like(y_scale),
        )

        n_fft = 1 << ((2 * half - 1).bit_length())
        source_fft = torch.fft.rfft(
            source,
            n=n_fft,
            dim=3,
        )
        target_fft = torch.fft.rfft(
            target,
            n=n_fft,
            dim=2,
        ).unsqueeze(0)
        correlation = torch.fft.irfft(
            torch.conj(source_fft) * target_fft,
            n=n_fft,
            dim=3,
        )
        correlation_sum = torch.sum(
            correlation,
            dim=(1, 2, 4),
            dtype=torch.float64,
        )

        negative_stop = min(half, seq_len)
        if negative_stop > 1:
            strict_lag[:, 1:negative_stop] += correlation_sum[
                :,
                n_fft + 1 - half : n_fft + negative_stop - half,
            ]

        positive_stop = min(block_len, seq_len)
        if half < positive_stop:
            strict_lag[:, half:positive_stop] += correlation_sum[
                :,
                : positive_stop - half,
            ]

        block_len *= 2

    return strict_lag


def _lag_self_energy_fft(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    attention_mask: torch.Tensor,
    shape: Any,
) -> tuple[torch.Tensor, float]:
    require(component.ndim == 4, "LAG_COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(width == shape.intermediate_size * shape.state_size, "LAG_WIDTH")
    require(
        tuple(log_a.shape)
        == (batch, seq_len, shape.intermediate_size, shape.state_size),
        "LAG_LOG_A_SHAPE",
    )
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "LAG_MASK_SHAPE",
    )
    require(
        bool(torch.all(torch.isfinite(log_a)).item()),
        "LAG_LOG_A_NONFINITE",
    )
    require(
        bool(torch.all(log_a <= 0.0).item()),
        "LAG_LOG_A_POSITIVE",
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
        "LAG_EMPTY_VALID_SEQUENCE",
    )
    n_fft = 1 << ((2 * seq_len - 1).bit_length())

    lag = torch.zeros(
        (k_count, seq_len),
        device=component.device,
        dtype=torch.float64,
    )
    max_half_span = 0.0

    for start in range(0, width, FFT_CHUNK_WIDTH):
        stop = min(start + FFT_CHUNK_WIDTH, width)
        p = 2.0 * torch.cumsum(
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
            "LAG_ACTIVE_PREFIX_NONFINITE",
        )
        half_span = 0.5 * (p_max - p_min)
        chunk_max = float(torch.amax(half_span).item())
        max_half_span = max(max_half_span, chunk_max)

        calc_dtype = torch.float64 if chunk_max > 60.0 else torch.float32
        centered_exp_limit = math.log(torch.finfo(calc_dtype).max)
        w_native = w[:, :, :, start:stop]

        if chunk_max <= centered_exp_limit:
            w_chunk = w_native.to(dtype=calc_dtype)
            p_calc = p.to(dtype=calc_dtype)
            center = 0.5 * (
                p_max.to(dtype=calc_dtype)
                + p_min.to(dtype=calc_dtype)
            )
            p_safe = torch.where(
                active,
                p_calc,
                center[:, None, :],
            )

            x_scale = torch.exp(-p_safe + center[:, None, :])
            y_scale = torch.exp(p_safe - center[:, None, :])
            require(
                bool(torch.all(torch.isfinite(x_scale)).item()),
                f"LAG_X_SCALE_NONFINITE:{chunk_max}",
            )
            require(
                bool(torch.all(torch.isfinite(y_scale)).item()),
                f"LAG_Y_SCALE_NONFINITE:{chunk_max}",
            )
            y = torch.where(
                target_mask[:, :, None],
                y_scale,
                torch.zeros_like(y_scale),
            )

            w_active = torch.where(
                source_active[None, :, :, None],
                w_chunk,
                torch.zeros_like(w_chunk),
            )
            x = w_active.square() * x_scale.unsqueeze(0)

            x_fft = torch.fft.rfft(x, n=n_fft, dim=2)
            y_fft = torch.fft.rfft(y, n=n_fft, dim=1).unsqueeze(0)
            corr = torch.fft.irfft(
                torch.conj(x_fft) * y_fft,
                n=n_fft,
                dim=2,
            )[:, :, :seq_len, :]
            lag += torch.sum(
                corr,
                dim=(1, 3),
                dtype=torch.float64,
            )
            continue

        lag[:, 0] += torch.sum(
            w_native.to(dtype=torch.float64).square()
            * target_mask[None, :, :, None],
            dim=(1, 2, 3),
            dtype=torch.float64,
        )
        lag += _lag_strict_energy_dyadic_fft(
            w_chunk=w_native.to(dtype=torch.float32),
            p_chunk=p,
            target_mask=target_mask,
            seq_len=seq_len,
        )

    lag.clamp_(min=0.0)
    return lag, max_half_span


def _component_recurrence_stats(
    *,
    component: torch.Tensor,
    context: Mapping[str, torch.Tensor],
    attention_mask: torch.Tensor,
    shape: Any,
) -> list[dict[str, Any]]:
    require(component.ndim == 4, "COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(width == shape.state_width, "COMPONENT_STATE_WIDTH")
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "COMPONENT_MASK_SHAPE",
    )
    require(
        shape.state_width == shape.intermediate_size * shape.state_size,
        "STATE_WIDTH_FACTORIZATION",
    )

    w = component.reshape(
        k_count,
        batch,
        seq_len,
        shape.intermediate_size,
        shape.state_size,
    )
    log_a = _discrete_log_a(context, shape=shape).to(device=component.device)

    state = torch.zeros(
        (k_count, batch, shape.intermediate_size, shape.state_size),
        device=component.device,
        dtype=component.dtype,
    )
    self_state = torch.zeros_like(state)
    raw = torch.zeros(k_count, device=component.device, dtype=torch.float64)
    recurrent = torch.zeros_like(raw)
    self_energy = torch.zeros_like(raw)
    j1_denom = torch.zeros_like(raw)

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

            if token_index + 1 < seq_len:
                next_valid = attention_mask[:, token_index + 1].to(
                    device=component.device,
                    dtype=component.dtype,
                ).reshape(1, batch, 1, 1)
                j1_denom += torch.sum(
                    write_t.square() * next_valid,
                    dim=(1, 2, 3),
                    dtype=torch.float64,
                )

    lag, max_half_span = _lag_self_energy_fft(
        component=component,
        log_a=log_a,
        attention_mask=attention_mask,
        shape=shape,
    )
    if seq_len > 0:
        for index in range(k_count):
            rel = _relative_error(
                float(lag[index, 0].item()),
                float(raw[index].item()),
            )
            require(rel <= LAG0_REPLAY_RTOL, f"LAG0_RAW_REPLAY:{index}:{rel}")

    rows: list[dict[str, Any]] = []
    for index in range(k_count):
        lag_values = [
            float(value)
            for value in lag[index].detach().to(device="cpu").tolist()
        ]
        rows.append(
            {
                "R": float(raw[index].item()),
                "S": float(self_energy[index].item()),
                "E": float(recurrent[index].item()),
                "C": float((recurrent[index] - self_energy[index]).item()),
                "J1_numer": lag_values[1] if len(lag_values) > 1 else 0.0,
                "J1_denom": float(j1_denom[index].item()),
                "lag": lag_values,
                "fft_half_log_span_max": max_half_span,
            }
        )
    return rows


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
    accumulator[f"J1_{prefix}_numer"] += float(stats["J1_numer"])
    accumulator[f"J1_{prefix}_denom"] += float(stats["J1_denom"])
    accumulator["fft_half_log_span_max"] = max(
        float(accumulator["fft_half_log_span_max"]),
        float(stats["fft_half_log_span_max"]),
    )

    key = f"lag_{prefix}"
    observed = [float(x) for x in stats["lag"]]
    if not accumulator[key]:
        accumulator[key] = [0.0 for _ in observed]
    require(len(accumulator[key]) == len(observed), f"LAG_LENGTH:{prefix}")
    for index, value in enumerate(observed):
        accumulator[key][index] += value


def _lag_summary(values: Sequence[float]) -> dict[str, Any]:
    require(len(values) > 0, "EMPTY_LAG_VECTOR")
    total = sum(float(x) for x in values)
    require(total > 0.0, "NONPOSITIVE_LAG_TOTAL")
    fractions = [float(x) / total for x in values]

    def bucket(start: int, stop: int | None) -> float:
        return sum(fractions[start:stop])

    cumulative = 0.0
    lag50 = len(fractions) - 1
    lag90 = len(fractions) - 1
    found50 = False
    for index, value in enumerate(fractions):
        cumulative += value
        if not found50 and cumulative >= 0.5:
            lag50 = index
            found50 = True
        if cumulative >= 0.9:
            lag90 = index
            break

    mean_lag = sum(index * value for index, value in enumerate(fractions))
    return {
        "lag0_fraction": bucket(0, 1),
        "lag1_fraction": bucket(1, 2),
        "lag2_4_fraction": bucket(2, 5),
        "lag5_8_fraction": bucket(5, 9),
        "lag9plus_fraction": bucket(9, None),
        "mean_lag": mean_lag,
        "lag50": lag50,
        "lag90": lag90,
        "total_self_energy": total,
    }


def _finalize_orientation(
    accumulator: Mapping[str, Any],
    reference: Mapping[str, float],
) -> dict[str, Any]:
    require(int(accumulator["example_count"]) == DEV_ROWS, "FINAL_EXAMPLE_COUNT")
    values = {key: float(accumulator[key]) for key in (
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    )}
    require(all(value > 0.0 for value in values.values()), "NONPOSITIVE_ENERGY")

    g_visible = values["S_visible"] / values["R_visible"]
    g_complement = values["S_complement"] / values["R_complement"]
    q_visible = values["E_visible"] / values["S_visible"]
    q_complement = values["E_complement"] / values["S_complement"]

    l_kernel = math.log(g_complement / g_visible)
    l_interference = math.log(q_complement / q_visible)
    log_selective = math.log(
        (values["E_complement"] / values["R_complement"])
        / (values["E_visible"] / values["R_visible"])
    )
    identity_error = abs(log_selective - (l_kernel + l_interference))
    require(
        identity_error <= LOG_IDENTITY_ATOL,
        f"LOG_DECOMPOSITION_IDENTITY:{identity_error}",
    )

    replay = {}
    for key in ("R_visible", "R_complement", "E_visible", "E_complement"):
        observed = values[key]
        expected = float(reference[key])
        rel = _relative_error(observed, expected)
        passed = (
            abs(observed - expected) <= REPLAY_ENERGY_ATOL
            or rel <= REPLAY_ENERGY_RTOL
        )
        replay[key] = {
            "observed": observed,
            "expected": expected,
            "abs_error": abs(observed - expected),
            "rel_error": rel,
            "pass": passed,
        }
        require(passed, f"VALIDATED_ENERGY_REPLAY:{key}:{rel}")

    validated_log = float(reference["LOG_SELECTIVE_RETENTION"])
    log_error = abs(log_selective - validated_log)
    require(
        log_error <= REPLAY_LOG_ATOL,
        f"VALIDATED_LOG_REPLAY:{log_error}",
    )

    j1_visible = float(accumulator["J1_visible_numer"]) / max(
        float(accumulator["J1_visible_denom"]),
        ENERGY_EPSILON,
    )
    j1_complement = float(accumulator["J1_complement_numer"]) / max(
        float(accumulator["J1_complement_denom"]),
        ENERGY_EPSILON,
    )

    return {
        "group": str(accumulator["group"]),
        "source": str(accumulator["source"]),
        "target": str(accumulator["target"]),
        "example_count": int(accumulator["example_count"]),
        **values,
        "C_visible": values["E_visible"] - values["S_visible"],
        "C_complement": values["E_complement"] - values["S_complement"],
        "G_visible": g_visible,
        "G_complement": g_complement,
        "Q_visible": q_visible,
        "Q_complement": q_complement,
        "L_kernel": l_kernel,
        "L_interference": l_interference,
        "LOG_SELECTIVE_RETENTION": log_selective,
        "SELECTIVE_RETENTION": math.exp(log_selective),
        "log_identity_abs_error": identity_error,
        "J1_visible": j1_visible,
        "J1_complement": j1_complement,
        "J1_ratio_complement_over_visible": (
            j1_complement / max(j1_visible, ENERGY_EPSILON)
        ),
        "lag_visible": _lag_summary(accumulator["lag_visible"]),
        "lag_complement": _lag_summary(accumulator["lag_complement"]),
        "raw_reconstruction_max_abs": float(
            accumulator["raw_reconstruction_max_abs"]
        ),
        "fft_half_log_span_max": float(accumulator["fft_half_log_span_max"]),
        "validated_transport_replay": {
            **replay,
            "LOG_SELECTIVE_RETENTION": {
                "observed": log_selective,
                "expected": validated_log,
                "abs_error": log_error,
                "pass": True,
            },
        },
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
    workers = {}
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
            "recurrence_group_decompositions": groups,
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
        "total_recurrence_group_decompositions": sum(
            row["recurrence_group_decompositions"] for row in workers.values()
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
    print("GEN5_NATIVE_RECURRENT_KERNEL_STATIC_CONTRACT_PASS")
    print(f"HEAD={head}")
    print("PRIMARY_ORIENTATIONS=18")
    print("CONTROL_ORIENTATIONS=18")
    print("TOTAL_ORIENTATIONS=36")
    print("WORKER0_ROWS=0:416")
    print("WORKER1_ROWS=416:840")
    print("VALIDATED_TRANSPORT_REFERENCE=AUTHENTICATED")
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
    print("GEN5_NATIVE_RECURRENT_KERNEL_CUDA_PREFLIGHT_PASS")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("EXECUTION_ENGINE=RECURRENCE_KERNEL_EXACT_DECOMPOSITION_V1")
    print("VALIDATED_TRANSPORT_REFERENCE=AUTHENTICATED")
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
                stats = _component_recurrence_stats(
                    component=combined,
                    context=context,
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

        del raw_by_cell
        print(
            "GEN5_RECURRENT_KERNEL_PROGRESS "
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
        "execution_engine": "RECURRENCE_KERNEL_EXACT_DECOMPOSITION_V1",
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
        "GEN5_NATIVE_RECURRENT_KERNEL_WORKER_PASS "
        f"worker={worker_id} rows={row_start}:{row_stop} "
        f"orientations=36 sha256={digest}"
    )


def _read_worker(scratch_root: Path, worker_id: int) -> dict[str, Any]:
    path = _worker_payload_path(scratch_root, worker_id)
    sidecar = path.with_suffix(".sha256")
    require(path.is_file(), f"WORKER_RESULT_MISSING:{worker_id}")
    require(sidecar.is_file(), f"WORKER_SHA_MISSING:{worker_id}")
    require(
        sidecar.read_text(encoding="utf-8").strip() == sha256_file(path),
        f"WORKER_SHA_MISMATCH:{worker_id}",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version") == WORKER_SCHEMA, "WORKER_SCHEMA")
    require(int(payload.get("worker_id", -1)) == worker_id, "WORKER_IDENTITY")
    require(
        payload.get("execution_head")
        == payload.get("implementation_freeze_commit"),
        "WORKER_HEAD_BINDING",
    )
    start, stop = worker_row_range(worker_id)
    require(
        (int(payload["row_start"]), int(payload["row_stop"]))
        == (start, stop),
        "WORKER_ROW_RANGE",
    )
    require(
        int(payload["valid_token_count"]) > 0,
        "WORKER_VALID_TOKEN_COUNT",
    )
    require(payload.get("recurrence_only") is True, "WORKER_RECURRENCE_ONLY")
    require(
        int(payload.get("downstream_stage_transport_calls", -1)) == 0,
        "WORKER_DOWNSTREAM_TRANSPORT",
    )
    require(
        int(payload.get("final_head_transport_calls", -1)) == 0,
        "WORKER_FINAL_HEAD_TRANSPORT",
    )
    require(payload.get("training_executed") is False, "WORKER_TRAINING")
    require(
        payload.get("parameter_gradients_accumulated") is False,
        "WORKER_PARAM_GRAD",
    )
    require(
        payload.get("confirmatory_9601_9900_loaded") is False,
        "WORKER_CONFIRMATORY",
    )
    require(payload.get("vitaminc_loaded") is False, "WORKER_VITAMINC")
    rows = payload.get("orientation_accumulators")
    require(
        isinstance(rows, list) and len(rows) == 36,
        "WORKER_ACCUMULATOR_COUNT",
    )
    return payload


def _merge_worker_accumulators(
    workers: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = {
        (group, cell_name(source), cell_name(target)): (group, source, target)
        for group, source, target in all_orientations()
    }
    merged = {
        key: _empty_orientation(*triple)
        for key, triple in keys.items()
    }
    worker_ids = set()

    for worker in workers:
        worker_id = int(worker["worker_id"])
        require(worker_id not in worker_ids, "MERGED_WORKER_DUPLICATE")
        worker_ids.add(worker_id)
        for part in worker["orientation_accumulators"]:
            key = (
                str(part["group"]),
                str(part["source"]),
                str(part["target"]),
            )
            require(key in merged, f"MERGED_ORIENTATION_COVERAGE:{key}")
            dst = merged[key]
            dst["example_count"] += int(part["example_count"])
            dst["raw_reconstruction_max_abs"] = max(
                float(dst["raw_reconstruction_max_abs"]),
                float(part["raw_reconstruction_max_abs"]),
            )
            dst["fft_half_log_span_max"] = max(
                float(dst["fft_half_log_span_max"]),
                float(part["fft_half_log_span_max"]),
            )
            for name in (
                "R_visible",
                "R_complement",
                "S_visible",
                "S_complement",
                "E_visible",
                "E_complement",
                "J1_visible_numer",
                "J1_visible_denom",
                "J1_complement_numer",
                "J1_complement_denom",
            ):
                dst[name] += float(part[name])
            for name in ("lag_visible", "lag_complement"):
                values = [float(x) for x in part[name]]
                if not dst[name]:
                    dst[name] = [0.0 for _ in values]
                require(
                    len(dst[name]) == len(values),
                    f"MERGED_LAG_LENGTH:{name}",
                )
                for index, value in enumerate(values):
                    dst[name][index] += value

    require(worker_ids == {0, 1}, "MERGED_WORKER_COVERAGE")
    require(len(merged) == 36, "MERGED_ORIENTATION_COUNT")
    require(
        all(int(row["example_count"]) == DEV_ROWS for row in merged.values()),
        "MERGED_EXAMPLE_COUNT",
    )

    reference = _load_validated_reference()
    rows = []
    for group, source, target in all_orientations():
        key = (group, cell_name(source), cell_name(target))
        rows.append(_finalize_orientation(merged[key], reference[key]))
    return rows


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(x) for x in values)
    require(ordered, "EMPTY_MEDIAN")
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(float(x) for x in values)
    require(len(ordered) >= 4, "IQR_REQUIRES_FOUR")
    mid = len(ordered) // 2
    return _median(ordered[:mid]), _median(ordered[-mid:])


def _symmetric_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        source = str(row["source"])
        target = str(row["target"])
        key = (str(row["group"]), min(source, target), max(source, target))
        grouped.setdefault(key, []).append(row)

    output = []
    for key, pair_rows in sorted(grouped.items()):
        require(
            len(pair_rows) == 2,
            f"PAIR_ORIENTATION_COVERAGE:{key}:{len(pair_rows)}",
        )
        metrics = {}
        for name in (
            "L_kernel",
            "L_interference",
            "LOG_SELECTIVE_RETENTION",
        ):
            metrics[name] = sum(float(row[name]) for row in pair_rows) / 2.0
        j1_logs = [
            math.log(
                max(
                    float(row["J1_ratio_complement_over_visible"]),
                    ENERGY_EPSILON,
                )
            )
            for row in pair_rows
        ]
        metrics["LOG_J1_RATIO"] = sum(j1_logs) / 2.0
        output.append(
            {
                "group": key[0],
                "left": key[1],
                "right": key[2],
                **metrics,
                "SELECTIVE_RETENTION": math.exp(
                    metrics["LOG_SELECTIVE_RETENTION"]
                ),
                "J1_RATIO": math.exp(metrics["LOG_J1_RATIO"]),
            }
        )
    require(len(output) == 18, "SYMMETRIC_PAIR_COUNT")
    return output


def _group_summary(
    pairs: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    values = [float(row[metric]) for row in pairs]
    require(len(values) == 9, f"GROUP_PAIR_COUNT:{metric}:{len(values)}")
    q1, q3 = _quartiles(values)
    return {
        "count": 9,
        "values": values,
        "median": _median(values),
        "minimum": min(values),
        "maximum": max(values),
        "iqr": [q1, q3],
    }


def _source_matched_controls(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
        metrics = {}
        for name in (
            "L_kernel",
            "L_interference",
            "LOG_SELECTIVE_RETENTION",
        ):
            p = sum(float(row[name]) for row in primary) / 2.0
            c = sum(float(row[name]) for row in control) / 2.0
            metrics[name] = {
                "primary_log_mean": p,
                "control_log_mean": c,
                "primary_minus_control": p - c,
            }
        output.append({"source": source, "metrics": metrics})
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
            all(
                bool(check["pass"])
                for check in row["validated_transport_replay"].values()
            )
            for row in rows
        ),
        "VALIDATED_TRANSPORT_REPLAY_FAIL",
    )
    require(
        max(float(row["raw_reconstruction_max_abs"]) for row in rows)
        <= transport.RAW_RECON_ATOL,
        "MERGED_RAW_RECONSTRUCTION_FAIL",
    )

    pairs = _symmetric_pairs(rows)
    grouped: dict[str, Any] = {}
    for group in ("PRIMARY_A", "CONTROL_R"):
        subset = [row for row in pairs if row["group"] == group]
        require(len(subset) == 9, f"PAIR_GROUP_COUNT:{group}")
        grouped[group] = {
            metric: _group_summary(subset, metric)
            for metric in (
                "L_kernel",
                "L_interference",
                "LOG_SELECTIVE_RETENTION",
                "LOG_J1_RATIO",
            )
        }

    matched = _source_matched_controls(rows)
    max_replay_rel = max(
        float(check["rel_error"])
        for row in rows
        for name, check in row["validated_transport_replay"].items()
        if name != "LOG_SELECTIVE_RETENTION"
    )
    max_log_replay = max(
        float(row["validated_transport_replay"]["LOG_SELECTIVE_RETENTION"]["abs_error"])
        for row in rows
    )

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_VALIDATED_RECURRENT_KERNEL_DECOMPOSITION",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_transport_run": VALIDATED_RUN,
        "population": "FROZEN_PHASE3A_P0_DEV",
        "dev_rows": DEV_ROWS,
        "valid_token_count": VALID_TOKEN_COUNT,
        "primary_orientations": 18,
        "control_orientations": 18,
        "decomposition_identity": (
            "LOG_SELECTIVE_RETENTION=L_kernel+L_interference"
        ),
        "grouped_symmetric_pair_log_terms": grouped,
        "symmetric_unordered_pairs": pairs,
        "source_cell_matched_primary_minus_control": matched,
        "validated_transport_replay": {
            "pass": True,
            "energy_rtol": REPLAY_ENERGY_RTOL,
            "energy_atol": REPLAY_ENERGY_ATOL,
            "log_atol": REPLAY_LOG_ATOL,
            "max_energy_rel_error": max_replay_rel,
            "max_log_abs_error": max_log_replay,
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
    summary_path = output_root / "recurrent_kernel_decomposition_summary.json"
    rows_path = output_root / "orientation_kernel_decomposition_metrics.jsonl"
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
        "validated_transport_script_blob": TRANSPORT_SCRIPT_BLOB,
        "validated_transport_artifact_sha256": VALIDATED_FILES,
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
        "energy_epsilon": ENERGY_EPSILON,
        "replay_energy_rtol": REPLAY_ENERGY_RTOL,
        "replay_energy_atol": REPLAY_ENERGY_ATOL,
        "replay_log_atol": REPLAY_LOG_ATOL,
        "fft_chunk_width": FFT_CHUNK_WIDTH,
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

    print("GEN5_NATIVE_RECURRENT_KERNEL_MERGE_PASS")
    print("ORIENTATIONS=36")
    print("UNORDERED_PAIRS=18")
    print("VALIDATED_TRANSPORT_REPLAY=PASS")
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
    else:
        run_full(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
