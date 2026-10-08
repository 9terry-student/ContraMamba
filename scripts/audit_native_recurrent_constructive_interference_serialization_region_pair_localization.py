#!/usr/bin/env python3
"""Exact serialization-region localization of recurrent cross-token interference.

This audit is bounded to the frozen raw_write -> recurrent_state transition.
It refines the validated pair-gap decomposition by partitioning strict token
pairs into pre-existing CLAIM/EOS/EVIDENCE serialization regions.

No downstream readout/gate/out-projection/head transport is executed. No
training, optimizer step, checkpoint mutation, projector refit, confirmatory
population, VitaminC access, or post-hoc semantic token labeling is allowed.
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

from scripts import audit_native_recurrent_constructive_interference_pair_gap_localization as pairgap  # noqa: E402

kernel = pairgap.kernel
transport = pairgap.transport
p3a = transport.p3a

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_COMMIT = "699b76d66b1204c33b4e01f04302cad52b47aa3c"
DESIGN_PATH = (
    "reports/"
    "native_recurrent_constructive_interference_serialization_region_pair_"
    "localization_design_candidate.md"
)
DESIGN_BLOB = "7755a7b5ff3f62010a63f5237ce233f5fce154c4"

VALIDATED_EVIDENCE_COMMIT = "9f60dacc98240eb9a087974e0996e5763f4d2b9a"
VALIDATED_PAIR_GAP_SCRIPT_PATH = (
    "scripts/audit_native_recurrent_constructive_interference_pair_gap_localization.py"
)
VALIDATED_PAIR_GAP_SCRIPT_BLOB = "a37970570ddd659a81523cf2e59e324a854c576e"
PHASE3A_RUNTIME_PATH = "scripts/train_reason_router_gen5_phase3a_contention.py"
PHASE3A_RUNTIME_BLOB = "45e333128c4fa31f4f502c5fdcf324243c1084fd"

VALIDATED_PAIR_GAP_RUN = "gen5-recurrent-pair-gap-localization-57ea0b5-r2"
VALIDATED_PAIR_GAP_ROOT = (
    ROOT
    / "reports/native_recurrent_constructive_interference_pair_gap_localization_runs"
    / VALIDATED_PAIR_GAP_RUN
)
VALIDATED_PAIR_GAP_FILES = {
    "pair_gap_localization_summary.json":
        "8c30db83f4cfb17089778a43199d61b043b15cc67340c1639470cdd556d40cab",
    "orientation_pair_gap_metrics.jsonl":
        "ede01dc8ba696293005b36558c622ba621943c1b8942edfd02b63489ab8d0949",
    "shard_manifest.json":
        "8e69d080bb6f8d0eb97688e53f4843654f2a895d66caef9adb0dda22fde8832c",
    "run_provenance.json":
        "7e81ff74037dee483f04c7ca9234f9f2c73f14ec9443d61d2310dedfe16b5ad0",
}

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_native_recurrent_constructive_interference_serialization_region_pair_localization.py",
        "tests/test_native_recurrent_constructive_interference_serialization_region_pair_localization.py",
    }
)

OUTPUT_SCHEMA = "GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION_V1"
WORKER_SCHEMA = "GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION_WORKER_V1"
PROVENANCE_SCHEMA = "GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION_PROVENANCE_V1"
SHARD_SCHEMA = "GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION_SHARD_V1"

REGION_NAMES = ("CLAIM", "EOS", "EVIDENCE")
REGION_PAIRS = (
    ("CLAIM", "CLAIM"),
    ("CLAIM", "EOS"),
    ("CLAIM", "EVIDENCE"),
    ("EOS", "EVIDENCE"),
    ("EVIDENCE", "EVIDENCE"),
)
REGION_PAIR_NAMES = tuple(f"{left}->{right}" for left, right in REGION_PAIRS)

WINDOWS = {
    "gap1_8": (1, 8),
    "gap9_16": (9, 16),
    "gap17_32": (17, 32),
    "gap33_64": (33, 64),
    "gap65_127": (65, 127),
    "gap1_32": (1, 32),
    "all": (1, 127),
}

PAIR_GAP_ENERGY_RTOL = pairgap.PAIR_GAP_ENERGY_RTOL
PAIR_GAP_ENERGY_ATOL = pairgap.PAIR_GAP_ENERGY_ATOL
PAIR_GAP_LOG_ATOL = pairgap.PAIR_GAP_LOG_ATOL
RAW_COMPONENT_MATCH_RTOL = pairgap.RAW_COMPONENT_MATCH_RTOL
ENERGY_EPSILON = pairgap.ENERGY_EPSILON
FFT_CHUNK_WIDTH = pairgap.FFT_CHUNK_WIDTH
FAST_FLOAT32_HALF_SPAN_LIMIT = pairgap.FAST_FLOAT32_HALF_SPAN_LIMIT

FACTOR_SEEDS = pairgap.FACTOR_SEEDS
FULL_FACTORIAL_CELLS = pairgap.FULL_FACTORIAL_CELLS
DEV_ROWS = pairgap.DEV_ROWS
VALID_TOKEN_COUNT = pairgap.VALID_TOKEN_COUNT
BATCH_ROWS = pairgap.BATCH_ROWS
TARGETS_PER_TRANSPORT = pairgap.TARGETS_PER_TRANSPORT
WORKER_ROW_RANGES = pairgap.WORKER_ROW_RANGES


class SerializationRegionPairLocalizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SerializationRegionPairLocalizationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SerializationRegionPairLocalizationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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
    return pairgap.cell_name(cell)


def all_orientations() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return pairgap.all_orientations()


def worker_row_range(worker_id: int) -> tuple[int, int]:
    return pairgap.worker_row_range(worker_id)


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

    for ancestor, label in (
        (DESIGN_COMMIT, "DESIGN"),
        (VALIDATED_EVIDENCE_COMMIT, "VALIDATED_EVIDENCE"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    require(
        git("rev-parse", f"HEAD:{DESIGN_PATH}") == DESIGN_BLOB,
        "DESIGN_BLOB_DRIFT",
    )
    require(
        git("rev-parse", f"HEAD:{VALIDATED_PAIR_GAP_SCRIPT_PATH}")
        == VALIDATED_PAIR_GAP_SCRIPT_BLOB,
        "VALIDATED_PAIR_GAP_SCRIPT_DRIFT",
    )
    require(
        git("rev-parse", f"HEAD:{PHASE3A_RUNTIME_PATH}") == PHASE3A_RUNTIME_BLOB,
        "PHASE3A_RUNTIME_DRIFT",
    )

    for name, digest in VALIDATED_PAIR_GAP_FILES.items():
        path = VALIDATED_PAIR_GAP_ROOT / name
        require(path.is_file(), f"VALIDATED_PAIR_GAP_ARTIFACT_MISSING:{name}")
        require(
            sha256_file(path) == digest,
            f"VALIDATED_PAIR_GAP_ARTIFACT_SHA:{name}",
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


def _load_validated_pair_gap_reference(
) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = VALIDATED_PAIR_GAP_ROOT / "orientation_pair_gap_metrics.jsonl"
    require(
        sha256_file(path) == VALIDATED_PAIR_GAP_FILES[path.name],
        "VALIDATED_PAIR_GAP_METRICS_SHA",
    )
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["group"]), str(row["source"]), str(row["target"]))
        require(key not in rows, f"VALIDATED_PAIR_GAP_DUPLICATE:{key}")
        rows[key] = row
    require(
        len(rows) == 36,
        f"VALIDATED_PAIR_GAP_REFERENCE_COUNT:{len(rows)}",
    )
    return rows


def validate_static_contract() -> None:
    pairgap.validate_static_contract()
    require(FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(len(all_orientations()) == 36, "ORIENTATION_COUNT")
    require(worker_row_range(0) == (0, 416), "WORKER0_RANGE")
    require(worker_row_range(1) == (416, 840), "WORKER1_RANGE")
    require(DEV_ROWS == p3a.DEV_ROWS == 840, "DEV_ROWS")
    require(VALID_TOKEN_COUNT == transport.VALID_TOKEN_COUNT == 60094, "VALID_TOKEN_COUNT")
    require(p3a.MAX_LENGTH == 128, "MAX_LENGTH")
    require(p3a.CLAIM_BUDGET == 63, "CLAIM_BUDGET")
    require(p3a.EVIDENCE_BUDGET == 64, "EVIDENCE_BUDGET")
    require(p3a.EOS_TOKEN_ID == 0, "EOS_TOKEN_ID")
    require(p3a.PAD_TOKEN_ID == 0, "PAD_TOKEN_ID")
    require(
        transport.DEV_ORDER_SHA256
        == "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25",
        "DEV_ORDER_SHA",
    )
    require(
        transport.DEV_ENCODING_SHA256
        == "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51",
        "DEV_ENCODING_SHA",
    )
    _load_validated_pair_gap_reference()


def _serialization_role_masks(
    features: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    required = {"input_ids", "attention_mask", "claim_mask", "evidence_mask"}
    require(required <= set(features), f"ROLE_FEATURE_KEYS:{sorted(features)}")

    input_ids = features["input_ids"]
    attention = features["attention_mask"].to(dtype=torch.bool)
    claim = features["claim_mask"].to(dtype=torch.bool)
    evidence = features["evidence_mask"].to(dtype=torch.bool)

    require(input_ids.ndim == 2, "ROLE_INPUT_IDS_RANK")
    require(attention.shape == input_ids.shape, "ROLE_ATTENTION_SHAPE")
    require(claim.shape == input_ids.shape, "ROLE_CLAIM_SHAPE")
    require(evidence.shape == input_ids.shape, "ROLE_EVIDENCE_SHAPE")
    require(not bool(torch.any(claim & evidence).item()), "ROLE_CLAIM_EVIDENCE_OVERLAP")
    require(
        not bool(torch.any(claim & ~attention).item()),
        "ROLE_CLAIM_OUTSIDE_ATTENTION",
    )
    require(
        not bool(torch.any(evidence & ~attention).item()),
        "ROLE_EVIDENCE_OUTSIDE_ATTENTION",
    )

    eos = attention & ~claim & ~evidence
    eos_per_row = torch.sum(eos.to(torch.int64), dim=1)
    require(
        bool(torch.all(eos_per_row == 1).item()),
        f"ROLE_EOS_COUNT:{eos_per_row.detach().cpu().tolist()}",
    )
    require(
        bool(torch.all(input_ids[eos] == p3a.EOS_TOKEN_ID).item()),
        "ROLE_EOS_TOKEN_ID",
    )

    batch, seq_len = input_ids.shape
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    claim_count = torch.sum(claim.to(torch.int64), dim=1)
    evidence_count = torch.sum(evidence.to(torch.int64), dim=1)
    eos_index = torch.argmax(eos.to(torch.int64), dim=1)

    require(bool(torch.all(claim_count > 0).item()), "ROLE_EMPTY_CLAIM")
    require(bool(torch.all(evidence_count > 0).item()), "ROLE_EMPTY_EVIDENCE")
    require(
        bool(torch.all(eos_index == claim_count).item()),
        "ROLE_EOS_NOT_AFTER_CLAIM",
    )
    require(
        bool(
            torch.all(
                claim
                == (
                    positions
                    < claim_count.unsqueeze(1)
                )
            ).item()
        ),
        "ROLE_CLAIM_NOT_PREFIX",
    )
    expected_evidence = (
        (positions > eos_index.unsqueeze(1))
        & (
            positions
            <= (eos_index + evidence_count).unsqueeze(1)
        )
    )
    require(
        bool(torch.all(evidence == expected_evidence).item()),
        "ROLE_EVIDENCE_NOT_CONTIGUOUS",
    )
    require(
        bool(torch.all(attention == (claim | eos | evidence)).item()),
        "ROLE_ACTIVE_PARTITION",
    )

    pad = ~attention
    if bool(torch.any(pad).item()):
        require(
            bool(torch.all(input_ids[pad] == p3a.PAD_TOKEN_ID).item()),
            "ROLE_PAD_TOKEN_ID",
        )

    return {
        "CLAIM": claim,
        "EOS": eos,
        "EVIDENCE": evidence,
    }


def _pair_opportunity_counts(
    role_masks: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    claim = role_masks["CLAIM"]
    eos = role_masks["EOS"]
    evidence = role_masks["EVIDENCE"]
    require(claim.shape == eos.shape == evidence.shape, "COUNT_ROLE_SHAPE")
    batch, seq_len = claim.shape
    require(seq_len == p3a.MAX_LENGTH, f"COUNT_SEQ_LEN:{seq_len}")

    positions = torch.arange(seq_len, device=claim.device)
    gap_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
    strict = gap_matrix > 0

    output: dict[str, dict[str, Any]] = {}
    for left, right in REGION_PAIRS:
        source = role_masks[left]
        target = role_masks[right]
        pair_mask = (
            source[:, :, None]
            & target[:, None, :]
            & strict.unsqueeze(0)
        )
        gaps = gap_matrix.unsqueeze(0).expand(batch, -1, -1)
        counts = torch.bincount(
            gaps[pair_mask].to(torch.int64),
            minlength=seq_len,
        )
        by_gap = [
            int(value)
            for value in counts[1:seq_len].detach().cpu().tolist()
        ]
        name = f"{left}->{right}"
        output[name] = {
            "by_gap": by_gap,
            "windows": _window_sums(by_gap),
            "all": int(sum(by_gap)),
        }

    total = sum(row["all"] for row in output.values())
    active = claim | eos | evidence
    active_lengths = torch.sum(active.to(torch.int64), dim=1)
    expected = int(
        torch.sum(active_lengths * (active_lengths - 1) // 2).item()
    )
    require(total == expected, f"COUNT_STRICT_PAIR_COVERAGE:{total}:{expected}")
    return output


def _window_sums(values: Sequence[float | int]) -> dict[str, float]:
    observed = [float(value) for value in values]
    require(len(observed) == p3a.MAX_LENGTH - 1, f"WINDOW_VECTOR_LEN:{len(observed)}")
    output: dict[str, float] = {}
    for name, (start, stop) in WINDOWS.items():
        lo = start - 1
        hi = min(stop, len(observed))
        output[name] = float(sum(observed[lo:hi]))
    return output


def _cross_strict_dyadic_fft(
    *,
    source_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    p_chunk: torch.Tensor,
    survival_chunk: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    require(source_chunk.shape == target_chunk.shape, "DYADIC_CROSS_COMPONENT_SHAPE")
    require(source_chunk.ndim == 4, "DYADIC_CROSS_RANK")
    k_count, batch, observed_seq_len, width = source_chunk.shape
    require(observed_seq_len == seq_len, "DYADIC_CROSS_SEQ_LEN")
    require(
        tuple(p_chunk.shape) == (batch, seq_len, width),
        "DYADIC_CROSS_P_SHAPE",
    )
    require(
        tuple(survival_chunk.shape) == (batch, seq_len, width),
        "DYADIC_CROSS_SURVIVAL_SHAPE",
    )

    padded_len = 1 << ((seq_len - 1).bit_length())
    if padded_len > seq_len:
        pad_count = padded_len - seq_len
        source_pad = torch.cat(
            (
                source_chunk,
                torch.zeros(
                    (k_count, batch, pad_count, width),
                    device=source_chunk.device,
                    dtype=source_chunk.dtype,
                ),
            ),
            dim=2,
        )
        target_pad = torch.cat(
            (
                target_chunk,
                torch.zeros(
                    (k_count, batch, pad_count, width),
                    device=target_chunk.device,
                    dtype=target_chunk.dtype,
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
                    device=source_chunk.device,
                    dtype=survival_chunk.dtype,
                ),
            ),
            dim=1,
        )
    else:
        source_pad = source_chunk
        target_pad = target_chunk
        p_pad = p_chunk
        survival_pad = survival_chunk

    gap = torch.zeros(
        (k_count, seq_len),
        device=source_chunk.device,
        dtype=torch.float64,
    )

    block_len = 2
    while block_len <= padded_len:
        half = block_len // 2
        block_count = padded_len // block_len

        p_blocks = p_pad.reshape(batch, block_count, block_len, width)
        source_blocks = source_pad.reshape(
            k_count, batch, block_count, block_len, width
        )
        target_blocks = target_pad.reshape(
            k_count, batch, block_count, block_len, width
        )
        survival_blocks = survival_pad.reshape(
            batch, block_count, block_len, width
        )

        left_p = p_blocks[:, :, :half, :]
        right_p = p_blocks[:, :, half:, :]
        shift = torch.amin(left_p, dim=2)

        left_exponent = -left_p + shift[:, :, None, :]
        right_exponent = right_p - shift[:, :, None, :]
        require(
            bool(torch.all(left_exponent <= 0.0).item()),
            "DYADIC_CROSS_LEFT_SCALE_SIGN",
        )
        require(
            bool(torch.all(right_exponent <= 0.0).item()),
            "DYADIC_CROSS_RIGHT_SCALE_SIGN",
        )

        x_scale = torch.exp(left_exponent).to(dtype=source_chunk.dtype)
        y_scale = torch.exp(right_exponent).to(dtype=source_chunk.dtype)
        require(
            bool(torch.all(torch.isfinite(x_scale)).item()),
            "DYADIC_CROSS_X_SCALE_NONFINITE",
        )
        require(
            bool(torch.all(torch.isfinite(y_scale)).item()),
            "DYADIC_CROSS_Y_SCALE_NONFINITE",
        )

        source = (
            source_blocks[:, :, :, :half, :]
            * x_scale.unsqueeze(0)
        )
        target = (
            target_blocks[:, :, :, half:, :]
            * survival_blocks[:, :, half:, :].to(
                dtype=source_chunk.dtype
            ).unsqueeze(0)
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


def _region_pair_gap_interference_fft(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
    shape: Any,
) -> tuple[dict[str, torch.Tensor], float]:
    """Return exact signed C_region(d), index 0 reserved as zero."""
    require(component.ndim == 4, "REGION_COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(seq_len == p3a.MAX_LENGTH, f"REGION_SEQ_LEN:{seq_len}")
    require(width == shape.intermediate_size * shape.state_size, "REGION_WIDTH")
    require(
        tuple(log_a.shape)
        == (batch, seq_len, shape.intermediate_size, shape.state_size),
        "REGION_LOG_A_SHAPE",
    )
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "REGION_MASK_SHAPE",
    )
    require(
        tuple(survival.shape) == (batch, seq_len, width),
        "REGION_SURVIVAL_SHAPE",
    )

    for role in REGION_NAMES:
        require(role in role_masks, f"REGION_ROLE_MISSING:{role}")
        require(
            tuple(role_masks[role].shape) == (batch, seq_len),
            f"REGION_ROLE_SHAPE:{role}",
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
        "REGION_EMPTY_VALID_SEQUENCE",
    )

    gap = torch.zeros(
        (len(REGION_PAIRS) * k_count, seq_len),
        device=component.device,
        dtype=torch.float64,
    )
    n_fft = 1 << ((2 * seq_len - 1).bit_length())
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
            "REGION_ACTIVE_PREFIX_NONFINITE",
        )
        half_span = 0.5 * (p_max - p_min)
        chunk_max = float(torch.amax(half_span).item())
        max_half_span = max(max_half_span, chunk_max)

        w_native = w[:, :, :, start:stop].to(dtype=torch.float32)
        base_active = torch.where(
            source_active[None, :, :, None],
            w_native,
            torch.zeros_like(w_native),
        )

        source_parts = []
        target_parts = []
        for left, right in REGION_PAIRS:
            source_role = role_masks[left].to(
                device=component.device,
                dtype=torch.bool,
            )
            target_role = role_masks[right].to(
                device=component.device,
                dtype=torch.bool,
            )
            source_parts.append(
                torch.where(
                    source_role[None, :, :, None],
                    base_active,
                    torch.zeros_like(base_active),
                )
            )
            target_parts.append(
                torch.where(
                    target_role[None, :, :, None],
                    base_active,
                    torch.zeros_like(base_active),
                )
            )

        source_stack = torch.cat(source_parts, dim=0)
        target_stack = torch.cat(target_parts, dim=0)
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
                f"REGION_X_SCALE_NONFINITE:{chunk_max}",
            )
            require(
                bool(torch.all(torch.isfinite(y_scale)).item()),
                f"REGION_Y_SCALE_NONFINITE:{chunk_max}",
            )

            source = source_stack * x_scale.unsqueeze(0)
            target = (
                target_stack
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
        else:
            gap += _cross_strict_dyadic_fft(
                source_chunk=source_stack,
                target_chunk=target_stack,
                p_chunk=p,
                survival_chunk=survival_chunk,
                seq_len=seq_len,
            )

        del source_stack, target_stack, source_parts, target_parts

    require(
        bool(torch.all(torch.isfinite(gap)).item()),
        "REGION_GAP_NONFINITE",
    )
    require(
        bool(torch.all(gap[:, 0] == 0.0).item()),
        "REGION_GAP_ZERO_INDEX_NONZERO",
    )

    reshaped = gap.reshape(len(REGION_PAIRS), k_count, seq_len)
    return {
        name: reshaped[index]
        for index, name in enumerate(REGION_PAIR_NAMES)
    }, max_half_span


def _component_region_pair_stats(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
    shape: Any,
) -> list[dict[str, Any]]:
    base_rows = pairgap._component_pair_gap_stats(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        shape=shape,
    )
    region_gap, region_span = _region_pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        role_masks=role_masks,
        shape=shape,
    )
    k_count = component.shape[0]
    require(len(base_rows) == k_count, "REGION_BASE_ROW_COUNT")

    rows = []
    for index in range(k_count):
        reconstructed = torch.zeros(
            p3a.MAX_LENGTH,
            device=component.device,
            dtype=torch.float64,
        )
        by_region: dict[str, list[float]] = {}
        for name in REGION_PAIR_NAMES:
            vector = region_gap[name][index]
            reconstructed += vector
            by_region[name] = [
                float(value)
                for value in vector[1:].detach().cpu().tolist()
            ]

        expected = torch.tensor(
            [0.0] + [float(x) for x in base_rows[index]["pair_gap_C"]],
            device=component.device,
            dtype=torch.float64,
        )
        diff = torch.abs(reconstructed - expected)
        denom = torch.clamp(torch.abs(expected), min=ENERGY_EPSILON)
        rel = diff / denom
        pass_mask = (diff <= PAIR_GAP_ENERGY_ATOL) | (rel <= PAIR_GAP_ENERGY_RTOL)
        require(
            bool(torch.all(pass_mask).item()),
            f"REGION_PAIR_GAP_RECON:{index}:"
            f"{float(torch.amax(diff).item())}:"
            f"{float(torch.amax(rel).item())}",
        )

        row = dict(base_rows[index])
        row["region_pair_C_by_gap"] = by_region
        row["region_pair_gap_reconstruction_abs_max"] = float(
            torch.amax(diff).item()
        )
        row["region_pair_gap_reconstruction_rel_max"] = float(
            torch.amax(rel).item()
        )
        row["region_pair_fft_half_log_span_max"] = region_span
        rows.append(row)
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
        "region_pair_C_visible": {
            name: [] for name in REGION_PAIR_NAMES
        },
        "region_pair_C_complement": {
            name: [] for name in REGION_PAIR_NAMES
        },
        "pair_gap_C_reconstruction_abs_max": 0.0,
        "pair_gap_Q_reconstruction_abs_max": 0.0,
        "pair_gap_fft_half_log_span_max": 0.0,
        "region_pair_gap_reconstruction_abs_max": 0.0,
        "region_pair_gap_reconstruction_rel_max": 0.0,
        "region_pair_fft_half_log_span_max": 0.0,
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

    gap_key = f"pair_gap_C_{prefix}"
    observed_gap = [float(x) for x in stats["pair_gap_C"]]
    if not accumulator[gap_key]:
        accumulator[gap_key] = [0.0 for _ in observed_gap]
    require(
        len(accumulator[gap_key]) == len(observed_gap),
        f"PAIR_GAP_LENGTH:{prefix}",
    )
    for index, value in enumerate(observed_gap):
        accumulator[gap_key][index] += value

    region_key = f"region_pair_C_{prefix}"
    for name in REGION_PAIR_NAMES:
        observed = [
            float(x)
            for x in stats["region_pair_C_by_gap"][name]
        ]
        if not accumulator[region_key][name]:
            accumulator[region_key][name] = [0.0 for _ in observed]
        require(
            len(accumulator[region_key][name]) == len(observed),
            f"REGION_PAIR_LENGTH:{prefix}:{name}",
        )
        for index, value in enumerate(observed):
            accumulator[region_key][name][index] += value

    for dst, src in (
        ("pair_gap_C_reconstruction_abs_max", "pair_gap_C_reconstruction_abs"),
        ("pair_gap_Q_reconstruction_abs_max", "pair_gap_Q_reconstruction_abs"),
        ("pair_gap_fft_half_log_span_max", "pair_gap_fft_half_log_span_max"),
        (
            "region_pair_gap_reconstruction_abs_max",
            "region_pair_gap_reconstruction_abs_max",
        ),
        (
            "region_pair_gap_reconstruction_rel_max",
            "region_pair_gap_reconstruction_rel_max",
        ),
        (
            "region_pair_fft_half_log_span_max",
            "region_pair_fft_half_log_span_max",
        ),
    ):
        accumulator[dst] = max(
            float(accumulator[dst]),
            float(stats[src]),
        )


def _merge_pair_counts(
    workers: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {
        name: {
            "by_gap": [0 for _ in range(p3a.MAX_LENGTH - 1)],
        }
        for name in REGION_PAIR_NAMES
    }
    for worker in workers:
        observed = worker["pair_opportunity_counts"]
        require(
            set(observed) == set(REGION_PAIR_NAMES),
            "WORKER_PAIR_COUNT_REGION_KEYS",
        )
        for name in REGION_PAIR_NAMES:
            vector = [int(x) for x in observed[name]["by_gap"]]
            require(
                len(vector) == p3a.MAX_LENGTH - 1,
                f"WORKER_PAIR_COUNT_LEN:{name}",
            )
            for index, value in enumerate(vector):
                merged[name]["by_gap"][index] += value

    for name in REGION_PAIR_NAMES:
        vector = merged[name]["by_gap"]
        merged[name]["windows"] = {
            key: int(value)
            for key, value in _window_sums(vector).items()
        }
        merged[name]["all"] = int(sum(vector))

    total = sum(row["all"] for row in merged.values())
    require(total > 0, "MERGED_PAIR_COUNT_EMPTY")
    return merged


def _replay_scalar(
    *,
    name: str,
    observed: float,
    expected: float,
    log_metric: bool = False,
) -> dict[str, Any]:
    abs_error = abs(observed - expected)
    rel_error = _relative_error(observed, expected)
    passed = (
        abs_error <= (PAIR_GAP_LOG_ATOL if log_metric else PAIR_GAP_ENERGY_ATOL)
        or (
            not log_metric
            and rel_error <= PAIR_GAP_ENERGY_RTOL
        )
    )
    require(
        passed,
        f"VALIDATED_PAIR_GAP_REPLAY:{name}:{observed}:{expected}:"
        f"{abs_error}:{rel_error}",
    )
    return {
        "observed": observed,
        "expected": expected,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "pass": True,
    }


def _replay_vector(
    *,
    name: str,
    observed: Sequence[float],
    expected: Sequence[float],
) -> dict[str, Any]:
    require(len(observed) == len(expected), f"REPLAY_VECTOR_LEN:{name}")
    max_abs = 0.0
    max_rel = 0.0
    for obs, exp in zip(observed, expected):
        abs_error = abs(float(obs) - float(exp))
        rel_error = _relative_error(float(obs), float(exp))
        require(
            abs_error <= PAIR_GAP_ENERGY_ATOL
            or rel_error <= PAIR_GAP_ENERGY_RTOL,
            f"VALIDATED_PAIR_GAP_VECTOR_REPLAY:{name}:{abs_error}:{rel_error}",
        )
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)
    return {
        "pass": True,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def _region_component_output(
    *,
    region_c: Mapping[str, Sequence[float]],
    self_energy: float,
    pair_counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in REGION_PAIR_NAMES:
        c_vector = [float(x) for x in region_c[name]]
        h_vector = [value / self_energy for value in c_vector]
        h_windows = _window_sums(h_vector)
        counts = {
            key: int(value)
            for key, value in pair_counts[name]["windows"].items()
        }
        mean_h = {}
        for window, h_value in h_windows.items():
            count = counts[window]
            mean_h[window] = (
                h_value / count
                if count > 0
                else None
            )
        output[name] = {
            "H_by_gap": h_vector,
            "H_windows": h_windows,
            "pair_counts": counts,
            "mean_h_windows": mean_h,
        }
    return output


def _finalize_orientation(
    accumulator: Mapping[str, Any],
    reference: Mapping[str, Any],
    pair_counts: Mapping[str, Mapping[str, Any]],
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
    require(all(value > 0.0 for value in values.values()), "FINAL_NONPOSITIVE_ENERGY")

    c_visible = values["E_visible"] - values["S_visible"]
    c_complement = values["E_complement"] - values["S_complement"]
    q_visible = values["E_visible"] / values["S_visible"]
    q_complement = values["E_complement"] / values["S_complement"]
    require(q_visible > 0.0 and q_complement > 0.0, "FINAL_NONPOSITIVE_Q")
    l_interference = math.log(q_complement / q_visible)

    c_gap_visible = [float(x) for x in accumulator["pair_gap_C_visible"]]
    c_gap_complement = [float(x) for x in accumulator["pair_gap_C_complement"]]
    require(
        len(c_gap_visible) == len(c_gap_complement) == p3a.MAX_LENGTH - 1,
        "FINAL_PAIR_GAP_LENGTH",
    )
    h_gap_visible = [x / values["S_visible"] for x in c_gap_visible]
    h_gap_complement = [x / values["S_complement"] for x in c_gap_complement]

    region_visible = _region_component_output(
        region_c=accumulator["region_pair_C_visible"],
        self_energy=values["S_visible"],
        pair_counts=pair_counts,
    )
    region_complement = _region_component_output(
        region_c=accumulator["region_pair_C_complement"],
        self_energy=values["S_complement"],
        pair_counts=pair_counts,
    )

    for prefix, expected_vector, region_output in (
        ("visible", h_gap_visible, region_visible),
        ("complement", h_gap_complement, region_complement),
    ):
        reconstructed = [
            sum(
                float(region_output[name]["H_by_gap"][index])
                for name in REGION_PAIR_NAMES
            )
            for index in range(p3a.MAX_LENGTH - 1)
        ]
        _replay_vector(
            name=f"REGION_H_RECON_{prefix}",
            observed=reconstructed,
            expected=expected_vector,
        )

    scalar_replay = {}
    for key in (
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    ):
        scalar_replay[key] = _replay_scalar(
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
        scalar_replay[key] = _replay_scalar(
            name=key,
            observed=observed,
            expected=float(reference[key]),
        )
    scalar_replay["L_interference"] = _replay_scalar(
        name="L_interference",
        observed=l_interference,
        expected=float(reference["L_interference"]),
        log_metric=True,
    )

    vector_replay = {
        "C_gap_visible": _replay_vector(
            name="C_gap_visible",
            observed=c_gap_visible,
            expected=[float(x) for x in reference["C_gap_visible"]],
        ),
        "C_gap_complement": _replay_vector(
            name="C_gap_complement",
            observed=c_gap_complement,
            expected=[float(x) for x in reference["C_gap_complement"]],
        ),
        "H_gap_visible": _replay_vector(
            name="H_gap_visible",
            observed=h_gap_visible,
            expected=[float(x) for x in reference["H_gap_visible"]],
        ),
        "H_gap_complement": _replay_vector(
            name="H_gap_complement",
            observed=h_gap_complement,
            expected=[float(x) for x in reference["H_gap_complement"]],
        ),
    }

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
        "H_gap_visible": h_gap_visible,
        "H_gap_complement": h_gap_complement,
        "region_pair_visible": region_visible,
        "region_pair_complement": region_complement,
        "batch_pair_gap_C_reconstruction_abs_max": float(
            accumulator["pair_gap_C_reconstruction_abs_max"]
        ),
        "batch_pair_gap_Q_reconstruction_abs_max": float(
            accumulator["pair_gap_Q_reconstruction_abs_max"]
        ),
        "pair_gap_fft_half_log_span_max": float(
            accumulator["pair_gap_fft_half_log_span_max"]
        ),
        "region_pair_gap_reconstruction_abs_max": float(
            accumulator["region_pair_gap_reconstruction_abs_max"]
        ),
        "region_pair_gap_reconstruction_rel_max": float(
            accumulator["region_pair_gap_reconstruction_rel_max"]
        ),
        "region_pair_fft_half_log_span_max": float(
            accumulator["region_pair_fft_half_log_span_max"]
        ),
        "validated_pair_gap_replay": {
            "scalars": scalar_replay,
            "vectors": vector_replay,
        },
    }


def _merge_worker_accumulators(
    workers: Sequence[Mapping[str, Any]],
    pair_counts: Mapping[str, Mapping[str, Any]],
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
        "region_pair_gap_reconstruction_abs_max",
        "region_pair_gap_reconstruction_rel_max",
        "region_pair_fft_half_log_span_max",
    )
    vector_sum = (
        "pair_gap_C_visible",
        "pair_gap_C_complement",
    )
    region_maps = (
        "region_pair_C_visible",
        "region_pair_C_complement",
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
                    **{
                        name: {
                            pair: [] for pair in REGION_PAIR_NAMES
                        }
                        for name in region_maps
                    },
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
                    f"MERGE_VECTOR_LEN:{name}:{key}",
                )
                for index, value in enumerate(observed):
                    dst[name][index] += value
            for map_name in region_maps:
                for pair in REGION_PAIR_NAMES:
                    observed = [float(x) for x in row[map_name][pair]]
                    if not dst[map_name][pair]:
                        dst[map_name][pair] = [0.0 for _ in observed]
                    require(
                        len(dst[map_name][pair]) == len(observed),
                        f"MERGE_REGION_VECTOR_LEN:{map_name}:{pair}:{key}",
                    )
                    for index, value in enumerate(observed):
                        dst[map_name][pair][index] += value

    require(len(merged) == 36, f"MERGED_ACCUMULATOR_COUNT:{len(merged)}")
    reference = _load_validated_pair_gap_reference()
    output = []
    for key in sorted(merged):
        require(key in reference, f"VALIDATED_REFERENCE_MISSING:{key}")
        output.append(
            _finalize_orientation(
                merged[key],
                reference[key],
                pair_counts,
            )
        )
    return output


def _mean(values: Sequence[float]) -> float:
    require(bool(values), "EMPTY_MEAN")
    return sum(float(x) for x in values) / len(values)


def _median(values: Sequence[float]) -> float:
    observed = sorted(float(x) for x in values)
    require(bool(observed), "EMPTY_MEDIAN")
    middle = len(observed) // 2
    if len(observed) % 2:
        return observed[middle]
    return 0.5 * (observed[middle - 1] + observed[middle])


def _grouped_region_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ("PRIMARY_A", "CONTROL_R"):
        subset = [row for row in rows if row["group"] == group]
        require(len(subset) == 18, f"GROUPED_COUNT:{group}")
        group_out: dict[str, Any] = {}
        for component in ("visible", "complement"):
            comp_out: dict[str, Any] = {}
            for pair in REGION_PAIR_NAMES:
                pair_out: dict[str, Any] = {}
                for window in WINDOWS:
                    values = [
                        float(
                            row[f"region_pair_{component}"][pair]["H_windows"][window]
                        )
                        for row in subset
                    ]
                    pair_out[window] = {
                        "count": len(values),
                        "median": _median(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                comp_out[pair] = pair_out
            group_out[component] = comp_out
        output[group] = group_out
    return output


def _source_matched(
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

        scalars = {}
        for name in ("log_Q_visible", "log_Q_complement", "L_interference"):
            p = _mean([float(row[name]) for row in primary])
            c = _mean([float(row[name]) for row in control])
            scalars[name] = {
                "primary_mean": p,
                "control_mean": c,
                "primary_minus_control": p - c,
            }

        region_metrics: dict[str, Any] = {}
        for component in ("visible", "complement"):
            comp_out: dict[str, Any] = {}
            for pair in REGION_PAIR_NAMES:
                pair_out = {}
                for window in WINDOWS:
                    p = _mean(
                        [
                            float(
                                row[f"region_pair_{component}"][pair]
                                ["H_windows"][window]
                            )
                            for row in primary
                        ]
                    )
                    c = _mean(
                        [
                            float(
                                row[f"region_pair_{component}"][pair]
                                ["H_windows"][window]
                            )
                            for row in control
                        ]
                    )
                    pair_out[window] = {
                        "primary_mean": p,
                        "control_mean": c,
                        "primary_minus_control": p - c,
                    }
                comp_out[pair] = pair_out
            region_metrics[component] = comp_out

        output.append(
            {
                "source": source,
                "metrics": scalars,
                "region_pair_metrics": region_metrics,
            }
        )
    return output


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
            "base_pair_gap_group_decompositions": groups,
            "serialization_region_group_decompositions": groups,
            "region_pair_classes_per_group": len(REGION_PAIRS),
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
        "total_base_pair_gap_group_decompositions": sum(
            row["base_pair_gap_group_decompositions"] for row in workers.values()
        ),
        "total_serialization_region_group_decompositions": sum(
            row["serialization_region_group_decompositions"]
            for row in workers.values()
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
    print("GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_STATIC_CONTRACT_PASS")
    print(f"HEAD={head}")
    print("REGION_PAIRS=" + ",".join(REGION_PAIR_NAMES))
    print("PRIMARY_WINDOW=gap1_32")
    print("DEV_ROWS=840")
    print("VALID_TOKEN_COUNT=60094")
    print("TOTAL_ORIENTATIONS=36")
    print("VALIDATED_PAIR_GAP_REFERENCE=AUTHENTICATED")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")


def run_preflight(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    transport.legacy._validate_two_t4s()
    transport._runtime_inputs(args)
    print("GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_CUDA_PREFLIGHT_PASS")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("EXECUTION_ENGINE=RECURRENT_SERIALIZATION_REGION_PAIR_EXACT_INTERFERENCE_V1")
    print("VALIDATED_PAIR_GAP_REFERENCE=AUTHENTICATED")
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
        require(len(rows) == 4, f"SOURCE_ORIENTATION_COUNT:{cell_name(source)}")

    row_start, row_stop = worker_row_range(worker_id)
    row_count = row_stop - row_start
    total_batches = math.ceil(row_count / BATCH_ROWS)
    valid_token_seen = 0
    pair_counts_acc = {
        name: {
            "by_gap": [0 for _ in range(p3a.MAX_LENGTH - 1)]
        }
        for name in REGION_PAIR_NAMES
    }

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

        role_masks = _serialization_role_masks(batch_features)
        batch_pair_counts = _pair_opportunity_counts(role_masks)
        for name in REGION_PAIR_NAMES:
            vector = batch_pair_counts[name]["by_gap"]
            for index, value in enumerate(vector):
                pair_counts_acc[name]["by_gap"][index] += int(value)

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
        pair_survival = pairgap._backward_survival_factor(
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

            for group_start in range(0, len(source_rows), TARGETS_PER_TRANSPORT):
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
                stats = _component_region_pair_stats(
                    component=combined,
                    log_a=pair_log_a,
                    survival=pair_survival,
                    attention_mask=batch_mask,
                    role_masks=role_masks,
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
                        _relative_error(float(visible_stats["R"]), raw_visible)
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

        del raw_by_cell, pair_log_a, pair_survival, role_masks
        print(
            "GEN5_RECURRENT_SERIALIZATION_REGION_PROGRESS "
            f"worker={worker_id} batch={batch_index}/{total_batches} "
            f"shard_rows={stop-row_start}/{row_count}",
            flush=True,
        )

    for name in REGION_PAIR_NAMES:
        vector = pair_counts_acc[name]["by_gap"]
        pair_counts_acc[name]["windows"] = {
            key: int(value)
            for key, value in _window_sums(vector).items()
        }
        pair_counts_acc[name]["all"] = int(sum(vector))

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
        "pair_opportunity_counts": pair_counts_acc,
        "valid_token_count": valid_token_seen,
        "execution_engine": "RECURRENT_SERIALIZATION_REGION_PAIR_EXACT_INTERFERENCE_V1",
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
        "GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_WORKER_PASS "
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
        require(worker["execution_head"] == args.expected_head, "MIXED_WORKER_HEAD")
    require(
        sum(int(worker["valid_token_count"]) for worker in workers)
        == VALID_TOKEN_COUNT,
        "MERGED_VALID_TOKEN_COUNT",
    )

    pair_counts = _merge_pair_counts(workers)
    rows = _merge_worker_accumulators(workers, pair_counts)
    require(len(rows) == 36, "MERGED_ORIENTATION_COUNT")
    require(
        max(float(row["raw_reconstruction_max_abs"]) for row in rows)
        <= transport.RAW_RECON_ATOL,
        "MERGED_RAW_RECONSTRUCTION_FAIL",
    )

    for row in rows:
        scalar_checks = row["validated_pair_gap_replay"]["scalars"].values()
        vector_checks = row["validated_pair_gap_replay"]["vectors"].values()
        require(
            all(bool(check["pass"]) for check in scalar_checks),
            "VALIDATED_PAIR_GAP_SCALAR_REPLAY_FAIL",
        )
        require(
            all(bool(check["pass"]) for check in vector_checks),
            "VALIDATED_PAIR_GAP_VECTOR_REPLAY_FAIL",
        )

    grouped = _grouped_region_summary(rows)
    matched = _source_matched(rows)

    max_region_abs = max(
        float(row["region_pair_gap_reconstruction_abs_max"])
        for row in rows
    )
    max_region_rel = max(
        float(row["region_pair_gap_reconstruction_rel_max"])
        for row in rows
    )
    max_scalar_replay_abs = max(
        float(check["abs_error"])
        for row in rows
        for check in row["validated_pair_gap_replay"]["scalars"].values()
    )
    max_vector_replay_abs = max(
        float(check["max_abs_error"])
        for row in rows
        for check in row["validated_pair_gap_replay"]["vectors"].values()
    )

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_VALIDATED_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_pair_gap_run": VALIDATED_PAIR_GAP_RUN,
        "population": "FROZEN_PHASE3A_P0_DEV",
        "dev_rows": DEV_ROWS,
        "valid_token_count": VALID_TOKEN_COUNT,
        "dev_order_sha256": transport.DEV_ORDER_SHA256,
        "dev_encoding_sha256": transport.DEV_ENCODING_SHA256,
        "serialization_contract": "claim[:63]+EOS(0)+evidence[:64]",
        "region_pairs": list(REGION_PAIR_NAMES),
        "windows": {
            key: [start, stop]
            for key, (start, stop) in WINDOWS.items()
        },
        "pair_opportunity_counts": pair_counts,
        "grouped_ordered_region_summary": grouped,
        "source_cell_matched_primary_minus_control": matched,
        "validated_pair_gap_replay": {
            "pass": True,
            "max_scalar_abs_error": max_scalar_replay_abs,
            "max_vector_abs_error": max_vector_replay_abs,
            "energy_rtol": PAIR_GAP_ENERGY_RTOL,
            "energy_atol": PAIR_GAP_ENERGY_ATOL,
            "log_atol": PAIR_GAP_LOG_ATOL,
        },
        "region_pair_reconstruction": {
            "pass": True,
            "max_abs_error": max_region_abs,
            "max_rel_error": max_region_rel,
        },
        "pair_count_normalization_role": "DESCRIPTIVE_DENSITY_CONTROL_ONLY",
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
    summary_path = output_root / "serialization_region_pair_summary.json"
    rows_path = output_root / "orientation_serialization_region_pair_metrics.jsonl"
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
        "validated_pair_gap_script_blob": VALIDATED_PAIR_GAP_SCRIPT_BLOB,
        "phase3a_runtime_blob": PHASE3A_RUNTIME_BLOB,
        "validated_pair_gap_artifact_sha256": VALIDATED_PAIR_GAP_FILES,
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

    print("GEN5_NATIVE_RECURRENT_SERIALIZATION_REGION_MERGE_PASS")
    print("ORIENTATIONS=36")
    print("VALIDATED_PAIR_GAP_REPLAY=PASS")
    print("REGION_PAIR_RECONSTRUCTION=PASS")
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


def _validate_args(args: argparse.Namespace) -> None:
    if args.static_contract_check:
        require(args.implementation_freeze_commit is None, "STATIC_IMPLEMENTATION_FREEZE_FORBIDDEN")
        require(args.model_snapshot is None, "STATIC_MODEL_SNAPSHOT_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(args.scratch_root is None, "STATIC_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_FORBIDDEN")
        require(args.worker_id is None, "STATIC_WORKER_FORBIDDEN")
        return

    require(args.expected_head is not None, "RUNTIME_EXPECTED_HEAD_REQUIRED")
    require(
        args.implementation_freeze_commit is not None,
        "RUNTIME_IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(args.model_snapshot is not None, "RUNTIME_MODEL_SNAPSHOT_REQUIRED")
    require(args.tokenizer_snapshot is not None, "RUNTIME_TOKENIZER_SNAPSHOT_REQUIRED")
    require(args.checkpoint is not None, "RUNTIME_CHECKPOINT_REQUIRED")

    if args.preflight:
        require(args.scratch_root is None, "PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.worker_id is None, "PREFLIGHT_WORKER_FORBIDDEN")
    elif args.run_worker:
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.output_root is None, "WORKER_OUTPUT_ROOT_FORBIDDEN")
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    elif args.merge_only:
        require(args.scratch_root is not None, "MERGE_SCRATCH_REQUIRED")
        require(args.output_root is not None, "MERGE_OUTPUT_REQUIRED")
        require(args.worker_id is None, "MERGE_WORKER_FORBIDDEN")
    elif args.run:
        require(args.scratch_root is not None, "RUN_SCRATCH_REQUIRED")
        require(args.output_root is not None, "RUN_OUTPUT_REQUIRED")
        require(args.worker_id is None, "RUN_WORKER_FORBIDDEN")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
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
