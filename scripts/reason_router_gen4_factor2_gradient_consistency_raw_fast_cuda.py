#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_factor2_small_alpha_behavioral_raw_fast_cuda
    as behavior_runner,
)
from scripts import (
    reason_router_gen4_factor2_small_alpha_native_readout_raw_fast_cuda
    as readout_runner,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "57644369580c6f560e8eeb1ddbfc09d714a4c789"

SCALES = ("mamba370m", "mamba14b")
SCALE_TO_PHYSICAL_GPU = {
    "mamba370m": 0,
    "mamba14b": 1,
}
TARGET_CELLS = ("C0_SHAM", "C2_NAME")

ALL_PAIRS = tuple(f"xg1_fact_{i}" for i in range(5701, 6001))
SUBSET_SELECTION_SALT = "factor2-gradient-consistency-v1"
SUBSET_SIZE = 32


def _subset_score(pair: str) -> str:
    return hashlib.sha256(
        f"{SUBSET_SELECTION_SALT}:{pair}".encode("utf-8")
    ).hexdigest()


_SUBSET_UNSORTED = sorted(ALL_PAIRS, key=_subset_score)[:SUBSET_SIZE]
SUBSET_PAIRS = tuple(
    sorted(_SUBSET_UNSORTED, key=lambda pair: int(pair.rsplit("_", 1)[1]))
)

EPSILONS = (0.03125, 0.015625, 0.0078125)
SIGNED_EPSILONS = tuple(
    value
    for epsilon in EPSILONS
    for value in (epsilon, -epsilon)
)

ROWS_PER_SCALE = SUBSET_SIZE * len(TARGET_CELLS)
TOTAL_ROWS = len(SCALES) * ROWS_PER_SCALE
FORWARDS_PER_ROW = 1 + 2 * len(EPSILONS)
FORWARDS_PER_SCALE = ROWS_PER_SCALE * FORWARDS_PER_ROW
TOTAL_FORWARD_BUDGET = len(SCALES) * FORWARDS_PER_SCALE

READOUT_DIR = ROOT / (
    "reports/reason_router_gen4_mamba370m14b_"
    "factor2_small_alpha_native_readout_raw_v1"
)
READOUT_ITEM_FILE = "factor2_small_alpha_native_readout_items.jsonl"
READOUT_SUMMARY_FILE = "raw_readout_summary.json"
READOUT_MANIFEST_FILE = "artifact_manifest.json"
READOUT_ITEM_SHA256 = (
    "cb4e4115d7f21cc66e63153f49942eb4e3e1f6d410c1475c24666cc4f76a9fd8"
)
READOUT_SUMMARY_SHA256 = (
    "63d5eeec7f8353380d166bb25c994726e3f2cf4280568793c773a1182af1b3f7"
)
READOUT_MANIFEST_SHA256 = (
    "a0aed230659c6f69b2bd21cba9326b34a6a264139274fd6a7ac3a594ade73cdd"
)

ROW_FILE = "gradient_consistency_rows.jsonl"
SUMMARY_FILE = "raw_gradient_consistency_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-factor2-gradient-consistency-row-v1"
SUMMARY_SCHEMA = "gen4-factor2-gradient-consistency-summary-v1"
MANIFEST_SCHEMA = "gen4-factor2-gradient-consistency-manifest-v1"

TECHNICAL_GATE_RESULT = (
    "PASS_GEN4_FACTOR2_GRADIENT_CONSISTENCY_TECHNICAL_GATE"
)
RAW_RESULT = "PASS_GEN4_FACTOR2_GRADIENT_CONSISTENCY_RAW"

NATIVE_MARGIN_REPLAY_TOL = 1.0e-5


class GradientConsistencyRawError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GradientConsistencyRawError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GradientConsistencyRawError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            expected_head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR_MISSING")


def validate_protocol() -> None:
    require(len(ALL_PAIRS) == 300, "ALL_PAIR_COUNT")
    require(SUBSET_SIZE == 32, "SUBSET_SIZE")
    require(len(SUBSET_PAIRS) == SUBSET_SIZE, "SUBSET_PAIR_COUNT")
    require(len(set(SUBSET_PAIRS)) == SUBSET_SIZE, "SUBSET_PAIR_UNIQUE")
    require(set(SUBSET_PAIRS) <= set(ALL_PAIRS), "SUBSET_PAIR_DOMAIN")
    expected = tuple(
        sorted(
            sorted(ALL_PAIRS, key=_subset_score)[:SUBSET_SIZE],
            key=lambda pair: int(pair.rsplit("_", 1)[1]),
        )
    )
    require(SUBSET_PAIRS == expected, "SUBSET_SELECTION")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(
        EPSILONS == (0.03125, 0.015625, 0.0078125),
        "EPSILONS",
    )
    require(all(value > 0.0 for value in EPSILONS), "EPSILON_POSITIVE")
    require(
        SIGNED_EPSILONS
        == (
            0.03125,
            -0.03125,
            0.015625,
            -0.015625,
            0.0078125,
            -0.0078125,
        ),
        "SIGNED_EPSILONS",
    )
    require(ROWS_PER_SCALE == 64, "ROWS_PER_SCALE")
    require(TOTAL_ROWS == 128, "TOTAL_ROWS")
    require(FORWARDS_PER_ROW == 7, "FORWARDS_PER_ROW")
    require(FORWARDS_PER_SCALE == 448, "FORWARDS_PER_SCALE")
    require(TOTAL_FORWARD_BUDGET == 896, "TOTAL_FORWARD_BUDGET")
    require(
        SCALE_TO_PHYSICAL_GPU
        == {
            "mamba370m": 0,
            "mamba14b": 1,
        },
        "GPU_MAPPING",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


def load_frozen_readout() -> dict[tuple[str, str, str], dict[str, Any]]:
    item_path = READOUT_DIR / READOUT_ITEM_FILE
    summary_path = READOUT_DIR / READOUT_SUMMARY_FILE
    manifest_path = READOUT_DIR / READOUT_MANIFEST_FILE

    require(
        sha256_file(item_path) == READOUT_ITEM_SHA256,
        "READOUT_ITEM_SHA",
    )
    require(
        sha256_file(summary_path) == READOUT_SUMMARY_SHA256,
        "READOUT_SUMMARY_SHA",
    )
    require(
        sha256_file(manifest_path) == READOUT_MANIFEST_SHA256,
        "READOUT_MANIFEST_SHA",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        summary["result"]
        == "PASS_GEN4_FACTOR2_SMALL_ALPHA_NATIVE_READOUT_RAW",
        "READOUT_RESULT",
    )
    require(
        summary["population"] == "xg1_fact_5701..xg1_fact_6000",
        "READOUT_POPULATION",
    )
    require(int(summary["row_count"]) == 1200, "READOUT_ROW_COUNT")
    require(summary["behavioral_response_accessed"] is False, "READOUT_BLIND")
    require(summary["behavioral_merge_computed"] is False, "READOUT_MERGE")
    require(int(summary["p_value_count_executed"]) == 0, "READOUT_P_VALUE")
    require(
        manifest["item_sha256"] == READOUT_ITEM_SHA256,
        "READOUT_MANIFEST_ITEM_SHA",
    )

    rows = read_jsonl(item_path)
    require(len(rows) == 1200, "READOUT_ITEMS")
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["scale"]),
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in by_key, f"READOUT_DUPLICATE:{key}")
        by_key[key] = dict(row)

    expected = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in ALL_PAIRS
        for cell in TARGET_CELLS
    }
    require(set(by_key) == expected, "READOUT_COVERAGE")
    return by_key


def fixed_margin(
    logits: Sequence[float],
    *,
    correct_label_id: int,
    wrong_class_id: int,
) -> float:
    require(len(logits) == 3, "FIXED_MARGIN_LOGITS")
    require(correct_label_id in (0, 1, 2), "FIXED_MARGIN_LABEL")
    require(wrong_class_id in (0, 1, 2), "FIXED_MARGIN_WRONG")
    require(wrong_class_id != correct_label_id, "FIXED_MARGIN_DISTINCT")
    value = float(logits[correct_label_id]) - float(logits[wrong_class_id])
    require(math.isfinite(value), "FIXED_MARGIN_NONFINITE")
    return value


def dynamic_margin(
    logits: Sequence[float],
    *,
    correct_label_id: int,
) -> tuple[float, int]:
    require(len(logits) == 3, "DYNAMIC_MARGIN_LOGITS")
    wrong_ids = [
        index
        for index in range(3)
        if index != correct_label_id
    ]
    require(
        float(logits[wrong_ids[0]]) != float(logits[wrong_ids[1]]),
        "DYNAMIC_WRONG_EXACT_TIE",
    )
    wrong_id = max(
        wrong_ids,
        key=lambda index: float(logits[index]),
    )
    value = fixed_margin(
        logits,
        correct_label_id=correct_label_id,
        wrong_class_id=wrong_id,
    )
    return value, int(wrong_id)


def central_difference_metrics(
    *,
    native_margin: float,
    plus_margin: float,
    minus_margin: float,
    epsilon: float,
    frozen_delta_l: float,
) -> dict[str, Any]:
    require(epsilon in EPSILONS, f"METRIC_EPSILON:{epsilon}")
    values = (
        native_margin,
        plus_margin,
        minus_margin,
        frozen_delta_l,
    )
    require(all(math.isfinite(value) for value in values), "METRIC_FINITE")

    d_plus = native_margin - plus_margin
    d_minus = native_margin - minus_margin
    central = (minus_margin - plus_margin) / (2.0 * epsilon)
    even_second = (
        plus_margin + minus_margin - 2.0 * native_margin
    ) / (epsilon * epsilon)

    ratio = None
    plus_ratio = None
    minus_ratio = None
    if frozen_delta_l != 0.0:
        ratio = central / frozen_delta_l
        plus_ratio = d_plus / (epsilon * frozen_delta_l)
        minus_ratio = d_minus / (-epsilon * frozen_delta_l)

    return {
        "epsilon": epsilon,
        "D_plus": d_plus,
        "D_minus": d_minus,
        "central_difference_Delta_L": central,
        "central_over_frozen_Delta_L": ratio,
        "positive_one_sided_over_frozen_Delta_L": plus_ratio,
        "negative_one_sided_over_frozen_Delta_L": minus_ratio,
        "even_second_difference": even_second,
    }


def signed_full_correction(
    unscaled_full: torch.Tensor,
    signed_epsilon: float,
) -> torch.Tensor:
    require(
        signed_epsilon in SIGNED_EPSILONS,
        f"SIGNED_EPSILON:{signed_epsilon}",
    )
    require(unscaled_full.ndim == 1, "UNSCALED_FULL_SHAPE")
    out = (unscaled_full * signed_epsilon).contiguous()
    require(bool(torch.isfinite(out).all().item()), "SIGNED_NONFINITE")
    return out


def _signed_control_hook(
    mixer: Any,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    native_full: torch.Tensor,
    unscaled_delta_full: torch.Tensor,
    signed_epsilon: float,
    intermediate_size: int,
    cast_tol: float,
    audit: dict[str, Any],
):
    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == intermediate_size, "MASK_WIDTH")
    intended_cpu = signed_full_correction(
        unscaled_delta_full,
        signed_epsilon,
    )

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        before = output.detach().clone()
        current_native = (
            before[0, token_index, :intermediate_size]
            .detach().cpu().to(native_full.dtype).contiguous()
        )
        replay_residual = float(
            torch.max(
                torch.abs(
                    current_native.to(torch.float64)
                    - native_full.to(torch.float64)
                )
            ).item()
        )
        require(
            replay_residual <= cast_tol,
            f"NATIVE_REPLAY:{replay_residual}",
        )

        intended = intended_cpu.to(
            device=output.device,
            dtype=output.dtype,
        )
        out = output.clone()
        out[0, token_index, :intermediate_size] += intended

        require(
            torch.equal(
                out[:, :, intermediate_size:],
                before[:, :, intermediate_size:],
            ),
            "GATE_CHANGED",
        )
        if token_index:
            require(
                torch.equal(
                    out[:, :token_index, :],
                    before[:, :token_index, :],
                ),
                "EARLIER_TOKEN_CHANGED",
            )
        if token_index + 1 < out.shape[1]:
            require(
                torch.equal(
                    out[:, token_index + 1:, :],
                    before[:, token_index + 1:, :],
                ),
                "LATER_TOKEN_CHANGED",
            )

        nonstrong = ~mask_cpu.to(output.device)
        require(
            torch.equal(
                out[0, token_index, :intermediate_size][nonstrong],
                before[0, token_index, :intermediate_size][nonstrong],
            ),
            "NONSTRONG_CHANGED",
        )

        applied = (
            out[0, token_index, :intermediate_size]
            - before[0, token_index, :intermediate_size]
        ).detach().cpu().to(torch.float64)
        intended64 = intended.detach().cpu().to(torch.float64)
        applied_residual = float(
            torch.max(torch.abs(applied - intended64)).item()
        )
        require(
            applied_residual <= cast_tol,
            f"APPLIED_RESIDUAL:{applied_residual}",
        )

        scaling_residual = float(
            torch.max(
                torch.abs(
                    intended_cpu.to(torch.float64)
                    - (
                        unscaled_delta_full.to(torch.float64)
                        * signed_epsilon
                    )
                )
            ).item()
        )
        require(
            scaling_residual <= cast_tol,
            f"SCALING_RESIDUAL:{scaling_residual}",
        )

        audit.clear()
        audit.update({
            "signed_epsilon": signed_epsilon,
            "native_replay_max_abs_residual": replay_residual,
            "applied_correction_max_abs_residual": applied_residual,
            "scaling_identity_max_abs_residual": scaling_residual,
            "applied_correction_l2": float(
                torch.linalg.vector_norm(applied).item()
            ),
        })
        return out

    return mixer.in_proj.register_forward_hook(hook)


def serialize_logits(output: Mapping[str, Any]) -> list[float]:
    logits = output["logits"]
    require(
        torch.is_tensor(logits)
        and tuple(logits.shape) == (1, 3),
        "LOGIT_SHAPE",
    )
    values = [
        float(value)
        for value in logits.detach().cpu()[0].tolist()
    ]
    require(
        all(math.isfinite(value) for value in values),
        "LOGIT_NONFINITE",
    )
    return values


def run_signed_forward(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row_index_value: int,
    token_index: int,
    native_full: torch.Tensor,
    unscaled_delta_full: torch.Tensor,
    signed_epsilon: float,
    device: torch.device,
) -> tuple[list[float], dict[str, Any]]:
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    adapter = spec["adapter"]
    audit: dict[str, Any] = {}

    handle = _signed_control_hook(
        runtime_ctx["intervention_mixer"],
        token_index=token_index,
        strong_mask=runtime_ctx["strong_mask"],
        native_full=native_full,
        unscaled_delta_full=unscaled_delta_full,
        signed_epsilon=signed_epsilon,
        intermediate_size=int(geom.INTERMEDIATE_SIZE),
        cast_tol=float(
            confirmation.transport_runtime.RUNTIME_CAST_TOL
        ),
        audit=audit,
    )
    try:
        with torch.inference_mode():
            output = adapter.historical_forward(
                model,
                bridge.feature_batch(
                    encoded,
                    row_index_value,
                    device,
                ),
                arm=geom.ARM,
            )
    finally:
        handle.remove()

    return serialize_logits(output), dict(audit)


def run_diagnostic_row(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    device: torch.device,
    frozen_readout: Mapping[str, Any],
    subset_row_index: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    geom = spec["geom"]
    adapter = spec["adapter"]
    target_index = int(anchor_index + geom.TARGET_OFFSET)
    strong_mask = (
        runtime_ctx["strong_mask"]
        .detach().cpu().bool().contiguous()
    )

    capture: dict[str, Any] = {}
    handle = behavior_runner._native_capture_hook(
        runtime_ctx["intervention_mixer"],
        token_index=target_index,
        strong_mask=strong_mask,
        spec=spec,
        frozen=frozen,
        capture=capture,
    )
    try:
        with torch.inference_mode():
            native_output = adapter.historical_forward(
                model,
                bridge.feature_batch(
                    encoded,
                    row_index_value,
                    device,
                ),
                arm=geom.ARM,
            )
    finally:
        handle.remove()

    require(
        {
            "native_full",
            "unscaled_delta_full",
            "unscaled_correction_l2",
        } <= set(capture),
        "NATIVE_CAPTURE",
    )

    native_logits = serialize_logits(native_output)
    correct_label_id = int(frozen_readout["correct_label_id"])
    frozen_wrong_id = int(
        frozen_readout["active_wrong_class_id"]
    )
    require(
        correct_label_id
        == readout_runner.study_b.LABEL_ID_BY_CELL[
            str(row["contrast_cell_id"])
        ],
        "CORRECT_LABEL_REPLAY",
    )

    native_fixed = fixed_margin(
        native_logits,
        correct_label_id=correct_label_id,
        wrong_class_id=frozen_wrong_id,
    )
    native_dynamic, native_dynamic_wrong = dynamic_margin(
        native_logits,
        correct_label_id=correct_label_id,
    )
    require(
        native_dynamic_wrong == frozen_wrong_id,
        "NATIVE_ACTIVE_WRONG_REPLAY",
    )

    frozen_native_margin = float(
        frozen_readout["native_correct_class_margin"]
    )
    native_margin_residual = abs(
        native_fixed - frozen_native_margin
    )
    require(
        native_margin_residual <= NATIVE_MARGIN_REPLAY_TOL,
        f"NATIVE_MARGIN_REPLAY:{native_margin_residual}",
    )

    frozen_delta_l = float(frozen_readout["Delta_L_row"])
    require(math.isfinite(frozen_delta_l), "FROZEN_DELTA_L")

    epsilon_results: dict[str, Any] = {}
    all_scaling_pass = True
    all_replay_pass = True
    all_apply_pass = True

    for epsilon in EPSILONS:
        plus_logits, plus_audit = run_signed_forward(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=target_index,
            native_full=capture["native_full"],
            unscaled_delta_full=capture["unscaled_delta_full"],
            signed_epsilon=epsilon,
            device=device,
        )
        minus_logits, minus_audit = run_signed_forward(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=target_index,
            native_full=capture["native_full"],
            unscaled_delta_full=capture["unscaled_delta_full"],
            signed_epsilon=-epsilon,
            device=device,
        )

        plus_fixed = fixed_margin(
            plus_logits,
            correct_label_id=correct_label_id,
            wrong_class_id=frozen_wrong_id,
        )
        minus_fixed = fixed_margin(
            minus_logits,
            correct_label_id=correct_label_id,
            wrong_class_id=frozen_wrong_id,
        )
        plus_dynamic, plus_dynamic_wrong = dynamic_margin(
            plus_logits,
            correct_label_id=correct_label_id,
        )
        minus_dynamic, minus_dynamic_wrong = dynamic_margin(
            minus_logits,
            correct_label_id=correct_label_id,
        )

        fixed_metrics = central_difference_metrics(
            native_margin=native_fixed,
            plus_margin=plus_fixed,
            minus_margin=minus_fixed,
            epsilon=epsilon,
            frozen_delta_l=frozen_delta_l,
        )
        dynamic_metrics = central_difference_metrics(
            native_margin=native_dynamic,
            plus_margin=plus_dynamic,
            minus_margin=minus_dynamic,
            epsilon=epsilon,
            frozen_delta_l=frozen_delta_l,
        )

        epsilon_results[str(epsilon)] = {
            "epsilon": epsilon,
            "plus": {
                "final_logits": plus_logits,
                "fixed_active_wrong_margin": plus_fixed,
                "dynamic_max_wrong_margin": plus_dynamic,
                "dynamic_wrong_class_id": plus_dynamic_wrong,
                "active_wrong_matches_frozen":
                    plus_dynamic_wrong == frozen_wrong_id,
                "intervention_audit": plus_audit,
            },
            "minus": {
                "final_logits": minus_logits,
                "fixed_active_wrong_margin": minus_fixed,
                "dynamic_max_wrong_margin": minus_dynamic,
                "dynamic_wrong_class_id": minus_dynamic_wrong,
                "active_wrong_matches_frozen":
                    minus_dynamic_wrong == frozen_wrong_id,
                "intervention_audit": minus_audit,
            },
            "fixed_active_wrong_metrics": fixed_metrics,
            "dynamic_max_wrong_metrics": dynamic_metrics,
        }

        for audit in (plus_audit, minus_audit):
            all_scaling_pass = all_scaling_pass and (
                float(audit["scaling_identity_max_abs_residual"])
                <= float(
                    spec["confirmation"]
                    .transport_runtime.RUNTIME_CAST_TOL
                )
            )
            all_replay_pass = all_replay_pass and (
                float(audit["native_replay_max_abs_residual"])
                <= float(
                    spec["confirmation"]
                    .transport_runtime.RUNTIME_CAST_TOL
                )
            )
            all_apply_pass = all_apply_pass and (
                float(audit["applied_correction_max_abs_residual"])
                <= float(
                    spec["confirmation"]
                    .transport_runtime.RUNTIME_CAST_TOL
                )
            )

    item = {
        "schema_version": ROW_SCHEMA,
        "scale": str(spec["scale"]),
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "subset_row_index": int(subset_row_index),
        "frozen_scale_row_index":
            int(frozen_readout["scale_row_index"]),
        "checkpoint_sha256": str(spec["checkpoint_sha256"]),
        "selected_plane": str(spec["selected_plane"]),
        "control_plane": str(spec["control_plane"]),
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": target_index,
        "correct_label_id": correct_label_id,
        "frozen_active_wrong_class_id": frozen_wrong_id,
        "frozen_Delta_L_row": frozen_delta_l,
        "frozen_native_correct_class_margin":
            frozen_native_margin,
        "native": {
            "final_logits": native_logits,
            "fixed_active_wrong_margin": native_fixed,
            "dynamic_max_wrong_margin": native_dynamic,
            "dynamic_wrong_class_id": native_dynamic_wrong,
            "margin_replay_abs_residual":
                native_margin_residual,
        },
        "unscaled_correction_l2":
            float(capture["unscaled_correction_l2"]),
        "epsilons": list(EPSILONS),
        "epsilon_results": epsilon_results,
        "scientific_full_model_forward_count":
            FORWARDS_PER_ROW,
        "local_backward_count": 0,
        "parameter_gradient_created": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "behavioral_calibration_result_accessed": False,
        "static_factor2_result_accessed": False,
        "scientific_conclusion": None,
    }
    gate = {
        "native_active_wrong_replay_pass": True,
        "native_margin_replay_pass":
            native_margin_residual <= NATIVE_MARGIN_REPLAY_TOL,
        "signed_scaling_pass": all_scaling_pass,
        "native_replay_pass": all_replay_pass,
        "applied_correction_pass": all_apply_pass,
        "seven_forward_execution_pass": True,
        "finite_frozen_Delta_L_pass": True,
    }
    return item, gate


def _prepare_scale(
    *,
    scale: str,
    model_snapshot: Path,
    physical_device: int,
):
    return readout_runner.prepare_scale(
        scale=scale,
        model_snapshot=model_snapshot,
        physical_device=physical_device,
    )


def worker_paths(temp_dir: Path, scale: str) -> dict[str, Path]:
    return {
        "rows": temp_dir / f"{scale}_rows.jsonl",
        "meta": temp_dir / f"{scale}_meta.json",
        "gate": temp_dir / f"{scale}_gate.json",
        "error": temp_dir / f"{scale}_error.txt",
    }


def technical_gate_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
) -> None:
    paths = worker_paths(Path(temp_dir), scale)
    try:
        frozen_readout = load_frozen_readout()
        (
            spec,
            model,
            runtime_ctx,
            frozen,
            rows,
            encoded,
            events,
            lookup,
            device,
            _provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        pair = SUBSET_PAIRS[0]
        cell = TARGET_CELLS[0]
        idx = lookup[(pair, cell)]
        anchor = int(
            events[(pair, cell, "A_IDENTITY")][
                "absolute_anchor_token_index"
            ]
        )
        _item, gate = run_diagnostic_row(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            frozen=frozen,
            encoded=encoded,
            row=rows[idx],
            row_index_value=idx,
            anchor_index=anchor,
            device=device,
            frozen_readout=frozen_readout[(scale, pair, cell)],
            subset_row_index=0,
        )
        torch.cuda.synchronize(device)
        require(all(bool(value) for value in gate.values()), "TECHNICAL_GATE")

        paths["gate"].write_bytes(pretty_json_bytes({
            "scale": scale,
            **gate,
            "numeric_logits_retained": False,
            "numeric_central_difference_retained": False,
            "numeric_ratio_retained": False,
            "behavioral_calibration_result_accessed": False,
            "static_factor2_result_accessed": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def raw_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    paths = worker_paths(Path(temp_dir), scale)
    try:
        authenticate_repo(expected_head)
        frozen_readout = load_frozen_readout()
        (
            spec,
            model,
            runtime_ctx,
            frozen,
            rows,
            encoded,
            events,
            lookup,
            device,
            provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        out_rows: list[dict[str, Any]] = []
        subset_row_index = 0
        for pair in SUBSET_PAIRS:
            for cell in TARGET_CELLS:
                idx = lookup[(pair, cell)]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                item, gate = run_diagnostic_row(
                    spec=spec,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    frozen=frozen,
                    encoded=encoded,
                    row=rows[idx],
                    row_index_value=idx,
                    anchor_index=anchor,
                    device=device,
                    frozen_readout=frozen_readout[(scale, pair, cell)],
                    subset_row_index=subset_row_index,
                )
                require(
                    all(bool(value) for value in gate.values()),
                    f"ROW_GATE:{pair}:{cell}",
                )
                out_rows.append(item)
                subset_row_index += 1

        require(len(out_rows) == ROWS_PER_SCALE, "WORKER_ROW_COUNT")
        torch.cuda.synchronize(device)

        paths["rows"].write_bytes(jsonl_bytes(out_rows))
        paths["meta"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "execution_head": expected_head,
            "subset_selection_salt": SUBSET_SELECTION_SALT,
            "subset_pairs": list(SUBSET_PAIRS),
            "subset_pair_count": SUBSET_SIZE,
            "target_cells": list(TARGET_CELLS),
            "epsilons": list(EPSILONS),
            "row_count": ROWS_PER_SCALE,
            "scientific_full_model_forward_count":
                FORWARDS_PER_SCALE,
            "local_backward_count": 0,
            "physical_device": SCALE_TO_PHYSICAL_GPU[scale],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "selected_plane": spec["selected_plane"],
            "control_plane": spec["control_plane"],
            "readout_item_sha256": READOUT_ITEM_SHA256,
            "model_provenance": provenance["model"],
            "tokenizer": provenance["population"]["tokenizer"],
            "training_executed": False,
            "parameter_update_executed": False,
            "inferential_test_performed": False,
            "p_value_count_executed": 0,
            "behavioral_calibration_result_accessed": False,
            "static_factor2_result_accessed": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def run_two_processes(
    *,
    target: Any,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    temp_dir: Path,
    expected_head: str | None,
) -> None:
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "GPU_COUNT")

    ctx = mp.get_context("spawn")
    snapshots = {
        "mamba370m": mamba370m_snapshot,
        "mamba14b": mamba14b_snapshot,
    }
    processes: list[tuple[str, mp.Process]] = []

    for scale in SCALES:
        kwargs: dict[str, Any] = {
            "scale": scale,
            "model_snapshot": str(snapshots[scale]),
            "temp_dir": str(temp_dir),
        }
        if expected_head is not None:
            kwargs["expected_head"] = expected_head
        process = ctx.Process(
            target=target,
            kwargs=kwargs,
            name=f"factor2-gradient-consistency-{scale}",
        )
        process.start()
        processes.append((scale, process))

    for scale, process in processes:
        process.join()
        if process.exitcode != 0:
            paths = worker_paths(temp_dir, scale)
            detail = (
                paths["error"].read_text(encoding="utf-8")
                if paths["error"].is_file()
                else "NO_WORKER_ERROR"
            )
            raise GradientConsistencyRawError(
                f"WORKER_FAILED:{scale}:\n{detail}"
            )


def run_technical_gate(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)

    with tempfile.TemporaryDirectory(
        prefix="gen4_factor2_gradient_consistency_gate_"
    ) as tmp:
        temp_dir = Path(tmp)
        run_two_processes(
            target=technical_gate_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=None,
        )

        for scale in SCALES:
            gate = json.loads(
                worker_paths(temp_dir, scale)["gate"].read_text(
                    encoding="utf-8"
                )
            )
            for key in (
                "native_active_wrong_replay_pass",
                "native_margin_replay_pass",
                "signed_scaling_pass",
                "native_replay_pass",
                "applied_correction_pass",
                "seven_forward_execution_pass",
                "finite_frozen_Delta_L_pass",
            ):
                require(gate[key] is True, f"GATE:{scale}:{key}")
            require(
                gate["numeric_logits_retained"] is False,
                "GATE_LOGITS",
            )
            require(
                gate["numeric_central_difference_retained"] is False,
                "GATE_CD",
            )
            require(
                gate["numeric_ratio_retained"] is False,
                "GATE_RATIO",
            )

    print("RESULT=" + TECHNICAL_GATE_RESULT)
    print("MAMBA370M=PASS")
    print("MAMBA14B=PASS")
    print("SUBSET_SELECTION=SHA256_OUTCOME_INDEPENDENT")
    print("SUBSET_PAIR_COUNT=32")
    print("TECHNICAL_ROWS_PER_SCALE=1")
    print("FORWARDS_PER_TECHNICAL_ROW=7")
    print("EPSILONS=0.03125,0.015625,0.0078125")
    print("SIGNED_PERTURBATIONS=True")
    print("NUMERIC_CENTRAL_DIFFERENCE_RETAINED=False")
    print("BEHAVIORAL_CALIBRATION_RESULT_ACCESSED=False")
    print("STATIC_FACTOR2_RESULT_ACCESSED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def write_sums(root: Path, names: Sequence[str]) -> None:
    (root / CHECKSUM_FILE).write_text(
        "".join(
            f"{sha256_file(root / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(
        prefix="gen4_factor2_gradient_consistency_raw_"
    ) as tmp:
        temp_dir = Path(tmp)
        run_two_processes(
            target=raw_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=expected_head,
        )

        all_rows: list[dict[str, Any]] = []
        workers: dict[str, Any] = {}
        for scale in SCALES:
            paths = worker_paths(temp_dir, scale)
            rows = read_jsonl(paths["rows"])
            require(
                len(rows) == ROWS_PER_SCALE,
                f"ROW_COUNT:{scale}",
            )
            all_rows.extend(rows)
            workers[scale] = json.loads(
                paths["meta"].read_text(encoding="utf-8")
            )

    require(len(all_rows) == TOTAL_ROWS, "TOTAL_ROW_COUNT")
    keys = {
        (
            row["scale"],
            row["source_pair_id"],
            row["contrast_cell_id"],
        )
        for row in all_rows
    }
    expected_keys = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in SUBSET_PAIRS
        for cell in TARGET_CELLS
    }
    require(keys == expected_keys, "RAW_KEY_COVERAGE")

    output_dir.mkdir(parents=True, exist_ok=False)
    raw = jsonl_bytes(all_rows)
    (output_dir / ROW_FILE).write_bytes(raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "population": "xg1_fact_5701..xg1_fact_6000",
        "subset_selection_method":
            "lowest SHA256(salt:pair_id), then numeric-order serialization",
        "subset_selection_salt": SUBSET_SELECTION_SALT,
        "subset_pairs": list(SUBSET_PAIRS),
        "subset_pair_count": SUBSET_SIZE,
        "cells": list(TARGET_CELLS),
        "scales": list(SCALES),
        "epsilons": list(EPSILONS),
        "signed_perturbations": True,
        "row_count": TOTAL_ROWS,
        "forwards_per_row": FORWARDS_PER_ROW,
        "scientific_full_model_forward_count":
            TOTAL_FORWARD_BUDGET,
        "local_backward_count": 0,
        "frozen_autograd_readout_reused": True,
        "readout_item_sha256": READOUT_ITEM_SHA256,
        "primary_numeric_target":
            "fixed-native-active-wrong central difference vs frozen Delta_L",
        "dynamic_max_wrong_margin_secondary_only": True,
        "even_order_curvature_recorded": True,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "row_filtering_performed": False,
        "rescue_performed": False,
        "adaptive_epsilon_selection_performed": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "behavioral_calibration_result_accessed": False,
        "static_factor2_result_accessed": False,
        "scientific_conclusion": None,
        "workers": workers,
    }
    (output_dir / SUMMARY_FILE).write_bytes(
        pretty_json_bytes(summary)
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "row_sha256": sha256_bytes(raw),
        "summary_sha256":
            sha256_file(output_dir / SUMMARY_FILE),
        "readout_item_sha256": READOUT_ITEM_SHA256,
        "subset_pair_count": SUBSET_SIZE,
        "row_count": TOTAL_ROWS,
        "epsilons": list(EPSILONS),
        "scientific_full_model_forward_count":
            TOTAL_FORWARD_BUDGET,
        "local_backward_count": 0,
        "p_value_count_executed": 0,
        "row_filtering_performed": False,
        "rescue_performed": False,
        "adaptive_epsilon_selection_performed": False,
        "behavioral_calibration_result_accessed": False,
        "static_factor2_result_accessed": False,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(
        pretty_json_bytes(manifest)
    )
    write_sums(
        output_dir,
        (
            ROW_FILE,
            SUMMARY_FILE,
            MANIFEST_FILE,
        ),
    )

    print("RESULT=" + RAW_RESULT)
    print("SUBSET_SELECTION=SHA256_OUTCOME_INDEPENDENT")
    print("SUBSET_PAIR_COUNT=32")
    print("ROW_COUNT=128")
    print("EPSILONS=0.03125,0.015625,0.0078125")
    print("SIGNED_PERTURBATIONS=True")
    print("FORWARDS_PER_ROW=7")
    print("MAMBA370M_FORWARD_COUNT=448")
    print("MAMBA14B_FORWARD_COUNT=448")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT_TOTAL=896")
    print("LOCAL_BACKWARD_COUNT=0")
    print("FROZEN_AUTOGRAD_READOUT_REUSED=True")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("ROW_FILTERING_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("ADAPTIVE_EPSILON_SELECTION_PERFORMED=False")
    print("BEHAVIORAL_CALIBRATION_RESULT_ACCESSED=False")
    print("STATIC_FACTOR2_RESULT_ACCESSED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded factor-2 gradient-consistency diagnostic. "
            "On an outcome-independent 32-pair subset, runs native plus "
            "symmetric +/- epsilon forward perturbations at the exact "
            "intervention boundary and retains raw fixed-active-wrong and "
            "dynamic-margin evidence for later CPU-only analysis."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("technical-gate", "raw"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--mamba370m-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mamba14b-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    if args.mode == "technical-gate":
        require(
            args.output_dir is None,
            "TECHNICAL_GATE_OUTPUT_ARGUMENT",
        )
        run_technical_gate(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
        )
        return 0

    require(args.output_dir is not None, "RAW_OUTPUT_REQUIRED")
    run_raw(
        expected_head=args.expected_head,
        mamba370m_snapshot=args.mamba370m_snapshot,
        mamba14b_snapshot=args.mamba14b_snapshot,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
