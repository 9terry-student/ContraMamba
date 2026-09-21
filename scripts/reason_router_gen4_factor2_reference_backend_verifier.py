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
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_factor2_gradient_consistency_raw_fast_cuda
    as gradient_diag,
)
from scripts import (
    reason_router_gen4_factor2_small_alpha_native_readout_raw_fast_cuda
    as native_readout,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)
from scripts import (
    reason_router_gen4_mamba370m14b_readout_alignment_raw_fast_cuda
    as study_b,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "9dbace6a97225668a626abc0632d0a7c1c6cff69"

SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
SCALE_TO_PHYSICAL_GPU = {"mamba370m": 0, "mamba14b": 1}

ALL_PAIRS = tuple(f"xg1_fact_{i}" for i in range(5701, 6001))
SUBSET_SELECTION_SALT = "factor2-reference-backend-v1"
SUBSET_SIZE = 4
EPSILON = 0.03125

READOUT_DIR = ROOT / (
    "reports/reason_router_gen4_mamba370m14b_"
    "factor2_small_alpha_native_readout_raw_v1"
)
READOUT_ITEM_FILE = "factor2_small_alpha_native_readout_items.jsonl"
READOUT_ITEM_SHA256 = (
    "cb4e4115d7f21cc66e63153f49942eb4e3e1f6d410c1475c24666cc4f76a9fd8"
)

ROW_FILE = "reference_backend_rows.jsonl"
SUMMARY_FILE = "raw_reference_backend_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-factor2-reference-backend-row-v1"
SUMMARY_SCHEMA = "gen4-factor2-reference-backend-summary-v1"
MANIFEST_SCHEMA = "gen4-factor2-reference-backend-manifest-v1"

TECHNICAL_GATE_RESULT = "PASS_GEN4_FACTOR2_REFERENCE_BACKEND_TECHNICAL_GATE"
RAW_RESULT = "PASS_GEN4_FACTOR2_REFERENCE_BACKEND_RAW"

NATIVE_MARGIN_REPLAY_TOL = 1.0e-5
FAST_DELTA_REPLAY_RTOL = 1.0e-6
FAST_DELTA_REPLAY_ATOL = 1.0e-8


class ReferenceBackendError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReferenceBackendError(message)


def _subset_score(pair: str) -> str:
    return hashlib.sha256(
        f"{SUBSET_SELECTION_SALT}:{pair}".encode("utf-8")
    ).hexdigest()


SUBSET_PAIRS = tuple(
    sorted(
        sorted(ALL_PAIRS, key=_subset_score)[:SUBSET_SIZE],
        key=lambda p: int(p.rsplit("_", 1)[1]),
    )
)
ROWS_PER_SCALE = SUBSET_SIZE * len(CELLS)
TOTAL_ROWS = len(SCALES) * ROWS_PER_SCALE
FORWARDS_PER_ROW = 6
BACKWARDS_PER_ROW = 2
FORWARDS_PER_SCALE = ROWS_PER_SCALE * FORWARDS_PER_ROW
BACKWARDS_PER_SCALE = ROWS_PER_SCALE * BACKWARDS_PER_ROW
TOTAL_FORWARDS = len(SCALES) * FORWARDS_PER_SCALE
TOTAL_BACKWARDS = len(SCALES) * BACKWARDS_PER_SCALE


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
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
        raise ReferenceBackendError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR_MISSING")


def validate_protocol() -> None:
    require(len(ALL_PAIRS) == 300, "ALL_PAIR_COUNT")
    require(SUBSET_SIZE == 4, "SUBSET_SIZE")
    require(len(SUBSET_PAIRS) == 4, "SUBSET_COUNT")
    require(len(set(SUBSET_PAIRS)) == 4, "SUBSET_UNIQUE")
    require(set(SUBSET_PAIRS) <= set(ALL_PAIRS), "SUBSET_DOMAIN")
    require(EPSILON == 0.03125, "EPSILON")
    require(ROWS_PER_SCALE == 8, "ROWS_PER_SCALE")
    require(TOTAL_ROWS == 16, "TOTAL_ROWS")
    require(FORWARDS_PER_ROW == 6, "FORWARDS_PER_ROW")
    require(BACKWARDS_PER_ROW == 2, "BACKWARDS_PER_ROW")
    require(FORWARDS_PER_SCALE == 48, "FORWARDS_PER_SCALE")
    require(BACKWARDS_PER_SCALE == 16, "BACKWARDS_PER_SCALE")
    require(TOTAL_FORWARDS == 96, "TOTAL_FORWARDS")
    require(TOTAL_BACKWARDS == 32, "TOTAL_BACKWARDS")


def load_frozen_readout() -> dict[tuple[str, str, str], dict[str, Any]]:
    path = READOUT_DIR / READOUT_ITEM_FILE
    require(sha256_file(path) == READOUT_ITEM_SHA256, "READOUT_SHA")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    require(len(rows) == 1200, "READOUT_COUNT")
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["scale"]),
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"READOUT_DUP:{key}")
        out[key] = row
    return out


def serialize_logits(output: Mapping[str, Any]) -> list[float]:
    logits = output["logits"]
    require(torch.is_tensor(logits), "LOGITS_TENSOR")
    require(tuple(logits.shape) == (1, 3), "LOGITS_SHAPE")
    values = [float(v) for v in logits.detach().cpu()[0].tolist()]
    require(all(math.isfinite(v) for v in values), "LOGITS_FINITE")
    return values


def fixed_margin(
    logits: Sequence[float],
    *,
    correct_label_id: int,
    wrong_class_id: int,
) -> float:
    require(len(logits) == 3, "MARGIN_LOGITS")
    value = float(logits[correct_label_id]) - float(logits[wrong_class_id])
    require(math.isfinite(value), "MARGIN_FINITE")
    return value


def _fixed_margin_tensor(
    logits: torch.Tensor,
    *,
    correct_label_id: int,
    wrong_class_id: int,
) -> torch.Tensor:
    require(tuple(logits.shape) == (1, 3), "MARGIN_TENSOR_SHAPE")
    margin = logits[0, correct_label_id] - logits[0, wrong_class_id]
    require(bool(torch.isfinite(margin.detach()).item()), "MARGIN_TENSOR_FINITE")
    return margin


def _model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    backbone = getattr(model, "mamba", None)
    require(backbone is not None, "MAMBA_BACKBONE")
    layers = list(getattr(backbone, "layers", []))
    require(len(layers) == 48, f"LAYER_COUNT:{len(layers)}")
    for idx, layer in enumerate(layers):
        require(getattr(layer, "mixer", None) is not None, f"MIXER:{idx}")
    return layers


@contextmanager
def count_fast_calls(model: torch.nn.Module):
    layers = _model_layers(model)
    counter = {"count": 0}
    patched: list[torch.nn.Module] = []
    for layer in layers:
        mixer = layer.mixer
        original = mixer.cuda_kernels_forward

        def wrapper(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
            *,
            _original=original,
        ):
            counter["count"] += 1
            return _original(
                hidden_states,
                cache_params,
                cache_position,
                attention_mask,
            )

        mixer.cuda_kernels_forward = types.MethodType(wrapper, mixer)
        patched.append(mixer)
    try:
        yield counter
    finally:
        for mixer in patched:
            if "cuda_kernels_forward" in mixer.__dict__:
                del mixer.__dict__["cuda_kernels_forward"]


@contextmanager
def force_all_slow(model: torch.nn.Module):
    layers = _model_layers(model)
    counter = {"count": 0}
    patched: list[torch.nn.Module] = []
    for layer in layers:
        mixer = layer.mixer

        def wrapper(
            self,
            hidden_states,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        ):
            counter["count"] += 1
            return self.slow_forward(
                hidden_states,
                cache_params,
                cache_position,
                attention_mask,
            )

        mixer.forward = types.MethodType(wrapper, mixer)
        patched.append(mixer)
    try:
        yield counter
    finally:
        for mixer in patched:
            if "forward" in mixer.__dict__:
                del mixer.__dict__["forward"]


def gradient_at_boundary(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    device: torch.device,
    correct_label_id: int,
    wrong_class_id: int,
) -> dict[str, Any]:
    geom = spec["geom"]
    intermediate_size = int(geom.INTERMEDIATE_SIZE)
    strong_mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    capture: dict[str, Any] = {}
    handle = study_b._install_local_leaf_hook(
        runtime_ctx["intervention_mixer"],
        token_index=anchor_index + geom.TARGET_OFFSET,
        intermediate_size=intermediate_size,
        strong_mask=strong_mask,
        capture=capture,
    )
    try:
        output = spec["adapter"].historical_forward(
            model,
            bridge.feature_batch(encoded, row_index_value, device),
            arm=geom.ARM,
        )
    finally:
        handle.remove()

    require("leaf" in capture and "native_full" in capture, "LEAF_CAPTURE")
    leaf = capture["leaf"]
    margin = _fixed_margin_tensor(
        output["logits"],
        correct_label_id=correct_label_id,
        wrong_class_id=wrong_class_id,
    )
    grad = torch.autograd.grad(
        margin,
        leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    require(tuple(grad.shape) == (intermediate_size,), "GRAD_SHAPE")
    require(bool(torch.isfinite(grad).all().item()), "GRAD_FINITE")
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "PARAM_GRAD_CREATED",
    )
    return {
        "logits": serialize_logits(output),
        "margin": float(margin.detach().cpu().item()),
        "grad_full": grad.detach().cpu().to(torch.float64).contiguous(),
        "native_full": capture["native_full"].detach().cpu().contiguous(),
    }


def project_delta_l(
    grad_full: torch.Tensor,
    delta_control_full: torch.Tensor,
) -> float:
    g = grad_full.detach().cpu().to(torch.float64).contiguous()
    d = delta_control_full.detach().cpu().to(torch.float64).contiguous()
    require(g.shape == d.shape and g.ndim == 1, "PROJECTION_SHAPE")
    value = float(-torch.dot(g, d).item())
    require(math.isfinite(value), "PROJECTION_FINITE")
    return value


def signed_forward_margin(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row_index_value: int,
    token_index: int,
    native_full: torch.Tensor,
    delta_control_full: torch.Tensor,
    signed_epsilon: float,
    device: torch.device,
    correct_label_id: int,
    wrong_class_id: int,
) -> dict[str, Any]:
    logits, audit = gradient_diag.run_signed_forward(
        spec=spec,
        model=model,
        runtime_ctx=runtime_ctx,
        encoded=encoded,
        row_index_value=row_index_value,
        token_index=token_index,
        native_full=native_full,
        unscaled_delta_full=delta_control_full,
        signed_epsilon=signed_epsilon,
        device=device,
    )
    margin = fixed_margin(
        logits,
        correct_label_id=correct_label_id,
        wrong_class_id=wrong_class_id,
    )
    return {"logits": logits, "margin": margin, "audit": audit}


def central_from_margins(
    plus_margin: float,
    minus_margin: float,
) -> float:
    value = (minus_margin - plus_margin) / (2.0 * EPSILON)
    require(math.isfinite(value), "CENTRAL_FINITE")
    return value


def run_reference_row(
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
) -> dict[str, Any]:
    geom = spec["geom"]
    correct = int(frozen_readout["correct_label_id"])
    wrong = int(frozen_readout["active_wrong_class_id"])
    frozen_delta_l = float(frozen_readout["Delta_L_row"])
    frozen_native_margin = float(frozen_readout["native_correct_class_margin"])
    token_index = int(anchor_index + geom.TARGET_OFFSET)

    with count_fast_calls(model) as fast_counter:
        fast_item, fast_arrays, _ = study_b.run_native_gradient_row(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            frozen=frozen,
            encoded=encoded,
            row=row,
            row_index_value=row_index_value,
            anchor_index=anchor_index,
            device=device,
        )
        require(int(fast_item["correct_label_id"]) == correct, "FAST_LABEL")
        require(int(fast_item["active_wrong_class_id"]) == wrong, "FAST_WRONG")
        fast_delta_l = float(fast_item["Delta_L_row"])
        require(
            math.isclose(
                fast_delta_l,
                frozen_delta_l,
                rel_tol=FAST_DELTA_REPLAY_RTOL,
                abs_tol=FAST_DELTA_REPLAY_ATOL,
            ),
            f"FAST_DELTA_REPLAY:{fast_delta_l}:{frozen_delta_l}",
        )
        fast_native_margin = float(fast_item["native_correct_class_margin"])
        require(
            abs(fast_native_margin - frozen_native_margin)
            <= NATIVE_MARGIN_REPLAY_TOL,
            "FAST_MARGIN_REPLAY",
        )
        fast_native_full = torch.from_numpy(
            fast_arrays["h_native_full"]
        ).contiguous()
        fast_delta_full = torch.from_numpy(
            fast_arrays["delta_control_full"]
        ).contiguous()
        fast_plus = signed_forward_margin(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=token_index,
            native_full=fast_native_full,
            delta_control_full=fast_delta_full,
            signed_epsilon=EPSILON,
            device=device,
            correct_label_id=correct,
            wrong_class_id=wrong,
        )
        fast_minus = signed_forward_margin(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=token_index,
            native_full=fast_native_full,
            delta_control_full=fast_delta_full,
            signed_epsilon=-EPSILON,
            device=device,
            correct_label_id=correct,
            wrong_class_id=wrong,
        )
    expected_fast_calls = 3 * 48
    require(
        fast_counter["count"] == expected_fast_calls,
        f"FAST_CALL_COUNT:{fast_counter['count']}",
    )
    fast_cd = central_from_margins(
        fast_plus["margin"],
        fast_minus["margin"],
    )

    with force_all_slow(model) as slow_counter:
        slow_native = gradient_at_boundary(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row=row,
            row_index_value=row_index_value,
            anchor_index=anchor_index,
            device=device,
            correct_label_id=correct,
            wrong_class_id=wrong,
        )
        slow_delta_l = project_delta_l(
            slow_native["grad_full"],
            fast_delta_full,
        )
        slow_plus = signed_forward_margin(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=token_index,
            native_full=slow_native["native_full"],
            delta_control_full=fast_delta_full,
            signed_epsilon=EPSILON,
            device=device,
            correct_label_id=correct,
            wrong_class_id=wrong,
        )
        slow_minus = signed_forward_margin(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            encoded=encoded,
            row_index_value=row_index_value,
            token_index=token_index,
            native_full=slow_native["native_full"],
            delta_control_full=fast_delta_full,
            signed_epsilon=-EPSILON,
            device=device,
            correct_label_id=correct,
            wrong_class_id=wrong,
        )
    expected_slow_calls = 3 * 48
    require(
        slow_counter["count"] == expected_slow_calls,
        f"SLOW_CALL_COUNT:{slow_counter['count']}",
    )
    slow_cd = central_from_margins(
        slow_plus["margin"],
        slow_minus["margin"],
    )

    # The fast native scalar margin is replayed against the frozen readout.
    # Plus/minus logits are retained for local forward-equivalence checks.
    slow_native_margin = float(slow_native["margin"])

    fast_slow_plus_logit_max_abs = max(
        abs(a - b)
        for a, b in zip(fast_plus["logits"], slow_plus["logits"])
    )
    fast_slow_minus_logit_max_abs = max(
        abs(a - b)
        for a, b in zip(fast_minus["logits"], slow_minus["logits"])
    )

    return {
        "schema_version": ROW_SCHEMA,
        "scale": str(spec["scale"]),
        "source_pair_id": str(row["source_pair_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "epsilon": EPSILON,
        "correct_label_id": correct,
        "frozen_active_wrong_class_id": wrong,
        "frozen_Delta_L_row": frozen_delta_l,
        "fast_recomputed_Delta_L_row": fast_delta_l,
        "slow_reference_Delta_L_on_fast_direction": slow_delta_l,
        "fast_central_difference_on_fast_direction": fast_cd,
        "slow_central_difference_on_fast_direction": slow_cd,
        "frozen_native_margin": frozen_native_margin,
        "fast_native_margin": fast_native_margin,
        "slow_native_margin": slow_native_margin,
        "fast_slow_native_margin_abs_diff":
            abs(fast_native_margin - slow_native_margin),
        "fast_plus_margin": fast_plus["margin"],
        "fast_minus_margin": fast_minus["margin"],
        "slow_plus_margin": slow_plus["margin"],
        "slow_minus_margin": slow_minus["margin"],
        "fast_slow_plus_logit_max_abs_diff":
            fast_slow_plus_logit_max_abs,
        "fast_slow_minus_logit_max_abs_diff":
            fast_slow_minus_logit_max_abs,
        "fast_path_mixer_call_count": expected_fast_calls,
        "slow_path_mixer_call_count": expected_slow_calls,
        "fast_forward_count": 3,
        "slow_forward_count": 3,
        "local_backward_count": 2,
        "parameter_gradient_created": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "adaptive_backend_selection_performed": False,
        "scientific_conclusion": None,
    }


def worker_paths(temp_dir: Path, scale: str) -> dict[str, Path]:
    return {
        "rows": temp_dir / f"{scale}_rows.jsonl",
        "meta": temp_dir / f"{scale}_meta.json",
        "gate": temp_dir / f"{scale}_gate.json",
        "error": temp_dir / f"{scale}_error.txt",
    }


def _prepare_scale(
    *,
    scale: str,
    model_snapshot: Path,
    physical_device: int,
):
    return native_readout.prepare_scale(
        scale=scale,
        model_snapshot=model_snapshot,
        physical_device=physical_device,
    )


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
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, _provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )
        pair = SUBSET_PAIRS[0]
        cell = CELLS[0]
        idx = lookup[(pair, cell)]
        anchor = int(
            events[(pair, cell, "A_IDENTITY")]["absolute_anchor_token_index"]
        )
        item = run_reference_row(
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
        )
        require(
            all(
                math.isfinite(float(item[name]))
                for name in (
                    "frozen_Delta_L_row",
                    "fast_recomputed_Delta_L_row",
                    "slow_reference_Delta_L_on_fast_direction",
                    "fast_central_difference_on_fast_direction",
                    "slow_central_difference_on_fast_direction",
                    "fast_slow_native_margin_abs_diff",
                    "fast_slow_plus_logit_max_abs_diff",
                    "fast_slow_minus_logit_max_abs_diff",
                )
            ),
            "GATE_NONFINITE",
        )
        paths["gate"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "fast_path_verified": True,
            "slow_path_verified": True,
            "frozen_fast_delta_replay_verified": True,
            "same_fast_direction_used_for_reference": True,
            "numeric_derivatives_retained": False,
            "numeric_forward_differences_retained": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
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
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, provenance,
        ) = _prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )
        out: list[dict[str, Any]] = []
        for pair in SUBSET_PAIRS:
            for cell in CELLS:
                idx = lookup[(pair, cell)]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                out.append(
                    run_reference_row(
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
                    )
                )
        require(len(out) == ROWS_PER_SCALE, "WORKER_ROWS")
        torch.cuda.synchronize(device)
        paths["rows"].write_bytes(jsonl_bytes(out))
        paths["meta"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "execution_head": expected_head,
            "subset_selection_salt": SUBSET_SELECTION_SALT,
            "subset_pairs": list(SUBSET_PAIRS),
            "subset_pair_count": SUBSET_SIZE,
            "cells": list(CELLS),
            "epsilon": EPSILON,
            "row_count": ROWS_PER_SCALE,
            "forward_count": FORWARDS_PER_SCALE,
            "local_backward_count": BACKWARDS_PER_SCALE,
            "fast_backend": "Transformers-5.0.0 cuda_kernels_forward",
            "reference_backend": "Transformers-5.0.0 MambaMixer.slow_forward",
            "reference_backend_forced_on_all_48_mixers": True,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "readout_item_sha256": READOUT_ITEM_SHA256,
            "model_provenance": provenance["model"],
            "training_executed": False,
            "parameter_update_executed": False,
            "p_value_count_executed": 0,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(traceback.format_exc(), encoding="utf-8")
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
        p = ctx.Process(target=target, kwargs=kwargs, name=f"ref-backend-{scale}")
        p.start()
        processes.append((scale, p))
    for scale, p in processes:
        p.join()
        if p.exitcode != 0:
            detail = worker_paths(temp_dir, scale)["error"]
            text = detail.read_text(encoding="utf-8") if detail.is_file() else "NO_WORKER_ERROR"
            raise ReferenceBackendError(f"WORKER_FAILED:{scale}:\n{text}")


def run_technical_gate(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    with tempfile.TemporaryDirectory(prefix="factor2_ref_backend_gate_") as tmp:
        td = Path(tmp)
        run_two_processes(
            target=technical_gate_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=td,
            expected_head=None,
        )
        for scale in SCALES:
            gate = json.loads(worker_paths(td, scale)["gate"].read_text(encoding="utf-8"))
            for key in (
                "fast_path_verified",
                "slow_path_verified",
                "frozen_fast_delta_replay_verified",
                "same_fast_direction_used_for_reference",
            ):
                require(gate[key] is True, f"GATE:{scale}:{key}")
            require(gate["numeric_derivatives_retained"] is False, "GATE_DERIV")
            require(gate["numeric_forward_differences_retained"] is False, "GATE_FORWARD_DIFF")
    print("RESULT=" + TECHNICAL_GATE_RESULT)
    print("MAMBA370M=PASS")
    print("MAMBA14B=PASS")
    print("SUBSET_SELECTION=SHA256_OUTCOME_INDEPENDENT")
    print("SUBSET_PAIR_COUNT=4")
    print("EPSILON=0.03125")
    print("FAST_BACKEND=cuda_kernels_forward")
    print("REFERENCE_BACKEND=MambaMixer.slow_forward")
    print("REFERENCE_BACKEND_FORCED_ALL_48_MIXERS=True")
    print("NUMERIC_DERIVATIVES_RETAINED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def write_sums(root: Path, names: Sequence[str]) -> None:
    (root / SUMS_FILE).write_text(
        "".join(f"{sha256_file(root / n)}  {n}\n" for n in sorted(names)),
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
    with tempfile.TemporaryDirectory(prefix="factor2_ref_backend_raw_") as tmp:
        td = Path(tmp)
        run_two_processes(
            target=raw_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=td,
            expected_head=expected_head,
        )
        rows: list[dict[str, Any]] = []
        workers: dict[str, Any] = {}
        for scale in SCALES:
            rows.extend(
                json.loads(line)
                for line in worker_paths(td, scale)["rows"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
            workers[scale] = json.loads(
                worker_paths(td, scale)["meta"].read_text(encoding="utf-8")
            )
    require(len(rows) == TOTAL_ROWS, "TOTAL_ROWS")
    keys = {
        (row["scale"], row["source_pair_id"], row["contrast_cell_id"])
        for row in rows
    }
    expected = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in SUBSET_PAIRS
        for cell in CELLS
    }
    require(keys == expected, "KEY_COVERAGE")

    output_dir.mkdir(parents=True, exist_ok=False)
    raw = jsonl_bytes(rows)
    (output_dir / ROW_FILE).write_bytes(raw)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "population": "xg1_fact_5701..xg1_fact_6000",
        "subset_selection_method": "lowest SHA256(salt:pair_id), then numeric-order serialization",
        "subset_selection_salt": SUBSET_SELECTION_SALT,
        "subset_pairs": list(SUBSET_PAIRS),
        "subset_pair_count": SUBSET_SIZE,
        "cells": list(CELLS),
        "scales": list(SCALES),
        "epsilon": EPSILON,
        "row_count": TOTAL_ROWS,
        "forward_count": TOTAL_FORWARDS,
        "local_backward_count": TOTAL_BACKWARDS,
        "fast_backend": "Transformers-5.0.0 cuda_kernels_forward",
        "reference_backend": "Transformers-5.0.0 MambaMixer.slow_forward",
        "reference_backend_forced_on_all_48_mixers": True,
        "same_fast_derived_direction_used_for_both_backends": True,
        "frozen_readout_item_sha256": READOUT_ITEM_SHA256,
        "adaptive_backend_selection_performed": False,
        "adaptive_epsilon_selection_performed": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
        "workers": workers,
    }
    (output_dir / SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "row_sha256": sha256_bytes(raw),
        "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
        "subset_pair_count": SUBSET_SIZE,
        "row_count": TOTAL_ROWS,
        "epsilon": EPSILON,
        "forward_count": TOTAL_FORWARDS,
        "local_backward_count": TOTAL_BACKWARDS,
        "reference_backend_forced_on_all_48_mixers": True,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    write_sums(output_dir, (ROW_FILE, SUMMARY_FILE, MANIFEST_FILE))

    print("RESULT=" + RAW_RESULT)
    print("SUBSET_SELECTION=SHA256_OUTCOME_INDEPENDENT")
    print("SUBSET_PAIR_COUNT=4")
    print("ROW_COUNT=16")
    print("EPSILON=0.03125")
    print("TOTAL_FORWARD_COUNT=96")
    print("TOTAL_LOCAL_BACKWARD_COUNT=32")
    print("FAST_BACKEND=cuda_kernels_forward")
    print("REFERENCE_BACKEND=MambaMixer.slow_forward")
    print("REFERENCE_BACKEND_FORCED_ALL_48_MIXERS=True")
    print("SAME_FAST_DERIVED_DIRECTION_USED_FOR_BOTH_BACKENDS=True")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("SCIENTIFIC_CONCLUSION=None")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent reference-backend verifier for the factor-2 gradient "
            "mismatch. Compares Transformers 5.0.0 CUDA-kernel execution with "
            "the sequential MambaMixer.slow_forward path on the same weights, "
            "same scalar margin, and same fast-derived intervention direction."
        )
    )
    parser.add_argument("--mode", choices=("technical-gate", "raw"), required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mamba370m-snapshot", type=Path, required=True)
    parser.add_argument("--mamba14b-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "technical-gate":
        require(args.output_dir is None, "GATE_OUTPUT_ARG")
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
