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
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "c337b8c894b63efc6f4ae9e8c014eaaef3362df3"

SCALE_ORDER = ("mamba370m", "mamba14b")
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_control",
)
TARGET_CELLS = bridge.TARGET_CELLS
PAIR_IDS = bridge.PAIR_IDS
PAIR_COUNT = bridge.PAIR_COUNT
SHARDS = bridge.SHARDS

INTERVENTION_BLOCK = 35
FINAL_BLOCK = 47
BLOCK_STAGE_NAMES = (
    ("pre_block_35",)
    + tuple(f"post_block_{i}" for i in range(INTERVENTION_BLOCK, FINAL_BLOCK + 1))
)
FINAL_NORM_STAGE = "post_final_norm"
STAGE_ORDER = BLOCK_STAGE_NAMES + (FINAL_NORM_STAGE,)

FULL_FORWARDS_PER_PAIR = len(TARGET_CELLS) * len(CONDITIONS)
FULL_FORWARDS_PER_SHARD = 150 * FULL_FORWARDS_PER_PAIR
FULL_FORWARDS_PER_SCALE = PAIR_COUNT * FULL_FORWARDS_PER_PAIR

DOWNSTREAM_REPLAYS_PER_FULL_FORWARD = 2
DOWNSTREAM_REPLAYS_PER_SHARD = (
    FULL_FORWARDS_PER_SHARD * DOWNSTREAM_REPLAYS_PER_FULL_FORWARD
)
DOWNSTREAM_REPLAYS_PER_SCALE = (
    FULL_FORWARDS_PER_SCALE * DOWNSTREAM_REPLAYS_PER_FULL_FORWARD
)

ROW_FILE = "stagewise_localization_rows.jsonl"
SUMMARY_FILE = "shard_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ROW_SCHEMA = "gen4-mamba370m14b-stagewise-coupling-localization-row-v1"
SUMMARY_SCHEMA = "gen4-mamba370m14b-stagewise-coupling-localization-summary-v1"
RESULT_PASS = "PASS_STAGEWISE_COUPLING_LOCALIZATION_RAW"

BASELINE_RUN_ROOT = Path(
    "reports/reason_router_gen4_mamba370m14b_behavioral_bridge_runs/"
    "g4k-mamba370m14b-behavioral-bridge-xg1-4801-5100-2gpu-2b41f28-retry1"
)
BASELINE_ROW_FILE = bridge.ROW_FILE
BASELINE_SUMMARY_FILE = bridge.SUMMARY_FILE
BASELINE_CHECKSUM_FILE = bridge.CHECKSUM_FILE

REPRO_TOL = 1.0e-6
LENS_TOL = 2.0e-6
PREFIX_TOL = 2.0e-6

PRIMITIVE_KEYS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "positive_energy",
    "negative_energy",
    "q_authorized",
    "entitlement_prob",
)


class LocalizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LocalizationError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL_MISSING:{path}")
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalizationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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
    require(rc == 0, "REQUIRED_ANCESTOR")


def validate_protocol() -> None:
    require(SCALE_ORDER == ("mamba370m", "mamba14b"), "SCALE_ORDER")
    require(CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_control",
    ), "CONDITIONS")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "CELLS")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(INTERVENTION_BLOCK == 35 and FINAL_BLOCK == 47, "BLOCK_RANGE")
    require(len(BLOCK_STAGE_NAMES) == 14, "BLOCK_STAGE_COUNT")
    require(len(STAGE_ORDER) == 15, "STAGE_COUNT")
    require(FULL_FORWARDS_PER_PAIR == 6, "FWD_PER_PAIR")
    require(FULL_FORWARDS_PER_SHARD == 900, "FWD_PER_SHARD")
    require(FULL_FORWARDS_PER_SCALE == 1800, "FWD_PER_SCALE")
    require(DOWNSTREAM_REPLAYS_PER_SHARD == 1800, "REPLAY_PER_SHARD")
    require(DOWNSTREAM_REPLAYS_PER_SCALE == 3600, "REPLAY_PER_SCALE")


def validate_checksums(shard_dir: Path) -> None:
    sums = shard_dir / BASELINE_CHECKSUM_FILE
    require(sums.is_file(), f"BASELINE_SUMS:{shard_dir}")
    seen: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        target = shard_dir / name
        require(target.is_file(), f"BASELINE_TARGET:{name}")
        require(sha256_file(target) == digest, f"BASELINE_SHA:{name}")
        seen.add(name)
    require(
        seen == {BASELINE_ROW_FILE, BASELINE_SUMMARY_FILE},
        f"BASELINE_SUM_SET:{seen}",
    )


def load_frozen_baseline(scale: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    spec = bridge.scale_spec(scale)
    scale_root = ROOT / BASELINE_RUN_ROOT / scale
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for shard_id, shard in enumerate(SHARDS):
        shard_dir = scale_root / f"shard{shard_id}"
        validate_checksums(shard_dir)
        summary = json.loads(
            (shard_dir / BASELINE_SUMMARY_FILE).read_text(encoding="utf-8")
        )
        require(summary["result"] == bridge.RESULT_PASS, "BASELINE_RESULT")
        require(summary["scale"] == scale, "BASELINE_SCALE")
        require(
            summary["checkpoint_sha256"] == spec["checkpoint_sha256"],
            "BASELINE_CHECKPOINT",
        )
        require(
            summary["selected_dominant_candidate"] == spec["selected_plane"],
            "BASELINE_SELECTED",
        )
        require(
            summary["response_blind_control_plane"] == spec["control_plane"],
            "BASELINE_CONTROL",
        )
        require(summary["selection_reopened"] is False, "BASELINE_SELECTION")
        require(summary["rescue_performed"] is False, "BASELINE_RESCUE")

        for row in read_jsonl(shard_dir / BASELINE_ROW_FILE):
            condition = str(row["condition"])
            if condition not in CONDITIONS:
                continue
            key = (
                str(row["source_pair_id"]),
                str(row["contrast_cell_id"]),
                condition,
            )
            require(key not in out, f"BASELINE_DUP:{key}")
            out[key] = row

    expected = {
        (pair, cell, condition)
        for pair in PAIR_IDS
        for cell in TARGET_CELLS
        for condition in CONDITIONS
    }
    require(set(out) == expected, "BASELINE_COVERAGE")
    return out


def _hook_tensor(output: Any, label: str) -> torch.Tensor:
    value = output
    if isinstance(value, tuple):
        require(bool(value), f"{label}:EMPTY_TUPLE")
        value = value[0]
    require(torch.is_tensor(value), f"{label}:NOT_TENSOR")
    require(value.ndim == 3 and value.shape[0] == 1, f"{label}:SHAPE")
    require(bool(torch.isfinite(value).all().item()), f"{label}:NONFINITE")
    return value


def install_capture_hooks(model: torch.nn.Module, captures: dict[str, torch.Tensor]):
    backbone = getattr(model, "mamba", None)
    require(backbone is not None, "CAPTURE_BACKBONE")
    layers = getattr(backbone, "layers", None)
    require(layers is not None and len(layers) == 48, "CAPTURE_LAYERS")
    norm_f = getattr(backbone, "norm_f", None)
    require(norm_f is not None, "CAPTURE_NORM_F")

    handles = []

    def pre_hook(_module, args):
        require(bool(args), "PRE_BLOCK_ARGS")
        tensor = _hook_tensor(args[0], "pre_block_35")
        captures["pre_block_35"] = tensor.detach().clone()

    handles.append(
        layers[INTERVENTION_BLOCK].register_forward_pre_hook(pre_hook)
    )

    for index in range(INTERVENTION_BLOCK, FINAL_BLOCK + 1):
        name = f"post_block_{index}"

        def make_hook(stage_name: str):
            def hook(_module, _args, output):
                tensor = _hook_tensor(output, stage_name)
                captures[stage_name] = tensor.detach().clone()
                return None
            return hook

        handles.append(
            layers[index].register_forward_hook(make_hook(name))
        )

    def norm_hook(_module, _args, output):
        tensor = _hook_tensor(output, FINAL_NORM_STAGE)
        captures[FINAL_NORM_STAGE] = tensor.detach().clone()
        return None

    handles.append(norm_f.register_forward_hook(norm_hook))
    return handles


def remove_handles(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def metrics_from_output(
    output: Mapping[str, Any],
    *,
    label_id: int,
    batch_index: int = 0,
) -> dict[str, Any]:
    logits = output["logits"]
    require(torch.is_tensor(logits) and logits.ndim == 2, "METRIC_LOGITS")
    require(0 <= batch_index < logits.shape[0], "METRIC_BATCH_INDEX")
    values = [
        float(x)
        for x in logits.detach().cpu()[batch_index].tolist()
    ]
    require(len(values) == 3 and all(math.isfinite(x) for x in values), "METRIC_LOGIT_VALUES")
    prediction_id = max(range(3), key=lambda i: values[i])
    result: dict[str, Any] = {
        "final_logits": values,
        "prediction_id": prediction_id,
        "correct_class_logit_margin": bridge.correct_margin(values, label_id),
    }
    for key in PRIMITIVE_KEYS:
        value = output[key]
        require(torch.is_tensor(value) and value.ndim == 1, f"METRIC:{key}")
        scalar = float(value.detach().cpu()[batch_index].item())
        require(math.isfinite(scalar), f"METRIC_FINITE:{key}")
        result[key] = scalar
    return result


def feature_for_replay(
    feature: Mapping[str, torch.Tensor],
    *,
    batch: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("attention_mask", "claim_mask", "evidence_mask"):
        tensor = feature[key]
        require(tuple(tensor.shape)[0] == 1, f"REPLAY_FEATURE:{key}")
        out[key] = tensor.expand(batch, -1).contiguous()
    return out


def downstream_replay(
    *,
    model: torch.nn.Module,
    adapter: Any,
    arm: str,
    feature: Mapping[str, torch.Tensor],
    hidden_states: torch.Tensor,
) -> Mapping[str, Any]:
    require(hidden_states.ndim == 3, "REPLAY_HIDDEN_SHAPE")
    batch = int(hidden_states.shape[0])
    masks = feature_for_replay(feature, batch=batch)
    edge_map = adapter.expected_edge_gradient_lambdas(arm)
    with torch.inference_mode():
        return model(
            input_ids=None,
            attention_mask=masks["attention_mask"],
            claim_mask=masks["claim_mask"],
            evidence_mask=masks["evidence_mask"],
            encoder_hidden_states=hidden_states,
            decision_mode=adapter.DECISION_MODE,
            gradient_ownership_mode=adapter.GRADIENT_OWNERSHIP_MODE,
            edge_gradient_lambdas=edge_map,
            return_q_diagnostics=True,
        )


def stage_lens(
    *,
    model: torch.nn.Module,
    adapter: Any,
    arm: str,
    feature: Mapping[str, torch.Tensor],
    captures: Mapping[str, torch.Tensor],
    label_id: int,
) -> dict[str, dict[str, Any]]:
    require(tuple(captures) == STAGE_ORDER, f"CAPTURE_ORDER:{tuple(captures)}")

    block_names = BLOCK_STAGE_NAMES
    block_batch = torch.cat(
        [captures[name] for name in block_names],
        dim=0,
    )
    block_output = downstream_replay(
        model=model,
        adapter=adapter,
        arm=arm,
        feature=feature,
        hidden_states=block_batch,
    )
    result = {
        name: metrics_from_output(
            block_output,
            label_id=label_id,
            batch_index=index,
        )
        for index, name in enumerate(block_names)
    }

    final_output = downstream_replay(
        model=model,
        adapter=adapter,
        arm=arm,
        feature=feature,
        hidden_states=captures[FINAL_NORM_STAGE],
    )
    result[FINAL_NORM_STAGE] = metrics_from_output(
        final_output,
        label_id=label_id,
        batch_index=0,
    )
    require(tuple(result) == STAGE_ORDER, "LENS_ORDER")
    return result


def max_metric_error(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    errors = [
        abs(float(x) - float(y))
        for x, y in zip(a["final_logits"], b["final_logits"], strict=True)
    ]
    errors.append(
        abs(
            float(a["correct_class_logit_margin"])
            - float(b["correct_class_logit_margin"])
        )
    )
    for key in PRIMITIVE_KEYS:
        if key in b:
            errors.append(abs(float(a[key]) - float(b[key])))
    return max(errors)


def baseline_error(actual: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    errors = [
        abs(float(x) - float(y))
        for x, y in zip(
            actual["final_logits"],
            baseline["final_logits"],
            strict=True,
        )
    ]
    errors.append(
        abs(
            float(actual["correct_class_logit_margin"])
            - float(baseline["correct_class_logit_margin"])
        )
    )
    errors.append(
        abs(float(actual["q_authorized"]) - float(baseline["q_authorized"]))
    )
    errors.append(
        abs(float(actual["entitlement_prob"]) - float(baseline["entitlement_prob"]))
    )
    return max(errors)


def run_condition_with_capture(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    condition: str,
    device: torch.device,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    adapter = spec["adapter"]
    cell = str(row["contrast_cell_id"])
    label_id = bridge.LABEL_ID_BY_CELL[cell]
    feature = bridge.feature_batch(encoded, row_index_value, device)

    audit: dict[str, Any] | None = None
    intervention_handle = None
    if condition != "native":
        audit = {}
        intervention_handle = bridge.install_behavior_hook(
            runtime_ctx["intervention_mixer"],
            token_index=anchor_index + geom.TARGET_OFFSET,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=frozen["planes"],
            selected_plane=str(spec["selected_plane"]),
            control_plane=str(spec["control_plane"]),
            dim=int(spec["dim"]),
            intermediate_size=int(geom.INTERMEDIATE_SIZE),
            tol=float(confirmation.TOL),
            cast_tol=float(confirmation.transport_runtime.RUNTIME_CAST_TOL),
            audit=audit,
        )

    captures: dict[str, torch.Tensor] = {}
    capture_handles = install_capture_hooks(model, captures)
    try:
        with torch.inference_mode():
            output = adapter.historical_forward(
                model,
                feature,
                arm=geom.ARM,
            )
    finally:
        remove_handles(capture_handles)
        if intervention_handle is not None:
            intervention_handle.remove()

    require(tuple(captures) == STAGE_ORDER, f"CAPTURE_COMPLETE:{tuple(captures)}")
    actual = metrics_from_output(output, label_id=label_id)
    lens = stage_lens(
        model=model,
        adapter=adapter,
        arm=geom.ARM,
        feature=feature,
        captures=captures,
        label_id=label_id,
    )

    lens_error = max_metric_error(actual, lens[FINAL_NORM_STAGE])
    require(lens_error <= LENS_TOL, f"FINAL_NORM_LENS_REPRO:{lens_error}")

    base_error = baseline_error(actual, baseline)
    require(base_error <= REPRO_TOL, f"FROZEN_BEHAVIOR_REPRO:{base_error}")

    return {
        "condition": condition,
        "actual_final": actual,
        "stage_lens": lens,
        "intervention_audit": audit,
        "baseline_reproduction_max_abs_error": base_error,
        "final_norm_lens_max_abs_error": lens_error,
        "_captures": captures,
        "_feature": feature,
    }


def propagation_stats(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    target_index: int,
    attended_length: int,
) -> dict[str, float]:
    require(a.shape == b.shape and a.ndim == 3 and a.shape[0] == 1, "PROP_SHAPE")
    require(0 <= target_index < attended_length <= a.shape[1], "PROP_INDEX")
    delta = (a - b).detach()
    require(bool(torch.isfinite(delta).all().item()), "PROP_NONFINITE")

    full = delta[0, :attended_length]
    target = delta[0, target_index]
    suffix = delta[0, target_index:attended_length]
    prefix = delta[0, :target_index]

    prefix_max = (
        float(torch.max(torch.abs(prefix)).item())
        if prefix.numel()
        else 0.0
    )
    return {
        "attended_sequence_l2":
            float(torch.linalg.vector_norm(full).item()),
        "target_token_l2":
            float(torch.linalg.vector_norm(target).item()),
        "attended_suffix_l2":
            float(torch.linalg.vector_norm(suffix).item()),
        "pre_target_max_abs": prefix_max,
    }


def hidden_propagation(
    condition_results: Mapping[str, Mapping[str, Any]],
    *,
    target_index: int,
) -> dict[str, Any]:
    native = condition_results["native"]
    neutral = condition_results["dominant_neutralized"]
    control = condition_results["dominant_control"]

    feature = native["_feature"]
    attended_length = int(feature["attention_mask"][0].sum().item())
    require(
        attended_length
        == int(neutral["_feature"]["attention_mask"][0].sum().item())
        == int(control["_feature"]["attention_mask"][0].sum().item()),
        "ATTENDED_LENGTH_DRIFT",
    )

    out: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        n = native["_captures"][stage]
        z = neutral["_captures"][stage]
        c = control["_captures"][stage]
        selected = propagation_stats(
            n, z,
            target_index=target_index,
            attended_length=attended_length,
        )
        ctrl = propagation_stats(
            c, z,
            target_index=target_index,
            attended_length=attended_length,
        )
        contrast = propagation_stats(
            n, c,
            target_index=target_index,
            attended_length=attended_length,
        )
        max_prefix = max(
            selected["pre_target_max_abs"],
            ctrl["pre_target_max_abs"],
            contrast["pre_target_max_abs"],
        )
        require(max_prefix <= PREFIX_TOL, f"CAUSAL_PREFIX_DRIFT:{stage}:{max_prefix}")
        out[stage] = {
            "selected_native_minus_neutralized": selected,
            "control_minus_neutralized": ctrl,
            "native_minus_control": contrast,
        }
    return out


def public_condition_result(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("_")
    }


def _worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "rows": temp_dir / f"shard_{shard_id}_rows.jsonl",
        "summary": temp_dir / f"shard_{shard_id}_summary.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    scale: str,
    shard: Mapping[str, Any],
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = _worker_paths(Path(temp_dir), shard_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        spec = bridge.scale_spec(scale)
        confirmation = spec["confirmation"]
        geom = spec["geom"]
        device = confirmation.runtime_gate_single_visible_gpu(physical_device)

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)
        geom.validate_snapshot(snapshot)
        require(
            checkpoint.resolve() == (ROOT / spec["checkpoint_rel"]).resolve(),
            "CHECKPOINT_PATH",
        )
        require(
            sha256_file(checkpoint) == spec["checkpoint_sha256"],
            "CHECKPOINT_SHA",
        )

        confirmation.load_frozen_selection()
        frozen = confirmation.load_frozen_geometry()
        rows, encoded, events, population_provenance = bridge.build_input_state(
            spec=spec,
            snapshot=snapshot,
        )
        lookup = bridge.row_index(rows)
        baseline = load_frozen_baseline(scale)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        confirmation.kernel_compat.validate_transformers_kernel_bindings(kernels)
        runtime_ctx = geom.runtime_components(model)
        confirmation.validate_runtime_geometry(runtime_ctx, frozen)

        require(len(model.mamba.layers) == 48, "MODEL_LAYER_COUNT")
        require(hasattr(model.mamba, "norm_f"), "MODEL_FINAL_NORM")

        output_rows: list[dict[str, Any]] = []
        max_baseline_error = 0.0
        max_lens_error = 0.0
        max_prefix_error = 0.0

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[global_index]
            for cell in TARGET_CELLS:
                index = lookup[(pair, cell)]
                row = rows[index]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                target_index = anchor + geom.TARGET_OFFSET

                condition_results: dict[str, dict[str, Any]] = {}
                for condition in CONDITIONS:
                    key = (pair, cell, condition)
                    result = run_condition_with_capture(
                        spec=spec,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        frozen=frozen,
                        encoded=encoded,
                        row=row,
                        row_index_value=index,
                        anchor_index=anchor,
                        condition=condition,
                        device=device,
                        baseline=baseline[key],
                    )
                    condition_results[condition] = result
                    max_baseline_error = max(
                        max_baseline_error,
                        float(result["baseline_reproduction_max_abs_error"]),
                    )
                    max_lens_error = max(
                        max_lens_error,
                        float(result["final_norm_lens_max_abs_error"]),
                    )

                propagation = hidden_propagation(
                    condition_results,
                    target_index=target_index,
                )
                for stage in STAGE_ORDER:
                    for key in (
                        "selected_native_minus_neutralized",
                        "control_minus_neutralized",
                        "native_minus_control",
                    ):
                        max_prefix_error = max(
                            max_prefix_error,
                            float(propagation[stage][key]["pre_target_max_abs"]),
                        )

                pre = "pre_block_35"
                for primitive in PRIMITIVE_KEYS:
                    values = [
                        float(condition_results[c]["stage_lens"][pre][primitive])
                        for c in CONDITIONS
                    ]
                    require(max(values) - min(values) <= LENS_TOL, f"PRE_BLOCK_PRIMITIVE:{primitive}")
                pre_margins = [
                    float(
                        condition_results[c]["stage_lens"][pre][
                            "correct_class_logit_margin"
                        ]
                    )
                    for c in CONDITIONS
                ]
                require(max(pre_margins) - min(pre_margins) <= LENS_TOL, "PRE_BLOCK_MARGIN")

                output_rows.append({
                    "schema_version": ROW_SCHEMA,
                    "scale": scale,
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "source_pair_id": pair,
                    "row_id": str(row["row_id"]),
                    "contrast_cell_id": cell,
                    "correct_label_id": bridge.LABEL_ID_BY_CELL[cell],
                    "correct_label": bridge.LABEL_NAME_BY_CELL[cell],
                    "anchor_name": "A_IDENTITY",
                    "absolute_anchor_token_index": anchor,
                    "target_intervention_token_index": target_index,
                    "selected_dominant_candidate": spec["selected_plane"],
                    "response_blind_control_plane": spec["control_plane"],
                    "stage_order": list(STAGE_ORDER),
                    "conditions": {
                        condition: public_condition_result(condition_results[condition])
                        for condition in CONDITIONS
                    },
                    "hidden_propagation": propagation,
                    "scientific_full_model_forward_count": len(CONDITIONS),
                    "downstream_only_replay_count":
                        len(CONDITIONS) * DOWNSTREAM_REPLAYS_PER_FULL_FORWARD,
                    "shard_index": shard_id,
                    "physical_device": physical_device,
                })

                # Release captured GPU tensors before the next row.
                del condition_results

        torch.cuda.synchronize(device)

        expected_rows = int(shard["pair_count"]) * len(TARGET_CELLS)
        require(len(output_rows) == expected_rows, "OUTPUT_ROW_COUNT")
        paths["rows"].write_bytes(
            b"".join(canonical_json_bytes(row) for row in output_rows)
        )

        manifest = population_provenance["manifest"]
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": expected_head,
            "scale": scale,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "hf_repo": spec["hf_repo"],
            "hf_revision": spec["hf_revision"],
            "selected_dominant_candidate": spec["selected_plane"],
            "response_blind_control_plane": spec["control_plane"],
            "selection_uses_response": False,
            "population_first": shard["pair_first"],
            "population_last": shard["pair_last"],
            "source_pair_count": shard["pair_count"],
            "target_cells": list(TARGET_CELLS),
            "condition_order": list(CONDITIONS),
            "stage_order": list(STAGE_ORDER),
            "label_contract": {
                "C0_SHAM": "SUPPORT",
                "C2_NAME": "NOT_ENTITLED",
            },
            "shard_index": shard_id,
            "physical_device": physical_device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": "cuda:0",
            "logical_device_name": torch.cuda.get_device_name(0),
            "tokenizer": population_provenance["tokenizer"],
            "model_provenance": model_provenance,
            "structural_source_sha256": manifest["source_file_sha256"],
            "structural_rows_sha256": manifest["row_file_sha256"],
            "scientific_full_model_forward_count_this_run":
                FULL_FORWARDS_PER_SHARD,
            "downstream_only_replay_count_this_run":
                DOWNSTREAM_REPLAYS_PER_SHARD,
            "frozen_behavioral_reproduction_verified": True,
            "max_frozen_behavioral_reproduction_abs_error": max_baseline_error,
            "max_final_norm_lens_abs_error": max_lens_error,
            "max_pre_target_hidden_abs_error": max_prefix_error,
            "primary_inference_executed": False,
            "p_value_count_added": 0,
            "scientific_conclusion": None,
            "selection_reopened": False,
            "rescue_performed": False,
            "training_executed": False,
            "backward_executed": False,
        }
        paths["summary"].write_bytes(pretty_json_bytes(summary))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def write_shard_dir(
    output_dir: Path,
    *,
    shard_id: int,
    rows_raw: bytes,
    summary_raw: bytes,
) -> None:
    shard_dir = output_dir / f"shard{shard_id}"
    require(not shard_dir.exists(), f"SHARD_COLLISION:{shard_id}")
    shard_dir.mkdir(parents=True, exist_ok=False)
    (shard_dir / ROW_FILE).write_bytes(rows_raw)
    (shard_dir / SUMMARY_FILE).write_bytes(summary_raw)
    hashes = {
        ROW_FILE: hashlib.sha256(rows_raw).hexdigest(),
        SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (shard_dir / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )


def run_scale(
    *,
    scale: str,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
    authenticate_checkout: bool = True,
) -> None:
    validate_protocol()
    if authenticate_checkout:
        authenticate_repo(expected_head)

    spec = bridge.scale_spec(scale)
    geom = spec["geom"]
    confirmation = spec["confirmation"]
    geom.validate_snapshot(model_snapshot)
    require(
        compact_checkpoint.resolve() == (ROOT / spec["checkpoint_rel"]).resolve(),
        "COMPACT_PATH",
    )
    require(
        sha256_file(compact_checkpoint) == spec["checkpoint_sha256"],
        "COMPACT_SHA",
    )
    confirmation.load_frozen_selection()
    confirmation.load_frozen_geometry()
    bridge.load_population()
    load_frozen_baseline(scale)

    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix=f"gen4_{scale}_stagewise_localization_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            p = ctx.Process(
                target=worker_run,
                kwargs={
                    "scale": scale,
                    "shard": shard,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                    "expected_head": expected_head,
                },
                name=f"{scale}-stagewise-shard{shard['shard_id']}",
            )
            p.start()
            processes.append(p)

        for shard, p in zip(SHARDS, processes, strict=True):
            p.join()
            if p.exitcode != 0:
                paths = _worker_paths(temp_dir, int(shard["shard_id"]))
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise LocalizationError(
                    f"WORKER_FAILED:{shard['shard_id']}:\n{detail}"
                )

        output_dir.mkdir(parents=True, exist_ok=False)
        for shard in SHARDS:
            shard_id = int(shard["shard_id"])
            paths = _worker_paths(temp_dir, shard_id)
            write_shard_dir(
                output_dir,
                shard_id=shard_id,
                rows_raw=paths["rows"].read_bytes(),
                summary_raw=paths["summary"].read_bytes(),
            )

    print("RESULT=PASS_STAGEWISE_COUPLING_LOCALIZATION_RAW")
    print("SCALE=" + scale)
    print("PAIR_RANGE=xg1_fact_4801..xg1_fact_5100")
    print("SOURCE_PAIR_COUNT=300")
    print("CONDITIONS=native,dominant_neutralized,dominant_control")
    print("STAGES=" + ",".join(STAGE_ORDER))
    print(f"SCIENTIFIC_FULL_MODEL_FORWARD_COUNT={FULL_FORWARDS_PER_SCALE}")
    print(f"DOWNSTREAM_ONLY_REPLAY_COUNT={DOWNSTREAM_REPLAYS_PER_SCALE}")
    print("P_VALUE_COUNT_ADDED=0")
    print("SELECTION_REOPENED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


def run_both(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    output_dir: Path,
) -> None:
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_ROOT_COLLISION")

    spec370 = bridge.scale_spec("mamba370m")
    spec14 = bridge.scale_spec("mamba14b")
    run_scale(
        scale="mamba370m",
        expected_head=expected_head,
        model_snapshot=mamba370m_snapshot,
        compact_checkpoint=ROOT / spec370["checkpoint_rel"],
        output_dir=output_dir / "mamba370m",
        authenticate_checkout=False,
    )
    run_scale(
        scale="mamba14b",
        expected_head=expected_head,
        model_snapshot=mamba14b_snapshot,
        compact_checkpoint=ROOT / spec14["checkpoint_rel"],
        output_dir=output_dir / "mamba14b",
        authenticate_checkout=False,
    )
    print("RESULT=PASS_STAGEWISE_COUPLING_LOCALIZATION_BOTH")
    print("TOTAL_SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=3600")
    print("TOTAL_DOWNSTREAM_ONLY_REPLAY_COUNT=7200")
    print("P_VALUE_COUNT_ADDED=0")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        choices=("mamba370m", "mamba14b", "both"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mamba370m-snapshot", type=Path)
    parser.add_argument("--mamba14b-snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scale == "both":
        require(args.mamba370m_snapshot is not None, "M370_SNAPSHOT_ARG")
        require(args.mamba14b_snapshot is not None, "M14_SNAPSHOT_ARG")
        run_both(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
            output_dir=args.output_dir,
        )
        return 0

    spec = bridge.scale_spec(args.scale)
    snapshot = (
        args.mamba370m_snapshot
        if args.scale == "mamba370m"
        else args.mamba14b_snapshot
    )
    require(snapshot is not None, "SCALE_SNAPSHOT_ARG")
    run_scale(
        scale=args.scale,
        expected_head=args.expected_head,
        model_snapshot=snapshot,
        compact_checkpoint=ROOT / spec["checkpoint_rel"],
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
