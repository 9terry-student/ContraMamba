#!/usr/bin/env python3
"""K0-RVG layer-22 current-token RMSNorm routing-source decomposition.

Frozen scientific question:
    Why does the layer-22 current-token mixer input delta-X arrive with the
    corr-specific direction that the fixed W_H routes toward downstream
    strong-kernel output rows and relatively away from weak-output rows?

Authenticated boundary:
    R22 -> RMSNorm22 -> X22 -> Mixer22
    X = gamma * (s * R), s = rsqrt(mean(R^2) + eps)

Scientific symmetric decomposition:
    Q_R = gamma * (s_bar * delta_R)
    Q_s = gamma * (R_bar * delta_s)

Explicit numerical bridge:
    Q_eps = (X_m - X_alg,m) - (X_s - X_alg,s)
    delta_X_obs = Q_R + Q_s + Q_eps

The three vectors are propagated only through the already-frozen bias-free
layer-22 hidden in-projection W_H.  The downstream strong/weak partition is
also inherited unchanged from frozen lag-0 evidence.

Observational/algebraic only.  No raw per-item vectors are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "176e47b068961d488f9a57d6181dcf3591cbb9a7"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_"
    "static_design_candidate.md"
)
AUTHORITY_SHA256 = "7965994e7f3fcfa9c29e0cf6125c8239a8cf8e98efe46777487a2a581760c821"
AUTHORITY_BLOB = "472512bfca8f3c83aeb0973e320372480725e5a1"

PARENT_EVIDENCE_FREEZE_COMMIT = "8b494a72d48528c3bdb8985a1907766fded040e0"
PARENT_IMPLEMENTATION_COMMIT = "cd602f036b03e36e171837d9532541a530799954"
PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_audit.py"
)
PARENT_RUNNER_SHA256 = "8d6c45636197ee6cc1d9e6f8c423a1f8fe57fc654ae450a4b3eb6b8657dfaf5b"
PARENT_RUNNER_BLOB = "5a7329083c79b0db6874fc7a95290ed9f7f26c36"

PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_cd602f0_v1"
)
PARENT_ITEM_REL = PARENT_RUN_DIR + "/layer22_current_token_inproj_routing_item_metrics.jsonl"
PARENT_ITEM_SHA256 = "f3a34918c02b1715af5ec57221d9d3a34b23afb71849c3962f1aedbe458b14c8"
PARENT_CHANNEL_REL = PARENT_RUN_DIR + "/layer22_current_token_inproj_routing_channel_summary.jsonl"
PARENT_CHANNEL_SHA256 = "2cf91c7611cd57cd8bc1deda9269cb7e36b6008af1baab523e8a6b4818752cf2"
PARENT_CUMULATIVE_REL = PARENT_RUN_DIR + "/layer22_current_token_inproj_kernel_rank_cumulative.jsonl"
PARENT_CUMULATIVE_SHA256 = "dac78490055ffbd0f723bdf07570305d1d51ff48e4550f9f980769355f4772b5"
PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = "c61f5faabe7a47fc71a516c9035bf739fc7c47f5707041558ab9f7d6f98cc47a"
PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/execution_manifest.json"
PARENT_MANIFEST_SHA256 = "91d91b4d52966242e3c2b7d2532d91e3070991eaae8f2dec4f5d5293fe4a83df"
PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_inproj_strong_routing_"
    "validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = "391f671d22f7b369f109288b649f61c667e83696c920684147445f7d3dc7e8ce"

RMS_PRECEDENT_FREEZE_COMMIT = "b29e05dd384cddddf8754a9e9925ff9753f49bdb"
RMS_PRECEDENT_REL = "scripts/longterm_k0_rvg_rmsnorm_residual_factorization_audit.py"
RMS_PRECEDENT_SHA256 = "a4c5a1aea466df2e70b933050714c564ccf2d7db85f8bb9d36dc8c5e06937c4c"

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_current_token_rmsnorm_routing_source_audit.py"
)

K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

EXPECTED_ITEM_COUNT = 336
EXPECTED_PAIR_ROLE_COUNT = 672
EXPECTED_COMMON_COUNT = 330
EXPECTED_FULL_FORWARD_COUNT = 1344
EXPECTED_PREFLIGHT_FORWARD_COUNT = 4
EXPECTED_LAYER_COUNT = 24

SOURCE_LAYER = 22
TARGET_K = 2
TARGET_LAG = 0
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
EXPECTED_STRONG_COUNT = 240
EXPECTED_WEAK_COUNT = 1296
EXPECTED_EQUAL_COUNT = 0
EXPECTED_LAG0_KERNEL_RMS = 0.24383223809052498
RMS_EPS = 1e-5
EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

RMS_BRANCH_RECON_REL_TOL = 1e-6
PARENT_SCALAR_REL_TOL = 1e-13
PARENT_SCALAR_ABS_TOL = 1e-13
VECTOR_ABS_TOL = 5e-12
SQUARED_ENERGY_REL_TOL = 1e-12
KERNEL_RMS_REL_TOL = 1e-13
KERNEL_RMS_ABS_TOL = 1e-13

QUESTION = (
    "Is the validated layer-22 current-token strong/weak W_H routing "
    "redistribution primarily associated with raw residual-stream difference "
    "delta-R22, branch-specific RMS scaling contrast delta-s22, their vector "
    "interaction, or a genuinely mixed combination?"
)

ITEM_SCHEMA = "k0-rvg-layer22-current-token-rmsnorm-routing-source-item-v1"
CHANNEL_SCHEMA = "k0-rvg-layer22-current-token-rmsnorm-routing-source-channel-v1"
CUMULATIVE_SCHEMA = "k0-rvg-layer22-current-token-rmsnorm-routing-source-cumulative-v1"
SUMMARY_SCHEMA = "k0-rvg-layer22-current-token-rmsnorm-routing-source-summary-v1"
MANIFEST_SCHEMA = "k0-rvg-layer22-current-token-rmsnorm-routing-source-execution-manifest-v1"


class RMSNormRoutingSourceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RMSNormRoutingSourceError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_module(path: Path, name: str):
    require(path.is_file(), f"MODULE_MISSING:{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"MODULE_SPEC_FAILURE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RMSNormRoutingSourceError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RMSNormRoutingSourceError(f"GIT_SHOW_FAILURE:{spec}") from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(root: Path, runtime_mode: bool) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"GIT_BRANCH_MISMATCH:{branch}")

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (PARENT_EVIDENCE_FREEZE_COMMIT, "PARENT_EVIDENCE"),
        (RMS_PRECEDENT_FREEZE_COMMIT, "RMS_PRECEDENT"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_FREEZE_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()

    for line in status:
        path = _status_path(line)
        xy = line[:2]
        if path in K1_UNTRACKED:
            require(xy == "??", f"K1_STATE_CHANGED:{line}")
            continue
        if not runtime_mode and path == RUNNER_REL:
            require(xy == "??", f"STATIC_RUNNER_STATE_UNEXPECTED:{line}")
            continue
        raise RMSNormRoutingSourceError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

    runner = root / RUNNER_REL
    require(runner.is_file(), "RUNNER_FILE_MISSING")

    if runtime_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(tracked == 0, "RUNTIME_REQUIRES_TRACKED_RUNNER")
        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(clean == 0, "RUNTIME_RUNNER_WORKTREE_DRIFT")

    return {
        "branch": branch,
        "head": head,
        "status": status,
        "runtime_mode": runtime_mode,
    }


def _authenticate_file(
    root: Path,
    commit: str,
    rel: str,
    expected_sha: str,
    label: str,
    expected_blob: str | None = None,
) -> bytes:
    frozen = git_bytes(root, f"{commit}:{rel}")
    require(sha256_bytes(frozen) == expected_sha, f"{label}_SHA256_MISMATCH")
    if expected_blob is not None:
        require(
            git(root, "rev-parse", f"{commit}:{rel}") == expected_blob,
            f"{label}_BLOB_MISMATCH",
        )
    current = root / rel
    require(current.is_file(), f"{label}_MISSING")
    require(current.read_bytes() == frozen, f"{label}_WORKTREE_DRIFT")
    return frozen


def authenticate_authority(root: Path) -> None:
    _authenticate_file(
        root,
        AUTHORITY_FREEZE_COMMIT,
        AUTHORITY_REL,
        AUTHORITY_SHA256,
        "AUTHORITY",
        AUTHORITY_BLOB,
    )


def load_parent(root: Path):
    raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_RUNNER_REL,
        PARENT_RUNNER_SHA256,
        "PARENT_RUNNER",
        PARENT_RUNNER_BLOB,
    )
    require(raw == (root / PARENT_RUNNER_REL).read_bytes(), "PARENT_RUNNER_BYTES_MISMATCH")
    return import_module(root / PARENT_RUNNER_REL, "k0_rvg_current_token_inproj_parent")


def authenticate_rms_precedent(root: Path) -> None:
    _authenticate_file(
        root,
        RMS_PRECEDENT_FREEZE_COMMIT,
        RMS_PRECEDENT_REL,
        RMS_PRECEDENT_SHA256,
        "RMS_PRECEDENT",
    )


def _load_json(raw: bytes, label: str):
    try:
        return json.loads(raw)
    except Exception as exc:
        raise RMSNormRoutingSourceError(f"{label}_JSON_PARSE_FAILURE") from exc


def authenticate_parent_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    item_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_ITEM_REL,
        PARENT_ITEM_SHA256,
        "PARENT_ITEM",
    )
    channel_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_CHANNEL_REL,
        PARENT_CHANNEL_SHA256,
        "PARENT_CHANNEL",
    )
    _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_CUMULATIVE_REL,
        PARENT_CUMULATIVE_SHA256,
        "PARENT_CUMULATIVE",
    )
    summary_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_SUMMARY_REL,
        PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY",
    )
    manifest_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_MANIFEST_REL,
        PARENT_MANIFEST_SHA256,
        "PARENT_MANIFEST",
    )
    _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE_COMMIT,
        PARENT_REPORT_REL,
        PARENT_REPORT_SHA256,
        "PARENT_REPORT",
    )

    summary = _load_json(summary_raw, "PARENT_SUMMARY")
    manifest = _load_json(manifest_raw, "PARENT_MANIFEST")
    require(
        summary.get("schema_version")
        == "k0-rvg-layer22-current-token-inproj-routing-summary-v1",
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-current-token-inproj-routing-execution-manifest-v1",
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("runtime_git_head") == PARENT_IMPLEMENTATION_COMMIT,
        "PARENT_RUNTIME_HEAD_MISMATCH",
    )
    require(
        manifest.get("runner_sha256") == PARENT_RUNNER_SHA256,
        "PARENT_MANIFEST_RUNNER_MISMATCH",
    )
    require(
        manifest.get("model_forward_count") == EXPECTED_FULL_FORWARD_COUNT,
        "PARENT_FORWARD_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "PARENT_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("strong_kernel_channel_count") == EXPECTED_STRONG_COUNT,
        "PARENT_STRONG_COUNT_MISMATCH",
    )
    require(
        summary.get("weak_kernel_channel_count") == EXPECTED_WEAK_COUNT,
        "PARENT_WEAK_COUNT_MISMATCH",
    )
    require(
        summary.get("equal_kernel_channel_count") == EXPECTED_EQUAL_COUNT,
        "PARENT_EQUAL_COUNT_MISMATCH",
    )

    parent_items: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(item_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise RMSNormRoutingSourceError(
                f"PARENT_ITEM_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        idx = int(row["local_template_index"])
        require(idx not in parent_items, f"PARENT_ITEM_DUPLICATE:{idx}")
        parent_items[idx] = row
    require(len(parent_items) == EXPECTED_COMMON_COUNT, "PARENT_ITEM_COUNT_MISMATCH")

    parent_channels: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(channel_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise RMSNormRoutingSourceError(
                f"PARENT_CHANNEL_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        j = int(row["channel_index"])
        require(j not in parent_channels, f"PARENT_CHANNEL_DUPLICATE:{j}")
        parent_channels[j] = row
    require(len(parent_channels) == INTERMEDIATE_SIZE, "PARENT_CHANNEL_COUNT_MISMATCH")

    return summary, manifest, parent_items, parent_channels


def build_parent_context(root: Path, parent: Any) -> dict[str, Any]:
    parent_lag0 = parent.load_parent(root)
    values = parent.build_plan(root, parent_lag0)
    require(len(values) == 24, "PARENT_BUILD_PLAN_ARITY_MISMATCH")
    (
        four_tap_parent,
        u_path_parent,
        write_parent,
        carry_parent,
        recurrent_parent,
        gate_parent,
        output_parent,
        residual_parent,
        rms_parent,
        hidden_parent,
        four_tap,
        postconv,
        time_step,
        dt_projection,
        operating,
        secant,
        discrete_b,
        write,
        carry,
        magnitude,
        base,
        plan,
        cohort,
        signatures,
    ) = values
    require(len(plan) == EXPECTED_PAIR_ROLE_COUNT, "PLAN_COUNT_MISMATCH")
    require(len(cohort) == EXPECTED_COMMON_COUNT, "COMMON_COHORT_COUNT_MISMATCH")
    return {
        "parent_lag0": parent_lag0,
        "four_tap_parent": four_tap_parent,
        "u_path_parent": u_path_parent,
        "write_parent": write_parent,
        "carry_parent": carry_parent,
        "recurrent_parent": recurrent_parent,
        "gate_parent": gate_parent,
        "output_parent": output_parent,
        "residual_parent": residual_parent,
        "rms_parent": rms_parent,
        "hidden_parent": hidden_parent,
        "four_tap": four_tap,
        "postconv": postconv,
        "time_step": time_step,
        "dt_projection": dt_projection,
        "operating": operating,
        "secant": secant,
        "discrete_b": discrete_b,
        "write": write,
        "carry": carry,
        "magnitude": magnitude,
        "base": base,
        "plan": plan,
        "cohort": cohort,
        "signatures": signatures,
    }


def resolve_runtime(
    root: Path,
    parent: Any,
    ctx: Mapping[str, Any],
    handoff_path: Path,
) -> dict[str, Any]:
    bundle = parent.resolve_runtime_bundle(
        root,
        ctx["parent_lag0"],
        ctx["four_tap_parent"],
        ctx["u_path_parent"],
        ctx["write_parent"],
        ctx["carry_parent"],
        ctx["recurrent_parent"],
        ctx["output_parent"],
        ctx["residual_parent"],
        ctx["rms_parent"],
        ctx["hidden_parent"],
        ctx["postconv"],
        ctx["base"],
        handoff_path,
    )
    require(len(bundle) == 14, "PARENT_RUNTIME_BUNDLE_ARITY_MISMATCH")
    (
        observer,
        k2s,
        model,
        binding,
        layer_map,
        handoff,
        encoder,
        snapshot_info,
        layer22,
        layer23,
        norm23,
        mixer23,
        mixer22,
        out_proj,
    ) = bundle
    require(len(layer_map) == EXPECTED_LAYER_COUNT, "LAYER_MAP_COUNT_MISMATCH")

    operator = parent.resolve_operator(
        ctx["parent_lag0"],
        ctx["four_tap_parent"],
        ctx["u_path_parent"],
        mixer22,
    )

    norm22 = getattr(layer22, "norm", None)
    block_mixer = getattr(layer22, "mixer", None)
    require(norm22 is not None, "LAYER22_NORM_MISSING")
    require(block_mixer is mixer22, "LAYER22_MIXER_IDENTITY_MISMATCH")

    return {
        "observer": observer,
        "k2s": k2s,
        "model": model,
        "binding": binding,
        "layer_map": layer_map,
        "handoff": handoff,
        "encoder": encoder,
        "snapshot_info": snapshot_info,
        "layer22": layer22,
        "layer23": layer23,
        "norm22": norm22,
        "norm23": norm23,
        "mixer23": mixer23,
        "mixer22": mixer22,
        "out_proj": out_proj,
        "operator": operator,
    }


def authenticate_operator_and_norm(
    runtime: Mapping[str, Any],
    parent_channels: Mapping[int, Mapping[str, Any]],
) -> None:
    import torch

    norm = runtime["norm22"]
    op = runtime["operator"]
    require(tuple(norm.weight.shape) == (HIDDEN_SIZE,), "RMS_WEIGHT_SHAPE_MISMATCH")
    require(norm.weight.dtype == torch.float32, "RMS_WEIGHT_DTYPE_MISMATCH")
    require(float(norm.variance_epsilon) == RMS_EPS, "RMS_EPS_MISMATCH")

    require(
        tuple(op["w_hidden64"].shape) == (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "W_H_SHAPE_MISMATCH",
    )
    require(
        int(op["strong_mask"].sum().item()) == EXPECTED_STRONG_COUNT,
        "STRONG_COUNT_RUNTIME_MISMATCH",
    )
    require(
        int(op["weak_mask"].sum().item()) == EXPECTED_WEAK_COUNT,
        "WEAK_COUNT_RUNTIME_MISMATCH",
    )
    require(
        int(op["equal_mask"].sum().item()) == EXPECTED_EQUAL_COUNT,
        "EQUAL_COUNT_RUNTIME_MISMATCH",
    )
    require(
        math.isclose(
            float(op["rms"]),
            EXPECTED_LAG0_KERNEL_RMS,
            rel_tol=KERNEL_RMS_REL_TOL,
            abs_tol=KERNEL_RMS_ABS_TOL,
        ),
        "KERNEL_RMS_RUNTIME_MISMATCH",
    )

    for j in range(INTERMEDIATE_SIZE):
        p = parent_channels[j]
        part = (
            "strong" if bool(op["strong_mask"][j].item())
            else "weak" if bool(op["weak_mask"][j].item())
            else "equal"
        )
        require(p["downstream_partition"] == part, f"PARENT_PARTITION_MISMATCH:{j}")
        require(
            int(p["kernel_magnitude_rank"])
            == int(op["kernel_order"].index(j) + 1),
            f"PARENT_KERNEL_RANK_MISMATCH:{j}",
        )
        require(
            math.isclose(
                float(p["inproj_row_gain_sq"]),
                float(op["row_r2"][j].item()),
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_ROW_GAIN_MISMATCH:{j}",
        )


def _prefixes(
    parent: Any,
    ctx: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    matched, swapped = parent._prefixes(
        ctx["parent_lag0"],
        ctx["four_tap_parent"],
        ctx["u_path_parent"],
        ctx["write_parent"],
        ctx["carry_parent"],
        ctx["recurrent_parent"],
        row,
    )
    return tuple(matched), tuple(swapped)


def capture_branch(
    parent: Any,
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    token_ids: Sequence[int],
    parent_targets: Sequence[int],
    rms_target: int,
) -> tuple[dict[int, Mapping[str, Any]], dict[str, Any]]:
    import torch

    norm = runtime["norm22"]
    r_holder: dict[str, Any] = {}
    x_holder: dict[str, Any] = {}
    counts = {"pre": 0, "post": 0}

    def norm_pre_hook(module, args):
        counts["pre"] += 1
        require(counts["pre"] == 1, "DUPLICATE_LAYER22_NORM_PRE_HOOK")
        require(len(args) == 1, "LAYER22_NORM_PRE_ARG_COUNT_MISMATCH")
        full = args[0].detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, f"R_DTYPE_MISMATCH:{full.dtype}")
        require(full.ndim == 3 and full.shape[0] == 1, "R_FULL_SHAPE_MISMATCH")
        require(full.shape[-1] == HIDDEN_SIZE, "R_WIDTH_MISMATCH")
        require(0 <= rms_target < full.shape[1], "R_TARGET_OUT_OF_RANGE")
        r_holder["R32"] = full[0, rms_target, :].contiguous().clone()

    def norm_post_hook(module, args, output):
        counts["post"] += 1
        require(counts["post"] == 1, "DUPLICATE_LAYER22_NORM_POST_HOOK")
        full = output.detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, f"X_DTYPE_MISMATCH:{full.dtype}")
        require(full.ndim == 3 and full.shape[0] == 1, "X_FULL_SHAPE_MISMATCH")
        require(full.shape[-1] == HIDDEN_SIZE, "X_WIDTH_MISMATCH")
        require(0 <= rms_target < full.shape[1], "X_TARGET_OUT_OF_RANGE")
        x_holder["X32"] = full[0, rms_target, :].contiguous().clone()

    pre_handle = norm.register_forward_pre_hook(norm_pre_hook)
    post_handle = norm.register_forward_hook(norm_post_hook)
    try:
        parent_records = parent.capture(
            ctx["parent_lag0"],
            ctx["four_tap_parent"],
            ctx["u_path_parent"],
            ctx["write_parent"],
            ctx["base"],
            runtime["model"],
            runtime["binding"],
            runtime["layer_map"],
            runtime["mixer22"],
            runtime["operator"],
            token_ids,
            parent_targets,
        )
    finally:
        pre_handle.remove()
        post_handle.remove()

    require(counts == {"pre": 1, "post": 1}, f"RMS_HOOK_COUNT_FAILURE:{counts}")
    require("R32" in r_holder and "X32" in x_holder, "RMS_CAPTURE_MISSING")
    require(rms_target in parent_records, "PARENT_TARGET_MISSING")

    parent_x = parent_records[rms_target]["X_RF32"][TARGET_LAG, :]
    require(
        torch.equal(x_holder["X32"], parent_x),
        "RMS_OUTPUT_PARENT_X_EXACT_MISMATCH",
    )

    gamma32 = norm.weight.detach().cpu().contiguous()
    r32 = r_holder["R32"]
    x32 = x_holder["X32"]
    variance32 = r32.pow(2).mean()
    scale32 = torch.rsqrt(variance32 + RMS_EPS)
    reconstructed32 = (gamma32 * (r32 * scale32)).to(x32.dtype).contiguous()

    residual64 = reconstructed32.to(torch.float64) - x32.to(torch.float64)
    denom = max(float(torch.linalg.vector_norm(x32.to(torch.float64)).item()), 1e-12)
    branch_rel = float(torch.linalg.vector_norm(residual64).item()) / denom
    require(
        branch_rel <= RMS_BRANCH_RECON_REL_TOL,
        f"RMS_BRANCH_RECONSTRUCTION_FAILURE:{rms_target}:{branch_rel}",
    )

    return parent_records, {
        "R32": r32,
        "X32": x32,
        "rms_scale32": scale32.detach().cpu().clone(),
        "rms_branch_reconstruction_relative_residual": branch_rel,
    }


def _isclose_parent(got, expected) -> bool:
    if got is None or expected is None:
        return got is None and expected is None
    return math.isclose(
        float(got),
        float(expected),
        rel_tol=PARENT_SCALAR_REL_TOL,
        abs_tol=PARENT_SCALAR_ABS_TOL,
    )


def _partition_sum(vec, op: Mapping[str, Any], part: str) -> float:
    import torch

    if part == "all":
        return float(torch.sum(vec).item())
    if part == "strong":
        return float(torch.sum(vec[op["strong_mask"]]).item())
    if part == "weak":
        return float(torch.sum(vec[op["weak_mask"]]).item())
    raise RMSNormRoutingSourceError(f"UNKNOWN_PARTITION:{part}")


def role_decomposition(
    row: Mapping[str, Any],
    matched_rms: Mapping[str, Any],
    swapped_rms: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_item: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])
    gamma = runtime["norm22"].weight.detach().cpu().to(torch.float64).contiguous()
    w = runtime["operator"]["w_hidden64"]

    rm = matched_rms["R32"].to(torch.float64).contiguous()
    rs = swapped_rms["R32"].to(torch.float64).contiguous()
    xm = matched_rms["X32"].to(torch.float64).contiguous()
    xs = swapped_rms["X32"].to(torch.float64).contiguous()
    sm = float(matched_rms["rms_scale32"].item())
    ss = float(swapped_rms["rms_scale32"].item())

    dr = rm - rs
    rbar = 0.5 * (rm + rs)
    ds = sm - ss
    sbar = 0.5 * (sm + ss)
    x_obs = xm - xs
    dx_sq = float(torch.dot(x_obs, x_obs).item())
    require(dx_sq > 0.0, f"DELTA_X_ZERO:{idx}:{role}")

    q_r = gamma * (sbar * dr)
    q_s = gamma * (rbar * ds)

    x_alg_m = gamma * (sm * rm)
    x_alg_s = gamma * (ss * rs)
    eps_m = xm - x_alg_m
    eps_s = xs - x_alg_s
    q_eps = eps_m - eps_s

    q_science = q_r + q_s
    bridge_vec = x_obs - q_science
    bridge_identity_abs = float(torch.max(torch.abs(bridge_vec - q_eps)).item())
    require(
        bridge_identity_abs <= VECTOR_ABS_TOL,
        f"RMS_ERROR_DIFFERENCE_IDENTITY_FAILURE:{idx}:{role}:{bridge_identity_abs}",
    )
    difference_rel_diag = float(
        torch.linalg.vector_norm(q_science - x_obs).item()
    ) / math.sqrt(dx_sq)

    def dot(a, b):
        return float(torch.dot(a, b).item())

    i_r = dot(q_r, q_r) / dx_sq
    i_s = dot(q_s, q_s) / dx_sq
    i_rs = 2.0 * dot(q_r, q_s) / dx_sq
    i_eps = (
        dot(q_eps, q_eps)
        + 2.0 * dot(q_r, q_eps)
        + 2.0 * dot(q_s, q_eps)
    ) / dx_sq
    input_closure = abs(1.0 - (i_r + i_s + i_rs + i_eps))
    require(
        input_closure <= SQUARED_ENERGY_REL_TOL,
        f"INPUT_ENERGY_CLOSURE_FAILURE:{idx}:{role}:{input_closure}",
    )

    h_r = torch.mv(w, q_r)
    h_s = torch.mv(w, q_s)
    h_eps = torch.mv(w, q_eps)
    h_total = torch.mv(w, x_obs)
    h_sum = h_r + h_s + h_eps
    h_vector_closure = float(torch.max(torch.abs(h_total - h_sum)).item())
    require(
        h_vector_closure <= VECTOR_ABS_TOL,
        f"H_VECTOR_CLOSURE_FAILURE:{idx}:{role}:{h_vector_closure}",
    )

    e_r = (h_r * h_r) / dx_sq
    e_s = (h_s * h_s) / dx_sq
    e_rs = (2.0 * h_r * h_s) / dx_sq
    e_eps = (
        h_eps * h_eps
        + 2.0 * h_r * h_eps
        + 2.0 * h_s * h_eps
    ) / dx_sq
    e_total = (h_total * h_total) / dx_sq

    channel_closure = float(
        torch.max(torch.abs(e_total - (e_r + e_s + e_rs + e_eps))).item()
    )
    require(
        channel_closure <= VECTOR_ABS_TOL,
        f"CHANNEL_COMPONENT_CLOSURE_FAILURE:{idx}:{role}:{channel_closure}",
    )

    t_h_sq = _partition_sum(e_total, runtime["operator"], "all")
    t_s_sq = _partition_sum(e_total, runtime["operator"], "strong")
    t_w_sq = _partition_sum(e_total, runtime["operator"], "weak")
    t_h = math.sqrt(max(t_h_sq, 0.0))
    t_s = math.sqrt(max(t_s_sq, 0.0))
    t_w = math.sqrt(max(t_w_sq, 0.0))
    p_s = t_s_sq / t_h_sq
    p_w = t_w_sq / t_h_sq
    dx_l2 = math.sqrt(dx_sq)

    for field, got in (
        (f"delta_x_current_l2_{role}", dx_l2),
        (f"t_h_{role}", t_h),
        (f"t_s_{role}", t_s),
        (f"t_w_{role}", t_w),
        (f"p_s_{role}", p_s),
        (f"p_w_{role}", p_w),
    ):
        expected = parent_item[field]
        require(
            _isclose_parent(got, expected),
            f"PARENT_SCALAR_REPRODUCTION_FAILURE:{idx}:{role}:{field}:{got}:{expected}",
        )

    return {
        "idx": idx,
        "stable_item_id": row["stable_item_id"],
        "role": role,
        "delta_r_l2": float(torch.linalg.vector_norm(dr).item()),
        "delta_x_l2": dx_l2,
        "delta_rms_scale": ds,
        "abs_delta_rms_scale": abs(ds),
        "q_r_l2": float(torch.linalg.vector_norm(q_r).item()),
        "q_s_l2": float(torch.linalg.vector_norm(q_s).item()),
        "q_eps_l2": float(torch.linalg.vector_norm(q_eps).item()),
        "rms_branch_reconstruction_relative_residual": max(
            float(matched_rms["rms_branch_reconstruction_relative_residual"]),
            float(swapped_rms["rms_branch_reconstruction_relative_residual"]),
        ),
        "difference_reconstruction_relative_residual_diagnostic": difference_rel_diag,
        "error_difference_identity_abs_residual": bridge_identity_abs,
        "input_energy_i_r": i_r,
        "input_energy_i_s": i_s,
        "input_energy_i_rs": i_rs,
        "input_energy_i_eps": i_eps,
        "input_energy_closure_abs_residual": input_closure,
        "h_vector_closure_abs_residual": h_vector_closure,
        "channel_component_closure_max_abs_residual": channel_closure,
        "t_h": t_h,
        "t_s": t_s,
        "t_w": t_w,
        "p_s": p_s,
        "p_w": p_w,
        "e_total": e_total,
        "e_r": e_r,
        "e_s": e_s,
        "e_rs": e_rs,
        "e_eps": e_eps,
    }


def pair_role_records(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_item: Mapping[str, Any],
):
    import torch

    require(corr["idx"] == ctrl["idx"], "PAIR_INDEX_MISMATCH")
    require(corr["stable_item_id"] == ctrl["stable_item_id"], "PAIR_STABLE_ID_MISMATCH")
    idx = int(corr["idx"])

    component_vectors = {}
    for key in ("total", "r", "s", "rs", "eps"):
        component_vectors[key] = corr[f"e_{key}"] - ctrl[f"e_{key}"]

    component_closure = float(
        torch.max(
            torch.abs(
                component_vectors["total"]
                - (
                    component_vectors["r"]
                    + component_vectors["s"]
                    + component_vectors["rs"]
                    + component_vectors["eps"]
                )
            )
        ).item()
    )
    require(
        component_closure <= VECTOR_ABS_TOL,
        f"PAIR_COMPONENT_CHANNEL_CLOSURE_FAILURE:{idx}:{component_closure}",
    )

    partition_values: dict[str, dict[str, float]] = {}
    max_partition_closure = 0.0
    for part in ("all", "strong", "weak"):
        vals = {
            key: _partition_sum(component_vectors[key], runtime["operator"], part)
            for key in ("total", "r", "s", "rs", "eps")
        }
        closure = abs(vals["total"] - (vals["r"] + vals["s"] + vals["rs"] + vals["eps"]))
        require(
            closure <= VECTOR_ABS_TOL,
            f"PARTITION_COMPONENT_CLOSURE_FAILURE:{idx}:{part}:{closure}",
        )
        vals["closure"] = closure
        partition_values[part] = vals
        max_partition_closure = max(max_partition_closure, closure)

    parent_targets = {
        "all": float(parent_item["delta_t_h_sq"]),
        "strong": float(parent_item["delta_t_s_sq"]),
        "weak": float(parent_item["delta_t_w_sq"]),
    }
    max_parent_bridge = 0.0
    for part in ("all", "strong", "weak"):
        residual = abs(partition_values[part]["total"] - parent_targets[part])
        require(
            residual <= VECTOR_ABS_TOL,
            f"PARENT_PARTITION_TOTAL_REPRODUCTION_FAILURE:{idx}:{part}:{residual}",
        )
        max_parent_bridge = max(max_parent_bridge, residual)

    row = {
        "schema_version": ITEM_SCHEMA,
        "local_template_index": idx,
        "stable_item_id": corr["stable_item_id"],
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
    }

    for role_name, rec in (("corr", corr), ("ctrl", ctrl)):
        for field in (
            "delta_r_l2",
            "delta_x_l2",
            "delta_rms_scale",
            "abs_delta_rms_scale",
            "q_r_l2",
            "q_s_l2",
            "q_eps_l2",
            "rms_branch_reconstruction_relative_residual",
            "difference_reconstruction_relative_residual_diagnostic",
            "error_difference_identity_abs_residual",
            "input_energy_i_r",
            "input_energy_i_s",
            "input_energy_i_rs",
            "input_energy_i_eps",
            "input_energy_closure_abs_residual",
            "h_vector_closure_abs_residual",
            "channel_component_closure_max_abs_residual",
            "t_h",
            "t_s",
            "t_w",
            "p_s",
            "p_w",
        ):
            row[f"{field}_{role_name}"] = rec[field]

    # Persist the frozen parent scalar bridges and their paired ordering
    # explicitly, in addition to the independently recomputed values above.
    row["parent_delta_x_current_l2_corr"] = parent_item["delta_x_current_l2_corr"]
    row["parent_delta_x_current_l2_ctrl"] = parent_item["delta_x_current_l2_ctrl"]
    for field in ("t_h", "t_s", "t_w", "p_s", "p_w"):
        pc = float(parent_item[f"{field}_corr"])
        pt = float(parent_item[f"{field}_ctrl"])
        row[f"parent_{field}_corr"] = pc
        row[f"parent_{field}_ctrl"] = pt
        row[f"parent_{field}_corr_gt_ctrl"] = bool(pc > pt)
        row[f"parent_{field}_corr_lt_ctrl"] = bool(pc < pt)
        row[f"parent_{field}_equal"] = bool(pc == pt)

    for part in ("all", "strong", "weak"):
        vals = partition_values[part]
        for key in ("total", "r", "s", "rs", "eps"):
            row[f"delta_{key}_{part}"] = vals[key]
        row[f"component_closure_abs_residual_{part}"] = vals["closure"]
        row[f"parent_total_{part}"] = parent_targets[part]
        row[f"parent_reproduction_abs_residual_{part}"] = abs(
            vals["total"] - parent_targets[part]
        )

    row["pair_channel_component_closure_max_abs_residual"] = component_closure
    row["max_partition_component_closure_abs_residual"] = max_partition_closure
    row["max_parent_partition_reproduction_abs_residual"] = max_parent_bridge
    row["raw_vectors_persisted"] = False

    return row, component_vectors


def _median(values):
    values = [float(v) for v in values]
    require(bool(values), "EMPTY_MEDIAN")
    return float(statistics.median(values))


def _aggregate(values):
    vals = [float(v) for v in values]
    require(bool(vals), "EMPTY_AGGREGATE")
    require(all(math.isfinite(v) for v in vals), "NONFINITE_AGGREGATE")
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _paired_count(rows, corr_field: str, ctrl_field: str) -> dict[str, int]:
    gt = lt = eq = 0
    for row in rows:
        a = float(row[corr_field])
        b = float(row[ctrl_field])
        if a > b:
            gt += 1
        elif a < b:
            lt += 1
        else:
            eq += 1
    return {
        "count": len(rows),
        "defined_pair_count": len(rows),
        "undefined_pair_count": 0,
        "corr_gt_ctrl": gt,
        "corr_lt_ctrl": lt,
        "equal": eq,
    }


def build_channel_outputs(
    component_matrices: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_channels: Mapping[int, Mapping[str, Any]],
):
    import torch

    expected = (EXPECTED_COMMON_COUNT, INTERMEDIATE_SIZE)
    for key, matrix in component_matrices.items():
        require(tuple(matrix.shape) == expected, f"{key.upper()}_MATRIX_SHAPE_MISMATCH")

    mean = {k: torch.mean(v, dim=0) for k, v in component_matrices.items()}
    median = {}
    pos = {}
    neg = {}
    zero = {}
    for key, matrix in component_matrices.items():
        sorted_v, _ = torch.sort(matrix, dim=0)
        median[key] = (
            sorted_v[(EXPECTED_COMMON_COUNT // 2) - 1, :]
            + sorted_v[EXPECTED_COMMON_COUNT // 2, :]
        ) / 2.0
        pos[key] = torch.sum(matrix > 0.0, dim=0)
        neg[key] = torch.sum(matrix < 0.0, dim=0)
        zero[key] = torch.sum(matrix == 0.0, dim=0)

    op = runtime["operator"]
    rank_by_kernel = [0] * INTERMEDIATE_SIZE
    for rank, channel in enumerate(op["kernel_order"], start=1):
        rank_by_kernel[int(channel)] = rank

    channel_rows = []
    for j in range(INTERMEDIATE_SIZE):
        p = parent_channels[j]
        part = (
            "strong" if bool(op["strong_mask"][j].item())
            else "weak" if bool(op["weak_mask"][j].item())
            else "equal"
        )
        row = {
            "schema_version": CHANNEL_SCHEMA,
            "channel_index": j,
            "kernel_magnitude_rank": int(rank_by_kernel[j]),
            "downstream_partition": part,
            "downstream_lag0_kernel_weight": float(op["lag0"][j].item()),
            "downstream_lag0_kernel_weight_sq": float(op["lag0_k2"][j].item()),
            "inproj_row_gain_sq": float(op["row_r2"][j].item()),
        }
        require(
            int(p["kernel_magnitude_rank"]) == row["kernel_magnitude_rank"],
            f"CHANNEL_PARENT_RANK_MISMATCH:{j}",
        )
        require(
            p["downstream_partition"] == part,
            f"CHANNEL_PARENT_PARTITION_MISMATCH:{j}",
        )
        for key in ("total", "r", "s", "rs", "eps"):
            row[f"mean_d_{key}"] = float(mean[key][j].item())
            row[f"median_d_{key}"] = float(median[key][j].item())
            row[f"positive_d_{key}_count"] = int(pos[key][j].item())
            row[f"negative_d_{key}_count"] = int(neg[key][j].item())
            row[f"zero_d_{key}_count"] = int(zero[key][j].item())
        channel_rows.append(row)

    cumulative_rows = []
    sums = {k: 0.0 for k in ("total", "r", "s", "rs", "eps")}
    for rank, channel in enumerate(op["kernel_order"], start=1):
        j = int(channel)
        for key in sums:
            sums[key] += float(mean[key][j].item())
        cumulative_rows.append(
            {
                "schema_version": CUMULATIVE_SCHEMA,
                "kernel_magnitude_rank": rank,
                "channel_index": j,
                "downstream_partition": channel_rows[j]["downstream_partition"],
                **{f"cumulative_mean_d_{k}": sums[k] for k in sums},
            }
        )

    return channel_rows, cumulative_rows


def component_attribution(item_rows) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in ("all", "strong", "weak"):
        means = {
            key: float(statistics.fmean(row[f"delta_{key}_{part}"] for row in item_rows))
            for key in ("total", "r", "s", "rs", "eps")
        }
        closure = abs(
            means["total"]
            - (means["r"] + means["s"] + means["rs"] + means["eps"])
        )
        require(
            closure <= VECTOR_ABS_TOL,
            f"AGGREGATE_COMPONENT_CLOSURE_FAILURE:{part}:{closure}",
        )
        a = sum(abs(means[k]) for k in ("r", "s", "rs", "eps"))
        shares = {
            k: (abs(means[k]) / a if a > 0.0 else None)
            for k in ("r", "s", "rs", "eps")
        }
        signed_to_net = {
            k: (means[k] / means["total"] if means["total"] != 0.0 else None)
            for k in ("r", "s", "rs", "eps")
        }
        scientific = ("r", "s", "rs")
        largest = max(scientific, key=lambda k: abs(means[k]))
        out[part] = {
            "mean_signed_components": means,
            "absolute_component_mass": a,
            "absolute_component_shares": shares,
            "signed_component_to_total_net_ratios": signed_to_net,
            "largest_absolute_scientific_component": largest,
            "aggregate_closure_abs_residual": closure,
        }
    return out


def make_summary(
    item_rows,
    runtime: Mapping[str, Any],
    attribution: Mapping[str, Any],
) -> dict[str, Any]:
    op = runtime["operator"]
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "scientific_question": QUESTION,
        "common_ddsssss_item_count": len(item_rows),
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "lag0_kernel_rms": float(op["rms"]),
        "strong_kernel_channel_count": int(op["strong_mask"].sum().item()),
        "weak_kernel_channel_count": int(op["weak_mask"].sum().item()),
        "equal_kernel_channel_count": int(op["equal_mask"].sum().item()),
        "parent_current_token_routing_match": True,
        "rmsnorm_parent_x_exact_match": True,
        "component_attribution": attribution,
        "role_diagnostics": {},
        "parent_paired_counts": {},
        "max_rms_branch_reconstruction_relative_residual": max(
            max(
                float(r["rms_branch_reconstruction_relative_residual_corr"]),
                float(r["rms_branch_reconstruction_relative_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_difference_reconstruction_relative_residual_diagnostic": max(
            max(
                float(r["difference_reconstruction_relative_residual_diagnostic_corr"]),
                float(r["difference_reconstruction_relative_residual_diagnostic_ctrl"]),
            )
            for r in item_rows
        ),
        "max_error_difference_identity_abs_residual": max(
            max(
                float(r["error_difference_identity_abs_residual_corr"]),
                float(r["error_difference_identity_abs_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_input_energy_closure_abs_residual": max(
            max(
                float(r["input_energy_closure_abs_residual_corr"]),
                float(r["input_energy_closure_abs_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_channel_component_closure_abs_residual": max(
            max(
                float(r["channel_component_closure_max_abs_residual_corr"]),
                float(r["channel_component_closure_max_abs_residual_ctrl"]),
                float(r["pair_channel_component_closure_max_abs_residual"]),
            )
            for r in item_rows
        ),
        "max_partition_component_closure_abs_residual": max(
            float(r["max_partition_component_closure_abs_residual"])
            for r in item_rows
        ),
        "max_parent_partition_reproduction_abs_residual": max(
            float(r["max_parent_partition_reproduction_abs_residual"])
            for r in item_rows
        ),
        "raw_vectors_persisted": False,
        "raw_per_item_channel_vectors_persisted": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed": False,
    }

    for role in ("corr", "ctrl"):
        summary["role_diagnostics"][role] = {}
        for field in (
            "delta_r_l2",
            "delta_x_l2",
            "delta_rms_scale",
            "abs_delta_rms_scale",
            "q_r_l2",
            "q_s_l2",
            "q_eps_l2",
            "input_energy_i_r",
            "input_energy_i_s",
            "input_energy_i_rs",
            "input_energy_i_eps",
            "t_h",
            "t_s",
            "t_w",
            "p_s",
            "p_w",
        ):
            summary["role_diagnostics"][role][field] = _aggregate(
                r[f"{field}_{role}"] for r in item_rows
            )

    for field in ("t_h", "t_s", "t_w", "p_s", "p_w"):
        summary["parent_paired_counts"][field] = _paired_count(
            item_rows, f"{field}_corr", f"{field}_ctrl"
        )

    return summary


def json_bytes(obj) -> bytes:
    return (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def _first_common_pair_rows(plan, cohort):
    by_idx: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in plan:
        idx = int(row["local_template_index"])
        if idx not in cohort:
            continue
        by_idx.setdefault(idx, {})[str(row["role"])] = row
    for idx in sorted(by_idx):
        pair = by_idx[idx]
        if "corr" in pair and "ctrl" in pair:
            return pair["corr"], pair["ctrl"]
    raise RMSNormRoutingSourceError("NO_COMMON_PREFLIGHT_PAIR")


def run_role(
    row: Mapping[str, Any],
    parent: Any,
    ctx: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    idx = int(row["local_template_index"])
    require(idx in parent_items, f"PARENT_ITEM_MISSING:{idx}")
    matched_prefix, swapped_prefix = _prefixes(parent, ctx, row)
    target = int(row["anchor"]) + TARGET_K

    matched_parent, matched_rms = capture_branch(
        parent,
        ctx,
        runtime,
        matched_prefix,
        row["targets"],
        target,
    )
    swapped_parent, swapped_rms = capture_branch(
        parent,
        ctx,
        runtime,
        swapped_prefix,
        row["targets"],
        target,
    )

    # Parent capture must have the target and already authenticates the
    # hidden in-projection branch reconstruction semantics.
    require(target in matched_parent and target in swapped_parent, "PARENT_CAPTURE_TARGET_MISSING")

    return role_decomposition(
        row,
        matched_rms,
        swapped_rms,
        runtime,
        parent_items[idx],
    )


def synthetic_factorization_check() -> dict[str, float]:
    import torch

    torch.manual_seed(71)
    gamma = torch.rand(HIDDEN_SIZE, dtype=torch.float64) + 0.5
    rm = torch.randn(HIDDEN_SIZE, dtype=torch.float64)
    rs = torch.randn(HIDDEN_SIZE, dtype=torch.float64)
    sm = 0.73
    ss = 0.69
    dr = rm - rs
    rbar = 0.5 * (rm + rs)
    sbar = 0.5 * (sm + ss)
    ds = sm - ss
    q_r = gamma * (sbar * dr)
    q_s = gamma * (rbar * ds)
    x_alg = gamma * (sm * rm - ss * rs)
    residual = float(torch.max(torch.abs((q_r + q_s) - x_alg)).item())
    require(residual <= VECTOR_ABS_TOL, f"SYNTHETIC_RMS_IDENTITY_FAILURE:{residual}")

    w = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.float64) * 0.01
    h_r = torch.mv(w, q_r)
    h_s = torch.mv(w, q_s)
    h_total = torch.mv(w, x_alg)
    dx_sq = float(torch.dot(x_alg, x_alg).item())
    e_r = h_r * h_r / dx_sq
    e_s = h_s * h_s / dx_sq
    e_rs = 2.0 * h_r * h_s / dx_sq
    e_total = h_total * h_total / dx_sq
    channel_residual = float(
        torch.max(torch.abs(e_total - (e_r + e_s + e_rs))).item()
    )
    require(
        channel_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_CHANNEL_IDENTITY_FAILURE:{channel_residual}",
    )
    return {
        "rms_identity_max_abs_residual": residual,
        "channel_identity_max_abs_residual": channel_residual,
    }


def print_plan(repo, parent_summary, parent_manifest, synth) -> None:
    print("=== LAYER22 CURRENT-TOKEN RMSNORM ROUTING-SOURCE AUDIT PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print("parent_evidence_freeze_commit =", PARENT_EVIDENCE_FREEZE_COMMIT)
    print("parent_implementation_commit =", PARENT_IMPLEMENTATION_COMMIT)
    print("parent_runner_sha256 =", PARENT_RUNNER_SHA256)
    print("parent_item_sha256 =", PARENT_ITEM_SHA256)
    print("parent_channel_sha256 =", PARENT_CHANNEL_SHA256)
    print("parent_cumulative_sha256 =", PARENT_CUMULATIVE_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("parent_report_sha256 =", PARENT_REPORT_SHA256)
    print("rms_precedent_freeze_commit =", RMS_PRECEDENT_FREEZE_COMMIT)
    print("rms_precedent_sha256 =", RMS_PRECEDENT_SHA256)
    print("pair_role_count =", EXPECTED_PAIR_ROLE_COUNT)
    print("common_ddsssss_item_count =", EXPECTED_COMMON_COUNT)
    print("source_layer =", SOURCE_LAYER)
    print("relative_coordinate =", TARGET_K)
    print("hidden_size =", HIDDEN_SIZE)
    print("intermediate_size =", INTERMEDIATE_SIZE)
    print("rms_eps =", RMS_EPS)
    print("scientific_question =", QUESTION)
    print("identity_delta_x = Q_R + Q_s + Q_eps")
    print("identity_parent_D = D_R + D_s + D_Rs + D_eps")
    print("strong_kernel_channel_count =", parent_summary["strong_kernel_channel_count"])
    print("weak_kernel_channel_count =", parent_summary["weak_kernel_channel_count"])
    print("expected_lag0_kernel_rms =", parent_summary["lag0_kernel_rms"])
    print("rms_branch_reconstruction_rel_tol =", RMS_BRANCH_RECON_REL_TOL)
    print("difference_reconstruction_relative_residual_gate = diagnostic_only")
    print("error_difference_identity_abs_tol =", VECTOR_ABS_TOL)
    print("channel_component_abs_tol =", VECTOR_ABS_TOL)
    print("squared_energy_rel_tol =", SQUARED_ENERGY_REL_TOL)
    print("parent_scalar_rel_tol =", PARENT_SCALAR_REL_TOL)
    print("parent_scalar_abs_tol =", PARENT_SCALAR_ABS_TOL)
    print("parent_runtime_head =", parent_manifest["runtime_git_head"])
    print("synthetic_rms_identity_max_abs_residual =", synth["rms_identity_max_abs_residual"])
    print("synthetic_channel_identity_max_abs_residual =", synth["channel_identity_max_abs_residual"])
    print("all_1536_channels_preregistered = True")
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("learned_geometry_executed = False")
    print("posthoc_search_executed = False")


def runtime_preflight(
    root: Path,
    parent: Any,
    ctx: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
    parent_channels: Mapping[int, Mapping[str, Any]],
    handoff_path: Path,
):
    runtime = resolve_runtime(root, parent, ctx, handoff_path)
    authenticate_operator_and_norm(runtime, parent_channels)

    corr_row, ctrl_row = _first_common_pair_rows(ctx["plan"], ctx["cohort"])
    records = {}
    forward_count = 0
    for row in (corr_row, ctrl_row):
        rec = run_role(row, parent, ctx, runtime, parent_items)
        records[str(row["role"])] = rec
        forward_count += 2

    require(forward_count == EXPECTED_PREFLIGHT_FORWARD_COUNT, "PREFLIGHT_FORWARD_COUNT_MISMATCH")
    require(set(records) == {"corr", "ctrl"}, "PREFLIGHT_ROLE_SET_MISMATCH")
    idx = int(records["corr"]["idx"])
    item, _ = pair_role_records(
        records["corr"],
        records["ctrl"],
        runtime,
        parent_items[idx],
    )

    print("PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_RUNTIME_PREFLIGHT")
    print("local_template_index =", idx)
    print("model_forward_count =", forward_count)
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("raw_vectors_persisted = False")
    print("lag0_kernel_rms =", runtime["operator"]["rms"])
    print("strong_kernel_channel_count =", int(runtime["operator"]["strong_mask"].sum().item()))
    print("weak_kernel_channel_count =", int(runtime["operator"]["weak_mask"].sum().item()))
    for role in ("corr", "ctrl"):
        print(f"delta_r_l2_{role} =", item[f"delta_r_l2_{role}"])
        print(f"delta_x_l2_{role} =", item[f"delta_x_l2_{role}"])
        print(f"delta_rms_scale_{role} =", item[f"delta_rms_scale_{role}"])
        print(f"q_r_l2_{role} =", item[f"q_r_l2_{role}"])
        print(f"q_s_l2_{role} =", item[f"q_s_l2_{role}"])
        print(f"q_eps_l2_{role} =", item[f"q_eps_l2_{role}"])
        print(f"input_energy_i_r_{role} =", item[f"input_energy_i_r_{role}"])
        print(f"input_energy_i_s_{role} =", item[f"input_energy_i_s_{role}"])
        print(f"input_energy_i_rs_{role} =", item[f"input_energy_i_rs_{role}"])
        print(f"input_energy_i_eps_{role} =", item[f"input_energy_i_eps_{role}"])
    for part in ("all", "strong", "weak"):
        for key in ("total", "r", "s", "rs", "eps"):
            print(f"delta_{key}_{part} =", item[f"delta_{key}_{part}"])
    print(
        "max_rms_branch_reconstruction_relative_residual =",
        max(
            item["rms_branch_reconstruction_relative_residual_corr"],
            item["rms_branch_reconstruction_relative_residual_ctrl"],
        ),
    )
    print(
        "max_difference_reconstruction_relative_residual_diagnostic =",
        max(
            item["difference_reconstruction_relative_residual_diagnostic_corr"],
            item["difference_reconstruction_relative_residual_diagnostic_ctrl"],
        ),
    )
    print(
        "max_error_difference_identity_abs_residual =",
        max(
            item["error_difference_identity_abs_residual_corr"],
            item["error_difference_identity_abs_residual_ctrl"],
        ),
    )
    print(
        "pair_channel_component_closure_max_abs_residual =",
        item["pair_channel_component_closure_max_abs_residual"],
    )
    print(
        "max_parent_partition_reproduction_abs_residual =",
        item["max_parent_partition_reproduction_abs_residual"],
    )


def execute(
    root: Path,
    parent: Any,
    ctx: Mapping[str, Any],
    repo: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
    parent_channels: Mapping[int, Mapping[str, Any]],
    handoff_path: Path,
    output_dir: Path,
):
    import torch

    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    require(handoff_path.is_file(), f"HANDOFF_MISSING:{handoff_path}")
    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_EXISTS:{partial_dir}")

    runtime = resolve_runtime(root, parent, ctx, handoff_path)
    authenticate_operator_and_norm(runtime, parent_channels)

    role_records: dict[tuple[int, str], dict[str, Any]] = {}
    forward_count = 0
    for n, row in enumerate(ctx["plan"], start=1):
        idx = int(row["local_template_index"])
        rec = run_role(row, parent, ctx, runtime, parent_items) if idx in ctx["cohort"] else None

        # Non-common rows still require the exact frozen execution plan and two
        # forwards, but need no scientific vector persistence.  Execute their
        # matched/swapped parent captures directly.
        if idx not in ctx["cohort"]:
            matched_prefix, swapped_prefix = _prefixes(parent, ctx, row)
            target = int(row["anchor"]) + TARGET_K
            capture_branch(
                parent, ctx, runtime, matched_prefix, row["targets"], target
            )
            capture_branch(
                parent, ctx, runtime, swapped_prefix, row["targets"], target
            )

        forward_count += 2
        if rec is not None:
            key = (idx, str(row["role"]))
            require(key not in role_records, f"ROLE_RECORD_DUPLICATE:{key}")
            role_records[key] = rec

        if n % 16 == 0 or n == len(ctx["plan"]):
            print(
                f"PROGRESS pair_roles={n}/{len(ctx['plan'])} model_forwards={forward_count}",
                flush=True,
            )

    require(forward_count == EXPECTED_FULL_FORWARD_COUNT, "FORWARD_COUNT_MISMATCH")
    require(
        len(role_records) == EXPECTED_COMMON_COUNT * 2,
        "COMMON_ROLE_RECORD_COUNT_MISMATCH",
    )

    item_rows = []
    vector_lists = {k: [] for k in ("total", "r", "s", "rs", "eps")}
    for idx in sorted(ctx["cohort"]):
        corr = role_records[(int(idx), "corr")]
        ctrl = role_records[(int(idx), "ctrl")]
        item, vectors = pair_role_records(corr, ctrl, runtime, parent_items[int(idx)])
        item_rows.append(item)
        for key in vector_lists:
            vector_lists[key].append(vectors[key])

    component_matrices = {
        key: torch.stack(values, dim=0)
        for key, values in vector_lists.items()
    }
    channel_rows, cumulative_rows = build_channel_outputs(
        component_matrices,
        runtime,
        parent_channels,
    )
    attribution = component_attribution(item_rows)
    summary = make_summary(item_rows, runtime, attribution)

    expected_parent_counts = {
        "p_s": 328,
        "t_h": 243,
        "t_s": 330,
        "t_w": 76,
    }
    for field, expected_gt in expected_parent_counts.items():
        got = summary["parent_paired_counts"][field]["corr_gt_ctrl"]
        require(got == expected_gt, f"PARENT_PAIRED_COUNT_MISMATCH:{field}:{got}:{expected_gt}")

    parent_mean_targets = {
        "all": 1.1589093253784961,
        "strong": 2.3029343955594896,
        "weak": -1.144025070180993,
    }
    for part, expected in parent_mean_targets.items():
        got = attribution[part]["mean_signed_components"]["total"]
        require(
            math.isclose(
                got,
                expected,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_MEAN_TOTAL_MISMATCH:{part}:{got}:{expected}",
        )

    partial_dir.mkdir(parents=True, exist_ok=False)
    item_path = partial_dir / "layer22_current_token_rmsnorm_routing_source_item_metrics.jsonl"
    channel_path = partial_dir / "layer22_current_token_rmsnorm_routing_source_channel_summary.jsonl"
    cumulative_path = partial_dir / "layer22_current_token_rmsnorm_routing_source_kernel_rank_cumulative.jsonl"
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    item_path.write_bytes(jsonl_bytes(item_rows))
    channel_path.write_bytes(jsonl_bytes(channel_rows))
    cumulative_path.write_bytes(jsonl_bytes(cumulative_rows))
    summary_path.write_bytes(json_bytes(summary))

    runner_sha = sha256_file(root / RUNNER_REL)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "runtime_git_head": repo["head"],
        "runtime_branch": repo["branch"],
        "runner_rel": RUNNER_REL,
        "runner_sha256": runner_sha,
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "authority_blob": AUTHORITY_BLOB,
        "parent_evidence_freeze_commit": PARENT_EVIDENCE_FREEZE_COMMIT,
        "parent_implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
        "parent_runner_blob": PARENT_RUNNER_BLOB,
        "parent_item_sha256": PARENT_ITEM_SHA256,
        "parent_channel_sha256": PARENT_CHANNEL_SHA256,
        "parent_cumulative_sha256": PARENT_CUMULATIVE_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "rms_precedent_freeze_commit": RMS_PRECEDENT_FREEZE_COMMIT,
        "rms_precedent_sha256": RMS_PRECEDENT_SHA256,
        "scientific_question": QUESTION,
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_layer": SOURCE_LAYER,
        "relative_coordinate": TARGET_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "rms_epsilon": RMS_EPS,
        "lag0_kernel_rms": float(runtime["operator"]["rms"]),
        "strong_kernel_channel_count": int(runtime["operator"]["strong_mask"].sum().item()),
        "weak_kernel_channel_count": int(runtime["operator"]["weak_mask"].sum().item()),
        "equal_kernel_channel_count": int(runtime["operator"]["equal_mask"].sum().item()),
        "mamba_source_sha256": runtime["binding"].source_sha256,
        "rms_branch_reconstruction_rel_tol": RMS_BRANCH_RECON_REL_TOL,
        "difference_reconstruction_relative_residual_is_diagnostic_only": True,
        "error_difference_identity_abs_tol": VECTOR_ABS_TOL,
        "channel_component_abs_tol": VECTOR_ABS_TOL,
        "squared_energy_rel_tol": SQUARED_ENERGY_REL_TOL,
        "parent_scalar_rel_tol": PARENT_SCALAR_REL_TOL,
        "parent_scalar_abs_tol": PARENT_SCALAR_ABS_TOL,
        "kernel_rms_rel_tol": KERNEL_RMS_REL_TOL,
        "kernel_rms_abs_tol": KERNEL_RMS_ABS_TOL,
        "handoff_zip_sha256": runtime["handoff"]["zip_sha256"],
        "checkpoint_sha256": runtime["handoff"]["checkpoint_sha256"],
        "encoder_canonical_digest": runtime["encoder"]["canonical_digest"],
        "encoder_raw_concat_digest": runtime["encoder"]["raw_concat_digest"],
        "parent_current_token_routing_match": True,
        "rmsnorm_parent_x_exact_match": True,
        "raw_vectors_persisted": False,
        "raw_per_item_channel_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed": False,
        "capture_method": (
            "read-only layer-22 RMSNorm pre/post hooks capture current-token R/X "
            "during the frozen parent U-path capture; RMS output X must exactly "
            "equal the frozen parent mixer-input X in float32; scientific "
            "Q_R/Q_s/Q_eps algebra and W_H propagation are evaluated in float64"
        ),
        "outputs": {
            item_path.name: sha256_file(item_path),
            channel_path.name: sha256_file(channel_path),
            cumulative_path.name: sha256_file(cumulative_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    print("PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_EXECUTION")
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("parent_current_token_routing_match = True")
    print("rmsnorm_parent_x_exact_match = True")
    print("lag0_kernel_rms =", runtime["operator"]["rms"])
    print("strong_kernel_channel_count =", int(runtime["operator"]["strong_mask"].sum().item()))
    print("weak_kernel_channel_count =", int(runtime["operator"]["weak_mask"].sum().item()))
    for part in ("all", "strong", "weak"):
        c = attribution[part]["mean_signed_components"]
        print(
            f"{part}_mean_components "
            f"total={c['total']} residual={c['r']} scale={c['s']} "
            f"interaction={c['rs']} numerical_bridge={c['eps']}"
        )
        print(
            f"{part}_largest_absolute_scientific_component =",
            attribution[part]["largest_absolute_scientific_component"],
        )
    print(
        "max_rms_branch_reconstruction_relative_residual =",
        summary["max_rms_branch_reconstruction_relative_residual"],
    )
    print(
        "max_difference_reconstruction_relative_residual_diagnostic =",
        summary["max_difference_reconstruction_relative_residual_diagnostic"],
    )
    print(
        "max_error_difference_identity_abs_residual =",
        summary["max_error_difference_identity_abs_residual"],
    )
    print(
        "max_input_energy_closure_abs_residual =",
        summary["max_input_energy_closure_abs_residual"],
    )
    print(
        "max_channel_component_closure_abs_residual =",
        summary["max_channel_component_closure_abs_residual"],
    )
    print(
        "max_parent_partition_reproduction_abs_residual =",
        summary["max_parent_partition_reproduction_abs_residual"],
    )


def parse_args():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-preflight", action="store_true")
    mode.add_argument("--runtime-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--handoff", type=Path)
    p.add_argument("--output-dir", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime_mode = bool(args.runtime_preflight or args.execute)

    repo = authenticate_repo(root, runtime_mode=runtime_mode)
    authenticate_authority(root)
    authenticate_rms_precedent(root)
    parent = load_parent(root)
    parent_summary, parent_manifest, parent_items, parent_channels = (
        authenticate_parent_evidence(root)
    )
    ctx = build_parent_context(root, parent)
    synth = synthetic_factorization_check()
    print_plan(repo, parent_summary, parent_manifest, synth)

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print("PASS_LAYER22_CURRENT_TOKEN_RMSNORM_ROUTING_SOURCE_STATIC_PREFLIGHT")
        return 0

    require(args.handoff is not None, "HANDOFF_REQUIRED")
    handoff = args.handoff.expanduser().resolve()
    require(handoff.is_file(), f"HANDOFF_MISSING:{handoff}")

    if args.runtime_preflight:
        require(args.output_dir is None, "RUNTIME_PREFLIGHT_OUTPUT_DIR_FORBIDDEN")
        runtime_preflight(
            root,
            parent,
            ctx,
            parent_items,
            parent_channels,
            handoff,
        )
        return 0

    require(args.execute, "UNKNOWN_MODE")
    require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
    execute(
        root,
        parent,
        ctx,
        repo,
        parent_items,
        parent_channels,
        handoff,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RMSNormRoutingSourceError as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        raise SystemExit(2)
