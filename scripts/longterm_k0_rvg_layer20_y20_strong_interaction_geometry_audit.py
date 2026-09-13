#!/usr/bin/env python3
"""K0-RVG layer20->21 frozen-strong interaction geometry localization.

Frozen target from the immediate parent:
    D_ry20,strong = corr - ctrl contrast of
        2 * <H_r20,strong, H_y20,strong> / ||delta-X22||^2

For each role define:
    x = H_r20,strong / sqrt(||delta-X22||^2)
    y = H_y20,strong / sqrt(||delta-X22||^2)
    A = ||x||, B = ||y||, C = cos(x,y), M = 2AB
    I = 2<x,y> = 2ABC = MC

For one paired item the exact midpoint identity is:
    delta-I = Q_A + Q_B + Q_C
    Q_A = 2*C_bar*B_bar*delta-A
    Q_B = 2*C_bar*A_bar*delta-B
    Q_C = M_bar*delta-C

This audit is observational/algebraic only.  It preserves the frozen 240-channel
strong partition, fixed layer22 parent map, checkpoint, handoff, and execution
protocol.  No raw vectors are persisted.
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

AUTHORITY_FREEZE_COMMIT = "2b239beb646c681f2032b6cfcaca15a34d25ad34"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer20_y20_strong_interaction_geometry_"
    "static_design_candidate.md"
)
AUTHORITY_SHA256 = "b5a35ede12b17e898f2d0f2632e303b58e702fc08963096b00157668e2d3425c"
AUTHORITY_BLOB = "38a8063f89c2332ca04210fa92076ec502ebd60e"

IMMEDIATE_PARENT_EVIDENCE_FREEZE = "f5ef304701c494b83b070a566265d652c03444dd"
IMMEDIATE_PARENT_IMPLEMENTATION = "845827d3d99de5fdf5409b901b7039197cf1c08e"
IMMEDIATE_PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer21_current_token_residual_construction_"
    "routing_source_audit.py"
)
IMMEDIATE_PARENT_RUNNER_SHA256 = (
    "788cd3b64883d2d4f8733f0787e0a44de25952e14cf48251b288ac36354388b8"
)
IMMEDIATE_PARENT_RUNNER_BLOB = "f52a0f0a25533ee14a0161cd550e88cfca052dc8"

IMMEDIATE_PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer21_current_token_residual_construction_"
    "routing_source_845827d_v1"
)
IMMEDIATE_PARENT_ITEM_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer21_current_token_residual_construction_routing_source_item_metrics.jsonl"
)
IMMEDIATE_PARENT_ITEM_SHA256 = (
    "58be748a721be37971f45bb4f5a2996f240fe91a0cd6ba99f33c0985b7538470"
)
IMMEDIATE_PARENT_CHANNEL_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer21_current_token_residual_construction_routing_source_channel_summary.jsonl"
)
IMMEDIATE_PARENT_CHANNEL_SHA256 = (
    "23f19995bf06d7fd10048e13411cbd0d4ec75676a99ae9d7032889f24fcb6537"
)
IMMEDIATE_PARENT_CUMULATIVE_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer21_current_token_residual_construction_routing_source_kernel_rank_cumulative.jsonl"
)
IMMEDIATE_PARENT_CUMULATIVE_SHA256 = (
    "81845540f4da7d5a43e1dcfa7893177a9ad88b619de8dff4a89e1be10477be4a"
)
IMMEDIATE_PARENT_SUMMARY_REL = IMMEDIATE_PARENT_RUN_DIR + "/summary.json"
IMMEDIATE_PARENT_SUMMARY_SHA256 = (
    "d65af91038d05bf3c92533ff8b48a0d495bd862dcbb779813bf56d337ab1855e"
)
IMMEDIATE_PARENT_MANIFEST_REL = IMMEDIATE_PARENT_RUN_DIR + "/execution_manifest.json"
IMMEDIATE_PARENT_MANIFEST_SHA256 = (
    "5268703eb069ad2d385bf15a2f9456adfd583a9fae5fda97c0e646ee05cc2300"
)
IMMEDIATE_PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer21_current_token_residual_construction_routing_source_"
    "validated_evidence_analysis_report_candidate.md"
)
IMMEDIATE_PARENT_REPORT_SHA256 = (
    "38b4bb275cac40cb2b20487f3b0fc378c269cce3a901b49566bc512cec4a0202"
)

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer20_y20_strong_interaction_geometry_audit.py"
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

SOURCE_BLOCK = 20
TARGET_RESIDUAL_LAYER = 21
PARENT_MAP_LAYER = 22
TARGET_K = 2
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536

EXPECTED_STRONG_COUNT = 240
EXPECTED_WEAK_COUNT = 1296
EXPECTED_EQUAL_COUNT = 0
EXPECTED_LAG0_KERNEL_RMS = 0.24383223809052498
EXPECTED_PARENT_STRONG_INTERACTION_MEAN = 0.760594723140676

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)
EXPECTED_BLOCK_FORWARD_SHA256 = (
    "0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4"
)
EXPECTED_BACKBONE_FORWARD_SHA256 = (
    "3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6"
)

EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

PARENT_SCALAR_REL_TOL = 1e-13
PARENT_SCALAR_ABS_TOL = 1e-13
VECTOR_ABS_TOL = 5e-12
CHANNEL_REPRO_ABS_TOL = 5e-12
COSINE_BOUNDARY_SLACK = 1e-12
KERNEL_RMS_REL_TOL = 1e-13
KERNEL_RMS_ABS_TOL = 1e-13

QUESTION = (
    "Why is the validated R20xY20 interaction contrast so strongly positive "
    "in the frozen strong-kernel partition: larger parent-normalized R20 "
    "projected magnitude, larger parent-normalized Y20 projected magnitude, "
    "stronger cosine alignment, or a genuine mixture?"
)

ITEM_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-item-v1"
)
CHANNEL_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-channel-validation-v1"
)
SUMMARY_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-summary-v1"
)
MANIFEST_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-execution-manifest-v1"
)


class StrongInteractionGeometryError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StrongInteractionGeometryError(message)


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
        raise StrongInteractionGeometryError(
            f"GIT_FAILURE:{' '.join(args)}"
        ) from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StrongInteractionGeometryError(
            f"GIT_SHOW_FAILURE:{spec}"
        ) from exc


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
        (IMMEDIATE_PARENT_EVIDENCE_FREEZE, "IMMEDIATE_PARENT_EVIDENCE"),
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
        raise StrongInteractionGeometryError(
            f"UNEXPECTED_WORKTREE_CHANGE:{line}"
        )

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
    expected_sha: str | None,
    label: str,
    expected_blob: str | None = None,
) -> bytes:
    frozen = git_bytes(root, f"{commit}:{rel}")
    if expected_sha is not None:
        require(
            sha256_bytes(frozen) == expected_sha,
            f"{label}_SHA256_MISMATCH",
        )
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


def load_immediate_parent(root: Path):
    _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_RUNNER_REL,
        IMMEDIATE_PARENT_RUNNER_SHA256,
        "IMMEDIATE_PARENT_RUNNER",
        IMMEDIATE_PARENT_RUNNER_BLOB,
    )
    return import_module(
        root / IMMEDIATE_PARENT_RUNNER_REL,
        "k0_rvg_layer21_residual_construction_parent",
    )


def _load_json(raw: bytes, label: str):
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise StrongInteractionGeometryError(
            f"{label}_JSON_PARSE_FAILURE"
        ) from exc
    require(isinstance(obj, dict), f"{label}_NOT_OBJECT")
    return obj


def authenticate_immediate_parent_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    item_raw = _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_ITEM_REL,
        IMMEDIATE_PARENT_ITEM_SHA256,
        "IMMEDIATE_PARENT_ITEM",
    )
    channel_raw = _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_CHANNEL_REL,
        IMMEDIATE_PARENT_CHANNEL_SHA256,
        "IMMEDIATE_PARENT_CHANNEL",
    )
    _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_CUMULATIVE_REL,
        IMMEDIATE_PARENT_CUMULATIVE_SHA256,
        "IMMEDIATE_PARENT_CUMULATIVE",
    )
    summary_raw = _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_SUMMARY_REL,
        IMMEDIATE_PARENT_SUMMARY_SHA256,
        "IMMEDIATE_PARENT_SUMMARY",
    )
    manifest_raw = _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_MANIFEST_REL,
        IMMEDIATE_PARENT_MANIFEST_SHA256,
        "IMMEDIATE_PARENT_MANIFEST",
    )
    _authenticate_file(
        root,
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        IMMEDIATE_PARENT_REPORT_REL,
        IMMEDIATE_PARENT_REPORT_SHA256,
        "IMMEDIATE_PARENT_REPORT",
    )

    summary = _load_json(summary_raw, "IMMEDIATE_PARENT_SUMMARY")
    manifest = _load_json(manifest_raw, "IMMEDIATE_PARENT_MANIFEST")

    require(
        summary.get("schema_version")
        == "k0-rvg-layer21-current-token-residual-construction-routing-source-summary-v1",
        "IMMEDIATE_PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer21-current-token-residual-construction-routing-source-execution-manifest-v1",
        "IMMEDIATE_PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("runtime_git_head") == IMMEDIATE_PARENT_IMPLEMENTATION,
        "IMMEDIATE_PARENT_RUNTIME_HEAD_MISMATCH",
    )
    require(
        manifest.get("runner_sha256") == IMMEDIATE_PARENT_RUNNER_SHA256,
        "IMMEDIATE_PARENT_MANIFEST_RUNNER_MISMATCH",
    )
    require(
        manifest.get("model_forward_count") == EXPECTED_FULL_FORWARD_COUNT,
        "IMMEDIATE_PARENT_FORWARD_COUNT_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_COMMON_COUNT,
        "IMMEDIATE_PARENT_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("strong_kernel_channel_count") == EXPECTED_STRONG_COUNT,
        "IMMEDIATE_PARENT_STRONG_COUNT_MISMATCH",
    )
    require(
        summary.get("weak_kernel_channel_count") == EXPECTED_WEAK_COUNT,
        "IMMEDIATE_PARENT_WEAK_COUNT_MISMATCH",
    )
    require(
        summary.get("equal_kernel_channel_count") == EXPECTED_EQUAL_COUNT,
        "IMMEDIATE_PARENT_EQUAL_COUNT_MISMATCH",
    )

    got_parent_mean = float(
        summary["component_attribution"]["strong"][
            "mean_signed_components"
        ]["ry20"]
    )
    require(
        math.isclose(
            got_parent_mean,
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
            rel_tol=PARENT_SCALAR_REL_TOL,
            abs_tol=PARENT_SCALAR_ABS_TOL,
        ),
        "IMMEDIATE_PARENT_STRONG_INTERACTION_MEAN_MISMATCH",
    )

    items: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(item_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise StrongInteractionGeometryError(
                f"IMMEDIATE_PARENT_ITEM_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        idx = int(row["local_template_index"])
        require(idx not in items, f"IMMEDIATE_PARENT_ITEM_DUPLICATE:{idx}")
        require("delta_ry20_strong" in row, f"IMMEDIATE_PARENT_ITEM_FIELD_MISSING:{idx}")
        items[idx] = row
    require(len(items) == EXPECTED_COMMON_COUNT, "IMMEDIATE_PARENT_ITEM_COUNT_MISMATCH")

    channels: dict[int, dict[str, Any]] = {}
    strong_count = 0
    for line_no, line in enumerate(channel_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise StrongInteractionGeometryError(
                f"IMMEDIATE_PARENT_CHANNEL_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        j = int(row["channel_index"])
        require(j not in channels, f"IMMEDIATE_PARENT_CHANNEL_DUPLICATE:{j}")
        require("mean_d_ry20" in row, f"IMMEDIATE_PARENT_CHANNEL_FIELD_MISSING:{j}")
        if row["downstream_partition"] == "strong":
            strong_count += 1
        channels[j] = row
    require(
        len(channels) == INTERMEDIATE_SIZE,
        "IMMEDIATE_PARENT_CHANNEL_COUNT_MISMATCH",
    )
    require(
        strong_count == EXPECTED_STRONG_COUNT,
        "IMMEDIATE_PARENT_STRONG_CHANNEL_COUNT_MISMATCH",
    )

    return summary, manifest, items, channels


def build_parent_stack(root: Path, parent: Any) -> dict[str, Any]:
    # Authenticate every frozen dependency used by the immediate-parent runner.
    parent.authenticate_authority(root)
    grandparent = parent.load_immediate_parent(root)
    (
        grandparent_summary,
        grandparent_manifest,
        grandparent_items,
        grandparent_channels,
        grandparent_cumulative,
    ) = parent.authenticate_immediate_parent_evidence(root)
    stack = parent.build_parent_stack(root, grandparent)
    return {
        "grandparent": grandparent,
        "grandparent_summary": grandparent_summary,
        "grandparent_manifest": grandparent_manifest,
        "grandparent_items": grandparent_items,
        "grandparent_channels": grandparent_channels,
        "grandparent_cumulative": grandparent_cumulative,
        "stack": stack,
    }


def resolve_runtime(
    root: Path,
    parent: Any,
    parent_stack: Mapping[str, Any],
    handoff_path: Path,
) -> dict[str, Any]:
    runtime = parent.resolve_runtime(
        root,
        parent_stack["grandparent"],
        parent_stack["stack"],
        handoff_path,
    )

    model = runtime["model"]
    backbone = model.mamba
    require(len(backbone.layers) == EXPECTED_LAYER_COUNT, "BACKBONE_LAYER_COUNT_MISMATCH")
    require(
        runtime["binding"].source_sha256 == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )

    op = runtime["operator"]
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

    return runtime


def _prefixes(
    parent: Any,
    parent_stack: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return parent._prefixes(
        parent_stack["grandparent"],
        parent_stack["stack"],
        row,
    )


def run_branch(
    row: Mapping[str, Any],
    parent: Any,
    parent_stack: Mapping[str, Any],
    runtime: Mapping[str, Any],
    token_ids: Sequence[int],
):
    return parent.run_branch(
        row,
        parent_stack["grandparent"],
        parent_stack["stack"],
        runtime,
        token_ids,
    )


def _safe_float(value: Any, label: str) -> float:
    out = float(value)
    require(math.isfinite(out), f"NONFINITE:{label}")
    return out


def role_geometry(
    row: Mapping[str, Any],
    parent: Any,
    parent_stack: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])

    matched_prefix, swapped_prefix = _prefixes(
        parent,
        parent_stack,
        row,
    )

    (
        _matched_parent_records,
        matched_rms,
        _matched_layer21,
        matched_layer20,
    ) = run_branch(
        row,
        parent,
        parent_stack,
        runtime,
        matched_prefix,
    )

    (
        _swapped_parent_records,
        swapped_rms,
        _swapped_layer21,
        swapped_layer20,
    ) = run_branch(
        row,
        parent,
        parent_stack,
        runtime,
        swapped_prefix,
    )

    gamma = (
        runtime["norm22"].weight.detach().cpu().to(torch.float64).contiguous()
    )
    w = runtime["operator"]["w_hidden64"]
    strong_mask = runtime["operator"]["strong_mask"]

    r20m = matched_layer20["R20"].to(torch.float64).contiguous()
    r20s = swapped_layer20["R20"].to(torch.float64).contiguous()
    y20m = matched_layer20["Y20"].to(torch.float64).contiguous()
    y20s = swapped_layer20["Y20"].to(torch.float64).contiguous()

    xm = matched_rms["X32"].to(torch.float64).contiguous()
    xs = swapped_rms["X32"].to(torch.float64).contiguous()
    sm = float(matched_rms["rms_scale32"].item())
    ss = float(swapped_rms["rms_scale32"].item())
    sbar = 0.5 * (sm + ss)

    dr20 = r20m - r20s
    dy20 = y20m - y20s
    x_obs = xm - xs
    dx_sq = float(torch.dot(x_obs, x_obs).item())
    require(dx_sq > 0.0, f"DELTA_X22_ZERO:{idx}:{role}")

    q_r20 = gamma * (sbar * dr20)
    q_y20 = gamma * (sbar * dy20)

    h_r20 = torch.mv(w, q_r20)
    h_y20 = torch.mv(w, q_y20)

    denom = math.sqrt(dx_sq)
    x = (h_r20[strong_mask] / denom).contiguous()
    y = (h_y20[strong_mask] / denom).contiguous()

    require(
        int(x.numel()) == EXPECTED_STRONG_COUNT,
        f"STRONG_X_WIDTH_MISMATCH:{idx}:{role}",
    )
    require(
        int(y.numel()) == EXPECTED_STRONG_COUNT,
        f"STRONG_Y_WIDTH_MISMATCH:{idx}:{role}",
    )

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0, f"ZERO_A:{idx}:{role}")
    require(b > 0.0, f"ZERO_B:{idx}:{role}")

    dot = float(torch.dot(x, y).item())
    c = dot / (a * b)
    require(
        -1.0 - COSINE_BOUNDARY_SLACK <= c <= 1.0 + COSINE_BOUNDARY_SLACK,
        f"COSINE_BOUNDARY_FAILURE:{idx}:{role}:{c}",
    )

    interaction_vec = 2.0 * x * y
    i_dot = 2.0 * dot
    i_channel = float(torch.sum(interaction_vec).item())
    i_abc = 2.0 * a * b * c
    m = 2.0 * a * b

    dot_channel_residual = abs(i_dot - i_channel)
    abc_residual = abs(i_dot - i_abc)

    require(
        dot_channel_residual <= VECTOR_ABS_TOL,
        f"ROLE_INTERACTION_CHANNEL_SUM_FAILURE:"
        f"{idx}:{role}:{dot_channel_residual}",
    )
    require(
        abc_residual <= VECTOR_ABS_TOL,
        f"ROLE_INTERACTION_ABC_FAILURE:{idx}:{role}:{abc_residual}",
    )

    positive = float(torch.sum(torch.clamp(interaction_vec, min=0.0)).item())
    negative = float(torch.sum(torch.clamp(interaction_vec, max=0.0)).item())
    mass_residual = abs(i_channel - (positive + negative))
    require(
        mass_residual <= VECTOR_ABS_TOL,
        f"ROLE_SIGNED_MASS_CLOSURE_FAILURE:{idx}:{role}:{mass_residual}",
    )

    positive_count = int(torch.sum(interaction_vec > 0.0).item())
    negative_count = int(torch.sum(interaction_vec < 0.0).item())
    zero_count = int(torch.sum(interaction_vec == 0.0).item())
    require(
        positive_count + negative_count + zero_count == EXPECTED_STRONG_COUNT,
        f"ROLE_SIGN_COUNT_FAILURE:{idx}:{role}",
    )

    return {
        "idx": idx,
        "stable_item_id": row["stable_item_id"],
        "role": role,
        "A": a,
        "B": b,
        "C": c,
        "M": m,
        "I": i_dot,
        "positive_interaction_mass": positive,
        "negative_interaction_mass": negative,
        "same_sign_channel_count": positive_count,
        "opposite_sign_channel_count": negative_count,
        "zero_product_channel_count": zero_count,
        "interaction_channel_sum_abs_residual": dot_channel_residual,
        "interaction_abc_abs_residual": abc_residual,
        "signed_mass_closure_abs_residual": mass_residual,
        "interaction_vec": interaction_vec,
    }


def pair_geometry(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    parent_item: Mapping[str, Any],
):
    import torch

    require(corr["idx"] == ctrl["idx"], "PAIR_INDEX_MISMATCH")
    require(
        corr["stable_item_id"] == ctrl["stable_item_id"],
        "PAIR_STABLE_ID_MISMATCH",
    )
    idx = int(corr["idx"])

    ac = _safe_float(corr["A"], f"A_CORR:{idx}")
    at = _safe_float(ctrl["A"], f"A_CTRL:{idx}")
    bc = _safe_float(corr["B"], f"B_CORR:{idx}")
    bt = _safe_float(ctrl["B"], f"B_CTRL:{idx}")
    cc = _safe_float(corr["C"], f"C_CORR:{idx}")
    ct = _safe_float(ctrl["C"], f"C_CTRL:{idx}")
    ic = _safe_float(corr["I"], f"I_CORR:{idx}")
    it = _safe_float(ctrl["I"], f"I_CTRL:{idx}")

    delta_a = ac - at
    delta_b = bc - bt
    delta_c = cc - ct

    a_bar = 0.5 * (ac + at)
    b_bar = 0.5 * (bc + bt)
    c_bar = 0.5 * (cc + ct)

    mc = 2.0 * ac * bc
    mt = 2.0 * at * bt
    m_bar = 0.5 * (mc + mt)
    delta_m = mc - mt

    delta_i = ic - it

    q_a = 2.0 * c_bar * b_bar * delta_a
    q_b = 2.0 * c_bar * a_bar * delta_b
    q_c = m_bar * delta_c

    magnitude_identity_residual = abs(
        delta_m
        - (
            2.0 * b_bar * delta_a
            + 2.0 * a_bar * delta_b
        )
    )
    first_level_residual = abs(
        delta_i
        - (
            c_bar * delta_m
            + m_bar * delta_c
        )
    )
    geometry_residual = abs(delta_i - (q_a + q_b + q_c))

    require(
        magnitude_identity_residual <= VECTOR_ABS_TOL,
        f"MAGNITUDE_MIDPOINT_IDENTITY_FAILURE:{idx}:{magnitude_identity_residual}",
    )
    require(
        first_level_residual <= VECTOR_ABS_TOL,
        f"FIRST_LEVEL_MIDPOINT_IDENTITY_FAILURE:{idx}:{first_level_residual}",
    )
    require(
        geometry_residual <= VECTOR_ABS_TOL,
        f"GEOMETRY_MIDPOINT_IDENTITY_FAILURE:{idx}:{geometry_residual}",
    )

    parent_target = float(parent_item["delta_ry20_strong"])
    parent_repro = abs(delta_i - parent_target)
    require(
        math.isclose(
            delta_i,
            parent_target,
            rel_tol=PARENT_SCALAR_REL_TOL,
            abs_tol=PARENT_SCALAR_ABS_TOL,
        ),
        f"PARENT_ITEM_STRONG_INTERACTION_REPRODUCTION_FAILURE:"
        f"{idx}:{delta_i}:{parent_target}",
    )

    delta_channel = corr["interaction_vec"] - ctrl["interaction_vec"]
    channel_sum_residual = abs(
        float(torch.sum(delta_channel).item()) - delta_i
    )
    require(
        channel_sum_residual <= VECTOR_ABS_TOL,
        f"PAIR_CHANNEL_SUM_FAILURE:{idx}:{channel_sum_residual}",
    )

    row = {
        "schema_version": ITEM_SCHEMA,
        "local_template_index": idx,
        "stable_item_id": corr["stable_item_id"],
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "parent_map_layer": PARENT_MAP_LAYER,
        "relative_coordinate": TARGET_K,
        "parent_delta_ry20_strong": parent_target,
        "reconstructed_delta_ry20_strong": delta_i,
        "parent_reproduction_abs_residual": parent_repro,
        "delta_A": delta_a,
        "delta_B": delta_b,
        "delta_C": delta_c,
        "A_bar": a_bar,
        "B_bar": b_bar,
        "C_bar": c_bar,
        "M_bar": m_bar,
        "delta_M": delta_m,
        "Q_A": q_a,
        "Q_B": q_b,
        "Q_C": q_c,
        "reconstructed_delta_I": delta_i,
        "magnitude_midpoint_identity_abs_residual": magnitude_identity_residual,
        "first_level_midpoint_identity_abs_residual": first_level_residual,
        "geometry_midpoint_identity_abs_residual": geometry_residual,
        "pair_channel_sum_abs_residual": channel_sum_residual,
        "raw_vectors_persisted": False,
    }

    for role_name, rec in (("corr", corr), ("ctrl", ctrl)):
        for field in (
            "A",
            "B",
            "C",
            "M",
            "I",
            "positive_interaction_mass",
            "negative_interaction_mass",
            "same_sign_channel_count",
            "opposite_sign_channel_count",
            "zero_product_channel_count",
            "interaction_channel_sum_abs_residual",
            "interaction_abc_abs_residual",
            "signed_mass_closure_abs_residual",
        ):
            row[f"{field}_{role_name}"] = rec[field]

    return row, {
        "delta_channel": delta_channel,
        "corr_channel": corr["interaction_vec"],
        "ctrl_channel": ctrl["interaction_vec"],
    }


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


def build_channel_validation(
    delta_matrix,
    corr_matrix,
    ctrl_matrix,
    runtime: Mapping[str, Any],
    parent_channels: Mapping[int, Mapping[str, Any]],
):
    import torch

    expected = (EXPECTED_COMMON_COUNT, EXPECTED_STRONG_COUNT)
    for label, matrix in (
        ("DELTA", delta_matrix),
        ("CORR", corr_matrix),
        ("CTRL", ctrl_matrix),
    ):
        require(
            tuple(matrix.shape) == expected,
            f"{label}_MATRIX_SHAPE_MISMATCH:{tuple(matrix.shape)}",
        )

    strong_indices = (
        torch.nonzero(runtime["operator"]["strong_mask"], as_tuple=False)
        .flatten()
        .tolist()
    )
    require(
        len(strong_indices) == EXPECTED_STRONG_COUNT,
        "STRONG_INDEX_COUNT_MISMATCH",
    )

    mean_delta = torch.mean(delta_matrix, dim=0)
    mean_corr = torch.mean(corr_matrix, dim=0)
    mean_ctrl = torch.mean(ctrl_matrix, dim=0)

    corr_pos = torch.sum(corr_matrix > 0.0, dim=0)
    corr_neg = torch.sum(corr_matrix < 0.0, dim=0)
    corr_zero = torch.sum(corr_matrix == 0.0, dim=0)
    ctrl_pos = torch.sum(ctrl_matrix > 0.0, dim=0)
    ctrl_neg = torch.sum(ctrl_matrix < 0.0, dim=0)
    ctrl_zero = torch.sum(ctrl_matrix == 0.0, dim=0)

    rows = []
    max_parent_repro = 0.0

    for local_j, channel_index in enumerate(strong_indices):
        p = parent_channels[int(channel_index)]
        require(
            p["downstream_partition"] == "strong",
            f"PARENT_CHANNEL_NOT_STRONG:{channel_index}",
        )
        parent_mean = float(p["mean_d_ry20"])
        new_mean = float(mean_delta[local_j].item())
        residual = abs(new_mean - parent_mean)
        require(
            math.isclose(
                new_mean,
                parent_mean,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=CHANNEL_REPRO_ABS_TOL,
            ),
            f"PARENT_STRONG_CHANNEL_REPRODUCTION_FAILURE:"
            f"{channel_index}:{new_mean}:{parent_mean}",
        )
        max_parent_repro = max(max_parent_repro, residual)

        require(
            int(corr_pos[local_j].item())
            + int(corr_neg[local_j].item())
            + int(corr_zero[local_j].item())
            == EXPECTED_COMMON_COUNT,
            f"CORR_CHANNEL_SIGN_COUNT_FAILURE:{channel_index}",
        )
        require(
            int(ctrl_pos[local_j].item())
            + int(ctrl_neg[local_j].item())
            + int(ctrl_zero[local_j].item())
            == EXPECTED_COMMON_COUNT,
            f"CTRL_CHANNEL_SIGN_COUNT_FAILURE:{channel_index}",
        )

        rows.append(
            {
                "schema_version": CHANNEL_SCHEMA,
                "channel_index": int(channel_index),
                "kernel_magnitude_rank": int(p["kernel_magnitude_rank"]),
                "downstream_partition": "strong",
                "parent_mean_d_ry20": parent_mean,
                "reconstructed_mean_d_ry20": new_mean,
                "parent_reproduction_abs_residual": residual,
                "corr_mean_signed_product": float(mean_corr[local_j].item()),
                "ctrl_mean_signed_product": float(mean_ctrl[local_j].item()),
                "corr_positive_product_count": int(corr_pos[local_j].item()),
                "corr_negative_product_count": int(corr_neg[local_j].item()),
                "corr_zero_product_count": int(corr_zero[local_j].item()),
                "ctrl_positive_product_count": int(ctrl_pos[local_j].item()),
                "ctrl_negative_product_count": int(ctrl_neg[local_j].item()),
                "ctrl_zero_product_count": int(ctrl_zero[local_j].item()),
            }
        )

    rows.sort(key=lambda r: r["channel_index"])

    channel_sum = float(
        math.fsum(r["reconstructed_mean_d_ry20"] for r in rows)
    )
    require(
        math.isclose(
            channel_sum,
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
            rel_tol=PARENT_SCALAR_REL_TOL,
            abs_tol=CHANNEL_REPRO_ABS_TOL,
        ),
        f"STRONG_CHANNEL_POPULATION_SUM_MISMATCH:{channel_sum}",
    )

    return rows, max_parent_repro, channel_sum


def component_attribution(item_rows) -> dict[str, Any]:
    means = {
        "total": float(
            statistics.fmean(
                float(r["reconstructed_delta_I"]) for r in item_rows
            )
        ),
        "Q_A": float(
            statistics.fmean(float(r["Q_A"]) for r in item_rows)
        ),
        "Q_B": float(
            statistics.fmean(float(r["Q_B"]) for r in item_rows)
        ),
        "Q_C": float(
            statistics.fmean(float(r["Q_C"]) for r in item_rows)
        ),
    }

    closure = abs(
        means["total"]
        - (means["Q_A"] + means["Q_B"] + means["Q_C"])
    )
    require(
        closure <= VECTOR_ABS_TOL,
        f"AGGREGATE_GEOMETRY_CLOSURE_FAILURE:{closure}",
    )

    mass = abs(means["Q_A"]) + abs(means["Q_B"]) + abs(means["Q_C"])
    shares = {
        k: (abs(means[k]) / mass if mass > 0.0 else None)
        for k in ("Q_A", "Q_B", "Q_C")
    }
    ratios = {
        k: (
            means[k] / means["total"]
            if means["total"] != 0.0
            else None
        )
        for k in ("Q_A", "Q_B", "Q_C")
    }
    largest = max(
        ("Q_A", "Q_B", "Q_C"),
        key=lambda k: abs(means[k]),
    )

    return {
        "mean_signed_components": means,
        "absolute_component_mass": mass,
        "absolute_component_shares": shares,
        "signed_component_to_total_net_ratios": ratios,
        "largest_absolute_scientific_component": largest,
        "aggregate_closure_abs_residual": closure,
    }


def make_summary(
    item_rows,
    attribution: Mapping[str, Any],
    runtime: Mapping[str, Any],
    max_parent_channel_repro: float,
    channel_sum: float,
) -> dict[str, Any]:
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "scientific_question": QUESTION,
        "common_ddsssss_item_count": len(item_rows),
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "parent_map_layer": PARENT_MAP_LAYER,
        "relative_coordinate": TARGET_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "strong_kernel_channel_count": int(
            runtime["operator"]["strong_mask"].sum().item()
        ),
        "weak_kernel_channel_count": int(
            runtime["operator"]["weak_mask"].sum().item()
        ),
        "equal_kernel_channel_count": int(
            runtime["operator"]["equal_mask"].sum().item()
        ),
        "lag0_kernel_rms": float(runtime["operator"]["rms"]),
        "expected_parent_strong_interaction_mean":
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
        "reconstructed_parent_strong_interaction_mean":
            attribution["mean_signed_components"]["total"],
        "strong_channel_population_sum": channel_sum,
        "component_attribution": attribution,
        "role_diagnostics": {},
        "max_parent_item_strong_interaction_reproduction_abs_residual": max(
            float(r["parent_reproduction_abs_residual"])
            for r in item_rows
        ),
        "max_role_interaction_channel_sum_abs_residual": max(
            max(
                float(r["interaction_channel_sum_abs_residual_corr"]),
                float(r["interaction_channel_sum_abs_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_role_interaction_abc_abs_residual": max(
            max(
                float(r["interaction_abc_abs_residual_corr"]),
                float(r["interaction_abc_abs_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_role_signed_mass_closure_abs_residual": max(
            max(
                float(r["signed_mass_closure_abs_residual_corr"]),
                float(r["signed_mass_closure_abs_residual_ctrl"]),
            )
            for r in item_rows
        ),
        "max_magnitude_midpoint_identity_abs_residual": max(
            float(r["magnitude_midpoint_identity_abs_residual"])
            for r in item_rows
        ),
        "max_first_level_midpoint_identity_abs_residual": max(
            float(r["first_level_midpoint_identity_abs_residual"])
            for r in item_rows
        ),
        "max_geometry_midpoint_identity_abs_residual": max(
            float(r["geometry_midpoint_identity_abs_residual"])
            for r in item_rows
        ),
        "max_pair_channel_sum_abs_residual": max(
            float(r["pair_channel_sum_abs_residual"])
            for r in item_rows
        ),
        "max_parent_strong_channel_mean_d_ry20_reproduction_abs_residual":
            max_parent_channel_repro,
        "parent_item_strong_interaction_match": True,
        "parent_strong_channel_mean_d_ry20_match": True,
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
            "A",
            "B",
            "C",
            "M",
            "I",
            "positive_interaction_mass",
            "negative_interaction_mass",
            "same_sign_channel_count",
            "opposite_sign_channel_count",
            "zero_product_channel_count",
        ):
            summary["role_diagnostics"][role][field] = _aggregate(
                r[f"{field}_{role}"] for r in item_rows
            )

    for field in (
        "delta_A",
        "delta_B",
        "delta_C",
        "A_bar",
        "B_bar",
        "C_bar",
        "M_bar",
        "delta_M",
        "Q_A",
        "Q_B",
        "Q_C",
        "reconstructed_delta_I",
    ):
        summary.setdefault("paired_diagnostics", {})[field] = _aggregate(
            r[field] for r in item_rows
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
    raise StrongInteractionGeometryError("NO_COMMON_PREFLIGHT_PAIR")


def synthetic_midpoint_check() -> dict[str, float]:
    # Deliberately use unequal positive magnitudes and nontrivial cosines.
    ac, at = 1.17, 0.91
    bc, bt = 0.84, 0.73
    cc, ct = 0.41, 0.16

    ic = 2.0 * ac * bc * cc
    it = 2.0 * at * bt * ct
    delta_i = ic - it

    da = ac - at
    db = bc - bt
    dc = cc - ct
    ab = 0.5 * (ac + at)
    bb = 0.5 * (bc + bt)
    cb = 0.5 * (cc + ct)

    mc = 2.0 * ac * bc
    mt = 2.0 * at * bt
    mb = 0.5 * (mc + mt)
    dm = mc - mt

    qa = 2.0 * cb * bb * da
    qb = 2.0 * cb * ab * db
    qc = mb * dc

    magnitude_residual = abs(
        dm - (2.0 * bb * da + 2.0 * ab * db)
    )
    first_level_residual = abs(
        delta_i - (cb * dm + mb * dc)
    )
    geometry_residual = abs(delta_i - (qa + qb + qc))

    require(
        magnitude_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_MAGNITUDE_IDENTITY_FAILURE:{magnitude_residual}",
    )
    require(
        first_level_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_FIRST_LEVEL_IDENTITY_FAILURE:{first_level_residual}",
    )
    require(
        geometry_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_GEOMETRY_IDENTITY_FAILURE:{geometry_residual}",
    )

    return {
        "magnitude_midpoint_identity_abs_residual": magnitude_residual,
        "first_level_midpoint_identity_abs_residual": first_level_residual,
        "geometry_midpoint_identity_abs_residual": geometry_residual,
    }


def print_plan(
    repo: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    synth: Mapping[str, float],
) -> None:
    print(
        "=== LAYER20->21 STRONG INTERACTION GEOMETRY AUDIT PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print(
        "immediate_parent_evidence_freeze =",
        IMMEDIATE_PARENT_EVIDENCE_FREEZE,
    )
    print(
        "immediate_parent_implementation =",
        IMMEDIATE_PARENT_IMPLEMENTATION,
    )
    print(
        "immediate_parent_runner_sha256 =",
        IMMEDIATE_PARENT_RUNNER_SHA256,
    )
    print(
        "immediate_parent_item_sha256 =",
        IMMEDIATE_PARENT_ITEM_SHA256,
    )
    print(
        "immediate_parent_channel_sha256 =",
        IMMEDIATE_PARENT_CHANNEL_SHA256,
    )
    print(
        "immediate_parent_cumulative_sha256 =",
        IMMEDIATE_PARENT_CUMULATIVE_SHA256,
    )
    print(
        "immediate_parent_summary_sha256 =",
        IMMEDIATE_PARENT_SUMMARY_SHA256,
    )
    print(
        "immediate_parent_manifest_sha256 =",
        IMMEDIATE_PARENT_MANIFEST_SHA256,
    )
    print(
        "immediate_parent_report_sha256 =",
        IMMEDIATE_PARENT_REPORT_SHA256,
    )
    print("pair_role_count =", EXPECTED_PAIR_ROLE_COUNT)
    print("common_ddsssss_item_count =", EXPECTED_COMMON_COUNT)
    print("source_block =", SOURCE_BLOCK)
    print("target_residual_layer =", TARGET_RESIDUAL_LAYER)
    print("parent_map_layer =", PARENT_MAP_LAYER)
    print("relative_coordinate =", TARGET_K)
    print("strong_kernel_channel_count =", EXPECTED_STRONG_COUNT)
    print("weak_kernel_channel_count =", EXPECTED_WEAK_COUNT)
    print("lag0_kernel_rms =", parent_summary["lag0_kernel_rms"])
    print("parent_runtime_head =", parent_manifest["runtime_git_head"])
    print(
        "expected_parent_strong_interaction_mean =",
        EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
    )
    print("scientific_question =", QUESTION)
    print("role_identity = I = 2<x,y> = 2*A*B*C = M*C")
    print(
        "paired_identity = delta_I = Q_A + Q_B + Q_C; "
        "Q_A=2*C_bar*B_bar*delta_A; "
        "Q_B=2*C_bar*A_bar*delta_B; "
        "Q_C=M_bar*delta_C"
    )
    print("parent_scalar_rel_tol =", PARENT_SCALAR_REL_TOL)
    print("parent_scalar_abs_tol =", PARENT_SCALAR_ABS_TOL)
    print("vector_abs_tol =", VECTOR_ABS_TOL)
    print("channel_reproduction_abs_tol =", CHANNEL_REPRO_ABS_TOL)
    print("cosine_boundary_slack =", COSINE_BOUNDARY_SLACK)
    for key, value in synth.items():
        print(f"synthetic_{key} =", value)
    print("parent_item_and_240_strong_channel_bridges_preregistered = True")
    print("new_channel_selection_executed = False")
    print("raw_vectors_persisted = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("pca_svd_or_learned_geometry_executed = False")
    print("posthoc_search_executed = False")


def runtime_preflight(
    root: Path,
    parent: Any,
    parent_stack: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
    handoff_path: Path,
):
    runtime = resolve_runtime(
        root,
        parent,
        parent_stack,
        handoff_path,
    )

    plan = parent_stack["stack"]["ctx"]["plan"]
    cohort = parent_stack["stack"]["ctx"]["cohort"]
    corr_row, ctrl_row = _first_common_pair_rows(plan, cohort)

    corr = role_geometry(
        corr_row,
        parent,
        parent_stack,
        runtime,
    )
    ctrl = role_geometry(
        ctrl_row,
        parent,
        parent_stack,
        runtime,
    )

    idx = int(corr["idx"])
    item, _vectors = pair_geometry(
        corr,
        ctrl,
        parent_items[idx],
    )

    print(
        "PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_RUNTIME_PREFLIGHT"
    )
    print("local_template_index =", idx)
    print("model_forward_count =", EXPECTED_PREFLIGHT_FORWARD_COUNT)
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("strong_kernel_channel_count =", EXPECTED_STRONG_COUNT)
    print("raw_vectors_persisted = False")

    for role in ("corr", "ctrl"):
        for field in (
            "A",
            "B",
            "C",
            "M",
            "I",
            "positive_interaction_mass",
            "negative_interaction_mass",
            "same_sign_channel_count",
            "opposite_sign_channel_count",
            "zero_product_channel_count",
        ):
            print(f"{field}_{role} =", item[f"{field}_{role}"])

    for field in (
        "delta_A",
        "delta_B",
        "delta_C",
        "A_bar",
        "B_bar",
        "C_bar",
        "M_bar",
        "delta_M",
        "Q_A",
        "Q_B",
        "Q_C",
        "reconstructed_delta_I",
        "parent_delta_ry20_strong",
        "parent_reproduction_abs_residual",
        "magnitude_midpoint_identity_abs_residual",
        "first_level_midpoint_identity_abs_residual",
        "geometry_midpoint_identity_abs_residual",
        "pair_channel_sum_abs_residual",
    ):
        print(field, "=", item[field])


def execute(
    root: Path,
    parent: Any,
    parent_stack: Mapping[str, Any],
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
    require(
        not partial_dir.exists(),
        f"PARTIAL_OUTPUT_EXISTS:{partial_dir}",
    )

    runtime = resolve_runtime(
        root,
        parent,
        parent_stack,
        handoff_path,
    )

    plan = parent_stack["stack"]["ctx"]["plan"]
    cohort = parent_stack["stack"]["ctx"]["cohort"]

    role_records: dict[tuple[int, str], dict[str, Any]] = {}
    forward_count = 0

    for n, row in enumerate(plan, start=1):
        idx = int(row["local_template_index"])

        if idx in cohort:
            rec = role_geometry(
                row,
                parent,
                parent_stack,
                runtime,
            )
            key = (idx, str(row["role"]))
            require(
                key not in role_records,
                f"ROLE_RECORD_DUPLICATE:{key}",
            )
            role_records[key] = rec
        else:
            matched_prefix, swapped_prefix = _prefixes(
                parent,
                parent_stack,
                row,
            )
            run_branch(
                row,
                parent,
                parent_stack,
                runtime,
                matched_prefix,
            )
            run_branch(
                row,
                parent,
                parent_stack,
                runtime,
                swapped_prefix,
            )

        forward_count += 2

        if n % 16 == 0 or n == len(plan):
            print(
                f"PROGRESS pair_roles={n}/{len(plan)} "
                f"model_forwards={forward_count}",
                flush=True,
            )

    require(
        forward_count == EXPECTED_FULL_FORWARD_COUNT,
        "FORWARD_COUNT_MISMATCH",
    )
    require(
        len(role_records) == EXPECTED_COMMON_COUNT * 2,
        "COMMON_ROLE_RECORD_COUNT_MISMATCH",
    )

    item_rows = []
    delta_vectors = []
    corr_vectors = []
    ctrl_vectors = []

    for idx in sorted(cohort):
        corr = role_records[(int(idx), "corr")]
        ctrl = role_records[(int(idx), "ctrl")]
        item, vectors = pair_geometry(
            corr,
            ctrl,
            parent_items[int(idx)],
        )
        item_rows.append(item)
        delta_vectors.append(vectors["delta_channel"])
        corr_vectors.append(vectors["corr_channel"])
        ctrl_vectors.append(vectors["ctrl_channel"])

    delta_matrix = torch.stack(delta_vectors, dim=0)
    corr_matrix = torch.stack(corr_vectors, dim=0)
    ctrl_matrix = torch.stack(ctrl_vectors, dim=0)

    (
        channel_rows,
        max_parent_channel_repro,
        channel_sum,
    ) = build_channel_validation(
        delta_matrix,
        corr_matrix,
        ctrl_matrix,
        runtime,
        parent_channels,
    )

    attribution = component_attribution(item_rows)

    require(
        math.isclose(
            attribution["mean_signed_components"]["total"],
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
            rel_tol=PARENT_SCALAR_REL_TOL,
            abs_tol=PARENT_SCALAR_ABS_TOL,
        ),
        "PARENT_STRONG_INTERACTION_POPULATION_MEAN_MISMATCH",
    )

    summary = make_summary(
        item_rows,
        attribution,
        runtime,
        max_parent_channel_repro,
        channel_sum,
    )

    partial_dir.mkdir(parents=True, exist_ok=False)

    item_path = (
        partial_dir
        / "layer20_y20_strong_interaction_geometry_item_metrics.jsonl"
    )
    channel_path = (
        partial_dir
        / "layer20_y20_strong_interaction_geometry_channel_validation.jsonl"
    )
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "execution_manifest.json"

    item_path.write_bytes(jsonl_bytes(item_rows))
    channel_path.write_bytes(jsonl_bytes(channel_rows))
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
        "immediate_parent_evidence_freeze":
            IMMEDIATE_PARENT_EVIDENCE_FREEZE,
        "immediate_parent_implementation":
            IMMEDIATE_PARENT_IMPLEMENTATION,
        "immediate_parent_runner_sha256":
            IMMEDIATE_PARENT_RUNNER_SHA256,
        "immediate_parent_runner_blob":
            IMMEDIATE_PARENT_RUNNER_BLOB,
        "immediate_parent_item_sha256":
            IMMEDIATE_PARENT_ITEM_SHA256,
        "immediate_parent_channel_sha256":
            IMMEDIATE_PARENT_CHANNEL_SHA256,
        "immediate_parent_cumulative_sha256":
            IMMEDIATE_PARENT_CUMULATIVE_SHA256,
        "immediate_parent_summary_sha256":
            IMMEDIATE_PARENT_SUMMARY_SHA256,
        "immediate_parent_manifest_sha256":
            IMMEDIATE_PARENT_MANIFEST_SHA256,
        "immediate_parent_report_sha256":
            IMMEDIATE_PARENT_REPORT_SHA256,
        "scientific_question": QUESTION,
        "execution_protocol": EXECUTION_PROTOCOL,
        "item_count": EXPECTED_ITEM_COUNT,
        "pair_role_count": EXPECTED_PAIR_ROLE_COUNT,
        "common_ddsssss_item_count": EXPECTED_COMMON_COUNT,
        "model_forward_count": forward_count,
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "parent_map_layer": PARENT_MAP_LAYER,
        "relative_coordinate": TARGET_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "strong_kernel_channel_count": int(
            runtime["operator"]["strong_mask"].sum().item()
        ),
        "weak_kernel_channel_count": int(
            runtime["operator"]["weak_mask"].sum().item()
        ),
        "equal_kernel_channel_count": int(
            runtime["operator"]["equal_mask"].sum().item()
        ),
        "lag0_kernel_rms": float(runtime["operator"]["rms"]),
        "expected_parent_strong_interaction_mean":
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
        "mamba_source_sha256": runtime["binding"].source_sha256,
        "block_forward_sha256": EXPECTED_BLOCK_FORWARD_SHA256,
        "backbone_forward_sha256":
            EXPECTED_BACKBONE_FORWARD_SHA256,
        "parent_scalar_rel_tol": PARENT_SCALAR_REL_TOL,
        "parent_scalar_abs_tol": PARENT_SCALAR_ABS_TOL,
        "vector_abs_tol": VECTOR_ABS_TOL,
        "channel_reproduction_abs_tol": CHANNEL_REPRO_ABS_TOL,
        "cosine_boundary_slack": COSINE_BOUNDARY_SLACK,
        "kernel_rms_rel_tol": KERNEL_RMS_REL_TOL,
        "kernel_rms_abs_tol": KERNEL_RMS_ABS_TOL,
        "handoff_zip_sha256": runtime["handoff"]["zip_sha256"],
        "checkpoint_sha256":
            runtime["handoff"]["checkpoint_sha256"],
        "encoder_canonical_digest":
            runtime["encoder"]["canonical_digest"],
        "encoder_raw_concat_digest":
            runtime["encoder"]["raw_concat_digest"],
        "parent_item_strong_interaction_match": True,
        "parent_strong_channel_mean_d_ry20_match": True,
        "raw_vectors_persisted": False,
        "raw_per_item_channel_vectors_persisted": False,
        "scientific_model_forward_executed": True,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_whitening_or_learned_geometry_executed": False,
        "posthoc_layer_lag_channel_item_or_window_search_executed":
            False,
        "capture_method": (
            "reuse frozen immediate-parent matched/swapped layer20/layer22 "
            "capture; reconstruct H_r20 and H_y20 in float64; restrict only "
            "to the frozen 240 strong output rows; divide by the frozen "
            "role-specific sqrt(||delta-X22||^2); decompose the exact strong "
            "interaction with the preregistered midpoint identity"
        ),
        "outputs": {
            item_path.name: sha256_file(item_path),
            channel_path.name: sha256_file(channel_path),
            summary_path.name: sha256_file(summary_path),
        },
    }

    manifest_path.write_bytes(json_bytes(manifest))

    os.replace(partial_dir, final_dir)

    print(
        "PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_EXECUTION"
    )
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("strong_kernel_channel_count =", EXPECTED_STRONG_COUNT)
    print("parent_item_strong_interaction_match = True")
    print("parent_strong_channel_mean_d_ry20_match = True")
    print(
        "reconstructed_parent_strong_interaction_mean =",
        attribution["mean_signed_components"]["total"],
    )
    print(
        "mean_Q_A =",
        attribution["mean_signed_components"]["Q_A"],
    )
    print(
        "mean_Q_B =",
        attribution["mean_signed_components"]["Q_B"],
    )
    print(
        "mean_Q_C =",
        attribution["mean_signed_components"]["Q_C"],
    )
    print(
        "largest_absolute_scientific_component =",
        attribution["largest_absolute_scientific_component"],
    )
    print(
        "absolute_component_shares =",
        attribution["absolute_component_shares"],
    )
    print(
        "max_parent_item_strong_interaction_reproduction_abs_residual =",
        summary[
            "max_parent_item_strong_interaction_reproduction_abs_residual"
        ],
    )
    print(
        "max_parent_strong_channel_mean_d_ry20_reproduction_abs_residual =",
        summary[
            "max_parent_strong_channel_mean_d_ry20_reproduction_abs_residual"
        ],
    )
    print(
        "max_geometry_midpoint_identity_abs_residual =",
        summary["max_geometry_midpoint_identity_abs_residual"],
    )
    print(
        "max_role_interaction_abc_abs_residual =",
        summary["max_role_interaction_abc_abs_residual"],
    )
    for role in ("corr", "ctrl"):
        for field in ("A", "B", "C", "I"):
            agg = summary["role_diagnostics"][role][field]
            print(
                f"{role}_{field}_mean =",
                agg["mean"],
            )
            print(
                f"{role}_{field}_median =",
                agg["median"],
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

    parent = load_immediate_parent(root)
    (
        parent_summary,
        parent_manifest,
        parent_items,
        parent_channels,
    ) = authenticate_immediate_parent_evidence(root)

    parent_stack = build_parent_stack(root, parent)
    synth = synthetic_midpoint_check()

    print_plan(
        repo,
        parent_summary,
        parent_manifest,
        synth,
    )

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print(
            "PASS_LAYER20_Y20_STRONG_INTERACTION_GEOMETRY_STATIC_PREFLIGHT"
        )
        return 0

    require(args.handoff is not None, "HANDOFF_REQUIRED")
    handoff = args.handoff.expanduser().resolve()
    require(handoff.is_file(), f"HANDOFF_MISSING:{handoff}")

    if args.runtime_preflight:
        require(
            args.output_dir is None,
            "RUNTIME_PREFLIGHT_OUTPUT_DIR_FORBIDDEN",
        )
        runtime_preflight(
            root,
            parent,
            parent_stack,
            parent_items,
            handoff,
        )
        return 0

    require(args.execute, "UNKNOWN_MODE")
    require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")

    execute(
        root,
        parent,
        parent_stack,
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
    except StrongInteractionGeometryError as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        raise SystemExit(2)
