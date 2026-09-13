#!/usr/bin/env python3
"""K0-RVG layer-21 current-token residual-construction routing-source decomposition.

Frozen target from the immediate parent:
    D_in21 := routing-energy contrast of incoming delta-R21 under the fixed
              layer-22 parent map s_bar22 * gamma22 * W_H22 / ||delta-X22||^2.

Authenticated candidate upstream boundary:
    R20 -> RMSNorm20 -> Mixer20 -> Y20
    R21 = R20 + Y20

This audit decomposes the already-frozen parent incoming source:
    Q_in21 = Q_r20 + Q_y20 + Q_eps20

and, after the fixed layer-22 W_H operator:
    D_in21 = D_r20 + D_y20 + D_ry20 + D_eps20.

The frozen 240/1296 downstream partition and fixed lag0 kernel-rank order are
preserved.  Observational/algebraic only.  No raw per-item vectors persist.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "a1dc479b25fb0b8277ff33102773ef2e05f5fc67"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_layer21_current_token_residual_construction_"
    "routing_source_static_design_candidate.md"
)
AUTHORITY_SHA256 = "50644b4d607aca3e9aabc2f8a0aab4c3ba0875b008f6fca85beca970c795a8da"
AUTHORITY_BLOB = "3c42fee0bc3710c9b643740c4a231981f1f086cb"

IMMEDIATE_PARENT_EVIDENCE_FREEZE = "e78549d64b9628125e619b23dca03b2d025171d3"
IMMEDIATE_PARENT_IMPLEMENTATION = "ab3c852b3655b587a7476ea1baca3bcc469e6a9b"
IMMEDIATE_PARENT_RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer22_current_token_residual_construction_"
    "routing_source_audit.py"
)
IMMEDIATE_PARENT_RUNNER_SHA256 = (
    "973692b8002572e0b229fb5ecf288457b063bae4043ccfeba56fa0c6e7448300"
)
IMMEDIATE_PARENT_RUNNER_BLOB = "7b19e810016a529dcad7b5f59214489d7e8bb44f"

IMMEDIATE_PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_residual_construction_"
    "routing_source_ab3c852_v1"
)
IMMEDIATE_PARENT_ITEM_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer22_current_token_residual_construction_routing_source_item_metrics.jsonl"
)
IMMEDIATE_PARENT_ITEM_SHA256 = (
    "063bf6355795c20adde263d8ceb8868be4e5cb42fe5c5fd46da8219cf5b58124"
)
IMMEDIATE_PARENT_CHANNEL_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer22_current_token_residual_construction_routing_source_channel_summary.jsonl"
)
IMMEDIATE_PARENT_CHANNEL_SHA256 = (
    "96a1308a5e384e59cd08e4df6e25d406ff6899e663326ecaff2c1702c8adcdfc"
)
IMMEDIATE_PARENT_CUMULATIVE_REL = (
    IMMEDIATE_PARENT_RUN_DIR
    + "/layer22_current_token_residual_construction_routing_source_kernel_rank_cumulative.jsonl"
)
IMMEDIATE_PARENT_CUMULATIVE_SHA256 = (
    "3f7be599d6f505a6f2e679bb24920f7ca09b8577f73289cc457a1138b4547f5d"
)
IMMEDIATE_PARENT_SUMMARY_REL = IMMEDIATE_PARENT_RUN_DIR + "/summary.json"
IMMEDIATE_PARENT_SUMMARY_SHA256 = (
    "364d3a961cc103c2760ea7d70a5547af52502d7e5eadcd2c6421e08ee4893ee0"
)
IMMEDIATE_PARENT_MANIFEST_REL = IMMEDIATE_PARENT_RUN_DIR + "/execution_manifest.json"
IMMEDIATE_PARENT_MANIFEST_SHA256 = (
    "55f992bddc7bdc232ce5a8c2d6555feb82026e6f14bb86424eed23d369368f4b"
)
IMMEDIATE_PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer22_current_token_residual_construction_routing_source_"
    "validated_evidence_analysis_report_candidate.md"
)
IMMEDIATE_PARENT_REPORT_SHA256 = (
    "bea953a6ba23705c31d5025b93a4511425412ae019acefcba5756a81690741ed"
)

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_layer21_current_token_residual_construction_"
    "routing_source_audit.py"
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

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)
EXPECTED_BLOCK_FORWARD_SHA256 = (
    "0f808b4d539a496e1681c81d39799d7072fa5c6da381a07cfa19f35fe825b3e4"
)
EXPECTED_BACKBONE_FORWARD_SHA256 = (
    "3f332bd50e6ea4ff64468c8d748672f90ffc43912d22c9d25d02b3e87d1661a6"
)

EXPECTED_PARENT_MEAN_IN = {
    "all": -0.2341372085181502,
    "strong": 0.9173468619072845,
    "weak": -1.1514840704254348,
}

EXECUTION_PROTOCOL = "EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6"

BRANCH_ADD_RECON_REL_TOL = 1e-7
PARENT_SCALAR_REL_TOL = 1e-13
PARENT_SCALAR_ABS_TOL = 1e-13
VECTOR_ABS_TOL = 5e-12
SQUARED_ENERGY_REL_TOL = 1e-12
KERNEL_RMS_REL_TOL = 1e-13
KERNEL_RMS_ABS_TOL = 1e-13

QUESTION = (
    "Is the validated incoming layer21 residual routing signature inherited "
    "from incoming delta-R20, produced by the layer20 mixer update delta-Y20, "
    "generated mainly by their downstream interaction under the fixed layer22 "
    "parent map, or genuinely mixed?"
)

ITEM_SCHEMA = (
    "k0-rvg-layer21-current-token-residual-construction-routing-source-item-v1"
)
CHANNEL_SCHEMA = (
    "k0-rvg-layer21-current-token-residual-construction-routing-source-channel-v1"
)
CUMULATIVE_SCHEMA = (
    "k0-rvg-layer21-current-token-residual-construction-routing-source-cumulative-v1"
)
SUMMARY_SCHEMA = (
    "k0-rvg-layer21-current-token-residual-construction-routing-source-summary-v1"
)
MANIFEST_SCHEMA = (
    "k0-rvg-layer21-current-token-residual-construction-routing-source-"
    "execution-manifest-v1"
)


class Layer21ResidualConstructionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Layer21ResidualConstructionError(message)


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
        raise Layer21ResidualConstructionError(
            f"GIT_FAILURE:{' '.join(args)}"
        ) from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Layer21ResidualConstructionError(
            f"GIT_SHOW_FAILURE:{spec}"
        ) from exc


def source_sha(fn: Any) -> str:
    src = textwrap.dedent(inspect.getsource(fn))
    return sha256_bytes(src.encode("utf-8"))


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
        raise Layer21ResidualConstructionError(
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
        "k0_rvg_layer22_residual_construction_parent",
    )


def _load_json(raw: bytes, label: str):
    try:
        return json.loads(raw)
    except Exception as exc:
        raise Layer21ResidualConstructionError(
            f"{label}_JSON_PARSE_FAILURE"
        ) from exc


def authenticate_immediate_parent_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
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
    cumulative_raw = _authenticate_file(
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
        == "k0-rvg-layer22-current-token-residual-construction-routing-source-summary-v1",
        "IMMEDIATE_PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("schema_version")
        == "k0-rvg-layer22-current-token-residual-construction-routing-source-execution-manifest-v1",
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

    attr = summary["component_attribution"]
    for part, expected in EXPECTED_PARENT_MEAN_IN.items():
        got = float(attr[part]["mean_signed_components"]["in"])
        require(
            math.isclose(
                got,
                expected,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"IMMEDIATE_PARENT_MEAN_IN_MISMATCH:{part}:{got}:{expected}",
        )

    items: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(item_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise Layer21ResidualConstructionError(
                f"IMMEDIATE_PARENT_ITEM_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        idx = int(row["local_template_index"])
        require(idx not in items, f"IMMEDIATE_PARENT_ITEM_DUPLICATE:{idx}")
        items[idx] = row
    require(len(items) == EXPECTED_COMMON_COUNT, "IMMEDIATE_PARENT_ITEM_COUNT_MISMATCH")

    channels: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(channel_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise Layer21ResidualConstructionError(
                f"IMMEDIATE_PARENT_CHANNEL_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        j = int(row["channel_index"])
        require(j not in channels, f"IMMEDIATE_PARENT_CHANNEL_DUPLICATE:{j}")
        channels[j] = row
    require(
        len(channels) == INTERMEDIATE_SIZE,
        "IMMEDIATE_PARENT_CHANNEL_COUNT_MISMATCH",
    )

    cumulative: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(cumulative_raw.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise Layer21ResidualConstructionError(
                f"IMMEDIATE_PARENT_CUMULATIVE_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        rank = int(row["kernel_magnitude_rank"])
        require(
            rank not in cumulative,
            f"IMMEDIATE_PARENT_CUMULATIVE_DUPLICATE:{rank}",
        )
        cumulative[rank] = row
    require(
        len(cumulative) == INTERMEDIATE_SIZE,
        "IMMEDIATE_PARENT_CUMULATIVE_COUNT_MISMATCH",
    )

    return summary, manifest, items, channels, cumulative


def build_parent_stack(root: Path, parent: Any) -> dict[str, Any]:
    # Authenticate every frozen dependency that the immediate-parent runner used.
    parent.authenticate_authority(root)
    grandparent = parent.load_immediate_parent(root)
    _residual_precedent, _residual_precedent_sha = parent.authenticate_residual_precedent(root)
    (
        grandparent_summary,
        grandparent_manifest,
        grandparent_items,
        grandparent_channels,
    ) = parent.authenticate_immediate_parent_evidence(root)
    ctx = parent.build_context(root, grandparent)
    return {
        "grandparent": grandparent,
        "grandparent_summary": grandparent_summary,
        "grandparent_manifest": grandparent_manifest,
        "grandparent_items": grandparent_items,
        "grandparent_channels": grandparent_channels,
        "ctx": ctx,
    }


def resolve_runtime(
    root: Path,
    parent: Any,
    stack: Mapping[str, Any],
    handoff_path: Path,
) -> dict[str, Any]:
    runtime = parent.resolve_runtime(
        root,
        stack["grandparent"],
        stack["ctx"],
        stack["grandparent_channels"],
        handoff_path,
    )

    model = runtime["model"]
    backbone = model.mamba
    require(
        len(backbone.layers) == EXPECTED_LAYER_COUNT,
        "BACKBONE_LAYER_COUNT_MISMATCH",
    )

    layer20 = backbone.layers[SOURCE_BLOCK]
    layer21 = backbone.layers[TARGET_RESIDUAL_LAYER]
    require(layer21 is runtime["layer21"], "LAYER21_RUNTIME_IDENTITY_MISMATCH")
    require(type(layer20) is type(layer21), "LAYER20_LAYER21_BLOCK_CLASS_MISMATCH")
    require(
        hasattr(layer20, "norm") and hasattr(layer20, "mixer"),
        "LAYER20_CHILDREN_MISSING",
    )
    require(
        bool(layer20.residual_in_fp32) is True,
        "LAYER20_RESIDUAL_IN_FP32_NOT_TRUE",
    )

    import torch

    require(
        tuple(layer20.norm.weight.shape) == (HIDDEN_SIZE,),
        "LAYER20_NORM_WEIGHT_SHAPE_MISMATCH",
    )
    require(
        layer20.norm.weight.dtype == torch.float32,
        "LAYER20_NORM_WEIGHT_DTYPE_MISMATCH",
    )
    require(
        int(layer20.mixer.hidden_size) == HIDDEN_SIZE,
        "LAYER20_MIXER_HIDDEN_SIZE_MISMATCH",
    )

    require(
        source_sha(type(layer20).forward) == EXPECTED_BLOCK_FORWARD_SHA256,
        "BLOCK_FORWARD_SHA256_MISMATCH",
    )
    require(
        source_sha(type(backbone).forward) == EXPECTED_BACKBONE_FORWARD_SHA256,
        "BACKBONE_FORWARD_SHA256_MISMATCH",
    )
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

    runtime = dict(runtime)
    runtime["backbone"] = backbone
    runtime["layer20"] = layer20
    return runtime


def _prefixes(
    parent: Any,
    stack: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return parent._prefixes(
        stack["grandparent"],
        stack["ctx"],
        row,
    )


def capture_branch_with_layer20(
    parent: Any,
    stack: Mapping[str, Any],
    runtime: Mapping[str, Any],
    token_ids: Sequence[int],
    parent_targets: Sequence[int],
    target: int,
):
    import torch

    layer20 = runtime["layer20"]
    holders: dict[str, Any] = {}
    counts = {"block_pre": 0, "mixer_post": 0, "block_post": 0}

    def block_pre_hook(module, args):
        counts["block_pre"] += 1
        require(counts["block_pre"] == 1, "DUPLICATE_LAYER20_BLOCK_PRE_HOOK")
        require(len(args) >= 1, "LAYER20_BLOCK_PRE_ARG_COUNT_MISMATCH")
        full = args[0].detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, f"R20_DTYPE_MISMATCH:{full.dtype}")
        require(full.ndim == 3 and full.shape[0] == 1, "R20_FULL_SHAPE_MISMATCH")
        require(full.shape[-1] == HIDDEN_SIZE, "R20_WIDTH_MISMATCH")
        require(0 <= target < full.shape[1], "R20_TARGET_OUT_OF_RANGE")
        holders["R20"] = full[0, target, :].contiguous().clone()

    def mixer_post_hook(module, args, output):
        counts["mixer_post"] += 1
        require(counts["mixer_post"] == 1, "DUPLICATE_LAYER20_MIXER_POST_HOOK")
        full = output.detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, f"Y20_DTYPE_MISMATCH:{full.dtype}")
        require(full.ndim == 3 and full.shape[0] == 1, "Y20_FULL_SHAPE_MISMATCH")
        require(full.shape[-1] == HIDDEN_SIZE, "Y20_WIDTH_MISMATCH")
        require(0 <= target < full.shape[1], "Y20_TARGET_OUT_OF_RANGE")
        holders["Y20"] = full[0, target, :].contiguous().clone()

    def block_post_hook(module, args, output):
        counts["block_post"] += 1
        require(counts["block_post"] == 1, "DUPLICATE_LAYER20_BLOCK_POST_HOOK")
        full = output.detach().cpu().contiguous().clone()
        require(full.dtype == torch.float32, f"R21_DTYPE_MISMATCH:{full.dtype}")
        require(full.ndim == 3 and full.shape[0] == 1, "R21_FULL_SHAPE_MISMATCH")
        require(full.shape[-1] == HIDDEN_SIZE, "R21_WIDTH_MISMATCH")
        require(0 <= target < full.shape[1], "R21_TARGET_OUT_OF_RANGE")
        holders["R21"] = full[0, target, :].contiguous().clone()

    h1 = layer20.register_forward_pre_hook(block_pre_hook)
    h2 = layer20.mixer.register_forward_hook(mixer_post_hook)
    h3 = layer20.register_forward_hook(block_post_hook)

    try:
        parent_records, rms_capture, layer21_capture = (
            parent.capture_branch_with_layer21(
                stack["grandparent"],
                stack["ctx"],
                runtime,
                token_ids,
                parent_targets,
                target,
            )
        )
    finally:
        h1.remove()
        h2.remove()
        h3.remove()

    require(
        counts == {"block_pre": 1, "mixer_post": 1, "block_post": 1},
        f"LAYER20_HOOK_COUNT_FAILURE:{counts}",
    )
    require(set(holders) == {"R20", "Y20", "R21"}, "LAYER20_CAPTURE_MISSING")

    require(
        torch.equal(holders["R21"], layer21_capture["R21"]),
        "LAYER20_R21_PARENT_LAYER21_INPUT_EXACT_MISMATCH",
    )

    reconstructed32 = (holders["R20"] + holders["Y20"]).to(torch.float32)
    branch_residual = (
        reconstructed32.to(torch.float64)
        - holders["R21"].to(torch.float64)
    )
    denom = max(
        float(
            torch.linalg.vector_norm(
                holders["R21"].to(torch.float64)
            ).item()
        ),
        1e-12,
    )
    branch_rel = float(torch.linalg.vector_norm(branch_residual).item()) / denom
    require(
        branch_rel <= BRANCH_ADD_RECON_REL_TOL,
        f"LAYER20_BRANCH_ADD_RECONSTRUCTION_FAILURE:{target}:{branch_rel}",
    )

    return parent_records, rms_capture, layer21_capture, {
        "R20": holders["R20"],
        "Y20": holders["Y20"],
        "R21": holders["R21"],
        "branch_add_reconstruction_relative_residual": branch_rel,
    }


def _isclose_parent(got, expected) -> bool:
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
    raise Layer21ResidualConstructionError(f"UNKNOWN_PARTITION:{part}")


def role_decomposition(
    row: Mapping[str, Any],
    matched_rms: Mapping[str, Any],
    swapped_rms: Mapping[str, Any],
    matched_layer21: Mapping[str, Any],
    swapped_layer21: Mapping[str, Any],
    matched_layer20: Mapping[str, Any],
    swapped_layer20: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_item: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    idx = int(row["local_template_index"])
    role = str(row["role"])

    gamma = (
        runtime["norm22"].weight.detach().cpu().to(torch.float64).contiguous()
    )
    w = runtime["operator"]["w_hidden64"]

    r20m = matched_layer20["R20"].to(torch.float64).contiguous()
    r20s = swapped_layer20["R20"].to(torch.float64).contiguous()
    y20m = matched_layer20["Y20"].to(torch.float64).contiguous()
    y20s = swapped_layer20["Y20"].to(torch.float64).contiguous()
    r21m = matched_layer20["R21"].to(torch.float64).contiguous()
    r21s = swapped_layer20["R21"].to(torch.float64).contiguous()

    # Exact bridge to the immediate-parent layer21 input capture.
    require(
        torch.equal(matched_layer20["R21"], matched_layer21["R21"]),
        f"MATCHED_R21_PARENT_EXACT_MISMATCH:{idx}:{role}",
    )
    require(
        torch.equal(swapped_layer20["R21"], swapped_layer21["R21"]),
        f"SWAPPED_R21_PARENT_EXACT_MISMATCH:{idx}:{role}",
    )

    xm = matched_rms["X32"].to(torch.float64).contiguous()
    xs = swapped_rms["X32"].to(torch.float64).contiguous()
    sm = float(matched_rms["rms_scale32"].item())
    ss = float(swapped_rms["rms_scale32"].item())
    sbar = 0.5 * (sm + ss)

    dr20 = r20m - r20s
    dy20 = y20m - y20s
    dr21 = r21m - r21s
    x_obs = xm - xs

    dr21_sq = float(torch.dot(dr21, dr21).item())
    dx_sq = float(torch.dot(x_obs, x_obs).item())
    require(dr21_sq > 0.0, f"DELTA_R21_ZERO:{idx}:{role}")
    require(dx_sq > 0.0, f"DELTA_X22_ZERO:{idx}:{role}")

    eps_m = r21m - (r20m + y20m)
    eps_s = r21s - (r20s + y20s)
    q_add20_eps = eps_m - eps_s

    dr_alg = dr20 + dy20
    error_diff = dr21 - dr_alg
    add_error_identity = float(
        torch.max(torch.abs(error_diff - q_add20_eps)).item()
    )
    require(
        add_error_identity <= VECTOR_ABS_TOL,
        f"ADD20_ERROR_DIFFERENCE_IDENTITY_FAILURE:"
        f"{idx}:{role}:{add_error_identity}",
    )

    add_difference_rel_diag = float(
        torch.linalg.vector_norm(dr_alg - dr21).item()
    ) / math.sqrt(dr21_sq)

    dr20_sq = float(torch.dot(dr20, dr20).item())
    dy20_sq = float(torch.dot(dy20, dy20).item())
    residual_cross = 2.0 * float(torch.dot(dr20, dy20).item())
    residual_rhs = dr20_sq + dy20_sq + residual_cross
    residual_alg_sq = float(torch.dot(dr_alg, dr_alg).item())
    residual_energy_closure = abs(residual_alg_sq - residual_rhs)
    require(
        residual_energy_closure
        <= SQUARED_ENERGY_REL_TOL
        * max(residual_alg_sq, dr20_sq + dy20_sq, 1.0),
        f"RESIDUAL20_SPACE_ENERGY_CLOSURE_FAILURE:"
        f"{idx}:{role}:{residual_energy_closure}",
    )

    rss_sq = dr20_sq + dy20_sq
    addition_ratio = (
        math.sqrt(residual_alg_sq) / math.sqrt(rss_sq)
        if rss_sq > 0.0
        else 0.0
    )
    residual_cross_fraction = (
        residual_cross / rss_sq if rss_sq > 0.0 else 0.0
    )

    # Fixed layer22 parent map.  No new layer20/layer21 normalization enters.
    q_r20 = gamma * (sbar * dr20)
    q_y20 = gamma * (sbar * dy20)
    q_eps20 = gamma * (sbar * q_add20_eps)
    q_in21 = gamma * (sbar * dr21)

    q_closure = float(
        torch.max(
            torch.abs(q_in21 - (q_r20 + q_y20 + q_eps20))
        ).item()
    )
    require(
        q_closure <= VECTOR_ABS_TOL,
        f"Q_IN21_SOURCE_CLOSURE_FAILURE:{idx}:{role}:{q_closure}",
    )

    delta_r21_l2 = math.sqrt(dr21_sq)
    q_in21_l2 = float(torch.linalg.vector_norm(q_in21).item())

    for field, got in (
        (f"delta_r21_l2_{role}", delta_r21_l2),
        (f"q_in_l2_{role}", q_in21_l2),
        (f"s22_bar_{role}", sbar),
    ):
        expected = parent_item[field]
        require(
            _isclose_parent(got, expected),
            f"IMMEDIATE_PARENT_SCALAR_REPRODUCTION_FAILURE:"
            f"{idx}:{role}:{field}:{got}:{expected}",
        )

    h_r20 = torch.mv(w, q_r20)
    h_y20 = torch.mv(w, q_y20)
    h_eps20 = torch.mv(w, q_eps20)
    h_in21 = torch.mv(w, q_in21)

    h_closure = float(
        torch.max(
            torch.abs(h_in21 - (h_r20 + h_y20 + h_eps20))
        ).item()
    )
    require(
        h_closure <= VECTOR_ABS_TOL,
        f"H_IN21_SOURCE_CLOSURE_FAILURE:{idx}:{role}:{h_closure}",
    )

    e_r20 = h_r20 * h_r20 / dx_sq
    e_y20 = h_y20 * h_y20 / dx_sq
    e_ry20 = 2.0 * h_r20 * h_y20 / dx_sq
    e_eps20 = (
        h_eps20 * h_eps20
        + 2.0 * h_r20 * h_eps20
        + 2.0 * h_y20 * h_eps20
    ) / dx_sq
    e_total = h_in21 * h_in21 / dx_sq

    channel_closure = float(
        torch.max(
            torch.abs(
                e_total - (e_r20 + e_y20 + e_ry20 + e_eps20)
            )
        ).item()
    )
    require(
        channel_closure <= VECTOR_ABS_TOL,
        f"ROLE_CHANNEL_SOURCE_CLOSURE_FAILURE:"
        f"{idx}:{role}:{channel_closure}",
    )

    return {
        "idx": idx,
        "stable_item_id": row["stable_item_id"],
        "role": role,
        "delta_r20_l2": math.sqrt(dr20_sq),
        "delta_y20_l2": math.sqrt(dy20_sq),
        "delta_r21_l2": delta_r21_l2,
        "q_add20_eps_l2": float(
            torch.linalg.vector_norm(q_add20_eps).item()
        ),
        "q_r20_l2": float(torch.linalg.vector_norm(q_r20).item()),
        "q_y20_l2": float(torch.linalg.vector_norm(q_y20).item()),
        "q_eps20_l2": float(torch.linalg.vector_norm(q_eps20).item()),
        "q_in21_l2": q_in21_l2,
        "s22_matched": sm,
        "s22_swapped": ss,
        "s22_bar": sbar,
        "branch_add_reconstruction_relative_residual": max(
            float(
                matched_layer20[
                    "branch_add_reconstruction_relative_residual"
                ]
            ),
            float(
                swapped_layer20[
                    "branch_add_reconstruction_relative_residual"
                ]
            ),
        ),
        "difference_add_reconstruction_relative_residual_diagnostic":
            add_difference_rel_diag,
        "add_error_difference_identity_abs_residual":
            add_error_identity,
        "residual_space_cross": residual_cross,
        "residual_space_cross_fraction": residual_cross_fraction,
        "residual_space_addition_ratio": addition_ratio,
        "residual_space_energy_closure_abs_residual":
            residual_energy_closure,
        "q_in21_source_closure_max_abs_residual": q_closure,
        "h_in21_source_closure_max_abs_residual": h_closure,
        "channel_source_closure_max_abs_residual":
            channel_closure,
        "e_total": e_total,
        "e_r20": e_r20,
        "e_y20": e_y20,
        "e_ry20": e_ry20,
        "e_eps20": e_eps20,
    }


def pair_role_records(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_item: Mapping[str, Any],
):
    import torch

    require(corr["idx"] == ctrl["idx"], "PAIR_INDEX_MISMATCH")
    require(
        corr["stable_item_id"] == ctrl["stable_item_id"],
        "PAIR_STABLE_ID_MISMATCH",
    )
    idx = int(corr["idx"])

    vectors = {
        key: corr[f"e_{key}"] - ctrl[f"e_{key}"]
        for key in ("total", "r20", "y20", "ry20", "eps20")
    }

    channel_closure = float(
        torch.max(
            torch.abs(
                vectors["total"]
                - (
                    vectors["r20"]
                    + vectors["y20"]
                    + vectors["ry20"]
                    + vectors["eps20"]
                )
            )
        ).item()
    )
    require(
        channel_closure <= VECTOR_ABS_TOL,
        f"PAIR_CHANNEL_SOURCE_CLOSURE_FAILURE:{idx}:{channel_closure}",
    )

    partition_values: dict[str, dict[str, float]] = {}
    max_partition_closure = 0.0
    max_parent_repro = 0.0

    for part in ("all", "strong", "weak"):
        vals = {
            key: _partition_sum(vectors[key], runtime["operator"], part)
            for key in ("total", "r20", "y20", "ry20", "eps20")
        }
        closure = abs(
            vals["total"]
            - (
                vals["r20"]
                + vals["y20"]
                + vals["ry20"]
                + vals["eps20"]
            )
        )
        require(
            closure <= VECTOR_ABS_TOL,
            f"PARTITION_SOURCE_CLOSURE_FAILURE:{idx}:{part}:{closure}",
        )

        parent_target = float(parent_item[f"delta_in_{part}"])
        parent_repro = abs(vals["total"] - parent_target)
        require(
            parent_repro <= VECTOR_ABS_TOL,
            f"PARENT_INCOMING_SOURCE_REPRODUCTION_FAILURE:"
            f"{idx}:{part}:{vals['total']}:{parent_target}:{parent_repro}",
        )

        vals["closure"] = closure
        vals["parent_target"] = parent_target
        vals["parent_repro"] = parent_repro
        partition_values[part] = vals
        max_partition_closure = max(max_partition_closure, closure)
        max_parent_repro = max(max_parent_repro, parent_repro)

    row = {
        "schema_version": ITEM_SCHEMA,
        "local_template_index": idx,
        "stable_item_id": corr["stable_item_id"],
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "parent_map_layer": PARENT_MAP_LAYER,
        "relative_coordinate": TARGET_K,
    }

    for role_name, rec in (("corr", corr), ("ctrl", ctrl)):
        for field in (
            "delta_r20_l2",
            "delta_y20_l2",
            "delta_r21_l2",
            "q_add20_eps_l2",
            "q_r20_l2",
            "q_y20_l2",
            "q_eps20_l2",
            "q_in21_l2",
            "s22_matched",
            "s22_swapped",
            "s22_bar",
            "branch_add_reconstruction_relative_residual",
            "difference_add_reconstruction_relative_residual_diagnostic",
            "add_error_difference_identity_abs_residual",
            "residual_space_cross",
            "residual_space_cross_fraction",
            "residual_space_addition_ratio",
            "residual_space_energy_closure_abs_residual",
            "q_in21_source_closure_max_abs_residual",
            "h_in21_source_closure_max_abs_residual",
            "channel_source_closure_max_abs_residual",
        ):
            row[f"{field}_{role_name}"] = rec[field]

    row["parent_delta_r21_l2_corr"] = float(
        parent_item["delta_r21_l2_corr"]
    )
    row["parent_delta_r21_l2_ctrl"] = float(
        parent_item["delta_r21_l2_ctrl"]
    )
    row["parent_q_in_l2_corr"] = float(parent_item["q_in_l2_corr"])
    row["parent_q_in_l2_ctrl"] = float(parent_item["q_in_l2_ctrl"])
    row["parent_s22_bar_corr"] = float(parent_item["s22_bar_corr"])
    row["parent_s22_bar_ctrl"] = float(parent_item["s22_bar_ctrl"])

    for part in ("all", "strong", "weak"):
        vals = partition_values[part]
        for key in ("total", "r20", "y20", "ry20", "eps20"):
            row[f"delta_{key}_{part}"] = vals[key]
        row[f"parent_delta_in_{part}"] = vals["parent_target"]
        row[f"component_closure_abs_residual_{part}"] = vals["closure"]
        row[f"parent_reproduction_abs_residual_{part}"] = vals[
            "parent_repro"
        ]

    row["pair_channel_source_closure_max_abs_residual"] = (
        channel_closure
    )
    row["max_partition_source_closure_abs_residual"] = (
        max_partition_closure
    )
    row["max_parent_incoming_source_reproduction_abs_residual"] = (
        max_parent_repro
    )
    row["raw_vectors_persisted"] = False

    return row, vectors


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


def build_channel_outputs(
    matrices: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_channels: Mapping[int, Mapping[str, Any]],
    parent_cumulative: Mapping[int, Mapping[str, Any]],
):
    import torch

    expected = (EXPECTED_COMMON_COUNT, INTERMEDIATE_SIZE)
    for key, matrix in matrices.items():
        require(
            tuple(matrix.shape) == expected,
            f"{key.upper()}_MATRIX_SHAPE_MISMATCH",
        )

    means = {k: torch.mean(v, dim=0) for k, v in matrices.items()}
    medians = {}
    pos = {}
    neg = {}
    zero = {}
    for key, matrix in matrices.items():
        sorted_v, _ = torch.sort(matrix, dim=0)
        medians[key] = (
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

    max_parent_channel_repro = 0.0
    channel_rows = []
    for j in range(INTERMEDIATE_SIZE):
        part = (
            "strong"
            if bool(op["strong_mask"][j].item())
            else "weak"
            if bool(op["weak_mask"][j].item())
            else "equal"
        )
        p = parent_channels[j]
        require(
            p["downstream_partition"] == part,
            f"PARENT_PARTITION_MISMATCH:{j}",
        )
        require(
            int(p["kernel_magnitude_rank"])
            == int(rank_by_kernel[j]),
            f"PARENT_KERNEL_RANK_MISMATCH:{j}",
        )

        parent_mean_in = float(p["mean_d_in"])
        new_mean_total = float(means["total"][j].item())
        parent_residual = abs(new_mean_total - parent_mean_in)
        require(
            math.isclose(
                new_mean_total,
                parent_mean_in,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_CHANNEL_MEAN_IN_REPRODUCTION_FAILURE:"
            f"{j}:{new_mean_total}:{parent_mean_in}",
        )
        max_parent_channel_repro = max(
            max_parent_channel_repro,
            parent_residual,
        )

        row = {
            "schema_version": CHANNEL_SCHEMA,
            "channel_index": j,
            "kernel_magnitude_rank": int(rank_by_kernel[j]),
            "downstream_partition": part,
            "downstream_lag0_kernel_weight": float(op["lag0"][j].item()),
            "downstream_lag0_kernel_weight_sq": float(
                op["lag0_k2"][j].item()
            ),
            "inproj_row_gain_sq": float(op["row_r2"][j].item()),
            "parent_mean_d_in": parent_mean_in,
            "parent_mean_d_in_reproduction_abs_residual":
                parent_residual,
        }
        for key in ("total", "r20", "y20", "ry20", "eps20"):
            row[f"mean_d_{key}"] = float(means[key][j].item())
            row[f"median_d_{key}"] = float(medians[key][j].item())
            row[f"positive_d_{key}_count"] = int(pos[key][j].item())
            row[f"negative_d_{key}_count"] = int(neg[key][j].item())
            row[f"zero_d_{key}_count"] = int(zero[key][j].item())
        channel_rows.append(row)

    cumulative_rows = []
    running = {
        k: 0.0
        for k in ("total", "r20", "y20", "ry20", "eps20")
    }
    max_parent_cumulative_repro = 0.0

    for rank, channel in enumerate(op["kernel_order"], start=1):
        j = int(channel)
        for key in running:
            running[key] += float(means[key][j].item())

        p = parent_cumulative[rank]
        require(
            int(p["channel_index"]) == j,
            f"PARENT_CUMULATIVE_CHANNEL_MISMATCH:{rank}",
        )
        parent_cumulative_in = float(p["cumulative_mean_d_in"])
        parent_residual = abs(
            running["total"] - parent_cumulative_in
        )
        require(
            math.isclose(
                running["total"],
                parent_cumulative_in,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_CUMULATIVE_IN_REPRODUCTION_FAILURE:"
            f"{rank}:{running['total']}:{parent_cumulative_in}",
        )
        max_parent_cumulative_repro = max(
            max_parent_cumulative_repro,
            parent_residual,
        )

        cumulative_rows.append(
            {
                "schema_version": CUMULATIVE_SCHEMA,
                "kernel_magnitude_rank": rank,
                "channel_index": j,
                "downstream_partition":
                    channel_rows[j]["downstream_partition"],
                **{
                    f"cumulative_mean_d_{k}": running[k]
                    for k in running
                },
                "parent_cumulative_mean_d_in":
                    parent_cumulative_in,
                "parent_cumulative_mean_d_in_reproduction_abs_residual":
                    parent_residual,
            }
        )

    return (
        channel_rows,
        cumulative_rows,
        max_parent_channel_repro,
        max_parent_cumulative_repro,
    )


def component_attribution(item_rows) -> dict[str, Any]:
    out = {}
    for part in ("all", "strong", "weak"):
        means = {
            key: float(
                statistics.fmean(
                    row[f"delta_{key}_{part}"]
                    for row in item_rows
                )
            )
            for key in ("total", "r20", "y20", "ry20", "eps20")
        }
        closure = abs(
            means["total"]
            - (
                means["r20"]
                + means["y20"]
                + means["ry20"]
                + means["eps20"]
            )
        )
        require(
            closure <= VECTOR_ABS_TOL,
            f"AGGREGATE_SOURCE_CLOSURE_FAILURE:{part}:{closure}",
        )
        mass = sum(
            abs(means[k])
            for k in ("r20", "y20", "ry20", "eps20")
        )
        shares = {
            k: (abs(means[k]) / mass if mass > 0.0 else None)
            for k in ("r20", "y20", "ry20", "eps20")
        }
        ratios = {
            k: (
                means[k] / means["total"]
                if means["total"] != 0.0
                else None
            )
            for k in ("r20", "y20", "ry20", "eps20")
        }
        largest = max(
            ("r20", "y20", "ry20"),
            key=lambda k: abs(means[k]),
        )
        out[part] = {
            "mean_signed_components": means,
            "absolute_component_mass": mass,
            "absolute_component_shares": shares,
            "signed_component_to_total_net_ratios": ratios,
            "largest_absolute_scientific_component": largest,
            "aggregate_closure_abs_residual": closure,
            "qualitative_parent_signature_component": {
                "r20": None,
                "y20": None,
                "ry20": None,
            },
        }

    for key in ("r20", "y20", "ry20"):
        carries = (
            out["strong"]["mean_signed_components"][key] > 0.0
            and out["weak"]["mean_signed_components"][key] < 0.0
        )
        for part in ("all", "strong", "weak"):
            out[part][
                "qualitative_parent_signature_component"
            ][key] = carries

    return out


def make_summary(
    item_rows,
    runtime: Mapping[str, Any],
    attribution: Mapping[str, Any],
    max_parent_channel_repro: float,
    max_parent_cumulative_repro: float,
) -> dict[str, Any]:
    op = runtime["operator"]
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
        "lag0_kernel_rms": float(op["rms"]),
        "strong_kernel_channel_count": int(
            op["strong_mask"].sum().item()
        ),
        "weak_kernel_channel_count": int(
            op["weak_mask"].sum().item()
        ),
        "equal_kernel_channel_count": int(
            op["equal_mask"].sum().item()
        ),
        "component_attribution": attribution,
        "role_diagnostics": {},
        "layer20_r21_equals_parent_layer21_input": True,
        "parent_incoming_source_routing_match": True,
        "parent_channel_mean_d_in_match": True,
        "parent_cumulative_mean_d_in_match": True,
        "max_branch_add_reconstruction_relative_residual": max(
            max(
                float(
                    r[
                        "branch_add_reconstruction_relative_residual_corr"
                    ]
                ),
                float(
                    r[
                        "branch_add_reconstruction_relative_residual_ctrl"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_difference_add_reconstruction_relative_residual_diagnostic":
            max(
                max(
                    float(
                        r[
                            "difference_add_reconstruction_relative_residual_diagnostic_corr"
                        ]
                    ),
                    float(
                        r[
                            "difference_add_reconstruction_relative_residual_diagnostic_ctrl"
                        ]
                    ),
                )
                for r in item_rows
            ),
        "max_add_error_difference_identity_abs_residual": max(
            max(
                float(
                    r[
                        "add_error_difference_identity_abs_residual_corr"
                    ]
                ),
                float(
                    r[
                        "add_error_difference_identity_abs_residual_ctrl"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_residual_space_energy_closure_abs_residual": max(
            max(
                float(
                    r[
                        "residual_space_energy_closure_abs_residual_corr"
                    ]
                ),
                float(
                    r[
                        "residual_space_energy_closure_abs_residual_ctrl"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_q_in21_source_closure_abs_residual": max(
            max(
                float(
                    r[
                        "q_in21_source_closure_max_abs_residual_corr"
                    ]
                ),
                float(
                    r[
                        "q_in21_source_closure_max_abs_residual_ctrl"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_h_in21_source_closure_abs_residual": max(
            max(
                float(
                    r[
                        "h_in21_source_closure_max_abs_residual_corr"
                    ]
                ),
                float(
                    r[
                        "h_in21_source_closure_max_abs_residual_ctrl"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_channel_source_closure_abs_residual": max(
            max(
                float(
                    r[
                        "channel_source_closure_max_abs_residual_corr"
                    ]
                ),
                float(
                    r[
                        "channel_source_closure_max_abs_residual_ctrl"
                    ]
                ),
                float(
                    r[
                        "pair_channel_source_closure_max_abs_residual"
                    ]
                ),
            )
            for r in item_rows
        ),
        "max_partition_source_closure_abs_residual": max(
            float(r["max_partition_source_closure_abs_residual"])
            for r in item_rows
        ),
        "max_parent_incoming_source_reproduction_abs_residual": max(
            float(
                r[
                    "max_parent_incoming_source_reproduction_abs_residual"
                ]
            )
            for r in item_rows
        ),
        "max_parent_channel_mean_d_in_reproduction_abs_residual":
            max_parent_channel_repro,
        "max_parent_cumulative_mean_d_in_reproduction_abs_residual":
            max_parent_cumulative_repro,
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
            "delta_r20_l2",
            "delta_y20_l2",
            "delta_r21_l2",
            "q_add20_eps_l2",
            "q_r20_l2",
            "q_y20_l2",
            "q_eps20_l2",
            "q_in21_l2",
            "s22_matched",
            "s22_swapped",
            "s22_bar",
            "residual_space_cross",
            "residual_space_cross_fraction",
            "residual_space_addition_ratio",
        ):
            summary["role_diagnostics"][role][field] = _aggregate(
                r[f"{field}_{role}"] for r in item_rows
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
    raise Layer21ResidualConstructionError("NO_COMMON_PREFLIGHT_PAIR")


def run_branch(
    row: Mapping[str, Any],
    parent: Any,
    stack: Mapping[str, Any],
    runtime: Mapping[str, Any],
    token_ids: Sequence[int],
):
    target = int(row["anchor"]) + TARGET_K
    return capture_branch_with_layer20(
        parent,
        stack,
        runtime,
        token_ids,
        row["targets"],
        target,
    )


def run_role(
    row: Mapping[str, Any],
    parent: Any,
    stack: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
):
    idx = int(row["local_template_index"])
    require(idx in parent_items, f"IMMEDIATE_PARENT_ITEM_MISSING:{idx}")

    matched_prefix, swapped_prefix = _prefixes(parent, stack, row)

    (
        _mpr,
        matched_rms,
        matched_layer21,
        matched_layer20,
    ) = run_branch(
        row,
        parent,
        stack,
        runtime,
        matched_prefix,
    )
    (
        _spr,
        swapped_rms,
        swapped_layer21,
        swapped_layer20,
    ) = run_branch(
        row,
        parent,
        stack,
        runtime,
        swapped_prefix,
    )

    return role_decomposition(
        row,
        matched_rms,
        swapped_rms,
        matched_layer21,
        swapped_layer21,
        matched_layer20,
        swapped_layer20,
        runtime,
        parent_items[idx],
    )


def synthetic_factorization_check() -> dict[str, float]:
    import torch

    torch.manual_seed(127)
    gamma = torch.rand(HIDDEN_SIZE, dtype=torch.float64) + 0.25
    w = (
        torch.randn(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            dtype=torch.float64,
        )
        * 0.01
    )

    dr20 = torch.randn(HIDDEN_SIZE, dtype=torch.float64)
    dy20 = torch.randn(HIDDEN_SIZE, dtype=torch.float64)
    eps = torch.randn(HIDDEN_SIZE, dtype=torch.float64) * 1e-8
    dr21 = dr20 + dy20 + eps
    sbar = 0.041

    q_r20 = gamma * (sbar * dr20)
    q_y20 = gamma * (sbar * dy20)
    q_eps20 = gamma * (sbar * eps)
    q_total = gamma * (sbar * dr21)

    q_residual = float(
        torch.max(
            torch.abs(q_total - (q_r20 + q_y20 + q_eps20))
        ).item()
    )
    require(
        q_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_Q_SOURCE_IDENTITY_FAILURE:{q_residual}",
    )

    h_r20 = torch.mv(w, q_r20)
    h_y20 = torch.mv(w, q_y20)
    h_eps20 = torch.mv(w, q_eps20)
    h_total = torch.mv(w, q_total)
    denom = 1.9

    e_r20 = h_r20 * h_r20 / denom
    e_y20 = h_y20 * h_y20 / denom
    e_ry20 = 2.0 * h_r20 * h_y20 / denom
    e_eps20 = (
        h_eps20 * h_eps20
        + 2.0 * h_r20 * h_eps20
        + 2.0 * h_y20 * h_eps20
    ) / denom
    e_total = h_total * h_total / denom

    e_residual = float(
        torch.max(
            torch.abs(
                e_total - (e_r20 + e_y20 + e_ry20 + e_eps20)
            )
        ).item()
    )
    require(
        e_residual <= VECTOR_ABS_TOL,
        f"SYNTHETIC_CHANNEL_SOURCE_IDENTITY_FAILURE:{e_residual}",
    )

    return {
        "q_source_identity_max_abs_residual": q_residual,
        "channel_source_identity_max_abs_residual": e_residual,
    }


def print_plan(
    repo: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    synth: Mapping[str, float],
) -> None:
    print(
        "=== LAYER21 CURRENT-TOKEN RESIDUAL-CONSTRUCTION "
        "ROUTING-SOURCE AUDIT PLAN ==="
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
    print("hidden_size =", HIDDEN_SIZE)
    print("intermediate_size =", INTERMEDIATE_SIZE)
    print("scientific_question =", QUESTION)
    print(
        "residual_identity = "
        "delta_R21 = delta_R20 + delta_Y20 + Q_add20_eps"
    )
    print(
        "routing_identity = "
        "D_in21 = D_r20 + D_y20 + D_ry20 + D_eps20"
    )
    print(
        "strong_kernel_channel_count =",
        parent_summary["strong_kernel_channel_count"],
    )
    print(
        "weak_kernel_channel_count =",
        parent_summary["weak_kernel_channel_count"],
    )
    print("lag0_kernel_rms =", parent_summary["lag0_kernel_rms"])
    print("parent_runtime_head =", parent_manifest["runtime_git_head"])
    for part, value in EXPECTED_PARENT_MEAN_IN.items():
        print(f"expected_parent_mean_delta_in_{part} =", value)
    print(
        "branch_add_reconstruction_rel_tol =",
        BRANCH_ADD_RECON_REL_TOL,
    )
    print(
        "difference_add_reconstruction_relative_residual_gate = "
        "diagnostic_only"
    )
    print("vector_abs_tol =", VECTOR_ABS_TOL)
    print("squared_energy_rel_tol =", SQUARED_ENERGY_REL_TOL)
    print("parent_scalar_rel_tol =", PARENT_SCALAR_REL_TOL)
    print("parent_scalar_abs_tol =", PARENT_SCALAR_ABS_TOL)
    print(
        "synthetic_q_source_identity_max_abs_residual =",
        synth["q_source_identity_max_abs_residual"],
    )
    print(
        "synthetic_channel_source_identity_max_abs_residual =",
        synth["channel_source_identity_max_abs_residual"],
    )
    print("parent_item_channel_cumulative_bridges_preregistered = True")
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
    stack: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
    handoff_path: Path,
):
    runtime = resolve_runtime(
        root,
        parent,
        stack,
        handoff_path,
    )

    corr_row, ctrl_row = _first_common_pair_rows(
        stack["ctx"]["plan"],
        stack["ctx"]["cohort"],
    )
    records = {}
    forward_count = 0

    for row in (corr_row, ctrl_row):
        rec = run_role(
            row,
            parent,
            stack,
            runtime,
            parent_items,
        )
        records[str(row["role"])] = rec
        forward_count += 2

    require(
        forward_count == EXPECTED_PREFLIGHT_FORWARD_COUNT,
        "PREFLIGHT_FORWARD_COUNT_MISMATCH",
    )
    require(
        set(records) == {"corr", "ctrl"},
        "PREFLIGHT_ROLE_SET_MISMATCH",
    )

    idx = int(records["corr"]["idx"])
    item, _ = pair_role_records(
        records["corr"],
        records["ctrl"],
        runtime,
        parent_items[idx],
    )

    print(
        "PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_"
        "ROUTING_SOURCE_RUNTIME_PREFLIGHT"
    )
    print("local_template_index =", idx)
    print("model_forward_count =", forward_count)
    print("scientific_population_accessed = True")
    print("scientific_evidence_emitted = False")
    print("layer20_r21_equals_parent_layer21_input = True")
    print("raw_vectors_persisted = False")
    print("lag0_kernel_rms =", runtime["operator"]["rms"])
    print(
        "strong_kernel_channel_count =",
        int(runtime["operator"]["strong_mask"].sum().item()),
    )
    print(
        "weak_kernel_channel_count =",
        int(runtime["operator"]["weak_mask"].sum().item()),
    )

    for role in ("corr", "ctrl"):
        for field in (
            "delta_r20_l2",
            "delta_y20_l2",
            "delta_r21_l2",
            "q_add20_eps_l2",
            "q_r20_l2",
            "q_y20_l2",
            "q_in21_l2",
            "s22_bar",
            "residual_space_cross",
            "residual_space_addition_ratio",
        ):
            print(f"{field}_{role} =", item[f"{field}_{role}"])

    for part in ("all", "strong", "weak"):
        for key in ("total", "r20", "y20", "ry20", "eps20"):
            print(
                f"delta_{key}_{part} =",
                item[f"delta_{key}_{part}"],
            )

    print(
        "max_branch_add_reconstruction_relative_residual =",
        max(
            item[
                "branch_add_reconstruction_relative_residual_corr"
            ],
            item[
                "branch_add_reconstruction_relative_residual_ctrl"
            ],
        ),
    )
    print(
        "max_difference_add_reconstruction_relative_residual_diagnostic =",
        max(
            item[
                "difference_add_reconstruction_relative_residual_diagnostic_corr"
            ],
            item[
                "difference_add_reconstruction_relative_residual_diagnostic_ctrl"
            ],
        ),
    )
    print(
        "max_add_error_difference_identity_abs_residual =",
        max(
            item[
                "add_error_difference_identity_abs_residual_corr"
            ],
            item[
                "add_error_difference_identity_abs_residual_ctrl"
            ],
        ),
    )
    print(
        "pair_channel_source_closure_max_abs_residual =",
        item["pair_channel_source_closure_max_abs_residual"],
    )
    print(
        "max_parent_incoming_source_reproduction_abs_residual =",
        item[
            "max_parent_incoming_source_reproduction_abs_residual"
        ],
    )


def execute(
    root: Path,
    parent: Any,
    stack: Mapping[str, Any],
    repo: Mapping[str, Any],
    parent_items: Mapping[int, Mapping[str, Any]],
    parent_channels: Mapping[int, Mapping[str, Any]],
    parent_cumulative: Mapping[int, Mapping[str, Any]],
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
        stack,
        handoff_path,
    )

    role_records: dict[tuple[int, str], dict[str, Any]] = {}
    forward_count = 0

    for n, row in enumerate(stack["ctx"]["plan"], start=1):
        idx = int(row["local_template_index"])
        matched_prefix, swapped_prefix = _prefixes(
            parent,
            stack,
            row,
        )

        if idx in stack["ctx"]["cohort"]:
            rec = run_role(
                row,
                parent,
                stack,
                runtime,
                parent_items,
            )
            key = (idx, str(row["role"]))
            require(
                key not in role_records,
                f"ROLE_RECORD_DUPLICATE:{key}",
            )
            role_records[key] = rec
        else:
            run_branch(
                row,
                parent,
                stack,
                runtime,
                matched_prefix,
            )
            run_branch(
                row,
                parent,
                stack,
                runtime,
                swapped_prefix,
            )

        forward_count += 2
        if n % 16 == 0 or n == len(stack["ctx"]["plan"]):
            print(
                f"PROGRESS pair_roles={n}/{len(stack['ctx']['plan'])} "
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
    vector_lists = {
        k: []
        for k in ("total", "r20", "y20", "ry20", "eps20")
    }

    for idx in sorted(stack["ctx"]["cohort"]):
        corr = role_records[(int(idx), "corr")]
        ctrl = role_records[(int(idx), "ctrl")]
        item, vectors = pair_role_records(
            corr,
            ctrl,
            runtime,
            parent_items[int(idx)],
        )
        item_rows.append(item)
        for key in vector_lists:
            vector_lists[key].append(vectors[key])

    matrices = {
        key: torch.stack(values, dim=0)
        for key, values in vector_lists.items()
    }

    (
        channel_rows,
        cumulative_rows,
        max_parent_channel_repro,
        max_parent_cumulative_repro,
    ) = build_channel_outputs(
        matrices,
        runtime,
        parent_channels,
        parent_cumulative,
    )

    attribution = component_attribution(item_rows)
    summary = make_summary(
        item_rows,
        runtime,
        attribution,
        max_parent_channel_repro,
        max_parent_cumulative_repro,
    )

    for part, expected in EXPECTED_PARENT_MEAN_IN.items():
        got = attribution[part]["mean_signed_components"]["total"]
        require(
            math.isclose(
                got,
                expected,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_MEAN_INCOMING_SOURCE_MISMATCH:"
            f"{part}:{got}:{expected}",
        )

    partial_dir.mkdir(parents=True, exist_ok=False)
    item_path = (
        partial_dir
        / "layer21_current_token_residual_construction_routing_source_item_metrics.jsonl"
    )
    channel_path = (
        partial_dir
        / "layer21_current_token_residual_construction_routing_source_channel_summary.jsonl"
    )
    cumulative_path = (
        partial_dir
        / "layer21_current_token_residual_construction_routing_source_kernel_rank_cumulative.jsonl"
    )
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
        "lag0_kernel_rms": float(runtime["operator"]["rms"]),
        "strong_kernel_channel_count": int(
            runtime["operator"]["strong_mask"].sum().item()
        ),
        "weak_kernel_channel_count": int(
            runtime["operator"]["weak_mask"].sum().item()
        ),
        "equal_kernel_channel_count": int(
            runtime["operator"]["equal_mask"].sum().item()
        ),
        "mamba_source_sha256": runtime["binding"].source_sha256,
        "block_forward_sha256": EXPECTED_BLOCK_FORWARD_SHA256,
        "backbone_forward_sha256":
            EXPECTED_BACKBONE_FORWARD_SHA256,
        "branch_add_reconstruction_rel_tol":
            BRANCH_ADD_RECON_REL_TOL,
        "difference_add_reconstruction_relative_residual_is_diagnostic_only":
            True,
        "vector_abs_tol": VECTOR_ABS_TOL,
        "squared_energy_rel_tol": SQUARED_ENERGY_REL_TOL,
        "parent_scalar_rel_tol": PARENT_SCALAR_REL_TOL,
        "parent_scalar_abs_tol": PARENT_SCALAR_ABS_TOL,
        "kernel_rms_rel_tol": KERNEL_RMS_REL_TOL,
        "kernel_rms_abs_tol": KERNEL_RMS_ABS_TOL,
        "handoff_zip_sha256": runtime["handoff"]["zip_sha256"],
        "checkpoint_sha256":
            runtime["handoff"]["checkpoint_sha256"],
        "encoder_canonical_digest":
            runtime["encoder"]["canonical_digest"],
        "encoder_raw_concat_digest":
            runtime["encoder"]["raw_concat_digest"],
        "layer20_r21_equals_parent_layer21_input": True,
        "parent_incoming_source_routing_match": True,
        "parent_channel_mean_d_in_match": True,
        "parent_cumulative_mean_d_in_match": True,
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
            "read-only layer20 block-pre/mixer-post/block-post hooks "
            "wrapped around the frozen immediate-parent layer21/layer22 "
            "capture; layer20 block output R21 must exactly equal parent "
            "layer21 block input in float32; source algebra and fixed "
            "layer22 W_H propagation are evaluated in float64"
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

    print(
        "PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_"
        "ROUTING_SOURCE_EXECUTION"
    )
    print("output_dir =", final_dir)
    print("model_forward_count =", forward_count)
    print("layer20_r21_equals_parent_layer21_input = True")
    print("parent_incoming_source_routing_match = True")
    print("parent_channel_mean_d_in_match = True")
    print("parent_cumulative_mean_d_in_match = True")
    print("lag0_kernel_rms =", runtime["operator"]["rms"])
    print(
        "strong_kernel_channel_count =",
        int(runtime["operator"]["strong_mask"].sum().item()),
    )
    print(
        "weak_kernel_channel_count =",
        int(runtime["operator"]["weak_mask"].sum().item()),
    )
    for part in ("all", "strong", "weak"):
        c = attribution[part]["mean_signed_components"]
        print(
            f"{part}_mean_components "
            f"total={c['total']} r20={c['r20']} y20={c['y20']} "
            f"interaction={c['ry20']} numerical_bridge={c['eps20']}"
        )
        print(
            f"{part}_largest_absolute_scientific_component =",
            attribution[part][
                "largest_absolute_scientific_component"
            ],
        )
    sign = attribution["strong"][
        "qualitative_parent_signature_component"
    ]
    print(
        "r20_carries_parent_sign_structure =",
        sign["r20"],
    )
    print(
        "y20_carries_parent_sign_structure =",
        sign["y20"],
    )
    print(
        "interaction_carries_parent_sign_structure =",
        sign["ry20"],
    )
    print(
        "max_branch_add_reconstruction_relative_residual =",
        summary[
            "max_branch_add_reconstruction_relative_residual"
        ],
    )
    print(
        "max_difference_add_reconstruction_relative_residual_diagnostic =",
        summary[
            "max_difference_add_reconstruction_relative_residual_diagnostic"
        ],
    )
    print(
        "max_add_error_difference_identity_abs_residual =",
        summary[
            "max_add_error_difference_identity_abs_residual"
        ],
    )
    print(
        "max_channel_source_closure_abs_residual =",
        summary["max_channel_source_closure_abs_residual"],
    )
    print(
        "max_parent_incoming_source_reproduction_abs_residual =",
        summary[
            "max_parent_incoming_source_reproduction_abs_residual"
        ],
    )
    print(
        "max_parent_channel_mean_d_in_reproduction_abs_residual =",
        summary[
            "max_parent_channel_mean_d_in_reproduction_abs_residual"
        ],
    )
    print(
        "max_parent_cumulative_mean_d_in_reproduction_abs_residual =",
        summary[
            "max_parent_cumulative_mean_d_in_reproduction_abs_residual"
        ],
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
        parent_cumulative,
    ) = authenticate_immediate_parent_evidence(root)
    stack = build_parent_stack(root, parent)
    synth = synthetic_factorization_check()

    print_plan(
        repo,
        parent_summary,
        parent_manifest,
        synth,
    )

    if args.static_preflight:
        print("scientific_model_forward_executed = False")
        print(
            "PASS_LAYER21_CURRENT_TOKEN_RESIDUAL_CONSTRUCTION_"
            "ROUTING_SOURCE_STATIC_PREFLIGHT"
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
            stack,
            parent_items,
            handoff,
        )
        return 0

    require(args.execute, "UNKNOWN_MODE")
    require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
    execute(
        root,
        parent,
        stack,
        repo,
        parent_items,
        parent_channels,
        parent_cumulative,
        handoff,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Layer21ResidualConstructionError as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        raise SystemExit(2)
