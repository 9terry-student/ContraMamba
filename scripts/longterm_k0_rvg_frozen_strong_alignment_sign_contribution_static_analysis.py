#!/usr/bin/env python3
"""K0-RVG frozen-strong alignment sign/co-contribution static analyzer.

This script performs a deterministic, model-free transform of the frozen
layer20->21 strong-interaction geometry evidence.

Primary exact decomposition:
    I_b = P_b + N_b
    delta-I = delta-P + delta-N

where:
    P_b >= 0 is positive same-sign co-contribution mass,
    N_b <= 0 is negative opposite-sign contribution mass,
    delta-P = P_corr - P_ctrl,
    delta-N = N_corr - N_ctrl.

Positive delta-N means cancellation relief because N is negative.

No model code, checkpoint, handoff ZIP, tokenizer, logits, task heads,
training, intervention, PCA/SVD, learned projection, or post-hoc subset search
is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "fe73f67e84176f5b7d468287d8084a8836e9a240"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_alignment_sign_contribution_"
    "static_design_candidate.md"
)
AUTHORITY_SHA256 = (
    "4298eecd25dce651f772d249c2b6e7c9830a447bd3595a344ca28ced94d76909"
)
AUTHORITY_BLOB = "7e9beaa78bad01b7ae4564de09635e0cdcc36b75"

AMENDMENT_FREEZE_COMMIT = "76ed124a3db439c6371bda4c5170a56455789b76"
AMENDMENT_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_alignment_sign_contribution_"
    "static_design_provenance_correction_amendment_candidate.md"
)
AMENDMENT_SHA256 = (
    "f5b6266d03d669f0580f8d40221e973ffff0f3a3ce1f3333fd722ff45bc290a5"
)
AMENDMENT_BLOB = "4bae34c23ce4045e1b43dbcef5249710708416e2"

PARENT_EVIDENCE_FREEZE = "431e8faa6e5c82a20d87f532b4ab960fcf641ec2"
PARENT_IMPLEMENTATION = "0b1168182a265bc405652d2ce519f63310433c3e"

PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1"
)

PARENT_ITEM_REL = (
    PARENT_RUN_DIR
    + "/layer20_y20_strong_interaction_geometry_item_metrics.jsonl"
)
PARENT_ITEM_SHA256 = (
    "e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe"
)

PARENT_CHANNEL_REL = (
    PARENT_RUN_DIR
    + "/layer20_y20_strong_interaction_geometry_channel_validation.jsonl"
)
PARENT_CHANNEL_SHA256 = (
    "111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942"
)

PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = (
    "35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798"
)

PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/execution_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee"
)

PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_layer20_y20_strong_interaction_geometry_"
    "validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = (
    "6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163"
)

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_frozen_strong_alignment_sign_contribution_static_analysis.py"
)

K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

EXPECTED_COMMON_COUNT = 330
EXPECTED_STRONG_COUNT = 240
EXPECTED_WEAK_COUNT = 1296
EXPECTED_EQUAL_COUNT = 0
EXPECTED_SOURCE_BLOCK = 20
EXPECTED_TARGET_RESIDUAL_LAYER = 21
EXPECTED_PARENT_MAP_LAYER = 22
EXPECTED_K = 2
EXPECTED_PARENT_STRONG_INTERACTION_MEAN = 0.760594723140676

EXPECTED_CORR_POS_MEAN = 2.267583155684786
EXPECTED_CTRL_POS_MEAN = 1.7361014562993915
EXPECTED_CORR_NEG_MEAN = -0.5449066994977081
EXPECTED_CTRL_NEG_MEAN = -0.7740197232529892

EXPECTED_CORR_SAME_COUNT_MEAN = 140.5818181818182
EXPECTED_CTRL_SAME_COUNT_MEAN = 136.03030303030303
EXPECTED_CORR_OPP_COUNT_MEAN = 99.41818181818182
EXPECTED_CTRL_OPP_COUNT_MEAN = 103.96969696969697
EXPECTED_ZERO_COUNT_MEAN = 0.0

PARENT_ITEM_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-item-v1"
)
PARENT_CHANNEL_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-channel-validation-v1"
)
PARENT_SUMMARY_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-summary-v1"
)
PARENT_MANIFEST_SCHEMA = (
    "k0-rvg-layer20-y20-strong-interaction-geometry-execution-manifest-v1"
)

ITEM_SCHEMA = (
    "k0-rvg-frozen-strong-alignment-sign-contribution-item-v1"
)
SUMMARY_SCHEMA = (
    "k0-rvg-frozen-strong-alignment-sign-contribution-summary-v1"
)
MANIFEST_SCHEMA = (
    "k0-rvg-frozen-strong-alignment-sign-contribution-static-analysis-manifest-v1"
)

PARENT_SCALAR_REL_TOL = 1e-13
PARENT_SCALAR_ABS_TOL = 1e-13
IDENTITY_ABS_TOL = 5e-12
CHANNEL_BRIDGE_ABS_TOL = 5e-12

QUESTION = (
    "Within the frozen strong-240 R20xY20 interaction, is corr's larger "
    "alignment-driven interaction caused primarily by increased positive "
    "same-sign co-contribution, by reduced negative opposite-sign "
    "cancellation, or by a mixture of both?"
)


class StaticAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticAnalysisError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticAnalysisError(
            f"GIT_FAILURE:{' '.join(args)}"
        ) from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", spec],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticAnalysisError(
            f"GIT_SHOW_FAILURE:{spec}"
        ) from exc


def _status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def authenticate_repo(
    root: Path,
    execute_mode: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(
        branch == EXPECTED_BRANCH,
        f"GIT_BRANCH_MISMATCH:{branch}",
    )

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (AMENDMENT_FREEZE_COMMIT, "AMENDMENT"),
        (PARENT_EVIDENCE_FREEZE, "PARENT_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            rc == 0,
            f"{label}_FREEZE_NOT_ANCESTOR",
        )

    allowed_output_prefix = None
    if output_dir is not None:
        try:
            allowed_output_prefix = (
                output_dir.resolve()
                .relative_to(root.resolve())
                .as_posix()
                .rstrip("/")
            )
        except ValueError:
            allowed_output_prefix = None

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()

    for line in status:
        path = _status_path(line)
        xy = line[:2]

        if path in K1_UNTRACKED:
            require(
                xy == "??",
                f"K1_STATE_CHANGED:{line}",
            )
            continue

        if not execute_mode and path == RUNNER_REL:
            require(
                xy == "??",
                f"STATIC_ANALYZER_STATE_UNEXPECTED:{line}",
            )
            continue

        if (
            execute_mode
            and allowed_output_prefix
            and path.startswith(allowed_output_prefix + "/")
        ):
            raise StaticAnalysisError(
                f"OUTPUT_ALREADY_PRESENT_IN_WORKTREE:{line}"
            )

        raise StaticAnalysisError(
            f"UNEXPECTED_WORKTREE_CHANGE:{line}"
        )

    analyzer = root / RUNNER_REL
    require(
        analyzer.is_file(),
        "STATIC_ANALYZER_FILE_MISSING",
    )

    if execute_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            tracked == 0,
            "EXECUTE_REQUIRES_TRACKED_ANALYZER",
        )
        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            clean == 0,
            "EXECUTE_ANALYZER_WORKTREE_DRIFT",
        )

    return {
        "branch": branch,
        "head": head,
        "status": status,
        "execute_mode": execute_mode,
    }


def _authenticate_file(
    root: Path,
    commit: str,
    rel: str,
    expected_sha256: str,
    label: str,
    expected_blob: str | None = None,
) -> bytes:
    raw = git_bytes(root, f"{commit}:{rel}")

    require(
        sha256_bytes(raw) == expected_sha256,
        f"{label}_SHA256_MISMATCH",
    )

    if expected_blob is not None:
        require(
            git(root, "rev-parse", f"{commit}:{rel}")
            == expected_blob,
            f"{label}_BLOB_MISMATCH",
        )

    current = root / rel
    require(
        current.is_file(),
        f"{label}_WORKTREE_FILE_MISSING",
    )
    require(
        current.read_bytes() == raw,
        f"{label}_WORKTREE_DRIFT",
    )
    return raw


def authenticate_authority(root: Path) -> None:
    _authenticate_file(
        root,
        AUTHORITY_FREEZE_COMMIT,
        AUTHORITY_REL,
        AUTHORITY_SHA256,
        "AUTHORITY",
        AUTHORITY_BLOB,
    )
    _authenticate_file(
        root,
        AMENDMENT_FREEZE_COMMIT,
        AMENDMENT_REL,
        AMENDMENT_SHA256,
        "AMENDMENT",
        AMENDMENT_BLOB,
    )


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise StaticAnalysisError(
            f"{label}_JSON_PARSE_FAILURE"
        ) from exc
    require(
        isinstance(obj, dict),
        f"{label}_NOT_OBJECT",
    )
    return obj


def _load_jsonl_bytes(
    raw: bytes,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise StaticAnalysisError(
                f"{label}_JSONL_PARSE_FAILURE:{line_no}"
            ) from exc
        require(
            isinstance(obj, dict),
            f"{label}_ROW_NOT_OBJECT:{line_no}",
        )
        rows.append(obj)
    return rows


def authenticate_parent_evidence(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    item_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_ITEM_REL,
        PARENT_ITEM_SHA256,
        "PARENT_ITEM",
    )
    channel_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_CHANNEL_REL,
        PARENT_CHANNEL_SHA256,
        "PARENT_CHANNEL",
    )
    summary_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_SUMMARY_REL,
        PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY",
    )
    manifest_raw = _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_MANIFEST_REL,
        PARENT_MANIFEST_SHA256,
        "PARENT_MANIFEST",
    )
    _authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_REPORT_REL,
        PARENT_REPORT_SHA256,
        "PARENT_REPORT",
    )

    items = _load_jsonl_bytes(
        item_raw,
        "PARENT_ITEM",
    )
    channels = _load_jsonl_bytes(
        channel_raw,
        "PARENT_CHANNEL",
    )
    summary = _load_json_bytes(
        summary_raw,
        "PARENT_SUMMARY",
    )
    manifest = _load_json_bytes(
        manifest_raw,
        "PARENT_MANIFEST",
    )

    return items, channels, summary, manifest


def close_parent(a: float, b: float) -> bool:
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PARENT_SCALAR_REL_TOL,
        abs_tol=PARENT_SCALAR_ABS_TOL,
    )


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    require(
        math.isfinite(result),
        f"NONFINITE:{label}",
    )
    return result


def validate_parent_manifest(
    manifest: Mapping[str, Any],
) -> None:
    require(
        manifest.get("schema_version")
        == PARENT_MANIFEST_SCHEMA,
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("runtime_git_head")
        == PARENT_IMPLEMENTATION,
        "PARENT_RUNTIME_HEAD_MISMATCH",
    )
    require(
        manifest.get("common_ddsssss_item_count")
        == EXPECTED_COMMON_COUNT,
        "PARENT_MANIFEST_COMMON_COUNT_MISMATCH",
    )
    require(
        manifest.get("strong_kernel_channel_count")
        == EXPECTED_STRONG_COUNT,
        "PARENT_MANIFEST_STRONG_COUNT_MISMATCH",
    )
    require(
        manifest.get("weak_kernel_channel_count")
        == EXPECTED_WEAK_COUNT,
        "PARENT_MANIFEST_WEAK_COUNT_MISMATCH",
    )
    require(
        manifest.get("equal_kernel_channel_count")
        == EXPECTED_EQUAL_COUNT,
        "PARENT_MANIFEST_EQUAL_COUNT_MISMATCH",
    )
    require(
        manifest.get("source_block")
        == EXPECTED_SOURCE_BLOCK,
        "PARENT_MANIFEST_SOURCE_BLOCK_MISMATCH",
    )
    require(
        manifest.get("target_residual_layer")
        == EXPECTED_TARGET_RESIDUAL_LAYER,
        "PARENT_MANIFEST_TARGET_LAYER_MISMATCH",
    )
    require(
        manifest.get("parent_map_layer")
        == EXPECTED_PARENT_MAP_LAYER,
        "PARENT_MANIFEST_PARENT_MAP_MISMATCH",
    )
    require(
        manifest.get("relative_coordinate")
        == EXPECTED_K,
        "PARENT_MANIFEST_K_MISMATCH",
    )
    require(
        manifest.get("parent_item_strong_interaction_match")
        is True,
        "PARENT_MANIFEST_ITEM_MATCH_FALSE",
    )
    require(
        manifest.get(
            "parent_strong_channel_mean_d_ry20_match"
        )
        is True,
        "PARENT_MANIFEST_CHANNEL_MATCH_FALSE",
    )

    forbidden_false = (
        "tokenizer_invoked",
        "logits_read",
        "task_heads_executed",
        "training_executed",
        "causal_intervention_executed",
        "pca_svd_whitening_or_learned_geometry_executed",
        "posthoc_layer_lag_channel_item_or_window_search_executed",
        "raw_vectors_persisted",
        "raw_per_item_channel_vectors_persisted",
    )
    for key in forbidden_false:
        require(
            manifest.get(key) is False,
            f"PARENT_MANIFEST_FORBIDDEN_FLAG:{key}",
        )


def validate_parent_summary(
    summary: Mapping[str, Any],
) -> None:
    require(
        summary.get("schema_version")
        == PARENT_SUMMARY_SCHEMA,
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count")
        == EXPECTED_COMMON_COUNT,
        "PARENT_SUMMARY_COMMON_COUNT_MISMATCH",
    )
    require(
        summary.get("strong_kernel_channel_count")
        == EXPECTED_STRONG_COUNT,
        "PARENT_SUMMARY_STRONG_COUNT_MISMATCH",
    )
    require(
        summary.get("weak_kernel_channel_count")
        == EXPECTED_WEAK_COUNT,
        "PARENT_SUMMARY_WEAK_COUNT_MISMATCH",
    )
    require(
        summary.get("equal_kernel_channel_count")
        == EXPECTED_EQUAL_COUNT,
        "PARENT_SUMMARY_EQUAL_COUNT_MISMATCH",
    )
    require(
        close_parent(
            summary[
                "reconstructed_parent_strong_interaction_mean"
            ],
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
        ),
        "PARENT_SUMMARY_TOTAL_MISMATCH",
    )

    role = summary["role_diagnostics"]

    expected = {
        ("corr", "positive_interaction_mass"):
            EXPECTED_CORR_POS_MEAN,
        ("ctrl", "positive_interaction_mass"):
            EXPECTED_CTRL_POS_MEAN,
        ("corr", "negative_interaction_mass"):
            EXPECTED_CORR_NEG_MEAN,
        ("ctrl", "negative_interaction_mass"):
            EXPECTED_CTRL_NEG_MEAN,
        ("corr", "same_sign_channel_count"):
            EXPECTED_CORR_SAME_COUNT_MEAN,
        ("ctrl", "same_sign_channel_count"):
            EXPECTED_CTRL_SAME_COUNT_MEAN,
        ("corr", "opposite_sign_channel_count"):
            EXPECTED_CORR_OPP_COUNT_MEAN,
        ("ctrl", "opposite_sign_channel_count"):
            EXPECTED_CTRL_OPP_COUNT_MEAN,
        ("corr", "zero_product_channel_count"):
            EXPECTED_ZERO_COUNT_MEAN,
        ("ctrl", "zero_product_channel_count"):
            EXPECTED_ZERO_COUNT_MEAN,
    }

    for (r, field), target in expected.items():
        got = role[r][field]["mean"]
        require(
            close_parent(got, target),
            f"PARENT_SUMMARY_ROLE_MEAN_MISMATCH:{r}:{field}",
        )


def validate_parent_channels(
    channels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        len(channels) == EXPECTED_STRONG_COUNT,
        f"PARENT_CHANNEL_COUNT_MISMATCH:{len(channels)}",
    )

    indices: set[int] = set()
    ranks: set[int] = set()
    max_bridge = 0.0

    corr_mean_sum = 0.0
    ctrl_mean_sum = 0.0
    delta_mean_sum = 0.0

    corr_positive_count_total = 0
    corr_negative_count_total = 0
    corr_zero_count_total = 0
    ctrl_positive_count_total = 0
    ctrl_negative_count_total = 0
    ctrl_zero_count_total = 0

    for row in channels:
        require(
            row.get("schema_version")
            == PARENT_CHANNEL_SCHEMA,
            "PARENT_CHANNEL_SCHEMA_MISMATCH",
        )

        j = int(row["channel_index"])
        require(
            j not in indices,
            f"PARENT_CHANNEL_DUPLICATE:{j}",
        )
        indices.add(j)

        require(
            row["downstream_partition"] == "strong",
            f"PARENT_CHANNEL_NOT_STRONG:{j}",
        )

        rank = int(row["kernel_magnitude_rank"])
        require(
            rank not in ranks,
            f"PARENT_CHANNEL_RANK_DUPLICATE:{rank}",
        )
        ranks.add(rank)

        corr = finite_float(
            row["corr_mean_signed_product"],
            f"CHANNEL_CORR:{j}",
        )
        ctrl = finite_float(
            row["ctrl_mean_signed_product"],
            f"CHANNEL_CTRL:{j}",
        )
        delta = finite_float(
            row["reconstructed_mean_d_ry20"],
            f"CHANNEL_DELTA:{j}",
        )

        residual = abs((corr - ctrl) - delta)
        require(
            residual <= CHANNEL_BRIDGE_ABS_TOL,
            f"PARENT_CHANNEL_BRIDGE_FAILURE:{j}:{residual}",
        )
        max_bridge = max(max_bridge, residual)

        require(
            math.isclose(
                delta,
                float(row["parent_mean_d_ry20"]),
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=CHANNEL_BRIDGE_ABS_TOL,
            ),
            f"PARENT_CHANNEL_REPRODUCTION_FAILURE:{j}",
        )

        for role in ("corr", "ctrl"):
            n_pos = int(row[f"{role}_positive_product_count"])
            n_neg = int(row[f"{role}_negative_product_count"])
            n_zero = int(row[f"{role}_zero_product_count"])
            require(
                n_pos + n_neg + n_zero
                == EXPECTED_COMMON_COUNT,
                f"PARENT_CHANNEL_SIGN_COUNT_FAILURE:{j}:{role}",
            )

        corr_mean_sum += corr
        ctrl_mean_sum += ctrl
        delta_mean_sum += delta

        corr_positive_count_total += int(
            row["corr_positive_product_count"]
        )
        corr_negative_count_total += int(
            row["corr_negative_product_count"]
        )
        corr_zero_count_total += int(
            row["corr_zero_product_count"]
        )
        ctrl_positive_count_total += int(
            row["ctrl_positive_product_count"]
        )
        ctrl_negative_count_total += int(
            row["ctrl_negative_product_count"]
        )
        ctrl_zero_count_total += int(
            row["ctrl_zero_product_count"]
        )

    require(
        len(indices) == EXPECTED_STRONG_COUNT,
        "PARENT_CHANNEL_UNIQUE_INDEX_COUNT_MISMATCH",
    )
    require(
        len(ranks) == EXPECTED_STRONG_COUNT,
        "PARENT_CHANNEL_UNIQUE_RANK_COUNT_MISMATCH",
    )
    require(
        math.isclose(
            delta_mean_sum,
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
            rel_tol=PARENT_SCALAR_REL_TOL,
            abs_tol=CHANNEL_BRIDGE_ABS_TOL,
        ),
        "PARENT_CHANNEL_POPULATION_SUM_MISMATCH",
    )

    return {
        "max_channel_bridge_abs_residual": max_bridge,
        "corr_mean_signed_product_sum": corr_mean_sum,
        "ctrl_mean_signed_product_sum": ctrl_mean_sum,
        "delta_mean_signed_product_sum": delta_mean_sum,
        "corr_positive_product_count_total":
            corr_positive_count_total,
        "corr_negative_product_count_total":
            corr_negative_count_total,
        "corr_zero_product_count_total":
            corr_zero_count_total,
        "ctrl_positive_product_count_total":
            ctrl_positive_count_total,
        "ctrl_negative_product_count_total":
            ctrl_negative_count_total,
        "ctrl_zero_product_count_total":
            ctrl_zero_count_total,
    }


def build_item_rows(
    parent_items: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    require(
        len(parent_items) == EXPECTED_COMMON_COUNT,
        f"PARENT_ITEM_COUNT_MISMATCH:{len(parent_items)}",
    )

    seen: set[int] = set()
    rows: list[dict[str, Any]] = []

    max_role_closure = 0.0
    max_pair_closure = 0.0
    max_parent_repro = 0.0
    max_count_closure = 0.0

    for parent in parent_items:
        require(
            parent.get("schema_version")
            == PARENT_ITEM_SCHEMA,
            "PARENT_ITEM_SCHEMA_MISMATCH",
        )

        idx = int(parent["local_template_index"])
        require(
            idx not in seen,
            f"PARENT_ITEM_DUPLICATE:{idx}",
        )
        seen.add(idx)

        require(
            parent["source_block"]
            == EXPECTED_SOURCE_BLOCK,
            f"PARENT_ITEM_SOURCE_BLOCK_MISMATCH:{idx}",
        )
        require(
            parent["target_residual_layer"]
            == EXPECTED_TARGET_RESIDUAL_LAYER,
            f"PARENT_ITEM_TARGET_LAYER_MISMATCH:{idx}",
        )
        require(
            parent["parent_map_layer"]
            == EXPECTED_PARENT_MAP_LAYER,
            f"PARENT_ITEM_PARENT_MAP_MISMATCH:{idx}",
        )
        require(
            parent["relative_coordinate"]
            == EXPECTED_K,
            f"PARENT_ITEM_K_MISMATCH:{idx}",
        )
        require(
            parent.get("raw_vectors_persisted") is False,
            f"PARENT_ITEM_RAW_VECTOR_FLAG:{idx}",
        )

        p_corr = finite_float(
            parent["positive_interaction_mass_corr"],
            f"P_CORR:{idx}",
        )
        p_ctrl = finite_float(
            parent["positive_interaction_mass_ctrl"],
            f"P_CTRL:{idx}",
        )
        n_corr = finite_float(
            parent["negative_interaction_mass_corr"],
            f"N_CORR:{idx}",
        )
        n_ctrl = finite_float(
            parent["negative_interaction_mass_ctrl"],
            f"N_CTRL:{idx}",
        )
        i_corr = finite_float(
            parent["I_corr"],
            f"I_CORR:{idx}",
        )
        i_ctrl = finite_float(
            parent["I_ctrl"],
            f"I_CTRL:{idx}",
        )

        require(
            p_corr >= 0.0,
            f"P_CORR_NEGATIVE:{idx}",
        )
        require(
            p_ctrl >= 0.0,
            f"P_CTRL_NEGATIVE:{idx}",
        )
        require(
            n_corr <= 0.0,
            f"N_CORR_POSITIVE:{idx}",
        )
        require(
            n_ctrl <= 0.0,
            f"N_CTRL_POSITIVE:{idx}",
        )

        corr_role_residual = abs(
            i_corr - (p_corr + n_corr)
        )
        ctrl_role_residual = abs(
            i_ctrl - (p_ctrl + n_ctrl)
        )
        role_residual = max(
            corr_role_residual,
            ctrl_role_residual,
        )
        require(
            role_residual <= IDENTITY_ABS_TOL,
            f"ROLE_SIGN_MASS_CLOSURE_FAILURE:{idx}:{role_residual}",
        )
        max_role_closure = max(
            max_role_closure,
            role_residual,
        )

        delta_p = p_corr - p_ctrl
        delta_n = n_corr - n_ctrl
        delta_i = i_corr - i_ctrl
        pair_residual = abs(
            delta_i - (delta_p + delta_n)
        )
        require(
            pair_residual <= IDENTITY_ABS_TOL,
            f"PAIR_SIGN_DECOMPOSITION_FAILURE:{idx}:{pair_residual}",
        )
        max_pair_closure = max(
            max_pair_closure,
            pair_residual,
        )

        parent_total = finite_float(
            parent["reconstructed_delta_I"],
            f"PARENT_TOTAL:{idx}",
        )
        parent_total_2 = finite_float(
            parent["reconstructed_delta_ry20_strong"],
            f"PARENT_TOTAL_2:{idx}",
        )
        require(
            abs(parent_total - parent_total_2)
            <= IDENTITY_ABS_TOL,
            f"PARENT_TOTAL_ALIAS_FAILURE:{idx}",
        )

        parent_repro = abs(
            delta_i - parent_total
        )
        require(
            math.isclose(
                delta_i,
                parent_total,
                rel_tol=PARENT_SCALAR_REL_TOL,
                abs_tol=PARENT_SCALAR_ABS_TOL,
            ),
            f"PARENT_ITEM_TOTAL_REPRODUCTION_FAILURE:"
            f"{idx}:{delta_i}:{parent_total}",
        )
        max_parent_repro = max(
            max_parent_repro,
            parent_repro,
        )

        n_same_corr = int(
            parent["same_sign_channel_count_corr"]
        )
        n_same_ctrl = int(
            parent["same_sign_channel_count_ctrl"]
        )
        n_opp_corr = int(
            parent["opposite_sign_channel_count_corr"]
        )
        n_opp_ctrl = int(
            parent["opposite_sign_channel_count_ctrl"]
        )
        n_zero_corr = int(
            parent["zero_product_channel_count_corr"]
        )
        n_zero_ctrl = int(
            parent["zero_product_channel_count_ctrl"]
        )

        for role, n_same, n_opp, n_zero in (
            (
                "corr",
                n_same_corr,
                n_opp_corr,
                n_zero_corr,
            ),
            (
                "ctrl",
                n_same_ctrl,
                n_opp_ctrl,
                n_zero_ctrl,
            ),
        ):
            require(
                n_same + n_opp + n_zero
                == EXPECTED_STRONG_COUNT,
                f"ITEM_CHANNEL_COUNT_FAILURE:{idx}:{role}",
            )
            require(
                n_same > 0,
                f"ZERO_SAME_SIGN_DENOMINATOR:{idx}:{role}",
            )
            require(
                n_opp > 0,
                f"ZERO_OPPOSITE_SIGN_DENOMINATOR:{idx}:{role}",
            )

        delta_n_same = n_same_corr - n_same_ctrl
        delta_n_opp = n_opp_corr - n_opp_ctrl
        delta_n_zero = n_zero_corr - n_zero_ctrl

        count_closure = abs(
            delta_n_same
            + delta_n_opp
            + delta_n_zero
        )
        require(
            count_closure == 0,
            f"COUNT_DIFFERENCE_CLOSURE_FAILURE:{idx}",
        )
        require(
            n_zero_corr == 0 and n_zero_ctrl == 0,
            f"UNEXPECTED_ZERO_PRODUCT_CHANNEL:{idx}",
        )
        require(
            delta_n_same + delta_n_opp == 0,
            f"SAME_OPPOSITE_COUNT_CLOSURE_FAILURE:{idx}",
        )
        max_count_closure = max(
            max_count_closure,
            float(count_closure),
        )

        pbar_corr = p_corr / n_same_corr
        pbar_ctrl = p_ctrl / n_same_ctrl
        k_corr = -n_corr
        k_ctrl = -n_ctrl
        kbar_corr = k_corr / n_opp_corr
        kbar_ctrl = k_ctrl / n_opp_ctrl

        require(
            pbar_corr >= 0.0 and pbar_ctrl >= 0.0,
            f"PBAR_NEGATIVE:{idx}",
        )
        require(
            kbar_corr >= 0.0 and kbar_ctrl >= 0.0,
            f"KBAR_NEGATIVE:{idx}",
        )

        row = {
            "schema_version": ITEM_SCHEMA,
            "local_template_index": idx,
            "stable_item_id": parent["stable_item_id"],
            "source_block": EXPECTED_SOURCE_BLOCK,
            "target_residual_layer":
                EXPECTED_TARGET_RESIDUAL_LAYER,
            "parent_map_layer":
                EXPECTED_PARENT_MAP_LAYER,
            "relative_coordinate": EXPECTED_K,

            "parent_interaction_total": parent_total,

            "P_corr": p_corr,
            "P_ctrl": p_ctrl,
            "N_corr": n_corr,
            "N_ctrl": n_ctrl,

            "delta_P": delta_p,
            "delta_N": delta_n,
            "reconstructed_delta_I": delta_i,

            "corr_role_closure_abs_residual":
                corr_role_residual,
            "ctrl_role_closure_abs_residual":
                ctrl_role_residual,
            "pair_decomposition_abs_residual":
                pair_residual,
            "parent_reproduction_abs_residual":
                parent_repro,

            "same_sign_channel_count_corr":
                n_same_corr,
            "same_sign_channel_count_ctrl":
                n_same_ctrl,
            "opposite_sign_channel_count_corr":
                n_opp_corr,
            "opposite_sign_channel_count_ctrl":
                n_opp_ctrl,
            "zero_product_channel_count_corr":
                n_zero_corr,
            "zero_product_channel_count_ctrl":
                n_zero_ctrl,

            "delta_same_sign_channel_count":
                delta_n_same,
            "delta_opposite_sign_channel_count":
                delta_n_opp,
            "delta_zero_product_channel_count":
                delta_n_zero,
            "count_difference_closure_abs_residual":
                float(count_closure),

            "Pbar_corr": pbar_corr,
            "Pbar_ctrl": pbar_ctrl,
            "delta_Pbar": pbar_corr - pbar_ctrl,

            "K_corr": k_corr,
            "K_ctrl": k_ctrl,
            "Kbar_corr": kbar_corr,
            "Kbar_ctrl": kbar_ctrl,
            "delta_Kbar": kbar_corr - kbar_ctrl,

            "new_model_execution": False,
            "new_channel_selection": False,
        }
        rows.append(row)

    rows.sort(
        key=lambda r: r["local_template_index"]
    )

    require(
        len(seen) == EXPECTED_COMMON_COUNT,
        "PARENT_ITEM_UNIQUE_COUNT_MISMATCH",
    )

    return rows, {
        "max_role_closure_abs_residual":
            max_role_closure,
        "max_pair_decomposition_abs_residual":
            max_pair_closure,
        "max_parent_reproduction_abs_residual":
            max_parent_repro,
        "max_count_difference_closure_abs_residual":
            max_count_closure,
    }


def aggregate(values: Sequence[float]) -> dict[str, float | int]:
    vals = [float(v) for v in values]
    require(
        bool(vals),
        "EMPTY_AGGREGATE",
    )
    require(
        all(math.isfinite(v) for v in vals),
        "NONFINITE_AGGREGATE",
    )
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def classify(
    g_pos: float,
    g_relief: float,
) -> tuple[str, str]:
    if g_pos * g_relief < 0.0:
        return (
            "OUTCOME_C_MIXED",
            "components oppose each other",
        )

    a_pos = abs(g_pos)
    a_relief = abs(g_relief)

    if a_pos > a_relief:
        return (
            "OUTCOME_A_POSITIVE_CO_CONTRIBUTION_DOMINANT",
            "positive same-sign co-contribution gain is the largest "
            "absolute scientific component",
        )

    if a_relief > a_pos:
        return (
            "OUTCOME_B_CANCELLATION_RELIEF_DOMINANT",
            "negative opposite-sign cancellation relief is the largest "
            "absolute scientific component",
        )

    return (
        "OUTCOME_C_MIXED",
        "components are equal in absolute magnitude",
    )


def build_summary(
    item_rows: Sequence[Mapping[str, Any]],
    parent_summary: Mapping[str, Any],
    channel_validation: Mapping[str, Any],
    maxima: Mapping[str, float],
) -> dict[str, Any]:
    g_total = float(
        statistics.fmean(
            float(r["reconstructed_delta_I"])
            for r in item_rows
        )
    )
    g_pos = float(
        statistics.fmean(
            float(r["delta_P"])
            for r in item_rows
        )
    )
    g_relief = float(
        statistics.fmean(
            float(r["delta_N"])
            for r in item_rows
        )
    )

    aggregate_closure = abs(
        g_total - (g_pos + g_relief)
    )
    require(
        aggregate_closure <= IDENTITY_ABS_TOL,
        f"AGGREGATE_SIGN_DECOMPOSITION_FAILURE:{aggregate_closure}",
    )

    require(
        close_parent(
            g_total,
            EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
        ),
        "PARENT_STRONG_INTERACTION_MEAN_REPRODUCTION_FAILURE",
    )

    role_aggregates: dict[str, dict[str, Any]] = {}
    for role in ("corr", "ctrl"):
        role_aggregates[role] = {}
        field_map = {
            "positive_interaction_mass":
                f"P_{role}",
            "negative_interaction_mass":
                f"N_{role}",
            "cancellation_magnitude":
                f"K_{role}",
            "same_sign_channel_count":
                f"same_sign_channel_count_{role}",
            "opposite_sign_channel_count":
                f"opposite_sign_channel_count_{role}",
            "zero_product_channel_count":
                f"zero_product_channel_count_{role}",
            "positive_mass_per_same_sign_channel":
                f"Pbar_{role}",
            "cancellation_magnitude_per_opposite_sign_channel":
                f"Kbar_{role}",
        }
        for out_name, source_name in field_map.items():
            role_aggregates[role][out_name] = aggregate(
                [
                    float(r[source_name])
                    for r in item_rows
                ]
            )

    expected_means = {
        ("corr", "positive_interaction_mass"):
            EXPECTED_CORR_POS_MEAN,
        ("ctrl", "positive_interaction_mass"):
            EXPECTED_CTRL_POS_MEAN,
        ("corr", "negative_interaction_mass"):
            EXPECTED_CORR_NEG_MEAN,
        ("ctrl", "negative_interaction_mass"):
            EXPECTED_CTRL_NEG_MEAN,
        ("corr", "same_sign_channel_count"):
            EXPECTED_CORR_SAME_COUNT_MEAN,
        ("ctrl", "same_sign_channel_count"):
            EXPECTED_CTRL_SAME_COUNT_MEAN,
        ("corr", "opposite_sign_channel_count"):
            EXPECTED_CORR_OPP_COUNT_MEAN,
        ("ctrl", "opposite_sign_channel_count"):
            EXPECTED_CTRL_OPP_COUNT_MEAN,
    }
    for (role, field), target in expected_means.items():
        require(
            close_parent(
                role_aggregates[role][field]["mean"],
                target,
            ),
            f"PARENT_AGGREGATE_REPRODUCTION_FAILURE:"
            f"{role}:{field}",
        )

    paired = {
        field: aggregate(
            [
                float(r[field])
                for r in item_rows
            ]
        )
        for field in (
            "delta_P",
            "delta_N",
            "reconstructed_delta_I",
            "delta_same_sign_channel_count",
            "delta_opposite_sign_channel_count",
            "delta_Pbar",
            "delta_Kbar",
        )
    }

    abs_mass = abs(g_pos) + abs(g_relief)
    require(
        abs_mass > 0.0,
        "ZERO_ABSOLUTE_COMPONENT_MASS",
    )

    shares = {
        "positive_co_contribution_gain":
            abs(g_pos) / abs_mass,
        "cancellation_relief":
            abs(g_relief) / abs_mass,
    }
    ratios = {
        "positive_co_contribution_gain":
            g_pos / g_total,
        "cancellation_relief":
            g_relief / g_total,
    }

    outcome, outcome_reason = classify(
        g_pos,
        g_relief,
    )

    both_reinforce = (
        g_total > 0.0
        and g_pos > 0.0
        and g_relief > 0.0
    )

    parent_summary_role = parent_summary[
        "role_diagnostics"
    ]
    for role in ("corr", "ctrl"):
        for field in (
            "positive_interaction_mass",
            "negative_interaction_mass",
            "same_sign_channel_count",
            "opposite_sign_channel_count",
            "zero_product_channel_count",
        ):
            require(
                close_parent(
                    role_aggregates[role][field]["mean"],
                    parent_summary_role[role][field]["mean"],
                ),
                f"PARENT_SUMMARY_ITEM_BRIDGE_FAILURE:"
                f"{role}:{field}",
            )

    return {
        "schema_version": SUMMARY_SCHEMA,
        "scientific_question": QUESTION,
        "analysis_type":
            "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",
        "common_ddsssss_item_count":
            EXPECTED_COMMON_COUNT,
        "strong_kernel_channel_count":
            EXPECTED_STRONG_COUNT,
        "source_block":
            EXPECTED_SOURCE_BLOCK,
        "target_residual_layer":
            EXPECTED_TARGET_RESIDUAL_LAYER,
        "parent_map_layer":
            EXPECTED_PARENT_MAP_LAYER,
        "relative_coordinate":
            EXPECTED_K,

        "primary_decomposition": {
            "G_total": g_total,
            "G_pos": g_pos,
            "G_relief": g_relief,
            "aggregate_closure_abs_residual":
                aggregate_closure,
            "absolute_component_mass":
                abs_mass,
            "absolute_component_shares":
                shares,
            "signed_component_to_total_ratios":
                ratios,
            "largest_absolute_scientific_component":
                (
                    "positive_co_contribution_gain"
                    if abs(g_pos) > abs(g_relief)
                    else (
                        "cancellation_relief"
                        if abs(g_relief) > abs(g_pos)
                        else "tie"
                    )
                ),
            "both_components_reinforce_parent_positive_total":
                both_reinforce,
        },

        "scientific_outcome": outcome,
        "scientific_outcome_reason":
            outcome_reason,

        "role_aggregates":
            role_aggregates,
        "paired_diagnostics":
            paired,

        "channel_bridge_validation":
            dict(channel_validation),

        "max_role_closure_abs_residual":
            maxima[
                "max_role_closure_abs_residual"
            ],
        "max_pair_decomposition_abs_residual":
            maxima[
                "max_pair_decomposition_abs_residual"
            ],
        "max_parent_reproduction_abs_residual":
            maxima[
                "max_parent_reproduction_abs_residual"
            ],
        "max_count_difference_closure_abs_residual":
            maxima[
                "max_count_difference_closure_abs_residual"
            ],

        "parent_item_reproduction_match": True,
        "parent_population_reproduction_match": True,
        "parent_summary_bridge_match": True,
        "parent_channel_bridge_match": True,

        "new_model_execution": False,
        "checkpoint_loaded": False,
        "handoff_opened": False,
        "transformers_imported": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_or_learned_geometry_executed": False,
        "posthoc_subset_search_executed": False,
        "raw_vectors_read_or_persisted": False,
    }


def synthetic_check() -> dict[str, float]:
    p_corr = 3.2
    p_ctrl = 2.5
    n_corr = -0.7
    n_ctrl = -1.1

    i_corr = p_corr + n_corr
    i_ctrl = p_ctrl + n_ctrl

    delta_p = p_corr - p_ctrl
    delta_n = n_corr - n_ctrl
    delta_i = i_corr - i_ctrl

    role_corr = abs(
        i_corr - (p_corr + n_corr)
    )
    role_ctrl = abs(
        i_ctrl - (p_ctrl + n_ctrl)
    )
    pair = abs(
        delta_i - (delta_p + delta_n)
    )

    require(
        role_corr <= IDENTITY_ABS_TOL,
        f"SYNTHETIC_CORR_ROLE_FAILURE:{role_corr}",
    )
    require(
        role_ctrl <= IDENTITY_ABS_TOL,
        f"SYNTHETIC_CTRL_ROLE_FAILURE:{role_ctrl}",
    )
    require(
        pair <= IDENTITY_ABS_TOL,
        f"SYNTHETIC_PAIR_FAILURE:{pair}",
    )
    require(
        delta_n > 0.0,
        "SYNTHETIC_RELIEF_SIGN_FAILURE",
    )

    return {
        "corr_role_closure_abs_residual":
            role_corr,
        "ctrl_role_closure_abs_residual":
            role_ctrl,
        "pair_decomposition_abs_residual":
            pair,
        "delta_N":
            delta_n,
    }


def json_bytes(obj: Any) -> bytes:
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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json_bytes(row)
        for row in rows
    )


def print_plan(
    repo: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
    synthetic: Mapping[str, float],
) -> None:
    print(
        "=== FROZEN-STRONG ALIGNMENT SIGN / "
        "CO-CONTRIBUTION STATIC ANALYSIS PLAN ==="
    )
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print(
        "authority_freeze_commit =",
        AUTHORITY_FREEZE_COMMIT,
    )
    print(
        "authority_sha256 =",
        AUTHORITY_SHA256,
    )
    print(
        "authority_blob =",
        AUTHORITY_BLOB,
    )
    print(
        "amendment_freeze_commit =",
        AMENDMENT_FREEZE_COMMIT,
    )
    print(
        "amendment_sha256 =",
        AMENDMENT_SHA256,
    )
    print(
        "amendment_blob =",
        AMENDMENT_BLOB,
    )
    print(
        "parent_evidence_freeze =",
        PARENT_EVIDENCE_FREEZE,
    )
    print(
        "parent_implementation =",
        PARENT_IMPLEMENTATION,
    )
    print(
        "parent_item_sha256 =",
        PARENT_ITEM_SHA256,
    )
    print(
        "parent_channel_sha256 =",
        PARENT_CHANNEL_SHA256,
    )
    print(
        "parent_summary_sha256 =",
        PARENT_SUMMARY_SHA256,
    )
    print(
        "parent_manifest_sha256 =",
        PARENT_MANIFEST_SHA256,
    )
    print(
        "parent_report_sha256 =",
        PARENT_REPORT_SHA256,
    )
    print(
        "common_ddsssss_item_count =",
        EXPECTED_COMMON_COUNT,
    )
    print(
        "strong_kernel_channel_count =",
        EXPECTED_STRONG_COUNT,
    )
    print(
        "expected_parent_strong_interaction_mean =",
        EXPECTED_PARENT_STRONG_INTERACTION_MEAN,
    )
    print(
        "parent_Q_C =",
        parent_summary[
            "component_attribution"
        ]["mean_signed_components"]["Q_C"],
    )
    print(
        "scientific_question =",
        QUESTION,
    )
    print(
        "primary_identity = "
        "delta_I = delta_P + delta_N"
    )
    print(
        "delta_N_interpretation = "
        "positive means reduced negative cancellation"
    )
    print(
        "parent_scalar_rel_tol =",
        PARENT_SCALAR_REL_TOL,
    )
    print(
        "parent_scalar_abs_tol =",
        PARENT_SCALAR_ABS_TOL,
    )
    print(
        "identity_abs_tol =",
        IDENTITY_ABS_TOL,
    )
    print(
        "channel_bridge_abs_tol =",
        CHANNEL_BRIDGE_ABS_TOL,
    )
    for key, value in synthetic.items():
        print(
            f"synthetic_{key} =",
            value,
        )
    print("new_model_execution = False")
    print("checkpoint_loaded = False")
    print("handoff_opened = False")
    print("transformers_imported = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("pca_svd_or_learned_geometry_executed = False")
    print("posthoc_subset_search_executed = False")


def execute(
    root: Path,
    repo: Mapping[str, Any],
    item_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    final_dir = output_dir.resolve()
    partial_dir = Path(
        str(final_dir) + ".partial"
    )

    require(
        not final_dir.exists(),
        f"OUTPUT_DIR_EXISTS:{final_dir}",
    )
    require(
        not partial_dir.exists(),
        f"PARTIAL_OUTPUT_EXISTS:{partial_dir}",
    )

    partial_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    item_path = (
        partial_dir
        / "frozen_strong_alignment_sign_contribution_item_metrics.jsonl"
    )
    summary_path = (
        partial_dir
        / "summary.json"
    )
    manifest_path = (
        partial_dir
        / "static_analysis_manifest.json"
    )

    item_path.write_bytes(
        jsonl_bytes(item_rows)
    )
    summary_path.write_bytes(
        json_bytes(summary)
    )

    analyzer_sha = sha256_file(
        root / RUNNER_REL
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_git_head": repo["head"],
        "analysis_branch": repo["branch"],
        "analyzer_rel": RUNNER_REL,
        "analyzer_sha256": analyzer_sha,

        "authority_freeze_commit":
            AUTHORITY_FREEZE_COMMIT,
        "authority_sha256":
            AUTHORITY_SHA256,
        "authority_blob":
            AUTHORITY_BLOB,

        "amendment_freeze_commit":
            AMENDMENT_FREEZE_COMMIT,
        "amendment_sha256":
            AMENDMENT_SHA256,
        "amendment_blob":
            AMENDMENT_BLOB,

        "parent_evidence_freeze":
            PARENT_EVIDENCE_FREEZE,
        "parent_implementation":
            PARENT_IMPLEMENTATION,
        "parent_item_sha256":
            PARENT_ITEM_SHA256,
        "parent_channel_sha256":
            PARENT_CHANNEL_SHA256,
        "parent_summary_sha256":
            PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256":
            PARENT_MANIFEST_SHA256,
        "parent_report_sha256":
            PARENT_REPORT_SHA256,

        "scientific_question":
            QUESTION,
        "analysis_type":
            "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",

        "common_ddsssss_item_count":
            EXPECTED_COMMON_COUNT,
        "strong_kernel_channel_count":
            EXPECTED_STRONG_COUNT,
        "source_block":
            EXPECTED_SOURCE_BLOCK,
        "target_residual_layer":
            EXPECTED_TARGET_RESIDUAL_LAYER,
        "parent_map_layer":
            EXPECTED_PARENT_MAP_LAYER,
        "relative_coordinate":
            EXPECTED_K,

        "parent_scalar_rel_tol":
            PARENT_SCALAR_REL_TOL,
        "parent_scalar_abs_tol":
            PARENT_SCALAR_ABS_TOL,
        "identity_abs_tol":
            IDENTITY_ABS_TOL,
        "channel_bridge_abs_tol":
            CHANNEL_BRIDGE_ABS_TOL,

        "new_model_execution": False,
        "model_forward_count": 0,
        "checkpoint_loaded": False,
        "handoff_opened": False,
        "transformers_imported": False,
        "tokenizer_invoked": False,
        "logits_read": False,
        "task_heads_executed": False,
        "training_executed": False,
        "causal_intervention_executed": False,
        "pca_svd_or_learned_geometry_executed": False,
        "posthoc_subset_search_executed": False,
        "raw_vectors_read_or_persisted": False,

        "parent_item_reproduction_match": True,
        "parent_population_reproduction_match": True,
        "parent_summary_bridge_match": True,
        "parent_channel_bridge_match": True,

        "outputs": {
            item_path.name:
                sha256_file(item_path),
            summary_path.name:
                sha256_file(summary_path),
        },
    }

    manifest_path.write_bytes(
        json_bytes(manifest)
    )

    os.replace(
        partial_dir,
        final_dir,
    )

    primary = summary["primary_decomposition"]

    print(
        "PASS_FROZEN_STRONG_ALIGNMENT_SIGN_CONTRIBUTION_STATIC_ANALYSIS"
    )
    print(
        "output_dir =",
        final_dir,
    )
    print(
        "new_model_execution = False"
    )
    print(
        "model_forward_count = 0"
    )
    print(
        "common_ddsssss_item_count =",
        EXPECTED_COMMON_COUNT,
    )
    print(
        "strong_kernel_channel_count =",
        EXPECTED_STRONG_COUNT,
    )
    print(
        "G_total =",
        primary["G_total"],
    )
    print(
        "G_pos =",
        primary["G_pos"],
    )
    print(
        "G_relief =",
        primary["G_relief"],
    )
    print(
        "absolute_component_shares =",
        primary["absolute_component_shares"],
    )
    print(
        "largest_absolute_scientific_component =",
        primary[
            "largest_absolute_scientific_component"
        ],
    )
    print(
        "both_components_reinforce_parent_positive_total =",
        primary[
            "both_components_reinforce_parent_positive_total"
        ],
    )
    print(
        "scientific_outcome =",
        summary["scientific_outcome"],
    )

    for role in ("corr", "ctrl"):
        role_agg = summary[
            "role_aggregates"
        ][role]
        for field in (
            "positive_interaction_mass",
            "negative_interaction_mass",
            "same_sign_channel_count",
            "opposite_sign_channel_count",
            "positive_mass_per_same_sign_channel",
            "cancellation_magnitude_per_opposite_sign_channel",
        ):
            print(
                f"{role}_{field}_mean =",
                role_agg[field]["mean"],
            )

    paired = summary["paired_diagnostics"]
    for field in (
        "delta_same_sign_channel_count",
        "delta_opposite_sign_channel_count",
        "delta_Pbar",
        "delta_Kbar",
    ):
        print(
            f"{field}_mean =",
            paired[field]["mean"],
        )

    print(
        "max_role_closure_abs_residual =",
        summary[
            "max_role_closure_abs_residual"
        ],
    )
    print(
        "max_pair_decomposition_abs_residual =",
        summary[
            "max_pair_decomposition_abs_residual"
        ],
    )
    print(
        "max_parent_reproduction_abs_residual =",
        summary[
            "max_parent_reproduction_abs_residual"
        ],
    )
    print(
        "max_channel_bridge_abs_residual =",
        summary[
            "channel_bridge_validation"
        ][
            "max_channel_bridge_abs_residual"
        ],
    )


def parse_args():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--static-preflight",
        action="store_true",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    require(
        not args.static_preflight
        or args.output_dir is None,
        "STATIC_PREFLIGHT_OUTPUT_DIR_FORBIDDEN",
    )
    require(
        not args.execute
        or args.output_dir is not None,
        "EXECUTE_OUTPUT_DIR_REQUIRED",
    )

    repo = authenticate_repo(
        root,
        execute_mode=bool(args.execute),
        output_dir=args.output_dir,
    )
    authenticate_authority(root)

    (
        parent_items,
        parent_channels,
        parent_summary,
        parent_manifest,
    ) = authenticate_parent_evidence(root)

    validate_parent_manifest(
        parent_manifest
    )
    validate_parent_summary(
        parent_summary
    )

    channel_validation = validate_parent_channels(
        parent_channels
    )

    item_rows, maxima = build_item_rows(
        parent_items
    )

    summary = build_summary(
        item_rows,
        parent_summary,
        channel_validation,
        maxima,
    )

    synthetic = synthetic_check()

    print_plan(
        repo,
        parent_summary,
        synthetic,
    )

    if args.static_preflight:
        print(
            "scientific_evidence_emitted = False"
        )
        print(
            "PASS_FROZEN_STRONG_ALIGNMENT_SIGN_CONTRIBUTION_STATIC_PREFLIGHT"
        )
        return 0

    execute(
        root,
        repo,
        item_rows,
        summary,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StaticAnalysisError as exc:
        print(
            "BLOCKED:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(2)
