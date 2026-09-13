#!/usr/bin/env python3
"""K0-RVG frozen-strong sign-contribution itemwise breadth static analyzer.

Model-free deterministic analysis over the frozen 330-item sign-contribution
artifact. No model, checkpoint, handoff, tokenizer, logits, training,
intervention, learned geometry, channel search, item search, or K1 work.
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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

AUTHORITY_FREEZE_COMMIT = "f34ad21beeda00e94904844375839f37c0fbee67"
AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_sign_contribution_"
    "itemwise_breadth_static_design_candidate.md"
)
AUTHORITY_SHA256 = (
    "8a7f6fcf85dc8954379699198ac5336bef54e86d7dad77402a9ed4d727575af5"
)
AUTHORITY_BLOB = "43132b38894f475c1f1755adb4e0a244a9636d18"

PARENT_EVIDENCE_FREEZE = "3f7a9eaa38935fc066f20f81cbce0a9f901909a9"
PARENT_IMPLEMENTATION = "3af43c91d791e2755b977529eb26d65db00c7870"

PARENT_RUN_DIR = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_alignment_sign_contribution_3af43c9_v1"
)
PARENT_ITEM_REL = (
    PARENT_RUN_DIR
    + "/frozen_strong_alignment_sign_contribution_item_metrics.jsonl"
)
PARENT_ITEM_SHA256 = (
    "cd9d3306c7ef2b2d8f2a759be5e7945ccfdb86f0009981f61247b9ddfcf03974"
)
PARENT_SUMMARY_REL = PARENT_RUN_DIR + "/summary.json"
PARENT_SUMMARY_SHA256 = (
    "0696e5887dfbd2fc7eb55d3c726e4e55415858e15522d52efca6ce38ecda1615"
)
PARENT_MANIFEST_REL = PARENT_RUN_DIR + "/static_analysis_manifest.json"
PARENT_MANIFEST_SHA256 = (
    "5e1fdd97236d0e6f2d7b7041207d25af78aba7dae3b15cff9a66d760c365d50f"
)
PARENT_REPORT_REL = (
    "reports/"
    "longterm_k0_rvg_frozen_strong_alignment_sign_contribution_"
    "validated_evidence_analysis_report_candidate.md"
)
PARENT_REPORT_SHA256 = (
    "a8e4e7fe528e9b01a16b26d7d3508938d7f63bbd4cc57153207c9ae4c58767d6"
)

RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_frozen_strong_sign_contribution_itemwise_breadth_static_analysis.py"
)

ALLOWED_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
    "validate_frozen_strong_alignment_sign_contribution_artifacts.py",
}

EXPECTED_ITEM_COUNT = 330
EXPECTED_STRONG_COUNT = 240
MAJORITY_BOUNDARY = 165
EXPECTED_SOURCE_BLOCK = 20
EXPECTED_TARGET_RESIDUAL_LAYER = 21
EXPECTED_PARENT_MAP_LAYER = 22
EXPECTED_K = 2

PARENT_ITEM_SCHEMA = "k0-rvg-frozen-strong-alignment-sign-contribution-item-v1"
PARENT_SUMMARY_SCHEMA = "k0-rvg-frozen-strong-alignment-sign-contribution-summary-v1"
PARENT_MANIFEST_SCHEMA = (
    "k0-rvg-frozen-strong-alignment-sign-contribution-static-analysis-manifest-v1"
)

ITEM_SCHEMA = "k0-rvg-frozen-strong-sign-contribution-itemwise-breadth-item-v1"
SUMMARY_SCHEMA = "k0-rvg-frozen-strong-sign-contribution-itemwise-breadth-summary-v1"
MANIFEST_SCHEMA = (
    "k0-rvg-frozen-strong-sign-contribution-itemwise-breadth-static-analysis-manifest-v1"
)

EXPECTED_MEANS = {
    "t": 0.760594723140676,
    "p": 0.5314816993853949,
    "r": 0.22911302375528114,
    "c": 4.551515151515152,
    "a": 0.0035643520982118104,
    "k": -0.001921654940814375,
}

EXPECTED_MEDIANS = {
    "t": 0.7522478498155687,
    "p": 0.5693259571352562,
    "r": 0.19919127457024965,
    "c": 3.0,
    "a": 0.0036918390252776183,
    "k": -0.0017159590409509089,
}

REL_TOL = 1e-13
ABS_TOL = 5e-12

QUESTION = (
    "Is the validated population-level positive co-contribution dominance "
    "broadly expressed across the frozen common-330 items, or is the positive "
    "population mean produced by a minority of large-effect items despite "
    "substantial itemwise heterogeneity?"
)


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


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
        raise AnalysisError(f"GIT_FAILURE:{' '.join(args)}") from exc


def git_bytes(root: Path, spec: str) -> bytes:
    try:
        return subprocess.check_output(["git", "show", spec], cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError(f"GIT_SHOW_FAILURE:{spec}") from exc


def status_path(line: str) -> str:
    raw = line[3:] if len(line) >= 4 else ""
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]
    return raw.strip('"').replace("\\", "/")


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=ABS_TOL)


def finite(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE:{label}")
    return x


def authenticate_repo(
    root: Path,
    execute_mode: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")

    for ancestor, label in (
        (AUTHORITY_FREEZE_COMMIT, "AUTHORITY"),
        (PARENT_EVIDENCE_FREEZE, "PARENT_EVIDENCE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    output_prefix = None
    if output_dir is not None:
        try:
            output_prefix = (
                output_dir.resolve()
                .relative_to(root.resolve())
                .as_posix()
                .rstrip("/")
            )
        except ValueError:
            output_prefix = None

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        text=True,
    ).splitlines()

    for line in status:
        path = status_path(line)
        xy = line[:2]

        if path in ALLOWED_UNTRACKED:
            require(xy == "??", f"ALLOWED_FILE_STATE_CHANGED:{line}")
            continue

        if not execute_mode and path == RUNNER_REL:
            require(xy == "??", f"ANALYZER_STATE_UNEXPECTED:{line}")
            continue

        if (
            execute_mode
            and output_prefix
            and path.startswith(output_prefix + "/")
        ):
            raise AnalysisError(f"OUTPUT_ALREADY_PRESENT:{line}")

        raise AnalysisError(f"UNEXPECTED_WORKTREE_CHANGE:{line}")

    runner = root / RUNNER_REL
    require(runner.is_file(), "ANALYZER_FILE_MISSING")

    if execute_mode:
        tracked = subprocess.call(
            ["git", "ls-files", "--error-unmatch", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(tracked == 0, "EXECUTE_REQUIRES_TRACKED_ANALYZER")
        clean = subprocess.call(
            ["git", "diff", "--quiet", "--", RUNNER_REL],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(clean == 0, "ANALYZER_WORKTREE_DRIFT")

    return {"branch": branch, "head": head, "status": status}


def authenticate_file(
    root: Path,
    commit: str,
    rel: str,
    expected_sha: str,
    label: str,
    expected_blob: str | None = None,
) -> bytes:
    raw = git_bytes(root, f"{commit}:{rel}")
    require(sha256_bytes(raw) == expected_sha, f"{label}_SHA256_MISMATCH")
    if expected_blob is not None:
        require(
            git(root, "rev-parse", f"{commit}:{rel}") == expected_blob,
            f"{label}_BLOB_MISMATCH",
        )
    current = root / rel
    require(current.is_file(), f"{label}_WORKTREE_FILE_MISSING")
    require(current.read_bytes() == raw, f"{label}_WORKTREE_DRIFT")
    return raw


def load_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise AnalysisError(f"{label}_JSON_PARSE_FAILURE") from exc
    require(isinstance(obj, dict), f"{label}_NOT_OBJECT")
    return obj


def load_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(raw.splitlines(), 1):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            raise AnalysisError(f"{label}_JSONL_PARSE_FAILURE:{i}") from exc
        require(isinstance(obj, dict), f"{label}_ROW_NOT_OBJECT:{i}")
        rows.append(obj)
    return rows


def authenticate_inputs(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    authenticate_file(
        root,
        AUTHORITY_FREEZE_COMMIT,
        AUTHORITY_REL,
        AUTHORITY_SHA256,
        "AUTHORITY",
        AUTHORITY_BLOB,
    )
    item_raw = authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_ITEM_REL,
        PARENT_ITEM_SHA256,
        "PARENT_ITEM",
    )
    summary_raw = authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_SUMMARY_REL,
        PARENT_SUMMARY_SHA256,
        "PARENT_SUMMARY",
    )
    manifest_raw = authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_MANIFEST_REL,
        PARENT_MANIFEST_SHA256,
        "PARENT_MANIFEST",
    )
    authenticate_file(
        root,
        PARENT_EVIDENCE_FREEZE,
        PARENT_REPORT_REL,
        PARENT_REPORT_SHA256,
        "PARENT_REPORT",
    )
    return (
        load_jsonl(item_raw, "PARENT_ITEM"),
        load_json(summary_raw, "PARENT_SUMMARY"),
        load_json(manifest_raw, "PARENT_MANIFEST"),
    )


def validate_parent_summary(summary: Mapping[str, Any]) -> None:
    require(
        summary.get("schema_version") == PARENT_SUMMARY_SCHEMA,
        "PARENT_SUMMARY_SCHEMA_MISMATCH",
    )
    require(
        summary.get("common_ddsssss_item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_SUMMARY_ITEM_COUNT_MISMATCH",
    )
    require(
        summary.get("strong_kernel_channel_count") == EXPECTED_STRONG_COUNT,
        "PARENT_SUMMARY_STRONG_COUNT_MISMATCH",
    )

    p = summary["paired_diagnostics"]
    bridge = {
        "t": p["reconstructed_delta_I"],
        "p": p["delta_P"],
        "r": p["delta_N"],
        "c": p["delta_same_sign_channel_count"],
        "a": p["delta_Pbar"],
        "k": p["delta_Kbar"],
    }
    for key, agg in bridge.items():
        require(close(agg["mean"], EXPECTED_MEANS[key]), f"PARENT_MEAN_MISMATCH:{key}")
        require(
            close(agg["median"], EXPECTED_MEDIANS[key]),
            f"PARENT_MEDIAN_MISMATCH:{key}",
        )


def validate_parent_manifest(manifest: Mapping[str, Any]) -> None:
    require(
        manifest.get("schema_version") == PARENT_MANIFEST_SCHEMA,
        "PARENT_MANIFEST_SCHEMA_MISMATCH",
    )
    require(
        manifest.get("analysis_git_head") == PARENT_IMPLEMENTATION,
        "PARENT_ANALYSIS_HEAD_MISMATCH",
    )
    require(
        manifest.get("parent_evidence_freeze")
        == "431e8faa6e5c82a20d87f532b4ab960fcf641ec2",
        "PARENT_PARENT_EVIDENCE_MISMATCH",
    )
    require(
        manifest.get("common_ddsssss_item_count") == EXPECTED_ITEM_COUNT,
        "PARENT_MANIFEST_ITEM_COUNT_MISMATCH",
    )
    require(
        manifest.get("strong_kernel_channel_count") == EXPECTED_STRONG_COUNT,
        "PARENT_MANIFEST_STRONG_COUNT_MISMATCH",
    )
    require(manifest.get("model_forward_count") == 0, "PARENT_MODEL_FORWARD_NONZERO")
    for key in (
        "new_model_execution",
        "checkpoint_loaded",
        "handoff_opened",
        "transformers_imported",
        "tokenizer_invoked",
        "logits_read",
        "task_heads_executed",
        "training_executed",
        "causal_intervention_executed",
        "pca_svd_or_learned_geometry_executed",
        "posthoc_subset_search_executed",
        "raw_vectors_read_or_persisted",
    ):
        require(manifest.get(key) is False, f"PARENT_FORBIDDEN_FLAG:{key}")


def sign_label(x: float) -> str:
    if x > 0.0:
        return "positive"
    if x < 0.0:
        return "negative"
    return "zero"


def quadrant(p: float, r: float) -> str:
    if p == 0.0 or r == 0.0:
        return "BOUNDARY"
    if p > 0.0 and r > 0.0:
        return "Q++"
    if p > 0.0 and r < 0.0:
        return "Q+-"
    if p < 0.0 and r > 0.0:
        return "Q-+"
    return "Q--"


def dominance(p: float, r: float) -> str:
    ap, ar = abs(p), abs(r)
    if ap > ar:
        return "POSITIVE_COMPONENT_DOMINANT"
    if ar > ap:
        return "CANCELLATION_RELIEF_DOMINANT"
    return "EXACT_TIE"


def positive_side_pattern(c: float, a: float) -> str:
    if c == 0.0 or a == 0.0:
        return "BOUNDARY"
    return ("C+" if c > 0.0 else "C-") + "/" + ("A+" if a > 0.0 else "A-")


def cancellation_side_pattern(o: float, k: float) -> str:
    if o == 0.0 or k == 0.0:
        return "BOUNDARY"
    return ("O_RELIEF" if o < 0.0 else "O_WORSE") + "/" + (
        "K_RELIEF" if k < 0.0 else "K_WORSE"
    )


def analyze_items(
    parent_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(
        len(parent_rows) == EXPECTED_ITEM_COUNT,
        f"PARENT_ITEM_COUNT_MISMATCH:{len(parent_rows)}",
    )

    stable_ids: set[str] = set()
    local_indices: set[int] = set()
    out: list[dict[str, Any]] = []

    max_pair_res = 0.0
    max_count_res = 0.0
    max_parent_res = 0.0

    for row in parent_rows:
        require(row.get("schema_version") == PARENT_ITEM_SCHEMA, "PARENT_ITEM_SCHEMA_MISMATCH")

        sid = str(row["stable_item_id"])
        idx = int(row["local_template_index"])
        require(sid not in stable_ids, f"DUPLICATE_STABLE_ITEM_ID:{sid}")
        require(idx not in local_indices, f"DUPLICATE_LOCAL_INDEX:{idx}")
        stable_ids.add(sid)
        local_indices.add(idx)

        require(row["source_block"] == EXPECTED_SOURCE_BLOCK, f"SOURCE_BLOCK_MISMATCH:{idx}")
        require(
            row["target_residual_layer"] == EXPECTED_TARGET_RESIDUAL_LAYER,
            f"TARGET_LAYER_MISMATCH:{idx}",
        )
        require(row["parent_map_layer"] == EXPECTED_PARENT_MAP_LAYER, f"PARENT_MAP_MISMATCH:{idx}")
        require(row["relative_coordinate"] == EXPECTED_K, f"K_MISMATCH:{idx}")
        require(row["new_model_execution"] is False, f"PARENT_NEW_MODEL_EXECUTION:{idx}")
        require(row["new_channel_selection"] is False, f"PARENT_NEW_CHANNEL_SELECTION:{idx}")

        p = finite(row["delta_P"], f"P:{idx}")
        r = finite(row["delta_N"], f"R:{idx}")
        t = finite(row["reconstructed_delta_I"], f"T:{idx}")
        c = finite(row["delta_same_sign_channel_count"], f"C:{idx}")
        o = finite(row["delta_opposite_sign_channel_count"], f"O:{idx}")
        a = finite(row["delta_Pbar"], f"A:{idx}")
        k = finite(row["delta_Kbar"], f"K:{idx}")

        pair_res = abs(t - (p + r))
        require(pair_res <= ABS_TOL, f"PAIR_IDENTITY_FAILURE:{idx}:{pair_res}")
        count_res = abs(c + o)
        require(count_res == 0.0, f"COUNT_IDENTITY_FAILURE:{idx}:{count_res}")

        parent_res = finite(row["parent_reproduction_abs_residual"], f"PARENT_RES:{idx}")
        require(parent_res <= ABS_TOL, f"PARENT_REPRO_FAILURE:{idx}:{parent_res}")

        max_pair_res = max(max_pair_res, pair_res)
        max_count_res = max(max_count_res, count_res)
        max_parent_res = max(max_parent_res, parent_res)

        dom = dominance(p, r)
        concordant = (
            t > 0.0
            and p > 0.0
            and abs(p) > abs(r)
        )

        out.append(
            {
                "schema_version": ITEM_SCHEMA,
                "local_template_index": idx,
                "stable_item_id": sid,
                "p_delta_P": p,
                "r_delta_N": r,
                "t_delta_I": t,
                "p_sign": sign_label(p),
                "r_sign": sign_label(r),
                "t_sign": sign_label(t),
                "component_sign_quadrant": quadrant(p, r),
                "component_dominance": dom,
                "parent_concordant_positive_co_contribution_led": concordant,
                "delta_same_sign_channel_count": c,
                "delta_opposite_sign_channel_count": o,
                "delta_Pbar": a,
                "delta_Kbar": k,
                "same_sign_count_direction": sign_label(c),
                "positive_strength_direction": sign_label(a),
                "cancellation_strength_direction": (
                    "relief" if k < 0.0 else ("worse" if k > 0.0 else "zero")
                ),
                "positive_side_count_strength_pattern": positive_side_pattern(c, a),
                "cancellation_side_count_strength_pattern": cancellation_side_pattern(o, k),
                "pair_identity_abs_residual": pair_res,
                "count_identity_abs_residual": count_res,
                "parent_reproduction_abs_residual": parent_res,
                "new_model_execution": False,
                "new_item_selection": False,
                "new_channel_selection": False,
            }
        )

    out.sort(key=lambda x: x["local_template_index"])

    return out, {
        "max_pair_identity_abs_residual": max_pair_res,
        "max_count_identity_abs_residual": max_count_res,
        "max_parent_reproduction_abs_residual": max_parent_res,
    }


def aggregate(values: Sequence[float]) -> dict[str, float | int]:
    vals = [float(x) for x in values]
    require(vals, "EMPTY_AGGREGATE")
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def sign_counts(values: Sequence[float]) -> dict[str, int | float]:
    pos = sum(1 for x in values if x > 0.0)
    neg = sum(1 for x in values if x < 0.0)
    zero = len(values) - pos - neg
    return {
        "positive": pos,
        "negative": neg,
        "zero": zero,
        "positive_fraction": pos / len(values),
    }


def mass_balance(values: Sequence[float]) -> dict[str, float]:
    pos = float(sum(max(float(x), 0.0) for x in values))
    neg = float(sum(min(float(x), 0.0) for x in values))
    require(pos > 0.0, "MASS_BALANCE_POSITIVE_SUM_NONPOSITIVE")
    return {
        "positive_mass": pos,
        "negative_mass": neg,
        "net_mass": pos + neg,
        "negative_to_positive_abs_ratio": abs(neg) / pos,
    }


def build_summary(
    item_rows: Sequence[Mapping[str, Any]],
    parent_summary: Mapping[str, Any],
    maxima: Mapping[str, float],
) -> dict[str, Any]:
    p = [float(r["p_delta_P"]) for r in item_rows]
    rr = [float(r["r_delta_N"]) for r in item_rows]
    t = [float(r["t_delta_I"]) for r in item_rows]
    c = [float(r["delta_same_sign_channel_count"]) for r in item_rows]
    o = [float(r["delta_opposite_sign_channel_count"]) for r in item_rows]
    a = [float(r["delta_Pbar"]) for r in item_rows]
    k = [float(r["delta_Kbar"]) for r in item_rows]

    aggregates = {
        "t": aggregate(t),
        "p": aggregate(p),
        "r": aggregate(rr),
        "c": aggregate(c),
        "o": aggregate(o),
        "a": aggregate(a),
        "k": aggregate(k),
    }

    for key in ("t", "p", "r", "c", "a", "k"):
        require(close(aggregates[key]["mean"], EXPECTED_MEANS[key]), f"ROW_MEAN_BRIDGE_FAILURE:{key}")
        require(
            close(aggregates[key]["median"], EXPECTED_MEDIANS[key]),
            f"ROW_MEDIAN_BRIDGE_FAILURE:{key}",
        )

    parent_pd = parent_summary["paired_diagnostics"]
    parent_map = {
        "t": parent_pd["reconstructed_delta_I"],
        "p": parent_pd["delta_P"],
        "r": parent_pd["delta_N"],
        "c": parent_pd["delta_same_sign_channel_count"],
        "a": parent_pd["delta_Pbar"],
        "k": parent_pd["delta_Kbar"],
    }
    for key in parent_map:
        require(close(aggregates[key]["mean"], parent_map[key]["mean"]), f"PARENT_SUMMARY_MEAN_BRIDGE:{key}")
        require(
            close(aggregates[key]["median"], parent_map[key]["median"]),
            f"PARENT_SUMMARY_MEDIAN_BRIDGE:{key}",
        )

    t_sign = sign_counts(t)
    p_sign = sign_counts(p)
    r_sign = sign_counts(rr)
    c_sign = sign_counts(c)
    a_sign = sign_counts(a)

    k_relief = sum(1 for x in k if x < 0.0)
    k_worse = sum(1 for x in k if x > 0.0)
    k_zero = len(k) - k_relief - k_worse

    quadrant_counts = Counter(str(r["component_sign_quadrant"]) for r in item_rows)
    dom_counts = Counter(str(r["component_dominance"]) for r in item_rows)
    positive_pattern_counts = Counter(
        str(r["positive_side_count_strength_pattern"]) for r in item_rows
    )
    cancellation_pattern_counts = Counter(
        str(r["cancellation_side_count_strength_pattern"]) for r in item_rows
    )

    concordant = sum(
        1 for r in item_rows
        if bool(r["parent_concordant_positive_co_contribution_led"])
    )

    n_total_pos = int(t_sign["positive"])
    axis_a = (
        "A1_MAJORITY_BROAD_POSITIVE_TOTAL"
        if n_total_pos > MAJORITY_BOUNDARY
        else "A2_MINORITY_DRIVEN_POSITIVE_TOTAL"
    )
    axis_b = (
        "B1_MAJORITY_PARENT_CONCORDANT"
        if concordant > MAJORITY_BOUNDARY
        else "B2_NOT_MAJORITY_PARENT_CONCORDANT"
    )

    if axis_a.startswith("A1") and axis_b.startswith("B1"):
        combined = "A1/B1_BROAD_POSITIVE_TOTAL_WITH_MAJORITY_PARENT_CONCORDANCE"
    elif axis_a.startswith("A1"):
        combined = "A1/B2_BROAD_POSITIVE_TOTAL_WITH_HETEROGENEOUS_COMPONENT_CONCORDANCE"
    elif axis_b.startswith("B1"):
        combined = "A2/B1_MINORITY_SIGN_BREADTH_WITH_MAJORITY_PARENT_CONCORDANCE"
    else:
        combined = "A2/B2_MINORITY_DRIVEN_OR_HETEROGENEOUS_ITEM_STRUCTURE"

    return {
        "schema_version": SUMMARY_SCHEMA,
        "scientific_question": QUESTION,
        "analysis_type": "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",
        "common_ddsssss_item_count": EXPECTED_ITEM_COUNT,
        "strong_kernel_channel_count": EXPECTED_STRONG_COUNT,
        "majority_boundary": MAJORITY_BOUNDARY,
        "source_block": EXPECTED_SOURCE_BLOCK,
        "target_residual_layer": EXPECTED_TARGET_RESIDUAL_LAYER,
        "parent_map_layer": EXPECTED_PARENT_MAP_LAYER,
        "relative_coordinate": EXPECTED_K,

        "primary_two_axis_classification": {
            "axis_A_total_effect_breadth": axis_a,
            "axis_B_parent_component_concordance": axis_b,
            "combined_classification": combined,
            "N_total_pos": n_total_pos,
            "F_total_pos": n_total_pos / EXPECTED_ITEM_COUNT,
            "N_parent_concordant": concordant,
            "F_parent_concordant": concordant / EXPECTED_ITEM_COUNT,
        },

        "component_sign_breadth": {
            "total_effect": t_sign,
            "positive_component": p_sign,
            "cancellation_relief_component": r_sign,
            "both_components_reinforce_count": int(quadrant_counts.get("Q++", 0)),
            "both_components_reinforce_fraction":
                int(quadrant_counts.get("Q++", 0)) / EXPECTED_ITEM_COUNT,
        },

        "component_sign_quadrants": {
            key: int(quadrant_counts.get(key, 0))
            for key in ("Q++", "Q+-", "Q-+", "Q--", "BOUNDARY")
        },

        "component_dominance": {
            "positive_component_dominant":
                int(dom_counts.get("POSITIVE_COMPONENT_DOMINANT", 0)),
            "cancellation_relief_dominant":
                int(dom_counts.get("CANCELLATION_RELIEF_DOMINANT", 0)),
            "exact_tie":
                int(dom_counts.get("EXACT_TIE", 0)),
            "positive_component_dominant_fraction":
                int(dom_counts.get("POSITIVE_COMPONENT_DOMINANT", 0))
                / EXPECTED_ITEM_COUNT,
        },

        "count_strength_breadth": {
            "same_sign_count": c_sign,
            "positive_mass_per_same_sign_channel": a_sign,
            "cancellation_magnitude_per_opposite_sign_channel": {
                "relief_negative_delta_Kbar": k_relief,
                "worse_positive_delta_Kbar": k_worse,
                "zero": k_zero,
                "relief_fraction": k_relief / EXPECTED_ITEM_COUNT,
            },
            "positive_side_joint_patterns": dict(sorted(positive_pattern_counts.items())),
            "cancellation_side_joint_patterns":
                dict(sorted(cancellation_pattern_counts.items())),
        },

        "cross_item_mass_balance": {
            "total_effect_t": mass_balance(t),
            "positive_component_p": mass_balance(p),
            "cancellation_relief_r": mass_balance(rr),
        },

        "population_bridges": aggregates,

        "max_pair_identity_abs_residual":
            maxima["max_pair_identity_abs_residual"],
        "max_count_identity_abs_residual":
            maxima["max_count_identity_abs_residual"],
        "max_parent_reproduction_abs_residual":
            maxima["max_parent_reproduction_abs_residual"],

        "parent_mean_bridge_match": True,
        "parent_median_bridge_match": True,
        "parent_item_identity_match": True,

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
        "new_item_selection": False,
        "new_channel_selection": False,
        "raw_vectors_read_or_persisted": False,
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
    return b"".join(json_bytes(r) for r in rows)


def print_plan(repo: Mapping[str, Any]) -> None:
    print("=== FROZEN-STRONG SIGN-CONTRIBUTION ITEMWISE BREADTH PLAN ===")
    print("branch =", repo["branch"])
    print("head =", repo["head"])
    print("authority_freeze_commit =", AUTHORITY_FREEZE_COMMIT)
    print("authority_sha256 =", AUTHORITY_SHA256)
    print("authority_blob =", AUTHORITY_BLOB)
    print("parent_evidence_freeze =", PARENT_EVIDENCE_FREEZE)
    print("parent_item_sha256 =", PARENT_ITEM_SHA256)
    print("parent_summary_sha256 =", PARENT_SUMMARY_SHA256)
    print("parent_manifest_sha256 =", PARENT_MANIFEST_SHA256)
    print("parent_report_sha256 =", PARENT_REPORT_SHA256)
    print("common_ddsssss_item_count =", EXPECTED_ITEM_COUNT)
    print("strong_kernel_channel_count =", EXPECTED_STRONG_COUNT)
    print("majority_boundary =", MAJORITY_BOUNDARY)
    print("scientific_question =", QUESTION)
    print("primary_identity = t_i = p_i + r_i")
    print("count_identity = c_i + o_i = 0")
    print("model_forward_count = 0")
    print("new_model_execution = False")
    print("checkpoint_loaded = False")
    print("handoff_opened = False")
    print("transformers_imported = False")
    print("tokenizer_invoked = False")
    print("training_executed = False")
    print("causal_intervention_executed = False")
    print("posthoc_subset_search_executed = False")


def execute(
    root: Path,
    repo: Mapping[str, Any],
    item_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    final_dir = output_dir.resolve()
    partial_dir = Path(str(final_dir) + ".partial")
    require(not final_dir.exists(), f"OUTPUT_DIR_EXISTS:{final_dir}")
    require(not partial_dir.exists(), f"PARTIAL_OUTPUT_EXISTS:{partial_dir}")

    partial_dir.mkdir(parents=True, exist_ok=False)

    item_path = (
        partial_dir
        / "frozen_strong_sign_contribution_itemwise_breadth_classification.jsonl"
    )
    summary_path = partial_dir / "summary.json"
    manifest_path = partial_dir / "static_analysis_manifest.json"

    item_path.write_bytes(jsonl_bytes(item_rows))
    summary_path.write_bytes(json_bytes(summary))

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "analysis_git_head": repo["head"],
        "analysis_branch": repo["branch"],
        "analyzer_rel": RUNNER_REL,
        "analyzer_sha256": sha256_file(root / RUNNER_REL),
        "authority_freeze_commit": AUTHORITY_FREEZE_COMMIT,
        "authority_sha256": AUTHORITY_SHA256,
        "authority_blob": AUTHORITY_BLOB,
        "parent_evidence_freeze": PARENT_EVIDENCE_FREEZE,
        "parent_implementation": PARENT_IMPLEMENTATION,
        "parent_item_sha256": PARENT_ITEM_SHA256,
        "parent_summary_sha256": PARENT_SUMMARY_SHA256,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_report_sha256": PARENT_REPORT_SHA256,
        "scientific_question": QUESTION,
        "analysis_type": "MODEL_FREE_FROZEN_ARTIFACT_STATIC_ANALYSIS",
        "common_ddsssss_item_count": EXPECTED_ITEM_COUNT,
        "strong_kernel_channel_count": EXPECTED_STRONG_COUNT,
        "majority_boundary": MAJORITY_BOUNDARY,
        "source_block": EXPECTED_SOURCE_BLOCK,
        "target_residual_layer": EXPECTED_TARGET_RESIDUAL_LAYER,
        "parent_map_layer": EXPECTED_PARENT_MAP_LAYER,
        "relative_coordinate": EXPECTED_K,
        "relative_tolerance": REL_TOL,
        "absolute_tolerance": ABS_TOL,
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
        "new_item_selection": False,
        "new_channel_selection": False,
        "raw_vectors_read_or_persisted": False,
        "parent_mean_bridge_match": True,
        "parent_median_bridge_match": True,
        "parent_item_identity_match": True,
        "outputs": {
            item_path.name: sha256_file(item_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    os.replace(partial_dir, final_dir)

    primary = summary["primary_two_axis_classification"]
    breadth = summary["component_sign_breadth"]
    dominance_summary = summary["component_dominance"]
    cs = summary["count_strength_breadth"]

    print("PASS_FROZEN_STRONG_SIGN_CONTRIBUTION_ITEMWISE_BREADTH_STATIC_ANALYSIS")
    print("output_dir =", final_dir)
    print("model_forward_count = 0")
    print("item_count =", EXPECTED_ITEM_COUNT)
    print("N_total_pos =", primary["N_total_pos"])
    print("F_total_pos =", primary["F_total_pos"])
    print("N_parent_concordant =", primary["N_parent_concordant"])
    print("F_parent_concordant =", primary["F_parent_concordant"])
    print("axis_A =", primary["axis_A_total_effect_breadth"])
    print("axis_B =", primary["axis_B_parent_component_concordance"])
    print("combined_classification =", primary["combined_classification"])
    print("positive_component_sign_breadth =", breadth["positive_component"])
    print("cancellation_relief_sign_breadth =", breadth["cancellation_relief_component"])
    print("both_components_reinforce_count =", breadth["both_components_reinforce_count"])
    print("component_sign_quadrants =", summary["component_sign_quadrants"])
    print("component_dominance =", dominance_summary)
    print("same_sign_count_breadth =", cs["same_sign_count"])
    print(
        "positive_strength_breadth =",
        cs["positive_mass_per_same_sign_channel"],
    )
    print(
        "cancellation_strength_breadth =",
        cs["cancellation_magnitude_per_opposite_sign_channel"],
    )
    print(
        "positive_side_joint_patterns =",
        cs["positive_side_joint_patterns"],
    )
    print(
        "cancellation_side_joint_patterns =",
        cs["cancellation_side_joint_patterns"],
    )
    print(
        "cross_item_mass_balance =",
        summary["cross_item_mass_balance"],
    )
    print(
        "max_pair_identity_abs_residual =",
        summary["max_pair_identity_abs_residual"],
    )
    print(
        "max_count_identity_abs_residual =",
        summary["max_count_identity_abs_residual"],
    )
    print(
        "max_parent_reproduction_abs_residual =",
        summary["max_parent_reproduction_abs_residual"],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--output-dir", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    require(
        not args.static_preflight or args.output_dir is None,
        "STATIC_PREFLIGHT_OUTPUT_DIR_FORBIDDEN",
    )
    require(
        not args.execute or args.output_dir is not None,
        "EXECUTE_OUTPUT_DIR_REQUIRED",
    )

    repo = authenticate_repo(
        root,
        execute_mode=bool(args.execute),
        output_dir=args.output_dir,
    )
    parent_rows, parent_summary, parent_manifest = authenticate_inputs(root)
    validate_parent_summary(parent_summary)
    validate_parent_manifest(parent_manifest)

    item_rows, maxima = analyze_items(parent_rows)
    summary = build_summary(item_rows, parent_summary, maxima)

    print_plan(repo)

    if args.static_preflight:
        print("scientific_evidence_emitted = False")
        print("PASS_FROZEN_STRONG_SIGN_CONTRIBUTION_ITEMWISE_BREADTH_STATIC_PREFLIGHT")
        return 0

    execute(root, repo, item_rows, summary, args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        raise SystemExit(2)
