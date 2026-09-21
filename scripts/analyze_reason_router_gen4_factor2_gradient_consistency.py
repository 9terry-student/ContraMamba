#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
EPSILONS = (0.03125, 0.015625, 0.0078125)
SUBSET_SIZE = 32
ROWS_PER_SCALE = 64

ROW_FILE = "gradient_consistency_rows.jsonl"
RAW_SUMMARY_FILE = "raw_gradient_consistency_summary.json"
RAW_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

ROW_DERIVATIVE_FILE = "row_level_derivatives.jsonl"
PAIR_DERIVATIVE_FILE = "pair_level_derivatives.jsonl"
ANALYSIS_FILE = "gradient_consistency_analysis.json"
MANIFEST_FILE = "artifact_manifest.json"

RESULT = "PASS_GEN4_FACTOR2_GRADIENT_CONSISTENCY_STATIC_ANALYSIS"
SCHEMA_VERSION = "gen4-factor2-gradient-consistency-static-analysis-v1"


class GradientConsistencyAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GradientConsistencyAnalysisError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GradientConsistencyAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(
    *,
    expected_head: str,
    raw_freeze_commit: str,
) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    for ancestor, label in (
        (raw_freeze_commit, "RAW_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise GradientConsistencyAnalysisError(
            f"INPUT_OUTSIDE_REPO:{path}"
        ) from exc


def git_blob_bytes(path: Path) -> bytes:
    rel = repo_relative(path)
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GradientConsistencyAnalysisError(
            f"GIT_BLOB_FAILURE:{rel}"
        ) from exc


def git_blob_text(path: Path) -> str:
    return git_blob_bytes(path).decode("utf-8")


def read_jsonl_blob(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(git_blob_text(path).splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


def validate_raw_bundle(raw_dir: Path) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    sums_text = git_blob_text(raw_dir / SUMS_FILE)
    sums: dict[str, str] = {}
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in sums, f"SUM_DUPLICATE:{name}")
        require(
            name in {
                ROW_FILE,
                RAW_SUMMARY_FILE,
                RAW_MANIFEST_FILE,
            },
            f"SUM_UNEXPECTED:{name}",
        )
        raw = git_blob_bytes(raw_dir / name)
        require(sha256_bytes(raw) == digest, f"SUM_DIGEST:{name}")
        sums[name] = digest
    require(
        set(sums)
        == {
            ROW_FILE,
            RAW_SUMMARY_FILE,
            RAW_MANIFEST_FILE,
        },
        "SUM_SET",
    )

    summary = json.loads(
        git_blob_text(raw_dir / RAW_SUMMARY_FILE)
    )
    manifest = json.loads(
        git_blob_text(raw_dir / RAW_MANIFEST_FILE)
    )
    require(
        summary["result"]
        == "PASS_GEN4_FACTOR2_GRADIENT_CONSISTENCY_RAW",
        "RAW_RESULT",
    )
    require(
        manifest["result"] == summary["result"],
        "RAW_MANIFEST_RESULT",
    )
    require(int(summary["subset_pair_count"]) == SUBSET_SIZE, "RAW_SUBSET")
    require(int(summary["row_count"]) == 128, "RAW_ROWS")
    require(summary["cells"] == list(CELLS), "RAW_CELLS")
    require(summary["scales"] == list(SCALES), "RAW_SCALES")
    require(summary["epsilons"] == list(EPSILONS), "RAW_EPS")
    require(summary["signed_perturbations"] is True, "RAW_SIGNED")
    require(
        summary["primary_numeric_target"]
        == "fixed-native-active-wrong central difference vs frozen Delta_L",
        "RAW_PRIMARY",
    )
    require(
        summary["dynamic_max_wrong_margin_secondary_only"] is True,
        "RAW_DYNAMIC_SECONDARY",
    )
    require(summary["even_order_curvature_recorded"] is True, "RAW_EVEN")
    require(int(summary["p_value_count_executed"]) == 0, "RAW_P")
    require(summary["row_filtering_performed"] is False, "RAW_FILTER")
    require(summary["rescue_performed"] is False, "RAW_RESCUE")
    require(
        summary["adaptive_epsilon_selection_performed"] is False,
        "RAW_ADAPTIVE",
    )
    require(
        summary["behavioral_calibration_result_accessed"] is False,
        "RAW_BEHAVIOR_ACCESS",
    )
    require(
        summary["static_factor2_result_accessed"] is False,
        "RAW_STATIC_ACCESS",
    )
    require(
        manifest["row_sha256"] == sums[ROW_FILE],
        "RAW_ROW_SHA",
    )
    require(
        manifest["summary_sha256"] == sums[RAW_SUMMARY_FILE],
        "RAW_SUMMARY_SHA",
    )

    rows = read_jsonl_blob(raw_dir / ROW_FILE)
    require(len(rows) == 128, "RAW_ROW_COUNT")
    return rows, summary, manifest


def sign_code(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def descriptive(
    values: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(
        a.ndim == 1
        and a.size > 0
        and np.isfinite(a).all(),
        "DESCRIPTIVE",
    )
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "mean_abs": float(np.mean(np.abs(a))),
        "sd_population": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
        "max_abs": float(np.max(np.abs(a))),
    }


def rank_average(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    require(
        a.ndim == 1
        and a.size > 0
        and np.isfinite(a).all(),
        "RANK_INPUT",
    )
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def pearson_coefficient(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(
        a.shape == b.shape
        and a.ndim == 1
        and a.size >= 2,
        "PEARSON_SHAPE",
    )
    require(
        np.isfinite(a).all() and np.isfinite(b).all(),
        "PEARSON_FINITE",
    )
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    denom = float(
        np.sqrt(np.dot(ac, ac) * np.dot(bc, bc))
    )
    require(denom > 0.0, "PEARSON_DENOM")
    value = float(np.dot(ac, bc) / denom)
    require(math.isfinite(value), "PEARSON_RESULT")
    return value


def spearman_coefficient(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    return pearson_coefficient(
        rank_average(x),
        rank_average(y),
    )


def origin_slope(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(
        a.shape == b.shape
        and a.ndim == 1
        and a.size > 0,
        "SLOPE_SHAPE",
    )
    require(
        np.isfinite(a).all() and np.isfinite(b).all(),
        "SLOPE_FINITE",
    )
    denom = float(np.dot(a, a))
    require(denom > 0.0, "SLOPE_DENOM")
    value = float(np.dot(a, b) / denom)
    require(math.isfinite(value), "SLOPE_RESULT")
    return value


def rmse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    require(
        a.shape == p.shape and a.ndim == 1,
        "RMSE_SHAPE",
    )
    return float(np.sqrt(np.mean((a - p) ** 2)))


def fixed_margin(
    logits: Sequence[float],
    *,
    correct_label_id: int,
    wrong_class_id: int,
) -> float:
    require(len(logits) == 3, "FIXED_MARGIN_LOGITS")
    value = (
        float(logits[correct_label_id])
        - float(logits[wrong_class_id])
    )
    require(math.isfinite(value), "FIXED_MARGIN_FINITE")
    return value


def dynamic_margin(
    logits: Sequence[float],
    *,
    correct_label_id: int,
) -> tuple[float, int]:
    wrong_ids = [
        index
        for index in range(3)
        if index != correct_label_id
    ]
    require(
        float(logits[wrong_ids[0]])
        != float(logits[wrong_ids[1]]),
        "DYNAMIC_TIE",
    )
    wrong = max(
        wrong_ids,
        key=lambda index: float(logits[index]),
    )
    return (
        fixed_margin(
            logits,
            correct_label_id=correct_label_id,
            wrong_class_id=wrong,
        ),
        int(wrong),
    )


def derive_one_epsilon(
    row: Mapping[str, Any],
    *,
    epsilon: float,
) -> dict[str, Any]:
    frozen_delta = float(row["frozen_Delta_L_row"])
    correct = int(row["correct_label_id"])
    wrong = int(row["frozen_active_wrong_class_id"])
    native_logits = row["native"]["final_logits"]
    entry = row["epsilon_results"][str(epsilon)]
    plus_logits = entry["plus"]["final_logits"]
    minus_logits = entry["minus"]["final_logits"]

    native_fixed = fixed_margin(
        native_logits,
        correct_label_id=correct,
        wrong_class_id=wrong,
    )
    plus_fixed = fixed_margin(
        plus_logits,
        correct_label_id=correct,
        wrong_class_id=wrong,
    )
    minus_fixed = fixed_margin(
        minus_logits,
        correct_label_id=correct,
        wrong_class_id=wrong,
    )

    native_dynamic, native_dynamic_wrong = dynamic_margin(
        native_logits,
        correct_label_id=correct,
    )
    plus_dynamic, plus_dynamic_wrong = dynamic_margin(
        plus_logits,
        correct_label_id=correct,
    )
    minus_dynamic, minus_dynamic_wrong = dynamic_margin(
        minus_logits,
        correct_label_id=correct,
    )
    require(
        native_dynamic_wrong == wrong,
        "NATIVE_WRONG_REPLAY",
    )

    fixed_cd = (
        minus_fixed - plus_fixed
    ) / (2.0 * epsilon)
    dynamic_cd = (
        minus_dynamic - plus_dynamic
    ) / (2.0 * epsilon)

    d_plus_fixed = native_fixed - plus_fixed
    d_minus_fixed = native_fixed - minus_fixed
    positive_estimate = d_plus_fixed / epsilon
    negative_estimate = -d_minus_fixed / epsilon

    even_second = (
        plus_fixed + minus_fixed - 2.0 * native_fixed
    ) / (epsilon * epsilon)

    return {
        "epsilon": epsilon,
        "frozen_Delta_L": frozen_delta,
        "fixed_central_difference": fixed_cd,
        "dynamic_central_difference": dynamic_cd,
        "fixed_positive_one_sided_estimate": positive_estimate,
        "fixed_negative_one_sided_estimate": negative_estimate,
        "fixed_even_second_difference": even_second,
        "plus_dynamic_wrong_class_id": plus_dynamic_wrong,
        "minus_dynamic_wrong_class_id": minus_dynamic_wrong,
        "plus_wrong_switch": plus_dynamic_wrong != wrong,
        "minus_wrong_switch": minus_dynamic_wrong != wrong,
    }


def vector_summary(
    delta: Sequence[float],
    estimate: Sequence[float],
) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    e = np.asarray(estimate, dtype=np.float64)
    require(d.shape == e.shape, "VECTOR_SHAPE")
    sign = np.asarray(
        [
            sign_code(float(x)) == sign_code(float(y))
            for x, y in zip(d, e)
        ],
        dtype=np.float64,
    )
    slope = origin_slope(d, e)
    return {
        "origin_slope_estimate_on_frozen_Delta_L": slope,
        "pearson": pearson_coefficient(d, e),
        "spearman_coefficient_only":
            spearman_coefficient(d, e),
        "sign_agreement": {
            "count": int(np.sum(sign)),
            "total": int(sign.size),
            "fraction": float(np.mean(sign)),
        },
        "rmse_vs_1x_frozen_Delta_L": rmse(e, d),
        "rmse_vs_2x_frozen_Delta_L": rmse(e, 2.0 * d),
        "distance_from_1x_slope": abs(slope - 1.0),
        "distance_from_2x_slope": abs(slope - 2.0),
        "estimate": descriptive(e),
        "frozen_Delta_L": descriptive(d),
        "p_value_count": 0,
        "inference_performed": False,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    subset_pairs: Sequence[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    require(len(rows) == 128, "ANALYSIS_ROW_COUNT")
    expected_keys = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in subset_pairs
        for cell in CELLS
    }
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    derived_rows: list[dict[str, Any]] = []

    for row in rows:
        key = (
            str(row["scale"]),
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in by_key, f"RAW_DUP:{key}")
        by_key[key] = row

    require(set(by_key) == expected_keys, "RAW_KEY_COVERAGE")

    scale_results: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []

    derived_by_key_eps: dict[
        tuple[str, str, str, float],
        dict[str, Any],
    ] = {}

    for scale in SCALES:
        scale_eps: dict[str, Any] = {}
        for epsilon in EPSILONS:
            delta: list[float] = []
            fixed_cd: list[float] = []
            dynamic_cd: list[float] = []
            positive: list[float] = []
            negative: list[float] = []
            even_second: list[float] = []
            plus_switch = 0
            minus_switch = 0

            for pair in subset_pairs:
                for cell in CELLS:
                    raw = by_key[(scale, pair, cell)]
                    derived = derive_one_epsilon(
                        raw,
                        epsilon=epsilon,
                    )
                    derived_by_key_eps[
                        (scale, pair, cell, epsilon)
                    ] = derived
                    delta.append(
                        float(derived["frozen_Delta_L"])
                    )
                    fixed_cd.append(
                        float(
                            derived[
                                "fixed_central_difference"
                            ]
                        )
                    )
                    dynamic_cd.append(
                        float(
                            derived[
                                "dynamic_central_difference"
                            ]
                        )
                    )
                    positive.append(
                        float(
                            derived[
                                "fixed_positive_one_sided_estimate"
                            ]
                        )
                    )
                    negative.append(
                        float(
                            derived[
                                "fixed_negative_one_sided_estimate"
                            ]
                        )
                    )
                    even_second.append(
                        float(
                            derived[
                                "fixed_even_second_difference"
                            ]
                        )
                    )
                    plus_switch += int(
                        bool(derived["plus_wrong_switch"])
                    )
                    minus_switch += int(
                        bool(derived["minus_wrong_switch"])
                    )

                    derived_rows.append({
                        "scale": scale,
                        "source_pair_id": pair,
                        "contrast_cell_id": cell,
                        **derived,
                    })

            scale_eps[str(epsilon)] = {
                "epsilon": epsilon,
                "primary_fixed_active_wrong_central":
                    vector_summary(delta, fixed_cd),
                "secondary_dynamic_max_wrong_central":
                    vector_summary(delta, dynamic_cd),
                "positive_one_sided_fixed":
                    vector_summary(delta, positive),
                "negative_one_sided_fixed":
                    vector_summary(delta, negative),
                "fixed_even_second_difference":
                    descriptive(even_second),
                "dynamic_wrong_switch_count_plus":
                    plus_switch,
                "dynamic_wrong_switch_count_minus":
                    minus_switch,
                "dynamic_wrong_switch_fraction_plus":
                    plus_switch / ROWS_PER_SCALE,
                "dynamic_wrong_switch_fraction_minus":
                    minus_switch / ROWS_PER_SCALE,
                "p_value_count": 0,
                "inference_performed": False,
            }
        scale_results[scale] = scale_eps

    pair_scale_vectors: dict[
        str,
        dict[float, dict[str, list[float]]],
    ] = {
        scale: {
            epsilon: {"delta": [], "cd": []}
            for epsilon in EPSILONS
        }
        for scale in SCALES
    }

    for pair in subset_pairs:
        record: dict[str, Any] = {
            "source_pair_id": pair,
        }
        for scale in SCALES:
            for epsilon in EPSILONS:
                delta_cells = [
                    float(
                        derived_by_key_eps[
                            (scale, pair, cell, epsilon)
                        ]["frozen_Delta_L"]
                    )
                    for cell in CELLS
                ]
                cd_cells = [
                    float(
                        derived_by_key_eps[
                            (scale, pair, cell, epsilon)
                        ][
                            "fixed_central_difference"
                        ]
                    )
                    for cell in CELLS
                ]
                pair_delta = float(
                    np.mean(
                        np.asarray(
                            delta_cells,
                            dtype=np.float64,
                        )
                    )
                )
                pair_cd = float(
                    np.mean(
                        np.asarray(
                            cd_cells,
                            dtype=np.float64,
                        )
                    )
                )
                tag = str(epsilon).replace(".", "_")
                record[
                    f"{scale}_frozen_Delta_L_alpha_{tag}"
                ] = pair_delta
                record[
                    f"{scale}_fixed_central_difference_alpha_{tag}"
                ] = pair_cd
                pair_scale_vectors[scale][epsilon][
                    "delta"
                ].append(pair_delta)
                pair_scale_vectors[scale][epsilon][
                    "cd"
                ].append(pair_cd)
        pair_rows.append(record)

    pair_level: dict[str, Any] = {}
    for scale in SCALES:
        pair_level[scale] = {}
        for epsilon in EPSILONS:
            vectors = pair_scale_vectors[scale][epsilon]
            pair_level[scale][str(epsilon)] = (
                vector_summary(
                    vectors["delta"],
                    vectors["cd"],
                )
            )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "subset_pair_count": SUBSET_SIZE,
        "subset_pairs": list(subset_pairs),
        "row_count": 128,
        "cells": list(CELLS),
        "scales": list(SCALES),
        "epsilons": list(EPSILONS),
        "primary_diagnostic": (
            "row-level fixed-native-active-wrong central-difference "
            "directional derivative vs frozen autograd Delta_L"
        ),
        "central_difference_definition": (
            "(M(h-epsilon*delta)-M(h+epsilon*delta))/(2*epsilon)"
        ),
        "margin_definition": (
            "correct logit minus native active wrong-class logit, "
            "with wrong-class identity frozen from native autograd readout"
        ),
        "one_sided_definitions": {
            "positive":
                "(M(h)-M(h+epsilon*delta))/epsilon",
            "negative":
                "-(M(h)-M(h-epsilon*delta))/epsilon",
        },
        "even_second_difference_definition": (
            "(M(h+epsilon*delta)+M(h-epsilon*delta)-2*M(h))/epsilon^2"
        ),
        "row_level_by_scale": scale_results,
        "pair_level_fixed_central_by_scale": pair_level,
        "dynamic_max_wrong_margin_secondary_only": True,
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "adaptive_epsilon_selection_performed": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "training_executed": False,
        "scientific_conclusion": None,
    }
    return analysis, derived_rows, pair_rows


def write_outputs(
    *,
    expected_head: str,
    raw_freeze_commit: str,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(
        expected_head=expected_head,
        raw_freeze_commit=raw_freeze_commit,
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    rows, summary, manifest = validate_raw_bundle(raw_dir)
    require(
        str(summary["execution_head"])
        == str(manifest["execution_head"]),
        "RAW_EXECUTION_HEAD",
    )

    subset_pairs = [
        str(value)
        for value in summary["subset_pairs"]
    ]
    require(len(subset_pairs) == SUBSET_SIZE, "SUBSET_PAIRS")

    analysis, derived_rows, pair_rows = analyze_rows(
        rows,
        subset_pairs=subset_pairs,
    )

    analysis["analysis_head"] = expected_head
    analysis["raw_freeze_commit"] = raw_freeze_commit
    analysis["raw_execution_head"] = summary["execution_head"]

    output_dir.mkdir(parents=True, exist_ok=False)

    row_raw = b"".join(
        canonical_json_bytes(row)
        for row in derived_rows
    )
    pair_raw = b"".join(
        canonical_json_bytes(row)
        for row in pair_rows
    )
    (output_dir / ROW_DERIVATIVE_FILE).write_bytes(
        row_raw
    )
    (output_dir / PAIR_DERIVATIVE_FILE).write_bytes(
        pair_raw
    )
    (output_dir / ANALYSIS_FILE).write_bytes(
        pretty_json_bytes(analysis)
    )

    out_manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "analysis_head": expected_head,
        "raw_freeze_commit": raw_freeze_commit,
        "raw_execution_head": summary["execution_head"],
        "frozen_input_read_mode": "git_blob_at_HEAD",
        "raw_input_sha256": {
            "rows": manifest["row_sha256"],
            "summary": manifest["summary_sha256"],
        },
        "row_level_derivatives_sha256":
            sha256_file(
                output_dir / ROW_DERIVATIVE_FILE
            ),
        "pair_level_derivatives_sha256":
            sha256_file(
                output_dir / PAIR_DERIVATIVE_FILE
            ),
        "analysis_sha256":
            sha256_file(output_dir / ANALYSIS_FILE),
        "subset_pair_count": SUBSET_SIZE,
        "row_count": 128,
        "epsilons": list(EPSILONS),
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "adaptive_epsilon_selection_performed": False,
        "model_forward_count": 0,
        "backward_count": 0,
        "training_executed": False,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(
        pretty_json_bytes(out_manifest)
    )

    names = (
        ROW_DERIVATIVE_FILE,
        PAIR_DERIVATIVE_FILE,
        ANALYSIS_FILE,
        MANIFEST_FILE,
    )
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static analysis of the bounded symmetric "
            "factor-2 gradient-consistency diagnostic. Compares the "
            "fixed-native-active-wrong central-difference derivative "
            "against frozen autograd Delta_L with zero p-values."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--raw-freeze-commit", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    result = write_outputs(
        expected_head=args.expected_head,
        raw_freeze_commit=args.raw_freeze_commit,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    print("RESULT=" + result["result"])
    for scale in SCALES:
        label = scale.upper()
        for epsilon in EPSILONS:
            metrics = result[
                "row_level_by_scale"
            ][scale][str(epsilon)]
            primary = metrics[
                "primary_fixed_active_wrong_central"
            ]
            pair = result[
                "pair_level_fixed_central_by_scale"
            ][scale][str(epsilon)]
            tag = str(epsilon).replace(".", "_")
            print(
                f"{label}_ROW_CD_SLOPE_EPS_{tag}="
                f"{primary['origin_slope_estimate_on_frozen_Delta_L']:.17g}"
            )
            print(
                f"{label}_PAIR_CD_SLOPE_EPS_{tag}="
                f"{pair['origin_slope_estimate_on_frozen_Delta_L']:.17g}"
            )
            print(
                f"{label}_POSITIVE_ONE_SIDED_SLOPE_EPS_{tag}="
                f"{metrics['positive_one_sided_fixed']['origin_slope_estimate_on_frozen_Delta_L']:.17g}"
            )
            print(
                f"{label}_NEGATIVE_ONE_SIDED_SLOPE_EPS_{tag}="
                f"{metrics['negative_one_sided_fixed']['origin_slope_estimate_on_frozen_Delta_L']:.17g}"
            )
            print(
                f"{label}_EVEN_SECOND_MEAN_ABS_EPS_{tag}="
                f"{metrics['fixed_even_second_difference']['mean_abs']:.17g}"
            )
            print(
                f"{label}_WRONG_SWITCH_PLUS_EPS_{tag}="
                f"{metrics['dynamic_wrong_switch_count_plus']}"
            )
            print(
                f"{label}_WRONG_SWITCH_MINUS_EPS_{tag}="
                f"{metrics['dynamic_wrong_switch_count_minus']}"
            )
    print("PRIMARY_P_VALUE_COUNT_ADDED=0")
    print("SECONDARY_P_VALUE_COUNT_ADDED=0")
    print("INFERENCE_PERFORMED=False")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("ADAPTIVE_EPSILON_SELECTION_PERFORMED=False")
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
