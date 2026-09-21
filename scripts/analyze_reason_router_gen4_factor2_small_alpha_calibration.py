#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
INPUT_FREEZE_COMMIT = "a88b71070310906b664c578c9627d9af3ad441d1"

SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(5701, 6001))
PAIR_COUNT = 300
ALPHAS = (0.25, 0.125, 0.0625, 0.03125)

READOUT_DIR = ROOT / (
    "reports/reason_router_gen4_mamba370m14b_"
    "factor2_small_alpha_native_readout_raw_v1"
)
BEHAVIOR_DIR = ROOT / (
    "reports/reason_router_gen4_mamba370m14b_"
    "factor2_small_alpha_behavioral_raw_v1"
)
STATE_DIR = ROOT / (
    "reports/reason_router_gen4_mamba370m14b_"
    "factor2_small_alpha_state_capture_raw_v1"
)

READOUT_ITEM_FILE = "factor2_small_alpha_native_readout_items.jsonl"
READOUT_SUMMARY_FILE = "raw_readout_summary.json"
READOUT_MANIFEST_FILE = "artifact_manifest.json"

BEHAVIOR_ITEM_FILE = "factor2_small_alpha_behavioral_items.jsonl"
BEHAVIOR_SUMMARY_FILE = "raw_behavioral_summary.json"
BEHAVIOR_MANIFEST_FILE = "artifact_manifest.json"

STATE_FILE = "factor2_small_alpha_states.npz"
STATE_INDEX_FILE = "capture_index.jsonl"
STATE_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_READOUT_SHA256 = {
    READOUT_MANIFEST_FILE:
        "a0aed230659c6f69b2bd21cba9326b34a6a264139274fd6a7ac3a594ade73cdd",
    READOUT_ITEM_FILE:
        "cb4e4115d7f21cc66e63153f49942eb4e3e1f6d410c1475c24666cc4f76a9fd8",
    READOUT_SUMMARY_FILE:
        "63d5eeec7f8353380d166bb25c994726e3f2cf4280568793c773a1182af1b3f7",
}
EXPECTED_BEHAVIOR_SHA256 = {
    BEHAVIOR_MANIFEST_FILE:
        "e3de43969bc375f519fb39cc416661a734cde65c9891276f800e8dd006d5bf25",
    BEHAVIOR_ITEM_FILE:
        "4be32c827d751b9b2f2f4a8211d594086635b87e6700c89bd581692e07558529",
    BEHAVIOR_SUMMARY_FILE:
        "a595acec31ca6ff74bd0fb8fdfe4c1aa6733dd42eebb019bb0421854c124ad95",
}
EXPECTED_STATE_SHA256 = {
    STATE_MANIFEST_FILE:
        "5f3c18b46f08acdda8a8148c607e015b0bba02dc68741eea0d79758ab98c7912",
    STATE_INDEX_FILE:
        "89927a67d16b68999eebdcc39cf034f722e268d607e007556d05b2277f3f4727",
    STATE_FILE:
        "0ee272ce9ba0f2eaecc183236f2e796c33c0dfe422158032727a733b31f38d02",
}

EXPECTED_SELECTED = {"mamba370m": "P3", "mamba14b": "P5"}
EXPECTED_CONTROL = {"mamba370m": "P5", "mamba14b": "P4"}
EXPECTED_CHECKPOINT = {
    "mamba370m":
        "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
    "mamba14b":
        "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a",
}

CONDITION_BY_ALPHA = {
    0.25: "control_alpha_0_25",
    0.125: "control_alpha_0_125",
    0.0625: "control_alpha_0_0625",
    0.03125: "control_alpha_0_03125",
}
RESTORED_CONDITION = "restored_native"

PAIR_FILE = "pair_level_merge.jsonl"
CALIBRATION_FILE = "calibration_analysis.json"
DISPLACEMENT_FILE = "displacement_context.json"
MANIFEST_FILE = "artifact_manifest.json"

RESULT = "PASS_GEN4_FACTOR2_SMALL_ALPHA_STATIC_CALIBRATION_ANALYSIS"
SCHEMA_VERSION = "gen4-factor2-small-alpha-static-calibration-v1"


class Factor2StaticCalibrationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Factor2StaticCalibrationError(message)


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
        raise Factor2StaticCalibrationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> str:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    head = git("rev-parse", "HEAD")
    require(head == expected_head, f"HEAD:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            INPUT_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "INPUT_FREEZE_NOT_ANCESTOR")
    return head


def validate_protocol() -> None:
    require(PAIR_IDS == tuple(f"xg1_fact_{i}" for i in range(5701, 6001)), "PAIR_IDS")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(CELLS == ("C0_SHAM", "C2_NAME"), "CELLS")
    require(SCALES == ("mamba370m", "mamba14b"), "SCALES")
    require(ALPHAS == (0.25, 0.125, 0.0625, 0.03125), "ALPHAS")
    require(all(alpha > 0.0 for alpha in ALPHAS), "POSITIVE_ALPHAS")
    require(1.0 not in ALPHAS, "NO_ALPHA1_BEHAVIORAL")
    require(set(CONDITION_BY_ALPHA) == set(ALPHAS), "CONDITION_ALPHA_SET")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise Factor2StaticCalibrationError(
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
        raise Factor2StaticCalibrationError(
            f"GIT_BLOB_FAILURE:{rel}"
        ) from exc


def git_blob_text(path: Path) -> str:
    return git_blob_bytes(path).decode("utf-8")


def read_jsonl_blob(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(git_blob_text(path).splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def validate_frozen_bundle(
    root: Path,
    expected_sha256: Mapping[str, str],
) -> None:
    sums = git_blob_text(root / SUMS_FILE)
    observed: dict[str, str] = {}
    for line in sums.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"SUM_DUPLICATE:{root}:{name}")
        require(name in expected_sha256, f"SUM_UNEXPECTED:{root}:{name}")
        raw = git_blob_bytes(root / name)
        actual = sha256_bytes(raw)
        require(actual == digest, f"SUM_DIGEST:{root}:{name}")
        require(
            expected_sha256[name] == digest,
            f"FROZEN_DIGEST:{root}:{name}",
        )
        observed[name] = digest
    require(set(observed) == set(expected_sha256), f"SUM_SET:{root}")


def validate_frozen_inputs() -> None:
    validate_frozen_bundle(READOUT_DIR, EXPECTED_READOUT_SHA256)
    validate_frozen_bundle(BEHAVIOR_DIR, EXPECTED_BEHAVIOR_SHA256)
    validate_frozen_bundle(STATE_DIR, EXPECTED_STATE_SHA256)

    readout_summary = json.loads(
        git_blob_text(READOUT_DIR / READOUT_SUMMARY_FILE)
    )
    readout_manifest = json.loads(
        git_blob_text(READOUT_DIR / READOUT_MANIFEST_FILE)
    )
    require(
        readout_summary["result"]
        == "PASS_GEN4_FACTOR2_SMALL_ALPHA_NATIVE_READOUT_RAW",
        "READOUT_RESULT",
    )
    require(readout_summary["population"] == "xg1_fact_5701..xg1_fact_6000", "READOUT_POP")
    require(int(readout_summary["pair_count"]) == 300, "READOUT_PAIR_COUNT")
    require(int(readout_summary["row_count"]) == 1200, "READOUT_ROWS")
    require(readout_summary["behavioral_response_accessed"] is False, "READOUT_BLIND")
    require(readout_summary["predicted_behavior_computed"] is False, "READOUT_PREDICTED")
    require(readout_summary["behavioral_merge_computed"] is False, "READOUT_MERGE")
    require(int(readout_summary["p_value_count_executed"]) == 0, "READOUT_P")
    require(readout_manifest["item_sha256"] == EXPECTED_READOUT_SHA256[READOUT_ITEM_FILE], "READOUT_ITEM_SHA")

    behavior_summary = json.loads(
        git_blob_text(BEHAVIOR_DIR / BEHAVIOR_SUMMARY_FILE)
    )
    behavior_manifest = json.loads(
        git_blob_text(BEHAVIOR_DIR / BEHAVIOR_MANIFEST_FILE)
    )
    require(
        behavior_summary["result"]
        == "PASS_GEN4_FACTOR2_SMALL_ALPHA_BEHAVIORAL_RAW",
        "BEHAVIOR_RESULT",
    )
    require(behavior_summary["population"] == "xg1_fact_5701..xg1_fact_6000", "BEHAVIOR_POP")
    require(behavior_summary["behavioral_alphas"] == list(ALPHAS), "BEHAVIOR_ALPHAS")
    require(behavior_summary["primary_curve_reserved_for_static_analysis"] == "K(alpha)", "PRIMARY_CURVE")
    require(int(behavior_summary["behavior_row_count"]) == 6000, "BEHAVIOR_ROWS")
    require(int(behavior_summary["p_value_count_executed"]) == 0, "BEHAVIOR_P")
    require(behavior_summary["negative_alpha_arm_executed"] is False, "NEGATIVE_ALPHA")
    require(behavior_summary["adaptive_rerun_executed"] is False, "ADAPTIVE_RERUN")
    require(behavior_manifest["item_sha256"] == EXPECTED_BEHAVIOR_SHA256[BEHAVIOR_ITEM_FILE], "BEHAVIOR_ITEM_SHA")

    state_manifest = json.loads(
        git_blob_text(STATE_DIR / STATE_MANIFEST_FILE)
    )
    require(
        state_manifest["result"]
        == "PASS_GEN4_FACTOR2_SMALL_ALPHA_STATE_CAPTURE_RAW",
        "STATE_RESULT",
    )
    require(state_manifest["population"] == "xg1_fact_5701..xg1_fact_6000", "STATE_POP")
    require(state_manifest["behavioral_alphas"] == list(ALPHAS), "STATE_ALPHAS")
    require(int(state_manifest["state_row_count"]) == 1200, "STATE_ROWS")
    require(state_manifest["negative_alpha_arm_executed"] is False, "STATE_NEGATIVE")
    require(state_manifest["adaptive_rerun_executed"] is False, "STATE_ADAPTIVE")
    require(int(state_manifest["p_value_count_executed"]) == 0, "STATE_P")


def sign_code(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def descriptive(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "DESCRIPTIVE")
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "sd_population": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
    }


def rank_average(values: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "RANK_INPUT")
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        average_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = average_rank
        i = j
    return ranks


def pearson_coefficient(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1 and a.size >= 2, "PEARSON_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "PEARSON_FINITE")
    ac = a - np.mean(a)
    bc = b - np.mean(b)
    denom = float(
        np.sqrt(
            np.dot(ac, ac)
            * np.dot(bc, bc)
        )
    )
    require(denom > 0.0 and math.isfinite(denom), "PEARSON_DENOM")
    result = float(np.dot(ac, bc) / denom)
    require(math.isfinite(result), "PEARSON_RESULT")
    return result


def spearman_coefficient(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    return pearson_coefficient(rank_average(x), rank_average(y))


def origin_slope(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1 and a.size > 0, "SLOPE_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "SLOPE_FINITE")
    denom = float(np.dot(a, a))
    require(denom > 0.0 and math.isfinite(denom), "SLOPE_DENOM")
    value = float(np.dot(a, b) / denom)
    require(math.isfinite(value), "SLOPE_RESULT")
    return value


def rmse(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    require(a.shape == p.shape and a.ndim == 1, "RMSE_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(p).all(), "RMSE_FINITE")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def read_pair_readout() -> dict[str, dict[str, dict[str, float]]]:
    rows = read_jsonl_blob(READOUT_DIR / READOUT_ITEM_FILE)
    require(len(rows) == 1200, "READOUT_ITEM_COUNT")
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        scale = str(row["scale"])
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        key = (scale, pair, cell)
        require(scale in SCALES, f"READOUT_SCALE:{key}")
        require(pair in PAIR_IDS, f"READOUT_PAIR:{key}")
        require(cell in CELLS, f"READOUT_CELL:{key}")
        require(key not in by_key, f"READOUT_DUP:{key}")
        require(row["selected_plane"] == EXPECTED_SELECTED[scale], f"READOUT_SELECTED:{key}")
        require(row["control_plane"] == EXPECTED_CONTROL[scale], f"READOUT_CONTROL:{key}")
        require(row["checkpoint_sha256"] == EXPECTED_CHECKPOINT[scale], f"READOUT_CHECKPOINT:{key}")
        require(row["behavioral_response_accessed"] is False, f"READOUT_BLIND:{key}")
        value = float(row["Delta_L_row"])
        require(math.isfinite(value), f"READOUT_NONFINITE:{key}")
        by_key[key] = row

    expected = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in PAIR_IDS
        for cell in CELLS
    }
    require(set(by_key) == expected, "READOUT_COVERAGE")

    out: dict[str, dict[str, dict[str, float]]] = {}
    for scale in SCALES:
        out[scale] = {}
        for pair in PAIR_IDS:
            c0 = float(by_key[(scale, pair, "C0_SHAM")]["Delta_L_row"])
            c2 = float(by_key[(scale, pair, "C2_NAME")]["Delta_L_row"])
            pair_value = float(np.mean(np.asarray([c0, c2], dtype=np.float64)))
            out[scale][pair] = {
                "Delta_L": pair_value,
                "Delta_L_C0_SHAM": c0,
                "Delta_L_C2_NAME": c2,
            }
    return out


def read_pair_behavior() -> dict[str, dict[str, dict[float, float]]]:
    rows = read_jsonl_blob(BEHAVIOR_DIR / BEHAVIOR_ITEM_FILE)
    require(len(rows) == 6000, "BEHAVIOR_ITEM_COUNT")
    by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}

    conditions = {RESTORED_CONDITION, *CONDITION_BY_ALPHA.values()}
    for row in rows:
        scale = str(row["scale"])
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        condition = str(row["condition"])
        key = (scale, pair, cell, condition)
        require(scale in SCALES, f"BEHAVIOR_SCALE:{key}")
        require(pair in PAIR_IDS, f"BEHAVIOR_PAIR:{key}")
        require(cell in CELLS, f"BEHAVIOR_CELL:{key}")
        require(condition in conditions, f"BEHAVIOR_CONDITION:{key}")
        require(key not in by_key, f"BEHAVIOR_DUP:{key}")
        margin = float(row["correct_class_logit_margin"])
        require(math.isfinite(margin), f"BEHAVIOR_MARGIN:{key}")
        require(row["primary_inference_executed"] is False, f"BEHAVIOR_INFERENCE:{key}")
        by_key[key] = row

    expected = {
        (scale, pair, cell, condition)
        for scale in SCALES
        for pair in PAIR_IDS
        for cell in CELLS
        for condition in conditions
    }
    require(set(by_key) == expected, "BEHAVIOR_COVERAGE")

    out: dict[str, dict[str, dict[float, float]]] = {}
    for scale in SCALES:
        out[scale] = {}
        for pair in PAIR_IDS:
            restored = float(np.mean([
                float(by_key[(scale, pair, cell, RESTORED_CONDITION)][
                    "correct_class_logit_margin"
                ])
                for cell in CELLS
            ]))
            alpha_values: dict[float, float] = {}
            for alpha in ALPHAS:
                condition = CONDITION_BY_ALPHA[alpha]
                control = float(np.mean([
                    float(by_key[(scale, pair, cell, condition)][
                        "correct_class_logit_margin"
                    ])
                    for cell in CELLS
                ]))
                d_beh = restored - control
                require(math.isfinite(d_beh), f"D_BEH:{scale}:{pair}:{alpha}")
                alpha_values[alpha] = d_beh
            out[scale][pair] = alpha_values
    return out


def summarize_calibration(
    delta_l: Sequence[float] | np.ndarray,
    d_beh: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    require(alpha in ALPHAS, f"SUMMARY_ALPHA:{alpha}")
    d = np.asarray(delta_l, dtype=np.float64)
    y = np.asarray(d_beh, dtype=np.float64)
    require(d.shape == (PAIR_COUNT,), "SUMMARY_DELTA_SHAPE")
    require(y.shape == (PAIR_COUNT,), "SUMMARY_BEHAVIOR_SHAPE")
    x = alpha * d
    k = origin_slope(x, y)
    slope_on_delta_l = origin_slope(d, y)

    residual_1x = y - x
    residual_2x = y - 2.0 * x
    sign_agree = np.asarray(
        [sign_code(float(a)) == sign_code(float(b)) for a, b in zip(d, y)],
        dtype=np.float64,
    )

    return {
        "alpha": alpha,
        "origin_slope_K": k,
        "origin_slope_D_BEH_on_Delta_L": slope_on_delta_l,
        "slope_identity_alpha_times_K": alpha * k,
        "pearson": pearson_coefficient(x, y),
        "spearman_coefficient_only": spearman_coefficient(x, y),
        "sign_agreement": {
            "count": int(np.sum(sign_agree)),
            "total": PAIR_COUNT,
            "fraction": float(np.mean(sign_agree)),
        },
        "rmse_vs_alpha_Delta_L": rmse(y, x),
        "rmse_vs_2alpha_Delta_L": rmse(y, 2.0 * x),
        "residual_D_minus_alpha_Delta_L": descriptive(residual_1x),
        "residual_D_minus_2alpha_Delta_L": descriptive(residual_2x),
        "Delta_L": descriptive(d),
        "alpha_Delta_L": descriptive(x),
        "D_BEH": descriptive(y),
        "distance_from_first_order_K1": abs(k - 1.0),
        "distance_from_factor2_K2": abs(k - 2.0),
        "p_value_count": 0,
        "inference_performed": False,
    }


def build_pair_merge(
    readout: Mapping[str, Mapping[str, Mapping[str, float]]],
    behavior: Mapping[str, Mapping[str, Mapping[float, float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vectors: dict[str, dict[float, list[float]]] = {
        scale: {alpha: [] for alpha in ALPHAS}
        for scale in SCALES
    }
    delta_vectors: dict[str, list[float]] = {scale: [] for scale in SCALES}

    for pair in PAIR_IDS:
        row: dict[str, Any] = {"source_pair_id": pair}
        for scale in SCALES:
            d = float(readout[scale][pair]["Delta_L"])
            row[f"{scale}_Delta_L"] = d
            row[f"{scale}_Delta_L_C0_SHAM"] = float(
                readout[scale][pair]["Delta_L_C0_SHAM"]
            )
            row[f"{scale}_Delta_L_C2_NAME"] = float(
                readout[scale][pair]["Delta_L_C2_NAME"]
            )
            delta_vectors[scale].append(d)
            for alpha in ALPHAS:
                y = float(behavior[scale][pair][alpha])
                tag = str(alpha).replace(".", "_")
                row[f"{scale}_D_BEH_alpha_{tag}"] = y
                row[f"{scale}_alpha_Delta_L_alpha_{tag}"] = alpha * d
                row[f"{scale}_residual_1x_alpha_{tag}"] = y - alpha * d
                row[f"{scale}_residual_2x_alpha_{tag}"] = y - 2.0 * alpha * d
                row[f"{scale}_sign_agree_alpha_{tag}"] = (
                    sign_code(d) == sign_code(y)
                )
                vectors[scale][alpha].append(y)
        rows.append(row)

    require(len(rows) == PAIR_COUNT, "PAIR_MERGE_COUNT")

    curves: dict[str, Any] = {}
    for scale in SCALES:
        d = np.asarray(delta_vectors[scale], dtype=np.float64)
        curve: dict[str, Any] = {}
        for alpha in ALPHAS:
            tag = str(alpha)
            y = np.asarray(vectors[scale][alpha], dtype=np.float64)
            curve[tag] = summarize_calibration(d, y, alpha=alpha)
        curves[scale] = curve
    return rows, curves


def pairwise_native_distances(native: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(native, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == PAIR_COUNT, "NATIVE_SHAPE")
    gram = x @ x.T
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, np.inf)
    index = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(PAIR_COUNT), index])
    require(np.isfinite(distance).all(), "NATIVE_DISTANCE")
    return distance, index


def intervention_to_native_distances(
    intervened: np.ndarray,
    native: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(intervened, dtype=np.float64)
    x = np.asarray(native, dtype=np.float64)
    require(y.shape == x.shape and x.shape[0] == PAIR_COUNT, "INTERVENTION_SHAPE")
    d2 = np.maximum(
        np.sum(y * y, axis=1)[:, None]
        + np.sum(x * x, axis=1)[None, :]
        - 2.0 * (y @ x.T),
        0.0,
    )
    np.fill_diagonal(d2, np.inf)
    index = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(PAIR_COUNT), index])
    require(np.isfinite(distance).all(), "INTERVENTION_DISTANCE")
    return distance, index


def summarize_displacement(
    native_full: np.ndarray,
    native_strong: np.ndarray,
    delta_full: np.ndarray,
    strong_indices: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    nf = np.asarray(native_full, dtype=np.float64)
    ns = np.asarray(native_strong, dtype=np.float64)
    du = np.asarray(delta_full, dtype=np.float64)
    strong = np.asarray(strong_indices, dtype=np.int64)
    require(nf.shape == du.shape and nf.shape[0] == PAIR_COUNT, "DISP_FULL_SHAPE")
    require(ns.shape[0] == PAIR_COUNT, "DISP_STRONG_ROWS")
    ds = du[:, strong]
    require(ds.shape == ns.shape, "DISP_STRONG_SHAPE")
    require(
        np.isfinite(nf).all()
        and np.isfinite(ns).all()
        and np.isfinite(du).all(),
        "DISP_FINITE",
    )

    df = alpha * du
    dstrong = alpha * ds
    r_rel_full = np.linalg.norm(df, axis=1) / np.maximum(
        np.linalg.norm(nf, axis=1),
        1e-12,
    )
    r_rel_strong = np.linalg.norm(dstrong, axis=1) / np.maximum(
        np.linalg.norm(ns, axis=1),
        1e-12,
    )

    native_nn_full, native_idx_full = pairwise_native_distances(nf)
    native_nn_strong, native_idx_strong = pairwise_native_distances(ns)
    int_nn_full, int_idx_full = intervention_to_native_distances(nf + df, nf)
    int_nn_strong, int_idx_strong = intervention_to_native_distances(
        ns + dstrong,
        ns,
    )

    r_nn_full = int_nn_full / np.maximum(native_nn_full, 1e-12)
    r_nn_strong = int_nn_strong / np.maximum(native_nn_strong, 1e-12)

    return {
        "alpha": alpha,
        "R_rel_full": descriptive(r_rel_full),
        "R_rel_strong": descriptive(r_rel_strong),
        "R_NN_full": {
            **descriptive(r_nn_full),
            "fraction_gt_1": float(np.mean(r_nn_full > 1.0)),
            "fraction_gt_2": float(np.mean(r_nn_full > 2.0)),
        },
        "R_NN_strong": {
            **descriptive(r_nn_strong),
            "fraction_gt_1": float(np.mean(r_nn_strong > 1.0)),
            "fraction_gt_2": float(np.mean(r_nn_strong > 2.0)),
        },
        "nearest_native_identity_change_rate_full": float(
            np.mean(native_idx_full != int_idx_full)
        ),
        "nearest_native_identity_change_rate_strong": float(
            np.mean(native_idx_strong != int_idx_strong)
        ),
        "p_value_count": 0,
        "inference_performed": False,
    }


def build_displacement_context() -> dict[str, Any]:
    index = read_jsonl_blob(STATE_DIR / STATE_INDEX_FILE)
    require(len(index) == 1200, "STATE_INDEX_COUNT")

    by_scale: dict[str, list[dict[str, Any]]] = {}
    for scale in SCALES:
        rows = [row for row in index if str(row["scale"]) == scale]
        rows.sort(key=lambda row: int(row["scale_row_index"]))
        require(len(rows) == 600, f"STATE_INDEX_SCALE:{scale}")
        require([int(row["scale_row_index"]) for row in rows] == list(range(600)), f"STATE_INDEX_ORDER:{scale}")
        by_scale[scale] = rows

    raw_npz = git_blob_bytes(STATE_DIR / STATE_FILE)
    require(
        sha256_bytes(raw_npz) == EXPECTED_STATE_SHA256[STATE_FILE],
        "STATE_NPZ_SHA",
    )

    result: dict[str, Any] = {
        "metric_family": [
            "R_rel_full",
            "R_rel_strong",
            "R_NN_full",
            "R_NN_strong",
            "nearest_native_identity_change_rate_full",
            "nearest_native_identity_change_rate_strong",
        ],
        "metric_family_source": (
            "Study-C low-displacement static displacement audit definitions"
        ),
        "alphas": list(ALPHAS),
        "scales": {},
        "p_value_count": 0,
        "inference_performed": False,
    }

    with np.load(io.BytesIO(raw_npz), allow_pickle=False) as data:
        for scale in SCALES:
            nf_all = np.asarray(data[f"{scale}__h_native_full"])
            ns_all = np.asarray(data[f"{scale}__h_native_strong"])
            du_all = np.asarray(data[f"{scale}__delta_control_full"])
            strong = np.asarray(
                data[f"{scale}__strong_indices"],
                dtype=np.int64,
            )
            require(nf_all.shape[0] == 600, f"STATE_FULL_ROWS:{scale}")
            require(ns_all.shape[0] == 600, f"STATE_STRONG_ROWS:{scale}")
            require(du_all.shape == nf_all.shape, f"STATE_DELTA_SHAPE:{scale}")
            require(np.array_equal(nf_all[:, strong], ns_all), f"STATE_GATHER:{scale}")

            scale_result: dict[str, Any] = {}
            rows_scale = by_scale[scale]
            for cell in CELLS:
                positions = [
                    int(row["scale_row_index"])
                    for row in rows_scale
                    if str(row["contrast_cell_id"]) == cell
                ]
                require(len(positions) == PAIR_COUNT, f"STATE_CELL_COUNT:{scale}:{cell}")
                pair_order = [
                    str(rows_scale[pos]["source_pair_id"])
                    for pos in positions
                ]
                require(pair_order == list(PAIR_IDS), f"STATE_PAIR_ORDER:{scale}:{cell}")
                pos = np.asarray(positions, dtype=np.int64)
                cell_result: dict[str, Any] = {}
                for alpha in ALPHAS:
                    cell_result[str(alpha)] = summarize_displacement(
                        nf_all[pos],
                        ns_all[pos],
                        du_all[pos],
                        strong,
                        alpha=alpha,
                    )
                scale_result[cell] = cell_result
            result["scales"][scale] = scale_result
    return result


def write_analysis(
    *,
    expected_head: str,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    analysis_head = authenticate_repo(expected_head)
    validate_frozen_inputs()
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    readout = read_pair_readout()
    behavior = read_pair_behavior()
    pair_rows, curves = build_pair_merge(readout, behavior)
    displacement = build_displacement_context()

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "analysis_head": analysis_head,
        "input_freeze_commit": INPUT_FREEZE_COMMIT,
        "population": "xg1_fact_5701..xg1_fact_6000",
        "source_pair_count": PAIR_COUNT,
        "cells": list(CELLS),
        "scales": list(SCALES),
        "alphas": list(ALPHAS),
        "primary_curve": "K(alpha)",
        "K_definition": (
            "origin_slope of pair-level D_BEH(alpha) on alpha*Delta_L: "
            "sum((alpha*Delta_L)*D_BEH)/sum((alpha*Delta_L)^2)"
        ),
        "pair_Delta_L_definition": "mean Delta_L_row over C0_SHAM,C2_NAME",
        "pair_D_BEH_definition": (
            "mean correct-class margin over cells at restored_native minus "
            "mean correct-class margin over cells at control_alpha"
        ),
        "first_order_reference_K": 1.0,
        "finite_range_factor2_reference_K": 2.0,
        "scale_curves": curves,
        "secondary_metric_family": [
            "pearson",
            "spearman_coefficient_only",
            "sign_agreement",
            "rmse_vs_alpha_Delta_L",
            "rmse_vs_2alpha_Delta_L",
            "descriptive_residuals",
        ],
        "displacement_context_file": DISPLACEMENT_FILE,
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "adaptive_rerun_performed": False,
        "negative_alpha_arm_added": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "training_executed": False,
        "scientific_conclusion": None,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    pair_raw = b"".join(canonical_json_bytes(row) for row in pair_rows)
    (output_dir / PAIR_FILE).write_bytes(pair_raw)
    (output_dir / CALIBRATION_FILE).write_bytes(pretty_json_bytes(analysis))
    (output_dir / DISPLACEMENT_FILE).write_bytes(pretty_json_bytes(displacement))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "analysis_head": analysis_head,
        "input_freeze_commit": INPUT_FREEZE_COMMIT,
        "frozen_input_read_mode": "git_blob_at_HEAD",
        "input_sha256": {
            "readout_items": EXPECTED_READOUT_SHA256[READOUT_ITEM_FILE],
            "readout_summary": EXPECTED_READOUT_SHA256[READOUT_SUMMARY_FILE],
            "readout_manifest": EXPECTED_READOUT_SHA256[READOUT_MANIFEST_FILE],
            "behavior_items": EXPECTED_BEHAVIOR_SHA256[BEHAVIOR_ITEM_FILE],
            "behavior_summary": EXPECTED_BEHAVIOR_SHA256[BEHAVIOR_SUMMARY_FILE],
            "behavior_manifest": EXPECTED_BEHAVIOR_SHA256[BEHAVIOR_MANIFEST_FILE],
            "state_npz": EXPECTED_STATE_SHA256[STATE_FILE],
            "state_index": EXPECTED_STATE_SHA256[STATE_INDEX_FILE],
            "state_manifest": EXPECTED_STATE_SHA256[STATE_MANIFEST_FILE],
        },
        "pair_level_merge_sha256": sha256_file(output_dir / PAIR_FILE),
        "calibration_analysis_sha256": sha256_file(output_dir / CALIBRATION_FILE),
        "displacement_context_sha256": sha256_file(output_dir / DISPLACEMENT_FILE),
        "source_pair_count": PAIR_COUNT,
        "alphas": list(ALPHAS),
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "adaptive_rerun_performed": False,
        "negative_alpha_arm_added": False,
        "model_forward_count": 0,
        "backward_count": 0,
        "training_executed": False,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (PAIR_FILE, CALIBRATION_FILE, DISPLACEMENT_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static factor-2 small-alpha calibration analysis. "
            "Merges frozen response-blind Delta_L with frozen four-alpha "
            "behavioral responses and computes K(alpha) plus descriptive "
            "geometric context. Adds zero p-values and executes no model."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_analysis(
        expected_head=args.expected_head,
        output_dir=args.output_dir,
    )
    print("RESULT=" + result["result"])
    for scale in SCALES:
        label = scale.upper()
        for alpha in ALPHAS:
            metrics = result["scale_curves"][scale][str(alpha)]
            tag = str(alpha).replace(".", "_")
            print(f"{label}_K_ALPHA_{tag}={metrics['origin_slope_K']:.17g}")
            print(
                f"{label}_PEARSON_ALPHA_{tag}="
                f"{metrics['pearson']:.17g}"
            )
            print(
                f"{label}_SPEARMAN_ALPHA_{tag}="
                f"{metrics['spearman_coefficient_only']:.17g}"
            )
            print(
                f"{label}_RMSE_1X_ALPHA_{tag}="
                f"{metrics['rmse_vs_alpha_Delta_L']:.17g}"
            )
            print(
                f"{label}_RMSE_2X_ALPHA_{tag}="
                f"{metrics['rmse_vs_2alpha_Delta_L']:.17g}"
            )
    print("PRIMARY_P_VALUE_COUNT_ADDED=0")
    print("SECONDARY_P_VALUE_COUNT_ADDED=0")
    print("INFERENCE_PERFORMED=False")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("ADAPTIVE_RERUN_PERFORMED=False")
    print("NEGATIVE_ALPHA_ARM_ADDED=False")
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
