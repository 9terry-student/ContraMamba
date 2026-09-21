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
ROW_FILE = "reference_backend_rows.jsonl"
RAW_SUMMARY_FILE = "raw_reference_backend_summary.json"
RAW_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

ANALYSIS_FILE = "reference_backend_analysis.json"
ROW_ANALYSIS_FILE = "reference_backend_row_analysis.jsonl"
MANIFEST_FILE = "artifact_manifest.json"

RESULT = "PASS_GEN4_FACTOR2_REFERENCE_BACKEND_STATIC_ANALYSIS"
SCHEMA_VERSION = "gen4-factor2-reference-backend-static-analysis-v1"


class ReferenceBackendAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReferenceBackendAnalysisError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReferenceBackendAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str, raw_freeze_commit: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", raw_freeze_commit, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git_blob_bytes(path: Path) -> bytes:
    rel = repo_relative(path)
    return subprocess.check_output(
        ["git", "show", f"HEAD:{rel}"],
        cwd=ROOT,
        stderr=subprocess.STDOUT,
    )


def git_blob_text(path: Path) -> str:
    return git_blob_bytes(path).decode("utf-8")


def read_jsonl_blob(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in git_blob_text(path).splitlines()
        if line.strip()
    ]


def origin_slope(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1 and a.size > 0, "SLOPE_SHAPE")
    denom = float(np.dot(a, a))
    require(denom > 0.0 and math.isfinite(denom), "SLOPE_DENOM")
    value = float(np.dot(a, b) / denom)
    require(math.isfinite(value), "SLOPE_FINITE")
    return value


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    require(a.shape == p.shape, "RMSE_SHAPE")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def descriptive(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "DESC")
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "mean_abs": float(np.mean(np.abs(a))),
        "max_abs": float(np.max(np.abs(a))),
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
    }


def vector_relation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    slope = origin_slope(x, y)
    return {
        "origin_slope": slope,
        "rmse_vs_1x": rmse(y, x),
        "rmse_vs_2x": rmse(y, 2.0 * np.asarray(x, dtype=np.float64)),
        "distance_from_1x": abs(slope - 1.0),
        "distance_from_2x": abs(slope - 2.0),
        "p_value_count": 0,
        "inference_performed": False,
    }


def validate_raw(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sums: dict[str, str] = {}
    for line in git_blob_text(raw_dir / SUMS_FILE).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(
            name in {ROW_FILE, RAW_SUMMARY_FILE, RAW_MANIFEST_FILE},
            f"SUM_NAME:{name}",
        )
        require(sha256_bytes(git_blob_bytes(raw_dir / name)) == digest, f"SUM_SHA:{name}")
        sums[name] = digest
    require(set(sums) == {ROW_FILE, RAW_SUMMARY_FILE, RAW_MANIFEST_FILE}, "SUM_SET")
    summary = json.loads(git_blob_text(raw_dir / RAW_SUMMARY_FILE))
    manifest = json.loads(git_blob_text(raw_dir / RAW_MANIFEST_FILE))
    require(summary["result"] == "PASS_GEN4_FACTOR2_REFERENCE_BACKEND_RAW", "RAW_RESULT")
    require(summary["subset_pair_count"] == 4, "RAW_SUBSET")
    require(summary["row_count"] == 16, "RAW_ROWS")
    require(summary["epsilon"] == 0.03125, "RAW_EPSILON")
    require(summary["forward_count"] == 96, "RAW_FORWARD")
    require(summary["local_backward_count"] == 32, "RAW_BACKWARD")
    require(summary["reference_backend_forced_on_all_48_mixers"] is True, "RAW_SLOW")
    require(summary["same_fast_derived_direction_used_for_both_backends"] is True, "RAW_DIRECTION")
    require(summary["p_value_count_executed"] == 0, "RAW_P")
    require(summary["scientific_conclusion"] is None, "RAW_CONCLUSION")
    require(manifest["row_sha256"] == sums[ROW_FILE], "ROW_SHA")
    require(manifest["summary_sha256"] == sums[RAW_SUMMARY_FILE], "SUMMARY_SHA")
    rows = read_jsonl_blob(raw_dir / ROW_FILE)
    require(len(rows) == 16, "ROW_COUNT")
    return rows, summary, manifest


def analyze(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scale: dict[str, Any] = {}
    for scale in SCALES:
        srows = [r for r in rows if r["scale"] == scale]
        require(len(srows) == 8, f"SCALE_ROWS:{scale}")

        frozen = [float(r["frozen_Delta_L_row"]) for r in srows]
        fast_grad = [float(r["fast_recomputed_Delta_L_row"]) for r in srows]
        slow_grad = [float(r["slow_reference_Delta_L_on_fast_direction"]) for r in srows]
        fast_cd = [float(r["fast_central_difference_on_fast_direction"]) for r in srows]
        slow_cd = [float(r["slow_central_difference_on_fast_direction"]) for r in srows]

        by_scale[scale] = {
            "fast_autograd_replay_vs_frozen":
                vector_relation(frozen, fast_grad),
            "fast_central_vs_frozen_autograd":
                vector_relation(frozen, fast_cd),
            "slow_reference_autograd_vs_frozen_fast_autograd":
                vector_relation(frozen, slow_grad),
            "slow_reference_central_vs_frozen_fast_autograd":
                vector_relation(frozen, slow_cd),
            "slow_reference_central_vs_slow_reference_autograd":
                vector_relation(slow_grad, slow_cd),
            "fast_central_vs_slow_central":
                vector_relation(fast_cd, slow_cd),
            "fast_slow_native_margin_abs_diff":
                descriptive([float(r["fast_slow_native_margin_abs_diff"]) for r in srows]),
            "fast_slow_plus_logit_max_abs_diff":
                descriptive([float(r["fast_slow_plus_logit_max_abs_diff"]) for r in srows]),
            "fast_slow_minus_logit_max_abs_diff":
                descriptive([float(r["fast_slow_minus_logit_max_abs_diff"]) for r in srows]),
            "p_value_count": 0,
            "inference_performed": False,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "scales": list(SCALES),
        "subset_pair_count": 4,
        "row_count": 16,
        "epsilon": 0.03125,
        "primary_question": (
            "Does sequential MambaMixer.slow_forward autograd recover the "
            "forward central derivative on the same fast-derived direction?"
        ),
        "by_scale": by_scale,
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "adaptive_backend_selection_performed": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "scientific_conclusion": None,
    }


def write_outputs(
    *,
    expected_head: str,
    raw_freeze_commit: str,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head, raw_freeze_commit)
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    rows, summary, manifest = validate_raw(raw_dir)
    analysis = analyze(rows)
    analysis["analysis_head"] = expected_head
    analysis["raw_freeze_commit"] = raw_freeze_commit
    analysis["raw_execution_head"] = summary["execution_head"]

    output_dir.mkdir(parents=True, exist_ok=False)
    row_out = b"".join(
        canonical_json_bytes({
            "scale": r["scale"],
            "source_pair_id": r["source_pair_id"],
            "contrast_cell_id": r["contrast_cell_id"],
            "frozen_Delta_L_row": r["frozen_Delta_L_row"],
            "fast_recomputed_Delta_L_row": r["fast_recomputed_Delta_L_row"],
            "slow_reference_Delta_L_on_fast_direction":
                r["slow_reference_Delta_L_on_fast_direction"],
            "fast_central_difference_on_fast_direction":
                r["fast_central_difference_on_fast_direction"],
            "slow_central_difference_on_fast_direction":
                r["slow_central_difference_on_fast_direction"],
            "fast_slow_native_margin_abs_diff":
                r["fast_slow_native_margin_abs_diff"],
            "fast_slow_plus_logit_max_abs_diff":
                r["fast_slow_plus_logit_max_abs_diff"],
            "fast_slow_minus_logit_max_abs_diff":
                r["fast_slow_minus_logit_max_abs_diff"],
        })
        for r in rows
    )
    (output_dir / ROW_ANALYSIS_FILE).write_bytes(row_out)
    (output_dir / ANALYSIS_FILE).write_bytes(pretty_json_bytes(analysis))
    out_manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "analysis_head": expected_head,
        "raw_freeze_commit": raw_freeze_commit,
        "raw_row_sha256": manifest["row_sha256"],
        "raw_summary_sha256": manifest["summary_sha256"],
        "analysis_sha256": sha256_file(output_dir / ANALYSIS_FILE),
        "row_analysis_sha256": sha256_file(output_dir / ROW_ANALYSIS_FILE),
        "p_value_count_added": 0,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(out_manifest))
    names = (ANALYSIS_FILE, ROW_ANALYSIS_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_text(
        "".join(f"{sha256_file(output_dir / n)}  {n}\n" for n in sorted(names)),
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--raw-freeze-commit", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_outputs(
        expected_head=args.expected_head,
        raw_freeze_commit=args.raw_freeze_commit,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    print("RESULT=" + result["result"])
    for scale in SCALES:
        metrics = result["by_scale"][scale]
        print(f"{scale.upper()}_FAST_AUTOGRAD_REPLAY_SLOPE="
              f"{metrics['fast_autograd_replay_vs_frozen']['origin_slope']:.17g}")
        print(f"{scale.upper()}_FAST_CD_VS_FROZEN_SLOPE="
              f"{metrics['fast_central_vs_frozen_autograd']['origin_slope']:.17g}")
        print(f"{scale.upper()}_SLOW_AUTOGRAD_VS_FROZEN_SLOPE="
              f"{metrics['slow_reference_autograd_vs_frozen_fast_autograd']['origin_slope']:.17g}")
        print(f"{scale.upper()}_SLOW_CD_VS_FROZEN_SLOPE="
              f"{metrics['slow_reference_central_vs_frozen_fast_autograd']['origin_slope']:.17g}")
        print(f"{scale.upper()}_SLOW_CD_VS_SLOW_AUTOGRAD_SLOPE="
              f"{metrics['slow_reference_central_vs_slow_reference_autograd']['origin_slope']:.17g}")
        print(f"{scale.upper()}_FAST_CD_VS_SLOW_CD_SLOPE="
              f"{metrics['fast_central_vs_slow_central']['origin_slope']:.17g}")
        print(f"{scale.upper()}_FAST_SLOW_NATIVE_MARGIN_MAX_ABS="
              f"{metrics['fast_slow_native_margin_abs_diff']['max_abs']:.17g}")
        print(f"{scale.upper()}_FAST_SLOW_PLUS_LOGIT_MAX_ABS="
              f"{metrics['fast_slow_plus_logit_max_abs_diff']['max']:.17g}")
        print(f"{scale.upper()}_FAST_SLOW_MINUS_LOGIT_MAX_ABS="
              f"{metrics['fast_slow_minus_logit_max_abs_diff']['max']:.17g}")
    print("P_VALUE_COUNT_ADDED=0")
    print("INFERENCE_PERFORMED=False")
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
