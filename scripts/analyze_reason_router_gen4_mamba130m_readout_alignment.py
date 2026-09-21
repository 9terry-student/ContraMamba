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
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

RAW_FREEZE_COMMIT = "32eb3baf678a44946f4ecd102f917832929624a7"
RAW_EXECUTION_HEAD = "ccbfe14389655e3a900882505a10df0fb645217b"
RAW_RUN_NAME = "g4k-mamba130m-readout-ccbfe14-r2-2t4"

RAW_DIR = (
    ROOT
    / "reports/reason_router_gen4_mamba130m_readout_alignment_raw_runs"
    / RAW_RUN_NAME
)

PLAN_PATH = ROOT / "reports/reason_router_gen4_mamba130m_readout_alignment_prospective_plan.md"
PLAN_SHA256 = "a8a2cb2af1ed35b82f404a713c15ebf176cd645f653c6eacf73722b6e34058b3"

ITEM_FILE = "readout_alignment_items.jsonl"
SUMMARY_FILE = "raw_readout_alignment_summary.json"
RAW_MANIFEST_FILE = "artifact_manifest.json"
RAW_SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_RAW_SHA256 = {
    RAW_MANIFEST_FILE:
        "571ca7d01fd92ee337f3f4a65e072e8f7b3ee28b18b784b0c60a118882224579",
    SUMMARY_FILE:
        "e37b69b3d0d5212da56a0587a76f3817ec1c3e9b8cee56cf92b8caf605ccecf0",
    ITEM_FILE:
        "27c428fbcc3f9e89cc4dad566666e4d5307727637cf8bf1dd0c347744da18c85",
}

PAIR_IDS = tuple(f"xg1_fact_{i:03d}" for i in range(2701, 3001))
CELLS = ("C0_SHAM", "C2_NAME")
N = 300
ALPHA = 0.05

RESULT_SUPPORTED = "MAMBA130M_READOUT_ALIGNMENT_POSITIVE_SUPPORTED"
RESULT_NOT_ESTABLISHED = "MAMBA130M_READOUT_ALIGNMENT_POSITIVE_NOT_ESTABLISHED"
ANALYSIS_RESULT = "PASS_GEN4_MAMBA130M_READOUT_ALIGNMENT_STATIC_ANALYSIS"

ANALYSIS_FILE = "readout_alignment_analysis.json"
PAIR_FILE = "readout_alignment_pair_values.jsonl"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"


class AnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
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
        raise AnalysisError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH_MISMATCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")


def validate_sums(root: Path) -> None:
    sums_path = root / RAW_SUMS_FILE
    require(sums_path.is_file(), "SUMS_MISSING")
    observed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"SUM_DUPLICATE:{name}")
        observed[name] = digest
    require(observed == EXPECTED_RAW_SHA256, "SUMS_CONTENT")
    for name, digest in observed.items():
        require((root / name).is_file(), f"SUM_TARGET_MISSING:{name}")
        require(sha256_file(root / name) == digest, f"SUM_MISMATCH:{name}")


def read_raw_bundle(
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected_files = {
        ITEM_FILE,
        SUMMARY_FILE,
        RAW_MANIFEST_FILE,
        RAW_SUMS_FILE,
    }
    require(raw_dir.is_dir(), "RAW_DIR_MISSING")
    require(
        {p.name for p in raw_dir.iterdir() if p.is_file()} == expected_files,
        "RAW_FILE_SET",
    )
    require(sha256_file(PLAN_PATH) == PLAN_SHA256, "PLAN_SHA256")
    validate_sums(raw_dir)

    summary = json.loads(
        (raw_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (raw_dir / RAW_MANIFEST_FILE).read_text(encoding="utf-8")
    )

    expected_result = "PASS_GEN4_MAMBA130M_READOUT_ALIGNMENT_RAW"
    require(summary["result"] == expected_result, "RAW_SUMMARY_RESULT")
    require(manifest["result"] == expected_result, "RAW_MANIFEST_RESULT")
    require(summary["execution_head"] == RAW_EXECUTION_HEAD, "RAW_SUMMARY_HEAD")
    require(manifest["execution_head"] == RAW_EXECUTION_HEAD, "RAW_MANIFEST_HEAD")
    require(summary["population"] == "xg1_fact_2701..xg1_fact_3000", "RAW_POPULATION")
    require(summary["pair_count"] == N, "RAW_PAIR_COUNT")
    require(summary["cells"] == list(CELLS), "RAW_CELLS")
    require(summary["scale"] == "mamba130m", "RAW_SCALE")
    require(summary["selected_plane"] == "P3", "RAW_SELECTED_PLANE")
    require(summary["control_plane"] == "P5", "RAW_CONTROL_PLANE")
    require(summary["row_count"] == 600, "RAW_ROW_COUNT")
    require(
        summary["primary_endpoint_reserved_for_static_analysis"] == "Delta_L_130M",
        "RAW_PRIMARY_ENDPOINT",
    )
    require(
        summary["primary_test_reserved_for_static_analysis"]
        == "one_sample_student_t_greater",
        "RAW_PRIMARY_TEST",
    )
    require(summary["primary_alpha"] == ALPHA, "RAW_ALPHA")
    require(summary["primary_p_value_count_reserved"] == 1, "RAW_RESERVED_P_COUNT")
    require(summary["inferential_test_performed"] is False, "RAW_INFERENCE_BOUNDARY")
    require(summary["p_value_count_executed"] == 0, "RAW_P_COUNT")
    require(summary["scientific_conclusion"] is None, "RAW_CONCLUSION")
    require(summary["behavioral_bridge_D_BEH_accessed"] is False, "RAW_BEHAVIOR_ACCESS")
    require(summary["cross_scale_readout_result_accessed"] is False, "RAW_CROSS_SCALE_ACCESS")
    require(manifest["primary_p_value_count_executed"] == 0, "RAW_MANIFEST_P_COUNT")
    require(manifest["scientific_conclusion"] is None, "RAW_MANIFEST_CONCLUSION")

    rows = [
        json.loads(line)
        for line in (raw_dir / ITEM_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == 600, "RAW_ITEMS_COUNT")
    return rows, summary, manifest


def finite_float(value: Any, name: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE:{name}")
    return x


def descriptive(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0, "DESCRIPTIVE_SHAPE")
    require(bool(np.isfinite(a).all()), "DESCRIPTIVE_NONFINITE")
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd_sample": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "sd_population": float(a.std(ddof=0)),
        "min": float(a.min()),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(a.max()),
        "fraction_positive": float(np.mean(a > 0.0)),
        "fraction_negative": float(np.mean(a < 0.0)),
        "fraction_zero": float(np.mean(a == 0.0)),
    }


def nullable_descriptive(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in finite), "NULLABLE_NONFINITE")
    if not finite:
        return {
            "non_null_count": 0,
            "null_count": len(values),
            "descriptive": None,
        }
    return {
        "non_null_count": len(finite),
        "null_count": len(values) - len(finite),
        "descriptive": descriptive(finite),
    }


def pair_values(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        require(str(row["scale"]) == "mamba130m", "ROW_SCALE")
        key = (str(row["source_pair_id"]), str(row["contrast_cell_id"]))
        require(key not in by_key, f"DUPLICATE:{key}")
        by_key[key] = row

    expected = {
        (pair, cell)
        for pair in PAIR_IDS
        for cell in CELLS
    }
    require(set(by_key) == expected, "PAIR_CELL_COVERAGE")

    out: list[dict[str, Any]] = []
    for pair in PAIR_IDS:
        values = [
            finite_float(
                by_key[(pair, cell)]["Delta_L_row"],
                f"Delta_L_row:{pair}:{cell}",
            )
            for cell in CELLS
        ]
        out.append({
            "source_pair_id": pair,
            "Delta_L_C0_SHAM": values[0],
            "Delta_L_C2_NAME": values[1],
            "Delta_L": float(np.mean(np.asarray(values, dtype=np.float64))),
        })
    require(len(out) == N, "PAIR_COUNT")
    return out


def primary_inference(delta_l: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(delta_l, dtype=np.float64)
    require(a.shape == (N,), "PRIMARY_SHAPE")
    require(bool(np.isfinite(a).all()), "PRIMARY_NONFINITE")
    sd = float(a.std(ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")
    test = stats.ttest_1samp(
        a,
        popmean=0.0,
        alternative="greater",
    )
    t = float(test.statistic)
    p = float(test.pvalue)
    require(math.isfinite(t) and 0.0 <= p <= 1.0, "PRIMARY_TEST")
    mean = float(a.mean())
    sign_gate = mean > 0.0
    p_gate = p < ALPHA
    supported = bool(sign_gate and p_gate)
    return {
        "test": "one_sample_student_t",
        "alternative": "greater",
        "endpoint": "Delta_L_130M",
        "n": N,
        "df": N - 1,
        "alpha": ALPHA,
        "p_value_count": 1,
        "mean": mean,
        "sd_sample": sd,
        "t_statistic": t,
        "p_value": p,
        "positive_mean_sign_gate_pass": sign_gate,
        "alpha_gate_pass": p_gate,
        "positive_alignment_supported": supported,
    }


def secondary_descriptive(
    rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = (
        "gradient_full_l2",
        "selected_projection_l2",
        "control_projection_l2",
        "selected_projection_fraction",
        "control_projection_fraction",
        "selected_component_l2",
        "control_component_l2",
        "L_selected",
        "L_control",
        "Delta_L_row",
    )
    row_desc = {
        metric: descriptive([
            finite_float(row[metric], metric)
            for row in rows
        ])
        for metric in metrics
    }

    selected_dir_0 = [
        finite_float(row["selected_directional_coordinates"][0], "selected_dir_0")
        for row in rows
    ]
    selected_dir_1 = [
        finite_float(row["selected_directional_coordinates"][1], "selected_dir_1")
        for row in rows
    ]
    control_dir_0 = [
        finite_float(row["control_directional_coordinates"][0], "control_dir_0")
        for row in rows
    ]
    control_dir_1 = [
        finite_float(row["control_directional_coordinates"][1], "control_dir_1")
        for row in rows
    ]

    selected_cos = [
        None if row["cosine_gradient_selected_component"] is None
        else finite_float(
            row["cosine_gradient_selected_component"],
            "cosine_gradient_selected_component",
        )
        for row in rows
    ]
    control_cos = [
        None if row["cosine_gradient_control_component"] is None
        else finite_float(
            row["cosine_gradient_control_component"],
            "cosine_gradient_control_component",
        )
        for row in rows
    ]

    gradient_norms = np.asarray(
        [finite_float(row["gradient_full_l2"], "gradient_full_l2") for row in rows],
        dtype=np.float64,
    )
    pair_delta = [
        finite_float(row["Delta_L"], "pair_Delta_L")
        for row in pair_rows
    ]

    active_wrong_counts: dict[str, int] = {}
    for row in rows:
        key = str(int(row["active_wrong_class_id"]))
        active_wrong_counts[key] = active_wrong_counts.get(key, 0) + 1

    by_cell: dict[str, Any] = {}
    for cell in CELLS:
        cell_rows = [row for row in rows if str(row["contrast_cell_id"]) == cell]
        require(len(cell_rows) == N, f"CELL_ROWS:{cell}")
        by_cell[cell] = {
            "Delta_L_row": descriptive([
                finite_float(row["Delta_L_row"], f"Delta_L_row:{cell}")
                for row in cell_rows
            ]),
            "gradient_full_l2": descriptive([
                finite_float(row["gradient_full_l2"], f"gradient:{cell}")
                for row in cell_rows
            ]),
        }

    return {
        "pair_Delta_L": descriptive(pair_delta),
        "row_metrics": row_desc,
        "directional_coordinates": {
            "selected_plus": descriptive(selected_dir_0),
            "selected_minus": descriptive(selected_dir_1),
            "control_plus": descriptive(control_dir_0),
            "control_minus": descriptive(control_dir_1),
        },
        "cosines": {
            "selected_component": nullable_descriptive(selected_cos),
            "control_component": nullable_descriptive(control_cos),
        },
        "by_cell": by_cell,
        "zero_gradient_count": int(np.sum(gradient_norms == 0.0)),
        "wrong_class_exact_tie_count": 0,
        "nonfinite_count": 0,
        "active_wrong_class_id_counts": active_wrong_counts,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_rows = pair_values(rows)
    delta = [finite_float(row["Delta_L"], "pair_Delta_L") for row in pair_rows]
    primary = primary_inference(delta)
    supported = bool(primary["positive_alignment_supported"])
    result_label = RESULT_SUPPORTED if supported else RESULT_NOT_ESTABLISHED

    result = {
        "schema_version": "gen4-mamba130m-readout-alignment-static-analysis-v1",
        "result": ANALYSIS_RESULT,
        "scientific_conclusion": result_label,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_execution_head": RAW_EXECUTION_HEAD,
        "raw_item_sha256": EXPECTED_RAW_SHA256[ITEM_FILE],
        "population": "xg1_fact_2701..xg1_fact_3000",
        "pair_count": N,
        "cells": list(CELLS),
        "scale": "mamba130m",
        "primary": primary,
        "secondary_descriptive": secondary_descriptive(rows, pair_rows),
        "total_p_value_count": 1,
        "behavioral_merge_performed": False,
        "three_scale_synthesis_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "training_executed": False,
    }
    return result, pair_rows


def write_analysis(
    *,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows, _summary, _manifest = read_raw_bundle(raw_dir)
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    result, pair_rows = analyze_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / ANALYSIS_FILE).write_bytes(pretty_json_bytes(result))
    (output_dir / PAIR_FILE).write_bytes(jsonl_bytes(pair_rows))

    manifest = {
        "schema_version": "gen4-mamba130m-readout-alignment-analysis-manifest-v1",
        "result": ANALYSIS_RESULT,
        "scientific_conclusion": result["scientific_conclusion"],
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_item_sha256": EXPECTED_RAW_SHA256[ITEM_FILE],
        "analysis_sha256": sha256_file(output_dir / ANALYSIS_FILE),
        "pair_values_sha256": sha256_file(output_dir / PAIR_FILE),
        "pair_count": N,
        "primary_p_value_count": 1,
        "total_p_value_count": 1,
        "behavioral_merge_performed": False,
        "three_scale_synthesis_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "training_executed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (ANALYSIS_FILE, PAIR_FILE, MANIFEST_FILE)
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static analysis of frozen Mamba-130M readout-alignment "
            "raw evidence. Executes exactly one prespecified one-sided "
            "one-sample Student t-test on 300 pair-level Delta_L values."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    result = write_analysis(raw_dir=args.raw_dir, output_dir=args.output_dir)
    primary = result["primary"]

    print("RESULT=" + result["result"])
    print(f"MEAN_DELTA_L_130M={primary['mean']:.17g}")
    print(f"SD_SAMPLE={primary['sd_sample']:.17g}")
    print(f"T_STATISTIC={primary['t_statistic']:.17g}")
    print(f"P_VALUE={primary['p_value']:.17g}")
    print("POSITIVE_MEAN_SIGN_GATE=" + str(primary["positive_mean_sign_gate_pass"]))
    print("ALPHA_GATE=" + str(primary["alpha_gate_pass"]))
    print("PRIMARY_SUPPORTED=" + str(primary["positive_alignment_supported"]))
    print("PRIMARY_P_VALUE_COUNT=1")
    print("TOTAL_P_VALUE_COUNT=1")
    print("SCIENTIFIC_CONCLUSION=" + result["scientific_conclusion"])
    print("BEHAVIORAL_MERGE_PERFORMED=False")
    print("THREE_SCALE_SYNTHESIS_PERFORMED=False")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
