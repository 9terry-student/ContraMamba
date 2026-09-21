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
RAW_FREEZE_COMMIT = "320b340d4b580beb2919daa0024cbf3c04ed65c1"
RAW_RUN_NAME = "g4k-lowdisp-behavioral-537e89b-r1-2t4"

RAW_DIR = (
    ROOT
    / "reports/reason_router_gen4_mamba370m14b_low_displacement_behavioral_raw_runs"
    / RAW_RUN_NAME
)
RAW_ITEM_FILE = "low_displacement_behavioral_items.jsonl"
RAW_SUMMARY_FILE = "raw_behavioral_summary.json"
RAW_MANIFEST_FILE = "artifact_manifest.json"
RAW_SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_RAW_SHA256 = {
    RAW_ITEM_FILE:
        "1d14bb251575021e79e0f72c33cd7141fad27a08750060a22fdd5d90a1012866",
    RAW_SUMMARY_FILE:
        "b6e22affd6e290ecfd8081ac82e655837d2b2ee6cb2b93bc979b7563ba04a6c3",
    RAW_MANIFEST_FILE:
        "02c893d1dfad27b41dd858028dfdf4986ca27ef835ae4342e2b2b0009fdeae65",
}

SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(5401, 5701))
N = 300
ALPHAS = (0.5, 0.25)
PRIMARY_ALPHA = 0.25
FAMILY_ALPHA = 0.05
CONDITION_BY_ALPHA = {
    0.5: "control_alpha_0_5",
    0.25: "control_alpha_0_25",
}
RESTORED_CONDITION = "restored_native"

RESULT_BOTH = "LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_PRESERVED_AT_ALPHA_0_25"
RESULT_PARTIAL = "LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_PARTIALLY_PRESERVED"
RESULT_NOT = "LOW_DISPLACEMENT_SCALE_SIGN_PATTERN_NOT_ESTABLISHED"
ANALYSIS_RESULT = "PASS_GEN4_LOW_DISPLACEMENT_BEHAVIORAL_STATIC_ANALYSIS"

PRIMARY_FILE = "primary_analysis.json"
PAIR_FILE = "pair_level_contrasts.jsonl"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"


class LowDisplacementBehaviorAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDisplacementBehaviorAnalysisError(message)


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
        raise LowDisplacementBehaviorAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def validate_raw_bundle(raw_dir: Path) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    expected_files = {
        RAW_ITEM_FILE,
        RAW_SUMMARY_FILE,
        RAW_MANIFEST_FILE,
        RAW_SUMS_FILE,
    }
    require(raw_dir.is_dir(), "RAW_DIR_MISSING")
    require(
        {p.name for p in raw_dir.iterdir() if p.is_file()} == expected_files,
        "RAW_FILE_SET",
    )
    for name, digest in EXPECTED_RAW_SHA256.items():
        require(sha256_file(raw_dir / name) == digest, f"RAW_SHA256:{name}")

    sums: dict[str, str] = {}
    for line in (raw_dir / RAW_SUMS_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        sums[name] = digest
    require(sums == EXPECTED_RAW_SHA256, "RAW_SUMS_CONTENT")

    summary = json.loads(
        (raw_dir / RAW_SUMMARY_FILE).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (raw_dir / RAW_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    require(
        summary["result"] == "PASS_GEN4_LOW_DISPLACEMENT_BEHAVIORAL_RAW",
        "RAW_SUMMARY_RESULT",
    )
    require(
        manifest["result"] == "PASS_GEN4_LOW_DISPLACEMENT_BEHAVIORAL_RAW",
        "RAW_MANIFEST_RESULT",
    )
    require(
        summary["execution_head"] == "537e89b4266ffef1b788832b5e00de411e5ff785",
        "RAW_EXECUTION_HEAD",
    )
    require(manifest["execution_head"] == summary["execution_head"], "RAW_HEAD_MISMATCH")
    require(summary["population"] == "xg1_fact_5401..xg1_fact_5700", "RAW_POPULATION")
    require(summary["pair_count"] == 300, "RAW_PAIR_COUNT")
    require(summary["cells"] == list(CELLS), "RAW_CELLS")
    require(summary["scales"] == list(SCALES), "RAW_SCALES")
    require(summary["behavioral_alphas"] == [0.5, 0.25], "RAW_ALPHAS")
    require(summary["primary_alpha_reserved_for_static_analysis"] == 0.25, "RAW_PRIMARY_ALPHA")
    require(summary["behavior_row_count"] == 3600, "RAW_ROW_COUNT")
    require(summary["p_value_count_executed"] == 0, "RAW_PVALUES")
    require(summary["primary_inference_executed"] is False, "RAW_INFERENCE")
    require(summary["alpha1_behavioral_forward_executed"] is False, "RAW_ALPHA1")
    require(summary["training_executed"] is False, "RAW_TRAINING")
    require(summary["backward_count"] == 0, "RAW_BACKWARD")

    rows = read_jsonl(raw_dir / RAW_ITEM_FILE)
    require(len(rows) == 3600, "RAW_ITEM_COUNT")
    return rows, summary, manifest


def descriptive(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all(), "DESCRIPTIVE")
    return {
        "mean": float(np.mean(a)),
        "sd_population": float(np.std(a, ddof=0)),
        "min": float(np.min(a)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
        "fraction_positive": float(np.mean(a > 0.0)),
        "fraction_negative": float(np.mean(a < 0.0)),
    }


def one_sample_directional(
    values: Sequence[float],
    *,
    alternative: str,
) -> dict[str, Any]:
    require(alternative in {"greater", "less"}, "PRIMARY_ALTERNATIVE")
    a = np.asarray(values, dtype=np.float64)
    require(a.shape == (N,) and np.isfinite(a).all(), "PRIMARY_VECTOR")
    sd = float(np.std(a, ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")
    test = stats.ttest_1samp(a, popmean=0.0, alternative=alternative)
    t = float(test.statistic)
    p = float(test.pvalue)
    require(math.isfinite(t) and 0.0 <= p <= 1.0, "PRIMARY_TEST")
    return {
        "n": N,
        "df": N - 1,
        "mean": float(np.mean(a)),
        "sd_sample": sd,
        "t_statistic": t,
        "p_raw": p,
        "alternative": alternative,
        "fraction_positive": float(np.mean(a > 0.0)),
        "fraction_negative": float(np.mean(a < 0.0)),
        "cohen_dz": float(np.mean(a)) / sd,
    }


def holm_adjust(p_by_scale: Mapping[str, float]) -> dict[str, float]:
    require(set(p_by_scale) == set(SCALES), "HOLM_SCALE_SET")
    ordered = sorted(
        ((scale, float(p)) for scale, p in p_by_scale.items()),
        key=lambda x: (x[1], x[0]),
    )
    require(all(0.0 <= p <= 1.0 for _, p in ordered), "HOLM_RANGE")
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (scale, p_raw) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_raw)
        running = max(running, candidate)
        adjusted[scale] = min(1.0, running)
    return adjusted


def build_pair_records(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["scale"]),
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        )
        require(key not in by_key, f"RAW_DUPLICATE:{key}")
        by_key[key] = row

    expected_keys = {
        (scale, pair, cell, condition)
        for scale in SCALES
        for pair in PAIR_IDS
        for cell in CELLS
        for condition in (
            RESTORED_CONDITION,
            CONDITION_BY_ALPHA[0.5],
            CONDITION_BY_ALPHA[0.25],
        )
    }
    require(set(by_key) == expected_keys, "RAW_KEY_COVERAGE")

    out: list[dict[str, Any]] = []
    for pair in PAIR_IDS:
        record: dict[str, Any] = {"source_pair_id": pair}
        for scale in SCALES:
            restored = float(np.mean([
                float(by_key[(scale, pair, cell, RESTORED_CONDITION)]
                      ["correct_class_logit_margin"])
                for cell in CELLS
            ]))
            record[f"{scale}_M_restored"] = restored
            for alpha in ALPHAS:
                condition = CONDITION_BY_ALPHA[alpha]
                control = float(np.mean([
                    float(by_key[(scale, pair, cell, condition)]
                          ["correct_class_logit_margin"])
                    for cell in CELLS
                ]))
                tag = "0_5" if alpha == 0.5 else "0_25"
                d = restored - control
                require(math.isfinite(control) and math.isfinite(d),
                        f"PAIR_NONFINITE:{scale}:{pair}:{alpha}")
                record[f"{scale}_M_control_alpha_{tag}"] = control
                record[f"{scale}_D_BEH_alpha_{tag}"] = d

        record["R_q_alpha_0_5"] = (
            float(record["mamba370m_D_BEH_alpha_0_5"])
            - float(record["mamba14b_D_BEH_alpha_0_5"])
        )
        record["R_q_alpha_0_25"] = (
            float(record["mamba370m_D_BEH_alpha_0_25"])
            - float(record["mamba14b_D_BEH_alpha_0_25"])
        )
        out.append(record)
    require(len(out) == N, "PAIR_RECORD_COUNT")
    return out


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_execution_head: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_records = build_pair_records(rows)

    vectors: dict[str, dict[float, np.ndarray]] = {
        scale: {} for scale in SCALES
    }
    for scale in SCALES:
        vectors[scale][0.5] = np.asarray(
            [r[f"{scale}_D_BEH_alpha_0_5"] for r in pair_records],
            dtype=np.float64,
        )
        vectors[scale][0.25] = np.asarray(
            [r[f"{scale}_D_BEH_alpha_0_25"] for r in pair_records],
            dtype=np.float64,
        )

    primary = {
        "mamba370m": one_sample_directional(
            vectors["mamba370m"][PRIMARY_ALPHA],
            alternative="greater",
        ),
        "mamba14b": one_sample_directional(
            vectors["mamba14b"][PRIMARY_ALPHA],
            alternative="less",
        ),
    }
    adjusted = holm_adjust({
        scale: float(primary[scale]["p_raw"])
        for scale in SCALES
    })

    support: dict[str, bool] = {}
    for scale in SCALES:
        primary[scale]["p_holm"] = adjusted[scale]
        sign_gate = (
            float(primary[scale]["mean"]) > 0.0
            if scale == "mamba370m"
            else float(primary[scale]["mean"]) < 0.0
        )
        primary[scale]["mandatory_sign_gate_pass"] = sign_gate
        primary[scale]["holm_alpha_pass"] = adjusted[scale] < FAMILY_ALPHA
        support[scale] = bool(sign_gate and adjusted[scale] < FAMILY_ALPHA)
        primary[scale]["primary_supported"] = support[scale]

    supported_count = sum(bool(v) for v in support.values())
    if supported_count == 2:
        conclusion = RESULT_BOTH
    elif supported_count == 1:
        conclusion = RESULT_PARTIAL
    else:
        conclusion = RESULT_NOT

    secondary = {
        scale: descriptive(vectors[scale][0.5])
        for scale in SCALES
    }

    cross_scale: dict[str, Any] = {}
    for alpha, tag in ((0.5, "alpha_0_5"), (0.25, "alpha_0_25")):
        r = vectors["mamba370m"][alpha] - vectors["mamba14b"][alpha]
        cross_scale[tag] = {
            "R_mean_370m_minus_14b":
                float(np.mean(vectors["mamba370m"][alpha]))
                - float(np.mean(vectors["mamba14b"][alpha])),
            "paired_R_q": descriptive(r),
            "p_value_added": False,
        }

    result = {
        "schema_version":
            "gen4-mamba370m14b-low-displacement-behavioral-static-analysis-v1",
        "result": ANALYSIS_RESULT,
        "raw_execution_head": raw_execution_head,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "population": "xg1_fact_5401..xg1_fact_5700",
        "pair_count": N,
        "cells": list(CELLS),
        "scales": list(SCALES),
        "primary_family": {
            "primary_alpha": PRIMARY_ALPHA,
            "family_alpha": FAMILY_ALPHA,
            "test": "one_sample_student_t",
            "multiplicity": "holm",
            "primary_p_value_count": 2,
            "mamba370m_alternative": "greater",
            "mamba14b_alternative": "less",
            "historical_p_values_in_family": 0,
        },
        "primary_by_scale": primary,
        "support_by_scale": support,
        "scientific_conclusion": conclusion,
        "alpha_0_5_secondary_descriptive": secondary,
        "cross_scale_descriptive": cross_scale,
        "alpha_0_5_p_value_count": 0,
        "cross_scale_p_value_count": 0,
        "total_p_value_count": 2,
        "alpha1_behavioral_endpoint_analyzed": False,
        "row_filtering_performed": False,
        "rescue_performed": False,
        "model_forward_count_this_analysis": 0,
        "backward_count_this_analysis": 0,
        "training_executed": False,
    }
    return result, pair_records


def write_analysis(
    *,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows, summary, _manifest = validate_raw_bundle(raw_dir)
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    result, pair_records = analyze_rows(
        rows,
        raw_execution_head=str(summary["execution_head"]),
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PRIMARY_FILE).write_bytes(pretty_json_bytes(result))
    pair_raw = jsonl_bytes(pair_records)
    (output_dir / PAIR_FILE).write_bytes(pair_raw)

    manifest = {
        "schema_version":
            "gen4-mamba370m14b-low-displacement-behavioral-analysis-manifest-v1",
        "result": ANALYSIS_RESULT,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_item_sha256": EXPECTED_RAW_SHA256[RAW_ITEM_FILE],
        "primary_analysis_sha256": sha256_file(output_dir / PRIMARY_FILE),
        "pair_level_contrasts_sha256": sha256_file(output_dir / PAIR_FILE),
        "pair_count": N,
        "primary_p_value_count": 2,
        "alpha_0_5_p_value_count": 0,
        "cross_scale_p_value_count": 0,
        "total_p_value_count": 2,
        "scientific_conclusion": result["scientific_conclusion"],
        "additional_model_forward_count": 0,
        "additional_backward_count": 0,
        "training_executed": False,
        "row_filtering_performed": False,
        "rescue_performed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (PRIMARY_FILE, PAIR_FILE, MANIFEST_FILE)
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
            "CPU-only static analysis of frozen Gen4 370M/1.4B "
            "low-displacement behavioral raw evidence. Executes exactly "
            "two primary p-values at alpha=0.25 with Holm correction."
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
    print("RESULT=" + result["result"])
    for scale in SCALES:
        p = result["primary_by_scale"][scale]
        label = scale.upper()
        print(f"{label}_ALPHA_0_25_MEAN={p['mean']:.17g}")
        print(f"{label}_T_STATISTIC={p['t_statistic']:.17g}")
        print(f"{label}_P_RAW={p['p_raw']:.17g}")
        print(f"{label}_P_HOLM={p['p_holm']:.17g}")
        print(f"{label}_SIGN_GATE={p['mandatory_sign_gate_pass']}")
        print(f"{label}_PRIMARY_SUPPORTED={p['primary_supported']}")
    print("PRIMARY_P_VALUE_COUNT=2")
    print("ALPHA_0_5_P_VALUE_COUNT=0")
    print("CROSS_SCALE_P_VALUE_COUNT=0")
    print("SCIENTIFIC_CONCLUSION=" + result["scientific_conclusion"])
    print("MODEL_FORWARD_COUNT_THIS_ANALYSIS=0")
    print("BACKWARD_COUNT_THIS_ANALYSIS=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
