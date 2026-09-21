#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


N = 300
SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_restored",
    "dominant_control",
)
EXPECTED_PAIRS = tuple(f"xg1_fact_{i}" for i in range(4801, 5101))

READOUT_PAIR_FILE = "readout_alignment_pair_values.jsonl"
READOUT_ANALYSIS_FILE = "readout_alignment_analysis.json"
READOUT_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

BEHAVIOR_ROW_FILE = "behavioral_rows.jsonl"
BEHAVIOR_SUMMARY_FILE = "shard_summary.json"

RESULT = "PASS_GEN4_READOUT_BEHAVIOR_PAIR_LEVEL_DESCRIPTIVE_MERGE"
SCHEMA_VERSION = "gen4-readout-behavior-pair-level-descriptive-merge-v1"
PLAN_FREEZE_COMMIT = "c24ce5f23c1c4035078fa74aa5d8dcdb7c4682d6"

EXPECTED_SELECTED = {"mamba370m": "P3", "mamba14b": "P5"}
EXPECTED_CONTROL = {"mamba370m": "P5", "mamba14b": "P4"}
EXPECTED_CHECKPOINT = {
    "mamba370m": "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
    "mamba14b": "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a",
}


class PairMergeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PairMergeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(row)
    return rows


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


def validate_sums(
    root: Path,
    *,
    expected_names: set[str],
) -> dict[str, str]:
    sums_path = root / SUMS_FILE
    require(sums_path.is_file(), f"SUMS_MISSING:{root}")
    seen: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in seen, f"SUM_DUPLICATE:{root}:{name}")
        target = root / name
        require(target.is_file(), f"SUM_TARGET_MISSING:{root}:{name}")
        require(sha256_file(target) == digest, f"SUM_MISMATCH:{root}:{name}")
        seen[name] = digest
    require(set(seen) == expected_names, f"SUM_SET:{root}:{set(seen)}")
    return seen


def read_readout_pairs(readout_dir: Path) -> list[dict[str, Any]]:
    validate_sums(
        readout_dir,
        expected_names={
            READOUT_PAIR_FILE,
            READOUT_ANALYSIS_FILE,
            READOUT_MANIFEST_FILE,
        },
    )
    manifest = json.loads(
        (readout_dir / READOUT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    require(int(manifest["primary_p_value_count"]) == 1, "READOUT_PRIMARY_P_COUNT")
    require(manifest["rescue_performed"] is False, "READOUT_RESCUE")
    require(manifest["row_filter_performed"] is False, "READOUT_ROW_FILTER")

    rows = jsonl_rows(readout_dir / READOUT_PAIR_FILE)
    require(len(rows) == N, "READOUT_PAIR_COUNT")

    expected = list(EXPECTED_PAIRS)
    observed = [str(row["source_pair_id"]) for row in rows]
    require(observed == expected, "READOUT_PAIR_ORDER")
    require(len(set(observed)) == N, "READOUT_PAIR_DUPLICATE")

    for row in rows:
        d370 = float(row["Delta_L_370M"])
        d14 = float(row["Delta_L_1.4B"])
        r = float(row["R"])
        require(
            all(math.isfinite(x) for x in (d370, d14, r)),
            f"READOUT_NONFINITE:{row['source_pair_id']}",
        )
        require(
            math.isclose(r, d370 - d14, rel_tol=0.0, abs_tol=1e-15),
            f"READOUT_R_IDENTITY:{row['source_pair_id']}",
        )
    return rows


def shard_expected_pairs(shard_id: int) -> tuple[str, ...]:
    require(shard_id in (0, 1), f"SHARD_ID:{shard_id}")
    if shard_id == 0:
        return tuple(f"xg1_fact_{i}" for i in range(4801, 4951))
    return tuple(f"xg1_fact_{i}" for i in range(4951, 5101))


def read_behavior_shard(
    shard_dir: Path,
    *,
    scale: str,
    shard_id: int,
) -> list[dict[str, Any]]:
    require(scale in SCALES, f"SCALE:{scale}")
    validate_sums(
        shard_dir,
        expected_names={BEHAVIOR_ROW_FILE, BEHAVIOR_SUMMARY_FILE},
    )

    summary = json.loads(
        (shard_dir / BEHAVIOR_SUMMARY_FILE).read_text(encoding="utf-8")
    )
    require(summary["result"] == "PASS_RAW_CROSS_SCALE_BEHAVIORAL_SHARD", "BEH_SUMMARY_RESULT")
    require(summary["scale"] == scale, "BEH_SUMMARY_SCALE")
    require(int(summary["shard_index"]) == shard_id, "BEH_SUMMARY_SHARD")
    require(int(summary["source_pair_count"]) == 150, "BEH_SUMMARY_PAIR_COUNT")
    require(tuple(summary["target_cells"]) == CELLS, "BEH_SUMMARY_CELLS")
    require(tuple(summary["condition_order"]) == CONDITIONS, "BEH_SUMMARY_CONDITIONS")
    require(summary["primary_inference_executed"] is False, "BEH_RAW_INFERENCE")
    require(summary["scientific_conclusion"] is None, "BEH_RAW_CONCLUSION")
    require(summary["selection_reopened"] is False, "BEH_SELECTION")
    require(summary["rescue_performed"] is False, "BEH_RESCUE")
    require(summary["training_executed"] is False, "BEH_TRAINING")
    require(summary["backward_executed"] is False, "BEH_BACKWARD")

    rows = jsonl_rows(shard_dir / BEHAVIOR_ROW_FILE)
    expected_pairs = shard_expected_pairs(shard_id)
    expected_tuples = {
        (pair, cell, condition)
        for pair in expected_pairs
        for cell in CELLS
        for condition in CONDITIONS
    }
    require(len(rows) == len(expected_tuples), f"BEH_ROW_COUNT:{scale}:{shard_id}")

    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        )
        require(key not in seen, f"BEH_DUPLICATE:{scale}:{shard_id}:{key}")
        seen.add(key)
        require(row["scale"] == scale, f"BEH_SCALE:{key}")
        require(int(row["shard_index"]) == shard_id, f"BEH_SHARD:{key}")
        require(
            row["selected_dominant_candidate"] == EXPECTED_SELECTED[scale],
            f"BEH_SELECTED:{key}",
        )
        require(
            row["response_blind_control_plane"] == EXPECTED_CONTROL[scale],
            f"BEH_CONTROL:{key}",
        )
        require(
            row["checkpoint_sha256"] == EXPECTED_CHECKPOINT[scale],
            f"BEH_CHECKPOINT:{key}",
        )
        margin = float(row["correct_class_logit_margin"])
        require(math.isfinite(margin), f"BEH_MARGIN_NONFINITE:{key}")

    require(seen == expected_tuples, f"BEH_TUPLE_COVERAGE:{scale}:{shard_id}")
    return rows


def reconstruct_behavior_pairs(
    behavior_run_dir: Path,
    *,
    scale: str,
) -> list[dict[str, Any]]:
    rows = (
        read_behavior_shard(
            behavior_run_dir / scale / "shard0",
            scale=scale,
            shard_id=0,
        )
        + read_behavior_shard(
            behavior_run_dir / scale / "shard1",
            scale=scale,
            shard_id=1,
        )
    )
    by_key = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    require(len(by_key) == N * len(CELLS) * len(CONDITIONS), f"BEH_KEY_COUNT:{scale}")

    out: list[dict[str, Any]] = []
    for pair in EXPECTED_PAIRS:
        restored = np.asarray(
            [
                float(
                    by_key[
                        (pair, cell, "dominant_restored")
                    ]["correct_class_logit_margin"]
                )
                for cell in CELLS
            ],
            dtype=np.float64,
        )
        control = np.asarray(
            [
                float(
                    by_key[
                        (pair, cell, "dominant_control")
                    ]["correct_class_logit_margin"]
                )
                for cell in CELLS
            ],
            dtype=np.float64,
        )
        d_beh = float(restored.mean() - control.mean())
        require(math.isfinite(d_beh), f"D_BEH_NONFINITE:{scale}:{pair}")
        out.append(
            {
                "source_pair_id": pair,
                "D_BEH": d_beh,
            }
        )

    require(
        [row["source_pair_id"] for row in out] == list(EXPECTED_PAIRS),
        f"BEH_PAIR_ORDER:{scale}",
    )
    return out


def sign_code(x: float) -> int:
    require(math.isfinite(x), "SIGN_NONFINITE")
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def sign_category(delta_l: float, d_beh: float) -> str:
    a = sign_code(delta_l)
    b = sign_code(d_beh)
    labels = {
        (1, 1): "both_positive",
        (-1, -1): "both_negative",
        (1, -1): "delta_positive_beh_negative",
        (-1, 1): "delta_negative_beh_positive",
        (0, 0): "both_zero",
        (0, 1): "delta_zero_beh_positive",
        (0, -1): "delta_zero_beh_negative",
        (1, 0): "delta_positive_beh_zero",
        (-1, 0): "delta_negative_beh_zero",
    }
    return labels[(a, b)]


def descriptive(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and x.size > 0 and bool(np.isfinite(x).all()), "DESC_INPUT")
    return {
        "mean": float(x.mean()),
        "sd_population": float(x.std(ddof=0)),
        "min": float(x.min()),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(x.max()),
    }


def finite_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    require(x.shape == y.shape == (N,), "CORR_SHAPE")
    require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), "CORR_NONFINITE")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    require(math.isfinite(value), "CORR_RESULT")
    return value


def spearman_without_p(x: np.ndarray, y: np.ndarray) -> float | None:
    rx = np.asarray(stats.rankdata(x, method="average"), dtype=np.float64)
    ry = np.asarray(stats.rankdata(y, method="average"), dtype=np.float64)
    return finite_correlation(rx, ry)


def summarize_scale(
    delta_l: Sequence[float],
    d_beh: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(delta_l, dtype=np.float64)
    y = np.asarray(d_beh, dtype=np.float64)
    require(x.shape == y.shape == (N,), "SCALE_VECTOR_SHAPE")
    require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), "SCALE_NONFINITE")
    residual = y - x

    categories = {
        "both_positive": 0,
        "both_negative": 0,
        "delta_positive_beh_negative": 0,
        "delta_negative_beh_positive": 0,
        "both_zero": 0,
        "delta_zero_beh_positive": 0,
        "delta_zero_beh_negative": 0,
        "delta_positive_beh_zero": 0,
        "delta_negative_beh_zero": 0,
    }
    agree = 0
    for a, b in zip(x.tolist(), y.tolist(), strict=True):
        cat = sign_category(float(a), float(b))
        categories[cat] += 1
        if sign_code(float(a)) == sign_code(float(b)):
            agree += 1

    require(sum(categories.values()) == N, "SIGN_CATEGORY_TOTAL")
    return {
        "pearson_Delta_L_vs_D_BEH": finite_correlation(x, y),
        "spearman_Delta_L_vs_D_BEH": spearman_without_p(x, y),
        "Delta_L": descriptive(x),
        "D_BEH": descriptive(y),
        "residual_D_BEH_minus_Delta_L": descriptive(residual),
        "sign_agreement": {
            "fraction": float(agree / N),
            "count": int(agree),
            "categories": categories,
        },
    }


def merge_and_analyze(
    readout_dir: Path,
    behavior_run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    readout = read_readout_pairs(readout_dir)
    behavior = {
        scale: reconstruct_behavior_pairs(behavior_run_dir, scale=scale)
        for scale in SCALES
    }
    for scale in SCALES:
        require(
            [row["source_pair_id"] for row in behavior[scale]] == list(EXPECTED_PAIRS),
            f"BEH_PAIR_ORDER_FINAL:{scale}",
        )

    merged: list[dict[str, Any]] = []
    for idx, pair in enumerate(EXPECTED_PAIRS):
        r = readout[idx]
        require(str(r["source_pair_id"]) == pair, f"READOUT_PAIR_ALIGN:{pair}")
        d370 = float(r["Delta_L_370M"])
        d14 = float(r["Delta_L_1.4B"])
        b370 = float(behavior["mamba370m"][idx]["D_BEH"])
        b14 = float(behavior["mamba14b"][idx]["D_BEH"])
        merged.append(
            {
                "source_pair_id": pair,
                "Delta_L_370M": d370,
                "D_BEH_370M": b370,
                "residual_370M": float(b370 - d370),
                "sign_agree_370M": sign_code(d370) == sign_code(b370),
                "Delta_L_1.4B": d14,
                "D_BEH_1.4B": b14,
                "residual_1.4B": float(b14 - d14),
                "sign_agree_1.4B": sign_code(d14) == sign_code(b14),
            }
        )

    scale_results = {
        "mamba370m": summarize_scale(
            [float(row["Delta_L_370M"]) for row in merged],
            [float(row["D_BEH_370M"]) for row in merged],
        ),
        "mamba14b": summarize_scale(
            [float(row["Delta_L_1.4B"]) for row in merged],
            [float(row["D_BEH_1.4B"]) for row in merged],
        ),
    }

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "population": "xg1_fact_4801..xg1_fact_5100",
        "source_pair_count": N,
        "scales": list(SCALES),
        "primary_p_value_count": 0,
        "secondary_p_value_count": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "scale_results": scale_results,
        "interpretation_threshold_applied": False,
        "scientific_conclusion": None,
    }
    return analysis, merged


def write_outputs(
    *,
    readout_dir: Path,
    behavior_run_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    analysis, merged = merge_and_analyze(readout_dir, behavior_run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    merge_raw = b"".join(canonical_json_bytes(row) for row in merged)
    (output_dir / "pair_level_merge.jsonl").write_bytes(merge_raw)
    (output_dir / "descriptive_analysis.json").write_bytes(
        pretty_json_bytes(analysis)
    )

    input_sha256 = {
        "readout_pair_values": sha256_file(readout_dir / READOUT_PAIR_FILE),
        "mamba370m_shard0_behavioral_rows": sha256_file(
            behavior_run_dir / "mamba370m" / "shard0" / BEHAVIOR_ROW_FILE
        ),
        "mamba370m_shard1_behavioral_rows": sha256_file(
            behavior_run_dir / "mamba370m" / "shard1" / BEHAVIOR_ROW_FILE
        ),
        "mamba14b_shard0_behavioral_rows": sha256_file(
            behavior_run_dir / "mamba14b" / "shard0" / BEHAVIOR_ROW_FILE
        ),
        "mamba14b_shard1_behavioral_rows": sha256_file(
            behavior_run_dir / "mamba14b" / "shard1" / BEHAVIOR_ROW_FILE
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "plan_freeze_commit": PLAN_FREEZE_COMMIT,
        "primary_p_value_count": 0,
        "secondary_p_value_count": 0,
        "inference_performed": False,
        "training_executed": False,
        "forward_executed": False,
        "backward_executed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "source_pair_count": N,
        "scales": list(SCALES),
        "input_sha256": input_sha256,
        "scientific_conclusion": None,
    }
    (output_dir / "artifact_manifest.json").write_bytes(
        pretty_json_bytes(manifest)
    )

    names = (
        "pair_level_merge.jsonl",
        "descriptive_analysis.json",
        "artifact_manifest.json",
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only descriptive merge of frozen Study-B pair-level Delta_L "
            "with frozen Experiment-1 pair-level D_BEH. Adds no p-values."
        )
    )
    parser.add_argument("--readout-analysis-dir", type=Path, required=True)
    parser.add_argument("--behavioral-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = write_outputs(
        readout_dir=args.readout_analysis_dir,
        behavior_run_dir=args.behavioral_run_dir,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(analysis["result"]))
    for scale in SCALES:
        result = analysis["scale_results"][scale]
        label = scale.upper()
        print(
            label
            + "_PEARSON_DELTA_L_VS_D_BEH="
            + repr(result["pearson_Delta_L_vs_D_BEH"])
        )
        print(
            label
            + "_SPEARMAN_DELTA_L_VS_D_BEH="
            + repr(result["spearman_Delta_L_vs_D_BEH"])
        )
        print(
            label
            + "_SIGN_AGREEMENT="
            + repr(result["sign_agreement"]["fraction"])
        )
        print(
            label
            + "_RESIDUAL_MEAN="
            + repr(result["residual_D_BEH_minus_Delta_L"]["mean"])
        )
    print("PRIMARY_P_VALUE_COUNT=0")
    print("SECONDARY_P_VALUE_COUNT=0")
    print("INFERENCE_PERFORMED=False")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("FORWARD_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
