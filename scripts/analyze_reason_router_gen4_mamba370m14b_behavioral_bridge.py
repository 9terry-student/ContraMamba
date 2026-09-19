#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as runner,
)


N = 300
ALPHA = 0.05
SCALES = ("mamba370m", "mamba14b")
EXPECTED_CONDITIONS = runner.CONDITIONS
EXPECTED_CELLS = runner.TARGET_CELLS
SHARDS = runner.SHARDS
FORWARDS_PER_SHARD = runner.FORWARDS_PER_SHARD
TOTAL_FORWARDS_PER_SCALE = runner.TOTAL_FORWARD_BUDGET
ROW_FILE = runner.ROW_FILE
SUMMARY_FILE = runner.SUMMARY_FILE
CHECKSUM_FILE = runner.CHECKSUM_FILE

RESULT_PASS = "PASS_MAMBA370M14B_BEHAVIORAL_BRIDGE_ANALYSIS"
RESULT_CROSS_SCALE = "CROSS_SCALE_BEHAVIORAL_BRIDGE_THROUGH_1.4B_SUPPORTED"
RESULT_PARTIAL = "SCALE_SPECIFIC_BEHAVIORAL_BRIDGE_ONLY"
RESULT_NOT = "CROSS_SCALE_BEHAVIORAL_BRIDGE_NOT_ESTABLISHED"


class BehavioralAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralAnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        out.append(value)
    return out


def validate_checksums(shard_dir: Path) -> None:
    path = shard_dir / CHECKSUM_FILE
    require(path.is_file(), f"CHECKSUM_MISSING:{shard_dir}")
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        target = shard_dir / name
        require(target.is_file(), f"CHECKSUM_TARGET:{name}")
        require(
            sha256_file(target) == digest,
            f"CHECKSUM:{name}",
        )
        seen.add(name)
    require(
        seen == {ROW_FILE, SUMMARY_FILE},
        f"CHECKSUM_SET:{seen}",
    )


def validate_shard(
    run_dir: Path,
    *,
    scale: str,
    shard_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard = SHARDS[shard_id]
    shard_dir = run_dir / f"shard{shard_id}"
    validate_checksums(shard_dir)

    rows = read_jsonl(shard_dir / ROW_FILE)
    summary = json.loads(
        (shard_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    )
    spec = runner.scale_spec(scale)

    require(summary["result"] == runner.RESULT_PASS, "SUMMARY_RESULT")
    require(summary["scale"] == scale, "SUMMARY_SCALE")
    require(
        summary["checkpoint_sha256"]
        == spec["checkpoint_sha256"],
        "SUMMARY_CHECKPOINT",
    )
    require(
        summary["selected_dominant_candidate"]
        == spec["selected_plane"],
        "SUMMARY_SELECTED",
    )
    require(
        summary["response_blind_control_plane"]
        == spec["control_plane"],
        "SUMMARY_CONTROL",
    )
    require(
        summary["selection_uses_behavioral_response"] is False,
        "SUMMARY_SELECTION_RESPONSE",
    )
    require(
        int(summary["shard_index"]) == shard_id,
        "SUMMARY_SHARD",
    )
    require(
        int(summary["physical_device"])
        == int(shard["physical_device"]),
        "SUMMARY_PHYSICAL",
    )
    require(
        summary["population_first"] == shard["pair_first"]
        and summary["population_last"] == shard["pair_last"],
        "SUMMARY_RANGE",
    )
    require(
        int(summary["source_pair_count"]) == 150,
        "SUMMARY_N",
    )
    require(
        tuple(summary["target_cells"]) == EXPECTED_CELLS,
        "SUMMARY_CELLS",
    )
    require(
        tuple(summary["condition_order"]) == EXPECTED_CONDITIONS,
        "SUMMARY_CONDITIONS",
    )
    require(
        summary["label_contract"]
        == {
            "C0_SHAM": "SUPPORT",
            "C2_NAME": "NOT_ENTITLED",
        },
        "SUMMARY_LABELS",
    )
    require(
        int(summary["scientific_full_model_forward_count_this_run"])
        == FORWARDS_PER_SHARD,
        "SUMMARY_BUDGET",
    )
    require(
        summary["primary_inference_executed"] is False,
        "RAW_INFERENCE_BOUNDARY",
    )
    require(
        summary["scientific_conclusion"] is None,
        "RAW_CONCLUSION",
    )
    require(
        summary["confirmation_inference_accessed"] is False,
        "CONFIRMATION_INFERENCE_BOUNDARY",
    )
    require(
        summary["selection_reopened"] is False,
        "SELECTION_BOUNDARY",
    )
    require(
        summary["rescue_performed"] is False,
        "RESCUE_BOUNDARY",
    )
    require(
        summary["training_executed"] is False
        and summary["backward_executed"] is False,
        "TRAINING_BOUNDARY",
    )
    require(len(rows) == FORWARDS_PER_SHARD, "ROW_COUNT")

    expected_pairs = runner.PAIR_IDS[
        int(shard["start_index"]):
        int(shard["end_index"])
    ]
    expected = {
        (pair, cell, condition)
        for pair in expected_pairs
        for cell in EXPECTED_CELLS
        for condition in EXPECTED_CONDITIONS
    }
    observed: set[tuple[str, str, str]] = set()

    for row in rows:
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        )
        require(key not in observed, f"DUPLICATE_TUPLE:{key}")
        observed.add(key)
        require(row["scale"] == scale, f"ROW_SCALE:{key}")
        require(
            row["checkpoint_sha256"]
            == spec["checkpoint_sha256"],
            f"ROW_CHECKPOINT:{key}",
        )
        require(
            int(row["shard_index"]) == shard_id,
            f"ROW_SHARD:{key}",
        )
        require(
            int(row["physical_device"])
            == int(shard["physical_device"]),
            f"ROW_PHYSICAL:{key}",
        )
        require(
            int(row["scientific_full_model_forward_count"]) == 1,
            f"ROW_BUDGET:{key}",
        )
        require(
            math.isfinite(
                float(row["correct_class_logit_margin"])
            ),
            f"ROW_MARGIN:{key}",
        )
        require(
            len(row["final_logits"]) == 3,
            f"ROW_LOGITS:{key}",
        )

    require(observed == expected, "TUPLE_COVERAGE")
    return rows, summary


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def one_sample_greater(
    values: Sequence[float],
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(
        array.shape == (N,)
        and bool(np.isfinite(array).all()),
        "PRIMARY_VECTOR",
    )
    mean_value = float(np.mean(array))
    sd = float(np.std(array, ddof=1))

    require(
        sd > 0.0 and math.isfinite(sd),
        "PRIMARY_SD",
    )
    test = stats.ttest_1samp(
        array,
        popmean=0.0,
        alternative="greater",
    )
    t_statistic = float(test.statistic)
    p_raw = float(test.pvalue)

    require(
        math.isfinite(t_statistic)
        and 0.0 <= p_raw <= 1.0,
        "PRIMARY_TEST_NONFINITE",
    )
    return {
        "n": N,
        "df": N - 1,
        "mean": mean_value,
        "sd": sd,
        "t_statistic": t_statistic,
        "p_raw": p_raw,
        "fraction_positive": float(np.mean(array > 0.0)),
        "cohen_dz": mean_value / sd,
    }


def holm_adjust(
    p_by_scale: Mapping[str, float],
) -> dict[str, float]:
    require(set(p_by_scale) == set(SCALES), "HOLM_SCALE_SET")
    ordered = sorted(
        (
            (scale, float(value))
            for scale, value in p_by_scale.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    require(
        all(0.0 <= p <= 1.0 for _, p in ordered),
        "HOLM_P_RANGE",
    )

    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)

    for rank, (scale, p_raw) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_raw)
        running = max(running, candidate)
        adjusted[scale] = min(1.0, running)

    return adjusted


def analyze_scale(
    run_dir: Path,
    *,
    scale: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shard0, summary0 = validate_shard(
        run_dir,
        scale=scale,
        shard_id=0,
    )
    shard1, summary1 = validate_shard(
        run_dir,
        scale=scale,
        shard_id=1,
    )

    for key in (
        "execution_head",
        "scale",
        "checkpoint_sha256",
        "structural_source_sha256",
        "structural_rows_sha256",
        "selected_dominant_candidate",
        "response_blind_control_plane",
        "tokenizer",
    ):
        require(
            summary0[key] == summary1[key],
            f"SHARD_MISMATCH:{scale}:{key}",
        )

    rows = shard0 + shard1
    require(
        len(rows) == TOTAL_FORWARDS_PER_SCALE,
        f"TOTAL_ROW_COUNT:{scale}",
    )
    pair_ids = sorted(
        {str(row["source_pair_id"]) for row in rows}
    )
    require(
        pair_ids == list(runner.PAIR_IDS),
        f"PAIR_UNION:{scale}",
    )

    by_key = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    require(
        len(by_key) == TOTAL_FORWARDS_PER_SCALE,
        f"KEY_COUNT:{scale}",
    )

    pair_records: list[dict[str, Any]] = []
    accuracy: dict[str, list[float]] = {
        condition: []
        for condition in EXPECTED_CONDITIONS
    }
    flip_counts = {
        condition: 0
        for condition in EXPECTED_CONDITIONS
        if condition != "native"
    }
    flip_denominator = N * len(EXPECTED_CELLS)

    for pair in pair_ids:
        margins: dict[str, float] = {}
        q_values: dict[str, float] = {}

        for condition in EXPECTED_CONDITIONS:
            margins[condition] = mean([
                float(
                    by_key[
                        (pair, cell, condition)
                    ]["correct_class_logit_margin"]
                )
                for cell in EXPECTED_CELLS
            ])
            q_values[condition] = mean([
                float(
                    by_key[
                        (pair, cell, condition)
                    ]["q_authorized"]
                )
                for cell in EXPECTED_CELLS
            ])

            for cell in EXPECTED_CELLS:
                row = by_key[(pair, cell, condition)]
                accuracy[condition].append(
                    1.0 if bool(row["is_correct"]) else 0.0
                )
                if condition != "native":
                    if (
                        int(row["prediction_id"])
                        != int(
                            by_key[
                                (pair, cell, "native")
                            ]["prediction_id"]
                        )
                    ):
                        flip_counts[condition] += 1

        d_beh = (
            margins["dominant_restored"]
            - margins["dominant_control"]
        )
        d_nec = (
            margins["native"]
            - margins["dominant_neutralized"]
        )
        pair_records.append({
            "source_pair_id": pair,
            "M_native": margins["native"],
            "M_neutralized":
                margins["dominant_neutralized"],
            "M_restored":
                margins["dominant_restored"],
            "M_control":
                margins["dominant_control"],
            "D_BEH": d_beh,
            "D_NEC_BEH": d_nec,
            "q_authorized_restored_minus_control":
                q_values["dominant_restored"]
                - q_values["dominant_control"],
        })

    d_values = [
        float(record["D_BEH"])
        for record in pair_records
    ]
    primary = one_sample_greater(d_values)

    q_delta = np.asarray(
        [
            float(
                record[
                    "q_authorized_restored_minus_control"
                ]
            )
            for record in pair_records
        ],
        dtype=np.float64,
    )
    d_array = np.asarray(d_values, dtype=np.float64)
    if (
        float(np.std(q_delta)) == 0.0
        or float(np.std(d_array)) == 0.0
    ):
        q_corr = None
    else:
        q_corr = float(np.corrcoef(q_delta, d_array)[0, 1])

    cell_means = {}
    for cell in EXPECTED_CELLS:
        values = [
            float(
                by_key[
                    (pair, cell, "dominant_restored")
                ]["correct_class_logit_margin"]
            )
            - float(
                by_key[
                    (pair, cell, "dominant_control")
                ]["correct_class_logit_margin"]
            )
            for pair in pair_ids
        ]
        cell_means[cell] = mean(values)

    result = {
        "scale": scale,
        "execution_head": summary0["execution_head"],
        "checkpoint_sha256": summary0["checkpoint_sha256"],
        "selected_dominant_candidate":
            summary0["selected_dominant_candidate"],
        "response_blind_control_plane":
            summary0["response_blind_control_plane"],
        "population": "xg1_fact_4801..xg1_fact_5100",
        "N": N,
        "primary_endpoint":
            "D_BEH=M_dominant_restored-M_dominant_control",
        "primary_test": {
            "test": "one_sample_student_t",
            "alternative": "greater",
            "alpha_family": ALPHA,
            **primary,
        },
        "secondary_descriptive": {
            "mean_native_minus_neutralized_pair_margin":
                mean([
                    float(record["D_NEC_BEH"])
                    for record in pair_records
                ]),
            "accuracy_by_condition": {
                condition: mean(values)
                for condition, values in accuracy.items()
            },
            "prediction_flip_rate_vs_native": {
                condition:
                    float(count / flip_denominator)
                for condition, count in flip_counts.items()
            },
            "mean_D_BEH_by_cell": cell_means,
            "q_authorized_effect_vs_margin_effect_pearson":
                q_corr,
            "q_authorized_note": (
                "This is the model q_authorized diagnostic from the "
                "same full forward, not the recurrent-trajectory Q endpoint."
            ),
        },
        "scientific_full_model_forward_count":
            TOTAL_FORWARDS_PER_SCALE,
    }
    return result, pair_records


def analyze(
    mamba370m_run_dir: Path,
    mamba14b_run_dir: Path,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    pair_records: dict[str, list[dict[str, Any]]] = {}

    for scale, run_dir in (
        ("mamba370m", mamba370m_run_dir),
        ("mamba14b", mamba14b_run_dir),
    ):
        result, records = analyze_scale(
            run_dir,
            scale=scale,
        )
        results[scale] = result
        pair_records[scale] = records

    require(
        results["mamba370m"]["execution_head"]
        == results["mamba14b"]["execution_head"],
        "CROSS_SCALE_EXECUTION_HEAD",
    )
    require(
        [
            record["source_pair_id"]
            for record in pair_records["mamba370m"]
        ]
        == [
            record["source_pair_id"]
            for record in pair_records["mamba14b"]
        ],
        "CROSS_SCALE_PAIR_ALIGNMENT",
    )

    raw_p = {
        scale: float(results[scale]["primary_test"]["p_raw"])
        for scale in SCALES
    }
    adjusted = holm_adjust(raw_p)

    supported: dict[str, bool] = {}
    for scale in SCALES:
        results[scale]["primary_test"]["p_holm"] = adjusted[scale]
        gate = bool(
            float(results[scale]["primary_test"]["mean"]) > 0.0
            and adjusted[scale] < ALPHA
        )
        results[scale]["behavioral_bridge_supported_holm"] = gate
        supported[scale] = gate

    both = all(supported.values())
    any_supported = any(supported.values())
    if both:
        label = RESULT_CROSS_SCALE
    elif any_supported:
        label = RESULT_PARTIAL
    else:
        label = RESULT_NOT

    d370 = np.asarray(
        [
            float(record["D_BEH"])
            for record in pair_records["mamba370m"]
        ],
        dtype=np.float64,
    )
    d14 = np.asarray(
        [
            float(record["D_BEH"])
            for record in pair_records["mamba14b"]
        ],
        dtype=np.float64,
    )
    if float(np.std(d370)) == 0.0 or float(np.std(d14)) == 0.0:
        cross_corr = None
    else:
        cross_corr = float(np.corrcoef(d370, d14)[0, 1])

    return {
        "schema_version":
            "gen4-mamba370m14b-behavioral-bridge-analysis-v1",
        "result": RESULT_PASS,
        "execution_head":
            results["mamba370m"]["execution_head"],
        "population": "xg1_fact_4801..xg1_fact_5100",
        "shared_population_across_scales": True,
        "primary_family": {
            "scale_order": list(SCALES),
            "test": "one_sample_student_t",
            "alternative": "greater",
            "family_alpha": ALPHA,
            "multiplicity": "holm",
            "primary_p_value_count": 2,
            "historical_behavioral_p_values_in_family": 0,
        },
        "scale_results": results,
        "support_by_scale": supported,
        "cross_scale_behavioral_bridge_supported": both,
        "scientific_conclusion": label,
        "cross_scale_descriptive": {
            "paired_D_BEH_pearson": cross_corr,
            "mean_D_BEH_14b_minus_370m":
                float(np.mean(d14 - d370)),
            "p_value_added_for_scale_difference": False,
        },
        "scientific_full_model_forward_count_total":
            2 * TOTAL_FORWARDS_PER_SCALE,
        "selection_reopened": False,
        "scale_specific_rescue_cohort_used": False,
        "additional_primary_p_values_executed": False,
        "training_executed": False,
        "backward_executed": False,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the two pre-specified 370M/1.4B behavioral "
            "bridge tests and apply the exact two-test Holm family."
        )
    )
    parser.add_argument(
        "--mamba370m-run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mamba14b-run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(not args.output.exists(), "OUTPUT_COLLISION")
    result = analyze(
        args.mamba370m_run_dir,
        args.mamba14b_run_dir,
    )
    args.output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("RESULT=" + result["result"])
    for scale in SCALES:
        primary = result["scale_results"][scale]["primary_test"]
        print(
            scale.upper()
            + "_MEAN_D_BEH="
            + format(float(primary["mean"]), ".17g")
        )
        print(
            scale.upper()
            + "_P_RAW="
            + format(float(primary["p_raw"]), ".17g")
        )
        print(
            scale.upper()
            + "_P_HOLM="
            + format(float(primary["p_holm"]), ".17g")
        )
        print(
            scale.upper()
            + "_SUPPORTED_HOLM="
            + str(bool(result["support_by_scale"][scale]))
        )

    print("PRIMARY_P_VALUE_COUNT=2")
    print("MULTIPLICITY=holm")
    print(
        "CROSS_SCALE_BEHAVIORAL_BRIDGE_SUPPORTED="
        + str(
            bool(
                result[
                    "cross_scale_behavioral_bridge_supported"
                ]
            )
        )
    )
    print(
        "SCIENTIFIC_CONCLUSION="
        + str(result["scientific_conclusion"])
    )
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT_TOTAL=4800")
    print("SELECTION_REOPENED=False")
    print("SCALE_SPECIFIC_RESCUE_COHORT_USED=False")
    print("ADDITIONAL_PRIMARY_P_VALUES_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()
