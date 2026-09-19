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
    reason_router_gen4_averitec_130m370m_external_transfer_fast_cuda
    as runner,
)


SCALES = runner.SCALE_ORDER
CONDITIONS = runner.CONDITIONS
N = runner.N
ALPHA = 0.05
RESULT_PASS = "PASS_AVERITEC_GOLD_EVIDENCE_EXTERNAL_TRANSFER_ANALYSIS"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL:{line_no}")
        rows.append(value)
    return rows


def validate_checksums(scale_dir: Path) -> None:
    sums = scale_dir / runner.CHECKSUM_FILE
    require(sums.is_file(), "SUMS_MISSING")
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        expected[name] = digest
    require(
        set(expected) == {runner.ROW_FILE, runner.SUMMARY_FILE},
        "SUM_SET",
    )
    for name, digest in expected.items():
        require(
            sha256_file(scale_dir / name) == digest,
            f"SUM_SHA:{name}",
        )


def validate_scale(
    scale_dir: Path,
    *,
    scale: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_checksums(scale_dir)
    rows = read_jsonl(scale_dir / runner.ROW_FILE)
    summary = json.loads(
        (scale_dir / runner.SUMMARY_FILE).read_text(encoding="utf-8")
    )

    require(summary["result"] == runner.RESULT_PASS, "SUMMARY_RESULT")
    require(summary["scale"] == scale, "SUMMARY_SCALE")
    require(summary["compatible_item_count"] == N, "SUMMARY_N")
    require(tuple(summary["condition_order"]) == CONDITIONS, "SUMMARY_CONDITIONS")
    require(
        summary["scientific_full_model_forward_count_this_run"]
        == runner.FORWARDS_PER_SCALE,
        "SUMMARY_FORWARDS",
    )
    require(summary["primary_inference_executed"] is False, "RAW_INFERENCE")
    require(summary["p_value_count_added"] == 0, "RAW_P")
    require(summary["scientific_conclusion"] is None, "RAW_CONCLUSION")
    require(summary["selection_reopened"] is False, "RAW_SELECTION")
    require(summary["rescue_performed"] is False, "RAW_RESCUE")
    require(summary["training_executed"] is False, "RAW_TRAIN")
    require(summary["backward_executed"] is False, "RAW_BACKWARD")
    require(summary["mamba14b_in_family"] is False, "RAW_14B")
    require(summary["cohort_sha256"] == runner.COHORT_SHA256, "SUMMARY_COHORT")
    require(
        summary["token_gate_manifest_sha256"] == runner.MANIFEST_SHA256,
        "SUMMARY_MANIFEST",
    )

    if scale == "mamba130m":
        require(
            summary["checkpoint_sha256"] == runner.MAMBA130_CHECKPOINT_SHA256,
            "M130_CKPT",
        )
        require(
            summary["selected_dominant_candidate"] == "P3"
            and summary["response_blind_control_plane"] == "P5"
            and summary["intervention_layer"] == 17,
            "M130_CAUSAL_OBJECT",
        )
    else:
        spec = runner.bridge_scale.scale_spec("mamba370m")
        require(
            summary["checkpoint_sha256"] == spec["checkpoint_sha256"],
            "M370_CKPT",
        )
        require(
            summary["selected_dominant_candidate"] == "P3"
            and summary["response_blind_control_plane"] == "P5"
            and summary["intervention_layer"] == 35,
            "M370_CAUSAL_OBJECT",
        )

    require(len(rows) == runner.FORWARDS_PER_SCALE, "ROW_COUNT")

    observed: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["example_id"]), str(row["condition"]))
        require(key not in observed, f"DUP:{key}")
        observed.add(key)
        require(row["scale"] == scale, f"ROW_SCALE:{key}")
        require(row["condition"] in CONDITIONS, f"ROW_CONDITION:{key}")
        require(row["anchor_name"] == runner.ANCHOR_NAME, f"ROW_ANCHOR:{key}")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + runner.TARGET_OFFSET,
            f"ROW_TARGET:{key}",
        )
        require(len(row["final_logits"]) == 3, f"ROW_LOGITS:{key}")
        require(
            math.isfinite(float(row["correct_class_logit_margin"])),
            f"ROW_MARGIN:{key}",
        )
        require(int(row["scientific_full_model_forward_count"]) == 1, f"ROW_FWD:{key}")

    expected = {
        (str(row["example_id"]), condition)
        for row in runner.read_jsonl(runner.ROOT / runner.COHORT_FILE)
        for condition in CONDITIONS
    }
    require(observed == expected, "ROW_COVERAGE")
    return rows, summary


def one_sample_greater(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (N,), "PRIMARY_SHAPE")
    require(bool(np.isfinite(array).all()), "PRIMARY_FINITE")
    sd = float(np.std(array, ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")

    test = stats.ttest_1samp(
        array,
        popmean=0.0,
        alternative="greater",
    )
    mean_value = float(np.mean(array))
    t_value = float(test.statistic)
    p_value = float(test.pvalue)
    require(math.isfinite(t_value), "T_NONFINITE")
    require(0.0 <= p_value <= 1.0, "P_RANGE")

    return {
        "n": N,
        "df": N - 1,
        "mean": mean_value,
        "sd": sd,
        "t_statistic": t_value,
        "p_raw": p_value,
        "cohen_dz": mean_value / sd,
        "fraction_positive": float(np.mean(array > 0.0)),
    }


def holm_adjust(p_by_scale: Mapping[str, float]) -> dict[str, float]:
    require(set(p_by_scale) == set(SCALES), "HOLM_SCALES")
    ordered = sorted(
        ((scale, float(p)) for scale, p in p_by_scale.items()),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (scale, p_raw) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_raw)
        running = max(running, candidate)
        adjusted[scale] = min(1.0, running)
    return adjusted


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or float(np.std(xa)) == 0.0 or float(np.std(ya)) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def analyze_scale(
    scale_dir: Path,
    *,
    scale: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, summary = validate_scale(scale_dir, scale=scale)
    by = {
        (str(row["example_id"]), str(row["condition"])): row
        for row in rows
    }
    cohort = runner.read_jsonl(runner.ROOT / runner.COHORT_FILE)
    item_ids = [str(row["example_id"]) for row in cohort]
    require(len(item_ids) == N and len(set(item_ids)) == N, "ITEM_IDS")

    d_ext: list[float] = []
    a_sel: list[float] = []
    b_ctrl: list[float] = []
    dq: list[float] = []
    records: list[dict[str, Any]] = []

    accuracy = {condition: [] for condition in CONDITIONS}
    flips = {
        "dominant_neutralized": 0,
        "dominant_control": 0,
    }
    label_d: dict[str, list[float]] = {
        "Refuted": [],
        "Not Enough Evidence": [],
        "Supported": [],
    }

    for item_id in item_ids:
        native = by[(item_id, "native")]
        neutral = by[(item_id, "dominant_neutralized")]
        control = by[(item_id, "dominant_control")]

        m_native = float(native["correct_class_logit_margin"])
        m_neutral = float(neutral["correct_class_logit_margin"])
        m_control = float(control["correct_class_logit_margin"])

        d = m_native - m_control
        a = m_native - m_neutral
        b = m_control - m_neutral
        require(abs((a - b) - d) <= 1.0e-12, f"DECOMPOSITION:{item_id}")

        q_delta = float(native["q_authorized"]) - float(control["q_authorized"])

        d_ext.append(d)
        a_sel.append(a)
        b_ctrl.append(b)
        dq.append(q_delta)

        source_label = str(native["source_label"])
        label_d[source_label].append(d)

        for condition in CONDITIONS:
            accuracy[condition].append(
                float(bool(by[(item_id, condition)]["is_correct"]))
            )
        for condition in ("dominant_neutralized", "dominant_control"):
            if (
                int(by[(item_id, condition)]["prediction_id"])
                != int(native["prediction_id"])
            ):
                flips[condition] += 1

        records.append({
            "example_id": item_id,
            "averitec_dev_index": int(native["averitec_dev_index"]),
            "source_label": source_label,
            "correct_label": str(native["correct_label"]),
            "M_native": m_native,
            "M_neutralized": m_neutral,
            "M_control": m_control,
            "A_sel": a,
            "B_ctrl": b,
            "D_EXT": d,
            "delta_q_authorized_native_minus_control": q_delta,
        })

    primary = one_sample_greater(d_ext)

    return {
        "scale": scale,
        "execution_head": summary["execution_head"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "primary_endpoint": "D_EXT=M_native-M_control",
        "primary": primary,
        "secondary_descriptive": {
            "mean_A_sel_native_minus_neutralized": float(np.mean(a_sel)),
            "mean_B_ctrl_control_minus_neutralized": float(np.mean(b_ctrl)),
            "accuracy_by_condition": {
                condition: float(np.mean(values))
                for condition, values in accuracy.items()
            },
            "prediction_flip_rate_vs_native": {
                condition: flips[condition] / N
                for condition in flips
            },
            "mean_D_EXT_by_source_label": {
                label: float(np.mean(values))
                for label, values in label_d.items()
            },
            "q_authorized_effect_vs_margin_effect_pearson":
                pearson(dq, d_ext),
        },
        "p_value_count": 1,
    }, records


def analyze(
    *,
    mamba130m_dir: Path,
    mamba370m_dir: Path,
) -> dict[str, Any]:
    scale_results: dict[str, Any] = {}
    item_records: dict[str, list[dict[str, Any]]] = {}

    result130, records130 = analyze_scale(
        mamba130m_dir,
        scale="mamba130m",
    )
    result370, records370 = analyze_scale(
        mamba370m_dir,
        scale="mamba370m",
    )
    scale_results["mamba130m"] = result130
    scale_results["mamba370m"] = result370
    item_records["mamba130m"] = records130
    item_records["mamba370m"] = records370

    p_raw = {
        scale: float(scale_results[scale]["primary"]["p_raw"])
        for scale in SCALES
    }
    p_holm = holm_adjust(p_raw)

    supported: dict[str, bool] = {}
    for scale in SCALES:
        scale_results[scale]["primary"]["p_holm"] = p_holm[scale]
        scale_results[scale]["primary"]["supported_holm"] = (
            float(scale_results[scale]["primary"]["mean"]) > 0.0
            and p_holm[scale] < ALPHA
        )
        supported[scale] = bool(
            scale_results[scale]["primary"]["supported_holm"]
        )

    support_count = sum(supported.values())
    if support_count == 2:
        conclusion = (
            "CROSS_SCALE_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_"
            "THROUGH_370M_SUPPORTED"
        )
    elif support_count == 1:
        conclusion = (
            "SCALE_SPECIFIC_AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_ONLY"
        )
    else:
        conclusion = (
            "AVERITEC_GOLD_EVIDENCE_CAUSAL_TRANSFER_NOT_ESTABLISHED"
        )

    return {
        "schema_version":
            "gen4-averitec-gold-evidence-external-transfer-analysis-v1",
        "result": RESULT_PASS,
        "population": "AVeriTeC official dev compatible 3-label gold-evidence cohort",
        "N_per_scale": N,
        "primary_scale_order": list(SCALES),
        "primary_endpoint": "D_EXT=M_native-M_control",
        "test": "one-sided one-sample Student t-test, alternative=greater",
        "family_alpha": ALPHA,
        "multiplicity": "Holm across exactly two scale-level primary p-values",
        "primary_p_value_count": 2,
        "historical_behavioral_p_values_in_family": 0,
        "mamba14b_in_family": False,
        "scale_results": scale_results,
        "supported_by_scale": supported,
        "scientific_conclusion": conclusion,
        "selection_reopened": False,
        "rescue_performed": False,
        "additional_primary_p_values": False,
        "training_executed": False,
        "backward_executed": False,
        "scientific_full_model_forward_count": runner.TOTAL_FORWARD_BUDGET,
        "item_records": item_records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba130m-dir", type=Path, required=True)
    parser.add_argument("--mamba370m-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(
        mamba130m_dir=args.mamba130m_dir,
        mamba370m_dir=args.mamba370m_dir,
    )
    require(not args.output.exists(), "OUTPUT_COLLISION")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("RESULT=" + result["result"])
    for scale in SCALES:
        primary = result["scale_results"][scale]["primary"]
        print(
            f"{scale.upper()}_MEAN_D_EXT="
            + format(float(primary["mean"]), ".17g")
        )
        print(
            f"{scale.upper()}_T="
            + format(float(primary["t_statistic"]), ".17g")
        )
        print(
            f"{scale.upper()}_P_RAW="
            + format(float(primary["p_raw"]), ".17g")
        )
        print(
            f"{scale.upper()}_P_HOLM="
            + format(float(primary["p_holm"]), ".17g")
        )
        print(
            f"{scale.upper()}_SUPPORTED_HOLM="
            + str(bool(primary["supported_holm"]))
        )
    print("PRIMARY_P_VALUE_COUNT=2")
    print("MULTIPLICITY=Holm")
    print("MAMBA14B_IN_FAMILY=False")
    print("SCIENTIFIC_CONCLUSION=" + result["scientific_conclusion"])
    print("SELECTION_REOPENED=False")
    print("RESCUE_PERFORMED=False")
    print("ADDITIONAL_PRIMARY_P_VALUES=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print(
        "SCIENTIFIC_FULL_MODEL_FORWARD_COUNT="
        + str(result["scientific_full_model_forward_count"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
