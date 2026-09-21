#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

RAW_FREEZE_COMMIT = "59c24ae696700dd722b8e1ca628fe495cfa65738"
PLAN_SHA256 = "11cae5cce793f3f27a3d8ec449f3dde1deab6aa162e37c97d6deaf9e04d6c845"

RAW_DIR = Path(
    "reports/reason_router_gen4_averitec_mamba14b_negative_sign_raw_runs/"
    "g4k-averitec14b-negtransfer-b6587da-r2-2t4"
)
RAW_SUMS_FILE = RAW_DIR / "SHA256SUMS.txt"
RAW_MANIFEST_FILE = RAW_DIR / "artifact_manifest.json"
RAW_ROWS_FILE = RAW_DIR / "external_transfer_rows.jsonl"
RAW_SUMMARY_FILE = RAW_DIR / "raw_external_transfer_summary.json"

RAW_GIT_BLOBS = {
    RAW_SUMS_FILE.as_posix():
        "a7f4b085f24e80a2c765e5c6f52d667e11a985a6",
    RAW_MANIFEST_FILE.as_posix():
        "969076247ef3daf339f6c98d1e5915b7950fa86f",
    RAW_ROWS_FILE.as_posix():
        "87c03efb0a5f74cf7f89a069bf5edfb166caaac0",
    RAW_SUMMARY_FILE.as_posix():
        "151881cbd88cadc9beafe64a6a785af61cbdd7dc",
}
RAW_SHA256 = {
    "artifact_manifest.json":
        "cf8d626b3b7908b9fca615319d03bada4e796df2beb50cf9961db410665c93f6",
    "external_transfer_rows.jsonl":
        "e2ef452a6dd3bb076924db5e4892741948e4a3c8c5ff11913a2c848cbc72980a",
    "raw_external_transfer_summary.json":
        "7bc2d0260dd664b2f0953ac224be44bd23de4b58998b42eada95ce35d21a761e",
}

RAW_RESULT = "PASS_GEN4_MAMBA14B_AVERITEC_NEGATIVE_SIGN_RAW"
RESULT_PASS = "PASS_GEN4_MAMBA14B_AVERITEC_NEGATIVE_SIGN_STATIC_ANALYSIS"
SUPPORTED = "MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_SUPPORTED"
NOT_ESTABLISHED = "MAMBA14B_AVERITEC_NEGATIVE_SIGN_TRANSFER_NOT_ESTABLISHED"

N = 462
DF = 461
ALPHA = 0.05
SCALE = "mamba14b"
CHECKPOINT_SHA256 = (
    "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a"
)
SELECTED_PLANE = "P5"
CONTROL_PLANE = "P4"
INTERVENTION_LAYER = 35
ANCHOR_NAME = "A_CLAIM_EVIDENCE_BOUNDARY"
TARGET_OFFSET = 2
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_control",
)

ANALYSIS_FILE = "external_transfer_analysis.json"
ITEM_FILE = "external_transfer_items.jsonl"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ANALYSIS_SCHEMA = "gen4-averitec-mamba14b-negative-sign-static-analysis-v1"
ITEM_SCHEMA = "gen4-averitec-mamba14b-negative-sign-static-item-v1"
MANIFEST_SCHEMA = "gen4-averitec-mamba14b-negative-sign-static-manifest-v1"


class StaticAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticAnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticAnalysisError(
            f"GIT_BLOB_READ:{path.as_posix()}"
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")

    for path, expected_blob in RAW_GIT_BLOBS.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"RAW_BLOB:{path}",
        )


def parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{label}:{line_no}")
        out.append(value)
    return out


def validate_raw() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sums_raw = git_blob_bytes(RAW_SUMS_FILE)
    parsed: dict[str, str] = {}
    for line in sums_raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in parsed, f"SUM_DUPLICATE:{name}")
        parsed[name] = digest
    require(parsed == RAW_SHA256, "RAW_SUMS_CONTENT")

    manifest_raw = git_blob_bytes(RAW_MANIFEST_FILE)
    rows_raw = git_blob_bytes(RAW_ROWS_FILE)
    summary_raw = git_blob_bytes(RAW_SUMMARY_FILE)

    require(
        sha256_bytes(manifest_raw) == RAW_SHA256["artifact_manifest.json"],
        "RAW_MANIFEST_SHA",
    )
    require(
        sha256_bytes(rows_raw) == RAW_SHA256["external_transfer_rows.jsonl"],
        "RAW_ROWS_SHA",
    )
    require(
        sha256_bytes(summary_raw)
        == RAW_SHA256["raw_external_transfer_summary.json"],
        "RAW_SUMMARY_SHA",
    )

    manifest = json.loads(manifest_raw.decode("utf-8"))
    summary = json.loads(summary_raw.decode("utf-8"))
    rows = parse_jsonl(rows_raw, "RAW_ROWS")

    require(manifest["result"] == RAW_RESULT, "RAW_MANIFEST_RESULT")
    require(summary["result"] == RAW_RESULT, "RAW_SUMMARY_RESULT")
    require(manifest["scale"] == SCALE, "RAW_MANIFEST_SCALE")
    require(summary["scale"] == SCALE, "RAW_SUMMARY_SCALE")
    require(int(manifest["compatible_item_count"]) == N, "RAW_MANIFEST_N")
    require(int(summary["compatible_item_count"]) == N, "RAW_SUMMARY_N")
    require(int(manifest["row_count"]) == N * len(CONDITIONS), "RAW_MANIFEST_ROWS")
    require(int(summary["row_count"]) == N * len(CONDITIONS), "RAW_SUMMARY_ROWS")
    require(tuple(manifest["condition_order"]) == CONDITIONS, "RAW_MANIFEST_CONDITIONS")
    require(tuple(summary["condition_order"]) == CONDITIONS, "RAW_SUMMARY_CONDITIONS")

    require(
        summary["checkpoint_sha256"] == CHECKPOINT_SHA256,
        "RAW_CHECKPOINT",
    )
    require(
        summary["selected_dominant_candidate"] == SELECTED_PLANE,
        "RAW_SELECTED",
    )
    require(
        summary["response_blind_control_plane"] == CONTROL_PLANE,
        "RAW_CONTROL",
    )
    require(int(summary["intervention_layer"]) == INTERVENTION_LAYER, "RAW_LAYER")
    require(summary["anchor_name"] == ANCHOR_NAME, "RAW_ANCHOR")
    require(int(summary["target_offset"]) == TARGET_OFFSET, "RAW_OFFSET")
    require(
        summary["primary_endpoint_reserved_for_static_analysis"]
        == "D_EXT_14B=M_native-M_control",
        "RAW_ENDPOINT",
    )
    require(
        summary["primary_alternative_reserved_for_static_analysis"] == "less",
        "RAW_ALTERNATIVE",
    )
    require(
        int(summary["primary_p_value_count_reserved"]) == 1,
        "RAW_RESERVED_P",
    )

    for obj, label in ((manifest, "MANIFEST"), (summary, "SUMMARY")):
        require(obj["training_executed"] is False, f"RAW_{label}_TRAIN")
        require(int(obj["backward_count"]) == 0, f"RAW_{label}_BACKWARD")
        require(obj["primary_inference_executed"] is False, f"RAW_{label}_INFERENCE")
        require(int(obj["p_value_count_executed"]) == 0, f"RAW_{label}_P")
        require(obj["scientific_conclusion"] is None, f"RAW_{label}_CONCLUSION")
        require(obj["selection_reopened"] is False, f"RAW_{label}_SELECTION")
        require(obj["rescue_performed"] is False, f"RAW_{label}_RESCUE")
        require(
            obj["historical_130m370m_averitec_family_reopened"] is False,
            f"RAW_{label}_HIST_FAMILY",
        )
        require(
            obj["historical_averitec_item_results_accessed"] is False,
            f"RAW_{label}_HIST_ITEMS",
        )

    require(len(rows) == N * len(CONDITIONS), "RAW_ROW_COUNT")
    observed: set[tuple[str, str]] = set()
    for row in rows:
        item_id = str(row["example_id"])
        condition = str(row["condition"])
        key = (item_id, condition)
        require(key not in observed, f"RAW_DUP:{key}")
        observed.add(key)

        require(row["scale"] == SCALE, f"RAW_ROW_SCALE:{key}")
        require(row["checkpoint_sha256"] == CHECKPOINT_SHA256, f"RAW_ROW_CKPT:{key}")
        require(condition in CONDITIONS, f"RAW_ROW_CONDITION:{key}")
        require(row["selected_dominant_candidate"] == SELECTED_PLANE, f"RAW_ROW_SELECTED:{key}")
        require(row["response_blind_control_plane"] == CONTROL_PLANE, f"RAW_ROW_CONTROL:{key}")
        require(int(row["intervention_layer"]) == INTERVENTION_LAYER, f"RAW_ROW_LAYER:{key}")
        require(row["anchor_name"] == ANCHOR_NAME, f"RAW_ROW_ANCHOR:{key}")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + TARGET_OFFSET,
            f"RAW_ROW_TARGET:{key}",
        )
        require(
            math.isfinite(float(row["correct_class_logit_margin"])),
            f"RAW_ROW_MARGIN:{key}",
        )
        require(row["primary_inference_executed"] is False, f"RAW_ROW_INFERENCE:{key}")
        require(int(row["p_value_count_executed"]) == 0, f"RAW_ROW_P:{key}")
        require(row["training_executed"] is False, f"RAW_ROW_TRAIN:{key}")
        require(row["backward_executed"] is False, f"RAW_ROW_BACKWARD:{key}")
        require("D_EXT" not in row and "D_EXT_14B" not in row, f"RAW_ROW_ENDPOINT:{key}")

    item_ids = {item for item, _ in observed}
    require(len(item_ids) == N, "RAW_ITEM_COUNT")
    expected = {
        (item_id, condition)
        for item_id in item_ids
        for condition in CONDITIONS
    }
    require(observed == expected, "RAW_COVERAGE")
    return rows, summary, manifest


def one_sample_less(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (N,), "PRIMARY_SHAPE")
    require(bool(np.isfinite(array).all()), "PRIMARY_FINITE")
    sd = float(np.std(array, ddof=1))
    require(sd > 0.0 and math.isfinite(sd), "PRIMARY_SD")

    test = stats.ttest_1samp(
        array,
        popmean=0.0,
        alternative="less",
    )
    mean_value = float(np.mean(array))
    t_value = float(test.statistic)
    p_value = float(test.pvalue)

    require(math.isfinite(mean_value), "PRIMARY_MEAN")
    require(math.isfinite(t_value), "PRIMARY_T")
    require(0.0 <= p_value <= 1.0, "PRIMARY_P")

    q25, median, q75 = [
        float(v)
        for v in np.quantile(array, [0.25, 0.5, 0.75])
    ]

    return {
        "n": N,
        "df": DF,
        "mean": mean_value,
        "sd_sample": sd,
        "t_statistic": t_value,
        "p_one_sided_less": p_value,
        "cohen_dz": mean_value / sd,
        "min": float(np.min(array)),
        "q25": q25,
        "median": median,
        "q75": q75,
        "max": float(np.max(array)),
        "fraction_negative": float(np.mean(array < 0.0)),
        "fraction_positive": float(np.mean(array > 0.0)),
        "fraction_zero": float(np.mean(array == 0.0)),
    }


def analyze() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, summary, _manifest = validate_raw()
    by = {
        (str(row["example_id"]), str(row["condition"])): row
        for row in rows
    }

    native_rows = [
        row for row in rows
        if str(row["condition"]) == "native"
    ]
    native_rows.sort(key=lambda row: int(row["averitec_dev_index"]))
    item_ids = [str(row["example_id"]) for row in native_rows]
    require(len(item_ids) == N and len(set(item_ids)) == N, "ITEM_IDS")

    d_ext: list[float] = []
    native_minus_neutralized: list[float] = []
    item_records: list[dict[str, Any]] = []
    label_values: dict[str, list[float]] = {
        "Refuted": [],
        "Supported": [],
        "Not Enough Evidence": [],
    }
    accuracy: dict[str, list[float]] = {
        condition: [] for condition in CONDITIONS
    }
    flips: dict[str, int] = {
        "dominant_neutralized": 0,
        "dominant_control": 0,
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
        require(math.isfinite(d) and math.isfinite(a), f"ITEM_FINITE:{item_id}")

        source_label = str(native["source_label"])
        require(source_label in label_values, f"SOURCE_LABEL:{source_label}")

        d_ext.append(d)
        native_minus_neutralized.append(a)
        label_values[source_label].append(d)

        for condition in CONDITIONS:
            accuracy[condition].append(
                float(bool(by[(item_id, condition)]["is_correct"]))
            )

        for condition in flips:
            if (
                int(by[(item_id, condition)]["prediction_id"])
                != int(native["prediction_id"])
            ):
                flips[condition] += 1

        item_records.append({
            "schema_version": ITEM_SCHEMA,
            "example_id": item_id,
            "averitec_dev_index": int(native["averitec_dev_index"]),
            "source_label": source_label,
            "correct_label": str(native["correct_label"]),
            "M_native": m_native,
            "M_neutralized": m_neutral,
            "M_control": m_control,
            "native_minus_neutralized": a,
            "D_EXT_14B": d,
        })

    primary = one_sample_less(d_ext)
    mean_negative_gate = float(primary["mean"]) < 0.0
    alpha_gate = float(primary["p_one_sided_less"]) < ALPHA
    supported = bool(mean_negative_gate and alpha_gate)
    conclusion = SUPPORTED if supported else NOT_ESTABLISHED

    descriptive = {
        "mean_native_minus_neutralized":
            float(np.mean(np.asarray(native_minus_neutralized, dtype=np.float64))),
        "accuracy_by_condition": {
            condition: float(np.mean(np.asarray(values, dtype=np.float64)))
            for condition, values in accuracy.items()
        },
        "prediction_flip_rate_vs_native": {
            condition: flips[condition] / N
            for condition in flips
        },
        "mean_D_EXT_14B_by_source_label": {
            label: float(np.mean(np.asarray(values, dtype=np.float64)))
            for label, values in label_values.items()
        },
        "source_label_counts": {
            label: len(values)
            for label, values in label_values.items()
        },
    }

    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "result": RESULT_PASS,
        "execution_type": "cpu_static_analysis",
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_execution_head": str(summary["execution_head"]),
        "prospective_plan_sha256": PLAN_SHA256,
        "scale": SCALE,
        "population":
            "AVeriTeC official dev compatible 3-label gold-evidence cohort",
        "N": N,
        "df": DF,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "primary_endpoint": "D_EXT_14B=M_native-M_control",
        "primary_test":
            "one-sided one-sample Student t-test, alternative=less",
        "alpha": ALPHA,
        "multiplicity": "none",
        "primary": primary,
        "negative_mean_sign_gate": mean_negative_gate,
        "alpha_gate": alpha_gate,
        "primary_supported": supported,
        "primary_p_value_count": 1,
        "secondary_p_value_count": 0,
        "historical_130m370m_averitec_family_reopened": False,
        "historical_p_values_in_new_family": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "training_executed": False,
        "model_forward_count": 0,
        "backward_count": 0,
        "scientific_conclusion": conclusion,
        "descriptive": descriptive,
    }
    return analysis, item_records


def write_outputs(output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    analysis, item_records = analyze()
    require(len(item_records) == N, "ITEM_RECORD_COUNT")

    output_dir.mkdir(parents=True, exist_ok=False)
    analysis_raw = pretty_json_bytes(analysis)
    items_raw = jsonl_bytes(item_records)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_type": "cpu_static_analysis",
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "prospective_plan_sha256": PLAN_SHA256,
        "input_git_blobs": dict(RAW_GIT_BLOBS),
        "input_sha256": dict(RAW_SHA256),
        "N": N,
        "primary_endpoint": "D_EXT_14B=M_native-M_control",
        "primary_test_alternative": "less",
        "primary_p_value_count": 1,
        "secondary_p_value_count": 0,
        "multiplicity": "none",
        "historical_130m370m_averitec_family_reopened": False,
        "historical_p_values_in_new_family": 0,
        "row_filter_performed": False,
        "rescue_performed": False,
        "selection_reopened": False,
        "training_executed": False,
        "model_forward_count": 0,
        "backward_count": 0,
        "scientific_conclusion": analysis["scientific_conclusion"],
        "output_sha256": {
            ANALYSIS_FILE: sha256_bytes(analysis_raw),
            ITEM_FILE: sha256_bytes(items_raw),
        },
    }
    manifest_raw = pretty_json_bytes(manifest)

    (output_dir / ANALYSIS_FILE).write_bytes(analysis_raw)
    (output_dir / ITEM_FILE).write_bytes(items_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        ANALYSIS_FILE: sha256_bytes(analysis_raw),
        ITEM_FILE: sha256_bytes(items_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single prespecified Mamba-1.4B AVeriTeC negative-sign "
            "external-transfer static analysis from frozen raw Git blobs."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    analysis = write_outputs(args.output_dir)

    primary = analysis["primary"]
    print("RESULT=" + analysis["result"])
    print("N=" + str(analysis["N"]))
    print("DF=" + str(analysis["df"]))
    print("MEAN_D_EXT_14B=" + format(float(primary["mean"]), ".17g"))
    print("SD_SAMPLE=" + format(float(primary["sd_sample"]), ".17g"))
    print("T=" + format(float(primary["t_statistic"]), ".17g"))
    print(
        "P_ONE_SIDED_LESS="
        + format(float(primary["p_one_sided_less"]), ".17g")
    )
    print("COHEN_DZ=" + format(float(primary["cohen_dz"]), ".17g"))
    print(
        "FRACTION_NEGATIVE="
        + format(float(primary["fraction_negative"]), ".17g")
    )
    print(
        "FRACTION_POSITIVE="
        + format(float(primary["fraction_positive"]), ".17g")
    )
    print(
        "NEGATIVE_MEAN_SIGN_GATE="
        + str(bool(analysis["negative_mean_sign_gate"]))
    )
    print("ALPHA_GATE=" + str(bool(analysis["alpha_gate"])))
    print("PRIMARY_SUPPORTED=" + str(bool(analysis["primary_supported"])))
    print("PRIMARY_P_VALUE_COUNT=1")
    print("SECONDARY_P_VALUE_COUNT=0")
    print("MULTIPLICITY=none")
    print("HISTORICAL_130M370M_FAMILY_REOPENED=False")
    print("HISTORICAL_P_VALUES_IN_NEW_FAMILY=0")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("SELECTION_REOPENED=False")
    print("TRAINING_EXECUTED=False")
    print("MODEL_FORWARD_COUNT=0")
    print("BACKWARD_COUNT=0")
    print("SCIENTIFIC_CONCLUSION=" + analysis["scientific_conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
