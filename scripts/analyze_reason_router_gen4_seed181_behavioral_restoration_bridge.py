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
EXPECTED_CHECKPOINT_SHA256 = "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
EXPECTED_DESIGN_COMMIT = "9ea1617f4485fa0b6093df0c70aa88a42260c710"
EXPECTED_CONDITIONS = ("native", "pp3_neutralized", "pp3_restored", "pp5_replacement")
EXPECTED_CELLS = ("C0_SHAM", "C2_NAME")
SHARD_RANGES = {0: (2701, 2850), 1: (2851, 3000)}
FORWARDS_PER_SHARD = 1200
TOTAL_FORWARDS = 2400
ROW_FILE = "behavioral_rows.jsonl"
SUMMARY_FILE = "shard_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class BehavioralAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralAnalysisError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
            out.append(value)
    return out


def validate_checksums(shard_dir: Path) -> None:
    path = shard_dir / CHECKSUM_FILE
    require(path.is_file(), f"CHECKSUM_MISSING:{shard_dir}")
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            target = shard_dir / name
            require(target.is_file(), f"CHECKSUM_TARGET:{name}")
            require(sha256_file(target) == digest, f"CHECKSUM:{name}")
            seen.add(name)
    require(seen == {ROW_FILE, SUMMARY_FILE}, f"CHECKSUM_SET:{seen}")


def validate_shard(shard_dir: Path, shard_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_checksums(shard_dir)
    rows = read_jsonl(shard_dir / ROW_FILE)
    summary = json.loads((shard_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    first, last = SHARD_RANGES[shard_index]
    expected_pairs = [f"xg1_fact_{i:03d}" for i in range(first, last + 1)]

    require(summary["result"] == "PASS_RAW_BEHAVIORAL_SHARD", f"SUMMARY_RESULT:{shard_index}")
    require(summary["design_commit"] == EXPECTED_DESIGN_COMMIT, f"SUMMARY_DESIGN:{shard_index}")
    require(summary["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, f"SUMMARY_CHECKPOINT:{shard_index}")
    require(int(summary["homolog_plane_index"]) == 3, f"SUMMARY_HOMOLOG:{shard_index}")
    require(int(summary["control_plane_index"]) == 5, f"SUMMARY_CONTROL:{shard_index}")
    require(summary["selection_uses_response"] is False, f"SUMMARY_SELECTION:{shard_index}")
    require(int(summary["shard_index"]) == shard_index, f"SUMMARY_SHARD:{shard_index}")
    require(int(summary["physical_device"]) == shard_index, f"SUMMARY_PHYSICAL:{shard_index}")
    require(summary["population_first"] == expected_pairs[0], f"SUMMARY_FIRST:{shard_index}")
    require(summary["population_last"] == expected_pairs[-1], f"SUMMARY_LAST:{shard_index}")
    require(int(summary["source_pair_count"]) == 150, f"SUMMARY_N:{shard_index}")
    require(tuple(summary["target_cells"]) == EXPECTED_CELLS, f"SUMMARY_CELLS:{shard_index}")
    require(tuple(summary["condition_order"]) == EXPECTED_CONDITIONS, f"SUMMARY_CONDITIONS:{shard_index}")
    require(int(summary["scientific_full_model_forward_count_this_run"]) == FORWARDS_PER_SHARD, f"SUMMARY_BUDGET:{shard_index}")
    require(int(summary["physical_gpu_inventory_count"]) >= 2, f"SUMMARY_GPU_INVENTORY:{shard_index}")
    require(summary["model_name"] == "state-spaces/mamba-130m-hf", f"SUMMARY_MODEL:{shard_index}")
    require(summary["label_contract"] == {"C0_SHAM": "SUPPORT", "C2_NAME": "NOT_ENTITLED"}, f"SUMMARY_LABELS:{shard_index}")
    require(summary["training_executed"] is False, f"TRAINING_BOUNDARY:{shard_index}")
    require(summary["backward_executed"] is False, f"BACKWARD_BOUNDARY:{shard_index}")
    require(summary["primary_inference_executed"] is False, f"RAW_BOUNDARY:{shard_index}")
    require(summary["scientific_conclusion"] is None, f"RAW_CONCLUSION:{shard_index}")
    require(len(rows) == FORWARDS_PER_SHARD, f"ROW_COUNT:{shard_index}")

    expected = {
        (pair, cell, condition)
        for pair in expected_pairs
        for cell in EXPECTED_CELLS
        for condition in EXPECTED_CONDITIONS
    }
    observed: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["source_pair_id"]), str(row["contrast_cell_id"]), str(row["condition"]))
        require(key not in observed, f"DUPLICATE_TUPLE:{key}")
        observed.add(key)
        require(row["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, f"ROW_CHECKPOINT:{key}")
        require(int(row["shard_index"]) == shard_index, f"ROW_SHARD:{key}")
        require(int(row["physical_device"]) == shard_index, f"ROW_PHYSICAL:{key}")
        require(int(row["scientific_full_model_forward_count"]) == 1, f"ROW_BUDGET:{key}")
        require(math.isfinite(float(row["correct_class_logit_margin"])), f"ROW_MARGIN:{key}")
        require(len(row["final_logits"]) == 3, f"ROW_LOGITS:{key}")
    require(observed == expected, f"TUPLE_COVERAGE:{shard_index}")
    return rows, summary


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def analyze(run_dir: Path) -> dict[str, Any]:
    shard0, summary0 = validate_shard(run_dir / "shard0", 0)
    shard1, summary1 = validate_shard(run_dir / "shard1", 1)
    require(summary0["execution_head"] == summary1["execution_head"], "EXECUTION_HEAD_MISMATCH")
    require(summary0["structural_source_sha256"] == summary1["structural_source_sha256"], "SOURCE_SHA_MISMATCH")
    require(summary0["structural_rows_sha256"] == summary1["structural_rows_sha256"], "ROWS_SHA_MISMATCH")
    require(summary0["geometry_json_sha256"] == summary1["geometry_json_sha256"], "GEOMETRY_JSON_MISMATCH")
    require(summary0["geometry_pt_sha256"] == summary1["geometry_pt_sha256"], "GEOMETRY_PT_MISMATCH")
    require(summary0["model_name"] == summary1["model_name"], "MODEL_IDENTITY_MISMATCH")
    require(summary0["model_contract"] == summary1["model_contract"], "MODEL_CONTRACT_MISMATCH")
    require(summary0["tokenizer"] == summary1["tokenizer"], "TOKENIZER_IDENTITY_MISMATCH")
    require(summary0["label_contract"] == summary1["label_contract"], "LABEL_CONTRACT_MISMATCH")

    rows = shard0 + shard1
    require(len(rows) == TOTAL_FORWARDS, "TOTAL_ROW_COUNT")
    pair_ids = sorted({str(row["source_pair_id"]) for row in rows})
    require(pair_ids == [f"xg1_fact_{i:03d}" for i in range(2701, 3001)], "MERGED_PAIR_UNION")

    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"]), str(row["condition"])): row
        for row in rows
    }
    require(len(by_key) == TOTAL_FORWARDS, "MERGED_KEY_COUNT")

    pair_records: list[dict[str, Any]] = []
    condition_accuracy: dict[str, list[float]] = {c: [] for c in EXPECTED_CONDITIONS}
    flip_counts: dict[str, int] = {c: 0 for c in EXPECTED_CONDITIONS if c != "native"}
    flip_denominator = N * len(EXPECTED_CELLS)

    for pair_id in pair_ids:
        margins: dict[str, float] = {}
        q_values: dict[str, float] = {}
        for condition in EXPECTED_CONDITIONS:
            row_margins = [
                float(by_key[(pair_id, cell, condition)]["correct_class_logit_margin"])
                for cell in EXPECTED_CELLS
            ]
            margins[condition] = mean(row_margins)
            q_values[condition] = mean(
                [float(by_key[(pair_id, cell, condition)]["q_authorized"]) for cell in EXPECTED_CELLS]
            )
            for cell in EXPECTED_CELLS:
                condition_accuracy[condition].append(
                    1.0 if bool(by_key[(pair_id, cell, condition)]["is_correct"]) else 0.0
                )
                if condition != "native":
                    if int(by_key[(pair_id, cell, condition)]["prediction_id"]) != int(by_key[(pair_id, cell, "native")]["prediction_id"]):
                        flip_counts[condition] += 1

        d_beh = margins["pp3_restored"] - margins["pp5_replacement"]
        necessity = margins["native"] - margins["pp3_neutralized"]
        q_delta = q_values["pp3_restored"] - q_values["pp5_replacement"]
        pair_records.append(
            {
                "source_pair_id": pair_id,
                "M_native": margins["native"],
                "M_neutralized": margins["pp3_neutralized"],
                "M_restored": margins["pp3_restored"],
                "M_control": margins["pp5_replacement"],
                "D_BEH": d_beh,
                "D_NEC_BEH": necessity,
                "q_authorized_restored_minus_control": q_delta,
            }
        )

    d = np.asarray([float(x["D_BEH"]) for x in pair_records], dtype=np.float64)
    require(d.shape == (N,) and np.isfinite(d).all(), "D_BEH_VECTOR")
    test = stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    mean_d = float(np.mean(d))
    p = float(test.pvalue)
    supported = mean_d > 0.0 and p < 0.05

    q_delta = np.asarray(
        [float(x["q_authorized_restored_minus_control"]) for x in pair_records],
        dtype=np.float64,
    )
    q_corr: float | None
    if float(np.std(q_delta)) == 0.0 or float(np.std(d)) == 0.0:
        q_corr = None
    else:
        q_corr = float(np.corrcoef(q_delta, d)[0, 1])

    c0_d = np.asarray(
        [
            float(by_key[(pair, "C0_SHAM", "pp3_restored")]["correct_class_logit_margin"])
            - float(by_key[(pair, "C0_SHAM", "pp5_replacement")]["correct_class_logit_margin"])
            for pair in pair_ids
        ],
        dtype=np.float64,
    )
    c2_d = np.asarray(
        [
            float(by_key[(pair, "C2_NAME", "pp3_restored")]["correct_class_logit_margin"])
            - float(by_key[(pair, "C2_NAME", "pp5_replacement")]["correct_class_logit_margin"])
            for pair in pair_ids
        ],
        dtype=np.float64,
    )

    return {
        "schema_version": "gen4-seed181-behavioral-restoration-analysis-v1",
        "execution_head": summary0["execution_head"],
        "design_commit": EXPECTED_DESIGN_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "N": N,
        "df": N - 1,
        "homolog_plane_index": 3,
        "control_plane_index": 5,
        "population": "xg1_fact_2701..xg1_fact_3000",
        "primary_endpoint": "D_BEH=M_restored-M_control",
        "mean_D_BEH": mean_d,
        "D_BEH_t": float(test.statistic),
        "D_BEH_one_sided_p": p,
        "primary_gates": {
            "mean_D_BEH_gt_0": mean_d > 0.0,
            "one_sided_p_lt_0_05": p < 0.05,
        },
        "behavioral_bridge_supported": supported,
        "behavioral_result_label": (
            "SEED181_BEHAVIORAL_RESTORATION_BRIDGE_SUPPORTED"
            if supported
            else "SEED181_BEHAVIORAL_RESTORATION_BRIDGE_NOT_ESTABLISHED"
        ),
        "secondary_descriptive": {
            "mean_native_minus_neutralized_pair_margin": mean(
                [float(x["D_NEC_BEH"]) for x in pair_records]
            ),
            "accuracy_by_condition": {
                condition: mean(values) for condition, values in condition_accuracy.items()
            },
            "prediction_flip_rate_vs_native": {
                condition: float(count / flip_denominator)
                for condition, count in flip_counts.items()
            },
            "mean_D_BEH_C0_SHAM": float(np.mean(c0_d)),
            "mean_D_BEH_C2_NAME": float(np.mean(c2_d)),
            "q_authorized_effect_vs_margin_effect_pearson": q_corr,
            "q_authorized_note": (
                "This is the model's q_authorized diagnostic from the same full forward; "
                "it is not the earlier recurrent-trajectory Q endpoint."
            ),
            "internal_recurrent_trajectory_Q_association": None,
            "internal_recurrent_trajectory_Q_note": (
                "Not measured on xg1_fact_2701..3000; no extra rescue endpoint is introduced."
            ),
        },
        "scientific_full_model_forward_count": TOTAL_FORWARDS,
        "seed180_behavioral_rescue_executed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(not args.output.exists(), "OUTPUT_COLLISION")
    result = analyze(args.run_dir)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("BEHAVIORAL_RESULT=" + result["behavioral_result_label"])
    print("MEAN_D_BEH=" + repr(result["mean_D_BEH"]))
    print("D_BEH_ONE_SIDED_P=" + repr(result["D_BEH_one_sided_p"]))


if __name__ == "__main__":
    main()
