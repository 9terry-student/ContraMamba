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

READOUT_FREEZE_COMMIT = "8a2043a261c2ff7a01ff3c24855aca8640e1ca87"
PLAN_PATH = ROOT / "reports/reason_router_gen4_mamba130m_readout_alignment_prospective_plan.md"
PLAN_SHA256 = "a8a2cb2af1ed35b82f404a713c15ebf176cd645f653c6eacf73722b6e34058b3"

N = 300
CELLS = ("C0_SHAM", "C2_NAME")
CONDITIONS = ("native", "pp3_neutralized", "pp3_restored", "pp5_replacement")
EXPECTED_PAIRS = tuple(f"xg1_fact_{i:03d}" for i in range(2701, 3001))

READOUT_DIR = ROOT / "reports/reason_router_gen4_mamba130m_readout_alignment_analysis_v1"
READOUT_PAIR_FILE = "readout_alignment_pair_values.jsonl"
READOUT_ANALYSIS_FILE = "readout_alignment_analysis.json"
READOUT_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_READOUT_SHA256 = {
    READOUT_MANIFEST_FILE:
        "8f9d54d69544b553773c3f214001cd5565a956087d934877dfee6003dda8021c",
    READOUT_ANALYSIS_FILE:
        "6ae8866d4c39cd1ffceb490c77ba9f8f207a14bce26059d96dcf845fa1946184",
    READOUT_PAIR_FILE:
        "8012e07d2a53b90d9d271ca4e4aa93af26f78c2fdfeaf0d149e38ff602c232b4",
}

BEHAVIOR_RUN_DIR = (
    ROOT
    / "reports/reason_router_gen4_seed181_behavioral_restoration_bridge_runs"
    / "g4k-seed181-behavioral-bridge-xg1-2701-3000-da9e36a-retry2"
)
BEHAVIOR_ROW_FILE = "behavioral_rows.jsonl"
BEHAVIOR_SUMMARY_FILE = "shard_summary.json"
EXPECTED_BEHAVIOR_SHA256 = {
    0: {
        BEHAVIOR_ROW_FILE:
            "f9e61c1979c5b3f4ce4b9c6b3f15829bbd014efe7674a38b19f276f948672dc8",
        BEHAVIOR_SUMMARY_FILE:
            "1bc03fd4578971ab0a6571fb32df1eab25faada609546d94cc6c71dc3e9760ff",
    },
    1: {
        BEHAVIOR_ROW_FILE:
            "b3e6873db21a892f21e4ac451d84491193e88ccaa917cb3cc30f16477f8b7fbe",
        BEHAVIOR_SUMMARY_FILE:
            "ec5eb86649569cc6033e11e35331a4960c510cad36e9d2df42027c95f25ee3ab",
    },
}

EXPECTED_CHECKPOINT_SHA256 = (
    "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
)
EXPECTED_DESIGN_COMMIT = "9ea1617f4485fa0b6093df0c70aa88a42260c710"
SHARD_RANGES = {0: (2701, 2850), 1: (2851, 3000)}

RESULT = "PASS_GEN4_MAMBA130M_READOUT_BEHAVIOR_PAIR_LEVEL_DESCRIPTIVE_MERGE"
SCHEMA_VERSION = "gen4-mamba130m-readout-behavior-pair-level-descriptive-merge-v1"

PAIR_OUT_FILE = "pair_level_merge.jsonl"
ANALYSIS_OUT_FILE = "descriptive_analysis.json"
MANIFEST_OUT_FILE = "artifact_manifest.json"


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


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PairMergeError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH_MISMATCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", READOUT_FREEZE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "READOUT_FREEZE_NOT_ANCESTOR")


def validate_exact_sums(
    root: Path,
    *,
    expected: Mapping[str, str],
) -> None:
    sums_path = root / SUMS_FILE
    require(sums_path.is_file(), f"SUMS_MISSING:{root}")
    observed: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"SUM_DUPLICATE:{root}:{name}")
        observed[name] = digest
    require(observed == dict(expected), f"SUMS_CONTENT:{root}")
    for name, digest in observed.items():
        target = root / name
        require(target.is_file(), f"SUM_TARGET_MISSING:{root}:{name}")
        require(sha256_file(target) == digest, f"SUM_MISMATCH:{root}:{name}")


def read_readout_pairs(readout_dir: Path) -> list[dict[str, Any]]:
    require(sha256_file(PLAN_PATH) == PLAN_SHA256, "PLAN_SHA256")
    validate_exact_sums(readout_dir, expected=EXPECTED_READOUT_SHA256)

    analysis = json.loads(
        (readout_dir / READOUT_ANALYSIS_FILE).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (readout_dir / READOUT_MANIFEST_FILE).read_text(encoding="utf-8")
    )

    require(
        analysis["result"] == "PASS_GEN4_MAMBA130M_READOUT_ALIGNMENT_STATIC_ANALYSIS",
        "READOUT_RESULT",
    )
    require(
        analysis["scientific_conclusion"]
        == "MAMBA130M_READOUT_ALIGNMENT_POSITIVE_SUPPORTED",
        "READOUT_CONCLUSION",
    )
    require(analysis["population"] == "xg1_fact_2701..xg1_fact_3000", "READOUT_POP")
    require(int(analysis["pair_count"]) == N, "READOUT_N")
    require(tuple(analysis["cells"]) == CELLS, "READOUT_CELLS")
    require(analysis["scale"] == "mamba130m", "READOUT_SCALE")
    require(int(analysis["total_p_value_count"]) == 1, "READOUT_P_COUNT")
    require(analysis["behavioral_merge_performed"] is False, "READOUT_MERGE_BOUNDARY")
    require(analysis["three_scale_synthesis_performed"] is False, "READOUT_SCALE_BOUNDARY")
    require(analysis["row_filter_performed"] is False, "READOUT_FILTER")
    require(analysis["rescue_performed"] is False, "READOUT_RESCUE")

    require(int(manifest["pair_count"]) == N, "READOUT_MANIFEST_N")
    require(int(manifest["primary_p_value_count"]) == 1, "READOUT_MANIFEST_PRIMARY_P")
    require(int(manifest["total_p_value_count"]) == 1, "READOUT_MANIFEST_TOTAL_P")
    require(manifest["behavioral_merge_performed"] is False, "READOUT_MANIFEST_MERGE")
    require(manifest["three_scale_synthesis_performed"] is False, "READOUT_MANIFEST_SCALE")
    require(manifest["row_filter_performed"] is False, "READOUT_MANIFEST_FILTER")
    require(manifest["rescue_performed"] is False, "READOUT_MANIFEST_RESCUE")

    rows = jsonl_rows(readout_dir / READOUT_PAIR_FILE)
    require(len(rows) == N, "READOUT_PAIR_COUNT")
    observed = [str(row["source_pair_id"]) for row in rows]
    require(observed == list(EXPECTED_PAIRS), "READOUT_PAIR_ORDER")
    require(len(set(observed)) == N, "READOUT_PAIR_DUPLICATE")

    for row in rows:
        values = (
            float(row["Delta_L"]),
            float(row["Delta_L_C0_SHAM"]),
            float(row["Delta_L_C2_NAME"]),
        )
        require(all(math.isfinite(x) for x in values), f"READOUT_NONFINITE:{row['source_pair_id']}")
        require(
            math.isclose(values[0], 0.5 * (values[1] + values[2]), rel_tol=0.0, abs_tol=1e-15),
            f"READOUT_PAIR_IDENTITY:{row['source_pair_id']}",
        )
    return rows


def shard_expected_pairs(shard_id: int) -> tuple[str, ...]:
    require(shard_id in SHARD_RANGES, f"SHARD_ID:{shard_id}")
    first, last = SHARD_RANGES[shard_id]
    return tuple(f"xg1_fact_{i:03d}" for i in range(first, last + 1))


def read_behavior_shard(
    shard_dir: Path,
    *,
    shard_id: int,
) -> list[dict[str, Any]]:
    validate_exact_sums(
        shard_dir,
        expected=EXPECTED_BEHAVIOR_SHA256[shard_id],
    )
    summary = json.loads(
        (shard_dir / BEHAVIOR_SUMMARY_FILE).read_text(encoding="utf-8")
    )
    expected_pairs = shard_expected_pairs(shard_id)

    require(summary["result"] == "PASS_RAW_BEHAVIORAL_SHARD", f"SUMMARY_RESULT:{shard_id}")
    require(summary["design_commit"] == EXPECTED_DESIGN_COMMIT, f"SUMMARY_DESIGN:{shard_id}")
    require(summary["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, f"SUMMARY_CHECKPOINT:{shard_id}")
    require(int(summary["homolog_plane_index"]) == 3, f"SUMMARY_HOMOLOG:{shard_id}")
    require(int(summary["control_plane_index"]) == 5, f"SUMMARY_CONTROL:{shard_id}")
    require(summary["selection_uses_response"] is False, f"SUMMARY_SELECTION:{shard_id}")
    require(int(summary["shard_index"]) == shard_id, f"SUMMARY_SHARD:{shard_id}")
    require(summary["population_first"] == expected_pairs[0], f"SUMMARY_FIRST:{shard_id}")
    require(summary["population_last"] == expected_pairs[-1], f"SUMMARY_LAST:{shard_id}")
    require(int(summary["source_pair_count"]) == 150, f"SUMMARY_PAIR_COUNT:{shard_id}")
    require(tuple(summary["target_cells"]) == CELLS, f"SUMMARY_CELLS:{shard_id}")
    require(tuple(summary["condition_order"]) == CONDITIONS, f"SUMMARY_CONDITIONS:{shard_id}")
    require(summary["training_executed"] is False, f"SUMMARY_TRAINING:{shard_id}")
    require(summary["backward_executed"] is False, f"SUMMARY_BACKWARD:{shard_id}")
    require(summary["primary_inference_executed"] is False, f"SUMMARY_INFERENCE:{shard_id}")
    require(summary["scientific_conclusion"] is None, f"SUMMARY_CONCLUSION:{shard_id}")

    rows = jsonl_rows(shard_dir / BEHAVIOR_ROW_FILE)
    expected_tuples = {
        (pair, cell, condition)
        for pair in expected_pairs
        for cell in CELLS
        for condition in CONDITIONS
    }
    require(len(rows) == len(expected_tuples), f"BEHAVIOR_ROW_COUNT:{shard_id}")

    observed: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        )
        require(key not in observed, f"BEHAVIOR_DUPLICATE:{shard_id}:{key}")
        observed.add(key)
        require(row["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, f"BEHAVIOR_CHECKPOINT:{key}")
        require(int(row["shard_index"]) == shard_id, f"BEHAVIOR_SHARD:{key}")
        require(math.isfinite(float(row["correct_class_logit_margin"])), f"BEHAVIOR_MARGIN:{key}")

    require(observed == expected_tuples, f"BEHAVIOR_COVERAGE:{shard_id}")
    return rows


def reconstruct_behavior_pairs(
    behavior_run_dir: Path,
) -> list[dict[str, Any]]:
    shard0 = read_behavior_shard(behavior_run_dir / "shard0", shard_id=0)
    shard1 = read_behavior_shard(behavior_run_dir / "shard1", shard_id=1)
    rows = shard0 + shard1

    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["condition"]),
        )
        require(key not in by_key, f"BEHAVIOR_MERGED_DUPLICATE:{key}")
        by_key[key] = row

    require(len(by_key) == N * len(CELLS) * len(CONDITIONS), "BEHAVIOR_KEY_COUNT")

    out: list[dict[str, Any]] = []
    for pair in EXPECTED_PAIRS:
        restored = np.asarray(
            [
                float(by_key[(pair, cell, "pp3_restored")]["correct_class_logit_margin"])
                for cell in CELLS
            ],
            dtype=np.float64,
        )
        control = np.asarray(
            [
                float(by_key[(pair, cell, "pp5_replacement")]["correct_class_logit_margin"])
                for cell in CELLS
            ],
            dtype=np.float64,
        )
        d_beh = float(restored.mean() - control.mean())
        require(math.isfinite(d_beh), f"D_BEH_NONFINITE:{pair}")
        out.append({
            "source_pair_id": pair,
            "D_BEH": d_beh,
            "D_BEH_C0_SHAM": float(restored[0] - control[0]),
            "D_BEH_C2_NAME": float(restored[1] - control[1]),
        })
    require([x["source_pair_id"] for x in out] == list(EXPECTED_PAIRS), "BEHAVIOR_PAIR_ORDER")
    return out


def sign_code(x: float) -> int:
    require(math.isfinite(x), "SIGN_NONFINITE")
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def sign_category(delta_l: float, d_beh: float) -> str:
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
    return labels[(sign_code(delta_l), sign_code(d_beh))]


def descriptive(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (N,), "DESCRIPTIVE_SHAPE")
    require(bool(np.isfinite(x).all()), "DESCRIPTIVE_NONFINITE")
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
    require(x.shape == y.shape == (N,), "CORRELATION_SHAPE")
    require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), "CORRELATION_NONFINITE")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    require(math.isfinite(value), "CORRELATION_RESULT")
    return value


def spearman_without_p(x: np.ndarray, y: np.ndarray) -> float | None:
    rx = np.asarray(stats.rankdata(x, method="average"), dtype=np.float64)
    ry = np.asarray(stats.rankdata(y, method="average"), dtype=np.float64)
    return finite_correlation(rx, ry)


def summarize(
    delta_l: Sequence[float],
    d_beh: Sequence[float],
) -> dict[str, Any]:
    x = np.asarray(delta_l, dtype=np.float64)
    y = np.asarray(d_beh, dtype=np.float64)
    require(x.shape == y.shape == (N,), "SUMMARY_SHAPE")
    require(bool(np.isfinite(x).all()) and bool(np.isfinite(y).all()), "SUMMARY_NONFINITE")
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
    behavior = reconstruct_behavior_pairs(behavior_run_dir)

    merged: list[dict[str, Any]] = []
    for idx, pair in enumerate(EXPECTED_PAIRS):
        r = readout[idx]
        b = behavior[idx]
        require(str(r["source_pair_id"]) == pair, f"READOUT_PAIR_ALIGN:{pair}")
        require(str(b["source_pair_id"]) == pair, f"BEHAVIOR_PAIR_ALIGN:{pair}")

        delta_l = float(r["Delta_L"])
        d_beh = float(b["D_BEH"])
        require(math.isfinite(delta_l) and math.isfinite(d_beh), f"MERGE_NONFINITE:{pair}")

        merged.append({
            "source_pair_id": pair,
            "Delta_L": delta_l,
            "D_BEH": d_beh,
            "residual_D_BEH_minus_Delta_L": float(d_beh - delta_l),
            "sign_agree": sign_code(delta_l) == sign_code(d_beh),
        })

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "population": "xg1_fact_2701..xg1_fact_3000",
        "source_pair_count": N,
        "scale": "mamba130m",
        "primary_p_value_count": 0,
        "secondary_p_value_count": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "descriptive": summarize(
            [float(row["Delta_L"]) for row in merged],
            [float(row["D_BEH"]) for row in merged],
        ),
        "scientific_conclusion": None,
        "interpretation_threshold_applied": False,
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

    (output_dir / PAIR_OUT_FILE).write_bytes(
        b"".join(canonical_json_bytes(row) for row in merged)
    )
    (output_dir / ANALYSIS_OUT_FILE).write_bytes(pretty_json_bytes(analysis))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "readout_freeze_commit": READOUT_FREEZE_COMMIT,
        "plan_sha256": PLAN_SHA256,
        "primary_p_value_count": 0,
        "secondary_p_value_count": 0,
        "inference_performed": False,
        "training_executed": False,
        "forward_executed": False,
        "backward_executed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "source_pair_count": N,
        "scale": "mamba130m",
        "input_sha256": {
            "readout_pair_values": EXPECTED_READOUT_SHA256[READOUT_PAIR_FILE],
            "behavior_shard0_rows":
                EXPECTED_BEHAVIOR_SHA256[0][BEHAVIOR_ROW_FILE],
            "behavior_shard1_rows":
                EXPECTED_BEHAVIOR_SHA256[1][BEHAVIOR_ROW_FILE],
        },
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_OUT_FILE).write_bytes(pretty_json_bytes(manifest))

    names = (PAIR_OUT_FILE, ANALYSIS_OUT_FILE, MANIFEST_OUT_FILE)
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
            "CPU-only descriptive merge of frozen Mamba-130M pair-level Delta_L "
            "with the frozen seed181 behavioral-bridge D_BEH on the same 300 pairs. "
            "Adds no p-values."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--readout-analysis-dir", type=Path, default=READOUT_DIR)
    parser.add_argument("--behavioral-run-dir", type=Path, default=BEHAVIOR_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    analysis = write_outputs(
        readout_dir=args.readout_analysis_dir,
        behavior_run_dir=args.behavioral_run_dir,
        output_dir=args.output_dir,
    )
    d = analysis["descriptive"]
    print("RESULT=" + analysis["result"])
    print("PEARSON_DELTA_L_VS_D_BEH=" + repr(d["pearson_Delta_L_vs_D_BEH"]))
    print("SPEARMAN_DELTA_L_VS_D_BEH=" + repr(d["spearman_Delta_L_vs_D_BEH"]))
    print("SIGN_AGREEMENT=" + repr(d["sign_agreement"]["fraction"]))
    print("SIGN_AGREEMENT_COUNT=" + repr(d["sign_agreement"]["count"]))
    print("RESIDUAL_MEAN=" + repr(d["residual_D_BEH_minus_Delta_L"]["mean"]))
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
