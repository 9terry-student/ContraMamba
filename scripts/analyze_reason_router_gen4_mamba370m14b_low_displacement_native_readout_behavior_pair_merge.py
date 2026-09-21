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

from scripts import (
    analyze_reason_router_gen4_mamba370m14b_readout_behavior_pair_merge
    as prior_merge,
)


ROOT = Path(__file__).resolve().parents[1]
N = 300
SCALES = ("mamba370m", "mamba14b")
CELLS = ("C0_SHAM", "C2_NAME")
EXPECTED_PAIRS = tuple(f"xg1_fact_{i}" for i in range(5401, 5701))
PRIMARY_ALPHA = 0.25

READOUT_ITEM_FILE = "low_displacement_native_readout_items.jsonl"
READOUT_SUMMARY_FILE = "raw_readout_summary.json"
READOUT_MANIFEST_FILE = "artifact_manifest.json"
BEHAVIOR_PAIR_FILE = "pair_level_contrasts.jsonl"
BEHAVIOR_ANALYSIS_FILE = "primary_analysis.json"
BEHAVIOR_MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"

RESULT = "PASS_GEN4_LOWDISP_NATIVE_READOUT_BEHAVIOR_PAIR_LEVEL_DESCRIPTIVE_MERGE"
SCHEMA_VERSION = "gen4-lowdisp-native-readout-behavior-pair-level-descriptive-merge-v1"

RAW_EXECUTION_HEAD = "5e6ff0617dc9d5e6db2b4f88e6afa6c9a5d9ea1b"
RAW_FREEZE_COMMIT = "e8f8877cb9a2e63e25c8f437edb627e820538f22"

EXPECTED_READOUT_SHA256 = {
    READOUT_MANIFEST_FILE: "0e59b3c5a1af8a419849f70e4d02cfcc4020f8d548c85d50d6f75842c24ea4cd",
    READOUT_ITEM_FILE: "0e69d42c09e445514cab799bc3646ef10500f35598cc1a10ce81502c2e2bb9cc",
    READOUT_SUMMARY_FILE: "c30211f92adbe55888126ff77076fd5e0cf2492e0857d2c2806b427f995720d0",
}

EXPECTED_BEHAVIOR_SHA256 = {
    BEHAVIOR_MANIFEST_FILE: "1097b4733b9961832e34b7d9155c3205ec5073843a4d131f1e72c7ffc75356c5",
    BEHAVIOR_PAIR_FILE: "712cce9c854f6d5a3b350c04bf11e0e06b19557c0a9e244dd3a93e195972c79d",
    BEHAVIOR_ANALYSIS_FILE: "ede3a33b201429b4b409d73b54213e85ba0da1a012aff8257b8c64490929e8fe",
}

EXPECTED_SELECTED = {"mamba370m": "P3", "mamba14b": "P5"}
EXPECTED_CONTROL = {"mamba370m": "P5", "mamba14b": "P4"}
EXPECTED_CHECKPOINT = {
    "mamba370m": "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
    "mamba14b": "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a",
}


class LowDispPairMergeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDispPairMergeError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDispPairMergeError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo_lineage() -> str:
    head = git("rev-parse", "HEAD")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", RAW_FREEZE_COMMIT, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "RAW_FREEZE_NOT_ANCESTOR")
    return head


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    root = ROOT.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise LowDispPairMergeError(
            f"INPUT_OUTSIDE_REPO:{path}"
        ) from exc
    return rel.as_posix()


def git_blob_bytes(path: Path) -> bytes:
    rel = repo_relative(path)
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDispPairMergeError(
            f"GIT_BLOB_FAILURE:{rel}"
        ) from exc


def git_blob_text(path: Path) -> str:
    return git_blob_bytes(path).decode("utf-8")


def jsonl_rows_from_text(text: str, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), f"JSONL_OBJECT:{label}:{line_no}")
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


def validate_frozen_sums(
    root: Path,
    *,
    expected_sha256: Mapping[str, str],
) -> None:
    sums_path = root / SUMS_FILE
    sums_text = git_blob_text(sums_path)
    seen: dict[str, str] = {}
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in seen, f"SUM_DUPLICATE:{root}:{name}")
        require(name in expected_sha256, f"SUM_UNEXPECTED:{root}:{name}")
        raw = git_blob_bytes(root / name)
        actual = sha256_bytes(raw)
        require(actual == digest, f"SUM_MISMATCH:{root}:{name}")
        require(
            expected_sha256[name] == digest,
            f"FROZEN_SHA_MISMATCH:{root}:{name}",
        )
        seen[name] = digest
    require(
        set(seen) == set(expected_sha256),
        f"SUM_SET:{root}:{set(seen)}",
    )


def validate_frozen_inputs(
    readout_raw_dir: Path,
    behavior_analysis_dir: Path,
) -> str:
    head = authenticate_repo_lineage()
    validate_frozen_sums(
        readout_raw_dir,
        expected_sha256=EXPECTED_READOUT_SHA256,
    )
    validate_frozen_sums(
        behavior_analysis_dir,
        expected_sha256=EXPECTED_BEHAVIOR_SHA256,
    )

    readout_summary = json.loads(
        git_blob_text(readout_raw_dir / READOUT_SUMMARY_FILE)
    )
    readout_manifest = json.loads(
        git_blob_text(readout_raw_dir / READOUT_MANIFEST_FILE)
    )

    require(
        readout_summary["result"]
        == "PASS_GEN4_LOWDISP_NATIVE_FUNCTIONAL_FAITHFULNESS_READOUT_RAW",
        "READOUT_RESULT",
    )
    require(readout_summary["execution_head"] == RAW_EXECUTION_HEAD, "READOUT_HEAD")
    require(
        readout_summary["population"] == "xg1_fact_5401..xg1_fact_5700",
        "READOUT_POP",
    )
    require(int(readout_summary["pair_count"]) == N, "READOUT_PAIR_COUNT")
    require(int(readout_summary["row_count"]) == 1200, "READOUT_ROW_COUNT")
    require(
        readout_summary["behavioral_response_accessed"] is False,
        "READOUT_BEHAVIOR_ACCESS",
    )
    require(
        readout_summary["predicted_behavior_computed"] is False,
        "READOUT_PREDICTED_BEHAVIOR",
    )
    require(
        readout_summary["behavioral_merge_computed"] is False,
        "READOUT_MERGE",
    )
    require(
        int(readout_summary["p_value_count_executed"]) == 0,
        "READOUT_P_COUNT",
    )
    require(
        readout_summary["scientific_conclusion"] is None,
        "READOUT_CONCLUSION",
    )

    require(
        readout_manifest["item_sha256"]
        == EXPECTED_READOUT_SHA256[READOUT_ITEM_FILE],
        "READOUT_ITEM_SHA",
    )
    require(
        readout_manifest["summary_sha256"]
        == EXPECTED_READOUT_SHA256[READOUT_SUMMARY_FILE],
        "READOUT_SUMMARY_SHA",
    )
    require(
        readout_manifest["behavioral_response_accessed"] is False,
        "READOUT_MANIFEST_BEHAVIOR_ACCESS",
    )
    require(
        int(readout_manifest["p_value_count_executed"]) == 0,
        "READOUT_MANIFEST_P_COUNT",
    )

    behavior_analysis = json.loads(
        git_blob_text(behavior_analysis_dir / BEHAVIOR_ANALYSIS_FILE)
    )
    behavior_manifest = json.loads(
        git_blob_text(behavior_analysis_dir / BEHAVIOR_MANIFEST_FILE)
    )

    require(
        behavior_analysis["result"]
        == "PASS_GEN4_LOW_DISPLACEMENT_BEHAVIORAL_STATIC_ANALYSIS",
        "BEHAVIOR_RESULT",
    )
    require(
        behavior_analysis["population"] == "xg1_fact_5401..xg1_fact_5700",
        "BEHAVIOR_POP",
    )
    require(int(behavior_analysis["pair_count"]) == N, "BEHAVIOR_PAIR_COUNT")
    require(
        float(behavior_analysis["primary_family"]["primary_alpha"])
        == PRIMARY_ALPHA,
        "BEHAVIOR_PRIMARY_ALPHA",
    )
    require(
        int(behavior_analysis["primary_family"]["primary_p_value_count"]) == 2,
        "BEHAVIOR_HISTORICAL_PRIMARY_P_COUNT",
    )
    require(
        behavior_analysis["row_filtering_performed"] is False,
        "BEHAVIOR_FILTER",
    )
    require(behavior_analysis["rescue_performed"] is False, "BEHAVIOR_RESCUE")
    require(
        behavior_analysis["training_executed"] is False,
        "BEHAVIOR_TRAINING",
    )

    require(
        behavior_manifest["pair_level_contrasts_sha256"]
        == EXPECTED_BEHAVIOR_SHA256[BEHAVIOR_PAIR_FILE],
        "BEHAVIOR_PAIR_SHA",
    )
    require(
        behavior_manifest["row_filtering_performed"] is False,
        "BEHAVIOR_MANIFEST_FILTER",
    )
    require(
        behavior_manifest["rescue_performed"] is False,
        "BEHAVIOR_MANIFEST_RESCUE",
    )
    return head


def pair_readout_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    require(len(rows) == 1200, "READOUT_ITEM_COUNT")

    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        scale = str(row["scale"])
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        key = (scale, pair, cell)
        require(scale in SCALES, f"READOUT_SCALE:{scale}")
        require(pair in EXPECTED_PAIRS, f"READOUT_PAIR:{pair}")
        require(cell in CELLS, f"READOUT_CELL:{cell}")
        require(key not in by_key, f"READOUT_DUPLICATE:{key}")
        require(
            row["selected_plane"] == EXPECTED_SELECTED[scale],
            f"READOUT_SELECTED:{key}",
        )
        require(
            row["control_plane"] == EXPECTED_CONTROL[scale],
            f"READOUT_CONTROL:{key}",
        )
        require(
            row["checkpoint_sha256"] == EXPECTED_CHECKPOINT[scale],
            f"READOUT_CHECKPOINT:{key}",
        )
        value = float(row["Delta_L_row"])
        require(math.isfinite(value), f"READOUT_DELTA_NONFINITE:{key}")
        require(
            row["behavioral_response_accessed"] is False,
            f"READOUT_BEHAVIOR_ACCESS:{key}",
        )
        require(
            int(row["p_value_count_executed"]) == 0,
            f"READOUT_P_COUNT:{key}",
        )
        by_key[key] = row

    expected_keys = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in EXPECTED_PAIRS
        for cell in CELLS
    }
    require(set(by_key) == expected_keys, "READOUT_COVERAGE")

    out: dict[str, list[dict[str, Any]]] = {}
    for scale in SCALES:
        records: list[dict[str, Any]] = []
        for pair in EXPECTED_PAIRS:
            values = [
                float(by_key[(scale, pair, cell)]["Delta_L_row"])
                for cell in CELLS
            ]
            pair_value = float(
                np.mean(np.asarray(values, dtype=np.float64))
            )
            require(
                math.isfinite(pair_value),
                f"PAIR_DELTA_NONFINITE:{scale}:{pair}",
            )
            records.append(
                {
                    "source_pair_id": pair,
                    "Delta_L": pair_value,
                    "Delta_L_C0_SHAM": values[0],
                    "Delta_L_C2_NAME": values[1],
                }
            )
        out[scale] = records
    return out


def read_readout_pair_values(
    readout_raw_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    text = git_blob_text(readout_raw_dir / READOUT_ITEM_FILE)
    rows = jsonl_rows_from_text(text, label=READOUT_ITEM_FILE)
    return pair_readout_from_rows(rows)


def behavior_values_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    require(len(rows) == N, "BEHAVIOR_PAIR_COUNT")

    observed = [str(row["source_pair_id"]) for row in rows]
    require(observed == list(EXPECTED_PAIRS), "BEHAVIOR_PAIR_ORDER")
    require(len(set(observed)) == N, "BEHAVIOR_PAIR_DUPLICATE")

    out: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for scale in SCALES:
            field = f"{scale}_D_BEH_alpha_0_25"
            value = float(copy[field])
            require(
                math.isfinite(value),
                f"BEHAVIOR_NONFINITE:{copy['source_pair_id']}:{field}",
            )
            copy[field] = value
        out.append(copy)
    return out


def read_behavior_pair_values(
    behavior_analysis_dir: Path,
) -> list[dict[str, Any]]:
    text = git_blob_text(behavior_analysis_dir / BEHAVIOR_PAIR_FILE)
    rows = jsonl_rows_from_text(text, label=BEHAVIOR_PAIR_FILE)
    return behavior_values_from_rows(rows)


def merge_and_analyze_values(
    readout: Mapping[str, Sequence[Mapping[str, Any]]],
    behavior: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(set(readout) == set(SCALES), "READOUT_SCALE_SET")
    require(len(behavior) == N, "BEHAVIOR_VECTOR_COUNT")

    for scale in SCALES:
        require(len(readout[scale]) == N, f"READOUT_VECTOR_COUNT:{scale}")

    merged: list[dict[str, Any]] = []
    for index, pair in enumerate(EXPECTED_PAIRS):
        r370 = readout["mamba370m"][index]
        r14 = readout["mamba14b"][index]
        b = behavior[index]

        require(str(r370["source_pair_id"]) == pair, f"READOUT_370_ALIGN:{pair}")
        require(str(r14["source_pair_id"]) == pair, f"READOUT_14_ALIGN:{pair}")
        require(str(b["source_pair_id"]) == pair, f"BEHAVIOR_ALIGN:{pair}")

        d370 = float(r370["Delta_L"])
        d14 = float(r14["Delta_L"])
        b370 = float(b["mamba370m_D_BEH_alpha_0_25"])
        b14 = float(b["mamba14b_D_BEH_alpha_0_25"])

        merged.append(
            {
                "source_pair_id": pair,
                "Delta_L_370M": d370,
                "D_BEH_alpha_0_25_370M": b370,
                "residual_D_BEH_minus_Delta_L_370M": float(b370 - d370),
                "sign_agree_370M": prior_merge.sign_code(d370)
                == prior_merge.sign_code(b370),
                "Delta_L_1.4B": d14,
                "D_BEH_alpha_0_25_1.4B": b14,
                "residual_D_BEH_minus_Delta_L_1.4B": float(b14 - d14),
                "sign_agree_1.4B": prior_merge.sign_code(d14)
                == prior_merge.sign_code(b14),
            }
        )

    scale_results = {
        "mamba370m": prior_merge.summarize_scale(
            [float(row["Delta_L_370M"]) for row in merged],
            [float(row["D_BEH_alpha_0_25_370M"]) for row in merged],
        ),
        "mamba14b": prior_merge.summarize_scale(
            [float(row["Delta_L_1.4B"]) for row in merged],
            [float(row["D_BEH_alpha_0_25_1.4B"]) for row in merged],
        ),
    }

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "population": "xg1_fact_5401..xg1_fact_5700",
        "source_pair_count": N,
        "scales": list(SCALES),
        "behavioral_alpha": PRIMARY_ALPHA,
        "pair_readout_definition": (
            "mean(Delta_L_row over C0_SHAM,C2_NAME), matching prior "
            "Mamba-370M/1.4B Study-B pair-level readout aggregation"
        ),
        "descriptive_metric_family": [
            "pearson_Delta_L_vs_D_BEH",
            "spearman_Delta_L_vs_D_BEH",
            "sign_agreement",
            "residual_D_BEH_minus_Delta_L",
        ],
        "metric_family_source": (
            "scripts/analyze_reason_router_gen4_mamba370m14b_"
            "readout_behavior_pair_merge.py"
        ),
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "interpretation_threshold_applied": False,
        "scale_results": scale_results,
        "scientific_conclusion": None,
    }
    return analysis, merged


def write_outputs(
    *,
    readout_raw_dir: Path,
    behavior_analysis_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    analysis_head = validate_frozen_inputs(
        readout_raw_dir,
        behavior_analysis_dir,
    )

    readout = read_readout_pair_values(readout_raw_dir)
    behavior = read_behavior_pair_values(behavior_analysis_dir)
    analysis, merged = merge_and_analyze_values(readout, behavior)

    output_dir.mkdir(parents=True, exist_ok=False)
    merge_raw = b"".join(canonical_json_bytes(row) for row in merged)
    (output_dir / "pair_level_merge.jsonl").write_bytes(merge_raw)
    (output_dir / "descriptive_analysis.json").write_bytes(
        pretty_json_bytes(analysis)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": RESULT,
        "execution_type": "cpu_static_analysis",
        "analysis_head": analysis_head,
        "raw_execution_head": RAW_EXECUTION_HEAD,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "population": "xg1_fact_5401..xg1_fact_5700",
        "source_pair_count": N,
        "behavioral_alpha": PRIMARY_ALPHA,
        "input_sha256": {
            "readout_items": EXPECTED_READOUT_SHA256[READOUT_ITEM_FILE],
            "readout_summary": EXPECTED_READOUT_SHA256[READOUT_SUMMARY_FILE],
            "readout_manifest": EXPECTED_READOUT_SHA256[READOUT_MANIFEST_FILE],
            "behavior_pair_level_contrasts": EXPECTED_BEHAVIOR_SHA256[
                BEHAVIOR_PAIR_FILE
            ],
            "behavior_primary_analysis": EXPECTED_BEHAVIOR_SHA256[
                BEHAVIOR_ANALYSIS_FILE
            ],
            "behavior_manifest": EXPECTED_BEHAVIOR_SHA256[
                BEHAVIOR_MANIFEST_FILE
            ],
        },
        "frozen_input_read_mode": "git_blob_at_HEAD",
        "metric_family_source": (
            "scripts/analyze_reason_router_gen4_mamba370m14b_"
            "readout_behavior_pair_merge.py"
        ),
        "primary_p_value_count_added": 0,
        "secondary_p_value_count_added": 0,
        "inference_performed": False,
        "training_executed": False,
        "forward_executed": False,
        "backward_executed": False,
        "row_filter_performed": False,
        "rescue_performed": False,
        "interpretation_threshold_applied": False,
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
            "CPU-only descriptive merge of frozen low-displacement native "
            "pair-level Delta_L with frozen alpha=0.25 pair-level D_BEH. "
            "Adds no p-values and performs no inference."
        )
    )
    parser.add_argument("--readout-raw-dir", type=Path, required=True)
    parser.add_argument("--behavior-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analysis = write_outputs(
        readout_raw_dir=args.readout_raw_dir,
        behavior_analysis_dir=args.behavior_analysis_dir,
        output_dir=args.output_dir,
    )

    print("RESULT=" + str(analysis["result"]))
    for scale in SCALES:
        result = analysis["scale_results"][scale]
        label = scale.upper()
        print(
            label
            + "_PEARSON_DELTA_L_VS_D_BEH_ALPHA_0_25="
            + repr(result["pearson_Delta_L_vs_D_BEH"])
        )
        print(
            label
            + "_SPEARMAN_DELTA_L_VS_D_BEH_ALPHA_0_25="
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
    print("PRIMARY_P_VALUE_COUNT_ADDED=0")
    print("SECONDARY_P_VALUE_COUNT_ADDED=0")
    print("INFERENCE_PERFORMED=False")
    print("ROW_FILTER_PERFORMED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("FORWARD_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
