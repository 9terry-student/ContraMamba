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
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

RAW_FREEZE_COMMIT = (
    "b245f3dc198eadf343bdbef490441819b0b75b8a"
)
RAW_EXECUTION_HEAD = (
    "1c3671ead1e08ba80651802a37147390806ed0e4"
)

RAW_RUN_NAME = (
    "g4k-mamba28b-readout-xg1-6601-6900-"
    "p3-p5-2gpu-1c3671e"
)

RAW_DIR = (
    ROOT
    / "reports/reason_router_gen4_mamba28b_readout_runs"
    / RAW_RUN_NAME
)

READOUT_PLAN_PATH = (
    ROOT
    / "data/reason_router_gen4_mamba28b_xg1_readout_v1"
    / "structural_manifest.json"
)

READOUT_PLAN_SHA256 = (
    "747064cd4af4112aa5a2de8cc5500d2d7734253f6a3920f27a4b129d749882b3"
)

ITEM_FILE = "readout_alignment_items.jsonl"
SUMMARY_FILE = "raw_readout_alignment_summary.json"
RAW_MANIFEST_FILE = "artifact_manifest.json"
RAW_SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_RAW_SHA256 = {
    RAW_MANIFEST_FILE:
        "583b72d27096862c87cb19b14ba6f5dce9f6183f5203aec6aa88832978c4f198",
    SUMMARY_FILE:
        "d0e2444667f66b61cb15a861b6b6b1a90f7f7dfb76419b266f1616a3dcb24826",
    ITEM_FILE:
        "6b66749e53816a2f0b9f4b35305b531250fce8e7b04fa5f2a98ce213bed6f883",
}

PAIR_IDS = tuple(
    f"xg1_fact_{index:03d}"
    for index in range(6601, 6901)
)
CELLS = ("C0_SHAM", "C2_NAME")
N = 300

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"

OWNERSHIP_ARM = "G3-GROUP-D-HALF"
OWNERSHIP_LAMBDA = 0.5
FORWARD_SCALE = 2.0

ANALYSIS_RESULT = (
    "PASS_GEN4_MAMBA28B_READOUT_ALIGNMENT_STATIC_DESCRIPTIVE"
)

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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
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


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
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


def jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(row)
        for row in rows
    )


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise AnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(
    expected_head: str,
) -> None:
    branch = git(
        "branch",
        "--show-current",
    )

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        git("rev-parse", "HEAD")
        == expected_head,
        "HEAD_MISMATCH",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            RAW_FREEZE_COMMIT,
            expected_head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    require(
        rc == 0,
        "RAW_FREEZE_NOT_ANCESTOR",
    )


def validate_readout_plan() -> dict[str, Any]:
    require(
        READOUT_PLAN_PATH.is_file(),
        "READOUT_PLAN_MISSING",
    )
    require(
        sha256_file(READOUT_PLAN_PATH)
        == READOUT_PLAN_SHA256,
        "READOUT_PLAN_SHA256",
    )

    manifest = json.loads(
        READOUT_PLAN_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        manifest["role"] == "readout",
        "READOUT_ROLE",
    )
    require(
        manifest["scale"] == "mamba28b",
        "READOUT_SCALE",
    )
    require(
        manifest["pair_id_first"]
        == "xg1_fact_6601",
        "READOUT_PAIR_FIRST",
    )
    require(
        manifest["pair_id_last"]
        == "xg1_fact_6900",
        "READOUT_PAIR_LAST",
    )

    require(
        manifest["planned_quantity"]
        == (
            "Delta_L_owned_with_forward_"
            "equivalent_recorded_if_applicable"
        ),
        "READOUT_PLANNED_QUANTITY",
    )

    require(
        "planned_primary_test"
        not in manifest,
        "UNEXPECTED_PLANNED_PRIMARY_TEST",
    )
    require(
        "success_rule"
        not in manifest,
        "UNEXPECTED_SUCCESS_RULE",
    )

    require(
        manifest[
            "selected_component_fixed_before_readout"
        ]
        is True,
        "SELECTED_NOT_FIXED",
    )
    require(
        manifest[
            "response_blind_control_fixed_before_readout"
        ]
        is True,
        "CONTROL_NOT_FIXED",
    )
    require(
        manifest["selection_allowed"]
        is False,
        "SELECTION_ALLOWED",
    )
    require(
        manifest[
            "response_guided_selection_allowed"
        ]
        is False,
        "RESPONSE_GUIDED_SELECTION",
    )
    require(
        manifest[
            "cohort_replacement_allowed"
        ]
        is False,
        "COHORT_REPLACEMENT",
    )
    require(
        manifest["row_filtering_allowed"]
        is False,
        "ROW_FILTERING",
    )
    require(
        manifest["rescue_policy"] == "none",
        "RESCUE_POLICY",
    )

    return manifest


def validate_sums(
    root: Path,
) -> None:
    sums_path = root / RAW_SUMS_FILE

    require(
        sums_path.is_file(),
        "SUMS_MISSING",
    )

    observed: dict[str, str] = {}

    for line in sums_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        parts = line.split("  ", 1)
        require(
            len(parts) == 2,
            "SUMS_FORMAT",
        )

        digest, name = parts

        require(
            name not in observed,
            f"SUMS_DUPLICATE:{name}",
        )

        observed[name] = digest

    require(
        observed == EXPECTED_RAW_SHA256,
        "SUMS_CONTENT",
    )

    for name, digest in observed.items():
        path = root / name

        require(
            path.is_file(),
            f"SUM_TARGET_MISSING:{name}",
        )
        require(
            sha256_file(path) == digest,
            f"SUM_MISMATCH:{name}",
        )


def read_raw_bundle(
    raw_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    validate_readout_plan()

    expected_files = {
        ITEM_FILE,
        SUMMARY_FILE,
        RAW_MANIFEST_FILE,
        RAW_SUMS_FILE,
    }

    require(
        raw_dir.is_dir(),
        "RAW_DIR_MISSING",
    )
    require(
        {
            path.name
            for path in raw_dir.iterdir()
            if path.is_file()
        }
        == expected_files,
        "RAW_FILE_SET",
    )

    validate_sums(raw_dir)

    summary = json.loads(
        (
            raw_dir
            / SUMMARY_FILE
        ).read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        (
            raw_dir
            / RAW_MANIFEST_FILE
        ).read_text(
            encoding="utf-8"
        )
    )

    expected_result = (
        "PASS_MAMBA28B_FRESH_DELTA_L_"
        "READOUT_RAW"
    )

    require(
        summary["result"]
        == expected_result,
        "RAW_SUMMARY_RESULT",
    )
    require(
        manifest["result"]
        == expected_result,
        "RAW_MANIFEST_RESULT",
    )

    require(
        summary["execution_head"]
        == RAW_EXECUTION_HEAD,
        "RAW_SUMMARY_HEAD",
    )
    require(
        manifest["execution_head"]
        == RAW_EXECUTION_HEAD,
        "RAW_MANIFEST_HEAD",
    )

    require(
        summary["population"]
        == "xg1_fact_6601..xg1_fact_6900",
        "RAW_POPULATION",
    )
    require(
        summary["pair_count"] == N,
        "RAW_PAIR_COUNT",
    )
    require(
        summary["target_cells"]
        == list(CELLS),
        "RAW_CELLS",
    )
    require(
        summary["item_count"] == 600,
        "RAW_ITEM_COUNT",
    )
    require(
        summary["scale"] == "mamba28b",
        "RAW_SCALE",
    )
    require(
        summary["selected_plane"]
        == SELECTED_PLANE,
        "RAW_SELECTED_PLANE",
    )
    require(
        summary["control_plane"]
        == CONTROL_PLANE,
        "RAW_CONTROL_PLANE",
    )

    require(
        summary["gradient_ownership_arm"]
        == OWNERSHIP_ARM,
        "RAW_OWNERSHIP_ARM",
    )
    require(
        summary["d_edge_gradient_lambda"]
        == OWNERSHIP_LAMBDA,
        "RAW_OWNERSHIP_LAMBDA",
    )
    require(
        summary["ownership_relation"]
        == (
            "Delta_L_forward_equivalent="
            "2*Delta_L_owned"
        ),
        "RAW_OWNERSHIP_RELATION",
    )

    require(
        summary[
            "pair_level_sign_aggregation_performed"
        ]
        is False,
        "RAW_SIGN_AGGREGATION_BOUNDARY",
    )
    require(
        summary["inferential_test_performed"]
        is False,
        "RAW_INFERENCE_BOUNDARY",
    )
    require(
        summary["p_value_count_executed"]
        == 0,
        "RAW_P_VALUE_COUNT",
    )
    require(
        summary["scientific_conclusion"]
        is None,
        "RAW_CONCLUSION",
    )

    for key in (
        "selection_reopened",
        "rescue_performed",
        "sweep_performed",
        "discovery_raw_response_accessed",
        "confirmation_raw_response_accessed",
    ):
        require(
            summary[key] is False,
            f"RAW_SUMMARY_{key}",
        )
        require(
            manifest[key] is False,
            f"RAW_MANIFEST_{key}",
        )

    require(
        manifest["source_pair_count"]
        == N,
        "RAW_MANIFEST_PAIR_COUNT",
    )
    require(
        manifest["item_count"] == 600,
        "RAW_MANIFEST_ITEM_COUNT",
    )
    require(
        manifest["pair_first"]
        == PAIR_IDS[0],
        "RAW_MANIFEST_PAIR_FIRST",
    )
    require(
        manifest["pair_last"]
        == PAIR_IDS[-1],
        "RAW_MANIFEST_PAIR_LAST",
    )
    require(
        manifest["selected_plane"]
        == SELECTED_PLANE,
        "RAW_MANIFEST_SELECTED",
    )
    require(
        manifest["control_plane"]
        == CONTROL_PLANE,
        "RAW_MANIFEST_CONTROL",
    )
    require(
        manifest["inferential_test_performed"]
        is False,
        "RAW_MANIFEST_INFERENCE",
    )
    require(
        manifest["p_value_count_executed"]
        == 0,
        "RAW_MANIFEST_PVALUE",
    )
    require(
        manifest["scientific_conclusion"]
        is None,
        "RAW_MANIFEST_CONCLUSION",
    )

    rows = [
        json.loads(line)
        for line in (
            raw_dir
            / ITEM_FILE
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(rows) == 600,
        "RAW_ROWS_COUNT",
    )

    return rows, summary, manifest


def finite_float(
    value: Any,
    label: str,
) -> float:
    result = float(value)

    require(
        math.isfinite(result),
        f"NONFINITE:{label}",
    )

    return result


def descriptive(
    values: Sequence[float],
) -> dict[str, Any]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    require(
        array.ndim == 1
        and array.size > 0,
        "DESCRIPTIVE_SHAPE",
    )
    require(
        bool(
            np.isfinite(
                array
            ).all()
        ),
        "DESCRIPTIVE_NONFINITE",
    )

    return {
        "n":
            int(array.size),
        "mean":
            float(array.mean()),
        "sd_sample":
            (
                float(
                    array.std(
                        ddof=1
                    )
                )
                if array.size > 1
                else 0.0
            ),
        "sd_population":
            float(
                array.std(
                    ddof=0
                )
            ),
        "min":
            float(array.min()),
        "q25":
            float(
                np.quantile(
                    array,
                    0.25,
                )
            ),
        "median":
            float(
                np.median(
                    array
                )
            ),
        "q75":
            float(
                np.quantile(
                    array,
                    0.75,
                )
            ),
        "max":
            float(array.max()),
        "fraction_positive":
            float(
                np.mean(
                    array > 0.0
                )
            ),
        "fraction_negative":
            float(
                np.mean(
                    array < 0.0
                )
            ),
        "fraction_zero":
            float(
                np.mean(
                    array == 0.0
                )
            ),
    }


def mean_sign(
    value: float,
) -> str:
    require(
        math.isfinite(value),
        "MEAN_SIGN_NONFINITE",
    )

    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def pair_values(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[
        tuple[str, str],
        Mapping[str, Any],
    ] = {}

    for row in rows:
        require(
            str(row["scale"])
            == "mamba28b",
            "ROW_SCALE",
        )

        pair = str(
            row[
                "source_pair_id"
            ]
        )
        cell = str(
            row[
                "contrast_cell_id"
            ]
        )

        key = (
            pair,
            cell,
        )

        require(
            key not in by_key,
            f"DUPLICATE:{key}",
        )

        require(
            str(
                row[
                    "selected_plane"
                ]
            )
            == SELECTED_PLANE,
            f"ROW_SELECTED:{key}",
        )
        require(
            str(
                row[
                    "control_plane"
                ]
            )
            == CONTROL_PLANE,
            f"ROW_CONTROL:{key}",
        )

        owned = finite_float(
            row[
                "Delta_L_owned"
            ],
            f"OWNED:{key}",
        )
        forward = finite_float(
            row[
                "Delta_L_forward_equivalent"
            ],
            f"FORWARD:{key}",
        )

        require(
            forward
            == FORWARD_SCALE
            * owned,
            f"OWNERSHIP_RELATION:{key}",
        )

        require(
            int(
                row[
                    "p_value_count_executed"
                ]
            )
            == 0,
            f"ROW_PVALUE:{key}",
        )
        require(
            row[
                "inferential_test_performed"
            ]
            is False,
            f"ROW_INFERENCE:{key}",
        )
        require(
            row[
                "selection_reopened"
            ]
            is False,
            f"ROW_SELECTION:{key}",
        )
        require(
            row[
                "rescue_performed"
            ]
            is False,
            f"ROW_RESCUE:{key}",
        )
        require(
            row[
                "sweep_performed"
            ]
            is False,
            f"ROW_SWEEP:{key}",
        )

        by_key[key] = row

    expected = {
        (
            pair,
            cell,
        )
        for pair in PAIR_IDS
        for cell in CELLS
    }

    require(
        set(by_key)
        == expected,
        "PAIR_CELL_COVERAGE",
    )

    out: list[
        dict[str, Any]
    ] = []

    for pair in PAIR_IDS:
        owned_values = [
            finite_float(
                by_key[
                    (
                        pair,
                        cell,
                    )
                ][
                    "Delta_L_owned"
                ],
                (
                    "OWNED:"
                    f"{pair}:{cell}"
                ),
            )
            for cell in CELLS
        ]

        forward_values = [
            finite_float(
                by_key[
                    (
                        pair,
                        cell,
                    )
                ][
                    "Delta_L_forward_equivalent"
                ],
                (
                    "FORWARD:"
                    f"{pair}:{cell}"
                ),
            )
            for cell in CELLS
        ]

        owned_pair = float(
            np.mean(
                np.asarray(
                    owned_values,
                    dtype=np.float64,
                )
            )
        )

        forward_pair = float(
            np.mean(
                np.asarray(
                    forward_values,
                    dtype=np.float64,
                )
            )
        )

        require(
            forward_pair
            == FORWARD_SCALE
            * owned_pair,
            f"PAIR_OWNERSHIP_RELATION:{pair}",
        )

        out.append({
            "source_pair_id":
                pair,
            "Delta_L_owned_C0_SHAM":
                owned_values[0],
            "Delta_L_owned_C2_NAME":
                owned_values[1],
            "Delta_L_owned":
                owned_pair,
            "Delta_L_forward_equivalent_C0_SHAM":
                forward_values[0],
            "Delta_L_forward_equivalent_C2_NAME":
                forward_values[1],
            "Delta_L_forward_equivalent":
                forward_pair,
        })

    require(
        len(out) == N,
        "PAIR_COUNT",
    )

    return out


def by_cell_descriptive(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[
        str,
        Any,
    ] = {}

    for cell in CELLS:
        cell_rows = [
            row
            for row in rows
            if str(
                row[
                    "contrast_cell_id"
                ]
            )
            == cell
        ]

        require(
            len(cell_rows)
            == N,
            f"CELL_COUNT:{cell}",
        )

        owned = [
            finite_float(
                row[
                    "Delta_L_owned"
                ],
                f"CELL_OWNED:{cell}",
            )
            for row in cell_rows
        ]

        forward = [
            finite_float(
                row[
                    "Delta_L_forward_equivalent"
                ],
                f"CELL_FORWARD:{cell}",
            )
            for row in cell_rows
        ]

        result[cell] = {
            "Delta_L_owned":
                descriptive(
                    owned
                ),
            "Delta_L_forward_equivalent":
                descriptive(
                    forward
                ),
        }

    return result


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    pairs = pair_values(
        rows
    )

    owned = [
        finite_float(
            row[
                "Delta_L_owned"
            ],
            "PAIR_OWNED",
        )
        for row in pairs
    ]

    forward = [
        finite_float(
            row[
                "Delta_L_forward_equivalent"
            ],
            "PAIR_FORWARD",
        )
        for row in pairs
    ]

    owned_desc = descriptive(
        owned
    )
    forward_desc = descriptive(
        forward
    )

    require(
        forward_desc[
            "mean"
        ]
        == FORWARD_SCALE
        * owned_desc[
            "mean"
        ],
        "MEAN_OWNERSHIP_RELATION",
    )

    owned_sign = mean_sign(
        float(
            owned_desc[
                "mean"
            ]
        )
    )
    forward_sign = mean_sign(
        float(
            forward_desc[
                "mean"
            ]
        )
    )

    require(
        owned_sign
        == forward_sign,
        "SIGN_INVARIANCE",
    )

    result = {
        "schema_version":
            (
                "gen4-mamba28b-readout-"
                "alignment-static-descriptive-v1"
            ),
        "result":
            ANALYSIS_RESULT,
        "scientific_conclusion":
            None,
        "formal_inference_performed":
            False,
        "p_value_count":
            0,
        "raw_freeze_commit":
            RAW_FREEZE_COMMIT,
        "raw_execution_head":
            RAW_EXECUTION_HEAD,
        "raw_item_sha256":
            EXPECTED_RAW_SHA256[
                ITEM_FILE
            ],
        "readout_plan_sha256":
            READOUT_PLAN_SHA256,
        "planned_primary_test_present":
            False,
        "success_rule_present":
            False,
        "population":
            "xg1_fact_6601..xg1_fact_6900",
        "pair_count":
            N,
        "cells":
            list(CELLS),
        "scale":
            "mamba28b",
        "selected_plane":
            SELECTED_PLANE,
        "control_plane":
            CONTROL_PLANE,
        "gradient_ownership_arm":
            OWNERSHIP_ARM,
        "d_edge_gradient_lambda":
            OWNERSHIP_LAMBDA,
        "ownership_relation":
            (
                "Delta_L_forward_equivalent="
                "2*Delta_L_owned"
            ),
        "pair_aggregation":
            (
                "arithmetic_mean_of_"
                "C0_SHAM_and_C2_NAME"
            ),
        "descriptive_mean_sign":
            owned_sign,
        "pair_Delta_L_owned":
            owned_desc,
        "pair_Delta_L_forward_equivalent":
            forward_desc,
        "by_cell":
            by_cell_descriptive(
                rows
            ),
        "row_filter_performed":
            False,
        "rescue_performed":
            False,
        "selection_reopened":
            False,
        "sweep_performed":
            False,
        "model_forward_count_this_analysis":
            0,
        "backward_count_this_analysis":
            0,
        "training_executed":
            False,
        "cross_scale_synthesis_performed":
            False,
        "threshold_estimation_performed":
            False,
        "monotonicity_test_performed":
            False,
    }

    return (
        result,
        pairs,
    )


def write_analysis(
    *,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows, _summary, _manifest = (
        read_raw_bundle(
            raw_dir
        )
    )

    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    result, pairs = (
        analyze_rows(
            rows
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        output_dir
        / ANALYSIS_FILE
    ).write_bytes(
        pretty_json_bytes(
            result
        )
    )

    (
        output_dir
        / PAIR_FILE
    ).write_bytes(
        jsonl_bytes(
            pairs
        )
    )

    manifest = {
        "schema_version":
            (
                "gen4-mamba28b-readout-"
                "alignment-analysis-manifest-v1"
            ),
        "result":
            ANALYSIS_RESULT,
        "scientific_conclusion":
            None,
        "descriptive_mean_sign":
            result[
                "descriptive_mean_sign"
            ],
        "raw_freeze_commit":
            RAW_FREEZE_COMMIT,
        "raw_execution_head":
            RAW_EXECUTION_HEAD,
        "raw_item_sha256":
            EXPECTED_RAW_SHA256[
                ITEM_FILE
            ],
        "analysis_sha256":
            sha256_file(
                output_dir
                / ANALYSIS_FILE
            ),
        "pair_values_sha256":
            sha256_file(
                output_dir
                / PAIR_FILE
            ),
        "pair_count":
            N,
        "formal_inference_performed":
            False,
        "p_value_count":
            0,
        "row_filter_performed":
            False,
        "rescue_performed":
            False,
        "selection_reopened":
            False,
        "sweep_performed":
            False,
        "additional_model_forward_count":
            0,
        "additional_backward_count":
            0,
        "training_executed":
            False,
        "cross_scale_synthesis_performed":
            False,
        "threshold_estimation_performed":
            False,
        "monotonicity_test_performed":
            False,
    }

    (
        output_dir
        / MANIFEST_FILE
    ).write_bytes(
        pretty_json_bytes(
            manifest
        )
    )

    names = (
        ANALYSIS_FILE,
        PAIR_FILE,
        MANIFEST_FILE,
    )

    (
        output_dir
        / SUMS_FILE
    ).write_text(
        "".join(
            (
                f"{sha256_file(output_dir / name)}"
                f"  {name}\n"
            )
            for name in sorted(
                names
            )
        ),
        encoding="utf-8",
        newline="\n",
    )

    return result


def parse_args(
    argv: Sequence[str]
    | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static descriptive analysis "
            "of frozen Mamba-2.8B fresh Delta_L "
            "readout evidence. Forms exactly 300 "
            "pair-level values by averaging C0_SHAM "
            "and C2_NAME. Executes no hypothesis "
            "test and no p-value."
        )
    )

    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Sequence[str]
    | None = None,
) -> int:
    args = parse_args(
        argv
    )

    authenticate_repo(
        args.expected_head
    )

    result = write_analysis(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )

    owned = result[
        "pair_Delta_L_owned"
    ]
    forward = result[
        "pair_Delta_L_forward_equivalent"
    ]

    print(
        "RESULT="
        + result["result"]
    )
    print(
        "PAIR_COUNT=300"
    )
    print(
        "PAIR_AGGREGATION="
        "mean(C0_SHAM,C2_NAME)"
    )
    print(
        "MEAN_DELTA_L_OWNED="
        f"{owned['mean']:.17g}"
    )
    print(
        "MEAN_DELTA_L_FORWARD_EQUIVALENT="
        f"{forward['mean']:.17g}"
    )
    print(
        "POSITIVE_FRACTION_OWNED="
        f"{owned['fraction_positive']:.17g}"
    )
    print(
        "DESCRIPTIVE_MEAN_SIGN="
        + result[
            "descriptive_mean_sign"
        ]
    )
    print(
        "FORMAL_INFERENCE_PERFORMED=False"
    )
    print(
        "P_VALUE_COUNT=0"
    )
    print(
        "SCIENTIFIC_CONCLUSION=None"
    )
    print(
        "ROW_FILTER_PERFORMED=False"
    )
    print(
        "RESCUE_PERFORMED=False"
    )
    print(
        "SELECTION_REOPENED=False"
    )
    print(
        "CROSS_SCALE_SYNTHESIS_PERFORMED=False"
    )
    print(
        "THRESHOLD_ESTIMATION_PERFORMED=False"
    )
    print(
        "MONOTONICITY_TEST_PERFORMED=False"
    )
    print(
        "MODEL_FORWARD_COUNT_THIS_ANALYSIS=0"
    )
    print(
        "BACKWARD_COUNT_THIS_ANALYSIS=0"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
