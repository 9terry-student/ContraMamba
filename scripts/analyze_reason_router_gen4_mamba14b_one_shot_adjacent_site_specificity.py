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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

DESIGN_FREEZE_COMMIT = "3ee2d7600de649a711752b8293687047c3c5ec4e"
RAW_FREEZE_COMMIT = "d0f7c086cc68476eeee33e7db65c7f21409bf42e"
ADJACENT_GEOMETRY_FREEZE_COMMIT = (
    "d434dfdc38f222419708ed67db87e4e28fb05a4d"
)

DESIGN_PATH = Path(
    "reports/reason_router_gen4_one_shot_adjacent_site_specificity_design.md"
)
DESIGN_BLOB = "9accf85f49e6278ee3d3b2b58482ffdb6b4085fc"

RAW_RUN_NAME = (
    "g4k-mamba14b-adjacent-specificity-raw-xg1-5101-5400-2gpu-d8327f9"
)
RAW_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_site_specificity_raw_runs"
) / RAW_RUN_NAME
RAW_ITEMS = RAW_ROOT / "paired_specificity_items.jsonl"
RAW_SUMMARY = RAW_ROOT / "raw_response_summary.json"
RAW_MANIFEST = RAW_ROOT / "artifact_manifest.json"
RAW_SUMS = RAW_ROOT / "SHA256SUMS.txt"

RAW_ITEMS_BLOB = "64da91ba82a17982feab0e7c8bd3bfba87bade00"
RAW_SUMMARY_BLOB = "f1f1e764c3cd5c412d13be67555004b10795b649"
RAW_MANIFEST_BLOB = "471d3fd0b73bc3bd28eb142593559cf588279e31"
RAW_SUMS_BLOB = "05f7422ab604eeb2075e1528589a3e5bdd6b1007"

RAW_ITEMS_SHA256 = (
    "49b8087a7928f104a88fca3e2966264f1dbd21a68ccce947e08197fb239686a3"
)
RAW_SUMMARY_SHA256 = (
    "70f4170bb2e1e4847da90026ec5cad7cbc973a847af6d66d306cb701074dccd9"
)
RAW_MANIFEST_SHA256 = (
    "445307d593d9e9560d47e34491fa2875cbce09c103fdb523e8d4dc67c5e01150"
)
RAW_SUMS_SHA256 = (
    "46ed4636ca1c99375f647f1a23d312f1e780aca292127347939bee0062619e09"
)

ADJACENT_GEOMETRY_SUMMARY = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_geometry_preparation_runs/"
    "g4k-mamba14b-adjacent-geometry-plus1-xg2xg4-2gpu-4de2451/"
    "geometry_summary.json"
)
ADJACENT_GEOMETRY_SUMMARY_BLOB = (
    "d6bd05aa8d47bdf80e088fbcfb5cd1e5e77a3866"
)
ADJACENT_GEOMETRY_SUMMARY_SHA256 = (
    "168aca69230b82eda29be2800b79b634362ac937335406f79f3e08a60d9f427e"
)

PAIR_COUNT = 300
PAIR_IDS = tuple(
    f"xg1_fact_{index:04d}"
    for index in range(5101, 5401)
)
ALPHA = 0.05

SUCCESS_LABEL = "MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_SUPPORTED"
FAILURE_LABEL = (
    "MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_NOT_ESTABLISHED"
)
RESULT_PASS = "PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_ANALYSIS"

ANALYSIS_SCHEMA = (
    "gen4-mamba14b-one-shot-adjacent-site-specificity-analysis-v1"
)
ANALYSIS_FILE = "adjacent_site_specificity_analysis.json"
REPORT_FILE = "adjacent_site_specificity_analysis.md"
SUMS_FILE = "SHA256SUMS.txt"


class SpecificityAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SpecificityAnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SpecificityAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SpecificityAnalysisError(
            f"GIT_BLOB_FAILURE:{path}"
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.decode("utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{line_no}")
        output.append(value)
    return output


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN"),
        (RAW_FREEZE_COMMIT, "RAW_FREEZE"),
        (ADJACENT_GEOMETRY_FREEZE_COMMIT, "ADJACENT_GEOMETRY"),
    ):
        require(
            git_rc(
                "merge-base",
                "--is-ancestor",
                ancestor,
                head,
            )
            == 0,
            f"{label}_NOT_ANCESTOR",
        )

    frozen_blobs = {
        DESIGN_PATH: DESIGN_BLOB,
        RAW_ITEMS: RAW_ITEMS_BLOB,
        RAW_SUMMARY: RAW_SUMMARY_BLOB,
        RAW_MANIFEST: RAW_MANIFEST_BLOB,
        RAW_SUMS: RAW_SUMS_BLOB,
        ADJACENT_GEOMETRY_SUMMARY:
            ADJACENT_GEOMETRY_SUMMARY_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        observed = git("rev-parse", f"HEAD:{path.as_posix()}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB:{path}:{observed}",
        )


def validate_sums(
    raw: bytes,
    expected: Mapping[str, str],
) -> None:
    observed: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"SUMS_DUPLICATE:{name}")
        observed[name] = digest
    require(observed == dict(expected), "SUMS_CONTENT")


def finite(value: Any, label: str) -> float:
    out = float(value)
    require(math.isfinite(out), f"NONFINITE:{label}")
    return out


def descriptive(
    values: Sequence[float],
) -> dict[str, Any]:
    from scipy import stats

    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and array.size > 1, "DESCRIPTIVE_SHAPE")

    finite_mask = np.isfinite(array)
    finite_count = int(np.sum(finite_mask))
    nonfinite_count = int(array.size - finite_count)
    require(nonfinite_count == 0, "DESCRIPTIVE_NONFINITE")

    n = int(array.size)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    require(math.isfinite(mean), "DESCRIPTIVE_MEAN")
    require(math.isfinite(sd) and sd >= 0.0, "DESCRIPTIVE_SD")

    se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    require(math.isfinite(critical), "CI_CRITICAL")

    return {
        "n": n,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "mean": mean,
        "sd": sd,
        "ci95_t_low": mean - critical * se,
        "ci95_t_high": mean + critical * se,
        "cohen_dz": None if sd == 0.0 else mean / sd,
        "fraction_positive": float(np.mean(array > 0.0)),
        "fraction_negative": float(np.mean(array < 0.0)),
        "fraction_zero": float(np.mean(array == 0.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def primary_one_sample_greater(
    values: Sequence[float],
    descriptive_stats: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy import stats

    array = np.asarray(values, dtype=np.float64)
    require(
        array.ndim == 1 and int(array.size) == PAIR_COUNT,
        "PRIMARY_SHAPE",
    )
    require(bool(np.isfinite(array).all()), "PRIMARY_NONFINITE")

    n = int(descriptive_stats["n"])
    mean = finite(descriptive_stats["mean"], "PRIMARY_MEAN")
    sd = finite(descriptive_stats["sd"], "PRIMARY_SD")
    require(n == PAIR_COUNT, "PRIMARY_N")
    require(sd > 0.0, "PRIMARY_ZERO_SD")

    t_stat = mean / (sd / math.sqrt(n))
    require(math.isfinite(t_stat), "PRIMARY_T")

    p_value = float(stats.t.sf(t_stat, df=n - 1))
    require(
        math.isfinite(p_value)
        and 0.0 <= p_value <= 1.0,
        "PRIMARY_P",
    )

    return {
        "endpoint": "S=D_CAN-D_ADJ",
        "test": "one_sample_student_t",
        "null": "E[S] <= 0",
        "alternative": "E[S] > 0",
        "tail": "greater",
        "alpha": ALPHA,
        "n": n,
        "df": n - 1,
        "null_mean": 0.0,
        "t_statistic": t_stat,
        "p_value": p_value,
        "p_value_count": 1,
        "multiplicity_correction": "none_single_test_family",
    }


def aggregate_q_components(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for site in ("canonical", "adjacent"):
        output[site] = {}
        for condition in ("dominant_restored", "dominant_control"):
            rows = [
                item[site]["conditions"][condition]
                for item in items
            ]
            output[site][condition] = {
                "Q": descriptive([
                    finite(row["Q"], f"{site}:{condition}:Q")
                    for row in rows
                ]),
                "E_XG2": descriptive([
                    finite(row["E_XG2"], f"{site}:{condition}:E_XG2")
                    for row in rows
                ]),
                "E_XG4": descriptive([
                    finite(row["E_XG4"], f"{site}:{condition}:E_XG4")
                    for row in rows
                ]),
            }
    return output


def aggregate_audit_maxima(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for site in ("canonical", "adjacent"):
        by_key: dict[str, list[float]] = {}
        for item in items:
            for condition in ("dominant_restored", "dominant_control"):
                audit = item[site]["conditions"][condition]["audit_maxima"]
                require(
                    isinstance(audit, dict),
                    f"AUDIT_TYPE:{site}:{condition}",
                )
                for key, value in audit.items():
                    by_key.setdefault(str(key), []).append(
                        abs(finite(value, f"AUDIT:{site}:{condition}:{key}"))
                    )

        output[site] = {
            key: {
                "max_abs": max(values),
                "observation_count": len(values),
            }
            for key, values in sorted(by_key.items())
        }

    return output


def validate_and_load_inputs() -> dict[str, Any]:
    items_raw = git_blob_bytes(RAW_ITEMS)
    summary_raw = git_blob_bytes(RAW_SUMMARY)
    manifest_raw = git_blob_bytes(RAW_MANIFEST)
    sums_raw = git_blob_bytes(RAW_SUMS)
    adjacent_geometry_raw = git_blob_bytes(
        ADJACENT_GEOMETRY_SUMMARY
    )

    require(
        sha256_bytes(items_raw) == RAW_ITEMS_SHA256,
        "RAW_ITEMS_SHA256",
    )
    require(
        sha256_bytes(summary_raw) == RAW_SUMMARY_SHA256,
        "RAW_SUMMARY_SHA256",
    )
    require(
        sha256_bytes(manifest_raw) == RAW_MANIFEST_SHA256,
        "RAW_MANIFEST_SHA256",
    )
    require(
        sha256_bytes(sums_raw) == RAW_SUMS_SHA256,
        "RAW_SUMS_SHA256",
    )
    require(
        sha256_bytes(adjacent_geometry_raw)
        == ADJACENT_GEOMETRY_SUMMARY_SHA256,
        "ADJACENT_GEOMETRY_SUMMARY_SHA256",
    )

    validate_sums(
        sums_raw,
        {
            "artifact_manifest.json": RAW_MANIFEST_SHA256,
            "paired_specificity_items.jsonl": RAW_ITEMS_SHA256,
            "raw_response_summary.json": RAW_SUMMARY_SHA256,
        },
    )

    items = read_jsonl_bytes(items_raw)
    summary = json.loads(summary_raw.decode("utf-8"))
    manifest = json.loads(manifest_raw.decode("utf-8"))
    adjacent_geometry = json.loads(
        adjacent_geometry_raw.decode("utf-8")
    )

    require(len(items) == PAIR_COUNT, "ITEM_COUNT")
    require(
        [str(item["source_pair_id"]) for item in items]
        == list(PAIR_IDS),
        "PAIR_ORDER",
    )

    require(
        summary["result"]
        == "PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_RAW_RESPONSE",
        "RAW_SUMMARY_RESULT",
    )
    require(summary["pair_count"] == PAIR_COUNT, "RAW_SUMMARY_N")
    require(
        summary["selected_causal_candidate"] == "P5",
        "RAW_SUMMARY_P5",
    )
    require(
        summary["canonical_response_blind_control_plane"] == "P4"
        and summary["adjacent_response_blind_control_plane"] == "P4",
        "RAW_SUMMARY_CONTROLS",
    )
    require(
        summary["canonical_triplet"] == [33, 34, 35]
        and summary["adjacent_triplet"] == [34, 35, 36],
        "RAW_SUMMARY_TRIPLETS",
    )
    require(
        summary["canonical_strong_dim"] == 829
        and summary["adjacent_strong_dim"] == 1205,
        "RAW_SUMMARY_STRONG_DIMS",
    )
    require(
        summary["forward_accounting"]["scientific_model_forward_count"]
        == 48000,
        "RAW_FORWARD_COUNT",
    )
    require(
        summary["inference"]["performed"] is False
        and summary["inference"]["primary_p_value_computed"] is False,
        "RAW_INFERENCE_ALREADY_RUN",
    )
    require(
        summary["scientific_conclusion"] is None,
        "RAW_PREMATURE_CONCLUSION",
    )
    require(
        summary["selection_reopened"] is False
        and summary["second_adjacent_site_executed"] is False
        and summary["layer_sweep_executed"] is False
        and summary["epsilon_sweep_executed"] is False
        and summary["token_sweep_executed"] is False
        and summary["rescue_performed"] is False,
        "RAW_BOUNDARY",
    )

    require(
        manifest["output_file_sha256"]
        == {
            "paired_specificity_items.jsonl": RAW_ITEMS_SHA256,
            "raw_response_summary.json": RAW_SUMMARY_SHA256,
        },
        "RAW_MANIFEST_OUTPUT_SHA",
    )
    require(
        manifest["scientific_model_forward_count"] == 48000,
        "RAW_MANIFEST_FORWARD_COUNT",
    )
    require(
        manifest["xg1_response_observed"] is True,
        "RAW_RESPONSE_NOT_OBSERVED",
    )
    require(
        manifest["inferential_test_performed"] is False
        and manifest["primary_p_value_computed"] is False,
        "RAW_MANIFEST_INFERENCE",
    )
    require(
        manifest["response_based_control_selection"] is False
        and manifest["second_adjacent_site_executed"] is False
        and manifest["rescue_performed"] is False,
        "RAW_MANIFEST_BOUNDARY",
    )

    require(
        adjacent_geometry["result"]
        == "PASS_MAMBA14B_ADJACENT_GEOMETRY_PREPARATION",
        "ADJACENT_GEOMETRY_RESULT",
    )
    require(
        adjacent_geometry["adjacent_triplet"] == [34, 35, 36],
        "ADJACENT_GEOMETRY_TRIPLET",
    )
    require(
        adjacent_geometry["strong_mask"]["strong_count"] == 1205,
        "ADJACENT_GEOMETRY_DIM",
    )
    require(
        adjacent_geometry["fixed_causal_candidate"] == "P5"
        and adjacent_geometry["response_blind_control_plane"] == "P4",
        "ADJACENT_GEOMETRY_PLANES",
    )
    require(
        adjacent_geometry["control_selection_uses_response"] is False,
        "ADJACENT_GEOMETRY_CONTROL_RESPONSE",
    )

    d_can: list[float] = []
    d_adj: list[float] = []
    s_values: list[float] = []

    for index, item in enumerate(items):
        pair = PAIR_IDS[index]
        require(item["pair_index"] == index, f"PAIR_INDEX:{pair}")
        require(item["epsilon"] == 0.025, f"EPSILON:{pair}")
        require(
            item["selected_causal_candidate"] == "P5",
            f"P5:{pair}",
        )
        require(
            item["canonical_response_blind_control_plane"] == "P4"
            and item["adjacent_response_blind_control_plane"] == "P4",
            f"CONTROL:{pair}",
        )
        require(
            item["canonical_triplet"] == [33, 34, 35]
            and item["adjacent_triplet"] == [34, 35, 36],
            f"TRIPLET:{pair}",
        )
        require(
            item["scientific_model_forward_count_this_run"] == 160,
            f"PAIR_FORWARD_COUNT:{pair}",
        )
        require(
            item["inferential_test_performed"] is False
            and item["primary_p_value_computed"] is False
            and item["selection_reopened"] is False
            and item["second_adjacent_site_executed"] is False
            and item["layer_sweep_executed"] is False
            and item["epsilon_sweep_executed"] is False
            and item["token_sweep_executed"] is False
            and item["rescue_performed"] is False,
            f"PAIR_BOUNDARY:{pair}",
        )

        can = finite(item["D_CAN"], f"D_CAN:{pair}")
        adj = finite(item["D_ADJ"], f"D_ADJ:{pair}")
        s = finite(item["S"], f"S:{pair}")

        require(
            abs(
                can
                - (
                    finite(
                        item["canonical"]["Q_restored"],
                        f"CAN_Q_RESTORED:{pair}",
                    )
                    - finite(
                        item["canonical"]["Q_control"],
                        f"CAN_Q_CONTROL:{pair}",
                    )
                )
            )
            <= 1.0e-18,
            f"D_CAN_ARITHMETIC:{pair}",
        )
        require(
            abs(
                adj
                - (
                    finite(
                        item["adjacent"]["Q_restored"],
                        f"ADJ_Q_RESTORED:{pair}",
                    )
                    - finite(
                        item["adjacent"]["Q_control"],
                        f"ADJ_Q_CONTROL:{pair}",
                    )
                )
            )
            <= 1.0e-18,
            f"D_ADJ_ARITHMETIC:{pair}",
        )
        require(
            abs(s - (can - adj)) <= 1.0e-18,
            f"S_ARITHMETIC:{pair}",
        )

        d_can.append(can)
        d_adj.append(adj)
        s_values.append(s)

    raw_desc = summary["raw_descriptive_only"]
    require(
        abs(
            math.fsum(d_can) / PAIR_COUNT
            - finite(raw_desc["mean_D_CAN"], "RAW_MEAN_D_CAN")
        )
        <= 1.0e-20,
        "RAW_MEAN_D_CAN_RECONSTRUCTION",
    )
    require(
        abs(
            math.fsum(d_adj) / PAIR_COUNT
            - finite(raw_desc["mean_D_ADJ"], "RAW_MEAN_D_ADJ")
        )
        <= 1.0e-20,
        "RAW_MEAN_D_ADJ_RECONSTRUCTION",
    )
    require(
        abs(
            math.fsum(s_values) / PAIR_COUNT
            - finite(raw_desc["mean_S"], "RAW_MEAN_S")
        )
        <= 1.0e-20,
        "RAW_MEAN_S_RECONSTRUCTION",
    )

    return {
        "items": items,
        "summary": summary,
        "manifest": manifest,
        "adjacent_geometry": adjacent_geometry,
        "D_CAN": d_can,
        "D_ADJ": d_adj,
        "S": s_values,
    }


def analyze(
    frozen: Mapping[str, Any],
    *,
    analysis_head: str,
) -> dict[str, Any]:
    d_can_stats = descriptive(frozen["D_CAN"])
    d_adj_stats = descriptive(frozen["D_ADJ"])
    s_stats = descriptive(frozen["S"])

    primary = primary_one_sample_greater(
        frozen["S"],
        s_stats,
    )

    canonical_sign_gate = bool(
        finite(d_can_stats["mean"], "D_CAN_MEAN") > 0.0
    )
    paired_specificity_gate = bool(
        finite(s_stats["mean"], "S_MEAN") > 0.0
        and finite(primary["p_value"], "PRIMARY_P_VALUE") < ALPHA
    )
    supported = bool(
        canonical_sign_gate
        and paired_specificity_gate
    )

    geometry = frozen["adjacent_geometry"]
    lambda_plus = {
        f"P{index}": finite(value, f"LAMBDA:P{index}")
        for index, value in enumerate(
            geometry["lambda_plus_by_plane"],
            1,
        )
    }

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "result": RESULT_PASS,
        "analysis_head": analysis_head,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "raw_run_name": RAW_RUN_NAME,
        "pair_first": PAIR_IDS[0],
        "pair_last": PAIR_IDS[-1],
        "pair_count": PAIR_COUNT,
        "selected_causal_candidate": "P5",
        "canonical_response_blind_control_plane": "P4",
        "adjacent_response_blind_control_plane": "P4",
        "canonical_triplet": [33, 34, 35],
        "adjacent_triplet": [34, 35, 36],
        "epsilon": 0.025,
        "endpoint_definitions": {
            "D_CAN":
                "Q_restored,canonical(P5)-Q_control,canonical(P4)",
            "D_ADJ":
                "Q_restored,adjacent(P5)-Q_control,adjacent(P4)",
            "S": "D_CAN-D_ADJ",
        },
        "descriptive": {
            "D_CAN": d_can_stats,
            "D_ADJ": d_adj_stats,
            "S": s_stats,
        },
        "primary_inference": primary,
        "decision_gates": {
            "canonical_sign_gate": {
                "rule": "mean(D_CAN) > 0",
                "passed": canonical_sign_gate,
                "adds_p_value": False,
            },
            "paired_specificity_gate": {
                "rule": "mean(S) > 0 and one-sided Student p < 0.05",
                "passed": paired_specificity_gate,
            },
            "both_required": True,
            "supported": supported,
        },
        "scientific_conclusion":
            SUCCESS_LABEL if supported else FAILURE_LABEL,
        "adjacent_geometry": {
            "strong_dimension": 1205,
            "strong_index_sha256":
                geometry["strong_mask"]["strong_index_sha256"],
            "lambda_plus_by_plane": lambda_plus,
            "response_blind_control_plane": "P4",
            "control_selection_uses_xg1_response": False,
        },
        "q_component_descriptive":
            aggregate_q_components(frozen["items"]),
        "intervention_audit_maxima":
            aggregate_audit_maxima(frozen["items"]),
        "forward_accounting": {
            "adjacent_geometry_model_forward_count": 2400,
            "canonical_geometry_rerun_count": 0,
            "canonical_response_model_forward_count": 24000,
            "adjacent_response_model_forward_count": 24000,
            "paired_response_model_forward_count": 48000,
            "total_new_scientific_model_forward_count": 50400,
            "analysis_model_forward_count": 0,
        },
        "inferential_accounting": {
            "primary_p_value_count": 1,
            "additional_p_value_count": 0,
            "multiplicity_correction": "none_single_test_family",
            "canonical_sign_gate_adds_p_value": False,
        },
        "boundaries": {
            "analysis_cpu_only": True,
            "model_execution_performed": False,
            "training_executed": False,
            "backward_executed": False,
            "selection_reopened": False,
            "response_based_control_selection": False,
            "second_adjacent_site_executed": False,
            "layer_sweep_executed": False,
            "token_sweep_executed": False,
            "epsilon_sweep_executed": False,
            "alternative_tail_executed": False,
            "additional_primary_endpoint_executed": False,
            "row_subset_rescue_executed": False,
            "rescue_performed": False,
            "experiments_1_to_3_rescued": False,
        },
        "claim_boundary": {
            "if_supported": (
                "On the prospectively frozen Mamba-1.4B XG1 "
                "5101..5400 cohort, the canonical (33,34,35) homologous "
                "site carried a stronger positive rank-aligned P5 core "
                "signal than the single architecture-predeclared adjacent "
                "+1 site (34,35,36) under the matched frozen measurement "
                "procedure."
            ),
            "does_not_establish": [
                "uniqueness across all layers",
                "a global layer optimum",
                "absence of causal signal at every other layer",
                "semantic identity of P5 across sites",
                "architectural universality",
                "improved downstream task behavior",
                "useful steering",
                (
                    "external natural-language transfer beyond already "
                    "frozen results"
                ),
            ],
        },
    }


def render_report(analysis: Mapping[str, Any]) -> str:
    desc = analysis["descriptive"]
    primary = analysis["primary_inference"]
    gates = analysis["decision_gates"]
    geometry = analysis["adjacent_geometry"]

    def fmt(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, float):
            return format(value, ".17g")
        return str(value)

    lines = [
        "# ContraMamba Gen4 Experiment 5 — One-Shot Adjacent-Site Specificity",
        "",
        f"Result: `{analysis['result']}`",
        "",
        f"Scientific conclusion: `{analysis['scientific_conclusion']}`",
        "",
        "## Frozen test",
        "",
        "- Cohort: `xg1_fact_5101..xg1_fact_5400`, N=300.",
        "- Canonical site: `(33,34,35)`.",
        "- Adjacent site: `(34,35,36)`.",
        "- Fixed rank-aligned causal candidate: `P5`.",
        "- Canonical control: `P4`.",
        "- Adjacent geometry-only control: `P4`.",
        "- epsilon: `0.025`.",
        "- Primary endpoint: `S = D_CAN - D_ADJ`.",
        "- Exactly one inferential p-value was computed.",
        "",
        "## Descriptive results",
        "",
        "| Endpoint | Mean | SD | 95% t-CI | Cohen dz | Positive fraction |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]

    for key in ("D_CAN", "D_ADJ", "S"):
        row = desc[key]
        lines.append(
            "| "
            + key
            + " | "
            + fmt(row["mean"])
            + " | "
            + fmt(row["sd"])
            + " | ["
            + fmt(row["ci95_t_low"])
            + ", "
            + fmt(row["ci95_t_high"])
            + "] | "
            + fmt(row["cohen_dz"])
            + " | "
            + fmt(row["fraction_positive"])
            + " |"
        )

    lines.extend([
        "",
        "## Primary inference",
        "",
        "- Test: one-sample Student t-test on paired `S`.",
        "- Null: `E[S] <= 0`.",
        "- Alternative: `E[S] > 0`.",
        "- Tail: one-sided greater.",
        f"- t({primary['df']}) = {fmt(primary['t_statistic'])}.",
        f"- p = {fmt(primary['p_value'])}.",
        "- Multiplicity correction: none; the inferential family contains one test.",
        "",
        "## Decision gates",
        "",
        (
            "- Fresh canonical sign gate `mean(D_CAN)>0`: "
            + ("PASS" if gates["canonical_sign_gate"]["passed"] else "FAIL")
            + "."
        ),
        (
            "- Paired specificity gate `mean(S)>0` and `p<0.05`: "
            + ("PASS" if gates["paired_specificity_gate"]["passed"] else "FAIL")
            + "."
        ),
        "",
        "## Adjacent geometry",
        "",
        f"- Strong dimension: `{geometry['strong_dimension']}`.",
        (
            "- Strong-index SHA256: `"
            + str(geometry["strong_index_sha256"])
            + "`."
        ),
        "- Geometry-only response-blind control: `P4`.",
    ])

    for plane, value in geometry["lambda_plus_by_plane"].items():
        lines.append(f"- `{plane}` lambda_plus: `{fmt(value)}`.")

    lines.extend([
        "",
        "## Forward accounting",
        "",
        "- Adjacent geometry: `2400` new scientific model forwards.",
        "- Canonical geometry reruns: `0`.",
        "- Canonical fresh response: `24000` forwards.",
        "- Adjacent fresh response: `24000` forwards.",
        "- Paired response total: `48000` forwards.",
        "- Experiment 5 total new scientific model forwards: `50400`.",
        "- Static analysis model forwards: `0`.",
        "",
        "## Claim boundary",
        "",
    ])

    if analysis["scientific_conclusion"] == SUCCESS_LABEL:
        lines.append(
            analysis["claim_boundary"]["if_supported"]
        )
    else:
        lines.append(
            "The one-shot adjacent-site specificity claim is not established "
            "under the frozen decision rule."
        )

    lines.extend([
        "",
        "This experiment does not rescue Experiments 1–3 and does not establish "
        "a global layer optimum or uniqueness across all layers.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    analysis_path = output_dir / ANALYSIS_FILE
    report_path = output_dir / REPORT_FILE
    sums_path = output_dir / SUMS_FILE

    analysis_path.write_bytes(pretty_json_bytes(analysis))
    report_path.write_text(
        render_report(analysis),
        encoding="utf-8",
        newline="\n",
    )

    hashes = {
        ANALYSIS_FILE: sha256_file(analysis_path),
        REPORT_FILE: sha256_file(report_path),
    }
    sums_path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only final inference for the frozen Mamba-1.4B one-shot "
            "adjacent-site specificity experiment. Computes exactly one "
            "inferential p-value: one-sided one-sample Student t-test on S."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    authenticate_repo(args.expected_head)
    frozen = validate_and_load_inputs()
    analysis = analyze(
        frozen,
        analysis_head=args.expected_head,
    )
    write_outputs(
        output_dir=args.output_dir,
        analysis=analysis,
    )

    primary = analysis["primary_inference"]
    desc = analysis["descriptive"]
    gates = analysis["decision_gates"]

    print("RESULT=" + analysis["result"])
    print(
        "SCIENTIFIC_CONCLUSION="
        + analysis["scientific_conclusion"]
    )
    print("PAIR_COUNT=300")
    print(
        "MEAN_D_CAN="
        + format(float(desc["D_CAN"]["mean"]), ".17g")
    )
    print(
        "MEAN_D_ADJ="
        + format(float(desc["D_ADJ"]["mean"]), ".17g")
    )
    print(
        "MEAN_S="
        + format(float(desc["S"]["mean"]), ".17g")
    )
    print(
        "SD_S="
        + format(float(desc["S"]["sd"]), ".17g")
    )
    print(
        "T_STATISTIC="
        + format(float(primary["t_statistic"]), ".17g")
    )
    print(
        "PRIMARY_P_VALUE="
        + format(float(primary["p_value"]), ".17g")
    )
    print("PRIMARY_P_VALUE_COUNT=1")
    print(
        "CANONICAL_SIGN_GATE="
        + ("PASS" if gates["canonical_sign_gate"]["passed"] else "FAIL")
    )
    print(
        "PAIRED_SPECIFICITY_GATE="
        + ("PASS" if gates["paired_specificity_gate"]["passed"] else "FAIL")
    )
    print("ADDITIONAL_P_VALUE_COUNT=0")
    print("ANALYSIS_MODEL_FORWARD_COUNT=0")
    print("TOTAL_NEW_EXPERIMENT5_MODEL_FORWARDS=50400")
    print("SECOND_ADJACENT_SITE_EXECUTED=False")
    print("RESCUE_PERFORMED=False")
    print("EXPERIMENTS_1_TO_3_RESCUED=False")


if __name__ == "__main__":
    main()
