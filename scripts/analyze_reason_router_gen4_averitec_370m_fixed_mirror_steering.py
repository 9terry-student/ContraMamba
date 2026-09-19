#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "f6209a5721a16519148880e8a7f0a7734d34c787"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_causal_atlas_guided_steering_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "0d35f9aebb37e6203bea40304d3a55ab2361b684"

SOURCE_ELIGIBILITY_CORRECTION = Path(
    "reports/reason_router_gen4_causal_atlas_guided_steering_source_eligibility_correction.md"
)
SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB = (
    "3a55da349d7c9780f8e51de76e3fd9abd67531aa"
)

RAW_RUN_NAME = (
    "g4k-averitec370-fixed-mirror-steering-raw-2799-25a626d"
)
RAW_ROOT = Path(
    "reports/reason_router_gen4_averitec_370m_fixed_mirror_steering_runs"
) / RAW_RUN_NAME

RAW_ROWS_FILE = RAW_ROOT / "steering_raw_rows.jsonl"
RAW_SUMMARY_FILE = RAW_ROOT / "execution_summary.json"
RAW_SUMS_FILE = RAW_ROOT / "SHA256SUMS.txt"

RAW_FREEZE_COMMIT = "f6209a5721a16519148880e8a7f0a7734d34c787"
RAW_EXECUTION_HEAD = "25a626dd52592c42318c80e14aa13c18b74fded9"

RAW_ROWS_GIT_BLOB = "52e07a7c93f96eb6d10f7b87e38a0aee7fb6ac2d"
RAW_SUMMARY_GIT_BLOB = "b35b4ed1aa5b53e896abd45fb8f22cb273c6660f"
RAW_SUMS_GIT_BLOB = "6c5d1825378f0207c0bd560e968b9c9360cb093d"

RAW_ROWS_SHA256 = (
    "912fb91bdfc5e778e4d2f99cb9a8f7a5bdb8a4a1389b39de1d47947eb9058d0c"
)
RAW_SUMMARY_SHA256 = (
    "ad9497956b8db7b9d3930e13437d7d6daf5c33e3b800999e0d3ee2787cca4f7e"
)
RAW_SUMS_SHA256 = (
    "372d243e5bb6acc1f0e4ad1309fb446f474e813acbdedb9466fbd56940614465"
)

N = 2799
CONDITIONS = (
    "native",
    "p3_mirror_steer",
    "p5_matched_control",
)
RAW_ROW_COUNT = N * len(CONDITIONS)
LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")

PRIMARY_ALPHA = 0.05
HARM_THRESHOLD = 0.05

SUPPORTED_RESULT = "AVERITEC_370M_FIXED_MIRROR_P3_STEERING_SUPPORTED"
NOT_ESTABLISHED_RESULT = (
    "AVERITEC_370M_FIXED_MIRROR_P3_STEERING_NOT_ESTABLISHED"
)

ANALYSIS_SCHEMA = "gen4-averitec-370m-fixed-mirror-steering-analysis-v1"
ANALYSIS_FILE = "steering_analysis.json"
REPORT_FILE = "steering_analysis.md"
CHECKSUM_FILE = "SHA256SUMS.txt"


class SteeringAnalysisError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SteeringAnalysisError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SteeringAnalysisError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    require(path.is_file(), f"JSONL_MISSING:{path}")
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    frozen_blobs = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        SOURCE_ELIGIBILITY_CORRECTION.as_posix():
            SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB,
        RAW_ROWS_FILE.as_posix(): RAW_ROWS_GIT_BLOB,
        RAW_SUMMARY_FILE.as_posix(): RAW_SUMMARY_GIT_BLOB,
        RAW_SUMS_FILE.as_posix(): RAW_SUMS_GIT_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(N == 2799, "N")
    require(RAW_ROW_COUNT == 8397, "RAW_ROW_COUNT")
    require(
        CONDITIONS
        == (
            "native",
            "p3_mirror_steer",
            "p5_matched_control",
        ),
        "CONDITIONS",
    )
    require(LABELS == ("REFUTE", "NOT_ENTITLED", "SUPPORT"), "LABELS")
    require(PRIMARY_ALPHA == 0.05, "PRIMARY_ALPHA")
    require(HARM_THRESHOLD == 0.05, "HARM_THRESHOLD")
    require(RAW_FREEZE_COMMIT == REQUIRED_ANCESTOR, "RAW_FREEZE_ANCESTOR")


def validate_raw_artifacts() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    rows_path = ROOT / RAW_ROWS_FILE
    summary_path = ROOT / RAW_SUMMARY_FILE
    sums_path = ROOT / RAW_SUMS_FILE

    require(sha256_file(rows_path) == RAW_ROWS_SHA256, "RAW_ROWS_SHA256")
    require(
        sha256_file(summary_path) == RAW_SUMMARY_SHA256,
        "RAW_SUMMARY_SHA256",
    )
    require(sha256_file(sums_path) == RAW_SUMS_SHA256, "RAW_SUMS_SHA256")

    declared: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in declared, "RAW_SUMS_DUPLICATE")
        declared[name] = digest

    require(
        declared
        == {
            "execution_summary.json": RAW_SUMMARY_SHA256,
            "steering_raw_rows.jsonl": RAW_ROWS_SHA256,
        },
        "RAW_SUMS_CONTENT",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(
        summary["schema_version"]
        == "gen4-averitec-370m-fixed-mirror-steering-raw-summary-v1",
        "RAW_SUMMARY_SCHEMA",
    )
    require(
        summary["result"]
        == "PASS_AVERITEC_370M_FIXED_MIRROR_STEERING_RAW",
        "RAW_SUMMARY_RESULT",
    )
    require(
        summary["execution_head"] == RAW_EXECUTION_HEAD,
        "RAW_EXECUTION_HEAD",
    )
    require(summary["fresh_cohort_count"] == N, "RAW_COHORT_N")
    require(summary["raw_row_count"] == RAW_ROW_COUNT, "RAW_SUMMARY_N")
    require(
        summary["full_model_forward_budget"] == RAW_ROW_COUNT,
        "RAW_FORWARD_BUDGET",
    )
    require(tuple(summary["conditions"]) == CONDITIONS, "RAW_CONDITIONS")
    require(summary["primary_inference_executed"] is False, "RAW_INFERENCE")
    require(summary["p_value_count_added"] == 0, "RAW_PVALUE")
    require(summary["scientific_conclusion"] is None, "RAW_CONCLUSION")
    require(
        summary["response_based_selection_performed"] is False,
        "RAW_RESPONSE_SELECTION",
    )
    require(summary["rescue_performed"] is False, "RAW_RESCUE")
    require(summary["magnitude_search_performed"] is False, "RAW_MAG_SEARCH")
    require(summary["sign_search_performed"] is False, "RAW_SIGN_SEARCH")
    require(summary["layer_search_performed"] is False, "RAW_LAYER_SEARCH")
    require(summary["token_search_performed"] is False, "RAW_TOKEN_SEARCH")

    rows = read_jsonl(rows_path)
    require(len(rows) == RAW_ROW_COUNT, "RAW_ROWS_N")

    expected_order: list[tuple[str, str]] = []
    seen_example_order: list[str] = []
    by_id: dict[str, list[dict[str, Any]]] = {}

    for index, row in enumerate(rows):
        require(
            row["schema_version"]
            == "gen4-averitec-370m-fixed-mirror-steering-raw-row-v1",
            f"ROW_SCHEMA:{index}",
        )
        require(row["condition"] in CONDITIONS, f"ROW_CONDITION:{index}")
        require(row["scale"] == "mamba370m", f"ROW_SCALE:{index}")
        require(
            row["selected_plane"] == "P3"
            and row["response_blind_control_plane"] == "P5",
            f"ROW_PLANES:{index}",
        )
        require(int(row["intervention_layer"]) == 35, f"ROW_LAYER:{index}")
        require(
            row["anchor_name"] == "A_CLAIM_EVIDENCE_BOUNDARY",
            f"ROW_ANCHOR:{index}",
        )
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + 2,
            f"ROW_TARGET:{index}",
        )
        require(int(row["prediction_id"]) in (0, 1, 2), f"ROW_PRED:{index}")
        require(
            str(row["prediction"]) in LABELS,
            f"ROW_PRED_LABEL:{index}",
        )
        require(
            int(row["correct_label_id"]) in (0, 1, 2),
            f"ROW_GOLD:{index}",
        )
        require(
            str(row["correct_label"]) in LABELS,
            f"ROW_GOLD_LABEL:{index}",
        )

        logits = [float(x) for x in row["final_logits"]]
        require(
            len(logits) == 3 and all(math.isfinite(x) for x in logits),
            f"ROW_LOGITS:{index}",
        )
        prediction_id = max(range(3), key=lambda j: logits[j])
        require(
            prediction_id == int(row["prediction_id"]),
            f"ROW_PRED_LOGIT:{index}",
        )

        gold_id = int(row["correct_label_id"])
        expected_margin = (
            logits[gold_id]
            - max(logits[j] for j in range(3) if j != gold_id)
        )
        require(
            math.isclose(
                float(row["correct_class_logit_margin"]),
                expected_margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"ROW_MARGIN:{index}",
        )
        require(
            bool(row["is_correct"]) == (prediction_id == gold_id),
            f"ROW_CORRECT:{index}",
        )

        example_id = str(row["example_id"])
        if not seen_example_order or seen_example_order[-1] != example_id:
            seen_example_order.append(example_id)

        by_id.setdefault(example_id, []).append(row)

    require(len(by_id) == N, "RAW_EXAMPLE_N")
    require(len(seen_example_order) == N, "RAW_EXAMPLE_ORDER_N")

    for example_id in seen_example_order:
        group = by_id[example_id]
        require(len(group) == 3, f"GROUP_N:{example_id}")
        require(
            tuple(str(row["condition"]) for row in group) == CONDITIONS,
            f"GROUP_ORDER:{example_id}",
        )
        gold_ids = {int(row["correct_label_id"]) for row in group}
        gold_labels = {str(row["correct_label"]) for row in group}
        source_indices = {int(row["averitec_train_index"]) for row in group}
        require(len(gold_ids) == 1, f"GROUP_GOLD_ID:{example_id}")
        require(len(gold_labels) == 1, f"GROUP_GOLD_LABEL:{example_id}")
        require(len(source_indices) == 1, f"GROUP_SOURCE:{example_id}")

        expected_order.extend(
            (example_id, condition)
            for condition in CONDITIONS
        )

    require(
        [
            (str(row["example_id"]), str(row["condition"]))
            for row in rows
        ] == expected_order,
        "RAW_GLOBAL_ORDER",
    )

    return rows, summary


def exact_one_sided_binomial_upper(
    corrections: int,
    damages: int,
) -> dict[str, Any]:
    require(corrections >= 0 and damages >= 0, "BINOMIAL_COUNTS")
    n = corrections + damages
    require(n > 0, "BINOMIAL_NOT_ESTIMABLE")

    numerator = sum(
        math.comb(n, k)
        for k in range(corrections, n + 1)
    )
    denominator = 1 << n
    exact = Fraction(numerator, denominator)

    p_value = float(exact)
    require(0.0 <= p_value <= 1.0, "BINOMIAL_P_RANGE")

    return {
        "test": "one_sided_exact_mcnemar_binomial",
        "alternative": "correction_probability_greater_than_0.5",
        "null_boundary_probability": 0.5,
        "discordant_count": n,
        "correction_count": corrections,
        "damage_count": damages,
        "exact_tail_numerator": str(exact.numerator),
        "exact_tail_denominator": str(exact.denominator),
        "p_value": p_value,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    require(numerator >= 0 and denominator >= 0, "RATE_COUNTS")
    require(numerator <= denominator, "RATE_ORDER")
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    require(all(math.isfinite(float(x)) for x in values), "MEAN_NONFINITE")
    return float(statistics.fmean(float(x) for x in values))


def _transition_table(
    pairs: Sequence[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    table = {
        source: {target: 0 for target in LABELS}
        for source in LABELS
    }
    for source, target in pairs:
        require(source in LABELS and target in LABELS, "TRANSITION_LABEL")
        table[source][target] += 1
    return table


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_head: str,
) -> dict[str, Any]:
    require(len(rows) == RAW_ROW_COUNT, "ANALYZE_RAW_N")

    grouped: list[tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
    ]] = []
    for start in range(0, len(rows), 3):
        group = rows[start:start + 3]
        require(len(group) == 3, "ANALYZE_GROUP_N")
        require(
            tuple(str(row["condition"]) for row in group) == CONDITIONS,
            "ANALYZE_CONDITION_ORDER",
        )
        require(
            len({str(row["example_id"]) for row in group}) == 1,
            "ANALYZE_EXAMPLE_ID",
        )
        grouped.append((group[0], group[1], group[2]))

    require(len(grouped) == N, "ANALYZE_EXAMPLE_N")

    native_correct_count = 0
    native_incorrect_count = 0
    steer_correct_count = 0
    control_correct_count = 0

    corrections = 0
    damages = 0

    label_corrections = {label: 0 for label in LABELS}
    label_damages = {label: 0 for label in LABELS}

    target_margin_changes: list[float] = []
    preservation_margin_changes: list[float] = []
    all_steer_margin_changes: list[float] = []
    native_minus_control_margins: list[float] = []

    steer_margin_exceeds_native_count = 0
    preferred_margin_order_count = 0

    native_to_steer: list[tuple[str, str]] = []
    native_to_control: list[tuple[str, str]] = []

    for native, steer, control in grouped:
        native_correct = bool(native["is_correct"])
        steer_correct = bool(steer["is_correct"])
        control_correct = bool(control["is_correct"])

        native_correct_count += int(native_correct)
        native_incorrect_count += int(not native_correct)
        steer_correct_count += int(steer_correct)
        control_correct_count += int(control_correct)

        gold = str(native["correct_label"])
        require(gold in LABELS, "ANALYZE_GOLD")

        if (not native_correct) and steer_correct:
            corrections += 1
            label_corrections[gold] += 1
        if native_correct and (not steer_correct):
            damages += 1
            label_damages[gold] += 1

        m_native = float(native["correct_class_logit_margin"])
        m_steer = float(steer["correct_class_logit_margin"])
        m_control = float(control["correct_class_logit_margin"])
        require(
            all(math.isfinite(x) for x in (m_native, m_steer, m_control)),
            "ANALYZE_MARGIN_NONFINITE",
        )

        delta_steer = m_steer - m_native
        delta_control = m_native - m_control

        all_steer_margin_changes.append(delta_steer)
        native_minus_control_margins.append(delta_control)

        if native_correct:
            preservation_margin_changes.append(delta_steer)
        else:
            target_margin_changes.append(delta_steer)

        steer_margin_exceeds_native_count += int(m_steer > m_native)
        preferred_margin_order_count += int(
            m_steer > m_native > m_control
        )

        native_to_steer.append(
            (str(native["prediction"]), str(steer["prediction"]))
        )
        native_to_control.append(
            (str(native["prediction"]), str(control["prediction"]))
        )

    require(
        native_correct_count + native_incorrect_count == N,
        "NATIVE_STRATA_CLOSURE",
    )
    require(corrections <= native_incorrect_count, "CORRECTION_BOUND")
    require(damages <= native_correct_count, "DAMAGE_BOUND")

    discordant = corrections + damages
    primary_test: dict[str, Any] | None
    p_value_count: int
    if discordant == 0:
        primary_test = None
        p_value_count = 0
        primary_significance_pass = False
    else:
        primary_test = exact_one_sided_binomial_upper(
            corrections,
            damages,
        )
        p_value_count = 1
        primary_significance_pass = (
            float(primary_test["p_value"]) < PRIMARY_ALPHA
        )

    damage_rate = _rate(damages, native_correct_count)
    correction_rate = _rate(corrections, native_incorrect_count)

    harm_gate_pass = (
        damage_rate is not None
        and damage_rate <= HARM_THRESHOLD
    )
    net_utility_gate_pass = corrections > damages
    estimable = discordant > 0

    supported = (
        estimable
        and primary_significance_pass
        and net_utility_gate_pass
        and harm_gate_pass
    )
    result = SUPPORTED_RESULT if supported else NOT_ESTABLISHED_RESULT

    native_accuracy = native_correct_count / N
    steer_accuracy = steer_correct_count / N
    control_accuracy = control_correct_count / N

    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "result": result,
        "analysis_head": analysis_head,
        "raw_freeze_commit": RAW_FREEZE_COMMIT,
        "raw_execution_head": RAW_EXECUTION_HEAD,
        "input_provenance": {
            "raw_run_name": RAW_RUN_NAME,
            "raw_rows_file": RAW_ROWS_FILE.as_posix(),
            "raw_rows_sha256": RAW_ROWS_SHA256,
            "raw_rows_git_blob": RAW_ROWS_GIT_BLOB,
            "raw_summary_file": RAW_SUMMARY_FILE.as_posix(),
            "raw_summary_sha256": RAW_SUMMARY_SHA256,
            "raw_summary_git_blob": RAW_SUMMARY_GIT_BLOB,
            "raw_sums_file": RAW_SUMS_FILE.as_posix(),
            "raw_sums_sha256": RAW_SUMS_SHA256,
            "raw_sums_git_blob": RAW_SUMS_GIT_BLOB,
            "design_artifact_git_blob": DESIGN_ARTIFACT_GIT_BLOB,
            "source_eligibility_correction_git_blob":
                SOURCE_ELIGIBILITY_CORRECTION_GIT_BLOB,
        },
        "population": {
            "N": N,
            "native_correct_count": native_correct_count,
            "native_incorrect_count": native_incorrect_count,
        },
        "primary_endpoint": {
            "definition": (
                "one-sided exact McNemar/binomial on native-vs-mirror-steer "
                "correctness discordances"
            ),
            "C_corrections": corrections,
            "D_damages": damages,
            "discordant_count": discordant,
            "estimable": estimable,
            "test": primary_test,
            "p_value_count_added": p_value_count,
            "alpha": PRIMARY_ALPHA,
            "significance_gate_pass": primary_significance_pass,
            "C_greater_than_D_gate_pass": net_utility_gate_pass,
        },
        "preservation_gate": {
            "damage_count": damages,
            "native_correct_denominator": native_correct_count,
            "damage_rate_on_native_correct": damage_rate,
            "threshold": HARM_THRESHOLD,
            "pass": harm_gate_pass,
        },
        "success_rule": {
            "discordant_positive": estimable,
            "primary_p_below_alpha": primary_significance_pass,
            "C_greater_than_D": net_utility_gate_pass,
            "damage_rate_at_most_threshold": harm_gate_pass,
            "all_four_pass": supported,
        },
        "descriptive_diagnostics": {
            "accuracy": {
                "native": native_accuracy,
                "p3_mirror_steer": steer_accuracy,
                "p5_matched_control": control_accuracy,
            },
            "correction": {
                "count": corrections,
                "native_error_denominator": native_incorrect_count,
                "rate_on_native_errors": correction_rate,
            },
            "damage": {
                "count": damages,
                "native_correct_denominator": native_correct_count,
                "rate_on_native_correct": damage_rate,
            },
            "net_accuracy_change": steer_accuracy - native_accuracy,
            "net_correct_count_change":
                steer_correct_count - native_correct_count,
            "mapped_label_specific": {
                label: {
                    "correction_count": label_corrections[label],
                    "damage_count": label_damages[label],
                }
                for label in LABELS
            },
            "correct_class_margin": {
                "mean_steer_minus_native_all":
                    _mean(all_steer_margin_changes),
                "mean_steer_minus_native_target_native_incorrect":
                    _mean(target_margin_changes),
                "mean_steer_minus_native_preservation_native_correct":
                    _mean(preservation_margin_changes),
                "mean_native_minus_matched_control":
                    _mean(native_minus_control_margins),
                "steer_margin_exceeds_native_count":
                    steer_margin_exceeds_native_count,
                "steer_margin_exceeds_native_fraction":
                    steer_margin_exceeds_native_count / N,
                "preferred_Msteer_gt_Mnative_gt_Mcontrol_count":
                    preferred_margin_order_count,
                "preferred_Msteer_gt_Mnative_gt_Mcontrol_fraction":
                    preferred_margin_order_count / N,
            },
            "prediction_transition_table": {
                "native_to_p3_mirror_steer":
                    _transition_table(native_to_steer),
                "native_to_p5_matched_control":
                    _transition_table(native_to_control),
            },
        },
        "analysis_execution": {
            "model_forward_count": 0,
            "cuda_executed": False,
            "training_executed": False,
            "backward_executed": False,
            "response_based_selection_performed": False,
            "rescue_performed": False,
            "magnitude_search_performed": False,
            "sign_search_performed": False,
            "layer_search_performed": False,
            "token_search_performed": False,
            "additional_p_value_count": 0,
        },
        "claim_boundary": {
            "positive_claim": (
                "On a fresh deduplicated AVeriTeC train-derived three-class "
                "gold-evidence cohort, the frozen Mamba-370M P3 causal atlas "
                "supported a pre-specified mirror intervention that produced "
                "more corrections than damages under the preregistered "
                "preservation gate."
            ),
            "negative_interpretation": (
                "The specific fixed mirror extrapolation did not establish "
                "useful steering under the preregistered utility/harm rule."
            ),
            "does_not_establish": [
                "general benchmark superiority",
                "four-class AVeriTeC performance",
                "retrieval competence",
                "free-form hallucination prevention",
                "a pre-emission precursor",
                "cross-scale steering",
                "universal P3 rank identity",
                "optimal steering magnitude",
            ],
        },
    }

    require(
        analysis["primary_endpoint"]["p_value_count_added"]
        in (0, 1),
        "PRIMARY_PVALUE_COUNT",
    )
    require(
        analysis["analysis_execution"]["additional_p_value_count"] == 0,
        "ADDITIONAL_PVALUES",
    )
    return analysis


def render_report(analysis: Mapping[str, Any]) -> str:
    primary = analysis["primary_endpoint"]
    preservation = analysis["preservation_gate"]
    diagnostics = analysis["descriptive_diagnostics"]
    accuracy = diagnostics["accuracy"]
    margin = diagnostics["correct_class_margin"]
    rule = analysis["success_rule"]

    if primary["estimable"]:
        p_line = (
            f"- one-sided exact p-value: `{primary['test']['p_value']:.17g}`\n"
            f"- exact tail fraction: "
            f"`{primary['test']['exact_tail_numerator']}/"
            f"{primary['test']['exact_tail_denominator']}`"
        )
    else:
        p_line = "- one-sided exact p-value: `NOT_ESTIMABLE` (`C + D = 0`)"

    if analysis["result"] == SUPPORTED_RESULT:
        interpretation = analysis["claim_boundary"]["positive_claim"]
    else:
        interpretation = analysis["claim_boundary"]["negative_interpretation"]

    lines = [
        "# Gen4 AVeriTeC 370M Fixed Mirror Steering — Frozen Analysis",
        "",
        "## Result",
        "",
        f"`{analysis['result']}`",
        "",
        "## Primary endpoint",
        "",
        f"- N: `{analysis['population']['N']}`",
        f"- native-correct: `{analysis['population']['native_correct_count']}`",
        f"- native-incorrect: `{analysis['population']['native_incorrect_count']}`",
        f"- C (native-wrong → steer-correct): `{primary['C_corrections']}`",
        f"- D (native-correct → steer-wrong): `{primary['D_damages']}`",
        f"- discordant C+D: `{primary['discordant_count']}`",
        p_line,
        f"- alpha: `{primary['alpha']}`",
        "",
        "## Preservation and success gates",
        "",
        (
            "- damage rate on native-correct: "
            f"`{preservation['damage_rate_on_native_correct']}`"
        ),
        f"- fixed damage threshold: `{preservation['threshold']}`",
        f"- C + D > 0: `{rule['discordant_positive']}`",
        f"- primary p < alpha: `{rule['primary_p_below_alpha']}`",
        f"- C > D: `{rule['C_greater_than_D']}`",
        (
            "- damage rate <= threshold: "
            f"`{rule['damage_rate_at_most_threshold']}`"
        ),
        f"- all four gates pass: `{rule['all_four_pass']}`",
        "",
        "## Required descriptive diagnostics",
        "",
        f"- native accuracy: `{accuracy['native']}`",
        f"- mirror-steer accuracy: `{accuracy['p3_mirror_steer']}`",
        f"- matched-control accuracy: `{accuracy['p5_matched_control']}`",
        (
            "- correction rate on native errors: "
            f"`{diagnostics['correction']['rate_on_native_errors']}`"
        ),
        (
            "- damage rate on native-correct: "
            f"`{diagnostics['damage']['rate_on_native_correct']}`"
        ),
        f"- net accuracy change: `{diagnostics['net_accuracy_change']}`",
        (
            "- mean correct-class margin change, all rows "
            f"(steer-native): `{margin['mean_steer_minus_native_all']}`"
        ),
        (
            "- mean margin change, target/native-incorrect: "
            f"`{margin['mean_steer_minus_native_target_native_incorrect']}`"
        ),
        (
            "- mean margin change, preservation/native-correct: "
            f"`{margin['mean_steer_minus_native_preservation_native_correct']}`"
        ),
        (
            "- mean matched-control contrast (native-control): "
            f"`{margin['mean_native_minus_matched_control']}`"
        ),
        (
            "- fraction with steer margin > native: "
            f"`{margin['steer_margin_exceeds_native_fraction']}`"
        ),
        (
            "- fraction with M_steer > M_native > M_control: "
            f"`{margin['preferred_Msteer_gt_Mnative_gt_Mcontrol_fraction']}`"
        ),
        "",
        "Mapped-label-specific correction/damage counts and full prediction "
        "transition tables are recorded in `steering_analysis.json`.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "This analysis adds exactly one primary p-value when the endpoint is "
        "estimable and adds no subgroup/control p-values. No rescue, tuning, "
        "new model forward, CUDA execution, training, or backward pass occurs.",
        "",
        "## Claim boundary",
        "",
        "This result does not establish:",
        "",
    ]
    lines.extend(
        f"- {item}"
        for item in analysis["claim_boundary"]["does_not_establish"]
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    analysis_raw = pretty_json_bytes(analysis)
    report_raw = render_report(analysis).encode("utf-8")

    (output_dir / ANALYSIS_FILE).write_bytes(analysis_raw)
    (output_dir / REPORT_FILE).write_bytes(report_raw)

    hashes = {
        ANALYSIS_FILE: hashlib.sha256(analysis_raw).hexdigest(),
        REPORT_FILE: hashlib.sha256(report_raw).hexdigest(),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_analysis(
    *,
    expected_head: str,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    rows, _summary = validate_raw_artifacts()

    analysis = analyze_rows(
        rows,
        analysis_head=expected_head,
    )
    write_outputs(
        output_dir=output_dir,
        analysis=analysis,
    )

    primary = analysis["primary_endpoint"]
    preservation = analysis["preservation_gate"]
    diagnostics = analysis["descriptive_diagnostics"]

    print("RESULT=" + str(analysis["result"]))
    print("N=2799")
    print("NATIVE_CORRECT=" + str(analysis["population"]["native_correct_count"]))
    print("NATIVE_INCORRECT=" + str(analysis["population"]["native_incorrect_count"]))
    print("C_CORRECTIONS=" + str(primary["C_corrections"]))
    print("D_DAMAGES=" + str(primary["D_damages"]))
    print("DISCORDANT=" + str(primary["discordant_count"]))
    if primary["test"] is None:
        print("PRIMARY_P_VALUE=NOT_ESTIMABLE")
    else:
        print("PRIMARY_P_VALUE=" + repr(float(primary["test"]["p_value"])))
    print("PRIMARY_P_VALUE_COUNT=" + str(primary["p_value_count_added"]))
    print(
        "DAMAGE_RATE_ON_NATIVE_CORRECT="
        + repr(preservation["damage_rate_on_native_correct"])
    )
    print(
        "NATIVE_ACCURACY="
        + repr(diagnostics["accuracy"]["native"])
    )
    print(
        "STEER_ACCURACY="
        + repr(diagnostics["accuracy"]["p3_mirror_steer"])
    )
    print(
        "CONTROL_ACCURACY="
        + repr(diagnostics["accuracy"]["p5_matched_control"])
    )
    print(
        "NET_ACCURACY_CHANGE="
        + repr(diagnostics["net_accuracy_change"])
    )
    print("ALL_FOUR_SUCCESS_GATES=" + str(analysis["success_rule"]["all_four_pass"]))
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("ADDITIONAL_P_VALUE_COUNT=0")
    return analysis


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single preregistered exact McNemar/binomial inference "
            "and required descriptive diagnostics for the frozen 370M "
            "AVeriTeC fixed mirror-steering raw artifact."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    run_analysis(
        expected_head=str(args.expected_head),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
