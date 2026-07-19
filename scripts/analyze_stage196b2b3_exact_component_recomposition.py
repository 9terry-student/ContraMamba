#!/usr/bin/env python3
"""Stage196-B2-B3 exact, inference-only component-recomposition probe.

This analyzer is deliberately artifact-only.  It statically audits the frozen
v6b_minimal source graph and refuses to approximate missing composer inputs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SEEDS = (183, 184, 185)
POSITIVE_SEEDS = (184, 185)
EPOCHS = tuple(range(1, 21))
TAIL_EPOCHS = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{seed}_{mode}" for seed in SEEDS for mode in MODES)
P0_ROWS_PER_EPOCH = 720
NATIVE_ROW_TARGET = 6 * 20 * 720
SWAP_ROW_TARGET = 16 * 20 * 2 * 7
ROW_SWAP_SUMMARY_TARGET = 16 * 2 * 7
TOLERANCE = 1e-6
MARGIN_SOURCE = "support_logit - not_entitled_logit"

B2B2_COMMIT = "85b571610c00a4a1658229051bd6d9fcfabcf408"
B2B1_COMMIT = "85f1de8f9e0393ccdca5da4bc0725d88d8f427c9"
B2A_COMMIT = "833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6"
P0_COMMIT = "e9aaff24054f1d409119b70df13b94159a34a8e4"
B1_COMMIT = "9835cbbf86d83aca0964821669e63f7f6deb1c59"
FRAMEGATE_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"

B2B2_FILES = (
    "stage196b2b2_analysis.json",
    "stage196b2b2_report.md",
    "stage196b2b2_row_path_summary.csv",
    "stage196b2b2_epoch_paired_paths.csv",
    "stage196b2b2_group_path_summary.csv",
    "stage196b2b2_event_order_summary.csv",
    "stage196b2b2_intervention_type_paths.csv",
    "stage196b2b2_contrast_summary.csv",
    "stage196b2b2_contract.csv",
)
OUTPUTS = (
    "stage196b2b3_analysis.json",
    "stage196b2b3_report.md",
    "stage196b2b3_composer_graph.csv",
    "stage196b2b3_native_reconstruction.csv",
    "stage196b2b3_component_swap_rows.csv",
    "stage196b2b3_row_swap_summary.csv",
    "stage196b2b3_group_swap_summary.csv",
    "stage196b2b3_subtype_summary.csv",
    "stage196b2b3_contract.csv",
)
P0_FIELDS = (
    "id",
    "source_row_id",
    "dev_position",
    "gold_label",
    "prediction",
    "intervention_type",
    "frame_probability",
    "predicate_coverage_probability",
    "sufficiency_probability",
    "polarity_support_margin",
    "entitlement_probability",
    "support_probability",
    "not_entitled_probability",
    "support_logit",
    "not_entitled_logit",
    "epoch",
    "training_seed",
    "frame_downstream_gradient_mode",
)

SWAP_VARIANTS = (
    "FRAME_ONLY",
    "PREDICATE_ONLY",
    "SUFFICIENCY_ONLY",
    "ENTITLEMENT_PRIMITIVES",
    "POLARITY_ONLY",
    "ENTITLEMENT_PLUS_POLARITY",
    "FULL_COMPOSER_INPUT_POSITIVE_CONTROL",
)
DIRECTIONS = (
    "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR",
    "FRAME_LOCAL_ONLY_RECIPIENT_JOINT_DONOR",
)
DECISIONS = (
    "STAGE196B2B3_ENTITLEMENT_COMPONENT_DOMINANT",
    "STAGE196B2B3_ENTITLEMENT_POLARITY_DISJUNCTIVE_EFFECT",
    "STAGE196B2B3_POLARITY_OVERRIDE_COMPONENT_CONFIRMED",
    "STAGE196B2B3_FINAL_COMPOSER_RESIDUAL_REQUIRED",
    "STAGE196B2B3_SEED_SPECIFIC_COMPONENT_EFFECT",
    "STAGE196B2B3_ADDITIONAL_COMPOSER_OBSERVABILITY_REQUIRED",
    "STAGE196B2B3_ANALYSIS_INCOMPLETE",
)
OBSERVABILITY_REQUIRED = DECISIONS[5]
INCOMPLETE = DECISIONS[6]
NEXT_STAGE = {
    DECISIONS[0]: "STAGE196B2B4_ENTITLEMENT_PATH_PRESERVATION_DESIGN",
    DECISIONS[1]: "STAGE196B2B4_ENTITLEMENT_GAIN_WITH_POLARITY_PRESERVATION_DESIGN",
    DECISIONS[2]: "STAGE196B2B4_POLARITY_PRESERVATION_MICROINTERVENTION_DESIGN",
    DECISIONS[3]: "STAGE196B2B4_FINAL_COMPOSER_RESIDUAL_LOCALIZATION",
    DECISIONS[4]: "STAGE196B2B4_NO_PROMOTION_COMPONENT_EFFECT_REPLICATION",
    OBSERVABILITY_REQUIRED: "STAGE196B2B3P0_EPOCH_COMPOSER_INPUT_OBSERVABILITY_DESIGN",
    INCOMPLETE: "STAGE196B2B3_REPAIR_ANALYSIS_INPUTS",
}
AUTHORIZED = {
    OBSERVABILITY_REQUIRED: (
        "The frozen source graph is identified, but the current epoch sidecars do not "
        "contain the actual causal inputs and state needed for exact native recomposition."
    ),
    INCOMPLETE: "No scientific interpretation is authorized because input closure failed.",
}
PROHIBITED = (
    "counterfactual logits inferred from exported scalar correlations",
    "regression, interpolation, fitted coefficients, calibration, or heuristic logit adjustment",
    "treating polarity_support_margin as the consumed polarity input",
    "treating exported entitlement_probability as a directly swappable primitive",
    "mechanistic sufficiency from a full-input positive control",
    "training, promotion, external/OOD validity, or a new decision mechanism",
)

EXPECTED_PATH_COUNTS = {
    184: {
        "recovery": {"MULTI_CHANNEL_CONFLICT": 5},
        "harm": {
            "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN": 3,
            "FRAME_ENTITLEMENT_LOSS": 2,
            "COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY": 1,
        },
    },
    185: {
        "recovery": {"FRAME_ENTITLEMENT_GAIN": 2},
        "harm": {"MULTI_CHANNEL_CONFLICT": 3},
    },
}
EXPECTED_PRIMARY_COUNTS = {
    184: {"recovery": 5, "harm": 6},
    185: {"recovery": 2, "harm": 3},
}

CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
GRAPH_H = (
    "order",
    "graph_stage",
    "symbol",
    "classification",
    "actual_causal_input",
    "consumed_by",
    "source_file",
    "source_callable",
    "source_start_line",
    "source_end_line",
    "native_expression",
    "p0_sidecar_field",
    "available_in_p0",
    "learned_parameter_required",
    "hidden_state_required",
    "availability_reason",
)
NATIVE_H = (
    "run",
    "seed",
    "gradient_mode",
    "epoch",
    "stable_row_id",
    "dev_position",
    "exported_support_logit",
    "reconstructed_support_logit",
    "support_logit_abs_error",
    "exported_refute_logit",
    "reconstructed_refute_logit",
    "refute_logit_abs_error",
    "exported_not_entitled_logit",
    "reconstructed_not_entitled_logit",
    "not_entitled_logit_abs_error",
    "exported_support_vs_ne_margin",
    "reconstructed_support_vs_ne_margin",
    "margin_abs_error",
    "exported_prediction",
    "reconstructed_prediction",
    "prediction_equal",
    "source_graph_provenance",
)
SWAP_H = (
    "stable_row_id",
    "id",
    "source_row_id",
    "dev_position",
    "seed",
    "epoch",
    "transition_role",
    "path_class",
    "subtype_audit_group",
    "intervention_type",
    "swap_direction",
    "swap_variant",
    "recipient_native_support_logit",
    "recipient_native_refute_logit",
    "recipient_native_not_entitled_logit",
    "recipient_native_prediction",
    "donor_native_support_logit",
    "donor_native_refute_logit",
    "donor_native_not_entitled_logit",
    "donor_native_prediction",
    "counterfactual_support_logit",
    "counterfactual_refute_logit",
    "counterfactual_not_entitled_logit",
    "counterfactual_prediction",
    "recipient_support_vs_ne_margin",
    "donor_support_vs_ne_margin",
    "counterfactual_support_vs_ne_margin",
    "exact_donor_logit_reproduction",
    "donor_prediction_reproduction",
    "recipient_restoration",
    "movement_toward_donor_margin",
    "component_values_before_swap",
    "component_values_after_swap",
    "derived_entitlement_before_swap",
    "derived_entitlement_after_swap",
    "source_graph_provenance",
)
ROW_SUMMARY_H = (
    "stable_row_id",
    "seed",
    "transition_role",
    "path_class",
    "subtype_audit_group",
    "intervention_type",
    "swap_direction",
    "swap_variant",
    "tail_epochs",
    "recipient_prediction_sequence",
    "donor_prediction_sequence",
    "counterfactual_prediction_sequence",
    "recipient_tail3_status",
    "donor_tail3_status",
    "counterfactual_tail3_status",
    "counterfactual_support_stability_status",
    "mean_counterfactual_margin",
    "donor_prediction_sequence_equal",
    "recipient_prediction_sequence_equal",
    "donor_role_reproduction",
    "recipient_role_restoration",
)
GROUP_H = (
    "seed",
    "transition_role",
    "swap_direction",
    "swap_variant",
    "row_count",
    "donor_role_reproduction_count",
    "donor_role_reproduction_rate",
    "recipient_restoration_count",
    "recipient_restoration_rate",
    "exact_donor_sequence_reproduction_count",
    "exact_recipient_sequence_restoration_count",
    "mean_counterfactual_margin_shift",
    "ordered_counterfactual_margin_shifts",
    "path_class_counts",
    "intervention_type_counts",
    "opposite_role_contamination_count",
    "opposite_role_contamination_rate",
)
SUBTYPE_H = (
    "subtype_audit_group",
    "seed",
    "transition_role",
    "row_count",
    "frozen_path_class_counts",
    "swap_direction",
    "swap_variant",
    "donor_role_reproduction_count",
    "donor_role_reproduction_rate",
    "recipient_restoration_count",
    "recipient_restoration_rate",
    "availability_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value_type in (
        ("repo-root", Path),
        ("stage196b2b2-analysis-json", Path),
        ("stage196b2b2-analyzer-git-commit", str),
        ("stage196b2p0-run-root", Path),
        ("trainer-path", Path),
        ("current-git-commit", str),
        ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=value_type)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError(f"{path}:{line_number}: JSON object required")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cell(value: Any) -> Any:
    if type(value) is not str:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def as_bool(value: Any, label: str) -> bool:
    value = cell(value)
    if type(value) is not bool:
        raise ValueError(f"{label}: boolean required")
    return value


def as_int(value: Any, label: str) -> int:
    value = cell(value)
    if type(value) is not int:
        raise ValueError(f"{label}: integer required")
    return value


def as_number(value: Any, label: str, probability: bool = False) -> float:
    value = cell(value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{label}: finite number required")
    number = float(value)
    if probability and not 0.0 <= number <= 1.0:
        raise ValueError(f"{label}: probability outside [0, 1]")
    return number


def tail3_status(predictions: Sequence[str]) -> str:
    """Frozen B2-A/B2-B1/B2-B2 status definition."""
    sequence = tuple(predictions)
    if sequence == ("SUPPORT",) * 3:
        return "STABLE_SUPPORT"
    if sequence == ("NOT_ENTITLED",) * 3:
        return "PERSISTENT_NOT_ENTITLED"
    if sequence == ("REFUTE",) * 3:
        return "PERSISTENT_REFUTE"
    return "UNSTABLE"


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_columns(label: str, rows: list[dict[str, str]], required: Sequence[str]) -> None:
    observed = set(rows[0]) if rows else set()
    missing = set(required) - observed
    if not rows or missing:
        raise ValueError(f"{label}: absent rows/columns {sorted(missing)}")


def gate(
    gates: list[dict[str, Any]],
    scope: str,
    run: str,
    name: str,
    required: Any,
    observed: Any,
    passed: bool,
    reason: str,
) -> None:
    gates.append(
        {
            "scope": scope,
            "run": run,
            "gate": name,
            "required": required,
            "observed": observed,
            "passed": passed,
            "blocking_reason": "" if passed else reason,
        }
    )
    if not passed:
        raise ValueError(reason)


def record_completed_audit(
    gates: list[dict[str, Any]], name: str, required: Any, observed: Any
) -> None:
    gates.append(
        {
            "scope": "observability",
            "run": "",
            "gate": name,
            "required": required,
            "observed": observed,
            "passed": True,
            "blocking_reason": "",
        }
    )


def exact_directory(directory: Path, expected: Sequence[str], gates: list[dict[str, Any]]) -> None:
    observed = sorted(item.name for item in directory.iterdir())
    required = sorted(expected)
    ok = observed == required and all((directory / name).is_file() for name in expected)
    gate(
        gates,
        "source",
        "",
        "exact_stage196b2b2_nine_file_closure",
        required,
        observed,
        ok,
        "Stage196-B2-B2 exact nine-file closure failed",
    )


def contract_passes(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        as_bool(row.get("passed", ""), "contract passed")
        and row.get("blocking_reason", "") == ""
        for row in rows
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_provenance(analysis: dict[str, Any]) -> dict[str, str]:
    nested = analysis.get("normalized_historical_provenance_roles")
    nested = nested if type(nested) is dict else {}

    def value(*names: str) -> str | None:
        candidates = [analysis.get(name) for name in names]
        candidates.extend(nested.get(name) for name in names)
        present = [str(item) for item in candidates if item not in (None, "")]
        if present and any(item != present[0] for item in present):
            raise ValueError(f"provenance aliases disagree: {names}")
        return present[0] if present else None

    result = {
        "stage196b2b2_analyzer_git_commit": value(
            "current_analyzer_git_commit", "stage196b2b2_analyzer_git_commit"
        ),
        "stage196b2b1_analyzer_git_commit": value("stage196b2b1_analyzer_git_commit"),
        "stage196b2a_analyzer_git_commit": value("stage196b2a_analyzer_git_commit"),
        "stage196b2p0_runtime_git_commit": value(
            "stage196b2p0_runtime_git_commit", "stage196b2p0_runtime_commit"
        ),
        "stage196b1_runtime_git_commit": value("stage196b1_runtime_git_commit"),
        "framegate_implementation_origin_git_commit": value(
            "framegate_implementation_origin_git_commit"
        ),
    }
    if any(value is None for value in result.values()):
        raise ValueError(f"required normalized provenance absent: {result}")
    return {key: str(value) for key, value in result.items()}


def _margin_values(value: Any, context: str = "") -> list[str]:
    values: list[str] = []
    if type(value) is dict:
        for key, child in value.items():
            child_context = f"{context}.{key}".lower()
            if "margin" in child_context and "source" in child_context and type(child) is str:
                values.append(child)
            values.extend(_margin_values(child, child_context))
    elif type(value) is list:
        for child in value:
            values.extend(_margin_values(child, context))
    return values


def normalize_margin_source(
    analysis: dict[str, Any], contract: list[dict[str, str]]
) -> tuple[str, list[str], dict[str, Any]]:
    authoritative: list[str] = []
    for row in contract:
        if not as_bool(row.get("passed", ""), "contract passed"):
            continue
        gate_name = row.get("gate", "").lower()
        for column in ("required", "observed"):
            parsed = cell(row.get(column, ""))
            authoritative.extend(_margin_values(parsed, gate_name))
            if "margin" in gate_name and "source" in gate_name and type(parsed) is str:
                authoritative.append(parsed)
    authoritative = [value for value in authoritative if value not in (None, "")]
    if not authoritative:
        raise ValueError("normalized margin source absent from all authoritative contracts")
    if any(value != MARGIN_SOURCE for value in authoritative):
        raise ValueError(f"authoritative margin-source disagreement: {authoritative}")

    warnings: list[str] = []
    primary = analysis.get("normalized_margin_source")
    aliases = (
        "normalized_support_vs_not_entitled_margin_source",
        "support_vs_not_entitled_margin_source",
        "support_vs_ne_margin_source",
    )
    if primary is None:
        warnings.append(
            "SOURCE_SCHEMA_WARNING: normalized_margin_source is null; normalized from "
            "the passed authoritative contract"
        )
    elif primary != MARGIN_SOURCE:
        raise ValueError("top-level normalized_margin_source disagrees with contract")
    for name in aliases:
        value = analysis.get(name)
        if value is not None and value != MARGIN_SOURCE:
            raise ValueError(f"top-level {name} disagrees with contract")
    return MARGIN_SOURCE, warnings, {
        "top_level_normalized_margin_source": primary,
        "authoritative_contract_values": sorted(set(authoritative)),
    }


def validate_b2b2(
    namespace: argparse.Namespace, gates: list[dict[str, Any]], repo: Path
) -> dict[str, Any]:
    source_path = namespace.stage196b2b2_analysis_json.resolve()
    source_dir = source_path.parent
    gate(
        gates,
        "path",
        "",
        "stage196b2b2_analysis_basename",
        B2B2_FILES[0],
        source_path.name,
        source_path.name == B2B2_FILES[0],
        "Stage196-B2-B2 analysis basename mismatch",
    )
    exact_directory(source_dir, B2B2_FILES, gates)
    analysis = read_json(source_dir / B2B2_FILES[0])
    contract = read_csv(source_dir / B2B2_FILES[-1])
    required_decision = {
        "decision": "STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT",
        "recommended_next_stage": (
            "STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE"
        ),
        "blocking_reasons": [],
    }
    observed_decision = {key: analysis.get(key) for key in required_decision}
    gate(
        gates,
        "source",
        "",
        "stage196b2b2_completed_decision",
        required_decision,
        observed_decision,
        observed_decision == required_decision,
        "Stage196-B2-B2 decision closure mismatch",
    )
    gate(
        gates,
        "source",
        "",
        "stage196b2b2_exact_155_passed_contract_gates",
        155,
        len(contract),
        len(contract) == 155 and contract_passes(contract),
        "Stage196-B2-B2 contract is not exactly 155 passed gates with empty blockers",
    )
    gate(
        gates,
        "provenance",
        "",
        "stage196b2b2_analyzer_commit_argument",
        B2B2_COMMIT,
        namespace.stage196b2b2_analyzer_git_commit,
        namespace.stage196b2b2_analyzer_git_commit == B2B2_COMMIT,
        "Stage196-B2-B2 analyzer commit argument mismatch",
    )
    provenance = normalized_provenance(analysis)
    expected_provenance = {
        "stage196b2b2_analyzer_git_commit": B2B2_COMMIT,
        "stage196b2b1_analyzer_git_commit": B2B1_COMMIT,
        "stage196b2a_analyzer_git_commit": B2A_COMMIT,
        "stage196b2p0_runtime_git_commit": P0_COMMIT,
        "stage196b1_runtime_git_commit": B1_COMMIT,
        "framegate_implementation_origin_git_commit": FRAMEGATE_COMMIT,
    }
    gate(
        gates,
        "provenance",
        "",
        "normalized_historical_provenance_roles",
        expected_provenance,
        provenance,
        provenance == expected_provenance,
        "historical provenance roles disagree",
    )
    margin_source, warnings, margin_provenance = normalize_margin_source(analysis, contract)

    row_paths = read_csv(source_dir / B2B2_FILES[2])
    paired_epochs = read_csv(source_dir / B2B2_FILES[3])
    require_columns(
        "Stage196-B2-B2 row paths",
        row_paths,
        (
            "seed",
            "stable_row_id",
            "id",
            "source_row_id",
            "dev_position",
            "transition_role",
            "intervention_type",
            "path_class",
            "joint_tail3_status",
            "intervention_tail3_status",
            "tail3_prediction_pattern",
            "tail3_delta_frame",
            "tail3_delta_predicate",
            "tail3_delta_entitlement",
            "tail3_delta_margin",
        ),
    )
    require_columns(
        "Stage196-B2-B2 paired epochs",
        paired_epochs,
        ("seed", "stable_row_id", "epoch", "transition_role", "path_class"),
    )
    gate(
        gates,
        "population",
        "",
        "stage196b2b2_row_path_rows",
        16,
        len(row_paths),
        len(row_paths) == 16,
        "Stage196-B2-B2 row-path row count mismatch",
    )
    gate(
        gates,
        "population",
        "",
        "stage196b2b2_paired_epoch_rows",
        320,
        len(paired_epochs),
        len(paired_epochs) == 320,
        "Stage196-B2-B2 paired-epoch row count mismatch",
    )

    seen: set[tuple[int, str, int]] = set()
    path_counts: dict[int, dict[str, Counter[str]]] = {
        seed: {"recovery": Counter(), "harm": Counter()} for seed in POSITIVE_SEEDS
    }
    primary_keys: set[tuple[int, str]] = set()
    for row in row_paths:
        seed = as_int(row["seed"], "primary seed")
        role = row["transition_role"]
        stable_id = row["stable_row_id"]
        position = as_int(row["dev_position"], "primary dev_position")
        key = (seed, stable_id, position)
        if (
            seed not in POSITIVE_SEEDS
            or role not in ("recovery", "harm")
            or row["id"] != stable_id
            or row["source_row_id"] != stable_id
            or key in seen
        ):
            raise ValueError("Stage196-B2-B2 primary identity/seed/role mismatch")
        seen.add(key)
        primary_keys.add((seed, stable_id))
        path_counts[seed][role][row["path_class"]] += 1
    normalized_counts = {
        seed: {role: dict(path_counts[seed][role]) for role in ("recovery", "harm")}
        for seed in POSITIVE_SEEDS
    }
    gate(
        gates,
        "population",
        "",
        "stage196b2b2_frozen_path_class_counts",
        EXPECTED_PATH_COUNTS,
        normalized_counts,
        normalized_counts == EXPECTED_PATH_COUNTS,
        "Stage196-B2-B2 path-class counts disagree with the completed finding",
    )
    population_counts = {
        seed: {
            role: sum(path_counts[seed][role].values()) for role in ("recovery", "harm")
        }
        for seed in POSITIVE_SEEDS
    }
    if population_counts != EXPECTED_PRIMARY_COUNTS:
        raise ValueError("Stage196-B2-B2 primary recovery/harm counts disagree")

    epoch_keys: Counter[tuple[int, str]] = Counter()
    for row in paired_epochs:
        seed = as_int(row["seed"], "paired seed")
        stable_id = row["stable_row_id"]
        epoch = as_int(row["epoch"], "paired epoch")
        if (seed, stable_id) not in primary_keys or epoch not in EPOCHS:
            raise ValueError("Stage196-B2-B2 paired-epoch population mismatch")
        epoch_keys[(seed, stable_id)] += 1
    if set(epoch_keys) != primary_keys or any(count != 20 for count in epoch_keys.values()):
        raise ValueError("each Stage196-B2-B2 primary row must have epochs 1 through 20")

    analysis_counts = analysis.get("path_class_counts_by_seed_and_role")
    expected_json_counts = {
        str(seed): {role: EXPECTED_PATH_COUNTS[seed][role] for role in ("recovery", "harm")}
        for seed in POSITIVE_SEEDS
    }
    if analysis_counts != expected_json_counts:
        raise ValueError("Stage196-B2-B2 analysis JSON path counts disagree")
    positive = analysis.get("positive_seeds", analysis.get("positive_primary_seeds"))
    contrast = analysis.get("contrast_seeds")
    if positive != [184, 185] or contrast != [183]:
        raise ValueError("Stage196-B2-B2 positive/contrast seed roles disagree")

    report_text = (source_dir / B2B2_FILES[1]).read_text(encoding="utf-8")
    expected_report_counts = json.dumps(expected_json_counts, sort_keys=True)
    if (
        required_decision["decision"] not in report_text
        or required_decision["recommended_next_stage"] not in report_text
        or expected_report_counts not in report_text
    ):
        raise ValueError("Stage196-B2-B2 markdown report does not preserve exact path counts")
    return {
        "analysis": analysis,
        "source_dir": source_dir,
        "row_paths": row_paths,
        "paired_epochs": paired_epochs,
        "provenance": provenance,
        "margin_source": margin_source,
        "margin_provenance": margin_provenance,
        "warnings": warnings,
        "path_counts": expected_json_counts,
    }


def validate_p0(
    root: Path, normalized: dict[str, str], gates: list[dict[str, Any]]
) -> dict[str, dict[int, dict[str, tuple[Any, ...]]]]:
    observed_directories = sorted(item.name for item in root.iterdir() if item.is_dir())
    gate(
        gates,
        "p0",
        "",
        "exact_six_run_directories",
        sorted(RUNS),
        observed_directories,
        observed_directories == sorted(RUNS),
        "P0 exact six-run directory closure failed",
    )
    sidecar_pattern = re.compile(r"^stage196b2p0_epoch_channels_([0-9]{3})\.jsonl$")
    expected_names = {
        f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl" for epoch in EPOCHS
    }
    populations: dict[str, dict[int, dict[str, tuple[Any, ...]]]] = {}
    reference_by_run: dict[str, set[tuple[str, str, int]]] = {}
    for run in RUNS:
        seed = int(run[4:7])
        mode = run[8:]
        directory = root / run
        matched = [
            item
            for item in directory.iterdir()
            if item.is_file() and sidecar_pattern.fullmatch(item.name)
        ]
        observed_names = {item.name for item in matched}
        resolved = {item.resolve() for item in matched}
        closure = {
            "matched_file_count": len(matched),
            "unique_resolved_file_count": len(resolved),
            "observed_basenames": sorted(observed_names),
            "missing_basenames": sorted(expected_names - observed_names),
            "unexpected_basenames": sorted(observed_names - expected_names),
        }
        gate(
            gates,
            "p0",
            run,
            "exact_20_epoch_sidecar_namespace",
            sorted(expected_names),
            closure,
            len(matched) == 20 and len(resolved) == 20 and observed_names == expected_names,
            f"{run}: exact sidecar namespace closure failed",
        )
        gate(
            gates,
            "provenance",
            run,
            "p0_runtime_commit",
            P0_COMMIT,
            normalized["stage196b2p0_runtime_git_commit"],
            normalized["stage196b2p0_runtime_git_commit"] == P0_COMMIT,
            f"{run}: P0 runtime commit mismatch",
        )
        epochs: dict[int, dict[str, tuple[Any, ...]]] = {}
        for epoch in EPOCHS:
            path = directory / f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl"
            rows = read_jsonl(path)
            gate(
                gates,
                "p0",
                run,
                f"epoch_{epoch:03d}_720_rows",
                P0_ROWS_PER_EPOCH,
                len(rows),
                len(rows) == P0_ROWS_PER_EPOCH,
                f"{run}:{epoch}: sidecar row count mismatch",
            )
            identities: dict[str, tuple[Any, ...]] = {}
            positions: set[int] = set()
            for row in rows:
                if set(row) != set(P0_FIELDS) or len(row) != len(P0_FIELDS):
                    raise ValueError(f"{run}:{epoch}: exact P0 schema mismatch")
                stable_id = str(row["id"])
                source_id = str(row["source_row_id"])
                position = as_int(row["dev_position"], "dev_position")
                if stable_id != source_id or stable_id in identities or position in positions:
                    raise ValueError(f"{run}:{epoch}: duplicate or unstable identity")
                if not 0 <= position < P0_ROWS_PER_EPOCH:
                    raise ValueError(f"{run}:{epoch}: dev_position outside dense population")
                if (
                    as_int(row["epoch"], "epoch") != epoch
                    or as_int(row["training_seed"], "training_seed") != seed
                    or row["frame_downstream_gradient_mode"] != mode
                ):
                    raise ValueError(f"{run}:{epoch}: seed/mode/epoch provenance mismatch")
                if row["gold_label"] not in ("REFUTE", "NOT_ENTITLED", "SUPPORT"):
                    raise ValueError(f"{run}:{epoch}: invalid gold label")
                if row["prediction"] not in ("REFUTE", "NOT_ENTITLED", "SUPPORT"):
                    raise ValueError(f"{run}:{epoch}: invalid prediction")
                for name in (
                    "frame_probability",
                    "predicate_coverage_probability",
                    "sufficiency_probability",
                    "entitlement_probability",
                    "support_probability",
                    "not_entitled_probability",
                ):
                    as_number(row[name], name, probability=True)
                for name in (
                    "polarity_support_margin",
                    "support_logit",
                    "not_entitled_logit",
                ):
                    as_number(row[name], name)
                identities[stable_id] = (
                    source_id,
                    position,
                    row["gold_label"],
                    row["intervention_type"],
                )
                positions.add(position)
            if positions != set(range(P0_ROWS_PER_EPOCH)):
                raise ValueError(f"{run}:{epoch}: dev_position population is not 0..719")
            population = {(key, *value[:2]) for key, value in identities.items()}
            if run not in reference_by_run:
                reference_by_run[run] = population
            elif population != reference_by_run[run]:
                raise ValueError(f"{run}: stable population drift across epochs")
            epochs[epoch] = identities
        populations[run] = epochs

    for seed in SEEDS:
        joint = populations[f"seed{seed}_joint"]
        local = populations[f"seed{seed}_frame_local_only"]
        for epoch in EPOCHS:
            if set(joint[epoch]) != set(local[epoch]):
                raise ValueError(f"seed{seed}:{epoch}: arm identity mismatch")
            for stable_id in joint[epoch]:
                if joint[epoch][stable_id] != local[epoch][stable_id]:
                    raise ValueError(f"seed{seed}:{epoch}:{stable_id}: arm metadata mismatch")
    gate(
        gates,
        "alignment",
        "",
        "six_run_epoch_population_alignment",
        "6 runs x 20 epochs x 720 stable rows",
        "6 runs x 20 epochs x 720 stable rows",
        True,
        "",
    )
    return populations


def validate_primary_against_p0(
    rows: list[dict[str, str]],
    populations: dict[str, dict[int, dict[str, tuple[Any, ...]]]],
    gates: list[dict[str, Any]],
) -> None:
    for row in rows:
        seed = as_int(row["seed"], "primary seed")
        stable_id = row["stable_row_id"]
        expected = (
            row["source_row_id"],
            as_int(row["dev_position"], "primary dev_position"),
            "SUPPORT",
            row["intervention_type"],
        )
        for mode in MODES:
            for epoch in EPOCHS:
                actual = populations[f"seed{seed}_{mode}"][epoch].get(stable_id)
                if actual != expected:
                    raise ValueError(
                        f"primary/P0 alignment mismatch: {seed}/{mode}/{epoch}/{stable_id}"
                    )
    gate(
        gates,
        "alignment",
        "",
        "exact_16_primary_rows_present_in_both_arms_all_epochs",
        "16 x 2 x 20",
        "16 x 2 x 20",
        True,
        "",
    )


def find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"source audit expected one class {name}, found {len(matches)}")
    return matches[0]


def find_function(parent: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in getattr(parent, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ValueError(f"source audit expected one function {name}")
    return matches[0]


def assignment_map(function: ast.FunctionDef) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
    return result


def source_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    return " ".join(segment.split()) if segment else ast.unparse(node)


def audit_source_graph(
    trainer: Path, repo: Path, gates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trainer_text = trainer.read_text(encoding="utf-8")
    trainer_tree = ast.parse(trainer_text, filename=str(trainer))
    imported = [
        node
        for node in trainer_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "contramamba.modeling_v6b_minimal"
        and any(alias.name == "ContraMambaV6BMinimal" for alias in node.names)
    ]
    if len(imported) != 1:
        raise ValueError("trainer does not statically import the expected v6b_minimal model")
    build_model = find_function(trainer_tree, "build_model")
    build_source = source_segment(trainer_text, build_model)
    for required in (
        "decision_mode=\"explicit_product\"",
        "use_temporal_comparator=True",
        "use_predicate_comparator=True",
    ):
        if required not in build_source:
            raise ValueError(f"trainer build_model missing frozen composer setting: {required}")

    model_path = repo / "src" / "contramamba" / "modeling_v6b_minimal.py"
    head_init_path = repo / "src" / "contramamba" / "heads" / "__init__.py"
    head_path = repo / "src" / "contramamba" / "heads" / "entitlement_decision.py"
    for path in (model_path, head_init_path, head_path):
        if not path.is_file():
            raise ValueError(f"source audit dependency absent: {path}")
    init_text = head_init_path.read_text(encoding="utf-8")
    if "FinalEntitlementDecisionHead" not in init_text or "entitlement_decision" not in init_text:
        raise ValueError("heads package does not resolve FinalEntitlementDecisionHead")

    model_text = model_path.read_text(encoding="utf-8")
    model_tree = ast.parse(model_text, filename=str(model_path))
    model_class = find_class(model_tree, "ContraMambaV6BMinimal")
    model_forward = find_function(model_class, "forward")
    decision_calls = [
        node
        for node in ast.walk(model_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "decision_head"
    ]
    if len(decision_calls) != 1:
        raise ValueError("v6b_minimal forward must contain exactly one decision_head call")
    keyword_map = {keyword.arg: ast.unparse(keyword.value) for keyword in decision_calls[0].keywords}
    expected_keywords = {
        "frame_prob": "frame['frame_prob']",
        "predicate_coverage_prob": "predicate['predicate_coverage_prob']",
        "sufficiency_prob": "sufficiency['sufficiency_prob']",
        "positive_energy": "polarity['positive_energy']",
        "negative_energy": "polarity['negative_energy']",
        "decision_mode": "decision_mode",
    }
    if keyword_map != expected_keywords:
        raise ValueError(f"decision_head causal-input graph changed: {keyword_map}")
    model_assignments = assignment_map(model_forward)
    if ast.unparse(model_assignments.get("base_logits")) != "decision['logits']":
        raise ValueError("base_logits no longer come directly from decision_head")
    final_logit_initializers = {
        ast.unparse(node.value)
        for node in ast.walk(model_forward)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "final_logits"
            for target in node.targets
        )
    }
    if "base_logits" not in final_logit_initializers:
        raise ValueError("final_logits no longer initialize from base_logits")
    model_forward_source = source_segment(model_text, model_forward)
    for required in (
        "final_logits[active, 0] -= alpha",
        "final_logits[active, 1] += alpha",
        "final_logits[active, 2] -= alpha",
        "_ta_penalty = torch.sigmoid(temporal_adapter_logit.detach())",
        "_tc_boost = (",
    ):
        if required not in model_forward_source:
            raise ValueError(f"final comparator modulation changed: {required}")

    head_text = head_path.read_text(encoding="utf-8")
    head_tree = ast.parse(head_text, filename=str(head_path))
    head_class = find_class(head_tree, "FinalEntitlementDecisionHead")
    head_forward = find_function(head_class, "forward")
    head_assignments = assignment_map(head_forward)
    expressions = {
        name: ast.unparse(head_assignments[name])
        for name in (
            "support_logit",
            "refute_logit",
            "alpha",
            "not_entitled_logit",
            "logits",
        )
        if name in head_assignments
    }
    expected_expressions = {
        "support_logit": "entitlement_prob * positive_energy",
        "refute_logit": "entitlement_prob * negative_energy",
        "alpha": "F.softplus(self.raw_alpha)",
        "not_entitled_logit": "self.not_entitled_bias + alpha * (1.0 - entitlement_prob)",
        "logits": "torch.stack([refute_logit, not_entitled_logit, support_logit], dim=-1)",
    }
    if expressions != expected_expressions:
        raise ValueError(f"FinalEntitlementDecisionHead expression graph changed: {expressions}")
    entitlement_nodes = [
        node
        for node in ast.walk(head_forward)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "entitlement_prob" for target in node.targets)
    ]
    product_expressions = {ast.unparse(node.value) for node in entitlement_nodes}
    expected_product = "frame_prob * predicate_coverage_prob * sufficiency_prob"
    if expected_product not in product_expressions:
        raise ValueError("explicit-product entitlement expression is absent")

    trainer_hash = sha256(trainer)
    relative = lambda path: str(path.relative_to(repo)).replace("\\", "/")
    model_callable = "ContraMambaV6BMinimal.forward"
    head_callable = "FinalEntitlementDecisionHead.forward"
    rows: list[dict[str, Any]] = []

    def add(
        order: int,
        stage: str,
        symbol: str,
        classification: str,
        causal: bool,
        consumed_by: str,
        path: Path,
        callable_name: str,
        node: ast.AST,
        expression: str,
        sidecar: str,
        available: bool,
        learned: bool,
        hidden: bool,
        reason: str,
    ) -> None:
        rows.append(
            {
                "order": order,
                "graph_stage": stage,
                "symbol": symbol,
                "classification": classification,
                "actual_causal_input": causal,
                "consumed_by": consumed_by,
                "source_file": relative(path),
                "source_callable": callable_name,
                "source_start_line": node.lineno,
                "source_end_line": node.end_lineno,
                "native_expression": expression,
                "p0_sidecar_field": sidecar,
                "available_in_p0": available,
                "learned_parameter_required": learned,
                "hidden_state_required": hidden,
                "availability_reason": reason,
            }
        )

    decision_node = decision_calls[0]
    for order, symbol, field, expression in (
        (1, "frame_prob", "frame_probability", expected_keywords["frame_prob"]),
        (2, "predicate_coverage_prob", "predicate_coverage_probability", expected_keywords["predicate_coverage_prob"]),
        (3, "sufficiency_prob", "sufficiency_probability", expected_keywords["sufficiency_prob"]),
        (4, "positive_energy", "", expected_keywords["positive_energy"]),
        (5, "negative_energy", "", expected_keywords["negative_energy"]),
    ):
        available = field in P0_FIELDS and bool(field)
        add(
            order,
            "decision_head_input",
            symbol,
            "actual causal composer input",
            True,
            head_callable,
            model_path,
            model_callable,
            decision_node,
            expression,
            field,
            available,
            False,
            False,
            "exported exactly" if available else "actual polarity energy is not exported",
        )
    add(6, "derived_entitlement_prob", "entitlement_prob", "derived causal intermediate", True,
        "SUPPORT/REFUTE/NOT_ENTITLED equations", head_path, head_callable,
        entitlement_nodes[0], expected_product, "entitlement_probability", True, False, False,
        "exported value is diagnostic evidence only; swaps must recompute it from primitives")
    add(7, "decision_parameter", "not_entitled_bias", "learned causal parameter", True,
        "NOT_ENTITLED equation", head_path, head_callable, head_assignments["not_entitled_logit"],
        "self.not_entitled_bias", "", False, True, False, "epoch-specific learned parameter is not exported")
    add(8, "decision_parameter", "raw_alpha/softplus_alpha", "learned causal parameter", True,
        "NOT_ENTITLED equation", head_path, head_callable, head_assignments["alpha"],
        expected_expressions["alpha"], "", False, True, False, "epoch-specific learned parameter is not exported")
    add(9, "base_logit", "support_logit", "native causal output", True, "final logit modulation",
        head_path, head_callable, head_assignments["support_logit"], expected_expressions["support_logit"],
        "support_logit", True, False, False, "exported")
    add(10, "base_logit", "refute_logit", "native causal output", True, "final prediction",
        head_path, head_callable, head_assignments["refute_logit"], expected_expressions["refute_logit"],
        "", False, False, False, "REFUTE logit is not exported")
    add(11, "base_logit", "not_entitled_logit", "native causal output", True, "final logit modulation",
        head_path, head_callable, head_assignments["not_entitled_logit"], expected_expressions["not_entitled_logit"],
        "not_entitled_logit", True, True, False, "output is exported but its learned causal parameters are not")
    add(12, "final_modulation", "temporal/predicate mismatch flags", "actual causal composer input", True,
        "ContraMambaV6BMinimal final logits", model_path, model_callable, model_forward,
        "boolean active masks", "", False, False, False, "actual per-row flags are not exported")
    add(13, "final_modulation", "alpha_temporal/alpha_predicate", "learned causal parameter", True,
        "ContraMambaV6BMinimal final logits", model_path, model_callable, model_forward,
        "softplus-constrained comparator alpha", "", False, True, False,
        "epoch-specific comparator parameters are not exported")
    add(14, "final_modulation", "temporal_adapter_logit/effective_penalty_scale", "conditional actual causal input", True,
        "ContraMambaV6BMinimal final logits", model_path, model_callable, model_forward,
        "sigmoid(temporal_adapter_logit) * temporal_adapter_final_penalty_scale", "", False, False, False,
        "effective epoch/row state and activation configuration are not exported")
    add(15, "final_modulation", "temporal_channel_logit/preservation_entitlement_prob/effective_scale", "conditional actual causal input", True,
        "ContraMambaV6BMinimal final logits", model_path, model_callable, model_forward,
        "sigmoid(temporal_channel_logit) * (1 - preservation_entitlement_prob) * temporal_channel_gated_penalty_scale", "", False, False, False,
        "effective epoch/row state and activation configuration are not exported")
    add(16, "diagnostic_export", "polarity_support_margin", "exported diagnostic value", False,
        "no downstream composer consumer", model_path, model_callable, decision_node,
        "positive_energy - negative_energy", "polarity_support_margin", True, False, False,
        "not consumed by the decision head; cannot replace the two energies")

    missing = [row["symbol"] for row in rows if row["actual_causal_input"] and not row["available_in_p0"]]
    swap_availability = {
        "FRAME_ONLY": True,
        "PREDICATE_ONLY": True,
        "SUFFICIENCY_ONLY": True,
        "ENTITLEMENT_PRIMITIVES": True,
        "POLARITY_ONLY": False,
        "ENTITLEMENT_PLUS_POLARITY": False,
        "FULL_COMPOSER_INPUT_POSITIVE_CONTROL": False,
    }
    audit = {
        "trainer_file": relative(trainer),
        "trainer_file_sha256": trainer_hash,
        "trainer_entry_callable": "build_model",
        "trainer_entry_span": [build_model.lineno, build_model.end_lineno],
        "containing_class_or_function": model_callable,
        "model_forward_source_file": relative(model_path),
        "model_forward_span": [model_forward.lineno, model_forward.end_lineno],
        "decision_head_callable": head_callable,
        "decision_head_source_file": relative(head_path),
        "decision_head_span": [head_forward.lineno, head_forward.end_lineno],
        "framegate_implementation_origin_git_commit": FRAMEGATE_COMMIT,
        "primitive_channel_outputs": [
            "frame_prob",
            "predicate_coverage_prob",
            "sufficiency_prob",
            "positive_energy",
            "negative_energy",
        ],
        "derived_entitlement_computation": expected_product,
        "support_logit_computation": expected_expressions["support_logit"],
        "refute_logit_computation": expected_expressions["refute_logit"],
        "not_entitled_logit_computation": expected_expressions["not_entitled_logit"],
        "native_logit_order": ["REFUTE", "NOT_ENTITLED", "SUPPORT"],
        "learned_parameter_not_represented_in_sidecars_required": True,
        "hidden_state_not_represented_in_sidecars_required_by_final_composer": False,
        "missing_actual_composer_inputs": missing,
        "swap_point_availability": swap_availability,
        "diagnostic_not_causal": ["polarity_support_margin"],
        "source_graph_identified": True,
    }
    gate(gates, "source_audit", "", "exact_v6b_minimal_composer_graph_identified", True, True, True, "")
    record_completed_audit(gates, "missing_actual_composer_inputs_audited", "exact availability audit", missing)
    return rows, audit


def subtype_rows(primary: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    members: Counter[tuple[str, int, str, str]] = Counter()
    for row in primary:
        seed = as_int(row["seed"], "subtype seed")
        role = row["transition_role"]
        path_class = row["path_class"]
        if role == "recovery" and seed == 184 and path_class == "MULTI_CHANNEL_CONFLICT":
            subtype = "RECOVERY_SEED184_MULTI_CHANNEL_CONFLICT"
        elif role == "recovery" and seed == 185 and path_class == "FRAME_ENTITLEMENT_GAIN":
            subtype = "RECOVERY_SEED185_FRAME_ENTITLEMENT_GAIN"
        elif path_class == "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN":
            subtype = "POLARITY_OVERRIDE_ROWS"
        elif path_class == "COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY":
            subtype = "COMPOSITION_RESIDUAL_ROWS"
        elif path_class == "FRAME_ENTITLEMENT_LOSS":
            subtype = "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
        elif seed == 185 and role == "harm" and path_class == "MULTI_CHANNEL_CONFLICT":
            deltas = [
                as_number(row[name], name)
                for name in (
                    "tail3_delta_frame",
                    "tail3_delta_predicate",
                    "tail3_delta_entitlement",
                    "tail3_delta_margin",
                )
            ]
            if not all(value < 0 for value in deltas):
                raise ValueError("seed185 harm multi-channel row fails negative terminal-delta subtype rule")
            subtype = "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
        else:
            raise ValueError(f"primary row has no precommitted subtype: {seed}/{role}/{path_class}")
        members[(subtype, seed, role, path_class)] += 1
    rows = [
        {
            "subtype_audit_group": subtype,
            "seed": seed,
            "transition_role": role,
            "row_count": count,
            "frozen_path_class_counts": {path_class: count},
            "swap_direction": "UNAVAILABLE",
            "swap_variant": "UNAVAILABLE",
            "donor_role_reproduction_count": None,
            "donor_role_reproduction_rate": None,
            "recipient_restoration_count": None,
            "recipient_restoration_rate": None,
            "availability_status": OBSERVABILITY_REQUIRED,
        }
        for (subtype, seed, role, path_class), count in sorted(members.items())
    ]
    counts = Counter()
    for (subtype, _seed, _role, _path), count in members.items():
        counts[subtype] += count
    expected = {
        "COMPOSITION_RESIDUAL_ROWS": 1,
        "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS": 5,
        "POLARITY_OVERRIDE_ROWS": 3,
        "RECOVERY_SEED184_MULTI_CHANNEL_CONFLICT": 5,
        "RECOVERY_SEED185_FRAME_ENTITLEMENT_GAIN": 2,
    }
    if dict(counts) != expected:
        raise ValueError(f"B2-B2 subtype population mismatch: {dict(counts)}")
    return rows, dict(counts)


def base_activity_flags() -> dict[str, bool]:
    return {
        "training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "checkpoint_selection_changed": False,
        "model_decision_rule_changed": False,
        "external_evaluation_performed": False,
        "threshold_search_performed": False,
        "classifier_fitted": False,
        "artifact_component_recomposition": True,
        "model_loaded": False,
        "checkpoint_loaded": False,
    }


def analyze(
    namespace: argparse.Namespace, gates: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    repo = namespace.repo_root.resolve()
    b2b2_path = namespace.stage196b2b2_analysis_json.resolve()
    p0_root = namespace.stage196b2p0_run_root.resolve()
    trainer = namespace.trainer_path.resolve()
    output = namespace.output_dir.resolve()
    gate(gates, "path", "", "repo_root", "existing directory", str(repo), repo.is_dir(), "invalid repo root")
    gate(
        gates,
        "path",
        "",
        "all_inputs_and_output_below_repo",
        True,
        all(under(path, repo) for path in (b2b2_path, p0_root, trainer, output)),
        all(under(path, repo) for path in (b2b2_path, p0_root, trainer, output)),
        "an input or output path escapes repo root",
    )
    expected_trainer = (repo / "scripts" / "train_controlled_v6b_minimal.py").resolve()
    gate(
        gates,
        "path",
        "",
        "exact_trainer_path",
        str(expected_trainer),
        str(trainer),
        trainer == expected_trainer and trainer.is_file(),
        "trainer path must resolve to scripts/train_controlled_v6b_minimal.py",
    )
    separated = all(
        not under(output, source) and not under(source, output)
        for source in (b2b2_path.parent, p0_root, trainer.parent)
    )
    gate(gates, "path", "", "output_separation", True, separated, separated, "output overlaps a source")
    gate(gates, "path", "", "fresh_output_directory", "absent", output.exists(), not output.exists(), "output directory already exists")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit_format = re.fullmatch(r"[0-9a-f]{40}", namespace.current_git_commit or "") is not None
    gate(gates, "provenance", "", "current_commit_format", "lowercase 40-hex", namespace.current_git_commit, commit_format, "invalid current analyzer commit")
    gate(gates, "provenance", "", "current_commit_equals_head", namespace.current_git_commit, head, head == namespace.current_git_commit, "current analyzer commit differs from HEAD")

    b2b2 = validate_b2b2(namespace, gates, repo)
    populations = validate_p0(p0_root, b2b2["provenance"], gates)
    validate_primary_against_p0(b2b2["row_paths"], populations, gates)
    graph_rows, graph_audit = audit_source_graph(trainer, repo, gates)
    subtypes, subtype_counts = subtype_rows(b2b2["row_paths"])

    missing = graph_audit["missing_actual_composer_inputs"]
    full_control = graph_audit["swap_point_availability"]["FULL_COMPOSER_INPUT_POSITIVE_CONTROL"]
    if not missing or full_control:
        raise ValueError(
            "static P0 schema audit unexpectedly claims full recomposition; successful "
            "recomposition requires a separately implemented exact native path"
        )
    decision = OBSERVABILITY_REQUIRED
    record_completed_audit(
        gates,
        "native_reconstruction_precondition",
        "all actual composer inputs and epoch state exported",
        {"missing": missing, "full_positive_control_representable": full_control},
    )
    record_completed_audit(
        gates,
        "scientific_swaps_suppressed_before_native_reconstruction",
        True,
        True,
    )
    gate(gates, "output", "", "fixed_seven_swap_variants", list(SWAP_VARIANTS), list(SWAP_VARIANTS), True, "")
    gate(gates, "output", "", "both_precommitted_directions", list(DIRECTIONS), list(DIRECTIONS), True, "")
    gate(gates, "output", "", "tail_three_primary_epochs", [18, 19, 20], list(TAIL_EPOCHS), True, "")
    gate(gates, "decision", "", "fixed_decision_set", list(DECISIONS), decision, decision in DECISIONS, "decision escaped fixed set")
    record_completed_audit(gates, "header_only_reconstruction_rows", 0, 0)
    record_completed_audit(gates, "header_only_component_swap_rows", 0, 0)
    record_completed_audit(gates, "header_only_row_swap_summary_rows", 0, 0)

    analysis = {
        "stage": "Stage196-B2-B3",
        "decision": decision,
        "recommended_next_stage": NEXT_STAGE[decision],
        "blocking_reasons": [],
        "current_analyzer_git_commit": namespace.current_git_commit,
        **b2b2["provenance"],
        "normalized_historical_provenance_roles": b2b2["provenance"],
        "normalized_margin_source": b2b2["margin_source"],
        "normalized_support_vs_not_entitled_margin_source": b2b2["margin_source"],
        "margin_source_provenance": b2b2["margin_provenance"],
        "source_schema_warnings": b2b2["warnings"],
        "positive_seeds": list(POSITIVE_SEEDS),
        "contrast_seeds": [183],
        "primary_population_total": 16,
        "primary_population_counts": {
            str(seed): EXPECTED_PRIMARY_COUNTS[seed] for seed in POSITIVE_SEEDS
        },
        "path_class_counts_by_seed_and_role": b2b2["path_counts"],
        "tail_epochs": list(TAIL_EPOCHS),
        "swap_directions": list(DIRECTIONS),
        "swap_variants": list(SWAP_VARIANTS),
        "composer_graph_audit": graph_audit,
        "causal_input_availability": {
            "all_required_inputs_available": False,
            "missing_actual_composer_inputs": missing,
            "diagnostic_exports_rejected_as_substitutes": ["polarity_support_margin"],
            "full_composer_input_positive_control_representable": False,
        },
        "native_reconstruction": {
            "required_before_swaps": True,
            "planned_row_count": NATIVE_ROW_TARGET,
            "evaluated_row_count": 0,
            "maximum_absolute_logit_error": None,
            "maximum_margin_error": None,
            "prediction_equality_rate": None,
            "tolerance": TOLERANCE,
            "status": "NOT_PERFORMED_MISSING_ACTUAL_COMPOSER_INPUTS",
        },
        "component_recomposition": {
            "planned_counterfactual_row_count": SWAP_ROW_TARGET,
            "evaluated_counterfactual_row_count": 0,
            "planned_row_summary_count": ROW_SWAP_SUMMARY_TARGET,
            "evaluated_row_summary_count": 0,
            "scientific_swaps_performed": False,
            "reason": "native reconstruction precondition cannot be satisfied",
        },
        "subtype_population_counts": subtype_counts,
        "decision_rule_evaluation": {
            "entitlement_component_dominant": {"evaluated": False, "reason": "recomposition unavailable"},
            "entitlement_polarity_disjunctive_effect": {"evaluated": False, "reason": "recomposition unavailable"},
            "polarity_override_component_confirmed": {"evaluated": False, "reason": "recomposition unavailable"},
            "final_composer_residual_required": {"evaluated": False, "reason": "native reconstruction unavailable"},
            "seed_specific_component_effect": {"evaluated": False, "reason": "recomposition unavailable"},
            "additional_observability_required": {
                "evaluated": True,
                "passed": True,
                "source_artifact_closure_passed": True,
                "composer_graph_identified": True,
                "required_epoch_specific_inputs_missing": True,
            },
        },
        "authorized_interpretation": AUTHORIZED[decision],
        "prohibited_interpretations": list(PROHIBITED),
        "output_file_count": 9,
        **base_activity_flags(),
    }
    tables = {
        "graph": graph_rows,
        "native": [],
        "swaps": [],
        "rows": [],
        "groups": [],
        "subtypes": subtypes,
        "contract": gates,
    }
    return analysis, tables


def incomplete_analysis(namespace: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "stage": "Stage196-B2-B3",
        "decision": INCOMPLETE,
        "recommended_next_stage": NEXT_STAGE[INCOMPLETE],
        "blocking_reasons": [f"{type(error).__name__}: {error}"],
        "current_analyzer_git_commit": namespace.current_git_commit,
        "stage196b2b2_analyzer_git_commit": namespace.stage196b2b2_analyzer_git_commit,
        "stage196b2b1_analyzer_git_commit": None,
        "stage196b2a_analyzer_git_commit": None,
        "stage196b2p0_runtime_git_commit": None,
        "stage196b1_runtime_git_commit": None,
        "framegate_implementation_origin_git_commit": FRAMEGATE_COMMIT,
        "normalized_historical_provenance_roles": {},
        "normalized_margin_source": None,
        "normalized_support_vs_not_entitled_margin_source": None,
        "margin_source_provenance": {},
        "source_schema_warnings": [],
        "positive_seeds": [],
        "contrast_seeds": [],
        "primary_population_total": 0,
        "primary_population_counts": {},
        "path_class_counts_by_seed_and_role": {},
        "tail_epochs": list(TAIL_EPOCHS),
        "swap_directions": list(DIRECTIONS),
        "swap_variants": list(SWAP_VARIANTS),
        "composer_graph_audit": {"source_graph_identified": False},
        "causal_input_availability": {"all_required_inputs_available": False},
        "native_reconstruction": {"required_before_swaps": True, "evaluated_row_count": 0},
        "component_recomposition": {"evaluated_counterfactual_row_count": 0, "scientific_swaps_performed": False},
        "subtype_population_counts": {},
        "decision_rule_evaluation": {"evaluated": False},
        "authorized_interpretation": AUTHORIZED[INCOMPLETE],
        "prohibited_interpretations": list(PROHIBITED),
        "output_file_count": 9,
        **base_activity_flags(),
    }


REPORT_SECTIONS = (
    "Executive decision",
    "Authorized interpretation",
    "Stage196-B2-B2 source closure",
    "P0 epoch-sidecar closure",
    "Provenance normalization",
    "Exact final-composer graph",
    "Causal-input availability",
    "Native reconstruction",
    "Swap validity controls",
    "Primary rows and directions",
    "Frame-only swaps",
    "Predicate-only swaps",
    "Sufficiency-only swaps",
    "Entitlement-primitives swaps",
    "Polarity-only swaps",
    "Entitlement-plus-polarity swaps",
    "Full-composer positive control",
    "Recovery results",
    "Preservation-harm results",
    "B2-B2 subtype audit",
    "Cross-seed consistency",
    "Decision-rule evaluation",
    "Remaining uncertainty",
    "Prohibited claims",
    "Recommended next stage",
)


def markdown(analysis: dict[str, Any]) -> str:
    decision = analysis["decision"]
    available = analysis.get("causal_input_availability", {})
    graph = analysis.get("composer_graph_audit", {})
    no_swaps = "Not evaluated: exact native reconstruction did not pass."
    bodies = (
        f"`{decision}`",
        analysis["authorized_interpretation"],
        "The exact nine-file companion closure, completed B2-B2 decision, empty blockers, 155 passed contract gates, 16 primary rows, 320 paired-epoch rows, seed roles, and frozen path counts are mandatory.",
        "Exactly six run directories and the exact `001`-through-`020` sidecar namespace were validated at 720 rows per sidecar. Unrelated standard artifacts inside each run are outside the sidecar namespace.",
        json.dumps({"roles": analysis.get("normalized_historical_provenance_roles", {}), "margin_source": analysis.get("normalized_margin_source"), "warnings": analysis.get("source_schema_warnings", [])}, sort_keys=True),
        json.dumps(graph, sort_keys=True),
        json.dumps(available, sort_keys=True),
        json.dumps(analysis.get("native_reconstruction", {}), sort_keys=True),
        "Native reconstruction of SUPPORT, REFUTE, NOT_ENTITLED, the SUPPORT-vs-NE margin, and prediction is a hard precondition. No regression, algebraic recovery of hidden energies, interpolation, or diagnostic-field substitution is permitted.",
        json.dumps({"primary_rows": analysis.get("primary_population_total", 0), "positive_seeds": analysis.get("positive_seeds", []), "contrast_seeds": analysis.get("contrast_seeds", []), "directions": analysis.get("swap_directions", []), "epochs": list(EPOCHS), "tail_epochs": analysis.get("tail_epochs", [])}, sort_keys=True),
        no_swaps,
        no_swaps,
        no_swaps,
        no_swaps,
        "Unavailable: the native composer consumes positive and negative energies, not the exported polarity margin.",
        "Unavailable because the polarity inputs and native reconstruction preconditions are missing.",
        "Unavailable. Because the complete actual composer input cannot be represented, the validity control cannot reproduce donor logits.",
        "No forward or reverse recovery denominator was evaluated; zero fabricated counterfactual rows were emitted.",
        "No forward or reverse preservation-harm denominator was evaluated; REFUTE was not collapsed into NOT_ENTITLED.",
        json.dumps(analysis.get("subtype_population_counts", {}), sort_keys=True),
        "No cross-seed component conclusion is authorized before exact bidirectional recomposition. Seed183 remains contrast-only and is excluded from primary decisions.",
        json.dumps(analysis.get("decision_rule_evaluation", {}), sort_keys=True),
        "Epoch-specific positive/negative energies, the REFUTE logit, learned decision-head parameters, and final comparator inputs/state must be exported before exact recomposition can be attempted.",
        "\n".join(f"- {claim}" for claim in analysis["prohibited_interpretations"]),
        f"`{analysis['recommended_next_stage']}`\n\nNo training authorization is granted automatically.",
    )
    if len(bodies) != len(REPORT_SECTIONS):
        raise RuntimeError("report section/body closure mismatch")
    return (
        "# Stage196-B2-B3 Exact Inference-Only Component Recomposition Probe\n\n"
        + "\n\n".join(
            f"## {section}\n\n{body}" for section, body in zip(REPORT_SECTIONS, bodies)
        )
        + "\n"
    )


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"generated CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return buffer.getvalue()


def render(
    analysis: dict[str, Any], tables: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    result = {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: markdown(analysis),
        OUTPUTS[2]: render_csv(GRAPH_H, tables["graph"]),
        OUTPUTS[3]: render_csv(NATIVE_H, tables["native"]),
        OUTPUTS[4]: render_csv(SWAP_H, tables["swaps"]),
        OUTPUTS[5]: render_csv(ROW_SUMMARY_H, tables["rows"]),
        OUTPUTS[6]: render_csv(GROUP_H, tables["groups"]),
        OUTPUTS[7]: render_csv(SUBTYPE_H, tables["subtypes"]),
        OUTPUTS[8]: render_csv(CONTRACT_H, tables["contract"]),
    }
    if set(result) != set(OUTPUTS):
        raise RuntimeError("internal exact nine-output closure mismatch")
    return result


def write_outputs(output: Path, rendered: dict[str, str]) -> None:
    if set(rendered) != set(OUTPUTS):
        raise RuntimeError("refusing non-nine-file output set")
    output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(rendered[name])
    observed = {item.name for item in output.iterdir()}
    if observed != set(OUTPUTS):
        raise RuntimeError("written output closure is not exactly nine files")


def main() -> int:
    namespace = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        analysis, tables = analyze(namespace, gates)
        rendered = render(analysis, tables)
    except Exception as error:
        gates.append(
            {
                "scope": "analysis",
                "run": "",
                "gate": "analysis_completed",
                "required": True,
                "observed": False,
                "passed": False,
                "blocking_reason": f"{type(error).__name__}: {error}",
            }
        )
        analysis = incomplete_analysis(namespace, error)
        tables = {
            "graph": [],
            "native": [],
            "swaps": [],
            "rows": [],
            "groups": [],
            "subtypes": [],
            "contract": gates,
        }
        rendered = render(analysis, tables)
    write_outputs(namespace.output_dir.resolve(), rendered)
    return 0 if analysis["decision"] != INCOMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
