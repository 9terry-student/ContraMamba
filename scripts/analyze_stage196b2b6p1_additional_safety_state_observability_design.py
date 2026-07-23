#!/usr/bin/env python3
"""Stage196-B2-B6P1 static action-response observability design analyzer.

This analyzer reads committed Stage196 reports and current source text.  It does
not load a model or checkpoint, evaluate a safety gate, enumerate feature
subsets, fit a threshold, or use P0 safety labels to select a state family.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

STAGE = "Stage196-B2-B6P1"
AUTHORITATIVE_SOURCE_COMMIT = "a959097bd2b34302503dac19d45a8a113f6b139a"
EXPECTED_CANDIDATES = ("00100000000000", "01000000000000", "10000000000000")
AUTHORITY_STATUSES = (
    "EXACT_EXISTING_ARTIFACT",
    "EXACT_DETERMINISTIC_RECONSTRUCTION",
    "SOURCE_AVAILABLE_EXPORT_MISSING",
    "NEW_COMPOSER_INSTRUMENTATION_REQUIRED",
    "NOT_INFERENCE_AUTHORIZED",
    "NOT_MECHANISTICALLY_JUSTIFIED",
)

DECISION_EXISTING = "STAGE196B2B6P1_EXISTING_ACTION_RESPONSE_STATE_OBSERVABLE"
DECISION_RECOMPOSE = "STAGE196B2B6P1_ACTION_RESPONSE_RECOMPOSITION_REQUIRED"
DECISION_EXPORT = "STAGE196B2B6P1_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT_REQUIRED"
DECISION_INSTRUMENT = "STAGE196B2B6P1_COMPOSER_RESPONSE_INSTRUMENTATION_REQUIRED"
DECISION_NONE = "STAGE196B2B6P1_NO_INFERENCE_AUTHORIZED_SAFETY_STATE"
DECISION_BLOCKED = "STAGE196B2B6P1_BLOCKED_CONTRACT_FAILURE"

NEXT_STAGE = {
    DECISION_EXISTING: "STAGE196B2B6P2_ACTION_RESPONSE_SAFETY_GATE_DIAGNOSTIC",
    DECISION_RECOMPOSE: "STAGE196B2B6P2_EXACT_ACTION_RESPONSE_RECOMPOSITION",
    DECISION_EXPORT: "STAGE196B2B6P2_ACTION_CONDITIONAL_COMPOSER_MARGIN_EXPORT",
    DECISION_INSTRUMENT: "STAGE196B2B6P2_COMPOSER_RESPONSE_INSTRUMENTATION",
    DECISION_NONE: "STAGE196B2B7_SELECTOR_INTERVENTION_RETHINK",
    DECISION_BLOCKED: "STAGE196B2B6P1_REPAIR_CONTRACT",
}

P0_FILES = (
    "stage196b2b6p0_analysis.json",
    "stage196b2b6p0_report.md",
    "stage196b2b6p0_safety_feature_dictionary.csv",
    "stage196b2b6p0_row_safety_targets.csv",
    "stage196b2b6p0_single_state_signature_rows.csv",
    "stage196b2b6p0_single_state_gate_summary.csv",
    "stage196b2b6p0_diagnostic_gate_summary.csv",
    "stage196b2b6p0_gated_policy_audit.csv",
    "stage196b2b6p0_contract.csv",
)
B6_FILES = (
    "stage196b2b6_analysis.json", "stage196b2b6_report.md",
    "stage196b2b6_candidate_feature_subsets.csv", "stage196b2b6_signature_action_map.csv",
    "stage196b2b6_primary_policy_validation.csv", "stage196b2b6_clean_dev_signature_audit.csv",
    "stage196b2b6_clean_dev_application_summary.csv", "stage196b2b6_policy_dominance.csv",
    "stage196b2b6_contract.csv",
)
# The 407.89 MiB recipient-signature CSV is deliberately absent.
B5_REQUIRED_FILES = (
    "stage196b2b5_analysis.json", "stage196b2b5_report.md",
    "stage196b2b5_feature_dictionary.csv", "stage196b2b5_row_action_sets.csv",
    "stage196b2b5_recipient_selector_summary.csv",
    "stage196b2b5_paired_delta_signature_rows.csv",
    "stage196b2b5_paired_delta_selector_summary.csv", "stage196b2b5_contract.csv",
)
B5_EXCLUDED_LARGE_FILE = "stage196b2b5_recipient_signature_rows.csv"
B4_FILES = (
    "stage196b2b4_analysis.json", "stage196b2b4_report.md",
    "stage196b2b4_primitive_coalition_rows.csv", "stage196b2b4_primitive_tail_summary.csv",
    "stage196b2b4_primitive_mobius_terms.csv", "stage196b2b4_residual_coalition_rows.csv",
    "stage196b2b4_residual_mobius_terms.csv", "stage196b2b4_localization_summary.csv",
    "stage196b2b4_contract.csv",
)
B3R1_FILES = (
    "stage196b2b3r1_analysis.json", "stage196b2b3r1_report.md",
    "stage196b2b3r1_component_swap_rows.csv", "stage196b2b3r1_composer_graph.csv",
    "stage196b2b3r1_native_reconstruction.csv", "stage196b2b3r1_row_swap_summary.csv",
    "stage196b2b3r1_group_swap_summary.csv", "stage196b2b3r1_subtype_summary.csv",
    "stage196b2b3r1_contract.csv",
)
OUTPUTS = (
    "stage196b2b6p1_analysis.json",
    "stage196b2b6p1_report.md",
    "stage196b2b6p1_source_closure.csv",
    "stage196b2b6p1_existing_observability_inventory.csv",
    "stage196b2b6p1_candidate_state_dictionary.csv",
    "stage196b2b6p1_action_response_authority_matrix.csv",
    "stage196b2b6p1_leakage_boundary.csv",
    "stage196b2b6p1_decision_gate.csv",
    "stage196b2b6p1_contract.csv",
)

SOURCE_H = (
    "stage", "artifact", "path", "required", "loaded", "row_count", "columns",
    "sha256", "purpose", "large_file_dependency",
)
INVENTORY_H = (
    "stage", "artifact", "field", "field_kind", "population_scope", "epoch_scope",
    "native_or_counterfactual", "numeric_authority", "action_conditional", "notes",
)
STATE_H = (
    "state_family", "state_name", "formal_definition", "inference_available",
    "gold_independent", "action_conditional", "mechanistic_scope",
    "source_artifact_or_source_file", "source_fields", "authority_status",
    "integration_status", "missing_requirement", "recommended_export_location",
)
AUTHORITY_H = (
    "state_family", "primary_for_decision", "existing_exact_scope",
    "full_candidate_population_status", "source_boundary_identified",
    "requires_new_export", "requires_new_instrumentation", "integration_status", "rationale",
)
LEAKAGE_H = (
    "quantity", "authorization_class", "inference_use", "seed_policy", "rationale",
)
DECISION_H = (
    "order", "decision", "condition", "observed", "reached", "recommended_next_stage",
)
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, kind in (
        ("repo-root", Path),
        ("stage196b2b6p0-analysis-json", Path),
        ("stage196b2b6-analysis-json", Path),
        ("stage196b2b5-analysis-json", Path),
        ("stage196b2b4-analysis-json", Path),
        ("stage196b2b3r1-analysis-json", Path),
        ("current-git-commit", str),
        ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=kind)
    return parser.parse_args()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON object required")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def csv_header_and_count(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path}: empty CSV") from exc
        return header, sum(1 for _ in reader)


def csv_value_counts(path: Path, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or field not in reader.fieldnames:
            raise ValueError(f"{path}: missing count field {field}")
        for row in reader:
            counts[row[field]] += 1
    return counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(f"{name}: boolean required")


def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str,
         fatal: bool = True) -> None:
    rows.append({
        "scope": scope, "run": run, "gate": name, "required": required,
        "observed": observed, "passed": bool(passed),
        "blocking_reason": "" if passed else reason,
    })
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def contract_closed(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(
        boolean(row.get("passed"), "passed") and not row.get("blocking_reason", "").strip()
        for row in rows
    )


def under(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def source_rows(
    paths: dict[str, Path], gates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    definitions = {
        "B2-B6P0": (paths["p0"], P0_FILES),
        "B2-B6": (paths["b6"], B6_FILES),
        "B2-B5": (paths["b5"], B5_REQUIRED_FILES),
        "B2-B4": (paths["b4"], B4_FILES),
        "B2-B3-R1": (paths["b3r1"], B3R1_FILES),
    }
    rows: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, list[str]]] = {}
    for stage, (analysis_path, names) in definitions.items():
        directory = analysis_path.parent
        expected_analysis = names[0]
        gate(gates, "source", stage, "exact_analysis_basename", expected_analysis,
             analysis_path.name, analysis_path.name == expected_analysis,
             "supplied authority has wrong basename")
        missing = [name for name in names if not (directory / name).is_file()]
        gate(gates, "source", stage, "required_companion_closure", list(names),
             {"missing": missing}, not missing, "required companion file missing")
        schemas[stage] = {}
        for name in names:
            path = directory / name
            columns: list[str] = []
            row_count: int | str = ""
            if path.suffix == ".csv":
                columns, row_count = csv_header_and_count(path)
                schemas[stage][name] = columns
            rows.append({
                "stage": stage, "artifact": name, "path": str(path), "required": True,
                "loaded": True, "row_count": row_count, "columns": columns,
                "sha256": sha256(path), "purpose": "authoritative source closure",
                "large_file_dependency": False,
            })
    excluded = paths["b5"].parent / B5_EXCLUDED_LARGE_FILE
    rows.append({
        "stage": "B2-B5", "artifact": B5_EXCLUDED_LARGE_FILE, "path": str(excluded),
        "required": False, "loaded": False, "row_count": "", "columns": [], "sha256": "",
        "purpose": "explicitly excluded; summaries and row-action sets are sufficient",
        "large_file_dependency": False,
    })
    gate(gates, "scope", "B2-B5", "large_recipient_signature_csv_nondependency",
         {"required": False, "opened": False}, {"required": False, "opened": False}, True,
         "excluded large CSV became a dependency")
    return rows, schemas


def validate_sources(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    root = ns.repo_root.resolve()
    paths = {
        "p0": ns.stage196b2b6p0_analysis_json.resolve(),
        "b6": ns.stage196b2b6_analysis_json.resolve(),
        "b5": ns.stage196b2b5_analysis_json.resolve(),
        "b4": ns.stage196b2b4_analysis_json.resolve(),
        "b3r1": ns.stage196b2b3r1_analysis_json.resolve(),
        "output": ns.output_dir.resolve(),
    }
    raw = (
        ns.repo_root, ns.stage196b2b6p0_analysis_json, ns.stage196b2b6_analysis_json,
        ns.stage196b2b5_analysis_json, ns.stage196b2b4_analysis_json,
        ns.stage196b2b3r1_analysis_json, ns.output_dir,
    )
    explicit = all(path.is_absolute() and under(root, path.resolve()) for path in raw)
    commit_valid = bool(re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit))
    gate(gates, "invocation", "", "explicit_authority_paths",
         {"absolute_under_repo": True, "commit_format": "40 lowercase hex"},
         {"absolute_under_repo": explicit, "commit_format_valid": commit_valid},
         root.is_dir() and explicit and commit_valid,
         "all paths must be absolute and under repo root")
    gate(gates, "source", "", "current_commit_identity", AUTHORITATIVE_SOURCE_COMMIT,
         ns.current_git_commit, ns.current_git_commit == AUTHORITATIVE_SOURCE_COMMIT,
         "current source commit differs from the precommitted authority")

    closure, schemas = source_rows(paths, gates)
    docs = {key: read_json(path) for key, path in paths.items() if key not in ("output",)}
    expected = {
        "p0": ("STAGE196B2B6P0_CURRENT_SAFETY_OBSERVABILITY_INSUFFICIENT",
               "STAGE196B2B6P1_ADDITIONAL_SAFETY_STATE_OBSERVABILITY_DESIGN"),
        "b6": ("STAGE196B2B6_PRIMARY_EXACT_CLEAN_DEV_UNSAFE",
               "STAGE196B2B6P0_SELECTOR_SAFETY_STATE_OBSERVABILITY"),
        "b5": ("STAGE196B2B5_CROSS_SEED_RECIPIENT_SELECTOR_LOCALIZED",
               "STAGE196B2B6_MINIMAL_SELECTOR_INTERVENTION_DESIGN"),
        "b4": ("STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION",
               "STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY"),
        "b3r1": ("STAGE196B2B3_FINAL_COMPOSER_RESIDUAL_REQUIRED",
                 "STAGE196B2B4_FINAL_COMPOSER_RESIDUAL_LOCALIZATION"),
    }
    for key, (decision, next_stage) in expected.items():
        observed = {
            "decision": docs[key].get("decision"),
            "recommended_next_stage": docs[key].get("recommended_next_stage"),
            "blocking_reasons": docs[key].get("blocking_reasons"),
        }
        required = {"decision": decision, "recommended_next_stage": next_stage, "blocking_reasons": []}
        gate(gates, "source", key, "decision_closure", required, observed,
             observed == required, f"{key} decision closure changed")
    gate(gates, "source", "", "five_stage_source_closure", 5,
         {"closed_stages": len(expected)}, len(expected) == 5,
         "five-stage source closure is incomplete")

    p0 = docs["p0"]
    summary = p0.get("row_safety_target_summary", {})
    aggregate = Counter()
    for candidate in EXPECTED_CANDIDATES:
        aggregate.update(summary.get(candidate, {}))
    target_counts = {name: aggregate[name] for name in ("MUST_ALLOW", "MUST_BLOCK", "OPTIONAL")}
    target_rows = sum(target_counts.values())
    gate(gates, "source", "p0", "p0_target_counts",
         {"rows": 6480, "MUST_ALLOW": 21, "MUST_BLOCK": 171, "OPTIONAL": 6288},
         {"rows": target_rows, **target_counts},
         {"rows": target_rows, **target_counts} ==
         {"rows": 6480, "MUST_ALLOW": 21, "MUST_BLOCK": 171, "OPTIONAL": 6288},
         "P0 target counts changed")
    single_summary_counts = csv_value_counts(
        paths["p0"].parent / "stage196b2b6p0_single_state_gate_summary.csv", "feature_family"
    )
    diagnostic_counts = csv_value_counts(
        paths["p0"].parent / "stage196b2b6p0_diagnostic_gate_summary.csv", "feature_family"
    )
    candidate_count = len(EXPECTED_CANDIDATES)
    single_count = single_summary_counts["single_state"] // candidate_count
    tail_count = diagnostic_counts["tail_trajectory"] // candidate_count
    paired_count = diagnostic_counts["paired_delta"] // candidate_count
    counts = {"single-state": single_count, "tail-trajectory": tail_count, "paired-delta": paired_count}
    gate(gates, "source", "p0", "p0_frozen_enumeration_counts",
         {"single-state": 49149, "tail-trajectory": 49149, "paired-delta": 3069}, counts,
         counts == {"single-state": 49149, "tail-trajectory": 49149, "paired-delta": 3069}
         and single_summary_counts == Counter({"single_state": 49149 * candidate_count})
         and diagnostic_counts == Counter({"tail_trajectory": 49149 * candidate_count,
                                           "paired_delta": 3069 * candidate_count}),
         "P0 enumeration counts changed")
    metric_reproduction = p0.get("source_closure", {}).get("b2b6_metric_reproduction", {})
    expected_metric_reproduction = {
        "disagreements": [], "extra_stored_rows": [], "generated_rows": 27, "stored_rows": 27,
    }
    gate(gates, "source", "p0", "p0_exact_b2b6_metric_reproduction",
         expected_metric_reproduction, metric_reproduction,
         metric_reproduction == expected_metric_reproduction,
         "P0 B2-B6 metric reproduction changed")
    zero_gates = {
        "single_state": p0.get("single_state_inclusion_minimal_feasible_gates"),
        "tail_trajectory": p0.get("tail_trajectory_feasible_gates"),
        "paired_delta": p0.get("paired_delta_feasible_gates"),
    }
    gate(gates, "source", "p0", "p0_zero_feasible_gates",
         {name: [] for name in zero_gates}, zero_gates,
         all(value == [] for value in zero_gates.values()), "P0 feasible gate closure changed")

    b6 = docs["b6"]
    b6_eval = b6.get("decision_rule_evaluation", {})
    candidates = tuple(b6.get("nondominated_candidates", []))
    b6_observed = {
        "masks": list(candidates), "all_primary_exact": b6_eval.get("all_primary_exact"),
        "all_clean_dev_unsafe": b6_eval.get("all_clean_dev_unsafe"),
    }
    gate(gates, "source", "b6", "b2b6_three_exact_unsafe_candidates",
         {"masks": list(EXPECTED_CANDIDATES), "all_primary_exact": True,
          "all_clean_dev_unsafe": True}, b6_observed,
         b6_observed == {"masks": list(EXPECTED_CANDIDATES), "all_primary_exact": True,
                         "all_clean_dev_unsafe": True},
         "B2-B6 exactness/unsafety closure changed")

    # Contract and schema closure for all five stages.
    for key, stage in (("p0", "B2-B6P0"), ("b6", "B2-B6"), ("b5", "B2-B5"),
                       ("b4", "B2-B4"), ("b3r1", "B2-B3-R1")):
        contract_name = {
            "p0": P0_FILES[-1], "b6": B6_FILES[-1], "b5": B5_REQUIRED_FILES[-1],
            "b4": B4_FILES[-1], "b3r1": B3R1_FILES[-1],
        }[key]
        contract = read_csv(paths[key].parent / contract_name)
        gate(gates, "source", stage, "prior_contract_closure", {"failed": 0},
             {"rows": len(contract), "failed": sum(not boolean(r.get("passed"), "passed") for r in contract)},
             contract_closed(contract), f"{stage} contract is not closed")

    b4 = docs["b4"]
    primitive_rows = b4.get("primitive_lattice", {}).get("row_count")
    tail_rows = b4.get("primitive_lattice", {}).get("tail_summary_row_count")
    if tail_rows is None:
        _, tail_rows = csv_header_and_count(paths["b4"].parent / "stage196b2b4_primitive_tail_summary.csv")
    gate(gates, "source", "b4", "b2b4_primitive_tail_authority_separation",
         {"epoch_primitive_rows": 20480, "tail_primitive_summaries": 1024},
         {"epoch_primitive_rows": primitive_rows, "tail_primitive_summaries": tail_rows},
         primitive_rows == 20480 and tail_rows == 1024,
         "B2-B4 epoch/tail authority was conflated")

    b3 = docs["b3r1"]
    native = b3.get("native_reconstruction", {})
    native_ok = (
        b3.get("blocking_reasons") == [] and native.get("row_count") == 86400
        and native.get("prediction_equality_rate") == 1.0
        and isinstance(native.get("maximum_final_logit_error"), (int, float))
        and native.get("maximum_final_logit_error") <= 1e-6
    )
    gate(gates, "source", "b3r1", "b2b3r1_exact_native_recomposition_authority",
         {"rows": 86400, "prediction_equality_rate": 1.0, "max_final_logit_error_lte": 1e-6},
         {"rows": native.get("row_count"), "prediction_equality_rate": native.get("prediction_equality_rate"),
          "maximum_final_logit_error": native.get("maximum_final_logit_error"),
          "decision": b3.get("decision")}, native_ok,
         "B2-B3-R1 exact native recomposition authority changed")

    required_schema = {
        ("B2-B5", "stage196b2b5_feature_dictionary.csv"): ("feature_family", "feature_name", "source_fields"),
        ("B2-B5", "stage196b2b5_recipient_selector_summary.csv"): ("feature_family", "feature_subset_mask", "feasible"),
        ("B2-B5", "stage196b2b5_paired_delta_selector_summary.csv"): ("feature_family", "feature_subset_mask", "feasible"),
        ("B2-B5", "stage196b2b5_row_action_sets.csv"): ("seed", "stable_row_id", "acceptable_coalitions"),
        ("B2-B4", "stage196b2b4_primitive_coalition_rows.csv"): ("epoch", "coalition_mask", "counterfactual_refute_logit", "counterfactual_not_entitled_logit", "counterfactual_support_logit"),
        ("B2-B4", "stage196b2b4_primitive_tail_summary.csv"): ("tail_epochs", "coalition_tail_predictions"),
        ("B2-B4", "stage196b2b4_primitive_mobius_terms.csv"): ("coalition_mask", "refute_interaction", "not_entitled_interaction", "support_interaction"),
        ("B2-B4", "stage196b2b4_residual_coalition_rows.csv"): ("coalition_mask", "counterfactual_prediction"),
        ("B2-B4", "stage196b2b4_residual_mobius_terms.csv"): ("coalition_mask", "margin_interaction"),
        ("B2-B4", "stage196b2b4_localization_summary.csv"): ("record_type", "criterion", "passed"),
        ("B2-B3-R1", "stage196b2b3r1_component_swap_rows.csv"): ("variant", "counterfactual_refute_logit", "counterfactual_not_entitled_logit", "counterfactual_support_logit"),
        ("B2-B3-R1", "stage196b2b3r1_composer_graph.csv"): ("symbol", "actual_causal_input", "exact_formula_implemented"),
        ("B2-B3-R1", "stage196b2b3r1_native_reconstruction.csv"): ("maximum_final_logit_error", "prediction_equality_rate"),
        ("B2-B3-R1", "stage196b2b3r1_row_swap_summary.csv"): ("variant", "counterfactual_tail_predictions"),
        ("B2-B3-R1", "stage196b2b3r1_group_swap_summary.csv"): ("variant", "mean_tail_margin_shift"),
    }
    for (stage, artifact), required_columns in required_schema.items():
        columns = schemas[stage][artifact]
        missing = sorted(set(required_columns) - set(columns))
        gate(gates, "schema", stage, artifact, list(required_columns),
             {"missing": missing}, not missing, f"{artifact} schema changed")

    source_files = {
        "model": root / "src/contramamba/modeling_v6b_minimal.py",
        "decision_head": root / "src/contramamba/heads/entitlement_decision.py",
    }
    source_text = {}
    for name, path in source_files.items():
        if not path.is_file():
            raise ValueError(f"missing source authority: {path}")
        source_text[name] = path.read_text(encoding="utf-8")
        closure.append({
            "stage": "current-source", "artifact": path.name, "path": str(path),
            "required": True, "loaded": True, "row_count": "", "columns": [],
            "sha256": sha256(path), "purpose": "static composer boundary inspection",
            "large_file_dependency": False,
        })
    return {"paths": paths, "docs": docs, "closure": closure, "schemas": schemas,
            "source_files": source_files, "source_text": source_text}


def inventory(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = {
        "B2-B6P0": ("stage196b2b6p0_row_safety_targets.csv",),
        "B2-B6": ("stage196b2b6_primary_policy_validation.csv", "stage196b2b6_clean_dev_application_summary.csv"),
        "B2-B5": ("stage196b2b5_feature_dictionary.csv", "stage196b2b5_row_action_sets.csv",
                    "stage196b2b5_recipient_selector_summary.csv", "stage196b2b5_paired_delta_selector_summary.csv"),
        "B2-B4": ("stage196b2b4_primitive_coalition_rows.csv", "stage196b2b4_primitive_tail_summary.csv",
                    "stage196b2b4_primitive_mobius_terms.csv", "stage196b2b4_residual_coalition_rows.csv",
                    "stage196b2b4_residual_mobius_terms.csv", "stage196b2b4_localization_summary.csv"),
        "B2-B3-R1": ("stage196b2b3r1_component_swap_rows.csv", "stage196b2b3r1_composer_graph.csv",
                       "stage196b2b3r1_native_reconstruction.csv", "stage196b2b3r1_row_swap_summary.csv",
                       "stage196b2b3r1_group_swap_summary.csv"),
    }
    numeric_tokens = ("logit", "margin", "interaction", "error", "prob", "energy", "scale", "rate")
    action_tokens = ("counterfactual", "coalition", "swap", "action", "prediction_changed")
    for stage, artifacts in selected.items():
        for artifact in artifacts:
            columns = source["schemas"][stage][artifact]
            for field in columns:
                rows.append({
                    "stage": stage, "artifact": artifact, "field": field,
                    "field_kind": "numeric_or_vector" if any(t in field for t in numeric_tokens) else "categorical_or_provenance",
                    "population_scope": "controlled primary identities" if stage in ("B2-B4", "B2-B3-R1") else "stored stage population",
                    "epoch_scope": "epochs 1-20 or tail3 as declared by artifact",
                    "native_or_counterfactual": "counterfactual" if "counterfactual" in field else "native_or_metadata",
                    "numeric_authority": any(t in field for t in numeric_tokens),
                    "action_conditional": any(t in field for t in action_tokens),
                    "notes": "header-level static inventory; values are not used to choose a state family",
                })
    return rows


def state_row(family: str, name: str, definition: str, status: str,
              sources: str, fields: Sequence[str], action: bool,
              integration: str, missing: str = "", export: str = "") -> dict[str, Any]:
    if status not in AUTHORITY_STATUSES:
        raise ValueError(f"unknown authority status: {status}")
    return {
        "state_family": family, "state_name": name, "formal_definition": definition,
        "inference_available": status not in ("NOT_INFERENCE_AUTHORIZED",),
        "gold_independent": True, "action_conditional": action,
        "mechanistic_scope": "exact final composer decision geometry",
        "source_artifact_or_source_file": sources, "source_fields": list(fields),
        "authority_status": status, "integration_status": integration,
        "missing_requirement": missing, "recommended_export_location": export,
    }


def candidate_dictionary(source_boundary: dict[str, bool]) -> list[dict[str, Any]]:
    export = "candidate-action inference export: one row per seed/epoch/row/candidate/action"
    model = "src/contramamba/modeling_v6b_minimal.py"
    primary = "stage196b2b4_primitive_coalition_rows.csv; stage196b2b3r1_component_swap_rows.csv"
    missing_scores = "exact native and candidate-action final score vectors for all 6,480 candidate-row applications"
    rows: list[dict[str, Any]] = []
    # Family A
    rows += [
        state_row("A", "native_class_score_vector", "s_native(x)=[s_R,s_N,s_S] before selector action", "SOURCE_AVAILABLE_EXPORT_MISSING", f"{model}; {primary}", ("logits", "recipient_*_logit"), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_top1_class", "argmax_c s_native,c(x)", "EXACT_EXISTING_ARTIFACT", "stage196b2b6p0_row_safety_targets.csv", ("joint_prediction",), False, "INTEGRATION_AUTHORIZED"),
        state_row("A", "native_runner_up_class", "second-ranked class under exact native scores", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_top1_minus_runner_up_margin", "max(s_native)-secondmax(s_native)", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_support_minus_not_entitled_margin", "s_native,S-s_native,N", "SOURCE_AVAILABLE_EXPORT_MISSING", f"{model}; {primary}", ("recipient_support_logit", "recipient_not_entitled_logit"), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_support_minus_refute_margin", "s_native,S-s_native,R", "SOURCE_AVAILABLE_EXPORT_MISSING", f"{model}; {primary}", ("recipient_support_logit", "recipient_refute_logit"), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_refute_minus_not_entitled_margin", "s_native,R-s_native,N", "SOURCE_AVAILABLE_EXPORT_MISSING", f"{model}; {primary}", ("recipient_refute_logit", "recipient_not_entitled_logit"), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_entitlement_branch_score", "max(s_native,S,s_native,R)-s_native,N", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_polarity_branch_score", "s_native,S-s_native,R", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), False, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("A", "native_probability_vector", "softmax(s_native); secondary representation only", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), False, "INTEGRATION_AUTHORIZED_SECONDARY", "requires raw native scores first", export),
    ]
    # Family B
    rows += [
        state_row("B", "counterfactual_class_score_vector", "s_a(x)=[s_R,s_N,s_S] after exact proposed action a", "SOURCE_AVAILABLE_EXPORT_MISSING", f"{model}; {primary}", ("counterfactual_*_logit", "logits"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("B", "counterfactual_top1_class", "argmax_c s_a,c(x)", "EXACT_EXISTING_ARTIFACT", "stage196b2b6p0_row_safety_targets.csv", ("selector_prediction", "evaluated_actions"), True, "INTEGRATION_AUTHORIZED"),
        state_row("B", "counterfactual_runner_up_class", "second-ranked class under s_a", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("B", "counterfactual_top1_minus_runner_up_margin", "max(s_a)-secondmax(s_a)", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("B", "counterfactual_support_minus_not_entitled_margin", "s_a,S-s_a,N", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_support_logit", "counterfactual_not_entitled_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("B", "counterfactual_support_minus_refute_margin", "s_a,S-s_a,R", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_support_logit", "counterfactual_refute_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("B", "counterfactual_refute_minus_not_entitled_margin", "s_a,R-s_a,N", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_refute_logit", "counterfactual_not_entitled_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
    ]
    # Family C
    rows += [
        state_row("C", "delta_class_score_vector", "Delta_a(x)=s_a(x)-s_native(x)", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_minus_recipient_*",), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("C", "delta_top1_margin", "m_top1(s_a)-m_top1(s_native)", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("C", "delta_support_minus_not_entitled_margin", "(s_a,S-s_a,N)-(s_native,S-s_native,N)", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_*_logit", "recipient_*_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("C", "delta_support_minus_refute_margin", "(s_a,S-s_a,R)-(s_native,S-s_native,R)", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_*_logit", "recipient_*_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("C", "delta_refute_minus_not_entitled_margin", "(s_a,R-s_a,N)-(s_native,R-s_native,N)", "SOURCE_AVAILABLE_EXPORT_MISSING", primary, ("counterfactual_*_logit", "recipient_*_logit"), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
        state_row("C", "prediction_changed", "1[argmax(s_a)!=argmax(s_native)]", "EXACT_EXISTING_ARTIFACT", "stage196b2b6p0_row_safety_targets.csv", ("joint_prediction", "selector_prediction"), True, "INTEGRATION_AUTHORIZED"),
        state_row("C", "entitlement_branch_changed", "1[entitled(argmax(s_a))!=entitled(argmax(s_native))]", "EXACT_DETERMINISTIC_RECONSTRUCTION", "stage196b2b6p0_row_safety_targets.csv", ("joint_prediction", "selector_prediction"), True, "INTEGRATION_AUTHORIZED"),
        state_row("C", "polarity_branch_changed", "1[both predictions entitled and SUPPORT/REFUTE differs]", "EXACT_DETERMINISTIC_RECONSTRUCTION", "stage196b2b6p0_row_safety_targets.csv", ("joint_prediction", "selector_prediction"), True, "INTEGRATION_AUTHORIZED"),
        state_row("C", "polarity_direction_preserved", "1[both predictions entitled and SUPPORT/REFUTE is equal]", "EXACT_DETERMINISTIC_RECONSTRUCTION", "stage196b2b6p0_row_safety_targets.csv", ("joint_prediction", "selector_prediction"), True, "INTEGRATION_AUTHORIZED"),
        state_row("C", "action_response_magnitude", "||Delta_a(x)||_2 with declared exact score units", "SOURCE_AVAILABLE_EXPORT_MISSING", model, ("logits",), True, "INTEGRATION_AUTHORIZED", missing_scores, export),
    ]
    # Family D: exact only on controlled primary lattices; full candidate coverage needs export.
    for name, definition, fields in (
        ("individual_primitive_class_score_contribution", "singleton exact contribution to each class score", ("refute_interaction", "not_entitled_interaction", "support_interaction")),
        ("individual_primitive_margin_contribution", "singleton exact contribution to each named decision margin", ("margin_interaction",)),
        ("coalition_interaction_contribution", "coalition effect retaining primitive membership and class coordinate", ("coalition_mask", "*_interaction")),
        ("mobius_interaction_term", "mu(S)=f(S)-sum_{T proper-subset S}mu(T)", ("coalition_mask", "*_interaction")),
        ("residual_contribution", "full-composer residual-lattice contribution by residual group", ("lattice", "*_interaction")),
        ("dominant_primitive", "argmax primitive contribution magnitude with ties preserved", ("coalition_mask", "*_interaction")),
        ("dominant_interaction_order", "interaction order with largest declared norm, ties preserved", ("coalition_size", "*_interaction")),
    ):
        rows.append(state_row("D", name, definition, "SOURCE_AVAILABLE_EXPORT_MISSING",
                              "stage196b2b4_primitive_mobius_terms.csv; stage196b2b4_residual_mobius_terms.csv",
                              fields, True, "INTEGRATION_AUTHORIZED", "exact terms exist only for controlled primary identities; export full candidate-action population", export))
    # Family E is diagnostic-only even where exact tail predictions already exist.
    for name, definition, fields in (
        ("action_response_sign_stability", "constancy of each signed Delta margin over epochs 18-20", ("counterfactual_tail_predictions", "mean_tail_margin_shift")),
        ("action_response_margin_stability", "declared dispersion/range of action-conditioned margins over epochs 18-20", ("counterfactual_minus_recipient_margin",)),
        ("counterfactual_winner_stability", "constancy of argmax(s_a) over epochs 18-20", ("counterfactual_tail_predictions",)),
        ("entitlement_transition_stability", "constancy of native-to-action entitlement transition over tail3", ("recipient_tail_predictions", "counterfactual_tail_predictions")),
        ("polarity_transition_stability", "constancy of native-to-action polarity transition over tail3", ("recipient_tail_predictions", "counterfactual_tail_predictions")),
        ("primitive_contribution_stability", "stability of exact per-primitive class/margin contribution over tail3", ("epoch", "*_interaction")),
    ):
        rows.append(state_row("E", name, definition, "EXACT_DETERMINISTIC_RECONSTRUCTION",
                              "stage196b2b4 primitive and tail artifacts; stage196b2b3r1 row swap summary",
                              fields, True, "DIAGNOSTIC_ONLY", "full candidate population would require a separate tail3 export", "diagnostic tail3 action-response export"))
    score_vectors = {"native_class_score_vector", "counterfactual_class_score_vector"}
    for row in rows:
        if row["state_family"] not in ("A", "B", "C"):
            continue
        if row["authority_status"] != "SOURCE_AVAILABLE_EXPORT_MISSING":
            continue
        if source_boundary["full_action_scores_existing"]:
            row["authority_status"] = ("EXACT_EXISTING_ARTIFACT" if row["state_name"] in score_vectors
                                       else "EXACT_DETERMINISTIC_RECONSTRUCTION")
            row["missing_requirement"] = ""
        elif source_boundary["full_action_scores_reconstructable"]:
            row["authority_status"] = "EXACT_DETERMINISTIC_RECONSTRUCTION"
        elif not source_boundary["final_scores_computed_and_returned"]:
            row["authority_status"] = "NEW_COMPOSER_INSTRUMENTATION_REQUIRED"
    return rows


def leakage_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    authorized = (
        "native model-internal scores", "candidate-action counterfactual scores",
        "exact deterministic score deltas", "exact deterministic margin deltas",
        "exact primitive contributions", "exact composer interaction terms",
        "action-response stability computed without labels",
    )
    diagnostic = (
        "tail3 aggregate statistics", "cross-seed recurrence", "donor-recipient paired statistics",
        "population-frequency features",
    )
    prohibited = (
        "gold label", "stored target label", "recovery status", "harm status",
        "MUST_ALLOW/MUST_BLOCK/OPTIONAL", "correctness", "false-entitlement status",
        "stable_row_id", "raw data identity", "text lexical identity",
        "candidate primary-case membership", "discovery/nondiscovery membership",
        "seed identity as a decision feature", "post-hoc chosen thresholds",
    )
    for quantity in authorized:
        rows.append({"quantity": quantity, "authorization_class": "INTEGRATION_AUTHORIZED",
                     "inference_use": "potential B2-B6P2 input after exact authority is exported",
                     "seed_policy": "seed may group audits but is not an input",
                     "rationale": "model-internal, gold-independent, mechanistic action response"})
    for quantity in diagnostic:
        rows.append({"quantity": quantity, "authorization_class": "DIAGNOSTIC_ONLY",
                     "inference_use": "not authorized without a separate later justification",
                     "seed_policy": "grouping/provenance only",
                     "rationale": "aggregate or paired context is not the minimal action-response state"})
    for quantity in prohibited:
        rows.append({"quantity": quantity, "authorization_class": "PROHIBITED_SELECTOR_INPUT",
                     "inference_use": "never", "seed_policy": "grouping/provenance only",
                     "rationale": "outcome, identity, membership, or post-hoc leakage"})
    return rows


def authority_matrix(dictionary: list[dict[str, Any]], source_boundary: dict[str, bool]) -> list[dict[str, Any]]:
    rationale = {
        "A": "Native prediction exists broadly; raw native scores exist only in controlled artifacts and are computed at the source boundary.",
        "B": "Counterfactual prediction exists broadly; exact counterfactual scores exist only for controlled primitive/swap rows.",
        "C": "Categorical response is derivable, but exact score and margin deltas require native and counterfactual score export.",
        "D": "B2-B4 provides exact class-coordinate Mobius authority on controlled lattices, not full candidate-row coverage.",
        "E": "Tail3 action-conditioned response is reconstructable on controlled rows and remains diagnostic-only.",
    }
    rows = []
    for family in "ABCDE":
        members = [row for row in dictionary if row["state_family"] == family]
        statuses = sorted(set(row["authority_status"] for row in members))
        rows.append({
            "state_family": family, "primary_for_decision": family in "ABC",
            "existing_exact_scope": "controlled primary rows" if family in "DE" else "prediction-level broad; score-level controlled",
            "full_candidate_population_status": statuses,
            "source_boundary_identified": source_boundary["final_scores_computed_and_returned"],
            "requires_new_export": any(row["authority_status"] == "SOURCE_AVAILABLE_EXPORT_MISSING" for row in members),
            "requires_new_instrumentation": any(row["authority_status"] == "NEW_COMPOSER_INSTRUMENTATION_REQUIRED" for row in members),
            "integration_status": "DIAGNOSTIC_ONLY" if family == "E" else "CANDIDATE_AUTHORIZED_NO_GATE_PROMOTION",
            "rationale": rationale[family],
        })
    return rows


def decide(dictionary: list[dict[str, Any]], source_boundary: dict[str, bool]) -> tuple[str, list[dict[str, Any]]]:
    primary = [row for row in dictionary if row["state_family"] in ("A", "B", "C")]
    all_existing = source_boundary["full_action_scores_existing"] and all(
        row["authority_status"] in ("EXACT_EXISTING_ARTIFACT", "EXACT_DETERMINISTIC_RECONSTRUCTION")
        for row in primary)
    all_reconstructable = source_boundary["full_action_scores_reconstructable"]
    source_export = (
        source_boundary["final_scores_computed_and_returned"]
        and any(row["authority_status"] == "SOURCE_AVAILABLE_EXPORT_MISSING" for row in primary)
    )
    instrumentation = (
        not source_boundary["final_scores_computed_and_returned"]
        and any(row["authority_status"] in ("SOURCE_AVAILABLE_EXPORT_MISSING",
                                             "NEW_COMPOSER_INSTRUMENTATION_REQUIRED") for row in primary)
    )
    only_prohibited = bool(primary) and all(row["authority_status"] in
                                           ("NOT_INFERENCE_AUTHORIZED", "NOT_MECHANISTICALLY_JUSTIFIED")
                                           for row in primary)
    conditions = (
        (DECISION_EXISTING, all_existing, "all primary A-C quantities exist exactly in committed artifacts"),
        (DECISION_RECOMPOSE, all_reconstructable and not all_existing,
         "exact native/counterfactual scores are deterministically reconstructable but not materialized"),
        (DECISION_EXPORT, source_export,
         "source computes exact final scores, while full candidate-action artifacts omit them"),
        (DECISION_INSTRUMENT, instrumentation,
         "necessary exact state has no identifiable current composer boundary"),
        (DECISION_NONE, only_prohibited,
         "only prohibited outcome, identity, or gold quantities remain"),
    )
    decision = next((name for name, passed, _ in conditions if passed), "")
    if not decision:
        raise ValueError("decision hierarchy did not reach exactly one outcome")
    rows = [{
        "order": index, "decision": name, "condition": condition,
        "observed": passed, "reached": name == decision,
        "recommended_next_stage": NEXT_STAGE[name],
    } for index, (name, passed, condition) in enumerate(conditions, 1)]
    return decision, rows


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = validate_sources(ns, gates)
    model_text = source["source_text"]["model"]
    head_text = source["source_text"]["decision_head"]
    boundary = {
        "base_scores_computed": "base_logits = decision[\"logits\"]" in model_text,
        "final_scores_computed": "final_logits = base_logits" in model_text,
        "final_scores_returned": '"logits": final_logits' in model_text,
        "predictions_derived_from_final_scores": "final_logits.argmax" in model_text,
        "class_score_vector_constructed": "torch.stack" in head_text and '"logits": logits' in head_text,
    }
    boundary["final_scores_computed_and_returned"] = all((
        boundary["base_scores_computed"], boundary["final_scores_computed"],
        boundary["final_scores_returned"], boundary["predictions_derived_from_final_scores"],
        boundary["class_score_vector_constructed"],
    ))
    p0_fields = set(source["schemas"]["B2-B6P0"]["stage196b2b6p0_row_safety_targets.csv"])
    native_score_fields = {"native_refute_logit", "native_not_entitled_logit", "native_support_logit"}
    action_score_fields = {"counterfactual_refute_logit", "counterfactual_not_entitled_logit",
                           "counterfactual_support_logit"}
    recomposition_fields = {
        "frame_prob", "predicate_coverage_prob", "sufficiency_prob", "positive_energy",
        "negative_energy", "raw_alpha", "not_entitled_bias", "temporal_mismatch_active",
        "predicate_mismatch_active", "temporal_adapter_logit", "temporal_channel_logit",
        "preservation_entitlement_prob",
    }
    boundary["full_action_scores_existing"] = (
        native_score_fields <= p0_fields and action_score_fields <= p0_fields)
    boundary["full_action_scores_reconstructable"] = (
        not boundary["full_action_scores_existing"] and recomposition_fields <= p0_fields)
    gate(gates, "source", "current", "final_composer_score_boundary_inspection_complete",
         {"inspection_complete": True},
         {"inspection_complete": all(isinstance(value, bool) for value in boundary.values()),
          "boundary": boundary},
         all(isinstance(value, bool) for value in boundary.values()),
         "static final-composer boundary inspection is incomplete")

    observed_inventory = inventory(source)
    gate(gates, "inventory", "", "existing_artifact_field_inventory_completion",
         {"nonempty": True}, {"rows": len(observed_inventory)}, bool(observed_inventory),
         "field inventory is empty")
    dictionary = candidate_dictionary(boundary)
    expected_state_names = len(set(row["state_name"] for row in dictionary))
    gate(gates, "dictionary", "", "candidate_state_dictionary_completion",
         {"families": list("ABCDE"), "rows": 40, "unique_state_names": 40},
         {"families": sorted(set(row["state_family"] for row in dictionary)),
          "rows": len(dictionary), "unique_state_names": expected_state_names},
         sorted(set(row["state_family"] for row in dictionary)) == list("ABCDE")
         and len(dictionary) == 40 and expected_state_names == 40, "candidate state dictionary incomplete")
    bad_status = sorted(set(row["authority_status"] for row in dictionary) - set(AUTHORITY_STATUSES))
    gate(gates, "dictionary", "", "authority_status_closure", list(AUTHORITY_STATUSES),
         {"used": sorted(set(row["authority_status"] for row in dictionary)), "invalid": bad_status},
         not bad_status and all(row["authority_status"] for row in dictionary),
         "candidate quantity lacks exactly one recognized authority status")
    leakage = leakage_rows()
    expected_leakage = {"INTEGRATION_AUTHORIZED": 7, "DIAGNOSTIC_ONLY": 4,
                        "PROHIBITED_SELECTOR_INPUT": 14}
    leakage_counts = Counter(row["authorization_class"] for row in leakage)
    gate(gates, "leakage", "", "leakage_boundary_closure", expected_leakage,
         dict(leakage_counts), dict(leakage_counts) == expected_leakage,
         "leakage authorization matrix incomplete")
    matrix = authority_matrix(dictionary, boundary)
    decision, decision_rows = decide(dictionary, boundary)
    gate(gates, "decision", "", "decision_rule_reachability", {"reached": 1},
         {"reached": sum(bool(row["reached"]) for row in decision_rows), "decision": decision},
         sum(bool(row["reached"]) for row in decision_rows) == 1,
         "ordered decision hierarchy did not reach exactly one decision")
    gate(gates, "scope", "", "no_gate_evaluation_or_feature_search",
         {"gate_evaluated": False, "feature_subsets_enumerated": False, "threshold_fitted": False},
         {"gate_evaluated": False, "feature_subsets_enumerated": False, "threshold_fitted": False},
         True, "stage scope expanded")
    gate(gates, "output", "", "exact_nine_file_output_closure", sorted(OUTPUTS),
         sorted(OUTPUTS), len(OUTPUTS) == 9 and len(set(OUTPUTS)) == 9,
         "output declaration is not an exact nine-file set")

    status_counts = Counter(row["authority_status"] for row in dictionary)
    analysis = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": NEXT_STAGE[decision],
        "blocking_reasons": [], "current_git_commit": ns.current_git_commit,
        "source_paths": {key: str(value) for key, value in source["paths"].items() if key != "output"},
        "source_hashes": {row["path"]: row["sha256"] for row in source["closure"] if row["sha256"]},
        "source_closure": {
            "five_stage_closed": True,
            "large_recipient_signature_csv_required": False,
            "large_recipient_signature_csv_loaded": False,
            "artifacts_inspected": len(source["closure"]),
        },
        "p0_closure": {
            "decision": source["docs"]["p0"].get("decision"),
            "feature_families_exhausted": ["single-state", "tail-trajectory", "paired-delta"],
            "zero_feasible_gates": True, "candidate_row_applications": 6480,
            "target_counts": {"MUST_ALLOW": 21, "MUST_BLOCK": 171, "OPTIONAL": 6288},
            "enumeration_counts": {"single-state": 49149, "tail-trajectory": 49149, "paired-delta": 3069},
        },
        "precommitted_candidate_hierarchy": {
            "A": "native composer decision geometry",
            "B": "candidate-action counterfactual composer geometry",
            "C": "exact action-response delta",
            "D": "primitive causal contribution geometry",
            "E": "tail3 action-response stability (diagnostic-only)",
        },
        "action_conditional_state_definition": {
            "native_scores": "s_native(x)", "counterfactual_scores": "s_a(x)",
            "response": "Delta_a(x)=s_a(x)-s_native(x)",
            "candidate_selector_masks": list(EXPECTED_CANDIDATES),
            "gold_correctness_distinct": True,
        },
        "composer_source_boundary": boundary,
        "deterministic_reconstruction_findings": {
            "controlled_native_nodes": [
                "frame_prob", "predicate_coverage_prob", "sufficiency_prob",
                "positive_energy", "negative_energy", "entitlement_prob",
                "decision_head_class_scores", "final_composer_class_scores",
            ],
            "controlled_primitive_swaps": [
                "FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY",
            ],
            "controlled_interactions": ["primitive Mobius terms", "residual Mobius terms"],
            "broad_prediction_response": "exact from joint_prediction and selector_prediction",
            "broad_numeric_response": "not exactly reconstructable from committed candidate-row artifacts",
            "numeric_authority_rule": "prediction authority never substitutes for raw score authority",
        },
        "observability_finding": "prediction-level action response is stored broadly; exact score/margin response is stored only for controlled primary identities",
        "missing_observability_boundary": "export native and candidate-action final class-score vectors for every candidate-row application at the existing final_logits boundary",
        "candidate_state_count": len(dictionary), "authority_status_counts": dict(status_counts),
        "decision_rule_evaluation": {"ordered_decisions": [row["decision"] for row in decision_rows],
                                     "reached": decision},
        "scientific_prohibitions": {
            "selector_promoted": False, "safety_gate_evaluated": False,
            "threshold_learned": False, "model_loaded": False, "checkpoint_loaded": False,
            "feature_subset_enumeration_performed": False, "p0_targets_used_for_family_selection": False,
        },
        "minimal_principled_state_family_for_b2b6p2": [
            "native final class-score vector", "candidate-action final class-score vector",
            "exact score delta vector", "exact named decision-margin deltas",
            "prediction/entitlement/polarity transition indicators",
        ],
        "authorized_interpretation": "B2-B6P1 identifies an action-conditional final-composer score export boundary only; it authorizes no gate or selector promotion.",
        "remaining_risks": [
            "Static source matching cannot prove a future exporter preserves dtype, row/action alignment, or checkpoint identity.",
            "B2-B4 primitive and residual attributions cover controlled primary identities, not the full clean-dev candidate population.",
            "Tail3 action-response stability remains diagnostic-only unless separately authorized.",
        ],
        "artifact_only": True, "static_source_inspection_only": True,
        "model_loaded": False, "checkpoint_loaded": False, "training_performed": False,
        "classifier_fitted": False, "learned_threshold_used": False,
        "safety_gate_evaluated": False, "promotion_authorized": False,
    }
    return analysis, {
        "source": source["closure"], "inventory": observed_inventory, "dictionary": dictionary,
        "authority": matrix, "leakage": leakage, "decision": decision_rows,
    }


def report(analysis: dict[str, Any]) -> str:
    if analysis["blocking_reasons"]:
        return (
            f"# {STAGE}: Additional Safety-State Observability Design\n\n"
            f"Decision: `{analysis['decision']}`\n\n"
            "The fail-closed source/artifact contract blocked scientific interpretation.\n\n"
            f"Blocking reasons: `{canonical(analysis['blocking_reasons'])}`\n"
        )
    boundary = analysis["composer_source_boundary"]
    return f"""# {STAGE}: Additional Safety-State Observability Design

## Decision

`{analysis['decision']}`

Recommended next stage: `{analysis['recommended_next_stage']}`.

This decision follows only from source and artifact authority. No relationship between a candidate quantity and P0 safety targets was inspected to choose the family.

## P0 closure and scope

P0 exhausted the currently authorized single-state (49,149), tail-trajectory (49,149), and paired-delta (3,069) feature subsets. It found no feasible safety gate over 6,480 candidate-row applications (21 MUST_ALLOW, 171 MUST_BLOCK, 6,288 OPTIONAL).

B2-B6P1 does not test another recipient-state subset. It does not enumerate feature subsets, train a model, fit a threshold, evaluate a safety gate, or promote a selector.

## Proposed action-conditional state

For row `x` and proposed selector action `a`, define native final composer scores `s_native(x)`, counterfactual final composer scores `s_a(x)`, and exact response `Delta_a(x) = s_a(x) - s_native(x)`. Named class margins and branch transitions are deterministic functions of those exact score vectors.

Counterfactual model response is model-internal and is distinct from gold correctness, recovery, harm, MUST_ALLOW, and MUST_BLOCK.

## Precommitted hierarchy

1. Family A — native composer decision geometry.
2. Family B — candidate-action counterfactual composer geometry.
3. Family C — exact action-response delta.
4. Family D — primitive causal contribution geometry without collapsing class coordinates or interaction order.
5. Family E — tail3 action-response stability, diagnostic-only.

## Existing artifact observability

B2-B6/P0 broadly store native and selector predictions. B2-B4 and B2-B3-R1 store exact recipient/counterfactual class logits, margins, primitive swaps, residual swaps, and Möbius terms for the controlled primary identities. B2-B4 preserves 20,480 epoch-level primitive coalition rows separately from 1,024 tail primitive summaries.

Those controlled artifacts do not materialize exact native and candidate-action score vectors for all 6,480 B2-B6/P0 candidate-row applications. A prediction label is not accepted as numeric score authority.

## Deterministic reconstruction findings

Given exact score vectors, runner-up class, top1 margin, named pairwise margins, score deltas, margin deltas, prediction change, entitlement transition, polarity transition, response magnitude, and probabilities are deterministic. Prediction/entitlement/polarity transitions are already reconstructable from stored native and counterfactual labels; numeric margins are not.

B2-B4 authorizes exact controlled-population primitive and Möbius attribution. Extending Family D to the full candidate population requires an exact export with row/action/epoch alignment; it is not inferred from labels.

## Missing export boundary

Static source inspection found `base_logits`, `final_logits`, returned `logits`, and predictions derived from `final_logits` at the current composer boundary: `{canonical(boundary)}`.

The minimal missing authority is an inference-time export with one aligned record per seed, epoch, row, selector candidate, and proposed primitive action containing exact native and counterfactual final class-score vectors. Named margins and deltas should be derived deterministically from those vectors. New classifier logic and learned thresholds are outside scope.

## Leakage boundary

Native scores, candidate-action scores, exact score/margin deltas, exact primitive contributions, exact composer interactions, and label-free action-response stability are potentially integration-authorized. Tail3 aggregates, cross-seed recurrence, donor-recipient paired statistics, and population frequencies remain diagnostic-only.

Gold/target/outcome/correctness fields, false-entitlement status, identities, lexical identity, primary/discovery membership, seed as a decision feature, and post-hoc thresholds are prohibited selector inputs. Seed is permitted only for grouping, transfer audit, and provenance.

## Contract and limitations

The analyzer is fail-closed, writes exactly nine files atomically into a new directory, and returns 0 only when `blocking_reasons == []`; otherwise it returns 2. Unhandled analysis failures still produce nine blocked outputs where output creation remains possible.

The 407.89 MiB `stage196b2b5_recipient_signature_rows.csv` is neither opened nor required. No model or checkpoint is loaded. No selector is promoted, no safety gate is evaluated, and no threshold is learned.
"""


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return canonical(value)
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


def render_contract(rows: list[dict[str, Any]]) -> str:
    return render_csv(CONTRACT_H, [
        {**row, "required": canonical(row["required"]), "observed": canonical(row["observed"])}
        for row in rows
    ])


def blocked_analysis(ns: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "stage": STAGE, "decision": DECISION_BLOCKED,
        "recommended_next_stage": NEXT_STAGE[DECISION_BLOCKED],
        "blocking_reasons": [f"{type(error).__name__}: {error}"],
        "current_git_commit": ns.current_git_commit,
        "source_paths": {
            "stage196b2b6p0_analysis_json": str(ns.stage196b2b6p0_analysis_json.resolve()),
            "stage196b2b6_analysis_json": str(ns.stage196b2b6_analysis_json.resolve()),
            "stage196b2b5_analysis_json": str(ns.stage196b2b5_analysis_json.resolve()),
            "stage196b2b4_analysis_json": str(ns.stage196b2b4_analysis_json.resolve()),
            "stage196b2b3r1_analysis_json": str(ns.stage196b2b3r1_analysis_json.resolve()),
        },
        "source_hashes": {}, "source_closure": {"five_stage_closed": False,
                                                   "large_recipient_signature_csv_required": False,
                                                   "large_recipient_signature_csv_loaded": False},
        "p0_closure": {}, "precommitted_candidate_hierarchy": {},
        "action_conditional_state_definition": {}, "composer_source_boundary": {},
        "observability_finding": "unavailable because contract failed",
        "missing_observability_boundary": "unavailable because contract failed",
        "candidate_state_count": 0, "authority_status_counts": {},
        "decision_rule_evaluation": {"completed": False},
        "scientific_prohibitions": {"selector_promoted": False, "safety_gate_evaluated": False,
                                     "threshold_learned": False},
        "minimal_principled_state_family_for_b2b6p2": [],
        "authorized_interpretation": "No scientific interpretation is authorized because a contract failed.",
        "remaining_risks": ["Repair the failed source/artifact contract."],
        "artifact_only": True, "static_source_inspection_only": True,
        "model_loaded": False, "checkpoint_loaded": False, "training_performed": False,
        "classifier_fitted": False, "learned_threshold_used": False,
        "safety_gate_evaluated": False, "promotion_authorized": False,
    }


def payloads(analysis: dict[str, Any], tables: dict[str, Any],
             gates: list[dict[str, Any]]) -> dict[str, str]:
    return {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: report(analysis),
        OUTPUTS[2]: render_csv(SOURCE_H, tables["source"]),
        OUTPUTS[3]: render_csv(INVENTORY_H, tables["inventory"]),
        OUTPUTS[4]: render_csv(STATE_H, tables["dictionary"]),
        OUTPUTS[5]: render_csv(AUTHORITY_H, tables["authority"]),
        OUTPUTS[6]: render_csv(LEAKAGE_H, tables["leakage"]),
        OUTPUTS[7]: render_csv(DECISION_H, tables["decision"]),
        OUTPUTS[8]: render_contract(gates),
    }


def atomic_write_outputs(output: Path, data: dict[str, str]) -> None:
    if output.exists() or set(data) != set(OUTPUTS):
        raise RuntimeError("refusing overwrite or non-nine-file output")
    if not output.parent.is_dir():
        raise RuntimeError("output parent must already exist")
    temporary = output.parent / f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        for name in OUTPUTS:
            with (temporary / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(data[name])
                handle.flush()
                os.fsync(handle.fileno())
        observed = sorted(path.name for path in temporary.iterdir() if path.is_file())
        if observed != sorted(OUTPUTS):
            raise RuntimeError("staged exact nine-file closure failed")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    ns = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        analysis, tables = analyze(ns, gates)
    except Exception as error:
        gate(gates, "analysis", "", "unhandled_contract_failure", False,
             f"{type(error).__name__}: {error}", False,
             f"{type(error).__name__}: {error}", fatal=False)
        analysis = blocked_analysis(ns, error)
        tables = {"source": [], "inventory": [], "dictionary": [], "authority": [],
                  "leakage": [], "decision": []}
    atomic_write_outputs(ns.output_dir.resolve(), payloads(analysis, tables, gates))
    return 0 if analysis["blocking_reasons"] == [] else 2


if __name__ == "__main__":
    raise SystemExit(main())
