#!/usr/bin/env python3
"""Stage196-B2-B1 artifact-only recovery/preservation-harm bifurcation audit."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

STAGE = "Stage196-B2-B1"
SOURCE_DECISION = "STAGE196B2A_SEED_SPECIFIC_MIXED_PROPAGATION"
SOURCE_NEXT = "STAGE196B2B_NO_PROMOTION_MINIMAL_SEED_SPECIFIC_FOLLOWUP"
SOURCE_ANALYZER_COMMIT = "833752ec25890e76ee3c2dc3f3bc3c3c1b7428a6"
P0_RUNTIME_COMMIT = "e9aaff24054f1d409119b70df13b94159a34a8e4"
B1_RUNTIME_COMMIT = "9835cbbf86d83aca0964821669e63f7f6deb1c59"
FRAMEGATE_ORIGIN_COMMIT = "5a39538ef3ca8f36cc2cc5d3290eae60d6a5f5c8"
MARGIN_SOURCE = "support_logit - not_entitled_logit"
TAIL = (18, 19, 20)
THRESHOLD = 0.5
EXPECTED_PRIMARY_COUNTS = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}
EXPECTED_DECISION_POPULATIONS = {184: 12, 185: 18}
LOCAL_CHANNELS = ("frame", "predicate", "sufficiency", "polarity", "entitlement")
RECURRENT_SETS = (
    "stage196a_baseline_recurrent", "stage196a_intervention_recurrent",
    "stage196a_common_recurrent", "stage196a_universal_all_six",
)
SOURCE_FILES = (
    "stage196b2a_analysis.json", "stage196b2a_report.md",
    "stage196b2a_seed_summary.csv", "stage196b2a_support_transition_rows.csv",
    "stage196b2a_channel_transition_summary.csv",
    "stage196b2a_recurrent_position_propagation.csv",
    "stage196b2a_harm_rescue_rows.csv", "stage196b2a_epoch_propagation.csv",
    "stage196b2a_contract.csv",
)
OUTPUTS = (
    "stage196b2b1_analysis.json", "stage196b2b1_report.md",
    "stage196b2b1_group_summary.csv", "stage196b2b1_row_profiles.csv",
    "stage196b2b1_signature_summary.csv", "stage196b2b1_cross_seed_transfer.csv",
    "stage196b2b1_intervention_type_summary.csv", "stage196b2b1_contract.csv",
)
INCOMPLETE = "STAGE196B2B1_ANALYSIS_INCOMPLETE"
DECISIONS = (
    "STAGE196B2B1_ISOLATED_FRAME_DEFICIT_BIFURCATION",
    "STAGE196B2B1_FRAME_BOUNDARY_BIFURCATION",
    "STAGE196B2B1_LOCAL_STATE_OVERLAP_COMPOSITION_ONLY_SEPARATION",
    "STAGE196B2B1_SEED_SPECIFIC_NO_STABLE_BIFURCATION",
    INCOMPLETE,
)
NEXT_STAGE = {
    DECISIONS[0]: "STAGE196B2B2_ERROR_CONDITIONED_FRAME_GRADIENT_ROUTING_DESIGN",
    DECISIONS[1]: "STAGE196B2B2_FRAME_BOUNDARY_CONDITIONED_INTERVENTION_DESIGN",
    DECISIONS[2]: "STAGE196B2B2_FRAME_COMPOSITION_JOINT_MICROINTERVENTION_DESIGN",
    DECISIONS[3]: "STAGE196B2B2_NO_PROMOTION_ROW_LEVEL_CAUSAL_PROBE",
    INCOMPLETE: "STAGE196B2B1_REPAIR_ANALYSIS_INPUTS",
}
AUTHORIZED = {
    DECISIONS[0]: ("Recovery and preservation harm occupy consistently distinct native local-channel "
                   "configurations, with recovery concentrated in isolated Frame deficit and harm "
                   "concentrated in fully passing local states. This authorizes design work only."),
    DECISIONS[1]: ("Native Frame pass/fail separates recovery from preservation harm, but downstream "
                   "local-state structure is not clean enough to support the stronger isolated-deficit "
                   "claim. This authorizes design work only."),
    DECISIONS[2]: ("Local Frame/predicate/sufficiency/polarity/entitlement states do not stably separate "
                   "recovery from preservation harm; distinction emerges only after final composition "
                   "is included. Selective Frame routing is not authorized."),
    DECISIONS[3]: ("Recovery and preservation harm do not exhibit a stable, transferable native-state "
                   "bifurcation across the positive Frame-shift seeds."),
    INCOMPLETE: "No scientific interpretation is authorized because required analysis closure failed.",
}
PROHIBITED = [
    "formal causal mediation", "deployable sample routing", "open-world failure detection",
    "unfrozen encoder behavior", "external/OOD improvement", "architecture superiority",
    "a selective intervention is already safe",
    "final prediction status itself constitutes a mechanistic separator",
    "automatic authorization of a new loss, trainer modification, global gradient detachment, "
    "checkpoint-selection changes, full retraining, or external evaluation",
]

CONTRACT_HEADER = ["scope", "run", "gate", "required", "observed", "passed", "blocking_reason"]
GROUP_HEADER = [
    "seed", "analysis_view", "transition_role", "row_count",
    "native_state_class_counts", "native_state_class_rates",
    "local_signature_counts", "local_signature_rates",
    *[f"{channel}_pass_rate" for channel in LOCAL_CHANNELS],
    *[f"{channel}_headroom_summary" for channel in LOCAL_CHANNELS],
    "intervention_type_counts", "intervention_type_rates",
    *[f"in_{name}_count" for name in RECURRENT_SETS],
    "universal_all_six_membership_count",
]
TRACE_CHANNELS = LOCAL_CHANNELS + (
    "support_vs_not_entitled_margin", "support_probability", "not_entitled_probability")
ROW_HEADER = [
    "id", "source_row_id", "stable_row_id", "dev_position", "seed",
    "analysis_view", "transition_role", "intervention_type",
    *[f"in_{name}" for name in RECURRENT_SETS],
    *[f"joint_{channel}_value" for channel in LOCAL_CHANNELS],
    *[f"{channel}_pass" for channel in LOCAL_CHANNELS],
    *[f"{channel}_headroom" for channel in LOCAL_CHANNELS],
    "local_signature", "mechanistic_state_class",
    *[f"trace_intervention_{channel}_value" for channel in TRACE_CHANNELS],
    *[f"trace_paired_{channel}_delta" for channel in TRACE_CHANNELS],
    "trace_joint_tail3_status", "trace_intervention_tail3_status",
    "trace_selected_joint_final", "trace_selected_intervention_final",
    "trace_selected_paired_transition",
    "secondary_joint_support_vs_not_entitled_margin",
    "secondary_joint_support_probability", "secondary_joint_not_entitled_probability",
    "secondary_final_composition_headroom", "secondary_composition_pass",
    "secondary_composition_augmented_signature", "secondary_view_label",
    "source_row_provenance",
]
SIGNATURE_HEADER = [
    "seed", "signature_view", "view_authority", "recovery_signature_set",
    "harm_signature_set", "shared_signature_set", "recovery_exclusive_signatures",
    "harm_exclusive_signatures", "signature_set_jaccard_overlap",
    "recovery_exclusive_row_coverage", "recovery_exclusive_coverage_numerator",
    "recovery_exclusive_coverage_denominator", "harm_exclusive_row_coverage",
    "harm_exclusive_coverage_numerator", "harm_exclusive_coverage_denominator",
    "collision_signatures", "collision_row_count",
]
TRANSFER_HEADER = [
    "source_seed", "target_seed", "signature_view", "view_authority",
    "source_recovery_exclusive_signatures", "source_harm_exclusive_signatures",
    "target_recovery_coverage", "target_recovery_coverage_numerator",
    "target_recovery_coverage_denominator", "target_recovery_purity",
    "target_recovery_purity_numerator", "target_recovery_purity_denominator",
    "target_recovery_collision_count", "target_harm_coverage",
    "target_harm_coverage_numerator", "target_harm_coverage_denominator",
    "target_harm_purity", "target_harm_purity_numerator",
    "target_harm_purity_denominator", "target_harm_collision_count",
    "target_uncovered_row_count", "target_uncovered_recovery_count",
    "target_uncovered_harm_count", "exact_transfer_stable",
]
INTERVENTION_HEADER = [
    "seed", "intervention_type", "recovery_count", "harm_count",
    "recovery_rate_within_seed_role", "harm_rate_within_seed_role",
    "role_presence", "recovery_only_in_seed", "harm_only_in_seed", "appears_in_both",
    "recovery_exclusive_persists_across_positive_seeds",
    "harm_exclusive_persists_across_positive_seeds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--stage196b2a-analysis-json", required=True, type=Path)
    parser.add_argument("--stage196b2a-analyzer-git-commit", required=True)
    parser.add_argument("--current-git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: CSV header absent")
        return list(reader)


def cell(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if text[0] in "[{\"" or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def boolean(value: Any, field: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is str and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{field}: expected serialized boolean")


def integer(value: Any, field: str) -> int:
    if type(value) is int:
        return value
    if type(value) is str and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    raise ValueError(f"{field}: expected integer")


def number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field}: expected finite number")
    return result


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def add_gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
             required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": passed,
                 "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(reason)


def require_columns(name: str, rows: list[dict[str, str]], required: Sequence[str]) -> None:
    if not rows:
        raise ValueError(f"{name}: no data rows")
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise ValueError(f"{name}: required columns absent: {missing}")


def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def git_head(repo: Path) -> str:
    process = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
    return process.stdout.strip()


def safe_paths(args: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    repo = args.repo_root.resolve()
    source = args.stage196b2a_analysis_json.resolve()
    output = args.output_dir.resolve()
    add_gate(gates, "path", "", "repo_root", "existing directory", str(repo), repo.is_dir(),
             "repository root is not a directory")
    add_gate(gates, "path", "", "source_analysis_name", "stage196b2a_analysis.json",
             source.name, source.name == "stage196b2a_analysis.json" and source.is_file(),
             "Stage196-B2-A analysis path/name mismatch")
    add_gate(gates, "path", "", "paths_below_repository", True,
             {"source": under(source, repo), "output": under(output, repo)},
             under(source, repo) and under(output, repo),
             "source and output paths must resolve below repository root")
    separated = not under(output, source.parent) and not under(source.parent, output)
    add_gate(gates, "path", "", "source_output_separation", "non-overlapping directory trees",
             {"source_directory": str(source.parent), "output_directory": str(output)}, separated,
             "output directory tree must not overlap the frozen source directory tree")
    output_ok = not output.exists() or (output.is_dir() and not any(output.iterdir()))
    add_gate(gates, "path", "", "new_empty_output", True, output_ok, output_ok,
             "output directory exists and is nonempty or is not a directory")
    return repo, source.parent, output


def contract_rows(contract: list[dict[str, str]], gate: str) -> list[dict[str, str]]:
    return [row for row in contract if row.get("gate") == gate]


def contract_scalar(contract: list[dict[str, str]], gate: str, column: str,
                    expected_count: int | None = None) -> Any:
    rows = contract_rows(contract, gate)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"source contract gate {gate}: expected {expected_count} rows, found {len(rows)}")
    if not rows:
        raise ValueError(f"source contract gate absent: {gate}")
    values = [cell(row.get(column, "")) for row in rows]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"source contract gate {gate}: nonuniform {column}")
    return first


def normalize_top_level(analysis: dict[str, Any], aliases: Sequence[str],
                        authoritative: str, warnings: list[str]) -> None:
    populated = [(name, analysis.get(name)) for name in aliases if analysis.get(name) is not None]
    null_aliases = [name for name in aliases if name in analysis and analysis[name] is None]
    for name, value in populated:
        if value != authoritative:
            raise ValueError(f"top-level {name} disagrees with authoritative contract value")
    if null_aliases or not populated:
        detail = ", ".join(null_aliases) if null_aliases else "/".join(aliases)
        warnings.append(f"SOURCE_SCHEMA_WARNING: top-level {detail} is null or absent; normalized from passed contract")


def normalize_provenance(args: argparse.Namespace, analysis: dict[str, Any],
                         contract: list[dict[str, str]], gates: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    source_commit = contract_scalar(contract, "analysis_runtime_commit_equals_head", "observed", 1)
    b1_commit = contract_scalar(contract, "stage196b1_runtime_commit_format", "observed", 1)
    p0_commit = contract_scalar(contract, "p0_runtime_commit", "required", 6)
    origin_required = contract_scalar(
        contract, "framegate_implementation_origin_commit_preserved", "required", 1)
    if type(origin_required) is not dict:
        raise ValueError("FrameGate origin contract row is not a mapping")
    origin_commit = origin_required.get("git_commit")
    p0_schema = contract_rows(contract, "p0_contract")
    p0_alignment = contract_rows(contract, "p0_sidecar_alignment")
    margin_contract_ok = len(p0_schema) == 6 and len(p0_alignment) == 6
    if margin_contract_ok:
        for row in p0_schema:
            required = cell(row.get("required", ""))
            required_fields = required.get("stage196b2p0_required_fields", []) if type(required) is dict else []
            margin_contract_ok = margin_contract_ok and type(required) is dict and (
                required.get("stage196b2p0_epoch_channel_observability_enabled") is True and
                "support_logit" in required_fields and "not_entitled_logit" in required_fields)
        margin_contract_ok = margin_contract_ok and all(
            cell(row.get("required", "")) == "20 x 720" and
            cell(row.get("observed", "")) == "20 x 720" for row in p0_alignment)
    add_gate(gates, "source", "", "contract_native_logit_margin_authority",
             MARGIN_SOURCE, {"p0_contract_rows": len(p0_schema),
                             "p0_sidecar_alignment_rows": len(p0_alignment)},
             margin_contract_ok,
             "passed source contract does not establish native paired sidecar composition values")
    normalized = {
        "stage196b2a_analyzer_git_commit": str(source_commit),
        "stage196b2p0_runtime_git_commit": str(p0_commit),
        "stage196b1_runtime_git_commit": str(b1_commit),
        "framegate_implementation_origin_git_commit": str(origin_commit),
        "support_vs_not_entitled_margin_source": MARGIN_SOURCE,
    }
    expected = {
        "stage196b2a_analyzer_git_commit": SOURCE_ANALYZER_COMMIT,
        "stage196b2p0_runtime_git_commit": P0_RUNTIME_COMMIT,
        "stage196b1_runtime_git_commit": B1_RUNTIME_COMMIT,
        "framegate_implementation_origin_git_commit": FRAMEGATE_ORIGIN_COMMIT,
        "support_vs_not_entitled_margin_source": MARGIN_SOURCE,
    }
    add_gate(gates, "provenance", "", "normalized_source_roles", expected, normalized,
             normalized == expected, "normalized source provenance roles disagree with the frozen contract")
    add_gate(gates, "provenance", "", "explicit_b2a_analyzer_commit",
             SOURCE_ANALYZER_COMMIT, args.stage196b2a_analyzer_git_commit,
             args.stage196b2a_analyzer_git_commit == SOURCE_ANALYZER_COMMIT == source_commit,
             "Stage196-B2-A analyzer commit argument/contract mismatch")
    normalize_top_level(analysis, ("analysis_runtime_commit", "analysis_runtime_git_commit"),
                        source_commit, warnings)
    normalize_top_level(analysis, ("stage196b2p0_runtime_commit",
                                   "stage196b2p0_runtime_git_commit"), p0_commit, warnings)
    normalize_top_level(analysis, ("stage196b1_runtime_git_commit",), b1_commit, warnings)
    normalize_top_level(analysis, ("framegate_implementation_origin_git_commit",),
                        origin_commit, warnings)
    normalize_top_level(analysis, ("support_vs_ne_margin_source",
                                   "support_vs_not_entitled_margin_source"),
                        MARGIN_SOURCE, warnings)
    return normalized, warnings


def load_source(args: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    repo, directory, output = safe_paths(args, gates)
    entries = sorted(path.name for path in directory.iterdir())
    exact_files = sorted(SOURCE_FILES)
    add_gate(gates, "source", "", "exact_nine_file_source_closure", exact_files, entries,
             entries == exact_files and all((directory / name).is_file() for name in SOURCE_FILES),
             "Stage196-B2-A source directory is not the exact nine-file closure")
    analysis = read_json(directory / SOURCE_FILES[0])
    if type(analysis) is not dict:
        raise ValueError("Stage196-B2-A analysis JSON must be an object")
    required_decision = {"decision": SOURCE_DECISION, "recommended_next_stage": SOURCE_NEXT,
                         "blocking_reasons": []}
    observed_decision = {key: analysis.get(key) for key in required_decision}
    add_gate(gates, "source", "", "completed_b2a_decision_closure", required_decision,
             observed_decision, observed_decision == required_decision,
             "Stage196-B2-A decision/next-stage/blocker closure mismatch")
    contract = read_csv(directory / "stage196b2a_contract.csv")
    contract_ok = bool(contract) and all(
        boolean(row.get("passed", ""), "source contract passed") and
        row.get("blocking_reason", "") == "" for row in contract)
    add_gate(gates, "source", "", "all_source_contract_gates_passed", True,
             contract_ok, contract_ok, "Stage196-B2-A source contract contains a failure")
    normalized, warnings = normalize_provenance(args, analysis, contract, gates)
    current_ok = re.fullmatch(r"[0-9a-f]{40}", args.current_git_commit or "") is not None
    add_gate(gates, "provenance", "", "current_analyzer_commit_format", "lowercase 40-hex",
             args.current_git_commit, current_ok, "current analyzer commit is not lowercase full 40-hex")
    head = git_head(repo)
    add_gate(gates, "provenance", "", "current_analyzer_commit_equals_head",
             args.current_git_commit, head, head == args.current_git_commit,
             "current analyzer commit differs from repository HEAD")
    seed_summary = read_csv(directory / "stage196b2a_seed_summary.csv")
    transitions = read_csv(directory / "stage196b2a_support_transition_rows.csv")
    channel_summary = read_csv(directory / "stage196b2a_channel_transition_summary.csv")
    recurrent = read_csv(directory / "stage196b2a_recurrent_position_propagation.csv")
    harm = read_csv(directory / "stage196b2a_harm_rescue_rows.csv")
    epoch = read_csv(directory / "stage196b2a_epoch_propagation.csv")
    add_gate(gates, "source", "", "three_seed_summary_rows", 3, len(seed_summary),
             len(seed_summary) == 3, "Stage196-B2-A seed summary must contain exactly three rows")
    add_gate(gates, "source", "", "sixty_epoch_propagation_rows", 60, len(epoch),
             len(epoch) == 60, "Stage196-B2-A epoch propagation must contain exactly 60 rows")
    add_gate(gates, "source", "", "support_transition_row_closure", 4320, len(transitions),
             len(transitions) == 4320,
             "Stage196-B2-A support-transition table must contain 3 x 720 x 2 rows")
    require_columns("seed summary", seed_summary, (
        "seed", "common_recurrent_frame_shift", "primary_decision_population_size",
        "largest_blocker", "largest_blocker_rate", "rescue_count", "harm_count"))
    by_seed = {integer(row["seed"], "seed summary seed"): row for row in seed_summary}
    if len(by_seed) != 3 or set(by_seed) != {183, 184, 185}:
        raise ValueError("Stage196-B2-A seed summary must uniquely cover seeds 183-185")
    raw_positive = analysis.get("positive_frame_shift_seeds")
    raw_contrast = analysis.get("negative_frame_shift_contrast_seeds")
    if type(raw_positive) is not list or type(raw_contrast) is not list:
        raise ValueError("Stage196-B2-A derived seed roles are absent")
    positive = [integer(value, "positive seed") for value in raw_positive]
    contrast = [integer(value, "contrast seed") for value in raw_contrast]
    derived_ok = (positive == [seed for seed, row in sorted(by_seed.items())
                               if row["common_recurrent_frame_shift"] == "positive"] and
                  contrast == [seed for seed, row in sorted(by_seed.items())
                               if row["common_recurrent_frame_shift"] == "negative"])
    add_gate(gates, "source", "", "derived_seed_roles", {"positive": [184, 185], "contrast": [183]},
             {"positive": positive, "contrast": contrast},
             derived_ok and positive == [184, 185] and contrast == [183],
             "positive/contrast seed roles do not reproduce the completed B2-A source")
    observed_population = {seed: integer(by_seed[seed]["primary_decision_population_size"],
                                         "primary population") for seed in positive}
    add_gate(gates, "source", "", "primary_decision_populations", EXPECTED_DECISION_POPULATIONS,
             observed_population, observed_population == EXPECTED_DECISION_POPULATIONS,
             "B2-A primary decision population counts disagree")
    blockers_ok = all(by_seed[seed]["largest_blocker"] == "FRAME_REMAINS_SUBTHRESHOLD" and
                      math.isclose(number(by_seed[seed]["largest_blocker_rate"], "blocker rate"),
                                   1.0, rel_tol=0.0, abs_tol=1e-12) for seed in positive)
    add_gate(gates, "source", "", "positive_seed_largest_blockers",
             {str(seed): ["FRAME_REMAINS_SUBTHRESHOLD", 1.0] for seed in positive},
             {str(seed): [by_seed[seed]["largest_blocker"], by_seed[seed]["largest_blocker_rate"]]
              for seed in positive}, blockers_ok, "B2-A largest-blocker closure disagreement")
    source_counts = {seed: {"recovery": integer(by_seed[seed]["rescue_count"], "rescue count"),
                            "harm": integer(by_seed[seed]["harm_count"], "harm count")}
                     for seed in positive}
    add_gate(gates, "source", "", "positive_seed_rescue_harm_counts",
             EXPECTED_PRIMARY_COUNTS, source_counts, source_counts == EXPECTED_PRIMARY_COUNTS,
             "B2-A rescue/harm count closure disagreement")
    epochs = {(integer(row.get("seed", ""), "epoch seed"),
               integer(row.get("epoch", ""), "epoch")) for row in epoch}
    epoch_ok = epochs == {(seed, value) for seed in (183, 184, 185) for value in range(1, 21)}
    add_gate(gates, "source", "", "epoch_seed_grid", "3 seeds x epochs 1-20",
             len(epochs), epoch_ok, "B2-A epoch table does not form the exact 3x20 grid")
    return {"repo": repo, "directory": directory, "output": output, "analysis": analysis,
            "contract": contract, "normalized": normalized, "warnings": warnings,
            "positive": positive, "contrast": contrast, "seed_summary": seed_summary,
            "transitions": transitions, "channel_summary": channel_summary,
            "recurrent": recurrent, "harm": harm, "epoch": epoch}


def transition_index(rows: list[dict[str, str]]) -> dict[tuple[int, str, str], dict[str, str]]:
    required = (
        "seed", "stable_row_id", "dev_position", "analysis_view", "gold_final_label",
        "intervention_type", "joint_final_class_or_pattern",
        "intervention_final_class_or_pattern", "joint_tail3_status",
        "intervention_tail3_status", "paired_transition_class",
        *[f"joint_{channel}_value" for channel in TRACE_CHANNELS],
        *[f"intervention_{channel}_value" for channel in TRACE_CHANNELS],
        *[f"paired_{channel}_delta" for channel in TRACE_CHANNELS],
        *[f"{channel}_joint_pass" for channel in LOCAL_CHANNELS],
        *[f"in_{name}" for name in RECURRENT_SETS],
    )
    require_columns("support transition", rows, required)
    result: dict[tuple[int, str, str], dict[str, str]] = {}
    identities: dict[tuple[int, str], int] = {}
    for row in rows:
        seed = integer(row["seed"], "transition seed")
        identifier = row["stable_row_id"]
        view = row["analysis_view"]
        position = integer(row["dev_position"], "transition dev position")
        if not identifier or view not in ("selected", "tail3"):
            raise ValueError("support transition has invalid stable identity or analysis view")
        key = (seed, identifier, view)
        if key in result:
            raise ValueError(f"duplicate support transition row: {key}")
        identity_key = (seed, identifier)
        if identity_key in identities and identities[identity_key] != position:
            raise ValueError(f"stable identity position drift: {identity_key}")
        identities[identity_key] = position
        result[key] = row
    return result


def recurrent_index(rows: list[dict[str, str]]) -> dict[tuple[int, str], set[tuple[str, int]]]:
    require_columns("recurrent position", rows,
                    ("seed", "recurrent_set", "stable_row_id", "dev_position"))
    result: dict[tuple[int, str], set[tuple[str, int]]] = {}
    for row in rows:
        seed = integer(row["seed"], "recurrent seed")
        name = row["recurrent_set"]
        if name not in RECURRENT_SETS:
            raise ValueError(f"unknown recurrent set: {name}")
        member = (row["stable_row_id"], integer(row["dev_position"], "recurrent position"))
        bucket = result.setdefault((seed, name), set())
        if member in bucket:
            raise ValueError(f"duplicate recurrent membership: {(seed, name, member)}")
        bucket.add(member)
    expected_keys = {(seed, name) for seed in (183, 184, 185) for name in RECURRENT_SETS}
    if set(result) != expected_keys:
        raise ValueError("recurrent-position table lacks exact seed-by-set closure")
    return result


def state(values: dict[str, float]) -> tuple[dict[str, bool], str, str, dict[str, float]]:
    passes = {"frame": values["frame"] >= THRESHOLD,
              "predicate": values["predicate"] >= THRESHOLD,
              "sufficiency": values["sufficiency"] >= THRESHOLD,
              "polarity": values["polarity"] >= 0.0,
              "entitlement": values["entitlement"] >= THRESHOLD}
    signature = "|".join("1" if passes[channel] else "0" for channel in LOCAL_CHANNELS)
    downstream = all(passes[channel] for channel in LOCAL_CHANNELS[1:])
    if not passes["frame"] and downstream:
        state_class = "ISOLATED_FRAME_DEFICIT"
    elif not passes["frame"]:
        state_class = "FRAME_PLUS_DOWNSTREAM_DEFICIT"
    elif downstream:
        state_class = "ALL_LOCAL_CHANNELS_PASS"
    else:
        state_class = "DOWNSTREAM_DEFICIT_WITH_FRAME_PASS"
    headrooms = {channel: values[channel] - (0.0 if channel == "polarity" else THRESHOLD)
                 for channel in LOCAL_CHANNELS}
    return passes, signature, state_class, headrooms


def build_profiles(context: dict[str, Any], gates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[int, str, str], dict[str, str]]]:
    transitions = transition_index(context["transitions"])
    recurrent = recurrent_index(context["recurrent"])
    for (seed, identifier, view), transition in transitions.items():
        position = integer(transition["dev_position"], "transition position")
        for name in RECURRENT_SETS:
            expected_member = (identifier, position) in recurrent[(seed, name)]
            if boolean(transition[f"in_{name}"], f"transition {name}") != expected_member:
                raise ValueError(f"whole-table recurrent membership disagreement: {(seed, identifier, view, name)}")
    for (seed, name), members in recurrent.items():
        for identifier, position in members:
            for view in ("selected", "tail3"):
                key = (seed, identifier, view)
                if key not in transitions or integer(transitions[key]["dev_position"], "recurrent join position") != position:
                    raise ValueError(f"recurrent identity absent or misaligned in transitions: {(seed, name, identifier, view)}")
    harm_rows = context["harm"]
    required = (
        "seed", "stable_row_id", "dev_position", "transition_role", "intervention_type",
        "joint_tail3_status", "intervention_tail3_status", "selected_class_transition",
        *[f"in_{name}" for name in RECURRENT_SETS],
        *[f"joint_{channel}_value" for channel in TRACE_CHANNELS],
        *[f"intervention_{channel}_value" for channel in TRACE_CHANNELS],
        *[f"paired_{channel}_delta" for channel in TRACE_CHANNELS],
    )
    require_columns("harm/rescue", harm_rows, required)
    seen: set[tuple[int, str, str, str]] = set()
    role_ids: dict[tuple[int, str], set[str]] = {}
    profiles: list[dict[str, Any]] = []
    source_harm_index: dict[tuple[int, str, str], dict[str, str]] = {}
    role_map = {"RESCUE": "recovery", "INTERVENTION_INDUCED_HARM": "harm"}
    for source_row in harm_rows:
        seed = integer(source_row["seed"], "harm/rescue seed")
        source_role = source_row["transition_role"]
        if source_role not in role_map:
            raise ValueError(f"unknown harm/rescue transition role: {source_role}")
        role = role_map[source_role]
        identifier = source_row["stable_row_id"]
        position = integer(source_row["dev_position"], "harm/rescue dev position")
        duplicate_key = (seed, "tail3", role, identifier)
        if duplicate_key in seen:
            raise ValueError(f"duplicate row within seed x tail3 x transition role: {duplicate_key}")
        seen.add(duplicate_key)
        role_ids.setdefault((seed, role), set()).add(identifier)
        source_harm_index[(seed, role, identifier)] = source_row
        tail_key = (seed, identifier, "tail3")
        selected_key = (seed, identifier, "selected")
        if tail_key not in transitions or selected_key not in transitions:
            raise ValueError(f"primary source row absent from support-transition table: {tail_key}")
        tail, selected = transitions[tail_key], transitions[selected_key]
        if integer(tail["dev_position"], "tail position") != position or tail["intervention_type"] != source_row["intervention_type"]:
            raise ValueError(f"stable identity/intervention type disagreement: {tail_key}")
        if (tail["joint_tail3_status"] != source_row["joint_tail3_status"] or
                tail["intervention_tail3_status"] != source_row["intervention_tail3_status"]):
            raise ValueError(f"tail-three status disagreement: {tail_key}")
        selected_pair = cell(source_row["selected_class_transition"])
        if selected_pair != [cell(selected["joint_final_class_or_pattern"]),
                             cell(selected["intervention_final_class_or_pattern"])]:
            raise ValueError(f"selected-view transition disagreement: {selected_key}")
        if tail["gold_final_label"] != "SUPPORT":
            raise ValueError(f"primary rescue/harm row is not gold SUPPORT: {tail_key}")
        if role == "recovery":
            definition_ok = (source_row["joint_tail3_status"] == "PERSISTENT_NOT_ENTITLED" and
                             source_row["intervention_tail3_status"] == "STABLE_SUPPORT" and
                             tail["paired_transition_class"] == "RESCUE_NE_TO_STABLE_SUPPORT")
        else:
            definition_ok = (source_row["joint_tail3_status"] == "STABLE_SUPPORT" and
                             source_row["intervention_tail3_status"] != "STABLE_SUPPORT")
        if not definition_ok:
            raise ValueError(f"source row violates precommitted {role} definition: {tail_key}")
        for name in RECURRENT_SETS:
            expected_member = (identifier, position) in recurrent[(seed, name)]
            source_member = boolean(source_row[f"in_{name}"], f"harm {name}")
            transition_member = boolean(tail[f"in_{name}"], f"transition {name}")
            if source_member != expected_member or transition_member != expected_member:
                raise ValueError(f"recurrent membership disagreement: {(seed, identifier, name)}")
        for channel in TRACE_CHANNELS:
            for prefix in ("joint", "intervention", "paired"):
                suffix = "delta" if prefix == "paired" else "value"
                key = f"{prefix}_{channel}_{suffix}"
                if not math.isclose(number(source_row[key], key), number(tail[key], key),
                                    rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"harm/support-transition scalar disagreement: {(tail_key, key)}")
        if seed not in context["positive"]:
            continue
        values = {channel: number(tail[f"joint_{channel}_value"], channel)
                  for channel in LOCAL_CHANNELS}
        passes, signature, state_class, headrooms = state(values)
        for channel in LOCAL_CHANNELS:
            if boolean(tail[f"{channel}_joint_pass"], f"{channel} joint pass") != passes[channel]:
                raise ValueError(f"native pass-state disagreement: {(tail_key, channel)}")
        final_margin = number(tail["joint_support_vs_not_entitled_margin_value"], "final margin")
        composition_pass = final_margin >= 0.0
        composition_signature = signature + ("|1" if composition_pass else "|0")
        profile: dict[str, Any] = {
            "id": identifier, "source_row_id": identifier, "stable_row_id": identifier,
            "dev_position": position, "seed": seed, "analysis_view": "tail3",
            "transition_role": role, "intervention_type": tail["intervention_type"],
            "local_signature": signature, "mechanistic_state_class": state_class,
            "trace_joint_tail3_status": tail["joint_tail3_status"],
            "trace_intervention_tail3_status": tail["intervention_tail3_status"],
            "trace_selected_joint_final": selected["joint_final_class_or_pattern"],
            "trace_selected_intervention_final": selected["intervention_final_class_or_pattern"],
            "trace_selected_paired_transition": selected["paired_transition_class"],
            "secondary_joint_support_vs_not_entitled_margin": final_margin,
            "secondary_joint_support_probability": number(tail["joint_support_probability_value"], "support probability"),
            "secondary_joint_not_entitled_probability": number(tail["joint_not_entitled_probability_value"], "NE probability"),
            "secondary_final_composition_headroom": final_margin,
            "secondary_composition_pass": composition_pass,
            "secondary_composition_augmented_signature": composition_signature,
            "secondary_view_label": "composition_augmented_non_authorizing",
            "source_row_provenance": {
                "identity_semantics": "B2-A stable_row_id = selected id cross-checked with trajectory source_row_id",
                "stage196b2a_analyzer_git_commit": context["normalized"]["stage196b2a_analyzer_git_commit"],
                "primary_population": "stage196b2a_harm_rescue_rows.csv",
                "native_state": "stage196b2a_support_transition_rows.csv tail3 joint arm",
                "recurrent_membership": "stage196b2a_recurrent_position_propagation.csv",
            },
        }
        for name in RECURRENT_SETS:
            profile[f"in_{name}"] = (identifier, position) in recurrent[(seed, name)]
        for channel in LOCAL_CHANNELS:
            profile[f"joint_{channel}_value"] = values[channel]
            profile[f"{channel}_pass"] = passes[channel]
            profile[f"{channel}_headroom"] = headrooms[channel]
        for channel in TRACE_CHANNELS:
            profile[f"trace_intervention_{channel}_value"] = number(
                tail[f"intervention_{channel}_value"], f"intervention {channel}")
            profile[f"trace_paired_{channel}_delta"] = number(
                tail[f"paired_{channel}_delta"], f"paired {channel}")
        profiles.append(profile)
    for seed in sorted({seed for seed, _ in role_ids}):
        overlap = role_ids.get((seed, "recovery"), set()) & role_ids.get((seed, "harm"), set())
        if overlap:
            raise ValueError(f"recovery/harm overlap for seed {seed}: {sorted(overlap)}")
    observed = {seed: {role: len(role_ids.get((seed, role), set())) for role in ("recovery", "harm")}
                for seed in context["positive"]}
    add_gate(gates, "population", "", "exact_disjoint_primary_populations",
             EXPECTED_PRIMARY_COUNTS, observed,
             observed == EXPECTED_PRIMARY_COUNTS and len(profiles) == 16,
             "primary tail-three recovery/harm populations do not reproduce 16 disjoint rows")
    return sorted(profiles, key=lambda row: (row["seed"], row["transition_role"], row["dev_position"])), source_harm_index


def distribution(values: Sequence[str]) -> tuple[dict[str, int], dict[str, float]]:
    counts = dict(sorted(Counter(values).items()))
    total = len(values)
    return counts, {key: value / total for key, value in counts.items()}


def headroom_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "minimum": ordered[0],
            "maximum": ordered[-1], "ordered_raw_values": ordered,
            "fail_side_count": sum(value < 0.0 for value in ordered),
            "pass_side_count": sum(value >= 0.0 for value in ordered)}


def group_summaries(profiles: list[dict[str, Any]], positive: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in positive:
        for role in ("recovery", "harm"):
            subset = [row for row in profiles if row["seed"] == seed and row["transition_role"] == role]
            class_counts, class_rates = distribution([row["mechanistic_state_class"] for row in subset])
            sig_counts, sig_rates = distribution([row["local_signature"] for row in subset])
            type_counts, type_rates = distribution([row["intervention_type"] for row in subset])
            result: dict[str, Any] = {
                "seed": seed, "analysis_view": "tail3", "transition_role": role,
                "row_count": len(subset), "native_state_class_counts": class_counts,
                "native_state_class_rates": class_rates, "local_signature_counts": sig_counts,
                "local_signature_rates": sig_rates, "intervention_type_counts": type_counts,
                "intervention_type_rates": type_rates,
                "universal_all_six_membership_count": sum(
                    row["in_stage196a_universal_all_six"] for row in subset),
            }
            for channel in LOCAL_CHANNELS:
                result[f"{channel}_pass_rate"] = rate(
                    sum(row[f"{channel}_pass"] for row in subset), len(subset))
                result[f"{channel}_headroom_summary"] = headroom_summary(
                    [row[f"{channel}_headroom"] for row in subset])
            for name in RECURRENT_SETS:
                result[f"in_{name}_count"] = sum(row[f"in_{name}"] for row in subset)
            rows.append(result)
    return rows


def overlap_rows(profiles: list[dict[str, Any]], positive: Sequence[int],
                 field: str, view: str, authority: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, set[str]]]]:
    rows: list[dict[str, Any]] = []
    sets_by_seed: dict[int, dict[str, set[str]]] = {}
    for seed in positive:
        recovery_rows = [row for row in profiles if row["seed"] == seed and row["transition_role"] == "recovery"]
        harm_rows = [row for row in profiles if row["seed"] == seed and row["transition_role"] == "harm"]
        recovery = {row[field] for row in recovery_rows}
        harm = {row[field] for row in harm_rows}
        shared, recovery_only, harm_only = recovery & harm, recovery - harm, harm - recovery
        union = recovery | harm
        recovery_n = sum(row[field] in recovery_only for row in recovery_rows)
        harm_n = sum(row[field] in harm_only for row in harm_rows)
        collision_rows = sum(row[field] in shared for row in recovery_rows + harm_rows)
        sets_by_seed[seed] = {"recovery": recovery, "harm": harm, "shared": shared,
                              "recovery_exclusive": recovery_only, "harm_exclusive": harm_only}
        rows.append({
            "seed": seed, "signature_view": view, "view_authority": authority,
            "recovery_signature_set": sorted(recovery), "harm_signature_set": sorted(harm),
            "shared_signature_set": sorted(shared),
            "recovery_exclusive_signatures": sorted(recovery_only),
            "harm_exclusive_signatures": sorted(harm_only),
            "signature_set_jaccard_overlap": len(shared) / len(union) if union else None,
            "recovery_exclusive_row_coverage": rate(recovery_n, len(recovery_rows)),
            "recovery_exclusive_coverage_numerator": recovery_n,
            "recovery_exclusive_coverage_denominator": len(recovery_rows),
            "harm_exclusive_row_coverage": rate(harm_n, len(harm_rows)),
            "harm_exclusive_coverage_numerator": harm_n,
            "harm_exclusive_coverage_denominator": len(harm_rows),
            "collision_signatures": sorted(shared), "collision_row_count": collision_rows,
        })
    return rows, sets_by_seed


def transfer_rows(profiles: list[dict[str, Any]], positive: Sequence[int], field: str,
                  view: str, authority: str,
                  signature_sets: dict[int, dict[str, set[str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_seed, target_seed in ((positive[0], positive[1]), (positive[1], positive[0])):
        recovery_signatures = signature_sets[source_seed]["recovery_exclusive"]
        harm_signatures = signature_sets[source_seed]["harm_exclusive"]
        target = [row for row in profiles if row["seed"] == target_seed]
        recovery = [row for row in target if row["transition_role"] == "recovery"]
        harm = [row for row in target if row["transition_role"] == "harm"]
        recovery_hits = [row for row in target if row[field] in recovery_signatures]
        harm_hits = [row for row in target if row[field] in harm_signatures]
        recovery_match = sum(row[field] in recovery_signatures for row in recovery)
        harm_match = sum(row[field] in harm_signatures for row in harm)
        recovery_collision = sum(row[field] in recovery_signatures for row in harm)
        harm_collision = sum(row[field] in harm_signatures for row in recovery)
        covered = [row for row in target if row[field] in recovery_signatures | harm_signatures]
        uncovered_recovery = sum(row[field] not in recovery_signatures | harm_signatures for row in recovery)
        uncovered_harm = sum(row[field] not in recovery_signatures | harm_signatures for row in harm)
        exact_stable = (recovery_match == len(recovery) and harm_match == len(harm) and
                        recovery_collision == 0 and harm_collision == 0 and len(covered) == len(target))
        rows.append({
            "source_seed": source_seed, "target_seed": target_seed,
            "signature_view": view, "view_authority": authority,
            "source_recovery_exclusive_signatures": sorted(recovery_signatures),
            "source_harm_exclusive_signatures": sorted(harm_signatures),
            "target_recovery_coverage": rate(recovery_match, len(recovery)),
            "target_recovery_coverage_numerator": recovery_match,
            "target_recovery_coverage_denominator": len(recovery),
            "target_recovery_purity": rate(recovery_match, len(recovery_hits)),
            "target_recovery_purity_numerator": recovery_match,
            "target_recovery_purity_denominator": len(recovery_hits),
            "target_recovery_collision_count": recovery_collision,
            "target_harm_coverage": rate(harm_match, len(harm)),
            "target_harm_coverage_numerator": harm_match,
            "target_harm_coverage_denominator": len(harm),
            "target_harm_purity": rate(harm_match, len(harm_hits)),
            "target_harm_purity_numerator": harm_match,
            "target_harm_purity_denominator": len(harm_hits),
            "target_harm_collision_count": harm_collision,
            "target_uncovered_row_count": len(target) - len(covered),
            "target_uncovered_recovery_count": uncovered_recovery,
            "target_uncovered_harm_count": uncovered_harm,
            "exact_transfer_stable": exact_stable,
        })
    return rows


def evaluate_rule(profiles: list[dict[str, Any]], positive: Sequence[int], name: str,
                  recovery_rule: Callable[[dict[str, Any]], bool],
                  harm_rule: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    all_pass = True
    for seed in positive:
        recovery = [row for row in profiles if row["seed"] == seed and row["transition_role"] == "recovery"]
        harm = [row for row in profiles if row["seed"] == seed and row["transition_role"] == "harm"]
        rc_n, rc_d = sum(recovery_rule(row) for row in recovery), len(recovery)
        rcont_n, rcont_d = sum(recovery_rule(row) for row in harm), len(harm)
        hc_n, hc_d = sum(harm_rule(row) for row in harm), len(harm)
        hcont_n, hcont_d = sum(harm_rule(row) for row in recovery), len(recovery)
        rc, rcont, hc, hcont = rate(rc_n, rc_d), rate(rcont_n, rcont_d), rate(hc_n, hc_d), rate(hcont_n, hcont_d)
        seed_pass = (rc is not None and rc >= 0.75 and rcont is not None and rcont <= 0.25 and
                     hc is not None and hc >= 0.75 and hcont is not None and hcont <= 0.25 and
                     rc_d >= 2 and hc_d >= 3)
        all_pass = all_pass and seed_pass
        per_seed[str(seed)] = {
            "recovery_coverage": rc, "recovery_coverage_fraction": [rc_n, rc_d],
            "recovery_rule_contamination_among_harms": rcont,
            "recovery_contamination_fraction": [rcont_n, rcont_d],
            "harm_coverage": hc, "harm_coverage_fraction": [hc_n, hc_d],
            "harm_rule_contamination_among_recoveries": hcont,
            "harm_contamination_fraction": [hcont_n, hcont_d],
            "balanced_coverage_mean": (rc + hc) / 2 if rc is not None and hc is not None else None,
            "minimum_population_requirements_met": rc_d >= 2 and hc_d >= 3,
            "all_four_conditions_pass": seed_pass,
        }
    return {"rule": name, "evaluated": True, "per_seed": per_seed,
            "passes_every_positive_seed": all_pass,
            "thresholds": {"recovery_coverage_minimum": 0.75,
                           "recovery_rule_harm_contamination_maximum": 0.25,
                           "harm_coverage_minimum": 0.75,
                           "harm_rule_recovery_contamination_maximum": 0.25,
                           "minimum_recovery_rows_per_seed": 2,
                           "minimum_harm_rows_per_seed": 3}}


def intervention_summary(profiles: list[dict[str, Any]], positive: Sequence[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed_sets: dict[int, dict[str, set[str]]] = {}
    for seed in positive:
        recovery = {row["intervention_type"] for row in profiles
                    if row["seed"] == seed and row["transition_role"] == "recovery"}
        harm = {row["intervention_type"] for row in profiles
                if row["seed"] == seed and row["transition_role"] == "harm"}
        per_seed_sets[seed] = {"recovery_only": recovery - harm, "harm_only": harm - recovery,
                               "both": recovery & harm}
    persistent_recovery = set.intersection(*(per_seed_sets[seed]["recovery_only"] for seed in positive))
    persistent_harm = set.intersection(*(per_seed_sets[seed]["harm_only"] for seed in positive))
    rows: list[dict[str, Any]] = []
    for seed in positive:
        seed_rows = [row for row in profiles if row["seed"] == seed]
        recovery_total = sum(row["transition_role"] == "recovery" for row in seed_rows)
        harm_total = sum(row["transition_role"] == "harm" for row in seed_rows)
        types = sorted({row["intervention_type"] for row in seed_rows})
        for intervention_type in types:
            recovery_count = sum(row["transition_role"] == "recovery" and
                                 row["intervention_type"] == intervention_type for row in seed_rows)
            harm_count = sum(row["transition_role"] == "harm" and
                             row["intervention_type"] == intervention_type for row in seed_rows)
            presence = "both" if recovery_count and harm_count else "recovery_only" if recovery_count else "harm_only"
            rows.append({
                "seed": seed, "intervention_type": intervention_type,
                "recovery_count": recovery_count, "harm_count": harm_count,
                "recovery_rate_within_seed_role": rate(recovery_count, recovery_total),
                "harm_rate_within_seed_role": rate(harm_count, harm_total),
                "role_presence": presence, "recovery_only_in_seed": presence == "recovery_only",
                "harm_only_in_seed": presence == "harm_only", "appears_in_both": presence == "both",
                "recovery_exclusive_persists_across_positive_seeds": intervention_type in persistent_recovery,
                "harm_exclusive_persists_across_positive_seeds": intervention_type in persistent_harm,
            })
    summary = {str(seed): {key: sorted(value) for key, value in per_seed_sets[seed].items()}
               for seed in positive}
    summary["cross_seed_persistent_recovery_only"] = sorted(persistent_recovery)
    summary["cross_seed_persistent_harm_only"] = sorted(persistent_harm)
    summary["authorizing"] = False
    return rows, summary


def contrast_summary(context: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in context["harm"]
            if integer(row["seed"], "contrast seed") in context["contrast"]]
    return {"seeds": context["contrast"], "decision_denominator_rows": 0,
            "harm_rescue_row_count": len(rows),
            "transition_role_counts": dict(sorted(Counter(row["transition_role"] for row in rows).items())),
            "intervention_type_counts": dict(sorted(Counter(row["intervention_type"] for row in rows).items())),
            "interpretation": "direction-visible contrast only; excluded from primary bifurcation decisions"}


def choose_decision(rule_a: dict[str, Any], rule_b: dict[str, Any],
                    local_overlap: list[dict[str, Any]], composition_overlap: list[dict[str, Any]],
                    local_transfer: list[dict[str, Any]], composition_transfer: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if rule_a["passes_every_positive_seed"]:
        return DECISIONS[0], {"rule_a_passed": True, "rule_b_evaluated": False,
                              "selection_order_respected": True}
    if rule_b["passes_every_positive_seed"]:
        return DECISIONS[1], {"rule_a_passed": False, "rule_b_passed": True,
                              "selection_order_respected": True}
    local_collisions = sum(row["collision_row_count"] for row in local_overlap)
    composition_collisions = sum(row["collision_row_count"] for row in composition_overlap)
    local_stable = all(row["exact_transfer_stable"] for row in local_transfer)
    composition_stable = all(row["exact_transfer_stable"] for row in composition_transfer)
    composition_role_exclusive = all(not row["shared_signature_set"] for row in composition_overlap)
    materially_reduced = composition_collisions < local_collisions
    composition_only = ((local_collisions > 0 or not local_stable) and
                        (composition_role_exclusive or materially_reduced) and composition_stable)
    decision = DECISIONS[2] if composition_only else DECISIONS[3]
    return decision, {
        "rule_a_passed": False, "rule_b_passed": False,
        "local_collision_row_count": local_collisions,
        "composition_collision_row_count": composition_collisions,
        "local_exact_cross_seed_transfer_stable": local_stable,
        "composition_exact_cross_seed_transfer_stable": composition_stable,
        "composition_role_exclusive_in_both_seeds": composition_role_exclusive,
        "composition_materially_reduces_collisions": materially_reduced,
        "composition_only_decision_conditions_met": composition_only,
        "selection_order_respected": True,
    }


def analyze_complete(args: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    context = load_source(args, gates)
    profiles, _ = build_profiles(context, gates)
    positive = context["positive"]
    groups = group_summaries(profiles, positive)
    local_rows, local_sets = overlap_rows(profiles, positive, "local_signature",
                                          "five_bit_local", "primary_authorizing")
    composition_rows, composition_sets = overlap_rows(
        profiles, positive, "secondary_composition_augmented_signature",
        "six_bit_composition_augmented", "composition_augmented_non_authorizing")
    local_transfer = transfer_rows(profiles, positive, "local_signature", "five_bit_local",
                                   "primary_authorizing", local_sets)
    composition_transfer = transfer_rows(
        profiles, positive, "secondary_composition_augmented_signature",
        "six_bit_composition_augmented", "composition_augmented_non_authorizing", composition_sets)
    rule_a = evaluate_rule(
        profiles, positive, "Rule A: isolated Frame deficit bifurcation",
        lambda row: row["mechanistic_state_class"] == "ISOLATED_FRAME_DEFICIT",
        lambda row: row["mechanistic_state_class"] == "ALL_LOCAL_CHANNELS_PASS")
    if rule_a["passes_every_positive_seed"]:
        rule_b = {"rule": "Rule B: Frame boundary bifurcation", "evaluated": False,
                  "skipped_reason": "Rule A passed; precommitted order prohibits evaluating Rule B",
                  "per_seed": {}, "passes_every_positive_seed": False}
    else:
        rule_b = evaluate_rule(
            profiles, positive, "Rule B: Frame boundary bifurcation",
            lambda row: not row["frame_pass"], lambda row: row["frame_pass"])
    intervention_rows, intervention = intervention_summary(profiles, positive)
    decision, decision_evaluation = choose_decision(
        rule_a, rule_b, local_rows, composition_rows, local_transfer, composition_transfer)
    add_gate(gates, "decision", "", "precommitted_decision_set", list(DECISIONS), decision,
             decision in DECISIONS[:-1], "decision escaped the precommitted scientific decision set")
    population_counts = {str(seed): {
        role: sum(row["seed"] == seed and row["transition_role"] == role for row in profiles)
        for role in ("recovery", "harm")} for seed in positive}
    report = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": NEXT_STAGE[decision],
        "blocking_reasons": [], "runnable": True,
        "current_analyzer_git_commit": args.current_git_commit,
        "stage196b2a_analyzer_git_commit": context["normalized"]["stage196b2a_analyzer_git_commit"],
        "stage196b2p0_runtime_git_commit": context["normalized"]["stage196b2p0_runtime_git_commit"],
        "stage196b1_runtime_git_commit": context["normalized"]["stage196b1_runtime_git_commit"],
        "framegate_implementation_origin_git_commit": context["normalized"]["framegate_implementation_origin_git_commit"],
        "normalized_support_vs_not_entitled_margin_source": MARGIN_SOURCE,
        "source_schema_warnings": context["warnings"],
        "analysis_view": "tail3", "tail_epochs": list(TAIL),
        "positive_primary_seeds": positive, "contrast_seeds": context["contrast"],
        "primary_population_counts": population_counts, "primary_population_total": len(profiles),
        "rule_a_evaluation": rule_a, "rule_b_evaluation": rule_b,
        "local_signature_overlap": {str(row["seed"]): row for row in local_rows},
        "composition_augmented_signature_overlap": {
            "view_label": "composition_augmented_non_authorizing",
            "per_seed": {str(row["seed"]): row for row in composition_rows}},
        "cross_seed_transfer_summary": {
            "local": local_transfer,
            "composition_augmented_non_authorizing": composition_transfer},
        "intervention_type_summary": intervention,
        "seed183_contrast": contrast_summary(context),
        "decision_rule_evaluation": decision_evaluation,
        "authorized_interpretation": AUTHORIZED[decision],
        "prohibited_interpretations": PROHIBITED,
        "output_file_count": 8,
        "training_performed": False, "checkpoint_loaded": False,
        "model_loaded": False, "external_evaluation_performed": False,
        "artifact_only_analysis": True, "threshold_search_performed": False,
        "classifier_fitted": False,
    }
    return report, {"group": groups, "profiles": profiles,
                    "signatures": local_rows + composition_rows,
                    "transfer": local_transfer + composition_transfer,
                    "intervention": intervention_rows, "contract": gates}


def incomplete_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "stage": STAGE, "decision": INCOMPLETE,
        "recommended_next_stage": NEXT_STAGE[INCOMPLETE],
        "blocking_reasons": [f"{type(exc).__name__}: {exc}"], "runnable": False,
        "current_analyzer_git_commit": args.current_git_commit,
        "stage196b2a_analyzer_git_commit": args.stage196b2a_analyzer_git_commit,
        "stage196b2p0_runtime_git_commit": None, "stage196b1_runtime_git_commit": None,
        "framegate_implementation_origin_git_commit": None,
        "normalized_support_vs_not_entitled_margin_source": None,
        "source_schema_warnings": [], "analysis_view": "tail3", "tail_epochs": list(TAIL),
        "positive_primary_seeds": [], "contrast_seeds": [],
        "primary_population_counts": {}, "primary_population_total": 0,
        "rule_a_evaluation": {"evaluated": False}, "rule_b_evaluation": {"evaluated": False},
        "local_signature_overlap": {},
        "composition_augmented_signature_overlap": {
            "view_label": "composition_augmented_non_authorizing", "per_seed": {}},
        "cross_seed_transfer_summary": {}, "intervention_type_summary": {},
        "seed183_contrast": {}, "decision_rule_evaluation": {"analysis_incomplete": True},
        "authorized_interpretation": AUTHORIZED[INCOMPLETE],
        "prohibited_interpretations": PROHIBITED, "output_file_count": 8,
        "training_performed": False, "checkpoint_loaded": False,
        "model_loaded": False, "external_evaluation_performed": False,
        "artifact_only_analysis": True, "threshold_search_performed": False,
        "classifier_fitted": False,
    }


def markdown(report: dict[str, Any]) -> str:
    complete = report["decision"] != INCOMPLETE
    source_closure = ("Exact nine-file Stage196-B2-A closure passed, including the completed mixed "
                      "decision, fixed next-stage recommendation, empty blockers, three seed rows, "
                      "60 epoch rows, contract gates, populations, blockers, and rescue/harm counts."
                      if complete else "Source closure failed: " + " ".join(report["blocking_reasons"]))
    sections = [
        ("Executive decision", f"`{report['decision']}`"),
        ("Authorized interpretation", report["authorized_interpretation"]),
        ("Stage196-B2-A source closure", source_closure),
        ("Provenance normalization", json.dumps({
            "stage196b2a_analyzer": report["stage196b2a_analyzer_git_commit"],
            "stage196b2p0_runtime": report["stage196b2p0_runtime_git_commit"],
            "stage196b1_runtime": report["stage196b1_runtime_git_commit"],
            "framegate_origin": report["framegate_implementation_origin_git_commit"],
            "margin_source": report["normalized_support_vs_not_entitled_margin_source"],
            "warnings": report["source_schema_warnings"]}, sort_keys=True)),
        ("Primary seeds and populations", json.dumps({"positive": report["positive_primary_seeds"],
            "contrast": report["contrast_seeds"], "counts": report["primary_population_counts"],
            "total": report["primary_population_total"]}, sort_keys=True)),
        ("Native-state feature policy", "Primary authorization uses only tail-three joint-arm Frame, predicate, sufficiency, polarity, and entitlement native values and their fixed pass states. Outcome, label, identity, intervention-arm, delta, final prediction/status, and final composition fields are excluded."),
        ("Recovery population", "Joint tail-three PERSISTENT_NOT_ENTITLED to intervention tail-three STABLE_SUPPORT, source role RESCUE, gold SUPPORT; selected-only rows are excluded."),
        ("Preservation-harm population", "Joint tail-three STABLE_SUPPORT to intervention tail-three non-STABLE_SUPPORT, source role INTERVENTION_INDUCED_HARM, gold SUPPORT; selected-only rows are excluded."),
        ("Local-state classes", "Exhaustive classes are ISOLATED_FRAME_DEFICIT, FRAME_PLUS_DOWNSTREAM_DEFICIT, ALL_LOCAL_CHANNELS_PASS, and DOWNSTREAM_DEFICIT_WITH_FRAME_PASS."),
        ("Fixed Rule A evaluation", json.dumps(report["rule_a_evaluation"], sort_keys=True)),
        ("Fixed Rule B evaluation", json.dumps(report["rule_b_evaluation"], sort_keys=True)),
        ("Signature overlap", json.dumps(report["local_signature_overlap"], sort_keys=True)),
        ("Cross-seed signature transfer", json.dumps(report["cross_seed_transfer_summary"].get("local", []), sort_keys=True)),
        ("Composition-augmented non-authorizing view", json.dumps({
            "overlap": report["composition_augmented_signature_overlap"],
            "transfer": report["cross_seed_transfer_summary"].get("composition_augmented_non_authorizing", [])}, sort_keys=True)),
        ("Intervention-type audit", json.dumps(report["intervention_type_summary"], sort_keys=True)),
        ("Seed183 contrast", json.dumps(report["seed183_contrast"], sort_keys=True)),
        ("Decision-rule evaluation", json.dumps(report["decision_rule_evaluation"], sort_keys=True)),
        ("Remaining uncertainty", "This frozen-Mamba artifact-only audit is descriptive. Exact native-state associations cannot establish formal mediation, safe routing, unfrozen behavior, or external validity."),
        ("Prohibited claims", "\n".join(f"- {claim}" for claim in report["prohibited_interpretations"])),
        ("Recommended next stage", f"`{report['recommended_next_stage']}`\n\nNo outcome automatically authorizes implementation, training, a new loss, checkpoint changes, or external evaluation."),
    ]
    return "# Stage196-B2-B1 Frame Recovery vs Preservation Harm Bifurcation Audit\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sections) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def render_csv(header: list[str], rows: Iterable[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=header, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header):
            raise ValueError(f"generated CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return handle.getvalue()


def empty_tables(gates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"group": [], "profiles": [], "signatures": [], "transfer": [],
            "intervention": [], "contract": gates}


def render_outputs(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {
        "stage196b2b1_analysis.json": json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "stage196b2b1_report.md": markdown(report),
        "stage196b2b1_group_summary.csv": render_csv(GROUP_HEADER, tables["group"]),
        "stage196b2b1_row_profiles.csv": render_csv(ROW_HEADER, tables["profiles"]),
        "stage196b2b1_signature_summary.csv": render_csv(SIGNATURE_HEADER, tables["signatures"]),
        "stage196b2b1_cross_seed_transfer.csv": render_csv(TRANSFER_HEADER, tables["transfer"]),
        "stage196b2b1_intervention_type_summary.csv": render_csv(INTERVENTION_HEADER, tables["intervention"]),
        "stage196b2b1_contract.csv": render_csv(CONTRACT_HEADER, tables["contract"]),
    }


def write_exact(output: Path, rendered: dict[str, str]) -> None:
    if set(rendered) != set(OUTPUTS):
        raise ValueError("internal eight-output closure mismatch")
    output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(rendered[name])
    if {path.name for path in output.iterdir()} != set(OUTPUTS):
        raise RuntimeError("written output closure mismatch")


def main() -> int:
    args = parse_args()
    gates: list[dict[str, Any]] = []
    try:
        report, tables = analyze_complete(args, gates)
        rendered = render_outputs(report, tables)
    except Exception as exc:
        gates.append({"scope": "analysis", "run": "", "gate": "analysis_completed",
                      "required": True, "observed": False, "passed": False,
                      "blocking_reason": f"{type(exc).__name__}: {exc}"})
        report, tables = incomplete_report(args, exc), empty_tables(gates)
        rendered = render_outputs(report, tables)
    output = args.output_dir.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("output directory exists and is nonempty")
        output.rmdir()
    write_exact(output, rendered)
    print(json.dumps({"decision": report["decision"], "output_dir": str(output),
                      "output_files": list(OUTPUTS)}, sort_keys=True))
    return 0 if report["decision"] != INCOMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
