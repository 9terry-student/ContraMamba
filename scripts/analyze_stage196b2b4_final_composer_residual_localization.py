#!/usr/bin/env python3
"""Stage196-B2-B4 artifact-only exact primitive/residual coalition localization.

The analyzer consumes only explicitly supplied artifacts.  It enumerates both
five-element Boolean lattices, independently recomposes every output from raw
composer inputs, and never imports a model or loads a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


STAGE = "Stage196-B2-B4"
TOL = 1e-6
P0_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
SEEDS = (183, 184, 185)
PRIMARY_SEEDS = (184, 185)
EPOCHS = tuple(range(1, 21))
TAIL = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{s}_{m}" for s in SEEDS for m in MODES)
DIRECTIONS = (
    "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR",
    "FRAME_LOCAL_ONLY_RECIPIENT_JOINT_DONOR",
)
DIRECTION_MODES = {
    DIRECTIONS[0]: ("joint", "frame_local_only"),
    DIRECTIONS[1]: ("frame_local_only", "joint"),
}
PRIMITIVES = (
    "FRAME", "PREDICATE", "SUFFICIENCY", "POSITIVE_ENERGY", "NEGATIVE_ENERGY",
)
PRIMITIVE_FIELDS = {
    "FRAME": ("frame_prob",),
    "PREDICATE": ("predicate_coverage_prob",),
    "SUFFICIENCY": ("sufficiency_prob",),
    "POSITIVE_ENERGY": ("positive_energy",),
    "NEGATIVE_ENERGY": ("negative_energy",),
}
RESIDUAL_GROUPS = (
    "DECISION_HEAD_CALIBRATION", "PREDICATE_COMPARATOR", "TEMPORAL_COMPARATOR",
    "TEMPORAL_ADAPTER", "TEMPORAL_CHANNEL",
)
# Only raw causal inputs are installed.  Every transformed/effective field is
# independently recomputed by reconstruct(); exported derived fields are targets.
RESIDUAL_FIELDS = {
    "DECISION_HEAD_CALIBRATION": ("not_entitled_bias", "raw_alpha"),
    "PREDICATE_COMPARATOR": ("predicate_mismatch_available", "predicate_mismatch_condition_input", "raw_alpha_predicate"),
    "TEMPORAL_COMPARATOR": ("temporal_mismatch_available", "temporal_mismatch_condition_input", "raw_alpha_temporal"),
    "TEMPORAL_ADAPTER": ("temporal_adapter_available", "temporal_adapter_logit", "temporal_adapter_final_penalty_scale"),
    "TEMPORAL_CHANNEL": ("temporal_channel_available", "temporal_channel_logit", "preservation_entitlement_prob", "temporal_channel_gated_penalty_scale"),
}
LEGACY_MASKS = {
    "FRAME_ONLY": "10000", "PREDICATE_ONLY": "01000", "SUFFICIENCY_ONLY": "00100",
    "ENTITLEMENT_PRIMITIVES": "11100", "POLARITY_ONLY": "00011",
    "ENTITLEMENT_PLUS_POLARITY": "11111",
}
R1_FILES = (
    "stage196b2b3r1_analysis.json", "stage196b2b3r1_report.md",
    "stage196b2b3r1_composer_graph.csv", "stage196b2b3r1_native_reconstruction.csv",
    "stage196b2b3r1_component_swap_rows.csv", "stage196b2b3r1_row_swap_summary.csv",
    "stage196b2b3r1_group_swap_summary.csv", "stage196b2b3r1_subtype_summary.csv",
    "stage196b2b3r1_contract.csv",
)
B2B2_FILES = (
    "stage196b2b2_analysis.json", "stage196b2b2_report.md",
    "stage196b2b2_row_path_summary.csv", "stage196b2b2_epoch_paired_paths.csv",
    "stage196b2b2_group_path_summary.csv", "stage196b2b2_event_order_summary.csv",
    "stage196b2b2_intervention_type_paths.csv", "stage196b2b2_contrast_summary.csv",
    "stage196b2b2_contract.csv",
)
OUTPUTS = (
    "stage196b2b4_analysis.json", "stage196b2b4_report.md",
    "stage196b2b4_primitive_coalition_rows.csv", "stage196b2b4_primitive_mobius_terms.csv",
    "stage196b2b4_primitive_tail_summary.csv", "stage196b2b4_residual_coalition_rows.csv",
    "stage196b2b4_residual_mobius_terms.csv", "stage196b2b4_localization_summary.csv",
    "stage196b2b4_contract.csv",
)
EXPECTED_PRIMARY = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}
EXPECTED_R1_NATIVE = {
    "row_count": 86400,
    "maximum_entitlement_error": 5.800120184140667e-08,
    "maximum_decision_head_error": 3.3065668159082406e-07,
    "maximum_branch_delta_sum_error": 0.0,
    "maximum_final_logit_error": 3.3065668159082406e-07,
    "maximum_margin_error": 3.610712273616201e-07,
    "prediction_equality_rate": 1.0,
    "tolerance": 1e-6,
    "passed": True,
}
EXPECTED_R1_POSITIVE = {
    "row_count": 86400, "maximum_absolute_donor_logit_error": 0.0,
    "maximum_donor_margin_error": 0.0, "prediction_equality_rate": 1.0,
    "failing_rows": [], "passed": True,
}
TRAJECTORY_FIELDS = {
    "id", "source_row_id", "dev_position", "gold_label", "prediction",
    "intervention_type", "frame_probability", "predicate_coverage_probability",
    "sufficiency_probability", "polarity_support_margin", "entitlement_probability",
    "support_probability", "not_entitled_probability", "support_logit",
    "not_entitled_logit", "epoch", "training_seed", "frame_downstream_gradient_mode",
}
BRANCHES = ("temporal_mismatch", "predicate_mismatch", "temporal_adapter", "temporal_channel")
CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
COALITION_H = (
    "seed", "epoch", "stable_row_id", "id", "source_row_id", "dev_position",
    "transition_role", "intervention_type", "path_class", "subtype", "direction",
    "recipient_mode", "donor_mode", "coalition_mask", "coalition_size", "coalition_members",
    "recipient_refute_logit", "recipient_not_entitled_logit", "recipient_support_logit",
    "recipient_margin", "recipient_prediction", "donor_refute_logit",
    "donor_not_entitled_logit", "donor_support_logit", "donor_margin", "donor_prediction",
    "counterfactual_refute_logit", "counterfactual_not_entitled_logit",
    "counterfactual_support_logit", "counterfactual_margin", "counterfactual_prediction",
    "counterfactual_minus_recipient_margin", "donor_minus_recipient_margin",
    "prediction_changed", "donor_prediction_reproduced", "recipient_prediction_preserved",
)
MOBIUS_H = (
    "seed", "epoch", "identity", "direction", "lattice", "coalition_mask",
    "coalition_size", "coalition_members", "refute_interaction",
    "not_entitled_interaction", "support_interaction", "margin_interaction",
    "full_effect_refute", "full_effect_not_entitled", "full_effect_support",
    "full_effect_margin", "interaction_sum_error_refute",
    "interaction_sum_error_not_entitled", "interaction_sum_error_support",
    "interaction_sum_error_margin",
)
TAIL_H = (
    "seed", "stable_row_id", "id", "source_row_id", "dev_position", "transition_role",
    "intervention_type", "path_class", "subtype", "direction", "coalition_mask",
    "coalition_size", "coalition_members", "tail_epochs", "recipient_tail_predictions",
    "donor_tail_predictions", "coalition_tail_predictions", "recipient_tail_status",
    "donor_tail_status", "coalition_tail_status", "donor_tail_reproduced",
    "recipient_tail_preserved", "minimal_donor_reproducing_coalition",
)
LOCALIZATION_H = (
    "record_type", "seed", "stable_row_id", "direction", "transition_role", "subtype",
    "coalition_mask", "coalition_members", "criterion", "numerator", "denominator",
    "passed", "details",
)
PROHIBITED = (
    "formal causal mediation", "external or OOD validity", "unfrozen-Mamba validity",
    "training improvement", "promotion", "a universal selector from seed-specific evidence",
    "a causal role for polarity_support_margin",
    "architectural sufficiency beyond the frozen composer",
    "a large interaction magnitude alone as a valid intervention",
    "continuous donor-logit closure as categorical selectivity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, typ in (
        ("repo-root", Path), ("stage196b2b3r1-analysis-json", Path),
        ("stage196b2b2-analysis-json", Path), ("stage196b2b3p0-run-root", Path),
        ("stage196b2b3p0-runtime-git-commit", str), ("current-git-commit", str),
        ("output-dir", Path),
    ):
        parser.add_argument(f"--{name}", required=True, type=typ)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if type(row) is not dict:
                raise ValueError(f"{path}:{line_number}: object required")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any, name: str, probability: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name}: finite JSON number required")
    result = float(value)
    if probability and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name}: probability outside [0,1]")
    return result


def csv_number(value: Any, name: str) -> float:
    if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.strip()):
        value = float(value)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name}: finite CSV number required")
    return float(value)


def integer(value: Any, name: str) -> int:
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        value = int(value)
    if type(value) is not int:
        raise ValueError(f"{name}: integer required")
    return value


def boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name}: JSON boolean required")
    return value


def softplus(value: float) -> float:
    return value + math.log1p(math.exp(-value)) if value > 0.0 else math.log1p(math.exp(value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0.0 else math.exp(value) / (1.0 + math.exp(value))


def argmax_label(logits: Sequence[float]) -> str:
    return ("REFUTE", "NOT_ENTITLED", "SUPPORT")[max(range(3), key=lambda i: logits[i])]


def tail_status(values: Sequence[str]) -> str:
    values = tuple(values)
    if values == ("SUPPORT",) * 3: return "STABLE_SUPPORT"
    if values == ("NOT_ENTITLED",) * 3: return "PERSISTENT_NOT_ENTITLED"
    if values == ("REFUTE",) * 3: return "PERSISTENT_REFUTE"
    return "UNSTABLE"


def mask_members(mask: int, names: Sequence[str]) -> tuple[str, list[str]]:
    bits = f"{mask:05b}"
    return bits, [name for bit, name in zip(bits, names) if bit == "1"]


def is_subset(left: str, right: str) -> bool:
    return all(a == "0" or b == "1" for a, b in zip(left, right))


def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str,
         fatal: bool = True) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": bool(passed),
                 "blocking_reason": "" if passed else reason})
    if fatal and not passed:
        raise ValueError(f"{name}: {reason}")


def exact_directory(path: Path, names: Sequence[str], gates: list[dict[str, Any]], label: str) -> None:
    observed = sorted(p.name for p in path.iterdir() if p.is_file())
    expected = sorted(names)
    gate(gates, "source", "", label, expected,
         {"files": observed, "missing": sorted(set(expected) - set(observed)),
          "unexpected": sorted(set(observed) - set(expected))},
         observed == expected, "exact file closure failed")


def contract_closed(rows: list[dict[str, str]], expected: int) -> bool:
    return len(rows) == expected and all(
        r.get("passed", "").strip().lower() == "true" and not r.get("blocking_reason", "").strip()
        for r in rows
    )


def subtype(meta: dict[str, str]) -> str:
    seed, role, path = integer(meta["seed"], "seed"), meta["transition_role"], meta["path_class"]
    if role == "recovery" and seed == 184 and path == "MULTI_CHANNEL_CONFLICT": return "RECOVERY_SEED184_MULTI_CHANNEL_CONFLICT"
    if role == "recovery" and seed == 185 and path == "FRAME_ENTITLEMENT_GAIN": return "RECOVERY_SEED185_FRAME_ENTITLEMENT_GAIN"
    if path == "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN": return "POLARITY_OVERRIDE_ROWS"
    if path == "COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY": return "COMPOSITION_RESIDUAL_ROWS"
    if path == "FRAME_ENTITLEMENT_LOSS": return "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
    if seed == 185 and role == "harm" and path == "MULTI_CHANNEL_CONFLICT":
        fields = ("tail3_delta_frame", "tail3_delta_predicate", "tail3_delta_entitlement", "tail3_delta_margin")
        if not all(csv_number(meta.get(field), field) < 0.0 for field in fields):
            raise ValueError("seed185 harm subtype signs changed")
        return "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
    raise ValueError("primary row has no frozen subtype")


def branch_magnitude(row: dict[str, Any], branch: str, validate_targets: bool = True) -> float:
    available = boolean(row[f"{branch}_available"], f"{branch}_available")
    if branch in ("temporal_mismatch", "predicate_mismatch"):
        condition = boolean(row[f"{branch}_condition_input"], f"{branch}_condition_input")
        stem = branch.split("_")[0]
        raw = row[f"raw_alpha_{stem}"]
        if not available:
            if raw is not None: raise ValueError(f"{branch}: unavailable branch has raw state")
            magnitude, active = 0.0, False
        else:
            magnitude, active = (softplus(number(raw, f"{branch} raw")) if condition else 0.0), condition
        if validate_targets:
            transformed = row[f"softplus_alpha_{stem}"]
            if available and abs(softplus(number(raw, f"{branch} raw")) - number(transformed, f"{branch} transformed")) > TOL:
                raise ValueError(f"{branch}: transform closure failed")
            if boolean(row[f"{branch}_active"], f"{branch}_active") != active:
                raise ValueError(f"{branch}: active closure failed")
        return magnitude
    if branch == "temporal_adapter":
        scale = number(row["temporal_adapter_final_penalty_scale"], "adapter scale")
        active = available and scale > 0.0
        if not available:
            if row["temporal_adapter_logit"] is not None or row["temporal_adapter_gate_probability"] is not None:
                raise ValueError("unavailable adapter has gate state")
            magnitude = 0.0
        else:
            logit = number(row["temporal_adapter_logit"], "adapter logit")
            probability = sigmoid(logit)
            if validate_targets and abs(probability - number(row["temporal_adapter_gate_probability"], "adapter probability", True)) > TOL:
                raise ValueError("adapter gate transform closure failed")
            magnitude = probability * scale if active else 0.0
        target = "temporal_adapter_effective_penalty_scale"
    else:
        scale = number(row["temporal_channel_gated_penalty_scale"], "channel scale")
        active = available and scale > 0.0
        if not available:
            if row["temporal_channel_logit"] is not None or row["temporal_channel_gate_probability"] is not None:
                raise ValueError("unavailable channel has gate state")
            magnitude = 0.0
        else:
            logit = number(row["temporal_channel_logit"], "channel logit")
            probability = sigmoid(logit)
            if validate_targets and abs(probability - number(row["temporal_channel_gate_probability"], "channel probability", True)) > TOL:
                raise ValueError("channel gate transform closure failed")
            magnitude = (probability
                         * (1.0 - number(row["preservation_entitlement_prob"], "preservation", True))
                         * scale if active else 0.0)
        target = "temporal_channel_effective_scale"
    if validate_targets:
        if boolean(row[f"{branch}_active"], f"{branch}_active") != active:
            raise ValueError(f"{branch}: active closure failed")
        if abs(magnitude - number(row[target], target)) > TOL:
            raise ValueError(f"{branch}: effective scale closure failed")
    return magnitude


def reconstruct(row: dict[str, Any], validate_targets: bool = True) -> dict[str, Any]:
    entitlement = (number(row["frame_prob"], "frame", True)
                   * number(row["predicate_coverage_prob"], "predicate", True)
                   * number(row["sufficiency_prob"], "sufficiency", True))
    alpha = softplus(number(row["raw_alpha"], "raw_alpha"))
    if validate_targets and abs(alpha - number(row["softplus_alpha"], "softplus_alpha")) > TOL:
        raise ValueError("decision-head transform closure failed")
    decision = (
        entitlement * number(row["negative_energy"], "negative_energy"),
        number(row["not_entitled_bias"], "not_entitled_bias") + alpha * (1.0 - entitlement),
        entitlement * number(row["positive_energy"], "positive_energy"),
    )
    deltas, target_error = {}, 0.0
    for branch in BRANCHES:
        magnitude = branch_magnitude(row, branch, validate_targets)
        delta = (-magnitude, magnitude, -magnitude)
        deltas[branch] = delta
        if validate_targets:
            for index, label in enumerate(("refute", "not_entitled", "support")):
                target_error = max(target_error, abs(delta[index] - number(row[f"{branch}_delta_{label}"], "branch delta")))
    total = tuple(sum(deltas[b][i] for b in BRANCHES) for i in range(3))
    if validate_targets:
        for index, label in enumerate(("refute", "not_entitled", "support")):
            target_error = max(target_error, abs(total[index] - number(row[f"total_final_delta_{label}"], "total delta")))
    final = tuple(decision[i] + total[i] for i in range(3))
    return {"entitlement": entitlement, "decision": decision, "branch_deltas": deltas,
            "branch_target_error": target_error, "final": final, "margin": final[2] - final[1],
            "prediction": argmax_label(final)}


def row_errors(row: dict[str, Any], rebuilt: dict[str, Any]) -> dict[str, Any]:
    native_decision = tuple(number(row[f"decision_head_{x}_logit"], "decision") for x in ("refute", "not_entitled", "support"))
    native_final = tuple(number(row[f"final_{x}_logit"], "final") for x in ("refute", "not_entitled", "support"))
    return {
        "entitlement": abs(rebuilt["entitlement"] - number(row["entitlement_prob_native"], "entitlement target")),
        "decision": max(abs(rebuilt["decision"][i] - native_decision[i]) for i in range(3)),
        "branch": rebuilt["branch_target_error"],
        "final": max(abs(rebuilt["final"][i] - native_final[i]) for i in range(3)),
        "margin": abs(rebuilt["margin"] - number(row["final_support_vs_not_entitled_margin"], "margin target")),
        "prediction": rebuilt["prediction"] == row["final_native_prediction"] == row["native_prediction"],
    }


def validate_sources(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    r1_path, b2_path = ns.stage196b2b3r1_analysis_json.resolve(), ns.stage196b2b2_analysis_json.resolve()
    if r1_path.name != R1_FILES[0] or b2_path.name != B2B2_FILES[0]:
        raise ValueError("analysis JSON basenames must be exact")
    exact_directory(r1_path.parent, R1_FILES, gates, "r1_exact_nine_file_closure")
    exact_directory(b2_path.parent, B2B2_FILES, gates, "b2b2_exact_nine_file_closure")
    r1, b2 = read_json(r1_path), read_json(b2_path)
    r1_required = {"decision": "STAGE196B2B3_FINAL_COMPOSER_RESIDUAL_REQUIRED",
                   "recommended_next_stage": "STAGE196B2B4_FINAL_COMPOSER_RESIDUAL_LOCALIZATION",
                   "blocking_reasons": []}
    r1_observed = {key: r1.get(key) for key in r1_required}
    gate(gates, "source", "", "r1_decision_closure", r1_required, r1_observed,
         r1_observed == r1_required, "R1 decision closure mismatch")
    r1_contract = read_csv(r1_path.parent / R1_FILES[-1])
    gate(gates, "source", "", "r1_55_of_55_contract_closure", {"rows": 55, "failed": 0},
         {"rows": len(r1_contract), "failed": sum(r.get("passed", "").lower() != "true" for r in r1_contract)},
         contract_closed(r1_contract, 55), "R1 contract closure mismatch")
    native, positive = r1.get("native_reconstruction", {}), r1.get("positive_control", {})
    native_observed = {key: native.get(key) for key in EXPECTED_R1_NATIVE}
    positive_observed = {key: positive.get(key) for key in EXPECTED_R1_POSITIVE}
    gate(gates, "source", "", "r1_native_reconstruction_exact", EXPECTED_R1_NATIVE,
         native_observed, native_observed == EXPECTED_R1_NATIVE, "R1 reconstruction authority changed")
    gate(gates, "source", "", "r1_positive_control_exact", EXPECTED_R1_POSITIVE,
         positive_observed, positive_observed == EXPECTED_R1_POSITIVE, "R1 positive control authority changed")
    expected_counts = {"component_swap": 3840, "paired_states": 43200,
                       "positive_control_directional": 86400, "row_swap_summary": 192,
                       "group_swap_summary": 48, "subtype_summary": 72}
    observed_counts = {key: r1.get("row_counts", {}).get(key) for key in expected_counts}
    gate(gates, "source", "", "r1_row_count_closure", expected_counts, observed_counts,
         observed_counts == expected_counts, "R1 row counts changed")
    r1_commit = r1.get("current_git_commit", r1.get("current_analyzer_git_commit"))
    gate(gates, "provenance", "", "r1_analyzer_source_commit_from_artifact", "40 lowercase hex in supplied R1 analysis",
         {"artifact_field_value": r1_commit}, isinstance(r1_commit, str) and re.fullmatch(r"[0-9a-f]{40}", r1_commit) is not None,
         "R1 analyzer source commit missing from supplied artifact")
    b2_required = {"decision": "STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT",
                   "recommended_next_stage": "STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE"}
    b2_observed = {key: b2.get(key) for key in b2_required}
    gate(gates, "source", "", "b2b2_decision_closure", b2_required, b2_observed,
         b2_observed == b2_required, "B2-B2 decision closure mismatch")
    b2_contract = read_csv(b2_path.parent / B2B2_FILES[-1])
    gate(gates, "source", "", "b2b2_155_of_155_contract_closure", 155,
         {"rows": len(b2_contract), "passed": sum(r.get("passed", "").lower() == "true" for r in b2_contract)},
         contract_closed(b2_contract, 155), "B2-B2 contract closure mismatch")
    static, epoch_rows = read_csv(b2_path.parent / B2B2_FILES[2]), read_csv(b2_path.parent / B2B2_FILES[3])
    gate(gates, "population", "", "b2b2_16_identity_closure", 16, len(static), len(static) == 16, "identity count mismatch")
    gate(gates, "population", "", "b2b2_320_epoch_row_closure", 320, len(epoch_rows), len(epoch_rows) == 320, "epoch row count mismatch")
    tail_values = sorted({integer(r["epoch"], "epoch") for r in epoch_rows if integer(r["epoch"], "epoch") in TAIL})
    gate(gates, "population", "", "b2b2_tail_epoch_closure", list(TAIL), tail_values, tail_values == list(TAIL), "tail epochs changed")
    primary: dict[tuple[int, str, str, int], dict[str, str]] = {}
    counts = {seed: Counter() for seed in PRIMARY_SEEDS}
    for row in static:
        seed, position = integer(row["seed"], "seed"), integer(row["dev_position"], "position")
        key = (seed, str(row["id"]), str(row["source_row_id"]), position)
        if seed not in PRIMARY_SEEDS or key in primary or row["transition_role"] not in ("recovery", "harm"):
            raise ValueError("invalid B2-B2 primary identity")
        for field in ("stable_row_id", "intervention_type", "path_class"):
            if not row.get(field): raise ValueError(f"missing frozen {field}")
        row["subtype"] = subtype(row)
        primary[key] = row
        counts[seed][row["transition_role"]] += 1
    normalized = {seed: dict(counts[seed]) for seed in PRIMARY_SEEDS}
    gate(gates, "population", "", "primary_population_exact", EXPECTED_PRIMARY, normalized,
         normalized == EXPECTED_PRIMARY and len(primary) == 16, "primary population changed")
    epoch_counts: Counter[tuple[int, str, str, int]] = Counter()
    for row in epoch_rows:
        key = (integer(row["seed"], "seed"), str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position"))
        if key not in primary or integer(row["epoch"], "epoch") not in EPOCHS:
            raise ValueError("B2-B2 epoch identity mismatch")
        for field in ("stable_row_id", "transition_role", "intervention_type", "path_class"):
            if field in row and row[field] != primary[key][field]: raise ValueError(f"frozen {field} changed")
        epoch_counts[key] += 1
    if len(epoch_counts) != 16 or set(epoch_counts.values()) != {20}:
        raise ValueError("B2-B2 identity/epoch closure failed")
    hashes = {str(path): sha256(path) for path in
              [*(r1_path.parent / name for name in R1_FILES), *(b2_path.parent / name for name in B2B2_FILES)]}
    return {"r1": r1, "b2": b2, "r1_path": r1_path, "b2_path": b2_path,
            "r1_commit": r1_commit, "primary": primary, "source_hashes": hashes,
            "legacy_rows": read_csv(r1_path.parent / R1_FILES[4])}


def validate_p0(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[Any, Any], dict[str, Any], dict[str, str]]:
    root = ns.stage196b2b3p0_run_root.resolve()
    observed_runs = sorted(p.name for p in root.iterdir() if p.is_dir())
    gate(gates, "p0", "", "p0_six_run_closure", sorted(RUNS), observed_runs,
         observed_runs == sorted(RUNS), "P0 run closure failed")
    gate(gates, "p0", "", "p0_runtime_commit_agreement", P0_COMMIT,
         ns.stage196b2b3p0_runtime_git_commit,
         ns.stage196b2b3p0_runtime_git_commit == P0_COMMIT, "P0 runtime commit mismatch")
    pairs: dict[Any, dict[str, dict[str, Any]]] = defaultdict(dict)
    composer_files = trajectory_files = composer_rows = trajectory_rows = prediction_equal = 0
    maxima = {key: 0.0 for key in ("entitlement", "decision", "branch", "final", "margin")}
    hashes: dict[str, str] = {}
    manifest_authorities = []
    for run in RUNS:
        seed, mode = int(run[4:7]), run[8:]
        composer_dir = root / run / "composer_inputs"
        trajectory_dir = root / run / "trajectory"
        composer_names = [f"stage196b2b3p0_epoch_composer_inputs_{epoch:03d}.jsonl" for epoch in EPOCHS]
        trajectory_names = [f"stage196b2p0_epoch_channels_{epoch:03d}.jsonl" for epoch in EPOCHS]
        manifest_name = "stage196b2b3p0_composer_input_manifest.json"
        exact_directory(composer_dir, (manifest_name, *composer_names), gates, f"{run}_composer_namespace")
        observed_trajectory = sorted(p.name for p in trajectory_dir.iterdir() if p.is_file())
        gate(gates, "p0", run, "trajectory_namespace", sorted(trajectory_names), observed_trajectory,
             observed_trajectory == sorted(trajectory_names), "trajectory namespace failed")
        manifest_path = composer_dir / manifest_name
        manifest = read_json(manifest_path)
        hashes[str(manifest_path)] = sha256(manifest_path)
        tolerance_fields = tuple(sorted(
            key for key in manifest
            if (key.startswith("maximum_") and key.endswith("_error")) or key.endswith("_tolerance")
        ))
        required_manifest_errors = {"maximum_decision_head_error", "maximum_final_logit_error", "maximum_margin_error"}
        manifest_ok = (manifest.get("completed") is True and manifest.get("prediction_equality_rate") == 1.0
                       and required_manifest_errors.issubset(tolerance_fields)
                       and all(number(manifest.get(field), field) <= TOL for field in tolerance_fields)
                       and manifest.get("current_git_commit") == P0_COMMIT and manifest.get("seed") == seed
                       and manifest.get("gradient_ownership_mode") == mode and manifest.get("sidecar_files") == composer_names)
        gate(gates, "p0", run, "manifest_and_tolerance_closure", True,
             {"completed": manifest.get("completed"), "commit": manifest.get("current_git_commit"),
              "seed": manifest.get("seed"), "mode": manifest.get("gradient_ownership_mode"),
              "tolerances": {field: manifest.get(field) for field in tolerance_fields}},
             manifest_ok, "manifest closure failed")
        authority = {key: value for key, value in manifest.items() if key.endswith("_file_sha256")}
        if not authority or any(re.fullmatch(r"[0-9a-f]{64}", str(v)) is None for v in authority.values()):
            raise ValueError(f"{run}: malformed manifest hash authority")
        manifest_authorities.append(authority)
        for epoch, composer_name, trajectory_name in zip(EPOCHS, composer_names, trajectory_names):
            composer_path, trajectory_path = composer_dir / composer_name, trajectory_dir / trajectory_name
            hashes[str(composer_path)] = sha256(composer_path)
            hashes[str(trajectory_path)] = sha256(trajectory_path)
            if manifest.get("sidecar_sha256", {}).get(composer_name) != hashes[str(composer_path)]:
                raise ValueError(f"{run}:{epoch}: composer sidecar hash mismatch")
            composer, trajectory = read_jsonl(composer_path), read_jsonl(trajectory_path)
            if len(composer) != 720 or len(trajectory) != 720 or any(set(row) != TRAJECTORY_FIELDS for row in trajectory):
                raise ValueError(f"{run}:{epoch}: sidecar row/schema closure failed")
            trajectory_index = {(str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position")): row for row in trajectory}
            if len(trajectory_index) != 720: raise ValueError("trajectory identity is not unique")
            schema, seen = set(composer[0]), set()
            for row in composer:
                if set(row) != schema: raise ValueError("composer schema drift")
                identity = (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position"))
                if identity in seen or identity not in trajectory_index: raise ValueError("strict composer identity failure")
                seen.add(identity)
                if (row.get("current_git_commit") != P0_COMMIT or integer(row.get("seed"), "seed") != seed
                        or integer(row.get("epoch"), "epoch") != epoch or row.get("gradient_ownership_mode") != mode):
                    raise ValueError("composer provenance mismatch")
                if trajectory_index[identity]["prediction"] != row["final_native_prediction"]:
                    raise ValueError("composer/trajectory prediction mismatch")
                rebuilt = reconstruct(row, True)
                errors = row_errors(row, rebuilt)
                for key in maxima: maxima[key] = max(maxima[key], float(errors[key]))
                prediction_equal += int(errors["prediction"])
                stored = dict(row); stored["_native"] = rebuilt
                pair_key = (seed, epoch, *identity)
                if mode in pairs[pair_key]: raise ValueError("duplicate treatment state")
                pairs[pair_key][mode] = stored
            composer_files += 1; trajectory_files += 1
            composer_rows += len(composer); trajectory_rows += len(trajectory)
    if len(manifest_authorities) != 6 or any(a != manifest_authorities[0] for a in manifest_authorities):
        raise ValueError("P0 manifest source hashes disagree")
    gate(gates, "p0", "", "p0_sidecar_closure", {"composer": 120, "trajectory": 120},
         {"composer": composer_files, "trajectory": trajectory_files}, composer_files == trajectory_files == 120,
         "P0 sidecar count failed")
    gate(gates, "p0", "", "p0_86400_composer_row_closure", {"composer": 86400, "trajectory": 86400},
         {"composer": composer_rows, "trajectory": trajectory_rows}, composer_rows == trajectory_rows == 86400,
         "P0 row count failed")
    pair_ok = len(pairs) == 43200 and all(set(arms) == set(MODES) for arms in pairs.values())
    gate(gates, "pairing", "", "strict_directional_pair_closure", 43200,
         {"pairs": len(pairs), "bad_pairs": sum(set(arms) != set(MODES) for arms in pairs.values())},
         pair_ok, "pair closure failed")
    reconstruction = {"row_count": composer_rows, "maximum_entitlement_error": maxima["entitlement"],
                      "maximum_decision_head_error": maxima["decision"],
                      "maximum_branch_delta_sum_error": maxima["branch"],
                      "maximum_final_logit_error": maxima["final"], "maximum_margin_error": maxima["margin"],
                      "prediction_equality_rate": prediction_equal / composer_rows, "tolerance": TOL,
                      "passed": max(maxima.values()) <= TOL and prediction_equal == composer_rows}
    gate(gates, "reconstruction", "", "native_reconstruction_pass", True, reconstruction,
         reconstruction["passed"], "native reconstruction failed")
    return pairs, reconstruction, hashes


def install_groups(recipient: dict[str, Any], donor: dict[str, Any], mask: int,
                   names: Sequence[str], field_map: dict[str, Sequence[str]]) -> dict[str, Any]:
    result = {key: value for key, value in recipient.items() if not key.startswith("_")}
    bits, members = mask_members(mask, names)
    for member in members:
        for field in field_map[member]:
            if field not in donor: raise ValueError(f"missing causal field {field}")
            result[field] = donor[field]
    # Derived targets stay recipient-valued and are ignored when validate_targets=False.
    result["_coalition_mask"] = bits
    return result


def primitive_state(recipient: dict[str, Any], donor: dict[str, Any], mask: int) -> dict[str, Any]:
    return install_groups(recipient, donor, mask, PRIMITIVES, PRIMITIVE_FIELDS)


def residual_state(recipient: dict[str, Any], donor: dict[str, Any], mask: int) -> dict[str, Any]:
    base = primitive_state(recipient, donor, 31)
    return install_groups(base, donor, mask, RESIDUAL_GROUPS, RESIDUAL_FIELDS)


def coalition_row(key: tuple[Any, ...], meta: dict[str, str], direction: str,
                  recipient_mode: str, donor_mode: str, mask: int, names: Sequence[str],
                  recipient: dict[str, Any], donor: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    seed, epoch, identity, source, position = key
    bits, members = mask_members(mask, names)
    rec, don = recipient["_native"], donor["_native"]
    return {
        "seed": seed, "epoch": epoch, "stable_row_id": meta["stable_row_id"], "id": identity,
        "source_row_id": source, "dev_position": position, "transition_role": meta["transition_role"],
        "intervention_type": meta["intervention_type"], "path_class": meta["path_class"],
        "subtype": meta["subtype"], "direction": direction, "recipient_mode": recipient_mode,
        "donor_mode": donor_mode, "coalition_mask": bits, "coalition_size": len(members),
        "coalition_members": members, "recipient_refute_logit": rec["final"][0],
        "recipient_not_entitled_logit": rec["final"][1], "recipient_support_logit": rec["final"][2],
        "recipient_margin": rec["margin"], "recipient_prediction": rec["prediction"],
        "donor_refute_logit": don["final"][0], "donor_not_entitled_logit": don["final"][1],
        "donor_support_logit": don["final"][2], "donor_margin": don["margin"],
        "donor_prediction": don["prediction"], "counterfactual_refute_logit": cf["final"][0],
        "counterfactual_not_entitled_logit": cf["final"][1], "counterfactual_support_logit": cf["final"][2],
        "counterfactual_margin": cf["margin"], "counterfactual_prediction": cf["prediction"],
        "counterfactual_minus_recipient_margin": cf["margin"] - rec["margin"],
        "donor_minus_recipient_margin": don["margin"] - rec["margin"],
        "prediction_changed": cf["prediction"] != rec["prediction"],
        "donor_prediction_reproduced": cf["prediction"] == don["prediction"],
        "recipient_prediction_preserved": cf["prediction"] == rec["prediction"],
    }


def enumerate_lattice(pairs: dict[Any, Any], primary: dict[Any, dict[str, str]],
                      lattice: str) -> list[dict[str, Any]]:
    names = PRIMITIVES if lattice == "primitive" else RESIDUAL_GROUPS
    output = []
    for key in sorted(pairs):
        seed, _, identity, source, position = key
        meta = primary.get((seed, identity, source, position))
        if meta is None: continue
        for direction in DIRECTIONS:
            recipient_mode, donor_mode = DIRECTION_MODES[direction]
            recipient, donor = pairs[key][recipient_mode], pairs[key][donor_mode]
            for mask in range(32):
                state = primitive_state(recipient, donor, mask) if lattice == "primitive" else residual_state(recipient, donor, mask)
                cf = reconstruct(state, False)
                output.append(coalition_row(key, meta, direction, recipient_mode, donor_mode,
                                            mask, names, recipient, donor, cf))
    return output


def mobius_rows(rows: list[dict[str, Any]], lattice: str, names: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["seed"], row["epoch"], row["stable_row_id"], row["direction"])
        buckets[key][int(row["coalition_mask"], 2)] = row
    output, maximum_error = [], 0.0
    metrics = ("refute", "not_entitled", "support", "margin")
    value_fields = {metric: f"counterfactual_{metric}_logit" for metric in metrics[:3]}
    value_fields["margin"] = "counterfactual_margin"
    extrema = {metric: {"maximum": None, "minimum": None} for metric in metrics}
    for key, values in sorted(buckets.items()):
        if set(values) != set(range(32)): raise ValueError(f"{lattice}: incomplete state lattice")
        interactions: dict[int, dict[str, float]] = {}
        for mask in range(32):
            interactions[mask] = {}
            subsets = [candidate for candidate in range(32) if candidate & mask == candidate]
            for metric in metrics:
                interactions[mask][metric] = math.fsum(
                    (-1.0 if ((mask.bit_count() - candidate.bit_count()) % 2) else 1.0)
                    * csv_number(values[candidate][value_fields[metric]], value_fields[metric])
                    for candidate in subsets
                )
        full_effect = {metric: csv_number(values[31][value_fields[metric]], "full")
                       - csv_number(values[0][value_fields[metric]], "empty") for metric in metrics}
        errors = {metric: abs(full_effect[metric] - math.fsum(interactions[mask][metric] for mask in range(1, 32)))
                  for metric in metrics}
        maximum_error = max(maximum_error, *errors.values())
        for mask in range(32):
            bits, members = mask_members(mask, names)
            row = {"seed": key[0], "epoch": key[1], "identity": key[2], "direction": key[3],
                   "lattice": lattice, "coalition_mask": bits, "coalition_size": len(members),
                   "coalition_members": members}
            row.update({f"{metric}_interaction": interactions[mask][metric] for metric in metrics})
            row.update({f"full_effect_{metric}": full_effect[metric] for metric in metrics})
            row.update({f"interaction_sum_error_{metric}": errors[metric] for metric in metrics})
            output.append(row)
            if mask:
                for metric in metrics:
                    descriptor = {"seed": key[0], "epoch": key[1], "identity": key[2],
                                  "direction": key[3], "coalition_mask": bits,
                                  "coalition_members": members, "value": interactions[mask][metric]}
                    if extrema[metric]["maximum"] is None or descriptor["value"] > extrema[metric]["maximum"]["value"]:
                        extrema[metric]["maximum"] = descriptor
                    if extrema[metric]["minimum"] is None or descriptor["value"] < extrema[metric]["minimum"]["value"]:
                        extrema[metric]["minimum"] = descriptor
    return output, {"row_count": len(output), "maximum_reconstruction_error": maximum_error,
                    "tolerance": TOL, "passed": maximum_error <= TOL, "largest_signed_terms": extrema}


def tail_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["seed"], row["stable_row_id"], row["direction"], row["coalition_mask"])].append(row)
    summaries = []
    for key, values in sorted(buckets.items()):
        values = sorted(values, key=lambda row: row["epoch"])
        if [row["epoch"] for row in values] != list(EPOCHS): raise ValueError("tail bucket lacks 20 epochs")
        tail = [row for row in values if row["epoch"] in TAIL]
        first = values[0]
        rec = [row["recipient_prediction"] for row in tail]
        don = [row["donor_prediction"] for row in tail]
        cf = [row["counterfactual_prediction"] for row in tail]
        summaries.append({"seed": key[0], "stable_row_id": key[1], "id": first["id"],
                          "source_row_id": first["source_row_id"], "dev_position": first["dev_position"],
                          "transition_role": first["transition_role"], "intervention_type": first["intervention_type"],
                          "path_class": first["path_class"], "subtype": first["subtype"], "direction": key[2],
                          "coalition_mask": key[3], "coalition_size": first["coalition_size"],
                          "coalition_members": first["coalition_members"], "tail_epochs": list(TAIL),
                          "recipient_tail_predictions": rec, "donor_tail_predictions": don,
                          "coalition_tail_predictions": cf, "recipient_tail_status": tail_status(rec),
                          "donor_tail_status": tail_status(don), "coalition_tail_status": tail_status(cf),
                          "donor_tail_reproduced": tail_status(cf) == tail_status(don),
                          "recipient_tail_preserved": tail_status(cf) == tail_status(rec),
                          "minimal_donor_reproducing_coalition": False})
    by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries: by_identity[(row["seed"], row["stable_row_id"], row["direction"])].append(row)
    minima = []
    for identity, values in sorted(by_identity.items()):
        reproducing = [row for row in values if row["donor_tail_reproduced"]]
        minimal_masks = [row["coalition_mask"] for row in reproducing
                         if not any(other["coalition_mask"] != row["coalition_mask"]
                                    and is_subset(other["coalition_mask"], row["coalition_mask"])
                                    for other in reproducing)]
        for row in values:
            row["minimal_donor_reproducing_coalition"] = row["coalition_mask"] in minimal_masks
            if row["minimal_donor_reproducing_coalition"]: minima.append(row)
    return summaries, minima


def legacy_agreement(primitive: list[dict[str, Any]], legacy: list[dict[str, str]]) -> dict[str, Any]:
    index = {(row["seed"], row["epoch"], row["stable_row_id"], row["direction"], row["coalition_mask"]): row for row in primitive}
    maximum = 0.0; prediction_equal = 0
    for row in legacy:
        mask = LEGACY_MASKS.get(row.get("variant"))
        if mask is None: raise ValueError("unexpected R1 legacy variant")
        key = (integer(row["seed"], "seed"), integer(row["epoch"], "epoch"), row["stable_row_id"], row["direction"], mask)
        current = index.get(key)
        if current is None: raise ValueError("R1 legacy row has no lattice match")
        for field in ("id", "source_row_id", "transition_role", "intervention_type", "path_class"):
            if str(row[field]) != str(current[field]): raise ValueError(f"R1/B2-B2 frozen {field} disagreement")
        if integer(row["dev_position"], "legacy position") != current["dev_position"]:
            raise ValueError("R1/B2-B2 frozen dev_position disagreement")
        for label in ("refute", "not_entitled", "support"):
            maximum = max(maximum, abs(csv_number(row[f"counterfactual_{label}_logit"], "legacy")
                                       - current[f"counterfactual_{label}_logit"]))
        maximum = max(maximum, abs(csv_number(row["counterfactual_margin"], "legacy") - current["counterfactual_margin"]))
        prediction_equal += int(row["counterfactual_prediction"] == current["counterfactual_prediction"])
    return {"row_count": len(legacy), "expected_row_count": 3840, "maximum_absolute_error": maximum,
            "prediction_equality_rate": prediction_equal / len(legacy),
            "passed": len(legacy) == 3840 and maximum == 0.0 and prediction_equal == len(legacy),
            "variant_masks": LEGACY_MASKS}


def seed_localization(tails: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    forward = [row for row in tails if row["direction"] == DIRECTIONS[0]]
    seed_results: dict[str, Any] = {}
    localization_rows: list[dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        masks: dict[str, Any] = {}
        for mask in (f"{value:05b}" for value in range(32)):
            selected = [row for row in forward if row["seed"] == seed and row["coalition_mask"] == mask]
            recovery = [row for row in selected if row["transition_role"] == "recovery"]
            harm = [row for row in selected if row["transition_role"] == "harm"]
            result = {"recovery_donor_reproduction_count": sum(row["donor_tail_reproduced"] for row in recovery),
                      "recovery_denominator": len(recovery),
                      "harm_recipient_preservation_count": sum(row["recipient_tail_preserved"] for row in harm),
                      "harm_denominator": len(harm)}
            result["recovery_all"] = result["recovery_donor_reproduction_count"] == result["recovery_denominator"] == EXPECTED_PRIMARY[seed]["recovery"]
            result["harm_all"] = result["harm_recipient_preservation_count"] == result["harm_denominator"] == EXPECTED_PRIMARY[seed]["harm"]
            result["passed"] = result["recovery_all"] and result["harm_all"]
            masks[mask] = result
            _, members = mask_members(int(mask, 2), PRIMITIVES)
            for criterion, numerator, denominator, passed in (
                ("all_recovery_donor_reproduced", result["recovery_donor_reproduction_count"], result["recovery_denominator"], result["recovery_all"]),
                ("all_harm_recipient_preserved", result["harm_recipient_preservation_count"], result["harm_denominator"], result["harm_all"]),
                ("exact_recovery_and_harm", result["recovery_donor_reproduction_count"] + result["harm_recipient_preservation_count"], result["recovery_denominator"] + result["harm_denominator"], result["passed"]),
            ):
                localization_rows.append({"record_type": "seed_fixed_primitive", "seed": seed, "stable_row_id": "",
                                          "direction": DIRECTIONS[0], "transition_role": "", "subtype": "",
                                          "coalition_mask": mask, "coalition_members": members, "criterion": criterion,
                                          "numerator": numerator, "denominator": denominator, "passed": passed,
                                          "details": result})
        passing = [mask for mask, result in masks.items() if result["passed"]]
        minimal = [mask for mask in passing if not any(other != mask and is_subset(other, mask) for other in passing)]
        seed_results[str(seed)] = {"coalitions": masks, "passing_coalitions": passing,
                                   "inclusion_minimal_passing_coalitions": minimal,
                                   "any_exact_fixed_coalition_passes": bool(passing)}
    cross_passing = [mask for mask in (f"{value:05b}" for value in range(32))
                     if all(seed_results[str(seed)]["coalitions"][mask]["passed"] for seed in PRIMARY_SEEDS)]
    cross_minimal = [mask for mask in cross_passing
                     if not any(other != mask and is_subset(other, mask) for other in cross_passing)]
    cross = {"passing_coalitions": cross_passing, "inclusion_minimal_passing_coalitions": cross_minimal,
             "same_fixed_coalition_passes_both_seeds": bool(cross_passing),
             "seed_agreement": seed_results["184"]["passing_coalitions"] == seed_results["185"]["passing_coalitions"]}
    return seed_results, cross, localization_rows


def residual_controls(primitive: list[dict[str, Any]], residual: list[dict[str, Any]]) -> dict[str, Any]:
    primitive_full = {(r["seed"], r["epoch"], r["stable_row_id"], r["direction"]): r for r in primitive if r["coalition_mask"] == "11111"}
    maximum_empty = maximum_donor_logit = maximum_donor_margin = 0.0
    empty_prediction = donor_prediction = 0; empty_total = donor_total = 0
    closing_masks = []
    for mask in (f"{value:05b}" for value in range(32)):
        selected = [row for row in residual if row["coalition_mask"] == mask]
        closes = True
        for row in selected:
            errors = [abs(row[f"counterfactual_{label}_logit"] - row[f"donor_{label}_logit"])
                      for label in ("refute", "not_entitled", "support")]
            closes = closes and max(errors) <= TOL and abs(row["counterfactual_margin"] - row["donor_margin"]) <= TOL and row["counterfactual_prediction"] == row["donor_prediction"]
        if len(selected) == 640 and closes: closing_masks.append(mask)
    for row in residual:
        key = (row["seed"], row["epoch"], row["stable_row_id"], row["direction"])
        if row["coalition_mask"] == "00000":
            target = primitive_full[key]
            errors = [abs(row[f"counterfactual_{label}_logit"] - target[f"counterfactual_{label}_logit"])
                      for label in ("refute", "not_entitled", "support")]
            maximum_empty = max(maximum_empty, *errors, abs(row["counterfactual_margin"] - target["counterfactual_margin"]))
            empty_prediction += int(row["counterfactual_prediction"] == target["counterfactual_prediction"]); empty_total += 1
        if row["coalition_mask"] == "11111":
            errors = [abs(row[f"counterfactual_{label}_logit"] - row[f"donor_{label}_logit"])
                      for label in ("refute", "not_entitled", "support")]
            maximum_donor_logit = max(maximum_donor_logit, *errors)
            maximum_donor_margin = max(maximum_donor_margin, abs(row["counterfactual_margin"] - row["donor_margin"]))
            donor_prediction += int(row["counterfactual_prediction"] == row["donor_prediction"]); donor_total += 1
    return {"empty": {"row_count": empty_total, "maximum_absolute_error": maximum_empty,
                       "prediction_equality_rate": empty_prediction / empty_total,
                       "passed": empty_total == 640 and maximum_empty <= TOL and empty_prediction == empty_total},
            "full": {"row_count": donor_total, "maximum_donor_logit_error": maximum_donor_logit,
                      "maximum_donor_margin_error": maximum_donor_margin,
                      "prediction_equality_rate": donor_prediction / donor_total,
                      "passed": donor_total == 640 and maximum_donor_logit <= TOL
                      and maximum_donor_margin <= TOL and donor_prediction == donor_total},
            "fixed_coalitions_closing_all_donor_outputs": closing_masks,
            "inclusion_minimal_closing_coalitions": [mask for mask in closing_masks
                if not any(other != mask and is_subset(other, mask) for other in closing_masks)]}


def residual_activity(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    index = {(r["seed"], r["epoch"], r["stable_row_id"], r["direction"], r["coalition_mask"]): r for r in rows}
    activity, localization = {}, []
    singleton_masks = ("10000", "01000", "00100", "00010", "00001")
    for group, mask in zip(RESIDUAL_GROUPS, singleton_masks):
        changes = []; categorical = 0
        for key, row in index.items():
            if key[-1] != mask: continue
            baseline = index[(*key[:-1], "00000")]
            delta = [row[f"counterfactual_{label}_logit"] - baseline[f"counterfactual_{label}_logit"]
                     for label in ("refute", "not_entitled", "support")]
            changes.extend(delta); categorical += int(row["counterfactual_prediction"] != baseline["counterfactual_prediction"])
        record = {"singleton_mask": mask, "row_count": len(changes) // 3,
                  "changes_final_logits": any(value != 0.0 for value in changes),
                  "structurally_inactive_under_observed_singleton": all(value == 0.0 for value in changes),
                  "numerically_zero": all(value == 0.0 for value in changes),
                  "maximum_signed_logit_change": max(changes), "minimum_signed_logit_change": min(changes),
                  "categorical_epoch_prediction_change_count": categorical}
        activity[group] = record
        localization.append({"record_type": "residual_group_activity", "seed": "", "stable_row_id": "",
                             "direction": "", "transition_role": "", "subtype": "", "coalition_mask": mask,
                             "coalition_members": [group], "criterion": "singleton_donor_substitution_activity",
                             "numerator": categorical, "denominator": record["row_count"],
                             "passed": record["changes_final_logits"], "details": record})
    tail_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["epoch"] in TAIL:
            tail_buckets[(row["seed"], row["stable_row_id"], row["direction"], row["coalition_mask"])].append(row)
    changed = total = 0; by_mask = {}
    for mask in (f"{value:05b}" for value in range(32)):
        count = denominator = 0
        for key, values in tail_buckets.items():
            if key[-1] != mask: continue
            baseline = tail_buckets[(*key[:-1], "00000")]
            if tail_status([r["counterfactual_prediction"] for r in values]) != tail_status([r["counterfactual_prediction"] for r in baseline]): count += 1
            denominator += 1
        by_mask[mask] = {"categorical_tail_status_change_count": count, "denominator": denominator}
        changed += count; total += denominator
    categorical = {"by_coalition": by_mask, "any_tail_status_change": changed > 0,
                   "total_changed_identity_coalitions": changed, "total_identity_coalitions": total}
    return activity, categorical, localization


def minimal_localization_rows(minima: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in minima:
        output.append({"record_type": "identity_minimal_primitive", "seed": row["seed"],
                       "stable_row_id": row["stable_row_id"], "direction": row["direction"],
                       "transition_role": row["transition_role"], "subtype": row["subtype"],
                       "coalition_mask": row["coalition_mask"], "coalition_members": row["coalition_members"],
                       "criterion": "donor_tail_status_reproduced_no_proper_subset",
                       "numerator": 1, "denominator": 1, "passed": True,
                       "details": {"recipient_tail_status": row["recipient_tail_status"],
                                   "donor_tail_status": row["donor_tail_status"],
                                   "coalition_tail_status": row["coalition_tail_status"]}})
    return output


def decide(seed_results: dict[str, Any], cross: dict[str, Any], minima: list[dict[str, Any]],
           controls: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if cross["inclusion_minimal_passing_coalitions"]:
        decision = "STAGE196B2B4_CROSS_SEED_PRIMITIVE_COALITION_LOCALIZED"
        next_stage = "STAGE196B2B5_PRIMITIVE_COALITION_INTERVENTION_DESIGN"
    elif all(seed_results[str(seed)]["any_exact_fixed_coalition_passes"] for seed in PRIMARY_SEEDS):
        decision = "STAGE196B2B4_SEED_SPECIFIC_PRIMITIVE_COALITIONS"
        next_stage = "STAGE196B2B5_SEED_SPECIFIC_MECHANISM_AUDIT"
    elif all(any(row["seed"] == seed and row["transition_role"] == "recovery"
                 and row["direction"] == DIRECTIONS[0] for row in minima) for seed in PRIMARY_SEEDS):
        decision = "STAGE196B2B4_ROW_SPECIFIC_PRIMITIVE_INTERACTION"
        next_stage = "STAGE196B2B5_ROW_SELECTOR_OBSERVABILITY"
    elif controls["inclusion_minimal_closing_coalitions"] == ["11111"]:
        decision = "STAGE196B2B4_DISTRIBUTED_FINAL_COMPOSER_RESIDUAL"
        next_stage = "STAGE196B2B5_NO_PROMOTION_RESIDUAL_STRUCTURE_AUDIT"
    else:
        raise ValueError("frozen decision rules are non-exhaustive for observed proper residual coalition")
    return decision, next_stage, {"ordered_rules_completed": True, "decision": decision}


def residual_classification(controls: dict[str, Any], activity: dict[str, Any], categorical: dict[str, Any]) -> dict[str, Any]:
    minimal = controls["inclusion_minimal_closing_coalitions"]
    active = [group for group, value in activity.items() if value["changes_final_logits"]]
    if len(minimal) == 1 and len([bit for bit in minimal[0] if bit == "1"]) == 1: kind = "single-group"
    elif minimal and any(mask != "11111" for mask in minimal): kind = "fixed proper coalition"
    elif minimal == ["11111"]: kind = "distributed across groups"
    else: kind = "row-specific"
    inert = bool(active) and not categorical["any_tail_status_change"]
    return {"classification": kind, "active_groups": active,
            "structurally_inactive_or_numerically_zero_groups": [g for g, v in activity.items() if v["numerically_zero"]],
            "numerically_nonzero_but_categorically_inert": inert}


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo, output = ns.repo_root.resolve(), ns.output_dir.resolve()
    gate(gates, "path", "", "repo_root", "existing directory", str(repo), repo.is_dir(), "invalid repo root")
    inputs = (ns.stage196b2b3r1_analysis_json.resolve(), ns.stage196b2b2_analysis_json.resolve(), ns.stage196b2b3p0_run_root.resolve())
    separate = not output.exists() and all(output != path and output not in path.parents and path not in output.parents for path in inputs)
    gate(gates, "path", "", "fresh_separate_output_directory", True,
         {"output": str(output), "exists": output.exists()}, separate, "output exists or overlaps source")
    gate(gates, "provenance", "", "current_git_commit_format", "40 lowercase hex", ns.current_git_commit,
         re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit or "") is not None, "invalid current commit")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    gate(gates, "provenance", "", "current_git_commit_equals_head", ns.current_git_commit, head,
         head == ns.current_git_commit, "current commit differs from HEAD")
    sources = validate_sources(ns, gates)
    pairs, native, p0_hashes = validate_p0(ns, gates)
    for (seed, identity, source, position), meta in sources["primary"].items():
        for epoch in EPOCHS:
            pair = pairs.get((seed, epoch, identity, source, position))
            if pair is None: raise ValueError("primary identity missing from P0")
            for arm in pair.values():
                if str(arm["stable_row_id"]) != str(meta["stable_row_id"]): raise ValueError("strict stable_row_id mismatch")
    gate(gates, "population", "", "primary_16_identity_closure", 16,
         len(sources["primary"]), len(sources["primary"]) == 16, "primary identity closure failed")
    directional_state_count = len(sources["primary"]) * len(EPOCHS) * len(DIRECTIONS)
    gate(gates, "population", "", "primary_640_directional_state_closure", 640,
         directional_state_count, directional_state_count == 640, "primary directional state closure failed")
    primitive = enumerate_lattice(pairs, sources["primary"], "primitive")
    gate(gates, "primitive", "", "primitive_exact_32_coalitions", [f"{m:05b}" for m in range(32)],
         sorted({row["coalition_mask"] for row in primitive}),
         sorted({row["coalition_mask"] for row in primitive}) == [f"{m:05b}" for m in range(32)], "primitive lattice incomplete")
    gate(gates, "primitive", "", "primitive_20480_row_closure", 20480, len(primitive), len(primitive) == 20480, "primitive row count failed")
    legacy = legacy_agreement(primitive, sources["legacy_rows"])
    gate(gates, "primitive", "", "legacy_variant_agreement", True, legacy, legacy["passed"], "legacy variants disagree")
    primitive_mobius, primitive_mobius_audit = mobius_rows(primitive, "primitive", PRIMITIVES)
    gate(gates, "primitive", "", "primitive_mobius_20480_and_exact", {"rows": 20480, "maximum_error": TOL},
         primitive_mobius_audit, len(primitive_mobius) == 20480 and primitive_mobius_audit["passed"], "primitive Mobius closure failed")
    tails, minima = tail_summaries(primitive)
    gate(gates, "primitive", "", "primitive_tail_summary_1024_row_closure", 1024, len(tails), len(tails) == 1024, "tail summary count failed")
    seed_results, cross, localization = seed_localization(tails)
    localization = minimal_localization_rows(minima) + localization
    residual = enumerate_lattice(pairs, sources["primary"], "residual")
    gate(gates, "residual", "", "residual_exact_32_coalitions", [f"{m:05b}" for m in range(32)],
         sorted({row["coalition_mask"] for row in residual}),
         sorted({row["coalition_mask"] for row in residual}) == [f"{m:05b}" for m in range(32)], "residual lattice incomplete")
    gate(gates, "residual", "", "residual_20480_row_closure", 20480, len(residual), len(residual) == 20480, "residual row count failed")
    controls = residual_controls(primitive, residual)
    gate(gates, "residual", "", "empty_residual_control", True, controls["empty"], controls["empty"]["passed"], "empty residual failed")
    gate(gates, "residual", "", "full_residual_donor_control", True, controls["full"], controls["full"]["passed"], "full residual failed")
    residual_mobius, residual_mobius_audit = mobius_rows(residual, "residual", RESIDUAL_GROUPS)
    gate(gates, "residual", "", "residual_mobius_20480_and_exact", {"rows": 20480, "maximum_error": TOL},
         residual_mobius_audit, len(residual_mobius) == 20480 and residual_mobius_audit["passed"], "residual Mobius closure failed")
    activity, categorical, activity_rows = residual_activity(residual)
    localization.extend(activity_rows)
    residual_kind = residual_classification(controls, activity, categorical)
    decision, next_stage, evaluation = decide(seed_results, cross, minima, controls)
    gate(gates, "decision", "", "decision_evaluation_completion", True, evaluation, True, "")
    gate(gates, "output", "", "exact_nine_output_plan", list(OUTPUTS), list(OUTPUTS), True, "")
    source_paths = {"stage196b2b3r1_analysis_json": str(sources["r1_path"]),
                    "stage196b2b2_analysis_json": str(sources["b2_path"]),
                    "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve())}
    analysis = {
        "stage": STAGE, "decision": decision, "recommended_next_stage": next_stage, "blocking_reasons": [],
        "current_git_commit": ns.current_git_commit,
        "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
        "r1_analyzer_source_git_commit_from_supplied_artifact": sources["r1_commit"],
        "source_paths": source_paths, "source_hashes": {**sources["source_hashes"], **p0_hashes},
        "native_reconstruction": native,
        "source_closure": {"r1_exact_files": list(R1_FILES), "b2b2_exact_files": list(B2B2_FILES),
                           "p0_runs": list(RUNS), "r1_decision": sources["r1"]["decision"],
                           "b2b2_decision": sources["b2"]["decision"]},
        "primary_population": {"identity_count": 16, "identity_epoch_count": 320,
                               "directional_state_count": 640, "tail_epochs": list(TAIL),
                               "seed_counts": EXPECTED_PRIMARY,
                               "seed_roles": {"183": "contrast-only excluded from decisions", "184": "primary", "185": "primary"}},
        "primitive_lattice": {"canonical_order": list(PRIMITIVES), "coalition_count": 32,
                              "row_count": len(primitive), "empty_mask": "00000", "full_mask": "11111",
                              "positive_and_negative_energy_separate": True, "legacy_agreement": legacy},
        "primitive_mobius": primitive_mobius_audit,
        "primitive_minimal_coalitions": [{"seed": r["seed"], "stable_row_id": r["stable_row_id"],
                                           "direction": r["direction"], "transition_role": r["transition_role"],
                                           "subtype": r["subtype"], "coalition_mask": r["coalition_mask"],
                                           "coalition_members": r["coalition_members"]} for r in minima],
        "primitive_seed_level_results": seed_results, "primitive_cross_seed_results": cross,
        "residual_lattice": {"canonical_order": list(RESIDUAL_GROUPS), "raw_causal_fields": RESIDUAL_FIELDS,
                             "coalition_count": 32, "row_count": len(residual), "baseline": "all donor primitives plus recipient residual groups",
                             "full": "all donor primitives plus donor residual groups"},
        "residual_positive_controls": controls, "residual_mobius": residual_mobius_audit,
        "residual_group_activity": activity, "residual_categorical_effect": categorical,
        "residual_localization": residual_kind, "decision_rule_evaluation": evaluation,
        "subtype_analysis": [
            {"seed": key[0], "transition_role": key[1], "subtype": key[2],
             "coalition_mask": key[3], "minimal_coalition_count": count}
            for key, count in sorted(Counter(
                (r["seed"], r["transition_role"], r["subtype"], r["coalition_mask"])
                for r in minima
            ).items())
        ],
        "authorized_interpretation": "Within the frozen-Mamba, frozen-composer six-run population, exact raw-input Boolean-lattice interventions descriptively localize primitive and final-composer residual structure under frozen tail and seed rules.",
        "remaining_uncertainty": ["inference-only within-model intervention probe", "observed clean-dev identities and epochs only",
                                  "continuous residual closure does not establish categorical selectivity"],
        "prohibited_claims": list(PROHIBITED), "training_performed": False, "model_loaded": False,
        "checkpoint_loaded": False, "classifier_fitted": False, "threshold_search_performed": False,
        "promotion_authorized": False, "artifact_only": True,
    }
    return analysis, {"primitive": primitive, "primitive_mobius": primitive_mobius, "tails": tails,
                      "residual": residual, "residual_mobius": residual_mobius, "localization": localization}


REPORT_SECTIONS = (
    "Executive decision", "Authorized interpretation", "R1 source result", "Source closure",
    "Primitive lattice definition", "Primitive coalition controls", "Primitive minimal coalitions",
    "Seed184 recovery/harm localization", "Seed185 recovery/harm localization",
    "Cross-seed primitive consistency", "Primitive M\u00f6bius interactions", "Residual group definition",
    "Residual empty-coalition control", "Residual full-donor control", "Residual M\u00f6bius interactions",
    "Categorical versus continuous residual", "Subtype analysis", "Remaining uncertainty",
    "Prohibited claims", "Recommended next stage",
)


def report(analysis: dict[str, Any]) -> str:
    bodies = (
        f"`{analysis['decision']}`", analysis["authorized_interpretation"],
        json.dumps({"decision": analysis["source_closure"].get("r1_decision"),
                    "analyzer_source_commit": analysis.get("r1_analyzer_source_git_commit_from_supplied_artifact")}, sort_keys=True),
        json.dumps(analysis["source_closure"], sort_keys=True), json.dumps(analysis["primitive_lattice"], sort_keys=True),
        json.dumps(analysis["primitive_lattice"].get("legacy_agreement"), sort_keys=True),
        json.dumps(analysis["primitive_minimal_coalitions"], sort_keys=True),
        json.dumps(analysis["primitive_seed_level_results"].get("184"), sort_keys=True),
        json.dumps(analysis["primitive_seed_level_results"].get("185"), sort_keys=True),
        json.dumps(analysis["primitive_cross_seed_results"], sort_keys=True),
        json.dumps(analysis["primitive_mobius"], sort_keys=True), json.dumps(analysis["residual_lattice"], sort_keys=True),
        json.dumps(analysis["residual_positive_controls"].get("empty"), sort_keys=True),
        json.dumps(analysis["residual_positive_controls"].get("full"), sort_keys=True),
        json.dumps(analysis["residual_mobius"], sort_keys=True),
        json.dumps({"group_activity": analysis["residual_group_activity"],
                    "categorical_effect": analysis["residual_categorical_effect"],
                    "classification": analysis["residual_localization"]}, sort_keys=True),
        json.dumps(analysis["subtype_analysis"], sort_keys=True),
        "\n".join(f"- {item}" for item in analysis["remaining_uncertainty"]),
        "\n".join(f"- {item}" for item in analysis["prohibited_claims"]),
        f"`{analysis['recommended_next_stage']}`\n\nNo training or promotion is authorized.",
    )
    return f"# {STAGE}: Exact Primitive-Coalition and Final-Composer Residual Localization\n\n" + "\n\n".join(
        f"## {heading}\n\n{body}" for heading, body in zip(REPORT_SECTIONS, bodies)
    ) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)): return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is True: return "true"
    if value is False: return "false"
    if value is None: return ""
    return value


def render_csv(header: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(header), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(header): raise ValueError(f"generated CSV schema mismatch: {sorted(set(row) ^ set(header))}")
        writer.writerow({key: csv_value(row[key]) for key in header})
    return buffer.getvalue()


def render_contract(rows: Iterable[dict[str, Any]]) -> str:
    """Render required/observed as valid JSON while preserving typed gate values."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(CONTRACT_H), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(CONTRACT_H): raise ValueError("generated contract schema mismatch")
        rendered = dict(row)
        rendered["required"] = json.dumps(row["required"], sort_keys=True, separators=(",", ":"))
        rendered["observed"] = json.dumps(row["observed"], sort_keys=True, separators=(",", ":"))
        rendered["passed"] = csv_value(row["passed"])
        writer.writerow(rendered)
    return buffer.getvalue()


def blocked_analysis(ns: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {"stage": STAGE, "decision": "STAGE196B2B4_BLOCKED_CONTRACT_FAILURE",
            "recommended_next_stage": "STAGE196B2B4_REPAIR",
            "blocking_reasons": [f"{type(error).__name__}: {error}"],
            "current_git_commit": ns.current_git_commit,
            "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
            "source_paths": {"stage196b2b3r1_analysis_json": str(ns.stage196b2b3r1_analysis_json.resolve()),
                             "stage196b2b2_analysis_json": str(ns.stage196b2b2_analysis_json.resolve()),
                             "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve())},
            "source_hashes": {}, "native_reconstruction": {"passed": False}, "source_closure": {},
            "primary_population": {}, "primitive_lattice": {}, "primitive_mobius": {},
            "primitive_minimal_coalitions": [], "primitive_seed_level_results": {},
            "primitive_cross_seed_results": {}, "residual_lattice": {}, "residual_positive_controls": {},
            "residual_mobius": {}, "residual_group_activity": {}, "residual_categorical_effect": {},
            "residual_localization": {}, "decision_rule_evaluation": {"ordered_rules_completed": False},
            "subtype_analysis": {}, "authorized_interpretation": "No scientific interpretation is authorized because a contract failed.",
            "remaining_uncertainty": ["Repair the failed artifact or reconstruction contract."],
            "prohibited_claims": list(PROHIBITED), "training_performed": False, "model_loaded": False,
            "checkpoint_loaded": False, "classifier_fitted": False, "threshold_search_performed": False,
            "promotion_authorized": False, "artifact_only": True}


def render(analysis: dict[str, Any], tables: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, str]:
    return {OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n", OUTPUTS[1]: report(analysis),
            OUTPUTS[2]: render_csv(COALITION_H, tables["primitive"]),
            OUTPUTS[3]: render_csv(MOBIUS_H, tables["primitive_mobius"]),
            OUTPUTS[4]: render_csv(TAIL_H, tables["tails"]),
            OUTPUTS[5]: render_csv(COALITION_H, tables["residual"]),
            OUTPUTS[6]: render_csv(MOBIUS_H, tables["residual_mobius"]),
            OUTPUTS[7]: render_csv(LOCALIZATION_H, tables["localization"]),
            OUTPUTS[8]: render_contract(gates)}


def atomic_write_outputs(output: Path, payloads: dict[str, str]) -> None:
    if set(payloads) != set(OUTPUTS) or output.exists(): raise RuntimeError("refusing non-nine-file or overwrite output")
    output.mkdir(parents=True, exist_ok=False)
    temporary = []
    try:
        for name in OUTPUTS:
            final = output / name
            temp = output / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
            with temp.open("x", encoding="utf-8", newline="") as handle:
                handle.write(payloads[name]); handle.flush(); os.fsync(handle.fileno())
            temporary.append((temp, final))
        for temp, final in temporary: os.replace(temp, final)
        if sorted(path.name for path in output.iterdir()) != sorted(OUTPUTS):
            raise RuntimeError("written output closure is not exactly nine files")
    finally:
        for temp, _ in temporary:
            if temp.exists(): temp.unlink()


def main() -> int:
    ns = parse_args(); gates: list[dict[str, Any]] = []
    empty = {"primitive": [], "primitive_mobius": [], "tails": [], "residual": [],
             "residual_mobius": [], "localization": []}
    try:
        analysis, tables = analyze(ns, gates)
    except Exception as error:
        analysis, tables = blocked_analysis(ns, error), empty
        if not any(not row["passed"] for row in gates):
            gate(gates, "analysis", "", "unhandled_contract_failure", True,
                 {"error_type": type(error).__name__, "message": str(error)}, False, str(error), fatal=False)
        gate(gates, "output", "", "exact_nine_output_plan", list(OUTPUTS), list(OUTPUTS), True, "", fatal=False)
    payloads = render(analysis, tables, gates)
    atomic_write_outputs(ns.output_dir.resolve(), payloads)
    return 3 if analysis["decision"] == "STAGE196B2B4_BLOCKED_CONTRACT_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
