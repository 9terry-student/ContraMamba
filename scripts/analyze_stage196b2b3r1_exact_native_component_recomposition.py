#!/usr/bin/env python3
"""Stage196-B2-B3-R1: artifact-only exact native component recomposition.

The program reads only explicitly named authorities, never imports the model, and
never trains or loads a checkpoint.  Scientific component rows are constructed
only after independent native reconstruction and the full donor control pass.
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


STAGE = "Stage196-B2-B3-R1"
LABEL = "Exact Native Component Recomposition"
P0_COMMIT = "fa16787efa84bb15d832b6d9382fafd77016c4e2"
TOL = 1e-6
SEEDS = (183, 184, 185)
PRIMARY_SEEDS = (184, 185)
EPOCHS = tuple(range(1, 21))
TAIL = (18, 19, 20)
MODES = ("joint", "frame_local_only")
RUNS = tuple(f"seed{s}_{m}" for s in SEEDS for m in MODES)
MARGIN_AUTHORITY = "support_logit - not_entitled_logit"
VARIANTS = (
    "FRAME_ONLY", "PREDICATE_ONLY", "SUFFICIENCY_ONLY",
    "ENTITLEMENT_PRIMITIVES", "POLARITY_ONLY", "ENTITLEMENT_PLUS_POLARITY",
)
PLANNED_VARIANTS = (*VARIANTS, "FULL_COMPOSER_INPUT_POSITIVE_CONTROL")
DIRECTIONS = (
    "JOINT_RECIPIENT_FRAME_LOCAL_ONLY_DONOR",
    "FRAME_LOCAL_ONLY_RECIPIENT_JOINT_DONOR",
)
DIRECTION_MODES = {
    DIRECTIONS[0]: ("joint", "frame_local_only"),
    DIRECTIONS[1]: ("frame_local_only", "joint"),
}
OUTPUTS = (
    "stage196b2b3r1_analysis.json", "stage196b2b3r1_report.md",
    "stage196b2b3r1_composer_graph.csv", "stage196b2b3r1_native_reconstruction.csv",
    "stage196b2b3r1_component_swap_rows.csv", "stage196b2b3r1_row_swap_summary.csv",
    "stage196b2b3r1_group_swap_summary.csv", "stage196b2b3r1_subtype_summary.csv",
    "stage196b2b3r1_contract.csv",
)
B2B3_FILES = (
    "stage196b2b3_analysis.json", "stage196b2b3_report.md",
    "stage196b2b3_composer_graph.csv", "stage196b2b3_native_reconstruction.csv",
    "stage196b2b3_component_swap_rows.csv", "stage196b2b3_row_swap_summary.csv",
    "stage196b2b3_group_swap_summary.csv", "stage196b2b3_subtype_summary.csv",
    "stage196b2b3_contract.csv",
)
B2B2_FILES = (
    "stage196b2b2_analysis.json", "stage196b2b2_report.md",
    "stage196b2b2_row_path_summary.csv", "stage196b2b2_epoch_paired_paths.csv",
    "stage196b2b2_group_path_summary.csv", "stage196b2b2_event_order_summary.csv",
    "stage196b2b2_intervention_type_paths.csv", "stage196b2b2_contrast_summary.csv",
    "stage196b2b2_contract.csv",
)
TRAJECTORY_FIELDS = {
    "id", "source_row_id", "dev_position", "gold_label", "prediction",
    "intervention_type", "frame_probability", "predicate_coverage_probability",
    "sufficiency_probability", "polarity_support_margin", "entitlement_probability",
    "support_probability", "not_entitled_probability", "support_logit",
    "not_entitled_logit", "epoch", "training_seed", "frame_downstream_gradient_mode",
}
IDENTITY = ("id", "source_row_id", "dev_position")
PRIMARY_META = (
    "seed", "stable_row_id", "id", "source_row_id", "dev_position",
    "transition_role", "intervention_type", "path_class",
)
EXPECTED_PRIMARY = {184: {"recovery": 5, "harm": 6}, 185: {"recovery": 2, "harm": 3}}
EXPECTED_PATHS = {
    184: {"recovery": {"MULTI_CHANNEL_CONFLICT": 5}, "harm": {
        "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN": 3, "FRAME_ENTITLEMENT_LOSS": 2,
        "COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY": 1}},
    185: {"recovery": {"FRAME_ENTITLEMENT_GAIN": 2},
          "harm": {"MULTI_CHANNEL_CONFLICT": 3}},
}
DECISIONS = (
    "STAGE196B2B3_ENTITLEMENT_COMPONENT_DOMINANT",
    "STAGE196B2B3_ENTITLEMENT_POLARITY_DISJUNCTIVE_EFFECT",
    "STAGE196B2B3_POLARITY_OVERRIDE_COMPONENT_CONFIRMED",
    "STAGE196B2B3_FINAL_COMPOSER_RESIDUAL_REQUIRED",
    "STAGE196B2B3_SEED_SPECIFIC_COMPONENT_EFFECT",
)
NEXT = {
    DECISIONS[0]: "STAGE196B2B4_ENTITLEMENT_PATH_PRESERVATION_DESIGN",
    DECISIONS[1]: "STAGE196B2B4_ENTITLEMENT_GAIN_WITH_POLARITY_PRESERVATION_DESIGN",
    DECISIONS[2]: "STAGE196B2B4_POLARITY_PRESERVATION_MICROINTERVENTION_DESIGN",
    DECISIONS[3]: "STAGE196B2B4_FINAL_COMPOSER_RESIDUAL_LOCALIZATION",
    DECISIONS[4]: "STAGE196B2B4_NO_PROMOTION_COMPONENT_EFFECT_REPLICATION",
}
POSITIVE_FAILED = "STAGE196B2B3R1_POSITIVE_CONTROL_FAILED"
POSITIVE_REPAIR = "STAGE196B2B3R1_REPAIR_RECOMPOSITION"
INCOMPLETE = "STAGE196B2B3R1_ANALYSIS_INCOMPLETE"
INCOMPLETE_NEXT = "STAGE196B2B3R1_REPAIR_ANALYSIS_INPUTS"
PROHIBITED = (
    "formal causal mediation", "architectural sufficiency beyond the frozen composer",
    "external or OOD validity", "unfrozen-Mamba validity", "training improvement",
    "promotion", "a new decision mechanism",
    "polarity causality from polarity_support_margin",
    "component causality when the full-composer positive control fails",
)

CONTRACT_H = ("scope", "run", "gate", "required", "observed", "passed", "blocking_reason")
NATIVE_H = (
    "run", "seed", "gradient_mode", "epoch", "row_count",
    "maximum_entitlement_error", "maximum_decision_head_error",
    "maximum_branch_delta_sum_error", "maximum_final_logit_error",
    "maximum_margin_error", "prediction_equality_rate",
)
SWAP_H = (
    "seed", "epoch", "stable_row_id", "id", "source_row_id", "dev_position",
    "transition_role", "intervention_type", "path_class", "direction", "variant",
    "recipient_mode", "donor_mode", "recipient_refute_logit",
    "recipient_not_entitled_logit", "recipient_support_logit", "recipient_margin",
    "recipient_prediction", "donor_refute_logit", "donor_not_entitled_logit",
    "donor_support_logit", "donor_margin", "donor_prediction",
    "counterfactual_refute_logit", "counterfactual_not_entitled_logit",
    "counterfactual_support_logit", "counterfactual_margin", "counterfactual_prediction",
    "counterfactual_minus_recipient_refute", "counterfactual_minus_recipient_not_entitled",
    "counterfactual_minus_recipient_support", "counterfactual_minus_recipient_margin",
    "donor_minus_recipient_margin", "counterfactual_toward_donor_margin",
    "signed_margin_closure_fraction", "recipient_prediction_changed",
    "donor_prediction_reproduced", "recipient_prediction_preserved",
)
ROW_H = (
    "seed", "stable_row_id", "id", "source_row_id", "dev_position", "transition_role",
    "intervention_type", "path_class", "subtype", "direction", "variant", "tail_epochs",
    "recipient_tail_predictions", "donor_tail_predictions", "counterfactual_tail_predictions",
    "recipient_tail_status", "donor_tail_status", "counterfactual_tail_status",
    "donor_tail_reproduced", "recipient_tail_preserved", "mean_tail_margin_shift",
)
GROUP_H = (
    "seed", "transition_role", "direction", "variant", "identity_count",
    "donor_tail_reproduced_count", "donor_tail_reproduced_rate",
    "recipient_tail_preserved_count", "recipient_tail_preserved_rate",
    "mean_tail_margin_shift", "path_class_counts", "intervention_type_counts",
)
SUBTYPE_H = (
    "subtype", "seed", "transition_role", "path_class", "direction", "variant",
    "identity_count", "donor_tail_reproduced_count", "donor_tail_reproduced_rate",
    "recipient_tail_preserved_count", "recipient_tail_preserved_rate",
)
GRAPH_EXTRA = (
    "exported_b2b3p0_field", "recomposition_field_role", "swapped_by_variants",
    "downstream_dependency", "exact_formula_implemented", "positive_control_inclusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, typ in (
        ("repo-root", Path), ("stage196b2b3-analysis-json", Path),
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
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL row")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{path}:{number}: object required")
            rows.append(value)
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
        raise ValueError(f"{name}: finite numeric value required")
    result = float(value)
    if probability and not 0.0 <= result <= 1.0:
        raise ValueError(f"{name}: probability outside [0,1]")
    return result


def csv_number(value: Any, name: str) -> float:
    """Parse one finite numeric CSV cell without weakening JSON typing."""
    original = value
    if type(value) is bool or value is None:
        raise ValueError(f"{name}: finite CSV numeric value required; observed={original!r}")
    if isinstance(value, str):
        value = value.strip()
        if not value or re.fullmatch(
            r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
        ) is None:
            raise ValueError(f"{name}: finite CSV numeric value required; observed={original!r}")
    elif type(value) not in (int, float):
        raise ValueError(f"{name}: finite CSV numeric value required; observed={original!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name}: finite CSV numeric value required; observed={original!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{name}: finite CSV numeric value required; observed={original!r}")
    return result


def integer(value: Any, name: str) -> int:
    if isinstance(value, str):
        if not re.fullmatch(r"-?[0-9]+", value):
            raise ValueError(f"{name}: integer required")
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
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def argmax_label(logits: Sequence[float]) -> str:
    return ("REFUTE", "NOT_ENTITLED", "SUPPORT")[max(range(3), key=lambda i: logits[i])]


def tail_status(values: Sequence[str]) -> str:
    values = tuple(values)
    if values == ("SUPPORT",) * 3:
        return "STABLE_SUPPORT"
    if values == ("NOT_ENTITLED",) * 3:
        return "PERSISTENT_NOT_ENTITLED"
    if values == ("REFUTE",) * 3:
        return "PERSISTENT_REFUTE"
    return "UNSTABLE"


def gate(rows: list[dict[str, Any]], scope: str, run: str, name: str,
         required: Any, observed: Any, passed: bool, reason: str) -> None:
    rows.append({"scope": scope, "run": run, "gate": name, "required": required,
                 "observed": observed, "passed": bool(passed),
                 "blocking_reason": "" if passed else reason})
    if not passed:
        raise ValueError(f"{name}: {reason}")


def exact_directory(path: Path, names: Sequence[str], gates: list[dict[str, Any]], name: str) -> None:
    observed = sorted(p.name for p in path.iterdir() if p.is_file())
    required = sorted(names)
    gate(gates, "source", "", name, required,
         {"files": observed, "missing": sorted(set(required) - set(observed)),
          "unexpected": sorted(set(observed) - set(required))},
         observed == required, "exact file closure failed")


def contract_closed(rows: list[dict[str, str]], expected: int) -> bool:
    return len(rows) == expected and all(
        str(r.get("passed", "")).strip().lower() == "true"
        and not str(r.get("blocking_reason", "")).strip() for r in rows
    )


def validate_authorities(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> dict[str, Any]:
    b3_path = ns.stage196b2b3_analysis_json.resolve()
    b2_path = ns.stage196b2b2_analysis_json.resolve()
    if b3_path.name != B2B3_FILES[0] or b2_path.name != B2B2_FILES[0]:
        raise ValueError("authoritative analysis basenames are not exact")
    exact_directory(b3_path.parent, B2B3_FILES, gates, "b2b3_exact_nine_file_closure")
    exact_directory(b2_path.parent, B2B2_FILES, gates, "b2b2_exact_nine_file_closure")
    b3, b2 = read_json(b3_path), read_json(b2_path)
    expected_b3 = {
        "decision": "STAGE196B2B3_ADDITIONAL_COMPOSER_OBSERVABILITY_REQUIRED",
        "recommended_next_stage": "STAGE196B2B3P0_EPOCH_COMPOSER_INPUT_OBSERVABILITY_DESIGN",
        "blocking_reasons": [],
    }
    observed_b3 = {key: b3.get(key) for key in expected_b3}
    gate(gates, "source", "", "b2b3_decision_closure", expected_b3, observed_b3,
         observed_b3 == expected_b3, "B2-B3 decision closure mismatch")
    b3_contract = read_csv(b3_path.parent / B2B3_FILES[-1])
    gate(gates, "source", "", "b2b3_193_of_193_contract_closure", 193,
         {"rows": len(b3_contract), "passed": sum(str(r.get("passed", "")).lower() == "true" for r in b3_contract)},
         contract_closed(b3_contract, 193), "B2-B3 contract closure mismatch")
    margins = [b3.get(k) for k in (
        "normalized_margin_source", "normalized_support_vs_not_entitled_margin_source") if b3.get(k) is not None]
    gate(gates, "source", "", "normalized_margin_authority", MARGIN_AUTHORITY, margins,
         bool(margins) and all(v == MARGIN_AUTHORITY for v in margins), "margin authority missing or conflicting")
    graph_audit = b3.get("composer_graph_audit", {})
    graph_identified = graph_audit.get("source_graph_identified") is True or bool(graph_audit.get("primitive_channel_outputs"))
    gate(gates, "source", "", "b2b3_source_graph_identified", True, graph_audit,
         graph_identified, "source graph was not identified")
    source_variants = b3.get("swap_variants")
    source_directions = b3.get("swap_directions")
    gate(gates, "source", "", "frozen_seven_variant_closure", list(PLANNED_VARIANTS), source_variants,
         source_variants == list(PLANNED_VARIANTS), "variant closure mismatch")
    gate(gates, "source", "", "frozen_two_direction_closure", list(DIRECTIONS), source_directions,
         source_directions == list(DIRECTIONS), "direction closure mismatch")
    expected_b2 = {
        "decision": "STAGE196B2B2_SEED_SPECIFIC_MULTIPATH_EFFECT",
        "recommended_next_stage": "STAGE196B2B3_NO_PROMOTION_INFERENCE_ONLY_COMPONENT_SWAP_PROBE",
        "blocking_reasons": [],
    }
    observed_b2 = {key: b2.get(key) for key in expected_b2}
    gate(gates, "source", "", "b2b2_decision_closure", expected_b2, observed_b2,
         observed_b2 == expected_b2, "B2-B2 decision closure mismatch")
    b2_contract = read_csv(b2_path.parent / B2B2_FILES[-1])
    gate(gates, "source", "", "b2b2_155_of_155_contract_closure", 155,
         {"rows": len(b2_contract), "passed": sum(str(r.get("passed", "")).lower() == "true" for r in b2_contract)},
         contract_closed(b2_contract, 155), "B2-B2 contract closure mismatch")
    static, epochs = read_csv(b2_path.parent / B2B2_FILES[2]), read_csv(b2_path.parent / B2B2_FILES[3])
    gate(gates, "population", "", "b2b2_16_static_primary_rows", 16, len(static), len(static) == 16, "static population mismatch")
    gate(gates, "population", "", "b2b2_320_paired_epoch_rows", 320, len(epochs), len(epochs) == 320, "epoch population mismatch")
    if not static or not set(PRIMARY_META).issubset(static[0]):
        raise ValueError("B2-B2 static metadata schema incomplete")
    index: dict[tuple[int, str, str, int], dict[str, str]] = {}
    counts = {s: {"recovery": Counter(), "harm": Counter()} for s in PRIMARY_SEEDS}
    for row in static:
        seed, pos = integer(row["seed"], "seed"), integer(row["dev_position"], "dev_position")
        key = (seed, str(row["id"]), str(row["source_row_id"]), pos)
        if key in index or seed not in PRIMARY_SEEDS or row["transition_role"] not in ("recovery", "harm"):
            raise ValueError("invalid or duplicate B2-B2 primary identity")
        if not row["path_class"]:
            raise ValueError("frozen row-level path_class is missing")
        index[key] = row
        counts[seed][row["transition_role"]][row["path_class"]] += 1
    normalized = {s: {r: dict(counts[s][r]) for r in ("recovery", "harm")} for s in PRIMARY_SEEDS}
    gate(gates, "population", "", "b2b2_frozen_path_counts", EXPECTED_PATHS, normalized,
         normalized == EXPECTED_PATHS, "frozen path-class counts changed")
    subtype_sign_fields = (
        "tail3_delta_frame", "tail3_delta_predicate",
        "tail3_delta_entitlement", "tail3_delta_margin",
    )
    seed185_harm_multichannel = [
        row for row in static
        if integer(row["seed"], "seed") == 185
        and row["transition_role"] == "harm"
        and row["path_class"] == "MULTI_CHANNEL_CONFLICT"
    ]
    sign_rows: list[dict[str, Any]] = []
    sign_error: str | None = None
    for row in seed185_harm_multichannel:
        identity = {
            "seed": integer(row["seed"], "seed"),
            "stable_row_id": row["stable_row_id"], "id": row["id"],
            "source_row_id": row["source_row_id"],
            "dev_position": integer(row["dev_position"], "dev_position"),
            "path_class": row["path_class"],
        }
        parsed = dict(identity)
        try:
            for field in subtype_sign_fields:
                parsed[field] = csv_number(row.get(field), field)
                if parsed[field] >= 0.0:
                    raise ValueError(
                        f"{field}: strictly negative frozen CSV value required; "
                        f"identity={identity!r}; observed={row.get(field)!r}"
                    )
        except ValueError as error:
            sign_error = f"identity={identity!r}; {error}"
            break
        sign_rows.append(parsed)
    sign_observed = {
        "expected_row_count": 3,
        "observed_row_count": len(seed185_harm_multichannel),
        "required_fields": list(subtype_sign_fields),
        "all_finite": sign_error is None and len(sign_rows) == len(seed185_harm_multichannel),
        "all_strictly_negative": sign_error is None and len(sign_rows) == 3,
        "rows": sign_rows,
    }
    gate(
        gates, "source", "", "b2b2_seed185_harm_multichannel_sign_closure",
        {"expected_row_count": 3, "required_fields": list(subtype_sign_fields),
         "all_finite": True, "all_strictly_negative": True},
        sign_observed,
        len(seed185_harm_multichannel) == 3 and sign_error is None and len(sign_rows) == 3,
        sign_error or "expected exactly three seed185 harm MULTI_CHANNEL_CONFLICT rows",
    )
    epoch_counts: Counter[tuple[int, str, str, int]] = Counter()
    for row in epochs:
        key = (integer(row["seed"], "epoch seed"), str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "epoch position"))
        if key not in index or integer(row["epoch"], "epoch") not in EPOCHS:
            raise ValueError("B2-B2 epoch row has no static identity or invalid epoch")
        for field in ("stable_row_id", "transition_role", "intervention_type"):
            if field in row and row[field] != index[key][field]:
                raise ValueError(f"B2-B2 epoch/static conflict: {field}")
        epoch_counts[key] += 1
    gate(gates, "population", "", "b2b2_twenty_epochs_per_primary_identity", 20,
         {"identities": len(epoch_counts), "counts": sorted(epoch_counts.values())},
         len(epoch_counts) == 16 and set(epoch_counts.values()) == {20}, "B2-B2 epoch closure failed")
    return {"b3": b3, "b2": b2, "primary": index, "b3_path": b3_path,
            "b2_path": b2_path, "graph": read_csv(b3_path.parent / B2B3_FILES[2])}


BRANCHES = ("temporal_mismatch", "predicate_mismatch", "temporal_adapter", "temporal_channel")


def branch_magnitude(row: dict[str, Any], branch: str) -> float:
    available = boolean(row[f"{branch}_available"], f"{branch}_available")
    active = boolean(row[f"{branch}_active"], f"{branch}_active")
    if branch in ("temporal_mismatch", "predicate_mismatch"):
        condition = boolean(row[f"{branch}_condition_input"], f"{branch}_condition_input")
        raw, transformed = row[f"raw_alpha_{branch.split('_')[0]}"], row[f"softplus_alpha_{branch.split('_')[0]}"]
        if not available:
            if active or raw is not None or transformed is not None:
                raise ValueError(f"{branch}: unavailable branch has causal state")
            return 0.0
        raw_n, transformed_n = number(raw, f"{branch} raw"), number(transformed, f"{branch} transformed")
        if active != condition or abs(softplus(raw_n) - transformed_n) > TOL:
            raise ValueError(f"{branch}: condition/transform closure failed")
        return transformed_n if active else 0.0
    if branch == "temporal_adapter":
        logit, probability = row["temporal_adapter_logit"], row["temporal_adapter_gate_probability"]
        scale = number(row["temporal_adapter_final_penalty_scale"], "adapter scale")
        expected_active = available and scale > 0.0
        if not available:
            if active or logit is not None or probability is not None:
                raise ValueError("unavailable temporal adapter has causal state")
            magnitude = 0.0
        else:
            logit_n, probability_n = number(logit, "adapter logit"), number(probability, "adapter probability", True)
            if abs(sigmoid(logit_n) - probability_n) > TOL:
                raise ValueError("adapter sigmoid closure failed")
            magnitude = probability_n * scale if expected_active else 0.0
        if active != expected_active or abs(magnitude - number(row["temporal_adapter_effective_penalty_scale"], "adapter effective")) > TOL:
            raise ValueError("adapter activity/effective-scale closure failed")
        return magnitude
    logit, probability, preservation = (row["temporal_channel_logit"], row["temporal_channel_gate_probability"], row["preservation_entitlement_prob"])
    scale = number(row["temporal_channel_gated_penalty_scale"], "channel scale")
    expected_active = available and scale > 0.0
    if not available:
        if active or logit is not None or probability is not None:
            raise ValueError("unavailable temporal channel has causal state")
        magnitude = 0.0
    else:
        logit_n = number(logit, "channel logit")
        probability_n = number(probability, "channel probability", True)
        if abs(sigmoid(logit_n) - probability_n) > TOL:
            raise ValueError("channel sigmoid closure failed")
        magnitude = probability_n * (1.0 - number(preservation, "preservation probability", True)) * scale if expected_active else 0.0
    if active != expected_active or abs(magnitude - number(row["temporal_channel_effective_scale"], "channel effective")) > TOL:
        raise ValueError("channel activity/effective-scale closure failed")
    return magnitude


def reconstruct(row: dict[str, Any]) -> dict[str, Any]:
    frame = number(row["frame_prob"], "frame_prob", True)
    predicate = number(row["predicate_coverage_prob"], "predicate_coverage_prob", True)
    sufficiency = number(row["sufficiency_prob"], "sufficiency_prob", True)
    entitlement = frame * predicate * sufficiency
    alpha = softplus(number(row["raw_alpha"], "raw_alpha"))
    if abs(alpha - number(row["softplus_alpha"], "softplus_alpha")) > TOL:
        raise ValueError("decision-head alpha transform mismatch")
    decision = (
        entitlement * number(row["negative_energy"], "negative_energy"),
        number(row["not_entitled_bias"], "not_entitled_bias") + alpha * (1.0 - entitlement),
        entitlement * number(row["positive_energy"], "positive_energy"),
    )
    branch_deltas, branch_error = {}, 0.0
    for branch in BRANCHES:
        magnitude = branch_magnitude(row, branch)
        delta = (-magnitude, magnitude, -magnitude)
        branch_deltas[branch] = delta
        for i, label in enumerate(("refute", "not_entitled", "support")):
            branch_error = max(branch_error, abs(delta[i] - number(row[f"{branch}_delta_{label}"], f"{branch} delta")))
    total = tuple(sum(branch_deltas[b][i] for b in BRANCHES) for i in range(3))
    total_error = max(abs(total[i] - number(row[f"total_final_delta_{('refute','not_entitled','support')[i]}"], "total delta")) for i in range(3))
    final = tuple(decision[i] + total[i] for i in range(3))
    margin = final[2] - final[1]
    return {"entitlement": entitlement, "decision": decision, "branch_deltas": branch_deltas,
            "branch_error": max(branch_error, total_error), "final": final, "margin": margin,
            "prediction": argmax_label(final)}


def row_errors(row: dict[str, Any], rebuilt: dict[str, Any]) -> dict[str, float | bool]:
    native_decision = tuple(number(row[f"decision_head_{x}_logit"], "decision logit") for x in ("refute", "not_entitled", "support"))
    native_final = tuple(number(row[f"final_{x}_logit"], "final logit") for x in ("refute", "not_entitled", "support"))
    return {
        "entitlement": abs(rebuilt["entitlement"] - number(row["entitlement_prob_native"], "native entitlement")),
        "decision": max(abs(rebuilt["decision"][i] - native_decision[i]) for i in range(3)),
        "branch": rebuilt["branch_error"],
        "final": max(abs(rebuilt["final"][i] - native_final[i]) for i in range(3)),
        "margin": abs(rebuilt["margin"] - number(row["final_support_vs_not_entitled_margin"], "native margin")),
        "prediction": rebuilt["prediction"] == row["final_native_prediction"] == row["native_prediction"],
    }


def validate_p0(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[tuple[int, int, str, str, int], dict[str, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    root = ns.stage196b2b3p0_run_root.resolve()
    observed_dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    gate(gates, "p0", "", "p0_exact_six_run_closure", sorted(RUNS), observed_dirs,
         observed_dirs == sorted(RUNS), "run-directory closure failed")
    if ns.stage196b2b3p0_runtime_git_commit != P0_COMMIT:
        raise ValueError("explicit P0 runtime commit disagrees with frozen authority")
    pairs: dict[tuple[int, int, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    summaries, manifest_hash_authorities = [], []
    composer_count = trajectory_count = composer_rows = trajectory_rows = prediction_equal = 0
    for run in RUNS:
        seed, mode = int(run[4:7]), run[8:]
        run_dir = root / run
        composer_dir, trajectory_dir = run_dir / "composer_inputs", run_dir / "trajectory"
        if not composer_dir.is_dir() or not trajectory_dir.is_dir():
            raise ValueError(f"{run}: composer_inputs/trajectory closure failed")
        composer_names = [f"stage196b2b3p0_epoch_composer_inputs_{e:03d}.jsonl" for e in EPOCHS]
        trajectory_names = [f"stage196b2p0_epoch_channels_{e:03d}.jsonl" for e in EPOCHS]
        manifest_name = "stage196b2b3p0_composer_input_manifest.json"
        observed_comp = sorted(p.name for p in composer_dir.iterdir() if p.is_file())
        observed_traj = sorted(p.name for p in trajectory_dir.iterdir() if p.is_file() and p.name.startswith("stage196b2p0_epoch_channels_"))
        gate(gates, "p0", run, "exact_composer_namespace", sorted([manifest_name, *composer_names]), observed_comp,
             observed_comp == sorted([manifest_name, *composer_names]), "composer namespace closure failed")
        gate(gates, "p0", run, "exact_trajectory_namespace", sorted(trajectory_names), observed_traj,
             observed_traj == sorted(trajectory_names), "trajectory namespace closure failed")
        manifest = read_json(composer_dir / manifest_name)
        manifest_commit = manifest.get("current_git_commit")
        manifest_ok = (
            manifest.get("completed") is True and manifest.get("prediction_equality_rate") == 1.0
            and number(manifest.get("maximum_decision_head_error"), "manifest decision error") <= TOL
            and number(manifest.get("maximum_final_logit_error"), "manifest final error") <= TOL
            and number(manifest.get("maximum_margin_error"), "manifest margin error") <= TOL
            and manifest_commit == P0_COMMIT and manifest.get("seed") == seed
            and manifest.get("gradient_ownership_mode") == mode
            and manifest.get("sidecar_files") == composer_names
        )
        gate(gates, "p0", run, "manifest_completion_and_provenance", True,
             {"completed": manifest.get("completed"), "commit": manifest_commit, "run_name": manifest.get("run_name"),
              "manifest_run_id": run, "seed": manifest.get("seed"), "mode": manifest.get("gradient_ownership_mode")},
             manifest_ok, "manifest completion/provenance mismatch")
        hash_authority = {k: v for k, v in manifest.items() if k.endswith("_file_sha256")}
        if not hash_authority or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in hash_authority.values()):
            raise ValueError(f"{run}: source-hash authority missing or malformed")
        manifest_hash_authorities.append(hash_authority)
        epoch_summaries = []
        for epoch, comp_name, traj_name in zip(EPOCHS, composer_names, trajectory_names):
            comp_path, traj_path = composer_dir / comp_name, trajectory_dir / traj_name
            expected_hash = manifest.get("sidecar_sha256", {}).get(comp_name)
            if expected_hash != sha256(comp_path):
                raise ValueError(f"{run}:{epoch}: composer sidecar hash mismatch")
            comp, traj = read_jsonl(comp_path), read_jsonl(traj_path)
            if len(comp) != 720 or len(traj) != 720:
                raise ValueError(f"{run}:{epoch}: expected 720 rows per sidecar")
            if any(set(t) != TRAJECTORY_FIELDS for t in traj):
                raise ValueError(f"{run}:{epoch}: trajectory schema mismatch")
            traj_index = {(str(t["id"]), str(t["source_row_id"]), integer(t["dev_position"], "trajectory position")): t for t in traj}
            if len(traj_index) != 720:
                raise ValueError(f"{run}:{epoch}: trajectory shared identity not unique")
            seen = {field: set() for field in ("stable_row_id", "id", "source_row_id", "dev_position")}
            maximum = {key: 0.0 for key in ("entitlement", "decision", "branch", "final", "margin")}
            equal = 0
            schema = set(comp[0])
            for row in comp:
                if set(row) != schema:
                    raise ValueError(f"{run}:{epoch}: composer schema drift")
                if row.get("current_git_commit") != P0_COMMIT or integer(row.get("seed"), "composer seed") != seed or integer(row.get("epoch"), "composer epoch") != epoch:
                    raise ValueError(f"{run}:{epoch}: row provenance mismatch")
                if row.get("gradient_ownership_mode") != mode:
                    raise ValueError(f"{run}:{epoch}: row mode mismatch")
                for field in seen:
                    value = integer(row[field], field) if field == "dev_position" else str(row[field])
                    if value in seen[field]:
                        raise ValueError(f"{run}:{epoch}: {field} is not unique")
                    seen[field].add(value)
                identity = (str(row["id"]), str(row["source_row_id"]), integer(row["dev_position"], "position"))
                trajectory = traj_index.get(identity)
                if trajectory is None:
                    raise ValueError(f"{run}:{epoch}: composer/trajectory identity mismatch")
                if trajectory["prediction"] != row["final_native_prediction"]:
                    raise ValueError(f"{run}:{epoch}: composer/trajectory prediction mismatch")
                rebuilt, errors = reconstruct(row), None
                errors = row_errors(row, rebuilt)
                for key in maximum:
                    maximum[key] = max(maximum[key], float(errors[key]))
                equal += int(errors["prediction"] is True)
                pair_key = (seed, epoch, *identity)
                if mode in pairs[pair_key]:
                    raise ValueError("duplicate treatment row in pair")
                stored = dict(row)
                stored["manifest_run_id"] = run
                stored["_native"] = rebuilt
                pairs[pair_key][mode] = stored
            if max(maximum.values()) > TOL or equal != 720:
                raise ValueError(f"{run}:{epoch}: independent native reconstruction failed: {maximum}")
            summaries.append({"run": run, "seed": seed, "gradient_mode": mode, "epoch": epoch,
                              "row_count": 720, "maximum_entitlement_error": maximum["entitlement"],
                              "maximum_decision_head_error": maximum["decision"],
                              "maximum_branch_delta_sum_error": maximum["branch"],
                              "maximum_final_logit_error": maximum["final"], "maximum_margin_error": maximum["margin"],
                              "prediction_equality_rate": equal / 720})
            epoch_summaries.append(maximum)
            composer_count += 1; trajectory_count += 1; composer_rows += len(comp); trajectory_rows += len(traj); prediction_equal += equal
    gate(gates, "p0", "", "p0_exact_120_composer_sidecars", 120, composer_count, composer_count == 120, "composer sidecar count mismatch")
    gate(gates, "p0", "", "p0_exact_120_trajectory_sidecars", 120, trajectory_count, trajectory_count == 120, "trajectory sidecar count mismatch")
    gate(gates, "p0", "", "p0_exact_86400_row_closure", {"composer": 86400, "trajectory": 86400},
         {"composer": composer_rows, "trajectory": trajectory_rows}, composer_rows == trajectory_rows == 86400, "global row closure failed")
    gate(gates, "provenance", "", "p0_runtime_commit_agreement", P0_COMMIT, ns.stage196b2b3p0_runtime_git_commit,
         ns.stage196b2b3p0_runtime_git_commit == P0_COMMIT, "runtime commit disagreement")
    gate(gates, "provenance", "", "p0_source_hash_agreement", "one identical source-hash mapping across six manifests",
         manifest_hash_authorities, len(manifest_hash_authorities) == 6 and all(x == manifest_hash_authorities[0] for x in manifest_hash_authorities),
         "manifest source hashes disagree")
    pair_ok = len(pairs) == 43200 and all(set(v) == set(MODES) for v in pairs.values())
    gate(gates, "pairing", "", "joint_frame_local_exact_pairing", 43200,
         {"pair_count": len(pairs), "bad_pairs": sum(set(v) != set(MODES) for v in pairs.values())}, pair_ok, "pair closure failed")
    global_row = {"run": "GLOBAL", "seed": "", "gradient_mode": "", "epoch": "", "row_count": composer_rows,
                  "maximum_entitlement_error": max(r["maximum_entitlement_error"] for r in summaries),
                  "maximum_decision_head_error": max(r["maximum_decision_head_error"] for r in summaries),
                  "maximum_branch_delta_sum_error": max(r["maximum_branch_delta_sum_error"] for r in summaries),
                  "maximum_final_logit_error": max(r["maximum_final_logit_error"] for r in summaries),
                  "maximum_margin_error": max(r["maximum_margin_error"] for r in summaries),
                  "prediction_equality_rate": prediction_equal / composer_rows}
    summaries.append(global_row)
    return pairs, summaries, global_row


def donor_state(recipient: dict[str, Any], donor: dict[str, Any], variant: str) -> dict[str, Any]:
    result = {k: v for k, v in recipient.items() if not k.startswith("_")}
    fields = {
        "FRAME_ONLY": ("frame_prob",),
        "PREDICATE_ONLY": ("predicate_coverage_prob",),
        "SUFFICIENCY_ONLY": ("sufficiency_prob",),
        "ENTITLEMENT_PRIMITIVES": ("frame_prob", "predicate_coverage_prob", "sufficiency_prob"),
        "POLARITY_ONLY": ("positive_energy", "negative_energy"),
        "ENTITLEMENT_PLUS_POLARITY": ("frame_prob", "predicate_coverage_prob", "sufficiency_prob", "positive_energy", "negative_energy"),
    }
    if variant == "FULL_COMPOSER_INPUT_POSITIVE_CONTROL":
        prohibited_derived = {
            "entitlement_prob_native", "entitlement_prob_recomputed",
            "decision_head_refute_logit", "decision_head_not_entitled_logit", "decision_head_support_logit",
            "total_final_delta_refute", "total_final_delta_not_entitled", "total_final_delta_support",
            "final_refute_logit", "final_not_entitled_logit", "final_support_logit",
            "final_support_vs_not_entitled_margin", "final_native_prediction", "native_prediction",
            "reconstructed_prediction", "prediction_equal",
        }
        causal = set(fields["ENTITLEMENT_PLUS_POLARITY"]) | {
            "not_entitled_bias", "raw_alpha", "softplus_alpha",
            *[f"{b}_{suffix}" for b in BRANCHES for suffix in (
                "available", "condition_input", "active", "logit", "gate_probability",
                "final_penalty_scale", "effective_penalty_scale", "gated_penalty_scale", "effective_scale")],
            "raw_alpha_temporal", "softplus_alpha_temporal", "raw_alpha_predicate", "softplus_alpha_predicate",
            "preservation_entitlement_prob",
        }
        for name in causal - prohibited_derived:
            if name in donor:
                result[name] = donor[name]
        return result
    for name in fields[variant]:
        result[name] = donor[name]
    return result


def run_positive_control(pairs: dict[Any, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    failures, max_logit, max_margin, equal, total = [], 0.0, 0.0, 0, 0
    for key, arms in pairs.items():
        for direction in DIRECTIONS:
            recipient_mode, donor_mode = DIRECTION_MODES[direction]
            donor = arms[donor_mode]
            result = reconstruct(donor_state(arms[recipient_mode], donor, "FULL_COMPOSER_INPUT_POSITIVE_CONTROL"))
            native = donor["_native"]
            errors = [abs(result["final"][i] - native["final"][i]) for i in range(3)]
            margin_error = abs(result["margin"] - native["margin"])
            prediction_ok = result["prediction"] == native["prediction"]
            max_logit, max_margin = max(max_logit, *errors), max(max_margin, margin_error)
            equal += int(prediction_ok); total += 1
            if max(errors) > TOL or margin_error > TOL or not prediction_ok:
                failures.append({"seed": key[0], "epoch": key[1], "id": key[2], "source_row_id": key[3],
                                 "dev_position": key[4], "direction": direction,
                                 "logit_errors": dict(zip(("REFUTE", "NOT_ENTITLED", "SUPPORT"), errors)),
                                 "margin_error": margin_error, "prediction_equal": prediction_ok})
    return {"row_count": total, "maximum_absolute_donor_logit_error": max_logit,
            "maximum_donor_margin_error": max_margin, "prediction_equality_rate": equal / total,
            "passed": total == 86400 and max_logit <= TOL and max_margin <= TOL and equal == total,
            "failing_rows": failures}


def subtype(meta: dict[str, str]) -> str:
    seed, role, path = integer(meta["seed"], "seed"), meta["transition_role"], meta["path_class"]
    if role == "recovery" and seed == 184 and path == "MULTI_CHANNEL_CONFLICT": return "RECOVERY_SEED184_MULTI_CHANNEL_CONFLICT"
    if role == "recovery" and seed == 185 and path == "FRAME_ENTITLEMENT_GAIN": return "RECOVERY_SEED185_FRAME_ENTITLEMENT_GAIN"
    if path == "POLARITY_OVERRIDE_DESPITE_FRAME_GAIN": return "POLARITY_OVERRIDE_ROWS"
    if path == "COMPOSITION_DIVERGENCE_WITHOUT_LOCAL_BOUNDARY": return "COMPOSITION_RESIDUAL_ROWS"
    if path == "FRAME_ENTITLEMENT_LOSS": return "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
    if seed == 185 and role == "harm" and path == "MULTI_CHANNEL_CONFLICT":
        required = ("tail3_delta_frame", "tail3_delta_predicate", "tail3_delta_entitlement", "tail3_delta_margin")
        if not all(csv_number(meta[x], x) < 0.0 for x in required):
            raise ValueError("seed185 harm MULTI_CHANNEL_CONFLICT subtype signs changed")
        return "FRAME_ENTITLEMENT_LOSS_LIKE_ROWS"
    raise ValueError("primary row has no frozen subtype")


def swap_rows(pairs: dict[Any, dict[str, dict[str, Any]]], primary: dict[Any, dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for key in sorted(pairs):
        seed, epoch, identity, source, position = key
        meta = primary.get((seed, identity, source, position))
        if meta is None:
            continue
        for direction in DIRECTIONS:
            recipient_mode, donor_mode = DIRECTION_MODES[direction]
            recipient, donor = pairs[key][recipient_mode], pairs[key][donor_mode]
            for variant in VARIANTS:
                cf = reconstruct(donor_state(recipient, donor, variant))
                rec, don = recipient["_native"], donor["_native"]
                denominator = don["margin"] - rec["margin"]
                movement = cf["margin"] - rec["margin"]
                fraction = None if abs(denominator) <= TOL else movement / denominator
                output.append({
                    "seed": seed, "epoch": epoch, "stable_row_id": meta["stable_row_id"], "id": identity,
                    "source_row_id": source, "dev_position": position, "transition_role": meta["transition_role"],
                    "intervention_type": meta["intervention_type"], "path_class": meta["path_class"],
                    "direction": direction, "variant": variant, "recipient_mode": recipient_mode, "donor_mode": donor_mode,
                    "recipient_refute_logit": rec["final"][0], "recipient_not_entitled_logit": rec["final"][1],
                    "recipient_support_logit": rec["final"][2], "recipient_margin": rec["margin"], "recipient_prediction": rec["prediction"],
                    "donor_refute_logit": don["final"][0], "donor_not_entitled_logit": don["final"][1],
                    "donor_support_logit": don["final"][2], "donor_margin": don["margin"], "donor_prediction": don["prediction"],
                    "counterfactual_refute_logit": cf["final"][0], "counterfactual_not_entitled_logit": cf["final"][1],
                    "counterfactual_support_logit": cf["final"][2], "counterfactual_margin": cf["margin"], "counterfactual_prediction": cf["prediction"],
                    "counterfactual_minus_recipient_refute": cf["final"][0] - rec["final"][0],
                    "counterfactual_minus_recipient_not_entitled": cf["final"][1] - rec["final"][1],
                    "counterfactual_minus_recipient_support": cf["final"][2] - rec["final"][2],
                    "counterfactual_minus_recipient_margin": movement, "donor_minus_recipient_margin": denominator,
                    "counterfactual_toward_donor_margin": (movement * denominator > 0.0) if abs(denominator) > TOL else None,
                    "signed_margin_closure_fraction": fraction,
                    "recipient_prediction_changed": cf["prediction"] != rec["prediction"],
                    "donor_prediction_reproduced": cf["prediction"] == don["prediction"],
                    "recipient_prediction_preserved": cf["prediction"] == rec["prediction"],
                })
    if len(output) != 3840:
        raise ValueError(f"component row count is {len(output)}, expected 3840")
    return output


def summarize(rows: list[dict[str, Any]], primary: dict[Any, dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["seed"], row["stable_row_id"], row["direction"], row["variant"])].append(row)
    row_summary = []
    meta_by_stable = {(integer(v["seed"], "seed"), v["stable_row_id"]): v for v in primary.values()}
    for (seed, stable, direction, variant), values in sorted(buckets.items()):
        values = sorted(values, key=lambda x: x["epoch"])
        if [v["epoch"] for v in values] != list(EPOCHS):
            raise ValueError("summary identity lacks 20 epochs")
        tail = [v for v in values if v["epoch"] in TAIL]
        meta = meta_by_stable[(seed, stable)]
        rec = [v["recipient_prediction"] for v in tail]; don = [v["donor_prediction"] for v in tail]; cf = [v["counterfactual_prediction"] for v in tail]
        row_summary.append({"seed": seed, "stable_row_id": stable, "id": meta["id"], "source_row_id": meta["source_row_id"],
                            "dev_position": integer(meta["dev_position"], "position"), "transition_role": meta["transition_role"],
                            "intervention_type": meta["intervention_type"], "path_class": meta["path_class"], "subtype": subtype(meta),
                            "direction": direction, "variant": variant, "tail_epochs": list(TAIL),
                            "recipient_tail_predictions": rec, "donor_tail_predictions": don, "counterfactual_tail_predictions": cf,
                            "recipient_tail_status": tail_status(rec), "donor_tail_status": tail_status(don), "counterfactual_tail_status": tail_status(cf),
                            "donor_tail_reproduced": cf == don, "recipient_tail_preserved": cf == rec,
                            "mean_tail_margin_shift": sum(v["counterfactual_minus_recipient_margin"] for v in tail) / 3.0})
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    subtypes: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in row_summary:
        groups[(row["seed"], row["transition_role"], row["direction"], row["variant"])].append(row)
        subtypes[(row["subtype"], row["seed"], row["transition_role"], row["path_class"], row["direction"], row["variant"])].append(row)
    group_rows = []
    for key, values in sorted(groups.items()):
        n = len(values)
        group_rows.append(dict(zip(("seed", "transition_role", "direction", "variant"), key), identity_count=n,
                               donor_tail_reproduced_count=sum(v["donor_tail_reproduced"] for v in values),
                               donor_tail_reproduced_rate=sum(v["donor_tail_reproduced"] for v in values) / n,
                               recipient_tail_preserved_count=sum(v["recipient_tail_preserved"] for v in values),
                               recipient_tail_preserved_rate=sum(v["recipient_tail_preserved"] for v in values) / n,
                               mean_tail_margin_shift=sum(v["mean_tail_margin_shift"] for v in values) / n,
                               path_class_counts=dict(Counter(v["path_class"] for v in values)),
                               intervention_type_counts=dict(Counter(v["intervention_type"] for v in values))))
    subtype_rows = []
    for key, values in sorted(subtypes.items()):
        n = len(values)
        subtype_rows.append(dict(zip(("subtype", "seed", "transition_role", "path_class", "direction", "variant"), key), identity_count=n,
                                 donor_tail_reproduced_count=sum(v["donor_tail_reproduced"] for v in values),
                                 donor_tail_reproduced_rate=sum(v["donor_tail_reproduced"] for v in values) / n,
                                 recipient_tail_preserved_count=sum(v["recipient_tail_preserved"] for v in values),
                                 recipient_tail_preserved_rate=sum(v["recipient_tail_preserved"] for v in values) / n))
    return row_summary, group_rows, subtype_rows


def decide(row_summary: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Resolve the old analyzer's unavailable-data placeholders with strict booleans.

    Frozen semantics: tail epochs only; forward recovery must reproduce its donor;
    forward harm must preserve its recipient; both primary seeds are required for a
    cross-seed claim.  No fitted threshold or score optimization is introduced.
    """
    forward = [r for r in row_summary if r["direction"] == DIRECTIONS[0]]
    seed_pass: dict[int, dict[str, bool]] = {s: {} for s in PRIMARY_SEEDS}
    detail: dict[str, Any] = {}
    for seed in PRIMARY_SEEDS:
        for variant in VARIANTS:
            selected = [r for r in forward if r["seed"] == seed and r["variant"] == variant]
            recoveries = [r for r in selected if r["transition_role"] == "recovery"]
            harms = [r for r in selected if r["transition_role"] == "harm"]
            passed = (len(recoveries) == EXPECTED_PRIMARY[seed]["recovery"] and len(harms) == EXPECTED_PRIMARY[seed]["harm"]
                      and all(r["donor_tail_reproduced"] for r in recoveries)
                      and all(r["recipient_tail_preserved"] for r in harms))
            seed_pass[seed][variant] = passed
            detail[f"seed{seed}:{variant}"] = {"recovery_donor_reproduction": sum(r["donor_tail_reproduced"] for r in recoveries),
                                               "recovery_denominator": len(recoveries), "harm_recipient_preservation": sum(r["recipient_tail_preserved"] for r in harms),
                                               "harm_denominator": len(harms), "passed": passed}
    cross = {variant: all(seed_pass[s][variant] for s in PRIMARY_SEEDS) for variant in VARIANTS}
    differing = [v for v in VARIANTS if seed_pass[184][v] != seed_pass[185][v]]
    if cross["ENTITLEMENT_PRIMITIVES"]:
        decision = DECISIONS[0]
    elif cross["ENTITLEMENT_PLUS_POLARITY"] and not cross["ENTITLEMENT_PRIMITIVES"] and not cross["POLARITY_ONLY"]:
        decision = DECISIONS[1]
    elif cross["POLARITY_ONLY"]:
        decision = DECISIONS[2]
    elif differing:
        decision = DECISIONS[4]
    else:
        decision = DECISIONS[3]
    return decision, {"tail_epochs": list(TAIL), "seed_level_predicates": detail,
                      "cross_seed_variant_pass": cross, "seed_conflicting_variants": differing,
                      "decision": decision}


def graph_rows(source: list[dict[str, str]]) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    if not source:
        raise ValueError("B2-B3 composer graph is empty")
    header = tuple(source[0]) + GRAPH_EXTRA
    mapping = {
        "frame_probability": ("frame_prob", "primitive", "FRAME_ONLY|ENTITLEMENT_PRIMITIVES|ENTITLEMENT_PLUS_POLARITY", "entitlement and decision head", "E=frame_prob*predicate_coverage_prob*sufficiency_prob"),
        "predicate_coverage_probability": ("predicate_coverage_prob", "primitive", "PREDICATE_ONLY|ENTITLEMENT_PRIMITIVES|ENTITLEMENT_PLUS_POLARITY", "entitlement and decision head", "same product"),
        "sufficiency_probability": ("sufficiency_prob", "primitive", "SUFFICIENCY_ONLY|ENTITLEMENT_PRIMITIVES|ENTITLEMENT_PLUS_POLARITY", "entitlement and decision head", "same product"),
        "positive_energy": ("positive_energy", "polarity primitive", "POLARITY_ONLY|ENTITLEMENT_PLUS_POLARITY", "decision SUPPORT", "E*positive_energy"),
        "negative_energy": ("negative_energy", "polarity primitive", "POLARITY_ONLY|ENTITLEMENT_PLUS_POLARITY", "decision REFUTE", "E*negative_energy"),
    }
    output = []
    for row in source:
        exported = str(row.get("p0_sidecar_field", ""))
        details = mapping.get(exported, (exported, "source-authority field", "", str(row.get("consumed_by", "")), str(row.get("native_expression", ""))))
        output.append({**row, "exported_b2b3p0_field": details[0], "recomposition_field_role": details[1],
                       "swapped_by_variants": details[2], "downstream_dependency": details[3],
                       "exact_formula_implemented": details[4], "positive_control_inclusion": True})
    # Explicit exhaustive final-modulation dependency audit; source classifications above remain untouched.
    for branch, fields, formula in (
        ("temporal_mismatch", "condition_input,raw_alpha_temporal", "active*softplus(raw);[-m,+m,-m]"),
        ("predicate_mismatch", "condition_input,raw_alpha_predicate", "active*softplus(raw);[-m,+m,-m]"),
        ("temporal_adapter", "adapter_logit,final_penalty_scale", "sigmoid(logit)*scale;[-m,+m,-m]"),
        ("temporal_channel", "channel_logit,preservation_entitlement_prob,gated_penalty_scale", "sigmoid(logit)*(1-preservation)*scale;[-m,+m,-m]"),
    ):
        blank = {name: "" for name in source[0]}
        output.append({**blank, "exported_b2b3p0_field": fields, "recomposition_field_role": "final-modulation branch",
                       "swapped_by_variants": "FULL_COMPOSER_INPUT_POSITIVE_CONTROL",
                       "downstream_dependency": "REFUTE-,NOT_ENTITLED+,SUPPORT-; independent of six component subsets",
                       "exact_formula_implemented": formula, "positive_control_inclusion": True})
    return header, output


def activity() -> dict[str, bool]:
    return {"training_performed": False, "model_loaded": False, "checkpoint_loaded": False,
            "regression_or_interpolation_used": False, "polarity_margin_used_as_causal_input": False,
            "frozen_mamba_scope": True, "artifact_only": True}


def analyze(ns: argparse.Namespace, gates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    repo, output = ns.repo_root.resolve(), ns.output_dir.resolve()
    gate(gates, "path", "", "repo_root", "existing directory", str(repo), repo.is_dir(), "invalid repo root")
    inputs = [ns.stage196b2b3_analysis_json.resolve(), ns.stage196b2b2_analysis_json.resolve(), ns.stage196b2b3p0_run_root.resolve()]
    separated = not output.exists() and all(output != p and output not in p.parents and p not in output.parents for p in inputs)
    gate(gates, "path", "", "fresh_separate_output_directory", True, {"output": str(output), "exists": output.exists()}, separated, "output exists or overlaps a source")
    commit_ok = re.fullmatch(r"[0-9a-f]{40}", ns.current_git_commit or "") is not None
    gate(gates, "provenance", "", "current_git_commit_format", "40 lowercase hex", ns.current_git_commit, commit_ok, "invalid current commit")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    gate(gates, "provenance", "", "current_git_commit_equals_head", ns.current_git_commit, head, head == ns.current_git_commit, "current commit differs from HEAD")
    authorities = validate_authorities(ns, gates)
    pairs, native, native_global = validate_p0(ns, gates)
    for key, meta in authorities["primary"].items():
        seed, identity, source, position = key
        for epoch in EPOCHS:
            pair = pairs.get((seed, epoch, identity, source, position))
            if pair is None:
                raise ValueError("B2-B2 primary identity missing from composer pairs")
            for arm in pair.values():
                if str(arm["stable_row_id"]) != str(meta["stable_row_id"]):
                    raise ValueError("B2-B2/composer stable_row_id disagreement")
    gate(gates, "population", "", "primary_16_identity_320_epoch_closure", {"identities": 16, "identity_epochs": 320},
         {"identities": len(authorities["primary"]), "identity_epochs": len(authorities["primary"]) * 20}, len(authorities["primary"]) == 16, "primary linkage failed")
    frozen_path = (repo / "scripts" / "analyze_stage196b2b3_exact_component_recomposition.py").resolve()
    if not frozen_path.is_file():
        raise ValueError("frozen B2-B3 decision-rule source is absent")
    provenance = {"source_path": str(frozen_path), "sha256": sha256(frozen_path),
                  "reuse": "decision names, tail status, seed roles, recovery/harm, subtype populations, promotion prohibitions; unavailable data placeholders replaced by exact rows"}
    gate(gates, "provenance", "", "frozen_decision_rule_provenance", "existing source path and SHA-256", provenance,
         bool(re.fullmatch(r"[0-9a-f]{64}", provenance["sha256"])), "decision provenance failed")
    positive = run_positive_control(pairs)
    gate(gates, "positive_control", "", "full_positive_control_row_count", 86400, positive["row_count"], positive["row_count"] == 86400, "positive-control row count mismatch")
    # Tolerance failures are scientific fail-closed outcomes, not malformed-input exceptions.
    positive_passed = positive["passed"]
    gates.extend([
        {"scope": "positive_control", "run": "", "gate": "donor_logit_tolerance", "required": TOL,
         "observed": positive["maximum_absolute_donor_logit_error"], "passed": positive["maximum_absolute_donor_logit_error"] <= TOL,
         "blocking_reason": "" if positive["maximum_absolute_donor_logit_error"] <= TOL else "donor-logit reproduction failed"},
        {"scope": "positive_control", "run": "", "gate": "donor_margin_tolerance", "required": TOL,
         "observed": positive["maximum_donor_margin_error"], "passed": positive["maximum_donor_margin_error"] <= TOL,
         "blocking_reason": "" if positive["maximum_donor_margin_error"] <= TOL else "donor-margin reproduction failed"},
        {"scope": "positive_control", "run": "", "gate": "donor_prediction_equality", "required": 1.0,
         "observed": positive["prediction_equality_rate"], "passed": positive["prediction_equality_rate"] == 1.0,
         "blocking_reason": "" if positive["prediction_equality_rate"] == 1.0 else "donor prediction reproduction failed"},
    ])
    swaps: list[dict[str, Any]] = []
    row_summary: list[dict[str, Any]] = []; groups: list[dict[str, Any]] = []; subtypes: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {"evaluated": False, "reason": "positive control failed"}
    if positive_passed:
        swaps = swap_rows(pairs, authorities["primary"])
        row_summary, groups, subtypes = summarize(swaps, authorities["primary"])
        decision, evaluation = decide(row_summary)
        recommended, authorized = NEXT[decision], (
            "Within the frozen-Mamba, frozen-composer six-run population, exact native-input interventions may be interpreted under the frozen tail, recovery/harm, subtype, direction, and seed rules."
        )
    else:
        decision, recommended = POSITIVE_FAILED, POSITIVE_REPAIR
        authorized = "No component counterfactual interpretation is authorized because the full-composer donor positive control failed."
    component_ok = (not positive_passed and len(swaps) == 0) or (positive_passed and len(swaps) == 3840)
    gate(gates, "component", "", "component_row_count_or_fail_closed", 3840 if positive_passed else 0, len(swaps), component_ok, "component execution did not obey positive-control gate")
    if positive_passed:
        gate(gates, "component", "", "two_direction_closure", list(DIRECTIONS), sorted(set(r["direction"] for r in swaps)), sorted(set(r["direction"] for r in swaps)) == sorted(DIRECTIONS), "direction closure failed")
        gate(gates, "component", "", "six_variant_closure", list(VARIANTS), sorted(set(r["variant"] for r in swaps)), sorted(set(r["variant"] for r in swaps)) == sorted(VARIANTS), "variant closure failed")
    graph_header, graph = graph_rows(authorities["graph"])
    source_paths = {"stage196b2b3_analysis": str(authorities["b3_path"]), "stage196b2b2_analysis": str(authorities["b2_path"]),
                    "stage196b2b3p0_run_root": str(ns.stage196b2b3p0_run_root.resolve()), "frozen_decision_rules": str(frozen_path)}
    source_hashes = {str(p): sha256(p) for p in [*(authorities["b3_path"].parent / n for n in B2B3_FILES),
                                                    *(authorities["b2_path"].parent / n for n in B2B2_FILES), frozen_path]}
    seed_results = {str(seed): {k.split(":", 1)[1]: v for k, v in evaluation.get("seed_level_predicates", {}).items() if k.startswith(f"seed{seed}:")} for seed in PRIMARY_SEEDS}
    analysis = {
        "stage": STAGE, "scientific_label": LABEL, "decision": decision, "recommended_next_stage": recommended,
        "blocking_reasons": [] if decision not in (INCOMPLETE,) else ["analysis incomplete"],
        "current_git_commit": ns.current_git_commit, "stage196b2b3p0_runtime_git_commit": ns.stage196b2b3p0_runtime_git_commit,
        "source_artifact_paths": source_paths, "source_hashes": source_hashes, "decision_rule_provenance": provenance,
        "native_reconstruction": {**native_global, "tolerance": TOL, "passed": True, "independent_of_sidecar_diagnostics": True},
        "positive_control": positive, "primary_population": {"identity_count": 16, "identity_epoch_count": 320,
            "seed_roles": {"183": "contrast-only", "184": "primary positive", "185": "primary positive"}},
        "component_variants": list(VARIANTS), "directions": list(DIRECTIONS),
        "row_counts": {"composer": 86400, "trajectory": 86400, "paired_states": 43200,
                       "positive_control_directional": positive["row_count"], "component_swap": len(swaps),
                       "row_swap_summary": len(row_summary), "group_swap_summary": len(groups), "subtype_summary": len(subtypes)},
        "seed_level_results": seed_results, "transition_role_results": groups, "path_class_results": groups,
        "subtype_results": subtypes, "cross_seed_consistency": {"seed183_excluded": True,
            "variant_pass": evaluation.get("cross_seed_variant_pass", {}), "conflicts": evaluation.get("seed_conflicting_variants", [])},
        "decision_rule_evaluation": evaluation, "authorized_interpretation": authorized,
        "remaining_uncertainty": ["inference-only within-model intervention probe", "frozen-Mamba and observed clean-dev epochs only"],
        "prohibited_claims": list(PROHIBITED), **activity(),
    }
    gate(gates, "analysis", "", "analysis_completion", True, True, True, "")
    gate(gates, "output", "", "exact_nine_output_plan", list(OUTPUTS), list(OUTPUTS), True, "")
    return analysis, {"graph_header": graph_header, "graph": graph, "native": native, "swaps": swaps,
                      "rows": row_summary, "groups": groups, "subtypes": subtypes}


REPORT_SECTIONS = (
    "Executive decision", "Authorized interpretation", "Source closure", "Native reconstruction",
    "Full-composer positive control", "Frame-only result", "Predicate-only result", "Sufficiency-only result",
    "Entitlement-primitives result", "Polarity-only result", "Entitlement-plus-polarity result",
    "Recovery analysis", "Preservation-harm analysis", "B2-B2 subtype analysis", "Seed184 result",
    "Seed185 result", "Cross-seed consistency", "Decision-rule evaluation", "Remaining uncertainty",
    "Prohibited claims", "Recommended next stage",
)


def report(analysis: dict[str, Any]) -> str:
    evaluation = analysis.get("decision_rule_evaluation", {})
    seed = analysis.get("seed_level_results", {})
    variant_text = {v: json.dumps({s: seed.get(s, {}).get(v) for s in ("184", "185")}, sort_keys=True) for v in VARIANTS}
    bodies = (
        f"`{analysis['decision']}`", analysis["authorized_interpretation"],
        json.dumps(analysis["source_artifact_paths"], sort_keys=True), json.dumps(analysis["native_reconstruction"], sort_keys=True),
        json.dumps(analysis["positive_control"], sort_keys=True), variant_text["FRAME_ONLY"], variant_text["PREDICATE_ONLY"],
        variant_text["SUFFICIENCY_ONLY"], variant_text["ENTITLEMENT_PRIMITIVES"], variant_text["POLARITY_ONLY"],
        variant_text["ENTITLEMENT_PLUS_POLARITY"], "Forward recovery uses donor-tail reproduction on the frozen B2-B2 recovery rows.",
        "Forward harm uses recipient-tail preservation; REFUTE and NOT_ENTITLED remain distinct.",
        json.dumps(analysis.get("subtype_results", []), sort_keys=True), json.dumps(seed.get("184", {}), sort_keys=True),
        json.dumps(seed.get("185", {}), sort_keys=True), json.dumps(analysis["cross_seed_consistency"], sort_keys=True),
        json.dumps(evaluation, sort_keys=True), "\n".join(f"- {x}" for x in analysis["remaining_uncertainty"]),
        "\n".join(f"- {x}" for x in analysis["prohibited_claims"]),
        f"`{analysis['recommended_next_stage']}`\n\nNo training or promotion is authorized.",
    )
    return f"# {STAGE}: {LABEL}\n\n" + "\n\n".join(f"## {h}\n\n{b}" for h, b in zip(REPORT_SECTIONS, bodies)) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is True: return "true"
    if value is False: return "false"
    if value is None: return ""
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


def render(analysis: dict[str, Any], tables: dict[str, Any], gates: list[dict[str, Any]]) -> dict[str, str]:
    return {
        OUTPUTS[0]: json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        OUTPUTS[1]: report(analysis), OUTPUTS[2]: render_csv(tables["graph_header"], tables["graph"]),
        OUTPUTS[3]: render_csv(NATIVE_H, tables["native"]), OUTPUTS[4]: render_csv(SWAP_H, tables["swaps"]),
        OUTPUTS[5]: render_csv(ROW_H, tables["rows"]), OUTPUTS[6]: render_csv(GROUP_H, tables["groups"]),
        OUTPUTS[7]: render_csv(SUBTYPE_H, tables["subtypes"]), OUTPUTS[8]: render_csv(CONTRACT_H, gates),
    }


def atomic_write_outputs(output: Path, payloads: dict[str, str]) -> None:
    if set(payloads) != set(OUTPUTS) or output.exists():
        raise RuntimeError("refusing non-nine-file or overwrite output")
    output.mkdir(parents=True, exist_ok=False)
    temporary: list[tuple[Path, Path]] = []
    try:
        for name in OUTPUTS:
            final = output / name
            temp = output / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
            with temp.open("x", encoding="utf-8", newline="") as handle:
                handle.write(payloads[name]); handle.flush(); os.fsync(handle.fileno())
            temporary.append((temp, final))
        for temp, final in temporary:
            os.replace(temp, final)
        if sorted(p.name for p in output.iterdir()) != sorted(OUTPUTS):
            raise RuntimeError("written output closure is not exactly nine files")
    finally:
        for temp, _ in temporary:
            if temp.exists(): temp.unlink()


def main() -> int:
    ns = parse_args()
    gates: list[dict[str, Any]] = []
    analysis, tables = analyze(ns, gates)
    payloads = render(analysis, tables, gates)
    atomic_write_outputs(ns.output_dir.resolve(), payloads)
    return 0 if analysis["decision"] != POSITIVE_FAILED else 3


if __name__ == "__main__":
    raise SystemExit(main())
