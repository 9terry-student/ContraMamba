#!/usr/bin/env python3
"""Stage183-A static compatible-frame positive-preservation design audit.

This script reads source, JSON/JSONL, and closure artifacts only.  It does not
import torch, load a checkpoint, instantiate a model, or run training/forward.
It writes a deterministic report package beneath --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage183-A"
DECISION = "STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST"
NEXT_STAGE = "STAGE184_CONTROLLED_TRAIN_INTEGRITY_MASK_SPEC"
AUTHORITATIVE_DATA = Path("data/controlled_v5_v3_without_time_swap.jsonl")
REQUIRED_SCHEMA = {
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
}
INTEGRITY_FIELDS = {
    "grammar_valid": ("grammar_valid", "stage183_grammar_valid"),
    "intervention_contract_exact": (
        "intervention_contract_exact",
        "stage183_intervention_contract_exact",
    ),
    "polarity_contamination_absent": (
        "polarity_contamination_absent",
        "stage183_polarity_contamination_absent",
    ),
    "schema_resolved": ("schema_resolved", "stage183_schema_resolved"),
    "canonical_row_valid": ("canonical_row_valid", "stage183_canonical_row_valid"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=AUTHORITATIVE_DATA)
    parser.add_argument("--split-seed", type=int, default=174)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    return parser.parse_args()


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_cell(row.get(key)) for key in headers})


def json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def split_by_pair_id(
    rows: list[dict[str, Any]], dev_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("--dev-ratio must be between 0 and 1")
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    if len(pair_ids) < 2:
        raise ValueError("at least two pair_id values are required")
    random.Random(seed).shuffle(pair_ids)
    dev_count = min(len(pair_ids) - 1, max(1, round(len(pair_ids) * dev_ratio)))
    dev_ids = set(pair_ids[:dev_count])
    train = [row for row in rows if str(row["pair_id"]) not in dev_ids]
    dev = [row for row in rows if str(row["pair_id"]) in dev_ids]
    return train, dev


def topology_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    labels = Counter(row.get("frame_compatible_label") for row in rows)
    pairs = {str(row.get("pair_id")) for row in rows}
    result = [
        {
            "split": split,
            "metric": "row_count",
            "value": len(rows),
            "denominator": "",
            "rate": "",
            "source": "frame_compatible_label",
        },
        {
            "split": split,
            "metric": "pair_count",
            "value": len(pairs),
            "denominator": "",
            "rate": "",
            "source": "pair_id",
        },
    ]
    for label, name in ((1, "compatible_positive"), (0, "incompatible_negative")):
        count = labels.get(label, 0)
        result.append(
            {
                "split": split,
                "metric": name,
                "value": count,
                "denominator": len(rows),
                "rate": count / len(rows) if rows else None,
                "source": "frame_compatible_label",
            }
        )
    return result


def find_field(row: dict[str, Any], aliases: tuple[str, ...]) -> tuple[bool, Any]:
    for field in aliases:
        if field in row:
            return True, row[field]
    return False, None


def contamination_audit(
    rows: list[dict[str, Any]], data_path: Path, authoritative_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_data = data_path.resolve()
    normalized_authoritative = authoritative_path.resolve()
    source_is_authoritative = normalized_data == normalized_authoritative
    no_time_swap = all(row.get("intervention_type") != "time_swap" for row in rows)
    schema_complete = all(REQUIRED_SCHEMA <= set(row) for row in rows)
    audits: list[dict[str, Any]] = []

    def add(criterion: str, available: bool, passed: bool, evidence: str) -> None:
        audits.append(
            {
                "criterion": criterion,
                "required": True,
                "authoritative_metadata_available": available,
                "passed": passed if available else False,
                "eligible_row_count": sum(
                    1
                    for row in rows
                    if row.get("frame_compatible_label") == 1
                )
                if available and passed
                else 0,
                "total_row_count": len(rows),
                "evidence": evidence,
                "fail_closed": True,
            }
        )

    add(
        "frame_label_compatible",
        all("frame_compatible_label" in row for row in rows),
        all(row.get("frame_compatible_label") in (0, 1) for row in rows),
        "row-level frame_compatible_label is present; loss eligibility additionally requires value 1",
    )
    for criterion, aliases in INTEGRITY_FIELDS.items():
        present_values = [find_field(row, aliases) for row in rows]
        available = bool(rows) and all(present for present, _ in present_values)
        passed = available and all(value is True for _, value in present_values)
        add(
            criterion,
            available,
            passed,
            f"accepted aliases={aliases}; missing metadata is not inferred from text or family",
        )
    add("time_swap_absent", True, no_time_swap, "intervention_type exact comparison")
    add(
        "authoritative_main_dataset",
        True,
        source_is_authoritative,
        f"resolved input equals {normalized_authoritative}",
    )
    add(
        "not_stage34_35_external_substitute",
        True,
        source_is_authoritative,
        "authoritative path identity; no external dataset inference",
    )
    add("required_schema_present", True, schema_complete, "required standard JSONL fields")

    full_metadata = all(
        row["authoritative_metadata_available"] and row["passed"]
        for row in audits
    )
    positive_count = sum(row.get("frame_compatible_label") == 1 for row in rows)
    fully_eligible = 0
    if full_metadata:
        for row in rows:
            integrity_true = all(find_field(row, aliases)[1] is True for aliases in INTEGRITY_FIELDS.values())
            if row.get("frame_compatible_label") == 1 and integrity_true:
                fully_eligible += 1
    summary = {
        "required_criteria_count": len(audits),
        "criteria_with_authoritative_metadata": sum(
            bool(row["authoritative_metadata_available"]) for row in audits
        ),
        "criteria_passed": sum(bool(row["passed"]) for row in audits),
        "compatible_positive_rows": positive_count,
        "fully_eligible_rows": fully_eligible,
        "complete_mask_constructible": full_metadata,
        "missing_integrity_fields": [
            name
            for name, aliases in INTEGRITY_FIELDS.items()
            if not all(find_field(row, aliases)[0] for row in rows)
        ],
        "policy": "fail_closed",
    }
    return audits, summary


def current_objective_rows(source_checks: dict[str, bool]) -> list[dict[str, Any]]:
    facts = [
        ("native_frame_loss", "F.binary_cross_entropy_with_logits(frame_logit, frame_compatible_labels)", "scripts/train_controlled_v5.py:465-468", True),
        ("positive_class_weight", "none", "no pos_weight in native frame call", True),
        ("reduction", "row mean (PyTorch default)", "no reduction override", True),
        ("native_frame_loss_weight", "1.0 implicit", "equal unweighted sum in controlled_losses", True),
        ("total_native_objective", "final CE + frame BCE + predicate BCE + sufficiency BCE + polarity CE", "scripts/train_controlled_v5.py:493", True),
        ("frame_classifier_bias", "trainable nn.Linear bias", "src/contramamba/heads/frame_gate.py:32", True),
        ("frame_probability", "sigmoid(frame_logit)", "src/contramamba/heads/frame_gate.py:69", True),
        ("checkpoint_selection", "maximize clean-dev final_macro_f1 by default", "trainer select_metric", True),
        ("frame_recall_directly_selected", "false", "not a select_metric choice", True),
        ("frame_margin_directly_selected", "false", "not a select_metric choice", True),
    ]
    rows = []
    for item, value, evidence, expected in facts:
        rows.append(
            {
                "item": item,
                "value": value,
                "evidence": evidence,
                "static_check_passed": source_checks.get(item, expected),
                "interpretation": "static_source_fact",
            }
        )
    return rows


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate": "A_positive_class_reweighting",
            "formula": "BCEWithLogits(pos_weight=alpha)",
            "scientific_target": "globally amplify compatible-positive BCE",
            "nonredundant_target": "low; changes scale but adds no margin",
            "expected_gradient_support": "all positive rows",
            "known_closed_path_overlap": "native BCE",
            "new_failure_risk": "global incompatible false positives and calibration shift",
            "requires_integrity_mask": False,
            "requires_teacher_or_counterpart": False,
            "fixed_without_tuning": False,
            "status": "not_selected",
        },
        {
            "candidate": "B_compatible_positive_absolute_margin_hinge",
            "formula": "mean_E relu(target_margin - frame_logit)",
            "scientific_target": "absolute clean compatible-positive frame-logit floor",
            "nonredundant_target": "high; absolute floor absent from BCE and Stage177",
            "expected_gradient_support": "eligible positive rows below margin",
            "known_closed_path_overlap": "same head as native BCE; not Stage177 ordering or Stage175 final SUPPORT anchor",
            "new_failure_risk": "false positives if mask/margin/weight are wrong",
            "requires_integrity_mask": True,
            "requires_teacher_or_counterpart": False,
            "fixed_without_tuning": False,
            "status": "contingent_preferred_after_integrity_gate",
        },
        {
            "candidate": "C_teacher_positive_logit_preservation",
            "formula": "mean_E relu(detached_reference - tolerance - frame_logit)",
            "scientific_target": "relative positive-logit preservation",
            "nonredundant_target": "medium; reference-relative rather than absolute",
            "expected_gradient_support": "eligible positives below reference tolerance",
            "known_closed_path_overlap": "Stage175 detached-reference preservation family",
            "new_failure_risk": "teacher drift/error preservation and provenance complexity",
            "requires_integrity_mask": True,
            "requires_teacher_or_counterpart": True,
            "fixed_without_tuning": False,
            "status": "not_selected",
        },
        {
            "candidate": "D_frame_classifier_bias_constraint",
            "formula": "constraint or penalty on frame_classifier.bias",
            "scientific_target": "global readout offset",
            "nonredundant_target": "unjustified; bias mechanism unavailable",
            "expected_gradient_support": "global through shared scalar bias",
            "known_closed_path_overlap": "calibration-like global shift",
            "new_failure_risk": "negative-class degradation",
            "requires_integrity_mask": False,
            "requires_teacher_or_counterpart": False,
            "fixed_without_tuning": False,
            "status": "rejected",
        },
        {
            "candidate": "E_false_negative_aware_checkpoint_selection",
            "formula": "predeclared clean-dev selection metric including positive recall",
            "scientific_target": "retain epochs with compatible-positive recall",
            "nonredundant_target": "selection rather than objective",
            "expected_gradient_support": "none; checkpoint selection only",
            "known_closed_path_overlap": "existing clean-dev selection infrastructure",
            "new_failure_risk": "selection leakage and incompatible precision loss",
            "requires_integrity_mask": False,
            "requires_teacher_or_counterpart": False,
            "fixed_without_tuning": False,
            "status": "not_selected",
        },
        {
            "candidate": "F_family_conditioned_weighting",
            "formula": "family-specific positive weights",
            "scientific_target": "selected family emphasis",
            "nonredundant_target": "low; hard-subset family association is not causal",
            "expected_gradient_support": "chosen family positives",
            "known_closed_path_overlap": "weighted native BCE",
            "new_failure_risk": "shortcut learning and 14-row hard-subset overfit",
            "requires_integrity_mask": True,
            "requires_teacher_or_counterpart": False,
            "fixed_without_tuning": False,
            "status": "rejected",
        },
    ]


def gradient_rows() -> list[dict[str, Any]]:
    return [
        {
            "loss_name": "native_positive_BCE",
            "activation_condition": "y=1",
            "dL_d_frame_logit": "sigmoid(z)-1",
            "gradient_near_zero": "-0.5",
            "gradient_strongly_negative": "approaches -1",
            "gradient_already_positive": "negative and approaches 0",
            "absolute_margin_enforced": False,
            "negative_rows_affected": True,
            "counterpart_or_teacher_required": False,
        },
        {
            "loss_name": "positive_reweighted_BCE",
            "activation_condition": "y=1",
            "dL_d_frame_logit": "alpha*(sigmoid(z)-1)",
            "gradient_near_zero": "-alpha/2",
            "gradient_strongly_negative": "approaches -alpha",
            "gradient_already_positive": "negative and approaches 0",
            "absolute_margin_enforced": False,
            "negative_rows_affected": "native negative BCE remains; no alpha multiplier",
            "counterpart_or_teacher_required": False,
        },
        {
            "loss_name": "clean_positive_absolute_margin_hinge",
            "activation_condition": "eligible y=1 and z<target_margin",
            "dL_d_frame_logit": "-1 if active; 0 if inactive",
            "gradient_near_zero": "-1 when target_margin>0",
            "gradient_strongly_negative": "-1",
            "gradient_already_positive": "0 once z>=target_margin",
            "absolute_margin_enforced": True,
            "negative_rows_affected": False,
            "counterpart_or_teacher_required": False,
        },
        {
            "loss_name": "teacher_positive_preservation_hinge",
            "activation_condition": "eligible y=1 and z<detached_reference-tolerance",
            "dL_d_frame_logit": "-1 if active; 0 if inactive",
            "gradient_near_zero": "reference-dependent",
            "gradient_strongly_negative": "-1 if active",
            "gradient_already_positive": "reference-dependent",
            "absolute_margin_enforced": False,
            "negative_rows_affected": False,
            "counterpart_or_teacher_required": True,
        },
    ]


def redundancy_rows() -> list[dict[str, Any]]:
    return [
        {"candidate": "A_positive_class_reweighting", "existing_objective": "native frame BCE", "overlap": "high", "nonredundant_scientific_target": "none; only positive scale", "known_result": "balanced frame-label topology", "conclusion": "not justified"},
        {"candidate": "B_absolute_margin_hinge", "existing_objective": "native frame BCE", "overlap": "medium", "nonredundant_scientific_target": "explicit absolute positive floor", "known_result": "BCE has no hard margin", "conclusion": "scientifically distinct"},
        {"candidate": "B_absolute_margin_hinge", "existing_objective": "Stage177 pairwise softplus", "overlap": "low", "nonredundant_scientific_target": "absolute z floor versus within-pair ordering", "known_result": "Stage177 margin_used=false and final predictions changed=0", "conclusion": "nonredundant"},
        {"candidate": "B_absolute_margin_hinge", "existing_objective": "Stage175 support anchor", "overlap": "low", "nonredundant_scientific_target": "native frame head versus final SUPPORT margin", "known_result": "Stage175 no clean benefit and reference-relative", "conclusion": "nonredundant"},
        {"candidate": "C_teacher_preservation", "existing_objective": "Stage175 support anchor", "overlap": "high", "nonredundant_scientific_target": "different head but same detached-reference family", "known_result": "Stage175 path closed", "conclusion": "avoid reopening without new evidence"},
        {"candidate": "D_bias_constraint", "existing_objective": "calibration/threshold family", "overlap": "medium", "nonredundant_scientific_target": "none established", "known_result": "bias decomposition unavailable", "conclusion": "rejected"},
        {"candidate": "E_checkpoint_selection", "existing_objective": "clean-dev checkpoint selection", "overlap": "low", "nonredundant_scientific_target": "positive recall retention", "known_result": "current metric is final macro-F1", "conclusion": "possible later, leakage contract required"},
        {"candidate": "F_family_weighting", "existing_objective": "weighted BCE", "overlap": "medium", "nonredundant_scientific_target": "family emphasis only", "known_result": "14 selected hard rows are not causal", "conclusion": "rejected"},
    ]


def risk_rows() -> list[dict[str, Any]]:
    dimensions = [
        "scientific_fit_to_stage182b", "redundancy_with_current_bce", "redundancy_with_stage175",
        "redundancy_with_stage177", "contamination_sensitivity", "false_positive_risk",
        "calibration_like_behavior_risk", "implementation_complexity", "checkpoint_dependency",
        "external_data_leakage_risk", "training_cost", "interpretability",
        "fixed_hyperparameter_feasibility",
    ]
    ratings = {
        "A_positive_class_reweighting": ["medium", "high", "low", "medium", "medium", "high", "high", "low", "low", "low", "low", "high", "low"],
        "B_absolute_margin_hinge": ["high", "medium", "low", "low", "high", "medium", "medium", "low", "low", "low", "low", "high", "medium"],
        "C_teacher_preservation": ["medium", "medium", "high", "low", "high", "medium", "medium", "high", "high", "medium", "high", "medium", "low"],
        "D_bias_constraint": ["low", "medium", "low", "low", "low", "high", "high", "medium", "low", "low", "low", "medium", "low"],
        "E_checkpoint_positive_recall": ["medium", "low", "low", "low", "medium", "high", "medium", "medium", "high", "high", "low", "high", "medium"],
        "F_family_conditioned_weighting": ["low", "medium", "low", "medium", "high", "high", "medium", "medium", "low", "medium", "low", "medium", "low"],
    }
    basis = {
        "A_positive_class_reweighting": "balanced topology; global shift without margin",
        "B_absolute_margin_hinge": "direct target but mask unavailable and fixed values unset",
        "C_teacher_preservation": "reference/checkpoint provenance and Stage175 overlap",
        "D_bias_constraint": "bias-specific evidence unavailable",
        "E_checkpoint_positive_recall": "clean-dev selection trade-off and leakage contract",
        "F_family_conditioned_weighting": "selected 14-row family mix is not causal",
    }
    rows = []
    for candidate, values in ratings.items():
        for dimension, rating in zip(dimensions, values):
            rows.append({"candidate": candidate, "dimension": dimension, "rating": rating, "basis": basis[candidate]})
    return rows


def source_audit(repo_root: Path) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    sources = {
        "trainer": repo_root / "scripts/train_controlled_v6b_minimal.py",
        "v5_trainer": repo_root / "scripts/train_controlled_v5.py",
        "model": repo_root / "src/contramamba/modeling_v6b_minimal.py",
        "frame_gate": repo_root / "src/contramamba/heads/frame_gate.py",
        "stage175": repo_root / "scripts/stage175b_support_anchor.py",
        "stage177": repo_root / "scripts/stage177c_frame_pairwise.py",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}
    checks = {
        "native_frame_loss": "F.binary_cross_entropy_with_logits(\n        output[\"frame_logit\"]" in text["v5_trainer"],
        "positive_class_weight": "output[\"frame_logit\"],\n        inputs[\"frame_compatible_labels\"],\n    )" in text["v5_trainer"],
        "reduction": "output[\"frame_logit\"],\n        inputs[\"frame_compatible_labels\"],\n    )" in text["v5_trainer"],
        "native_frame_loss_weight": "total = label_loss + frame_loss + predicate_loss + sufficiency_loss + polarity_loss" in text["v5_trainer"],
        "total_native_objective": "total = label_loss + frame_loss + predicate_loss + sufficiency_loss + polarity_loss" in text["v5_trainer"],
        "frame_classifier_bias": "self.frame_classifier = nn.Linear(frame_size, 1)" in text["frame_gate"],
        "frame_probability": '"frame_prob": torch.sigmoid(frame_logit)' in text["frame_gate"],
        "checkpoint_selection": 'select_metric="final_macro_f1"' in text["trainer"],
        "frame_recall_directly_selected": "choices=(\"final_macro_f1\", \"final_accuracy\")" in text["v5_trainer"],
        "frame_margin_directly_selected": "choices=(\"final_macro_f1\", \"final_accuracy\")" in text["v5_trainer"],
    }
    files = [
        {"path": str(path.relative_to(repo_root)), "sha256": sha256(path), "exists": True}
        for path in sources.values()
    ]
    return checks, files


def markdown_report(report: dict[str, Any]) -> str:
    topology = report["train_label_topology"]
    contamination = report["contamination_eligibility"]
    return f"""# Stage183-A positive-preservation design report

## Decision

`{report['decision']}`

Authorized next route: `{report['stage184_gate']['authorized_next_stage']}`.

The compatible-positive absolute-margin hinge is the contingent best scientific fit, but implementation is not authorized because the current main JSONL cannot construct a complete contamination-safe clean-compatible mask. No nonzero target margin or weight is selected.

## Current objective

The native frame loss is row-mean `BCEWithLogits(frame_logit, frame_compatible_label)` with no `pos_weight`; it enters the five-term native objective with implicit weight 1.0. `frame_prob` is `sigmoid(frame_logit)`, and the linear frame classifier has a trainable bias. Default checkpoint selection maximizes clean-dev final macro-F1, not frame-positive recall or margin.

## Label topology

- Full rows/pairs: {topology['all_row_count']} / {topology['all_pair_count']}
- Train rows/pairs: {topology['train_row_count']} / {topology['train_pair_count']}
- Train compatible/incompatible: {topology['train_compatible_count']} / {topology['train_incompatible_count']}
- Dev rows/pairs: {topology['dev_row_count']} / {topology['dev_pair_count']}

The frozen pair topology is balanced, so positive-class reweighting has no imbalance-based justification.

## Integrity gate

Complete mask constructible: `{str(contamination['complete_mask_constructible']).lower()}`.
Missing authoritative integrity fields: `{', '.join(contamination['missing_integrity_fields'])}`.

Stage182-A showed that generator equality is not cleanliness and found 22 deterministic contaminated review items. Missing grammar, contract, polarity-contamination, or canonical-valid metadata is fail-closed; it may not be inferred from family names or text heuristics.

## Candidate selection

Candidate B adds an absolute compatible-positive logit floor that native BCE and Stage177 pairwise ordering do not guarantee, and it targets a different head from Stage175's final SUPPORT anchor. It requires no teacher or counterpart and adds no direct negative-row loss. It remains contingent until the integrity mask exists and fixed no-sweep hyperparameters are justified.

Candidates A, C, D, E, and F are not selected: respectively they lack an imbalance rationale, reopen reference preservation, lack bias evidence, introduce a selection/leakage trade-off, or overfit noncausal family composition.

## Safety

Static audit only. No model/Torch execution, checkpoint load, forward, training, smoke, dataset mutation, relabeling, threshold fitting, calibration, external evaluation, `time_swap`, multi-seed run, or hyperparameter sweep is authorized.
"""


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve(repo_root, args.data).resolve()
    authoritative_path = (repo_root / AUTHORITATIVE_DATA).resolve()

    required_artifacts = [
        repo_root / "reports/stage182b_compatible_positive_margin_collapse_closure.json",
        repo_root / "reports/stage182a_data_contamination_clean_failure_set_closure.json",
        repo_root / "reports/stage175c_support_anchor_path_closure.json",
        repo_root / "reports/stage177e_frame_pairwise_redundancy_closure.json",
    ]
    missing = [str(path) for path in [data_path, *required_artifacts] if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required Stage183-A inputs: " + ", ".join(missing))

    closure182b = read_json(required_artifacts[0])
    closure182a = read_json(required_artifacts[1])
    closure175 = read_json(required_artifacts[2])
    closure177 = read_json(required_artifacts[3])
    rows = read_jsonl(data_path)
    train_rows, dev_rows = split_by_pair_id(rows, args.dev_ratio, args.split_seed)
    source_checks, source_files = source_audit(repo_root)
    objective = current_objective_rows(source_checks)
    topology_csv = (
        topology_rows(rows, "all")
        + topology_rows(train_rows, "train")
        + topology_rows(dev_rows, "dev")
    )
    contamination_csv, contamination_summary = contamination_audit(
        train_rows, data_path, authoritative_path
    )
    candidates = candidate_rows()
    gradients = gradient_rows()
    redundancy = redundancy_rows()
    risks = risk_rows()

    label_counts = Counter(row.get("frame_compatible_label") for row in train_rows)
    topology_summary = {
        "data": str(data_path),
        "split_seed": args.split_seed,
        "dev_ratio": args.dev_ratio,
        "all_row_count": len(rows),
        "all_pair_count": len({row["pair_id"] for row in rows}),
        "train_row_count": len(train_rows),
        "train_pair_count": len({row["pair_id"] for row in train_rows}),
        "dev_row_count": len(dev_rows),
        "dev_pair_count": len({row["pair_id"] for row in dev_rows}),
        "train_compatible_count": label_counts.get(1, 0),
        "train_incompatible_count": label_counts.get(0, 0),
        "targets_inferred_from_family": False,
    }
    shared_paths = [
        {"loss": "native_frame_BCE", "direct_target": "frame_classifier and frame_pair_repr", "shared_upstream": "FrameGate project/pair_projector and encoder token states", "causal_conflict_established": False},
        {"loss": "final_classifier_CE", "direct_target": 'output["logits"]', "shared_upstream": "frame representation/probability and encoder", "causal_conflict_established": False},
        {"loss": "predicate_sufficiency_polarity", "direct_target": "respective heads", "shared_upstream": "encoder token states; decision representation paths", "causal_conflict_established": False},
    ]
    selection_evidence = [
        {"evidence": "Stage182-B compatible false negatives", "value": "13/14", "supports": "positive absolute target", "opposes": "bias/family causal claim", "source": "Stage182-B closure"},
        {"evidence": "paired frame-logit median and CI", "value": "-0.555523656308651; CI [-0.7878567576408386,-0.3871966600418091]", "supports": "systematic scalar collapse signal", "opposes": "none", "source": "Stage182-B closure"},
        {"evidence": "centroid correct", "value": "candidate=1/14; control=1/14", "supports": "none", "opposes": "representation/readout causal route", "source": "Stage182-B closure"},
        {"evidence": "train frame topology", "value": f"positive={label_counts.get(1,0)} negative={label_counts.get(0,0)}", "supports": "no global imbalance remedy", "opposes": "Candidate A", "source": "authoritative data labels"},
        {"evidence": "Stage177 objective", "value": "relative pair ordering; no absolute margin", "supports": "Candidate B nonredundancy", "opposes": "reopening pairwise path", "source": closure177.get("decision")},
        {"evidence": "Stage175 objective", "value": "final SUPPORT detached-reference anchor; no clean benefit", "supports": "different-head distinction", "opposes": "Candidate C", "source": closure175.get("decision")},
        {"evidence": "train integrity metadata", "value": f"missing={contamination_summary['missing_integrity_fields']}", "supports": "integrity mask first", "opposes": "loss implementation now", "source": "authoritative JSONL static audit"},
    ]
    gate_rows = [
        {"gate": "complete_contamination_safe_mask", "required": True, "passed": contamination_summary["complete_mask_constructible"], "authorization": "required before loss smoke", "next_stage": NEXT_STAGE},
        {"gate": "fixed_target_margin_justified_without_sweep", "required": True, "passed": False, "authorization": "no value selected", "next_stage": NEXT_STAGE},
        {"gate": "fixed_nonzero_weight_justified_without_sweep", "required": True, "passed": False, "authorization": "default remains 0", "next_stage": NEXT_STAGE},
        {"gate": "training_authorized", "required": True, "passed": False, "authorization": "forbidden", "next_stage": NEXT_STAGE},
    ]

    report: dict[str, Any] = {
        "stage": STAGE,
        "decision": DECISION,
        "scope": {"static_design_audit_only": True, "implementation": False, "training": False, "model_execution": False},
        "input_validation": {
            "status": "passed",
            "data_sha256": sha256(data_path),
            "source_files": source_files,
            "artifact_sha256": {str(path.relative_to(repo_root)): sha256(path) for path in required_artifacts},
            "stage182b_decision": closure182b.get("decision"),
            "stage182a_decision": closure182a.get("decision"),
            "source_static_checks": source_checks,
        },
        "current_frame_objective": {"formula": "mean BCEWithLogits(frame_logit, frame_compatible_label)", "rows": objective},
        "train_label_topology": topology_summary,
        "shared_gradient_paths": shared_paths,
        "contamination_eligibility": {**contamination_summary, "criteria": contamination_csv},
        "candidate_interventions": candidates,
        "gradient_analysis": {"variable": "z=frame_logit", "outer_weights_excluded": True, "rows": gradients},
        "redundancy_analysis": redundancy,
        "risk_analysis": {"scale": ["low", "medium", "high"], "scores_fitted": False, "rows": risks},
        "selected_design": {
            "primary_recommendation": DECISION,
            "contingent_post_integrity_candidate": "B_compatible_positive_absolute_margin_hinge",
            "implementation_authorized": False,
            "target_margin": None,
            "weight": 0.0,
            "mode_default": "off",
            "reason": "contamination-safe eligibility and fixed no-sweep hyperparameters are unavailable",
        },
        "stage184_gate": {
            "authorized_next_stage": NEXT_STAGE,
            "positive_margin_hinge_smoke_authorized": False,
            "training_pilot_authorized": False,
            "rows": gate_rows,
        },
        "limitations": [
            "Static code topology does not establish runtime gradient dominance.",
            "The Stage182-B 14-pair signal is not a population causal estimate.",
            "Bias-specific and polarity-conditioned causal mechanisms are not established.",
            "No fixed nonzero margin or weight is justified.",
        ],
        "safety_policy": {
            "no_model_execution": True, "no_torch_execution": True, "no_checkpoint_load": True,
            "no_forward": True, "no_training": True, "no_smoke": True, "no_dataset_modification": True,
            "no_generator_modification": True, "no_relabeling": True, "no_threshold_fitting": True,
            "no_calibration": True, "no_external_evaluation": True, "no_time_swap": True,
            "no_multi_seed": True, "no_hyperparameter_sweep": True,
        },
    }

    write_json(output_dir / "stage183a_positive_preservation_design_report.json", report)
    (output_dir / "stage183a_positive_preservation_design_report.md").write_text(markdown_report(report), encoding="utf-8")
    write_csv(output_dir / "stage183a_current_frame_objective_audit.csv", ["item", "value", "evidence", "static_check_passed", "interpretation"], objective)
    write_csv(output_dir / "stage183a_train_label_topology.csv", ["split", "metric", "value", "denominator", "rate", "source"], topology_csv)
    write_csv(output_dir / "stage183a_candidate_intervention_matrix.csv", ["candidate", "formula", "scientific_target", "nonredundant_target", "expected_gradient_support", "known_closed_path_overlap", "new_failure_risk", "requires_integrity_mask", "requires_teacher_or_counterpart", "fixed_without_tuning", "status"], candidates)
    write_csv(output_dir / "stage183a_gradient_comparison.csv", ["loss_name", "activation_condition", "dL_d_frame_logit", "gradient_near_zero", "gradient_strongly_negative", "gradient_already_positive", "absolute_margin_enforced", "negative_rows_affected", "counterpart_or_teacher_required"], gradients)
    write_csv(output_dir / "stage183a_redundancy_audit.csv", ["candidate", "existing_objective", "overlap", "nonredundant_scientific_target", "known_result", "conclusion"], redundancy)
    write_csv(output_dir / "stage183a_contamination_eligibility_audit.csv", ["criterion", "required", "authoritative_metadata_available", "passed", "eligible_row_count", "total_row_count", "evidence", "fail_closed"], contamination_csv)
    write_csv(output_dir / "stage183a_risk_matrix.csv", ["candidate", "dimension", "rating", "basis"], risks)
    write_csv(output_dir / "stage183a_selection_evidence.csv", ["evidence", "value", "supports", "opposes", "source"], selection_evidence)
    write_csv(output_dir / "stage183a_stage184_gate.csv", ["gate", "required", "passed", "authorization", "next_stage"], gate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
