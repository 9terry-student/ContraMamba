#!/usr/bin/env python3
"""Audit the Stage186-A fixed compatible-positive absolute-margin specification.

This script is stdlib-only.  It performs static source inspection and artifact
validation; it never imports the trainer, model, torch, or checkpoint code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage186-A"
READY = "STAGE186A_FIXED_NO_SWEEP_COMPATIBLE_POSITIVE_MARGIN_SPEC_READY"
BLOCKED = "STAGE186A_COMPATIBLE_POSITIVE_MARGIN_SPEC_BLOCKED"
NEXT = "STAGE187_COMPATIBLE_POSITIVE_MARGIN_DEFAULT_OFF_IMPLEMENTATION"
STAGE185_DECISION = "STAGE185A_INTEGRITY_SIDECAR_BUILT_AND_POSITIVE_ELIGIBILITY_MATERIALIZED"
STAGE185_NEXT = "STAGE186_COMPATIBLE_POSITIVE_MARGIN_FIXED_SPEC_AUDIT"
STAGE183_DECISION = "STAGE183A_CONTROLLED_TRAIN_INTEGRITY_MASK_REQUIRED_FIRST"
STAGE183_CANDIDATE = "B_compatible_positive_absolute_margin_hinge"
STAGE182B_DECISION = "STAGE182B_COMPATIBLE_POSITIVE_MARGIN_COLLAPSE_SIGNAL"
DEFAULT_DATASET_SHA256 = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
DEFAULT_SIDECAR_SEMANTIC_SHA256 = "5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc"
FIXED_MARGIN_LOGIT = 0.0
FIXED_MARGIN_WEIGHT = 0.05
DEFAULT_MARGIN_WEIGHT = 0.0
LOGIT_GRID = (-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0)

STAGE185_FILES = (
    "stage185a_controlled_train_integrity_sidecar_report.json",
    "stage185a_controlled_train_integrity_sidecar.jsonl",
    "stage185a_controlled_train_integrity_sidecar.csv",
    "stage185a_provenance.json",
    "stage185a_positive_eligibility_summary.csv",
    "stage185a_family_coverage.csv",
    "stage185a_pair_integrity_audit.csv",
    "stage185a_stage182_overlap_regression.csv",
    "stage185a_stage186_gate.csv",
)
OUTPUTS = (
    "stage186a_compatible_positive_margin_fixed_spec_report.json",
    "stage186a_compatible_positive_margin_fixed_spec_report.md",
    "stage186a_input_identity_audit.csv",
    "stage186a_eligible_cohort_audit.csv",
    "stage186a_pair_family_balance.csv",
    "stage186a_fixed_hyperparameter_spec.csv",
    "stage186a_gradient_scale_audit.csv",
    "stage186a_sidecar_consumption_contract.csv",
    "stage186a_objective_integration_contract.csv",
    "stage186a_default_off_contract.csv",
    "stage186a_checkpoint_selection_contract.csv",
    "stage186a_prior_intervention_nonredundancy.csv",
    "stage186a_stage187_gate.csv",
)
CSV_HEADERS = {
    "stage186a_input_identity_audit.csv": ["check", "observed", "expected", "passed", "evidence"],
    "stage186a_eligible_cohort_audit.csv": ["check", "observed", "expected", "passed", "fail_closed_behavior"],
    "stage186a_pair_family_balance.csv": ["dimension", "identifier", "eligible_rows", "expected_rows", "passed", "interpretation"],
    "stage186a_fixed_hyperparameter_spec.csv": ["parameter", "value", "default_value", "fixed_no_sweep", "rationale", "forbidden_adaptation"],
    "stage186a_gradient_scale_audit.csv": ["frame_logit", "sigmoid_probability", "positive_bce_loss", "positive_bce_gradient_magnitude", "unweighted_hinge_loss", "weighted_hinge_loss", "weighted_hinge_gradient_magnitude", "weighted_hinge_to_bce_gradient_ratio", "hinge_active", "boundary_convention"],
    "stage186a_sidecar_consumption_contract.csv": ["requirement", "authoritative_field", "validation", "failure_behavior", "eligibility_inference_forbidden"],
    "stage186a_objective_integration_contract.csv": ["component", "current_source", "current_semantics", "stage187_contract", "changed", "evidence"],
    "stage186a_default_off_contract.csv": ["setting", "recommended_name", "default", "fixed_intervention_value", "activation_requirement", "ambiguous_alias_forbidden"],
    "stage186a_checkpoint_selection_contract.csv": ["property", "current_value", "stage187_required_value", "changed", "evidence"],
    "stage186a_prior_intervention_nonredundancy.csv": ["intervention", "target", "comparison_topology", "output_space", "distinct_from_stage186", "evidence"],
    "stage186a_stage187_gate.csv": ["gate", "observed", "required", "passed", "authorized_next_stage"],
}


class AuditBlocked(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage185a-dir", type=Path, required=True)
    parser.add_argument("--stage183a-dir", type=Path, required=True)
    parser.add_argument("--stage182b-dir", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-sha256", default=DEFAULT_DATASET_SHA256)
    parser.add_argument("--expected-sidecar-semantic-sha256", default=DEFAULT_SIDECAR_SEMANTIC_SHA256)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_dataset_sha256):
        parser.error("--expected-dataset-sha256 must be a lowercase SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_sidecar_semantic_sha256):
        parser.error("--expected-sidecar-semantic-sha256 must be a lowercase SHA-256")
    return args


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def read_csv(path: Path, fields: set[str] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = set(reader.fieldnames or [])
        if fields:
            require(fields <= actual, f"{path}: missing columns {sorted(fields - actual)}")
        return [dict(row) for row in reader]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def cell(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "" if value is None else value


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: cell(row.get(field)) for field in fields})


def require_files(directory: Path, names: Iterable[str]) -> dict[str, Path]:
    result = {name: directory / name for name in names}
    missing = [str(path) for path in result.values() if not path.is_file()]
    require(not missing, "missing required inputs: " + ", ".join(missing))
    return result


def find_one(directory: Path, name: str) -> Path:
    direct = directory / name
    if direct.is_file():
        return direct
    matches = list(directory.rglob(name))
    require(len(matches) == 1, f"expected exactly one {name} under {directory}, found {len(matches)}")
    return matches[0]


def semantic_sha(rows: list[dict[str, Any]]) -> str:
    canonical = [
        {key: row[key] for key in sorted(row) if key != "created_at"}
        for row in rows
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_ids(rows: Iterable[dict[str, Any]], key: str) -> set[str]:
    values = [str(row.get(key, "")) for row in rows]
    require(all(values), f"missing {key}")
    require(len(values) == len(set(values)), f"duplicate {key}")
    return set(values)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def evidence(text: str, pattern: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    require(match is not None, f"trainer static evidence missing: {label}")
    line = text.count("\n", 0, match.start()) + 1
    snippet = text[match.start():match.end()].replace("\n", " ").strip()
    return f"line {line}: {snippet[:180]}"


def optional_evidence(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        return None
    line = text.count("\n", 0, match.start()) + 1
    snippet = text[match.start():match.end()].replace("\n", " ").strip()
    return f"line {line}: {snippet[:180]}"


def stage183_frame_bce_crosscheck(report: dict[str, Any]) -> dict[str, Any]:
    require(report.get("decision") == STAGE183_DECISION, "Stage183-A decision mismatch in delegated frame-BCE cross-check")
    objective = report.get("current_frame_objective") or {}
    rows = objective.get("rows") or []
    by_item = {row.get("item"): row for row in rows if isinstance(row, dict)}

    def fact(item: str, required_text: str) -> dict[str, Any]:
        row = by_item.get(item) or {}
        value = str(row.get("value", ""))
        passed = row.get("static_check_passed") is True and required_text.lower() in value.lower()
        require(passed, f"Stage183-A native frame-loss evidence mismatch: {item}")
        return row

    native = fact("native_frame_loss", "binary_cross_entropy_with_logits")
    pos_weight = fact("positive_class_weight", "none")
    reduction = fact("reduction", "row mean")
    weight = fact("native_frame_loss_weight", "1.0")
    bias = fact("frame_classifier_bias", "trainable")
    formula = str(objective.get("formula", ""))
    gradient_variable = str((report.get("gradient_analysis") or {}).get("variable", ""))
    pre_sigmoid = "BCEWithLogits" in formula and "frame_logit" in formula and gradient_variable == "z=frame_logit"
    require(pre_sigmoid, "Stage183-A pre-sigmoid frame-logit evidence mismatch")
    require("pos_weight" not in str(native.get("value", "")), "Stage183-A native frame loss unexpectedly specifies pos_weight")
    return {
        "passed": True,
        "bce_with_logits": True,
        "row_mean": True,
        "pos_weight": None,
        "native_weight": 1.0,
        "pre_sigmoid_frame_logit": True,
        "frame_classifier_bias_trainable": True,
        "evidence": {
            "formula": formula,
            "native_frame_loss": native.get("evidence"),
            "positive_class_weight": pos_weight.get("evidence"),
            "reduction": reduction.get("evidence"),
            "native_frame_loss_weight": weight.get("evidence"),
            "frame_classifier_bias": bias.get("evidence"),
            "gradient_variable": gradient_variable,
        },
    }


def trainer_audit(path: Path, stage183_report: dict[str, Any]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    require("compatible_positive_margin" not in text and "compatible-positive-margin" not in text, "equivalent compatible-positive margin intervention already exists")
    frame_output = evidence(text, r'output\["frame_logit"\]', "native frame_logit")
    direct_frame_bce = optional_evidence(
        text, r'F\.binary_cross_entropy_with_logits\(\s*output\["frame_logit"\]'
    )
    final_ce = evidence(text, r'F\.cross_entropy\(\s*output\["logits"\]', "final CE")
    delegated_consumption = optional_evidence(
        text, r'losses\s*=\s*[A-Za-z0-9_\.]+\.controlled_losses\(\s*output\s*,'
    )
    delegated_total = optional_evidence(
        text, r'total_loss\s*=\s*losses\["total"\]\s*\+\s*active_intervention_loss'
    )
    weight_arg = optional_evidence(
        text, r'parser\.add_argument\(\s*"--v7-frame-loss-weight"'
    ) or optional_evidence(text, r'\bv7_frame_loss_weight\b')
    stage183_crosscheck = stage183_frame_bce_crosscheck(stage183_report)
    if direct_frame_bce is not None:
        detection_mode = "direct_trainer"
        frame_bce_evidence = direct_frame_bce
        frame_aux_wiring = delegated_total or optional_evidence(
            text, r'total\s*=\s*label_loss\s*\+\s*frame_loss\s*\+\s*predicate_loss\s*\+\s*sufficiency_loss\s*\+\s*polarity_loss'
        )
        require(frame_aux_wiring is not None, "trainer static evidence missing: direct frame auxiliary total-objective wiring")
    else:
        detection_mode = "delegated_model_forward_stage183_crosscheck"
        require(weight_arg is not None, "trainer static evidence missing: v7 frame loss weight argument")
        require(delegated_consumption is not None, "trainer static evidence missing: delegated model-native auxiliary-loss consumption")
        require(delegated_total is not None, "trainer static evidence missing: frame auxiliary total-objective wiring")
        frame_bce_evidence = "Stage183-A native frame BCE cross-check + delegated trainer auxiliary-loss consumption"
        frame_aux_wiring = delegated_total
    assembly = evidence(text, r'total_loss\s*=\s*losses\["total"\]\s*\+\s*active_intervention_loss', "trainer objective assembly")
    selection = (
        optional_evidence(text, r'select_metric:\s*str\s*=\s*"final_macro_f1"')
        or optional_evidence(text, r'parser\.add_argument\(\s*"--select-metric".{0,500}?default\s*=\s*"final_macro_f1"')
        or optional_evidence(text, r'select_metric\s*=\s*"final_macro_f1"')
    )
    require(selection is not None, "trainer static evidence missing: checkpoint default final_macro_f1")
    selection_use = evidence(text, r'score\s*=\s*float\(dev_metrics\[select_metric\]\)', "checkpoint scoring")
    naming = evidence(text, r'parser\.add_argument\("--[a-z0-9-]+"', "CLI naming convention")
    prior_flags = {
        "stage175_like_anchor_flags_present": "--lambda-frame-anchor" in text or "--lambda-logit-preserve" in text or "--stage175b-support-anchor" in text,
        "stage177_like_pairwise_flags_present": "--ranking-margin" in text or "--use-intervention-loss" in text or "--stage177c" in text,
    }
    return {
        "path": str(path), "static_text_only": True, "imported": False,
        "native_frame_output_key": "frame_logit", "frame_output_evidence": frame_output,
        "native_frame_bce_detection_mode": detection_mode,
        "direct_native_frame_bce_call_found": direct_frame_bce is not None,
        "stage183_native_frame_bce_crosscheck": stage183_crosscheck,
        "frame_logit_access_found": True,
        "frame_loss_weight_arg_found": weight_arg is not None,
        "frame_aux_objective_wiring_found": frame_aux_wiring is not None,
        "final_ce_output_logits_found": True,
        "checkpoint_selection_unchanged": True,
        "frame_bce": frame_bce_evidence, "native_frame_loss_weight": 1.0,
        "final_ce": final_ce,
        "existing_total": delegated_consumption or frame_aux_wiring,
        "assembly": assembly,
        "checkpoint_default": "final_macro_f1", "checkpoint_evidence": selection,
        "checkpoint_score_evidence": selection_use, "cli_naming_evidence": naming,
        "equivalent_absolute_compatible_positive_hinge_present": False,
        "evidence_lines": {
            "frame_logit_access": frame_output,
            "direct_native_frame_bce": direct_frame_bce,
            "frame_loss_weight_argument": weight_arg,
            "delegated_auxiliary_consumption": delegated_consumption,
            "frame_auxiliary_objective_wiring": frame_aux_wiring,
            "final_ce_output_logits": final_ce,
            "checkpoint_selection": selection,
            "checkpoint_score": selection_use,
        },
        "limitations": [
            "Delegated mode does not treat generic BCE calls as native frame-BCE evidence.",
            "Stage22 frame-violation, entitlement, temporal, and location-boundary BCE calls are excluded from native evidence.",
            "The delegated BCE semantic is established by Stage183-A facts plus trainer auxiliary-loss wiring; the trainer is not imported or executed.",
        ],
        **prior_flags,
    }

def gradient_rows() -> list[dict[str, Any]]:
    rows = []
    for z in LOGIT_GRID:
        probability = 1.0 / (1.0 + math.exp(-z))
        bce = max(0.0, -z) + math.log1p(math.exp(-abs(z)))
        bce_gradient = 1.0 - probability
        hinge = max(0.0, FIXED_MARGIN_LOGIT - z)
        if z < FIXED_MARGIN_LOGIT:
            weighted_gradient: float | None = FIXED_MARGIN_WEIGHT
            active = "true"
            boundary = "active z<0"
        elif z > FIXED_MARGIN_LOGIT:
            weighted_gradient = 0.0
            active = "false"
            boundary = "inactive z>0"
        else:
            weighted_gradient = None
            active = "boundary"
            boundary = "ReLU subgradient at z=0 is implementation-dependent; spec fixes only z<0 active and z>0 inactive"
        rows.append({
            "frame_logit": z, "sigmoid_probability": probability,
            "positive_bce_loss": bce, "positive_bce_gradient_magnitude": bce_gradient,
            "unweighted_hinge_loss": hinge, "weighted_hinge_loss": FIXED_MARGIN_WEIGHT * hinge,
            "weighted_hinge_gradient_magnitude": weighted_gradient,
            "weighted_hinge_to_bce_gradient_ratio": None if weighted_gradient is None else weighted_gradient / bce_gradient,
            "hinge_active": active, "boundary_convention": boundary,
        })
    return rows


def markdown(report: dict[str, Any]) -> str:
    cohort = report["eligible_cohort"]
    direction = report["stage182b_directional_evidence"]
    return f"""# Stage186-A compatible-positive margin fixed-spec report

## Decision

`{report['decision']}`

Authorized next: `{report['stage187_gate']['authorized_next_stage']}`. Loss implementation, checkpoint execution, and training remain unauthorized.

## Stage185 closure and sidecar identity

The authoritative 3,600-row sidecar exact-joins the source by unique row ID. Dataset SHA-256: `{report['stage185_closure']['dataset_sha256']}`. Semantic sidecar SHA-256: `{report['stage185_closure']['sidecar_semantic_sha256']}`. Stage182 overlap regression and 22/22 deterministic contamination recovery passed with zero blocked invariants.

## Native frame BCE detection

The trainer does not directly recompute native frame BCE as `F.binary_cross_entropy_with_logits(output["frame_logit"], ...)`. Detection mode is `{report['trainer_static_audit']['native_frame_bce_detection_mode']}`: the model-native auxiliary loss is consumed through the trainer's delegated loss path and wired into `losses["total"]`. Stage183-A static facts establish row-mean BCEWithLogits, no `pos_weight`, native weight 1.0, the pre-sigmoid frame logit, and a trainable frame-classifier bias. Generic BCE substrings—including Stage22 frame-violation, entitlement, temporal, and location-boundary losses—are not used as native frame-BCE evidence.

## Eligible cohort

Exactly {cohort['eligible_rows']} of {cohort['train_compatible_rows']} train-compatible rows are eligible ({cohort['eligible_rate']:.16f}). They cover {cohort['eligible_pairs']} pairs and {cohort['eligible_families']} families; each eligible pair contributes five rows and each family contributes 121. Families: {', '.join(cohort['family_names'])}.

Only 121 of 240 train pairs contribute, so this is pair-concentrated integrity-filtered coverage, not population-wide cleanliness. Exact five-family balance does not remove that limitation.

## Directional evidence and fixed specification

Stage182-B reports {direction['compatible_false_negatives']} compatible false negatives versus {direction['incompatible_false_positives']} incompatible false positives. Its candidate-minus-control direction and bootstrap interval are negative. This justifies intervention direction only; no Stage182-B statistic is fitted into a hyperparameter.

The fixed target is native pre-sigmoid `output["frame_logit"]` with margin `0.0`:

```text
L_margin = mean(relu(-frame_logit[eligible_mask]))
L_total = L_existing + 0.05 * L_margin
```

Zero is the sigmoid boundary, not a confidence-inflation target. The weight is fixed at 0.05 while native frame BCE remains weight 1.0. Near zero on the active side, the hinge gradient is 10% of positive BCE's magnitude. No sweep, calibration, schedule, family weight, or automatic rescaling is permitted.

## Normalization and zero-eligible batches

Normalization is eligible-row mean only. Current 5-per-pair topology makes row mean equal to equal-pair mean, but later topology changes require a new gate. An empty eligible mask returns graph-compatible scalar zero `frame_logit.sum() * 0.0`; division by zero, NaN, batch skipping, and optimizer-step changes are forbidden.

## Consumption and objective integration

Activation requires the authoritative eligibility boolean plus train/compatible/integrity/time/source checks, exact dataset and semantic SHA values, and an exact one-to-one row-ID join. Family names, reason-code text, Stage182 membership, predictions, probabilities, final labels, heuristics, and row order may not infer eligibility.

Static trainer inspection identified native `frame_logit`, row-mean frame BCE, final CE from `output["logits"]`, existing objective assembly, and clean-dev `final_macro_f1` selection. Stage187 may append the default-off term only; final CE, existing auxiliary weights, optimizer/scheduler, architecture, and checkpoint selection remain unchanged.

Stage175 targets a final-classifier SUPPORT anchor against a detached reference. Stage177 targets within-pair ordering. Stage186 targets an absolute native frame-head boundary on independently eligible rows, so it is nonredundant with both.

## Default-off contract and safety

Implementation defaults: `--compatible-positive-margin-weight 0.0` and `--compatible-positive-margin-logit 0.0`. The fixed intervention setting is weight 0.05 and logit 0.0. Sidecar path and expected semantic SHA are separate activation inputs. No trainer/model/loss modification, Torch import, checkpoint load, forward, training, smoke run, evaluation, fitting, sweep, or annotation occurred in this audit.
"""


def run(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    stage185_dir = resolve(root, args.stage185a_dir)
    stage183_dir = resolve(root, args.stage183a_dir)
    stage182_dir = resolve(root, args.stage182b_dir)
    trainer_path = resolve(root, args.trainer_source)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(trainer_path.is_file(), f"trainer source missing: {trainer_path}")
    files185 = require_files(stage185_dir, STAGE185_FILES)
    report185 = read_json(files185[STAGE185_FILES[0]])
    provenance = read_json(files185["stage185a_provenance.json"])
    report183 = read_json(find_one(stage183_dir, "stage183a_positive_preservation_design_report.json"))
    report182 = read_json(find_one(stage182_dir, "stage182b_clean_frame_failure_localization_report.json"))

    input_rows: list[dict[str, Any]] = []
    def check(name: str, observed: Any, expected: Any, source: str) -> None:
        passed = observed == expected
        input_rows.append({"check": name, "observed": observed, "expected": expected, "passed": passed, "evidence": source})
        require(passed, f"input identity mismatch: {name}: {observed!r} != {expected!r}")

    check("stage185_decision", report185.get("decision"), STAGE185_DECISION, str(files185[STAGE185_FILES[0]]))
    check("stage185_authorized_next", report185.get("stage186_gate", {}).get("authorized_next_stage"), STAGE185_NEXT, "stage186_gate")
    check("dataset_sha_report", report185.get("dataset_identity", {}).get("sha256"), args.expected_dataset_sha256, "Stage185 report")
    check("dataset_sha_provenance", provenance.get("source_dataset_sha256"), args.expected_dataset_sha256, "Stage185 provenance")
    check("semantic_sha_report", report185.get("sidecar_hashes", {}).get("semantic_sha256"), args.expected_sidecar_semantic_sha256, "Stage185 report")
    check("semantic_sha_provenance", provenance.get("sidecar_hashes", {}).get("semantic_sha256"), args.expected_sidecar_semantic_sha256, "Stage185 provenance")
    check("stage183_decision", report183.get("decision"), STAGE183_DECISION, "Stage183 report")
    selected = report183.get("selected_design", {})
    check("stage183_candidate", selected.get("contingent_post_integrity_candidate"), STAGE183_CANDIDATE, "Stage183 selected_design")
    check("stage183_margin_unset", selected.get("target_margin"), None, "Stage183 selected_design")
    check("stage183_implementation_unauthorized", selected.get("implementation_authorized"), False, "Stage183 selected_design")
    check("stage182b_decision", report182.get("decision"), STAGE182B_DECISION, "Stage182-B report")

    rows = read_jsonl(files185["stage185a_controlled_train_integrity_sidecar.jsonl"])
    csv_rows = read_csv(files185["stage185a_controlled_train_integrity_sidecar.csv"], {"row_id", "eligible_for_positive_margin"})
    json_ids = exact_ids(rows, "row_id")
    csv_ids = exact_ids(csv_rows, "row_id")
    check("sidecar_rows", len(rows), 3600, "sidecar JSONL")
    check("sidecar_csv_rows", len(csv_rows), 3600, "sidecar CSV")
    check("jsonl_csv_row_id_set", json_ids == csv_ids, True, "exact row-ID join over 3,600 unique IDs")
    actual_semantic_sha = semantic_sha(rows)
    check("recomputed_semantic_sha", actual_semantic_sha, args.expected_sidecar_semantic_sha256, "canonical JSONL rows excluding created_at")

    eligible = [row for row in rows if row.get("eligible_for_positive_margin") is True]
    train_compatible = [row for row in rows if row.get("split") == "train" and row.get("frame_compatible_label") == 1]
    unresolved_positive = [row for row in train_compatible if row.get("integrity_status") == "UNRESOLVED"]
    ineligible_positive = [row for row in train_compatible if row.get("integrity_status") == "INELIGIBLE"]
    pair_counts = Counter(str(row["pair_id"]) for row in eligible)
    family_counts = Counter(str(row["intervention_type"]) for row in eligible)
    invalid_eligible = [row for row in eligible if not (
        row.get("split") == "train" and row.get("frame_compatible_label") == 1
        and row.get("integrity_status") == "ELIGIBLE" and row.get("time_swap_status") == "PASS"
        and row.get("dataset_source_status") == "PASS"
    )]
    cohort_checks = (
        ("eligible_rows", len(eligible), 605),
        ("train_compatible_rows", len(train_compatible), 1440),
        ("eligible_pairs", len(pair_counts), 121),
        ("eligible_families", len(family_counts), 5),
        ("each_eligible_pair_exactly_five", sorted(set(pair_counts.values())), [5]),
        ("each_eligible_family_exactly_121", sorted(set(family_counts.values())), [121]),
        ("invalid_eligible_rows", len(invalid_eligible), 0),
        ("dev_eligible_rows", sum(row.get("split") == "dev" for row in eligible), 0),
        ("incompatible_eligible_rows", sum(row.get("frame_compatible_label") != 1 for row in eligible), 0),
        ("unresolved_eligible_rows", sum(row.get("integrity_status") == "UNRESOLVED" for row in eligible), 0),
        ("ineligible_eligible_rows", sum(row.get("integrity_status") == "INELIGIBLE" for row in eligible), 0),
        ("unresolved_train_compatible", len(unresolved_positive), 119),
        ("ineligible_train_compatible", len(ineligible_positive), 716),
    )
    cohort_rows = []
    for name, observed, expected in cohort_checks:
        passed = observed == expected
        cohort_rows.append({"check": name, "observed": observed, "expected": expected, "passed": passed, "fail_closed_behavior": "block Stage186 readiness"})
        require(passed, f"eligible cohort mismatch: {name}: {observed!r} != {expected!r}")

    gate185 = read_csv(files185["stage185a_stage186_gate.csv"], {"gate", "passed", "observed"})
    gate_by_name = {row["gate"]: row for row in gate185}
    require(gate_by_name.get("blocked_invariants", {}).get("observed") == "0", "Stage185 blocked invariant is nonzero")
    require(all(as_bool(row["passed"]) for row in gate185), "Stage185 gate contains failure")
    overlap = read_csv(files185["stage185a_stage182_overlap_regression.csv"], {"passed", "stage182_integrity_status", "stage185_contamination_codes"})
    require(len(overlap) == 78 and all(as_bool(row["passed"]) for row in overlap), "Stage182 overlap regression did not pass")
    contaminated = [row for row in overlap if row["stage182_integrity_status"] == "CONTAMINATED_CONSTRUCTION"]
    require(len(contaminated) == 22, "Stage182 contaminated overlap is not 22")

    native_direction = report182.get("native_error_direction", {})
    margin_signal = report182.get("compatible_positive_margin", {})
    compatible_fn = native_direction.get("compatible_false_negative_count")
    incompatible_fp = native_direction.get("incompatible_false_positive_count")
    margin_rows = margin_signal.get("rows") or []
    require(compatible_fn == 13 and incompatible_fp == 1, "Stage182-B error direction mismatch")
    require(margin_signal.get("compatible_positive_dominant") is True and margin_signal.get("negative_margin_systematic") is True, "Stage182-B compatible-positive direction is not established")
    require(margin_rows and float(margin_rows[0]["median_paired_logit_difference"]) < 0, "Stage182-B median paired gap is not negative")
    ci = margin_signal.get("paired_logit_bootstrap_ci") or []
    require(len(ci) == 2 and max(float(value) for value in ci) < 0, "Stage182-B CI is not entirely negative")
    directional = {
        "decision": STAGE182B_DECISION, "compatible_false_negatives": compatible_fn,
        "incompatible_false_positives": incompatible_fp,
        "median_paired_logit_difference": margin_rows[0]["median_paired_logit_difference"],
        "bootstrap_ci": ci, "direction_only_not_hyperparameter_fit": True,
    }

    trainer = trainer_audit(trainer_path, report183)
    gradients = gradient_rows()
    balance_rows = [
        {"dimension": "pair", "identifier": key, "eligible_rows": count, "expected_rows": 5, "passed": count == 5, "interpretation": "equal row and pair mean under current topology"}
        for key, count in sorted(pair_counts.items())
    ] + [
        {"dimension": "family", "identifier": key, "eligible_rows": count, "expected_rows": 121, "passed": count == 121, "interpretation": "reported balance; never an eligibility rule"}
        for key, count in sorted(family_counts.items())
    ]
    fixed_rows = [
        {"parameter": "margin_logit", "value": FIXED_MARGIN_LOGIT, "default_value": FIXED_MARGIN_LOGIT, "fixed_no_sweep": True, "rationale": "native sigmoid decision boundary", "forbidden_adaptation": "dataset/dev/Stage182 gap fitting or confidence target"},
        {"parameter": "compatible_positive_margin_weight", "value": FIXED_MARGIN_WEIGHT, "default_value": DEFAULT_MARGIN_WEIGHT, "fixed_no_sweep": True, "rationale": "active-side gradient is 10% of positive BCE at z=0", "forbidden_adaptation": "sweep, schedule, family/pair/rate rescale, learned weight"},
    ]
    consumption_rows = [
        {"requirement": "authoritative eligibility", "authoritative_field": "eligible_for_positive_margin", "validation": "must be JSON boolean true", "failure_behavior": "exclude row", "eligibility_inference_forbidden": "family, reason code, Stage182 membership, prediction, probability, final label, text"},
        {"requirement": "row identity", "authoritative_field": "row_id", "validation": "exact unique one-to-one source/sidecar join; no missing or extra IDs", "failure_behavior": "block training start", "eligibility_inference_forbidden": "row-order join"},
        {"requirement": "dataset provenance", "authoritative_field": "source_dataset_sha256", "validation": f"exact {args.expected_dataset_sha256}", "failure_behavior": "block training start", "eligibility_inference_forbidden": "path-only acceptance"},
        {"requirement": "sidecar provenance", "authoritative_field": "semantic sidecar SHA-256", "validation": f"exact {args.expected_sidecar_semantic_sha256}", "failure_behavior": "block training start", "eligibility_inference_forbidden": "file-name or file-order acceptance"},
        {"requirement": "defense in depth", "authoritative_field": "split/frame_compatible_label/integrity_status/time_swap_status/dataset_source_status", "validation": "train, 1, ELIGIBLE, PASS, PASS", "failure_behavior": "exclude row or block inconsistent sidecar", "eligibility_inference_forbidden": "derived substitute fields"},
    ]
    objective_rows = [
        {"component": "native frame signal", "current_source": 'output["frame_logit"]', "current_semantics": "pre-sigmoid scalar", "stage187_contract": "masked absolute hinge input", "changed": False, "evidence": trainer["frame_output_evidence"]},
        {"component": "native frame BCE", "current_source": "binary_cross_entropy_with_logits", "current_semantics": "row mean, implicit weight 1.0, no pos_weight", "stage187_contract": "unchanged", "changed": False, "evidence": trainer["frame_bce"]},
        {"component": "final CE", "current_source": 'cross_entropy(output["logits"], labels)', "current_semantics": "final classifier objective", "stage187_contract": "unchanged; never use loss_logits", "changed": False, "evidence": trainer["final_ce"]},
        {"component": "total objective", "current_source": "existing total plus existing intervention", "current_semantics": "L_existing", "stage187_contract": "L_existing + lambda_margin * eligible-row mean relu(-frame_logit)", "changed": "default-off append only", "evidence": trainer["assembly"]},
    ]
    default_rows = [
        {"setting": "weight", "recommended_name": "--compatible-positive-margin-weight", "default": 0.0, "fixed_intervention_value": 0.05, "activation_requirement": "nonzero requires sidecar path and expected semantic SHA", "ambiguous_alias_forbidden": "--margin-weight and prior intervention flags"},
        {"setting": "margin logit", "recommended_name": "--compatible-positive-margin-logit", "default": 0.0, "fixed_intervention_value": 0.0, "activation_requirement": "fixed constant; no CLI-selected alternative in Stage186 audit", "ambiguous_alias_forbidden": "probability/confidence margin aliases"},
        {"setting": "sidecar path", "recommended_name": "controlled_integrity_sidecar_path", "default": None, "fixed_intervention_value": "authoritative Stage185 sidecar", "activation_requirement": "required for nonzero weight", "ambiguous_alias_forbidden": "implicit latest-folder discovery"},
        {"setting": "semantic SHA", "recommended_name": "expected_integrity_sidecar_semantic_sha256", "default": None, "fixed_intervention_value": args.expected_sidecar_semantic_sha256, "activation_requirement": "required for nonzero weight", "ambiguous_alias_forbidden": "path-only trust"},
    ]
    checkpoint_rows = [
        {"property": "best epoch metric", "current_value": "clean-dev final_macro_f1", "stage187_required_value": "clean-dev final_macro_f1", "changed": False, "evidence": trainer["checkpoint_evidence"]},
        {"property": "tie breaks", "current_value": "none specified for frame recall/margin", "stage187_required_value": "unchanged; no margin or cohort tie-break", "changed": False, "evidence": trainer["checkpoint_score_evidence"]},
        {"property": "coverage feedback", "current_value": "not a selection signal", "stage187_required_value": "not a selection signal", "changed": False, "evidence": "Stage185 coverage is identity/audit metadata only"},
    ]
    prior_rows = [
        {"intervention": "Stage175 SUPPORT anchor", "target": "final SUPPORT margin against detached canonical reference", "comparison_topology": "anchor/reference", "output_space": "final classifier", "distinct_from_stage186": True, "evidence": "Stage183 closure candidate assessment"},
        {"intervention": "Stage177 pair ordering", "target": "compatible above incompatible", "comparison_topology": "within-pair ranking", "output_space": "native frame_logit", "distinct_from_stage186": True, "evidence": "ordering has no absolute boundary"},
        {"intervention": "Stage186 fixed margin", "target": "eligible compatible positive at or above logit zero", "comparison_topology": "individual eligible row", "output_space": "native frame_logit", "distinct_from_stage186": True, "evidence": "absolute boundary hinge; no reference or pair opponent"},
    ]
    gate_rows = [
        {"gate": "Stage185 identity and SHA", "observed": "passed", "required": "passed", "passed": True, "authorized_next_stage": NEXT},
        {"gate": "605 rows / 121 pairs / 5 families", "observed": f"{len(eligible)}/{len(pair_counts)}/{len(family_counts)}", "required": "605/121/5", "passed": True, "authorized_next_stage": NEXT},
        {"gate": "121 x 5 topology", "observed": sorted(set(pair_counts.values())), "required": [5], "passed": True, "authorized_next_stage": NEXT},
        {"gate": "fixed margin/weight/default", "observed": "0.0/0.05/0.0", "required": "0.0/0.05/0.0", "passed": True, "authorized_next_stage": NEXT},
        {"gate": "eligible normalization and empty behavior", "observed": "specified", "required": "eligible-row mean and graph zero", "passed": True, "authorized_next_stage": NEXT},
        {"gate": "trainer integration and selection invariance", "observed": "static evidence passed", "required": "passed", "passed": True, "authorized_next_stage": NEXT},
        {"gate": "implementation or training performed", "observed": False, "required": False, "passed": True, "authorized_next_stage": NEXT},
    ]

    report = {
        "stage": STAGE, "decision": READY,
        "scope": {"static_specification_audit_only": True, "loss_implemented": False, "training": False},
        "input_validation": {"status": "passed", "rows": input_rows},
        "stage185_closure": {"decision": STAGE185_DECISION, "dataset_sha256": args.expected_dataset_sha256, "sidecar_semantic_sha256": actual_semantic_sha, "rows": len(rows), "one_to_one_join_passed": True, "stage182_regression_passed": True, "contamination_recovery": "22/22", "blocked_invariants": 0},
        "stage183_candidate": {"decision": STAGE183_DECISION, "candidate": STAGE183_CANDIDATE, "margin_was_unset": True, "nonzero_weight_was_unset": True, "implementation_was_unauthorized": True},
        "stage182b_directional_evidence": directional,
        "trainer_static_audit": trainer,
        "eligible_cohort": {"eligible_rows": len(eligible), "train_compatible_rows": len(train_compatible), "eligible_rate": len(eligible) / len(train_compatible), "eligible_pairs": len(pair_counts), "eligible_families": len(family_counts), "family_names": sorted(family_counts), "eligible_rows_per_pair": sorted(set(pair_counts.values())), "eligible_rows_per_family": dict(sorted(family_counts.items())), "unresolved_train_compatible": len(unresolved_positive), "ineligible_train_compatible": len(ineligible_positive), "dev_or_incompatible_or_noneligible_leakage": len(invalid_eligible)},
        "pair_family_balance": {"eligible_pair_share_of_train_pairs": len(pair_counts) / 240, "largest_family_share": max(family_counts.values()) / len(eligible), "family_concentration_warning": False, "row_mean_equals_equal_pair_mean_under_current_topology": True, "future_topology_change_requires_new_gate": True},
        "fixed_hyperparameters": {"margin_logit": FIXED_MARGIN_LOGIT, "weight": FIXED_MARGIN_WEIGHT, "default_weight": DEFAULT_MARGIN_WEIGHT, "no_sweep": True, "no_adaptive_rescale": True},
        "margin_formula": {"signal": 'output["frame_logit"]', "formula": "mean_{i in eligible} relu(0.0 - frame_logit_i)", "probability_target": False, "final_classifier_target": False},
        "normalization_contract": {"denominator": "eligible_count", "family_or_pair_reweight": False, "current_equal_pair_equivalence": True},
        "zero_eligible_contract": {"semantic_form": "frame_logit.sum() * 0.0", "graph_compatible": True, "division_by_zero": False, "batch_skip": False, "optimizer_semantics_changed": False},
        "gradient_scale": {"rows": gradients, "boundary_subgradient_fixed": False, "active_region": "z<0", "inactive_region": "z>0", "weight_selected_from_grid": False},
        "sidecar_consumption": {"rows": consumption_rows, "exact_row_id_join": True, "dataset_sha_gate": True, "semantic_sha_gate": True, "fail_closed_before_training": True},
        "objective_integration": {"formula": "L_total = L_existing + 0.05 * L_margin", "rows": objective_rows, "existing_auxiliary_weights_changed": False, "final_ce_changed": False},
        "default_off_contract": {"default_weight": 0.0, "fixed_run_weight": 0.05, "margin_logit": 0.0, "rows": default_rows},
        "checkpoint_selection": {"unchanged": True, "metric": "clean-dev final_macro_f1", "rows": checkpoint_rows},
        "prior_intervention_nonredundancy": {"equivalent_existing_intervention": False, "rows": prior_rows},
        "implementation_readiness": {"default_off_trainer_implementation_ready": True, "loss_implementation_performed": False, "training_performed": False, "checkpoint_evaluation_performed": False},
        "stage187_gate": {"authorized_next_stage": NEXT, "scope": "default-off trainer implementation only; no training or checkpoint evaluation", "rows": gate_rows},
        "limitations": ["Only 121 of 240 train pairs contribute.", "Exact family balance does not establish population-wide cleanliness.", "The z=0 ReLU subgradient convention remains an implementation detail to record in Stage187.", "This audit did not import, compile, or execute the trainer; the analyzer itself performed static text parsing only."],
        "safety_policy": {"static_only": True, "no_source_modification": True, "no_torch_or_model_import": True, "no_checkpoint_or_forward": True, "no_loss_implementation": True, "no_training": True, "no_evaluation_or_calibration": True, "no_margin_or_weight_sweep": True, "no_annotation_or_llm_labeling": True},
    }

    write_json(output_dir / OUTPUTS[0], report)
    (output_dir / OUTPUTS[1]).write_text(markdown(report), encoding="utf-8")
    csv_rows_by_name = {
        OUTPUTS[2]: input_rows, OUTPUTS[3]: cohort_rows, OUTPUTS[4]: balance_rows,
        OUTPUTS[5]: fixed_rows, OUTPUTS[6]: gradients, OUTPUTS[7]: consumption_rows,
        OUTPUTS[8]: objective_rows, OUTPUTS[9]: default_rows, OUTPUTS[10]: checkpoint_rows,
        OUTPUTS[11]: prior_rows, OUTPUTS[12]: gate_rows,
    }
    for name, values in csv_rows_by_name.items():
        write_csv(output_dir / name, CSV_HEADERS[name], values)
    require(all((output_dir / name).is_file() for name in OUTPUTS), "13-file output contract failure")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except AuditBlocked as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked_report = {
            "stage": STAGE, "decision": BLOCKED,
            "scope": {"static_specification_audit_only": True, "loss_implemented": False, "training": False},
            "input_validation": {"status": "blocked", "reason": str(exc)},
            "stage185_closure": None, "stage183_candidate": None,
            "stage182b_directional_evidence": None, "trainer_static_audit": None,
            "eligible_cohort": None, "pair_family_balance": None,
            "fixed_hyperparameters": {"margin_logit": FIXED_MARGIN_LOGIT, "weight": FIXED_MARGIN_WEIGHT, "default_weight": DEFAULT_MARGIN_WEIGHT},
            "margin_formula": None, "normalization_contract": None,
            "zero_eligible_contract": None, "gradient_scale": None,
            "sidecar_consumption": None, "objective_integration": None,
            "default_off_contract": None, "checkpoint_selection": None,
            "prior_intervention_nonredundancy": None,
            "implementation_readiness": {"ready": False, "loss_implementation_performed": False, "training_performed": False},
            "stage187_gate": {"authorized_next_stage": None, "blocked_reason": str(exc)},
            "limitations": [str(exc)],
            "safety_policy": {"static_only": True, "no_loss_implementation": True, "no_training": True},
        }
        write_json(output_dir / OUTPUTS[0], blocked_report)
        (output_dir / OUTPUTS[1]).write_text(f"# Stage186-A blocked\n\n`{BLOCKED}`\n\nReason: {exc}\n", encoding="utf-8")
        for name, fields in CSV_HEADERS.items():
            values = [{"gate": "audit_blocked", "observed": str(exc), "required": "all gates pass", "passed": False, "authorized_next_stage": ""}] if name == OUTPUTS[12] else []
            write_csv(output_dir / name, fields, values)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
