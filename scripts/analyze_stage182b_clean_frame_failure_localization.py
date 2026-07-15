"""Localize Stage182-A clean native-frame failures from frozen artifacts only.

This module deliberately imports no model or tensor library.  It loads existing
CSV/JSON products, performs deterministic identity joins and fixed statistical
summaries, and writes evaluation-only Stage182-B artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE = "Stage182-B"
EXPECTED_182A = "STAGE182A_DATA_CONTAMINATION_CONFIRMED_AND_CLEAN_MODEL_FAILURE_SET_READY"
EXPECTED_179A = "STAGE179A_FRAME_SEMANTICS_REPRESENTATION_CAUSE_MIXED_OR_INSUFFICIENT"
BLOCKED = "STAGE182B_CLEAN_FRAME_FAILURE_LOCALIZATION_BLOCKED"
REPRESENTATION = "STAGE182B_CLEAN_FRAME_REPRESENTATION_MISLOCALIZATION_SIGNAL"
READOUT = "STAGE182B_CLEAN_FRAME_READOUT_ALIGNMENT_FAILURE_SIGNAL"
MARGIN = "STAGE182B_COMPATIBLE_POSITIVE_MARGIN_COLLAPSE_SIGNAL"
POLARITY = "STAGE182B_POLARITY_CONDITIONED_FRAME_INTERFERENCE_SIGNAL"
MIXED = "STAGE182B_CLEAN_FRAME_FAILURE_LOCALIZATION_MIXED"
INSUFFICIENT = "STAGE182B_NATIVE_FRAME_LOCALIZATION_ARTIFACTS_INSUFFICIENT"

OUTPUT_JSON = "stage182b_clean_frame_failure_localization_report.json"
OUTPUT_MD = "stage182b_clean_frame_failure_localization_report.md"
OUTPUTS = {
    "candidate": "stage182b_candidate_localization.csv",
    "pairs": "stage182b_matched_control_pairs.csv",
    "direction": "stage182b_native_error_direction.csv",
    "margin": "stage182b_compatible_positive_margin.csv",
    "projection": "stage182b_projection_bias_decomposition.csv",
    "centroid": "stage182b_centroid_head_agreement.csv",
    "movement": "stage182b_representation_movement.csv",
    "contrast": "stage182b_clean_hard_correct_contrast.csv",
    "family": "stage182b_family_localization.csv",
    "cohort": "stage182b_cohort_localization.csv",
    "polarity": "stage182b_polarity_conditioned_comparison.csv",
    "coverage": "stage182b_artifact_coverage.csv",
    "gate": "stage182b_stage183_gate_evidence.csv",
}

BASE = ["row_id", "item_role", "stage176_cohort", "intervention_type", "frame_label",
        "frame_prediction", "frame_logit", "frame_probability", "frame_head_projection",
        "representation_movement_from_none", "centroid_prediction", "centroid_correct"]
SCHEMAS = {
    "candidate": BASE + ["native_error_direction", "matched_control_row_id", "scalar_source"],
    "pairs": ["candidate_row_id", "control_row_id", "intervention_type", "stage176_cohort",
              "candidate_frame_logit", "control_frame_logit", "candidate_minus_control_frame_logit",
              "candidate_frame_probability", "control_frame_probability",
              "candidate_minus_control_frame_probability", "candidate_minus_control_projection",
              "candidate_minus_control_centroid_correct", "candidate_and_control_same_native_label",
              "candidate_and_control_same_family", "candidate_and_control_same_construction_status"],
    "direction": ["native_error_direction", "count", "rate", "family_composition",
                  "stage176_cohort_composition", "median_frame_logit", "median_frame_probability"],
    "margin": ["subset", "count", "rate", "median_frame_logit", "logit_bootstrap_ci_lower",
               "logit_bootstrap_ci_upper", "median_frame_probability", "below_zero_logit_count",
               "matched_control_median_logit", "median_paired_logit_difference",
               "median_paired_probability_difference", "exact_sign_test_p", "family_composition",
               "stage176_cohort_composition"],
    "projection": ["row_id", "subset", "frame_logit", "frame_head_projection", "inferred_bias",
                   "projection_sign", "logit_sign", "projection_positive_logit_negative",
                   "projection_negative_logit_negative", "relationship_verified", "reconstruction_error"],
    "centroid": ["row_id", "item_role", "intervention_type", "stage176_cohort", "frame_prediction",
                 "frame_label", "centroid_prediction", "centroid_correct", "agreement_class"],
    "movement": ["row_id", "subset", "intervention_type", "movement_status",
                 "representation_movement_from_none"],
    "contrast": ["metric", "candidate_value", "clean_hard_correct_value", "interpretation"],
    "family": ["intervention_type", "candidate_count", "compatible_false_negative_count",
               "incompatible_false_positive_count", "median_frame_logit", "median_matched_control_logit",
               "median_paired_logit_difference", "paired_sign_test_p", "paired_sign_test_bh_q",
               "centroid_correct_rate", "projection_negative_rate", "representation_movement_median",
               "stage176_cohort_composition", "support_interpretation"],
    "cohort": ["stage176_cohort", "candidate_count", "clean_hard_correct_count",
               "candidate_rate_within_clean_hard", "compatible_false_negative_count",
               "median_frame_logit", "median_paired_logit_difference", "centroid_correct_rate",
               "fisher_exact_p", "interpretation"],
    "polarity": ["polarity_support", "none_support", "polarity_median_paired_logit_gap",
                 "none_median_paired_logit_gap", "gap_difference", "pooled_scale",
                 "bootstrap_ci_lower", "bootstrap_ci_upper", "centroid_correctness_difference",
                 "projection_difference", "effect_gate_passed", "interpretation"],
    "coverage": ["artifact", "cohort", "field", "available_count", "required_count", "coverage_rate",
                 "threshold", "passed"],
    "gate": ["decision", "criterion", "observed_value", "threshold", "passed", "authorized_next_stage"],
}

ALIASES = {
    "frame_label": ("native_frame_label", "dataset_native_frame_label", "gold_frame_label", "frame_compatible_label"),
    "frame_prediction": ("native_frame_prediction", "frame_prediction"),
    "frame_logit": ("native_frame_logit", "frame_logit"),
    "frame_probability": ("native_frame_probability", "native_frame_prob", "frame_probability", "frame_prob"),
    "frame_head_projection": ("frame_head_projection", "head_direction_projection", "unit_head_direction_projection"),
    "representation_movement_from_none": ("representation_movement_from_none", "representation_displacement_from_none"),
    "centroid_prediction": ("stage179_centroid_prediction", "centroid_prediction"),
    "centroid_correct": ("stage179_centroid_correct", "centroid_correct"),
}
INT_FIELDS = {"frame_label", "frame_prediction", "centroid_prediction"}
BOOL_FIELDS = {"centroid_correct"}


class LocalizationBlocked(ValueError):
    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


def require(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise LocalizationBlocked(message, **details)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalizationBlocked(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise LocalizationBlocked(f"cannot read CSV {path}: {exc}") from exc
    require(bool(header), f"CSV has no header: {path}")
    return rows, header


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LocalizationBlocked(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: encode(row.get(field, "")) for field in fields})


def encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return "" if value is None else value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    require(text in {"true", "false", "1", "0"}, f"invalid boolean value {value!r}")
    return text in {"true", "1"}


def value_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
    return a == b


def normalize_value(field: str, raw: Any) -> Any:
    if raw is None or str(raw).strip() == "":
        return None
    if field in INT_FIELDS:
        value = float(raw)
        require(value.is_integer(), f"non-integral {field}: {raw}")
        return int(value)
    if field in BOOL_FIELDS:
        return as_bool(raw)
    return float(raw)


def normalize_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, aliases in ALIASES.items():
        populated = [(alias, normalize_value(field, row.get(alias))) for alias in aliases
                     if row.get(alias) is not None and str(row.get(alias)).strip() != ""]
        if populated:
            first = populated[0][1]
            require(all(value_equal(first, value) for _, value in populated[1:]),
                    f"conflicting aliases for {field} in {source}", schema_mismatch={field: populated})
            result[field] = first
        else:
            result[field] = None
    return result


def identity(row: dict[str, Any]) -> str | None:
    for key in ("row_id", "id", "source_row_id"):
        if str(row.get(key, "")).strip():
            return str(row[key]).strip()
    return None


def merge_normalized(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = identity(row)
        if key:
            grouped[key].append(normalize_row(row, source))
    merged: dict[str, dict[str, Any]] = {}
    for key, values in grouped.items():
        item: dict[str, Any] = {}
        for field in ALIASES:
            present = [row[field] for row in values if row[field] is not None]
            require(not present or all(value_equal(present[0], value) for value in present[1:]),
                    f"duplicate identity has conflicting {field} in {source}", duplicate_ids=[key])
            item[field] = present[0] if present else None
        merged[key] = item
    return merged


def med(values: Iterable[Any]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return statistics.median(data) if data else None


def mean(values: Iterable[Any]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return statistics.fmean(data) if data else None


def rate(n: int, d: int) -> float | None:
    return n / d if d else None


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return data[low] if low == high else data[low] + (data[high] - data[low]) * (pos - low)


def bootstrap(values: list[float], samples: int, seed: int, statistic=statistics.median) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    estimates = [float(statistic([values[rng.randrange(len(values))] for _ in values])) for _ in range(samples)]
    return quantile(estimates, .025), quantile(estimates, .975)


def sign_test(values: list[float]) -> float | None:
    nonzero = [value for value in values if not math.isclose(value, 0.0, abs_tol=1e-12)]
    if not nonzero:
        return 1.0 if values else None
    n = len(nonzero)
    k = min(sum(value > 0 for value in nonzero), sum(value < 0 for value in nonzero))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def fisher(a: int, b: int, c: int, d: int) -> float:
    r1, r2, c1, total = a + b, c + d, a + c, a + b + c + d
    lo, hi = max(0, c1 - r2), min(r1, c1)
    probability = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(total, c1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-15))


def bh(pairs: list[tuple[str, float | None]]) -> dict[str, float | None]:
    valid = sorted((p, key) for key, p in pairs if p is not None)
    result: dict[str, float | None] = {key: None for key, _ in pairs}
    running = 1.0
    for rank in range(len(valid), 0, -1):
        p, key = valid[rank - 1]
        running = min(running, p * len(valid) / rank)
        result[key] = running
    return result


def composition(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def scalar(row: dict[str, Any], field: str) -> Any:
    return row.get("scalars", {}).get(field)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage182a-report", type=Path, required=True)
    parser.add_argument("--stage182a-clean-candidates", type=Path, required=True)
    parser.add_argument("--stage182a-clean-controls", type=Path, required=True)
    parser.add_argument("--stage182a-unique-item-integrity", type=Path, required=True)
    parser.add_argument("--stage182a-canonical-control-audit", type=Path, required=True)
    parser.add_argument("--stage180a-hidden-item-key", type=Path, required=True)
    parser.add_argument("--stage180a-pass2-packet", type=Path, required=True)
    parser.add_argument("--stage179a-report", type=Path, default=None)
    parser.add_argument("--stage179a-hard39-attribution", type=Path, default=None)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=182)
    parser.add_argument("--minimum-candidates", type=int, default=8)
    return parser.parse_args()


def markdown(report: dict[str, Any]) -> str:
    top = report["candidate_topology"]
    direction = report["native_error_direction"]
    paired = report["matched_control_analysis"]
    centroid = report["centroid_readout_analysis"]
    projection = report["projection_bias_analysis"]
    movement = report["representation_movement_analysis"]
    coverage = report["artifact_coverage"]
    provenance = report.get("provenance", {})
    return f"""# Stage182-B clean native-frame failure localization

**Decision:** `{report['decision']}`

**Artifact source mode:** `{coverage.get('artifact_source_mode')}`

## Artifact provenance

Stage180-A manifest SHA: `{provenance.get('stage180a_manifest_sha256')}`. Stage180-A Pass-2 packet SHA: `{provenance.get('stage180a_pass2_packet_sha256')}`. Stage182-A candidate CSV SHA: `{provenance.get('stage182a_candidate_csv_sha256')}`. Stage182-A decision: `{provenance.get('stage182a_report_decision')}`.

{coverage.get('provenance_limitation', '')}

## Fixed clean cohorts

The analysis retained `{top.get('candidate_count')}` Stage182-A clean model-failure candidates, `{top.get('matched_candidate_control_count')}` linked clean controls, `{top.get('clean_hard_native_frame_correct_count')}` clean hard native-frame-correct references, and `{top.get('all_clean_control_count')}` clean control references. Every candidate is a hard row with a resolved schema, valid grammar and intervention contract, valid canonical control, and a native-frame label/prediction mismatch. Contaminated rows were excluded from every comparison.

Compatible false negatives: `{direction.get('compatible_false_negative_count')}`. Incompatible false positives: `{direction.get('incompatible_false_positive_count')}`.

## Paired margin localization

The median candidate-minus-control frame-logit difference was `{paired.get('median_paired_logit_difference')}` (fixed-seed bootstrap 95% CI `{paired.get('paired_logit_bootstrap_ci')}`); the two-sided exact sign-test p-value was `{paired.get('exact_sign_test_p')}`. These are descriptive localization statistics, not causal tests.

## Centroid and readout

Centroid/head subanalysis availability: `{centroid.get('subanalysis_available')}`. Candidate centroid-correct rate was `{centroid.get('candidate_centroid_correct_rate')}` and matched-control centroid-correct rate was `{centroid.get('matched_control_centroid_correct_rate')}`. Stage179 centroid outputs are gold-conditioned, leave-one-row-out, transductive diagnostics—not a deployable classifier or a Stage182 fitted probe.

Projection subanalysis availability: `{projection.get('subanalysis_available')}`. Bias-specific decomposition availability: `{projection.get('bias_specific_subanalysis_available')}`. A bias-specific conclusion is reported only when the stored projection reconstructs the native logit with an effectively constant inferred bias and maximum error at most `1e-5`.

Representation-movement subanalysis availability: `{movement.get('subanalysis_available')}`. Missing centroid, projection, or movement fields do not suppress scalar-margin analysis.

## Stratified evidence

Family results keep support below three descriptive only. Benjamini-Hochberg correction is applied only to family-specific sign tests. The `none`/`polarity_flip` comparison and harmful/beneficial cohort results are associations, not causal polarity-channel or training-mechanism findings. Representation movement uses only the frozen scalar magnitude and does not infer vector direction.

## Authorization and limitations

Authorized design-only route: `{report['stage183_gate'].get('authorized_next_stage')}`. Training remains unauthorized. No annotation, model, checkpoint, dataset, label, generator, calibration, or threshold state was modified. Stage182-B distinguishes scalar-margin, centroid/representation, readout, and family-conditioned evidence but makes no causal mechanism claim.
"""


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    current = "argument_validation"
    diagnostics: dict[str, Any] = {}
    try:
        require(args.bootstrap_samples >= 1000, "--bootstrap-samples must be at least 1000")
        require(args.minimum_candidates >= 1, "--minimum-candidates must be at least 1")
        external_report = args.stage179a_report is not None
        external_hard39 = args.stage179a_hard39_attribution is not None
        require(external_report == external_hard39,
                "--stage179a-report and --stage179a-hard39-attribution must be provided together")
        artifact_source_mode = ("external_stage179_artifacts" if external_report
                                else "embedded_stage180a_pass2")
        current = "input_read"
        report182 = read_json(args.stage182a_report)
        report179 = read_json(args.stage179a_report) if external_report else None
        require(report182.get("decision") == EXPECTED_182A, "unexpected Stage182-A decision")
        if report179 is not None:
            require(report179.get("decision") == EXPECTED_179A, "unexpected Stage179-A decision")
        candidates_raw, _ = read_csv(args.stage182a_clean_candidates)
        controls_raw, _ = read_csv(args.stage182a_clean_controls)
        items_raw, _ = read_csv(args.stage182a_unique_item_integrity)
        canonical_raw, _ = read_csv(args.stage182a_canonical_control_audit)
        hidden_raw, _ = read_csv(args.stage180a_hidden_item_key)
        pass2_raw, _ = read_csv(args.stage180a_pass2_packet)
        hard179_raw, _ = (read_csv(args.stage179a_hard39_attribution)
                          if external_hard39 else ([], []))
        transitions_raw, _ = read_csv(args.stage176a_row_transitions)
        provenance_limitation = (
            "" if external_report else
            "Stage179-derived scalar values were frozen into the Stage180-A packet. "
            "Direct Stage179 runtime report provenance was unavailable. "
            "No scalar was reconstructed or recomputed."
        )
        input_hashes = report182.get("input_validation", {}).get("input_sha256", {})
        provenance = {
            "artifact_source_mode": artifact_source_mode,
            "stage180a_manifest_sha256": input_hashes.get("stage180a_manifest"),
            "stage180a_pass2_packet_sha256": sha256(args.stage180a_pass2_packet),
            "stage182a_candidate_csv_sha256": sha256(args.stage182a_clean_candidates),
            "stage182a_report_decision": report182.get("decision"),
            "stage179a_report_decision": report179.get("decision") if report179 else None,
            "scalar_reconstructed_or_recomputed": False,
        }

        current = "identity_validation"
        def unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
            ids = [identity(row) for row in rows]
            require(all(ids), f"{name} lacks row identity")
            duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
            require(not duplicates, f"duplicate row_id in {name}", duplicate_ids=duplicates)
            return {str(identity(row)): row for row in rows}

        candidate_by_id = unique(candidates_raw, "Stage182-A candidates")
        control_by_id = unique(controls_raw, "Stage182-A clean controls")
        item_by_id = unique(items_raw, "Stage182-A unique items")
        canonical_by_id = unique(canonical_raw, "Stage182-A canonical audit")
        candidate_ids = set(candidate_by_id)
        report_ids = set(report182.get("clean_failure_set_readiness", {}).get("candidate_row_ids", []))
        require(candidate_ids == report_ids, "candidate IDs disagree with Stage182-A report",
                missing_ids=sorted(report_ids - candidate_ids), extra_ids=sorted(candidate_ids - report_ids))
        require(len(candidate_ids) == 14, "candidate topology must contain 14 rows")
        require(len(control_by_id) == 30, "clean-control topology must contain 30 rows")
        clean_correct_ids = {key for key, row in item_by_id.items()
                             if row.get("final_diagnostic_class") == "CLEAN_HARD_NATIVE_FRAME_CORRECT"}
        require(len(clean_correct_ids) == 7, "clean-hard-correct topology must contain seven rows")

        matched_ids: set[str] = set()
        offending: list[dict[str, Any]] = []
        for key, row in candidate_by_id.items():
            flags = (row.get("item_role") == "hard",
                     row.get("final_diagnostic_class") == "CLEAN_MODEL_FAILURE_CANDIDATE",
                     as_bool(row.get("grammar_valid")), as_bool(row.get("contract_exact_match")),
                     as_bool(row.get("canonical_control_valid")), as_bool(row.get("schema_resolved")),
                     int(row.get("native_frame_label", -1)) != int(row.get("native_frame_prediction", -1)))
            if not all(flags):
                offending.append({"row_id": key, "native_frame_label": row.get("native_frame_label"),
                                  "native_frame_prediction": row.get("native_frame_prediction"), "checks": flags})
            matched = str(row.get("matched_source_row_id", ""))
            require(matched in control_by_id, f"candidate {key} lacks its matched clean control", missing_ids=[matched])
            require(matched not in candidate_ids, "candidate/control overlap", extra_ids=[matched])
            require(canonical_by_id.get(key, {}).get("matched_control_row_id") == matched,
                    f"canonical audit link mismatch for {key}")
            matched_ids.add(matched)
        require(not offending, "candidate mismatch invariant failed", offending_candidate_rows=offending)
        require(len(matched_ids) == 14, "matched candidate controls must be one-to-one")
        if external_hard39:
            require(candidate_ids <= set(identity(row) for row in hard179_raw),
                    "Stage179 hard-39 attribution lacks candidate IDs",
                    missing_ids=sorted(candidate_ids - set(identity(row) for row in hard179_raw)))

        current = "artifact_normalization"
        review_to_id: dict[str, str] = {}
        for row in hidden_raw:
            review = str(row.get("review_instance_id", ""))
            row_id = identity(row)
            require(review and row_id, "hidden key lacks review or row identity")
            require(review not in review_to_id, "duplicate review ID in hidden key", duplicate_ids=[review])
            review_to_id[review] = str(row_id)
        pass2_with_ids: list[dict[str, Any]] = []
        pass2_by_source_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pass2_raw:
            review = str(row.get("review_instance_id", ""))
            require(review in review_to_id, "Pass-2 review ID absent from hidden key", missing_ids=[review])
            source_id = review_to_id[review]
            normalized_row = {**row, "row_id": source_id}
            pass2_with_ids.append(normalized_row)
            pass2_by_source_id[source_id].append(normalized_row)
        require(candidate_ids <= set(pass2_by_source_id),
                "Stage180-A Pass-2 packet lacks candidate IDs",
                missing_ids=sorted(candidate_ids - set(pass2_by_source_id)))

        matched_aliases = {
            "frame_label": ("matched_control_native_frame_label",),
            "frame_prediction": ("matched_control_frame_prediction",),
            "frame_logit": ("matched_control_frame_logit",),
            "frame_probability": ("matched_control_frame_prob", "matched_control_frame_probability"),
        }
        embedded_matched: dict[str, dict[str, Any]] = {}
        for candidate_id in sorted(candidate_ids):
            expected_control = str(candidate_by_id[candidate_id]["matched_source_row_id"])
            packet_rows = pass2_by_source_id[candidate_id]
            linked_control_ids = {review_to_id[str(row["matched_control_review_instance_id"])]
                                  for row in packet_rows
                                  if str(row.get("matched_control_review_instance_id", "")) in review_to_id}
            require(linked_control_ids == {expected_control},
                    f"Pass-2 matched-control identity mismatch for {candidate_id}",
                    missing_ids=[expected_control],
                    extra_ids=sorted(linked_control_ids - {expected_control}))
            values: dict[str, Any] = {}
            for field, aliases in matched_aliases.items():
                observed = [normalize_value(field, row[alias]) for row in packet_rows for alias in aliases
                            if str(row.get(alias, "")).strip()]
                require(not observed or all(value_equal(observed[0], value) for value in observed[1:]),
                        f"repeat rows conflict for embedded matched-control {field}: {candidate_id}",
                        schema_mismatch={"row_id": candidate_id, "field": field})
                values[field] = observed[0] if observed else None
            embedded_matched[expected_control] = values

        pass2_normalized = merge_normalized(pass2_with_ids, "Stage180-A Pass-2")
        sources = [("stage180a_pass2", pass2_normalized)]
        if external_hard39:
            sources.append(("stage179a_hard39", merge_normalized(hard179_raw, "Stage179-A hard39")))
        sources.append(("stage176a_transitions", merge_normalized(transitions_raw, "Stage176-A transitions")))
        sources.append(("stage182a_identity_fields",
                        merge_normalized(candidates_raw + controls_raw, "Stage182-A identity fields")))
        wanted_ids = candidate_ids | matched_ids | clean_correct_ids | set(control_by_id)
        normalized: dict[str, dict[str, Any]] = {}
        for row_id in wanted_ids:
            chosen: dict[str, Any] = {field: None for field in ALIASES}
            field_source: dict[str, str] = {}
            for source_name, mapping in sources:
                source_row = mapping.get(row_id, {})
                for field in ALIASES:
                    value = source_row.get(field)
                    if value is not None and chosen[field] is None:
                        chosen[field], field_source[field] = value, source_name
                    elif value is not None and field in {"frame_label", "frame_prediction"}:
                        require(value_equal(chosen[field], value), f"cross-artifact {field} mismatch for {row_id}",
                                schema_mismatch={"row_id": row_id, "field": field})
            normalized[row_id] = {"values": chosen, "sources": field_source}

        for control_id, embedded_values in embedded_matched.items():
            for field, value in embedded_values.items():
                if value is None:
                    continue
                existing = normalized[control_id]["values"].get(field)
                require(existing is None or value_equal(existing, value),
                        f"embedded matched-control {field} disagrees with control row: {control_id}",
                        schema_mismatch={"row_id": control_id, "field": field})
                if existing is None:
                    normalized[control_id]["values"][field] = value
                    normalized[control_id]["sources"][field] = "embedded_stage180a_matched_control"

        def enriched(row_id: str, row: dict[str, Any]) -> dict[str, Any]:
            values = normalized[row_id]["values"]
            return {"row_id": row_id, "item_role": row.get("item_role", ""),
                    "stage176_cohort": row.get("stage176_cohort", "none"),
                    "intervention_type": row.get("intervention_type", ""), "scalars": values,
                    "scalar_source": normalized[row_id]["sources"]}

        candidates = [enriched(key, candidate_by_id[key]) for key in sorted(candidate_ids)]
        controls = [enriched(key, control_by_id[key]) for key in sorted(control_by_id)]
        clean_correct = [enriched(key, item_by_id[key]) for key in sorted(clean_correct_ids)]
        matched_controls = {key: enriched(key, control_by_id[key]) for key in matched_ids}
        transition_ids = {identity(row) for row in transitions_raw}
        require(candidate_ids <= transition_ids, "Stage176 transitions lack candidate IDs",
                missing_ids=sorted(candidate_ids - transition_ids))
        require(composition(candidates, "stage176_cohort") == {"beneficial_correction": 1, "harmful_regression": 13},
                "Stage176 candidate cohort topology mismatch")

        embedded_candidate_required = {
            "native_frame_label": "frame_label",
            "native_frame_prediction": "frame_prediction",
            "native_frame_logit": "frame_logit",
            "native_frame_probability": "frame_probability",
        }
        embedded_control_required = {
            "matched_control_frame_prediction": "frame_prediction",
            "matched_control_frame_logit": "frame_logit",
            "matched_control_frame_prob": "frame_probability",
        }
        embedded_candidate_field_coverage = {
            artifact_field: rate(
                sum(pass2_normalized.get(row_id, {}).get(canonical_field) is not None
                    for row_id in candidate_ids), 14)
            for artifact_field, canonical_field in embedded_candidate_required.items()
        }
        embedded_control_field_coverage = {
            artifact_field: rate(
                sum(embedded_matched.get(row_id, {}).get(canonical_field) is not None
                    for row_id in matched_ids), 14)
            for artifact_field, canonical_field in embedded_control_required.items()
        }
        embedded_required_scalar_coverage = min(
            list(embedded_candidate_field_coverage.values()) + list(embedded_control_field_coverage.values())
        )
        embedded_centroid_coverage = rate(
            sum(pass2_normalized.get(row_id, {}).get("centroid_correct") is not None
                and pass2_normalized.get(row_id, {}).get("centroid_prediction") is not None
                for row_id in candidate_ids | matched_ids), 28)
        embedded_projection_coverage = rate(
            sum(pass2_normalized.get(row_id, {}).get("frame_head_projection") is not None
                for row_id in candidate_ids | matched_ids), 28)
        embedded_movement_coverage = rate(
            sum(pass2_normalized.get(row_id, {}).get("representation_movement_from_none") is not None
                for row_id in candidate_ids | matched_ids), 28)

        current = "paired_analysis"
        candidate_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        for row in candidates:
            label, prediction = scalar(row, "frame_label"), scalar(row, "frame_prediction")
            require(label is not None and prediction is not None and label != prediction,
                    f"normalized native mismatch unavailable for {row['row_id']}")
            direction = "compatible_false_negative" if label == 1 and prediction == 0 else "incompatible_false_positive"
            matched_id = str(candidate_by_id[row["row_id"]]["matched_source_row_id"])
            control = matched_controls[matched_id]
            candidate_rows.append({**{key: row.get(key) for key in BASE if key in row},
                                   **row["scalars"], "row_id": row["row_id"], "item_role": "hard",
                                   "stage176_cohort": row["stage176_cohort"],
                                   "intervention_type": row["intervention_type"],
                                   "native_error_direction": direction, "matched_control_row_id": matched_id,
                                   "scalar_source": row["scalar_source"]})
            def diff(field: str) -> float | None:
                a, b = scalar(row, field), scalar(control, field)
                return float(a) - float(b) if a is not None and b is not None else None
            pair_rows.append({"candidate_row_id": row["row_id"], "control_row_id": matched_id,
                              "intervention_type": row["intervention_type"], "stage176_cohort": row["stage176_cohort"],
                              "candidate_frame_logit": scalar(row, "frame_logit"), "control_frame_logit": scalar(control, "frame_logit"),
                              "candidate_minus_control_frame_logit": diff("frame_logit"),
                              "candidate_frame_probability": scalar(row, "frame_probability"),
                              "control_frame_probability": scalar(control, "frame_probability"),
                              "candidate_minus_control_frame_probability": diff("frame_probability"),
                              "candidate_minus_control_projection": diff("frame_head_projection"),
                              "candidate_minus_control_centroid_correct": diff("centroid_correct"),
                              "candidate_and_control_same_native_label": scalar(row, "frame_label") == scalar(control, "frame_label"),
                              "candidate_and_control_same_family": row["intervention_type"] == control["intervention_type"],
                              "candidate_and_control_same_construction_status": True})
        logit_diffs = [float(row["candidate_minus_control_frame_logit"]) for row in pair_rows
                       if row["candidate_minus_control_frame_logit"] is not None]
        prob_diffs = [float(row["candidate_minus_control_frame_probability"]) for row in pair_rows
                      if row["candidate_minus_control_frame_probability"] is not None]
        proj_diffs = [float(row["candidate_minus_control_projection"]) for row in pair_rows
                      if row["candidate_minus_control_projection"] is not None]
        paired_ci = bootstrap(logit_diffs, args.bootstrap_samples, args.seed)
        paired_analysis = {"pair_count": len(pair_rows), "median_paired_logit_difference": med(logit_diffs),
                           "mean_paired_logit_difference": mean(logit_diffs), "paired_logit_bootstrap_ci": list(paired_ci),
                           "negative_difference_count": sum(value < 0 for value in logit_diffs),
                           "negative_difference_rate": rate(sum(value < 0 for value in logit_diffs), len(logit_diffs)),
                           "exact_sign_test_p": sign_test(logit_diffs),
                           "median_paired_probability_difference": med(prob_diffs),
                           "median_paired_projection_difference": med(proj_diffs),
                           "centroid_agreement_patterns": dict(Counter(str(row["candidate_minus_control_centroid_correct"]) for row in pair_rows))}

        current = "localization_summaries"
        direction_rows = []
        for name in ("compatible_false_negative", "incompatible_false_positive"):
            group = [row for row in candidate_rows if row["native_error_direction"] == name]
            direction_rows.append({"native_error_direction": name, "count": len(group), "rate": rate(len(group), len(candidate_rows)),
                                   "family_composition": composition(group, "intervention_type"),
                                   "stage176_cohort_composition": composition(group, "stage176_cohort"),
                                   "median_frame_logit": med(row["frame_logit"] for row in group),
                                   "median_frame_probability": med(row["frame_probability"] for row in group)})
        fn_rows = [row for row in candidate_rows if row["native_error_direction"] == "compatible_false_negative"]
        fn_ids = {row["row_id"] for row in fn_rows}
        fn_pairs = [row for row in pair_rows if row["candidate_row_id"] in fn_ids]
        fn_logits = [float(row["frame_logit"]) for row in fn_rows if row["frame_logit"] is not None]
        fn_logit_diffs = [float(row["candidate_minus_control_frame_logit"]) for row in fn_pairs if row["candidate_minus_control_frame_logit"] is not None]
        fn_prob_diffs = [float(row["candidate_minus_control_frame_probability"]) for row in fn_pairs if row["candidate_minus_control_frame_probability"] is not None]
        fn_ci = bootstrap(fn_logits, args.bootstrap_samples, args.seed + 1)
        fn_pair_ci = bootstrap(fn_logit_diffs, args.bootstrap_samples, args.seed + 2)
        margin_rows = [{"subset": "compatible_false_negative", "count": len(fn_rows), "rate": rate(len(fn_rows), 14),
                        "median_frame_logit": med(fn_logits), "logit_bootstrap_ci_lower": fn_ci[0],
                        "logit_bootstrap_ci_upper": fn_ci[1],
                        "median_frame_probability": med(row["frame_probability"] for row in fn_rows),
                        "below_zero_logit_count": sum(value < 0 for value in fn_logits),
                        "matched_control_median_logit": med(row["control_frame_logit"] for row in fn_pairs),
                        "median_paired_logit_difference": med(fn_logit_diffs),
                        "median_paired_probability_difference": med(fn_prob_diffs),
                        "exact_sign_test_p": sign_test(fn_logit_diffs), "family_composition": composition(fn_rows, "intervention_type"),
                        "stage176_cohort_composition": composition(fn_rows, "stage176_cohort")}]

        projection_population = candidates + list(matched_controls.values())
        inferred = [float(scalar(row, "frame_logit")) - float(scalar(row, "frame_head_projection"))
                    for row in projection_population if scalar(row, "frame_logit") is not None and scalar(row, "frame_head_projection") is not None]
        bias = med(inferred)
        errors = [abs(value - float(bias)) for value in inferred] if bias is not None else []
        normalized_projection_coverage = rate(len(inferred), len(projection_population))
        projection_subanalysis_available = (normalized_projection_coverage is not None and
                                            normalized_projection_coverage >= .90)
        relationship_verified = (projection_subanalysis_available and
                                 len(inferred) == len(projection_population) and
                                 bool(errors) and max(errors) <= 1e-5)
        projection_rows = []
        for row in projection_population:
            logit, projection = scalar(row, "frame_logit"), scalar(row, "frame_head_projection")
            inferred_bias = float(logit) - float(projection) if logit is not None and projection is not None else None
            projection_rows.append({"row_id": row["row_id"], "subset": "candidate" if row["row_id"] in candidate_ids else "matched_control",
                                    "frame_logit": logit, "frame_head_projection": projection, "inferred_bias": inferred_bias,
                                    "projection_sign": None if projection is None else ("positive" if projection > 0 else "negative" if projection < 0 else "zero"),
                                    "logit_sign": None if logit is None else ("positive" if logit > 0 else "negative" if logit < 0 else "zero"),
                                    "projection_positive_logit_negative": projection is not None and logit is not None and projection > 0 and logit < 0,
                                    "projection_negative_logit_negative": projection is not None and logit is not None and projection < 0 and logit < 0,
                                    "relationship_verified": relationship_verified,
                                    "reconstruction_error": None if inferred_bias is None or bias is None else abs(inferred_bias - bias)})
        fn_projection = [row for row in projection_rows if row["row_id"] in fn_ids]
        projection_negative_rate = rate(sum(row["frame_head_projection"] is not None and row["frame_head_projection"] < 0 for row in fn_projection),
                                        sum(row["frame_head_projection"] is not None for row in fn_projection))
        positive_negative_rate = rate(sum(row["projection_positive_logit_negative"] for row in fn_projection),
                                      sum(row["frame_head_projection"] is not None and row["frame_logit"] is not None for row in fn_projection))
        bias_dominant = relationship_verified and positive_negative_rate is not None and positive_negative_rate >= .70
        projection_analysis = {"available_count": len(inferred),
                               "coverage": normalized_projection_coverage,
                               "subanalysis_available": projection_subanalysis_available,
                               "inferred_bias_median": bias,
                               "inferred_bias_variance": statistics.pvariance(inferred) if inferred else None,
                               "maximum_reconstruction_error": max(errors) if errors else None,
                               "relationship_verified": relationship_verified,
                               "bias_specific_subanalysis_available": relationship_verified,
                               "compatible_fn_projection_positive_logit_negative_rate": positive_negative_rate,
                               "compatible_fn_projection_negative_rate": projection_negative_rate,
                               "bias_dominant_evidence_gate_passed": bias_dominant}

        centroid_rows = []
        for row in candidates + list(matched_controls.values()):
            correct = scalar(row, "centroid_correct")
            centroid_rows.append({"row_id": row["row_id"], "item_role": "hard" if row["row_id"] in candidate_ids else "control",
                                  "intervention_type": row["intervention_type"], "stage176_cohort": row["stage176_cohort"],
                                  "frame_prediction": scalar(row, "frame_prediction"), "frame_label": scalar(row, "frame_label"),
                                  "centroid_prediction": scalar(row, "centroid_prediction"), "centroid_correct": correct,
                                  "agreement_class": "HEAD_WRONG_CENTROID_UNAVAILABLE" if correct is None else
                                  "HEAD_WRONG_CENTROID_CORRECT" if correct else "HEAD_WRONG_CENTROID_WRONG"})
        candidate_centroid = [row for row in centroid_rows if row["row_id"] in candidate_ids and row["centroid_correct"] is not None]
        control_centroid = [row for row in centroid_rows if row["row_id"] in matched_ids and row["centroid_correct"] is not None]
        candidate_centroid_coverage = rate(len(candidate_centroid), 14)
        control_centroid_coverage = rate(len(control_centroid), 14)
        centroid_subanalysis_available = (candidate_centroid_coverage is not None and
                                          control_centroid_coverage is not None and
                                          candidate_centroid_coverage >= .90 and
                                          control_centroid_coverage >= .90)
        centroid_analysis = {"candidate_available_count": len(candidate_centroid),
                             "candidate_coverage": candidate_centroid_coverage,
                             "matched_control_coverage": control_centroid_coverage,
                             "subanalysis_available": centroid_subanalysis_available,
                             "candidate_centroid_correct_rate": rate(sum(row["centroid_correct"] for row in candidate_centroid), len(candidate_centroid)),
                             "candidate_centroid_wrong_rate": rate(sum(not row["centroid_correct"] for row in candidate_centroid), len(candidate_centroid)),
                             "matched_control_available_count": len(control_centroid),
                             "matched_control_centroid_correct_rate": rate(sum(row["centroid_correct"] for row in control_centroid), len(control_centroid)),
                             "head_centroid_disagreement_rate": rate(sum(row["centroid_correct"] for row in candidate_centroid), len(candidate_centroid)),
                             "gold_conditioned_transductive_diagnostic": True, "deployable_classifier": False}

        movement_rows = []
        for subset, group in (("candidate", candidates), ("matched_control", list(matched_controls.values())),
                              ("clean_hard_correct", clean_correct)):
            for row in group:
                value = scalar(row, "representation_movement_from_none")
                status = "not_applicable" if row["intervention_type"] == "none" else "available" if value is not None else "missing"
                movement_rows.append({"row_id": row["row_id"], "subset": subset, "intervention_type": row["intervention_type"],
                                      "movement_status": status, "representation_movement_from_none": value})
        movement_available = sum(row["movement_status"] == "available" for row in movement_rows)
        movement_applicable = sum(row["movement_status"] != "not_applicable" for row in movement_rows)
        normalized_movement_coverage = rate(movement_available, movement_applicable)
        movement_analysis: dict[str, Any] = {
            "vector_direction_inference_authorized": False,
            "coverage": normalized_movement_coverage,
            "subanalysis_available": (normalized_movement_coverage is not None and
                                      normalized_movement_coverage >= .90),
            "subsets": {},
        }
        for subset in ("candidate", "matched_control", "clean_hard_correct"):
            values = [float(row["representation_movement_from_none"]) for row in movement_rows
                      if row["subset"] == subset and row["movement_status"] == "available"]
            movement_analysis["subsets"][subset] = {"available_count": len(values), "median": med(values),
                                                      "iqr": [quantile(values, .25), quantile(values, .75)],
                                                      "bootstrap_ci": list(bootstrap(values, args.bootstrap_samples, args.seed + 3))}

        family_rows = []
        family_p: list[tuple[str, float | None]] = []
        for family in sorted({row["intervention_type"] for row in candidates}):
            group = [row for row in candidate_rows if row["intervention_type"] == family]
            group_pairs = [row for row in pair_rows if row["intervention_type"] == family]
            diffs = [float(row["candidate_minus_control_frame_logit"]) for row in group_pairs if row["candidate_minus_control_frame_logit"] is not None]
            p = sign_test(diffs)
            family_p.append((family, p))
            family_rows.append({"intervention_type": family, "candidate_count": len(group),
                                "compatible_false_negative_count": sum(row["native_error_direction"] == "compatible_false_negative" for row in group),
                                "incompatible_false_positive_count": sum(row["native_error_direction"] == "incompatible_false_positive" for row in group),
                                "median_frame_logit": med(row["frame_logit"] for row in group),
                                "median_matched_control_logit": med(row["control_frame_logit"] for row in group_pairs),
                                "median_paired_logit_difference": med(diffs), "paired_sign_test_p": p,
                                "centroid_correct_rate": rate(sum(row["centroid_correct"] is True for row in centroid_rows if row["row_id"] in {g["row_id"] for g in group}),
                                                                sum(row["centroid_correct"] is not None for row in centroid_rows if row["row_id"] in {g["row_id"] for g in group})),
                                "projection_negative_rate": rate(sum(row["frame_head_projection"] is not None and row["frame_head_projection"] < 0 for row in projection_rows if row["row_id"] in {g["row_id"] for g in group}),
                                                                 sum(row["frame_head_projection"] is not None for row in projection_rows if row["row_id"] in {g["row_id"] for g in group})),
                                "representation_movement_median": med(row["representation_movement_from_none"] for row in movement_rows if row["row_id"] in {g["row_id"] for g in group}),
                                "stage176_cohort_composition": composition(group, "stage176_cohort"),
                                "support_interpretation": "descriptive_only" if len(group) < 3 else "family_conditioned_association"})
        qvalues = bh(family_p)
        for row in family_rows:
            row["paired_sign_test_bh_q"] = qvalues[row["intervention_type"]]

        cohort_rows = []
        for cohort in ("harmful_regression", "beneficial_correction"):
            group = [row for row in candidate_rows if row["stage176_cohort"] == cohort]
            reference = [row for row in clean_correct if row["stage176_cohort"] == cohort]
            other_group = 14 - len(group)
            other_reference = 7 - len(reference)
            group_pairs = [row for row in pair_rows if row["stage176_cohort"] == cohort]
            cohort_rows.append({"stage176_cohort": cohort, "candidate_count": len(group), "clean_hard_correct_count": len(reference),
                                "candidate_rate_within_clean_hard": rate(len(group), len(group) + len(reference)),
                                "compatible_false_negative_count": sum(row["native_error_direction"] == "compatible_false_negative" for row in group),
                                "median_frame_logit": med(row["frame_logit"] for row in group),
                                "median_paired_logit_difference": med(row["candidate_minus_control_frame_logit"] for row in group_pairs),
                                "centroid_correct_rate": rate(sum(row["centroid_correct"] is True for row in centroid_rows if row["row_id"] in {g["row_id"] for g in group}),
                                                                sum(row["centroid_correct"] is not None for row in centroid_rows if row["row_id"] in {g["row_id"] for g in group})),
                                "fisher_exact_p": fisher(len(group), len(reference), other_group, other_reference),
                                "interpretation": "descriptive_only"})

        polarity_pairs = [row for row in pair_rows if row["intervention_type"] == "polarity_flip" and row["candidate_minus_control_frame_logit"] is not None]
        none_pairs = [row for row in pair_rows if row["intervention_type"] == "none" and row["candidate_minus_control_frame_logit"] is not None]
        polarity_values = [float(row["candidate_minus_control_frame_logit"]) for row in polarity_pairs]
        none_values = [float(row["candidate_minus_control_frame_logit"]) for row in none_pairs]
        gap_difference = (med(polarity_values) - med(none_values)) if polarity_values and none_values else None
        pooled_scale = None
        if len(polarity_values) >= 2 and len(none_values) >= 2:
            pooled_scale = math.sqrt(((len(polarity_values) - 1) * statistics.variance(polarity_values) +
                                      (len(none_values) - 1) * statistics.variance(none_values)) /
                                     (len(polarity_values) + len(none_values) - 2))
        rng = random.Random(args.seed + 4)
        boot_gap = []
        if polarity_values and none_values:
            for _ in range(args.bootstrap_samples):
                p_sample = [polarity_values[rng.randrange(len(polarity_values))] for _ in polarity_values]
                n_sample = [none_values[rng.randrange(len(none_values))] for _ in none_values]
                boot_gap.append(statistics.median(p_sample) - statistics.median(n_sample))
        polarity_ci = (quantile(boot_gap, .025), quantile(boot_gap, .975))
        ci_excludes_zero = polarity_ci[0] is not None and (polarity_ci[1] < 0 or polarity_ci[0] > 0)
        polarity_gate = (len(polarity_values) >= 5 and pooled_scale is not None and pooled_scale > 0 and
                         gap_difference is not None and abs(gap_difference) >= .5 * pooled_scale and ci_excludes_zero)
        family_candidate_centroid = lambda family: [row for row in centroid_rows if row["row_id"] in candidate_ids and
                                                     candidate_by_id[row["row_id"]].get("intervention_type") == family and row["centroid_correct"] is not None]
        pc, nc = family_candidate_centroid("polarity_flip"), family_candidate_centroid("none")
        polarity_row = {"polarity_support": len(polarity_values), "none_support": len(none_values),
                        "polarity_median_paired_logit_gap": med(polarity_values), "none_median_paired_logit_gap": med(none_values),
                        "gap_difference": gap_difference, "pooled_scale": pooled_scale, "bootstrap_ci_lower": polarity_ci[0],
                        "bootstrap_ci_upper": polarity_ci[1],
                        "centroid_correctness_difference": (rate(sum(row["centroid_correct"] for row in pc), len(pc)) - rate(sum(row["centroid_correct"] for row in nc), len(nc))) if pc and nc else None,
                        "projection_difference": med(row["candidate_minus_control_projection"] for row in polarity_pairs) - med(row["candidate_minus_control_projection"] for row in none_pairs) if
                        any(row["candidate_minus_control_projection"] is not None for row in polarity_pairs) and any(row["candidate_minus_control_projection"] is not None for row in none_pairs) else None,
                        "effect_gate_passed": polarity_gate, "interpretation": "family_conditioned_association_not_causal_interference"}

        contrast_rows = []
        def contrast(metric: str, c_value: Any, r_value: Any, note: str = "fixed_clean_hard_reference") -> None:
            contrast_rows.append({"metric": metric, "candidate_value": c_value, "clean_hard_correct_value": r_value, "interpretation": note})
        contrast("count", 14, 7)
        contrast("frame_label_distribution", Counter(row["frame_label"] for row in candidate_rows), Counter(scalar(row, "frame_label") for row in clean_correct))
        contrast("median_frame_logit", med(row["frame_logit"] for row in candidate_rows), med(scalar(row, "frame_logit") for row in clean_correct))
        contrast("median_frame_probability", med(row["frame_probability"] for row in candidate_rows), med(scalar(row, "frame_probability") for row in clean_correct))
        contrast("centroid_correct_rate", centroid_analysis["candidate_centroid_correct_rate"],
                 rate(sum(scalar(row, "centroid_correct") is True for row in clean_correct), sum(scalar(row, "centroid_correct") is not None for row in clean_correct)))
        contrast("family_distribution", composition(candidate_rows, "intervention_type"), composition(clean_correct, "intervention_type"))
        contrast("stage176_cohort_distribution", composition(candidate_rows, "stage176_cohort"), composition(clean_correct, "stage176_cohort"))

        coverage_rows = []
        for cohort_name, group, threshold in (("candidates", candidates, .90), ("matched_candidate_controls", list(matched_controls.values()), .90)):
            for field in ALIASES:
                available = sum(scalar(row, field) is not None for row in group)
                coverage_rows.append({"artifact": "normalized_priority_sources", "cohort": cohort_name, "field": field,
                                      "available_count": available, "required_count": len(group), "coverage_rate": rate(available, len(group)),
                                      "threshold": threshold, "passed": rate(available, len(group)) >= threshold})
        candidate_scalar_fields = ("frame_label", "frame_prediction", "frame_logit", "frame_probability")
        control_scalar_fields = ("frame_prediction", "frame_logit", "frame_probability")
        candidate_scalar_coverage = min(
            rate(sum(scalar(row, field) is not None for row in candidates), 14)
            for field in candidate_scalar_fields
        )
        control_scalar_coverage = min(
            rate(sum(scalar(row, field) is not None for row in matched_controls.values()), 14)
            for field in control_scalar_fields
        )
        insufficient = (candidate_scalar_coverage < .90 or control_scalar_coverage < .90 or
                        (artifact_source_mode == "embedded_stage180a_pass2" and
                         embedded_required_scalar_coverage < .90))

        compatible_dominant = len(fn_rows) / 14 >= .75
        negative_systematic = (len(fn_rows) >= args.minimum_candidates and med(fn_logits) is not None and med(fn_logits) < 0 and
                               fn_pair_ci[1] is not None and fn_pair_ci[1] < 0)
        rep_gate = (not insufficient and centroid_subanalysis_available and
                    len(candidates) >= args.minimum_candidates and centroid_analysis["candidate_centroid_wrong_rate"] is not None and
                    centroid_analysis["candidate_centroid_wrong_rate"] >= .65 and
                    centroid_analysis["matched_control_centroid_correct_rate"] is not None and
                    centroid_analysis["matched_control_centroid_correct_rate"] >= .75 and paired_ci[1] is not None and paired_ci[1] < 0 and
                    centroid_analysis["candidate_coverage"] >= .80)
        readout_gate = (not insufficient and centroid_subanalysis_available and
                        len(candidates) >= args.minimum_candidates and centroid_analysis["candidate_centroid_correct_rate"] is not None and
                        centroid_analysis["candidate_centroid_correct_rate"] >= .65 and
                        centroid_analysis["matched_control_centroid_correct_rate"] is not None and
                        centroid_analysis["matched_control_centroid_correct_rate"] >= .75 and paired_ci[1] is not None and paired_ci[1] < 0 and
                        ((projection_subanalysis_available and projection_negative_rate is not None and
                          projection_negative_rate >= .70) or bias_dominant))
        margin_gate = (not insufficient and compatible_dominant and negative_systematic and
                       candidate_scalar_coverage >= .90 and control_scalar_coverage >= .90)
        polarity_gate = not insufficient and polarity_gate
        if insufficient:
            decision, next_stage = INSUFFICIENT, "STAGE183_NATIVE_FRAME_ARTIFACT_RECOVERY_SPEC"
        elif rep_gate:
            decision, next_stage = REPRESENTATION, "STAGE183_FRAME_REPRESENTATION_LOCALIZATION_DESIGN_AUDIT"
        elif readout_gate:
            decision, next_stage = READOUT, "STAGE183_FRAME_READOUT_AND_POSITIVE_MARGIN_DESIGN_AUDIT"
        elif margin_gate:
            decision, next_stage = MARGIN, "STAGE183_COMPATIBLE_FRAME_POSITIVE_PRESERVATION_DESIGN_AUDIT"
        elif polarity_gate:
            decision, next_stage = POLARITY, "STAGE183_POLARITY_FRAME_DISENTANGLEMENT_DESIGN_AUDIT"
        else:
            decision, next_stage = MIXED, "STAGE183_STRATIFIED_CLEAN_FRAME_FAILURE_DESIGN_AUDIT"
        gates = {REPRESENTATION: rep_gate, READOUT: readout_gate, MARGIN: margin_gate, POLARITY: polarity_gate,
                 INSUFFICIENT: insufficient, MIXED: decision == MIXED}
        gate_rows = [{"decision": name, "criterion": "fixed_gate", "observed_value": passed,
                      "threshold": "all_declared_conditions", "passed": passed,
                      "authorized_next_stage": next_stage if name == decision else ""} for name, passed in gates.items()]

        report = {"stage": STAGE, "decision": decision,
                  "scope": {"artifact_only": True, "evaluation_only_failure_localization": True,
                            "artifact_source_mode": artifact_source_mode,
                            "bootstrap_samples": args.bootstrap_samples, "seed": args.seed, "minimum_candidates": args.minimum_candidates},
                  "input_validation": {"status": "passed", "stage182a_decision": EXPECTED_182A,
                                       "stage179a_decision": report179.get("decision") if report179 else None,
                                       "stage179_runtime_decision_directly_validated": report179 is not None,
                                       "candidate_native_frame_mismatch_invariant": True, "contaminated_comparison_rows": 0},
                  "provenance": provenance,
                  "artifact_coverage": {
                                        "artifact_source_mode": artifact_source_mode,
                                        "external_stage179_report_available": external_report,
                                        "external_stage179_hard39_available": external_hard39,
                                        "embedded_pass2_scalar_coverage": {
                                            "candidate_native_fields": embedded_candidate_field_coverage,
                                            "matched_control_fields": embedded_control_field_coverage,
                                            "minimum_required_coverage": embedded_required_scalar_coverage,
                                        },
                                        "embedded_centroid_coverage": embedded_centroid_coverage,
                                        "embedded_projection_coverage": embedded_projection_coverage,
                                        "embedded_movement_coverage": embedded_movement_coverage,
                                        "provenance_limitation": provenance_limitation,
                                        "candidate_scalar_coverage": candidate_scalar_coverage,
                                        "matched_control_scalar_coverage": control_scalar_coverage,
                                        "rows": coverage_rows},
                  "candidate_topology": {"candidate_count": 14, "matched_candidate_control_count": 14,
                                         "clean_hard_native_frame_correct_count": 7, "all_clean_control_count": 30,
                                         "candidate_family_count": len({row["intervention_type"] for row in candidates}),
                                         "schema_unresolved_count": 0},
                  "native_error_direction": {"compatible_false_negative_count": len(fn_rows),
                                             "incompatible_false_positive_count": 14 - len(fn_rows), "rows": direction_rows},
                  "matched_control_analysis": paired_analysis,
                  "compatible_positive_margin": {"compatible_positive_dominant": compatible_dominant,
                                                 "negative_margin_systematic": negative_systematic,
                                                 "paired_logit_bootstrap_ci": list(fn_pair_ci), "rows": margin_rows},
                  "projection_bias_analysis": projection_analysis, "centroid_readout_analysis": centroid_analysis,
                  "representation_movement_analysis": movement_analysis,
                  "clean_hard_correct_contrast": {"reference_is_stage176_hard_subset": True, "rows": contrast_rows},
                  "family_analysis": {"multiple_comparison_correction": "Benjamini-Hochberg_family_tests_only", "rows": family_rows},
                  "cohort_analysis": {"fisher_exact_interpretation": "descriptive_only", "rows": cohort_rows},
                  "polarity_conditioned_analysis": polarity_row,
                  "diagnosis": {"representation_gate": rep_gate, "readout_gate": readout_gate, "compatible_margin_gate": margin_gate,
                                "polarity_gate": polarity_gate, "causal_mechanism_established": False},
                  "stage183_gate": {"authorized_next_stage": next_stage, "authorization_scope": "design_audit_only",
                                    "training_authorized": False, "rows": gate_rows},
                  "limitations": ([provenance_limitation] if provenance_limitation else []) +
                                 ["Stage179 centroid is gold-conditioned and transductive, not deployable.",
                                  "Scalar movement magnitude does not establish representation direction.",
                                  "Family and cohort results are associations, not causal mechanisms.",
                                  "No final-classifier output is used as localization evidence."],
                  "safety_policy": {"annotation": False, "model_import": False, "torch_import": False,
                                    "checkpoint_load": False, "forward": False, "embedding_recomputation": False,
                                    "learned_probe": False, "classifier_fitting": False, "threshold_optimization": False,
                                    "calibration": False, "relabeling": False, "dataset_mutation": False, "training": False}}
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = {"candidate": candidate_rows, "pairs": pair_rows, "direction": direction_rows, "margin": margin_rows,
                  "projection": projection_rows, "centroid": centroid_rows, "movement": movement_rows,
                  "contrast": contrast_rows, "family": family_rows, "cohort": cohort_rows, "polarity": [polarity_row],
                  "coverage": coverage_rows, "gate": gate_rows}
        for name, filename in OUTPUTS.items():
            write_csv(output_dir / filename, tables[name], SCHEMAS[name])
        (output_dir / OUTPUT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output_dir / OUTPUT_MD).write_text(markdown(report), encoding="utf-8")
        return 0
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        details = getattr(error, "details", {})
        blocked = {"stage": STAGE, "decision": BLOCKED, "scope": {"artifact_only": True},
                   "input_validation": {"status": "blocked", "failure_stage": current}, "artifact_coverage": {},
                   "candidate_topology": {}, "native_error_direction": {}, "matched_control_analysis": {},
                   "compatible_positive_margin": {}, "projection_bias_analysis": {}, "centroid_readout_analysis": {},
                   "representation_movement_analysis": {}, "clean_hard_correct_contrast": {}, "family_analysis": {},
                   "cohort_analysis": {}, "polarity_conditioned_analysis": {},
                   "diagnosis": {"error_type": type(error).__name__, "error": str(error), "failure_stage": current,
                                 "traceback": traceback.format_exc(), "missing_ids": details.get("missing_ids", []),
                                 "extra_ids": details.get("extra_ids", []), "duplicate_ids": details.get("duplicate_ids", []),
                                 "schema_mismatch": details.get("schema_mismatch", {}),
                                 "offending_candidate_rows": details.get("offending_candidate_rows", []),
                                 "contaminated_rows_accidentally_included": details.get("contaminated_rows_accidentally_included", [])},
                   "stage183_gate": {"authorized_next_stage": "", "training_authorized": False},
                   "limitations": ["Validation failure prevents a scientific localization conclusion."],
                   "safety_policy": {"model_import": False, "checkpoint_load": False, "forward": False, "training": False}}
        (output_dir / OUTPUT_JSON).write_text(json.dumps(blocked, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / OUTPUT_MD).write_text(f"# Stage182-B blocked\n\n**Decision:** `{BLOCKED}`\n\nFailure stage: `{current}`\n\nError: `{error}`\n", encoding="utf-8")
        for name, filename in OUTPUTS.items():
            rows = [{"decision": BLOCKED, "criterion": current, "observed_value": str(error), "threshold": "validation_pass", "passed": False,
                     "authorized_next_stage": ""}] if name == "gate" else []
            write_csv(output_dir / filename, rows, SCHEMAS[name])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
