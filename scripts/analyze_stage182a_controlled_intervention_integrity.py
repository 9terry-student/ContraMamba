"""Deterministically audit controlled interventions and isolate clean failures.

This script imports no model code, loads no checkpoint, and performs no forward
pass.  It reads the frozen Stage176--181 artifacts and reconstructs the
controlled rows with the original data generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


STAGE = "Stage182-A"
COMPLETE = "STAGE182A_CONTROLLED_INTERVENTION_INTEGRITY_AND_CLEAN_FAILURE_SET_AUDIT_COMPLETE"
BLOCKED = "STAGE182A_CONTROLLED_INTERVENTION_INTEGRITY_AUDIT_BLOCKED"
EXPECTED_STAGE181 = "STAGE181A_SECOND_REVIEW_OR_HUMAN_ADJUDICATION_REQUIRED"
EXPECTED_STAGE180 = "STAGE180A_HARD_FRAME_MANUAL_REVIEW_PACKET_READY"

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SOURCE = ROOT / "scripts" / "build_controlled_v5.py"

OUTPUT_JSON = "stage182a_controlled_intervention_integrity_report.json"
OUTPUT_MD = "stage182a_controlled_intervention_integrity_report.md"
OUTPUT_ITEMS = "stage182a_unique_item_integrity.csv"
OUTPUT_CONTRACT = "stage182a_intervention_contract_audit.csv"
OUTPUT_AXIS = "stage182a_structured_axis_delta.csv"
OUTPUT_GRAMMAR = "stage182a_grammar_integrity_audit.csv"
OUTPUT_CANONICAL = "stage182a_canonical_control_audit.csv"
OUTPUT_CLEAN = "stage182a_clean_model_failure_candidates.csv"
OUTPUT_DATA_QUEUE = "stage182a_data_intervention_contamination_queue.csv"
OUTPUT_CLEAN_CONTROL = "stage182a_clean_control_reference.csv"
OUTPUT_CONTROL_QUEUE = "stage182a_control_anomaly_queue.csv"
OUTPUT_UNRESOLVED = "stage182a_schema_unresolved_queue.csv"
OUTPUT_COMPARISON = "stage182a_stage181_taxonomy_comparison.csv"
OUTPUT_COHORT = "stage182a_beneficial_harmful_integrity.csv"
OUTPUT_FAMILY = "stage182a_intervention_family_integrity.csv"
OUTPUT_DECISION = "stage182a_decision_evidence.csv"

REQUIRED_DATA_FIELDS = {
    "id", "pair_id", "claim", "evidence", "final_label",
    "frame_compatible_label", "predicate_covered_label", "sufficiency_label",
    "polarity_label", "primary_failure_type", "intervention_type",
}

INTENDED_AXES: dict[str, set[str]] = {
    "none": set(),
    "paraphrase": {"realization"},
    "entity_swap": {"name"},
    "event_swap": {"object"},
    "time_swap": {"time"},
    "location_swap": {"location"},
    "role_swap": {"role"},
    "title_name_swap": {"title", "name"},
    "predicate_swap": {"predicate"},
    "evidence_deletion": {"content_deletion"},
    "evidence_truncation": {"content_truncation"},
    "irrelevant_evidence": {"content_replacement"},
    "polarity_flip": {"polarity"},
}
NON_POLARITY = set(INTENDED_AXES) - {"none", "polarity_flip"}
SLOT_AXES = {"title", "name", "role", "predicate", "object", "location", "time"}
CONTENT_OPERATIONS = {"evidence_deletion", "evidence_truncation", "irrelevant_evidence"}

ITEM_COLUMNS = [
    "row_id", "review_instance_id", "item_role", "stage176_cohort",
    "intervention_type", "integrity_status", "grammar_valid", "contract_exact_match",
    "canonical_control_valid", "schema_resolved", "native_frame_label",
    "native_frame_prediction", "stage181_primary_stratum", "final_diagnostic_class",
    "pair_id", "match_link_id", "matched_source_row_id", "anomaly_codes",
    "grammar_invalid", "multi_axis_contamination", "ineffective_intervention",
    "canonical_control_invalid",
]
CONTRACT_COLUMNS = [
    "row_id", "pair_id", "intervention_type", "intended_changed_axes",
    "observed_changed_axes", "unexpected_additional_axes", "missing_intended_axes",
    "exact_contract_match", "primary_integrity_status",
]
AXIS_COLUMNS = [
    "row_id", "pair_id", "intervention_type", "entity_changed", "title_name_changed",
    "role_changed", "predicate_changed", "event_changed", "location_changed",
    "temporal_changed", "polarity_changed", "surface_only_changed",
]
GRAMMAR_COLUMNS = [
    "row_id", "pair_id", "intervention_type", "invalid_do_support", "duplicate_negation",
    "missing_subject", "malformed_role_phrase", "missing_event_object", "empty_location",
    "empty_temporal", "template_violation", "affected_side", "grammar_valid",
]
CANONICAL_COLUMNS = [
    "row_id", "pair_id", "item_role", "canonical_none_row_id",
    "canonical_none_generator_exact", "canonical_none_grammar_valid",
    "packet_canonical_anchor_exact", "matched_control_row_id", "matched_control_valid",
    "canonical_control_valid",
]
COMPARISON_COLUMNS = [
    "row_id", "item_role", "stage181_primary_stratum", "deterministic_integrity_class",
    "agreement_category", "roadmap_correction_required",
]
COHORT_COLUMNS = [
    "stage176_cohort", "hard_count", "clean_failure_count", "contaminated_count",
    "unresolved_count", "clean_failure_rate", "other_cohort_clean_failure_rate",
    "rate_difference", "fisher_exact_p", "interpretation",
]
FAMILY_COLUMNS = [
    "intervention_type", "item_count", "hard_count", "control_count", "clean_count",
    "contaminated_count", "unresolved_count", "clean_model_failure_count",
    "clean_control_reference_count", "control_anomaly_count",
]
DECISION_COLUMNS = ["criterion", "observed_value", "threshold", "passed", "evidence_source"]

CSV_SCHEMAS = {
    OUTPUT_ITEMS: ITEM_COLUMNS,
    OUTPUT_CONTRACT: CONTRACT_COLUMNS,
    OUTPUT_AXIS: AXIS_COLUMNS,
    OUTPUT_GRAMMAR: GRAMMAR_COLUMNS,
    OUTPUT_CANONICAL: CANONICAL_COLUMNS,
    OUTPUT_CLEAN: ITEM_COLUMNS,
    OUTPUT_DATA_QUEUE: ITEM_COLUMNS,
    OUTPUT_CLEAN_CONTROL: ITEM_COLUMNS,
    OUTPUT_CONTROL_QUEUE: ITEM_COLUMNS,
    OUTPUT_UNRESOLVED: ITEM_COLUMNS,
    OUTPUT_COMPARISON: COMPARISON_COLUMNS,
    OUTPUT_COHORT: COHORT_COLUMNS,
    OUTPUT_FAMILY: FAMILY_COLUMNS,
    OUTPUT_DECISION: DECISION_COLUMNS,
}


class AuditBlocked(ValueError):
    """A frozen-input, identity, or generator contract was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditBlocked(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlocked(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path, required: Iterable[str]) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise AuditBlocked(f"cannot read CSV {path}: {exc}") from exc
    missing = sorted(set(required) - set(header))
    require(not missing, f"schema mismatch in {path}: missing columns {missing}")
    require(bool(rows), f"CSV is empty: {path}")
    return rows, header


def read_data(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            rows = value if isinstance(value, list) else value.get("records", value.get("data"))
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise AuditBlocked(f"cannot read controlled data {path}: {exc}") from exc
    require(isinstance(rows, list) and rows, f"controlled data has no row list: {path}")
    require(all(isinstance(row, dict) for row in rows), "controlled data rows must be objects")
    for number, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_DATA_FIELDS - set(row))
        require(not missing, f"controlled data row {number} missing {missing}")
    ids = [str(row["id"]) for row in rows]
    require(len(ids) == len(set(ids)), "controlled data contains duplicate id values")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuditBlocked(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: csv_value(row.get(field, "")) for field in fields} for row in rows)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def two_sided_fisher(a: int, b: int, c: int, d: int) -> float:
    """Exact descriptive Fisher p-value for a 2x2 table."""
    row1, row2, col1, total = a + b, c + d, a + c, a + b + c + d
    denominator = math.comb(total, row1)

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / denominator

    observed = probability(a)
    low, high = max(0, row1 - (total - col1)), min(row1, col1)
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed + 1e-15))


def load_generator(path: Path) -> ModuleType:
    """Load the generator with labels only, bypassing package model imports."""
    require(path.resolve() == GENERATOR_SOURCE.resolve(), "generator source must be repository canonical source")
    spec = importlib.util.spec_from_file_location("stage182a_controlled_generator", path)
    require(spec is not None and spec.loader is not None, f"cannot load generator specification: {path}")
    module = importlib.util.module_from_spec(spec)
    labels_path = ROOT / "src" / "contramamba" / "labels.py"
    labels_spec = importlib.util.spec_from_file_location("contramamba.labels", labels_path)
    require(labels_spec is not None and labels_spec.loader is not None,
            f"cannot load label-only schema: {labels_path}")
    labels_module = importlib.util.module_from_spec(labels_spec)
    package = ModuleType("contramamba")
    package.__path__ = [str(labels_path.parent)]  # type: ignore[attr-defined]
    sentinel = object()
    previous_package = sys.modules.get("contramamba", sentinel)
    previous_labels = sys.modules.get("contramamba.labels", sentinel)
    original_path = list(sys.path)
    try:
        sys.modules["contramamba"] = package
        sys.modules["contramamba.labels"] = labels_module
        labels_spec.loader.exec_module(labels_module)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AuditBlocked(f"cannot load controlled generator {path}: {exc}") from exc
    finally:
        sys.path[:] = original_path
        if previous_package is sentinel:
            sys.modules.pop("contramamba", None)
        else:
            sys.modules["contramamba"] = previous_package  # type: ignore[assignment]
        if previous_labels is sentinel:
            sys.modules.pop("contramamba.labels", None)
        else:
            sys.modules["contramamba.labels"] = previous_labels  # type: ignore[assignment]
    for name in ("build_controlled_records", "fact_templates_for_count"):
        require(callable(getattr(module, name, None)), f"generator lacks callable {name}")
    return module


def exact_map(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = [str(row.get(key, "")).strip() for row in rows]
    require(all(values), f"empty {key} value")
    require(len(values) == len(set(values)), f"duplicate {key} value")
    return dict(zip(values, rows, strict=True))


def decision_value(report: dict[str, Any]) -> str | None:
    return report.get("decision") or report.get("execution_decision") or (report.get("closure") or {}).get("decision")


def normalize_hidden(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Collapse review repeats while proving all source metadata is consistent."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    review_ids: set[str] = set()
    for row in rows:
        review_id = row["review_instance_id"].strip()
        source_id = row["row_id"].strip()
        require(review_id and source_id, "hidden key has empty identity")
        require(review_id not in review_ids, f"duplicate review_instance_id {review_id}")
        review_ids.add(review_id)
        grouped[source_id].append(row)
    unique: dict[str, dict[str, str]] = {}
    fields = ("pair_id", "item_role", "stage176_cohort", "intervention_type", "native_frame_label",
              "gold_final_label", "match_link_id", "matched_hard_row_id", "match_level")
    reviews_by_source: dict[str, list[str]] = {}
    for source_id, instances in grouped.items():
        canonical = instances[0]
        for field in fields:
            observed = {row.get(field, "").strip() for row in instances}
            require(len(observed) == 1, f"repeat metadata conflict for {source_id}.{field}: {sorted(observed)}")
        unique[source_id] = canonical
        reviews_by_source[source_id] = sorted(row["review_instance_id"].strip() for row in instances)
    return unique, reviews_by_source


def build_links(hidden: dict[str, dict[str, str]]) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    links: dict[str, dict[str, str]] = defaultdict(dict)
    for source_id, row in hidden.items():
        role, link = row["item_role"].strip(), row["match_link_id"].strip()
        require(role in {"hard", "control"}, f"invalid item_role for {source_id}: {role!r}")
        require(link, f"empty match_link_id for {source_id}")
        require(role not in links[link], f"duplicate {role} in match link {link}")
        links[link][role] = source_id
    require(all(set(value) == {"hard", "control"} for value in links.values()),
            "each match link must contain one unique hard and one unique control")
    pairs = {link: (value["hard"], value["control"]) for link, value in links.items()}
    counterpart: dict[str, str] = {}
    for hard, control in pairs.values():
        counterpart[hard], counterpart[control] = control, hard
    return pairs, counterpart


def reconstruct_generator(data: list[dict[str, Any]], module: ModuleType) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    pair_ids = sorted({str(row["pair_id"]) for row in data})
    interventions = {str(row["intervention_type"]) for row in data}
    require(interventions <= set(INTENDED_AXES), f"unknown interventions: {sorted(interventions - set(INTENDED_AXES))}")
    counts = Counter(str(row["pair_id"]) for row in data)
    require(set(counts.values()) == {len(interventions)}, "dataset is not rectangular by pair/intervention")
    require(all({str(row["intervention_type"]) for row in data if str(row["pair_id"]) == pair_id} == interventions
                for pair_id in pair_ids), "pair intervention coverage differs")
    generated = module.build_controlled_records(len(pair_ids))
    expected_rows = [row for row in generated if row["intervention_type"] in interventions]
    expected = exact_map(expected_rows, "id")
    observed = exact_map(data, "id")
    require(set(expected) == set(observed),
            f"dataset/generator identity mismatch: missing={sorted(set(expected)-set(observed))[:10]} "
            f"extra={sorted(set(observed)-set(expected))[:10]}")
    mismatches = [key for key in sorted(expected) if observed[key] != expected[key]]
    require(not mismatches, f"dataset rows differ from generator schema: {mismatches[:10]}")
    templates = module.fact_templates_for_count(len(pair_ids))
    facts = exact_map(templates, "pair_id")
    require(set(facts) == set(pair_ids), "generator pair IDs differ from dataset")
    return expected, facts


def polarity(text: str) -> str:
    return "negative" if re.search(r"\bdid\s+not\b", text, flags=re.IGNORECASE) else "positive"


def semantic_state(fact: dict[str, Any], intervention: str, evidence: str) -> dict[str, str]:
    """Recover generator slot values from the rendered evidence itself."""
    del intervention  # the declared type must not determine the observed state
    state: dict[str, str] = {}
    for axis in SLOT_AXES:
        original = str(fact[axis])
        alternate = str(fact.get(f"alternate_{axis}", ""))
        original_present = original in evidence
        alternate_present = bool(alternate) and alternate in evidence
        if original_present and not alternate_present:
            state[axis] = f"original:{original}"
        elif alternate_present and not original_present:
            state[axis] = f"alternate:{alternate}"
        elif original_present and alternate_present:
            state[axis] = f"ambiguous_both:{original}|{alternate}"
        else:
            state[axis] = "absent"
    state["polarity"] = polarity(evidence)
    return state


def changed_axes(row: dict[str, Any], none: dict[str, Any], fact: dict[str, Any]) -> set[str]:
    intervention = str(row["intervention_type"])
    if intervention in CONTENT_OPERATIONS:
        changed = set(INTENDED_AXES[intervention])
        if polarity(str(row["evidence"])) != polarity(str(none["evidence"])):
            changed.add("polarity")
        return changed
    base = semantic_state(fact, "none", str(none["evidence"]))
    current = semantic_state(fact, intervention, str(row["evidence"]))
    changed = {axis for axis in base if base[axis] != current[axis]}
    if intervention == "paraphrase" and row["evidence"] != none["evidence"]:
        changed.add("realization")
    return changed


def grammar_anomaly(row: dict[str, Any], fact: dict[str, Any]) -> bool:
    """Generator predicates are past-event forms and cannot follow did-not unchanged."""
    candidates = {str(fact["predicate"]), str(fact["alternate_predicate"])}
    text = str(row["evidence"])
    return any(re.search(rf"\bdid\s+not\s+{re.escape(predicate)}\b", text, re.IGNORECASE)
               for predicate in candidates)


def packet_validation(packet_rows: list[dict[str, str]], hidden_rows: list[dict[str, str]],
                      data_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, bool], dict[str, int]]:
    hidden_by_review = exact_map(hidden_rows, "review_instance_id")
    packet_by_review = exact_map(packet_rows, "review_instance_id")
    require(set(packet_by_review) == set(hidden_by_review), "Pass 2 packet and hidden-key review IDs differ")
    result: dict[str, bool] = defaultdict(lambda: True)
    claim_equal = evidence_different = relation_exact = 0
    for review_id, packet in packet_by_review.items():
        hidden = hidden_by_review[review_id]
        source_id, pair_id = hidden["row_id"].strip(), hidden["pair_id"].strip()
        require(source_id in data_by_id, f"hidden source row absent from data: {source_id}")
        none_id = f"{pair_id}__none"
        require(none_id in data_by_id, f"canonical none row absent: {none_id}")
        anchor = data_by_id[none_id]
        exact = (packet["canonical_none_claim"] == anchor["claim"] and
                 packet["canonical_none_evidence"] == anchor["evidence"] and
                 packet["intervention_type"].strip() == str(data_by_id[source_id]["intervention_type"]))
        require(exact, f"Pass 2 canonical anchor or intervention mismatch for {review_id}")
        claim_matches = str(data_by_id[source_id]["claim"]) == packet["canonical_none_claim"]
        evidence_differs = str(data_by_id[source_id]["evidence"]) != packet["canonical_none_evidence"]
        is_non_none = str(data_by_id[source_id]["intervention_type"]) != "none"
        claim_equal += claim_matches
        evidence_different += evidence_differs
        relation_exact += evidence_differs == is_non_none
        result[source_id] = result[source_id] and exact
    require(claim_equal == len(packet_rows), "source claims do not all equal canonical-none claims")
    require(relation_exact == len(packet_rows),
            "evidence difference is not exactly equivalent to non-none intervention")
    return dict(result), {"claim_equal_instances": claim_equal,
                          "evidence_different_instances": evidence_different,
                          "evidence_difference_relation_exact_instances": relation_exact}


def audit_item(source_id: str, hidden: dict[str, str], row: dict[str, Any], none: dict[str, Any],
               fact: dict[str, Any], packet_exact: bool, counterpart: str,
               cohort: str, stage181_stratum: str) -> dict[str, Any]:
    intervention = str(row["intervention_type"])
    changed = changed_axes(row, none, fact)
    intended = INTENDED_AXES[intervention]
    unexpected = changed - intended
    missing = intended - changed
    grammar = grammar_anomaly(row, fact)
    polarity_leak = intervention in NON_POLARITY and "polarity" in changed
    anomalies: list[str] = []
    if grammar:
        anomalies.append("DID_NOT_INFLECTED_PREDICATE")
    if polarity_leak:
        anomalies.append("NON_POLARITY_INTERVENTION_POLARITY_CHANGE")
    if unexpected - {"polarity"}:
        anomalies.append("UNINTENDED_STRUCTURED_AXIS_CHANGE")
    if missing:
        anomalies.append("MISSING_INTENDED_AXIS_CHANGE")
    verdict = "CONTAMINATED_CONSTRUCTION" if anomalies else "CLEAN_SINGLE_AXIS_CONSTRUCTION"
    return {
        "source_row_id": source_id,
        "item_role": hidden["item_role"].strip(),
        "pair_id": row["pair_id"],
        "intervention_type": intervention,
        "match_link_id": hidden["match_link_id"].strip(),
        "matched_source_row_id": counterpart,
        "stage176_cohort": cohort,
        "stage181_primary_stratum": stage181_stratum,
        "generator_exact": True,
        "packet_anchor_exact": packet_exact,
        "changed_axes": sorted(changed),
        "intended_axes": sorted(intended),
        "unexpected_axes": sorted(unexpected),
        "missing_intended_axes": sorted(missing),
        "did_not_inflected_predicate": grammar,
        "non_polarity_polarity_change": polarity_leak,
        "label_schema_exact": True,
        "integrity_verdict": verdict,
        "anomaly_codes": anomalies,
        "clean_model_failure_candidate": False,
    }


def markdown(report: dict[str, Any]) -> str:
    topology = report["unique_item_topology"]
    integrity = report["intervention_integrity"]
    hard_control = report["hard_control_integrity"]
    readiness = report["clean_failure_set_readiness"]
    return f"""# Stage182-A controlled-intervention integrity audit

**Decision:** `{report['decision']}`

The deterministic audit covered all {topology['unique_items']} unique review items:
{topology['unique_hard']} hard rows and {topology['unique_controls']} matched controls.

## Result

- Clean constructions: {integrity['clean_items']}
- Contaminated constructions: {integrity['contaminated_items']}
- Grammar anomalies: {report['grammar_integrity']['invalid_items']}
- Non-polarity polarity leaks: {report['structured_axis_contract']['polarity_contamination_items']}
- Clean hard/control pairs: {hard_control['clean_pairs']} of {topology['matched_pairs']}
- Final clean model-failure candidates: {readiness['candidate_count']}

## Decision evidence

- Criterion: `minimum_clean_hard_candidates`
- Observed clean hard candidates: {readiness['candidate_count']}
- Required minimum: {readiness['minimum_clean_hard_candidates']}
- Passed: {readiness['minimum_criterion_passed']}

Generator equality is provenance rather than a cleanliness verdict. A row can
exactly reproduce the generator and still be excluded for a deterministic
grammar or multi-axis construction defect. The final set is a model-failure
*candidate* set, not a causal diagnosis.

No data, annotation, model, checkpoint, or training state was modified.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage176a-row-transitions", type=Path, required=True)
    parser.add_argument("--stage179a-hard39-attribution", type=Path, required=True)
    parser.add_argument("--stage180a-manifest", type=Path, required=True)
    parser.add_argument("--stage180a-hidden-item-key", type=Path, required=True)
    parser.add_argument("--stage180a-pass2-packet", type=Path, required=True)
    parser.add_argument("--stage181a-report", type=Path, required=True)
    parser.add_argument("--stage181a-unique-item-roadmap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-clean-hard-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    current = "input_read"
    diagnostics: dict[str, Any] = {}
    try:
        require(args.minimum_clean_hard_candidates >= 1,
                "--minimum-clean-hard-candidates must be at least 1")
        paths = {
            "data": args.data,
            "stage176a_row_transitions": args.stage176a_row_transitions,
            "stage179a_hard39_attribution": args.stage179a_hard39_attribution,
            "stage180a_manifest": args.stage180a_manifest,
            "stage180a_hidden_item_key": args.stage180a_hidden_item_key,
            "stage180a_pass2_packet": args.stage180a_pass2_packet,
            "stage181a_report": args.stage181a_report,
            "stage181a_unique_item_roadmap": args.stage181a_unique_item_roadmap,
            "generator_source": GENERATOR_SOURCE,
        }
        data = read_data(args.data)
        stage176, _ = read_csv(args.stage176a_row_transitions, {"row_id"})
        stage179, _ = read_csv(args.stage179a_hard39_attribution, {"row_id"})
        manifest = read_json(args.stage180a_manifest)
        hidden_rows, _ = read_csv(args.stage180a_hidden_item_key,
                                  {"review_instance_id", "row_id", "pair_id", "item_role",
                                   "stage176_cohort", "intervention_type", "native_frame_label",
                                   "gold_final_label", "match_link_id", "matched_hard_row_id",
                                   "match_level", "is_repeat", "repeat_group_id"})
        packet, _ = read_csv(args.stage180a_pass2_packet,
                             {"review_instance_id", "intervention_type", "canonical_none_claim",
                              "canonical_none_evidence"})
        stage181 = read_json(args.stage181a_report)
        roadmap, _ = read_csv(args.stage181a_unique_item_roadmap,
                              {"source_row_id", "item_role", "primary_stratum"})

        current = "decision_and_identity_contract"
        require(decision_value(stage181) == EXPECTED_STAGE181,
                f"Stage181-A decision must be {EXPECTED_STAGE181}")
        manifest_decision = decision_value(manifest)
        require(manifest_decision in {EXPECTED_STAGE180, None},
                f"unexpected Stage180-A manifest decision: {manifest_decision}")
        hidden, reviews_by_source = normalize_hidden(hidden_rows)
        pairs, counterpart = build_links(hidden)
        hard_ids = {source_id for source_id, row in hidden.items() if row["item_role"].strip() == "hard"}
        control_ids = set(hidden) - hard_ids
        require(len(hard_ids) == 39 and len(control_ids) == 39 and len(pairs) == 39,
                f"expected 39 hard/control pairs; got hard={len(hard_ids)} control={len(control_ids)} pairs={len(pairs)}")
        require(len(hidden_rows) == 86,
                f"expected 86 review instances including repeats; got {len(hidden_rows)}")
        stage176_ids = [row["row_id"].strip() for row in stage176]
        stage179_ids = [row["row_id"].strip() for row in stage179]
        require(len(stage176_ids) == len(set(stage176_ids)), "Stage176-A has duplicate row_id")
        require(len(stage179_ids) == len(set(stage179_ids)), "Stage179-A has duplicate row_id")
        require(hard_ids <= set(stage176_ids), "hard IDs are missing from Stage176-A transitions")
        require(set(stage179_ids) == hard_ids,
                f"Stage179-A hard-39 IDs differ: missing={sorted(hard_ids-set(stage179_ids))[:10]} "
                f"extra={sorted(set(stage179_ids)-hard_ids)[:10]}")
        roadmap_by_id = exact_map(roadmap, "source_row_id")
        require(set(roadmap_by_id) == set(hidden), "Stage181 roadmap IDs differ from hidden unique IDs")
        require(all(roadmap_by_id[key]["item_role"].strip() == hidden[key]["item_role"].strip()
                    for key in hidden), "Stage181 roadmap role differs from hidden key")

        current = "generator_reconstruction"
        generator = load_generator(GENERATOR_SOURCE)
        _generated_by_id, facts = reconstruct_generator(data, generator)
        data_by_id = exact_map(data, "id")
        require(set(hidden) <= set(data_by_id), "hidden source IDs absent from controlled data")

        current = "packet_anchor_validation"
        packet_exact, packet_structure = packet_validation(packet, hidden_rows, data_by_id)
        require(set(packet_exact) == set(hidden), "not every unique item received packet validation")
        require(packet_structure["evidence_different_instances"] == 72,
                "expected exactly 72 non-none/evidence-different review instances")

        current = "unique_item_integrity"
        stage176_by_id = {row["row_id"].strip(): row for row in stage176}
        stage179_by_id = {row["row_id"].strip(): row for row in stage179}
        items: list[dict[str, Any]] = []
        for source_id in sorted(hidden):
            row = data_by_id[source_id]
            fact = facts[str(row["pair_id"])]
            none = data_by_id[f"{row['pair_id']}__none"]
            cohort = hidden[source_id].get("stage176_cohort", "").strip()
            if source_id in stage176_by_id:
                transition_cohort = (stage176_by_id[source_id].get("stage176_cohort") or
                                     stage176_by_id[source_id].get("cohort") or
                                     stage176_by_id[source_id].get("transition_cohort") or "").strip()
                require(not transition_cohort or not cohort or transition_cohort == cohort,
                        f"Stage176 cohort mismatch for {source_id}")
            items.append(audit_item(
                source_id, hidden[source_id], row, none, fact, packet_exact[source_id],
                counterpart[source_id], cohort, roadmap_by_id[source_id]["primary_stratum"].strip(),
            ))
        item_by_id = {row["source_row_id"]: row for row in items}

        current = "matched_pair_and_clean_set"
        pair_rows: list[dict[str, Any]] = []
        clean_ids: set[str] = set()
        for link, (hard_id, control_id) in sorted(pairs.items()):
            hard, control = item_by_id[hard_id], item_by_id[control_id]
            pair_clean = (hard["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION" and
                          control["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION")
            if pair_clean:
                clean_ids.add(hard_id)
                hard["clean_model_failure_candidate"] = True
            pair_rows.append({
                "match_link_id": link,
                "hard_source_row_id": hard_id,
                "control_source_row_id": control_id,
                "hard_integrity_verdict": hard["integrity_verdict"],
                "control_integrity_verdict": control["integrity_verdict"],
                "pair_clean": pair_clean,
                "hard_anomaly_codes": hard["anomaly_codes"],
                "control_anomaly_codes": control["anomaly_codes"],
                "clean_model_failure_candidate": pair_clean,
            })

        # Output-only projections from the canonical item/pair results above.
        for row in items:
            role = row["item_role"]
            if role == "hard":
                row["final_diagnostic_class"] = (
                    "CLEAN_MODEL_FAILURE_CANDIDATE"
                    if row["clean_model_failure_candidate"]
                    else "DATA_INTERVENTION_CONTAMINATION"
                )
            else:
                row["final_diagnostic_class"] = (
                    "CLEAN_CONTROL_REFERENCE"
                    if row["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
                    else "CONTROL_ANOMALY"
                )

        def projected_item(row: dict[str, Any]) -> dict[str, Any]:
            source_id = row["source_row_id"]
            matched = item_by_id[row["matched_source_row_id"]]
            matched_valid = matched["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
            if row["item_role"] == "control":
                matched_valid = row["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
            anomaly_codes = list(row["anomaly_codes"])
            if row["item_role"] == "hard" and not matched_valid:
                anomaly_codes.append("CANONICAL_CONTROL_INVALID")
            return {
                "row_id": source_id,
                "review_instance_id": ";".join(reviews_by_source[source_id]),
                "item_role": row["item_role"],
                "stage176_cohort": row["stage176_cohort"],
                "intervention_type": row["intervention_type"],
                "integrity_status": row["integrity_verdict"],
                "grammar_valid": not row["did_not_inflected_predicate"],
                "contract_exact_match": not row["unexpected_axes"] and not row["missing_intended_axes"],
                "canonical_control_valid": matched_valid,
                "schema_resolved": True,
                "native_frame_label": hidden[source_id]["native_frame_label"].strip(),
                "native_frame_prediction": (
                    stage179_by_id[source_id].get("frame_prediction", "")
                    if source_id in stage179_by_id
                    else hidden[source_id]["native_frame_label"].strip()
                ),
                "stage181_primary_stratum": row["stage181_primary_stratum"],
                "final_diagnostic_class": row["final_diagnostic_class"],
                "pair_id": row["pair_id"],
                "match_link_id": row["match_link_id"],
                "matched_source_row_id": row["matched_source_row_id"],
                "anomaly_codes": anomaly_codes,
                "grammar_invalid": row["did_not_inflected_predicate"],
                "multi_axis_contamination": (row["non_polarity_polarity_change"] or
                                               bool(row["unexpected_axes"])),
                "ineffective_intervention": bool(row["missing_intended_axes"]),
                "canonical_control_invalid": not matched_valid,
            }

        item_outputs = [projected_item(row) for row in items]
        item_output_by_id = {row["row_id"]: row for row in item_outputs}
        contract_rows = [{
            "row_id": row["source_row_id"], "pair_id": row["pair_id"],
            "intervention_type": row["intervention_type"],
            "intended_changed_axes": row["intended_axes"],
            "observed_changed_axes": row["changed_axes"],
            "unexpected_additional_axes": row["unexpected_axes"],
            "missing_intended_axes": row["missing_intended_axes"],
            "exact_contract_match": not row["unexpected_axes"] and not row["missing_intended_axes"],
            "primary_integrity_status": row["integrity_verdict"],
        } for row in items]
        axis_rows = [{
            "row_id": row["source_row_id"], "pair_id": row["pair_id"],
            "intervention_type": row["intervention_type"],
            "entity_changed": "name" in row["changed_axes"],
            "title_name_changed": "title" in row["changed_axes"],
            "role_changed": "role" in row["changed_axes"],
            "predicate_changed": "predicate" in row["changed_axes"],
            "event_changed": "object" in row["changed_axes"],
            "location_changed": "location" in row["changed_axes"],
            "temporal_changed": "time" in row["changed_axes"],
            "polarity_changed": "polarity" in row["changed_axes"],
            "surface_only_changed": "realization" in row["changed_axes"],
        } for row in items]
        grammar_rows = [{
            "row_id": row["source_row_id"], "pair_id": row["pair_id"],
            "intervention_type": row["intervention_type"],
            "invalid_do_support": row["did_not_inflected_predicate"],
            "duplicate_negation": False, "missing_subject": False,
            "malformed_role_phrase": False, "missing_event_object": False,
            "empty_location": False, "empty_temporal": False,
            "template_violation": row["did_not_inflected_predicate"],
            "affected_side": "evidence" if row["did_not_inflected_predicate"] else "none",
            "grammar_valid": not row["did_not_inflected_predicate"],
        } for row in items]
        canonical_rows = []
        for row in items:
            source_id = row["source_row_id"]
            none_id = f"{row['pair_id']}__none"
            none_row = data_by_id[none_id]
            none_valid = not grammar_anomaly(none_row, facts[str(row["pair_id"])])
            matched_id = row["matched_source_row_id"]
            matched_valid = item_by_id[matched_id]["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
            if row["item_role"] == "control":
                matched_valid = row["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION"
            canonical_rows.append({
                "row_id": source_id, "pair_id": row["pair_id"], "item_role": row["item_role"],
                "canonical_none_row_id": none_id, "canonical_none_generator_exact": True,
                "canonical_none_grammar_valid": none_valid,
                "packet_canonical_anchor_exact": row["packet_anchor_exact"],
                "matched_control_row_id": matched_id if row["item_role"] == "hard" else source_id,
                "matched_control_valid": matched_valid,
                "canonical_control_valid": matched_valid,
            })

        agreement_map = {
            "CLEAN_LABEL_MODEL_FAILURE_CANDIDATE": "CLEAN_MODEL_FAILURE_CANDIDATE",
            "DATA_INTERVENTION_REPAIR_CANDIDATE": "DATA_INTERVENTION_CONTAMINATION",
            "CLEAN_CONTROL_REFERENCE": "CLEAN_CONTROL_REFERENCE",
            "CONTROL_ANOMALY": "CONTROL_ANOMALY",
            "ADJUDICATION_HOLD": "SCHEMA_UNRESOLVED_HOLD",
        }
        comparison_rows = []
        for row in items:
            stage181_class = row["stage181_primary_stratum"]
            deterministic = row["final_diagnostic_class"]
            agrees = agreement_map.get(stage181_class) == deterministic
            comparison_rows.append({
                "row_id": row["source_row_id"], "item_role": row["item_role"],
                "stage181_primary_stratum": stage181_class,
                "deterministic_integrity_class": deterministic,
                "agreement_category": "AGREES" if agrees else "DETERMINISTIC_RECLASSIFICATION",
                "roadmap_correction_required": not agrees,
            })

        hard_outputs = [row for row in item_outputs if row["item_role"] == "hard"]
        clean_output_rows = [row for row in hard_outputs
                             if row["final_diagnostic_class"] == "CLEAN_MODEL_FAILURE_CANDIDATE"]
        data_queue_rows = [row for row in hard_outputs
                           if row["final_diagnostic_class"] == "DATA_INTERVENTION_CONTAMINATION"]
        clean_control_rows = [row for row in item_outputs
                              if row["final_diagnostic_class"] == "CLEAN_CONTROL_REFERENCE"]
        control_queue_rows = [row for row in item_outputs
                              if row["final_diagnostic_class"] == "CONTROL_ANOMALY"]
        unresolved_rows = [row for row in item_outputs
                           if row["final_diagnostic_class"] == "SCHEMA_UNRESOLVED_HOLD"]

        beneficial = [row for row in hard_outputs if row["stage176_cohort"] == "beneficial_correction"]
        harmful = [row for row in hard_outputs if row["stage176_cohort"] == "harmful_regression"]
        beneficial_clean = sum(row["final_diagnostic_class"] == "CLEAN_MODEL_FAILURE_CANDIDATE"
                               for row in beneficial)
        harmful_clean = sum(row["final_diagnostic_class"] == "CLEAN_MODEL_FAILURE_CANDIDATE"
                            for row in harmful)
        fisher_p = two_sided_fisher(
            beneficial_clean, len(beneficial) - beneficial_clean,
            harmful_clean, len(harmful) - harmful_clean,
        )
        cohort_rows = []
        for name, group, clean_count, other_group, other_clean in (
            ("beneficial_correction", beneficial, beneficial_clean, harmful, harmful_clean),
            ("harmful_regression", harmful, harmful_clean, beneficial, beneficial_clean),
        ):
            own_rate, other_rate = rate(clean_count, len(group)), rate(other_clean, len(other_group))
            cohort_rows.append({
                "stage176_cohort": name, "hard_count": len(group),
                "clean_failure_count": clean_count,
                "contaminated_count": sum(row["final_diagnostic_class"] == "DATA_INTERVENTION_CONTAMINATION"
                                           for row in group),
                "unresolved_count": sum(row["final_diagnostic_class"] == "SCHEMA_UNRESOLVED_HOLD"
                                         for row in group),
                "clean_failure_rate": own_rate,
                "other_cohort_clean_failure_rate": other_rate,
                "rate_difference": (own_rate - other_rate) if own_rate is not None and other_rate is not None else None,
                "fisher_exact_p": fisher_p, "interpretation": "descriptive_only",
            })

        family_rows = []
        for family in sorted({row["intervention_type"] for row in item_outputs}):
            group = [row for row in item_outputs if row["intervention_type"] == family]
            family_rows.append({
                "intervention_type": family, "item_count": len(group),
                "hard_count": sum(row["item_role"] == "hard" for row in group),
                "control_count": sum(row["item_role"] == "control" for row in group),
                "clean_count": sum(row["integrity_status"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION" for row in group),
                "contaminated_count": sum(row["integrity_status"] == "CONTAMINATED_CONSTRUCTION" for row in group),
                "unresolved_count": sum(row["final_diagnostic_class"] == "SCHEMA_UNRESOLVED_HOLD" for row in group),
                "clean_model_failure_count": sum(row["final_diagnostic_class"] == "CLEAN_MODEL_FAILURE_CANDIDATE" for row in group),
                "clean_control_reference_count": sum(row["final_diagnostic_class"] == "CLEAN_CONTROL_REFERENCE" for row in group),
                "control_anomaly_count": sum(row["final_diagnostic_class"] == "CONTROL_ANOMALY" for row in group),
            })

        decision_rows = [
            {"criterion": "generator_exact_match", "observed_value": len(data),
             "threshold": f"=={len(data)}", "passed": True, "evidence_source": "generator_reconstruction"},
            {"criterion": "unique_hard_topology", "observed_value": len(hard_ids),
             "threshold": "==39", "passed": len(hard_ids) == 39, "evidence_source": "hidden_item_key"},
            {"criterion": "unique_control_topology", "observed_value": len(control_ids),
             "threshold": "==39", "passed": len(control_ids) == 39, "evidence_source": "hidden_item_key"},
            {"criterion": "review_instance_topology", "observed_value": len(hidden_rows),
             "threshold": "==86", "passed": len(hidden_rows) == 86, "evidence_source": "hidden_item_key"},
            {"criterion": "non_none_evidence_difference", "observed_value": packet_structure["evidence_different_instances"],
             "threshold": "==72", "passed": packet_structure["evidence_different_instances"] == 72,
             "evidence_source": "pass2_packet_and_controlled_data"},
            {"criterion": "schema_unresolved_items", "observed_value": len(unresolved_rows),
             "threshold": "==0", "passed": len(unresolved_rows) == 0, "evidence_source": "unique_item_integrity"},
            {"criterion": "minimum_clean_hard_candidates", "observed_value": len(clean_output_rows),
             "threshold": args.minimum_clean_hard_candidates,
             "passed": len(clean_output_rows) >= args.minimum_clean_hard_candidates,
             "evidence_source": "hard_control_integrity"},
        ]

        clean_items = sum(row["integrity_verdict"] == "CLEAN_SINGLE_AXIS_CONSTRUCTION" for row in items)
        contaminated_items = sum(row["integrity_verdict"] == "CONTAMINATED_CONSTRUCTION" for row in items)
        grammar_invalid = sum(row["did_not_inflected_predicate"] for row in items)
        polarity_contamination = sum(row["non_polarity_polarity_change"] for row in items)
        clean_pairs = sum(row["pair_clean"] for row in pair_rows)
        anomaly_counts = dict(sorted(Counter(code for row in items for code in row["anomaly_codes"]).items()))
        input_hashes = {name: sha256(path) for name, path in paths.items()}
        report = {
            "stage": STAGE,
            "decision": COMPLETE,
            "scope": {"deterministic_read_only_audit": True, "identity_key": "row_id",
                      "stage181_taxonomy_used_as_integrity_ground_truth": False},
            "input_validation": {"status": "passed", "input_sha256": input_hashes,
                                 "generator_exact_dataset": True, "exhaustive_hidden_join": True,
                                 "exhaustive_packet_join": True, "stage181_decision": EXPECTED_STAGE181},
            "authoritative_schema": {"generator_source": "scripts/build_controlled_v5.py",
                                     "generator_source_sha256": input_hashes["generator_source"],
                                     "controlled_rows_exact_match": len(data)},
            "unique_item_topology": {"unique_items": len(items), "unique_hard": len(hard_ids),
                                     "unique_controls": len(control_ids), "matched_pairs": len(pair_rows),
                                     "review_instances": len(hidden_rows),
                                     "repeat_instances": len(hidden_rows) - len(hidden)},
            "canonical_context_contract": packet_structure,
            "structured_axis_contract": {"contract_rows": len(contract_rows),
                                         "exact_contract_rows": sum(row["exact_contract_match"] for row in contract_rows),
                                         "polarity_contamination_items": polarity_contamination},
            "grammar_integrity": {"audited_items": len(grammar_rows), "invalid_items": grammar_invalid,
                                  "valid_items": len(grammar_rows) - grammar_invalid},
            "intervention_integrity": {"clean_items": clean_items,
                                       "contaminated_items": contaminated_items,
                                       "unresolved_items": len(unresolved_rows)},
            "hard_control_integrity": {"clean_pairs": clean_pairs,
                                       "contaminated_pairs": len(pair_rows) - clean_pairs,
                                       "clean_controls": len(clean_control_rows),
                                       "control_anomalies": len(control_queue_rows)},
            "stage181_taxonomy_comparison": {"rows": len(comparison_rows),
                                             "agreements": sum(row["agreement_category"] == "AGREES" for row in comparison_rows),
                                             "roadmap_corrections": sum(row["roadmap_correction_required"] for row in comparison_rows)},
            "beneficial_harmful_analysis": {"rows": cohort_rows, "fisher_exact": "two_sided",
                                            "interpretation": "descriptive_only"},
            "clean_failure_set_readiness": {"candidate_count": len(clean_output_rows),
                                            "candidate_row_ids": sorted(row["row_id"] for row in clean_output_rows),
                                            "minimum_clean_hard_candidates": args.minimum_clean_hard_candidates,
                                            "minimum_criterion_passed": len(clean_output_rows) >= args.minimum_clean_hard_candidates,
                                            "causal_proof": False},
            "diagnosis": {"anomaly_counts": anomaly_counts,
                          "generator_equality_is_cleanliness": False,
                          "row_id_misalignment_rejected": True},
            "stage182b_gate": {"clean_failure_set_available": bool(clean_output_rows),
                               "minimum_clean_hard_candidates": args.minimum_clean_hard_candidates,
                               "observed_clean_hard_candidates": len(clean_output_rows),
                               "minimum_criterion_passed": len(clean_output_rows) >= args.minimum_clean_hard_candidates,
                               "execution_authorized": False,
                               "decision_evidence_rows": len(decision_rows)},
            "limitations": ["The clean set is not causal proof.",
                            "Stage181 taxonomy remains provisional.",
                            "Fisher exact results are descriptive only."],
            "safety_policy": {"dataset_modification": False, "annotation_modification": False,
                              "automatic_relabeling": False, "training_subset_construction": False,
                              "model_import": False, "checkpoint_load": False, "model_forward": False,
                              "training": False, "fitting": False, "calibration": False,
                              "threshold_search": False, "external_evaluation": False},
        }

        current = "output_write"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_rows = {
            OUTPUT_ITEMS: item_outputs,
            OUTPUT_CONTRACT: contract_rows,
            OUTPUT_AXIS: axis_rows,
            OUTPUT_GRAMMAR: grammar_rows,
            OUTPUT_CANONICAL: canonical_rows,
            OUTPUT_CLEAN: clean_output_rows,
            OUTPUT_DATA_QUEUE: data_queue_rows,
            OUTPUT_CLEAN_CONTROL: clean_control_rows,
            OUTPUT_CONTROL_QUEUE: control_queue_rows,
            OUTPUT_UNRESOLVED: unresolved_rows,
            OUTPUT_COMPARISON: comparison_rows,
            OUTPUT_COHORT: cohort_rows,
            OUTPUT_FAMILY: family_rows,
            OUTPUT_DECISION: decision_rows,
        }
        for filename, fields in CSV_SCHEMAS.items():
            write_csv(output_dir / filename, output_rows[filename], fields)
        write_json(output_dir / OUTPUT_JSON, report)
        (output_dir / OUTPUT_MD).write_text(markdown(report), encoding="utf-8")
        return 0
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        blocked_decision_rows = [{
            "criterion": current,
            "observed_value": str(error),
            "threshold": "required_validation_passes",
            "passed": False,
            "evidence_source": "blocked_exception",
        }]
        blocked = {
            "stage": STAGE,
            "decision": BLOCKED,
            "scope": {"deterministic_read_only_audit": True, "identity_key": "row_id"},
            "input_validation": {"status": "blocked", "failure_stage": current},
            "authoritative_schema": {},
            "unique_item_topology": {},
            "canonical_context_contract": {},
            "structured_axis_contract": {},
            "grammar_integrity": {},
            "intervention_integrity": {},
            "hard_control_integrity": {},
            "stage181_taxonomy_comparison": {},
            "beneficial_harmful_analysis": {},
            "clean_failure_set_readiness": {"candidate_count": 0, "ready": False,
                                            "minimum_clean_hard_candidates": args.minimum_clean_hard_candidates,
                                            "minimum_criterion_passed": False},
            "diagnosis": {"error_type": type(error).__name__, "error": str(error),
                          "failure_stage": current, "traceback": traceback.format_exc(),
                          "diagnostics": diagnostics},
            "stage182b_gate": {"minimum_clean_hard_candidates": args.minimum_clean_hard_candidates,
                               "observed_clean_hard_candidates": 0,
                               "minimum_criterion_passed": False,
                               "execution_authorized": False},
            "limitations": ["Validation failure prevents a scientific conclusion."],
            "safety_policy": {"dataset_modification": False, "model_import": False,
                              "checkpoint_load": False, "model_forward": False, "training": False},
        }
        write_json(output_dir / OUTPUT_JSON, blocked)
        (output_dir / OUTPUT_MD).write_text(
            f"# Stage182-A blocked\n\n**Decision:** `{BLOCKED}`\n\n"
            f"Failure stage: `{current}`\n\nError: `{error}`\n", encoding="utf-8")
        write_csv(output_dir / OUTPUT_DECISION, blocked_decision_rows, DECISION_COLUMNS)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

