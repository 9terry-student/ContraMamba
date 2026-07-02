"""Stage57: generate a synthetic, non-leaking external bridge dataset.

Stage55 found that VitaminC external failure is dominated by NOT_ENTITLED
overprediction/collapse, not label mapping, prediction path, truncation,
composer, or Stage45C recovery failure. Stage56 froze a non-leaking bridge
design with five bridge families (entity_attribute, numeric_comparison,
temporal_comparison, lexical_paraphrase, distractor_evidence) meant to teach
entitlement under open-domain-like wording without ever training, tuning, or
selecting on VitaminC (or any other external dataset).

This script is a pure data generator + audit report. It:
  - builds every claim/evidence pair from synthetic templates and synthetic
    entities/values (invented names, invented places, invented numbers),
  - never reads VitaminC/Climate-FEVER text, labels, ids, or examples,
  - never reads Stage53A/Stage55 external example rows,
  - writes a JSONL dataset plus a JSON+Markdown audit report.

It does not train, evaluate, or tune anything, and it does not modify any
existing data or results file.

Outputs (created, parent directories made as needed):
  - data/stage57_nonleaking_external_bridge.jsonl
  - results/stage57_nonleaking_external_bridge_audit.json
  - results/stage57_nonleaking_external_bridge_audit.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_JSONL = ROOT / "data" / "stage57_nonleaking_external_bridge.jsonl"
DEFAULT_AUDIT_JSON = ROOT / "results" / "stage57_nonleaking_external_bridge_audit.json"
DEFAULT_AUDIT_MD = ROOT / "results" / "stage57_nonleaking_external_bridge_audit.md"
DEFAULT_SEED = 57
DEFAULT_EXAMPLES_PER_FAMILY_LABEL = 40

# Only read for design-metadata confirmation; never for text/labels/examples.
STAGE56_DESIGN_JSON = ROOT / "results" / "stage56_nonleaking_external_transfer_bridge_design.json"

STAGE57_DECISION_READY = "STAGE57_NONLEAKING_EXTERNAL_BRIDGE_DATA_READY"

# REFUTE=0, NOT_ENTITLED=1, SUPPORT=2 (matches v5.FINAL_LABEL_TO_ID / _S28E_LABEL_TO_ID
# used elsewhere in this repo, e.g. scripts/build_stage31c_coverage_entailment_aux.py).
FINAL_LABEL_TO_ID: dict[str, int] = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

BRIDGE_FAMILIES = [
    "entity_attribute_bridge",
    "numeric_comparison_bridge",
    "temporal_comparison_bridge",
    "lexical_paraphrase_bridge",
    "distractor_evidence_bridge",
]

FAMILY_SHORT_CODE = {
    "entity_attribute_bridge": "eab",
    "numeric_comparison_bridge": "ncb",
    "temporal_comparison_bridge": "tcb",
    "lexical_paraphrase_bridge": "lpb",
    "distractor_evidence_bridge": "deb",
}

LEAKAGE_POLICY: dict[str, Any] = {
    "vitaminc_text_used_for_generation": False,
    "vitaminc_labels_used_for_generation": False,
    "external_metrics_used_for_threshold_tuning": False,
    "synthetic_only": True,
}

STAGE57_GENERATION_SOURCE = "synthetic_nonleaking_bridge"
STAGE57_LEAKAGE_POLICY_TAG = "no_vitaminc_text_or_labels_used"

# ---------------------------------------------------------------------------
# Synthetic vocabularies. All names/places/orgs are invented for this script;
# none are copied from VitaminC, Climate-FEVER, or any other external corpus.
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Sera", "Talen", "Mira", "Doran", "Kestrel", "Ilana", "Vasho", "Renna",
    "Corvin", "Yulan", "Petra", "Osric", "Lira", "Bramwell", "Ansel",
    "Thessaly", "Norin", "Calla", "Ember", "Halden", "Isolde", "Rasmin",
    "Verona", "Oskar", "Meret", "Fennick", "Adair", "Solene", "Torvald", "Nyla",
]
LAST_NAMES = [
    "Marrow", "Vance", "Ashcombe", "Delmar", "Korrin", "Whitlock", "Sable",
    "Ferro", "Lindqvist", "Brackwater", "Osgood", "Thane", "Callahan",
    "Merrow", "Stavros", "Undset", "Kestner", "Halvard", "Orrin", "Vesque",
    "Marlow", "Anders", "Kovac", "Reyven", "Solari", "Brenning", "Dashiell",
    "Fen", "Aldric", "Tarn",
]
PLACE_NAMES = [
    "Norvale", "Kaldris", "Fennmoor", "Aurelith", "Brackenholt", "Solvane",
    "Iskerra", "Wrenfield", "Dunmara", "Corvashire", "Talwick", "Ombervale",
    "Rustholm", "Vintershed", "Halloway", "Merenth", "Adderbrook",
    "Sillowgate", "Karrowdale", "Emberlyn",
]
ORG_ROOTS = [
    "Verdant", "Kestrel", "Northgate", "Solace", "Ironvale", "Brightwell",
    "Halcyon", "Amberline", "Frostpine", "Cobalt",
]
ORG_SUFFIXES = [
    "Holdings", "Dynamics", "Collective", "Foundry", "Systems", "Works",
    "Institute", "Guild", "Union", "Assembly",
]
PROFESSIONS = [
    "cellist", "sculptor", "chemist", "novelist", "cyclist", "cartographer",
    "surgeon", "glassblower", "archivist", "aviator", "botanist",
    "choreographer", "locksmith", "typographer", "falconer",
]
ARTIFACT_CATEGORIES = [
    "studio album", "documentary film", "graphic novel", "opera",
    "video game", "symphony", "anthology", "stage musical", "short film",
    "concept album",
]
CRIME_TYPES = [
    "fraud", "embezzlement", "smuggling", "forgery", "bribery", "tax evasion",
]
TOPIC_WORDS = [
    "renewable energy", "urban planning", "marine biology", "textile design",
    "aviation safety", "folk music", "glacier research", "public health",
    "robotics", "ceramics",
]
NUMERIC_DOMAINS = [
    ("quarterly revenue", "$ million", 10, 500),
    ("reported case count", "cases", 50, 5000),
    ("membership count", "members", 100, 20000),
    ("review score", "points", 40, 100),
    ("event attendance", "attendees", 200, 50000),
]
NUMERIC_OPS = ["less than", "more than", "at least", "fewer than", "exactly"]
TEMPORAL_EVENTS = [
    ("founded", "organization"),
    ("released", "artifact"),
    ("born", "person"),
    ("died", "person"),
]
TEMPORAL_OPS = ["before", "after", "as_of"]


def distribute(total: int, buckets: int) -> list[int]:
    base, extra = divmod(total, buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


class NamePool:
    """Deterministic, non-repeating supplier of synthetic entity names."""

    def __init__(self, rng: random.Random, combos: list[tuple[str, str]], sep: str = " "):
        pool = list(combos)
        rng.shuffle(pool)
        self._pool = [f"{a}{sep}{b}" for a, b in pool]
        self._i = 0

    def next(self) -> str:
        name = self._pool[self._i % len(self._pool)]
        self._i += 1
        return name


class CyclicPool:
    """Deterministic, order-preserved-but-shuffled cycling supplier."""

    def __init__(self, rng: random.Random, items: list[Any]):
        pool = list(items)
        rng.shuffle(pool)
        self._pool = pool
        self._i = 0

    def next(self) -> Any:
        item = self._pool[self._i % len(self._pool)]
        self._i += 1
        return item


def make_row(
    *,
    family: str,
    subtype: str,
    pair_id: str,
    intervention_type: str,
    claim: str,
    evidence: str,
    label_name: str,
    frame_compatible_label: int,
    predicate_covered_label: int,
    sufficiency_label: int,
    polarity_label: str,
    primary_failure_type: str,
) -> dict[str, Any]:
    row_id = f"{pair_id}__{intervention_type}"
    return {
        "id": row_id,
        "pair_id": pair_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": label_name,
        "label": FINAL_LABEL_TO_ID[label_name],
        "frame_compatible_label": frame_compatible_label,
        "predicate_covered_label": predicate_covered_label,
        "sufficiency_label": sufficiency_label,
        "polarity_label": polarity_label,
        "primary_failure_type": primary_failure_type,
        "intervention_type": intervention_type,
        "stage57_family": "bridge",
        "stage57_bridge_family": family,
        "stage57_subtype": subtype,
        "stage57_generation_source": STAGE57_GENERATION_SOURCE,
        "stage57_leakage_policy": STAGE57_LEAKAGE_POLICY_TAG,
    }


# ---------------------------------------------------------------------------
# Family 1: entity_attribute_bridge
# ---------------------------------------------------------------------------
_EAB_SUBTYPES = ["birth", "founding", "location", "profession", "artifact_category"]


def build_entity_attribute_bridge(rng: random.Random, n_pairs: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    other_orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    other_places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    other_categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    artifacts = CyclicPool(rng, PLACE_NAMES)  # invented single-word artifact titles

    rows: list[dict[str, Any]] = []
    subtype_counts = distribute(n_pairs, len(_EAB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_EAB_SUBTYPES, subtype_counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage57_eab_{subtype}_{idx:04d}"

            if subtype == "birth":
                name = people.next()
                year = rng.randint(1900, 2015)
                city = places.next()
                other_year = year + rng.choice([-40, -12, 7, 25, 33])
                claim = f"{name} was born in {year}."
                support_ev = f"{name} was born in {year} in {city}."
                refute_ev = f"{name} was born in {other_year} in {city}."
                ne_ev = f"{name} is a {professions.next()} based in {city}."
            elif subtype == "founding":
                org = orgs.next()
                year = rng.randint(1900, 2015)
                city = places.next()
                other_year = year + rng.choice([-30, -9, 11, 22, 41])
                claim = f"{org} was founded in {year}."
                support_ev = f"{org} was founded in {year} by {people.next()}."
                refute_ev = f"{org} was founded in {other_year}."
                ne_ev = f"{org} operates across {city} and {other_places.next()}."
            elif subtype == "location":
                org = orgs.next()
                city = places.next()
                other_city = other_places.next()
                claim = f"{org} is located in {city}."
                support_ev = f"{org}, based in {city}, employs staff across the region."
                refute_ev = f"{org} is located in {other_city}."
                ne_ev = f"{org} was founded in {rng.randint(1900, 2015)}."
            elif subtype == "profession":
                name = people.next()
                profession = professions.next()
                other_profession = other_professions.next()
                city = places.next()
                claim = f"{name} is a {profession}."
                support_ev = f"{name} works as a {profession} in {city}."
                refute_ev = f"{name} is a {other_profession}."
                ne_ev = f"{name} attended a public event in {city}."
            else:  # artifact_category
                artifact = artifacts.next()
                category = categories.next()
                other_category = other_categories.next()
                name = people.next()
                year = rng.randint(1950, 2020)
                claim = f"{artifact} is a {category}."
                support_ev = f"{artifact}, released in {year}, is a {category} by {name}."
                refute_ev = f"{artifact} is a {other_category}."
                ne_ev = f"{artifact} received favorable coverage in {places.next()}."

            rows.append(make_row(
                family="entity_attribute_bridge", subtype=subtype, pair_id=pair_id,
                intervention_type="bridge_direct", claim=claim, evidence=support_ev,
                label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label="SUPPORT", primary_failure_type="none",
            ))
            rows.append(make_row(
                family="entity_attribute_bridge", subtype=subtype, pair_id=pair_id,
                intervention_type="bridge_attribute_contradiction", claim=claim, evidence=refute_ev,
                label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label="REFUTE", primary_failure_type="polarity",
            ))
            rows.append(make_row(
                family="entity_attribute_bridge", subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_missing_{subtype}_predicate", claim=claim, evidence=ne_ev,
                label_name="NOT_ENTITLED", frame_compatible_label=1, predicate_covered_label=0,
                sufficiency_label=0, polarity_label="NONE", primary_failure_type="sufficiency",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 2: numeric_comparison_bridge
# ---------------------------------------------------------------------------
def _numeric_threshold(rng: random.Random, evidence_number: int, op: str, want_support: bool) -> int:
    margin = max(1, round(evidence_number * rng.uniform(0.05, 0.25)))
    if op in ("less than", "fewer than"):
        return evidence_number + margin if want_support else max(1, evidence_number - margin)
    if op == "more than":
        return max(0, evidence_number - margin) if want_support else evidence_number + margin
    if op == "at least":
        return max(0, evidence_number - margin) if want_support else evidence_number + margin
    # exactly
    return evidence_number if want_support else evidence_number + margin


def build_numeric_comparison_bridge(rng: random.Random, n_pairs: int) -> list[dict[str, Any]]:
    subjects = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    other_topics = CyclicPool(rng, TOPIC_WORDS)
    places = CyclicPool(rng, PLACE_NAMES)
    ops = CyclicPool(rng, NUMERIC_OPS)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    for idx in range(1, n_pairs + 1):
        subject = subjects.next()
        domain_name, unit, low, high = domains.next()
        op = ops.next()
        evidence_number = rng.randint(low, high)

        support_threshold = _numeric_threshold(rng, evidence_number, op, want_support=True)
        refute_threshold = _numeric_threshold(rng, evidence_number, op, want_support=False)

        claim_support = f"{subject} reported {op} {support_threshold} {unit} in {domain_name}."
        claim_refute = f"{subject} reported {op} {refute_threshold} {unit} in {domain_name}."
        evidence_text = f"{subject} recorded {evidence_number} {unit} in {domain_name} this period."
        ne_evidence = f"{subject} discussed its outlook on {other_topics.next()} in {places.next()}, without citing figures."

        pair_id = f"stage57_ncb_{domain_name.replace(' ', '_')}_{idx:04d}"
        rows.append(make_row(
            family="numeric_comparison_bridge", subtype=domain_name, pair_id=pair_id,
            intervention_type="bridge_direct", claim=claim_support, evidence=evidence_text,
            label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
            sufficiency_label=1, polarity_label="SUPPORT", primary_failure_type="none",
        ))
        rows.append(make_row(
            family="numeric_comparison_bridge", subtype=domain_name, pair_id=pair_id,
            intervention_type="bridge_numeric_contradiction", claim=claim_refute, evidence=evidence_text,
            label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
            sufficiency_label=1, polarity_label="REFUTE", primary_failure_type="polarity",
        ))
        rows.append(make_row(
            family="numeric_comparison_bridge", subtype=domain_name, pair_id=pair_id,
            intervention_type="bridge_no_comparable_number", claim=claim_support, evidence=ne_evidence,
            label_name="NOT_ENTITLED", frame_compatible_label=1, predicate_covered_label=0,
            sufficiency_label=0, polarity_label="NONE", primary_failure_type="sufficiency",
        ))
    return rows


# ---------------------------------------------------------------------------
# Family 3: temporal_comparison_bridge
# ---------------------------------------------------------------------------
def _temporal_threshold(rng: random.Random, evidence_year: int, op: str, want_support: bool) -> int:
    margin = rng.randint(2, 15)
    if op == "before":
        return evidence_year + margin if want_support else max(1900, evidence_year - margin)
    if op == "after":
        return max(1900, evidence_year - margin) if want_support else evidence_year + margin
    # as_of: event must have occurred by threshold year (evidence_year <= threshold)
    return evidence_year + margin if want_support else max(1900, evidence_year - margin)


def build_temporal_comparison_bridge(rng: random.Random, n_pairs: int) -> list[dict[str, Any]]:
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    artifacts = CyclicPool(rng, PLACE_NAMES)
    places = CyclicPool(rng, PLACE_NAMES)
    ops = CyclicPool(rng, TEMPORAL_OPS)
    events = CyclicPool(rng, TEMPORAL_EVENTS)

    op_phrase = {"before": "before", "after": "after", "as_of": "as of"}
    rows: list[dict[str, Any]] = []
    for idx in range(1, n_pairs + 1):
        verb, entity_type = events.next()
        op = ops.next()
        if entity_type == "organization":
            subject = orgs.next()
        elif entity_type == "artifact":
            subject = artifacts.next()
        else:
            subject = people.next()

        evidence_year = rng.randint(1900, 2015)
        support_threshold = _temporal_threshold(rng, evidence_year, op, want_support=True)
        refute_threshold = _temporal_threshold(rng, evidence_year, op, want_support=False)

        claim_support = f"{subject} was {verb} {op_phrase[op]} {support_threshold}."
        claim_refute = f"{subject} was {verb} {op_phrase[op]} {refute_threshold}."
        evidence_text = f"{subject} was {verb} in {evidence_year}."
        ne_evidence = f"{subject} was widely covered in {places.next()}, with no confirmed year for when it was {verb}."

        pair_id = f"stage57_tcb_{verb}_{op}_{idx:04d}"
        rows.append(make_row(
            family="temporal_comparison_bridge", subtype=f"{verb}_{op}", pair_id=pair_id,
            intervention_type="bridge_direct", claim=claim_support, evidence=evidence_text,
            label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
            sufficiency_label=1, polarity_label="SUPPORT", primary_failure_type="none",
        ))
        rows.append(make_row(
            family="temporal_comparison_bridge", subtype=f"{verb}_{op}", pair_id=pair_id,
            intervention_type="bridge_temporal_contradiction", claim=claim_refute, evidence=evidence_text,
            label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
            sufficiency_label=1, polarity_label="REFUTE", primary_failure_type="polarity",
        ))
        rows.append(make_row(
            family="temporal_comparison_bridge", subtype=f"{verb}_{op}", pair_id=pair_id,
            intervention_type="bridge_no_comparable_date", claim=claim_support, evidence=ne_evidence,
            label_name="NOT_ENTITLED", frame_compatible_label=1, predicate_covered_label=0,
            sufficiency_label=0, polarity_label="NONE", primary_failure_type="sufficiency",
        ))
    return rows


# ---------------------------------------------------------------------------
# Family 4: lexical_paraphrase_bridge
# ---------------------------------------------------------------------------
_LPB_PATTERNS = ["sentence_duration", "album_ordinal", "founding_threshold", "competition_genre"]


def build_lexical_paraphrase_bridge(rng: random.Random, n_pairs: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    artifacts = CyclicPool(rng, PLACE_NAMES)
    other_artifacts = CyclicPool(rng, PLACE_NAMES)
    crimes = CyclicPool(rng, CRIME_TYPES)
    topics = CyclicPool(rng, TOPIC_WORDS)
    places = CyclicPool(rng, PLACE_NAMES)
    ordinals = ["second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    ordinal_pool = CyclicPool(rng, ordinals)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_pairs, len(_LPB_PATTERNS))
    idx = 0
    for pattern, count in zip(_LPB_PATTERNS, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage57_lpb_{pattern}_{idx:04d}"

            if pattern == "sentence_duration":
                name = people.next()
                years = rng.randint(2, 20)
                other_years = years + rng.choice([-6, -3, 3, 5, 9])
                crime = crimes.next()
                claim = f"{name} was sentenced to {years} years for {crime}."
                support_ev = f"{name} had a {years}-year sentence for {crime}."
                refute_ev = f"{name} had a {max(1, other_years)}-year sentence for {crime}."
                ne_ev = f"{name} is currently on trial for {crime}."
            elif pattern == "album_ordinal":
                artifact = artifacts.next()
                name = people.next()
                ordinal = ordinal_pool.next()
                other_artifact = other_artifacts.next()
                claim = f"{artifact} is the {ordinal} studio album by {name}."
                support_ev = f"{artifact} is an album released by {name}."
                refute_ev = f"{artifact} is a documentary film directed by {name}, titled after {other_artifact}."
                ne_ev = f"{name} released new material in {places.next()}."
            elif pattern == "founding_threshold":
                org = orgs.next()
                year = rng.randint(1900, 2010)
                later_year = year + rng.randint(3, 25)
                other_year = later_year + rng.randint(1, 15)
                claim = f"{org} was founded before {later_year}."
                support_ev = f"{org} was founded in {year}."
                refute_ev = f"{org} was founded in {other_year}."
                ne_ev = f"{org} operates across {places.next()}."
            else:  # competition_genre
                show = artifacts.next()
                topic = topics.next()
                claim = f"{show} is a competition."
                support_ev = f"{show} is a televised competition show broadcast in {places.next()}."
                refute_ev = f"{show} is a documentary series about {topic}."
                ne_ev = f"{show} premiered in {rng.randint(1970, 2020)}."

            rows.append(make_row(
                family="lexical_paraphrase_bridge", subtype=pattern, pair_id=pair_id,
                intervention_type="bridge_paraphrase", claim=claim, evidence=support_ev,
                label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label="SUPPORT", primary_failure_type="none",
            ))
            rows.append(make_row(
                family="lexical_paraphrase_bridge", subtype=pattern, pair_id=pair_id,
                intervention_type="bridge_paraphrase_contradiction", claim=claim, evidence=refute_ev,
                label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label="REFUTE", primary_failure_type="polarity",
            ))
            rows.append(make_row(
                family="lexical_paraphrase_bridge", subtype=pattern, pair_id=pair_id,
                intervention_type="bridge_paraphrase_insufficient", claim=claim, evidence=ne_ev,
                label_name="NOT_ENTITLED", frame_compatible_label=1, predicate_covered_label=0,
                sufficiency_label=0, polarity_label="NONE", primary_failure_type="sufficiency",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 5: distractor_evidence_bridge (NOT_ENTITLED only)
# ---------------------------------------------------------------------------
_DEB_SUBTYPES = ["missing_predicate", "wrong_subject_same_domain", "partial_list_or_count", "related_insufficient_context"]


def build_distractor_evidence_bridge(rng: random.Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    other_orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    topics = CyclicPool(rng, TOPIC_WORDS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_DEB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_DEB_SUBTYPES, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage57_deb_{subtype}_{idx:04d}"

            if subtype == "missing_predicate":
                name = people.next()
                profession = professions.next()
                year = rng.randint(1950, 2015)
                claim = f"{name} was born in {year}."
                evidence = f"{name} is a {profession} who lives in {places.next()}."
                frame_compatible, predicate_covered, sufficiency = 1, 0, 0
            elif subtype == "wrong_subject_same_domain":
                org = orgs.next()
                wrong_org = other_orgs.next()
                year = rng.randint(1950, 2015)
                claim = f"{org} was founded in {year}."
                evidence = f"{wrong_org} was founded in {year}."
                frame_compatible, predicate_covered, sufficiency = 0, 0, 1
            elif subtype == "partial_list_or_count":
                org = orgs.next()
                total = rng.randint(20, 200)
                claim = f"{org} has {total} member organizations."
                evidence = f"{org}'s member organizations include entities based in {places.next()} and {places.next()}."
                frame_compatible, predicate_covered, sufficiency = 1, 1, 0
            else:  # related_insufficient_context
                name = people.next()
                topic = topics.next()
                profession = professions.next()
                claim = f"{name} is a {profession}."
                evidence = f"{name} spoke at a conference on {topic} in {places.next()}."
                frame_compatible, predicate_covered, sufficiency = 1, 0, 0

            rows.append(make_row(
                family="distractor_evidence_bridge", subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_{subtype}", claim=claim, evidence=evidence,
                label_name="NOT_ENTITLED", frame_compatible_label=frame_compatible,
                predicate_covered_label=predicate_covered, sufficiency_label=sufficiency,
                polarity_label="NONE", primary_failure_type="sufficiency" if sufficiency == 0 else "frame",
            ))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_dataset(seed: int, examples_per_family_label: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    rows.extend(build_entity_attribute_bridge(random.Random(rng.random()), examples_per_family_label))
    rows.extend(build_numeric_comparison_bridge(random.Random(rng.random()), examples_per_family_label))
    rows.extend(build_temporal_comparison_bridge(random.Random(rng.random()), examples_per_family_label))
    rows.extend(build_lexical_paraphrase_bridge(random.Random(rng.random()), examples_per_family_label))
    rows.extend(build_distractor_evidence_bridge(random.Random(rng.random()), examples_per_family_label))

    seen_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise RuntimeError(f"duplicate id generated: {row['id']}")
        seen_ids.add(row["id"])
    return rows


def build_audit(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    label_counts = Counter(row["final_label"] for row in rows)
    family_counts = Counter(row["stage57_bridge_family"] for row in rows)
    family_label_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        fam = row["stage57_bridge_family"]
        lbl = row["final_label"]
        family_label_counts.setdefault(fam, Counter())[lbl] += 1
    family_label_counts = {fam: dict(counter) for fam, counter in family_label_counts.items()}

    stage56_confirmed = STAGE56_DESIGN_JSON.exists()

    return {
        "stage": "Stage57",
        "decision": STAGE57_DECISION_READY,
        "source_stage56_design": str(STAGE56_DESIGN_JSON.relative_to(ROOT)).replace("\\", "/"),
        "stage56_design_file_found": stage56_confirmed,
        "generation_config": {
            "seed": args.seed,
            "examples_per_family_label": args.examples_per_family_label,
        },
        "output_jsonl": str(Path(args.output_jsonl)).replace("\\", "/"),
        "total_rows": len(rows),
        "counts_by_label": dict(label_counts),
        "counts_by_bridge_family": dict(family_counts),
        "counts_by_bridge_family_and_label": family_label_counts,
        "bridge_families": BRIDGE_FAMILIES,
        "label_mapping": FINAL_LABEL_TO_ID,
        "leakage_policy": LEAKAGE_POLICY,
        "notes": [
            "This dataset is synthetic training/diagnostic data only. It is NOT an "
            "external evaluation result and must not be reported as VitaminC or any "
            "other external-benchmark metric.",
            "This dataset must not be mixed with corrupted time_swap rows from "
            "data/controlled_v5_v3.jsonl; temporal bridge rows here are freshly "
            "generated and independent of that corruption.",
            "No VitaminC/Climate-FEVER text, labels, ids, or examples were read or "
            "used to produce any row in this file.",
        ],
        "recommended_next_stage": {
            "stage": "Stage58",
            "name": "bridge dataset static audit / schema check",
            "scope": "Validate schema, label balance, and non-leakage of this bridge "
                     "dataset before any training uses it.",
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def render_markdown(audit: dict[str, Any]) -> str:
    lines = []
    lines.append("# Stage57: Non-leaking External Bridge Dataset Generator Audit")
    lines.append("")
    lines.append(f"**Decision:** `{audit['decision']}`")
    lines.append("")
    lines.append(f"- Output JSONL: `{audit['output_jsonl']}`")
    lines.append(f"- Total rows: {audit['total_rows']}")
    lines.append(f"- Seed: {audit['generation_config']['seed']}")
    lines.append(f"- Examples per family/label: {audit['generation_config']['examples_per_family_label']}")
    lines.append(f"- Stage56 design file found: {audit['stage56_design_file_found']}")
    lines.append("")
    lines.append("## Counts by label")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|---|---|")
    for label, count in sorted(audit["counts_by_label"].items()):
        lines.append(f"| {label} | {count} |")
    lines.append("")
    lines.append("## Counts by bridge family")
    lines.append("")
    lines.append("| Family | Count |")
    lines.append("|---|---|")
    for family, count in sorted(audit["counts_by_bridge_family"].items()):
        lines.append(f"| {family} | {count} |")
    lines.append("")
    lines.append("## Counts by bridge family x label")
    lines.append("")
    lines.append("| Family | REFUTE | NOT_ENTITLED | SUPPORT |")
    lines.append("|---|---|---|---|")
    for family in audit["bridge_families"]:
        counts = audit["counts_by_bridge_family_and_label"].get(family, {})
        lines.append(
            f"| {family} | {counts.get('REFUTE', 0)} | {counts.get('NOT_ENTITLED', 0)} | {counts.get('SUPPORT', 0)} |"
        )
    lines.append("")
    lines.append("## Leakage policy")
    lines.append("")
    for key, value in audit["leakage_policy"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in audit["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Recommended next stage")
    lines.append("")
    next_stage = audit["recommended_next_stage"]
    lines.append(f"- **{next_stage['stage']}**: {next_stage['name']} — {next_stage['scope']}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--output-audit-json", type=str, default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--output-audit-md", type=str, default=str(DEFAULT_AUDIT_MD))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--examples-per-family-label", type=int, default=DEFAULT_EXAMPLES_PER_FAMILY_LABEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = generate_dataset(seed=args.seed, examples_per_family_label=args.examples_per_family_label)

    output_jsonl_path = Path(args.output_jsonl)
    write_jsonl(rows, output_jsonl_path)

    audit = build_audit(rows, args)

    output_audit_json_path = Path(args.output_audit_json)
    output_audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    output_audit_md_path = Path(args.output_audit_md)
    output_audit_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_md_path.write_text(render_markdown(audit), encoding="utf-8")

    print(f"[Stage57] wrote {len(rows)} rows to {output_jsonl_path}")
    print(f"[Stage57] wrote audit JSON to {output_audit_json_path}")
    print(f"[Stage57] wrote audit Markdown to {output_audit_md_path}")


if __name__ == "__main__":
    main()
