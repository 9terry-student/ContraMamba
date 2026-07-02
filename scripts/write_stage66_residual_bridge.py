"""Stage67: generate the synthetic Stage66 residual bridge dataset.

Stage65 found that residual VitaminC external failure (after Stage57's
non-leaking bridge) is still dominated by false NOT_ENTITLED on gold
SUPPORT/REFUTE plus a smaller REFUTE/SUPPORT polarity-confusion band and a
bounded false-entitlement tail. Stage66 froze a non-leaking residual bridge
*design* (five families, 720 rows, quotas only) meant to close that residual
gap without ever training, tuning, or generating from VitaminC (or any other
external dataset) text or labels.

This script is a pure data generator + audit report. It:
  - builds every claim/evidence pair from freshly invented synthetic
    templates and synthetic entities/values (invented names, invented
    places, invented organizations, invented artifact titles, invented
    numbers/dates),
  - reads the Stage66 design JSON only for its quota/family taxonomy (row
    counts, label plan, family names, target-error labels) -- never for
    example text,
  - never reads VitaminC/Climate-FEVER/FEVEROUS text, labels, ids, or
    examples, and never reads Stage65 residual sample rows,
  - never uses data/controlled_v5_v3.jsonl time_swap rows,
  - writes a JSONL dataset plus a JSON+Markdown audit report.

It does not train, evaluate, or tune anything, and it does not modify any
existing data or results file (aside from the new output files it writes,
and only when --overwrite is passed if those files already exist).

Outputs (created, parent directories made as needed):
  - data/stage66_residual_bridge.jsonl
  - results/stage67_stage66_residual_bridge_generation_audit.json
  - results/stage67_stage66_residual_bridge_generation_audit.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_JSONL = ROOT / "data" / "stage66_residual_bridge.jsonl"
DEFAULT_AUDIT_JSON = ROOT / "results" / "stage67_stage66_residual_bridge_generation_audit.json"
DEFAULT_AUDIT_MD = ROOT / "results" / "stage67_stage66_residual_bridge_generation_audit.md"
DEFAULT_DESIGN_JSON = ROOT / "results" / "stage66_residual_bridge_expansion_design.json"
DEFAULT_SEED = 660067

STAGE67_DECISION_READY = "STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_READY"
STAGE67_DECISION_FAILED = "STAGE67_STAGE66_RESIDUAL_BRIDGE_DATA_FAILED"

# REFUTE=0, NOT_ENTITLED=1, SUPPORT=2 (matches v5.FINAL_LABEL_TO_ID used
# elsewhere in this repo, e.g. scripts/write_stage57_nonleaking_external_bridge.py).
FINAL_LABEL_TO_ID: dict[str, int] = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

STAGE66_GENERATION_SOURCE = "synthetic_nonleaking_residual_bridge"
STAGE66_LEAKAGE_POLICY_TAG = "no_vitaminc_text_or_labels_used_taxonomy_only"

# Fields whose *value* is a fixed, mandated policy tag rather than generated
# row content. STAGE66_LEAKAGE_POLICY_TAG intentionally contains the
# substring "vitaminc" (it documents *non*-use), so it is excluded from the
# forbidden-marker text scan; every other field is scanned.
FORBIDDEN_SCAN_EXCLUDE_FIELDS = {"stage66_leakage_policy"}

FORBIDDEN_MARKERS = [
    "vitaminc",
    "vitamin-c",
    "climate_fever",
    "climate-fever",
    "feverous",
    "stage43",
    "stage53",
    "stage55",
    "stage63",
    "stage65",
    "time_swap",
]

REQUIRED_FIELDS = [
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
    "stage66_family",
    "stage66_bridge_family",
    "stage66_subtype",
    "stage66_target_error",
    "stage66_generation_source",
    "stage66_leakage_policy",
]

# Family plan, mirrored from results/stage66_residual_bridge_expansion_design.json
# (quota/taxonomy only -- no example text is read from that file).
FAMILY_PLAN: dict[str, dict[str, Any]] = {
    "support_entitlement_recovery_bridge": {
        "planned_rows": 200,
        "label_plan": {"SUPPORT": 200, "REFUTE": 0, "NOT_ENTITLED": 0},
        "target_error": "false_NE_on_SUPPORT",
    },
    "refute_entitlement_recovery_bridge": {
        "planned_rows": 160,
        "label_plan": {"SUPPORT": 0, "REFUTE": 160, "NOT_ENTITLED": 0},
        "target_error": "false_NE_on_REFUTE",
    },
    "polarity_disambiguation_bridge": {
        "planned_rows": 200,
        "label_plan": {"SUPPORT": 100, "REFUTE": 100, "NOT_ENTITLED": 0},
        "target_error": "REFUTE_SUPPORT_polarity_confusion",
    },
    "numeric_temporal_comparison_bridge": {
        "planned_rows": 120,
        "label_plan": {"SUPPORT": 60, "REFUTE": 60, "NOT_ENTITLED": 0},
        "target_error": "numeric_temporal_comparative_residuals",
    },
    "strict_ne_frame_safety_bridge": {
        "planned_rows": 40,
        "label_plan": {"SUPPORT": 0, "REFUTE": 0, "NOT_ENTITLED": 40},
        "target_error": "false_SUPPORT_or_REFUTE_on_NE",
    },
}
BRIDGE_FAMILY_ORDER = list(FAMILY_PLAN.keys())
STAGE66_INCREMENTAL_LABEL_PLAN = {"SUPPORT": 360, "REFUTE": 320, "NOT_ENTITLED": 40}
STAGE66_TOTAL_ROWS = 720

LEAKAGE_POLICY: dict[str, Any] = {
    "synthetic_only": True,
    "no_vitaminc_text_or_labels_used": True,
    "stage65_residual_samples_used_as_templates": False,
    "time_swap_used": False,
    "external_metrics_used_for_threshold_tuning": False,
}

# ---------------------------------------------------------------------------
# Synthetic vocabularies. Every name/place/org/title below is newly invented
# for this script. None are copied, paraphrased, or templated from VitaminC,
# Climate-FEVER, FEVEROUS, Stage65 residual samples, or Stage57's bridge
# vocabulary; no real-world proper nouns are used.
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Branwen", "Ilse", "Thoric", "Sable", "Meren", "Cassia", "Oswin", "Livia",
    "Torsten", "Amara", "Denholm", "Faye", "Garrick", "Helewise", "Ionna",
    "Jorund", "Kiona", "Lucan", "Mirel", "Nairne", "Orlagh", "Peregrine",
    "Quilla", "Roswitha", "Saskia", "Tavish", "Ulrika", "Varek", "Wrenna",
    "Yorick",
]
LAST_NAMES = [
    "Blackthorn", "Ravensworth", "Corrigan", "Dunwoody", "Elderfield",
    "Farrow", "Grimsby", "Hollowell", "Ingram", "Juniper", "Kestrion",
    "Larkspur", "Moorland", "Nightshade", "Ostrander", "Pemberton",
    "Quarrymoor", "Rathbone", "Stonebridge", "Thistlewood", "Underhill",
    "Vosberg", "Wraithmoor", "Ashgrove", "Briarwood", "Corvane", "Duskfell",
    "Emberfall", "Fenwick", "Graywick",
]
PLACE_NAMES = [
    "Aldenmere", "Brookhaven", "Cindervale", "Dellmoor", "Everwatch",
    "Faelan", "Grovehallow", "Havenridge", "Ironcliff", "Jasperwell",
    "Kettlebrook", "Larkmoor", "Mistfen", "Northshale", "Oakenfold",
    "Pinehollow", "Quillmarsh", "Ridgewater", "Stormhaven", "Thornfield",
]
ORG_ROOTS = [
    "Vantage", "Meridian", "Cascade", "Silverline", "Anchorpoint", "Fernwood",
    "Highmark", "Ledgerock", "Novabright", "Quarrystone",
]
ORG_SUFFIXES = [
    "Alliance", "Consortium", "Cooperative", "Enterprises", "Federation",
    "Laboratories", "Partners", "Syndicate", "Trust", "Ventures",
]
PROFESSIONS = [
    "violinist", "potter", "biologist", "playwright", "marathoner",
    "geologist", "orthopedist", "weaver", "curator", "navigator",
    "horticulturist", "puppeteer", "blacksmith", "calligrapher", "beekeeper",
]
ARTIFACT_CATEGORIES = [
    "live album", "animated feature", "memoir", "chamber opera",
    "tabletop game", "tone poem", "essay collection", "radio drama",
    "one-act play", "remix album",
]
ARTIFACT_TITLES = [
    "Emberfall", "Glasswing", "Driftlight", "Hollowmere", "Ashenvale",
    "Wrenfeather", "Cinderpath", "Moongate", "Rivenshore", "Thistledown",
    "Windmere", "Starfallow", "Duskwood", "Larkspire", "Frostglen",
    "Amberwake", "Nightloom", "Sablewind", "Grovemark", "Ironbloom",
]
PRODUCTS = [
    "solar panels", "water filters", "bicycle frames", "circuit boards",
    "ceramic tiles", "wind turbines", "battery packs", "greenhouse kits",
]
QUALITIES = ["pacing", "craftsmanship", "originality", "structure", "tone", "depth"]
POSITIVE_ADJECTIVES = ["luminous", "assured", "inventive", "graceful", "vivid", "meticulous"]
NUMERIC_DOMAINS = [
    ("annual harvest yield", "tons", 50, 900),
    ("festival ticket sales", "tickets", 200, 20000),
    ("laboratory sample count", "samples", 30, 3000),
    ("critic rating", "points", 40, 100),
    ("subscriber count", "subscribers", 500, 100000),
]

def distribute(total: int, buckets: int) -> list[int]:
    base, extra = divmod(total, buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


class NamePool:
    """Deterministic, non-repeating (until exhausted) supplier of combined names."""

    def __init__(self, rng: Random, combos: list[tuple[str, str]], sep: str = " "):
        pool = list(combos)
        rng.shuffle(pool)
        self._pool = [f"{a}{sep}{b}" for a, b in pool]
        self._i = 0

    def next(self) -> str:
        name = self._pool[self._i % len(self._pool)]
        self._i += 1
        return name

    def next_distinct_from(self, other: Any) -> str:
        name = self.next()
        guard = 0
        while name == other and guard < len(self._pool):
            name = self.next()
            guard += 1
        return name


class CyclicPool:
    """Deterministic, shuffled cycling supplier over a flat item list."""

    def __init__(self, rng: Random, items: list[Any]):
        pool = list(items)
        rng.shuffle(pool)
        self._pool = pool
        self._i = 0

    def next(self) -> Any:
        item = self._pool[self._i % len(self._pool)]
        self._i += 1
        return item

    def next_distinct_from(self, other: Any) -> Any:
        item = self.next()
        guard = 0
        while item == other and guard < len(self._pool):
            item = self.next()
            guard += 1
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
    polarity_label: int,
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
        "stage66_family": family,
        "stage66_bridge_family": family,
        "stage66_subtype": subtype,
        "stage66_target_error": FAMILY_PLAN[family]["target_error"],
        "stage66_generation_source": STAGE66_GENERATION_SOURCE,
        "stage66_leakage_policy": STAGE66_LEAKAGE_POLICY_TAG,
    }


# ---------------------------------------------------------------------------
# Family 1: support_entitlement_recovery_bridge (200 rows, SUPPORT only)
# ---------------------------------------------------------------------------
_SERB_SUBTYPES = [
    "direct_attribute_support",
    "profession_role_support",
    "album_film_work_support",
    "quantity_paraphrase_support",
    "date_release_birth_support",
    "review_sentiment_support",
]
_SERB_FAMILY = "support_entitlement_recovery_bridge"


def build_support_entitlement_recovery_bridge(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    products = CyclicPool(rng, PRODUCTS)
    qualities = CyclicPool(rng, QUALITIES)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_SERB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_SERB_SUBTYPES, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage66_serb_{subtype}_{idx:04d}"

            if subtype == "direct_attribute_support":
                name = people.next()
                place = places.next()
                year = rng.randint(1955, 2012)
                claim = f"{name} holds citizenship of {place}."
                evidence = f"{name} has held {place} citizenship since {year} and resides there year-round."
            elif subtype == "profession_role_support":
                name = people.next()
                profession = professions.next()
                org = orgs.next()
                year = rng.randint(1990, 2020)
                claim = f"{name} is a {profession}."
                evidence = f"{name} has worked as a {profession} at {org} since {year}."
            elif subtype == "album_film_work_support":
                artifact = artifacts.next()
                category = categories.next()
                name = people.next()
                year = rng.randint(1960, 2023)
                claim = f"{artifact} is a {category} by {name}."
                evidence = f"{artifact}, released in {year}, is a {category} created by {name}."
            elif subtype == "quantity_paraphrase_support":
                org = orgs.next()
                product = products.next()
                threshold = rng.randint(100, 5000)
                actual = threshold + rng.randint(50, 500)
                claim = f"{org} sold more than {threshold} {product}."
                evidence = f"{org} sold {actual} {product} last quarter, well above {threshold}."
            elif subtype == "date_release_birth_support":
                if rng.random() < 0.5:
                    name = people.next()
                    year = rng.randint(1930, 2005)
                    place = places.next()
                    claim = f"{name} was born in {year}."
                    evidence = f"{name} came into the world in {year}, in {place}."
                else:
                    artifact = artifacts.next()
                    year = rng.randint(1960, 2023)
                    claim = f"{artifact} was released in {year}."
                    evidence = f"{artifact} debuted in {year} to wide circulation."
            else:  # review_sentiment_support
                artifact = artifacts.next()
                quality = qualities.next()
                claim = f"{artifact} received a positive critical reception."
                evidence = f"Critics praised {artifact} for its {quality}, calling it a genuine triumph."

            rows.append(make_row(
                family=_SERB_FAMILY, subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_support_{subtype}", claim=claim, evidence=evidence,
                label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=1, primary_failure_type="none",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 2: refute_entitlement_recovery_bridge (160 rows, REFUTE only)
# ---------------------------------------------------------------------------
_RERB_SUBTYPES = [
    "wrong_type_refute",
    "wrong_role_refute",
    "wrong_location_refute",
    "wrong_date_refute",
    "wrong_numeric_threshold_refute",
    "wrong_creator_director_refute",
]
_RERB_FAMILY = "refute_entitlement_recovery_bridge"


def build_refute_entitlement_recovery_bridge(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    other_places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    other_categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_RERB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_RERB_SUBTYPES, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage66_rerb_{subtype}_{idx:04d}"

            if subtype == "wrong_type_refute":
                artifact = artifacts.next()
                category = categories.next()
                other_category = other_categories.next_distinct_from(category)
                claim = f"{artifact} is a {category}."
                evidence = f"{artifact} is in fact a {other_category}, not a {category}."
            elif subtype == "wrong_role_refute":
                name = people.next()
                profession = professions.next()
                other_profession = other_professions.next_distinct_from(profession)
                place = places.next()
                claim = f"{name} is a {profession}."
                evidence = f"{name} actually works as a {other_profession} in {place}."
            elif subtype == "wrong_location_refute":
                org = orgs.next()
                place = places.next()
                other_place = other_places.next_distinct_from(place)
                claim = f"{org} is headquartered in {place}."
                evidence = f"{org} is headquartered in {other_place}, not {place}."
            elif subtype == "wrong_date_refute":
                name = other_people.next()
                year = rng.randint(1930, 2010)
                other_year = year + rng.choice([-22, -11, 6, 14, 27])
                claim = f"{name} was born in {year}."
                evidence = f"{name} was born in {other_year}."
            elif subtype == "wrong_numeric_threshold_refute":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                threshold = rng.randint(low + 20, high)
                gap = rng.randint(10, max(11, int(threshold * 0.3) + 10))
                actual = max(low, threshold - gap)
                claim = f"{org} reported more than {threshold} {unit} in {domain_name}."
                evidence = f"{org} recorded only {actual} {unit} in {domain_name}, short of {threshold}."
            else:  # wrong_creator_director_refute
                artifact = artifacts.next()
                name = people.next()
                other_name = other_people.next_distinct_from(name)
                claim = f"{artifact} was created by {name}."
                evidence = f"{artifact} was actually created by {other_name}."

            rows.append(make_row(
                family=_RERB_FAMILY, subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_refute_{subtype}", claim=claim, evidence=evidence,
                label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=0, primary_failure_type="polarity",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 3: polarity_disambiguation_bridge (200 rows: 100 SUPPORT + 100 REFUTE)
# ---------------------------------------------------------------------------
_PDB_SUBTYPES = [
    "more_than_less_than_pair",
    "before_after_pair",
    "same_value_vs_conflicting_value_pair",
    "positive_review_vs_negative_review_pair",
    "is_a_vs_is_not_a_type_pair",
    "founded_released_born_date_pair",
]
_PDB_FAMILY = "polarity_disambiguation_bridge"


def build_polarity_disambiguation_bridge(rng: Random, n_pairs: int) -> list[dict[str, Any]]:
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    other_categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    qualities = CyclicPool(rng, QUALITIES)
    adjectives = CyclicPool(rng, POSITIVE_ADJECTIVES)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_pairs, len(_PDB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_PDB_SUBTYPES, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage66_pdb_{subtype}_{idx:04d}"

            if subtype == "more_than_less_than_pair":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                actual = rng.randint(low + 50, high)
                margin = rng.randint(10, max(11, int(actual * 0.2) + 10))
                threshold = actual - margin
                evidence = f"{org} recorded {actual} {unit} in {domain_name}."
                support_claim = f"{org} reported more than {threshold} {unit} in {domain_name}."
                refute_claim = f"{org} reported less than {threshold} {unit} in {domain_name}."
            elif subtype == "before_after_pair":
                place = places.next()
                year = rng.randint(1930, 2010)
                margin = rng.randint(3, 20)
                threshold = year + margin
                evidence = f"An assembly hall in {place} was completed in {year}."
                support_claim = f"The assembly hall in {place} was completed before {threshold}."
                refute_claim = f"The assembly hall in {place} was completed after {threshold}."
            elif subtype == "same_value_vs_conflicting_value_pair":
                place = places.next()
                value = rng.randint(4000, 900000)
                other_value = value + rng.choice([-1500, -300, 300, 1500, 5000])
                evidence = f"{place} has a population of {value} residents."
                support_claim = f"{place} has a population of {value} residents."
                refute_claim = f"{place} has a population of {other_value} residents."
            elif subtype == "positive_review_vs_negative_review_pair":
                artifact = artifacts.next()
                adjective = adjectives.next()
                quality = qualities.next()
                evidence = f"Critics described {artifact} as {adjective}, highlighting its {quality}."
                support_claim = f"{artifact} received a positive critical response."
                refute_claim = f"{artifact} received a negative critical response."
            elif subtype == "is_a_vs_is_not_a_type_pair":
                artifact = artifacts.next()
                category = categories.next()
                other_category = other_categories.next_distinct_from(category)
                evidence = f"{artifact} is classified as a {category}."
                support_claim = f"{artifact} is a {category}."
                refute_claim = f"{artifact} is a {other_category}."
            else:  # founded_released_born_date_pair
                org = orgs.next()
                year = rng.randint(1900, 2015)
                other_year = year + rng.choice([-19, -8, 9, 17, 24])
                evidence = f"{org} was founded in {year}."
                support_claim = f"{org} was founded in {year}."
                refute_claim = f"{org} was founded in {other_year}."

            rows.append(make_row(
                family=_PDB_FAMILY, subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_polarity_support_{subtype}", claim=support_claim, evidence=evidence,
                label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=1, primary_failure_type="none",
            ))
            rows.append(make_row(
                family=_PDB_FAMILY, subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_polarity_refute_{subtype}", claim=refute_claim, evidence=evidence,
                label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=0, primary_failure_type="polarity",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 4: numeric_temporal_comparison_bridge (120 rows: 60 SUPPORT + 60 REFUTE)
# ---------------------------------------------------------------------------
_NTCB_PAIR_TYPES = [
    ("number_word_equivalence_support", "numeric_threshold_refute"),
    ("year_exact_match_support", "year_mismatch_refute"),
    ("before_after_support", "before_after_refute"),
]
_NTCB_FAMILY = "numeric_temporal_comparison_bridge"


def build_numeric_temporal_comparison_bridge(rng: Random, n_pairs_per_type: int) -> list[dict[str, Any]]:
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    idx = 0
    for support_subtype, refute_subtype in _NTCB_PAIR_TYPES:
        for _ in range(n_pairs_per_type):
            idx += 1

            if support_subtype == "number_word_equivalence_support":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                actual = rng.randint(low, high)
                delta = rng.choice([-40, -15, 15, 40, 90])
                other_value = actual + delta if actual + delta > 0 else actual + abs(delta) + 10
                evidence = f"{org} recorded {actual} {unit} in {domain_name}."
                support_claim = f"{org} recorded exactly {actual} {unit} in {domain_name}."
                refute_claim = f"{org} recorded exactly {other_value} {unit} in {domain_name}."
            elif support_subtype == "year_exact_match_support":
                place = places.next()
                year = rng.randint(1900, 2020)
                other_year = year + rng.choice([-13, -5, 4, 10, 21])
                evidence = f"A survey station in {place} was recorded active in {year}."
                support_claim = f"The survey station in {place} was recorded active in {year}."
                refute_claim = f"The survey station in {place} was recorded active in {other_year}."
            else:  # before_after_support / before_after_refute
                place = places.next()
                year = rng.randint(1900, 2015)
                margin = rng.randint(2, 15)
                threshold = year + margin
                evidence = f"A canal route near {place} opened in {year}."
                support_claim = f"The canal route near {place} opened before {threshold}."
                refute_claim = f"The canal route near {place} opened after {threshold}."

            support_pair_id = f"stage66_ntcb_{support_subtype}_{idx:04d}"
            refute_pair_id = f"stage66_ntcb_{refute_subtype}_{idx:04d}"

            rows.append(make_row(
                family=_NTCB_FAMILY, subtype=support_subtype, pair_id=support_pair_id,
                intervention_type=f"bridge_numeric_temporal_support_{support_subtype}",
                claim=support_claim, evidence=evidence,
                label_name="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=1, primary_failure_type="none",
            ))
            rows.append(make_row(
                family=_NTCB_FAMILY, subtype=refute_subtype, pair_id=refute_pair_id,
                intervention_type=f"bridge_numeric_temporal_refute_{refute_subtype}",
                claim=refute_claim, evidence=evidence,
                label_name="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
                sufficiency_label=1, polarity_label=0, primary_failure_type="polarity",
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 5: strict_ne_frame_safety_bridge (40 rows, NOT_ENTITLED only)
# ---------------------------------------------------------------------------
_SNFSB_SUBTYPES = [
    "wrong_subject_same_predicate_ne",
    "related_entity_distractor_ne",
    "same_domain_missing_predicate_ne",
    "partial_attribute_without_entitlement_ne",
]
_SNFSB_FAMILY = "strict_ne_frame_safety_bridge"

# polarity_label is not applicable for NOT_ENTITLED rows; -1 keeps it
# distinct from REFUTE's 0 and SUPPORT's 1 so no ambiguous label id is
# introduced on the polarity axis.
_NE_POLARITY_LABEL = -1


def build_strict_ne_frame_safety_bridge(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    collaborators = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    other_orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    other_places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_SNFSB_SUBTYPES))
    idx = 0
    for subtype, count in zip(_SNFSB_SUBTYPES, counts):
        for _ in range(count):
            idx += 1
            pair_id = f"stage66_snfsb_{subtype}_{idx:04d}"

            if subtype == "wrong_subject_same_predicate_ne":
                org = orgs.next()
                other_org = other_orgs.next_distinct_from(org)
                year = rng.randint(1950, 2020)
                claim = f"{org} was founded in {year}."
                evidence = f"{other_org} was founded in {year}."
                frame_compatible, predicate_covered, sufficiency, failure = 0, 0, 0, "frame"
            elif subtype == "related_entity_distractor_ne":
                name = people.next()
                collaborator = collaborators.next()
                profession = professions.next()
                other_profession = other_professions.next()
                place = places.next()
                claim = f"{name} is a {profession}."
                evidence = f"{collaborator}, a frequent collaborator of {name}, is a {other_profession} based in {place}."
                frame_compatible, predicate_covered, sufficiency, failure = 1, 0, 0, "sufficiency"
            elif subtype == "same_domain_missing_predicate_ne":
                name = people.next()
                year = rng.randint(1950, 2015)
                place = places.next()
                profession = professions.next()
                claim = f"{name} was born in {year}."
                evidence = f"{name} is a {profession} who currently resides in {place}."
                frame_compatible, predicate_covered, sufficiency, failure = 1, 0, 0, "sufficiency"
            else:  # partial_attribute_without_entitlement_ne
                org = orgs.next()
                total = rng.randint(20, 400)
                place_a = places.next()
                place_b = other_places.next()
                claim = f"{org} has {total} member organizations."
                evidence = f"{org}'s membership includes organizations based in {place_a} and {place_b}, among others."
                frame_compatible, predicate_covered, sufficiency, failure = 1, 1, 0, "sufficiency"

            rows.append(make_row(
                family=_SNFSB_FAMILY, subtype=subtype, pair_id=pair_id,
                intervention_type=f"bridge_ne_{subtype}", claim=claim, evidence=evidence,
                label_name="NOT_ENTITLED", frame_compatible_label=frame_compatible,
                predicate_covered_label=predicate_covered, sufficiency_label=sufficiency,
                polarity_label=_NE_POLARITY_LABEL, primary_failure_type=failure,
            ))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_dataset(seed: int) -> list[dict[str, Any]]:
    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    rows.extend(build_support_entitlement_recovery_bridge(
        Random(rng.random()), FAMILY_PLAN[_SERB_FAMILY]["planned_rows"]))
    rows.extend(build_refute_entitlement_recovery_bridge(
        Random(rng.random()), FAMILY_PLAN[_RERB_FAMILY]["planned_rows"]))
    rows.extend(build_polarity_disambiguation_bridge(
        Random(rng.random()), FAMILY_PLAN[_PDB_FAMILY]["planned_rows"] // 2))
    rows.extend(build_numeric_temporal_comparison_bridge(
        Random(rng.random()), FAMILY_PLAN[_NTCB_FAMILY]["planned_rows"] // 2 // 3))
    rows.extend(build_strict_ne_frame_safety_bridge(
        Random(rng.random()), FAMILY_PLAN[_SNFSB_FAMILY]["planned_rows"]))

    if len(rows) != STAGE66_TOTAL_ROWS:
        raise RuntimeError(f"generated {len(rows)} rows, expected {STAGE66_TOTAL_ROWS}")
    return rows


def load_design(design_json_path: Path) -> dict[str, Any] | None:
    if not design_json_path.exists():
        return None
    return json.loads(design_json_path.read_text(encoding="utf-8"))


def check_design_plan_match(design: dict[str, Any] | None) -> dict[str, Any]:
    if design is None:
        return {"design_file_found": False, "matches": None, "mismatches": ["design file not found"]}

    mismatches: list[str] = []
    design_label_plan = design.get("stage66_incremental_label_plan", {})
    if design_label_plan != STAGE66_INCREMENTAL_LABEL_PLAN:
        mismatches.append("stage66_incremental_label_plan differs from design file")

    design_families = {f["stage66_family"]: f for f in design.get("proposed_bridge_families", [])}
    for family, plan in FAMILY_PLAN.items():
        design_family = design_families.get(family)
        if design_family is None:
            mismatches.append(f"family '{family}' missing from design file")
            continue
        if design_family.get("planned_rows") != plan["planned_rows"]:
            mismatches.append(f"family '{family}' planned_rows differs from design file")
        if design_family.get("label_plan") != plan["label_plan"]:
            mismatches.append(f"family '{family}' label_plan differs from design file")
        if design_family.get("primary_target_error") != plan["target_error"]:
            mismatches.append(f"family '{family}' primary_target_error differs from design file")

    return {"design_file_found": True, "matches": len(mismatches) == 0, "mismatches": mismatches}


def check_required_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[str] = []
    for row in rows:
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] is None or row[field] == "":
                missing.append(f"{row.get('id', '<unknown>')}: missing {field}")
    return {"passed": len(missing) == 0, "missing_field_examples": missing[:20], "missing_count": len(missing)}


def check_duplicate_ids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["id"] for row in rows)
    duplicates = [row_id for row_id, count in counts.items() if count > 1]
    return {"passed": len(duplicates) == 0, "duplicate_count": len(duplicates), "duplicate_examples": duplicates[:20]}


def check_duplicate_claim_evidence_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((row["claim"], row["evidence"]) for row in rows)
    duplicates = [pair for pair, count in counts.items() if count > 1]
    return {
        "passed": len(duplicates) == 0,
        "duplicate_count": len(duplicates),
        "duplicate_examples": [{"claim": c, "evidence": e} for c, e in duplicates[:20]],
    }


def check_label_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[str] = []
    for row in rows:
        expected = FINAL_LABEL_TO_ID.get(row["final_label"])
        if expected is None or row["label"] != expected:
            bad.append(row["id"])
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20]}


def check_axis_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[str] = []
    for row in rows:
        label = row["final_label"]
        if label == "SUPPORT":
            ok = (
                row["frame_compatible_label"] == 1
                and row["predicate_covered_label"] == 1
                and row["sufficiency_label"] == 1
                and row["polarity_label"] == 1
            )
        elif label == "REFUTE":
            ok = (
                row["frame_compatible_label"] == 1
                and row["predicate_covered_label"] == 1
                and row["sufficiency_label"] == 1
                and row["polarity_label"] == 0
            )
        elif label == "NOT_ENTITLED":
            blocked = (
                row["frame_compatible_label"] == 0
                or row["predicate_covered_label"] == 0
                or row["sufficiency_label"] == 0
            )
            ok = blocked and row["polarity_label"] == _NE_POLARITY_LABEL
        else:
            ok = False
        if not ok:
            bad.append(row["id"])
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20]}


def check_family_and_label_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[str] = []
    family_counts = Counter(row["stage66_bridge_family"] for row in rows)
    family_label_counts: dict[str, Counter] = {}
    for row in rows:
        family_label_counts.setdefault(row["stage66_bridge_family"], Counter())[row["final_label"]] += 1

    extra_families = sorted(set(family_counts.keys()) - set(FAMILY_PLAN.keys()))
    missing_families = sorted(set(FAMILY_PLAN.keys()) - set(family_counts.keys()))
    if extra_families:
        mismatches.append(f"unexpected stage66_bridge_family value(s) present: {extra_families}")
    if missing_families:
        mismatches.append(f"expected stage66_bridge_family value(s) missing: {missing_families}")

    for family, plan in FAMILY_PLAN.items():
        if family_counts.get(family, 0) != plan["planned_rows"]:
            mismatches.append(f"family '{family}' row count {family_counts.get(family, 0)} != planned {plan['planned_rows']}")
        actual_label_plan = dict(family_label_counts.get(family, {}))
        for label_name, planned_count in plan["label_plan"].items():
            if actual_label_plan.get(label_name, 0) != planned_count:
                mismatches.append(
                    f"family '{family}' label '{label_name}' count {actual_label_plan.get(label_name, 0)} != planned {planned_count}"
                )

    total_label_counts = Counter(row["final_label"] for row in rows)
    for label_name, planned_count in STAGE66_INCREMENTAL_LABEL_PLAN.items():
        if total_label_counts.get(label_name, 0) != planned_count:
            mismatches.append(
                f"overall label '{label_name}' count {total_label_counts.get(label_name, 0)} != planned {planned_count}"
            )

    return {"passed": len(mismatches) == 0, "mismatches": mismatches}


def check_bridge_family_matches_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[str] = []
    wrong_target_error: list[str] = []
    for row in rows:
        if row["stage66_bridge_family"] != row["stage66_family"]:
            bad.append(row["id"])
        expected_target_error = FAMILY_PLAN.get(row["stage66_family"], {}).get("target_error")
        if row["stage66_target_error"] != expected_target_error:
            wrong_target_error.append(row["id"])
    return {
        "passed": len(bad) == 0 and len(wrong_target_error) == 0,
        "bridge_family_mismatch_count": len(bad),
        "bridge_family_mismatch_examples": bad[:20],
        "target_error_mismatch_count": len(wrong_target_error),
        "target_error_mismatch_examples": wrong_target_error[:20],
    }


def scan_forbidden_markers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for row in rows:
        for field, value in row.items():
            if field in FORBIDDEN_SCAN_EXCLUDE_FIELDS:
                continue
            text = str(value).lower()
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    hits.append({"id": row["id"], "field": field, "marker": marker})
    return {"passed": len(hits) == 0, "markers_checked": FORBIDDEN_MARKERS, "hit_count": len(hits), "hit_examples": hits[:20]}


def build_audit(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    design: dict[str, Any] | None,
) -> dict[str, Any]:
    label_counts = Counter(row["final_label"] for row in rows)
    family_counts = Counter(row["stage66_bridge_family"] for row in rows)
    subtype_counts = Counter(row["stage66_subtype"] for row in rows)

    family_label_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        fam = row["stage66_bridge_family"]
        lbl = row["final_label"]
        family_label_counts.setdefault(fam, Counter())[lbl] += 1
    family_label_counts = {fam: dict(counter) for fam, counter in family_label_counts.items()}

    required_fields_check = check_required_fields(rows)
    duplicate_id_check = check_duplicate_ids(rows)
    duplicate_pair_check = check_duplicate_claim_evidence_pairs(rows)
    label_mapping_check = check_label_mapping(rows)
    axis_consistency_check = check_axis_consistency(rows)
    family_label_count_check = check_family_and_label_counts(rows)
    bridge_family_identity_check = check_bridge_family_matches_family(rows)
    forbidden_marker_scan = scan_forbidden_markers(rows)
    design_plan_check = check_design_plan_match(design)

    all_checks_passed = (
        required_fields_check["passed"]
        and duplicate_id_check["passed"]
        and duplicate_pair_check["passed"]
        and label_mapping_check["passed"]
        and axis_consistency_check["passed"]
        and family_label_count_check["passed"]
        and bridge_family_identity_check["passed"]
        and forbidden_marker_scan["passed"]
        and len(rows) == STAGE66_TOTAL_ROWS
    )
    decision = STAGE67_DECISION_READY if all_checks_passed else STAGE67_DECISION_FAILED

    return {
        "stage": "Stage67",
        "decision": decision,
        "source_design_json": str(Path(args.design_json)).replace("\\", "/"),
        "output_jsonl": str(Path(args.output_jsonl)).replace("\\", "/"),
        "generation_config": {"seed": args.seed},
        "total_rows": len(rows),
        "expected_total_rows": STAGE66_TOTAL_ROWS,
        "counts_by_label": dict(label_counts),
        "expected_counts_by_label": STAGE66_INCREMENTAL_LABEL_PLAN,
        "counts_by_bridge_family": dict(family_counts),
        "counts_by_bridge_family_and_label": family_label_counts,
        "counts_by_subtype": dict(subtype_counts),
        "bridge_families": BRIDGE_FAMILY_ORDER,
        "label_mapping": FINAL_LABEL_TO_ID,
        "checks": {
            "required_fields_present": required_fields_check,
            "duplicate_id_check": duplicate_id_check,
            "duplicate_claim_evidence_pair_check": duplicate_pair_check,
            "label_mapping_check": label_mapping_check,
            "axis_consistency_check": axis_consistency_check,
            "family_and_label_count_check": family_label_count_check,
            "bridge_family_identity_check": bridge_family_identity_check,
            "forbidden_marker_scan": forbidden_marker_scan,
            "design_plan_match_check": design_plan_check,
        },
        "leakage_checks": LEAKAGE_POLICY,
        "notes": [
            "This dataset is synthetic training/diagnostic data only. It is NOT an "
            "external evaluation result and must not be reported as VitaminC or any "
            "other external-benchmark metric.",
            "Stage65 residual error taxonomy/counts were used only to set family "
            "quotas in the Stage66 design; no Stage65 residual sample text, "
            "VitaminC text, or VitaminC labels were read or used to produce any "
            "row in this file.",
            "This dataset must not be mixed with corrupted time_swap rows from "
            "data/controlled_v5_v3.jsonl.",
            "Per the Stage66 design, this bridge is intended to be appended to "
            "the training split only, after a clean main split (Stage69 scope); "
            "this script performs no such integration and no training.",
        ],
        "recommended_next_stage": {
            "stage": "Stage68",
            "name": "static audit of generated Stage66 residual bridge",
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
    lines: list[str] = []
    lines.append("# Stage67: Stage66 Residual Bridge Generation Audit")
    lines.append("")
    lines.append(f"**Decision:** `{audit['decision']}`")
    lines.append("")
    lines.append(f"- Source design JSON: `{audit['source_design_json']}`")
    lines.append(f"- Output JSONL: `{audit['output_jsonl']}`")
    lines.append(f"- Seed: {audit['generation_config']['seed']}")
    lines.append(f"- Total rows: {audit['total_rows']} (expected {audit['expected_total_rows']})")
    lines.append("")
    lines.append("## Counts by label")
    lines.append("")
    lines.append("| Label | Count | Expected |")
    lines.append("|---|---|---|")
    for label in ("SUPPORT", "REFUTE", "NOT_ENTITLED"):
        lines.append(
            f"| {label} | {audit['counts_by_label'].get(label, 0)} | {audit['expected_counts_by_label'].get(label, 0)} |"
        )
    lines.append("")
    lines.append("## Counts by bridge family")
    lines.append("")
    lines.append("| Family | Count |")
    lines.append("|---|---|")
    for family in audit["bridge_families"]:
        lines.append(f"| {family} | {audit['counts_by_bridge_family'].get(family, 0)} |")
    lines.append("")
    lines.append("## Counts by bridge family x label")
    lines.append("")
    lines.append("| Family | SUPPORT | REFUTE | NOT_ENTITLED |")
    lines.append("|---|---|---|---|")
    for family in audit["bridge_families"]:
        counts = audit["counts_by_bridge_family_and_label"].get(family, {})
        lines.append(
            f"| {family} | {counts.get('SUPPORT', 0)} | {counts.get('REFUTE', 0)} | {counts.get('NOT_ENTITLED', 0)} |"
        )
    lines.append("")
    lines.append("## Counts by subtype")
    lines.append("")
    lines.append("| Subtype | Count |")
    lines.append("|---|---|")
    for subtype, count in sorted(audit["counts_by_subtype"].items()):
        lines.append(f"| {subtype} | {count} |")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Passed |")
    lines.append("|---|---|")
    for name, result in audit["checks"].items():
        passed = result.get("passed", result.get("matches"))
        lines.append(f"| {name} | {passed} |")
    lines.append("")
    lines.append("## Leakage checks")
    lines.append("")
    for key, value in audit["leakage_checks"].items():
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
    parser.add_argument("--audit-json", type=str, default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--audit-md", type=str, default=str(DEFAULT_AUDIT_MD))
    parser.add_argument("--design-json", type=str, default=str(DEFAULT_DESIGN_JSON))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting existing output files. Without this flag, "
                              "the script fails safely if any output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_paths = [Path(args.output_jsonl), Path(args.audit_json), Path(args.audit_md)]
    if not args.overwrite:
        existing = [str(p) for p in output_paths if p.exists()]
        if existing:
            print(
                "[Stage67] refusing to overwrite existing output file(s) without --overwrite: "
                + ", ".join(existing),
                file=sys.stderr,
            )
            raise SystemExit(1)

    design = load_design(Path(args.design_json))

    rows = generate_dataset(seed=args.seed)

    output_jsonl_path = Path(args.output_jsonl)
    write_jsonl(rows, output_jsonl_path)

    audit = build_audit(rows, args, design)

    output_audit_json_path = Path(args.audit_json)
    output_audit_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    output_audit_md_path = Path(args.audit_md)
    output_audit_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_md_path.write_text(render_markdown(audit), encoding="utf-8")

    print(f"[Stage67] decision: {audit['decision']}")
    print(f"[Stage67] wrote {len(rows)} rows to {output_jsonl_path}")
    print(f"[Stage67] wrote audit JSON to {output_audit_json_path}")
    print(f"[Stage67] wrote audit Markdown to {output_audit_md_path}")


if __name__ == "__main__":
    main()
