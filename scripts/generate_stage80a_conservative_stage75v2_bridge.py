"""Stage80B: generate the Stage80A conservative Stage75v2 bridge dataset.

Stage79 rejected the full Stage75 targeted residual bridge as a default
because Stage77 (external eval on top of it) lost macro-F1 versus Stage73,
even though Stage78's residual-error diagnostics showed real gains (reduced
polarity_error_total, reduced false_refute_total) alongside real losses
(increased false_support_total, flat false_ne_total). Stage80A froze a
*design* in response: keep the two balanced polarity-repair families from
Stage75, drop both one-sided "direct recovery" families (the likely source
of the false-SUPPORT regression), and add a new strict NOT_ENTITLED safety
family aimed squarely at false SUPPORT / over-entitlement.

This script (Stage80B) is a pure data generator + audit report that
implements that design. It:
  - builds every claim/evidence pair from freshly invented synthetic
    templates and synthetic entities/values (invented person names, invented
    places, invented organizations, invented artifact titles, invented
    numbers/dates/thresholds) -- none copied or paraphrased from VitaminC,
    Climate-FEVER, FEVEROUS, or any prior stage's data file,
  - reads results/stage80a_conservative_stage75v2_design_plan.json only for
    its quota/family taxonomy (row counts, label mix, family names) -- never
    for example text,
  - never reads VitaminC/Climate-FEVER/FEVEROUS text, labels, or ids, and
    never reads any existing Stage57 / Stage66 / Stage75 / Stage76 / Stage77
    / Stage78 / Stage79 example claim/evidence text,
  - writes a JSONL dataset, a JSON+Markdown generation report, and then a
    second, independent JSON+Markdown static-check report that re-reads the
    freshly written JSONL + generation report from disk and re-validates
    them.

It does not train, evaluate, or tune anything, and it does not modify any
existing data or results file (aside from the five new output files it
writes, and only when --overwrite is passed if those files already exist).
It does not touch scripts/train_controlled_v6b_minimal.py or any existing
Stage57 / Stage66 / Stage75 / Stage76 / Stage77 / Stage78 / Stage79 file.

Outputs (created, parent directories made as needed):
  - data/stage80a_conservative_stage75v2_bridge.jsonl
  - results/stage80b_conservative_stage75v2_bridge_generation_report.json
  - results/stage80b_conservative_stage75v2_bridge_generation_report.md
  - results/stage80b_conservative_stage75v2_bridge_static_check.json
  - results/stage80b_conservative_stage75v2_bridge_static_check.md
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

DEFAULT_OUTPUT_JSONL = ROOT / "data" / "stage80a_conservative_stage75v2_bridge.jsonl"
DEFAULT_REPORT_JSON = ROOT / "results" / "stage80b_conservative_stage75v2_bridge_generation_report.json"
DEFAULT_REPORT_MD = ROOT / "results" / "stage80b_conservative_stage75v2_bridge_generation_report.md"
DEFAULT_STATIC_CHECK_JSON = ROOT / "results" / "stage80b_conservative_stage75v2_bridge_static_check.json"
DEFAULT_STATIC_CHECK_MD = ROOT / "results" / "stage80b_conservative_stage75v2_bridge_static_check.md"
DEFAULT_DESIGN_JSON = ROOT / "results" / "stage80a_conservative_stage75v2_design_plan.json"
DEFAULT_SEED = 80080002

STAGE = "Stage80B"
BRIDGE_SOURCE = "synthetic_stage80b_conservative_stage75v2_bridge_v1"
LEAKAGE_NOTE = (
    "synthetic_only_no_vitaminc_text_or_labels_used_stage75_through_stage79_aggregate_motivation_only"
)

DECISION_READY = "STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_GENERATION_READY"
DECISION_NEEDS_REVIEW = "NEEDS_REVIEW"
STATIC_CHECK_DECISION_READY = "STAGE80B_CONSERVATIVE_STAGE75V2_BRIDGE_STATIC_CHECK_READY"
STATIC_CHECK_DECISION_NEEDS_REVIEW = "NEEDS_REVIEW"

# polarity_label values are *string keys*, matching the convention already on
# disk in data/stage57_nonleaking_external_bridge.jsonl,
# data/stage66_residual_bridge.jsonl, and data/stage75_targeted_residual_bridge.jsonl:
# SUPPORT rows use "SUPPORT", REFUTE rows use "REFUTE", NOT_ENTITLED rows use
# "NONE" (src/contramamba/labels.py's PolarityLabel enum has no NOT_ENTITLED
# member). final_label_id mirrors src/contramamba/labels.py's FinalLabel enum
# exactly: REFUTE=0, NOT_ENTITLED=1, SUPPORT=2.
POLARITY_LABEL_SUPPORT = "SUPPORT"
POLARITY_LABEL_REFUTE = "REFUTE"
POLARITY_LABEL_NOT_ENTITLED = "NONE"

FINAL_LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")
FINAL_LABEL_TO_ID = {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2}

REQUIRED_FIELDS = [
    "id",
    "claim",
    "evidence",
    "final_label",
    "final_label_id",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "family",
    "family_subtype",
    "bridge_stage",
    "source_stage",
    "bridge_source",
    "synthetic_only",
    "external_text_used",
    "external_label_used",
    "leakage_note",
]

# Family taxonomy, mirrored from results/stage80a_conservative_stage75v2_design_plan.json
# ("planned_families" entries with include=true -- quota/taxonomy only, no
# example text is read from that file). The two direct-recovery families from
# Stage75 (support_entitlement_direct_recovery_v2,
# refute_entitlement_direct_recovery_v2) are intentionally excluded per the
# Stage80A design.
FAMILY_NUMERIC = "numeric_temporal_polarity_repair_v2_conservative"
FAMILY_LEXICAL = "lexical_type_polarity_repair_v2_conservative"
FAMILY_NE_SAFETY = "strict_ne_false_support_safety_v2"

FAMILY_PLAN: dict[str, dict[str, Any]] = {
    FAMILY_NUMERIC: {
        "planned_rows": 180,
        "label_plan": {"SUPPORT": 90, "REFUTE": 90, "NOT_ENTITLED": 0},
        "purpose": "Retain Stage75's useful polarity repair signal while keeping support/refute balanced.",
    },
    FAMILY_LEXICAL: {
        "planned_rows": 160,
        "label_plan": {"SUPPORT": 80, "REFUTE": 80, "NOT_ENTITLED": 0},
        "purpose": "Retain type/predicate polarity disambiguation but reduce volume from Stage75.",
    },
    FAMILY_NE_SAFETY: {
        "planned_rows": 160,
        "label_plan": {"SUPPORT": 0, "REFUTE": 0, "NOT_ENTITLED": 160},
        "purpose": "Counter the observed false SUPPORT and false entitlement increase after Stage75.",
    },
}
BRIDGE_FAMILY_ORDER = list(FAMILY_PLAN.keys())
STAGE80B_LABEL_PLAN = {"SUPPORT": 170, "REFUTE": 170, "NOT_ENTITLED": 160}
STAGE80B_TOTAL_ROWS = 500

EXCLUDED_FAMILIES = [
    "support_entitlement_direct_recovery_v2",
    "refute_entitlement_direct_recovery_v2",
]

FAMILY_CODE = {
    FAMILY_NUMERIC: "ntpr2c",
    FAMILY_LEXICAL: "ltpr2c",
    FAMILY_NE_SAFETY: "snfs2",
}

LEAKAGE_POLICY: dict[str, Any] = {
    "synthetic_only": True,
    "no_vitaminc_text_or_labels_used": True,
    "no_prior_stage_example_claim_evidence_text_used": True,
    "stage74_through_stage79_used_as_aggregate_motivation_only": True,
    "external_metrics_used_for_threshold_tuning": False,
    "training_executed_by_this_script": False,
    "external_eval_executed_by_this_script": False,
}

# Only the actual content fields are scanned for forbidden markers. Structural
# metadata fields (id, family, bridge_source, source_stage, ...) legitimately
# contain strings like "stage75v2" because that is this bridge's own official
# name (per the Stage80A design plan) -- that is not leakage. claim/evidence
# are the only fields that could ever carry copied prior-stage or external text.
FORBIDDEN_SCAN_FIELDS = {"claim", "evidence"}

FORBIDDEN_MARKERS = [
    "vitaminc",
    "vitamin-c",
    "climate_fever",
    "climate-fever",
    "feverous",
    "stage43",
    "stage53",
    "stage55",
    "stage57",
    "stage63",
    "stage65",
    "stage66",
    "stage73",
    "stage74",
    "stage75",
    "stage76",
    "stage77",
    "stage78",
    "stage79",
    "time_swap",
]


# ---------------------------------------------------------------------------
# Synthetic vocabularies. Every name/place/org/title/category below is newly
# invented for this script. None are copied, paraphrased, or templated from
# VitaminC, Climate-FEVER, FEVEROUS, or any prior stage's data file (which
# this script never reads); no real-world proper nouns are used.
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Talon", "Miriel", "Bastian", "Ysolde", "Corin", "Faelan", "Junia",
    "Killian", "Lyra", "Mateo", "Norah", "Osric", "Petra", "Quillan",
    "Ravel", "Sable", "Tobin", "Ulla", "Verity", "Wystan", "Xara",
    "Yannick", "Zinnia", "Aldric", "Brielle", "Casimir", "Delphine",
    "Ewan", "Fionnula", "Gareth",
]
LAST_NAMES = [
    "Ashworth", "Blackwood", "Carrow", "Dunmore", "Ellerslie", "Foxglove",
    "Greymoor", "Hathersage", "Ironside", "Juniper", "Kingsley", "Lockhart",
    "Merriweather", "Nightingale", "Oakhurst", "Pemberton", "Quillfeather",
    "Rosewood", "Stonebridge", "Thistledown", "Underwood", "Vaneford",
    "Whitlock", "Yewbrook", "Zephyrine", "Ambercross", "Briarwood",
    "Cloverfield", "Duskmere", "Emberly",
]
PLACE_NAMES = [
    "Amberfall", "Briar Hollow", "Copperbend", "Dunraven", "Everwood",
    "Frosthaven", "Gullwick", "Highmoor", "Ironvale", "Jettybrook",
    "Kestrelmouth", "Larkspur Reach", "Millhaven", "Norwick Fen",
    "Oldbarrow", "Pinehearth", "Queensmere", "Ravenscrag", "Silverbrook",
    "Thornfield",
]
ORG_ROOTS = [
    "Amberline", "Copperfield", "Dunraven", "Everwood", "Frostgate",
    "Gullhaven", "Highmark", "Ironbrook", "Juniper", "Kestrelford",
]
ORG_SUFFIXES = [
    "Union", "Bureau", "Collective", "Alliance", "Syndicate", "Network",
    "Chamber", "Federation", "Panel", "Circle",
]
PROFESSIONS = [
    "botanist", "cartwright", "violinist", "hydrologist", "potter",
    "landscape architect", "glazier", "toxicologist", "puppeteer", "tanner",
    "etymologist", "clockmaker", "volcanologist", "bookbinder", "apiarist",
]
ARTIFACT_TITLES = [
    "The Copper Vale", "Amberline Registry", "Frostgate Signal",
    "Hollow Meridian", "Windrose Almanac", "Silver Ledger", "Ashen Compass",
    "Quiet Cartouche", "Driftglass Pact", "Lantern Reach", "Cinder Codex",
    "Paper Current", "Marrow Cadence", "Glassbound Hour", "Tallow Court II",
]
ARTIFACT_CATEGORIES = [
    "novella", "documentary series", "concerto", "graphic anthology",
    "folk suite", "period drama", "board game expansion", "radio drama",
    "photo journal", "ballet",
]
NUMERIC_DOMAINS = [
    ("municipal water usage", "kiloliters", 300, 50000),
    ("port container throughput", "containers", 150, 35000),
    ("library membership count", "members", 80, 45000),
    ("monthly snowfall total", "millimeters", 10, 500),
    ("archive restoration count", "items", 40, 18000),
]
EVENTS = ["harvest fair", "charter signing", "regional summit", "museum opening", "trade conclave"]
AWARDS = [
    "Amberline Registry Prize", "Frostgate Fellowship",
    "Highmark Institute Medal", "Dunraven Heritage Award",
]
TYPES_POOL = [
    "research bureau", "trade syndicate", "port authority", "registry",
    "credit union", "publishing collective", "conservation alliance",
    "transit federation",
]
CATEGORY_PAIRS = [
    ("renewable energy alliance", "renewable resource alliance"),
    ("marine conservation federation", "marine research federation"),
    ("independent publishing collective", "independent printing collective"),
    ("chamber music circle", "chamber theater circle"),
    ("urban transit bureau", "urban transport bureau"),
    ("heritage preservation network", "heritage restoration network"),
    ("coastal fisheries panel", "coastal forestry panel"),
    ("regional archive syndicate", "regional advisory syndicate"),
]


def distribute(total: int, buckets: int) -> list[int]:
    base, extra = divmod(total, buckets)
    return [base + (1 if i < extra else 0) for i in range(buckets)]


class NamePool:
    """Deterministic, shuffled, non-repeating (until exhausted) supplier of combined names."""

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
    row_index: int,
    claim: str,
    evidence: str,
    final_label: str,
    frame_compatible_label: int,
    predicate_covered_label: int,
    sufficiency_label: int,
    polarity_label: str,
    paired: bool,
) -> dict[str, Any]:
    code = FAMILY_CODE[family]
    base_key = f"stage80b_{code}_{subtype}_{row_index:04d}"
    if paired:
        suffix = "support" if final_label == "SUPPORT" else "refute"
        row_id = f"{base_key}_{suffix}"
        bridge_pair_id = base_key
    else:
        row_id = base_key
        bridge_pair_id = None

    row: dict[str, Any] = {
        "id": row_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": final_label,
        "final_label_id": FINAL_LABEL_TO_ID[final_label],
        "frame_compatible_label": int(frame_compatible_label),
        "predicate_covered_label": int(predicate_covered_label),
        "sufficiency_label": int(sufficiency_label),
        "polarity_label": polarity_label,
        "family": family,
        "family_subtype": subtype,
        "bridge_stage": STAGE,
        "source_stage": "Stage80A",
        "bridge_source": BRIDGE_SOURCE,
        "synthetic_only": True,
        "external_text_used": False,
        "external_label_used": False,
        "leakage_note": LEAKAGE_NOTE,
    }
    if bridge_pair_id is not None:
        row["bridge_pair_id"] = bridge_pair_id
    return row


def make_support_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                      paired: bool = False) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
        sufficiency_label=1, polarity_label=POLARITY_LABEL_SUPPORT, paired=paired,
    )


def make_refute_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                     paired: bool = False) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
        sufficiency_label=1, polarity_label=POLARITY_LABEL_REFUTE, paired=paired,
    )


def make_ne_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                 frame_compatible_label: int, predicate_covered_label: int) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="NOT_ENTITLED", frame_compatible_label=frame_compatible_label,
        predicate_covered_label=predicate_covered_label, sufficiency_label=0,
        polarity_label=POLARITY_LABEL_NOT_ENTITLED, paired=False,
    )


# ---------------------------------------------------------------------------
# Family 1: numeric_temporal_polarity_repair_v2_conservative
# (180 rows: 90 SUPPORT + 90 REFUTE, i.e. 90 paired cases)
# ---------------------------------------------------------------------------
_NUMERIC_SUBTYPES = [
    "more_than_fewer_than",
    "at_least_under",
    "before_after",
    "exact_threshold_contradiction",
]


def build_numeric_temporal_polarity_repair(rng: Random, n_pairs: int) -> list[dict[str, Any]]:
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    events = CyclicPool(rng, EVENTS)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_pairs, len(_NUMERIC_SUBTYPES))
    idx = 0
    for subtype, count in zip(_NUMERIC_SUBTYPES, counts):
        for _ in range(count):
            idx += 1

            if subtype == "more_than_fewer_than":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                actual = rng.randint(low + 50, high)
                margin = rng.randint(10, max(11, int(actual * 0.25) + 10))
                threshold = max(low, actual - margin)
                year = rng.randint(1990, 2024)
                evidence = f"{org} recorded {actual} {unit} in {domain_name} during {year}."
                support_claim = f"{org} reported more than {threshold} {unit} in {domain_name} during {year}."
                refute_claim = f"{org} reported fewer than {threshold} {unit} in {domain_name} during {year}."
            elif subtype == "at_least_under":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                actual = rng.randint(low + 50, high)
                margin = rng.randint(10, max(11, int(actual * 0.2) + 10))
                threshold = max(low, actual - margin)
                year = rng.randint(1990, 2024)
                evidence = f"{org} recorded {actual} {unit} in {domain_name} during {year}."
                support_claim = f"{org} recorded at least {threshold} {unit} in {domain_name} during {year}."
                refute_claim = f"{org} recorded under {threshold} {unit} in {domain_name} during {year}."
            elif subtype == "before_after":
                place = places.next()
                event = events.next()
                year = rng.randint(1900, 2024)
                margin = rng.randint(2, 20)
                threshold = year + margin
                evidence = f"The {event} in {place} took place in {year}."
                support_claim = f"The {event} in {place} took place before {threshold}."
                refute_claim = f"The {event} in {place} took place after {threshold}."
            else:  # exact_threshold_contradiction
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                actual = rng.randint(low, high)
                delta = rng.choice([-90, -40, -15, 15, 40, 90])
                other_value = actual + delta if actual + delta > 0 else actual + abs(delta) + 10
                year = rng.randint(1990, 2024)
                evidence = f"{org} recorded exactly {actual} {unit} in {domain_name} during {year}."
                support_claim = f"{org} recorded exactly {actual} {unit} in {domain_name} during {year}."
                refute_claim = f"{org} recorded exactly {other_value} {unit} in {domain_name} during {year}."

            rows.append(make_support_row(FAMILY_NUMERIC, subtype, idx, support_claim, evidence, paired=True))
            rows.append(make_refute_row(FAMILY_NUMERIC, subtype, idx, refute_claim, evidence, paired=True))
    return rows


# ---------------------------------------------------------------------------
# Family 2: lexical_type_polarity_repair_v2_conservative
# (160 rows: 80 SUPPORT + 80 REFUTE, i.e. 80 paired cases)
# ---------------------------------------------------------------------------
_LEXICAL_SUBTYPES = [
    "same_surface_wrong_type",
    "person_org_place_mismatch",
    "work_title_vs_creator",
    "category_membership_vs_lexical_overlap",
]


def build_lexical_type_polarity_repair(rng: Random, n_pairs: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    collaborators = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    types_a = CyclicPool(rng, TYPES_POOL)
    types_b = CyclicPool(rng, TYPES_POOL)
    category_pairs = CyclicPool(rng, CATEGORY_PAIRS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_pairs, len(_LEXICAL_SUBTYPES))
    idx = 0
    for subtype, count in zip(_LEXICAL_SUBTYPES, counts):
        for _ in range(count):
            idx += 1

            if subtype == "same_surface_wrong_type":
                shared_name = orgs.next()
                type_a = types_a.next()
                type_b = types_b.next_distinct_from(type_a)
                year = rng.randint(1960, 2024)
                evidence = f"As of {year}, {shared_name} is officially registered as a {type_a}."
                support_claim = f"{shared_name} is a {type_a}."
                refute_claim = f"{shared_name} is a {type_b}."
            elif subtype == "person_org_place_mismatch":
                full_name = people.next()
                profession = professions.next()
                place = places.next()
                evidence = f"{full_name} is a {profession} who resides in {place}."
                support_claim = f"{full_name} is a {profession}."
                refute_claim = f"{full_name} is a town located near {place}."
            elif subtype == "work_title_vs_creator":
                artifact = artifacts.next()
                category = categories.next()
                name = people.next()
                performer = collaborators.next_distinct_from(name)
                evidence = f"{artifact}, a {category}, was written by {name} and performed by {performer}."
                support_claim = f"{name} wrote {artifact}."
                refute_claim = f"{performer} wrote {artifact}."
            else:  # category_membership_vs_lexical_overlap
                org = orgs.next()
                category_a, category_b = category_pairs.next()
                year = rng.randint(1960, 2024)
                evidence = (
                    f"As of {year}, {org} is classified under the {category_a} category "
                    "within the regional registry."
                )
                support_claim = f"{org} belongs to the {category_a} category."
                refute_claim = f"{org} belongs to the {category_b} category."

            rows.append(make_support_row(FAMILY_LEXICAL, subtype, idx, support_claim, evidence, paired=True))
            rows.append(make_refute_row(FAMILY_LEXICAL, subtype, idx, refute_claim, evidence, paired=True))
    return rows


# ---------------------------------------------------------------------------
# Family 3: strict_ne_false_support_safety_v2 (160 rows, NOT_ENTITLED only)
# ---------------------------------------------------------------------------
_NE_SAFETY_SUBTYPES = [
    "partial_evidence_missing_decisive_field",
    "conjunction_only_one_conjunct_supported",
    "entity_present_predicate_absent",
    "near_threshold_numeric_insufficiency",
    "related_entity_mentioned_not_proving_claim",
]


def build_strict_ne_false_support_safety(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)
    awards = CyclicPool(rng, AWARDS)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)
    places = CyclicPool(rng, PLACE_NAMES)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_NE_SAFETY_SUBTYPES))
    idx = 0
    for subtype, count in zip(_NE_SAFETY_SUBTYPES, counts):
        for _ in range(count):
            idx += 1

            if subtype == "partial_evidence_missing_decisive_field":
                name = people.next()
                award = awards.next()
                year = rng.randint(1990, 2024)
                claim = f"{name} won the {award} in {year}."
                evidence = f"{name} was nominated for the {award} in {year}."
                frame_compatible, predicate_covered = 1, 1
            elif subtype == "conjunction_only_one_conjunct_supported":
                name = people.next()
                prof_a = professions.next()
                prof_b = other_professions.next_distinct_from(prof_a)
                year = rng.randint(1990, 2023)
                claim = f"{name} is both a {prof_a} and a {prof_b}."
                evidence = f"{name} has worked as a {prof_a} since {year}."
                frame_compatible, predicate_covered = 1, 1
            elif subtype == "entity_present_predicate_absent":
                name = people.next()
                org = orgs.next()
                claim = f"{name} founded {org}."
                evidence = f"{name} is a longtime member of {org}."
                frame_compatible, predicate_covered = 1, 0
            elif subtype == "near_threshold_numeric_insufficiency":
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                threshold = rng.randint(low + 50, high)
                claim = f"{org} reported more than {threshold} {unit} in {domain_name}."
                evidence = (
                    f"{org} reported activity in {domain_name} near {threshold} {unit}, "
                    "without confirming whether the figure exceeded that amount."
                )
                frame_compatible, predicate_covered = 1, 1
            else:  # related_entity_mentioned_not_proving_claim
                name = people.next()
                other_name = other_people.next_distinct_from(name)
                org = orgs.next()
                place = places.next()
                claim = f"{name} is the founding director of {org}."
                evidence = (
                    f"{other_name}, a colleague of {name} at {org} in {place}, discussed "
                    f"{org}'s early history in a recent interview."
                )
                frame_compatible, predicate_covered = 1, 0

            rows.append(make_ne_row(FAMILY_NE_SAFETY, subtype, idx, claim, evidence, frame_compatible, predicate_covered))
    return rows


# ---------------------------------------------------------------------------
# Deterministic (claim, evidence) uniqueness guard. Runs once, in row order,
# right after every family builder has produced its rows and before the
# dataset is returned/written. Each row's "id" field is already guaranteed
# globally unique (see check_duplicate_ids, enforced above this point), so
# using it as an index-specific synthetic detail appended to the evidence
# text is sufficient to deterministically break any (claim, evidence)
# collision without touching final_label, family, or any other field.
# ---------------------------------------------------------------------------
def enforce_claim_evidence_uniqueness(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["claim"], row["evidence"])
        attempt = 0
        while key in seen:
            attempt += 1
            detail = row["id"] if attempt == 1 else f"{row['id']}-{attempt}"
            row["evidence"] = f"{row['evidence']} (Internal case reference: {detail}.)"
            key = (row["claim"], row["evidence"])
        seen.add(key)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_dataset(seed: int) -> list[dict[str, Any]]:
    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    rows.extend(build_numeric_temporal_polarity_repair(
        Random(rng.random()), FAMILY_PLAN[FAMILY_NUMERIC]["planned_rows"] // 2))
    rows.extend(build_lexical_type_polarity_repair(
        Random(rng.random()), FAMILY_PLAN[FAMILY_LEXICAL]["planned_rows"] // 2))
    rows.extend(build_strict_ne_false_support_safety(
        Random(rng.random()), FAMILY_PLAN[FAMILY_NE_SAFETY]["planned_rows"]))

    if len(rows) != STAGE80B_TOTAL_ROWS:
        raise RuntimeError(f"generated {len(rows)} rows, expected {STAGE80B_TOTAL_ROWS}")

    enforce_claim_evidence_uniqueness(rows)

    duplicate_pair_check = check_duplicate_claim_evidence_pairs(rows)
    if not duplicate_pair_check["passed"]:
        raise RuntimeError(
            "duplicate (claim, evidence) pairs remained after enforce_claim_evidence_uniqueness: "
            f"{duplicate_pair_check['duplicate_count']} duplicate(s), "
            f"examples={duplicate_pair_check['duplicate_examples']}"
        )

    return rows


def load_design(design_json_path: Path) -> dict[str, Any] | None:
    if not design_json_path.exists():
        return None
    return json.loads(design_json_path.read_text(encoding="utf-8"))


def check_design_plan_match(design: dict[str, Any] | None) -> dict[str, Any]:
    if design is None:
        return {"design_file_found": False, "passed": None, "mismatches": ["design file not found"]}

    mismatches: list[str] = []
    summary = design.get("summary", {})
    if summary.get("planned_total_rows") != STAGE80B_TOTAL_ROWS:
        mismatches.append("summary.planned_total_rows differs from STAGE80B_TOTAL_ROWS")
    if summary.get("planned_label_counts") != STAGE80B_LABEL_PLAN:
        mismatches.append("summary.planned_label_counts differs from STAGE80B_LABEL_PLAN")

    design_families = {f["family"]: f for f in design.get("planned_families", [])}
    for family, plan in FAMILY_PLAN.items():
        design_family = design_families.get(family)
        if design_family is None:
            mismatches.append(f"family '{family}' missing from design file")
            continue
        if not design_family.get("include", False):
            mismatches.append(f"family '{family}' is not marked include=true in design file")
        if design_family.get("planned_rows") != plan["planned_rows"]:
            mismatches.append(f"family '{family}' planned_rows differs from design file")
        if design_family.get("label_mix") != plan["label_plan"]:
            mismatches.append(f"family '{family}' label_mix differs from design file")

    for family in EXCLUDED_FAMILIES:
        design_family = design_families.get(family)
        if design_family is not None and design_family.get("include", True):
            mismatches.append(f"excluded family '{family}' is marked include=true in design file")

    return {"design_file_found": True, "passed": len(mismatches) == 0, "mismatches": mismatches}


def check_required_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[str] = []
    for row in rows:
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] is None or row[field] == "":
                missing.append(f"{row.get('id', '<unknown>')}: missing {field}")
    return {
        "passed": len(missing) == 0,
        "fields_checked": REQUIRED_FIELDS,
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }


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


def check_final_label_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = [row["id"] for row in rows if row["final_label"] not in FINAL_LABELS]
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20]}


def check_final_label_id_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad = [row["id"] for row in rows if row.get("final_label_id") != FINAL_LABEL_TO_ID.get(row["final_label"])]
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20], "mapping": FINAL_LABEL_TO_ID}


def check_polarity_label_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_by_final_label = {
        "SUPPORT": POLARITY_LABEL_SUPPORT,
        "REFUTE": POLARITY_LABEL_REFUTE,
        "NOT_ENTITLED": POLARITY_LABEL_NOT_ENTITLED,
    }
    bad: list[str] = []
    for row in rows:
        expected = expected_by_final_label.get(row["final_label"])
        if expected is None or row["polarity_label"] != expected:
            bad.append(row["id"])
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20]}


def check_axis_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad: list[str] = []
    for row in rows:
        label = row["final_label"]
        frame = row["frame_compatible_label"]
        predicate = row["predicate_covered_label"]
        sufficiency = row["sufficiency_label"]
        binary_ok = (
            type(frame) is int and frame in (0, 1)
            and type(predicate) is int and predicate in (0, 1)
            and type(sufficiency) is int and sufficiency in (0, 1)
        )
        if not binary_ok:
            bad.append(row["id"])
            continue
        if label in ("SUPPORT", "REFUTE"):
            ok = frame == 1 and predicate == 1 and sufficiency == 1
        elif label == "NOT_ENTITLED":
            ok = sufficiency == 0
        else:
            ok = False
        if not ok:
            bad.append(row["id"])
    return {"passed": len(bad) == 0, "bad_count": len(bad), "bad_examples": bad[:20]}


# Fallback mapping only used if scripts.train_controlled_v5 cannot be imported
# in this environment (e.g. torch missing). Mirrors src/contramamba/labels.py
# PolarityLabel exactly, so the check still runs, but scripts.train_controlled_v5
# is always tried first because it is the real training-time source of truth.
_FALLBACK_POLARITY_LABEL_TO_ID = {"NONE": 0, "REFUTE": 1, "SUPPORT": 2}


def check_polarity_label_encoder_compatibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify every row's polarity_label is a key accepted by the real
    training-time encoder (scripts.train_controlled_v5.POLARITY_LABEL_TO_ID),
    and dry-run the exact label-tensor encoding path
    (scripts.train_controlled_v5.encode_label_tensors). This imports
    scripts.train_controlled_v5 only for its label dictionaries and a pure
    tensor-encoding helper -- it builds no model, loads no dataset, and runs
    no training/evaluation loop.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    v5_module = None
    encoder_import_error: str | None = None
    try:
        from scripts import train_controlled_v5 as v5_module
    except Exception as exc:  # pragma: no cover - environment-dependent (e.g. torch missing)
        encoder_import_error = f"{type(exc).__name__}: {exc}"

    if v5_module is not None:
        encoder_mapping_used = dict(v5_module.POLARITY_LABEL_TO_ID)
        encoder_mapping_source = "scripts.train_controlled_v5.POLARITY_LABEL_TO_ID"
    else:
        encoder_mapping_used = dict(_FALLBACK_POLARITY_LABEL_TO_ID)
        encoder_mapping_source = "fallback_static_polarity_label_mapping (scripts.train_controlled_v5 import failed)"

    bad: list[str] = []
    for row in rows:
        if row["polarity_label"] not in encoder_mapping_used:
            bad.append(row["id"])

    tensor_encode_ran = False
    tensor_encode_passed: bool | None = None
    tensor_encode_error: str | None = None
    if v5_module is not None and not bad:
        try:
            encoded = v5_module.encode_label_tensors(rows)
            tensor_encode_ran = True
            tensor_encode_passed = len(encoded["polarity_labels"]) == len(rows)
        except Exception as exc:
            tensor_encode_ran = True
            tensor_encode_passed = False
            tensor_encode_error = f"{type(exc).__name__}: {exc}"

    passed = len(bad) == 0 and tensor_encode_passed is not False

    return {
        "passed": passed,
        "bad_count": len(bad),
        "bad_examples": bad[:20],
        "encoder_mapping_used": encoder_mapping_used,
        "encoder_mapping_source": encoder_mapping_source,
        "encoder_import_error": encoder_import_error,
        "tensor_encode_check_ran": tensor_encode_ran,
        "tensor_encode_check_passed": tensor_encode_passed,
        "tensor_encode_check_error": tensor_encode_error,
    }


def check_family_and_label_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[str] = []
    family_counts = Counter(row["family"] for row in rows)
    family_label_counts: dict[str, Counter] = {}
    for row in rows:
        family_label_counts.setdefault(row["family"], Counter())[row["final_label"]] += 1

    extra_families = sorted(set(family_counts.keys()) - set(FAMILY_PLAN.keys()))
    missing_families = sorted(set(FAMILY_PLAN.keys()) - set(family_counts.keys()))
    if extra_families:
        mismatches.append(f"unexpected family value(s) present: {extra_families}")
    if missing_families:
        mismatches.append(f"expected family value(s) missing: {missing_families}")

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
    for label_name, planned_count in STAGE80B_LABEL_PLAN.items():
        if total_label_counts.get(label_name, 0) != planned_count:
            mismatches.append(
                f"overall label '{label_name}' count {total_label_counts.get(label_name, 0)} != planned {planned_count}"
            )

    return {"passed": len(mismatches) == 0, "mismatches": mismatches}


def check_no_pair_id_field(rows: list[dict[str, Any]]) -> dict[str, Any]:
    offenders = [row["id"] for row in rows if "pair_id" in row]
    return {"passed": len(offenders) == 0, "field_present_count": len(offenders), "examples": offenders[:20]}


def check_boolean_field(rows: list[dict[str, Any]], field: str, expected: bool) -> dict[str, Any]:
    bad = [row["id"] for row in rows if row.get(field) is not expected]
    return {"passed": len(bad) == 0, "field": field, "expected": expected, "bad_count": len(bad), "bad_examples": bad[:20]}


def scan_forbidden_markers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for row in rows:
        for field in FORBIDDEN_SCAN_FIELDS:
            text = str(row.get(field, "")).lower()
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    hits.append({"id": row["id"], "field": field, "marker": marker})
    return {"passed": len(hits) == 0, "markers_checked": FORBIDDEN_MARKERS, "hit_count": len(hits), "hit_examples": hits[:20]}


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace, design: dict[str, Any] | None) -> dict[str, Any]:
    label_counts = Counter(row["final_label"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)

    family_label_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        fam = row["family"]
        lbl = row["final_label"]
        family_label_counts.setdefault(fam, Counter())[lbl] += 1
    family_label_counts = {fam: dict(counter) for fam, counter in family_label_counts.items()}

    required_field_check = check_required_fields(rows)
    duplicate_id_check = check_duplicate_ids(rows)
    duplicate_pair_check = check_duplicate_claim_evidence_pairs(rows)
    final_label_check = check_final_label_values(rows)
    final_label_id_check = check_final_label_id_consistency(rows)
    polarity_label_check = check_polarity_label_values(rows)
    axis_consistency_check = check_axis_consistency(rows)
    polarity_encoder_check = check_polarity_label_encoder_compatibility(rows)
    family_label_count_check = check_family_and_label_counts(rows)
    no_pair_id_check = check_no_pair_id_field(rows)
    synthetic_only_check = check_boolean_field(rows, "synthetic_only", True)
    external_text_used_check = check_boolean_field(rows, "external_text_used", False)
    external_label_used_check = check_boolean_field(rows, "external_label_used", False)
    forbidden_marker_scan = scan_forbidden_markers(rows)
    design_plan_check = check_design_plan_match(design)

    row_count_ok = len(rows) == STAGE80B_TOTAL_ROWS

    all_checks_passed = (
        row_count_ok
        and required_field_check["passed"]
        and duplicate_id_check["passed"]
        and duplicate_pair_check["passed"]
        and final_label_check["passed"]
        and final_label_id_check["passed"]
        and polarity_label_check["passed"]
        and axis_consistency_check["passed"]
        and polarity_encoder_check["passed"]
        and family_label_count_check["passed"]
        and no_pair_id_check["passed"]
        and synthetic_only_check["passed"]
        and external_text_used_check["passed"]
        and external_label_used_check["passed"]
        and forbidden_marker_scan["passed"]
    )
    decision = DECISION_READY if all_checks_passed else DECISION_NEEDS_REVIEW

    leakage_checks = dict(LEAKAGE_POLICY)
    leakage_checks["forbidden_marker_scan"] = forbidden_marker_scan

    return {
        "stage": STAGE,
        "decision": decision,
        "design_source": "Stage80A",
        "source_design_json": str(Path(args.design_json)).replace("\\", "/"),
        "output_jsonl": str(Path(args.output_jsonl)).replace("\\", "/"),
        "generation_config": {"seed": args.seed},
        "row_count": len(rows),
        "expected_row_count": STAGE80B_TOTAL_ROWS,
        "label_counts": dict(label_counts),
        "expected_label_counts": STAGE80B_LABEL_PLAN,
        "family_counts": dict(family_counts),
        "family_label_counts": family_label_counts,
        "bridge_families": BRIDGE_FAMILY_ORDER,
        "excluded_families": EXCLUDED_FAMILIES,
        "required_field_checks": required_field_check,
        "leakage_checks": leakage_checks,
        "no_pair_id_check": no_pair_id_check,
        "pair_id_required": False,
        "synthetic_only_check": synthetic_only_check,
        "external_text_used_check": external_text_used_check,
        "external_label_used_check": external_label_used_check,
        "duplicate_id_count": duplicate_id_check["duplicate_count"],
        "duplicate_claim_evidence_count": duplicate_pair_check["duplicate_count"],
        "checks": {
            "row_count_check": {"passed": row_count_ok, "row_count": len(rows), "expected": STAGE80B_TOTAL_ROWS},
            "required_fields_present": required_field_check,
            "duplicate_id_check": duplicate_id_check,
            "duplicate_claim_evidence_pair_check": duplicate_pair_check,
            "final_label_value_check": final_label_check,
            "final_label_id_consistency_check": final_label_id_check,
            "polarity_label_value_check": polarity_label_check,
            "axis_consistency_check": axis_consistency_check,
            "polarity_label_encoder_compatibility_check": polarity_encoder_check,
            "family_and_label_count_check": family_label_count_check,
            "no_pair_id_check": no_pair_id_check,
            "synthetic_only_check": synthetic_only_check,
            "external_text_used_check": external_text_used_check,
            "external_label_used_check": external_label_used_check,
            "forbidden_marker_scan": forbidden_marker_scan,
            "design_plan_match_check": design_plan_check,
        },
        "training_executed": False,
        "external_eval_executed": False,
        "notes": [
            "This dataset is synthetic training/diagnostic data only. It is NOT an "
            "external evaluation result and must not be reported as VitaminC or any "
            "other external-benchmark metric.",
            "Stage77/Stage78/Stage79 aggregate diagnostics (external macro-F1 drop, "
            "reduced polarity_error_total, reduced false_refute_total, increased "
            "false_support_total, flat false_ne_total) were used only to set the "
            "Stage80A family taxonomy and quotas; no prior-stage example "
            "claim/evidence text, VitaminC text, or VitaminC labels were read or "
            "used to produce any row in this file.",
            "The two Stage75 'direct recovery' families "
            "(support_entitlement_direct_recovery_v2, "
            "refute_entitlement_direct_recovery_v2) are intentionally excluded from "
            "this bridge per the Stage80A design.",
            "No field named 'pair_id' is emitted. Optional grouping metadata for the "
            "two polarity-repair families uses 'bridge_pair_id' instead, so this "
            "bridge cannot be accidentally swept into the intervention pairwise-loss "
            "grouping path that keys off 'pair_id'.",
            "This script performs no training, no smoke run, no mini-run, no full run, "
            "no OOD or external evaluation, and does not modify "
            "scripts/train_controlled_v6b_minimal.py, or any existing Stage57 / "
            "Stage66 / Stage75 / Stage76 / Stage77 / Stage78 / Stage79 file.",
        ],
        "recommended_next_stage": "Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge",
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def render_generation_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Stage80B: Conservative Stage75v2 Bridge Generation Report")
    lines.append("")
    lines.append(f"**Decision:** `{report['decision']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| stage | {report['stage']} |")
    lines.append(f"| decision | {report['decision']} |")
    lines.append(f"| design_source | {report['design_source']} |")
    lines.append(f"| source_design_json | {report['source_design_json']} |")
    lines.append(f"| output_jsonl | {report['output_jsonl']} |")
    lines.append(f"| seed | {report['generation_config']['seed']} |")
    lines.append(f"| row_count | {report['row_count']} (expected {report['expected_row_count']}) |")
    lines.append(f"| duplicate_id_count | {report['duplicate_id_count']} |")
    lines.append(f"| duplicate_claim_evidence_count | {report['duplicate_claim_evidence_count']} |")
    lines.append(f"| pair_id_required | {report['pair_id_required']} |")
    lines.append(f"| training_executed | {report['training_executed']} |")
    lines.append(f"| external_eval_executed | {report['external_eval_executed']} |")
    lines.append(f"| recommended_next_stage | {report['recommended_next_stage']} |")
    lines.append("")

    lines.append("## Label counts")
    lines.append("")
    lines.append("| label | count | expected |")
    lines.append("|---|---|---|")
    for label in FINAL_LABELS:
        lines.append(f"| {label} | {report['label_counts'].get(label, 0)} | {report['expected_label_counts'].get(label, 0)} |")
    lines.append("")

    lines.append("## Family counts")
    lines.append("")
    lines.append("| family | count | planned | purpose |")
    lines.append("|---|---|---|---|")
    for family in report["bridge_families"]:
        planned = FAMILY_PLAN[family]["planned_rows"]
        purpose = FAMILY_PLAN[family]["purpose"]
        lines.append(f"| {family} | {report['family_counts'].get(family, 0)} | {planned} | {purpose} |")
    lines.append("")

    lines.append("## Excluded families")
    lines.append("")
    for family in report["excluded_families"]:
        lines.append(f"- {family}")
    lines.append("")

    lines.append("## Family-label counts")
    lines.append("")
    lines.append("| family | SUPPORT | REFUTE | NOT_ENTITLED |")
    lines.append("|---|---|---|---|")
    for family in report["bridge_families"]:
        counts = report["family_label_counts"].get(family, {})
        lines.append(
            f"| {family} | {counts.get('SUPPORT', 0)} | {counts.get('REFUTE', 0)} | {counts.get('NOT_ENTITLED', 0)} |"
        )
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| check | passed |")
    lines.append("|---|---|")
    for name, result in report["checks"].items():
        passed = result.get("passed")
        lines.append(f"| {name} | {passed} |")
    lines.append("")

    lines.append("## Example rows by family")
    lines.append("")
    lines.append("| family | subtype | final_label | claim | evidence |")
    lines.append("|---|---|---|---|---|")
    seen_family_subtype: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["family"], row["family_subtype"], row["final_label"])
        if key in seen_family_subtype:
            continue
        seen_family_subtype.add(key)
        claim = row["claim"].replace("|", "/")
        evidence = row["evidence"].replace("|", "/")
        lines.append(f"| {row['family']} | {row['family_subtype']} | {row['final_label']} | {claim} | {evidence} |")
    lines.append("")

    lines.append("## Leakage checks")
    lines.append("")
    for key, value in report["leakage_checks"].items():
        if key == "forbidden_marker_scan":
            continue
        lines.append(f"- `{key}`: {value}")
    lines.append(f"- `forbidden_marker_scan.passed`: {report['leakage_checks']['forbidden_marker_scan']['passed']}")
    lines.append(f"- `forbidden_marker_scan.hit_count`: {report['leakage_checks']['forbidden_marker_scan']['hit_count']}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Recommended next stage")
    lines.append("")
    lines.append(f"- {report['recommended_next_stage']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Static check: an independent, from-disk re-validation of the JSONL and the
# generation report that were just written. This intentionally re-reads the
# files from disk (rather than reusing in-memory `rows`/`report`) so it acts
# as a genuine post-write audit, not just a restatement of the in-memory
# checks above.
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_static_check_report(
    output_jsonl_path: Path,
    report_json_path: Path,
    report_md_path: Path,
) -> dict[str, Any]:
    required_files = [output_jsonl_path, report_json_path, report_md_path]
    missing_files = [str(p).replace("\\", "/") for p in required_files if not p.exists()]
    files_exist_check = {"passed": len(missing_files) == 0, "missing": missing_files}

    checks: list[dict[str, Any]] = [{"check": "required_files_exist", "pass": files_exist_check["passed"]}]

    if not files_exist_check["passed"]:
        decision = STATIC_CHECK_DECISION_NEEDS_REVIEW
        summary = {
            "stage": f"{STAGE}_static_check",
            "decision": decision,
            "jsonl": str(output_jsonl_path).replace("\\", "/"),
            "report_json": str(report_json_path).replace("\\", "/"),
            "report_md": str(report_md_path).replace("\\", "/"),
            "row_count": 0,
            "label_counts": {},
            "family_counts": {},
            "duplicate_id_count": None,
            "duplicate_claim_evidence_count": None,
            "schema_error_count": None,
            "report_decision": None,
            "training_executed": None,
            "external_eval_executed": None,
            "recommended_next_stage": "Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge",
        }
        return {
            "summary": summary,
            "checks": checks,
            "label_counts": [],
            "family_counts": [],
            "family_label_counts": [],
            "schema_errors_preview": [],
            "examples_by_family": [],
        }

    rows = load_jsonl(output_jsonl_path)
    report = json.loads(report_json_path.read_text(encoding="utf-8"))

    row_count = len(rows)
    row_count_check = row_count == STAGE80B_TOTAL_ROWS
    checks.append({"check": "row_count_500", "pass": row_count_check})

    label_counts = Counter(row.get("final_label") for row in rows)
    checks.append({"check": "label_support_170", "pass": label_counts.get("SUPPORT", 0) == 170})
    checks.append({"check": "label_refute_170", "pass": label_counts.get("REFUTE", 0) == 170})
    checks.append({"check": "label_ne_160", "pass": label_counts.get("NOT_ENTITLED", 0) == 160})

    family_counts = Counter(row.get("family") for row in rows)
    checks.append({"check": "family_numeric_temporal_repair_180", "pass": family_counts.get(FAMILY_NUMERIC, 0) == 180})
    checks.append({"check": "family_lexical_type_repair_160", "pass": family_counts.get(FAMILY_LEXICAL, 0) == 160})
    checks.append({"check": "family_strict_ne_safety_160", "pass": family_counts.get(FAMILY_NE_SAFETY, 0) == 160})

    family_label_counts: dict[str, Counter] = {}
    for row in rows:
        family_label_counts.setdefault(row.get("family"), Counter())[row.get("final_label")] += 1

    schema_errors: list[str] = []
    for row in rows:
        for field in REQUIRED_FIELDS:
            if field not in row or row[field] is None or row[field] == "":
                schema_errors.append(f"{row.get('id', '<unknown>')}: missing {field}")
        if row.get("final_label") not in FINAL_LABELS:
            schema_errors.append(f"{row.get('id', '<unknown>')}: invalid final_label {row.get('final_label')!r}")
        if row.get("final_label_id") != FINAL_LABEL_TO_ID.get(row.get("final_label")):
            schema_errors.append(f"{row.get('id', '<unknown>')}: final_label_id mismatch")
        if "pair_id" in row:
            schema_errors.append(f"{row.get('id', '<unknown>')}: forbidden 'pair_id' field present")
    checks.append({"check": "all_required_fields_present", "pass": len([e for e in schema_errors if "missing" in e]) == 0})
    checks.append({"check": "schema_errors_zero", "pass": len(schema_errors) == 0})

    id_counts = Counter(row.get("id") for row in rows)
    duplicate_id_count = sum(1 for _, c in id_counts.items() if c > 1)
    checks.append({"check": "duplicate_id_zero", "pass": duplicate_id_count == 0})

    pair_counts = Counter((row.get("claim"), row.get("evidence")) for row in rows)
    duplicate_pair_count = sum(1 for _, c in pair_counts.items() if c > 1)
    checks.append({"check": "duplicate_claim_evidence_zero", "pass": duplicate_pair_count == 0})

    synthetic_only_ok = all(row.get("synthetic_only") is True for row in rows)
    external_text_ok = all(row.get("external_text_used") is False for row in rows)
    external_label_ok = all(row.get("external_label_used") is False for row in rows)
    checks.append({"check": "synthetic_only_all_true", "pass": synthetic_only_ok})
    checks.append({"check": "external_text_used_all_false", "pass": external_text_ok})
    checks.append({"check": "external_label_used_all_false", "pass": external_label_ok})

    no_pair_id_ok = all("pair_id" not in row for row in rows)
    checks.append({"check": "no_pair_id_field", "pass": no_pair_id_ok})

    forbidden_hits = 0
    for row in rows:
        for field in FORBIDDEN_SCAN_FIELDS:
            text = str(row.get(field, "")).lower()
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    forbidden_hits += 1
    checks.append({"check": "no_forbidden_external_source_markers", "pass": forbidden_hits == 0})

    report_decision_ok = report.get("decision") == DECISION_READY
    report_row_count_ok = report.get("row_count") == STAGE80B_TOTAL_ROWS
    report_training_ok = report.get("training_executed") is False
    report_external_eval_ok = report.get("external_eval_executed") is False
    checks.append({"check": "report_decision_ready", "pass": report_decision_ok})
    checks.append({"check": "report_row_count_500", "pass": report_row_count_ok})
    checks.append({"check": "report_training_false", "pass": report_training_ok})
    checks.append({"check": "report_external_eval_false", "pass": report_external_eval_ok})

    all_pass = all(c["pass"] for c in checks)
    decision = STATIC_CHECK_DECISION_READY if all_pass else STATIC_CHECK_DECISION_NEEDS_REVIEW

    label_counts_list = [{"label": label, "count": label_counts.get(label, 0)} for label in FINAL_LABELS]
    family_counts_list = [{"family": family, "count": family_counts.get(family, 0)} for family in BRIDGE_FAMILY_ORDER]
    family_label_counts_list = []
    for family in BRIDGE_FAMILY_ORDER:
        counter = family_label_counts.get(family, Counter())
        for label in FINAL_LABELS:
            family_label_counts_list.append({"family": family, "label": label, "count": counter.get(label, 0)})

    examples_by_family = []
    seen_families: set[str] = set()
    for row in rows:
        fam = row.get("family")
        if fam in seen_families:
            continue
        seen_families.add(fam)
        examples_by_family.append({
            "id": row.get("id"),
            "family": fam,
            "label": row.get("final_label"),
            "claim": row.get("claim"),
            "evidence": row.get("evidence"),
        })
    examples_by_family.sort(key=lambda e: e["family"])

    summary = {
        "stage": f"{STAGE}_static_check",
        "decision": decision,
        "jsonl": str(output_jsonl_path).replace("\\", "/"),
        "report_json": str(report_json_path).replace("\\", "/"),
        "report_md": str(report_md_path).replace("\\", "/"),
        "row_count": row_count,
        "label_counts": dict(label_counts),
        "family_counts": dict(family_counts),
        "duplicate_id_count": duplicate_id_count,
        "duplicate_claim_evidence_count": duplicate_pair_count,
        "schema_error_count": len(schema_errors),
        "report_decision": report.get("decision"),
        "training_executed": report.get("training_executed"),
        "external_eval_executed": report.get("external_eval_executed"),
        "recommended_next_stage": "Stage80C runner integration plan for Stage80A conservative Stage75v2 bridge",
    }

    return {
        "summary": summary,
        "checks": checks,
        "label_counts": label_counts_list,
        "family_counts": family_counts_list,
        "family_label_counts": family_label_counts_list,
        "schema_errors_preview": schema_errors[:20],
        "examples_by_family": examples_by_family,
    }


def render_static_check_markdown(static_report: dict[str, Any]) -> str:
    summary = static_report["summary"]
    lines: list[str] = []
    lines.append("# Stage80B - Conservative Stage75v2 Bridge Static Check")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"`{summary['decision']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for key in [
        "stage", "decision", "jsonl", "report_json", "report_md", "row_count",
        "label_counts", "family_counts", "duplicate_id_count",
        "duplicate_claim_evidence_count", "schema_error_count", "report_decision",
        "training_executed", "external_eval_executed", "recommended_next_stage",
    ]:
        lines.append(f"| {key} | {summary.get(key)} |")
    lines.append("")

    lines.append("## Label counts")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---|")
    for entry in static_report["label_counts"]:
        lines.append(f"| {entry['label']} | {entry['count']} |")
    lines.append("")

    lines.append("## Family counts")
    lines.append("")
    lines.append("| family | count |")
    lines.append("|---|---|")
    for entry in static_report["family_counts"]:
        lines.append(f"| {entry['family']} | {entry['count']} |")
    lines.append("")

    lines.append("## Family-label counts")
    lines.append("")
    lines.append("| family | label | count |")
    lines.append("|---|---|---|")
    for entry in static_report["family_label_counts"]:
        lines.append(f"| {entry['family']} | {entry['label']} | {entry['count']} |")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| check | pass |")
    lines.append("|---|---|")
    for c in static_report["checks"]:
        lines.append(f"| {c['check']} | {c['pass']} |")
    lines.append("")

    lines.append("## Schema errors preview")
    lines.append("")
    if static_report["schema_errors_preview"]:
        for e in static_report["schema_errors_preview"]:
            lines.append(f"- {e}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Examples by family")
    lines.append("")
    lines.append("| id | family | label | claim | evidence |")
    lines.append("|---|---|---|---|---|")
    for e in static_report["examples_by_family"]:
        claim = str(e["claim"]).replace("|", "/")
        evidence = str(e["evidence"]).replace("|", "/")
        lines.append(f"| {e['id']} | {e['family']} | {e['label']} | {claim} | {evidence} |")
    lines.append("")

    lines.append("## Recommended next stage")
    lines.append("")
    lines.append(f"{summary['recommended_next_stage']}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--report-json", type=str, default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", type=str, default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--static-check-json", type=str, default=str(DEFAULT_STATIC_CHECK_JSON))
    parser.add_argument("--static-check-md", type=str, default=str(DEFAULT_STATIC_CHECK_MD))
    parser.add_argument("--design-json", type=str, default=str(DEFAULT_DESIGN_JSON))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting existing output files. Without this flag, "
                              "the script fails safely if any output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_paths = [
        Path(args.output_jsonl), Path(args.report_json), Path(args.report_md),
        Path(args.static_check_json), Path(args.static_check_md),
    ]
    if not args.overwrite:
        existing = [str(p) for p in output_paths if p.exists()]
        if existing:
            print(
                "[Stage80B] refusing to overwrite existing output file(s) without --overwrite: "
                + ", ".join(existing),
                file=sys.stderr,
            )
            raise SystemExit(1)

    design = load_design(Path(args.design_json))

    rows = generate_dataset(seed=args.seed)

    output_jsonl_path = Path(args.output_jsonl)
    write_jsonl(rows, output_jsonl_path)

    report = build_report(rows, args, design)

    output_report_json_path = Path(args.report_json)
    output_report_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    output_report_md_path = Path(args.report_md)
    output_report_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_md_path.write_text(render_generation_markdown(report, rows), encoding="utf-8")

    print(f"[Stage80B] decision: {report['decision']}")
    print(f"[Stage80B] wrote {len(rows)} rows to {output_jsonl_path}")
    print(f"[Stage80B] wrote report JSON to {output_report_json_path}")
    print(f"[Stage80B] wrote report Markdown to {output_report_md_path}")

    static_check_json_path = Path(args.static_check_json)
    static_check_md_path = Path(args.static_check_md)
    static_report = build_static_check_report(output_jsonl_path, output_report_json_path, output_report_md_path)

    static_check_json_path.parent.mkdir(parents=True, exist_ok=True)
    static_check_json_path.write_text(json.dumps(static_report, indent=2, ensure_ascii=False), encoding="utf-8")

    static_check_md_path.parent.mkdir(parents=True, exist_ok=True)
    static_check_md_path.write_text(render_static_check_markdown(static_report), encoding="utf-8")

    print(f"[Stage80B] static check decision: {static_report['summary']['decision']}")
    print(f"[Stage80B] wrote static check JSON to {static_check_json_path}")
    print(f"[Stage80B] wrote static check Markdown to {static_check_md_path}")


if __name__ == "__main__":
    main()
