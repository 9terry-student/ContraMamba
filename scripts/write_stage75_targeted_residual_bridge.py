"""Stage75B: generate the synthetic Stage75 targeted residual bridge dataset.

Stage74 ran a residual external (VitaminC) error audit on the Stage71-retry
checkpoint and found the residual failure mix was dominated by false
NOT_ENTITLED on gold SUPPORT/REFUTE (false_NE_total=323), followed by a large
polarity-confusion band (polarity_error_total=244) and a much smaller
false-entitlement tail (false_entitlement_total=80). Stage75A froze a
*design* on top of those aggregate counts only: five bridge families, 1020
rows total, quotas/label-mix only -- no VitaminC text or labels.

This script (Stage75B) is a pure data generator + audit report that
implements that design. It:
  - builds every claim/evidence pair from freshly invented synthetic
    templates and synthetic entities/values (invented person names, invented
    places, invented organizations, invented artifact titles, invented
    numbers/dates/thresholds),
  - reads results/stage75a_targeted_bridge_design_plan.json only for its
    quota/family taxonomy (row counts, label plan, family names, target-error
    labels) -- never for example text,
  - never reads VitaminC/Climate-FEVER/FEVEROUS text, labels, or ids, and
    never reads any Stage74 example claim/evidence text (Stage74 is used only
    as aggregate motivation, recorded as metadata in this script's report),
  - writes a JSONL dataset plus a JSON+Markdown audit report.

It does not train, evaluate, or tune anything, and it does not modify any
existing data or results file (aside from the new output files it writes,
and only when --overwrite is passed if those files already exist). It does
not touch scripts/train_controlled_v6b_minimal.py or any existing Stage57 /
Stage66 / Stage73 / Stage74 / Stage75A file.

Outputs (created, parent directories made as needed):
  - data/stage75_targeted_residual_bridge.jsonl
  - results/stage75b_targeted_residual_bridge_generation_report.json
  - results/stage75b_targeted_residual_bridge_generation_report.md
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

DEFAULT_OUTPUT_JSONL = ROOT / "data" / "stage75_targeted_residual_bridge.jsonl"
DEFAULT_REPORT_JSON = ROOT / "results" / "stage75b_targeted_residual_bridge_generation_report.json"
DEFAULT_REPORT_MD = ROOT / "results" / "stage75b_targeted_residual_bridge_generation_report.md"
DEFAULT_DESIGN_JSON = ROOT / "results" / "stage75a_targeted_bridge_design_plan.json"
DEFAULT_SEED = 75075002

STAGE75B_DECISION_READY = "STAGE75B_TARGETED_RESIDUAL_BRIDGE_GENERATION_READY"
STAGE75B_DECISION_FAILED = "STAGE75B_TARGETED_RESIDUAL_BRIDGE_GENERATION_FAILED"

STAGE = "Stage75B"
BRIDGE_SOURCE = "synthetic_stage75b_targeted_residual_bridge_v2"
LEAKAGE_NOTE = (
    "synthetic_only_no_vitaminc_text_or_labels_used_stage74_aggregate_motivation_only"
)

# polarity_label values are *string keys*, matching the convention already on
# disk in data/controlled_v5_v3_without_time_swap.jsonl,
# data/stage57_nonleaking_external_bridge.jsonl, and
# data/stage66_residual_bridge.jsonl: SUPPORT rows use "SUPPORT", REFUTE rows
# use "REFUTE", NOT_ENTITLED rows use "NONE" (src/contramamba/labels.py's
# PolarityLabel enum has no NOT_ENTITLED member).
POLARITY_LABEL_SUPPORT = "SUPPORT"
POLARITY_LABEL_REFUTE = "REFUTE"
POLARITY_LABEL_NOT_ENTITLED = "NONE"

FINAL_LABELS = ("SUPPORT", "REFUTE", "NOT_ENTITLED")

REQUIRED_FIELDS = [
    "id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "bridge_family",
    "target_error_type",
    "bridge_source",
    "synthetic_only",
    "external_text_used",
    "external_label_used",
    "leakage_note",
    "stage",
]

# Family taxonomy, mirrored from results/stage75a_targeted_bridge_design_plan.json
# ("families" / "family_plan" entries -- quota/taxonomy only, no example text
# is read from that file). target_error is the family-level string exactly as
# frozen by Stage75A; families 3 and 4 mix two polarity-error directions, so
# individual rows in those two families are tagged with the more specific
# per-row direction (see ROW_TARGET_ERROR_SUPPORT / ROW_TARGET_ERROR_REFUTE
# below) rather than the combined family-level string.
FAMILY_SEDR = "support_entitlement_direct_recovery_v2"
FAMILY_REDR = "refute_entitlement_direct_recovery_v2"
FAMILY_NTPC = "numeric_temporal_polarity_comparison_v2"
FAMILY_LTPD = "lexical_type_polarity_disambiguation_v2"
FAMILY_SNES = "strict_ne_external_style_safety_v2"

FAMILY_PLAN: dict[str, dict[str, Any]] = {
    FAMILY_SEDR: {
        "planned_rows": 240,
        "label_plan": {"SUPPORT": 240, "REFUTE": 0, "NOT_ENTITLED": 0},
        "target_error": "false_NE_on_SUPPORT",
        "purpose": "Recover obvious SUPPORT cases that are currently over-abstained as NOT_ENTITLED.",
    },
    FAMILY_REDR: {
        "planned_rows": 220,
        "label_plan": {"SUPPORT": 0, "REFUTE": 220, "NOT_ENTITLED": 0},
        "target_error": "false_NE_on_REFUTE",
        "purpose": "Recover obvious REFUTE cases that are currently over-abstained as NOT_ENTITLED.",
    },
    FAMILY_NTPC: {
        "planned_rows": 260,
        "label_plan": {"SUPPORT": 130, "REFUTE": 130, "NOT_ENTITLED": 0},
        "target_error": "wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT",
        "purpose": "Reduce polarity flips on before/after, greater/less, at-least/under, exact-count comparisons.",
    },
    FAMILY_LTPD: {
        "planned_rows": 220,
        "label_plan": {"SUPPORT": 110, "REFUTE": 110, "NOT_ENTITLED": 0},
        "target_error": "wrong_polarity_SUPPORT_to_REFUTE + wrong_polarity_REFUTE_to_SUPPORT",
        "purpose": "Reduce polarity flips where lexical overlap hides type mismatch or direct equivalence.",
    },
    FAMILY_SNES: {
        "planned_rows": 80,
        "label_plan": {"SUPPORT": 0, "REFUTE": 0, "NOT_ENTITLED": 80},
        "target_error": "false_SUPPORT_on_NE + false_REFUTE_on_NE",
        "purpose": "Preserve NE safety without overcorrecting into more false_NE.",
    },
}
BRIDGE_FAMILY_ORDER = list(FAMILY_PLAN.keys())
STAGE75B_LABEL_PLAN = {"SUPPORT": 480, "REFUTE": 460, "NOT_ENTITLED": 80}
STAGE75B_TOTAL_ROWS = 1020

FAMILY_CODE = {
    FAMILY_SEDR: "sedr2",
    FAMILY_REDR: "redr2",
    FAMILY_NTPC: "ntpc2",
    FAMILY_LTPD: "ltpd2",
    FAMILY_SNES: "snes2",
}

# Per-row target_error_type used for families 3 and 4, which mix both
# polarity-flip directions per the Stage75A design.
ROW_TARGET_ERROR_SUPPORT = "wrong_polarity_SUPPORT_to_REFUTE"
ROW_TARGET_ERROR_REFUTE = "wrong_polarity_REFUTE_to_SUPPORT"

LEAKAGE_POLICY: dict[str, Any] = {
    "synthetic_only": True,
    "no_vitaminc_text_or_labels_used": True,
    "no_stage74_example_claim_evidence_text_used": True,
    "stage74_used_as_aggregate_motivation_only": True,
    "external_metrics_used_for_threshold_tuning": False,
    "training_executed_by_this_script": False,
    "external_eval_executed_by_this_script": False,
}

# Fields whose *value* is a fixed, mandated policy tag rather than generated
# row content. LEAKAGE_NOTE intentionally contains the substring "vitaminc"
# (it documents *non*-use), so it is excluded from the forbidden-marker text
# scan; every other field is scanned.
FORBIDDEN_SCAN_EXCLUDE_FIELDS = {"leakage_note"}

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
    "stage73",
    "stage74",
    "time_swap",
]


# ---------------------------------------------------------------------------
# Synthetic vocabularies. Every name/place/org/title/category below is newly
# invented for this script. None are copied, paraphrased, or templated from
# VitaminC, Climate-FEVER, FEVEROUS, or the Stage74 residual example CSV
# (which this script never reads); no real-world proper nouns are used.
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Mira", "Daren", "Selene", "Bryn", "Corwin", "Elowen", "Fenn", "Isolde",
    "Joran", "Kestrel", "Liora", "Nyle", "Odessa", "Percy", "Quinby",
    "Rosalind", "Soren", "Thalia", "Ulric", "Vesper", "Wrenna", "Xanthe",
    "Yara", "Zeke", "Arlo", "Briony", "Callum", "Dessa", "Emrys", "Faela",
]
LAST_NAMES = [
    "Quill", "Holt", "Voss", "Marrow", "Sterling", "Ashcombe", "Larkin",
    "Faircloth", "Winslow", "Hallow", "Brackenridge", "Corvel", "Delacourt",
    "Everhart", "Fenmore", "Gladstone", "Harrowgate", "Ivywood", "Jessup",
    "Kirrin", "Lansdown", "Merrow", "Norwood", "Osprey", "Penrose",
    "Quarry", "Ridgeway", "Steepleton", "Thackeray", "Underbridge",
]
PLACE_NAMES = [
    "Arlen City", "Kepler Harbor", "Northbridge", "Windmere Bay",
    "Thorncastle", "Hallowgate", "Brightwater", "Cindermark", "Fenhollow",
    "Graywick Point", "Ivoryreach", "Jasperfield", "Larkholt", "Mossgate",
    "Overwick", "Pallowmere", "Quaybrook", "Ridgemarsh", "Silverfen",
    "Tanglewood Cross",
]
ORG_ROOTS = [
    "Merrow", "Luma", "Ledgerfield", "Cindermark", "Hallowgate",
    "Brightwater", "Kestrel", "Thistlemoor", "Ashcombe", "Ravenwell",
]
ORG_SUFFIXES = [
    "Institute", "Archive", "Trust", "Cooperative", "Assembly", "Guild",
    "Foundation", "Registry", "Society", "Consortium",
]
PROFESSIONS = [
    "archivist", "cartographer", "cellist", "marine biologist", "ceramicist",
    "urban planner", "glassblower", "epidemiologist", "choreographer",
    "blacksmith", "lexicographer", "horologist", "seismologist",
    "typesetter", "falconer",
]
ARTIFACT_TITLES = [
    "The Glass Orchard", "Northbridge Ledger", "Ember Relay", "Hollow Star",
    "Windfall Chronicle", "Silver Loom", "Ashen Meridian",
    "Quiet Cartographer", "Driftwood Accord", "Lantern Bearer",
    "Cinder Atlas", "Paper Tide", "Marrow Season", "Glasswing Hour",
    "Tallow Court",
]
ARTIFACT_CATEGORIES = [
    "novel", "documentary film", "chamber symphony", "graphic memoir",
    "folk album", "historical drama", "tabletop expansion", "radio serial",
    "photo essay", "opera",
]
NUMERIC_DOMAINS = [
    ("annual grant funding", "dollars", 5000, 900000),
    ("harbor cargo throughput", "containers", 200, 40000),
    ("reading room visitor count", "visitors", 100, 50000),
    ("quarterly rainfall total", "millimeters", 20, 600),
    ("archive digitization count", "scans", 50, 20000),
]
QUALITIES = ["pacing", "structure", "clarity", "originality", "craftsmanship", "tone"]
POSITIVE_ADJECTIVES = ["luminous", "assured", "meticulous", "vivid", "inventive", "graceful"]
ROLES = ["director", "curator", "chief archivist", "artistic director", "board chair", "lead researcher"]
AWARDS = [
    "Northbridge Ledger Prize", "Kepler Harbor Fellowship",
    "Merrow Institute Medal", "Cindermark Heritage Award",
]
EVENTS = ["harbor festival", "founding ceremony", "regional assembly", "archive dedication", "trade summit"]
PRODUCTS = [
    "solar water heaters", "tidal generators", "archive scanners",
    "harbor buoys", "ceramic filters", "wind gauges", "greenhouse kits",
    "seed vaults",
]
TYPES_POOL = [
    "research institute", "trade guild", "harbor authority", "archive",
    "cooperative bank", "publishing house", "conservation trust",
    "transit authority",
]
CATEGORY_PAIRS = [
    ("renewable energy cooperative", "renewable resource cooperative"),
    ("marine conservation trust", "marine research trust"),
    ("independent publishing house", "independent printing house"),
    ("chamber music ensemble", "chamber theater ensemble"),
    ("urban transit authority", "urban transport union"),
    ("heritage preservation society", "heritage restoration society"),
    ("coastal fisheries council", "coastal forestry council"),
    ("regional archive network", "regional advisory network"),
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
    target_error_type: str,
    paired: bool,
) -> dict[str, Any]:
    code = FAMILY_CODE[family]
    base_key = f"stage75b_{code}_{subtype}_{row_index:04d}"
    if paired:
        suffix = "support" if final_label == "SUPPORT" else "refute"
        row_id = f"{base_key}_{suffix}"
        bridge_pair_id = base_key
    else:
        row_id = base_key
        bridge_pair_id = base_key

    return {
        "id": row_id,
        "claim": claim,
        "evidence": evidence,
        "final_label": final_label,
        "frame_compatible_label": int(frame_compatible_label),
        "predicate_covered_label": int(predicate_covered_label),
        "sufficiency_label": int(sufficiency_label),
        "polarity_label": polarity_label,
        "bridge_family": family,
        "target_error_type": target_error_type,
        "bridge_source": BRIDGE_SOURCE,
        "synthetic_only": True,
        "external_text_used": False,
        "external_label_used": False,
        "leakage_note": LEAKAGE_NOTE,
        "stage": STAGE,
        "bridge_pair_id": bridge_pair_id,
        "bridge_subtype": subtype,
    }


def make_support_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                      target_error_type: str, paired: bool = False) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="SUPPORT", frame_compatible_label=1, predicate_covered_label=1,
        sufficiency_label=1, polarity_label=POLARITY_LABEL_SUPPORT,
        target_error_type=target_error_type, paired=paired,
    )


def make_refute_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                     target_error_type: str, paired: bool = False) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="REFUTE", frame_compatible_label=1, predicate_covered_label=1,
        sufficiency_label=1, polarity_label=POLARITY_LABEL_REFUTE,
        target_error_type=target_error_type, paired=paired,
    )


def make_ne_row(family: str, subtype: str, row_index: int, claim: str, evidence: str,
                 frame_compatible_label: int, predicate_covered_label: int) -> dict[str, Any]:
    return make_row(
        family=family, subtype=subtype, row_index=row_index, claim=claim, evidence=evidence,
        final_label="NOT_ENTITLED", frame_compatible_label=frame_compatible_label,
        predicate_covered_label=predicate_covered_label, sufficiency_label=0,
        polarity_label=POLARITY_LABEL_NOT_ENTITLED,
        target_error_type=FAMILY_PLAN[FAMILY_SNES]["target_error"], paired=False,
    )


# ---------------------------------------------------------------------------
# Family 1: support_entitlement_direct_recovery_v2 (240 rows, SUPPORT only)
# ---------------------------------------------------------------------------
_SEDR_SUBTYPES = [
    "entity_attribute_support",
    "role_work_title_membership_support",
    "date_numeric_support",
    "paraphrased_evidence_support",
]


def build_support_entitlement_direct_recovery(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    places = CyclicPool(rng, PLACE_NAMES)
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    roles = CyclicPool(rng, ROLES)
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)
    products = CyclicPool(rng, PRODUCTS)
    qualities = CyclicPool(rng, QUALITIES)
    adjectives = CyclicPool(rng, POSITIVE_ADJECTIVES)

    target_error = FAMILY_PLAN[FAMILY_SEDR]["target_error"]
    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_SEDR_SUBTYPES))
    idx = 0
    for subtype, count in zip(_SEDR_SUBTYPES, counts):
        for _ in range(count):
            idx += 1

            if subtype == "entity_attribute_support":
                if rng.random() < 0.5:
                    name = people.next()
                    place = places.next()
                    year = rng.randint(1955, 2015)
                    claim = f"{name} holds citizenship of {place}."
                    evidence = f"{name} has held {place} citizenship since {year} and resides there."
                else:
                    org = orgs.next()
                    place = places.next()
                    year = rng.randint(1900, 2020)
                    claim = f"{org} is headquartered in {place}."
                    evidence = f"{org}'s headquarters has been located in {place} since it was founded in {year}."
            elif subtype == "role_work_title_membership_support":
                if rng.random() < 0.5:
                    name = people.next()
                    role = roles.next()
                    org = orgs.next()
                    year = rng.randint(1990, 2023)
                    claim = f"{name} is the {role} of {org}."
                    evidence = f"{name} has served as {org}'s {role} since {year}."
                else:
                    artifact = artifacts.next()
                    category = categories.next()
                    name = people.next()
                    year = rng.randint(1960, 2024)
                    claim = f"{artifact} is a {category} by {name}."
                    evidence = f"{artifact}, first released in {year}, is a {category} credited to {name}."
            elif subtype == "date_numeric_support":
                if rng.random() < 0.5:
                    name = people.next()
                    year = rng.randint(1930, 2010)
                    place = places.next()
                    claim = f"{name} was born in {year}."
                    evidence = f"{name} was born in {year}, in {place}."
                else:
                    org = orgs.next()
                    domain_name, unit, low, high = domains.next()
                    value = rng.randint(low, high)
                    year = rng.randint(1995, 2024)
                    claim = f"{org} recorded {value} {unit} in {domain_name} during {year}."
                    evidence = f"{org}'s {year} annual report lists {value} {unit} in {domain_name}."
            else:  # paraphrased_evidence_support
                if rng.random() < 0.5:
                    org = orgs.next()
                    product = products.next()
                    threshold = rng.randint(100, 5000)
                    actual = threshold + rng.randint(50, 800)
                    claim = f"{org} sold more than {threshold} {product}."
                    evidence = f"{org} moved {actual} units of {product}, comfortably above the {threshold} mark."
                else:
                    artifact = artifacts.next()
                    quality = qualities.next()
                    adjective = adjectives.next()
                    score = rng.randint(70, 99)
                    claim = f"{artifact} received a positive critical reception."
                    evidence = (
                        f"Reviewers called {artifact} {adjective}, singling out its {quality} for "
                        f"praise, and audience scores climbed to {score} percent."
                    )

            rows.append(make_support_row(FAMILY_SEDR, subtype, idx, claim, evidence, target_error))
    return rows


# ---------------------------------------------------------------------------
# Family 2: refute_entitlement_direct_recovery_v2 (220 rows, REFUTE only)
# ---------------------------------------------------------------------------
_REDR_SUBTYPES = [
    "type_mismatch_refute",
    "wrong_date_count_refute",
    "exclusive_only_refute",
    "wrong_entity_work_role_refute",
]


def build_refute_entitlement_direct_recovery(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    other_people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    other_orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)
    roles = CyclicPool(rng, ROLES)
    artifacts = CyclicPool(rng, ARTIFACT_TITLES)
    categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    other_categories = CyclicPool(rng, ARTIFACT_CATEGORIES)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    target_error = FAMILY_PLAN[FAMILY_REDR]["target_error"]
    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_REDR_SUBTYPES))
    idx = 0
    for subtype, count in zip(_REDR_SUBTYPES, counts):
        for _ in range(count):
            idx += 1

            if subtype == "type_mismatch_refute":
                artifact = artifacts.next()
                category = categories.next()
                other_category = other_categories.next_distinct_from(category)
                year = rng.randint(1960, 2024)
                claim = f"{artifact} is a {category}."
                evidence = f"As of {year}, {artifact} is in fact a {other_category}, not a {category}."
            elif subtype == "wrong_date_count_refute":
                if rng.random() < 0.5:
                    name = people.next()
                    year = rng.randint(1930, 2015)
                    other_year = year + rng.choice([-25, -14, -7, 8, 16, 29])
                    claim = f"{name} was born in {year}."
                    evidence = f"{name} was actually born in {other_year}, not {year}."
                else:
                    org = orgs.next()
                    domain_name, unit, low, high = domains.next()
                    actual = rng.randint(low, max(low + 1, high - 500))
                    threshold = actual + rng.randint(50, 500)
                    claim = f"{org} recorded {threshold} {unit} in {domain_name}."
                    evidence = f"{org} actually recorded {actual} {unit} in {domain_name}, not {threshold}."
            elif subtype == "exclusive_only_refute":
                if rng.random() < 0.5:
                    org = orgs.next()
                    place = places.next()
                    other_org = other_orgs.next_distinct_from(org)
                    year = rng.randint(1960, 2024)
                    claim = f"{org} is the only organization based in {place}."
                    evidence = f"As of {year}, both {org} and {other_org} are based in {place}."
                else:
                    name = people.next()
                    profession = professions.next()
                    other_profession = other_professions.next_distinct_from(profession)
                    year = rng.randint(1990, 2024)
                    claim = f"{name} works exclusively as a {profession}."
                    evidence = (
                        f"According to a {year} directory listing, {name} works as both a "
                        f"{profession} and a {other_profession}."
                    )
            else:  # wrong_entity_work_role_refute
                if rng.random() < 0.5:
                    artifact = artifacts.next()
                    name = people.next()
                    other_name = other_people.next_distinct_from(name)
                    claim = f"{artifact} was created by {name}."
                    evidence = f"{artifact} was actually created by {other_name}, not {name}."
                else:
                    name = people.next()
                    role = roles.next()
                    org = orgs.next()
                    other_name = other_people.next_distinct_from(name)
                    claim = f"{name} is the {role} of {org}."
                    evidence = f"{other_name}, not {name}, is the {role} of {org}."

            rows.append(make_refute_row(FAMILY_REDR, subtype, idx, claim, evidence, target_error))
    return rows


# ---------------------------------------------------------------------------
# Family 3: numeric_temporal_polarity_comparison_v2 (260 rows: 130 SUPPORT + 130 REFUTE)
# ---------------------------------------------------------------------------
_NTPC_SUBTYPES = [
    "more_than_fewer_than",
    "at_least_under",
    "before_after",
    "exact_threshold_contradiction",
]


def build_numeric_temporal_polarity_comparison(rng: Random, n_pairs: int) -> list[dict[str, Any]]:
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    places = CyclicPool(rng, PLACE_NAMES)
    events = CyclicPool(rng, EVENTS)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_pairs, len(_NTPC_SUBTYPES))
    idx = 0
    for subtype, count in zip(_NTPC_SUBTYPES, counts):
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

            rows.append(make_support_row(
                FAMILY_NTPC, subtype, idx, support_claim, evidence,
                ROW_TARGET_ERROR_SUPPORT, paired=True,
            ))
            rows.append(make_refute_row(
                FAMILY_NTPC, subtype, idx, refute_claim, evidence,
                ROW_TARGET_ERROR_REFUTE, paired=True,
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 4: lexical_type_polarity_disambiguation_v2 (220 rows: 110 SUPPORT + 110 REFUTE)
# ---------------------------------------------------------------------------
_LTPD_SUBTYPES = [
    "same_surface_wrong_type",
    "person_org_place_mismatch",
    "work_title_vs_creator",
    "category_membership_vs_lexical_overlap",
]


def build_lexical_type_polarity_disambiguation(rng: Random, n_pairs: int) -> list[dict[str, Any]]:
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
    counts = distribute(n_pairs, len(_LTPD_SUBTYPES))
    idx = 0
    for subtype, count in zip(_LTPD_SUBTYPES, counts):
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
                refute_claim = f"{full_name} is a city located near {place}."
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

            rows.append(make_support_row(
                FAMILY_LTPD, subtype, idx, support_claim, evidence,
                ROW_TARGET_ERROR_SUPPORT, paired=True,
            ))
            rows.append(make_refute_row(
                FAMILY_LTPD, subtype, idx, refute_claim, evidence,
                ROW_TARGET_ERROR_REFUTE, paired=True,
            ))
    return rows


# ---------------------------------------------------------------------------
# Family 5: strict_ne_external_style_safety_v2 (80 rows, NOT_ENTITLED only)
# ---------------------------------------------------------------------------
_SNES_SUBTYPES = [
    "partial_evidence_missing_decisive_field",
    "conjunction_only_one_conjunct_supported",
    "entity_present_predicate_absent",
    "near_threshold_numeric_insufficiency",
]


def build_strict_ne_external_style_safety(rng: Random, n_total: int) -> list[dict[str, Any]]:
    people = NamePool(rng, list(itertools.product(FIRST_NAMES, LAST_NAMES)))
    orgs = NamePool(rng, list(itertools.product(ORG_ROOTS, ORG_SUFFIXES)))
    professions = CyclicPool(rng, PROFESSIONS)
    other_professions = CyclicPool(rng, PROFESSIONS)
    awards = CyclicPool(rng, AWARDS)
    domains = CyclicPool(rng, NUMERIC_DOMAINS)

    rows: list[dict[str, Any]] = []
    counts = distribute(n_total, len(_SNES_SUBTYPES))
    idx = 0
    for subtype, count in zip(_SNES_SUBTYPES, counts):
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
            else:  # near_threshold_numeric_insufficiency
                org = orgs.next()
                domain_name, unit, low, high = domains.next()
                threshold = rng.randint(low + 50, high)
                claim = f"{org} reported more than {threshold} {unit} in {domain_name}."
                evidence = (
                    f"{org} reported activity in {domain_name} near {threshold} {unit}, "
                    "without confirming whether the figure exceeded that amount."
                )
                frame_compatible, predicate_covered = 1, 1

            rows.append(make_ne_row(
                FAMILY_SNES, subtype, idx, claim, evidence, frame_compatible, predicate_covered,
            ))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_dataset(seed: int) -> list[dict[str, Any]]:
    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    rows.extend(build_support_entitlement_direct_recovery(
        Random(rng.random()), FAMILY_PLAN[FAMILY_SEDR]["planned_rows"]))
    rows.extend(build_refute_entitlement_direct_recovery(
        Random(rng.random()), FAMILY_PLAN[FAMILY_REDR]["planned_rows"]))
    rows.extend(build_numeric_temporal_polarity_comparison(
        Random(rng.random()), FAMILY_PLAN[FAMILY_NTPC]["planned_rows"] // 2))
    rows.extend(build_lexical_type_polarity_disambiguation(
        Random(rng.random()), FAMILY_PLAN[FAMILY_LTPD]["planned_rows"] // 2))
    rows.extend(build_strict_ne_external_style_safety(
        Random(rng.random()), FAMILY_PLAN[FAMILY_SNES]["planned_rows"]))

    if len(rows) != STAGE75B_TOTAL_ROWS:
        raise RuntimeError(f"generated {len(rows)} rows, expected {STAGE75B_TOTAL_ROWS}")
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
    if summary.get("planned_row_count") != STAGE75B_TOTAL_ROWS:
        mismatches.append("summary.planned_row_count differs from STAGE75B_TOTAL_ROWS")
    if summary.get("planned_label_mix") != STAGE75B_LABEL_PLAN:
        mismatches.append("summary.planned_label_mix differs from STAGE75B_LABEL_PLAN")

    design_families = {f["family"]: f for f in design.get("families", [])}
    for family, plan in FAMILY_PLAN.items():
        design_family = design_families.get(family)
        if design_family is None:
            mismatches.append(f"family '{family}' missing from design file")
            continue
        if design_family.get("planned_rows") != plan["planned_rows"]:
            mismatches.append(f"family '{family}' planned_rows differs from design file")
        if design_family.get("label_mix") != plan["label_plan"]:
            mismatches.append(f"family '{family}' label_mix differs from design file")
        if design_family.get("target_error") != plan["target_error"]:
            mismatches.append(f"family '{family}' target_error differs from design file")

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
        # Deliberately a local, lazy import (not moved to module scope) so this
        # generator has no hard torch/train_controlled_v5 dependency for its
        # core row-building path; only this compatibility check needs it.
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
    family_counts = Counter(row["bridge_family"] for row in rows)
    family_label_counts: dict[str, Counter] = {}
    for row in rows:
        family_label_counts.setdefault(row["bridge_family"], Counter())[row["final_label"]] += 1

    extra_families = sorted(set(family_counts.keys()) - set(FAMILY_PLAN.keys()))
    missing_families = sorted(set(FAMILY_PLAN.keys()) - set(family_counts.keys()))
    if extra_families:
        mismatches.append(f"unexpected bridge_family value(s) present: {extra_families}")
    if missing_families:
        mismatches.append(f"expected bridge_family value(s) missing: {missing_families}")

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
    for label_name, planned_count in STAGE75B_LABEL_PLAN.items():
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
        for field, value in row.items():
            if field in FORBIDDEN_SCAN_EXCLUDE_FIELDS:
                continue
            text = str(value).lower()
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    hits.append({"id": row["id"], "field": field, "marker": marker})
    return {"passed": len(hits) == 0, "markers_checked": FORBIDDEN_MARKERS, "hit_count": len(hits), "hit_examples": hits[:20]}


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace, design: dict[str, Any] | None) -> dict[str, Any]:
    label_counts = Counter(row["final_label"] for row in rows)
    family_counts = Counter(row["bridge_family"] for row in rows)

    family_label_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        fam = row["bridge_family"]
        lbl = row["final_label"]
        family_label_counts.setdefault(fam, Counter())[lbl] += 1
    family_label_counts = {fam: dict(counter) for fam, counter in family_label_counts.items()}

    required_field_check = check_required_fields(rows)
    duplicate_id_check = check_duplicate_ids(rows)
    duplicate_pair_check = check_duplicate_claim_evidence_pairs(rows)
    final_label_check = check_final_label_values(rows)
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

    row_count_ok = len(rows) == STAGE75B_TOTAL_ROWS

    all_checks_passed = (
        row_count_ok
        and required_field_check["passed"]
        and duplicate_id_check["passed"]
        and duplicate_pair_check["passed"]
        and final_label_check["passed"]
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
    decision = STAGE75B_DECISION_READY if all_checks_passed else STAGE75B_DECISION_FAILED

    leakage_checks = dict(LEAKAGE_POLICY)
    leakage_checks["forbidden_marker_scan"] = forbidden_marker_scan

    return {
        "stage": STAGE,
        "decision": decision,
        "source_stage75a": str(Path(args.design_json)).replace("\\", "/"),
        "source_stage74_aggregate_only": True,
        "output_jsonl": str(Path(args.output_jsonl)).replace("\\", "/"),
        "generation_config": {"seed": args.seed},
        "row_count": len(rows),
        "expected_row_count": STAGE75B_TOTAL_ROWS,
        "label_counts": dict(label_counts),
        "expected_label_counts": STAGE75B_LABEL_PLAN,
        "family_counts": dict(family_counts),
        "family_label_counts": family_label_counts,
        "bridge_families": BRIDGE_FAMILY_ORDER,
        "required_field_checks": required_field_check,
        "leakage_checks": leakage_checks,
        "no_pair_id_check": no_pair_id_check,
        "synthetic_only_check": synthetic_only_check,
        "external_text_used_check": external_text_used_check,
        "external_label_used_check": external_label_used_check,
        "duplicate_id_count": duplicate_id_check["duplicate_count"],
        "duplicate_claim_evidence_count": duplicate_pair_check["duplicate_count"],
        "checks": {
            "row_count_check": {"passed": row_count_ok, "row_count": len(rows), "expected": STAGE75B_TOTAL_ROWS},
            "required_fields_present": required_field_check,
            "duplicate_id_check": duplicate_id_check,
            "duplicate_claim_evidence_pair_check": duplicate_pair_check,
            "final_label_value_check": final_label_check,
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
            "Stage74 residual external error counts (false_NE_total=323, "
            "polarity_error_total=244, false_entitlement_total=80) were used only to "
            "set family quotas in the Stage75A design; no Stage74 residual example "
            "claim/evidence text, VitaminC text, or VitaminC labels were read or used "
            "to produce any row in this file.",
            "No field named 'pair_id' is emitted. Optional grouping metadata for the "
            "two polarity-comparison families uses 'bridge_pair_id' instead, so this "
            "bridge cannot be accidentally swept into the intervention pairwise-loss "
            "grouping path that keys off 'pair_id'.",
            "This script performs no training, no smoke run, no mini-run, no full run, "
            "no OOD or external evaluation, and does not modify "
            "scripts/train_controlled_v6b_minimal.py, existing Stage57/66 data, or any "
            "existing Stage73/74/75A report.",
        ],
        "recommended_next_stage": "Stage75C static audit and runner integration plan",
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def render_markdown(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Stage75B: Targeted Residual Bridge Generation Report")
    lines.append("")
    lines.append(f"**Decision:** `{report['decision']}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| stage | {report['stage']} |")
    lines.append(f"| decision | {report['decision']} |")
    lines.append(f"| source_stage75a | {report['source_stage75a']} |")
    lines.append(f"| source_stage74_aggregate_only | {report['source_stage74_aggregate_only']} |")
    lines.append(f"| output_jsonl | {report['output_jsonl']} |")
    lines.append(f"| seed | {report['generation_config']['seed']} |")
    lines.append(f"| row_count | {report['row_count']} (expected {report['expected_row_count']}) |")
    lines.append(f"| duplicate_id_count | {report['duplicate_id_count']} |")
    lines.append(f"| duplicate_claim_evidence_count | {report['duplicate_claim_evidence_count']} |")
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
    lines.append("| family | count | planned | target_error |")
    lines.append("|---|---|---|---|")
    for family in report["bridge_families"]:
        planned = FAMILY_PLAN[family]["planned_rows"]
        target_error = FAMILY_PLAN[family]["target_error"]
        lines.append(f"| {family} | {report['family_counts'].get(family, 0)} | {planned} | {target_error} |")
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
    lines.append("| family | subtype | final_label | target_error_type | claim | evidence |")
    lines.append("|---|---|---|---|---|---|")
    seen_family_subtype: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["bridge_family"], row["bridge_subtype"], row["final_label"])
        if key in seen_family_subtype:
            continue
        seen_family_subtype.add(key)
        claim = row["claim"].replace("|", "/")
        evidence = row["evidence"].replace("|", "/")
        lines.append(
            f"| {row['bridge_family']} | {row['bridge_subtype']} | {row['final_label']} | "
            f"{row['target_error_type']} | {claim} | {evidence} |"
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--report-json", type=str, default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", type=str, default=str(DEFAULT_REPORT_MD))
    parser.add_argument("--design-json", type=str, default=str(DEFAULT_DESIGN_JSON))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting existing output files. Without this flag, "
                              "the script fails safely if any output file already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_paths = [Path(args.output_jsonl), Path(args.report_json), Path(args.report_md)]
    if not args.overwrite:
        existing = [str(p) for p in output_paths if p.exists()]
        if existing:
            print(
                "[Stage75B] refusing to overwrite existing output file(s) without --overwrite: "
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
    output_report_md_path.write_text(render_markdown(report, rows), encoding="utf-8")

    print(f"[Stage75B] decision: {report['decision']}")
    print(f"[Stage75B] wrote {len(rows)} rows to {output_jsonl_path}")
    print(f"[Stage75B] wrote report JSON to {output_report_json_path}")
    print(f"[Stage75B] wrote report Markdown to {output_report_md_path}")


if __name__ == "__main__":
    main()
