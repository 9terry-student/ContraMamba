from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

DESIGN_FREEZE_COMMIT = "4b6f0c831ebc89f0e786e1eb526739c7c9c06413"
DESIGN_REL = Path(
    "reports/reason_router_gen4_generator_family_prevalence_transportability_design_candidate.md"
)
DESIGN_SHA256 = "ffc1705ef5c68ff0a737bee088d93285f18f4093e0127dc94f99fc60be7b53d5"

SOURCE_PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = SOURCE_PAIR_COUNT * ROWS_PER_PAIR
MECHANISM_ID = "masked_slot_substitution_v1"

SOURCE_FILE = "structured_source_facts.jsonl"
ROW_FILE = "synthetic_reason_router_six_cell.jsonl"

SOURCE_FIELDS = (
    "schema_version",
    "generator_family",
    "pair_id",
    "title",
    "name",
    "role",
    "predicate",
    "object",
    "time",
    "location",
    "alternate_title",
    "alternate_name",
    "alternate_role",
    "alternate_predicate",
)

ROW_FIELDS = (
    "schema_version",
    "generator_family",
    "mechanism_id",
    "source_pair_id",
    "row_id",
    "contrast_cell_id",
    "axis_mask",
    "intended_changed_axes",
    "generator_source_fields",
    "claim",
    "evidence",
)

CANONICAL_AXIS_ORDER = ("title", "name", "role", "predicate")

CELL_SPEC = (
    ("C0_SHAM", (0, 0, 0, 0), ()),
    ("C1_TITLE", (1, 0, 0, 0), (("title", "alternate_title"),)),
    ("C2_NAME", (0, 1, 0, 0), (("name", "alternate_name"),)),
    ("C3_ROLE", (0, 0, 1, 0), (("role", "alternate_role"),)),
    ("C4_PREDICATE", (0, 0, 0, 1), (("predicate", "alternate_predicate"),)),
    (
        "C5_TITLE_NAME",
        (1, 1, 0, 0),
        (("title", "alternate_title"), ("name", "alternate_name")),
    ),
)
CELL_IDS = tuple(cell_id for cell_id, _, _ in CELL_SPEC)

FORBIDDEN_FIELDS = frozenset(
    {
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "primary_failure_type",
        "predictions",
        "prediction",
        "logits",
        "probabilities",
        "evaluator_outputs",
        "error_cohort",
        "training_outcomes",
        "evaluation_outcomes",
        "target_C",
        "reference_C",
        "alignment_shift_abs",
        "regime",
        "baseline_plus_path_efficiency",
        "baseline_minus_path_efficiency",
        "delta_baseline",
        "alignment_plus_path_efficiency",
        "alignment_minus_path_efficiency",
        "delta_alignment",
        "R_ALIGN",
        "absolute_anchor_token_index",
        "terminal_index",
        "post4_eligible",
        "exclusion_code",
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    }
)


class PrevalenceCohortBuildError(ValueError):
    pass


@dataclass(frozen=True)
class Schedule:
    name_stride: int
    name_offset: int
    name_alt_offset: int
    title_stride: int
    title_offset: int
    title_alt_offset: int
    role_stride: int
    role_offset: int
    role_alt_offset: int
    predicate_stride: int
    predicate_offset: int
    location_stride: int
    location_offset: int
    time_stride: int
    time_offset: int
    object_stride: int
    object_offset: int


@dataclass(frozen=True)
class FamilySpec:
    key: str
    generator_family: str
    source_schema: str
    row_schema: str
    names: tuple[str, ...]
    titles: tuple[str, ...]
    roles: tuple[str, ...]
    predicate_pairs: tuple[tuple[str, str], ...]
    object_stems: tuple[str, ...]
    locations: tuple[str, ...]
    times: tuple[str, ...]
    object_tag: str
    schedule: Schedule
    renderer: Callable[[Mapping[str, Any], Mapping[str, str]], str]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrevalenceCohortBuildError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def _render_xg2(fact: Mapping[str, Any], overrides: Mapping[str, str]) -> str:
    v = {**dict(fact), **dict(overrides)}
    return (
        f"At {v['time']}, the bulletin from {v['location']} lists "
        f"{v['title']} {v['name']} as {v['role']} and records that this person "
        f"{v['predicate']} {v['object']}."
    )


def _render_xg3(fact: Mapping[str, Any], overrides: Mapping[str, str]) -> str:
    v = {**dict(fact), **dict(overrides)}
    return (
        f"A dispatch dated {v['time']} in {v['location']} names {v['name']}, "
        f"bearing the title {v['title']}, as {v['role']}; the dispatch states that "
        f"this person {v['predicate']} {v['object']}."
    )


def _render_xg4(fact: Mapping[str, Any], overrides: Mapping[str, str]) -> str:
    v = {**dict(fact), **dict(overrides)}
    return (
        f"From {v['location']} during {v['time']}, the register records "
        f"{v['title']} {v['name']} in the role of {v['role']}; its entry says "
        f"this person {v['predicate']} {v['object']}."
    )


XG2_NAMES = (
    "Arel Quorin",
    "Bexa Nymor",
    "Ciren Talvek",
    "Dova Iskar",
    "Eris Pellan",
    "Faro Quess",
    "Galen Orrix",
    "Hessa Vey",
    "Joren Calyx",
    "Kiva Morren",
    "Leto Pyran",
    "Mina Sorrel",
    "Oren Tavis",
    "Pela Zorin",
    "Rian Volis",
    "Sera Kyne",
)
XG2_TITLES = (
    "Beacon Keeper",
    "Civic Reader",
    "Harbor Signer",
    "Ledger Speaker",
    "Route Keeper",
    "Signal Reader",
    "Treaty Bearer",
    "Vault Speaker",
)
XG2_ROLES = (
    "beacon archive keeper",
    "civic route recorder",
    "harbor signal custodian",
    "ledger review convenor",
    "public notice curator",
    "route ledger mediator",
    "signal docket keeper",
    "treaty record examiner",
)
XG2_PREDICATE_PAIRS = (
    ("beacon-countersigned", "beacon-voided"),
    ("beacon-blueprinted", "beacon-redrafted"),
    ("beacon-indexed", "beacon-delisted"),
    ("beacon-ratified", "beacon-rescinded"),
    ("beacon-sequenced", "beacon-desynchronized"),
    ("beacon-sealed", "beacon-unsealed"),
    ("beacon-stamped", "beacon-countermanded"),
    ("beacon-tabulated", "beacon-recounted"),
    ("beacon-witnessed", "beacon-withdrawn"),
)
XG2_OBJECT_STEMS = (
    "the amber beacon atlas",
    "the birch route tablet",
    "the copper notice folio",
    "the drift signal chart",
    "the ember treaty sheet",
    "the flax harbor ledger",
    "the garnet civic docket",
    "the hazel transit roll",
    "the indigo beacon folio",
    "the jade route slate",
    "the kelp notice register",
    "the lilac signal tablet",
    "the ochre treaty ledger",
    "the pearl harbor chart",
    "the russet civic roll",
    "the silver transit folio",
)
XG2_LOCATIONS = (
    "Amber Loom District",
    "Birch Crown Harbor",
    "Copper Finch Reach",
    "Drift Lantern Quay",
    "Ember Reed Ward",
    "Flax Meridian Port",
    "Garnet Hollow Gate",
    "Hazel Spire Basin",
    "Indigo Crest Dock",
    "Jade Lantern Reach",
    "Kelp Crown Station",
    "Lilac Meridian Ward",
    "Ochre Finch Port",
    "Pearl Hollow Quay",
    "Russet Spire Gate",
    "Silver Reed Basin",
)
XG2_TIMES = (
    "the aurora census interval",
    "the beacon tally interval",
    "the copper review span",
    "the drift audit span",
    "the ember docket interval",
    "the flax notice cycle",
    "the garnet survey span",
    "the hazel ledger cycle",
    "the indigo registry span",
    "the jade census cycle",
    "the kelp audit interval",
    "the lilac notice span",
    "the ochre docket cycle",
    "the pearl survey interval",
    "the russet registry cycle",
    "the silver tally span",
)

XG3_NAMES = (
    "Tovan Elric",
    "Ulya Ferren",
    "Varek Juno",
    "Willa Brast",
    "Xeran Dovel",
    "Yara Fenwick",
    "Zevan Harl",
    "Amina Ilex",
    "Brio Jast",
    "Celia Korven",
    "Darin Lox",
    "Evia Marn",
    "Fenix Neral",
    "Greta Ovik",
    "Halen Prist",
    "Iria Rovel",
)
XG3_TITLES = (
    "Archive Caller",
    "Boundary Reader",
    "Charter Voice",
    "Dispatch Keeper",
    "Field Reader",
    "Marker Voice",
    "Notice Caller",
    "Survey Keeper",
)
XG3_ROLES = (
    "archive route auditor",
    "boundary file keeper",
    "charter signal reader",
    "dispatch record arbiter",
    "field notice custodian",
    "marker docket reviewer",
    "survey file mediator",
    "transit charter examiner",
)
XG3_PREDICATE_PAIRS = (
    ("dispatch-abstracted", "dispatch-restored"),
    ("dispatch-booked", "dispatch-expunged"),
    ("dispatch-crosschecked", "dispatch-unchecked"),
    ("dispatch-docketed", "dispatch-undocketed"),
    ("dispatch-flagged", "dispatch-unflagged"),
    ("dispatch-inscribed", "dispatch-erased"),
    ("dispatch-notarized", "dispatch-disavowed"),
    ("dispatch-plotted", "dispatch-unplotted"),
    ("dispatch-serialized", "dispatch-deserialized"),
)
XG3_OBJECT_STEMS = (
    "the topaz boundary card",
    "the umbra charter map",
    "the verdant dispatch sheet",
    "the winter field tablet",
    "the xenon marker roll",
    "the yellow notice atlas",
    "the zircon survey card",
    "the apricot transit map",
    "the bronze boundary sheet",
    "the coral charter tablet",
    "the denim dispatch roll",
    "the ecru field atlas",
    "the fawn marker card",
    "the grape notice map",
    "the ivory survey sheet",
    "the khaki transit tablet",
)
XG3_LOCATIONS = (
    "Topaz Vale Annex",
    "Umbra Pike Crossing",
    "Verdant Bell Yard",
    "Winter Arch Point",
    "Xenon Vale Depot",
    "Yellow Pike Annex",
    "Zircon Bell Crossing",
    "Apricot Arch Yard",
    "Bronze Vale Point",
    "Coral Pike Depot",
    "Denim Bell Annex",
    "Ecru Arch Crossing",
    "Fawn Vale Yard",
    "Grape Pike Point",
    "Ivory Bell Depot",
    "Khaki Arch Annex",
)
XG3_TIMES = (
    "the topaz filing passage",
    "the umbra dispatch passage",
    "the verdant marker passage",
    "the winter survey passage",
    "the xenon boundary passage",
    "the yellow charter passage",
    "the zircon transit passage",
    "the apricot filing passage",
    "the bronze dispatch passage",
    "the coral marker passage",
    "the denim survey passage",
    "the ecru boundary passage",
    "the fawn charter passage",
    "the grape transit passage",
    "the ivory filing passage",
    "the khaki dispatch passage",
)

XG4_NAMES = (
    "Jessa Auvin",
    "Kalen Brex",
    "Luma Corris",
    "Merek Dain",
    "Nira Esol",
    "Olan Fray",
    "Perri Gant",
    "Quila Hest",
    "Rovan Jex",
    "Suri Kael",
    "Taren Lume",
    "Una Mox",
    "Vela Neris",
    "Wren Ordan",
    "Xara Peth",
    "Yorin Rell",
)
XG4_TITLES = (
    "Circuit Herald",
    "Dock Herald",
    "Index Herald",
    "Panel Keeper",
    "Quorum Reader",
    "Record Herald",
    "Station Reader",
    "Vector Keeper",
)
XG4_ROLES = (
    "circuit record custodian",
    "dock index examiner",
    "index panel mediator",
    "panel vector auditor",
    "quorum station keeper",
    "record circuit reviewer",
    "station quorum curator",
    "vector dock recorder",
)
XG4_PREDICATE_PAIRS = (
    ("register-annotated", "register-stripped"),
    ("register-catalogued", "register-decatalogued"),
    ("register-encoded", "register-decoded"),
    ("register-gridded", "register-ungridded"),
    ("register-keyed", "register-unkeyed"),
    ("register-mapped", "register-demapped"),
    ("register-numbered", "register-denumbered"),
    ("register-posted", "register-unposted"),
    ("register-tagged", "register-untagged"),
)
XG4_OBJECT_STEMS = (
    "the aqua circuit panel",
    "the black dock index",
    "the cream quorum card",
    "the dusk record vector",
    "the eggshell station panel",
    "the fern circuit index",
    "the gold dock card",
    "the heather quorum vector",
    "the ice record panel",
    "the jet station index",
    "the lime circuit card",
    "the mauve dock vector",
    "the navy quorum panel",
    "the peach record index",
    "the rose station card",
    "the teal circuit vector",
)
XG4_LOCATIONS = (
    "Aqua Ridge Forum",
    "Black Cedar Court",
    "Cream Delta Hall",
    "Dusk Ridge Forum",
    "Eggshell Cedar Court",
    "Fern Delta Hall",
    "Gold Ridge Forum",
    "Heather Cedar Court",
    "Ice Delta Hall",
    "Jet Ridge Forum",
    "Lime Cedar Court",
    "Mauve Delta Hall",
    "Navy Ridge Forum",
    "Peach Cedar Court",
    "Rose Delta Hall",
    "Teal Ridge Forum",
)
XG4_TIMES = (
    "the aqua registry turn",
    "the black index turn",
    "the cream panel turn",
    "the dusk vector turn",
    "the eggshell station turn",
    "the fern circuit turn",
    "the gold dock turn",
    "the heather quorum turn",
    "the ice registry turn",
    "the jet index turn",
    "the lime panel turn",
    "the mauve vector turn",
    "the navy station turn",
    "the peach circuit turn",
    "the rose dock turn",
    "the teal quorum turn",
)


FAMILY_SPECS: dict[str, FamilySpec] = {
    "xg2": FamilySpec(
        key="xg2",
        generator_family="xg2_prevalence_bulletin_v1",
        source_schema="GEN4_XG2_PREVALENCE_STRUCTURED_SOURCE_FACT_V1",
        row_schema="GEN4_XG2_PREVALENCE_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1",
        names=XG2_NAMES,
        titles=XG2_TITLES,
        roles=XG2_ROLES,
        predicate_pairs=XG2_PREDICATE_PAIRS,
        object_stems=XG2_OBJECT_STEMS,
        locations=XG2_LOCATIONS,
        times=XG2_TIMES,
        object_tag="B",
        schedule=Schedule(5, 1, 9, 3, 2, 5, 5, 1, 6, 7, 2, 9, 3, 11, 4, 13, 6),
        renderer=_render_xg2,
    ),
    "xg3": FamilySpec(
        key="xg3",
        generator_family="xg3_prevalence_dispatch_v1",
        source_schema="GEN4_XG3_PREVALENCE_STRUCTURED_SOURCE_FACT_V1",
        row_schema="GEN4_XG3_PREVALENCE_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1",
        names=XG3_NAMES,
        titles=XG3_TITLES,
        roles=XG3_ROLES,
        predicate_pairs=XG3_PREDICATE_PAIRS,
        object_stems=XG3_OBJECT_STEMS,
        locations=XG3_LOCATIONS,
        times=XG3_TIMES,
        object_tag="D",
        schedule=Schedule(7, 3, 11, 5, 1, 6, 3, 2, 7, 5, 4, 11, 5, 13, 7, 9, 2),
        renderer=_render_xg3,
    ),
    "xg4": FamilySpec(
        key="xg4",
        generator_family="xg4_prevalence_register_v1",
        source_schema="GEN4_XG4_PREVALENCE_STRUCTURED_SOURCE_FACT_V1",
        row_schema="GEN4_XG4_PREVALENCE_SIX_CELL_MASKED_SLOT_SUBSTITUTION_V1",
        names=XG4_NAMES,
        titles=XG4_TITLES,
        roles=XG4_ROLES,
        predicate_pairs=XG4_PREDICATE_PAIRS,
        object_stems=XG4_OBJECT_STEMS,
        locations=XG4_LOCATIONS,
        times=XG4_TIMES,
        object_tag="R",
        schedule=Schedule(3, 4, 12, 7, 0, 3, 7, 3, 5, 4, 1, 13, 6, 9, 5, 7, 4),
        renderer=_render_xg4,
    ),
}
FAMILY_KEYS = tuple(FAMILY_SPECS)


def family_spec(key: str) -> FamilySpec:
    try:
        return FAMILY_SPECS[key]
    except KeyError as exc:
        raise PrevalenceCohortBuildError(f"UNKNOWN_FAMILY:{key}") from exc


def expected_pair_ids(spec: FamilySpec) -> list[str]:
    return [f"{spec.key}_fact_{index:03d}" for index in range(1, SOURCE_PAIR_COUNT + 1)]


def render_statement(
    spec: FamilySpec,
    fact: Mapping[str, Any],
    **overrides: str,
) -> str:
    return spec.renderer(fact, overrides)


def _select(values: Sequence[str], i: int, stride: int, offset: int) -> str:
    return str(values[(stride * i + offset) % len(values)])


def _source_fact(spec: FamilySpec, index_zero_based: int) -> dict[str, Any]:
    require(0 <= index_zero_based < SOURCE_PAIR_COUNT, "SOURCE_INDEX_RANGE")
    i = index_zero_based
    s = spec.schedule

    name = _select(spec.names, i, s.name_stride, s.name_offset)
    alternate_name = _select(spec.names, i, s.name_stride, s.name_alt_offset)
    title = _select(spec.titles, i, s.title_stride, s.title_offset)
    alternate_title = _select(spec.titles, i, s.title_stride, s.title_alt_offset)
    role = _select(spec.roles, i, s.role_stride, s.role_offset)
    alternate_role = _select(spec.roles, i, s.role_stride, s.role_alt_offset)
    predicate, alternate_predicate = spec.predicate_pairs[
        (s.predicate_stride * i + s.predicate_offset) % len(spec.predicate_pairs)
    ]
    location = _select(spec.locations, i, s.location_stride, s.location_offset)
    time = _select(spec.times, i, s.time_stride, s.time_offset)
    stem = _select(spec.object_stems, i, s.object_stride, s.object_offset)
    object_text = f"{stem} packet {spec.object_tag}{i + 1:03d}"

    row = {
        "schema_version": spec.source_schema,
        "generator_family": spec.generator_family,
        "pair_id": f"{spec.key}_fact_{i + 1:03d}",
        "title": title,
        "name": name,
        "role": role,
        "predicate": predicate,
        "object": object_text,
        "time": time,
        "location": location,
        "alternate_title": alternate_title,
        "alternate_name": alternate_name,
        "alternate_role": alternate_role,
        "alternate_predicate": alternate_predicate,
    }
    validate_source_fact(spec, row)
    return row


def build_source_facts(spec: FamilySpec) -> list[dict[str, Any]]:
    rows = [_source_fact(spec, i) for i in range(SOURCE_PAIR_COUNT)]
    validate_source_facts(spec, rows)
    return rows


def validate_source_fact(spec: FamilySpec, fact: Mapping[str, Any]) -> None:
    require(set(fact.keys()) == set(SOURCE_FIELDS), "SOURCE_FIELD_SET")
    require(not (FORBIDDEN_FIELDS & set(fact)), "SOURCE_FORBIDDEN_FIELD")
    require(fact["schema_version"] == spec.source_schema, "SOURCE_SCHEMA")
    require(fact["generator_family"] == spec.generator_family, "SOURCE_FAMILY")
    for field in SOURCE_FIELDS[2:]:
        require(
            isinstance(fact[field], str) and bool(fact[field]),
            f"SOURCE_STRING:{field}",
        )
    for axis in CANONICAL_AXIS_ORDER:
        alternate = f"alternate_{axis}"
        require(fact[axis] != fact[alternate], f"SOURCE_AXIS_EQUAL:{axis}")


def validate_source_facts(
    spec: FamilySpec,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    require(len(rows) == SOURCE_PAIR_COUNT, "SOURCE_PAIR_COUNT")
    for row in rows:
        validate_source_fact(spec, row)
    ids = [str(row["pair_id"]) for row in rows]
    require(ids == expected_pair_ids(spec), "SOURCE_PAIR_IDS")
    require(len(set(ids)) == SOURCE_PAIR_COUNT, "SOURCE_PAIR_ID_UNIQUENESS")


def cell_spec(
    cell_id: str,
) -> tuple[tuple[int, int, int, int], tuple[tuple[str, str], ...]]:
    for candidate, mask, substitutions in CELL_SPEC:
        if candidate == cell_id:
            return mask, substitutions
    raise PrevalenceCohortBuildError(f"UNKNOWN_CELL:{cell_id}")


def _row_id(pair_id: str, cell_id: str) -> str:
    return f"{pair_id}__{MECHANISM_ID}__{cell_id.lower()}"


def materialize_fact(
    spec: FamilySpec,
    fact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validate_source_fact(spec, fact)
    pair_id = str(fact["pair_id"])
    claim = render_statement(spec, fact)
    rows: list[dict[str, Any]] = []

    for cell_id, mask, substitutions in CELL_SPEC:
        overrides = {axis: str(fact[source]) for axis, source in substitutions}
        evidence = render_statement(spec, fact, **overrides)
        rows.append(
            {
                "schema_version": spec.row_schema,
                "generator_family": spec.generator_family,
                "mechanism_id": MECHANISM_ID,
                "source_pair_id": pair_id,
                "row_id": _row_id(pair_id, cell_id),
                "contrast_cell_id": cell_id,
                "axis_mask": list(mask),
                "intended_changed_axes": [axis for axis, _ in substitutions],
                "generator_source_fields": {
                    axis: source for axis, source in substitutions
                },
                "claim": claim,
                "evidence": evidence,
            }
        )

    validate_materialized_rows(spec, rows, expected_pairs=1)
    return rows


def materialize_facts(
    spec: FamilySpec,
    facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_source_facts(spec, facts)
    output: list[dict[str, Any]] = []
    for fact in facts:
        output.extend(materialize_fact(spec, fact))
    validate_materialized_rows(spec, output, expected_pairs=SOURCE_PAIR_COUNT)
    return output


def validate_materialized_rows(
    spec: FamilySpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
) -> None:
    require(
        len(rows) == expected_pairs * ROWS_PER_PAIR,
        "MATERIALIZED_ROW_COUNT",
    )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()

    for row in rows:
        require(set(row.keys()) == set(ROW_FIELDS), "ROW_FIELD_SET")
        require(not (FORBIDDEN_FIELDS & set(row)), "ROW_FORBIDDEN_FIELD")
        require(row["schema_version"] == spec.row_schema, "ROW_SCHEMA")
        require(row["generator_family"] == spec.generator_family, "ROW_FAMILY")
        require(row["mechanism_id"] == MECHANISM_ID, "ROW_MECHANISM")
        pair_id = str(row["source_pair_id"])
        row_id = str(row["row_id"])
        cell_id = str(row["contrast_cell_id"])
        require(row_id not in seen_ids, f"DUPLICATE_ROW_ID:{row_id}")
        seen_ids.add(row_id)
        mask, substitutions = cell_spec(cell_id)
        require(row["axis_mask"] == list(mask), f"ROW_MASK:{row_id}")
        require(
            row["intended_changed_axes"] == [axis for axis, _ in substitutions],
            f"ROW_AXES:{row_id}",
        )
        require(
            row["generator_source_fields"]
            == {axis: source for axis, source in substitutions},
            f"ROW_SOURCE_FIELDS:{row_id}",
        )
        require(row_id == _row_id(pair_id, cell_id), f"ROW_ID:{row_id}")
        require(
            isinstance(row["claim"], str) and bool(row["claim"]),
            f"ROW_CLAIM:{row_id}",
        )
        require(
            isinstance(row["evidence"], str) and bool(row["evidence"]),
            f"ROW_EVIDENCE:{row_id}",
        )
        grouped[pair_id].append(row)

    require(len(grouped) == expected_pairs, "MATERIALIZED_PAIR_COUNT")
    for pair_id, block in grouped.items():
        require(len(block) == ROWS_PER_PAIR, f"CELL_COUNT:{pair_id}")
        require(
            [str(row["contrast_cell_id"]) for row in block] == list(CELL_IDS),
            f"CELL_ORDER:{pair_id}",
        )
        require(
            len({str(row["claim"]) for row in block}) == 1,
            f"CLAIM_INVARIANCE:{pair_id}",
        )


def configured_inventory_values(spec: FamilySpec) -> set[str]:
    values = (
        set(spec.names)
        | set(spec.titles)
        | set(spec.roles)
        | set(spec.object_stems)
        | set(spec.locations)
        | set(spec.times)
    )
    for pair in spec.predicate_pairs:
        values.update(pair)
    return values


def write_cohorts(output_dir: Path) -> dict[str, dict[str, str]]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    design = ROOT / DESIGN_REL
    require(design.is_file(), "DESIGN_MISSING")
    require(
        hashlib.sha256(design.read_bytes()).hexdigest() == DESIGN_SHA256,
        "DESIGN_SHA256",
    )

    payloads: dict[str, tuple[bytes, bytes]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for key in FAMILY_KEYS:
        spec = family_spec(key)
        facts = build_source_facts(spec)
        rows = materialize_facts(spec, facts)
        source_raw = jsonl_bytes(facts)
        row_raw = jsonl_bytes(rows)
        payloads[key] = (source_raw, row_raw)
        hashes[key] = {
            SOURCE_FILE: sha256_bytes(source_raw),
            ROW_FILE: sha256_bytes(row_raw),
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    for key in FAMILY_KEYS:
        family_dir = output_dir / key
        family_dir.mkdir()
        source_raw, row_raw = payloads[key]
        (family_dir / SOURCE_FILE).write_bytes(source_raw)
        (family_dir / ROW_FILE).write_bytes(row_raw)

    return hashes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind XG2/XG3/XG4 300-pair / 1800-row structural "
            "cohorts for the Gen4-K prevalence-transportability study. "
            "No tokenizer, model, CUDA, or intervention code is used."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hashes = write_cohorts(args.output_dir)
    print("RESULT=PASS_PREVALENCE_COHORT_BUILD")
    for key in FAMILY_KEYS:
        print(f"{key.upper()}_SOURCE_PAIR_COUNT=300")
        print(f"{key.upper()}_ROW_COUNT=1800")
        print(f"{key.upper()}_SOURCE_SHA256={hashes[key][SOURCE_FILE]}")
        print(f"{key.upper()}_ROW_SHA256={hashes[key][ROW_FILE]}")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("INTERVENTION_EXECUTED=False")


if __name__ == "__main__":
    main()
