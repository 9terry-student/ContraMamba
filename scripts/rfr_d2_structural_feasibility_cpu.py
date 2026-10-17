"""RFR-D2 in-memory planning audits; NOT a corpus or a feasibility certificate.

Public API and exact metadata schemas (extra keys are rejected):

* audit_family_registry(families): nonempty sequence of mappings with required
  case_family_id, split_group_id, surface_domain, renderer_family_id,
  lexicon_namespace_id, root_initial_A, root_initial_C, root_initial_U,
  semantic_world_fingerprint, entity_tuple_fingerprint,
  source_attestation_fingerprint, template_family_id, derivation_rule_id,
  oracle_version. Optional: split (train/dev/test), holdout_contract
  (NONE/RENDERER/LEXICON/TEMPLATE/ALL). Every root is one planned family.
* audit_variant_plan(variants, families): metadata-only declarations requiring
  case_family_id, split_group_id, variant_id, update_arm, readout_time,
  query_target, presentation_order_id, paraphrase_family_id. Optional split
  and reverse_polarity (strict bool). One order/paraphrase *realization* must
  declare all 27 arm/readout/query cells; this never creates samples.
* audit_counterbalance_plan(plan, families): plan mapping with plan_version
  'RFR_D2_COUNTERBALANCE_PLAN_V1' and optional inventory. Inventory entries
  declare case_family_id, split_group_id, split ('UNASSIGNED' if absent),
  root_truth_stratum (three-letter T/F), surface_domain, renderer_family_id,
  template_family_id, authentication_marker_family_id, source_name_family_id,
  polarity_wording_family_id, timestamp_presentation_pattern_id, update_arm,
  readout_time, query_target, count (strict positive integer). They contain
  NO source text, labels, model outputs, or generated case rows. Expected
  answer statuses are derived independently from fixed Boolean design logic.

All functions validate without mutation and return reports with status in
PASS_PLAN_CHECK / FAIL / NOT_EVALUATED, plus explicit NOT_EVALUATED for actual
text leakage, statistical independence, tokenizer, model, and provenance.
Malformed metadata and identity collisions raise ValueError (fail closed).
No file I/O, random sampling, corpus materialization, or external execution.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping

FAMILY_KEYS = frozenset((
    "case_family_id", "split_group_id", "surface_domain",
    "renderer_family_id", "lexicon_namespace_id", "root_initial_A",
    "root_initial_C", "root_initial_U", "semantic_world_fingerprint",
    "entity_tuple_fingerprint", "source_attestation_fingerprint",
    "template_family_id", "derivation_rule_id", "oracle_version",
))
VARIANT_KEYS = frozenset((
    "case_family_id", "split_group_id", "variant_id", "update_arm",
    "readout_time", "query_target", "presentation_order_id",
    "paraphrase_family_id",
))
INVENTORY_KEYS = frozenset((
    "case_family_id", "split_group_id", "split", "root_truth_stratum",
    "surface_domain", "renderer_family_id", "template_family_id",
    "authentication_marker_family_id", "source_name_family_id",
    "polarity_wording_family_id", "timestamp_presentation_pattern_id",
    "update_arm", "readout_time", "query_target", "count",
))
ARMS = frozenset(("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"))
READOUTS = frozenset(("PRE", "POST", "DELAYED"))
QUERIES = frozenset(("A", "B", "U"))
SPLITS = frozenset(("train", "dev", "test"))
HOLDOUTS = frozenset(("NONE", "RENDERER", "LEXICON", "TEMPLATE", "ALL"))
CUE_FIELDS = (
    "surface_domain", "renderer_family_id", "template_family_id",
    "authentication_marker_family_id", "source_name_family_id",
    "polarity_wording_family_id", "timestamp_presentation_pattern_id",
    "update_arm", "readout_time", "split",
)
SHORTCUT_FAMILIES = (
    "majority", "t1_assertion_only", "authentication_only",
    "last_mention", "query_only", "rule_agnostic",
    "source_template_only",
)
TRUTH_STRATA = tuple("".join(bits) for bits in (
    (a, c, u) for a in "FT" for c in "FT" for u in "FT"
))
ALL_CELLS = frozenset((a, r, q) for a in ARMS for r in READOUTS for q in QUERIES)
UNVERIFIED = {
    "CONTENT_INDEPENDENCE": "UNVERIFIED",
    "REALIZED_SPLIT_ISOLATION": "NOT_EVALUATED",
    "ACTUAL_LEXICAL_LEAKAGE": "NOT_EVALUATED",
    "ACTUAL_SHORTCUT_PERFORMANCE": "NOT_EVALUATED",
    "TOKENIZER_SERIALIZATION": "NOT_EVALUATED",
    "MODEL_EVALUATION": "NOT_EVALUATED",
    "MATERIALIZATION_PROVENANCE": "NOT_EVALUATED",
    "SCIENCE_RESULT": "NOT_EVALUATED",
}


def _sequence(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("INVALID_NONEMPTY_SEQUENCE:" + name)
    return value


def _schema(value, required, optional, name):
    if not isinstance(value, Mapping):
        raise ValueError("EXPECTED_MAPPING:" + name)
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        raise ValueError("SCHEMA_MISMATCH:" + name)


def _id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("INVALID_IDENTIFIER:" + name)
    return value


def _enum(value, options, name):
    if type(value) is not str or value not in options:
        raise ValueError("INVALID_ENUM:" + name)
    return value


def _strict_bool(value, name):
    if type(value) is not bool:
        raise ValueError("INVALID_BOOLEAN:" + name)
    return value


def _stratum(family):
    return "".join("T" if family[key] else "F" for key in (
        "root_initial_A", "root_initial_C", "root_initial_U"
    ))


def _family_index(families):
    _sequence(families, "families")
    families_by_id = {}
    by_group = {}
    fingerprints = {field: {} for field in (
        "semantic_world_fingerprint", "entity_tuple_fingerprint",
        "source_attestation_fingerprint",
    )}
    per_split = defaultdict(Counter)
    renderers = defaultdict(Counter)
    holdouts = set()
    assigned = 0
    for family in families:
        _schema(family, FAMILY_KEYS, {"split", "holdout_contract"}, "family")
        for field in FAMILY_KEYS - {"root_initial_A", "root_initial_C", "root_initial_U"}:
            _id(family[field], field)
        for field in ("root_initial_A", "root_initial_C", "root_initial_U"):
            _strict_bool(family[field], field)
        _enum(family["derivation_rule_id"], {"B_AND_A_C"}, "derivation_rule_id")
        _enum(family["oracle_version"], {"RFR_D1_ORACLE_V1"}, "oracle_version")
        split = family.get("split")
        if split is not None:
            _enum(split, SPLITS, "split")
            assigned += 1
        holdout = _enum(family.get("holdout_contract", "NONE"), HOLDOUTS,
                        "holdout_contract")
        holdouts.add(holdout)
        fid = family["case_family_id"]
        gid = family["split_group_id"]
        if fid in families_by_id:
            raise ValueError("DUPLICATE_FAMILY_ID")
        if gid in by_group:
            raise ValueError("GROUP_REUSED_ACROSS_FAMILIES")
        families_by_id[fid] = family
        by_group[gid] = fid
        for field, known in fingerprints.items():
            fp = family[field]
            if fp in known:
                raise ValueError("CROSS_FAMILY_FINGERPRINT_COLLISION:" + field)
            known[fp] = fid
        stratum = _stratum(family)
        bucket = split if split is not None else "UNASSIGNED"
        per_split[bucket][stratum] += 1
        renderers[bucket][family["renderer_family_id"]] += 1
    if len(holdouts) != 1:
        raise ValueError("INCONSISTENT_HOLDOUT_CONTRACT")
    holdout = next(iter(holdouts))
    if 0 < assigned < len(families_by_id):
        raise ValueError("PARTIAL_SPLIT_ASSIGNMENT")
    cross_split_renderer = set()
    cross_split_lexicon = set()
    cross_split_template = set()
    for field, policy in (
        ("renderer_family_id", "RENDERER"),
        ("lexicon_namespace_id", "LEXICON"),
        ("template_family_id", "TEMPLATE"),
    ):
        lookup = defaultdict(set)
        for family in families_by_id.values():
            if "split" in family and family["split"] is not None:
                lookup[family[field]].add(family["split"])
        repeated = {k for k, v in lookup.items() if len(v) > 1}
        if field == "renderer_family_id":
            cross_split_renderer = repeated
        if field == "lexicon_namespace_id":
            cross_split_lexicon = repeated
        if field == "template_family_id":
            cross_split_template = repeated
        if holdout in (policy, "ALL") and repeated:
            raise ValueError("CLAIMED_HOLDOUT_REUSE:" + field)
    return (families_by_id, per_split, renderers, holdout,
            cross_split_renderer, cross_split_lexicon, cross_split_template)


def audit_family_registry(families) -> dict:
    """Audit declared *root* identities and a prospective 8-stratum plan."""
    by_id, per_split, renderers, holdout, cross_renderers, cross_lex, cross_template = (
        _family_index(families)
    )
    issues = []
    for split, counts in sorted(per_split.items()):
        if set(counts) != set(TRUTH_STRATA):
            issues.append("MISSING_TRUTH_STRATA:" + split)
        if len(set(counts.values())) != 1:
            issues.append("UNEQUAL_TRUTH_STRATA:" + split)
    status = "FAIL" if issues else "PASS_PLAN_CHECK"
    return {
        "status": status,
        "G2_FAMILY_IDENTITY": "PASS_PLAN_CHECK",
        "G6_DECLARED_STRATUM_CAPACITY": status,
        "declared_root_family_count": len(by_id),
        "declared_family_counts_by_split_stratum": {
            split: dict(sorted(counts.items())) for split, counts in sorted(per_split.items())
        },
        "declared_family_counts_by_split_renderer": {
            split: dict(sorted(counts.items())) for split, counts in sorted(renderers.items())
        },
        "split_assignment_plan": "PASS_PLAN_CHECK" if "UNASSIGNED" not in per_split else "NOT_EVALUATED",
        "holdout_contract": holdout,
        "shared_renderer_across_assigned_splits": sorted(cross_renderers),
        "shared_lexicon_across_assigned_splits": sorted(cross_lex),
        "shared_template_across_assigned_splits": sorted(cross_template),
        "issues": sorted(issues),
        **UNVERIFIED,
    }


def audit_variant_plan(variants, families) -> dict:
    """Check 27 metadata cells per root and order/paraphrase realization."""
    root_audit = audit_family_registry(families)
    by_family = _family_index(families)[0]
    if not isinstance(variants, (list, tuple)):
        raise ValueError("INVALID_VARIANTS_SEQUENCE")
    if not variants:
        return {
            "status": "NOT_EVALUATED", "G1_PLANNED_FACTORIAL": "NOT_EVALUATED",
            "declared_variant_count": 0, "issues": ["NO_VARIANT_PLAN"],
            **UNVERIFIED,
        }
    seen_ids = set()
    cells = defaultdict(set)
    for variant in variants:
        _schema(variant, VARIANT_KEYS, {"split", "reverse_polarity"}, "variant")
        for field in VARIANT_KEYS - {"update_arm", "readout_time", "query_target"}:
            _id(variant[field], field)
        fid = variant["case_family_id"]
        if fid not in by_family:
            raise ValueError("UNKNOWN_ROOT_FAMILY")
        root = by_family[fid]
        if variant["split_group_id"] != root["split_group_id"]:
            raise ValueError("VARIANT_GROUP_INHERITANCE_CONFLICT")
        if "split" in variant:
            _enum(variant["split"], SPLITS, "variant_split")
            if variant["split"] != root.get("split"):
                raise ValueError("VARIANT_SPLIT_INHERITANCE_CONFLICT")
        if "reverse_polarity" in variant:
            _strict_bool(variant["reverse_polarity"], "reverse_polarity")
        vid = variant["variant_id"]
        if vid in seen_ids:
            raise ValueError("DUPLICATE_VARIANT_ID")
        seen_ids.add(vid)
        cell = (
            _enum(variant["update_arm"], ARMS, "update_arm"),
            _enum(variant["readout_time"], READOUTS, "readout_time"),
            _enum(variant["query_target"], QUERIES, "query_target"),
        )
        key = (fid, variant["presentation_order_id"], variant["paraphrase_family_id"])
        if cell in cells[key]:
            raise ValueError("DUPLICATE_FACTORIAL_CELL")
        cells[key].add(cell)
    issues = []
    for fid in by_family:
        if not any(key[0] == fid for key in cells):
            issues.append("MISSING_FAMILY_VARIANTS:" + fid)
    for key, found in sorted(cells.items()):
        if found != ALL_CELLS:
            issues.append("INCOMPLETE_27_CELL_REALIZATION:" + "/".join(key))
    if root_audit["status"] == "FAIL":
        issues.append("ROOT_REGISTRY_PLAN_INCOMPLETE")
    status = "FAIL" if issues else "PASS_PLAN_CHECK"
    return {
        "status": status,
        "G1_PLANNED_FACTORIAL": status,
        "declared_root_family_count": len(by_family),
        "declared_variant_count": len(variants),
        "declared_realization_count": len(cells),
        "counted_independent_units": len(by_family),
        "missing_cell_count": sum(27 - len(found) for found in cells.values()),
        "issues": sorted(issues),
        **UNVERIFIED,
    }


def _expected_boolean(family, arm, readout, query):
    # Frozen D1 truth semantics, expressed without importing the D1 code.
    a = family["root_initial_A"]
    c = family["root_initial_C"]
    u = family["root_initial_U"]
    if arm == "VALID_FLIP" and readout != "PRE":
        a = not a
    return {"A": a, "B": a and c, "U": u}[query]


def audit_counterbalance_plan(plan, families) -> dict:
    """Audit *declared* cue-vs-answer cross-tab, never measured text leakage.

    For each nonsemantic cue value and query A/B/U across planned inventory,
    both attainable Boolean target answers must occur. Authentication *truth*
    itself is legitimately task-relevant; a cue family's surface form must
    not deterministically encode an answer. No forced B=50/50 balance.
    """
    root_report = audit_family_registry(families)
    by_family = _family_index(families)[0]
    _schema(plan, {"plan_version"}, {"inventory"}, "counterbalance_plan")
    _enum(plan["plan_version"], {"RFR_D2_COUNTERBALANCE_PLAN_V1"}, "plan_version")
    if "inventory" not in plan or plan["inventory"] is None:
        return {
            "status": "NOT_EVALUATED", "G3_PLANNED_SURFACE": "NOT_EVALUATED",
            "G4_SHORTCUT_ATTACKS": {s: "NOT_EVALUATED" for s in SHORTCUT_FAMILIES},
            "issues": ["MISSING_CONTINGENCY_INVENTORY"],
            **UNVERIFIED,
        }
    inventory = _sequence(plan["inventory"], "inventory")
    cue_labels = defaultdict(lambda: Counter())
    expected_answers = Counter()
    total_cells = 0
    coverage = defaultdict(set)
    joint_labels = defaultdict(Counter)
    for row in inventory:
        _schema(row, INVENTORY_KEYS, frozenset(), "inventory")
        for field in INVENTORY_KEYS - {"count"}:
            _id(row[field], field)
        fid = row["case_family_id"]
        if fid not in by_family:
            raise ValueError("UNKNOWN_INVENTORY_FAMILY")
        root = by_family[fid]
        for field in ("split_group_id", "surface_domain", "renderer_family_id",
                      "template_family_id"):
            if row[field] != root[field]:
                raise ValueError("INVENTORY_FAMILY_METADATA_CONFLICT:" + field)
        declared_split = root.get("split") or "UNASSIGNED"
        if row["split"] != declared_split:
            raise ValueError("INVENTORY_SPLIT_CONFLICT")
        if row["root_truth_stratum"] != _stratum(root):
            raise ValueError("INVENTORY_TRUTH_STRATUM_CONFLICT")
        arm = _enum(row["update_arm"], ARMS, "update_arm")
        readout = _enum(row["readout_time"], READOUTS, "readout_time")
        query = _enum(row["query_target"], QUERIES, "query_target")
        count = row["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("INVALID_POSITIVE_INVENTORY_COUNT")
        label = "SUPPORT" if _expected_boolean(root, arm, readout, query) else "REFUTE"
        expected_answers[(query, label)] += count
        coverage[fid].add((arm, readout, query))
        joint_key = ("JOINT_CUE", row["surface_domain"],
                     row["renderer_family_id"], row["template_family_id"],
                     row["authentication_marker_family_id"],
                     row["source_name_family_id"],
                     row["polarity_wording_family_id"],
                     row["timestamp_presentation_pattern_id"], query)
        joint_labels[joint_key][label] += count
        for field in CUE_FIELDS:
            cue_labels[(field, row[field], query)][label] += count
        total_cells += count
    issues = []
    for fid in by_family:
        if coverage[fid] != ALL_CELLS:
            issues.append("INCOMPLETE_INVENTORY_27_CELLS:" + fid)
    for key, histogram in sorted(cue_labels.items()):
        if set(histogram) != {"REFUTE", "SUPPORT"}:
            issues.append("DETERMINISTIC_CUE_FOR_QUERY:" + "/".join(key))
    for key, histogram in sorted(joint_labels.items()):
        if set(histogram) != {"REFUTE", "SUPPORT"}:
            issues.append("DETERMINISTIC_JOINT_CUE_FOR_QUERY:" + "/".join(key))
    if root_report["status"] == "FAIL":
        issues.append("ROOT_REGISTRY_PLAN_INCOMPLETE")
    status = "FAIL" if issues else "PASS_PLAN_CHECK"
    return {
        "status": status,
        "G3_PLANNED_SURFACE": status,
        "G4_SHORTCUT_ATTACKS": {s: "NOT_EVALUATED" for s in SHORTCUT_FAMILIES},
        "declared_inventory_count": len(inventory),
        "declared_inventory_weight": total_cells,
        "design_derived_target_answer_counts": {
            query: {label: expected_answers[(query, label)] for label in ("REFUTE", "SUPPORT")}
            for query in sorted(QUERIES)
        },
        "structural_B_label_imbalance": "75_PERCENT_REFUTE_FOR_FULL_PRIMARY_FACTORIAL",
        "primary_NOT_ENTITLED": "ABSENT_BY_DESIGN",
        "unavoidable_family_identity_cues": "NOT_EVALUATED",
        "issues": sorted(issues),
        **UNVERIFIED,
    }
