"""RFR-D3-M4: fail-closed, metadata-only independent-root feasibility audit.

No roots, prompts, splits, random seeds, reports or files are created here.
The current M2/M3 pilot must not be promoted to an independently sampled
cohort. A prospective sampling design is PLAN_ONLY even when structurally valid.
"""

from collections import Counter
from collections.abc import Mapping
from itertools import product

DOMAINS = frozenset(("REGISTRY_RECORD", "INSTRUMENT_LOG"))
SPLITS = frozenset(("train", "dev", "test"))
WORLD_FIELDS = ("root_initial_A", "root_initial_C", "root_initial_U")
DRAW_COMPONENTS = frozenset((
    "ENTITY_CONFIGURATION", "BASE_RECORD_IDENTITIES", "SOURCE_ATTESTATION_IDENTITIES",
    "GRAMMAR_AND_TEMPLATE_CHOICES",
))
PLAN_KEYS = frozenset((
    "purpose", "root_unit", "derivation_rule_id", "oracle_version", "surface_domains",
    "planned_splits", "split_assignments", "split_group_inheritance", "holdout_contract",
    "root_draw_protocol", "draw_components", "root_manifest_policy",
    "logical_generalization_claim", "realized_independence_evidence",
))


def _mapping(value, required, name):
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise ValueError("M4_SCHEMA:" + name)


def _identifier(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError("M4_IDENTIFIER:" + name)
    return value


def _exact_set(value, required, name):
    if type(value) not in (list, tuple) or len(value) != len(required) or set(value) != required:
        raise ValueError("M4_SET:" + name)
    if any(type(x) is not str for x in value):
        raise ValueError("M4_STRING_SET:" + name)


def audit_existing_m2_frame(roots, lineage, m3_report):
    """Review current frozen M2 root roster and validated M3 overlap evidence.

    Requires the caller first to execute M2/M3 validators on the *same* roots
    and rows. This function checks metadata consistency, not text by itself.
    """
    if type(roots) is not list or type(lineage) is not list:
        raise ValueError("M4_ROOT_INPUT_TYPE")
    if len(roots) != 16 or len(lineage) != 16:
        raise ValueError("M4_CURRENT_M2_ROSTER_COUNT")
    if not isinstance(m3_report, Mapping):
        raise ValueError("M4_M3_REPORT_TYPE")
    if m3_report.get("status") != "M3_BOUNDED_AUDIT_COMPLETE_WITH_COHORT_BLOCKERS":
        raise ValueError("M4_M3_NOT_VALIDATED")
    if m3_report.get("provisional_root_count") != 16 or m3_report.get("distinct_boolean_worlds") != 8:
        raise ValueError("M4_M3_ROSTER_DISAGREEMENT")
    if m3_report.get("independent_cohort_feasibility") != "NOT_ESTABLISHED":
        raise ValueError("M4_M3_CLAIM_PROMOTION")
    overlap = m3_report.get("structural_overlap")
    neighbors = m3_report.get("controlled_near_neighbors")
    if not isinstance(overlap, Mapping) or not isinstance(neighbors, Mapping):
        raise ValueError("M4_MISSING_OVERLAP_EVIDENCE")
    if (overlap.get("cross_family_equal_role_structure_row_pairs") != 4608 or
        overlap.get("semantic_independence") != "NOT_ESTABLISHED" or
        neighbors.get("distinct_provisional_root_pairs_differing_only_U") != 8 or
        neighbors.get("matched_row_pairs_exactly_one_U_literal_apart") != 1728):
        raise ValueError("M4_M3_OVERLAP_DISAGREEMENT")

    by_id, domain_world = {}, Counter()
    for root in roots:
        if not isinstance(root, Mapping):
            raise ValueError("M4_ROOT_SCHEMA")
        fid = _identifier(root.get("case_family_id"), "case_family_id")
        _identifier(root.get("split_group_id"), "split_group_id")
        if fid in by_id:
            raise ValueError("M4_DUPLICATE_ROOT_ID")
        if root.get("surface_domain") not in DOMAINS:
            raise ValueError("M4_DOMAIN")
        if root.get("holdout_contract") != "NONE" or "split" in root:
            raise ValueError("M4_SPLITS_NOT_AUTHORIZED")
        world = []
        for key in WORLD_FIELDS:
            if type(root.get(key)) is not bool:
                raise ValueError("M4_BOOLEAN:" + key)
            world.append(root[key])
        if (root.get("derivation_rule_id") != "B_AND_A_C" or
            root.get("oracle_version") != "RFR_D1_ORACLE_V1"):
            raise ValueError("M4_FROZEN_SEMANTICS")
        domain_world[(root["surface_domain"], tuple(world))] += 1
        by_id[fid] = root
    expected_world = set(product((False, True), repeat=3))
    if (len(domain_world) != 16 or any(
        domain_world[(d, world)] != 1 for d in DOMAINS for world in expected_world
    )):
        raise ValueError("M4_NOT_TWO_DETERMINISTIC_COPIES_OF_EIGHT_WORLDS")

    seen_lineages, seen_groups, seen_world_ids, seen_substreams = set(), set(), set(), set()
    for entry in lineage:
        if not isinstance(entry, Mapping):
            raise ValueError("M4_LINEAGE_SCHEMA")
        fid = _identifier(entry.get("case_family_id"), "lineage.case_family_id")
        if fid not in by_id or fid in seen_lineages:
            raise ValueError("M4_LINEAGE_ROOT_ID")
        seen_lineages.add(fid)
        group = _identifier(entry.get("split_group_id"), "lineage.split_group_id")
        if group != by_id[fid]["split_group_id"] or group in seen_groups:
            raise ValueError("M4_LINEAGE_GROUP")
        seen_groups.add(group)
        world_id = _identifier(entry.get("semantic_world_id"), "semantic_world_id")
        substream = _identifier(entry.get("root_substream_id"), "root_substream_id")
        if world_id in seen_world_ids or substream in seen_substreams:
            raise ValueError("M4_LINEAGE_IDENTITY_REUSE")
        seen_world_ids.add(world_id)
        seen_substreams.add(substream)
        if entry.get("root_kind") != "PRIMARY_ROOT":
            raise ValueError("M4_DERIVED_ROOT_FALSELY_INDEPENDENT")
        if not substream.startswith("m1_enumeration_only_"):
            raise ValueError("M4_M2_PROVENANCE_MISMATCH")
    if seen_lineages != set(by_id):
        raise ValueError("M4_MISSING_LINEAGE")

    return {
        "status": "M4_CURRENT_FRAME_INDEPENDENCE_BLOCKED",
        "provisional_roots": 16,
        "independent_logical_laws": 1,
        "distinct_boolean_worlds": 8,
        "domain_world_replications": 8,
        "independently_drawn_root_families_verified": 0,
        "root_generation": "DETERMINISTIC_ENUMERATION_NOT_SAMPLING",
        "cross_family_equal_role_structure_pairs": 4608,
        "source_m3_status": m3_report["status"],
        "split_assignments": "NONE",
        "within_renderer_three_split_full_stratum_grid_minimum_roots": 48,
        "grid_count_shortfall_if_one_root_per_cell": 32,
        "three_way_renderer_disjointness_with_two_renderers": "IMPOSSIBLE",
        "independent_surface_cohort": "NOT_ESTABLISHED",
        "broader_logical_generalization": "BLOCKED_BY_FROZEN_PRIMARY_RULE",
        "materialized_cohort_feasibility": "NOT_EVALUATED",
        "tokenizer_serialization": "NOT_EVALUATED",
        "model_evaluation": "NOT_EVALUATED",
        "PHASE3B": "BLOCKED_UNCHANGED",
    }


def audit_prospective_sampling_contract(plan):
    """Check *proposed* root draw and holdout policy, without executing it.

    A PASS_PLAN_ONLY proves no random independence and authorizes no sampling.
    In particular, three-way renderer holdout is impossible with two renderers.
    """
    _mapping(plan, PLAN_KEYS, "sampling_plan")
    fixed = {
        "purpose": "SURFACE_GENERALIZATION_ONLY",
        "root_unit": "INDEPENDENT_SYNTHETIC_MICROWORLD_DRAW",
        "derivation_rule_id": "B_AND_A_C",
        "oracle_version": "RFR_D1_ORACLE_V1",
        "split_assignments": "NONE",
        "split_group_inheritance": "ALL_DESCENDANTS_SAME_ROOT_GROUP",
        "holdout_contract": "NONE",
        "root_draw_protocol": "PROSPECTIVE_SEPARATE_ROOT_SUBSTREAMS",
        "root_manifest_policy": "DECLARATION_ONLY_PIN_BEFORE_REALIZATION",
        "realized_independence_evidence": "NONE",
    }
    for key, expectation in fixed.items():
        if plan[key] != expectation:
            raise ValueError("M4_UNAUTHORIZED_PLAN_CLAIM:" + key)
    _exact_set(plan["surface_domains"], DOMAINS, "surface_domains")
    _exact_set(plan["planned_splits"], SPLITS, "planned_splits")
    _exact_set(plan["draw_components"], DRAW_COMPONENTS, "draw_components")
    if type(plan["logical_generalization_claim"]) is not bool or plan["logical_generalization_claim"]:
        raise ValueError("M4_LOGICAL_GENERALIZATION_NOT_JUSTIFIED")
    return {
        "status": "M4_PROSPECTIVE_SURFACE_FRAME_PLAN_ONLY",
        "candidate_root_draw_components": len(DRAW_COMPONENTS),
        "logical_law_count": 1,
        "logical_world_count": 8,
        "minimum_root_slots_for_full_domain_stratum_split_cross": 48,
        "minimum_root_slots_are_statistical_power_analysis": False,
        "split_assignments": "NONE",
        "renderer_holdout_across_three_splits": "NOT_SUPPORTED",
        "independently_drawn_roots": "NOT_EVALUATED",
        "independent_surface_cohort": "NOT_ESTABLISHED",
        "semantic_generalization": "BLOCKED_UNDER_FROZEN_PRIMARY_RULE",
        "materialized_cohort_feasibility": "NOT_EVALUATED",
        "tokenizer_serialization": "NOT_EVALUATED",
        "model_evaluation": "NOT_EVALUATED",
        "PHASE3B": "BLOCKED_UNCHANGED",
    }
