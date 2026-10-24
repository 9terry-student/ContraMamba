"""RFR-D3-M6: prospective, metadata-only root provenance/graph contract audits.

No independent draws, manifest files, random seeds, corpus rows, splits,
model inference, tokenizer, or IO. A valid hypothetical plan is never evidence
that an independent population was realized. Frozen M4 owns the frame check.
"""

from collections.abc import Mapping
from itertools import permutations

from scripts import rfr_d3_m4_root_frame_feasibility_cpu as m4

COMPONENTS = frozenset((
    "ENTITY_CONFIGURATION", "BASE_RECORD_IDENTITIES",
    "SOURCE_ATTESTATION_IDENTITIES", "GRAMMAR_AND_TEMPLATE_CHOICES",
))
ROOT_KEYS = frozenset((
    "case_family_id", "split_group_id", "semantic_world_id", "surface_domain",
    "renderer_family_id", "lexicon_namespace_id", "template_family_id",
    "root_initial_A", "root_initial_C", "root_initial_U",
    "derivation_rule_id", "oracle_version", "root_substream_id", "root_kind",
    "ancestry_parent_id", "split_assignment", "component_draw_refs",
    "instance_graph", "declaration_status", "plan_identity",
    "generator_version", "renderer_version", "grammar_version",
    "canonicalization_version", "root_draw_protocol_id",
))
MANIFEST_KEYS = frozenset((
    "manifest_schema_version", "plan_identity", "generator_version",
    "renderer_version", "grammar_version", "canonicalization_version",
    "root_draw_protocol_id", "seed_derivation_spec_state", "attempt_ledger_policy",
    "component_ref_policy", "root_lineage_policy", "split_policy",
    "realization_state", "claim_scope",
))
ATTEMPT_KEYS = frozenset((
    "attempt_id", "case_family_id", "root_substream_id", "ordinal",
    "accepted", "reject_reason",
))
REF_KEYS = frozenset(("source_stream_id", "counter_index", "draw_record_ref",
                       "parent_root_substream_id"))
NODE_KEYS = frozenset(("node_id", "role"))
EDGE_KEYS = frozenset(("from", "to", "relation"))
GRAPH_KEYS = frozenset(("nodes", "edges"))
PRIMARY_ROLES = frozenset((
    "ENTITY_MAIN", "RECORD_A", "RECORD_C", "RECORD_U",
    "ATTEST_A", "ATTEST_C", "ATTEST_U",
))
OPTIONAL_ROLES = frozenset(("ATTEST_HOP", "AUX_NEUTRAL"))
STATUS = "M6_PASS_PLAN_ONLY_PROVENANCE_NOT_VERIFIED"


def _schema(value, keys, label):
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("M6_SCHEMA:" + label)


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError("M6_TEXT:" + label)
    return value


def _items(value, label, allow_empty=False):
    if type(value) not in (list, tuple) or (not allow_empty and not value):
        raise ValueError("M6_SEQUENCE:" + label)
    return value


def audit_manifest_contract(plan, manifest):
    """Check a prospective manifest specification; do not pin or choose a seed."""
    frame = m4.audit_prospective_sampling_contract(plan)
    if frame["status"] != "M4_PROSPECTIVE_SURFACE_FRAME_PLAN_ONLY":
        raise ValueError("M6_M4_FRAME_NOT_PLAN_ONLY")
    _schema(manifest, MANIFEST_KEYS, "manifest")
    exact = {
        "manifest_schema_version": "RFR_D3_M6_ROOT_PLAN_V1",
        "canonicalization_version": "M6_TYPED_GRAPH_V1",
        "seed_derivation_spec_state": "NOT_FROZEN_NO_SEED_SELECTED",
        "attempt_ledger_policy": "ALL_ATTEMPTS_WITH_REJECTION_REASONS",
        "component_ref_policy": "FOUR_DISTINCT_COMPONENT_DRAWS_PER_ROOT",
        "root_lineage_policy": "PRIMARY_ROOT_NO_DERIVED_RECOUNT",
        "split_policy": "UNASSIGNED_INHERIT_ON_LATER_AUTHORITY",
        "realization_state": "DECLARATIONS_ONLY_NO_DRAWS",
        "claim_scope": "SURFACE_GENERALIZATION_PLAN_ONLY",
    }
    for key, val in exact.items():
        if manifest[key] != val:
            raise ValueError("M6_UNAUTHORIZED_MANIFEST_CLAIM:" + key)
    for key in ("plan_identity", "generator_version", "renderer_version",
                "grammar_version", "root_draw_protocol_id"):
        _text(manifest[key], key)
    return {"status": STATUS, "M4_FRAME": frame["status"],
            "MANIFEST_EXECUTED": "NO", "DRAW_SEED_SELECTED": "NO",
            "ACTUAL_ROOT_PROVENANCE": "NOT_EVALUATED",
            "INDEPENDENT_SURFACE_COHORT": "NOT_ESTABLISHED",
            "MODEL_EVALUATION": "NOT_EVALUATED", "PHASE3B": "BLOCKED_UNCHANGED"}


def _graph_signature(graph):
    """Canonical task-bearing typed graph, invariant to node IDs/reorderings.

    Neutral decoration is deliberately excluded: changing only AUX_NEUTRAL
    nodes cannot turn a copied root into a distinct instance structure.
    Anonymous attestation hops are exhaustively permuted (at most three).
    Equal signatures are evidence of structural reuse, not scientific proof.
    """
    _schema(graph, GRAPH_KEYS, "graph")
    nodes = _items(graph["nodes"], "nodes")
    edges = _items(graph["edges"], "edges")
    ids = {}
    mandatory = {}
    hops = []
    for node in nodes:
        _schema(node, NODE_KEYS, "node")
        nid = _text(node["node_id"], "node_id")
        role = _text(node["role"], "role")
        if nid in ids or role not in PRIMARY_ROLES | OPTIONAL_ROLES:
            raise ValueError("M6_NODE_DUPLICATE_OR_ROLE")
        ids[nid] = role
        if role in PRIMARY_ROLES:
            if role in mandatory:
                raise ValueError("M6_DUPLICATE_PRIMARY_ROLE")
            mandatory[role] = nid
        elif role == "ATTEST_HOP":
            hops.append(nid)
    if set(mandatory) != PRIMARY_ROLES or len(hops) > 3:
        raise ValueError("M6_GRAPH_ROLE_COVERAGE")
    seen_edges = set()
    for edge in edges:
        _schema(edge, EDGE_KEYS, "edge")
        src, dst, rel = edge["from"], edge["to"], edge["relation"]
        if src not in ids or dst not in ids or src == dst:
            raise ValueError("M6_GRAPH_ENDPOINT")
        _text(rel, "relation")
        item = (src, dst, rel)
        if item in seen_edges:
            raise ValueError("M6_DUPLICATE_EDGE")
        seen_edges.add(item)
        if "AUX_NEUTRAL" in (ids[src], ids[dst]):
            if rel != "NEUTRAL_CONTEXT":
                raise ValueError("M6_AUXILIARY_ORACLE_CONTAMINATION")
        elif rel == "HAS_RECORD":
            if ids[src] != "ENTITY_MAIN" or ids[dst] not in {"RECORD_A", "RECORD_C", "RECORD_U"}:
                raise ValueError("M6_INVALID_HAS_RECORD")
        elif rel == "HAS_ATTESTATION":
            if (ids[src], ids[dst]) not in {(f"RECORD_{x}", f"ATTEST_{x}") for x in "ACU"}:
                raise ValueError("M6_INVALID_ATTESTATION_LINK")
        elif rel == "CHAIN_LINK":
            if ids[src] not in {"ATTEST_A", "ATTEST_C", "ATTEST_U", "ATTEST_HOP"} or ids[dst] != "ATTEST_HOP":
                raise ValueError("M6_INVALID_CHAIN")
        else:
            raise ValueError("M6_UNKNOWN_GRAPH_RELATION")
    for target in "ACU":
        if (mandatory["ENTITY_MAIN"], mandatory[f"RECORD_{target}"], "HAS_RECORD") not in seen_edges:
            raise ValueError("M6_MISSING_ENTITY_RECORD_EDGE")
        if (mandatory[f"RECORD_{target}"], mandatory[f"ATTEST_{target}"], "HAS_ATTESTATION") not in seen_edges:
            raise ValueError("M6_MISSING_RECORD_ATTESTATION_EDGE")
    active_hops = {n for n in hops if any(n in (src,dst) for src,dst,rel in seen_edges if rel == "CHAIN_LINK")}
    if active_hops != set(hops):
        raise ValueError("M6_DANGLING_ATTESTATION_HOP")
    # A prospective chain may branch, but may not contain a cycle.
    chain = [(src,dst) for src,dst,rel in seen_edges if rel == "CHAIN_LINK"]
    adjacency = {}
    for src,dst in chain:
        adjacency.setdefault(src, []).append(dst)
    def visit(n, path):
        if n in path:
            raise ValueError("M6_ATTESTATION_CYCLE")
        for successor in adjacency.get(n, ()):
            visit(successor, path | {n})
    for n in adjacency:
        visit(n, set())
    primary_order = sorted(PRIMARY_ROLES)
    results = []
    for perm in permutations(hops):
        coding = {mandatory[role]: role for role in primary_order}
        coding.update({nid: f"ATTEST_HOP_{i}" for i,nid in enumerate(perm)})
        encoded = tuple(sorted((coding[src], rel, coding[dst]) for src,dst,rel in seen_edges
                               if src in coding and dst in coding))
        results.append(encoded)
    return ("M6_TYPED_GRAPH_V1", tuple(primary_order), len(hops), min(results))


def audit_declared_root_roster(plan, manifest, roots, attempts):
    """Audit *hypothetical declarations*, never certify observed random draws.

    All roots/attempts must carry an explicit non-realized declaration marker.
    Even all-distinct graphs/substreams merely make a prospective plan plausible.
    """
    audit_manifest_contract(plan, manifest)
    roots = _items(roots, "roots")
    attempts = _items(attempts, "attempts")
    by_id, graph_seen, groups, world_ids, substreams = {}, {}, set(), set(), set()
    ref_seen, streams_seen, global_streams = set(), set(), set()
    core_worlds = set()
    for root in roots:
        _schema(root, ROOT_KEYS, "root")
        for field in ("plan_identity", "generator_version", "renderer_version",
                      "grammar_version", "canonicalization_version", "root_draw_protocol_id"):
            if root[field] != manifest[field]:
                raise ValueError("M6_ROOT_MANIFEST_VERSION_MISMATCH:" + field)
        if root["declaration_status"] != "HYPOTHETICAL_NOT_DRAWN":
            raise ValueError("M6_FALSE_REALIZATION_CLAIM")
        if root["root_kind"] != "PRIMARY_ROOT" or root["ancestry_parent_id"] is not None:
            raise ValueError("M6_DERIVED_ROOT_RECOUNT")
        if root["split_assignment"] is not None:
            raise ValueError("M6_SPLIT_ASSIGNMENT_NOT_AUTHORIZED")
        for field in ("case_family_id", "split_group_id", "semantic_world_id",
                      "renderer_family_id", "lexicon_namespace_id", "template_family_id", "root_substream_id"):
            _text(root[field], field)
        fid = root["case_family_id"]
        if fid in by_id or root["split_group_id"] in groups or root["semantic_world_id"] in world_ids or root["root_substream_id"] in substreams:
            raise ValueError("M6_ROOT_OR_LINEAGE_ID_REUSE")
        if root["surface_domain"] not in m4.DOMAINS:
            raise ValueError("M6_SURFACE_DOMAIN")
        if root["derivation_rule_id"] != "B_AND_A_C" or root["oracle_version"] != "RFR_D1_ORACLE_V1":
            raise ValueError("M6_FROZEN_SEMANTICS")
        world = []
        for field in m4.WORLD_FIELDS:
            if type(root[field]) is not bool:
                raise ValueError("M6_STRICT_BOOLEAN:" + field)
            world.append(root[field])
        core_worlds.add(tuple(world))
        refs = root["component_draw_refs"]
        _schema(refs, COMPONENTS, "draw_components")
        root_streams = set()
        for kind in sorted(COMPONENTS):
            record = refs[kind]
            _schema(record, REF_KEYS, "component_ref")
            if record["parent_root_substream_id"] != root["root_substream_id"]:
                raise ValueError("M6_COMPONENT_ROOT_SUBSTREAM_CONFLICT")
            stream = _text(record["source_stream_id"], "source_stream_id")
            token = _text(record["draw_record_ref"], "draw_record_ref")
            counter = record["counter_index"]
            if type(counter) is not int or counter < 0:
                raise ValueError("M6_BAD_DRAW_COUNTER")
            if stream in root_streams or stream in global_streams or (stream, counter) in ref_seen or token in streams_seen:
                raise ValueError("M6_COPIED_COMPONENT_DRAW")
            root_streams.add(stream)
            global_streams.add(stream)
            ref_seen.add((stream, counter))
            streams_seen.add(token)
        signature = _graph_signature(root["instance_graph"])
        if signature in graph_seen:
            raise ValueError("M6_BLOCKED_ISOMORPHIC_INSTANCE_GRAPH")
        graph_seen[signature] = fid
        by_id[fid] = root
        groups.add(root["split_group_id"])
        world_ids.add(root["semantic_world_id"])
        substreams.add(root["root_substream_id"])
    attempt_ids, logs = set(), {}
    for item in attempts:
        _schema(item, ATTEMPT_KEYS, "attempt")
        aid = _text(item["attempt_id"], "attempt_id")
        fid = _text(item["case_family_id"], "attempt.case_family_id")
        if aid in attempt_ids or fid not in by_id:
            raise ValueError("M6_ATTEMPT_ID_OR_ROOT")
        attempt_ids.add(aid)
        if item["root_substream_id"] != by_id[fid]["root_substream_id"]:
            raise ValueError("M6_ATTEMPT_SUBSTREAM_CONFLICT")
        if type(item["ordinal"]) is not int or item["ordinal"] < 0 or type(item["accepted"]) is not bool:
            raise ValueError("M6_ATTEMPT_TYPES")
        reason = item["reject_reason"]
        if item["accepted"] and reason is not None:
            raise ValueError("M6_ACCEPTED_WITH_REJECTION")
        if not item["accepted"]:
            _text(reason, "reject_reason")
        logs.setdefault(fid, []).append(item)
    for fid in by_id:
        events = sorted(logs.get(fid, []), key=lambda a:a["ordinal"])
        if not events or [e["ordinal"] for e in events] != list(range(len(events))):
            raise ValueError("M6_MISSING_OR_GAPPED_ATTEMPT_LEDGER")
        if sum(e["accepted"] for e in events) != 1 or not events[-1]["accepted"]:
            raise ValueError("M6_ATTEMPT_ACCEPTANCE_SEQUENCE")
    return {"status": STATUS, "declared_candidate_roots": len(by_id),
            "declared_attempts": len(attempts), "distinct_declared_core_logic_worlds": len(core_worlds),
            "distinct_declared_instance_graphs": len(graph_seen),
            "verified_independent_root_draws": 0,
            "draw_provenance_observed": "NO", "selection_ledger_exhaustiveness": "NOT_VERIFIABLE_FROM_DECLARATIONS",
            "family_split_isolation": "NOT_EVALUATED_NO_SPLITS",
            "raw_text_duplicate_audit": "NOT_EVALUATED_NO_TEXT",
            "semantic_independence": "NOT_ESTABLISHED",
            "cohort_materialization": "NOT_EVALUATED", "model_evaluation": "NOT_EVALUATED",
            "PHASE3B": "BLOCKED_UNCHANGED"}
