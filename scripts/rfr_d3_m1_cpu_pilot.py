"""RFR-D3-M1: strictly in-memory, deterministic synthetic rendering PILOT.

No I/O, RNG, split assignment, model/tokenizer calls or scientific claims.
16 provisional roots = 2 surface domains x the same 8 Boolean logical worlds.
432 related rows = 16 roots x 3 arms x 3 readouts x 3 queries.
Every generated row is checked against the frozen D1 oracle and D3 protocol.
D2 structural feasibility is delegated, never reimplemented.

This module creates *pilot examples in RAM* only. Its returned report is a
bounded implementation diagnostic, never MATERIALIZED_COHORT_FEASIBILITY_PASS.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json

from scripts import rfr_d1_deterministic_oracle_cpu as d1
from scripts import rfr_d3_cohort_contract_cpu as d3

DOMAINS = ("REGISTRY_RECORD", "INSTRUMENT_LOG")
ARMS = ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")
READOUTS = ("PRE", "POST", "DELAYED")
QUERIES = ("A", "B", "U")

# Symbolic D3 grammar contract IDs, intentionally NOT sentence templates.
GRAMMAR_IDS = {
    "REGISTRY_RECORD": ("ledger", "register_renderer_m1"),
    "INSTRUMENT_LOG": ("apparatus", "instrument_renderer_m1"),
}
# Opaque-looking invented identifiers; they are NOT random independent draws.
NAMES = ("Cedar", "Nimbus", "Topaz", "Quill", "Mosaic", "Harbor", "Orchid", "Flint")


def grammar_definitions():
    """Declare disjoint form IDs for the frozen D3 static grammar audit."""
    output = []
    for domain, (prefix, renderer) in GRAMMAR_IDS.items():
        output.append({
            "surface_domain": domain,
            "renderer_family_id": renderer,
            "lexical_atoms": [f"{prefix}_{part}" for part in (
                "world", "fact", "source", "context")],
            "template_family_ids": [f"{prefix}_record_v1", f"{prefix}_query_v1"],
            "positive_form_id": f"{prefix}_explicit_true",
            "explicit_negative_form_id": f"{prefix}_explicit_false",
            "absence_form_id": f"{prefix}_no_record",
            "authenticated_marker_id": f"{prefix}_certified",
            "unauthenticated_marker_id": f"{prefix}_unverified",
            "rule_conjunction_form_ids": [f"{prefix}_and_A_C", f"{prefix}_both_A_C"],
            "presentation_order_ids": [f"{prefix}_chronological", f"{prefix}_reverse"],
            "query_position_ids": [f"{prefix}_query_first", f"{prefix}_query_last"],
            "neutral_distractor_family_ids": [f"{prefix}_neutral"],
        })
    return output


def _worlds():
    return tuple(product((False, True), repeat=3))


def root_specs():
    """Return declared PILOT roots; no sampling, no train/dev/test assignment."""
    roots, lineage = [], []
    for domain in DOMAINS:
        prefix, renderer = GRAMMAR_IDS[domain]
        for i, (a, c, u) in enumerate(_worlds()):
            key = f"m1_{prefix}_{i}"
            roots.append({
                "case_family_id": key,
                "split_group_id": f"m1_group_{key}",
                "surface_domain": domain,
                "renderer_family_id": renderer,
                "lexicon_namespace_id": f"m1_lexicon_{key}",
                "template_family_id": f"m1_template_{key}",
                "root_initial_A": a,
                "root_initial_C": c,
                "root_initial_U": u,
                "semantic_world_fingerprint": f"m1_world_{key}",
                "entity_tuple_fingerprint": f"m1_entity_{key}",
                "source_attestation_fingerprint": f"m1_attestation_{key}",
                "derivation_rule_id": "B_AND_A_C",
                "oracle_version": "RFR_D1_ORACLE_V1",
                "holdout_contract": "NONE",
            })
            lineage.append({
                "case_family_id": key,
                "split_group_id": f"m1_group_{key}",
                "semantic_world_id": f"m1_world_id_{key}",
                "root_substream_id": f"m1_enumeration_only_{key}",
                "root_kind": "PRIMARY_ROOT",
            })
    return roots, lineage


def _parity(root):
    return sum(root[f"root_initial_{k}"] for k in ("A", "C", "U")) % 2


def _event(root, tag, target, value, auth, time):
    fid = root["case_family_id"]
    source = f"{fid}_alpha" if auth else f"{fid}_beta"
    return {
        "event_id": f"{fid}_{tag}", "target": target,
        "asserted_value": value, "source_authenticated": auth,
        "event_time": time, "source_id": f"source_{source}",
        "provenance_id": f"proof_{source}",
        "oracle_version": "RFR_D1_ORACLE_V1",
        "derivation_rule_id": "B_AND_A_C",
    }


def event_declaration(root, arm, readout, query):
    """Construct a typed descendant; D3 validates before any text is emitted."""
    if arm not in ARMS or readout not in READOUTS or query not in QUERIES:
        raise ValueError("INVALID_PILOT_FACTORS")
    events = [_event(root, f"t0_{k}", k, root[f"root_initial_{k}"], True, 10)
              for k in ("A", "C", "U")]
    if readout != "PRE":
        value = root["root_initial_A"] if arm == "VALID_SHAM" else not root["root_initial_A"]
        events.append(_event(root, "t1_A", "A", value, arm != "INVALID_FLIP", 20))
    order = [event["event_id"] for event in events]
    if _parity(root):
        order.reverse()
    declaration = {
        "query_target": query,
        "events": events,
        "presentation_order": order,
        "visible_sections": ["rule", "records", "authentication", "logical_time", "query"]
            + (["neutral_delay"] if readout == "DELAYED" else []),
        "sealed_field_names": sorted(d3.SEALED_REQUIRED),
        "distractors": ([{"distractor_id": "neutral_m1", "kind": "NEUTRAL"}]
                        if readout == "DELAYED" else []),
    }
    d3.audit_event_declaration(root, arm, readout, declaration)
    return declaration


def _rule(domain, name):
    if domain == "REGISTRY_RECORD":
        return (f"Rule: In the synthetic register for {name}, B holds exactly if "
                "filing A is true AND requirement C is true. There are no other conditions.")
    return (f"Rule: In the synthetic instrument {name}, B holds exactly if "
            "channel A is true AND calibration C is true. There are no other conditions.")


def _record(domain, event):
    value = "explicitly TRUE" if event["asserted_value"] else "explicitly FALSE"
    # Authentication is explicit in the local document protocol, NOT encoded
    # as words such as 'trusted' or 'invalid' within the source identifier.
    # These pilot markers are not sufficient to prove cue counterbalance.
    if domain == "REGISTRY_RECORD":
        cert = "register seal verified" if event["source_authenticated"] else "register seal not verified"
    else:
        cert = "instrument signature passed" if event["source_authenticated"] else "instrument signature failed"
    if domain == "REGISTRY_RECORD":
        noun = {"A": "filing A", "C": "requirement C", "U": "unrelated entry U"}[event["target"]]
        stem = "Register record"
    else:
        noun = {"A": "channel A", "C": "calibration C", "U": "unrelated sensor U"}[event["target"]]
        stem = "Instrument log"
    return (f"{stem}: logical-time={event['event_time']}; {noun} is {value}; "
            f"source={event['source_id']}; provenance={event['provenance_id']}; "
            f"authentication={cert}.")


def _query(domain, query):
    words = ({"A": "filing A", "B": "joint register eligibility B", "U": "unrelated entry U"}
             if domain == "REGISTRY_RECORD" else
             {"A": "channel A", "B": "joint instrument readiness B", "U": "unrelated sensor U"})
    return f"Question: Is {words[query]} true based on the authenticated records and rule?"


def render_visible_input(root, declaration):
    """Deterministic grammatical realization. No sealed gold fields are read."""
    domain = root["surface_domain"]
    prefix = GRAMMAR_IDS[domain][0]
    i = int(root["case_family_id"].rsplit("_", 1)[1])
    name = f"{prefix.title()}-{NAMES[i]}"
    header = ("Synthetic register evidence:" if domain == "REGISTRY_RECORD"
              else "Synthetic instrument evidence:")
    records = {event["event_id"]: event for event in declaration["events"]}
    lines = [header, _rule(domain, name)]
    question = _query(domain, declaration["query_target"])
    if _parity(root):
        lines.append(question)
    lines.extend(_record(domain, records[key]) for key in declaration["presentation_order"])
    if declaration["distractors"]:
        lines.append("Neutral note: the margin carries a decorative dash with no A/C/U assertion.")
    if not _parity(root):
        lines.append(question)
    return "\n".join(lines)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(value):
    return sha256(value).hexdigest()


def _variant_plan(roots):
    return [{
        "case_family_id": root["case_family_id"],
        "split_group_id": root["split_group_id"],
        "variant_id": f"{root['case_family_id']}:{arm}:{readout}:{query}",
        "update_arm": arm, "readout_time": readout, "query_target": query,
        "presentation_order_id": "m1_reverse" if _parity(root) else "m1_chronological",
        "paraphrase_family_id": "m1_fixed_pilot",
    } for root in roots for arm, readout, query in product(ARMS, READOUTS, QUERIES)]


def generate_pilot():
    """Produce bounded pilot rows in memory only; not a saved RFR dataset."""
    roots, lineage = root_specs()
    grammar = grammar_definitions()
    d3.audit_grammar_definitions(grammar)
    bridge = d3.audit_d2_plan_bridge(roots, _variant_plan(roots), grammar, lineage)
    if bridge["status"] != "PASS_STATIC_CONTRACT":
        raise ValueError("FROZEN_D2_D3_PLAN_BLOCKED")
    rows = []
    for root in roots:
        for query in QUERIES:
            contrasts = {arm: event_declaration(root, arm, "POST", query) for arm in ARMS}
            d3.audit_matched_contrasts(root, contrasts)
        for arm, readout, query in product(ARMS, READOUTS, QUERIES):
            declaration = event_declaration(root, arm, readout, query)
            answer = d1.evaluate_records(
                declaration["events"], query, readout, declaration["distractors"])
            visible = render_visible_input(root, declaration)
            vid = f"{root['case_family_id']}:{arm}:{readout}:{query}"
            rows.append({
                "visible_input": visible,
                "sealed_metadata": {
                    "case_family_id": root["case_family_id"],
                    "split_group_id": root["split_group_id"],
                    "case_id": f"case_{root['case_family_id']}",
                    "variant_id": vid,
                    "surface_domain": root["surface_domain"],
                    "root_truth_stratum": "".join("T" if root[f"root_initial_{k}"] else "F"
                                                   for k in ("A", "C", "U")),
                    "update_arm": arm,
                    "readout_time": readout,
                    "query_target": query,
                    "target_status": answer["target_status"],
                    "final_label": answer["final_label"],
                    "final_label_id": answer["final_label_id"],
                    "events": declaration["events"],
                    "distractors": declaration["distractors"],
                    "presentation_order": declaration["presentation_order"],
                    "prompt_sha256": _sha(visible.encode("utf-8")),
                    "canonical_event_sha256": _sha(_canonical_json(declaration["events"])),
                },
            })
    return roots, rows


def audit_pilot(roots, rows):
    """Independently recheck in-memory pilot structure and observable content.

    This is a PILOT audit only; it does not verify independently sampled
    populations, full leakage freedom, real data manifests or serialization.
    """
    if type(roots) is not list or type(rows) is not list:
        raise ValueError("INVALID_PILOT_INPUTS")
    if len(roots) != 16 or len(rows) != 432:
        raise ValueError("INCORRECT_PILOT_COUNTS")
    roots_by_id = {r["case_family_id"]: r for r in roots}
    if len(roots_by_id) != 16:
        raise ValueError("DUPLICATE_ROOTS")
    if Counter(r["surface_domain"] for r in roots) != Counter({d: 8 for d in DOMAINS}):
        raise ValueError("DOMAIN_CAPACITY_MISMATCH")
    if any(Counter(tuple(r[f"root_initial_{k}"] for k in ("A", "C", "U"))
                   for r in roots if r["surface_domain"] == d) !=
           Counter({x: 1 for x in _worlds()}) for d in DOMAINS):
        raise ValueError("TRUTH_STRATA_MISMATCH")
    seen_variants = set()
    buckets = defaultdict(list)
    labels = Counter()
    global_prompt_hashes = defaultdict(set)
    for row in rows:
        if set(row) != {"visible_input", "sealed_metadata"}:
            raise ValueError("ROW_SCHEMA_MISMATCH")
        visible, meta = row["visible_input"], row["sealed_metadata"]
        if type(visible) is not str or not visible:
            raise ValueError("INVALID_VISIBLE_INPUT")
        fid = meta["case_family_id"]
        if fid not in roots_by_id:
            raise ValueError("UNKNOWN_FAMILY")
        root = roots_by_id[fid]
        arm, readout, query = (meta[k] for k in ("update_arm", "readout_time", "query_target"))
        vid = f"{fid}:{arm}:{readout}:{query}"
        if meta["variant_id"] != vid or vid in seen_variants:
            raise ValueError("VARIANT_IDENTITY_CONFLICT")
        seen_variants.add(vid)
        if meta["split_group_id"] != root["split_group_id"]:
            raise ValueError("SPLIT_GROUP_CONFLICT")
        if "split" in meta:
            raise ValueError("PILOT_SPLIT_NOT_AUTHORIZED")
        if meta["surface_domain"] != root["surface_domain"]:
            raise ValueError("DOMAIN_CONFLICT")
        if meta["prompt_sha256"] != _sha(visible.encode("utf-8")) or (
            meta["canonical_event_sha256"] != _sha(_canonical_json(meta["events"]))
        ):
            raise ValueError("PILOT_HASH_MISMATCH")
        if any(token in visible for token in (
            "case_family_id", "split_group_id", "target_status", "final_label",
            "final_label_id", "VALID_FLIP", "INVALID_FLIP", "VALID_SHAM",
            "NOT_ENTITLED", "REFUTE", "SUPPORT", "root_initial_",
        )):
            raise ValueError("SEALED_METADATA_EXPOSED")
        if "Rule:" not in visible or "Question:" not in visible:
            raise ValueError("MISSING_VISIBLE_RULE_OR_QUERY")
        if readout == "PRE" and "logical-time=20" in visible:
            raise ValueError("PRE_FUTURE_LEAKAGE")
        if readout != "PRE" and "logical-time=20" not in visible:
            raise ValueError("T1_NOT_VISIBLE")
        if ("Neutral note:" in visible) != (readout == "DELAYED"):
            raise ValueError("DELAYED_SURFACE_CONFLICT")
        declaration = {
            "query_target": query,
            "events": meta["events"],
            "presentation_order": meta["presentation_order"],
            "visible_sections": ["rule", "records", "authentication", "logical_time", "query"]
                + (["neutral_delay"] if readout == "DELAYED" else []),
            "sealed_field_names": sorted(d3.SEALED_REQUIRED),
            "distractors": meta["distractors"],
        }
        d3.audit_event_declaration(root, arm, readout, declaration)
        if render_visible_input(root, declaration) != visible:
            raise ValueError("RENDERED_EVENT_OR_QUERY_MISMATCH")
        oracle = d1.evaluate_records(meta["events"], query, readout, meta["distractors"])
        if (meta["target_status"], meta["final_label"], meta["final_label_id"]) != (
            oracle["target_status"], oracle["final_label"], oracle["final_label_id"]
        ):
            raise ValueError("FROZEN_ORACLE_MISMATCH")
        buckets[(fid, arm, readout, query)].append(row)
        labels[(query, meta["final_label"])] += 1
        global_prompt_hashes[meta["prompt_sha256"]].add(fid)
    if any(len(bucket) != 1 for bucket in buckets.values()):
        raise ValueError("INCOMPLETE_OR_DUPLICATE_FACTORIAL")
    if any(len(fids) != 1 for fids in global_prompt_hashes.values()):
        raise ValueError("CROSS_FAMILY_EXACT_DUPLICATE")
    # Identical PRE inputs across arms in the same family/query are expected.
    for root in roots:
        fid = root["case_family_id"]
        for query in QUERIES:
            hashes = {buckets[(fid, arm, "PRE", query)][0]["sealed_metadata"]["prompt_sha256"]
                      for arm in ARMS}
            if len(hashes) != 1:
                raise ValueError("PRE_ARM_LEAKAGE")
    expected_labels = {("A", "REFUTE"): 72, ("A", "SUPPORT"): 72,
                       ("B", "REFUTE"): 108, ("B", "SUPPORT"): 36,
                       ("U", "REFUTE"): 72, ("U", "SUPPORT"): 72}
    if labels != Counter(expected_labels):
        raise ValueError("PRIMARY_LABEL_DISTRIBUTION_CONFLICT")
    unique_prompts = len(global_prompt_hashes)
    if unique_prompts != 336:
        raise ValueError("PILOT_UNEXPECTED_CONTENT_EQUIVALENCE")
    return {
        "status": "PASS_BOUNDED_IN_MEMORY_PILOT",
        "pilot_roots_declared": len(roots),
        "logical_boolean_worlds": 8,
        "model_facing_rows_in_memory": len(rows),
        "unique_raw_prompt_bytes": unique_prompts,
        "expected_within_family_pre_repeat_rows": len(rows) - unique_prompts,
        "cross_family_exact_byte_duplicates": 0,
        "labels_by_query": {
            q: {label: labels[(q, label)] for label in ("REFUTE", "SUPPORT")}
            for q in QUERIES
        },
        "split_assignments": "NONE",
        "independent_synthetic_population": "NOT_ESTABLISHED",
        "surface_near_duplicate_review": "NOT_EVALUATED",
        "shortcut_attack_performance": "NOT_EVALUATED",
        "complete_lexical_leakage_freedom": "NOT_ESTABLISHED",
        "tokenizer_serialization": "NOT_EVALUATED",
        "model_evaluation": "NOT_EVALUATED",
        "materialized_cohort_feasibility": "NOT_EVALUATED",
        "native_state_causal_mediation": "NOT_ESTABLISHED",
        "PHASE3B": "BLOCKED_UNCHANGED",
    }
