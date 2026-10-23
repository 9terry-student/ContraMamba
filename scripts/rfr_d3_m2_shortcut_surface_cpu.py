"""RFR-D3-M2: bounded, deterministic in-memory surface-counterbalance pilot.

No persistence, RNG, train/dev/test assignment, tokenizer, model or GPU.
The 16 M1 root identities are used ONLY in sealed metadata; this stage does
not claim independently sampled micro-worlds or eliminate semantic shortcuts.
Every order, query-position and t1 source-alias rotation is crossed within each
root, arm, readout and query. Frozen D1, D2 and D3 remain authoritative.
"""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json

from scripts import rfr_d1_deterministic_oracle_cpu as d1
from scripts import rfr_d3_cohort_contract_cpu as d3
from scripts import rfr_d3_m1_cpu_pilot as m1

ORDERS = ("CHRONOLOGICAL", "REVERSED")
POSITIONS = ("QUERY_FIRST", "QUERY_LAST")
SOURCE_ROTATIONS = ("COPPER_VALID", "SLATE_VALID")
ARMS = ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")
READOUTS = ("PRE", "POST", "DELAYED")
QUERIES = ("A", "B", "U")
ROOT_COUNT = 16
EXPECTED_ROWS = ROOT_COUNT * len(ORDERS) * len(POSITIONS) * len(SOURCE_ROTATIONS) * 27
SOURCE_ALIASES = ("SRC_COPPER", "SRC_SLATE")
PROVENANCE_ALIASES = ("PROV_COPPER", "PROV_SLATE")
VISIBLE_DENY = ("case_family_id", "split_group_id", "final_label", "target_status",
                "root_initial_", "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP",
                "NOT_ENTITLED", "REFUTE", "SUPPORT")


def _sha(data):
    return sha256(data).hexdigest()


def _canonical(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def _event(event_id, target, value, authenticated, time, source_id, provenance_id):
    return {"event_id": event_id, "target": target, "asserted_value": value,
            "source_authenticated": authenticated, "event_time": time,
            "source_id": source_id, "provenance_id": provenance_id,
            "oracle_version": "RFR_D1_ORACLE_V1", "derivation_rule_id": "B_AND_A_C"}


def _t1_source(arm, rotation):
    if rotation not in SOURCE_ROTATIONS or arm not in ARMS:
        raise ValueError("INVALID_SOURCE_ROTATION_OR_ARM")
    authentic = arm != "INVALID_FLIP"
    index = (0 if authentic else 1) if rotation == "COPPER_VALID" else (1 if authentic else 0)
    return SOURCE_ALIASES[index], PROVENANCE_ALIASES[index]


def declare(root, arm, readout, query, order, position, rotation):
    """Construct one typed protocol with no root-world-derived surface choice."""
    if (arm not in ARMS or readout not in READOUTS or query not in QUERIES
            or order not in ORDERS or position not in POSITIONS
            or rotation not in SOURCE_ROTATIONS):
        raise ValueError("INVALID_M2_FACTOR")
    events = [_event("t0_" + k, k, root["root_initial_" + k], True, 10,
                     "BASE_SOURCE_" + k, "BASE_PROV_" + k) for k in ("A", "C", "U")]
    if readout != "PRE":
        new_a = root["root_initial_A"] if arm == "VALID_SHAM" else not root["root_initial_A"]
        src, prov = _t1_source(arm, rotation)
        events.append(_event("t1_A", "A", new_a, arm != "INVALID_FLIP", 20, src, prov))
    ordered = [item["event_id"] for item in events]
    if order == "REVERSED":
        ordered.reverse()
    declaration = {
        "query_target": query, "events": events, "presentation_order": ordered,
        "visible_sections": ["rule", "records", "authentication", "logical_time", "query"]
                            + (["neutral_delay"] if readout == "DELAYED" else []),
        "sealed_field_names": sorted(d3.SEALED_REQUIRED),
        "distractors": ([{"distractor_id": "neutral_m2", "kind": "NEUTRAL"}]
                        if readout == "DELAYED" else []),
    }
    d3.audit_event_declaration(root, arm, readout, declaration)
    return declaration


def render(root, declaration, position):
    """No family ID, root index or unique entity name is included in the text."""
    if position not in POSITIONS:
        raise ValueError("INVALID_POSITION")
    domain = root["surface_domain"]
    header = ("Synthetic register evidence:" if domain == "REGISTRY_RECORD"
              else "Synthetic instrument evidence:")
    rule = m1._rule(domain, "the anonymous case")
    question = m1._query(domain, declaration["query_target"])
    records = {record["event_id"]: record for record in declaration["events"]}
    lines = [header, rule]
    if position == "QUERY_FIRST":
        lines.append(question)
    lines.extend(m1._record(domain, records[eid]) for eid in declaration["presentation_order"])
    if declaration["distractors"]:
        lines.append("Neutral note: a decorative margin mark contains no A/C/U assertion.")
    if position == "QUERY_LAST":
        lines.append(question)
    return "\n".join(lines)


def _plan(roots):
    return [{
        "case_family_id": root["case_family_id"],
        "split_group_id": root["split_group_id"],
        "variant_id": ":".join((root["case_family_id"], arm, readout, query,
                                order, position, rotation)),
        "update_arm": arm, "readout_time": readout, "query_target": query,
        "presentation_order_id": order,
        "paraphrase_family_id": position + ":" + rotation,
    } for root in roots for order, position, rotation, arm, readout, query in
        product(ORDERS, POSITIONS, SOURCE_ROTATIONS, ARMS, READOUTS, QUERIES)]


def generate_pilot():
    """Generate M2 rows exclusively in RAM, reusing the frozen M1 root roster."""
    roots, lineage = m1.root_specs()
    grammars = m1.grammar_definitions()
    bridge = d3.audit_d2_plan_bridge(roots, _plan(roots), grammars, lineage)
    if bridge["status"] != "PASS_STATIC_CONTRACT":
        raise ValueError("D2_D3_PLAN_BLOCKED")
    rows = []
    for root in roots:
        for order, position, rotation in product(ORDERS, POSITIONS, SOURCE_ROTATIONS):
            for query in QUERIES:
                triple = {arm: declare(root, arm, "POST", query, order, position, rotation)
                          for arm in ARMS}
                d3.audit_matched_contrasts(root, triple)
            for arm, readout, query in product(ARMS, READOUTS, QUERIES):
                decl = declare(root, arm, readout, query, order, position, rotation)
                result = d1.evaluate_records(decl["events"], query, readout, decl["distractors"])
                visible = render(root, decl, position)
                meta = {
                    "case_family_id": root["case_family_id"],
                    "split_group_id": root["split_group_id"],
                    "variant_id": ":".join((root["case_family_id"], arm, readout, query,
                                            order, position, rotation)),
                    "update_arm": arm, "readout_time": readout, "query_target": query,
                    "surface_domain": root["surface_domain"],
                    "root_truth_stratum": "".join("T" if root["root_initial_" + k] else "F"
                                                      for k in ("A", "C", "U")),
                    "presentation_order_id": order, "query_position_id": position,
                    "source_rotation_id": rotation,
                    "target_status": result["target_status"],
                    "final_label": result["final_label"],
                    "final_label_id": result["final_label_id"],
                    "events": decl["events"], "distractors": decl["distractors"],
                    "presentation_order": decl["presentation_order"],
                    "prompt_sha256": _sha(visible.encode("utf-8")),
                    "canonical_event_sha256": _sha(_canonical(decl["events"])),
                }
                rows.append({"visible_input": visible, "sealed_metadata": meta})
    return roots, rows


def audit_pilot(roots, rows):
    """Adversarial bounded audit; still NOT a cohort-independent validity gate."""
    if type(roots) is not list or type(rows) is not list:
        raise ValueError("INVALID_INPUT_COLLECTION")
    if len(roots) != ROOT_COUNT or len(rows) != EXPECTED_ROWS:
        raise ValueError("M2_COUNT_MISMATCH")
    by_id = {r["case_family_id"]: r for r in roots}
    if len(by_id) != ROOT_COUNT:
        raise ValueError("ROOT_COLLISION")
    seen = set()
    seen_prompt_families = defaultdict(set)
    positions = defaultdict(set)
    source_auth = defaultdict(set)
    label_count = Counter()
    label_by_source_query = defaultdict(set)
    for row in rows:
        if set(row) != {"visible_input", "sealed_metadata"}:
            raise ValueError("ROW_SCHEMA_MISMATCH")
        visible, meta = row["visible_input"], row["sealed_metadata"]
        if type(visible) is not str or not visible or not isinstance(meta, dict):
            raise ValueError("ROW_TYPE_MISMATCH")
        fid = meta["case_family_id"]
        if fid not in by_id:
            raise ValueError("UNKNOWN_ROOT")
        root = by_id[fid]
        arm, readout, query = (meta[key] for key in ("update_arm", "readout_time", "query_target"))
        order, position, rotation = (meta[key] for key in (
            "presentation_order_id", "query_position_id", "source_rotation_id"))
        unique = (fid, arm, readout, query, order, position, rotation)
        if unique in seen or meta["variant_id"] != ":".join(unique):
            raise ValueError("DUPLICATE_OR_INCONSISTENT_VARIANT")
        seen.add(unique)
        if meta["split_group_id"] != root["split_group_id"] or "split" in meta:
            raise ValueError("SPLIT_GROUP_OR_ASSIGNMENT")
        if meta["surface_domain"] != root["surface_domain"]:
            raise ValueError("DOMAIN_CONFLICT")
        expected_stratum = "".join("T" if root["root_initial_" + k] else "F"
                                   for k in ("A", "C", "U"))
        if meta["root_truth_stratum"] != expected_stratum:
            raise ValueError("ROOT_STRATUM_CONFLICT")
        if meta["prompt_sha256"] != _sha(visible.encode("utf-8")) or meta[
            "canonical_event_sha256"] != _sha(_canonical(meta["events"])):
            raise ValueError("HASH_MISMATCH")
        if any(word in visible for word in VISIBLE_DENY) or fid in visible or "m1_" in visible:
            raise ValueError("VISIBLE_ID_OR_GOLD_LEAKAGE")
        expected_declaration = declare(root, arm, readout, query, order, position, rotation)
        # Compare typed events and whole displayed text; hashes alone do not authenticate meaning.
        if (meta["events"] != expected_declaration["events"] or
            meta["presentation_order"] != expected_declaration["presentation_order"] or
            meta["distractors"] != expected_declaration["distractors"] or
            visible != render(root, expected_declaration, position)):
            raise ValueError("RENDER_OR_DECLARATION_MISMATCH")
        oracle = d1.evaluate_records(meta["events"], query, readout, meta["distractors"])
        if (meta["target_status"], meta["final_label"], meta["final_label_id"]) != (
                oracle["target_status"], oracle["final_label"], oracle["final_label_id"]):
            raise ValueError("FROZEN_ORACLE_MISMATCH")
        if (readout == "PRE") != ("logical-time=20" not in visible):
            raise ValueError("PRE_POST_EVENT_LEAKAGE")
        if (readout == "DELAYED") != ("Neutral note:" in visible):
            raise ValueError("DELAYED_SURFACE_MISMATCH")
        lines = visible.splitlines()
        question_at = next((i for i, line in enumerate(lines) if line.startswith("Question:")), -1)
        expected_at = 2 if position == "QUERY_FIRST" else len(lines) - 1
        if question_at != expected_at:
            raise ValueError("QUERY_POSITION_NOT_CROSSED")
        displayed = [line for line in lines if "logical-time=" in line]
        timestamps = [20 if "logical-time=20" in line else 10 for line in displayed]
        if order == "CHRONOLOGICAL" and timestamps != sorted(timestamps):
            raise ValueError("CHRONOLOGICAL_ORDER_CONFLICT")
        if order == "REVERSED" and timestamps != sorted(timestamps, reverse=True):
            raise ValueError("REVERSED_ORDER_CONFLICT")
        positions[(fid, arm, readout, query)].add((order, position, rotation))
        seen_prompt_families[meta["prompt_sha256"]].add(fid)
        label_count[(query, meta["final_label"])] += 1
        if readout != "PRE":
            t1 = meta["events"][-1]
            source_auth[(fid, arm, t1["source_id"])].add(t1["source_authenticated"])
            label_by_source_query[(t1["source_id"], query, readout)].add(meta["final_label"])
    if len(seen) != EXPECTED_ROWS:
        raise ValueError("INCOMPLETE_FACTORIAL")
    expected_cross = set(product(ORDERS, POSITIONS, SOURCE_ROTATIONS))
    if len(positions) != ROOT_COUNT * 27 or any(v != expected_cross for v in positions.values()):
        raise ValueError("INCOMPLETE_SURFACE_COUNTERBALANCE")
    if any(len(families) != 1 for families in seen_prompt_families.values()):
        raise ValueError("CROSS_FAMILY_EXACT_PROMPT_DUPLICATE")
    # Each alias must be observed with BOTH authentications across arm realizations
    # and across each root via source rotation; cue alone cannot encode authentication.
    by_source_auth = defaultdict(set)
    for (fid, arm, source), flags in source_auth.items():
        by_source_auth[(fid, source)].update(flags)
    if len(by_source_auth) != ROOT_COUNT * 2 or any(flags != {True, False}
                                                 for flags in by_source_auth.values()):
        raise ValueError("SOURCE_ALIAS_AUTHENTICATION_SHORTCUT")
    for source, query, readout in product(SOURCE_ALIASES, QUERIES, ("POST", "DELAYED")):
        if label_by_source_query[(source, query, readout)] != {"REFUTE", "SUPPORT"}:
            raise ValueError("SOURCE_ALIAS_DETERMINISTIC_LABEL_CUE")
    expected_labels = {("A", "REFUTE"): 72 * 8, ("A", "SUPPORT"): 72 * 8,
                       ("B", "REFUTE"): 108 * 8, ("B", "SUPPORT"): 36 * 8,
                       ("U", "REFUTE"): 72 * 8, ("U", "SUPPORT"): 72 * 8}
    if label_count != Counter(expected_labels):
        raise ValueError("LABEL_DISTRIBUTION_MISMATCH")
    # Do not turn this diagnostic into a fake independently materialized corpus PASS.
    return {
        "status": "PASS_BOUNDED_M2_SURFACE_PILOT",
        "provisional_roots": ROOT_COUNT,
        "distinct_boolean_worlds": 8,
        "related_rows_in_memory": len(rows),
        "factorial_surface_realizations_per_root": 8,
        "unique_raw_prompt_bytes": len(seen_prompt_families),
        "cross_family_exact_prompt_duplicates": 0,
        "source_alias_authentication_counterbalance": "PASS_IN_PILOT",
        "order_and_position_counterbalance": "PASS_IN_PILOT",
        "label_counts_by_query": {q: {label: label_count[(q, label)]
                                     for label in ("REFUTE", "SUPPORT")} for q in QUERIES},
        "semantic_root_independence": "NOT_ESTABLISHED",
        "cross_family_near_duplicate_safety": "NOT_ESTABLISHED",
        "actual_shortcut_predictor_test": "NOT_EVALUATED",
        "tokenizer_serialization": "NOT_EVALUATED",
        "materialized_cohort_feasibility": "NOT_EVALUATED",
        "model_evaluation": "NOT_EVALUATED",
        "native_state_causal_mediation": "NOT_ESTABLISHED",
        "PHASE3B": "BLOCKED_UNCHANGED",
    }


def _surface_records(visible):
    """Read only visible textual records for fixed, deliberately weak attacks."""
    import re
    pattern = re.compile(
        r"^(?:Register record|Instrument log): logical-time=(\d+); "
        r"(.+?) is explicitly (TRUE|FALSE); source=([^;]+); provenance=([^;]+); "
        r"authentication=(.+)\.$"
    )
    out = []
    for line in visible.splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            continue
        time, noun, value, source, provenance, auth = match.groups()
        target = noun[-1]
        if target not in ("A", "C", "U"):
            raise ValueError("INVALID_VISIBLE_PREDICATE")
        out.append({"time": int(time), "target": target, "value": value == "TRUE",
                    "source": source, "provenance": provenance,
                    "authenticated": auth in ("register seal verified", "instrument signature passed")})
    return out


def _visible_query(visible):
    question = next((line for line in visible.splitlines() if line.startswith("Question:")), "")
    import re
    match = re.search(r"\b([ABU]) true based on", question)
    if match is None:
        raise ValueError("INVALID_VISIBLE_QUERY")
    return match.group(1)


def _predict_shortcuts(visible):
    """Seven fixed no-fit heuristics, not learned models or scientific evaluation."""
    events = _surface_records(visible)
    query = _visible_query(visible)
    updates = [e for e in events if e["time"] == 20]
    update = updates[0] if updates else None
    desired_base = "A" if query == "B" else query
    relevant = [e for e in events if e["target"] == desired_base]
    newest = [e for e in relevant if e["authenticated"]]
    newest = max(newest, key=lambda e: e["time"]) if newest else None
    last = relevant[-1] if relevant else None
    return {
        "majority": False,
        "t1_assertion_only": update["value"] if update and query != "U" else False,
        "authentication_only": bool(update and update["authenticated"]),
        "last_mention": last["value"] if last else False,
        "query_only": query == "A",
        "rule_agnostic": newest["value"] if newest else False,
        "source_template_only": bool(update and update["source"] == "SRC_COPPER"),
    }


def diagnose_shortcuts(rows):
    """Report conditional confusion of seven fixed surface-only attacks.

    Caller should independently audit rows first. The fixed heuristics neither
    train nor tune on gold outcomes, and their failure is NOT proof of no leakage.
    """
    if type(rows) is not list or not rows:
        raise ValueError("EMPTY_SHORTCUT_DIAGNOSTIC")
    confusions = defaultdict(Counter)
    for row in rows:
        visible, meta = row["visible_input"], row["sealed_metadata"]
        predicted = _predict_shortcuts(visible)
        gold = meta["final_label"]
        for name, prediction in predicted.items():
            result = "SUPPORT" if prediction else "REFUTE"
            confusions[(name, meta["query_target"], meta["update_arm"],
                        "C_" + meta["root_truth_stratum"][1])][
                (gold, result)] += 1
    totals = defaultdict(Counter)
    for (name, query, arm, c_stratum), hist in confusions.items():
        totals[name]["correct"] += sum(count for (gold, pred), count in hist.items()
                                        if gold == pred)
        totals[name]["total"] += sum(hist.values())
    if set(totals) != {"majority", "t1_assertion_only", "authentication_only",
                       "last_mention", "query_only", "rule_agnostic",
                       "source_template_only"}:
        raise ValueError("INCOMPLETE_SHORTCUT_SET")
    return {
        "status": "BOUNDED_FIXED_SHORTCUT_DIAGNOSTIC_ONLY",
        "fixed_heuristics": {
            name: {"correct": totals[name]["correct"],
                   "incorrect": totals[name]["total"] - totals[name]["correct"],
                   "total": totals[name]["total"]}
            for name in sorted(totals)
        },
        "conditional_confusions": {
            "/".join((name, query, arm, c_stratum)): {
                gold + "->" + pred: count for (gold, pred), count in sorted(hist.items())
            }
            for (name, query, arm, c_stratum), hist in sorted(confusions.items())
        },
        "trained_model_evaluation": "NOT_EVALUATED",
        "general_shortcut_freedom": "NOT_ESTABLISHED",
        "statistical_independence": "NOT_ESTABLISHED",
    }
