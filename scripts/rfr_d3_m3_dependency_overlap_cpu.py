"""RFR-D3-M3: independent conditional and structural-overlap audit of M2 RAM rows.

No new root or corpus generation, no file I/O, no RNG, no split assignment,
no tokenization/model calls, no metric fitting. Uses frozen D1-D3/M1/M2 and
M2's existing in-memory API; distinguishes deterministic logical properties
from empirical performance and cross-family independence.
"""

from collections import Counter, defaultdict
from itertools import product
import re

from scripts import rfr_d3_m2_shortcut_surface_cpu as m2

RECORD = re.compile(
    r"^(Register record|Instrument log): logical-time=(10|20); "
    r"(filing A|requirement C|unrelated entry U|channel A|calibration C|unrelated sensor U) "
    r"is explicitly (TRUE|FALSE); source=([^;]+); provenance=([^;]+); "
    r"authentication=(register seal verified|register seal not verified|"
    r"instrument signature passed|instrument signature failed)\.$"
)
QUERY = re.compile(r"^Question: Is .+? ([ABU]) true based on the authenticated records and rule\?$")
AUTHENTIC = frozenset(("register seal verified", "instrument signature passed"))
PREDICATES = {"filing A": "A", "requirement C": "C", "unrelated entry U": "U",
              "channel A": "A", "calibration C": "C", "unrelated sensor U": "U"}


def _visible_signature(visible):
    """Independent role-level parser, not M2's parser/rendering implementation.

    A shared structure across domains indicates the SAME realized Boolean
    problem presented using different domain vocabulary; this is not a
    similarity-threshold or a declaration of text-level near-duplicates.
    """
    if type(visible) is not str:
        raise ValueError("M3_VISIBLE_TYPE")
    lines = visible.splitlines()
    if len(lines) < 6 or not lines[1].startswith("Rule:"):
        raise ValueError("M3_VISIBLE_STRUCTURE")
    if not lines[0] in ("Synthetic register evidence:", "Synthetic instrument evidence:"):
        raise ValueError("M3_VISIBLE_HEADER")
    domain = ("REGISTRY_RECORD" if lines[0] == "Synthetic register evidence:"
              else "INSTRUMENT_LOG")
    events = []
    questions = []
    delay_count = 0
    for i, line in enumerate(lines[2:], start=2):
        match = RECORD.fullmatch(line)
        if match:
            stem, time, noun, val, source, prov, auth = match.groups()
            if (domain == "REGISTRY_RECORD" and stem != "Register record") or (
                domain == "INSTRUMENT_LOG" and stem != "Instrument log"):
                raise ValueError("M3_DOMAIN_STEM_MISMATCH")
            events.append((int(time), PREDICATES[noun], val == "TRUE",
                           auth in AUTHENTIC, source, prov))
        elif (query_match := QUERY.fullmatch(line)):
            questions.append((i, query_match.group(1)))
        elif line.startswith("Neutral note:"):
            if line != "Neutral note: a decorative margin mark contains no A/C/U assertion.":
                raise ValueError("M3_NON_NEUTRAL_DELAY")
            delay_count += 1
        else:
            raise ValueError("M3_UNPARSEABLE_VISIBLE_LINE")
    if len(questions) != 1 or len(events) not in (3, 4) or delay_count not in (0, 1):
        raise ValueError("M3_STRUCTURE_COUNTS")
    question_index, query = questions[0]
    if question_index not in (2, len(lines)-1):
        raise ValueError("M3_QUERY_POSITION")
    position = "QUERY_FIRST" if question_index == 2 else "QUERY_LAST"
    return {
        "domain": domain, "query": query, "position": position,
        "events": tuple(events), "delayed": bool(delay_count),
        "skeleton": (domain, query, position, bool(delay_count),
                     tuple((time, target) for time, target, *_ in events)),
        "cross_domain_structure": (query, position, bool(delay_count), tuple(events)),
    }


def _index_rows(roots, rows):
    if type(roots) is not list or type(rows) is not list:
        raise ValueError("M3_INVALID_COLLECTION")
    rmap = {r["case_family_id"]: r for r in roots}
    if len(rmap) != len(roots):
        raise ValueError("M3_DUPLICATE_ROOT")
    indexed = {}
    for row in rows:
        meta = row["sealed_metadata"]
        factors = tuple(meta[k] for k in (
            "case_family_id", "update_arm", "readout_time", "query_target",
            "presentation_order_id", "query_position_id", "source_rotation_id"))
        if factors in indexed:
            raise ValueError("M3_DUPLICATE_FACTOR")
        indexed[factors] = row
    return rmap, indexed


def _paired_dependency(rmap, indexed):
    """Check literal A/B/U Boolean expectations for matched PRE/POST/DELAYED.

    Count paired comparisons separately from *unique* logical worlds and roots.
    The check is independent of M2's rule-agnostic shortcut predictor.
    """
    matrix = Counter()
    paired_count = 0
    for fid, root in rmap.items():
        a, c, u = (root["root_initial_"+k] for k in ("A", "C", "U"))
        if any(type(z) is not bool for z in (a,c,u)):
            raise ValueError("M3_ROOT_TYPE")
        for order, position, rotation, arm, query in product(
            m2.ORDERS, m2.POSITIONS, m2.SOURCE_ROTATIONS, m2.ARMS, m2.QUERIES
        ):
            base = (fid, arm, "PRE", query, order, position, rotation)
            post = (fid, arm, "POST", query, order, position, rotation)
            delayed = (fid, arm, "DELAYED", query, order, position, rotation)
            try:
                pre_row, post_row, delay_row = (indexed[k] for k in (base,post,delayed))
            except KeyError as exc:
                raise ValueError("M3_MISSING_MATCHED_VARIANT") from exc
            initial = {"A":a, "B":a and c, "U":u}[query]
            next_a = not a if arm == "VALID_FLIP" else a
            after = {"A":next_a, "B":next_a and c, "U":u}[query]
            values = tuple(x["sealed_metadata"]["final_label"] for x in (
                pre_row, post_row, delay_row))
            expected = tuple("SUPPORT" if x else "REFUTE" for x in (
                initial, after, after))
            if values != expected:
                raise ValueError("M3_PAIRED_BOOLEAN_MISMATCH")
            if pre_row["sealed_metadata"]["root_truth_stratum"] != (
                "".join("T" if x else "F" for x in (a,c,u))):
                raise ValueError("M3_ROOT_STRATUM_MISMATCH")
            matrix[(query,"C_T" if c else "C_F",arm,
                    "CHANGED" if initial != after else "UNCHANGED")] += 1
            paired_count += 1
    return {"paired_pre_post_delayed_comparisons": paired_count,
            "conditional_transition_counts": {
                "/".join(k): n for k,n in sorted(matrix.items())
            }, "status": "PASS_DETERMINISTIC_PAIRED_LOGIC_ONLY"}


def _independent_c_omission_heuristic(signature):
    """Read newest authenticated A for A/B, newest U for U; deliberately omit C."""
    target = "U" if signature["query"] == "U" else "A"
    eligible = [(time, value) for time, name, value, authenticated, *_ in
                signature["events"] if name == target and authenticated]
    if not eligible:
        raise ValueError("M3_BASE_FACT_MISSING")
    return max(eligible, key=lambda pair: pair[0])[1]


def _structural_overlap(rows):
    group = defaultdict(list)
    skeleton_families = defaultdict(set)
    heuristic = Counter()
    risk_examples = []
    for row in rows:
        meta = row["sealed_metadata"]
        signature = _visible_signature(row["visible_input"])
        if signature["domain"] != meta["surface_domain"] or signature["query"] != meta["query_target"]:
            raise ValueError("M3_VISIBLE_METADATA_DISAGREES")
        fid = meta["case_family_id"]
        group[signature["cross_domain_structure"]].append((fid, signature["domain"],
                                                              meta["root_truth_stratum"]))
        skeleton_families[signature["skeleton"]].add(fid)
        prediction = _independent_c_omission_heuristic(signature)
        gold = meta["final_label"] == "SUPPORT"
        k = (meta["query_target"], "C_"+meta["root_truth_stratum"][1])
        heuristic[(k,"CORRECT" if prediction == gold else "WRONG")] += 1
    cross_pairs = 0
    shared_signatures = 0
    covered_rows = 0
    family_pairs = set()
    for signature, records in group.items():
        by_fid = Counter(fid for fid,_,_ in records)
        by_domain = defaultdict(set)
        for fid, domain, stratum in records:
            by_domain[domain].add(fid)
        if len(by_fid) > 1:
            pairs = (sum(by_fid.values())**2 - sum(n*n for n in by_fid.values()))//2
            cross_pairs += pairs
            shared_signatures += 1
            covered_rows += len(records)
            for left in by_domain.get("REGISTRY_RECORD",set()):
                for right in by_domain.get("INSTRUMENT_LOG",set()):
                    family_pairs.add(tuple(sorted((left,right))))
            if len(risk_examples) < 3:
                risk_examples.append({"query":signature[0],
                                      "record_count":len(records),
                                      "root_count":len(by_fid)})
    # All cross-domain signatures map to matching logical worlds, not to
    # independently drawn case-families. No near-duplicate threshold is selected.
    if not family_pairs or not shared_signatures:
        raise ValueError("M3_NO_STRUCTURAL_REPLICATION_FOUND")
    by_condition = {}
    for query in m2.QUERIES:
        for c in ("F","T"):
            k = (query, "C_"+c)
            correct, wrong = heuristic[(k,"CORRECT")], heuristic[(k,"WRONG")]
            by_condition["/".join(k)] = {"correct":correct,"incorrect":wrong,"total":correct+wrong}
    total = sum(v["total"] for v in by_condition.values())
    return {
        "status":"STRUCTURAL_REPLICATION_DETECTED_NOT_INDEPENDENCE_PASS",
        "raw_prompt_rows":len(rows),
        "cross_family_equal_role_structure_signatures":shared_signatures,
        "cross_family_equal_role_structure_rows":covered_rows,
        "cross_family_equal_role_structure_row_pairs":cross_pairs,
        "distinct_cross_domain_semantic_world_pairings":len(family_pairs),
        "template_skeleton_classes":len(skeleton_families),
        "max_families_sharing_template_skeleton":max(map(len,skeleton_families.values())),
        "c_omission_heuristic_by_query_and_C":by_condition,
        "c_omission_heuristic_pooled":{
            "correct":sum(v["correct"] for v in by_condition.values()),
            "incorrect":sum(v["incorrect"] for v in by_condition.values()),
            "total":total,
        },
        "structural_match_examples_without_text":risk_examples,
        "semantic_independence":"NOT_ESTABLISHED",
        "textual_near_duplicate_similarity":"NOT_EVALUATED",
        "no_leakage_claim":"NOT_ESTABLISHED",
    }



def _u_literal_mask(text, domain):
    """Mask ONLY the displayed U-record Boolean, not other truths/cues."""
    marker = "unrelated entry U is explicitly " if domain == "REGISTRY_RECORD" else "unrelated sensor U is explicitly "
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if marker in line]
    if len(matches) != 1:
        raise ValueError("M3_U_RECORD_COUNT")
    index = matches[0]
    line = lines[index]
    if marker + "TRUE" in line:
        lines[index] = line.replace(marker + "TRUE", marker + "<U_BOOL>", 1)
    elif marker + "FALSE" in line:
        lines[index] = line.replace(marker + "FALSE", marker + "<U_BOOL>", 1)
    else:
        raise ValueError("M3_U_RECORD_LITERAL")
    return tuple(lines)


def _controlled_one_literal_neighbors(rmap, indexed):
    """Count exact single-U-truth edits across distinct families, not edit-distance.

    This is an intentionally constrained close-neighbor audit, not a global
    near-duplicate scan, and it does not set a similarity threshold.
    """
    world = defaultdict(dict)
    for fid, root in rmap.items():
        key = (root["surface_domain"], root["root_initial_A"], root["root_initial_C"])
        u = root["root_initial_U"]
        if u in world[key]:
            raise ValueError("M3_REUSED_WORLD_WITHIN_DOMAIN")
        world[key][u] = fid
    paired_families = 0
    near_rows = 0
    label_discordant = 0
    for key, sibling in world.items():
        if set(sibling) != {False, True}:
            raise ValueError("M3_MISSING_U_SIBLING")
        paired_families += 1
        for arm, readout, query, order, position, rotation in product(
            m2.ARMS, m2.READOUTS, m2.QUERIES, m2.ORDERS,
            m2.POSITIONS, m2.SOURCE_ROTATIONS
        ):
            tail = (arm, readout, query, order, position, rotation)
            left = indexed[(sibling[False],)+tail]
            right = indexed[(sibling[True],)+tail]
            a, b = left["visible_input"], right["visible_input"]
            if a == b or _u_literal_mask(a,key[0]) != _u_literal_mask(b,key[0]):
                continue
            near_rows += 1
            if left["sealed_metadata"]["final_label"] != right["sealed_metadata"]["final_label"]:
                label_discordant += 1
    return {
        "distinct_provisional_root_pairs_differing_only_U":paired_families,
        "matched_row_pairs_exactly_one_U_literal_apart":near_rows,
        "matched_row_pairs_differing_in_gold_label":label_discordant,
        "near_duplicate_global_search":"NOT_EVALUATED",
        "cross_split_near_duplicate_risk":"NOT_EVALUATED_NO_SPLITS",
    }


def audit_m3(roots, rows):
    """Read-only analysis of supplied, already authorized M2 in-memory rows."""
    if m2.audit_pilot(roots, rows)["status"] != "PASS_BOUNDED_M2_SURFACE_PILOT":
        raise ValueError("M3_UPSTREAM_PILOT_BLOCKED")
    rmap, index = _index_rows(roots, rows)
    paired = _paired_dependency(rmap, index)
    overlap = _structural_overlap(rows)
    neighbors = _controlled_one_literal_neighbors(rmap, index)
    if paired["paired_pre_post_delayed_comparisons"] != len(roots)*8*3*3:
        raise ValueError("M3_PAIRED_COVERAGE")
    if overlap["c_omission_heuristic_pooled"]["total"] != len(rows):
        raise ValueError("M3_SHORTCUT_COVERAGE")
    return {
        "status":"M3_BOUNDED_AUDIT_COMPLETE_WITH_COHORT_BLOCKERS",
        "baseline":"RFR_D3_M2_IN_MEMORY_ONLY",
        "provisional_root_count":len(roots),
        "distinct_boolean_worlds":8,
        "paired_dependency":paired,
        "structural_overlap":overlap,
        "controlled_near_neighbors":neighbors,
        "independent_cohort_feasibility":"NOT_ESTABLISHED",
        "materialized_cohort_feasibility":"NOT_EVALUATED",
        "trained_model_performance":"NOT_EVALUATED",
        "tokenizer_serialization":"NOT_EVALUATED",
        "native_state_causal_mediation":"NOT_ESTABLISHED",
        "PHASE3B":"BLOCKED_UNCHANGED",
    }
