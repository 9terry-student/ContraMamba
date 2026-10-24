"""RFR-H2-P1: CPU-only temporal serialization and readout *plan* contract.

Audit externally supplied, hypothetical single-case declarations only.
No root draw, RNG, corpus/row file, tokenizer, model, checkpoint, training,
model evaluation, native state, cache, filesystem IO, or split assignment.
Frozen RFR-D1 owns the logical oracle. This module does not score an LM.
"""

from hashlib import sha256
from itertools import product
from collections.abc import Mapping

RULE_TEXT = (
    "Rule: B is true exactly when both A and C are true.\n"
    "Evidence policy: Apply authenticated records only; the newest logical-time "
    "authenticated record for a variable takes precedence.\n"
)
DELAY_TEXT = "Neutral interval: an unrelated blank sheet was turned.\n"
ARMS = ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")
READOUTS = ("PRE", "POST", "DELAYED")
QUERIES = ("A", "B", "U")
ROW_KEYS = frozenset((
    "arm", "readout", "query_target", "rule_text", "t0_text",
    "t1_text", "delay_text", "query_text", "byte_anchors",
    "pre_query_sha256", "declared_label", "declared_label_id",
))
ANCHOR_KEYS = frozenset((
    "rule_end", "t0_end", "t1_end", "delay_end", "query_start",
))
CLASS_LABELS = ("REFUTE", "NOT_ENTITLED", "SUPPORT")
STATUS = "H2_P1_STATIC_CONTRACT_PASS_NO_MODEL_OR_TOKENIZER"


def _need(ok, code):
    if not ok:
        raise ValueError("H2_P1_" + code)


def _bool_world(world):
    _need(isinstance(world, Mapping) and set(world) == {"A", "C", "U"}, "WORLD_SCHEMA")
    _need(all(type(world[k]) is bool for k in ("A", "C", "U")), "WORLD_STRICT_BOOLEAN")


def _value(v):
    return "TRUE" if v else "FALSE"


def _t0_text(world):
    return (
        "Initial authenticated records (logical time 10):\n"
        + "".join(
            f"{key}={_value(world[key])}; authenticated=yes; "
            f"source=BASE_{key}.\n"
            for key in ("A", "C", "U")
        )
    )


def _t1_text(world, arm):
    value = world["A"] if arm == "VALID_SHAM" else not world["A"]
    authenticated = arm != "INVALID_FLIP"
    return (
        f"Update (logical time 20): A={_value(value)}; "
        f"authenticated={'yes' if authenticated else 'no'}; "
        f"source={'REV_A' if authenticated else 'UNVERIFIED_A'}.\n"
    )


def _query_text(query):
    return f"Query: Is {query} true under the current verified record?\nAnswer: "


def _byte_anchors(rule, t0, t1, delay):
    a = len(rule.encode("utf-8"))
    b = a + len(t0.encode("utf-8"))
    c = b + len(t1.encode("utf-8")) if t1 is not None else None
    d = c + len(delay.encode("utf-8")) if delay is not None else None
    return {
        "rule_end": a,
        "t0_end": b,
        "t1_end": c,
        "delay_end": d,
        "query_start": d if delay is not None else c if t1 is not None else b,
    }


def _event(name, value, authenticated, time, *, source):
    return {
        "event_id": f"t{time}_{name}", "target": name,
        "asserted_value": value, "source_authenticated": authenticated,
        "event_time": time, "source_id": source,
        "provenance_id": "PROV_" + source,
        "oracle_version": "RFR_D1_ORACLE_V1", "derivation_rule_id": "B_AND_A_C",
    }


def _compare_oracle(world, arm, readout, query):
    """Compare physical PRE exclusion with the frozen oracle's time-filter logic."""
    from scripts import rfr_d1_deterministic_oracle_cpu as d1

    events = [_event(k, world[k], True, 10, source="BASE_" + k) for k in "ACU"]
    if readout != "PRE":
        authenticated = arm != "INVALID_FLIP"
        value = world["A"] if arm == "VALID_SHAM" else not world["A"]
        events.append(_event("A", value, authenticated, 20,
                             source="REV_A" if authenticated else "UNVERIFIED_A"))
    distractors = ([{"distractor_id": "hypothetical_neutral_1", "kind": "NEUTRAL"}]
                   if readout == "DELAYED" else [])
    observed = d1.evaluate_records(events, query, readout, distractors)
    reference = d1.primary_fixture(world["A"], world["C"], world["U"],
                                   arm, readout, query)
    for key in ("A", "B", "C", "U", "target_status", "final_label", "final_label_id"):
        _need(observed[key] == reference[key], "ORACLE_DISAGREEMENT_" + key)
    return observed


def audit_hypothetical_case(world, rows):
    """Verify a full 3x3x3 *declared* single-world table, not a sampled cohort.

    Only fixed-form provisional text can PASS. All literal input strings are
    supplied by the caller; no model-ready corpus or root is produced here.
    Real-world semantic neutrality, tokenizer anchoring, and LM head remain untested.
    """
    _bool_world(world)
    _need(type(rows) in (list, tuple) and len(rows) == 27, "EXACT_27_DECLARATIONS")
    observed = {}
    for row in rows:
        _need(isinstance(row, Mapping) and set(row) == ROW_KEYS, "ROW_SCHEMA")
        arm, readout, query = (row[k] for k in ("arm", "readout", "query_target"))
        _need(arm in ARMS and readout in READOUTS and query in QUERIES,
              "FACTOR_VALUE")
        key = (arm, readout, query)
        _need(key not in observed, "DUPLICATE_FACTOR")
        _need(row["rule_text"] == RULE_TEXT, "RULE_CHANGED_OR_LEAKED")
        _need(row["t0_text"] == _t0_text(world), "T0_CHANGED_OR_LEAKED")
        expected_t1 = None if readout == "PRE" else _t1_text(world, arm)
        expected_delay = DELAY_TEXT if readout == "DELAYED" else None
        _need(row["t1_text"] == expected_t1, "T1_WRONG_OR_FUTURE_LEAKAGE")
        _need(row["delay_text"] == expected_delay, "DELAY_NON_NEUTRAL_OR_MISSING")
        _need(row["query_text"] == _query_text(query), "QUERY_WRONG_OR_ANSWER_LEAKAGE")
        anchors = _byte_anchors(RULE_TEXT, row["t0_text"], expected_t1, expected_delay)
        _need(isinstance(row["byte_anchors"], Mapping)
              and set(row["byte_anchors"]) == ANCHOR_KEYS
              and all(type(v) is int or v is None for v in row["byte_anchors"].values())
              and dict(row["byte_anchors"]) == anchors, "BYTE_ANCHOR_MISMATCH")
        prefix = RULE_TEXT + row["t0_text"] + (expected_t1 or "") + (expected_delay or "")
        expected_hash = sha256(prefix.encode("utf-8")).hexdigest()
        _need(type(row["pre_query_sha256"]) is str
              and row["pre_query_sha256"] == expected_hash, "PREFIX_HASH_MISMATCH")
        gold = _compare_oracle(world, arm, readout, query)
        _need(type(row["declared_label"]) is str
              and row["declared_label"] == gold["final_label"], "LABEL_MISMATCH")
        _need(type(row["declared_label_id"]) is int
              and row["declared_label_id"] == gold["final_label_id"], "LABEL_ID_MISMATCH")
        _need(gold["final_label"] in ("REFUTE", "SUPPORT"), "UNEXPECTED_UNKNOWN_PRIMARY")
        observed[key] = prefix

    _need(set(observed) == set(product(ARMS, READOUTS, QUERIES)), "FACTOR_COVERAGE")
    for arm, readout in product(ARMS, READOUTS):
        _need(len({observed[arm, readout, q] for q in QUERIES}) == 1,
              "QUERY_CONTAMINATED_PREFIX")
    for query in QUERIES:
        _need(len({observed[arm, "PRE", query] for arm in ARMS}) == 1,
              "PRE_ARM_LEAKAGE")
    for arm in ARMS:
        p = observed[arm, "POST", "A"]
        delayed = observed[arm, "DELAYED", "A"]
        _need(delayed == p + DELAY_TEXT, "DELAY_DID_NOT_EXTEND_POST_PREFIX")
    return {
        "status": STATUS,
        "hypothetical_worlds_checked": 1,
        "declared_related_variants_checked": 27,
        "independently_sampled_roots": 0,
        "actual_prompt_corpus_created": "NO",
        "physical_pre_future_exclusion": "PASS_STATIC",
        "query_branch_isolation": "PASS_STATIC_BYTES_ONLY",
        "d1_label_consistency": "PASS_STATIC",
        "tokenizer_token_anchor_verification": "NOT_EVALUATED",
        "language_model_head_and_scoring": "NOT_EVALUATED",
        "temporal_model_behavior": "NOT_EVALUATED",
        "native_state_causal_mediation": "NOT_ESTABLISHED",
        "phase3b": "BLOCKED_UNCHANGED",
    }


READOUT_SPEC = {
    "mode": "CAUSAL_TERMINAL_TEACHER_FORCED_CANDIDATE",
    "label_order": ("REFUTE", "NOT_ENTITLED", "SUPPORT"),
    "candidate_continuations": "UNSET_NO_TOKENIZER",
    "score_aggregation": "UNSET_NO_TOKENIZER",
    "tokenizer_identity": "NOT_VERIFIED",
    "lm_head_identity": "NOT_VERIFIED",
    "trainable_probe": False,
    "evaluation_enabled": False,
}


def audit_readout_declaration(spec):
    """Fail closed on even a simulated tokenized or evaluated readout claim."""
    _need(isinstance(spec, Mapping) and set(spec) == set(READOUT_SPEC),
          "READOUT_SCHEMA")
    _need(all(type(spec[k]) is type(v) and spec[k] == v
              for k, v in READOUT_SPEC.items()), "READOUT_PREMATURE_PROMOTION")
    return {
        "status": "H2_P1_READOUT_PLAN_ONLY",
        "candidate_labels": CLASS_LABELS,
        "tokenizer_verified": "NO",
        "causal_lm_head_verified": "NO",
        "candidate_scores_comparable": "NOT_EVALUATED",
        "model_executed": "NO",
        "phase3b": "BLOCKED_UNCHANGED",
    }
