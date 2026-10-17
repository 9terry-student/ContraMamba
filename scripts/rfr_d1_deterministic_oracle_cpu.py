"""RFR-D1 deterministic, in-memory three-valued evidence oracle.

Scope: no corpus creation, I/O, model, tokenizer, network, GPU, or training.
The source-authentication protocol is synthetic, not a real-world trust claim.
"""

from collections.abc import Mapping, Sequence

ORACLE_VERSION = "RFR_D1_ORACLE_V1"
RULE_ID = "B_AND_A_C"
TRUTHS = frozenset(("T", "F", "?"))
BASE_TARGETS = frozenset(("A", "C", "U"))
QUERY_TARGETS = frozenset(("A", "B", "U"))
READOUTS = frozenset(("PRE", "POST", "DELAYED"))
ARMS = frozenset(("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"))
SPLITS = frozenset(("train", "dev", "test"))
EVENT_KEYS = frozenset((
    "event_id", "target", "asserted_value", "source_authenticated",
    "event_time", "source_id", "provenance_id",
))
OPTIONAL_EVENT_KEYS = frozenset(("oracle_version", "derivation_rule_id"))


def _require_identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"INVALID_IDENTIFIER:{name}")
    return value


def _require_enum(value, values, name):
    if not isinstance(value, str) or value not in values:
        raise ValueError(f"INVALID_{name}")
    return value


def _require_boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"INVALID_BOOLEAN:{name}")
    return value


def _require_version_and_rule(record):
    if "oracle_version" in record and record["oracle_version"] != ORACLE_VERSION:
        raise ValueError("UNKNOWN_ORACLE_VERSION")
    if "derivation_rule_id" in record and record["derivation_rule_id"] != RULE_ID:
        raise ValueError("UNKNOWN_DERIVATION_RULE")


def strong_kleene_and(a, c) -> str:
    """Strong-Kleene conjunction over exactly T, F and unknown ?."""
    _require_enum(a, TRUTHS, "TRUTH_A")
    _require_enum(c, TRUTHS, "TRUTH_C")
    if a == "F" or c == "F":
        return "F"
    if a == "T" and c == "T":
        return "T"
    return "?"


def truth_to_label(value) -> tuple[str, int]:
    """Map epistemic truth, not mere lack of positive evidence, to label."""
    _require_enum(value, TRUTHS, "TRUTH_LABEL")
    return {
        "F": ("REFUTE", 0),
        "?": ("NOT_ENTITLED", 1),
        "T": ("SUPPORT", 2),
    }[value]


def _validated_events(events):
    if not isinstance(events, (tuple, list)):
        raise ValueError("EVENTS_MUST_BE_SEQUENCE")
    by_id = {}
    accepted = []
    authenticated_at_time = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("EVENT_MUST_BE_MAPPING")
        keys = set(event)
        if not EVENT_KEYS <= keys or not keys <= EVENT_KEYS | OPTIONAL_EVENT_KEYS:
            raise ValueError("EVENT_SCHEMA_MISMATCH")
        for key in ("event_id", "source_id", "provenance_id"):
            _require_identifier(event[key], key)
        _require_enum(event["target"], BASE_TARGETS, "EVENT_TARGET")
        _require_boolean(event["asserted_value"], "asserted_value")
        _require_boolean(event["source_authenticated"], "source_authenticated")
        if type(event["event_time"]) is not int:
            raise ValueError("INVALID_EVENT_TIME")
        _require_version_and_rule(event)
        if event["target"] in ("C", "U") and event["event_time"] != 10:
            raise ValueError("UNAUTHORIZED_NON_A_UPDATE")
        payload = dict(event)
        prev = by_id.get(payload["event_id"])
        if prev is not None:
            if prev != payload:
                raise ValueError("DUPLICATE_EVENT_ID_CONFLICT")
            continue
        by_id[payload["event_id"]] = payload
        if payload["source_authenticated"]:
            locus = (payload["target"], payload["event_time"])
            value = payload["asserted_value"]
            if locus in authenticated_at_time and authenticated_at_time[locus] != value:
                raise ValueError("SAME_TIME_AUTHORITY_CONFLICT")
            authenticated_at_time[locus] = value
        accepted.append(payload)
    return accepted


def _validate_distractors(distractors, readout):
    if not isinstance(distractors, (tuple, list)):
        raise ValueError("DISTRACTORS_MUST_BE_SEQUENCE")
    if readout != "DELAYED" and distractors:
        raise ValueError("DISTRACTORS_ONLY_DELAYED")
    seen = set()
    for distractor in distractors:
        if not isinstance(distractor, Mapping):
            raise ValueError("DISTRACTOR_MUST_BE_MAPPING")
        if set(distractor) != {"distractor_id", "kind"}:
            raise ValueError("DISTRACTOR_NOT_NEUTRAL_METADATA")
        did = _require_identifier(distractor["distractor_id"], "distractor_id")
        if distractor["kind"] != "NEUTRAL" or did in seen:
            raise ValueError("DISTRACTOR_NOT_NEUTRAL_METADATA")
        seen.add(did)


def evaluate_records(events, query_target, readout, distractors=()) -> dict:
    """Validate every event, then compute state at PRE/POST/DELAYED.

    PRE observes only admitted time-10 baselines. POST/DELAYED observe all
    authenticated events, using logical time rather than presentation order.
    Non-A base variables cannot be updated beyond baseline time 10.
    """
    _require_enum(query_target, QUERY_TARGETS, "QUERY_TARGET")
    _require_enum(readout, READOUTS, "READOUT")
    all_events = _validated_events(events)
    _validate_distractors(distractors, readout)
    states = {}
    for target in ("A", "C", "U"):
        eligible = [
            event for event in all_events
            if event["target"] == target
            and event["source_authenticated"]
            and (readout != "PRE" or event["event_time"] == 10)
        ]
        if eligible:
            # Equal-time authenticated conflicting statements were rejected
            # while validating the full event set, including future records.
            newest_time = max(event["event_time"] for event in eligible)
            newest = next(event for event in eligible if event["event_time"] == newest_time)
            states[target] = "T" if newest["asserted_value"] else "F"
        else:
            states[target] = "?"
    states["B"] = strong_kleene_and(states["A"], states["C"])
    answer = states[query_target]
    label, label_id = truth_to_label(answer)
    return {
        "A": states["A"], "C": states["C"], "B": states["B"],
        "U": states["U"], "query_target": query_target,
        "target_status": answer, "final_label": label,
        "final_label_id": label_id, "oracle_version": ORACLE_VERSION,
        "derivation_rule_id": RULE_ID,
    }


def _event(event_id, target, asserted_value, authenticated, event_time):
    return {
        "event_id": event_id, "target": target,
        "asserted_value": asserted_value,
        "source_authenticated": authenticated,
        "event_time": event_time,
        "source_id": "synthetic_source_" + event_id,
        "provenance_id": "synthetic_provenance_" + event_id,
        "oracle_version": ORACLE_VERSION,
        "derivation_rule_id": RULE_ID,
    }


def primary_fixture(a0, c0, u0, arm, readout, query_target) -> dict:
    """In-memory exhaustive logical fixture, never a saved dataset row."""
    _require_boolean(a0, "a0")
    _require_boolean(c0, "c0")
    _require_boolean(u0, "u0")
    _require_enum(arm, ARMS, "ARM")
    _require_enum(readout, READOUTS, "READOUT")
    _require_enum(query_target, QUERY_TARGETS, "QUERY_TARGET")
    events = [
        _event("t0_A", "A", a0, True, 10),
        _event("t0_C", "C", c0, True, 10),
        _event("t0_U", "U", u0, True, 10),
        _event("t1_A", "A", a0 if arm == "VALID_SHAM" else not a0,
               arm != "INVALID_FLIP", 20),
    ]
    distractors = ({"distractor_id": "neutral_1", "kind": "NEUTRAL"},) if readout == "DELAYED" else ()
    return evaluate_records(events, query_target, readout, distractors)


def verify_split_isolation(assignments) -> bool:
    """Reject cross-split or contradictory case/family/group identities."""
    if not isinstance(assignments, (list, tuple)) or not assignments:
        raise ValueError("EMPTY_OR_INVALID_ASSIGNMENTS")
    by_case = {}
    by_family = {}
    by_group = {}
    required = {"case_id", "case_family_id", "split_group_id", "split"}
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or not required <= set(assignment):
            raise ValueError("ASSIGNMENT_SCHEMA_MISMATCH")
        case = _require_identifier(assignment["case_id"], "case_id")
        family = _require_identifier(assignment["case_family_id"], "case_family_id")
        group = _require_identifier(assignment["split_group_id"], "split_group_id")
        split = _require_enum(assignment["split"], SPLITS, "SPLIT")
        if case in by_case and by_case[case] != (family, group, split):
            raise ValueError("CASE_IDENTITY_OR_SPLIT_CONFLICT")
        if family in by_family and by_family[family] != (group, split):
            raise ValueError("FAMILY_GROUP_OR_SPLIT_CONFLICT")
        if group in by_group and by_group[group] != (family, split):
            raise ValueError("GROUP_FAMILY_OR_SPLIT_CONFLICT")
        by_case[case] = (family, group, split)
        by_family[family] = (group, split)
        by_group[group] = (family, split)
    return True
