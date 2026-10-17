"""Independent, in-memory tests for the frozen RFR-D1 CPU-only contract."""

import ast
import copy
import itertools
import unittest
from pathlib import Path

from scripts import rfr_d1_deterministic_oracle_cpu as oracle


# These expectations are literal Boolean logic / separately frozen labels.
# Do not use production oracle functions to compute any expected outcome.
LABELS = {False: ("F", "REFUTE", 0), True: ("T", "SUPPORT", 2)}
TRUTH_TABLE = {
    ("T", "T"): "T", ("T", "F"): "F", ("T", "?"): "?",
    ("F", "T"): "F", ("F", "F"): "F", ("F", "?"): "F",
    ("?", "T"): "?", ("?", "F"): "F", ("?", "?"): "?",
}
LABEL_BY_TRUTH = {
    "F": ("REFUTE", 0), "?": ("NOT_ENTITLED", 1), "T": ("SUPPORT", 2),
}


def event(event_id, target, value, auth, time, **extra):
    result = {
        "event_id": event_id, "target": target,
        "asserted_value": value, "source_authenticated": auth,
        "event_time": time, "source_id": "src_" + event_id,
        "provenance_id": "prov_" + event_id,
    }
    result.update(extra)
    return result


def baseline(a=True, c=True, u=False):
    return [
        event("base_A", "A", a, True, 10),
        event("base_C", "C", c, True, 10),
        event("base_U", "U", u, True, 10),
    ]


def assign(case, family, group, split):
    return {
        "case_id": case, "case_family_id": family,
        "split_group_id": group, "split": split,
    }


class TestRfrD1PrimaryContract(unittest.TestCase):
    def test_exact_216_independent_fixture_checks(self):
        checked = 0
        for a0, c0, u0 in itertools.product((False, True), repeat=3):
            for arm in ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"):
                for readout in ("PRE", "POST", "DELAYED"):
                    a = (not a0) if arm == "VALID_FLIP" and readout != "PRE" else a0
                    expected = {"A": a, "B": a and c0, "U": u0, "C": c0}
                    for query in ("A", "B", "U"):
                        with self.subTest(a0=a0, c0=c0, u0=u0, arm=arm,
                                          readout=readout, query=query):
                            actual = oracle.primary_fixture(
                                a0, c0, u0, arm, readout, query
                            )
                            for prop, val in expected.items():
                                self.assertEqual(actual[prop], LABELS[val][0])
                            status, label, label_id = LABELS[expected[query]]
                            self.assertEqual(actual["query_target"], query)
                            self.assertEqual(actual["target_status"], status)
                            self.assertEqual(actual["final_label"], label)
                            self.assertEqual(actual["final_label_id"], label_id)
                            self.assertEqual(actual["oracle_version"], "RFR_D1_ORACLE_V1")
                            self.assertEqual(actual["derivation_rule_id"], "B_AND_A_C")
                        checked += 1
        self.assertEqual(checked, 216)

    def test_pre_sham_invalid_and_delayed_invariance(self):
        for a0, c0, u0 in itertools.product((False, True), repeat=3):
            for query in ("A", "B", "U"):
                pre = [oracle.primary_fixture(a0, c0, u0, arm, "PRE", query)
                       for arm in ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")]
                self.assertEqual(pre[0], pre[1])
                self.assertEqual(pre[1], pre[2])
                sham = oracle.primary_fixture(a0, c0, u0, "VALID_SHAM", "POST", query)
                invalid = oracle.primary_fixture(a0, c0, u0, "INVALID_FLIP", "POST", query)
                self.assertEqual(sham, invalid)
                for arm in ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"):
                    self.assertEqual(
                        oracle.primary_fixture(a0, c0, u0, arm, "POST", query),
                        oracle.primary_fixture(a0, c0, u0, arm, "DELAYED", query),
                    )

    def test_full_nine_row_kleene_truth_table(self):
        self.assertEqual(len(TRUTH_TABLE), 9)
        for (a, c), expected in TRUTH_TABLE.items():
            with self.subTest(a=a, c=c):
                self.assertEqual(oracle.strong_kleene_and(a, c), expected)
        for truth, expected_label in LABEL_BY_TRUTH.items():
            self.assertEqual(oracle.truth_to_label(truth), expected_label)
        for invalid in (None, True, False, "true", "", 0, 1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    oracle.strong_kleene_and(invalid, "T")
                with self.assertRaises(ValueError):
                    oracle.truth_to_label(invalid)


class TestRfrD1EventContract(unittest.TestCase):
    def test_time_order_invariance_without_last_mention_shortcut(self):
        original = baseline() + [
            event("stale", "A", False, True, 5),
            event("newer", "A", False, True, 20),
            event("untrusted_last", "A", True, False, 50),
        ]
        reference = oracle.evaluate_records(original, "B", "POST")
        self.assertEqual(reference["A"], "F")
        self.assertEqual(reference["B"], "F")
        for perm in itertools.permutations(original):
            self.assertEqual(oracle.evaluate_records(perm, "B", "POST"), reference)
        self.assertEqual(oracle.evaluate_records(original, "B", "PRE")["B"], "T")

    def test_pre_does_not_see_future_post_events(self):
        records = baseline() + [event("t20", "A", False, True, 20)]
        before = oracle.evaluate_records(records, "A", "PRE")
        after = oracle.evaluate_records(records, "A", "POST")
        self.assertEqual((before["A"], after["A"]), ("T", "F"))

    def test_untrusted_and_stale_cannot_override(self):
        for additional in (
            event("untrusted_new", "A", False, False, 50),
            event("stale_trusted", "A", False, True, 5),
        ):
            result = oracle.evaluate_records(baseline() + [additional], "A", "POST")
            self.assertEqual(result["A"], "T")

    def test_trusted_same_time_conflict_fails_in_any_presentation_order(self):
        records = baseline() + [
            event("eq1", "A", True, True, 20),
            event("eq2", "A", False, True, 20),
        ]
        for permutation in (records, list(reversed(records))):
            with self.assertRaisesRegex(ValueError, "SAME_TIME_AUTHORITY_CONFLICT"):
                oracle.evaluate_records(permutation, "A", "POST")
            # Validate entire supplied set, even if PRE would not expose t20.
            with self.assertRaisesRegex(ValueError, "SAME_TIME_AUTHORITY_CONFLICT"):
                oracle.evaluate_records(permutation, "A", "PRE")

    def test_identical_same_time_statements_are_idempotent(self):
        records = baseline() + [event("t20", "A", False, True, 20)]
        reference = oracle.evaluate_records(records, "B", "POST")
        duplicate = copy.deepcopy(records[-1])
        self.assertEqual(oracle.evaluate_records(records + [duplicate], "B", "POST"), reference)
        different_id = event("t20_other", "A", False, True, 20)
        self.assertEqual(oracle.evaluate_records(records + [different_id], "B", "POST"), reference)

    def test_duplicate_event_identity_mismatch_fails_closed(self):
        for key, value in (("asserted_value", False), ("source_id", "elsewhere"),
                           ("source_authenticated", False), ("event_time", 21)):
            initial = baseline()
            modified = dict(initial[0], **{key: value})
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "DUPLICATE_EVENT_ID_CONFLICT"):
                    oracle.evaluate_records(initial + [modified], "A", "POST")

    def test_missing_premises_and_explicit_false_separation(self):
        for known_a, known_c, expected_b in (
            (True, None, "?"), (False, None, "F"),
            (None, False, "F"), (None, None, "?"),
            (None, True, "?"),
        ):
            records = [event("u", "U", False, True, 10)]
            if known_a is not None:
                records.append(event("a", "A", known_a, True, 10))
            if known_c is not None:
                records.append(event("c", "C", known_c, True, 10))
            result = oracle.evaluate_records(records, "B", "POST")
            self.assertEqual(result["B"], expected_b)
            self.assertEqual(
                (result["final_label"], result["final_label_id"]),
                LABEL_BY_TRUTH[expected_b],
            )

    def test_c_u_b_updates_illegal(self):
        for target in ("C", "U", "B"):
            records = baseline() + [event("late_" + target, target, False, True, 20)]
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    oracle.evaluate_records(records, "A", "POST")
        with self.assertRaises(ValueError):
            oracle.evaluate_records(baseline() + [event("bad", "D", True, True, 20)], "A", "POST")

    def test_record_strict_types_and_identifiers(self):
        cases = (
            {"asserted_value": 1}, {"asserted_value": "True"},
            {"source_authenticated": 1}, {"event_time": True},
            {"event_time": 20.0}, {"event_time": "20"},
            {"event_id": " "}, {"source_id": ""},
            {"provenance_id": None}, {"target": "B"},
            {"oracle_version": "RFR_D1_ORACLE_V2"},
            {"derivation_rule_id": "B_OR_A_C"},
            {"unexpected_hidden_gold_label": "SUPPORT"},
        )
        for patch in cases:
            row = dict(event("a", "A", True, True, 10), **patch)
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    oracle.evaluate_records([row], "A", "POST")
        missing = event("a", "A", True, True, 10)
        del missing["source_id"]
        with self.assertRaises(ValueError):
            oracle.evaluate_records([missing], "A", "POST")
        with self.assertRaises(ValueError):
            oracle.evaluate_records([{**event("a", "A", True, True, 10),
                                      "oracle_version": None}], "A", "POST")
        with self.assertRaises(ValueError):
            oracle.evaluate_records([dict(event("a", "A", True, True, 10),
                                         derivation_rule_id="UNRECOGNIZED")], "A", "POST")

    def test_invalid_enums_and_arms(self):
        for query in ("C", "D", "", None, 2):
            with self.assertRaises(ValueError):
                oracle.evaluate_records(baseline(), query, "POST")
        for readout in ("PAST", None, 1):
            with self.assertRaises(ValueError):
                oracle.evaluate_records(baseline(), "A", readout)
        for arm in ("UNKNOWN", "MISSING_PREMISE", None):
            with self.assertRaises(ValueError):
                oracle.primary_fixture(True, True, False, arm, "POST", "A")
        for boolean in (0, 1, "T", None):
            with self.assertRaises(ValueError):
                oracle.primary_fixture(boolean, True, True, "VALID_FLIP", "POST", "A")
        for incorrect in (None, {}, "hi", ["event"]):
            with self.assertRaises(ValueError):
                oracle.evaluate_records(incorrect, "A", "POST")

    def test_distractor_neutrality_and_noncontamination(self):
        records = baseline() + [event("t20", "A", False, True, 20)]
        neutral = ({"distractor_id": "neutral001", "kind": "NEUTRAL"},)
        after = oracle.evaluate_records(records, "B", "POST")
        delayed = oracle.evaluate_records(records, "B", "DELAYED", neutral)
        self.assertEqual(after, delayed)
        bad = (
            {"distractor_id": "d", "kind": "EVIDENCE"},
            {"distractor_id": "d", "kind": "NEUTRAL", "target": "A"},
            {"distractor_id": "d", "kind": "NEUTRAL", "event_time": 30},
            {"distractor_id": "d", "kind": "NEUTRAL", "text": "A is false"},
            {"distractor_id": " ", "kind": "NEUTRAL"},
        )
        for item in bad:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    oracle.evaluate_records(records, "A", "DELAYED", [item])
        with self.assertRaises(ValueError):
            oracle.evaluate_records(records, "A", "DELAYED", [neutral[0], neutral[0]])
        with self.assertRaises(ValueError):
            oracle.evaluate_records(records, "A", "POST", neutral)

    def test_does_not_mutate_inputs_and_is_deterministic(self):
        records = baseline() + [event("new", "A", False, True, 20)]
        distractors = [{"distractor_id": "neutral", "kind": "NEUTRAL"}]
        saved_events, saved_distractors = copy.deepcopy(records), copy.deepcopy(distractors)
        first = oracle.evaluate_records(records, "B", "DELAYED", distractors)
        second = oracle.evaluate_records(records, "B", "DELAYED", distractors)
        self.assertEqual(first, second)
        self.assertEqual((records, distractors), (saved_events, saved_distractors))


class TestRfrD1SplitContract(unittest.TestCase):
    def test_family_group_split_positive_and_read_only(self):
        assignments = [
            assign("case1", "family1", "group1", "train"),
            assign("case1", "family1", "group1", "train"),
            assign("case2", "family1", "group1", "train"),
            assign("case3", "family2", "group2", "dev"),
            assign("case4", "family3", "group3", "test"),
        ]
        preserved = copy.deepcopy(assignments)
        self.assertIs(oracle.verify_split_isolation(assignments), True)
        self.assertEqual(assignments, preserved)

    def test_cross_split_or_reused_case_family_group_rejected(self):
        first = assign("case1", "family1", "group1", "train")
        failures = (
            assign("case1", "family1", "group1", "test"),
            assign("case1", "family2", "group2", "train"),
            assign("case2", "family1", "group2", "train"),
            assign("case2", "family1", "group1", "dev"),
            assign("case2", "family2", "group1", "train"),
            assign("case2", "family2", "group1", "test"),
        )
        for second in failures:
            with self.subTest(second=second):
                with self.assertRaises(ValueError):
                    oracle.verify_split_isolation([first, second])

    def test_malformed_and_empty_split_input_rejected(self):
        for assignment in (
            assign("", "family", "group", "train"),
            assign("case", None, "group", "train"),
            assign("case", "family", " ", "train"),
            assign("case", "family", "group", "validation"),
            {"case_id": "case", "case_family_id": "family", "split": "train"},
        ):
            with self.subTest(assignment=assignment):
                with self.assertRaises(ValueError):
                    oracle.verify_split_isolation([assignment])
        for data in ([], (), None, "test"):
            with self.assertRaises(ValueError):
                oracle.verify_split_isolation(data)


class TestRfrD1NoSideEffects(unittest.TestCase):
    def test_implementation_stdlib_only_and_no_write_or_external_execution_calls(self):
        path = Path(oracle.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed_imports = {"collections", "typing"}
        disallowed_call_names = {
            "open", "exec", "eval", "compile", "__import__", "torch", "requests",
            "system", "popen", "run", "write", "write_text", "write_bytes",
            "mkdir", "makedirs", "dump", "dumps", "save", "savez",
            "touch", "unlink", "remove", "replace", "subprocess", "cuda",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed_imports)
            if isinstance(node, ast.ImportFrom):
                self.assertIn((node.module or "").split(".")[0], allowed_imports)
                self.assertEqual(node.level, 0)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, disallowed_call_names)
                if isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, disallowed_call_names)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)


if __name__ == "__main__":
    unittest.main()
