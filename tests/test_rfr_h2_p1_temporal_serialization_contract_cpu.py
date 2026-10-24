"""Adversarial RFR-H2-P1 metadata-only tests. Fixtures are NOT sampled roots.

Exactly one hand-authored logical world at a time; its 27 related variants
are ephemeral unit-test declarations, never a corpus or evaluation population.
"""

import ast
import copy
import hashlib
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

from scripts import rfr_h2_p1_temporal_serialization_contract_cpu as p1


def world(a=True, c=True, u=False):
    return {"A": a, "C": c, "U": u}


def _lit(v):
    return "TRUE" if v else "FALSE"


def _t0(w):
    lines = ["Initial authenticated records (logical time 10):"]
    for k in ("A", "C", "U"):
        lines.append(f"{k}={_lit(w[k])}; authenticated=yes; source=BASE_{k}.")
    return "\n".join(lines) + "\n"


def _t1(w, arm):
    value = w["A"] if arm == "VALID_SHAM" else not w["A"]
    trusted = arm != "INVALID_FLIP"
    return (f"Update (logical time 20): A={_lit(value)}; "
            f"authenticated={'yes' if trusted else 'no'}; "
            f"source={'REV_A' if trusted else 'UNVERIFIED_A'}.\n")


def _independent_answer(w, arm, readout, query):
    a = w["A"]
    if readout != "PRE" and arm == "VALID_FLIP":
        a = not a
    values = {"A": a, "B": a and w["C"], "U": w["U"]}
    return ("SUPPORT", 2) if values[query] else ("REFUTE", 0)


def declarations(w):
    """Unit-test-only, provisional literal clauses; no production renderer."""
    entries = []
    for arm, readout, query in product(p1.ARMS, p1.READOUTS, p1.QUERIES):
        rule = p1.RULE_TEXT
        initial = _t0(w)
        correction = None if readout == "PRE" else _t1(w, arm)
        delay = p1.DELAY_TEXT if readout == "DELAYED" else None
        prefix = rule + initial + (correction or "") + (delay or "")
        rule_end = len(rule.encode("utf-8"))
        t0_end = rule_end + len(initial.encode("utf-8"))
        t1_end = t0_end + len(correction.encode("utf-8")) if correction is not None else None
        delay_end = t1_end + len(delay.encode("utf-8")) if delay is not None else None
        label, number = _independent_answer(w, arm, readout, query)
        entries.append({
            "arm": arm, "readout": readout, "query_target": query,
            "rule_text": rule, "t0_text": initial, "t1_text": correction,
            "delay_text": delay,
            "query_text": f"Query: Is {query} true under the current verified record?\nAnswer: ",
            "byte_anchors": {
                "rule_end": rule_end,
                "t0_end": t0_end,
                "t1_end": t1_end,
                "delay_end": delay_end,
                "query_start": len(prefix.encode("utf-8")),
            },
            "pre_query_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "declared_label": label, "declared_label_id": number,
        })
    return entries


def chosen(rows, *, arm="VALID_FLIP", readout="POST", query="A"):
    return next(row for row in rows if row["arm"] == arm and
                row["readout"] == readout and row["query_target"] == query)


class TestPositiveStaticContract(unittest.TestCase):
    def test_complete_case_and_status(self):
        w = world()
        rows = declarations(w)
        before = copy.deepcopy((w, rows))
        result = p1.audit_hypothetical_case(w, rows)
        self.assertEqual(result["status"], "H2_P1_STATIC_CONTRACT_PASS_NO_MODEL_OR_TOKENIZER")
        self.assertEqual(result["declared_related_variants_checked"], 27)
        self.assertEqual(result["independently_sampled_roots"], 0)
        self.assertEqual(result["tokenizer_token_anchor_verification"], "NOT_EVALUATED")
        self.assertEqual(result["native_state_causal_mediation"], "NOT_ESTABLISHED")
        self.assertEqual((w, rows), before)

    def test_all_eight_hypothetical_strata(self):
        for a, c, u in product((False, True), repeat=3):
            with self.subTest(world=(a, c, u)):
                result = p1.audit_hypothetical_case(world(a, c, u), declarations(world(a, c, u)))
                self.assertEqual(result["hypothetical_worlds_checked"], 1)

    def test_pre_bytes_same_across_arms(self):
        rows = declarations(world())
        self.assertEqual({chosen(rows, arm=arm, readout="PRE")["pre_query_sha256"]
                          for arm in p1.ARMS}.__len__(), 1)

    def test_query_branches_same_prefix_at_every_arm_readout(self):
        rows = declarations(world())
        for arm, readout in product(p1.ARMS, p1.READOUTS):
            hashes = {chosen(rows, arm=arm, readout=readout, query=q)["pre_query_sha256"]
                      for q in p1.QUERIES}
            self.assertEqual(len(hashes), 1)

    def test_conditional_b_gate_and_unauthorized_contradiction(self):
        for c in (True, False):
            w = world(False, c, True)
            rows = declarations(w)
            self.assertEqual(chosen(rows, readout="POST", query="B")["declared_label"],
                             "SUPPORT" if c else "REFUTE")
            self.assertEqual(chosen(rows, arm="INVALID_FLIP", readout="POST", query="B")["declared_label"],
                             "REFUTE")
            self.assertEqual(p1.audit_hypothetical_case(w, rows)["d1_label_consistency"], "PASS_STATIC")

    def test_never_claim_three_way_epistemic_coverage(self):
        w = world()
        labels = {row["declared_label"] for row in declarations(w)}
        self.assertEqual(labels, {"REFUTE", "SUPPORT"})
        self.assertNotIn("NOT_ENTITLED", labels)

    def test_readout_fails_closed_on_unverified_dependencies(self):
        report = p1.audit_readout_declaration(copy.deepcopy(p1.READOUT_SPEC))
        self.assertEqual(report["status"], "H2_P1_READOUT_PLAN_ONLY")
        self.assertEqual(report["causal_lm_head_verified"], "NO")
        self.assertEqual(report["candidate_scores_comparable"], "NOT_EVALUATED")


class TestAdversarialStaticContract(unittest.TestCase):
    def setUp(self):
        self.w = world()
        self.rows = declarations(self.w)

    def rejects(self):
        with self.assertRaisesRegex(ValueError, "H2_P1_"):
            p1.audit_hypothetical_case(self.w, self.rows)

    def test_world_missing_and_extra_keys(self):
        for corrupt in ({"A": True, "C": False}, {"A": True, "C": False, "U": True, "B": True}):
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, "WORLD_SCHEMA"):
                p1.audit_hypothetical_case(corrupt, self.rows)

    def test_world_bool_strictness(self):
        self.w["C"] = 1
        self.rejects()

    def test_missing_row(self):
        self.rows.pop()
        self.rejects()

    def test_duplicate_row(self):
        self.rows[-1] = copy.deepcopy(self.rows[0])
        self.rejects()

    def test_extra_sealed_label_or_root_id_key(self):
        self.rows[0]["case_family_id"] = "injected"
        self.rejects()

    def test_unknown_arm(self):
        self.rows[0]["arm"] = "VALID_NONPRIMARY"
        self.rejects()

    def test_unknown_readout(self):
        self.rows[0]["readout"] = "AFTER"
        self.rejects()

    def test_unknown_query(self):
        self.rows[0]["query_target"] = "C"
        self.rejects()

    def test_rule_change_or_gold_label_leak(self):
        chosen(self.rows)["rule_text"] += "Answer=SUPPORT.\n"
        self.rejects()

    def test_t0_value_change(self):
        chosen(self.rows)["t0_text"] = chosen(self.rows)["t0_text"].replace("A=TRUE", "A=FALSE")
        self.rejects()

    def test_pre_future_t1_included(self):
        chosen(self.rows, readout="PRE")["t1_text"] = _t1(self.w, "VALID_FLIP")
        self.rejects()

    def test_pre_future_arm_hint_in_initial_record(self):
        chosen(self.rows, readout="PRE")["t0_text"] += "future arm: VALID_FLIP\n"
        self.rejects()

    def test_post_t1_missing(self):
        chosen(self.rows)["t1_text"] = None
        self.rejects()

    def test_post_t1_repeats_baseline(self):
        chosen(self.rows)["t1_text"] += _t0(self.w)
        self.rejects()

    def test_flip_authentication_changed(self):
        chosen(self.rows)["t1_text"] = chosen(self.rows)["t1_text"].replace("authenticated=yes", "authenticated=no")
        self.rejects()

    def test_source_id_shortcut_mutation(self):
        chosen(self.rows)["t1_text"] = chosen(self.rows)["t1_text"].replace("REV_A", "GOLD_REFUTE")
        self.rejects()

    def test_delayed_interval_missing(self):
        chosen(self.rows, readout="DELAYED")["delay_text"] = None
        self.rejects()

    def test_delayed_interval_asserts_contradictory_fact(self):
        chosen(self.rows, readout="DELAYED")["delay_text"] += "A=TRUE.\n"
        self.rejects()

    def test_post_has_delay(self):
        chosen(self.rows)["delay_text"] = p1.DELAY_TEXT
        self.rejects()

    def test_query_contains_answer(self):
        chosen(self.rows)["query_text"] += "SUPPORT"
        self.rejects()

    def test_query_target_mismatch(self):
        chosen(self.rows)["query_text"] = "Query: Is B true under the current verified record?\nAnswer: "
        self.rejects()

    def test_query_metadata_in_prefix(self):
        chosen(self.rows)["t1_text"] += "query_target=A\n"
        self.rejects()

    def test_wrong_anchor_offset(self):
        chosen(self.rows)["byte_anchors"]["t0_end"] += 1
        self.rejects()

    def test_extra_anchor(self):
        chosen(self.rows)["byte_anchors"]["query_end"] = 999
        self.rejects()

    def test_bool_anchor_rejected(self):
        chosen(self.rows)["byte_anchors"]["rule_end"] = True
        self.rejects()

    def test_wrong_prefix_hash(self):
        chosen(self.rows)["pre_query_sha256"] = "0" * 64
        self.rejects()

    def test_unknown_gold_label(self):
        chosen(self.rows)["declared_label"] = "NOT_ENTITLED"
        self.rejects()

    def test_wrong_gold_id(self):
        chosen(self.rows)["declared_label_id"] = 2
        self.rejects()

    def test_bool_gold_id_not_a_numeric_id(self):
        chosen(self.rows)["declared_label_id"] = True
        self.rejects()

    def test_oracle_disagreement_fails_closed(self):
        from scripts import rfr_d1_deterministic_oracle_cpu as d1
        real = d1.primary_fixture
        def corrupted(*args, **kwargs):
            d = real(*args, **kwargs)
            d["B"] = "?" if d["B"] != "?" else "T"
            return d
        with patch.object(d1, "primary_fixture", side_effect=corrupted):
            self.rejects()

    def test_wrong_rows_type(self):
        self.rows = iter(self.rows)
        self.rejects()

    def test_untrusted_label_id_calibration_rejected(self):
        chosen(self.rows, arm="INVALID_FLIP")["declared_label_id"] = 1
        self.rejects()


class TestReadoutBoundaries(unittest.TestCase):
    def test_readout_extra_key(self):
        spec = dict(p1.READOUT_SPEC, trained_probe=True)
        with self.assertRaisesRegex(ValueError, "H2_P1_READOUT_SCHEMA"):
            p1.audit_readout_declaration(spec)

    def test_premature_readout_claims(self):
        changes = {
            "tokenizer_identity": "PINNED",
            "lm_head_identity": "VERIFIED",
            "candidate_continuations": ("False", "Unknown", "True"),
            "score_aggregation": "SUM",
            "trainable_probe": True,
            "evaluation_enabled": True,
            "label_order": ("SUPPORT", "NOT_ENTITLED", "REFUTE"),
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                spec = copy.deepcopy(p1.READOUT_SPEC)
                spec[key] = value
                with self.assertRaisesRegex(ValueError, "H2_P1_READOUT_PREMATURE_PROMOTION"):
                    p1.audit_readout_declaration(spec)

    def test_not_a_model_or_tokenizer_runner(self):
        source = Path(p1.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(n.name.split(".")[0] for n in node.names)
            if isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertFalse(imported & {"torch", "transformers", "tokenizers", "random", "numpy", "subprocess", "os"})
        self.assertNotIn("model.forward(", source)
        self.assertNotIn("open(", source)
        self.assertNotIn("write_bytes(", source)
        self.assertNotIn("write_text(", source)


if __name__ == "__main__":
    unittest.main()
