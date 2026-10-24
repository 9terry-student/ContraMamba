"""New P/Q 16-item readout control: CPU, static, fake logits only."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import rfr_h2_basic_readout_eligibility_cpu as basic
from scripts import rfr_h2_p6_exploratory_grid_cpu as p6


class StaticGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = basic.build_grid()

    def test_exact_16_grid_and_balance(self):
        self.assertEqual(len(self.cases), 16)
        self.assertEqual(len({x["case_id"] for x in self.cases}), 16)
        self.assertEqual(len({x["prompt_sha256"] for x in self.cases}), 16)
        self.assertEqual(Counter(x["gold"] for x in self.cases),
                         Counter({"REFUTE": 8, "SUPPORT": 8}))
        self.assertEqual(set(x["order"] for x in self.cases), {"PQ", "QP"})
        self.assertEqual(set(x["query"] for x in self.cases), {"P", "Q"})

    def test_exact_text_bytes_and_answer_not_appended(self):
        s = basic.make_prompt(False, True, ("P", "Q"), "P")
        self.assertEqual(s, "Facts:\nP is FALSE.\nQ is TRUE.\nQuestion: Is P true?\nAnswer:")
        for x in self.cases:
            self.assertEqual(hashlib.sha256(x["prompt"].encode()).hexdigest(),
                             x["prompt_sha256"])
            self.assertTrue(x["prompt"].endswith("\nAnswer:"))
            self.assertNotIn("Answer: True", x["prompt"])
            self.assertNotIn("Answer: False", x["prompt"])

    def test_query_switch_and_record_order_are_independent(self):
        for p, q in itertools.product((False, True), repeat=2):
            for order in ("PQ", "QP"):
                rows = [c for c in self.cases if
                        (c["p"], c["q"], c["order"]) == (p, q, order)]
                self.assertEqual(len(rows), 2)
                self.assertEqual({x["gold"] for x in rows} == {"REFUTE", "SUPPORT"},
                                 p != q)
            for target in ("P", "Q"):
                rows = [c for c in self.cases if
                        (c["p"], c["q"], c["query"]) == (p, q, target)]
                self.assertEqual(len(rows), 2)
                self.assertEqual(len({r["gold"] for r in rows}), 1)

    def test_invalid_factors_fail_closed(self):
        for args in ((0, True, ("P", "Q"), "P"),
                     (False, True, ("P", "P"), "P"),
                     (False, True, ("P", "Q"), "B")):
            with self.assertRaisesRegex(basic.EligibilityError, "H2_BASIC_"):
                basic.make_prompt(*args)

    def test_template_mutations_fail_closed(self):
        with patch.object(basic, "QUESTION_TEMPLATE", "Question? {query} Answer:"):
            with self.assertRaisesRegex(basic.EligibilityError, "TEMPLATE_CHANGED"):
                basic.build_grid()
        with patch.object(basic.p3, "CONTINUATIONS", {
            "REFUTE": " No", "NOT_ENTITLED": " Unknown", "SUPPORT": " True"}):
            with self.assertRaisesRegex(basic.EligibilityError, "FROZEN_READOUT_CHANGED"):
                basic.build_grid()


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = basic.build_grid()

    def test_perfect_and_one_failure(self):
        good = {r["case_id"]: tuple(10. if label == r["gold"] else 0.
                                    for label in basic.LABELS) for r in self.cases}
        result = basic.parse_scores(self.cases, good)
        self.assertTrue(result["summary"]["strict_pass"])
        self.assertEqual(result["summary"]["correct"], 16)
        self.assertEqual(sum(x["both_correct"] for x in
                             result["summary"]["opposite_facts_query_switch_pairs"]), 4)
        bad = dict(good)
        bad[self.cases[0]["case_id"]] = (0., 0., 0.)
        result = basic.parse_scores(self.cases, bad)
        self.assertFalse(result["summary"]["strict_pass"])
        self.assertEqual(result["summary"]["correct"], 15)
        self.assertEqual(result["summary"]["status"], "INELIGIBLE_STOP_RFR_PATH")

    def test_constant_support_and_invalid_counts_as_failure(self):
        always = {r["case_id"]: (0., -2., 5.) for r in self.cases}
        result = basic.parse_scores(self.cases, always)
        self.assertEqual(result["summary"]["correct"], 8)
        self.assertEqual(result["summary"]["prediction_counts"], {"SUPPORT": 16})
        first = self.cases[0]["case_id"]
        always[first] = (float("nan"), 1., 2.)
        result = basic.parse_scores(self.cases, always)
        self.assertEqual(result["summary"]["prediction_counts"],
                         {"INVALID": 1, "SUPPORT": 15})
        self.assertFalse(result["summary"]["strict_pass"])

    def test_score_coverage_fail_closed(self):
        with self.assertRaisesRegex(basic.EligibilityError, "SCORES_COVERAGE"):
            basic.parse_scores(self.cases, {})


class GateTests(unittest.TestCase):
    def test_static_default_no_model(self):
        self.assertEqual(basic.main([]), 0)
        self.assertEqual(basic.main(["--approved-commit", "a" * 40]), 2)
        self.assertEqual(basic.main(["--execute"]), 2)

    def test_commit_guard_before_runtime_reads(self):
        with patch.object(p6, "authentication", side_effect=p6.ContractError("HEAD_MISMATCH")):
            with self.assertRaisesRegex(p6.ContractError, "HEAD_MISMATCH"):
                basic.authenticate(Path("."), Path("."), "a" * 40)

    def test_no_overwrite_checksum_and_complete_set(self):
        cases = basic.build_grid()
        good = {r["case_id"]: tuple(10. if label == r["gold"] else 0.
                                    for label in basic.LABELS) for r in cases}
        data = basic.parse_scores(cases, good)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dest = basic.save_run(root, "eligible16_test", {"head": "a" * 40}, data)
            names = {x.name for x in dest.iterdir()}
            self.assertEqual(names, {"manifest.json", "predictions.jsonl", "summary.json",
                                     "report.md", "SHA256SUMS.txt"})
            checks = (dest / "SHA256SUMS.txt").read_text().splitlines()
            self.assertEqual(len(checks), 4)
            for line in checks:
                sha, name = line.split("  ", 1)
                self.assertEqual(sha, basic.sha_file(dest / name))
            with self.assertRaisesRegex(basic.EligibilityError, "RUN_EXISTS"):
                basic.save_run(root, "eligible16_test", {"head": "a" * 40}, data)
            with self.assertRaisesRegex(basic.EligibilityError, "RUN_NAME"):
                basic.save_run(root, "../../escape", {"head": "a" * 40}, data)


if __name__ == "__main__":
    unittest.main()
