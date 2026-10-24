"""P6 adversarial, model-free tests. Never loads tokenizer, checkpoint, or torch."""

from collections import Counter
import importlib.util
from pathlib import Path
import hashlib
import itertools
import json
import math
import tempfile
import unittest
from unittest.mock import patch

from scripts import rfr_h2_p6_exploratory_grid_cpu as h2
from scripts import rfr_h2_p1_temporal_serialization_contract_cpu as p1
from scripts import rfr_h2_p3_tokenizer_continuation_conformance_cpu as p3


def frozen_mock_oracle(world, arm, readout, query):
    answer = h2.expected_label(world["A"], world["C"], world["U"], arm, readout, query)
    return {"final_label": answer, "final_label_id": h2.LABELS.index(answer)}


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = h2.build_grid(oracle_fn=frozen_mock_oracle)

    def test_unique_168_and_pre_no_arm_leakage(self):
        self.assertEqual(len(self.cases), 168)
        self.assertEqual(len({c["case_id"] for c in self.cases}), 168)
        self.assertEqual(len({c["prompt_sha256"] for c in self.cases}), 168)
        for a, c, u in itertools.product((False, True), repeat=3):
            family = [x for x in self.cases if (x["a0"], x["c0"], x["u0"]) == (a, c, u)]
            self.assertEqual(len(family), 21)
            self.assertEqual(Counter(x["readout"] for x in family),
                             Counter({"PRE": 3, "POST": 9, "DELAYED": 9}))
            self.assertEqual(set(x["arm"] for x in family if x["readout"] == "PRE"),
                             {h2.PRE_ARM})
            self.assertTrue(all("Update (logical time 20):" not in x["prompt"]
                                for x in family if x["readout"] == "PRE"))

    def test_answer_seam_and_bytes(self):
        for x in self.cases:
            self.assertTrue(x["prompt"].endswith("\nAnswer:"))
            self.assertEqual(hashlib.sha256(x["prompt"].encode()).hexdigest(),
                             x["prompt_sha256"])
            self.assertTrue(x["prompt"] + " True" in
                            (x["prompt"] + p3.CONTINUATIONS["SUPPORT"],))
            self.assertNotIn("Answer: True", x["prompt"])

    def test_boolean_dependency_and_trust_controls(self):
        for case in self.cases:
            expected = h2.expected_label(case["a0"], case["c0"], case["u0"],
                                         case["arm"] if case["readout"] != "PRE"
                                         else "VALID_SHAM", case["readout"], case["query"])
            self.assertEqual(case["gold"], expected)
            if case["query"] == "U":
                self.assertEqual(case["gold"], "SUPPORT" if case["u0"] else "REFUTE")
            if case["query"] == "B" and not case["c0"]:
                self.assertEqual(case["gold"], "REFUTE")
        self.assertEqual(set(x["gold"] for x in self.cases), {"REFUTE", "SUPPORT"})

    @unittest.skipUnless(importlib.util.find_spec("scripts.rfr_d1_deterministic_oracle_cpu"),
                         "Full repository D1 source unavailable in isolated fixture")
    def test_frozen_d1_real_integration(self):
        actual = h2.build_grid()
        self.assertEqual([x["case_id"] for x in actual],
                         [x["case_id"] for x in self.cases])
        self.assertEqual(Counter(x["gold"] for x in actual),
                         Counter({"REFUTE": 98, "SUPPORT": 70}))

    def test_oracle_drift_is_blocked(self):
        def dishonest(world, arm, readout, query):
            r = frozen_mock_oracle(world, arm, readout, query)
            r["final_label_id"] = 1
            return r
        with self.assertRaisesRegex(h2.ContractError, "D1_ORACLE_CONTRADICTION"):
            h2.build_grid(oracle_fn=dishonest)

    def test_p1_source_mutations_block(self):
        with patch.object(p1, "DELAY_TEXT", "Fake event A=true.\n"):
            with self.assertRaisesRegex(h2.ContractError, "DELAY_DRIFT"):
                h2.build_grid(oracle_fn=frozen_mock_oracle)
        with patch.object(p3, "QUERY_SUFFIX_REASSIGNED", "Answer: "):
            with self.assertRaisesRegex(h2.ContractError, "SEAM_DRIFT"):
                h2.build_grid(oracle_fn=frozen_mock_oracle)
        with patch.dict(p3.CONTINUATIONS, {"SUPPORT": " Yes"}):
            with self.assertRaisesRegex(h2.ContractError, "CONTINUATION_DRIFT"):
                h2.build_grid(oracle_fn=frozen_mock_oracle)
        with patch.object(p1, "_t1_text", return_value="A=TRUE; authenticated=no; source=GOLD\n"):
            with self.assertRaisesRegex(h2.ContractError, "T1_OR_PRE_LEAK|PROMPT_COLLISION"):
                h2.build_grid(oracle_fn=frozen_mock_oracle)

    def test_duplicate_pre_nominal_is_identical(self):
        world = {"A": False, "C": True, "U": True}
        for q in h2.QUERIES:
            self.assertEqual(len({"".join(p3._segments(world, arm, "PRE")) +
                                  p1._query_text(q)[:-1] for arm in h2.ARMS}), 1)


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = h2.build_grid(oracle_fn=frozen_mock_oracle)

    def test_all_eight_baselines_and_fixed_outputs(self):
        rows = h2.baseline_rows(self.cases)
        self.assertEqual(len(rows), 168)
        self.assertEqual(set(rows[0]["predictions"]), set(h2.BASELINES))
        self.assertEqual({name: sum(row["predictions"][name]["correct"] for row in rows)
                          for name in h2.BASELINES},
                         {"ALWAYS_REFUTE": 98, "ALWAYS_NOT_ENTITLED": 0,
                          "ALWAYS_SUPPORT": 70, "T0_ONLY": 144,
                          "LATEST_ASSERTION": 144, "IGNORE_C": 154,
                          "QUERY_ONLY": 98, "AUTH_FLAG_ONLY": 82})
        for c in self.cases:
            self.assertEqual(h2.baseline_prediction(c, "T0_ONLY"),
                             h2.expected_label(c["a0"], c["c0"], c["u0"],
                                               "VALID_SHAM", c["readout"], c["query"]))
            if c["arm"] == "INVALID_FLIP" and c["query"] == "A":
                self.assertEqual(h2.baseline_prediction(c, "LATEST_ASSERTION"),
                                 "SUPPORT" if not c["a0"] else "REFUTE")
            if c["query"] == "B" and not c["c0"]:
                if c["readout"] == "PRE" or c["arm"] != "VALID_FLIP":
                    a = c["a0"]
                else:
                    a = not c["a0"]
                self.assertEqual(h2.baseline_prediction(c, "IGNORE_C"),
                                 "SUPPORT" if a else "REFUTE")

    def test_unknown_baseline_rejected(self):
        with self.assertRaisesRegex(h2.ContractError, "UNKNOWN_BASELINE"):
            h2.baseline_prediction(self.cases[0], "BEST_AFTER_SEEING_GOLD")
        with self.assertRaisesRegex(h2.ContractError, "BASELINE_INPUT_COVERAGE"):
            h2.baseline_rows(self.cases + self.cases[:1])


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = h2.build_grid(oracle_fn=frozen_mock_oracle)

    def test_score_mapping_and_margin(self):
        for gold, target in zip(h2.LABELS, range(3)):
            z = [0., 0., 0.]
            z[target] = 4.
            result = h2.score_triplet(tuple(z), gold=gold)
            self.assertEqual(result["prediction"], gold)
            self.assertTrue(result["correct"])
            self.assertEqual(result["gold_margin"], 4.)
            self.assertAlmostEqual(sum(result["candidate_softmax"]), 1.)

    def test_nonfinite_ties_and_wrong_shapes_fail_closed(self):
        for triple, reason in [((math.nan, 0., 2.), "NONFINITE"),
                               ((-math.inf, 0., 2.), "NONFINITE"),
                               ((math.inf, 0., 2.), "NONFINITE"),
                               ((1., 1., 0.), "TIED_MAX")]:
            value = h2.score_triplet(triple, gold="REFUTE")
            self.assertFalse(value["correct"])
            self.assertFalse(value["valid"])
            self.assertEqual(value["reason"], reason)
        for bad in [(), (1.,), (1., 2., 3., 4.), (True, 2., 3.), ("bad", 0., 1.)]:
            with self.assertRaisesRegex(h2.ContractError, "SCORE_SCHEMA"):
                h2.score_triplet(bad, gold="REFUTE")

    def test_joint_success_and_failed_one_case(self):
        perfect = {c["case_id"]: tuple(4. if label == c["gold"] else 0.
                                        for label in h2.LABELS) for c in self.cases}
        report = h2.score_report(self.cases, perfect)
        self.assertEqual(report["summary"]["CASE_joint_count"], 8)
        self.assertEqual(report["summary"]["correct"], 168)
        self.assertEqual(len(report["predictions"]), 168)
        self.assertEqual(len(report["baseline_predictions"]), 168)
        self.assertTrue(all(v["all_correct"] for v in
                            report["summary"]["conditional_diagnostics"].values()))
        perturbed = dict(perfect)
        item = self.cases[0]
        perturbed[item["case_id"]] = (0., 0., 0.)
        failed = h2.score_report(self.cases, perturbed)
        self.assertEqual(failed["summary"]["CASE_joint_count"], 7)
        self.assertEqual(failed["summary"]["correct"], 167)
        self.assertEqual(failed["predictions"][0]["reason"], "TIED_MAX")
        self.assertEqual(len(failed["summary"]["failures"]), 1)

    def test_score_coverage_mismatch_rejected(self):
        with self.assertRaisesRegex(h2.ContractError, "SCORE_COVERAGE"):
            h2.score_report(self.cases, {})


class GateAndArtifactTests(unittest.TestCase):
    def test_no_run_main_disallows_execution_flags(self):
        with patch.object(h2, "build_grid", return_value=[{}] * 168), \
             patch.object(h2, "baseline_rows", return_value=[
                 {"predictions": {name: {"correct": True} for name in h2.BASELINES}}
             ] * 168):
            self.assertEqual(h2.main([]), 0)
            self.assertEqual(h2.main(["--approved-commit", "a" * 40]), 2)
            self.assertEqual(h2.main(["--execute"]), 2)

    def test_commit_mismatch_blocks_before_files(self):
        with patch.object(h2, "git_output", side_effect=[str(Path("/repo").resolve()), h2.EXPECTED_BRANCH, "a" * 40]):
            with self.assertRaisesRegex(h2.ContractError, "HEAD_MISMATCH"):
                h2.authentication(Path("/repo"), Path("/snapshot"), "b" * 40)
        with self.assertRaisesRegex(h2.ContractError, "COMMIT_SCHEMA"):
            h2.authentication(Path("/repo"), Path("/snapshot"), "invalid")

    def test_artifact_checksums_nonoverwrite(self):
        with tempfile.TemporaryDirectory() as td:
            report = {
                "predictions": [{"case_id": "one", "prediction": "REFUTE"}],
                "baseline_predictions": [{"case_id": "one"}],
                "summary": {"CASE_joint_count": 0, "n_unique_contexts": 168,
                            "strata": {"A0C0U0": {"PRE_joint": False,
                                                   "POST_joint": False,
                                                   "DELAYED_joint": False,
                                                   "CASE_joint": False}},
                            "conditional_diagnostics": {},
                            "baseline_summary": {},
                            "correct": 0, "failures": []},
            }
            dest = h2.save_evidence(Path(td), "h2p6_run", {"head": "a" * 40}, report)
            self.assertTrue(dest.is_dir())
            self.assertEqual(len(list(dest.iterdir())), 6)
            checks = (dest / "SHA256SUMS.txt").read_text().splitlines()
            for line in checks:
                digest, name = line.split("  ", 1)
                self.assertEqual(digest, h2.sha_file(dest / name))
            with self.assertRaisesRegex(h2.ContractError, "RUN_DIRECTORY_EXISTS"):
                h2.save_evidence(Path(td), "h2p6_run", {"head": "a" * 40}, report)

    def test_run_name_path_injection_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(h2.ContractError, "RUN_NAME"):
                h2.save_evidence(Path(td), "../../bad", {"head": "a" * 40}, {})


if __name__ == "__main__":
    unittest.main()
