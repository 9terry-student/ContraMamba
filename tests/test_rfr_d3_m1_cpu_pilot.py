"""Independent bounded M1 tests: real in-memory text, no corpus files or models."""

import ast
from collections import Counter, defaultdict
import copy
from hashlib import sha256
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import rfr_d3_m1_cpu_pilot as m1
from scripts import rfr_d1_deterministic_oracle_cpu as d1
from scripts import rfr_d3_cohort_contract_cpu as d3


class TestM1Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m1.generate_pilot()
        cls.by_root = {r["case_family_id"]: r for r in cls.roots}
        cls.by_cell = {(x["sealed_metadata"]["case_family_id"],
                        x["sealed_metadata"]["update_arm"],
                        x["sealed_metadata"]["readout_time"],
                        x["sealed_metadata"]["query_target"]): x for x in cls.rows}

    def test_exact_16_provisional_roots_and_8_logical_worlds(self):
        self.assertEqual(len(self.roots), 16)
        counts = Counter(r["surface_domain"] for r in self.roots)
        self.assertEqual(counts, {"REGISTRY_RECORD": 8, "INSTRUMENT_LOG": 8})
        for domain in counts:
            worlds = {tuple(r[f"root_initial_{k}"] for k in "ACU")
                      for r in self.roots if r["surface_domain"] == domain}
            self.assertEqual(worlds, set(itertools.product((False, True), repeat=3)))
        self.assertEqual(len({tuple(r[f"root_initial_{k}"] for k in "ACU")
                              for r in self.roots}), 8)

    def test_no_fake_random_seed_or_split(self):
        self.assertTrue(all("split" not in r for r in self.roots))
        for row in self.rows:
            self.assertNotIn("split", row["sealed_metadata"])
            self.assertNotIn("random_seed", row["sealed_metadata"])

    def test_exact_432_related_rows_and_27_cells_per_root(self):
        self.assertEqual(len(self.rows), 432)
        self.assertEqual(len(self.by_cell), 432)
        for root in self.roots:
            fid = root["case_family_id"]
            actual = {(a, t, q) for rf, a, t, q in self.by_cell if rf == fid}
            expected = set(itertools.product(m1.ARMS, m1.READOUTS, m1.QUERIES))
            self.assertEqual(actual, expected)

    def test_independent_truth_oracle_each_primary_cell(self):
        # Deliberately DO NOT use d1 to compute expectations.
        for root in self.roots:
            fid = root["case_family_id"]
            a0, c0, u0 = (root[f"root_initial_{k}"] for k in "ACU")
            for arm, readout, query in itertools.product(m1.ARMS, m1.READOUTS, m1.QUERIES):
                a = (not a0) if arm == "VALID_FLIP" and readout != "PRE" else a0
                value = {"A": a, "B": a and c0, "U": u0}[query]
                meta = self.by_cell[(fid, arm, readout, query)]["sealed_metadata"]
                self.assertEqual(meta["target_status"], "T" if value else "F")
                self.assertEqual(meta["final_label"], "SUPPORT" if value else "REFUTE")
                self.assertEqual(meta["final_label_id"], 2 if value else 0)

    def test_stratum_sensitive_B_transitions_and_C_false_block(self):
        for root in self.roots:
            fid = root["case_family_id"]
            before = self.by_cell[(fid, "VALID_FLIP", "PRE", "B")]["sealed_metadata"]["target_status"]
            after = self.by_cell[(fid, "VALID_FLIP", "POST", "B")]["sealed_metadata"]["target_status"]
            self.assertEqual(before != after, root["root_initial_C"])
            if not root["root_initial_C"]:
                self.assertEqual(after, "F")

    def test_A_flip_both_directions_and_U_retention(self):
        directions = set()
        for root in self.roots:
            fid = root["case_family_id"]
            pre = self.by_cell[(fid, "VALID_FLIP", "PRE", "A")]["sealed_metadata"]["target_status"]
            post = self.by_cell[(fid, "VALID_FLIP", "POST", "A")]["sealed_metadata"]["target_status"]
            directions.add((pre, post))
            self.assertNotEqual(pre, post)
            for arm in m1.ARMS:
                for readout in m1.READOUTS:
                    u = self.by_cell[(fid, arm, readout, "U")]["sealed_metadata"]["target_status"]
                    self.assertEqual(u, "T" if root["root_initial_U"] else "F")
        self.assertEqual(directions, {("T", "F"), ("F", "T")})

    def test_sham_invalid_no_revision_and_delayed_same_oracle(self):
        for root in self.roots:
            fid = root["case_family_id"]
            for q in m1.QUERIES:
                baseline = self.by_cell[(fid, "VALID_FLIP", "PRE", q)]["sealed_metadata"]["target_status"]
                for arm in ("VALID_SHAM", "INVALID_FLIP"):
                    self.assertEqual(
                        self.by_cell[(fid, arm, "POST", q)]["sealed_metadata"]["target_status"], baseline)
                for arm in m1.ARMS:
                    self.assertEqual(
                        self.by_cell[(fid, arm, "POST", q)]["sealed_metadata"]["target_status"],
                        self.by_cell[(fid, arm, "DELAYED", q)]["sealed_metadata"]["target_status"])

    def test_pre_physically_excludes_t1_and_same_bytes_across_arms(self):
        for root in self.roots:
            fid = root["case_family_id"]
            for q in m1.QUERIES:
                pre = [self.by_cell[(fid, a, "PRE", q)]["visible_input"] for a in m1.ARMS]
                self.assertEqual(len(set(pre)), 1)
                self.assertNotIn("logical-time=20", pre[0])
                self.assertNotIn("t1_A", pre[0])
            for arm in m1.ARMS:
                for q in m1.QUERIES:
                    post = self.by_cell[(fid, arm, "POST", q)]["visible_input"]
                    delayed = self.by_cell[(fid, arm, "DELAYED", q)]["visible_input"]
                    self.assertIn("logical-time=20", post)
                    self.assertNotEqual(post, delayed)
                    self.assertIn("Neutral note:", delayed)

    def test_matched_contrast_same_t1_content_auth_difference(self):
        for root in self.roots:
            fid = root["case_family_id"]
            for query in m1.QUERIES:
                triple = {a: m1.event_declaration(root, a, "POST", query) for a in m1.ARMS}
                d3.audit_matched_contrasts(root, triple)
                t1_flip = triple["VALID_FLIP"]["events"][-1]
                t1_invalid = triple["INVALID_FLIP"]["events"][-1]
                for key in ("event_id", "target", "asserted_value", "event_time"):
                    self.assertEqual(t1_flip[key], t1_invalid[key])
                self.assertTrue(t1_flip["source_authenticated"])
                self.assertFalse(t1_invalid["source_authenticated"])

    def test_event_timestamp_not_presentation_order(self):
        both_orders = set()
        for root in self.roots:
            d = m1.event_declaration(root, "VALID_FLIP", "POST", "A")
            events = {e["event_id"]: e for e in d["events"]}
            times = [events[key]["event_time"] for key in d["presentation_order"]]
            both_orders.add(times[0] == 20)
            self.assertEqual(sorted(times), [10, 10, 10, 20])
        self.assertEqual(both_orders, {False, True})

    def test_explicit_false_and_missing_evidence_are_not_confused(self):
        for row in self.rows:
            meta = row["sealed_metadata"]
            self.assertNotEqual(meta["target_status"], "?")
            self.assertNotEqual(meta["final_label"], "NOT_ENTITLED")
            if any(not e["asserted_value"] for e in meta["events"]):
                self.assertIn("explicitly FALSE", row["visible_input"])

    def test_two_domains_actual_text_not_merely_renamed_headers(self):
        by_domain = defaultdict(list)
        for row in self.rows:
            by_domain[row["sealed_metadata"]["surface_domain"]].append(row["visible_input"])
        self.assertTrue(all("Register record:" in s and "filing A" in s and "Instrument log:" not in s
                            for s in by_domain["REGISTRY_RECORD"]))
        self.assertTrue(all("Instrument log:" in s and "channel A" in s and "Register record:" not in s
                            for s in by_domain["INSTRUMENT_LOG"]))
        self.assertEqual(d3.audit_grammar_definitions(m1.grammar_definitions())["status"],
                         "PASS_STATIC_CONTRACT")

    def test_no_hidden_gold_metadata_in_visible_input(self):
        forbidden = ("case_family_id", "split_group_id", "final_label", "target_status",
                     "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP", "SUPPORT", "REFUTE",
                     "NOT_ENTITLED", "root_initial_")
        for row in self.rows:
            text = row["visible_input"]
            self.assertTrue(all(word not in text for word in forbidden))
            self.assertEqual(set(row), {"visible_input", "sealed_metadata"})

    def test_realized_hash_integrity_and_precise_duplicate_semantics(self):
        by_hash = defaultdict(set)
        for row in self.rows:
            meta = row["sealed_metadata"]
            self.assertEqual(meta["prompt_sha256"],
                             sha256(row["visible_input"].encode("utf-8")).hexdigest())
            by_hash[meta["prompt_sha256"]].add(meta["case_family_id"])
        self.assertEqual(len(by_hash), 336)
        self.assertTrue(all(len(fids) == 1 for fids in by_hash.values()))

    def test_bounded_pilot_report_is_not_science_promotion(self):
        report = m1.audit_pilot(self.roots, self.rows)
        self.assertEqual(report["status"], "PASS_BOUNDED_IN_MEMORY_PILOT")
        self.assertEqual(report["model_facing_rows_in_memory"], 432)
        self.assertEqual(report["unique_raw_prompt_bytes"], 336)
        self.assertEqual(report["logical_boolean_worlds"], 8)
        self.assertEqual(report["labels_by_query"], {
            "A": {"REFUTE": 72, "SUPPORT": 72},
            "B": {"REFUTE": 108, "SUPPORT": 36},
            "U": {"REFUTE": 72, "SUPPORT": 72},
        })
        for key in ("materialized_cohort_feasibility", "tokenizer_serialization",
                    "model_evaluation", "surface_near_duplicate_review"):
            self.assertEqual(report[key], "NOT_EVALUATED")
        self.assertEqual(report["PHASE3B"], "BLOCKED_UNCHANGED")


class TestM1Adversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m1.generate_pilot()

    def _tampered(self, index=0):
        rows = list(self.rows)
        rows[index] = copy.deepcopy(rows[index])
        return rows

    def test_missing_row_block(self):
        with self.assertRaisesRegex(ValueError, "INCORRECT_PILOT_COUNTS"):
            m1.audit_pilot(self.roots, self.rows[:-1])

    def test_duplicate_variant_block(self):
        rows = self._tampered(0)
        rows[0]["sealed_metadata"]["variant_id"] = self.rows[1]["sealed_metadata"]["variant_id"]
        with self.assertRaises(ValueError):
            m1.audit_pilot(self.roots, rows)

    def test_gold_tampering_block(self):
        rows = self._tampered(0)
        rows[0]["sealed_metadata"]["final_label"] = "NOT_ENTITLED"
        with self.assertRaisesRegex(ValueError, "FROZEN_ORACLE_MISMATCH"):
            m1.audit_pilot(self.roots, rows)

    def test_metadata_leak_with_rehashed_text_block(self):
        rows = self._tampered(0)
        rows[0]["visible_input"] += "\nfinal_label=SUPPORT"
        rows[0]["sealed_metadata"]["prompt_sha256"] = sha256(
            rows[0]["visible_input"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "SEALED_METADATA_EXPOSED"):
            m1.audit_pilot(self.roots, rows)

    def test_modifying_visible_text_and_rehashing_is_not_enough(self):
        rows = self._tampered(0)
        rows[0]["visible_input"] = rows[0]["visible_input"].replace("explicitly FALSE", "explicitly TRUE")
        rows[0]["sealed_metadata"]["prompt_sha256"] = sha256(
            rows[0]["visible_input"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "RENDERED_EVENT_OR_QUERY_MISMATCH"):
            m1.audit_pilot(self.roots, rows)

    def test_PRE_future_text_leak_fails(self):
        rows = self._tampered(0)
        rows[0]["visible_input"] += "\nRegister record: logical-time=20; hidden update"
        rows[0]["sealed_metadata"]["prompt_sha256"] = sha256(
            rows[0]["visible_input"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "PRE_FUTURE_LEAKAGE"):
            m1.audit_pilot(self.roots, rows)

    def test_hash_tamper_fails(self):
        rows = self._tampered(0)
        rows[0]["sealed_metadata"]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PILOT_HASH_MISMATCH"):
            m1.audit_pilot(self.roots, rows)

    def test_group_and_split_tamper_fail(self):
        for key, value, expected in (("split_group_id", "wrong", "SPLIT_GROUP_CONFLICT"),
                                     ("split", "test", "PILOT_SPLIT_NOT_AUTHORIZED")):
            rows = self._tampered(0)
            rows[0]["sealed_metadata"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, expected):
                m1.audit_pilot(self.roots, rows)

    def test_cross_family_prompt_collision_block(self):
        # Replacing a family prompt with another's identical bytes, and
        # recomputing the hash, must still fail at the render/source check.
        rows = self._tampered(27)
        rows[27]["visible_input"] = rows[0]["visible_input"]
        rows[27]["sealed_metadata"]["prompt_sha256"] = rows[0]["sealed_metadata"]["prompt_sha256"]
        with self.assertRaises(ValueError):
            m1.audit_pilot(self.roots, rows)

    def test_event_authority_flip_fails(self):
        rows = self._tampered(3)  # First family POST event by generation order
        target = rows[3]["sealed_metadata"]["events"][-1]
        target["source_authenticated"] = not target["source_authenticated"]
        with self.assertRaises(ValueError):
            m1.audit_pilot(self.roots, rows)

    def test_grammar_id_collision_fails_d3_gate(self):
        grammars = m1.grammar_definitions()
        grammars[1]["lexical_atoms"][0] = grammars[0]["lexical_atoms"][0]
        with self.assertRaisesRegex(ValueError, "CROSS_DOMAIN_LEXICON_COLLISION"):
            d3.audit_grammar_definitions(grammars)

    def test_no_file_io_rng_gpu_or_model_paths_in_module(self):
        source = Path(m1.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        import_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_roots.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                import_roots.add(node.module.split(".")[0])
        self.assertFalse(import_roots & {"os", "pathlib", "subprocess", "torch", "transformers",
                                         "random", "numpy", "socket", "requests", "pickle"})
        self.assertNotIn("if __name__ ==", source)  # No implicit CLI execution.
        with patch("builtins.open", side_effect=AssertionError("unexpected I/O")):
            rs, rows = m1.generate_pilot()
            self.assertEqual(m1.audit_pilot(rs, rows)["status"], "PASS_BOUNDED_IN_MEMORY_PILOT")


if __name__ == "__main__":
    unittest.main()
