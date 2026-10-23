"""RFR-D3-M2 adversarial tests; no persistence, GPU, model or split assignment."""

import ast
from collections import Counter, defaultdict
import copy
from hashlib import sha256
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import rfr_d3_m2_shortcut_surface_cpu as m2
from scripts import rfr_d3_cohort_contract_cpu as d3


class TestM2Primary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m2.generate_pilot()

    def test_fixed_exact_scope(self):
        self.assertEqual(len(self.roots), 16)
        self.assertEqual(len(self.rows), 3456)
        strata = defaultdict(set)
        for root in self.roots:
            strata[root["surface_domain"]].add(tuple(root[f"root_initial_{k}"] for k in ("A", "C", "U")))
        self.assertEqual(set(strata), {"REGISTRY_RECORD", "INSTRUMENT_LOG"})
        self.assertTrue(all(len(v) == 8 for v in strata.values()))

    def test_m2_pilot_report_preserves_nonclaims(self):
        report = m2.audit_pilot(self.roots, self.rows)
        self.assertEqual(report["status"], "PASS_BOUNDED_M2_SURFACE_PILOT")
        self.assertEqual(report["related_rows_in_memory"], 3456)
        self.assertEqual(report["unique_raw_prompt_bytes"], 2496)
        self.assertEqual(report["cross_family_exact_prompt_duplicates"], 0)
        self.assertEqual(report["factorial_surface_realizations_per_root"], 8)
        for key in ("semantic_root_independence", "cross_family_near_duplicate_safety"):
            self.assertEqual(report[key], "NOT_ESTABLISHED")
        self.assertEqual(report["materialized_cohort_feasibility"], "NOT_EVALUATED")
        self.assertEqual(report["PHASE3B"], "BLOCKED_UNCHANGED")

    def test_independent_boolean_expected_labels_all_rows(self):
        by_id = {r["case_family_id"]: r for r in self.roots}
        for row in self.rows:
            m = row["sealed_metadata"]
            root = by_id[m["case_family_id"]]
            a, c, u = (root[f"root_initial_{k}"] for k in ("A", "C", "U"))
            if m["update_arm"] == "VALID_FLIP" and m["readout_time"] != "PRE":
                a = not a
            expected = {"A": a, "B": a and c, "U": u}[m["query_target"]]
            self.assertEqual(m["final_label"], "SUPPORT" if expected else "REFUTE")
            self.assertEqual(m["final_label_id"], 2 if expected else 0)

    def test_all_8_surface_crosses_each_factorial_cell(self):
        observed = defaultdict(set)
        for row in self.rows:
            meta = row["sealed_metadata"]
            key = tuple(meta[k] for k in (
                "case_family_id", "update_arm", "readout_time", "query_target"))
            observed[key].add(tuple(meta[k] for k in (
                "presentation_order_id", "query_position_id", "source_rotation_id")))
        self.assertEqual(len(observed), 16 * 27)
        expected = set(product(m2.ORDERS, m2.POSITIONS, m2.SOURCE_ROTATIONS))
        self.assertTrue(all(v == expected for v in observed.values()))

    def test_source_ids_not_root_id_or_fixed_auth_cue(self):
        labels = defaultdict(set)
        for row in self.rows:
            meta = row["sealed_metadata"]
            visible = row["visible_input"]
            self.assertNotIn(meta["case_family_id"], visible)
            self.assertNotIn("m1_", visible)
            if meta["readout_time"] != "PRE":
                t1 = meta["events"][-1]
                self.assertIn(t1["source_id"], m2.SOURCE_ALIASES)
                labels[(meta["case_family_id"], t1["source_id"])].add(
                    t1["source_authenticated"])
        self.assertEqual(len(labels), 32)
        self.assertTrue(all(x == {True, False} for x in labels.values()))

    def test_query_position_independent_of_baseline(self):
        by_stratum = defaultdict(set)
        for row in self.rows:
            meta = row["sealed_metadata"]
            root = next(r for r in self.roots if r["case_family_id"] == meta["case_family_id"])
            stratum = tuple(root[f"root_initial_{k}"] for k in ("A", "C", "U"))
            by_stratum[stratum].add(meta["query_position_id"])
        self.assertEqual(len(by_stratum), 8)
        self.assertTrue(all(x == set(m2.POSITIONS) for x in by_stratum.values()))

    def test_reversal_not_equal_to_logical_event_priority(self):
        for row in self.rows:
            m = row["sealed_metadata"]
            if m["readout_time"] == "PRE":
                continue
            time_order = [next(e["event_time"] for e in m["events"] if e["event_id"] == eid)
                          for eid in m["presentation_order"]]
            if m["presentation_order_id"] == "REVERSED":
                self.assertEqual(time_order[0], 20)
            else:
                self.assertEqual(time_order[-1], 20)

    def test_pre_no_t1_and_identical_across_arms_and_source_rotations(self):
        families = defaultdict(set)
        for row in self.rows:
            m = row["sealed_metadata"]
            if m["readout_time"] != "PRE":
                continue
            self.assertNotIn("logical-time=20", row["visible_input"])
            key = tuple(m[x] for x in ("case_family_id", "query_target",
                                        "presentation_order_id", "query_position_id"))
            families[key].add(row["visible_input"])
        self.assertEqual(len(families), 16 * 3 * 2 * 2)
        self.assertTrue(all(len(v) == 1 for v in families.values()))

    def test_post_delayed_same_gold_with_neutral_only(self):
        grouped = defaultdict(dict)
        for row in self.rows:
            m = row["sealed_metadata"]
            key = tuple(m[x] for x in ("case_family_id", "update_arm", "query_target",
                                        "presentation_order_id", "query_position_id",
                                        "source_rotation_id"))
            grouped[key][m["readout_time"]] = row
        for block in grouped.values():
            self.assertEqual(block["POST"]["sealed_metadata"]["final_label"],
                             block["DELAYED"]["sealed_metadata"]["final_label"])
            self.assertNotIn("Neutral note:", block["POST"]["visible_input"])
            self.assertIn("Neutral note:", block["DELAYED"]["visible_input"])

    def test_matched_valid_invalid_same_claim_different_attestation(self):
        root = self.roots[0]
        for order, pos, rotation in product(m2.ORDERS, m2.POSITIONS, m2.SOURCE_ROTATIONS):
            triple = {arm: m2.declare(root, arm, "POST", "B", order, pos, rotation)
                      for arm in m2.ARMS}
            self.assertEqual(d3.audit_matched_contrasts(root, triple)["status"],
                             "PASS_STATIC_CONTRACT")
            flip, invalid = (triple[a]["events"][-1] for a in ("VALID_FLIP", "INVALID_FLIP"))
            self.assertEqual(flip["asserted_value"], invalid["asserted_value"])
            self.assertEqual(flip["event_time"], invalid["event_time"])
            self.assertNotEqual(flip["source_id"], invalid["source_id"])

    def test_two_domains_have_different_actual_grammar(self):
        examples = {}
        for row in self.rows:
            m = row["sealed_metadata"]
            examples.setdefault(m["surface_domain"], row["visible_input"])
        self.assertIn("Synthetic register evidence:", examples["REGISTRY_RECORD"])
        self.assertIn("Synthetic instrument evidence:", examples["INSTRUMENT_LOG"])
        self.assertNotIn("Register record:", examples["INSTRUMENT_LOG"])
        self.assertNotIn("Instrument log:", examples["REGISTRY_RECORD"])

    def test_fixed_shortcut_diagnostics_are_conditional_and_all_imperfect(self):
        result = m2.diagnose_shortcuts(self.rows)
        self.assertEqual(result["status"], "BOUNDED_FIXED_SHORTCUT_DIAGNOSTIC_ONLY")
        self.assertEqual(len(result["fixed_heuristics"]), 7)
        self.assertEqual(len(result["conditional_confusions"]), 7 * 3 * 3 * 2)
        self.assertTrue(all(x["total"] == 3456 and x["incorrect"] > 0
                            for x in result["fixed_heuristics"].values()))
        self.assertEqual(result["general_shortcut_freedom"], "NOT_ESTABLISHED")
        self.assertEqual(result["trained_model_evaluation"], "NOT_EVALUATED")


class TestM2Adversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m2.generate_pilot()

    def mutate(self, index=0):
        rows = list(self.rows)
        rows[index] = copy.deepcopy(rows[index])
        return rows

    def test_bad_surface_factor_is_rejected(self):
        root = self.roots[0]
        for patch in ({"order": "PARITY"}, {"position": "CENTER"},
                      {"rotation": "AUTH_FOLLOWS_SOURCE"}, {"arm": "INVALID"}):
            options = {"arm": "VALID_FLIP", "readout": "POST", "query": "A",
                       "order": "REVERSED", "position": "QUERY_LAST", "rotation": "COPPER_VALID"}
            options.update(patch)
            with self.assertRaises(ValueError):
                m2.declare(root, **options)

    def test_invalid_counts_and_duplicate_variant_block(self):
        with self.assertRaisesRegex(ValueError, "M2_COUNT_MISMATCH"):
            m2.audit_pilot(self.roots, self.rows[:-1])
        rows = self.mutate(0)
        rows[0]["sealed_metadata"]["variant_id"] = self.rows[1]["sealed_metadata"]["variant_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_INCONSISTENT_VARIANT"):
            m2.audit_pilot(self.roots, rows)

    def test_gold_label_tampering_block(self):
        rows = self.mutate(0)
        rows[0]["sealed_metadata"]["final_label"] = "NOT_ENTITLED"
        with self.assertRaisesRegex(ValueError, "FROZEN_ORACLE_MISMATCH"):
            m2.audit_pilot(self.roots, rows)

    def test_prompt_rehashed_metadata_leak_rejected(self):
        rows = self.mutate(0)
        rows[0]["visible_input"] += "\ncase_family_id=m1_ledger_0"
        rows[0]["sealed_metadata"]["prompt_sha256"] = sha256(
            rows[0]["visible_input"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "VISIBLE_ID_OR_GOLD_LEAKAGE"):
            m2.audit_pilot(self.roots, rows)

    def test_rehashed_boolean_surface_tampering_rejected(self):
        rows = self.mutate(0)
        rows[0]["visible_input"] = rows[0]["visible_input"].replace(
            "explicitly FALSE", "explicitly TRUE", 1)
        rows[0]["sealed_metadata"]["prompt_sha256"] = sha256(
            rows[0]["visible_input"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "RENDER_OR_DECLARATION_MISMATCH"):
            m2.audit_pilot(self.roots, rows)

    def test_rehashed_source_swap_rejected(self):
        row_index = next(i for i, row in enumerate(self.rows) if
                         row["sealed_metadata"]["readout_time"] == "POST")
        rows = self.mutate(row_index)
        m = rows[row_index]["sealed_metadata"]
        event = m["events"][-1]
        event["source_id"] = "SRC_IMPOSTOR"
        m["canonical_event_sha256"] = sha256(m2._canonical(m["events"])).hexdigest()
        with self.assertRaisesRegex(ValueError, "RENDER_OR_DECLARATION_MISMATCH"):
            m2.audit_pilot(self.roots, rows)

    def test_query_position_forgery_fails(self):
        rows = self.mutate(0)
        rows[0]["sealed_metadata"]["query_position_id"] = "QUERY_LAST"
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_INCONSISTENT_VARIANT"):
            m2.audit_pilot(self.roots, rows)

    def test_no_split_or_group_tampering(self):
        for key, value in (("split", "test"), ("split_group_id", "another_group")):
            rows = self.mutate(0)
            rows[0]["sealed_metadata"][key] = value
            with self.assertRaisesRegex(ValueError, "SPLIT_GROUP_OR_ASSIGNMENT"):
                m2.audit_pilot(self.roots, rows)

    def test_hash_mutation_fails(self):
        rows = self.mutate(0)
        rows[0]["sealed_metadata"]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"):
            m2.audit_pilot(self.roots, rows)

    def test_no_file_io_random_or_scientific_runtime(self):
        tree = ast.parse(Path(m2.__file__).read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"os", "pathlib", "subprocess", "torch", "transformers",
                                    "random", "numpy", "socket", "requests", "pickle"})
        self.assertNotIn("if __name__ ==", Path(m2.__file__).read_text(encoding="utf-8"))
        with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            roots, rows = m2.generate_pilot()
            self.assertEqual(len(rows), 3456)
            self.assertEqual(m2.audit_pilot(roots, rows)["status"],
                             "PASS_BOUNDED_M2_SURFACE_PILOT")


if __name__ == "__main__":
    unittest.main()
