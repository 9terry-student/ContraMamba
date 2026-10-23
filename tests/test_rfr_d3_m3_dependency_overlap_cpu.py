"""RFR-D3-M3 bounded independent role-structure and Boolean-control tests."""

import ast
from collections import Counter, defaultdict
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import rfr_d3_m2_shortcut_surface_cpu as m2
from scripts import rfr_d3_m3_dependency_overlap_cpu as m3


class TestM3Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m2.generate_pilot()
        cls.result = m3.audit_m3(cls.roots, cls.rows)

    def test_m3_bounded_status_and_nonclaims(self):
        r = self.result
        self.assertEqual(r["status"], "M3_BOUNDED_AUDIT_COMPLETE_WITH_COHORT_BLOCKERS")
        self.assertEqual(r["provisional_root_count"], 16)
        self.assertEqual(r["distinct_boolean_worlds"], 8)
        self.assertEqual(r["independent_cohort_feasibility"], "NOT_ESTABLISHED")
        self.assertEqual(r["materialized_cohort_feasibility"], "NOT_EVALUATED")
        self.assertEqual(r["trained_model_performance"], "NOT_EVALUATED")
        self.assertEqual(r["native_state_causal_mediation"], "NOT_ESTABLISHED")
        self.assertEqual(r["PHASE3B"], "BLOCKED_UNCHANGED")

    def test_exact_1152_three_time_matched_comparisons(self):
        paired = self.result["paired_dependency"]
        self.assertEqual(paired["status"], "PASS_DETERMINISTIC_PAIRED_LOGIC_ONLY")
        self.assertEqual(paired["paired_pre_post_delayed_comparisons"], 1152)
        self.assertEqual(sum(paired["conditional_transition_counts"].values()), 1152)

    def test_C_false_B_always_unchanged_all_arms(self):
        counts = self.result["paired_dependency"]["conditional_transition_counts"]
        for arm in ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"):
            self.assertEqual(counts[f"B/C_F/{arm}/UNCHANGED"], 64)
            self.assertNotIn(f"B/C_F/{arm}/CHANGED", counts)

    def test_C_true_B_changes_only_under_valid_flip(self):
        counts = self.result["paired_dependency"]["conditional_transition_counts"]
        self.assertEqual(counts["B/C_T/VALID_FLIP/CHANGED"], 64)
        self.assertNotIn("B/C_T/VALID_FLIP/UNCHANGED", counts)
        for arm in ("VALID_SHAM", "INVALID_FLIP"):
            self.assertEqual(counts[f"B/C_T/{arm}/UNCHANGED"], 64)
            self.assertNotIn(f"B/C_T/{arm}/CHANGED", counts)

    def test_A_flips_only_if_valid_and_U_always_retained(self):
        counts = self.result["paired_dependency"]["conditional_transition_counts"]
        for condition in ("C_T", "C_F"):
            self.assertEqual(counts[f"A/{condition}/VALID_FLIP/CHANGED"], 64)
            for arm in ("VALID_SHAM", "INVALID_FLIP"):
                self.assertEqual(counts[f"A/{condition}/{arm}/UNCHANGED"], 64)
            for arm in m2.ARMS:
                self.assertEqual(counts[f"U/{condition}/{arm}/UNCHANGED"], 64)

    def test_structural_duplicate_signatures_not_text_duplicate_claim(self):
        x = self.result["structural_overlap"]
        self.assertEqual(x["status"], "STRUCTURAL_REPLICATION_DETECTED_NOT_INDEPENDENCE_PASS")
        self.assertEqual(x["cross_family_equal_role_structure_signatures"], 1248)
        self.assertEqual(x["cross_family_equal_role_structure_rows"], 3456)
        self.assertEqual(x["cross_family_equal_role_structure_row_pairs"], 4608)
        self.assertEqual(x["distinct_cross_domain_semantic_world_pairings"], 8)
        self.assertEqual(x["textual_near_duplicate_similarity"], "NOT_EVALUATED")
        self.assertEqual(x["semantic_independence"], "NOT_ESTABLISHED")
        self.assertEqual(m2.audit_pilot(self.roots, self.rows)["cross_family_exact_prompt_duplicates"], 0)

    def test_template_reuse_spans_eight_provisional_roots(self):
        x = self.result["structural_overlap"]
        self.assertEqual(x["template_skeleton_classes"], 72)
        self.assertEqual(x["max_families_sharing_template_skeleton"], 8)

    def test_one_U_literal_near_neighbor_counts(self):
        r = self.result["controlled_near_neighbors"]
        self.assertEqual(r["distinct_provisional_root_pairs_differing_only_U"], 8)
        self.assertEqual(r["matched_row_pairs_exactly_one_U_literal_apart"], 1728)
        self.assertEqual(r["matched_row_pairs_differing_in_gold_label"], 576)
        self.assertEqual(r["near_duplicate_global_search"], "NOT_EVALUATED")
        self.assertEqual(r["cross_split_near_duplicate_risk"], "NOT_EVALUATED_NO_SPLITS")

    def test_independent_C_omission_diagnostic_exact_conditional_counts(self):
        c = self.result["structural_overlap"]["c_omission_heuristic_by_query_and_C"]
        expected = {
            "A/C_F": (576,0), "A/C_T": (576,0),
            "B/C_F": (288,288), "B/C_T": (576,0),
            "U/C_F": (576,0), "U/C_T": (576,0),
        }
        self.assertEqual(set(c),set(expected))
        for name,(correct,incorrect) in expected.items():
            self.assertEqual((c[name]["correct"],c[name]["incorrect"]),(correct,incorrect))
        self.assertEqual(self.result["structural_overlap"]["c_omission_heuristic_pooled"],
                         {"correct":3168,"incorrect":288,"total":3456})

    def test_source_and_query_position_not_part_of_root_truth_index(self):
        signatures = [m3._visible_signature(row["visible_input"])
                      for row in self.rows]
        self.assertEqual(set(sig["domain"] for sig in signatures),
                         {"REGISTRY_RECORD","INSTRUMENT_LOG"})
        by_world = defaultdict(set)
        for row, sig in zip(self.rows,signatures):
            key = row["sealed_metadata"]["root_truth_stratum"]
            by_world[key].add(sig["position"])
        self.assertEqual(len(by_world),8)
        self.assertTrue(all(v == {"QUERY_FIRST","QUERY_LAST"} for v in by_world.values()))

    def test_no_input_mutation(self):
        sample = copy.deepcopy(self.rows[0])
        a = self.rows[0]
        m3._visible_signature(a["visible_input"])
        self.assertEqual(self.rows[0],sample)
        self.assertEqual(self.roots[0]["root_initial_A"],False)


class TestM3Adversarial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots, cls.rows = m2.generate_pilot()
        cls.by_roots, cls.idx = m3._index_rows(cls.roots,cls.rows)

    def test_visible_parser_rejects_malformed_domain(self):
        text = self.rows[0]["visible_input"].replace(
            "Synthetic register evidence:", "Unrecognized evidence:")
        with self.assertRaisesRegex(ValueError,"M3_VISIBLE_HEADER"):
            m3._visible_signature(text)

    def test_visible_parser_rejects_additional_unrecognized_line(self):
        text = self.rows[0]["visible_input"] + "\nGOLD_LABEL=REFUTE"
        with self.assertRaisesRegex(ValueError,"M3_UNPARSEABLE_VISIBLE_LINE"):
            m3._visible_signature(text)

    def test_visible_parser_rejects_malformed_query(self):
        text = self.rows[0]["visible_input"].replace("Question:","Answer:")
        with self.assertRaisesRegex(ValueError,"M3_UNPARSEABLE_VISIBLE_LINE"):
            m3._visible_signature(text)

    def test_visible_parser_rejects_non_neutral_distractor(self):
        row = next(x for x in self.rows if x["sealed_metadata"]["readout_time"] == "DELAYED")
        text = row["visible_input"].replace("decorative margin mark", "new fact A")
        with self.assertRaisesRegex(ValueError,"M3_NON_NEUTRAL_DELAY"):
            m3._visible_signature(text)

    def test_visible_parser_requires_exactly_one_question(self):
        text = self.rows[0]["visible_input"] + "\nQuestion: Is filing A true based on the authenticated records and rule?"
        with self.assertRaisesRegex(ValueError,"M3_STRUCTURE_COUNTS"):
            m3._visible_signature(text)

    def test_single_U_mask_changes_only_U_line(self):
        row = next(x for x in self.rows if x["sealed_metadata"]["surface_domain"] == "REGISTRY_RECORD")
        domain = "REGISTRY_RECORD"
        text = row["visible_input"]
        masked = m3._u_literal_mask(text, domain)
        self.assertEqual(sum("<U_BOOL>" in line for line in masked),1)
        self.assertEqual(len(masked),len(text.splitlines()))
        self.assertNotEqual(masked,tuple(text.splitlines()))

    def test_single_U_mask_rejects_removed_U_record(self):
        text = self.rows[0]["visible_input"]
        text = "\n".join(l for l in text.splitlines() if "unrelated entry U is" not in l)
        with self.assertRaisesRegex(ValueError,"M3_U_RECORD_COUNT"):
            m3._u_literal_mask(text, "REGISTRY_RECORD")

    def test_paired_logic_independently_detects_wrong_gold(self):
        row = self.idx[(self.roots[0]["case_family_id"],"VALID_FLIP","POST","B",
                        "CHRONOLOGICAL","QUERY_FIRST","COPPER_VALID")]
        corrupted = copy.deepcopy(row)
        label = corrupted["sealed_metadata"]["final_label"]
        corrupted["sealed_metadata"]["final_label"] = "REFUTE" if label == "SUPPORT" else "SUPPORT"
        index = dict(self.idx)
        meta = corrupted["sealed_metadata"]
        key = tuple(meta[k] for k in (
            "case_family_id","update_arm","readout_time","query_target",
            "presentation_order_id","query_position_id","source_rotation_id"))
        index[key] = corrupted
        with self.assertRaisesRegex(ValueError,"M3_PAIRED_BOOLEAN_MISMATCH"):
            m3._paired_dependency(self.by_roots,index)

    def test_index_detects_duplicate_variant(self):
        with self.assertRaisesRegex(ValueError,"M3_DUPLICATE_FACTOR"):
            m3._index_rows(self.roots,[self.rows[0],self.rows[0]])

    def test_m3_fail_closed_if_m2_guard_is_not_pass(self):
        with patch.object(m3.m2,"audit_pilot",return_value={"status":"FAIL"}):
            with self.assertRaisesRegex(ValueError,"M3_UPSTREAM_PILOT_BLOCKED"):
                m3.audit_m3(self.roots,self.rows)

    def test_rehashed_gold_tamper_is_not_allowed_by_upstream(self):
        changed = list(self.rows)
        changed[0] = copy.deepcopy(changed[0])
        changed[0]["sealed_metadata"]["final_label"] = "NOT_ENTITLED"
        with self.assertRaises(ValueError):
            m3.audit_m3(self.roots,changed)

    def test_no_new_generation_file_io_rng_or_model_paths(self):
        source = Path(m3.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"open","write_text","write_bytes","mkdir","dump","save",
                     "fit","train","infer","tokenize","render","generate_pilot",
                     "eval","exec","Popen","run","system"}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name = (node.func.id if isinstance(node.func,ast.Name) else
                        node.func.attr if isinstance(node.func,ast.Attribute) else "")
                self.assertNotIn(name,forbidden)
        self.assertNotIn("if __name__ ==",source)
        self.assertNotIn("random",source.lower())
        self.assertNotIn("torch",source)
        self.assertNotIn("transformers",source)


if __name__ == "__main__":
    unittest.main()
