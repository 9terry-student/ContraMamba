"""RFR-D3-M4 independent adversarial tests for root-frame feasibility gates."""

import ast
import copy
import importlib.util
import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import rfr_d3_m4_root_frame_feasibility_cpu as m4


# Fixture deliberately reconstructed without importing M1/M2/M3 generators.
def frame():
    roots, ancestry = [], []
    for domain, prefix in (("REGISTRY_RECORD", "ledger"),
                           ("INSTRUMENT_LOG", "apparatus")):
        for i, (a, c, u) in enumerate(itertools.product((False, True), repeat=3)):
            fid = f"m1_{prefix}_{i}"
            roots.append({
                "case_family_id": fid, "split_group_id": "m1_group_" + fid,
                "surface_domain": domain, "root_initial_A": a,
                "root_initial_C": c, "root_initial_U": u,
                "derivation_rule_id": "B_AND_A_C", "oracle_version": "RFR_D1_ORACLE_V1",
                "holdout_contract": "NONE",
            })
            ancestry.append({
                "case_family_id": fid, "split_group_id": "m1_group_" + fid,
                "semantic_world_id": "m1_world_id_" + fid,
                "root_substream_id": "m1_enumeration_only_" + fid,
                "root_kind": "PRIMARY_ROOT",
            })
    return roots, ancestry


def m3_report():
    return {
        "status": "M3_BOUNDED_AUDIT_COMPLETE_WITH_COHORT_BLOCKERS",
        "provisional_root_count": 16, "distinct_boolean_worlds": 8,
        "independent_cohort_feasibility": "NOT_ESTABLISHED",
        "structural_overlap": {
            "cross_family_equal_role_structure_row_pairs": 4608,
            "semantic_independence": "NOT_ESTABLISHED",
        },
        "controlled_near_neighbors": {
            "distinct_provisional_root_pairs_differing_only_U": 8,
            "matched_row_pairs_exactly_one_U_literal_apart": 1728,
        },
    }


def prospective():
    return {
        "purpose": "SURFACE_GENERALIZATION_ONLY",
        "root_unit": "INDEPENDENT_SYNTHETIC_MICROWORLD_DRAW",
        "derivation_rule_id": "B_AND_A_C", "oracle_version": "RFR_D1_ORACLE_V1",
        "surface_domains": ["INSTRUMENT_LOG", "REGISTRY_RECORD"],
        "planned_splits": ["test", "train", "dev"],
        "split_assignments": "NONE",
        "split_group_inheritance": "ALL_DESCENDANTS_SAME_ROOT_GROUP",
        "holdout_contract": "NONE",
        "root_draw_protocol": "PROSPECTIVE_SEPARATE_ROOT_SUBSTREAMS",
        "draw_components": [
            "ENTITY_CONFIGURATION", "BASE_RECORD_IDENTITIES",
            "SOURCE_ATTESTATION_IDENTITIES", "GRAMMAR_AND_TEMPLATE_CHOICES",
        ],
        "root_manifest_policy": "DECLARATION_ONLY_PIN_BEFORE_REALIZATION",
        "logical_generalization_claim": False,
        "realized_independence_evidence": "NONE",
    }


class TestExistingM2Frame(unittest.TestCase):
    def setUp(self):
        self.roots, self.lineage = frame()
        self.report = m3_report()

    def test_true_8_worlds_and_no_independence_promotion(self):
        r = m4.audit_existing_m2_frame(self.roots, self.lineage, self.report)
        self.assertEqual(r["status"], "M4_CURRENT_FRAME_INDEPENDENCE_BLOCKED")
        self.assertEqual(r["distinct_boolean_worlds"], 8)
        self.assertEqual(r["independent_logical_laws"], 1)
        self.assertEqual(r["independently_drawn_root_families_verified"], 0)
        self.assertEqual(r["domain_world_replications"], 8)
        self.assertEqual(r["cross_family_equal_role_structure_pairs"], 4608)
        self.assertEqual(r["grid_count_shortfall_if_one_root_per_cell"], 32)
        self.assertEqual(r["independent_surface_cohort"], "NOT_ESTABLISHED")
        self.assertEqual(r["PHASE3B"], "BLOCKED_UNCHANGED")

    def test_never_mutates_inputs(self):
        before = copy.deepcopy((self.roots, self.lineage, self.report))
        m4.audit_existing_m2_frame(self.roots, self.lineage, self.report)
        self.assertEqual((self.roots, self.lineage, self.report), before)

    def test_m3_claim_promotion_rejected(self):
        for field, value in (("status", "PASS"),
                             ("independent_cohort_feasibility", "PASS"),
                             ("provisional_root_count", 32),
                             ("distinct_boolean_worlds", 16)):
            bad = copy.deepcopy(self.report)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(self.roots, self.lineage, bad)

    def test_overlap_audit_evidence_cannot_be_removed(self):
        for group, key, replacement in (
            ("structural_overlap", "cross_family_equal_role_structure_row_pairs", 0),
            ("structural_overlap", "semantic_independence", "PASS"),
            ("controlled_near_neighbors", "matched_row_pairs_exactly_one_U_literal_apart", 0),
            ("controlled_near_neighbors", "distinct_provisional_root_pairs_differing_only_U", 0),
        ):
            bad = copy.deepcopy(self.report)
            bad[group][key] = replacement
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(self.roots, self.lineage, bad)

    def test_duplicate_or_missing_roots_rejected(self):
        bad = copy.deepcopy(self.roots)
        bad[-1]["case_family_id"] = bad[0]["case_family_id"]
        with self.assertRaises(ValueError):
            m4.audit_existing_m2_frame(bad, self.lineage, self.report)
        with self.assertRaises(ValueError):
            m4.audit_existing_m2_frame(self.roots[:-1], self.lineage, self.report)

    def test_duplicated_world_domain_or_nonboolean_rejected(self):
        for key, value in (("root_initial_A", self.roots[0]["root_initial_A"]),
                           ("surface_domain", "unknown"),
                           ("root_initial_C", "F")):
            bad = copy.deepcopy(self.roots)
            bad[1][key] = value
            if key == "root_initial_A":
                bad[1]["root_initial_A"] = not bad[1]["root_initial_A"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(bad, self.lineage, self.report)

    def test_preserve_frozen_rule_and_oracle(self):
        for key, val in (("derivation_rule_id", "B_OR_A_C"),
                         ("oracle_version", "RFR_D1_ORACLE_V2")):
            bad = copy.deepcopy(self.roots)
            bad[0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(bad, self.lineage, self.report)

    def test_no_unauthorized_split_or_renderer_holdout(self):
        for key, val in (("split", "train"), ("holdout_contract", "RENDERER")):
            bad = copy.deepcopy(self.roots)
            bad[0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(bad, self.lineage, self.report)

    def test_missing_or_corrupt_ancestry_rejected(self):
        for key, val in (("case_family_id", "unknown"),
                         ("split_group_id", "different"),
                         ("root_substream_id", "pretend_random_seed"),
                         ("root_kind", "TRUTH_REVERSAL")):
            bad = copy.deepcopy(self.lineage)
            bad[0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_existing_m2_frame(self.roots, bad, self.report)

    def test_distinct_ids_are_not_semantic_independence(self):
        bad = copy.deepcopy(self.roots)
        for index, root in enumerate(bad):
            root["case_family_id"] = "unique_" + str(index)
        with self.assertRaises(ValueError):
            m4.audit_existing_m2_frame(bad, self.lineage, self.report)


class TestProspectivePlan(unittest.TestCase):
    def test_valid_plan_does_not_establish_any_realized_evidence(self):
        plan = prospective()
        before = copy.deepcopy(plan)
        r = m4.audit_prospective_sampling_contract(plan)
        self.assertEqual(plan, before)
        self.assertEqual(r["status"], "M4_PROSPECTIVE_SURFACE_FRAME_PLAN_ONLY")
        self.assertEqual(r["minimum_root_slots_for_full_domain_stratum_split_cross"], 48)
        self.assertFalse(r["minimum_root_slots_are_statistical_power_analysis"])
        self.assertEqual(r["independently_drawn_roots"], "NOT_EVALUATED")
        self.assertEqual(r["materialized_cohort_feasibility"], "NOT_EVALUATED")
        self.assertEqual(r["renderer_holdout_across_three_splits"], "NOT_SUPPORTED")

    def test_three_way_renderer_holdout_is_false_with_two_domains(self):
        bad = prospective()
        bad["holdout_contract"] = "RENDERER"
        with self.assertRaisesRegex(ValueError, "M4_UNAUTHORIZED_PLAN_CLAIM"):
            m4.audit_prospective_sampling_contract(bad)

    def test_claiming_logical_independence_fails(self):
        for bad_value in (True, 1, "false"):
            bad = prospective()
            bad["logical_generalization_claim"] = bad_value
            with self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                m4.audit_prospective_sampling_contract(bad)

    def test_claiming_realized_roots_without_data_fails(self):
        for key, val in (("realized_independence_evidence", "VERIFIED"),
                         ("split_assignments", "COMPLETE"),
                         ("root_manifest_policy", "DONE")):
            bad = prospective()
            bad[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_prospective_sampling_contract(bad)

    def test_required_root_components_are_not_optional(self):
        bad = prospective()
        bad["draw_components"] = bad["draw_components"][:-1]
        with self.assertRaises(ValueError):
            m4.audit_prospective_sampling_contract(bad)

    def test_scope_is_closed_to_frozen_semantics(self):
        for key, value in (("purpose", "LOGICAL_GENERALIZATION"),
                           ("derivation_rule_id", "B_OR_A_C"),
                           ("oracle_version", "RFR_D1_ORACLE_V2"),
                           ("root_draw_protocol", "NAME_ROTATION_ONLY")):
            bad = prospective()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_prospective_sampling_contract(bad)

    def test_domain_set_and_split_set_must_be_exact(self):
        for key, val in (("surface_domains", ["REGISTRY_RECORD"]),
                         ("planned_splits", ["train", "dev", "dev"])):
            bad = prospective()
            bad[key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                m4.audit_prospective_sampling_contract(bad)

    def test_no_unexpected_fields_or_side_effects(self):
        bad = prospective()
        bad["actual_seed"] = 123
        with self.assertRaises(ValueError):
            m4.audit_prospective_sampling_contract(bad)
        with patch("builtins.open", side_effect=AssertionError("Unexpected I/O")):
            self.assertEqual(m4.audit_prospective_sampling_contract(prospective())["status"],
                             "M4_PROSPECTIVE_SURFACE_FRAME_PLAN_ONLY")


class TestFrozenRepositoryBridge(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("scripts.rfr_d3_m2_shortcut_surface_cpu") is not None,
        "Frozen M2/M3 modules not present in the isolated artifact sandbox",
    )
    def test_real_m2_roots_and_m3_report_not_promoted(self):
        from scripts import rfr_d3_m1_cpu_pilot as m1
        from scripts import rfr_d3_m2_shortcut_surface_cpu as m2
        from scripts import rfr_d3_m3_dependency_overlap_cpu as m3

        roots, lineage = m1.root_specs()
        verified_roots, rows = m2.generate_pilot()
        self.assertEqual(roots, verified_roots)
        upstream = m3.audit_m3(verified_roots, rows)
        report = m4.audit_existing_m2_frame(roots, lineage, upstream)
        self.assertEqual(report["status"], "M4_CURRENT_FRAME_INDEPENDENCE_BLOCKED")
        self.assertEqual(report["cross_family_equal_role_structure_pairs"], 4608)

    def test_m4_module_contains_no_random_io_model_or_sampling(self):
        source = Path(m4.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertEqual([alias.name for alias in node.names], [])
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, ("collections", "collections.abc", "itertools"))
            elif isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name) else
                        node.func.attr if isinstance(node.func, ast.Attribute) else "")
                self.assertNotIn(name, ("open", "write", "write_text", "write_bytes", "save",
                                        "dump", "dumps", "seed", "shuffle", "choice", "sample",
                                        "tokenize", "render", "run", "Popen", "system"))


if __name__ == "__main__":
    unittest.main()
