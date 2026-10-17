"""RFR-D2 adversarial metadata-only audits; never create corpus instances."""

import ast
import copy
import itertools
import unittest
from pathlib import Path

from scripts import rfr_d2_structural_feasibility_cpu as d2


# Independent test expectations: use literal Boolean logic and fixed D1 enums.
ARMS = ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")
READOUTS = ("PRE", "POST", "DELAYED")
QUERIES = ("A", "B", "U")


def roots(splits=(None,), holdout="NONE"):
    rows = []
    for split in splits:
        for i, (a, c, u) in enumerate(itertools.product((False, True), repeat=3)):
            suffix = (split or "unassigned") + "_" + str(i)
            domain = "domain" + str(int(a != u))
            row = {
                "case_family_id": "family_" + suffix,
                "split_group_id": "group_" + suffix,
                "surface_domain": domain,
                "renderer_family_id": "renderer_" + domain,
                "lexicon_namespace_id": "lex_" + suffix,
                "root_initial_A": a,
                "root_initial_C": c,
                "root_initial_U": u,
                "semantic_world_fingerprint": "world_" + suffix,
                "entity_tuple_fingerprint": "entity_" + suffix,
                "source_attestation_fingerprint": "attest_" + suffix,
                "template_family_id": "template_shared",
                "derivation_rule_id": "B_AND_A_C",
                "oracle_version": "RFR_D1_ORACLE_V1",
                "holdout_contract": holdout,
            }
            if split is not None:
                row["split"] = split
            rows.append(row)
    return rows


def variants(families, extra_realization=False):
    result = []
    for fam in families:
        for order in (("order1", "order2") if extra_realization else ("order1",)):
            for arm, readout, query in itertools.product(ARMS, READOUTS, QUERIES):
                item = {
                    "case_family_id": fam["case_family_id"],
                    "split_group_id": fam["split_group_id"],
                    "variant_id": fam["case_family_id"] + ":" + order + ":" + arm + ":" + readout + ":" + query,
                    "update_arm": arm,
                    "readout_time": readout,
                    "query_target": query,
                    "presentation_order_id": order,
                    "paraphrase_family_id": "paraphrase1",
                    "reverse_polarity": order == "order2",
                }
                if "split" in fam:
                    item["split"] = fam["split"]
                result.append(item)
    return result


def inventory(families):
    result = []
    for fam in families:
        triple = "".join("T" if fam[x] else "F" for x in (
            "root_initial_A", "root_initial_C", "root_initial_U"
        ))
        for arm, readout, query in itertools.product(ARMS, READOUTS, QUERIES):
            result.append({
                "case_family_id": fam["case_family_id"],
                "split_group_id": fam["split_group_id"],
                "split": fam.get("split", "UNASSIGNED"),
                "root_truth_stratum": triple,
                "surface_domain": fam["surface_domain"],
                "renderer_family_id": fam["renderer_family_id"],
                "template_family_id": fam["template_family_id"],
                "authentication_marker_family_id": "marker_shared",
                "source_name_family_id": "source_shared",
                "polarity_wording_family_id": "polarity_shared",
                "timestamp_presentation_pattern_id": "order_shared",
                "update_arm": arm,
                "readout_time": readout,
                "query_target": query,
                "count": 1,
            })
    return result


def plan(rows=None):
    result = {"plan_version": "RFR_D2_COUNTERBALANCE_PLAN_V1"}
    if rows is not None:
        result["inventory"] = rows
    return result


class TestFamilyRegistry(unittest.TestCase):
    def test_eight_strata_counts_and_no_claim_of_real_independence(self):
        families = roots()
        saved = copy.deepcopy(families)
        result = d2.audit_family_registry(families)
        self.assertEqual(result["status"], "PASS_PLAN_CHECK")
        self.assertEqual(result["declared_root_family_count"], 8)
        self.assertEqual(sum(result["declared_family_counts_by_split_stratum"]["UNASSIGNED"].values()), 8)
        self.assertEqual(len(result["declared_family_counts_by_split_stratum"]["UNASSIGNED"]), 8)
        self.assertEqual(result["CONTENT_INDEPENDENCE"], "UNVERIFIED")
        self.assertEqual(result["REALIZED_SPLIT_ISOLATION"], "NOT_EVALUATED")
        self.assertEqual(result["split_assignment_plan"], "NOT_EVALUATED")
        self.assertEqual(families, saved)

    def test_missing_and_unequal_truth_strata_fail_plan_check(self):
        families = roots()
        self.assertEqual(d2.audit_family_registry(families[:-1])["status"], "FAIL")
        extra = copy.deepcopy(families[0])
        for key in ("case_family_id", "split_group_id", "lexicon_namespace_id",
                    "semantic_world_fingerprint", "entity_tuple_fingerprint",
                    "source_attestation_fingerprint"):
            extra[key] = "another_" + extra[key]
        self.assertEqual(d2.audit_family_registry(families + [extra])["status"], "FAIL")

    def test_assigned_balanced_splits_and_renderer_warning(self):
        families = roots(("train", "test"))
        result = d2.audit_family_registry(families)
        self.assertEqual(result["status"], "PASS_PLAN_CHECK")
        self.assertEqual(result["split_assignment_plan"], "PASS_PLAN_CHECK")
        self.assertEqual(len(result["shared_renderer_across_assigned_splits"]), 2)
        self.assertEqual(result["REALIZED_SPLIT_ISOLATION"], "NOT_EVALUATED")

    def test_claimed_renderer_template_or_namespace_holdout_fails_closed(self):
        for contract in ("RENDERER", "TEMPLATE", "ALL"):
            with self.subTest(contract=contract):
                with self.assertRaisesRegex(ValueError, "CLAIMED_HOLDOUT_REUSE"):
                    d2.audit_family_registry(roots(("train", "dev"), contract))
        families = roots(("train", "dev"), "LEXICON")
        families[8]["lexicon_namespace_id"] = families[0]["lexicon_namespace_id"]
        with self.assertRaisesRegex(ValueError, "CLAIMED_HOLDOUT_REUSE"):
            d2.audit_family_registry(families)

    def test_identifier_and_fingerprint_collisions_fail_closed(self):
        for field in ("case_family_id", "split_group_id",
                      "semantic_world_fingerprint", "entity_tuple_fingerprint",
                      "source_attestation_fingerprint"):
            families = roots()
            families[1][field] = families[0][field]
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    d2.audit_family_registry(families)

    def test_missing_fields_types_versions_and_partial_assignment(self):
        keys = ("root_initial_A", "root_initial_C", "root_initial_U")
        for key in keys:
            families = roots()
            families[0][key] = 1
            with self.assertRaises(ValueError):
                d2.audit_family_registry(families)
        for key, bad in (("oracle_version", "RFR_D1_ORACLE_V2"),
                         ("derivation_rule_id", "B_OR_A_C"),
                         ("case_family_id", ""), ("holdout_contract", "bogus"),
                         ("unexpected", "anything")):
            families = roots()
            families[0][key] = bad
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    d2.audit_family_registry(families)
        families = roots()
        del families[0]["surface_domain"]
        with self.assertRaises(ValueError):
            d2.audit_family_registry(families)
        families = roots()
        families[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "PARTIAL_SPLIT_ASSIGNMENT"):
            d2.audit_family_registry(families)
        for invalid in ([], {}, None, "text"):
            with self.assertRaises(ValueError):
                d2.audit_family_registry(invalid)


class TestVariantPlan(unittest.TestCase):
    def test_complete_27_cells_is_plan_only_and_reversals_not_independent(self):
        families = roots()
        items = variants(families, extra_realization=True)
        original = copy.deepcopy(items)
        result = d2.audit_variant_plan(items, families)
        self.assertEqual(result["status"], "PASS_PLAN_CHECK")
        self.assertEqual(result["declared_variant_count"], 432)
        self.assertEqual(result["declared_realization_count"], 16)
        self.assertEqual(result["counted_independent_units"], 8)
        self.assertEqual(result["MATERIALIZATION_PROVENANCE"], "NOT_EVALUATED")
        self.assertEqual(items, original)

    def test_missing_cell_and_missing_family_returns_fail(self):
        families = roots()
        items = variants(families)
        self.assertEqual(d2.audit_variant_plan(items[:-1], families)["status"], "FAIL")
        removed_family = [v for v in items if v["case_family_id"] != families[0]["case_family_id"]]
        self.assertEqual(d2.audit_variant_plan(removed_family, families)["status"], "FAIL")
        self.assertEqual(d2.audit_variant_plan([], families)["status"], "NOT_EVALUATED")

    def test_duplicate_cell_or_variant_id_fails_closed(self):
        families = roots()
        items = variants(families)
        duplicate_id = copy.deepcopy(items[0])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_VARIANT_ID"):
            d2.audit_variant_plan(items + [duplicate_id], families)
        collision = copy.deepcopy(items[0])
        collision["variant_id"] = "not_same_id"
        with self.assertRaisesRegex(ValueError, "DUPLICATE_FACTORIAL_CELL"):
            d2.audit_variant_plan(items + [collision], families)

    def test_wrong_family_group_split_and_illegal_metadata(self):
        families = roots(("train",))
        items = variants(families)
        failures = (
            {"split_group_id": "another_group"},
            {"split": "test"},
            {"update_arm": "ANOTHER"},
            {"query_target": "C"},
            {"readout_time": "LATER"},
            {"reverse_polarity": 1},
            {"target_status": "T"},
            {"case_family_id": "not_registered"},
        )
        for patch in failures:
            candidate = copy.deepcopy(items[0])
            candidate.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    d2.audit_variant_plan([candidate] + items[1:], families)
        del items[0]["variant_id"]
        with self.assertRaises(ValueError):
            d2.audit_variant_plan(items, families)


class TestCounterbalancePlan(unittest.TestCase):
    def test_missing_inventory_not_evaluated(self):
        result = d2.audit_counterbalance_plan(plan(), roots())
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertEqual(result["G3_PLANNED_SURFACE"], "NOT_EVALUATED")
        self.assertTrue(all(s == "NOT_EVALUATED" for s in result["G4_SHORTCUT_ATTACKS"].values()))

    def test_full_planned_inventory_and_design_derived_counts(self):
        families = roots()
        rows = inventory(families)
        original = copy.deepcopy(rows)
        result = d2.audit_counterbalance_plan(plan(rows), families)
        self.assertEqual(result["status"], "PASS_PLAN_CHECK")
        self.assertEqual(result["declared_inventory_count"], 216)
        self.assertEqual(result["design_derived_target_answer_counts"], {
            "A": {"REFUTE": 36, "SUPPORT": 36},
            "B": {"REFUTE": 54, "SUPPORT": 18},
            "U": {"REFUTE": 36, "SUPPORT": 36},
        })
        self.assertEqual(result["ACTUAL_LEXICAL_LEAKAGE"], "NOT_EVALUATED")
        self.assertEqual(result["CONTENT_INDEPENDENCE"], "UNVERIFIED")
        self.assertEqual(result["TOKENIZER_SERIALIZATION"], "NOT_EVALUATED")
        self.assertEqual(result["MODEL_EVALUATION"], "NOT_EVALUATED")
        self.assertEqual(rows, original)

    def test_missing_family_factorial_cells_never_pass(self):
        families = roots()
        rows = inventory(families)
        self.assertEqual(d2.audit_counterbalance_plan(plan(rows[:-1]), families)["status"], "FAIL")
        with self.assertRaises(ValueError):
            d2.audit_counterbalance_plan(plan([]), families)

    def test_deterministic_label_cue_fail_even_when_all_27_cells_exist(self):
        families = roots()
        rows = inventory(families)
        # An invented cue family directly distinguishes each expected answer.
        for row in rows:
            fam = next(f for f in families if f["case_family_id"] == row["case_family_id"])
            a = fam["root_initial_A"]
            if row["update_arm"] == "VALID_FLIP" and row["readout_time"] != "PRE":
                a = not a
            expected = {"A": a, "B": a and fam["root_initial_C"],
                        "U": fam["root_initial_U"]}[row["query_target"]]
            row["source_name_family_id"] = "source_positive" if expected else "source_negative"
        result = d2.audit_counterbalance_plan(plan(rows), families)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("DETERMINISTIC_CUE_FOR_QUERY:source_name_family_id" in x
                            for x in result["issues"]))

    def test_joint_shortcut_rejected_when_each_single_cue_is_balanced(self):
        families = roots()
        rows = inventory(families)
        by_id = {f["case_family_id"]: f for f in families}
        for row in rows:
            root = by_id[row["case_family_id"]]
            a = root["root_initial_A"]
            u = root["root_initial_U"]
            row["source_name_family_id"] = "source" + str(int(a != u))
            row["polarity_wording_family_id"] = "word" + str(int(a))
        result = d2.audit_counterbalance_plan(plan(rows), families)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("DETERMINISTIC_JOINT_CUE_FOR_QUERY" in issue
                            for issue in result["issues"]))

    def test_invalid_counts_truth_fingerprint_and_no_gold_text(self):
        families = roots()
        original = inventory(families)
        for patch in ({"root_truth_stratum": "TTT"}, {"split_group_id": "wrong"},
                      {"surface_domain": "WRONG"}, {"count": 0}, {"count": True},
                      {"count": "1"}, {"final_label": "SUPPORT"},
                      {"update_arm": "NOPE"}, {"split": "train"}):
            rows = copy.deepcopy(original)
            rows[0].update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    d2.audit_counterbalance_plan(plan(rows), families)
        for invalid_plan in ({"plan_version": "another"},
                             {"plan_version": "RFR_D2_COUNTERBALANCE_PLAN_V1", "bad": 1},
                             None):
            with self.assertRaises(ValueError):
                d2.audit_counterbalance_plan(invalid_plan, families)


class TestStaticNoSideEffects(unittest.TestCase):
    def test_module_imports_and_calls_do_not_enable_execution_or_writes(self):
        tree = ast.parse(Path(d2.__file__).read_text(encoding="utf-8"))
        allowed = {"collections"}
        forbidden = {"open", "eval", "exec", "compile", "__import__", "write", "write_text",
                     "write_bytes", "save", "dump", "dumps", "mkdir", "touch", "unlink",
                     "remove", "replace", "system", "run", "popen", "subprocess", "cuda",
                     "read_text", "read_bytes"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertIn(alias.name.split(".")[0], allowed)
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                self.assertIn((node.module or "").split(".")[0], allowed)
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name)
                        else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                self.assertNotIn(name, forbidden)

    def test_no_false_full_feasibility_promotion_for_any_returned_report(self):
        families = roots()
        reports = (
            d2.audit_family_registry(families),
            d2.audit_variant_plan(variants(families), families),
            d2.audit_counterbalance_plan(plan(inventory(families)), families),
        )
        for report in reports:
            self.assertEqual(report["status"], "PASS_PLAN_CHECK")
            self.assertEqual(report["REALIZED_SPLIT_ISOLATION"], "NOT_EVALUATED")
            self.assertEqual(report["ACTUAL_LEXICAL_LEAKAGE"], "NOT_EVALUATED")
            self.assertEqual(report["TOKENIZER_SERIALIZATION"], "NOT_EVALUATED")
            self.assertEqual(report["MATERIALIZATION_PROVENANCE"], "NOT_EVALUATED")
            self.assertEqual(report["CONTENT_INDEPENDENCE"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
