"""Independent, adversarial D3 *metadata-only* contract tests.

No generated dataset, rendered prompts, random sampling, or model output.
The integration test uses the real frozen D2 validator when it is present.
"""

import ast
import copy
import importlib.util
import itertools
import unittest
from pathlib import Path

try:
    from scripts import rfr_d3_cohort_contract_cpu as d3
except ModuleNotFoundError:
    import rfr_d3_cohort_contract_cpu as d3


def grammar(domain, prefix):
    return {
        "surface_domain": domain,
        "renderer_family_id": f"renderer_{prefix}",
        "lexical_atoms": [f"{prefix}_{i}" for i in range(6)],
        "template_family_ids": [f"{prefix}_template1", f"{prefix}_template2"],
        "positive_form_id": f"{prefix}_positive",
        "explicit_negative_form_id": f"{prefix}_verified_false",
        "absence_form_id": f"{prefix}_missing",
        "authenticated_marker_id": f"{prefix}_certified",
        "unauthenticated_marker_id": f"{prefix}_uncertified",
        "rule_conjunction_form_ids": [f"{prefix}_and1", f"{prefix}_and2"],
        "presentation_order_ids": [f"{prefix}_chronological", f"{prefix}_reverse"],
        "query_position_ids": [f"{prefix}_query_first", f"{prefix}_query_last"],
        "neutral_distractor_family_ids": [f"{prefix}_neutral"],
    }


def grammars():
    return [grammar("REGISTRY_RECORD", "ledger"),
            grammar("INSTRUMENT_LOG", "apparatus")]


def roots():
    rows = []
    for i, (a, c, u) in enumerate(itertools.product((False, True), repeat=3)):
        domain = "REGISTRY_RECORD" if i % 2 == 0 else "INSTRUMENT_LOG"
        renderer = "renderer_ledger" if i % 2 == 0 else "renderer_apparatus"
        rows.append({
            "case_family_id": f"f{i}", "split_group_id": f"g{i}",
            "surface_domain": domain, "renderer_family_id": renderer,
            "lexicon_namespace_id": f"lex{i}",
            "root_initial_A": a, "root_initial_C": c, "root_initial_U": u,
            "semantic_world_fingerprint": f"world_fp{i}",
            "entity_tuple_fingerprint": f"entity_fp{i}",
            "source_attestation_fingerprint": f"attest_fp{i}",
            "template_family_id": f"templ_{i}",
            "derivation_rule_id": "B_AND_A_C",
            "oracle_version": "RFR_D1_ORACLE_V1", "holdout_contract": "NONE",
        })
    return rows


def lineage(families=None):
    return [{"case_family_id": row["case_family_id"],
             "split_group_id": row["split_group_id"],
             "semantic_world_id": f"world{i}",
             "root_substream_id": f"substream{i}", "root_kind": "PRIMARY_ROOT"}
            for i, row in enumerate(families if families is not None else roots())]


def event(eid, target, value, auth, time):
    return {"event_id": eid, "target": target, "asserted_value": value,
            "source_authenticated": auth, "event_time": time,
            "source_id": f"src_{eid}", "provenance_id": f"prov_{eid}",
            "oracle_version": "RFR_D1_ORACLE_V1",
            "derivation_rule_id": "B_AND_A_C"}


def declared(root, arm="VALID_FLIP", readout="POST", query="B"):
    records = [event(f"e_{target}", target, root[f"root_initial_{target}"], True, 10)
               for target in ("A", "C", "U")]
    if readout != "PRE":
        changed = root["root_initial_A"] if arm == "VALID_SHAM" else not root["root_initial_A"]
        t1 = event("e_t1", "A", changed, arm != "INVALID_FLIP", 20)
        if arm == "INVALID_FLIP":
            t1["source_id"] = "src_alt_e_t1"
            t1["provenance_id"] = "prov_alt_e_t1"
        records.append(t1)
    return {
        "query_target": query,
        "events": records,
        "presentation_order": [e["event_id"] for e in reversed(records)],
        "visible_sections": ["rule", "records", "authentication", "logical_time", "query"]
                            + (["neutral_delay"] if readout == "DELAYED" else []),
        "sealed_field_names": ["case_family_id", "split_group_id", "update_arm",
                               "readout_time", "query_target", "target_status",
                               "final_label", "final_label_id"],
        "distractors": ([{"distractor_id": "d1", "kind": "NEUTRAL"}]
                        if readout == "DELAYED" else []),
    }


def variants(families):
    return [
        {"case_family_id": root["case_family_id"],
         "split_group_id": root["split_group_id"],
         "variant_id": root["case_family_id"] + ":" + arm + ":" + readout + ":" + query,
         "update_arm": arm, "readout_time": readout,
         "query_target": query, "presentation_order_id": "reversed",
         "paraphrase_family_id": "base"}
        for root in families
        for arm, readout, query in itertools.product(
            ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"),
            ("PRE", "POST", "DELAYED"), ("A", "B", "U"))
    ]


class TestGrammar(unittest.TestCase):
    def test_two_domains_pass_and_inputs_unchanged(self):
        items = grammars()
        before = copy.deepcopy(items)
        result = d3.audit_grammar_definitions(items)
        self.assertEqual(result["status"], "PASS_STATIC_CONTRACT")
        self.assertEqual(result["TEXT_RENDERING"], "NOT_EVALUATED")
        self.assertEqual(result["MATERIALIZED_COHORT_FEASIBILITY"], "NOT_EVALUATED")
        self.assertEqual(items, before)

    def test_domain_count_missing_or_unknown_fails(self):
        for candidate in ([], grammars()[:1], grammars() + [grammars()[0]],
                          [grammar("REGISTRY_RECORD", "ledger"),
                           grammar("SOMETHING_ELSE", "apparatus")]):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    d3.audit_grammar_definitions(candidate)

    def test_lexicon_and_template_collisions_fail(self):
        for field in ("lexical_atoms", "template_family_ids", "rule_conjunction_form_ids"):
            items = grammars()
            items[1][field][0] = items[0][field][0]
            with self.subTest(field=field), self.assertRaises(ValueError):
                d3.audit_grammar_definitions(items)

    def test_polarity_distinct_from_unknown_and_authentication(self):
        for key in ("absence_form_id", "explicit_negative_form_id",
                    "unauthenticated_marker_id"):
            items = grammars()
            items[0][key] = items[0]["positive_form_id"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                d3.audit_grammar_definitions(items)

    def test_no_gold_label_vocab_or_accidental_metadata(self):
        items = grammars()
        items[1]["lexical_atoms"][0] = "final_label"
        with self.assertRaisesRegex(ValueError, "GOLD_LABEL_LEXICON"):
            d3.audit_grammar_definitions(items)
        items = grammars()
        items[0]["target_status"] = "REFUTE"
        with self.assertRaisesRegex(ValueError, "SCHEMA_MISMATCH"):
            d3.audit_grammar_definitions(items)

    def test_cross_domain_auth_surface_id_collision(self):
        items = grammars()
        items[1]["authenticated_marker_id"] = items[0]["authenticated_marker_id"]
        with self.assertRaisesRegex(ValueError, "CROSS_DOMAIN_AUTH_POLARITY"):
            d3.audit_grammar_definitions(items)

    def test_required_counterbalance_declarations(self):
        for key in ("rule_conjunction_form_ids", "presentation_order_ids", "query_position_ids"):
            items = grammars()
            items[0][key] = ["one"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                d3.audit_grammar_definitions(items)


class TestRootLineage(unittest.TestCase):
    def test_lineage_pass_without_independence_claim(self):
        families = roots()
        ancestry = lineage(families)
        prior = copy.deepcopy(families), copy.deepcopy(ancestry)
        result = d3.audit_root_lineage(families, ancestry)
        self.assertEqual(result["status"], "PASS_STATIC_CONTRACT")
        self.assertEqual(result["declared_root_count"], 8)
        self.assertEqual(result["CONTENT_INDEPENDENCE"], "UNVERIFIED")
        self.assertEqual((families, ancestry), prior)

    def test_dependent_root_and_cross_family_substream_fail(self):
        for key, value in (("root_kind", "TRUTH_REVERSAL"),
                           ("root_substream_id", "substream0"),
                           ("semantic_world_id", "world0")):
            items = lineage()
            items[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d3.audit_root_lineage(roots(), items)

    def test_split_group_identity_and_missing_lineage_fail(self):
        items = lineage()
        items[1]["split_group_id"] = "g0"
        with self.assertRaisesRegex(ValueError, "LINEAGE_GROUP_CONFLICT"):
            d3.audit_root_lineage(roots(), items)
        with self.assertRaisesRegex(ValueError, "MISSING_ROOT_LINEAGE"):
            d3.audit_root_lineage(roots(), lineage()[:-1])

    def test_strict_root_boolean_and_version(self):
        for patch in ({"root_initial_A": 1}, {"root_initial_U": "?"},
                      {"derivation_rule_id": "B_OR_A_C"},
                      {"oracle_version": "RFR_D1_ORACLE_V2"}):
            rs = roots()
            rs[0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                d3.audit_root_lineage(rs, lineage())


class TestEvents(unittest.TestCase):
    def test_all_primary_arms_readouts_and_queries_metadata_only(self):
        for root in roots():
            for arm, readout, query in itertools.product(
                ("VALID_FLIP", "VALID_SHAM", "INVALID_FLIP"),
                ("PRE", "POST", "DELAYED"), ("A", "B", "U")):
                item = declared(root, arm, readout, query)
                saved = copy.deepcopy(item)
                result = d3.audit_event_declaration(root, arm, readout, item)
                self.assertEqual(result["status"], "PASS_STATIC_CONTRACT")
                self.assertEqual(result["ACTUAL_PRE_INPUT_BYTES"], "NOT_EVALUATED")
                self.assertEqual(item, saved)

    def test_pre_physically_excludes_t1_even_for_sham(self):
        root = roots()[0]
        item = declared(root, "VALID_SHAM", "PRE")
        item["events"].append(event("e_t1", "A", root["root_initial_A"], True, 20))
        item["presentation_order"].append("e_t1")
        with self.assertRaisesRegex(ValueError, "PRE_FUTURE"):
            d3.audit_event_declaration(root, "VALID_SHAM", "PRE", item)

    def test_post_requires_exact_t1_and_preserves_authentication(self):
        root = roots()[0]
        for arm, patch in (("VALID_FLIP", {"asserted_value": root["root_initial_A"]}),
                           ("INVALID_FLIP", {"source_authenticated": True}),
                           ("VALID_SHAM", {"source_authenticated": False}),
                           ("VALID_FLIP", {"event_time": 10})):
            item = declared(root, arm)
            item["events"][-1].update(patch)
            with self.subTest(arm=arm, patch=patch), self.assertRaises(ValueError):
                d3.audit_event_declaration(root, arm, "POST", item)

    def test_explicit_false_not_unknown(self):
        root = roots()[0]
        item = declared(root)
        item["events"][0]["asserted_value"] = "?"
        with self.assertRaisesRegex(ValueError, "INVALID_EVENT_BOOLEAN"):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)
        item = declared(root)
        del item["events"][0]
        item["presentation_order"].remove("e_A")
        with self.assertRaises(ValueError):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)

    def test_event_identity_provenance_timestamp_and_target(self):
        root = roots()[0]
        patches = ({"target": "B"}, {"event_time": True}, {"source_id": ""},
                   {"derivation_rule_id": "B_OR_A_C"}, {"oracle_version": "v2"},
                   {"extra_gold": 1}, {"event_id": "e_A"})
        for patch in patches:
            item = declared(root)
            item["events"][-1].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)

    def test_permutation_and_query_must_be_valid(self):
        root = roots()[0]
        item = declared(root)
        item["presentation_order"] = ["e_A", "e_C", "e_U", "e_U"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_ENTRY"):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)
        item = declared(root)
        item["presentation_order"][-1] = "missing_from_events"
        with self.assertRaisesRegex(ValueError, "PRESENTATION_ORDER"):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)
        item = declared(root)
        item["query_target"] = "C"
        with self.assertRaises(ValueError):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)

    def test_visible_fields_and_sealed_fields_are_separate(self):
        root = roots()[0]
        item = declared(root)
        item["visible_sections"].append("final_label")
        with self.assertRaisesRegex(ValueError, "VISIBLE_METADATA"):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)
        item = declared(root)
        item["sealed_field_names"].remove("final_label_id")
        with self.assertRaisesRegex(ValueError, "SEALED_METADATA"):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)

    def test_delayed_requires_only_neutral_distractors(self):
        root = roots()[0]
        item = declared(root, readout="DELAYED")
        item["distractors"][0]["kind"] = "CHANGED_A"
        with self.assertRaises(ValueError):
            d3.audit_event_declaration(root, "VALID_FLIP", "DELAYED", item)
        item = declared(root, readout="DELAYED")
        item["distractors"] = []
        with self.assertRaises(ValueError):
            d3.audit_event_declaration(root, "VALID_FLIP", "DELAYED", item)
        item = declared(root, readout="POST")
        item["distractors"] = [{"distractor_id": "d1", "kind": "NEUTRAL"}]
        with self.assertRaises(ValueError):
            d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)

    def test_t0_values_must_match_root_both_polarities(self):
        for root in (roots()[0], roots()[-1]):
            item = declared(root)
            item["events"][0]["asserted_value"] = not root["root_initial_A"]
            with self.assertRaisesRegex(ValueError, "ROOT_TRUTH_CONFLICT"):
                d3.audit_event_declaration(root, "VALID_FLIP", "POST", item)


class TestMatchedContrasts(unittest.TestCase):
    def test_full_triple_passes(self):
        root = roots()[0]
        triple = {arm: declared(root, arm) for arm in (
            "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")}
        prior = copy.deepcopy(triple)
        report = d3.audit_matched_contrasts(root, triple)
        self.assertEqual(report["status"], "PASS_STATIC_CONTRACT")
        self.assertEqual(report["ACTUAL_AUTHENTICATION_EVIDENCE"], "NOT_EVALUATED")
        self.assertEqual(triple, prior)

    def test_invalid_provenance_cannot_impersonate_valid_provenance(self):
        root = roots()[0]
        triple = {arm: declared(root, arm) for arm in (
            "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")}
        for key in ("source_id", "provenance_id"):
            triple["INVALID_FLIP"]["events"][-1][key] = triple["VALID_FLIP"]["events"][-1][key]
        with self.assertRaisesRegex(ValueError, "PROVENANCE_NOT_DISTINCT"):
            d3.audit_matched_contrasts(root, triple)

    def test_contrast_query_and_baseline_cannot_change(self):
        root = roots()[0]
        triple = {arm: declared(root, arm) for arm in (
            "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")}
        triple["INVALID_FLIP"]["query_target"] = "A"
        with self.assertRaisesRegex(ValueError, "MATCHED_QUERY"):
            d3.audit_matched_contrasts(root, triple)
        triple = {arm: declared(root, arm) for arm in (
            "VALID_FLIP", "VALID_SHAM", "INVALID_FLIP")}
        triple["INVALID_FLIP"]["events"][0]["source_id"] = "another_verified_source"
        with self.assertRaisesRegex(ValueError, "MATCHED_BASELINE"):
            d3.audit_matched_contrasts(root, triple)


class TestD2Bridge(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scripts") is not None, "Frozen D2 module not in local artifact sandbox")
    def test_exact_d2_plan_delegation_when_repo_available(self):
        rs = roots()
        report = d3.audit_d2_plan_bridge(rs, variants(rs), grammars(), lineage(rs))
        self.assertEqual(report["status"], "PASS_STATIC_CONTRACT")
        self.assertEqual(report["D2_VARIANT_PLAN"], "PASS_PLAN_CHECK")
        self.assertEqual(report["counted_independent_units"], 8)
        self.assertEqual(report["GENERATED_CASE_ROWS"], "NONE")

    @unittest.skipUnless(importlib.util.find_spec("scripts") is not None, "Frozen D2 module not in local artifact sandbox")
    def test_false_three_way_renderer_holdout_fails_closed(self):
        rs = roots()
        for root in rs:
            root["holdout_contract"] = "RENDERER"
            root["split"] = "train"
        other = copy.deepcopy(rs)
        for i, root in enumerate(other):
            root["split"] = "dev"
            for key in ("case_family_id", "split_group_id", "lexicon_namespace_id",
                        "semantic_world_fingerprint", "entity_tuple_fingerprint",
                        "source_attestation_fingerprint", "template_family_id"):
                root[key] += "_dev"
        combined = rs + other
        with self.assertRaisesRegex(ValueError, "CLAIMED_HOLDOUT_REUSE"):
            d3.audit_d2_plan_bridge(combined, variants(combined), grammars(), lineage(combined))

    @unittest.skipUnless(importlib.util.find_spec("scripts") is not None, "Frozen D2 module not in local artifact sandbox")
    def test_d2_incomplete_factorial_does_not_promote(self):
        rs = roots()
        report = d3.audit_d2_plan_bridge(rs, variants(rs)[:-1], grammars(), lineage(rs))
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["MATERIALIZED_COHORT_FEASIBILITY"], "NOT_EVALUATED")


class TestNoSideEffects(unittest.TestCase):
    def test_no_materialization_or_execution_functions(self):
        source = Path(d3.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_calls = {"open", "exec", "eval", "write_text", "write_bytes",
                           "dump", "dumps", "save", "mkdir", "run", "Popen",
                           "system", "seed", "shuffle", "tokenize", "render"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call = (node.func.id if isinstance(node.func, ast.Name) else
                        node.func.attr if isinstance(node.func, ast.Attribute) else "")
                self.assertNotIn(call, forbidden_calls)
            if isinstance(node, ast.Import):
                self.fail("unexpected import in static contract module")
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, ("collections.abc", "scripts"))
        self.assertNotIn("def render_primary_variant(", source)
        self.assertNotIn("def audit_materialized_cohort(", source)


if __name__ == "__main__":
    unittest.main()
