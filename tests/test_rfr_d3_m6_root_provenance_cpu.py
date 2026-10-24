"""RFR-D3-M6 adversarial metadata-only tests. Never produce sampled roots."""

import ast
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import rfr_d3_m6_root_provenance_cpu as m6


def plan():
    # Reconstructed literal contract; no M4 test fixture is imported.
    return {
        "purpose": "SURFACE_GENERALIZATION_ONLY",
        "root_unit": "INDEPENDENT_SYNTHETIC_MICROWORLD_DRAW",
        "derivation_rule_id": "B_AND_A_C", "oracle_version": "RFR_D1_ORACLE_V1",
        "surface_domains": ["REGISTRY_RECORD", "INSTRUMENT_LOG"],
        "planned_splits": ["train", "dev", "test"], "split_assignments": "NONE",
        "split_group_inheritance": "ALL_DESCENDANTS_SAME_ROOT_GROUP",
        "holdout_contract": "NONE",
        "root_draw_protocol": "PROSPECTIVE_SEPARATE_ROOT_SUBSTREAMS",
        "draw_components": ["ENTITY_CONFIGURATION", "BASE_RECORD_IDENTITIES",
                            "SOURCE_ATTESTATION_IDENTITIES", "GRAMMAR_AND_TEMPLATE_CHOICES"],
        "root_manifest_policy": "DECLARATION_ONLY_PIN_BEFORE_REALIZATION",
        "logical_generalization_claim": False, "realized_independence_evidence": "NONE",
    }


def manifest():
    return {
        "manifest_schema_version": "RFR_D3_M6_ROOT_PLAN_V1",
        "plan_identity": "frozen_prospective_plan_name_only",
        "generator_version": "planned_gen_version", "renderer_version": "planned_renderer_version",
        "grammar_version": "planned_grammar_version",
        "canonicalization_version": "M6_TYPED_GRAPH_V1",
        "root_draw_protocol_id": "proposed_component_substreams_v1",
        "seed_derivation_spec_state": "NOT_FROZEN_NO_SEED_SELECTED",
        "attempt_ledger_policy": "ALL_ATTEMPTS_WITH_REJECTION_REASONS",
        "component_ref_policy": "FOUR_DISTINCT_COMPONENT_DRAWS_PER_ROOT",
        "root_lineage_policy": "PRIMARY_ROOT_NO_DERIVED_RECOUNT",
        "split_policy": "UNASSIGNED_INHERIT_ON_LATER_AUTHORITY",
        "realization_state": "DECLARATIONS_ONLY_NO_DRAWS",
        "claim_scope": "SURFACE_GENERALIZATION_PLAN_ONLY",
    }


def graph(suffix, hops=0, add_decoration=False):
    nodes = [{"node_id": f"{suffix}_{role}", "role": role} for role in (
        "ENTITY_MAIN", "RECORD_A", "RECORD_C", "RECORD_U",
        "ATTEST_A", "ATTEST_C", "ATTEST_U")]
    edges = []
    for target in "ACU":
        edges.append({"from": f"{suffix}_ENTITY_MAIN", "to": f"{suffix}_RECORD_{target}", "relation": "HAS_RECORD"})
        edges.append({"from": f"{suffix}_RECORD_{target}", "to": f"{suffix}_ATTEST_{target}", "relation": "HAS_ATTESTATION"})
    parent = f"{suffix}_ATTEST_A"
    for i in range(hops):
        hop = f"{suffix}_hop_{i}"
        nodes.append({"node_id": hop, "role": "ATTEST_HOP"})
        edges.append({"from": parent, "to": hop, "relation": "CHAIN_LINK"})
        parent = hop
    if add_decoration:
        nodes.append({"node_id": f"{suffix}_ornament", "role": "AUX_NEUTRAL"})
        edges.append({"from": f"{suffix}_ENTITY_MAIN", "to": f"{suffix}_ornament", "relation": "NEUTRAL_CONTEXT"})
    return {"nodes": nodes, "edges": edges}


def root(i, hops, values=(False, False, False)):
    d = "REGISTRY_RECORD" if i == 0 else "INSTRUMENT_LOG"
    versions = manifest()
    row = {
        "case_family_id": f"planned_fam_{i}", "split_group_id": f"planned_group_{i}",
        "semantic_world_id": f"planned_instance_{i}", "surface_domain": d,
        "renderer_family_id": f"renderer_for_{d}",
        "lexicon_namespace_id": f"lexicon_{i}", "template_family_id": f"template_{i}",
        "root_initial_A": values[0], "root_initial_C": values[1], "root_initial_U": values[2],
        "derivation_rule_id": "B_AND_A_C", "oracle_version": "RFR_D1_ORACLE_V1",
        "root_substream_id": f"future_stream_{i}", "root_kind": "PRIMARY_ROOT",
        "ancestry_parent_id": None, "split_assignment": None,
        "instance_graph": graph(f"root_{i}", hops),
        "declaration_status": "HYPOTHETICAL_NOT_DRAWN",
        "component_draw_refs": {
            component: {
                "source_stream_id": f"future_{i}_{j}_stream",
                "counter_index": 0, "draw_record_ref": f"future_{i}_{j}_record",
                "parent_root_substream_id": f"future_stream_{i}",
            }
            for j, component in enumerate(sorted(m6.COMPONENTS))
        },
    }
    for key in ("plan_identity", "generator_version", "renderer_version",
                "grammar_version", "canonicalization_version", "root_draw_protocol_id"):
        row[key] = versions[key]
    return row


def attempt(i, ordinal, accepted=True, reason=None):
    return {"attempt_id": f"proposed_{i}_{ordinal}", "case_family_id": f"planned_fam_{i}",
            "root_substream_id": f"future_stream_{i}", "ordinal": ordinal,
            "accepted": accepted, "reject_reason": reason}


def sample_declarations():
    # Hypothetical, hand-authored structures; *not* executed draws or generated prompts.
    return [root(0, 0), root(1, 1)], [attempt(0, 0), attempt(1, 0)]


class TestManifestContract(unittest.TestCase):
    def test_plan_only_is_not_a_realized_manifest(self):
        p, m = plan(), manifest()
        before = copy.deepcopy((p, m))
        result = m6.audit_manifest_contract(p, m)
        self.assertEqual(result["status"], "M6_PASS_PLAN_ONLY_PROVENANCE_NOT_VERIFIED")
        self.assertEqual(result["ACTUAL_ROOT_PROVENANCE"], "NOT_EVALUATED")
        self.assertEqual(result["DRAW_SEED_SELECTED"], "NO")
        self.assertEqual((p, m), before)

    def test_false_seed_selection_or_realization_blocked(self):
        for key, value in (("seed_derivation_spec_state", "PINNED_SEED_42"),
                           ("realization_state", "REALIZED"),
                           ("claim_scope", "INDEPENDENT_PASS"),
                           ("split_policy", "ASSIGNED")):
            m = manifest()
            m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.audit_manifest_contract(plan(), m)

    def test_m4_frame_failure_is_not_overridden(self):
        p = plan()
        p["holdout_contract"] = "RENDERER"
        with self.assertRaises(ValueError):
            m6.audit_manifest_contract(p, manifest())

    def test_schema_version_and_missing_plan_identity(self):
        for key, value in (("canonicalization_version", "V2"),
                           ("plan_identity", ""),
                           ("generator_version", None)):
            m = manifest()
            m[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.audit_manifest_contract(plan(), m)
        m = manifest()
        m["seed_value"] = 1234
        with self.assertRaises(ValueError):
            m6.audit_manifest_contract(plan(), m)


class TestProspectiveRoster(unittest.TestCase):
    def test_two_structurally_distinct_declarations_only(self):
        roots, ledger = sample_declarations()
        snapshot = copy.deepcopy((roots, ledger))
        report = m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        self.assertEqual(report["status"], "M6_PASS_PLAN_ONLY_PROVENANCE_NOT_VERIFIED")
        self.assertEqual(report["declared_candidate_roots"], 2)
        self.assertEqual(report["distinct_declared_core_logic_worlds"], 1)
        self.assertEqual(report["distinct_declared_instance_graphs"], 2)
        self.assertEqual(report["verified_independent_root_draws"], 0)
        self.assertEqual(report["draw_provenance_observed"], "NO")
        self.assertEqual(report["semantic_independence"], "NOT_ESTABLISHED")
        self.assertEqual((roots, ledger), snapshot)

    def test_isomorphic_root_with_renamed_nodes_blocked(self):
        roots, ledger = sample_declarations()
        roots[1]["instance_graph"] = graph("different_cosmetic_names", hops=0)
        with self.assertRaisesRegex(ValueError, "ISOMORPHIC_INSTANCE_GRAPH"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_neutral_decoration_cannot_fake_distinct_instance(self):
        roots, ledger = sample_declarations()
        roots[1]["instance_graph"] = graph("another", hops=0, add_decoration=True)
        with self.assertRaisesRegex(ValueError, "ISOMORPHIC_INSTANCE_GRAPH"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_graph_node_and_edge_order_independent(self):
        roots, ledger = sample_declarations()
        roots[1]["instance_graph"]["nodes"].reverse()
        roots[1]["instance_graph"]["edges"].reverse()
        self.assertEqual(m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)["declared_candidate_roots"], 2)

    def test_core_logic_repeat_allowed_even_different_domain(self):
        roots, ledger = sample_declarations()
        self.assertEqual(tuple(roots[0][f"root_initial_{v}"] for v in "ACU"),
                         tuple(roots[1][f"root_initial_{v}"] for v in "ACU"))
        self.assertEqual(m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)["distinct_declared_core_logic_worlds"], 1)

    def test_strict_boolean_and_semantic_rule(self):
        for key, value in (("root_initial_A", 0), ("root_initial_C", "F"),
                           ("derivation_rule_id", "B_OR_A_C"),
                           ("oracle_version", "RFR_D1_ORACLE_V2")):
            roots, ledger = sample_declarations()
            roots[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_no_realized_claim_or_split_or_descendant(self):
        for key,value in (("declaration_status", "REALIZED"),
                          ("split_assignment", "train"),
                          ("root_kind", "PARAPHRASE_DESCENDANT"),
                          ("ancestry_parent_id", "some_root")):
            roots, ledger = sample_declarations()
            roots[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_version_and_provenance_links(self):
        for key,value in (("generator_version", "mismatched"),
                          ("plan_identity", "different_plan"),
                          ("root_draw_protocol_id", "different_draw")):
            roots, ledger = sample_declarations()
            roots[0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ROOT_MANIFEST_VERSION"):
                m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_duplicate_component_record_or_stream_blocked(self):
        for key in ("draw_record_ref", "source_stream_id"):
            roots, ledger = sample_declarations()
            component = sorted(m6.COMPONENTS)[0]
            roots[1]["component_draw_refs"][component][key] = roots[0]["component_draw_refs"][component][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "COPIED_COMPONENT_DRAW"):
                m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_component_root_substream_linkage(self):
        roots, ledger = sample_declarations()
        component = sorted(m6.COMPONENTS)[0]
        roots[1]["component_draw_refs"][component]["parent_root_substream_id"] = roots[0]["root_substream_id"]
        with self.assertRaisesRegex(ValueError, "COMPONENT_ROOT_SUBSTREAM_CONFLICT"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_missing_component_and_nonint_counter_blocked(self):
        roots, ledger = sample_declarations()
        del roots[0]["component_draw_refs"]["BASE_RECORD_IDENTITIES"]
        with self.assertRaises(ValueError):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        roots, ledger = sample_declarations()
        component = sorted(m6.COMPONENTS)[0]
        roots[0]["component_draw_refs"][component]["counter_index"] = True
        with self.assertRaisesRegex(ValueError, "BAD_DRAW_COUNTER"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_root_group_world_or_substream_reuse_blocked(self):
        for key in ("case_family_id", "split_group_id", "semantic_world_id", "root_substream_id"):
            roots, ledger = sample_declarations()
            roots[1][key] = roots[0][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_missing_attempt_or_noncontiguous_attempts_blocked(self):
        roots, ledger = sample_declarations()
        with self.assertRaisesRegex(ValueError, "MISSING_OR_GAPPED_ATTEMPT_LEDGER"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger[:-1])
        roots, ledger = sample_declarations()
        ledger[1]["ordinal"] = 2
        with self.assertRaisesRegex(ValueError, "MISSING_OR_GAPPED_ATTEMPT_LEDGER"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_rejected_attempt_accounted_but_not_proven(self):
        roots, ledger = sample_declarations()
        ledger[0] = attempt(0, 0, False, "declared_duplicate_instance_graph")
        ledger.append(attempt(0, 1))
        report = m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        self.assertEqual(report["declared_attempts"], 3)
        self.assertEqual(report["selection_ledger_exhaustiveness"], "NOT_VERIFIABLE_FROM_DECLARATIONS")

    def test_silent_retry_and_accepted_with_reason_blocked(self):
        roots, ledger = sample_declarations()
        ledger.append(attempt(0, 1, False, "late_reject"))
        with self.assertRaisesRegex(ValueError, "ATTEMPT_ACCEPTANCE_SEQUENCE"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        roots, ledger = sample_declarations()
        ledger[0]["reject_reason"] = "wrong"
        with self.assertRaisesRegex(ValueError, "ACCEPTED_WITH_REJECTION"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        roots, ledger = sample_declarations()
        ledger[0]["accepted"] = False
        with self.assertRaises(ValueError):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_invalid_graph_topology_and_oracle_contamination(self):
        roots, ledger = sample_declarations()
        roots[0]["instance_graph"]["edges"][0]["relation"] = "LOGICAL_DEPENDENCY"
        with self.assertRaises(ValueError):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)
        roots, ledger = sample_declarations()
        g = roots[0]["instance_graph"] = graph("n", add_decoration=True)
        g["edges"][-1]["relation"] = "HAS_ATTESTATION"
        with self.assertRaisesRegex(ValueError, "AUXILIARY_ORACLE_CONTAMINATION"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_attempt_substream_conflict(self):
        roots, ledger = sample_declarations()
        ledger[0]["root_substream_id"] = "some_other_stream"
        with self.assertRaisesRegex(ValueError, "ATTEMPT_SUBSTREAM_CONFLICT"):
            m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)

    def test_no_unwanted_execution_paths(self):
        src = Path(m6.__file__).read_text(encoding="utf-8")
        syntax = ast.parse(src)
        modules = set()
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        self.assertFalse(modules & {"random", "numpy", "torch", "transformers",
                                     "subprocess", "pathlib", "os", "socket", "pickle"})
        forbidden = {"open", "write", "write_text", "write_bytes", "mkdir",
                     "save", "seed", "shuffle", "fit", "tokenize", "render"}
        for node in ast.walk(syntax):
            if isinstance(node, ast.Call):
                operation = (node.func.id if isinstance(node.func, ast.Name) else
                             node.func.attr if isinstance(node.func, ast.Attribute) else None)
                self.assertNotIn(operation, forbidden)
        with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            roots, ledger = sample_declarations()
            self.assertEqual(m6.audit_declared_root_roster(plan(), manifest(), roots, ledger)["verified_independent_root_draws"], 0)


if __name__ == "__main__":
    unittest.main()
