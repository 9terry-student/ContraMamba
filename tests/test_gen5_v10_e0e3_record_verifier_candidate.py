#!/usr/bin/env python3
"""CPU-only verifier tests; no torch, GPU, dataset or model loading."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from scripts import gen5_v10_e0e3_record_verifier_candidate as v

class TestE0E3RecordCandidate(unittest.TestCase):
    def setUp(self):
        self.fx=v.synthetic_fixture()

    def test_known_diagnostic_mismatch_never_promotes(self):
        r=v.evaluate_v10(self.fx)
        self.assertEqual(r["LEGACY_FULL_RECORD"],"FAIL")
        self.assertEqual(r["RAW_PAYLOAD_NUMERIC_BOUNDED"],"PASS_PROVISIONAL")
        self.assertEqual(r["V10_E0_PROVENANCE"],"NOT_EVALUATED")
        self.assertEqual(r["V10_E1_FROZEN_GUARDS"],"NOT_EVALUATED")
        self.assertEqual(r["V10_E2_PAYLOAD"],"NOT_EVALUATED")
        self.assertEqual(r["V10_E3_DIAGNOSTICS"],"NOT_EVALUATED")
        self.assertEqual(r["V10_PAYLOAD_EQUIVALENCE"],"NOT_EVALUATED")

    def test_replay_tamper_blocks(self):
        self.fx["gemm_records"][0]["fine_pair_C_by_gap"][v.PARENTS[0]][v.FINE[0]][7]=.1
        with self.assertRaisesRegex(v.ContractBlocked,"FROZEN_PER_GAP_REPLAY_FAIL"):
            v.evaluate_v10(self.fx)

    def test_diagnostic_forgery_blocks(self):
        self.fx["gemm_records"][0][v.DIAGNOSTIC]=0.
        with self.assertRaisesRegex(v.ContractBlocked,"DIAGNOSTIC_NOT_TRUTHFUL"):
            v.evaluate_v10(self.fx)

    def test_missing_fine_cell_blocks(self):
        self.fx["gemm_records"][0]["fine_pair_C_by_gap"][v.PARENTS[0]].pop(v.FINE[-1])
        with self.assertRaisesRegex(v.ContractBlocked,"FINE_CLASSES"):
            v.evaluate_v10(self.fx)

    def test_missing_record_key_blocks(self):
        self.fx["gemm_records"][0].pop("R")
        with self.assertRaisesRegex(v.ContractBlocked,"RECORD_SCHEMA_KEYS"):
            v.evaluate_v10(self.fx)

    def test_nonfinite_blocks(self):
        self.fx["gemm_records"][0]["R"]=float("nan")
        with self.assertRaisesRegex(v.ContractBlocked,"NONFINITE"):
            v.evaluate_v10(self.fx)

    def test_payload_mismatch_is_failure_not_pass(self):
        self.fx["gemm_records"][0]["R"]=4.0
        r=v.evaluate_v10(self.fx)
        self.assertEqual(r["V10_E2_PAYLOAD"],"FAIL")
        self.assertEqual(r["V10_PAYLOAD_EQUIVALENCE"],"FAIL")

    def test_missing_provenance_pair_blocks(self):
        with self.assertRaisesRegex(v.ContractBlocked,"REPO_AND_EXPECTED_HEAD_MUST_BE_TOGETHER"):
            v.evaluate_v10(self.fx,expected_repo_head="0"*40)

    def test_wrong_top_level_schema_blocks(self):
        self.fx["unexpected"]=True
        with self.assertRaisesRegex(v.ContractBlocked,"BUNDLE_TOPLEVEL_SCHEMA"):
            v.evaluate_v10(self.fx)

    def test_k_count_reject(self):
        self.fx["gemm_records"].pop()
        with self.assertRaisesRegex(v.ContractBlocked,"K_COUNT"):
            v.evaluate_v10(self.fx)

    def test_report_path_inside_source_worktree_blocks(self):
        source_repo = Path(v.__file__).resolve().parents[1]
        with self.assertRaisesRegex(v.ContractBlocked,"REPORT_INSIDE_SOURCE_WORKTREE_FORBIDDEN"):
            v.validate_report_destination(source_repo / "reports" / "probe.json")

    def test_report_path_inside_authenticated_worktree_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            repo = base / "authenticated"
            repo.mkdir()
            with self.assertRaisesRegex(v.ContractBlocked,"REPORT_INSIDE_AUTHENTICATED_WORKTREE_FORBIDDEN"):
                v.validate_report_destination(repo / "reports" / "probe.json", repo=repo)
            v.validate_report_destination(base / "outside.json", repo=repo)

    def test_derived_normalized_h_detects_drift(self):
        # Paired tiny S and C can mask an H drift in the original C-field tolerance.
        for row in self.fx["frozen_records"]:
            row["S"]=1e-12
        for row in self.fx["gemm_records"]:
            row["S"]=1e-12
        # expected row diagnostic must be recomputed after changing S.
        # Here use the normalized comparator in isolation to test delta detection.
        self.fx["gemm_records"][0]["fine_pair_C_by_gap"][v.PARENTS[0]][v.FINE[1]][4] = 1e-12
        x=v.derived_normalized_payload(self.fx["frozen_records"], self.fx["gemm_records"])
        self.assertEqual(x["status"],"FAIL")
        self.assertGreater(x["counts"]["failed"],0)

if __name__=="__main__": unittest.main(verbosity=2)
