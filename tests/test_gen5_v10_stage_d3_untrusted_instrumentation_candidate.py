#!/usr/bin/env python3
"""Stage D3 untrusted observation contract, entirely CPU/static."""
from __future__ import annotations

import copy
import io
import contextlib
import os
import sys
import unittest
from unittest.mock import patch

from scripts import gen5_v10_stage_d3_untrusted_instrumentation_candidate as d3
from scripts import gen5_v10_stage_d2_static_runtime_preflight_candidate as d2


class TestD3StaticInstrumentation(unittest.TestCase):
    def make(self, runtime=False):
        claims = None
        if runtime:
            claims = {
                "checkpoint_sha256": d2.CHECKPOINT_SHA256,
                "dev_order_sha256": d2.DEV_ORDER_SHA256,
                "active_encoding_sha256": d2.ACTIVE_ENCODING_SHA256,
                "model_revision": d2.MODEL_REVISION,
                "worker_id": 1, "batch_index_one_based": 13,
                "row_start": 800, "row_stop": 832, "source_cell": "6201_6201",
                "target_cells": ["6201_6202", "6201_6203"],
                "code_head": "0" * 40, "claimed_branch": d2.BRANCH,
                "claimed_clean_worktree": True,
                "source_status": "SELF_ASSERTED_NOT_AUTHENTICATED",
                "physical_device": "NVIDIA Tesla T4 claimed", "torch_version": "2.x claimed",
                "cuda_runtime": "12.x claimed", "cublas_version": "12.x claimed",
                "allow_tf32": False,
            }
        return d3.observation(
            source_kind="UNVERIFIED_RUNTIME_CLAIM" if runtime else "CPU_SYNTHETIC_FIXTURE",
            claimed_code_head="0"*40,
            selector=copy.deepcopy(d3.BOUNDS), entries=d3.synthetic_events(memory=runtime),
            source_claims=claims)

    def mutated(self, fn, reason, runtime=False):
        obj = self.make(runtime)
        fn(obj)
        obj["observation_sha256"] = d3._hash({k:v for k,v in obj.items() if k != "observation_sha256"})
        with self.assertRaisesRegex(d3.D3Blocked, reason): d3.validate(obj)

    def test_mock_trace(self):
        report = d3.validate(self.make())
        self.assertEqual(report["event_count"],17)
        self.assertEqual(report["capture_phases"],3)
        self.assertFalse(report["source_tensor_bytes_authenticated"])
        self.assertEqual(report["readiness"], "BLOCKED_INDEPENDENT_RUNTIME_AND_MEMORY_EVIDENCE")

    def test_repeatability_and_no_mutation(self):
        a = self.make(); b = self.make()
        self.assertEqual(a,b)
        copy_a = copy.deepcopy(a)
        self.assertEqual(d3.validate(a), d3.validate(b))
        self.assertEqual(copy_a,a)

    def test_claimed_gpu_memory_never_certifies(self):
        r = d3.validate(self.make(runtime=True))
        self.assertFalse(r["observed_gpu_peak_authenticated"])
        self.assertEqual(r["GPU_EXECUTION"],"NOT_AUTHORIZED")

    def test_missing_event(self):
        self.mutated(lambda x:x["events"].pop(), "EVENT_COUNT")

    def test_extra_event(self):
        self.mutated(lambda x:x["events"].append(x["events"][-1]), "EVENT_COUNT")

    def test_event_reorder(self):
        self.mutated(lambda x:x["events"].reverse(), "EVENT_ORDER")

    def test_wrong_event_ordinal(self):
        self.mutated(lambda x:x["events"][3].__setitem__("ordinal", 9), "EVENT_ORDER")

    def test_duplicate_elapsed(self):
        self.mutated(lambda x:x["events"][2].__setitem__("elapsed_ns",2), "NONMONOTONIC")

    def test_negative_elapsed(self):
        self.mutated(lambda x:x["events"][2].__setitem__("elapsed_ns",-1), "ELAPSED_NS")

    def test_float_elapsed(self):
        self.mutated(lambda x:x["events"][2].__setitem__("elapsed_ns",2.0), "ELAPSED_NS")

    def test_false_gpu_on_cpu_fake(self):
        self.mutated(lambda x:x["events"][2].__setitem__("memory",{}),"CPU_FAKE_MEMORY")

    def test_memory_dtype(self):
        self.mutated(lambda x:x["events"][0]["memory"].__setitem__("allocated_bytes",True), "NONNEG_INT",runtime=True)

    def test_memory_alloc_reserved(self):
        self.mutated(lambda x:x["events"][0]["memory"].__setitem__("allocated_bytes",21), "MEMORY_INCONSISTENT",runtime=True)

    def test_memory_reserved_total(self):
        self.mutated(lambda x:x["events"][0]["memory"].__setitem__("reserved_bytes",101), "MEMORY_INCONSISTENT",runtime=True)

    def test_memory_driver_free_total(self):
        self.mutated(lambda x:x["events"][0]["memory"].__setitem__("driver_free_bytes",101), "MEMORY_INCONSISTENT",runtime=True)

    def test_memory_peak_below_alloc(self):
        self.mutated(lambda x:x["events"][0]["memory"].__setitem__("peak_allocated_bytes",9), "MEMORY_INCONSISTENT",runtime=True)

    def test_memory_missing_field(self):
        self.mutated(lambda x:x["events"][0]["memory"].pop("driver_free_bytes"), "RUNTIME_MEMORY_FIELDS",runtime=True)

    def test_memory_peak_cannot_reset(self):
        self.mutated(lambda x:x["events"][2]["memory"].__setitem__("peak_allocated_bytes",29),
                     "RESET_OR_NONMONOTONIC_PEAK",runtime=True)

    def test_missing_d2_lineage_claims(self):
        self.mutated(lambda x:x.__setitem__("source_claims",None),
                     "D2_SOURCE_CLAIMS",runtime=True)

    def test_d2_checkpoint_claim_drift(self):
        self.mutated(lambda x:x["source_claims"].__setitem__("checkpoint_sha256","0"*64),
                     "D2_SOURCE_CLAIMS",runtime=True)

    def test_claimed_head_mismatch(self):
        self.mutated(lambda x:x["source_claims"].__setitem__("code_head","f"*40),
                     "CLAIMED_HEAD_OR_SOURCE_AUTHENTICATION",runtime=True)

    def test_cpu_cannot_claim_runtime_lineage(self):
        self.mutated(lambda x:x.__setitem__("source_claims",{}),"CPU_FAKE_SOURCE_CLAIMS")

    def test_missing_guard(self):
        self.mutated(lambda x:x["guard_trace"].pop("source_gradient_projector"),"GUARD_FIELDS")

    def test_guard_false_pass(self):
        self.mutated(lambda x:x["guard_trace"].__setitem__("source_gradient_projector","PASS"),
                     "GUARD_FALSE_ATTESTATION")

    def test_all_guard_claims_do_not_promote(self):
        x=self.make(runtime=True)
        x["guard_trace"]={name:"OBSERVED_CLAIM_ONLY" for name in d3.GUARD_KEYS}
        x["observation_sha256"]=d3._hash({k:v for k,v in x.items() if k!="observation_sha256"})
        r=d3.validate(x)
        self.assertEqual(r["E1"],"NOT_EVALUATED")

    def test_manifest_capture_hash_mismatch(self):
        self.mutated(lambda x:x["events"][d3.EVENTS.index("POST_V10_CAPTURED")].__setitem__("capture_manifest_sha256","2"*64), "INPUT_CAPTURE_VALUE_DRIFT")

    def test_capture_hash_missing(self):
        self.mutated(lambda x:x["events"][d3.EVENTS.index("PRE_FROZEN_CAPTURED")].__setitem__("capture_manifest_sha256",None), "CAPTURE_DIGEST_MISSING")

    def test_non_capture_digest_injected(self):
        self.mutated(lambda x:x["events"][0].__setitem__("capture_manifest_sha256","0"*64), "NONCAPTURE_DIGEST")

    def test_selector_wrong_batch(self):
        self.mutated(lambda x:x["selector"].__setitem__("batch_index_one_based",12), "SELECTOR_MISMATCH")

    def test_selector_wrong_source(self):
        self.mutated(lambda x:x["selector"].__setitem__("source_cell","6202_6202"), "SELECTOR_MISMATCH")

    def test_selector_reversed_targets(self):
        self.mutated(lambda x:x["selector"]["target_cells"].reverse(), "SELECTOR_MISMATCH")

    def test_selected_scope_counts(self):
        self.mutated(lambda x:x["work_counts"].__setitem__("source_gradient_calls",2), "WORK_COUNTS")

    def test_missing_work_count(self):
        self.mutated(lambda x:x["work_counts"].pop("raw_write_cells"), "WORK_COUNTS")

    def test_code_head_must_not_be_base(self):
        self.mutated(lambda x:x.__setitem__("claimed_code_head",d3.BASE), "EXECUTION_HEAD")

    def test_invalid_code_head(self):
        self.mutated(lambda x:x.__setitem__("claimed_code_head","invalid"), "EXECUTION_HEAD")

    def test_not_authorized_even_self_rehashed(self):
        for fld in ("E0","E1","E2","E3","GPU_EXECUTION","attestation","readiness"):
            with self.subTest(field=fld):
                self.mutated(lambda x:x.__setitem__(fld,"PASS"), "FORBIDDEN_PROMOTION")

    def test_tamper_without_rehash(self):
        x=self.make();x["events"][2]["elapsed_ns"] += 10
        with self.assertRaises(d3.D3Blocked): d3.validate(x)

    def test_extra_schema_field(self):
        self.mutated(lambda x:x.__setitem__("actual_gpu_ready",True), "OBSERVATION_FIELDS")

    def test_json_nan_rejected(self):
        with self.assertRaisesRegex(d3.D3Blocked,"NONCANONICAL_JSON"):
            d3._canonical({"bad":float("nan")})

    def test_entry_missing_field(self):
        self.mutated(lambda x:x["events"][0].pop("memory"), "EVENT_FIELDS")

    def test_wrong_event_source_kind(self):
        self.mutated(lambda x:x.__setitem__("source_kind","AUTHENTICATED_CUDA"), "SOURCE_KIND")

    def test_static_cli_is_inert(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}), patch.object(sys,"argv",["d3","--static-contract-check"]):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(d3.main(),0)
            self.assertIn("GPU=OFF",output.getvalue())

    def test_cli_rejects_cuda_visibility(self):
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"0"}), patch.object(sys,"argv",["d3","--static-contract-check"]):
            with self.assertRaisesRegex(d3.D3Blocked,"CUDA_DISABLED"):
                d3.main()

    def test_cli_rejects_run_mode(self):
        with patch.object(sys,"argv",["d3","--run-bounded"]),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): d3.main()


if __name__ == "__main__":
    unittest.main()
