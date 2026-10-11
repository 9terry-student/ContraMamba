#!/usr/bin/env python3
"""Gen5 v10 Stage B: offline strict V2 paired-bundle verifier candidate.

STATIC / synthetic CPU contract only. Metadata digests and self-asserted replay
cannot authenticate model/source tensors, GPU execution, or original guards.
No CUDA import, no model loading, and no scientific promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any

from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
from scripts import gen5_v10_bounded_worker_export_candidate as worker

SCHEMA = "GEN5_V10_BOUNDED_PAIRED_EXPORT_V2"
REPORT_SCHEMA = "GEN5_V10_BOUNDED_PAIRED_VERIFICATION_V2"
AUTH_BASE = "f8e60f173d3a72aa17779449ec9b931c7fa4d043"
PARENTS = ("CLAIM->CLAIM", "EVIDENCE->EVIDENCE", "CLAIM->EVIDENCE")
CLASSES = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "RESIDUAL")
FINE = tuple(f"{a}->{b}" for a in CLASSES for b in CLASSES)
REGION = ("CLAIM->CLAIM", "CLAIM->EOS", "CLAIM->EVIDENCE",
          "EOS->EVIDENCE", "EVIDENCE->EVIDENCE")
GAPS = 127
ATOL = 2e-5
RTOL = 2e-5
EPSILON = 1e-30
DIAGNOSTIC = "fine_parent_reconstruction_rel_max"
FIELDS = frozenset((
    "R", "S", "E", "C", "Q", "pair_gap_C",
    "pair_gap_C_reconstruction_abs", "pair_gap_Q_reconstruction_abs",
    "pair_gap_fft_half_log_span_max", "region_pair_C_by_gap",
    "region_pair_gap_reconstruction_abs_max", "region_pair_gap_reconstruction_rel_max",
    "region_pair_fft_half_log_span_max", "fine_pair_C_by_gap",
    "fine_parent_reconstruction_abs_max", DIAGNOSTIC,
    "fine_pair_fft_half_log_span_max",
))
TOPLEVEL = frozenset(("schema_version", "candidate_base", "selector", "runtime_claims",
                       "input_capture", "engines", "evidence_status", "bundle_sha256"))
ENGINE_FIELDS = frozenset(("engine_id", "records", "records_sha256", "dispatch", "guards"))
RUNTIME_FIELDS = frozenset(("source_kind", "execution", "model_revision", "checkpoint_sha256",
                            "dev_order_sha256", "active_encoding_sha256", "code_head",
                            "physical_device", "cuda_runtime", "torch_version", "tf32"))
CAPTURE_FIELDS = frozenset(("pre_frozen", "post_frozen", "post_v10"))
GUARD_FIELDS = frozenset(("fine_parent_replay_claim", "upstream_guard_coverage"))

class V2Blocked(ValueError):
    pass


def demand(condition: bool, code: str) -> None:
    if not condition:
        raise V2Blocked("GEN5_V10_V2_BLOCKED:" + code)


def bytes_canonical(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise V2Blocked("NONCANONICAL_JSON") from exc


def digest(obj: Any) -> str:
    return hashlib.sha256(bytes_canonical(obj)).hexdigest()


def finite(v: Any, loc: str) -> float:
    demand(type(v) in (int, float) and math.isfinite(v), "NONFINITE_OR_NOT_NUMERIC:" + loc)
    return float(v)


def vector(v: Any, loc: str) -> list[float]:
    demand(type(v) is list and len(v) == GAPS, "VECTOR_GEOMETRY:" + loc)
    return [finite(x, loc + f"[{i}]") for i, x in enumerate(v)]


def pair_error(reference: float, observed: float) -> tuple[bool, float, float]:
    delta = abs(reference-observed)
    rel = delta/max(abs(reference), EPSILON)
    return (delta <= ATOL or rel <= RTOL, delta, rel)


def validate_row(rec: Any, loc: str) -> dict[str, Any]:
    demand(type(rec) is dict and set(rec) == FIELDS, "RECORD_FIELDS:" + loc)
    for key in FIELDS - {"fine_pair_C_by_gap", "region_pair_C_by_gap", "pair_gap_C"}:
        finite(rec[key], loc + "." + key)
    s = finite(rec["S"], loc + ".S")
    demand(s > 0, "NONPOSITIVE_S:" + loc)
    vector(rec["pair_gap_C"], loc + ".pair_gap_C")
    coarse = rec["region_pair_C_by_gap"]
    demand(type(coarse) is dict and set(coarse) == set(REGION),
           "COARSE_KEYS:" + loc)
    for region in REGION:
        vector(coarse[region], loc + "." + region)
    fine = rec["fine_pair_C_by_gap"]
    demand(type(fine) is dict and set(fine) == set(PARENTS),
           "FINE_PARENTS:" + loc)
    max_abs = 0.0
    max_rel = 0.0
    for parent in PARENTS:
        cells = fine[parent]
        demand(type(cells) is dict and set(cells) == set(FINE),
               "FINE_CELLS:" + loc + "." + parent)
        vals = [vector(cells[cell], f"{loc}.{parent}.{cell}") for cell in FINE]
        expected = vector(coarse[parent], f"{loc}.{parent}.coarse")
        summed = [0.0] * GAPS
        # Frozen sequential float64 sum, in its fixed fine-cell order.
        for cells_vector in vals:
            for j in range(GAPS):
                summed[j] += cells_vector[j]
        for gap, (a, b) in enumerate(zip(expected, summed), 1):
            eh, oh = a/s, b/s
            passed, abs_error, rel_error = pair_error(eh, oh)
            demand(passed, f"FROZEN_PARENT_REPLAY:{loc}:{parent}:gap{gap}")
            max_abs = max(max_abs, abs_error)
            max_rel = max(max_rel, rel_error)
    # Authentic diagnostics must not be replaced by the other engine's values.
    demand(math.isclose(float(rec["fine_parent_reconstruction_abs_max"]),
                        max_abs, rel_tol=1e-12, abs_tol=0.0),
           "FORGED_ABS_DIAGNOSTIC:" + loc)
    demand(math.isclose(float(rec[DIAGNOSTIC]), max_rel,
                        rel_tol=1e-12, abs_tol=0.0),
           "FORGED_REL_DIAGNOSTIC:" + loc)
    return {"recomputed_abs": max_abs, "recomputed_rel": max_rel,
            "per_gap_parent_replay": "PASS_RECOMPUTED_FROM_RECORDS"}


def _compare(a: Any, b: Any, path: str, stats: dict[str, Any]):
    if type(a) is dict:
        demand(type(b) is dict and set(a) == set(b), "COMPARE_KEYS:" + path)
        for key in a:
            _compare(a[key], b[key], path + "." + key, stats)
    elif type(a) is list:
        demand(type(b) is list and len(a) == len(b), "COMPARE_LEN:" + path)
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, path + f"[{i}]", stats)
    elif type(a) in (float, int) and type(b) in (float, int):
        if type(a) is int or type(b) is int:
            demand(type(a) is int and type(b) is int and a == b,
                   "EXACT_ID_OR_INT_MISMATCH:" + path)
            stats["exact_integers"] += 1
            return
        x = finite(a, path); y = finite(b, path)
        passed, delta, rel = pair_error(x,y)
        stats["numeric"] += 1
        special = path.endswith("." + DIAGNOSTIC)
        if not passed:
            stats["legacy_failed"] += 1
            if special:
                stats["diagnostic_differences"] += 1
            else:
                stats["payload_failed"] += 1
            if len(stats["mismatches"]) < 20:
                stats["mismatches"].append({"path": path, "kind":
                    "DIAGNOSTIC_DIFF" if special else "PAYLOAD_DIFF",
                    "frozen": x, "v10": y, "abs_diff": delta, "rel_diff": rel})
    else:
        demand(type(a) is type(b) and a == b, "METADATA_MISMATCH:" + path)


def compare_normalized(frozen: list[dict[str, Any]],
                       v10: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"H_by_gap": 0, "H_windows": 0, "fail": 0}
    windows = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 127))
    for fr, vr in zip(frozen, v10):
        for parent in PARENTS:
            for cell in FINE:
                fa = fr["fine_pair_C_by_gap"][parent][cell]
                va = vr["fine_pair_C_by_gap"][parent][cell]
                hf = [x/fr["S"] for x in fa]
                hv = [x/vr["S"] for x in va]
                for x, y in zip(hf, hv):
                    stats["H_by_gap"] += 1
                    if not pair_error(x,y)[0]: stats["fail"] += 1
                for lo, hi in windows:
                    stats["H_windows"] += 1
                    if not pair_error(sum(hf[lo-1:hi]),sum(hv[lo-1:hi]))[0]:
                        stats["fail"] += 1
    return stats


def validate(bundle: Any) -> dict[str, Any]:
    """Untrusted offline bundle inspection. Never promotes E0/E1/E2/E3."""
    demand(type(bundle) is dict and set(bundle) == TOPLEVEL, "BUNDLE_TOPLEVEL")
    demand(bundle["schema_version"] == SCHEMA, "BUNDLE_SCHEMA")
    demand(bundle["candidate_base"] == AUTH_BASE, "CANDIDATE_BASE")
    own = bundle["bundle_sha256"]
    demand(type(own) is str and len(own) == 64 and all(ch in '0123456789abcdef' for ch in own),
           "BUNDLE_HASH_FORMAT")
    demand(own == digest({k:v for k,v in bundle.items() if k != "bundle_sha256"}),
           "BUNDLE_HASH_FORGERY")
    locus = bundle["selector"]
    demand(type(locus) is dict and set(locus) == {
        "worker_id", "batch_index_one_based", "row_start", "row_stop",
        "row_ids", "source_cell", "target_cells", "target_group_index",
        "group", "expected_component_count", "expected_batch_rows"
    }, "LOCUS_SCHEMA")
    for name, expected in (("worker_id",1),("batch_index_one_based",13),
                           ("row_start",800),("row_stop",832),
                           ("target_group_index",0),("expected_component_count",4),
                           ("expected_batch_rows",32)):
        demand(type(locus[name]) is int and locus[name] == expected,
               "LOCUS_GEOMETRY:"+name)
    ids = locus["row_ids"]
    demand(type(ids) is list and len(ids) == 32 and all(type(v) is str and bool(v) for v in ids)
           and len(set(ids)) == 32, "LOCUS_ROW_IDS")
    demand(locus["source_cell"] == "6201_6201", "LOCUS_SOURCE")
    demand(type(locus["target_cells"]) is list and len(locus["target_cells"]) == 2 and
           len(set(locus["target_cells"])) == 2 and
           all(type(v) is str and bool(v) for v in locus["target_cells"]),
           "LOCUS_TARGETS")
    demand(locus["target_cells"] == ["6201_6202", "6201_6203"], "LOCUS_TARGET_ORDER")
    demand(locus["group"] == "CONTROL_R", "LOCUS_GROUP")
    runtime = bundle["runtime_claims"]
    demand(type(runtime) is dict and set(runtime) == RUNTIME_FIELDS,
           "RUNTIME_SCHEMA")
    demand(runtime["source_kind"] == "CPU_SYNTHETIC_ONLY" and
           runtime["execution"] == "NO" and runtime["code_head"] == AUTH_BASE and
           runtime["physical_device"] == "CPU" and runtime["cuda_runtime"] is None and
           runtime["tf32"] is False, "RUNTIME_NOT_STATIC")
    for field in ("model_revision", "checkpoint_sha256", "dev_order_sha256",
                  "active_encoding_sha256", "torch_version"):
        demand(runtime[field] == "NOT_AUTHENTICATED", "RUNTIME_FALSE_ATTEST:" + field)
    capture = bundle["input_capture"]
    demand(type(capture) is dict and set(capture) == CAPTURE_FIELDS, "CAPTURE_FIELDS")
    for name in CAPTURE_FIELDS:
        stage_a.validate_manifest_structure(capture[name])
    stage_a.compare_manifest_identity(capture["pre_frozen"],capture["post_frozen"])
    stage_a.compare_manifest_identity(capture["pre_frozen"],capture["post_v10"])
    geo = capture["pre_frozen"]["geometry"]
    demand(geo["synthetic_cpu_geometry"] is True and geo["k"] == 4 and
           geo["batch_rows"] == 32 and geo["sequence_length"] == 128,
           "INPUT_GEOMETRY")
    eng = bundle["engines"]
    demand(type(eng) is dict and set(eng) == {"frozen", "v10"}, "ENGINE_KEYS")
    verified = {}
    for name in ("frozen", "v10"):
        entry=eng[name]
        demand(type(entry) is dict and set(entry) == ENGINE_FIELDS,
               "ENGINE_SCHEMA:"+name)
        demand(entry["engine_id"] == ("FROZEN_FFT_V1" if name == "frozen" else "V10_HYBRID_CANDIDATE"),
               "ENGINE_ID:"+name)
        records = entry["records"]
        demand(type(records) is list and len(records) == 4, "ENGINE_RECORD_K4:"+name)
        demand(type(entry["records_sha256"]) is str and
               entry["records_sha256"] == digest(records), "RECORD_HASH:"+name)
        verified[name]=[validate_row(rec, name+f"[{i}]") for i,rec in enumerate(records)]
        guards=entry["guards"]
        demand(type(guards) is dict and set(guards) == GUARD_FIELDS,
               "GUARD_FIELDS:"+name)
        demand(guards["fine_parent_replay_claim"] == "PASS" and
               guards["upstream_guard_coverage"] == "NOT_AUTHENTICATED",
               "GUARD_ATTESTATION:"+name)
        route=entry["dispatch"]
        demand(type(route) is dict, "DISPATCH_RECORD:"+name)
        if name == "frozen":
            demand(route == {"mode":"FROZEN_WHOLE", "batch_rows":32},
                   "FROZEN_ROUTE")
        else:
            demand(set(route)=={"mode","batch_rows","fast_chunks","gemm_chunks",
                                "frozen_tail_calls","total_chunks","max_half_span"},
                   "V10_ROUTE_FIELDS")
            for key in ("batch_rows","fast_chunks","gemm_chunks","frozen_tail_calls","total_chunks"):
                demand(type(route[key]) is int, "ROUTE_INTEGER:"+key)
            demand(route["mode"] == "CHUNKWISE_HYBRID" and route["batch_rows"] == 32 and
                   route["total_chunks"] == math.ceil(geo["state_width"] / 4) and route["fast_chunks"] >= 0 and
                   route["gemm_chunks"] >= 0 and route["frozen_tail_calls"] == 0 and
                   route["fast_chunks"] + route["gemm_chunks"] == route["total_chunks"],
                   "ROUTE_COVERAGE")
            finite(route["max_half_span"], "ROUTE_HALF_SPAN")
    stats = {"numeric":0,"exact_integers":0,"legacy_failed":0,"payload_failed":0,
             "diagnostic_differences":0,"mismatches":[]}
    _compare(eng["frozen"]["records"],eng["v10"]["records"],"records",stats)
    derived = compare_normalized(eng["frozen"]["records"],eng["v10"]["records"])
    payload_bad = bool(stats["payload_failed"] or derived["fail"])
    demand(bundle["evidence_status"] == "CPU_SYNTHETIC_STATIC_ONLY_NOT_APPROVED",
           "EVIDENCE_STATUS_FALSE")
    return {
        "schema_version": REPORT_SCHEMA,
        "bundle_sha256": own,
        "candidate_base": AUTH_BASE,
        "bounded_selector": locus,
        "record_truth": {"frozen":verified["frozen"],"v10":verified["v10"]},
        "comparison": stats,
        "derived_H":derived,
        "LEGACY_FULL_RECORD":"FAIL" if stats["legacy_failed"] else "PASS",
        "V10_E0_PROVENANCE":"NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS":"NOT_EVALUATED",
        "V10_E2_PAYLOAD":"FAIL" if payload_bad else "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS":"NOT_EVALUATED",
        "V10_PAYLOAD_EQUIVALENCE":"FAIL" if payload_bad else "NOT_EVALUATED",
        "V10_E2_BOUNDED_NUMERIC":"FAIL" if payload_bad else "PASS_PROVISIONAL",
        "V10_E3_BOUNDED_RAW":"DIVERGENCE" if stats["diagnostic_differences"] else "AGREEMENT",
        "V10_E0_MISSING":["authenticated_source_tensor_bytes","frozen_model_checkpoint_tokenizer_lineage",
                            "gpu_runtime_and_exact_worker_seam"],
        "V10_E1_MISSING":["actual_upstream_guard_execution","authenticated_runtime_guard_trace"],
        "population_scope":"NOT_EVALUATED", "PRODUCTION":"BLOCKED",
        "SCIENTIFIC_EXECUTION":"NOT_AUTHORIZED",
    }


def seal_synthetic_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """CPU synthetic helper, never a real/runtime attestation."""
    demand(type(payload) is dict and "bundle_sha256" not in payload,
           "SEAL_PAYLOAD_SCHEMA")
    obj=dict(payload)
    obj["bundle_sha256"]=digest(obj)
    return obj


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check",action="store_true",required=True)
    args=parser.parse_args()
    demand(args.static_contract_check,"STATIC_ONLY")
    demand(os.environ.get("CUDA_VISIBLE_DEVICES","") in ("","-1"),
           "CUDA_DISABLED_REQUIRED")
    print("GEN5_V10_STAGE_B_V2_VERIFIER_CPU_STATIC_CONTRACT_PASS")
    print("SCHEMA="+SCHEMA+";VERIFICATION="+REPORT_SCHEMA)
    print("V10_E0_PROVENANCE=NOT_EVALUATED;V10_E1_FROZEN_GUARDS=NOT_EVALUATED")
    print("GPU=OFF;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;PRODUCTION=BLOCKED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
