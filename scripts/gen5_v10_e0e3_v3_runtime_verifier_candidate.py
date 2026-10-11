#!/usr/bin/env python3
"""Stage C strict offline V3 bounded runtime *candidate* verifier.

Accepts self-described mock or unverified-source records for schema diagnostics
only. Digests on JSON values are not third-party attestations. Does not accept
V2, load models or GPUs, write files, or promote E0/E1/E2/E3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from typing import Any

SCHEMA = "GEN5_V10_BOUNDED_RUNTIME_EXPORT_V3"
REPORT_SCHEMA = "GEN5_V10_BOUNDED_RUNTIME_VERIFICATION_V3"
BASE = "6683d4c0dc9e742f5ac503fe4851e1c5218b99d8"
SIDECAR_DOMAIN = b"GEN5_V10_BOUNDED_V3_SIDECAR_V1\x00"
HEX40 = re.compile(r"[a-f0-9]{40}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
CLASSES = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "RESIDUAL")
PARENTS = ("CLAIM->CLAIM", "EVIDENCE->EVIDENCE", "CLAIM->EVIDENCE")
FINE = tuple(f"{a}->{b}" for a in CLASSES for b in CLASSES)
REGIONS = ("CLAIM->CLAIM", "CLAIM->EOS", "CLAIM->EVIDENCE",
           "EOS->EVIDENCE", "EVIDENCE->EVIDENCE")
SCALARS = ("R", "S", "E", "C", "Q", "pair_gap_C_reconstruction_abs",
           "pair_gap_Q_reconstruction_abs", "pair_gap_fft_half_log_span_max",
           "region_pair_gap_reconstruction_abs_max", "region_pair_gap_reconstruction_rel_max",
           "region_pair_fft_half_log_span_max", "fine_parent_reconstruction_abs_max",
           "fine_parent_reconstruction_rel_max", "fine_pair_fft_half_log_span_max")
RECORD_FIELDS = frozenset(SCALARS + ("pair_gap_C", "region_pair_C_by_gap", "fine_pair_C_by_gap"))
ROOT_FIELDS = frozenset(("schema_version", "candidate_base", "selector", "code_identity",
                         "runtime_claims", "input_capture", "engines",
                         "guard_trace", "failure_evidence", "evidence_status",
                         "bundle_sha256", "sidecar"))
CAPTURE_STAGES = ("pre_frozen", "post_frozen", "post_v10")
CAPTURE_FIELDS = frozenset(("content_sha256", "byte_count", "shape", "dtype",
                             "source_device", "source_strides", "phase",
                             "synchronized", "chunk_cap_bytes", "V10_E0_PROVENANCE"))
GUARD_KEYS = ("frozen_original_full_fine_stats", "frozen_coarse_replay",
              "frozen_fine_parent_replay", "v10_coarse_replay", "v10_fine_parent_replay",
              "raw_reconstruction_auth", "visible_raw_energy_internal_replay",
              "complement_raw_energy_internal_replay", "source_gradient_projector")
ATOL, RTOL, EPSILON = 2e-5, 2e-5, 1e-30
WINDOWS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 127))
FROZEN_BLOBS = {
    "scripts/audit_native_recurrent_generator_anchor_pair_localization.py": "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635",
    "scripts/gen5_v10_hybrid_fine_pair_candidate.py": "56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0",
    "scripts/gen5_v10_shared_dyadic_gemm_candidate.py": "9003fb9bd23b2a6680ef210a51845058622badd2",
    "scripts/gen5_v10_bounded_tensor_provenance_candidate.py": "703e0f6faf15ee499ec6a97f1aa8edc8d2a93333",
    "scripts/gen5_v10_bounded_worker_export_candidate.py": "cbfc6004c6ab177eea8e7def1cc001049ec4bc4e",
    "scripts/gen5_v10_e0e3_v2_bundle_verifier_candidate.py": "674f8a22b0b9ddaa51f60ff126c0f152c0b49e31",
}


class V3Blocked(ValueError):
    pass


def demand(ok: bool, code: str) -> None:
    if not ok:
        raise V3Blocked("GEN5_V10_V3_BLOCKED:" + code)


def canonical(obj: Any) -> bytes:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise V3Blocked("NONCANONICAL_JSON") from exc


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def validhex(x: Any, length: int = 64) -> bool:
    return type(x) is str and (HEX64 if length == 64 else HEX40).fullmatch(x) is not None


def number(x: Any, loc: str) -> float:
    demand(type(x) in (int, float) and math.isfinite(x), "NONFINITE:" + loc)
    return float(x)


def vector(x: Any, loc: str) -> list[float]:
    demand(type(x) is list and len(x) == 127, "VECTOR_GEOMETRY:" + loc)
    return [number(v, f"{loc}:{i}") for i, v in enumerate(x)]


def eq(x: float, y: float) -> tuple[bool, float, float]:
    diff = abs(x-y)
    rel = diff/max(abs(x), EPSILON)
    return (math.isfinite(diff) and math.isfinite(rel) and
            (diff <= ATOL or rel <= RTOL), diff, rel)


def validate_record(row: Any, loc: str) -> dict[str, Any]:
    demand(type(row) is dict and set(row) == RECORD_FIELDS, "RECORD_SCHEMA:"+loc)
    for k in SCALARS:
        number(row[k], loc+"."+k)
    s = number(row["S"], loc+".S")
    demand(s > 0, "SELF_ENERGY:"+loc)
    vector(row["pair_gap_C"], loc+".pair_gap_C")
    coarse = row["region_pair_C_by_gap"]
    fine = row["fine_pair_C_by_gap"]
    demand(type(coarse) is dict and set(coarse) == set(REGIONS), "COARSE_KEYS:"+loc)
    demand(type(fine) is dict and set(fine) == set(PARENTS), "FINE_PARENTS:"+loc)
    for key in REGIONS:
        vector(coarse[key], loc+".coarse."+key)
    maxima = [0.0, 0.0]
    for parent in PARENTS:
        cells = fine[parent]
        demand(type(cells) is dict and set(cells) == set(FINE), "FINE_CELLS:"+loc+parent)
        summed = [0.] * 127
        for name in FINE:
            vals = vector(cells[name], loc+".fine."+name)
            for i, v in enumerate(vals):
                summed[i] += v
        for i, (raw_e, raw_o) in enumerate(zip(coarse[parent], summed)):
            e, o = raw_e / s, raw_o / s
            demand(math.isfinite(e) and math.isfinite(o), "NORMALIZED_NONFINITE:"+loc)
            passed, abs_err, rel_err = eq(e, o)
            demand(passed, f"FROZEN_PARENT_REPLAY:{loc}:{parent}:{i+1}")
            maxima[0] = max(maxima[0], abs_err)
            maxima[1] = max(maxima[1], rel_err)
    for key, expected in (("fine_parent_reconstruction_abs_max", maxima[0]),
                          ("fine_parent_reconstruction_rel_max", maxima[1])):
        observed = number(row[key], loc+"."+key)
        demand(math.isclose(observed, expected, rel_tol=1e-12, abs_tol=0.0),
               "DIAGNOSTIC_FORGERY:"+loc+"."+key)
    return {"status":"RECOMPUTED_FROM_UNAUTHENTICATED_RECORDS", "abs_max":maxima[0],
            "rel_max":maxima[1]}


def _compare(a: Any, b: Any, path: str, stats: dict[str, Any]) -> None:
    if type(a) is dict:
        demand(type(b) is dict and set(a) == set(b), "PAYLOAD_KEYS:"+path)
        for name in sorted(a):
            _compare(a[name], b[name], path+"."+name, stats)
    elif type(a) is list:
        demand(type(b) is list and len(a) == len(b), "PAYLOAD_LIST:"+path)
        for i, (x,y) in enumerate(zip(a,b)):
            _compare(x,y,path+f"[{i}]",stats)
    elif type(a) in (int, float) and type(b) in (int, float):
        if type(a) is int or type(b) is int:
            demand(type(a) is int and type(b) is int and a == b,"INTEGER_DRIFT:"+path)
            stats["integers"] += 1
        else:
            x, y = number(a,path), number(b,path)
            passed, abs_err, rel_err = eq(x,y)
            stats["raw_numeric"] += 1
            if not passed:
                stats["legacy_failures"] += 1
                kind = "DIAGNOSTIC" if path.endswith(".fine_parent_reconstruction_rel_max") else "PAYLOAD"
                stats["diagnostic_failures" if kind == "DIAGNOSTIC" else "payload_failures"] += 1
                if len(stats["first_failures"]) < 20:
                    stats["first_failures"].append({"path":path,"kind":kind,
                        "abs_error":abs_err,"rel_error":rel_err})
    else:
        demand(type(a) is type(b) and a == b,"NONNUMERIC_DRIFT:"+path)


def _normalized(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"H_by_gap":0,"H_windows":0,"failures":0}
    for fr, vr in zip(a,b):
        fs, vs = float(fr["S"]), float(vr["S"])
        for parent in PARENTS:
            for cell in FINE:
                x = [v/fs for v in fr["fine_pair_C_by_gap"][parent][cell]]
                y = [v/vs for v in vr["fine_pair_C_by_gap"][parent][cell]]
                for e,o in zip(x,y):
                    stats["H_by_gap"] += 1
                    if not eq(e,o)[0]: stats["failures"] += 1
                for lo,hi in WINDOWS:
                    stats["H_windows"] += 1
                    if not eq(sum(x[lo-1:hi]),sum(y[lo-1:hi]))[0]:
                        stats["failures"] += 1
    demand(stats["H_by_gap"] == 38100 and stats["H_windows"] == 1500,
           "NORMALIZED_COVERAGE")
    return stats


def _capture(bundle: Any, *, source_kind: str) -> dict[str, Any]:
    from scripts import gen5_v10_bounded_tensor_provenance_candidate as stage_a
    demand(type(bundle) is dict and set(bundle) == set(CAPTURE_STAGES),"CAPTURE_PHASE_KEYS")
    first = None
    geometry = None
    for stage in CAPTURE_STAGES:
        part = bundle[stage]
        demand(type(part) is dict and set(part) == set(stage_a.TENSOR_KEYS),"CAPTURE_TENSOR_KEYS")
        for key in stage_a.TENSOR_KEYS:
            t = part[key]
            demand(type(t) is dict and set(t) == CAPTURE_FIELDS,"CAPTURE_ENTRY:"+key)
            expected_phase = {"pre_frozen":"PRE_FROZEN", "post_frozen":"POST_FROZEN",
                              "post_v10":"POST_V10"}[stage]
            demand(t["phase"] == expected_phase and t["synchronized"] is True and
                   t["V10_E0_PROVENANCE"] == "NOT_EVALUATED", "CAPTURE_CLAIM:"+key)
            expected_dtype = "float32" if key in ("component", "log_a", "survival") else "bool"
            demand(t["dtype"] == expected_dtype and type(t["shape"]) is list and
                   len(t["shape"]) == (4 if key in ("component","log_a") else
                                        3 if key == "survival" else 2) and
                   all(type(x) is int and x > 0 for x in t["shape"]),"CAPTURE_GEOMETRY:"+key)
            demand(type(t["source_strides"]) is list and
                   len(t["source_strides"]) == len(t["shape"]) and
                   all(type(x) is int for x in t["source_strides"]), "CAPTURE_STRIDES:"+key)
            demand(validhex(t["content_sha256"]) and type(t["byte_count"]) is int and
                   t["byte_count"] == math.prod(t["shape"]) * (4 if expected_dtype == "float32" else 1),
                   "CAPTURE_HASH_SIZE:"+key)
            demand(type(t["chunk_cap_bytes"]) is int and
                   4 <= t["chunk_cap_bytes"] <= 8*1024*1024,"CAPTURE_CHUNK_CAP:"+key)
            demand(t["source_device"] == ("cpu" if source_kind == "CPU_MOCK_ONLY" else "cuda:0"),
                   "CAPTURE_SOURCE_KIND:"+key)
            bind = (t["shape"],t["dtype"],t["content_sha256"],t["byte_count"],
                    t["source_device"])
            if first is None: first = {}
            if stage == "pre_frozen": first[key] = bind
            else: demand(first[key] == bind,"INPUT_VALUE_DRIFT:"+key)
        shape = part["component"]["shape"]
        demand(shape[:3] == [4,32,128],"CAPTURE_KBS")
        width = shape[3]
        demand(width == 24576 if source_kind == "RUNTIME_CLAIM_UNAUTHENTICATED"
               else width > 0 and width != 24576,"CAPTURE_SOURCE_WIDTH")
        b, s = 32, 128
        demand(part["survival"]["shape"] == [b,s,width],"CAPTURE_SURVIVAL_SHAPE")
        demand(part["attention_mask"]["shape"] == [b,s],"CAPTURE_ATTENTION_SHAPE")
        for key in stage_a.TENSOR_KEYS:
            if key.startswith(("role_masks.","fine_masks.")):
                demand(part[key]["shape"] == [b,s],"CAPTURE_MASK_SHAPE:"+key)
        la = part["log_a"]["shape"]
        demand(la[0] == b and la[1] == s and la[2]*la[3] == width,"CAPTURE_LOG_SHAPE")
        geometry = {"K":4,"B":32,"S":128,"W":width}
    return geometry


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    """Only a synthetic helper for CPU fixtures. Never source attestation."""
    demand(type(payload) is dict and "bundle_sha256" not in payload and
           "sidecar" not in payload,"SEAL_FIELDS")
    value = dict(payload)
    body = digest(value)
    sidecar_payload = {"domain":"GEN5_V10_V3_SIDECAR_V1","bundle_sha256":body}
    sidecar_hash = hashlib.sha256(SIDECAR_DOMAIN + canonical(sidecar_payload)).hexdigest()
    value["bundle_sha256"] = body
    value["sidecar"] = dict(sidecar_payload, sidecar_sha256=sidecar_hash)
    return value


def validate(bundle: Any) -> dict[str, Any]:
    demand(type(bundle) is dict and set(bundle) == ROOT_FIELDS,"ROOT_SCHEMA")
    demand(bundle["schema_version"] == SCHEMA and bundle["candidate_base"] == BASE,
           "SCHEMA_OR_BASE")
    demand(validhex(bundle["bundle_sha256"]),"BUNDLE_HASH_FORMAT")
    nohash = {k:v for k,v in bundle.items() if k not in ("bundle_sha256","sidecar")}
    demand(bundle["bundle_sha256"] == digest(nohash),"BUNDLE_HASH")
    sidecar = bundle["sidecar"]
    demand(type(sidecar) is dict and set(sidecar) ==
           {"domain","bundle_sha256","sidecar_sha256"} and
           sidecar["domain"] == "GEN5_V10_V3_SIDECAR_V1" and
           sidecar["bundle_sha256"] == bundle["bundle_sha256"],"SIDECAR_SCHEMA")
    sidecar_without = {k:v for k,v in sidecar.items() if k != "sidecar_sha256"}
    demand(validhex(sidecar["sidecar_sha256"]) and sidecar["sidecar_sha256"] ==
           hashlib.sha256(SIDECAR_DOMAIN + canonical(sidecar_without)).hexdigest(),
           "SIDECAR_HASH")
    from scripts import gen5_v10_bounded_worker_export_candidate as stage_b
    locus = bundle["selector"]
    demand(type(locus) is dict,"SELECTOR_TYPE")
    demand(type(locus.get("row_ids")) is list,"SELECTOR_ROWS")
    try:
        authenticated_locus = stage_b.choose_bounded_locus(
        worker_id=locus.get("worker_id"),
        batch_index=locus.get("batch_index_one_based"),
        row_start=locus.get("row_start"),row_stop=locus.get("row_stop"),
        row_ids=locus["row_ids"],source_cells=stage_b.SOURCE_CELLS,
        orientations=stage_b.FROZEN_ORIENTATIONS,
        selected_source=stage_b.SOURCE_CELLS[0],
        selected_targets=(stage_b.SOURCE_CELLS[1],stage_b.SOURCE_CELLS[2]),
        target_group_index=locus.get("target_group_index"),
        )
    except stage_b.StageBBlocked as exc:
        raise V3Blocked("GEN5_V10_V3_BLOCKED:SELECTOR:" + str(exc)) from exc
    demand(locus == authenticated_locus,"SELECTOR_IDENTITY")
    code = bundle["code_identity"]
    demand(type(code) is dict and set(code) == {"branch","frozen_base","code_head",
           "frozen_blobs","identity_status"},"CODE_ID_FIELDS")
    demand(code["branch"] == "gen5-causal-role-state-ownership" and
           code["frozen_base"] == BASE and code["frozen_blobs"] == FROZEN_BLOBS,
           "CODE_IDENTITY_FALSE_ATTESTATION")
    runtime = bundle["runtime_claims"]
    demand(type(runtime) is dict and set(runtime) == {"source_kind","execution_status",
           "physical_device","model_revision","checkpoint_sha256",
           "dev_order_sha256","active_encoding_sha256","torch_version",
           "cuda_runtime","cublas_version","tf32","determinism"},"RUNTIME_FIELDS")
    kind = runtime["source_kind"]
    demand(kind in ("CPU_MOCK_ONLY", "RUNTIME_CLAIM_UNAUTHENTICATED"),"SOURCE_KIND")
    demand(runtime["execution_status"] == ("NO" if kind == "CPU_MOCK_ONLY" else "UNVERIFIED"),
           "EXECUTION_CLAIM")
    demand(runtime["tf32"] is False, "TF32_CLAIM")
    if kind == "CPU_MOCK_ONLY":
        demand(code["code_head"] == "NOT_AUTHENTICATED" and
               code["identity_status"] == "NOT_AUTHENTICATED",
               "CODE_IDENTITY_FALSE_ATTESTATION")
        demand(runtime["physical_device"] == "CPU" and
               runtime["determinism"] == "NOT_AUTHENTICATED",
               "PHYSICAL_DEVICE_CLAIM")
        for name in ("model_revision","checkpoint_sha256","dev_order_sha256",
                     "active_encoding_sha256","torch_version","cuda_runtime","cublas_version"):
            demand(runtime[name] == "NOT_AUTHENTICATED", "RUNTIME_FALSE_ATTEST:"+name)
    else:
        # Structurally inspect an UNAUTHENTICATED runtime claim, never certify it.
        # Actual E0 requires an independent, commit-pinned execution authority.
        demand(validhex(code["code_head"],40) and code["code_head"] != BASE and
               code["identity_status"] == "SELF_ASSERTED_NOT_AUTHENTICATED",
               "RUNTIME_CODE_NOT_VERIFIED")
        demand(runtime["model_revision"] ==
               "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37" and
               runtime["checkpoint_sha256"] ==
               "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f" and
               runtime["dev_order_sha256"] ==
               "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25" and
               runtime["active_encoding_sha256"] ==
               "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51",
               "RUNTIME_PINNED_SOURCE_CLAIMS")
        demand(type(runtime["physical_device"]) is str and
               bool(runtime["physical_device"].strip()) and
               runtime["physical_device"] not in ("CPU","NOT_AUTHENTICATED") and
               runtime["determinism"] in ("ENABLED_CLAIM", "DISABLED_CLAIM") and
               all(type(runtime[key]) is str and bool(runtime[key].strip()) and
                   runtime[key] != "NOT_AUTHENTICATED" for key in
                   ("torch_version","cuda_runtime","cublas_version")),
               "RUNTIME_ENVIRONMENT_CLAIMS")
    geo = _capture(bundle["input_capture"],source_kind=kind)
    engines = bundle["engines"]
    demand(type(engines) is dict and set(engines) == {"frozen","v10"},"ENGINE_SET")
    diagnostics = {}; rows = {}
    for engine in ("frozen","v10"):
        item = engines[engine]
        demand(type(item) is dict and set(item) == {"engine_id","records","records_sha256",
                                                "dispatch","guard_status"},"ENGINE_FIELDS")
        demand(item["engine_id"] == ("FROZEN_FFT_V1" if engine == "frozen" else
                                    "V10_HYBRID_CANDIDATE"),"ENGINE_ID")
        rows[engine] = item["records"]
        demand(type(rows[engine]) is list and len(rows[engine]) == 4,"RECORD_K4")
        demand(validhex(item["records_sha256"]) and
               item["records_sha256"] == digest(rows[engine]),"RECORD_HASH")
        diagnostics[engine] = [validate_record(row,engine+f"[{i}]")
                               for i,row in enumerate(rows[engine])]
        demand(item["guard_status"] == "UNAUTHENTICATED_RUNTIME_CLAIM",
               "GUARD_FALSE_CLAIM")
        route = item["dispatch"]
        demand(type(route) is dict,"ROUTE_SCHEMA")
        if engine == "frozen":
            demand(route == {"mode":"FROZEN_WHOLE","batch_rows":32},"FROZEN_ROUTE")
        else:
            demand(set(route) == {"mode","batch_rows","fast_chunks","gemm_chunks",
                                  "frozen_tail_calls","total_chunks","max_half_span"},
                   "V10_ROUTE_FIELDS")
            demand(route["mode"] == "CHUNKWISE_HYBRID" and
                   type(route["batch_rows"]) is int and route["batch_rows"] == 32 and
                   type(route["frozen_tail_calls"]) is int and route["frozen_tail_calls"] == 0,
                   "V10_ROUTE_MODE")
            for name in ("fast_chunks","gemm_chunks","total_chunks"):
                demand(type(route[name]) is int and route[name] >= 0,"V10_CHUNK_COUNT")
            width_chunk = 4 if kind == "CPU_MOCK_ONLY" else 256
            demand(route["total_chunks"] == math.ceil(geo["W"]/width_chunk) and
                   route["fast_chunks"]+route["gemm_chunks"] == route["total_chunks"],
                   "V10_ROUTE_COVERAGE")
            demand(number(route["max_half_span"],"V10_HALF_SPAN") >= 0.0,
                   "V10_HALF_SPAN_NEGATIVE")
    guards = bundle["guard_trace"]
    demand(type(guards) is dict and set(guards) == set(GUARD_KEYS),"GUARD_COVERAGE")
    demand(all(v in ("OBSERVED_CPU_MOCK", "NOT_AUTHENTICATED") for v in guards.values()),
           "GUARD_FALSE_ATTEST")
    failures = bundle["failure_evidence"]
    demand(type(failures) is list and all(type(x) is str and bool(x) for x in failures),
           "FAILURE_EVIDENCE")
    demand(bundle["evidence_status"] == "UNAUTHENTICATED_BOUNDED_RECORDS_NO_PROMOTION",
           "EVIDENCE_STATUS")
    stats = {"raw_numeric":0,"integers":0,"payload_failures":0,"diagnostic_failures":0,
             "legacy_failures":0,"first_failures":[]}
    _compare(rows["frozen"],rows["v10"],"records",stats)
    normalized = _normalized(rows["frozen"],rows["v10"])
    payload_fail = bool(stats["payload_failures"] or normalized["failures"])
    return {
        "schema_version":REPORT_SCHEMA,"candidate_base":BASE,
        "bundle_sha256":bundle["bundle_sha256"], "bounded_selector":locus,
        "geometry":geo,"input_source_authenticated":False,
        "recomputed_diagnostics":diagnostics,"raw_comparison":stats,
        "normalized_H":normalized,
        "LEGACY_FULL_RECORD":"FAIL" if stats["legacy_failures"] else "PASS",
        "V10_E0_PROVENANCE":"NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS":"FAIL" if failures else "NOT_EVALUATED",
        "V10_E2_PAYLOAD":"FAIL" if payload_fail else "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS":"NOT_EVALUATED",
        "V10_PAYLOAD_EQUIVALENCE":"FAIL" if payload_fail or failures else "NOT_EVALUATED",
        "V10_E2_BOUNDED_NUMERIC":"FAIL" if payload_fail else "PASS_PROVISIONAL",
        "V10_E3_BOUNDED_RAW":"DIVERGENCE" if stats["diagnostic_failures"] else "AGREEMENT",
        "population_scope":"NOT_EVALUATED",
        "missing_evidence":["source_tensor_bytes_authenticated","runtime_code_head_authenticated",
                            "frozen_model_tokenizer_checkpoint_lineage",
                            "all_original_upstream_guards_authenticated",
                            "gpu_bounded_execution_gate"],
        "PRODUCTION":"BLOCKED","SCIENTIFIC_EXECUTION":"NOT_AUTHORIZED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-contract-check",action="store_true",required=True)
    args = parser.parse_args()
    demand(os.environ.get("CUDA_VISIBLE_DEVICES","") in ("","-1"),"GPU_NOT_DISABLED")
    print("GEN5_V10_STAGE_C_V3_VERIFIER_CPU_STATIC_CONTRACT_PASS")
    print("SCHEMA="+SCHEMA+";VERIFICATION="+REPORT_SCHEMA)
    print("GPU=OFF;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;PRODUCTION=BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
