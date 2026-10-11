#!/usr/bin/env python3
"""Gen5 v10 bounded CPU E0-E3 *candidate* record verifier.

Inputs are independently exported four-component records from the frozen and
candidate engines, NOT the v7/v8 summary logs. All JSON values are inspected.
This tool cannot execute CUDA, authenticate stored input tensors, or authorize
production/scientific execution. Run --self-test for synthetic contract tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "GEN5_SHARED_GEMM_CPU_EQUIVALENCE_DRAFT_V1" # frozen export input schema
VERIFIER_VERSION = "GEN5_V10_E0E3_CPU_RECORD_CANDIDATE_V1"
REPO_HEAD = "14608d234f6c0e1e8d7cd446b3e384cf77fc86da"
HEAD = "e145d7d22b256c4ce5854939148a07d972bb75c8"
BLOBS = {
    "scripts/audit_native_recurrent_generator_anchor_pair_localization.py": "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635",
    "tests/test_native_recurrent_generator_anchor_pair_localization.py": "26c90ada9a74a2ac2716c880e58d15df5bd01704",
    "scripts/gen5_v10_hybrid_fine_pair_candidate.py": "56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0",
    "scripts/gen5_v10_shared_dyadic_gemm_candidate.py": "9003fb9bd23b2a6680ef210a51845058622badd2",
    "tests/test_gen5_v10_hybrid_candidate.py": "9d329222bae47d3fe282619ff7de982863a35144",
    "reports/native_recurrent_generator_anchor_pair_localization_v10_numeric_contract_and_candidate_freeze_spec.md": "661c627f09034c34f9830b18b7e9938c72c05925",
}
PARENTS = ("CLAIM->CLAIM", "EVIDENCE->EVIDENCE", "CLAIM->EVIDENCE")
CLASSES = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE", "RESIDUAL")
FINE = tuple(f"{a}->{b}" for a in CLASSES for b in CLASSES)
ATOL = 2.0e-5
RTOL = 2.0e-5
EPSILON = 1.0e-30
GAPS = 127
EXPECTED_EXPORTED_NUMERIC_FIELDS = 41204
REGION_PAIRS = ("CLAIM->CLAIM", "CLAIM->EOS", "CLAIM->EVIDENCE",
                "EOS->EVIDENCE", "EVIDENCE->EVIDENCE")
EXPORTED_ROW_FIELDS = frozenset((
    "R", "S", "E", "C", "Q", "pair_gap_C",
    "pair_gap_C_reconstruction_abs", "pair_gap_Q_reconstruction_abs",
    "pair_gap_fft_half_log_span_max", "region_pair_C_by_gap",
    "region_pair_gap_reconstruction_abs_max", "region_pair_gap_reconstruction_rel_max",
    "region_pair_fft_half_log_span_max", "fine_pair_C_by_gap",
    "fine_parent_reconstruction_abs_max", "fine_parent_reconstruction_rel_max",
    "fine_pair_fft_half_log_span_max",
))
K = 4
DIAGNOSTIC = "fine_parent_reconstruction_rel_max"


class ContractBlocked(ValueError):
    pass


def demand(ok: bool, reason: str) -> None:
    if not ok:
        raise ContractBlocked(reason)


def finite_number(value: Any, where: str) -> float:
    demand(type(value) in (float, int), f"NOT_NUMERIC:{where}")
    demand(math.isfinite(value), f"NONFINITE:{where}")
    return float(value)


def vector(value: Any, where: str) -> list[float]:
    demand(type(value) is list and len(value) == GAPS, f"VECTOR_GEOMETRY:{where}")
    return [finite_number(x, f"{where}[{i}]") for i, x in enumerate(value)]


def agreement(x: float, y: float) -> tuple[bool, float, float]:
    diff = abs(y - x)
    relative = diff / max(abs(x), EPSILON)
    return diff <= ATOL or relative <= RTOL, diff, relative


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_identity(repo: Path, expected_repo_head: str) -> dict[str, Any]:
    demand(isinstance(expected_repo_head, str) and len(expected_repo_head) == 40
           and all(ch in "0123456789abcdef" for ch in expected_repo_head),
           "EXPECTED_REPO_HEAD_FULL_SHA_REQUIRED")
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
        demand(repo.resolve() == top, "REPO_PATH_NOT_WORKTREE_ROOT")
        demand(git("rev-parse", "HEAD") == expected_repo_head, "VERSIONED_REPO_HEAD_MISMATCH")
        demand(git("merge-base", "--is-ancestor", REPO_HEAD, "HEAD") == "",
               "FROZEN_CANDIDATE_BASE_NOT_ANCESTOR")
        demand(not git("status", "--porcelain"), "FROZEN_WORKTREE_DIRTY")
        demand(git("branch", "--show-current") in ("gen5-causal-role-state-ownership", ""),
               "FROZEN_BRANCH_MISMATCH")
        for path, expected in BLOBS.items():
            demand(git("rev-parse", "HEAD:" + path) == expected, "FROZEN_BLOB_MISMATCH:" + path)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ContractBlocked(f"REPO_GIT_READ_FAILED:{type(exc).__name__}") from exc
    return {"status": "PASS_READ_ONLY", "head": expected_repo_head, "candidate_base": REPO_HEAD, "blob_count": len(BLOBS)}


def exported_schema_check(records: Any, engine: str) -> None:
    demand(type(records) is list and len(records) == K, f"EXPORTED_K:{engine}")
    for i, rec in enumerate(records):
        where = f"{engine}[{i}]"
        demand(type(rec) is dict and set(rec) == EXPORTED_ROW_FIELDS,
               f"EXPORTED_EXACT_ROW_FIELDS:{where}")
        vector(rec["pair_gap_C"], where + ".pair_gap_C")
        region = rec["region_pair_C_by_gap"]
        demand(type(region) is dict and set(region) == set(REGION_PAIRS),
               f"EXPORTED_REGION_PARENT_KEYS:{where}")
        for parent in REGION_PAIRS:
            vector(region[parent], where + ".region_pair_C_by_gap." + parent)
        for name in EXPORTED_ROW_FIELDS - {"pair_gap_C", "region_pair_C_by_gap", "fine_pair_C_by_gap"}:
            finite_number(rec[name], where + "." + name)


def actual_replay(records: Any, engine: str) -> list[dict[str, Any]]:
    """Replicate frozen fine-parent *H* replay and diagnostic extrema from C vectors.

    The C vectors are 127 gaps (1..127). Frozen uses sequential float64 addition
    over its 25 ordered fine pairs, then normalizes both sides by S.
    """
    demand(type(records) is list and len(records) == K, f"K_COUNT:{engine}")
    results = []
    for i, record in enumerate(records):
        loc = f"{engine}[{i}]"
        demand(type(record) is dict, f"RECORD_NOT_DICT:{loc}")
        for field in ("S", "fine_pair_C_by_gap", "region_pair_C_by_gap",
                      "fine_parent_reconstruction_abs_max", DIAGNOSTIC):
            demand(field in record, f"REQUIRED_FIELD_MISSING:{loc}.{field}")
        s = finite_number(record["S"], loc + ".S")
        demand(s > 0.0, f"NONPOSITIVE_SELF_ENERGY:{loc}")
        fine = record["fine_pair_C_by_gap"]
        coarse = record["region_pair_C_by_gap"]
        demand(type(fine) is dict and set(fine) == set(PARENTS), f"FINE_PARENTS:{loc}")
        demand(type(coarse) is dict and set(PARENTS) <= set(coarse), f"COARSE_PARENTS:{loc}")
        max_abs, max_rel = 0.0, 0.0
        parents = []
        for parent in PARENTS:
            item = fine[parent]
            demand(type(item) is dict and set(item) == set(FINE), f"FINE_CLASSES:{loc}.{parent}")
            vectors = [vector(item[name], f"{loc}.{parent}.{name}") for name in FINE]
            expected = vector(coarse[parent], f"{loc}.coarse.{parent}")
            combined = [0.0] * GAPS
            for value in vectors:
                for gap_idx, elem in enumerate(value):
                    combined[gap_idx] += elem  # same fixed fine-cell order as frozen
            pm_abs, pm_rel = 0.0, 0.0
            worst_abs, worst_rel = None, None
            for gap_idx, (obs_c, exp_c) in enumerate(zip(combined, expected)):
                obs_h = obs_c / s
                exp_h = exp_c / s
                passed, abs_error, rel_error = agreement(exp_h, obs_h)
                demand(passed, f"FROZEN_PER_GAP_REPLAY_FAIL:{loc}:{parent}:gap{gap_idx + 1}:"
                       f"abs={abs_error}:rel={rel_error}")
                if abs_error > pm_abs:
                    pm_abs = abs_error
                    worst_abs = {"gap": gap_idx + 1, "absolute_error": abs_error}
                if rel_error > pm_rel:
                    pm_rel = rel_error
                    worst_rel = {
                        "gap": gap_idx + 1, "expected_h": exp_h,
                        "observed_h": obs_h, "absolute_error": abs_error,
                        "relative_error": rel_error,
                        "near_zero_denominator": abs(exp_h) < ATOL,
                    }
            max_abs = max(max_abs, pm_abs)
            max_rel = max(max_rel, pm_rel)
            parents.append({"parent": parent, "max_abs": pm_abs,
                            "max_rel": pm_rel, "worst_abs": worst_abs, "worst_rel": worst_rel})
        # Truthfulness uses an independent re-computation of the *actual* emitted
        # values, not equality to the other engine's diagnostics.
        for key, actual in (("fine_parent_reconstruction_abs_max", max_abs),
                            (DIAGNOSTIC, max_rel)):
            claimed = finite_number(record[key], loc + "." + key)
            demand(math.isclose(claimed, actual, rel_tol=1e-12, abs_tol=0.0),
                   f"DIAGNOSTIC_NOT_TRUTHFUL:{loc}.{key}:recorded={claimed}:recomputed={actual}")
        results.append({"component_index": i, "max_abs": max_abs, "max_rel": max_rel,
                        "parents": parents})
    return results


def compare_all(frozen: Any, gemm: Any) -> dict[str, Any]:
    counts = {"numeric": 0, "exact_integer": 0, "identity": 0, "legacy_failed": 0, "payload_failed": 0,
              "diagnostic_noncomparable": 0, "max_payload_abs": 0.0,
              "max_payload_rel": 0.0, "sign_flip_near_zero": 0}
    mismatches: list[dict[str, Any]] = []

    def add_mismatch(path: str, status: str, a: Any, b: Any, diff: Any, rel: Any) -> None:
        mismatches.append({"path": path, "classification": status,
                           "frozen": a, "gemm": b, "abs_diff": diff, "relative_diff": rel})

    def walk(a: Any, b: Any, path: str) -> None:
        if type(a) is dict:
            demand(type(b) is dict and set(a) == set(b), f"RECORD_SCHEMA_KEYS:{path}")
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
        elif type(a) is list:
            demand(type(b) is list and len(a) == len(b), f"RECORD_SCHEMA_SEQUENCE:{path}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif type(a) in (int, float) and type(b) in (int, float):
            # Scientific count / ID fields must not drift, even by one.
            if type(a) is int or type(b) is int:
                demand(type(a) is int and type(b) is int and a == b,
                       f"EXACT_INTEGER_MISMATCH:{path}:{a}:{b}")
                counts["identity"] += 1
                counts["exact_integer"] += 1
                return
            x, y = finite_number(a, path), finite_number(b, path)
            ok, diff, rel = agreement(x, y)
            counts["numeric"] += 1
            special = (path in {f"root[{i}].{DIAGNOSTIC}" for i in range(K)})
            if not ok:
                counts["legacy_failed"] += 1
                if special:
                    counts["diagnostic_noncomparable"] += 1
                    add_mismatch(path, "DIAGNOSTIC_NONCOMPARABLE", a, b, diff, rel)
                else:
                    counts["payload_failed"] += 1
                    add_mismatch(path, "PAYLOAD_MISMATCH", a, b, diff, rel)
            if not special:
                counts["max_payload_abs"] = max(counts["max_payload_abs"], diff)
                counts["max_payload_rel"] = max(counts["max_payload_rel"], rel)
                if x != 0.0 and y != 0.0 and (x < 0.0) != (y < 0.0):
                    counts["sign_flip_near_zero"] += 1
        else:
            demand(type(a) is type(b) and a == b, f"EXACT_METADATA_MISMATCH:{path}")
            counts["identity"] += 1

    walk(frozen, gemm, "root")
    return {"counts": counts, "mismatches": mismatches,
            "E2_PAYLOAD": "PASS_PROVISIONAL" if not counts["payload_failed"] else "FAIL",
            "E3_CROSS_ENGINE_DIAGNOSTIC": (
                "DIAGNOSTIC_NONCOMPARABLE" if counts["diagnostic_noncomparable"] else "AGREEMENT"
            ),
            "LEGACY_FULL_RECORD": "FAIL" if counts["legacy_failed"] else "PASS"}


def evaluate(bundle: Any, *, repo: Path | None = None, expected_repo_head: str | None = None) -> dict[str, Any]:
    demand(type(bundle) is dict and set(bundle) == {
        "schema_version", "expected_head", "geometry", "frozen_records", "gemm_records"
    }, "BUNDLE_TOPLEVEL_SCHEMA")
    demand(bundle["schema_version"] == SCHEMA, "BUNDLE_SCHEMA_VERSION")
    demand(bundle["expected_head"] == HEAD, "BUNDLE_HEAD")
    g = bundle["geometry"]
    demand(type(g) is dict and set(g) == {"k", "gap_count", "batch_rows", "source_group",
                                             "worker_id", "batch_index", "evidence_kind"},
           "GEOMETRY_SCHEMA")
    demand(type(g["k"]) is int and g["k"] == K, "GEOMETRY_K")
    demand(type(g["gap_count"]) is int and g["gap_count"] == GAPS, "GEOMETRY_GAPS")
    demand(type(g["batch_rows"]) is int and g["batch_rows"] in (8, 32), "GEOMETRY_BATCH")
    demand(type(g["worker_id"]) is int and g["worker_id"] in (0, 1), "GEOMETRY_WORKER")
    demand(type(g["batch_index"]) is int and g["batch_index"] >= 0, "GEOMETRY_BATCH_INDEX")
    demand(type(g["source_group"]) is str and bool(g["source_group"].strip()), "GEOMETRY_SOURCE_GROUP")
    demand(g["evidence_kind"] in ("SYNTHETIC_CPU", "EXPORTED_BOUNDED_RECORDS"),
           "GEOMETRY_EVIDENCE_KIND")
    # A bundle's provenance fields are assertions by the exporter; the repo
    # authentication is independently read-only but cannot authenticate tensors.
    demand((repo is None and expected_repo_head is None) or
           (repo is not None and expected_repo_head is not None),
           "REPO_AND_EXPECTED_HEAD_MUST_BE_TOGETHER")
    ident = repo_identity(repo, expected_repo_head) if repo is not None else {"status": "NOT_AUTHENTICATED"}
    if g["evidence_kind"] == "EXPORTED_BOUNDED_RECORDS":
        exported_schema_check(bundle["frozen_records"], "frozen")
        exported_schema_check(bundle["gemm_records"], "gemm")
    baseline = actual_replay(bundle["frozen_records"], "frozen")
    candidate = actual_replay(bundle["gemm_records"], "gemm")
    compare = compare_all(bundle["frozen_records"], bundle["gemm_records"])
    if g["evidence_kind"] == "EXPORTED_BOUNDED_RECORDS":
        actual_count = compare["counts"]["numeric"] + compare["counts"]["exact_integer"]
        demand(actual_count == EXPECTED_EXPORTED_NUMERIC_FIELDS,
               f"EXPORTED_RECORD_NUMERIC_FIELD_COUNT:{actual_count}")
    return {
        "contract": SCHEMA, "status": "CPU_RECORD_DIAGNOSTIC_ONLY_NOT_APPROVED",
        "E0_REPO": ident, "E0_RECORD_SOURCE": "UNAUTHENTICATED_INPUT_BYTES",
        "E1_FINE_PARENT_REPLAY": "PASS", "E1_OTHER_FROZEN_GUARDS": "NOT_EVALUATED",
        "E3_DIAGNOSTIC_TRUTH": "PASS", "compare": compare,
        "replay": {"frozen": baseline, "gemm": candidate},
        "PROMOTION": "BLOCKED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
    }



WINDOWS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 127))


def derived_normalized_payload(frozen: Any, gemm: Any) -> dict[str, Any]:
    """Compare recomputed fine H(d) and five windows, never proxy raw records."""
    counts = {"H_by_gap": 0, "H_windows": 0, "failed": 0, "max_abs": 0.0}
    failures = []
    for i, (f, g) in enumerate(zip(frozen, gemm)):
        fs = finite_number(f["S"], f"frozen[{i}].S")
        gs = finite_number(g["S"], f"gemm[{i}].S")
        demand(fs > 0 and gs > 0, f"DERIVED_S_NONPOSITIVE:{i}")
        for parent in PARENTS:
            for cell in FINE:
                fv = vector(f["fine_pair_C_by_gap"][parent][cell], f"frozen[{i}].{parent}.{cell}")
                gv = vector(g["fine_pair_C_by_gap"][parent][cell], f"gemm[{i}].{parent}.{cell}")
                fh = [v / fs for v in fv]
                gh = [v / gs for v in gv]
                for j, (a, b) in enumerate(zip(fh, gh)):
                    ok, delta, _ = agreement(a, b)
                    counts["H_by_gap"] += 1
                    counts["max_abs"] = max(counts["max_abs"], delta)
                    if not ok:
                        counts["failed"] += 1
                        if len(failures) < 20:
                            failures.append(f"component[{i}].{parent}.{cell}.H_by_gap[{j+1}]")
                for lo, hi in WINDOWS:
                    # Fixed frozen windows are inclusive 1-indexed gaps.
                    a, b = sum(fh[lo-1:hi]), sum(gh[lo-1:hi])
                    ok, delta, _ = agreement(a, b)
                    counts["H_windows"] += 1
                    counts["max_abs"] = max(counts["max_abs"], delta)
                    if not ok:
                        counts["failed"] += 1
                        if len(failures) < 20:
                            failures.append(f"component[{i}].{parent}.{cell}.H_window[{lo}_{hi}]")
    return {"counts": counts, "first_failure_paths": failures,
            "status": "PASS_BOUNDED_NUMERIC_ONLY" if not counts["failed"] else "FAIL"}


def evaluate_v10(bundle: Any, *, repo: Path | None = None, expected_repo_head: str | None = None) -> dict[str, Any]:
    """Strict numeric comparison without manufacturing unauthenticated E0/E1 passes."""
    old = evaluate(bundle, repo=repo, expected_repo_head=expected_repo_head)
    derived = derived_normalized_payload(bundle["frozen_records"], bundle["gemm_records"])
    old_compare = old["compare"]
    payload_bad = bool(old_compare["counts"]["payload_failed"] or derived["counts"]["failed"])
    diagnostic_diff = bool(old_compare["counts"]["diagnostic_noncomparable"])
    # GPU, tensor provenance, original upstream guards, full aggregation unavailable
    # in a standalone raw-record export. Do not call any of those PASS.
    return {
        "schema_version": VERIFIER_VERSION,
        "input_export_schema": SCHEMA,
        "current_repo_code_identity": old["E0_REPO"],
        "evidence_frozen_head_claim": bundle["expected_head"],
        "V10_E0_PROVENANCE": "NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
        "V10_E2_PAYLOAD": "FAIL" if payload_bad else "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
        "E3_RAW_NUMERIC_COMPARISON": "DIVERGENCE" if diagnostic_diff else "AGREEMENT",
        "V10_PAYLOAD_EQUIVALENCE": "NOT_EVALUATED" if not payload_bad else "FAIL",
        "LEGACY_FULL_RECORD": old_compare["LEGACY_FULL_RECORD"],
        "E1_RECONSTRUCTED_FINE_PARENT_FROM_RECORDS": old["E1_FINE_PARENT_REPLAY"],
        "E3_RECOMPUTED_DIAGNOSTIC_TRUTH": old["E3_DIAGNOSTIC_TRUTH"],
        "E3_DIAGNOSTIC_DIFFERENCE_OBSERVED": diagnostic_diff,
        "E3_NUMERIC_CAUSE_NOT_AUTHENTICATED": diagnostic_diff,
        "RAW_PAYLOAD_NUMERIC_BOUNDED": "FAIL" if payload_bad else "PASS_PROVISIONAL",
        "raw_full_comparison": old_compare,
        "derived_normalized_H_and_windows": derived,
        "FROZEN_RECORD_REPLAY_DETAILS": old["replay"]["frozen"],
        "V10_RECORD_REPLAY_DETAILS": old["replay"]["gemm"],
        "population_scope": "NOT_EVALUATED",
        "missing_evidence": ["source-tensor-byte-identity", "actual-all-frozen-upstream-guards",
                             "versioned-runner-provenance", "source-matched-orientation-population-aggregation"],
        "PRODUCTION": "BLOCKED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
    }


def synthetic_fixture() -> dict[str, Any]:
    """Tiny-value record fixture; NOT derived from any experiment."""
    def row(component: int, *, gemm: bool) -> dict[str, Any]:
        fine = {parent: {name: [0.0] * GAPS for name in FINE} for parent in PARENTS}
        coarse = {parent: [0.0] * GAPS for parent in PARENTS}
        for pi, parent in enumerate(PARENTS):
            gap = 59 + pi * 4  # 1-based
            coarse[parent][gap-1] = 2.562915613051649e-13 * (component + 1)
            fine[parent][FINE[0]][gap-1] = (
                0.0 if gemm else -2.941668544685262e-11 * (component + 1)
            )
        rec = {"component_index": component, "S": 1.0, "R": 2.0, "E": 3.0,
               "metadata": "CPU_SYNTHETIC_ONLY", "region_pair_C_by_gap": coarse,
               "fine_pair_C_by_gap": fine,
               "fine_parent_reconstruction_abs_max": 0.0,
               DIAGNOSTIC: 0.0}
        # Truth values will be explicitly calculated by a function which does
        # not inspect the recorded maxima, then assigned for synthetic tests.
        am, rm = 0.0, 0.0
        for parent in PARENTS:
            obs = fine[parent][FINE[0]]
            exp = coarse[parent]
            for x, y in zip(obs, exp):
                d = abs(x-y)
                am = max(am,d)
                rm = max(rm,d/max(abs(y),EPSILON))
        rec["fine_parent_reconstruction_abs_max"] = am
        rec[DIAGNOSTIC] = rm
        return rec
    return {"schema_version": SCHEMA, "expected_head": HEAD,
            "geometry": {"k": K, "gap_count": GAPS, "batch_rows": 32, "source_group": "TEST",
                         "worker_id": 0, "batch_index": 0, "evidence_kind": "SYNTHETIC_CPU"},
            "frozen_records": [row(i, gemm=False) for i in range(K)],
            "gemm_records": [row(i, gemm=True) for i in range(K)]}


def self_test() -> None:
    fixture = synthetic_fixture()
    output = evaluate(fixture)
    counts = output["compare"]["counts"]
    demand(output["E1_FINE_PARENT_REPLAY"] == "PASS", "TEST_REPLAY")
    demand(output["compare"]["E2_PAYLOAD"] == "PASS_PROVISIONAL", "TEST_PAYLOAD")
    demand(output["compare"]["LEGACY_FULL_RECORD"] == "FAIL", "TEST_LEGACY_MUST_FAIL")
    demand(counts["diagnostic_noncomparable"] == 4 and counts["payload_failed"] == 0,
           "TEST_DIAGNOSTIC_PATHS")
    cases = {}

    def blocked(name: str, change, expected: str) -> None:
        trial = copy.deepcopy(fixture)
        change(trial)
        try:
            evaluate(trial)
        except ContractBlocked as exc:
            demand(expected in str(exc), "WRONG_BLOCKER:" + name + ":" + str(exc))
        else:
            raise ContractBlocked("FAIL_OPEN_SELF_TEST:" + name)
        cases[name] = "BLOCKED_AS_EXPECTED"

    blocked("parent_gap_guard", lambda a: a["gemm_records"][0]
            ["fine_pair_C_by_gap"][PARENTS[0]][FINE[0]].__setitem__(7, 0.1),
            "FROZEN_PER_GAP_REPLAY_FAIL")
    blocked("diagnostic_forgery", lambda a: a["gemm_records"][0]
            .__setitem__(DIAGNOSTIC, a["frozen_records"][0][DIAGNOSTIC]),
            "DIAGNOSTIC_NOT_TRUTHFUL")
    blocked("missing_fine_cell", lambda a: a["gemm_records"][0]
            ["fine_pair_C_by_gap"][PARENTS[0]].pop(FINE[-1]), "FINE_CLASSES")
    blocked("wrong_gap_geometry", lambda a: a["gemm_records"][0]
            ["fine_pair_C_by_gap"][PARENTS[0]][FINE[0]].pop(), "VECTOR_GEOMETRY")
    blocked("nonfinite", lambda a: a["gemm_records"][0].__setitem__("R", float("nan")),
            "NONFINITE")
    blocked("int_id_mismatch", lambda a: a["gemm_records"][0].__setitem__("component_index", 999),
            "EXACT_INTEGER_MISMATCH")
    blocked("extra_field", lambda a: a["gemm_records"][0].__setitem__("unknown", 11),
            "RECORD_SCHEMA_KEYS")
    payload = copy.deepcopy(fixture)
    payload["gemm_records"][0]["R"] = 4.0
    result = evaluate(payload)
    demand(result["compare"]["E2_PAYLOAD"] == "FAIL", "PAYLOAD_NOT_DETECTED")
    cases["payload_mismatch"] = "DETECTED"
    v10_out = evaluate_v10(fixture)
    demand(v10_out["V10_E0_PROVENANCE"] == "NOT_EVALUATED", "NO_FAKE_E0_PASS")
    demand(v10_out["V10_E1_FROZEN_GUARDS"] == "NOT_EVALUATED", "NO_FAKE_E1_PASS")
    demand(v10_out["V10_E2_PAYLOAD"] == "NOT_EVALUATED", "NO_FAKE_E2_PASS")
    demand(v10_out["LEGACY_FULL_RECORD"] == "FAIL", "MUST_KEEP_LEGACY_FAIL")
    print("GEN5_V10_E0E3_CPU_RECORD_SELF_TEST_PASS")
    print(json.dumps({"synthetic_demo_status": output["status"],
                      "legacy": output["compare"]["LEGACY_FULL_RECORD"],
                      "payload": output["compare"]["E2_PAYLOAD"],
                      "diagnostic_noncomparable": counts["diagnostic_noncomparable"],
                      "negative_cases": cases, "cuda": False, "research_execution": False}, sort_keys=True))


def validate_report_destination(report: Path, repo: Path | None = None) -> None:
    """Prevent report writes into the source or authenticated Git worktree.

    Canonicalize the not-yet-created path to catch parent-directory symlinks.
    This is a safety check, not a substitute for Git status authentication.
    """
    target = report.resolve(strict=False)
    roots = (("SOURCE", Path(__file__).resolve().parents[1]), ("AUTHENTICATED", repo))
    for label, root in roots:
        if root is not None:
            canonical = root.resolve()
            demand(target != canonical and canonical not in target.parents,
                   f"REPORT_INSIDE_{label}_WORKTREE_FORBIDDEN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--bundle", type=Path)
    parser.add_argument("--repo", type=Path, help="optional read-only HEAD/blob/clean authentication")
    parser.add_argument("--expected-repo-head", help="full 40-char SHA of clean versioned-code commit (requires --repo)")
    parser.add_argument("--report", type=Path, help="write CPU-only JSON report outside frozen repository")
    args = parser.parse_args()
    try:
        if args.self_test:
            demand(args.repo is None and args.report is None and args.expected_repo_head is None,
                   "SELF_TEST_TAKES_NO_REPO_OR_REPORT")
            self_test()
            return 0
        assert args.bundle is not None
        data = args.bundle.read_bytes()
        bundle = json.loads(data.decode("utf-8"), parse_constant=lambda s: (_ for _ in ()).throw(ContractBlocked("NONFINITE_JSON:"+s)))
        result = evaluate_v10(bundle, repo=args.repo, expected_repo_head=args.expected_repo_head)
        result["input_sha256"] = sha256(data)
        if args.report is not None:
            validate_report_destination(args.report, repo=args.repo)
            demand(not args.report.exists(), "REPORT_EXISTS_NO_OVERWRITE")
            demand(args.report.parent.is_dir(), "REPORT_PARENT_MISSING")
            with args.report.open("x", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
                fh.write("\n")
        counts = result["raw_full_comparison"]["counts"]
        print("GEN5_V10_E0E3_CPU_RECORD_VERIFIER_CHECK_COMPLETE")
        print("INPUT_SHA256="+result["input_sha256"])
        print("E0="+result["V10_E0_PROVENANCE"]+";E1="+result["V10_E1_FROZEN_GUARDS"])
        print("E2="+result["V10_E2_PAYLOAD"]+";E3="+result["V10_E3_DIAGNOSTICS"])
        print("LEGACY_FULL_RECORD="+result["LEGACY_FULL_RECORD"])
        print("COUNTS="+json.dumps(counts, sort_keys=True))
        print("PROMOTION=BLOCKED;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED;E1_OTHER_FROZEN_GUARDS=NOT_EVALUATED")
        return 0 if result["RAW_PAYLOAD_NUMERIC_BOUNDED"] == "PASS_PROVISIONAL" else 2
    except (ContractBlocked, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"GEN5_GEMM_CPU_CONTRACT_BLOCKED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
