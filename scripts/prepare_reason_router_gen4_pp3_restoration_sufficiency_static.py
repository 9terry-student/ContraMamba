from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base
from scripts import build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh
from scripts import prepare_reason_router_gen4_pp3_necessity_static as prior

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
DESIGN_COMMIT = "854bcd5585512155776c34e46846817c54b75cf6"
DESIGN_PATH = Path("reports/reason_router_gen4_pp3_restoration_sufficiency_design.md")
DESIGN_BLOB = "f58931df229a09705edb94346c23c0dd9e67e799"

OLD_DIRS = [
    Path("data/reason_router_gen4_xg1_cross_generator_v1"),
    Path("data/reason_router_gen4_xg1_fresh_specificity_v1"),
    Path("data/reason_router_gen4_xg1_necessity_v1"),
]
OLD_601_SOURCE_SHA = "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd"
OLD_601_ROWS_SHA = "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224"

OUTPUT_DATA_DIR = Path("data/reason_router_gen4_xg1_restoration_sufficiency_v1")
OUTPUT_REPORT_DIR = Path("reports/reason_router_gen4_pp3_restoration_sufficiency_static_preparation_854bcd5")
PAIR_FIRST = "xg1_fact_901"
PAIR_LAST = "xg1_fact_1200"
PAIR_COUNT = 300
ROW_COUNT = 1800
TOL = 1.0e-12


class StaticPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError("GIT_FAILURE:" + " ".join(args)) from exc


def git_show_bytes(path: Path, ref: str = "HEAD") -> bytes:
    try:
        return subprocess.check_output(["git", "show", f"{ref}:{path.as_posix()}"], cwd=ROOT, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(f"GIT_SHOW_FAILURE:{ref}:{path.as_posix()}") from exc


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}")
            out.append(value)
    return out


def authenticate_design() -> dict[str, str]:
    branch = git("branch", "--show-current")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    head = git("rev-parse", "HEAD")
    rc = subprocess.call(["git", "merge-base", "--is-ancestor", DESIGN_COMMIT, head], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(rc == 0, "DESIGN_COMMIT_NOT_ANCESTOR")
    blob = git("rev-parse", f"HEAD:{DESIGN_PATH.as_posix()}")
    require(blob == DESIGN_BLOB, f"DESIGN_BLOB_DRIFT:{blob}")
    return {"branch": branch, "head": head, "design_commit": DESIGN_COMMIT, "design_blob": blob}


def prove_prior_population_identity() -> dict[str, Any]:
    identity = dict(prior.prove_generator_identity())
    d = OLD_DIRS[2]
    source_raw = git_show_bytes(d / base.SOURCE_FILE)
    rows_raw = git_show_bytes(d / base.ROW_FILE)
    require(sha256_bytes(source_raw) == OLD_601_SOURCE_SHA, "OLD_601_SOURCE_SHA")
    require(sha256_bytes(rows_raw) == OLD_601_ROWS_SHA, "OLD_601_ROWS_SHA")
    facts = fresh.build_source_facts(start=601, end=900)
    rows = fresh.materialize_facts(facts)
    require(base.jsonl_bytes(facts) == source_raw, "REGENERATED_601_SOURCE_DRIFT")
    require(base.jsonl_bytes(rows) == rows_raw, "REGENERATED_601_ROWS_DRIFT")
    identity.update({
        "fresh_601_900_source_byte_identity": True,
        "fresh_601_900_row_byte_identity": True,
        "fresh_601_900_source_sha256": OLD_601_SOURCE_SHA,
        "fresh_601_900_rows_sha256": OLD_601_ROWS_SHA,
    })
    return identity


def build_new_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    facts = fresh.build_source_facts(start=901, end=1200)
    rows = fresh.materialize_facts(facts)
    require(len(facts) == PAIR_COUNT, "NEW_FACT_COUNT")
    require(len(rows) == ROW_COUNT, "NEW_ROW_COUNT")
    require([str(x["pair_id"]) for x in facts] == [f"xg1_fact_{i:03d}" for i in range(901, 1201)], "NEW_PAIR_ORDER")

    old_rows: list[dict[str, Any]] = []
    for d in OLD_DIRS:
        old_rows.extend(read_jsonl(ROOT / d / base.ROW_FILE))

    old_pair_ids = {str(r["source_pair_id"]) for r in old_rows}
    new_pair_ids = {str(r["source_pair_id"]) for r in rows}
    old_claims = {str(r["claim"]) for r in old_rows}
    new_claims = {str(r["claim"]) for r in rows}
    old_evidence = {str(r["evidence"]) for r in old_rows}
    new_evidence = {str(r["evidence"]) for r in rows}
    old_pairs = {(str(r["claim"]), str(r["evidence"])) for r in old_rows}
    new_pairs = {(str(r["claim"]), str(r["evidence"])) for r in rows}

    require(not (old_pair_ids & new_pair_ids), "PAIR_ID_OVERLAP")
    require(not (old_claims & new_claims), "CLAIM_OVERLAP")
    require(not (old_evidence & new_evidence), "EVIDENCE_OVERLAP")
    require(not (old_pairs & new_pairs), "CLAIM_EVIDENCE_ROW_OVERLAP")

    return facts, rows, {
        "source_raw": base.jsonl_bytes(facts),
        "rows_raw": base.jsonl_bytes(rows),
        "pair_id_overlap_count": 0,
        "claim_overlap_count": 0,
        "evidence_overlap_count": 0,
        "claim_evidence_row_overlap_count": 0,
    }


def geometry_and_restoration_audit() -> dict[str, Any]:
    g0 = prior.geometry_audit()
    p3p = prior.load_vector(ROOT / prior.PP3_DIR / "pp3_plus.f64le", prior.PP3_PLUS_SHA)
    p3m = prior.load_vector(ROOT / prior.PP3_DIR / "pp3_minus.f64le", prior.PP3_MINUS_SHA)
    p5p = prior.load_vector(ROOT / prior.PP5_DIR / "pp5_plus.f64le", prior.PP5_PLUS_SHA)
    p5m = prior.load_vector(ROOT / prior.PP5_DIR / "pp5_minus.f64le", prior.PP5_MINUS_SHA)
    b3 = np.stack([p3p, p3m], axis=1)
    b5 = np.stack([p5p, p5m], axis=1)
    probes = np.asarray([[1.0,0.0],[0.0,1.0],[1.0,1.0],[3.0,-4.0],[math.pi,math.e],[-0.125,0.875]], dtype=np.float64)

    r3_res: list[float] = []
    r5_res: list[float] = []
    norm_res: list[float] = []
    coeff_res: list[float] = []
    for i, coeff in enumerate(probes):
        nuisance = np.asarray([0.37*(i+1), -0.19*(i+1)], dtype=np.float64)
        h = b3 @ coeff + b5 @ nuisance
        recovered = np.asarray([float(h @ p3p), float(h @ p3m)], dtype=np.float64)
        c3 = b3 @ recovered
        c5 = b5 @ recovered
        background = h - c3
        r3 = background + c3
        r5 = background + c5
        coeff_res.append(float(np.max(np.abs(recovered - coeff))))
        r3_res.append(float(np.max(np.abs(r3 - h))))
        r5_res.append(float(np.max(np.abs((r5 - background) - c5))))
        norm_res.append(abs(float(np.linalg.norm(r3-background)) - float(np.linalg.norm(r5-background))))

    coeff_max, r3_max, r5_max, norm_max = map(max, (coeff_res, r3_res, r5_res, norm_res))
    require(coeff_max <= TOL, f"PP3_COEFFICIENT_RECOVERY:{coeff_max}")
    require(r3_max <= TOL, f"R3_IDENTITY:{r3_max}")
    require(r5_max <= TOL, f"R5_INCREMENT:{r5_max}")
    require(norm_max <= TOL, f"RESTORATION_NORM_MATCH:{norm_max}")

    return {
        "schema_version": "GEN4_PP3_RESTORATION_SUFFICIENCY_GEOMETRY_STATIC_V1",
        "result": "PASS_PP3_RESTORATION_SUFFICIENCY_GEOMETRY_STATIC",
        "pp3_plus_sha256": prior.PP3_PLUS_SHA,
        "pp3_minus_sha256": prior.PP3_MINUS_SHA,
        "pp5_plus_sha256": prior.PP5_PLUS_SHA,
        "pp5_minus_sha256": prior.PP5_MINUS_SHA,
        "ambient_dim": int(g0["ambient_dim"]),
        "tolerance": TOL,
        "pp3_gram_max_abs_residual": float(g0["pp3_gram_max_abs_residual"]),
        "pp5_gram_max_abs_residual": float(g0["pp5_gram_max_abs_residual"]),
        "pp3_pp5_cross_plane_max_abs_dot": float(g0["pp3_pp5_cross_plane_max_abs_dot"]),
        "coefficient_transfer_max_abs_l2_mismatch": float(g0["coefficient_transfer_max_abs_l2_mismatch"]),
        "restoration_probe_count": int(probes.shape[0]),
        "pp3_coefficient_recovery_max_abs_residual": coeff_max,
        "r3_equals_h_max_abs_residual": r3_max,
        "r5_minus_background_equals_c5_max_abs_residual": r5_max,
        "restoration_addition_norm_max_abs_mismatch": norm_max,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_inference_executed": False,
    }


def tokenizer_eligibility(tokenizer_snapshot: Path, facts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_authority, old_first, old_last = prior.AUTHORITY_COMMIT, prior.PAIR_FIRST, prior.PAIR_LAST
    try:
        prior.AUTHORITY_COMMIT = DESIGN_COMMIT
        prior.PAIR_FIRST = PAIR_FIRST
        prior.PAIR_LAST = PAIR_LAST
        anchor_rows, summary = prior.tokenizer_eligibility(tokenizer_snapshot, facts, rows)
    finally:
        prior.AUTHORITY_COMMIT, prior.PAIR_FIRST, prior.PAIR_LAST = old_authority, old_first, old_last

    for row in anchor_rows:
        row["schema_version"] = "GEN4_PP3_RESTORATION_SUFFICIENCY_XG1_TOKENIZER_ANCHOR_V1"
    anchor_raw = b"".join(canonical_json_bytes(row) for row in anchor_rows)
    summary = dict(summary)
    summary.update({
        "schema_version": "GEN4_PP3_RESTORATION_SUFFICIENCY_XG1_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "authority_commit": DESIGN_COMMIT,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
    })
    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(int(summary["source_pair_count"]) == PAIR_COUNT, "TOKENIZER_PAIR_COUNT")
    require(int(summary["anchor_row_count"]) == ROW_COUNT, "TOKENIZER_ANCHOR_COUNT")
    return anchor_rows, summary


def write_outputs(provenance: Mapping[str, Any], identity: Mapping[str, Any], population: Mapping[str, Any], geometry: Mapping[str, Any], anchor_rows: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Any]) -> dict[str, Any]:
    data_dir = ROOT / OUTPUT_DATA_DIR
    report_dir = ROOT / OUTPUT_REPORT_DIR
    require(not data_dir.exists(), f"DATA_OUTPUT_COLLISION:{data_dir}")
    require(not report_dir.exists(), f"REPORT_OUTPUT_COLLISION:{report_dir}")
    data_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]
    structural = {
        "schema_version": "GEN4_PP3_RESTORATION_SUFFICIENCY_XG1_STRUCTURAL_V1",
        "result": "PASS_XG1_901_1200_RESTORATION_SUFFICIENCY_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": base.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        **dict(identity),
        "pair_id_overlap_with_001_900": population["pair_id_overlap_count"],
        "claim_overlap_with_001_900": population["claim_overlap_count"],
        "evidence_overlap_with_001_900": population["evidence_overlap_count"],
        "claim_evidence_row_overlap_with_001_900": population["claim_evidence_row_overlap_count"],
        "deterministic_generator_semantics": True,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    data_payloads = {base.SOURCE_FILE: source_raw, base.ROW_FILE: rows_raw, "structural_manifest.json": pretty_json_bytes(structural)}
    data_hashes: dict[str, str] = {}
    for name, raw in data_payloads.items():
        (data_dir / name).write_bytes(raw)
        data_hashes[name] = sha256_bytes(raw)
    (data_dir / "SHA256SUMS.txt").write_bytes("".join(f"{digest}  {name}\n" for name, digest in sorted(data_hashes.items())).encode("utf-8"))

    anchor_raw = b"".join(canonical_json_bytes(row) for row in anchor_rows)
    preparation = {
        "schema_version": "GEN4_PP3_RESTORATION_SUFFICIENCY_STATIC_PREPARATION_V1",
        "result": "PASS_PP3_RESTORATION_SUFFICIENCY_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": OUTPUT_DATA_DIR.as_posix(),
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_sha256": data_hashes[base.SOURCE_FILE],
            "rows_sha256": data_hashes[base.ROW_FILE],
            "structural_manifest_sha256": data_hashes["structural_manifest.json"],
        },
        "geometry": {
            "result": geometry["result"],
            "pp3_plus_sha256": prior.PP3_PLUS_SHA,
            "pp3_minus_sha256": prior.PP3_MINUS_SHA,
            "pp5_plus_sha256": prior.PP5_PLUS_SHA,
            "pp5_minus_sha256": prior.PP5_MINUS_SHA,
            "pp3_gram_max_abs_residual": geometry["pp3_gram_max_abs_residual"],
            "pp5_gram_max_abs_residual": geometry["pp5_gram_max_abs_residual"],
            "pp3_pp5_cross_plane_max_abs_dot": geometry["pp3_pp5_cross_plane_max_abs_dot"],
            "coefficient_transfer_max_abs_l2_mismatch": geometry["coefficient_transfer_max_abs_l2_mismatch"],
            "r3_equals_h_max_abs_residual": geometry["r3_equals_h_max_abs_residual"],
            "r5_minus_background_equals_c5_max_abs_residual": geometry["r5_minus_background_equals_c5_max_abs_residual"],
            "restoration_addition_norm_max_abs_mismatch": geometry["restoration_addition_norm_max_abs_mismatch"],
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "anchor_manifest_sha256": eligibility["anchor_manifest_sha256"],
            "source_pair_count": eligibility["source_pair_count"],
            "anchor_row_count": eligibility["anchor_row_count"],
            "tokenizer_revision": prior.TOKENIZER_REVISION,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }
    report_payloads = {
        "geometry_manifest.json": pretty_json_bytes(geometry),
        "tokenizer_anchor_manifest.jsonl": anchor_raw,
        "tokenizer_eligibility_summary.json": pretty_json_bytes(eligibility),
        "preparation_manifest.json": pretty_json_bytes(preparation),
    }
    report_hashes: dict[str, str] = {}
    for name, raw in report_payloads.items():
        (report_dir / name).write_bytes(raw)
        report_hashes[name] = sha256_bytes(raw)
    (report_dir / "SHA256SUMS.txt").write_bytes("".join(f"{digest}  {name}\n" for name, digest in sorted(report_hashes.items())).encode("utf-8"))
    return {"data_hashes": data_hashes, "report_hashes": report_hashes}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only static preparation for PP3 restoration sufficiency.")
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.tokenizer_snapshot.resolve().is_relative_to(ROOT.resolve()), "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO")

    provenance = authenticate_design()
    identity = prove_prior_population_identity()
    facts, rows, population = build_new_population()
    geometry = geometry_and_restoration_audit()
    anchor_rows, eligibility = tokenizer_eligibility(args.tokenizer_snapshot, facts, rows)
    outputs = write_outputs(provenance, identity, population, geometry, anchor_rows, eligibility)

    print("RESULT=PASS_PP3_RESTORATION_SUFFICIENCY_STATIC_PREPARATION")
    print(f"PAIR_ID_FIRST={PAIR_FIRST}")
    print(f"PAIR_ID_LAST={PAIR_LAST}")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PAIR_ID_OVERLAP_WITH_001_900=0")
    print("CLAIM_OVERLAP_WITH_001_900=0")
    print("EVIDENCE_OVERLAP_WITH_001_900=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_900=0")
    print(f"PP3_GRAM_MAX_ABS_RESIDUAL={geometry['pp3_gram_max_abs_residual']:.17g}")
    print(f"PP5_GRAM_MAX_ABS_RESIDUAL={geometry['pp5_gram_max_abs_residual']:.17g}")
    print(f"PP3_PP5_CROSS_PLANE_MAX_ABS_DOT={geometry['pp3_pp5_cross_plane_max_abs_dot']:.17g}")
    print(f"COEFFICIENT_TRANSFER_MAX_ABS_L2_MISMATCH={geometry['coefficient_transfer_max_abs_l2_mismatch']:.17g}")
    print(f"R3_EQUALS_H_MAX_ABS_RESIDUAL={geometry['r3_equals_h_max_abs_residual']:.17g}")
    print(f"R5_MINUS_BACKGROUND_EQUALS_C5_MAX_ABS_RESIDUAL={geometry['r5_minus_background_equals_c5_max_abs_residual']:.17g}")
    print(f"RESTORATION_ADDITION_NORM_MAX_ABS_MISMATCH={geometry['restoration_addition_norm_max_abs_mismatch']:.17g}")
    print(f"TOKENIZER_ELIGIBILITY={eligibility['result']}")
    print(f"ANCHOR_ROW_COUNT={eligibility['anchor_row_count']}")
    print(f"ANCHOR_MANIFEST_SHA256={eligibility['anchor_manifest_sha256']}")
    print(f"SOURCE_SHA256={outputs['data_hashes'][base.SOURCE_FILE]}")
    print(f"ROWS_SHA256={outputs['data_hashes'][base.ROW_FILE]}")
    print(f"STRUCTURAL_MANIFEST_SHA256={outputs['data_hashes']['structural_manifest.json']}")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
