#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import build_reason_router_gen4_factor2_small_alpha_holdout as previous

ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"
REQUIRED_BASE = "684d69eb777e0ff868bf7b41d3e55ca23bd805ca"

PREVIOUS_BUILDER_PATH = "scripts/build_reason_router_gen4_factor2_small_alpha_holdout.py"
PREVIOUS_BUILDER_BLOB = "db3ff0952a9ddec2241d0693c266a0ea396c5778"
PREVIOUS_DIR = previous.OUTPUT_DIR
PREVIOUS_SOURCE_SHA256 = "05026973b2ec61847c85d6aab800eada130aad9e9c7edb14a4d8d88f41544c4e"
PREVIOUS_ROWS_SHA256 = "08da7b3b1d9d92f189b6481abd0b889aeac908519eeae804b8574328ce497f51"
PREVIOUS_MANIFEST_SHA256 = "ffa42c683f76d8efdb77cb30404f63a6d0dc3915f5b27953ae215ef1f45aeaa9"

PAIR_COUNT = 300
ROWS_PER_PAIR = 6
ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

SOURCE_FILE = previous.SOURCE_FILE
ROW_FILE = previous.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
MANIFEST_SCHEMA = "GEN4_MAMBA1_FIVE_SCALE_LADDER_XG1_PROSPECTIVE_HOLDOUT_V1"
RESULT = "PASS_MAMBA1_FIVE_SCALE_LADDER_XG1_PROSPECTIVE_HOLDOUTS"

COHORTS = (
    ("mamba790m", "discovery", 6901, 7200),
    ("mamba790m", "confirmation", 7201, 7500),
    ("mamba790m", "readout", 7501, 7800),
    ("mamba28b", "discovery", 6001, 6300),
    ("mamba28b", "confirmation", 6301, 6600),
    ("mamba28b", "readout", 6601, 6900),
)

OUTPUT_DIRS = {
    ("mamba790m", "discovery"): Path("data/reason_router_gen4_mamba790m_xg1_discovery_v1"),
    ("mamba790m", "confirmation"): Path("data/reason_router_gen4_mamba790m_xg1_confirmation_v1"),
    ("mamba790m", "readout"): Path("data/reason_router_gen4_mamba790m_xg1_readout_v1"),
    ("mamba28b", "discovery"): Path("data/reason_router_gen4_mamba28b_xg1_discovery_v1"),
    ("mamba28b", "confirmation"): Path("data/reason_router_gen4_mamba28b_xg1_confirmation_v1"),
    ("mamba28b", "readout"): Path("data/reason_router_gen4_mamba28b_xg1_readout_v1"),
}

FORBIDDEN_OUTCOME_FIELDS = previous.FORBIDDEN_OUTCOME_FIELDS
PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")


class FiveScaleHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FiveScaleHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FiveScaleHoldoutError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FiveScaleHoldoutError(f"GIT_BLOB_FAILURE:{path.as_posix()}") from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return previous.jsonl_bytes(rows)


def canonical_manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def expected_pair_ids(first: int, last: int) -> list[str]:
    return previous.expected_pair_ids(first, last)


def build_population(first: int, last: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = previous.build_population(first, last)
    require([str(row["pair_id"]) for row in facts] == expected_pair_ids(first, last), "PAIR_ID_SEQUENCE")
    require(len(facts) == last - first + 1, "SOURCE_COUNT")
    require(len(rows) == len(facts) * ROWS_PER_PAIR, "ROW_COUNT")
    for row in [*facts, *rows]:
        require(not (FORBIDDEN_OUTCOME_FIELDS & set(row)), "OUTCOME_FIELD_PRESENT")
    return facts, rows


def _parse_jsonl_bytes(raw: bytes, label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{label}:OBJECT:{line_no}")
        out.append(value)
    return out


def authenticate_repo() -> None:
    require(git("branch", "--show-current") == EXPECTED_BRANCH, "BRANCH_MISMATCH")
    require(
        git_rc("merge-base", "--is-ancestor", REQUIRED_BASE, "HEAD") == 0,
        "REQUIRED_BASE_NOT_ANCESTOR",
    )
    require(
        git("rev-parse", f"HEAD:{PREVIOUS_BUILDER_PATH}") == PREVIOUS_BUILDER_BLOB,
        "PREVIOUS_BUILDER_BLOB_DRIFT",
    )
    for path in (PREVIOUS_BUILDER_PATH, PREVIOUS_DIR.as_posix()):
        require(git_rc("diff", "--quiet", REQUIRED_BASE, "HEAD", "--", path) == 0,
                f"FROZEN_DEPENDENCY_COMMIT_DRIFT:{path}")
        require(git_rc("diff", "--quiet", "--", path) == 0,
                f"FROZEN_DEPENDENCY_WORKTREE_DRIFT:{path}")
        require(git_rc("diff", "--cached", "--quiet", "--", path) == 0,
                f"FROZEN_DEPENDENCY_INDEX_DRIFT:{path}")


def prior_inventory_through_6000() -> tuple[set[str], set[str], set[str]]:
    pair_ids, claims, evidence = previous.prior_inventory_through_5700()
    source_raw = git_blob_bytes(PREVIOUS_DIR / SOURCE_FILE)
    rows_raw = git_blob_bytes(PREVIOUS_DIR / ROW_FILE)
    manifest_raw = git_blob_bytes(PREVIOUS_DIR / MANIFEST_FILE)
    require(sha256_bytes(source_raw) == PREVIOUS_SOURCE_SHA256, "PREVIOUS_SOURCE_SHA")
    require(sha256_bytes(rows_raw) == PREVIOUS_ROWS_SHA256, "PREVIOUS_ROWS_SHA")
    require(sha256_bytes(manifest_raw) == PREVIOUS_MANIFEST_SHA256, "PREVIOUS_MANIFEST_SHA")
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    require(manifest["result"] == previous.RESULT, "PREVIOUS_RESULT")
    require(
        manifest["pair_id_first"] == "xg1_fact_5701"
        and manifest["pair_id_last"] == "xg1_fact_6000",
        "PREVIOUS_RANGE",
    )
    require(manifest["source_file_sha256"] == PREVIOUS_SOURCE_SHA256, "PREVIOUS_MANIFEST_SOURCE_SHA")
    require(manifest["row_file_sha256"] == PREVIOUS_ROWS_SHA256, "PREVIOUS_MANIFEST_ROWS_SHA")

    regen_facts, regen_rows = previous.build_population(previous.FIRST_PAIR, previous.LAST_PAIR)
    require(previous.jsonl_bytes(regen_facts) == source_raw, "PREVIOUS_SOURCE_REGENERATION")
    require(previous.jsonl_bytes(regen_rows) == rows_raw, "PREVIOUS_ROWS_REGENERATION")

    frozen_facts = _parse_jsonl_bytes(source_raw, "PREVIOUS_SOURCE")
    frozen_rows = _parse_jsonl_bytes(rows_raw, "PREVIOUS_ROWS")
    new_pairs = {str(row["pair_id"]) for row in frozen_facts}
    new_claims = {str(row["claim"]) for row in frozen_rows}
    new_evidence = {str(row["evidence"]) for row in frozen_rows}
    require(not (pair_ids & new_pairs), "PREVIOUS_PAIR_OVERLAP")
    require(not (claims & new_claims), "PREVIOUS_CLAIM_OVERLAP")
    require(not (evidence & new_evidence), "PREVIOUS_EVIDENCE_OVERLAP")
    pair_ids |= new_pairs
    claims |= new_claims
    evidence |= new_evidence
    require(pair_ids == set(expected_pair_ids(1, 6000)), "PRIOR_PAIR_COVERAGE_001_6000")
    return pair_ids, claims, evidence


def _sets_for_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(row["pair_id"]) for row in facts},
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _role_fields(role: str) -> dict[str, Any]:
    if role == "discovery":
        return {
            "prospective_use": "dominant_component_discovery_only",
            "all_five_planes_required": list(PLANE_ORDER),
            "unique_argmax_required": True,
            "positivity_gate": False,
            "response_blind_control_from_geometry_only": True,
            "selection_allowed": True,
            "formal_inference_allowed": False,
            "p_value_count": 0,
            "confirmation_response_access_allowed": False,
            "readout_response_access_allowed": False,
            "rescue_policy": "none",
        }
    if role == "confirmation":
        return {
            "prospective_use": "core_confirmation_only_after_discovery_selection_freeze",
            "planned_endpoint": "D_CORE=Q_restored(k*)-Q_control(c*)",
            "planned_primary_test": {
                "test": "one_sample_student_t",
                "alternative": "greater",
                "alpha": 0.05,
                "n": 300,
                "primary_p_value_count": 1,
            },
            "success_rule": "mean(D_CORE)>0 and one-sided primary p<0.05",
            "selection_allowed": False,
            "discovery_raw_response_access_allowed": False,
            "readout_response_access_allowed": False,
            "rescue_policy": "none",
        }
    if role == "readout":
        return {
            "prospective_use": "fresh_readout_only_after_core_confirmation",
            "planned_quantity": "Delta_L_owned_with_forward_equivalent_recorded_if_applicable",
            "selected_component_fixed_before_readout": True,
            "response_blind_control_fixed_before_readout": True,
            "selection_allowed": False,
            "discovery_raw_response_access_allowed": False,
            "confirmation_raw_response_access_allowed": False,
            "response_guided_selection_allowed": False,
            "rescue_policy": "none",
        }
    raise FiveScaleHoldoutError(f"ROLE:{role}")


def build_holdout_payloads() -> dict[tuple[str, str], dict[str, Any]]:
    inventory = prior_inventory_through_6000()
    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    # Generate in numerical pair order, independent of execution order.
    specs = sorted(COHORTS, key=lambda x: x[2])
    prior_last = 6000
    for scale, role, first, last in specs:
        facts_a, rows_a = build_population(first, last)
        facts_b, rows_b = build_population(first, last)
        source_raw = jsonl_bytes(facts_a)
        row_raw = jsonl_bytes(rows_a)
        require(source_raw == jsonl_bytes(facts_b), f"{scale}:{role}:SOURCE_REGENERATION")
        require(row_raw == jsonl_bytes(rows_b), f"{scale}:{role}:ROW_REGENERATION")

        current = _sets_for_rows(facts_a, rows_a)
        require(len(current[0]) == PAIR_COUNT, f"{scale}:{role}:PAIR_COUNT")
        require(len(current[1]) == PAIR_COUNT, f"{scale}:{role}:CLAIM_COUNT")
        require(len(current[2]) == ROW_COUNT, f"{scale}:{role}:EVIDENCE_COUNT")
        require(not (inventory[0] & current[0]), f"{scale}:{role}:PAIR_OVERLAP")
        require(not (inventory[1] & current[1]), f"{scale}:{role}:CLAIM_OVERLAP")
        require(not (inventory[2] & current[2]), f"{scale}:{role}:EVIDENCE_OVERLAP")

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "result": "PASS_MAMBA1_FIVE_SCALE_LADDER_XG1_STRUCTURAL",
            "scale": scale,
            "role": role,
            "generator_family": facts_a[0]["generator_family"],
            "deterministic_generator_semantics": True,
            "pair_id_first": f"xg1_fact_{first}",
            "pair_id_last": f"xg1_fact_{last}",
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "rows_per_pair": ROWS_PER_PAIR,
            "source_file_sha256": sha256_bytes(source_raw),
            "row_file_sha256": sha256_bytes(row_raw),
            "prior_pair_range": f"xg1_fact_001..xg1_fact_{prior_last}",
            "pair_id_overlap_with_prior": 0,
            "claim_overlap_with_prior": 0,
            "evidence_overlap_with_prior": 0,
            "labels_present": False,
            "response_fields_present": False,
            "endpoint_values_present": False,
            "tokenizer_executed": False,
            "checkpoint_loaded": False,
            "model_executed": False,
            "cuda_executed": False,
            "cohort_replacement_allowed": False,
            "row_filtering_allowed": False,
            **_role_fields(role),
        }
        payloads[(scale, role)] = {
            "source": source_raw,
            "rows": row_raw,
            "manifest": manifest,
        }
        inventory = (
            inventory[0] | current[0],
            inventory[1] | current[1],
            inventory[2] | current[2],
        )
        prior_last = last

    require(inventory[0] == set(expected_pair_ids(1, 7800)), "FINAL_PAIR_COVERAGE_001_7800")
    return payloads


def _write_one(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    source_raw = bytes(payload["source"])
    row_raw = bytes(payload["rows"])
    manifest_raw = canonical_manifest_bytes(payload["manifest"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / SOURCE_FILE).write_bytes(source_raw)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )
    return hashes


def validate_written(output_dir: Path, scale: str, role: str, first: int, last: int) -> dict[str, Any]:
    required = {SOURCE_FILE, ROW_FILE, MANIFEST_FILE, CHECKSUM_FILE}
    require(output_dir.is_dir(), f"OUTPUT_DIR_MISSING:{output_dir}")
    require({p.name for p in output_dir.iterdir() if p.is_file()} == required,
            f"OUTPUT_FILE_SET:{output_dir}")
    source_raw = (output_dir / SOURCE_FILE).read_bytes()
    row_raw = (output_dir / ROW_FILE).read_bytes()
    manifest_raw = (output_dir / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    facts, rows = build_population(first, last)
    require(source_raw == jsonl_bytes(facts), f"WRITTEN_SOURCE_REGENERATION:{output_dir}")
    require(row_raw == jsonl_bytes(rows), f"WRITTEN_ROWS_REGENERATION:{output_dir}")
    require(manifest["scale"] == scale and manifest["role"] == role, f"WRITTEN_ROLE:{output_dir}")
    require(manifest["pair_id_first"] == f"xg1_fact_{first}", f"WRITTEN_FIRST:{output_dir}")
    require(manifest["pair_id_last"] == f"xg1_fact_{last}", f"WRITTEN_LAST:{output_dir}")
    require(manifest["source_file_sha256"] == sha256_bytes(source_raw), f"WRITTEN_SOURCE_SHA:{output_dir}")
    require(manifest["row_file_sha256"] == sha256_bytes(row_raw), f"WRITTEN_ROW_SHA:{output_dir}")
    for key in (
        "labels_present", "response_fields_present", "endpoint_values_present",
        "tokenizer_executed", "checkpoint_loaded", "model_executed", "cuda_executed",
        "cohort_replacement_allowed", "row_filtering_allowed",
    ):
        require(manifest[key] is False, f"BOUNDARY:{key}:{output_dir}")
    expected_hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    expected_sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(expected_hashes.items()))
    require((output_dir / CHECKSUM_FILE).read_text(encoding="utf-8") == expected_sums,
            f"CHECKSUMS:{output_dir}")
    return manifest


def write_holdouts() -> dict[str, Any]:
    authenticate_repo()
    for key, rel in OUTPUT_DIRS.items():
        require(not (ROOT / rel).exists(), f"OUTPUT_COLLISION:{key}:{rel}")
    payloads = build_holdout_payloads()
    out: dict[str, Any] = {"result": RESULT, "cohorts": {}}
    for scale, role, first, last in COHORTS:
        rel = OUTPUT_DIRS[(scale, role)]
        hashes = _write_one(ROOT / rel, payloads[(scale, role)])
        manifest = validate_written(ROOT / rel, scale, role, first, last)
        out["cohorts"][f"{scale}:{role}"] = {
            "manifest": manifest,
            "artifact_sha256": hashes,
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Materialize all six outcome-blind XG1 cohorts for the Mamba-1 five-scale ladder extension."
    )
    parser.parse_args(argv)
    result = write_holdouts()
    print("RESULT=" + result["result"])
    for scale, role, first, last in COHORTS:
        print(f"{scale.upper()}_{role.upper()}=xg1_fact_{first}..xg1_fact_{last}")
    print("PRIOR_PAIR_COVERAGE=xg1_fact_001..xg1_fact_6000")
    print("FINAL_PAIR_COVERAGE=xg1_fact_001..xg1_fact_7800")
    print("ALL_COHORT_PAIR_CLAIM_EVIDENCE_OVERLAP=0")
    print("LABELS_PRESENT=False")
    print("RESPONSE_FIELDS_PRESENT=False")
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
