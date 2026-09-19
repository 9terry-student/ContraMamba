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

from scripts import build_reason_router_gen4_mamba370m_xg1_holdouts as m370
from scripts import (
    build_reason_router_gen4_mamba370m_xg1_residual_decomposition_holdout
    as m370_residual,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "f97b597fb4da08a8d360ed07e48c6727e15ae0be"

DISCOVERY_FIRST = 3901
DISCOVERY_LAST = 4200
CONFIRMATION_FIRST = 4201
CONFIRMATION_LAST = 4500
RESIDUAL_FIRST = 4501
RESIDUAL_LAST = 4800

PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

DISCOVERY_DIR = Path("data/reason_router_gen4_mamba14b_xg1_discovery_v1")
CONFIRMATION_DIR = Path("data/reason_router_gen4_mamba14b_xg1_confirmation_v1")
RESIDUAL_DIR = Path("data/reason_router_gen4_mamba14b_xg1_residual_characterization_v1")

SOURCE_FILE = m370.SOURCE_FILE
ROW_FILE = m370.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

MANIFEST_SCHEMA = "GEN4_MAMBA14B_XG1_PROSPECTIVE_HOLDOUT_V1"
RESULT = "PASS_MAMBA14B_XG1_PROSPECTIVE_HOLDOUTS"

RESIDUAL_3601_3900_DIR = m370_residual.OUTPUT_DIR
RESIDUAL_3601_3900_SOURCE_SHA256 = (
    "2eb3c50a95b325f2bb0a3f478443bb2c2be3f2dab0c134a0b760649db65bc9e6"
)
RESIDUAL_3601_3900_ROWS_SHA256 = (
    "9caaff06c9f4ae02015542ac5f7f193a6d97666f1dc3ae96e126b7eb4df9590b"
)
RESIDUAL_3601_3900_MANIFEST_SHA256 = (
    "0ede4e6095dbe1194164d4249e4a47982688a06f3406c94ec6c14b01f2d655d3"
)

FORBIDDEN_OUTCOME_FIELDS = m370.FORBIDDEN_OUTCOME_FIELDS
PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")


class Mamba14BXG1HoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Mamba14BXG1HoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Mamba14BXG1HoldoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authenticate_repo() -> None:
    require(
        git("branch", "--show-current") == EXPECTED_BRANCH,
        "BRANCH_MISMATCH",
    )
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            "HEAD",
        )
        == 0,
        "REQUIRED_ANCESTOR_MISSING",
    )

    for path in m370.FROZEN_DEPENDENCIES:
        require(
            git_rc(
                "diff",
                "--quiet",
                m370.FROZEN_PARENT_COMMIT,
                "HEAD",
                "--",
                path,
            )
            == 0,
            f"FROZEN_DEPENDENCY_COMMIT_DRIFT:{path}",
        )
        require(
            git_rc("diff", "--quiet", "--", path) == 0,
            f"FROZEN_DEPENDENCY_WORKTREE_DRIFT:{path}",
        )
        require(
            git_rc("diff", "--cached", "--quiet", "--", path) == 0,
            f"FROZEN_DEPENDENCY_INDEX_DRIFT:{path}",
        )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_manifest_bytes(
    manifest: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            dict(manifest),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def expected_pair_ids(first: int, last: int) -> list[str]:
    return m370.expected_pair_ids(first, last)


def build_population(
    first: int,
    last: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts, rows = m370.build_population(first, last)
    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(first, last),
        "PAIR_ID_SEQUENCE",
    )
    require(len(facts) == PAIR_COUNT, "SOURCE_COUNT")
    require(len(rows) == EXPECTED_ROW_COUNT, "ROW_COUNT")

    for fact in facts:
        require(
            not (FORBIDDEN_OUTCOME_FIELDS & set(fact)),
            f"SOURCE_OUTCOME_FIELD:{fact['pair_id']}",
        )
    for row in rows:
        require(
            not (FORBIDDEN_OUTCOME_FIELDS & set(row)),
            f"ROW_OUTCOME_FIELD:{row['row_id']}",
        )
    return facts, rows


def _git_population_inventory(
    directory: Path,
    *,
    first: int,
    last: int,
    source_sha: str,
    rows_sha: str,
    manifest_sha: str,
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    source_raw = m370.prior.git_blob_bytes(directory / SOURCE_FILE)
    rows_raw = m370.prior.git_blob_bytes(directory / ROW_FILE)
    manifest_raw = m370.prior.git_blob_bytes(directory / MANIFEST_FILE)

    require(
        sha256_bytes(source_raw) == source_sha,
        f"{label}:SOURCE_SHA",
    )
    require(
        sha256_bytes(rows_raw) == rows_sha,
        f"{label}:ROWS_SHA",
    )
    require(
        sha256_bytes(manifest_raw) == manifest_sha,
        f"{label}:MANIFEST_SHA",
    )

    manifest = json.loads(manifest_raw.decode("utf-8-sig"))
    require(
        manifest["source_file_sha256"] == source_sha,
        f"{label}:MANIFEST_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"] == rows_sha,
        f"{label}:MANIFEST_ROWS_SHA",
    )
    require(
        manifest["pair_id_first"] == f"xg1_fact_{first:03d}"
        and manifest["pair_id_last"] == f"xg1_fact_{last:03d}",
        f"{label}:MANIFEST_RANGE",
    )
    require(
        manifest["labels_present"] is False
        and manifest["response_fields_present"] is False
        and manifest["endpoint_values_present"] is False
        and manifest["model_executed"] is False
        and manifest["cuda_executed"] is False,
        f"{label}:MANIFEST_OUTCOME_BLINDNESS",
    )

    return m370._inventory_from_serialized(
        source_raw,
        rows_raw,
        first=first,
        last=last,
        label=label,
    )


def _merge_disjoint(
    current: tuple[set[str], set[str], set[str]],
    addition: tuple[set[str], set[str], set[str]],
    *,
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    pairs, claims, evidence = current
    p2, c2, e2 = addition

    require(not (pairs & p2), f"{label}:PAIR_OVERLAP")
    require(not (claims & c2), f"{label}:CLAIM_OVERLAP")
    require(not (evidence & e2), f"{label}:EVIDENCE_OVERLAP")

    return pairs | p2, claims | c2, evidence | e2


def prior_inventory_through_3900(
) -> tuple[set[str], set[str], set[str]]:
    inventory = m370.prior_inventory_through_3000()

    discovery = _git_population_inventory(
        m370.DISCOVERY_DIR,
        first=3001,
        last=3300,
        source_sha=m370_residual.DISCOVERY_SOURCE_SHA256,
        rows_sha=m370_residual.DISCOVERY_ROWS_SHA256,
        manifest_sha=m370_residual.DISCOVERY_MANIFEST_SHA256,
        label="MAMBA370_DISCOVERY_3001_3300",
    )
    inventory = _merge_disjoint(
        inventory,
        discovery,
        label="MAMBA370_DISCOVERY_3001_3300",
    )

    confirmation = _git_population_inventory(
        m370.CONFIRMATION_DIR,
        first=3301,
        last=3600,
        source_sha=m370_residual.CONFIRMATION_SOURCE_SHA256,
        rows_sha=m370_residual.CONFIRMATION_ROWS_SHA256,
        manifest_sha=m370_residual.CONFIRMATION_MANIFEST_SHA256,
        label="MAMBA370_CONFIRMATION_3301_3600",
    )
    inventory = _merge_disjoint(
        inventory,
        confirmation,
        label="MAMBA370_CONFIRMATION_3301_3600",
    )

    residual = _git_population_inventory(
        RESIDUAL_3601_3900_DIR,
        first=3601,
        last=3900,
        source_sha=RESIDUAL_3601_3900_SOURCE_SHA256,
        rows_sha=RESIDUAL_3601_3900_ROWS_SHA256,
        manifest_sha=RESIDUAL_3601_3900_MANIFEST_SHA256,
        label="MAMBA370_RESIDUAL_3601_3900",
    )
    inventory = _merge_disjoint(
        inventory,
        residual,
        label="MAMBA370_RESIDUAL_3601_3900",
    )

    pairs, claims, evidence = inventory
    require(
        pairs == set(expected_pair_ids(1, 3900)),
        "PRIOR_PAIR_COVERAGE_001_3900",
    )
    return pairs, claims, evidence


def _sets_for_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    return m370._sets_for_rows(facts, rows)


def _build_deterministic_population(
    first: int,
    last: int,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts_a, rows_a = build_population(first, last)
    facts_b, rows_b = build_population(first, last)

    require(
        m370.prior.jsonl_bytes(facts_a)
        == m370.prior.jsonl_bytes(facts_b),
        f"{label}:SOURCE_REGENERATION",
    )
    require(
        m370.prior.jsonl_bytes(rows_a)
        == m370.prior.jsonl_bytes(rows_b),
        f"{label}:ROW_REGENERATION",
    )
    return facts_a, rows_a


def _validate_fresh(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_inventory: tuple[set[str], set[str], set[str]],
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    prior_pairs, prior_claims, prior_evidence = prior_inventory
    pairs, claims, evidence = _sets_for_rows(facts, rows)

    require(not (pairs & prior_pairs), f"{label}:PAIR_OVERLAP")
    require(not (claims & prior_claims), f"{label}:CLAIM_OVERLAP")
    require(not (evidence & prior_evidence), f"{label}:EVIDENCE_OVERLAP")
    require(len(pairs) == PAIR_COUNT, f"{label}:PAIR_COUNT")
    require(bool(claims), f"{label}:CLAIMS_EMPTY")
    require(bool(evidence), f"{label}:EVIDENCE_EMPTY")
    return pairs, claims, evidence


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
            "residual_response_access_allowed": False,
            "rescue_policy": "none",
        }

    if role == "confirmation":
        return {
            "prospective_use":
                "core_confirmation_only_after_discovery_selection_freeze",
            "planned_endpoint":
                "D_CORE=Q_restored(k*)-Q_control(c*)",
            "planned_primary_test": {
                "test": "one_sample_student_t",
                "alternative": "greater",
                "alpha": 0.05,
                "n": 300,
                "primary_p_value_count": 1,
            },
            "success_rule":
                "mean(D_CORE)>0 and one-sided primary p<0.05",
            "selection_allowed": False,
            "discovery_raw_response_access_allowed": False,
            "residual_response_access_allowed": False,
            "rescue_policy": "none",
        }

    if role == "residual_characterization":
        return {
            "prospective_use": "descriptive_residual_characterization_only",
            "dominant_plane_preselected": False,
            "residual_plane_set_preselected": False,
            "p3_preselected": False,
            "p5_predesignated": False,
            "planned_symbolic_endpoints": {
                "S_k": "Q_native-Q_k-neutralized for k != k*",
                "S_RES":
                    "Q_native-Q_all-residual-neutralized",
                "S_SUM": "sum_{k != k*} S_k",
                "I_RES": "S_RES-S_SUM",
                "C_k":
                    "mean_branch[a_k^2+b_k^2]",
            },
            "formal_inference_allowed": False,
            "p_value_count": 0,
            "selection_allowed": False,
            "core_confirmation_inference_access_allowed": False,
            "rescue_of_failed_core_test": False,
            "layer_sweep_allowed": False,
            "token_sweep_allowed": False,
            "epsilon_sweep_allowed": False,
        }

    raise Mamba14BXG1HoldoutError(f"ROLE:{role}")


def build_manifest(
    *,
    role: str,
    first: int,
    last: int,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    prior_last: int,
) -> dict[str, Any]:
    require(
        role in {
            "discovery",
            "confirmation",
            "residual_characterization",
        },
        "ROLE",
    )
    require(len(facts) == PAIR_COUNT, "MANIFEST_PAIR_COUNT")
    require(len(rows) == EXPECTED_ROW_COUNT, "MANIFEST_ROW_COUNT")

    source_raw = m370.prior.jsonl_bytes(facts)
    row_raw = m370.prior.jsonl_bytes(rows)

    result_by_role = {
        "discovery":
            "PASS_MAMBA14B_XG1_DISCOVERY_3901_4200_STRUCTURAL",
        "confirmation":
            "PASS_MAMBA14B_XG1_CONFIRMATION_4201_4500_STRUCTURAL",
        "residual_characterization":
            "PASS_MAMBA14B_XG1_RESIDUAL_4501_4800_STRUCTURAL",
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": result_by_role[role],
        "role": role,
        "generator_family": m370.prior.base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{first:03d}",
        "pair_id_last": f"xg1_fact_{last:03d}",
        "source_pair_count": PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range":
            f"xg1_fact_001..xg1_fact_{prior_last:03d}",
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
        **_role_fields(role),
    }
    return manifest


def build_holdout_payloads() -> dict[str, dict[str, Any]]:
    inventory = prior_inventory_through_3900()
    payloads: dict[str, dict[str, Any]] = {}

    specs = (
        ("discovery", DISCOVERY_FIRST, DISCOVERY_LAST, 3900),
        ("confirmation", CONFIRMATION_FIRST, CONFIRMATION_LAST, 4200),
        (
            "residual_characterization",
            RESIDUAL_FIRST,
            RESIDUAL_LAST,
            4500,
        ),
    )

    for role, first, last, prior_last in specs:
        facts, rows = _build_deterministic_population(
            first,
            last,
            label=role.upper(),
        )
        fresh = _validate_fresh(
            facts,
            rows,
            prior_inventory=inventory,
            label=role.upper(),
        )
        inventory = _merge_disjoint(
            inventory,
            fresh,
            label=role.upper(),
        )

        manifest = build_manifest(
            role=role,
            first=first,
            last=last,
            facts=facts,
            rows=rows,
            prior_last=prior_last,
        )
        payloads[role] = {
            "source": m370.prior.jsonl_bytes(facts),
            "rows": m370.prior.jsonl_bytes(rows),
            "manifest": manifest,
        }

    pairs, _, _ = inventory
    require(
        pairs == set(expected_pair_ids(1, 4800)),
        "FINAL_PAIR_COVERAGE_001_4800",
    )
    return payloads


def _write_one(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )

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
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return hashes


def validate_written(
    output_dir: Path,
    *,
    expected_first: int,
    expected_last: int,
    expected_role: str,
) -> dict[str, Any]:
    required = {
        SOURCE_FILE,
        ROW_FILE,
        MANIFEST_FILE,
        CHECKSUM_FILE,
    }
    require(output_dir.is_dir(), f"OUTPUT_DIR_MISSING:{output_dir}")
    require(
        {
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        }
        == required,
        f"OUTPUT_FILE_SET:{output_dir}",
    )

    source_raw = (output_dir / SOURCE_FILE).read_bytes()
    row_raw = (output_dir / ROW_FILE).read_bytes()
    manifest_raw = (output_dir / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8-sig"))

    facts = m370.prior.read_jsonl_bytes(
        source_raw,
        f"{output_dir}:{SOURCE_FILE}",
    )
    rows = m370.prior.read_jsonl_bytes(
        row_raw,
        f"{output_dir}:{ROW_FILE}",
    )

    require(
        [str(row["pair_id"]) for row in facts]
        == expected_pair_ids(expected_first, expected_last),
        f"WRITTEN_PAIR_IDS:{output_dir}",
    )
    m370.prior.base.validate_materialized_rows(
        rows,
        expected_pairs=PAIR_COUNT,
    )
    require(
        manifest["role"] == expected_role,
        f"WRITTEN_ROLE:{output_dir}",
    )
    require(
        manifest["source_file_sha256"] == sha256_bytes(source_raw),
        f"WRITTEN_SOURCE_SHA:{output_dir}",
    )
    require(
        manifest["row_file_sha256"] == sha256_bytes(row_raw),
        f"WRITTEN_ROW_SHA:{output_dir}",
    )
    require(
        manifest["labels_present"] is False
        and manifest["response_fields_present"] is False
        and manifest["endpoint_values_present"] is False
        and manifest["tokenizer_executed"] is False
        and manifest["checkpoint_loaded"] is False
        and manifest["model_executed"] is False
        and manifest["cuda_executed"] is False,
        f"WRITTEN_OUTCOME_BLINDNESS:{output_dir}",
    )

    expected_hashes = {
        SOURCE_FILE: sha256_bytes(source_raw),
        ROW_FILE: sha256_bytes(row_raw),
        MANIFEST_FILE: sha256_bytes(manifest_raw),
    }
    expected_sums = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(expected_hashes.items())
    )
    require(
        (output_dir / CHECKSUM_FILE).read_text(encoding="utf-8")
        == expected_sums,
        f"WRITTEN_CHECKSUMS:{output_dir}",
    )
    return manifest


def write_holdouts(
    discovery_dir: Path = ROOT / DISCOVERY_DIR,
    confirmation_dir: Path = ROOT / CONFIRMATION_DIR,
    residual_dir: Path = ROOT / RESIDUAL_DIR,
) -> dict[str, Any]:
    authenticate_repo()

    outputs = {
        "discovery": discovery_dir,
        "confirmation": confirmation_dir,
        "residual_characterization": residual_dir,
    }
    for output_dir in outputs.values():
        require(
            not output_dir.exists(),
            f"OUTPUT_COLLISION:{output_dir}",
        )

    payloads = build_holdout_payloads()
    specs = {
        "discovery": (DISCOVERY_FIRST, DISCOVERY_LAST),
        "confirmation": (CONFIRMATION_FIRST, CONFIRMATION_LAST),
        "residual_characterization": (RESIDUAL_FIRST, RESIDUAL_LAST),
    }

    result: dict[str, Any] = {"result": RESULT}
    for role, output_dir in outputs.items():
        hashes = _write_one(output_dir, payloads[role])
        first, last = specs[role]
        manifest = validate_written(
            output_dir,
            expected_first=first,
            expected_last=last,
            expected_role=role,
        )
        result[role] = {
            "manifest": manifest,
            "artifact_sha256": hashes,
        }
    return result


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize outcome-blind Mamba-1.4B XG1 discovery "
            "3901..4200, core-confirmation 4201..4500, and residual "
            "characterization 4501..4800 structural holdouts. Static only."
        )
    )
    parser.add_argument(
        "--discovery-output-dir",
        type=Path,
        default=ROOT / DISCOVERY_DIR,
    )
    parser.add_argument(
        "--confirmation-output-dir",
        type=Path,
        default=ROOT / CONFIRMATION_DIR,
    )
    parser.add_argument(
        "--residual-output-dir",
        type=Path,
        default=ROOT / RESIDUAL_DIR,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    result = write_holdouts(
        discovery_dir=args.discovery_output_dir,
        confirmation_dir=args.confirmation_output_dir,
        residual_dir=args.residual_output_dir,
    )

    d = result["discovery"]["manifest"]
    c = result["confirmation"]["manifest"]
    r = result["residual_characterization"]["manifest"]

    print("RESULT=" + RESULT)
    print(
        "DISCOVERY_RANGE="
        f"{d['pair_id_first']}..{d['pair_id_last']}"
    )
    print(
        "CONFIRMATION_RANGE="
        f"{c['pair_id_first']}..{c['pair_id_last']}"
    )
    print(
        "RESIDUAL_RANGE="
        f"{r['pair_id_first']}..{r['pair_id_last']}"
    )
    print("SOURCE_PAIR_COUNT_PER_COHORT=300")
    print("ROW_COUNT_PER_COHORT=1800")
    print("PRIOR_PAIR_COVERAGE_001_3900=True")
    print("FINAL_PAIR_COVERAGE_001_4800=True")
    print("CROSS_COHORT_PAIR_OVERLAP=0")
    print("CROSS_COHORT_CLAIM_OVERLAP=0")
    print("CROSS_COHORT_EVIDENCE_OVERLAP=0")
    print("LABELS_PRESENT=False")
    print("RESPONSE_FIELDS_PRESENT=False")
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
