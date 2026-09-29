#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_reason_router_gen4_mamba1_five_scale_ladder_holdouts as ladder
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as eligibility

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"
REQUIRED_ANCESTOR = "c8dc7a4bb69dd4e86f4bbbcf48b88940bc007cd8"

FROZEN_DEPENDENCY_BLOBS = {
    "scripts/build_reason_router_gen4_mamba1_five_scale_ladder_holdouts.py":
        "4ff5ee70f463cae94d5188999b350efecbe212bb",
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py":
        "6c98ce022ca134e385db28851fd364dc6daff423",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py":
        "7f51683ecbf60d63cdc0399d2b0e89f8d42dd0ab",
    "reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/"
    "native_backbone_identity_manifest_candidate.jsonl":
        "351b28ad14c6a1b963559165195d0f5942772be3",
}

PLANE_FILES = {
    "pp3_plus": (
        Path("reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89/pp3_plus.f64le"),
        "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff",
    ),
    "pp3_minus": (
        Path("reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89/pp3_minus.f64le"),
        "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7",
    ),
    "pp5_plus": (
        Path("reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab/pp5_plus.f64le"),
        "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2",
    ),
    "pp5_minus": (
        Path("reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab/pp5_minus.f64le"),
        "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855",
    ),
}

CHECKPOINT_PATH = (
    "reports/reason_router_gen3_grouped_factorial_runs/seed180/"
    "G3-GROUP-D-HALF/selected_checkpoint.pt"
)
CHECKPOINT_SHA256 = "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
NATIVE_BACKBONE_SIGNATURE_SHA256 = (
    "81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415"
)

COHORTS = (
    ("construction", 7801, 8100),
    ("necessity_confirmation", 8101, 8400),
    ("restoration_confirmation", 8401, 8700),
)

OUTPUT_DIRS = {
    "construction": Path("data/reason_router_gen5_phase1b_xg1_construction_v1"),
    "necessity_confirmation": Path(
        "data/reason_router_gen5_phase1b_xg1_necessity_confirmation_v1"
    ),
    "restoration_confirmation": Path(
        "data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1"
    ),
}
REPORT_DIR = Path(
    "reports/reason_router_gen5_phase1b_static_preparation_c8dc7a4_v1"
)

SOURCE_FILE = ladder.SOURCE_FILE
ROW_FILE = ladder.ROW_FILE
STRUCTURAL_MANIFEST_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
TOKENIZER_SUMMARY_FILE = "tokenizer_eligibility_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
PREP_SUMMARY_FILE = "static_preparation_summary.json"

FORBIDDEN_OUTCOME_FIELDS = ladder.FORBIDDEN_OUTCOME_FIELDS
ANCHOR_SCHEMA = "GEN5_PHASE1B_TOKENIZER_ANCHOR_ELIGIBILITY_V1"


class Phase1BStaticPrepError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase1BStaticPrepError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase1BStaticPrepError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def authenticate_repository() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(
        git_rc("merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head) == 0,
        "REQUIRED_ANCESTOR_NOT_PRESENT",
    )

    observed_blobs: dict[str, str] = {}
    for path, expected_blob in FROZEN_DEPENDENCY_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(observed == expected_blob, f"FROZEN_DEPENDENCY_BLOB_DRIFT:{path}")
        observed_blobs[path] = observed

    return {
        "branch": branch,
        "head": head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "frozen_dependency_blobs": observed_blobs,
    }


def verify_plane_identities() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (rel, expected) in PLANE_FILES.items():
        path = ROOT / rel
        require(path.is_file(), f"MISSING_PLANE:{name}:{rel}")
        observed = sha256_file(path)
        require(observed == expected, f"PLANE_SHA256:{name}")
        out[name] = {
            "path": rel.as_posix(),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return out


def verify_checkpoint_binding() -> dict[str, Any]:
    manifest_rel = Path(
        "reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/"
        "native_backbone_identity_manifest_candidate.jsonl"
    )
    rows = [
        json.loads(line)
        for line in (ROOT / manifest_rel).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if int(row.get("seed", -1)) == 180
        and row.get("arm") == "G3-GROUP-D-HALF"
    ]
    require(len(matches) == 1, "REPRESENTATIVE_BACKBONE_MANIFEST_ROW")
    row = matches[0]
    require(row["checkpoint_path"] == CHECKPOINT_PATH, "CHECKPOINT_PATH_BINDING")
    require(row["expected_checkpoint_sha256"] == CHECKPOINT_SHA256, "CHECKPOINT_SHA_BINDING")
    require(
        row["native_backbone_signature_sha256"] == NATIVE_BACKBONE_SIGNATURE_SHA256,
        "BACKBONE_SIGNATURE_BINDING",
    )
    require(row["identity_vs_reference"] == "IDENTICAL", "BACKBONE_REFERENCE_IDENTITY")

    extraction = (
        ROOT / "scripts/reason_router_gen4_native_mamba_state_extraction.py"
    ).read_text(encoding="utf-8")
    require(CHECKPOINT_SHA256 in extraction, "EXTRACTION_CHECKPOINT_SHA_BINDING")
    require(
        "G3-GROUP-D-HALF/selected_checkpoint.pt" in extraction,
        "EXTRACTION_CHECKPOINT_PATH_BINDING",
    )

    return {
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "native_backbone_signature_sha256": NATIVE_BACKBONE_SIGNATURE_SHA256,
        "checkpoint_payload_loaded": False,
        "model_instantiated": False,
    }


def row_sets(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(row["pair_id"]) for row in facts},
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def prior_inventory_through_7800() -> tuple[set[str], set[str], set[str]]:
    inventory = ladder.prior_inventory_through_6000()
    for _scale, _role, first, last in sorted(ladder.COHORTS, key=lambda x: x[2]):
        facts, rows = ladder.build_population(first, last)
        current = row_sets(facts, rows)
        require(not (inventory[0] & current[0]), f"PRIOR_PAIR_OVERLAP:{first}:{last}")
        require(not (inventory[1] & current[1]), f"PRIOR_CLAIM_OVERLAP:{first}:{last}")
        require(not (inventory[2] & current[2]), f"PRIOR_EVIDENCE_OVERLAP:{first}:{last}")
        inventory = tuple(a | b for a, b in zip(inventory, current))
    require(
        inventory[0] == set(ladder.expected_pair_ids(1, 7800)),
        "PRIOR_PAIR_COVERAGE_001_7800",
    )
    return inventory  # type: ignore[return-value]


def role_contract(role: str) -> dict[str, Any]:
    if role == "construction":
        return {
            "prospective_use": "R22_C22_construction_only",
            "scientific_confirmation_allowed": False,
            "R22_construction_allowed": True,
            "C22_construction_allowed": True,
            "confirmation_response_access_allowed": False,
            "rank_fixed": 2,
        }
    if role == "necessity_confirmation":
        return {
            "prospective_use": "R22_local_necessity_confirmation_only",
            "scientific_confirmation_allowed": True,
            "R22_construction_allowed": False,
            "C22_construction_allowed": False,
            "construction_responses_access_allowed": False,
            "primary_endpoint": "D_NEC22=Q_C22_CONTROL-Q_R22_NEUTRALIZED",
            "primary_p_value_count": 1,
        }
    if role == "restoration_confirmation":
        return {
            "prospective_use": "R22_restoration_confirmation_only",
            "scientific_confirmation_allowed": True,
            "R22_construction_allowed": False,
            "C22_construction_allowed": False,
            "construction_responses_access_allowed": False,
            "necessity_responses_access_allowed": False,
            "primary_endpoint": "D_SUF22=Q_R22_RESTORED-Q_C22_REPLACEMENT",
            "primary_p_value_count": 1,
        }
    raise Phase1BStaticPrepError(f"UNKNOWN_ROLE:{role}")


def build_cohorts() -> tuple[
    dict[str, dict[str, Any]],
    tuple[set[str], set[str], set[str]],
]:
    inventory = prior_inventory_through_7800()
    payloads: dict[str, dict[str, Any]] = {}
    prior_last = 7800

    for role, first, last in COHORTS:
        facts_a, rows_a = ladder.build_population(first, last)
        facts_b, rows_b = ladder.build_population(first, last)
        source_raw = ladder.jsonl_bytes(facts_a)
        row_raw = ladder.jsonl_bytes(rows_a)
        require(source_raw == ladder.jsonl_bytes(facts_b), f"{role}:SOURCE_REGEN")
        require(row_raw == ladder.jsonl_bytes(rows_b), f"{role}:ROW_REGEN")
        require(len(facts_a) == 300, f"{role}:PAIR_COUNT")
        require(len(rows_a) == 1800, f"{role}:ROW_COUNT")

        for row in [*facts_a, *rows_a]:
            require(
                not (FORBIDDEN_OUTCOME_FIELDS & set(row)),
                f"{role}:FORBIDDEN_OUTCOME_FIELD",
            )

        current = row_sets(facts_a, rows_a)
        require(len(current[0]) == 300, f"{role}:UNIQUE_PAIRS")
        require(len(current[1]) == 300, f"{role}:UNIQUE_CLAIMS")
        require(len(current[2]) == 1800, f"{role}:UNIQUE_EVIDENCE")
        require(not (inventory[0] & current[0]), f"{role}:PAIR_OVERLAP")
        require(not (inventory[1] & current[1]), f"{role}:CLAIM_OVERLAP")
        require(not (inventory[2] & current[2]), f"{role}:EVIDENCE_OVERLAP")

        manifest = {
            "schema_version": "GEN5_PHASE1B_XG1_STRUCTURAL_HOLDOUT_V1",
            "result": "PASS_GEN5_PHASE1B_XG1_STRUCTURAL_HOLDOUT",
            "phase1b_design_commit": REQUIRED_ANCESTOR,
            "role": role,
            "generator_family": facts_a[0]["generator_family"],
            "deterministic_generator_semantics": True,
            "pair_id_first": f"xg1_fact_{first}",
            "pair_id_last": f"xg1_fact_{last}",
            "source_pair_count": 300,
            "row_count": 1800,
            "rows_per_pair": 6,
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
            "training_executed": False,
            "backward_executed": False,
            "cohort_replacement_allowed": False,
            "row_filtering_allowed": False,
            **role_contract(role),
        }
        payloads[role] = {
            "facts": facts_a,
            "rows": rows_a,
            "source_raw": source_raw,
            "row_raw": row_raw,
            "structural_manifest": manifest,
        }
        inventory = tuple(a | b for a, b in zip(inventory, current))
        prior_last = last

    require(
        inventory[0] == set(ladder.expected_pair_ids(1, 8700)),
        "FINAL_PAIR_COVERAGE_001_8700",
    )
    return payloads, inventory  # type: ignore[return-value]


def compute_tokenizer_eligibility(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts_by_id = {str(f["pair_id"]): f for f in facts}
    require(len(facts_by_id) == 300, f"{role}:FACT_ID_COUNT")

    anchors: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in facts_by_id, f"{role}:MISSING_FACT:{pair_id}")
        produced = eligibility.analyze_required_anchors_for_row(
            row, facts_by_id[pair_id], tokenizer
        )
        for item in produced:
            item = dict(item)
            item["schema_version"] = ANCHOR_SCHEMA
            item["phase1b_role"] = role
            anchors.append(item)

    require(len(anchors) == 1800, f"{role}:ANCHOR_ROW_COUNT")
    counts = Counter(str(row["anchor_name"]) for row in anchors)
    require(
        dict(counts) == eligibility.ANCHOR_EXPECTED_COUNTS,
        f"{role}:ANCHOR_COUNTS:{dict(counts)}",
    )
    exclusion = Counter(
        str(row["exclusion_code"])
        for row in anchors
        if row["exclusion_code"] is not None
    )
    eligible = sum(bool(row["post4_eligible"]) for row in anchors)
    require(eligible == 1800, f"{role}:ANCHOR_INELIGIBILITY:{dict(exclusion)}")

    lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in anchors
    }
    require(len(lookup) == 1800, f"{role}:ANCHOR_KEY_DUPLICATE")

    mismatch = 0
    for pair_id in facts_by_id:
        for cell in eligibility.TARGET_IDENTITY_NAME_CELLS:
            a = lookup[(pair_id, cell, "A_IDENTITY")]
            b = lookup[(pair_id, cell, "A_NAME")]
            mismatch += int(
                a["absolute_anchor_token_index"] != b["absolute_anchor_token_index"]
            )
    require(mismatch == 0, f"{role}:IDENTITY_NAME_COORD_MISMATCH:{mismatch}")

    anchor_raw = jsonl_bytes(anchors)
    summary = {
        "schema_version": "GEN5_PHASE1B_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "result": "PASS_GEN5_PHASE1B_TOKENIZER_ANCHOR_ELIGIBILITY",
        "role": role,
        "source_pair_count": 300,
        "required_anchor_row_count": 1800,
        "eligible_anchor_row_count": eligible,
        "eligible_anchor_counts": dict(counts),
        "exclusion_counts": dict(exclusion),
        "identity_name_coordinate_mismatch_count": mismatch,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
        "tokenizer": dict(tokenizer_provenance),
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    return anchors, summary


def checksum_text(files: Mapping[str, bytes]) -> str:
    return "".join(f"{sha256_bytes(raw)}  {name}\n" for name, raw in sorted(files.items()))


def write_outputs(
    payloads: Mapping[str, Mapping[str, Any]],
    anchors_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer_summaries: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    planes: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    for rel in [*OUTPUT_DIRS.values(), REPORT_DIR]:
        require(not (ROOT / rel).exists(), f"OUTPUT_COLLISION:{rel}")

    cohort_records: dict[str, Any] = {}
    for role, _first, _last in COHORTS:
        out = ROOT / OUTPUT_DIRS[role]
        out.mkdir(parents=True, exist_ok=False)

        structural_raw = canonical_json_bytes(payloads[role]["structural_manifest"])
        anchor_raw = jsonl_bytes(anchors_by_role[role])
        token_raw = canonical_json_bytes(tokenizer_summaries[role])

        files = {
            SOURCE_FILE: payloads[role]["source_raw"],
            ROW_FILE: payloads[role]["row_raw"],
            STRUCTURAL_MANIFEST_FILE: structural_raw,
            ANCHOR_FILE: anchor_raw,
            TOKENIZER_SUMMARY_FILE: token_raw,
        }
        for name, raw in files.items():
            (out / name).write_bytes(raw)
        sums = checksum_text(files)
        (out / CHECKSUM_FILE).write_text(sums, encoding="utf-8")

        cohort_records[role] = {
            "directory": OUTPUT_DIRS[role].as_posix(),
            "files": {
                name: {"sha256": sha256_bytes(raw), "bytes": len(raw)}
                for name, raw in sorted(files.items())
            },
            "checksums_sha256": sha256_bytes(sums.encode("utf-8")),
        }

    REPORT_DIR_ABS = ROOT / REPORT_DIR
    REPORT_DIR_ABS.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "GEN5_PHASE1B_STATIC_PREPARATION_SUMMARY_V1",
        "result": "PASS_GEN5_PHASE1B_STATIC_PREPARATION",
        "phase1b_design_commit": REQUIRED_ANCESTOR,
        "execution_head": provenance["head"],
        "repository": dict(provenance),
        "cohorts": cohort_records,
        "prior_pair_coverage": "xg1_fact_001..xg1_fact_7800",
        "final_pair_coverage": "xg1_fact_001..xg1_fact_8700",
        "plane_identities": dict(planes),
        "checkpoint_binding": dict(checkpoint_binding),
        "R22_rank_frozen": 2,
        "R22_constructed": False,
        "C22_constructed": False,
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "backward_executed": False,
        "task_evaluation_executed": False,
        "cuda_executed": False,
        "kaggle_executed": False,
        "parent_native_backbone_identity_audit": "DEFERRED_NOT_REQUIRED_FOR_PHASE1B_STATIC_PREP",
        "next_stage": "FREEZE_STATIC_PREPARATION_THEN_IMPLEMENT_R22_C22_CONSTRUCTION_RUNNER",
    }
    raw = canonical_json_bytes(summary)
    (REPORT_DIR_ABS / PREP_SUMMARY_FILE).write_bytes(raw)
    sums = checksum_text({PREP_SUMMARY_FILE: raw})
    (REPORT_DIR_ABS / CHECKSUM_FILE).write_text(sums, encoding="utf-8")
    return summary


def run(tokenizer_snapshot: Path | None) -> dict[str, Any]:
    provenance = authenticate_repository()
    planes = verify_plane_identities()
    checkpoint_binding = verify_checkpoint_binding()
    payloads, _inventory = build_cohorts()

    tokenizer, tokenizer_provenance = eligibility.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    anchors_by_role: dict[str, list[dict[str, Any]]] = {}
    token_summaries: dict[str, dict[str, Any]] = {}
    for role, _first, _last in COHORTS:
        anchors, summary = compute_tokenizer_eligibility(
            payloads[role]["facts"],
            payloads[role]["rows"],
            tokenizer,
            tokenizer_provenance,
            role,
        )
        anchors_by_role[role] = anchors
        token_summaries[role] = summary

    return write_outputs(
        payloads,
        anchors_by_role,
        token_summaries,
        provenance,
        planes,
        checkpoint_binding,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only Gen5 Phase 1B cohort/identity/tokenizer static preparation."
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=None,
        help="Canonical Mamba-130M tokenizer snapshot. Defaults to frozen HF cache revision.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args.tokenizer_snapshot)
    print("RESULT=" + summary["result"])
    for role, first, last in COHORTS:
        print(f"{role.upper()}=xg1_fact_{first}..xg1_fact_{last}")
        cohort = summary["cohorts"][role]
        print(
            f"{role.upper()}_SOURCE_SHA256="
            + cohort["files"][SOURCE_FILE]["sha256"]
        )
        print(
            f"{role.upper()}_ROWS_SHA256="
            + cohort["files"][ROW_FILE]["sha256"]
        )
        print(
            f"{role.upper()}_ANCHOR_SHA256="
            + cohort["files"][ANCHOR_FILE]["sha256"]
        )
    print("PRIOR_PAIR_COVERAGE=xg1_fact_001..xg1_fact_7800")
    print("FINAL_PAIR_COVERAGE=xg1_fact_001..xg1_fact_8700")
    print("CHECKPOINT_SHA256=" + CHECKPOINT_SHA256)
    print("NATIVE_BACKBONE_SIGNATURE_SHA256=" + NATIVE_BACKBONE_SIGNATURE_SHA256)
    print("MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("CUDA_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
