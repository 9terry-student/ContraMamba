from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base
from scripts import build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_SCHEMA = "GEN4_XG1_FRESH_SPECIFICITY_STRUCTURAL_MANIFEST_V1"
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

FORBIDDEN_FIELDS = base.FORBIDDEN_FIELDS | frozenset(
    {
        "absolute_anchor_token_index",
        "terminal_index",
        "post4_eligible",
        "exclusion_code",
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
        "J_PP3_PLUS",
        "J_PP3_MINUS",
        "J_PP5_PLUS",
        "J_PP5_MINUS",
        "C_PP3",
        "C_PP5",
        "D_SPEC",
    }
)


class FreshXG1ValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshXG1ValidationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), "JSONL_OBJECT")
        rows.append(value)
    return rows


def validate_substitutions(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_pair = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(len(by_pair) == 300, "SUBSTITUTION_FACT_COUNT")

    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in by_pair, f"ROW_PAIR_REFERENCE:{pair_id}")

        fact = by_pair[pair_id]
        cell_id = str(row["contrast_cell_id"])
        mask, substitutions = base.cell_spec(cell_id)

        overrides = {
            axis: str(fact[source])
            for axis, source in substitutions
        }

        require(
            row["claim"] == base.render_statement(fact),
            f"CLAIM_RENDER:{row['row_id']}",
        )
        require(
            row["evidence"]
            == base.render_statement(fact, **overrides),
            f"EVIDENCE_RENDER:{row['row_id']}",
        )
        require(
            row["axis_mask"] == list(mask),
            f"SUBSTITUTION_MASK:{row['row_id']}",
        )


def validate_cohort(output_dir: Path) -> dict[str, Any]:
    fresh.authenticate_static_authority()
    original_identity = fresh.prove_original_byte_identity()

    source_path = output_dir / fresh.SOURCE_FILE
    row_path = output_dir / fresh.ROW_FILE
    manifest_path = output_dir / MANIFEST_FILE
    checksum_path = output_dir / CHECKSUM_FILE

    require(source_path.is_file(), "SOURCE_FILE_MISSING")
    require(row_path.is_file(), "ROW_FILE_MISSING")
    require(not manifest_path.exists(), "MANIFEST_COLLISION")
    require(not checksum_path.exists(), "CHECKSUM_COLLISION")

    source_raw = source_path.read_bytes()
    row_raw = row_path.read_bytes()

    facts = read_jsonl_bytes(source_raw)
    rows = read_jsonl_bytes(row_raw)

    fresh.validate_source_facts(facts)
    base.validate_materialized_rows(
        rows,
        expected_pairs=fresh.SOURCE_PAIR_COUNT,
    )
    validate_substitutions(facts, rows)

    require(len(facts) == 300, "SOURCE_COUNT")
    require(len(rows) == 1800, "ROW_COUNT")

    expected_pairs = fresh.expected_pair_ids()
    require(
        [str(fact["pair_id"]) for fact in facts]
        == expected_pairs,
        "PAIR_IDS",
    )

    observed_row_pairs = []
    for offset in range(0, len(rows), 6):
        block = rows[offset : offset + 6]
        require(len(block) == 6, "ROW_BLOCK_SIZE")
        pair_id = str(block[0]["source_pair_id"])
        require(
            all(str(row["source_pair_id"]) == pair_id for row in block),
            f"ROW_BLOCK_PAIR:{pair_id}",
        )
        observed_row_pairs.append(pair_id)

    require(
        observed_row_pairs == expected_pairs,
        "ROW_PAIR_ORDER",
    )
    require(
        len({str(row["row_id"]) for row in rows}) == 1800,
        "ROW_ID_UNIQUENESS",
    )

    require(
        not any(FORBIDDEN_FIELDS & set(row) for row in rows),
        "ROW_FORBIDDEN_FIELDS",
    )
    require(
        not any(FORBIDDEN_FIELDS & set(fact) for fact in facts),
        "SOURCE_FORBIDDEN_FIELDS",
    )

    regenerated_facts = fresh.build_source_facts()
    regenerated_rows = fresh.materialize_facts(
        regenerated_facts
    )

    require(
        source_raw == base.jsonl_bytes(regenerated_facts),
        "SOURCE_REGEN_BYTE_DRIFT",
    )
    require(
        row_raw == base.jsonl_bytes(regenerated_rows),
        "ROW_REGEN_BYTE_DRIFT",
    )

    frozen_original_rows = read_jsonl_bytes(
        fresh.git_show_bytes(
            fresh.DESIGN_FREEZE_COMMIT,
            fresh.ORIGINAL_ROW_REL,
        )
    )

    original_claims = {
        str(row["claim"])
        for row in frozen_original_rows
    }
    original_evidence = {
        str(row["evidence"])
        for row in frozen_original_rows
    }

    fresh_claims = {
        str(row["claim"])
        for row in rows
    }
    fresh_evidence = {
        str(row["evidence"])
        for row in rows
    }

    claim_overlap = sorted(
        fresh_claims & original_claims
    )
    evidence_overlap = sorted(
        fresh_evidence & original_evidence
    )

    require(
        claim_overlap == [],
        "ORIGINAL_CLAIM_OVERLAP",
    )
    require(
        evidence_overlap == [],
        "ORIGINAL_EVIDENCE_OVERLAP",
    )

    source_sha = sha256_bytes(source_raw)
    row_sha = sha256_bytes(row_raw)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "result": "PASS_XG1_FRESH_SPECIFICITY_STRUCTURAL_FREEZE",
        "design_freeze_commit": fresh.DESIGN_FREEZE_COMMIT,
        "design_git_blob": fresh.DESIGN_GIT_BLOB,
        "generator_family": base.GENERATOR_FAMILY,
        "source_pair_count": 300,
        "row_count": 1800,
        "rows_per_pair": 6,
        "pair_id_first": "xg1_fact_301",
        "pair_id_last": "xg1_fact_600",
        "source_file_sha256": source_sha,
        "row_file_sha256": row_sha,
        "original_001_300_source_sha256": (
            original_identity["source_sha256"]
        ),
        "original_001_300_row_sha256": (
            original_identity["row_sha256"]
        ),
        "original_001_300_source_byte_identity": True,
        "original_001_300_row_byte_identity": True,
        "original_claim_overlap_count": 0,
        "original_evidence_overlap_count": 0,
        "deterministic_byte_regeneration": True,
        "labels_present": False,
        "token_fields_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }

    manifest_raw = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    manifest_sha = sha256_bytes(manifest_raw)

    checksums = {
        fresh.SOURCE_FILE: source_sha,
        fresh.ROW_FILE: row_sha,
        MANIFEST_FILE: manifest_sha,
    }

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(checksums.items())
    ).encode("utf-8")

    checksum_path.write_bytes(checksum_raw)

    return {
        **manifest,
        "manifest_sha256": manifest_sha,
        "checksum_sha256": sha256_bytes(checksum_raw),
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and freeze fresh XG1 301..600 structural cohort. "
            "Static only."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_cohort(args.output_dir)

    print("RESULT=", report["result"], sep="")
    print("PAIR_ID_FIRST=xg1_fact_301")
    print("PAIR_ID_LAST=xg1_fact_600")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("ORIGINAL_001_300_SOURCE_BYTE_IDENTITY=True")
    print("ORIGINAL_001_300_ROW_BYTE_IDENTITY=True")
    print("ORIGINAL_CLAIM_OVERLAP_COUNT=0")
    print("ORIGINAL_EVIDENCE_OVERLAP_COUNT=0")
    print("DETERMINISTIC_BYTE_REGENERATION=True")
    print(
        "SOURCE_FILE_SHA256=",
        report["source_file_sha256"],
        sep="",
    )
    print(
        "ROW_FILE_SHA256=",
        report["row_file_sha256"],
        sep="",
    )
    print(
        "MANIFEST_SHA256=",
        report["manifest_sha256"],
        sep="",
    )
    print(
        "CHECKSUM_SHA256=",
        report["checksum_sha256"],
        sep="",
    )
    print("TOKENIZER_EXECUTED=False")
    print("CHECKPOINT_LOADED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
