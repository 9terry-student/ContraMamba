from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as xg1
from scripts import build_controlled_v5 as historical_builder
from scripts import materialize_reason_router_gen4_six_cell_contrast as historical_materializer


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
DESIGN_FREEZE_COMMIT = "a31d2bc5ab4b939f52e969c89f3783feb9c3b233"
DESIGN_REL = xg1.DESIGN_REL
DESIGN_SHA256 = xg1.DESIGN_SHA256

PRIOR_HOLDOUT_REL = Path(
    "data/reason_router_gen4_large_correction_prospective_holdout_v1/"
    "synthetic_reason_router_six_cell.jsonl"
)
PRIOR_HOLDOUT_SHA256 = (
    "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
)

MANIFEST_SCHEMA = "GEN4_XG1_STRUCTURAL_MANIFEST_V1"
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

DISCOVERY_REFERENCE_PAIR_COUNT = 300
FULL_REFERENCE_SOURCE_COUNT = 600

FORBIDDEN_FIELDS = xg1.FORBIDDEN_FIELDS | frozenset(
    {
        "absolute_anchor_token_index",
        "terminal_index",
        "post4_eligible",
        "exclusion_code",
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    }
)


class XG1ValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG1ValidationError(message)


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
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path.name}")
            rows.append(value)
    return rows


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XG1ValidationError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_authority() -> None:
    branch = git("branch", "--show-current")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")

    head = git("rev-parse", "HEAD")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", DESIGN_FREEZE_COMMIT, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "DESIGN_FREEZE_NOT_ANCESTOR")

    design = ROOT / DESIGN_REL
    require(design.is_file(), "DESIGN_MISSING")
    require(sha256_file(design) == DESIGN_SHA256, "DESIGN_SHA256")


def _casefold_strings(values: Iterable[str]) -> set[str]:
    return {value.casefold() for value in values}


def _historical_source_values() -> set[str]:
    facts = historical_builder.fact_templates_for_count(FULL_REFERENCE_SOURCE_COUNT)
    require(len(facts) == FULL_REFERENCE_SOURCE_COUNT, "HISTORICAL_SOURCE_COUNT")
    values: set[str] = set()
    for fact in facts:
        for key, raw in fact.items():
            if key == "pair_id":
                continue
            if isinstance(raw, str) and raw:
                values.add(raw)
    return values


def _xg1_source_values(facts: Sequence[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for fact in facts:
        for field in (
            "title",
            "name",
            "role",
            "predicate",
            "object",
            "time",
            "location",
            "alternate_title",
            "alternate_name",
            "alternate_role",
            "alternate_predicate",
        ):
            raw = fact[field]
            require(isinstance(raw, str) and bool(raw), f"XG1_SOURCE_VALUE:{field}")
            values.add(raw)
    return values


def _historical_discovery_strings() -> tuple[set[str], set[str]]:
    facts = historical_builder.fact_templates_for_count(DISCOVERY_REFERENCE_PAIR_COUNT)
    require(len(facts) == DISCOVERY_REFERENCE_PAIR_COUNT, "DISCOVERY_FACT_COUNT")
    rows = historical_materializer.materialize_facts(facts)
    require(len(rows) == DISCOVERY_REFERENCE_PAIR_COUNT * 6, "DISCOVERY_ROW_COUNT")
    return (
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _prior_holdout_strings() -> tuple[set[str], set[str]]:
    path = ROOT / PRIOR_HOLDOUT_REL
    require(path.is_file(), "PRIOR_HOLDOUT_MISSING")
    require(sha256_file(path) == PRIOR_HOLDOUT_SHA256, "PRIOR_HOLDOUT_SHA256")
    rows = read_jsonl(path)
    require(len(rows) == 1800, "PRIOR_HOLDOUT_ROW_COUNT")
    pairs = {str(row["source_pair_id"]) for row in rows}
    require(len(pairs) == 300, "PRIOR_HOLDOUT_PAIR_COUNT")
    return (
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _validate_substitutions(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_pair = {str(fact["pair_id"]): fact for fact in facts}
    require(len(by_pair) == xg1.SOURCE_PAIR_COUNT, "SUBSTITUTION_FACT_COUNT")

    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in by_pair, f"ROW_PAIR_REFERENCE:{pair_id}")
        fact = by_pair[pair_id]
        cell_id = str(row["contrast_cell_id"])
        mask, substitutions = xg1.cell_spec(cell_id)
        overrides = {axis: str(fact[source]) for axis, source in substitutions}
        require(
            row["claim"] == xg1.render_statement(fact),
            f"CLAIM_RENDER:{row['row_id']}",
        )
        require(
            row["evidence"] == xg1.render_statement(fact, **overrides),
            f"EVIDENCE_RENDER:{row['row_id']}",
        )
        require(row["axis_mask"] == list(mask), f"SUBSTITUTION_MASK:{row['row_id']}")


def _production_builder_independence() -> None:
    source = (
        ROOT / "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "build_controlled_v5",
        "materialize_reason_router_gen4_six_cell_contrast",
        "FACT_TEMPLATES",
        "fact_templates_for_count",
        "_generated_fact_template",
        "._statement",
        "._paraphrase",
    )
    for token in forbidden:
        require(token not in source, f"PRODUCTION_BUILDER_DEPENDENCY:{token}")


def validate_cohort(output_dir: Path) -> dict[str, Any]:
    authenticate_authority()
    _production_builder_independence()

    source_path = output_dir / xg1.SOURCE_FILE
    row_path = output_dir / xg1.ROW_FILE
    manifest_path = output_dir / MANIFEST_FILE
    checksum_path = output_dir / CHECKSUM_FILE

    require(source_path.is_file(), "SOURCE_FILE_MISSING")
    require(row_path.is_file(), "ROW_FILE_MISSING")
    require(not manifest_path.exists(), "MANIFEST_COLLISION")
    require(not checksum_path.exists(), "CHECKSUM_COLLISION")

    facts = read_jsonl(source_path)
    rows = read_jsonl(row_path)

    xg1.validate_source_facts(facts)
    xg1.validate_materialized_rows(rows, expected_pairs=xg1.SOURCE_PAIR_COUNT)
    _validate_substitutions(facts, rows)

    require(len(facts) == 300, "SOURCE_COUNT")
    require(len(rows) == 1800, "ROW_COUNT")
    require(
        [str(fact["pair_id"]) for fact in facts] == xg1.expected_pair_ids(),
        "PAIR_IDS",
    )
    require(len({str(row["row_id"]) for row in rows}) == 1800, "ROW_ID_UNIQUENESS")
    require(not any(FORBIDDEN_FIELDS & set(row) for row in rows), "ROW_FORBIDDEN_FIELDS")
    require(not any(FORBIDDEN_FIELDS & set(fact) for fact in facts), "SOURCE_FORBIDDEN_FIELDS")

    regenerated_facts = xg1.build_source_facts()
    regenerated_rows = xg1.materialize_facts(regenerated_facts)
    require(
        source_path.read_bytes() == xg1.jsonl_bytes(regenerated_facts),
        "SOURCE_REGEN_BYTE_DRIFT",
    )
    require(
        row_path.read_bytes() == xg1.jsonl_bytes(regenerated_rows),
        "ROW_REGEN_BYTE_DRIFT",
    )

    historical_values = _historical_source_values()
    xg1_values = _xg1_source_values(facts)
    exact_inventory_overlap = sorted(xg1_values & historical_values)
    folded_historical = _casefold_strings(historical_values)
    casefold_inventory_overlap = sorted(
        value for value in xg1_values if value.casefold() in folded_historical
    )
    require(exact_inventory_overlap == [], "EXACT_SOURCE_INVENTORY_OVERLAP")
    require(casefold_inventory_overlap == [], "CASEFOLD_SOURCE_INVENTORY_OVERLAP")

    discovery_claims, discovery_evidence = _historical_discovery_strings()
    holdout_claims, holdout_evidence = _prior_holdout_strings()

    xg1_claims = {str(row["claim"]) for row in rows}
    xg1_evidence = {str(row["evidence"]) for row in rows}

    require(not (xg1_claims & discovery_claims), "DISCOVERY_CLAIM_OVERLAP")
    require(not (xg1_evidence & discovery_evidence), "DISCOVERY_EVIDENCE_OVERLAP")
    require(not (xg1_claims & holdout_claims), "HOLDOUT_CLAIM_OVERLAP")
    require(not (xg1_evidence & holdout_evidence), "HOLDOUT_EVIDENCE_OVERLAP")

    source_sha = sha256_file(source_path)
    row_sha = sha256_file(row_path)

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "result": "PASS_XG1_STRUCTURAL_INDEPENDENCE",
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "design_sha256": DESIGN_SHA256,
        "generator_family": xg1.GENERATOR_FAMILY,
        "source_pair_count": len(facts),
        "row_count": len(rows),
        "rows_per_pair": 6,
        "pair_id_first": xg1.expected_pair_ids()[0],
        "pair_id_last": xg1.expected_pair_ids()[-1],
        "source_file_sha256": source_sha,
        "row_file_sha256": row_sha,
        "prior_holdout_sha256": PRIOR_HOLDOUT_SHA256,
        "historical_reference_source_count": FULL_REFERENCE_SOURCE_COUNT,
        "discovery_reference_pair_count": DISCOVERY_REFERENCE_PAIR_COUNT,
        "xg1_unique_source_value_count": len(xg1_values),
        "historical_unique_source_value_count": len(historical_values),
        "exact_inventory_overlap_count": 0,
        "casefold_inventory_overlap_count": 0,
        "discovery_claim_overlap_count": 0,
        "discovery_evidence_overlap_count": 0,
        "prior_holdout_claim_overlap_count": 0,
        "prior_holdout_evidence_overlap_count": 0,
        "deterministic_byte_regeneration": True,
        "production_builder_uses_historical_generator": False,
        "labels_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "model_executed": False,
        "cuda_executed": False,
    }

    manifest_raw = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_raw)
    manifest_sha = sha256_bytes(manifest_raw)

    checksums = {
        xg1.SOURCE_FILE: source_sha,
        xg1.ROW_FILE: row_sha,
        MANIFEST_FILE: manifest_sha,
    }
    checksum_raw = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums.items())
    ).encode("utf-8")
    checksum_path.write_bytes(checksum_raw)

    return {
        **manifest,
        "manifest_sha256": manifest_sha,
        "checksum_sha256": sha256_bytes(checksum_raw),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Gen4-K XG1 structural independence and freeze a static manifest. "
            "No tokenizer/model/CUDA execution occurs."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_cohort(args.output_dir)
    print("RESULT=", report["result"], sep="")
    print("SOURCE_PAIR_COUNT=", report["source_pair_count"], sep="")
    print("ROW_COUNT=", report["row_count"], sep="")
    print("EXACT_INVENTORY_OVERLAP_COUNT=0")
    print("CASEFOLD_INVENTORY_OVERLAP_COUNT=0")
    print("DISCOVERY_CLAIM_OVERLAP_COUNT=0")
    print("DISCOVERY_EVIDENCE_OVERLAP_COUNT=0")
    print("PRIOR_HOLDOUT_CLAIM_OVERLAP_COUNT=0")
    print("PRIOR_HOLDOUT_EVIDENCE_OVERLAP_COUNT=0")
    print("DETERMINISTIC_BYTE_REGENERATION=True")
    print("SOURCE_FILE_SHA256=", report["source_file_sha256"], sep="")
    print("ROW_FILE_SHA256=", report["row_file_sha256"], sep="")
    print("MANIFEST_SHA256=", report["manifest_sha256"], sep="")
    print("CHECKSUM_SHA256=", report["checksum_sha256"], sep="")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
