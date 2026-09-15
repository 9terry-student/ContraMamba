from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts import build_controlled_v5 as historical_builder
from scripts import materialize_reason_router_gen4_six_cell_contrast as historical_materializer
from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as cohorts


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
DESIGN_FREEZE_COMMIT = cohorts.DESIGN_FREEZE_COMMIT
DESIGN_REL = cohorts.DESIGN_REL
DESIGN_SHA256 = cohorts.DESIGN_SHA256

PRIOR_HOLDOUT_REL = Path(
    "data/reason_router_gen4_large_correction_prospective_holdout_v1/"
    "synthetic_reason_router_six_cell.jsonl"
)
PRIOR_HOLDOUT_SHA256 = (
    "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
)

XG1_DIR = Path("data/reason_router_gen4_xg1_cross_generator_v1")
XG1_SOURCE_SHA256 = (
    "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
)
XG1_ROW_SHA256 = (
    "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
)
XG1_STRUCTURAL_MANIFEST_SHA256 = (
    "f7c881dd1a4a400e600eea4b03b46e05a48bf0da07d29905462fc3543d8af822"
)

DISCOVERY_REFERENCE_PAIR_COUNT = 300
FULL_REFERENCE_SOURCE_COUNT = 600

FAMILY_MANIFEST_SCHEMA = "GEN4_PREVALENCE_FAMILY_STRUCTURAL_MANIFEST_V1"
CROSS_MANIFEST_SCHEMA = "GEN4_PREVALENCE_CROSS_FAMILY_STRUCTURAL_MANIFEST_V1"
MANIFEST_FILE = "structural_manifest.json"
FAMILY_CHECKSUM_FILE = "SHA256SUMS.txt"
CROSS_MANIFEST_FILE = "cross_family_structural_manifest.json"
ROOT_CHECKSUM_FILE = "SHA256SUMS.txt"

FORBIDDEN_FIELDS = cohorts.FORBIDDEN_FIELDS


class PrevalenceStructuralValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrevalenceStructuralValidationError(message)


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
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}")
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
        raise PrevalenceStructuralValidationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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


def _source_values(
    facts: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> set[str]:
    values: set[str] = set()
    fields = (
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
    )
    for fact in facts:
        for field in fields:
            raw = fact[field]
            require(
                isinstance(raw, str) and bool(raw),
                f"{label}_SOURCE_VALUE:{field}",
            )
            values.add(raw)
    return values


def _historical_source_values() -> set[str]:
    facts = historical_builder.fact_templates_for_count(FULL_REFERENCE_SOURCE_COUNT)
    require(
        len(facts) == FULL_REFERENCE_SOURCE_COUNT,
        "HISTORICAL_SOURCE_COUNT",
    )
    values: set[str] = set()
    for fact in facts:
        for key, raw in fact.items():
            if key == "pair_id":
                continue
            if isinstance(raw, str) and raw:
                values.add(raw)
    return values


def _historical_discovery_strings() -> tuple[set[str], set[str]]:
    facts = historical_builder.fact_templates_for_count(
        DISCOVERY_REFERENCE_PAIR_COUNT
    )
    require(
        len(facts) == DISCOVERY_REFERENCE_PAIR_COUNT,
        "DISCOVERY_FACT_COUNT",
    )
    rows = historical_materializer.materialize_facts(facts)
    require(
        len(rows) == DISCOVERY_REFERENCE_PAIR_COUNT * 6,
        "DISCOVERY_ROW_COUNT",
    )
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
    require(
        len({str(row["source_pair_id"]) for row in rows}) == 300,
        "PRIOR_HOLDOUT_PAIR_COUNT",
    )
    return (
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _xg1_reference() -> tuple[set[str], set[str], set[str]]:
    source_path = ROOT / XG1_DIR / cohorts.SOURCE_FILE
    row_path = ROOT / XG1_DIR / cohorts.ROW_FILE
    manifest_path = ROOT / XG1_DIR / MANIFEST_FILE

    require(source_path.is_file(), "XG1_SOURCE_MISSING")
    require(row_path.is_file(), "XG1_ROWS_MISSING")
    require(manifest_path.is_file(), "XG1_MANIFEST_MISSING")
    require(sha256_file(source_path) == XG1_SOURCE_SHA256, "XG1_SOURCE_SHA256")
    require(sha256_file(row_path) == XG1_ROW_SHA256, "XG1_ROW_SHA256")
    require(
        sha256_file(manifest_path) == XG1_STRUCTURAL_MANIFEST_SHA256,
        "XG1_STRUCTURAL_MANIFEST_SHA256",
    )

    facts = read_jsonl(source_path)
    rows = read_jsonl(row_path)
    require(len(facts) == 300, "XG1_SOURCE_COUNT")
    require(len(rows) == 1800, "XG1_ROW_COUNT")
    return (
        _source_values(facts, label="XG1"),
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def _validate_substitutions(
    spec: cohorts.FamilySpec,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    by_pair = {str(fact["pair_id"]): fact for fact in facts}
    require(
        len(by_pair) == cohorts.SOURCE_PAIR_COUNT,
        f"{spec.key}:SUBSTITUTION_FACT_COUNT",
    )
    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in by_pair, f"{spec.key}:ROW_PAIR_REFERENCE:{pair_id}")
        fact = by_pair[pair_id]
        cell_id = str(row["contrast_cell_id"])
        mask, substitutions = cohorts.cell_spec(cell_id)
        overrides = {axis: str(fact[source]) for axis, source in substitutions}
        require(
            row["claim"] == cohorts.render_statement(spec, fact),
            f"{spec.key}:CLAIM_RENDER:{row['row_id']}",
        )
        require(
            row["evidence"]
            == cohorts.render_statement(spec, fact, **overrides),
            f"{spec.key}:EVIDENCE_RENDER:{row['row_id']}",
        )
        require(
            row["axis_mask"] == list(mask),
            f"{spec.key}:SUBSTITUTION_MASK:{row['row_id']}",
        )


def _production_builder_independence() -> None:
    source = (
        ROOT
        / "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "build_controlled_v5",
        "materialize_reason_router_gen4_six_cell_contrast",
        "build_reason_router_gen4_xg1_cross_generator_cohort",
        "FACT_TEMPLATES",
        "fact_templates_for_count",
        "_generated_fact_template",
        "._statement",
        "._paraphrase",
    )
    for token in forbidden:
        require(
            token not in source,
            f"PRODUCTION_BUILDER_DEPENDENCY:{token}",
        )


def _validate_one_family(
    output_dir: Path,
    spec: cohorts.FamilySpec,
    earlier_values: set[str],
    earlier_claims: set[str],
    earlier_evidence: set[str],
) -> dict[str, Any]:
    family_dir = output_dir / spec.key
    source_path = family_dir / cohorts.SOURCE_FILE
    row_path = family_dir / cohorts.ROW_FILE
    manifest_path = family_dir / MANIFEST_FILE
    checksum_path = family_dir / FAMILY_CHECKSUM_FILE

    require(family_dir.is_dir(), f"{spec.key}:FAMILY_DIR_MISSING")
    require(source_path.is_file(), f"{spec.key}:SOURCE_FILE_MISSING")
    require(row_path.is_file(), f"{spec.key}:ROW_FILE_MISSING")
    require(not manifest_path.exists(), f"{spec.key}:MANIFEST_COLLISION")
    require(not checksum_path.exists(), f"{spec.key}:CHECKSUM_COLLISION")

    facts = read_jsonl(source_path)
    rows = read_jsonl(row_path)

    cohorts.validate_source_facts(spec, facts)
    cohorts.validate_materialized_rows(
        spec,
        rows,
        expected_pairs=cohorts.SOURCE_PAIR_COUNT,
    )
    _validate_substitutions(spec, facts, rows)

    require(len(facts) == 300, f"{spec.key}:SOURCE_COUNT")
    require(len(rows) == 1800, f"{spec.key}:ROW_COUNT")
    require(
        [str(fact["pair_id"]) for fact in facts]
        == cohorts.expected_pair_ids(spec),
        f"{spec.key}:PAIR_IDS",
    )
    require(
        len({str(row["row_id"]) for row in rows}) == 1800,
        f"{spec.key}:ROW_ID_UNIQUENESS",
    )
    require(
        not any(FORBIDDEN_FIELDS & set(row) for row in rows),
        f"{spec.key}:ROW_FORBIDDEN_FIELDS",
    )
    require(
        not any(FORBIDDEN_FIELDS & set(fact) for fact in facts),
        f"{spec.key}:SOURCE_FORBIDDEN_FIELDS",
    )

    regenerated_facts = cohorts.build_source_facts(spec)
    regenerated_rows = cohorts.materialize_facts(spec, regenerated_facts)
    require(
        source_path.read_bytes() == cohorts.jsonl_bytes(regenerated_facts),
        f"{spec.key}:SOURCE_REGEN_BYTE_DRIFT",
    )
    require(
        row_path.read_bytes() == cohorts.jsonl_bytes(regenerated_rows),
        f"{spec.key}:ROW_REGEN_BYTE_DRIFT",
    )

    values = _source_values(facts, label=spec.key.upper())
    exact_inventory_overlap = sorted(values & earlier_values)
    folded_earlier = _casefold_strings(earlier_values)
    casefold_inventory_overlap = sorted(
        value for value in values if value.casefold() in folded_earlier
    )
    require(
        exact_inventory_overlap == [],
        f"{spec.key}:EARLIER_EXACT_SOURCE_INVENTORY_OVERLAP",
    )
    require(
        casefold_inventory_overlap == [],
        f"{spec.key}:EARLIER_CASEFOLD_SOURCE_INVENTORY_OVERLAP",
    )

    claims = {str(row["claim"]) for row in rows}
    evidence = {str(row["evidence"]) for row in rows}
    require(
        not (claims & earlier_claims),
        f"{spec.key}:EARLIER_CLAIM_OVERLAP",
    )
    require(
        not (evidence & earlier_evidence),
        f"{spec.key}:EARLIER_EVIDENCE_OVERLAP",
    )

    source_sha = sha256_file(source_path)
    row_sha = sha256_file(row_path)

    return {
        "spec": spec,
        "facts": facts,
        "rows": rows,
        "values": values,
        "claims": claims,
        "evidence": evidence,
        "source_sha": source_sha,
        "row_sha": row_sha,
        "source_path": source_path,
        "row_path": row_path,
        "manifest_path": manifest_path,
        "checksum_path": checksum_path,
    }


def _validate_cross_family(
    validated: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    exact_inventory = 0
    casefold_inventory = 0
    claim_overlap = 0
    evidence_overlap = 0

    for left, right in combinations(cohorts.FAMILY_KEYS, 2):
        lv = set(validated[left]["values"])
        rv = set(validated[right]["values"])
        exact = lv & rv
        folded_r = _casefold_strings(rv)
        folded = {value for value in lv if value.casefold() in folded_r}
        claims = set(validated[left]["claims"]) & set(validated[right]["claims"])
        evidence = (
            set(validated[left]["evidence"])
            & set(validated[right]["evidence"])
        )

        require(not exact, f"CROSS_FAMILY_EXACT_INVENTORY:{left}:{right}")
        require(not folded, f"CROSS_FAMILY_CASEFOLD_INVENTORY:{left}:{right}")
        require(not claims, f"CROSS_FAMILY_CLAIM_OVERLAP:{left}:{right}")
        require(
            not evidence,
            f"CROSS_FAMILY_EVIDENCE_OVERLAP:{left}:{right}",
        )

        exact_inventory += len(exact)
        casefold_inventory += len(folded)
        claim_overlap += len(claims)
        evidence_overlap += len(evidence)

    return {
        "exact_inventory_overlap_count": exact_inventory,
        "casefold_inventory_overlap_count": casefold_inventory,
        "claim_overlap_count": claim_overlap,
        "evidence_overlap_count": evidence_overlap,
    }


def validate_cohorts(output_dir: Path) -> dict[str, Any]:
    authenticate_authority()
    _production_builder_independence()

    require(output_dir.is_dir(), "OUTPUT_DIR_MISSING")
    require(
        not (output_dir / CROSS_MANIFEST_FILE).exists(),
        "CROSS_MANIFEST_COLLISION",
    )
    require(
        not (output_dir / ROOT_CHECKSUM_FILE).exists(),
        "ROOT_CHECKSUM_COLLISION",
    )

    historical_values = _historical_source_values()
    discovery_claims, discovery_evidence = _historical_discovery_strings()
    holdout_claims, holdout_evidence = _prior_holdout_strings()
    xg1_values, xg1_claims, xg1_evidence = _xg1_reference()

    earlier_values = historical_values | xg1_values
    earlier_claims = discovery_claims | holdout_claims | xg1_claims
    earlier_evidence = discovery_evidence | holdout_evidence | xg1_evidence

    validated: dict[str, dict[str, Any]] = {}
    for key in cohorts.FAMILY_KEYS:
        validated[key] = _validate_one_family(
            output_dir,
            cohorts.family_spec(key),
            earlier_values,
            earlier_claims,
            earlier_evidence,
        )

    cross_counts = _validate_cross_family(validated)
    builder_path = (
        ROOT
        / "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
    )
    validator_path = (
        ROOT
        / "scripts/validate_reason_router_gen4_generator_family_prevalence_cohorts.py"
    )
    builder_sha = sha256_file(builder_path)
    validator_sha = sha256_file(validator_path)

    family_reports: dict[str, dict[str, Any]] = {}
    for key in cohorts.FAMILY_KEYS:
        item = validated[key]
        spec = item["spec"]
        manifest: dict[str, Any] = {
            "schema_version": FAMILY_MANIFEST_SCHEMA,
            "result": "PASS_PREVALENCE_FAMILY_STRUCTURAL_INDEPENDENCE",
            "design_freeze_commit": DESIGN_FREEZE_COMMIT,
            "design_sha256": DESIGN_SHA256,
            "family_key": key,
            "generator_family": spec.generator_family,
            "source_pair_count": 300,
            "row_count": 1800,
            "rows_per_pair": 6,
            "pair_id_first": cohorts.expected_pair_ids(spec)[0],
            "pair_id_last": cohorts.expected_pair_ids(spec)[-1],
            "source_file_sha256": item["source_sha"],
            "row_file_sha256": item["row_sha"],
            "builder_sha256": builder_sha,
            "validator_sha256": validator_sha,
            "prior_holdout_sha256": PRIOR_HOLDOUT_SHA256,
            "xg1_source_sha256": XG1_SOURCE_SHA256,
            "xg1_row_sha256": XG1_ROW_SHA256,
            "xg1_structural_manifest_sha256": XG1_STRUCTURAL_MANIFEST_SHA256,
            "historical_reference_source_count": FULL_REFERENCE_SOURCE_COUNT,
            "discovery_reference_pair_count": DISCOVERY_REFERENCE_PAIR_COUNT,
            "unique_source_value_count": len(item["values"]),
            "earlier_exact_inventory_overlap_count": 0,
            "earlier_casefold_inventory_overlap_count": 0,
            "earlier_claim_overlap_count": 0,
            "earlier_evidence_overlap_count": 0,
            "deterministic_byte_regeneration": True,
            "production_builder_uses_historical_generator": False,
            "production_builder_uses_xg1_generator": False,
            "labels_present": False,
            "model_geometry_present": False,
            "endpoint_values_present": False,
            "response_fields_present": False,
            "tokenizer_executed": False,
            "model_executed": False,
            "cuda_executed": False,
            "intervention_executed": False,
        }
        raw = canonical_json_bytes(manifest)
        item["manifest_path"].write_bytes(raw)
        manifest_sha = sha256_bytes(raw)

        family_checksums = {
            cohorts.SOURCE_FILE: item["source_sha"],
            cohorts.ROW_FILE: item["row_sha"],
            MANIFEST_FILE: manifest_sha,
        }
        checksum_raw = "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(family_checksums.items())
        ).encode("utf-8")
        item["checksum_path"].write_bytes(checksum_raw)

        family_reports[key] = {
            **manifest,
            "manifest_sha256": manifest_sha,
            "checksum_sha256": sha256_bytes(checksum_raw),
        }

    cross_manifest: dict[str, Any] = {
        "schema_version": CROSS_MANIFEST_SCHEMA,
        "result": "PASS_PREVALENCE_CROSS_FAMILY_STRUCTURAL_INDEPENDENCE",
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "design_sha256": DESIGN_SHA256,
        "families": list(cohorts.FAMILY_KEYS),
        "family_count": 3,
        "source_pair_count_per_family": 300,
        "row_count_per_family": 1800,
        "total_source_pair_count": 900,
        "total_row_count": 5400,
        "builder_sha256": builder_sha,
        "validator_sha256": validator_sha,
        "cross_family_exact_inventory_overlap_count": cross_counts[
            "exact_inventory_overlap_count"
        ],
        "cross_family_casefold_inventory_overlap_count": cross_counts[
            "casefold_inventory_overlap_count"
        ],
        "cross_family_claim_overlap_count": cross_counts["claim_overlap_count"],
        "cross_family_evidence_overlap_count": cross_counts[
            "evidence_overlap_count"
        ],
        "all_families_deterministic_byte_regeneration": True,
        "tokenizer_executed": False,
        "model_executed": False,
        "cuda_executed": False,
        "intervention_executed": False,
        "family_artifacts": {
            key: {
                "source_file_sha256": family_reports[key]["source_file_sha256"],
                "row_file_sha256": family_reports[key]["row_file_sha256"],
                "manifest_sha256": family_reports[key]["manifest_sha256"],
                "checksum_sha256": family_reports[key]["checksum_sha256"],
            }
            for key in cohorts.FAMILY_KEYS
        },
    }
    cross_raw = canonical_json_bytes(cross_manifest)
    cross_path = output_dir / CROSS_MANIFEST_FILE
    cross_path.write_bytes(cross_raw)
    cross_sha = sha256_bytes(cross_raw)

    root_checksums: dict[str, str] = {
        CROSS_MANIFEST_FILE: cross_sha,
    }
    for key in cohorts.FAMILY_KEYS:
        for name in (
            cohorts.SOURCE_FILE,
            cohorts.ROW_FILE,
            MANIFEST_FILE,
            FAMILY_CHECKSUM_FILE,
        ):
            path = output_dir / key / name
            root_checksums[f"{key}/{name}"] = sha256_file(path)

    root_checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(root_checksums.items())
    ).encode("utf-8")
    (output_dir / ROOT_CHECKSUM_FILE).write_bytes(root_checksum_raw)

    return {
        **cross_manifest,
        "cross_manifest_sha256": cross_sha,
        "root_checksum_sha256": sha256_bytes(root_checksum_raw),
        "family_reports": family_reports,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate XG2/XG3/XG4 structural independence and cross-family "
            "non-overlap for Gen4-K prevalence transportability. "
            "No tokenizer, model, CUDA, baseline geometry, or intervention runs."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_cohorts(args.output_dir)
    print("RESULT=", report["result"], sep="")
    print("FAMILY_COUNT=3")
    print("TOTAL_SOURCE_PAIR_COUNT=900")
    print("TOTAL_ROW_COUNT=5400")
    for key in cohorts.FAMILY_KEYS:
        item = report["family_reports"][key]
        print(f"{key.upper()}_SOURCE_SHA256={item['source_file_sha256']}")
        print(f"{key.upper()}_ROW_SHA256={item['row_file_sha256']}")
        print(f"{key.upper()}_MANIFEST_SHA256={item['manifest_sha256']}")
        print(f"{key.upper()}_CHECKSUM_SHA256={item['checksum_sha256']}")
    print("CROSS_FAMILY_EXACT_INVENTORY_OVERLAP_COUNT=0")
    print("CROSS_FAMILY_CASEFOLD_INVENTORY_OVERLAP_COUNT=0")
    print("CROSS_FAMILY_CLAIM_OVERLAP_COUNT=0")
    print("CROSS_FAMILY_EVIDENCE_OVERLAP_COUNT=0")
    print("CROSS_MANIFEST_SHA256=", report["cross_manifest_sha256"], sep="")
    print("ROOT_CHECKSUM_SHA256=", report["root_checksum_sha256"], sep="")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")
    print("INTERVENTION_EXECUTED=False")


if __name__ == "__main__":
    main()
