from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-behavioral-restoration-bridge"
DESIGN_COMMIT = "9ea1617f4485fa0b6093df0c70aa88a42260c710"
DESIGN_PATH = "reports/reason_router_gen4_seed181_behavioral_restoration_bridge_design.md"
DESIGN_BLOB = "9f4e7493893b1d324da3c8fbb981010c380edfae"

FIRST_PAIR = 2701
LAST_PAIR = 3000
PAIR_COUNT = 300
ROWS_PER_PAIR = 6
EXPECTED_ROW_COUNT = PAIR_COUNT * ROWS_PER_PAIR

OUTPUT_DIR = Path("data/reason_router_gen4_xg1_behavioral_bridge_v1")
SOURCE_FILE = base.SOURCE_FILE
ROW_FILE = base.ROW_FILE
MANIFEST_FILE = "structural_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
MANIFEST_SCHEMA = "GEN4_SEED181_BEHAVIORAL_BRIDGE_XG1_STRUCTURAL_V1"

# Exact prior XG1 inventory through 2700. These file hashes were frozen before
# this prospective population was materialized.
PRIOR_COHORTS: tuple[tuple[str, int, int, str, str], ...] = (
    (
        "data/reason_router_gen4_xg1_cross_generator_v1",
        1,
        300,
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0",
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f",
    ),
    (
        "data/reason_router_gen4_xg1_fresh_specificity_v1",
        301,
        600,
        "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7",
        "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6",
    ),
    (
        "data/reason_router_gen4_xg1_necessity_v1",
        601,
        900,
        "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd",
        "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224",
    ),
    (
        "data/reason_router_gen4_xg1_restoration_sufficiency_v1",
        901,
        1200,
        "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21",
        "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c",
    ),
    (
        "data/reason_router_gen4_xg1_residual_template_transport_v1",
        1201,
        1500,
        "9b93eff2399b7f90fb3d63f28f834fb986efcac9d347038c4b265519062d71a6",
        "f60cc028e8276279499290976f18ba503fc7a35a4232c1d768b822075d87ec5e",
    ),
    (
        "data/reason_router_gen4_xg1_residual_aggregate_necessity_v1",
        1501,
        1800,
        "b8b2186fb4bdf5d9781efb9ab6eb56eb8a3d51052618ab7cf7d257aa6e72df21",
        "16d9646cab9fa10d0241db2eeecf15164a04eb80740b873c2db6e8f02acab578",
    ),
    (
        "data/reason_router_gen4_xg1_residual_individual_plane_necessity_v1",
        1801,
        2100,
        "18381b4d31bf6b5b5975edf29dd23ad9a4cfbece23cf8c119cda4fb2539fcc9f",
        "f3b67944c05e0198f9f33a3b544ddee344a3741ceccb00fe2a74191f177b18b5",
    ),
    (
        "data/reason_router_gen4_xg1_residual_individual_plane_restoration_sufficiency_v1",
        2101,
        2400,
        "d2cc7276254bb96c32ea54af3c3fb768cbe005dcd7cb329512232df18629ea81",
        "56957588464df337ad65509e377b534e49c264ddaac766e3131633dd5e2ac087",
    ),
    (
        "data/reason_router_gen4_xg1_residual_aggregate_restoration_sufficiency_v1",
        2401,
        2700,
        "eb77056732740f501066026a3b65ea3522cd203916828f9d8d56a6e746c79a87",
        "de175b9817e6b589f4580247775adf760ece7a5b929c59dc28bdad6a7fe763e7",
    ),
)


class BehavioralHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BehavioralHoldoutError("GIT_FAILURE:" + " ".join(args)) from exc


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BehavioralHoldoutError(
            f"GIT_BLOB_FAILURE:{path.as_posix()}"
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl_bytes(
    raw: bytes,
    label: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    text = raw.decode("utf-8-sig")
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{label}:{line_no}",
        )
        out.append(value)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_bytes(
        path.read_bytes(),
        path.as_posix(),
    )


def authenticate_design() -> None:
    branch = git("branch", "--show-current")
    require(branch == EXPECTED_BRANCH, f"BRANCH:{branch}")
    require(
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", DESIGN_COMMIT, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "DESIGN_COMMIT_NOT_ANCESTOR",
    )
    require(git("rev-parse", f"HEAD:{DESIGN_PATH}") == DESIGN_BLOB, "DESIGN_BLOB_DRIFT")


def expected_pair_ids(first: int = FIRST_PAIR, last: int = LAST_PAIR) -> list[str]:
    return [f"xg1_fact_{i:03d}" for i in range(first, last + 1)]


def source_fact(global_index_one_based: int) -> dict[str, Any]:
    require(global_index_one_based >= 1, "GLOBAL_INDEX")
    i = global_index_one_based - 1

    name = base.NAMES[i % len(base.NAMES)]
    alternate_name = base.NAMES[(i + 7) % len(base.NAMES)]
    title = base.TITLES[(3 * i + 1) % len(base.TITLES)]
    alternate_title = base.TITLES[(3 * i + 4) % len(base.TITLES)]
    role = base.ROLES[(5 * i + 2) % len(base.ROLES)]
    alternate_role = base.ROLES[(5 * i + 5) % len(base.ROLES)]
    predicate, alternate_predicate = base.PREDICATE_PAIRS[
        (7 * i + 3) % len(base.PREDICATE_PAIRS)
    ]
    location = base.LOCATIONS[(5 * i + 4) % len(base.LOCATIONS)]
    time = base.TIMES[(7 * i + 5) % len(base.TIMES)]
    object_text = (
        f"{base.OBJECT_STEMS[(11 * i + 6) % len(base.OBJECT_STEMS)]} "
        f"unit {global_index_one_based:03d}"
    )

    row = {
        "schema_version": base.SOURCE_SCHEMA,
        "generator_family": base.GENERATOR_FAMILY,
        "pair_id": f"xg1_fact_{global_index_one_based:03d}",
        "title": title,
        "name": name,
        "role": role,
        "predicate": predicate,
        "object": object_text,
        "time": time,
        "location": location,
        "alternate_title": alternate_title,
        "alternate_name": alternate_name,
        "alternate_role": alternate_role,
        "alternate_predicate": alternate_predicate,
    }
    base.validate_source_fact(row)
    return row


def build_population(first: int = FIRST_PAIR, last: int = LAST_PAIR) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(last >= first, "RANGE")
    facts = [source_fact(i) for i in range(first, last + 1)]
    require(
        [str(row["pair_id"]) for row in facts] == expected_pair_ids(first, last),
        "PAIR_IDS",
    )
    require(len({str(row["pair_id"]) for row in facts}) == len(facts), "PAIR_ID_UNIQUENESS")

    rows: list[dict[str, Any]] = []
    for fact in facts:
        rows.extend(base.materialize_fact(fact))
    base.validate_materialized_rows(rows, expected_pairs=len(facts))
    return facts, rows


def validate_semantic_labels(facts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    facts_by_id = {str(row["pair_id"]): row for row in facts}
    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    for pair_id, fact in facts_by_id.items():
        c0 = by_key[(pair_id, "C0_SHAM")]
        c2 = by_key[(pair_id, "C2_NAME")]
        require(c0["claim"] == c0["evidence"], f"C0_POSITIVE_IDENTITY:{pair_id}")
        require(
            c2["claim"] == base.render_statement(fact),
            f"C2_CLAIM_IDENTITY:{pair_id}",
        )
        require(
            c2["evidence"] == base.render_statement(fact, name=str(fact["alternate_name"])),
            f"C2_NAME_SWAP:{pair_id}",
        )
        require(fact["name"] != fact["alternate_name"], f"C2_NAME_DIFFERENCE:{pair_id}")


def prior_inventory() -> tuple[set[str], set[str], set[str]]:
    pair_ids: set[str] = set()
    claims: set[str] = set()
    evidence: set[str] = set()
    expected_next = 1

    for relative, first, last, source_sha, row_sha in PRIOR_COHORTS:
        require(first == expected_next, f"PRIOR_RANGE_GAP:{relative}")
        expected_next = last + 1
        source_rel = Path(relative) / SOURCE_FILE
        row_rel = Path(relative) / ROW_FILE

        # Frozen prior-cohort identity is commit/Git identity, not
        # platform-dependent working-tree line-ending identity.
        source_raw = git_blob_bytes(source_rel)
        row_raw = git_blob_bytes(row_rel)

        require(
            sha256_bytes(source_raw) == source_sha,
            f"PRIOR_SOURCE_SHA:{relative}",
        )
        require(
            sha256_bytes(row_raw) == row_sha,
            f"PRIOR_ROW_SHA:{relative}",
        )

        facts = read_jsonl_bytes(
            source_raw,
            f"HEAD:{source_rel.as_posix()}",
        )
        rows = read_jsonl_bytes(
            row_raw,
            f"HEAD:{row_rel.as_posix()}",
        )
        expected_ids = expected_pair_ids(first, last)
        require([str(x["pair_id"]) for x in facts] == expected_ids, f"PRIOR_PAIR_IDS:{relative}")
        pair_ids.update(expected_ids)
        claims.update(str(x["claim"]) for x in rows)
        evidence.update(str(x["evidence"]) for x in rows)

    require(expected_next == FIRST_PAIR, "PRIOR_COVERAGE_NOT_001_2700")
    require(len(pair_ids) == FIRST_PAIR - 1, "PRIOR_PAIR_COUNT")
    return pair_ids, claims, evidence


def validate_continuation_identity() -> None:
    # Compare against immutable Git blob bytes rather than working-tree bytes.
    # This preserves exact serialized identity across LF/CRLF checkout policy.
    latest = Path(PRIOR_COHORTS[-1][0])
    facts, rows = build_population(2401, 2700)
    require(
        jsonl_bytes(facts) == git_blob_bytes(latest / SOURCE_FILE),
        "CONTINUATION_SOURCE_IDENTITY",
    )
    require(
        jsonl_bytes(rows) == git_blob_bytes(latest / ROW_FILE),
        "CONTINUATION_ROW_IDENTITY",
    )


def build_manifest(facts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prior_pairs, prior_claims, prior_evidence = prior_inventory()
    current_pairs = {str(x["pair_id"]) for x in facts}
    current_claims = {str(x["claim"]) for x in rows}
    current_evidence = {str(x["evidence"]) for x in rows}

    pair_overlap = current_pairs & prior_pairs
    claim_overlap = current_claims & prior_claims
    evidence_overlap = current_evidence & prior_evidence
    require(not pair_overlap, "PRIOR_PAIR_OVERLAP")
    require(not claim_overlap, "PRIOR_CLAIM_OVERLAP")
    require(not evidence_overlap, "PRIOR_EVIDENCE_OVERLAP")

    validate_semantic_labels(facts, rows)
    source_raw = jsonl_bytes(facts)
    row_raw = jsonl_bytes(rows)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "result": "PASS_SEED181_BEHAVIORAL_BRIDGE_XG1_2701_3000_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": base.GENERATOR_FAMILY,
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{FIRST_PAIR:03d}",
        "pair_id_last": f"xg1_fact_{LAST_PAIR:03d}",
        "source_pair_count": PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "prior_pair_range": "xg1_fact_001..xg1_fact_2700",
        "pair_id_overlap_with_001_2700": 0,
        "claim_overlap_with_001_2700": 0,
        "evidence_overlap_with_001_2700": 0,
        "continuation_byte_identity_with_2401_2700": True,
        "labels_present": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
        "behavioral_label_contract": {
            "C0_SHAM": "SUPPORT",
            "C2_NAME": "NOT_ENTITLED",
            "labels_embedded_in_structural_rows": False,
        },
    }


def write_cohort(output_dir: Path = ROOT / OUTPUT_DIR) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    authenticate_design()
    validate_continuation_identity()

    facts_a, rows_a = build_population()
    facts_b, rows_b = build_population()
    require(jsonl_bytes(facts_a) == jsonl_bytes(facts_b), "SOURCE_REGENERATION")
    require(jsonl_bytes(rows_a) == jsonl_bytes(rows_b), "ROW_REGENERATION")

    manifest = build_manifest(facts_a, rows_a)
    source_raw = jsonl_bytes(facts_a)
    row_raw = jsonl_bytes(rows_a)
    manifest_raw = json.dumps(
        manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"

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
    return {**manifest, "artifact_sha256": hashes}


def validate_written(output_dir: Path = ROOT / OUTPUT_DIR) -> dict[str, Any]:
    required = {SOURCE_FILE, ROW_FILE, MANIFEST_FILE, CHECKSUM_FILE}
    require(output_dir.is_dir(), "OUTPUT_DIR_MISSING")
    require({p.name for p in output_dir.iterdir() if p.is_file()} == required, "OUTPUT_FILE_SET")

    checksum_rows: dict[str, str] = {}
    for line in (output_dir / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            checksum_rows[name] = digest
    require(set(checksum_rows) == required - {CHECKSUM_FILE}, "CHECKSUM_SET")
    for name, digest in checksum_rows.items():
        require(sha256_file(output_dir / name) == digest, f"CHECKSUM:{name}")

    manifest = json.loads((output_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    require(manifest["schema_version"] == MANIFEST_SCHEMA, "MANIFEST_SCHEMA")
    require(manifest["design_commit"] == DESIGN_COMMIT, "MANIFEST_DESIGN")
    require(manifest["pair_id_first"] == "xg1_fact_2701", "MANIFEST_FIRST")
    require(manifest["pair_id_last"] == "xg1_fact_3000", "MANIFEST_LAST")
    require(int(manifest["source_pair_count"]) == PAIR_COUNT, "MANIFEST_PAIR_COUNT")
    require(int(manifest["row_count"]) == EXPECTED_ROW_COUNT, "MANIFEST_ROW_COUNT")
    require(manifest["labels_present"] is False, "MANIFEST_LABEL_BOUNDARY")
    require(manifest["model_executed"] is False, "MANIFEST_MODEL_BOUNDARY")

    facts = read_jsonl(output_dir / SOURCE_FILE)
    rows = read_jsonl(output_dir / ROW_FILE)
    require([x["pair_id"] for x in facts] == expected_pair_ids(), "WRITTEN_PAIR_IDS")
    base.validate_materialized_rows(rows, expected_pairs=PAIR_COUNT)
    validate_semantic_labels(facts, rows)
    require(sha256_file(output_dir / SOURCE_FILE) == manifest["source_file_sha256"], "MANIFEST_SOURCE_SHA")
    require(sha256_file(output_dir / ROW_FILE) == manifest["row_file_sha256"], "MANIFEST_ROW_SHA")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the prospective XG1 2701..3000 behavioral holdout. Static only.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.validate_only:
        result = validate_written(output_dir)
    else:
        result = write_cohort(output_dir)
        validate_written(output_dir)
    print("RESULT=" + str(result["result"]))
    print("PAIR_RANGE=xg1_fact_2701..xg1_fact_3000")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("PRIOR_PAIR_OVERLAP=0")
    print("PRIOR_CLAIM_OVERLAP=0")
    print("PRIOR_EVIDENCE_OVERLAP=0")
    print("TOKENIZER_EXECUTED=False")
    print("MODEL_EXECUTED=False")
    print("CUDA_EXECUTED=False")


if __name__ == "__main__":
    main()
