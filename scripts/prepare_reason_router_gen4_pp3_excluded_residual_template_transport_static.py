from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT_ENV = os.environ.get("CONTRAMAMBA_REPO_ROOT")
if not _ROOT_ENV:
    raise RuntimeError("CONTRAMAMBA_REPO_ROOT is required")
ROOT = Path(_ROOT_ENV).resolve()
sys.path.insert(0, str(ROOT))

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base
from scripts import build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh
from scripts import prepare_reason_router_gen4_pp3_necessity_static as prior

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
EXPECTED_HEAD = "d3cc008fad221862e6fe9718b67b6ba0c87d0368"

DESIGN_COMMIT = EXPECTED_HEAD
DESIGN_PATH = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_template_transport_design.md"
)
DESIGN_BLOB = "8b99238182027191fa9d5f3014774ce7b2e9a342"

TEMPLATE_COMMIT = "82957bdb57cd83367293753c3b0596722810f7d3"
TEMPLATE_PATH = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_static_analysis_7a6c30f.json"
)
TEMPLATE_SHA256 = (
    "712be029cbb52b30a41e9322c58978392f0c61bb76eefa15ffb8340c91899dc6"
)

OLD_COHORTS = (
    (
        1,
        300,
        Path("data/reason_router_gen4_xg1_cross_generator_v1"),
        "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0",
        "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f",
    ),
    (
        301,
        600,
        Path("data/reason_router_gen4_xg1_fresh_specificity_v1"),
        "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7",
        "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6",
    ),
    (
        601,
        900,
        Path("data/reason_router_gen4_xg1_necessity_v1"),
        "49bec37150630d31bb5f502f49ef23ffc9a75bb93e079127c8c430aae3da6abd",
        "e03534599c07201e371eb07938d8492d22a30de39dcbbb3c0c22300a4ff94224",
    ),
    (
        901,
        1200,
        Path("data/reason_router_gen4_xg1_restoration_sufficiency_v1"),
        "2c700452d818531c46a8ffd473eb6d64d8af29f3a9284da8371f5c9eb2610c21",
        "7ec2ea86f35562394244f6df6e8b098ea3ba8a9bf358868fc614f5029746241c",
    ),
)

PAIR_FIRST = "xg1_fact_1201"
PAIR_LAST = "xg1_fact_1500"
PAIR_START = 1201
PAIR_END = 1500
PAIR_COUNT = 300
ROW_COUNT = 1800

OUTPUT_DATA_DIR = Path(
    "data/reason_router_gen4_xg1_residual_template_transport_v1"
)
OUTPUT_REPORT_DIR = Path(
    "reports/reason_router_gen4_pp3_excluded_residual_template_transport_static_preparation_d3cc008"
)

RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")
FROZEN_XG2_VECTOR = (
    1.5920587453157877e-08,
    2.0711288247409458e-08,
    -8.756974184315863e-10,
    2.9916623498998595e-08,
)
FROZEN_XG4_VECTOR = (
    1.8670022911077672e-08,
    -2.177503237973276e-08,
    1.1605587525048473e-07,
    4.28622819921435e-08,
)
FROZEN_XG2_UNIT = (
    0.4007549782451175,
    0.5213470856800944,
    -0.022043162722820656,
    0.7530649126348957,
)
FROZEN_XG4_UNIT = (
    0.14700867790791192,
    -0.17145767505443896,
    0.9138296650882098,
    0.337499714799086,
)
FROZEN_TEMPLATE_COSINE = 0.2035409975407082
TOL = 2.0e-15


class StaticPreparationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StaticPreparationError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_show_bytes(path: Path, ref: str = "HEAD") -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StaticPreparationError(
            f"GIT_SHOW_FAILURE:{ref}:{path.as_posix()}"
        ) from exc


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


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_from_bytes(raw: bytes) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{line_no}",
        )
        out.append(value)
    return out


def authenticate() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(head == EXPECTED_HEAD, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    blob = git(
        "rev-parse",
        f"HEAD:{DESIGN_PATH.as_posix()}",
    )
    require(blob == DESIGN_BLOB, f"DESIGN_BLOB_DRIFT:{blob}")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            TEMPLATE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "TEMPLATE_COMMIT_NOT_ANCESTOR")

    return {
        "branch": branch,
        "head": head,
        "design_commit": DESIGN_COMMIT,
        "design_path": DESIGN_PATH.as_posix(),
        "design_blob": blob,
        "template_commit": TEMPLATE_COMMIT,
        "template_path": TEMPLATE_PATH.as_posix(),
        "template_sha256": TEMPLATE_SHA256,
    }


def validate_old_populations() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_old_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []

    for start, end, directory, expected_source_sha, expected_rows_sha in OLD_COHORTS:
        source_path = directory / base.SOURCE_FILE
        rows_path = directory / base.ROW_FILE

        frozen_source = git_show_bytes(source_path)
        frozen_rows = git_show_bytes(rows_path)

        require(
            sha256_bytes(frozen_source) == expected_source_sha,
            f"OLD_SOURCE_SHA:{start}_{end}",
        )
        require(
            sha256_bytes(frozen_rows) == expected_rows_sha,
            f"OLD_ROWS_SHA:{start}_{end}",
        )

        facts = fresh.build_source_facts(start=start, end=end)
        rows = fresh.materialize_facts(facts)

        regen_source = base.jsonl_bytes(facts)
        regen_rows = base.jsonl_bytes(rows)

        require(
            regen_source == frozen_source,
            f"OLD_SOURCE_REGEN_DRIFT:{start}_{end}",
        )
        require(
            regen_rows == frozen_rows,
            f"OLD_ROWS_REGEN_DRIFT:{start}_{end}",
        )

        parsed_rows = jsonl_from_bytes(frozen_rows)
        require(
            len(parsed_rows) == 6 * (end - start + 1),
            f"OLD_ROW_COUNT:{start}_{end}",
        )
        all_old_rows.extend(parsed_rows)

        identities.append({
            "pair_first": f"xg1_fact_{start:03d}",
            "pair_last": f"xg1_fact_{end:03d}",
            "source_sha256": expected_source_sha,
            "rows_sha256": expected_rows_sha,
            "source_byte_regeneration_identity": True,
            "row_byte_regeneration_identity": True,
        })

    require(len(all_old_rows) == 4 * 1800, "OLD_TOTAL_ROW_COUNT")
    return all_old_rows, {
        "cohorts": identities,
        "old_pair_range": "xg1_fact_001..xg1_fact_1200",
        "old_source_pair_count": 1200,
        "old_row_count": 7200,
        "all_old_cohorts_byte_regeneration_identity": True,
    }


def build_new_population(
    old_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    facts = fresh.build_source_facts(
        start=PAIR_START,
        end=PAIR_END,
    )
    rows = fresh.materialize_facts(facts)

    require(len(facts) == PAIR_COUNT, "NEW_FACT_COUNT")
    require(len(rows) == ROW_COUNT, "NEW_ROW_COUNT")

    expected_ids = [
        f"xg1_fact_{i:03d}"
        for i in range(PAIR_START, PAIR_END + 1)
    ]
    require(
        [str(row["pair_id"]) for row in facts]
        == expected_ids,
        "NEW_PAIR_ORDER",
    )

    old_pair_ids = {str(row["source_pair_id"]) for row in old_rows}
    new_pair_ids = {str(row["source_pair_id"]) for row in rows}

    old_claims = {str(row["claim"]) for row in old_rows}
    new_claims = {str(row["claim"]) for row in rows}

    old_evidence = {str(row["evidence"]) for row in old_rows}
    new_evidence = {str(row["evidence"]) for row in rows}

    old_claim_evidence = {
        (str(row["claim"]), str(row["evidence"]))
        for row in old_rows
    }
    new_claim_evidence = {
        (str(row["claim"]), str(row["evidence"]))
        for row in rows
    }

    pair_overlap = old_pair_ids & new_pair_ids
    claim_overlap = old_claims & new_claims
    evidence_overlap = old_evidence & new_evidence
    row_overlap = old_claim_evidence & new_claim_evidence

    require(
        not pair_overlap,
        f"PAIR_ID_OVERLAP:{len(pair_overlap)}",
    )
    require(
        not claim_overlap,
        f"CLAIM_OVERLAP:{len(claim_overlap)}",
    )
    require(
        not evidence_overlap,
        f"EVIDENCE_OVERLAP:{len(evidence_overlap)}",
    )
    require(
        not row_overlap,
        f"CLAIM_EVIDENCE_ROW_OVERLAP:{len(row_overlap)}",
    )

    return facts, rows, {
        "source_raw": base.jsonl_bytes(facts),
        "rows_raw": base.jsonl_bytes(rows),
        "pair_id_overlap_with_001_1200": 0,
        "claim_overlap_with_001_1200": 0,
        "evidence_overlap_with_001_1200": 0,
        "claim_evidence_row_overlap_with_001_1200": 0,
    }


def norm(values: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def unit(values: Sequence[float]) -> list[float]:
    n = norm(values)
    require(n > 0.0, "ZERO_TEMPLATE_NORM")
    return [float(x) / n for x in values]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(
        float(x) * float(y)
        for x, y in zip(a, b, strict=True)
    )


def validate_templates() -> dict[str, Any]:
    raw = git_show_bytes(TEMPLATE_PATH, TEMPLATE_COMMIT)
    require(
        sha256_bytes(raw) == TEMPLATE_SHA256,
        "TEMPLATE_ARTIFACT_SHA256",
    )
    source = json.loads(raw.decode("utf-8-sig"))
    require(
        source["status"]
        == "STATIC_EXPLORATORY_RESIDUAL_CHARACTERIZATION_NO_INFERENCE",
        "TEMPLATE_SOURCE_STATUS",
    )
    require(
        source["geometry"]["residual_planes"]
        == list(RESIDUAL_PLANES),
        "TEMPLATE_PLANE_ORDER",
    )

    observed_xg2 = tuple(
        float(x)
        for x in source["cohorts"]["XG2_601_900"]["residual_net_vector"]
    )
    observed_xg4 = tuple(
        float(x)
        for x in source["cohorts"]["XG4_601_900"]["residual_net_vector"]
    )
    require(
        observed_xg2 == FROZEN_XG2_VECTOR,
        "XG2_TEMPLATE_VECTOR_DRIFT",
    )
    require(
        observed_xg4 == FROZEN_XG4_VECTOR,
        "XG4_TEMPLATE_VECTOR_DRIFT",
    )

    xg2_unit = unit(observed_xg2)
    xg4_unit = unit(observed_xg4)

    require(
        max(
            abs(a - b)
            for a, b in zip(xg2_unit, FROZEN_XG2_UNIT, strict=True)
        )
        <= TOL,
        "XG2_UNIT_TEMPLATE_DRIFT",
    )
    require(
        max(
            abs(a - b)
            for a, b in zip(xg4_unit, FROZEN_XG4_UNIT, strict=True)
        )
        <= TOL,
        "XG4_UNIT_TEMPLATE_DRIFT",
    )

    template_cosine = dot(xg2_unit, xg4_unit)
    require(
        abs(template_cosine - FROZEN_TEMPLATE_COSINE) <= TOL,
        "TEMPLATE_COSINE_DRIFT",
    )

    return {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_TEMPLATE_STATIC_V1",
        "result": "PASS_FROZEN_RESIDUAL_TEMPLATES",
        "source_commit": TEMPLATE_COMMIT,
        "source_path": TEMPLATE_PATH.as_posix(),
        "source_sha256": TEMPLATE_SHA256,
        "residual_plane_order": list(RESIDUAL_PLANES),
        "xg2_residual_mean_net_vector": list(observed_xg2),
        "xg4_residual_mean_net_vector": list(observed_xg4),
        "xg2_template_norm": norm(observed_xg2),
        "xg4_template_norm": norm(observed_xg4),
        "xg2_unit_template": xg2_unit,
        "xg4_unit_template": xg4_unit,
        "template_cosine": template_cosine,
        "xg1_outcome_used_for_template_construction": False,
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_inference_executed": False,
    }


def tokenizer_eligibility(
    tokenizer_snapshot: Path,
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_authority = prior.AUTHORITY_COMMIT
    old_first = prior.PAIR_FIRST
    old_last = prior.PAIR_LAST

    try:
        prior.AUTHORITY_COMMIT = DESIGN_COMMIT
        prior.PAIR_FIRST = PAIR_FIRST
        prior.PAIR_LAST = PAIR_LAST
        anchor_rows, summary = prior.tokenizer_eligibility(
            tokenizer_snapshot,
            facts,
            rows,
        )
    finally:
        prior.AUTHORITY_COMMIT = old_authority
        prior.PAIR_FIRST = old_first
        prior.PAIR_LAST = old_last

    for row in anchor_rows:
        row["schema_version"] = (
            "GEN4_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_XG1_"
            "TOKENIZER_ANCHOR_V1"
        )

    anchor_raw = b"".join(
        canonical_json_bytes(row)
        for row in anchor_rows
    )

    summary = dict(summary)
    summary.update({
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_XG1_"
            "TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "authority_commit": DESIGN_COMMIT,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
    })

    require(summary["result"] == "PASS_300_OF_300", "TOKENIZER_RESULT")
    require(
        int(summary["source_pair_count"]) == PAIR_COUNT,
        "TOKENIZER_PAIR_COUNT",
    )
    require(
        int(summary["anchor_row_count"]) == ROW_COUNT,
        "TOKENIZER_ANCHOR_COUNT",
    )
    require(
        summary["model_forward_count"] == 0,
        "TOKENIZER_MODEL_FORWARD",
    )
    require(
        summary["checkpoint_load_count"] == 0,
        "TOKENIZER_CHECKPOINT_LOAD",
    )
    require(summary["gpu_used"] is False, "TOKENIZER_GPU")

    return anchor_rows, summary


def write_outputs(
    provenance: Mapping[str, Any],
    old_identity: Mapping[str, Any],
    population: Mapping[str, Any],
    templates: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    data_dir = ROOT / OUTPUT_DATA_DIR
    report_dir = ROOT / OUTPUT_REPORT_DIR

    require(
        not data_dir.exists(),
        f"DATA_OUTPUT_COLLISION:{data_dir}",
    )
    require(
        not report_dir.exists(),
        f"REPORT_OUTPUT_COLLISION:{report_dir}",
    )

    data_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    source_raw = population["source_raw"]
    rows_raw = population["rows_raw"]

    structural = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_XG1_STRUCTURAL_V1",
        "result":
            "PASS_XG1_1201_1500_RESIDUAL_TEMPLATE_TRANSPORT_STRUCTURAL",
        "design_commit": DESIGN_COMMIT,
        "generator_family": base.GENERATOR_FAMILY,
        "pair_id_first": PAIR_FIRST,
        "pair_id_last": PAIR_LAST,
        "source_pair_count": PAIR_COUNT,
        "row_count": ROW_COUNT,
        "rows_per_pair": 6,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(rows_raw),
        "prior_population_identity": dict(old_identity),
        "pair_id_overlap_with_001_1200":
            population["pair_id_overlap_with_001_1200"],
        "claim_overlap_with_001_1200":
            population["claim_overlap_with_001_1200"],
        "evidence_overlap_with_001_1200":
            population["evidence_overlap_with_001_1200"],
        "claim_evidence_row_overlap_with_001_1200":
            population["claim_evidence_row_overlap_with_001_1200"],
        "deterministic_generator_semantics": True,
        "labels_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }

    data_payloads = {
        base.SOURCE_FILE: source_raw,
        base.ROW_FILE: rows_raw,
        "structural_manifest.json": pretty_json_bytes(structural),
    }
    data_hashes: dict[str, str] = {}
    for name, raw in data_payloads.items():
        (data_dir / name).write_bytes(raw)
        data_hashes[name] = sha256_bytes(raw)

    (data_dir / "SHA256SUMS.txt").write_bytes(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(data_hashes.items())
        ).encode("utf-8")
    )

    anchor_raw = b"".join(
        canonical_json_bytes(row)
        for row in anchor_rows
    )

    preparation = {
        "schema_version":
            "GEN4_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_STATIC_PREPARATION_V1",
        "result":
            "PASS_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_STATIC_PREPARATION",
        "design": dict(provenance),
        "population": {
            "data_dir": OUTPUT_DATA_DIR.as_posix(),
            "pair_id_first": PAIR_FIRST,
            "pair_id_last": PAIR_LAST,
            "source_pair_count": PAIR_COUNT,
            "row_count": ROW_COUNT,
            "source_sha256": data_hashes[base.SOURCE_FILE],
            "rows_sha256": data_hashes[base.ROW_FILE],
            "structural_manifest_sha256":
                data_hashes["structural_manifest.json"],
            "pair_id_overlap_with_001_1200": 0,
            "claim_overlap_with_001_1200": 0,
            "evidence_overlap_with_001_1200": 0,
            "claim_evidence_row_overlap_with_001_1200": 0,
        },
        "templates": {
            "result": templates["result"],
            "source_commit": templates["source_commit"],
            "source_sha256": templates["source_sha256"],
            "residual_plane_order":
                templates["residual_plane_order"],
            "xg2_unit_template":
                templates["xg2_unit_template"],
            "xg4_unit_template":
                templates["xg4_unit_template"],
            "template_cosine":
                templates["template_cosine"],
            "xg1_outcome_used_for_template_construction": False,
        },
        "tokenizer_eligibility": {
            "result": eligibility["result"],
            "anchor_manifest_sha256":
                eligibility["anchor_manifest_sha256"],
            "source_pair_count":
                eligibility["source_pair_count"],
            "anchor_row_count":
                eligibility["anchor_row_count"],
            "tokenizer_revision": prior.TOKENIZER_REVISION,
        },
        "future_execution_contract": {
            "scientific_model_forward_budget": 12000,
            "pair_count": 300,
            "directions_per_pair": 10,
            "forwards_per_direction": 4,
            "planned_gpu_count": 2,
            "planned_pair_shards": [
                {
                    "gpu": 0,
                    "pair_first": "xg1_fact_1201",
                    "pair_last": "xg1_fact_1350",
                    "pair_count": 150,
                    "scientific_model_forward_budget": 6000,
                },
                {
                    "gpu": 1,
                    "pair_first": "xg1_fact_1351",
                    "pair_last": "xg1_fact_1500",
                    "pair_count": 150,
                    "scientific_model_forward_budget": 6000,
                },
            ],
            "execution_authorized": False,
        },
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "primary_inference_executed": False,
    }

    report_payloads = {
        "template_manifest.json": pretty_json_bytes(templates),
        "tokenizer_anchor_manifest.jsonl": anchor_raw,
        "tokenizer_eligibility_summary.json":
            pretty_json_bytes(eligibility),
        "preparation_manifest.json": pretty_json_bytes(preparation),
    }

    report_hashes: dict[str, str] = {}
    for name, raw in report_payloads.items():
        (report_dir / name).write_bytes(raw)
        report_hashes[name] = sha256_bytes(raw)

    (report_dir / "SHA256SUMS.txt").write_bytes(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(report_hashes.items())
        ).encode("utf-8")
    )

    return {
        "data_hashes": data_hashes,
        "report_hashes": report_hashes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only static preparation for prospective PP3-excluded "
            "residual template transport on fresh XG1 1201..1500."
        )
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    tokenizer_snapshot = args.tokenizer_snapshot.resolve()
    require(
        tokenizer_snapshot.is_dir(),
        f"TOKENIZER_SNAPSHOT_MISSING:{tokenizer_snapshot}",
    )
    require(
        not tokenizer_snapshot.is_relative_to(ROOT.resolve()),
        "TOKENIZER_SNAPSHOT_MUST_BE_EXTERNAL_TO_REPO",
    )

    provenance = authenticate()
    old_rows, old_identity = validate_old_populations()
    facts, rows, population = build_new_population(old_rows)
    templates = validate_templates()
    anchor_rows, eligibility = tokenizer_eligibility(
        tokenizer_snapshot,
        facts,
        rows,
    )
    outputs = write_outputs(
        provenance,
        old_identity,
        population,
        templates,
        anchor_rows,
        eligibility,
    )

    print(
        "RESULT="
        "PASS_PP3_EXCLUDED_RESIDUAL_TEMPLATE_TRANSPORT_STATIC_PREPARATION"
    )
    print(f"PAIR_ID_FIRST={PAIR_FIRST}")
    print(f"PAIR_ID_LAST={PAIR_LAST}")
    print("SOURCE_PAIR_COUNT=300")
    print("ROW_COUNT=1800")
    print("OLD_XG1_001_1200_BYTE_REGENERATION_IDENTITY=True")
    print("PAIR_ID_OVERLAP_WITH_001_1200=0")
    print("CLAIM_OVERLAP_WITH_001_1200=0")
    print("EVIDENCE_OVERLAP_WITH_001_1200=0")
    print("CLAIM_EVIDENCE_ROW_OVERLAP_WITH_001_1200=0")
    print(f"TEMPLATE_SOURCE_SHA256={TEMPLATE_SHA256}")
    print(
        "TEMPLATE_COSINE="
        f"{templates['template_cosine']:.17g}"
    )
    print(
        "XG2_UNIT_TEMPLATE="
        f"{templates['xg2_unit_template']}"
    )
    print(
        "XG4_UNIT_TEMPLATE="
        f"{templates['xg4_unit_template']}"
    )
    print(
        "TOKENIZER_ELIGIBILITY="
        f"{eligibility['result']}"
    )
    print(
        "ANCHOR_ROW_COUNT="
        f"{eligibility['anchor_row_count']}"
    )
    print(
        "ANCHOR_MANIFEST_SHA256="
        f"{eligibility['anchor_manifest_sha256']}"
    )
    print(
        "SOURCE_SHA256="
        f"{outputs['data_hashes'][base.SOURCE_FILE]}"
    )
    print(
        "ROWS_SHA256="
        f"{outputs['data_hashes'][base.ROW_FILE]}"
    )
    print(
        "STRUCTURAL_MANIFEST_SHA256="
        f"{outputs['data_hashes']['structural_manifest.json']}"
    )
    print(
        "TEMPLATE_MANIFEST_SHA256="
        f"{outputs['report_hashes']['template_manifest.json']}"
    )
    print(
        "PREPARATION_MANIFEST_SHA256="
        f"{outputs['report_hashes']['preparation_manifest.json']}"
    )
    print("PLANNED_GPU_COUNT=2")
    print("PLANNED_GPU0_PAIRS=xg1_fact_1201..xg1_fact_1350")
    print("PLANNED_GPU1_PAIRS=xg1_fact_1351..xg1_fact_1500")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("EXECUTION_AUTHORIZED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
