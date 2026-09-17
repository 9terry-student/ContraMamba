from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as base
from scripts import build_reason_router_gen4_xg1_fresh_specificity_cohort as fresh
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as legacy


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"
STRUCTURAL_FREEZE_COMMIT = "ddb1404800af6dbd89982bbbcdd8262d203577f6"
DESIGN_FREEZE_COMMIT = "0bc49ab95cbb2c8735b4bc79422d660fa64e3e01"

FRESH_BUILDER_PATH = (
    "scripts/build_reason_router_gen4_xg1_fresh_specificity_cohort.py"
)
FRESH_BUILDER_BLOB = "e4eff4bb909165ede5a792069cbc58f8b02ed002"

LEGACY_GATE_PATH = (
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py"
)
LEGACY_GATE_BLOB = "6c98ce022ca134e385db28851fd364dc6daff423"

COHORT_DIR = Path("data/reason_router_gen4_xg1_fresh_specificity_v1")
SOURCE_FACTS_PATH = COHORT_DIR / "structured_source_facts.jsonl"
ROWS_PATH = COHORT_DIR / "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_MANIFEST_PATH = COHORT_DIR / "structural_manifest.json"

EXPECTED_SOURCE_FACTS_SHA256 = (
    "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7"
)
EXPECTED_ROWS_SHA256 = (
    "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6"
)
EXPECTED_STRUCTURAL_MANIFEST_SHA256 = (
    "92ae0137641691c2a9d0254e87739e3e89d90556dc0579a2877433b86ff97dfb"
)

EXPECTED_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = 1800
EXPECTED_ANCHOR_ROWS = 1800

PAIR_ID_FIRST = "xg1_fact_301"
PAIR_ID_LAST = "xg1_fact_600"

ANCHOR_MANIFEST_SCHEMA = (
    "GEN4_XG1_FRESH_SPECIFICITY_TOKENIZER_ANCHOR_ELIGIBILITY_V1"
)
SUMMARY_SCHEMA = (
    "GEN4_XG1_FRESH_SPECIFICITY_TOKENIZER_ANCHOR_ELIGIBILITY_SUMMARY_V1"
)


class FreshEligibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshEligibilityError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshEligibilityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repository() -> dict[str, str]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )

    for commit, label in (
        (STRUCTURAL_FREEZE_COMMIT, "STRUCTURAL_FREEZE"),
        (DESIGN_FREEZE_COMMIT, "DESIGN_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    observed_fresh_builder_blob = git(
        "rev-parse",
        f"HEAD:{FRESH_BUILDER_PATH}",
    )
    require(
        observed_fresh_builder_blob == FRESH_BUILDER_BLOB,
        "FRESH_BUILDER_BLOB_DRIFT:"
        f"{observed_fresh_builder_blob}",
    )

    observed_legacy_gate_blob = git(
        "rev-parse",
        f"HEAD:{LEGACY_GATE_PATH}",
    )
    require(
        observed_legacy_gate_blob == LEGACY_GATE_BLOB,
        "LEGACY_GATE_BLOB_DRIFT:"
        f"{observed_legacy_gate_blob}",
    )

    for path, label in (
        (FRESH_BUILDER_PATH, "FRESH_BUILDER"),
        (LEGACY_GATE_PATH, "LEGACY_GATE"),
    ):
        for cached, suffix in (
            (False, "WORKTREE"),
            (True, "INDEX"),
        ):
            args = ["git", "diff"]
            if cached:
                args.append("--cached")
            args += ["--quiet", "--", path]

            rc = subprocess.call(
                args,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            require(rc == 0, f"{label}_{suffix}_DRIFT")

    return {
        "branch": branch,
        "head": head,
        "structural_freeze_commit": STRUCTURAL_FREEZE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "fresh_builder_blob": FRESH_BUILDER_BLOB,
        "legacy_gate_blob": LEGACY_GATE_BLOB,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        isinstance(value, dict),
        f"JSON_OBJECT_REQUIRED:{path}",
    )
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT_REQUIRED:{path}:{line_no}",
        )
        rows.append(value)

    return rows


def validate_structural_manifest(
    manifest: Mapping[str, Any],
) -> None:
    exact = {
        "schema_version":
            "GEN4_XG1_FRESH_SPECIFICITY_STRUCTURAL_MANIFEST_V1",
        "result":
            "PASS_XG1_FRESH_SPECIFICITY_STRUCTURAL_FREEZE",
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "generator_family": base.GENERATOR_FAMILY,
        "source_pair_count": EXPECTED_PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": 6,
        "pair_id_first": PAIR_ID_FIRST,
        "pair_id_last": PAIR_ID_LAST,
        "source_file_sha256": EXPECTED_SOURCE_FACTS_SHA256,
        "row_file_sha256": EXPECTED_ROWS_SHA256,
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

    for key, expected in exact.items():
        require(
            manifest.get(key) == expected,
            f"STRUCTURAL_MANIFEST_DRIFT:{key}",
        )


def load_frozen_fresh_inputs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    source_path = ROOT / SOURCE_FACTS_PATH
    rows_path = ROOT / ROWS_PATH
    manifest_path = ROOT / STRUCTURAL_MANIFEST_PATH

    for path in (source_path, rows_path, manifest_path):
        require(
            path.is_file(),
            f"MISSING_FROZEN_INPUT:{path}",
        )

    require(
        sha256_file(source_path)
        == EXPECTED_SOURCE_FACTS_SHA256,
        "SOURCE_FACTS_SHA256_MISMATCH",
    )
    require(
        sha256_file(rows_path)
        == EXPECTED_ROWS_SHA256,
        "ROWS_SHA256_MISMATCH",
    )
    require(
        sha256_file(manifest_path)
        == EXPECTED_STRUCTURAL_MANIFEST_SHA256,
        "STRUCTURAL_MANIFEST_SHA256_MISMATCH",
    )

    manifest = _load_json(manifest_path)
    validate_structural_manifest(manifest)

    facts = _load_jsonl(source_path)
    rows = _load_jsonl(rows_path)

    fresh.validate_source_facts(facts)
    base.validate_materialized_rows(
        rows,
        expected_pairs=EXPECTED_PAIR_COUNT,
    )

    require(
        len(facts) == EXPECTED_PAIR_COUNT,
        "SOURCE_FACT_COUNT",
    )
    require(
        len(rows) == EXPECTED_ROW_COUNT,
        "ROW_COUNT",
    )

    expected_ids = [
        f"xg1_fact_{index:03d}"
        for index in range(301, 601)
    ]
    require(
        [str(fact["pair_id"]) for fact in facts]
        == expected_ids,
        "PAIR_ID_ORDER",
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_pair_id"])].append(
            str(row["contrast_cell_id"])
        )

    for pair_id in expected_ids:
        require(
            grouped[pair_id] == list(base.CELL_IDS),
            f"SIX_CELL_ORDER_DRIFT:{pair_id}",
        )

    return facts, rows, manifest


def serialize_anchor_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    if not rows:
        return b""

    text = "\n".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    ) + "\n"

    return text.encode("utf-8")


def compute_eligibility(
    tokenizer_snapshot: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance = authenticate_repository()
    facts, rows, _manifest = load_frozen_fresh_inputs()

    tokenizer, tokenizer_provenance = (
        legacy.load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(
        len(facts_by_id) == EXPECTED_PAIR_COUNT,
        "FACT_ID_COUNT_MISMATCH",
    )

    anchor_rows: list[dict[str, Any]] = []

    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(
            pair_id in facts_by_id,
            f"MISSING_FACT_FOR_PAIR:{pair_id}",
        )

        produced = legacy.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair_id],
            tokenizer,
        )

        for anchor in produced:
            anchor = dict(anchor)
            anchor["schema_version"] = ANCHOR_MANIFEST_SCHEMA
            anchor_rows.append(anchor)

    require(
        len(anchor_rows) == EXPECTED_ANCHOR_ROWS,
        "ANCHOR_ROW_COUNT_MISMATCH",
    )

    anchor_counts = Counter(
        str(row["anchor_name"])
        for row in anchor_rows
    )
    require(
        dict(anchor_counts)
        == legacy.ANCHOR_EXPECTED_COUNTS,
        f"ANCHOR_COUNT_MISMATCH:{dict(anchor_counts)}",
    )

    eligible_anchor_counts = Counter(
        str(row["anchor_name"])
        for row in anchor_rows
        if row["post4_eligible"]
    )

    exclusion_counts = Counter(
        str(row["exclusion_code"])
        for row in anchor_rows
        if row["exclusion_code"] is not None
    )

    pair_ok = {
        str(fact["pair_id"]): True
        for fact in facts
    }

    for row in anchor_rows:
        if not row["post4_eligible"]:
            pair_ok[str(row["source_pair_id"])] = False

    event_lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in anchor_rows
    }

    require(
        len(event_lookup) == EXPECTED_ANCHOR_ROWS,
        "DUPLICATE_ANCHOR_EVENT_KEY",
    )

    mismatches: list[dict[str, Any]] = []

    for pair_id in pair_ok:
        for cell_id in legacy.TARGET_IDENTITY_NAME_CELLS:
            identity = event_lookup[
                (pair_id, cell_id, "A_IDENTITY")
            ]
            name = event_lookup[
                (pair_id, cell_id, "A_NAME")
            ]

            if (
                identity["absolute_anchor_token_index"]
                != name["absolute_anchor_token_index"]
            ):
                pair_ok[pair_id] = False
                mismatches.append(
                    {
                        "source_pair_id": pair_id,
                        "contrast_cell_id": cell_id,
                        "identity_absolute_anchor_token_index":
                            identity[
                                "absolute_anchor_token_index"
                            ],
                        "name_absolute_anchor_token_index":
                            name[
                                "absolute_anchor_token_index"
                            ],
                    }
                )

    complete_pair_count = sum(pair_ok.values())

    all_eligible = (
        complete_pair_count == EXPECTED_PAIR_COUNT
        and sum(eligible_anchor_counts.values())
        == EXPECTED_ANCHOR_ROWS
        and not mismatches
    )

    verdict = (
        "PASS_300_OF_300"
        if all_eligible
        else "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"
    )

    claim_truncation_count = sum(
        1
        for row in rows
        if len(
            legacy._encoding(
                tokenizer,
                str(row["claim"]),
            )[0]
        )
        > legacy.CLAIM_BUDGET
    )

    evidence_truncation_count = sum(
        1
        for row in rows
        if len(
            legacy._encoding(
                tokenizer,
                str(row["evidence"]),
            )[0]
        )
        > legacy.EVIDENCE_BUDGET
    )

    manifest_bytes = serialize_anchor_manifest(
        anchor_rows
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "phase":
            "XG1_FRESH_SPECIFICITY_TOKENIZER_ANCHOR_ELIGIBILITY",
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
        "provenance": provenance,
        "frozen_input": {
            "source_facts_path":
                SOURCE_FACTS_PATH.as_posix(),
            "source_facts_sha256":
                EXPECTED_SOURCE_FACTS_SHA256,
            "rows_path":
                ROWS_PATH.as_posix(),
            "rows_sha256":
                EXPECTED_ROWS_SHA256,
            "structural_manifest_path":
                STRUCTURAL_MANIFEST_PATH.as_posix(),
            "structural_manifest_sha256":
                EXPECTED_STRUCTURAL_MANIFEST_SHA256,
            "row_count": EXPECTED_ROW_COUNT,
            "source_pair_count": EXPECTED_PAIR_COUNT,
            "pair_id_first": PAIR_ID_FIRST,
            "pair_id_last": PAIR_ID_LAST,
        },
        "tokenizer": tokenizer_provenance,
        "coordinate_contract": {
            "active_serialization":
                "claim[:63]+EOS(0)+evidence[:64]",
            "anchor_mapping":
                "xg1_generator_declared_span_final_overlapping_"
                "evidence_token_mapped_independently_per_cell",
            "post4_rule":
                "a+4 <= terminal_index-1",
            "shortened_post_window_allowed": False,
        },
        "required_anchor_row_count":
            EXPECTED_ANCHOR_ROWS,
        "required_anchor_counts":
            legacy.ANCHOR_EXPECTED_COUNTS,
        "eligible_anchor_counts": {
            anchor: int(
                eligible_anchor_counts.get(anchor, 0)
            )
            for anchor in legacy.ANCHOR_EXPECTED_COUNTS
        },
        "target_identity_name_cells":
            list(legacy.TARGET_IDENTITY_NAME_CELLS),
        "target_identity_name_mismatch_count":
            len(mismatches),
        "target_identity_name_mismatches":
            mismatches,
        "exclusion_counts":
            dict(sorted(exclusion_counts.items())),
        "claim_truncation_row_count":
            claim_truncation_count,
        "evidence_truncation_row_count":
            evidence_truncation_count,
        "complete_source_pair_count":
            complete_pair_count,
        "complete_source_pair_target":
            EXPECTED_PAIR_COUNT,
        "primary_complete_pair_prefix_feasibility":
            verdict,
        "anchor_manifest_sha256":
            sha256_bytes(manifest_bytes),
        "claim_boundary": [
            "This artifact establishes fresh XG1 301..600 tokenizer/event-anchor prefix eligibility only.",
            "The tokenizer and coordinate contract are inherited unchanged from the completed XG1 001..300 eligibility gate.",
            "No model forward, checkpoint load, native-state extraction, susceptibility endpoint computation, or scientific inference is performed.",
            "Eligibility PASS does not itself authorize scientific model execution.",
        ],
    }

    return anchor_rows, summary


def write_outputs(
    *,
    anchor_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_jsonl: Path,
    summary_path: Path,
) -> None:
    require(
        not output_jsonl.exists(),
        f"OUTPUT_COLLISION:{output_jsonl}",
    )
    require(
        not summary_path.exists(),
        f"OUTPUT_COLLISION:{summary_path}",
    )

    output_jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_bytes = serialize_anchor_manifest(
        anchor_rows
    )
    output_jsonl.write_bytes(manifest_bytes)

    require(
        sha256_file(output_jsonl)
        == summary["anchor_manifest_sha256"],
        "WRITTEN_ANCHOR_MANIFEST_SHA256_MISMATCH",
    )

    summary_path.write_text(
        json.dumps(
            dict(summary),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh XG1 301..600 tokenizer/event-anchor "
            "eligibility gate. No model execution."
        )
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    anchor_rows, summary = compute_eligibility(
        args.tokenizer_snapshot
    )

    write_outputs(
        anchor_rows=anchor_rows,
        summary=summary,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
    )

    print(
        "RESULT=",
        summary[
            "primary_complete_pair_prefix_feasibility"
        ],
        sep="",
    )
    print(
        "SOURCE_PAIR_COUNT=",
        summary["complete_source_pair_count"],
        sep="",
    )
    print(
        "PAIR_ID_FIRST=",
        summary["frozen_input"]["pair_id_first"],
        sep="",
    )
    print(
        "PAIR_ID_LAST=",
        summary["frozen_input"]["pair_id_last"],
        sep="",
    )
    print(
        "ANCHOR_ROW_COUNT=",
        summary["required_anchor_row_count"],
        sep="",
    )
    print(
        "ELIGIBLE_ANCHOR_COUNTS=",
        summary["eligible_anchor_counts"],
        sep="",
    )
    print(
        "EXCLUSION_COUNTS=",
        summary["exclusion_counts"],
        sep="",
    )
    print(
        "CLAIM_TRUNCATION_ROW_COUNT=",
        summary["claim_truncation_row_count"],
        sep="",
    )
    print(
        "EVIDENCE_TRUNCATION_ROW_COUNT=",
        summary["evidence_truncation_row_count"],
        sep="",
    )
    print(
        "ANCHOR_MANIFEST_SHA256=",
        summary["anchor_manifest_sha256"],
        sep="",
    )
    print("MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")

    return (
        0
        if summary[
            "primary_complete_pair_prefix_feasibility"
        ]
        == "PASS_300_OF_300"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
