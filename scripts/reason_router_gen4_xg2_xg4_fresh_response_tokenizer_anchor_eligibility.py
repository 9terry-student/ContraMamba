from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg2_xg4_fresh_response_holdouts as holdouts
from scripts import reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility as inherited


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

STRUCTURAL_FREEZE_COMMIT = (
    "4bda8dd4b46d56ffe9bb37cffc91f8506026ca69"
)
DESIGN_FREEZE_COMMIT = (
    "2074f52d39bf0fca6d63248016ce47c54bff5e06"
)

FRESH_BUILDER_PATH = (
    "scripts/build_reason_router_gen4_xg2_xg4_fresh_response_holdouts.py"
)
FRESH_BUILDER_BLOB = (
    "911846cf0b3304caaab0399535bd9f3de7f8249e"
)

INHERITED_GATE_PATH = (
    "scripts/reason_router_gen4_generator_family_prevalence_"
    "tokenizer_anchor_eligibility.py"
)
INHERITED_GATE_BLOB = (
    "29a97f343f372718ca56436c505893136cb505b6"
)

XG1_GATE_PATH = (
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py"
)
XG1_GATE_BLOB = (
    "6c98ce022ca134e385db28851fd364dc6daff423"
)

COHORT_ROOT = Path(
    "data/reason_router_gen4_xg2_xg4_fresh_response_holdouts_v1"
)

FROZEN = {
    "xg2": {
        "source_sha256":
            "c02d2fea5a7f3c8b5243598ad5505bba3f276c099be7b8c428c06eb39ffc7141",
        "rows_sha256":
            "1ca4f1c79caf5719e8970bf889c75b0e59f8711e5c78a36f7578727020e23670",
        "manifest_sha256":
            "7df94b0788a55c6a4927926f6b92ef7e90d66346c5342608f5716ddd517e47f9",
    },
    "xg4": {
        "source_sha256":
            "0666c9345505f993bce70d66b5b1f9b784edca782eecb13a750ecf827f14ba3a",
        "rows_sha256":
            "b407613cdcee15d193847130f64e6aa674e666f72d6d08a30b573005e5e9de7d",
        "manifest_sha256":
            "ee8d9bc4ae0ca135c335cf9eaa6bd9025cd5354304d435f1e3bb703a8b11e80d",
    },
}

FAMILY_KEYS = ("xg2", "xg4")

EXPECTED_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = 1800
EXPECTED_ANCHOR_ROWS = 1800

ANCHOR_EXPECTED_COUNTS = inherited.ANCHOR_EXPECTED_COUNTS
TARGET_IDENTITY_NAME_CELLS = inherited.TARGET_IDENTITY_NAME_CELLS

ANCHOR_SCHEMA = (
    "GEN4_XG2_XG4_FRESH_RESPONSE_TOKENIZER_ANCHOR_ELIGIBILITY_V1"
)
SUMMARY_SCHEMA = (
    "GEN4_XG2_XG4_FRESH_RESPONSE_TOKENIZER_ANCHOR_"
    "ELIGIBILITY_SUMMARY_V1"
)

EligibilityError = inherited.EligibilityError
require = inherited.require


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EligibilityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate(root: Path = ROOT) -> dict[str, str]:
    root = root.resolve()

    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(
        branch == EXPECTED_BRANCH,
        f"BRANCH_MISMATCH:{branch}",
    )

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            STRUCTURAL_FREEZE_COMMIT,
            head,
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "STRUCTURAL_FREEZE_NOT_ANCESTOR",
    )

    pinned = (
        (
            FRESH_BUILDER_PATH,
            FRESH_BUILDER_BLOB,
            "FRESH_BUILDER",
        ),
        (
            INHERITED_GATE_PATH,
            INHERITED_GATE_BLOB,
            "INHERITED_GATE",
        ),
        (
            XG1_GATE_PATH,
            XG1_GATE_BLOB,
            "XG1_GATE",
        ),
    )

    observed: dict[str, str] = {}

    for path, expected, label in pinned:
        blob = git(
            root,
            "rev-parse",
            f"HEAD:{path}",
        )
        require(
            blob == expected,
            f"{label}_BLOB_DRIFT:{blob}",
        )

        for diff_args, state in (
            (
                ("diff", "--quiet", "--", path),
                "WORKTREE",
            ),
            (
                (
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    path,
                ),
                "INDEX",
            ),
        ):
            rc = subprocess.call(
                ["git", *diff_args],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            require(
                rc == 0,
                f"{label}_{state}_DRIFT",
            )

        observed[
            label.lower() + "_blob"
        ] = blob

    return {
        "branch": branch,
        "head": head,
        "structural_freeze_commit":
            STRUCTURAL_FREEZE_COMMIT,
        "design_freeze_commit":
            DESIGN_FREEZE_COMMIT,
        **observed,
    }


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for number, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)

        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{number}",
        )
        output.append(value)

    return output


def load_family(
    family: str,
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    require(
        family in FAMILY_KEYS,
        f"UNKNOWN_FAMILY:{family}",
    )

    frozen = FROZEN[family]
    base = root / COHORT_ROOT / family

    source_path = (
        base / holdouts.SOURCE_FILE
    )
    rows_path = (
        base / holdouts.ROW_FILE
    )
    manifest_path = (
        base / "structural_manifest.json"
    )

    for path in (
        source_path,
        rows_path,
        manifest_path,
    ):
        require(
            path.is_file(),
            f"MISSING_FROZEN_INPUT:{path}",
        )

    require(
        inherited.xg1.sha256_file(
            source_path
        )
        == frozen["source_sha256"],
        f"SOURCE_SHA256:{family}",
    )
    require(
        inherited.xg1.sha256_file(
            rows_path
        )
        == frozen["rows_sha256"],
        f"ROWS_SHA256:{family}",
    )
    require(
        inherited.xg1.sha256_file(
            manifest_path
        )
        == frozen["manifest_sha256"],
        f"MANIFEST_SHA256:{family}",
    )

    facts = _load_jsonl(source_path)
    rows = _load_jsonl(rows_path)

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )

    spec = holdouts.family_spec(family)

    holdouts.validate_source_facts(
        spec,
        facts,
    )
    holdouts.validate_materialized_rows(
        spec,
        rows,
    )

    require(
        len(facts) == EXPECTED_PAIR_COUNT,
        f"SOURCE_COUNT:{family}",
    )
    require(
        len(rows) == EXPECTED_ROW_COUNT,
        f"ROW_COUNT:{family}",
    )

    expected_ids = (
        holdouts.expected_pair_ids(family)
    )

    require(
        [
            str(fact["pair_id"])
            for fact in facts
        ]
        == expected_ids,
        f"PAIR_IDS:{family}",
    )

    require(
        manifest.get("schema_version")
        == (
            "GEN4_XG2_XG4_FRESH_RESPONSE_"
            "HOLDOUT_STRUCTURAL_MANIFEST_V1"
        ),
        f"MANIFEST_SCHEMA:{family}",
    )
    require(
        manifest.get("result")
        == (
            "PASS_FRESH_RESPONSE_HOLDOUT_"
            "STRUCTURAL_GATE"
        ),
        f"MANIFEST_RESULT:{family}",
    )
    require(
        manifest.get("family_key")
        == family,
        f"MANIFEST_FAMILY:{family}",
    )
    require(
        manifest.get("pair_id_first")
        == f"{family}_fact_301",
        f"MANIFEST_FIRST_PAIR:{family}",
    )
    require(
        manifest.get("pair_id_last")
        == f"{family}_fact_600",
        f"MANIFEST_LAST_PAIR:{family}",
    )
    require(
        manifest.get(
            "source_file_sha256"
        )
        == frozen["source_sha256"],
        f"MANIFEST_SOURCE_SHA:{family}",
    )
    require(
        manifest.get("row_file_sha256")
        == frozen["rows_sha256"],
        f"MANIFEST_ROWS_SHA:{family}",
    )
    require(
        manifest.get(
            "absolute_schedule_indexing_301_600"
        )
        is True,
        f"MANIFEST_ABSOLUTE_INDEX:{family}",
    )
    require(
        manifest.get("tokenizer_executed")
        is False,
        f"MANIFEST_TOKENIZER_BOUNDARY:{family}",
    )
    require(
        manifest.get("model_executed")
        is False,
        f"MANIFEST_MODEL_BOUNDARY:{family}",
    )
    require(
        manifest.get("cuda_executed")
        is False,
        f"MANIFEST_CUDA_BOUNDARY:{family}",
    )
    require(
        manifest.get("response_fields_present")
        is False,
        f"MANIFEST_RESPONSE_BOUNDARY:{family}",
    )

    return facts, rows, manifest


def _base_summary(
    family: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    frozen = FROZEN[family]
    base = COHORT_ROOT / family

    return {
        "schema_version": SUMMARY_SCHEMA,
        "phase": (
            "XG2_XG4_FRESH_RESPONSE_"
            "TOKENIZER_ANCHOR_ELIGIBILITY"
        ),
        "family_key": family,
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
        "provenance": dict(provenance),
        "frozen_input": {
            "source_facts_path": (
                base
                / holdouts.SOURCE_FILE
            ).as_posix(),
            "source_facts_sha256":
                frozen["source_sha256"],
            "rows_path": (
                base
                / holdouts.ROW_FILE
            ).as_posix(),
            "rows_sha256":
                frozen["rows_sha256"],
            "structural_manifest_path": (
                base
                / "structural_manifest.json"
            ).as_posix(),
            "structural_manifest_sha256":
                frozen["manifest_sha256"],
            "source_pair_count":
                EXPECTED_PAIR_COUNT,
            "row_count":
                EXPECTED_ROW_COUNT,
            "pair_id_first":
                f"{family}_fact_301",
            "pair_id_last":
                f"{family}_fact_600",
        },
    }


def compute_eligibility(
    *,
    family: str,
    root: Path = ROOT,
    tokenizer_snapshot:
        str | Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    provenance = authenticate(root)

    facts, rows, _manifest = (
        load_family(
            family,
            root,
        )
    )

    failures = (
        inherited.topology_failures(
            family,
            facts,
        )
    )

    summary = _base_summary(
        family,
        provenance,
    )

    if failures:
        empty = (
            inherited.xg1
            .serialize_anchor_manifest([])
        )

        summary.update({
            "tokenizer_executed":
                False,
            "structural_anchor_topology_failure_count":
                len(failures),
            "structural_anchor_topology_failures":
                failures,
            "required_anchor_row_count":
                EXPECTED_ANCHOR_ROWS,
            "required_anchor_counts":
                ANCHOR_EXPECTED_COUNTS,
            "eligible_anchor_counts": {
                "A_IDENTITY": 0,
                "A_NAME": 0,
            },
            "target_identity_name_mismatch_count":
                0,
            "target_identity_name_mismatches":
                [],
            "exclusion_counts": {},
            "complete_source_pair_count":
                0,
            "complete_source_pair_target":
                EXPECTED_PAIR_COUNT,
            "primary_complete_pair_prefix_feasibility":
                "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY",
            "anchor_manifest_sha256":
                inherited.xg1.sha256_bytes(
                    empty
                ),
        })

        return [], summary

    tokenizer, tokenizer_provenance = (
        inherited.xg1
        .load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }

    require(
        len(facts_by_id)
        == EXPECTED_PAIR_COUNT,
        "FACT_ID_COUNT",
    )

    anchor_rows: list[
        dict[str, Any]
    ] = []

    for row in rows:
        pair_id = str(
            row["source_pair_id"]
        )

        require(
            pair_id in facts_by_id,
            f"MISSING_FACT:{pair_id}",
        )

        events = inherited.analyze_row(
            family,
            row,
            facts_by_id[pair_id],
            tokenizer,
        )

        for event in events:
            event = dict(event)
            event["schema_version"] = (
                ANCHOR_SCHEMA
            )
            anchor_rows.append(event)

    require(
        len(anchor_rows)
        == EXPECTED_ANCHOR_ROWS,
        "ANCHOR_ROW_COUNT",
    )

    counts = Counter(
        row["anchor_name"]
        for row in anchor_rows
    )

    require(
        dict(counts)
        == ANCHOR_EXPECTED_COUNTS,
        f"ANCHOR_COUNTS:{dict(counts)}",
    )

    eligible_counts = Counter(
        row["anchor_name"]
        for row in anchor_rows
        if row["post4_eligible"]
    )

    exclusions = Counter(
        row["exclusion_code"]
        for row in anchor_rows
        if row["exclusion_code"]
        is not None
    )

    pair_ok = {
        pair_id: True
        for pair_id in facts_by_id
    }

    for row in anchor_rows:
        if not row["post4_eligible"]:
            pair_ok[
                str(
                    row[
                        "source_pair_id"
                    ]
                )
            ] = False

    lookup = {
        (
            str(
                row[
                    "source_pair_id"
                ]
            ),
            str(
                row[
                    "contrast_cell_id"
                ]
            ),
            str(
                row["anchor_name"]
            ),
        ): row
        for row in anchor_rows
    }

    require(
        len(lookup)
        == EXPECTED_ANCHOR_ROWS,
        "DUPLICATE_ANCHOR_EVENT_KEY",
    )

    mismatches: list[
        dict[str, Any]
    ] = []

    for pair_id in facts_by_id:
        for cell_id in (
            TARGET_IDENTITY_NAME_CELLS
        ):
            identity = lookup[
                (
                    pair_id,
                    cell_id,
                    "A_IDENTITY",
                )
            ]
            name = lookup[
                (
                    pair_id,
                    cell_id,
                    "A_NAME",
                )
            ]

            if (
                identity[
                    "absolute_anchor_token_index"
                ]
                != name[
                    "absolute_anchor_token_index"
                ]
            ):
                pair_ok[
                    pair_id
                ] = False

                mismatches.append({
                    "source_pair_id":
                        pair_id,
                    "contrast_cell_id":
                        cell_id,
                    "identity_absolute_anchor_token_index":
                        identity[
                            "absolute_anchor_token_index"
                        ],
                    "name_absolute_anchor_token_index":
                        name[
                            "absolute_anchor_token_index"
                        ],
                })

    complete = sum(
        pair_ok.values()
    )

    all_eligible = (
        complete
        == EXPECTED_PAIR_COUNT
        and sum(
            eligible_counts.values()
        )
        == EXPECTED_ANCHOR_ROWS
        and not mismatches
    )

    verdict = (
        "PASS_300_OF_300"
        if all_eligible
        else
        "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"
    )

    manifest_bytes = (
        inherited.xg1
        .serialize_anchor_manifest(
            anchor_rows
        )
    )

    summary.update({
        "tokenizer_executed":
            True,
        "tokenizer":
            tokenizer_provenance,
        "structural_anchor_topology_failure_count":
            0,
        "structural_anchor_topology_failures":
            [],
        "required_anchor_row_count":
            EXPECTED_ANCHOR_ROWS,
        "required_anchor_counts":
            ANCHOR_EXPECTED_COUNTS,
        "eligible_anchor_counts": {
            key: int(
                eligible_counts.get(
                    key,
                    0,
                )
            )
            for key
            in ANCHOR_EXPECTED_COUNTS
        },
        "target_identity_name_cells":
            list(
                TARGET_IDENTITY_NAME_CELLS
            ),
        "target_identity_name_mismatch_count":
            len(mismatches),
        "target_identity_name_mismatches":
            mismatches,
        "exclusion_counts":
            dict(
                sorted(
                    exclusions.items()
                )
            ),
        "complete_source_pair_count":
            complete,
        "complete_source_pair_target":
            EXPECTED_PAIR_COUNT,
        "primary_complete_pair_prefix_feasibility":
            verdict,
        "anchor_manifest_sha256":
            inherited.xg1
            .sha256_bytes(
                manifest_bytes
            ),
        "coordinate_contract": {
            "active_serialization":
                "claim[:63]+EOS(0)+evidence[:64]",
            "anchor_mapping":
                (
                    "frozen_contiguous_title_name_span_"
                    "final_overlapping_evidence_token"
                ),
            "post4_rule":
                "a+4 <= terminal_index-1",
            "shortened_post_window_allowed":
                False,
        },
        "claim_boundary": [
            (
                "This artifact establishes fresh XG2/XG4 "
                "tokenizer/event-anchor prefix eligibility only."
            ),
            (
                "The canonical tokenizer is the frozen "
                "active-encoding analysis reference."
            ),
            (
                "No model forward, checkpoint load, CUDA "
                "execution, intervention response, baseline "
                "geometry, R_ALIGN, or inferential test is performed."
            ),
            (
                "Eligibility PASS does not itself authorize "
                "scientific model execution."
            ),
        ],
    })

    return anchor_rows, summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        choices=FAMILY_KEYS,
        required=True,
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
        default=None,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    anchor_rows, summary = (
        compute_eligibility(
            family=args.family,
            tokenizer_snapshot=(
                args.tokenizer_snapshot
            ),
        )
    )

    inherited.xg1.write_outputs(
        anchor_rows=anchor_rows,
        summary=summary,
        output_jsonl=(
            args.output_jsonl
        ),
        summary_path=args.summary,
    )

    print(
        "FAMILY =",
        args.family.upper(),
    )
    print(
        "RESULT =",
        summary[
            "primary_complete_pair_prefix_feasibility"
        ],
    )
    print(
        "TOKENIZER_EXECUTED =",
        summary[
            "tokenizer_executed"
        ],
    )
    print(
        "STRUCTURAL_ANCHOR_TOPOLOGY_FAILURE_COUNT =",
        summary[
            "structural_anchor_topology_failure_count"
        ],
    )
    print(
        "COMPLETE_SOURCE_PAIR_COUNT =",
        summary[
            "complete_source_pair_count"
        ],
    )
    print(
        "MODEL_FORWARD_COUNT = 0"
    )
    print(
        "GPU_USED = FALSE"
    )

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