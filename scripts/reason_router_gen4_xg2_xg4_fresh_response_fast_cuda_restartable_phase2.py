from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

PHASE1_ARTIFACT_FREEZE_COMMIT = (
    "f4f5aef2025a66e8b7e7ed6d077523eec46eb6f0"
)
IMPLEMENTATION_SCOPE_COMMIT = (
    "7d43d2a9a180c976a23884a0005e83c6a82390c9"
)

PHASE1_ARTIFACT_ROOT = Path(
    "reports/"
    "reason_router_gen4_xg2_xg4_fresh_response_"
    "restartable_phase1_d217bad_r3"
)
PHASE1_RUNNER_REL = (
    "scripts/"
    "reason_router_gen4_xg2_xg4_fresh_response_"
    "fast_cuda_restartable_phase1.py"
)
PHASE1_RUNNER_BLOB = (
    "62df03e9c48014650dee69ae007b91d87d5cb7ac"
)
IMPLEMENTATION_SCOPE_REL = (
    "reports/"
    "reason_router_gen4_xg2_xg4_fresh_response_"
    "restartable_phase2_implementation_spec_candidate.md"
)
IMPLEMENTATION_SCOPE_BLOB = (
    "46e1856a70d5abd0b870b2c328b2a5e52268e716"
)

SOURCE_PAIR_COUNT = phase1.SOURCE_PAIR_COUNT
ALIGNMENT_FORWARDS_PER_PAIR = 2
PHASE2_ALIGNMENT_FORWARD_BUDGET = (
    SOURCE_PAIR_COUNT * ALIGNMENT_FORWARDS_PER_PAIR
)
PERSISTED_PHASE1_BASELINE_FORWARD_COUNT = (
    phase1.FULL_BASELINE_FORWARD_BUDGET
)
COMPLETE_PROTOCOL_FORWARD_COUNT = (
    PERSISTED_PHASE1_BASELINE_FORWARD_COUNT
    + PHASE2_ALIGNMENT_FORWARD_BUDGET
)

ALIGNMENT_SHIFT_THRESHOLD = phase1.ALIGNMENT_SHIFT_THRESHOLD
REGIME_LARGE = phase1.REGIME_LARGE
REGIME_SMALL = phase1.REGIME_SMALL
FROZEN_REGIME_COUNTS = phase1.FROZEN_REGIME_COUNTS

ITEM_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase2-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase2-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-"
    "restartable-phase2-manifest-v1"
)
RESULT_PASS = (
    "PASS_XG2_XG4_FRESH_RESPONSE_"
    "RESTARTABLE_PHASE2_FREEZE"
)

ITEM_FILE = "response_items.jsonl"
SUMMARY_FILE = "phase2_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESPONSE_FIELDS = (
    "alignment_plus_path_efficiency",
    "alignment_minus_path_efficiency",
    "delta_alignment",
    "R_ALIGN",
)

FORBIDDEN_ANALYSIS_FIELDS = frozenset(
    {
        "h1_p_value",
        "h2_p_value",
        "h1_pass",
        "h2_pass",
        "replication_conclusion",
    }
)


class RestartablePhase2Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RestartablePhase2Error(message)


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
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(dict(row))
        for row in rows
    )


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise RestartablePhase2Error(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(
    expected_head: str,
) -> None:
    phase1.authenticate_repo(expected_head)

    branch = git("branch", "--show-current")
    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )

    for ancestor, label in (
        (
            PHASE1_ARTIFACT_FREEZE_COMMIT,
            "PHASE1_ARTIFACT_FREEZE",
        ),
        (
            IMPLEMENTATION_SCOPE_COMMIT,
            "IMPLEMENTATION_SCOPE",
        ),
    ):
        require(
            _git_is_ancestor(
                ancestor,
                expected_head,
            ),
            f"{label}_NOT_ANCESTOR",
        )

    observed_phase1_blob = git(
        "rev-parse",
        f"HEAD:{PHASE1_RUNNER_REL}",
    )
    require(
        observed_phase1_blob == PHASE1_RUNNER_BLOB,
        (
            "FROZEN_PHASE1_RUNNER_DRIFT:"
            f"{observed_phase1_blob}"
        ),
    )

    observed_scope_blob = git(
        "rev-parse",
        f"HEAD:{IMPLEMENTATION_SCOPE_REL}",
    )
    require(
        observed_scope_blob
        == IMPLEMENTATION_SCOPE_BLOB,
        (
            "IMPLEMENTATION_SCOPE_DRIFT:"
            f"{observed_scope_blob}"
        ),
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            PHASE1_ARTIFACT_FREEZE_COMMIT,
            expected_head,
            "--",
            PHASE1_ARTIFACT_ROOT.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "FROZEN_PHASE1_ARTIFACT_TREE_DRIFT",
    )


def _phase1_dir(
    family: str,
) -> Path:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    return ROOT / PHASE1_ARTIFACT_ROOT / family


def _expected_pairs(
    family: str,
) -> tuple[str, ...]:
    return phase1._expected_pairs(family)


def _validate_phase1_loaded(
    family: str,
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    summary = loaded["summary"]
    items = loaded["items"]
    plans = loaded["alignment_delta_h"]

    require(
        len(items) == SOURCE_PAIR_COUNT,
        "PHASE1_ITEM_COUNT",
    )
    require(
        torch.is_tensor(plans),
        "PHASE1_PLAN_NOT_TENSOR",
    )
    require(
        int(plans.shape[0])
        == SOURCE_PAIR_COUNT,
        "PHASE1_PLAN_COUNT",
    )

    require(
        summary["family_key"] == family,
        "PHASE1_SUMMARY_FAMILY",
    )
    require(
        summary["source_pair_count"]
        == SOURCE_PAIR_COUNT,
        "PHASE1_SUMMARY_PAIR_COUNT",
    )
    require(
        summary["support_gate_pass"] is True,
        "PHASE1_SUPPORT_GATE",
    )
    require(
        summary["phase2_restartable"] is True,
        "PHASE1_NOT_RESTARTABLE",
    )
    require(
        summary["baseline_model_forward_count"]
        == PERSISTED_PHASE1_BASELINE_FORWARD_COUNT,
        "PHASE1_BASELINE_FORWARD_COUNT",
    )
    require(
        summary["alignment_model_forward_count"]
        == 0,
        "PHASE1_ALIGNMENT_FORWARD_COUNT",
    )
    require(
        float(summary["threshold"])
        == ALIGNMENT_SHIFT_THRESHOLD,
        "PHASE1_THRESHOLD",
    )

    expected_counts = (
        FROZEN_REGIME_COUNTS[family]
    )
    require(
        summary["n_LARGE"]
        == expected_counts[REGIME_LARGE],
        "PHASE1_LARGE_COUNT",
    )
    require(
        summary["n_SMALL"]
        == expected_counts[REGIME_SMALL],
        "PHASE1_SMALL_COUNT",
    )

    observed_pairs = tuple(
        str(row["source_pair_id"])
        for row in items
    )
    require(
        observed_pairs
        == _expected_pairs(family),
        "PHASE1_PAIR_ORDER",
    )

    for index, row in enumerate(items):
        require(
            row["alignment_plan_index"]
            == index,
            f"PHASE1_PLAN_INDEX:{index}",
        )

        plus = float(
            row[
                "baseline_plus_path_efficiency"
            ]
        )
        minus = float(
            row[
                "baseline_minus_path_efficiency"
            ]
        )
        delta = float(
            row["delta_baseline"]
        )
        require(
            all(
                math.isfinite(value)
                for value in (
                    plus,
                    minus,
                    delta,
                )
            ),
            f"PHASE1_NONFINITE_BASELINE:{index}",
        )
        require(
            delta == plus - minus,
            f"PHASE1_DELTA_BASELINE:{index}",
        )

        plan = plans[index]
        require(
            (
                str(plan.dtype)
                .removeprefix("torch.")
            )
            == row[
                "alignment_delta_h_dtype"
            ],
            f"PHASE1_PLAN_DTYPE:{index}",
        )
        require(
            list(plan.shape)
            == row[
                "alignment_delta_h_shape"
            ],
            f"PHASE1_PLAN_SHAPE:{index}",
        )
        require(
            bool(
                torch.isfinite(plan)
                .all()
                .item()
            ),
            f"PHASE1_PLAN_NONFINITE:{index}",
        )

    counts = Counter(
        str(row["regime"])
        for row in items
    )
    require(
        counts[REGIME_LARGE]
        == expected_counts[REGIME_LARGE]
        and counts[REGIME_SMALL]
        == expected_counts[REGIME_SMALL],
        "PHASE1_REGIME_COUNTS",
    )

    return {
        "summary": summary,
        "items": items,
        "alignment_delta_h": plans,
        "manifest": loaded["manifest"],
    }


def load_phase1_artifact(
    family: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    directory = (
        artifact_dir
        if artifact_dir is not None
        else _phase1_dir(family)
    )

    loaded = (
        phase1.validate_restartable_artifact(
            directory,
            family,
        )
    )
    validated = _validate_phase1_loaded(
        family,
        loaded,
    )

    validated["artifact_dir"] = directory
    validated[
        "artifact_manifest_sha256"
    ] = sha256_file(
        directory / phase1.MANIFEST_FILE
    )
    validated[
        "artifact_checksum_sha256"
    ] = sha256_file(
        directory / phase1.CHECKSUM_FILE
    )
    return validated


def _input_row(
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    return (
        phase1.base.prevalence_eq
        ._input_row(
            encoded,
            row_index,
            pair,
            cell,
        )
    )


def _run_alignment_pair(
    family: str,
    baseline_item: Mapping[str, Any],
    alignment_delta_h: torch.Tensor,
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[tuple[str, str], int],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(
        baseline_item["family_key"]
        == family,
        "BASELINE_ITEM_FAMILY",
    )

    pair = str(
        baseline_item["source_pair_id"]
    )
    core = phase1.base.prevalence_eq.core
    parent = phase1.base.prevalence_eq.parent
    runtime = (
        phase1.base.prevalence_eq
        .transport_runtime
    )

    cells = phase1._cells()
    anchors = phase1._anchors_for_pair(
        pair,
        events,
    )

    for role, field in (
        ("tp", "target_plus_anchor"),
        ("tm", "target_minus_anchor"),
        ("rp", "reference_plus_anchor"),
        ("rm", "reference_minus_anchor"),
    ):
        require(
            anchors[role]
            == int(baseline_item[field]),
            f"FROZEN_ANCHOR_IDENTITY:{pair}:{role}",
        )

    plan = (
        alignment_delta_h.detach()
        .cpu()
        .contiguous()
        .clone()
    )
    require(
        bool(torch.isfinite(plan).all().item()),
        f"NONFINITE_PLAN:{pair}",
    )
    require(
        (
            str(plan.dtype)
            .removeprefix("torch.")
        )
        == baseline_item[
            "alignment_delta_h_dtype"
        ],
        f"PLAN_DTYPE:{pair}",
    )
    require(
        list(plan.shape)
        == baseline_item[
            "alignment_delta_h_shape"
        ],
        f"PLAN_SHAPE:{pair}",
    )

    alignment: dict[str, Any] = {}

    for role, plus_branch in (
        ("tp", True),
        ("tm", False),
    ):
        alignment[role] = (
            parent.capture_branch(
                model,
                runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                input_ids=_input_row(
                    encoded,
                    row_index,
                    pair,
                    cells[role],
                ),
                anchor=anchors[role],
                budget=budget,
                capture_states=True,
                delta_h=plan,
                plus_branch=plus_branch,
            )
        )

    paired_audit = (
        runtime.paired_intervention_audit(
            alignment["tp"][
                "intervention_audit"
            ],
            alignment["tm"][
                "intervention_audit"
            ],
            plan,
            plus_expected_token_index=(
                anchors["tp"]
                + core.TARGET_OFFSET
            ),
            minus_expected_token_index=(
                anchors["tm"]
                + core.TARGET_OFFSET
            ),
        )
    )

    alignment_plus = float(
        parent.path_efficiency(
            alignment["tp"]
        )
    )
    alignment_minus = float(
        parent.path_efficiency(
            alignment["tm"]
        )
    )
    delta_alignment = (
        alignment_plus
        - alignment_minus
    )
    delta_baseline = float(
        baseline_item["delta_baseline"]
    )
    r_align = (
        delta_alignment
        - delta_baseline
    )

    require(
        all(
            math.isfinite(value)
            for value in (
                alignment_plus,
                alignment_minus,
                delta_alignment,
                delta_baseline,
                r_align,
            )
        ),
        f"NONFINITE_RESPONSE:{pair}",
    )

    item = dict(baseline_item)
    item["phase1_schema_version"] = (
        item["schema_version"]
    )
    item["schema_version"] = ITEM_SCHEMA
    item[
        "phase1_artifact_freeze_commit"
    ] = PHASE1_ARTIFACT_FREEZE_COMMIT
    item[
        "alignment_plus_path_efficiency"
    ] = alignment_plus
    item[
        "alignment_minus_path_efficiency"
    ] = alignment_minus
    item["delta_alignment"] = (
        delta_alignment
    )
    item["R_ALIGN"] = r_align
    item[
        "alignment_midpoint_max_abs_residual"
    ] = float(
        paired_audit[
            "midpoint_max_abs_residual"
        ]
    )
    item[
        "alignment_pair_delta_max_abs_residual"
    ] = float(
        paired_audit[
            "pair_delta_max_abs_residual"
        ]
    )
    item[
        "alignment_applied_correction_max_abs_residual"
    ] = float(
        paired_audit[
            "applied_correction_max_abs_residual"
        ]
    )
    item[
        "alignment_runtime_correction_l2"
    ] = float(
        paired_audit[
            "runtime_correction_l2"
        ]
    )

    require(
        item["delta_baseline"]
        == (
            item[
                "baseline_plus_path_efficiency"
            ]
            - item[
                "baseline_minus_path_efficiency"
            ]
        ),
        f"DELTA_BASELINE_IDENTITY:{pair}",
    )
    require(
        item["delta_alignment"]
        == (
            item[
                "alignment_plus_path_efficiency"
            ]
            - item[
                "alignment_minus_path_efficiency"
            ]
        ),
        f"DELTA_ALIGNMENT_IDENTITY:{pair}",
    )
    require(
        item["R_ALIGN"]
        == (
            item["delta_alignment"]
            - item["delta_baseline"]
        ),
        f"R_ALIGN_IDENTITY:{pair}",
    )

    return item


def _validate_response_rows(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "RESPONSE_ITEM_COUNT",
    )

    expected_pairs = _expected_pairs(family)
    expected_counts = (
        FROZEN_REGIME_COUNTS[family]
    )

    for index, (
        expected_pair,
        raw,
    ) in enumerate(
        zip(
            expected_pairs,
            items,
            strict=True,
        )
    ):
        row = dict(raw)

        require(
            row.get("schema_version")
            == ITEM_SCHEMA,
            f"ITEM_SCHEMA:{index}",
        )
        require(
            row.get("phase1_schema_version")
            == phase1.ITEM_SCHEMA,
            f"PHASE1_SCHEMA:{index}",
        )
        require(
            row.get("family_key")
            == family,
            f"ITEM_FAMILY:{index}",
        )
        require(
            row.get("source_pair_id")
            == expected_pair,
            f"PAIR_ORDER:{index}",
        )
        require(
            row.get(
                "alignment_plan_index"
            )
            == index,
            f"PLAN_INDEX:{index}",
        )
        require(
            float(row["threshold"])
            == ALIGNMENT_SHIFT_THRESHOLD,
            f"THRESHOLD:{index}",
        )
        require(
            not (
                FORBIDDEN_ANALYSIS_FIELDS
                & set(row)
            ),
            f"ANALYSIS_FIELD_LEAK:{index}",
        )

        values = [
            float(
                row[
                    "baseline_plus_path_efficiency"
                ]
            ),
            float(
                row[
                    "baseline_minus_path_efficiency"
                ]
            ),
            float(row["delta_baseline"]),
            float(
                row[
                    "alignment_plus_path_efficiency"
                ]
            ),
            float(
                row[
                    "alignment_minus_path_efficiency"
                ]
            ),
            float(row["delta_alignment"]),
            float(row["R_ALIGN"]),
            float(
                row[
                    "alignment_midpoint_max_abs_residual"
                ]
            ),
            float(
                row[
                    "alignment_pair_delta_max_abs_residual"
                ]
            ),
            float(
                row[
                    "alignment_applied_correction_max_abs_residual"
                ]
            ),
            float(
                row[
                    "alignment_runtime_correction_l2"
                ]
            ),
        ]
        require(
            all(
                math.isfinite(value)
                for value in values
            ),
            f"NONFINITE_ITEM:{index}",
        )

        require(
            row["delta_baseline"]
            == (
                row[
                    "baseline_plus_path_efficiency"
                ]
                - row[
                    "baseline_minus_path_efficiency"
                ]
            ),
            f"DELTA_BASELINE:{index}",
        )
        require(
            row["delta_alignment"]
            == (
                row[
                    "alignment_plus_path_efficiency"
                ]
                - row[
                    "alignment_minus_path_efficiency"
                ]
            ),
            f"DELTA_ALIGNMENT:{index}",
        )
        require(
            row["R_ALIGN"]
            == (
                row["delta_alignment"]
                - row["delta_baseline"]
            ),
            f"R_ALIGN:{index}",
        )

    counts = Counter(
        str(row["regime"])
        for row in items
    )
    require(
        counts[REGIME_LARGE]
        == expected_counts[REGIME_LARGE]
        and counts[REGIME_SMALL]
        == expected_counts[REGIME_SMALL],
        "FROZEN_REGIME_COUNTS",
    )


def _write_outputs(
    output_dir: Path,
    *,
    family: str,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    _validate_response_rows(
        family,
        items,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payloads = {
        ITEM_FILE: jsonl_bytes(items),
        SUMMARY_FILE: canonical_json_bytes(
            summary
        ),
    }

    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int(
                    (output_dir / name)
                    .stat()
                    .st_size
                ),
            }
            for name, digest in sorted(
                hashes.items()
            )
        },
    }

    manifest_raw = canonical_json_bytes(
        manifest
    )
    (
        output_dir / MANIFEST_FILE
    ).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = (
        sha256_bytes(manifest_raw)
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(
            hashes.items()
        )
    ).encode("utf-8")
    (
        output_dir / CHECKSUM_FILE
    ).write_bytes(checksum_raw)

    return hashes


def _load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        require(
            isinstance(row, dict),
            (
                "JSONL_OBJECT_REQUIRED:"
                f"{line_no}"
            ),
        )
        rows.append(row)
    return rows


def validate_phase2_artifact(
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    manifest_path = (
        output_dir / MANIFEST_FILE
    )
    checksum_path = (
        output_dir / CHECKSUM_FILE
    )

    require(
        manifest_path.is_file(),
        "MANIFEST_MISSING",
    )
    require(
        checksum_path.is_file(),
        "CHECKSUM_MISSING",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )
    require(
        manifest.get("schema_version")
        == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )

    files = manifest.get("files")
    require(
        isinstance(files, dict),
        "MANIFEST_FILES",
    )
    require(
        set(files)
        == {
            ITEM_FILE,
            SUMMARY_FILE,
        },
        "MANIFEST_FILE_SET",
    )

    observed_hashes: dict[str, str] = {}

    for name in (
        ITEM_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / name
        require(
            path.is_file(),
            f"ARTIFACT_MISSING:{name}",
        )
        observed_sha = sha256_file(path)
        observed_bytes = int(
            path.stat().st_size
        )
        expected = files[name]
        require(
            observed_sha
            == expected["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            observed_bytes
            == int(expected["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )
        observed_hashes[name] = (
            observed_sha
        )

    observed_hashes[MANIFEST_FILE] = (
        sha256_file(manifest_path)
    )

    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split(
            "  ",
            1,
        )
        require(
            name not in checksum_rows,
            f"CHECKSUM_DUPLICATE:{name}",
        )
        checksum_rows[name] = digest

    require(
        checksum_rows
        == {
            name: digest
            for name, digest in sorted(
                observed_hashes.items()
            )
        },
        "CHECKSUM_CONTENT",
    )

    items = _load_jsonl(
        output_dir / ITEM_FILE
    )
    _validate_response_rows(
        family,
        items,
    )

    summary = json.loads(
        (
            output_dir
            / SUMMARY_FILE
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("schema_version")
        == SUMMARY_SCHEMA,
        "SUMMARY_SCHEMA",
    )
    require(
        summary.get("result")
        == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary.get("family_key")
        == family,
        "SUMMARY_FAMILY",
    )
    require(
        summary.get("source_pair_count")
        == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        summary.get(
            "phase1_artifact_freeze_commit"
        )
        == PHASE1_ARTIFACT_FREEZE_COMMIT,
        "SUMMARY_PHASE1_FREEZE",
    )
    require(
        summary.get(
            "baseline_model_forward_count_this_run"
        )
        == 0,
        "SUMMARY_BASELINE_THIS_RUN",
    )
    require(
        summary.get(
            "alignment_model_forward_count_this_run"
        )
        == PHASE2_ALIGNMENT_FORWARD_BUDGET,
        "SUMMARY_ALIGNMENT_THIS_RUN",
    )
    require(
        summary.get(
            "current_run_scientific_forward_count"
        )
        == PHASE2_ALIGNMENT_FORWARD_BUDGET,
        "SUMMARY_CURRENT_RUN_BUDGET",
    )
    require(
        summary.get(
            "persisted_phase1_baseline_forward_count"
        )
        == PERSISTED_PHASE1_BASELINE_FORWARD_COUNT,
        "SUMMARY_PERSISTED_BASELINE",
    )
    require(
        summary.get(
            "complete_protocol_forward_count"
        )
        == COMPLETE_PROTOCOL_FORWARD_COUNT,
        "SUMMARY_COMPLETE_PROTOCOL",
    )
    require(
        summary.get(
            "response_fields_observed"
        )
        is True,
        "SUMMARY_RESPONSE_FIELDS",
    )
    require(
        summary.get("r_align_observed")
        is True,
        "SUMMARY_R_ALIGN",
    )
    require(
        summary.get("h1_test_executed")
        is False
        and summary.get(
            "h2_test_executed"
        )
        is False,
        "SUMMARY_H1_H2_BOUNDARY",
    )
    require(
        summary.get("training_executed")
        is False
        and summary.get(
            "backward_executed"
        )
        is False
        and summary.get(
            "task_heads_executed"
        )
        is False
        and summary.get("logits_read")
        is False,
        "SUMMARY_EXECUTION_BOUNDARY",
    )
    require(
        summary.get(
            "scientific_conclusion"
        )
        is None,
        "SUMMARY_CONCLUSION_BOUNDARY",
    )

    counts = Counter(
        str(row["regime"])
        for row in items
    )
    require(
        summary["n_LARGE"]
        == counts[REGIME_LARGE]
        and summary["n_SMALL"]
        == counts[REGIME_SMALL],
        "SUMMARY_REGIME_COUNTS",
    )

    return {
        "summary": summary,
        "items": items,
        "manifest": manifest,
    }


def run_restartable_phase2(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(
        family in FROZEN_REGIME_COUNTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    authenticate_repo(expected_head)

    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    # Phase-1 artifact validation is deliberately completed before
    # runtime/model construction or any scientific model forward.
    phase1_loaded = load_phase1_artifact(
        family
    )

    phase1_summary = (
        phase1_loaded["summary"]
    )
    phase1_items = phase1_loaded["items"]
    phase1_plans = (
        phase1_loaded["alignment_delta_h"]
    )

    phase1.base.prevalence_eq.backend.runtime_gate()

    with (
        phase1.base.prevalence_eq.backend
        .parent_runtime_rebind()
    ):
        rows, encoded, event_rows = (
            phase1.base.fresh_eq
            .load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )

        pairs = (
            phase1.base.fresh_eq
            ._pair_order(
                family,
                rows,
            )
        )
        require(
            tuple(pairs)
            == _expected_pairs(family),
            "PAIR_ORDER",
        )
        require(
            tuple(
                str(row["source_pair_id"])
                for row in phase1_items
            )
            == tuple(pairs),
            "PHASE1_INPUT_PAIR_ORDER",
        )

        parent = (
            phase1.base.prevalence_eq
            .parent
        )
        events = parent.event_lookup(
            event_rows
        )
        parent.validate_transport_event_plan(
            pairs,
            events,
        )
        row_index = parent.build_row_index(
            rows
        )
        trace_code, trace_line = (
            phase1.base.prevalence_eq
            .measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            phase1.base.prevalence_eq
            .kernel_compat
            .load_exact_fast_kernels()
        )

        with (
            phase1.base.prevalence_eq
            .kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                parent
                .load_representative_model_external(
                    model_snapshot=(
                        model_snapshot
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                )
            )

            require(
                checkpoint_sha
                == (
                    phase1.base.prevalence_eq
                    .extraction
                    .REPRESENTATIVE_CHECKPOINT_SHA256
                ),
                "CHECKPOINT_IDENTITY",
            )

            runtime_ctx = (
                phase1.base.prevalence_eq
                .transport_runtime
                .validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(
            constructor_kernel_calls
        )
        require(
            set(constructor_counts)
            == {
                "causal-conv1d",
                "mamba-ssm",
            },
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_NAMES:"
                f"{dict(constructor_counts)}"
            ),
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ] > 0
            and constructor_counts[
                "causal-conv1d"
            ]
            == constructor_counts[
                "mamba-ssm"
            ],
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_CALL_COUNT:"
                f"{dict(constructor_counts)}"
            ),
        )

        (
            phase1.base.prevalence_eq
            .kernel_compat
            .validate_transformers_kernel_bindings(
                kernels
            )
        )

        model.to(torch.device("cuda:0"))
        model.eval()

        require(
            all(
                parameter.device.type
                == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            phase1.base.prevalence_eq
            .backend
            ._make_fast_capture(
                kernels
            )
        )

        original_capture = (
            parent.capture_branch
        )
        budget = parent.ForwardBudget(
            PHASE2_ALIGNMENT_FORWARD_BUDGET
        )
        items: list[dict[str, Any]] = []

        parent.capture_branch = (
            fast_capture
        )
        try:
            for index, pair in enumerate(
                pairs
            ):
                require(
                    phase1_items[index][
                        "source_pair_id"
                    ]
                    == pair,
                    f"PAIR_IDENTITY:{index}",
                )

                item = _run_alignment_pair(
                    family,
                    phase1_items[index],
                    phase1_plans[index],
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                items.append(item)

            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            parent.capture_branch = (
                original_capture
            )

    counts = Counter(
        str(row["regime"])
        for row in items
    )
    expected_counts = (
        FROZEN_REGIME_COUNTS[family]
    )
    require(
        counts[REGIME_LARGE]
        == expected_counts[REGIME_LARGE]
        and counts[REGIME_SMALL]
        == expected_counts[REGIME_SMALL],
        "POSTRUN_REGIME_COUNTS",
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "implementation_scope_commit": (
            IMPLEMENTATION_SCOPE_COMMIT
        ),
        "phase1_artifact_freeze_commit": (
            PHASE1_ARTIFACT_FREEZE_COMMIT
        ),
        "phase1_artifact_path": (
            PHASE1_ARTIFACT_ROOT
            .joinpath(family)
            .as_posix()
        ),
        "phase1_artifact_manifest_sha256": (
            phase1_loaded[
                "artifact_manifest_sha256"
            ]
        ),
        "phase1_artifact_checksum_sha256": (
            phase1_loaded[
                "artifact_checksum_sha256"
            ]
        ),
        "phase1_result": (
            phase1_summary["result"]
        ),
        "source_pair_count": (
            SOURCE_PAIR_COUNT
        ),
        "pair_id_first": (
            items[0]["source_pair_id"]
        ),
        "pair_id_last": (
            items[-1]["source_pair_id"]
        ),
        "threshold": (
            ALIGNMENT_SHIFT_THRESHOLD
        ),
        "threshold_reestimated": False,
        "n_LARGE": (
            counts[REGIME_LARGE]
        ),
        "n_SMALL": (
            counts[REGIME_SMALL]
        ),
        "support_gate_pass": True,
        "phase2_restartable_input": True,
        "baseline_model_forward_count_this_run": 0,
        "alignment_model_forward_count_this_run": (
            PHASE2_ALIGNMENT_FORWARD_BUDGET
        ),
        "current_run_scientific_forward_count": (
            PHASE2_ALIGNMENT_FORWARD_BUDGET
        ),
        "persisted_phase1_baseline_forward_count": (
            PERSISTED_PHASE1_BASELINE_FORWARD_COUNT
        ),
        "complete_protocol_forward_count": (
            COMPLETE_PROTOCOL_FORWARD_COUNT
        ),
        "response_fields_observed": True,
        "r_align_observed": True,
        "alignment_intervention_executed": True,
        "magnitude_intervention_executed": False,
        "h1_test_executed": False,
        "h2_test_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
        "scientific_conclusion_scope": (
            "PHASE2_RESPONSE_OBSERVATION_ONLY"
        ),
        "representative_checkpoint_sha256": (
            checkpoint_sha
        ),
    }

    _write_outputs(
        output_dir,
        family=family,
        items=items,
        summary=summary,
    )

    validated = validate_phase2_artifact(
        output_dir,
        family,
    )
    require(
        validated["summary"]["result"]
        == RESULT_PASS,
        "POSTWRITE_VALIDATION_RESULT",
    )

    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restartable fresh XG2/XG4 Phase-2 response run. "
            "Consumes frozen Phase-1 baseline PE and lossless "
            "alignment_delta_h. Exactly 300 pairs x two alignment "
            "branches = 600 scientific forwards per family. "
            "No baseline re-execution, H1/H2, training, backward, "
            "task-head, or logit execution."
        )
    )
    parser.add_argument(
        "--family",
        choices=("xg2", "xg4"),
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    summary = run_restartable_phase2(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=(
            args.tokenizer_snapshot
        ),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT =", summary["result"])
    print(
        "FAMILY =",
        summary["family_key"],
    )
    print(
        "N_LARGE =",
        summary["n_LARGE"],
    )
    print(
        "N_SMALL =",
        summary["n_SMALL"],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary[
            "baseline_model_forward_count_this_run"
        ],
    )
    print(
        "ALIGNMENT_MODEL_FORWARD_COUNT_THIS_RUN =",
        summary[
            "alignment_model_forward_count_this_run"
        ],
    )
    print(
        "COMPLETE_PROTOCOL_FORWARD_COUNT =",
        summary[
            "complete_protocol_forward_count"
        ],
    )
    print(
        "R_ALIGN_OBSERVED =",
        summary["r_align_observed"],
    )
    print("H1/H2 = NOT EXECUTED")


if __name__ == "__main__":
    main()
