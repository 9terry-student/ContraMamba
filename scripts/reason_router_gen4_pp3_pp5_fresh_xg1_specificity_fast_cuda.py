from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_pp3_xg1_external_transport_fast_cuda
    as prior,
)
from scripts import (
    reason_router_gen4_xg1_fresh_specificity_tokenizer_anchor_eligibility
    as eligibility,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

DESIGN_FREEZE_COMMIT = "0bc49ab95cbb2c8735b4bc79422d660fa64e3e01"
STATIC_PREPARATION_FREEZE_COMMIT = (
    "ddb1404800af6dbd89982bbbcdd8262d203577f6"
)
ELIGIBILITY_FREEZE_COMMIT = (
    "3613d3a6a21d6fb2bef3692e3c8203b5ce0f37ec"
)
IMPLEMENTATION_AUTHORITY_COMMIT = (
    "1db39ffbbcc0c0558edc859d3260e50132d8c858"
)

AUTHORITY_PATH = (
    "reports/"
    "reason_router_gen4_pp3_pp5_fresh_xg1_specificity_"
    "implementation_authority.md"
)
AUTHORITY_BLOB = "a02787f969fb2674d10803f3f62ea21e3cafb7ce"

PRIOR_RUNNER_PATH = (
    "scripts/reason_router_gen4_pp3_xg1_external_transport_fast_cuda.py"
)
PRIOR_RUNNER_BLOB = "1b81deacc330beb9a7cf1d09b520ec55aaf2cc0d"

FRESH_ELIGIBILITY_GATE_PATH = (
    "scripts/"
    "reason_router_gen4_xg1_fresh_specificity_"
    "tokenizer_anchor_eligibility.py"
)
FRESH_ELIGIBILITY_GATE_BLOB = (
    "5e404406638740f8043fe2870b2e2cad21f4f924"
)

FRESH_COHORT_DIR = Path(
    "data/reason_router_gen4_xg1_fresh_specificity_v1"
)
FRESH_SOURCE_SHA256 = (
    "aa5b8e3cfcbf19e71335ecdbea659326925f8bea33312c2354de670fa7a15cf7"
)
FRESH_ROWS_SHA256 = (
    "3f28d8a75008d383855313a08168fef1a2b9b37257103636a7f2edb65ce76ad6"
)
FRESH_STRUCTURAL_MANIFEST_SHA256 = (
    "92ae0137641691c2a9d0254e87739e3e89d90556dc0579a2877433b86ff97dfb"
)

FRESH_ELIGIBILITY_DIR = Path(
    "reports/"
    "reason_router_gen4_xg1_fresh_specificity_"
    "tokenizer_anchor_eligibility_ddb1404"
)
FRESH_ANCHOR_MANIFEST = FRESH_ELIGIBILITY_DIR / "anchor_manifest.jsonl"
FRESH_ELIGIBILITY_SUMMARY = (
    FRESH_ELIGIBILITY_DIR / "eligibility_summary.json"
)
FRESH_ANCHOR_MANIFEST_SHA256 = (
    "4f7eb6b8660212a9db64b679370262fe4ec9b57cff0ab99121b039c0db141e04"
)
FRESH_ELIGIBILITY_SUMMARY_SHA256 = (
    "d546db6cce692e7000d382570ef15802770f5e6267d266e66f08b5efe4fbe5d2"
)

PP5_PREPARATION_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab"
)
PP5_PREPARATION_MANIFEST_FILE = "preparation_manifest.json"
PP5_PREPARATION_CHECKSUM_FILE = "SHA256SUMS.txt"
PP5_PLUS_FILE = "pp5_plus.f64le"
PP5_MINUS_FILE = "pp5_minus.f64le"

PP5_PREPARATION_MANIFEST_SHA256 = (
    "f403969f2c099d5227c28edae70696347b57bb7d96e1ab18fd5dcf24299365be"
)
PP5_PLUS_SHA256 = (
    "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2"
)
PP5_MINUS_SHA256 = (
    "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855"
)

PP3_PLUS_SHA256 = prior.PP3_PLUS_SHA256
PP3_MINUS_SHA256 = prior.PP3_MINUS_SHA256

SOURCE_PAIR_COUNT = 300
AMBIENT_DIM = prior.AMBIENT_DIM
EPSILON = prior.EPSILON
S3 = prior.S3
S5 = 0.99986792842854511

VECTOR_NORM_TOL = prior.VECTOR_NORM_TOL
VECTOR_DOT_TOL = prior.VECTOR_DOT_TOL

FORWARDS_PER_SIGNED_PROBE = prior.FORWARDS_PER_SIGNED_PROBE
FORWARDS_PER_DIRECTION = prior.FORWARDS_PER_DIRECTION
DIRECTIONS_PER_PAIR = 4
FORWARDS_PER_PAIR = FORWARDS_PER_DIRECTION * DIRECTIONS_PER_PAIR
SCIENTIFIC_FORWARD_BUDGET = SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
BASELINE_FORWARD_BUDGET_THIS_RUN = 0

DIRECTION_ORDER = (
    "pp3_plus",
    "pp3_minus",
    "pp5_plus",
    "pp5_minus",
)
DIRECTION_SHA256 = {
    "pp3_plus": PP3_PLUS_SHA256,
    "pp3_minus": PP3_MINUS_SHA256,
    "pp5_plus": PP5_PLUS_SHA256,
    "pp5_minus": PP5_MINUS_SHA256,
}

PROBE_SEED_SCHEMA = (
    "gen4-pp3-pp5-fresh-xg1-specificity-probe-seed-v1"
)
DIRECTION_PROBE_SCHEMA = (
    "gen4-pp3-pp5-fresh-xg1-specificity-direction-probe-v1"
)
ITEM_SCHEMA = (
    "gen4-pp3-pp5-fresh-xg1-specificity-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-pp3-pp5-fresh-xg1-specificity-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-pp3-pp5-fresh-xg1-specificity-manifest-v1"
)

RESULT_PASS = (
    "PASS_PP3_PP5_FRESH_XG1_SPECIFICITY_OBSERVATION"
)

ITEM_FILE = "pp3_pp5_fresh_xg1_specificity_items.jsonl"
SUMMARY_FILE = "pp3_pp5_fresh_xg1_specificity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class FreshSpecificityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshSpecificityError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        raise FreshSpecificityError(
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


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        head == expected_head,
        f"HEAD_MISMATCH:{head}",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    for commit, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN_FREEZE"),
        (
            STATIC_PREPARATION_FREEZE_COMMIT,
            "STATIC_PREPARATION_FREEZE",
        ),
        (
            ELIGIBILITY_FREEZE_COMMIT,
            "ELIGIBILITY_FREEZE",
        ),
        (
            IMPLEMENTATION_AUTHORITY_COMMIT,
            "IMPLEMENTATION_AUTHORITY",
        ),
    ):
        require(
            _git_is_ancestor(commit, expected_head),
            f"{label}_NOT_ANCESTOR",
        )

    frozen_blobs = {
        AUTHORITY_PATH: AUTHORITY_BLOB,
        PRIOR_RUNNER_PATH: PRIOR_RUNNER_BLOB,
        FRESH_ELIGIBILITY_GATE_PATH:
            FRESH_ELIGIBILITY_GATE_BLOB,
    }

    for path, expected_blob in frozen_blobs.items():
        observed = git(
            "rev-parse",
            f"HEAD:{path}",
        )
        require(
            observed == expected_blob,
            f"FROZEN_BLOB_DRIFT:{path}:{observed}",
        )

    # Reuse the previously validated CUDA/runtime dependency chain.
    prior.authenticate_repo(expected_head)


def _expected_pairs() -> tuple[str, ...]:
    return tuple(
        f"xg1_fact_{index:03d}"
        for index in range(301, 601)
    )


def _read_jsonl(
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

        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT_REQUIRED:{path}:{line_no}",
        )
        rows.append(value)

    return rows


def _read_checksum_file(
    path: Path,
) -> dict[str, str]:
    rows: dict[str, str] = {}

    for line in path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue

        digest, name = line.split("  ", 1)
        require(
            name not in rows,
            f"CHECKSUM_DUPLICATE:{name}",
        )
        rows[name] = digest

    return rows


def _vector_from_raw(
    raw: bytes,
    label: str,
) -> torch.Tensor:
    require(
        len(raw) == AMBIENT_DIM * 8,
        f"VECTOR_BYTE_LENGTH:{label}:{len(raw)}",
    )

    values = struct.unpack(
        f"<{AMBIENT_DIM}d",
        raw,
    )
    vector = torch.tensor(
        values,
        dtype=torch.float64,
    ).contiguous()

    require(
        tuple(vector.shape) == (AMBIENT_DIM,),
        f"VECTOR_SHAPE:{label}",
    )
    require(
        bool(torch.isfinite(vector).all().item()),
        f"VECTOR_NONFINITE:{label}",
    )

    return vector


def _validate_unit_pair(
    plus: torch.Tensor,
    minus: torch.Tensor,
    *,
    label: str,
) -> dict[str, float | int]:
    plus_norm = float(
        torch.linalg.vector_norm(
            plus,
            ord=2,
        ).item()
    )
    minus_norm = float(
        torch.linalg.vector_norm(
            minus,
            ord=2,
        ).item()
    )
    dot = float(
        torch.dot(
            plus,
            minus,
        ).item()
    )

    require(
        math.isfinite(plus_norm)
        and abs(plus_norm - 1.0)
        <= VECTOR_NORM_TOL,
        f"{label}_PLUS_NORM:{plus_norm}",
    )
    require(
        math.isfinite(minus_norm)
        and abs(minus_norm - 1.0)
        <= VECTOR_NORM_TOL,
        f"{label}_MINUS_NORM:{minus_norm}",
    )
    require(
        math.isfinite(dot)
        and abs(dot) <= VECTOR_DOT_TOL,
        f"{label}_ORTHOGONALITY:{dot}",
    )

    plus_pivot = int(
        torch.argmax(
            torch.abs(plus)
        ).item()
    )
    minus_pivot = int(
        torch.argmax(
            torch.abs(minus)
        ).item()
    )

    require(
        float(plus[plus_pivot].item()) > 0.0,
        f"{label}_PLUS_SIGN",
    )
    require(
        float(minus[minus_pivot].item()) > 0.0,
        f"{label}_MINUS_SIGN",
    )

    return {
        "plus_norm": plus_norm,
        "minus_norm": minus_norm,
        "plus_minus_dot": dot,
        "plus_pivot": plus_pivot,
        "minus_pivot": minus_pivot,
    }


def load_frozen_vectors() -> dict[str, Any]:
    pp3 = prior.load_pp3_vectors()

    pp3_metrics = _validate_unit_pair(
        pp3["pp3_plus"],
        pp3["pp3_minus"],
        label="PP3",
    )
    require(
        pp3_metrics["plus_pivot"] == 267,
        "PP3_PLUS_PIVOT",
    )
    require(
        pp3_metrics["minus_pivot"] == 23,
        "PP3_MINUS_PIVOT",
    )

    prep = ROOT / PP5_PREPARATION_ROOT
    manifest_path = (
        prep / PP5_PREPARATION_MANIFEST_FILE
    )
    checksum_path = (
        prep / PP5_PREPARATION_CHECKSUM_FILE
    )
    plus_path = prep / PP5_PLUS_FILE
    minus_path = prep / PP5_MINUS_FILE

    for path in (
        manifest_path,
        checksum_path,
        plus_path,
        minus_path,
    ):
        require(
            path.is_file(),
            f"PP5_PREPARATION_FILE_MISSING:{path}",
        )

    require(
        sha256_file(manifest_path)
        == PP5_PREPARATION_MANIFEST_SHA256,
        "PP5_PREPARATION_MANIFEST_SHA256",
    )
    require(
        sha256_file(plus_path)
        == PP5_PLUS_SHA256,
        "PP5_PLUS_SHA256",
    )
    require(
        sha256_file(minus_path)
        == PP5_MINUS_SHA256,
        "PP5_MINUS_SHA256",
    )

    checksums = _read_checksum_file(
        checksum_path
    )
    require(
        checksums
        == {
            PP5_MINUS_FILE:
                PP5_MINUS_SHA256,
            PP5_PLUS_FILE:
                PP5_PLUS_SHA256,
            PP5_PREPARATION_MANIFEST_FILE:
                PP5_PREPARATION_MANIFEST_SHA256,
        },
        "PP5_PREPARATION_CHECKSUM_CONTENT",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        manifest.get("schema_version")
        == (
            "GEN4_PP3_PP5_FRESH_XG1_"
            "SPECIFICITY_PREPARATION_V1"
        ),
        "PP5_PREPARATION_SCHEMA",
    )
    require(
        manifest.get("result")
        == (
            "PASS_PP3_PP5_FRESH_XG1_"
            "OUTCOME_BLIND_PREPARATION"
        ),
        "PP5_PREPARATION_RESULT",
    )
    require(
        manifest.get("design_freeze_commit")
        == DESIGN_FREEZE_COMMIT,
        "PP5_PREPARATION_DESIGN_FREEZE",
    )
    require(
        manifest.get("ambient_dim")
        == AMBIENT_DIM,
        "PP5_PREPARATION_AMBIENT_DIM",
    )

    pp3_reproduction = manifest.get(
        "pp3_reproduction"
    )
    require(
        isinstance(
            pp3_reproduction,
            dict,
        ),
        "PP3_REPRODUCTION_OBJECT",
    )
    require(
        pp3_reproduction.get(
            "frozen_bytes_reproduced_exactly"
        )
        is True,
        "PP3_REPRODUCTION_NOT_EXACT",
    )
    require(
        pp3_reproduction.get(
            "plus_sha256"
        )
        == PP3_PLUS_SHA256,
        "PP3_REPRODUCTION_PLUS_SHA",
    )
    require(
        pp3_reproduction.get(
            "minus_sha256"
        )
        == PP3_MINUS_SHA256,
        "PP3_REPRODUCTION_MINUS_SHA",
    )

    pp5 = manifest.get("pp5")
    require(
        isinstance(pp5, dict),
        "PP5_MANIFEST_OBJECT",
    )
    require(
        pp5.get(
            "scientific_principal_pair_number"
        )
        == 5,
        "PP5_NUMBER",
    )
    require(
        float(pp5["s5"]) == S5,
        "PP5_S5",
    )
    require(
        pp5.get("plus_sha256")
        == PP5_PLUS_SHA256,
        "PP5_MANIFEST_PLUS_SHA",
    )
    require(
        pp5.get("minus_sha256")
        == PP5_MINUS_SHA256,
        "PP5_MANIFEST_MINUS_SHA",
    )

    science = manifest.get(
        "scientific_execution"
    )
    require(
        isinstance(science, dict),
        "PREPARATION_SCIENCE_OBJECT",
    )
    require(
        science.get("model_forward_count")
        == 0,
        "PREPARATION_MODEL_FORWARD",
    )
    require(
        science.get("checkpoint_load_count")
        == 0,
        "PREPARATION_CHECKPOINT_LOAD",
    )
    require(
        science.get("cuda_executed")
        is False,
        "PREPARATION_CUDA",
    )
    require(
        science.get(
            "scientific_outcomes_observed"
        )
        is False,
        "PREPARATION_OUTCOME_BOUNDARY",
    )

    pp5_plus = _vector_from_raw(
        plus_path.read_bytes(),
        "pp5_plus",
    )
    pp5_minus = _vector_from_raw(
        minus_path.read_bytes(),
        "pp5_minus",
    )

    pp5_metrics = _validate_unit_pair(
        pp5_plus,
        pp5_minus,
        label="PP5",
    )

    require(
        pp5_metrics["plus_pivot"] == 319,
        "PP5_PLUS_PIVOT",
    )
    require(
        pp5_metrics["minus_pivot"] == 197,
        "PP5_MINUS_PIVOT",
    )

    return {
        "pp3_plus": pp3["pp3_plus"],
        "pp3_minus": pp3["pp3_minus"],
        "pp5_plus": pp5_plus,
        "pp5_minus": pp5_minus,
        "pp3_metrics": pp3_metrics,
        "pp5_metrics": pp5_metrics,
    }


def load_fresh_anchor_manifest() -> list[
    dict[str, Any]
]:
    manifest_path = (
        ROOT / FRESH_ANCHOR_MANIFEST
    )
    summary_path = (
        ROOT / FRESH_ELIGIBILITY_SUMMARY
    )

    require(
        manifest_path.is_file(),
        "FRESH_ANCHOR_MANIFEST_MISSING",
    )
    require(
        summary_path.is_file(),
        "FRESH_ELIGIBILITY_SUMMARY_MISSING",
    )

    require(
        sha256_file(manifest_path)
        == FRESH_ANCHOR_MANIFEST_SHA256,
        "FRESH_ANCHOR_MANIFEST_SHA256",
    )
    require(
        sha256_file(summary_path)
        == FRESH_ELIGIBILITY_SUMMARY_SHA256,
        "FRESH_ELIGIBILITY_SUMMARY_SHA256",
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get(
            "primary_complete_pair_prefix_feasibility"
        )
        == "PASS_300_OF_300",
        "FRESH_ELIGIBILITY_NOT_PASS",
    )
    require(
        summary.get(
            "complete_source_pair_count"
        )
        == SOURCE_PAIR_COUNT,
        "FRESH_ELIGIBILITY_PAIR_COUNT",
    )
    require(
        summary.get(
            "required_anchor_row_count"
        )
        == 1800,
        "FRESH_ELIGIBILITY_ANCHOR_COUNT",
    )
    require(
        summary.get(
            "eligible_anchor_counts"
        )
        == {
            "A_IDENTITY": 1200,
            "A_NAME": 600,
        },
        "FRESH_ELIGIBILITY_ANCHOR_COUNTS",
    )
    require(
        summary.get("exclusion_counts")
        == {},
        "FRESH_ELIGIBILITY_EXCLUSIONS",
    )
    require(
        summary.get(
            "target_identity_name_mismatch_count"
        )
        == 0,
        "FRESH_ELIGIBILITY_TARGET_MISMATCH",
    )
    require(
        summary.get("model_forward_count")
        == 0,
        "FRESH_ELIGIBILITY_MODEL_FORWARD",
    )
    require(
        summary.get("checkpoint_load_count")
        == 0,
        "FRESH_ELIGIBILITY_CHECKPOINT",
    )
    require(
        summary.get("gpu_used")
        is False,
        "FRESH_ELIGIBILITY_GPU",
    )
    require(
        summary.get(
            "scientific_outcomes_observed"
        )
        is False,
        "FRESH_ELIGIBILITY_OUTCOME_BOUNDARY",
    )
    require(
        summary.get(
            "anchor_manifest_sha256"
        )
        == FRESH_ANCHOR_MANIFEST_SHA256,
        "FRESH_ELIGIBILITY_MANIFEST_IDENTITY",
    )

    frozen_input = summary.get(
        "frozen_input"
    )
    require(
        isinstance(frozen_input, dict),
        "FRESH_FROZEN_INPUT_OBJECT",
    )
    require(
        frozen_input.get(
            "source_facts_sha256"
        )
        == FRESH_SOURCE_SHA256,
        "FRESH_SOURCE_SHA",
    )
    require(
        frozen_input.get("rows_sha256")
        == FRESH_ROWS_SHA256,
        "FRESH_ROWS_SHA",
    )
    require(
        frozen_input.get(
            "structural_manifest_sha256"
        )
        == FRESH_STRUCTURAL_MANIFEST_SHA256,
        "FRESH_STRUCTURAL_SHA",
    )
    require(
        frozen_input.get("pair_id_first")
        == "xg1_fact_301",
        "FRESH_PAIR_FIRST",
    )
    require(
        frozen_input.get("pair_id_last")
        == "xg1_fact_600",
        "FRESH_PAIR_LAST",
    )

    rows = _read_jsonl(
        manifest_path
    )
    require(
        len(rows) == 1800,
        "FRESH_ANCHOR_ROW_COUNT",
    )

    require(
        Counter(
            row["anchor_name"]
            for row in rows
        )
        == Counter(
            {
                "A_IDENTITY": 1200,
                "A_NAME": 600,
            }
        ),
        "FRESH_ANCHOR_NAME_COUNTS",
    )

    for index, row in enumerate(rows):
        require(
            row.get("schema_version")
            == eligibility.ANCHOR_MANIFEST_SCHEMA,
            f"FRESH_ANCHOR_SCHEMA:{index}",
        )
        require(
            row.get("post4_eligible")
            is True,
            f"FRESH_ANCHOR_INELIGIBLE:{index}",
        )
        require(
            row.get("exclusion_code")
            is None,
            f"FRESH_ANCHOR_EXCLUSION:{index}",
        )
        require(
            type(
                row.get(
                    "absolute_anchor_token_index"
                )
            )
            is int,
            f"FRESH_ANCHOR_INDEX_TYPE:{index}",
        )

    return rows


def _pair_order(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()

    for row in rows:
        pair = str(
            row["source_pair_id"]
        )
        if pair not in seen:
            seen.add(pair)
            order.append(pair)

    expected = _expected_pairs()
    require(
        tuple(order) == expected,
        "FRESH_XG1_PAIR_ORDER",
    )
    return tuple(order)


def validate_fresh_population(
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    adapter = prior.xg1_eq.adapter
    parent = (
        prior.holdout
        .phase1
        .base
        .prevalence_eq
        .parent
    )

    normalized = (
        adapter.validate_gen4_rows(
            rows,
            require_canonical_shape=True,
        )
    )
    pairs = _pair_order(normalized)

    require(
        list(encoded["row_id"])
        == [
            str(row["row_id"])
            for row in normalized
        ],
        "ENCODED_ROW_ORDER",
    )
    require(
        list(encoded["source_pair_id"])
        == [
            str(row["source_pair_id"])
            for row in normalized
        ],
        "ENCODED_PAIR_ORDER",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [
            str(row["contrast_cell_id"])
            for row in normalized
        ],
        "ENCODED_CELL_ORDER",
    )

    input_ids = encoded["input_ids"]
    attention = encoded[
        "attention_mask"
    ]
    claim_mask = encoded[
        "claim_mask"
    ]
    evidence_mask = encoded[
        "evidence_mask"
    ]

    require(
        torch.is_tensor(input_ids),
        "INPUT_IDS_TENSOR",
    )
    require(
        tuple(input_ids.shape)
        == (
            1800,
            adapter.MAX_LENGTH,
        ),
        "INPUT_IDS_SHAPE",
    )

    row_index = {
        str(row["row_id"]): index
        for index, row
        in enumerate(normalized)
    }
    require(
        len(row_index) == 1800,
        "ROW_ID_CARDINALITY",
    )

    events = parent.event_lookup(
        event_rows
    )
    parent.validate_transport_event_plan(
        pairs,
        events,
    )

    for event in event_rows:
        row_id = str(event["row_id"])
        require(
            row_id in row_index,
            f"EVENT_ROW_ID:{row_id}",
        )

        index = row_index[row_id]

        anchor = int(
            event[
                "absolute_anchor_token_index"
            ]
        )
        terminal = int(
            event["terminal_index"]
        )
        claim_count = int(
            event[
                "claim_consumed_token_count"
            ]
        )
        evidence_count = int(
            event[
                "evidence_consumed_token_count"
            ]
        )
        evidence_anchor = int(
            event[
                "anchor_evidence_token_index"
            ]
        )

        require(
            anchor
            == claim_count
            + 1
            + evidence_anchor,
            "ANCHOR_COORDINATE",
        )
        require(
            terminal
            == claim_count
            + evidence_count,
            "TERMINAL_COORDINATE",
        )
        require(
            anchor + 4
            <= terminal - 1,
            "POST4_RULE",
        )
        require(
            int(
                attention[index]
                .sum()
                .item()
            )
            == terminal + 1,
            "ATTENTION_TERMINAL",
        )
        require(
            int(
                claim_mask[index]
                .sum()
                .item()
            )
            == claim_count,
            "CLAIM_MASK_COUNT",
        )
        require(
            int(
                evidence_mask[index]
                .sum()
                .item()
            )
            == evidence_count,
            "EVIDENCE_MASK_COUNT",
        )

    return pairs


def load_fresh_inputs(
    tokenizer_snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    facts, rows, _manifest = (
        eligibility.load_frozen_fresh_inputs()
    )

    require(
        len(facts) == SOURCE_PAIR_COUNT,
        "FRESH_FACT_COUNT",
    )

    tokenizer, _provenance = (
        eligibility
        .legacy
        .load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    encoded = (
        prior.xg1_eq
        .adapter
        .encode_gen4_rows(
            rows,
            tokenizer,
        )
    )

    event_rows = (
        load_fresh_anchor_manifest()
    )

    validate_fresh_population(
        rows,
        encoded,
        event_rows,
    )

    return rows, encoded, event_rows


def _probe_seed(
    index: int,
    pair: str,
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    require(
        pair == _expected_pairs()[index],
        f"PROBE_SEED_PAIR:{index}:{pair}",
    )

    anchors = (
        prior.holdout
        .phase1
        ._anchors_for_pair(
            pair,
            events,
        )
    )

    return {
        "schema_version":
            PROBE_SEED_SCHEMA,
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "pair_ordinal": index + 1,
        "target_plus_anchor":
            int(anchors["tp"]),
        "target_minus_anchor":
            int(anchors["tm"]),
        "reference_plus_anchor":
            int(anchors["rp"]),
        "reference_minus_anchor":
            int(anchors["rm"]),
    }


def _run_direction_j(
    seed: Mapping[str, Any],
    unit_direction: torch.Tensor,
    *,
    direction_key: str,
    vector_sha256: str,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[
        tuple[str, str],
        int,
    ],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(
        direction_key
        in DIRECTION_ORDER,
        f"BAD_DIRECTION:{direction_key}",
    )
    require(
        vector_sha256
        == DIRECTION_SHA256[
            direction_key
        ],
        f"DIRECTION_SHA256:{direction_key}",
    )

    direction = (
        unit_direction
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
        .clone()
    )

    require(
        tuple(direction.shape)
        == (AMBIENT_DIM,),
        f"DIRECTION_SHAPE:{direction_key}",
    )

    norm = float(
        torch.linalg.vector_norm(
            direction,
            ord=2,
        ).item()
    )
    require(
        math.isfinite(norm)
        and abs(norm - 1.0)
        <= VECTOR_NORM_TOL,
        f"DIRECTION_NORM:{direction_key}:{norm}",
    )

    positive = prior._run_signed_probe(
        seed,
        direction,
        orientation=1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )

    negative = prior._run_signed_probe(
        seed,
        direction,
        orientation=-1,
        model=model,
        runtime_ctx=runtime_ctx,
        trace_code=trace_code,
        trace_line=trace_line,
        encoded=encoded,
        row_index=row_index,
        events=events,
        budget=budget,
    )

    f_plus = float(
        positive["F"]
    )
    f_minus = float(
        negative["F"]
    )

    j_value = (
        f_plus - f_minus
    ) / (2.0 * EPSILON)
    j_squared = (
        j_value * j_value
    )

    require(
        all(
            math.isfinite(value)
            for value in (
                f_plus,
                f_minus,
                j_value,
                j_squared,
            )
        ),
        f"DIRECTION_NONFINITE:{direction_key}",
    )

    return {
        "schema_version":
            DIRECTION_PROBE_SCHEMA,
        "direction_key":
            direction_key,
        "vector_sha256":
            vector_sha256,
        "epsilon":
            EPSILON,
        "F_plus":
            f_plus,
        "F_minus":
            f_minus,
        "J":
            j_value,
        "J_squared":
            j_squared,
        "positive_probe":
            positive,
        "negative_probe":
            negative,
        "model_forward_count":
            FORWARDS_PER_DIRECTION,
    }


def _run_pair(
    seed: Mapping[str, Any],
    *,
    vectors: Mapping[str, torch.Tensor],
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[
        tuple[str, str],
        int,
    ],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    probes: dict[str, Any] = {}

    for direction_key in DIRECTION_ORDER:
        probes[direction_key] = (
            _run_direction_j(
                seed,
                vectors[direction_key],
                direction_key=
                    direction_key,
                vector_sha256=
                    DIRECTION_SHA256[
                        direction_key
                    ],
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=budget,
            )
        )

    j3p = float(
        probes["pp3_plus"]["J"]
    )
    j3m = float(
        probes["pp3_minus"]["J"]
    )
    j5p = float(
        probes["pp5_plus"]["J"]
    )
    j5m = float(
        probes["pp5_minus"]["J"]
    )

    c3 = (
        S3 / 5.0
    ) * (
        j3p * j3p
        - j3m * j3m
    )
    c5 = (
        S5 / 5.0
    ) * (
        j5p * j5p
        - j5m * j5m
    )
    d_spec = c3 - c5

    require(
        all(
            math.isfinite(value)
            for value in (
                j3p,
                j3m,
                j5p,
                j5m,
                c3,
                c5,
                d_spec,
            )
        ),
        "PAIR_NONFINITE",
    )

    item = dict(seed)
    item[
        "probe_seed_schema_version"
    ] = item["schema_version"]
    item["schema_version"] = (
        ITEM_SCHEMA
    )

    item[
        "implementation_authority_commit"
    ] = IMPLEMENTATION_AUTHORITY_COMMIT
    item[
        "static_preparation_freeze_commit"
    ] = STATIC_PREPARATION_FREEZE_COMMIT
    item[
        "eligibility_freeze_commit"
    ] = ELIGIBILITY_FREEZE_COMMIT

    item["epsilon"] = EPSILON
    item["s3"] = S3
    item["s5"] = S5

    item[
        "pp3_plus_sha256"
    ] = PP3_PLUS_SHA256
    item[
        "pp3_minus_sha256"
    ] = PP3_MINUS_SHA256
    item[
        "pp5_plus_sha256"
    ] = PP5_PLUS_SHA256
    item[
        "pp5_minus_sha256"
    ] = PP5_MINUS_SHA256

    item["direction_order"] = list(
        DIRECTION_ORDER
    )
    item[
        "direction_probes"
    ] = probes

    item["J_PP3_PLUS"] = j3p
    item["J_PP3_MINUS"] = j3m
    item["J_PP5_PLUS"] = j5p
    item["J_PP5_MINUS"] = j5m

    item["C_PP3"] = c3
    item["C_PP5"] = c5
    item["D_SPEC"] = d_spec

    item[
        "baseline_model_forward_count_this_run"
    ] = 0
    item[
        "scientific_model_forward_count_this_run"
    ] = FORWARDS_PER_PAIR

    return item


def _validate_signed_probe(
    probe: Mapping[str, Any],
    orientation: int,
) -> None:
    require(
        int(probe["orientation"])
        == orientation,
        "SIGNED_PROBE_ORIENTATION",
    )
    require(
        int(probe["model_forward_count"])
        == FORWARDS_PER_SIGNED_PROBE,
        "SIGNED_PROBE_FORWARD_COUNT",
    )
    require(
        float(probe["delta_h_l2"])
        == 2.0 * EPSILON,
        "SIGNED_PROBE_DELTA_L2",
    )

    for field in (
        "plus_path_efficiency",
        "minus_path_efficiency",
        "F",
        "midpoint_max_abs_residual",
        "pair_delta_max_abs_residual",
        "applied_correction_max_abs_residual",
        "runtime_correction_l2",
    ):
        require(
            math.isfinite(
                float(probe[field])
            ),
            f"SIGNED_PROBE_NONFINITE:{field}",
        )

    require(
        abs(
            float(
                probe[
                    "runtime_correction_l2"
                ]
            )
            - 2.0 * EPSILON
        )
        <= 1.0e-12,
        "SIGNED_PROBE_CORRECTION_L2",
    )


def _validate_direction_probe(
    probe: Mapping[str, Any],
    direction_key: str,
) -> None:
    require(
        probe.get("schema_version")
        == DIRECTION_PROBE_SCHEMA,
        "DIRECTION_SCHEMA",
    )
    require(
        probe.get("direction_key")
        == direction_key,
        "DIRECTION_KEY",
    )
    require(
        probe.get("vector_sha256")
        == DIRECTION_SHA256[
            direction_key
        ],
        "DIRECTION_VECTOR_SHA",
    )
    require(
        float(probe["epsilon"])
        == EPSILON,
        "DIRECTION_EPSILON",
    )
    require(
        int(
            probe[
                "model_forward_count"
            ]
        )
        == FORWARDS_PER_DIRECTION,
        "DIRECTION_FORWARD_COUNT",
    )

    _validate_signed_probe(
        probe["positive_probe"],
        1,
    )
    _validate_signed_probe(
        probe["negative_probe"],
        -1,
    )

    f_plus = float(
        probe["F_plus"]
    )
    f_minus = float(
        probe["F_minus"]
    )
    j_value = float(
        probe["J"]
    )
    j_squared = float(
        probe["J_squared"]
    )

    require(
        j_value
        == (
            f_plus - f_minus
        )
        / (2.0 * EPSILON),
        "DIRECTION_J_IDENTITY",
    )
    require(
        j_squared
        == j_value * j_value,
        "DIRECTION_J_SQUARED_IDENTITY",
    )


def _validate_items(
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        len(items)
        == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    expected_pairs = (
        _expected_pairs()
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
            row.get(
                "probe_seed_schema_version"
            )
            == PROBE_SEED_SCHEMA,
            f"PROBE_SEED_SCHEMA:{index}",
        )
        require(
            row.get("family_key")
            == "xg1",
            f"ITEM_FAMILY:{index}",
        )
        require(
            row.get("source_pair_id")
            == expected_pair,
            f"PAIR_ORDER:{index}",
        )
        require(
            row.get("pair_index")
            == index,
            f"PAIR_INDEX:{index}",
        )
        require(
            row.get("pair_ordinal")
            == index + 1,
            f"PAIR_ORDINAL:{index}",
        )

        require(
            row.get(
                "implementation_authority_commit"
            )
            == IMPLEMENTATION_AUTHORITY_COMMIT,
            f"IMPLEMENTATION_AUTHORITY:{index}",
        )
        require(
            row.get(
                "static_preparation_freeze_commit"
            )
            == STATIC_PREPARATION_FREEZE_COMMIT,
            f"STATIC_FREEZE:{index}",
        )
        require(
            row.get(
                "eligibility_freeze_commit"
            )
            == ELIGIBILITY_FREEZE_COMMIT,
            f"ELIGIBILITY_FREEZE:{index}",
        )

        require(
            float(row["epsilon"])
            == EPSILON,
            f"EPSILON:{index}",
        )
        require(
            float(row["s3"])
            == S3,
            f"S3:{index}",
        )
        require(
            float(row["s5"])
            == S5,
            f"S5:{index}",
        )

        require(
            row.get("direction_order")
            == list(DIRECTION_ORDER),
            f"DIRECTION_ORDER:{index}",
        )

        probes = row.get(
            "direction_probes"
        )
        require(
            isinstance(probes, dict),
            f"DIRECTION_PROBES:{index}",
        )
        require(
            set(probes)
            == set(DIRECTION_ORDER),
            f"DIRECTION_PROBE_KEY_SET:{index}",
        )

        for key in DIRECTION_ORDER:
            _validate_direction_probe(
                probes[key],
                key,
            )

        j3p = float(
            row["J_PP3_PLUS"]
        )
        j3m = float(
            row["J_PP3_MINUS"]
        )
        j5p = float(
            row["J_PP5_PLUS"]
        )
        j5m = float(
            row["J_PP5_MINUS"]
        )

        require(
            j3p
            == float(
                probes[
                    "pp3_plus"
                ]["J"]
            ),
            f"J3P_IDENTITY:{index}",
        )
        require(
            j3m
            == float(
                probes[
                    "pp3_minus"
                ]["J"]
            ),
            f"J3M_IDENTITY:{index}",
        )
        require(
            j5p
            == float(
                probes[
                    "pp5_plus"
                ]["J"]
            ),
            f"J5P_IDENTITY:{index}",
        )
        require(
            j5m
            == float(
                probes[
                    "pp5_minus"
                ]["J"]
            ),
            f"J5M_IDENTITY:{index}",
        )

        expected_c3 = (
            S3 / 5.0
        ) * (
            j3p * j3p
            - j3m * j3m
        )
        expected_c5 = (
            S5 / 5.0
        ) * (
            j5p * j5p
            - j5m * j5m
        )
        expected_d = (
            expected_c3
            - expected_c5
        )

        require(
            float(row["C_PP3"])
            == expected_c3,
            f"C_PP3_IDENTITY:{index}",
        )
        require(
            float(row["C_PP5"])
            == expected_c5,
            f"C_PP5_IDENTITY:{index}",
        )
        require(
            float(row["D_SPEC"])
            == expected_d,
            f"D_SPEC_IDENTITY:{index}",
        )

        require(
            row.get(
                "baseline_model_forward_count_this_run"
            )
            == 0,
            f"BASELINE_FORWARD:{index}",
        )
        require(
            row.get(
                "scientific_model_forward_count_this_run"
            )
            == FORWARDS_PER_PAIR,
            f"SCIENTIFIC_FORWARD:{index}",
        )


def _canonical_json_bytes(
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


def _jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        _canonical_json_bytes(
            dict(row)
        )
        for row in rows
    )


def _make_summary(
    items: Sequence[Mapping[str, Any]],
    *,
    expected_head: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    _validate_items(items)

    return {
        "schema_version":
            SUMMARY_SCHEMA,
        "result":
            RESULT_PASS,
        "execution_head":
            expected_head,
        "design_freeze_commit":
            DESIGN_FREEZE_COMMIT,
        "static_preparation_freeze_commit":
            STATIC_PREPARATION_FREEZE_COMMIT,
        "eligibility_freeze_commit":
            ELIGIBILITY_FREEZE_COMMIT,
        "implementation_authority_commit":
            IMPLEMENTATION_AUTHORITY_COMMIT,
        "source_pair_count":
            SOURCE_PAIR_COUNT,
        "pair_id_first":
            items[0]["source_pair_id"],
        "pair_id_last":
            items[-1]["source_pair_id"],
        "epsilon":
            EPSILON,
        "s3":
            S3,
        "s5":
            S5,
        "pp3_plus_sha256":
            PP3_PLUS_SHA256,
        "pp3_minus_sha256":
            PP3_MINUS_SHA256,
        "pp5_plus_sha256":
            PP5_PLUS_SHA256,
        "pp5_minus_sha256":
            PP5_MINUS_SHA256,
        "direction_order":
            list(DIRECTION_ORDER),
        "model_forwards_per_direction":
            FORWARDS_PER_DIRECTION,
        "model_forwards_per_pair":
            FORWARDS_PER_PAIR,
        "scientific_model_forward_count_this_run":
            SCIENTIFIC_FORWARD_BUDGET,
        "baseline_model_forward_count_this_run":
            0,
        "primary_endpoint_definition":
            (
                "D_SPEC=C_PP3-C_PP5; "
                "C_PPk=(s_k/5)*(J_PLUS^2-J_MINUS^2)"
            ),
        "primary_endpoint_D_SPEC_observed":
            True,
        "C_PP3_observed":
            True,
        "C_PP5_observed":
            True,
        "finite_value_audit":
            "PASS",
        "primary_inference_executed":
            False,
        "multiplicity_correction_executed":
            False,
        "training_executed":
            False,
        "backward_executed":
            False,
        "task_heads_executed":
            False,
        "logits_read":
            False,
        "scientific_conclusion":
            None,
        "representative_checkpoint_sha256":
            checkpoint_sha256,
        "fresh_source_facts_sha256":
            FRESH_SOURCE_SHA256,
        "fresh_rows_sha256":
            FRESH_ROWS_SHA256,
        "fresh_structural_manifest_sha256":
            FRESH_STRUCTURAL_MANIFEST_SHA256,
        "fresh_anchor_manifest_sha256":
            FRESH_ANCHOR_MANIFEST_SHA256,
        "fresh_eligibility_summary_sha256":
            FRESH_ELIGIBILITY_SUMMARY_SHA256,
    }


def _write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )
    _validate_items(items)

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payloads = {
        ITEM_FILE:
            _jsonl_bytes(items),
        SUMMARY_FILE:
            _canonical_json_bytes(
                summary
            ),
    }

    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = (
            sha256_bytes(raw)
        )

    manifest = {
        "schema_version":
            MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int(
                    (
                        output_dir
                        / name
                    ).stat().st_size
                ),
            }
            for name, digest
            in sorted(
                hashes.items()
            )
        },
    }

    manifest_raw = (
        _canonical_json_bytes(
            manifest
        )
    )
    (
        output_dir
        / MANIFEST_FILE
    ).write_bytes(
        manifest_raw
    )

    hashes[MANIFEST_FILE] = (
        sha256_bytes(
            manifest_raw
        )
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest
        in sorted(
            hashes.items()
        )
    ).encode("utf-8")

    (
        output_dir
        / CHECKSUM_FILE
    ).write_bytes(
        checksum_raw
    )

    return hashes


def validate_artifact(
    output_dir: Path,
) -> dict[str, Any]:
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

    observed_hashes: dict[
        str,
        str,
    ] = {}

    for name in (
        ITEM_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / name
        require(
            path.is_file(),
            f"ARTIFACT_MISSING:{name}",
        )

        observed_sha = (
            sha256_file(path)
        )
        require(
            observed_sha
            == files[name]["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            int(path.stat().st_size)
            == int(
                files[name]["bytes"]
            ),
            f"ARTIFACT_BYTES:{name}",
        )

        observed_hashes[
            name
        ] = observed_sha

    observed_hashes[
        MANIFEST_FILE
    ] = sha256_file(
        manifest_path
    )

    require(
        _read_checksum_file(
            checksum_path
        )
        == {
            name: digest
            for name, digest
            in sorted(
                observed_hashes.items()
            )
        },
        "CHECKSUM_CONTENT",
    )

    items = _read_jsonl(
        output_dir / ITEM_FILE
    )
    _validate_items(items)

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
        summary.get(
            "source_pair_count"
        )
        == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        summary.get("pair_id_first")
        == "xg1_fact_301",
        "SUMMARY_FIRST_PAIR",
    )
    require(
        summary.get("pair_id_last")
        == "xg1_fact_600",
        "SUMMARY_LAST_PAIR",
    )
    require(
        summary.get(
            "direction_order"
        )
        == list(DIRECTION_ORDER),
        "SUMMARY_DIRECTION_ORDER",
    )
    require(
        summary.get(
            "model_forwards_per_direction"
        )
        == FORWARDS_PER_DIRECTION,
        "SUMMARY_FORWARD_DIRECTION",
    )
    require(
        summary.get(
            "model_forwards_per_pair"
        )
        == FORWARDS_PER_PAIR,
        "SUMMARY_FORWARD_PAIR",
    )
    require(
        summary.get(
            "scientific_model_forward_count_this_run"
        )
        == SCIENTIFIC_FORWARD_BUDGET,
        "SUMMARY_FORWARD_TOTAL",
    )
    require(
        summary.get(
            "baseline_model_forward_count_this_run"
        )
        == 0,
        "SUMMARY_BASELINE_FORWARD",
    )
    require(
        summary.get(
            "primary_endpoint_D_SPEC_observed"
        )
        is True,
        "SUMMARY_ENDPOINT_OBSERVED",
    )
    require(
        summary.get(
            "primary_inference_executed"
        )
        is False,
        "SUMMARY_INFERENCE_BOUNDARY",
    )
    require(
        summary.get(
            "multiplicity_correction_executed"
        )
        is False,
        "SUMMARY_MULTIPLICITY_BOUNDARY",
    )
    require(
        summary.get(
            "training_executed"
        )
        is False
        and summary.get(
            "backward_executed"
        )
        is False
        and summary.get(
            "task_heads_executed"
        )
        is False
        and summary.get(
            "logits_read"
        )
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

    return {
        "summary": summary,
        "items": items,
        "manifest": manifest,
    }


def run_observation(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(
        expected_head
    )
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    vectors = (
        load_frozen_vectors()
    )

    runtime = (
        prior.holdout
        .phase1
        .base
        .prevalence_eq
    )
    runtime.backend.runtime_gate()

    with (
        runtime.backend
        .parent_runtime_rebind()
    ):
        (
            rows,
            encoded,
            event_rows,
        ) = load_fresh_inputs(
            tokenizer_snapshot
        )

        pairs = _pair_order(rows)
        require(
            pairs == _expected_pairs(),
            "FRESH_PAIR_POPULATION",
        )

        parent = runtime.parent

        events = (
            parent.event_lookup(
                event_rows
            )
        )
        parent.validate_transport_event_plan(
            pairs,
            events,
        )

        row_index = (
            parent.build_row_index(
                rows
            )
        )

        trace_code, trace_line = (
            runtime.measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            runtime.kernel_compat
            .load_exact_fast_kernels()
        )

        with (
            runtime.kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_calls:
            model, checkpoint_sha = (
                parent
                .load_representative_model_external(
                    model_snapshot=
                        model_snapshot,
                    checkpoint_path=
                        checkpoint_path,
                )
            )

            require(
                checkpoint_sha
                == runtime.extraction
                .REPRESENTATIVE_CHECKPOINT_SHA256,
                "CHECKPOINT_IDENTITY",
            )

            runtime_ctx = (
                runtime
                .transport_runtime
                .validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(
            constructor_calls
        )
        require(
            set(
                constructor_counts
            )
            == {
                "causal-conv1d",
                "mamba-ssm",
            },
            "TRANSFORMERS_CONSTRUCTOR_KERNEL_NAMES",
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ]
            > 0
            and constructor_counts[
                "causal-conv1d"
            ]
            == constructor_counts[
                "mamba-ssm"
            ],
            "TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT",
        )

        runtime.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )

        model.to(
            torch.device(
                "cuda:0"
            )
        )
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
            runtime.backend
            ._make_fast_capture(
                kernels
            )
        )
        original_capture = (
            parent.capture_branch
        )

        budget = (
            parent.ForwardBudget(
                SCIENTIFIC_FORWARD_BUDGET
            )
        )

        items: list[
            dict[str, Any]
        ] = []

        parent.capture_branch = (
            fast_capture
        )

        try:
            for index, pair in enumerate(
                pairs
            ):
                seed = _probe_seed(
                    index,
                    pair,
                    events,
                )

                items.append(
                    _run_pair(
                        seed,
                        vectors={
                            key:
                                vectors[key]
                            for key
                            in DIRECTION_ORDER
                        },
                        model=model,
                        runtime_ctx=
                            runtime_ctx,
                        trace_code=
                            trace_code,
                        trace_line=
                            trace_line,
                        encoded=encoded,
                        row_index=
                            row_index,
                        events=events,
                        budget=budget,
                    )
                )

            budget.assert_exact()
            torch.cuda.synchronize()

        finally:
            parent.capture_branch = (
                original_capture
            )

    _validate_items(items)

    summary = _make_summary(
        items,
        expected_head=
            expected_head,
        checkpoint_sha256=
            checkpoint_sha,
    )

    hashes = _write_outputs(
        output_dir,
        items=items,
        summary=summary,
    )

    validated = (
        validate_artifact(
            output_dir
        )
    )

    return {
        "summary":
            validated["summary"],
        "hashes":
            hashes,
    }


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen PP3-vs-PP5 specificity observation "
            "on fresh XG1 301..600. "
            "No inferential test is executed."
        )
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
        "--checkpoint-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(
        argv
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    report = run_observation(
        expected_head=
            args.expected_head,
        model_snapshot=
            args.model_snapshot,
        tokenizer_snapshot=
            args.tokenizer_snapshot,
        checkpoint_path=
            args.checkpoint_path,
        output_dir=
            args.output_dir,
    )

    summary = report["summary"]

    print(
        "RESULT=",
        summary["result"],
        sep="",
    )
    print(
        "SOURCE_PAIR_COUNT=",
        summary[
            "source_pair_count"
        ],
        sep="",
    )
    print(
        "PAIR_ID_FIRST=",
        summary[
            "pair_id_first"
        ],
        sep="",
    )
    print(
        "PAIR_ID_LAST=",
        summary[
            "pair_id_last"
        ],
        sep="",
    )
    print(
        "DIRECTION_ORDER=",
        summary[
            "direction_order"
        ],
        sep="",
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=",
        summary[
            "scientific_model_forward_count_this_run"
        ],
        sep="",
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT_THIS_RUN=0"
    )
    print(
        "PRIMARY_INFERENCE_EXECUTED=False"
    )
    print(
        "SCIENTIFIC_CONCLUSION=None"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
