#!/usr/bin/env python3
"""Exact generator-anchor localization of recurrent constructive interference.

This audit refines the validated serialization-region decomposition into the
already-frozen XG1 atomic generator anchors A_TITLE/A_NAME/A_ROLE/A_PREDICATE
plus a mechanical RESIDUAL class.

The implementation is recurrence-only. It does not train, mutate checkpoints,
refit projectors, execute downstream transport, inspect decoded token strings,
load confirmatory seeds 9601/9900, or access VitaminC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import (
    audit_native_recurrent_constructive_interference_serialization_region_pair_localization
    as coarse,
)
from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as xg1
from scripts import prepare_reason_router_gen5_phase3_static as p3static
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as anchor

pairgap = coarse.pairgap
kernel = coarse.kernel
transport = coarse.transport
p3a = coarse.p3a

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_COMMIT = "55b8784aa154d766cd189f60b2919451720598de"
DESIGN_PATH = "reports/native_recurrent_generator_anchor_pair_localization_design_candidate.md"
DESIGN_BLOB = "39690866075d65366db5e7c631e8c0e2415f7e16"

VALIDATED_EVIDENCE_COMMIT = "ee61ac1f70816d7150ab9dc5af57a4b9f0714515"
VALIDATED_EVIDENCE_REPORT_PATH = (
    "reports/native_recurrent_constructive_interference_serialization_region_pair_"
    "localization_validated_evidence_analysis_report_candidate.md"
)
VALIDATED_EVIDENCE_REPORT_BLOB = "f16d035b5f02f6e55138ca283dcdf971418eaa12"

VALIDATED_COARSE_SCRIPT_PATH = (
    "scripts/audit_native_recurrent_constructive_interference_serialization_region_pair_localization.py"
)
VALIDATED_COARSE_SCRIPT_BLOB = "353663e225870d22241ca894e2774bed7a979e4d"

FROZEN_DEPENDENCY_BLOBS = {
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py":
        "6c98ce022ca134e385db28851fd364dc6daff423",
    "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py":
        "c830026935a6c9f4990c6a3315c75fd5580e7264",
    "scripts/prepare_reason_router_gen5_phase3_static.py":
        "5f2ec6498af475579b07e16b122be201a0ff656a",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/"
    "phase3_seven_cell_labeled_training.jsonl":
        "62c31fe06b5d83f76c1b75d4c8333aa6e31473cd",
    "data/reason_router_gen5_phase3_xg1_contention_training_v1/"
    "pair_split_manifest.json":
        "1f881fc46d7e4093d201eaf7cc9e98ca4294c424",
}

VALIDATED_COARSE_RUN = "gen5-recurrent-serialization-region-localization-032e68e-r1"
VALIDATED_COARSE_ROOT = (
    ROOT
    / "reports/native_recurrent_constructive_interference_serialization_region_pair_localization_runs"
    / VALIDATED_COARSE_RUN
)
VALIDATED_COARSE_FILES = {
    "serialization_region_pair_summary.json":
        "24006b8c650582348bb6b9754b4568f30c5bf079329eb1a02f7e0f1b536d78fa",
    "orientation_serialization_region_pair_metrics.jsonl":
        "a2eae4c48bbbcb3d677f44fc8189f344201d3e7b3991a4509906959d13d4eb13",
    "shard_manifest.json":
        "d0781d2c20d654e570b656d5b32feb5494078a01465aecbe6f5bdfb954c673da",
    "run_provenance.json":
        "fdaa9016fd003a16cc66ce0df85736add658a95d6ee8c6f04d0fc0d592c903a8",
}

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_native_recurrent_generator_anchor_pair_localization.py",
        "tests/test_native_recurrent_generator_anchor_pair_localization.py",
    }
)

OUTPUT_SCHEMA = "GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_PAIR_LOCALIZATION_V1"
WORKER_SCHEMA = "GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_PAIR_LOCALIZATION_WORKER_V1"
PROVENANCE_SCHEMA = "GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_PAIR_LOCALIZATION_PROVENANCE_V1"
SHARD_SCHEMA = "GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_PAIR_LOCALIZATION_SHARD_V1"

ATOMIC_ANCHORS = ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE")
FINE_CLASSES = ATOMIC_ANCHORS + ("RESIDUAL",)
PARENT_PAIRS = (
    ("CLAIM", "CLAIM"),
    ("EVIDENCE", "EVIDENCE"),
    ("CLAIM", "EVIDENCE"),
)
PARENT_PAIR_NAMES = tuple(f"{left}->{right}" for left, right in PARENT_PAIRS)
FINE_PAIR_NAMES = tuple(
    f"{left}->{right}"
    for left in FINE_CLASSES
    for right in FINE_CLASSES
)
FINE_CELL_NAMES = tuple(
    f"{parent}:{fine}"
    for parent in PARENT_PAIR_NAMES
    for fine in FINE_PAIR_NAMES
)

WINDOWS = coarse.WINDOWS
PAIR_GAP_ENERGY_RTOL = coarse.PAIR_GAP_ENERGY_RTOL
PAIR_GAP_ENERGY_ATOL = coarse.PAIR_GAP_ENERGY_ATOL
PAIR_GAP_LOG_ATOL = coarse.PAIR_GAP_LOG_ATOL
RAW_COMPONENT_MATCH_RTOL = coarse.RAW_COMPONENT_MATCH_RTOL
ENERGY_EPSILON = coarse.ENERGY_EPSILON
FFT_CHUNK_WIDTH = coarse.FFT_CHUNK_WIDTH
FAST_FLOAT32_HALF_SPAN_LIMIT = coarse.FAST_FLOAT32_HALF_SPAN_LIMIT

FACTOR_SEEDS = coarse.FACTOR_SEEDS
FULL_FACTORIAL_CELLS = coarse.FULL_FACTORIAL_CELLS
DEV_ROWS = coarse.DEV_ROWS
VALID_TOKEN_COUNT = coarse.VALID_TOKEN_COUNT
BATCH_ROWS = coarse.BATCH_ROWS
TARGETS_PER_TRANSPORT = coarse.TARGETS_PER_TRANSPORT
WORKER_ROW_RANGES = coarse.WORKER_ROW_RANGES


class GeneratorAnchorLocalizationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GeneratorAnchorLocalizationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeneratorAnchorLocalizationError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def cell_name(cell: tuple[int, int]) -> str:
    return coarse.cell_name(cell)


def all_orientations() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return coarse.all_orientations()


def worker_row_range(worker_id: int) -> tuple[int, int]:
    return coarse.worker_row_range(worker_id)


def _relative_error(observed: float, expected: float) -> float:
    return coarse._relative_error(observed, expected)


def authenticate_repo(
    *,
    expected_head: str | None,
    allow_implementation_worktree: bool,
) -> str:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH:{branch}")
    if expected_head is not None:
        require(head == expected_head, f"HEAD:{head}:EXPECTED:{expected_head}")

    for ancestor, label in (
        (DESIGN_COMMIT, "DESIGN"),
        (VALIDATED_EVIDENCE_COMMIT, "VALIDATED_EVIDENCE"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    frozen_blobs = {
        DESIGN_PATH: DESIGN_BLOB,
        VALIDATED_EVIDENCE_REPORT_PATH: VALIDATED_EVIDENCE_REPORT_BLOB,
        VALIDATED_COARSE_SCRIPT_PATH: VALIDATED_COARSE_SCRIPT_BLOB,
        **FROZEN_DEPENDENCY_BLOBS,
    }
    for path, expected_blob in frozen_blobs.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_BLOB_DRIFT:{path}:{observed}",
        )

    for name, digest in VALIDATED_COARSE_FILES.items():
        path = VALIDATED_COARSE_ROOT / name
        require(path.is_file(), f"VALIDATED_COARSE_ARTIFACT_MISSING:{name}")
        require(
            sha256_file(path) == digest,
            f"VALIDATED_COARSE_ARTIFACT_SHA:{name}",
        )

    dirty = transport.status_paths()
    if allow_implementation_worktree:
        require(
            dirty <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(dirty)}",
        )
    else:
        require(not dirty, f"WORKTREE_NOT_CLEAN:{sorted(dirty)}")
    return head


def _load_validated_coarse_reference(
) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = VALIDATED_COARSE_ROOT / "orientation_serialization_region_pair_metrics.jsonl"
    require(
        sha256_file(path) == VALIDATED_COARSE_FILES[path.name],
        "VALIDATED_COARSE_METRICS_SHA",
    )
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["group"]), str(row["source"]), str(row["target"]))
        require(key not in rows, f"VALIDATED_COARSE_DUPLICATE:{key}")
        rows[key] = row
    require(len(rows) == 36, f"VALIDATED_COARSE_REFERENCE_COUNT:{len(rows)}")
    return rows


def _load_validated_coarse_summary() -> dict[str, Any]:
    path = VALIDATED_COARSE_ROOT / "serialization_region_pair_summary.json"
    require(
        sha256_file(path) == VALIDATED_COARSE_FILES[path.name],
        "VALIDATED_COARSE_SUMMARY_SHA",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value["status"]
        == "PASS_VALIDATED_RECURRENT_SERIALIZATION_REGION_PAIR_LOCALIZATION",
        "VALIDATED_COARSE_STATUS",
    )
    return value


def validate_static_contract() -> None:
    coarse.validate_static_contract()
    require(FACTOR_SEEDS == (6201, 6202, 6203), "FACTOR_SEEDS")
    require(len(all_orientations()) == 36, "ORIENTATION_COUNT")
    require(worker_row_range(0) == (0, 416), "WORKER0_RANGE")
    require(worker_row_range(1) == (416, 840), "WORKER1_RANGE")
    require(DEV_ROWS == p3a.DEV_ROWS == 840, "DEV_ROWS")
    require(VALID_TOKEN_COUNT == transport.VALID_TOKEN_COUNT == 60094, "VALID_TOKEN_COUNT")
    require(BATCH_ROWS == 32, "BATCH_ROWS")
    require(TARGETS_PER_TRANSPORT == 2, "TARGETS_PER_TRANSPORT")
    require(p3a.MAX_LENGTH == 128, "MAX_LENGTH")
    require(p3a.CLAIM_BUDGET == 63, "CLAIM_BUDGET")
    require(p3a.EVIDENCE_BUDGET == 64, "EVIDENCE_BUDGET")
    require(p3a.EOS_TOKEN_ID == 0, "EOS_TOKEN_ID")
    require(p3a.PAD_TOKEN_ID == 0, "PAD_TOKEN_ID")
    require(ATOMIC_ANCHORS == ("A_TITLE", "A_NAME", "A_ROLE", "A_PREDICATE"), "ATOMIC_ANCHORS")
    require(FINE_CLASSES[-1] == "RESIDUAL", "RESIDUAL_CLASS")
    require(len(FINE_PAIR_NAMES) == 25, "FINE_PAIR_COUNT")
    require(len(FINE_CELL_NAMES) == 75, "FINE_CELL_COUNT")
    require(
        PARENT_PAIR_NAMES
        == ("CLAIM->CLAIM", "EVIDENCE->EVIDENCE", "CLAIM->EVIDENCE"),
        "PARENT_PAIRS",
    )
    require(
        transport.DEV_ORDER_SHA256
        == "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25",
        "DEV_ORDER_SHA",
    )
    require(
        transport.DEV_ENCODING_SHA256
        == "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51",
        "DEV_ENCODING_SHA",
    )
    require(tuple(p3static.BASE_CELLS) == tuple(xg1.CELL_IDS), "BASE_CELL_IDENTITY")
    require("C6_EXPLICIT_DENIAL" in p3static.TRAINING_CELLS, "C6_MISSING")
    _load_validated_coarse_reference()
    _load_validated_coarse_summary()


def _c6_statement_and_spans(
    fact: Mapping[str, Any],
) -> tuple[str, dict[str, tuple[int, int]]]:
    time = str(fact["time"])
    location = str(fact["location"])
    title = str(fact["title"])
    name = str(fact["name"])
    role = str(fact["role"])
    predicate = str(fact["predicate"])
    obj = str(fact["object"])

    prefix = f"During {time}, records from {location} identify "
    title_start = len(prefix)
    title_end = title_start + len(title)
    name_start = title_end + 1
    name_end = name_start + len(name)
    role_start = name_end + len(" as ")
    role_end = role_start + len(role)
    predicate_start = role_end + len(
        "; the record explicitly denies that this person "
    )
    predicate_end = predicate_start + len(predicate)

    rendered = (
        f"During {time}, records from {location} identify "
        f"{title} {name} as {role}; "
        f"the record explicitly denies that this person "
        f"{predicate} {obj}."
    )
    require(
        rendered == p3static.explicit_denial_evidence(fact),
        "C6_RENDERER_IDENTITY",
    )
    spans = {
        "A_TITLE": (title_start, title_end),
        "A_NAME": (name_start, name_end),
        "A_ROLE": (role_start, role_end),
        "A_PREDICATE": (predicate_start, predicate_end),
    }
    for anchor_name, (start, stop) in spans.items():
        require(0 <= start < stop <= len(rendered), f"C6_SPAN:{anchor_name}")
    return rendered, spans


def _base_statement_and_spans(
    fact: Mapping[str, Any],
    *,
    cell_id: str | None,
) -> tuple[str, dict[str, tuple[int, int]]]:
    overrides: dict[str, str] = {}
    if cell_id is not None:
        _mask, substitutions = xg1.cell_spec(cell_id)
        overrides = {
            axis: str(fact[source])
            for axis, source in substitutions
        }
    rendered, all_spans = anchor.realized_statement_and_spans(fact, overrides)
    spans = {
        name: tuple(all_spans[name])
        for name in ATOMIC_ANCHORS
    }
    return rendered, spans


def _encode_with_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(value) for value in encoded.ids]
    offsets = [(int(a), int(b)) for a, b in encoded.offsets]
    require(len(ids) == len(offsets), "TOKEN_ID_OFFSET_LENGTH")
    require(bool(ids), "EMPTY_TOKENIZATION")
    return ids, offsets


def _assign_token_classes(
    offsets: Sequence[tuple[int, int]],
    spans: Mapping[str, tuple[int, int]],
    *,
    kept_count: int,
) -> list[str]:
    require(0 <= kept_count <= len(offsets), "KEPT_COUNT")
    require(set(spans) == set(ATOMIC_ANCHORS), "ANCHOR_SPAN_KEYS")
    output: list[str] = []
    for token_index, (token_start, token_stop) in enumerate(offsets[:kept_count]):
        overlaps = []
        if token_stop > token_start:
            for anchor_name in ATOMIC_ANCHORS:
                span_start, span_stop = spans[anchor_name]
                if token_stop > span_start and token_start < span_stop:
                    overlaps.append(anchor_name)
        require(
            len(overlaps) <= 1,
            f"MULTI_ANCHOR_TOKEN:{token_index}:{overlaps}",
        )
        output.append(overlaps[0] if overlaps else "RESIDUAL")
    require(len(output) == kept_count, "TOKEN_CLASS_COUNT")
    return output


def _expected_side_ids(
    bundle: Mapping[str, Any],
    row_index: int,
    side: str,
) -> list[int]:
    inputs = bundle["model_inputs"]
    if side == "CLAIM":
        mask = inputs["claim_mask"][row_index]
    elif side == "EVIDENCE":
        mask = inputs["evidence_mask"][row_index]
    else:
        raise GeneratorAnchorLocalizationError(f"UNKNOWN_SIDE:{side}")
    return [
        int(value)
        for value in inputs["input_ids"][row_index][mask].tolist()
    ]


def _build_anchor_partition(
    *,
    static: Mapping[str, Any],
    encoded: Mapping[str, Any],
    tokenizer_snapshot: Path | None,
    row_start: int,
    row_stop: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    require(0 <= row_start < row_stop <= DEV_ROWS, "ANCHOR_ROW_RANGE")
    dev_rows = static["dev_rows"]
    bundle = encoded["dev_bundle"]
    require(len(dev_rows) == DEV_ROWS, "ANCHOR_DEV_ROWS")
    require(len(bundle["row_ids"]) == DEV_ROWS, "ANCHOR_BUNDLE_ROWS")

    tokenizer, tokenizer_provenance = anchor.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    facts, _base_rows = p3static.build_population(
        p3static.TRAIN_FIRST,
        p3static.TRAIN_LAST,
    )
    facts_by_id = {str(fact["pair_id"]): fact for fact in facts}
    require(len(facts_by_id) == p3static.TRAIN_PAIR_COUNT, "FACT_MAP_COUNT")

    row_count = row_stop - row_start
    masks = {
        side: {
            class_name: torch.zeros(
                (row_count, p3a.MAX_LENGTH),
                dtype=torch.bool,
            )
            for class_name in FINE_CLASSES
        }
        for side in ("CLAIM", "EVIDENCE")
    }

    claim_truncated = 0
    evidence_truncated = 0
    anchor_token_counts = {
        side: {name: 0 for name in FINE_CLASSES}
        for side in ("CLAIM", "EVIDENCE")
    }

    inputs = bundle["model_inputs"]

    for local_index, global_index in enumerate(range(row_start, row_stop)):
        row = dev_rows[global_index]
        require(
            str(bundle["row_ids"][global_index]) == str(row["row_id"]),
            f"ANCHOR_ROW_ID:{global_index}",
        )
        pair_id = str(row["source_pair_id"])
        require(pair_id in facts_by_id, f"ANCHOR_FACT_MISSING:{pair_id}")
        fact = facts_by_id[pair_id]
        cell_id = str(row["contrast_cell_id"])

        claim_text, claim_spans = _base_statement_and_spans(
            fact,
            cell_id=None,
        )
        require(
            claim_text == str(row["claim"]),
            f"CLAIM_RENDER_IDENTITY:{row['row_id']}",
        )

        if cell_id in p3static.BASE_CELLS:
            evidence_text, evidence_spans = _base_statement_and_spans(
                fact,
                cell_id=cell_id,
            )
        elif cell_id == "C6_EXPLICIT_DENIAL":
            evidence_text, evidence_spans = _c6_statement_and_spans(fact)
        else:
            raise GeneratorAnchorLocalizationError(
                f"UNAUTHORIZED_CELL:{cell_id}"
            )
        require(
            evidence_text == str(row["evidence"]),
            f"EVIDENCE_RENDER_IDENTITY:{row['row_id']}",
        )

        claim_ids, claim_offsets = _encode_with_offsets(tokenizer, claim_text)
        evidence_ids, evidence_offsets = _encode_with_offsets(
            tokenizer,
            evidence_text,
        )
        claim_kept = claim_ids[:p3a.CLAIM_BUDGET]
        evidence_kept = evidence_ids[:p3a.EVIDENCE_BUDGET]
        claim_truncated += int(len(claim_ids) > p3a.CLAIM_BUDGET)
        evidence_truncated += int(len(evidence_ids) > p3a.EVIDENCE_BUDGET)

        require(
            claim_kept == _expected_side_ids(bundle, global_index, "CLAIM"),
            f"CLAIM_ACTIVE_ENCODING_REPLAY:{row['row_id']}",
        )
        require(
            evidence_kept == _expected_side_ids(bundle, global_index, "EVIDENCE"),
            f"EVIDENCE_ACTIVE_ENCODING_REPLAY:{row['row_id']}",
        )

        claim_classes = _assign_token_classes(
            claim_offsets,
            claim_spans,
            kept_count=len(claim_kept),
        )
        evidence_classes = _assign_token_classes(
            evidence_offsets,
            evidence_spans,
            kept_count=len(evidence_kept),
        )

        for token_index, class_name in enumerate(claim_classes):
            masks["CLAIM"][class_name][local_index, token_index] = True
            anchor_token_counts["CLAIM"][class_name] += 1

        evidence_start = len(claim_kept) + 1
        for token_index, class_name in enumerate(evidence_classes):
            absolute_index = evidence_start + token_index
            require(absolute_index < p3a.MAX_LENGTH, "EVIDENCE_ABSOLUTE_INDEX")
            masks["EVIDENCE"][class_name][local_index, absolute_index] = True
            anchor_token_counts["EVIDENCE"][class_name] += 1

        claim_union = torch.zeros(p3a.MAX_LENGTH, dtype=torch.bool)
        evidence_union = torch.zeros(p3a.MAX_LENGTH, dtype=torch.bool)
        for class_name in FINE_CLASSES:
            claim_union |= masks["CLAIM"][class_name][local_index]
            evidence_union |= masks["EVIDENCE"][class_name][local_index]

        require(
            torch.equal(claim_union, inputs["claim_mask"][global_index].cpu()),
            f"CLAIM_PARTITION_RECON:{row['row_id']}",
        )
        require(
            torch.equal(evidence_union, inputs["evidence_mask"][global_index].cpu()),
            f"EVIDENCE_PARTITION_RECON:{row['row_id']}",
        )
        require(
            not bool(torch.any(claim_union & evidence_union).item()),
            f"SIDE_PARTITION_OVERLAP:{row['row_id']}",
        )

        eos_index = len(claim_kept)
        require(bool(inputs["attention_mask"][global_index, eos_index]), "EOS_ACTIVE")
        require(
            not bool(claim_union[eos_index] or evidence_union[eos_index]),
            f"EOS_FINE_ASSIGNMENT:{row['row_id']}",
        )

    provenance = {
        "row_start": row_start,
        "row_stop": row_stop,
        "row_count": row_count,
        "tokenizer_provenance": tokenizer_provenance,
        "claim_truncation_row_count": claim_truncated,
        "evidence_truncation_row_count": evidence_truncated,
        "anchor_token_counts": anchor_token_counts,
        "active_encoding_replay": True,
        "decoded_token_strings_inspected": False,
    }
    return masks, provenance


def _batch_anchor_masks(
    partition: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    local_start: int,
    local_stop: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        side: {
            class_name: partition[side][class_name][local_start:local_stop].to(
                device=device
            )
            for class_name in FINE_CLASSES
        }
        for side in ("CLAIM", "EVIDENCE")
    }


def _fine_pair_opportunity_counts(
    fine_masks: Mapping[str, Mapping[str, torch.Tensor]],
    coarse_counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    sample = fine_masks["CLAIM"]["A_TITLE"]
    batch, seq_len = sample.shape
    require(seq_len == p3a.MAX_LENGTH, f"FINE_COUNT_SEQ:{seq_len}")

    positions = torch.arange(seq_len, device=sample.device)
    gap_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
    strict = gap_matrix > 0
    gaps = gap_matrix.unsqueeze(0).expand(batch, -1, -1)

    output: dict[str, dict[str, dict[str, Any]]] = {}
    for source_side, target_side in PARENT_PAIRS:
        parent = f"{source_side}->{target_side}"
        parent_out: dict[str, dict[str, Any]] = {}
        reconstructed = [0 for _ in range(seq_len - 1)]

        for source_class in FINE_CLASSES:
            source = fine_masks[source_side][source_class]
            for target_class in FINE_CLASSES:
                target = fine_masks[target_side][target_class]
                pair_mask = (
                    source[:, :, None]
                    & target[:, None, :]
                    & strict.unsqueeze(0)
                )
                counts = torch.bincount(
                    gaps[pair_mask].to(torch.int64),
                    minlength=seq_len,
                )
                by_gap = [
                    int(value)
                    for value in counts[1:seq_len].detach().cpu().tolist()
                ]
                fine_name = f"{source_class}->{target_class}"
                parent_out[fine_name] = {
                    "by_gap": by_gap,
                    "windows": {
                        key: int(value)
                        for key, value in coarse._window_sums(by_gap).items()
                    },
                    "all": int(sum(by_gap)),
                }
                for index, value in enumerate(by_gap):
                    reconstructed[index] += value

        expected = [int(x) for x in coarse_counts[parent]["by_gap"]]
        require(
            reconstructed == expected,
            f"FINE_COUNT_PARENT_RECON:{parent}",
        )
        output[parent] = parent_out

    return output


def _class_stack(
    *,
    base_active: torch.Tensor,
    side_masks: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    parts = []
    for class_name in FINE_CLASSES:
        mask = side_masks[class_name].to(
            device=base_active.device,
            dtype=torch.bool,
        )
        parts.append(
            torch.where(
                mask[None, :, :, None],
                base_active,
                torch.zeros_like(base_active),
            )
        )
    return torch.stack(parts, dim=0)


def _fast_class_ffts(
    *,
    class_stack: torch.Tensor,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
    survival_chunk: torch.Tensor,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = class_stack * x_scale[None, None, :, :, :]
    target = (
        class_stack
        * survival_chunk[None, None, :, :, :]
        * y_scale[None, None, :, :, :]
    )
    source_fft = torch.fft.rfft(source, n=n_fft, dim=3)
    target_fft = torch.fft.rfft(target, n=n_fft, dim=3)
    del source, target
    return source_fft, target_fft


def _spectral_pair_matrix(
    source_fft: torch.Tensor,
    target_fft: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    require(source_fft.shape == target_fft.shape, "SPECTRAL_CLASS_SHAPE")
    require(source_fft.ndim == 5, "SPECTRAL_CLASS_RANK")
    class_count, k_count, _batch, _freq, _width = source_fft.shape
    require(class_count == len(FINE_CLASSES), "SPECTRAL_CLASS_COUNT")

    # Avoid materializing 25 time-domain correlation tensors. Promote FFT
    # coefficients before the batch/channel contraction: casting the reduced
    # spectrum afterward cannot recover complex64 summation roundoff.
    source_high = torch.conj(source_fft).to(dtype=torch.complex128)
    target_high = target_fft.to(dtype=torch.complex128)
    spectrum = torch.einsum(
        "akbfw,ckbfw->ackf",
        source_high,
        target_high,
    )
    del source_high, target_high
    correlation = torch.fft.irfft(
        spectrum,
        n=1 << ((2 * seq_len - 1).bit_length()),
        dim=-1,
    )
    return 2.0 * correlation[..., 1:seq_len]


def _fine_pair_gap_interference_fft(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    fine_masks: Mapping[str, Mapping[str, torch.Tensor]],
    shape: Any,
) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    """Return signed C(d) for 75 fine cells; gap index zero is reserved."""
    require(component.ndim == 4, "FINE_COMPONENT_RANK")
    k_count, batch, seq_len, width = component.shape
    require(seq_len == p3a.MAX_LENGTH, f"FINE_SEQ_LEN:{seq_len}")
    require(width == shape.intermediate_size * shape.state_size, "FINE_WIDTH")
    require(
        tuple(log_a.shape)
        == (batch, seq_len, shape.intermediate_size, shape.state_size),
        "FINE_LOG_A_SHAPE",
    )
    require(
        tuple(attention_mask.shape) == (batch, seq_len),
        "FINE_ATTENTION_SHAPE",
    )
    require(
        tuple(survival.shape) == (batch, seq_len, width),
        "FINE_SURVIVAL_SHAPE",
    )

    for side in ("CLAIM", "EVIDENCE"):
        require(side in fine_masks, f"FINE_SIDE_MISSING:{side}")
        for class_name in FINE_CLASSES:
            require(
                tuple(fine_masks[side][class_name].shape) == (batch, seq_len),
                f"FINE_MASK_SHAPE:{side}:{class_name}",
            )

    w = component.reshape(k_count, batch, seq_len, width)
    log_a_flat = log_a.reshape(batch, seq_len, width)
    target_mask = attention_mask.to(device=component.device, dtype=torch.bool)
    source_active = (
        torch.flip(
            torch.cumsum(
                torch.flip(target_mask.to(dtype=torch.int64), dims=(1,)),
                dim=1,
            ),
            dims=(1,),
        )
        > 0
    )
    require(
        bool(torch.all(torch.any(source_active, dim=1)).item()),
        "FINE_EMPTY_VALID_SEQUENCE",
    )

    gap = {
        parent: {
            fine: torch.zeros(
                (k_count, seq_len),
                device=component.device,
                dtype=torch.float64,
            )
            for fine in FINE_PAIR_NAMES
        }
        for parent in PARENT_PAIR_NAMES
    }

    n_fft = 1 << ((2 * seq_len - 1).bit_length())
    max_half_span = 0.0

    for start in range(0, width, FFT_CHUNK_WIDTH):
        stop = min(start + FFT_CHUNK_WIDTH, width)
        p = torch.cumsum(
            log_a_flat[:, :, start:stop].to(dtype=torch.float64),
            dim=1,
        )
        active = source_active[:, :, None]
        p_max = torch.amax(
            torch.where(active, p, torch.full_like(p, -torch.inf)),
            dim=1,
        )
        p_min = torch.amin(
            torch.where(active, p, torch.full_like(p, torch.inf)),
            dim=1,
        )
        require(
            bool(torch.all(torch.isfinite(p_max) & torch.isfinite(p_min)).item()),
            "FINE_ACTIVE_PREFIX_NONFINITE",
        )
        half_span = 0.5 * (p_max - p_min)
        chunk_max = float(torch.amax(half_span).item())
        max_half_span = max(max_half_span, chunk_max)

        w_native = w[:, :, :, start:stop].to(dtype=torch.float32)
        base_active = torch.where(
            source_active[None, :, :, None],
            w_native,
            torch.zeros_like(w_native),
        )
        survival_chunk = survival[:, :, start:stop].to(dtype=torch.float32)

        claim_stack = _class_stack(
            base_active=base_active,
            side_masks=fine_masks["CLAIM"],
        )
        evidence_stack = _class_stack(
            base_active=base_active,
            side_masks=fine_masks["EVIDENCE"],
        )

        if chunk_max <= FAST_FLOAT32_HALF_SPAN_LIMIT:
            center = 0.5 * (
                p_max.to(dtype=torch.float32)
                + p_min.to(dtype=torch.float32)
            )
            p_calc = p.to(dtype=torch.float32)
            p_safe = torch.where(active, p_calc, center[:, None, :])
            x_scale = torch.exp(-p_safe + center[:, None, :])
            y_scale = torch.exp(p_safe - center[:, None, :])
            require(bool(torch.all(torch.isfinite(x_scale)).item()), "FINE_X_SCALE")
            require(bool(torch.all(torch.isfinite(y_scale)).item()), "FINE_Y_SCALE")

            claim_source_fft, claim_target_fft = _fast_class_ffts(
                class_stack=claim_stack,
                x_scale=x_scale,
                y_scale=y_scale,
                survival_chunk=survival_chunk,
                n_fft=n_fft,
            )
            evidence_source_fft, evidence_target_fft = _fast_class_ffts(
                class_stack=evidence_stack,
                x_scale=x_scale,
                y_scale=y_scale,
                survival_chunk=survival_chunk,
                n_fft=n_fft,
            )

            parent_spectra = {
                "CLAIM->CLAIM": _spectral_pair_matrix(
                    claim_source_fft,
                    claim_target_fft,
                    seq_len=seq_len,
                ),
                "EVIDENCE->EVIDENCE": _spectral_pair_matrix(
                    evidence_source_fft,
                    evidence_target_fft,
                    seq_len=seq_len,
                ),
                "CLAIM->EVIDENCE": _spectral_pair_matrix(
                    claim_source_fft,
                    evidence_target_fft,
                    seq_len=seq_len,
                ),
            }
            for parent, values in parent_spectra.items():
                for left_index, left in enumerate(FINE_CLASSES):
                    for right_index, right in enumerate(FINE_CLASSES):
                        fine = f"{left}->{right}"
                        gap[parent][fine][:, 1:] += values[
                            left_index,
                            right_index,
                        ]
            del (
                claim_source_fft,
                claim_target_fft,
                evidence_source_fft,
                evidence_target_fft,
                parent_spectra,
            )
        else:
            # Rare large-log-span fallback. Pair batches are deliberately
            # bounded to five so memory never scales with all 75 fine cells.
            for source_side, target_side in PARENT_PAIRS:
                parent = f"{source_side}->{target_side}"
                source_stack = claim_stack if source_side == "CLAIM" else evidence_stack
                target_stack = claim_stack if target_side == "CLAIM" else evidence_stack

                specs = [
                    (left_index, right_index, f"{left}->{right}")
                    for left_index, left in enumerate(FINE_CLASSES)
                    for right_index, right in enumerate(FINE_CLASSES)
                ]
                for pair_start in range(0, len(specs), 5):
                    subset = specs[pair_start:pair_start + 5]
                    source_parts = torch.cat(
                        [source_stack[left_index] for left_index, _, _ in subset],
                        dim=0,
                    )
                    target_parts = torch.cat(
                        [target_stack[right_index] for _, right_index, _ in subset],
                        dim=0,
                    )
                    observed = coarse._cross_strict_dyadic_fft(
                        source_chunk=source_parts,
                        target_chunk=target_parts,
                        p_chunk=p,
                        survival_chunk=survival_chunk,
                        seq_len=seq_len,
                    )
                    reshaped = observed.reshape(len(subset), k_count, seq_len)
                    for subset_index, (_left, _right, fine) in enumerate(subset):
                        gap[parent][fine] += reshaped[subset_index]
                    del source_parts, target_parts, observed, reshaped

        del claim_stack, evidence_stack, base_active

    for parent in PARENT_PAIR_NAMES:
        for fine in FINE_PAIR_NAMES:
            value = gap[parent][fine]
            require(bool(torch.all(torch.isfinite(value)).item()), f"FINE_NONFINITE:{parent}:{fine}")
            require(bool(torch.all(value[:, 0] == 0.0).item()), f"FINE_ZERO_INDEX:{parent}:{fine}")

    return gap, max_half_span


def _component_fine_pair_stats(
    *,
    component: torch.Tensor,
    log_a: torch.Tensor,
    survival: torch.Tensor,
    attention_mask: torch.Tensor,
    role_masks: Mapping[str, torch.Tensor],
    fine_masks: Mapping[str, Mapping[str, torch.Tensor]],
    shape: Any,
) -> list[dict[str, Any]]:
    coarse_rows = coarse._component_region_pair_stats(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        role_masks=role_masks,
        shape=shape,
    )
    fine_gap, fine_span = _fine_pair_gap_interference_fft(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention_mask,
        fine_masks=fine_masks,
        shape=shape,
    )
    k_count = component.shape[0]
    require(len(coarse_rows) == k_count, "FINE_COARSE_ROW_COUNT")

    rows = []
    for index in range(k_count):
        fine_out: dict[str, dict[str, list[float]]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for parent in PARENT_PAIR_NAMES:
            reconstructed = torch.zeros(
                p3a.MAX_LENGTH,
                device=component.device,
                dtype=torch.float64,
            )
            parent_out: dict[str, list[float]] = {}
            for fine in FINE_PAIR_NAMES:
                vector = fine_gap[parent][fine][index]
                reconstructed += vector
                parent_out[fine] = [
                    float(value)
                    for value in vector[1:].detach().cpu().tolist()
                ]

            expected = torch.tensor(
                [0.0] + [
                    float(x)
                    for x in coarse_rows[index]["region_pair_C_by_gap"][parent]
                ],
                device=component.device,
                dtype=torch.float64,
            )
            self_energy = float(coarse_rows[index]["S"])
            require(self_energy > 0.0, f"FINE_PARENT_SELF_ENERGY:{index}")
            reconstructed_h = [
                float(value) / self_energy
                for value in reconstructed[1:].detach().cpu().tolist()
            ]
            expected_h = [
                float(value) / self_energy
                for value in expected[1:].detach().cpu().tolist()
            ]
            replay = coarse._replay_vector(
                name=f"FINE_PARENT_H_RECON_{parent}_{index}",
                observed=reconstructed_h,
                expected=expected_h,
            )
            max_abs = max(max_abs, float(replay["max_abs_error"]))
            max_rel = max(max_rel, float(replay["max_rel_error"]))
            fine_out[parent] = parent_out

        row = dict(coarse_rows[index])
        row["fine_pair_C_by_gap"] = fine_out
        row["fine_parent_reconstruction_abs_max"] = max_abs
        row["fine_parent_reconstruction_rel_max"] = max_rel
        row["fine_pair_fft_half_log_span_max"] = fine_span
        rows.append(row)
    return rows


def _empty_orientation(
    group: str,
    source: tuple[int, int],
    target: tuple[int, int],
) -> dict[str, Any]:
    return {
        "group": group,
        "source": cell_name(source),
        "target": cell_name(target),
        "example_count": 0,
        "raw_reconstruction_max_abs": 0.0,
        "R_visible": 0.0,
        "R_complement": 0.0,
        "S_visible": 0.0,
        "S_complement": 0.0,
        "E_visible": 0.0,
        "E_complement": 0.0,
        "fine_pair_C_visible": {
            parent: {fine: [] for fine in FINE_PAIR_NAMES}
            for parent in PARENT_PAIR_NAMES
        },
        "fine_pair_C_complement": {
            parent: {fine: [] for fine in FINE_PAIR_NAMES}
            for parent in PARENT_PAIR_NAMES
        },
        "fine_parent_reconstruction_abs_max": 0.0,
        "fine_parent_reconstruction_rel_max": 0.0,
        "fine_pair_fft_half_log_span_max": 0.0,
    }


def _accumulate_component_row(
    accumulator: dict[str, Any],
    *,
    prefix: str,
    stats: Mapping[str, Any],
) -> None:
    require(prefix in {"visible", "complement"}, f"COMPONENT_PREFIX:{prefix}")
    for energy in ("R", "S", "E"):
        accumulator[f"{energy}_{prefix}"] += float(stats[energy])

    dst = accumulator[f"fine_pair_C_{prefix}"]
    observed = stats["fine_pair_C_by_gap"]
    for parent in PARENT_PAIR_NAMES:
        for fine in FINE_PAIR_NAMES:
            vector = [float(x) for x in observed[parent][fine]]
            if not dst[parent][fine]:
                dst[parent][fine] = [0.0 for _ in vector]
            require(
                len(dst[parent][fine]) == len(vector),
                f"FINE_ACCUM_LEN:{prefix}:{parent}:{fine}",
            )
            for index, value in enumerate(vector):
                dst[parent][fine][index] += value

    for dst_name, src_name in (
        ("fine_parent_reconstruction_abs_max", "fine_parent_reconstruction_abs_max"),
        ("fine_parent_reconstruction_rel_max", "fine_parent_reconstruction_rel_max"),
        ("fine_pair_fft_half_log_span_max", "fine_pair_fft_half_log_span_max"),
    ):
        accumulator[dst_name] = max(
            float(accumulator[dst_name]),
            float(stats[src_name]),
        )


def _merge_pair_counts(
    workers: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    merged = {
        parent: {
            fine: {"by_gap": [0 for _ in range(p3a.MAX_LENGTH - 1)]}
            for fine in FINE_PAIR_NAMES
        }
        for parent in PARENT_PAIR_NAMES
    }
    for worker in workers:
        observed = worker["pair_opportunity_counts"]
        require(set(observed) == set(PARENT_PAIR_NAMES), "WORKER_PARENT_COUNT_KEYS")
        for parent in PARENT_PAIR_NAMES:
            require(set(observed[parent]) == set(FINE_PAIR_NAMES), f"WORKER_FINE_KEYS:{parent}")
            for fine in FINE_PAIR_NAMES:
                vector = [int(x) for x in observed[parent][fine]["by_gap"]]
                require(len(vector) == p3a.MAX_LENGTH - 1, f"WORKER_FINE_COUNT_LEN:{parent}:{fine}")
                for index, value in enumerate(vector):
                    merged[parent][fine]["by_gap"][index] += value

    coarse_summary = _load_validated_coarse_summary()
    coarse_counts = coarse_summary["pair_opportunity_counts"]

    for parent in PARENT_PAIR_NAMES:
        reconstructed = [0 for _ in range(p3a.MAX_LENGTH - 1)]
        for fine in FINE_PAIR_NAMES:
            vector = merged[parent][fine]["by_gap"]
            merged[parent][fine]["windows"] = {
                key: int(value)
                for key, value in coarse._window_sums(vector).items()
            }
            merged[parent][fine]["all"] = int(sum(vector))
            for index, value in enumerate(vector):
                reconstructed[index] += value
        require(
            reconstructed == [int(x) for x in coarse_counts[parent]["by_gap"]],
            f"MERGED_FINE_COUNT_PARENT_RECON:{parent}",
        )
    return merged


def _fine_component_output(
    *,
    fine_c: Mapping[str, Mapping[str, Sequence[float]]],
    self_energy: float,
    pair_counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for parent in PARENT_PAIR_NAMES:
        parent_out: dict[str, Any] = {}
        for fine in FINE_PAIR_NAMES:
            c_vector = [float(x) for x in fine_c[parent][fine]]
            h_vector = [value / self_energy for value in c_vector]
            h_windows = coarse._window_sums(h_vector)
            counts = {
                key: int(value)
                for key, value in pair_counts[parent][fine]["windows"].items()
            }
            mean_h = {}
            for window, h_value in h_windows.items():
                count = counts[window]
                mean_h[window] = h_value / count if count > 0 else None
            parent_out[fine] = {
                "H_by_gap": h_vector,
                "H_windows": h_windows,
                "pair_counts": counts,
                "mean_h_windows": mean_h,
            }
        output[parent] = parent_out
    return output


def _finalize_orientation(
    accumulator: Mapping[str, Any],
    reference: Mapping[str, Any],
    pair_counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    require(int(accumulator["example_count"]) == DEV_ROWS, "FINAL_EXAMPLE_COUNT")
    values = {
        key: float(accumulator[key])
        for key in (
            "R_visible",
            "R_complement",
            "S_visible",
            "S_complement",
            "E_visible",
            "E_complement",
        )
    }
    require(all(value > 0.0 for value in values.values()), "FINAL_NONPOSITIVE_ENERGY")

    c_visible = values["E_visible"] - values["S_visible"]
    c_complement = values["E_complement"] - values["S_complement"]
    q_visible = values["E_visible"] / values["S_visible"]
    q_complement = values["E_complement"] / values["S_complement"]
    require(q_visible > 0.0 and q_complement > 0.0, "FINAL_NONPOSITIVE_Q")
    l_interference = math.log(q_complement / q_visible)

    fine_visible = _fine_component_output(
        fine_c=accumulator["fine_pair_C_visible"],
        self_energy=values["S_visible"],
        pair_counts=pair_counts,
    )
    fine_complement = _fine_component_output(
        fine_c=accumulator["fine_pair_C_complement"],
        self_energy=values["S_complement"],
        pair_counts=pair_counts,
    )

    scalar_replay = {}
    for key in (
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    ):
        scalar_replay[key] = coarse._replay_scalar(
            name=key,
            observed=values[key],
            expected=float(reference[key]),
        )
    for key, observed in (
        ("C_visible", c_visible),
        ("C_complement", c_complement),
        ("Q_visible", q_visible),
        ("Q_complement", q_complement),
    ):
        scalar_replay[key] = coarse._replay_scalar(
            name=key,
            observed=observed,
            expected=float(reference[key]),
        )
    scalar_replay["L_interference"] = coarse._replay_scalar(
        name="L_interference",
        observed=l_interference,
        expected=float(reference["L_interference"]),
        log_metric=True,
    )

    parent_replay: dict[str, Any] = {"visible": {}, "complement": {}}
    for prefix, observed_output in (
        ("visible", fine_visible),
        ("complement", fine_complement),
    ):
        for parent in PARENT_PAIR_NAMES:
            reconstructed = [
                sum(
                    float(observed_output[parent][fine]["H_by_gap"][index])
                    for fine in FINE_PAIR_NAMES
                )
                for index in range(p3a.MAX_LENGTH - 1)
            ]
            expected = [
                float(x)
                for x in reference[f"region_pair_{prefix}"][parent]["H_by_gap"]
            ]
            parent_replay[prefix][parent] = coarse._replay_vector(
                name=f"FINE_PARENT_{prefix}_{parent}",
                observed=reconstructed,
                expected=expected,
            )

    return {
        "group": str(accumulator["group"]),
        "source": str(accumulator["source"]),
        "target": str(accumulator["target"]),
        "example_count": int(accumulator["example_count"]),
        "raw_reconstruction_max_abs": float(accumulator["raw_reconstruction_max_abs"]),
        **values,
        "C_visible": c_visible,
        "C_complement": c_complement,
        "Q_visible": q_visible,
        "Q_complement": q_complement,
        "log_Q_visible": math.log(q_visible),
        "log_Q_complement": math.log(q_complement),
        "L_interference": l_interference,
        "fine_pair_visible": fine_visible,
        "fine_pair_complement": fine_complement,
        "batch_fine_parent_reconstruction_abs_max": float(
            accumulator["fine_parent_reconstruction_abs_max"]
        ),
        "batch_fine_parent_reconstruction_rel_max": float(
            accumulator["fine_parent_reconstruction_rel_max"]
        ),
        "fine_pair_fft_half_log_span_max": float(
            accumulator["fine_pair_fft_half_log_span_max"]
        ),
        "validated_serialization_region_replay": {
            "scalars": scalar_replay,
            "parent_vectors": parent_replay,
        },
    }


def _merge_worker_accumulators(
    workers: Sequence[Mapping[str, Any]],
    pair_counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    scalar_sum = (
        "example_count",
        "R_visible",
        "R_complement",
        "S_visible",
        "S_complement",
        "E_visible",
        "E_complement",
    )
    scalar_max = (
        "raw_reconstruction_max_abs",
        "fine_parent_reconstruction_abs_max",
        "fine_parent_reconstruction_rel_max",
        "fine_pair_fft_half_log_span_max",
    )
    fine_maps = ("fine_pair_C_visible", "fine_pair_C_complement")

    for worker in workers:
        for row in worker["orientation_accumulators"]:
            key = (str(row["group"]), str(row["source"]), str(row["target"]))
            if key not in merged:
                merged[key] = {
                    "group": key[0],
                    "source": key[1],
                    "target": key[2],
                    **{name: 0 for name in scalar_sum},
                    **{name: 0.0 for name in scalar_max},
                    **{
                        map_name: {
                            parent: {fine: [] for fine in FINE_PAIR_NAMES}
                            for parent in PARENT_PAIR_NAMES
                        }
                        for map_name in fine_maps
                    },
                }
            dst = merged[key]
            for name in scalar_sum:
                dst[name] += row[name]
            for name in scalar_max:
                dst[name] = max(float(dst[name]), float(row[name]))

            for map_name in fine_maps:
                for parent in PARENT_PAIR_NAMES:
                    for fine in FINE_PAIR_NAMES:
                        observed = [float(x) for x in row[map_name][parent][fine]]
                        target = dst[map_name][parent][fine]
                        if not target:
                            target.extend([0.0 for _ in observed])
                        require(
                            len(target) == len(observed),
                            f"MERGE_FINE_LEN:{map_name}:{parent}:{fine}:{key}",
                        )
                        for index, value in enumerate(observed):
                            target[index] += value

    require(len(merged) == 36, f"MERGED_ACCUMULATOR_COUNT:{len(merged)}")
    reference = _load_validated_coarse_reference()
    output = []
    for key in sorted(merged):
        require(key in reference, f"VALIDATED_COARSE_REFERENCE_MISSING:{key}")
        output.append(
            _finalize_orientation(
                merged[key],
                reference[key],
                pair_counts,
            )
        )
    return output


def _mean(values: Sequence[float]) -> float:
    require(bool(values), "MEAN_EMPTY")
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float:
    require(bool(values), "MEDIAN_EMPTY")
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _grouped_fine_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ("PRIMARY_A", "CONTROL_R"):
        subset = [row for row in rows if row["group"] == group]
        require(len(subset) == 18, f"GROUPED_COUNT:{group}")
        group_out: dict[str, Any] = {}
        for component in ("visible", "complement"):
            comp_out: dict[str, Any] = {}
            for parent in PARENT_PAIR_NAMES:
                parent_out: dict[str, Any] = {}
                for fine in FINE_PAIR_NAMES:
                    fine_out: dict[str, Any] = {}
                    for window in WINDOWS:
                        values = [
                            float(
                                row[f"fine_pair_{component}"][parent][fine]
                                ["H_windows"][window]
                            )
                            for row in subset
                        ]
                        fine_out[window] = {
                            "count": len(values),
                            "median": _median(values),
                            "minimum": min(values),
                            "maximum": max(values),
                        }
                    parent_out[fine] = fine_out
                comp_out[parent] = parent_out
            group_out[component] = comp_out
        output[group] = group_out
    return output


def _source_matched(
    rows: Sequence[Mapping[str, Any]],
    pair_counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)
    require(len(by_source) == 9, "SOURCE_MATCHED_CELL_COUNT")

    output = []
    for source, source_rows in sorted(by_source.items()):
        primary = [row for row in source_rows if row["group"] == "PRIMARY_A"]
        control = [row for row in source_rows if row["group"] == "CONTROL_R"]
        require(len(primary) == 2 and len(control) == 2, f"SOURCE_MATCH_COUNTS:{source}")

        scalars = {}
        for name in ("log_Q_visible", "log_Q_complement", "L_interference"):
            p = _mean([float(row[name]) for row in primary])
            c = _mean([float(row[name]) for row in control])
            scalars[name] = {
                "primary_mean": p,
                "control_mean": c,
                "primary_minus_control": p - c,
            }

        fine_metrics: dict[str, Any] = {}
        for component in ("visible", "complement"):
            comp_out: dict[str, Any] = {}
            for parent in PARENT_PAIR_NAMES:
                parent_out: dict[str, Any] = {}
                for fine in FINE_PAIR_NAMES:
                    fine_out = {}
                    for window in WINDOWS:
                        p = _mean([
                            float(
                                row[f"fine_pair_{component}"][parent][fine]
                                ["H_windows"][window]
                            )
                            for row in primary
                        ])
                        c = _mean([
                            float(
                                row[f"fine_pair_{component}"][parent][fine]
                                ["H_windows"][window]
                            )
                            for row in control
                        ])
                        delta = p - c
                        count = int(pair_counts[parent][fine]["windows"][window])
                        fine_out[window] = {
                            "primary_mean": p,
                            "control_mean": c,
                            "primary_minus_control": delta,
                            "pair_opportunity_count": count,
                            "primary_minus_control_mean_h":
                                (delta / count if count > 0 else None),
                        }
                    parent_out[fine] = fine_out
                comp_out[parent] = parent_out
            fine_metrics[component] = comp_out

        output.append({
            "source": source,
            "metrics": scalars,
            "fine_pair_metrics": fine_metrics,
        })
    return output


def _worker_payload_path(scratch_root: Path, worker_id: int) -> Path:
    return scratch_root / f"worker{worker_id}" / "worker_result.json"


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _worker_sha_sidecar_bytes(digest: str) -> bytes:
    return (digest + "\n").encode("ascii")


def _validate_runtime_head(args: argparse.Namespace) -> str:
    require(bool(args.expected_head), "EXPECTED_HEAD_REQUIRED")
    require(bool(args.implementation_freeze_commit), "IMPLEMENTATION_FREEZE_REQUIRED")
    require(
        args.expected_head == args.implementation_freeze_commit,
        "EXECUTION_HEAD_MUST_EQUAL_IMPLEMENTATION_FREEZE",
    )
    return authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=False,
    )


def _expected_counts() -> dict[str, Any]:
    workers: dict[str, Any] = {}
    groups_per_source = math.ceil(4 / TARGETS_PER_TRANSPORT)
    for worker_id in (0, 1):
        start, stop = worker_row_range(worker_id)
        batches = math.ceil((stop - start) / BATCH_ROWS)
        groups = 9 * groups_per_source * batches
        workers[str(worker_id)] = {
            "row_start": start,
            "row_stop": stop,
            "row_count": stop - start,
            "batches": batches,
            "source_gradient_forwards": 9 * batches,
            "coarse_group_decompositions": groups,
            "fine_group_decompositions": groups,
            "fine_cells_per_group": len(FINE_CELL_NAMES),
            "downstream_stage_transport_calls": 0,
            "final_head_transport_calls": 0,
        }
    return {
        "batch_rows": BATCH_ROWS,
        "targets_per_group": TARGETS_PER_TRANSPORT,
        "fine_classes": len(FINE_CLASSES),
        "fine_pairs_per_parent": len(FINE_PAIR_NAMES),
        "parent_pairs": len(PARENT_PAIR_NAMES),
        "fine_cells_total": len(FINE_CELL_NAMES),
        "workers": workers,
        "total_source_gradient_forwards": sum(
            row["source_gradient_forwards"] for row in workers.values()
        ),
        "total_fine_group_decompositions": sum(
            row["fine_group_decompositions"] for row in workers.values()
        ),
        "total_downstream_stage_transport_calls": 0,
        "total_final_head_transport_calls": 0,
    }


def run_static_contract_check(args: argparse.Namespace) -> None:
    head = authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=True,
    )
    validate_static_contract()
    print("GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_STATIC_CONTRACT_PASS")
    print(f"HEAD={head}")
    print("FINE_CLASSES=" + ",".join(FINE_CLASSES))
    print("PARENT_PAIRS=" + ",".join(PARENT_PAIR_NAMES))
    print("FINE_CELLS=75")
    print("DEV_ROWS=840")
    print("VALID_TOKEN_COUNT=60094")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")


def run_preflight(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    transport.legacy._validate_two_t4s()
    static, encoded, _model, _wrapper, _strong_mask, _planes = transport._runtime_inputs(args)

    partitions = []
    for worker_id in (0, 1):
        start, stop = worker_row_range(worker_id)
        _masks, provenance = _build_anchor_partition(
            static=static,
            encoded=encoded,
            tokenizer_snapshot=args.tokenizer_snapshot,
            row_start=start,
            row_stop=stop,
        )
        partitions.append(provenance)

    require(
        sum(int(row["row_count"]) for row in partitions) == DEV_ROWS,
        "PREFLIGHT_ANCHOR_ROW_COVERAGE",
    )
    print("GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_CUDA_PREFLIGHT_PASS")
    print("ANCHOR_ACTIVE_ENCODING_REPLAY=PASS")
    print("ANCHOR_PARTITION_ROWS=840")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("RECURRENCE_ONLY=True")
    print("TRAINING_EXECUTED=False")
    print("EXPECTED_COUNTS=" + json.dumps(_expected_counts(), sort_keys=True))


def run_worker(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    require(torch.cuda.device_count() == 1, "WORKER_VISIBLE_GPU_COUNT_MUST_BE_ONE")

    worker_id = int(args.worker_id)
    scratch_root = Path(args.scratch_root)
    worker_root = scratch_root / f"worker{worker_id}"
    require(not worker_root.exists(), f"WORKER_OUTPUT_COLLISION:{worker_root}")

    static, encoded, model, wrapper, strong_mask, planes = transport._runtime_inputs(args)
    device = torch.device("cuda:0")
    features, _labels, active, targets = p3a._feature_batch_to_device(
        encoded["dev_bundle"],
        device,
    )

    row_start, row_stop = worker_row_range(worker_id)
    anchor_partition, anchor_provenance = _build_anchor_partition(
        static=static,
        encoded=encoded,
        tokenizer_snapshot=args.tokenizer_snapshot,
        row_start=row_start,
        row_stop=row_stop,
    )

    weights = transport._cell_weights()
    local_sources = tuple(FULL_FACTORIAL_CELLS)
    orientations = all_orientations()
    accumulators = {
        (group, source, target): _empty_orientation(group, source, target)
        for group, source, target in orientations
    }
    by_source = {
        source: tuple(row for row in orientations if row[1] == source)
        for source in local_sources
    }
    for source, rows in by_source.items():
        require(len(rows) == 4, f"SOURCE_ORIENTATION_COUNT:{cell_name(source)}")

    row_count = row_stop - row_start
    total_batches = math.ceil(row_count / BATCH_ROWS)
    valid_token_seen = 0
    pair_counts_acc = {
        parent: {
            fine: {"by_gap": [0 for _ in range(p3a.MAX_LENGTH - 1)]}
            for fine in FINE_PAIR_NAMES
        }
        for parent in PARENT_PAIR_NAMES
    }

    for batch_index, start in enumerate(
        range(row_start, row_stop, BATCH_ROWS),
        start=1,
    ):
        stop = min(start + BATCH_ROWS, row_stop)
        batch_features = transport.legacy._phase_b_batch_features(
            features,
            start,
            stop,
        )
        batch_mask = batch_features["attention_mask"]
        valid_token_seen += int(torch.count_nonzero(batch_mask).item())

        role_masks = coarse._serialization_role_masks(batch_features)
        coarse_counts = coarse._pair_opportunity_counts(role_masks)
        local_start = start - row_start
        local_stop = stop - row_start
        fine_masks = _batch_anchor_masks(
            anchor_partition,
            local_start=local_start,
            local_stop=local_stop,
            device=batch_mask.device,
        )
        batch_pair_counts = _fine_pair_opportunity_counts(
            fine_masks,
            coarse_counts,
        )
        for parent in PARENT_PAIR_NAMES:
            for fine in FINE_PAIR_NAMES:
                vector = batch_pair_counts[parent][fine]["by_gap"]
                target = pair_counts_acc[parent][fine]["by_gap"]
                for index, value in enumerate(vector):
                    target[index] += int(value)

        context = transport.legacy._phase_b_prepare_common_context(
            model=model,
            wrapper=wrapper,
            features=batch_features,
            stressor_active=active[start:stop],
            target_indices=targets[start:stop],
            strong_mask=strong_mask,
            planes=planes,
        )
        pair_log_a = kernel._discrete_log_a(
            context,
            shape=wrapper.correction.shape,
        ).to(device=batch_mask.device)
        pair_survival = pairgap._backward_survival_factor(
            log_a_flat=pair_log_a.reshape(
                pair_log_a.shape[0],
                pair_log_a.shape[1],
                -1,
            ),
            attention_mask=batch_mask,
        )

        raw_by_cell: dict[tuple[int, int], torch.Tensor] = {}
        with torch.no_grad():
            for cell in FULL_FACTORIAL_CELLS:
                a_weight, b_weight = weights[cell]
                raw_by_cell[cell] = transport.legacy._phase_b_raw_write(
                    context["mixer_input"],
                    batch_mask,
                    a_weight,
                    b_weight,
                ).detach()

        for source in local_sources:
            source_rows = by_source[source]
            raw_source = raw_by_cell[source]
            _, g_refute, g_support = transport._source_gradient_bundle(
                model=model,
                wrapper=wrapper,
                features=batch_features,
                context=context,
                raw_source=raw_source,
            )
            g_refute, g_support, pinv_cpu = transport._prepare_projector_pinv(
                grad_refute=g_refute,
                grad_support=g_support,
                attention_mask=batch_mask,
            )

            for group_start in range(0, len(source_rows), TARGETS_PER_TRANSPORT):
                group_rows = source_rows[group_start:group_start + TARGETS_PER_TRANSPORT]
                raw_targets = [raw_by_cell[row[2]] for row in group_rows]
                projector_accs = [
                    transport._empty_orientation(*row)
                    for row in group_rows
                ]
                visible, complement = transport._project_visible_group(
                    grad_refute=g_refute,
                    grad_support=g_support,
                    pinv_cpu=pinv_cpu,
                    attention_mask=batch_mask,
                    raw_source=raw_source,
                    raw_targets=raw_targets,
                    accumulators=projector_accs,
                )

                combined = torch.cat((visible, complement), dim=0)
                stats = _component_fine_pair_stats(
                    component=combined,
                    log_a=pair_log_a,
                    survival=pair_survival,
                    attention_mask=batch_mask,
                    role_masks=role_masks,
                    fine_masks=fine_masks,
                    shape=wrapper.correction.shape,
                )
                target_count = len(group_rows)

                for index, row in enumerate(group_rows):
                    accumulator = accumulators[row]
                    visible_stats = stats[index]
                    complement_stats = stats[target_count + index]
                    _accumulate_component_row(
                        accumulator,
                        prefix="visible",
                        stats=visible_stats,
                    )
                    _accumulate_component_row(
                        accumulator,
                        prefix="complement",
                        stats=complement_stats,
                    )

                    projector_acc = projector_accs[index]
                    accumulator["raw_reconstruction_max_abs"] = max(
                        float(accumulator["raw_reconstruction_max_abs"]),
                        float(projector_acc["raw_reconstruction_max_abs"]),
                    )
                    require(
                        float(projector_acc["raw_reconstruction_max_abs"])
                        <= transport.RAW_RECON_ATOL,
                        "RAW_RECONSTRUCTION_AUTH",
                    )

                    raw_visible = float(projector_acc["stage"]["raw_write"]["D_visible"])
                    raw_complement = float(projector_acc["stage"]["raw_write"]["D_complement"])
                    require(
                        _relative_error(float(visible_stats["R"]), raw_visible)
                        <= RAW_COMPONENT_MATCH_RTOL,
                        "VISIBLE_RAW_ENERGY_INTERNAL_REPLAY",
                    )
                    require(
                        _relative_error(float(complement_stats["R"]), raw_complement)
                        <= RAW_COMPONENT_MATCH_RTOL,
                        "COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY",
                    )
                    accumulator["example_count"] += stop - start

                del combined, visible, complement, stats, projector_accs

            del g_refute, g_support, pinv_cpu

        del raw_by_cell, pair_log_a, pair_survival, role_masks, fine_masks
        print(
            "GEN5_RECURRENT_GENERATOR_ANCHOR_PROGRESS "
            f"worker={worker_id} batch={batch_index}/{total_batches} "
            f"shard_rows={stop-row_start}/{row_count}",
            flush=True,
        )

    for parent in PARENT_PAIR_NAMES:
        for fine in FINE_PAIR_NAMES:
            vector = pair_counts_acc[parent][fine]["by_gap"]
            pair_counts_acc[parent][fine]["windows"] = {
                key: int(value)
                for key, value in coarse._window_sums(vector).items()
            }
            pair_counts_acc[parent][fine]["all"] = int(sum(vector))

    for accumulator in accumulators.values():
        require(
            int(accumulator["example_count"]) == row_count,
            "WORKER_ORIENTATION_EXAMPLE_COUNT",
        )
    require(
        not any(parameter.grad is not None for parameter in model.parameters()),
        "PARAMETER_GRADIENT_ACCUMULATED_FINAL",
    )

    result = {
        "schema_version": WORKER_SCHEMA,
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "worker_id": worker_id,
        "row_start": row_start,
        "row_stop": row_stop,
        "row_count": row_count,
        "orientation_count": 36,
        "orientation_accumulators": list(accumulators.values()),
        "pair_opportunity_counts": pair_counts_acc,
        "anchor_partition_provenance": anchor_provenance,
        "valid_token_count": valid_token_seen,
        "execution_engine": "RECURRENT_GENERATOR_ANCHOR_PAIR_EXACT_INTERFERENCE_V1",
        "recurrence_only": True,
        "decoded_token_strings_inspected": False,
        "downstream_stage_transport_calls": 0,
        "final_head_transport_calls": 0,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }

    payload_path = _worker_payload_path(scratch_root, worker_id)
    _atomic_write(payload_path, canonical_json_bytes(result))
    digest = sha256_file(payload_path)
    _atomic_write(
        payload_path.with_suffix(".sha256"),
        _worker_sha_sidecar_bytes(digest),
    )
    print(
        "GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_WORKER_PASS "
        f"worker={worker_id} rows={row_start}:{row_stop} "
        f"orientations=36 sha256={digest}"
    )


def _read_worker(
    scratch_root: Path,
    worker_id: int,
) -> dict[str, Any]:
    path = _worker_payload_path(scratch_root, worker_id)
    sidecar = path.with_suffix(".sha256")
    require(path.is_file(), f"WORKER_RESULT_MISSING:{worker_id}")
    require(sidecar.is_file(), f"WORKER_SHA_MISSING:{worker_id}")
    observed = sha256_file(path)
    expected = sidecar.read_text(encoding="ascii").strip()
    require(observed == expected, f"WORKER_SHA:{worker_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value["schema_version"] == WORKER_SCHEMA, f"WORKER_SCHEMA:{worker_id}")
    require(int(value["worker_id"]) == worker_id, f"WORKER_ID:{worker_id}")
    start, stop = worker_row_range(worker_id)
    require(
        (int(value["row_start"]), int(value["row_stop"])) == (start, stop),
        f"WORKER_RANGE:{worker_id}",
    )
    return value


def run_merge_only(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")

    scratch_root = Path(args.scratch_root)
    output_root = Path(args.output_root)
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")

    workers = [_read_worker(scratch_root, worker_id) for worker_id in (0, 1)]
    for worker in workers:
        require(worker["execution_head"] == args.expected_head, "MIXED_WORKER_HEAD")
    require(
        sum(int(worker["valid_token_count"]) for worker in workers)
        == VALID_TOKEN_COUNT,
        "MERGED_VALID_TOKEN_COUNT",
    )

    pair_counts = _merge_pair_counts(workers)
    rows = _merge_worker_accumulators(workers, pair_counts)
    require(len(rows) == 36, "MERGED_ORIENTATION_COUNT")
    require(
        max(float(row["raw_reconstruction_max_abs"]) for row in rows)
        <= transport.RAW_RECON_ATOL,
        "MERGED_RAW_RECONSTRUCTION_FAIL",
    )

    for row in rows:
        scalar_checks = row["validated_serialization_region_replay"]["scalars"].values()
        vector_groups = row["validated_serialization_region_replay"]["parent_vectors"].values()
        require(
            all(bool(check["pass"]) for check in scalar_checks),
            "VALIDATED_COARSE_SCALAR_REPLAY_FAIL",
        )
        require(
            all(
                bool(check["pass"])
                for group in vector_groups
                for check in group.values()
            ),
            "VALIDATED_COARSE_PARENT_REPLAY_FAIL",
        )

    grouped = _grouped_fine_summary(rows)
    matched = _source_matched(rows, pair_counts)

    max_fine_abs = max(
        float(row["batch_fine_parent_reconstruction_abs_max"])
        for row in rows
    )
    max_fine_rel = max(
        float(row["batch_fine_parent_reconstruction_rel_max"])
        for row in rows
    )
    max_scalar_replay_abs = max(
        float(check["abs_error"])
        for row in rows
        for check in row["validated_serialization_region_replay"]["scalars"].values()
    )
    max_parent_replay_abs = max(
        float(check["max_abs_error"])
        for row in rows
        for group in row["validated_serialization_region_replay"]["parent_vectors"].values()
        for check in group.values()
    )

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_VALIDATED_RECURRENT_GENERATOR_ANCHOR_PAIR_LOCALIZATION",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_coarse_run": VALIDATED_COARSE_RUN,
        "population": "FROZEN_PHASE3A_P0_DEV",
        "dev_rows": DEV_ROWS,
        "valid_token_count": VALID_TOKEN_COUNT,
        "dev_order_sha256": transport.DEV_ORDER_SHA256,
        "dev_encoding_sha256": transport.DEV_ENCODING_SHA256,
        "serialization_contract": "claim[:63]+EOS(0)+evidence[:64]",
        "fine_classes": list(FINE_CLASSES),
        "parent_pairs": list(PARENT_PAIR_NAMES),
        "fine_pair_count_per_parent": len(FINE_PAIR_NAMES),
        "fine_cell_count": len(FINE_CELL_NAMES),
        "windows": {
            key: [start, stop]
            for key, (start, stop) in WINDOWS.items()
        },
        "pair_opportunity_counts": pair_counts,
        "grouped_fine_pair_summary": grouped,
        "source_cell_matched_primary_minus_control": matched,
        "validated_serialization_region_replay": {
            "pass": True,
            "max_scalar_abs_error": max_scalar_replay_abs,
            "max_parent_vector_abs_error": max_parent_replay_abs,
        },
        "fine_parent_reconstruction": {
            "pass": True,
            "max_abs_error": max_fine_abs,
            "max_rel_error": max_fine_rel,
        },
        "pair_count_normalization_role": "DESCRIPTIVE_DENSITY_CONTROL_ONLY",
        "mechanism_classification_thresholds": None,
        "scientific_p_value_count": 0,
        "recurrence_only": True,
        "decoded_token_strings_inspected": False,
        "downstream_stage_transport_calls": 0,
        "final_head_transport_calls": 0,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }

    shard_manifest = {
        "schema_version": SHARD_SCHEMA,
        "execution_head": args.expected_head,
        "workers": {
            str(worker["worker_id"]): {
                "row_start": worker["row_start"],
                "row_stop": worker["row_stop"],
                "row_count": worker["row_count"],
                "orientation_count": worker["orientation_count"],
                "valid_token_count": worker["valid_token_count"],
                "worker_result_sha256": sha256_file(
                    _worker_payload_path(
                        scratch_root,
                        int(worker["worker_id"]),
                    )
                ),
                "anchor_partition_provenance":
                    worker["anchor_partition_provenance"],
            }
            for worker in workers
        },
        "total_orientations": 36,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = output_root / "generator_anchor_pair_summary.json"
    rows_path = output_root / "orientation_generator_anchor_pair_metrics.jsonl"
    shard_path = output_root / "shard_manifest.json"
    provenance_path = output_root / "run_provenance.json"

    summary_path.write_bytes(canonical_json_bytes(summary))
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(row) for row in rows)
    )
    shard_path.write_bytes(canonical_json_bytes(shard_manifest))

    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "design_blob": DESIGN_BLOB,
        "validated_evidence_commit": VALIDATED_EVIDENCE_COMMIT,
        "validated_evidence_report_blob": VALIDATED_EVIDENCE_REPORT_BLOB,
        "validated_coarse_script_blob": VALIDATED_COARSE_SCRIPT_BLOB,
        "frozen_dependency_blobs": FROZEN_DEPENDENCY_BLOBS,
        "validated_coarse_artifact_sha256": VALIDATED_COARSE_FILES,
        "summary_sha256": sha256_file(summary_path),
        "orientation_metrics_sha256": sha256_file(rows_path),
        "shard_manifest_sha256": sha256_file(shard_path),
        "worker_result_sha256": {
            str(worker_id): sha256_file(
                _worker_payload_path(scratch_root, worker_id)
            )
            for worker_id in (0, 1)
        },
        "projector_backend": transport.PROJECTOR_BACKEND,
        "projector_pinv_rtol": transport.PROJECTOR_PINV_RTOL,
        "pair_gap_energy_rtol": PAIR_GAP_ENERGY_RTOL,
        "pair_gap_energy_atol": PAIR_GAP_ENERGY_ATOL,
        "pair_gap_log_atol": PAIR_GAP_LOG_ATOL,
        "fft_chunk_width": FFT_CHUNK_WIDTH,
        "fast_float32_half_span_limit": FAST_FLOAT32_HALF_SPAN_LIMIT,
        "worker_row_ranges": {
            str(worker_id): list(worker_row_range(worker_id))
            for worker_id in (0, 1)
        },
        "dev_order_sha256": transport.DEV_ORDER_SHA256,
        "dev_encoding_sha256": transport.DEV_ENCODING_SHA256,
        "parent_checkpoint_sha256": transport.PARENT_CHECKPOINT_SHA256,
        "decoded_token_strings_inspected": False,
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    print("GEN5_NATIVE_RECURRENT_GENERATOR_ANCHOR_MERGE_PASS")
    print("ORIENTATIONS=36")
    print("FINE_CELLS=75")
    print("VALIDATED_SERIALIZATION_REGION_REPLAY=PASS")
    print("FINE_PARENT_RECONSTRUCTION=PASS")
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={rows_path}")
    print(f"SHARD_MANIFEST={shard_path}")
    print(f"PROVENANCE={provenance_path}")


def _spawn_worker(
    args: argparse.Namespace,
    worker_id: int,
) -> subprocess.Popen:
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-worker",
        "--expected-head",
        str(args.expected_head),
        "--implementation-freeze-commit",
        str(args.implementation_freeze_commit),
        "--model-snapshot",
        str(args.model_snapshot),
        "--tokenizer-snapshot",
        str(args.tokenizer_snapshot),
        "--checkpoint",
        str(args.checkpoint),
        "--scratch-root",
        str(args.scratch_root),
        "--worker-id",
        str(worker_id),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    return subprocess.Popen(command, cwd=ROOT, env=env)


def run_full(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    transport.legacy._validate_two_t4s()
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(args.output_root is not None, "OUTPUT_ROOT_REQUIRED")
    scratch_root = Path(args.scratch_root)
    output_root = Path(args.output_root)
    require(not scratch_root.exists(), f"SCRATCH_COLLISION:{scratch_root}")
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    scratch_root.mkdir(parents=True, exist_ok=False)

    workers = [_spawn_worker(args, worker_id) for worker_id in (0, 1)]
    return_codes = [process.wait() for process in workers]
    require(return_codes == [0, 0], f"WORKER_FAILURE:{return_codes}")
    run_merge_only(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-contract-check", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run-worker", action="store_true")
    modes.add_argument("--merge-only", action="store_true")
    modes.add_argument("--run", action="store_true")

    parser.add_argument("--expected-head")
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.static_contract_check:
        require(args.implementation_freeze_commit is None, "STATIC_IMPLEMENTATION_FREEZE_FORBIDDEN")
        require(args.model_snapshot is None, "STATIC_MODEL_SNAPSHOT_FORBIDDEN")
        require(args.tokenizer_snapshot is None, "STATIC_TOKENIZER_SNAPSHOT_FORBIDDEN")
        require(args.checkpoint is None, "STATIC_CHECKPOINT_FORBIDDEN")
        require(args.scratch_root is None, "STATIC_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_FORBIDDEN")
        require(args.worker_id is None, "STATIC_WORKER_FORBIDDEN")
        return

    require(args.expected_head is not None, "RUNTIME_EXPECTED_HEAD_REQUIRED")
    require(
        args.implementation_freeze_commit is not None,
        "RUNTIME_IMPLEMENTATION_FREEZE_REQUIRED",
    )
    require(args.model_snapshot is not None, "RUNTIME_MODEL_SNAPSHOT_REQUIRED")
    require(args.tokenizer_snapshot is not None, "RUNTIME_TOKENIZER_SNAPSHOT_REQUIRED")
    require(args.checkpoint is not None, "RUNTIME_CHECKPOINT_REQUIRED")

    if args.preflight:
        require(args.scratch_root is None, "PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_FORBIDDEN")
        require(args.worker_id is None, "PREFLIGHT_WORKER_FORBIDDEN")
    elif args.run_worker:
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.output_root is None, "WORKER_OUTPUT_ROOT_FORBIDDEN")
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    elif args.merge_only:
        require(args.scratch_root is not None, "MERGE_SCRATCH_REQUIRED")
        require(args.output_root is not None, "MERGE_OUTPUT_REQUIRED")
        require(args.worker_id is None, "MERGE_WORKER_FORBIDDEN")
    elif args.run:
        require(args.scratch_root is not None, "RUN_SCRATCH_REQUIRED")
        require(args.output_root is not None, "RUN_OUTPUT_REQUIRED")
        require(args.worker_id is None, "RUN_WORKER_FORBIDDEN")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.static_contract_check:
        run_static_contract_check(args)
    elif args.preflight:
        run_preflight(args)
    elif args.run_worker:
        run_worker(args)
    elif args.merge_only:
        run_merge_only(args)
    else:
        run_full(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
