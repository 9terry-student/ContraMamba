#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    build_reason_router_gen4_mamba14b_adjacent_site_specificity_holdout
    as holdout_builder,
)
from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import (
    reason_router_gen4_mamba14b_confirmation_fast_cuda
    as core,
)
from scripts import (
    reason_router_gen4_mamba14b_discovery_fast_cuda
    as discovery,
)
from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda
    as geom,
)
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import (
    reason_router_gen4_xg1_tokenizer_anchor_eligibility
    as tokenizer_gate,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

DESIGN_FREEZE_COMMIT = "3ee2d7600de649a711752b8293687047c3c5ec4e"
STATIC_PREPARATION_COMMIT = "b6c1c7bdb65f1edaeed78044d0b699578552ac5f"
EQUIVALENCE_FREEZE_COMMIT = "4de24510cf94e326472b387104b458f52b1220ea"
ADJACENT_GEOMETRY_FREEZE_COMMIT = "d434dfdc38f222419708ed67db87e4e28fb05a4d"
CANONICAL_GEOMETRY_FREEZE_COMMIT = "f97b597fb4da08a8d360ed07e48c6727e15ae0be"

HOLDOUT_ROOT = holdout_builder.OUTPUT_DIR
HOLDOUT_SOURCE_SHA256 = (
    "dd89f33baebd1d3df5a42eccbbc6e65efa2aa59fcc3d59bdd49cbd23a5d19a23"
)
HOLDOUT_ROWS_SHA256 = (
    "f014f66d16521db3935a1098e51d34552545fb78f6a0821e739035e16627b3a2"
)
HOLDOUT_MANIFEST_SHA256 = (
    "d1033694e4db56008fef222f39d2e4d2e2e1a0c3587e30587704d32b07e3b89b"
)

PAIR_FIRST = 5101
PAIR_LAST = 5400
PAIR_COUNT = 300
ROWS = 1800
PAIR_IDS = tuple(
    f"xg1_fact_{index:04d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
SELECTED_PLANE = "P5"
CANONICAL_CONTROL_PLANE = "P4"
ADJACENT_CONTROL_PLANE = "P4"
K = 5
EPS = 0.025
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2

CONDITION_ORDER = (
    "dominant_restored",
    "dominant_control",
)
DIRECTION_ORDER = tuple(
    [f"xg2_{index}" for index in range(K)]
    + [f"xg4_{index}" for index in range(K)]
)

FORWARDS_PER_SIGNED = 2
FORWARDS_PER_DIRECTION = 4
FORWARDS_PER_CONDITION = len(DIRECTION_ORDER) * FORWARDS_PER_DIRECTION
FORWARDS_PER_PAIR_PER_SITE = len(CONDITION_ORDER) * FORWARDS_PER_CONDITION
FORWARDS_PER_SITE = PAIR_COUNT * FORWARDS_PER_PAIR_PER_SITE
TOTAL_FORWARD_BUDGET = 2 * FORWARDS_PER_SITE

CANONICAL_GEOMETRY_ROOT = discovery.GEOMETRY_ROOT
CANONICAL_DIM = 829
CANONICAL_STRONG_INDEX_SHA256 = (
    "ceaebe6046c747e09a169e8b5c0e59d68c9d82185dc5a9017373f497cc0a96c7"
)
CANONICAL_TRIPLET = (33, 34, 35)
CANONICAL_GEOMETRY_SUMMARY_SHA256 = discovery.GEOMETRY_SUMMARY_SHA256

ADJACENT_GEOMETRY_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_geometry_preparation_runs/"
    "g4k-mamba14b-adjacent-geometry-plus1-xg2xg4-2gpu-4de2451"
)
ADJACENT_DIM = 1205
ADJACENT_STRONG_INDEX_SHA256 = (
    "3c1d39df9b9b7a9acd3200b9b8cb31576bd6cc81fcd4518781f2cbbee3533141"
)
ADJACENT_TRIPLET = (34, 35, 36)
ADJACENT_GEOMETRY_SUMMARY_SHA256 = (
    "168aca69230b82eda29be2800b79b634362ac937335406f79f3e08a60d9f427e"
)
ADJACENT_GEOMETRY_MANIFEST_SHA256 = (
    "9e3f3590d604781f1c54980d454af0c07374feafffeb4843779d4c14461914ff"
)
ADJACENT_GEOMETRY_SUMS_SHA256 = (
    "74c7b90a055456c6c2545707542ecd9a6a279ffc9ece3b9c079155ae8d804637"
)

SITE_CONFIG = {
    "canonical": {
        "site": "canonical",
        "physical_device": 0,
        "triplet": CANONICAL_TRIPLET,
        "dim": CANONICAL_DIM,
        "strong_index_sha256": CANONICAL_STRONG_INDEX_SHA256,
        "control_plane": CANONICAL_CONTROL_PLANE,
        "geometry_summary_sha256": CANONICAL_GEOMETRY_SUMMARY_SHA256,
        "geometry_freeze_commit": CANONICAL_GEOMETRY_FREEZE_COMMIT,
    },
    "adjacent": {
        "site": "adjacent",
        "physical_device": 1,
        "triplet": ADJACENT_TRIPLET,
        "dim": ADJACENT_DIM,
        "strong_index_sha256": ADJACENT_STRONG_INDEX_SHA256,
        "control_plane": ADJACENT_CONTROL_PLANE,
        "geometry_summary_sha256": ADJACENT_GEOMETRY_SUMMARY_SHA256,
        "geometry_freeze_commit": ADJACENT_GEOMETRY_FREEZE_COMMIT,
    },
}

ITEM_FILE = "paired_specificity_items.jsonl"
SUMMARY_FILE = "raw_response_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESULT_PASS = "PASS_MAMBA14B_ONE_SHOT_ADJACENT_SITE_SPECIFICITY_RAW_RESPONSE"
ITEM_SCHEMA = "gen4-mamba14b-one-shot-adjacent-site-specificity-raw-item-v1"
SUMMARY_SCHEMA = "gen4-mamba14b-one-shot-adjacent-site-specificity-raw-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba14b-one-shot-adjacent-site-specificity-raw-manifest-v1"

PLANE_TOL = 1.0e-9


class SpecificityRawError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SpecificityRawError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        output.append(value)
    return output


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SpecificityRawError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        git("rev-parse", "HEAD") == expected_head,
        "HEAD_MISMATCH",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN"),
        (STATIC_PREPARATION_COMMIT, "STATIC_PREPARATION"),
        (EQUIVALENCE_FREEZE_COMMIT, "EQUIVALENCE"),
        (ADJACENT_GEOMETRY_FREEZE_COMMIT, "ADJACENT_GEOMETRY"),
        (CANONICAL_GEOMETRY_FREEZE_COMMIT, "CANONICAL_GEOMETRY"),
    ):
        require(
            git_rc(
                "merge-base",
                "--is-ancestor",
                ancestor,
                expected_head,
            )
            == 0,
            f"{label}_NOT_ANCESTOR",
        )


def validate_protocol() -> None:
    require(PAIR_FIRST == 5101 and PAIR_LAST == 5400, "PAIR_RANGE")
    require(PAIR_COUNT == 300 and ROWS == 1800, "COHORT_SIZE")
    require(SELECTED_PLANE == "P5", "SELECTED_PLANE")
    require(CANONICAL_CONTROL_PLANE == "P4", "CANONICAL_CONTROL")
    require(ADJACENT_CONTROL_PLANE == "P4", "ADJACENT_CONTROL")
    require(EPS == 0.025, "EPSILON")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(CANONICAL_TRIPLET == (33, 34, 35), "CANONICAL_TRIPLET")
    require(ADJACENT_TRIPLET == (34, 35, 36), "ADJACENT_TRIPLET")
    require(CANONICAL_DIM == 829, "CANONICAL_DIM")
    require(ADJACENT_DIM == 1205, "ADJACENT_DIM")
    require(FORWARDS_PER_CONDITION == 40, "FORWARDS_PER_CONDITION")
    require(FORWARDS_PER_PAIR_PER_SITE == 80, "FORWARDS_PER_PAIR_PER_SITE")
    require(FORWARDS_PER_SITE == 24000, "FORWARDS_PER_SITE")
    require(TOTAL_FORWARD_BUDGET == 48000, "TOTAL_FORWARD_BUDGET")
    require(
        tuple(SITE_CONFIG) == ("canonical", "adjacent"),
        "SITE_ORDER",
    )
    require(
        SITE_CONFIG["canonical"]["physical_device"] == 0
        and SITE_CONFIG["adjacent"]["physical_device"] == 1,
        "SITE_DEVICE_ASSIGNMENT",
    )


def load_fresh_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_path = ROOT / HOLDOUT_ROOT / holdout_builder.SOURCE_FILE
    rows_path = ROOT / HOLDOUT_ROOT / holdout_builder.ROW_FILE
    manifest_path = ROOT / HOLDOUT_ROOT / holdout_builder.MANIFEST_FILE

    require(source_path.is_file(), "HOLDOUT_SOURCE_MISSING")
    require(rows_path.is_file(), "HOLDOUT_ROWS_MISSING")
    require(manifest_path.is_file(), "HOLDOUT_MANIFEST_MISSING")

    require(
        sha256_file(source_path) == HOLDOUT_SOURCE_SHA256,
        "HOLDOUT_SOURCE_SHA",
    )
    require(
        sha256_file(rows_path) == HOLDOUT_ROWS_SHA256,
        "HOLDOUT_ROWS_SHA",
    )
    require(
        sha256_file(manifest_path) == HOLDOUT_MANIFEST_SHA256,
        "HOLDOUT_MANIFEST_SHA",
    )

    facts = read_jsonl(source_path)
    rows = read_jsonl(rows_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    require(len(facts) == PAIR_COUNT, "HOLDOUT_FACT_COUNT")
    require(len(rows) == ROWS, "HOLDOUT_ROW_COUNT")
    require(
        [str(value["pair_id"]) for value in facts]
        == list(PAIR_IDS),
        "HOLDOUT_PAIR_ORDER",
    )

    observed_pairs: list[str] = []
    seen: set[str] = set()
    per_pair: dict[str, int] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        per_pair[pair] = per_pair.get(pair, 0) + 1
        if pair not in seen:
            seen.add(pair)
            observed_pairs.append(pair)

    require(observed_pairs == list(PAIR_IDS), "HOLDOUT_ROW_PAIR_ORDER")
    require(
        all(per_pair.get(pair) == 6 for pair in PAIR_IDS),
        "HOLDOUT_ROWS_PER_PAIR",
    )

    require(
        manifest["schema_version"]
        == "GEN4_MAMBA14B_XG1_ADJACENT_SITE_SPECIFICITY_STRUCTURAL_V1",
        "HOLDOUT_SCHEMA",
    )
    require(
        manifest["result"]
        == "PASS_MAMBA14B_XG1_ADJACENT_SITE_SPECIFICITY_5101_5400_STRUCTURAL",
        "HOLDOUT_RESULT",
    )
    require(manifest["role"] == "one_shot_adjacent_site_specificity", "HOLDOUT_ROLE")
    require(manifest["canonical_triplet"] == [33, 34, 35], "HOLDOUT_CANONICAL")
    require(manifest["adjacent_triplet"] == [34, 35, 36], "HOLDOUT_ADJACENT")
    require(manifest["selected_causal_candidate_frozen"] == "P5", "HOLDOUT_P5")
    require(manifest["epsilon"] == EPS, "HOLDOUT_EPS")
    require(manifest["anchor_name"] == ANCHOR_NAME, "HOLDOUT_ANCHOR")
    require(manifest["target_offset"] == TARGET_OFFSET, "HOLDOUT_TARGET_OFFSET")
    require(manifest["response_based_control_selection_allowed"] is False, "HOLDOUT_CONTROL")
    require(manifest["selection_allowed"] is False, "HOLDOUT_SELECTION")
    require(manifest["second_adjacent_site_allowed"] is False, "HOLDOUT_SECOND_SITE")
    require(manifest["epsilon_sweep_allowed"] is False, "HOLDOUT_EPS_SWEEP")
    require(manifest["layer_sweep_allowed"] is False, "HOLDOUT_LAYER_SWEEP")
    require(manifest["token_sweep_allowed"] is False, "HOLDOUT_TOKEN_SWEEP")
    require(manifest["response_fields_present"] is False, "HOLDOUT_RESPONSE_FIELDS")
    require(manifest["endpoint_values_present"] is False, "HOLDOUT_ENDPOINT_FIELDS")
    return facts, rows


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, rows = load_fresh_population()
    tokenizer, _provenance = geom.load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(rows, tokenizer)

    require(
        tuple(encoded["input_ids"].shape) == (ROWS, 128),
        "ENCODED_INPUT_SHAPE",
    )
    require(
        list(encoded["source_pair_id"])
        == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(len(facts_by_id) == PAIR_COUNT, "FACT_LOOKUP_COUNT")

    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        require(pair in facts_by_id, f"MISSING_FACT:{pair}")

        analyzed = tokenizer_gate.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair],
            tokenizer,
        )
        for event in analyzed:
            key = (
                pair,
                cell,
                str(event["anchor_name"]),
            )
            require(key not in events, f"ANCHOR_DUPLICATE:{key}")
            require(
                bool(event["post4_eligible"]),
                f"ANCHOR_INELIGIBLE:{key}",
            )
            events[key] = dict(event)

    required_cells = (
        core.TARGET_PLUS_CELL,
        core.TARGET_MINUS_CELL,
        core.REFERENCE_PLUS_CELL,
        core.REFERENCE_MINUS_CELL,
    )
    for pair in PAIR_IDS:
        for cell in required_cells:
            key = (pair, cell, ANCHOR_NAME)
            require(key in events, f"MISSING_REQUIRED_ANCHOR:{key}")

    return rows, encoded, events


def row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW:{key}")
        out[key] = index
    require(len(out) == ROWS, "ROW_INDEX_COUNT")
    return out


def _load_f64(
    path: Path,
    shape: tuple[int, ...],
    expected_sha: str,
) -> torch.Tensor:
    require(path.is_file(), f"GEOMETRY_FILE_MISSING:{path.name}")
    require(
        sha256_file(path) == expected_sha,
        f"GEOMETRY_FILE_SHA:{path.name}",
    )
    array = np.fromfile(path, dtype="<f8")
    require(
        int(array.size) == math.prod(shape),
        f"GEOMETRY_FILE_SIZE:{path.name}:{array.size}",
    )
    value = torch.from_numpy(
        array.copy().reshape(shape)
    ).to(torch.float64).contiguous()
    require(
        bool(torch.isfinite(value).all().item()),
        f"GEOMETRY_NONFINITE:{path.name}",
    )
    return value


def load_adjacent_geometry() -> dict[str, Any]:
    root = ROOT / ADJACENT_GEOMETRY_ROOT
    summary_path = root / "geometry_summary.json"
    manifest_path = root / "artifact_manifest.json"
    strong_path = root / "strong_indices.json"
    sums_path = root / "SHA256SUMS.txt"

    require(
        sha256_file(summary_path) == ADJACENT_GEOMETRY_SUMMARY_SHA256,
        "ADJACENT_SUMMARY_SHA",
    )
    require(
        sha256_file(manifest_path) == ADJACENT_GEOMETRY_MANIFEST_SHA256,
        "ADJACENT_MANIFEST_SHA",
    )
    require(
        sha256_file(sums_path) == ADJACENT_GEOMETRY_SUMS_SHA256,
        "ADJACENT_SUMS_SHA",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    strong = json.loads(strong_path.read_text(encoding="utf-8-sig"))

    require(
        summary["result"]
        == "PASS_MAMBA14B_ADJACENT_GEOMETRY_PREPARATION",
        "ADJACENT_RESULT",
    )
    require(summary["execution_head"] == EQUIVALENCE_FREEZE_COMMIT, "ADJACENT_EXECUTION_HEAD")
    require(summary["adjacent_triplet"] == [34, 35, 36], "ADJACENT_TRIPLET")
    require(summary["fixed_causal_candidate"] == SELECTED_PLANE, "ADJACENT_SELECTED")
    require(
        summary["response_blind_control_plane"] == ADJACENT_CONTROL_PLANE,
        "ADJACENT_CONTROL",
    )
    require(
        summary["control_selection_uses_response"] is False,
        "ADJACENT_CONTROL_RESPONSE",
    )
    require(summary["xg1_model_forward_count"] == 0, "ADJACENT_XG1_FORWARD")
    require(summary["causal_response_observed"] is False, "ADJACENT_RESPONSE")
    require(summary["statistical_testing_performed"] is False, "ADJACENT_STATS")

    require(
        manifest["result"]
        == "PASS_MAMBA14B_ADJACENT_GEOMETRY_PREPARATION",
        "ADJACENT_MANIFEST_RESULT",
    )
    require(manifest["execution_head"] == EQUIVALENCE_FREEZE_COMMIT, "ADJACENT_MANIFEST_HEAD")
    require(manifest["fixed_causal_candidate"] == SELECTED_PLANE, "ADJACENT_MANIFEST_SELECTED")
    require(manifest["response_blind_control_plane"] == ADJACENT_CONTROL_PLANE, "ADJACENT_MANIFEST_CONTROL")
    require(manifest["xg1_accessed"] is False, "ADJACENT_MANIFEST_XG1")
    require(manifest["response_observed"] is False, "ADJACENT_MANIFEST_RESPONSE")

    declared: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in declared, f"ADJACENT_SUMS_DUPLICATE:{name}")
        declared[name] = digest
        require(
            sha256_file(root / name) == digest,
            f"ADJACENT_SUMS_FILE:{name}",
        )

    require(
        declared["artifact_manifest.json"]
        == ADJACENT_GEOMETRY_MANIFEST_SHA256,
        "ADJACENT_MANIFEST_SUMS_SHA",
    )

    for name, digest in manifest["output_file_sha256"].items():
        require(
            declared.get(name) == digest,
            f"ADJACENT_MANIFEST_FILE_SHA:{name}",
        )

    indices = [int(value) for value in strong["strong_indices"]]
    require(len(indices) == ADJACENT_DIM, "ADJACENT_STRONG_COUNT")
    require(indices == sorted(indices), "ADJACENT_STRONG_ORDER")
    require(len(set(indices)) == ADJACENT_DIM, "ADJACENT_STRONG_DUPLICATE")
    require(
        geom.strong_index_sha256(indices)
        == ADJACENT_STRONG_INDEX_SHA256,
        "ADJACENT_STRONG_HASH",
    )

    mask = torch.zeros(
        geom.INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    mask[torch.tensor(indices, dtype=torch.long)] = True
    require(int(mask.sum().item()) == ADJACENT_DIM, "ADJACENT_STRONG_MASK")

    bases = {
        "xg2": _load_f64(
            root / "xg2_basis.f64le",
            (ADJACENT_DIM, K),
            manifest["output_file_sha256"]["xg2_basis.f64le"],
        ),
        "xg4": _load_f64(
            root / "xg4_basis.f64le",
            (ADJACENT_DIM, K),
            manifest["output_file_sha256"]["xg4_basis.f64le"],
        ),
    }

    planes: dict[str, dict[str, torch.Tensor]] = {}
    vectors: list[torch.Tensor] = []
    for plane in PLANE_ORDER:
        number = plane[1:]
        plus = _load_f64(
            root / f"p{number}_plus.f64le",
            (ADJACENT_DIM,),
            manifest["output_file_sha256"][f"p{number}_plus.f64le"],
        )
        minus = _load_f64(
            root / f"p{number}_minus.f64le",
            (ADJACENT_DIM,),
            manifest["output_file_sha256"][f"p{number}_minus.f64le"],
        )
        require(
            abs(float(torch.linalg.vector_norm(plus).item()) - 1.0)
            <= PLANE_TOL,
            f"ADJACENT_PLANE_PLUS_NORM:{plane}",
        )
        require(
            abs(float(torch.linalg.vector_norm(minus).item()) - 1.0)
            <= PLANE_TOL,
            f"ADJACENT_PLANE_MINUS_NORM:{plane}",
        )
        require(
            abs(float(torch.dot(plus, minus).item()))
            <= PLANE_TOL,
            f"ADJACENT_PLANE_ORTHOGONALITY:{plane}",
        )
        planes[plane] = {
            "plus": plus,
            "minus": minus,
        }
        vectors.extend([plus, minus])

    matrix = torch.stack(vectors, dim=1)
    gram_residual = float(
        torch.max(
            torch.abs(
                matrix.T @ matrix
                - torch.eye(10, dtype=torch.float64)
            )
        ).item()
    )
    require(
        gram_residual <= PLANE_TOL,
        f"ADJACENT_FULL_PLANE_GRAM:{gram_residual}",
    )

    lambda_plus = {
        plane: float(value)
        for plane, value in zip(
            PLANE_ORDER,
            summary["lambda_plus_by_plane"],
            strict=True,
        )
    }
    require(
        max(
            (
                plane
                for plane in PLANE_ORDER
                if plane != SELECTED_PLANE
            ),
            key=lambda plane: lambda_plus[plane],
        )
        == ADJACENT_CONTROL_PLANE,
        "ADJACENT_CONTROL_RECONSTRUCTION",
    )

    return {
        "summary": summary,
        "strong_indices": indices,
        "strong_mask": mask,
        "bases": bases,
        "planes": planes,
        "lambda_plus": lambda_plus,
        "full_plane_gram_max_abs_residual": gram_residual,
    }


def load_site_geometry(site: str) -> dict[str, Any]:
    require(site in SITE_CONFIG, f"SITE:{site}")
    if site == "canonical":
        frozen = discovery.load_frozen_geometry()
        require(len(frozen["strong_indices"]) == CANONICAL_DIM, "CANONICAL_DIM")
        require(
            geom.strong_index_sha256(frozen["strong_indices"])
            == CANONICAL_STRONG_INDEX_SHA256,
            "CANONICAL_STRONG_HASH",
        )
        require(
            frozen["summary"]["execution_head"] == "c758d5e81b0ad18f9993846789b28efdad53e3e9",
            "CANONICAL_EXECUTION_HEAD",
        )
        require(
            max(
                (
                    plane
                    for plane in PLANE_ORDER
                    if plane != SELECTED_PLANE
                ),
                key=lambda plane: frozen["lambda_plus"][plane],
            )
            == CANONICAL_CONTROL_PLANE,
            "CANONICAL_CONTROL_RECONSTRUCTION",
        )
        return frozen
    return load_adjacent_geometry()


def configure_site(site: str) -> Mapping[str, Any]:
    require(site in SITE_CONFIG, f"SITE:{site}")
    cfg = SITE_CONFIG[site]
    source, target, intervention = cfg["triplet"]

    geom.SOURCE_BLOCK = int(source)
    geom.TARGET_RESIDUAL_LAYER = int(target)
    geom.INTERVENTION_LAYER = int(intervention)
    geom.LOCAL_LAYER_OFFSETS = (-2, -1, 0)
    geom.TARGET_OFFSET = TARGET_OFFSET

    core.DIM = int(cfg["dim"])
    core.PAIR_IDS = PAIR_IDS
    core.PAIR_COUNT = PAIR_COUNT
    core.ROWS = ROWS
    core.SELECTED_PLANE = SELECTED_PLANE
    core.CONTROL_PLANE = str(cfg["control_plane"])
    core.CONDITION_ORDER = CONDITION_ORDER
    core.EPS = EPS

    require(core.FORWARDS_PER_CONDITION == FORWARDS_PER_CONDITION, "CORE_CONDITION_BUDGET")
    require(core.FORWARDS_PER_PAIR == FORWARDS_PER_PAIR_PER_SITE, "CORE_PAIR_BUDGET")
    return cfg


def validate_runtime_geometry(
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> None:
    mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    observed = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten().tolist()

    require(
        observed == frozen["strong_indices"],
        f"RUNTIME_STRONG_INDICES:{cfg['site']}",
    )
    require(
        len(observed) == int(cfg["dim"]),
        f"RUNTIME_STRONG_DIM:{cfg['site']}",
    )
    require(
        geom.strong_index_sha256(observed)
        == str(cfg["strong_index_sha256"]),
        f"RUNTIME_STRONG_HASH:{cfg['site']}",
    )


def _worker_paths(temp_dir: Path, site: str) -> dict[str, Path]:
    return {
        "items": temp_dir / f"{site}_items.jsonl",
        "meta": temp_dir / f"{site}_meta.json",
        "error": temp_dir / f"{site}.error.txt",
    }


def site_worker(
    *,
    site: str,
    expected_head: str,
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), site)
    try:
        validate_protocol()
        authenticate_repo(expected_head)
        cfg = configure_site(site)

        physical_device = int(cfg["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        device = core.runtime_gate_single_visible_gpu(physical_device)

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        frozen = load_site_geometry(site)
        rows, encoded, events = build_input_state(snapshot)
        lookup = row_index(rows)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        runtime_ctx = geom.runtime_components(model)
        validate_runtime_geometry(runtime_ctx, frozen, cfg)

        budget = core.ForwardBudget(FORWARDS_PER_SITE)
        items: list[dict[str, Any]] = []

        for pair_index, pair in enumerate(PAIR_IDS):
            raw = core.run_pair(
                pair_index=pair_index,
                pair=pair,
                bases=frozen["bases"],
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                planes=frozen["planes"],
                encoded=encoded,
                lookup=lookup,
                events=events,
                device=device,
                budget=budget,
            )
            items.append({
                "schema_version":
                    "gen4-mamba14b-specificity-site-response-item-v1",
                "site": site,
                "triplet": list(cfg["triplet"]),
                "strong_dim": int(cfg["dim"]),
                "source_pair_id": pair,
                "pair_index": pair_index,
                "epsilon": EPS,
                "selected_causal_candidate": SELECTED_PLANE,
                "response_blind_control_plane": str(cfg["control_plane"]),
                "condition_order": list(CONDITION_ORDER),
                "conditions": raw["conditions"],
                "Q_restored": float(raw["Q_restored"]),
                "Q_control": float(raw["Q_control"]),
                "D": float(raw["D_CORE"]),
                "scientific_model_forward_count_this_run":
                    FORWARDS_PER_PAIR_PER_SITE,
                "inferential_test_performed": False,
                "selection_reopened": False,
                "response_based_control_selection": False,
                "rescue_performed": False,
            })

        budget.assert_exact()
        torch.cuda.synchronize(device)

        require(len(items) == PAIR_COUNT, f"WORKER_ITEM_COUNT:{site}")
        require(
            [item["source_pair_id"] for item in items] == list(PAIR_IDS),
            f"WORKER_PAIR_ORDER:{site}",
        )

        paths["items"].write_bytes(jsonl_bytes(items))
        meta = {
            "schema_version":
                "gen4-mamba14b-specificity-site-response-worker-v1",
            "site": site,
            "physical_device": physical_device,
            "logical_device": 0,
            "cuda_visible_devices": str(physical_device),
            "device_name": torch.cuda.get_device_name(0),
            "triplet": list(cfg["triplet"]),
            "strong_dim": int(cfg["dim"]),
            "strong_index_sha256":
                str(cfg["strong_index_sha256"]),
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "pair_count": PAIR_COUNT,
            "scientific_model_forward_count": FORWARDS_PER_SITE,
            "items_sha256": sha256_file(paths["items"]),
            "model_provenance": model_provenance,
            "geometry_summary_sha256":
                str(cfg["geometry_summary_sha256"]),
            "geometry_freeze_commit":
                str(cfg["geometry_freeze_commit"]),
            "selected_causal_candidate": SELECTED_PLANE,
            "response_blind_control_plane":
                str(cfg["control_plane"]),
            "response_based_control_selection": False,
            "inferential_test_performed": False,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def merge_site_outputs(
    temp_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    site_items: dict[str, list[dict[str, Any]]] = {}
    site_meta: dict[str, dict[str, Any]] = {}

    for site in ("canonical", "adjacent"):
        paths = _worker_paths(temp_dir, site)
        require(paths["items"].is_file(), f"WORKER_ITEMS_MISSING:{site}")
        require(paths["meta"].is_file(), f"WORKER_META_MISSING:{site}")

        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        items = read_jsonl(paths["items"])
        cfg = SITE_CONFIG[site]

        require(meta["site"] == site, f"MERGE_SITE:{site}")
        require(
            meta["physical_device"] == cfg["physical_device"],
            f"MERGE_DEVICE:{site}",
        )
        require(meta["logical_device"] == 0, f"MERGE_LOGICAL_DEVICE:{site}")
        require(meta["triplet"] == list(cfg["triplet"]), f"MERGE_TRIPLET:{site}")
        require(meta["strong_dim"] == cfg["dim"], f"MERGE_DIM:{site}")
        require(
            meta["strong_index_sha256"]
            == cfg["strong_index_sha256"],
            f"MERGE_STRONG_SHA:{site}",
        )
        require(meta["pair_count"] == PAIR_COUNT, f"MERGE_PAIR_COUNT:{site}")
        require(
            meta["scientific_model_forward_count"] == FORWARDS_PER_SITE,
            f"MERGE_FORWARD_COUNT:{site}",
        )
        require(
            meta["items_sha256"] == sha256_file(paths["items"]),
            f"MERGE_ITEMS_SHA:{site}",
        )
        require(len(items) == PAIR_COUNT, f"MERGE_ITEMS_COUNT:{site}")
        require(
            [item["source_pair_id"] for item in items] == list(PAIR_IDS),
            f"MERGE_PAIR_ORDER:{site}",
        )

        site_items[site] = items
        site_meta[site] = meta

    paired: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(PAIR_IDS):
        canonical = site_items["canonical"][pair_index]
        adjacent = site_items["adjacent"][pair_index]

        require(
            canonical["source_pair_id"] == pair
            and adjacent["source_pair_id"] == pair,
            f"PAIR_ALIGNMENT:{pair}",
        )

        d_can = float(canonical["D"])
        d_adj = float(adjacent["D"])
        s_value = d_can - d_adj
        require(
            all(math.isfinite(v) for v in (d_can, d_adj, s_value)),
            f"PAIR_ENDPOINT_NONFINITE:{pair}",
        )

        paired.append({
            "schema_version": ITEM_SCHEMA,
            "source_pair_id": pair,
            "pair_index": pair_index,
            "epsilon": EPS,
            "selected_causal_candidate": SELECTED_PLANE,
            "canonical_response_blind_control_plane":
                CANONICAL_CONTROL_PLANE,
            "adjacent_response_blind_control_plane":
                ADJACENT_CONTROL_PLANE,
            "canonical_triplet": list(CANONICAL_TRIPLET),
            "adjacent_triplet": list(ADJACENT_TRIPLET),
            "canonical": canonical,
            "adjacent": adjacent,
            "D_CAN": d_can,
            "D_ADJ": d_adj,
            "S": s_value,
            "endpoint_definition": {
                "D_CAN": "Q_restored,canonical(P5)-Q_control,canonical(P4)",
                "D_ADJ": "Q_restored,adjacent(P5)-Q_control,adjacent(P4)",
                "S": "D_CAN-D_ADJ",
            },
            "scientific_model_forward_count_this_run":
                2 * FORWARDS_PER_PAIR_PER_SITE,
            "inferential_test_performed": False,
            "primary_p_value_computed": False,
            "selection_reopened": False,
            "second_adjacent_site_executed": False,
            "layer_sweep_executed": False,
            "epsilon_sweep_executed": False,
            "token_sweep_executed": False,
            "rescue_performed": False,
        })

    return paired, site_meta


def raw_descriptive_summary(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    require(len(items) == PAIR_COUNT, "DESCRIPTIVE_ITEM_COUNT")
    d_can = [float(item["D_CAN"]) for item in items]
    d_adj = [float(item["D_ADJ"]) for item in items]
    s_values = [float(item["S"]) for item in items]

    require(
        all(
            math.isfinite(value)
            for values in (d_can, d_adj, s_values)
            for value in values
        ),
        "DESCRIPTIVE_NONFINITE",
    )

    return {
        "mean_D_CAN": math.fsum(d_can) / PAIR_COUNT,
        "mean_D_ADJ": math.fsum(d_adj) / PAIR_COUNT,
        "mean_S": math.fsum(s_values) / PAIR_COUNT,
        "fraction_D_CAN_positive":
            sum(value > 0.0 for value in d_can) / PAIR_COUNT,
        "fraction_D_ADJ_positive":
            sum(value > 0.0 for value in d_adj) / PAIR_COUNT,
        "fraction_S_positive":
            sum(value > 0.0 for value in s_values) / PAIR_COUNT,
    }


def write_output_bundle(
    *,
    output_dir: Path,
    expected_head: str,
    items: Sequence[Mapping[str, Any]],
    site_meta: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(len(items) == PAIR_COUNT, "OUTPUT_ITEM_COUNT")
    require(
        [item["source_pair_id"] for item in items] == list(PAIR_IDS),
        "OUTPUT_PAIR_ORDER",
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    item_path = output_dir / ITEM_FILE
    summary_path = output_dir / SUMMARY_FILE
    manifest_path = output_dir / MANIFEST_FILE
    sums_path = output_dir / CHECKSUM_FILE

    item_path.write_bytes(jsonl_bytes(items))

    descriptive = raw_descriptive_summary(items)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "phase": "one_shot_adjacent_site_specificity_raw_response",
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "static_preparation_commit": STATIC_PREPARATION_COMMIT,
        "equivalence_freeze_commit": EQUIVALENCE_FREEZE_COMMIT,
        "canonical_geometry_freeze_commit":
            CANONICAL_GEOMETRY_FREEZE_COMMIT,
        "adjacent_geometry_freeze_commit":
            ADJACENT_GEOMETRY_FREEZE_COMMIT,
        "pair_first": PAIR_IDS[0],
        "pair_last": PAIR_IDS[-1],
        "pair_count": PAIR_COUNT,
        "epsilon": EPS,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "selected_causal_candidate": SELECTED_PLANE,
        "canonical_response_blind_control_plane":
            CANONICAL_CONTROL_PLANE,
        "adjacent_response_blind_control_plane":
            ADJACENT_CONTROL_PLANE,
        "control_selection_uses_xg1_response": False,
        "canonical_triplet": list(CANONICAL_TRIPLET),
        "adjacent_triplet": list(ADJACENT_TRIPLET),
        "canonical_strong_dim": CANONICAL_DIM,
        "adjacent_strong_dim": ADJACENT_DIM,
        "endpoint_definitions": {
            "D_CAN": "Q_restored,canonical(P5)-Q_control,canonical(P4)",
            "D_ADJ": "Q_restored,adjacent(P5)-Q_control,adjacent(P4)",
            "S": "D_CAN-D_ADJ",
        },
        "raw_descriptive_only": descriptive,
        "site_workers": {
            site: dict(meta)
            for site, meta in site_meta.items()
        },
        "forward_accounting": {
            "canonical_model_forward_count": FORWARDS_PER_SITE,
            "adjacent_model_forward_count": FORWARDS_PER_SITE,
            "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
            "per_pair_per_site": FORWARDS_PER_PAIR_PER_SITE,
        },
        "inference": {
            "performed": False,
            "primary_p_value_computed": False,
            "primary_test_reserved_for_static_analyzer": True,
            "planned_test":
                "one-sample Student t-test on S, one-sided greater",
            "planned_alpha": 0.05,
            "planned_primary_p_value_count": 1,
        },
        "scientific_conclusion": None,
        "selection_reopened": False,
        "second_adjacent_site_executed": False,
        "layer_sweep_executed": False,
        "epsilon_sweep_executed": False,
        "token_sweep_executed": False,
        "rescue_performed": False,
    }
    summary_path.write_bytes(pretty_json_bytes(summary))

    payload_hashes = {
        ITEM_FILE: sha256_file(item_path),
        SUMMARY_FILE: sha256_file(summary_path),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "output_file_sha256": dict(sorted(payload_hashes.items())),
        "pair_count": PAIR_COUNT,
        "selected_causal_candidate": SELECTED_PLANE,
        "canonical_control": CANONICAL_CONTROL_PLANE,
        "adjacent_control": ADJACENT_CONTROL_PLANE,
        "canonical_triplet": list(CANONICAL_TRIPLET),
        "adjacent_triplet": list(ADJACENT_TRIPLET),
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "canonical_model_forward_count": FORWARDS_PER_SITE,
        "adjacent_model_forward_count": FORWARDS_PER_SITE,
        "xg1_response_observed": True,
        "inferential_test_performed": False,
        "primary_p_value_computed": False,
        "scientific_conclusion": None,
        "selection_reopened": False,
        "response_based_control_selection": False,
        "second_adjacent_site_executed": False,
        "rescue_performed": False,
    }
    manifest_path.write_bytes(pretty_json_bytes(manifest))
    payload_hashes[MANIFEST_FILE] = sha256_file(manifest_path)

    sums_path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(payload_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    return summary


def run_raw_response(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    geom.validate_snapshot(model_snapshot)

    require(
        compact_checkpoint.resolve()
        == (ROOT / geom.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        geom.sha256_file(compact_checkpoint)
        == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "CUDA_DEVICE_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba14b_specificity_raw_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for site in ("canonical", "adjacent"):
            process = ctx.Process(
                target=site_worker,
                kwargs={
                    "site": site,
                    "expected_head": expected_head,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"mamba14b-specificity-{site}",
            )
            process.start()
            processes.append(process)

        for site, process in zip(
            ("canonical", "adjacent"),
            processes,
            strict=True,
        ):
            process.join()
            if process.exitcode != 0:
                paths = _worker_paths(temp_dir, site)
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise SpecificityRawError(
                    f"WORKER_FAILED:{site}:\n{detail}"
                )

        items, site_meta = merge_site_outputs(temp_dir)
        summary = write_output_bundle(
            output_dir=output_dir,
            expected_head=expected_head,
            items=items,
            site_meta=site_meta,
        )

    require(
        summary["forward_accounting"]["scientific_model_forward_count"]
        == TOTAL_FORWARD_BUDGET,
        "SUMMARY_FORWARD_BUDGET",
    )
    require(summary["inference"]["performed"] is False, "SUMMARY_INFERENCE")
    require(summary["scientific_conclusion"] is None, "SUMMARY_CONCLUSION")
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen Mamba-1.4B one-shot adjacent-site paired raw "
            "specificity response on fresh XG1 5101-5400. GPU0 runs canonical "
            "(33,34,35); GPU1 runs adjacent (34,35,36). No inferential test."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / geom.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_raw_response(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    desc = summary["raw_descriptive_only"]
    print("RESULT=" + str(summary["result"]))
    print("PAIR_RANGE=xg1_fact_5101..xg1_fact_5400")
    print("PAIR_COUNT=300")
    print("CANONICAL_TRIPLET=33,34,35")
    print("ADJACENT_TRIPLET=34,35,36")
    print("CANONICAL_STRONG_DIM=829")
    print("ADJACENT_STRONG_DIM=1205")
    print("FIXED_CAUSAL_CANDIDATE=P5")
    print("CANONICAL_CONTROL=P4")
    print("ADJACENT_CONTROL=P4")
    print("CONTROL_SELECTION_USES_XG1_RESPONSE=False")
    print("MEAN_D_CAN=" + format(float(desc["mean_D_CAN"]), ".17g"))
    print("MEAN_D_ADJ=" + format(float(desc["mean_D_ADJ"]), ".17g"))
    print("MEAN_S=" + format(float(desc["mean_S"]), ".17g"))
    print("CANONICAL_MODEL_FORWARD_COUNT=24000")
    print("ADJACENT_MODEL_FORWARD_COUNT=24000")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=48000")
    print("XG1_RESPONSE_OBSERVED=True")
    print("INFERENTIAL_TEST_PERFORMED=False")
    print("PRIMARY_P_VALUE_COMPUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    print("SECOND_ADJACENT_SITE_EXECUTED=False")
    print("RESCUE_PERFORMED=False")


if __name__ == "__main__":
    main()
