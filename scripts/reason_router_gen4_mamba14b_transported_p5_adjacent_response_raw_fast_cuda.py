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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import (
    reason_router_gen4_mamba14b_confirmation_fast_cuda as core,
)
from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda as geom,
)
from scripts import (
    reason_router_gen4_mamba14b_one_shot_adjacent_site_specificity_raw_fast_cuda
    as specificity,
)
from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility
    as direct,
)
from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence
    as ref_fd,
)

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_PROGRAM_ANCESTOR = "fa4a68054176412ec1e30be4d14dd7202314435f"

PROGRAM_PATH = Path(
    "reports/reason_router_gen4_next_mechanistic_program_prospective_freeze.md"
)
PROGRAM_GIT_BLOB = "487bf844e27279392a4203c1dc70e5702e3ad325"

SPECIFICITY_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_one_shot_adjacent_site_specificity_raw_fast_cuda.py"
)
SPECIFICITY_SCRIPT_GIT_BLOB = "99dc4175530153aff61357582efa869d7f0bf8de"

CORE_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_confirmation_fast_cuda.py"
)
CORE_SCRIPT_GIT_BLOB = "0302ceb263cdc0e5f37eb4ef0dae4f61aeab55d8"

DIRECT_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py"
)
DIRECT_SCRIPT_GIT_BLOB = "db523f66d9a05297077af18b547b4f2b2c9379ea"

REFERENCE_FD_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence.py"
)
REFERENCE_FD_SCRIPT_GIT_BLOB = "3dc94d833f81d829e81c85c0c3641fce861e8eb7"

TRANSPORT_SUMMARY_PATH = Path(
    "reports/reason_router_gen4_mamba14b_p5_cross_block_population_transport_runs/"
    "g4k-mamba14b-p5-crossblock-population-859831c-2t4/transport_summary.json"
)
TRANSPORT_SUMMARY_GIT_BLOB = "14fbb34cb2fff430009af7e3c4bfcdac707387dd"

HISTORICAL_RAW_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_site_specificity_raw_runs/"
    "g4k-mamba14b-adjacent-specificity-raw-xg1-5101-5400-2gpu-d8327f9"
)
HISTORICAL_ITEMS_PATH = HISTORICAL_RAW_ROOT / "paired_specificity_items.jsonl"
HISTORICAL_SUMMARY_PATH = HISTORICAL_RAW_ROOT / "raw_response_summary.json"
HISTORICAL_ITEMS_GIT_BLOB = "64da91ba82a17982feab0e7c8bd3bfba87bade00"
HISTORICAL_SUMMARY_GIT_BLOB = "f1f1e764c3cd5c412d13be67555004b10795b649"

PAIR_FIRST = 5101
PAIR_LAST = 5400
PAIR_COUNT = 300
PAIR_IDS = tuple(
    f"xg1_fact_{index:04d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

CELL = "C2_NAME"
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2
SOURCE_BLOCK = 35
TARGET_BLOCK = 36
SOURCE_PLANE = "P5"
ADJACENT_CONTROL_PLANE = "P4"
EPSILON = 0.025

ADJACENT_STRONG_DIM = 1205
AMBIENT_DIM = 4096
BASIS_NAMES = ("plus", "minus")

RESPONSE_FORWARDS_PER_PAIR = 80
TRANSPORT_BOUNDARY_FORWARDS_PER_PAIR = 1
LOCAL_FD_FORWARDS_PER_PAIR = 4
TOTAL_RESPONSE_FORWARDS = PAIR_COUNT * RESPONSE_FORWARDS_PER_PAIR
TOTAL_TRANSPORT_BOUNDARY_FORWARDS = (
    PAIR_COUNT * TRANSPORT_BOUNDARY_FORWARDS_PER_PAIR
)
TOTAL_LOCAL_FD_FORWARDS = PAIR_COUNT * LOCAL_FD_FORWARDS_PER_PAIR
TOTAL_FULL_MODEL_FORWARDS = (
    TOTAL_RESPONSE_FORWARDS + TOTAL_TRANSPORT_BOUNDARY_FORWARDS
)

GPU_COUNT = 2
PAIRS_PER_SHARD = PAIR_COUNT // GPU_COUNT
SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 150,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
    },
)

# Frozen before any transported-basis response is observed.
PLANNED_PRIMARY_ENDPOINT = "G=D_TRANSPORT-D_ADJ"
PLANNED_PRIMARY_TEST = "paired_one_sample_student_t_greater"
PLANNED_ALPHA = 0.05
PLANNED_PRIMARY_P_VALUE_COUNT = 1
PLANNED_SIGN_GATE = "mean(D_TRANSPORT)>0"
PLANNED_HISTORICAL_REFERENCE = (
    "same-pair frozen Experiment-5 D_ADJ on XG1 5101..5400"
)
D_CAN_ROLE = "descriptive_only"
CONTROL_EXTENSION_RULE = (
    "orthogonal_residualization_of_frozen_adjacent_P4_against_"
    "row_conditioned_transported_P5"
)

RESULT_PASS = (
    "PASS_MAMBA14B_TRANSPORTED_P5_ADJACENT_RESPONSE_RAW"
)
ITEM_SCHEMA = "gen4-mamba14b-transported-p5-adjacent-response-raw-item-v2"
SUMMARY_SCHEMA = "gen4-mamba14b-transported-p5-adjacent-response-raw-summary-v2"
MANIFEST_SCHEMA = "gen4-mamba14b-transported-p5-adjacent-response-raw-manifest-v2"

ITEM_FILE = "transported_adjacent_response_items.jsonl"
SUMMARY_FILE = "raw_response_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"


class TransportedResponseError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportedResponseError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransportedResponseError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_PROGRAM_ANCESTOR,
            expected_head,
        )
        == 0,
        "PROGRAM_FREEZE_NOT_ANCESTOR",
    )

    pinned = {
        PROGRAM_PATH: PROGRAM_GIT_BLOB,
        SPECIFICITY_SCRIPT_PATH: SPECIFICITY_SCRIPT_GIT_BLOB,
        CORE_SCRIPT_PATH: CORE_SCRIPT_GIT_BLOB,
        DIRECT_SCRIPT_PATH: DIRECT_SCRIPT_GIT_BLOB,
        REFERENCE_FD_SCRIPT_PATH: REFERENCE_FD_SCRIPT_GIT_BLOB,
        TRANSPORT_SUMMARY_PATH: TRANSPORT_SUMMARY_GIT_BLOB,
        HISTORICAL_ITEMS_PATH: HISTORICAL_ITEMS_GIT_BLOB,
        HISTORICAL_SUMMARY_PATH: HISTORICAL_SUMMARY_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )

    transport_summary = json.loads(
        TRANSPORT_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    require(
        transport_summary["result"]
        == "PASS_MAMBA14B_P5_CROSS_BLOCK_POPULATION_TRANSPORT_MEASUREMENT",
        "TRANSPORT_POPULATION_NOT_PASS",
    )
    require(
        int(transport_summary["pooled"]["rank_counts"]["2"]) == 600,
        "TRANSPORT_POPULATION_RANK2_NOT_600",
    )
    require(
        float(transport_summary["transport"]["epsilon"]) == EPSILON,
        "TRANSPORT_EPSILON",
    )


def validate_protocol() -> None:
    require(PAIR_FIRST == 5101 and PAIR_LAST == 5400, "PAIR_RANGE")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(len(PAIR_IDS) == PAIR_COUNT, "PAIR_IDS")
    require(CELL == "C2_NAME", "CELL")
    require(ANCHOR_NAME == "A_IDENTITY", "ANCHOR_NAME")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(SOURCE_BLOCK == 35, "SOURCE_BLOCK")
    require(TARGET_BLOCK == 36, "TARGET_BLOCK")
    require(SOURCE_PLANE == "P5", "SOURCE_PLANE")
    require(ADJACENT_CONTROL_PLANE == "P4", "ADJ_CONTROL")
    require(EPSILON == ref_fd.EPSILON == 0.025, "EPSILON")
    require(ADJACENT_STRONG_DIM == 1205, "ADJ_STRONG_DIM")
    require(AMBIENT_DIM == direct.INTERMEDIATE_SIZE == 4096, "AMBIENT_DIM")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")
    require(RESPONSE_FORWARDS_PER_PAIR == 80, "RESPONSE_FORWARDS_PER_PAIR")
    require(TOTAL_RESPONSE_FORWARDS == 24000, "TOTAL_RESPONSE_FORWARDS")
    require(TOTAL_TRANSPORT_BOUNDARY_FORWARDS == 300, "TRANSPORT_BOUNDARY_FORWARDS")
    require(TOTAL_LOCAL_FD_FORWARDS == 1200, "TOTAL_LOCAL_FD_FORWARDS")
    require(TOTAL_FULL_MODEL_FORWARDS == 24300, "TOTAL_FULL_MODEL_FORWARDS")
    require(GPU_COUNT == 2 and PAIRS_PER_SHARD == 150, "SHARDS")
    require(PLANNED_PRIMARY_ENDPOINT == "G=D_TRANSPORT-D_ADJ", "PRIMARY_ENDPOINT")
    require(
        PLANNED_PRIMARY_TEST == "paired_one_sample_student_t_greater",
        "PRIMARY_TEST",
    )
    require(PLANNED_ALPHA == 0.05, "ALPHA")
    require(PLANNED_PRIMARY_P_VALUE_COUNT == 1, "P_VALUE_COUNT")
    require(PLANNED_SIGN_GATE == "mean(D_TRANSPORT)>0", "SIGN_GATE")
    require(D_CAN_ROLE == "descriptive_only", "D_CAN_ROLE")
    require(
        CONTROL_EXTENSION_RULE
        == (
            "orthogonal_residualization_of_frozen_adjacent_P4_against_"
            "row_conditioned_transported_P5"
        ),
        "CONTROL_EXTENSION_RULE",
    )


def _rank_two(matrix: torch.Tensor) -> tuple[int, float, torch.Tensor]:
    value = matrix.detach().cpu().to(torch.float64).contiguous()
    require(value.ndim == 2 and value.shape[1] == 2, "RANK_MATRIX_SHAPE")
    require(bool(torch.isfinite(value).all().item()), "RANK_NONFINITE")
    s = torch.linalg.svdvals(value)
    tol = (
        max(value.shape)
        * torch.finfo(torch.float64).eps
        * float(torch.max(s).item())
    )
    rank = int(torch.sum(s > tol).item())
    return rank, tol, s


def _canonical_qr(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = matrix.detach().cpu().to(torch.float64).contiguous()
    q, r = torch.linalg.qr(value, mode="reduced")
    diag = torch.diagonal(r)
    signs = torch.where(
        diag < 0.0,
        -torch.ones_like(diag),
        torch.ones_like(diag),
    )
    q = q * signs.unsqueeze(0)
    r = signs.unsqueeze(1) * r
    return q.contiguous(), r.contiguous()


def restrict_transported_plane_to_adjacent_strong(
    transported_plus: torch.Tensor,
    transported_minus: torch.Tensor,
    strong_indices: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    plus = transported_plus.detach().cpu().to(torch.float64).contiguous()
    minus = transported_minus.detach().cpu().to(torch.float64).contiguous()
    require(tuple(plus.shape) == (AMBIENT_DIM,), "PLUS_SHAPE")
    require(tuple(minus.shape) == (AMBIENT_DIM,), "MINUS_SHAPE")
    require(
        bool(torch.isfinite(plus).all().item())
        and bool(torch.isfinite(minus).all().item()),
        "TRANSPORT_NONFINITE",
    )

    w = torch.stack([plus, minus], dim=1)
    ambient_rank, ambient_tol, ambient_s = _rank_two(w)
    require(ambient_rank == 2, f"AMBIENT_TRANSPORT_RANK:{ambient_rank}")

    ambient_q, _ambient_r = _canonical_qr(w)
    indices = [int(value) for value in strong_indices]
    require(len(indices) == ADJACENT_STRONG_DIM, "STRONG_INDEX_COUNT")
    require(indices == sorted(indices), "STRONG_INDEX_ORDER")
    require(len(set(indices)) == ADJACENT_STRONG_DIM, "STRONG_INDEX_DUPLICATE")
    require(
        all(0 <= index < AMBIENT_DIM for index in indices),
        "STRONG_INDEX_RANGE",
    )

    restricted = ambient_q[
        torch.tensor(indices, dtype=torch.long),
        :,
    ].contiguous()
    strong_rank, strong_tol, strong_s = _rank_two(restricted)
    require(strong_rank == 2, f"STRONG_PROJECTED_RANK:{strong_rank}")

    strong_q, _strong_r = _canonical_qr(restricted)
    gram_residual = float(
        torch.max(
            torch.abs(
                strong_q.T @ strong_q
                - torch.eye(2, dtype=torch.float64)
            )
        ).item()
    )
    require(gram_residual <= 1e-10, f"STRONG_Q_GRAM:{gram_residual}")

    condition = float(strong_s[0].item() / strong_s[1].item())
    require(math.isfinite(condition) and condition >= 1.0, "STRONG_CONDITION")

    return {
        "plus": strong_q[:, 0].contiguous(),
        "minus": strong_q[:, 1].contiguous(),
    }, {
        "ambient_w_rank": ambient_rank,
        "ambient_w_rank_tolerance": ambient_tol,
        "ambient_w_singular_values": [
            float(value) for value in ambient_s.tolist()
        ],
        "strong_projection_rank": strong_rank,
        "strong_projection_rank_tolerance": strong_tol,
        "strong_projection_singular_values": [
            float(value) for value in strong_s.tolist()
        ],
        "strong_projection_condition_number": condition,
        "strong_plane_gram_max_abs_residual": gram_residual,
    }


def build_pair_transported_plane(
    *,
    pair: str,
    model: Any,
    kernels: Mapping[str, Any],
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    device: torch.device,
    strong_indices: Sequence[int],
    source_p5: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    row_key = (pair, CELL)
    require(row_key in lookup, f"TRANSPORT_ROW_MISSING:{pair}")
    event_key = (pair, CELL, ANCHOR_NAME)
    require(event_key in events, f"TRANSPORT_EVENT_MISSING:{pair}")

    row_index = int(lookup[row_key])
    input_ids = encoded["input_ids"][row_index].unsqueeze(0).contiguous()
    require(tuple(input_ids.shape) == (1, 128), "TRANSPORT_INPUT_SHAPE")
    anchor = int(events[event_key]["absolute_anchor_token_index"])
    target_abs = anchor + TARGET_OFFSET
    require(0 <= target_abs < 128, "TRANSPORT_TARGET_RANGE")

    baseline = direct.capture_baseline_boundary(
        model=model,
        input_ids=input_ids,
        target_abs=target_abs,
        device=device,
    )

    projected35 = baseline["projected35"].to(device).contiguous()
    residual35 = baseline["residual35"].to(device).contiguous()
    base_content = (
        projected35[
            0,
            target_abs,
            :AMBIENT_DIM,
        ]
        .detach()
        .clone()
    )

    layer35 = model.mamba.layers[SOURCE_BLOCK]
    layer36 = model.mamba.layers[TARGET_BLOCK]

    def fast_phi(content: torch.Tensor) -> torch.Tensor:
        projected = direct.replace_target_content(
            projected35,
            content,
            target_abs,
        )
        return direct.local_block35_to_block36_map(
            projected35=projected,
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
            kernels=kernels,
        )

    transported: dict[str, torch.Tensor] = {}
    for name in BASIS_NAMES:
        _plus, _minus, derivative = ref_fd.fast_symmetric_fd(
            fast_phi,
            base_content,
            source_p5[name],
        )
        transported[name] = (
            derivative.detach().cpu().to(torch.float64).contiguous()
        )

    plane, diagnostics = restrict_transported_plane_to_adjacent_strong(
        transported["plus"],
        transported["minus"],
        strong_indices,
    )
    return plane, {
        "transport_source_cell": CELL,
        "transport_anchor_name": ANCHOR_NAME,
        "transport_anchor": anchor,
        "transport_target_abs": target_abs,
        "transport_epsilon": EPSILON,
        "transport_full_model_forward_count": 1,
        "transport_local_perturbed_forward_count": 4,
        **diagnostics,
    }


def _validated_plane_matrix(
    plane: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> torch.Tensor:
    require(set(plane) == {"plus", "minus"}, f"{label}_PLANE_KEYS")
    matrix = torch.stack(
        [
            plane["plus"].detach().cpu().to(torch.float64).contiguous(),
            plane["minus"].detach().cpu().to(torch.float64).contiguous(),
        ],
        dim=1,
    )
    require(
        tuple(matrix.shape) == (ADJACENT_STRONG_DIM, 2),
        f"{label}_PLANE_SHAPE:{tuple(matrix.shape)}",
    )
    require(bool(torch.isfinite(matrix).all().item()), f"{label}_PLANE_NONFINITE")
    gram_residual = float(
        torch.max(
            torch.abs(
                matrix.T @ matrix
                - torch.eye(2, dtype=torch.float64)
            )
        ).item()
    )
    require(gram_residual <= 1e-10, f"{label}_PLANE_GRAM:{gram_residual}")
    return matrix


def residualize_control_against_selected(
    selected_plane: Mapping[str, torch.Tensor],
    frozen_control_plane: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected = _validated_plane_matrix(selected_plane, label="SELECTED")
    control = _validated_plane_matrix(frozen_control_plane, label="CONTROL")

    cross_before = (selected.T @ control).contiguous()
    residual = (
        control
        - selected @ cross_before
    ).contiguous()

    residual_rank, residual_tol, residual_s = _rank_two(residual)
    require(
        residual_rank == 2,
        f"CONTROL_RESIDUAL_RANK:{residual_rank}",
    )

    # Polar orthonormalization gives the closest orthonormal basis to the
    # residualized frozen P4 columns, preserving the fixed P4 source while
    # extending the historical operator to a nonorthogonal transported P5.
    u, _s, vh = torch.linalg.svd(
        residual,
        full_matrices=False,
    )
    control_q = (u @ vh).contiguous()

    gram_after = float(
        torch.max(
            torch.abs(
                control_q.T @ control_q
                - torch.eye(2, dtype=torch.float64)
            )
        ).item()
    )
    cross_after = (selected.T @ control_q).contiguous()
    cross_after_max = float(torch.max(torch.abs(cross_after)).item())

    require(gram_after <= 1e-10, f"CONTROL_Q_GRAM:{gram_after}")
    require(
        cross_after_max <= 1e-10,
        f"CONTROL_SELECTED_ORTHOGONALITY:{cross_after_max}",
    )

    return {
        "plus": control_q[:, 0].contiguous(),
        "minus": control_q[:, 1].contiguous(),
    }, {
        "control_source_plane": ADJACENT_CONTROL_PLANE,
        "control_extension_rule": CONTROL_EXTENSION_RULE,
        "selected_control_cross_gram_before": [
            [float(value) for value in row]
            for row in cross_before.tolist()
        ],
        "selected_control_cross_max_abs_before":
            float(torch.max(torch.abs(cross_before)).item()),
        "control_residual_rank": residual_rank,
        "control_residual_rank_tolerance": residual_tol,
        "control_residual_singular_values": [
            float(value) for value in residual_s.tolist()
        ],
        "control_residual_condition_number":
            float(residual_s[0].item() / residual_s[1].item()),
        "control_orthonormal_gram_max_abs_residual": gram_after,
        "selected_control_cross_max_abs_after": cross_after_max,
    }


def pair_planes(
    frozen_planes: Mapping[str, Mapping[str, torch.Tensor]],
    transported_plane: Mapping[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    require("P5" in frozen_planes and "P4" in frozen_planes, "FROZEN_PLANES")
    require(set(transported_plane) == {"plus", "minus"}, "TRANSPORTED_PLANE_KEYS")
    selected = {
        "plus": transported_plane["plus"],
        "minus": transported_plane["minus"],
    }
    control, control_diag = residualize_control_against_selected(
        selected,
        frozen_planes["P4"],
    )
    out = {
        plane: {
            "plus": values["plus"],
            "minus": values["minus"],
        }
        for plane, values in frozen_planes.items()
    }
    out["P5"] = selected
    out["P4"] = control
    return out, control_diag


def _worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "items": temp_dir / f"shard_{shard_id}_items.jsonl",
        "meta": temp_dir / f"shard_{shard_id}_meta.json",
        "error": temp_dir / f"shard_{shard_id}_error.txt",
    }


def worker_run(
    *,
    shard_id: int,
    expected_head: str,
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    paths = _worker_paths(Path(temp_dir), shard_id)
    try:
        validate_protocol()
        authenticate_repo(expected_head)
        require(0 <= shard_id < len(SHARDS), f"SHARD_ID:{shard_id}")
        shard = SHARDS[shard_id]

        cfg = specificity.configure_site("adjacent")
        require(tuple(cfg["triplet"]) == (34, 35, 36), "ADJACENT_TRIPLET")
        require(int(cfg["dim"]) == ADJACENT_STRONG_DIM, "ADJACENT_DIM")
        require(
            str(cfg["control_plane"]) == ADJACENT_CONTROL_PLANE,
            "ADJACENT_CONTROL",
        )

        physical_device = int(shard["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        device = core.runtime_gate_single_visible_gpu(physical_device)

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        frozen = specificity.load_adjacent_geometry()
        rows, encoded, events = specificity.build_input_state(snapshot)
        lookup = specificity.row_index(rows)
        source_p5 = direct.load_canonical_p5_ambient()

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        runtime_ctx = geom.runtime_components(model)
        specificity.validate_runtime_geometry(
            runtime_ctx,
            frozen,
            cfg,
        )

        strong_indices = [
            int(value)
            for value in frozen["strong_indices"]
        ]
        require(
            len(strong_indices) == ADJACENT_STRONG_DIM,
            "STRONG_DIM",
        )

        start = int(shard["start_index"])
        end = int(shard["end_index"])
        require(end - start == PAIRS_PER_SHARD, "SHARD_PAIR_COUNT")

        budget = core.ForwardBudget(
            PAIRS_PER_SHARD * RESPONSE_FORWARDS_PER_PAIR
        )
        items: list[dict[str, Any]] = []

        for pair_index in range(start, end):
            pair = PAIR_IDS[pair_index]

            transported_plane, transport_diag = build_pair_transported_plane(
                pair=pair,
                model=model,
                kernels=kernels,
                encoded=encoded,
                lookup=lookup,
                events=events,
                device=device,
                strong_indices=strong_indices,
                source_p5=source_p5,
            )
            planes, control_diag = pair_planes(
                frozen["planes"],
                transported_plane,
            )
            transport_diag = {
                **transport_diag,
                "control_orthogonalization": control_diag,
            }

            raw = core.run_pair(
                pair_index=pair_index,
                pair=pair,
                bases=frozen["bases"],
                model=model,
                runtime_ctx=runtime_ctx,
                kernels=kernels,
                planes=planes,
                encoded=encoded,
                lookup=lookup,
                events=events,
                device=device,
                budget=budget,
            )

            items.append({
                "schema_version": ITEM_SCHEMA,
                "source_pair_id": pair,
                "pair_index": pair_index,
                "site": "adjacent",
                "triplet": [34, 35, 36],
                "strong_dim": ADJACENT_STRONG_DIM,
                "selected_plane_role": "row_conditioned_transported_canonical_P5",
                "transport_plane_source_row": {
                    "contrast_cell_id": CELL,
                    "anchor_name": ANCHOR_NAME,
                    "target_offset": TARGET_OFFSET,
                },
                "response_blind_control_plane": ADJACENT_CONTROL_PLANE,
                "control_extension_rule": CONTROL_EXTENSION_RULE,
                "epsilon": EPSILON,
                "conditions": raw["conditions"],
                "Q_restored": float(raw["Q_restored"]),
                "Q_control": float(raw["Q_control"]),
                "D_TRANSPORT": float(raw["D_CORE"]),
                "transport_diagnostics": transport_diag,
                "response_model_forward_count": RESPONSE_FORWARDS_PER_PAIR,
                "transport_boundary_full_model_forward_count": 1,
                "transport_local_perturbed_forward_count": 4,
                "inferential_test_performed": False,
                "historical_D_ADJ_accessed_during_execution": False,
                "historical_D_CAN_accessed_during_execution": False,
                "selection_reopened": False,
                "rescue_performed": False,
            })

        budget.assert_exact()
        torch.cuda.synchronize(device)

        require(len(items) == PAIRS_PER_SHARD, "WORKER_ITEM_COUNT")
        require(
            [item["source_pair_id"] for item in items]
            == list(PAIR_IDS[start:end]),
            "WORKER_PAIR_ORDER",
        )

        items_bytes = jsonl_bytes(items)
        paths["items"].write_bytes(items_bytes)

        meta = {
            "schema_version":
                "gen4-mamba14b-transported-p5-adjacent-response-worker-v1",
            "shard_id": shard_id,
            "physical_device": physical_device,
            "logical_device": 0,
            "cuda_visible_devices": str(physical_device),
            "device_name": torch.cuda.get_device_name(0),
            "pair_first": PAIR_IDS[start],
            "pair_last": PAIR_IDS[end - 1],
            "pair_count": PAIRS_PER_SHARD,
            "response_model_forward_count":
                PAIRS_PER_SHARD * RESPONSE_FORWARDS_PER_PAIR,
            "transport_boundary_full_model_forward_count":
                PAIRS_PER_SHARD,
            "transport_local_perturbed_forward_count":
                PAIRS_PER_SHARD * LOCAL_FD_FORWARDS_PER_PAIR,
            "items_sha256": sha256_bytes(items_bytes),
            "model_provenance": model_provenance,
            "inferential_test_performed": False,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def _descriptive(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.shape == (PAIR_COUNT,), "DESCRIPTIVE_SHAPE")
    require(bool(np.isfinite(array).all()), "DESCRIPTIVE_NONFINITE")
    return {
        "mean": float(array.mean()),
        "sd_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def run_raw(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    require(model_snapshot.is_dir(), "SNAPSHOT_DIR")
    require(
        compact_checkpoint.resolve()
        == (ROOT / geom.COMPACT_CHECKPOINT_REL).resolve(),
        "CHECKPOINT_PATH",
    )
    require(
        geom.sha256_file(compact_checkpoint)
        == geom.COMPACT_CHECKPOINT_SHA256,
        "CHECKPOINT_SHA",
    )
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba14b_transported_p5_adjacent_response_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard_id": int(shard["shard_id"]),
                    "expected_head": expected_head,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"transported-p5-adjacent-shard-{shard['shard_id']}",
            )
            process.start()
            processes.append(process)

        for shard_id, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                error_path = _worker_paths(
                    temp_dir,
                    shard_id,
                )["error"]
                detail = (
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise TransportedResponseError(
                    f"WORKER_{shard_id}_FAILED:\n{detail}"
                )

        items: list[dict[str, Any]] = []
        worker_meta: list[dict[str, Any]] = []

        for shard_id in range(GPU_COUNT):
            paths = _worker_paths(temp_dir, shard_id)
            require(paths["items"].is_file(), f"ITEMS_MISSING:{shard_id}")
            require(paths["meta"].is_file(), f"META_MISSING:{shard_id}")

            raw_items = paths["items"].read_bytes()
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            require(
                meta["items_sha256"] == sha256_bytes(raw_items),
                f"ITEM_SHA:{shard_id}",
            )
            shard_items = [
                json.loads(line)
                for line in raw_items.decode("utf-8").splitlines()
                if line.strip()
            ]
            require(
                len(shard_items) == PAIRS_PER_SHARD,
                f"SHARD_ITEMS:{shard_id}",
            )
            items.extend(shard_items)
            worker_meta.append(meta)

    require(len(items) == PAIR_COUNT, "ITEM_COUNT")
    require(
        [item["source_pair_id"] for item in items] == list(PAIR_IDS),
        "PAIR_ORDER",
    )
    require(
        all(
            int(item["transport_diagnostics"]["ambient_w_rank"]) == 2
            and int(item["transport_diagnostics"]["strong_projection_rank"]) == 2
            for item in items
        ),
        "TRANSPORT_RANK_NOT_TWO",
    )

    d_transport = [float(item["D_TRANSPORT"]) for item in items]
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "phase": "study_a_optional_transported_basis_adjacent_response_raw",
        "claim_boundary": (
            "Raw transported-basis adjacent response only. "
            "No primary inference is executed here. "
            "Historical Experiment-5 D_ADJ and D_CAN are not read during GPU execution."
        ),
        "population": {
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "pair_count": PAIR_COUNT,
            "cohort": "same frozen Experiment-5 XG1 5101..5400",
        },
        "transported_plane": {
            "source_block": SOURCE_BLOCK,
            "target_block": TARGET_BLOCK,
            "source_plane": SOURCE_PLANE,
            "source_row_cell": CELL,
            "anchor_name": ANCHOR_NAME,
            "target_offset": TARGET_OFFSET,
            "epsilon": EPSILON,
            "ambient_dimension": AMBIENT_DIM,
            "adjacent_strong_dimension": ADJACENT_STRONG_DIM,
            "restriction_rule": (
                "orth(J_q U35) in ambient coordinates; restrict to frozen adjacent "
                "strong indices; require rank 2; orthonormalize restricted plane"
            ),
            "same_pair_plane_applied_to_both_target_branches": True,
        },
        "response": {
            "site": "adjacent",
            "triplet": [34, 35, 36],
            "response_blind_control_plane": ADJACENT_CONTROL_PLANE,
            "control_extension_rule": CONTROL_EXTENSION_RULE,
            "control_extension_interpretation": (
                "Frozen adjacent P4 is projected into the orthogonal complement "
                "of each row-conditioned transported P5 and symmetrically "
                "orthonormalized. This is response-blind and reduces to the "
                "historical P4 operator when selected and control planes are "
                "already orthogonal."
            ),
            "probe_bases": "frozen adjacent XG2/XG4 K=5 bases",
            "endpoint": (
                "D_TRANSPORT=Q_restored(transported_P5)-"
                "Q_control(orthogonalized_frozen_adjacent_P4)"
            ),
            "descriptive_D_TRANSPORT": _descriptive(d_transport),
        },
        "planned_static_analysis": {
            "primary_endpoint": PLANNED_PRIMARY_ENDPOINT,
            "historical_reference": PLANNED_HISTORICAL_REFERENCE,
            "primary_test": PLANNED_PRIMARY_TEST,
            "alternative": "E[G] > 0",
            "alpha": PLANNED_ALPHA,
            "primary_p_value_count": PLANNED_PRIMARY_P_VALUE_COUNT,
            "sign_gate": PLANNED_SIGN_GATE,
            "D_CAN_role": D_CAN_ROLE,
            "reserved_for_cpu_after_raw_freeze": True,
        },
        "historical_artifacts_pinned_but_not_read_during_execution": {
            "items_path": HISTORICAL_ITEMS_PATH.as_posix(),
            "items_git_blob": HISTORICAL_ITEMS_GIT_BLOB,
            "summary_path": HISTORICAL_SUMMARY_PATH.as_posix(),
            "summary_git_blob": HISTORICAL_SUMMARY_GIT_BLOB,
        },
        "forward_accounting": {
            "response_model_forward_count": TOTAL_RESPONSE_FORWARDS,
            "transport_boundary_full_model_forward_count":
                TOTAL_TRANSPORT_BOUNDARY_FORWARDS,
            "scientific_full_model_forward_count": TOTAL_FULL_MODEL_FORWARDS,
            "transport_local_perturbed_forward_count": TOTAL_LOCAL_FD_FORWARDS,
        },
        "workers": worker_meta,
        "boundary": {
            "inferential_test_performed": False,
            "p_value_count_added": 0,
            "historical_D_ADJ_accessed_during_execution": False,
            "historical_D_CAN_accessed_during_execution": False,
            "new_cohort_selected": False,
            "layer_sweep_executed": False,
            "token_sweep_executed": False,
            "epsilon_sweep_executed": False,
            "row_subset_selection_executed": False,
            "control_reselection_executed": False,
            "training_executed": False,
            "parameter_update_executed": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    item_bytes = jsonl_bytes(items)
    summary_bytes = pretty_json_bytes(summary)
    (output_dir / ITEM_FILE).write_bytes(item_bytes)
    (output_dir / SUMMARY_FILE).write_bytes(summary_bytes)

    file_hashes = {
        ITEM_FILE: sha256_bytes(item_bytes),
        SUMMARY_FILE: sha256_bytes(summary_bytes),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_program_ancestor": REQUIRED_PROGRAM_ANCESTOR,
        "output_file_sha256": dict(sorted(file_hashes.items())),
        "planned_primary_endpoint": PLANNED_PRIMARY_ENDPOINT,
        "planned_primary_test": PLANNED_PRIMARY_TEST,
        "planned_alpha": PLANNED_ALPHA,
        "planned_primary_p_value_count": PLANNED_PRIMARY_P_VALUE_COUNT,
        "historical_items_git_blob": HISTORICAL_ITEMS_GIT_BLOB,
        "historical_summary_git_blob": HISTORICAL_SUMMARY_GIT_BLOB,
        "transport_summary_git_blob": TRANSPORT_SUMMARY_GIT_BLOB,
        "inferential_test_performed": False,
        "scientific_full_model_forward_count": TOTAL_FULL_MODEL_FORWARDS,
        "transport_local_perturbed_forward_count": TOTAL_LOCAL_FD_FORWARDS,
        "rescue_performed": False,
    }
    manifest_bytes = pretty_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_bytes)
    file_hashes[MANIFEST_FILE] = sha256_bytes(manifest_bytes)

    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(file_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prospectively frozen optional Study-A transported-canonical-P5 "
            "adjacent response on the same XG1 5101..5400 cohort. The row-conditioned "
            "transported plane is built from C2_NAME at block35->36 with epsilon=0.025, "
            "restricted to the frozen adjacent strong mask, then used as the selected "
            "adjacent plane. Raw response only; no p-value is computed."
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_raw(
        expected_head=str(args.expected_head),
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(summary["result"]))
    print("PAIR_FIRST=xg1_fact_5101")
    print("PAIR_LAST=xg1_fact_5400")
    print("PAIR_COUNT=300")
    print("SITE=adjacent")
    print("TRIPLET=34,35,36")
    print("TRANSPORT_SOURCE_CELL=C2_NAME")
    print("TRANSPORT_SOURCE_BLOCK=35")
    print("TRANSPORT_TARGET_BLOCK=36")
    print("TRANSPORT_EPSILON=0.025")
    print("ADJACENT_CONTROL=P4")
    print("RESPONSE_MODEL_FORWARD_COUNT=24000")
    print("TRANSPORT_BOUNDARY_FULL_MODEL_FORWARD_COUNT=300")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=24300")
    print("TRANSPORT_LOCAL_PERTURBED_FORWARD_COUNT=1200")
    print("PLANNED_PRIMARY_ENDPOINT=G=D_TRANSPORT-D_ADJ")
    print("PLANNED_PRIMARY_TEST=one-sided paired Student greater")
    print("PLANNED_PRIMARY_P_VALUE_COUNT=1")
    print("INFERENTIAL_TEST_PERFORMED=False")
    print("HISTORICAL_D_ADJ_ACCESSED_DURING_EXECUTION=False")
    print("HISTORICAL_D_CAN_ACCESSED_DURING_EXECUTION=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
