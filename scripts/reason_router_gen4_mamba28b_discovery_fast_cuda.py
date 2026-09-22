#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import tempfile
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    build_reason_router_gen4_mamba370m_xg1_holdouts
    as holdout_builder,
)
from scripts import (
    reason_router_gen4_generator_family_prevalence_kernel_compat
    as kernel_compat,
)
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runtime
    as transport_runtime,
)
from scripts import (
    reason_router_gen4_mamba28b_geometry_prepare_fast_cuda
    as geom,
)
from scripts import reason_router_gen4_native_mamba_state_measurement as measurement
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

GEOMETRY_FREEZE_COMMIT = "06082ee152866ba77ebb1d6a342a4387ded140e3"
DISCOVERY_HOLDOUT_FREEZE_COMMIT = "6c0989a45382db31c9052b89b088e999b3e7cf59"

GEOMETRY_ROOT = Path(
    "reports/reason_router_gen4_mamba28b_geometry_preparation_runs/"
    "g4k-mamba28b-geometry-xg2xg4-2gpu-7744d94"
)
GEOMETRY_SUMMARY_SHA256 = (
    "6d60009fd3bea86442997d5c9585991b90696b9e97ccc9d4ac511d9cc4843d3c"
)
STRONG_INDICES_SHA256 = (
    "cb8a2fb157ffe22b33021cb8b884665d3c7469b3dbc3e82f91ef10a856cbdbfa"
)
XG2_BASIS_SHA256 = (
    "9fbef631420de05fe54e0aa663a21d5def11b951d502a1c23537057f544dc209"
)
XG4_BASIS_SHA256 = (
    "c9ffafa4416a4ccef5f3e14652b0379e8cf7b1c4b4e16d91f72e7aba952478fe"
)
PLANE_FILE_SHA256 = {
    "P1": {
        "plus": "a1a0de78a2f68c48bd9c6388b1654221436fa203fa096eef59f4311766222689",
        "minus": "019c1bbf566079008d830454d082c614d08c0979f20f710443b0b5506789cec3",
    },
    "P2": {
        "plus": "63a06c4a6c067549092bed55a22182de03702b14d264e2c038867afbd6db3c20",
        "minus": "8785d40b531fec48ae6b55d3f612b3bab3701409792321c60ece9a0c39aca68d",
    },
    "P3": {
        "plus": "92d232e34c56843ce9f54f66a764b2be28f424f13e91a3dd10abd230ce057239",
        "minus": "458ba21a2046e518a03e3e82ee91e43157ff4449a886ebda78efcde23ec75092",
    },
    "P4": {
        "plus": "957269f29c61992f37708b7a92f84f62c864cb45103d684615a2e92040442e54",
        "minus": "67ae3bcb89d4983b3c3456a2a8e5b7fef29fa3750e65e7cedf150445c2efbed3",
    },
    "P5": {
        "plus": "634d8f60b223867f291cd86f3be77a646f25488a6bff192cd77dd0a7e20fd72f",
        "minus": "7a6940e58bf4592f0b22832c3cddbf3d1c136376f604dc635cd81fbfbb9a34b2",
    },
}

DISCOVERY_ROOT = Path(
    "data/reason_router_gen4_mamba28b_xg1_discovery_v1"
)
DISCOVERY_SOURCE_SHA256 = (
    "c4a22871c79b158bffc69629307defa56d61cf2851ad3e7bdb7c260a0584a754"
)
DISCOVERY_ROWS_SHA256 = (
    "b6a07a6877f5e179baca44964ae7d27ea6f4a0836999c1cf72af3c66de612e3e"
)
DISCOVERY_MANIFEST_SHA256 = (
    "f91c1a47bfedc06de9037e60b9e4caf77b5ab9f6a4ba0b3db470cf925671e61d"
)

PAIR_FIRST = 6001
PAIR_LAST = 6300
PAIR_COUNT = 300
ROWS = 1800
PAIR_IDS = tuple(
    f"xg1_fact_{index:03d}"
    for index in range(PAIR_FIRST, PAIR_LAST + 1)
)

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
K = 5
EPS = 0.025
DIM = 1003
STATE_WIDTH = geom.INTERMEDIATE_SIZE * geom.STATE_SIZE
DIRECTION_ORDER = tuple(
    [f"xg2_{index}" for index in range(K)]
    + [f"xg4_{index}" for index in range(K)]
)
CONDITION_ORDER = (
    "restored",
    "p1_neutralized",
    "p2_neutralized",
    "p3_neutralized",
    "p4_neutralized",
    "p5_neutralized",
)

TARGET_PLUS_CELL = geom.TARGET_PLUS_CELL
TARGET_MINUS_CELL = geom.TARGET_MINUS_CELL
REFERENCE_PLUS_CELL = geom.REFERENCE_PLUS_CELL
REFERENCE_MINUS_CELL = geom.REFERENCE_MINUS_CELL
ANCHOR_NAME = "A_IDENTITY"

FORWARDS_PER_SIGNED = 2
FORWARDS_PER_DIRECTION = 4
FORWARDS_PER_CONDITION = len(DIRECTION_ORDER) * FORWARDS_PER_DIRECTION
FORWARDS_PER_PAIR = len(CONDITION_ORDER) * FORWARDS_PER_CONDITION
TOTAL_FORWARD_BUDGET = PAIR_COUNT * FORWARDS_PER_PAIR

GPU_COUNT = 2
PAIRS_PER_SHARD = PAIR_COUNT // GPU_COUNT
FORWARDS_PER_SHARD = PAIRS_PER_SHARD * FORWARDS_PER_PAIR

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_6001",
        "pair_last": "xg1_fact_6150",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_6151",
        "pair_last": "xg1_fact_6300",
        "pair_count": 150,
        "forward_budget": FORWARDS_PER_SHARD,
    },
)

ITEM_FILE = "discovery_items.jsonl"
SELECTION_FILE = "discovery_selection.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

RESULT_PASS = "PASS_MAMBA28B_DISCOVERY_SELECTION_FREEZE"
ITEM_SCHEMA = "gen4-mamba28b-discovery-item-v1"
SELECTION_SCHEMA = "gen4-mamba28b-discovery-selection-v1"
MANIFEST_SCHEMA = "gen4-mamba28b-discovery-manifest-v1"

TOL = 1.0e-12
PLANE_TOL = 1.0e-9


class DiscoveryError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DiscoveryError(message)


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


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DiscoveryError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    for freeze_commit, label in (
        (GEOMETRY_FREEZE_COMMIT, "GEOMETRY_FREEZE"),
        (DISCOVERY_HOLDOUT_FREEZE_COMMIT, "DISCOVERY_HOLDOUT_FREEZE"),
    ):
        rc = subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                freeze_commit,
                head,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")


def validate_protocol() -> None:
    require(geom.SOURCE_BLOCK == 45, "SOURCE_BLOCK")
    require(geom.TARGET_RESIDUAL_LAYER == 46, "TARGET_LAYER")
    require(geom.INTERVENTION_LAYER == 47, "INTERVENTION_LAYER")
    require(geom.TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(K == 5, "K")
    require(EPS == 0.025, "EPS")
    require(DIM == 1003, "DIM")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(FORWARDS_PER_CONDITION == 40, "FORWARDS_PER_CONDITION")
    require(FORWARDS_PER_PAIR == 240, "FORWARDS_PER_PAIR")
    require(TOTAL_FORWARD_BUDGET == 72000, "TOTAL_FORWARD_BUDGET")
    require(FORWARDS_PER_SHARD == 36000, "FORWARDS_PER_SHARD")
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")

    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["physical_device"] == expected_id, "SHARD_DEVICE")
        require(
            shard["end_index"] - shard["start_index"]
            == shard["pair_count"],
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["pair_count"] * FORWARDS_PER_PAIR
            == shard["forward_budget"],
            "SHARD_FORWARD_BUDGET",
        )
        covered.extend(
            range(shard["start_index"], shard["end_index"])
        )
    require(covered == list(range(PAIR_COUNT)), "SHARD_COVERAGE")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def load_discovery_population() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = DISCOVERY_ROOT / holdout_builder.SOURCE_FILE
    rows = DISCOVERY_ROOT / holdout_builder.ROW_FILE
    manifest_path = DISCOVERY_ROOT / "structural_manifest.json"

    require(source.is_file(), "DISCOVERY_SOURCE_MISSING")
    require(rows.is_file(), "DISCOVERY_ROWS_MISSING")
    require(manifest_path.is_file(), "DISCOVERY_MANIFEST_MISSING")

    require(
        sha256_file(source) == DISCOVERY_SOURCE_SHA256,
        "DISCOVERY_SOURCE_SHA",
    )
    require(
        sha256_file(rows) == DISCOVERY_ROWS_SHA256,
        "DISCOVERY_ROWS_SHA",
    )
    require(
        sha256_file(manifest_path) == DISCOVERY_MANIFEST_SHA256,
        "DISCOVERY_MANIFEST_SHA",
    )

    facts = _read_jsonl(source)
    materialized = _read_jsonl(rows)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    require(len(facts) == PAIR_COUNT, "DISCOVERY_FACT_COUNT")
    require(len(materialized) == ROWS, "DISCOVERY_ROW_COUNT")
    require(
        [str(row["pair_id"]) for row in facts] == list(PAIR_IDS),
        "DISCOVERY_PAIR_ORDER",
    )
    holdout_builder.prior.base.validate_materialized_rows(
        materialized,
        expected_pairs=PAIR_COUNT,
    )

    require(
        manifest["schema_version"]
        == "GEN4_MAMBA1_FIVE_SCALE_LADDER_XG1_PROSPECTIVE_HOLDOUT_V1",
        "DISCOVERY_SCHEMA",
    )
    require(
        manifest["result"]
        == "PASS_MAMBA1_FIVE_SCALE_LADDER_XG1_STRUCTURAL",
        "DISCOVERY_RESULT",
    )
    require(manifest["role"] == "discovery", "DISCOVERY_ROLE")
    require(
        manifest["prospective_use"]
        == "dominant_component_discovery_only",
        "DISCOVERY_USE",
    )
    require(
        manifest["all_five_planes_required"] == list(PLANE_ORDER),
        "DISCOVERY_ALL_FIVE_PLANES",
    )
    require(
        manifest["unique_argmax_required"] is True,
        "DISCOVERY_UNIQUE_ARGMAX",
    )
    require(
        manifest["positivity_gate"] is False,
        "DISCOVERY_POSITIVITY_GATE",
    )
    require(
        manifest["response_blind_control_from_geometry_only"] is True,
        "DISCOVERY_CONTROL_RULE",
    )
    require(
        manifest["selection_allowed"] is True
        and manifest["formal_inference_allowed"] is False
        and manifest["p_value_count"] == 0,
        "DISCOVERY_SELECTION_BOUNDARY",
    )
    require(
        manifest["confirmation_response_access_allowed"] is False,
        "CONFIRMATION_RESPONSE_BOUNDARY",
    )
    require(
        manifest["readout_response_access_allowed"] is False,
        "READOUT_RESPONSE_BOUNDARY",
    )
    require(manifest["rescue_policy"] == "none", "DISCOVERY_RESCUE")

    return facts, materialized


    return facts, materialized


def _load_f64(path: Path, shape: tuple[int, ...], expected_sha: str) -> torch.Tensor:
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


def load_frozen_geometry() -> dict[str, Any]:
    summary_path = GEOMETRY_ROOT / "geometry_summary.json"
    strong_path = GEOMETRY_ROOT / "strong_indices.json"

    require(
        sha256_file(summary_path) == GEOMETRY_SUMMARY_SHA256,
        "GEOMETRY_SUMMARY_SHA",
    )
    require(
        sha256_file(strong_path) == STRONG_INDICES_SHA256,
        "STRONG_INDICES_FILE_SHA",
    )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8-sig")
    )
    strong = json.loads(
        strong_path.read_text(encoding="utf-8-sig")
    )

    require(
        summary["result"] == "PASS_MAMBA28B_GEOMETRY_PREPARATION",
        "GEOMETRY_RESULT",
    )
    require(
        summary["xg1_discovery_accessed"] is False
        and summary["xg1_confirmation_accessed"] is False
        and summary["causal_response_observed"] is False
        and summary["plane_selection_performed"] is False,
        "GEOMETRY_RESPONSE_BOUNDARY",
    )
    require(
        summary["layer_mapping"]["source_block"] == 45
        and summary["layer_mapping"]["target_residual_layer"] == 46
        and summary["layer_mapping"]["intervention_layer"] == 47,
        "GEOMETRY_LAYER_MAPPING",
    )

    indices = [int(value) for value in strong["strong_indices"]]
    require(len(indices) == DIM, "FROZEN_STRONG_COUNT")
    require(indices == sorted(indices), "FROZEN_STRONG_ORDER")
    require(len(set(indices)) == DIM, "FROZEN_STRONG_DUPLICATE")
    require(
        geom.strong_index_sha256(indices)
        == "ea1b35e632f9b4bf3a16f781a86278d250c0ce2e734b2996ef8dc75233a26f50",
        "FROZEN_STRONG_HASH",
    )

    mask = torch.zeros(
        geom.INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    mask[torch.tensor(indices, dtype=torch.long)] = True
    require(int(mask.sum().item()) == DIM, "FROZEN_STRONG_MASK")

    bases = {
        "xg2": _load_f64(
            GEOMETRY_ROOT / "xg2_basis.f64le",
            (DIM, K),
            XG2_BASIS_SHA256,
        ),
        "xg4": _load_f64(
            GEOMETRY_ROOT / "xg4_basis.f64le",
            (DIM, K),
            XG4_BASIS_SHA256,
        ),
    }

    planes: dict[str, dict[str, torch.Tensor]] = {}
    for plane in PLANE_ORDER:
        number = plane[1:]
        plus = _load_f64(
            GEOMETRY_ROOT / f"p{number}_plus.f64le",
            (DIM,),
            PLANE_FILE_SHA256[plane]["plus"],
        )
        minus = _load_f64(
            GEOMETRY_ROOT / f"p{number}_minus.f64le",
            (DIM,),
            PLANE_FILE_SHA256[plane]["minus"],
        )
        require(
            abs(float(torch.linalg.vector_norm(plus).item()) - 1.0)
            <= PLANE_TOL,
            f"PLANE_PLUS_NORM:{plane}",
        )
        require(
            abs(float(torch.linalg.vector_norm(minus).item()) - 1.0)
            <= PLANE_TOL,
            f"PLANE_MINUS_NORM:{plane}",
        )
        require(
            abs(float(torch.dot(plus, minus).item())) <= PLANE_TOL,
            f"PLANE_ORTHOGONALITY:{plane}",
        )
        planes[plane] = {
            "plus": plus,
            "minus": minus,
        }

    lambda_plus = {
        plane: float(value)
        for plane, value in zip(
            PLANE_ORDER,
            summary["lambda_plus_by_plane"],
            strict=True,
        )
    }
    require(
        all(
            math.isfinite(lambda_plus[plane])
            and lambda_plus[plane] > 0.0
            for plane in PLANE_ORDER
        ),
        "LAMBDA_PLUS",
    )

    return {
        "summary": summary,
        "strong_indices": indices,
        "strong_mask": mask,
        "bases": bases,
        "planes": planes,
        "lambda_plus": lambda_plus,
    }


def runtime_gate_single_visible_gpu(physical_device: int) -> torch.device:
    require(physical_device in (0, 1), "PHYSICAL_DEVICE")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(
        visible == str(physical_device),
        f"CUDA_VISIBLE_DEVICES:{visible}",
    )

    import transformers

    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    require(
        observed == backend.EXPECTED_RUNTIME,
        f"RUNTIME_MISMATCH:{observed}",
    )
    require(
        kernel_compat._kernel_package_version() == backend.KERNELS_VERSION,
        "KERNELS_VERSION",
    )
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() == 1,
        f"LOGICAL_CUDA_DEVICE_COUNT:{torch.cuda.device_count()}",
    )
    require(torch.version.cuda == backend.EXPECTED_CUDA_RUNTIME, "CUDA_RUNTIME")
    require(
        torch.cuda.get_device_name(0) == backend.EXPECTED_DEVICE_NAME,
        f"CUDA_DEVICE_NAME:{torch.cuda.get_device_name(0)}",
    )
    require(
        tuple(torch.cuda.get_device_capability(0))
        == backend.EXPECTED_CAPABILITY,
        "CUDA_CAPABILITY",
    )

    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def validate_runtime_geometry(
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> None:
    mask = runtime_ctx["strong_mask"].detach().cpu().bool().contiguous()
    observed = torch.nonzero(
        mask,
        as_tuple=False,
    ).flatten().tolist()
    require(
        observed == frozen["strong_indices"],
        "RUNTIME_STRONG_INDICES",
    )
    require(
        geom.strong_index_sha256(observed)
        == "ea1b35e632f9b4bf3a16f781a86278d250c0ce2e734b2996ef8dc75233a26f50",
        "RUNTIME_STRONG_HASH",
    )


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, rows = load_discovery_population()
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
            require(bool(event["post4_eligible"]), f"ANCHOR_INELIGIBLE:{key}")
            events[key] = dict(event)

    required_cells = (
        TARGET_PLUS_CELL,
        TARGET_MINUS_CELL,
        REFERENCE_PLUS_CELL,
        REFERENCE_MINUS_CELL,
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


def input_row(
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    pair: str,
    cell: str,
) -> torch.Tensor:
    key = (pair, cell)
    require(key in lookup, f"MISSING_INPUT:{key}")
    index = lookup[key]
    return (
        encoded["input_ids"][index:index + 1]
        .detach().cpu().contiguous()
    )


def anchors_for_pair(
    pair: str,
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, int]:
    cells = {
        "tp": TARGET_PLUS_CELL,
        "tm": TARGET_MINUS_CELL,
        "rp": REFERENCE_PLUS_CELL,
        "rm": REFERENCE_MINUS_CELL,
    }
    out = {
        role: int(
            events[(pair, cell, ANCHOR_NAME)][
                "absolute_anchor_token_index"
            ]
        )
        for role, cell in cells.items()
    }
    return out


def plane_component(
    h: torch.Tensor,
    plane: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    require(plane in PLANE_ORDER, f"PLANE:{plane}")
    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (DIM,), f"H_SHAPE:{tuple(value.shape)}")
    require(bool(torch.isfinite(value).all().item()), "H_NONFINITE")

    plus = planes[plane]["plus"]
    minus = planes[plane]["minus"]
    a = float(torch.dot(value, plus))
    b = float(torch.dot(value, minus))
    component = (a * plus + b * minus).contiguous()

    return {
        "a": a,
        "b": b,
        "component": component,
        "component_l2": float(torch.linalg.vector_norm(component).item()),
    }


def condition_correction(
    h: torch.Tensor,
    *,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    value = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(value.shape) == (DIM,), "CONDITION_H_SHAPE")

    if condition == "restored":
        # The restored state is explicitly constructed as neutralize + restore
        # for every plane. The net correction must be identically zero.
        round_trip = torch.zeros_like(value)
        max_round_trip = 0.0
        native_coefficients: dict[str, list[float]] = {}
        for plane in PLANE_ORDER:
            info = plane_component(value, plane, planes)
            component = info["component"]
            plane_round_trip = (-component + component).contiguous()
            max_round_trip = max(
                max_round_trip,
                float(torch.max(torch.abs(plane_round_trip)).item()),
            )
            round_trip += plane_round_trip
            native_coefficients[plane] = [
                float(info["a"]),
                float(info["b"]),
            ]
        require(max_round_trip <= TOL, f"RESTORATION_ROUND_TRIP:{max_round_trip}")
        correction = round_trip.contiguous()
        selected_plane = None
        neutral_residual = None
    else:
        require(
            condition.endswith("_neutralized"),
            f"CONDITION:{condition}",
        )
        selected_plane = condition[:2].upper()
        require(selected_plane in PLANE_ORDER, f"CONDITION_PLANE:{condition}")
        info = plane_component(value, selected_plane, planes)
        correction = (-info["component"]).contiguous()
        post = value + correction
        plus = planes[selected_plane]["plus"]
        minus = planes[selected_plane]["minus"]
        neutral_residual = max(
            abs(float(torch.dot(post, plus).item())),
            abs(float(torch.dot(post, minus).item())),
        )
        require(
            neutral_residual <= TOL,
            f"NEUTRALIZATION:{selected_plane}:{neutral_residual}",
        )
        native_coefficients = {
            selected_plane: [
                float(info["a"]),
                float(info["b"]),
            ]
        }
        max_round_trip = None

    require(
        bool(torch.isfinite(correction).all().item()),
        "CORRECTION_NONFINITE",
    )

    return {
        "condition": condition,
        "selected_plane": selected_plane,
        "correction": correction,
        "correction_l2": float(
            torch.linalg.vector_norm(correction).item()
        ),
        "native_coefficients": native_coefficients,
        "neutralization_max_abs_projection": neutral_residual,
        "restoration_round_trip_max_abs": max_round_trip,
    }


def install_probe_hook(
    mixer: Any,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
    audit: dict[str, Any],
):
    require(orientation in (-1, 1), "ORIENTATION")
    require(branch_sign in (-1, 1), "BRANCH_SIGN")
    vector = direction.detach().cpu().to(torch.float64).contiguous()
    require(tuple(vector.shape) == (DIM,), "DIRECTION_SHAPE")
    require(
        abs(float(torch.linalg.vector_norm(vector).item()) - 1.0)
        <= PLANE_TOL,
        "DIRECTION_NORM",
    )

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * geom.INTERMEDIATE_SIZE,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")

        before = output.detach().clone()
        mask = strong_mask.detach().cpu().bool().contiguous()
        require(
            mask.numel() == geom.INTERMEDIATE_SIZE
            and int(mask.sum().item()) == DIM,
            "STRONG_MASK",
        )
        mask_device = mask.to(before.device)

        h = (
            before[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
            .detach().cpu().to(torch.float64).contiguous()
        )

        info = condition_correction(
            h,
            condition=condition,
            planes=planes,
        )
        probe = vector * (
            float(branch_sign)
            * float(orientation)
            * EPS
        )
        total = (info["correction"] + probe).contiguous()

        out = output.clone()
        intended = total.to(
            device=out.device,
            dtype=out.dtype,
        )
        out[
            0,
            token_index,
            :geom.INTERMEDIATE_SIZE,
        ][mask_device] += intended

        require(
            torch.equal(
                out[:, :, geom.INTERMEDIATE_SIZE:],
                before[:, :, geom.INTERMEDIATE_SIZE:],
            ),
            "GATE_CHANGED",
        )
        require(
            torch.equal(
                out[:, :, :geom.INTERMEDIATE_SIZE][
                    :, :, ~mask_device
                ],
                before[:, :, :geom.INTERMEDIATE_SIZE][
                    :, :, ~mask_device
                ],
            ),
            "NONSTRONG_CHANGED",
        )
        if token_index:
            require(
                torch.equal(
                    out[:, :token_index, :],
                    before[:, :token_index, :],
                ),
                "EARLIER_TOKEN_CHANGED",
            )
        if token_index + 1 < out.shape[1]:
            require(
                torch.equal(
                    out[:, token_index + 1:, :],
                    before[:, token_index + 1:, :],
                ),
                "LATER_TOKEN_CHANGED",
            )

        applied = (
            out[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
            - before[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
        ).detach().cpu().to(torch.float64)

        residual = float(
            torch.max(
                torch.abs(
                    applied
                    - intended.detach().cpu().to(torch.float64)
                )
            ).item()
        )
        require(
            residual <= transport_runtime.RUNTIME_CAST_TOL,
            f"APPLIED_RESIDUAL:{residual}",
        )

        audit.clear()
        audit.update({
            "condition": condition,
            "selected_plane": info["selected_plane"],
            "orientation": int(orientation),
            "branch_sign": int(branch_sign),
            "token_index": int(token_index),
            "condition_correction_l2": float(info["correction_l2"]),
            "probe_correction_l2": float(
                torch.linalg.vector_norm(probe).item()
            ),
            "neutralization_max_abs_projection":
                info["neutralization_max_abs_projection"],
            "restoration_round_trip_max_abs":
                info["restoration_round_trip_max_abs"],
            "applied_correction_max_abs_residual": residual,
        })
        return out

    return mixer.in_proj.register_forward_hook(hook)


def flatten_state(state: torch.Tensor) -> np.ndarray:
    out = (
        state.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .reshape(-1)
        .copy()
    )
    require(out.size == STATE_WIDTH, f"STATE_WIDTH:{out.size}")
    require(bool(np.isfinite(out).all()), "STATE_NONFINITE")
    return out


class ForwardBudget:
    def __init__(self, expected: int) -> None:
        self.expected = int(expected)
        self.count = 0

    def consume(self) -> None:
        require(self.count < self.expected, "FORWARD_BUDGET_OVERFLOW")
        self.count += 1

    def assert_exact(self) -> None:
        require(
            self.count == self.expected,
            f"FORWARD_BUDGET:{self.count}:{self.expected}",
        )


def capture_branch(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    input_ids: torch.Tensor,
    anchor: int,
    device: torch.device,
    budget: ForwardBudget,
    condition: str,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import transformers.models.mamba.modeling_mamba as mm

    mixer = runtime_ctx["intervention_mixer"]
    target_abs = int(anchor) + geom.TARGET_OFFSET
    require(tuple(input_ids.shape) == (1, 128), "INPUT_SHAPE")

    intervention_audit: dict[str, Any] = {}
    handle = install_probe_hook(
        mixer,
        token_index=target_abs,
        strong_mask=runtime_ctx["strong_mask"],
        condition=condition,
        planes=planes,
        direction=direction,
        orientation=orientation,
        branch_sign=branch_sign,
        audit=intervention_audit,
    )

    kernel_scan = kernels["selective_scan_fn"]
    kernel_update = kernels["selective_state_update"]

    active = {"value": False}
    captured: list[tuple[torch.Tensor, ...]] = []
    fast_calls = {"count": 0}

    def scan_wrapper(*args, **kwargs):
        if active["value"]:
            require(len(captured) == 0, "SCAN_CAPTURE_DUPLICATE")
            require(len(args) >= 8, "SCAN_ARG_COUNT")
            captured.append(
                tuple(
                    value.detach().clone()
                    for value in args[:8]
                )
            )
        return kernel_scan(*args, **kwargs)

    original_scan = mm.selective_scan_fn
    original_cuda = mixer.cuda_kernels_forward

    def cuda_wrapper(
        _self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        require(not active["value"], "CUDA_FAST_REENTRY")
        fast_calls["count"] += 1
        active["value"] = True
        try:
            return original_cuda(
                hidden_states,
                cache_params,
                cache_position,
                attention_mask,
            )
        finally:
            active["value"] = False

    mm.selective_scan_fn = scan_wrapper
    mixer.cuda_kernels_forward = types.MethodType(
        cuda_wrapper,
        mixer,
    )

    budget.consume()
    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().to(device).contiguous()
            )
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
        mm.selective_scan_fn = original_scan
        if "cuda_kernels_forward" in mixer.__dict__:
            del mixer.__dict__["cuda_kernels_forward"]

    require(bool(intervention_audit), "INTERVENTION_AUDIT_MISSING")
    require(fast_calls["count"] == 1, f"CUDA_FAST_PATH_CALLS:{fast_calls['count']}")
    require(len(captured) == 1, f"SCAN_CAPTURE_COUNT:{len(captured)}")

    (
        u,
        delta,
        a_matrix,
        b_scan,
        c_scan,
        d_vector,
        gate,
        delta_bias,
    ) = captured[0]

    prefix_end = int(anchor) + 1
    require(prefix_end + 4 <= u.shape[-1], "POST4_RANGE")

    _, state = kernel_scan(
        u[..., :prefix_end].contiguous(),
        delta[..., :prefix_end].contiguous(),
        a_matrix,
        b_scan[..., :prefix_end].contiguous(),
        c_scan[..., :prefix_end].contiguous(),
        d_vector,
        gate[..., :prefix_end].contiguous(),
        delta_bias,
        delta_softplus=True,
        return_last_state=True,
    )

    window = [flatten_state(state)]

    for token in range(prefix_end, prefix_end + 4):
        _ = kernel_update(
            state,
            u[..., token],
            delta[..., token],
            a_matrix,
            b_scan[..., token],
            c_scan[..., token],
            d_vector,
            gate[..., token],
            delta_bias,
            dt_softplus=True,
        )
        window.append(flatten_state(state))

    filler = window[0]
    states = [
        filler.copy()
        for _ in range(int(input_ids.shape[1]))
    ]
    for offset, vector in enumerate(window):
        states[int(anchor) + offset] = vector

    path_efficiency = measurement.post4_path_efficiency(
        states,
        int(anchor),
    )
    require(math.isfinite(path_efficiency), "PATH_EFFICIENCY_NONFINITE")

    return {
        "anchor": int(anchor),
        "target_abs": target_abs,
        "path_efficiency": float(path_efficiency),
        "fast_path_calls": 1,
    }, intervention_audit


def run_signed(
    *,
    pair: str,
    condition: str,
    direction: torch.Tensor,
    orientation: int,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    kernels: Mapping[str, Any],
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    device: torch.device,
    budget: ForwardBudget,
) -> dict[str, Any]:
    anchors = anchors_for_pair(pair, events)
    cells = {
        "tp": TARGET_PLUS_CELL,
        "tm": TARGET_MINUS_CELL,
    }
    efficiencies: dict[str, float] = {}
    audits: list[dict[str, Any]] = []

    for role, branch_sign in (("tp", 1), ("tm", -1)):
        branch, audit = capture_branch(
            model=model,
            runtime_ctx=runtime_ctx,
            kernels=kernels,
            input_ids=input_row(
                encoded,
                lookup,
                pair,
                cells[role],
            ),
            anchor=anchors[role],
            device=device,
            budget=budget,
            condition=condition,
            planes=planes,
            direction=direction,
            orientation=orientation,
            branch_sign=branch_sign,
        )
        efficiencies[role] = float(branch["path_efficiency"])
        audits.append(audit)

    value = efficiencies["tp"] - efficiencies["tm"]
    require(math.isfinite(value), "F_NONFINITE")

    return {
        "F": value,
        "plus_path_efficiency": efficiencies["tp"],
        "minus_path_efficiency": efficiencies["tm"],
        "max_applied_residual": max(
            float(a["applied_correction_max_abs_residual"])
            for a in audits
        ),
        "max_neutralization_residual": max(
            (
                float(a["neutralization_max_abs_projection"])
                for a in audits
                if a["neutralization_max_abs_projection"] is not None
            ),
            default=0.0,
        ),
        "max_restoration_round_trip": max(
            (
                float(a["restoration_round_trip_max_abs"])
                for a in audits
                if a["restoration_round_trip_max_abs"] is not None
            ),
            default=0.0,
        ),
        "model_forward_count": FORWARDS_PER_SIGNED,
    }


def run_condition(
    *,
    pair: str,
    condition: str,
    bases: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    max_applied = 0.0
    max_neutral = 0.0
    max_round_trip = 0.0

    for family in ("xg2", "xg4"):
        for index in range(K):
            direction = bases[family][:, index]
            positive = run_signed(
                pair=pair,
                condition=condition,
                direction=direction,
                orientation=1,
                **kwargs,
            )
            negative = run_signed(
                pair=pair,
                condition=condition,
                direction=direction,
                orientation=-1,
                **kwargs,
            )

            f_plus = float(positive["F"])
            f_minus = float(negative["F"])
            j_value = (f_plus - f_minus) / (2.0 * EPS)
            require(math.isfinite(j_value), "J_NONFINITE")

            probes.append({
                "direction_key": f"{family}_{index}",
                "basis_family": family,
                "basis_index": index,
                "F_plus": f_plus,
                "F_minus": f_minus,
                "J": j_value,
                "J_squared": j_value * j_value,
            })
            max_applied = max(
                max_applied,
                float(positive["max_applied_residual"]),
                float(negative["max_applied_residual"]),
            )
            max_neutral = max(
                max_neutral,
                float(positive["max_neutralization_residual"]),
                float(negative["max_neutralization_residual"]),
            )
            max_round_trip = max(
                max_round_trip,
                float(positive["max_restoration_round_trip"]),
                float(negative["max_restoration_round_trip"]),
            )

    require(
        [probe["direction_key"] for probe in probes]
        == list(DIRECTION_ORDER),
        "DIRECTION_ORDER",
    )

    e_xg2 = math.fsum(
        float(probe["J_squared"])
        for probe in probes[:K]
    ) / K
    e_xg4 = math.fsum(
        float(probe["J_squared"])
        for probe in probes[K:]
    ) / K
    q_value = e_xg2 - e_xg4
    require(
        all(math.isfinite(value) for value in (e_xg2, e_xg4, q_value)),
        "Q_NONFINITE",
    )

    return {
        "condition": condition,
        "direction_order": list(DIRECTION_ORDER),
        "direction_probes": probes,
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": q_value,
        "max_applied_correction_residual": max_applied,
        "max_neutralization_projection": max_neutral,
        "max_restoration_round_trip": max_round_trip,
        "model_forward_count": FORWARDS_PER_CONDITION,
    }


def run_pair(
    *,
    pair_index: int,
    pair: str,
    bases: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    require(pair == PAIR_IDS[pair_index], f"PAIR:{pair_index}:{pair}")

    restored = run_condition(
        pair=pair,
        condition="restored",
        bases=bases,
        **kwargs,
    )
    neutralized: dict[str, dict[str, Any]] = {}
    for plane in PLANE_ORDER:
        neutralized[plane] = run_condition(
            pair=pair,
            condition=f"{plane.lower()}_neutralized",
            bases=bases,
            **kwargs,
        )

    q_restored = float(restored["Q"])
    plane_effects: dict[str, dict[str, float]] = {}
    for plane in PLANE_ORDER:
        q_neutralized = float(neutralized[plane]["Q"])
        effect = q_restored - q_neutralized
        require(math.isfinite(effect), f"S_NONFINITE:{plane}")
        plane_effects[plane] = {
            "Q_restored": q_restored,
            "Q_neutralized": q_neutralized,
            "S": effect,
        }

    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "pair_index": pair_index,
        "epsilon": EPS,
        "plane_order": list(PLANE_ORDER),
        "direction_order": list(DIRECTION_ORDER),
        "restored_condition": restored,
        "neutralized_conditions": neutralized,
        "plane_effects": plane_effects,
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "selection_performed": False,
        "confirmation_accessed": False,
        "inferential_test_performed": False,
        "positive_gate_applied": False,
        "rescue_performed": False,
    }


def descriptive_stats(values: Sequence[float]) -> dict[str, float]:
    require(len(values) == PAIR_COUNT, "DESCRIPTIVE_N")
    array = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(array).all()), "DESCRIPTIVE_NONFINITE")
    mean = float(math.fsum(float(v) for v in values) / len(values))
    sd = float(np.std(array, ddof=1))
    se = sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "ci95_normal_low": mean - 1.96 * se,
        "ci95_normal_high": mean + 1.96 * se,
        "fraction_positive": float(np.mean(array > 0.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def select_dominant(
    items: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, float]]]:
    require(len(items) == PAIR_COUNT, "SELECTION_ITEM_COUNT")
    stats: dict[str, dict[str, float]] = {}

    for plane in PLANE_ORDER:
        values = [
            float(item["plane_effects"][plane]["S"])
            for item in items
        ]
        stats[plane] = descriptive_stats(values)

    means = {
        plane: float(stats[plane]["mean"])
        for plane in PLANE_ORDER
    }
    best = max(means.values())
    winners = [
        plane
        for plane in PLANE_ORDER
        if means[plane] == best
    ]
    require(
        len(winners) == 1,
        f"SELECTION_NOT_UNIQUE:{winners}",
    )
    return winners[0], stats


def select_response_blind_control(
    selected_plane: str,
    lambda_plus: Mapping[str, float],
) -> str:
    require(selected_plane in PLANE_ORDER, "SELECTED_PLANE")
    candidates = [
        plane
        for plane in PLANE_ORDER
        if plane != selected_plane
    ]
    values = {
        plane: float(lambda_plus[plane])
        for plane in candidates
    }
    require(
        all(math.isfinite(value) for value in values.values()),
        "CONTROL_LAMBDA_NONFINITE",
    )
    best = max(values.values())
    winners = [
        plane
        for plane in candidates
        if values[plane] == best
    ]
    require(
        len(winners) == 1,
        f"CONTROL_SELECTION_NOT_UNIQUE:{winners}",
    )
    return winners[0]


def _worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "items": temp_dir / f"shard_{shard_id}_items.jsonl",
        "meta": temp_dir / f"shard_{shard_id}_meta.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = _worker_paths(Path(temp_dir), shard_id)

    try:
        # Each worker exposes exactly one physical T4 as logical cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

        validate_protocol()
        authenticate_repo(expected_head)
        device = runtime_gate_single_visible_gpu(physical_device)

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        frozen = load_frozen_geometry()
        rows, encoded, events = build_input_state(snapshot)
        lookup = row_index(rows)

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        kernel_compat.validate_transformers_kernel_bindings(kernels)

        runtime_ctx = geom.runtime_components(model)
        validate_runtime_geometry(runtime_ctx, frozen)

        budget = ForwardBudget(int(shard["forward_budget"]))
        items: list[dict[str, Any]] = []

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[global_index]
            items.append(
                run_pair(
                    pair_index=global_index,
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
            )

        budget.assert_exact()
        torch.cuda.synchronize(device)

        require(len(items) == int(shard["pair_count"]), "WORKER_ITEM_COUNT")
        require(
            items[0]["source_pair_id"] == shard["pair_first"]
            and items[-1]["source_pair_id"] == shard["pair_last"],
            "WORKER_PAIR_RANGE",
        )

        paths["items"].write_bytes(jsonl_bytes(items))
        meta = {
            "schema_version": "gen4-mamba28b-discovery-worker-v1",
            "shard_id": shard_id,
            "physical_device": physical_device,
            "logical_device": 0,
            "cuda_visible_devices": str(physical_device),
            "device_name": torch.cuda.get_device_name(0),
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": len(items),
            "scientific_model_forward_count":
                int(shard["forward_budget"]),
            "items_sha256": sha256_file(paths["items"]),
            "model_provenance": model_provenance,
            "strong_index_sha256":
                frozen["summary"]["strong_mask"][
                    "strong_index_sha256"
                ],
            "geometry_summary_sha256": GEOMETRY_SUMMARY_SHA256,
            "confirmation_accessed": False,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def merge_worker_items(
    temp_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []

    for shard in SHARDS:
        shard_id = int(shard["shard_id"])
        paths = _worker_paths(temp_dir, shard_id)
        require(paths["meta"].is_file(), f"WORKER_META_MISSING:{shard_id}")
        require(paths["items"].is_file(), f"WORKER_ITEMS_MISSING:{shard_id}")

        meta = json.loads(
            paths["meta"].read_text(encoding="utf-8")
        )
        require(meta["shard_id"] == shard_id, "MERGE_SHARD_ID")
        require(
            meta["physical_device"] == shard["physical_device"],
            "MERGE_PHYSICAL_DEVICE",
        )
        require(meta["logical_device"] == 0, "MERGE_LOGICAL_DEVICE")
        require(
            meta["pair_count"] == shard["pair_count"],
            "MERGE_PAIR_COUNT",
        )
        require(
            meta["scientific_model_forward_count"]
            == shard["forward_budget"],
            "MERGE_FORWARD_COUNT",
        )
        require(
            meta["items_sha256"] == sha256_file(paths["items"]),
            "MERGE_ITEMS_SHA",
        )
        require(meta["confirmation_accessed"] is False, "MERGE_CONFIRMATION")

        shard_items = _read_jsonl(paths["items"])
        require(
            len(shard_items) == shard["pair_count"],
            "MERGE_SHARD_ITEM_COUNT",
        )
        items.extend(shard_items)
        metas.append(meta)

    require(len(items) == PAIR_COUNT, "MERGED_ITEM_COUNT")
    require(
        [str(item["source_pair_id"]) for item in items]
        == list(PAIR_IDS),
        "MERGED_PAIR_ORDER",
    )
    require(
        sum(
            int(item["scientific_model_forward_count_this_run"])
            for item in items
        )
        == TOTAL_FORWARD_BUDGET,
        "MERGED_FORWARD_SUM",
    )
    return items, metas


def write_outputs(
    *,
    output_dir: Path,
    expected_head: str,
    items: Sequence[Mapping[str, Any]],
    worker_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    frozen = load_frozen_geometry()
    selected_plane, stats = select_dominant(items)
    control_plane = select_response_blind_control(
        selected_plane,
        frozen["lambda_plus"],
    )

    selection = {
        "schema_version": SELECTION_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "geometry_freeze_commit": GEOMETRY_FREEZE_COMMIT,
        "discovery_holdout_freeze_commit": DISCOVERY_HOLDOUT_FREEZE_COMMIT,
        "geometry_summary_sha256": GEOMETRY_SUMMARY_SHA256,
        "discovery_population": {
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "source_pair_count": PAIR_COUNT,
            "source_sha256": DISCOVERY_SOURCE_SHA256,
            "rows_sha256": DISCOVERY_ROWS_SHA256,
            "structural_manifest_sha256": DISCOVERY_MANIFEST_SHA256,
        },
        "epsilon": EPS,
        "selection_definition":
            "k*=argmax_k mean_i(Q_restored,k,i-Q_neutralized,k,i)",
        "restored_semantics":
            "branch-local plane neutralize-plus-exact-restore round trip; net native state",
        "plane_order": list(PLANE_ORDER),
        "per_plane_descriptive_effects": stats,
        "selected_dominant_candidate": selected_plane,
        "selection_unique": True,
        "positivity_gate_applied": False,
        "control_definition":
            "c*=argmax_{k!=k*} lambda_plus_k from frozen geometry only",
        "lambda_plus_by_plane": frozen["lambda_plus"],
        "response_blind_control_plane": control_plane,
        "control_selection_uses_response": False,
        "confirmation_accessed": False,
        "confirmation_response_accessed": False,
        "inferential_test_performed": False,
        "scientific_conclusion_established": False,
        "rescue_performed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / ITEM_FILE).write_bytes(
        jsonl_bytes(items)
    )
    (output_dir / SELECTION_FILE).write_bytes(
        pretty_json_bytes(selection)
    )

    file_hashes = {
        ITEM_FILE: sha256_file(output_dir / ITEM_FILE),
        SELECTION_FILE: sha256_file(output_dir / SELECTION_FILE),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "geometry_freeze_commit": GEOMETRY_FREEZE_COMMIT,
        "discovery_holdout_freeze_commit": DISCOVERY_HOLDOUT_FREEZE_COMMIT,
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "worker_count": GPU_COUNT,
        "workers": list(worker_meta),
        "output_file_sha256": dict(sorted(file_hashes.items())),
        "selected_dominant_candidate": selected_plane,
        "response_blind_control_plane": control_plane,
        "selection_unique": True,
        "positive_gate_applied": False,
        "confirmation_accessed": False,
        "inferential_test_performed": False,
        "training_executed": False,
        "backward_executed": False,
        "rescue_performed": False,
    }
    (output_dir / MANIFEST_FILE).write_bytes(
        pretty_json_bytes(manifest)
    )
    file_hashes[MANIFEST_FILE] = sha256_file(
        output_dir / MANIFEST_FILE
    )

    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(file_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    return selection


def run_discovery(
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
        sha256_file(compact_checkpoint)
        == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    # Parent sees both physical GPUs. Workers remap independently to one
    # logical cuda:0 each using CUDA_VISIBLE_DEVICES.
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba28b_discovery_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard": shard,
                    "expected_head": expected_head,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=(
                    "mamba28b-discovery-"
                    f"shard{shard['shard_id']}"
                ),
            )
            process.start()
            processes.append(process)

        for shard, process in zip(
            SHARDS,
            processes,
            strict=True,
        ):
            process.join()
            if process.exitcode != 0:
                paths = _worker_paths(
                    temp_dir,
                    int(shard["shard_id"]),
                )
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise DiscoveryError(
                    "WORKER_FAILED:"
                    f"{shard['shard_id']}:\n{detail}"
                )

        items, worker_meta = merge_worker_items(temp_dir)
        return write_outputs(
            output_dir=output_dir,
            expected_head=expected_head,
            items=items,
            worker_meta=worker_meta,
        )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mamba-2.8B prospective discovery on XG1 6001..6300. "
            "Selects exactly one dominant candidate from response; "
            "confirmation 6301..6600 is not accessed."
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


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)
    selection = run_discovery(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT=" + str(selection["result"]))
    print(
        "SELECTED_DOMINANT_CANDIDATE="
        + str(selection["selected_dominant_candidate"])
    )
    print(
        "RESPONSE_BLIND_CONTROL_PLANE="
        + str(selection["response_blind_control_plane"])
    )
    for plane in PLANE_ORDER:
        stats = selection["per_plane_descriptive_effects"][plane]
        print(
            f"{plane}_MEAN_S="
            + format(float(stats["mean"]), ".17g")
        )
    print("SELECTION_UNIQUE=True")
    print("POSITIVITY_GATE_APPLIED=False")
    print("CONTROL_SELECTION_USES_RESPONSE=False")
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(TOTAL_FORWARD_BUDGET)
    )
    print("DISCOVERY_RANGE=xg1_fact_6001..xg1_fact_6300")
    print("CONFIRMATION_ACCESSED=False")
    print("INFERENTIAL_TEST_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION_ESTABLISHED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("RESCUE_PERFORMED=False")


if __name__ == "__main__":
    main()
