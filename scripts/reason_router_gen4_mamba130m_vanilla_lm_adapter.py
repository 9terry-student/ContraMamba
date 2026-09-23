from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)
from scripts import (
    reason_router_gen4_seed181_behavioral_restoration_bridge_fast_cuda
    as bridge,
)


ROOT = Path(__file__).resolve().parents[1]

HF_REPO = "state-spaces/mamba-130m-hf"
HF_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"

HF_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

MODEL_SAFETENSORS_SHA256 = (
    "1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea"
)

HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
LAYER_COUNT = 24
INTERVENTION_LAYER = 17
TARGET_OFFSET = 2

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
DIM = 395

PAIR_IDS = tuple(
    f"xg1_fact_{i:03d}"
    for i in range(2701, 3001)
)

TOL = 1.0e-12

GEOMETRY_JSON_SHA256 = (
    "e6e9db909eb7d2c6bbdb493a4efeca8c18e4d474cf99a943be0f3f7b9dee1012"
)
GEOMETRY_PT_SHA256 = (
    "de3ae6a450c2ba0a85b4f53919e3765e6e7dfb6dba3676535554c437b1647a1c"
)
GEOMETRY_SOURCE_CHECKPOINT_SHA256 = (
    "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
)

EXPECTED_STRONG_INDEX_SHA256 = (
    "6950bb6c6cc777375f5e4ce18f22fd3272d7b80c5aff0726c25a6ec77d5813ce"
)
EXPECTED_LAYER17_MU_K2 = 0.027899337798707836


class Mamba130VanillaLMAdapterError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Mamba130VanillaLMAdapterError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_fast_runtime_for_device(gpu_id: int) -> None:
    require(gpu_id in (0, 1), f"GPU_ID:{gpu_id}")
    runtime = (
        bridge.restoration
        .holdout.phase1.base
        .prevalence_eq
    )
    runtime.backend.runtime_gate()
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        gpu_id < torch.cuda.device_count(),
        f"CUDA_DEVICE_COUNT:{torch.cuda.device_count()}",
    )


def validate_snapshot(snapshot: Path) -> dict[str, dict[str, Any]]:
    require(snapshot.is_dir(), f"SNAPSHOT_MISSING:{snapshot}")
    require(
        snapshot.name == HF_REVISION,
        f"SNAPSHOT_REVISION_PATH:{snapshot.name}",
    )

    observed: dict[str, dict[str, Any]] = {}

    for name in HF_FILES:
        path = snapshot / name
        require(path.is_file(), f"SNAPSHOT_FILE_MISSING:{name}")
        observed[name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    require(
        observed["model.safetensors"]["sha256"]
        == MODEL_SAFETENSORS_SHA256,
        "MODEL_SAFETENSORS_SHA256",
    )

    config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8")
    )

    exact_config = {
        "architectures": ["MambaForCausalLM"],
        "conv_kernel": 4,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": LAYER_COUNT,
        "state_size": STATE_SIZE,
        "vocab_size": 50280,
    }

    for key, expected in exact_config.items():
        require(
            config.get(key) == expected,
            f"CONFIG:{key}:{config.get(key)!r}",
        )

    require(
        "tie_word_embeddings" not in config,
        "HF_TIE_METADATA_UNEXPECTED",
    )

    return observed


def strong_partition(conv_weight: torch.Tensor) -> dict[str, Any]:
    partition = core.strong_partition(conv_weight)
    core.validate_frozen_layer17_partition(partition)

    require(
        partition["strong_index_sha256"]
        == EXPECTED_STRONG_INDEX_SHA256,
        "STRONG_INDEX_SHA256",
    )
    require(
        abs(
            float(partition["mu_k2"])
            - EXPECTED_LAYER17_MU_K2
        )
        <= 1e-15,
        "LAYER17_MU_K2",
    )

    mask = torch.zeros(
        INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    mask[
        partition["strong"]
        .detach()
        .cpu()
        .long()
    ] = True

    require(
        int(mask.sum().item()) == DIM,
        "STRONG_MASK_COUNT",
    )

    return {
        **partition,
        "mask": mask.contiguous(),
    }


def build_input_state(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    facts, all_rows, _manifest = (
        bridge.validate_fresh_data()
    )
    selected_rows, facts_by_id = (
        bridge.validate_label_semantics(
            facts,
            all_rows,
        )
    )
    events, encoded, _tokenizer_provenance = (
        bridge.build_anchor_and_encoding(
            selected_rows,
            facts_by_id,
            snapshot,
        )
    )

    require(
        len(selected_rows) == 600,
        "ROW_COUNT",
    )

    return (
        selected_rows,
        encoded,
        events,
    )


def row_index(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        ): index
        for index, row in enumerate(rows)
    }

    require(
        len(out) == 600,
        "ROW_INDEX_COUNT",
    )

    expected = {
        (pair, cell)
        for pair in PAIR_IDS
        for cell in (
            "C0_SHAM",
            "C2_NAME",
        )
    }
    require(
        set(out) == expected,
        "ROW_INDEX_COVERAGE",
    )

    return out


def load_frozen_geometry() -> dict[str, Any]:
    aliases = bridge.load_seed181_planes()

    planes = {
        "P3": {
            "plus":
                aliases[
                    "pp3_plus"
                ].detach().cpu().to(
                    torch.float64
                ).contiguous(),
            "minus":
                aliases[
                    "pp3_minus"
                ].detach().cpu().to(
                    torch.float64
                ).contiguous(),
        },
        "P5": {
            "plus":
                aliases[
                    "pp5_plus"
                ].detach().cpu().to(
                    torch.float64
                ).contiguous(),
            "minus":
                aliases[
                    "pp5_minus"
                ].detach().cpu().to(
                    torch.float64
                ).contiguous(),
        },
    }

    for plane in ("P3", "P5"):
        for sign in ("plus", "minus"):
            value = planes[plane][sign]
            require(
                tuple(value.shape) == (DIM,),
                f"PLANE_SHAPE:{plane}:{sign}",
            )
            require(
                abs(
                    float(
                        torch.linalg.vector_norm(
                            value
                        ).item()
                    )
                    - 1.0
                )
                <= TOL,
                f"PLANE_NORM:{plane}:{sign}",
            )

    return {
        "planes": planes,
        "geometry_json_sha256":
            GEOMETRY_JSON_SHA256,
        "geometry_pt_sha256":
            GEOMETRY_PT_SHA256,
        "geometry_source_checkpoint_sha256":
            GEOMETRY_SOURCE_CHECKPOINT_SHA256,
    }
