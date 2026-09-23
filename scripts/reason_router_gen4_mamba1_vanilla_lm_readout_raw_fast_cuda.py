#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba1-five-scale-ladder-extension"

PLAN_FREEZE_COMMIT = (
    "1fe9a198a15c9cea0e5451d918cd949bc21bf7e0"
)

SCALES = (
    "mamba370m",
    "mamba790m",
    "mamba14b",
    "mamba28b",
)

TARGET_CELLS = (
    "C0_SHAM",
    "C2_NAME",
)

ANCHOR_NAME = "A_IDENTITY"

COMMON_SELECTED_PLANE = "P3"
COMMON_CONTROL_PLANE = "P5"

GPU_COUNT = 2
PAIRS_PER_SHARD = 150
ITEMS_PER_SCALE = 600
FORWARDS_PER_SCALE = 600
BACKWARDS_PER_SCALE = 600

ITEM_FILE = "vanilla_lm_readout_items.jsonl"
SUMMARY_FILE = "raw_vanilla_lm_readout_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-mamba1-vanilla-lm-functional-readout-item-v1"
SUMMARY_SCHEMA = "gen4-mamba1-vanilla-lm-functional-readout-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba1-vanilla-lm-functional-readout-manifest-v1"

RESULT_PASS = "PASS_MAMBA1_VANILLA_LM_FUNCTIONAL_READOUT_RAW"


class VanillaLMReadoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise VanillaLMReadoutError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
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
    return b"".join(
        canonical_json_bytes(row)
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
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VanillaLMReadoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH:{branch}",
    )
    require(
        head == expected_head,
        f"HEAD:{head}",
    )
    require(
        status == "",
        "WORKTREE_NOT_CLEAN",
    )

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PLAN_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "PLAN_FREEZE_NOT_ANCESTOR",
    )


def scale_spec(scale: str) -> dict[str, Any]:
    require(
        scale in SCALES,
        f"SCALE:{scale}",
    )

    if scale in (
        "mamba370m",
        "mamba14b",
    ):
        from scripts import (
            reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
            as bridge
        )

        spec = bridge.scale_spec(scale)

        return {
            "scale": scale,
            "geom": spec["geom"],
            "confirmation": spec["confirmation"],
            "bridge": bridge,
            "readout": None,
            "selected_plane": spec["selected_plane"],
            "control_plane": spec["control_plane"],
            "pair_ids": tuple(bridge.PAIR_IDS),
            "dim": int(spec["dim"]),
            "hf_repo": str(spec["hf_repo"]),
            "hf_revision": str(spec["hf_revision"]),
        }

    if scale == "mamba790m":
        from scripts import (
            reason_router_gen4_mamba790m_readout_alignment_raw_fast_cuda
            as readout
        )

        return {
            "scale": scale,
            "geom": readout.geom,
            "confirmation": readout.confirmation,
            "bridge": None,
            "readout": readout,
            "selected_plane": readout.SELECTED_PLANE,
            "control_plane": readout.CONTROL_PLANE,
            "pair_ids": tuple(readout.PAIR_IDS),
            "dim": int(readout.DIM),
            "hf_repo": str(readout.geom.HF_REPO),
            "hf_revision": str(readout.geom.HF_REVISION),
        }

    from scripts import (
        reason_router_gen4_mamba28b_readout_alignment_raw_fast_cuda
        as readout
    )

    return {
        "scale": scale,
        "geom": readout.geom,
        "confirmation": readout.confirmation,
        "bridge": None,
        "readout": readout,
        "selected_plane": readout.SELECTED_PLANE,
        "control_plane": readout.CONTROL_PLANE,
        "pair_ids": tuple(readout.PAIR_IDS),
        "dim": int(readout.DIM),
        "hf_repo": str(readout.geom.HF_REPO),
        "hf_revision": str(readout.geom.HF_REVISION),
    }


def validate_protocol() -> None:
    expected = {
        "mamba370m": {
            "selected": "P3",
            "control": "P5",
            "first": "xg1_fact_4801",
            "last": "xg1_fact_5100",
            "layer": 35,
            "dim": 650,
        },
        "mamba790m": {
            "selected": "P2",
            "control": "P5",
            "first": "xg1_fact_7501",
            "last": "xg1_fact_7800",
            "layer": 35,
            "dim": 975,
        },
        "mamba14b": {
            "selected": "P5",
            "control": "P4",
            "first": "xg1_fact_4801",
            "last": "xg1_fact_5100",
            "layer": 35,
            "dim": 829,
        },
        "mamba28b": {
            "selected": "P3",
            "control": "P5",
            "first": "xg1_fact_6601",
            "last": "xg1_fact_6900",
            "layer": 47,
            "dim": 1003,
        },
    }

    for scale in SCALES:
        spec = scale_spec(scale)
        exp = expected[scale]

        require(
            spec["selected_plane"] == exp["selected"],
            f"SELECTED:{scale}",
        )
        require(
            spec["control_plane"] == exp["control"],
            f"CONTROL:{scale}",
        )
        require(
            spec["pair_ids"][0] == exp["first"]
            and spec["pair_ids"][-1] == exp["last"]
            and len(spec["pair_ids"]) == 300,
            f"PAIR_RANGE:{scale}",
        )
        require(
            spec["geom"].INTERVENTION_LAYER == exp["layer"],
            f"LAYER:{scale}",
        )
        require(
            spec["geom"].TARGET_OFFSET == 2,
            f"TARGET_OFFSET:{scale}",
        )
        require(
            spec["dim"] == exp["dim"],
            f"DIM:{scale}",
        )

    require(
        COMMON_SELECTED_PLANE == "P3"
        and COMMON_CONTROL_PLANE == "P5",
        "COMMON_CONTRAST",
    )
    require(
        GPU_COUNT == 2
        and PAIRS_PER_SHARD == 150,
        "SHARD_PROTOCOL",
    )
    require(
        ITEMS_PER_SCALE == 600
        and FORWARDS_PER_SCALE == 600
        and BACKWARDS_PER_SCALE == 600,
        "EXECUTION_COUNTS",
    )


def build_input_state(
    spec: Mapping[str, Any],
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str], int],
]:
    if spec["bridge"] is not None:
        rows, encoded, events, _provenance = (
            spec["bridge"].build_input_state(
                spec=spec["bridge"].scale_spec(
                    str(spec["scale"])
                ),
                snapshot=snapshot,
            )
        )

        lookup = spec["bridge"].row_index(rows)

    else:
        readout = spec["readout"]

        rows, encoded, events = (
            readout.build_input_state(snapshot)
        )
        lookup = readout.row_index(rows)

    for pair in spec["pair_ids"]:
        for cell in TARGET_CELLS:
            require(
                (pair, cell) in lookup,
                f"ROW_MISSING:{pair}:{cell}",
            )
            require(
                (pair, cell, ANCHOR_NAME) in events,
                f"ANCHOR_MISSING:{pair}:{cell}",
            )

    return rows, encoded, events, lookup


def feature_inputs(
    encoded: Mapping[str, Any],
    index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = (
        encoded["input_ids"][index:index + 1]
        .detach()
        .to(device)
        .contiguous()
    )
    attention_mask = (
        encoded["attention_mask"][index:index + 1]
        .detach()
        .to(device)
        .contiguous()
    )

    require(
        tuple(input_ids.shape) == (1, 128),
        "INPUT_SHAPE",
    )
    require(
        tuple(attention_mask.shape) == (1, 128),
        "MASK_SHAPE",
    )

    return input_ids, attention_mask


def install_local_leaf_hook(
    mixer: Any,
    *,
    token_index: int,
    intermediate_size: int,
    capture: dict[str, Any],
):
    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1]
            == 2 * intermediate_size,
            "INPROJ_SHAPE",
        )
        require(
            0 <= token_index < output.shape[1],
            "TOKEN_INDEX",
        )

        before = output.detach()

        native_full = (
            before[
                0,
                token_index,
                :intermediate_size,
            ]
            .clone()
            .contiguous()
        )

        leaf = (
            native_full.clone()
            .requires_grad_(True)
        )

        gate = (
            before[
                0,
                token_index,
                intermediate_size:,
            ]
            .clone()
            .contiguous()
        )

        token = torch.cat(
            [leaf, gate],
            dim=0,
        ).view(1, 1, -1)

        out = torch.cat(
            [
                before[:, :token_index, :],
                token,
                before[:, token_index + 1:, :],
            ],
            dim=1,
        )

        require(
            tuple(out.shape)
            == tuple(output.shape),
            "HOOK_OUTPUT_SHAPE",
        )
        require(
            torch.equal(
                out.detach(),
                before,
            ),
            "HOOK_VALUE_DRIFT",
        )

        capture.clear()
        capture["leaf"] = leaf
        capture["native_full"] = (
            native_full.detach()
            .cpu()
            .contiguous()
        )

        return out

    return mixer.in_proj.register_forward_hook(
        hook
    )


def cosine(
    dot: float,
    x_norm: float,
    y_norm: float,
) -> float | None:
    if x_norm == 0.0 or y_norm == 0.0:
        return None

    value = dot / (
        x_norm * y_norm
    )

    require(
        math.isfinite(value),
        "COSINE_NONFINITE",
    )
    require(
        -1.0 - 1e-9
        <= value
        <= 1.0 + 1e-9,
        "COSINE_RANGE",
    )

    return float(
        min(1.0, max(-1.0, value))
    )


def matched_components(
    native_strong: torch.Tensor,
    *,
    selected_plane: str,
    control_plane: str,
    planes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
    dim: int,
    tol: float,
) -> dict[str, Any]:
    value = (
        native_strong.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    require(
        tuple(value.shape) == (dim,),
        "NATIVE_STRONG_SHAPE",
    )

    selected_plus = (
        planes[selected_plane]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    selected_minus = (
        planes[selected_plane]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    a = float(
        torch.dot(
            value,
            selected_plus,
        ).item()
    )
    b = float(
        torch.dot(
            value,
            selected_minus,
        ).item()
    )

    selected = (
        a * selected_plus
        + b * selected_minus
    ).contiguous()

    control = (
        a
        * planes[control_plane]["plus"]
        .detach()
        .cpu()
        .to(torch.float64)
        + b
        * planes[control_plane]["minus"]
        .detach()
        .cpu()
        .to(torch.float64)
    ).contiguous()

    selected_norm = float(
        torch.linalg.vector_norm(
            selected
        ).item()
    )
    control_norm = float(
        torch.linalg.vector_norm(
            control
        ).item()
    )

    mismatch = abs(
        selected_norm
        - control_norm
    )

    require(
        mismatch <= tol,
        (
            "MATCHED_COMPONENT_NORM:"
            f"{selected_plane}:{control_plane}:"
            f"{mismatch}"
        ),
    )

    return {
        "a": a,
        "b": b,
        "selected_component": selected,
        "control_component": control,
        "component_norm": selected_norm,
        "norm_mismatch": mismatch,
    }


def contrast_readout(
    grad_strong: torch.Tensor,
    native_strong: torch.Tensor,
    *,
    selected_plane: str,
    control_plane: str,
    planes: Mapping[
        str,
        Mapping[str, torch.Tensor],
    ],
    dim: int,
    tol: float,
) -> dict[str, Any]:
    component = matched_components(
        native_strong,
        selected_plane=selected_plane,
        control_plane=control_plane,
        planes=planes,
        dim=dim,
        tol=tol,
    )

    selected = component[
        "selected_component"
    ]
    control = component[
        "control_component"
    ]

    l_selected = float(
        torch.dot(
            grad_strong,
            selected,
        ).item()
    )
    l_control = float(
        torch.dot(
            grad_strong,
            control,
        ).item()
    )
    delta = (
        l_selected
        - l_control
    )

    grad_norm = float(
        torch.linalg.vector_norm(
            grad_strong
        ).item()
    )
    component_norm = float(
        component["component_norm"]
    )

    cos_selected = cosine(
        l_selected,
        grad_norm,
        component_norm,
    )
    cos_control = cosine(
        l_control,
        grad_norm,
        component_norm,
    )

    cosine_gap = (
        None
        if (
            cos_selected is None
            or cos_control is None
        )
        else float(
            cos_selected
            - cos_control
        )
    )

    return {
        "selected_plane": selected_plane,
        "control_plane": control_plane,
        "native_selected_coordinates": [
            float(component["a"]),
            float(component["b"]),
        ],
        "selected_component_l2":
            component_norm,
        "control_component_l2":
            float(
                torch.linalg.vector_norm(
                    control
                ).item()
            ),
        "matched_component_norm_mismatch":
            float(
                component[
                    "norm_mismatch"
                ]
            ),
        "L_selected_LM":
            l_selected,
        "L_control_LM":
            l_control,
        "Delta_L_LM":
            float(delta),
        "cosine_gradient_selected_component":
            cos_selected,
        "cosine_gradient_control_component":
            cos_control,
        "cosine_gap":
            cosine_gap,
    }


def freeze_parameters(
    model: torch.nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)

    require(
        all(
            not p.requires_grad
            for p in model.parameters()
        ),
        "MODEL_PARAMETER_REQUIRES_GRAD",
    )
    require(
        all(
            p.grad is None
            for p in model.parameters()
        ),
        "MODEL_PARAMETER_GRAD_PREEXISTS",
    )


def load_vanilla_model(
    spec: Mapping[str, Any],
    *,
    snapshot: Path,
    gpu_id: int,
) -> tuple[
    torch.nn.Module,
    torch.device,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    geom = spec["geom"]

    geom.validate_fast_runtime_for_device(
        gpu_id
    )
    snapshot_identity = (
        geom.validate_snapshot(snapshot)
    )

    from transformers import (
        AutoModelForCausalLM,
    )

    device = torch.device(
        f"cuda:{gpu_id}"
    )

    try:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(snapshot),
                local_files_only=True,
                dtype=torch.float32,
            )
        )
    except TypeError:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(snapshot),
                local_files_only=True,
                torch_dtype=torch.float32,
            )
        )

    model.eval()
    model.to(device)
    freeze_parameters(model)

    require(
        model.__class__.__name__
        == "MambaForCausalLM",
        (
            "MODEL_CLASS:"
            + model.__class__.__name__
        ),
    )

    backbone = getattr(
        model,
        "backbone",
        None,
    )
    require(
        backbone is not None,
        "LM_BACKBONE_MISSING",
    )

    layers = getattr(
        backbone,
        "layers",
        None,
    )
    require(
        layers is not None
        and len(layers)
        == spec["geom"].LAYER_COUNT,
        "LM_LAYER_COUNT",
    )

    mixer = (
        layers[
            spec["geom"].INTERVENTION_LAYER
        ]
        .mixer
    )

    require(
        tuple(
            mixer.in_proj.weight.shape
        )
        == (
            2
            * spec["geom"].INTERMEDIATE_SIZE,
            spec["geom"].HIDDEN_SIZE,
        ),
        "LM_IN_PROJ_SHAPE",
    )

    frozen = (
        spec["confirmation"]
        .load_frozen_geometry()
    )

    runtime_partition = (
        spec["geom"].strong_partition(
            mixer.conv1d.weight
        )
    )

    runtime_mask = (
        runtime_partition["mask"]
        .detach()
        .cpu()
        .bool()
        .contiguous()
    )
    frozen_mask = (
        frozen["strong_mask"]
        .detach()
        .cpu()
        .bool()
        .contiguous()
    )

    require(
        torch.equal(
            runtime_mask,
            frozen_mask,
        ),
        "VANILLA_STRONG_MASK_DRIFT",
    )
    require(
        int(
            frozen_mask.sum().item()
        )
        == int(spec["dim"]),
        "FROZEN_STRONG_DIM",
    )

    return (
        model,
        device,
        {
            "intervention_mixer":
                mixer,
            "strong_mask":
                frozen_mask,
        },
        {
            "snapshot_identity":
                snapshot_identity,
            "model_class":
                model.__class__.__name__,
            "hf_repo":
                spec["hf_repo"],
            "hf_revision":
                spec["hf_revision"],
            "contra_downstream_loaded":
                False,
            "additional_head_trained":
                False,
        },
    )


def run_row(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    lookup: Mapping[
        tuple[str, str],
        int,
    ],
    pair: str,
    cell: str,
) -> dict[str, Any]:
    index = int(
        lookup[(pair, cell)]
    )

    input_ids, attention_mask = (
        feature_inputs(
            encoded,
            index,
            device,
        )
    )

    anchor_index = int(
        events[
            (
                pair,
                cell,
                ANCHOR_NAME,
            )
        ][
            "absolute_anchor_token_index"
        ]
    )

    target_index = (
        anchor_index
        + spec["geom"].TARGET_OFFSET
    )

    seq_len = int(
        input_ids.shape[1]
    )

    require(
        0 <= target_index
        < seq_len - 1,
        "LM_TARGET_INDEX",
    )
    require(
        int(
            attention_mask[
                0,
                target_index,
            ].item()
        )
        == 1,
        "LM_TARGET_MASK",
    )
    require(
        int(
            attention_mask[
                0,
                target_index + 1,
            ].item()
        )
        == 1,
        "LM_NEXT_MASK",
    )

    next_token_id = int(
        input_ids[
            0,
            target_index + 1,
        ].item()
    )

    capture: dict[str, Any] = {}

    handle = install_local_leaf_hook(
        runtime["intervention_mixer"],
        token_index=target_index,
        intermediate_size=(
            spec["geom"].INTERMEDIATE_SIZE
        ),
        capture=capture,
    )

    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        handle.remove()

    require(
        "leaf" in capture
        and "native_full" in capture,
        "HOOK_CAPTURE",
    )

    logits = output.logits

    require(
        torch.is_tensor(logits)
        and logits.ndim == 3
        and logits.shape[0] == 1
        and logits.shape[1] == seq_len,
        "LM_LOGITS_SHAPE",
    )
    require(
        0 <= next_token_id
        < logits.shape[-1],
        "NEXT_TOKEN_VOCAB",
    )

    log_probs = torch.log_softmax(
        logits[
            0,
            target_index,
        ].float(),
        dim=-1,
    )

    objective = (
        log_probs[
            next_token_id
        ]
    )

    require(
        bool(
            torch.isfinite(
                objective.detach()
            ).item()
        ),
        "LM_OBJECTIVE_NONFINITE",
    )

    leaf = capture["leaf"]

    grad = torch.autograd.grad(
        objective,
        leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]

    require(
        tuple(grad.shape)
        == (
            spec["geom"].INTERMEDIATE_SIZE,
        ),
        "LM_GRAD_SHAPE",
    )
    require(
        bool(
            torch.isfinite(
                grad
            ).all().item()
        ),
        "LM_GRAD_NONFINITE",
    )
    require(
        all(
            parameter.grad is None
            for parameter in model.parameters()
        ),
        "PARAMETER_GRAD_CREATED",
    )

    strong_mask = (
        runtime["strong_mask"]
    )

    native_full = (
        capture["native_full"]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    grad_full = (
        grad.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    native_strong = (
        native_full[
            strong_mask
        ]
        .contiguous()
    )
    grad_strong = (
        grad_full[
            strong_mask
        ]
        .contiguous()
    )

    require(
        native_strong.numel()
        == spec["dim"],
        "NATIVE_STRONG_DIM",
    )
    require(
        grad_strong.numel()
        == spec["dim"],
        "GRAD_STRONG_DIM",
    )

    tol = float(
        spec["confirmation"].TOL
    )

    task = contrast_readout(
        grad_strong,
        native_strong,
        selected_plane=(
            spec["selected_plane"]
        ),
        control_plane=(
            spec["control_plane"]
        ),
        planes=frozen["planes"],
        dim=int(spec["dim"]),
        tol=tol,
    )

    common = contrast_readout(
        grad_strong,
        native_strong,
        selected_plane=(
            COMMON_SELECTED_PLANE
        ),
        control_plane=(
            COMMON_CONTROL_PLANE
        ),
        planes=frozen["planes"],
        dim=int(spec["dim"]),
        tol=tol,
    )

    return {
        "schema_version":
            ITEM_SCHEMA,
        "scale":
            spec["scale"],
        "source_pair_id":
            pair,
        "contrast_cell_id":
            cell,
        "anchor_name":
            ANCHOR_NAME,
        "anchor_token_index":
            anchor_index,
        "target_token_index":
            target_index,
        "next_token_id":
            next_token_id,
        "lm_objective":
            "teacher_forced_next_token_log_probability",
        "lm_log_probability":
            float(
                objective.detach().item()
            ),
        "gradient_full_l2":
            float(
                torch.linalg.vector_norm(
                    grad_full
                ).item()
            ),
        "gradient_strong_l2":
            float(
                torch.linalg.vector_norm(
                    grad_strong
                ).item()
            ),
        "native_strong_l2":
            float(
                torch.linalg.vector_norm(
                    native_strong
                ).item()
            ),
        "task_matched":
            task,
        "common_p3_p5":
            common,
        "parameter_grad_created":
            False,
        "model_forward_count":
            1,
        "local_backward_count":
            1,
        "intervention_condition_forward_count":
            0,
        "training_step_count":
            0,
        "p_value_count":
            0,
    }


def worker_paths(
    root: Path,
    worker: int,
) -> dict[str, Path]:
    return {
        "items":
            root
            / f"worker_{worker}_items.jsonl",
        "meta":
            root
            / f"worker_{worker}_meta.json",
        "error":
            root
            / f"worker_{worker}_error.txt",
    }


def worker_run(
    *,
    scale: str,
    gpu_id: int,
    start_index: int,
    end_index: int,
    expected_head: str,
    snapshot: str,
    temp_dir: str,
) -> None:
    paths = worker_paths(
        Path(temp_dir),
        gpu_id,
    )

    try:
        authenticate_repo(expected_head)
        spec = scale_spec(scale)
        snapshot_path = Path(snapshot)

        (
            rows,
            encoded,
            events,
            lookup,
        ) = build_input_state(
            spec,
            snapshot_path,
        )

        (
            model,
            device,
            runtime,
            model_provenance,
        ) = load_vanilla_model(
            spec,
            snapshot=snapshot_path,
            gpu_id=gpu_id,
        )

        frozen = (
            spec["confirmation"]
            .load_frozen_geometry()
        )

        output_rows: list[
            dict[str, Any]
        ] = []

        for pair in spec["pair_ids"][
            start_index:end_index
        ]:
            for cell in TARGET_CELLS:
                item = run_row(
                    spec=spec,
                    model=model,
                    device=device,
                    runtime=runtime,
                    frozen=frozen,
                    encoded=encoded,
                    events=events,
                    lookup=lookup,
                    pair=pair,
                    cell=cell,
                )
                item["shard_index"] = gpu_id
                item["physical_device"] = gpu_id
                output_rows.append(item)

        require(
            len(output_rows)
            == (
                end_index
                - start_index
            )
            * len(TARGET_CELLS),
            "WORKER_ITEM_COUNT",
        )

        torch.cuda.synchronize(
            device
        )

        paths["items"].write_bytes(
            jsonl_bytes(
                output_rows
            )
        )

        meta = {
            "schema_version":
                "gen4-mamba1-vanilla-lm-worker-v1",
            "scale":
                scale,
            "gpu_id":
                gpu_id,
            "pair_first":
                spec["pair_ids"][
                    start_index
                ],
            "pair_last":
                spec["pair_ids"][
                    end_index - 1
                ],
            "pair_count":
                end_index
                - start_index,
            "item_count":
                len(output_rows),
            "model_forward_count":
                len(output_rows),
            "local_backward_count":
                len(output_rows),
            "parameter_grad_count":
                0,
            "intervention_condition_forward_count":
                0,
            "training_step_count":
                0,
            "p_value_count":
                0,
            "model_provenance":
                model_provenance,
            "items_sha256":
                sha256_file(
                    paths["items"]
                ),
        }

        paths["meta"].write_bytes(
            pretty_json_bytes(meta)
        )

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_bundle(
    *,
    scale: str,
    expected_head: str,
    output_dir: Path,
    metas: Sequence[
        Mapping[str, Any]
    ],
    items: Sequence[
        Mapping[str, Any]
    ],
) -> None:
    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    spec = scale_spec(scale)

    require(
        len(items)
        == ITEMS_PER_SCALE,
        "TOTAL_ITEM_COUNT",
    )

    expected_keys = {
        (pair, cell)
        for pair in spec["pair_ids"]
        for cell in TARGET_CELLS
    }

    observed_keys = {
        (
            str(item["source_pair_id"]),
            str(item["contrast_cell_id"]),
        )
        for item in items
    }

    require(
        observed_keys
        == expected_keys,
        "ITEM_COVERAGE",
    )

    require(
        sum(
            int(
                item[
                    "model_forward_count"
                ]
            )
            for item in items
        )
        == FORWARDS_PER_SCALE,
        "TOTAL_FORWARD_COUNT",
    )
    require(
        sum(
            int(
                item[
                    "local_backward_count"
                ]
            )
            for item in items
        )
        == BACKWARDS_PER_SCALE,
        "TOTAL_BACKWARD_COUNT",
    )
    require(
        all(
            item[
                "parameter_grad_created"
            ]
            is False
            for item in items
        ),
        "PARAMETER_GRAD_FLAG",
    )
    require(
        all(
            item["p_value_count"] == 0
            for item in items
        ),
        "ITEM_P_VALUE_COUNT",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    item_path = (
        output_dir / ITEM_FILE
    )
    item_path.write_bytes(
        jsonl_bytes(items)
    )

    summary = {
        "schema_version":
            SUMMARY_SCHEMA,
        "result":
            RESULT_PASS,
        "execution_head":
            expected_head,
        "scale":
            scale,
        "population_first":
            spec["pair_ids"][0],
        "population_last":
            spec["pair_ids"][-1],
        "pair_count":
            300,
        "target_cells":
            list(TARGET_CELLS),
        "item_count":
            ITEMS_PER_SCALE,
        "model_forward_count":
            FORWARDS_PER_SCALE,
        "local_backward_count":
            BACKWARDS_PER_SCALE,
        "parameter_grad_count":
            0,
        "intervention_condition_forward_count":
            0,
        "training_step_count":
            0,
        "p_value_count":
            0,
        "formal_inference_performed":
            False,
        "pair_aggregation_performed":
            False,
        "mean_delta_l_computed":
            False,
        "sign_vector_computed":
            False,
        "selection_reopened":
            False,
        "row_filter_performed":
            False,
        "cohort_replacement_performed":
            False,
        "rescue_performed":
            False,
        "task_matched_selected_plane":
            spec["selected_plane"],
        "task_matched_control_plane":
            spec["control_plane"],
        "common_selected_plane":
            COMMON_SELECTED_PLANE,
        "common_control_plane":
            COMMON_CONTROL_PLANE,
        "gradient_source":
            "pretrained_vanilla_mamba_teacher_forced_next_token_log_probability",
        "contra_downstream_loaded":
            False,
        "additional_head_trained":
            False,
    }

    summary_path = (
        output_dir / SUMMARY_FILE
    )
    summary_path.write_bytes(
        pretty_json_bytes(summary)
    )

    manifest = {
        "schema_version":
            MANIFEST_SCHEMA,
        "result":
            RESULT_PASS,
        "execution_head":
            expected_head,
        "plan_freeze_commit":
            PLAN_FREEZE_COMMIT,
        "scale":
            scale,
        "hf_repo":
            spec["hf_repo"],
        "hf_revision":
            spec["hf_revision"],
        "model_class":
            "MambaForCausalLM",
        "item_file":
            ITEM_FILE,
        "item_sha256":
            sha256_file(item_path),
        "summary_file":
            SUMMARY_FILE,
        "summary_sha256":
            sha256_file(summary_path),
        "pair_count":
            300,
        "item_count":
            ITEMS_PER_SCALE,
        "model_forward_count":
            FORWARDS_PER_SCALE,
        "local_backward_count":
            BACKWARDS_PER_SCALE,
        "parameter_grad_count":
            0,
        "intervention_condition_forward_count":
            0,
        "training_step_count":
            0,
        "p_value_count":
            0,
        "formal_inference_performed":
            False,
        "pair_aggregation_performed":
            False,
        "mean_delta_l_computed":
            False,
        "sign_vector_computed":
            False,
        "selection_reopened":
            False,
        "row_filter_performed":
            False,
        "cohort_replacement_performed":
            False,
        "rescue_performed":
            False,
        "contra_downstream_loaded":
            False,
        "contra_logits_accessed":
            False,
        "g3_ownership_used":
            False,
        "additional_head_trained":
            False,
        "worker_meta":
            list(metas),
    }

    manifest_path = (
        output_dir / MANIFEST_FILE
    )
    manifest_path.write_bytes(
        pretty_json_bytes(manifest)
    )

    sums = {
        ITEM_FILE:
            sha256_file(item_path),
        SUMMARY_FILE:
            sha256_file(summary_path),
        MANIFEST_FILE:
            sha256_file(manifest_path),
    }

    (
        output_dir
        / CHECKSUM_FILE
    ).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest
            in sorted(
                sums.items()
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    scale: str,
    expected_head: str,
    snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(
        expected_head
    )

    spec = scale_spec(scale)

    spec["geom"].validate_snapshot(
        snapshot
    )

    require(
        torch.cuda.is_available(),
        "CUDA_UNAVAILABLE",
    )
    require(
        torch.cuda.device_count()
        >= GPU_COUNT,
        "CUDA_DEVICE_COUNT",
    )
    require(
        not output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    ctx = mp.get_context(
        "spawn"
    )

    with tempfile.TemporaryDirectory(
        prefix=(
            "contramamba_"
            "vanilla_lm_"
        )
    ) as temp:
        temp_dir = Path(temp)

        processes = []

        shards = (
            (0, 0, 150),
            (1, 150, 300),
        )

        for (
            gpu_id,
            start,
            end,
        ) in shards:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "scale":
                        scale,
                    "gpu_id":
                        gpu_id,
                    "start_index":
                        start,
                    "end_index":
                        end,
                    "expected_head":
                        expected_head,
                    "snapshot":
                        str(snapshot),
                    "temp_dir":
                        str(temp_dir),
                },
            )
            process.start()
            processes.append(
                (
                    gpu_id,
                    process,
                )
            )

        failures = []

        for gpu_id, process in processes:
            process.join()

            if process.exitcode != 0:
                paths = worker_paths(
                    temp_dir,
                    gpu_id,
                )

                detail = (
                    paths["error"]
                    .read_text(
                        encoding="utf-8"
                    )
                    if paths[
                        "error"
                    ].is_file()
                    else (
                        "no worker error file"
                    )
                )

                failures.append(
                    (
                        gpu_id,
                        process.exitcode,
                        detail,
                    )
                )

        require(
            not failures,
            "WORKER_FAILURE:"
            + repr(failures),
        )

        metas = []
        all_items = []

        for gpu_id in (0, 1):
            paths = worker_paths(
                temp_dir,
                gpu_id,
            )

            meta = json.loads(
                paths["meta"]
                .read_text(
                    encoding="utf-8"
                )
            )
            shard_items = read_jsonl(
                paths["items"]
            )

            require(
                sha256_file(
                    paths["items"]
                )
                == meta[
                    "items_sha256"
                ],
                "WORKER_ITEMS_SHA",
            )

            metas.append(meta)
            all_items.extend(
                shard_items
            )

        pair_order = {
            pair: index
            for index, pair
            in enumerate(
                spec["pair_ids"]
            )
        }
        cell_order = {
            cell: index
            for index, cell
            in enumerate(
                TARGET_CELLS
            )
        }

        all_items.sort(
            key=lambda row: (
                pair_order[
                    str(
                        row[
                            "source_pair_id"
                        ]
                    )
                ],
                cell_order[
                    str(
                        row[
                            "contrast_cell_id"
                        ]
                    )
                ],
            )
        )

        write_bundle(
            scale=scale,
            expected_head=expected_head,
            output_dir=output_dir,
            metas=metas,
            items=all_items,
        )

    print(
        "RESULT="
        + RESULT_PASS
    )
    print(
        "SCALE="
        + scale
    )
    print(
        "PAIR_COUNT=300"
    )
    print(
        "ITEM_COUNT=600"
    )
    print(
        "MODEL_FORWARD_COUNT=600"
    )
    print(
        "LOCAL_BACKWARD_COUNT=600"
    )
    print(
        "PARAMETER_GRAD_COUNT=0"
    )
    print(
        "INTERVENTION_CONDITION_FORWARD_COUNT=0"
    )
    print(
        "TRAINING_STEP_COUNT=0"
    )
    print(
        "P_VALUE_COUNT=0"
    )
    print(
        "PAIR_AGGREGATION_PERFORMED=False"
    )
    print(
        "MEAN_DELTA_L_COMPUTED=False"
    )
    print(
        "SIGN_VECTOR_COMPUTED=False"
    )
    print(
        "CONTRA_DOWNSTREAM_LOADED=False"
    )
    print(
        "G3_OWNERSHIP_USED=False"
    )
    print(
        "SELECTION_REOPENED=False"
    )
    print(
        "ROW_FILTER_PERFORMED=False"
    )
    print(
        "RESCUE_PERFORMED=False"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh vanilla-Mamba causal-LM "
            "functional readout using frozen "
            "ContraMamba native geometry, "
            "without ContraMamba downstream "
            "weights or logits."
        )
    )

    parser.add_argument(
        "--scale",
        required=True,
        choices=SCALES,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    run_raw(
        scale=args.scale,
        expected_head=args.expected_head,
        snapshot=args.model_snapshot.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
