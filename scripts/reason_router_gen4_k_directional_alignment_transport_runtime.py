from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)
from scripts import (
    reason_router_gen4_native_mamba_state_measurement
    as measurement,
)


RUNTIME_CAST_TOL = 5e-6
MIDPOINT_TOL = 5e-6


class TransportRuntimeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportRuntimeError(message)


def validate_runtime_components(model: Any) -> dict[str, Any]:
    import torch

    measurement.runtime_gate()

    backbone = getattr(model, "mamba", None)
    require(backbone is not None, "MAMBA_BACKBONE_MISSING")

    layers = getattr(backbone, "layers", None)
    require(
        layers is not None
        and len(layers) == core.LAYER_COUNT,
        "LAYER_COUNT_MISMATCH",
    )

    layer15 = layers[core.SOURCE_BLOCK]
    layer17 = layers[core.INTERVENTION_LAYER]

    mixer15 = getattr(layer15, "mixer", None)
    mixer17 = getattr(layer17, "mixer", None)
    norm17 = getattr(layer17, "norm", None)

    require(mixer15 is not None, "LAYER15_MIXER_MISSING")
    require(mixer17 is not None, "LAYER17_MIXER_MISSING")
    require(norm17 is not None, "LAYER17_NORM_MISSING")

    require(
        int(getattr(mixer17, "intermediate_size", -1))
        == core.INTERMEDIATE_SIZE,
        "INTERMEDIATE_SIZE_MISMATCH",
    )

    in_proj = getattr(mixer17, "in_proj", None)
    require(in_proj is not None, "LAYER17_INPROJ_MISSING")

    weight = in_proj.weight.detach().cpu()
    require(
        tuple(weight.shape)
        == (
            2 * core.INTERMEDIATE_SIZE,
            core.HIDDEN_SIZE,
        ),
        "LAYER17_INPROJ_SHAPE",
    )

    conv = getattr(mixer17, "conv1d", None)
    require(conv is not None, "LAYER17_CONV_MISSING")

    partition = core.strong_partition(
        conv.weight
    )
    core.validate_frozen_layer17_partition(
        partition
    )

    strong_mask = torch.zeros(
        core.INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    strong_mask[partition["strong"]] = True

    w_hidden = (
        weight[: core.INTERMEDIATE_SIZE]
        .to(torch.float64)
        .contiguous()
    )

    gamma = (
        norm17.weight.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    require(
        tuple(gamma.shape)
        == (core.HIDDEN_SIZE,),
        "LAYER17_GAMMA_SHAPE",
    )

    return {
        "backbone": backbone,
        "layer15": layer15,
        "mixer15": mixer15,
        "layer17": layer17,
        "mixer17": mixer17,
        "norm17": norm17,
        "strong_mask": strong_mask,
        "partition": partition,
        "w_hidden": w_hidden,
        "gamma": gamma,
    }


def make_layer17_observer(
    mixer17: Any,
):
    code, capture_line = (
        measurement._resolve_and_validate_runtime_binding()
    )

    return measurement._TraceCollector(
        code,
        capture_line,
        {
            id(mixer17): {
                "layer_index":
                    core.INTERVENTION_LAYER,
            }
        },
        enabled=True,
    )


def flatten_layer17_snapshots(
    snapshots: Mapping[
        tuple[int, int, int],
        Any,
    ],
    *,
    token_count: int,
) -> list[np.ndarray]:
    require(
        isinstance(snapshots, Mapping)
        and bool(snapshots),
        "EMPTY_LAYER17_CAPTURE",
    )

    forward_ids = {
        key[0]
        for key in snapshots
    }
    require(
        len(forward_ids) == 1,
        "AMBIGUOUS_FORWARD_ID",
    )

    forward_id = next(iter(forward_ids))

    expected = [
        (
            forward_id,
            core.INTERVENTION_LAYER,
            token_index,
        )
        for token_index
        in range(token_count)
    ]

    require(
        list(snapshots) == expected,
        "LAYER17_CAPTURE_COORDINATES",
    )

    return [
        measurement.flatten_scientific_state(
            snapshots[key]
        )
        for key in expected
    ]


def apply_inproj_intervention(
    inproj_output: Any,
    *,
    token_index: int,
    strong_mask: Any,
    delta_h: Any,
    plus_branch: bool,
    audit: dict[str, Any],
):
    import torch

    require(
        torch.is_tensor(inproj_output),
        "INPROJ_OUTPUT_NOT_TENSOR",
    )
    require(
        inproj_output.ndim == 3
        and inproj_output.shape[0] == 1
        and inproj_output.shape[-1]
        == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_OUTPUT_SHAPE",
    )
    require(
        0 <= int(token_index)
        < inproj_output.shape[1],
        "TARGET_TOKEN_RANGE",
    )

    mask_cpu = (
        strong_mask.detach()
        .cpu()
        .bool()
        .contiguous()
    )
    require(
        mask_cpu.numel()
        == core.INTERMEDIATE_SIZE,
        "STRONG_MASK_WIDTH",
    )
    require(
        int(mask_cpu.sum().item())
        == core.EXPECTED_STRONG_COUNT,
        "STRONG_MASK_COUNT",
    )

    correction64 = (
        delta_h.detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    require(
        correction64.numel()
        == core.EXPECTED_STRONG_COUNT,
        "DELTA_H_WIDTH",
    )
    require(
        bool(torch.isfinite(
            correction64
        ).all().item()),
        "DELTA_H_NONFINITE",
    )

    before = inproj_output.detach().clone()
    out = inproj_output.clone()

    mask = mask_cpu.to(
        device=out.device
    )
    correction = correction64.to(
        device=out.device,
        dtype=out.dtype,
    )

    sign = 0.5 if plus_branch else -0.5

    out_x = out[
        0,
        int(token_index),
        : core.INTERMEDIATE_SIZE,
    ]
    out_x[mask] = (
        out_x[mask]
        + sign * correction
    )

    # Gate half must remain exact.
    require(
        torch.equal(
            out[
                :,
                :,
                core.INTERMEDIATE_SIZE :,
            ],
            before[
                :,
                :,
                core.INTERMEDIATE_SIZE :,
            ],
        ),
        "GATE_BRANCH_CHANGED",
    )

    # All non-strong x channels must remain exact.
    nonstrong = ~mask
    require(
        torch.equal(
            out[
                :,
                :,
                : core.INTERMEDIATE_SIZE,
            ][
                :,
                :,
                nonstrong,
            ],
            before[
                :,
                :,
                : core.INTERMEDIATE_SIZE,
            ][
                :,
                :,
                nonstrong,
            ],
        ),
        "NONSTRONG_X_CHANGED",
    )

    # No other token may change.
    if token_index > 0:
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
                out[
                    :,
                    token_index + 1 :,
                    :,
                ],
                before[
                    :,
                    token_index + 1 :,
                    :,
                ],
            ),
            "LATER_TOKEN_CHANGED",
        )

    before_strong = (
        before[
            0,
            token_index,
            : core.INTERMEDIATE_SIZE,
        ][mask]
        .detach()
        .cpu()
        .clone()
    )
    after_strong = (
        out[
            0,
            token_index,
            : core.INTERMEDIATE_SIZE,
        ][mask]
        .detach()
        .cpu()
        .clone()
    )

    intended = (
        sign * correction
    ).detach().cpu()

    applied = (
        after_strong
        - before_strong
    )

    applied_residual = float(
        torch.max(
            torch.abs(
                applied.to(torch.float64)
                - intended.to(torch.float64)
            )
        ).item()
    )

    require(
        applied_residual
        <= RUNTIME_CAST_TOL,
        "APPLIED_CORRECTION_RESIDUAL",
    )

    audit.clear()
    audit.update({
        "token_index":
            int(token_index),
        "plus_branch":
            bool(plus_branch),
        "before_strong":
            before_strong,
        "after_strong":
            after_strong,
        "applied_correction_max_abs_residual":
            applied_residual,
        "runtime_correction_l2":
            float(
                torch.linalg.vector_norm(
                    correction64
                ).item()
            ),
    })

    return out


def install_inproj_hook(
    mixer17: Any,
    *,
    token_index: int,
    strong_mask: Any,
    delta_h: Any,
    plus_branch: bool,
    audit: dict[str, Any],
):
    def hook(
        _module,
        _args,
        output,
    ):
        return apply_inproj_intervention(
            output,
            token_index=token_index,
            strong_mask=strong_mask,
            delta_h=delta_h,
            plus_branch=plus_branch,
            audit=audit,
        )

    return mixer17.in_proj.register_forward_hook(
        hook
    )


def paired_intervention_audit(
    plus_audit: Mapping[str, Any],
    minus_audit: Mapping[str, Any],
    delta_h: Any,
    *,
    plus_expected_token_index: int,
    minus_expected_token_index: int,
) -> dict[str, float]:
    import torch

    require(
        plus_audit["plus_branch"] is True,
        "PLUS_AUDIT_ROLE",
    )
    require(
        minus_audit["plus_branch"] is False,
        "MINUS_AUDIT_ROLE",
    )
    require(
        type(plus_expected_token_index) is int
        and plus_expected_token_index >= 0
        and type(minus_expected_token_index) is int
        and minus_expected_token_index >= 0,
        "EXPECTED_TOKEN_RANGE",
    )
    require(
        plus_audit["token_index"]
        == plus_expected_token_index,
        "PLUS_AUDIT_TOKEN_MISMATCH",
    )
    require(
        minus_audit["token_index"]
        == minus_expected_token_index,
        "MINUS_AUDIT_TOKEN_MISMATCH",
    )

    bp = (
        plus_audit["before_strong"]
        .detach()
        .cpu()
        .to(torch.float64)
    )
    bm = (
        minus_audit["before_strong"]
        .detach()
        .cpu()
        .to(torch.float64)
    )
    ap = (
        plus_audit["after_strong"]
        .detach()
        .cpu()
        .to(torch.float64)
    )
    am = (
        minus_audit["after_strong"]
        .detach()
        .cpu()
        .to(torch.float64)
    )

    before_mid = 0.5 * (bp + bm)
    after_mid = 0.5 * (ap + am)

    midpoint_residual = float(
        torch.max(
            torch.abs(
                after_mid - before_mid
            )
        ).item()
    )

    require(
        midpoint_residual
        <= MIDPOINT_TOL,
        "PAIR_MIDPOINT_RESIDUAL",
    )

    intended = (
        delta_h.detach()
        .cpu()
        .to(
            dtype=plus_audit[
                "after_strong"
            ].dtype
        )
        .to(torch.float64)
    )

    realized_delta = (
        (ap - am)
        - (bp - bm)
    )

    pair_delta_residual = float(
        torch.max(
            torch.abs(
                realized_delta - intended
            )
        ).item()
    )

    require(
        pair_delta_residual
        <= RUNTIME_CAST_TOL,
        "PAIR_DELTA_RESIDUAL",
    )

    applied_residual = max(
        float(
            plus_audit[
                "applied_correction_max_abs_residual"
            ]
        ),
        float(
            minus_audit[
                "applied_correction_max_abs_residual"
            ]
        ),
    )

    require(
        applied_residual
        <= RUNTIME_CAST_TOL,
        "PAIR_APPLIED_RESIDUAL",
    )

    return {
        "plus_token_index":
            float(plus_expected_token_index),
        "minus_token_index":
            float(minus_expected_token_index),
        "midpoint_max_abs_residual":
            midpoint_residual,
        "pair_delta_max_abs_residual":
            pair_delta_residual,
        "applied_correction_max_abs_residual":
            applied_residual,
        "runtime_correction_l2":
            float(
                plus_audit[
                    "runtime_correction_l2"
                ]
            ),
    }
