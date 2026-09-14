from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

GEN4_PARENT = "a738b4c169d30b0a7e21563c3aba2c29c481c456"
K_CAUSAL_PARENT = "5f5f4d6a80085ad8baf43445475d1c3049535c22"

SOURCE_BLOCK = 15
TARGET_RESIDUAL_LAYER = 16
INTERVENTION_LAYER = 17
TARGET_OFFSET = 2
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 1536
STATE_SIZE = 16
LAYER_COUNT = 24
LAG0_KERNEL_INDEX = 3

TARGET_PLUS_CELL = "C2_NAME"
TARGET_MINUS_CELL = "C0_SHAM"
REFERENCE_PLUS_CELL = "C5_TITLE_NAME"
REFERENCE_MINUS_CELL = "C1_TITLE"
ANCHOR_NAME = "A_IDENTITY"

EXPECTED_STRONG_COUNT = 395
EXPECTED_WEAK_COUNT = 1141
EXPECTED_EQUAL_COUNT = 0
EXPECTED_STRONG_INDEX_SHA256 = (
    "6950bb6c6cc777375f5e4ce18f22fd3272d7b80c5aff0726c25a6ec77d5813ce"
)
EXPECTED_LAYER17_MU_K2 = 0.027899337798707836

FROZEN_LAYER17_ENDPOINT_REL = (
    "reports/reason_router_gen4_native_mamba_state_bridge_name_q1_q3_"
    "scientific_extraction_v1/kinematic_endpoints.jsonl"
)
FROZEN_LAYER17_ENDPOINT_SHA256 = (
    "b47e32496f73f5493e8737b040b30bc4a23dd8cd22454a85244fa812273b6605"
)
FROZEN_DELTA_NAME_PATH_EFFICIENCY_MEAN = -0.013998957453394283

COSINE_SLACK = 1e-12
VECTOR_TOL = 5e-12


class TransportCoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportCoreError(message)


def strong_index_sha256(indices: Sequence[int]) -> str:
    raw = ",".join(str(int(v)) for v in indices).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def strong_partition(conv_weight: Any) -> dict[str, Any]:
    import torch

    require(torch.is_tensor(conv_weight), "CONV_WEIGHT_NOT_TENSOR")
    weight = conv_weight.detach().cpu().to(torch.float64).contiguous()
    require(
        tuple(weight.shape) == (INTERMEDIATE_SIZE, 1, 4),
        "CONV_WEIGHT_SHAPE",
    )

    k = weight[:, 0, LAG0_KERNEL_INDEX]
    k2 = k.square()
    mu = float(k2.mean().item())

    strong = torch.nonzero(k2 > mu, as_tuple=False).flatten()
    weak = torch.nonzero(k2 < mu, as_tuple=False).flatten()
    equal = torch.nonzero(k2 == mu, as_tuple=False).flatten()

    return {
        "mu_k2": mu,
        "strong": strong,
        "weak": weak,
        "equal": equal,
        "strong_index_sha256": strong_index_sha256(strong.tolist()),
    }


def validate_frozen_layer17_partition(
    partition: Mapping[str, Any],
) -> None:
    require(
        len(partition["strong"]) == EXPECTED_STRONG_COUNT,
        "STRONG_COUNT",
    )
    require(
        len(partition["weak"]) == EXPECTED_WEAK_COUNT,
        "WEAK_COUNT",
    )
    require(
        len(partition["equal"]) == EXPECTED_EQUAL_COUNT,
        "EQUAL_COUNT",
    )
    require(
        partition["strong_index_sha256"]
        == EXPECTED_STRONG_INDEX_SHA256,
        "STRONG_INDEX_IDENTITY",
    )
    require(
        math.isclose(
            float(partition["mu_k2"]),
            EXPECTED_LAYER17_MU_K2,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ),
        "MU_K2_IDENTITY",
    )


def _finite64(value: Any, label: str):
    import torch

    require(torch.is_tensor(value), f"{label}_NOT_TENSOR")
    out = value.detach().cpu().to(torch.float64).contiguous()
    require(
        bool(torch.isfinite(out).all().item()),
        f"{label}_NONFINITE",
    )
    return out


def cosine(x: Any, y: Any) -> float:
    import torch

    xx = _finite64(x, "COS_X")
    yy = _finite64(y, "COS_Y")

    require(
        xx.shape == yy.shape and xx.numel() > 0,
        "COS_SHAPE",
    )

    a = float(torch.linalg.vector_norm(xx).item())
    b = float(torch.linalg.vector_norm(yy).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")

    value = float(torch.dot(xx, yy).item() / (a * b))
    require(
        -1.0 - COSINE_SLACK
        <= value
        <= 1.0 + COSINE_SLACK,
        "COS_RANGE",
    )
    return min(1.0, max(-1.0, value))


def reconstruct_pair_geometry(
    plus: Mapping[str, Any],
    minus: Mapping[str, Any],
    *,
    gamma: Any,
    w_hidden: Any,
    strong_mask: Any,
) -> dict[str, Any]:
    import torch

    rp = _finite64(plus["R"], "R_PLUS")
    rm = _finite64(minus["R"], "R_MINUS")
    yp = _finite64(plus["Y"], "Y_PLUS")
    ym = _finite64(minus["Y"], "Y_MINUS")
    xp = _finite64(plus["X"], "X_PLUS")
    xm = _finite64(minus["X"], "X_MINUS")
    gamma = _finite64(gamma, "GAMMA")
    w = _finite64(w_hidden, "W_HIDDEN")
    mask = strong_mask.detach().cpu().bool().contiguous()

    require(tuple(rp.shape) == (HIDDEN_SIZE,), "R_SHAPE")
    require(tuple(yp.shape) == (HIDDEN_SIZE,), "Y_SHAPE")
    require(tuple(xp.shape) == (HIDDEN_SIZE,), "X_SHAPE")
    require(tuple(gamma.shape) == (HIDDEN_SIZE,), "GAMMA_SHAPE")
    require(
        tuple(w.shape) == (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        "W_HIDDEN_SHAPE",
    )
    require(mask.numel() == INTERMEDIATE_SIZE, "MASK_WIDTH")
    require(
        int(mask.sum().item()) == EXPECTED_STRONG_COUNT,
        "MASK_COUNT",
    )

    sp = float(plus["rms_scale"])
    sm = float(minus["rms_scale"])
    require(
        math.isfinite(sp) and math.isfinite(sm),
        "RMS_SCALE_NONFINITE",
    )
    sbar = 0.5 * (sp + sm)

    dr = rp - rm
    dy = yp - ym
    dx = xp - xm

    d = float(torch.linalg.vector_norm(dx).item())
    require(
        math.isfinite(d) and d > 0.0,
        "DELTA_X_ZERO",
    )

    q_r = gamma * (sbar * dr)
    q_y = gamma * (sbar * dy)

    h_r = torch.mv(w, q_r)
    h_y = torch.mv(w, q_y)

    x = (h_r[mask] / d).contiguous()
    y = (h_y[mask] / d).contiguous()

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_GEOMETRY_NORM")

    c = cosine(x, y)
    interaction = 2.0 * float(torch.dot(x, y).item())

    require(
        abs(interaction - 2.0 * a * b * c)
        <= VECTOR_TOL,
        "INTERACTION_CLOSURE",
    )

    return {
        "x": x,
        "y": y,
        "d": d,
        "A": a,
        "B": b,
        "C": c,
        "I": interaction,
    }


def alignment_delta(
    x_target: Any,
    y_target: Any,
    target_cosine: float,
):
    import torch

    x = _finite64(x_target, "ALIGN_X")
    y = _finite64(y_target, "ALIGN_Y")

    require(
        x.numel() == EXPECTED_STRONG_COUNT
        and y.numel() == EXPECTED_STRONG_COUNT,
        "ALIGN_WIDTH",
    )

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")

    c = cosine(x, y)
    require(
        -1.0 - COSINE_SLACK
        <= float(target_cosine)
        <= 1.0 + COSINE_SLACK,
        "TARGET_COS_RANGE",
    )
    ct = min(1.0, max(-1.0, float(target_cosine)))

    u = x / a
    raw = y / b - c * u

    n = float(torch.linalg.vector_norm(raw).item())
    require(
        math.isfinite(n) and n > COSINE_SLACK,
        "DEGENERATE_ORTHOGONAL_COMPONENT",
    )
    v = raw / n

    y_star = b * (
        ct * u
        + math.sqrt(max(0.0, 1.0 - ct * ct)) * v
    )

    require(
        abs(
            float(torch.linalg.vector_norm(y_star).item())
            - b
        )
        <= VECTOR_TOL,
        "ALIGN_B_PRESERVATION",
    )

    realized_a = float(
        torch.linalg.vector_norm(x).item()
    )
    realized_b = float(
        torch.linalg.vector_norm(y_star).item()
    )
    a_residual = abs(realized_a - a)
    b_residual = abs(realized_b - b)

    require(
        a_residual <= VECTOR_TOL,
        "ALIGN_A_PRESERVATION",
    )
    require(
        b_residual <= VECTOR_TOL,
        "ALIGN_B_PRESERVATION",
    )

    realized = cosine(x, y_star)
    require(
        abs(realized - ct) <= VECTOR_TOL,
        "ALIGN_TARGET_COSINE",
    )

    return y_star - y, {
        "baseline_A": a,
        "baseline_B": b,
        "baseline_C": c,
        "realized_A": realized_a,
        "realized_B": realized_b,
        "A_preservation_abs_residual":
            a_residual,
        "B_preservation_abs_residual":
            b_residual,
        "target_C": ct,
        "realized_C": realized,
    }


def magnitude_delta(
    x_target: Any,
    y_target: Any,
    target_a: float,
    target_b: float,
):
    import torch

    x = _finite64(x_target, "MAG_X")
    y = _finite64(y_target, "MAG_Y")

    require(
        x.numel() == EXPECTED_STRONG_COUNT
        and y.numel() == EXPECTED_STRONG_COUNT,
        "MAG_WIDTH",
    )

    a = float(torch.linalg.vector_norm(x).item())
    b = float(torch.linalg.vector_norm(y).item())
    require(a > 0.0 and b > 0.0, "ZERO_SOURCE_NORM")
    require(
        math.isfinite(float(target_a))
        and target_a > 0.0,
        "BAD_TARGET_A",
    )
    require(
        math.isfinite(float(target_b))
        and target_b > 0.0,
        "BAD_TARGET_B",
    )

    xm = (float(target_a) / a) * x
    ym = (float(target_b) / b) * y

    baseline_c = cosine(x, y)
    realized_c = cosine(xm, ym)

    require(
        abs(realized_c - baseline_c) <= VECTOR_TOL,
        "MAG_COSINE_PRESERVATION",
    )
    require(
        abs(
            float(torch.linalg.vector_norm(xm).item())
            - target_a
        )
        <= VECTOR_TOL,
        "MAG_A_TARGET",
    )
    require(
        abs(
            float(torch.linalg.vector_norm(ym).item())
            - target_b
        )
        <= VECTOR_TOL,
        "MAG_B_TARGET",
    )

    return (xm - x) + (ym - y), {
        "target_A": float(target_a),
        "target_B": float(target_b),
        "baseline_C": baseline_c,
        "realized_C": realized_c,
    }


def branch_symmetric_pair(
    plus: Any,
    minus: Any,
    delta: Any,
):
    pp = _finite64(plus, "BRANCH_PLUS")
    mm = _finite64(minus, "BRANCH_MINUS")
    dd = _finite64(delta, "BRANCH_DELTA")

    require(
        pp.shape == mm.shape == dd.shape,
        "BRANCH_SHAPE",
    )

    before_mid = 0.5 * (pp + mm)
    pp2 = pp + 0.5 * dd
    mm2 = mm - 0.5 * dd
    after_mid = 0.5 * (pp2 + mm2)

    residual = float(
        np.max(
            np.abs(
                (after_mid - before_mid).numpy()
            )
        )
    )
    require(
        residual <= VECTOR_TOL,
        "BRANCH_MIDPOINT",
    )
    return pp2, mm2, residual


def post4_path_efficiency(
    states: Sequence[Any],
    anchor: int,
) -> float:
    vectors = [
        np.ascontiguousarray(
            np.asarray(v, dtype=np.float32)
        )
        for v in states
    ]

    require(
        type(anchor) is int and anchor >= 1,
        "ANCHOR",
    )
    require(
        anchor + 4 < len(vectors),
        "POST4_RANGE",
    )

    for vector in vectors:
        require(
            vector.ndim == 1
            and vector.size > 0
            and np.isfinite(vector).all(),
            "STATE_VECTOR",
        )

    speeds = [
        float(
            np.linalg.norm(
                np.ascontiguousarray(
                    vectors[t] - vectors[t - 1],
                    dtype=np.float32,
                )
            )
        )
        for t in range(anchor + 1, anchor + 5)
    ]

    denominator = float(
        np.sum(
            np.asarray(
                speeds,
                dtype=np.float32,
            )
        )
    )
    require(
        math.isfinite(denominator)
        and denominator > 0.0,
        "ZERO_PATH",
    )

    displacement = float(
        np.linalg.norm(
            np.ascontiguousarray(
                vectors[anchor + 4]
                - vectors[anchor],
                dtype=np.float32,
            )
        )
    )

    value = displacement / denominator
    require(
        math.isfinite(value),
        "NONFINITE_PATH_EFFICIENCY",
    )
    return float(value)


def causal_reductions(
    baseline_plus: float,
    baseline_minus: float,
    alignment_plus: float,
    alignment_minus: float,
    magnitude_plus: float,
    magnitude_minus: float,
) -> dict[str, float]:
    values = (
        baseline_plus,
        baseline_minus,
        alignment_plus,
        alignment_minus,
        magnitude_plus,
        magnitude_minus,
    )
    require(
        all(math.isfinite(float(v)) for v in values),
        "NONFINITE_ENDPOINT",
    )

    delta0 = float(baseline_plus) - float(baseline_minus)
    delta_a = float(alignment_plus) - float(alignment_minus)
    delta_m = float(magnitude_plus) - float(magnitude_minus)

    # Frozen DELTA_NAME path-efficiency is negative.
    # Positive R means movement toward zero: attenuation.
    r_align = delta_a - delta0
    r_mag = delta_m - delta0

    return {
        "delta_baseline": delta0,
        "delta_alignment": delta_a,
        "delta_magnitude": delta_m,
        "R_ALIGN": r_align,
        "R_MAG": r_mag,
        "ALIGNMENT_SPECIFICITY": r_align - r_mag,
    }
