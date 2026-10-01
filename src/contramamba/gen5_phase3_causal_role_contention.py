"""Gen5 Phase 3A frozen causal-role contention primitives.

Implementation authority:
    ccda16d6c7ad321588ec6f5321457e0afb7a1d9e
"""

from __future__ import annotations

import hashlib
import math
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

IMPLEMENTATION_AUTHORITY_COMMIT = "ccda16d6c7ad321588ec6f5321457e0afb7a1d9e"
INTERVENTION_LAYER = 17
INTERMEDIATE_SIZE = 1536
INPROJ_WIDTH = 3072
STRONG_DIM = 395
LAG0_KERNEL_INDEX = 3
TARGET_OFFSET = 2
PRESSURES = ("P0", "PR", "PC")

PP3_ROOT = Path("reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89")
PP5_ROOT = Path("reports/reason_router_gen4_pp3_pp5_fresh_xg1_specificity_preparation_0bc49ab")
PLANE_FILES = {
    "pp3_plus": (PP3_ROOT / "pp3_plus.f64le", "66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff"),
    "pp3_minus": (PP3_ROOT / "pp3_minus.f64le", "ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7"),
    "pp5_plus": (PP5_ROOT / "pp5_plus.f64le", "7eb8154a10f647a4b732f7a7b7e34087840a513b88177da06633d8f7b28a4df2"),
    "pp5_minus": (PP5_ROOT / "pp5_minus.f64le", "311a41c37b9586206ea9bfc9da390688d9a6bb5672a5f63bb589c9d019478855"),
}
PLANE_TOL = 1e-12
RUNTIME_CAST_TOL = 5e-6
EXPECTED_STRONG_COUNT = 395
EXPECTED_STRONG_INDEX_SHA256 = "6950bb6c6cc777375f5e4ce18f22fd3272d7b80c5aff0726c25a6ec77d5813ce"
EXPECTED_LAYER17_MU_K2 = 0.027899337798707836


class Phase3ContentionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase3ContentionError(message)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plane_from_bytes(raw: bytes, *, label: str) -> torch.Tensor:
    require(len(raw) == STRONG_DIM * 8, f"{label}_BYTES:{len(raw)}")
    values = struct.unpack(f"<{STRONG_DIM}d", raw)
    vector = torch.tensor(values, dtype=torch.float64).contiguous()
    require(bool(torch.isfinite(vector).all().item()), f"{label}_NONFINITE")
    require(abs(float(torch.linalg.vector_norm(vector).item()) - 1.0) <= PLANE_TOL, f"{label}_NORM")
    return vector


def validate_plane_geometry(planes: Mapping[str, torch.Tensor], *, tol: float = PLANE_TOL) -> dict[str, float]:
    required = {"pp3_plus", "pp3_minus", "pp5_plus", "pp5_minus"}
    require(set(planes) == required, f"PLANE_KEYS:{sorted(planes)}")
    q = {}
    for key in sorted(required):
        value = planes[key].detach().cpu().to(torch.float64).contiguous()
        require(tuple(value.shape) == (STRONG_DIM,), f"{key}_SHAPE")
        require(bool(torch.isfinite(value).all().item()), f"{key}_NONFINITE")
        require(abs(float(torch.linalg.vector_norm(value).item()) - 1.0) <= tol, f"{key}_NORM")
        q[key] = value

    pp3_dot = abs(float(torch.dot(q["pp3_plus"], q["pp3_minus"]).item()))
    pp5_dot = abs(float(torch.dot(q["pp5_plus"], q["pp5_minus"]).item()))
    pp3 = torch.stack([q["pp3_plus"], q["pp3_minus"]], dim=1)
    pp5 = torch.stack([q["pp5_plus"], q["pp5_minus"]], dim=1)
    cross = float(torch.max(torch.abs(pp3.T @ pp5)).item())
    require(pp3_dot <= tol, f"PP3_ORTHOGONALITY:{pp3_dot}")
    require(pp5_dot <= tol, f"PP5_ORTHOGONALITY:{pp5_dot}")
    require(cross <= tol, f"PP3_PP5_CROSS_ORTHOGONALITY:{cross}")
    return {
        "pp3_plus_minus_abs_dot": pp3_dot,
        "pp5_plus_minus_abs_dot": pp5_dot,
        "pp3_pp5_cross_max_abs": cross,
    }


def load_frozen_planes(repo_root: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    root = Path(repo_root)
    planes = {}
    for name, (relative, expected_sha) in PLANE_FILES.items():
        path = root / relative
        require(path.is_file(), f"PLANE_MISSING:{relative.as_posix()}")
        observed = sha256_file(path)
        require(observed == expected_sha, f"PLANE_SHA256:{name}:{observed}")
        planes[name] = _plane_from_bytes(path.read_bytes(), label=name)
    return planes, validate_plane_geometry(planes)


def strong_index_sha256(indices: torch.Tensor | list[int]) -> str:
    values = [int(v) for v in (indices.detach().cpu().flatten().tolist() if torch.is_tensor(indices) else indices)]
    return hashlib.sha256(",".join(str(v) for v in values).encode("ascii")).hexdigest()


def derive_strong_partition(conv_weight: torch.Tensor, *, require_frozen_identity: bool = True) -> dict[str, Any]:
    weight = conv_weight.detach().cpu().to(torch.float64).contiguous()
    require(tuple(weight.shape) == (INTERMEDIATE_SIZE, 1, 4), f"CONV_WEIGHT_SHAPE:{tuple(weight.shape)}")
    k2 = weight[:, 0, LAG0_KERNEL_INDEX].square()
    mu = float(k2.mean().item())
    strong = torch.nonzero(k2 > mu, as_tuple=False).flatten().to(torch.long)
    weak = torch.nonzero(k2 < mu, as_tuple=False).flatten().to(torch.long)
    equal = torch.nonzero(k2 == mu, as_tuple=False).flatten().to(torch.long)
    digest = strong_index_sha256(strong)

    if require_frozen_identity:
        require(len(strong) == EXPECTED_STRONG_COUNT, f"STRONG_COUNT:{len(strong)}")
        require(len(weak) == INTERMEDIATE_SIZE - EXPECTED_STRONG_COUNT, "WEAK_COUNT")
        require(len(equal) == 0, f"EQUAL_COUNT:{len(equal)}")
        require(digest == EXPECTED_STRONG_INDEX_SHA256, f"STRONG_INDEX_SHA256:{digest}")
        require(math.isclose(mu, EXPECTED_LAYER17_MU_K2, rel_tol=1e-15, abs_tol=1e-15), f"MU_K2:{mu}")

    mask = torch.zeros(INTERMEDIATE_SIZE, dtype=torch.bool)
    mask[strong] = True
    return {
        "mu_k2": mu,
        "strong": strong,
        "weak": weak,
        "equal": equal,
        "strong_index_sha256": digest,
        "strong_mask": mask,
    }


def _native_pp3_coefficients(native_strong: torch.Tensor, planes: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    h = native_strong.detach().cpu().to(torch.float64).contiguous()
    require(h.ndim == 2 and h.shape[1] == STRONG_DIM, "NATIVE_STRONG_SHAPE")
    a = h @ planes["pp3_plus"].detach().cpu().to(torch.float64)
    b = h @ planes["pp3_minus"].detach().cpu().to(torch.float64)
    return a.contiguous(), b.contiguous()


def pressure_delta(native_strong: torch.Tensor, *, pressure: str, planes: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    require(pressure in PRESSURES, f"PRESSURE:{pressure}")
    validate_plane_geometry(planes)
    h = native_strong.detach().cpu().to(torch.float64).contiguous()
    require(h.ndim == 2 and h.shape[1] == STRONG_DIM, "NATIVE_STRONG_SHAPE")
    a, b = _native_pp3_coefficients(h, planes)

    if pressure == "P0":
        delta = torch.zeros_like(h)
    elif pressure == "PR":
        delta = -a[:, None] * planes["pp3_plus"][None, :] - b[:, None] * planes["pp3_minus"][None, :]
    else:
        delta = -a[:, None] * planes["pp5_plus"][None, :] - b[:, None] * planes["pp5_minus"][None, :]

    post = h + delta
    post_plus = post @ planes["pp3_plus"]
    post_minus = post @ planes["pp3_minus"]
    if pressure == "PR":
        require(float(torch.max(torch.abs(post_plus)).item()) <= PLANE_TOL, "PR_PLUS_RESIDUAL")
        require(float(torch.max(torch.abs(post_minus)).item()) <= PLANE_TOL, "PR_MINUS_RESIDUAL")

    return {
        "a": a,
        "b": b,
        "delta": delta.contiguous(),
        "post_pp3_plus": post_plus.contiguous(),
        "post_pp3_minus": post_minus.contiguous(),
    }


def apply_stressor_batch(
    output: torch.Tensor,
    *,
    pressure: str,
    strong_mask: torch.Tensor,
    active_rows: torch.Tensor,
    target_indices: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(pressure in PRESSURES, f"PRESSURE:{pressure}")
    require(output.ndim == 3 and output.shape[-1] == INPROJ_WIDTH, f"INPROJ_SHAPE:{tuple(output.shape)}")
    batch, seq, _ = output.shape
    mask_cpu = strong_mask.detach().cpu().bool().contiguous()
    require(mask_cpu.numel() == INTERMEDIATE_SIZE, "STRONG_MASK_WIDTH")
    require(int(mask_cpu.sum().item()) == STRONG_DIM, "STRONG_MASK_COUNT")
    active_cpu = active_rows.detach().cpu().bool().contiguous()
    targets_cpu = target_indices.detach().cpu().to(torch.long).contiguous()
    require(tuple(active_cpu.shape) == (batch,), "ACTIVE_ROWS_SHAPE")
    require(tuple(targets_cpu.shape) == (batch,), "TARGET_INDICES_SHAPE")
    active_index = torch.nonzero(active_cpu, as_tuple=False).flatten()

    if pressure == "P0" or active_index.numel() == 0:
        return output, {
            "pressure": pressure,
            "active_row_count": int(active_index.numel()),
            "coefficient_source": "native_pp3_coordinates",
            "delta_l2": [],
            "post_pp3_plus_max_abs": 0.0,
            "post_pp3_minus_max_abs": 0.0,
        }

    active_targets = targets_cpu[active_index]
    require(bool(torch.all(active_targets >= 0).item()), "NEGATIVE_TARGET_INDEX")
    require(bool(torch.all(active_targets < seq).item()), "TARGET_INDEX_RANGE")

    rows_device = active_index.to(output.device)
    targets_device = active_targets.to(output.device)
    mask_device = mask_cpu.to(output.device)

    native_hidden = output[rows_device, targets_device, :INTERMEDIATE_SIZE]
    native_strong = native_hidden[:, mask_device].detach().cpu().to(torch.float64)
    details = pressure_delta(native_strong, pressure=pressure, planes=planes)

    delta_hidden_cpu = torch.zeros((active_index.numel(), INTERMEDIATE_SIZE), dtype=torch.float64)
    delta_hidden_cpu[:, mask_cpu] = details["delta"]
    delta_hidden = delta_hidden_cpu.to(device=output.device, dtype=output.dtype)
    delta_output = torch.zeros_like(output)
    delta_output[rows_device, targets_device, :INTERMEDIATE_SIZE] = delta_hidden
    result = output + delta_output

    require(torch.equal(result[:, :, INTERMEDIATE_SIZE:], output[:, :, INTERMEDIATE_SIZE:]), "GATE_HALF_CHANGED")
    nonstrong = (~mask_cpu).to(output.device)
    require(
        torch.equal(
            result[:, :, :INTERMEDIATE_SIZE][:, :, nonstrong],
            output[:, :, :INTERMEDIATE_SIZE][:, :, nonstrong],
        ),
        "NONSTRONG_CHANNEL_CHANGED",
    )

    # Validate the correction that was actually realized after casting back
    # into the runtime activation dtype.  This inherits the frozen Gen4
    # transport-runtime tolerance instead of validating only the idealized
    # float64 delta.
    after_hidden = result[
        rows_device,
        targets_device,
        :INTERMEDIATE_SIZE,
    ]
    after_strong = (
        after_hidden[:, mask_device]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )
    applied = after_strong - native_strong
    applied_residual = float(
        torch.max(torch.abs(applied - details["delta"])).item()
    )
    require(
        applied_residual <= RUNTIME_CAST_TOL,
        f"APPLIED_CORRECTION_RESIDUAL:{applied_residual}",
    )

    actual_post_plus = (
        after_strong
        @ planes["pp3_plus"].detach().cpu().to(torch.float64)
    )
    actual_post_minus = (
        after_strong
        @ planes["pp3_minus"].detach().cpu().to(torch.float64)
    )
    actual_post_plus_max = float(
        torch.max(torch.abs(actual_post_plus)).item()
    )
    actual_post_minus_max = float(
        torch.max(torch.abs(actual_post_minus)).item()
    )

    if pressure == "PR":
        require(
            actual_post_plus_max <= RUNTIME_CAST_TOL,
            f"PR_RUNTIME_PLUS_RESIDUAL:{actual_post_plus_max}",
        )
        require(
            actual_post_minus_max <= RUNTIME_CAST_TOL,
            f"PR_RUNTIME_MINUS_RESIDUAL:{actual_post_minus_max}",
        )

    return result, {
        "pressure": pressure,
        "active_row_count": int(active_index.numel()),
        "coefficient_source": "native_pp3_coordinates",
        "delta_l2": [float(v) for v in torch.linalg.vector_norm(details["delta"], dim=1).tolist()],
        "native_pp3_a": [float(v) for v in details["a"].tolist()],
        "native_pp3_b": [float(v) for v in details["b"].tolist()],
        "ideal_post_pp3_plus_max_abs": float(torch.max(torch.abs(details["post_pp3_plus"])).item()),
        "ideal_post_pp3_minus_max_abs": float(torch.max(torch.abs(details["post_pp3_minus"])).item()),
        "applied_correction_max_abs_residual": applied_residual,
        "actual_post_pp3_plus_max_abs": actual_post_plus_max,
        "actual_post_pp3_minus_max_abs": actual_post_minus_max,
    }


@contextmanager
def batch_stressor_hook(
    in_proj: torch.nn.Module,
    *,
    pressure: str,
    strong_mask: torch.Tensor,
    active_rows: torch.Tensor,
    target_indices: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
    audit_sink: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    require(pressure in PRESSURES, f"PRESSURE:{pressure}")
    if pressure == "P0":
        yield
        return

    def hook(_module, _args, output):
        result, audit = apply_stressor_batch(
            output,
            pressure=pressure,
            strong_mask=strong_mask,
            active_rows=active_rows,
            target_indices=target_indices,
            planes=planes,
        )
        if audit_sink is not None:
            audit_sink.append(audit)
        return result

    handle = in_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def contention_fractions(
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    r22: torch.Tensor,
    c22: torch.Tensor,
) -> dict[str, float]:
    a = a_weight.detach().cpu().to(torch.float64).contiguous()
    b = b_weight.detach().cpu().to(torch.float64).contiguous()
    r = r22.detach().cpu().to(torch.float64).contiguous()
    c = c22.detach().cpu().to(torch.float64).contiguous()

    require(a.ndim == 2 and b.ndim == 2, "CORRECTION_WEIGHT_RANK")
    rank, hidden = a.shape
    state_width, b_rank = b.shape
    require(rank == b_rank, "CORRECTION_RANK_MISMATCH")
    require(tuple(r.shape) == (state_width, rank), "R22_SHAPE")
    require(tuple(c.shape) == (state_width, rank), "C22_SHAPE")
    require(hidden > 0 and state_width > 0 and rank > 0, "EMPTY_GEOMETRY")

    g_a = a @ a.T
    g_b = b.T @ b
    map_energy = float(torch.trace(g_b @ g_a).item())
    require(math.isfinite(map_energy) and map_energy > 0.0, "ZERO_CORRECTION_MAP")

    rb = r.T @ b
    cb = c.T @ b
    r_energy = float(torch.trace((rb.T @ rb) @ g_a).item())
    c_energy = float(torch.trace((cb.T @ cb) @ g_a).item())
    r_energy = max(0.0, r_energy)
    c_energy = max(0.0, c_energy)

    return {
        "map_energy": map_energy,
        "map_frobenius_norm": math.sqrt(map_energy),
        "r22_projected_energy": r_energy,
        "c22_projected_energy": c_energy,
        "F_R": r_energy / map_energy,
        "F_C": c_energy / map_energy,
    }
