#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (ROOT, SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

STAGE_C_FREEZE_COMMIT = "bc226f5a19f42d26c916fe7b339b26c41a8fdfe8"
STAGE_B_FREEZE_COMMIT = "b49c1339231a1c5dbb7f870ddc81159109b08c1e"
PHASE3A_EXECUTION_COMMIT = "d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e"

EXECUTION_AUTHORITY_PATH = Path(
    "reports/reason_router_gen5_optimization_path_bypass_"
    "stage_d_downstream_image_execution_authority_spec_candidate.md"
)

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset({
    "scripts/reason_router_gen5_optimization_path_bypass_stage_d_downstream_image_equivalence.py",
    "tests/test_reason_router_gen5_optimization_path_bypass_stage_d_downstream_image_equivalence.py",
})

SEEDS = (6201, 6202, 6203)
PRESSURES = ("P0", "PR", "PC")
CELLS = tuple((s, p) for s in SEEDS for p in PRESSURES)

GPU_QUEUES = {
    0: (
        (6201, "P0"),
        (6201, "PC"),
        (6202, "PR"),
        (6203, "P0"),
        (6203, "PC"),
    ),
    1: (
        (6201, "PR"),
        (6202, "P0"),
        (6202, "PC"),
        (6203, "PR"),
    ),
}

PRIMARY_RADIUS = 0.025
AUDIT_RADIUS = 0.05
AUDIT_ROWS = 32
EXPECTED_DEV_ROWS = 840
EXPECTED_STRESSOR_ROWS = 480
PROBE_STREAM_ROWS = 16
STATE_WIDTH = 24576
RANK = 2
HIDDEN_SIZE = 768
TASK_REPR_SIZE = 384
PRIMITIVE_SIZE = 5
LOGIT_SIZE = 3

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
R22_SHA256 = "a69232900e8b5a91ec5248e36facee4d421eabd502829e6abf3d2719fdb02214"
C22_SHA256 = "c692d39a7387e32e9bf76fe4db7ce2d30d5f7d22a4389e8af363a80b394155c4"

SUMMARY_FILE = "downstream_image_summary.json"
ROWS_FILE = "downstream_image_rows.jsonl"
PROVENANCE_FILE = "run_provenance.json"

SURFACES = ("final_backbone_hidden", "task_repr", "decision_primitives")
SCALE_SURFACES = SURFACES + ("centered_logits",)


class StageDError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageDError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageDError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def status_paths() -> set[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )
    out: set[str] = set()
    for line in raw.splitlines():
        if not line:
            continue
        require(len(line) >= 4, f"STATUS_ROW:{line}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.add(path)
    return out


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(tensor.numpy().tobytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def authenticate_repo(expected_head: str, *, allow_implementation_worktree: bool) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH:{branch}")
    require(head == expected_head, f"HEAD:{head}")
    for ancestor, label in (
        (STAGE_C_FREEZE_COMMIT, "STAGE_C_FREEZE"),
        (STAGE_B_FREEZE_COMMIT, "STAGE_B_FREEZE"),
        (PHASE3A_EXECUTION_COMMIT, "PHASE3A_EXECUTION"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, expected_head) == 0,
            f"{label}_NOT_ANCESTOR",
        )
    observed = status_paths()
    if allow_implementation_worktree:
        require(
            observed <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(observed)}",
        )
    else:
        require(not observed, f"WORKTREE_NOT_CLEAN:{sorted(observed)}")


def validate_gpu_queues() -> None:
    flat = tuple(GPU_QUEUES[0]) + tuple(GPU_QUEUES[1])
    require(len(flat) == 9, "GPU_QUEUE_COUNT")
    require(set(flat) == set(CELLS), "GPU_QUEUE_MATRIX")
    require(len(set(flat)) == 9, "GPU_QUEUE_DUPLICATE")
    require(len(GPU_QUEUES[0]) == 5, "GPU0_QUEUE")
    require(len(GPU_QUEUES[1]) == 4, "GPU1_QUEUE")


def validate_execution_authority(
    *,
    expected_head: str,
    implementation_freeze_commit: str,
    execution_authority_commit: str,
) -> None:
    require(
        git_rc(
            "merge-base", "--is-ancestor",
            implementation_freeze_commit, execution_authority_commit,
        ) == 0,
        "IMPLEMENTATION_FREEZE_NOT_ANCESTOR",
    )
    require(
        git_rc(
            "merge-base", "--is-ancestor",
            execution_authority_commit, expected_head,
        ) == 0,
        "EXECUTION_AUTHORITY_NOT_ANCESTOR",
    )
    for rel in sorted(AUTHORIZED_IMPLEMENTATION_PATHS):
        require(
            git_rc(
                "diff", "--quiet",
                implementation_freeze_commit, expected_head, "--", rel,
            ) == 0,
            f"IMPLEMENTATION_DRIFT:{rel}",
        )
    rel = EXECUTION_AUTHORITY_PATH.as_posix()
    require((ROOT / rel).is_file(), "AUTHORITY_MISSING")
    require(
        git("rev-parse", f"{execution_authority_commit}:{rel}")
        == git("rev-parse", f"HEAD:{rel}"),
        "AUTHORITY_BLOB_DRIFT",
    )
    text = git("show", f"{execution_authority_commit}:{rel}")
    tokens = (
        "SCIENTIFIC_EXECUTION_ALLOWED=YES_FORWARD_ONLY_STAGE_D_DOWNSTREAM_IMAGE_MATRIX",
        f"IMPLEMENTATION_FREEZE_COMMIT={implementation_freeze_commit}",
        "TRAINING_ALLOWED=NO",
        "BACKWARD_ALLOWED=NO",
        "OPTIMIZER_ALLOWED=NO",
        "CONFIRMATORY_9601_9900_ALLOWED=NO",
        "BASELINE_TRAJECTORY=PHASE3A_ZERO_CORRECTION_DEV_EVAL",
        "POPULATION=FROZEN_PHASE3A_DEV_STRESSOR_DOMAIN_480_ROWS",
        "GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
    )
    for token in tokens:
        require(token in text, f"AUTHORITY_TOKEN:{token}")


def effective_rank_and_singular_values(
    matrix: torch.Tensor,
) -> tuple[int, list[float], float]:
    x = matrix.detach().to(torch.float64)
    require(x.ndim == 2, "RANK_MATRIX")
    require(bool(torch.isfinite(x).all()), "RANK_NONFINITE")
    singular = torch.linalg.svdvals(x)
    values = [float(v) for v in singular.tolist()]
    if not values or values[0] == 0.0:
        return 0, values, 0.0
    tol = max(x.shape) * torch.finfo(torch.float32).eps * values[0]
    rank = sum(value > tol for value in values)
    return int(rank), values, float(tol)


def orthonormal_basis_from_b(b_weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    b = b_weight.detach().cpu().to(torch.float64).contiguous()
    require(tuple(b.shape) == (STATE_WIDTH, RANK), "B_SHAPE")
    rank, singular, tol = effective_rank_and_singular_values(b)
    require(rank == 2, f"B_EFFECTIVE_RANK:{rank}")
    u, _s, _vh = torch.linalg.svd(b, full_matrices=False)
    q = u[:, :2].contiguous()
    residual = float(
        torch.max(torch.abs(q.T @ q - torch.eye(2, dtype=torch.float64))).item()
    )
    require(residual <= 1e-10, f"QB_ORTHONORMALITY:{residual}")
    return q, {
        "effective_rank": rank,
        "singular_values": singular,
        "rank_tolerance": tol,
        "orthonormality_max_abs": residual,
    }


def _surface_geometry_one(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, Any]:
    """Basis-invariant subspace geometry from D x 2 response matrices."""
    l = left.detach().to(torch.float64)
    r = right.detach().to(torch.float64)
    require(l.ndim == 2 and r.ndim == 2, "SURFACE_MATRIX_RANK")
    require(l.shape[1] == r.shape[1] == 2, "SURFACE_MATRIX_WIDTH")
    require(l.shape[0] == r.shape[0], "SURFACE_MATRIX_HEIGHT")
    require(bool(torch.isfinite(l).all()) and bool(torch.isfinite(r).all()), "SURFACE_NONFINITE")

    def gram_decomp(x: torch.Tensor):
        gram = x.T @ x
        vals, vecs = torch.linalg.eigh(gram)
        order = torch.argsort(vals, descending=True)
        vals = torch.clamp(vals[order], min=0.0)
        vecs = vecs[:, order]
        singular = torch.sqrt(vals)
        sigma_max = float(singular[0].item()) if singular.numel() else 0.0
        tol = max(int(x.shape[0]), 2) * torch.finfo(torch.float32).eps * sigma_max
        rank = int(torch.count_nonzero(singular > tol).item())
        return gram, singular, vecs, rank, float(tol)

    lg, ls, lv, lr, ltol = gram_decomp(l)
    rg, rs, rv, rr, rtol = gram_decomp(r)
    cross = l.T @ r

    k = min(lr, rr)
    if k == 0:
        cosines: list[float] = []
        angles: list[float] = []
        affinity = None
    else:
        lvec = lv[:, :lr]
        rvec = rv[:, :rr]
        lscale = torch.diag(1.0 / ls[:lr])
        rscale = torch.diag(1.0 / rs[:rr])
        whitened = lscale @ lvec.T @ cross @ rvec @ rscale
        c = torch.linalg.svdvals(whitened)
        c = torch.clamp(c[:k], min=0.0, max=1.0)
        cosines = [float(v) for v in c.tolist()]
        angles = [
            float(torch.rad2deg(torch.acos(v)).item())
            for v in c
        ]
        affinity = float(torch.mean(c * c).item())

    return {
        "left_effective_rank": lr,
        "right_effective_rank": rr,
        "left_singular_values": [float(v) for v in ls.tolist()],
        "right_singular_values": [float(v) for v in rs.tolist()],
        "left_rank_tolerance": ltol,
        "right_rank_tolerance": rtol,
        "principal_cosines": cosines,
        "principal_angles_deg": angles,
        "affinity": affinity,
        "rank2_interpretation_valid": lr == 2 and rr == 2,
    }


def surface_geometry_batch(
    left_columns: Sequence[torch.Tensor],
    right_columns: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    require(len(left_columns) == len(right_columns) == 2, "SURFACE_COLUMN_COUNT")
    l0, l1 = left_columns
    r0, r1 = right_columns
    require(l0.shape == l1.shape == r0.shape == r1.shape, "SURFACE_COLUMN_SHAPE")
    require(l0.ndim >= 2, "SURFACE_BATCH_RANK")
    batch = int(l0.shape[0])
    out = []
    for i in range(batch):
        left = torch.stack(
            [l0[i].reshape(-1), l1[i].reshape(-1)], dim=1
        )
        right = torch.stack(
            [r0[i].reshape(-1), r1[i].reshape(-1)], dim=1
        )
        out.append(_surface_geometry_one(left, right))
    return out


def scale_consistency_batch(
    primary_columns: Sequence[torch.Tensor],
    audit_columns: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    require(len(primary_columns) == len(audit_columns) == 2, "SCALE_COLUMN_COUNT")
    p0, p1 = primary_columns
    a0, a1 = audit_columns
    require(p0.shape == p1.shape == a0.shape == a1.shape, "SCALE_COLUMN_SHAPE")
    batch = int(p0.shape[0])
    rows = []
    for i in range(batch):
        p = torch.stack([p0[i].reshape(-1), p1[i].reshape(-1)], dim=1).to(torch.float64)
        a = torch.stack([a0[i].reshape(-1), a1[i].reshape(-1)], dim=1).to(torch.float64)
        pn = float(torch.linalg.vector_norm(p).item())
        diff = float(torch.linalg.vector_norm(p - a).item())
        rel = diff / max(pn, 1e-30)
        dot = float(torch.sum(p * a).item())
        an = float(torch.linalg.vector_norm(a).item())
        cosine = None if pn == 0.0 or an == 0.0 else dot / (pn * an)
        rows.append({
            "relative_frobenius_difference": rel,
            "frobenius_cosine": cosine,
            "primary_frobenius_norm": pn,
            "audit_frobenius_norm": an,
        })
    return rows


def _load_sources():
    from scripts import (
        reason_router_gen5_optimization_path_bypass_stage_b_functional_decomposition as stage_b
    )
    from scripts import train_reason_router_gen5_phase3a_contention as p3a

    artifacts = stage_b.load_phase3a_artifacts()
    static = p3a.validate_static_artifacts()
    require(len(static["dev_rows"]) == EXPECTED_DEV_ROWS, "DEV_ROWS")
    return stage_b, p3a, artifacts, static


def _stressor_subset_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    active = bundle["stressor_active"].detach().cpu().bool()
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    require(int(indices.numel()) == EXPECTED_STRESSOR_ROWS, f"STRESSOR_ROWS:{indices.numel()}")
    targets = bundle["target_indices"].detach().cpu()[indices]
    require(bool(torch.all(targets >= 0)), "STRESSOR_TARGET_NEGATIVE")
    inputs = {
        key: value[indices].contiguous()
        for key, value in bundle["model_inputs"].items()
    }
    return {
        "model_inputs": inputs,
        "stressor_active": torch.ones(EXPECTED_STRESSOR_ROWS, dtype=torch.bool),
        "target_indices": targets.contiguous(),
        "row_ids": [bundle["row_ids"][int(i)] for i in indices.tolist()],
        "pair_ids": [bundle["pair_ids"][int(i)] for i in indices.tolist()],
        "contrast_cell_ids": [bundle["contrast_cell_ids"][int(i)] for i in indices.tolist()],
    }


def subset_order_sha256(bundle: Mapping[str, Any]) -> str:
    value = [
        {
            "row_id": str(bundle["row_ids"][i]),
            "pair_id": str(bundle["pair_ids"][i]),
            "cell": str(bundle["contrast_cell_ids"][i]),
            "target": int(bundle["target_indices"][i]),
        }
        for i in range(len(bundle["row_ids"]))
    ]
    return sha256_bytes(canonical_json_bytes(value))


def _feature_slice_to_device(
    bundle: Mapping[str, Any],
    start: int,
    stop: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = bundle["model_inputs"]
    features = {
        key: inputs[key][start:stop].to(device)
        for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask")
    }
    active = bundle["stressor_active"][start:stop].to(device)
    targets = bundle["target_indices"][start:stop].to(device)
    return features, active, targets


def _write_probe_contribution(
    native_mixer: torch.nn.Module,
    mixer_input: torch.Tensor,
    *,
    direction: torch.Tensor,
    radius: float,
    target_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Additional layer-22 mixer output from one controlled WRITE22 perturbation."""
    require(mixer_input.ndim == 3, "PROBE_INPUT_RANK")
    batch, seq_len, hidden = mixer_input.shape
    require(hidden == HIDDEN_SIZE, "PROBE_HIDDEN_SIZE")
    require(tuple(attention_mask.shape) == (batch, seq_len), "PROBE_MASK_SHAPE")
    require(tuple(target_indices.shape) == (batch,), "PROBE_TARGET_SHAPE")
    require(abs(float(radius)) > 0.0 and math.isfinite(float(radius)), "PROBE_RADIUS")
    require(tuple(direction.shape) == (STATE_WIDTH,), "PROBE_DIRECTION_SHAPE")

    projected = native_mixer.in_proj(mixer_input).transpose(1, 2)
    hidden_states, gate = projected.chunk(2, dim=1)
    active = attention_mask.to(hidden_states.dtype)
    hidden_states = hidden_states * active.unsqueeze(1)

    conv_hidden = native_mixer.act(
        native_mixer.conv1d(hidden_states)[..., :seq_len]
    )
    conv_hidden = conv_hidden * active.unsqueeze(1)

    ssm_parameters = native_mixer.x_proj(conv_hidden.transpose(1, 2))
    time_step, _native_b, c_readout = torch.split(
        ssm_parameters,
        [
            int(native_mixer.time_step_rank),
            int(native_mixer.ssm_state_size),
            int(native_mixer.ssm_state_size),
        ],
        dim=-1,
    )
    discrete_time_step = F.softplus(
        F.linear(
            time_step,
            native_mixer.dt_proj.weight,
            native_mixer.dt_proj.bias,
        )
    ).transpose(1, 2)
    a_continuous = -torch.exp(native_mixer.A_log.float())

    direction_live = (
        direction.to(device=mixer_input.device, dtype=mixer_input.dtype)
        .reshape(int(native_mixer.intermediate_size), int(native_mixer.ssm_state_size))
    )
    state = torch.zeros(
        (
            batch,
            int(native_mixer.intermediate_size),
            int(native_mixer.ssm_state_size),
        ),
        device=mixer_input.device,
        dtype=mixer_input.dtype,
    )
    outputs = []

    for token_index in range(seq_len):
        discrete_a_t = torch.exp(
            a_continuous[None, :, :]
            * discrete_time_step[:, :, token_index, None].float()
        ).to(dtype=mixer_input.dtype)
        row_selector = (target_indices == token_index).to(mixer_input.dtype)
        write_t = (
            direction_live[None, :, :]
            * row_selector[:, None, None]
            * float(radius)
        )
        state = discrete_a_t * state + write_t
        read_t = torch.sum(
            state.to(c_readout.dtype)
            * c_readout[:, token_index, None, :],
            dim=-1,
        )
        scan_t = read_t * native_mixer.act(gate[:, :, token_index])
        outputs.append(
            F.linear(scan_t, native_mixer.out_proj.weight, bias=None)
        )

    return torch.stack(outputs, dim=1)


@contextlib.contextmanager
def layer22_write_probe(
    wrapper: Any,
    *,
    direction: torch.Tensor,
    radius: float,
    target_indices: torch.Tensor,
) -> Iterator[None]:
    require(int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0, "BASELINE_B_NONZERO")
    prior = wrapper.__dict__.get("forward", None)

    def patched(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        require(cache_params is None, "PROBE_CACHE_UNSUPPORTED")
        require(cache_position is None, "PROBE_CACHE_POSITION_UNSUPPORTED")
        effective_mask = (
            attention_mask if attention_mask is not None
            else self._phase2_active_mask
        )
        require(effective_mask is not None, "PROBE_ACTIVE_MASK_REQUIRED")
        native_output = self.native_mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        contribution = _write_probe_contribution(
            self.native_mixer,
            hidden_states,
            direction=direction,
            radius=radius,
            target_indices=target_indices,
            attention_mask=effective_mask,
        )
        return native_output + contribution

    wrapper.forward = types.MethodType(patched, wrapper)
    try:
        yield
    finally:
        if prior is None:
            del wrapper.__dict__["forward"]
        else:
            wrapper.forward = prior


def _extract_surfaces(
    hidden: torch.Tensor,
    downstream: Mapping[str, Any],
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    masked_hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
    task_repr = torch.cat(
        [
            downstream["frame_pair_repr"],
            downstream["predicate_pair_repr"],
            downstream["sufficiency_repr"],
        ],
        dim=-1,
    )
    require(task_repr.shape[-1] == TASK_REPR_SIZE, "TASK_REPR_SIZE")
    primitives = torch.stack(
        [
            downstream["frame_prob"],
            downstream["predicate_coverage_prob"],
            downstream["sufficiency_prob"],
            downstream["positive_energy"],
            downstream["negative_energy"],
        ],
        dim=-1,
    )
    require(primitives.shape[-1] == PRIMITIVE_SIZE, "PRIMITIVE_SIZE")
    logits = downstream["logits"]
    require(logits.shape[-1] == LOGIT_SIZE, "LOGIT_SIZE")
    centered = logits - logits.mean(dim=-1, keepdim=True)
    return {
        "final_backbone_hidden": masked_hidden,
        "task_repr": task_repr,
        "decision_primitives": primitives,
        "centered_logits": centered,
    }


def _probe_one_direction(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    p3a: Any,
    features: Mapping[str, torch.Tensor],
    active: torch.Tensor,
    targets: torch.Tensor,
    pressure: str,
    strong_mask: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    radius: float,
) -> dict[str, torch.Tensor]:
    require(pressure in PRESSURES, f"PRESSURE:{pressure}")
    mixer17 = model.mamba.layers[17].mixer

    def run(sign: float) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            with p3a.phase2_active_mask(model, features["attention_mask"]):
                with p3a.batch_stressor_hook(
                    mixer17.in_proj,
                    pressure=pressure,
                    strong_mask=strong_mask,
                    active_rows=active,
                    target_indices=targets,
                    planes=planes,
                ):
                    with layer22_write_probe(
                        wrapper,
                        direction=direction,
                        radius=float(sign) * float(radius),
                        target_indices=targets,
                    ):
                        result = model.mamba(input_ids=features["input_ids"])
            hidden = result.last_hidden_state
            downstream = p3a._historical_forward_from_hidden(model, features, hidden)
            return _extract_surfaces(hidden, downstream, features["attention_mask"])

    plus = run(+1.0)
    minus = run(-1.0)
    out = {}
    for name in SCALE_SURFACES:
        out[name] = (plus[name] - minus[name]) / (2.0 * float(radius))
        require(bool(torch.isfinite(out[name]).all()), f"DERIVATIVE_NONFINITE:{name}")
    return out


def _row_geometry_records(
    *,
    row_ids: Sequence[str],
    pair_ids: Sequence[str],
    cells: Sequence[str],
    targets: torch.Tensor,
    r_columns: Sequence[dict[str, torch.Tensor]],
    b_columns: Sequence[dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    for surface in SURFACES:
        rg = [r_columns[i][surface] for i in range(2)]
        bg = [b_columns[i][surface] for i in range(2)]
        geometry = surface_geometry_batch(rg, bg)
        if not rows:
            rows = [
                {
                    "row_id": str(row_ids[i]),
                    "source_pair_id": str(pair_ids[i]),
                    "contrast_cell_id": str(cells[i]),
                    "target_token_index": int(targets[i].item()),
                    "surfaces": {},
                }
                for i in range(len(row_ids))
            ]
        for i, geo in enumerate(geometry):
            rows[i]["surfaces"][surface] = geo

    # Centered logits are reported descriptively only; no rank-2 verdict.
    rlog = [r_columns[i]["centered_logits"] for i in range(2)]
    blog = [b_columns[i]["centered_logits"] for i in range(2)]
    for i in range(len(rows)):
        rm = torch.stack([rlog[0][i], rlog[1][i]], dim=1).to(torch.float64)
        bm = torch.stack([blog[0][i], blog[1][i]], dim=1).to(torch.float64)
        rows[i]["centered_logits"] = _surface_geometry_one(rm, bm)
        rows[i]["centered_logits"]["rank2_equivalence_interpretation_allowed"] = False
    return rows


def _summarize_geometry(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for surface in SURFACES:
        geos = [row["surfaces"][surface] for row in rows]
        valid = [g for g in geos if g["rank2_interpretation_valid"]]
        affinities = [float(g["affinity"]) for g in geos if g["affinity"] is not None]
        first_angles = [
            float(g["principal_angles_deg"][0])
            for g in valid
            if len(g["principal_angles_deg"]) >= 2
        ]
        second_angles = [
            float(g["principal_angles_deg"][1])
            for g in valid
            if len(g["principal_angles_deg"]) >= 2
        ]
        rank_pairs: dict[str, int] = {}
        for g in geos:
            key = f"{g['left_effective_rank']}x{g['right_effective_rank']}"
            rank_pairs[key] = rank_pairs.get(key, 0) + 1
        out[surface] = {
            "row_count": len(geos),
            "rank2_valid_count": len(valid),
            "rank2_valid_fraction": len(valid) / max(len(geos), 1),
            "rank_pair_counts": rank_pairs,
            "mean_affinity": None if not affinities else sum(affinities) / len(affinities),
            "mean_principal_angle_1_deg":
                None if not first_angles else sum(first_angles) / len(first_angles),
            "mean_principal_angle_2_deg":
                None if not second_angles else sum(second_angles) / len(second_angles),
        }
    return out


def _run_cell(args: argparse.Namespace, seed: int, pressure: str) -> dict[str, Any]:
    from contramamba.gen5_phase2_state_update_ownership import (
        load_frozen_owner_bases,
        parent_parameter_fingerprint,
    )
    from scripts import train_reason_router_gen5_phase2_state_update_ownership as p2train

    stage_b, p3a, artifacts, static = _load_sources()
    snapshot = p2train.resolve_exact_snapshot(args.model_snapshot)
    checkpoint = Path(args.checkpoint)
    p3a.validate_checkpoint(checkpoint)
    encoded = p3a.load_runtime_encoding(static, args.tokenizer_snapshot)
    subset = _stressor_subset_bundle(encoded["dev_bundle"])

    artifact = artifacts[(seed, pressure)]
    state = artifact["payload"]["state_dict"]
    b_final = state["B_theta.weight"].detach().cpu().contiguous()
    q_b, q_b_audit = orthonormal_basis_from_b(b_final)
    r22, _c22, _basis_geometry = load_frozen_owner_bases(ROOT)
    r22 = r22.detach().cpu().to(torch.float64).contiguous()

    input_geo = _surface_geometry_one(r22, q_b)

    model, wrapper, runtime_meta, strong_mask, planes = p3a._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint,
        seed=seed,
    )
    model.eval()
    model.mamba.config.use_cache = False
    require(int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0, "STEP0_B_NONZERO")
    parent_before = parent_parameter_fingerprint(model)

    device = torch.device("cuda:0")
    row_records: list[dict[str, Any]] = []
    primary_hashers = {
        f"{family}_{basis}_{surface}": hashlib.sha256()
        for family in ("R22", "BFINAL")
        for basis in range(2)
        for surface in SCALE_SURFACES
    }
    audit_rows_public: list[dict[str, Any]] = []

    for start in range(0, EXPECTED_STRESSOR_ROWS, PROBE_STREAM_ROWS):
        stop = min(start + PROBE_STREAM_ROWS, EXPECTED_STRESSOR_ROWS)
        features, active, targets = _feature_slice_to_device(subset, start, stop, device)
        require(bool(torch.all(active)), "SUBSET_ACTIVE_FALSE")

        r_columns = []
        b_columns = []
        for family, basis_matrix, collector in (
            ("R22", r22, r_columns),
            ("BFINAL", q_b, b_columns),
        ):
            for basis_index in range(2):
                deriv = _probe_one_direction(
                    model=model,
                    wrapper=wrapper,
                    p3a=p3a,
                    features=features,
                    active=active,
                    targets=targets,
                    pressure=pressure,
                    strong_mask=strong_mask,
                    planes=planes,
                    direction=basis_matrix[:, basis_index],
                    radius=PRIMARY_RADIUS,
                )
                for surface in SCALE_SURFACES:
                    primary_hashers[f"{family}_{basis_index}_{surface}"].update(
                        deriv[surface].detach().cpu().to(torch.float32).contiguous().numpy().tobytes()
                    )
                collector.append(deriv)

        chunk_rows = _row_geometry_records(
            row_ids=subset["row_ids"][start:stop],
            pair_ids=subset["pair_ids"][start:stop],
            cells=subset["contrast_cell_ids"][start:stop],
            targets=targets.detach().cpu(),
            r_columns=r_columns,
            b_columns=b_columns,
        )

        # Strong causal audit: no response is allowed before the target coordinate.
        pretarget_max = 0.0
        for deriv in r_columns + b_columns:
            hidden = deriv["final_backbone_hidden"].detach()
            for local_i in range(int(hidden.shape[0])):
                t = int(targets[local_i].item())
                if t > 0:
                    pretarget_max = max(
                        pretarget_max,
                        float(torch.max(torch.abs(hidden[local_i, :t])).item()),
                    )
        require(pretarget_max <= 1e-6, f"PRETARGET_CAUSALITY:{pretarget_max}")

        for row in chunk_rows:
            row["seed"] = seed
            row["pressure"] = pressure
            row["primary_radius"] = PRIMARY_RADIUS
            row["pretarget_response_max_abs_chunk"] = pretarget_max
        row_records.extend(chunk_rows)

        # Scale audit only on the frozen first AUDIT_ROWS rows.
        if start < AUDIT_ROWS:
            audit_stop = min(stop, AUDIT_ROWS)
            audit_n = audit_stop - start
            audit_features = {
                key: value[:audit_n]
                for key, value in features.items()
            }
            audit_active = active[:audit_n]
            audit_targets = targets[:audit_n]
            r_audit = []
            b_audit = []
            for family, basis_matrix, collector in (
                ("R22", r22, r_audit),
                ("BFINAL", q_b, b_audit),
            ):
                for basis_index in range(2):
                    collector.append(_probe_one_direction(
                        model=model,
                        wrapper=wrapper,
                        p3a=p3a,
                        features=audit_features,
                        active=audit_active,
                        targets=audit_targets,
                        pressure=pressure,
                        strong_mask=strong_mask,
                        planes=planes,
                        direction=basis_matrix[:, basis_index],
                        radius=AUDIT_RADIUS,
                    ))
            for local_i in range(audit_n):
                record = {
                    "row_id": subset["row_ids"][start + local_i],
                    "primary_radius": PRIMARY_RADIUS,
                    "audit_radius": AUDIT_RADIUS,
                    "families": {},
                }
                for family, primary, audit in (
                    ("R22", r_columns, r_audit),
                    ("BFINAL", b_columns, b_audit),
                ):
                    fam = {}
                    for surface in SCALE_SURFACES:
                        pcols = [
                            primary[j][surface][local_i:local_i+1]
                            for j in range(2)
                        ]
                        acols = [
                            audit[j][surface][local_i:local_i+1]
                            for j in range(2)
                        ]
                        fam[surface] = scale_consistency_batch(pcols, acols)[0]
                    record["families"][family] = fam
                audit_rows_public.append(record)

        del r_columns, b_columns
        torch.cuda.synchronize()

    require(len(row_records) == EXPECTED_STRESSOR_ROWS, "ROW_RECORD_COUNT")
    require(len(audit_rows_public) == AUDIT_ROWS, "AUDIT_ROW_COUNT")
    require(parent_parameter_fingerprint(model) == parent_before, "PARENT_MUTATION")
    require(int(torch.count_nonzero(wrapper.correction.B_theta.weight).item()) == 0, "POST_STEP0_B_NONZERO")

    scale_summary: dict[str, Any] = {}
    for family in ("R22", "BFINAL"):
        scale_summary[family] = {}
        for surface in SCALE_SURFACES:
            values = [
                row["families"][family][surface]
                for row in audit_rows_public
            ]
            diffs = [float(v["relative_frobenius_difference"]) for v in values]
            cos = [
                float(v["frobenius_cosine"])
                for v in values
                if v["frobenius_cosine"] is not None
            ]
            scale_summary[family][surface] = {
                "audit_row_count": len(values),
                "mean_relative_frobenius_difference": sum(diffs) / len(diffs),
                "max_relative_frobenius_difference": max(diffs),
                "mean_frobenius_cosine": None if not cos else sum(cos) / len(cos),
                "min_frobenius_cosine": None if not cos else min(cos),
            }

    cell_summary = {
        "seed": seed,
        "pressure": pressure,
        "population": "FROZEN_PHASE3A_DEV_STRESSOR_DOMAIN",
        "row_count": EXPECTED_STRESSOR_ROWS,
        "row_order_sha256": subset_order_sha256(subset),
        "baseline_trajectory": "PHASE3A_ZERO_CORRECTION_DEV_EVAL",
        "final_B_source_sha256": artifact["checkpoint_sha256"],
        "final_B_basis_audit": q_b_audit,
        "input_write_space_R22_vs_Bfinal": input_geo,
        "downstream_surfaces": _summarize_geometry(row_records),
        "scale_audit": scale_summary,
        "primary_radius": PRIMARY_RADIUS,
        "audit_radius": AUDIT_RADIUS,
        "audit_rows": AUDIT_ROWS,
        "primary_derivative_stream_sha256": {
            key: hasher.hexdigest()
            for key, hasher in sorted(primary_hashers.items())
        },
        "model_eval_mode": True,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "training_executed": False,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
        "runtime": runtime_meta["runtime"],
    }

    del model, wrapper
    torch.cuda.empty_cache()

    return {
        "summary": cell_summary,
        "rows": row_records,
        "scale_audit_rows": audit_rows_public,
    }


def run_static_verify(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=True)
    validate_gpu_queues()
    stage_b, p3a, artifacts, static = _load_sources()
    del stage_b

    from contramamba.gen5_phase2_state_update_ownership import load_frozen_owner_bases

    r22, c22, geometry = load_frozen_owner_bases(ROOT)
    require(tuple(r22.shape) == (STATE_WIDTH, 2), "R22_SHAPE")
    require(tuple(c22.shape) == (STATE_WIDTH, 2), "C22_SHAPE")
    require(geometry["r22_c22_cross_max_abs"] <= 1e-10, "R22_C22_GEOMETRY")

    for seed, pressure in CELLS:
        b = artifacts[(seed, pressure)]["payload"]["state_dict"]["B_theta.weight"]
        orthonormal_basis_from_b(b)

    # Static bundle already fixes 840 dev rows; runtime tokenizer is intentionally not run.
    stressor_count = sum(bool(row["stressor_domain"]) for row in static["dev_rows"])
    require(stressor_count == EXPECTED_STRESSOR_ROWS, f"STATIC_STRESSOR_ROWS:{stressor_count}")

    print("GEN5_STAGE_D_STATIC_VERIFY_PASS")
    print(f"HEAD={args.expected_head}")
    print("STAGE_C_FREEZE=" + STAGE_C_FREEZE_COMMIT)
    print("CELLS=9")
    print(f"DEV_ROWS={EXPECTED_DEV_ROWS}")
    print(f"STRESSOR_ROWS={EXPECTED_STRESSOR_ROWS}")
    print(f"PRIMARY_RADIUS={PRIMARY_RADIUS}")
    print(f"AUDIT_RADIUS={AUDIT_RADIUS}")
    print(f"AUDIT_ROWS={AUDIT_ROWS}")
    print("BASELINE_TRAJECTORY=PHASE3A_ZERO_CORRECTION_DEV_EVAL")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def run_worker(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_gpu_queues()
    require(args.worker_id in (0, 1), f"WORKER_ID:{args.worker_id}")

    scratch = Path(args.scratch_root) / f"worker{args.worker_id}"
    require(not scratch.exists(), f"WORKER_OUTPUT_COLLISION:{scratch}")
    scratch.mkdir(parents=True, exist_ok=False)

    summaries = []
    rows = []
    scale_rows = []
    for seed, pressure in GPU_QUEUES[args.worker_id]:
        result = _run_cell(args, seed, pressure)
        summaries.append(result["summary"])
        rows.extend(result["rows"])
        for row in result["scale_audit_rows"]:
            public = dict(row)
            public["seed"] = seed
            public["pressure"] = pressure
            scale_rows.append(public)

    (scratch / "worker_summary.json").write_bytes(canonical_json_bytes({
        "worker_id": args.worker_id,
        "queue": [[s, p] for s, p in GPU_QUEUES[args.worker_id]],
        "cells": summaries,
        "scale_audit_rows": scale_rows,
    }))
    with (scratch / "worker_rows.jsonl").open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))

    print(f"GEN5_STAGE_D_WORKER_PASS worker={args.worker_id} cells={len(summaries)} rows={len(rows)}")


def _validate_two_t4s() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 2, f"CUDA_DEVICE_COUNT:{torch.cuda.device_count()}")
    devices = []
    for index in (0, 1):
        name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        require(name == "Tesla T4", f"GPU_NAME:{index}:{name}")
        require(capability == (7, 5), f"GPU_CAPABILITY:{index}:{capability}")
        devices.append({
            "index": index,
            "name": name,
            "capability": list(capability),
        })
    return {"device_count": 2, "devices": devices}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_matrix(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head, allow_implementation_worktree=False)
    validate_execution_authority(
        expected_head=args.expected_head,
        implementation_freeze_commit=args.implementation_freeze_commit,
        execution_authority_commit=args.execution_authority_commit,
    )
    validate_gpu_queues()
    gpu_meta = _validate_two_t4s()

    output_root = Path(args.output_root)
    scratch_root = Path(str(output_root) + "_scratch")
    require(not output_root.exists(), f"OUTPUT_COLLISION:{output_root}")
    require(not scratch_root.exists(), f"SCRATCH_COLLISION:{scratch_root}")
    scratch_root.mkdir(parents=True, exist_ok=False)

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-worker",
        "--expected-head", args.expected_head,
        "--implementation-freeze-commit", args.implementation_freeze_commit,
        "--execution-authority-commit", args.execution_authority_commit,
        "--model-snapshot", str(args.model_snapshot),
        "--tokenizer-snapshot", str(args.tokenizer_snapshot),
        "--checkpoint", str(args.checkpoint),
        "--scratch-root", str(scratch_root),
    ]

    workers = []
    for worker_id in (0, 1):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        proc = subprocess.Popen(
            common + ["--worker-id", str(worker_id)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        workers.append((worker_id, proc))

    failures = []
    for worker_id, proc in workers:
        stdout, _ = proc.communicate()
        print(f"=== STAGE D WORKER {worker_id} ===")
        print(stdout, end="" if stdout.endswith("\n") else "\n")
        if proc.returncode != 0:
            failures.append((worker_id, proc.returncode))
    require(not failures, f"WORKER_FAILURES:{failures}")

    cell_summaries = []
    rows = []
    scale_rows = []
    for worker_id in (0, 1):
        worker = scratch_root / f"worker{worker_id}"
        summary = json.loads((worker / "worker_summary.json").read_text(encoding="utf-8"))
        cell_summaries.extend(summary["cells"])
        scale_rows.extend(summary["scale_audit_rows"])
        rows.extend(_read_jsonl(worker / "worker_rows.jsonl"))

    require(len(cell_summaries) == 9, f"CELL_SUMMARY_COUNT:{len(cell_summaries)}")
    require(len(rows) == 9 * EXPECTED_STRESSOR_ROWS, f"ROW_COUNT:{len(rows)}")
    require(len(scale_rows) == 9 * AUDIT_ROWS, f"SCALE_ROW_COUNT:{len(scale_rows)}")
    require(
        {(int(x["seed"]), str(x["pressure"])) for x in cell_summaries} == set(CELLS),
        "CELL_MATRIX",
    )

    cell_summaries.sort(key=lambda x: (int(x["seed"]), PRESSURES.index(str(x["pressure"]))))
    rows.sort(key=lambda x: (
        int(x["seed"]), PRESSURES.index(str(x["pressure"])), str(x["row_id"])
    ))
    scale_rows.sort(key=lambda x: (
        int(x["seed"]), PRESSURES.index(str(x["pressure"])), str(x["row_id"])
    ))

    output_root.mkdir(parents=True, exist_ok=False)
    rows_path = output_root / ROWS_FILE
    with rows_path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))

    summary = {
        "schema_version": "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_D_DOWNSTREAM_IMAGE_V1",
        "result": "PASS_GEN5_STAGE_D_DOWNSTREAM_IMAGE_MATRIX",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "stage_c_freeze_commit": STAGE_C_FREEZE_COMMIT,
        "stage_b_freeze_commit": STAGE_B_FREEZE_COMMIT,
        "source_phase3a_execution_commit": PHASE3A_EXECUTION_COMMIT,
        "population": "FROZEN_PHASE3A_DEV_STRESSOR_DOMAIN_480_ROWS",
        "baseline_trajectory": "PHASE3A_ZERO_CORRECTION_DEV_EVAL",
        "primary_radius": PRIMARY_RADIUS,
        "audit_radius": AUDIT_RADIUS,
        "audit_rows_per_cell": AUDIT_ROWS,
        "cell_count": 9,
        "rows_per_cell": EXPECTED_STRESSOR_ROWS,
        "row_record_count": len(rows),
        "cells": cell_summaries,
        "scale_audit_rows": scale_rows,
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "gpu_queues": {
            str(k): [[s, p] for s, p in v]
            for k, v in GPU_QUEUES.items()
        },
        "gpu_runtime": gpu_meta,
        "rows_file_sha256": sha256_file(rows_path),
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "task_evaluation_executed": False,
        "confirmatory_9601_9900_loaded": False,
        "scientific_p_value_count": 0,
        "scientific_conclusion": None,
    }
    summary_path = output_root / SUMMARY_FILE
    summary_path.write_bytes(canonical_json_bytes(summary))

    provenance = {
        "schema_version": "GEN5_OPTIMIZATION_PATH_BYPASS_STAGE_D_PROVENANCE_V1",
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "execution_authority_commit": args.execution_authority_commit,
        "summary_sha256": sha256_file(summary_path),
        "rows_sha256": sha256_file(rows_path),
        "cell_count": 9,
        "row_record_count": len(rows),
        "gpu_topology": "TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP",
        "training_executed": False,
        "backward_executed": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "confirmatory_9601_9900_loaded": False,
    }
    provenance_path = output_root / PROVENANCE_FILE
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    shutil.rmtree(scratch_root)

    print("GEN5_STAGE_D_DOWNSTREAM_IMAGE_MATRIX_PASS")
    print("CELLS=9")
    print(f"ROWS_PER_CELL={EXPECTED_STRESSOR_ROWS}")
    print(f"ROW_RECORD_COUNT={len(rows)}")
    print("GPU_WORKERS=2")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("BASELINE_TRAJECTORY=PHASE3A_ZERO_CORRECTION_DEV_EVAL")
    print("BACKWARD_EXECUTED=False")
    print("OPTIMIZER_CONSTRUCTED=False")
    print("OPTIMIZER_STEP_COUNT=0")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print(f"SUMMARY={summary_path}")
    print(f"ROWS={rows_path}")
    print(f"PROVENANCE={provenance_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-verify-only", action="store_true")
    modes.add_argument("--run-matrix", action="store_true")
    modes.add_argument("--run-worker", action="store_true")

    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--implementation-freeze-commit")
    parser.add_argument("--execution-authority-commit")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--tokenizer-snapshot", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--worker-id", type=int)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.static_verify_only:
        for name in (
            "implementation_freeze_commit",
            "execution_authority_commit",
            "model_snapshot",
            "tokenizer_snapshot",
            "checkpoint",
            "output_root",
            "scratch_root",
            "worker_id",
        ):
            require(getattr(args, name) is None, f"STATIC_FORBIDDEN:{name}")
        return

    for name in (
        "implementation_freeze_commit",
        "execution_authority_commit",
        "model_snapshot",
        "tokenizer_snapshot",
        "checkpoint",
    ):
        require(getattr(args, name) is not None, f"RUNTIME_REQUIRED:{name}")

    if args.run_matrix:
        require(args.output_root is not None, "MATRIX_OUTPUT_REQUIRED")
        require(args.scratch_root is None, "MATRIX_SCRATCH_FORBIDDEN")
        require(args.worker_id is None, "MATRIX_WORKER_FORBIDDEN")
    else:
        require(args.output_root is None, "WORKER_OUTPUT_FORBIDDEN")
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.static_verify_only:
        run_static_verify(args)
    elif args.run_matrix:
        run_matrix(args)
    else:
        run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
