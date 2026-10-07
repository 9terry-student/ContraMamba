#!/usr/bin/env python3
"""Fixed raw-write transport assay for native downstream functional reconvergence.

This runner implements the frozen contract at:
reports/native_downstream_functional_reconvergence_raw_write_transport_
implementation_preexecution_contract_candidate.md

It deliberately reuses the already-validated Gen5 raw-write replay,
true-forward projector, and ordered internal-stage helpers. It does not train,
mutate checkpoints, access the confirmatory population, or refit any downstream
task-visible basis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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

from scripts import audit_reason_router_gen5_ainit_temporal_birth as legacy  # noqa: E402
from scripts import train_reason_router_gen5_ainit_rng_causal_intervention as base  # noqa: E402
from scripts import train_reason_router_gen5_phase3a_contention as p3a  # noqa: E402


EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

DESIGN_COMMIT = "c9459d6788abdf4d62cfb38c5b2c2c2075dc9638"
DESIGN_PATH = (
    "reports/native_downstream_functional_reconvergence_"
    "raw_write_transport_design_candidate.md"
)
CONTRACT_COMMIT = "15e4077e58b0ece5143863a94ae1ffae9a307149"
CONTRACT_PATH = (
    "reports/native_downstream_functional_reconvergence_"
    "raw_write_transport_implementation_preexecution_contract_candidate.md"
)

LEGACY_HELPER_PATH = "scripts/audit_reason_router_gen5_ainit_temporal_birth.py"
LEGACY_HELPER_BLOB = "bbe2f4b0265fc0aba9c5ac87dec9545870035aea"
BASE_RUNTIME_PATH = "scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py"
BASE_RUNTIME_BLOB = "ae5066595f62089ef0844895441f7b30b663b5f3"
P3A_RUNTIME_PATH = "scripts/train_reason_router_gen5_phase3a_contention.py"
P3A_RUNTIME_BLOB = "45e333128c4fa31f4f502c5fdcf324243c1084fd"

AUTHORIZED_IMPLEMENTATION_PATHS = frozenset(
    {
        "scripts/audit_native_downstream_functional_reconvergence_raw_write_transport.py",
        "tests/test_native_downstream_functional_reconvergence_raw_write_transport.py",
    }
)

FACTOR_SEEDS = (6201, 6202, 6203)
FULL_FACTORIAL_CELLS = tuple(
    (a_seed, r_seed)
    for a_seed in FACTOR_SEEDS
    for r_seed in FACTOR_SEEDS
)

STAGE_ORDER = (
    "raw_write",
    "recurrent_state",
    "c_readout_pre_gate",
    "gated_scan",
    "layer22_out_proj",
)

WORKER_ROW_RANGES = {
    0: (0, 416),
    1: (416, 840),
}

CHECKPOINT_SHA256 = {
    (6201, 6201): "157ca1c945c7f70b03638ef7e06750a4f4105e5504904272529ecd82bfba5ddf",
    (6201, 6202): "ef03fbbedf3fab8efb255f2ccb6ec33cdb65ee6881a92e6b4d43fe1e719e40d4",
    (6201, 6203): "15582eda034befb1c8d202f04c494f7fd5882bf9fd60ddc963232761057b3df7",
    (6202, 6201): "3c217b43eb164583980bee91c39d53a7a3dc20d181e0341db35ffe8ece162ed3",
    (6202, 6202): "1aa16196aa5aa338c30ac71dd20c7a6fb36a62eafdbe52bcf5696cfc3b87c214",
    (6202, 6203): "d1c478b448c5f53e2a97552196a454f63f9d9081de614a6d9a5cd4e389e30ee2",
    (6203, 6201): "d9e1062baf554867b212da57c9d30fe8efeccafc77255ba869affac691606359",
    (6203, 6202): "32943df3558ef72eb7f7b7bfb70c6a1a03185a1ea6ca4dd83b9677cd59d13698",
    (6203, 6203): "c536464dd8541423d18a2bfbfee40211f068885baa37730315dbf7cf0a784770",
}

PARENT_CHECKPOINT_SHA256 = (
    "1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f"
)
DEV_ORDER_SHA256 = (
    "b42f64ec4961907fb59eb5fdf9e2e1714649b7952e551c4e9abf7e99c1456e25"
)
DEV_ENCODING_SHA256 = (
    "e3162804bfd184907ee1b22b3f4b4cf3ecee1069fe55661a1a4dbaeecb2cca51"
)
DEV_ROWS = 840
VALID_TOKEN_COUNT = 60094
BATCH_ROWS = 32
TARGETS_PER_TRANSPORT = 2
ENERGY_EPSILON = 1.0e-30

PROJECTOR_PINV_RTOL = 1.0e-12
PROJECTOR_PINV_ATOL = 0.0
PROJECTOR_BACKEND = "gpu_f32_product_f64_reduce_cpu_2x2_pinv"

JOINT_EDGE_FORWARD_ATOL = 5.0e-6
SOURCE_TARGET_REPLAY_ATOL = 5.0e-5
RESIDUAL_CHAIN_ATOL = 5.0e-4
RAW_WRITE_PROJECTOR_ATOL = 5.0e-5
RAW_WRITE_EFFECT_ATOL = 2.0e-3
RAW_RECON_ATOL = 5.0e-5

RESIDUAL_CHAIN_TARGETS = {
    "raw_write": 0.459225933132,
    "recurrent_state": 0.235282854833,
    "c_readout_pre_gate": 0.201189474462,
    "gated_scan": 0.139629099045,
    "layer22_out_proj": 0.155009066845,
}
RAW_WRITE_PROJECTOR_TARGET = 0.000323695863574
PRECURSOR_SUMMARY_SHA256 = "25c3e86d8d4ebbd19f7945755b5ca8033d5b05b0a6dd2a6804709b61c34293d2"
RAW_WRITE_MARGIN_TARGETS = {
    "R_visible": 0.894958000002,
    "R_complement": 0.00440660190005,
    "R_interaction": 0.00843951843385,
}

OUTPUT_SCHEMA = "GEN5_NATIVE_RECONVERGENCE_RAW_WRITE_TRANSPORT_V1"
WORKER_SCHEMA = "GEN5_NATIVE_RECONVERGENCE_RAW_WRITE_TRANSPORT_WORKER_V1"
PROVENANCE_SCHEMA = "GEN5_NATIVE_RECONVERGENCE_RAW_WRITE_TRANSPORT_PROVENANCE_V1"


class TransportError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransportError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def status_paths() -> set[str]:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )
    result: set[str] = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        require(len(line) >= 4, f"MALFORMED_STATUS:{line!r}")
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result.add(path)
    return result


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
    return f"A{cell[0]}-R{cell[1]}"


def pair_group(
    source: tuple[int, int],
    target: tuple[int, int],
) -> str | None:
    if source == target:
        return None
    same_a = source[0] == target[0]
    same_r = source[1] == target[1]
    if same_r and not same_a:
        return "PRIMARY_A"
    if same_a and not same_r:
        return "CONTROL_R"
    return None


def all_orientations() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    rows = []
    for source in FULL_FACTORIAL_CELLS:
        for target in FULL_FACTORIAL_CELLS:
            group = pair_group(source, target)
            if group is not None:
                rows.append((group, source, target))
    require(
        sum(1 for group, _, _ in rows if group == "PRIMARY_A") == 18,
        "PRIMARY_ORIENTATION_COUNT",
    )
    require(
        sum(1 for group, _, _ in rows if group == "CONTROL_R") == 18,
        "CONTROL_ORIENTATION_COUNT",
    )
    require(len(rows) == 36, "TOTAL_ORIENTATION_COUNT")
    return tuple(rows)


def worker_row_range(worker_id: int) -> tuple[int, int]:
    require(worker_id in (0, 1), f"WORKER_ID:{worker_id}")
    return WORKER_ROW_RANGES[worker_id]


def worker_orientations(
    worker_id: int,
) -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    require(worker_id in (0, 1), f"WORKER_ID:{worker_id}")
    return all_orientations()


def _assert_blob(path: str, expected_blob: str) -> None:
    live = git("rev-parse", f"HEAD:{path}")
    require(live == expected_blob, f"BLOB_DRIFT:{path}:{live}")


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

    require(
        git_rc("merge-base", "--is-ancestor", DESIGN_COMMIT, head) == 0,
        "DESIGN_NOT_ANCESTOR",
    )
    require(
        git_rc("merge-base", "--is-ancestor", CONTRACT_COMMIT, head) == 0,
        "CONTRACT_NOT_ANCESTOR",
    )

    frozen_design_blob = git("rev-parse", f"{DESIGN_COMMIT}:{DESIGN_PATH}")
    live_design_blob = git("rev-parse", f"HEAD:{DESIGN_PATH}")
    require(frozen_design_blob == live_design_blob, "DESIGN_BLOB_DRIFT")

    frozen_contract_blob = git("rev-parse", f"{CONTRACT_COMMIT}:{CONTRACT_PATH}")
    live_contract_blob = git("rev-parse", f"HEAD:{CONTRACT_PATH}")
    require(frozen_contract_blob == live_contract_blob, "CONTRACT_BLOB_DRIFT")

    _assert_blob(LEGACY_HELPER_PATH, LEGACY_HELPER_BLOB)
    _assert_blob(BASE_RUNTIME_PATH, BASE_RUNTIME_BLOB)
    _assert_blob(P3A_RUNTIME_PATH, P3A_RUNTIME_BLOB)

    dirty = status_paths()
    if allow_implementation_worktree:
        require(
            dirty <= AUTHORIZED_IMPLEMENTATION_PATHS,
            f"IMPLEMENTATION_SCOPE:{sorted(dirty)}",
        )
    else:
        require(not dirty, f"WORKTREE_NOT_CLEAN:{sorted(dirty)}")
    return head


def validate_static_contract() -> None:
    require(FACTOR_SEEDS == legacy.FACTOR_SEEDS, "FACTOR_SEEDS")
    require(FULL_FACTORIAL_CELLS == legacy.FULL_FACTORIAL_CELLS, "FULL_GRID")
    require(STAGE_ORDER == legacy.TEMPORAL_MECHANISM_INTERNAL_STAGES, "STAGE_ORDER")
    require(DEV_ROWS == legacy.DEV_ROWS == p3a.DEV_ROWS, "DEV_ROWS")
    require(VALID_TOKEN_COUNT == legacy.PHASE_B_VALID_TOKEN_COUNT, "VALID_TOKENS")
    require(PROJECTOR_PINV_RTOL == legacy.PHASE_B_PROJECTOR_PINV_RTOL, "PROJECTOR_RTOL")
    require(PROJECTOR_BACKEND == "gpu_f32_product_f64_reduce_cpu_2x2_pinv", "PROJECTOR_BACKEND")
    for cell in FULL_FACTORIAL_CELLS:
        path, frozen_sha, _schema = legacy.FROZEN_FINAL_SOURCES[cell]
        del path
        require(frozen_sha == CHECKPOINT_SHA256[cell], f"CHECKPOINT_SHA_BINDING:{cell_name(cell)}")
    a0,b0=worker_row_range(0); a1,b1=worker_row_range(1)
    require((a0,b0,a1,b1)==(0,416,416,DEV_ROWS),"WORKER_ROW_RANGES")
    expected=set(all_orientations())
    require(set(worker_orientations(0))==expected,"WORKER0_ORIENTATION_COVERAGE")
    require(set(worker_orientations(1))==expected,"WORKER1_ORIENTATION_COVERAGE")


def _runtime_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    torch.nn.Module,
    Any,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    static, encoded, snapshot, checkpoint_path = base._prepare_runtime_inputs(args)

    require(len(static["dev_rows"]) == DEV_ROWS, "RUNTIME_DEV_ROWS")
    require(
        p3a.row_order_sha256(static["dev_rows"]) == DEV_ORDER_SHA256,
        "DEV_ORDER_SHA",
    )
    require(
        encoded["dev_encoding_sha256"] == DEV_ENCODING_SHA256,
        "DEV_ENCODING_SHA",
    )
    require(
        sha256_file(Path(checkpoint_path)) == PARENT_CHECKPOINT_SHA256,
        "PARENT_CHECKPOINT_SHA",
    )

    model, wrapper, _runtime_meta, strong_mask, planes = base._prepare_runtime_model(
        snapshot=snapshot,
        checkpoint_path=checkpoint_path,
        a_init_seed=6201,
        training_rng_seed=6201,
    )
    model.eval()
    model.mamba.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    weights: dict[str, Any] = {}
    for cell in FULL_FACTORIAL_CELLS:
        payload = legacy.load_frozen_final_checkpoint(*cell)
        require(
            payload["file_sha256"] == CHECKPOINT_SHA256[cell],
            f"CHECKPOINT_LOAD_SHA:{cell_name(cell)}",
        )
        weights[cell_name(cell)] = payload

    return static, encoded, model, wrapper, strong_mask, planes


def _valid_stage(
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    require(value.ndim == 3, "STAGE_RANK")
    require(
        attention_mask.ndim == 2
        and tuple(attention_mask.shape) == tuple(value.shape[:2]),
        "STAGE_MASK_SHAPE",
    )
    mask = attention_mask.bool().unsqueeze(-1)
    return torch.where(mask, value, torch.zeros_like(value))


def _sq_energy_by_example(
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    masked = _valid_stage(value, attention_mask)
    flat = masked.to(device="cpu", dtype=torch.float64).reshape(masked.shape[0], -1)
    return torch.sum(flat * flat, dim=1)


def _stage_energy_sums(
    *,
    source: torch.Tensor,
    full: torch.Tensor,
    visible: torch.Tensor,
    complement: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, float]:
    d_full = _valid_stage(full - source, attention_mask)
    d_visible = _valid_stage(visible - source, attention_mask)
    d_complement = _valid_stage(complement - source, attention_mask)
    interaction = d_visible + d_complement - d_full

    return {
        "D_full": float(_sq_energy_by_example(d_full, attention_mask).sum().item()),
        "D_visible": float(_sq_energy_by_example(d_visible, attention_mask).sum().item()),
        "D_complement": float(_sq_energy_by_example(d_complement, attention_mask).sum().item()),
        "I_abs": float(_sq_energy_by_example(interaction, attention_mask).sum().item()),
    }


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
        "task_row_energy_sum": 0.0,
        "task_row_energy_numer_sum": 0.0,
        "task_row_energy_denom_sum": 0.0,
        "raw_reconstruction_max_abs": 0.0,
        "stage": {
            stage: {
                "D_full": 0.0,
                "D_visible": 0.0,
                "D_complement": 0.0,
                "I_abs": 0.0,
            }
            for stage in STAGE_ORDER
        },
        "finite": {
            "centered_logits": {
                "E_full": 0.0,
                "E_visible": 0.0,
                "E_complement": 0.0,
                "E_interaction": 0.0,
            },
            "two_margins": {
                "E_full": 0.0,
                "E_visible": 0.0,
                "E_complement": 0.0,
                "E_interaction": 0.0,
            },
        },
        "visible_prediction_disagreement_vs_source": 0,
        "complement_prediction_disagreement_vs_source": 0,
    }


def _prepare_projector_pinv(
    *,
    grad_refute: torch.Tensor,
    grad_support: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(
        grad_refute.shape == grad_support.shape,
        "PROJECTOR_GRAD_SHAPE",
    )
    require(
        attention_mask.ndim == 2
        and tuple(attention_mask.shape) == tuple(grad_refute.shape[:2]),
        "PROJECTOR_MASK_SHAPE",
    )

    valid = attention_mask.to(
        device=grad_refute.device,
        dtype=grad_refute.dtype,
    ).unsqueeze(-1)

    # Detached analysis gradients: mask in place to avoid duplicate state-width
    # tensors on GPU.
    grad_refute.mul_(valid)
    grad_support.mul_(valid)

    reduce_dims = tuple(range(1, grad_refute.ndim))
    g00 = torch.sum(
        grad_refute * grad_refute,
        dim=reduce_dims,
        dtype=torch.float64,
    )
    g01 = torch.sum(
        grad_refute * grad_support,
        dim=reduce_dims,
        dtype=torch.float64,
    )
    g11 = torch.sum(
        grad_support * grad_support,
        dim=reduce_dims,
        dtype=torch.float64,
    )

    gram_cpu = torch.stack(
        (
            torch.stack((g00, g01), dim=-1),
            torch.stack((g01, g11), dim=-1),
        ),
        dim=-2,
    ).to(device="cpu")

    pinv_cpu = torch.linalg.pinv(
        gram_cpu,
        rtol=PROJECTOR_PINV_RTOL,
        atol=PROJECTOR_PINV_ATOL,
        hermitian=True,
    )
    return grad_refute, grad_support, pinv_cpu


def _project_visible_group(
    *,
    grad_refute: torch.Tensor,
    grad_support: torch.Tensor,
    pinv_cpu: torch.Tensor,
    attention_mask: torch.Tensor,
    raw_source: torch.Tensor,
    raw_targets: Sequence[torch.Tensor],
    accumulators: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    require(len(raw_targets) == len(accumulators), "PROJECTOR_GROUP_LENGTH")
    require(1 <= len(raw_targets) <= TARGETS_PER_TRANSPORT, "PROJECTOR_GROUP_SIZE")

    residuals = torch.stack(
        [target - raw_source for target in raw_targets],
        dim=0,
    )
    valid = attention_mask.to(
        device=residuals.device,
        dtype=residuals.dtype,
    ).unsqueeze(0).unsqueeze(-1)
    residuals.mul_(valid)

    jd0 = torch.sum(
        residuals * grad_refute.unsqueeze(0),
        dim=(2, 3),
        dtype=torch.float64,
    )
    jd1 = torch.sum(
        residuals * grad_support.unsqueeze(0),
        dim=(2, 3),
        dtype=torch.float64,
    )
    denom = torch.sum(
        residuals * residuals,
        dim=(2, 3),
        dtype=torch.float64,
    )

    jd_cpu = torch.stack((jd0, jd1), dim=-1).to(device="cpu")
    denom_cpu = denom.to(device="cpu")
    alpha_cpu = torch.einsum("bij,kbj->kbi", pinv_cpu, jd_cpu)

    # For the orthogonal row-space projection, ||J^T alpha||^2 = alpha^T Jd.
    # This keeps the exact 2x2 CPU solve while avoiding any second pseudoinverse.
    numer_cpu = torch.sum(alpha_cpu * jd_cpu, dim=-1)

    alpha = alpha_cpu.to(
        device=raw_source.device,
        dtype=raw_source.dtype,
    )
    visible = (
        alpha[..., 0, None, None] * grad_refute.unsqueeze(0)
        + alpha[..., 1, None, None] * grad_support.unsqueeze(0)
    )

    # Preserve the masked FULL residual for exact reconstruction and
    # raw-write energy accounting.
    complement = residuals - visible

    for index, accumulator in enumerate(accumulators):
        denom_row = denom_cpu[index]
        numer_row = numer_cpu[index]
        ratio_row = torch.where(
            denom_row > 0.0,
            numer_row / denom_row,
            torch.zeros_like(denom_row),
        )
        accumulator["task_row_energy_sum"] += float(ratio_row.sum().item())
        accumulator["task_row_energy_numer_sum"] += float(numer_row.sum().item())
        accumulator["task_row_energy_denom_sum"] += float(denom_row.sum().item())

        full = residuals[index]
        interaction = visible[index] + complement[index] - full

        raw_metrics = {
            "D_full": float(torch.sum(full * full, dtype=torch.float64).item()),
            "D_visible": float(
                torch.sum(visible[index] * visible[index], dtype=torch.float64).item()
            ),
            "D_complement": float(
                torch.sum(
                    complement[index] * complement[index],
                    dtype=torch.float64,
                ).item()
            ),
            "I_abs": float(
                torch.sum(interaction * interaction, dtype=torch.float64).item()
            ),
        }
        for key, value in raw_metrics.items():
            accumulator["stage"]["raw_write"][key] += value

        recon_error = float(torch.max(torch.abs(interaction)).item())
        accumulator["raw_reconstruction_max_abs"] = max(
            float(accumulator["raw_reconstruction_max_abs"]),
            recon_error,
        )
        require(recon_error <= RAW_RECON_ATOL, f"RAW_RECON:{recon_error}")

    return visible, complement

def _accumulate_finite(
    accumulator: dict[str, Any],
    *,
    source_logits: torch.Tensor,
    full_logits: torch.Tensor,
    visible_logits: torch.Tensor,
    complement_logits: torch.Tensor,
) -> None:
    effects = legacy._phase_b_effect_sums(
        source_logits,
        full_logits,
        visible_logits,
        complement_logits,
    )
    for coordinate in ("centered_logits", "two_margins"):
        for key in ("E_full", "E_visible", "E_complement", "E_interaction"):
            accumulator["finite"][coordinate][key] += float(effects[coordinate][key])

    source_pred = torch.argmax(source_logits, dim=-1)
    visible_pred = torch.argmax(visible_logits, dim=-1)
    complement_pred = torch.argmax(complement_logits, dim=-1)
    accumulator["visible_prediction_disagreement_vs_source"] += int(
        torch.count_nonzero(source_pred != visible_pred).item()
    )
    accumulator["complement_prediction_disagreement_vs_source"] += int(
        torch.count_nonzero(source_pred != complement_pred).item()
    )


def _finalize_orientation(value: Mapping[str, Any]) -> dict[str, Any]:
    count = int(value["example_count"])
    require(count == DEV_ROWS, f"ORIENTATION_EXAMPLE_COUNT:{count}")

    raw_visible = float(value["stage"]["raw_write"]["D_visible"])
    raw_complement = float(value["stage"]["raw_write"]["D_complement"])

    stage_rows: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        row = dict(value["stage"][stage])
        ret_visible = row["D_visible"] / max(raw_visible, ENERGY_EPSILON)
        ret_complement = row["D_complement"] / max(raw_complement, ENERGY_EPSILON)
        selective = ret_complement / max(ret_visible, ENERGY_EPSILON)
        i_rel = row["I_abs"] / max(row["D_full"], ENERGY_EPSILON)
        stage_rows[stage] = {
            **row,
            "RET_VISIBLE": ret_visible,
            "RET_COMPLEMENT": ret_complement,
            "SELECTIVE_RETENTION": selective,
            "LOG_SELECTIVE_RETENTION": (
                None if selective <= 0.0 else math.log(selective)
            ),
            "I_REL": i_rel,
            "sum_D_full": row["D_full"],
        }

    finite = {}
    for coordinate, metrics in value["finite"].items():
        denom = max(float(metrics["E_full"]), ENERGY_EPSILON)
        finite[coordinate] = {
            **metrics,
            "R_visible": float(metrics["E_visible"]) / denom,
            "R_complement": float(metrics["E_complement"]) / denom,
            "R_interaction": float(metrics["E_interaction"]) / denom,
        }

    return {
        "group": value["group"],
        "source": value["source"],
        "target": value["target"],
        "example_count": count,
        "task_row_energy_mean": float(value["task_row_energy_sum"]) / count,
        "task_row_energy_ratio_of_sums": (
            float(value["task_row_energy_numer_sum"])
            / max(float(value["task_row_energy_denom_sum"]), ENERGY_EPSILON)
        ),
        "task_row_energy_numer_sum": float(value["task_row_energy_numer_sum"]),
        "task_row_energy_denom_sum": float(value["task_row_energy_denom_sum"]),
        "raw_reconstruction_max_abs": float(value["raw_reconstruction_max_abs"]),
        "stage": stage_rows,
        "finite": finite,
        "visible_prediction_disagreement_vs_source": int(
            value["visible_prediction_disagreement_vs_source"]
        ),
        "complement_prediction_disagreement_vs_source": int(
            value["complement_prediction_disagreement_vs_source"]
        ),
    }


def _cell_weights() -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    result = {}
    for cell in FULL_FACTORIAL_CELLS:
        payload = legacy.load_frozen_final_checkpoint(*cell)
        result[cell] = (
            payload["A_theta.weight"],
            payload["B_theta.weight"],
        )
    return result


def _prepare_worker_runtime(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    torch.nn.Module,
    Any,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    _static, encoded, model, wrapper, strong_mask, planes = _runtime_inputs(args)
    device = torch.device("cuda:0")
    features, _labels, active, targets = p3a._feature_batch_to_device(
        encoded["dev_bundle"],
        device,
    )
    return encoded, model, wrapper, strong_mask, planes, features, active, targets


def _source_gradient_bundle(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    raw_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leaf = raw_source.detach().clone().requires_grad_(True)
    source_logits = legacy._phase_b_resume_from_raw_write(
        model=model,
        wrapper=wrapper,
        features=features,
        context=context,
        raw_write=leaf,
    )
    margins = legacy._phase_b_margin_vector(source_logits)
    g_refute = torch.autograd.grad(
        margins[:, 0].sum(),
        leaf,
        retain_graph=True,
        create_graph=False,
    )[0]
    g_support = torch.autograd.grad(
        margins[:, 1].sum(),
        leaf,
        retain_graph=False,
        create_graph=False,
    )[0]
    require(
        not any(parameter.grad is not None for parameter in model.parameters()),
        "PARAMETER_GRADIENT_ACCUMULATED",
    )
    return source_logits.detach(), g_refute.detach(), g_support.detach()


def _stage_final_logits(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    stages: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return legacy.temporal_mechanism_resume_from_stage(
        model=model,
        wrapper=wrapper,
        features=features,
        context=context,
        stage="layer22_out_proj",
        stage_value=stages["layer22_out_proj"],
    )


def _trajectory_stage_energy(
    value: torch.Tensor,
    *,
    target_count: int,
    valid_token: torch.Tensor,
) -> torch.Tensor:
    require(value.ndim == 3, "STREAM_STAGE_RANK")
    expected = 1 + 3 * target_count
    require(value.shape[0] == expected, "STREAM_STAGE_TRAJECTORY_COUNT")

    source = value[0].unsqueeze(0)
    full = value[1 : 1 + target_count]
    visible = value[1 + target_count : 1 + 2 * target_count]
    complement = value[1 + 2 * target_count : 1 + 3 * target_count]

    d_full = full - source
    d_visible = visible - source
    d_complement = complement - source
    interaction = d_visible + d_complement - d_full

    diffs = torch.stack(
        (d_full, d_visible, d_complement, interaction),
        dim=0,
    )
    valid = valid_token.to(
        device=value.device,
        dtype=value.dtype,
    ).reshape(1, 1, value.shape[1], 1)
    diffs = diffs * valid

    return torch.sum(
        diffs * diffs,
        dim=(2, 3),
        dtype=torch.float64,
    )


def _repeat_feature_batch(
    features: Mapping[str, torch.Tensor],
    copies: int,
) -> dict[str, torch.Tensor]:
    batch = next(iter(features.values())).shape[0]
    output: dict[str, torch.Tensor] = {}
    for key, value in features.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == batch
        ):
            output[key] = (
                value.unsqueeze(0)
                .expand(copies, *value.shape)
                .reshape(copies * batch, *value.shape[1:])
            )
        else:
            output[key] = value
    return output


def _stream_transport_group(
    *,
    model: torch.nn.Module,
    wrapper: Any,
    features: Mapping[str, torch.Tensor],
    context: Mapping[str, torch.Tensor],
    attention_mask: torch.Tensor,
    raw_source: torch.Tensor,
    raw_targets: Sequence[torch.Tensor],
    visible: torch.Tensor,
    complement: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, dict[str, float]]],
]:
    target_count = len(raw_targets)
    require(1 <= target_count <= TARGETS_PER_TRANSPORT, "STREAM_TARGET_COUNT")
    require(visible.shape[0] == target_count, "STREAM_VISIBLE_COUNT")
    require(complement.shape[0] == target_count, "STREAM_COMPLEMENT_COUNT")

    native_mixer = wrapper.native_mixer
    shape = wrapper.correction.shape
    batch, seq_len, width = raw_source.shape
    trajectory_count = 1 + 3 * target_count
    require(width == shape.state_width, "STREAM_RAW_WIDTH")

    state = torch.zeros(
        (
            trajectory_count,
            batch,
            shape.intermediate_size,
            shape.state_size,
        ),
        device=raw_source.device,
        dtype=raw_source.dtype,
    )

    stage_energy = {
        stage: torch.zeros(
            (4, target_count),
            device=raw_source.device,
            dtype=torch.float64,
        )
        for stage in STAGE_ORDER[1:]
    }
    out_rows: list[torch.Tensor] = []
    gate_all = native_mixer.act(context["gate"])

    with torch.no_grad():
        for token_index in range(seq_len):
            write = torch.cat(
                (
                    raw_source[:, token_index, :].unsqueeze(0),
                    torch.stack(
                        [target[:, token_index, :] for target in raw_targets],
                        dim=0,
                    ),
                    visible[:, :, token_index, :],
                    complement[:, :, token_index, :],
                ),
                dim=0,
            )

            discrete_a_t = torch.exp(
                context["a_continuous"][None, :, :]
                * context["discrete_time_step"][
                    :, :, token_index, None
                ].float()
            ).to(dtype=raw_source.dtype)

            state = (
                discrete_a_t.unsqueeze(0) * state
                + write.reshape(
                    trajectory_count,
                    batch,
                    shape.intermediate_size,
                    shape.state_size,
                )
            )

            valid_token = attention_mask[:, token_index]

            recurrent = state.reshape(
                trajectory_count,
                batch,
                shape.state_width,
            )
            stage_energy["recurrent_state"] += _trajectory_stage_energy(
                recurrent,
                target_count=target_count,
                valid_token=valid_token,
            )

            read = torch.sum(
                state.to(context["c_readout"].dtype)
                * context["c_readout"][
                    :, token_index, None, :
                ].unsqueeze(0),
                dim=-1,
            )
            stage_energy["c_readout_pre_gate"] += _trajectory_stage_energy(
                read,
                target_count=target_count,
                valid_token=valid_token,
            )

            gated = read * gate_all[:, :, token_index].unsqueeze(0)
            stage_energy["gated_scan"] += _trajectory_stage_energy(
                gated,
                target_count=target_count,
                valid_token=valid_token,
            )

            out = torch.nn.functional.linear(
                gated,
                native_mixer.out_proj.weight,
                bias=None,
            )
            stage_energy["layer22_out_proj"] += _trajectory_stage_energy(
                out,
                target_count=target_count,
                valid_token=valid_token,
            )
            out_rows.append(out)

        correction22 = torch.stack(out_rows, dim=2)
        base22 = context["residual22"] + context["native22"]
        hidden22 = (
            base22.unsqueeze(0) + correction22
        ).reshape(
            trajectory_count * batch,
            seq_len,
            correction22.shape[-1],
        )

        hidden23 = model.mamba.layers[23](
            hidden22,
            cache_params=None,
            cache_position=None,
            attention_mask=None,
        )
        final_hidden = model.mamba.norm_f(hidden23)
        expanded_features = _repeat_feature_batch(features, trajectory_count)
        output = legacy._phase_b_joint_forward_from_hidden(
            model,
            expanded_features,
            final_hidden,
        )
        logits = output["logits"].reshape(
            trajectory_count,
            batch,
            3,
        )

    rows: list[dict[str, dict[str, float]]] = []
    cpu_energy = {
        stage: tensor.detach().to(device="cpu")
        for stage, tensor in stage_energy.items()
    }
    for target_index in range(target_count):
        row: dict[str, dict[str, float]] = {}
        for stage in STAGE_ORDER[1:]:
            values = cpu_energy[stage][:, target_index]
            row[stage] = {
                "D_full": float(values[0].item()),
                "D_visible": float(values[1].item()),
                "D_complement": float(values[2].item()),
                "I_abs": float(values[3].item()),
            }
        rows.append(row)

    source_logits = logits[0]
    full_logits = logits[1 : 1 + target_count]
    visible_logits = logits[
        1 + target_count : 1 + 2 * target_count
    ]
    complement_logits = logits[
        1 + 2 * target_count : 1 + 3 * target_count
    ]
    return (
        source_logits,
        full_logits,
        visible_logits,
        complement_logits,
        rows,
    )


def _expected_counts() -> dict[str, Any]:
    result = {
        "batch_rows": BATCH_ROWS,
        "targets_per_transport": TARGETS_PER_TRANSPORT,
        "sharding_mode": "DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION",
        "workers": {},
    }
    groups_per_source = math.ceil(4 / TARGETS_PER_TRANSPORT)
    for worker_id in (0, 1):
        row_start, row_stop = worker_row_range(worker_id)
        row_count = row_stop - row_start
        batches = math.ceil(row_count / BATCH_ROWS)
        fused_calls = 9 * groups_per_source * batches
        result["workers"][str(worker_id)] = {
            "row_start": row_start,
            "row_stop": row_stop,
            "row_count": row_count,
            "batches": batches,
            "source_cells": 9,
            "orientations": 36,
            "common_context_batches": batches,
            "raw_write_cell_evaluations": 9 * batches,
            "source_gradient_forwards": 9 * batches,
            "fused_transport_calls": fused_calls,
            "fused_transport_trajectories": fused_calls * (1 + 3 * TARGETS_PER_TRANSPORT),
            "batched_final_head_calls": fused_calls,
            "legacy_full_stage_tensor_offloads": 0,
        }
    result["total_orientations"] = 36
    result["total_rows"] = DEV_ROWS
    result["total_common_context_batches"] = sum(x["common_context_batches"] for x in result["workers"].values())
    result["total_source_gradient_forwards"] = sum(x["source_gradient_forwards"] for x in result["workers"].values())
    result["total_fused_transport_calls"] = sum(x["fused_transport_calls"] for x in result["workers"].values())
    return result


def run_static_contract_check(args: argparse.Namespace) -> None:
    head = authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=True,
    )
    validate_static_contract()
    print("GEN5_NATIVE_RECONVERGENCE_STATIC_CONTRACT_PASS")
    print(f"HEAD={head}")
    print("PRIMARY_ORIENTATIONS=18")
    print("CONTROL_ORIENTATIONS=18")
    print("TOTAL_ORIENTATIONS=36")
    print("SHARDING_MODE=DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION")
    print("WORKER0_ROWS=0:416")
    print("WORKER1_ROWS=416:840")
    print("WORKER0_ORIENTATIONS=36")
    print("WORKER1_ORIENTATIONS=36")
    print(f"PROJECTOR_BACKEND={PROJECTOR_BACKEND}")
    print(f"PROJECTOR_PINV_RTOL={PROJECTOR_PINV_RTOL:.17g}")
    print("MODEL_FORWARD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")


def _validate_runtime_head(args: argparse.Namespace) -> str:
    require(args.expected_head is not None, "EXPECTED_HEAD_REQUIRED")
    require(args.implementation_freeze_commit is not None, "IMPLEMENTATION_FREEZE_REQUIRED")
    require(
        args.expected_head == args.implementation_freeze_commit,
        "EXECUTION_HEAD_MUST_EQUAL_IMPLEMENTATION_FREEZE",
    )
    return authenticate_repo(
        expected_head=args.expected_head,
        allow_implementation_worktree=False,
    )


def run_preflight(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    legacy._validate_two_t4s()
    _runtime_inputs(args)
    counts = _expected_counts()
    print("GEN5_NATIVE_RECONVERGENCE_CUDA_PREFLIGHT_PASS")
    print("GPU_TOPOLOGY=TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP")
    print("EXECUTION_ENGINE=FUSED_STREAMING_TRANSPORT_V2")
    print("SHARDING_MODE=DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION")
    print(f"TARGETS_PER_TRANSPORT={TARGETS_PER_TRANSPORT}")
    print("TRAINING_EXECUTED=False")
    print("PARAMETER_GRADIENTS_ACCUMULATED=False")
    print("CONFIRMATORY_9601_9900_LOADED=False")
    print("EXPECTED_FORWARD_COUNTS=" + json.dumps(counts, sort_keys=True))


def _worker_payload_path(scratch_root: Path, worker_id: int) -> Path:
    return scratch_root / f"worker{worker_id}" / "worker_result.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name + ".tmp.",
        delete=False,
    ) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def run_worker(args: argparse.Namespace) -> None:
    _validate_runtime_head(args)
    validate_static_contract()
    require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
    require(args.scratch_root is not None, "SCRATCH_ROOT_REQUIRED")
    require(torch.cuda.is_available(), "CUDA_REQUIRED")
    require(torch.cuda.device_count() == 1, "WORKER_VISIBLE_GPU_COUNT_MUST_BE_ONE")
    worker_id=int(args.worker_id)
    scratch_root=Path(args.scratch_root)
    worker_root=scratch_root/f"worker{worker_id}"
    require(not worker_root.exists(), f"WORKER_OUTPUT_COLLISION:{worker_root}")
    (_encoded,model,wrapper,strong_mask,planes,features,active,targets)=_prepare_worker_runtime(args)
    weights=_cell_weights()
    local_sources=tuple(FULL_FACTORIAL_CELLS)
    orientations=all_orientations()
    accumulators={(g,src,tgt):_empty_orientation(g,src,tgt) for g,src,tgt in orientations}
    by_source={src:tuple(row for row in orientations if row[1]==src) for src in local_sources}
    for src,rows in by_source.items(): require(len(rows)==4,f"SOURCE_ORIENTATION_COUNT:{cell_name(src)}")
    row_start,row_stop=worker_row_range(worker_id)
    row_count=row_stop-row_start
    total_batches=math.ceil(row_count/BATCH_ROWS)
    cell_replay_max={cell_name(cell):0.0 for cell in local_sources}
    valid_token_seen=0
    for batch_index,start in enumerate(range(row_start,row_stop,BATCH_ROWS),start=1):
        stop=min(start+BATCH_ROWS,row_stop)
        batch_features=legacy._phase_b_batch_features(features,start,stop)
        batch_mask=batch_features["attention_mask"]
        valid_token_seen += int(torch.count_nonzero(batch_mask).item())
        context=legacy._phase_b_prepare_common_context(
            model=model,wrapper=wrapper,features=batch_features,
            stressor_active=active[start:stop],target_indices=targets[start:stop],
            strong_mask=strong_mask,planes=planes,
        )
        raw_by_cell={}
        with torch.no_grad():
            for cell in FULL_FACTORIAL_CELLS:
                a_weight,b_weight=weights[cell]
                raw_by_cell[cell]=legacy._phase_b_raw_write(context["mixer_input"],batch_mask,a_weight,b_weight).detach()
        for source in local_sources:
            source_rows=by_source[source]
            raw_source=raw_by_cell[source]
            source_logits_grad,g_refute,g_support=_source_gradient_bundle(
                model=model,wrapper=wrapper,features=batch_features,context=context,raw_source=raw_source,
            )
            g_refute,g_support,pinv_cpu=_prepare_projector_pinv(
                grad_refute=g_refute,grad_support=g_support,attention_mask=batch_mask,
            )
            for group_start in range(0,len(source_rows),TARGETS_PER_TRANSPORT):
                group_rows=source_rows[group_start:group_start+TARGETS_PER_TRANSPORT]
                group_accs=[accumulators[row] for row in group_rows]
                raw_targets=[raw_by_cell[row[2]] for row in group_rows]
                visible,complement=_project_visible_group(
                    grad_refute=g_refute,grad_support=g_support,pinv_cpu=pinv_cpu,
                    attention_mask=batch_mask,raw_source=raw_source,raw_targets=raw_targets,accumulators=group_accs,
                )
                source_logits,full_logits,visible_logits,complement_logits,stage_rows=_stream_transport_group(
                    model=model,wrapper=wrapper,features=batch_features,context=context,attention_mask=batch_mask,
                    raw_source=raw_source,raw_targets=raw_targets,visible=visible,complement=complement,
                )
                replay_error=float(torch.max(torch.abs(source_logits-source_logits_grad)).item())
                source_name=cell_name(source)
                cell_replay_max[source_name]=max(cell_replay_max[source_name],replay_error)
                require(replay_error<=SOURCE_TARGET_REPLAY_ATOL,f"SOURCE_REPLAY_AUTH:{source_name}:{replay_error}")
                for index,row in enumerate(group_rows):
                    acc=accumulators[row]
                    for stage in STAGE_ORDER[1:]:
                        for key,value in stage_rows[index][stage].items(): acc["stage"][stage][key]+=value
                    _accumulate_finite(acc,source_logits=source_logits_grad,full_logits=full_logits[index],visible_logits=visible_logits[index],complement_logits=complement_logits[index])
                    acc["example_count"] += int(stop-start)
                del visible,complement,source_logits,full_logits,visible_logits,complement_logits,stage_rows
            del source_logits_grad,g_refute,g_support,pinv_cpu
        del raw_by_cell
        print("GEN5_TRANSPORT_PROGRESS " f"worker={worker_id} batch={batch_index}/{total_batches} " f"shard_rows={stop-row_start}/{row_count}",flush=True)
    for acc in accumulators.values(): require(int(acc["example_count"])==row_count,"WORKER_ORIENTATION_EXAMPLE_COUNT")
    require(not any(p.grad is not None for p in model.parameters()),"PARAMETER_GRADIENT_ACCUMULATED_FINAL")
    result={
        "schema_version":WORKER_SCHEMA,"execution_head":args.expected_head,
        "implementation_freeze_commit":args.implementation_freeze_commit,"worker_id":worker_id,
        "sharding_mode":"DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION",
        "row_start":row_start,"row_stop":row_stop,"row_count":row_count,
        "source_cells":[cell_name(cell) for cell in local_sources],"orientation_count":36,
        "orientation_accumulators":list(accumulators.values()),"cell_source_replay_max_abs":cell_replay_max,
        "valid_token_count":valid_token_seen,"downstream_map_shared_across_grid":True,
        "downstream_map_reason":"frozen parent/native coefficient path is common; cell-specific checkpoint payload contributes only A_theta/B_theta raw-write weights",
        "execution_engine":"FUSED_STREAMING_TRANSPORT_V2","targets_per_transport":TARGETS_PER_TRANSPORT,
        "full_stage_tensor_cpu_offload":False,"training_executed":False,"optimizer_constructed":False,
        "backward_method_called":False,"parameter_gradients_accumulated":False,"checkpoint_mutation":False,
        "confirmatory_9601_9900_loaded":False,"vitaminc_loaded":False,
    }
    payload_path=_worker_payload_path(scratch_root,worker_id)
    _atomic_write(payload_path,canonical_json_bytes(result))
    digest=sha256_file(payload_path)
    _atomic_write(payload_path.with_suffix(".sha256"),(digest+"\\n").encode("utf-8"))
    print(f"GEN5_NATIVE_RECONVERGENCE_WORKER_PASS worker={worker_id} rows={row_start}:{row_stop} orientations=36 sha256={digest}")

def _read_worker(scratch_root: Path, worker_id: int) -> dict[str, Any]:
    path=_worker_payload_path(scratch_root,worker_id); sidecar=path.with_suffix(".sha256")
    require(path.is_file(),f"WORKER_RESULT_MISSING:{worker_id}"); require(sidecar.is_file(),f"WORKER_SHA_MISSING:{worker_id}")
    require(sidecar.read_text(encoding="utf-8").strip()==sha256_file(path),f"WORKER_SHA_MISMATCH:{worker_id}")
    payload=json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema_version")==WORKER_SCHEMA,"WORKER_SCHEMA"); require(int(payload.get("worker_id",-1))==worker_id,"WORKER_IDENTITY")
    require(payload.get("execution_head")==payload.get("implementation_freeze_commit"),"WORKER_HEAD_BINDING")
    require(payload.get("sharding_mode")=="DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION","WORKER_SHARDING_MODE")
    a,b=worker_row_range(worker_id); require((int(payload["row_start"]),int(payload["row_stop"]))==(a,b),"WORKER_ROW_RANGE")
    require(int(payload["row_count"])==b-a,"WORKER_ROW_COUNT")
    rows=payload.get("orientation_accumulators"); require(isinstance(rows,list) and len(rows)==36,"WORKER_ACCUMULATORS_COUNT")
    ids={(str(x["group"]),str(x["source"]),str(x["target"])) for x in rows}; expected={(g,cell_name(src),cell_name(tgt)) for g,src,tgt in all_orientations()}
    require(len(ids)==36,"WORKER_ORIENTATION_DUPLICATE"); require(ids==expected,"WORKER_ORIENTATION_COVERAGE")
    require(all(int(x["example_count"])==b-a for x in rows),"WORKER_ORIENTATION_EXAMPLE_COUNT")
    require(payload.get("training_executed") is False,"WORKER_TRAINING"); require(payload.get("parameter_gradients_accumulated") is False,"WORKER_PARAM_GRAD")
    require(payload.get("confirmatory_9601_9900_loaded") is False,"WORKER_CONFIRMATORY"); require(payload.get("vitaminc_loaded") is False,"WORKER_VITAMINC")
    return payload


def _merge_worker_accumulators(workers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    triples={(g,cell_name(src),cell_name(tgt)):(g,src,tgt) for g,src,tgt in all_orientations()}
    merged={k:_empty_orientation(*v) for k,v in triples.items()}; worker_ids=set()
    for worker in workers:
        wid=int(worker["worker_id"]); require(wid not in worker_ids,"MERGED_WORKER_DUPLICATE"); worker_ids.add(wid)
        for part in worker["orientation_accumulators"]:
            key=(str(part["group"]),str(part["source"]),str(part["target"])); require(key in merged,"MERGED_ORIENTATION_COVERAGE"); dst=merged[key]
            dst["example_count"]+=int(part["example_count"])
            for k in ("task_row_energy_sum","task_row_energy_numer_sum","task_row_energy_denom_sum"): dst[k]+=float(part[k])
            dst["raw_reconstruction_max_abs"]=max(float(dst["raw_reconstruction_max_abs"]),float(part["raw_reconstruction_max_abs"]))
            for stage in STAGE_ORDER:
                for k in ("D_full","D_visible","D_complement","I_abs"): dst["stage"][stage][k]+=float(part["stage"][stage][k])
            for coord in ("centered_logits","two_margins"):
                for k in ("E_full","E_visible","E_complement","E_interaction"): dst["finite"][coord][k]+=float(part["finite"][coord][k])
            for k in ("visible_prediction_disagreement_vs_source","complement_prediction_disagreement_vs_source"): dst[k]+=int(part[k])
    require(worker_ids=={0,1},"MERGED_WORKER_COVERAGE"); require(len(merged)==36,"MERGED_ORIENTATION_COUNT"); require(len(set(merged))==36,"MERGED_ORIENTATION_DUPLICATE")
    require(set(merged)==set(triples),"MERGED_ORIENTATION_COVERAGE"); require(all(int(x["example_count"])==DEV_ROWS for x in merged.values()),"MERGED_ORIENTATION_EXAMPLE_COUNT")
    return [_finalize_orientation(merged[(g,cell_name(src),cell_name(tgt))]) for g,src,tgt in all_orientations()]


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    require(n > 0, "EMPTY_MEDIAN")
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _quartiles(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(float(x) for x in values)
    require(len(ordered) >= 4, "IQR_REQUIRES_FOUR")
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[-mid:]
    return _median(lower), _median(upper)


def _group_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    values = [
        float(row["stage"][stage]["pair_selective_retention"])
        for row in pair_rows
    ]
    require(len(values) == 9, f"GROUP_PAIR_COUNT:{len(values)}")
    q1, q3 = _quartiles(values)
    return {
        "count": len(values),
        "values": values,
        "median": _median(values),
        "minimum": min(values),
        "maximum": max(values),
        "iqr": [q1, q3],
    }


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    left = str(row["source"])
    right = str(row["target"])
    return str(row["group"]), min(left, right), max(left, right)


def _symmetric_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_pair_key(row), []).append(row)

    result = []
    for key, pair_rows in sorted(grouped.items()):
        require(len(pair_rows) == 2, f"PAIR_ORIENTATION_COVERAGE:{key}:{len(pair_rows)}")
        stage = {}
        for stage_name in STAGE_ORDER:
            logs = [
                row["stage"][stage_name]["LOG_SELECTIVE_RETENTION"]
                for row in pair_rows
            ]
            require(all(value is not None for value in logs), f"NONPOSITIVE_SELECTIVE_RETENTION:{key}:{stage_name}")
            mean_log = sum(float(value) for value in logs) / 2.0
            stage[stage_name] = {
                "pair_log_selective_retention": mean_log,
                "pair_selective_retention": math.exp(mean_log),
            }
        result.append(
            {
                "group": key[0],
                "left": key[1],
                "right": key[2],
                "stage": stage,
            }
        )
    return result


def _source_matched_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    output = []
    require(len(by_source) == 9, "SOURCE_MATCHED_CELL_COUNT")
    for source, source_rows in sorted(by_source.items()):
        primary = [row for row in source_rows if row["group"] == "PRIMARY_A"]
        control = [row for row in source_rows if row["group"] == "CONTROL_R"]
        require(len(primary) == 2 and len(control) == 2, f"SOURCE_MATCH_COUNTS:{source}")
        stage = {}
        for stage_name in STAGE_ORDER:
            primary_log = sum(
                float(row["stage"][stage_name]["LOG_SELECTIVE_RETENTION"])
                for row in primary
            ) / 2.0
            control_log = sum(
                float(row["stage"][stage_name]["LOG_SELECTIVE_RETENTION"])
                for row in control
            ) / 2.0
            stage[stage_name] = {
                "primary_log_mean": primary_log,
                "control_log_mean": control_log,
                "primary_minus_control": primary_log - control_log,
            }
        output.append({"source": source, "stage": stage})
    return output


def _normalized_residual(
    *,
    diff_sq: float,
    source_sq: float,
    target_sq: float,
) -> float:
    denom = 0.5 * (source_sq + target_sq)
    require(denom > 0.0, "RESIDUAL_ZERO_DENOM")
    return math.sqrt(diff_sq / denom)


def _load_precursor_summary() -> Mapping[str, Any]:
    path = (
        ROOT
        / "reports/reason_router_gen5_ainit_internal_precursor_localization_runs"
        / "gen5-ainit-internal-precursor-e2c5631-r1"
        / "ainit_internal_precursor_localization_summary.json"
    )
    require(path.is_file(), "PRECURSOR_SUMMARY_MISSING")
    require(
        sha256_file(path) == PRECURSOR_SUMMARY_SHA256,
        "PRECURSOR_SUMMARY_SHA",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version")
        == "GEN5_AINIT_INTERNAL_PRECURSOR_LOCALIZATION_SUMMARY_V1",
        "PRECURSOR_SUMMARY_SCHEMA",
    )
    require(payload.get("result") == "RAW_WRITE_PRECURSOR", "PRECURSOR_RESULT")
    return payload


def _authentication_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["group"] == "PRIMARY_A"]
    require(len(primary) == 18, "AUTH_PRIMARY_ORIENTATION_COUNT")

    # The prior frozen target is an orientation-symmetric grouped mean.
    # Pair symmetry is therefore used here before the 9-pair mean.
    pair_rows = [
        row
        for row in _symmetric_pairs(primary)
        if row["group"] == "PRIMARY_A"
    ]
    require(len(pair_rows) == 9, "AUTH_PRIMARY_UNORDERED_PAIR_COUNT")

    # Residual-chain and joint-forward authentication are immutable source
    # evidence. The exact summary SHA and the exact legacy helper blob are
    # hard-bound above; this run freshly authenticates source replay while
    # reusing those already-validated invariant targets.
    precursor = _load_precursor_summary()
    frozen_chain = precursor.get("residual_chain_authentication")
    require(isinstance(frozen_chain, Mapping), "PRECURSOR_RESIDUAL_AUTH")

    functional_auth = precursor.get("functional_authentication")
    require(isinstance(functional_auth, Mapping), "PRECURSOR_FUNCTIONAL_AUTH")
    require(functional_auth.get("pass") is True, "PRECURSOR_FUNCTIONAL_AUTH_FAIL")
    require(
        float(functional_auth.get("edge_joint_forward_max_abs", math.inf))
        <= JOINT_EDGE_FORWARD_ATOL,
        "PRECURSOR_EDGE_JOINT_FORWARD_AUTH",
    )

    residual_checks = {}
    for stage in STAGE_ORDER:
        frozen = frozen_chain.get(stage)
        require(isinstance(frozen, Mapping), f"PRECURSOR_STAGE_MISSING:{stage}")
        observed = float(frozen["observed"])
        target = RESIDUAL_CHAIN_TARGETS[stage]
        residual_checks[stage] = {
            "observed": observed,
            "target": target,
            "atol": RESIDUAL_CHAIN_ATOL,
            "abs_error": abs(observed - target),
            "pass": abs(observed - target) <= RESIDUAL_CHAIN_ATOL,
        }

    projector_mean = sum(float(row["task_row_energy_mean"]) for row in primary) / len(primary)
    projector_check = {
        "observed": projector_mean,
        "target": RAW_WRITE_PROJECTOR_TARGET,
        "atol": RAW_WRITE_PROJECTOR_ATOL,
        "abs_error": abs(projector_mean - RAW_WRITE_PROJECTOR_TARGET),
        "pass": abs(projector_mean - RAW_WRITE_PROJECTOR_TARGET) <= RAW_WRITE_PROJECTOR_ATOL,
    }

    aggregate_finite = {
        coordinate: {key: 0.0 for key in ("E_full", "E_visible", "E_complement", "E_interaction")}
        for coordinate in ("centered_logits", "two_margins")
    }
    for row in primary:
        for coordinate in aggregate_finite:
            for key in aggregate_finite[coordinate]:
                aggregate_finite[coordinate][key] += float(row["finite"][coordinate][key])

    margin = aggregate_finite["two_margins"]
    denom = max(margin["E_full"], ENERGY_EPSILON)
    effect_observed = {
        "R_visible": margin["E_visible"] / denom,
        "R_complement": margin["E_complement"] / denom,
        "R_interaction": margin["E_interaction"] / denom,
    }
    effect_checks = {}
    for name, target in RAW_WRITE_MARGIN_TARGETS.items():
        observed = effect_observed[name]
        effect_checks[name] = {
            "observed": observed,
            "target": target,
            "atol": RAW_WRITE_EFFECT_ATOL,
            "abs_error": abs(observed - target),
            "pass": abs(observed - target) <= RAW_WRITE_EFFECT_ATOL,
        }

    passed = (
        all(row["pass"] for row in residual_checks.values())
        and projector_check["pass"]
        and all(row["pass"] for row in effect_checks.values())
    )
    return {
        "pass": bool(passed),
        "residual_chain": residual_checks,
        "raw_write_projector": projector_check,
        "raw_write_two_margin_effect": effect_checks,
        "frozen_functional_authentication": {
            "source_summary_sha256": PRECURSOR_SUMMARY_SHA256,
            "edge_joint_forward_max_abs": float(
                functional_auth["edge_joint_forward_max_abs"]
            ),
            "source_fingerprint_max_abs": float(
                functional_auth["source_fingerprint_max_abs"]
            ),
            "full_target_replay_max_abs": float(
                functional_auth["full_target_replay_max_abs"]
            ),
            "streaming_semantic_max_abs": float(
                functional_auth["streaming_semantic_max_abs"]
            ),
            "pass": bool(functional_auth["pass"]),
        },
        "symmetric_pair_count": len(pair_rows),
    }


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
        require(
            worker["execution_head"] == args.expected_head,
            "MIXED_WORKER_HEAD",
        )

    rows = _merge_worker_accumulators(workers)
    require(len(rows)==36,f"MERGED_ORIENTATION_COUNT:{len(rows)}")
    identities={(row["group"],row["source"],row["target"]) for row in rows}
    require(len(identities)==36,"MERGED_ORIENTATION_DUPLICATE")
    expected={(g,cell_name(src),cell_name(tgt)) for g,src,tgt in all_orientations()}
    require(identities==expected,"MERGED_ORIENTATION_COVERAGE")
    require(sum(int(w["valid_token_count"]) for w in workers)==VALID_TOKEN_COUNT,"MERGED_VALID_TOKEN_COUNT")
    replay_cells={cell_name(cell):max(float(w["cell_source_replay_max_abs"][cell_name(cell)]) for w in workers) for cell in FULL_FACTORIAL_CELLS}
    require(max(replay_cells.values())<=SOURCE_TARGET_REPLAY_ATOL,"CELL_REPLAY_AUTH_FAIL")

    authentication = _authentication_from_rows(rows)
    require(authentication["pass"], "FROZEN_AUTHENTICATION_FAIL")

    symmetric = _symmetric_pairs(rows)
    require(
        sum(1 for row in symmetric if row["group"] == "PRIMARY_A") == 9,
        "PRIMARY_UNORDERED_PAIR_COUNT",
    )
    require(
        sum(1 for row in symmetric if row["group"] == "CONTROL_R") == 9,
        "CONTROL_UNORDERED_PAIR_COUNT",
    )
    matched = _source_matched_controls(rows)

    pair_stage_rows = []
    for row in rows:
        for stage in STAGE_ORDER:
            pair_stage_rows.append(
                {
                    "group": row["group"],
                    "source": row["source"],
                    "target": row["target"],
                    "stage": stage,
                    **row["stage"][stage],
                }
            )
    require(len(pair_stage_rows) == 180, "PAIR_STAGE_ROW_COUNT")

    grouped = {}
    for group in ("PRIMARY_A", "CONTROL_R"):
        subset = [row for row in symmetric if row["group"] == group]
        grouped[group] = {
            stage: _group_summary(subset, stage)
            for stage in STAGE_ORDER
        }

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS_VALIDATED_TRANSPORT_ARTIFACT",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "contract_commit": CONTRACT_COMMIT,
        "population": "FROZEN_PHASE3A_P0_DEV",
        "dev_rows": DEV_ROWS,
        "valid_token_count": VALID_TOKEN_COUNT,
        "sharding_mode": "DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION",
        "worker_row_ranges": {str(i): list(worker_row_range(i)) for i in (0, 1)},
        "primary_orientations": 18,
        "control_orientations": 18,
        "stage_order": list(STAGE_ORDER),
        "grouped_stagewise_selective_retention": grouped,
        "symmetric_unordered_pairs": symmetric,
        "source_cell_matched_primary_minus_control": matched,
        "authentication": authentication,
        "cell_source_replay_max_abs": replay_cells,
        "mechanism_classification_thresholds": None,
        "scientific_p_value_count": 0,
        "uncertainty_method": "DESCRIPTIVE_PAIR_DISTRIBUTION_ONLY_NO_CI_NO_PVALUE",
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }

    shard_manifest = {
        "schema_version": "GEN5_NATIVE_RECONVERGENCE_SHARD_MANIFEST_V1",
        "execution_head": args.expected_head,
        "workers": {
            str(worker["worker_id"]): {
                "sharding_mode": worker["sharding_mode"],
                "row_start": worker["row_start"],
                "row_stop": worker["row_stop"],
                "row_count": worker["row_count"],
                "source_cells": worker["source_cells"],
                "orientation_count": worker["orientation_count"],
                "valid_token_count": worker["valid_token_count"],
                "worker_result_sha256": sha256_file(
                    _worker_payload_path(scratch_root, int(worker["worker_id"]))
                ),
            }
            for worker in workers
        },
        "total_orientations": 36,
    }

    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = output_root / "transport_summary.json"
    rows_path = output_root / "pair_stage_transport_metrics.jsonl"
    shard_path = output_root / "shard_manifest.json"
    provenance_path = output_root / "run_provenance.json"

    summary_path.write_bytes(canonical_json_bytes(summary))
    rows_path.write_bytes(
        b"".join(canonical_json_bytes(row) for row in pair_stage_rows)
    )
    shard_path.write_bytes(canonical_json_bytes(shard_manifest))

    provenance = {
        "schema_version": PROVENANCE_SCHEMA,
        "status": "PASS",
        "execution_head": args.expected_head,
        "implementation_freeze_commit": args.implementation_freeze_commit,
        "design_commit": DESIGN_COMMIT,
        "contract_commit": CONTRACT_COMMIT,
        "summary_sha256": sha256_file(summary_path),
        "pair_stage_metrics_sha256": sha256_file(rows_path),
        "shard_manifest_sha256": sha256_file(shard_path),
        "worker_result_sha256": {
            str(worker_id): sha256_file(_worker_payload_path(scratch_root, worker_id))
            for worker_id in (0, 1)
        },
        "projector": {
            "backend": PROJECTOR_BACKEND,
            "rtol": PROJECTOR_PINV_RTOL,
            "atol": PROJECTOR_PINV_ATOL,
            "matrix": "2x2_gram",
            "hermitian": True,
        },
        "energy_epsilon": ENERGY_EPSILON,
        "sharding_mode": "DISJOINT_EXAMPLE_RANGES_SUFFICIENT_STAT_REDUCTION",
        "worker_row_ranges": {str(i): list(worker_row_range(i)) for i in (0, 1)},
        "dev_order_sha256": DEV_ORDER_SHA256,
        "dev_encoding_sha256": DEV_ENCODING_SHA256,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "checkpoint_sha256": {
            cell_name(cell): CHECKPOINT_SHA256[cell]
            for cell in FULL_FACTORIAL_CELLS
        },
        "training_executed": False,
        "optimizer_constructed": False,
        "backward_method_called": False,
        "parameter_gradients_accumulated": False,
        "checkpoint_mutation": False,
        "confirmatory_9601_9900_loaded": False,
        "vitaminc_loaded": False,
    }
    provenance_path.write_bytes(canonical_json_bytes(provenance))

    print("GEN5_NATIVE_RECONVERGENCE_MERGE_PASS")
    print(f"ORIENTATIONS={len(rows)}")
    print(f"PAIR_STAGE_ROWS={len(pair_stage_rows)}")
    print(f"SUMMARY={summary_path}")
    print(f"METRICS={rows_path}")
    print(f"SHARD_MANIFEST={shard_path}")
    print(f"PROVENANCE={provenance_path}")


def _spawn_worker(args: argparse.Namespace, worker_id: int) -> subprocess.Popen:
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
    legacy._validate_two_t4s()
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


def validate_args(args: argparse.Namespace) -> None:
    if args.static_contract_check:
        require(args.worker_id is None, "STATIC_WORKER_ID_FORBIDDEN")
        require(args.scratch_root is None, "STATIC_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "STATIC_OUTPUT_FORBIDDEN")
        return

    require(args.expected_head is not None, "EXPECTED_HEAD_REQUIRED")
    require(args.implementation_freeze_commit is not None, "IMPLEMENTATION_FREEZE_REQUIRED")
    require(args.model_snapshot is not None, "MODEL_SNAPSHOT_REQUIRED")
    require(args.tokenizer_snapshot is not None, "TOKENIZER_SNAPSHOT_REQUIRED")
    require(args.checkpoint is not None, "PARENT_CHECKPOINT_REQUIRED")

    if args.preflight:
        require(args.worker_id is None, "PREFLIGHT_WORKER_FORBIDDEN")
        require(args.scratch_root is None, "PREFLIGHT_SCRATCH_FORBIDDEN")
        require(args.output_root is None, "PREFLIGHT_OUTPUT_FORBIDDEN")
    elif args.run_worker:
        require(args.worker_id in (0, 1), "WORKER_ID_REQUIRED")
        require(args.scratch_root is not None, "WORKER_SCRATCH_REQUIRED")
        require(args.output_root is None, "WORKER_OUTPUT_FORBIDDEN")
    elif args.merge_only:
        require(args.worker_id is None, "MERGE_WORKER_FORBIDDEN")
        require(args.scratch_root is not None, "MERGE_SCRATCH_REQUIRED")
        require(args.output_root is not None, "MERGE_OUTPUT_REQUIRED")
    elif args.run:
        require(args.worker_id is None, "RUN_WORKER_ID_FORBIDDEN")
        require(args.scratch_root is not None, "RUN_SCRATCH_REQUIRED")
        require(args.output_root is not None, "RUN_OUTPUT_REQUIRED")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

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
