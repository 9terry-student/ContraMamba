#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_finite_epsilon_five_plane_decomposition_fast_cuda
    as reference
)

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "7254f89c352de5e6c6594e43d01a6a49ea5aeeb4"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_small_epsilon_robustness_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "b0e67c1d0aae2df08ae22d44d411b455db1c3cbd"

REFERENCE_RUN_NAME = (
    "g4k-finite-epsilon-five-plane-decomposition-xg1-2401-2700-"
    "e27dbbd-retry1"
)
REFERENCE_ROOT = Path(
    "reports/reason_router_gen4_finite_epsilon_five_plane_decomposition_runs"
) / REFERENCE_RUN_NAME

REFERENCE_ITEMS = (
    REFERENCE_ROOT / "finite_epsilon_principal_decomposition_items.jsonl"
)
REFERENCE_SUMMARY = (
    REFERENCE_ROOT / "finite_epsilon_principal_decomposition_summary.json"
)
REFERENCE_MANIFEST = REFERENCE_ROOT / "artifact_manifest.json"
REFERENCE_SUMS = REFERENCE_ROOT / "SHA256SUMS.txt"

REFERENCE_ITEMS_GIT_BLOB = "edd46066805082734f624a54736df8a6b82544e5"
REFERENCE_SUMMARY_GIT_BLOB = "996cc03c15d7cc8a04109325e35d6b0512e00ea2"
REFERENCE_MANIFEST_GIT_BLOB = "4967aca0c5634d136867e500548c757489053b6b"
REFERENCE_SUMS_GIT_BLOB = "9b19f40ac9843ccc74a72118e3ce4683bb79bea9"

REFERENCE_ITEMS_SHA256 = (
    "8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17"
)
REFERENCE_SUMMARY_SHA256 = (
    "7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e"
)
REFERENCE_MANIFEST_SHA256 = (
    "bcf6facd7d7a506b2dcbd1da4ce6eca67749eaa8e0e014bfb0ffac0467fa1d10"
)

REFERENCE_EPSILON = 0.025
NEW_EPSILONS = (0.0125, 0.00625)

N = reference.N
DIM = reference.DIM
K = reference.K
TOL = reference.TOL

PLANE_ORDER = reference.PLANE_ORDER
DIRECTION_ORDER = reference.DIRECTION_ORDER
EXPECTED_EIGENVALUES = reference.EXPECTED_EIGENVALUES

F_SIGNED = 2
F_DIRECTION = 4
F_PAIR_PER_EPSILON = 40
F_PAIR = F_PAIR_PER_EPSILON * len(NEW_EPSILONS)
F_TOTAL_PER_EPSILON = N * F_PAIR_PER_EPSILON
F_TOTAL = N * F_PAIR
GPU_COUNT = 2

SHARDS = (
    {
        "shard_id": 0,
        "gpu_id": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_2401",
        "pair_last": "xg1_fact_2550",
        "pair_count": 150,
        "forward_budget": 12000,
    },
    {
        "shard_id": 1,
        "gpu_id": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_2551",
        "pair_last": "xg1_fact_2700",
        "pair_count": 150,
        "forward_budget": 12000,
    },
)

ITEM_FILE = "small_epsilon_robustness_items.jsonl"
SUMMARY_FILE = "small_epsilon_robustness_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-small-epsilon-robustness-item-v1"
EPSILON_SCHEMA = "gen4-small-epsilon-robustness-epsilon-observation-v1"
SUMMARY_SCHEMA = "gen4-small-epsilon-robustness-summary-v1"
MANIFEST_SCHEMA = "gen4-small-epsilon-robustness-manifest-v1"
RESULT_PASS = "PASS_SMALL_EPSILON_ROBUSTNESS_RAW_OBSERVATION"


class SmallEpsilonRobustnessError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SmallEpsilonRobustnessError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmallEpsilonRobustnessError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_file(path: Path) -> str:
    return reference.base.sha256_file(path)


def expected_pairs() -> tuple[str, ...]:
    return reference.expected_pairs()


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(head == expected_head, f"HEAD:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "DESIGN_NOT_ANCESTOR")

    frozen_blobs = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        REFERENCE_ITEMS.as_posix(): REFERENCE_ITEMS_GIT_BLOB,
        REFERENCE_SUMMARY.as_posix(): REFERENCE_SUMMARY_GIT_BLOB,
        REFERENCE_MANIFEST.as_posix(): REFERENCE_MANIFEST_GIT_BLOB,
        REFERENCE_SUMS.as_posix(): REFERENCE_SUMS_GIT_BLOB,
        (
            "scripts/"
            "reason_router_gen4_finite_epsilon_five_plane_decomposition_fast_cuda.py"
        ): "7e1aa50a7c12a5c593a2abddda64fd52f1173adb",
    }
    for path, blob in frozen_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}") == blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(N == 300, "N")
    require(DIM == 395, "DIM")
    require(K == 5, "K")
    require(REFERENCE_EPSILON == 0.025, "REFERENCE_EPSILON")
    require(NEW_EPSILONS == (0.0125, 0.00625), "NEW_EPSILONS")
    require(len(set(NEW_EPSILONS)) == 2, "EPSILON_UNIQUENESS")
    require(
        all(0.0 < epsilon < REFERENCE_EPSILON for epsilon in NEW_EPSILONS),
        "EPSILON_RANGE",
    )
    require(
        math.isclose(
            NEW_EPSILONS[0],
            REFERENCE_EPSILON / 2.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "EPSILON_HALF_1",
    )
    require(
        math.isclose(
            NEW_EPSILONS[1],
            REFERENCE_EPSILON / 4.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "EPSILON_HALF_2",
    )
    require(F_SIGNED == 2, "F_SIGNED")
    require(F_DIRECTION == 4, "F_DIRECTION")
    require(F_PAIR_PER_EPSILON == 40, "F_PAIR_PER_EPSILON")
    require(F_PAIR == 80, "F_PAIR")
    require(F_TOTAL_PER_EPSILON == 12000, "F_TOTAL_PER_EPSILON")
    require(F_TOTAL == 24000, "F_TOTAL")
    require(GPU_COUNT == 2, "GPU_COUNT")


def validate_shards() -> None:
    validate_protocol()
    require(len(SHARDS) == GPU_COUNT, "SHARD_COUNT")
    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(int(shard["shard_id"]) == expected_id, "SHARD_ID")
        require(int(shard["gpu_id"]) == expected_id, "SHARD_GPU")
        start = int(shard["start_index"])
        end = int(shard["end_index"])
        count = int(shard["pair_count"])
        require(end - start == count, "SHARD_PAIR_COUNT")
        require(
            count * F_PAIR == int(shard["forward_budget"]),
            "SHARD_FORWARD_BUDGET",
        )
        pair_slice = expected_pairs()[start:end]
        require(
            len(pair_slice) == count
            and pair_slice[0] == shard["pair_first"]
            and pair_slice[-1] == shard["pair_last"],
            "SHARD_PAIR_RANGE",
        )
        covered.extend(range(start, end))
    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        sum(int(shard["forward_budget"]) for shard in SHARDS) == F_TOTAL,
        "TOTAL_FORWARD_BUDGET",
    )


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmallEpsilonRobustnessError(
            f"GIT_BLOB_FAILURE:{path}"
        ) from exc


def validate_reference_artifact() -> dict[str, Any]:
    canonical = {
        REFERENCE_ITEMS.name: git_blob_bytes(REFERENCE_ITEMS),
        REFERENCE_SUMMARY.name: git_blob_bytes(REFERENCE_SUMMARY),
        REFERENCE_MANIFEST.name: git_blob_bytes(REFERENCE_MANIFEST),
        REFERENCE_SUMS.name: git_blob_bytes(REFERENCE_SUMS),
    }

    require(
        reference.base.sha256_bytes(canonical[REFERENCE_ITEMS.name])
        == REFERENCE_ITEMS_SHA256,
        "REFERENCE_ITEMS_SHA256",
    )
    require(
        reference.base.sha256_bytes(canonical[REFERENCE_SUMMARY.name])
        == REFERENCE_SUMMARY_SHA256,
        "REFERENCE_SUMMARY_SHA256",
    )
    require(
        reference.base.sha256_bytes(canonical[REFERENCE_MANIFEST.name])
        == REFERENCE_MANIFEST_SHA256,
        "REFERENCE_MANIFEST_SHA256",
    )

    manifest = json.loads(
        canonical[REFERENCE_MANIFEST.name].decode("utf-8")
    )
    require(
        manifest["schema_version"] == reference.MANIFEST_SCHEMA,
        "REFERENCE_MANIFEST_SCHEMA",
    )
    require(
        manifest["files"]
        == {
            reference.ITEM_FILE: {
                "sha256": REFERENCE_ITEMS_SHA256,
                "bytes": len(canonical[REFERENCE_ITEMS.name]),
            },
            reference.SUMMARY_FILE: {
                "sha256": REFERENCE_SUMMARY_SHA256,
                "bytes": len(canonical[REFERENCE_SUMMARY.name]),
            },
        },
        "REFERENCE_MANIFEST_CONTENT",
    )

    observed_sums: dict[str, str] = {}
    for line in canonical[REFERENCE_SUMS.name].decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed_sums, "REFERENCE_SUMS_DUPLICATE")
        observed_sums[name] = digest

    require(
        observed_sums
        == {
            reference.MANIFEST_FILE: REFERENCE_MANIFEST_SHA256,
            reference.ITEM_FILE: REFERENCE_ITEMS_SHA256,
            reference.SUMMARY_FILE: REFERENCE_SUMMARY_SHA256,
        },
        "REFERENCE_SUMS_CONTENT",
    )

    items = [
        json.loads(line)
        for line in canonical[REFERENCE_ITEMS.name]
        .decode("utf-8")
        .splitlines()
        if line.strip()
    ]
    require(len(items) == N, "REFERENCE_ITEM_COUNT")

    for index, (pair, item) in enumerate(
        zip(expected_pairs(), items, strict=True)
    ):
        require(
            item["prior_raw_run_name"] == reference.PRIOR_RUN_NAME,
            f"REFERENCE_PRIOR_RUN:{index}",
        )
        require(
            item["prior_items_sha256"] == reference.PRIOR_ITEMS_SHA256,
            f"REFERENCE_PRIOR_ITEMS_SHA:{index}",
        )
        reference.validate_item(
            item,
            pair,
            index,
            float(item["prior_native_q0"]),
        )

    summary = json.loads(
        canonical[REFERENCE_SUMMARY.name].decode("utf-8")
    )
    require(
        summary["schema_version"] == reference.SUMMARY_SCHEMA
        and summary["result"] == reference.RESULT_PASS,
        "REFERENCE_SUMMARY_RESULT",
    )
    require(
        summary["execution_head"]
        == "e27dbbd45635da4beb03691f5a266a7d72824424",
        "REFERENCE_EXECUTION_HEAD",
    )
    require(summary["source_pair_count"] == N, "REFERENCE_SUMMARY_N")
    require(summary["epsilon"] == REFERENCE_EPSILON, "REFERENCE_EPSILON_VALUE")
    require(
        summary["plane_order"] == list(PLANE_ORDER)
        and tuple(summary["principal_direction_order"]) == DIRECTION_ORDER,
        "REFERENCE_GEOMETRY",
    )
    require(
        summary["prior_native_q0_reused"] is True
        and summary["prior_raw_run_name"] == reference.PRIOR_RUN_NAME
        and summary["prior_items_sha256"] == reference.PRIOR_ITEMS_SHA256,
        "REFERENCE_PRIOR_PROVENANCE",
    )
    require(
        summary["scientific_model_forward_count_this_run"] == 12000,
        "REFERENCE_FORWARD_COUNT",
    )
    require(
        summary["new_original_basis_scientific_model_forward_count"] == 0
        and summary["baseline_model_forward_count_this_run"] == 0,
        "REFERENCE_NO_EXTRA_FORWARD",
    )
    require(summary["primary_inference_executed"] is False, "REFERENCE_INFERENCE")
    require(
        summary["multiplicity_correction_executed"] is False,
        "REFERENCE_MULTIPLICITY",
    )
    require(summary["scientific_conclusion"] is None, "REFERENCE_CONCLUSION")

    return {
        "items": items,
        "summary": summary,
        "manifest": manifest,
        "canonical_git_blob_validation": True,
    }


def central_difference(
    f_plus: float,
    f_minus: float,
    epsilon: float,
) -> float:
    require(
        math.isfinite(float(f_plus))
        and math.isfinite(float(f_minus))
        and math.isfinite(float(epsilon))
        and float(epsilon) > 0.0,
        "CENTRAL_DIFFERENCE_INPUT",
    )
    value = (float(f_plus) - float(f_minus)) / (2.0 * float(epsilon))
    require(math.isfinite(value), "CENTRAL_DIFFERENCE_NONFINITE")
    return value


def apply_probe_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
    epsilon: float,
    audit: dict[str, Any],
) -> torch.Tensor:
    runtime = reference.base.holdout.phase1.base.prevalence_eq
    core = runtime.core

    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_SHAPE",
    )
    require(orientation in (-1, 1) and branch_sign in (-1, 1), "SIGN")
    require(float(epsilon) in NEW_EPSILONS, f"EPSILON:{epsilon}")

    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == core.INTERMEDIATE_SIZE
        and int(mask.sum().item()) == DIM,
        "MASK",
    )

    before = output.detach().clone()
    mask_device = mask.to(before.device)
    h = (
        before[0, token_index, :core.INTERMEDIATE_SIZE][mask_device]
        .detach()
        .cpu()
        .to(torch.float64)
        .contiguous()
    )

    condition_info = reference.base.condition_correction(
        h,
        condition,
        planes,
    )

    vector_value = (
        direction.detach().cpu().to(torch.float64).contiguous()
    )
    require(tuple(vector_value.shape) == (DIM,), "DIRECTION_SHAPE")
    require(
        abs(float(torch.linalg.vector_norm(vector_value).item()) - 1.0)
        <= TOL,
        "DIRECTION_NORM",
    )

    probe = (
        vector_value
        * float(branch_sign)
        * float(orientation)
        * float(epsilon)
    ).contiguous()
    total = (condition_info["d"] + probe).contiguous()

    out = output.clone()
    intended = total.to(device=out.device, dtype=out.dtype)
    out[0, token_index, :core.INTERMEDIATE_SIZE][mask_device] += intended

    require(
        torch.equal(
            out[:, :, core.INTERMEDIATE_SIZE:],
            before[:, :, core.INTERMEDIATE_SIZE:],
        ),
        "GATE_CHANGED",
    )
    require(
        torch.equal(
            out[:, :, :core.INTERMEDIATE_SIZE][:, :, ~mask_device],
            before[:, :, :core.INTERMEDIATE_SIZE][:, :, ~mask_device],
        ),
        "NONSTRONG_CHANGED",
    )
    if token_index:
        require(
            torch.equal(out[:, :token_index, :], before[:, :token_index, :]),
            "EARLIER_CHANGED",
        )
    if token_index + 1 < out.shape[1]:
        require(
            torch.equal(
                out[:, token_index + 1:, :],
                before[:, token_index + 1:, :],
            ),
            "LATER_CHANGED",
        )

    applied = (
        out[0, token_index, :core.INTERMEDIATE_SIZE][mask_device]
        - before[0, token_index, :core.INTERMEDIATE_SIZE][mask_device]
    ).detach().cpu().to(torch.float64)

    residual = float(
        torch.max(
            torch.abs(
                applied - intended.detach().cpu().to(torch.float64)
            )
        ).item()
    )
    require(
        residual <= runtime.transport_runtime.RUNTIME_CAST_TOL,
        f"APPLIED_RESIDUAL:{residual}",
    )

    audit.clear()
    audit.update({
        "condition": condition,
        "token_index": int(token_index),
        "orientation": int(orientation),
        "branch_sign": int(branch_sign),
        "epsilon": float(epsilon),
        "coefficient_source":
            "branch_local_native_residual_coordinates",
        "native_residual_coefficients":
            condition_info["native_residual_coefficients"],
        "native_component_l2":
            condition_info["native_component_l2"],
        "quarter_turn_component_l2":
            condition_info["quarter_turn_component_l2"],
        "restoration_replacement_addition_l2_mismatch":
            condition_info[
                "restoration_replacement_addition_l2_mismatch"
            ],
        "native_quarter_turn_dot":
            condition_info["native_quarter_turn_dot"],
        "condition_correction_l2":
            condition_info["condition_correction_l2"],
        "pp3_native_coefficients":
            condition_info["pp3_native_coefficients"],
        "pp3_post_coefficients":
            condition_info["pp3_post_coefficients"],
        "pp3_coefficient_drift_max_abs":
            condition_info["pp3_coefficient_drift_max_abs"],
        "residual_post_condition_projections":
            condition_info["residual_post_condition_projections"],
        "residual_neutralization_max_abs_projection":
            condition_info["residual_neutralization_max_abs_projection"],
        "replacement_coordinate_max_abs_residual":
            condition_info["replacement_coordinate_max_abs_residual"],
        "probe_correction_l2":
            float(torch.linalg.vector_norm(probe).item()),
        "applied_correction_max_abs_residual": residual,
    })
    return out


def install_probe_hook(mixer17: Any, **kwargs: Any):
    def hook(_module, _args, output):
        return apply_probe_hook(output, **kwargs)
    return mixer17.in_proj.register_forward_hook(hook)


def run_signed_epsilon(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    orientation: int,
    epsilon: float,
    planes: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[Any, Any],
    events: Mapping[Any, Any],
    budget: Any,
) -> dict[str, Any]:
    runtime = reference.base.holdout.phase1.base.prevalence_eq
    parent = runtime.parent
    core = runtime.core

    pair = str(seed["source_pair_id"])
    cells = reference.base.holdout.phase1._cells()
    anchors = reference.base.holdout.phase1._anchors_for_pair(pair, events)

    captured: dict[str, Any] = {}
    audits: dict[str, Any] = {}

    for role, branch_sign in (("tp", 1), ("tm", -1)):
        audit: dict[str, Any] = {}
        target = anchors[role] + core.TARGET_OFFSET
        handle = install_probe_hook(
            runtime_ctx["mixer17"],
            token_index=target,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=planes,
            direction=direction,
            orientation=orientation,
            branch_sign=branch_sign,
            epsilon=epsilon,
            audit=audit,
        )
        try:
            captured[role] = parent.capture_branch(
                model,
                runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                input_ids=reference.base.input_row(
                    encoded,
                    row_index,
                    pair,
                    cells[role],
                ),
                anchor=anchors[role],
                budget=budget,
                capture_states=True,
            )
        finally:
            handle.remove()

        require(
            bool(audit)
            and captured[role]["intervention_audit"] is None,
            "HOOK_AUDIT",
        )
        audits[role] = dict(audit)

    plus_efficiency = float(parent.path_efficiency(captured["tp"]))
    minus_efficiency = float(parent.path_efficiency(captured["tm"]))

    return {
        "condition": condition,
        "orientation": int(orientation),
        "epsilon": float(epsilon),
        "F": plus_efficiency - minus_efficiency,
        "plus_path_efficiency": plus_efficiency,
        "minus_path_efficiency": minus_efficiency,
        "branch_audits": audits,
        "model_forward_count": F_SIGNED,
    }


def run_principal_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    direction_key: str,
    epsilon: float,
    planes: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    positive = run_signed_epsilon(
        seed,
        direction,
        condition="native",
        orientation=1,
        epsilon=epsilon,
        planes=planes,
        **kwargs,
    )
    negative = run_signed_epsilon(
        seed,
        direction,
        condition="native",
        orientation=-1,
        epsilon=epsilon,
        planes=planes,
        **kwargs,
    )

    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    numerator = f_plus - f_minus
    j_value = central_difference(f_plus, f_minus, epsilon)

    return {
        "direction_key": direction_key,
        "epsilon": float(epsilon),
        "F_plus": f_plus,
        "F_minus": f_minus,
        "central_difference_numerator": numerator,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": F_DIRECTION,
    }


def run_epsilon_observation(
    seed: Mapping[str, Any],
    *,
    epsilon: float,
    planes: Mapping[str, torch.Tensor],
    eigenvalues: Sequence[float],
    q0: float,
    **kwargs: Any,
) -> dict[str, Any]:
    require(float(epsilon) in NEW_EPSILONS, f"RUN_EPSILON:{epsilon}")
    direction_map = reference.principal_direction_map(planes)

    probes = [
        run_principal_direction(
            seed,
            direction_map[key],
            direction_key=key,
            epsilon=epsilon,
            planes=planes,
            **kwargs,
        )
        for key in DIRECTION_ORDER
    ]
    require(
        tuple(probe["direction_key"] for probe in probes) == DIRECTION_ORDER,
        "PROBE_ORDER",
    )

    j_by_direction = {
        probe["direction_key"]: float(probe["J"])
        for probe in probes
    }
    reconstruction = reference.decomposition_from_j(
        j_by_direction,
        eigenvalues,
        q0,
    )

    return {
        "schema_version": EPSILON_SCHEMA,
        "epsilon": float(epsilon),
        "principal_direction_order": list(DIRECTION_ORDER),
        "principal_direction_probes": probes,
        **reconstruction,
        "scientific_model_forward_count_this_epsilon":
            F_PAIR_PER_EPSILON,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "scientific_conclusion": None,
    }


def run_pair(
    seed: Mapping[str, Any],
    *,
    planes: Mapping[str, torch.Tensor],
    eigenvalues: Sequence[float],
    q0: float,
    **kwargs: Any,
) -> dict[str, Any]:
    observations = [
        run_epsilon_observation(
            seed,
            epsilon=epsilon,
            planes=planes,
            eigenvalues=eigenvalues,
            q0=q0,
            **kwargs,
        )
        for epsilon in NEW_EPSILONS
    ]

    return {
        **dict(seed),
        "schema_version": ITEM_SCHEMA,
        "reference_epsilon": REFERENCE_EPSILON,
        "reference_run_name": REFERENCE_RUN_NAME,
        "reference_items_sha256": REFERENCE_ITEMS_SHA256,
        "reference_scientific_model_forward_count_this_run": 0,
        "new_epsilons": list(NEW_EPSILONS),
        "plane_order": list(PLANE_ORDER),
        "principal_direction_order": list(DIRECTION_ORDER),
        "positive_contrast_eigenvalues":
            [float(x) for x in eigenvalues],
        "prior_native_q0": float(q0),
        "prior_native_q0_reused": True,
        "epsilon_observations": observations,
        "new_original_basis_scientific_model_forward_count": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
        "baseline_model_forward_count_this_run": 0,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "multiplicity_correction_executed": False,
        "scientific_conclusion": None,
    }


def validate_probe(
    probe: Mapping[str, Any],
    *,
    epsilon: float,
) -> None:
    require(float(probe["epsilon"]) == float(epsilon), "PROBE_EPSILON")
    require(int(probe["model_forward_count"]) == F_DIRECTION, "PROBE_BUDGET")
    f_plus = float(probe["F_plus"])
    f_minus = float(probe["F_minus"])
    numerator = f_plus - f_minus
    require(
        float(probe["central_difference_numerator"]) == numerator,
        "PROBE_NUMERATOR",
    )
    require(
        float(probe["J"]) == central_difference(f_plus, f_minus, epsilon),
        "PROBE_J",
    )
    require(
        float(probe["J_squared"])
        == float(probe["J"]) * float(probe["J"]),
        "PROBE_J2",
    )
    for sign_key in ("positive_probe", "negative_probe"):
        signed = probe[sign_key]
        require(float(signed["epsilon"]) == float(epsilon), "SIGNED_EPSILON")
        require(int(signed["model_forward_count"]) == F_SIGNED, "SIGNED_BUDGET")
        for audit in signed["branch_audits"].values():
            require(float(audit["epsilon"]) == float(epsilon), "AUDIT_EPSILON")
            require(
                math.isclose(
                    float(audit["probe_correction_l2"]),
                    float(epsilon),
                    rel_tol=0.0,
                    abs_tol=2.0e-12,
                ),
                "AUDIT_PROBE_NORM",
            )


def validate_epsilon_observation(
    observation: Mapping[str, Any],
    *,
    epsilon: float,
    eigenvalues: Sequence[float],
    q0: float,
) -> None:
    require(observation["schema_version"] == EPSILON_SCHEMA, "EPS_SCHEMA")
    require(float(observation["epsilon"]) == float(epsilon), "EPS_VALUE")
    require(
        tuple(observation["principal_direction_order"]) == DIRECTION_ORDER,
        "EPS_DIRECTION_ORDER",
    )
    probes = observation["principal_direction_probes"]
    require(len(probes) == 2 * K, "EPS_PROBE_COUNT")
    require(
        tuple(probe["direction_key"] for probe in probes) == DIRECTION_ORDER,
        "EPS_PROBE_ORDER",
    )
    for probe in probes:
        validate_probe(probe, epsilon=epsilon)

    j_by_direction = {
        probe["direction_key"]: float(probe["J"])
        for probe in probes
    }
    expected = reference.decomposition_from_j(
        j_by_direction,
        eigenvalues,
        q0,
    )
    require(
        observation["plane_contributions"] == expected["plane_contributions"],
        "EPS_CONTRIBUTIONS",
    )
    for key in (
        "Q_principal",
        "reconstruction_residual",
        "absolute_reconstruction_residual",
        "relative_reconstruction_residual_to_Q0",
        "absolute_relative_reconstruction_residual_to_Q0",
    ):
        require(observation[key] == expected[key], f"EPS_RECON:{key}")

    require(
        int(observation["scientific_model_forward_count_this_epsilon"])
        == F_PAIR_PER_EPSILON,
        "EPS_FORWARD_BUDGET",
    )
    require(observation["primary_inference_executed"] is False, "EPS_INFERENCE")
    require(int(observation["p_value_count_added"]) == 0, "EPS_PVALUE")
    require(observation["scientific_conclusion"] is None, "EPS_CONCLUSION")


def validate_item(
    item: Mapping[str, Any],
    expected_pair: str,
    index: int,
    expected_q0: float,
) -> None:
    require(
        item["schema_version"] == ITEM_SCHEMA
        and item["source_pair_id"] == expected_pair
        and int(item["pair_index"]) == index
        and item["family_key"] == "xg1",
        f"ITEM_ID:{index}",
    )
    require(
        float(item["reference_epsilon"]) == REFERENCE_EPSILON,
        f"ITEM_REF_EPS:{index}",
    )
    require(
        item["reference_run_name"] == REFERENCE_RUN_NAME
        and item["reference_items_sha256"] == REFERENCE_ITEMS_SHA256,
        f"ITEM_REF:{index}",
    )
    require(
        int(item["reference_scientific_model_forward_count_this_run"]) == 0,
        f"ITEM_REF_FORWARD:{index}",
    )
    require(
        tuple(float(x) for x in item["new_epsilons"]) == NEW_EPSILONS,
        f"ITEM_EPSILONS:{index}",
    )
    require(
        item["plane_order"] == list(PLANE_ORDER)
        and tuple(item["principal_direction_order"]) == DIRECTION_ORDER,
        f"ITEM_GEOMETRY:{index}",
    )
    require(float(item["prior_native_q0"]) == float(expected_q0), "ITEM_Q0")
    require(item["prior_native_q0_reused"] is True, "ITEM_Q0_REUSE")

    eigenvalues = [float(x) for x in item["positive_contrast_eigenvalues"]]
    require(len(eigenvalues) == K, "ITEM_EIGENVALUES")

    observations = item["epsilon_observations"]
    require(len(observations) == len(NEW_EPSILONS), "ITEM_OBS_COUNT")
    require(
        tuple(float(obs["epsilon"]) for obs in observations) == NEW_EPSILONS,
        "ITEM_OBS_ORDER",
    )
    for epsilon, observation in zip(
        NEW_EPSILONS,
        observations,
        strict=True,
    ):
        validate_epsilon_observation(
            observation,
            epsilon=epsilon,
            eigenvalues=eigenvalues,
            q0=float(expected_q0),
        )

    require(
        int(item["scientific_model_forward_count_this_run"]) == F_PAIR,
        "ITEM_FORWARD_BUDGET",
    )
    require(
        int(item["new_original_basis_scientific_model_forward_count"]) == 0,
        "ITEM_ORIGINAL_BASIS",
    )
    require(
        int(item["baseline_model_forward_count_this_run"]) == 0,
        "ITEM_BASELINE",
    )
    require(item["primary_inference_executed"] is False, "ITEM_INFERENCE")
    require(int(item["p_value_count_added"]) == 0, "ITEM_PVALUE")
    require(
        item["multiplicity_correction_executed"] is False,
        "ITEM_MULTIPLICITY",
    )
    require(item["scientific_conclusion"] is None, "ITEM_CONCLUSION")


def validate_items(
    items: Sequence[Mapping[str, Any]],
    prior_q0: Sequence[float],
) -> None:
    require(len(items) == len(prior_q0) == N, "ITEM_COUNT")
    for index, (pair, item, q0) in enumerate(
        zip(expected_pairs(), items, prior_q0, strict=True)
    ):
        validate_item(item, pair, index, float(q0))


def validate_shard_items(
    items: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
    prior_q0: Sequence[float],
) -> None:
    require(len(items) == int(shard["pair_count"]), "SHARD_ITEM_COUNT")
    start = int(shard["start_index"])
    for local_index, item in enumerate(items):
        global_index = start + local_index
        validate_item(
            item,
            expected_pairs()[global_index],
            global_index,
            float(prior_q0[global_index]),
        )
    require(
        sum(
            int(item["scientific_model_forward_count_this_run"])
            for item in items
        ) == int(shard["forward_budget"]),
        "SHARD_FORWARD_SUM",
    )


def write_shard_payload(
    temp_dir: Path,
    shard_id: int,
    payload: Mapping[str, Any],
) -> None:
    (temp_dir / f"shard_{shard_id}.json").write_bytes(
        reference.base.canonical(payload)
    )


def read_shard_payload(
    temp_dir: Path,
    shard_id: int,
) -> dict[str, Any]:
    path = temp_dir / f"shard_{shard_id}.json"
    require(path.is_file(), f"SHARD_MISSING:{shard_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"SHARD_OBJECT:{shard_id}")
    return value


def worker_run(
    *,
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    prior_q0: Sequence[float],
    temp_dir: Path,
) -> None:
    shard_id = int(shard["shard_id"])
    error_path = temp_dir / f"shard_{shard_id}.error.txt"

    try:
        gpu_id = int(shard["gpu_id"])
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        authenticate_repo(expected_head)
        validate_shards()
        reference.base.validate_static_inputs()
        planes, eigenvalues = reference.principal_geometry()

        runtime = reference.base.holdout.phase1.base.prevalence_eq
        reference.base.runtime_gate_for_device(runtime, gpu_id)

        with runtime.backend.parent_runtime_rebind():
            rows, encoded, event_rows = reference.base.load_inputs(
                tokenizer_snapshot
            )
            pairs = reference.base.pair_order(rows)
            parent = runtime.parent
            events = parent.event_lookup(event_rows)
            row_index = parent.build_row_index(rows)

            trace_code, trace_line = (
                runtime.measurement._resolve_and_validate_runtime_binding()
            )
            kernels = runtime.kernel_compat.load_exact_fast_kernels()

            with runtime.kernel_compat.exact_transformers_kernel_loader(
                kernels
            ) as calls:
                model, checkpoint_sha = (
                    parent.load_representative_model_external(
                        model_snapshot=model_snapshot,
                        checkpoint_path=checkpoint_path,
                    )
                )
                require(
                    checkpoint_sha
                    == runtime.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
                    "CHECKPOINT",
                )
                runtime_ctx = (
                    runtime.transport_runtime.validate_runtime_components(
                        model
                    )
                )

            counts = Counter(calls)
            require(
                set(counts) == {"causal-conv1d", "mamba-ssm"}
                and counts["causal-conv1d"] > 0
                and counts["causal-conv1d"] == counts["mamba-ssm"],
                "KERNEL_CONSTRUCTOR",
            )
            runtime.kernel_compat.validate_transformers_kernel_bindings(
                kernels
            )

            model.to(device)
            model.eval()

            fast_capture = reference.base.make_fast_capture_for_device(
                runtime,
                kernels,
                device,
            )
            original_capture = parent.capture_branch
            budget = parent.ForwardBudget(int(shard["forward_budget"]))
            items: list[dict[str, Any]] = []

            parent.capture_branch = fast_capture
            try:
                for global_index in range(
                    int(shard["start_index"]),
                    int(shard["end_index"]),
                ):
                    pair = pairs[global_index]
                    items.append(
                        run_pair(
                            reference.base.probe_seed(
                                global_index,
                                pair,
                                events,
                            ),
                            planes=planes,
                            eigenvalues=eigenvalues,
                            q0=float(prior_q0[global_index]),
                            model=model,
                            runtime_ctx=runtime_ctx,
                            trace_code=trace_code,
                            trace_line=trace_line,
                            encoded=encoded,
                            row_index=row_index,
                            events=events,
                            budget=budget,
                        )
                    )
                budget.assert_exact()
                torch.cuda.synchronize(device)
            finally:
                parent.capture_branch = original_capture

        validate_shard_items(items, shard, prior_q0)
        payload = {
            "shard_id": shard_id,
            "gpu_id": gpu_id,
            "device_name": torch.cuda.get_device_name(gpu_id),
            "pair_first": items[0]["source_pair_id"],
            "pair_last": items[-1]["source_pair_id"],
            "pair_count": len(items),
            "scientific_model_forward_count_this_run":
                int(shard["forward_budget"]),
            "checkpoint_sha256": checkpoint_sha,
            "items": items,
        }
        write_shard_payload(temp_dir, shard_id, payload)

    except BaseException:
        error_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def worker_entry(
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: str,
    tokenizer_snapshot: str,
    checkpoint_path: str,
    prior_q0: Sequence[float],
    temp_dir: str,
) -> None:
    worker_run(
        shard=shard,
        expected_head=expected_head,
        model_snapshot=Path(model_snapshot),
        tokenizer_snapshot=Path(tokenizer_snapshot),
        checkpoint_path=Path(checkpoint_path),
        prior_q0=prior_q0,
        temp_dir=Path(temp_dir),
    )


def merge_shards(
    payloads: Sequence[Mapping[str, Any]],
    prior_q0: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(payloads) == GPU_COUNT, "MERGE_COUNT")
    by_id = {
        int(payload["shard_id"]): payload
        for payload in payloads
    }
    require(set(by_id) == {0, 1}, "MERGE_IDS")

    merged: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    checkpoint_shas: set[str] = set()

    for shard in SHARDS:
        payload = by_id[int(shard["shard_id"])]
        items = payload["items"]
        validate_shard_items(items, shard, prior_q0)
        checkpoint_shas.add(str(payload["checkpoint_sha256"]))
        merged.extend(items)
        shard_meta.append({
            "shard_id": int(payload["shard_id"]),
            "gpu_id": int(payload["gpu_id"]),
            "device_name": str(payload["device_name"]),
            "pair_first": str(payload["pair_first"]),
            "pair_last": str(payload["pair_last"]),
            "pair_count": int(payload["pair_count"]),
            "scientific_model_forward_count_this_run":
                int(payload["scientific_model_forward_count_this_run"]),
            "checkpoint_sha256": str(payload["checkpoint_sha256"]),
        })

    require(len(checkpoint_shas) == 1, "CHECKPOINT_MISMATCH")
    validate_items(merged, prior_q0)
    return merged, shard_meta


def write_outputs(
    out: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    prior_q0: Sequence[float],
) -> None:
    require(not out.exists(), "OUTPUT_COLLISION")
    validate_items(items, prior_q0)
    out.mkdir(parents=True, exist_ok=False)

    payloads = {
        ITEM_FILE: reference.base.jsonl(items),
        SUMMARY_FILE: reference.base.canonical(summary),
    }
    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        (out / name).write_bytes(raw)
        hashes[name] = reference.base.sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((out / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    manifest_raw = reference.base.canonical(manifest)
    (out / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = reference.base.sha256_bytes(manifest_raw)

    (out / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def validate_artifact(out: Path) -> dict[str, Any]:
    reference_validated = validate_reference_artifact()
    prior_q0 = [
        float(item["prior_native_q0"])
        for item in reference_validated["items"]
    ]
    require(len(prior_q0) == N, "PRIOR_Q0_N")

    manifest = json.loads(
        (out / MANIFEST_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        manifest["schema_version"] == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )

    hashes: dict[str, str] = {}
    for name in (ITEM_FILE, SUMMARY_FILE):
        path = out / name
        require(path.is_file(), f"FILE_MISSING:{name}")
        digest = sha256_file(path)
        require(
            digest == manifest["files"][name]["sha256"]
            and path.stat().st_size == manifest["files"][name]["bytes"],
            f"FILE:{name}",
        )
        hashes[name] = digest

    hashes[MANIFEST_FILE] = sha256_file(out / MANIFEST_FILE)

    observed: dict[str, str] = {}
    for line in (out / CHECKSUM_FILE).read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in observed, f"CHECKSUM_DUPLICATE:{name}")
        observed[name] = digest

    require(
        observed == {
            name: digest
            for name, digest in sorted(hashes.items())
        },
        "CHECKSUMS",
    )

    items = reference.base.read_jsonl(out / ITEM_FILE)
    validate_items(items, prior_q0)

    summary = json.loads(
        (out / SUMMARY_FILE).read_text(encoding="utf-8-sig")
    )
    require(
        summary["schema_version"] == SUMMARY_SCHEMA
        and summary["result"] == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary["source_pair_count"] == N
        and summary["pair_id_first"] == "xg1_fact_2401"
        and summary["pair_id_last"] == "xg1_fact_2700",
        "SUMMARY_POPULATION",
    )
    require(
        float(summary["reference_epsilon"]) == REFERENCE_EPSILON
        and tuple(float(x) for x in summary["new_epsilons"]) == NEW_EPSILONS,
        "SUMMARY_EPSILONS",
    )
    require(
        summary["reference_run_name"] == REFERENCE_RUN_NAME
        and summary["reference_items_sha256"] == REFERENCE_ITEMS_SHA256
        and summary["reference_summary_sha256"] == REFERENCE_SUMMARY_SHA256,
        "SUMMARY_REFERENCE",
    )
    require(
        int(summary["reference_scientific_model_forward_count_this_run"]) == 0,
        "SUMMARY_REFERENCE_FORWARD",
    )
    require(
        int(summary["scientific_model_forward_count_per_new_epsilon"])
        == F_TOTAL_PER_EPSILON,
        "SUMMARY_EPS_FORWARD",
    )
    require(
        int(summary["scientific_model_forward_count_this_run"]) == F_TOTAL,
        "SUMMARY_TOTAL_FORWARD",
    )
    require(
        int(summary["new_original_basis_scientific_model_forward_count"]) == 0
        and int(summary["baseline_model_forward_count_this_run"]) == 0,
        "SUMMARY_NO_EXTRA_FORWARD",
    )
    require(
        int(summary["gpu_count"]) == GPU_COUNT
        and len(summary["shards"]) == GPU_COUNT,
        "SUMMARY_GPU",
    )
    require(summary["primary_inference_executed"] is False, "SUMMARY_INFERENCE")
    require(int(summary["p_value_count_added"]) == 0, "SUMMARY_PVALUE")
    require(
        summary["multiplicity_correction_executed"] is False,
        "SUMMARY_MULTIPLICITY",
    )
    require(summary["training_executed"] is False, "SUMMARY_TRAINING")
    require(summary["backward_executed"] is False, "SUMMARY_BACKWARD")
    require(summary["task_heads_executed"] is False, "SUMMARY_TASK_HEADS")
    require(summary["logits_read"] is False, "SUMMARY_LOGITS")
    require(summary["scientific_conclusion"] is None, "SUMMARY_CONCLUSION")

    return {
        "items": items,
        "summary": summary,
        "manifest": manifest,
    }


def run_observation(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_shards()
    reference.base.validate_static_inputs()
    _, eigenvalues = reference.principal_geometry()

    reference_validated = validate_reference_artifact()
    prior_q0 = [
        float(item["prior_native_q0"])
        for item in reference_validated["items"]
    ]
    require(len(prior_q0) == N, "PRIOR_Q0_COUNT")

    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gen4_small_epsilon_robustness_"
    ) as temp_name:
        temp_dir = Path(temp_name)
        processes = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_entry,
                args=(
                    dict(shard),
                    expected_head,
                    str(model_snapshot),
                    str(tokenizer_snapshot),
                    str(checkpoint_path),
                    list(prior_q0),
                    str(temp_dir),
                ),
                name=f"gen4-small-epsilon-gpu{shard['gpu_id']}",
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failures = []
        for shard, process in zip(SHARDS, processes, strict=True):
            if process.exitcode != 0:
                error_path = (
                    temp_dir / f"shard_{int(shard['shard_id'])}.error.txt"
                )
                detail = (
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else f"exitcode={process.exitcode}"
                )
                failures.append(
                    f"SHARD_{shard['shard_id']}_FAILED:\n{detail}"
                )
        require(not failures, "\n".join(failures))

        payloads = [
            read_shard_payload(temp_dir, int(shard["shard_id"]))
            for shard in SHARDS
        ]
        items, shard_meta = merge_shards(payloads, prior_q0)

    checkpoint_sha = shard_meta[0]["checkpoint_sha256"]

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "source_pair_count": N,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "reference_epsilon": REFERENCE_EPSILON,
        "new_epsilons": list(NEW_EPSILONS),
        "epsilon_execution_order": list(NEW_EPSILONS),
        "plane_order": list(PLANE_ORDER),
        "principal_direction_order": list(DIRECTION_ORDER),
        "positive_contrast_eigenvalues": list(eigenvalues),
        "reference_run_name": REFERENCE_RUN_NAME,
        "reference_items_sha256": REFERENCE_ITEMS_SHA256,
        "reference_summary_sha256": REFERENCE_SUMMARY_SHA256,
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "reference_scientific_model_forward_count_this_run": 0,
        "prior_native_q0_reused": True,
        "model_forwards_per_direction": F_DIRECTION,
        "model_forwards_per_pair_per_new_epsilon": F_PAIR_PER_EPSILON,
        "model_forwards_per_pair_this_run": F_PAIR,
        "scientific_model_forward_count_per_new_epsilon":
            F_TOTAL_PER_EPSILON,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "new_original_basis_scientific_model_forward_count": 0,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shard_meta,
        "representative_checkpoint_sha256": checkpoint_sha,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }

    write_outputs(output_dir, items, summary, prior_q0)
    validate_artifact(output_dir)
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw five-plane finite-difference observations at the "
            "two preregistered smaller epsilons 0.0125 and 0.00625. "
            "The frozen epsilon=0.025 reference is reused without rerun."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_observation(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + summary["result"])
    print("REFERENCE_EPSILON=0.025")
    print("REFERENCE_EPSILON_RERUN=False")
    print("NEW_EPSILONS=0.0125,0.00625")
    print("GPU_COUNT=" + str(summary["gpu_count"]))
    for shard in summary["shards"]:
        print(
            f"SHARD_{shard['shard_id']}_GPU="
            f"{shard['gpu_id']}:"
            f"{shard['pair_first']}..{shard['pair_last']}:"
            f"FORWARDS={shard['scientific_model_forward_count_this_run']}"
        )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_PER_NEW_EPSILON="
        + str(summary["scientific_model_forward_count_per_new_epsilon"])
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("PRIOR_NATIVE_Q0_REUSED=True")
    print("NEW_ORIGINAL_BASIS_SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
