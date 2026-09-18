from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_finite_epsilon_five_plane_decomposition_fast_cuda
    as decomposition
)
from scripts import (
    reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda
    as base
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

STAGE2_RESULT_COMMIT = "fb6498e52d0410c64d458896b555f4cbdbf5e407"
STAGE2_RUN_NAME = (
    "g4k-finite-epsilon-five-plane-decomposition-"
    "xg1-2401-2700-e27dbbd-retry1"
)
STAGE2_ROOT = Path(
    "reports/reason_router_gen4_finite_epsilon_five_plane_decomposition_runs"
) / STAGE2_RUN_NAME

STAGE2_ITEMS_SHA256 = (
    "8db506d872ff81e72b08d44f9ff0af907cb1a65086c0071e3e365a75bd166e17"
)
STAGE2_SUMMARY_SHA256 = (
    "7f9e01dd28ffb2d2a286b464aa4701c639e68f265e11de9eef67ca1e3c1e448e"
)
STAGE2_MANIFEST_SHA256 = (
    "bcf6facd7d7a506b2dcbd1da4ce6eca67749eaa8e0e014bfb0ffac0467fa1d10"
)

N = 300
DIM = 395
K = 5
EPS = 0.025
TOL = 1.0e-12
FLOAT_IDENTITY_ULPS = 8

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")
PAIR_PARTNERS = ("P1", "P2", "P4", "P5")

CONDITION_PLAN = {
    "p1_neutralized": ("P1",),
    "p2_neutralized": ("P2",),
    "p3_neutralized": ("P3",),
    "p4_neutralized": ("P4",),
    "p5_neutralized": ("P5",),
    "p3_p1_joint_neutralized": ("P3", "P1"),
    "p3_p2_joint_neutralized": ("P3", "P2"),
    "p3_p4_joint_neutralized": ("P3", "P4"),
    "p3_p5_joint_neutralized": ("P3", "P5"),
}
CONDITIONS = tuple(CONDITION_PLAN)

DIRECTIONS = tuple(
    [f"xg2_{index}" for index in range(K)]
    + [f"xg4_{index}" for index in range(K)]
)

F_SIGNED = 2
F_DIRECTION = 4
F_CONDITION = 40
F_PAIR = 9 * F_CONDITION
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
        "forward_budget": 54000,
    },
    {
        "shard_id": 1,
        "gpu_id": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_2551",
        "pair_last": "xg1_fact_2700",
        "pair_count": 150,
        "forward_budget": 54000,
    },
)

ITEM_FILE = "pp3_anchored_plane_additivity_items.jsonl"
SUMMARY_FILE = "pp3_anchored_plane_additivity_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-pp3-anchored-plane-additivity-item-v1"
SUMMARY_SCHEMA = "gen4-pp3-anchored-plane-additivity-summary-v1"
MANIFEST_SCHEMA = "gen4-pp3-anchored-plane-additivity-manifest-v1"
RESULT_PASS = "PASS_PP3_ANCHORED_PLANE_ADDITIVITY_RAW_OBSERVATION"


class AdditivityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AdditivityError(message)


def git(*args: str) -> str:
    try:
        return base.git(*args)
    except Exception as exc:
        raise AdditivityError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    rc = base.subprocess.call(
        ["git", "merge-base", "--is-ancestor", STAGE2_RESULT_COMMIT, head],
        cwd=ROOT,
        stdout=base.subprocess.DEVNULL,
        stderr=base.subprocess.DEVNULL,
    )
    require(rc == 0, "STAGE2_RESULT_NOT_ANCESTOR")


def expected_pairs() -> tuple[str, ...]:
    return tuple(
        f"xg1_fact_{index:03d}"
        for index in range(2401, 2701)
    )


def validate_shards() -> None:
    require(len(SHARDS) == GPU_COUNT == 2, "SHARD_COUNT")
    covered: list[int] = []

    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["gpu_id"] == expected_id, "SHARD_GPU")
        require(
            int(shard["end_index"]) - int(shard["start_index"])
            == int(shard["pair_count"]),
            "SHARD_PAIR_COUNT",
        )
        require(
            int(shard["pair_count"]) * F_PAIR
            == int(shard["forward_budget"]),
            "SHARD_FORWARD_BUDGET",
        )

        pair_slice = expected_pairs()[
            int(shard["start_index"]):int(shard["end_index"])
        ]
        require(
            len(pair_slice) == int(shard["pair_count"])
            and pair_slice[0] == shard["pair_first"]
            and pair_slice[-1] == shard["pair_last"],
            "SHARD_PAIR_RANGE",
        )
        covered.extend(
            range(int(shard["start_index"]), int(shard["end_index"]))
        )

    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        sum(int(shard["forward_budget"]) for shard in SHARDS) == F_TOTAL,
        "TOTAL_FORWARD_BUDGET",
    )


def load_stage2_q0() -> list[float]:
    root = ROOT / STAGE2_ROOT
    require(root.is_dir(), "STAGE2_ROOT_MISSING")

    items_path = root / decomposition.ITEM_FILE
    summary_path = root / decomposition.SUMMARY_FILE
    manifest_path = root / decomposition.MANIFEST_FILE

    require(
        base.sha256_file(items_path) == STAGE2_ITEMS_SHA256,
        "STAGE2_ITEMS_SHA",
    )
    require(
        base.sha256_file(summary_path) == STAGE2_SUMMARY_SHA256,
        "STAGE2_SUMMARY_SHA",
    )
    require(
        base.sha256_file(manifest_path) == STAGE2_MANIFEST_SHA256,
        "STAGE2_MANIFEST_SHA",
    )

    validated = decomposition.validate_artifact(root)
    items = validated["items"]
    summary = validated["summary"]

    require(len(items) == N, "STAGE2_ITEM_COUNT")
    require(
        summary["execution_head"]
        == "e27dbbd45635da4beb03691f5a266a7d72824424",
        "STAGE2_EXECUTION_HEAD",
    )
    require(
        summary["primary_inference_executed"] is False
        and summary["scientific_conclusion"] is None,
        "STAGE2_RAW_BOUNDARY",
    )

    q0: list[float] = []
    for index, (pair, item) in enumerate(
        zip(expected_pairs(), items, strict=True)
    ):
        require(
            item["source_pair_id"] == pair
            and int(item["pair_index"]) == index,
            f"STAGE2_PAIR:{index}",
        )
        value = float(item["prior_native_q0"])
        require(math.isfinite(value), f"STAGE2_Q0_FINITE:{index}")
        q0.append(value)

    return q0


def plane_keys(plane: str) -> tuple[str, str]:
    require(plane in PLANE_ORDER, f"PLANE:{plane}")
    if plane == "P3":
        return "pp3_plus", "pp3_minus"
    key = plane.lower()
    return f"{key}_plus", f"{key}_minus"


def all_plane_components(
    h: torch.Tensor,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    h = h.detach().cpu().to(torch.float64).contiguous()
    require(tuple(h.shape) == (DIM,), "H_SHAPE")

    components: dict[str, torch.Tensor] = {}
    coefficients: dict[str, list[float]] = {}

    for plane in PLANE_ORDER:
        plus_key, minus_key = plane_keys(plane)
        plus = planes[plus_key]
        minus = planes[minus_key]
        a = float(torch.dot(h, plus))
        b = float(torch.dot(h, minus))
        component = (a * plus + b * minus).contiguous()

        coefficients[plane] = [a, b]
        components[plane] = component

    return {
        "h": h,
        "coefficients": coefficients,
        "components": components,
    }


def condition_correction(
    h: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    require(condition in CONDITION_PLAN, f"CONDITION:{condition}")
    selected = CONDITION_PLAN[condition]
    require(
        len(selected) in {1, 2}
        and len(set(selected)) == len(selected),
        f"SELECTED_PLANES:{condition}",
    )

    info = all_plane_components(h, planes)

    correction = torch.zeros(DIM, dtype=torch.float64)
    component_l2: dict[str, float] = {}
    for plane in selected:
        component = info["components"][plane]
        correction = correction - component
        component_l2[plane] = float(torch.linalg.vector_norm(component))

    correction = correction.contiguous()
    post = (info["h"] + correction).contiguous()

    post_coefficients: dict[str, list[float]] = {}
    selected_residual = 0.0
    unselected_drift = 0.0

    for plane in PLANE_ORDER:
        plus_key, minus_key = plane_keys(plane)
        post_pair = [
            float(torch.dot(post, planes[plus_key])),
            float(torch.dot(post, planes[minus_key])),
        ]
        post_coefficients[plane] = post_pair

        native_pair = info["coefficients"][plane]
        if plane in selected:
            selected_residual = max(
                selected_residual,
                abs(post_pair[0]),
                abs(post_pair[1]),
            )
        else:
            unselected_drift = max(
                unselected_drift,
                abs(post_pair[0] - native_pair[0]),
                abs(post_pair[1] - native_pair[1]),
            )

    require(
        selected_residual <= TOL,
        f"SELECTED_RESIDUAL:{condition}:{selected_residual}",
    )
    require(
        unselected_drift <= TOL,
        f"UNSELECTED_DRIFT:{condition}:{unselected_drift}",
    )

    squared_component_norm_sum = math.fsum(
        value * value for value in component_l2.values()
    )
    correction_norm = float(torch.linalg.vector_norm(correction))
    require(
        abs(correction_norm * correction_norm - squared_component_norm_sum)
        <= 5.0e-12,
        f"ORTHOGONAL_SUM_NORM:{condition}",
    )

    return {
        "h": info["h"],
        "d": correction,
        "selected_planes": list(selected),
        "native_plane_coefficients": info["coefficients"],
        "post_plane_coefficients": post_coefficients,
        "component_l2": component_l2,
        "condition_correction_l2": correction_norm,
        "selected_plane_residual_max_abs": selected_residual,
        "unselected_plane_drift_max_abs": unselected_drift,
    }


def apply_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    direction: torch.Tensor,
    orientation: int,
    branch_sign: int,
    audit: dict[str, Any],
) -> torch.Tensor:
    runtime = base.holdout.phase1.base.prevalence_eq
    core = runtime.core

    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_SHAPE",
    )
    require(
        orientation in {-1, 1} and branch_sign in {-1, 1},
        "SIGN",
    )

    mask = strong_mask.detach().cpu().bool().contiguous()
    require(
        mask.numel() == core.INTERMEDIATE_SIZE
        and int(mask.sum()) == DIM,
        "MASK",
    )

    before = output.detach().clone()
    mask_device = mask.to(before.device)

    h = (
        before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )
    condition_info = condition_correction(h, condition, planes)

    vector = (
        direction.detach().cpu().to(torch.float64).contiguous()
    )
    require(
        tuple(vector.shape) == (DIM,)
        and abs(float(torch.linalg.vector_norm(vector)) - 1.0) <= TOL,
        "DIRECTION",
    )

    probe = vector * (
        float(branch_sign) * float(orientation) * EPS
    )
    total = (condition_info["d"] + probe).contiguous()

    out = output.clone()
    intended = total.to(device=out.device, dtype=out.dtype)
    out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device] += intended

    require(
        torch.equal(
            out[:, :, core.INTERMEDIATE_SIZE:],
            before[:, :, core.INTERMEDIATE_SIZE:],
        ),
        "GATE_CHANGED",
    )
    require(
        torch.equal(
            out[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device],
            before[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device],
        ),
        "NONSTRONG_CHANGED",
    )

    if token_index:
        require(
            torch.equal(
                out[:, :token_index, :],
                before[:, :token_index, :],
            ),
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
        out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        - before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
    ).detach().cpu().to(torch.float64)

    residual = float(
        torch.max(
            torch.abs(
                applied
                - intended.detach().cpu().to(torch.float64)
            )
        )
    )
    require(
        residual <= runtime.transport_runtime.RUNTIME_CAST_TOL,
        f"APPLIED_RESIDUAL:{residual}",
    )

    audit.clear()
    audit.update({
        "condition": condition,
        "selected_planes": condition_info["selected_planes"],
        "token_index": int(token_index),
        "orientation": int(orientation),
        "branch_sign": int(branch_sign),
        "coefficient_source": "branch_local_native_five_plane_coordinates",
        "native_plane_coefficients":
            condition_info["native_plane_coefficients"],
        "post_plane_coefficients":
            condition_info["post_plane_coefficients"],
        "component_l2": condition_info["component_l2"],
        "condition_correction_l2":
            condition_info["condition_correction_l2"],
        "selected_plane_residual_max_abs":
            condition_info["selected_plane_residual_max_abs"],
        "unselected_plane_drift_max_abs":
            condition_info["unselected_plane_drift_max_abs"],
        "probe_correction_l2":
            float(torch.linalg.vector_norm(probe)),
        "applied_correction_max_abs_residual": residual,
    })

    return out


def install_hook(mixer17: Any, **kwargs: Any):
    def hook(_module: Any, _args: Any, output: torch.Tensor):
        return apply_hook(output, **kwargs)
    return mixer17.in_proj.register_forward_hook(hook)


def run_signed(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    orientation: int,
    planes: Mapping[str, torch.Tensor],
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: Any,
    encoded: Any,
    row_index: Any,
    events: Any,
    budget: Any,
) -> dict[str, Any]:
    runtime = base.holdout.phase1.base.prevalence_eq
    parent = runtime.parent
    core = runtime.core

    pair = str(seed["source_pair_id"])
    cells = base.holdout.phase1._cells()
    anchors = base.holdout.phase1._anchors_for_pair(pair, events)

    captured: dict[str, Any] = {}
    audits: dict[str, Any] = {}

    for role, branch_sign in (("tp", 1), ("tm", -1)):
        audit: dict[str, Any] = {}
        target = anchors[role] + core.TARGET_OFFSET

        handle = install_hook(
            runtime_ctx["mixer17"],
            token_index=target,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=planes,
            direction=direction,
            orientation=orientation,
            branch_sign=branch_sign,
            audit=audit,
        )
        try:
            captured[role] = parent.capture_branch(
                model,
                runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                input_ids=base.input_row(
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

    plus_efficiency = float(
        parent.path_efficiency(captured["tp"])
    )
    minus_efficiency = float(
        parent.path_efficiency(captured["tm"])
    )

    return {
        "condition": condition,
        "orientation": int(orientation),
        "F": plus_efficiency - minus_efficiency,
        "plus_path_efficiency": plus_efficiency,
        "minus_path_efficiency": minus_efficiency,
        "branch_audits": audits,
        "model_forward_count": F_SIGNED,
    }


def run_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    condition: str,
    family: str,
    index: int,
    **kwargs: Any,
) -> dict[str, Any]:
    positive = run_signed(
        seed,
        direction,
        condition=condition,
        orientation=1,
        **kwargs,
    )
    negative = run_signed(
        seed,
        direction,
        condition=condition,
        orientation=-1,
        **kwargs,
    )

    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (f_plus - f_minus) / (2.0 * EPS)

    return {
        "direction_key": f"{family}_{index}",
        "basis_family": family,
        "basis_index": index,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": F_DIRECTION,
    }


def run_condition(
    seed: Mapping[str, Any],
    *,
    condition: str,
    bases: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []

    for family in ("xg2", "xg4"):
        for index in range(K):
            probes.append(
                run_direction(
                    seed,
                    bases[family][:, index],
                    condition=condition,
                    family=family,
                    index=index,
                    **kwargs,
                )
            )

    require(
        tuple(p["direction_key"] for p in probes) == DIRECTIONS,
        "DIRECTION_ORDER",
    )

    e_xg2 = math.fsum(
        float(p["J_squared"]) for p in probes[:K]
    ) / K
    e_xg4 = math.fsum(
        float(p["J_squared"]) for p in probes[K:]
    ) / K

    return {
        "condition": condition,
        "selected_planes": list(CONDITION_PLAN[condition]),
        "direction_order": list(DIRECTIONS),
        "direction_probes": probes,
        "E_XG2": e_xg2,
        "E_XG4": e_xg4,
        "Q": e_xg2 - e_xg4,
        "scientific_model_forward_count": F_CONDITION,
    }


def endpoints_from_q(
    q0: float,
    q_by_condition: Mapping[str, float],
) -> dict[str, Any]:
    main_effects = {
        plane: float(q0) - float(q_by_condition[f"{plane.lower()}_neutralized"])
        for plane in PLANE_ORDER
    }

    pair_effects: dict[str, dict[str, float]] = {}
    for partner in PAIR_PARTNERS:
        condition = f"p3_{partner.lower()}_joint_neutralized"
        q_joint = float(q_by_condition[condition])
        joint_effect = float(q0) - q_joint
        additive_effect = (
            main_effects["P3"] + main_effects[partner]
        )
        interaction_effect = joint_effect - additive_effect
        q_additive_prediction = float(q0) - additive_effect
        q_interaction_residual = q_joint - q_additive_prediction

        algebra_ulp = max(
            math.ulp(value)
            for value in (
                float(q0),
                q_joint,
                joint_effect,
                additive_effect,
                interaction_effect,
                q_additive_prediction,
                q_interaction_residual,
            )
        )
        require(
            abs(interaction_effect + q_interaction_residual)
            <= FLOAT_IDENTITY_ULPS * algebra_ulp,
            f"INTERACTION_ALGEBRA:{partner}",
        )

        pair_effects[partner] = {
            "Q_joint": q_joint,
            "joint_effect": joint_effect,
            "additive_effect_prediction": additive_effect,
            "interaction_effect": interaction_effect,
            "Q_additive_prediction": q_additive_prediction,
            "Q_interaction_residual": q_interaction_residual,
        }

    return {
        "Q0": float(q0),
        "main_effects": main_effects,
        "pp3_pair_interactions": pair_effects,
    }


def validate_condition(
    condition: Mapping[str, Any],
) -> None:
    name = str(condition["condition"])
    require(name in CONDITION_PLAN, f"COND_NAME:{name}")
    require(
        condition["selected_planes"] == list(CONDITION_PLAN[name]),
        f"COND_PLAN:{name}",
    )
    require(
        condition["direction_order"] == list(DIRECTIONS)
        and int(condition["scientific_model_forward_count"]) == F_CONDITION,
        f"COND_META:{name}",
    )

    probes = condition["direction_probes"]
    require(len(probes) == 2 * K, f"PROBE_COUNT:{name}")

    for expected_key, probe in zip(DIRECTIONS, probes, strict=True):
        require(
            probe["direction_key"] == expected_key
            and int(probe["model_forward_count"]) == F_DIRECTION,
            f"PROBE_META:{name}:{expected_key}",
        )

        for orientation, probe_name in (
            (1, "positive_probe"),
            (-1, "negative_probe"),
        ):
            signed = probe[probe_name]
            require(
                int(signed["orientation"]) == orientation
                and int(signed["model_forward_count"]) == F_SIGNED,
                f"SIGNED_META:{name}:{expected_key}:{orientation}",
            )
            for role, branch_sign in (("tp", 1), ("tm", -1)):
                audit = signed["branch_audits"][role]
                require(
                    audit["condition"] == name
                    and audit["selected_planes"]
                    == list(CONDITION_PLAN[name])
                    and int(audit["orientation"]) == orientation
                    and int(audit["branch_sign"]) == branch_sign,
                    f"AUDIT_COORD:{name}:{expected_key}:{role}",
                )
                require(
                    audit["coefficient_source"]
                    == "branch_local_native_five_plane_coordinates",
                    f"AUDIT_SOURCE:{name}",
                )
                require(
                    float(audit["selected_plane_residual_max_abs"]) <= TOL
                    and float(audit["unselected_plane_drift_max_abs"]) <= TOL,
                    f"AUDIT_GEOMETRY:{name}",
                )
                require(
                    abs(float(audit["probe_correction_l2"]) - EPS) <= TOL,
                    f"AUDIT_PROBE_L2:{name}",
                )
                require(
                    float(audit["applied_correction_max_abs_residual"])
                    <= base.holdout.phase1.base.prevalence_eq
                    .transport_runtime.RUNTIME_CAST_TOL,
                    f"AUDIT_APPLIED:{name}",
                )

        f_plus = float(probe["F_plus"])
        f_minus = float(probe["F_minus"])
        j_value = float(probe["J"])
        require(
            j_value == (f_plus - f_minus) / (2.0 * EPS)
            and float(probe["J_squared"]) == j_value * j_value,
            f"J_ID:{name}:{expected_key}",
        )

    e_xg2 = math.fsum(
        float(p["J_squared"]) for p in probes[:K]
    ) / K
    e_xg4 = math.fsum(
        float(p["J_squared"]) for p in probes[K:]
    ) / K
    require(
        float(condition["E_XG2"]) == e_xg2
        and float(condition["E_XG4"]) == e_xg4
        and float(condition["Q"]) == e_xg2 - e_xg4,
        f"Q_ID:{name}",
    )


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
        float(item["epsilon"]) == EPS
        and item["condition_order"] == list(CONDITIONS)
        and item["direction_order"] == list(DIRECTIONS),
        f"ITEM_META:{index}",
    )
    require(
        item["baseline_q0_reused"] is True
        and item["baseline_run_name"] == STAGE2_RUN_NAME
        and item["baseline_items_sha256"] == STAGE2_ITEMS_SHA256
        and float(item["Q0"]) == float(expected_q0),
        f"BASELINE:{index}",
    )
    require(
        int(item["scientific_model_forward_count_this_run"]) == F_PAIR
        and int(item["new_baseline_forward_count"]) == 0,
        f"ITEM_BUDGET:{index}",
    )
    require(
        item["primary_inference_executed"] is False
        and item["multiplicity_correction_executed"] is False
        and item["scientific_conclusion"] is None,
        f"ITEM_BOUNDARY:{index}",
    )

    conditions = item["conditions"]
    require(
        tuple(condition["condition"] for condition in conditions)
        == CONDITIONS,
        f"ITEM_CONDITION_ORDER:{index}",
    )

    for condition in conditions:
        validate_condition(condition)

    q_by_condition = {
        str(condition["condition"]): float(condition["Q"])
        for condition in conditions
    }
    expected = endpoints_from_q(float(expected_q0), q_by_condition)

    require(
        item["main_effects"] == expected["main_effects"],
        f"MAIN_EFFECTS:{index}",
    )
    require(
        item["pp3_pair_interactions"]
        == expected["pp3_pair_interactions"],
        f"PAIR_INTERACTIONS:{index}",
    )


def run_pair(
    seed: Mapping[str, Any],
    *,
    bases: Mapping[str, torch.Tensor],
    planes: Mapping[str, torch.Tensor],
    q0: float,
    **kwargs: Any,
) -> dict[str, Any]:
    conditions = [
        run_condition(
            seed,
            condition=condition,
            bases=bases,
            planes=planes,
            **kwargs,
        )
        for condition in CONDITIONS
    ]

    require(
        tuple(condition["condition"] for condition in conditions)
        == CONDITIONS,
        "CONDITION_ORDER",
    )

    q_by_condition = {
        str(condition["condition"]): float(condition["Q"])
        for condition in conditions
    }
    endpoints = endpoints_from_q(float(q0), q_by_condition)

    item = {
        **dict(seed),
        "schema_version": ITEM_SCHEMA,
        "epsilon": EPS,
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "baseline_q0_reused": True,
        "baseline_run_name": STAGE2_RUN_NAME,
        "baseline_items_sha256": STAGE2_ITEMS_SHA256,
        "conditions": conditions,
        **endpoints,
        "new_baseline_forward_count": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "scientific_conclusion": None,
    }
    return item


def validate_items(
    items: Sequence[Mapping[str, Any]],
    q0_values: Sequence[float],
) -> None:
    require(len(items) == len(q0_values) == N, "ITEM_COUNT")
    for index, (pair, item, q0) in enumerate(
        zip(expected_pairs(), items, q0_values, strict=True)
    ):
        validate_item(item, pair, index, float(q0))


def validate_shard_items(
    items: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
    q0_values: Sequence[float],
) -> None:
    require(
        len(items) == int(shard["pair_count"]),
        "SHARD_ITEM_COUNT",
    )
    start = int(shard["start_index"])
    for local_index, item in enumerate(items):
        global_index = start + local_index
        validate_item(
            item,
            expected_pairs()[global_index],
            global_index,
            float(q0_values[global_index]),
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
        base.canonical(payload)
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
    q0_values: Sequence[float],
    temp_dir: Path,
) -> None:
    shard_id = int(shard["shard_id"])
    error_path = temp_dir / f"shard_{shard_id}.error.txt"

    try:
        gpu_id = int(shard["gpu_id"])
        require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
        require(
            torch.cuda.device_count() >= GPU_COUNT,
            "CUDA_DEVICE_COUNT",
        )
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        authenticate_repo(expected_head)
        validate_shards()
        base.validate_static_inputs()

        planes = base.load_planes()
        bases = base.load_bases()

        runtime = base.holdout.phase1.base.prevalence_eq
        base.runtime_gate_for_device(runtime, gpu_id)

        with runtime.backend.parent_runtime_rebind():
            rows, encoded, event_rows = base.load_inputs(
                tokenizer_snapshot
            )
            pairs = base.pair_order(rows)

            parent = runtime.parent
            events = parent.event_lookup(event_rows)
            row_index = parent.build_row_index(rows)

            trace_code, trace_line = (
                runtime.measurement
                ._resolve_and_validate_runtime_binding()
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
                    runtime.transport_runtime
                    .validate_runtime_components(model)
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

            fast_capture = base.make_fast_capture_for_device(
                runtime, kernels, device
            )
            original_capture = parent.capture_branch
            budget = parent.ForwardBudget(
                int(shard["forward_budget"])
            )

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
                            base.probe_seed(
                                global_index,
                                pair,
                                events,
                            ),
                            bases=bases,
                            planes=planes,
                            q0=float(q0_values[global_index]),
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

        validate_shard_items(items, shard, q0_values)

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
    q0_values: Sequence[float],
    temp_dir: str,
) -> None:
    worker_run(
        shard=shard,
        expected_head=expected_head,
        model_snapshot=Path(model_snapshot),
        tokenizer_snapshot=Path(tokenizer_snapshot),
        checkpoint_path=Path(checkpoint_path),
        q0_values=q0_values,
        temp_dir=Path(temp_dir),
    )


def merge_shards(
    payloads: Sequence[Mapping[str, Any]],
    q0_values: Sequence[float],
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

        validate_shard_items(items, shard, q0_values)
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
    validate_items(merged, q0_values)

    return merged, shard_meta


def write_outputs(
    out: Path,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    q0_values: Sequence[float],
) -> None:
    require(not out.exists(), "OUTPUT_COLLISION")
    validate_items(items, q0_values)
    out.mkdir(parents=True, exist_ok=False)

    payloads = {
        ITEM_FILE: base.jsonl(items),
        SUMMARY_FILE: base.canonical(summary),
    }
    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        (out / name).write_bytes(raw)
        hashes[name] = base.sha256_bytes(raw)

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

    manifest_raw = base.canonical(manifest)
    (out / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = base.sha256_bytes(manifest_raw)

    (out / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def validate_artifact(out: Path) -> dict[str, Any]:
    q0_values = load_stage2_q0()

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

        digest = base.sha256_file(path)
        require(
            digest == manifest["files"][name]["sha256"]
            and path.stat().st_size == manifest["files"][name]["bytes"],
            f"FILE:{name}",
        )
        hashes[name] = digest

    hashes[MANIFEST_FILE] = base.sha256_file(out / MANIFEST_FILE)

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

    items = base.read_jsonl(out / ITEM_FILE)
    validate_items(items, q0_values)

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
        summary["epsilon"] == EPS
        and summary["condition_order"] == list(CONDITIONS)
        and summary["direction_order"] == list(DIRECTIONS),
        "SUMMARY_DESIGN",
    )
    require(
        summary["baseline_q0_reused"] is True
        and summary["baseline_run_name"] == STAGE2_RUN_NAME
        and summary["baseline_items_sha256"] == STAGE2_ITEMS_SHA256,
        "SUMMARY_BASELINE",
    )
    require(
        summary["scientific_model_forward_count_this_run"] == F_TOTAL
        and summary["new_baseline_forward_count"] == 0
        and summary["gpu_count"] == GPU_COUNT,
        "SUMMARY_BUDGET",
    )
    require(
        summary["primary_inference_executed"] is False
        and summary["multiplicity_correction_executed"] is False
        and summary["scientific_conclusion"] is None,
        "SUMMARY_BOUNDARY",
    )

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
    base.validate_static_inputs()

    q0_values = load_stage2_q0()

    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() >= GPU_COUNT,
        "CUDA_DEVICE_COUNT",
    )

    ctx = mp.get_context("spawn")

    with tempfile.TemporaryDirectory(
        prefix="gen4_pp3_anchored_plane_additivity_"
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
                    list(q0_values),
                    str(temp_dir),
                ),
                name=f"gen4-additivity-gpu{shard['gpu_id']}",
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failures = []
        for shard, process in zip(SHARDS, processes, strict=True):
            if process.exitcode != 0:
                error_path = (
                    temp_dir
                    / f"shard_{int(shard['shard_id'])}.error.txt"
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
        items, shard_meta = merge_shards(payloads, q0_values)

    checkpoint_sha = shard_meta[0]["checkpoint_sha256"]

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "source_pair_count": N,
        "pair_id_first": items[0]["source_pair_id"],
        "pair_id_last": items[-1]["source_pair_id"],
        "epsilon": EPS,
        "plane_order": list(PLANE_ORDER),
        "pair_anchor_plane": "P3",
        "pair_partner_planes": list(PAIR_PARTNERS),
        "condition_order": list(CONDITIONS),
        "direction_order": list(DIRECTIONS),
        "baseline_q0_reused": True,
        "baseline_run_name": STAGE2_RUN_NAME,
        "baseline_result_commit": STAGE2_RESULT_COMMIT,
        "baseline_items_sha256": STAGE2_ITEMS_SHA256,
        "model_forwards_per_direction": F_DIRECTION,
        "model_forwards_per_condition": F_CONDITION,
        "model_forwards_per_pair": F_PAIR,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "new_baseline_forward_count": 0,
        "gpu_count": GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shard_meta,
        "representative_checkpoint_sha256": checkpoint_sha,
        "interaction_definition":
            "I_3k=(Q0-Q_3k)-(Q0-Q_3)-(Q0-Q_k)",
        "interaction_scope":
            "P3_anchored_single_and_pair_neutralization_screen",
        "matched_control_interactions_tested": False,
        "full_factorial_executed": False,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion": None,
    }

    write_outputs(output_dir, items, summary, q0_values)
    validate_artifact(output_dir)

    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw P3-anchored five-plane main-effect and pair-interaction "
            "neutralization screen on XG1 2401..2700."
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
    print("GPU_COUNT=" + str(summary["gpu_count"]))
    for shard in summary["shards"]:
        print(
            f"SHARD_{shard['shard_id']}_GPU="
            f"{shard['gpu_id']}:"
            f"{shard['pair_first']}..{shard['pair_last']}:"
            f"FORWARDS={shard['scientific_model_forward_count_this_run']}"
        )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("NEW_BASELINE_FORWARD_COUNT=0")
    print("BASELINE_Q0_REUSED=True")
    print("FULL_FACTORIAL_EXECUTED=False")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
