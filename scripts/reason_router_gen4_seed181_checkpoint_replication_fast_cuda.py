from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_family_subspace_sensitivity_fast_cuda as subspace
from scripts import reason_router_gen4_pp3_restoration_sufficiency_fast_cuda as restoration
from scripts import reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1 as phase1


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "g4k-seed181-checkpoint-replication-prep"
BASE_COMMIT = "fb6498e52d0410c64d458896b555f4cbdbf5e407"

SEED = 181
ARM = "G3-GROUP-D-HALF"
CHECKPOINT_SHA256 = "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
CHECKPOINT_BYTES = 518270455
SELECTED_EPOCH = 19

FAMILIES = ("xg2", "xg4")
N = 300
DIM = 395
K = 5
EPS = 0.025
TOL = 1.0e-12
EIG_TOL = 1.0e-10

XG1_DATA_ROOT = restoration.DATA_ROOT
XG1_STATIC_ROOT = restoration.STATIC_ROOT

GEOMETRY_FORWARD_PER_FAMILY = 1200
PRINCIPAL_FORWARD_PER_PAIR = 40
RESTORATION_FORWARD_PER_PAIR = 120
TOTAL_FORWARD_BUDGET = 2 * GEOMETRY_FORWARD_PER_FAMILY + N * (
    PRINCIPAL_FORWARD_PER_PAIR + RESTORATION_FORWARD_PER_PAIR
)

ITEM_FILE = "seed181_checkpoint_replication_items.jsonl"
SUMMARY_FILE = "seed181_checkpoint_replication_summary.json"
GEOMETRY_FILE = "seed181_principal_geometry.json"
GEOMETRY_TENSOR_FILE = "seed181_principal_geometry.pt"
CHECKSUM_FILE = "SHA256SUMS.txt"
XG2_ITEMS_FILE = "seed181_xg2_geometry_items.jsonl"
XG4_ITEMS_FILE = "seed181_xg4_geometry_items.jsonl"
XG2_PLANS_FILE = "seed181_xg2_alignment_delta_h.pt"
XG4_PLANS_FILE = "seed181_xg4_alignment_delta_h.pt"

REUSED_PATHS = (
    "scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py",
    "scripts/reason_router_gen4_family_subspace_sensitivity_fast_cuda.py",
    "scripts/reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py",
    "scripts/reason_router_gen4_native_mamba_state_extraction.py",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py",
)


class ReplicationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ReplicationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplicationError("GIT_FAILURE:" + " ".join(args)) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def authenticate_repo(expected_head: str) -> None:
    require(git("branch", "--show-current") == EXPECTED_BRANCH, "BRANCH")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    require(
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "BASE_NOT_ANCESTOR",
    )
    require(
        subprocess.call(
            ["git", "diff", "--quiet", BASE_COMMIT, expected_head, "--", *REUSED_PATHS],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "HISTORICAL_DEPENDENCY_DRIFT",
    )


def authenticate_checkpoint(path: Path) -> None:
    require(path.is_file(), "CHECKPOINT_MISSING")
    require(path.stat().st_size == CHECKPOINT_BYTES, "CHECKPOINT_BYTES")
    require(sha256_file(path) == CHECKPOINT_SHA256, "CHECKPOINT_SHA256")
    adapter = restoration.adapter
    require(adapter.expected_checkpoint_sha256(SEED, ARM) == CHECKPOINT_SHA256, "REGISTRY_SHA")


@contextlib.contextmanager
def seed181_binding():
    extraction = restoration.holdout.phase1.base.prevalence_eq.extraction
    old = (
        extraction.REPRESENTATIVE_SEED,
        extraction.REPRESENTATIVE_ARM,
        extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
    )
    extraction.REPRESENTATIVE_SEED = SEED
    extraction.REPRESENTATIVE_ARM = ARM
    extraction.REPRESENTATIVE_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    try:
        yield extraction
    finally:
        (
            extraction.REPRESENTATIVE_SEED,
            extraction.REPRESENTATIVE_ARM,
            extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        ) = old


def geometry_extract_family(
    family: str,
    *,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    require(family in FAMILIES, "FAMILY")
    runtime = phase1.base.prevalence_eq
    phase1.base.validate_equivalence_artifact(family)
    runtime.backend.runtime_gate()

    with runtime.backend.parent_runtime_rebind():
        rows, encoded, event_rows = phase1.base.fresh_eq.load_family_inputs(
            family, tokenizer_snapshot
        )
        pairs = phase1.base.fresh_eq._pair_order(family, rows)
        require(tuple(pairs) == phase1._expected_pairs(family), "PAIR_ORDER")
        events = runtime.parent.event_lookup(event_rows)
        runtime.parent.validate_transport_event_plan(pairs, events)
        row_index = runtime.parent.build_row_index(rows)
        trace_code, trace_line = runtime.measurement._resolve_and_validate_runtime_binding()
        kernels = runtime.kernel_compat.load_exact_fast_kernels()

        with runtime.kernel_compat.exact_transformers_kernel_loader(kernels) as calls:
            model, checkpoint_sha = runtime.parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
            require(checkpoint_sha == CHECKPOINT_SHA256, "CHECKPOINT_LOAD_IDENTITY")
            runtime_ctx = runtime.transport_runtime.validate_runtime_components(model)

        counts = Counter(calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"}
            and counts["causal-conv1d"] > 0
            and counts["causal-conv1d"] == counts["mamba-ssm"],
            "KERNEL_CONSTRUCTOR",
        )
        runtime.kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(device)
        model.eval()
        fast_capture = runtime.backend._make_fast_capture(kernels)
        original_capture = runtime.parent.capture_branch
        budget = runtime.parent.ForwardBudget(GEOMETRY_FORWARD_PER_FAMILY)
        items: list[dict[str, Any]] = []
        plans: list[torch.Tensor] = []
        runtime.parent.capture_branch = fast_capture
        try:
            for index, pair in enumerate(pairs):
                item, plan = phase1._run_restartable_baseline_pair(
                    family,
                    pair,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                item = dict(item)
                item["alignment_plan_index"] = index
                item["replication_seed"] = SEED
                item["replication_arm"] = ARM
                items.append(item)
                plans.append(plan.detach().cpu().to(torch.float64).contiguous())
            budget.assert_exact()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            runtime.parent.capture_branch = original_capture

    plan_tensor = torch.stack(plans, dim=0).contiguous()
    require(tuple(plan_tensor.shape) == (N, DIM), "PLAN_SHAPE")
    return {
        "family": family,
        "items": items,
        "plans": plan_tensor,
        "checkpoint_sha256": checkpoint_sha,
        "scientific_model_forward_count": GEOMETRY_FORWARD_PER_FAMILY,
    }


def canonicalize_columns(x: torch.Tensor) -> torch.Tensor:
    out = x.detach().cpu().to(torch.float64).contiguous().clone()
    for column in range(out.shape[1]):
        v = out[:, column]
        pivot = int(torch.argmax(torch.abs(v)).item())
        require(float(v[pivot]) != 0.0, "SIGN_PIVOT")
        if float(v[pivot]) < 0.0:
            out[:, column].mul_(-1.0)
    return out


def projector_modes(
    b2: torch.Tensor, b4: torch.Tensor
) -> dict[str, Any]:
    cross = b2.T @ b4
    u_small, cosines, vh = torch.linalg.svd(cross, full_matrices=False)
    v_small = vh.T
    u = b2 @ u_small
    v = b4 @ v_small

    planes: dict[str, torch.Tensor] = {}
    eigenvalues: list[float] = []
    angles: list[float] = []
    for i in range(K):
        c = float(cosines[i])
        c = max(-1.0, min(1.0, c))
        s = math.sqrt(max(0.0, 1.0 - c * c))
        require(s > EIG_TOL, f"DEGENERATE_PRINCIPAL_PLANE:{i}")
        e = u[:, i]
        f = (v[:, i] - c * e) / s
        block = torch.tensor(
            [[s * s, -c * s], [-c * s, -s * s]], dtype=torch.float64
        )
        vals, vecs = torch.linalg.eigh(block)
        minus_idx = int(torch.argmin(vals).item())
        plus_idx = int(torch.argmax(vals).item())
        z_minus = e * vecs[0, minus_idx] + f * vecs[1, minus_idx]
        z_plus = e * vecs[0, plus_idx] + f * vecs[1, plus_idx]
        pair = canonicalize_columns(torch.stack([z_plus, z_minus], dim=1))
        gram = pair.T @ pair
        require(
            float(torch.max(torch.abs(gram - torch.eye(2, dtype=torch.float64))))
            <= 1.0e-10,
            f"PLANE_GRAM:{i}",
        )
        require(abs(float(vals[plus_idx]) - s) <= 1.0e-10, f"PLUS_EIG:{i}")
        require(abs(float(vals[minus_idx]) + s) <= 1.0e-10, f"MINUS_EIG:{i}")
        planes[f"P{i+1}_plus"] = pair[:, 0].contiguous()
        planes[f"P{i+1}_minus"] = pair[:, 1].contiguous()
        eigenvalues.append(s)
        angles.append(math.degrees(math.acos(c)))

    matrix = torch.stack(
        [planes[f"P{i}_plus"] for i in range(1, 6)]
        + [planes[f"P{i}_minus"] for i in range(1, 6)],
        dim=1,
    )
    require(bool(torch.isfinite(matrix).all()), "NONFINITE_MODES")

    full_gram = matrix.T @ matrix
    gram_residual = float(
        torch.max(
            torch.abs(full_gram - torch.eye(2 * K, dtype=torch.float64))
        ).item()
    )
    require(gram_residual <= 1.0e-10, f"PRINCIPAL_MODE_GRAM:{gram_residual}")

    contrast = b2 @ b2.T - b4 @ b4.T
    reconstructed = torch.zeros_like(contrast)
    for i, eig in enumerate(eigenvalues, 1):
        z_plus = planes[f"P{i}_plus"][:, None]
        z_minus = planes[f"P{i}_minus"][:, None]
        reconstructed += float(eig) * (z_plus @ z_plus.T - z_minus @ z_minus.T)
    operator_residual = float(torch.max(torch.abs(contrast - reconstructed)).item())
    require(
        operator_residual <= 1.0e-10,
        f"PROJECTOR_CONTRAST_RECONSTRUCTION:{operator_residual}",
    )

    sorted_eigs = sorted(float(x) for x in eigenvalues)
    min_positive_eigengap = min(
        right - left for left, right in zip(sorted_eigs[:-1], sorted_eigs[1:], strict=True)
    )
    require(
        min_positive_eigengap > EIG_TOL,
        f"PRINCIPAL_PLANE_NONIDENTIFIABLE:{min_positive_eigengap}",
    )

    return {
        "planes": planes,
        "positive_eigenvalues": tuple(eigenvalues),
        "principal_angles_deg": tuple(angles),
        "principal_cosines": tuple(float(x) for x in cosines.tolist()),
        "principal_mode_gram_max_abs_residual": gram_residual,
        "projector_contrast_reconstruction_max_abs_residual": operator_residual,
        "minimum_positive_eigengap": min_positive_eigengap,
    }


def plane_matrix(planes: Mapping[str, torch.Tensor], index: int) -> torch.Tensor:
    return torch.stack(
        [planes[f"P{index}_plus"], planes[f"P{index}_minus"]], dim=1
    ).to(torch.float64)


def match_homolog(planes: Mapping[str, torch.Tensor], eigenvalues: Sequence[float]) -> dict[str, Any]:
    frozen = restoration.load_planes()
    h180 = torch.stack([frozen["pp3_plus"], frozen["pp3_minus"]], dim=1).to(
        torch.float64
    )
    overlaps: list[float] = []
    for i in range(1, 6):
        zi = plane_matrix(planes, i)
        overlaps.append(float(torch.sum((h180.T @ zi) ** 2)))

    order = sorted(range(5), key=lambda i: (-overlaps[i], i))
    require(overlaps[order[0]] - overlaps[order[1]] > TOL, "HOMOLOG_TIE")
    homolog = order[0] + 1

    control_candidates = [i for i in range(1, 6) if i != homolog]
    control_order = sorted(
        control_candidates, key=lambda i: (-float(eigenvalues[i - 1]), i)
    )
    require(
        len(control_order) >= 2
        and float(eigenvalues[control_order[0] - 1])
        - float(eigenvalues[control_order[1] - 1])
        > TOL,
        "CONTROL_TIE",
    )
    control = control_order[0]
    return {
        "homolog_plane_index": homolog,
        "control_plane_index": control,
        "seed180_pp3_overlap_by_seed181_plane": overlaps,
        "homolog_overlap": overlaps[homolog - 1],
        "control_positive_eigenvalue": float(eigenvalues[control - 1]),
    }


def build_geometry(xg2_plans: torch.Tensor, xg4_plans: torch.Tensor) -> dict[str, Any]:
    b2_info = subspace.reconstruct_family_basis("xg2", xg2_plans)
    b4_info = subspace.reconstruct_family_basis("xg4", xg4_plans)
    b2 = b2_info["basis"].detach().cpu().to(torch.float64).contiguous()
    b4 = b4_info["basis"].detach().cpu().to(torch.float64).contiguous()
    modes = projector_modes(b2, b4)
    match = match_homolog(modes["planes"], modes["positive_eigenvalues"])
    return {
        "bases": {"xg2": b2, "xg4": b4},
        **modes,
        **match,
        "xg2_basis_info": {k: v for k, v in b2_info.items() if k != "basis"},
        "xg4_basis_info": {k: v for k, v in b4_info.items() if k != "basis"},
    }


def validate_checkpoint_independent_xg1_inputs() -> None:
    checks = {
        XG1_DATA_ROOT / "structured_source_facts.jsonl": restoration.SOURCE_SHA,
        XG1_DATA_ROOT / "synthetic_reason_router_six_cell.jsonl": restoration.ROWS_SHA,
        XG1_DATA_ROOT / "structural_manifest.json": restoration.STRUCT_SHA,
        XG1_STATIC_ROOT / "tokenizer_anchor_manifest.jsonl": restoration.ANCHOR_SHA,
        XG1_STATIC_ROOT / "tokenizer_eligibility_summary.json": restoration.ELIG_SHA,
    }
    for relative, expected in checks.items():
        path = ROOT / relative
        require(path.is_file(), f"MISSING_INPUT:{relative}")
        require(sha256_file(path) == expected, f"INPUT_SHA:{relative}")


def load_xg1_inputs(tokenizer_snapshot: Path):
    validate_checkpoint_independent_xg1_inputs()
    rows = restoration.adapter.validate_gen4_rows(
        restoration.read_jsonl(ROOT / XG1_DATA_ROOT / "synthetic_reason_router_six_cell.jsonl"),
        require_canonical_shape=True,
    )
    pairs = restoration.pair_order(rows)
    tokenizer, _ = restoration.tokenizer_gate.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    encoded = restoration.adapter.encode_gen4_rows(rows, tokenizer)
    event_rows = restoration.read_jsonl(
        ROOT / XG1_STATIC_ROOT / "tokenizer_anchor_manifest.jsonl"
    )
    require(len(rows) == restoration.ROWS and len(event_rows) == restoration.ROWS, "ROW_COUNTS")
    require(
        list(encoded["source_pair_id"]) == [str(row["source_pair_id"]) for row in rows],
        "ENCODED_PAIR_ORDER",
    )
    return rows, encoded, event_rows, pairs


def alias_planes(geometry: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    planes = geometry["planes"]
    h = int(geometry["homolog_plane_index"])
    c = int(geometry["control_plane_index"])
    return {
        "pp3_plus": planes[f"P{h}_plus"],
        "pp3_minus": planes[f"P{h}_minus"],
        "pp5_plus": planes[f"P{c}_plus"],
        "pp5_minus": planes[f"P{c}_minus"],
    }


def principal_probe_pair(seed: Mapping[str, Any], *, geometry: Mapping[str, Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    planes = geometry["planes"]
    aliases = alias_planes(geometry)
    j: dict[str, float] = {}
    probe_rows: list[dict[str, Any]] = []
    for i in range(1, 6):
        for sign in ("plus", "minus"):
            key = f"P{i}_{sign}"
            probe = restoration.run_direction(
                seed,
                planes[key],
                condition="pp3_restored",
                family=f"P{i}_{sign}",
                index=0,
                planes=aliases,
                **dict(kwargs),
            )
            j[key] = float(probe["J"])
            probe_rows.append(probe)

    contributions: dict[str, float] = {}
    for i, eig in enumerate(geometry["positive_eigenvalues"], 1):
        contributions[f"P{i}"] = float(eig) * (
            j[f"P{i}_plus"] ** 2 - j[f"P{i}_minus"] ** 2
        ) / K
    return {
        "principal_direction_probes": probe_rows,
        "principal_J": j,
        "plane_contributions": contributions,
        "Q_principal": math.fsum(contributions.values()),
        "scientific_model_forward_count": PRINCIPAL_FORWARD_PER_PAIR,
    }


def run_causal_core(
    *,
    geometry: Mapping[str, Any],
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    runtime = restoration.holdout.phase1.base.prevalence_eq
    runtime.backend.runtime_gate()
    aliases = alias_planes(geometry)
    bases = geometry["bases"]

    with runtime.backend.parent_runtime_rebind():
        rows, encoded, event_rows, pairs = load_xg1_inputs(tokenizer_snapshot)
        parent = runtime.parent
        events = parent.event_lookup(event_rows)
        parent.validate_transport_event_plan(pairs, events)
        row_index = parent.build_row_index(rows)
        trace_code, trace_line = runtime.measurement._resolve_and_validate_runtime_binding()
        kernels = runtime.kernel_compat.load_exact_fast_kernels()
        with runtime.kernel_compat.exact_transformers_kernel_loader(kernels) as calls:
            model, checkpoint_sha = parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
            require(checkpoint_sha == CHECKPOINT_SHA256, "CAUSAL_CHECKPOINT")
            runtime_ctx = runtime.transport_runtime.validate_runtime_components(model)
        counts = Counter(calls)
        require(
            set(counts) == {"causal-conv1d", "mamba-ssm"}
            and counts["causal-conv1d"] > 0
            and counts["causal-conv1d"] == counts["mamba-ssm"],
            "CAUSAL_KERNELS",
        )
        runtime.kernel_compat.validate_transformers_kernel_bindings(kernels)
        model.to(device)
        model.eval()
        fast_capture = runtime.backend._make_fast_capture(kernels)
        original_capture = parent.capture_branch
        budget = parent.ForwardBudget(N * (PRINCIPAL_FORWARD_PER_PAIR + RESTORATION_FORWARD_PER_PAIR))
        items: list[dict[str, Any]] = []
        parent.capture_branch = fast_capture
        try:
            for index, pair in enumerate(pairs):
                seed = restoration.probe_seed(index, pair, events)
                common = {
                    "model": model,
                    "runtime_ctx": runtime_ctx,
                    "trace_code": trace_code,
                    "trace_line": trace_line,
                    "encoded": encoded,
                    "row_index": row_index,
                    "events": events,
                    "budget": budget,
                }
                rest_item = restoration.run_pair(seed, bases=bases, planes=aliases, **common)
                principal = principal_probe_pair(seed, geometry=geometry, kwargs=common)
                q_native = float(rest_item["Q_R3"])
                q_principal = float(principal["Q_principal"])
                items.append(
                    {
                        "schema_version": "gen4-seed181-checkpoint-replication-item-v1",
                        "source_pair_id": pair,
                        "pair_index": index,
                        "replication_seed": SEED,
                        "replication_arm": ARM,
                        "checkpoint_sha256": checkpoint_sha,
                        "homolog_plane_index": int(geometry["homolog_plane_index"]),
                        "control_plane_index": int(geometry["control_plane_index"]),
                        "Q_neutralized": float(rest_item["Q_B"]),
                        "Q_restored": q_native,
                        "Q_control": float(rest_item["Q_R5"]),
                        "S_homolog": float(rest_item["S3"]),
                        "S_control": float(rest_item["S5"]),
                        "D_SUF": float(rest_item["D_SUF"]),
                        "Q_principal": q_principal,
                        "reconstruction_residual": q_native - q_principal,
                        "restoration_raw": rest_item,
                        "principal_raw": principal,
                        "scientific_model_forward_count_this_run": PRINCIPAL_FORWARD_PER_PAIR + RESTORATION_FORWARD_PER_PAIR,
                    }
                )
            budget.assert_exact()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        finally:
            parent.capture_branch = original_capture
    return items


def serialize_geometry(geometry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    json_value = {
        "schema_version": "gen4-seed181-principal-geometry-v1",
        "seed": SEED,
        "arm": ARM,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "principal_cosines": list(geometry["principal_cosines"]),
        "principal_angles_deg": list(geometry["principal_angles_deg"]),
        "positive_eigenvalues": list(geometry["positive_eigenvalues"]),
        "principal_mode_gram_max_abs_residual": float(
            geometry["principal_mode_gram_max_abs_residual"]
        ),
        "projector_contrast_reconstruction_max_abs_residual": float(
            geometry["projector_contrast_reconstruction_max_abs_residual"]
        ),
        "minimum_positive_eigengap": float(geometry["minimum_positive_eigengap"]),
        "seed180_pp3_overlap_by_seed181_plane": list(
            geometry["seed180_pp3_overlap_by_seed181_plane"]
        ),
        "homolog_plane_index": int(geometry["homolog_plane_index"]),
        "homolog_overlap": float(geometry["homolog_overlap"]),
        "control_plane_index": int(geometry["control_plane_index"]),
        "control_positive_eigenvalue": float(geometry["control_positive_eigenvalue"]),
        "homolog_selection_rule": "argmax_trace_seed180_PP3_projector_times_seed181_principal_plane_projector",
        "control_selection_rule": "max_positive_projector_contrast_eigenvalue_among_nonhomolog_planes",
        "selection_uses_response": False,
        "scientific_model_forward_count": 0,
    }
    tensor_value: dict[str, Any] = {
        "bases": geometry["bases"],
        "planes": geometry["planes"],
        "positive_eigenvalues": torch.tensor(geometry["positive_eigenvalues"], dtype=torch.float64),
    }
    return json_value, tensor_value


def write_result(
    output_dir: Path,
    *,
    expected_head: str,
    xg2: Mapping[str, Any],
    xg4: Mapping[str, Any],
    geometry: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    output_dir.mkdir(parents=True)
    (output_dir / ITEM_FILE).write_bytes(jsonl_bytes(items))
    (output_dir / XG2_ITEMS_FILE).write_bytes(jsonl_bytes(xg2["items"]))
    (output_dir / XG4_ITEMS_FILE).write_bytes(jsonl_bytes(xg4["items"]))
    torch.save(xg2["plans"], output_dir / XG2_PLANS_FILE)
    torch.save(xg4["plans"], output_dir / XG4_PLANS_FILE)
    geometry_json, geometry_tensor = serialize_geometry(geometry)
    (output_dir / GEOMETRY_FILE).write_bytes(canonical_json_bytes(geometry_json))
    torch.save(geometry_tensor, output_dir / GEOMETRY_TENSOR_FILE)

    summary = {
        "schema_version": "gen4-seed181-checkpoint-replication-summary-v1",
        "result": "PASS_RAW_OBSERVATION_ONLY",
        "execution_head": expected_head,
        "base_commit": BASE_COMMIT,
        "replication_seed": SEED,
        "replication_arm": ARM,
        "selected_epoch": SELECTED_EPOCH,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_pair_count": N,
        "xg1_pair_first": "xg1_fact_901",
        "xg1_pair_last": "xg1_fact_1200",
        "homolog_plane_index": int(geometry["homolog_plane_index"]),
        "control_plane_index": int(geometry["control_plane_index"]),
        "geometry_model_forward_count": 2 * GEOMETRY_FORWARD_PER_FAMILY,
        "principal_model_forward_count": N * PRINCIPAL_FORWARD_PER_PAIR,
        "restoration_model_forward_count": N * RESTORATION_FORWARD_PER_PAIR,
        "scientific_model_forward_count_this_run": TOTAL_FORWARD_BUDGET,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "primary_inference_executed": False,
        "scientific_conclusion": None,
    }
    (output_dir / SUMMARY_FILE).write_bytes(canonical_json_bytes(summary))

    checks = {}
    for name in (
        ITEM_FILE,
        SUMMARY_FILE,
        GEOMETRY_FILE,
        GEOMETRY_TENSOR_FILE,
        XG2_ITEMS_FILE,
        XG4_ITEMS_FILE,
        XG2_PLANS_FILE,
        XG4_PLANS_FILE,
    ):
        path = output_dir / name
        checks[name] = sha256_file(path)
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checks.items())),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)
    authenticate_checkpoint(args.checkpoint)
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    device = torch.device(args.device)
    require(str(device) == "cuda:0", "DEVICE_MUST_BE_CUDA0_FOR_FROZEN_FAST_CAPTURE")

    with seed181_binding():
        xg2 = geometry_extract_family(
            "xg2",
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        xg4 = geometry_extract_family(
            "xg4",
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        geometry = build_geometry(xg2["plans"], xg4["plans"])
        items = run_causal_core(
            geometry=geometry,
            model_snapshot=args.model_snapshot,
            tokenizer_snapshot=args.tokenizer_snapshot,
            checkpoint_path=args.checkpoint,
            device=device,
        )

    require(len(items) == N, "FINAL_ITEM_COUNT")
    require(
        sum(int(row["scientific_model_forward_count_this_run"]) for row in items)
        + xg2["scientific_model_forward_count"]
        + xg4["scientific_model_forward_count"]
        == TOTAL_FORWARD_BUDGET,
        "TOTAL_FORWARD_COUNT",
    )
    write_result(
        args.output_dir,
        expected_head=args.expected_head,
        xg2=xg2,
        xg4=xg4,
        geometry=geometry,
        items=items,
    )
    print("RESULT=PASS_RAW_OBSERVATION_ONLY")
    print(f"CHECKPOINT_SHA256={CHECKPOINT_SHA256}")
    print(f"HOMOLOG_PLANE=P{geometry['homolog_plane_index']}")
    print(f"CONTROL_PLANE=P{geometry['control_plane_index']}")
    print(f"SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN={TOTAL_FORWARD_BUDGET}")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
