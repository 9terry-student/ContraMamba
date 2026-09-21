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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda as geom,
)
from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility
    as direct,
)
from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence
    as ref_fd,
)

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_PROGRAM_ANCESTOR = "fa4a68054176412ec1e30be4d14dd7202314435f"

PROGRAM_PATH = Path(
    "reports/reason_router_gen4_next_mechanistic_program_prospective_freeze.md"
)
PROGRAM_GIT_BLOB = "487bf844e27279392a4203c1dc70e5702e3ad325"

GEOM_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_geometry_prepare_fast_cuda.py"
)
GEOM_SCRIPT_GIT_BLOB = "d221c28657cf9bb157d517b635bae42671563903"

DIRECT_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py"
)
DIRECT_SCRIPT_GIT_BLOB = "db523f66d9a05297077af18b547b4f2b2c9379ea"

REFERENCE_FD_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_equivalence.py"
)
REFERENCE_FD_SCRIPT_GIT_BLOB = "3dc94d833f81d829e81c85c0c3641fce861e8eb7"

REFERENCE_FD_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_p5_cross_block_reference_fd_one_row_runs/"
    "g4k-mamba14b-p5-crossblock-ref-fd-gate-4ca4ffd-2t4"
)
REFERENCE_FD_REPORT = REFERENCE_FD_ROOT / "reference_fd_equivalence_report.json"
REFERENCE_FD_SUMS = REFERENCE_FD_ROOT / "SHA256SUMS.txt"
REFERENCE_FD_REPORT_GIT_BLOB = "11ecd4b0122f5d9db126e3ab3dd6fef5f048b8ad"
REFERENCE_FD_SUMS_GIT_BLOB = "6bd895c372542ee1f584f4170ef1192106ba639f"

ADJACENT_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_adjacent_geometry_preparation_runs/"
    "g4k-mamba14b-adjacent-geometry-plus1-xg2xg4-2gpu-4de2451"
)
ADJACENT_STRONG = ADJACENT_ROOT / "strong_indices.json"
ADJACENT_P5_PLUS = ADJACENT_ROOT / "p5_plus.f64le"
ADJACENT_P5_MINUS = ADJACENT_ROOT / "p5_minus.f64le"
ADJACENT_STRONG_GIT_BLOB = "dc5f8f0af4d0375a88fbee93c8769bf4b64492cd"
ADJACENT_P5_PLUS_GIT_BLOB = "6d83d1fc64bcbb7b1c9a779399dd405d82a063a6"
ADJACENT_P5_MINUS_GIT_BLOB = "0ff81176cd5be099eb0da344293621d74b3ce39b"

FAMILIES = ("xg2", "xg4")
SOURCE_PAIR_FIRST = 301
SOURCE_PAIR_LAST = 600
SOURCE_PAIR_COUNT_PER_FAMILY = 300
TOTAL_ROW_COUNT = 600
CELL = "C2_NAME"
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2

SOURCE_BLOCK = 35
TARGET_BLOCK = 36
AMBIENT_DIM = 4096
SOURCE_PLANE = "P5"
TARGET_PLANE = "P5"
BASIS_NAMES = ("plus", "minus")
EPSILON = 0.025
GPU_COUNT = 2

RESULT_PASS = "PASS_MAMBA14B_P5_CROSS_BLOCK_POPULATION_TRANSPORT_MEASUREMENT"
ITEM_SCHEMA = "gen4-mamba14b-p5-cross-block-population-transport-item-v1"
SUMMARY_SCHEMA = "gen4-mamba14b-p5-cross-block-population-transport-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba14b-p5-cross-block-population-transport-manifest-v1"

ITEMS_FILE = "transport_items.jsonl"
SUMMARY_FILE = "transport_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
SUMS_FILE = "SHA256SUMS.txt"


class PopulationTransportError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PopulationTransportError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PopulationTransportError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
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
    return b"".join(canonical_json_bytes(row) for row in rows)


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_PROGRAM_ANCESTOR,
            expected_head,
        )
        == 0,
        "PROGRAM_FREEZE_NOT_ANCESTOR",
    )

    pinned = {
        PROGRAM_PATH: PROGRAM_GIT_BLOB,
        GEOM_SCRIPT_PATH: GEOM_SCRIPT_GIT_BLOB,
        DIRECT_SCRIPT_PATH: DIRECT_SCRIPT_GIT_BLOB,
        REFERENCE_FD_SCRIPT_PATH: REFERENCE_FD_SCRIPT_GIT_BLOB,
        REFERENCE_FD_REPORT: REFERENCE_FD_REPORT_GIT_BLOB,
        REFERENCE_FD_SUMS: REFERENCE_FD_SUMS_GIT_BLOB,
        ADJACENT_STRONG: ADJACENT_STRONG_GIT_BLOB,
        ADJACENT_P5_PLUS: ADJACENT_P5_PLUS_GIT_BLOB,
        ADJACENT_P5_MINUS: ADJACENT_P5_MINUS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )

    report = json.loads(
        REFERENCE_FD_REPORT.read_text(encoding="utf-8")
    )
    require(
        report["result"]
        == "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REFERENCE_FD_EQUIVALENCE",
        "REFERENCE_FD_GATE_NOT_PASS",
    )
    require(
        float(report["fast_estimator"]["epsilon"]) == EPSILON,
        "REFERENCE_FD_EPSILON",
    )
    require(
        report["fast_estimator"]["epsilon_sweep_performed"] is False,
        "REFERENCE_FD_EPSILON_SWEEP",
    )


def validate_protocol() -> None:
    require(FAMILIES == ("xg2", "xg4"), "FAMILIES")
    require(SOURCE_PAIR_FIRST == 301, "SOURCE_PAIR_FIRST")
    require(SOURCE_PAIR_LAST == 600, "SOURCE_PAIR_LAST")
    require(SOURCE_PAIR_COUNT_PER_FAMILY == 300, "SOURCE_PAIR_COUNT_PER_FAMILY")
    require(TOTAL_ROW_COUNT == 600, "TOTAL_ROW_COUNT")
    require(CELL == "C2_NAME", "CELL")
    require(ANCHOR_NAME == "A_IDENTITY", "ANCHOR_NAME")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(SOURCE_BLOCK == 35, "SOURCE_BLOCK")
    require(TARGET_BLOCK == 36, "TARGET_BLOCK")
    require(AMBIENT_DIM == direct.INTERMEDIATE_SIZE == 4096, "AMBIENT_DIM")
    require(SOURCE_PLANE == TARGET_PLANE == "P5", "PLANE")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")
    require(EPSILON == ref_fd.EPSILON == 0.025, "EPSILON")
    require(GPU_COUNT == 2, "GPU_COUNT")


def _read_f64le(path: Path, count: int) -> torch.Tensor:
    raw = path.read_bytes()
    expected_bytes = count * 8
    require(
        len(raw) == expected_bytes,
        f"F64LE_SIZE:{path}:{len(raw)}:{expected_bytes}",
    )
    arr = np.frombuffer(raw, dtype=np.dtype("<f8")).copy()
    require(arr.shape == (count,), f"F64LE_SHAPE:{path}:{arr.shape}")
    value = torch.from_numpy(arr).to(torch.float64).contiguous()
    require(bool(torch.isfinite(value).all().item()), f"F64LE_NONFINITE:{path}")
    return value


def load_adjacent_p5_ambient() -> dict[str, torch.Tensor]:
    strong = json.loads(ADJACENT_STRONG.read_text(encoding="utf-8"))
    require(
        strong["schema_version"] == "gen4-mamba14b-adjacent-strong-mask-v1",
        "ADJ_STRONG_SCHEMA",
    )
    require(int(strong["intervention_layer"]) == 36, "ADJ_INTERVENTION_LAYER")
    require(int(strong["strong_count"]) == 1205, "ADJ_STRONG_COUNT")

    indices = [int(value) for value in strong["strong_indices"]]
    require(len(indices) == 1205, "ADJ_STRONG_INDEX_COUNT")

    out = {
        "plus": direct.scatter_strong_vector(
            _read_f64le(ADJACENT_P5_PLUS, 1205),
            indices,
        ),
        "minus": direct.scatter_strong_vector(
            _read_f64le(ADJACENT_P5_MINUS, 1205),
            indices,
        ),
    }
    for name, value in out.items():
        require(
            tuple(value.shape) == (AMBIENT_DIM,),
            f"ADJ_P5_SHAPE:{name}",
        )
        require(
            abs(float(torch.linalg.vector_norm(value).item()) - 1.0)
            <= 2e-10,
            f"ADJ_P5_NORM:{name}",
        )
    require(
        abs(float(torch.dot(out["plus"], out["minus"]).item()))
        <= 2e-10,
        "ADJ_P5_ORTHOGONALITY",
    )
    return out


def plane_matrix(vectors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    require(set(vectors) == set(BASIS_NAMES), "PLANE_VECTOR_KEYS")
    matrix = torch.stack(
        [
            vectors["plus"].detach().cpu().to(torch.float64).contiguous(),
            vectors["minus"].detach().cpu().to(torch.float64).contiguous(),
        ],
        dim=1,
    )
    require(matrix.ndim == 2 and matrix.shape[1] == 2, "PLANE_MATRIX_SHAPE")
    require(bool(torch.isfinite(matrix).all().item()), "PLANE_MATRIX_NONFINITE")
    gram = matrix.T @ matrix
    require(
        float(torch.max(torch.abs(gram - torch.eye(2, dtype=torch.float64))).item())
        <= 2e-10,
        "PLANE_MATRIX_ORTHONORMALITY",
    )
    return matrix


def numerical_rank(
    matrix: torch.Tensor,
) -> tuple[int, float, torch.Tensor]:
    value = matrix.detach().cpu().to(torch.float64).contiguous()
    require(value.ndim == 2 and value.shape[1] == 2, "RANK_MATRIX_SHAPE")
    require(bool(torch.isfinite(value).all().item()), "RANK_MATRIX_NONFINITE")
    singular = torch.linalg.svdvals(value)
    require(tuple(singular.shape) == (2,), "RANK_SINGULAR_SHAPE")
    tolerance = (
        max(value.shape)
        * torch.finfo(torch.float64).eps
        * float(torch.max(singular).item())
    )
    rank = int(torch.sum(singular > tolerance).item())
    return rank, tolerance, singular


def transport_metrics(
    transported_plus: torch.Tensor,
    transported_minus: torch.Tensor,
    target_plane: torch.Tensor,
) -> dict[str, Any]:
    w_plus = transported_plus.detach().cpu().to(torch.float64).contiguous()
    w_minus = transported_minus.detach().cpu().to(torch.float64).contiguous()
    require(w_plus.ndim == w_minus.ndim == 1, "W_VECTOR_RANK")
    require(w_plus.shape == w_minus.shape, "W_VECTOR_SHAPE")
    require(
        target_plane.ndim == 2
        and target_plane.shape == (w_plus.numel(), 2),
        "TARGET_PLANE_SHAPE",
    )
    require(
        bool(torch.isfinite(w_plus).all().item())
        and bool(torch.isfinite(w_minus).all().item()),
        "W_NONFINITE",
    )

    w = torch.stack([w_plus, w_minus], dim=1)
    rank, rank_tol, singular = numerical_rank(w)
    plus_norm = float(torch.linalg.vector_norm(w_plus).item())
    minus_norm = float(torch.linalg.vector_norm(w_minus).item())
    require(
        math.isfinite(plus_norm)
        and math.isfinite(minus_norm)
        and plus_norm > 0.0
        and minus_norm > 0.0,
        "TRANSPORTED_NORM",
    )

    result: dict[str, Any] = {
        "jv_plus_l2": plus_norm,
        "jv_minus_l2": minus_norm,
        "w_singular_values": [float(value) for value in singular.tolist()],
        "w_rank": rank,
        "w_rank_tolerance": rank_tol,
        "w_condition_number": None,
        "qt_u36_singular_values": None,
        "principal_angles_radians": None,
        "principal_angles_degrees": None,
        "projector_overlap": None,
        "procrustes_residual_fro": None,
        "procrustes_residual_normalized": None,
    }

    if rank < 2:
        return result

    s1 = float(singular[0].item())
    s2 = float(singular[1].item())
    require(s2 > 0.0, "W_S2_ZERO")
    result["w_condition_number"] = s1 / s2

    q, _ = torch.linalg.qr(w, mode="reduced")
    require(q.shape == w.shape, "Q_SHAPE")
    gram = q.T @ q
    require(
        float(torch.max(torch.abs(gram - torch.eye(2, dtype=torch.float64))).item())
        <= 1e-10,
        "Q_ORTHONORMALITY",
    )

    cross = q.T @ target_plane
    sigma = torch.linalg.svdvals(cross)
    require(tuple(sigma.shape) == (2,), "CROSS_SINGULAR_SHAPE")
    sigma = torch.clamp(sigma, 0.0, 1.0)
    angles = torch.acos(sigma)
    overlap = 0.5 * float(torch.sum(sigma * sigma).item())

    u, _s, vh = torch.linalg.svd(cross)
    rotation = u @ vh
    aligned = q @ rotation
    residual = float(
        torch.linalg.matrix_norm(
            aligned - target_plane,
            ord="fro",
        ).item()
    )
    normalized_residual = residual / math.sqrt(2.0)

    require(
        0.0 <= overlap <= 1.0 + 1e-12,
        "PROJECTOR_OVERLAP_RANGE",
    )
    require(math.isfinite(residual), "PROCRUSTES_NONFINITE")

    result.update({
        "qt_u36_singular_values": [
            float(value) for value in sigma.tolist()
        ],
        "principal_angles_radians": [
            float(value) for value in angles.tolist()
        ],
        "principal_angles_degrees": [
            math.degrees(float(value)) for value in angles.tolist()
        ],
        "projector_overlap": min(1.0, max(0.0, overlap)),
        "procrustes_residual_fro": residual,
        "procrustes_residual_normalized": normalized_residual,
    })
    return result


def expected_pairs(family: str) -> tuple[str, ...]:
    require(family in FAMILIES, f"FAMILY:{family}")
    return tuple(
        f"{family}_fact_{index:03d}"
        for index in range(SOURCE_PAIR_FIRST, SOURCE_PAIR_LAST + 1)
    )


def row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW:{key}")
        out[key] = index
    return out


def run_family(
    *,
    family: str,
    gpu_id: int,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(family in FAMILIES, f"FAMILY:{family}")
    require(gpu_id in (0, 1), f"GPU_ID:{gpu_id}")
    geom.validate_fast_runtime_for_device(gpu_id)

    rows, encoded, events, tokenizer_provenance = geom.load_family_inputs(
        family,
        snapshot,
    )
    pairs = expected_pairs(family)
    lookup = row_index(rows)

    model, kernels, model_provenance = geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=compact_checkpoint,
        gpu_id=gpu_id,
    )
    device = torch.device(f"cuda:{gpu_id}")
    layers = model.mamba.layers
    layer35 = layers[SOURCE_BLOCK]
    layer36 = layers[TARGET_BLOCK]

    source_p5 = direct.load_canonical_p5_ambient()
    target_p5 = load_adjacent_p5_ambient()
    target_u36 = plane_matrix(target_p5)

    items: list[dict[str, Any]] = []
    full_model_forward_count = 0
    local_perturbed_forward_count = 0

    for pair in pairs:
        key = (pair, CELL)
        require(key in lookup, f"ROW_MISSING:{family}:{pair}")
        event_key = (pair, CELL, ANCHOR_NAME)
        require(event_key in events, f"EVENT_MISSING:{family}:{pair}")

        index = lookup[key]
        input_ids = encoded["input_ids"][index].unsqueeze(0).contiguous()
        require(tuple(input_ids.shape) == (1, 128), "INPUT_SHAPE")
        anchor = int(events[event_key]["absolute_anchor_token_index"])
        target_abs = anchor + TARGET_OFFSET
        require(0 <= target_abs < 128, "TARGET_RANGE")

        baseline = direct.capture_baseline_boundary(
            model=model,
            input_ids=input_ids,
            target_abs=target_abs,
            device=device,
        )
        full_model_forward_count += 1

        baseline_projected35 = baseline["projected35"].to(device).contiguous()
        residual35 = baseline["residual35"].to(device).contiguous()
        base_content = (
            baseline_projected35[
                0,
                target_abs,
                :AMBIENT_DIM,
            ]
            .detach()
            .clone()
        )

        def fast_phi(content: torch.Tensor) -> torch.Tensor:
            projected = direct.replace_target_content(
                baseline_projected35,
                content,
                target_abs,
            )
            return direct.local_block35_to_block36_map(
                projected35=projected,
                residual35=residual35,
                target_abs=target_abs,
                layer35=layer35,
                layer36=layer36,
                kernels=kernels,
            )

        transported: dict[str, torch.Tensor] = {}
        for basis_name in BASIS_NAMES:
            _plus, _minus, derivative = ref_fd.fast_symmetric_fd(
                fast_phi,
                base_content,
                source_p5[basis_name],
            )
            local_perturbed_forward_count += 2
            transported[basis_name] = (
                derivative.detach().cpu().to(torch.float64).contiguous()
            )

        metrics = transport_metrics(
            transported["plus"],
            transported["minus"],
            target_u36,
        )
        items.append({
            "schema_version": ITEM_SCHEMA,
            "family_key": family,
            "source_pair_id": pair,
            "contrast_cell_id": CELL,
            "anchor_name": ANCHOR_NAME,
            "absolute_anchor_token_index": anchor,
            "target_offset": TARGET_OFFSET,
            "target_abs": target_abs,
            "source_block": SOURCE_BLOCK,
            "target_block": TARGET_BLOCK,
            "source_plane": SOURCE_PLANE,
            "target_plane": TARGET_PLANE,
            "epsilon": EPSILON,
            **metrics,
        })

    require(
        len(items) == SOURCE_PAIR_COUNT_PER_FAMILY,
        f"ITEM_COUNT:{family}:{len(items)}",
    )
    require(
        [str(item["source_pair_id"]) for item in items] == list(pairs),
        f"ITEM_ORDER:{family}",
    )
    require(
        full_model_forward_count == SOURCE_PAIR_COUNT_PER_FAMILY,
        f"FULL_FORWARD_COUNT:{family}",
    )
    require(
        local_perturbed_forward_count
        == SOURCE_PAIR_COUNT_PER_FAMILY * len(BASIS_NAMES) * 2,
        f"LOCAL_FORWARD_COUNT:{family}",
    )
    require(
        not any(parameter.grad is not None for parameter in model.parameters()),
        "PARAMETER_GRAD_CREATED",
    )

    meta = {
        "family_key": family,
        "gpu_id": gpu_id,
        "pair_count": len(items),
        "pair_first": pairs[0],
        "pair_last": pairs[-1],
        "full_model_forward_count": full_model_forward_count,
        "local_perturbed_forward_count": local_perturbed_forward_count,
        "tokenizer_provenance": tokenizer_provenance,
        "model_provenance": model_provenance,
        "parameter_gradient_created": False,
        "training_executed": False,
    }
    return items, meta


def worker_run(
    *,
    family: str,
    gpu_id: int,
    expected_head: str,
    snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    root = Path(temp_dir)
    items_path = root / f"worker_{gpu_id}_items.jsonl"
    meta_path = root / f"worker_{gpu_id}_meta.json"
    error_path = root / f"worker_{gpu_id}_error.txt"
    try:
        items, meta = run_family(
            family=family,
            gpu_id=gpu_id,
            expected_head=expected_head,
            snapshot=Path(snapshot),
            compact_checkpoint=Path(compact_checkpoint),
        )
        item_bytes = jsonl_bytes(items)
        items_path.write_bytes(item_bytes)
        meta = {
            **meta,
            "items_sha256": sha256_bytes(item_bytes),
        }
        meta_path.write_bytes(pretty_json_bytes(meta))
    except Exception:
        error_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def _finite_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        x = float(value)
        require(math.isfinite(x), f"SUMMARY_NONFINITE:{key}")
        out.append(x)
    return out


def descriptive(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(arr).all()), "DESCRIPTIVE_NONFINITE")
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std_population": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_counts = Counter(int(row["w_rank"]) for row in rows)
    metric_keys = (
        "jv_plus_l2",
        "jv_minus_l2",
        "w_condition_number",
        "projector_overlap",
        "procrustes_residual_fro",
        "procrustes_residual_normalized",
    )
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "rank_counts": {
            str(rank): int(rank_counts.get(rank, 0))
            for rank in (0, 1, 2)
        },
    }
    for key in metric_keys:
        summary[key] = descriptive(_finite_values(rows, key))

    for index in (0, 1):
        summary[f"w_singular_value_{index + 1}"] = descriptive(
            [
                float(row["w_singular_values"][index])
                for row in rows
            ]
        )
        sigma_values = [
            float(row["qt_u36_singular_values"][index])
            for row in rows
            if row["qt_u36_singular_values"] is not None
        ]
        summary[f"qt_u36_singular_value_{index + 1}"] = descriptive(
            sigma_values
        )
        angle_deg = [
            float(row["principal_angles_degrees"][index])
            for row in rows
            if row["principal_angles_degrees"] is not None
        ]
        summary[f"principal_angle_degrees_{index + 1}"] = descriptive(
            angle_deg
        )
    return summary


def run_population(
    *,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    require(snapshot.is_dir(), f"SNAPSHOT_DIR:{snapshot}")
    require(
        compact_checkpoint.resolve()
        == (ROOT / geom.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        geom.sha256_file(compact_checkpoint) == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "CUDA_DEVICE_COUNT")

    source_p5 = plane_matrix(direct.load_canonical_p5_ambient())
    target_p5 = plane_matrix(load_adjacent_p5_ambient())
    require(source_p5.shape == target_p5.shape == (AMBIENT_DIM, 2), "PLANE_SHAPE")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba14b_p5_cross_block_population_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for gpu_id, family in enumerate(FAMILIES):
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "family": family,
                    "gpu_id": gpu_id,
                    "expected_head": expected_head,
                    "snapshot": str(snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"p5-cross-block-transport-{family}-gpu{gpu_id}",
            )
            process.start()
            processes.append(process)

        for gpu_id, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                error_path = temp_dir / f"worker_{gpu_id}_error.txt"
                detail = (
                    error_path.read_text(encoding="utf-8")
                    if error_path.is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise PopulationTransportError(
                    f"WORKER_{gpu_id}_FAILED:\n{detail}"
                )

        all_rows: list[dict[str, Any]] = []
        worker_meta: list[dict[str, Any]] = []
        for gpu_id, family in enumerate(FAMILIES):
            items_path = temp_dir / f"worker_{gpu_id}_items.jsonl"
            meta_path = temp_dir / f"worker_{gpu_id}_meta.json"
            require(items_path.is_file(), f"WORKER_ITEMS_MISSING:{gpu_id}")
            require(meta_path.is_file(), f"WORKER_META_MISSING:{gpu_id}")

            raw_items = items_path.read_bytes()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            require(meta["family_key"] == family, "WORKER_FAMILY")
            require(int(meta["gpu_id"]) == gpu_id, "WORKER_GPU")
            require(
                meta["items_sha256"] == sha256_bytes(raw_items),
                "WORKER_ITEMS_SHA",
            )
            rows = [
                json.loads(line)
                for line in raw_items.decode("utf-8").splitlines()
                if line.strip()
            ]
            require(
                len(rows) == SOURCE_PAIR_COUNT_PER_FAMILY,
                "WORKER_ROW_COUNT",
            )
            all_rows.extend(rows)
            worker_meta.append(meta)

    require(len(all_rows) == TOTAL_ROW_COUNT, "TOTAL_ROW_COUNT")
    require(
        [row["family_key"] for row in all_rows[:300]] == ["xg2"] * 300,
        "XG2_ORDER",
    )
    require(
        [row["family_key"] for row in all_rows[300:]] == ["xg4"] * 300,
        "XG4_ORDER",
    )

    family_summary = {
        family: summarize_rows(
            [row for row in all_rows if row["family_key"] == family]
        )
        for family in FAMILIES
    }
    pooled_summary = summarize_rows(all_rows)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "phase": "study_a_cross_layer_causal_transport_population_measurement",
        "claim_boundary": (
            "Descriptive N=600 row-conditioned transport measurement only; "
            "no p-value, response endpoint, or preserved/reoriented/collapse "
            "classification is produced by this runner."
        ),
        "population": {
            "families": list(FAMILIES),
            "pair_first": SOURCE_PAIR_FIRST,
            "pair_last": SOURCE_PAIR_LAST,
            "rows_per_family": SOURCE_PAIR_COUNT_PER_FAMILY,
            "total_rows": TOTAL_ROW_COUNT,
            "cell": CELL,
            "anchor_name": ANCHOR_NAME,
            "target_offset": TARGET_OFFSET,
        },
        "transport": {
            "source_block": SOURCE_BLOCK,
            "target_block": TARGET_BLOCK,
            "source_tensor":
                "block35_mixer_in_proj_output_content_half",
            "target_tensor":
                "block36_mixer_in_proj_output_content_half",
            "ambient_dimension": AMBIENT_DIM,
            "source_plane": SOURCE_PLANE,
            "target_plane": TARGET_PLANE,
            "epsilon": EPSILON,
            "estimator": "authenticated_fast_cuda_symmetric_central_difference",
        },
        "families": family_summary,
        "pooled": pooled_summary,
        "worker_meta": worker_meta,
        "accounting": {
            "scientific_full_model_forward_count": TOTAL_ROW_COUNT,
            "scientific_local_map_perturbed_forward_count":
                TOTAL_ROW_COUNT * len(BASIS_NAMES) * 2,
            "finite_difference_direction_count":
                TOTAL_ROW_COUNT * len(BASIS_NAMES),
            "xg1_response_forward_count": 0,
            "p_value_count_added": 0,
        },
        "boundary": {
            "xg1_experiment5_response_accessed": False,
            "population_transport_executed": True,
            "adjacent_p5_geometry_accessed": True,
            "principal_angles_computed": True,
            "projector_overlap_computed": True,
            "procrustes_alignment_computed": True,
            "statistical_testing_performed": False,
            "p_value_count_added": 0,
            "interpretation_threshold_applied": False,
            "preserved_reoriented_collapse_label_emitted": False,
            "training_executed": False,
            "parameter_update_executed": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    items_bytes = jsonl_bytes(all_rows)
    summary_bytes = pretty_json_bytes(summary)

    (output_dir / ITEMS_FILE).write_bytes(items_bytes)
    (output_dir / SUMMARY_FILE).write_bytes(summary_bytes)

    file_hashes = {
        ITEMS_FILE: sha256_bytes(items_bytes),
        SUMMARY_FILE: sha256_bytes(summary_bytes),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_program_ancestor": REQUIRED_PROGRAM_ANCESTOR,
        "reference_fd_evidence": {
            "report_path": REFERENCE_FD_REPORT.as_posix(),
            "report_git_blob": REFERENCE_FD_REPORT_GIT_BLOB,
            "sums_path": REFERENCE_FD_SUMS.as_posix(),
            "sums_git_blob": REFERENCE_FD_SUMS_GIT_BLOB,
            "result":
                "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REFERENCE_FD_EQUIVALENCE",
            "epsilon": EPSILON,
        },
        "plane_artifacts": {
            "canonical_p5": {
                "strong_indices_path": direct.CANONICAL_STRONG.as_posix(),
                "plus_path": direct.CANONICAL_P5_PLUS.as_posix(),
                "minus_path": direct.CANONICAL_P5_MINUS.as_posix(),
            },
            "adjacent_p5": {
                "strong_indices_path": ADJACENT_STRONG.as_posix(),
                "plus_path": ADJACENT_P5_PLUS.as_posix(),
                "minus_path": ADJACENT_P5_MINUS.as_posix(),
            },
        },
        "output_file_sha256": dict(sorted(file_hashes.items())),
        "accounting": dict(summary["accounting"]),
        "boundary": dict(summary["boundary"]),
    }
    manifest_bytes = pretty_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_bytes)
    file_hashes[MANIFEST_FILE] = sha256_bytes(manifest_bytes)

    sums = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(file_hashes.items())
    )
    (output_dir / SUMS_FILE).write_text(
        sums,
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen Study-A Mamba-1.4B N=600 row-conditioned "
            "canonical-P5 block35-to-block36 transport measurement using the "
            "validated fixed-epsilon fast-CUDA finite-difference estimator. "
            "Descriptive geometry only; no p-values or response endpoints."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / geom.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_population(
        expected_head=str(args.expected_head),
        snapshot=args.snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(summary["result"]))
    print("STUDY=A_CROSS_LAYER_CAUSAL_TRANSPORT")
    print("FAMILIES=xg2,xg4")
    print("ROWS_PER_FAMILY=300")
    print("TOTAL_ROWS=600")
    print("CELL=C2_NAME")
    print("ANCHOR=A_IDENTITY")
    print("TARGET_OFFSET=2")
    print("SOURCE_BLOCK=35")
    print("TARGET_BLOCK=36")
    print("SOURCE_PLANE=P5")
    print("TARGET_PLANE=P5")
    print("EPSILON=0.025")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=600")
    print("SCIENTIFIC_LOCAL_MAP_PERTURBED_FORWARD_COUNT=2400")
    print("XG1_EXPERIMENT5_RESPONSE_ACCESSED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("INTERPRETATION_THRESHOLD_APPLIED=False")
    print("SCIENTIFIC_INTERPRETATION_EMITTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
