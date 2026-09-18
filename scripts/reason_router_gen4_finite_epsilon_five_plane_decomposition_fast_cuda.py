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
    reason_router_gen4_pp3_excluded_residual_aggregate_restoration_sufficiency_fast_cuda
    as base
)

ROOT = Path(__file__).resolve().parents[1]

N = 300
DIM = 395
K = 5
EPS = 0.025
TOL = 1.0e-12

PLANE_ORDER = ("P1", "P2", "P3", "P4", "P5")
DIRECTION_ORDER = tuple(
    direction
    for plane in PLANE_ORDER
    for direction in (f"{plane}_plus", f"{plane}_minus")
)

EXPECTED_EIGENVALUES = (
    0.87061814189182785,
    0.94755022112376275,
    0.98692852916688512,
    0.99848952673382474,
    0.99986792842854511,
)

F_SIGNED = 2
F_DIRECTION = 4
F_PAIR = 40
F_TOTAL = 12000
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
        "forward_budget": 6000,
    },
    {
        "shard_id": 1,
        "gpu_id": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_2551",
        "pair_last": "xg1_fact_2700",
        "pair_count": 150,
        "forward_budget": 6000,
    },
)

PRIOR_RUN_NAME = (
    "g4k-residual-aggregate-restoration-sufficiency-"
    "xg1-2401-2700-92efdd0-retry3"
)
PRIOR_ROOT = Path(
    "reports/"
    "reason_router_gen4_pp3_excluded_residual_aggregate_restoration_"
    "sufficiency_runs/"
) / PRIOR_RUN_NAME

PRIOR_ITEMS_SHA256 = (
    "9d5dabbef82a8fcfaccc4e610bbb4d91f1e7ee9ea2f2627a92f999c88034ac49"
)
PRIOR_SUMMARY_SHA256 = (
    "e38d8c3943594e934535ecf1682788868f815e28630e4ac44cc92864f44bd7d9"
)
PRIOR_MANIFEST_SHA256 = (
    "24efaa5b982ab8b6ce2b5c0796b1b6b46e9f1cdfe047a07aab5c8742eaadb064"
)
PRIOR_EXECUTION_HEAD = "92efdd06974f3db96937d07f8df00b8ca0fea6ff"

ITEM_FILE = "finite_epsilon_principal_decomposition_items.jsonl"
SUMMARY_FILE = "finite_epsilon_principal_decomposition_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-finite-epsilon-five-plane-decomposition-item-v1"
SUMMARY_SCHEMA = "gen4-finite-epsilon-five-plane-decomposition-summary-v1"
MANIFEST_SCHEMA = "gen4-finite-epsilon-five-plane-decomposition-manifest-v1"
RESULT_PASS = "PASS_FINITE_EPSILON_FIVE_PLANE_DECOMPOSITION_RAW_OBSERVATION"


class DecompositionError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise DecompositionError(message)


def expected_pairs() -> tuple[str, ...]:
    return tuple(f"xg1_fact_{index:03d}" for index in range(2401, 2701))


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
            "SHARD_BUDGET",
        )
        pair_slice = expected_pairs()[
            int(shard["start_index"]):int(shard["end_index"])
        ]
        require(
            len(pair_slice) == int(shard["pair_count"])
            and pair_slice[0] == shard["pair_first"]
            and pair_slice[-1] == shard["pair_last"],
            "SHARD_RANGE",
        )
        covered.extend(
            range(int(shard["start_index"]), int(shard["end_index"]))
        )
    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        sum(int(shard["forward_budget"]) for shard in SHARDS) == F_TOTAL,
        "TOTAL_BUDGET",
    )


def authenticate_repo(expected_head: str) -> None:
    base.authenticate_repo(expected_head)


def validate_prior_native_q0() -> list[float]:
    root = ROOT / PRIOR_ROOT
    require(root.is_dir(), f"PRIOR_ROOT_MISSING:{root}")

    items_path = root / (
        "pp3_excluded_residual_aggregate_restoration_sufficiency_items.jsonl"
    )
    summary_path = root / (
        "pp3_excluded_residual_aggregate_restoration_sufficiency_summary.json"
    )
    manifest_path = root / "artifact_manifest.json"

    require(
        base.sha256_file(items_path) == PRIOR_ITEMS_SHA256,
        "PRIOR_ITEMS_SHA",
    )
    require(
        base.sha256_file(summary_path) == PRIOR_SUMMARY_SHA256,
        "PRIOR_SUMMARY_SHA",
    )
    require(
        base.sha256_file(manifest_path) == PRIOR_MANIFEST_SHA256,
        "PRIOR_MANIFEST_SHA",
    )

    validated = base.validate_artifact(root)
    items = validated["items"]
    summary = validated["summary"]

    require(
        summary["execution_head"] == PRIOR_EXECUTION_HEAD,
        "PRIOR_EXECUTION_HEAD",
    )
    require(
        summary["scientific_model_forward_count_this_run"] == 36000
        and summary["primary_inference_executed"] is False
        and summary["scientific_conclusion"] is None,
        "PRIOR_RAW_BOUNDARY",
    )
    require(len(items) == N, "PRIOR_ITEM_COUNT")

    q0: list[float] = []
    for index, (pair, item) in enumerate(
        zip(expected_pairs(), items, strict=True)
    ):
        require(
            item["source_pair_id"] == pair
            and int(item["pair_index"]) == index,
            f"PRIOR_PAIR:{index}",
        )
        native = {
            row["condition"]: row
            for row in item["conditions"]
        }["native"]
        require(
            float(item["Q0"]) == float(native["Q"]),
            f"PRIOR_Q0_ID:{index}",
        )
        value = float(item["Q0"])
        require(math.isfinite(value), f"PRIOR_Q0_FINITE:{index}")
        q0.append(value)
    return q0


def principal_geometry() -> tuple[
    dict[str, torch.Tensor],
    tuple[float, ...],
]:
    planes = base.load_planes()
    family_bases = base.load_bases()

    b2 = family_bases["xg2"].to(torch.float64)
    b4 = family_bases["xg4"].to(torch.float64)
    contrast = b2 @ b2.T - b4 @ b4.T

    vectors = []
    eigenvalues = []

    plane_keys = {
        "P1": ("p1_plus", "p1_minus"),
        "P2": ("p2_plus", "p2_minus"),
        "P3": ("pp3_plus", "pp3_minus"),
        "P4": ("p4_plus", "p4_minus"),
        "P5": ("p5_plus", "p5_minus"),
    }

    for plane, expected in zip(
        PLANE_ORDER, EXPECTED_EIGENVALUES, strict=True
    ):
        plus_key, minus_key = plane_keys[plane]
        plus = planes[plus_key].to(torch.float64).contiguous()
        minus = planes[minus_key].to(torch.float64).contiguous()

        lambda_plus = float(torch.dot(plus, contrast @ plus))
        lambda_minus = -float(torch.dot(minus, contrast @ minus))

        plus_residual = float(
            torch.max(torch.abs(contrast @ plus - expected * plus))
        )
        minus_residual = float(
            torch.max(torch.abs(contrast @ minus + expected * minus))
        )

        require(
            abs(lambda_plus - expected) <= 2.0e-12,
            f"EIGENVALUE_PLUS:{plane}:{lambda_plus}",
        )
        require(
            abs(lambda_minus - expected) <= 2.0e-12,
            f"EIGENVALUE_MINUS:{plane}:{lambda_minus}",
        )
        require(
            plus_residual <= 2.0e-12,
            f"EIGENVECTOR_PLUS:{plane}:{plus_residual}",
        )
        require(
            minus_residual <= 2.0e-12,
            f"EIGENVECTOR_MINUS:{plane}:{minus_residual}",
        )

        vectors.extend([plus, minus])
        eigenvalues.append((lambda_plus + lambda_minus) / 2.0)

    matrix = torch.stack(vectors, dim=1)
    gram_residual = float(
        torch.max(
            torch.abs(
                matrix.T @ matrix
                - torch.eye(2 * K, dtype=torch.float64)
            )
        )
    )
    require(
        gram_residual <= TOL,
        f"PRINCIPAL_GRAM:{gram_residual}",
    )

    return planes, tuple(eigenvalues)


def principal_direction_map(
    planes: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "P1_plus": planes["p1_plus"],
        "P1_minus": planes["p1_minus"],
        "P2_plus": planes["p2_plus"],
        "P2_minus": planes["p2_minus"],
        "P3_plus": planes["pp3_plus"],
        "P3_minus": planes["pp3_minus"],
        "P4_plus": planes["p4_plus"],
        "P4_minus": planes["p4_minus"],
        "P5_plus": planes["p5_plus"],
        "P5_minus": planes["p5_minus"],
    }


def run_principal_direction(
    seed: Mapping[str, Any],
    direction: torch.Tensor,
    *,
    direction_key: str,
    planes: Mapping[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, Any]:
    positive = base.run_signed(
        seed,
        direction,
        condition="native",
        orientation=1,
        planes=planes,
        **kwargs,
    )
    negative = base.run_signed(
        seed,
        direction,
        condition="native",
        orientation=-1,
        planes=planes,
        **kwargs,
    )

    f_plus = float(positive["F"])
    f_minus = float(negative["F"])
    j_value = (f_plus - f_minus) / (2.0 * EPS)

    return {
        "direction_key": direction_key,
        "F_plus": f_plus,
        "F_minus": f_minus,
        "J": j_value,
        "J_squared": j_value * j_value,
        "positive_probe": positive,
        "negative_probe": negative,
        "model_forward_count": F_DIRECTION,
    }


def decomposition_from_j(
    j_by_direction: Mapping[str, float],
    eigenvalues: Sequence[float],
    q0: float,
) -> dict[str, Any]:
    require(
        tuple(j_by_direction.keys()) == DIRECTION_ORDER,
        "J_DIRECTION_ORDER",
    )
    require(len(eigenvalues) == K, "EIGENVALUE_COUNT")

    contributions: dict[str, float] = {}
    for plane_index, plane in enumerate(PLANE_ORDER):
        j_plus = float(j_by_direction[f"{plane}_plus"])
        j_minus = float(j_by_direction[f"{plane}_minus"])
        contribution = (
            float(eigenvalues[plane_index])
            * (j_plus * j_plus - j_minus * j_minus)
            / K
        )
        require(math.isfinite(contribution), f"CONTRIBUTION:{plane}")
        contributions[plane] = contribution

    q_principal = math.fsum(contributions.values())
    residual = float(q0) - q_principal
    abs_residual = abs(residual)
    relative = (
        residual / float(q0)
        if float(q0) != 0.0
        else None
    )
    abs_relative = (
        abs_residual / abs(float(q0))
        if float(q0) != 0.0
        else None
    )

    return {
        "plane_contributions": contributions,
        "Q_principal": q_principal,
        "reconstruction_residual": residual,
        "absolute_reconstruction_residual": abs_residual,
        "relative_reconstruction_residual_to_Q0": relative,
        "absolute_relative_reconstruction_residual_to_Q0": abs_relative,
    }


def run_pair(
    seed: Mapping[str, Any],
    *,
    planes: Mapping[str, torch.Tensor],
    eigenvalues: Sequence[float],
    q0: float,
    **kwargs: Any,
) -> dict[str, Any]:
    direction_map = principal_direction_map(planes)
    probes = [
        run_principal_direction(
            seed,
            direction_map[key],
            direction_key=key,
            planes=planes,
            **kwargs,
        )
        for key in DIRECTION_ORDER
    ]

    require(
        tuple(probe["direction_key"] for probe in probes)
        == DIRECTION_ORDER,
        "PROBE_ORDER",
    )

    j_by_direction = {
        probe["direction_key"]: float(probe["J"])
        for probe in probes
    }
    values = decomposition_from_j(
        j_by_direction,
        eigenvalues,
        q0,
    )

    return {
        **dict(seed),
        "schema_version": ITEM_SCHEMA,
        "epsilon": EPS,
        "plane_order": list(PLANE_ORDER),
        "principal_direction_order": list(DIRECTION_ORDER),
        "positive_contrast_eigenvalues": list(eigenvalues),
        "prior_native_q0": float(q0),
        "prior_native_q0_reused": True,
        "prior_raw_run_name": PRIOR_RUN_NAME,
        "prior_items_sha256": PRIOR_ITEMS_SHA256,
        "principal_direction_probes": probes,
        **values,
        "new_original_basis_scientific_model_forward_count": 0,
        "scientific_model_forward_count_this_run": F_PAIR,
        "baseline_model_forward_count_this_run": 0,
        "primary_inference_executed": False,
        "multiplicity_correction_executed": False,
        "scientific_conclusion": None,
    }


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
        and item["plane_order"] == list(PLANE_ORDER)
        and item["principal_direction_order"] == list(DIRECTION_ORDER),
        f"ITEM_META:{index}",
    )
    require(
        item["prior_native_q0_reused"] is True
        and item["prior_raw_run_name"] == PRIOR_RUN_NAME
        and item["prior_items_sha256"] == PRIOR_ITEMS_SHA256
        and float(item["prior_native_q0"]) == float(expected_q0),
        f"PRIOR_Q0:{index}",
    )
    require(
        item["scientific_model_forward_count_this_run"] == F_PAIR
        and item["new_original_basis_scientific_model_forward_count"] == 0
        and item["baseline_model_forward_count_this_run"] == 0,
        f"BUDGET:{index}",
    )
    require(
        item["primary_inference_executed"] is False
        and item["multiplicity_correction_executed"] is False
        and item["scientific_conclusion"] is None,
        f"BOUNDARY:{index}",
    )

    probes = item["principal_direction_probes"]
    require(len(probes) == 2 * K, f"PROBE_COUNT:{index}")
    require(
        tuple(probe["direction_key"] for probe in probes)
        == DIRECTION_ORDER,
        f"PROBE_ORDER:{index}",
    )
    require(
        all(int(probe["model_forward_count"]) == F_DIRECTION for probe in probes),
        f"PROBE_BUDGET:{index}",
    )

    j_by_direction = {
        probe["direction_key"]: float(probe["J"])
        for probe in probes
    }
    expected = decomposition_from_j(
        j_by_direction,
        [float(x) for x in item["positive_contrast_eigenvalues"]],
        float(expected_q0),
    )

    require(
        item["plane_contributions"] == expected["plane_contributions"],
        f"CONTRIBUTIONS:{index}",
    )
    for key in (
        "Q_principal",
        "reconstruction_residual",
        "absolute_reconstruction_residual",
        "relative_reconstruction_residual_to_Q0",
        "absolute_relative_reconstruction_residual_to_Q0",
    ):
        require(item[key] == expected[key], f"RECON:{key}:{index}")


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
    prior_q0: Sequence[float],
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
        planes, eigenvalues = principal_geometry()

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
    prior_q0 = validate_prior_native_q0()

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
        summary["epsilon"] == EPS
        and summary["plane_order"] == list(PLANE_ORDER)
        and summary["principal_direction_order"] == list(DIRECTION_ORDER),
        "SUMMARY_GEOMETRY",
    )
    require(
        summary["prior_native_q0_reused"] is True
        and summary["prior_raw_run_name"] == PRIOR_RUN_NAME
        and summary["prior_items_sha256"] == PRIOR_ITEMS_SHA256,
        "SUMMARY_PRIOR",
    )
    require(
        summary["scientific_model_forward_count_this_run"] == F_TOTAL
        and summary["new_original_basis_scientific_model_forward_count"] == 0
        and summary["baseline_model_forward_count_this_run"] == 0,
        "SUMMARY_BUDGET",
    )
    require(
        summary["gpu_count"] == GPU_COUNT
        and len(summary["shards"]) == GPU_COUNT,
        "SUMMARY_GPU",
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
    _, eigenvalues = principal_geometry()
    prior_q0 = validate_prior_native_q0()

    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(
        torch.cuda.device_count() >= GPU_COUNT,
        "CUDA_DEVICE_COUNT",
    )

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="gen4_finite_epsilon_five_plane_decomposition_"
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
                name=f"gen4-principal-decomposition-gpu{shard['gpu_id']}",
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
        items, shard_meta = merge_shards(payloads, prior_q0)

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
        "principal_direction_order": list(DIRECTION_ORDER),
        "positive_contrast_eigenvalues": list(eigenvalues),
        "prior_native_q0_reused": True,
        "prior_raw_run_name": PRIOR_RUN_NAME,
        "prior_execution_head": PRIOR_EXECUTION_HEAD,
        "prior_items_sha256": PRIOR_ITEMS_SHA256,
        "model_forwards_per_direction": F_DIRECTION,
        "model_forwards_per_pair": F_PAIR,
        "scientific_model_forward_count_this_run": F_TOTAL,
        "new_original_basis_scientific_model_forward_count": 0,
        "baseline_model_forward_count_this_run": 0,
        "gpu_count": GPU_COUNT,
        "parallelization": "independent_pair_shards_spawn",
        "shards": shard_meta,
        "representative_checkpoint_sha256": checkpoint_sha,
        "primary_inference_executed": False,
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
            "Direct finite-epsilon P1..P5 principal-direction decomposition "
            "on XG1 2401..2700 using frozen native Q0 evidence."
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
    print("PRIOR_NATIVE_Q0_REUSED=True")
    print("NEW_ORIGINAL_BASIS_SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


if __name__ == "__main__":
    main()
