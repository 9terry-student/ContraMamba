#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import tempfile
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts import (
    reason_router_gen4_mamba370m_confirmation_fast_cuda
    as confirmation,
)
from scripts import (
    reason_router_gen4_mamba370m_discovery_fast_cuda
    as discovery,
)
from scripts import (
    reason_router_gen4_mamba370m_geometry_prepare_fast_cuda
    as geom,
)
from scripts import (
    reason_router_gen4_mamba370m_residual_decomposition_fast_cuda
    as decomposition,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

REQUIRED_EVIDENCE_FREEZE_COMMIT = (
    "b92ed1aded48de898cfac9cc85da373cc675b484"
)

PAIR_FIRST = 3601
PAIR_LAST = 3900
PAIR_COUNT = 300
PAIR_IDS = tuple(
    f"xg1_fact_{i:03d}"
    for i in range(PAIR_FIRST, PAIR_LAST + 1)
)

RESIDUAL_PLANES = ("P1", "P2", "P4", "P5")
BRANCH_ROLES = ("tp", "tm")
BRANCH_CELLS = {
    "tp": geom.TARGET_PLUS_CELL,
    "tm": geom.TARGET_MINUS_CELL,
}

FORWARDS_PER_PAIR = 2
TOTAL_FORWARD_BUDGET = PAIR_COUNT * FORWARDS_PER_PAIR
GPU_COUNT = 2
PAIRS_PER_SHARD = PAIR_COUNT // GPU_COUNT
FORWARDS_PER_SHARD = PAIRS_PER_SHARD * FORWARDS_PER_PAIR

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 150,
        "pair_first": "xg1_fact_3601",
        "pair_last": "xg1_fact_3750",
        "pair_count": 150,
        "forward_budget": 300,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 150,
        "end_index": 300,
        "pair_first": "xg1_fact_3751",
        "pair_last": "xg1_fact_3900",
        "pair_count": 150,
        "forward_budget": 300,
    },
)

DECOMP_ROOT = Path(
    "reports/reason_router_gen4_mamba370m_residual_decomposition_runs/"
    "g4k-mamba370-residual-decomp-xg1-3601-3900-2gpu-f42c94f"
)
DECOMP_ITEMS = DECOMP_ROOT / "residual_decomposition_items.jsonl"
DECOMP_ITEMS_SHA256 = (
    "89c9fa60efbb90a4c6fd993d5f71dbb7e10ee1c1f5a52932172566acd75b97e9"
)
DECOMP_SUMMARY = DECOMP_ROOT / "residual_decomposition_summary.json"
DECOMP_SUMMARY_SHA256 = (
    "4d9d9be6735356ee80b151db6f43bcbdb768e8c283a3e6fe41c3ae4588ff3c6d"
)

ITEM_FILE = "native_coefficient_census_items.jsonl"
SUMMARY_FILE = "native_coefficient_census_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-mamba370m-native-coefficient-census-item-v1"
SUMMARY_SCHEMA = "gen4-mamba370m-native-coefficient-census-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba370m-native-coefficient-census-manifest-v1"
RESULT_PASS = "PASS_MAMBA370M_NATIVE_COEFFICIENT_CENSUS_RAW_OBSERVATION"

TOL = 1.0e-12


class NativeCoefficientCensusError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise NativeCoefficientCensusError(message)


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


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeCoefficientCensusError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")

    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(status == "", "WORKTREE_NOT_CLEAN")

    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_EVIDENCE_FREEZE_COMMIT,
            head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "EVIDENCE_FREEZE_NOT_ANCESTOR")


def validate_protocol() -> None:
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(RESIDUAL_PLANES == ("P1", "P2", "P4", "P5"), "PLANES")
    require(FORWARDS_PER_PAIR == 2, "FORWARDS_PER_PAIR")
    require(TOTAL_FORWARD_BUDGET == 600, "TOTAL_FORWARD_BUDGET")
    require(FORWARDS_PER_SHARD == 300, "FORWARDS_PER_SHARD")
    require(len(SHARDS) == 2, "SHARD_COUNT")

    covered = []
    for expected_id, shard in enumerate(SHARDS):
        require(shard["shard_id"] == expected_id, "SHARD_ID")
        require(shard["physical_device"] == expected_id, "SHARD_DEVICE")
        require(
            shard["end_index"] - shard["start_index"]
            == shard["pair_count"],
            "SHARD_PAIR_COUNT",
        )
        require(
            shard["pair_count"] * FORWARDS_PER_PAIR
            == shard["forward_budget"],
            "SHARD_FORWARD_BUDGET",
        )
        covered.extend(range(shard["start_index"], shard["end_index"]))

    require(covered == list(range(PAIR_COUNT)), "SHARD_COVERAGE")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        result.append(value)
    return result


def load_frozen_effects() -> dict[str, dict[str, float]]:
    require(DECOMP_ITEMS.is_file(), "DECOMP_ITEMS_MISSING")
    require(DECOMP_SUMMARY.is_file(), "DECOMP_SUMMARY_MISSING")
    require(
        sha256_file(DECOMP_ITEMS) == DECOMP_ITEMS_SHA256,
        "DECOMP_ITEMS_SHA",
    )
    require(
        sha256_file(DECOMP_SUMMARY) == DECOMP_SUMMARY_SHA256,
        "DECOMP_SUMMARY_SHA",
    )

    rows = _read_jsonl(DECOMP_ITEMS)
    require(len(rows) == PAIR_COUNT, "DECOMP_ITEM_COUNT")
    require(
        [str(row["source_pair_id"]) for row in rows]
        == list(PAIR_IDS),
        "DECOMP_PAIR_ORDER",
    )

    out: dict[str, dict[str, float]] = {}
    for row in rows:
        pair = str(row["source_pair_id"])
        out[pair] = {
            plane: float(row[f"S_{plane}"])
            for plane in RESIDUAL_PLANES
        }
        require(
            all(math.isfinite(v) for v in out[pair].values()),
            f"DECOMP_EFFECT_NONFINITE:{pair}",
        )

    summary = json.loads(
        DECOMP_SUMMARY.read_text(encoding="utf-8-sig")
    )
    require(
        summary["result"]
        == decomposition.RESULT_PASS,
        "DECOMP_SUMMARY_RESULT",
    )
    require(
        summary["formal_inference_performed"] is False,
        "DECOMP_SUMMARY_INFERENCE",
    )
    require(
        int(summary["p_value_count"]) == 0,
        "DECOMP_SUMMARY_PVALUES",
    )

    return out


class ForwardBudget:
    def __init__(self, expected: int) -> None:
        self.expected = int(expected)
        self.count = 0

    def consume(self) -> None:
        require(self.count < self.expected, "FORWARD_BUDGET_OVERFLOW")
        self.count += 1

    def assert_exact(self) -> None:
        require(
            self.count == self.expected,
            f"FORWARD_BUDGET:{self.count}:{self.expected}",
        )


def capture_native_coefficients(
    *,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    input_ids: torch.Tensor,
    token_index: int,
    device: torch.device,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    budget: ForwardBudget,
) -> dict[str, Any]:
    mixer = runtime_ctx["intervention_mixer"]
    strong_mask = (
        runtime_ctx["strong_mask"]
        .detach().cpu().bool().contiguous()
    )

    require(
        strong_mask.numel() == geom.INTERMEDIATE_SIZE,
        "STRONG_MASK_SIZE",
    )
    require(
        int(strong_mask.sum().item()) == decomposition.DIM,
        "STRONG_MASK_COUNT",
    )

    captured: list[torch.Tensor] = []
    fast_calls = {"count": 0}

    def hook(_module, _args, output):
        require(
            torch.is_tensor(output)
            and output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * geom.INTERMEDIATE_SIZE,
            "INPROJ_SHAPE",
        )
        require(0 <= token_index < output.shape[1], "TOKEN_INDEX")
        require(len(captured) == 0, "HOOK_DUPLICATE")

        mask_device = strong_mask.to(output.device)
        h = (
            output[
                0,
                token_index,
                :geom.INTERMEDIATE_SIZE,
            ][mask_device]
            .detach().cpu().to(torch.float64).contiguous()
        )
        require(
            tuple(h.shape) == (decomposition.DIM,),
            "CAPTURE_SHAPE",
        )
        require(
            bool(torch.isfinite(h).all().item()),
            "CAPTURE_NONFINITE",
        )
        captured.append(h)
        return output

    original_cuda = mixer.cuda_kernels_forward

    def cuda_wrapper(
        _self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
    ):
        fast_calls["count"] += 1
        return original_cuda(
            hidden_states,
            cache_params,
            cache_position,
            attention_mask,
        )

    handle = mixer.in_proj.register_forward_hook(hook)
    mixer.cuda_kernels_forward = types.MethodType(
        cuda_wrapper,
        mixer,
    )

    budget.consume()

    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().to(device).contiguous()
            )
        torch.cuda.synchronize(device)
    finally:
        handle.remove()
        if "cuda_kernels_forward" in mixer.__dict__:
            del mixer.__dict__["cuda_kernels_forward"]

    require(
        fast_calls["count"] == 1,
        f"CUDA_FAST_PATH_CALLS:{fast_calls['count']}",
    )
    require(len(captured) == 1, "CAPTURE_COUNT")

    h = captured[0]

    coeffs = {}
    for plane in RESIDUAL_PLANES:
        plus = planes[plane]["plus"]
        minus = planes[plane]["minus"]
        a = float(torch.dot(h, plus).item())
        b = float(torch.dot(h, minus).item())
        energy = a * a + b * b
        require(
            all(math.isfinite(v) for v in (a, b, energy))
            and energy >= 0.0,
            f"COEFFICIENT_NONFINITE:{plane}",
        )
        coeffs[plane] = {
            "a": a,
            "b": b,
            "energy": energy,
            "l2": math.sqrt(energy),
        }

    return {
        "token_index": int(token_index),
        "coefficients": coeffs,
        "fast_path_calls": 1,
    }


def run_pair(
    *,
    pair_index: int,
    pair: str,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    planes: Mapping[str, Mapping[str, torch.Tensor]],
    encoded: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], int],
    events: Mapping[tuple[str, str, str], Mapping[str, Any]],
    frozen_effects: Mapping[str, Mapping[str, float]],
    device: torch.device,
    budget: ForwardBudget,
) -> dict[str, Any]:
    require(pair == PAIR_IDS[pair_index], f"PAIR:{pair_index}")
    anchors = confirmation.anchors_for_pair(pair, events)

    branches = {}
    for role in BRANCH_ROLES:
        branches[role] = capture_native_coefficients(
            model=model,
            runtime_ctx=runtime_ctx,
            input_ids=confirmation.input_row(
                encoded,
                lookup,
                pair,
                BRANCH_CELLS[role],
            ),
            token_index=int(anchors[role]) + geom.TARGET_OFFSET,
            device=device,
            planes=planes,
            budget=budget,
        )

    pair_energy = {}
    pair_l2 = {}
    for plane in RESIDUAL_PLANES:
        energies = [
            float(branches[role]["coefficients"][plane]["energy"])
            for role in BRANCH_ROLES
        ]
        value = math.fsum(energies) / len(energies)
        pair_energy[plane] = value
        pair_l2[plane] = math.sqrt(value)

    effects = {
        plane: float(frozen_effects[pair][plane])
        for plane in RESIDUAL_PLANES
    }

    require(
        all(math.isfinite(v) for v in pair_energy.values()),
        "PAIR_ENERGY_NONFINITE",
    )
    require(
        all(math.isfinite(v) for v in effects.values()),
        "PAIR_EFFECT_NONFINITE",
    )

    return {
        "schema_version": ITEM_SCHEMA,
        "source_pair_id": pair,
        "pair_index": pair_index,
        "branch_roles": list(BRANCH_ROLES),
        "branches": branches,
        "pair_mean_coefficient_energy": pair_energy,
        "pair_rms_coefficient_l2": pair_l2,
        "frozen_residual_effects": effects,
        "scientific_model_forward_count_this_run": FORWARDS_PER_PAIR,
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "scientific_conclusion_established": False,
        "rescue_performed": False,
    }


def descriptive(x: np.ndarray) -> dict[str, float]:
    require(x.shape == (PAIR_COUNT,), f"DESC_SHAPE:{x.shape}")
    require(bool(np.isfinite(x).all()), "DESC_NONFINITE")
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    require(a.shape == b.shape == (PAIR_COUNT,), "CORR_SHAPE")
    require(
        bool(np.isfinite(a).all()) and bool(np.isfinite(b).all()),
        "CORR_NONFINITE",
    )
    require(
        float(np.std(a, ddof=1)) > 0.0
        and float(np.std(b, ddof=1)) > 0.0,
        "CORR_ZERO_SD",
    )
    value = float(np.corrcoef(a, b)[0, 1])
    require(math.isfinite(value), "CORR_RESULT_NONFINITE")
    return value


def summarize_items(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(items) == PAIR_COUNT, "SUMMARY_ITEM_COUNT")

    energy = {
        plane: np.asarray(
            [
                float(item["pair_mean_coefficient_energy"][plane])
                for item in items
            ],
            dtype=np.float64,
        )
        for plane in RESIDUAL_PLANES
    }
    effect = {
        plane: np.asarray(
            [
                float(item["frozen_residual_effects"][plane])
                for item in items
            ],
            dtype=np.float64,
        )
        for plane in RESIDUAL_PLANES
    }

    mean_energy = np.asarray(
        [float(np.mean(energy[p])) for p in RESIDUAL_PLANES],
        dtype=np.float64,
    )
    total_mean_energy = float(np.sum(mean_energy))
    require(total_mean_energy > 0.0, "TOTAL_MEAN_ENERGY")

    energy_share = {
        plane: float(mean_energy[index] / total_mean_energy)
        for index, plane in enumerate(RESIDUAL_PLANES)
    }

    largest_energy = Counter()
    largest_abs_effect = Counter()
    same_winner = 0

    for i in range(PAIR_COUNT):
        energy_winner = max(
            RESIDUAL_PLANES,
            key=lambda p: float(energy[p][i]),
        )
        effect_winner = max(
            RESIDUAL_PLANES,
            key=lambda p: abs(float(effect[p][i])),
        )
        largest_energy[energy_winner] += 1
        largest_abs_effect[effect_winner] += 1
        if energy_winner == effect_winner:
            same_winner += 1

    plane_summary = {}
    for plane in RESIDUAL_PLANES:
        mean_e = float(np.mean(energy[plane]))
        mean_s = float(np.mean(effect[plane]))
        require(mean_e > 0.0, f"MEAN_ENERGY_ZERO:{plane}")

        plane_summary[plane] = {
            "coefficient_energy": descriptive(energy[plane]),
            "mean_coefficient_l2": float(
                np.mean(np.sqrt(energy[plane]))
            ),
            "energy_share_of_residual_mean_total":
                energy_share[plane],
            "frozen_effect_mean": mean_s,
            "mean_effect_over_mean_energy":
                mean_s / mean_e,
            "corr_energy_effect":
                safe_corr(energy[plane], effect[plane]),
            "corr_energy_abs_effect":
                safe_corr(energy[plane], np.abs(effect[plane])),
            "largest_energy_count":
                int(largest_energy[plane]),
            "largest_energy_fraction":
                float(largest_energy[plane] / PAIR_COUNT),
            "largest_abs_effect_count":
                int(largest_abs_effect[plane]),
            "largest_abs_effect_fraction":
                float(largest_abs_effect[plane] / PAIR_COUNT),
        }

    return {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "source_pair_count": PAIR_COUNT,
        "pair_id_first": PAIR_IDS[0],
        "pair_id_last": PAIR_IDS[-1],
        "residual_planes": list(RESIDUAL_PLANES),
        "plane_summary": plane_summary,
        "largest_energy_and_largest_abs_effect_same_plane": {
            "count": int(same_winner),
            "fraction": float(same_winner / PAIR_COUNT),
        },
        "total_mean_residual_coefficient_energy":
            total_mean_energy,
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "scientific_conclusion_established": False,
        "rescue_performed": False,
    }


def _worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "items": temp_dir / f"shard_{shard_id}_items.jsonl",
        "meta": temp_dir / f"shard_{shard_id}_meta.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    shard: Mapping[str, Any],
    expected_head: str,
    model_snapshot: str,
    compact_checkpoint: str,
    temp_dir: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = _worker_paths(Path(temp_dir), shard_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

        validate_protocol()
        authenticate_repo(expected_head)

        device = discovery.runtime_gate_single_visible_gpu(
            physical_device
        )

        snapshot = Path(model_snapshot)
        checkpoint = Path(compact_checkpoint)

        frozen = decomposition.confirmation.load_frozen_geometry()
        rows, encoded, events = decomposition.build_input_state(snapshot)
        lookup = confirmation.row_index(rows)
        frozen_effects = load_frozen_effects()

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        decomposition.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )

        runtime_ctx = geom.runtime_components(model)
        confirmation.validate_runtime_geometry(
            runtime_ctx,
            frozen,
        )

        budget = ForwardBudget(int(shard["forward_budget"]))
        items = []

        for global_index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            pair = PAIR_IDS[global_index]
            items.append(
                run_pair(
                    pair_index=global_index,
                    pair=pair,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    planes=frozen["planes"],
                    encoded=encoded,
                    lookup=lookup,
                    events=events,
                    frozen_effects=frozen_effects,
                    device=device,
                    budget=budget,
                )
            )

        budget.assert_exact()
        torch.cuda.synchronize(device)

        require(
            len(items) == int(shard["pair_count"]),
            "WORKER_ITEM_COUNT",
        )

        paths["items"].write_bytes(jsonl_bytes(items))
        meta = {
            "schema_version":
                "gen4-mamba370m-native-coefficient-census-worker-v1",
            "shard_id": shard_id,
            "physical_device": physical_device,
            "logical_device": 0,
            "cuda_visible_devices": str(physical_device),
            "device_name": torch.cuda.get_device_name(0),
            "pair_first": shard["pair_first"],
            "pair_last": shard["pair_last"],
            "pair_count": len(items),
            "scientific_model_forward_count":
                int(shard["forward_budget"]),
            "items_sha256": sha256_file(paths["items"]),
            "model_provenance": model_provenance,
            "decomposition_items_sha256": DECOMP_ITEMS_SHA256,
            "decomposition_summary_sha256": DECOMP_SUMMARY_SHA256,
            "formal_inference_performed": False,
            "p_value_count": 0,
            "selection_performed": False,
            "rescue_performed": False,
        }
        paths["meta"].write_bytes(pretty_json_bytes(meta))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise


def merge_workers(
    temp_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = []
    metas = []

    for shard in SHARDS:
        shard_id = int(shard["shard_id"])
        paths = _worker_paths(temp_dir, shard_id)

        require(paths["items"].is_file(), f"ITEMS_MISSING:{shard_id}")
        require(paths["meta"].is_file(), f"META_MISSING:{shard_id}")

        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        require(meta["shard_id"] == shard_id, "MERGE_SHARD_ID")
        require(
            meta["physical_device"] == shard["physical_device"],
            "MERGE_DEVICE",
        )
        require(meta["logical_device"] == 0, "MERGE_LOGICAL_DEVICE")
        require(
            meta["pair_count"] == shard["pair_count"],
            "MERGE_PAIR_COUNT",
        )
        require(
            meta["scientific_model_forward_count"]
            == shard["forward_budget"],
            "MERGE_FORWARD_COUNT",
        )
        require(
            meta["items_sha256"] == sha256_file(paths["items"]),
            "MERGE_ITEMS_SHA",
        )

        shard_items = _read_jsonl(paths["items"])
        require(
            len(shard_items) == shard["pair_count"],
            "MERGE_SHARD_ITEMS",
        )

        items.extend(shard_items)
        metas.append(meta)

    require(len(items) == PAIR_COUNT, "MERGED_ITEM_COUNT")
    require(
        [str(item["source_pair_id"]) for item in items]
        == list(PAIR_IDS),
        "MERGED_PAIR_ORDER",
    )
    require(
        sum(
            int(item["scientific_model_forward_count_this_run"])
            for item in items
        )
        == TOTAL_FORWARD_BUDGET,
        "MERGED_FORWARD_SUM",
    )

    return items, metas


def write_outputs(
    *,
    output_dir: Path,
    expected_head: str,
    items: Sequence[Mapping[str, Any]],
    worker_meta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    summary = summarize_items(items)

    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / ITEM_FILE).write_bytes(jsonl_bytes(items))
    (output_dir / SUMMARY_FILE).write_bytes(pretty_json_bytes(summary))

    hashes = {
        ITEM_FILE: sha256_file(output_dir / ITEM_FILE),
        SUMMARY_FILE: sha256_file(output_dir / SUMMARY_FILE),
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_evidence_freeze_commit":
            REQUIRED_EVIDENCE_FREEZE_COMMIT,
        "decomposition_items_sha256": DECOMP_ITEMS_SHA256,
        "decomposition_summary_sha256": DECOMP_SUMMARY_SHA256,
        "pair_first": PAIR_IDS[0],
        "pair_last": PAIR_IDS[-1],
        "source_pair_count": PAIR_COUNT,
        "residual_planes": list(RESIDUAL_PLANES),
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "worker_count": GPU_COUNT,
        "workers": list(worker_meta),
        "scientific_question":
            "Is the dominant negative P5/P2 residual effect in Mamba-370M "
            "explained by native coefficient mass, or by stronger/opposite "
            "effect coupling per unit plane coefficient energy?",
        "formal_inference_performed": False,
        "p_value_count": 0,
        "selection_performed": False,
        "scientific_conclusion_established": False,
        "rescue_of_failed_cross_backbone_claim": False,
        "training_executed": False,
        "backward_executed": False,
        "output_file_sha256": dict(sorted(hashes.items())),
    }

    (output_dir / MANIFEST_FILE).write_bytes(pretty_json_bytes(manifest))
    hashes[MANIFEST_FILE] = sha256_file(output_dir / MANIFEST_FILE)

    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    return summary


def run_census(
    *,
    expected_head: str,
    model_snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    load_frozen_effects()
    decomposition.load_population()
    geom.validate_snapshot(model_snapshot)

    require(
        compact_checkpoint.resolve()
        == (ROOT / geom.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        sha256_file(compact_checkpoint)
        == geom.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba370m_native_coeff_census_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard": shard,
                    "expected_head": expected_head,
                    "model_snapshot": str(model_snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"mamba370-native-coeff-census-{shard['shard_id']}",
            )
            process.start()
            processes.append(process)

        for shard, process in zip(SHARDS, processes, strict=True):
            process.join()
            if process.exitcode != 0:
                paths = _worker_paths(
                    temp_dir,
                    int(shard["shard_id"]),
                )
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise NativeCoefficientCensusError(
                    f"WORKER_FAILED:{shard['shard_id']}:\n{detail}"
                )

        items, metas = merge_workers(temp_dir)

        return write_outputs(
            output_dir=output_dir,
            expected_head=expected_head,
            items=items,
            worker_meta=metas,
        )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mamba-370M native residual-plane coefficient census on the "
            "already-frozen XG1 3601..3900 decomposition cohort. "
            "Descriptive only: 600 forwards, no p-values, no selection."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / geom.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    summary = run_census(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT=" + summary["result"])

    for plane in RESIDUAL_PLANES:
        row = summary["plane_summary"][plane]
        print(
            f"{plane}_MEAN_COEFF_ENERGY="
            + format(
                float(row["coefficient_energy"]["mean"]),
                ".17g",
            )
        )
        print(
            f"{plane}_ENERGY_SHARE="
            + format(
                float(row["energy_share_of_residual_mean_total"]),
                ".17g",
            )
        )
        print(
            f"{plane}_MEAN_EFFECT="
            + format(float(row["frozen_effect_mean"]), ".17g")
        )
        print(
            f"{plane}_MEAN_EFFECT_OVER_MEAN_ENERGY="
            + format(
                float(row["mean_effect_over_mean_energy"]),
                ".17g",
            )
        )
        print(
            f"{plane}_CORR_ENERGY_EFFECT="
            + format(float(row["corr_energy_effect"]), ".17g")
        )
        print(
            f"{plane}_CORR_ENERGY_ABS_EFFECT="
            + format(float(row["corr_energy_abs_effect"]), ".17g")
        )
        print(
            f"{plane}_LARGEST_ENERGY_FRACTION="
            + format(float(row["largest_energy_fraction"]), ".17g")
        )
        print(
            f"{plane}_LARGEST_ABS_EFFECT_FRACTION="
            + format(float(row["largest_abs_effect_fraction"]), ".17g")
        )

    same = summary[
        "largest_energy_and_largest_abs_effect_same_plane"
    ]
    print(
        "LARGEST_ENERGY_EFFECT_WINNER_SAME_FRACTION="
        + format(float(same["fraction"]), ".17g")
    )

    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN="
        + str(TOTAL_FORWARD_BUDGET)
    )
    print("FORMAL_INFERENCE_PERFORMED=False")
    print("P_VALUE_COUNT=0")
    print("SELECTION_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION_ESTABLISHED=False")
    print("RESCUE_PERFORMED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


if __name__ == "__main__":
    main()
