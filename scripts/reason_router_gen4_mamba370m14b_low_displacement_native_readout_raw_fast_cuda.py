#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_mamba370m14b_low_displacement_behavioral_raw_fast_cuda
    as lowdisp,
)
from scripts import (
    reason_router_gen4_mamba370m14b_readout_alignment_raw_fast_cuda
    as study_b,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_BASE_COMMIT = "1f9c42cd502bcf204c05baa2c6cbbc7e9274bd7c"

SCALES = ("mamba370m", "mamba14b")
SCALE_TO_PHYSICAL_GPU = {
    "mamba370m": 0,
    "mamba14b": 1,
}
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
PAIR_FIRST = 5401
PAIR_LAST = 5700
PAIR_COUNT = 300
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(PAIR_FIRST, PAIR_LAST + 1))
ROWS_PER_SCALE = PAIR_COUNT * len(TARGET_CELLS)
TOTAL_ROWS = len(SCALES) * ROWS_PER_SCALE

EXPECTED_SELECTED = {
    "mamba370m": "P3",
    "mamba14b": "P5",
}
EXPECTED_CONTROL = {
    "mamba370m": "P5",
    "mamba14b": "P4",
}
EXPECTED_CHECKPOINT = {
    "mamba370m":
        "9d8e3db22af4636938679aac6a8a97dd45344937d434fab29eac2ddc41a52a72",
    "mamba14b":
        "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a",
}

STRUCTURAL_ROW_SHA256 = (
    "1d0f21ba44b4a282f42edb7e56626bb7d522ce09e25e1611bf1424ead4da2b28"
)
STRUCTURAL_SOURCE_SHA256 = (
    "eddd6a264130e6451c72aeab010758dac43de82d9521898dfd1489c86717d11a"
)
TOKEN_GATE_CROSS_SCALE_SHA256 = (
    "6657df0844a4fc8f41488b6fe119322ffa99583c83c7fdb4de683e9f7bb20b00"
)

ITEM_FILE = "low_displacement_native_readout_items.jsonl"
SUMMARY_FILE = "raw_readout_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ITEM_SCHEMA = "gen4-lowdisp-native-functional-faithfulness-readout-row-v1"
SUMMARY_SCHEMA = "gen4-lowdisp-native-functional-faithfulness-readout-summary-v1"
MANIFEST_SCHEMA = "gen4-lowdisp-native-functional-faithfulness-readout-manifest-v1"

GATE_RESULT = "PASS_GEN4_LOWDISP_NATIVE_FUNCTIONAL_FAITHFULNESS_TECHNICAL_GATE"
RAW_RESULT = "PASS_GEN4_LOWDISP_NATIVE_FUNCTIONAL_FAITHFULNESS_READOUT_RAW"


class LowDispReadoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDispReadoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDispReadoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_BASE_COMMIT, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "BASE_COMMIT_NOT_ANCESTOR")


def validate_protocol() -> None:
    require(PAIR_FIRST == 5401 and PAIR_LAST == 5700, "PAIR_RANGE")
    require(PAIR_COUNT == 300, "PAIR_COUNT")
    require(PAIR_IDS == tuple(f"xg1_fact_{i}" for i in range(5401, 5701)), "PAIR_IDS")
    require(TARGET_CELLS == ("C0_SHAM", "C2_NAME"), "TARGET_CELLS")
    require(ROWS_PER_SCALE == 600 and TOTAL_ROWS == 1200, "ROW_COUNTS")
    require(
        SCALE_TO_PHYSICAL_GPU == {
            "mamba370m": 0,
            "mamba14b": 1,
        },
        "GPU_MAPPING",
    )

    for scale in SCALES:
        spec = lowdisp.scale_spec(scale)
        require(
            str(spec["selected_plane"]) == EXPECTED_SELECTED[scale],
            f"SELECTED:{scale}",
        )
        require(
            str(spec["control_plane"]) == EXPECTED_CONTROL[scale],
            f"CONTROL:{scale}",
        )
        require(
            str(spec["checkpoint_sha256"]) == EXPECTED_CHECKPOINT[scale],
            f"CHECKPOINT:{scale}",
        )

    structural = ROOT / lowdisp.holdout.OUTPUT_DIR
    require(
        sha256_file(structural / lowdisp.holdout.ROW_FILE)
        == STRUCTURAL_ROW_SHA256,
        "STRUCTURAL_ROW_SHA",
    )
    require(
        sha256_file(structural / lowdisp.holdout.SOURCE_FILE)
        == STRUCTURAL_SOURCE_SHA256,
        "STRUCTURAL_SOURCE_SHA",
    )

    token_summary = ROOT / (
        "reports/reason_router_gen4_mamba370m14b_"
        "low_displacement_tokenizer_anchor_eligibility_v1/"
        "cross_scale_summary.json"
    )
    require(
        sha256_file(token_summary) == TOKEN_GATE_CROSS_SCALE_SHA256,
        "TOKEN_GATE_SUMMARY_SHA",
    )
    gate = json.loads(token_summary.read_text(encoding="utf-8"))
    require(
        gate["result"]
        == "PASS_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_ELIGIBILITY",
        "TOKEN_GATE_RESULT",
    )
    require(gate["scale_results"]["mamba370m"] == "PASS_600_OF_600", "TOKEN_370")
    require(gate["scale_results"]["mamba14b"] == "PASS_600_OF_600", "TOKEN_14")


def worker_paths(temp_dir: Path, scale: str) -> dict[str, Path]:
    return {
        "items": temp_dir / f"{scale}_items.jsonl",
        "meta": temp_dir / f"{scale}_meta.json",
        "gate": temp_dir / f"{scale}_gate.json",
        "error": temp_dir / f"{scale}.error.txt",
    }


def prepare_scale(
    *,
    scale: str,
    model_snapshot: Path,
    physical_device: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    spec = lowdisp.scale_spec(scale)
    confirmation = spec["confirmation"]
    geom = spec["geom"]

    device = confirmation.runtime_gate_single_visible_gpu(physical_device)
    snapshot = Path(model_snapshot)
    checkpoint = ROOT / spec["checkpoint_rel"]

    geom.validate_snapshot(snapshot)
    require(
        sha256_file(checkpoint) == str(spec["checkpoint_sha256"]),
        "CHECKPOINT_SHA",
    )

    confirmation.load_frozen_selection()
    frozen = confirmation.load_frozen_geometry()

    rows, encoded, events, population_provenance = lowdisp.build_input_state(
        spec=spec,
        snapshot=snapshot,
    )
    lookup = lowdisp.row_index(rows)

    model, kernels, model_provenance = geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=checkpoint,
        gpu_id=0,
    )
    confirmation.kernel_compat.validate_transformers_kernel_bindings(kernels)
    runtime_ctx = geom.runtime_components(model)
    confirmation.validate_runtime_geometry(runtime_ctx, frozen)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "PARAM_REQUIRES_GRAD",
    )
    model.eval()

    require(len(rows) == ROWS_PER_SCALE, "INPUT_ROWS")
    require(
        tuple(str(row["source_pair_id"]) for row in rows[::2])
        == PAIR_IDS,
        "INPUT_PAIR_ORDER",
    )

    return (
        spec,
        model,
        runtime_ctx,
        frozen,
        rows,
        encoded,
        events,
        lookup,
        device,
        {
            "population": population_provenance,
            "model": model_provenance,
        },
    )


def normalize_item(
    item: Mapping[str, Any],
    *,
    scale_row_index: int,
) -> dict[str, Any]:
    out = {
        "schema_version": ITEM_SCHEMA,
        "scale": str(item["scale"]),
        "source_pair_id": str(item["source_pair_id"]),
        "contrast_cell_id": str(item["contrast_cell_id"]),
        "correct_label_id": int(item["correct_label_id"]),
        "active_wrong_class_id": int(item["active_wrong_class_id"]),
        "native_correct_class_margin": float(item["native_correct_class_margin"]),
        "checkpoint_sha256": str(item["checkpoint_sha256"]),
        "selected_plane": str(item["selected_plane"]),
        "control_plane": str(item["control_plane"]),
        "absolute_anchor_token_index": int(item["absolute_anchor_token_index"]),
        "target_intervention_token_index":
            int(item["target_intervention_token_index"]),
        "gradient_full_l2": float(item["gradient_full_l2"]),
        "selected_projection_l2": float(item["selected_projection_l2"]),
        "control_projection_l2": float(item["control_projection_l2"]),
        "selected_projection_fraction":
            float(item["selected_projection_fraction"]),
        "control_projection_fraction":
            float(item["control_projection_fraction"]),
        "selected_directional_coordinates":
            [float(v) for v in item["selected_directional_coordinates"]],
        "control_directional_coordinates":
            [float(v) for v in item["control_directional_coordinates"]],
        "selected_component_l2": float(item["selected_component_l2"]),
        "control_component_l2": float(item["control_component_l2"]),
        "L_selected": float(item["L_selected"]),
        "L_control": float(item["L_control"]),
        "Delta_L_row": float(item["Delta_L_row"]),
        "cosine_gradient_selected_component":
            item["cosine_gradient_selected_component"],
        "cosine_gradient_control_component":
            item["cosine_gradient_control_component"],
        "scale_row_index": int(scale_row_index),
        "scientific_full_model_forward_count": 1,
        "local_backward_count": 1,
        "parameter_gradient_created": False,
        "training_executed": False,
        "parameter_update_executed": False,
        "behavioral_response_accessed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
    }
    return out


def run_one_row(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index_value: int,
    anchor_index: int,
    device: torch.device,
    scale_row_index: int,
) -> dict[str, Any]:
    item, _arrays, _capture = study_b.run_native_gradient_row(
        spec=spec,
        model=model,
        runtime_ctx=runtime_ctx,
        frozen=frozen,
        encoded=encoded,
        row=row,
        row_index_value=row_index_value,
        anchor_index=anchor_index,
        device=device,
    )
    return normalize_item(item, scale_row_index=scale_row_index)


def technical_gate_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
) -> None:
    paths = worker_paths(Path(temp_dir), scale)
    try:
        (
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, _provenance,
        ) = prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        pair = PAIR_IDS[0]
        cell = TARGET_CELLS[0]
        idx = lookup[(pair, cell)]
        anchor = int(
            events[(pair, cell, "A_IDENTITY")]["absolute_anchor_token_index"]
        )
        _ = run_one_row(
            spec=spec,
            model=model,
            runtime_ctx=runtime_ctx,
            frozen=frozen,
            encoded=encoded,
            row=rows[idx],
            row_index_value=idx,
            anchor_index=anchor,
            device=device,
            scale_row_index=0,
        )

        paths["gate"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "model_reconstruction_pass": True,
            "frozen_lowdisp_population_pass": True,
            "token_anchor_pass": True,
            "native_local_leaf_gradient_pass": True,
            "parameter_gradient_absent_pass": True,
            "frozen_plane_identity_pass": True,
            "numeric_margin_retained": False,
            "numeric_gradient_retained": False,
            "numeric_Delta_L_retained": False,
            "behavioral_response_accessed": False,
            "scientific_conclusion": None,
        }))
    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def raw_worker(
    *,
    scale: str,
    model_snapshot: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    paths = worker_paths(Path(temp_dir), scale)
    try:
        authenticate_repo(expected_head)
        (
            spec, model, runtime_ctx, frozen, rows, encoded, events,
            lookup, device, provenance,
        ) = prepare_scale(
            scale=scale,
            model_snapshot=Path(model_snapshot),
            physical_device=SCALE_TO_PHYSICAL_GPU[scale],
        )

        items: list[dict[str, Any]] = []
        scale_row_index = 0
        for pair in PAIR_IDS:
            for cell in TARGET_CELLS:
                idx = lookup[(pair, cell)]
                anchor = int(
                    events[(pair, cell, "A_IDENTITY")][
                        "absolute_anchor_token_index"
                    ]
                )
                item = run_one_row(
                    spec=spec,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    frozen=frozen,
                    encoded=encoded,
                    row=rows[idx],
                    row_index_value=idx,
                    anchor_index=anchor,
                    device=device,
                    scale_row_index=scale_row_index,
                )
                items.append(item)
                scale_row_index += 1

        require(len(items) == ROWS_PER_SCALE, "WORKER_ITEM_COUNT")
        torch.cuda.synchronize(device)

        paths["items"].write_bytes(jsonl_bytes(items))
        paths["meta"].write_bytes(pretty_json_bytes({
            "scale": scale,
            "execution_head": expected_head,
            "pair_first": PAIR_IDS[0],
            "pair_last": PAIR_IDS[-1],
            "pair_count": PAIR_COUNT,
            "row_count": ROWS_PER_SCALE,
            "physical_device": SCALE_TO_PHYSICAL_GPU[scale],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "selected_plane": spec["selected_plane"],
            "control_plane": spec["control_plane"],
            "scientific_full_model_forward_count": ROWS_PER_SCALE,
            "local_backward_count": ROWS_PER_SCALE,
            "training_executed": False,
            "parameter_update_executed": False,
            "inferential_test_performed": False,
            "p_value_count_executed": 0,
            "behavioral_response_accessed": False,
            "scientific_conclusion": None,
            "model_provenance": provenance["model"],
            "tokenizer": provenance["population"]["tokenizer"],
        }))
    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        raise


def run_two_processes(
    *,
    target,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    temp_dir: Path,
    expected_head: str | None,
) -> None:
    ctx = mp.get_context("spawn")
    snapshots = {
        "mamba370m": mamba370m_snapshot,
        "mamba14b": mamba14b_snapshot,
    }
    processes = []

    for scale in SCALES:
        kwargs = {
            "scale": scale,
            "model_snapshot": str(snapshots[scale]),
            "temp_dir": str(temp_dir),
        }
        if expected_head is not None:
            kwargs["expected_head"] = expected_head

        process = ctx.Process(
            target=target,
            kwargs=kwargs,
            name=f"lowdisp-native-readout-{scale}",
        )
        process.start()
        processes.append((scale, process))

    for scale, process in processes:
        process.join()
        if process.exitcode != 0:
            paths = worker_paths(temp_dir, scale)
            detail = (
                paths["error"].read_text(encoding="utf-8")
                if paths["error"].is_file()
                else "NO_WORKER_ERROR"
            )
            raise LowDispReadoutError(
                f"WORKER_FAILED:{scale}:\n{detail}"
            )


def run_technical_gate(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)

    with tempfile.TemporaryDirectory(
        prefix="gen4_lowdisp_native_readout_gate_"
    ) as tmp:
        temp_dir = Path(tmp)
        run_two_processes(
            target=technical_gate_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=None,
        )
        for scale in SCALES:
            gate = json.loads(
                worker_paths(temp_dir, scale)["gate"].read_text(
                    encoding="utf-8"
                )
            )
            for key in (
                "model_reconstruction_pass",
                "frozen_lowdisp_population_pass",
                "token_anchor_pass",
                "native_local_leaf_gradient_pass",
                "parameter_gradient_absent_pass",
                "frozen_plane_identity_pass",
            ):
                require(gate[key] is True, f"GATE:{scale}:{key}")
            require(gate["numeric_margin_retained"] is False, "GATE_MARGIN")
            require(gate["numeric_gradient_retained"] is False, "GATE_GRAD")
            require(gate["numeric_Delta_L_retained"] is False, "GATE_DELTA_L")
            require(
                gate["behavioral_response_accessed"] is False,
                "GATE_BEHAVIOR",
            )

    print("RESULT=" + GATE_RESULT)
    print("MAMBA370M=PASS")
    print("MAMBA14B=PASS")
    print("NUMERIC_DELTA_L_RETAINED=False")
    print("BEHAVIORAL_RESPONSE_ACCESSED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def write_sums(root: Path, names: Sequence[str]) -> None:
    (root / CHECKSUM_FILE).write_text(
        "".join(
            f"{sha256_file(root / name)}  {name}\n"
            for name in sorted(names)
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    expected_head: str,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), "OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(
        prefix="gen4_lowdisp_native_readout_raw_"
    ) as tmp:
        temp_dir = Path(tmp)
        run_two_processes(
            target=raw_worker,
            mamba370m_snapshot=mamba370m_snapshot,
            mamba14b_snapshot=mamba14b_snapshot,
            temp_dir=temp_dir,
            expected_head=expected_head,
        )

        all_items: list[dict[str, Any]] = []
        worker_meta: dict[str, Any] = {}
        for scale in SCALES:
            paths = worker_paths(temp_dir, scale)
            scale_items = [
                json.loads(line)
                for line in paths["items"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            require(
                len(scale_items) == ROWS_PER_SCALE,
                f"ITEM_COUNT:{scale}",
            )
            all_items.extend(scale_items)
            worker_meta[scale] = json.loads(
                paths["meta"].read_text(encoding="utf-8")
            )

    require(len(all_items) == TOTAL_ROWS, "TOTAL_ITEM_COUNT")
    expected_keys = {
        (scale, pair, cell)
        for scale in SCALES
        for pair in PAIR_IDS
        for cell in TARGET_CELLS
    }
    observed_keys = {
        (
            str(item["scale"]),
            str(item["source_pair_id"]),
            str(item["contrast_cell_id"]),
        )
        for item in all_items
    }
    require(observed_keys == expected_keys, "TOTAL_COVERAGE")

    output_dir.mkdir(parents=True, exist_ok=False)
    item_raw = jsonl_bytes(all_items)
    (output_dir / ITEM_FILE).write_bytes(item_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "population": "xg1_fact_5401..xg1_fact_5700",
        "pair_count": PAIR_COUNT,
        "cells": list(TARGET_CELLS),
        "scales": list(SCALES),
        "row_count": TOTAL_ROWS,
        "scientific_full_model_forward_count": TOTAL_ROWS,
        "local_backward_count": TOTAL_ROWS,
        "parameter_gradient_count": 0,
        "training_executed": False,
        "parameter_update_executed": False,
        "inferential_test_performed": False,
        "p_value_count_executed": 0,
        "behavioral_response_accessed": False,
        "predicted_behavior_computed": False,
        "behavioral_merge_computed": False,
        "scientific_conclusion": None,
        "workers": worker_meta,
    }
    (output_dir / SUMMARY_FILE).write_bytes(
        pretty_json_bytes(summary)
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RAW_RESULT,
        "execution_head": expected_head,
        "required_base_commit": REQUIRED_BASE_COMMIT,
        "structural_row_sha256": STRUCTURAL_ROW_SHA256,
        "structural_source_sha256": STRUCTURAL_SOURCE_SHA256,
        "token_gate_cross_scale_sha256": TOKEN_GATE_CROSS_SCALE_SHA256,
        "item_sha256": sha256_bytes(item_raw),
        "summary_sha256": sha256_file(output_dir / SUMMARY_FILE),
        "row_count": TOTAL_ROWS,
        "behavioral_response_accessed": False,
        "predicted_behavior_computed": False,
        "behavioral_merge_computed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
    }
    (output_dir / MANIFEST_FILE).write_bytes(
        pretty_json_bytes(manifest)
    )
    write_sums(
        output_dir,
        (ITEM_FILE, SUMMARY_FILE, MANIFEST_FILE),
    )

    print("RESULT=" + RAW_RESULT)
    print("PAIR_RANGE=xg1_fact_5401..xg1_fact_5700")
    print("PAIR_COUNT=300")
    print("ROW_COUNT=1200")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=1200")
    print("LOCAL_BACKWARD_COUNT=1200")
    print("PARAMETER_GRADIENT_COUNT=0")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("BEHAVIORAL_RESPONSE_ACCESSED=False")
    print("PREDICTED_BEHAVIOR_COMPUTED=False")
    print("BEHAVIORAL_MERGE_COMPUTED=False")
    print("TRAINING_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure native local task-margin readout Delta_L on the frozen "
            "XG1 5401..5700 low-displacement cohort, without reading any "
            "behavioral response artifact. No p-value."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("gate", "raw"),
        required=True,
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument(
        "--mamba370m-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mamba14b-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args(argv)
    if args.mode == "raw":
        require(args.output_dir is not None, "OUTPUT_DIR_REQUIRED")
    else:
        require(args.output_dir is None, "GATE_OUTPUT_FORBIDDEN")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "gate":
        run_technical_gate(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
        )
    else:
        run_raw(
            expected_head=args.expected_head,
            mamba370m_snapshot=args.mamba370m_snapshot,
            mamba14b_snapshot=args.mamba14b_snapshot,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
