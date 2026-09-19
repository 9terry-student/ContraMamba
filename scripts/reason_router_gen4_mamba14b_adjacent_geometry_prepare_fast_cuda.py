#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_mamba14b_geometry_prepare_fast_cuda
    as canonical,
)

ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
DESIGN_FREEZE_COMMIT = "3ee2d7600de649a711752b8293687047c3c5ec4e"

CANONICAL_TRIPLET = (33, 34, 35)
ADJACENT_TRIPLET = (34, 35, 36)
SOURCE_BLOCK = 34
TARGET_RESIDUAL_LAYER = 35
INTERVENTION_LAYER = 36
LOCAL_LAYER_OFFSETS = (-2, -1, 0)
TARGET_OFFSET = 2

SELECTED_CAUSAL_CANDIDATE = "P5"
RESULT_PASS = "PASS_MAMBA14B_ADJACENT_GEOMETRY_PREPARATION"
SUMMARY_SCHEMA = "gen4-mamba14b-adjacent-geometry-preparation-summary-v1"
MANIFEST_SCHEMA = "gen4-mamba14b-adjacent-geometry-preparation-manifest-v1"
STRONG_SCHEMA = "gen4-mamba14b-adjacent-strong-mask-v1"

TOTAL_FORWARD_BUDGET = 2400
XG1_FORWARD_BUDGET = 0


class AdjacentGeometryError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AdjacentGeometryError(message)


def _configure_canonical_module() -> None:
    canonical.SOURCE_BLOCK = SOURCE_BLOCK
    canonical.TARGET_RESIDUAL_LAYER = TARGET_RESIDUAL_LAYER
    canonical.INTERVENTION_LAYER = INTERVENTION_LAYER
    canonical.LOCAL_LAYER_OFFSETS = LOCAL_LAYER_OFFSETS
    canonical.TARGET_OFFSET = TARGET_OFFSET
    canonical.RESULT_PASS = RESULT_PASS
    canonical.validate_protocol_constants = validate_protocol_constants


def validate_protocol_constants() -> None:
    require(
        (SOURCE_BLOCK, TARGET_RESIDUAL_LAYER, INTERVENTION_LAYER)
        == ADJACENT_TRIPLET,
        "ADJACENT_TRIPLET",
    )
    require(
        tuple(
            value - INTERVENTION_LAYER
            for value in ADJACENT_TRIPLET
        )
        == LOCAL_LAYER_OFFSETS,
        "LOCAL_LAYER_OFFSETS",
    )
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(canonical.K == 5, "K")
    require(
        canonical.TOTAL_FORWARD_BUDGET == TOTAL_FORWARD_BUDGET,
        "FORWARD_BUDGET",
    )


def select_response_blind_control(
    lambda_plus: Sequence[float],
) -> str:
    require(len(lambda_plus) == 5, "LAMBDA_COUNT")
    planes = tuple(f"P{i}" for i in range(1, 6))
    values = {
        plane: float(value)
        for plane, value in zip(planes, lambda_plus, strict=True)
        if plane != SELECTED_CAUSAL_CANDIDATE
    }
    require(
        all(math.isfinite(value) and value > 0.0 for value in values.values()),
        "CONTROL_LAMBDA_INVALID",
    )
    best = max(values.values())
    winners = [
        plane for plane, value in values.items()
        if value == best
    ]
    require(len(winners) == 1, f"CONTROL_SELECTION_NOT_UNIQUE:{winners}")
    return winners[0]


def _adjacent_worker_run(**kwargs: Any) -> None:
    _configure_canonical_module()
    canonical.worker_run(**kwargs)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rewrite_adjacent_bundle(
    *,
    output_dir: Path,
    expected_head: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = output_dir / canonical.SUMMARY_FILE
    manifest_path = output_dir / canonical.MANIFEST_FILE
    strong_path = output_dir / canonical.STRONG_INDEX_FILE
    sums_path = output_dir / canonical.CHECKSUM_FILE

    out = dict(summary)
    lambdas = [
        float(value)
        for value in out["lambda_plus_by_plane"]
    ]
    control = select_response_blind_control(lambdas)

    out["schema_version"] = SUMMARY_SCHEMA
    out["result"] = RESULT_PASS
    out["phase"] = "one_shot_adjacent_site_geometry_preparation"
    out["claim_boundary"] = (
        "Response-blind Mamba-1.4B adjacent-site geometry reconstruction "
        "for fixed +1 triplet (34,35,36); no XG1 specificity response observed."
    )
    out["design_freeze_commit"] = DESIGN_FREEZE_COMMIT
    out["canonical_triplet"] = list(CANONICAL_TRIPLET)
    out["adjacent_triplet"] = list(ADJACENT_TRIPLET)
    out["adjacent_shift"] = 1
    out["layer_mapping"] = {
        "rule": (
            "shift canonical triplet (33,34,35) exactly +1 downstream "
            "while preserving offsets (-2,-1,0)"
        ),
        "canonical_triplet": list(CANONICAL_TRIPLET),
        "adjacent_triplet": list(ADJACENT_TRIPLET),
        "preserved_local_offsets": list(LOCAL_LAYER_OFFSETS),
        "source_block": SOURCE_BLOCK,
        "target_residual_layer": TARGET_RESIDUAL_LAYER,
        "intervention_layer": INTERVENTION_LAYER,
    }
    out["fixed_causal_candidate"] = SELECTED_CAUSAL_CANDIDATE
    out["response_blind_control_plane"] = control
    out["control_selection_rule"] = (
        "c_adj=argmax_{k!=P5} lambda_plus_adj,k"
    )
    out["control_selection_uses_response"] = False
    out["plane_selection_performed"] = False
    out["control_selection_performed"] = True
    out["xg1_model_forward_count"] = 0
    out["xg1_specificity_accessed"] = False
    out["causal_response_observed"] = False
    out["statistical_testing_performed"] = False
    out["rescue_performed"] = False

    summary_path.write_bytes(canonical.pretty_json_bytes(out))

    strong = json.loads(strong_path.read_text(encoding="utf-8"))
    strong["schema_version"] = STRONG_SCHEMA
    strong["intervention_layer"] = INTERVENTION_LAYER
    strong["site_role"] = "one_shot_adjacent_plus1"
    strong_path.write_bytes(canonical.pretty_json_bytes(strong))

    file_names = [
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name not in {
            canonical.MANIFEST_FILE,
            canonical.CHECKSUM_FILE,
        }
    ]
    file_hashes = {
        name: _sha256_file(output_dir / name)
        for name in file_names
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "canonical_triplet": list(CANONICAL_TRIPLET),
        "adjacent_triplet": list(ADJACENT_TRIPLET),
        "fixed_causal_candidate": SELECTED_CAUSAL_CANDIDATE,
        "response_blind_control_plane": control,
        "control_selection_uses_response": False,
        "output_file_sha256": dict(sorted(file_hashes.items())),
        "scientific_model_forward_count": TOTAL_FORWARD_BUDGET,
        "xg1_model_forward_count": XG1_FORWARD_BUDGET,
        "xg1_accessed": False,
        "response_observed": False,
        "plane_selection_performed": False,
        "control_selection_performed": True,
        "statistical_testing_performed": False,
        "training_executed": False,
        "backward_executed": False,
        "rescue_performed": False,
    }
    manifest_path.write_bytes(canonical.pretty_json_bytes(manifest))
    file_hashes[canonical.MANIFEST_FILE] = _sha256_file(manifest_path)

    sums_path.write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(file_hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )

    require(out["scientific_model_forward_count_this_run"] == 2400, "SUMMARY_BUDGET")
    require(out["xg1_model_forward_count"] == 0, "SUMMARY_XG1_BUDGET")
    require(out["layer_mapping"]["adjacent_triplet"] == [34, 35, 36], "SUMMARY_TRIPLET")
    require(out["fixed_causal_candidate"] == "P5", "SUMMARY_CANDIDATE")
    require(out["response_blind_control_plane"] == control, "SUMMARY_CONTROL")
    return out


def run_geometry_preparation(
    *,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _configure_canonical_module()
    validate_protocol_constants()
    canonical.authenticate_repo(expected_head)
    canonical.validate_snapshot(snapshot)

    require(
        compact_checkpoint.resolve()
        == (ROOT / canonical.COMPACT_CHECKPOINT_REL).resolve(),
        "COMPACT_CHECKPOINT_PATH",
    )
    require(
        canonical.sha256_file(compact_checkpoint)
        == canonical.COMPACT_CHECKPOINT_SHA256,
        "COMPACT_CHECKPOINT_SHA",
    )
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= canonical.GPU_COUNT, "CUDA_DEVICE_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="gen4_mamba14b_adjacent_geometry_"
    ) as tmp:
        temp_dir = Path(tmp)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for gpu_id, family in enumerate(canonical.FAMILIES):
            process = ctx.Process(
                target=_adjacent_worker_run,
                kwargs={
                    "gpu_id": gpu_id,
                    "family": family,
                    "expected_head": expected_head,
                    "snapshot": str(snapshot),
                    "compact_checkpoint": str(compact_checkpoint),
                    "temp_dir": str(temp_dir),
                },
                name=f"mamba14b-adjacent-geometry-{family}-gpu{gpu_id}",
            )
            process.start()
            processes.append(process)

        for gpu_id, process in enumerate(processes):
            process.join()
            if process.exitcode != 0:
                paths = canonical._worker_payload_paths(temp_dir, gpu_id)
                detail = (
                    paths["error"].read_text(encoding="utf-8")
                    if paths["error"].is_file()
                    else "NO_WORKER_ERROR_FILE"
                )
                raise AdjacentGeometryError(
                    f"WORKER_{gpu_id}_FAILED:\n{detail}"
                )

        worker_payloads: list[dict[str, Any]] = []
        plans: dict[str, torch.Tensor] = {}
        items: dict[str, list[dict[str, Any]]] = {}

        for gpu_id, family in enumerate(canonical.FAMILIES):
            paths = canonical._worker_payload_paths(temp_dir, gpu_id)
            require(paths["meta"].is_file(), f"WORKER_META_MISSING:{gpu_id}")
            require(paths["plan"].is_file(), f"WORKER_PLAN_MISSING:{gpu_id}")
            require(paths["items"].is_file(), f"WORKER_ITEMS_MISSING:{gpu_id}")

            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            require(meta["gpu_id"] == gpu_id, "WORKER_META_GPU")
            require(meta["family_key"] == family, "WORKER_META_FAMILY")
            require(
                meta["model_forward_count"] == canonical.FORWARDS_PER_FAMILY,
                "WORKER_FORWARD_BUDGET",
            )
            require(meta["xg1_accessed"] is False, "WORKER_XG1_ACCESS")
            require(meta["response_observed"] is False, "WORKER_RESPONSE_ACCESS")
            require(
                meta["plan_sha256"] == canonical.sha256_file(paths["plan"]),
                "WORKER_PLAN_SHA",
            )
            require(
                meta["items_sha256"] == canonical.sha256_file(paths["items"]),
                "WORKER_ITEMS_SHA",
            )
            worker_payloads.append(meta)

            try:
                plan = torch.load(
                    paths["plan"],
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                plan = torch.load(paths["plan"], map_location="cpu")
            require(torch.is_tensor(plan), f"PLAN_NOT_TENSOR:{family}")
            plans[family] = plan.to(torch.float64).contiguous()
            items[family] = canonical.read_jsonl(paths["items"])

        summary = canonical.write_output_bundle(
            output_dir=output_dir,
            expected_head=expected_head,
            snapshot=snapshot,
            worker_payloads=worker_payloads,
            plans=plans,
            items=items,
        )
        return _rewrite_adjacent_bundle(
            output_dir=output_dir,
            expected_head=expected_head,
            summary=summary,
        )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare response-blind Mamba-1.4B geometry for the frozen "
            "one-shot adjacent +1 site (34,35,36). Two GPUs; no XG1 response."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=ROOT / canonical.COMPACT_CHECKPOINT_REL,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_geometry_preparation(
        expected_head=args.expected_head,
        snapshot=args.model_snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
    )

    print("RESULT=" + str(summary["result"]))
    print("CANONICAL_TRIPLET=33,34,35")
    print("ADJACENT_TRIPLET=34,35,36")
    print(
        "STRONG_COUNT="
        + str(summary["strong_mask"]["strong_count"])
    )
    print(
        "STRONG_INDEX_SHA256="
        + str(summary["strong_mask"]["strong_index_sha256"])
    )
    for plane in range(1, canonical.K + 1):
        item = summary["principal_planes"][f"P{plane}"]
        print(
            f"P{plane}_LAMBDA_PLUS="
            + format(float(item["lambda_plus"]), ".17g")
        )
    print("FIXED_CAUSAL_CANDIDATE=P5")
    print(
        "RESPONSE_BLIND_CONTROL_PLANE="
        + str(summary["response_blind_control_plane"])
    )
    print("CONTROL_SELECTION_USES_RESPONSE=False")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT_THIS_RUN=2400")
    print("XG1_MODEL_FORWARD_COUNT=0")
    print("XG1_SPECIFICITY_ACCESSED=False")
    print("CAUSAL_RESPONSE_OBSERVED=False")
    print("PLANE_SELECTION_PERFORMED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("CUDA_FAST_PATH_MANDATORY=True")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("RESCUE_PERFORMED=False")


if __name__ == "__main__":
    main()
