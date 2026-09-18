from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import build_reason_router_gen4_seed181_behavioral_bridge_holdout as holdout
from scripts import reason_router_gen4_pp3_restoration_sufficiency_fast_cuda as restoration
from scripts import reason_router_gen4_seed181_checkpoint_replication_fast_cuda as seed181
from scripts import reason_router_gen4_six_cell_tier2_inference_adapter as adapter
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as tokenizer_gate


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-behavioral-restoration-bridge"
DESIGN_COMMIT = "9ea1617f4485fa0b6093df0c70aa88a42260c710"
DESIGN_PATH = "reports/reason_router_gen4_seed181_behavioral_restoration_bridge_design.md"
DESIGN_BLOB = "9f4e7493893b1d324da3c8fbb981010c380edfae"

SEED = 181
ARM = "G3-GROUP-D-HALF"
CHECKPOINT_SHA256 = seed181.CHECKPOINT_SHA256
CHECKPOINT_BYTES = seed181.CHECKPOINT_BYTES

GEOMETRY_ROOT = Path(
    "reports/reason_router_gen4_seed181_checkpoint_replication_runs/"
    "g4k-seed181-checkpoint-replication-8e96fd1-retry1"
)
GEOMETRY_JSON = GEOMETRY_ROOT / "seed181_principal_geometry.json"
GEOMETRY_PT = GEOMETRY_ROOT / "seed181_principal_geometry.pt"
GEOMETRY_JSON_SHA256 = "e6e9db909eb7d2c6bbdb493a4efeca8c18e4d474cf99a943be0f3f7b9dee1012"
GEOMETRY_PT_SHA256 = "de3ae6a450c2ba0a85b4f53919e3765e6e7dfb6dba3676535554c437b1647a1c"

CONDITIONS = ("native", "pp3_neutralized", "pp3_restored", "pp5_replacement")
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
LABEL_ID_BY_CELL = {"C0_SHAM": 2, "C2_NAME": 1}
LABEL_NAME_BY_CELL = {"C0_SHAM": "SUPPORT", "C2_NAME": "NOT_ENTITLED"}
SHARD_RANGES = {0: (2701, 2850), 1: (2851, 3000)}
PAIRS_PER_SHARD = 150
ROWS_PER_PAIR = len(TARGET_CELLS) * len(CONDITIONS)
FORWARDS_PER_PAIR = ROWS_PER_PAIR
FORWARDS_PER_SHARD = PAIRS_PER_SHARD * FORWARDS_PER_PAIR
TOTAL_FORWARD_BUDGET = FORWARDS_PER_SHARD * 2
MIN_PHYSICAL_GPU_COUNT = 2

ROW_FILE = "behavioral_rows.jsonl"
SUMMARY_FILE = "shard_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ROW_SCHEMA = "gen4-seed181-behavioral-restoration-row-v1"
SUMMARY_SCHEMA = "gen4-seed181-behavioral-restoration-shard-summary-v1"

FROZEN_DEPENDENCIES = (
    DESIGN_PATH,
    "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py",
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py",
    "scripts/reason_router_gen4_pp3_restoration_sufficiency_fast_cuda.py",
    "scripts/reason_router_gen4_seed181_checkpoint_replication_fast_cuda.py",
    "scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_core.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runtime.py",
    "scripts/reason_router_gen4_k_directional_alignment_transport_runner.py",
    GEOMETRY_JSON.as_posix(),
    GEOMETRY_PT.as_posix(),
)


class BehavioralBridgeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise BehavioralBridgeError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BehavioralBridgeError("GIT_FAILURE:" + " ".join(args)) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(x) for x in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        out.append(value)
    return out


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    require(
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", DESIGN_COMMIT, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "DESIGN_NOT_ANCESTOR",
    )
    require(git("rev-parse", f"HEAD:{DESIGN_PATH}") == DESIGN_BLOB, "DESIGN_BLOB")
    require(
        subprocess.call(
            ["git", "diff", "--quiet", DESIGN_COMMIT, expected_head, "--", *FROZEN_DEPENDENCIES],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "FROZEN_DEPENDENCY_DRIFT",
    )


def authenticate_checkpoint(path: Path) -> None:
    require(path.is_file(), "CHECKPOINT_MISSING")
    require(path.stat().st_size == CHECKPOINT_BYTES, "CHECKPOINT_BYTES")
    require(sha256_file(path) == CHECKPOINT_SHA256, "CHECKPOINT_SHA256")
    require(adapter.expected_checkpoint_sha256(SEED, ARM) == CHECKPOINT_SHA256, "CHECKPOINT_REGISTRY")


def physical_gpu_inventory_count() -> int:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BehavioralBridgeError("PHYSICAL_GPU_INVENTORY_UNAVAILABLE") from exc
    count = sum(1 for line in raw.splitlines() if line.strip().startswith("GPU "))
    require(count >= MIN_PHYSICAL_GPU_COUNT, f"PHYSICAL_GPU_COUNT:{count}")
    return count


def validate_cuda_partition(shard_index: int, physical_device: int) -> torch.device:
    require(shard_index in SHARD_RANGES, f"SHARD:{shard_index}")
    require(physical_device == shard_index, "PHYSICAL_DEVICE_SHARD_BINDING")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible == str(physical_device), f"CUDA_VISIBLE_DEVICES:{visible}")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() == 1, f"LOGICAL_CUDA_DEVICE_COUNT:{torch.cuda.device_count()}")
    return torch.device("cuda:0")


def validate_fresh_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = ROOT / holdout.OUTPUT_DIR
    manifest = holdout.validate_written(root)
    require(manifest["result"] == "PASS_SEED181_BEHAVIORAL_BRIDGE_XG1_2701_3000_STRUCTURAL", "DATA_RESULT")
    facts = read_jsonl(root / holdout.SOURCE_FILE)
    rows = read_jsonl(root / holdout.ROW_FILE)
    require(len(facts) == 300 and len(rows) == 1800, "DATA_COUNTS")
    return facts, rows, manifest


def load_seed181_planes() -> dict[str, torch.Tensor]:
    json_path = ROOT / GEOMETRY_JSON
    tensor_path = ROOT / GEOMETRY_PT
    require(sha256_file(json_path) == GEOMETRY_JSON_SHA256, "GEOMETRY_JSON_SHA")
    require(sha256_file(tensor_path) == GEOMETRY_PT_SHA256, "GEOMETRY_PT_SHA")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    require(metadata["checkpoint_sha256"] == CHECKPOINT_SHA256, "GEOMETRY_CHECKPOINT")
    require(metadata["homolog_plane_index"] == 3, "GEOMETRY_HOMOLOG")
    require(metadata["control_plane_index"] == 5, "GEOMETRY_CONTROL")
    require(metadata["selection_uses_response"] is False, "GEOMETRY_RESPONSE_BOUNDARY")

    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    geometry = {
        "planes": tensor["planes"],
        "homolog_plane_index": int(metadata["homolog_plane_index"]),
        "control_plane_index": int(metadata["control_plane_index"]),
    }
    aliases = seed181.alias_planes(geometry)
    for name, value in aliases.items():
        aliases[name] = value.detach().cpu().to(torch.float64).contiguous()
        require(tuple(aliases[name].shape) == (restoration.DIM,), f"PLANE_SHAPE:{name}")
        require(abs(float(torch.linalg.vector_norm(aliases[name])) - 1.0) <= restoration.TOL, f"PLANE_NORM:{name}")
    return aliases


def validate_label_semantics(
    facts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    facts_by_id = {str(x["pair_id"]): x for x in facts}
    selected: list[dict[str, Any]] = []
    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    for pair_id in holdout.expected_pair_ids():
        fact = facts_by_id[pair_id]
        c0 = dict(by_key[(pair_id, "C0_SHAM")])
        c2 = dict(by_key[(pair_id, "C2_NAME")])
        require(c0["claim"] == c0["evidence"], f"C0_SUPPORT_SEMANTICS:{pair_id}")
        require(
            c2["evidence"] == holdout.base.render_statement(fact, name=str(fact["alternate_name"])),
            f"C2_NOT_ENTITLED_SEMANTICS:{pair_id}",
        )
        require(c2["claim"] == c0["claim"], f"PAIR_CLAIM_IDENTITY:{pair_id}")
        selected.extend([c0, c2])
    return selected, facts_by_id


def build_anchor_and_encoding(
    selected_rows: Sequence[Mapping[str, Any]],
    facts_by_id: Mapping[str, Mapping[str, Any]],
    tokenizer_snapshot: Path,
):
    tokenizer, tokenizer_provenance = tokenizer_gate.load_canonical_analysis_tokenizer(tokenizer_snapshot)
    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected_rows:
        pair_id = str(row["source_pair_id"])
        analyzed = tokenizer_gate.analyze_required_anchors_for_row(row, facts_by_id[pair_id], tokenizer)
        require({x["anchor_name"] for x in analyzed} == {"A_IDENTITY", "A_NAME"}, f"ANCHOR_SET:{pair_id}:{row['contrast_cell_id']}")
        for item in analyzed:
            key = (pair_id, str(row["contrast_cell_id"]), str(item["anchor_name"]))
            require(key not in events, f"ANCHOR_DUPLICATE:{key}")
            require(bool(item["post4_eligible"]), f"ANCHOR_INELIGIBLE:{key}")
            events[key] = dict(item)
        identity = events[(pair_id, str(row["contrast_cell_id"]), "A_IDENTITY")]
        name = events[(pair_id, str(row["contrast_cell_id"]), "A_NAME")]
        require(
            int(identity["absolute_anchor_token_index"]) == int(name["absolute_anchor_token_index"]),
            f"IDENTITY_NAME_ANCHOR_MISMATCH:{pair_id}:{row['contrast_cell_id']}",
        )
    encoded = adapter.encode_gen4_rows(selected_rows, tokenizer)
    return events, encoded, tokenizer_provenance


def shard_pairs(shard_index: int) -> tuple[str, ...]:
    first, last = SHARD_RANGES[shard_index]
    pairs = tuple(f"xg1_fact_{i:03d}" for i in range(first, last + 1))
    require(len(pairs) == PAIRS_PER_SHARD, "SHARD_PAIR_COUNT")
    return pairs


def feature_batch(encoded: Mapping[str, Any], index: int, device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
        tensor = encoded[key]
        require(torch.is_tensor(tensor), f"ENCODED_TENSOR:{key}")
        out[key] = tensor[index : index + 1].detach().to(device).contiguous()
    return out


def behavior_hook(
    output: torch.Tensor,
    *,
    token_index: int,
    strong_mask: torch.Tensor,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    audit: dict[str, Any],
) -> torch.Tensor:
    require(condition in {"pp3_neutralized", "pp3_restored", "pp5_replacement"}, f"HOOK_CONDITION:{condition}")
    runtime = restoration.holdout.phase1.base.prevalence_eq
    core = runtime.core
    require(
        output.ndim == 3
        and output.shape[0] == 1
        and output.shape[-1] == 2 * core.INTERMEDIATE_SIZE,
        "INPROJ_SHAPE",
    )
    require(0 <= token_index < output.shape[1], "TOKEN_INDEX")
    mask = strong_mask.detach().cpu().bool().contiguous()
    require(mask.numel() == core.INTERMEDIATE_SIZE and int(mask.sum()) == restoration.DIM, "STRONG_MASK")

    before = output.detach().clone()
    mask_device = mask.to(before.device)
    h = (
        before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )
    info = restoration.condition_correction(h, condition, planes)
    correction = info["d"].detach().cpu().to(torch.float64).contiguous()

    out = output.clone()
    intended = correction.to(device=out.device, dtype=out.dtype)
    out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device] += intended

    require(torch.equal(out[:, :, core.INTERMEDIATE_SIZE :], before[:, :, core.INTERMEDIATE_SIZE :]), "GATE_CHANGED")
    require(torch.equal(out[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device], before[:, :, : core.INTERMEDIATE_SIZE][:, :, ~mask_device]), "NONSTRONG_CHANGED")
    if token_index:
        require(torch.equal(out[:, :token_index, :], before[:, :token_index, :]), "EARLIER_CHANGED")
    if token_index + 1 < out.shape[1]:
        require(torch.equal(out[:, token_index + 1 :, :], before[:, token_index + 1 :, :]), "LATER_CHANGED")

    applied = (
        out[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
        - before[0, token_index, : core.INTERMEDIATE_SIZE][mask_device]
    ).detach().cpu().to(torch.float64)
    residual = float(torch.max(torch.abs(applied - intended.detach().cpu().to(torch.float64))))
    require(residual <= runtime.transport_runtime.RUNTIME_CAST_TOL, f"APPLIED_RESIDUAL:{residual}")

    audit.clear()
    audit.update(
        {
            "condition": condition,
            "token_index": int(token_index),
            "coefficient_source": "native_seed181_P3_coordinates",
            "native_p3_a": float(info["a"]),
            "native_p3_b": float(info["b"]),
            "p3_component_l2": float(info["c3_l2"]),
            "p5_component_l2": float(info["c5_l2"]),
            "matched_component_norm_abs_difference": float(info["restoration_addition_norm_mismatch"]),
            "correction_l2": float(info["direct_final_state_correction_l2"]),
            "applied_correction_max_abs_residual": residual,
            "probe_correction_l2": 0.0,
        }
    )
    return out


def install_behavior_hook(mixer17: Any, **kwargs: Any):
    def hook(_module, _args, output):
        return behavior_hook(output, **kwargs)

    return mixer17.in_proj.register_forward_hook(hook)


def correct_margin(logits: Sequence[float], label_id: int) -> float:
    require(len(logits) == 3 and label_id in {0, 1, 2}, "MARGIN_INPUT")
    correct = float(logits[label_id])
    wrong = max(float(logits[i]) for i in range(3) if i != label_id)
    value = correct - wrong
    require(math.isfinite(value), "MARGIN_NONFINITE")
    return value


def run_condition(
    *,
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    encoded: Mapping[str, Any],
    row: Mapping[str, Any],
    row_index: int,
    anchor_index: int,
    condition: str,
    planes: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    audit: dict[str, Any] | None = None
    handle = None
    if condition != "native":
        audit = {}
        handle = install_behavior_hook(
            runtime_ctx["mixer17"],
            token_index=anchor_index + restoration.holdout.phase1.base.prevalence_eq.core.TARGET_OFFSET,
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=planes,
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = adapter.historical_forward(
                model,
                feature_batch(encoded, row_index, device),
                arm=ARM,
            )
    finally:
        if handle is not None:
            handle.remove()

    serialized = adapter.serialize_model_outputs(
        output,
        [row],
        seed=SEED,
        arm=ARM,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )[0]
    label_id = LABEL_ID_BY_CELL[str(row["contrast_cell_id"])]
    margin = correct_margin(serialized["final_logits"], label_id)
    return {
        "schema_version": ROW_SCHEMA,
        "seed": SEED,
        "arm": ARM,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_pair_id": str(row["source_pair_id"]),
        "row_id": str(row["row_id"]),
        "contrast_cell_id": str(row["contrast_cell_id"]),
        "condition": condition,
        "correct_label_id": label_id,
        "correct_label": LABEL_NAME_BY_CELL[str(row["contrast_cell_id"])],
        "refute_logit": float(serialized["refute_logit"]),
        "not_entitled_logit": float(serialized["ne_logit"]),
        "support_logit": float(serialized["support_logit"]),
        "final_logits": [float(x) for x in serialized["final_logits"]],
        "prediction_id": int(serialized["prediction_id"]),
        "prediction": str(serialized["prediction"]),
        "is_correct": int(serialized["prediction_id"]) == label_id,
        "correct_class_logit_margin": margin,
        "q_authorized": float(serialized["q_authorized"]),
        "entitlement_prob": float(serialized["entitlement_prob"]),
        "anchor_name": "A_IDENTITY",
        "absolute_anchor_token_index": int(anchor_index),
        "target_intervention_token_index": int(anchor_index + restoration.holdout.phase1.base.prevalence_eq.core.TARGET_OFFSET),
        "intervention_audit": audit,
        "scientific_full_model_forward_count": 1,
    }


def write_shard(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    row_raw = jsonl_bytes(rows)
    summary_raw = json.dumps(
        dict(summary), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)
    hashes = {
        ROW_FILE: hashlib.sha256(row_raw).hexdigest(),
        SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )


def run_shard(args: argparse.Namespace) -> None:
    authenticate_repo(args.expected_head)
    authenticate_checkpoint(args.checkpoint)
    physical_gpu_count = physical_gpu_inventory_count()
    device = validate_cuda_partition(args.shard_index, args.physical_device)
    facts, all_rows, structural_manifest = validate_fresh_data()
    selected_rows, facts_by_id = validate_label_semantics(facts, all_rows)
    events, encoded, tokenizer_provenance = build_anchor_and_encoding(
        selected_rows, facts_by_id, args.tokenizer_snapshot
    )
    planes = load_seed181_planes()

    row_index = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): i
        for i, row in enumerate(selected_rows)
    }
    require(len(row_index) == 600, "ROW_INDEX_COUNT")

    runtime = restoration.holdout.phase1.base.prevalence_eq
    runtime.backend.runtime_gate()
    kernels = runtime.kernel_compat.load_exact_fast_kernels()
    output_rows: list[dict[str, Any]] = []
    pairs = shard_pairs(args.shard_index)

    with seed181.seed181_binding():
        with runtime.backend.parent_runtime_rebind():
            with runtime.kernel_compat.exact_transformers_kernel_loader(kernels) as calls:
                model, checkpoint_sha = runtime.parent.load_representative_model_external(
                    model_snapshot=args.model_snapshot,
                    checkpoint_path=args.checkpoint,
                )
                require(checkpoint_sha == CHECKPOINT_SHA256, "MODEL_CHECKPOINT_SHA")
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

            for pair_id in pairs:
                for cell in TARGET_CELLS:
                    row = selected_rows[row_index[(pair_id, cell)]]
                    anchor = events[(pair_id, cell, "A_IDENTITY")]
                    anchor_index = int(anchor["absolute_anchor_token_index"])
                    for condition in CONDITIONS:
                        item = run_condition(
                            model=model,
                            runtime_ctx=runtime_ctx,
                            encoded=encoded,
                            row=row,
                            row_index=row_index[(pair_id, cell)],
                            anchor_index=anchor_index,
                            condition=condition,
                            planes=planes,
                            device=device,
                        )
                        item["shard_index"] = args.shard_index
                        item["physical_device"] = args.physical_device
                        output_rows.append(item)

            torch.cuda.synchronize(device)

    require(len(output_rows) == FORWARDS_PER_SHARD, "OUTPUT_ROW_COUNT")
    require(sum(int(x["scientific_full_model_forward_count"]) for x in output_rows) == FORWARDS_PER_SHARD, "FORWARD_BUDGET")

    expected_tuples = {
        (pair, cell, condition)
        for pair in pairs
        for cell in TARGET_CELLS
        for condition in CONDITIONS
    }
    observed_tuples = {
        (str(x["source_pair_id"]), str(x["contrast_cell_id"]), str(x["condition"]))
        for x in output_rows
    }
    require(observed_tuples == expected_tuples, "SHARD_TUPLE_COVERAGE")

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": "PASS_RAW_BEHAVIORAL_SHARD",
        "execution_head": args.expected_head,
        "design_commit": DESIGN_COMMIT,
        "seed": SEED,
        "arm": ARM,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "geometry_json_sha256": GEOMETRY_JSON_SHA256,
        "geometry_pt_sha256": GEOMETRY_PT_SHA256,
        "homolog_plane_index": 3,
        "control_plane_index": 5,
        "selection_uses_response": False,
        "structural_source_sha256": structural_manifest["source_file_sha256"],
        "structural_rows_sha256": structural_manifest["row_file_sha256"],
        "population_first": pairs[0],
        "population_last": pairs[-1],
        "source_pair_count": len(pairs),
        "target_cells": list(TARGET_CELLS),
        "condition_order": list(CONDITIONS),
        "label_contract": dict(LABEL_NAME_BY_CELL),
        "shard_index": args.shard_index,
        "physical_device": args.physical_device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "logical_device_name": torch.cuda.get_device_name(0),
        "physical_gpu_inventory_count": physical_gpu_count,
        "model_name": adapter.MODEL_NAME,
        "model_contract": "historical_v6b_minimal_strict_checkpoint_load",
        "tokenizer": tokenizer_provenance,
        "scientific_full_model_forward_count_this_run": FORWARDS_PER_SHARD,
        "training_executed": False,
        "backward_executed": False,
        "primary_inference_executed": False,
        "scientific_conclusion": None,
    }
    write_shard(args.output_dir, output_rows, summary)
    print("RESULT=PASS_RAW_BEHAVIORAL_SHARD")
    print(f"SHARD_INDEX={args.shard_index}")
    print(f"PHYSICAL_DEVICE={args.physical_device}")
    print(f"PAIR_RANGE={pairs[0]}..{pairs[-1]}")
    print(f"SCIENTIFIC_FULL_MODEL_FORWARD_COUNT_THIS_RUN={FORWARDS_PER_SHARD}")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("SCIENTIFIC_CONCLUSION=None")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    parser.add_argument("--physical-device", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_shard(parse_args(argv))


if __name__ == "__main__":
    main()
