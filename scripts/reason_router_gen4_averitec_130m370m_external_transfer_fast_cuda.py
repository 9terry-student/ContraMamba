#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    build_reason_router_gen4_averitec_gold_evidence_130m370m_token_gate
    as gate,
)
from scripts import (
    reason_router_gen4_seed181_behavioral_restoration_bridge_fast_cuda
    as bridge130,
)
from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge_scale,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "4be69b8662251b88348d15e98b31ffb2eeeb29b7"

SCALE_ORDER = ("mamba130m", "mamba370m")
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_control",
)
N = 462
GPU_BY_SCALE = {
    "mamba130m": 0,
    "mamba370m": 1,
}
FORWARDS_PER_SCALE = N * len(CONDITIONS)
TOTAL_FORWARD_BUDGET = FORWARDS_PER_SCALE * len(SCALE_ORDER)

COHORT_ROOT = Path(
    "data/reason_router_gen4_averitec_gold_evidence_130m370m_boundary_v1"
)
COHORT_FILE = COHORT_ROOT / "compatible_cohort.jsonl"
MANIFEST_FILE = COHORT_ROOT / "token_gate_manifest.json"
COHORT_SUMS_FILE = COHORT_ROOT / "SHA256SUMS.txt"

COHORT_GIT_BLOB = "19ea13642b1b1628e4af93d4184b284ff222980c"
MANIFEST_GIT_BLOB = "30031c4354dc4cdf67b5ea569b4c825df2c14c3b"
COHORT_SUMS_GIT_BLOB = "33ed62087d91f83978c041ee4c5709dbcc07d3e7"
COHORT_SHA256 = "726ead51008b56e84a91989349d534ea160dd4f1b987bed21ef1cad8e6e3016c"
MANIFEST_SHA256 = "658381dc2b0c2e57413aaf48f137eaa66112fc3b56ca8f45f232cfdb69d0490b"

ANCHOR_NAME = "A_CLAIM_EVIDENCE_BOUNDARY"
TARGET_OFFSET = 2

MAMBA130_REPO = "state-spaces/mamba-130m-hf"
MAMBA130_REVISION = "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
MAMBA130_CONFIG_BYTES = 895
MAMBA130_CONFIG_SHA256 = (
    "784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a"
)
MAMBA130_REQUIRED_SNAPSHOT_SHA256 = {
    "config.json": MAMBA130_CONFIG_SHA256,
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

MAMBA130_CHECKPOINT_RELEASE_TAG = "gen4-r5-evaluator-checkpoints-cf08261"
MAMBA130_CHECKPOINT_RELEASE_ID = 387669220
MAMBA130_CHECKPOINT_ASSET_ID = 559773417
MAMBA130_CHECKPOINT_ASSET_NAME = (
    "seed181__G3-GROUP-D-HALF__selected_checkpoint.pt"
)
MAMBA130_CHECKPOINT_BYTES = 518270455
MAMBA130_CHECKPOINT_SHA256 = (
    "afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f"
)
MAMBA130_CHECKPOINT_URL = (
    "https://github.com/9terry-student/ContraMamba/releases/download/"
    f"{MAMBA130_CHECKPOINT_RELEASE_TAG}/{MAMBA130_CHECKPOINT_ASSET_NAME}"
)

MAMBA130_SELECTED = "P3"
MAMBA130_CONTROL = "P5"
MAMBA130_INTERVENTION_LAYER = 17

MAMBA370_SELECTED = "P3"
MAMBA370_CONTROL = "P5"
MAMBA370_INTERVENTION_LAYER = 35

ROW_FILE = "external_transfer_rows.jsonl"
SUMMARY_FILE = "scale_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ROW_SCHEMA = "gen4-averitec-gold-evidence-external-transfer-row-v1"
SUMMARY_SCHEMA = "gen4-averitec-gold-evidence-external-transfer-scale-summary-v1"
RESULT_PASS = "PASS_AVERITEC_GOLD_EVIDENCE_EXTERNAL_TRANSFER_RAW_SCALE"

PRIMARY_INFERENCE_EXECUTED = False
P_VALUE_COUNT_ADDED = 0


class ExternalTransferError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ExternalTransferError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalTransferError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"JSONL_MISSING:{path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{line_no}")
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, expected_head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "REQUIRED_ANCESTOR")

    frozen_blobs = {
        COHORT_FILE.as_posix(): COHORT_GIT_BLOB,
        MANIFEST_FILE.as_posix(): MANIFEST_GIT_BLOB,
        COHORT_SUMS_FILE.as_posix(): COHORT_SUMS_GIT_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(SCALE_ORDER == ("mamba130m", "mamba370m"), "SCALE_ORDER")
    require(CONDITIONS == (
        "native",
        "dominant_neutralized",
        "dominant_control",
    ), "CONDITIONS")
    require(N == 462, "N")
    require(FORWARDS_PER_SCALE == 1386, "FORWARDS_PER_SCALE")
    require(TOTAL_FORWARD_BUDGET == 2772, "TOTAL_FORWARD_BUDGET")
    require(GPU_BY_SCALE == {"mamba130m": 0, "mamba370m": 1}, "GPU_MAP")
    require(MAMBA130_SELECTED == MAMBA370_SELECTED == "P3", "SELECTED")
    require(MAMBA130_CONTROL == MAMBA370_CONTROL == "P5", "CONTROL")
    require(MAMBA130_INTERVENTION_LAYER == 17, "M130_LAYER")
    require(MAMBA370_INTERVENTION_LAYER == 35, "M370_LAYER")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")


def validate_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sums = (ROOT / COHORT_SUMS_FILE).read_text(encoding="utf-8").splitlines()
    parsed: dict[str, str] = {}
    for line in sums:
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        parsed[name] = digest
    require(
        parsed == {
            "compatible_cohort.jsonl": COHORT_SHA256,
            "token_gate_manifest.json": MANIFEST_SHA256,
        },
        "COHORT_SUMS",
    )
    require(sha256_file(ROOT / COHORT_FILE) == COHORT_SHA256, "COHORT_SHA")
    require(sha256_file(ROOT / MANIFEST_FILE) == MANIFEST_SHA256, "MANIFEST_SHA")

    manifest = json.loads(
        (ROOT / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    require(manifest["result"] == "PASS_462_OF_462", "GATE_RESULT")
    require(manifest["compatible_count"] == N, "GATE_N")
    require(manifest["token_gate"]["pass_count"] == N, "GATE_PASS")
    require(manifest["token_gate"]["fail_count"] == 0, "GATE_FAIL")
    require(
        manifest["primary_external_transfer_family"] == {
            "historical_behavioral_p_values_in_family": 0,
            "mamba14b_in_family": False,
            "multiplicity": "holm",
            "primary_p_value_count_planned": 2,
            "scale_order": ["mamba130m", "mamba370m"],
        },
        "PRIMARY_FAMILY",
    )
    require(
        manifest["anchor_contract"]["anchor_name"] == ANCHOR_NAME
        and manifest["anchor_contract"]["target_offset"] == TARGET_OFFSET,
        "ANCHOR_CONTRACT",
    )
    require(
        manifest["shared_tokenizer_bytes_across_primary_scales"] is True,
        "SHARED_TOKENIZER",
    )

    rows = read_jsonl(ROOT / COHORT_FILE)
    require(len(rows) == N, "COHORT_N")
    require(len({str(row["example_id"]) for row in rows}) == N, "COHORT_IDS")

    counts = Counter(str(row["source_label"]) for row in rows)
    require(
        dict(counts)
        == {
            "Refuted": 305,
            "Supported": 122,
            "Not Enough Evidence": 35,
        },
        f"LABEL_COUNTS:{dict(counts)}",
    )
    for row in rows:
        require(row["token_gate_pass"] is True, "ROW_GATE")
        require(row["token_gate_reasons"] == [], "ROW_GATE_REASONS")
        require(row["anchor_name"] == ANCHOR_NAME, "ROW_ANCHOR")
        require(int(row["target_offset"]) == TARGET_OFFSET, "ROW_OFFSET")
        require(
            int(row["target_intervention_token_index"])
            == int(row["absolute_anchor_token_index"]) + TARGET_OFFSET,
            "ROW_TARGET",
        )
        require(int(row["correct_label_id"]) in (0, 1, 2), "ROW_LABEL_ID")
        require(
            str(row["correct_label"])
            in ("REFUTE", "NOT_ENTITLED", "SUPPORT"),
            "ROW_LABEL",
        )
    return rows, manifest


def validate_mamba130_snapshot(snapshot: Path) -> None:
    require(snapshot.is_dir(), f"M130_SNAPSHOT:{snapshot}")
    for filename, expected in MAMBA130_REQUIRED_SNAPSHOT_SHA256.items():
        path = snapshot / filename
        require(path.is_file(), f"M130_SNAPSHOT_FILE:{filename}")
        require(sha256_file(path) == expected, f"M130_SNAPSHOT_SHA:{filename}")
    require(
        (snapshot / "config.json").stat().st_size == MAMBA130_CONFIG_BYTES,
        "M130_CONFIG_BYTES",
    )


def authenticate_mamba130_checkpoint(path: Path) -> None:
    require(path.is_file(), "M130_CHECKPOINT_MISSING")
    require(path.stat().st_size == MAMBA130_CHECKPOINT_BYTES, "M130_CHECKPOINT_BYTES")
    require(
        sha256_file(path) == MAMBA130_CHECKPOINT_SHA256,
        "M130_CHECKPOINT_SHA",
    )
    require(
        bridge130.adapter.expected_checkpoint_sha256(181, "G3-GROUP-D-HALF")
        == MAMBA130_CHECKPOINT_SHA256,
        "M130_CHECKPOINT_REGISTRY",
    )


def model_rows(
    cohort: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cohort:
        example_id = str(row["example_id"])
        out.append({
            "row_id": example_id,
            "source_pair_id": example_id,
            "contrast_cell_id": "AVERITEC",
            "claim": str(row["claim"]),
            "evidence": str(row["evidence"]),
        })
    require(len(out) == N, "MODEL_ROWS_N")
    return out


def validate_encoding(
    *,
    cohort: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
) -> None:
    require(tuple(encoded["input_ids"].shape) == (N, 128), "INPUT_SHAPE")
    require(tuple(encoded["attention_mask"].shape) == (N, 128), "ATTN_SHAPE")
    require(tuple(encoded["claim_mask"].shape) == (N, 128), "CLAIM_SHAPE")
    require(tuple(encoded["evidence_mask"].shape) == (N, 128), "EVIDENCE_SHAPE")

    for index, row in enumerate(cohort):
        attention = encoded["attention_mask"][index]
        claim = encoded["claim_mask"][index]
        evidence = encoded["evidence_mask"][index]

        attended = int(attention.sum().item())
        claim_count = int(claim.sum().item())
        evidence_count = int(evidence.sum().item())
        anchor = int(row["absolute_anchor_token_index"])
        target = int(row["target_intervention_token_index"])

        require(attended == int(row["serialized_attended_length"]), f"ATTENDED:{index}")
        require(claim_count == int(row["claim_consumed_token_count"]), f"CLAIM_COUNT:{index}")
        require(evidence_count == int(row["evidence_consumed_token_count"]), f"EVIDENCE_COUNT:{index}")
        require(anchor == claim_count, f"ANCHOR_INDEX:{index}")
        require(target == anchor + TARGET_OFFSET, f"TARGET_INDEX:{index}")
        require(target < attended, f"TARGET_ATTENDED:{index}")
        require(bool(evidence[target].item()), f"TARGET_EVIDENCE:{index}")


def feature_batch(
    encoded: Mapping[str, Any],
    index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "claim_mask", "evidence_mask"):
        tensor = encoded[key]
        require(torch.is_tensor(tensor), f"FEATURE:{key}")
        out[key] = (
            tensor[index:index + 1]
            .detach()
            .to(device)
            .contiguous()
        )
    return out


def correct_margin(logits: Sequence[float], label_id: int) -> float:
    require(len(logits) == 3 and label_id in (0, 1, 2), "MARGIN_INPUT")
    correct = float(logits[label_id])
    wrong = max(float(logits[i]) for i in range(3) if i != label_id)
    value = correct - wrong
    require(math.isfinite(value), "MARGIN_NONFINITE")
    return value


def build_input_130(
    *,
    snapshot: Path,
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    validate_mamba130_snapshot(snapshot)
    tokenizer, tokenizer_provenance = (
        bridge130.tokenizer_gate.load_canonical_analysis_tokenizer(snapshot)
    )
    rows = model_rows(cohort)
    encoded = bridge130.adapter.encode_gen4_rows(rows, tokenizer)
    validate_encoding(cohort=cohort, encoded=encoded)
    return rows, encoded, tokenizer_provenance


def build_input_370(
    *,
    snapshot: Path,
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = bridge_scale.scale_spec("mamba370m")
    spec["geom"].validate_snapshot(snapshot)
    tokenizer, tokenizer_provenance = spec["geom"].load_tokenizer(snapshot)
    rows = model_rows(cohort)
    encoded = spec["adapter"].encode_gen4_rows(rows, tokenizer)
    validate_encoding(cohort=cohort, encoded=encoded)
    return rows, encoded, tokenizer_provenance


def serialize_output(
    output: Mapping[str, Any],
    *,
    scale: str,
    checkpoint_sha256: str,
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    condition: str,
    intervention_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    serialized = bridge_scale.serialize_output(
        output,
        row=model_row,
        scale=scale,
        checkpoint_sha256=checkpoint_sha256,
    )
    label_id = int(cohort_row["correct_label_id"])
    serialized.update({
        "schema_version": ROW_SCHEMA,
        "condition": condition,
        "averitec_dev_index": int(cohort_row["averitec_dev_index"]),
        "example_id": str(cohort_row["example_id"]),
        "source_label": str(cohort_row["source_label"]),
        "correct_label_id": label_id,
        "correct_label": str(cohort_row["correct_label"]),
        "is_correct": int(serialized["prediction_id"]) == label_id,
        "correct_class_logit_margin":
            correct_margin(serialized["final_logits"], label_id),
        "anchor_name": ANCHOR_NAME,
        "absolute_anchor_token_index":
            int(cohort_row["absolute_anchor_token_index"]),
        "target_intervention_token_index":
            int(cohort_row["target_intervention_token_index"]),
        "selected_dominant_candidate": "P3",
        "response_blind_control_plane": "P5",
        "intervention_audit":
            None if intervention_audit is None else dict(intervention_audit),
        "scientific_full_model_forward_count": 1,
    })
    return serialized


def run_condition_130(
    *,
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    planes: Mapping[str, torch.Tensor],
    encoded: Mapping[str, Any],
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    row_index: int,
    condition: str,
    device: torch.device,
) -> dict[str, Any]:
    mapping = {
        "dominant_neutralized": "pp3_neutralized",
        "dominant_control": "pp5_replacement",
    }
    handle = None
    audit: dict[str, Any] | None = None

    if condition != "native":
        audit = {}
        handle = bridge130.install_behavior_hook(
            runtime_ctx["mixer17"],
            token_index=int(cohort_row["target_intervention_token_index"]),
            strong_mask=runtime_ctx["strong_mask"],
            condition=mapping[condition],
            planes=planes,
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = bridge130.adapter.historical_forward(
                model,
                feature_batch(encoded, row_index, device),
                arm="G3-GROUP-D-HALF",
            )
    finally:
        if handle is not None:
            handle.remove()

    return serialize_output(
        output,
        scale="mamba130m",
        checkpoint_sha256=MAMBA130_CHECKPOINT_SHA256,
        model_row=model_row,
        cohort_row=cohort_row,
        condition=condition,
        intervention_audit=audit,
    )


def run_condition_370(
    *,
    spec: Mapping[str, Any],
    model: torch.nn.Module,
    runtime_ctx: Mapping[str, Any],
    frozen: Mapping[str, Any],
    encoded: Mapping[str, Any],
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    row_index: int,
    condition: str,
    device: torch.device,
) -> dict[str, Any]:
    handle = None
    audit: dict[str, Any] | None = None

    if condition != "native":
        audit = {}
        handle = bridge_scale.install_behavior_hook(
            runtime_ctx["intervention_mixer"],
            token_index=int(cohort_row["target_intervention_token_index"]),
            strong_mask=runtime_ctx["strong_mask"],
            condition=condition,
            planes=frozen["planes"],
            selected_plane=MAMBA370_SELECTED,
            control_plane=MAMBA370_CONTROL,
            dim=int(spec["dim"]),
            intermediate_size=int(spec["geom"].INTERMEDIATE_SIZE),
            tol=float(spec["confirmation"].TOL),
            cast_tol=float(
                spec["confirmation"].transport_runtime.RUNTIME_CAST_TOL
            ),
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = spec["adapter"].historical_forward(
                model,
                feature_batch(encoded, row_index, device),
                arm=spec["geom"].ARM,
            )
    finally:
        if handle is not None:
            handle.remove()

    return serialize_output(
        output,
        scale="mamba370m",
        checkpoint_sha256=str(spec["checkpoint_sha256"]),
        model_row=model_row,
        cohort_row=cohort_row,
        condition=condition,
        intervention_audit=audit,
    )


def write_scale_output(
    output_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=False, exist_ok=False)

    rows_raw = jsonl_bytes(rows)
    summary_raw = pretty_json_bytes(summary)

    (output_dir / ROW_FILE).write_bytes(rows_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: hashlib.sha256(rows_raw).hexdigest(),
        SUMMARY_FILE: hashlib.sha256(summary_raw).hexdigest(),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def worker_130(
    *,
    snapshot: str,
    checkpoint: str,
    output_dir: str,
    expected_head: str,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        require(torch.cuda.is_available(), "M130_CUDA")
        require(torch.cuda.device_count() == 1, "M130_CUDA_COUNT")
        device = torch.device("cuda:0")

        snapshot_path = Path(snapshot)
        checkpoint_path = Path(checkpoint)
        authenticate_mamba130_checkpoint(checkpoint_path)
        cohort, manifest = validate_cohort()
        model_rows_value, encoded, tokenizer_provenance = build_input_130(
            snapshot=snapshot_path,
            cohort=cohort,
        )
        planes = bridge130.load_seed181_planes()

        runtime = bridge130.restoration.holdout.phase1.base.prevalence_eq
        runtime.backend.runtime_gate()
        kernels = runtime.kernel_compat.load_exact_fast_kernels()

        with bridge130.seed181.seed181_binding():
            with runtime.backend.parent_runtime_rebind():
                with runtime.kernel_compat.exact_transformers_kernel_loader(
                    kernels
                ) as calls:
                    model, checkpoint_sha = (
                        runtime.parent.load_representative_model_external(
                            model_snapshot=snapshot_path,
                            checkpoint_path=checkpoint_path,
                        )
                    )
                    require(
                        checkpoint_sha == MAMBA130_CHECKPOINT_SHA256,
                        "M130_MODEL_CHECKPOINT",
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
                    "M130_KERNEL_CONSTRUCTOR",
                )
                runtime.kernel_compat.validate_transformers_kernel_bindings(
                    kernels
                )
                model.to(device)
                model.eval()

                output_rows: list[dict[str, Any]] = []
                for index, (cohort_row, model_row) in enumerate(
                    zip(cohort, model_rows_value, strict=True)
                ):
                    for condition in CONDITIONS:
                        output_rows.append(
                            run_condition_130(
                                model=model,
                                runtime_ctx=runtime_ctx,
                                planes=planes,
                                encoded=encoded,
                                model_row=model_row,
                                cohort_row=cohort_row,
                                row_index=index,
                                condition=condition,
                                device=device,
                            )
                        )

                torch.cuda.synchronize(device)

        require(len(output_rows) == FORWARDS_PER_SCALE, "M130_ROW_COUNT")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": expected_head,
            "scale": "mamba130m",
            "physical_device": 0,
            "logical_device": "cuda:0",
            "logical_device_name": torch.cuda.get_device_name(0),
            "hf_repo": MAMBA130_REPO,
            "hf_revision": MAMBA130_REVISION,
            "checkpoint_sha256": MAMBA130_CHECKPOINT_SHA256,
            "checkpoint_bytes": MAMBA130_CHECKPOINT_BYTES,
            "checkpoint_release_tag": MAMBA130_CHECKPOINT_RELEASE_TAG,
            "checkpoint_release_id": MAMBA130_CHECKPOINT_RELEASE_ID,
            "checkpoint_release_asset_id": MAMBA130_CHECKPOINT_ASSET_ID,
            "checkpoint_release_asset_name": MAMBA130_CHECKPOINT_ASSET_NAME,
            "selected_dominant_candidate": MAMBA130_SELECTED,
            "response_blind_control_plane": MAMBA130_CONTROL,
            "intervention_layer": MAMBA130_INTERVENTION_LAYER,
            "anchor_name": ANCHOR_NAME,
            "target_offset": TARGET_OFFSET,
            "condition_order": list(CONDITIONS),
            "compatible_item_count": N,
            "cohort_sha256": COHORT_SHA256,
            "token_gate_manifest_sha256": MANIFEST_SHA256,
            "tokenizer": tokenizer_provenance,
            "scientific_full_model_forward_count_this_run": FORWARDS_PER_SCALE,
            "primary_inference_executed": False,
            "p_value_count_added": 0,
            "scientific_conclusion": None,
            "selection_reopened": False,
            "rescue_performed": False,
            "training_executed": False,
            "backward_executed": False,
            "mamba14b_in_family": False,
        }
        write_scale_output(Path(output_dir), rows=output_rows, summary=summary)

    except BaseException:
        traceback.print_exc()
        raise


def worker_370(
    *,
    snapshot: str,
    output_dir: str,
    expected_head: str,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        spec = bridge_scale.scale_spec("mamba370m")
        device = spec["confirmation"].runtime_gate_single_visible_gpu(1)
        snapshot_path = Path(snapshot)
        spec["geom"].validate_snapshot(snapshot_path)

        checkpoint = ROOT / spec["checkpoint_rel"]
        require(checkpoint.is_file(), "M370_CHECKPOINT")
        require(
            sha256_file(checkpoint) == spec["checkpoint_sha256"],
            "M370_CHECKPOINT_SHA",
        )

        cohort, manifest = validate_cohort()
        model_rows_value, encoded, tokenizer_provenance = build_input_370(
            snapshot=snapshot_path,
            cohort=cohort,
        )

        spec["confirmation"].load_frozen_selection()
        frozen = spec["confirmation"].load_frozen_geometry()

        model, kernels, model_provenance = spec["geom"].reconstruct_model(
            snapshot=snapshot_path,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        spec["confirmation"].kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )
        runtime_ctx = spec["geom"].runtime_components(model)
        spec["confirmation"].validate_runtime_geometry(runtime_ctx, frozen)

        output_rows: list[dict[str, Any]] = []
        for index, (cohort_row, model_row) in enumerate(
            zip(cohort, model_rows_value, strict=True)
        ):
            for condition in CONDITIONS:
                output_rows.append(
                    run_condition_370(
                        spec=spec,
                        model=model,
                        runtime_ctx=runtime_ctx,
                        frozen=frozen,
                        encoded=encoded,
                        model_row=model_row,
                        cohort_row=cohort_row,
                        row_index=index,
                        condition=condition,
                        device=device,
                    )
                )

        torch.cuda.synchronize(device)
        require(len(output_rows) == FORWARDS_PER_SCALE, "M370_ROW_COUNT")

        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "result": RESULT_PASS,
            "execution_head": expected_head,
            "scale": "mamba370m",
            "physical_device": 1,
            "logical_device": "cuda:0",
            "logical_device_name": torch.cuda.get_device_name(0),
            "hf_repo": str(spec["hf_repo"]),
            "hf_revision": str(spec["hf_revision"]),
            "checkpoint_sha256": str(spec["checkpoint_sha256"]),
            "model_provenance": model_provenance,
            "selected_dominant_candidate": MAMBA370_SELECTED,
            "response_blind_control_plane": MAMBA370_CONTROL,
            "intervention_layer": MAMBA370_INTERVENTION_LAYER,
            "anchor_name": ANCHOR_NAME,
            "target_offset": TARGET_OFFSET,
            "condition_order": list(CONDITIONS),
            "compatible_item_count": N,
            "cohort_sha256": COHORT_SHA256,
            "token_gate_manifest_sha256": MANIFEST_SHA256,
            "tokenizer": tokenizer_provenance,
            "scientific_full_model_forward_count_this_run": FORWARDS_PER_SCALE,
            "primary_inference_executed": False,
            "p_value_count_added": 0,
            "scientific_conclusion": None,
            "selection_reopened": False,
            "rescue_performed": False,
            "training_executed": False,
            "backward_executed": False,
            "mamba14b_in_family": False,
        }
        write_scale_output(Path(output_dir), rows=output_rows, summary=summary)

    except BaseException:
        traceback.print_exc()
        raise


def run_both(
    *,
    expected_head: str,
    mamba130m_snapshot: Path,
    mamba130m_checkpoint: Path,
    mamba370m_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    validate_cohort()
    validate_mamba130_snapshot(mamba130m_snapshot)
    authenticate_mamba130_checkpoint(mamba130m_checkpoint)
    bridge_scale.scale_spec("mamba370m")["geom"].validate_snapshot(
        mamba370m_snapshot
    )
    require(not output_dir.exists(), "OUTPUT_ROOT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= 2, "PHYSICAL_GPU_COUNT")
    output_dir.mkdir(parents=True, exist_ok=False)

    ctx = mp.get_context("spawn")
    p130 = ctx.Process(
        target=worker_130,
        kwargs={
            "snapshot": str(mamba130m_snapshot),
            "checkpoint": str(mamba130m_checkpoint),
            "output_dir": str(output_dir / "mamba130m"),
            "expected_head": expected_head,
        },
        name="averitec-mamba130m-gpu0",
    )
    p370 = ctx.Process(
        target=worker_370,
        kwargs={
            "snapshot": str(mamba370m_snapshot),
            "output_dir": str(output_dir / "mamba370m"),
            "expected_head": expected_head,
        },
        name="averitec-mamba370m-gpu1",
    )

    p130.start()
    p370.start()
    p130.join()
    p370.join()

    require(p130.exitcode == 0, f"M130_WORKER_EXIT:{p130.exitcode}")
    require(p370.exitcode == 0, f"M370_WORKER_EXIT:{p370.exitcode}")

    print("RESULT=PASS_AVERITEC_GOLD_EVIDENCE_EXTERNAL_TRANSFER_RAW_BOTH")
    print("PRIMARY_SCALES=mamba130m,mamba370m")
    print("COMPATIBLE_ITEMS_PER_SCALE=462")
    print("CONDITIONS=native,dominant_neutralized,dominant_control")
    print("MAMBA130M_GPU=0")
    print("MAMBA370M_GPU=1")
    print(f"FULL_MODEL_FORWARDS_PER_SCALE={FORWARDS_PER_SCALE}")
    print(f"FULL_MODEL_FORWARDS_TOTAL={TOTAL_FORWARD_BUDGET}")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("MAMBA14B_IN_FAMILY=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen AVeriTeC gold-evidence external causal transfer "
            "raw response collection for Mamba-130M and Mamba-370M."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mamba130m-snapshot", type=Path, required=True)
    parser.add_argument("--mamba130m-checkpoint", type=Path, required=True)
    parser.add_argument("--mamba370m-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_both(
        expected_head=args.expected_head,
        mamba130m_snapshot=args.mamba130m_snapshot,
        mamba130m_checkpoint=args.mamba130m_checkpoint,
        mamba370m_snapshot=args.mamba370m_snapshot,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
