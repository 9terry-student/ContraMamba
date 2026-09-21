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
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import (
    reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda
    as bridge_scale,
)


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_GATE_FREEZE_COMMIT = "c260488e91c610eaaea1d7a194a46003c96660c5"

PLAN_PATH = Path(
    "reports/reason_router_gen4_averitec_"
    "mamba14b_negative_sign_transfer_prospective_plan.md"
)
PLAN_SHA256 = "11cae5cce793f3f27a3d8ec449f3dde1deab6aa162e37c97d6deaf9e04d6c845"

COHORT_ROOT = Path(
    "data/reason_router_gen4_averitec_gold_evidence_mamba14b_boundary_v1"
)
COHORT_FILE = COHORT_ROOT / "compatible_cohort.jsonl"
GATE_MANIFEST_FILE = COHORT_ROOT / "token_gate_manifest.json"
COHORT_SUMS_FILE = COHORT_ROOT / "SHA256SUMS.txt"

COHORT_GIT_BLOB = "1628526a4eaea3e81497e304f7cb83fd4f911ad6"
GATE_MANIFEST_GIT_BLOB = "8bbd6396341e67f5616914a8f16056017beb369a"
COHORT_SUMS_GIT_BLOB = "35f9b926aa82263776b3926c9d0d971377cc5c40"

COHORT_SHA256 = "86da8d1435e5722bef5f35db204d097600f152361b147e84d0bb3bbe714c4f00"
GATE_MANIFEST_SHA256 = "88e9f668ac430c7d59796e4b35da790375f6af92349eca3ab021e3935af2e604"

SCALE = "mamba14b"
N = 462
CONDITIONS = (
    "native",
    "dominant_neutralized",
    "dominant_control",
)
SELECTED_PLANE = "P5"
CONTROL_PLANE = "P4"
INTERVENTION_LAYER = 35
ANCHOR_NAME = "A_CLAIM_EVIDENCE_BOUNDARY"
TARGET_OFFSET = 2
CHECKPOINT_SHA256 = (
    "915c9de38d9dc7ee9da26ba4328e74549864c6bd29723f3b7a4b4e0050efce0a"
)
HF_REPO = "state-spaces/mamba-1.4b-hf"
HF_REVISION = "6e46eae61c27280517feef46f536d16b91076f08"

GPU_COUNT = 2
ITEMS_PER_SHARD = N // GPU_COUNT
FORWARDS_PER_ITEM = len(CONDITIONS)
FORWARDS_PER_SHARD = ITEMS_PER_SHARD * FORWARDS_PER_ITEM
TOTAL_FORWARD_BUDGET = N * FORWARDS_PER_ITEM

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start_index": 0,
        "end_index": 231,
        "item_count": 231,
        "forward_budget": 693,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start_index": 231,
        "end_index": 462,
        "item_count": 231,
        "forward_budget": 693,
    },
)

ROW_FILE = "external_transfer_rows.jsonl"
SUMMARY_FILE = "raw_external_transfer_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-averitec-mamba14b-negative-sign-raw-row-v1"
SUMMARY_SCHEMA = "gen4-averitec-mamba14b-negative-sign-raw-summary-v1"
MANIFEST_SCHEMA = "gen4-averitec-mamba14b-negative-sign-raw-manifest-v1"
RESULT_PASS = "PASS_GEN4_MAMBA14B_AVERITEC_NEGATIVE_SIGN_RAW"

PRIMARY_ENDPOINT_RESERVED = "D_EXT_14B=M_native-M_control"
PRIMARY_ALTERNATIVE_RESERVED = "less"
PRIMARY_P_VALUE_COUNT_RESERVED = 1


class ExternalTransfer14BError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ExternalTransfer14BError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalTransfer14BError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_bytes(path: Path) -> bytes:
    rel = path.as_posix()
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalTransfer14BError(
            f"GIT_BLOB_READ:{rel}"
        ) from exc


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


def jsonl_from_bytes(raw: bytes, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{name}:{line_no}")
        rows.append(value)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")

    rc = subprocess.call(
        [
            "git", "merge-base", "--is-ancestor",
            REQUIRED_GATE_FREEZE_COMMIT,
            expected_head,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "GATE_FREEZE_NOT_ANCESTOR")

    frozen_blobs = {
        COHORT_FILE.as_posix(): COHORT_GIT_BLOB,
        GATE_MANIFEST_FILE.as_posix(): GATE_MANIFEST_GIT_BLOB,
        COHORT_SUMS_FILE.as_posix(): COHORT_SUMS_GIT_BLOB,
    }
    for path, expected_blob in frozen_blobs.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )

    plan_raw = git_blob_bytes(PLAN_PATH)
    require(sha256_bytes(plan_raw) == PLAN_SHA256, "PLAN_SHA256")


def validate_protocol() -> None:
    require(SCALE == "mamba14b", "SCALE")
    require(N == 462, "N")
    require(
        CONDITIONS == (
            "native",
            "dominant_neutralized",
            "dominant_control",
        ),
        "CONDITIONS",
    )
    require(SELECTED_PLANE == "P5", "SELECTED")
    require(CONTROL_PLANE == "P4", "CONTROL")
    require(INTERVENTION_LAYER == 35, "LAYER")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(GPU_COUNT == 2, "GPU_COUNT")
    require(ITEMS_PER_SHARD == 231, "ITEMS_PER_SHARD")
    require(FORWARDS_PER_ITEM == 3, "FORWARDS_PER_ITEM")
    require(FORWARDS_PER_SHARD == 693, "FORWARDS_PER_SHARD")
    require(TOTAL_FORWARD_BUDGET == 1386, "TOTAL_FORWARD_BUDGET")

    covered: list[int] = []
    for expected_id, shard in enumerate(SHARDS):
        require(int(shard["shard_id"]) == expected_id, "SHARD_ID")
        require(int(shard["physical_device"]) == expected_id, "SHARD_GPU")
        require(
            int(shard["end_index"]) - int(shard["start_index"])
            == int(shard["item_count"]),
            "SHARD_ITEM_COUNT",
        )
        require(
            int(shard["item_count"]) * FORWARDS_PER_ITEM
            == int(shard["forward_budget"]),
            "SHARD_FORWARD_BUDGET",
        )
        covered.extend(
            range(
                int(shard["start_index"]),
                int(shard["end_index"]),
            )
        )
    require(covered == list(range(N)), "SHARD_COVERAGE")


def validate_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sums_raw = git_blob_bytes(COHORT_SUMS_FILE)
    parsed: dict[str, str] = {}
    for line in sums_raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        require(name not in parsed, f"SUM_DUPLICATE:{name}")
        parsed[name] = digest
    require(
        parsed == {
            "compatible_cohort.jsonl": COHORT_SHA256,
            "token_gate_manifest.json": GATE_MANIFEST_SHA256,
        },
        "COHORT_SUMS",
    )

    cohort_raw = git_blob_bytes(COHORT_FILE)
    manifest_raw = git_blob_bytes(GATE_MANIFEST_FILE)
    require(sha256_bytes(cohort_raw) == COHORT_SHA256, "COHORT_SHA")
    require(
        sha256_bytes(manifest_raw) == GATE_MANIFEST_SHA256,
        "GATE_MANIFEST_SHA",
    )

    manifest = json.loads(manifest_raw.decode("utf-8"))
    require(manifest["result"] == "PASS_462_OF_462", "GATE_RESULT")
    require(int(manifest["compatible_count"]) == N, "GATE_N")
    require(int(manifest["token_gate"]["pass_count"]) == N, "GATE_PASS")
    require(int(manifest["token_gate"]["fail_count"]) == 0, "GATE_FAIL")
    require(
        manifest["token_gate"]["failure_counts"] == {},
        "GATE_FAILURE_COUNTS",
    )

    ext = manifest["primary_external_transfer_extension"]
    require(ext["scale"] == SCALE, "GATE_SCALE")
    require(
        ext["primary_endpoint"] == PRIMARY_ENDPOINT_RESERVED,
        "GATE_ENDPOINT",
    )
    require(
        ext["alternative"] == PRIMARY_ALTERNATIVE_RESERVED,
        "GATE_ALTERNATIVE",
    )
    require(
        int(ext["primary_p_value_count_planned"])
        == PRIMARY_P_VALUE_COUNT_RESERVED,
        "GATE_P_COUNT",
    )
    require(ext["multiplicity"] == "none", "GATE_MULTIPLICITY")
    require(
        ext["historical_130m370m_averitec_family_reopened"] is False,
        "GATE_HISTORICAL_FAMILY",
    )
    require(
        int(ext["historical_p_values_in_new_family"]) == 0,
        "GATE_HISTORICAL_P",
    )

    causal = manifest["frozen_causal_object"]
    require(causal["scale"] == SCALE, "CAUSAL_SCALE")
    require(causal["checkpoint_sha256"] == CHECKPOINT_SHA256, "CAUSAL_CKPT")
    require(causal["selected_plane"] == SELECTED_PLANE, "CAUSAL_SELECTED")
    require(causal["control_plane"] == CONTROL_PLANE, "CAUSAL_CONTROL")
    require(int(causal["intervention_layer"]) == INTERVENTION_LAYER, "CAUSAL_LAYER")
    require(int(causal["target_offset"]) == TARGET_OFFSET, "CAUSAL_OFFSET")

    tokenizer = manifest["tokenizer"]
    require(tokenizer["repo"] == HF_REPO, "TOKENIZER_REPO")
    require(tokenizer["revision"] == HF_REVISION, "TOKENIZER_REVISION")
    require(tokenizer["tokenizers_version"] == "0.22.2", "TOKENIZER_VERSION")

    rows = jsonl_from_bytes(cohort_raw, COHORT_FILE.as_posix())
    require(len(rows) == N, "COHORT_N")
    require(len({str(row["example_id"]) for row in rows}) == N, "COHORT_IDS")

    label_counts = Counter(str(row["source_label"]) for row in rows)
    require(
        dict(label_counts)
        == {
            "Refuted": 305,
            "Supported": 122,
            "Not Enough Evidence": 35,
        },
        f"LABEL_COUNTS:{dict(label_counts)}",
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
        require(
            0
            <= int(row["target_intervention_token_index"])
            < int(row["serialized_attended_length"]),
            "ROW_TARGET_ATTENDED",
        )
        require(int(row["correct_label_id"]) in (0, 1, 2), "ROW_LABEL_ID")

    return rows, manifest


def model_rows(
    cohort: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out = [
        {
            "row_id": str(row["example_id"]),
            "source_pair_id": str(row["example_id"]),
            "contrast_cell_id": "AVERITEC",
            "claim": str(row["claim"]),
            "evidence": str(row["evidence"]),
        }
        for row in cohort
    ]
    require(len(out) == N, "MODEL_ROWS_N")
    return out


def validate_encoding(
    *,
    cohort: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
) -> None:
    for key in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
        require(
            torch.is_tensor(encoded[key])
            and tuple(encoded[key].shape) == (N, 128),
            f"ENCODED:{key}",
        )

    for index, row in enumerate(cohort):
        attention = encoded["attention_mask"][index]
        claim = encoded["claim_mask"][index]
        evidence = encoded["evidence_mask"][index]

        attended = int(attention.sum().item())
        claim_count = int(claim.sum().item())
        evidence_count = int(evidence.sum().item())
        anchor = int(row["absolute_anchor_token_index"])
        target = int(row["target_intervention_token_index"])

        require(
            attended == int(row["serialized_attended_length"]),
            f"ATTENDED:{index}",
        )
        require(
            claim_count == int(row["claim_consumed_token_count"]),
            f"CLAIM:{index}",
        )
        require(
            evidence_count == int(row["evidence_consumed_token_count"]),
            f"EVIDENCE:{index}",
        )
        require(anchor == claim_count, f"ANCHOR:{index}")
        require(target == anchor + TARGET_OFFSET, f"TARGET:{index}")
        require(target < attended, f"TARGET_ATTENDED:{index}")
        require(bool(evidence[target].item()), f"TARGET_EVIDENCE:{index}")


def build_input(
    *,
    spec: Mapping[str, Any],
    snapshot: Path,
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    geom = spec["geom"]
    geom.validate_snapshot(snapshot)
    tokenizer, tokenizer_provenance = geom.load_tokenizer(snapshot)
    rows = model_rows(cohort)
    encoded = spec["adapter"].encode_gen4_rows(rows, tokenizer)
    validate_encoding(cohort=cohort, encoded=encoded)
    return rows, encoded, tokenizer_provenance


def feature_batch(
    encoded: Mapping[str, Any],
    index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in (
        "input_ids",
        "attention_mask",
        "claim_mask",
        "evidence_mask",
    ):
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


def serialize_output(
    output: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    model_row: Mapping[str, Any],
    cohort_row: Mapping[str, Any],
    condition: str,
    intervention_audit: Mapping[str, Any] | None,
    shard_id: int,
    physical_device: int,
) -> dict[str, Any]:
    serialized = bridge_scale.serialize_output(
        output,
        row=model_row,
        scale=SCALE,
        checkpoint_sha256=CHECKPOINT_SHA256,
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
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "intervention_audit":
            None if intervention_audit is None else dict(intervention_audit),
        "shard_index": shard_id,
        "physical_device": physical_device,
        "scientific_full_model_forward_count": 1,
        "primary_inference_executed": False,
        "p_value_count_executed": 0,
        "training_executed": False,
        "backward_executed": False,
    })
    require("D_EXT" not in serialized and "D_EXT_14B" not in serialized, "ROW_D_EXT")
    return serialized


def run_condition(
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
    shard_id: int,
    physical_device: int,
) -> dict[str, Any]:
    require(condition in CONDITIONS, f"CONDITION:{condition}")
    confirmation = spec["confirmation"]
    geom = spec["geom"]

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
            selected_plane=SELECTED_PLANE,
            control_plane=CONTROL_PLANE,
            dim=int(spec["dim"]),
            intermediate_size=int(geom.INTERMEDIATE_SIZE),
            tol=float(confirmation.TOL),
            cast_tol=float(
                confirmation.transport_runtime.RUNTIME_CAST_TOL
            ),
            audit=audit,
        )

    try:
        with torch.inference_mode():
            output = spec["adapter"].historical_forward(
                model,
                feature_batch(encoded, row_index, device),
                arm=geom.ARM,
            )
    finally:
        if handle is not None:
            handle.remove()

    return serialize_output(
        output,
        spec=spec,
        model_row=model_row,
        cohort_row=cohort_row,
        condition=condition,
        intervention_audit=audit,
        shard_id=shard_id,
        physical_device=physical_device,
    )


def worker_paths(temp_dir: Path, shard_id: int) -> dict[str, Path]:
    return {
        "rows": temp_dir / f"shard_{shard_id}_rows.jsonl",
        "summary": temp_dir / f"shard_{shard_id}_summary.json",
        "error": temp_dir / f"shard_{shard_id}.error.txt",
    }


def worker_run(
    *,
    shard: Mapping[str, Any],
    model_snapshot: str,
    temp_dir: str,
    expected_head: str,
) -> None:
    shard_id = int(shard["shard_id"])
    physical_device = int(shard["physical_device"])
    paths = worker_paths(Path(temp_dir), shard_id)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        spec = bridge_scale.scale_spec(SCALE)
        confirmation = spec["confirmation"]
        geom = spec["geom"]

        require(str(spec["selected_plane"]) == SELECTED_PLANE, "SPEC_SELECTED")
        require(str(spec["control_plane"]) == CONTROL_PLANE, "SPEC_CONTROL")
        require(str(spec["checkpoint_sha256"]) == CHECKPOINT_SHA256, "SPEC_CKPT")
        require(str(spec["hf_repo"]) == HF_REPO, "SPEC_REPO")
        require(str(spec["hf_revision"]) == HF_REVISION, "SPEC_REVISION")
        require(int(geom.INTERVENTION_LAYER) == INTERVENTION_LAYER, "SPEC_LAYER")

        device = confirmation.runtime_gate_single_visible_gpu(
            physical_device
        )
        snapshot = Path(model_snapshot)
        geom.validate_snapshot(snapshot)

        checkpoint = ROOT / spec["checkpoint_rel"]
        require(checkpoint.is_file(), "COMPACT_CHECKPOINT_MISSING")
        require(
            sha256_file(checkpoint) == CHECKPOINT_SHA256,
            "COMPACT_CHECKPOINT_SHA",
        )

        cohort, _gate_manifest = validate_cohort()
        model_rows_value, encoded, tokenizer_provenance = build_input(
            spec=spec,
            snapshot=snapshot,
            cohort=cohort,
        )

        confirmation.load_frozen_selection()
        frozen = confirmation.load_frozen_geometry()

        model, kernels, model_provenance = geom.reconstruct_model(
            snapshot=snapshot,
            compact_checkpoint=checkpoint,
            gpu_id=0,
        )
        confirmation.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )
        runtime_ctx = geom.runtime_components(model)
        confirmation.validate_runtime_geometry(runtime_ctx, frozen)

        output_rows: list[dict[str, Any]] = []
        for index in range(
            int(shard["start_index"]),
            int(shard["end_index"]),
        ):
            cohort_row = cohort[index]
            model_row = model_rows_value[index]
            for condition in CONDITIONS:
                output_rows.append(
                    run_condition(
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
                        shard_id=shard_id,
                        physical_device=physical_device,
                    )
                )

        torch.cuda.synchronize(device)

        require(
            len(output_rows) == int(shard["forward_budget"]),
            "WORKER_ROW_COUNT",
        )
        expected = {
            (str(cohort[index]["example_id"]), condition)
            for index in range(
                int(shard["start_index"]),
                int(shard["end_index"]),
            )
            for condition in CONDITIONS
        }
        observed = {
            (str(row["example_id"]), str(row["condition"]))
            for row in output_rows
        }
        require(observed == expected, "WORKER_COVERAGE")

        paths["rows"].write_bytes(jsonl_bytes(output_rows))
        summary = {
            "shard_id": shard_id,
            "physical_device": physical_device,
            "start_index": int(shard["start_index"]),
            "end_index": int(shard["end_index"]),
            "item_count": int(shard["item_count"]),
            "forward_count": int(shard["forward_budget"]),
            "logical_device": "cuda:0",
            "logical_device_name": torch.cuda.get_device_name(0),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "tokenizer": tokenizer_provenance,
            "model_provenance": model_provenance,
            "execution_head": expected_head,
            "primary_inference_executed": False,
            "p_value_count_executed": 0,
            "training_executed": False,
            "backward_executed": False,
            "scientific_conclusion": None,
        }
        paths["summary"].write_bytes(pretty_json_bytes(summary))

    except BaseException:
        paths["error"].write_text(
            traceback.format_exc(),
            encoding="utf-8",
            newline="\n",
        )
        traceback.print_exc()
        raise


def read_temp_rows(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"TEMP_ROWS_MISSING:{path}")
    return jsonl_from_bytes(path.read_bytes(), str(path))


def merge_worker_outputs(
    *,
    temp_dir: Path,
    expected_head: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []

    for shard in SHARDS:
        shard_id = int(shard["shard_id"])
        paths = worker_paths(temp_dir, shard_id)
        require(not paths["error"].exists(), f"WORKER_ERROR_FILE:{shard_id}")
        rows = read_temp_rows(paths["rows"])
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        require(len(rows) == int(shard["forward_budget"]), f"SHARD_ROWS:{shard_id}")
        require(summary["execution_head"] == expected_head, f"SHARD_HEAD:{shard_id}")
        require(int(summary["p_value_count_executed"]) == 0, f"SHARD_P:{shard_id}")
        require(summary["scientific_conclusion"] is None, f"SHARD_CONCLUSION:{shard_id}")
        all_rows.extend(rows)
        shard_summaries.append(summary)

    condition_order = {name: i for i, name in enumerate(CONDITIONS)}
    all_rows.sort(
        key=lambda row: (
            int(row["averitec_dev_index"]),
            condition_order[str(row["condition"])],
        )
    )

    require(len(all_rows) == TOTAL_FORWARD_BUDGET, "MERGED_ROW_COUNT")
    require(
        len({
            (str(row["example_id"]), str(row["condition"]))
            for row in all_rows
        }) == TOTAL_FORWARD_BUDGET,
        "MERGED_UNIQUENESS",
    )

    cohort, _ = validate_cohort()
    expected = {
        (str(row["example_id"]), condition)
        for row in cohort
        for condition in CONDITIONS
    }
    observed = {
        (str(row["example_id"]), str(row["condition"]))
        for row in all_rows
    }
    require(observed == expected, "MERGED_COVERAGE")
    return all_rows, shard_summaries


def write_outputs(
    *,
    output_dir: Path,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    shard_summaries: Sequence[Mapping[str, Any]],
) -> None:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    require(len(rows) == TOTAL_FORWARD_BUDGET, "OUTPUT_ROWS")
    require(len(shard_summaries) == 2, "OUTPUT_SHARDS")

    output_dir.mkdir(parents=True, exist_ok=False)

    rows_raw = jsonl_bytes(rows)
    rows_sha = sha256_bytes(rows_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "scale": SCALE,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_dominant_candidate": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_layer": INTERVENTION_LAYER,
        "anchor_name": ANCHOR_NAME,
        "target_offset": TARGET_OFFSET,
        "condition_order": list(CONDITIONS),
        "compatible_item_count": N,
        "row_count": TOTAL_FORWARD_BUDGET,
        "cohort_sha256": COHORT_SHA256,
        "token_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "primary_endpoint_reserved_for_static_analysis":
            PRIMARY_ENDPOINT_RESERVED,
        "primary_alternative_reserved_for_static_analysis":
            PRIMARY_ALTERNATIVE_RESERVED,
        "primary_p_value_count_reserved":
            PRIMARY_P_VALUE_COUNT_RESERVED,
        "scientific_full_model_forward_count": TOTAL_FORWARD_BUDGET,
        "backward_count": 0,
        "training_executed": False,
        "parameter_update_executed": False,
        "primary_inference_executed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
        "selection_reopened": False,
        "rescue_performed": False,
        "historical_130m370m_averitec_family_reopened": False,
        "historical_averitec_item_results_accessed": False,
        "shards": [dict(x) for x in shard_summaries],
    }
    summary_raw = pretty_json_bytes(summary)
    summary_sha = sha256_bytes(summary_raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "scale": SCALE,
        "required_gate_freeze_commit": REQUIRED_GATE_FREEZE_COMMIT,
        "prospective_plan_sha256": PLAN_SHA256,
        "input_sha256": {
            "compatible_cohort.jsonl": COHORT_SHA256,
            "token_gate_manifest.json": GATE_MANIFEST_SHA256,
        },
        "output_sha256": {
            ROW_FILE: rows_sha,
            SUMMARY_FILE: summary_sha,
        },
        "compatible_item_count": N,
        "row_count": TOTAL_FORWARD_BUDGET,
        "condition_order": list(CONDITIONS),
        "shard_count": 2,
        "scientific_full_model_forward_count": TOTAL_FORWARD_BUDGET,
        "backward_count": 0,
        "training_executed": False,
        "parameter_update_executed": False,
        "primary_inference_executed": False,
        "p_value_count_executed": 0,
        "scientific_conclusion": None,
        "selection_reopened": False,
        "rescue_performed": False,
        "historical_130m370m_averitec_family_reopened": False,
        "historical_averitec_item_results_accessed": False,
    }
    manifest_raw = pretty_json_bytes(manifest)
    manifest_sha = sha256_bytes(manifest_raw)

    (output_dir / ROW_FILE).write_bytes(rows_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)

    hashes = {
        ROW_FILE: rows_sha,
        SUMMARY_FILE: summary_sha,
        MANIFEST_FILE: manifest_sha,
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )


def run_raw(
    *,
    expected_head: str,
    model_snapshot: Path,
    output_dir: Path,
) -> None:
    validate_protocol()
    authenticate_repo(expected_head)
    validate_cohort()

    spec = bridge_scale.scale_spec(SCALE)
    require(str(spec["selected_plane"]) == SELECTED_PLANE, "TOP_SELECTED")
    require(str(spec["control_plane"]) == CONTROL_PLANE, "TOP_CONTROL")
    require(str(spec["checkpoint_sha256"]) == CHECKPOINT_SHA256, "TOP_CKPT")
    require(str(spec["hf_repo"]) == HF_REPO, "TOP_REPO")
    require(str(spec["hf_revision"]) == HF_REVISION, "TOP_REVISION")
    require(int(spec["geom"].INTERVENTION_LAYER) == INTERVENTION_LAYER, "TOP_LAYER")

    spec["geom"].validate_snapshot(model_snapshot)
    checkpoint = ROOT / spec["checkpoint_rel"]
    require(checkpoint.is_file(), "TOP_CHECKPOINT_MISSING")
    require(sha256_file(checkpoint) == CHECKPOINT_SHA256, "TOP_CHECKPOINT_SHA")

    require(not output_dir.exists(), "OUTPUT_ROOT_COLLISION")
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE")
    require(torch.cuda.device_count() >= GPU_COUNT, "PHYSICAL_GPU_COUNT")

    with tempfile.TemporaryDirectory(
        prefix="contramamba_m14b_averitec_raw_"
    ) as temp_name:
        temp_dir = Path(temp_name)
        ctx = mp.get_context("spawn")
        processes: list[mp.Process] = []

        for shard in SHARDS:
            process = ctx.Process(
                target=worker_run,
                kwargs={
                    "shard": dict(shard),
                    "model_snapshot": str(model_snapshot),
                    "temp_dir": str(temp_dir),
                    "expected_head": expected_head,
                },
                name=f"averitec-mamba14b-shard{shard['shard_id']}",
            )
            processes.append(process)

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        for shard, process in zip(SHARDS, processes, strict=True):
            require(
                process.exitcode == 0,
                f"WORKER_EXIT:{shard['shard_id']}:{process.exitcode}",
            )

        rows, shard_summaries = merge_worker_outputs(
            temp_dir=temp_dir,
            expected_head=expected_head,
        )
        write_outputs(
            output_dir=output_dir,
            expected_head=expected_head,
            rows=rows,
            shard_summaries=shard_summaries,
        )

    print("RESULT=" + RESULT_PASS)
    print("SCALE=mamba14b")
    print("COMPATIBLE_ITEM_COUNT=462")
    print("CONDITIONS=native,dominant_neutralized,dominant_control")
    print("SHARD0_GPU=0")
    print("SHARD1_GPU=1")
    print("ITEMS_PER_SHARD=231")
    print("SCIENTIFIC_FULL_MODEL_FORWARD_COUNT=1386")
    print("BACKWARD_COUNT=0")
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_EXECUTED=0")
    print("SCIENTIFIC_CONCLUSION=None")
    print("HISTORICAL_130M370M_FAMILY_REOPENED=False")
    print("HISTORICAL_AVERITEC_ITEM_RESULTS_ACCESSED=False")
    print("TRAINING_EXECUTED=False")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute raw-only Mamba-1.4B AVeriTeC negative-sign external "
            "transfer response collection on the frozen 462-item token-gated "
            "cohort. No inferential statistics are executed."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_raw(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
