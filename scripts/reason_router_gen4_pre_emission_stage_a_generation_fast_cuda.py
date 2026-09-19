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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import (
    reason_router_gen4_averitec_130m370m_external_transfer_fast_cuda
    as averitec,
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
    reason_router_gen4_pre_emission_causal_lm_identity_grammar_gate
    as identity_gate,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "f3f38b3a96cd2436ffbbe9de0c229a78378abf61"

GATE_ARTIFACT = Path(
    "reports/reason_router_gen4_pre_emission_identity_grammar_gate_runs/"
    "g4k-preemission-causallm-grammar-gate-370m-5fb9f9c-cpu3/"
    "identity_grammar_gate.json"
)
GATE_ARTIFACT_GIT_BLOB = "c95e6095290d5502b6c4b9dad4bf38570c24d8eb"
GATE_MANIFEST_SHA256 = (
    "a7529f96702444ccb1a963b87fa8ffacd37eaf4eef4a26144e47cf94abd61312"
)
GRAMMAR_SHA256 = (
    "ddf5e25a3fcf9d4ddf943438a8a184c2d4a8a7c2c7beb778aa0806e0b8e3906e"
)
BACKBONE_SHA256 = geom.MAMBA_STATE_CANONICAL_SHA256
LM_HEAD_SHA256 = (
    "a473cbd256224b82e4ce004797629aee049c99db483bb5dd05fe7dcc67bdddeb"
)

N = 462
SHARD_COUNT = 2
SHARD_SIZE = N // SHARD_COUNT
SHARDS = (
    {"shard_id": 0, "physical_device": 0, "start": 0, "end": 231},
    {"shard_id": 1, "physical_device": 1, "start": 231, "end": 462},
)

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
EARLY_BLOCK = 35
LATE_BLOCK = 47
DIM = discovery.DIM

CLASS_ORDER = identity_gate.CLASS_ORDER
DECISIVE_CLASSES = identity_gate.DECISIVE_CLASSES
COMMON_PREFIX_LENGTH = 8
DECISIVE_TSTAR_ZERO_BASED = 8
OBSERVATION_OFFSETS = (-4, -3, -2, -1)
OBSERVATION_GENERATED_PREFIX_LENGTHS = tuple(
    DECISIVE_TSTAR_ZERO_BASED + offset + 1
    for offset in OBSERVATION_OFFSETS
)
MAX_GENERATED_TOKENS = 16

PARTITION_SALT = "gen4-pre-emission-stage-a-partition-v1"
PARTITION_ORDER = ("calibration", "confirmatory")
CALIBRATION_COUNTS = {
    "REFUTE": 153,
    "SUPPORT": 61,
    "NOT_ENTITLED": 17,
}
EXPECTED_PARTITION_COUNTS = {
    "calibration": 231,
    "confirmatory": 231,
}
EXPECTED_PARTITION_LABEL_COUNTS = {
    "calibration": {
        "REFUTE": 153,
        "SUPPORT": 61,
        "NOT_ENTITLED": 17,
    },
    "confirmatory": {
        "REFUTE": 152,
        "SUPPORT": 61,
        "NOT_ENTITLED": 18,
    },
}

# Frozen before scientific Stage-A generation outputs are observed.
PRIMARY_STAGE_A_PARTITION = "confirmatory"
PRIMARY_STAGE_A_SIGNAL = "p3_component_l2"
PRIMARY_STAGE_A_OFFSETS = OBSERVATION_OFFSETS
PRIMARY_STAGE_A_TEST = "two_sided_welch_t"
PRIMARY_STAGE_A_MULTIPLICITY = "holm"
PRIMARY_STAGE_A_P_VALUE_COUNT = 4
PRIMARY_STAGE_A_ALPHA = 0.05
PRIMARY_STAGE_A_SUPPORT_RULE = (
    "at_least_one_holm_adjusted_p_below_alpha_at_a_preregistered_pre_emission_offset"
)

ROW_FILE = "stage_a_generation_rows.jsonl"
SUMMARY_FILE = "execution_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"
ROW_SCHEMA = "gen4-pre-emission-stage-a-generation-row-v1"
SUMMARY_SCHEMA = "gen4-pre-emission-stage-a-generation-summary-v1"
RESULT_PASS = "PASS_PRE_EMISSION_STAGE_A_RAW_GENERATION"

LATE_READOUT_REPRO_ATOL = 1.0e-5


class StageAGenerationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StageAGenerationError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageAGenerationError(
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


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


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

    require(
        git("rev-parse", f"HEAD:{GATE_ARTIFACT.as_posix()}")
        == GATE_ARTIFACT_GIT_BLOB,
        "GATE_ARTIFACT_BLOB",
    )

    # Reuse the already-frozen AVeriTeC cohort/blob authentication.
    averitec.authenticate_repo(expected_head)


def validate_protocol() -> None:
    require(N == 462, "N")
    require(SHARD_COUNT == 2 and SHARD_SIZE == 231, "SHARDS")
    require(
        [(s["start"], s["end"]) for s in SHARDS]
        == [(0, 231), (231, 462)],
        "SHARD_RANGES",
    )
    require(SELECTED_PLANE == "P3", "SELECTED_PLANE")
    require(CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(EARLY_BLOCK == 35 and LATE_BLOCK == 47, "SITE_LOCK")
    require(DIM == 650, "DIM")
    require(CLASS_ORDER == ("REFUTE", "NOT_ENTITLED", "SUPPORT"), "CLASS_ORDER")
    require(DECISIVE_CLASSES == ("REFUTE", "SUPPORT"), "DECISIVE_CLASSES")
    require(COMMON_PREFIX_LENGTH == 8, "COMMON_PREFIX_LENGTH")
    require(DECISIVE_TSTAR_ZERO_BASED == 8, "DECISIVE_TSTAR")
    require(OBSERVATION_OFFSETS == (-4, -3, -2, -1), "OBSERVATION_OFFSETS")
    require(
        OBSERVATION_GENERATED_PREFIX_LENGTHS == (5, 6, 7, 8),
        "OBSERVATION_PREFIX_LENGTHS",
    )
    require(MAX_GENERATED_TOKENS == 16, "MAX_GENERATED_TOKENS")
    require(PRIMARY_STAGE_A_PARTITION == "confirmatory", "PRIMARY_PARTITION")
    require(PRIMARY_STAGE_A_SIGNAL == "p3_component_l2", "PRIMARY_SIGNAL")
    require(PRIMARY_STAGE_A_OFFSETS == (-4, -3, -2, -1), "PRIMARY_OFFSETS")
    require(PRIMARY_STAGE_A_TEST == "two_sided_welch_t", "PRIMARY_TEST")
    require(PRIMARY_STAGE_A_MULTIPLICITY == "holm", "PRIMARY_MULTIPLICITY")
    require(PRIMARY_STAGE_A_P_VALUE_COUNT == 4, "PRIMARY_P_VALUE_COUNT")
    require(PRIMARY_STAGE_A_ALPHA == 0.05, "PRIMARY_ALPHA")


def load_gate_artifact() -> dict[str, Any]:
    path = ROOT / GATE_ARTIFACT
    require(path.is_file(), "GATE_ARTIFACT_MISSING")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), "GATE_ARTIFACT_TYPE")

    stored_manifest_sha = str(value.get("manifest_sha256", ""))
    require(stored_manifest_sha == GATE_MANIFEST_SHA256, "GATE_MANIFEST_SHA")
    check = dict(value)
    check.pop("manifest_sha256", None)
    require(
        sha256_bytes(canonical_json_bytes(check)) == GATE_MANIFEST_SHA256,
        "GATE_INTERNAL_MANIFEST_SHA",
    )

    require(
        value.get("result")
        == identity_gate.RESULT_PASS,
        "GATE_RESULT",
    )
    require(value.get("hf_repo") == geom.HF_REPO, "GATE_HF_REPO")
    require(value.get("hf_revision") == geom.HF_REVISION, "GATE_HF_REVISION")

    identity = value["causal_lm_identity"]
    require(identity["status"] == "PASS", "GATE_CAUSAL_LM")
    require(
        identity["backbone_canonical_sha256"] == BACKBONE_SHA256,
        "GATE_BACKBONE_SHA",
    )
    require(identity["lm_head_tied_to_input_embeddings"] is True, "GATE_TIED")
    require(identity["lm_head_weight_sha256"] == LM_HEAD_SHA256, "GATE_LM_HEAD_SHA")
    require(
        identity["input_embedding_weight_sha256"] == LM_HEAD_SHA256,
        "GATE_EMBEDDING_SHA",
    )

    grammar = value["commitment_grammar"]
    require(grammar["grammar_sha256"] == GRAMMAR_SHA256, "GATE_GRAMMAR_SHA")
    require(
        int(grammar["longest_common_prefix_length"]) == COMMON_PREFIX_LENGTH,
        "GATE_COMMON_PREFIX",
    )
    require(
        grammar["decisive_t_star_zero_based"]
        == {"REFUTE": 8, "SUPPORT": 8},
        "GATE_DECISIVE_TSTAR",
    )
    require(
        grammar["surfaces"] == identity_gate.COMMITMENT_SURFACES,
        "GATE_SURFACES",
    )
    require(
        grammar["token_ids"]
        == {
            "NOT_ENTITLED": [
                15545, 327, 253, 1941, 13, 253, 11844, 310,
                5803, 64, 3489, 1433, 23167, 15,
            ],
            "REFUTE": [
                15545, 327, 253, 1941, 13, 253, 11844, 310,
                5689, 39, 23638, 15,
            ],
            "SUPPORT": [
                15545, 327, 253, 1941, 13, 253, 11844, 310,
                9242, 27425, 15,
            ],
        },
        "GATE_TOKEN_IDS",
    )

    require(value["scientific_model_forward_count"] == 0, "GATE_FORWARD_COUNT")
    require(value["scientific_generation_executed"] is False, "GATE_GENERATION")
    require(value["generation_response_inspected"] is False, "GATE_INSPECTION")
    require(value["cuda_executed"] is False, "GATE_CUDA")
    require(value["training_executed"] is False, "GATE_TRAINING")
    require(value["evaluation_executed"] is False, "GATE_EVALUATION")
    return value


def partition_rank_digest(example_id: str, label: str) -> str:
    require(example_id != "", "PARTITION_EXAMPLE_ID")
    require(label in CLASS_ORDER, f"PARTITION_LABEL:{label}")
    raw = (
        PARTITION_SALT
        + "\0"
        + label
        + "\0"
        + example_id
    ).encode("utf-8")
    return sha256_bytes(raw)


def build_partition_assignments(
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    require(len(cohort) == N, "PARTITION_COHORT_N")
    by_label: dict[str, list[tuple[str, str]]] = {
        label: []
        for label in CLASS_ORDER
    }
    seen: set[str] = set()
    for row in cohort:
        example_id = str(row["example_id"])
        label = str(row["correct_label"])
        require(example_id not in seen, f"PARTITION_DUPLICATE:{example_id}")
        seen.add(example_id)
        require(label in by_label, f"PARTITION_LABEL:{label}")
        by_label[label].append(
            (partition_rank_digest(example_id, label), example_id)
        )

    assignments: dict[str, str] = {}
    rank_rows: list[dict[str, Any]] = []
    for label in CLASS_ORDER:
        ranked = sorted(by_label[label])
        calibration_count = CALIBRATION_COUNTS[label]
        require(0 < calibration_count < len(ranked), f"PARTITION_COUNT:{label}")
        for rank_index, (digest, example_id) in enumerate(ranked):
            partition = (
                "calibration"
                if rank_index < calibration_count
                else "confirmatory"
            )
            assignments[example_id] = partition
            rank_rows.append({
                "example_id": example_id,
                "correct_label": label,
                "rank_sha256": digest,
                "rank_within_label_zero_based": rank_index,
                "partition": partition,
            })

    require(len(assignments) == N, "PARTITION_ASSIGNMENT_N")
    partition_counts = Counter(assignments.values())
    require(
        dict(partition_counts) == EXPECTED_PARTITION_COUNTS,
        f"PARTITION_COUNTS:{dict(partition_counts)}",
    )

    label_counts: dict[str, Counter[str]] = {
        partition: Counter()
        for partition in PARTITION_ORDER
    }
    lookup = {
        str(row["example_id"]): str(row["correct_label"])
        for row in cohort
    }
    for example_id, partition in assignments.items():
        label_counts[partition][lookup[example_id]] += 1
    normalized_label_counts = {
        partition: {
            label: int(label_counts[partition][label])
            for label in CLASS_ORDER
        }
        for partition in PARTITION_ORDER
    }
    require(
        normalized_label_counts == EXPECTED_PARTITION_LABEL_COUNTS,
        f"PARTITION_LABEL_COUNTS:{normalized_label_counts}",
    )

    rank_rows = sorted(
        rank_rows,
        key=lambda row: (
            CLASS_ORDER.index(str(row["correct_label"])),
            int(row["rank_within_label_zero_based"]),
        ),
    )
    partition_manifest = {
        "schema_version": "gen4-pre-emission-stage-a-partition-v1",
        "salt": PARTITION_SALT,
        "algorithm": (
            "within_gold_label_sort_by_sha256_salt_nul_label_nul_example_id_"
            "then_fixed_calibration_prefix"
        ),
        "calibration_counts_by_gold_label": dict(CALIBRATION_COUNTS),
        "partition_counts": dict(EXPECTED_PARTITION_COUNTS),
        "partition_label_counts": normalized_label_counts,
        "rank_rows": rank_rows,
    }
    partition_manifest["partition_sha256"] = sha256_bytes(
        canonical_json_bytes(partition_manifest)
    )
    return assignments, partition_manifest


def grammar_token_ids(
    gate_artifact: Mapping[str, Any],
) -> dict[str, list[int]]:
    grammar = gate_artifact["commitment_grammar"]
    out = {
        label: [int(x) for x in grammar["token_ids"][label]]
        for label in CLASS_ORDER
    }
    validated = identity_gate.validate_token_sequences(out)
    require(validated["grammar_sha256"] == GRAMMAR_SHA256, "GRAMMAR_REVALIDATION")
    return out


def completed_label(
    prefix: Sequence[int],
    token_ids: Mapping[str, Sequence[int]],
) -> str | None:
    values = [int(x) for x in prefix]
    matches = [
        label
        for label in CLASS_ORDER
        if values == [int(x) for x in token_ids[label]]
    ]
    require(len(matches) <= 1, "COMPLETE_LABEL_AMBIGUOUS")
    return matches[0] if matches else None


def compatible_labels(
    prefix: Sequence[int],
    token_ids: Mapping[str, Sequence[int]],
) -> tuple[str, ...]:
    values = [int(x) for x in prefix]
    out = []
    for label in CLASS_ORDER:
        sequence = [int(x) for x in token_ids[label]]
        if len(values) <= len(sequence) and sequence[: len(values)] == values:
            out.append(label)
    require(out, "GRAMMAR_PREFIX_NO_COMPATIBLE_LABEL")
    return tuple(out)


def branch_token_by_label(
    token_ids: Mapping[str, Sequence[int]],
) -> dict[str, int]:
    out = {
        label: int(token_ids[label][COMMON_PREFIX_LENGTH])
        for label in CLASS_ORDER
    }
    require(len(set(out.values())) == len(CLASS_ORDER), "BRANCH_TOKENS_NOT_UNIQUE")
    return out


def greedy_allowed_token(
    logits: torch.Tensor,
    allowed: Sequence[int],
) -> tuple[int, dict[str, float]]:
    require(torch.is_tensor(logits) and logits.ndim == 1, "GREEDY_LOGITS_SHAPE")
    ids = tuple(sorted(int(x) for x in allowed))
    require(ids, "GREEDY_ALLOWED_EMPTY")
    require(ids[-1] < logits.shape[0], "GREEDY_TOKEN_RANGE")

    values = {
        str(token_id): float(logits[token_id].detach().cpu().item())
        for token_id in ids
    }
    require(all(math.isfinite(x) for x in values.values()), "GREEDY_NONFINITE")
    best = max(ids, key=lambda token_id: values[str(token_id)])
    tied = [
        token_id
        for token_id in ids
        if values[str(token_id)] == values[str(best)]
    ]
    require(best == min(tied), "GREEDY_TIE_BREAK")
    return best, values


def classify_outcome(gold: str, emitted: str) -> dict[str, Any]:
    require(gold in CLASS_ORDER, f"GOLD:{gold}")
    require(emitted in CLASS_ORDER, f"EMITTED:{emitted}")

    decisive = emitted in DECISIVE_CLASSES
    unsupported = (
        (gold == "NOT_ENTITLED" and decisive)
        or (gold == "SUPPORT" and emitted == "REFUTE")
        or (gold == "REFUTE" and emitted == "SUPPORT")
    )
    supported = (
        (gold == "SUPPORT" and emitted == "SUPPORT")
        or (gold == "REFUTE" and emitted == "REFUTE")
    )
    require(not (unsupported and supported), "OUTCOME_OVERLAP")

    if unsupported:
        category = "unsupported_decisive"
    elif supported:
        category = "supported_decisive"
    elif gold == "NOT_ENTITLED" and emitted == "NOT_ENTITLED":
        category = "nondecisive_correct_abstention"
    else:
        require(emitted == "NOT_ENTITLED", "OUTCOME_UNCLASSIFIED")
        category = "nondecisive_abstention_error"

    return {
        "emitted_is_decisive": decisive,
        "unsupported_decisive_commitment": unsupported,
        "supported_decisive_commitment": supported,
        "primary_stage_a_group": (
            "unsupported"
            if unsupported
            else "supported"
            if supported
            else None
        ),
        "outcome_category": category,
        "t_star_zero_based": (
            DECISIVE_TSTAR_ZERO_BASED
            if decisive
            else None
        ),
    }


def unsupported_branch_margin(
    gold: str,
    branch_logits: Mapping[str, float],
) -> float:
    require(set(branch_logits) == set(CLASS_ORDER), "BRANCH_LOGIT_KEYS")
    values = {label: float(branch_logits[label]) for label in CLASS_ORDER}
    require(all(math.isfinite(x) for x in values.values()), "BRANCH_LOGIT_FINITE")

    if gold == "REFUTE":
        unsupported = values["SUPPORT"]
        nonunsupported = max(values["REFUTE"], values["NOT_ENTITLED"])
    elif gold == "SUPPORT":
        unsupported = values["REFUTE"]
        nonunsupported = max(values["SUPPORT"], values["NOT_ENTITLED"])
    elif gold == "NOT_ENTITLED":
        unsupported = max(values["REFUTE"], values["SUPPORT"])
        nonunsupported = values["NOT_ENTITLED"]
    else:
        raise StageAGenerationError(f"BRANCH_GOLD:{gold}")

    value = unsupported - nonunsupported
    require(math.isfinite(value), "BRANCH_MARGIN_FINITE")
    return value


def decisive_vs_abstain_margin(
    branch_logits: Mapping[str, float],
) -> float:
    require(set(branch_logits) == set(CLASS_ORDER), "DECISIVE_MARGIN_KEYS")
    value = max(
        float(branch_logits["REFUTE"]),
        float(branch_logits["SUPPORT"]),
    ) - float(branch_logits["NOT_ENTITLED"])
    require(math.isfinite(value), "DECISIVE_MARGIN_FINITE")
    return value


def observation_offset_for_generated_prefix_length(length: int) -> int | None:
    if length not in OBSERVATION_GENERATED_PREFIX_LENGTHS:
        return None
    last_emitted_index = length - 1
    offset = last_emitted_index - DECISIVE_TSTAR_ZERO_BASED
    require(offset in OBSERVATION_OFFSETS, "OBSERVATION_OFFSET")
    return offset


def extract_prompt_ids(
    encoded: Mapping[str, Any],
    row_index: int,
    expected_attended_length: int,
) -> torch.Tensor:
    input_ids = encoded["input_ids"][row_index].detach().cpu().contiguous()
    attention = (
        encoded["attention_mask"][row_index]
        .detach().cpu().bool().contiguous()
    )
    require(input_ids.ndim == 1 and attention.ndim == 1, "PROMPT_RANK")
    require(input_ids.shape == attention.shape, "PROMPT_SHAPE")

    attended = int(attention.sum().item())
    require(attended == expected_attended_length, "PROMPT_ATTENDED_LENGTH")
    require(attended > 0, "PROMPT_EMPTY")
    require(bool(attention[:attended].all().item()), "PROMPT_LEFT_CONTIGUOUS")
    require(
        not bool(attention[attended:].any().item()),
        "PROMPT_PADDING_NONCONTIGUOUS",
    )
    prompt = input_ids[:attended].clone().to(torch.long).contiguous()
    require(prompt.numel() == attended, "PROMPT_LENGTH")
    return prompt


def _hook_tensor(output: Any, label: str) -> torch.Tensor:
    value = output
    if isinstance(value, tuple):
        require(bool(value), f"{label}:EMPTY_TUPLE")
        value = value[0]
    require(torch.is_tensor(value), f"{label}:NOT_TENSOR")
    require(
        value.ndim == 3 and value.shape[0] == 1,
        f"{label}:SHAPE:{tuple(value.shape)}",
    )
    require(bool(torch.isfinite(value).all().item()), f"{label}:NONFINITE")
    return value


def early_plane_measurements(
    in_proj_output: torch.Tensor,
    *,
    strong_mask: torch.Tensor,
    planes: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    require(
        torch.is_tensor(in_proj_output)
        and in_proj_output.ndim == 3
        and in_proj_output.shape[0] == 1
        and in_proj_output.shape[-1] == 2 * geom.INTERMEDIATE_SIZE,
        "EARLY_INPROJ_SHAPE",
    )
    mask = strong_mask.detach().cpu().bool().contiguous()
    require(mask.numel() == geom.INTERMEDIATE_SIZE, "EARLY_MASK_WIDTH")
    require(int(mask.sum().item()) == DIM, "EARLY_MASK_COUNT")
    mask_device = mask.to(in_proj_output.device)

    h = (
        in_proj_output[
            0,
            -1,
            :geom.INTERMEDIATE_SIZE,
        ][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )
    require(tuple(h.shape) == (DIM,), "EARLY_H_SHAPE")

    out: dict[str, dict[str, float]] = {}
    for plane in (SELECTED_PLANE, CONTROL_PLANE):
        info = discovery.plane_component(h, plane, planes)
        out[plane] = {
            "a": float(info["a"]),
            "b": float(info["b"]),
            "component_l2": float(info["component_l2"]),
        }
    return out


def runtime_components_for_causal_lm(
    model: Any,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    backbone = getattr(model, "backbone", None)
    require(backbone is not None, "RUNTIME_BACKBONE")
    layers = getattr(backbone, "layers", None)
    require(
        layers is not None and len(layers) == geom.LAYER_COUNT,
        "RUNTIME_LAYERS",
    )
    require(EARLY_BLOCK < len(layers) and LATE_BLOCK < len(layers), "RUNTIME_SITES")

    mixer = getattr(layers[EARLY_BLOCK], "mixer", None)
    require(mixer is not None, "RUNTIME_EARLY_MIXER")
    in_proj = getattr(mixer, "in_proj", None)
    conv1d = getattr(mixer, "conv1d", None)
    require(in_proj is not None and conv1d is not None, "RUNTIME_EARLY_MODULES")
    require(
        tuple(in_proj.weight.shape)
        == (2 * geom.INTERMEDIATE_SIZE, geom.HIDDEN_SIZE),
        "RUNTIME_INPROJ_SHAPE",
    )

    runtime_partition = geom.strong_partition(conv1d.weight)
    runtime_indices = torch.nonzero(
        runtime_partition["mask"],
        as_tuple=False,
    ).flatten().tolist()
    require(
        runtime_indices == list(frozen["strong_indices"]),
        "RUNTIME_STRONG_INDICES",
    )

    norm_f = getattr(backbone, "norm_f", None)
    lm_head = getattr(model, "lm_head", None)
    require(norm_f is not None, "RUNTIME_FINAL_NORM")
    require(lm_head is not None, "RUNTIME_LM_HEAD")

    return {
        "early_in_proj": in_proj,
        "late_layer": layers[LATE_BLOCK],
        "final_norm": norm_f,
        "lm_head": lm_head,
        "strong_mask": runtime_partition["mask"].detach().cpu().bool().contiguous(),
    }


def load_generation_model(
    *,
    snapshot: Path,
    physical_device: int,
) -> tuple[Any, torch.device, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from transformers import MambaConfig, MambaForCausalLM

    device = discovery.runtime_gate_single_visible_gpu(physical_device)
    identity_gate.validate_snapshot_and_config(snapshot)

    snapshot_config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8-sig")
    )
    config = MambaConfig.from_pretrained(snapshot, local_files_only=True)
    tie_compat = identity_gate.apply_legacy_tied_lm_head_compat(
        config,
        snapshot_config=snapshot_config,
    )
    config.use_mamba_kernels = True

    kernels = discovery.kernel_compat.load_exact_fast_kernels()
    with discovery.kernel_compat.exact_transformers_kernel_loader(
        kernels
    ) as calls:
        loaded = MambaForCausalLM.from_pretrained(
            snapshot,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )

    require(
        isinstance(loaded, tuple) and len(loaded) == 2,
        "GENERATION_MODEL_LOAD_RETURN",
    )
    model, loading_info = loaded
    require(isinstance(loading_info, Mapping), "GENERATION_LOADING_INFO")

    counts = Counter(calls)
    require(
        set(counts) == {"causal-conv1d", "mamba-ssm"}
        and counts["causal-conv1d"] > 0
        and counts["causal-conv1d"] == counts["mamba-ssm"],
        f"GENERATION_KERNEL_CONSTRUCTOR:{dict(counts)}",
    )
    discovery.kernel_compat.validate_transformers_kernel_bindings(kernels)

    model.eval()
    model.requires_grad_(False)
    identity = identity_gate.validate_loaded_causal_lm(
        model,
        loading_info=loading_info,
    )
    require(
        identity["backbone_canonical_sha256"] == BACKBONE_SHA256,
        "GENERATION_BACKBONE_SHA",
    )
    require(identity["lm_head_weight_sha256"] == LM_HEAD_SHA256, "GENERATION_LM_HEAD_SHA")
    identity["legacy_tied_lm_head_compat"] = tie_compat
    identity["kernel_constructor_calls"] = dict(counts)

    frozen = discovery.load_frozen_geometry()
    runtime = runtime_components_for_causal_lm(model, frozen)

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    require(
        all(parameter.device == device for parameter in model.parameters()),
        "GENERATION_MODEL_DEVICE",
    )
    return model, device, runtime, frozen, identity


def forward_next_token(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
    observe: bool,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1,
        "FORWARD_INPUT_SHAPE",
    )
    require(input_ids.dtype == torch.long, "FORWARD_INPUT_DTYPE")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    capture: dict[str, Any] = {}
    handles: list[Any] = []

    if observe:
        def early_hook(_module, _args, output):
            value = _hook_tensor(output, "EARLY_INPROJ")
            capture["early"] = early_plane_measurements(
                value,
                strong_mask=runtime["strong_mask"],
                planes=frozen["planes"],
            )
            return None

        def late_hook(_module, _args, output):
            value = _hook_tensor(output, "POST_BLOCK_47")
            capture["post_block_47"] = (
                value[:, -1:, :]
                .detach().clone()
            )
            return None

        handles.append(runtime["early_in_proj"].register_forward_hook(early_hook))
        handles.append(runtime["late_layer"].register_forward_hook(late_hook))

    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()

    logits = output.logits
    require(
        torch.is_tensor(logits)
        and tuple(logits.shape[:2]) == (1, 1),
        f"FORWARD_LOGITS_SHAPE:{tuple(logits.shape)}",
    )
    next_logits = logits[0, -1].detach()

    if not observe:
        return next_logits, None

    require(set(capture) == {"early", "post_block_47"}, "OBSERVATION_CAPTURE_KEYS")
    post47 = capture["post_block_47"]
    with torch.inference_mode():
        normed = runtime["final_norm"](post47)
        manual_logits = runtime["lm_head"](
            normed.to(runtime["lm_head"].weight.dtype)
        ).float()[0, -1]

    require(manual_logits.shape == next_logits.shape, "LATE_LOGIT_SHAPE")
    max_error = float(
        torch.max(torch.abs(manual_logits - next_logits)).detach().cpu().item()
    )
    require(
        max_error <= LATE_READOUT_REPRO_ATOL,
        f"LATE_READOUT_REPRO:{max_error}",
    )

    branch_logits = {
        label: float(manual_logits[int(branch_tokens[label])].detach().cpu().item())
        for label in CLASS_ORDER
    }
    require(all(math.isfinite(x) for x in branch_logits.values()), "BRANCH_LOGITS_NONFINITE")

    observation = {
        "p3_a": capture["early"][SELECTED_PLANE]["a"],
        "p3_b": capture["early"][SELECTED_PLANE]["b"],
        "p3_component_l2":
            capture["early"][SELECTED_PLANE]["component_l2"],
        "p5_a": capture["early"][CONTROL_PLANE]["a"],
        "p5_b": capture["early"][CONTROL_PLANE]["b"],
        "p5_component_l2":
            capture["early"][CONTROL_PLANE]["component_l2"],
        "branch_start_logits": branch_logits,
        "unsupported_branch_margin":
            unsupported_branch_margin(gold_label, branch_logits),
        "decisive_vs_abstain_margin":
            decisive_vs_abstain_margin(branch_logits),
        "late_readout_reproduction_max_abs_error": max_error,
    }
    return next_logits, observation


def generate_one(
    *,
    model: Any,
    device: torch.device,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    prompt_ids: torch.Tensor,
    cohort_row: Mapping[str, Any],
    partition: str,
    token_ids: Mapping[str, Sequence[int]],
    branch_tokens: Mapping[str, int],
) -> dict[str, Any]:
    require(partition in PARTITION_ORDER, f"ROW_PARTITION:{partition}")
    gold = str(cohort_row["correct_label"])
    require(gold in CLASS_ORDER, f"ROW_GOLD:{gold}")

    prompt = prompt_ids.detach().cpu().to(torch.long).contiguous()
    generated: list[int] = []
    observations: list[dict[str, Any]] = []
    forward_count = 0
    branch_choice_allowed_logits: dict[str, float] | None = None

    while True:
        complete = completed_label(generated, token_ids) if generated else None
        if complete is not None:
            emitted = complete
            break

        require(
            len(generated) < MAX_GENERATED_TOKENS,
            "GENERATION_MAX_TOKENS",
        )
        allowed = identity_gate.allowed_next_tokens(generated, token_ids)
        require(allowed, "GENERATION_ALLOWED_EMPTY_BEFORE_COMPLETE")

        combined = torch.cat(
            [
                prompt,
                torch.tensor(generated, dtype=torch.long),
            ]
        ).unsqueeze(0).to(device)

        offset = observation_offset_for_generated_prefix_length(
            len(generated)
        )
        next_logits, observation = forward_next_token(
            model=model,
            runtime=runtime,
            frozen=frozen,
            input_ids=combined,
            gold_label=gold,
            branch_tokens=branch_tokens,
            observe=offset is not None,
        )
        forward_count += 1

        next_token, allowed_logits = greedy_allowed_token(
            next_logits,
            allowed,
        )

        if len(generated) == COMMON_PREFIX_LENGTH:
            require(
                set(int(x) for x in allowed)
                == set(branch_tokens.values()),
                "BRANCH_ALLOWED_SET",
            )
            branch_choice_allowed_logits = {
                label: float(
                    allowed_logits[str(branch_tokens[label])]
                )
                for label in CLASS_ORDER
            }

        if observation is not None:
            require(offset is not None, "OBSERVATION_OFFSET_MISSING")
            observation.update({
                "relative_offset": int(offset),
                "generated_prefix_length": len(generated),
                "last_emitted_token_index_zero_based":
                    len(generated) - 1,
                "current_next_token_is_commitment_branch":
                    len(generated) == COMMON_PREFIX_LENGTH,
            })
            observations.append(observation)

        generated.append(int(next_token))
        compatible_labels(generated, token_ids)

    require(emitted in CLASS_ORDER, "EMITTED_LABEL")
    require(
        generated == [int(x) for x in token_ids[emitted]],
        "EMITTED_TOKEN_SEQUENCE",
    )
    require(
        [row["relative_offset"] for row in observations]
        == list(OBSERVATION_OFFSETS),
        "OBSERVATION_OFFSET_COVERAGE",
    )
    require(
        [row["generated_prefix_length"] for row in observations]
        == list(OBSERVATION_GENERATED_PREFIX_LENGTHS),
        "OBSERVATION_PREFIX_COVERAGE",
    )
    require(branch_choice_allowed_logits is not None, "BRANCH_CHOICE_LOGITS")

    outcome = classify_outcome(gold, emitted)

    return {
        "schema_version": ROW_SCHEMA,
        "averitec_dev_index": int(cohort_row["averitec_dev_index"]),
        "example_id": str(cohort_row["example_id"]),
        "source_label": str(cohort_row["source_label"]),
        "correct_label": gold,
        "correct_label_id": int(cohort_row["correct_label_id"]),
        "partition": partition,
        "prompt_attended_token_count": int(prompt.numel()),
        "generated_token_ids": list(generated),
        "generated_token_count": len(generated),
        "generated_commitment_surface":
            identity_gate.COMMITMENT_SURFACES[emitted],
        "emitted_commitment_label": emitted,
        "branch_decision_token_index_zero_based":
            DECISIVE_TSTAR_ZERO_BASED,
        "branch_choice_allowed_logits": branch_choice_allowed_logits,
        **outcome,
        "observations": observations,
        "scientific_model_forward_count": forward_count,
        "generate_api_call_count": 0,
        "manual_constrained_decoding": True,
        "sampling": False,
        "temperature": None,
        "beam_search": False,
    }


def load_population_and_encoding(
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    cohort, cohort_manifest = averitec.validate_cohort()
    require(len(cohort) == N, "COHORT_N")
    _rows, encoded, tokenizer_provenance = averitec.build_input_370(
        snapshot=snapshot,
        cohort=cohort,
    )
    assignments, partition_manifest = build_partition_assignments(cohort)
    return (
        cohort,
        encoded,
        tokenizer_provenance,
        assignments,
        {
            "cohort_manifest": cohort_manifest,
            "partition_manifest": partition_manifest,
        },
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT:{line_no}")
        rows.append(value)
    return rows


def shard_worker(
    *,
    shard: Mapping[str, int],
    snapshot: str,
    temp_output: str,
    gate_artifact: Mapping[str, Any],
) -> None:
    try:
        physical_device = int(shard["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)

        snapshot_path = Path(snapshot)
        cohort, encoded, tokenizer_provenance, assignments, extra = (
            load_population_and_encoding(snapshot_path)
        )
        token_ids = grammar_token_ids(gate_artifact)
        branch_tokens = branch_token_by_label(token_ids)

        model, device, runtime, frozen, model_identity = (
            load_generation_model(
                snapshot=snapshot_path,
                physical_device=physical_device,
            )
        )

        rows: list[dict[str, Any]] = []
        for index in range(int(shard["start"]), int(shard["end"])):
            cohort_row = cohort[index]
            prompt = extract_prompt_ids(
                encoded,
                index,
                int(cohort_row["serialized_attended_length"]),
            )
            rows.append(
                generate_one(
                    model=model,
                    device=device,
                    runtime=runtime,
                    frozen=frozen,
                    prompt_ids=prompt,
                    cohort_row=cohort_row,
                    partition=assignments[str(cohort_row["example_id"])],
                    token_ids=token_ids,
                    branch_tokens=branch_tokens,
                )
            )

        torch.cuda.synchronize(device)
        require(
            len(rows) == int(shard["end"]) - int(shard["start"]),
            "SHARD_ROW_COUNT",
        )

        payload = {
            "schema_version": "gen4-pre-emission-stage-a-shard-v1",
            "shard": dict(shard),
            "model_identity": model_identity,
            "tokenizer": tokenizer_provenance,
            "partition_sha256":
                extra["partition_manifest"]["partition_sha256"],
            "rows": rows,
        }
        Path(temp_output).write_bytes(canonical_json_bytes(payload))

    except BaseException:
        traceback.print_exc()
        raise


def launch_shards(
    *,
    snapshot: Path,
    temp_dir: Path,
    gate_artifact: Mapping[str, Any],
) -> list[Path]:
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    outputs: list[Path] = []

    for shard in SHARDS:
        out = temp_dir / f"shard{shard['shard_id']}.json"
        outputs.append(out)
        process = ctx.Process(
            target=shard_worker,
            kwargs={
                "shard": shard,
                "snapshot": str(snapshot),
                "temp_output": str(out),
                "gate_artifact": dict(gate_artifact),
            },
            name=f"stage-a-shard-{shard['shard_id']}",
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        require(
            process.exitcode == 0,
            f"SHARD_EXIT:{process.name}:{process.exitcode}",
        )

    require(all(path.is_file() for path in outputs), "SHARD_OUTPUT_MISSING")
    return outputs


def merge_shards(
    *,
    shard_paths: Sequence[Path],
    cohort: Sequence[Mapping[str, Any]],
    partition_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(shard_paths) == SHARD_COUNT, "MERGE_SHARD_COUNT")
    rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    identity_ref: Mapping[str, Any] | None = None

    for shard_id, path in enumerate(shard_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(
            payload["schema_version"]
            == "gen4-pre-emission-stage-a-shard-v1",
            "MERGE_SHARD_SCHEMA",
        )
        require(int(payload["shard"]["shard_id"]) == shard_id, "MERGE_SHARD_ID")
        require(
            payload["partition_sha256"]
            == partition_manifest["partition_sha256"],
            "MERGE_PARTITION_SHA",
        )

        identity = payload["model_identity"]
        if identity_ref is None:
            identity_ref = identity
        else:
            require(
                identity["backbone_canonical_sha256"]
                == identity_ref["backbone_canonical_sha256"],
                "MERGE_BACKBONE_IDENTITY",
            )
            require(
                identity["lm_head_weight_sha256"]
                == identity_ref["lm_head_weight_sha256"],
                "MERGE_LM_HEAD_IDENTITY",
            )

        shard_rows = payload["rows"]
        require(
            len(shard_rows)
            == int(payload["shard"]["end"]) - int(payload["shard"]["start"]),
            "MERGE_SHARD_ROWS",
        )
        rows.extend(shard_rows)
        shard_meta.append({
            "shard": payload["shard"],
            "model_identity": identity,
            "tokenizer": payload["tokenizer"],
            "row_count": len(shard_rows),
        })

    require(len(rows) == N, "MERGE_N")
    expected_ids = [str(row["example_id"]) for row in cohort]
    require(
        [str(row["example_id"]) for row in rows] == expected_ids,
        "MERGE_ROW_ORDER",
    )
    require(len(set(expected_ids)) == N, "MERGE_UNIQUE_IDS")
    return rows, shard_meta


def validate_raw_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == N, "RAW_N")
    outcome_counts = Counter(str(row["outcome_category"]) for row in rows)
    emitted_counts = Counter(str(row["emitted_commitment_label"]) for row in rows)
    partition_counts = Counter(str(row["partition"]) for row in rows)

    require(
        dict(partition_counts) == EXPECTED_PARTITION_COUNTS,
        f"RAW_PARTITIONS:{dict(partition_counts)}",
    )
    for row in rows:
        require(row["schema_version"] == ROW_SCHEMA, "RAW_SCHEMA")
        require(
            [x["relative_offset"] for x in row["observations"]]
            == list(OBSERVATION_OFFSETS),
            "RAW_OBSERVATIONS",
        )
        require(
            row["generate_api_call_count"] == 0
            and row["manual_constrained_decoding"] is True,
            "RAW_DECODER",
        )
        require(row["sampling"] is False, "RAW_SAMPLING")
        require(row["beam_search"] is False, "RAW_BEAM")
        require(
            row["generated_token_count"]
            == row["scientific_model_forward_count"],
            "RAW_FORWARD_COUNT",
        )

    total_forwards = sum(
        int(row["scientific_model_forward_count"])
        for row in rows
    )
    require(total_forwards > N, "RAW_FORWARD_TOTAL")
    return {
        "emitted_commitment_counts": dict(sorted(emitted_counts.items())),
        "outcome_category_counts": dict(sorted(outcome_counts.items())),
        "partition_counts": dict(partition_counts),
        "scientific_model_forward_count_this_run": total_forwards,
        "branch_decision_count": N,
        "generate_api_call_count": 0,
    }


def write_output(
    *,
    output_dir: Path,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    shard_meta: Sequence[Mapping[str, Any]],
    tokenizer_provenance: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_summary = validate_raw_rows(rows)
    row_raw = jsonl_bytes(rows)
    row_sha = sha256_bytes(row_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "hf_repo": geom.HF_REPO,
        "hf_revision": geom.HF_REVISION,
        "backbone_canonical_sha256": BACKBONE_SHA256,
        "lm_head_weight_sha256": LM_HEAD_SHA256,
        "identity_gate": {
            "artifact": GATE_ARTIFACT.as_posix(),
            "artifact_git_blob": GATE_ARTIFACT_GIT_BLOB,
            "manifest_sha256": GATE_MANIFEST_SHA256,
            "grammar_sha256": GRAMMAR_SHA256,
        },
        "cohort": {
            "count": N,
            "cohort_sha256": averitec.COHORT_SHA256,
            "token_gate_manifest_sha256": averitec.MANIFEST_SHA256,
        },
        "partition": dict(partition_manifest),
        "frozen_sites": {
            "early_block": EARLY_BLOCK,
            "early_plane": SELECTED_PLANE,
            "control_plane": CONTROL_PLANE,
            "late_block": LATE_BLOCK,
        },
        "generation_protocol": {
            "decoder": "manual_full_prefix_finite_grammar_greedy",
            "use_cache": False,
            "generate_api_used": False,
            "sampling": False,
            "temperature": None,
            "beam_search": False,
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "common_prefix_length": COMMON_PREFIX_LENGTH,
            "branch_decision_token_index_zero_based":
                DECISIVE_TSTAR_ZERO_BASED,
            "observation_offsets": list(OBSERVATION_OFFSETS),
            "observation_generated_prefix_lengths":
                list(OBSERVATION_GENERATED_PREFIX_LENGTHS),
        },
        "planned_stage_a_inference_not_executed": {
            "primary_partition": PRIMARY_STAGE_A_PARTITION,
            "primary_signal": PRIMARY_STAGE_A_SIGNAL,
            "relative_offsets": list(PRIMARY_STAGE_A_OFFSETS),
            "test": PRIMARY_STAGE_A_TEST,
            "multiplicity": PRIMARY_STAGE_A_MULTIPLICITY,
            "planned_p_value_count": PRIMARY_STAGE_A_P_VALUE_COUNT,
            "alpha": PRIMARY_STAGE_A_ALPHA,
            "support_rule": PRIMARY_STAGE_A_SUPPORT_RULE,
        },
        "late_readout": {
            "source": "post_block_47",
            "readout": "frozen_terminal_norm_plus_tied_lm_head",
            "p3_projected_at_block47": False,
            "reproduction_atol": LATE_READOUT_REPRO_ATOL,
            "branch_start_logits_recorded_at_all_offsets": True,
            "current_next_token_commitment_margin_offset": -1,
            "earlier_offsets_role":
                "prospective_branch_start_readout_only_not_current_grammar_choice",
        },
        "tokenizer": dict(tokenizer_provenance),
        "shards": list(shard_meta),
        **raw_summary,
        "row_file_sha256": row_sha,
        "scientific_generation_executed": True,
        "generation_response_inspected_by_selection_logic": False,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "training_executed": False,
        "backward_executed": False,
        "selection_reopened": False,
        "layer_scan_executed": False,
        "rescue_performed": False,
        "stage_b_executed": False,
        "stage_c_executed": False,
        "scientific_conclusion": None,
    }

    summary_raw = pretty_json_bytes(summary)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: sha256_bytes(row_raw),
        SUMMARY_FILE: sha256_bytes(summary_raw),
    }
    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute raw Stage-A finite-grammar generation for the frozen "
            "Mamba-370M pre-emission precursor design. No statistical inference."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_protocol()
    authenticate_repo(str(args.expected_head))

    snapshot = args.snapshot.resolve()
    geom.validate_snapshot(snapshot)
    gate_artifact = load_gate_artifact()

    cohort, _encoded, tokenizer_provenance, _assignments, extra = (
        load_population_and_encoding(snapshot)
    )
    partition_manifest = extra["partition_manifest"]

    require(not args.output_dir.exists(), "OUTPUT_COLLISION")

    with tempfile.TemporaryDirectory(
        prefix="gen4_pre_emission_stage_a_"
    ) as temp:
        shard_paths = launch_shards(
            snapshot=snapshot,
            temp_dir=Path(temp),
            gate_artifact=gate_artifact,
        )
        rows, shard_meta = merge_shards(
            shard_paths=shard_paths,
            cohort=cohort,
            partition_manifest=partition_manifest,
        )

    summary = write_output(
        output_dir=args.output_dir,
        expected_head=str(args.expected_head),
        rows=rows,
        shard_meta=shard_meta,
        tokenizer_provenance=tokenizer_provenance,
        partition_manifest=partition_manifest,
    )

    print(f"RESULT={summary['result']}")
    print(f"ROWS={N}")
    print(
        "PARTITION_SHA256="
        + summary["partition"]["partition_sha256"]
    )
    print(
        "EMITTED_COUNTS="
        + json.dumps(
            summary["emitted_commitment_counts"],
            sort_keys=True,
        )
    )
    print(
        "OUTCOME_COUNTS="
        + json.dumps(
            summary["outcome_category_counts"],
            sort_keys=True,
        )
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT="
        + str(summary["scientific_model_forward_count_this_run"])
    )
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("TRAINING_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
