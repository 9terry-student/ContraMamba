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
    reason_router_gen4_pre_emission_stage_a_generation_fast_cuda
    as base,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "e04179dc8cd1454caf8bd71d36114dc0b1da7b5d"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_design.md"
)
DESIGN_ARTIFACT_GIT_BLOB = "b4f9c3ebf2e43a556fbfeb43c5146b4a04f4c620"

BASE_GENERATOR = Path(
    "scripts/reason_router_gen4_pre_emission_stage_a_generation_fast_cuda.py"
)
BASE_GENERATOR_GIT_BLOB = "aa6725bdc1cbfa43547e89293f3b7911a4cdd6ed"

PRIOR_RAW_FREEZE_COMMIT = "a58b1ebef0957a8437e0255fb4069d2fe9cf8449"
PRIOR_CLOSURE_COMMIT = "130aa76cf3bb0816505a45a18df8334445e6f0ce"

N = base.N
SHARDS = base.SHARDS
SHARD_COUNT = base.SHARD_COUNT

GOLD_CLASS_ORDER = base.CLASS_ORDER
FORCED_CLASS_ORDER = ("REFUTE", "SUPPORT")
COMMON_PREFIX_LENGTH = 8
DECISIVE_TSTAR_ZERO_BASED = 8
OBSERVATION_OFFSETS = (-4, -3, -2, -1)
OBSERVATION_GENERATED_PREFIX_LENGTHS = (5, 6, 7, 8)
MAX_GENERATED_TOKENS = 16

EXPECTED_PARTITION_SHA256 = (
    "871fb5c1e2c62f247c284ceae409ef9ec19acc75f63ffc832829c2664db46311"
)

FORCED_SURFACES = {
    "REFUTE": "Based on the evidence, the verdict is REFUTE.",
    "SUPPORT": "Based on the evidence, the verdict is SUPPORT.",
}
FORCED_TOKEN_IDS = {
    "REFUTE": [
        15545, 327, 253, 1941, 13, 253, 11844, 310,
        5689, 39, 23638, 15,
    ],
    "SUPPORT": [
        15545, 327, 253, 1941, 13, 253, 11844, 310,
        9242, 27425, 15,
    ],
}
FORCED_GRAMMAR_SHA256 = (
    "62c9c53871f68fcc0f57d38c96b75d2391ee6ddc473a4325a522f91c9249bd00"
)

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

ROW_FILE = "forced_decisive_stage_a_generation_rows.jsonl"
SUMMARY_FILE = "execution_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-pre-emission-forced-decisive-stage-a-generation-row-v1"
SUMMARY_SCHEMA = (
    "gen4-pre-emission-forced-decisive-stage-a-generation-summary-v1"
)
RESULT_PASS = "PASS_PRE_EMISSION_FORCED_DECISIVE_STAGE_A_RAW_GENERATION"


class ForcedDecisiveStageAError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ForcedDecisiveStageAError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ForcedDecisiveStageAError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
    return b"".join(
        canonical_json_bytes(dict(row))
        for row in rows
    )


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

    pinned = {
        DESIGN_ARTIFACT.as_posix(): DESIGN_ARTIFACT_GIT_BLOB,
        BASE_GENERATOR.as_posix(): BASE_GENERATOR_GIT_BLOB,
        base.GATE_ARTIFACT.as_posix(): base.GATE_ARTIFACT_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )

    # Reuse the frozen cohort/blob authentication without invoking the
    # prior three-class generator's own branch/ancestor policy.
    base.averitec.authenticate_repo(expected_head)


def validate_protocol() -> None:
    require(N == 462, "N")
    require(SHARD_COUNT == 2, "SHARD_COUNT")
    require(
        [(int(x["start"]), int(x["end"])) for x in SHARDS]
        == [(0, 231), (231, 462)],
        "SHARD_RANGES",
    )

    require(
        GOLD_CLASS_ORDER == ("REFUTE", "NOT_ENTITLED", "SUPPORT"),
        "GOLD_CLASS_ORDER",
    )
    require(FORCED_CLASS_ORDER == ("REFUTE", "SUPPORT"), "FORCED_CLASS_ORDER")
    require(
        tuple(FORCED_SURFACES) == FORCED_CLASS_ORDER,
        "FORCED_SURFACE_ORDER",
    )
    require(
        tuple(FORCED_TOKEN_IDS) == FORCED_CLASS_ORDER,
        "FORCED_TOKEN_ORDER",
    )

    require(base.SELECTED_PLANE == "P3", "SELECTED_PLANE")
    require(base.CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(base.EARLY_BLOCK == 35, "EARLY_BLOCK")
    require(base.LATE_BLOCK == 47, "LATE_BLOCK")
    require(base.DIM == 650, "DIM")
    require(base.LATE_READOUT_REPRO_ATOL == 1.0e-5, "LATE_REPRO_ATOL")

    require(COMMON_PREFIX_LENGTH == 8, "COMMON_PREFIX_LENGTH")
    require(DECISIVE_TSTAR_ZERO_BASED == 8, "DECISIVE_TSTAR")
    require(OBSERVATION_OFFSETS == (-4, -3, -2, -1), "OBSERVATION_OFFSETS")
    require(
        OBSERVATION_GENERATED_PREFIX_LENGTHS == (5, 6, 7, 8),
        "OBSERVATION_PREFIX_LENGTHS",
    )
    require(MAX_GENERATED_TOKENS == 16, "MAX_GENERATED_TOKENS")

    require(base.PARTITION_SALT == "gen4-pre-emission-stage-a-partition-v1", "PARTITION_SALT")
    require(
        base.EXPECTED_PARTITION_COUNTS
        == {"calibration": 231, "confirmatory": 231},
        "PARTITION_COUNTS",
    )

    require(PRIMARY_STAGE_A_PARTITION == "confirmatory", "PRIMARY_PARTITION")
    require(PRIMARY_STAGE_A_SIGNAL == "p3_component_l2", "PRIMARY_SIGNAL")
    require(PRIMARY_STAGE_A_OFFSETS == (-4, -3, -2, -1), "PRIMARY_OFFSETS")
    require(PRIMARY_STAGE_A_TEST == "two_sided_welch_t", "PRIMARY_TEST")
    require(PRIMARY_STAGE_A_MULTIPLICITY == "holm", "PRIMARY_MULTIPLICITY")
    require(PRIMARY_STAGE_A_P_VALUE_COUNT == 4, "PRIMARY_P_VALUE_COUNT")
    require(PRIMARY_STAGE_A_ALPHA == 0.05, "PRIMARY_ALPHA")


def forced_grammar_manifest(
    gate_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    grammar = gate_artifact["commitment_grammar"]

    source_tokens = {
        label: [int(x) for x in grammar["token_ids"][label]]
        for label in FORCED_CLASS_ORDER
    }
    source_surfaces = {
        label: str(grammar["surfaces"][label])
        for label in FORCED_CLASS_ORDER
    }

    require(source_tokens == FORCED_TOKEN_IDS, "FORCED_TOKEN_IDS")
    require(source_surfaces == FORCED_SURFACES, "FORCED_SURFACES")

    sequences = [source_tokens[label] for label in FORCED_CLASS_ORDER]
    common_prefix = base.identity_gate.longest_common_prefix_length(
        sequences
    )
    require(common_prefix == COMMON_PREFIX_LENGTH, "FORCED_COMMON_PREFIX")

    for left in FORCED_CLASS_ORDER:
        for right in FORCED_CLASS_ORDER:
            if left == right:
                continue
            require(
                not base.identity_gate.is_complete_prefix(
                    source_tokens[left],
                    source_tokens[right],
                ),
                f"FORCED_COMPLETE_PREFIX:{left}:{right}",
            )

    unique_indices = {
        label: base.identity_gate.unique_commitment_index(
            source_tokens[label],
            source_tokens,
        )
        for label in FORCED_CLASS_ORDER
    }
    require(
        unique_indices == {"REFUTE": 8, "SUPPORT": 8},
        f"FORCED_UNIQUE_INDICES:{unique_indices}",
    )

    manifest = {
        "schema_version": "gen4-pre-emission-forced-decisive-grammar-v1",
        "class_order": list(FORCED_CLASS_ORDER),
        "surfaces": dict(source_surfaces),
        "token_ids": dict(source_tokens),
        "longest_common_prefix_length": COMMON_PREFIX_LENGTH,
        "longest_common_prefix_token_ids":
            list(source_tokens["REFUTE"][:COMMON_PREFIX_LENGTH]),
        "unique_commitment_token_index_zero_based": dict(unique_indices),
        "decisive_t_star_zero_based": dict(unique_indices),
        "minimum_precommit_generated_tokens":
            base.identity_gate.MIN_PRECOMMIT_GENERATED_TOKENS,
        "not_entitled_available": False,
    }
    observed = sha256_bytes(canonical_json_bytes(manifest))
    require(
        observed == FORCED_GRAMMAR_SHA256,
        f"FORCED_GRAMMAR_SHA256:{observed}",
    )
    manifest["grammar_sha256"] = observed
    return manifest


def forced_token_ids(
    grammar_manifest: Mapping[str, Any],
) -> dict[str, list[int]]:
    return {
        label: [int(x) for x in grammar_manifest["token_ids"][label]]
        for label in FORCED_CLASS_ORDER
    }


def completed_label(
    prefix: Sequence[int],
    token_ids: Mapping[str, Sequence[int]],
) -> str | None:
    values = [int(x) for x in prefix]
    matches = [
        label
        for label in FORCED_CLASS_ORDER
        if values == [int(x) for x in token_ids[label]]
    ]
    require(len(matches) <= 1, "COMPLETE_LABEL_AMBIGUOUS")
    return matches[0] if matches else None


def compatible_labels(
    prefix: Sequence[int],
    token_ids: Mapping[str, Sequence[int]],
) -> tuple[str, ...]:
    values = [int(x) for x in prefix]
    out: list[str] = []
    for label in FORCED_CLASS_ORDER:
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
        for label in FORCED_CLASS_ORDER
    }
    require(
        len(set(out.values())) == len(FORCED_CLASS_ORDER),
        "BRANCH_TOKENS_NOT_UNIQUE",
    )
    return out


def classify_forced_outcome(
    gold: str,
    emitted: str,
) -> dict[str, Any]:
    require(gold in GOLD_CLASS_ORDER, f"GOLD:{gold}")
    require(emitted in FORCED_CLASS_ORDER, f"EMITTED:{emitted}")

    supported = (
        (gold == "REFUTE" and emitted == "REFUTE")
        or (gold == "SUPPORT" and emitted == "SUPPORT")
    )
    unsupported = (
        gold == "NOT_ENTITLED"
        or (gold == "REFUTE" and emitted == "SUPPORT")
        or (gold == "SUPPORT" and emitted == "REFUTE")
    )

    require(
        supported != unsupported,
        f"FORCED_OUTCOME_XOR:{gold}:{emitted}",
    )

    return {
        "emitted_is_decisive": True,
        "unsupported_forced_decisive_commitment": unsupported,
        "supported_forced_decisive_commitment": supported,
        "primary_stage_a_group":
            "unsupported" if unsupported else "supported",
        "outcome_category":
            "unsupported_forced_decisive"
            if unsupported
            else "supported_forced_decisive",
        "t_star_zero_based": DECISIVE_TSTAR_ZERO_BASED,
    }


def m47_support_minus_refute(
    branch_logits: Mapping[str, float],
) -> float:
    require(
        tuple(branch_logits) == FORCED_CLASS_ORDER,
        "M47_BRANCH_LOGIT_ORDER",
    )
    refute = float(branch_logits["REFUTE"])
    support = float(branch_logits["SUPPORT"])
    require(
        math.isfinite(refute) and math.isfinite(support),
        "M47_BRANCH_LOGIT_FINITE",
    )
    value = support - refute
    require(math.isfinite(value), "M47_MARGIN_FINITE")
    return value


def forward_next_token(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
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
            value = base._hook_tensor(output, "EARLY_INPROJ")
            capture["early"] = base.early_plane_measurements(
                value,
                strong_mask=runtime["strong_mask"],
                planes=frozen["planes"],
            )
            return None

        def late_hook(_module, _args, output):
            value = base._hook_tensor(output, "POST_BLOCK_47")
            capture["post_block_47"] = value.detach().clone()
            return None

        handles.append(
            runtime["early_in_proj"].register_forward_hook(early_hook)
        )
        handles.append(
            runtime["late_layer"].register_forward_hook(late_hook)
        )

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

    require(
        set(capture) == {"early", "post_block_47"},
        "OBSERVATION_CAPTURE_KEYS",
    )

    manual_logits = base.late_readout_from_post_block_47(
        capture["post_block_47"],
        final_norm=runtime["final_norm"],
        lm_head=runtime["lm_head"],
    )
    require(
        manual_logits.shape == next_logits.shape,
        "LATE_LOGIT_SHAPE",
    )

    max_error = float(
        torch.max(
            torch.abs(manual_logits - next_logits)
        ).detach().cpu().item()
    )
    require(
        max_error <= base.LATE_READOUT_REPRO_ATOL,
        f"LATE_READOUT_REPRO:{max_error}",
    )

    branch_logits = {
        label: float(
            manual_logits[int(branch_tokens[label])]
            .detach().cpu().item()
        )
        for label in FORCED_CLASS_ORDER
    }
    require(
        all(math.isfinite(x) for x in branch_logits.values()),
        "BRANCH_LOGITS_NONFINITE",
    )

    observation = {
        "p3_a":
            capture["early"][base.SELECTED_PLANE]["a"],
        "p3_b":
            capture["early"][base.SELECTED_PLANE]["b"],
        "p3_component_l2":
            capture["early"][base.SELECTED_PLANE]["component_l2"],
        "p5_a":
            capture["early"][base.CONTROL_PLANE]["a"],
        "p5_b":
            capture["early"][base.CONTROL_PLANE]["b"],
        "p5_component_l2":
            capture["early"][base.CONTROL_PLANE]["component_l2"],
        "branch_start_logits": branch_logits,
        "m47_support_minus_refute":
            m47_support_minus_refute(branch_logits),
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
    require(
        partition in base.PARTITION_ORDER,
        f"ROW_PARTITION:{partition}",
    )

    gold = str(cohort_row["correct_label"])
    require(gold in GOLD_CLASS_ORDER, f"ROW_GOLD:{gold}")

    prompt = (
        prompt_ids.detach().cpu().to(torch.long).contiguous()
    )
    generated: list[int] = []
    observations: list[dict[str, Any]] = []
    forward_count = 0
    branch_choice_allowed_logits: dict[str, float] | None = None

    while True:
        complete = (
            completed_label(generated, token_ids)
            if generated
            else None
        )
        if complete is not None:
            emitted = complete
            break

        require(
            len(generated) < MAX_GENERATED_TOKENS,
            "GENERATION_MAX_TOKENS",
        )

        allowed = base.identity_gate.allowed_next_tokens(
            generated,
            token_ids,
        )
        require(
            allowed,
            "GENERATION_ALLOWED_EMPTY_BEFORE_COMPLETE",
        )

        combined = torch.cat([
            prompt,
            torch.tensor(generated, dtype=torch.long),
        ]).unsqueeze(0).to(device)

        offset = base.observation_offset_for_generated_prefix_length(
            len(generated)
        )
        next_logits, observation = forward_next_token(
            model=model,
            runtime=runtime,
            frozen=frozen,
            input_ids=combined,
            branch_tokens=branch_tokens,
            observe=offset is not None,
        )
        forward_count += 1

        next_token, allowed_logits = base.greedy_allowed_token(
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
                for label in FORCED_CLASS_ORDER
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

    require(emitted in FORCED_CLASS_ORDER, "EMITTED_LABEL")
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
    require(
        branch_choice_allowed_logits is not None,
        "BRANCH_CHOICE_LOGITS",
    )

    outcome = classify_forced_outcome(gold, emitted)

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
            FORCED_SURFACES[emitted],
        "emitted_commitment_label": emitted,
        "branch_decision_token_index_zero_based":
            DECISIVE_TSTAR_ZERO_BASED,
        "branch_choice_allowed_logits":
            branch_choice_allowed_logits,
        **outcome,
        "observations": observations,
        "scientific_model_forward_count": forward_count,
        "generate_api_call_count": 0,
        "manual_constrained_decoding": True,
        "forced_decisive_two_class_grammar": True,
        "not_entitled_available": False,
        "sampling": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
        "beam_search": False,
        "repetition_penalty": None,
        "logit_bias": None,
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
    cohort, encoded, tokenizer_provenance, assignments, extra = (
        base.load_population_and_encoding(snapshot)
    )
    partition_manifest = extra["partition_manifest"]
    require(
        partition_manifest["partition_sha256"]
        == EXPECTED_PARTITION_SHA256,
        "PARTITION_SHA256",
    )
    return (
        cohort,
        encoded,
        tokenizer_provenance,
        assignments,
        extra,
    )


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

        (
            cohort,
            encoded,
            tokenizer_provenance,
            assignments,
            extra,
        ) = load_population_and_encoding(snapshot_path)

        grammar_manifest = forced_grammar_manifest(gate_artifact)
        token_ids = forced_token_ids(grammar_manifest)
        branch_tokens = branch_token_by_label(token_ids)

        model, device, runtime, frozen, model_identity = (
            base.load_generation_model(
                snapshot=snapshot_path,
                physical_device=physical_device,
            )
        )

        rows: list[dict[str, Any]] = []
        for index in range(
            int(shard["start"]),
            int(shard["end"]),
        ):
            cohort_row = cohort[index]
            prompt = base.extract_prompt_ids(
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
                    partition=assignments[
                        str(cohort_row["example_id"])
                    ],
                    token_ids=token_ids,
                    branch_tokens=branch_tokens,
                )
            )

        torch.cuda.synchronize(device)
        require(
            len(rows)
            == int(shard["end"]) - int(shard["start"]),
            "SHARD_ROW_COUNT",
        )

        payload = {
            "schema_version":
                "gen4-pre-emission-forced-decisive-stage-a-shard-v1",
            "shard": dict(shard),
            "model_identity": model_identity,
            "tokenizer": tokenizer_provenance,
            "partition_sha256":
                extra["partition_manifest"]["partition_sha256"],
            "forced_grammar_sha256":
                grammar_manifest["grammar_sha256"],
            "rows": rows,
        }
        Path(temp_output).write_bytes(
            canonical_json_bytes(payload)
        )

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
            name=(
                "forced-decisive-stage-a-shard-"
                f"{shard['shard_id']}"
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        require(
            process.exitcode == 0,
            f"SHARD_EXIT:{process.name}:{process.exitcode}",
        )

    require(
        all(path.is_file() for path in outputs),
        "SHARD_OUTPUT_MISSING",
    )
    return outputs


def merge_shards(
    *,
    shard_paths: Sequence[Path],
    cohort: Sequence[Mapping[str, Any]],
    partition_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(
        len(shard_paths) == SHARD_COUNT,
        "MERGE_SHARD_COUNT",
    )

    rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    identity_ref: Mapping[str, Any] | None = None

    for shard_id, path in enumerate(shard_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))

        require(
            payload["schema_version"]
            == "gen4-pre-emission-forced-decisive-stage-a-shard-v1",
            "MERGE_SHARD_SCHEMA",
        )
        require(
            int(payload["shard"]["shard_id"]) == shard_id,
            "MERGE_SHARD_ID",
        )
        require(
            payload["partition_sha256"]
            == partition_manifest["partition_sha256"],
            "MERGE_PARTITION_SHA",
        )
        require(
            payload["forced_grammar_sha256"]
            == FORCED_GRAMMAR_SHA256,
            "MERGE_GRAMMAR_SHA",
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
            == (
                int(payload["shard"]["end"])
                - int(payload["shard"]["start"])
            ),
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

    expected_ids = [
        str(row["example_id"])
        for row in cohort
    ]
    require(
        [str(row["example_id"]) for row in rows]
        == expected_ids,
        "MERGE_ROW_ORDER",
    )
    require(
        len(set(expected_ids)) == N,
        "MERGE_UNIQUE_IDS",
    )

    return rows, shard_meta


def validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(rows) == N, "RAW_N")

    emitted_counts = Counter(
        str(row["emitted_commitment_label"])
        for row in rows
    )
    outcome_counts = Counter(
        str(row["outcome_category"])
        for row in rows
    )
    partition_counts = Counter(
        str(row["partition"])
        for row in rows
    )
    group_counts = Counter(
        str(row["primary_stage_a_group"])
        for row in rows
    )
    confirmatory_group_counts = Counter(
        str(row["primary_stage_a_group"])
        for row in rows
        if row["partition"] == "confirmatory"
    )

    require(
        set(emitted_counts).issubset(set(FORCED_CLASS_ORDER)),
        f"RAW_EMITTED_CLASSES:{dict(emitted_counts)}",
    )
    require(
        sum(emitted_counts.values()) == N,
        "RAW_EMITTED_N",
    )
    require(
        set(outcome_counts).issubset({
            "unsupported_forced_decisive",
            "supported_forced_decisive",
        }),
        f"RAW_OUTCOME_CLASSES:{dict(outcome_counts)}",
    )
    require(
        set(group_counts) == {"unsupported", "supported"}
        or set(group_counts) in ({"unsupported"}, {"supported"}),
        f"RAW_GROUP_CLASSES:{dict(group_counts)}",
    )
    require(
        dict(partition_counts)
        == {"calibration": 231, "confirmatory": 231},
        f"RAW_PARTITIONS:{dict(partition_counts)}",
    )

    for row in rows:
        require(
            row["schema_version"] == ROW_SCHEMA,
            "RAW_SCHEMA",
        )
        require(
            row["emitted_commitment_label"]
            in FORCED_CLASS_ORDER,
            "RAW_EMITTED_LABEL",
        )
        require(
            row["emitted_is_decisive"] is True,
            "RAW_DECISIVE_FLAG",
        )
        require(
            row["primary_stage_a_group"]
            in ("unsupported", "supported"),
            "RAW_PRIMARY_GROUP",
        )
        require(
            (
                bool(row["unsupported_forced_decisive_commitment"])
                != bool(row["supported_forced_decisive_commitment"])
            ),
            "RAW_OUTCOME_XOR",
        )
        require(
            int(row["t_star_zero_based"])
            == DECISIVE_TSTAR_ZERO_BASED,
            "RAW_TSTAR",
        )
        require(
            [x["relative_offset"] for x in row["observations"]]
            == list(OBSERVATION_OFFSETS),
            "RAW_OBSERVATIONS",
        )
        require(
            row["generate_api_call_count"] == 0,
            "RAW_GENERATE_API",
        )
        require(
            row["manual_constrained_decoding"] is True,
            "RAW_MANUAL_DECODER",
        )
        require(
            row["forced_decisive_two_class_grammar"] is True,
            "RAW_FORCED_GRAMMAR",
        )
        require(
            row["not_entitled_available"] is False,
            "RAW_NOT_ENTITLED_AVAILABLE",
        )
        require(row["sampling"] is False, "RAW_SAMPLING")
        require(row["temperature"] is None, "RAW_TEMPERATURE")
        require(row["top_k"] is None, "RAW_TOP_K")
        require(row["top_p"] is None, "RAW_TOP_P")
        require(row["beam_search"] is False, "RAW_BEAM")
        require(
            row["repetition_penalty"] is None,
            "RAW_REPETITION_PENALTY",
        )
        require(row["logit_bias"] is None, "RAW_LOGIT_BIAS")
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
        "emitted_commitment_counts":
            dict(sorted(emitted_counts.items())),
        "outcome_category_counts":
            dict(sorted(outcome_counts.items())),
        "forced_decisive_group_counts":
            dict(sorted(group_counts.items())),
        "confirmatory_forced_decisive_group_counts":
            dict(sorted(confirmatory_group_counts.items())),
        "partition_counts": dict(partition_counts),
        "scientific_model_forward_count_this_run":
            total_forwards,
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
    forced_grammar: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_summary = validate_raw_rows(rows)
    row_raw = jsonl_bytes(rows)
    row_sha = sha256_bytes(row_raw)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "design": {
            "artifact": DESIGN_ARTIFACT.as_posix(),
            "artifact_git_blob": DESIGN_ARTIFACT_GIT_BLOB,
        },
        "base_generator_dependency": {
            "path": BASE_GENERATOR.as_posix(),
            "git_blob": BASE_GENERATOR_GIT_BLOB,
            "purpose":
                "reuse_frozen_model_loading_partition_and_full_sequence_late_readout",
        },
        "prior_closed_stage_a": {
            "raw_freeze_commit": PRIOR_RAW_FREEZE_COMMIT,
            "closure_commit": PRIOR_CLOSURE_COMMIT,
            "reopened": False,
            "continued_as_same_experiment": False,
        },
        "hf_repo": base.geom.HF_REPO,
        "hf_revision": base.geom.HF_REVISION,
        "backbone_canonical_sha256":
            base.BACKBONE_SHA256,
        "lm_head_weight_sha256":
            base.LM_HEAD_SHA256,
        "source_identity_gate": {
            "artifact": base.GATE_ARTIFACT.as_posix(),
            "artifact_git_blob":
                base.GATE_ARTIFACT_GIT_BLOB,
            "manifest_sha256":
                base.GATE_MANIFEST_SHA256,
            "three_class_grammar_sha256":
                base.GRAMMAR_SHA256,
        },
        "forced_decisive_grammar": dict(forced_grammar),
        "cohort": {
            "count": N,
            "cohort_sha256":
                base.averitec.COHORT_SHA256,
            "token_gate_manifest_sha256":
                base.averitec.MANIFEST_SHA256,
        },
        "partition": dict(partition_manifest),
        "frozen_sites": {
            "early_block": base.EARLY_BLOCK,
            "early_plane": base.SELECTED_PLANE,
            "control_plane": base.CONTROL_PLANE,
            "late_block": base.LATE_BLOCK,
        },
        "generation_protocol": {
            "decoder":
                "manual_full_prefix_forced_decisive_two_class_finite_grammar_greedy",
            "use_cache": False,
            "generate_api_used": False,
            "sampling": False,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "beam_search": False,
            "repetition_penalty": None,
            "logit_bias": None,
            "not_entitled_available": False,
            "max_generated_tokens":
                MAX_GENERATED_TOKENS,
            "common_prefix_length":
                COMMON_PREFIX_LENGTH,
            "branch_decision_token_index_zero_based":
                DECISIVE_TSTAR_ZERO_BASED,
            "observation_offsets":
                list(OBSERVATION_OFFSETS),
            "observation_generated_prefix_lengths":
                list(OBSERVATION_GENERATED_PREFIX_LENGTHS),
        },
        "planned_stage_a_inference_not_executed": {
            "primary_partition":
                PRIMARY_STAGE_A_PARTITION,
            "primary_comparison":
                "unsupported_forced_decisive_vs_supported_forced_decisive",
            "primary_signal":
                PRIMARY_STAGE_A_SIGNAL,
            "relative_offsets":
                list(PRIMARY_STAGE_A_OFFSETS),
            "test":
                PRIMARY_STAGE_A_TEST,
            "multiplicity":
                PRIMARY_STAGE_A_MULTIPLICITY,
            "planned_p_value_count":
                PRIMARY_STAGE_A_P_VALUE_COUNT,
            "alpha":
                PRIMARY_STAGE_A_ALPHA,
            "support_rule":
                PRIMARY_STAGE_A_SUPPORT_RULE,
            "estimability_rule":
                "both_confirmatory_forced_decisive_groups_must_have_positive_count_before_any_p_value",
        },
        "late_readout": {
            "source": "post_block_47",
            "readout":
                "full_sequence_post_block_47_to_frozen_terminal_norm_then_last_token_tied_lm_head",
            "p3_projected_at_block47": False,
            "full_sequence_norm_replay": True,
            "reproduction_atol":
                base.LATE_READOUT_REPRO_ATOL,
            "diagnostic_margin":
                "logit_SUPPORT_minus_logit_REFUTE",
            "diagnostic_only": True,
        },
        "tokenizer": dict(tokenizer_provenance),
        "shards": list(shard_meta),
        **raw_summary,
        "row_file_sha256": row_sha,
        "scientific_generation_executed": True,
        "generation_response_inspected_by_selection_logic": False,
        "primary_inference_executed": False,
        "forced_decisive_estimability_evaluation_deferred_until_raw_artifact_frozen":
            True,
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


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute raw prospective forced-decisive two-class Stage-A "
            "generation for frozen Mamba-370M P3. No statistical inference."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    validate_protocol()
    authenticate_repo(str(args.expected_head))

    snapshot = args.snapshot.resolve()
    base.geom.validate_snapshot(snapshot)

    gate_artifact = base.load_gate_artifact()
    grammar_manifest = forced_grammar_manifest(gate_artifact)

    (
        cohort,
        _encoded,
        tokenizer_provenance,
        _assignments,
        extra,
    ) = load_population_and_encoding(snapshot)

    partition_manifest = extra["partition_manifest"]
    require(
        not args.output_dir.exists(),
        "OUTPUT_COLLISION",
    )

    with tempfile.TemporaryDirectory(
        prefix="gen4_pre_emission_forced_decisive_stage_a_"
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
        forced_grammar=grammar_manifest,
    )

    print(f"RESULT={summary['result']}")
    print(f"ROWS={N}")
    print(
        "FORCED_GRAMMAR_SHA256="
        + summary["forced_decisive_grammar"]["grammar_sha256"]
    )
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
        "CONFIRMATORY_GROUP_COUNTS="
        + json.dumps(
            summary["confirmatory_forced_decisive_group_counts"],
            sort_keys=True,
        )
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT="
        + str(
            summary[
                "scientific_model_forward_count_this_run"
            ]
        )
    )
    print("PRIMARY_INFERENCE_EXECUTED=False")
    print("ESTIMABILITY_EVALUATION_DEFERRED=True")
    print("P_VALUE_COUNT_ADDED=0")
    print("TRAINING_EXECUTED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
