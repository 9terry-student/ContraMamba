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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_precursor_v2_dcs_stage_a_fast_cuda
    as cohort_source,
)
from scripts import (
    reason_router_gen4_precursor_v4_analytic_vjp_one_row_equivalence
    as v4eq,
)
from scripts import (
    reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_fast_cuda
    as forced,
)
from scripts import (
    reason_router_gen4_pre_emission_stage_a_generation_fast_cuda
    as base,
)

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "b9e7ef119fcb99cb92b2a18cd524f22c9c203d78"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_precursor_v4_analytic_vjp_susceptibility_design_candidate.md"
)
DESIGN_GIT_BLOB = "4f2e04de11f668e2a592971ad7952eb1863bd538"

EQUIVALENCE_SCRIPT = Path(
    "scripts/reason_router_gen4_precursor_v4_analytic_vjp_one_row_equivalence.py"
)
EQUIVALENCE_SCRIPT_GIT_BLOB = "92ede8e49fa5a329a5d292db714c7c854138e369"

EQUIVALENCE_ROOT = Path(
    "reports/reason_router_gen4_precursor_v4_analytic_vjp_equivalence_runs/"
    "g4k-precursorv4-analytic-vjp-eq-historical-3f591ff-gpu1"
)
EQUIVALENCE_REPORT = (
    EQUIVALENCE_ROOT / "precursor_v4_analytic_vjp_one_row_equivalence_report.json"
)
EQUIVALENCE_SUMS = EQUIVALENCE_ROOT / "SHA256SUMS.txt"
EQUIVALENCE_REPORT_GIT_BLOB = "eaa509568dd0d035db29417dc091878909b62da4"
EQUIVALENCE_SUMS_GIT_BLOB = "9df6136403f1e038f4d7748b5e3874c2c6e2449d"
EQUIVALENCE_REPORT_SHA256 = (
    "9604d8a3d4eb13fd8fd25d5829e546da928dbc7e00fe89dfaa8ab427b2fa1f32"
)
EQUIVALENCE_SUMS_SHA256 = (
    "76df92da072845654770b529f0c1d81e0e5663db3c171c65e72be6acab34e6bd"
)
EQUIVALENCE_EXECUTION_HEAD = "3f591ff629660e2cf9dafc7951f5f2c1b65aa454"
EQUIVALENCE_RESULT = "PASS_PRECURSOR_V4_ANALYTIC_VJP_CPU_CUDA_EQUIVALENCE"

COHORT_RUNNER = Path(
    "scripts/reason_router_gen4_precursor_v2_dcs_stage_a_fast_cuda.py"
)
COHORT_RUNNER_GIT_BLOB = "835d8faba49b70ded8540924233015d06b6c2b68"

COHORT_FILE = cohort_source.COHORT_FILE
COHORT_MANIFEST = cohort_source.COHORT_MANIFEST
COHORT_SUMS = cohort_source.COHORT_SUMS
COHORT_GIT_BLOB = "842e66434af5e142f9b8b0abff1a57b986884e4f"
COHORT_MANIFEST_GIT_BLOB = "7b86a94add40f3f89f2c32f2a37b67c4a7c22b6f"
COHORT_SUMS_GIT_BLOB = "885a0816a689391e53562f2ec66b3db37e05b3bf"

BASE_GENERATOR = Path(
    "scripts/reason_router_gen4_pre_emission_stage_a_generation_fast_cuda.py"
)
BASE_GENERATOR_GIT_BLOB = "aa6725bdc1cbfa43547e89293f3b7911a4cdd6ed"

FORCED_RUNNER = Path(
    "scripts/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_fast_cuda.py"
)
FORCED_RUNNER_GIT_BLOB = "99fb5a7b43d35d667d3dda646adea0e37392e647"

IDENTITY_GATE = Path(
    "scripts/reason_router_gen4_pre_emission_causal_lm_identity_grammar_gate.py"
)
IDENTITY_GATE_GIT_BLOB = "fa5fd6972715247c1d23f60dabbef74717609dcb"

N = 800
SOURCE_LABEL_COUNTS = {"Refuted": 400, "Supported": 400}
CORRECT_LABEL_COUNTS = {"REFUTE": 400, "SUPPORT": 400}

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
PLANES = (SELECTED_PLANE, CONTROL_PLANE)
BASIS_NAMES = ("plus", "minus")
EARLY_BLOCK = 35
LATE_BLOCK = 47
DIM = 650

OBSERVATION_OFFSETS = (-4, -3, -2, -1)
OBSERVATION_PREFIX_LENGTHS = (5, 6, 7, 8)
PREFIX_LENGTH_TO_OFFSET = dict(
    zip(OBSERVATION_PREFIX_LENGTHS, OBSERVATION_OFFSETS)
)

FORCED_CLASS_ORDER = ("REFUTE", "SUPPORT")
FULL_MODEL_FORWARDS_PER_ROW = 12
LOCAL_VJPS_PER_ROW = 4
SCIENTIFIC_FORWARD_BUDGET = N * FULL_MODEL_FORWARDS_PER_ROW
SCIENTIFIC_LOCAL_VJP_BUDGET = N * LOCAL_VJPS_PER_ROW

SHARDS = (
    {
        "shard_id": 0,
        "physical_device": 0,
        "start": 0,
        "end": 400,
        "example_count": 400,
        "forward_budget": 4800,
        "local_vjp_budget": 1600,
    },
    {
        "shard_id": 1,
        "physical_device": 1,
        "start": 400,
        "end": 800,
        "example_count": 400,
        "forward_budget": 4800,
        "local_vjp_budget": 1600,
    },
)
SHARD_COUNT = len(SHARDS)

ROW_FILE = "precursor_v4_analytic_stage_a_raw_rows.jsonl"
SUMMARY_FILE = "execution_summary.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

ROW_SCHEMA = "gen4-precursor-v4-analytic-stage-a-raw-row-v1"
SHARD_SCHEMA = "gen4-precursor-v4-analytic-stage-a-shard-v1"
SUMMARY_SCHEMA = "gen4-precursor-v4-analytic-stage-a-raw-summary-v1"
RESULT_PASS = "PASS_PRECURSOR_V4_ANALYTIC_STAGE_A_RAW"


class PrecursorV4StageAError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorV4StageAError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecursorV4StageAError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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

    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            expected_head,
        )
        == 0,
        "REQUIRED_ANCESTOR",
    )

    pinned = {
        DESIGN_ARTIFACT: DESIGN_GIT_BLOB,
        EQUIVALENCE_SCRIPT: EQUIVALENCE_SCRIPT_GIT_BLOB,
        EQUIVALENCE_REPORT: EQUIVALENCE_REPORT_GIT_BLOB,
        EQUIVALENCE_SUMS: EQUIVALENCE_SUMS_GIT_BLOB,
        COHORT_RUNNER: COHORT_RUNNER_GIT_BLOB,
        COHORT_FILE: COHORT_GIT_BLOB,
        COHORT_MANIFEST: COHORT_MANIFEST_GIT_BLOB,
        COHORT_SUMS: COHORT_SUMS_GIT_BLOB,
        BASE_GENERATOR: BASE_GENERATOR_GIT_BLOB,
        FORCED_RUNNER: FORCED_RUNNER_GIT_BLOB,
        IDENTITY_GATE: IDENTITY_GATE_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_equivalence_freeze() -> dict[str, Any]:
    report_path = ROOT / EQUIVALENCE_REPORT
    sums_path = ROOT / EQUIVALENCE_SUMS
    require(report_path.is_file(), "EQUIVALENCE_REPORT_MISSING")
    require(sums_path.is_file(), "EQUIVALENCE_SUMS_MISSING")
    require(
        sha256_file(report_path) == EQUIVALENCE_REPORT_SHA256,
        "EQUIVALENCE_REPORT_SHA256",
    )
    require(
        sha256_file(sums_path) == EQUIVALENCE_SUMS_SHA256,
        "EQUIVALENCE_SUMS_SHA256",
    )

    expected_line = (
        EQUIVALENCE_REPORT_SHA256
        + "  "
        + EQUIVALENCE_REPORT.name
    )
    lines = [
        line.strip()
        for line in sums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(lines == [expected_line], "EQUIVALENCE_SUMS_CONTENT")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["result"] == EQUIVALENCE_RESULT, "EQUIVALENCE_RESULT")
    require(
        report["execution_head"] == EQUIVALENCE_EXECUTION_HEAD,
        "EQUIVALENCE_EXECUTION_HEAD",
    )
    require(
        report["comparison"]["result"] == "PASS",
        "EQUIVALENCE_COMPARISON",
    )
    require(
        report["measurement"]["name"]
        == "analytic_local_differential_susceptibility",
        "EQUIVALENCE_MEASUREMENT",
    )
    require(report["measurement"]["epsilon"] is None, "EQUIVALENCE_EPSILON")
    require(
        report["boundary"]["fresh_n800_scientific_cohort_accessed"] is False,
        "EQUIVALENCE_N800_ACCESS",
    )
    require(
        report["boundary"]["scientific_inference_executed"] is False,
        "EQUIVALENCE_INFERENCE",
    )
    require(
        report["boundary"]["statistical_testing_performed"] is False,
        "EQUIVALENCE_STATS",
    )
    require(
        report["boundary"]["p_value_count_added"] == 0,
        "EQUIVALENCE_PVALUES",
    )
    return report


def validate_protocol() -> None:
    require(N == 800, "N")
    require(
        SOURCE_LABEL_COUNTS == {"Refuted": 400, "Supported": 400},
        "SOURCE_LABEL_COUNTS",
    )
    require(
        CORRECT_LABEL_COUNTS == {"REFUTE": 400, "SUPPORT": 400},
        "CORRECT_LABEL_COUNTS",
    )
    require(PLANES == ("P3", "P5"), "PLANES")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")
    require(EARLY_BLOCK == 35, "EARLY_BLOCK")
    require(LATE_BLOCK == 47, "LATE_BLOCK")
    require(DIM == 650, "DIM")
    require(
        OBSERVATION_OFFSETS == (-4, -3, -2, -1),
        "OBSERVATION_OFFSETS",
    )
    require(
        OBSERVATION_PREFIX_LENGTHS == (5, 6, 7, 8),
        "OBSERVATION_PREFIX_LENGTHS",
    )
    require(
        FULL_MODEL_FORWARDS_PER_ROW == 12,
        "FULL_MODEL_FORWARDS_PER_ROW",
    )
    require(LOCAL_VJPS_PER_ROW == 4, "LOCAL_VJPS_PER_ROW")
    require(SCIENTIFIC_FORWARD_BUDGET == 9600, "FORWARD_BUDGET")
    require(
        SCIENTIFIC_LOCAL_VJP_BUDGET == 3200,
        "VJP_BUDGET",
    )
    require(SHARD_COUNT == 2, "SHARD_COUNT")

    require(cohort_source.N == N, "COHORT_SOURCE_N")
    require(
        cohort_source.SOURCE_LABEL_COUNTS == SOURCE_LABEL_COUNTS,
        "COHORT_SOURCE_LABELS",
    )
    require(
        cohort_source.CORRECT_LABEL_COUNTS == CORRECT_LABEL_COUNTS,
        "COHORT_CORRECT_LABELS",
    )
    require(base.SELECTED_PLANE == SELECTED_PLANE, "BASE_P3")
    require(base.CONTROL_PLANE == CONTROL_PLANE, "BASE_P5")
    require(base.EARLY_BLOCK == EARLY_BLOCK, "BASE_EARLY_BLOCK")
    require(base.LATE_BLOCK == LATE_BLOCK, "BASE_LATE_BLOCK")
    require(base.DIM == DIM, "BASE_DIM")
    require(
        forced.OBSERVATION_OFFSETS == OBSERVATION_OFFSETS,
        "FORCED_OFFSETS",
    )
    require(
        forced.OBSERVATION_GENERATED_PREFIX_LENGTHS
        == OBSERVATION_PREFIX_LENGTHS,
        "FORCED_PREFIX_LENGTHS",
    )
    require(
        forced.FORCED_CLASS_ORDER == FORCED_CLASS_ORDER,
        "FORCED_CLASS_ORDER",
    )

    covered: list[int] = []
    total_forward_budget = 0
    total_vjp_budget = 0
    for expected_id, shard in enumerate(SHARDS):
        require(int(shard["shard_id"]) == expected_id, "SHARD_ID")
        require(int(shard["physical_device"]) == expected_id, "SHARD_DEVICE")
        start = int(shard["start"])
        end = int(shard["end"])
        count = int(shard["example_count"])
        require(end - start == count, f"SHARD_COUNT:{expected_id}")
        require(
            int(shard["forward_budget"])
            == count * FULL_MODEL_FORWARDS_PER_ROW,
            f"SHARD_FORWARD_BUDGET:{expected_id}",
        )
        require(
            int(shard["local_vjp_budget"])
            == count * LOCAL_VJPS_PER_ROW,
            f"SHARD_VJP_BUDGET:{expected_id}",
        )
        covered.extend(range(start, end))
        total_forward_budget += int(shard["forward_budget"])
        total_vjp_budget += int(shard["local_vjp_budget"])

    require(covered == list(range(N)), "SHARD_COVERAGE")
    require(
        total_forward_budget == SCIENTIFIC_FORWARD_BUDGET,
        "SHARD_FORWARD_SUM",
    )
    require(
        total_vjp_budget == SCIENTIFIC_LOCAL_VJP_BUDGET,
        "SHARD_VJP_SUM",
    )


def analytic_measure_and_logits(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1,
        "MEASURE_INPUT_SHAPE",
    )
    require(input_ids.dtype == torch.long, "MEASURE_INPUT_DTYPE")
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "MODEL_PARAMETERS_REQUIRE_GRAD",
    )
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "MODEL_PARAMETER_GRAD_PREEXISTS",
    )

    token_index = int(input_ids.shape[1]) - 1
    require(token_index >= 0, "MEASURE_EMPTY_INPUT")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    capture: dict[str, torch.Tensor] = {}

    def early_hook(_module, _args, output):
        require(torch.is_tensor(output), "EARLY_OUTPUT_NOT_TENSOR")
        require(
            output.ndim == 3
            and output.shape[0] == 1
            and output.shape[-1] == 2 * base.geom.INTERMEDIATE_SIZE,
            f"EARLY_OUTPUT_SHAPE:{tuple(output.shape)}",
        )
        leaf = output.detach().clone().requires_grad_(True)
        capture["leaf"] = leaf
        return leaf

    handle = runtime["early_in_proj"].register_forward_hook(early_hook)
    try:
        with torch.enable_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            require(set(capture) == {"leaf"}, "LEAF_CAPTURE")
            logits = output.logits
            require(
                torch.is_tensor(logits)
                and tuple(logits.shape[:2]) == (1, 1),
                f"LOGITS_SHAPE:{tuple(logits.shape)}",
            )
            next_logits = logits[0, -1].float()
            margin = v4eq.differentiable_gold_margin(
                next_logits,
                gold_label=gold_label,
                branch_tokens=branch_tokens,
            )
            require(bool(torch.isfinite(margin).item()), "MARGIN_NONFINITE")
            gradient, = torch.autograd.grad(
                margin,
                capture["leaf"],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
    finally:
        handle.remove()

    require(
        torch.is_tensor(gradient)
        and gradient.shape == capture["leaf"].shape,
        "GRADIENT_SHAPE",
    )
    require(
        bool(torch.isfinite(gradient).all().item()),
        "GRADIENT_NONFINITE",
    )

    mask_cpu = runtime["strong_mask"].detach().cpu().bool().contiguous()
    require(
        mask_cpu.numel() == base.geom.INTERMEDIATE_SIZE
        and int(mask_cpu.sum().item()) == DIM,
        "STRONG_MASK",
    )
    mask_device = mask_cpu.to(gradient.device)
    strong_gradient = (
        gradient[
            0,
            token_index,
            :base.geom.INTERMEDIATE_SIZE,
        ][mask_device]
        .detach().cpu().to(torch.float64).contiguous()
    )
    require(
        tuple(strong_gradient.shape) == (DIM,),
        "STRONG_GRADIENT_SHAPE",
    )
    require(
        bool(torch.isfinite(strong_gradient).all().item()),
        "STRONG_GRADIENT_NONFINITE",
    )

    directional: dict[str, dict[str, float]] = {}
    for plane in PLANES:
        directional[plane] = {}
        for basis_name in BASIS_NAMES:
            basis = (
                frozen["planes"][plane][basis_name]
                .detach().cpu().to(torch.float64).contiguous()
            )
            require(
                tuple(basis.shape) == (DIM,),
                f"BASIS_SHAPE:{plane}:{basis_name}",
            )
            value = float(torch.dot(strong_gradient, basis).item())
            require(
                math.isfinite(value),
                f"DIRECTIONAL_NONFINITE:{plane}:{basis_name}",
            )
            directional[plane][basis_name] = value

    derived = v4eq.summarize_directional_values(directional)
    branch_logits = {
        label: float(
            next_logits[int(branch_tokens[label])].detach().cpu().item()
        )
        for label in FORCED_CLASS_ORDER
    }
    f_t = float(margin.detach().cpu().item())
    require(math.isfinite(f_t), "F_T_NONFINITE")
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "MODEL_PARAMETER_GRAD_CREATED",
    )

    measurement = {
        "gold_aligned_f_t": f_t,
        "branch_start_logits": branch_logits,
        "strong_gradient_l2": float(
            torch.linalg.vector_norm(strong_gradient).item()
        ),
        **derived,
        "local_vjp_count": 1,
        "model_forward_count": 1,
        "training_executed": False,
        "parameter_gradient_created": False,
        "parameter_update_executed": False,
    }
    return next_logits.detach(), measurement


def summarize_item_endpoint(
    measurements: Sequence[Mapping[str, Any]],
) -> float:
    require(
        len(measurements) == len(OBSERVATION_OFFSETS),
        "ITEM_MEASUREMENT_COUNT",
    )
    require(
        [int(x["relative_offset"]) for x in measurements]
        == list(OBSERVATION_OFFSETS),
        "ITEM_OFFSET_ORDER",
    )
    values = [float(x["d_t"]) for x in measurements]
    require(all(math.isfinite(x) for x in values), "ITEM_D_T_FINITE")
    z_i = sum(values) / float(len(values))
    require(math.isfinite(z_i), "ITEM_Z_FINITE")
    return z_i


def run_one(
    *,
    model: Any,
    device: torch.device,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    prompt_ids: torch.Tensor,
    cohort_row: Mapping[str, Any],
    token_ids: Mapping[str, Sequence[int]],
    branch_tokens: Mapping[str, int],
) -> dict[str, Any]:
    gold = str(cohort_row["correct_label"])
    require(gold in FORCED_CLASS_ORDER, f"ROW_GOLD:{gold}")

    prompt = prompt_ids.detach().cpu().to(torch.long).contiguous()
    generated: list[int] = []
    measurements: list[dict[str, Any]] = []
    full_model_forward_count = 0
    local_vjp_count = 0
    branch_choice_allowed_logits: dict[str, float] | None = None

    while True:
        complete = (
            forced.completed_label(generated, token_ids)
            if generated
            else None
        )
        if complete is not None:
            emitted = complete
            break

        require(
            len(generated) < forced.MAX_GENERATED_TOKENS,
            "GENERATION_MAX_TOKENS",
        )
        allowed = base.identity_gate.allowed_next_tokens(
            generated,
            token_ids,
        )
        require(allowed, "GENERATION_ALLOWED_EMPTY")

        combined = torch.cat(
            [
                prompt,
                torch.tensor(generated, dtype=torch.long),
            ]
        ).unsqueeze(0).to(device)

        generated_prefix_length = len(generated)
        if generated_prefix_length in PREFIX_LENGTH_TO_OFFSET:
            next_logits, measurement = analytic_measure_and_logits(
                model=model,
                runtime=runtime,
                frozen=frozen,
                input_ids=combined,
                gold_label=gold,
                branch_tokens=branch_tokens,
            )
            offset = PREFIX_LENGTH_TO_OFFSET[generated_prefix_length]
            measurement.update({
                "relative_offset": int(offset),
                "generated_prefix_length": int(generated_prefix_length),
                "last_emitted_token_index_zero_based":
                    int(generated_prefix_length) - 1,
                "current_next_token_is_commitment_branch":
                    int(generated_prefix_length)
                    == forced.COMMON_PREFIX_LENGTH,
            })
            measurements.append(measurement)
            local_vjp_count += 1
        else:
            next_logits, observation = forced.forward_next_token(
                model=model,
                runtime=runtime,
                frozen=frozen,
                input_ids=combined,
                branch_tokens=branch_tokens,
                observe=False,
            )
            require(observation is None, "NATIVE_OBSERVATION_UNEXPECTED")

        full_model_forward_count += 1

        next_token, allowed_logits = base.greedy_allowed_token(
            next_logits,
            allowed,
        )

        if generated_prefix_length == forced.COMMON_PREFIX_LENGTH:
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

        generated.append(int(next_token))
        forced.compatible_labels(generated, token_ids)

    require(emitted in FORCED_CLASS_ORDER, "EMITTED_LABEL")
    require(
        generated == [int(x) for x in token_ids[emitted]],
        "EMITTED_TOKEN_SEQUENCE",
    )
    require(
        full_model_forward_count == FULL_MODEL_FORWARDS_PER_ROW,
        f"FULL_MODEL_FORWARD_COUNT:{full_model_forward_count}",
    )
    require(
        local_vjp_count == LOCAL_VJPS_PER_ROW,
        f"LOCAL_VJP_COUNT:{local_vjp_count}",
    )
    require(
        [int(x["relative_offset"]) for x in measurements]
        == list(OBSERVATION_OFFSETS),
        "MEASUREMENT_OFFSET_COVERAGE",
    )
    require(
        [int(x["generated_prefix_length"]) for x in measurements]
        == list(OBSERVATION_PREFIX_LENGTHS),
        "MEASUREMENT_PREFIX_COVERAGE",
    )
    require(
        branch_choice_allowed_logits is not None,
        "BRANCH_CHOICE_LOGITS",
    )

    z_i = summarize_item_endpoint(measurements)
    supported = emitted == gold

    return {
        "schema_version": ROW_SCHEMA,
        "averitec_train_index": int(cohort_row["averitec_train_index"]),
        "example_id": str(cohort_row["example_id"]),
        "source_label": str(cohort_row["source_label"]),
        "correct_label": gold,
        "correct_label_id": int(cohort_row["correct_label_id"]),
        "prompt_attended_token_count": int(prompt.numel()),
        "generated_token_ids": list(generated),
        "generated_token_count": len(generated),
        "generated_commitment_surface": forced.FORCED_SURFACES[emitted],
        "emitted_commitment_label": emitted,
        "branch_decision_token_index_zero_based":
            forced.DECISIVE_TSTAR_ZERO_BASED,
        "branch_choice_allowed_logits": branch_choice_allowed_logits,
        "supported_forced_decisive_commitment": bool(supported),
        "unsupported_forced_decisive_commitment": bool(not supported),
        "primary_stage_a_group":
            "supported" if supported else "unsupported",
        "outcome_category":
            "supported_forced_decisive"
            if supported
            else "unsupported_forced_decisive",
        "analytic_local_differential_susceptibility_measurements":
            measurements,
        "z_i": z_i,
        "scientific_full_model_forward_count": full_model_forward_count,
        "scientific_local_vjp_count": local_vjp_count,
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
        "scientific_measurement_executed": True,
        "primary_inference_executed": False,
        "p_value_count_added": 0,
        "training_executed": False,
        "parameter_gradient_created": False,
        "parameter_update_executed": False,
    }


def shard_worker(
    *,
    shard: Mapping[str, int],
    snapshot: str,
    temp_output: str,
) -> None:
    try:
        physical_device = int(shard["physical_device"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        snapshot_path = Path(snapshot)

        cohort, cohort_manifest = cohort_source.validate_frozen_cohort()
        _rows, encoded, tokenizer_provenance = cohort_source.build_input(
            snapshot=snapshot_path,
            cohort=cohort,
            cohort_manifest=cohort_manifest,
        )

        gate_artifact = base.load_gate_artifact()
        grammar_manifest = forced.forced_grammar_manifest(gate_artifact)
        token_ids = forced.forced_token_ids(grammar_manifest)
        branch_tokens = forced.branch_token_by_label(token_ids)

        model, device, runtime, frozen, model_identity = (
            base.load_generation_model(
                snapshot=snapshot_path,
                physical_device=physical_device,
            )
        )
        require(
            int(runtime["strong_mask"].detach().cpu().sum().item()) == DIM,
            "RUNTIME_STRONG_DIM",
        )
        require(
            SELECTED_PLANE in frozen["planes"]
            and CONTROL_PLANE in frozen["planes"],
            "RUNTIME_PLANES",
        )

        rows: list[dict[str, Any]] = []
        for index in range(
            int(shard["start"]),
            int(shard["end"]),
        ):
            cohort_row = cohort[index]
            prompt = cohort_source.extract_prompt_ids(
                encoded,
                index=index,
                expected_attended=int(
                    cohort_row["serialized_attended_length"]
                ),
            )
            rows.append(
                run_one(
                    model=model,
                    device=device,
                    runtime=runtime,
                    frozen=frozen,
                    prompt_ids=prompt,
                    cohort_row=cohort_row,
                    token_ids=token_ids,
                    branch_tokens=branch_tokens,
                )
            )

        torch.cuda.synchronize(device)
        require(
            len(rows) == int(shard["example_count"]),
            "SHARD_ROW_COUNT",
        )
        require(
            sum(
                int(row["scientific_full_model_forward_count"])
                for row in rows
            )
            == int(shard["forward_budget"]),
            "SHARD_FORWARD_COUNT",
        )
        require(
            sum(
                int(row["scientific_local_vjp_count"])
                for row in rows
            )
            == int(shard["local_vjp_budget"]),
            "SHARD_VJP_COUNT",
        )

        payload = {
            "schema_version": SHARD_SCHEMA,
            "shard": dict(shard),
            "model_identity": dict(model_identity),
            "tokenizer": dict(tokenizer_provenance),
            "forced_grammar_sha256":
                grammar_manifest["grammar_sha256"],
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
) -> list[Path]:
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    outputs: list[Path] = []

    for shard in SHARDS:
        output = temp_dir / f"shard{shard['shard_id']}.json"
        outputs.append(output)
        process = ctx.Process(
            target=shard_worker,
            kwargs={
                "shard": shard,
                "snapshot": str(snapshot),
                "temp_output": str(output),
            },
            name=f"precursor-v4-analytic-stage-a-shard-{shard['shard_id']}",
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(len(shard_paths) == SHARD_COUNT, "MERGE_SHARD_COUNT")

    rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    model_identity_ref: Mapping[str, Any] | None = None
    tokenizer_ref: Mapping[str, Any] | None = None

    for shard_id, path in enumerate(shard_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["schema_version"] == SHARD_SCHEMA, "MERGE_SCHEMA")
        require(
            int(payload["shard"]["shard_id"]) == shard_id,
            "MERGE_SHARD_ID",
        )
        require(payload["shard"] == SHARDS[shard_id], "MERGE_SHARD_SPEC")
        require(
            payload["forced_grammar_sha256"]
            == forced.FORCED_GRAMMAR_SHA256,
            "MERGE_GRAMMAR_SHA",
        )

        identity = payload["model_identity"]
        tokenizer = payload["tokenizer"]
        if model_identity_ref is None:
            model_identity_ref = identity
            tokenizer_ref = tokenizer
        else:
            for key in (
                "backbone_canonical_sha256",
                "lm_head_weight_sha256",
                "input_embedding_weight_sha256",
            ):
                require(
                    identity[key] == model_identity_ref[key],
                    f"MERGE_MODEL_IDENTITY:{key}",
                )
            require(
                tokenizer == tokenizer_ref,
                "MERGE_TOKENIZER_IDENTITY",
            )

        shard_rows = payload["rows"]
        require(
            len(shard_rows) == int(SHARDS[shard_id]["example_count"]),
            "MERGE_SHARD_ROWS",
        )
        rows.extend(shard_rows)
        shard_meta.append({
            "shard": dict(payload["shard"]),
            "model_identity": dict(identity),
            "tokenizer": dict(tokenizer),
            "row_count": len(shard_rows),
        })

    require(len(rows) == N, "MERGE_N")
    expected_ids = [str(row["example_id"]) for row in cohort]
    observed_ids = [str(row["example_id"]) for row in rows]
    require(observed_ids == expected_ids, "MERGE_ROW_ORDER")
    require(len(set(observed_ids)) == N, "MERGE_ROW_UNIQUENESS")
    return rows, shard_meta


def validate_raw_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(rows) == N, "RAW_N")
    group_counts = Counter(
        str(row["primary_stage_a_group"])
        for row in rows
    )
    emitted_counts = Counter(
        str(row["emitted_commitment_label"])
        for row in rows
    )

    for row in rows:
        require(row["schema_version"] == ROW_SCHEMA, "RAW_SCHEMA")
        require(
            row["emitted_commitment_label"] in FORCED_CLASS_ORDER,
            "RAW_EMITTED",
        )
        require(
            row["primary_stage_a_group"]
            in ("supported", "unsupported"),
            "RAW_GROUP",
        )
        require(
            bool(row["supported_forced_decisive_commitment"])
            != bool(row["unsupported_forced_decisive_commitment"]),
            "RAW_OUTCOME_XOR",
        )
        measurements = row[
            "analytic_local_differential_susceptibility_measurements"
        ]
        require(
            len(measurements) == LOCAL_VJPS_PER_ROW,
            "RAW_MEASUREMENT_COUNT",
        )
        require(
            [int(x["relative_offset"]) for x in measurements]
            == list(OBSERVATION_OFFSETS),
            "RAW_OFFSET_COVERAGE",
        )
        for measurement in measurements:
            require(
                set(measurement["directional_derivatives"])
                == set(PLANES),
                "RAW_DIRECTIONAL_PLANES",
            )
            require(
                math.isfinite(float(measurement["chi_p3"]))
                and math.isfinite(float(measurement["chi_p5"]))
                and math.isfinite(float(measurement["d_t"])),
                "RAW_DERIVED_FINITE",
            )
        require(math.isfinite(float(row["z_i"])), "RAW_Z_FINITE")
        require(
            int(row["scientific_full_model_forward_count"])
            == FULL_MODEL_FORWARDS_PER_ROW,
            "RAW_FORWARD_COUNT",
        )
        require(
            int(row["scientific_local_vjp_count"])
            == LOCAL_VJPS_PER_ROW,
            "RAW_VJP_COUNT",
        )
        require(row["generate_api_call_count"] == 0, "RAW_GENERATE_API")
        require(
            row["manual_constrained_decoding"] is True,
            "RAW_MANUAL_DECODE",
        )
        require(
            row["scientific_measurement_executed"] is True,
            "RAW_MEASUREMENT_EXECUTION",
        )
        require(
            row["primary_inference_executed"] is False,
            "RAW_INFERENCE",
        )
        require(row["p_value_count_added"] == 0, "RAW_P_VALUE")
        require(row["training_executed"] is False, "RAW_TRAINING")
        require(
            row["parameter_gradient_created"] is False,
            "RAW_PARAMETER_GRADIENT",
        )
        require(
            row["parameter_update_executed"] is False,
            "RAW_PARAMETER_UPDATE",
        )

    total_forwards = sum(
        int(row["scientific_full_model_forward_count"])
        for row in rows
    )
    total_vjps = sum(
        int(row["scientific_local_vjp_count"])
        for row in rows
    )
    require(
        total_forwards == SCIENTIFIC_FORWARD_BUDGET,
        f"RAW_FORWARD_BUDGET:{total_forwards}",
    )
    require(
        total_vjps == SCIENTIFIC_LOCAL_VJP_BUDGET,
        f"RAW_VJP_BUDGET:{total_vjps}",
    )
    return {
        "supported_unsupported_group_counts":
            dict(sorted(group_counts.items())),
        "emitted_commitment_counts":
            dict(sorted(emitted_counts.items())),
        "scientific_full_model_forward_count_this_run":
            total_forwards,
        "scientific_local_vjp_count_this_run":
            total_vjps,
    }


def write_output(
    *,
    output_dir: Path,
    expected_head: str,
    rows: Sequence[Mapping[str, Any]],
    shard_meta: Sequence[Mapping[str, Any]],
    equivalence_report: Mapping[str, Any],
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
        "experiment":
            "PRECURSOR_V4_ANALYTIC_LOCAL_DIFFERENTIAL_SUSCEPTIBILITY_STAGE_A",
        "model_scale": "mamba370m",
        "hf_repo": "state-spaces/mamba-370m-hf",
        "hf_revision": "589179554943157be31701edd8b4558889276674",
        "selected_plane": SELECTED_PLANE,
        "response_blind_control_plane": CONTROL_PLANE,
        "intervention_block": EARLY_BLOCK,
        "readout_block": LATE_BLOCK,
        "estimator": "analytic_local_vjp",
        "epsilon": None,
        "observation_offsets": list(OBSERVATION_OFFSETS),
        "observation_generated_prefix_lengths":
            list(OBSERVATION_PREFIX_LENGTHS),
        "endpoint":
            "Z_i=mean(D_t over t*-4,t*-3,t*-2,t*-1)",
        "forward_accounting": {
            "full_model_forwards_per_row": FULL_MODEL_FORWARDS_PER_ROW,
            "local_vjps_per_row": LOCAL_VJPS_PER_ROW,
            "cohort_count": N,
            "scientific_forward_budget": SCIENTIFIC_FORWARD_BUDGET,
            "scientific_local_vjp_budget":
                SCIENTIFIC_LOCAL_VJP_BUDGET,
            "observation_vjp_forward_reuses_decoding_forward": True,
        },
        "cohort": {
            "path": COHORT_FILE.as_posix(),
            "git_blob": COHORT_GIT_BLOB,
            "sha256": cohort_source.COHORT_SHA256,
            "count": N,
            "source_label_counts": SOURCE_LABEL_COUNTS,
            "correct_label_counts": CORRECT_LABEL_COUNTS,
            "fresh_response_free_at_selection": True,
        },
        "design": {
            "path": DESIGN_ARTIFACT.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
        },
        "equivalence_prerequisite": {
            "result": equivalence_report["result"],
            "execution_head": equivalence_report["execution_head"],
            "report_path": EQUIVALENCE_REPORT.as_posix(),
            "report_git_blob": EQUIVALENCE_REPORT_GIT_BLOB,
            "report_sha256": EQUIVALENCE_REPORT_SHA256,
            "comparison_result":
                equivalence_report["comparison"]["result"],
        },
        "dependencies": {
            "equivalence_script_git_blob":
                EQUIVALENCE_SCRIPT_GIT_BLOB,
            "cohort_runner_git_blob": COHORT_RUNNER_GIT_BLOB,
            "base_generator_git_blob": BASE_GENERATOR_GIT_BLOB,
            "forced_runner_git_blob": FORCED_RUNNER_GIT_BLOB,
            "identity_gate_git_blob": IDENTITY_GATE_GIT_BLOB,
        },
        "forced_grammar_sha256": forced.FORCED_GRAMMAR_SHA256,
        "row_file": ROW_FILE,
        "row_file_sha256": row_sha,
        "shards": [dict(x) for x in shard_meta],
        **raw_summary,
        "scientific_measurement_executed": True,
        "scientific_inference_executed": False,
        "primary_test_executed": False,
        "p_value_count_added": 0,
        "training_executed": False,
        "local_vjp_backward_executed": True,
        "parameter_gradient_created": False,
        "parameter_update_executed": False,
        "generation_api_used": False,
        "scientific_conclusion": None,
    }

    summary_raw = pretty_json_bytes(summary)
    (output_dir / ROW_FILE).write_bytes(row_raw)
    (output_dir / SUMMARY_FILE).write_bytes(summary_raw)

    hashes = {
        ROW_FILE: row_sha,
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


def execute(
    *,
    expected_head: str,
    snapshot: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_protocol()
    equivalence_report = validate_equivalence_freeze()
    cohort, _manifest = cohort_source.validate_frozen_cohort()

    require(snapshot.is_dir(), f"SNAPSHOT_DIR:{snapshot}")
    with tempfile.TemporaryDirectory(
        prefix="precursor-v4-analytic-stage-a-"
    ) as tmp:
        shard_paths = launch_shards(
            snapshot=snapshot,
            temp_dir=Path(tmp),
        )
        rows, shard_meta = merge_shards(
            shard_paths=shard_paths,
            cohort=cohort,
        )

    return write_output(
        output_dir=output_dir,
        expected_head=expected_head,
        rows=rows,
        shard_meta=shard_meta,
        equivalence_report=equivalence_report,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Precursor-v4 analytic local differential susceptibility "
            "Stage A raw CUDA generation on the frozen response-free N=800 "
            "cohort. Produces raw measurements and future outcomes only. "
            "No primary statistical test or p-value is computed here."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = execute(
        expected_head=str(args.expected_head),
        snapshot=args.snapshot,
        output_dir=args.output_dir,
    )
    print("RESULT=" + str(summary["result"]))
    print("N=" + str(summary["cohort"]["count"]))
    print(
        "GROUP_COUNTS="
        + json.dumps(
            summary["supported_unsupported_group_counts"],
            sort_keys=True,
        )
    )
    print(
        "EMITTED_COUNTS="
        + json.dumps(
            summary["emitted_commitment_counts"],
            sort_keys=True,
        )
    )
    print(
        "SCIENTIFIC_FULL_MODEL_FORWARD_COUNT="
        + str(summary["scientific_full_model_forward_count_this_run"])
    )
    print(
        "SCIENTIFIC_LOCAL_VJP_COUNT="
        + str(summary["scientific_local_vjp_count_this_run"])
    )
    print("PRIMARY_TEST_EXECUTED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
