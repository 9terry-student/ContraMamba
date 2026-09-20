#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_pre_emission_causal_lm_identity_grammar_gate
    as identity_gate,
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
REQUIRED_ANCESTOR = "42e8166667ea1c394088ac6092d8e3afb3394fff"

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_precursor_v4_analytic_vjp_susceptibility_design_candidate.md"
)
DESIGN_GIT_BLOB = "4f2e04de11f668e2a592971ad7952eb1863bd538"

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

HISTORICAL_RAW_ROOT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_runs/"
    "g4k-preemission-forceddecisive-stagea-raw-370m-0d3a094-gpu1"
)
HISTORICAL_RAW_SUMMARY = HISTORICAL_RAW_ROOT / "execution_summary.json"
HISTORICAL_RAW_ROWS = HISTORICAL_RAW_ROOT / "forced_decisive_stage_a_generation_rows.jsonl"
HISTORICAL_RAW_SUMS = HISTORICAL_RAW_ROOT / "SHA256SUMS.txt"
HISTORICAL_RAW_SUMMARY_GIT_BLOB = "725e5454a824adc18814d3a20931bb2ab65209f7"
HISTORICAL_RAW_ROWS_GIT_BLOB = "1691f8e9582f786760df19bee778f247d53b238b"
HISTORICAL_RAW_SUMS_GIT_BLOB = "a7c41fb72d08bc8f5083f45d96bf39aca693bd64"

EQUIVALENCE_ATOL = 1.0e-4
EQUIVALENCE_RTOL = 1.0e-4

SELECTED_PLANE = "P3"
CONTROL_PLANE = "P5"
PLANES = (SELECTED_PLANE, CONTROL_PLANE)
BASIS_NAMES = ("plus", "minus")
DIM = 650
RELATIVE_OFFSET = -4
GENERATED_PREFIX_LENGTH = 5

CPU_MODEL_FORWARD_COUNT = 1
GPU_MODEL_FORWARD_COUNT = 1
CPU_VJP_COUNT = 1
GPU_VJP_COUNT = 1
TOTAL_EQUIVALENCE_FORWARD_COUNT = 2
TOTAL_EQUIVALENCE_VJP_COUNT = 2

REPORT_FILE = "precursor_v4_analytic_vjp_one_row_equivalence_report.json"
SUMS_FILE = "SHA256SUMS.txt"
REPORT_SCHEMA = "gen4-precursor-v4-analytic-vjp-one-row-equivalence-v1"
RESULT_PASS = "PASS_PRECURSOR_V4_ANALYTIC_VJP_CPU_CUDA_EQUIVALENCE"


class PrecursorV4AnalyticVJPError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorV4AnalyticVJPError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecursorV4AnalyticVJPError(
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


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(git("rev-parse", "HEAD") == expected_head, "HEAD_MISMATCH")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_ANCESTOR,
            expected_head,
        ) == 0,
        "IMPLEMENTATION_NOT_DESCENDANT_OF_V3_FREEZE",
    )

    pinned = {
        DESIGN_ARTIFACT: DESIGN_GIT_BLOB,
        BASE_GENERATOR: BASE_GENERATOR_GIT_BLOB,
        FORCED_RUNNER: FORCED_RUNNER_GIT_BLOB,
        IDENTITY_GATE: IDENTITY_GATE_GIT_BLOB,
        HISTORICAL_RAW_SUMMARY: HISTORICAL_RAW_SUMMARY_GIT_BLOB,
        HISTORICAL_RAW_ROWS: HISTORICAL_RAW_ROWS_GIT_BLOB,
        HISTORICAL_RAW_SUMS: HISTORICAL_RAW_SUMS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(EQUIVALENCE_ATOL == 1e-4, "ATOL")
    require(EQUIVALENCE_RTOL == 1e-4, "RTOL")
    require(SELECTED_PLANE == "P3", "SELECTED_PLANE")
    require(CONTROL_PLANE == "P5", "CONTROL_PLANE")
    require(PLANES == ("P3", "P5"), "PLANES")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")
    require(DIM == 650, "DIM")
    require(RELATIVE_OFFSET == -4, "RELATIVE_OFFSET")
    require(GENERATED_PREFIX_LENGTH == 5, "GENERATED_PREFIX_LENGTH")
    require(
        forced.OBSERVATION_GENERATED_PREFIX_LENGTHS[0]
        == GENERATED_PREFIX_LENGTH,
        "PREFIX_LENGTH_LOCK",
    )
    require(
        forced.OBSERVATION_OFFSETS[0] == RELATIVE_OFFSET,
        "OFFSET_LOCK",
    )
    require(base.SELECTED_PLANE == SELECTED_PLANE, "BASE_P3")
    require(base.CONTROL_PLANE == CONTROL_PLANE, "BASE_P5")
    require(base.EARLY_BLOCK == 35, "EARLY_BLOCK")
    require(base.LATE_BLOCK == 47, "LATE_BLOCK")
    require(base.DIM == DIM, "BASE_DIM")
    require(CPU_MODEL_FORWARD_COUNT == 1, "CPU_FORWARD_COUNT")
    require(GPU_MODEL_FORWARD_COUNT == 1, "GPU_FORWARD_COUNT")
    require(CPU_VJP_COUNT == 1, "CPU_VJP_COUNT")
    require(GPU_VJP_COUNT == 1, "GPU_VJP_COUNT")
    require(TOTAL_EQUIVALENCE_FORWARD_COUNT == 2, "TOTAL_FORWARD_COUNT")
    require(TOTAL_EQUIVALENCE_VJP_COUNT == 2, "TOTAL_VJP_COUNT")


def select_historical_decisive_row(
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    require(len(cohort) == base.N, "HISTORICAL_COHORT_N")
    for index, row in enumerate(cohort):
        if str(row["correct_label"]) in forced.FORCED_CLASS_ORDER:
            return index, row
    raise PrecursorV4AnalyticVJPError("NO_HISTORICAL_DECISIVE_ROW")


def historical_equivalence_input(
    *,
    snapshot: Path,
) -> tuple[
    torch.Tensor,
    Mapping[str, Any],
    Mapping[str, int],
    Mapping[str, Any],
]:
    cohort, encoded, tokenizer_provenance, _assignments, extra = (
        base.load_population_and_encoding(snapshot)
    )
    index, row = select_historical_decisive_row(cohort)
    prompt = base.extract_prompt_ids(
        encoded,
        index,
        int(row["serialized_attended_length"]),
    )

    gate_artifact = base.load_gate_artifact()
    grammar_manifest = forced.forced_grammar_manifest(gate_artifact)
    token_ids = forced.forced_token_ids(grammar_manifest)
    branch_tokens = forced.branch_token_by_label(token_ids)

    refute_prefix = [
        int(x)
        for x in token_ids["REFUTE"][:GENERATED_PREFIX_LENGTH]
    ]
    support_prefix = [
        int(x)
        for x in token_ids["SUPPORT"][:GENERATED_PREFIX_LENGTH]
    ]
    require(refute_prefix == support_prefix, "PREFIX_NOT_CLASS_BLIND")

    combined = torch.cat(
        [
            prompt.detach().cpu().to(torch.long),
            torch.tensor(refute_prefix, dtype=torch.long),
        ]
    ).unsqueeze(0).contiguous()

    provenance = {
        "selection_rule":
            "first_decisive_gold_row_in_frozen_historical_precursor_population",
        "selection_uses_generation_response": False,
        "historical_population_count": base.N,
        "historical_row_index_zero_based": int(index),
        "example_id": str(row["example_id"]),
        "averitec_dev_index": int(row["averitec_dev_index"]),
        "source_label": str(row["source_label"]),
        "correct_label": str(row["correct_label"]),
        "prompt_attended_token_count": int(prompt.numel()),
        "generated_prefix_length": GENERATED_PREFIX_LENGTH,
        "relative_offset": RELATIVE_OFFSET,
        "generated_prefix_token_ids": refute_prefix,
        "combined_input_token_count": int(combined.shape[1]),
        "partition_sha256":
            extra["partition_manifest"]["partition_sha256"],
        "tokenizer": dict(tokenizer_provenance),
        "fresh_n800_scientific_cohort_accessed": False,
    }
    return combined, row, branch_tokens, provenance


def differentiable_gold_margin(
    logits: torch.Tensor,
    *,
    gold_label: str,
    branch_tokens: Mapping[str, int],
) -> torch.Tensor:
    require(
        torch.is_tensor(logits) and logits.ndim == 1,
        "MARGIN_LOGITS_SHAPE",
    )
    require(
        gold_label in forced.FORCED_CLASS_ORDER,
        f"GOLD_LABEL:{gold_label}",
    )
    refute = logits[int(branch_tokens["REFUTE"])]
    support = logits[int(branch_tokens["SUPPORT"])]
    if gold_label == "SUPPORT":
        return support - refute
    return refute - support


def summarize_directional_values(
    directional: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    require(set(directional) == set(PLANES), "DIRECTIONAL_PLANES")
    chi: dict[str, float] = {}
    normalized: dict[str, dict[str, float]] = {}
    for plane in PLANES:
        require(
            set(directional[plane]) == set(BASIS_NAMES),
            f"DIRECTIONAL_BASIS:{plane}",
        )
        normalized[plane] = {}
        squares = 0.0
        for basis_name in BASIS_NAMES:
            value = float(directional[plane][basis_name])
            require(math.isfinite(value), f"DIRECTIONAL_FINITE:{plane}:{basis_name}")
            normalized[plane][basis_name] = value
            squares += value * value
        chi_value = math.sqrt(squares)
        require(math.isfinite(chi_value), f"CHI_FINITE:{plane}")
        chi[plane] = chi_value

    d_t = chi[SELECTED_PLANE] - chi[CONTROL_PLANE]
    require(math.isfinite(d_t), "D_T_FINITE")
    return {
        "directional_derivatives": normalized,
        "chi_p3": chi[SELECTED_PLANE],
        "chi_p5": chi[CONTROL_PLANE],
        "d_t": d_t,
    }


def analytic_vjp_observation(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
) -> dict[str, Any]:
    require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1,
        "INPUT_SHAPE",
    )
    require(input_ids.dtype == torch.long, "INPUT_DTYPE")
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "MODEL_PARAMETERS_REQUIRE_GRAD",
    )
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "MODEL_PARAMETER_GRAD_PREEXISTS",
    )

    token_index = int(input_ids.shape[1]) - 1
    require(token_index >= 0, "EMPTY_INPUT")
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
            margin = differentiable_gold_margin(
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
    require(bool(torch.isfinite(gradient).all().item()), "GRADIENT_NONFINITE")

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
    require(tuple(strong_gradient.shape) == (DIM,), "STRONG_GRADIENT_SHAPE")
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
            require(tuple(basis.shape) == (DIM,), f"BASIS_SHAPE:{plane}:{basis_name}")
            require(
                bool(torch.isfinite(basis).all().item()),
                f"BASIS_NONFINITE:{plane}:{basis_name}",
            )
            value = float(torch.dot(strong_gradient, basis).item())
            require(
                math.isfinite(value),
                f"DIRECTIONAL_NONFINITE:{plane}:{basis_name}",
            )
            directional[plane][basis_name] = value

    summary = summarize_directional_values(directional)

    branch_logits = {
        label: float(
            next_logits[int(branch_tokens[label])].detach().cpu().item()
        )
        for label in forced.FORCED_CLASS_ORDER
    }
    f_t = float(margin.detach().cpu().item())
    require(math.isfinite(f_t), "F_T_NONFINITE")
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "MODEL_PARAMETER_GRAD_CREATED",
    )

    return {
        "gold_aligned_f_t": f_t,
        "branch_start_logits": branch_logits,
        "strong_gradient_l2": float(
            torch.linalg.vector_norm(strong_gradient).item()
        ),
        **summary,
        "model_forward_count": 1,
        "local_vjp_count": 1,
        "training_executed": False,
        "parameter_gradient_created": False,
        "parameter_update_executed": False,
    }


def _compare_scalar(
    cpu_value: Any,
    gpu_value: Any,
    *,
    label: str,
) -> float:
    cpu = float(cpu_value)
    gpu = float(gpu_value)
    require(
        math.isfinite(cpu) and math.isfinite(gpu),
        f"NONFINITE:{label}",
    )
    diff = abs(gpu - cpu)
    limit = EQUIVALENCE_ATOL + EQUIVALENCE_RTOL * abs(cpu)
    require(
        diff <= limit,
        f"EQUIVALENCE_FAILURE:{label}:{diff}:{limit}",
    )
    return diff


def compare_observations(
    cpu: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    diffs: dict[str, float] = {
        "f_t_abs_diff": _compare_scalar(
            cpu["gold_aligned_f_t"],
            gpu["gold_aligned_f_t"],
            label="f_t",
        ),
    }

    for plane in PLANES:
        for basis_name in BASIS_NAMES:
            key = f"{plane.lower()}_{basis_name}_abs_diff"
            diffs[key] = _compare_scalar(
                cpu["directional_derivatives"][plane][basis_name],
                gpu["directional_derivatives"][plane][basis_name],
                label=f"directional:{plane}:{basis_name}",
            )

    diffs["chi_p3_abs_diff"] = _compare_scalar(
        cpu["chi_p3"], gpu["chi_p3"], label="chi_p3"
    )
    diffs["chi_p5_abs_diff"] = _compare_scalar(
        cpu["chi_p5"], gpu["chi_p5"], label="chi_p5"
    )
    diffs["d_t_abs_diff"] = _compare_scalar(
        cpu["d_t"], gpu["d_t"], label="d_t"
    )
    return {
        **diffs,
        "result": "PASS",
    }


def _identity_exact(
    cpu_identity: Mapping[str, Any],
    gpu_identity: Mapping[str, Any],
) -> None:
    for key in (
        "backbone_canonical_sha256",
        "lm_head_weight_sha256",
        "input_embedding_weight_sha256",
        "lm_head_tied_to_input_embeddings",
        "tie_word_embeddings",
    ):
        require(
            cpu_identity[key] == gpu_identity[key],
            f"MODEL_IDENTITY:{key}",
        )


def run_gate(
    *,
    expected_head: str,
    snapshot: Path,
    physical_device: int,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    validate_protocol()
    require(not output_dir.exists(), "OUTPUT_COLLISION")
    require(snapshot.is_dir(), f"SNAPSHOT_DIR:{snapshot}")

    (
        input_ids_cpu,
        historical_row,
        branch_tokens,
        historical_provenance,
    ) = historical_equivalence_input(snapshot=snapshot)
    gold_label = str(historical_row["correct_label"])
    require(gold_label in forced.FORCED_CLASS_ORDER, "HISTORICAL_GOLD")

    cpu_model, cpu_identity = identity_gate.load_causal_lm_identity(snapshot)
    frozen_cpu = base.discovery.load_frozen_geometry()
    cpu_runtime = base.runtime_components_for_causal_lm(
        cpu_model,
        frozen_cpu,
    )
    require(
        all(parameter.device.type == "cpu" for parameter in cpu_model.parameters()),
        "CPU_MODEL_DEVICE",
    )
    cpu_observation = analytic_vjp_observation(
        model=cpu_model,
        runtime=cpu_runtime,
        frozen=frozen_cpu,
        input_ids=input_ids_cpu,
        gold_label=gold_label,
        branch_tokens=branch_tokens,
    )
    cpu_strong_mask = (
        cpu_runtime["strong_mask"].detach().cpu().bool().contiguous().clone()
    )
    del cpu_runtime
    del frozen_cpu
    del cpu_model
    gc.collect()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    gpu_model, device, gpu_runtime, frozen_gpu, gpu_identity = (
        base.load_generation_model(
            snapshot=snapshot,
            physical_device=physical_device,
        )
    )
    require(str(device) == "cuda:0", f"GPU_LOGICAL_DEVICE:{device}")
    input_ids_gpu = input_ids_cpu.to(device).contiguous()

    gpu_observation = analytic_vjp_observation(
        model=gpu_model,
        runtime=gpu_runtime,
        frozen=frozen_gpu,
        input_ids=input_ids_gpu,
        gold_label=gold_label,
        branch_tokens=branch_tokens,
    )
    torch.cuda.synchronize(device)

    _identity_exact(cpu_identity, gpu_identity)
    gpu_strong_mask = (
        gpu_runtime["strong_mask"].detach().cpu().bool().contiguous()
    )
    require(
        torch.equal(cpu_strong_mask, gpu_strong_mask),
        "STRONG_MASK_MISMATCH",
    )
    require(
        int(cpu_strong_mask.sum().item()) == DIM,
        "STRONG_MASK_DIM",
    )

    comparison = compare_observations(cpu_observation, gpu_observation)

    kernel_compat = base.discovery.kernel_compat
    report = {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_ancestor": REQUIRED_ANCESTOR,
        "design": {
            "path": DESIGN_ARTIFACT.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
        },
        "historical_input": historical_provenance,
        "historical_raw_evidence_reference": {
            "root": HISTORICAL_RAW_ROOT.as_posix(),
            "summary_git_blob": HISTORICAL_RAW_SUMMARY_GIT_BLOB,
            "rows_git_blob": HISTORICAL_RAW_ROWS_GIT_BLOB,
            "sums_git_blob": HISTORICAL_RAW_SUMS_GIT_BLOB,
            "raw_response_used_for_selection": False,
        },
        "measurement": {
            "name": "analytic_local_differential_susceptibility",
            "relative_offset": RELATIVE_OFFSET,
            "generated_prefix_length": GENERATED_PREFIX_LENGTH,
            "epsilon": None,
            "planes": list(PLANES),
            "basis_names": list(BASIS_NAMES),
            "strong_dim": DIM,
            "intervention_block": 35,
            "readout_block": 47,
            "local_leaf": "block35_in_proj_full_output",
            "projected_gradient_slice": "content_half_strong_channels_at_current_token",
            "projection_reduction": "canonical_cpu_float64",
            "model_parameters_require_grad": False,
        },
        "tolerances": {
            "atol": EQUIVALENCE_ATOL,
            "rtol": EQUIVALENCE_RTOL,
        },
        "comparison": comparison,
        "cpu": {
            "execution_mode": "cpu_slow_transformers_mamba_fallback",
            "use_mamba_kernels": False,
            "model_identity": cpu_identity,
            "observation": cpu_observation,
        },
        "cuda": {
            "execution_mode": "cuda_fast_exact_project_kernels",
            "logical_device": str(device),
            "physical_device_requested": int(physical_device),
            "model_identity": gpu_identity,
            "observation": gpu_observation,
        },
        "kernel_provenance": {
            "kernels_version": kernel_compat.KERNELS_VERSION,
            "build_variant": kernel_compat.BUILD_VARIANT,
            "mamba_scientific_revision":
                kernel_compat.MAMBA_SPEC.scientific_revision,
            "mamba_binary_sha256":
                kernel_compat.MAMBA_SPEC.binary_sha256,
            "causal_conv_scientific_revision":
                kernel_compat.CONV_SPEC.scientific_revision,
            "causal_conv_binary_sha256":
                kernel_compat.CONV_SPEC.binary_sha256,
            "gpu_kernel_constructor_calls":
                dict(gpu_identity["kernel_constructor_calls"]),
        },
        "accounting": {
            "cpu_model_forward_count": CPU_MODEL_FORWARD_COUNT,
            "gpu_model_forward_count": GPU_MODEL_FORWARD_COUNT,
            "cpu_local_vjp_count": CPU_VJP_COUNT,
            "gpu_local_vjp_count": GPU_VJP_COUNT,
            "total_equivalence_model_forward_count":
                TOTAL_EQUIVALENCE_FORWARD_COUNT,
            "total_equivalence_local_vjp_count":
                TOTAL_EQUIVALENCE_VJP_COUNT,
            "scientific_model_forward_count": 0,
            "scientific_backward_count": 0,
        },
        "boundary": {
            "fresh_n800_scientific_cohort_accessed": False,
            "historical_generation_response_used_for_selection": False,
            "scientific_inference_executed": False,
            "statistical_testing_performed": False,
            "p_value_count_added": 0,
            "training_executed": False,
            "parameter_gradient_created": False,
            "parameter_update_executed": False,
            "scientific_conclusion": None,
            "n800_runner_implemented_by_this_artifact": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / REPORT_FILE
    sums_path = output_dir / SUMS_FILE
    report_path.write_text(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sums_path.write_text(
        f"{sha256_file(report_path)}  {REPORT_FILE}\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Historical one-row CPU-slow versus CUDA-fast analytic VJP "
            "equivalence gate for Precursor-v4. No N=800 access."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_gate(
        expected_head=str(args.expected_head),
        snapshot=args.model_snapshot,
        physical_device=int(args.physical_device),
        output_dir=args.output_dir,
    )
    comparison = report["comparison"]
    print("RESULT=" + str(report["result"]))
    print("HISTORICAL_EXAMPLE_ID=" + str(report["historical_input"]["example_id"]))
    print("RELATIVE_OFFSET=-4")
    print("CPU_MODEL_FORWARD_COUNT=1")
    print("GPU_MODEL_FORWARD_COUNT=1")
    print("CPU_LOCAL_VJP_COUNT=1")
    print("GPU_LOCAL_VJP_COUNT=1")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("F_T_ABS_DIFF=" + repr(comparison["f_t_abs_diff"]))
    print("P3_PLUS_ABS_DIFF=" + repr(comparison["p3_plus_abs_diff"]))
    print("P3_MINUS_ABS_DIFF=" + repr(comparison["p3_minus_abs_diff"]))
    print("P5_PLUS_ABS_DIFF=" + repr(comparison["p5_plus_abs_diff"]))
    print("P5_MINUS_ABS_DIFF=" + repr(comparison["p5_minus_abs_diff"]))
    print("CHI_P3_ABS_DIFF=" + repr(comparison["chi_p3_abs_diff"]))
    print("CHI_P5_ABS_DIFF=" + repr(comparison["chi_p5_abs_diff"]))
    print("D_T_ABS_DIFF=" + repr(comparison["d_t_abs_diff"]))
    print("FRESH_N800_SCIENTIFIC_COHORT_ACCESSED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
