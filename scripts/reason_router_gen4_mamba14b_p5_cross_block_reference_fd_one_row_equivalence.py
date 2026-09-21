#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (
    reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility
    as direct,
)

EXPECTED_BRANCH = direct.EXPECTED_BRANCH
REQUIRED_PROGRAM_ANCESTOR = "fa4a68054176412ec1e30be4d14dd7202314435f"

PROGRAM_PATH = Path(
    "reports/reason_router_gen4_next_mechanistic_program_prospective_freeze.md"
)
PROGRAM_GIT_BLOB = "487bf844e27279392a4203c1dc70e5702e3ad325"

ROR_SPEC_PATH = Path(
    "reports/reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_execution_spec_candidate.md"
)
ROR_SPEC_GIT_BLOB = "ae42198e7c55a560aec0aa72bf770a02cdb225a8"

ROR_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_reverse_over_reverse_one_row_feasibility.py"
)
ROR_SCRIPT_GIT_BLOB = "20f6ba4fcd2aad677d0e20e0c73f9f62d05ceb8b"

TRANSFORMERS_REFERENCE_REPO = "huggingface/transformers"
TRANSFORMERS_REFERENCE_TAG = "v5.0.0"
TRANSFORMERS_REFERENCE_COMMIT = "08810b1e278938278c50153ee1edfd7a20a759da"
TRANSFORMERS_MAMBA_SOURCE_PATH = (
    "src/transformers/models/mamba/modeling_mamba.py"
)
TRANSFORMERS_MAMBA_SOURCE_GIT_BLOB = (
    "ae80aa74f651f15a82bfc41ece60ba51ed5bb206"
)

FAMILY = direct.FAMILY
PAIR = direct.PAIR
CELL = direct.CELL
ANCHOR_NAME = direct.ANCHOR_NAME
TARGET_OFFSET = direct.TARGET_OFFSET
SOURCE_BLOCK = direct.SOURCE_BLOCK
TARGET_BLOCK = direct.TARGET_BLOCK
INTERMEDIATE_SIZE = direct.INTERMEDIATE_SIZE
PROJECTED_WIDTH = direct.PROJECTED_WIDTH
SOURCE_PLANE = direct.SOURCE_PLANE
BASIS_NAMES = direct.BASIS_NAMES

EPSILON = 0.025

PRIMAL_ATOL = 1e-4
PRIMAL_RTOL = 1e-4

DIRECTION_COSINE_MIN = 0.9998
DIRECTION_NORM_RATIO_MIN = 0.98
DIRECTION_NORM_RATIO_MAX = 1.02
DIRECTION_REL_L2_MAX = 0.02

RESULT_PASS = (
    "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REFERENCE_FD_EQUIVALENCE"
)

REPORT_FILE = "reference_fd_equivalence_report.json"
REF_PLUS_FILE = "reference_exact_p5_plus.f64le"
REF_MINUS_FILE = "reference_exact_p5_minus.f64le"
FD_PLUS_FILE = "fast_fd_p5_plus.f64le"
FD_MINUS_FILE = "fast_fd_p5_minus.f64le"
SUMS_FILE = "SHA256SUMS.txt"


class TransportReferenceFDError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportReferenceFDError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransportReferenceFDError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def authenticate_repo(expected_head: str) -> None:
    direct.authenticate_repo(expected_head)
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_PROGRAM_ANCESTOR,
            expected_head,
        )
        == 0,
        "PROGRAM_FREEZE_NOT_ANCESTOR",
    )

    pinned = {
        PROGRAM_PATH: PROGRAM_GIT_BLOB,
        ROR_SPEC_PATH: ROR_SPEC_GIT_BLOB,
        ROR_SCRIPT_PATH: ROR_SCRIPT_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    direct.validate_protocol()

    require(FAMILY == "xg2", "FAMILY")
    require(PAIR == "xg2_fact_301", "PAIR")
    require(CELL == "C2_NAME", "CELL")
    require(ANCHOR_NAME == "A_IDENTITY", "ANCHOR_NAME")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(SOURCE_BLOCK == 35, "SOURCE_BLOCK")
    require(TARGET_BLOCK == 36, "TARGET_BLOCK")
    require(INTERMEDIATE_SIZE == 4096, "INTERMEDIATE_SIZE")
    require(PROJECTED_WIDTH == 8192, "PROJECTED_WIDTH")
    require(SOURCE_PLANE == "P5", "SOURCE_PLANE")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")

    require(EPSILON == 0.025, "EPSILON")
    require(PRIMAL_ATOL == 1e-4, "PRIMAL_ATOL")
    require(PRIMAL_RTOL == 1e-4, "PRIMAL_RTOL")
    require(DIRECTION_COSINE_MIN == 0.9998, "DIRECTION_COSINE_MIN")
    require(
        DIRECTION_NORM_RATIO_MIN == 0.98,
        "DIRECTION_NORM_RATIO_MIN",
    )
    require(
        DIRECTION_NORM_RATIO_MAX == 1.02,
        "DIRECTION_NORM_RATIO_MAX",
    )
    require(DIRECTION_REL_L2_MAX == 0.02, "DIRECTION_REL_L2_MAX")


def reference_slow_block35_to_block36_map(
    *,
    projected35: torch.Tensor,
    residual35: torch.Tensor,
    target_abs: int,
    layer35: Any,
    layer36: Any,
) -> torch.Tensor:
    """
    Differentiable unfused reference for the exact frozen local map.

    This is the Transformers v5.0.0 MambaMixer.slow_forward semantics from
    immediately after in_proj through block35 output projection, residual
    addition, block36 norm, and block36 in_proj.

    Reference source:
      huggingface/transformers v5.0.0
      commit 08810b1e278938278c50153ee1edfd7a20a759da
      modeling_mamba.py blob ae80aa74f651f15a82bfc41ece60ba51ed5bb206
    """
    require(
        tuple(projected35.shape) == (1, 128, PROJECTED_WIDTH),
        "REFERENCE_PROJECTED_SHAPE",
    )
    require(
        tuple(residual35.shape) == (1, 128, direct.geom.HIDDEN_SIZE),
        "REFERENCE_RESIDUAL_SHAPE",
    )
    require(0 <= target_abs < 128, "REFERENCE_TARGET_RANGE")

    mixer35 = layer35.mixer
    projected_states = projected35.transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    batch_size = int(hidden_states.shape[0])
    seq_len = int(hidden_states.shape[-1])
    dtype = hidden_states.dtype

    # Transformers v5.0.0 slow_forward, no cache and no attention mask.
    hidden_states = mixer35.act(
        mixer35.conv1d(hidden_states)[..., :seq_len]
    )

    ssm_parameters = mixer35.x_proj(
        hidden_states.transpose(1, 2)
    )
    time_step, b_scan, c_scan = torch.split(
        ssm_parameters,
        [
            mixer35.time_step_rank,
            mixer35.ssm_state_size,
            mixer35.ssm_state_size,
        ],
        dim=-1,
    )

    discrete_time_step = F.softplus(
        mixer35.dt_proj(time_step)
    ).transpose(1, 2)

    a_matrix = -torch.exp(mixer35.A_log.float())
    discrete_a = torch.exp(
        a_matrix[None, :, None, :]
        * discrete_time_step[:, :, :, None]
    )
    discrete_b = (
        discrete_time_step[:, :, :, None]
        * b_scan[:, None, :, :].float()
    )
    delta_b_u = (
        discrete_b
        * hidden_states[:, :, :, None].float()
    )

    ssm_state = torch.zeros(
        (
            batch_size,
            mixer35.intermediate_size,
            mixer35.ssm_state_size,
        ),
        device=hidden_states.device,
        dtype=dtype,
    )

    scan_outputs: list[torch.Tensor] = []
    for token_index in range(seq_len):
        ssm_state = (
            discrete_a[:, :, token_index, :] * ssm_state
            + delta_b_u[:, :, token_index, :]
        )
        scan_output = torch.matmul(
            ssm_state.to(dtype),
            c_scan[:, token_index, :].unsqueeze(-1),
        )
        scan_outputs.append(scan_output[:, :, 0])

    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = (
        scan_output
        + hidden_states * mixer35.D[None, :, None]
    )
    scan_output = scan_output * mixer35.act(gate)

    contextualized = mixer35.out_proj(
        scan_output.transpose(1, 2)
    )
    block35_output = residual35 + contextualized

    norm36 = layer36.norm(
        block35_output.to(dtype=layer36.norm.weight.dtype)
    )
    projected36 = layer36.mixer.in_proj(norm36)

    require(
        tuple(projected36.shape) == (1, 128, PROJECTED_WIDTH),
        "REFERENCE_PROJECTED36_SHAPE",
    )
    return projected36[
        0,
        target_abs,
        :INTERMEDIATE_SIZE,
    ]


def exact_reference_jvp(
    phi: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = direction.to(
        device=base.device,
        dtype=base.dtype,
    ).contiguous()
    require(base.shape == tangent.shape, "REFERENCE_JVP_SHAPE")

    try:
        primal, transported = torch.func.jvp(
            phi,
            (base,),
            (tangent,),
        )
    except Exception as exc:
        raise TransportReferenceFDError(
            "REFERENCE_EXACT_JVP_UNSUPPORTED:"
            f"{type(exc).__name__}:{exc}"
        ) from exc

    require(
        primal.ndim == 1,
        "REFERENCE_PRIMAL_RANK",
    )
    require(
        transported.shape == primal.shape,
        "REFERENCE_JVP_OUTPUT_SHAPE",
    )
    require(
        bool(torch.isfinite(primal).all().item()),
        "REFERENCE_PRIMAL_NONFINITE",
    )
    require(
        bool(torch.isfinite(transported).all().item()),
        "REFERENCE_JVP_NONFINITE",
    )
    return primal, transported


def fast_symmetric_fd(
    phi: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tangent = direction.to(
        device=base.device,
        dtype=base.dtype,
    ).contiguous()
    require(base.shape == tangent.shape, "FD_SHAPE")

    with torch.inference_mode():
        plus = phi(base + EPSILON * tangent)
        minus = phi(base - EPSILON * tangent)

    require(
        plus.ndim == 1,
        "FD_PLUS_RANK",
    )
    require(
        minus.shape == plus.shape,
        "FD_MINUS_SHAPE",
    )
    require(
        bool(torch.isfinite(plus).all().item()),
        "FD_PLUS_NONFINITE",
    )
    require(
        bool(torch.isfinite(minus).all().item()),
        "FD_MINUS_NONFINITE",
    )

    derivative = (plus - minus) / (2.0 * EPSILON)
    require(
        bool(torch.isfinite(derivative).all().item()),
        "FD_DERIVATIVE_NONFINITE",
    )
    return plus, minus, derivative


def compare_direction(
    reference: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, float]:
    ref = reference.detach().cpu().to(torch.float64).contiguous()
    obs = observed.detach().cpu().to(torch.float64).contiguous()

    require(
        tuple(ref.shape) == (INTERMEDIATE_SIZE,),
        "COMPARE_REFERENCE_SHAPE",
    )
    require(
        tuple(obs.shape) == (INTERMEDIATE_SIZE,),
        "COMPARE_OBSERVED_SHAPE",
    )
    require(
        bool(torch.isfinite(ref).all().item())
        and bool(torch.isfinite(obs).all().item()),
        "COMPARE_NONFINITE",
    )

    ref_norm = float(torch.linalg.vector_norm(ref).item())
    obs_norm = float(torch.linalg.vector_norm(obs).item())
    require(ref_norm > 0.0, "REFERENCE_NORM_ZERO")
    require(obs_norm > 0.0, "OBSERVED_NORM_ZERO")

    diff = obs - ref
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    relative_l2 = diff_norm / ref_norm
    cosine = float(
        torch.dot(ref, obs).item()
        / (ref_norm * obs_norm)
    )
    norm_ratio = obs_norm / ref_norm
    max_abs = float(torch.max(torch.abs(diff)).item())

    require(
        math.isfinite(relative_l2)
        and math.isfinite(cosine)
        and math.isfinite(norm_ratio)
        and math.isfinite(max_abs),
        "COMPARE_METRIC_NONFINITE",
    )

    passed = (
        cosine >= DIRECTION_COSINE_MIN
        and DIRECTION_NORM_RATIO_MIN
        <= norm_ratio
        <= DIRECTION_NORM_RATIO_MAX
        and relative_l2 <= DIRECTION_REL_L2_MAX
    )

    return {
        "reference_l2": ref_norm,
        "fast_fd_l2": obs_norm,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "norm_ratio_fast_fd_over_reference": norm_ratio,
        "max_abs_error": max_abs,
        "pass": bool(passed),
    }


def compare_primal(
    reference: torch.Tensor,
    fast: torch.Tensor,
) -> dict[str, float | bool]:
    ref = reference.detach().cpu().to(torch.float64).contiguous()
    obs = fast.detach().cpu().to(torch.float64).contiguous()

    require(ref.shape == obs.shape, "PRIMAL_SHAPE")
    diff = torch.abs(obs - ref)
    max_abs = float(torch.max(diff).item())
    scale = float(torch.max(torch.abs(ref)).item())
    limit = PRIMAL_ATOL + PRIMAL_RTOL * scale

    require(
        math.isfinite(max_abs)
        and math.isfinite(scale)
        and math.isfinite(limit),
        "PRIMAL_METRIC_NONFINITE",
    )
    return {
        "max_abs_error": max_abs,
        "reference_max_abs": scale,
        "allclose_limit_at_reference_max": limit,
        "pass": bool(
            torch.allclose(
                obs,
                ref,
                atol=PRIMAL_ATOL,
                rtol=PRIMAL_RTOL,
            )
        ),
    }


def run_gate(
    *,
    expected_head: str,
    snapshot: Path,
    compact_checkpoint: Path,
    output_dir: Path,
    gpu_id: int,
) -> dict[str, Any]:
    validate_protocol()
    authenticate_repo(expected_head)
    require(not output_dir.exists(), f"OUTPUT_COLLISION:{output_dir}")
    require(gpu_id == 0, "GPU_ID_GATE_FIXED")
    direct.geom.validate_fast_runtime_for_device(gpu_id)

    input_ids, target_abs, input_meta = direct.load_gate_input(snapshot)
    p5 = direct.load_canonical_p5_ambient()

    model, kernels, model_provenance = direct.geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=compact_checkpoint,
        gpu_id=gpu_id,
    )
    device = torch.device(f"cuda:{gpu_id}")
    layers = model.mamba.layers
    layer35 = layers[SOURCE_BLOCK]
    layer36 = layers[TARGET_BLOCK]

    require(
        not any(
            parameter.requires_grad
            for parameter in model.mamba.parameters()
        ),
        "MODEL_PARAMETERS_REQUIRE_GRAD",
    )

    baseline = direct.capture_baseline_boundary(
        model=model,
        input_ids=input_ids,
        target_abs=target_abs,
        device=device,
    )
    baseline_projected35 = (
        baseline["projected35"].to(device).contiguous()
    )
    residual35 = baseline["residual35"].to(device).contiguous()
    base_content = (
        baseline_projected35[
            0,
            target_abs,
            :INTERMEDIATE_SIZE,
        ]
        .detach()
        .clone()
    )

    def build_projected(content: torch.Tensor) -> torch.Tensor:
        return direct.replace_target_content(
            baseline_projected35,
            content,
            target_abs,
        )

    def reference_phi(content: torch.Tensor) -> torch.Tensor:
        return reference_slow_block35_to_block36_map(
            projected35=build_projected(content),
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
        )

    def fast_phi(content: torch.Tensor) -> torch.Tensor:
        return direct.local_block35_to_block36_map(
            projected35=build_projected(content),
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
            kernels=kernels,
        )

    with torch.inference_mode():
        fast_primal = fast_phi(base_content)

    reference_vectors: dict[str, torch.Tensor] = {}
    fd_vectors: dict[str, torch.Tensor] = {}
    reference_primals: list[torch.Tensor] = []
    comparisons: dict[str, dict[str, float | bool]] = {}

    for basis_name in BASIS_NAMES:
        reference_primal, reference_jv = exact_reference_jvp(
            reference_phi,
            base_content,
            p5[basis_name],
        )
        _, _, fd_jv = fast_symmetric_fd(
            fast_phi,
            base_content,
            p5[basis_name],
        )

        reference_primals.append(reference_primal.detach())
        reference_vectors[basis_name] = (
            reference_jv.detach()
            .cpu().to(torch.float64).contiguous()
        )
        fd_vectors[basis_name] = (
            fd_jv.detach()
            .cpu().to(torch.float64).contiguous()
        )
        comparisons[basis_name] = compare_direction(
            reference_vectors[basis_name],
            fd_vectors[basis_name],
        )

    primal_repeat_diff = float(
        torch.max(
            torch.abs(
                reference_primals[0].detach().float()
                - reference_primals[1].detach().float()
            )
        ).item()
    )
    require(
        primal_repeat_diff == 0.0,
        f"REFERENCE_PRIMAL_REPEAT:{primal_repeat_diff}",
    )

    primal_comparison = compare_primal(
        reference_primals[0],
        fast_primal,
    )
    require(
        bool(primal_comparison["pass"]),
        "REFERENCE_FAST_PRIMAL_EQUIVALENCE",
    )

    for basis_name in BASIS_NAMES:
        require(
            bool(comparisons[basis_name]["pass"]),
            f"REFERENCE_FD_DIRECTION_EQUIVALENCE:{basis_name}",
        )

    reference_rank = direct.numerical_rank_two(
        [
            reference_vectors["plus"],
            reference_vectors["minus"],
        ]
    )
    fd_rank = direct.numerical_rank_two(
        [
            fd_vectors["plus"],
            fd_vectors["minus"],
        ]
    )

    raw_files = {
        REF_PLUS_FILE: direct.raw_f64le(
            reference_vectors["plus"]
        ),
        REF_MINUS_FILE: direct.raw_f64le(
            reference_vectors["minus"]
        ),
        FD_PLUS_FILE: direct.raw_f64le(fd_vectors["plus"]),
        FD_MINUS_FILE: direct.raw_f64le(fd_vectors["minus"]),
    }

    report = {
        "schema_version":
            "gen4-mamba14b-p5-cross-block-one-row-reference-fd-equivalence-v1",
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_program_ancestor": REQUIRED_PROGRAM_ANCESTOR,
        "gate_row": {
            "family": FAMILY,
            "source_pair_id": PAIR,
            "contrast_cell_id": CELL,
            "anchor_name": ANCHOR_NAME,
            "target_offset": TARGET_OFFSET,
            **input_meta,
        },
        "transport_definition": {
            "source_block": SOURCE_BLOCK,
            "target_block": TARGET_BLOCK,
            "source_tensor":
                "block35_mixer_in_proj_output_content_half",
            "target_tensor":
                "block36_mixer_in_proj_output_content_half",
            "ambient_dimension": INTERMEDIATE_SIZE,
            "source_plane": SOURCE_PLANE,
            "basis_order": list(BASIS_NAMES),
            "gate_half_fixed": True,
            "non_target_tokens_fixed_at_source_boundary": True,
        },
        "reference": {
            "backend":
                "transformers_v5.0.0_unfused_slow_forward_semantics",
            "exact_autodiff": "torch.func.jvp",
            "repo": TRANSFORMERS_REFERENCE_REPO,
            "tag": TRANSFORMERS_REFERENCE_TAG,
            "commit": TRANSFORMERS_REFERENCE_COMMIT,
            "source_path": TRANSFORMERS_MAMBA_SOURCE_PATH,
            "source_git_blob": TRANSFORMERS_MAMBA_SOURCE_GIT_BLOB,
        },
        "fast_estimator": {
            "backend": "authenticated_exact_fast_cuda",
            "estimator": "symmetric_central_difference",
            "epsilon": EPSILON,
            "epsilon_selection":
                "prospectively_frozen_from_mamba14b_experiment5_intervention_scale",
            "epsilon_sweep_performed": False,
        },
        "equivalence_thresholds": {
            "primal_atol": PRIMAL_ATOL,
            "primal_rtol": PRIMAL_RTOL,
            "direction_cosine_min": DIRECTION_COSINE_MIN,
            "direction_norm_ratio_min":
                DIRECTION_NORM_RATIO_MIN,
            "direction_norm_ratio_max":
                DIRECTION_NORM_RATIO_MAX,
            "direction_relative_l2_max":
                DIRECTION_REL_L2_MAX,
            "threshold_tuning_after_result": False,
        },
        "primal_comparison": primal_comparison,
        "direction_comparisons": comparisons,
        "reference_span_rank_diagnostic": reference_rank,
        "fast_fd_span_rank_diagnostic": fd_rank,
        "model_provenance": model_provenance,
        "accounting": {
            "baseline_full_model_forward_count": 1,
            "reference_exact_jvp_count": 2,
            "fast_local_baseline_forward_count": 1,
            "fast_local_perturbed_forward_count": 4,
            "fast_local_fd_direction_count": 2,
            "scientific_model_forward_count": 0,
            "scientific_inference_count": 0,
            "p_value_count_added": 0,
        },
        "boundary": {
            "experiment5_xg1_response_accessed": False,
            "population_transport_executed": False,
            "adjacent_p5_accessed": False,
            "principal_angles_computed": False,
            "projector_overlap_computed": False,
            "procrustes_alignment_computed": False,
            "statistical_testing_performed": False,
            "scientific_conclusion": None,
            "training_executed": False,
            "parameter_gradient_created": any(
                parameter.grad is not None
                for parameter in model.parameters()
            ),
            "parameter_update_executed": False,
        },
    }

    require(
        report["boundary"]["parameter_gradient_created"] is False,
        "PARAMETER_GRAD_CREATED",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name, raw in raw_files.items():
        (output_dir / name).write_bytes(raw)
        hashes[name] = direct.sha256_bytes(raw)

    report_bytes = direct.pretty_json_bytes(report)
    (output_dir / REPORT_FILE).write_bytes(report_bytes)
    hashes[REPORT_FILE] = direct.sha256_bytes(report_bytes)

    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded one-row Mamba-1.4B canonical-P5 "
            "block35-to-block36 reference exact-autodiff versus fixed-"
            "epsilon fast-CUDA finite-difference equivalence gate. "
            "No population transport metric or statistical test is computed."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_gate(
        expected_head=str(args.expected_head),
        snapshot=args.snapshot,
        compact_checkpoint=args.compact_checkpoint,
        output_dir=args.output_dir,
        gpu_id=int(args.gpu_id),
    )

    print("RESULT=" + str(report["result"]))
    print("FAMILY=xg2")
    print("PAIR=xg2_fact_301")
    print("CELL=C2_NAME")
    print("ANCHOR=A_IDENTITY")
    print("TARGET_OFFSET=2")
    print("SOURCE_BLOCK=35")
    print("TARGET_BLOCK=36")
    print("EPSILON=0.025")
    print(
        "PLUS_COSINE="
        + repr(
            report["direction_comparisons"]["plus"][
                "cosine_similarity"
            ]
        )
    )
    print(
        "PLUS_REL_L2="
        + repr(
            report["direction_comparisons"]["plus"][
                "relative_l2_error"
            ]
        )
    )
    print(
        "MINUS_COSINE="
        + repr(
            report["direction_comparisons"]["minus"][
                "cosine_similarity"
            ]
        )
    )
    print(
        "MINUS_REL_L2="
        + repr(
            report["direction_comparisons"]["minus"][
                "relative_l2_error"
            ]
        )
    )
    print("REFERENCE_EXACT_JVP_COUNT=2")
    print("FAST_LOCAL_PERTURBED_FORWARD_COUNT=4")
    print("EPSILON_SWEEP_PERFORMED=False")
    print("POPULATION_TRANSPORT_EXECUTED=False")
    print("ADJACENT_P5_ACCESSED=False")
    print("PRINCIPAL_ANGLES_COMPUTED=False")
    print("PROJECTOR_OVERLAP_COMPUTED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
