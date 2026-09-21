#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]

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
DIRECT_JVP_CLOSURE_PATH = Path(
    "reports/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_technical_closure_report_candidate.md"
)
DIRECT_JVP_CLOSURE_GIT_BLOB = "eba094c517b38d8730eb9b8362f9fa6f69b295b5"
DIRECT_JVP_SCRIPT_PATH = Path(
    "scripts/reason_router_gen4_mamba14b_p5_cross_block_jvp_one_row_feasibility.py"
)
DIRECT_JVP_SCRIPT_GIT_BLOB = "db523f66d9a05297077af18b547b4f2b2c9379ea"

FAMILY = direct.FAMILY
PAIR = direct.PAIR
CELL = direct.CELL
ANCHOR_NAME = direct.ANCHOR_NAME
TARGET_OFFSET = direct.TARGET_OFFSET
SOURCE_BLOCK = direct.SOURCE_BLOCK
TARGET_BLOCK = direct.TARGET_BLOCK
INTERMEDIATE_SIZE = direct.INTERMEDIATE_SIZE
SOURCE_PLANE = direct.SOURCE_PLANE
BASIS_NAMES = direct.BASIS_NAMES

RESULT_PASS = (
    "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_REVERSE_OVER_REVERSE_FEASIBILITY"
)
REPORT_FILE = "reverse_over_reverse_feasibility_report.json"
PLUS_FILE = "transported_p5_plus.f64le"
MINUS_FILE = "transported_p5_minus.f64le"
SUMS_FILE = "SHA256SUMS.txt"


class TransportReverseOverReverseError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportReverseOverReverseError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransportReverseOverReverseError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def authenticate_repo(expected_head: str) -> None:
    direct.authenticate_repo(expected_head)
    require(
        git_rc("merge-base", "--is-ancestor", REQUIRED_PROGRAM_ANCESTOR, expected_head) == 0,
        "PROGRAM_FREEZE_NOT_ANCESTOR",
    )
    pinned = {
        PROGRAM_PATH: PROGRAM_GIT_BLOB,
        DIRECT_JVP_CLOSURE_PATH: DIRECT_JVP_CLOSURE_GIT_BLOB,
        DIRECT_JVP_SCRIPT_PATH: DIRECT_JVP_SCRIPT_GIT_BLOB,
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
    require(SOURCE_PLANE == "P5", "SOURCE_PLANE")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")


def reverse_over_reverse_jvp(
    phi: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Jv via grad_w[v^T grad_x(w^T Phi(x))]."""
    require(torch.is_tensor(base), "BASE_NOT_TENSOR")
    require(torch.is_tensor(direction), "DIRECTION_NOT_TENSOR")
    require(base.shape == direction.shape, "BASE_DIRECTION_SHAPE")
    require(bool(torch.isfinite(base).all().item()), "BASE_NONFINITE")
    require(bool(torch.isfinite(direction).all().item()), "DIRECTION_NONFINITE")

    x = base.detach().clone().requires_grad_(True)
    tangent = direction.detach().to(device=x.device, dtype=x.dtype).contiguous()

    try:
        with torch.enable_grad():
            primal = phi(x)
            require(torch.is_tensor(primal), "PRIMAL_NOT_TENSOR")
            require(bool(torch.isfinite(primal).all().item()), "PRIMAL_NONFINITE")

            auxiliary = torch.ones_like(primal).detach().requires_grad_(True)
            first_pairing = torch.sum(auxiliary * primal)
            grad_x, = torch.autograd.grad(
                first_pairing,
                x,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )
            require(grad_x.shape == x.shape, "FIRST_REVERSE_SHAPE")
            require(
                bool(torch.isfinite(grad_x).all().item()),
                "FIRST_REVERSE_NONFINITE",
            )

            second_pairing = torch.sum(grad_x * tangent)
            transported, = torch.autograd.grad(
                second_pairing,
                auxiliary,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
    except TransportReverseOverReverseError:
        raise
    except Exception as exc:
        raise TransportReverseOverReverseError(
            "DOUBLE_BACKWARD_UNSUPPORTED:"
            f"{type(exc).__name__}:{exc}"
        ) from exc

    require(transported.shape == primal.shape, "TRANSPORTED_SHAPE")
    require(
        bool(torch.isfinite(transported).all().item()),
        "TRANSPORTED_NONFINITE",
    )
    return primal, transported


def compute_one_direction_reverse_over_reverse(
    *,
    base_content: torch.Tensor,
    direction: torch.Tensor,
    baseline_projected35: torch.Tensor,
    residual35: torch.Tensor,
    target_abs: int,
    layer35: Any,
    layer36: Any,
    kernels: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    tangent = direction.to(
        device=base_content.device,
        dtype=base_content.dtype,
    ).contiguous()
    require(tuple(base_content.shape) == (INTERMEDIATE_SIZE,), "BASE_CONTENT_SHAPE")
    require(tuple(tangent.shape) == (INTERMEDIATE_SIZE,), "TANGENT_SHAPE")

    def phi(content: torch.Tensor) -> torch.Tensor:
        projected = direct.replace_target_content(
            baseline_projected35, content, target_abs
        )
        return direct.local_block35_to_block36_map(
            projected35=projected,
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
            kernels=kernels,
        )

    primal, transported = reverse_over_reverse_jvp(phi, base_content, tangent)
    require(tuple(primal.shape) == (INTERMEDIATE_SIZE,), "ROR_PRIMAL_SHAPE")
    require(
        tuple(transported.shape) == (INTERMEDIATE_SIZE,),
        "ROR_TRANSPORTED_SHAPE",
    )
    return primal, transported


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

    baseline = direct.capture_baseline_boundary(
        model=model,
        input_ids=input_ids,
        target_abs=target_abs,
        device=device,
    )
    baseline_projected35 = baseline["projected35"].to(device).contiguous()
    residual35 = baseline["residual35"].to(device).contiguous()
    base_content = (
        baseline_projected35[0, target_abs, :INTERMEDIATE_SIZE]
        .detach().clone()
    )

    transported: dict[str, torch.Tensor] = {}
    primals: list[torch.Tensor] = []
    for basis_name in BASIS_NAMES:
        primal, tangent = compute_one_direction_reverse_over_reverse(
            base_content=base_content,
            direction=p5[basis_name],
            baseline_projected35=baseline_projected35,
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
            kernels=kernels,
        )
        primals.append(primal.detach())
        transported[basis_name] = (
            tangent.detach().cpu().to(torch.float64).contiguous()
        )

    primal_diff = float(
        torch.max(torch.abs(primals[0].detach().float() - primals[1].detach().float())).item()
    )
    require(primal_diff == 0.0, f"ROR_PRIMAL_MISMATCH:{primal_diff}")

    rank_info = direct.numerical_rank_two(
        [transported["plus"], transported["minus"]]
    )
    require(rank_info["rank"] == 2, f"TRANSPORTED_RANK:{rank_info['rank']}")

    norms = {
        name: float(torch.linalg.vector_norm(vector).item())
        for name, vector in transported.items()
    }
    require(
        all(math.isfinite(value) and value > 0.0 for value in norms.values()),
        "TRANSPORTED_NORM",
    )

    plus_raw = direct.raw_f64le(transported["plus"])
    minus_raw = direct.raw_f64le(transported["minus"])

    report = {
        "schema_version":
            "gen4-mamba14b-p5-cross-block-one-row-reverse-over-reverse-feasibility-v1",
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_program_ancestor": REQUIRED_PROGRAM_ANCESTOR,
        "direct_jvp_closure_blob": DIRECT_JVP_CLOSURE_GIT_BLOB,
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
            "source_tensor": "block35_mixer_in_proj_output_content_half",
            "target_tensor": "block36_mixer_in_proj_output_content_half",
            "ambient_dimension": INTERMEDIATE_SIZE,
            "source_plane": SOURCE_PLANE,
            "basis_order": list(BASIS_NAMES),
            "gate_half_fixed": True,
            "non_target_tokens_fixed_at_source_boundary": True,
            "estimator": "reverse_over_reverse_exact_autodiff_Jv",
            "identity": "grad_w(v^T grad_x(w^T Phi(x))) = Jv",
            "auxiliary_vector":
                "ones_like(Phi(x)); exact result is independent of w",
            "direct_torch_func_jvp_used": False,
            "finite_difference_used": False,
            "slow_backend_used": False,
            "alternate_kernel_bundle_used": False,
        },
        "transported_vector_norms": norms,
        "transported_span_numerical_rank": int(rank_info["rank"]),
        "transported_span_rank_tolerance": float(rank_info["rank_tolerance"]),
        "transported_singular_values_for_rank_only": rank_info["singular_values"],
        "model_provenance": model_provenance,
        "accounting": {
            "baseline_full_model_forward_count": 1,
            "local_exact_jv_count": 2,
            "first_reverse_grad_count": 2,
            "second_reverse_grad_count": 2,
            "direct_forward_jvp_count": 0,
            "scientific_model_forward_count": 0,
            "scientific_inference_count": 0,
            "p_value_count_added": 0,
        },
        "boundary": {
            "experiment5_xg1_response_accessed": False,
            "population_transport_executed": False,
            "principal_angles_computed": False,
            "projector_overlap_computed": False,
            "procrustes_alignment_computed": False,
            "statistical_testing_performed": False,
            "scientific_conclusion": None,
            "training_executed": False,
            "parameter_gradient_created": any(
                parameter.grad is not None for parameter in model.parameters()
            ),
            "parameter_update_executed": False,
        },
    }
    require(
        report["boundary"]["parameter_gradient_created"] is False,
        "PARAMETER_GRAD_CREATED",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PLUS_FILE).write_bytes(plus_raw)
    (output_dir / MINUS_FILE).write_bytes(minus_raw)
    report_bytes = direct.pretty_json_bytes(report)
    (output_dir / REPORT_FILE).write_bytes(report_bytes)

    hashes = {
        PLUS_FILE: direct.sha256_bytes(plus_raw),
        MINUS_FILE: direct.sha256_bytes(minus_raw),
        REPORT_FILE: direct.sha256_bytes(report_bytes),
    }
    (output_dir / SUMS_FILE).write_text(
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(hashes.items())
        ),
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded one-row response-free Mamba-1.4B canonical-P5 "
            "block35-to-block36 reverse-over-reverse exact-autodiff Jv "
            "feasibility gate. No population transport metric or statistical "
            "test is computed."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path, required=True)
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
    print("TRANSPORTED_P5_PLUS_L2=" + repr(report["transported_vector_norms"]["plus"]))
    print("TRANSPORTED_P5_MINUS_L2=" + repr(report["transported_vector_norms"]["minus"]))
    print("TRANSPORTED_SPAN_NUMERICAL_RANK=" + str(report["transported_span_numerical_rank"]))
    print("LOCAL_EXACT_JV_COUNT=2")
    print("FIRST_REVERSE_GRAD_COUNT=2")
    print("SECOND_REVERSE_GRAD_COUNT=2")
    print("DIRECT_FORWARD_JVP_COUNT=0")
    print("FINITE_DIFFERENCE_USED=False")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("PRINCIPAL_ANGLES_COMPUTED=False")
    print("PROJECTOR_OVERLAP_COMPUTED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
