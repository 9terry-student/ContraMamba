#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import reason_router_gen4_mamba14b_geometry_prepare_fast_cuda as geom
from scripts import reason_router_gen4_mamba14b_discovery_fast_cuda as discovery

EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_AUDIT_ANCESTOR = "36480ac7a68bc8475f8a2ead4b3ae6ddab333684"

AUDIT_PATH = Path(
    "reports/reason_router_gen4_mamba14b_p5_cross_block_jvp_transport_feasibility_audit_candidate.md"
)
AUDIT_GIT_BLOB = "f461b14d34ca58f8cce3711795dc7bd0cab3cd1a"
GEOM_SCRIPT = Path("scripts/reason_router_gen4_mamba14b_geometry_prepare_fast_cuda.py")
GEOM_SCRIPT_GIT_BLOB = "d221c28657cf9bb157d517b635bae42671563903"
DISCOVERY_SCRIPT = Path("scripts/reason_router_gen4_mamba14b_discovery_fast_cuda.py")
DISCOVERY_SCRIPT_GIT_BLOB = "6ddc06d009b720844eb529cf855d4badc8744602"

CANONICAL_ROOT = Path(
    "reports/reason_router_gen4_mamba14b_geometry_preparation_runs/"
    "g4k-mamba14b-geometry-xg2xg4-2gpu-c758d5e-retry1"
)
CANONICAL_STRONG = CANONICAL_ROOT / "strong_indices.json"
CANONICAL_P5_PLUS = CANONICAL_ROOT / "p5_plus.f64le"
CANONICAL_P5_MINUS = CANONICAL_ROOT / "p5_minus.f64le"
CANONICAL_STRONG_GIT_BLOB = "7d86541ff368c550860dcf666ce418c5a1217ec0"
CANONICAL_P5_PLUS_GIT_BLOB = "007c15f802bc2a07e1770586b995f1fe9d9c0b90"
CANONICAL_P5_MINUS_GIT_BLOB = "0a2a3278c01b520f0916cd48051a9a9cef77041a"

FAMILY = "xg2"
PAIR = "xg2_fact_301"
CELL = "C2_NAME"
ANCHOR_NAME = "A_IDENTITY"
TARGET_OFFSET = 2
SOURCE_BLOCK = 35
TARGET_BLOCK = 36
INTERMEDIATE_SIZE = 4096
PROJECTED_WIDTH = 8192
SOURCE_PLANE = "P5"
BASIS_NAMES = ("plus", "minus")

RESULT_PASS = "PASS_MAMBA14B_P5_CROSS_BLOCK_ONE_ROW_JVP_FEASIBILITY"
REPORT_FILE = "jvp_feasibility_report.json"
PLUS_FILE = "transported_p5_plus.f64le"
MINUS_FILE = "transported_p5_minus.f64le"
SUMS_FILE = "SHA256SUMS.txt"


class TransportJVPError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise TransportJVPError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TransportJVPError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(branch in ("", EXPECTED_BRANCH), f"BRANCH:{branch}")
    require(git("rev-parse", "HEAD") == expected_head, "HEAD")
    require(git("status", "--porcelain") == "", "WORKTREE")
    require(
        git_rc(
            "merge-base",
            "--is-ancestor",
            REQUIRED_AUDIT_ANCESTOR,
            expected_head,
        )
        == 0,
        "AUDIT_NOT_ANCESTOR",
    )

    pinned = {
        AUDIT_PATH: AUDIT_GIT_BLOB,
        GEOM_SCRIPT: GEOM_SCRIPT_GIT_BLOB,
        DISCOVERY_SCRIPT: DISCOVERY_SCRIPT_GIT_BLOB,
        CANONICAL_STRONG: CANONICAL_STRONG_GIT_BLOB,
        CANONICAL_P5_PLUS: CANONICAL_P5_PLUS_GIT_BLOB,
        CANONICAL_P5_MINUS: CANONICAL_P5_MINUS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}") == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(FAMILY == "xg2", "FAMILY")
    require(PAIR == "xg2_fact_301", "PAIR")
    require(CELL == "C2_NAME", "CELL")
    require(ANCHOR_NAME == "A_IDENTITY", "ANCHOR_NAME")
    require(TARGET_OFFSET == 2, "TARGET_OFFSET")
    require(SOURCE_BLOCK == 35, "SOURCE_BLOCK")
    require(TARGET_BLOCK == 36, "TARGET_BLOCK")
    require(INTERMEDIATE_SIZE == geom.INTERMEDIATE_SIZE == 4096, "INTERMEDIATE_SIZE")
    require(PROJECTED_WIDTH == 2 * INTERMEDIATE_SIZE, "PROJECTED_WIDTH")
    require(SOURCE_PLANE == "P5", "SOURCE_PLANE")
    require(BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")


def scatter_strong_vector(
    strong_vector: torch.Tensor,
    strong_indices: Sequence[int],
    *,
    ambient_dim: int = INTERMEDIATE_SIZE,
) -> torch.Tensor:
    value = strong_vector.detach().cpu().to(torch.float64).contiguous()
    indices = [int(index) for index in strong_indices]
    require(value.ndim == 1, "SCATTER_VECTOR_NDIM")
    require(len(indices) == int(value.numel()), "SCATTER_LENGTH")
    require(indices == sorted(indices), "SCATTER_INDEX_ORDER")
    require(len(set(indices)) == len(indices), "SCATTER_INDEX_DUPLICATE")
    require(
        all(0 <= index < ambient_dim for index in indices),
        "SCATTER_INDEX_RANGE",
    )
    require(bool(torch.isfinite(value).all().item()), "SCATTER_NONFINITE")

    out = torch.zeros(ambient_dim, dtype=torch.float64)
    out[torch.tensor(indices, dtype=torch.long)] = value
    require(
        math.isclose(
            float(torch.linalg.vector_norm(out).item()),
            float(torch.linalg.vector_norm(value).item()),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "SCATTER_NORM",
    )
    return out.contiguous()


def numerical_rank_two(vectors: Sequence[torch.Tensor]) -> dict[str, Any]:
    require(len(vectors) == 2, "RANK_VECTOR_COUNT")
    cols = [
        vector.detach().cpu().to(torch.float64).contiguous()
        for vector in vectors
    ]
    require(
        all(tuple(vector.shape) == (INTERMEDIATE_SIZE,) for vector in cols),
        "RANK_VECTOR_SHAPE",
    )
    matrix = torch.stack(cols, dim=1)
    require(bool(torch.isfinite(matrix).all().item()), "RANK_NONFINITE")
    singular = torch.linalg.svdvals(matrix)
    require(tuple(singular.shape) == (2,), "RANK_SINGULAR_SHAPE")
    eps = torch.finfo(torch.float64).eps
    tolerance = (
        max(matrix.shape)
        * eps
        * float(torch.max(singular).item())
    )
    rank = int(torch.sum(singular > tolerance).item())
    return {
        "rank": rank,
        "rank_tolerance": tolerance,
        "singular_values": [float(value) for value in singular.tolist()],
    }


def _row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW:{key}")
        out[key] = index
    return out


def load_gate_input(
    snapshot: Path,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    rows, encoded, events, tokenizer_provenance = geom.load_family_inputs(
        FAMILY,
        snapshot,
    )
    lookup = _row_index(rows)
    require((PAIR, CELL) in lookup, "GATE_ROW_MISSING")
    event_key = (PAIR, CELL, ANCHOR_NAME)
    require(event_key in events, "GATE_EVENT_MISSING")

    row_index = lookup[(PAIR, CELL)]
    input_ids = encoded["input_ids"][row_index].unsqueeze(0).contiguous()
    require(tuple(input_ids.shape) == (1, 128), "GATE_INPUT_SHAPE")
    anchor = int(events[event_key]["absolute_anchor_token_index"])
    target_abs = anchor + TARGET_OFFSET
    require(0 <= target_abs < input_ids.shape[1], "TARGET_RANGE")

    return input_ids, target_abs, {
        "row_index": row_index,
        "anchor": anchor,
        "target_abs": target_abs,
        "tokenizer_provenance": tokenizer_provenance,
    }


def load_canonical_p5_ambient() -> dict[str, torch.Tensor]:
    frozen = discovery.load_frozen_geometry()
    indices = [int(value) for value in frozen["strong_indices"]]
    require(len(indices) == 829, "CANONICAL_STRONG_COUNT")
    require(SOURCE_PLANE in frozen["planes"], "P5_MISSING")

    out: dict[str, torch.Tensor] = {}
    for basis_name in BASIS_NAMES:
        strong = (
            frozen["planes"][SOURCE_PLANE][basis_name]
            .detach().cpu().to(torch.float64).contiguous()
        )
        require(tuple(strong.shape) == (829,), f"P5_SHAPE:{basis_name}")
        ambient = scatter_strong_vector(strong, indices)
        require(
            abs(float(torch.linalg.vector_norm(ambient).item()) - 1.0)
            <= 2e-10,
            f"P5_NORM:{basis_name}",
        )
        out[basis_name] = ambient

    require(
        abs(float(torch.dot(out["plus"], out["minus"]).item()))
        <= 2e-10,
        "P5_ORTHOGONALITY",
    )
    return out


def capture_baseline_boundary(
    *,
    model: Any,
    input_ids: torch.Tensor,
    target_abs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    backbone = model.mamba
    layers = backbone.layers
    require(len(layers) == geom.LAYER_COUNT, "LAYER_COUNT")
    layer35 = layers[SOURCE_BLOCK]
    mixer35 = layer35.mixer

    captured: dict[str, torch.Tensor] = {}
    counts = {"layer35_pre": 0, "in_proj35": 0}

    def layer35_pre(_module, args):
        counts["layer35_pre"] += 1
        require(
            counts["layer35_pre"] == 1 and len(args) >= 1,
            "LAYER35_PRE_HOOK",
        )
        value = args[0]
        require(torch.is_tensor(value), "LAYER35_INPUT_TENSOR")
        captured["residual35"] = value.detach().clone()

    def in_proj35_hook(_module, _args, output):
        counts["in_proj35"] += 1
        require(counts["in_proj35"] == 1, "INPROJ35_HOOK")
        require(
            torch.is_tensor(output)
            and tuple(output.shape) == (1, 128, PROJECTED_WIDTH),
            f"INPROJ35_SHAPE:{tuple(output.shape)}",
        )
        captured["projected35"] = output.detach().clone()

    handles = [
        layer35.register_forward_pre_hook(layer35_pre),
        mixer35.in_proj.register_forward_hook(in_proj35_hook),
    ]
    try:
        model.mamba.eval()
        with torch.inference_mode():
            _ = model.mamba(
                input_ids=input_ids.detach().to(device).contiguous()
            )
        torch.cuda.synchronize(device)
    finally:
        for handle in reversed(handles):
            handle.remove()

    require(
        counts == {"layer35_pre": 1, "in_proj35": 1},
        f"BASELINE_HOOK_COUNTS:{counts}",
    )
    require(set(captured) == {"residual35", "projected35"}, "BASELINE_CAPTURE")
    require(
        tuple(captured["residual35"].shape) == (1, 128, geom.HIDDEN_SIZE),
        "RESIDUAL35_SHAPE",
    )
    require(
        tuple(captured["projected35"].shape) == (1, 128, PROJECTED_WIDTH),
        "PROJECTED35_SHAPE",
    )
    require(0 <= target_abs < 128, "BASELINE_TARGET_RANGE")
    return captured


def replace_target_content(
    baseline_projected: torch.Tensor,
    target_content: torch.Tensor,
    target_abs: int,
) -> torch.Tensor:
    require(
        baseline_projected.ndim == 3
        and baseline_projected.shape[0] == 1
        and baseline_projected.shape[-1] == PROJECTED_WIDTH,
        "REPLACE_BASELINE_SHAPE",
    )
    require(
        tuple(target_content.shape) == (INTERMEDIATE_SIZE,),
        "REPLACE_TARGET_SHAPE",
    )
    require(0 <= target_abs < baseline_projected.shape[1], "REPLACE_TARGET_RANGE")

    gate = baseline_projected[
        0,
        target_abs,
        INTERMEDIATE_SIZE:,
    ]
    target_row = torch.cat(
        [target_content, gate],
        dim=0,
    ).reshape(1, 1, PROJECTED_WIDTH)
    return torch.cat(
        [
            baseline_projected[:, :target_abs, :],
            target_row,
            baseline_projected[:, target_abs + 1 :, :],
        ],
        dim=1,
    )


def local_block35_to_block36_map(
    *,
    projected35: torch.Tensor,
    residual35: torch.Tensor,
    target_abs: int,
    layer35: Any,
    layer36: Any,
    kernels: Mapping[str, Any],
) -> torch.Tensor:
    require(
        tuple(projected35.shape) == (1, 128, PROJECTED_WIDTH),
        "LOCAL_PROJECTED_SHAPE",
    )
    require(
        tuple(residual35.shape) == (1, 128, geom.HIDDEN_SIZE),
        "LOCAL_RESIDUAL_SHAPE",
    )

    mixer35 = layer35.mixer
    projected_states = projected35.transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    conv_weights = mixer35.conv1d.weight.view(
        mixer35.conv1d.weight.size(0),
        mixer35.conv1d.weight.size(2),
    )
    hidden_states = kernels["causal_conv1d_fn"](
        hidden_states,
        conv_weights,
        mixer35.conv1d.bias,
        activation=mixer35.activation,
    )

    ssm_parameters = mixer35.x_proj(hidden_states.transpose(1, 2))
    time_step, b_scan, c_scan = torch.split(
        ssm_parameters,
        [
            mixer35.time_step_rank,
            mixer35.ssm_state_size,
            mixer35.ssm_state_size,
        ],
        dim=-1,
    )
    discrete_time_step = mixer35.dt_proj.weight @ time_step.transpose(1, 2)
    a_matrix = -torch.exp(mixer35.A_log.float())
    time_proj_bias = (
        mixer35.dt_proj.bias.float()
        if mixer35.dt_proj.bias is not None
        else None
    )

    scan_outputs, _ = kernels["selective_scan_fn"](
        hidden_states,
        discrete_time_step,
        a_matrix,
        b_scan.transpose(1, 2),
        c_scan.transpose(1, 2),
        mixer35.D.float(),
        gate,
        time_proj_bias,
        delta_softplus=True,
        return_last_state=True,
    )
    contextualized = mixer35.out_proj(scan_outputs.transpose(1, 2))
    block35_output = residual35 + contextualized

    norm36 = layer36.norm(
        block35_output.to(dtype=layer36.norm.weight.dtype)
    )
    projected36 = layer36.mixer.in_proj(norm36)
    require(
        tuple(projected36.shape) == (1, 128, PROJECTED_WIDTH),
        "PROJECTED36_SHAPE",
    )
    return projected36[
        0,
        target_abs,
        :INTERMEDIATE_SIZE,
    ]


def compute_one_direction_jvp(
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
    require(tuple(tangent.shape) == (INTERMEDIATE_SIZE,), "TANGENT_SHAPE")

    def phi(content: torch.Tensor) -> torch.Tensor:
        projected = replace_target_content(
            baseline_projected35,
            content,
            target_abs,
        )
        return local_block35_to_block36_map(
            projected35=projected,
            residual35=residual35,
            target_abs=target_abs,
            layer35=layer35,
            layer36=layer36,
            kernels=kernels,
        )

    try:
        primal, transported = torch.func.jvp(
            phi,
            (base_content,),
            (tangent,),
        )
    except Exception as exc:
        raise TransportJVPError(
            "FORWARD_JVP_UNSUPPORTED:"
            f"{type(exc).__name__}:{exc}"
        ) from exc

    require(
        tuple(primal.shape) == (INTERMEDIATE_SIZE,),
        "JVP_PRIMAL_SHAPE",
    )
    require(
        tuple(transported.shape) == (INTERMEDIATE_SIZE,),
        "JVP_TANGENT_SHAPE",
    )
    require(bool(torch.isfinite(primal).all().item()), "JVP_PRIMAL_NONFINITE")
    require(
        bool(torch.isfinite(transported).all().item()),
        "JVP_TANGENT_NONFINITE",
    )
    return primal, transported


def raw_f64le(value: torch.Tensor) -> bytes:
    out = value.detach().cpu().to(torch.float64).contiguous().numpy()
    return out.astype("<f8", copy=False).tobytes(order="C")


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
    geom.validate_fast_runtime_for_device(gpu_id)

    input_ids, target_abs, input_meta = load_gate_input(snapshot)
    p5 = load_canonical_p5_ambient()

    model, kernels, model_provenance = geom.reconstruct_model(
        snapshot=snapshot,
        compact_checkpoint=compact_checkpoint,
        gpu_id=gpu_id,
    )
    device = torch.device(f"cuda:{gpu_id}")
    layers = model.mamba.layers
    layer35 = layers[SOURCE_BLOCK]
    layer36 = layers[TARGET_BLOCK]

    baseline = capture_baseline_boundary(
        model=model,
        input_ids=input_ids,
        target_abs=target_abs,
        device=device,
    )
    baseline_projected35 = baseline["projected35"].to(device).contiguous()
    residual35 = baseline["residual35"].to(device).contiguous()
    base_content = baseline_projected35[
        0,
        target_abs,
        :INTERMEDIATE_SIZE,
    ].detach().clone()

    transported: dict[str, torch.Tensor] = {}
    primals: list[torch.Tensor] = []
    for basis_name in BASIS_NAMES:
        primal, tangent = compute_one_direction_jvp(
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
        torch.max(
            torch.abs(
                primals[0].detach().float()
                - primals[1].detach().float()
            )
        ).item()
    )
    require(primal_diff == 0.0, f"JVP_PRIMAL_MISMATCH:{primal_diff}")

    rank_info = numerical_rank_two(
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

    plus_raw = raw_f64le(transported["plus"])
    minus_raw = raw_f64le(transported["minus"])
    report = {
        "schema_version": "gen4-mamba14b-p5-cross-block-one-row-jvp-feasibility-v1",
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "required_audit_ancestor": REQUIRED_AUDIT_ANCESTOR,
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
            "estimator": "torch.func.jvp",
            "fallback_estimator_used": False,
        },
        "transported_vector_norms": norms,
        "transported_span_numerical_rank": int(rank_info["rank"]),
        "transported_span_rank_tolerance": float(rank_info["rank_tolerance"]),
        "transported_singular_values_for_rank_only": rank_info["singular_values"],
        "model_provenance": model_provenance,
        "accounting": {
            "baseline_full_model_forward_count": 1,
            "local_jvp_count": 2,
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
    report_bytes = pretty_json_bytes(report)
    (output_dir / REPORT_FILE).write_bytes(report_bytes)

    hashes = {
        PLUS_FILE: sha256_bytes(plus_raw),
        MINUS_FILE: sha256_bytes(minus_raw),
        REPORT_FILE: sha256_bytes(report_bytes),
    }
    (output_dir / SUMS_FILE).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
        newline="\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded one-row response-free Mamba-1.4B canonical-P5 "
            "block35-to-block36 forward-JVP feasibility gate. "
            "No population transport metric or statistical test is computed."
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
    print(
        "TRANSPORTED_P5_PLUS_L2="
        + repr(report["transported_vector_norms"]["plus"])
    )
    print(
        "TRANSPORTED_P5_MINUS_L2="
        + repr(report["transported_vector_norms"]["minus"])
    )
    print(
        "TRANSPORTED_SPAN_NUMERICAL_RANK="
        + str(report["transported_span_numerical_rank"])
    )
    print("LOCAL_JVP_COUNT=2")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print("PRINCIPAL_ANGLES_COMPUTED=False")
    print("PROJECTOR_OVERLAP_COMPUTED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
