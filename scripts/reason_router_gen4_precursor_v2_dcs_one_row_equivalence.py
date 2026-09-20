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
from scripts import (
    reason_router_gen4_precursor_v2_dcs_stage_a_fast_cuda
    as dcs,
)


EXPECTED_BRANCH = "gen4-mamba370m-core-replication"
REQUIRED_ANCESTOR = "083edb64b75b5a5967068d0b5fb7b97668395018"

DCS_RUNNER = Path(
    "scripts/reason_router_gen4_precursor_v2_dcs_stage_a_fast_cuda.py"
)
DCS_RUNNER_GIT_BLOB = "835d8faba49b70ded8540924233015d06b6c2b68"

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

DESIGN_ARTIFACT = Path(
    "reports/reason_router_gen4_precursor_v2_dynamic_causal_susceptibility_stage_a_design.md"
)
DESIGN_GIT_BLOB = "d24011d698de1a53436a6bb997b2bb400881a718"

FORWARD_CORRECTION = Path(
    "reports/reason_router_gen4_precursor_v2_dcs_forward_accounting_correction.md"
)
FORWARD_CORRECTION_GIT_BLOB = "a37393140dc05a19889805206b5ec3070575dbe1"

HISTORICAL_RAW_ROOT = Path(
    "reports/reason_router_gen4_pre_emission_forced_decisive_stage_a_generation_runs/"
    "g4k-preemission-forceddecisive-stagea-raw-370m-0d3a094-gpu1"
)
HISTORICAL_RAW_SUMMARY = HISTORICAL_RAW_ROOT / "execution_summary.json"
HISTORICAL_RAW_ROWS = (
    HISTORICAL_RAW_ROOT / "forced_decisive_stage_a_generation_rows.jsonl"
)
HISTORICAL_RAW_SUMS = HISTORICAL_RAW_ROOT / "SHA256SUMS.txt"
HISTORICAL_RAW_SUMMARY_GIT_BLOB = "725e5454a824adc18814d3a20931bb2ab65209f7"
HISTORICAL_RAW_ROWS_GIT_BLOB = "1691f8e9582f786760df19bee778f247d53b238b"
HISTORICAL_RAW_SUMS_GIT_BLOB = "a7c41fb72d08bc8f5083f45d96bf39aca693bd64"

EQUIVALENCE_ATOL = 1.0e-4
EQUIVALENCE_RTOL = 1.0e-4

RELATIVE_OFFSET = -4
GENERATED_PREFIX_LENGTH = 5
PROBES_PER_BACKEND = 8
CPU_MODEL_FORWARD_COUNT = PROBES_PER_BACKEND
GPU_MODEL_FORWARD_COUNT = PROBES_PER_BACKEND
TOTAL_EQUIVALENCE_FORWARD_COUNT = (
    CPU_MODEL_FORWARD_COUNT + GPU_MODEL_FORWARD_COUNT
)

REPORT_FILE = "precursor_v2_dcs_one_row_equivalence_report.json"
SUMS_FILE = "SHA256SUMS.txt"
REPORT_SCHEMA = "gen4-precursor-v2-dcs-one-row-equivalence-v1"
RESULT_PASS = "PASS_PRECURSOR_V2_DCS_ONE_ROW_CPU_CUDA_EQUIVALENCE"


class PrecursorV2DCSEquivalenceError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrecursorV2DCSEquivalenceError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrecursorV2DCSEquivalenceError(
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
        )
        == 0,
        "RUNNER_IMPLEMENTATION_NOT_ANCESTOR",
    )

    pinned = {
        DCS_RUNNER: DCS_RUNNER_GIT_BLOB,
        BASE_GENERATOR: BASE_GENERATOR_GIT_BLOB,
        FORCED_RUNNER: FORCED_RUNNER_GIT_BLOB,
        IDENTITY_GATE: IDENTITY_GATE_GIT_BLOB,
        DESIGN_ARTIFACT: DESIGN_GIT_BLOB,
        FORWARD_CORRECTION: FORWARD_CORRECTION_GIT_BLOB,
        HISTORICAL_RAW_SUMMARY: HISTORICAL_RAW_SUMMARY_GIT_BLOB,
        HISTORICAL_RAW_ROWS: HISTORICAL_RAW_ROWS_GIT_BLOB,
        HISTORICAL_RAW_SUMS: HISTORICAL_RAW_SUMS_GIT_BLOB,
    }
    for path, expected_blob in pinned.items():
        require(
            git("rev-parse", f"HEAD:{path.as_posix()}")
            == expected_blob,
            f"FROZEN_BLOB:{path}",
        )


def validate_protocol() -> None:
    require(EQUIVALENCE_ATOL == 1.0e-4, "ATOL")
    require(EQUIVALENCE_RTOL == 1.0e-4, "RTOL")
    require(RELATIVE_OFFSET == -4, "RELATIVE_OFFSET")
    require(GENERATED_PREFIX_LENGTH == 5, "GENERATED_PREFIX_LENGTH")
    require(
        dcs.PREFIX_LENGTH_TO_OFFSET[GENERATED_PREFIX_LENGTH]
        == RELATIVE_OFFSET,
        "PREFIX_OFFSET_MAPPING",
    )
    require(dcs.EPSILON == 0.025, "EPSILON")
    require(dcs.PLANES == ("P3", "P5"), "PLANES")
    require(dcs.BASIS_NAMES == ("plus", "minus"), "BASIS_NAMES")
    require(dcs.EPSILON_SIGNS == (-1, 1), "EPSILON_SIGNS")
    require(dcs.PROBES_PER_OFFSET == 8, "PROBES_PER_OFFSET")
    require(PROBES_PER_BACKEND == dcs.PROBES_PER_OFFSET, "PROBE_COUNT")
    require(CPU_MODEL_FORWARD_COUNT == 8, "CPU_FORWARD_COUNT")
    require(GPU_MODEL_FORWARD_COUNT == 8, "GPU_FORWARD_COUNT")
    require(TOTAL_EQUIVALENCE_FORWARD_COUNT == 16, "TOTAL_FORWARD_COUNT")
    require(forced.FORCED_CLASS_ORDER == ("REFUTE", "SUPPORT"), "CLASS_ORDER")
    require(forced.COMMON_PREFIX_LENGTH == 8, "COMMON_PREFIX_LENGTH")
    require(
        GENERATED_PREFIX_LENGTH < forced.COMMON_PREFIX_LENGTH,
        "PREFIX_NOT_PREDECISION",
    )


def select_historical_decisive_row(
    cohort: Sequence[Mapping[str, Any]],
) -> tuple[int, Mapping[str, Any]]:
    require(len(cohort) == base.N, "HISTORICAL_COHORT_N")
    for index, row in enumerate(cohort):
        if str(row["correct_label"]) in forced.FORCED_CLASS_ORDER:
            return index, row
    raise PrecursorV2DCSEquivalenceError(
        "NO_HISTORICAL_DECISIVE_ROW"
    )


def historical_equivalence_input(
    *,
    snapshot: Path,
) -> tuple[
    torch.Tensor,
    Mapping[str, Any],
    Mapping[str, Sequence[int]],
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
    require(
        refute_prefix == support_prefix,
        "HISTORICAL_PREFIX_NOT_CLASS_BLIND",
    )

    combined = torch.cat(
        [
            prompt.detach().cpu().to(torch.long),
            torch.tensor(refute_prefix, dtype=torch.long),
        ]
    ).unsqueeze(0).contiguous()
    require(
        combined.ndim == 2
        and combined.shape[0] == 1,
        "EQUIVALENCE_INPUT_SHAPE",
    )

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
        "new_precursor_v2_cohort_accessed": False,
    }
    return combined, row, token_ids, branch_tokens, provenance


def _convert_probe_for_equivalence(
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(probe)
    count = int(out.pop("scientific_full_model_forward_count"))
    require(count == 1, "PROBE_FORWARD_COUNT")
    out["equivalence_model_forward_count"] = 1
    return out


def run_probe_set(
    *,
    model: Any,
    runtime: Mapping[str, Any],
    frozen: Mapping[str, Any],
    input_ids: torch.Tensor,
    gold_label: str,
    branch_tokens: Mapping[str, int],
    cast_tol: float,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for plane in dcs.PLANES:
        for basis_name in dcs.BASIS_NAMES:
            for epsilon_sign in dcs.EPSILON_SIGNS:
                probe = dcs.probe_forward(
                    model=model,
                    runtime=runtime,
                    frozen=frozen,
                    input_ids=input_ids,
                    gold_label=gold_label,
                    branch_tokens=branch_tokens,
                    plane=plane,
                    basis_name=basis_name,
                    epsilon_sign=epsilon_sign,
                    cast_tol=cast_tol,
                )
                probes.append(_convert_probe_for_equivalence(probe))
    require(len(probes) == PROBES_PER_BACKEND, "PROBE_SET_COUNT")
    require(
        sum(int(p["equivalence_model_forward_count"]) for p in probes)
        == PROBES_PER_BACKEND,
        "PROBE_SET_FORWARD_COUNT",
    )
    return probes


def _probe_key(probe: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(probe["plane"]),
        str(probe["basis"]),
        int(probe["epsilon_sign"]),
    )


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


def compare_probe_sets(
    cpu_probes: Sequence[Mapping[str, Any]],
    gpu_probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        len(cpu_probes) == len(gpu_probes) == PROBES_PER_BACKEND,
        "COMPARE_PROBE_COUNT",
    )
    cpu_by_key = {_probe_key(p): p for p in cpu_probes}
    gpu_by_key = {_probe_key(p): p for p in gpu_probes}
    require(len(cpu_by_key) == PROBES_PER_BACKEND, "CPU_PROBE_KEYS")
    require(len(gpu_by_key) == PROBES_PER_BACKEND, "GPU_PROBE_KEYS")
    require(set(cpu_by_key) == set(gpu_by_key), "PROBE_KEY_SET")

    max_branch_logit_diff = 0.0
    max_margin_diff = 0.0
    max_applied_l2_diff = 0.0

    for key in sorted(cpu_by_key):
        cpu = cpu_by_key[key]
        gpu = gpu_by_key[key]
        require(
            float(cpu["epsilon"]) == float(gpu["epsilon"]) == dcs.EPSILON,
            f"EPSILON:{key}",
        )
        for label in forced.FORCED_CLASS_ORDER:
            max_branch_logit_diff = max(
                max_branch_logit_diff,
                _compare_scalar(
                    cpu["branch_start_logits"][label],
                    gpu["branch_start_logits"][label],
                    label=f"{key}:branch:{label}",
                ),
            )
        max_margin_diff = max(
            max_margin_diff,
            _compare_scalar(
                cpu["gold_aligned_decisive_margin"],
                gpu["gold_aligned_decisive_margin"],
                label=f"{key}:gold_aligned_margin",
            ),
        )
        max_applied_l2_diff = max(
            max_applied_l2_diff,
            _compare_scalar(
                cpu["intervention_audit"]["applied_delta_l2"],
                gpu["intervention_audit"]["applied_delta_l2"],
                label=f"{key}:applied_delta_l2",
            ),
        )

    cpu_derived = dcs.summarize_offset_from_probe_values(
        probes=cpu_probes
    )
    gpu_derived = dcs.summarize_offset_from_probe_values(
        probes=gpu_probes
    )

    derived_diffs = {
        "chi_p3_abs_diff": _compare_scalar(
            cpu_derived["chi_p3"],
            gpu_derived["chi_p3"],
            label="derived:chi_p3",
        ),
        "chi_p5_abs_diff": _compare_scalar(
            cpu_derived["chi_p5"],
            gpu_derived["chi_p5"],
            label="derived:chi_p5",
        ),
        "d_t_abs_diff": _compare_scalar(
            cpu_derived["d_t"],
            gpu_derived["d_t"],
            label="derived:d_t",
        ),
    }
    for plane in dcs.PLANES:
        for basis_name in dcs.BASIS_NAMES:
            derived_diffs[
                f"{plane.lower()}_{basis_name}_derivative_abs_diff"
            ] = _compare_scalar(
                cpu_derived["derivatives"][plane][basis_name],
                gpu_derived["derivatives"][plane][basis_name],
                label=f"derived:{plane}:{basis_name}",
            )

    return {
        "max_branch_token_logit_abs_diff": max_branch_logit_diff,
        "max_gold_aligned_margin_abs_diff": max_margin_diff,
        "max_applied_delta_l2_abs_diff": max_applied_l2_diff,
        **derived_diffs,
        "cpu_derived": cpu_derived,
        "gpu_derived": gpu_derived,
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
        _token_ids,
        branch_tokens,
        historical_provenance,
    ) = historical_equivalence_input(snapshot=snapshot)
    gold_label = str(historical_row["correct_label"])
    require(gold_label in forced.FORCED_CLASS_ORDER, "HISTORICAL_GOLD")

    spec = dcs.steering.bridge.scale_spec("mamba370m")
    cast_tol = float(
        spec["confirmation"].transport_runtime.RUNTIME_CAST_TOL
    )
    require(
        math.isfinite(cast_tol) and cast_tol > 0.0,
        "RUNTIME_CAST_TOL",
    )

    cpu_model, cpu_identity = identity_gate.load_causal_lm_identity(
        snapshot
    )
    frozen_cpu = base.discovery.load_frozen_geometry()
    cpu_runtime = base.runtime_components_for_causal_lm(
        cpu_model,
        frozen_cpu,
    )
    require(
        all(parameter.device.type == "cpu" for parameter in cpu_model.parameters()),
        "CPU_MODEL_DEVICE",
    )
    cpu_probes = run_probe_set(
        model=cpu_model,
        runtime=cpu_runtime,
        frozen=frozen_cpu,
        input_ids=input_ids_cpu,
        gold_label=gold_label,
        branch_tokens=branch_tokens,
        cast_tol=cast_tol,
    )
    cpu_strong_mask = (
        cpu_runtime["strong_mask"]
        .detach().cpu().bool().contiguous().clone()
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

    gpu_probes = run_probe_set(
        model=gpu_model,
        runtime=gpu_runtime,
        frozen=frozen_gpu,
        input_ids=input_ids_gpu,
        gold_label=gold_label,
        branch_tokens=branch_tokens,
        cast_tol=cast_tol,
    )
    torch.cuda.synchronize(device)

    _identity_exact(cpu_identity, gpu_identity)
    gpu_strong_mask = (
        gpu_runtime["strong_mask"]
        .detach().cpu().bool().contiguous()
    )
    require(
        torch.equal(cpu_strong_mask, gpu_strong_mask),
        "STRONG_MASK_MISMATCH",
    )
    require(
        int(cpu_strong_mask.sum().item()) == dcs.DIM,
        "STRONG_MASK_DIM",
    )

    comparison = compare_probe_sets(cpu_probes, gpu_probes)

    kernel_compat = base.discovery.kernel_compat
    report = {
        "schema_version": REPORT_SCHEMA,
        "result": RESULT_PASS,
        "execution_head": expected_head,
        "runner_implementation_ancestor": REQUIRED_ANCESTOR,
        "design": {
            "path": DESIGN_ARTIFACT.as_posix(),
            "git_blob": DESIGN_GIT_BLOB,
        },
        "forward_accounting_correction": {
            "path": FORWARD_CORRECTION.as_posix(),
            "git_blob": FORWARD_CORRECTION_GIT_BLOB,
        },
        "dcs_runner": {
            "path": DCS_RUNNER.as_posix(),
            "git_blob": DCS_RUNNER_GIT_BLOB,
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
            "relative_offset": RELATIVE_OFFSET,
            "generated_prefix_length": GENERATED_PREFIX_LENGTH,
            "epsilon": dcs.EPSILON,
            "planes": list(dcs.PLANES),
            "basis_names": list(dcs.BASIS_NAMES),
            "epsilon_signs": list(dcs.EPSILON_SIGNS),
            "probes_per_backend": PROBES_PER_BACKEND,
            "gold_aligned_readout": True,
            "readout_block": dcs.LATE_BLOCK,
            "intervention_block": dcs.EARLY_BLOCK,
        },
        "tolerances": {
            "atol": EQUIVALENCE_ATOL,
            "rtol": EQUIVALENCE_RTOL,
            "runtime_cast_tolerance": cast_tol,
        },
        "comparison": comparison,
        "cpu": {
            "execution_mode": "cpu_slow_transformers_mamba_fallback",
            "use_mamba_kernels": False,
            "model_identity": cpu_identity,
            "probes": cpu_probes,
        },
        "cuda": {
            "execution_mode": "cuda_fast_exact_project_kernels",
            "logical_device": str(device),
            "physical_device_requested": int(physical_device),
            "model_identity": gpu_identity,
            "probes": gpu_probes,
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
        "forward_accounting": {
            "cpu_model_forward_count": CPU_MODEL_FORWARD_COUNT,
            "gpu_model_forward_count": GPU_MODEL_FORWARD_COUNT,
            "total_equivalence_model_forward_count":
                TOTAL_EQUIVALENCE_FORWARD_COUNT,
            "scientific_model_forward_count": 0,
        },
        "boundary": {
            "new_precursor_v2_cohort_accessed": False,
            "historical_generation_response_used_for_selection": False,
            "scientific_inference_executed": False,
            "statistical_testing_performed": False,
            "p_value_count_added": 0,
            "training_executed": False,
            "backward_executed": False,
            "scientific_conclusion": None,
            "stage_a_scientific_execution_authorized_by_this_artifact":
                False,
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
        )
        + "\n",
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
            "Bounded one-row CPU-slow versus CUDA-fast equivalence gate "
            "for the frozen Precursor-v2 dynamic causal susceptibility "
            "measurement. Uses the first decisive-gold row from the old "
            "historical precursor population and only offset t*-4. "
            "Does not access the new N=800 Precursor-v2 cohort."
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
    print(
        "HISTORICAL_EXAMPLE_ID="
        + str(report["historical_input"]["example_id"])
    )
    print("RELATIVE_OFFSET=-4")
    print("CPU_MODEL_FORWARD_COUNT=8")
    print("GPU_MODEL_FORWARD_COUNT=8")
    print("TOTAL_EQUIVALENCE_MODEL_FORWARD_COUNT=16")
    print("SCIENTIFIC_MODEL_FORWARD_COUNT=0")
    print(
        "MAX_BRANCH_TOKEN_LOGIT_ABS_DIFF="
        + repr(comparison["max_branch_token_logit_abs_diff"])
    )
    print(
        "MAX_GOLD_ALIGNED_MARGIN_ABS_DIFF="
        + repr(comparison["max_gold_aligned_margin_abs_diff"])
    )
    print("CHI_P3_ABS_DIFF=" + repr(comparison["chi_p3_abs_diff"]))
    print("CHI_P5_ABS_DIFF=" + repr(comparison["chi_p5_abs_diff"]))
    print("D_T_ABS_DIFF=" + repr(comparison["d_t_abs_diff"]))
    print("NEW_PRECURSOR_V2_COHORT_ACCESSED=False")
    print("STATISTICAL_TESTING_PERFORMED=False")
    print("P_VALUE_COUNT_ADDED=0")
    print("SCIENTIFIC_CONCLUSION=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
