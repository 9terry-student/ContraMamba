from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence as eq


ROOT = Path(__file__).resolve().parents[1]

R5_FREEZE_COMMIT = "71f4e5106e36d36003e49e7424ab98d3df0dbe90"
R5_EXECUTION_HEAD = "e646fc900d12b42621a6a58ee31eff255bb52bfd"

SOURCE_PAIR_COUNT = 300
BASELINE_FORWARDS_PER_PAIR = 4
FULL_BASELINE_FORWARD_BUDGET = SOURCE_PAIR_COUNT * BASELINE_FORWARDS_PER_PAIR
ALIGNMENT_SHIFT_THRESHOLD = 0.11228626366380845
MIN_GROUP_SIZE = 30
REGIME_LARGE = "LARGE"
REGIME_SMALL = "SMALL"

ITEM_SCHEMA = "gen4-generator-family-prevalence-full-baseline-item-v1"
SUMMARY_SCHEMA = "gen4-generator-family-prevalence-full-baseline-summary-v1"
MANIFEST_SCHEMA = "gen4-generator-family-prevalence-full-baseline-manifest-v1"
RESULT_PASS = "PASS_GENERATOR_FAMILY_PREVALENCE_FULL_BASELINE"
GLOBAL_PRIMARY_CLASSIFICATION = "PREVALENCE_TRANSPORTABILITY_INCOMPLETE"

ITEM_FILE = "baseline_items.jsonl"
SUMMARY_FILE = "prevalence_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

R5_GATE_ARTIFACTS: dict[str, dict[str, str]] = {
    "xg2": {
        "path": (
            "reports/"
            "reason_router_gen4_generator_family_prevalence_baseline_fast_cuda_"
            "one_pair_equivalence_e646fc9_r5/xg2/equivalence_report.json"
        ),
        "sha256": "1395b2ca9c20d501e21058250a972a7af5648b57d900b79d2cc04f960f5032fa",
    },
    "xg4": {
        "path": (
            "reports/"
            "reason_router_gen4_generator_family_prevalence_baseline_fast_cuda_"
            "one_pair_equivalence_e646fc9_r5/xg4/equivalence_report.json"
        ),
        "sha256": "d3d18a90c44207654a39a01e6ed38bebc50b096e1b12c9bb55d70c0b4a10bf4d",
    },
}

FROZEN_PREVALENCE_DEPENDENCY_PATHS = (
    "scripts/reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py",
    "scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py",
    "scripts/reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility.py",
    "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py",
    R5_GATE_ARTIFACTS["xg2"]["path"],
    R5_GATE_ARTIFACTS["xg4"]["path"],
)


class PrevalenceFullBaselineError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PrevalenceFullBaselineError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def authenticate_repo(expected_head: str) -> None:
    # Reuse the already-frozen generator-family/backend authentication first.
    eq.authenticate_repo(expected_head)
    require(
        _git_is_ancestor(R5_FREEZE_COMMIT, expected_head),
        "R5_FREEZE_NOT_ANCESTOR",
    )
    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            R5_FREEZE_COMMIT,
            expected_head,
            "--",
            *FROZEN_PREVALENCE_DEPENDENCY_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_PREVALENCE_DEPENDENCY_DRIFT")


def validate_r5_gate_artifact(family: str) -> dict[str, Any]:
    require(family in R5_GATE_ARTIFACTS, f"UNSUPPORTED_FAMILY:{family}")
    spec = R5_GATE_ARTIFACTS[family]
    path = ROOT / spec["path"]
    require(path.is_file(), f"R5_GATE_ARTIFACT_MISSING:{family}")
    require(
        sha256_file(path) == spec["sha256"],
        f"R5_GATE_ARTIFACT_SHA256:{family}",
    )
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(report.get("result") == eq.RESULT_PASS, f"R5_GATE_RESULT:{family}")
    require(report.get("family_key") == family, f"R5_GATE_FAMILY:{family}")
    require(
        report.get("source_pair_id") == eq.FAMILY_CONFIG[family]["pair_id"],
        f"R5_GATE_PAIR:{family}",
    )
    require(
        report.get("execution_head") == R5_EXECUTION_HEAD,
        f"R5_GATE_EXECUTION_HEAD:{family}",
    )
    require(
        report.get("cpu_model_forward_count") == eq.FORWARDS_PER_BACKEND,
        f"R5_GATE_CPU_FORWARD_COUNT:{family}",
    )
    require(
        report.get("gpu_model_forward_count") == eq.FORWARDS_PER_BACKEND,
        f"R5_GATE_GPU_FORWARD_COUNT:{family}",
    )
    require(
        report.get("total_model_forward_count") == eq.TOTAL_MODEL_FORWARDS,
        f"R5_GATE_TOTAL_FORWARD_COUNT:{family}",
    )
    require(
        report.get("scientific_budget_forward_count") == 0,
        f"R5_GATE_SCIENTIFIC_BUDGET:{family}",
    )
    require(
        report.get("scientific_conclusion") is None,
        f"R5_GATE_SCIENTIFIC_CONCLUSION:{family}",
    )
    require(
        report.get("representative_checkpoint_sha256")
        == eq.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        f"R5_GATE_CHECKPOINT:{family}",
    )
    return report


def classify_regime(alignment_shift_abs: float) -> str:
    value = float(alignment_shift_abs)
    require(math.isfinite(value), "NONFINITE_ALIGNMENT_SHIFT")
    return REGIME_LARGE if value >= ALIGNMENT_SHIFT_THRESHOLD else REGIME_SMALL


def summarize_prevalence(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(items) == SOURCE_PAIR_COUNT, "ITEM_COUNT")
    pair_ids = [str(item["source_pair_id"]) for item in items]
    require(len(set(pair_ids)) == SOURCE_PAIR_COUNT, "PAIR_ID_UNIQUENESS")

    counts = Counter(str(item["regime"]) for item in items)
    require(set(counts) <= {REGIME_LARGE, REGIME_SMALL}, "REGIME_VALUE")
    n_large = int(counts[REGIME_LARGE])
    n_small = int(counts[REGIME_SMALL])
    require(n_large + n_small == SOURCE_PAIR_COUNT, "REGIME_COUNT_TOTAL")

    viable = n_large >= MIN_GROUP_SIZE and n_small >= MIN_GROUP_SIZE
    return {
        "n_LARGE": n_large,
        "n_SMALL": n_small,
        "p_LARGE": n_large / SOURCE_PAIR_COUNT,
        "p_SMALL": n_small / SOURCE_PAIR_COUNT,
        "minimum_group_size": MIN_GROUP_SIZE,
        "VIABLE": viable,
        "family_level_classification": "VIABLE" if viable else "NOT_VIABLE",
    }


def _write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    payloads = {
        ITEM_FILE: jsonl_bytes(items),
        SUMMARY_FILE: canonical_json_bytes(summary),
    }
    hashes: dict[str, str] = {}
    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = sha256_bytes(raw)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int((output_dir / name).stat().st_size),
            }
            for name, digest in sorted(hashes.items())
        },
    }
    manifest_raw = canonical_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(manifest_raw)
    hashes[MANIFEST_FILE] = sha256_bytes(manifest_raw)

    checksum_raw = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
    ).encode("utf-8")
    (output_dir / CHECKSUM_FILE).write_bytes(checksum_raw)
    return hashes


def run_full_baseline(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(family in eq.FAMILY_CONFIG, f"UNSUPPORTED_FAMILY:{family}")
    authenticate_repo(expected_head)
    gate_report = validate_r5_gate_artifact(family)
    eq.backend.runtime_gate()
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    with eq.backend.parent_runtime_rebind():
        rows, encoded, event_rows = eq.load_family_inputs(
            family,
            tokenizer_snapshot,
        )
        pairs = eq._pair_order(family, rows)
        require(len(pairs) == SOURCE_PAIR_COUNT, "PAIR_COUNT")

        events = eq.parent.event_lookup(event_rows)
        eq.parent.validate_transport_event_plan(pairs, events)
        row_index = eq.parent.build_row_index(rows)
        trace_code, trace_line = (
            eq.measurement._resolve_and_validate_runtime_binding()
        )

        kernels = eq.kernel_compat.load_exact_fast_kernels()
        with eq.kernel_compat.exact_transformers_kernel_loader(
            kernels
        ) as constructor_kernel_calls:
            model, checkpoint_sha = eq.parent.load_representative_model_external(
                model_snapshot=model_snapshot,
                checkpoint_path=checkpoint_path,
            )
            require(
                checkpoint_sha == eq.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
                "CHECKPOINT_IDENTITY",
            )
            runtime_ctx = eq.transport_runtime.validate_runtime_components(model)

        constructor_counts = Counter(constructor_kernel_calls)
        require(
            set(constructor_counts) == {"causal-conv1d", "mamba-ssm"},
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_NAMES:{dict(constructor_counts)}",
        )
        require(
            constructor_counts["causal-conv1d"] > 0,
            "TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT_ZERO",
        )
        require(
            constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"],
            f"TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT_MISMATCH:{dict(constructor_counts)}",
        )
        eq.kernel_compat.validate_transformers_kernel_bindings(kernels)

        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(
                parameter.device.type == "cuda"
                for parameter in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = eq.backend._make_fast_capture(kernels)
        original_capture = eq.parent.capture_branch
        budget = eq.parent.ForwardBudget(FULL_BASELINE_FORWARD_BUDGET)
        items: list[dict[str, Any]] = []

        eq.parent.capture_branch = fast_capture
        try:
            for pair in pairs:
                raw = eq.run_baseline_geometry_pair(
                    family,
                    pair,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )
                shift = float(raw["alignment_shift_abs"])
                item = dict(raw)
                item["schema_version"] = ITEM_SCHEMA
                item["threshold"] = ALIGNMENT_SHIFT_THRESHOLD
                item["regime"] = classify_regime(shift)
                item["classification_from_baseline_only"] = True
                items.append(item)
            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            eq.parent.capture_branch = original_capture

    prevalence = summarize_prevalence(items)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "r5_freeze_commit": R5_FREEZE_COMMIT,
        "r5_gate_execution_head": R5_EXECUTION_HEAD,
        "r5_gate_artifact_sha256": R5_GATE_ARTIFACTS[family]["sha256"],
        "representative_checkpoint_sha256": checkpoint_sha,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "threshold_reestimated": False,
        **prevalence,
        "baseline_model_forward_count": FULL_BASELINE_FORWARD_BUDGET,
        "total_model_forward_count": FULL_BASELINE_FORWARD_BUDGET,
        "scientific_budget_forward_count": FULL_BASELINE_FORWARD_BUDGET,
        "baseline_only": True,
        "alignment_intervention_executed": False,
        "magnitude_intervention_executed": False,
        "response_endpoint_computed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "xg3_model_forward_count": 0,
        "global_primary_classification": GLOBAL_PRIMARY_CLASSIFICATION,
        "scientific_conclusion_scope": "FAMILY_LEVEL_PREVALENCE_ONLY",
        "r5_gate_result": gate_report["result"],
    }
    _write_outputs(output_dir, items=items, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full fast-CUDA baseline prevalence execution for one frozen eligible "
            "generator family (XG2 or XG4). Executes exactly 300 pairs x 4 "
            "baseline cells = 1200 scientific model forwards. No intervention, "
            "response endpoint, task head, training, or backward path exists."
        )
    )
    parser.add_argument("--family", choices=tuple(sorted(eq.FAMILY_CONFIG)), required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_full_baseline(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT =", summary["result"])
    print("FAMILY =", summary["family_key"])
    print("N_LARGE =", summary["n_LARGE"])
    print("N_SMALL =", summary["n_SMALL"])
    print("P_LARGE =", summary["p_LARGE"])
    print("VIABLE =", summary["VIABLE"])
    print("TOTAL_MODEL_FORWARD_COUNT =", summary["total_model_forward_count"])
    print("GLOBAL_PRIMARY_CLASSIFICATION =", summary["global_primary_classification"])


if __name__ == "__main__":
    main()
