from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import reason_router_gen4_large_correction_prospective_fast_cuda_full as base
from scripts import reason_router_gen4_xg1_fast_cuda_one_pair_equivalence as xg1_eq


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"

DESIGN_FREEZE_COMMIT = "a31d2bc5ab4b939f52e969c89f3783feb9c3b233"
STRUCTURAL_FREEZE_COMMIT = "d9029801fd47636c155b1c846c433fc561424c8f"
ELIGIBILITY_FREEZE_COMMIT = "ee7c2c10a0cbb4930b78eb0047ec0603b63e0d41"
GATE_FREEZE_COMMIT = "f41aa2abec3fa372699913c387082131b0447051"
GATE_EXECUTION_HEAD = "6afb6191d1ef8c0aa39d4c5497a9250fda192110"

DESIGN_REL = Path(
    "reports/reason_router_gen4_xg1_cross_generator_prospective_replication_design.md"
)
GATE_ARTIFACT_REL = Path(
    "reports/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence_6afb619_r1/"
    "equivalence_report.json"
)
GATE_ARTIFACT_SHA256 = (
    "865124fd1804c2198d58ae01f3846319770fb3eda6482363b4036ba35f721877"
)

SOURCE_PAIR_COUNT = base.SOURCE_PAIR_COUNT
BASELINE_FORWARDS_PER_PAIR = base.BASELINE_FORWARDS_PER_PAIR
ALIGNMENT_FORWARDS_PER_PAIR = base.ALIGNMENT_FORWARDS_PER_PAIR
FORWARDS_PER_PAIR = base.FORWARDS_PER_PAIR
BASELINE_FORWARD_BUDGET = base.BASELINE_FORWARD_BUDGET
ALIGNMENT_FORWARD_BUDGET = base.ALIGNMENT_FORWARD_BUDGET
FULL_FORWARD_BUDGET = base.FULL_FORWARD_BUDGET

ALIGNMENT_SHIFT_THRESHOLD = base.ALIGNMENT_SHIFT_THRESHOLD
MIN_GROUP_SIZE = base.MIN_GROUP_SIZE
FAMILY_ALPHA = base.FAMILY_ALPHA

REGIME_LARGE = base.REGIME_LARGE
REGIME_SMALL = base.REGIME_SMALL

REGIME_ITEM_SCHEMA = "gen4-k-xg1-cross-generator-regime-item-v1"
REGIME_FREEZE_SCHEMA = "gen4-k-xg1-cross-generator-regime-freeze-v1"
ITEM_SCHEMA = "gen4-k-xg1-cross-generator-full-item-v1"
SUMMARY_SCHEMA = "gen4-k-xg1-cross-generator-full-summary-v1"
MANIFEST_SCHEMA = "gen4-k-xg1-cross-generator-full-manifest-v1"

REGIME_FILE = base.REGIME_FILE
REGIME_FREEZE_FILE = base.REGIME_FREEZE_FILE
ITEM_FILE = base.ITEM_FILE
SUMMARY_FILE = base.SUMMARY_FILE
MANIFEST_FILE = base.MANIFEST_FILE
CHECKSUM_FILE = base.CHECKSUM_FILE

_RESPONSE_FIELDS = base._RESPONSE_FIELDS

FROZEN_FULL_DEPENDENCY_PATHS = tuple(
    dict.fromkeys(
        (
            *xg1_eq.BACKEND_FROZEN_PATHS,
            "scripts/reason_router_gen4_large_correction_prospective_fast_cuda_full.py",
            "scripts/reason_router_gen4_xg1_fast_cuda_one_pair_equivalence.py",
            "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py",
            "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py",
            DESIGN_REL.as_posix(),
            xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST.as_posix(),
            xg1_eq.ELIGIBILITY_SUMMARY.as_posix(),
            xg1_eq.eligibility.SOURCE_FACTS_PATH.as_posix(),
            xg1_eq.eligibility.ROWS_PATH.as_posix(),
            xg1_eq.eligibility.STRUCTURAL_MANIFEST_PATH.as_posix(),
            GATE_ARTIFACT_REL.as_posix(),
        )
    )
)


class XG1FullError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG1FullError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XG1FullError("GIT_FAILURE:" + " ".join(args)) from exc


def validate_gate_artifact() -> dict[str, Any]:
    path = ROOT / GATE_ARTIFACT_REL
    require(path.is_file(), "GATE_ARTIFACT_MISSING")
    require(
        base.sha256_file(path) == GATE_ARTIFACT_SHA256,
        "GATE_ARTIFACT_SHA256",
    )
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(
        report.get("result") == "PASS_XG1_FAST_CUDA_ONE_PAIR_EQUIVALENCE",
        "GATE_RESULT_NOT_PASS",
    )
    require(
        report.get("execution_head") == GATE_EXECUTION_HEAD,
        "GATE_EXECUTION_HEAD",
    )
    require(report.get("source_pair_id") == "xg1_fact_001", "GATE_PAIR_ID")
    require(report.get("cpu_model_forward_count") == 6, "GATE_CPU_FORWARD_COUNT")
    require(report.get("gpu_model_forward_count") == 6, "GATE_GPU_FORWARD_COUNT")
    require(report.get("total_model_forward_count") == 12, "GATE_TOTAL_FORWARD_COUNT")
    require(
        report.get("scientific_budget_forward_count") == 0,
        "GATE_SCIENTIFIC_BUDGET",
    )
    require(
        report.get("scientific_conclusion") is None,
        "GATE_SCIENTIFIC_CONCLUSION",
    )
    require(
        report.get("representative_checkpoint_sha256")
        == base.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
        "GATE_CHECKPOINT_IDENTITY",
    )
    require(
        report.get("xg1_rows_sha256")
        == xg1_eq.eligibility.EXPECTED_ROWS_SHA256,
        "GATE_XG1_ROWS_IDENTITY",
    )
    require(
        report.get("eligibility_anchor_manifest_sha256")
        == xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "GATE_ELIGIBILITY_MANIFEST_IDENTITY",
    )
    require(float(report["state_atol"]) == 1e-4, "GATE_STATE_ATOL")
    require(float(report["state_rtol"]) == 1e-4, "GATE_STATE_RTOL")
    require(float(report["geometry_atol"]) == 1e-4, "GATE_GEOMETRY_ATOL")
    require(float(report["geometry_rtol"]) == 1e-4, "GATE_GEOMETRY_RTOL")
    require(float(report["pe_atol"]) == 1e-4, "GATE_PE_ATOL")
    require(
        float(report["max_state_abs_diff"]) <= float(report["state_atol"]),
        "GATE_STATE_TOLERANCE",
    )
    require(
        float(report["max_geometry_abs_diff"]) <= float(report["geometry_atol"]),
        "GATE_GEOMETRY_TOLERANCE",
    )
    require(
        float(report["max_pe_abs_diff"]) <= float(report["pe_atol"]),
        "GATE_PE_TOLERANCE",
    )
    return report


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch in {"", EXPECTED_BRANCH}, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (DESIGN_FREEZE_COMMIT, "DESIGN_FREEZE"),
        (STRUCTURAL_FREEZE_COMMIT, "STRUCTURAL_FREEZE"),
        (ELIGIBILITY_FREEZE_COMMIT, "ELIGIBILITY_FREEZE"),
        (GATE_FREEZE_COMMIT, "GATE_FREEZE"),
    ):
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", ancestor, expected_head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"{label}_NOT_ANCESTOR")

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            GATE_FREEZE_COMMIT,
            expected_head,
            "--",
            *FROZEN_FULL_DEPENDENCY_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "FROZEN_FULL_DEPENDENCY_DRIFT")
    validate_gate_artifact()


def classify_regime(alignment_shift_abs: float) -> str:
    return base.classify_regime(alignment_shift_abs)


def group_size_passes(counts: Mapping[str, int]) -> bool:
    return base.group_size_passes(counts)


def _run_baseline_pair(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    public, plan = base._run_baseline_pair(*args, **kwargs)
    public = dict(public)
    public["schema_version"] = REGIME_ITEM_SCHEMA
    return public, plan


def freeze_regimes(
    pairs: Sequence[str],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require(len(pairs) == SOURCE_PAIR_COUNT, "FREEZE_PAIR_COUNT")
    require(len(baseline_rows) == SOURCE_PAIR_COUNT, "FREEZE_BASELINE_ROW_COUNT")
    frozen: list[dict[str, Any]] = []
    counts = {REGIME_LARGE: 0, REGIME_SMALL: 0}

    for pair, raw in zip(pairs, baseline_rows, strict=True):
        require(raw.get("source_pair_id") == pair, f"FREEZE_PAIR_ORDER:{pair}")
        require(
            not (_RESPONSE_FIELDS & set(raw)),
            f"FREEZE_RESPONSE_FIELD_LEAK:{pair}",
        )
        shift = float(raw["alignment_shift_abs"])
        regime = classify_regime(shift)
        require(raw.get("regime") == regime, f"FREEZE_REGIME_DRIFT:{pair}")
        row = {
            "schema_version": REGIME_ITEM_SCHEMA,
            "source_pair_id": pair,
            "target_C": float(raw["target_C"]),
            "reference_C": float(raw["reference_C"]),
            "alignment_shift_abs": shift,
            "threshold": ALIGNMENT_SHIFT_THRESHOLD,
            "regime": regime,
            "classification_frozen_before_alignment": True,
        }
        frozen.append(row)
        counts[regime] += 1

    require(sum(counts.values()) == SOURCE_PAIR_COUNT, "FREEZE_COUNT_TOTAL")
    return frozen, counts


def _write_bytes(path: Path, raw: bytes) -> str:
    require(not path.exists(), f"OUTPUT_COLLISION:{path.name}")
    path.write_bytes(raw)
    return base.sha256_bytes(raw)


def _write_regime_freeze(
    output_dir: Path,
    regime_rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")
    output_dir.mkdir(parents=True, exist_ok=False)

    regime_raw = base.jsonl_bytes(regime_rows)
    regime_sha = _write_bytes(output_dir / REGIME_FILE, regime_raw)
    freeze = {
        "schema_version": REGIME_FREEZE_SCHEMA,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "threshold_source": "original_frozen_discovery_q75",
        "source_pair_count": SOURCE_PAIR_COUNT,
        "n_LARGE": int(counts[REGIME_LARGE]),
        "n_SMALL": int(counts[REGIME_SMALL]),
        "minimum_group_size": MIN_GROUP_SIZE,
        "group_size_gate_pass": group_size_passes(counts),
        "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
        "alignment_model_forward_count_at_freeze": 0,
        "response_fields_observed_at_freeze": False,
        "regime_manifest_sha256": regime_sha,
    }
    freeze_sha = _write_bytes(
        output_dir / REGIME_FREEZE_FILE,
        base.canonical_json_bytes(freeze),
    )
    return {
        "regime_manifest_sha256": regime_sha,
        "regime_freeze_sha256": freeze_sha,
        "regime_freeze": freeze,
    }


def _run_alignment_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    item = dict(base._run_alignment_pair(*args, **kwargs))
    item["schema_version"] = ITEM_SCHEMA
    return item


def analyze_responses(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    analysis = dict(base.analyze_responses(items))
    supported = analysis["scientific_conclusion"] == "PROSPECTIVE_ADVERSE_REGIME_SUPPORTED"
    analysis.pop("scientific_conclusion", None)
    prospective = str(analysis.pop("prospective_regime_test"))
    return {
        **analysis,
        "xg1_prospective_regime_test": prospective,
        "scientific_conclusion": (
            "CROSS_GENERATOR_ADVERSE_REGIME_REPLICATED"
            if supported
            else "CROSS_GENERATOR_ADVERSE_REGIME_NOT_ESTABLISHED"
        ),
    }


def _write_final_artifacts(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]] | None,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in (REGIME_FILE, REGIME_FREEZE_FILE):
        path = output_dir / filename
        require(path.is_file(), f"MISSING_PREINTERVENTION_ARTIFACT:{filename}")
        hashes[filename] = base.sha256_file(path)

    if items is not None:
        raw = base.jsonl_bytes(items)
        hashes[ITEM_FILE] = _write_bytes(output_dir / ITEM_FILE, raw)

    hashes[SUMMARY_FILE] = _write_bytes(
        output_dir / SUMMARY_FILE,
        base.canonical_json_bytes(summary),
    )

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
    hashes[MANIFEST_FILE] = _write_bytes(
        output_dir / MANIFEST_FILE,
        base.canonical_json_bytes(manifest),
    )

    checksum_lines = [
        f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
    ]
    _write_bytes(
        output_dir / CHECKSUM_FILE,
        "".join(checksum_lines).encode("utf-8"),
    )
    return hashes


def _summary_provenance(
    *,
    expected_head: str,
    checkpoint_sha: str,
    freeze_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_head": expected_head,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "structural_freeze_commit": STRUCTURAL_FREEZE_COMMIT,
        "eligibility_freeze_commit": ELIGIBILITY_FREEZE_COMMIT,
        "gate_freeze_commit": GATE_FREEZE_COMMIT,
        "gate_execution_head": GATE_EXECUTION_HEAD,
        "gate_artifact_sha256": GATE_ARTIFACT_SHA256,
        "xg1_source_facts_sha256":
            xg1_eq.eligibility.EXPECTED_SOURCE_FACTS_SHA256,
        "xg1_rows_sha256": xg1_eq.eligibility.EXPECTED_ROWS_SHA256,
        "xg1_structural_manifest_sha256":
            xg1_eq.eligibility.EXPECTED_STRUCTURAL_MANIFEST_SHA256,
        "eligibility_anchor_manifest_sha256":
            xg1_eq.ELIGIBILITY_ANCHOR_MANIFEST_SHA256,
        "eligibility_summary_sha256": xg1_eq.ELIGIBILITY_SUMMARY_SHA256,
        "representative_checkpoint_sha256": checkpoint_sha,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "minimum_group_size": MIN_GROUP_SIZE,
        "regime_manifest_sha256": freeze_info["regime_manifest_sha256"],
        "regime_freeze_sha256": freeze_info["regime_freeze_sha256"],
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "response_dependent_exclusion": False,
        "threshold_reestimated": False,
    }


def run_full(
    *,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    authenticate_repo(expected_head)
    base.backend.runtime_gate()
    require(not output_dir.exists(), "OUTPUT_DIR_COLLISION")

    with base.backend.parent_runtime_rebind():
        rows, encoded, event_rows = xg1_eq.load_xg1_inputs(tokenizer_snapshot)
        pairs = xg1_eq._pair_order(rows)
        require(len(pairs) == SOURCE_PAIR_COUNT, "PAIR_COUNT")
        events = base.parent.event_lookup(event_rows)
        base.parent.validate_transport_event_plan(pairs, events)
        row_index = base.parent.build_row_index(rows)
        trace_code, trace_line = base.measurement._resolve_and_validate_runtime_binding()

        model, checkpoint_sha = base.parent.load_representative_model_external(
            model_snapshot=model_snapshot,
            checkpoint_path=checkpoint_path,
        )
        require(
            checkpoint_sha == base.extraction.REPRESENTATIVE_CHECKPOINT_SHA256,
            "CHECKPOINT_IDENTITY",
        )
        runtime_ctx = base.transport_runtime.validate_runtime_components(model)
        kernels = base.backend.load_exact_fast_kernels()
        model.to(torch.device("cuda:0"))
        model.eval()
        require(
            all(parameter.device.type == "cuda" for parameter in model.mamba.parameters()),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = base.backend._make_fast_capture(kernels)
        original_capture = base.parent.capture_branch
        budget = base.parent.ForwardBudget(FULL_FORWARD_BUDGET)
        baseline_rows: list[dict[str, Any]] = []
        runtime_plans: dict[str, dict[str, Any]] = {}

        base.parent.capture_branch = fast_capture
        try:
            # Phase 1: all 300 XG1 pairs complete baseline geometry first.
            # No intervention response exists before the persisted regime freeze.
            for pair in pairs:
                public, plan = _run_baseline_pair(
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
                baseline_rows.append(public)
                runtime_plans[pair] = plan

            require(
                budget.used == BASELINE_FORWARD_BUDGET,
                "BASELINE_FORWARD_BUDGET",
            )
            regime_rows, counts = freeze_regimes(pairs, baseline_rows)
            freeze_info = _write_regime_freeze(output_dir, regime_rows, counts)

            if not group_size_passes(counts):
                summary = {
                    "schema_version": SUMMARY_SCHEMA,
                    "result":
                        "PASS_XG1_BASELINE_PHASE_BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE",
                    **_summary_provenance(
                        expected_head=expected_head,
                        checkpoint_sha=checkpoint_sha,
                        freeze_info=freeze_info,
                    ),
                    "n_LARGE": int(counts[REGIME_LARGE]),
                    "n_SMALL": int(counts[REGIME_SMALL]),
                    "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
                    "alignment_model_forward_count": 0,
                    "total_model_forward_count": BASELINE_FORWARD_BUDGET,
                    "scientific_budget_forward_count": BASELINE_FORWARD_BUDGET,
                    "xg1_prospective_regime_test":
                        "BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE",
                    "scientific_conclusion": None,
                }
                _write_final_artifacts(
                    output_dir,
                    items=None,
                    summary=summary,
                )
                torch.cuda.synchronize()
                return summary

            # Phase 2 begins only after the persisted XG1 regime freeze and
            # fixed group-size gate have both passed.
            items: list[dict[str, Any]] = []
            for pair, baseline_row in zip(
                pairs,
                baseline_rows,
                strict=True,
            ):
                items.append(
                    _run_alignment_pair(
                        pair,
                        baseline_row=baseline_row,
                        runtime_plan=runtime_plans[pair],
                        model=model,
                        runtime_ctx=runtime_ctx,
                        trace_code=trace_code,
                        trace_line=trace_line,
                        encoded=encoded,
                        row_index=row_index,
                        budget=budget,
                    )
                )
            budget.assert_exact()
            torch.cuda.synchronize()
        finally:
            base.parent.capture_branch = original_capture

    analysis = analyze_responses(items)
    item_raw = base.jsonl_bytes(items)
    item_sha = base.sha256_bytes(item_raw)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": "PASS_XG1_FULL_EXECUTION",
        **_summary_provenance(
            expected_head=expected_head,
            checkpoint_sha=checkpoint_sha,
            freeze_info=freeze_info,
        ),
        "kernels_version": base.backend.KERNELS_VERSION,
        "mamba_revision": base.backend.MAMBA_REV,
        "mamba_binary_sha256": base.backend.MAMBA_BINARY_SHA256,
        "causal_conv_revision": base.backend.CONV_REV,
        "causal_conv_binary_sha256": base.backend.CONV_BINARY_SHA256,
        "build_variant": base.backend.BUILD_VARIANT,
        "python_version": base.backend.EXPECTED_RUNTIME["python"],
        "numpy_version": base.backend.EXPECTED_RUNTIME["numpy"],
        "torch_version": base.backend.EXPECTED_RUNTIME["torch"],
        "transformers_version": base.backend.EXPECTED_RUNTIME["transformers"],
        "cuda_runtime": base.backend.EXPECTED_CUDA_RUNTIME,
        "cuda_device": base.backend.EXPECTED_DEVICE_NAME,
        "cuda_capability": list(base.backend.EXPECTED_CAPABILITY),
        "baseline_model_forward_count": BASELINE_FORWARD_BUDGET,
        "alignment_model_forward_count": ALIGNMENT_FORWARD_BUDGET,
        "total_model_forward_count": FULL_FORWARD_BUDGET,
        "scientific_budget_forward_count": FULL_FORWARD_BUDGET,
        "item_metrics_sha256": item_sha,
        **analysis,
    }
    _write_final_artifacts(output_dir, items=items, summary=summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full 300-pair fast-CUDA XG1 cross-generator prospective "
            "replication. Executes 1200 baseline scientific forwards, freezes "
            "the fixed-threshold regimes, and executes the additional 600 "
            "alignment forwards only if both fixed groups contain at least 30 "
            "pairs."
        )
    )
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_full(
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=args.tokenizer_snapshot,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    print("RESULT =", summary["result"])
    print("N_LARGE =", summary["n_LARGE"])
    print("N_SMALL =", summary["n_SMALL"])
    print(
        "TOTAL_MODEL_FORWARD_COUNT =",
        summary["total_model_forward_count"],
    )
    print(
        "XG1_PROSPECTIVE_REGIME_TEST =",
        summary["xg1_prospective_regime_test"],
    )
    print(
        "SCIENTIFIC_CONCLUSION =",
        summary["scientific_conclusion"],
    )
    if "inferential_family" in summary:
        for test in summary["inferential_family"]["tests"]:
            print(
                test["hypothesis"],
                "RAW_P =",
                test["raw_p"],
                "HOLM_ADJUSTED_P =",
                test["holm_adjusted_p"],
                "HOLM_REJECT =",
                test["holm_reject"],
            )


if __name__ == "__main__":
    main()
