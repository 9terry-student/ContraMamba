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

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence as prevalence_eq
from scripts import reason_router_gen4_xg2_xg4_fresh_response_full_cuda_one_pair_equivalence as fresh_eq


ROOT = Path(__file__).resolve().parents[1]

DESIGN_FREEZE_COMMIT = "2074f52d39bf0fca6d63248016ce47c54bff5e06"
EQUIVALENCE_ARTIFACT_FREEZE_COMMIT = (
    "9ab93bcbb05fe2ba69e1d8f0c77b62626edd9945"
)
EQUIVALENCE_EXECUTION_HEAD = (
    "0385dbe8989c4468b471eef2e2a7196bac20826d"
)

SOURCE_PAIR_COUNT = 300
BASELINE_FORWARDS_PER_PAIR = 4
FULL_BASELINE_FORWARD_BUDGET = (
    SOURCE_PAIR_COUNT * BASELINE_FORWARDS_PER_PAIR
)

ALIGNMENT_SHIFT_THRESHOLD = 0.11228626366380845
MIN_GROUP_SIZE = 30
REGIME_LARGE = "LARGE"
REGIME_SMALL = "SMALL"

ITEM_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-baseline-regime-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-baseline-regime-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-xg2-xg4-fresh-response-baseline-regime-manifest-v1"
)

RESULT_PASS = (
    "PASS_XG2_XG4_FRESH_RESPONSE_BASELINE_REGIME_FREEZE"
)
REGIME_TEST_READY = "READY_FOR_ALIGNMENT_RESPONSE"
REGIME_TEST_BLOCKED = "BLOCKED_INSUFFICIENT_FIXED_GROUP_SIZE"

ITEM_FILE = "baseline_items.jsonl"
SUMMARY_FILE = "regime_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"

EQUIVALENCE_ARTIFACTS: dict[str, dict[str, str]] = {
    "xg2": {
        "path": (
            "reports/"
            "reason_router_gen4_xg2_xg4_fresh_response_"
            "full_cuda_one_pair_equivalence_0385dbe_r3/"
            "xg2/report.json"
        ),
        "sha256": (
            "db3fbbaeccd72341d8580151e83aa6cc"
            "54b0b4653598653bf8e516f8fd3b843a"
        ),
        "pair_id": "xg2_fact_301",
    },
    "xg4": {
        "path": (
            "reports/"
            "reason_router_gen4_xg2_xg4_fresh_response_"
            "full_cuda_one_pair_equivalence_0385dbe_r3/"
            "xg4/report.json"
        ),
        "sha256": (
            "9211d969cccb17b1a5000973b1e187d6"
            "ac1b7ab550c2e13ff3ade87fe503c053"
        ),
        "pair_id": "xg4_fact_301",
    },
}

REUSED_BLOBS = {
    (
        "scripts/"
        "reason_router_gen4_xg2_xg4_fresh_response_"
        "full_cuda_one_pair_equivalence.py"
    ): "e8afdcad264458c6008516684bf73f22ce240402",
    (
        "scripts/"
        "reason_router_gen4_generator_family_prevalence_"
        "fast_cuda_one_pair_equivalence.py"
    ): "41142f6090d7e3496e56c9e2871299279a4756f8",
    (
        "scripts/"
        "reason_router_gen4_generator_family_prevalence_"
        "kernel_compat.py"
    ): "d46cbe75a2697b9fb38ac2d4021de4779030fc12",
}


class FreshBaselineError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise FreshBaselineError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
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


def jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(dict(row))
        for row in rows
    )


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshBaselineError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    rc = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return rc == 0


def authenticate_repo(
    expected_head: str,
) -> None:
    fresh_eq.authenticate_repo(expected_head)

    require(
        _git_is_ancestor(
            EQUIVALENCE_ARTIFACT_FREEZE_COMMIT,
            expected_head,
        ),
        "EQUIVALENCE_ARTIFACT_FREEZE_NOT_ANCESTOR",
    )

    for path, expected_blob in REUSED_BLOBS.items():
        observed = git(
            "rev-parse",
            f"HEAD:{path}",
        )
        require(
            observed == expected_blob,
            f"REUSED_BLOB_DRIFT:{path}:{observed}",
        )


def validate_equivalence_artifact(
    family: str,
) -> dict[str, Any]:
    require(
        family in EQUIVALENCE_ARTIFACTS,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    spec = EQUIVALENCE_ARTIFACTS[family]
    path = ROOT / spec["path"]

    require(
        path.is_file(),
        f"EQUIVALENCE_ARTIFACT_MISSING:{family}",
    )
    require(
        sha256_file(path) == spec["sha256"],
        f"EQUIVALENCE_ARTIFACT_SHA256:{family}",
    )

    report = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    require(
        report.get("result")
        == fresh_eq.RESULT_PASS,
        f"EQUIVALENCE_RESULT:{family}",
    )
    require(
        report.get("family_key") == family,
        f"EQUIVALENCE_FAMILY:{family}",
    )
    require(
        report.get("source_pair_id")
        == spec["pair_id"],
        f"EQUIVALENCE_PAIR:{family}",
    )
    require(
        report.get("execution_head")
        == EQUIVALENCE_EXECUTION_HEAD,
        f"EQUIVALENCE_EXECUTION_HEAD:{family}",
    )
    require(
        report.get("cpu_model_forward_count") == 6,
        f"EQUIVALENCE_CPU_FORWARDS:{family}",
    )
    require(
        report.get("gpu_model_forward_count") == 6,
        f"EQUIVALENCE_GPU_FORWARDS:{family}",
    )
    require(
        report.get("total_model_forward_count") == 12,
        f"EQUIVALENCE_TOTAL_FORWARDS:{family}",
    )
    require(
        report.get("scientific_budget_forward_count") == 0,
        f"EQUIVALENCE_SCIENTIFIC_BUDGET:{family}",
    )
    require(
        report.get("scientific_conclusion") is None,
        f"EQUIVALENCE_SCIENTIFIC_CONCLUSION:{family}",
    )
    require(
        report.get("training_executed") is False,
        f"EQUIVALENCE_TRAINING:{family}",
    )
    require(
        report.get("backward_executed") is False,
        f"EQUIVALENCE_BACKWARD:{family}",
    )
    return report


def classify_regime(
    alignment_shift_abs: float,
) -> str:
    value = float(alignment_shift_abs)
    require(
        math.isfinite(value),
        "NONFINITE_ALIGNMENT_SHIFT",
    )
    return (
        REGIME_LARGE
        if value >= ALIGNMENT_SHIFT_THRESHOLD
        else REGIME_SMALL
    )


def summarize_regimes(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    pair_ids = [
        str(item["source_pair_id"])
        for item in items
    ]
    require(
        len(set(pair_ids)) == SOURCE_PAIR_COUNT,
        "PAIR_ID_UNIQUENESS",
    )

    counts = Counter(
        str(item["regime"])
        for item in items
    )
    require(
        set(counts)
        <= {REGIME_LARGE, REGIME_SMALL},
        "REGIME_VALUE",
    )

    n_large = int(counts[REGIME_LARGE])
    n_small = int(counts[REGIME_SMALL])

    require(
        n_large + n_small == SOURCE_PAIR_COUNT,
        "REGIME_COUNT_TOTAL",
    )

    viable = (
        n_large >= MIN_GROUP_SIZE
        and n_small >= MIN_GROUP_SIZE
    )

    return {
        "n_LARGE": n_large,
        "n_SMALL": n_small,
        "p_LARGE": n_large / SOURCE_PAIR_COUNT,
        "p_SMALL": n_small / SOURCE_PAIR_COUNT,
        "minimum_group_size": MIN_GROUP_SIZE,
        "support_gate_pass": viable,
        "prospective_regime_test": (
            REGIME_TEST_READY
            if viable
            else REGIME_TEST_BLOCKED
        ),
    }


def _write_outputs(
    output_dir: Path,
    *,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

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
                "bytes": int(
                    (output_dir / name).stat().st_size
                ),
            }
            for name, digest in sorted(hashes.items())
        },
    }

    manifest_raw = canonical_json_bytes(manifest)
    (output_dir / MANIFEST_FILE).write_bytes(
        manifest_raw
    )
    hashes[MANIFEST_FILE] = sha256_bytes(
        manifest_raw
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(hashes.items())
    ).encode("utf-8")

    (output_dir / CHECKSUM_FILE).write_bytes(
        checksum_raw
    )

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
    require(
        family in fresh_eq.FAMILY_CONFIG,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    authenticate_repo(expected_head)
    gate_report = validate_equivalence_artifact(
        family
    )

    prevalence_eq.backend.runtime_gate()

    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    with prevalence_eq.backend.parent_runtime_rebind():
        rows, encoded, event_rows = (
            fresh_eq.load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )

        pairs = fresh_eq._pair_order(
            family,
            rows,
        )

        require(
            len(pairs) == SOURCE_PAIR_COUNT,
            "PAIR_COUNT",
        )
        require(
            pairs[0] == f"{family}_fact_301",
            "FIRST_PAIR",
        )
        require(
            pairs[-1] == f"{family}_fact_600",
            "LAST_PAIR",
        )

        events = prevalence_eq.parent.event_lookup(
            event_rows
        )
        prevalence_eq.parent.validate_transport_event_plan(
            pairs,
            events,
        )

        row_index = prevalence_eq.parent.build_row_index(
            rows
        )

        trace_code, trace_line = (
            prevalence_eq.measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            prevalence_eq.kernel_compat
            .load_exact_fast_kernels()
        )

        with (
            prevalence_eq.kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                prevalence_eq.parent
                .load_representative_model_external(
                    model_snapshot=model_snapshot,
                    checkpoint_path=checkpoint_path,
                )
            )

            require(
                checkpoint_sha
                == prevalence_eq.extraction
                .REPRESENTATIVE_CHECKPOINT_SHA256,
                "CHECKPOINT_IDENTITY",
            )

            runtime_ctx = (
                prevalence_eq.transport_runtime
                .validate_runtime_components(model)
            )

        constructor_counts = Counter(
            constructor_kernel_calls
        )

        require(
            set(constructor_counts)
            == {"causal-conv1d", "mamba-ssm"},
            (
                "TRANSFORMERS_CONSTRUCTOR_KERNEL_NAMES:"
                f"{dict(constructor_counts)}"
            ),
        )
        require(
            constructor_counts["causal-conv1d"] > 0,
            "TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT_ZERO",
        )
        require(
            constructor_counts["causal-conv1d"]
            == constructor_counts["mamba-ssm"],
            (
                "TRANSFORMERS_CONSTRUCTOR_KERNEL_CALL_COUNT_MISMATCH:"
                f"{dict(constructor_counts)}"
            ),
        )

        prevalence_eq.kernel_compat.validate_transformers_kernel_bindings(
            kernels
        )

        model.to(
            torch.device("cuda:0")
        )
        model.eval()

        require(
            all(
                parameter.device.type == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            prevalence_eq.backend
            ._make_fast_capture(kernels)
        )

        original_capture = (
            prevalence_eq.parent.capture_branch
        )

        budget = prevalence_eq.parent.ForwardBudget(
            FULL_BASELINE_FORWARD_BUDGET
        )

        items: list[dict[str, Any]] = []

        prevalence_eq.parent.capture_branch = (
            fast_capture
        )

        try:
            for pair in pairs:
                raw = (
                    prevalence_eq
                    .run_baseline_geometry_pair(
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
                )

                shift = float(
                    raw["alignment_shift_abs"]
                )

                item = dict(raw)
                item["schema_version"] = ITEM_SCHEMA
                item["threshold"] = (
                    ALIGNMENT_SHIFT_THRESHOLD
                )
                item["regime"] = classify_regime(
                    shift
                )
                item[
                    "classification_from_baseline_only"
                ] = True

                items.append(item)

            budget.assert_exact()
            torch.cuda.synchronize()

        finally:
            prevalence_eq.parent.capture_branch = (
                original_capture
            )

    support = summarize_regimes(items)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "result": RESULT_PASS,
        "family_key": family,
        "execution_head": expected_head,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "equivalence_artifact_freeze_commit": (
            EQUIVALENCE_ARTIFACT_FREEZE_COMMIT
        ),
        "equivalence_execution_head": (
            EQUIVALENCE_EXECUTION_HEAD
        ),
        "equivalence_artifact_sha256": (
            EQUIVALENCE_ARTIFACTS[family]["sha256"]
        ),
        "equivalence_result": gate_report["result"],
        "representative_checkpoint_sha256": (
            checkpoint_sha
        ),
        "pair_id_first": pairs[0],
        "pair_id_last": pairs[-1],
        "source_pair_count": SOURCE_PAIR_COUNT,
        "threshold": ALIGNMENT_SHIFT_THRESHOLD,
        "threshold_reestimated": False,
        **support,
        "baseline_model_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "total_model_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "scientific_budget_forward_count": (
            FULL_BASELINE_FORWARD_BUDGET
        ),
        "alignment_model_forward_count": 0,
        "baseline_only": True,
        "alignment_intervention_executed": False,
        "magnitude_intervention_executed": False,
        "response_fields_observed": False,
        "r_align_observed": False,
        "h1_test_executed": False,
        "h2_test_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "scientific_conclusion_scope": (
            "FRESH_BASELINE_REGIME_SUPPORT_ONLY"
        ),
    }

    _write_outputs(
        output_dir,
        items=items,
        summary=summary,
    )

    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh XG2/XG4 Phase-1 baseline regime freeze. "
            "Runs exactly 300 fresh pairs x C0/C1/C2/C5 "
            "= 1200 scientific forwards for one family. "
            "No alignment response, R_ALIGN, H1/H2, "
            "task head, training, or backward execution."
        )
    )

    parser.add_argument(
        "--family",
        choices=tuple(
            sorted(fresh_eq.FAMILY_CONFIG)
        ),
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
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
    print(
        "PROSPECTIVE_REGIME_TEST =",
        summary["prospective_regime_test"],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT =",
        summary["baseline_model_forward_count"],
    )
    print(
        "ALIGNMENT_MODEL_FORWARD_COUNT =",
        summary["alignment_model_forward_count"],
    )
    print(
        "RESPONSE_FIELDS_OBSERVED =",
        summary["response_fields_observed"],
    )


if __name__ == "__main__":
    main()
