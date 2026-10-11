#!/usr/bin/env python3
"""E1: bounded original-helper CPU dispatch with a hard, observable call budget.

This invokes Stage C's injected original-helper transcription through an isolated
facade and checks each upstream call; it never calls original run_worker().
It cannot load a model or execute CUDA, authenticate sources, or issue E0-E3 PASS.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BASE = "c3c75af0e232252c6e1c0ec0ff3f0d1544c3a8fd"
BRANCH = "gen5-causal-role-state-ownership"
SCHEMA = "GEN5_V10_E1_BOUNDED_CPU_ORIGINAL_HELPER_BUDGET_V1"
ORIGINAL_BLOB = "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635"

# This is the *exact* audited upstream path through Stage C. The callback
# names below correspond to frozen helper functions, not fabricated D3 events.
ORDER = (
    "batch_features", "anchor_partition", "fine_masks", "context",
    *("raw_write" for _ in range(9)),
    "source_gradient", "projector_pinv",
    "orientation_accumulator", "orientation_accumulator", "project_group",
    "frozen_full_reducer",
)
CALL_BUDGET = {
    "batch_features": 1, "anchor_partition": 1, "fine_masks": 1,
    "context": 1, "raw_write": 9, "source_gradient": 1,
    "projector_pinv": 1, "orientation_accumulator": 2,
    "project_group": 1, "frozen_full_reducer": 1,
}


class E1Blocked(ValueError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise E1Blocked("GEN5_V10_E1_BOUNDED_BLOCKED:" + why)


class _BudgetedFrozen:
    """Proxy frozen helpers without monkeypatching module/global symbols.

    All methods are delegated to the injected original-function object. The
    proxy itself is NOT an authenticity check: test doubles remain test doubles.
    """

    def __init__(self, original: Any):
        self._original = original
        self.trace: list[str] = []
        self.counts: Counter[str] = Counter()

        legacy = original.transport.legacy
        self.transport = SimpleNamespace(
            RAW_RECON_ATOL=original.transport.RAW_RECON_ATOL,
            legacy=SimpleNamespace(
                _phase_b_batch_features=self._wrap("batch_features", legacy._phase_b_batch_features),
                _phase_b_prepare_common_context=self._wrap("context", legacy._phase_b_prepare_common_context),
                _phase_b_raw_write=self._wrap("raw_write", legacy._phase_b_raw_write),
            ),
            _cell_weights=original.transport._cell_weights,
            _source_gradient_bundle=self._wrap("source_gradient", original.transport._source_gradient_bundle),
            _prepare_projector_pinv=self._wrap("projector_pinv", original.transport._prepare_projector_pinv),
            _empty_orientation=self._wrap("orientation_accumulator", original.transport._empty_orientation),
            _project_visible_group=self._wrap("project_group", original.transport._project_visible_group),
        )

    def _wrap(self, label: str, fn: Any):
        require(callable(fn), "FROZEN_HELPER_NOT_CALLABLE:" + label)
        def invoke(*args: Any, **kwargs: Any):
            i = len(self.trace)
            require(i < len(ORDER) and ORDER[i] == label, "ORIGINAL_HELPER_ORDER:" + label)
            require(self.counts[label] < CALL_BUDGET[label], "CALL_BUDGET:" + label)
            # Record only AFTER successful return; rejected or raising functions
            # never create an apparently complete trace.
            value = fn(*args, **kwargs)
            self.trace.append(label)
            self.counts[label] += 1
            return value
        return invoke

    def _build_anchor_partition(self, *args: Any, **kwargs: Any):
        return self._wrap("anchor_partition", self._original._build_anchor_partition)(*args, **kwargs)

    def _batch_anchor_masks(self, *args: Any, **kwargs: Any):
        return self._wrap("fine_masks", self._original._batch_anchor_masks)(*args, **kwargs)

    def _component_fine_pair_stats(self, *args: Any, **kwargs: Any):
        return self._wrap("frozen_full_reducer", self._original._component_fine_pair_stats)(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def finish(self) -> dict[str, int]:
        require(tuple(self.trace) == ORDER, "INCOMPLETE_ORIGINAL_HELPER_TRACE")
        require(dict(self.counts) == CALL_BUDGET, "CALL_BUDGET_NOT_EXACT")
        return dict(self.counts)


def execute_injected_cpu_one_group(*, frozen: Any, hybrid: Any,
                                   row_ids: list[str], static: Any,
                                   encoded: Any, tokenizer_snapshot: Any,
                                   model: Any, wrapper: Any,
                                   strong_mask: Any, planes: Any,
                                   features: Any, active: Any,
                                   targets: Any) -> dict[str, Any]:
    """CPU injected one-group helper execution; never an actual science worker.

    All loaders are intentionally absent. A genuine CUDA tensor or production
    width is rejected by frozen Stage C before any reducer is invoked. Does
    not execute original run_worker, run_full, or Kaggle work.
    """
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "GPU_VISIBLE")
    require(type(row_ids) is list and len(row_ids) == 32 and
            all(type(x) is str and x for x in row_ids) and len(set(row_ids)) == 32,
            "ROW_IDS_EXACT")
    require(isinstance(active, (list, tuple)) and len(active) >= 832 and
            isinstance(targets, (list, tuple)) and len(targets) >= 832,
            "INJECTED_ROW_CAPACITY")
    require(hasattr(wrapper, "correction") and
            hasattr(wrapper.correction, "shape") and
            hasattr(wrapper.correction.shape, "intermediate_size") and
            hasattr(wrapper.correction.shape, "state_size"),
            "INJECTED_SHAPE")
    shape = wrapper.correction.shape
    require(type(shape.intermediate_size) is int and
            type(shape.state_size) is int and
            shape.intermediate_size > 0 and shape.state_size > 0,
            "CPU_SYNTHETIC_SHAPE_INTEGER")
    w = shape.intermediate_size * shape.state_size
    require(0 < w < 24576, "CPU_SYNTHETIC_WIDTH_ONLY")

    from scripts import gen5_v10_bounded_runtime_bridge_candidate as c
    guarded = _BudgetedFrozen(frozen)
    # E0's source-locus contract is guaranteed only for the immutable Git
    # bytes verified independently via the static CLI, not this injected object.
    result = c.execute_one_group_cpu_mock(
        frozen=guarded, hybrid=hybrid, row_ids=row_ids, static=static,
        encoded=encoded, tokenizer_snapshot=tokenizer_snapshot,
        model=model, wrapper=wrapper, strong_mask=strong_mask,
        planes=planes, features=features, active=active, targets=targets,
    )
    counts = guarded.finish()
    require(result.get("raw_projector_guard_status") ==
            "PASS_CPU_MOCK_ONLY_NOT_AUTHENTICATED", "ORIGINAL_GUARD_NOT_REPLAYED")
    require(result.get("V10_E0_PROVENANCE") == "NOT_EVALUATED" and
            result.get("V10_E1_FROZEN_GUARDS") == "NOT_EVALUATED" and
            result.get("SCIENTIFIC_EXECUTION") == "NOT_AUTHORIZED",
            "FORBIDDEN_SCIENCE_PROMOTION")
    require(set(result["input_capture"]) ==
            {"pre_frozen", "post_frozen", "post_v10"}, "CAPTURE_COVERAGE")
    require(all(len(result["input_capture"][p]) == 17 for p in
                ("pre_frozen", "post_frozen", "post_v10")), "CAPTURE_KEYS")
    return {
        "schema_version": SCHEMA,
        "frozen_git_blob_expected": ORIGINAL_BLOB,
        "selected_locus": dict(result["selector"]),
        "original_helper_trace": list(guarded.trace),
        "original_helper_counts": counts,
        "input_manifest": result["input_manifest"],
        "frozen_records": result["frozen_records"],
        "v10_records": result["v10_records"],
        "v10_dispatch": result["v10_dispatch"],
        "upstream_guard_status": "CPU_MOCK_REPLAY_ONLY_NOT_AUTHENTICATED",
        "runtime_origin_attested": False,
        "GPU_EXECUTION": "NOT_AUTHORIZED",
        "V10_E0_PROVENANCE": "NOT_EVALUATED",
        "V10_E1_FROZEN_GUARDS": "NOT_EVALUATED",
        "V10_E2_PAYLOAD": "NOT_EVALUATED",
        "V10_E3_DIAGNOSTICS": "NOT_EVALUATED",
        "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
        "PRODUCTION": "BLOCKED",
    }


def inspect_local_source() -> dict[str, Any]:
    """Inspect original committed bytes, rather than importing run_worker."""
    import subprocess
    from scripts import gen5_v10_stage_e0_frozen_locus_adapter_candidate as e0
    repo = Path(__file__).resolve().parents[1]
    def git(*parts: str) -> bytes:
        proc = subprocess.run(["git", "-C", str(repo), *parts],
                              capture_output=True, timeout=20, check=False)
        require(proc.returncode == 0, "LOCAL_GIT_READ_FAILURE")
        return proc.stdout
    require(git("rev-parse", "HEAD").strip().decode("ascii") == BASE,
            "LOCAL_HEAD_MISMATCH")
    require(git("branch", "--show-current").strip().decode("ascii") == BRANCH,
            "LOCAL_BRANCH_MISMATCH")
    require(git("rev-parse", "HEAD:" + e0.ORIGINAL).strip().decode("ascii") ==
            ORIGINAL_BLOB, "ORIGINAL_BLOB_MISMATCH")
    require(git("diff", "--name-only").strip() == b"" and
            git("diff", "--cached", "--name-only").strip() == b"",
            "UNEXPECTED_TRACKED_CHANGES")
    raw = git("show", "HEAD:" + e0.ORIGINAL)
    report = e0.inspect_frozen_source(raw)
    require(report["frozen_git_blob"] == ORIGINAL_BLOB and
            not report["frozen_worker_direct_invocation_allowed"],
            "FROZEN_STATIC_SOURCE_REJECTED")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--static-contract-check", action="store_true")
    flags.add_argument("--inspect-local-frozen-source", action="store_true")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "GPU_VISIBLE")
    if args.inspect_local_frozen_source:
        r = inspect_local_source()
        print("GEN5_V10_STAGE_E1_FROZEN_SOURCE_INTEGRATION_PASS")
        print("HEAD=" + BASE + ";FROZEN_BLOB=" + r["frozen_git_blob"])
        print("STATIC_SOURCE_ONLY=YES;GPU_EXECUTION=NOT_AUTHORIZED")
    else:
        require(len(ORDER) == 19 and sum(CALL_BUDGET.values()) == 19,
                "INTERNAL_BUDGET_DRIFT")
        print("GEN5_V10_STAGE_E1_CPU_BOUNDED_DISPATCH_STATIC_PASS")
        print("BASE=" + BASE + ";ORIGINAL_HELPER_CALLS=" + str(len(ORDER)))
        print("GPU=OFF;MODEL_LOAD=NO;E0_E3=NOT_EVALUATED;SCIENTIFIC_EXECUTION=NOT_AUTHORIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
