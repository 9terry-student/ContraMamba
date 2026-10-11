#!/usr/bin/env python3
"""Source-bound, non-executing Gen5 v10 E0 bounded-worker integration adapter.

Reads the exact frozen *Git blob* and independently locates the real worker's
batch/source/group and guard call sites. Does not load the original Python
module, models, torch, CUDA, checkpoints or execute any worker callback.
This is a static link contract, not an executable GPU worker or E0 attestation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "GEN5_V10_STAGE_E0_SOURCE_BOUND_WORKER_ADAPTER_V1"
BASE = "59bf37f426ccc1620a66be21e131b3a3d6b4fdd8"
BRANCH = "gen5-causal-role-state-ownership"
ORIGINAL = "scripts/audit_native_recurrent_generator_anchor_pair_localization.py"
ORIGINAL_BLOB = "f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635"
FIXED = {"worker_id": 1, "batch_index_one_based": 13,
         "row_start": 800, "row_stop": 832, "source_cell": "6201_6201",
         "target_cells": ["6201_6202", "6201_6203"],
         "K": 4, "B": 32, "S": 128, "W": 24576}
# Audited from the unchanged source at ORIGINAL_BLOB: explicit lexical ordering.
# These are callsites, NOT claims that the code or guards have executed.
ORDER = (
    "transport.legacy._phase_b_batch_features",
    "coarse._serialization_role_masks",
    "_batch_anchor_masks",
    "transport.legacy._phase_b_prepare_common_context",
    "kernel._discrete_log_a",
    "pairgap._backward_survival_factor",
    "transport.legacy._phase_b_raw_write",
    "transport._source_gradient_bundle",
    "transport._prepare_projector_pinv",
    "transport._project_visible_group",
    "torch.cat",
    "_component_fine_pair_stats",
    "_accumulate_component_row",
)
RAW_GUARDS = (
    "RAW_RECONSTRUCTION_AUTH",
    "VISIBLE_RAW_ENERGY_INTERNAL_REPLAY",
    "COMPLEMENT_RAW_ENERGY_INTERNAL_REPLAY",
)
RUNTIME_BLOCKERS = (
    "EXPLICIT_BOUNDED_GPU_EXECUTION_AUTHORITY_MISSING",
    "REAL_BOUNDARY_INSTRUMENTATION_NOT_IMPLEMENTED",
    "ACTUAL_MODEL_CHECKPOINT_TOKENIZER_BYTES_UNAUTHENTICATED",
    "FROZEN_ALL_NINE_RAW_WRITE_GPU_MEMORY_UNMEASURED",
    "FROZEN_FULL_FFT_V10_GEMM_AND_CAPTURE_PEAK_UNMEASURED",
    "ORIGINAL_UPSTREAM_GUARD_EXECUTION_UNAUTHENTICATED",
    "TRUSTED_EXTERNAL_ARTIFACT_PUBLICATION_RECEIPT_MISSING",
)

class E0Blocked(ValueError):
    pass


def require(ok: bool, code: str) -> None:
    if not ok:
        raise E0Blocked("GEN5_V10_E0_SOURCE_BOUND_BLOCKED:" + code)


def git_blob_sha(raw: bytes) -> str:
    require(type(raw) is bytes, "SOURCE_BYTES")
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\x00" + raw).hexdigest()


def _ancestry(fn: ast.AST) -> dict[int, tuple[ast.AST, ...]]:
    parents: dict[int, tuple[ast.AST, ...]] = {}
    def visit(node: ast.AST, chain: tuple[ast.AST, ...]) -> None:
        parents[id(node)] = chain
        for child in ast.iter_child_nodes(node):
            visit(child, chain + (node,))
    visit(fn, ())
    return parents


def _under(node: ast.AST, ancestor: ast.AST, parents: dict[int, tuple[ast.AST, ...]]) -> bool:
    return ancestor in parents[id(node)]


def _loops(fn: ast.AST, target: str, expr: str) -> list[ast.For]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.For)
            and ast.unparse(n.target) == target and ast.unparse(n.iter) == expr]


def _calls(fn: ast.AST, qualified: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and ast.unparse(n.func) == qualified]


def inspect_frozen_source(raw: bytes, *, enforce_blob: bool = True) -> dict[str, Any]:
    """Read-only AST verification of exact original run_worker nesting and guards.

    Can run on injected test snippets when enforce_blob=False. Never eval/exec.
    """
    require(type(raw) is bytes and 0 < len(raw) < 2_000_000, "SOURCE_BYTES")
    sha = git_blob_sha(raw)
    if enforce_blob:
        require(sha == ORIGINAL_BLOB, "FROZEN_SOURCE_BLOB_MISMATCH")
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=ORIGINAL)
    except (SyntaxError, UnicodeError) as exc:
        raise E0Blocked("GEN5_V10_E0_SOURCE_BOUND_BLOCKED:INVALID_SOURCE") from exc
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "run_worker"]
    require(len(funcs) == 1 and funcs[0].args.args and
            funcs[0].args.args[0].arg == "args", "ORIGINAL_WORKER_ENTRYPOINT")
    fn = funcs[0]
    parents = _ancestry(fn)
    batch = _loops(fn, "(batch_index, start)",
                   "enumerate(range(row_start, row_stop, BATCH_ROWS), start=1)")
    if not batch:
        batch = _loops(fn, "batch_index, start",
                       "enumerate(range(row_start, row_stop, BATCH_ROWS), start=1)")
    source_loops = _loops(fn, "source", "local_sources")
    raw_loops = _loops(fn, "cell", "FULL_FACTORIAL_CELLS")
    groups = _loops(fn, "group_start", "range(0, len(source_rows), TARGETS_PER_TRANSPORT)")
    require(len(batch) == len(source_loops) == len(raw_loops) == len(groups) == 1,
            "EXACT_FROZEN_LOOP_NESTING")
    batch, sources, raw_loop, group = batch[0], source_loops[0], raw_loops[0], groups[0]
    require(_under(sources, batch, parents) and _under(raw_loop, batch, parents) and
            _under(group, sources, parents) and not _under(raw_loop, sources, parents),
            "LOOP_PARENT_ORDER")
    callsites: dict[str, int] = {}
    callnodes: dict[str, ast.Call] = {}
    # The authentic worker accumulates each target's visible and complement
    # components at TWO lexical call sites within the target loop. Requiring
    # one site spuriously blocked the frozen Git blob. Keep the stronger
    # two-call structural contract instead of silently accepting either count.
    component_calls: list[ast.Call] = []
    for name in ORDER:
        matches = _calls(fn, name)
        if name == "_accumulate_component_row":
            require(len(matches) == 2, "COMPONENT_ACCUMULATION_CARDINALITY")
            component_calls = sorted(matches, key=lambda n: (n.lineno, n.col_offset))
            callnodes[name] = component_calls[0]
        else:
            require(len(matches) == 1, "CALLSITE_CARDINALITY:" + name)
            callnodes[name] = matches[0]
        callsites[name] = callnodes[name].lineno
    require(list(callsites.values()) == sorted(callsites.values()) and
            len(set(callsites.values())) == len(ORDER), "CALLSITE_ORDER")
    for name in ORDER:
        require(_under(callnodes[name], batch, parents), "CALL_OUTSIDE_BATCH:" + name)
    require(_under(callnodes["transport.legacy._phase_b_raw_write"], raw_loop, parents),
            "RAW_WRITE_NOT_ALL_NINE")
    for name in ("transport._source_gradient_bundle", "transport._prepare_projector_pinv"):
        require(_under(callnodes[name], sources, parents) and
                not _under(callnodes[name], group, parents), "SOURCE_GRADIENT_SCOPE:" + name)
    for name in ("transport._project_visible_group", "torch.cat",
                 "_component_fine_pair_stats", "_accumulate_component_row"):
        require(_under(callnodes[name], group, parents), "TARGET_GROUP_SCOPE:" + name)
    # Frozen target loop: for EACH of two target rows, visible stats[index]
    # and complement stats[target_count + index] are accumulated separately.
    target_loops = _loops(fn, "(index, row)", "enumerate(group_rows)")
    if not target_loops:
        target_loops = _loops(fn, "index, row", "enumerate(group_rows)")
    require(len(target_loops) == 1 and _under(target_loops[0], group, parents),
            "COMPONENT_TARGET_LOOP")
    target_loop = target_loops[0]
    require(all(_under(node, target_loop, parents) for node in component_calls),
            "COMPONENT_CALLS_NOT_IN_TARGET_LOOP")
    assignments = [node for node in ast.walk(group)
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)]
    def one_assignment(name: str, expression: str, parent: ast.AST) -> ast.Assign:
        found = [node for node in assignments if node.targets[0].id == name and
                 ast.unparse(node.value) == expression and _under(node, parent, parents)]
        require(len(found) == 1, "COMPONENT_INDEX_BINDING:" + name)
        return found[0]
    count_binding = one_assignment("target_count", "len(group_rows)", group)
    visible_binding = one_assignment("visible_stats", "stats[index]", target_loop)
    complement_binding = one_assignment("complement_stats", "stats[target_count + index]", target_loop)
    require(count_binding.lineno < target_loop.lineno and
            visible_binding.lineno < complement_binding.lineno < component_calls[0].lineno,
            "COMPONENT_BINDING_ORDER")
    for node, prefix, source in zip(component_calls,
                                    ("visible", "complement"),
                                    ("visible_stats", "complement_stats")):
        kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
        require(len(node.args) == 1 and ast.unparse(node.args[0]) == "accumulator" and
                kw == {"prefix": repr(prefix), "stats": source},
                "COMPONENT_ACCUMULATION_BINDING:" + prefix)
    require(component_calls[0].lineno < component_calls[1].lineno,
            "COMPONENT_ACCUMULATION_ORDER")
    # Neither reducer nor projector may be redirected to a local surrogate.
    cat = callnodes["torch.cat"]
    require(cat.args and ast.unparse(cat.args[0]) == "(visible, complement)" and
            any(k.arg == "dim" and isinstance(k.value, ast.Constant) and k.value.value == 0
                for k in cat.keywords), "EXACT_K4_CONCAT")
    stats = callnodes["_component_fine_pair_stats"]
    kwargs = {k.arg: ast.unparse(k.value) for k in stats.keywords}
    require(kwargs == {"component":"combined","log_a":"pair_log_a",
                       "survival":"pair_survival","attention_mask":"batch_mask",
                       "role_masks":"role_masks","fine_masks":"fine_masks",
                       "shape":"wrapper.correction.shape"}, "FROZEN_STATS_ARGUMENT_BINDING")
    guards: dict[str, int] = {}
    for label in RAW_GUARDS:
        matches = [n for n in ast.walk(group) if isinstance(n, ast.Constant) and
                   n.value == label and _under(n, group, parents)]
        require(len(matches) == 1 and matches[0].lineno > component_calls[1].lineno,
                "FROZEN_GUARD_MISSING:" + label)
        guards[label] = matches[0].lineno
    # Source validation must remain before original model/data initialization.
    auth = _calls(fn, "_validate_runtime_head")
    inputs = _calls(fn, "transport._runtime_inputs")
    require(len(auth) == len(inputs) == 1 and auth[0].lineno < inputs[0].lineno and
            inputs[0].lineno < callsites[ORDER[0]], "PRELOAD_SOURCE_GATE")
    return {
        "schema_version": SCHEMA, "frozen_git_blob": sha,
        "frozen_worker_function": "run_worker",
        "static_callsites": callsites,
        "component_accumulation_callsites": {
            "visible": component_calls[0].lineno,
            "complement": component_calls[1].lineno,
        },
        "frozen_raw_guard_literals": guards,
        "selected_locus": FIXED.copy(),
        "original_batch_loop_traverses_multiple_rows": True,
        "original_source_loop_traverses_multiple_sources": True,
        "new_adapter_must_bound_before_model_or_gradient": True,
        "frozen_worker_direct_invocation_allowed": False,
        "static_source_semantics": "SOURCE_NESTING_VERIFIED_NOT_EXECUTED",
        "runtime_source_authenticated": False,
        "real_cuda_peak_measured": False,
        "readiness": "BLOCKED_BOUND_EXECUTION_AND_INDEPENDENT_EVIDENCE",
        "missing_evidence": list(RUNTIME_BLOCKERS),
        "GPU_EXECUTION": "NOT_AUTHORIZED", "SCIENTIFIC_EXECUTION": "NOT_AUTHORIZED",
        "E0": "NOT_EVALUATED", "E1": "NOT_EVALUATED",
        "E2": "NOT_EVALUATED", "E3": "NOT_EVALUATED", "PRODUCTION": "BLOCKED",
    }


def _git(repo: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args],
                                capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise E0Blocked("GEN5_V10_E0_SOURCE_BOUND_BLOCKED:GIT_IO") from exc
    require(result.returncode == 0, "GIT_COMMAND_FAILED:" + args[0])
    return result.stdout.strip() if args[0] != "show" else result.stdout


def inspect_pinned_repo(repo: Path) -> dict[str, Any]:
    """Read committed original as raw bytes; never import/execute the worker.

    New E0 candidate files may be untracked during local validation. Only
    unrelated tracked differences and original source modifications block.
    """
    require(isinstance(repo, Path) and repo.is_dir() and not repo.is_symlink(),
            "REPO_DIRECTORY")
    require(_git(repo,"rev-parse","HEAD").decode("ascii") == BASE,"HEAD_MISMATCH")
    require(_git(repo,"branch","--show-current").decode("ascii") == BRANCH,
            "BRANCH_MISMATCH")
    require(_git(repo,"diff","--name-only") == b"" and
            _git(repo,"diff","--cached","--name-only") == b"", "TRACKED_MODIFICATIONS")
    require(_git(repo,"rev-parse","HEAD:" + ORIGINAL).decode("ascii") == ORIGINAL_BLOB,
            "FROZEN_BLOB_IN_COMMIT")
    raw = _git(repo,"show","HEAD:" + ORIGINAL)
    report = inspect_frozen_source(raw)
    report["source_read_mode"] = "GIT_SHOW_COMMITTED_BLOB_NOT_RUNTIME_LOADED_SOURCE"
    report["committed_head"] = BASE
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    flags = p.add_mutually_exclusive_group(required=True)
    flags.add_argument("--static-contract-check", action="store_true")
    flags.add_argument("--inspect-local-frozen-source", action="store_true")
    args = p.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
            "CUDA_VISIBLE_DISABLED_REQUIRED")
    if args.inspect_local_frozen_source:
        report = inspect_pinned_repo(Path(__file__).resolve().parents[1])
        print("GEN5_V10_STAGE_E0_FROZEN_SOURCE_AST_PASS")
        print("COMMITTED_HEAD=" + report["committed_head"] + ";BLOB=" + report["frozen_git_blob"])
        print("FROZEN_CALLSITES=" + str(len(report["static_callsites"])) + ";RAW_GUARDS=" + str(len(report["frozen_raw_guard_literals"])))
        print("GPU_EXECUTION=" + report["GPU_EXECUTION"] + ";E0=" + report["E0"])
        return 0
    print("GEN5_V10_STAGE_E0_SOURCE_BOUND_STATIC_PASS")
    print("SCHEMA=" + SCHEMA + ";BASE=" + BASE + ";FROZEN_BLOB=" + ORIGINAL_BLOB)
    print("GPU=OFF;MODEL_LOAD=NO;ORIGINAL_WORKER_NOT_INVOKED;E0_E3=NOT_EVALUATED")
    print("READINESS=BLOCKED_BOUND_EXECUTION_AND_INDEPENDENT_EVIDENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
