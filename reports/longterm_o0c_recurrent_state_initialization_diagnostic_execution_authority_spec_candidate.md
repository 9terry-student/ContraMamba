# Longterm O0c Recurrent-State Initialization Diagnostic Execution Authority Spec Candidate

## 1. Overall Authoring Verdict

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

Phase:

`REPORT_ONLY_RECURRENT_STATE_INITIALIZATION_DIAGNOSTIC_EXECUTION_AUTHORITY_AUTHORING`

This candidate authorizes exactly one future CPU-only, read-only diagnostic to determine why the corrected exact-runtime preflight at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948` returned:

`BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`

with blocker:

`recurrent_state_initialization`

This candidate does not classify the blocker in advance as an implementation defect, source incompatibility, validator false negative, or scientific result.

This authoring task does not run Kaggle, register a run, execute the diagnostic, rerun the preflight, load a model/tokenizer/dataset, perform forward/generation, train/evaluate, mutate packages, mutate implementation/tests/prior reports, stage, commit, or push.

## 2. Repo/HEAD And Starting State

Working repository:

`C:\o0c-preflight-auth-c551747`

Expected HEAD:

`52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`

Starting HEAD observed before authoring:

`52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`

Starting `git status --short`:

empty.

Required final repository state:

- HEAD unchanged at `52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`;
- no tracked modifications;
- nothing staged;
- exactly one task-attributable untracked candidate:
  `reports/longterm_o0c_recurrent_state_initialization_diagnostic_execution_authority_spec_candidate.md`;
- no temporary files inside the repository.

## 3. Authority Chain

Authority order used:

1. Current controller instruction for this task.
2. Primary frozen corrected-preflight execution authority: `52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`.
3. Frozen corrected implementation / future diagnostic execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.
4. Consumed validated run: `longterm-o0c-runtime-source-provenance-preflight-c551747-v3`.
5. Imported provenance audit: `C:\Users\Home1\.contramamba\imports\longterm-o0c-runtime-source-provenance-preflight-c551747-v3_c551747180ce_20260907_093530`.
6. `reports/longterm_o0c_runtime_source_provenance_corrected_preflight_execution_authority_spec_candidate.md`.
7. `scripts/preflight_longterm_o0c_runtime_source_provenance.py` at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.
8. `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` at `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.
9. Prior distribution-root diagnostic execution authority and interpretation reports, used only as workflow/provenance pattern.
10. `AGENTS.md` and applicable runbook constraints available in the repository.

No contradictory higher-priority authority was found during authoring.

## 4. Consumed V3 Provenance Facts

The consumed v3 run is preserved exactly as:

- run name: `longterm-o0c-runtime-source-provenance-preflight-c551747-v3`;
- execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`;
- command SHA256: `369bd49fda005c35d22894e0dca0a105472f72daa6eab55a19821c7b697c9db0`;
- command bytes: `4152`;
- command LF: `130`;
- command CR: `0`;
- command final LF: `false`;
- started UTC: `2026-09-07T00:33:07Z`;
- finished UTC: `2026-09-07T00:33:30Z`;
- exit code: `2`;
- status: `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`;
- blocker: `recurrent_state_initialization`;
- run log SHA256: `f475b6b1f025756d2abc8eceb813b65704f0b2a1d173c57bc6a8c085d5f371eb`;
- run meta SHA256: `112edde6121303cab2da65b62e679e03c990c9d4e7475378893b3eee66076b98`;
- handoff ZIP SHA256: `0e9afcd07e4f596497401a20b232d869f8238a78d56346e8f3597f9cc40463c9`;
- collector: `PASS, FILES_COLLECTED=0`;
- import: `PASS`;
- imported execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`.

Direct audit evidence read during authoring:

- `command.sh`: SHA256 `369bd49fda005c35d22894e0dca0a105472f72daa6eab55a19821c7b697c9db0`, bytes `4152`, LF `130`, CR `0`, final LF `false`;
- `run.log`: SHA256 `f475b6b1f025756d2abc8eceb813b65704f0b2a1d173c57bc6a8c085d5f371eb`;
- `run.meta`: SHA256 `112edde6121303cab2da65b62e679e03c990c9d4e7475378893b3eee66076b98`;
- `manifest.json`: records `file_count: 0` and `files: []`;
- `import.json`: records import schema `contramamba-local-import-v2`, copied files `0`, identical files `0`, and source ZIP SHA256 `0e9afcd07e4f596497401a20b232d869f8238a78d56346e8f3597f9cc40463c9`.

V3 is consumed and must not be rerun, reused, overwritten, deleted, aliased, or reinterpreted. `FILES_COLLECTED=0` is expected because the frozen corrected preflight blocked before canonical JSON publication.

Preserved separation:

- code correctness remains separate;
- execution result remains separate;
- provenance validity remains separate;
- scientific conclusion remains `NONE`.

## 5. Frozen Validator Predicate

The frozen `c551747180ce1e8fe4eed5f7aa5ab6294cd89948` source implements recurrent-state initialization proof in `_recurrent_proof_nodes`.

The current recurrent proof logic requires exactly one statement in the direct body of `MambaMixer.slow_forward` that:

- is `Assign` or `AnnAssign`;
- assigns the bare name `ssm_state`;
- contains a call whose terminal attribute/name is one of `new_zeros`, `zeros`, or `empty`.

Zero matching statements produces:

`BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED("recurrent_state_initialization")`

More than one matching statement produces:

`BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS("recurrent_state_initialization")`

This is recorded only as the frozen validator contract. It is not evidence that installed Transformers `5.0.0` is defective, and it is not evidence that the frozen validator is defective.

## 6. Static Hypothesis, Non-Conclusive

Static authoring hypothesis:

The installed `Transformers 5.0.0` `MambaMixer.slow_forward` source may initialize recurrent state through a syntactic shape that differs from the frozen validator's narrow direct-body assignment predicate.

This is non-conclusive. The future diagnostic must determine, from exact installed source evidence only, whether the blocker reflects:

- a validator false negative over a deterministic recurrent-state initialization shape;
- genuinely incompatible recurrent initialization semantics for the frozen O0c design;
- unresolved or ambiguous evidence.

No implementation defect, package defect, scientific conclusion, or corrective action is predeclared here.

## 7. Exact Future Diagnostic Scope

This candidate authorizes exactly one future diagnostic run named:

`longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`

Execution commit exactly:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

The diagnostic is read-only source inspection and must not:

- rerun the preflight;
- load or instantiate a model;
- load or instantiate a tokenizer;
- load a dataset;
- perform model forward/generation;
- train or evaluate;
- import optional kernels;
- initialize CUDA;
- install/uninstall/upgrade/downgrade packages;
- modify `site-packages`;
- modify repository files;
- create repository artifacts;
- repair or reinterpret the validator during execution.

Allowed inspection mechanisms:

- Python stdlib;
- `importlib.metadata`;
- `importlib.util` only where non-instantiating source-origin lookup is safe;
- `pathlib`;
- `hashlib`;
- `ast`;
- inspect-like analysis over raw source text, without importing the Mamba model module;
- filesystem metadata reads.

The future diagnostic may reimplement only the frozen predicate as a read-only observation. It must not weaken or modify that predicate.

## 8. Exact Run Name And Execution Commit

Reserved run name:

`longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`

Exact execution commit:

`c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

The run name is reserved by this candidate only after independent verification and controller activation. If the run name is found already registered or consumed incompatibly, the workflow must `BLOCK` and must not silently rename.

## 9. Runtime/CPU/GPU Contract

Runtime:

- Python `3.12.13`;
- NumPy `2.0.2`;
- torch `2.10.0+cpu`;
- Transformers `5.0.0`;
- CPU only;
- GPU OFF.

The future command must verify expected runtime strings before source interpretation. Runtime version checks must use distribution metadata and platform identity, not package imports that would initialize model stacks or optional kernels.

GPU exposure must fail closed. The command must not import torch to discover GPU state; it must use non-mutating filesystem, command-availability, and environment checks.

## 10. Required Diagnostic Evidence

The future diagnostic must deterministically observe and print to stdout at least:

1. Runtime/source identity:
   - Python version;
   - NumPy, torch, and Transformers distribution versions;
   - `sys.executable`;
   - resolved Transformers distribution root;
   - resolved `modeling_mamba.py` path;
   - `modeling_mamba.py` SHA256, bytes, LF, CR, and final-LF.
2. Exact `MambaMixer.slow_forward` structure:
   - function existence and uniqueness;
   - start/end line;
   - deterministic line-numbered source excerpt;
   - argument names;
   - ordered direct-body statement types;
   - AST facts for every direct-body assignment/annassign involving `ssm_state`;
   - call paths within each candidate;
   - number of statements satisfying the frozen validator predicate.
3. All recurrent-state-related occurrences within `MambaMixer.slow_forward` and directly relevant helper functions:
   - every load/store of `ssm_state`;
   - assignments to `ssm_state`;
   - calls producing or transforming recurrent state;
   - cache-derived recurrent-state accesses;
   - helper calls whose return value becomes recurrent state;
   - branch conditions affecting recurrent initialization;
   - initialization before/inside token loops;
   - whether initialization is direct, helper-mediated, cache-mediated, conditional, tuple-unpacked, annotated, or otherwise structurally different from the frozen validator pattern.
4. Sequential update evidence, without executing the model:
   - locate the sequential loop used by the CPU slow path, if present;
   - identify recurrent-state update statements;
   - identify how the initial recurrent state reaches the first update;
   - identify any cache/non-cache split;
   - record whether the source statically establishes a deterministic initial recurrent state or whether that cannot be established.
5. Frozen-validator replay:
   - candidate count;
   - matching statement spans;
   - why each near-match does or does not satisfy the frozen predicate.

The diagnostic must print evidence only. Final classification may be assigned only under the criteria in Section 11.

## 11. Classification Criteria

The future diagnostic evidence must support exactly one of the following classifications.

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`

Use only if exact installed source statically provides a deterministic recurrent state initialization compatible with the relevant O0c state semantics, but the frozen validator misses it solely because the syntactic initialization shape is outside its narrow predicate.

`ACTUAL_RECURRENT_STATE_INITIALIZATION_SEMANTICS_INCOMPATIBLE`

Use only if exact installed source demonstrates recurrent initialization semantics that are genuinely incompatible with the frozen O0c design, not merely syntactically different.

`BLOCKED_RECURRENT_STATE_INITIALIZATION_ROOT_CAUSE_UNRESOLVED`

Use if the source evidence is insufficient, ambiguous, helper-mediated in a way that cannot be statically proven, contradictory, or otherwise cannot justify either classification above.

This candidate does not predeclare the first or second outcome.

## 12. Exact Future Command

The exact future Kaggle bash command is the command between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`. This authoring task records it but does not execute it.

BEGIN_EXACT_COMMAND
```bash
set -euo pipefail

RUN_NAME="longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1"
EXPECTED_COMMIT="c551747180ce1e8fe4eed5f7aa5ab6294cd89948"

actual_commit="$(git rev-parse HEAD)"
if [ "${actual_commit}" != "${EXPECTED_COMMIT}" ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_COMMIT_MISMATCH\",\"expected\":\"${EXPECTED_COMMIT}\",\"actual\":\"${actual_commit}\"}"
  exit 1
fi

status_porcelain="$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)"
if [ -n "${status_porcelain}" ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_REPOSITORY_STATE_DIRTY\",\"details\":$(python -B -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines(), sort_keys=True))' <<< "${status_porcelain}")}"
  exit 1
fi

if [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ] || [ -e /proc/driver/nvidia/version ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_GPU_EXPOSED\",\"probe\":\"nvidia_device_or_driver_path\"}"
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  printf '%s\n' "{\"status\":\"BLOCKED_GPU_EXPOSED\",\"probe\":\"nvidia-smi\"}"
  exit 1
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_GPU_EXPOSED\",\"probe\":\"CUDA_VISIBLE_DEVICES\",\"value\":\"${CUDA_VISIBLE_DEVICES}\"}"
  exit 1
fi

python -B - <<'PY'
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

RUN_NAME = "longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1"
EXPECTED_COMMIT = "c551747180ce1e8fe4eed5f7aa5ab6294cd89948"
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cpu",
    "transformers": "5.0.0",
}
STATE_TERMS = ("ssm_state", "cache", "cache_params", "seqlen_offset", "conv_state")
INIT_TERMINALS = {"new_zeros", "zeros", "empty"}


def fail(status, **extra):
    payload = {"diagnostic_name": RUN_NAME, "execution_commit": EXPECTED_COMMIT, "status": status}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True, indent=2))
    raise SystemExit(2)


def runtime_versions():
    try:
        return {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
        }
    except Exception as exc:
        fail("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", error=f"{type(exc).__name__}: {exc}")


runtime = runtime_versions()
mismatches = {key: {"expected": EXPECTED_RUNTIME[key], "actual": runtime.get(key)} for key in EXPECTED_RUNTIME if runtime.get(key) != EXPECTED_RUNTIME[key]}
if mismatches:
    fail("BLOCKED_RUNTIME_VERSION_MISMATCH", mismatches=mismatches, sys_executable=sys.executable)

try:
    dist = importlib.metadata.distribution("transformers")
except Exception as exc:
    fail("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", package="transformers", error=f"{type(exc).__name__}: {exc}")

try:
    files = list(dist.files or [])
except Exception as exc:
    fail("BLOCKED_DISTRIBUTION_METADATA_UNAVAILABLE", error=f"{type(exc).__name__}: {exc}")

root_counter = Counter()
root_samples = {}
for file in sorted(files, key=lambda item: str(item).replace("\\", "/")):
    parts = tuple(getattr(file, "parts", Path(str(file)).parts))
    if not parts or parts[0] != "transformers":
        continue
    try:
        located = Path(dist.locate_file(file)).resolve(strict=True)
        root = located.parents[len(parts) - 2]
    except Exception as exc:
        fail("BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED", relative_path=str(file), error=f"{type(exc).__name__}: {exc}")
    root_text = str(root)
    root_counter[root_text] += 1
    root_samples.setdefault(root_text, str(file).replace("\\", "/"))

if len(root_counter) != 1:
    fail(
        "BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS",
        distinct_roots=[{"root": root, "entry_count": root_counter[root], "first_sample_entry": root_samples[root]} for root in sorted(root_counter)],
    )
if root_counter:
    transformers_root = Path(next(iter(root_counter))).resolve(strict=True)
else:
    try:
        transformers_root = Path(dist.locate_file("transformers")).resolve(strict=True)
    except Exception as exc:
        fail("BLOCKED_SOURCE_FILE_UNRESOLVED", module="transformers", error=f"{type(exc).__name__}: {exc}")

mamba_path = (transformers_root / "models" / "mamba" / "modeling_mamba.py").resolve(strict=True)
if not mamba_path.is_file():
    fail("BLOCKED_SOURCE_FILE_UNRESOLVED", module="transformers.models.mamba.modeling_mamba", path=str(mamba_path))

try:
    data = mamba_path.read_bytes()
except OSError as exc:
    fail("BLOCKED_SOURCE_HASH_UNAVAILABLE", path=str(mamba_path), error=f"{type(exc).__name__}: {exc}")
try:
    text = data.decode("utf-8")
    tree = ast.parse(text)
except Exception as exc:
    fail("BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE", path=str(mamba_path), error=f"{type(exc).__name__}: {exc}")

lines = text.splitlines()


def node_span(node):
    return {"start_line": getattr(node, "lineno", None), "end_line": getattr(node, "end_lineno", getattr(node, "lineno", None))}


def excerpt(start, end):
    return [{"line": number, "text": lines[number - 1]} for number in range(start, end + 1)]


def attr_path(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return attr_path(node.value) + [node.attr]
    if isinstance(node, ast.Subscript):
        return attr_path(node.value) + ["[]"]
    return []


def call_path(node):
    return attr_path(node.func) if isinstance(node, ast.Call) else []


def target_paths(node):
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        return [attr_path(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for item in node.elts:
            out.extend(target_paths(item))
        return out
    return []


def assigned_paths(stmt):
    if isinstance(stmt, ast.Assign):
        out = []
        for target in stmt.targets:
            out.extend(target_paths(target))
        return out
    if isinstance(stmt, ast.AnnAssign):
        return target_paths(stmt.target)
    if isinstance(stmt, ast.AugAssign):
        return target_paths(stmt.target)
    return []


def assigns_name(stmt, name):
    return any(path == [name] for path in assigned_paths(stmt))


def all_call_paths(node):
    return [call_path(child) for child in ast.walk(node) if isinstance(child, ast.Call)]


def calls_terminal(node, terminals):
    return any(path and path[-1] in terminals for path in all_call_paths(node))


def names_in(node, wanted):
    return [
        {"name": child.id, "ctx": type(child.ctx).__name__, **node_span(child)}
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id in wanted
    ]


def stmt_source(stmt):
    return "\n".join(lines[stmt.lineno - 1 : getattr(stmt, "end_lineno", stmt.lineno)])


def assignment_record(stmt):
    terminals = [path[-1] for path in all_call_paths(stmt) if path]
    satisfies = isinstance(stmt, (ast.Assign, ast.AnnAssign)) and assigns_name(stmt, "ssm_state") and any(term in INIT_TERMINALS for term in terminals)
    reasons = []
    if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        reasons.append("not Assign or AnnAssign")
    if not assigns_name(stmt, "ssm_state"):
        reasons.append("does not assign bare name ssm_state")
    if not any(term in INIT_TERMINALS for term in terminals):
        reasons.append("contains no call ending in new_zeros, zeros, or empty")
    if satisfies:
        reasons.append("satisfies frozen predicate")
    return {
        **node_span(stmt),
        "statement_type": type(stmt).__name__,
        "assigned_paths": [".".join(path) for path in assigned_paths(stmt)],
        "call_paths": [".".join(path) for path in all_call_paths(stmt)],
        "source": stmt_source(stmt),
        "satisfies_frozen_recurrent_initialization_predicate": satisfies,
        "predicate_reasons": reasons,
    }


def function_qualnames(root):
    found = []
    def visit(node, stack):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit(child, stack + [node.name])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join(stack + [node.name])
            found.append((qualname, node))
            for child in node.body:
                visit(child, stack + [node.name])
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, stack)
    visit(root, [])
    return found

mixer_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MambaMixer"]
if len(mixer_classes) != 1:
    fail("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS" if mixer_classes else "BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", symbol="MambaMixer", count=len(mixer_classes))

slow_matches = [node for name, node in function_qualnames(tree) if name == "MambaMixer.slow_forward"]
if len(slow_matches) != 1:
    fail("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS" if slow_matches else "BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", symbol="MambaMixer.slow_forward", count=len(slow_matches))
slow = slow_matches[0]

init_candidates = [
    stmt for stmt in slow.body
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and assigns_name(stmt, "ssm_state") and calls_terminal(stmt, INIT_TERMINALS)
]
near_matches = []
for stmt in slow.body:
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        terminals = [path[-1] for path in all_call_paths(stmt) if path]
        if assigns_name(stmt, "ssm_state") or any(term in INIT_TERMINALS for term in terminals) or names_in(stmt, {"ssm_state"}):
            near_matches.append(assignment_record(stmt))

state_occurrences = []
for node in ast.walk(slow):
    if isinstance(node, ast.Name) and node.id == "ssm_state":
        state_occurrences.append({"kind": "name", "ctx": type(node.ctx).__name__, **node_span(node)})
    elif isinstance(node, ast.Attribute) and any(term in node.attr for term in STATE_TERMS):
        state_occurrences.append({"kind": "attribute", "path": ".".join(attr_path(node)), **node_span(node)})

branch_conditions = []
loops = []
updates = []
for node in ast.walk(slow):
    if isinstance(node, ast.If) and any(names_in(node.test, set(STATE_TERMS)) or term in stmt_source(node.test) for term in STATE_TERMS):
        branch_conditions.append({**node_span(node), "test": stmt_source(node.test)})
    if isinstance(node, (ast.For, ast.While)):
        body_assigns = [assignment_record(stmt) for stmt in node.body if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and (assigns_name(stmt, "ssm_state") or names_in(stmt, {"ssm_state"}))]
        loops.append({**node_span(node), "loop_type": type(node).__name__, "header": lines[node.lineno - 1], "ssm_state_assignments_or_reads_in_body": body_assigns})
        updates.extend(record for record in body_assigns if any(path == "ssm_state" for path in record["assigned_paths"]))

call_assignments = []
for stmt in ast.walk(slow):
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and names_in(stmt, {"ssm_state"}):
        call_assignments.append(assignment_record(stmt))

called_names = set()
for path in all_call_paths(slow):
    if path:
        called_names.add(path[-1])
helper_records = []
for qualname, func in function_qualnames(tree):
    short = qualname.split(".")[-1]
    if func is slow or short not in called_names:
        continue
    func_text = "\n".join(lines[func.lineno - 1 : getattr(func, "end_lineno", func.lineno)])
    if any(term in func_text for term in STATE_TERMS) or any(term in short.lower() for term in ("state", "cache", "scan")):
        helper_records.append({
            "qualname": qualname,
            **node_span(func),
            "argument_names": [arg.arg for arg in func.args.args],
            "ssm_state_occurrences": names_in(func, {"ssm_state"}),
            "excerpt": excerpt(func.lineno, getattr(func, "end_lineno", func.lineno)),
        })

first_loop_start = min([item["start_line"] for item in loops], default=None)
init_before_first_loop = [item for item in near_matches if item["satisfies_frozen_recurrent_initialization_predicate"] and (first_loop_start is None or item["start_line"] < first_loop_start)]

payload = {
    "diagnostic_name": RUN_NAME,
    "execution_commit": EXPECTED_COMMIT,
    "status": "DIAGNOSTIC_OBSERVATION_ONLY",
    "classification_note": "Evidence only; final classification must be assigned separately under the authority criteria.",
    "runtime_source_identity": {
        "runtime": runtime,
        "sys_executable": sys.executable,
        "transformers_distribution_root": str(transformers_root),
        "modeling_mamba_path": str(mamba_path),
        "modeling_mamba_sha256": hashlib.sha256(data).hexdigest(),
        "modeling_mamba_bytes": len(data),
        "modeling_mamba_lf": data.count(b"\x0a"),
        "modeling_mamba_cr": data.count(b"\x0d"),
        "modeling_mamba_final_lf": data.endswith(b"\x0a"),
    },
    "slow_forward_structure": {
        "function_exists_unique": True,
        **node_span(slow),
        "argument_names": [arg.arg for arg in slow.args.args],
        "direct_body_statement_types": [{"index": index, "type": type(stmt).__name__, **node_span(stmt)} for index, stmt in enumerate(slow.body)],
        "line_numbered_excerpt": excerpt(slow.lineno, getattr(slow, "end_lineno", slow.lineno)),
    },
    "recurrent_state_occurrences": {
        "within_slow_forward": state_occurrences,
        "assignments_annassign_augassign_involving_ssm_state": call_assignments,
        "cache_related_occurrences_text_scan": [
            {"line": number, "text": line}
            for number, line in enumerate(lines[slow.lineno - 1 : getattr(slow, "end_lineno", slow.lineno)], start=slow.lineno)
            if any(term in line for term in STATE_TERMS)
        ],
        "branch_conditions_affecting_state_or_cache": branch_conditions,
        "directly_relevant_helper_functions": helper_records,
    },
    "sequential_update_evidence": {
        "sequential_loops_in_slow_forward": loops,
        "recurrent_state_update_statements": updates,
        "first_loop_start_line": first_loop_start,
        "initial_state_reaches_first_update_static_observation": "DIRECT_FROZEN_PATTERN_BEFORE_FIRST_LOOP" if init_before_first_loop and updates else "NOT_ESTABLISHED_BY_FROZEN_PATTERN",
        "cache_non_cache_split_text_observations": [
            {"line": number, "text": line}
            for number, line in enumerate(lines[slow.lineno - 1 : getattr(slow, "end_lineno", slow.lineno)], start=slow.lineno)
            if "cache" in line or "seqlen_offset" in line
        ],
    },
    "frozen_validator_replay": {
        "predicate": "direct MambaMixer.slow_forward body statement is Assign or AnnAssign, assigns bare name ssm_state, and contains a call ending in new_zeros, zeros, or empty",
        "candidate_count": len(init_candidates),
        "matching_statement_spans": [node_span(stmt) for stmt in init_candidates],
        "near_matches": near_matches,
        "zero_match_frozen_status": "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED(recurrent_state_initialization)",
        "multiple_match_frozen_status": "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS(recurrent_state_initialization)",
    },
}
print(json.dumps(payload, sort_keys=True, indent=2))
PY
```
END_EXACT_COMMAND

Normative command identity for `cm run save` is the exact command text inside the fenced block above after removing only the Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF, consistent with the repository's established `BEGIN_EXACT_COMMAND` / `END_EXACT_COMMAND` convention.

The command verifies:

- exact HEAD `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`;
- clean repository state;
- GPU not exposed;
- exact runtime strings before source interpretation;
- read-only source diagnostic only;
- deterministic JSON stdout;
- no repository output file creation;
- fail-closed nonzero errors.

## 13. Command SHA256/Bytes/LF/CR/Final-LF

Exact command SHA256:

`0795668fa8e7770bed5dabde1991d0447f5c42d30d6062f499ff0033ca016247`

Exact command bytes:

`15564`

Exact command LF:

`364`

Exact command CR:

`0`

Exact command final LF:

`false`

The command identity was computed over the Section 12 command bytes after removing only the opening and closing Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF.

## 14. Run-Name Collision Check

Collision search performed during authoring:

- repository text search for `longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`;
- local `.contramamba` import/registry text search for `longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`;
- local `.contramamba` text search for `recurrent-state-initialization-diagnostic`.

Observed collision state:

`NO_COLLISION_OBSERVED`

If independent verification finds this run name already registered or consumed incompatibly:

`BLOCK`

Do not silently rename.

## 15. Scientific/Non-Execution Boundaries

Scientific conclusion:

`NONE`

This candidate does not authorize:

- scientific O0c execution;
- O0c recurrent-state instrumentation implementation;
- model/tokenizer/dataset loading;
- model forward/generation;
- training;
- evaluation;
- package mutation;
- repository mutation beyond this single candidate file;
- validator repair;
- preflight rerun;
- Kaggle execution during this authoring task.

Code correctness, execution result, provenance validity, and scientific conclusion remain separate.

## 16. Candidate Path And Raw Identity

Candidate path:

`reports/longterm_o0c_recurrent_state_initialization_diagnostic_execution_authority_spec_candidate.md`

Candidate raw identity must be recomputed after final authoring edits:

- SHA256;
- bytes;
- LF count;
- CR count;
- final-LF fact.

The final raw identity is reported by the authoring agent after validation rather than embedded here, to avoid self-referential hash churn.

## 17. Git Diff Check

Required authoring validation:

```powershell
git diff --check
```

Expected result:

PASS with no output.

## 18. Tracked/Staged/Untracked State

Required authoring validation:

```powershell
git diff --name-status
git diff --cached --name-status
git status --short
```

Expected result:

- no tracked modifications;
- no staged changes;
- exactly one task-attributable untracked candidate:
  `?? reports/longterm_o0c_recurrent_state_initialization_diagnostic_execution_authority_spec_candidate.md`.

## 19. Explicit No-Execution Attestation

NO DIAGNOSTIC EXECUTION.

NO PREFLIGHT RERUN.

NO KAGGLE.

NO `cm run save`.

NO `cm run`.

NO `cm collect`.

NO `cm import`.

NO MODEL LOADING.

NO TOKENIZER LOADING.

NO DATASET LOADING.

NO MODEL FORWARD.

NO GENERATION.

NO TRAINING.

NO EVALUATION.

NO PACKAGE INSTALL.

NO PACKAGE UNINSTALL.

NO PACKAGE UPGRADE OR DOWNGRADE.

NO OPTIONAL KERNEL IMPORT.

NO CUDA INITIALIZATION.

NO SITE-PACKAGES MUTATION.

NO IMPLEMENTATION MODIFICATION.

NO TEST MODIFICATION.

NO PRIOR REPORT MODIFICATION.

NO STAGING.

NO COMMIT.

NO PUSH.

## 20. Discrepancies/Blockers

No blocking discrepancy was found during authoring.

The imported v3 run blocked with `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED("recurrent_state_initialization")`. That result is consumed and valid as a blocked preflight result. It must not be upgraded into a scientific or implementation conclusion without the future diagnostic evidence authorized here.

## 21. Exact Next Authorized Action

Independent verification of this candidate's exact bytes, authority sufficiency, consumed-v3 evidence, frozen-validator predicate, command identity, run-name collision state, and final git state.

Only after independent verification and controller activation may the future diagnostic be registered/executed under:

`longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`
