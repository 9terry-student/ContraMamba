#!/usr/bin/env python
"""Deterministic O0c runtime-source provenance preflight.

This script is provenance infrastructure only. It resolves package/source
identity, computes raw-byte source facts, and statically classifies source
structure without loading models, tokenizers, datasets, or optional kernels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "o0c_runtime_source_provenance_preflight_v1"

PASS_STATUS = "PASS_SOURCE_IDENTITY_FROZEN"

STATUSES = {
    PASS_STATUS,
    "BLOCKED_RUNTIME_VERSION_UNAVAILABLE",
    "BLOCKED_RUNTIME_VERSION_MISMATCH",
    "BLOCKED_SOURCE_FILE_UNRESOLVED",
    "BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED",
    "BLOCKED_SOURCE_HASH_UNAVAILABLE",
    "BLOCKED_TRANSFORMERS_SOURCE_SHADOWING",
    "BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS",
    "BLOCKED_REQUIRED_SYMBOL_UNRESOLVED",
    "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS",
    "BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE",
    "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED",
    "BLOCKED_BACKEND_PATH_UNRESOLVED",
    "BLOCKED_O0C_INDEXING_INCOMPATIBLE",
    "BLOCKED_ARTIFACT_SERIALIZATION_NONDETERMINISTIC",
    "BLOCKED_OUTPUT_COLLISION",
    "BLOCKED_FORBIDDEN_MODEL_TOKENIZER_INVOCATION",
    "BLOCKED_FORBIDDEN_PACKAGE_MUTATION",
    "BLOCKED_IMPLEMENTATION_SCOPE_WOULD_WIDEN",
}

RECURRENT_CLASSIFICATIONS = {
    "SOURCE_SUPPORTS_O0C_CONVENTION",
    "SOURCE_REQUIRES_IMPLEMENTATION_REVIEW",
    "SOURCE_INCOMPATIBLE_WITH_FROZEN_O0C_DESIGN",
}

BACKEND_CLASSIFICATIONS = {
    "BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN",
    "BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN",
    "BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE",
    "BACKEND_STATICALLY_UNRESOLVED",
    "BACKEND_INCOMPATIBLE_WITH_O0C",
}

SYMBOL_KEYS = (
    "mixer_forward_dispatch",
    "sequential_slow_path",
    "recurrent_state_initialization",
    "recurrent_state_update",
    "convolution_cache_initialization_update",
    "cache_recurrent_state_storage",
    "hidden_state_output_path",
    "backend_kernel_selection",
)

SOURCE_KEYS = {"mamba": "transformers.models.mamba.modeling_mamba", "cache": "transformers.cache_utils"}


class PreflightBlocked(Exception):
    def __init__(self, status: str, note: str):
        if status not in STATUSES:
            raise ValueError(f"unknown preflight status: {status}")
        self.status = status
        self.note = note
        super().__init__(f"{status}: {note}")


@dataclass(frozen=True)
class SourceFacts:
    module: str
    path: Path
    sha256: str
    bytes: int
    lf_count: int
    cr_count: int
    final_lf: bool
    text: str


@dataclass(frozen=True)
class SourceResolution:
    distribution_root: Path
    import_root: Path
    mamba_source: Path
    cache_source: Path
    distribution_version: str


def canonical_path(path: Path | str) -> Path:
    try:
        return Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightBlocked(
            "BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED",
            f"could not canonicalize path: {path}",
        ) from exc


def _case_key(path: Path) -> str:
    return os.path.normcase(str(path)) if os.name == "nt" else str(path)


def same_path(left: Path, right: Path) -> bool:
    return _case_key(left) == _case_key(right)


def is_descendant_or_equal(source: Path, root: Path) -> bool:
    if same_path(source, root):
        return True
    if os.name == "nt":
        source_parts = tuple(os.path.normcase(part) for part in source.parts)
        root_parts = tuple(os.path.normcase(part) for part in root.parts)
        return len(source_parts) >= len(root_parts) and source_parts[: len(root_parts)] == root_parts
    try:
        source.relative_to(root)
        return True
    except ValueError:
        return False


def runtime_versions(
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    python_version: Callable[[], str] = platform.python_version,
) -> dict[str, str]:
    try:
        return {
            "python": str(python_version()),
            "numpy": str(version_lookup("numpy")),
            "torch": str(version_lookup("torch")),
            "transformers": str(version_lookup("transformers")),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreflightBlocked("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", str(exc)) from exc
    except Exception as exc:
        raise PreflightBlocked("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", str(exc)) from exc


def check_runtime(expected: Mapping[str, str], actual: Mapping[str, str]) -> None:
    missing = [key for key in ("python", "numpy", "torch", "transformers") if key not in actual]
    if missing:
        raise PreflightBlocked("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", ",".join(missing))
    mismatches = [key for key in ("python", "numpy", "torch", "transformers") if str(actual[key]) != expected[key]]
    if mismatches:
        raise PreflightBlocked("BLOCKED_RUNTIME_VERSION_MISMATCH", ",".join(mismatches))


def _spec_origin(spec: object, module: str) -> Path:
    origin = getattr(spec, "origin", None)
    if not origin or origin in {"built-in", "frozen", "namespace"}:
        locations = getattr(spec, "submodule_search_locations", None)
        if locations:
            location = next(iter(locations), None)
            if location:
                return Path(location) / "__init__.py"
        raise PreflightBlocked("BLOCKED_SOURCE_FILE_UNRESOLVED", module)
    return Path(origin)


def _transformers_import_root(spec: object) -> Path:
    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen", "namespace"}:
        return Path(origin).parent
    locations = getattr(spec, "submodule_search_locations", None)
    if not locations:
        raise PreflightBlocked("BLOCKED_SOURCE_FILE_UNRESOLVED", "transformers")
    first = next(iter(locations), None)
    if not first:
        raise PreflightBlocked("BLOCKED_SOURCE_FILE_UNRESOLVED", "transformers")
    return Path(first)


def _distribution_root(distribution: object) -> Path:
    roots: set[Path] = set()
    locate_file = getattr(distribution, "locate_file")
    files = getattr(distribution, "files", None) or []
    for file in files:
        parts = getattr(file, "parts", Path(str(file)).parts)
        if parts and parts[0] == "transformers":
            located = canonical_path(Path(locate_file(file)))
            root = located.parent if located.name == "__init__.py" else located.parents[len(parts) - 2]
            roots.add(root)
    if not roots:
        roots.add(canonical_path(Path(locate_file("transformers"))))
    canonical_roots = {_case_key(root): root for root in roots}
    if len(canonical_roots) != 1:
        raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS", "distribution roots")
    return next(iter(canonical_roots.values()))


def resolve_transformers_sources(
    repo_root: Path,
    distribution_provider: Callable[[str], object] = importlib.metadata.distribution,
    spec_provider: Callable[[str], object | None] = importlib.util.find_spec,
) -> SourceResolution:
    repo = canonical_path(repo_root)
    try:
        distribution = distribution_provider("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreflightBlocked("BLOCKED_RUNTIME_VERSION_UNAVAILABLE", "transformers") from exc
    dist_root = _distribution_root(distribution)
    dist_version = str(getattr(distribution, "version", importlib.metadata.version("transformers")))

    specs = {name: spec_provider(name) for name in ("transformers", *SOURCE_KEYS.values())}
    for name, spec in specs.items():
        if spec is None:
            raise PreflightBlocked("BLOCKED_SOURCE_FILE_UNRESOLVED", name)

    import_root = canonical_path(_transformers_import_root(specs["transformers"]))
    mamba_source = canonical_path(_spec_origin(specs[SOURCE_KEYS["mamba"]], SOURCE_KEYS["mamba"]))
    cache_source = canonical_path(_spec_origin(specs[SOURCE_KEYS["cache"]], SOURCE_KEYS["cache"]))

    roots = {key: value for key, value in {"distribution": dist_root, "import": import_root}.items()}
    if not (is_descendant_or_equal(import_root, dist_root) or is_descendant_or_equal(dist_root, import_root)):
        raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_SHADOWING", "import root outside distribution root")

    for label, path in (("import_root", import_root), ("mamba", mamba_source), ("cache", cache_source)):
        if is_descendant_or_equal(path, repo):
            raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_SHADOWING", f"{label} inside repository")
        if not is_descendant_or_equal(path, import_root):
            raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_SHADOWING", f"{label} outside import root")
        if not is_descendant_or_equal(path, dist_root):
            raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_SHADOWING", f"{label} outside distribution root")

    if len({_case_key(path) for path in roots.values()}) > 1:
        if not is_descendant_or_equal(import_root, dist_root):
            raise PreflightBlocked("BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS", "unreconciled roots")

    return SourceResolution(dist_root, import_root, mamba_source, cache_source, dist_version)


def raw_source_identity(path: Path, module: str) -> SourceFacts:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PreflightBlocked("BLOCKED_SOURCE_HASH_UNAVAILABLE", str(path)) from exc
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8")
        ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise PreflightBlocked("BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE", str(path)) from exc
    return SourceFacts(
        module=module,
        path=path,
        sha256=sha256,
        bytes=len(data),
        lf_count=data.count(b"\x0a"),
        cr_count=data.count(b"\x0d"),
        final_lf=bool(data.endswith(b"\x0a")),
        text=text,
    )


def _qualname_stack(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    out: list[tuple[str, ast.AST]] = []

    def visit(node: ast.AST, stack: list[str]) -> None:
        if isinstance(node, ast.ClassDef):
            stack.append(node.name)
            for child in node.body:
                visit(child, stack)
            stack.pop()
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join((*stack, node.name))
            out.append((qualname, node))
            stack.append(node.name)
            for child in node.body:
                visit(child, stack)
            stack.pop()
            return
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    visit(tree, [])
    return out


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        return (*_attribute_path(node.value), node.attr)
    return ()


def _call_path(node: ast.AST) -> tuple[str, ...]:
    return _attribute_path(node.func) if isinstance(node, ast.Call) else ()


def _target_paths(node: ast.AST) -> list[tuple[str, ...]]:
    if isinstance(node, (ast.Name, ast.Attribute)):
        return [_attribute_path(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        paths: list[tuple[str, ...]] = []
        for elt in node.elts:
            paths.extend(_target_paths(elt))
        return paths
    if isinstance(node, ast.Subscript):
        return [_attribute_path(node.value)]
    return []


def _assigned_paths(node: ast.AST) -> list[tuple[str, ...]]:
    if isinstance(node, ast.Assign):
        paths: list[tuple[str, ...]] = []
        for target in node.targets:
            paths.extend(_target_paths(target))
        return paths
    if isinstance(node, ast.AnnAssign):
        return _target_paths(node.target)
    if isinstance(node, ast.AugAssign):
        return _target_paths(node.target)
    return []


def _assigns_name(node: ast.AST, name: str) -> bool:
    return any(path == (name,) for path in _assigned_paths(node))


def _loads_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id == name for child in ast.walk(node))


def _calls_attr(node: ast.AST, attrs: set[str]) -> bool:
    return any(isinstance(child, ast.Call) and _call_path(child)[-1:] and _call_path(child)[-1] in attrs for child in ast.walk(node))


def _branch_calls_slow_forward(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _call_path(child)[-2:] == ("self", "slow_forward"):
            return True
    return False


def _all_statements_raise(statements: Sequence[ast.stmt]) -> bool:
    return bool(statements) and all(isinstance(stmt, ast.Raise) for stmt in statements)


def _is_cpu_device_test(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    if not isinstance(node.ops[0], ast.Eq):
        return False
    left = _attribute_path(node.left)
    comparator = node.comparators[0]
    return left[-2:] == ("device", "type") and isinstance(comparator, ast.Constant) and comparator.value == "cpu"


def _find_unique_function(tree: ast.AST, qualname: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    matches = [node for candidate, node in _qualname_stack(tree) if candidate == qualname]
    if len(matches) > 1:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS", qualname)
    if not matches:
        return None
    node = matches[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return node


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in node.elts:
            names.update(_target_names(elt))
        return names
    return set()


def _loop_target_names(loop: ast.For) -> set[str]:
    return _target_names(loop.target)


def _is_range_iteration(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_path(node)[-1:] == ("range",)


def _contains_index_name(node: ast.AST, index_names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in index_names for child in ast.walk(node))


def _expr_dependencies(node: ast.AST | None, env: Mapping[str, set[str]], index_names: set[str]) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return set(env.get(node.id, set()))
    if isinstance(node, ast.Constant):
        return set()
    if isinstance(node, ast.Subscript):
        deps = _expr_dependencies(node.value, env, index_names)
        deps.update(_expr_dependencies(node.slice, env, index_names))
        if _contains_index_name(node.slice, index_names):
            deps.add("current")
        return deps
    if isinstance(node, ast.Call):
        deps = _expr_dependencies(node.func, env, index_names)
        for arg in node.args:
            deps.update(_expr_dependencies(arg, env, index_names))
        for keyword in node.keywords:
            deps.update(_expr_dependencies(keyword.value, env, index_names))
        return deps
    if isinstance(node, ast.Attribute):
        return _expr_dependencies(node.value, env, index_names)
    if isinstance(node, ast.BinOp):
        return _expr_dependencies(node.left, env, index_names) | _expr_dependencies(node.right, env, index_names)
    if isinstance(node, ast.UnaryOp):
        return _expr_dependencies(node.operand, env, index_names)
    if isinstance(node, ast.BoolOp):
        deps: set[str] = set()
        for value in node.values:
            deps.update(_expr_dependencies(value, env, index_names))
        return deps
    if isinstance(node, ast.Compare):
        deps = _expr_dependencies(node.left, env, index_names)
        for comparator in node.comparators:
            deps.update(_expr_dependencies(comparator, env, index_names))
        return deps
    if isinstance(node, ast.IfExp):
        return (
            _expr_dependencies(node.test, env, index_names)
            | _expr_dependencies(node.body, env, index_names)
            | _expr_dependencies(node.orelse, env, index_names)
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        deps: set[str] = set()
        for elt in node.elts:
            deps.update(_expr_dependencies(elt, env, index_names))
        return deps
    if isinstance(node, ast.Dict):
        deps: set[str] = set()
        for key in node.keys:
            deps.update(_expr_dependencies(key, env, index_names))
        for value in node.values:
            deps.update(_expr_dependencies(value, env, index_names))
        return deps
    return set()


def _assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    if isinstance(node, ast.AugAssign):
        return ast.BinOp(left=node.target, op=node.op, right=node.value)
    return None


def _bind_assignment_dependencies(stmt: ast.AST, env: dict[str, set[str]], index_names: set[str]) -> set[str]:
    deps = _expr_dependencies(_assignment_value(stmt), env, index_names)
    for path in _assigned_paths(stmt):
        if len(path) == 1:
            env[path[0]] = set(deps)
    return deps


def _loop_body_update_and_readout(loop: ast.For) -> tuple[ast.AST | None, ast.AST | None]:
    env: dict[str, set[str]] = {"ssm_state": {"prior"}}
    loop_targets = _loop_target_names(loop)
    index_names = loop_targets if _is_range_iteration(loop.iter) else set()
    if not _is_range_iteration(loop.iter):
        for name in loop_targets:
            env[name] = {"current"}
    updates: list[ast.AST] = []
    readout: ast.AST | None = None
    for stmt in loop.body:
        if updates and isinstance(stmt, (ast.Assign, ast.AnnAssign)) and not _assigns_name(stmt, "ssm_state") and _loads_name(stmt, "ssm_state"):
            readout = stmt
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            deps = _bind_assignment_dependencies(stmt, env, index_names)
            if _assigns_name(stmt, "ssm_state") and {"prior", "current"} <= deps:
                updates.append(stmt)
            continue
        if updates:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                path = _call_path(stmt.value)
                if path[-1:] == ("append",) and any(_loads_name(arg, "ssm_state") for arg in stmt.value.args):
                    readout = stmt
                    break
    if len(updates) > 1:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS", "recurrent_state_update")
    return (updates[0] if updates else None), readout


def _recurrent_proof_nodes(tree: ast.AST) -> tuple[ast.AST, ast.AST, ast.For, ast.AST, ast.AST]:
    slow = _find_unique_function(tree, "MambaMixer.slow_forward")
    if slow is None:
        raise PreflightBlocked("BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED", "missing MambaMixer.slow_forward")

    init_nodes = [
        stmt
        for stmt in slow.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and _assigns_name(stmt, "ssm_state")
        and _calls_attr(stmt, {"new_zeros", "zeros", "empty"})
    ]
    if len(init_nodes) != 1:
        status = "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS" if len(init_nodes) > 1 else "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"
        raise PreflightBlocked(status, "recurrent_state_initialization")

    loop_candidates: list[tuple[ast.For, ast.AST, ast.AST]] = []
    for stmt in slow.body:
        if isinstance(stmt, ast.For):
            update, readout = _loop_body_update_and_readout(stmt)
            if update is not None and readout is not None:
                loop_candidates.append((stmt, update, readout))
            elif update is not None:
                raise PreflightBlocked("BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED", "post-update readout unresolved")
    if len(loop_candidates) != 1:
        status = "BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS" if len(loop_candidates) > 1 else "BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED"
        raise PreflightBlocked(status, "recurrent_state_update")

    loop, update, readout = loop_candidates[0]
    return slow, init_nodes[0], loop, update, readout


def _backend_proof_if(tree: ast.AST) -> ast.If | None:
    forward = _find_unique_function(tree, "MambaMixer.forward")
    if forward is None:
        return None
    matches = [
        stmt
        for stmt in forward.body
        if isinstance(stmt, ast.If)
        and _is_cpu_device_test(stmt.test)
        and any(_branch_calls_slow_forward(branch_stmt) for branch_stmt in stmt.body)
        and _all_statements_raise(stmt.orelse)
    ]
    if len(matches) > 1:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS", "backend_kernel_selection")
    return matches[0] if matches else None


def _has_optional_backend_path(tree: ast.AST) -> bool:
    optional_names = {
        "selective_scan_fn",
        "selective_scan",
        "mamba_inner_fn",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "associative_scan",
        "use_mamba_kernels",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in optional_names:
            return True
        if isinstance(node, ast.Attribute) and node.attr in optional_names:
            return True
    return False


def _location(module: str, qualname: str, source_file_key: str, source_sha256: str, node: ast.AST) -> dict[str, object]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if start is None or end is None:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", qualname)
    return {
        "module": module,
        "qualname": qualname,
        "source_file_key": source_file_key,
        "source_sha256": source_sha256,
        "start_line": int(start),
        "end_line": int(end),
    }


def _exactly_one(candidates: Sequence[tuple[str, ast.AST]], family: str) -> tuple[str, ast.AST]:
    if not candidates:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", family)
    if len(candidates) > 1:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS", family)
    return candidates[0]


def bind_symbol_locations(mamba: SourceFacts, cache: SourceFacts) -> dict[str, dict[str, object]]:
    try:
        mamba_tree = ast.parse(mamba.text)
        cache_tree = ast.parse(cache.text)
    except SyntaxError as exc:
        raise PreflightBlocked("BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE", str(exc)) from exc

    functions = _qualname_stack(mamba_tree)
    slow = _find_unique_function(mamba_tree, "MambaMixer.slow_forward")
    forward = _find_unique_function(mamba_tree, "MambaMixer.forward")
    if slow is None or forward is None:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", "MambaMixer forward/slow_forward")
    _, init_node, _, update_node, readout_node = _recurrent_proof_nodes(mamba_tree)
    backend_node = _backend_proof_if(mamba_tree)
    cache_functions = _qualname_stack(cache_tree)

    if backend_node is None:
        raise PreflightBlocked("BLOCKED_REQUIRED_SYMBOL_UNRESOLVED", "backend_kernel_selection")

    families: dict[str, tuple[str, str, list[tuple[str, ast.AST]]]] = {
        "mixer_forward_dispatch": (
            mamba.module,
            "mamba",
            [(q, n) for q, n in functions if q == "MambaMixer.forward" and _branch_calls_slow_forward(n)],
        ),
        "sequential_slow_path": (
            mamba.module,
            "mamba",
            [(q, n) for q, n in functions if q == "MambaMixer.slow_forward"],
        ),
        "recurrent_state_initialization": (
            mamba.module,
            "mamba",
            [("MambaMixer.slow_forward", init_node)],
        ),
        "recurrent_state_update": (
            mamba.module,
            "mamba",
            [("MambaMixer.slow_forward", update_node)],
        ),
        "convolution_cache_initialization_update": (
            mamba.module,
            "mamba",
            [
                ("MambaMixer.slow_forward", n)
                for n in slow.body
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                and any(path[-1:] and path[-1][:4] == "conv" for path in _assigned_paths(n))
            ],
        ),
        "hidden_state_output_path": (
            mamba.module,
            "mamba",
            [("MambaMixer.slow_forward", readout_node)],
        ),
        "backend_kernel_selection": (
            mamba.module,
            "mamba",
            [("MambaMixer.forward", backend_node)],
        ),
        "cache_recurrent_state_storage": (
            cache.module,
            "cache",
            [
                (q, n)
                for q, func in cache_functions
                for n in getattr(func, "body", [])
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                and any(path[-1:] == ("ssm_state",) or path[-2:] == ("self", "ssm_state") for path in _assigned_paths(n))
                and _loads_name(n, "ssm_state")
            ],
        ),
    }

    out: dict[str, dict[str, object]] = {}
    for family in SYMBOL_KEYS:
        module, source_key, candidates = families[family]
        qualname, node = _exactly_one(candidates, family)
        sha = mamba.sha256 if source_key == "mamba" else cache.sha256
        out[family] = _location(module, qualname, source_key, sha, node)
    return out


def classify_recurrent_semantics(mamba_text: str) -> str:
    try:
        tree = ast.parse(mamba_text)
    except SyntaxError as exc:
        raise PreflightBlocked("BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE", str(exc)) from exc
    _recurrent_proof_nodes(tree)
    return "SOURCE_SUPPORTS_O0C_CONVENTION"


def classify_backend(mamba_text: str) -> str:
    try:
        tree = ast.parse(mamba_text)
    except SyntaxError as exc:
        raise PreflightBlocked("BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE", str(exc)) from exc
    if _backend_proof_if(tree) is not None and not _has_optional_backend_path(tree):
        return "BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN"
    if _has_optional_backend_path(tree):
        return "BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE"
    if _find_unique_function(tree, "MambaMixer.slow_forward") is not None:
        return "BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN"
    return "BACKEND_STATICALLY_UNRESOLVED"


def source_json(facts: SourceFacts) -> dict[str, object]:
    return {
        "module": facts.module,
        "path": str(facts.path),
        "sha256": facts.sha256,
        "bytes": facts.bytes,
        "lf_count": facts.lf_count,
        "cr_count": facts.cr_count,
        "final_lf": facts.final_lf,
    }


def build_artifact(expected: Mapping[str, str], runtime: Mapping[str, str], resolution: SourceResolution) -> dict[str, object]:
    mamba = raw_source_identity(resolution.mamba_source, SOURCE_KEYS["mamba"])
    cache = raw_source_identity(resolution.cache_source, SOURCE_KEYS["cache"])
    recurrent = classify_recurrent_semantics(mamba.text)
    backend = classify_backend(mamba.text)
    if recurrent == "SOURCE_INCOMPATIBLE_WITH_FROZEN_O0C_DESIGN":
        raise PreflightBlocked("BLOCKED_O0C_INDEXING_INCOMPATIBLE", recurrent)
    if recurrent != "SOURCE_SUPPORTS_O0C_CONVENTION":
        raise PreflightBlocked("BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED", recurrent)
    if backend != "BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN":
        raise PreflightBlocked("BLOCKED_BACKEND_PATH_UNRESOLVED", backend)
    symbols = bind_symbol_locations(mamba, cache)
    return {
        "backend_static_classification": backend,
        "cache_source": source_json(cache),
        "expected_runtime": dict(expected),
        "mamba_source": source_json(mamba),
        "notes": sorted(["static_provenance_only", "no_model_tokenizer_dataset_network"]),
        "o0c_full_sequence_capture_feasibility": recurrent,
        "o0c_state_indexing_compatibility": recurrent,
        "optimized_kernel_availability": "NOT_IMPORTED_OBSERVATION_ONLY",
        "preflight_status": PASS_STATUS,
        "runtime": dict(runtime),
        "schema_version": SCHEMA_VERSION,
        "source_resolution": {
            "transformers_distribution_root": str(resolution.distribution_root),
            "transformers_import_root": str(resolution.import_root),
            "transformers_distribution_version": resolution.distribution_version,
            "shadowing_status": "PASS_RECONCILED_UNIQUE_TRANSFORMERS_SOURCE",
        },
        "symbol_locations": symbols,
    }


def serialize_artifact(artifact: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PreflightBlocked("BLOCKED_ARTIFACT_SERIALIZATION_NONDETERMINISTIC", str(exc)) from exc
    return (text + "\n").encode("utf-8")


def publish_artifact(output: Path, data: bytes) -> None:
    if output.exists():
        raise PreflightBlocked("BLOCKED_OUTPUT_COLLISION", str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=".o0c_preflight.", suffix=".tmp", dir=str(output.parent))
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise PreflightBlocked("BLOCKED_OUTPUT_COLLISION", str(output))
        os.replace(tmp_name, output)
        tmp_name = ""
    finally:
        if fd != -1:
            os.close(fd)
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass


def run_preflight(expected: Mapping[str, str], output: Path, repo_root: Path) -> str:
    if output.exists():
        raise PreflightBlocked("BLOCKED_OUTPUT_COLLISION", str(output))
    actual = runtime_versions()
    check_runtime(expected, actual)
    resolution = resolve_transformers_sources(repo_root)
    artifact = build_artifact(expected, actual, resolution)
    publish_artifact(output, serialize_artifact(artifact))
    return PASS_STATUS


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-python", required=True)
    parser.add_argument("--expected-numpy", required=True)
    parser.add_argument("--expected-torch", required=True)
    parser.add_argument("--expected-transformers", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = {
        "python": args.expected_python,
        "numpy": args.expected_numpy,
        "torch": args.expected_torch,
        "transformers": args.expected_transformers,
    }
    output = Path(args.output)
    repo_root = Path(__file__).resolve(strict=True).parents[1]
    try:
        status = run_preflight(expected, output, repo_root)
    except PreflightBlocked as exc:
        print(f"preflight_status={exc.status}")
        print(f"output={output}")
        print(f"blocker={exc.note}")
        return 2
    print(f"preflight_status={status}")
    print(f"output={output}")
    print("blocker=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
