"""K0-RVG raw native Mamba recurrence observer.

Synthetic-validation implementation only.

This module observes the unmodified CPU slow-path recurrence:
    S_t = G_t * S_(t-1) + W_t
at exact CPython line events bound to the frozen Transformers 5.12.1
MambaMixer.slow_forward source.

No scientific-population execution path exists in this module.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

AUTHORITY_COMMIT = "d8d717a09516b8562f64f7c503d4bbdcc9c34c5d"
PARENT_STATIC_COMMIT = "e048e83083da105ee44ef53e9e32193591c79a90"
A0_COMMIT = "55debe94f0d19d16a334395e8561901fed6b52fa"
A0_MODEL_REL = "src/contramamba/modeling_v6b_minimal.py"
A0_HEADS_REL = "src/contramamba/heads"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"

EXPECTED_MAMBA_SOURCE_SHA256 = (
    "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"
)
EXPECTED_UPSTREAM_BLOB = "87987e3e6646d8d0f9f0048bdd8a155d99c845db"

K2S_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
COMMON_ENCODER_CANONICAL_SHA256 = (
    "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
)
COMMON_ENCODER_RAW_CONCAT_SHA256 = (
    "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"
)

EXPECTED_SLOW_FORWARD_LINE = 270
EXPECTED_DISCRETE_A_LINE = 322
EXPECTED_DELTAB_U_LINE = 324
EXPECTED_LOOP_LINE = 349
EXPECTED_UPDATE_LINE = 350
EXPECTED_READOUT_LINE = 351
EXPECTED_FORWARD_LINE = 366

N_LAYERS = 24
VELOCITY_ATOL = 1e-6
VELOCITY_RTOL = 1e-5

OBSERVER_REL = "scripts/longterm_k0_rvg_raw_recurrence_observer.py"
TEST_REL = "tests/test_longterm_k0_rvg_raw_recurrence_observer.py"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
RUNNER_REL = "scripts/longterm_k0_rvg_p1_raw_vector_execution.py"
RUNNER_TEST_REL = "tests/test_longterm_k0_rvg_p1_raw_vector_execution.py"
R2_IMPLEMENTATION_FILES = {OBSERVER_REL, TEST_REL, RUNNER_REL, RUNNER_TEST_REL}


class ContractError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tensor_sha256(tensor: Any) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def _assignment_target(node: ast.AST) -> str:
    target: ast.AST | None = None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


@dataclass(frozen=True)
class SourceBinding:
    code: Any
    source_path: Path
    source_sha256: str
    source_bytes: int
    slow_forward_line: int
    discrete_a_line: int
    deltab_u_line: int
    loop_line: int
    update_line: int
    readout_line: int
    forward_line: int
    qualname: str = "MambaMixer.slow_forward"


def analyze_mamba_source(source: bytes) -> dict[str, int]:
    """Statically prove the unique slow-path recurrence and its readout."""
    try:
        tree = ast.parse(source.decode("utf-8", "strict"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ContractError("MAMBA_SOURCE_PARSE_FAILURE") from exc

    mixers = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MambaMixer"
    ]
    require(len(mixers) == 1, "MAMBA_MIXER_CLASS_AMBIGUOUS")
    mixer = mixers[0]

    slow = [
        node
        for node in mixer.body
        if isinstance(node, ast.FunctionDef) and node.name == "slow_forward"
    ]
    forward = [
        node
        for node in mixer.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    require(len(slow) == 1 and len(forward) == 1, "MAMBA_FORWARD_SOURCE_AMBIGUOUS")
    slow_fn = slow[0]
    forward_fn = forward[0]

    discrete_a = [
        node
        for node in ast.walk(slow_fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assignment_target(node) == "discrete_A"
    ]
    deltab_u = [
        node
        for node in ast.walk(slow_fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and _assignment_target(node) == "deltaB_u"
    ]
    require(len(discrete_a) == 1, "MAMBA_DISCRETE_A_AMBIGUOUS")
    require(len(deltab_u) == 1, "MAMBA_DELTAB_U_AMBIGUOUS")
    require(
        {"A", "discrete_time_step"} <= _names(_value(discrete_a[0])),
        "MAMBA_DISCRETE_A_STRUCTURE_MISMATCH",
    )
    require(
        {"discrete_B", "hidden_states"} <= _names(_value(deltab_u[0])),
        "MAMBA_DELTAB_U_STRUCTURE_MISMATCH",
    )

    candidates: list[tuple[ast.For, ast.AST, ast.AST]] = []
    for node in ast.walk(slow_fn):
        if (
            not isinstance(node, ast.For)
            or not isinstance(node.target, ast.Name)
            or node.target.id != "i"
        ):
            continue
        for pos in range(len(node.body) - 1):
            update = node.body[pos]
            readout = node.body[pos + 1]
            update_value = _value(update)
            readout_value = _value(readout)
            if _assignment_target(update) != "ssm_state":
                continue
            if _assignment_target(readout) != "scan_output":
                continue

            update_ok = (
                isinstance(update_value, ast.BinOp)
                and isinstance(update_value.op, ast.Add)
                and isinstance(update_value.left, ast.BinOp)
                and isinstance(update_value.left.op, ast.Mult)
                and {"discrete_A", "ssm_state", "i"} <= _names(update_value.left)
                and {"deltaB_u", "i"} <= _names(update_value.right)
            )
            readout_ok = (
                isinstance(readout_value, ast.Call)
                and isinstance(readout_value.func, ast.Attribute)
                and readout_value.func.attr == "matmul"
                and len(readout_value.args) >= 2
                and "ssm_state" in _names(readout_value.args[0])
                and {"C", "i"} <= _names(readout_value.args[1])
            )
            if update_ok and readout_ok:
                candidates.append((node, update, readout))

    require(len(candidates) == 1, "MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS")
    loop, update, readout = candidates[0]

    slow_calls = [
        n
        for n in ast.walk(forward_fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "slow_forward"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    ]
    require(len(slow_calls) == 1, "MAMBA_FORWARD_SLOW_FALLBACK_MISSING")

    return {
        "slow_forward_line": int(slow_fn.lineno),
        "discrete_a_line": int(discrete_a[0].lineno),
        "deltab_u_line": int(deltab_u[0].lineno),
        "loop_line": int(loop.lineno),
        "update_line": int(update.lineno),
        "readout_line": int(readout.lineno),
        "forward_line": int(forward_fn.lineno),
    }


def resolve_source_binding() -> SourceBinding:
    import transformers
    import transformers.models.mamba.modeling_mamba as module

    require(transformers.__version__ == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    function = module.MambaMixer.slow_forward
    require(inspect.isfunction(function), "MAMBA_SLOW_FORWARD_NOT_FUNCTION")
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    require(source_path.is_file(), "MAMBA_SOURCE_FILE_MISSING")
    raw = source_path.read_bytes()
    require(
        sha256_bytes(raw) == EXPECTED_MAMBA_SOURCE_SHA256,
        "MAMBA_SOURCE_SHA256_MISMATCH",
    )
    analysis = analyze_mamba_source(raw)

    expected = {
        "slow_forward_line": EXPECTED_SLOW_FORWARD_LINE,
        "discrete_a_line": EXPECTED_DISCRETE_A_LINE,
        "deltab_u_line": EXPECTED_DELTAB_U_LINE,
        "loop_line": EXPECTED_LOOP_LINE,
        "update_line": EXPECTED_UPDATE_LINE,
        "readout_line": EXPECTED_READOUT_LINE,
        "forward_line": EXPECTED_FORWARD_LINE,
    }
    require(analysis == expected, "MAMBA_SOURCE_LINE_BINDING_MISMATCH")
    require(function.__code__.co_filename == str(source_path), "MAMBA_CODE_SOURCE_MISMATCH")
    code_lines = {
        line for _, _, line in function.__code__.co_lines() if line is not None
    }
    require(EXPECTED_UPDATE_LINE in code_lines, "MAMBA_UPDATE_LINE_NOT_EXECUTABLE")
    require(EXPECTED_READOUT_LINE in code_lines, "MAMBA_READOUT_LINE_NOT_EXECUTABLE")

    return SourceBinding(
        code=function.__code__,
        source_path=source_path,
        source_sha256=sha256_bytes(raw),
        source_bytes=len(raw),
        **analysis,
    )


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    device: str


@dataclass(frozen=True)
class _PreCapture:
    s_prev: Any
    g: Any
    w: Any
    s_prev_meta: TensorMeta
    g_meta: TensorMeta
    w_meta: TensorMeta


@dataclass(frozen=True)
class RecurrenceRecord:
    layer_index: int
    token_index: int
    s_prev: Any
    g: Any
    w: Any
    s_post: Any
    s_prev_meta: TensorMeta
    g_meta: TensorMeta
    w_meta: TensorMeta
    s_post_meta: TensorMeta


def _snapshot_tensor(value: Any, role: str) -> tuple[Any, TensorMeta]:
    import torch

    require(value is not None and isinstance(value, torch.Tensor), f"{role}_NOT_TENSOR")
    meta = TensorMeta(
        shape=tuple(int(v) for v in value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
    )
    require(bool(torch.isfinite(value).all().item()), f"{role}_NONFINITE")
    snapshot = value.detach().cpu().contiguous().clone()
    require(snapshot is not value, f"{role}_SNAPSHOT_ALIAS")
    require(snapshot.device.type == "cpu", f"{role}_SNAPSHOT_NOT_CPU")
    return snapshot, meta


def registered_mamba_layers(model: Any) -> dict[int, int]:
    layers = getattr(getattr(model, "mamba", None), "layers", None)
    require(layers is not None and len(layers) == N_LAYERS, "MAMBA_LAYER_COUNT_MISMATCH")
    mapping: dict[int, int] = {}
    for index, block in enumerate(layers):
        mixer = getattr(block, "mixer", None)
        require(mixer is not None, "MAMBA_MIXER_MISSING")
        mapping[id(mixer)] = index
    require(len(mapping) == N_LAYERS, "MAMBA_MIXER_ID_COLLISION")
    require(sorted(mapping.values()) == list(range(N_LAYERS)), "MAMBA_LAYER_INDEX_MISMATCH")
    return mapping


class RawRecurrenceCollector:
    """Single-use CPython line observer for natural recurrence tuples."""

    def __init__(
        self,
        binding: SourceBinding,
        registered_layers: Mapping[int, int],
        target_indices: Iterable[int],
    ) -> None:
        self.binding = binding
        self.layers = dict(registered_layers)
        require(len(self.layers) == N_LAYERS, "REGISTERED_LAYER_COUNT_MISMATCH")
        self.target_indices = frozenset(int(v) for v in target_indices)
        require(bool(self.target_indices), "TRACE_TARGET_INDICES_EMPTY")
        require(min(self.target_indices) >= 0, "TRACE_TARGET_INDEX_NEGATIVE")
        self.records: dict[tuple[int, int], RecurrenceRecord] | None = None
        self._pending: dict[tuple[int, int], _PreCapture] | None = None
        self._prior_trace: Any = None
        self._used = False

    def _coordinate(self, frame: Any) -> tuple[int, int] | None:
        layer_index = self.layers.get(id(frame.f_locals.get("self")))
        if layer_index is None:
            return None
        token_index = frame.f_locals.get("i")
        require(type(token_index) is int and token_index >= 0, "AMBIGUOUS_TOKEN_INDEX")
        if token_index not in self.target_indices:
            return None
        return int(layer_index), int(token_index)

    def _trace(self, frame: Any, event: str, arg: Any):
        if frame.f_code is not self.binding.code or event != "line":
            return self._trace

        if frame.f_lineno == self.binding.update_line:
            key = self._coordinate(frame)
            if key is None:
                return self._trace
            require(self._pending is not None and self.records is not None, "TRACE_NOT_ACTIVE")
            require(key not in self._pending, "DUPLICATE_PRE_UPDATE_CAPTURE")
            require(key not in self.records, "DUPLICATE_RECURRENCE_COORDINATE")

            token_index = key[1]
            s_prev, s_prev_meta = _snapshot_tensor(
                frame.f_locals.get("ssm_state"), "S_PREV"
            )
            discrete_a = frame.f_locals.get("discrete_A")
            deltab_u = frame.f_locals.get("deltaB_u")
            require(discrete_a is not None, "DISCRETE_A_MISSING")
            require(deltab_u is not None, "DELTAB_U_MISSING")
            g, g_meta = _snapshot_tensor(
                discrete_a[:, :, token_index, :], "G"
            )
            w, w_meta = _snapshot_tensor(
                deltab_u[:, :, token_index, :], "W"
            )
            self._pending[key] = _PreCapture(
                s_prev=s_prev,
                g=g,
                w=w,
                s_prev_meta=s_prev_meta,
                g_meta=g_meta,
                w_meta=w_meta,
            )
            return self._trace

        if frame.f_lineno == self.binding.readout_line:
            key = self._coordinate(frame)
            if key is None:
                return self._trace
            require(self._pending is not None and self.records is not None, "TRACE_NOT_ACTIVE")
            require(key in self._pending, "POST_WITHOUT_PRE_CAPTURE")
            require(key not in self.records, "DUPLICATE_POST_UPDATE_CAPTURE")
            pre = self._pending.pop(key)
            s_post, s_post_meta = _snapshot_tensor(
                frame.f_locals.get("ssm_state"), "S_POST"
            )
            record = RecurrenceRecord(
                layer_index=key[0],
                token_index=key[1],
                s_prev=pre.s_prev,
                g=pre.g,
                w=pre.w,
                s_post=s_post,
                s_prev_meta=pre.s_prev_meta,
                g_meta=pre.g_meta,
                w_meta=pre.w_meta,
                s_post_meta=s_post_meta,
            )
            mixer = frame.f_locals.get("self")
            require(mixer is not None, "MAMBA_MIXER_LOCAL_MISSING")
            expected_shape = (
                1,
                int(mixer.intermediate_size),
                int(mixer.ssm_state_size),
            )
            _validate_record_metadata(record, expected_shape=expected_shape)
            self.records[key] = record

        return self._trace

    @contextmanager
    def capture(self):
        require(not self._used, "TRACE_COLLECTOR_REUSE")
        require(self.records is None and self._pending is None, "TRACE_COLLECTOR_DIRTY")
        self.records = {}
        self._pending = {}
        self._prior_trace = sys.gettrace()
        sys.settrace(self._trace)
        completed = False
        try:
            yield self
            completed = True
        finally:
            sys.settrace(self._prior_trace)
            self._used = True
        if completed:
            require(self._pending == {}, "INCOMPLETE_PRE_POST_PAIR")


def _validate_record_metadata(
    record: RecurrenceRecord,
    expected_shape: tuple[int, int, int] | None = None,
) -> None:
    metas = (
        record.s_prev_meta,
        record.g_meta,
        record.w_meta,
        record.s_post_meta,
    )
    require(len({m.shape for m in metas}) == 1, "RECURRENCE_SHAPE_MISMATCH")
    require(len(record.s_prev_meta.shape) == 3, "RECURRENCE_RANK_MISMATCH")
    require(record.s_prev_meta.shape[0] == 1, "RECURRENCE_BATCH_NOT_ONE")
    if expected_shape is not None:
        require(
            record.s_prev_meta.shape == expected_shape,
            "RECURRENCE_EXPECTED_SHAPE_MISMATCH",
        )
    require(
        {m.dtype for m in metas} == {"torch.float32"},
        "RECURRENCE_DTYPE_MISMATCH",
    )
    require(
        {m.device for m in metas} == {"cpu"},
        "RECURRENCE_DEVICE_MISMATCH",
    )


def validate_recurrence_record(record: RecurrenceRecord) -> dict[str, float | str | bool]:
    import torch

    _validate_record_metadata(record)
    reconstructed = record.g * record.s_prev + record.w
    require(torch.equal(reconstructed, record.s_post), "RECURRENCE_EXACT_RECONSTRUCTION_FAILURE")

    v_raw = record.s_post - record.s_prev
    v_carry = (record.g - 1.0) * record.s_prev
    v_write = record.w
    rearranged = v_carry + v_write
    diff = (v_raw - rearranged).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    raw_norm = float(torch.linalg.vector_norm(v_raw).item())
    diff_norm = float(torch.linalg.vector_norm(diff).item())
    relative_frobenius = diff_norm / max(raw_norm, 1e-12)
    tolerance_scale = VELOCITY_ATOL + VELOCITY_RTOL * rearranged.abs()
    max_scaled = float((diff / tolerance_scale).max().item()) if diff.numel() else 0.0
    rearrangement_allclose = bool(
        torch.allclose(
            v_raw,
            rearranged,
            atol=VELOCITY_ATOL,
            rtol=VELOCITY_RTOL,
        )
    )
    rearrangement_status = (
        "PASS_TOLERANCE"
        if rearrangement_allclose
        else "DIAGNOSTIC_TOLERANCE_EXCEEDED"
    )
    return {
        "recurrence_exact": "PASS_EXACT",
        "velocity_rearrangement": rearrangement_status,
        "velocity_rearrangement_allclose": rearrangement_allclose,
        "velocity_atol": VELOCITY_ATOL,
        "velocity_rtol": VELOCITY_RTOL,
        "max_abs_residual": max_abs,
        "max_relative_residual": relative_frobenius,
        "max_scaled_tolerance_residual": max_scaled,
    }


def record_hashes(record: RecurrenceRecord) -> dict[str, str]:
    return {
        "S_prev": tensor_sha256(record.s_prev),
        "G": tensor_sha256(record.g),
        "W": tensor_sha256(record.w),
        "S_post": tensor_sha256(record.s_post),
    }


def _import_k2s(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import longterm_k2s_pair_specific_event_dynamics as k2s

    return k2s


def _repo_contract(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    authority_ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        authority_ancestor == 0,
        "IMPLEMENTATION_AUTHORITY_NOT_ANCESTOR",
    )

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        if path in HISTORICAL_K1_UNTRACKED:
            require(line[:2] == "??", "GIT_DIRTY_CONTRACT_MISMATCH")
            continue
        if path in R2_IMPLEMENTATION_FILES:
            require(line[:2] == " M", "GIT_DIRTY_CONTRACT_MISMATCH")
            continue
        raise ContractError("GIT_DIRTY_CONTRACT_MISMATCH")

    helper = root / K2S_REL
    require(helper.is_file(), "K2S_HELPER_MISSING")
    require(file_sha256(helper) == K2S_SHA256, "K2S_HELPER_SHA256_MISMATCH")
    require(_git(root, "rev-parse", f"HEAD:{K2S_REL}") == K2S_GIT_BLOB, "K2S_HELPER_BLOB_MISMATCH")

    current_model_blob = _git(root, "rev-parse", f"{head}:{A0_MODEL_REL}")
    frozen_model_blob = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_MODEL_REL}")
    current_heads_tree = _git(root, "rev-parse", f"{head}:{A0_HEADS_REL}")
    frozen_heads_tree = _git(root, "rev-parse", f"{A0_COMMIT}:{A0_HEADS_REL}")
    require(current_model_blob == frozen_model_blob, "A0_MODEL_SOURCE_DRIFT")
    require(current_heads_tree == frozen_heads_tree, "A0_HEADS_SOURCE_DRIFT")

    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "implementation_authority_commit": AUTHORITY_COMMIT,
        "implementation_authority_is_ancestor": True,
        "runtime_dirty_contract": status,
        "k2s_helper_sha256": K2S_SHA256,
        "k2s_helper_git_blob": K2S_GIT_BLOB,
        "a0_model_blob_sha": current_model_blob,
        "a0_heads_tree_sha": current_heads_tree,
    }


def _synthetic_bundle_set(k2s: Any, tokenizer: Any) -> tuple[list[Mapping[str, Sequence[int]]], list[int]]:
    prefix = "Claim: synthetic blorp\nEvidence: synthetic snarp\nAdditional evidence:\n"
    continuations = (
        " synthetic corrective alpha beta gamma delta epsilon zeta eta theta iota.",
        " synthetic control kappa lambda mu nu xi omicron pi rho sigma.",
        " synthetic swapped tau upsilon phi chi psi omega amber cobalt jade.",
        " synthetic alternate cedar quartz river summit valley willow ember frost.",
    )
    bundles = [k2s.task_mask_bundle(tokenizer, prefix + c) for c in continuations]
    prefix_ids = k2s._token_ids(tokenizer, prefix)
    require(len(prefix_ids) >= 2, "SYNTHETIC_PREFIX_TOO_SHORT")
    require(
        all(list(bundle["input_ids"])[: len(prefix_ids)] == prefix_ids for bundle in bundles),
        "SYNTHETIC_PREFIX_TOKEN_MISMATCH",
    )
    return bundles, prefix_ids


def _compare_record_maps(
    a: Mapping[tuple[int, int], RecurrenceRecord],
    b: Mapping[tuple[int, int], RecurrenceRecord],
    failure: str,
) -> None:
    import torch

    require(set(a) == set(b), failure + "_COORDINATE")
    for key in a:
        for role in ("s_prev", "g", "w", "s_post"):
            require(torch.equal(getattr(a[key], role), getattr(b[key], role)), failure + "_" + role.upper())


def run_synthetic_preflight(root: Path, handoff_path: Path, hf_revision: str) -> dict[str, Any]:
    import torch

    provenance = _repo_contract(root)
    k2s = _import_k2s(root)

    require(k2s.file_sha256(root / K2S_REL) == K2S_SHA256, "K2S_RUNTIME_SHA_MISMATCH")
    require(k2s.EXPECTED_ZIP_SHA256 == EXPECTED_ZIP_SHA256, "HANDOFF_CONSTANT_DRIFT")
    require(
        k2s.EXPECTED_CHECKPOINT_SHA256 == EXPECTED_CHECKPOINT_SHA256,
        "CHECKPOINT_CONSTANT_DRIFT",
    )

    handoff = k2s.audit_handoff(handoff_path)
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    require(
        encoder["canonical_digest"] == COMMON_ENCODER_CANONICAL_SHA256,
        "ENCODER_CANONICAL_DRIFT",
    )
    require(
        encoder["raw_concat_digest"] == COMMON_ENCODER_RAW_CONCAT_SHA256,
        "ENCODER_RAW_DRIFT",
    )

    snapshot, hf = k2s.resolve_hf_snapshot(hf_revision)
    require(hf["hf_model_id"] == HF_MODEL, "HF_MODEL_MISMATCH")
    require(hf["resolved_hf_revision"] == HF_REVISION, "HF_REVISION_MISMATCH")
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    tokenizer = hf["tokenizer"]
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()

    binding = resolve_source_binding()
    old_binding = k2s.resolve_capture_binding()
    require(old_binding.source_sha256 == binding.source_sha256, "K2S_SOURCE_BINDING_DRIFT")
    require(old_binding.recurrence_update_line == binding.update_line, "K2S_UPDATE_LINE_DRIFT")
    require(old_binding.capture_line == binding.readout_line, "K2S_READOUT_LINE_DRIFT")

    layer_map = registered_mamba_layers(model)
    old_layer_map = k2s.registered_mamba_layers(model)
    require(layer_map == old_layer_map, "K2S_LAYER_MAP_DRIFT")

    bundles, prefix_ids = _synthetic_bundle_set(k2s, tokenizer)
    first = bundles[0]
    token_count = len(first["input_ids"])
    target_all = range(token_count)

    baseline = k2s._full_model_forward(model, first)
    baseline_logits = k2s._logits(baseline).detach().cpu().clone()

    prior_trace = sys.gettrace()
    collector = RawRecurrenceCollector(binding, layer_map, target_all)
    with collector.capture():
        traced = k2s._full_model_forward(model, first)
    require(sys.gettrace() is prior_trace, "TRACE_RESTORATION_FAILURE")
    traced_logits = k2s._logits(traced).detach().cpu()
    require(torch.equal(baseline_logits, traced_logits), "RAW_RECURRENCE_OBSERVER_NONINTERFERENCE_FAILURE")
    require(collector.records is not None, "RAW_RECORDS_MISSING")
    expected_count = N_LAYERS * token_count
    require(len(collector.records) == expected_count, "RAW_CAPTURE_INCOMPLETE")

    max_abs = 0.0
    max_rel = 0.0
    max_scaled = 0.0
    diagnostic_pass_count = 0
    diagnostic_exceedance_count = 0
    for record in collector.records.values():
        result = validate_recurrence_record(record)
        status = result["velocity_rearrangement"]
        if status == "PASS_TOLERANCE":
            diagnostic_pass_count += 1
        elif status == "DIAGNOSTIC_TOLERANCE_EXCEEDED":
            diagnostic_exceedance_count += 1
        else:
            raise ContractError("UNKNOWN_VELOCITY_REARRANGEMENT_STATUS")
        max_abs = max(max_abs, float(result["max_abs_residual"]))
        max_rel = max(max_rel, float(result["max_relative_residual"]))
        max_scaled = max(
            max_scaled,
            float(result["max_scaled_tolerance_residual"]),
        )
    require(
        diagnostic_pass_count + diagnostic_exceedance_count == len(collector.records),
        "VELOCITY_DIAGNOSTIC_ACCOUNTING_MISMATCH",
    )

    fresh = RawRecurrenceCollector(binding, layer_map, target_all)
    with fresh.capture():
        k2s._full_model_forward(model, first)
    require(fresh.records is not None, "FRESH_RAW_RECORDS_MISSING")
    _compare_record_maps(
        collector.records,
        fresh.records,
        "FRESH_FORWARD_IDENTITY_FAILURE",
    )

    old = k2s.TraceCollector(old_binding, old_layer_map, target_all, enabled=True)
    with old.capture():
        k2s._full_model_forward(model, first)
    require(old.snapshots is not None, "K2S_BRIDGE_SNAPSHOTS_MISSING")
    require(set(old.snapshots) == set(collector.records), "K2S_BRIDGE_COORDINATE_MISMATCH")
    for key, old_state in old.snapshots.items():
        require(
            torch.equal(old_state, collector.records[key].s_post),
            "K2S_POST_CONSUMPTION_BRIDGE_FAILURE",
        )

    prefix_targets = range(len(prefix_ids))
    reference_hashes: dict[tuple[int, int], dict[str, str]] | None = None
    for branch_index, bundle in enumerate(bundles):
        current = RawRecurrenceCollector(binding, layer_map, prefix_targets)
        with current.capture():
            k2s._full_model_forward(model, bundle)
        require(current.records is not None, "PREFIX_RAW_RECORDS_MISSING")
        require(
            len(current.records) == N_LAYERS * len(prefix_ids),
            "PREFIX_RAW_CAPTURE_INCOMPLETE",
        )
        hashes = {key: record_hashes(value) for key, value in current.records.items()}
        if branch_index == 0:
            reference_hashes = hashes
        else:
            require(hashes == reference_hashes, "CAUSAL_PREFIX_RAW_RECURRENCE_IDENTITY_FAILURE")

    reuse_rejected = False
    try:
        with collector.capture():
            pass
    except ContractError as exc:
        reuse_rejected = str(exc) == "TRACE_COLLECTOR_REUSE"
    require(reuse_rejected, "TRACE_COLLECTOR_REUSE_NOT_REJECTED")

    return {
        "schema_version": "k0-rvg-raw-recurrence-synthetic-preflight-v1",
        "status": "PASS_SYNTHETIC_RAW_RECURRENCE_OBSERVER",
        "scientific_population_accessed": False,
        "scientific_recurrent_state_read": False,
        "provenance": provenance,
        "hf": {
            "model_id": HF_MODEL,
            "revision": HF_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
        },
        "source_binding": {
            "source_path": str(binding.source_path),
            "source_sha256": binding.source_sha256,
            "source_bytes": binding.source_bytes,
            "upstream_git_blob": EXPECTED_UPSTREAM_BLOB,
            "qualname": binding.qualname,
            "slow_forward_line": binding.slow_forward_line,
            "discrete_a_line": binding.discrete_a_line,
            "deltab_u_line": binding.deltab_u_line,
            "loop_line": binding.loop_line,
            "update_line": binding.update_line,
            "readout_line": binding.readout_line,
            "forward_line": binding.forward_line,
        },
        "capture": {
            "layer_count": N_LAYERS,
            "synthetic_token_count": token_count,
            "record_count": len(collector.records),
            "state_timing": "paired_pre_update_and_post_consumption_s_t",
            "tensor_shape": list(next(iter(collector.records.values())).s_post.shape),
            "dtype": str(next(iter(collector.records.values())).s_post.dtype),
            "snapshot_device": str(next(iter(collector.records.values())).s_post.device),
        },
        "checks": {
            "source_sha_binding": "PASS_EXACT",
            "source_ast_structure": "PASS_EXACT",
            "slow_path_traversal": "PASS",
            "layer_registration": "PASS_EXACT_24",
            "pre_post_pair_completeness": "PASS_EXACT",
            "tensor_shape_dtype_finite": "PASS",
            "snapshot_nonaliasing": "PASS",
            "recurrence_reconstruction": "PASS_EXACT",
            "velocity_rearrangement_blocking": False,
            "velocity_rearrangement_diagnostic_pass_count": diagnostic_pass_count,
            "velocity_rearrangement_diagnostic_exceedance_count": diagnostic_exceedance_count,
            "velocity_atol": VELOCITY_ATOL,
            "velocity_rtol": VELOCITY_RTOL,
            "velocity_max_abs_residual": max_abs,
            "velocity_max_relative_frobenius_residual": max_rel,
            "velocity_max_scaled_tolerance_residual": max_scaled,
            "logit_noninterference": "PASS_EXACT",
            "fresh_forward_identity": "PASS_EXACT_ALL_ROLES",
            "causal_prefix_identity": "PASS_EXACT_ALL_ROLES_ALL_LAYERS",
            "k2s_post_consumption_bridge": "PASS_BYTE_IDENTICAL",
            "trace_restoration": "PASS",
            "collector_reuse_rejection": "PASS",
        },
        "handoff": {
            "zip_sha256": EXPECTED_ZIP_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "encoder": encoder,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K0-RVG synthetic-only raw Mamba recurrence observer validation."
    )
    parser.add_argument(
        "--synthetic-preflight",
        action="store_true",
        help="Run fabricated-text synthetic observer validation only.",
    )
    parser.add_argument(
        "--seed180-handoff",
        type=Path,
        help="Validated seed180 handoff ZIP used only to build the frozen model.",
    )
    parser.add_argument(
        "--hf-revision",
        default=HF_REVISION,
        help="Must equal the frozen HF revision.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.synthetic_preflight:
        parser.error("only --synthetic-preflight is implemented")
    if args.seed180_handoff is None:
        parser.error("--seed180-handoff is required for --synthetic-preflight")
    root = Path(__file__).resolve().parents[1]
    result = run_synthetic_preflight(root, args.seed180_handoff.resolve(), args.hf_revision)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
