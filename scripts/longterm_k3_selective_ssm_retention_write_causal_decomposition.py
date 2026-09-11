"""K3 selective-SSM retention-vs-write causal decomposition.

Implementation authority: frozen K3 preregistration at commit
20032bb53d77416bb7eb25411eb4f77b008648b4.

This module implements the structural capture/replay machinery and the complete
pre-registered K3 statistics, but the only CLI action authorized in this phase
is --replay-preflight. Scientific intervention execution is intentionally not
reachable from this implementation-phase CLI.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

K3_PREREG_AUTHORITY_COMMIT = "20032bb53d77416bb7eb25411eb4f77b008648b4"
K3_PREREG_REL = "reports/longterm_k3_selective_ssm_retention_write_causal_decomposition_prereg_candidate.md"
K3_PREREG_SHA256 = "7561f4188921b645eba3d7108bcd007b7c223389b5e1abc512cc8d61af976c84"

K2R_RESULT_ARCHIVE_COMMIT = "ba0ad9052a3b8a5eb80fef46dea461f372f95ee8"
K2R_IMPLEMENTATION_COMMIT = "52bd363bb3690b4c85dbdb6add686ecb31088627"
K2R_RUNNER_REL = "scripts/longterm_k2r_claim_disjoint_dissociation_replication.py"
K2R_RUNNER_SHA256 = "557d537e3dde30fdfd1f3e03fe9a2e7d019499e8e53c76242ff5eee9b8c87eeb"
K2R_RUNNER_GIT_BLOB = "0c45e6a5538697033b776127a2daea407cf778cd"

K2S_RUNNER_REL = "scripts/longterm_k2s_pair_specific_event_dynamics.py"
K2S_RUNNER_SHA256 = "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
K2S_RUNNER_GIT_BLOB = "3a651fb508669bdcf72441a4869b863d6eee6c1f"

K2R_ARCHIVE_DIR = "reports/longterm_k2r_claim_disjoint_replication_52bd363_v1"
K2R_CANDIDATE_REL = f"{K2R_ARCHIVE_DIR}/candidate_pool.jsonl"
K2R_CANDIDATE_SHA256 = "00bbdc9977679aa0561be413e7cef8e710dc7987618fe9593e8bf0852a5a7de4"
K2R_BLOCK_REL = f"{K2R_ARCHIVE_DIR}/block_metrics.jsonl"
K2R_BLOCK_SHA256 = "e4982e8a57e15863227d17f080e7a7a699fb14100387fc36c7c47ab68962c8de"
K2R_MANIFEST_REL = f"{K2R_ARCHIVE_DIR}/manifest.json"
K2R_MANIFEST_SHA256 = "fecf202165256de086d49d60e0b8613765bb869c4988744b92aabed3f1931919"
K2R_CLOSURE_REL = "reports/longterm_k2r_claim_disjoint_replication_closure_report_candidate.md"
K2R_CLOSURE_SHA256 = "49d4f7f1c8d8053b62796009526f86f2fd5a1345728f1c14f6b0ebf981b7976b"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}
K3_UNTRACKED = {
    "scripts/longterm_k3_selective_ssm_retention_write_causal_decomposition.py",
    "tests/test_longterm_k3_selective_ssm_retention_write_causal_decomposition.py",
}

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MAMBA_SOURCE_SHA256 = "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"

PRIMARY_LAYER = 23
N_LAYERS = 24
N_ITEMS = 300
N_BLOCKS = 150
W = 8
EPSILON = 1e-12

METRICS = ("R", "D", "DISP", "P")
EXPECTED_DIRECTION = {"R": 1, "D": 1, "DISP": -1, "P": -1}
DOMINANT_COMPONENT = {"R": "W", "D": "W", "DISP": "G", "P": "G"}
PRIMARY_TEST_ORDER = (
    "R_ATT", "R_SEL",
    "D_ATT", "D_SEL",
    "DISP_ATT", "DISP_SEL",
    "P_ATT", "P_SEL",
)

SUCCESS_VERDICT = "LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CAUSALLY_SUPPORTED"
CONTRADICTION_VERDICT = "LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_CONTRADICTED"
NOT_ESTABLISHED_VERDICT = "LAYER23_NATIVE_RECURRENCE_COMPONENT_SPECIALIZATION_NOT_ESTABLISHED"

EXPECTED_ZIP_SHA256 = "96859bad3e400613b4c981990e56aaf35b1d92baaaa930eb11448cf38a63b861"
EXPECTED_CHECKPOINT_SHA256 = "4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c"
COMMON_ENCODER_CANONICAL_SHA256 = "48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597"
COMMON_ENCODER_RAW_CONCAT_SHA256 = "968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae"

class ContractError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc

def parse_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    require(not raw.startswith(b"\xef\xbb\xbf"), "JSONL_BOM_FORBIDDEN")
    require(b"\r" not in raw, "JSONL_CR_FORBIDDEN")
    require(raw.endswith(b"\n"), "JSONL_FINAL_LF_REQUIRED")
    lines = raw[:-1].split(b"\n")
    require(bool(lines) and all(lines), "JSONL_BLANK_LINE_FORBIDDEN")
    out: list[dict[str, Any]] = []
    for line in lines:
        value = json.loads(line.decode("utf-8", "strict"))
        require(isinstance(value, dict), "JSONL_ROW_NOT_OBJECT")
        out.append(value)
    return out

def git_provenance(root: Path, replay_preflight: bool) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    head = _git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", K3_PREREG_AUTHORITY_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "K3_PREREG_NOT_ANCESTOR")

    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    allowed = set(HISTORICAL_K1_UNTRACKED)
    if replay_preflight:
        allowed |= K3_UNTRACKED
    for line in status:
        path = line[3:].replace("\\", "/") if len(line) >= 4 else ""
        require(line[:2] == "??" and path in allowed, "GIT_DIRTY_CONTRACT_MISMATCH")

    exact_files = {
        K3_PREREG_REL: K3_PREREG_SHA256,
        K2R_RUNNER_REL: K2R_RUNNER_SHA256,
        K2S_RUNNER_REL: K2S_RUNNER_SHA256,
        K2R_CANDIDATE_REL: K2R_CANDIDATE_SHA256,
        K2R_BLOCK_REL: K2R_BLOCK_SHA256,
        K2R_MANIFEST_REL: K2R_MANIFEST_SHA256,
        K2R_CLOSURE_REL: K2R_CLOSURE_SHA256,
    }
    for rel, expected in exact_files.items():
        path = root / rel
        require(path.is_file(), f"REQUIRED_FILE_MISSING:{rel}")
        require(file_sha256(path) == expected, f"REQUIRED_FILE_SHA_MISMATCH:{rel}")

    require(
        _git(root, "rev-parse", f"{head}:{K2R_RUNNER_REL}") == K2R_RUNNER_GIT_BLOB,
        "K2R_RUNNER_BLOB_DRIFT",
    )
    require(
        _git(root, "rev-parse", f"{head}:{K2S_RUNNER_REL}") == K2S_RUNNER_GIT_BLOB,
        "K2S_RUNNER_BLOB_DRIFT",
    )
    return {
        "runtime_branch": branch,
        "runtime_git_head": head,
        "runtime_dirty_contract": status,
        "k3_prereg_sha256": K3_PREREG_SHA256,
        "k2r_runner_sha256": K2R_RUNNER_SHA256,
        "k2r_runner_git_blob": K2R_RUNNER_GIT_BLOB,
        "k2s_runner_sha256": K2S_RUNNER_SHA256,
        "k2s_runner_git_blob": K2S_RUNNER_GIT_BLOB,
        "k2r_candidate_sha256": K2R_CANDIDATE_SHA256,
        "k2r_block_sha256": K2R_BLOCK_SHA256,
    }

def load_frozen_dependencies(root: Path) -> tuple[Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    k2r = importlib.import_module("scripts.longterm_k2r_claim_disjoint_dissociation_replication")
    k2s = importlib.import_module("scripts.longterm_k2s_pair_specific_event_dynamics")
    require(Path(k2r.__file__).resolve() == (root / K2R_RUNNER_REL).resolve(), "K2R_IMPORT_PATH_MISMATCH")
    require(Path(k2s.__file__).resolve() == (root / K2S_RUNNER_REL).resolve(), "K2S_IMPORT_PATH_MISMATCH")
    require(file_sha256(Path(k2r.__file__)) == K2R_RUNNER_SHA256, "K2R_IMPORT_BYTE_MISMATCH")
    require(file_sha256(Path(k2s.__file__)) == K2S_RUNNER_SHA256, "K2S_IMPORT_BYTE_MISMATCH")
    require(tuple(k2r.PRIMARY_ORDER) == METRICS, "K2R_PRIMARY_METRIC_DRIFT")
    require(k2r.PRIMARY_LAYER == PRIMARY_LAYER and k2r.W == W, "K2R_GEOMETRY_DRIFT")
    require(k2s.PRIMARY_LAYER == PRIMARY_LAYER and k2s.W == W, "K2S_GEOMETRY_DRIFT")
    return k2r, k2s

def audit_population_without_outcomes(root: Path) -> dict[str, Any]:
    raw = (root / K2R_CANDIDATE_REL).read_bytes()
    require(sha256_bytes(raw) == K2R_CANDIDATE_SHA256, "K2R_CANDIDATE_SHA_MISMATCH")
    rows = parse_jsonl_bytes(raw)
    require(len(rows) == N_ITEMS, "K2R_CANDIDATE_COUNT_MISMATCH")
    stable = [str(row["stable_item_id"]) for row in rows]
    require(stable == sorted(stable), "K2R_CANDIDATE_ORDER_MISMATCH")
    require(len(set(stable)) == N_ITEMS, "K2R_STABLE_ID_DUPLICATE")
    blocks = []
    for block_index in range(N_BLOCKS):
        a, b = 2 * block_index, 2 * block_index + 1
        require((a ^ 1) == b and (b ^ 1) == a, "K2R_RECIPROCAL_MAPPING_BROKEN")
        blocks.append({
            "block_index": block_index,
            "item_a_stable_id": stable[a],
            "item_b_stable_id": stable[b],
        })
    mapping_sha = sha256_bytes(canonical_json(blocks))
    return {
        "candidate_pool_sha256": K2R_CANDIDATE_SHA256,
        "candidate_pool_count": N_ITEMS,
        "reciprocal_block_count": N_BLOCKS,
        "reciprocal_mapping_canonical_sha256": mapping_sha,
        "scientific_outcome_values_read": False,
    }

def _names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }

def _target_name(node: ast.AST) -> str:
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

def analyze_component_source(source: bytes) -> dict[str, int]:
    try:
        tree = ast.parse(source.decode("utf-8", "strict"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ContractError("MAMBA_SOURCE_PARSE_FAILURE") from exc
    mixers = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaMixer"]
    require(len(mixers) == 1, "MAMBA_MIXER_CLASS_AMBIGUOUS")
    slow = [n for n in mixers[0].body if isinstance(n, ast.FunctionDef) and n.name == "slow_forward"]
    require(len(slow) == 1, "MAMBA_SLOW_FORWARD_AMBIGUOUS")

    discrete_a_lines = [
        n for n in ast.walk(slow[0])
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and _target_name(n) == "discrete_A"
        and {"A", "discrete_time_step", "exp"} <= _names(n.value)
    ]
    delta_b_u_lines = [
        n for n in ast.walk(slow[0])
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and _target_name(n) == "deltaB_u"
        and {"discrete_B", "hidden_states"} <= _names(n.value)
    ]
    require(len(discrete_a_lines) == 1, "DISCRETE_A_DEFINITION_AMBIGUOUS")
    require(len(delta_b_u_lines) == 1, "DELTAB_U_DEFINITION_AMBIGUOUS")

    candidates: list[tuple[ast.For, ast.AST, ast.AST]] = []
    for node in ast.walk(slow[0]):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != "i":
            continue
        for pos in range(len(node.body) - 1):
            update = node.body[pos]
            readout = node.body[pos + 1]
            if _target_name(update) != "ssm_state" or _target_name(readout) != "scan_output":
                continue
            value = update.value if isinstance(update, (ast.Assign, ast.AnnAssign)) else None
            require(value is not None, "RECURRENCE_UPDATE_VALUE_MISSING")
            update_ok = (
                isinstance(value, ast.BinOp)
                and isinstance(value.op, ast.Add)
                and isinstance(value.left, ast.BinOp)
                and isinstance(value.left.op, ast.Mult)
                and {"discrete_A", "ssm_state", "i"} <= _names(value.left)
                and {"deltaB_u", "i"} <= _names(value.right)
            )
            if update_ok:
                candidates.append((node, update, readout))
    require(len(candidates) == 1, "MAMBA_SEQUENTIAL_RECURRENCE_AMBIGUOUS")
    _, update, readout = candidates[0]
    return {
        "discrete_A_line": int(discrete_a_lines[0].lineno),
        "deltaB_u_line": int(delta_b_u_lines[0].lineno),
        "recurrence_update_line": int(update.lineno),
        "post_update_line": int(readout.lineno),
    }

@dataclass(frozen=True)
class ComponentBinding:
    code: Any
    source_path: Path
    source_sha256: str
    source_bytes: int
    recurrence_update_line: int
    post_update_line: int
    discrete_A_line: int
    deltaB_u_line: int
    qualname: str

def resolve_component_binding(k2s: Any) -> ComponentBinding:
    import transformers
    import transformers.models.mamba.modeling_mamba as module
    require(transformers.__version__ == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    function = module.MambaMixer.slow_forward
    require(inspect.isfunction(function), "MAMBA_SLOW_FORWARD_NOT_FUNCTION")
    source_path = Path(inspect.getsourcefile(function) or "").resolve()
    require(source_path.is_file(), "MAMBA_SOURCE_FILE_MISSING")
    raw = source_path.read_bytes()
    require(sha256_bytes(raw) == MAMBA_SOURCE_SHA256, "MAMBA_SOURCE_SHA256_MISMATCH")
    source_analysis = analyze_component_source(raw)
    k2s_binding = k2s.resolve_capture_binding()
    require(k2s_binding.source_sha256 == MAMBA_SOURCE_SHA256, "K2S_CAPTURE_SOURCE_SHA_DRIFT")
    require(
        k2s_binding.recurrence_update_line == source_analysis["recurrence_update_line"]
        and k2s_binding.capture_line == source_analysis["post_update_line"],
        "K2S_K3_CAPTURE_LINE_DISAGREEMENT",
    )
    code_lines = {line for _, _, line in function.__code__.co_lines() if line is not None}
    require(source_analysis["recurrence_update_line"] in code_lines, "RECURRENCE_UPDATE_LINE_NOT_EXECUTABLE")
    require(source_analysis["post_update_line"] in code_lines, "POST_UPDATE_LINE_NOT_EXECUTABLE")
    return ComponentBinding(
        code=function.__code__,
        source_path=source_path,
        source_sha256=MAMBA_SOURCE_SHA256,
        source_bytes=len(raw),
        recurrence_update_line=source_analysis["recurrence_update_line"],
        post_update_line=source_analysis["post_update_line"],
        discrete_A_line=source_analysis["discrete_A_line"],
        deltaB_u_line=source_analysis["deltaB_u_line"],
        qualname="MambaMixer.slow_forward",
    )

def tensor_sha256(tensor: Any) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())

class ComponentCollector:
    """Capture natural layer-23 G, W, pre-state and post-state without mutation."""

    def __init__(
        self,
        binding: ComponentBinding,
        target_mixer: Any,
        component_indices: Iterable[int],
        state_indices: Iterable[int],
    ) -> None:
        self.binding = binding
        self.target_mixer_id = id(target_mixer)
        self.component_indices = frozenset(int(i) for i in component_indices)
        self.state_indices = frozenset(int(i) for i in state_indices)
        require(self.component_indices, "COMPONENT_TARGETS_EMPTY")
        require(self.state_indices, "STATE_TARGETS_EMPTY")
        self.G: dict[int, Any] = {}
        self.W: dict[int, Any] = {}
        self.pre_state: dict[int, Any] = {}
        self.post_state: dict[int, Any] = {}
        self._prior = None

    @staticmethod
    def _clone_cpu(tensor: Any) -> Any:
        require(tensor is not None and hasattr(tensor, "detach"), "COMPONENT_TENSOR_MISSING")
        out = tensor.detach().cpu().contiguous().clone()
        require(out is not tensor, "COMPONENT_SNAPSHOT_ALIAS")
        return out

    def _trace(self, frame: Any, event: str, arg: Any):
        if frame.f_code is not self.binding.code or event != "line":
            return self._trace
        mixer = frame.f_locals.get("self")
        if id(mixer) != self.target_mixer_id:
            return self._trace
        token_index = frame.f_locals.get("i")
        if type(token_index) is not int or token_index < 0:
            return self._trace
        i = int(token_index)
        if frame.f_lineno == self.binding.recurrence_update_line and i in self.component_indices:
            discrete_A = frame.f_locals.get("discrete_A")
            deltaB_u = frame.f_locals.get("deltaB_u")
            ssm_state = frame.f_locals.get("ssm_state")
            require(discrete_A is not None and deltaB_u is not None and ssm_state is not None, "COMPONENT_LOCALS_MISSING")
            require(i not in self.G and i not in self.W and i not in self.pre_state, "DUPLICATE_COMPONENT_COORDINATE")
            self.G[i] = self._clone_cpu(discrete_A[:, :, i, :])
            self.W[i] = self._clone_cpu(deltaB_u[:, :, i, :])
            self.pre_state[i] = self._clone_cpu(ssm_state)
        elif frame.f_lineno == self.binding.post_update_line and i in self.state_indices:
            ssm_state = frame.f_locals.get("ssm_state")
            require(i not in self.post_state, "DUPLICATE_POST_STATE_COORDINATE")
            self.post_state[i] = self._clone_cpu(ssm_state)
        return self._trace

    @contextmanager
    def capture(self):
        self._prior = sys.gettrace()
        sys.settrace(self._trace)
        try:
            yield self
        finally:
            sys.settrace(self._prior)

def validate_component_tensor(tensor: Any, mixer: Any, label: str) -> None:
    import torch
    expected = (1, int(mixer.intermediate_size), int(mixer.ssm_state_size))
    require(tuple(tensor.shape) == expected, f"{label}_SHAPE_MISMATCH")
    require(tensor.dtype == torch.float32, f"{label}_DTYPE_MISMATCH")
    require(tensor.device.type == "cpu", f"{label}_DEVICE_MISMATCH")
    require(bool(torch.isfinite(tensor).all().item()), f"{label}_NONFINITE")

def first_divergence(corr: Sequence[int], ctrl: Sequence[int], prefix_len: int, end_inclusive: int) -> int:
    stop = min(len(corr), len(ctrl), end_inclusive + 1)
    value = next((i for i in range(prefix_len, stop) if corr[i] != ctrl[i]), None)
    require(value is not None, "PAIR_DIVERGENCE_MISSING")
    return int(value)

def arithmetic_midpoint(a: Any, b: Any) -> Any:
    return 0.5 * (a + b)

def structural_replay(
    initial_state: Any,
    G: Mapping[int, Any],
    W_terms: Mapping[int, Any],
    start: int,
    end: int,
) -> dict[int, Any]:
    require(start <= end, "REPLAY_RANGE_INVALID")
    require(set(range(start, end + 1)) <= set(G), "REPLAY_G_INCOMPLETE")
    require(set(range(start, end + 1)) <= set(W_terms), "REPLAY_W_INCOMPLETE")
    current = initial_state.detach().cpu().contiguous().clone()
    out: dict[int, Any] = {}
    for t in range(start, end + 1):
        current = G[t] * current + W_terms[t]
        out[t] = current.detach().cpu().contiguous().clone()
    return out

def equalized_terms(
    corr_G: Mapping[int, Any],
    corr_W: Mapping[int, Any],
    ctrl_G: Mapping[int, Any],
    ctrl_W: Mapping[int, Any],
    start: int,
    end: int,
    condition: str,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any], dict[int, Any]]:
    require(condition in {"BASE", "W_EQ", "G_EQ", "GW_EQ"}, "INTERVENTION_CONDITION_INVALID")
    cg, cw, ng, nw = {}, {}, {}, {}
    for t in range(start, end + 1):
        require(t in corr_G and t in corr_W and t in ctrl_G and t in ctrl_W, "INTERVENTION_TERMS_INCOMPLETE")
        gbar = arithmetic_midpoint(corr_G[t], ctrl_G[t])
        wbar = arithmetic_midpoint(corr_W[t], ctrl_W[t])
        cg[t] = corr_G[t] if condition in {"BASE", "W_EQ"} else gbar
        ng[t] = ctrl_G[t] if condition in {"BASE", "W_EQ"} else gbar
        cw[t] = corr_W[t] if condition in {"BASE", "G_EQ"} else wbar
        nw[t] = ctrl_W[t] if condition in {"BASE", "G_EQ"} else wbar
    return cg, cw, ng, nw

def replay_pair(
    corr: Mapping[str, Any],
    ctrl: Mapping[str, Any],
    start: int,
    end: int,
    condition: str,
) -> tuple[dict[int, Any], dict[int, Any]]:
    import torch
    require(start - 1 in corr["post_state"] and start - 1 in ctrl["post_state"], "PAIR_INITIAL_STATE_MISSING")
    require(torch.equal(corr["post_state"][start - 1], ctrl["post_state"][start - 1]), "PAIR_INITIAL_STATE_NOT_EQUAL")
    cg, cw, ng, nw = equalized_terms(
        corr["G"], corr["W"], ctrl["G"], ctrl["W"], start, end, condition
    )
    corr_out = structural_replay(corr["post_state"][start - 1], cg, cw, start, end)
    ctrl_out = structural_replay(ctrl["post_state"][start - 1], ng, nw, start, end)
    return corr_out, ctrl_out

def combine_natural_and_replay(
    natural_post: Mapping[int, Any],
    replayed: Mapping[int, Any],
    start: int,
    p: int,
) -> dict[int, Any]:
    required = range(p - 1, p + W + 1)
    out = {}
    for t in required:
        if t < start:
            require(t in natural_post, "NATURAL_PREINTERVENTION_STATE_MISSING")
            out[t] = natural_post[t]
        else:
            require(t in replayed, "REPLAYED_STATE_MISSING")
            out[t] = replayed[t]
    return out

def branch_metrics(k2s: Any, states: Mapping[int, Any], p: int) -> dict[str, Any]:
    return k2s.compute_branch_metrics(states, p)

def x_pair_specificity(
    branch_metrics_by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, float | None]:
    required = {"matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl"}
    require(set(branch_metrics_by_name) == required, "BRANCH_SET_INVALID")
    fields = {
        "R": "R_mean_speed",
        "D": "D_mean_turn",
        "DISP": "displacement",
        "P": "P_efficiency",
    }
    out = {}
    for metric, field in fields.items():
        mc, mn = branch_metrics_by_name["matched_corr"][field], branch_metrics_by_name["matched_ctrl"][field]
        sc, sn = branch_metrics_by_name["swapped_corr"][field], branch_metrics_by_name["swapped_ctrl"][field]
        if None in (mc, mn, sc, sn):
            out[metric] = None
        else:
            dm = float(mc - mn)
            ds = float(sc - sn)
            value = abs(dm) - abs(ds)
            require(math.isfinite(value), "X_NONFINITE")
            out[metric] = value
    return out

def aligned_signal(metric: str, block_value: float | None) -> float | None:
    require(metric in METRICS, "METRIC_INVALID")
    return None if block_value is None else EXPECTED_DIRECTION[metric] * float(block_value)

def attenuation_and_selectivity(
    metric: str,
    z_base: float | None,
    z_w_eq: float | None,
    z_g_eq: float | None,
) -> dict[str, float | None]:
    require(metric in METRICS, "METRIC_INVALID")
    if None in (z_base, z_w_eq, z_g_eq):
        return {"ATT_W": None, "ATT_G": None, "ATT_DOM": None, "ATT_OTHER": None, "SEL": None}
    att_w = float(z_base - z_w_eq)
    att_g = float(z_base - z_g_eq)
    if DOMINANT_COMPONENT[metric] == "W":
        att_dom, att_other = att_w, att_g
    else:
        att_dom, att_other = att_g, att_w
    return {
        "ATT_W": att_w,
        "ATT_G": att_g,
        "ATT_DOM": att_dom,
        "ATT_OTHER": att_other,
        "SEL": float(att_dom - att_other),
    }

def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    require(positive >= 0 and negative >= 0, "SIGN_COUNTS_INVALID")
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    numerator = sum(math.comb(n, k) for k in range(tail + 1))
    return min(1.0, 2.0 * numerator / (2 ** n))

def summarize_test(values: Sequence[float | None]) -> dict[str, Any]:
    undefined = sum(v is None for v in values)
    valid = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in valid), "PRIMARY_VALUE_NONFINITE")
    pos = sum(v > 0.0 for v in valid)
    neg = sum(v < 0.0 for v in valid)
    zero = sum(v == 0.0 for v in valid)
    n_eff = pos + neg
    floor = len(valid) >= 120 and n_eff >= 30
    raw = exact_two_sided_sign_p(pos, neg) if floor else 1.0
    effect = None if n_eff == 0 else (pos - neg) / n_eff
    return {
        "n_valid": len(valid),
        "n_eff": n_eff,
        "positive_count": pos,
        "negative_count": neg,
        "zero_count": zero,
        "undefined_count": undefined,
        "promotion_floor_pass": floor,
        "raw_p": raw,
        "rank_biserial_sign_effect": effect,
    }

def holm_adjust_eight(raw: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    require(set(raw) == set(PRIMARY_TEST_ORDER), "HOLM_PRIMARY_SET_MISMATCH")
    tie = {name: i for i, name in enumerate(PRIMARY_TEST_ORDER)}
    ordered = sorted(PRIMARY_TEST_ORDER, key=lambda name: (float(raw[name]), tie[name]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    require(m == 8, "HOLM_M_MISMATCH")
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(raw[name]))
        running = max(running, candidate)
        adjusted[name] = min(1.0, running)
    return {
        name: {
            "raw_p": float(raw[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "holm_reject": bool(adjusted[name] <= 0.05),
        }
        for name in PRIMARY_TEST_ORDER
    }

def k3_primary_statistics(values: Mapping[str, Sequence[float | None]]) -> dict[str, Any]:
    require(set(values) == set(PRIMARY_TEST_ORDER), "PRIMARY_TEST_VALUES_SET_MISMATCH")
    tests = {name: summarize_test(values[name]) for name in PRIMARY_TEST_ORDER}
    holm = holm_adjust_eight({name: tests[name]["raw_p"] for name in PRIMARY_TEST_ORDER})
    matches, contradictions = [], []
    for name in PRIMARY_TEST_ORDER:
        tests[name].update(holm[name])
        effect = tests[name]["rank_biserial_sign_effect"]
        match = bool(
            tests[name]["promotion_floor_pass"]
            and tests[name]["holm_reject"]
            and effect is not None
            and effect > 0
        )
        contradiction = bool(
            tests[name]["promotion_floor_pass"]
            and tests[name]["holm_reject"]
            and effect is not None
            and effect < 0
        )
        tests[name]["direction_match"] = match
        tests[name]["directional_contradiction"] = contradiction
        if match:
            matches.append(name)
        if contradiction:
            contradictions.append(name)
    full = matches == list(PRIMARY_TEST_ORDER)
    if full:
        verdict = SUCCESS_VERDICT
    elif contradictions:
        verdict = CONTRADICTION_VERDICT
    else:
        verdict = NOT_ESTABLISHED_VERDICT
    return {
        "primary_test_order": list(PRIMARY_TEST_ORDER),
        "holm_m": 8,
        "holm_alpha": 0.05,
        "tests": tests,
        "direction_matched_tests": matches,
        "directional_contradiction_tests": contradictions,
        "full_support": full,
        "scientific_verdict": verdict,
    }

def _branch_capture(
    model: Any,
    token_ids: Sequence[int],
    component_binding: ComponentBinding,
    component_indices: Iterable[int],
    state_indices: Iterable[int],
) -> dict[str, Any]:
    import torch
    mixer = model.mamba.layers[PRIMARY_LAYER].mixer
    collector = ComponentCollector(component_binding, mixer, component_indices, state_indices)
    input_tensor = torch.tensor(list(token_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode(), collector.capture():
        output = model.mamba(input_ids=input_tensor)
    for t, tensor in collector.G.items():
        validate_component_tensor(tensor, mixer, f"G_{t}")
    for t, tensor in collector.W.items():
        validate_component_tensor(tensor, mixer, f"W_{t}")
    for t, tensor in collector.pre_state.items():
        validate_component_tensor(tensor, mixer, f"PRE_{t}")
    for t, tensor in collector.post_state.items():
        validate_component_tensor(tensor, mixer, f"POST_{t}")
    return {
        "G": collector.G,
        "W": collector.W,
        "pre_state": collector.pre_state,
        "post_state": collector.post_state,
        "last_hidden_state": output.last_hidden_state.detach().cpu().clone(),
    }

def run_replay_preflight(
    root: Path,
    k2s: Any,
    model: Any,
    tokenizer: Any,
    component_binding: ComponentBinding,
) -> dict[str, Any]:
    import torch

    prefix = "Claim: synthetic k3 blorp\nEvidence: synthetic k3 snarp\nAdditional evidence:\n"
    corr = " synthetic corrective alpha beta gamma delta epsilon zeta eta theta iota kappa."
    ctrl = " synthetic control lambda mu nu xi omicron pi rho sigma tau upsilon."
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    corr_ids = list(tokenizer(prefix + corr, add_special_tokens=False)["input_ids"])
    ctrl_ids = list(tokenizer(prefix + ctrl, add_special_tokens=False)["input_ids"])
    require(corr_ids[: len(prefix_ids)] == prefix_ids == ctrl_ids[: len(prefix_ids)], "SYNTHETIC_PREFIX_TOKEN_MISMATCH")
    p = len(prefix_ids) - 1
    require(p >= 1, "SYNTHETIC_PREFIX_TOO_SHORT")
    require(len(corr_ids) >= p + W + 1 and len(ctrl_ids) >= p + W + 1, "SYNTHETIC_WINDOW_TOO_SHORT")
    d = first_divergence(corr_ids, ctrl_ids, len(prefix_ids), p + W)
    end = p + W
    component_indices = range(d, end + 1)
    state_indices = range(p - 1, end + 1)

    # Noninterference: ordinary task logits must be bit-exact with component trace enabled.
    bundle = k2s.task_mask_bundle(tokenizer, prefix + corr)
    baseline = k2s._full_model_forward(model, bundle)
    baseline_logits = k2s._logits(baseline).detach().cpu().clone()
    mixer = model.mamba.layers[PRIMARY_LAYER].mixer
    trace = ComponentCollector(component_binding, mixer, component_indices, state_indices)
    with trace.capture():
        traced = k2s._full_model_forward(model, bundle)
    traced_logits = k2s._logits(traced).detach().cpu()
    require(torch.equal(baseline_logits, traced_logits), "COMPONENT_TRACE_LOGIT_NONINTERFERENCE_FAILURE")

    corr_cap = _branch_capture(model, corr_ids, component_binding, component_indices, state_indices)
    ctrl_cap = _branch_capture(model, ctrl_ids, component_binding, component_indices, state_indices)
    require(
        set(corr_cap["G"]) == set(component_indices)
        and set(corr_cap["W"]) == set(component_indices)
        and set(ctrl_cap["G"]) == set(component_indices)
        and set(ctrl_cap["W"]) == set(component_indices),
        "SYNTHETIC_COMPONENT_CAPTURE_INCOMPLETE",
    )
    require(
        set(corr_cap["post_state"]) == set(state_indices)
        and set(ctrl_cap["post_state"]) == set(state_indices),
        "SYNTHETIC_STATE_CAPTURE_INCOMPLETE",
    )
    require(torch.equal(corr_cap["post_state"][d - 1], ctrl_cap["post_state"][d - 1]), "SYNTHETIC_D_MINUS_1_STATE_MISMATCH")

    # Captured recurrence pre-state must equal post state at t-1 where both are available.
    for branch in (corr_cap, ctrl_cap):
        for t in component_indices:
            if t - 1 in branch["post_state"]:
                require(torch.equal(branch["pre_state"][t], branch["post_state"][t - 1]), "PRE_POST_RECURRENCE_COORDINATE_MISMATCH")

    # Natural structural replay exact.
    for branch in (corr_cap, ctrl_cap):
        natural = structural_replay(
            branch["post_state"][d - 1], branch["G"], branch["W"], d, end
        )
        for t in range(d, end + 1):
            require(torch.equal(natural[t], branch["post_state"][t]), "NATURAL_STRUCTURAL_REPLAY_FAILURE")

        sham_g = {t: branch["G"][t].clone() for t in component_indices}
        sham_w = {t: branch["W"][t].clone() for t in component_indices}
        sham = structural_replay(branch["post_state"][d - 1], sham_g, sham_w, d, end)
        for t in range(d, end + 1):
            require(torch.equal(sham[t], branch["post_state"][t]), "SHAM_REPLAY_FAILURE")

    # GW equalization must collapse paired state exactly from d onward.
    gw_corr, gw_ctrl = replay_pair(corr_cap, ctrl_cap, d, end, "GW_EQ")
    for t in range(d, end + 1):
        require(torch.equal(gw_corr[t], gw_ctrl[t]), "GW_EQ_PAIR_STATE_COLLAPSE_FAILURE")

    # W_EQ and G_EQ must be executable and finite.
    for condition in ("W_EQ", "G_EQ"):
        a, b = replay_pair(corr_cap, ctrl_cap, d, end, condition)
        for t in range(d, end + 1):
            require(bool(torch.isfinite(a[t]).all().item()), f"{condition}_CORR_NONFINITE")
            require(bool(torch.isfinite(b[t]).all().item()), f"{condition}_CTRL_NONFINITE")

    # Known-term algebraic control independent of model values.
    g0 = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    w0 = torch.full((1, 2, 2), 1.0, dtype=torch.float32)
    init = torch.zeros((1, 2, 2), dtype=torch.float32)
    known = structural_replay(init, {0: g0, 1: g0}, {0: w0, 1: w0}, 0, 1)
    require(torch.equal(known[0], torch.ones_like(init)), "KNOWN_REPLAY_STEP0_FAILURE")
    require(torch.equal(known[1], torch.full_like(init, 1.5)), "KNOWN_REPLAY_STEP1_FAILURE")

    return {
        "status": "PASS_K3_REPLAY_PREFLIGHT",
        "scientific_population_intervention_executed": False,
        "component_capture_noninterference": "PASS_EXACT",
        "natural_structural_replay": "PASS_EXACT",
        "sham_replay": "PASS_EXACT",
        "gw_eq_pair_state_collapse": "PASS_EXACT",
        "synthetic_known_term_replay": "PASS_EXACT",
        "capture_layer": PRIMARY_LAYER,
        "synthetic_prefix_token_count": len(prefix_ids),
        "synthetic_corr_token_count": len(corr_ids),
        "synthetic_ctrl_token_count": len(ctrl_ids),
        "synthetic_p": p,
        "synthetic_d": d,
        "synthetic_d_minus_p": d - p,
        "synthetic_replay_end": end,
        "component_dtype": "torch.float32",
        "device": "cpu",
        "recurrence_source": {
            "qualname": component_binding.qualname,
            "path": str(component_binding.source_path),
            "sha256": component_binding.source_sha256,
            "bytes": component_binding.source_bytes,
            "discrete_A_line": component_binding.discrete_A_line,
            "deltaB_u_line": component_binding.deltaB_u_line,
            "recurrence_update_line": component_binding.recurrence_update_line,
            "post_update_line": component_binding.post_update_line,
        },
    }

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="K3 selective-SSM retention-vs-write causal decomposition")
    p.add_argument("--seed180-handoff", required=True)
    p.add_argument("--hf-revision", required=True)
    p.add_argument("--replay-preflight", action="store_true")
    return p

def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(args.replay_preflight, "K3_SCIENTIFIC_EXECUTION_NOT_AUTHORIZED_IN_IMPLEMENTATION_PHASE")
    root = Path(__file__).resolve().parents[1]
    runtime = git_provenance(root, replay_preflight=True)
    population = audit_population_without_outcomes(root)
    k2r, k2s = load_frozen_dependencies(root)

    snapshot, hf = k2s.resolve_hf_snapshot(args.hf_revision)
    require(hf["hf_model_id"] == HF_MODEL and hf["resolved_hf_revision"] == HF_REVISION, "HF_IDENTITY_MISMATCH")
    require(hf["transformers_version"] == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")

    handoff = k2s.audit_handoff(Path(args.seed180_handoff))
    require(handoff["zip_sha256"] == EXPECTED_ZIP_SHA256, "HANDOFF_ZIP_SHA_MISMATCH")
    require(handoff["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "CHECKPOINT_SHA_MISMATCH")
    checkpoint = k2s.load_authenticated_checkpoint(handoff)
    encoder = k2s.encoder_fingerprint(checkpoint["model_state_dict"])
    require(encoder["canonical_digest"] == COMMON_ENCODER_CANONICAL_SHA256, "ENCODER_CANONICAL_SHA_MISMATCH")
    require(encoder["raw_concat_digest"] == COMMON_ENCODER_RAW_CONCAT_SHA256, "ENCODER_RAW_SHA_MISMATCH")
    model = k2s.build_a0_model(root, snapshot, checkpoint)
    model.eval()

    # CPU is part of the frozen scientific/replay semantics.
    first_parameter = next(model.parameters())
    require(first_parameter.device.type == "cpu", "K3_CPU_REQUIRED")

    binding = resolve_component_binding(k2s)
    preflight = run_replay_preflight(root, k2s, model, hf["tokenizer"], binding)

    result = {
        "k3_replay_preflight": "PASS",
        "runtime": runtime,
        "population": population,
        "handoff": {
            **handoff,
            "encoder": encoder,
            "strict_load": "PASS",
        },
        "hf": {
            key: value
            for key, value in hf.items()
            if key not in {"config", "tokenizer"}
        },
        "dependency_binding": {
            "k2r_implementation_commit": K2R_IMPLEMENTATION_COMMIT,
            "k2r_runner_sha256": K2R_RUNNER_SHA256,
            "k2r_runner_git_blob": K2R_RUNNER_GIT_BLOB,
            "k2s_runner_sha256": K2S_RUNNER_SHA256,
            "k2s_runner_git_blob": K2S_RUNNER_GIT_BLOB,
            "k2r_result_archive_commit": K2R_RESULT_ARCHIVE_COMMIT,
        },
        "replay": preflight,
        "execution_boundary": {
            "k3_design_frozen": True,
            "k3_implementation_validation_authorized": True,
            "k3_scientific_intervention_execution_authorized": False,
            "scientific_cli_reachable": False,
        },
    }
    print(canonical_json(result).decode("utf-8"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
