"""Bounded Gen4 native-Mamba-state measurement implementation.

Phase C only:
- exact runtime/source validation for the frozen future scientific runtime;
- default-disabled post-update native recurrent-state observation;
- primary-layer binding;
- deterministic local POST4 kinematics;
- no scientific model execution or artifact publication.

Importing this module does not import torch or transformers.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PHASE_C_AUTHORITY_COMMIT = (
    "480ff74aebf5ef942aa9f47fa78c06612a8f97a4"
)
PHASE_AB_FEASIBILITY_COMMIT = (
    "26fd55803acd05febefc8bd031f2fc23c17b0ef4"
)

EXPECTED_VERSIONS = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cpu",
    "transformers": "5.0.0",
}

MAMBA_MODULE = "transformers.models.mamba.modeling_mamba"
CACHE_MODULE = "transformers.cache_utils"

MAMBA_SHA256 = (
    "4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83"
)
MAMBA_BYTES = 39500

CACHE_SHA256 = (
    "6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc"
)
CACHE_BYTES = 60432

CAPTURE_QUALNAME = "MambaMixer.slow_forward"
RECURRENT_UPDATE_LINE = 409
CAPTURE_LINE = 410
FINAL_CACHE_PERSISTENCE_LINE = 417

PRIMARY_LAYER_INDEX = 11
NATIVE_MAMBA_LAYER_COUNT = 24

STATE_SOURCE = "native_selective_ssm_recurrent_state"
STATE_TIMING = "post_consumption_s_t"

STATE_SHAPE = (1, 1536, 16)
PER_EXAMPLE_STATE_SHAPE = (1536, 16)
FLATTENED_STATE_SIZE = 24576

BLOCKED_ZERO_TRANSITION = (
    "BLOCKED_UNDEFINED_ZERO_TRANSITION_NORM"
)
BLOCKED_ZERO_PATH = (
    "BLOCKED_UNDEFINED_ZERO_PATH_LENGTH"
)
BLOCKED_SCIENTIFIC_EXECUTION = (
    "PHASE_C_SCIENTIFIC_EXECUTION_PROHIBITED"
)


class ContractError(RuntimeError):
    """Fail-closed Phase C contract violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(module: Any) -> tuple[Path, bytes]:
    path = Path(getattr(module, "__file__", "")).resolve()
    require(path.is_file(), "source root")
    return path, path.read_bytes()


def _node_names(node: Any) -> set[str]:
    if node is None:
        return set()
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
    }


def _assignment_target(node: Any) -> str:
    if isinstance(node, ast.Assign):
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return ""

    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _assignment_value(node: Any) -> Any:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _validate_source_roles(data: bytes, code: Any) -> None:
    """Prove update -> readout -> recurrent-cache persistence roles."""
    try:
        tree = ast.parse(data.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ContractError("source role syntax") from exc

    nodes_by_line: dict[int, list[Any]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            line = getattr(node, "lineno", None)
            if isinstance(line, int):
                nodes_by_line.setdefault(line, []).append(node)

    updates = nodes_by_line.get(RECURRENT_UPDATE_LINE, [])
    readouts = nodes_by_line.get(CAPTURE_LINE, [])
    persists = nodes_by_line.get(FINAL_CACHE_PERSISTENCE_LINE, [])

    require(len(updates) == 1, "source update role")
    require(len(readouts) == 1, "source readout role")
    require(len(persists) == 1, "source recurrent cache role")

    update = updates[0]
    readout = readouts[0]
    persist = persists[0]

    update_value = _assignment_value(update)
    update_names = _node_names(update_value)

    require(
        isinstance(update, (ast.Assign, ast.AnnAssign))
        and _assignment_target(update) == "ssm_state"
        and isinstance(update_value, ast.BinOp)
        and isinstance(update_value.op, ast.Add)
        and {
            "ssm_state",
            "discrete_A",
            "discrete_B",
            "hidden_states",
        }
        <= update_names,
        "source update role",
    )

    readout_value = _assignment_value(readout)
    readout_names = _node_names(readout_value)

    require(
        isinstance(readout, (ast.Assign, ast.AnnAssign))
        and _assignment_target(readout) not in {"", "ssm_state"}
        and {"ssm_state", "discrete_C"} <= readout_names,
        "source readout role",
    )

    call = persist.value if isinstance(persist, ast.Expr) else None

    cache_target = (
        call.func.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "copy_"
        )
        else None
    )

    require(
        isinstance(cache_target, ast.Subscript)
        and isinstance(cache_target.value, ast.Attribute)
        and cache_target.value.attr == "ssm_states"
        and isinstance(cache_target.value.value, ast.Name)
        and cache_target.value.value.id == "cache_params"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "ssm_state",
        "source recurrent cache role",
    )

    line_events = {
        line
        for _, _, line in code.co_lines()
        if line is not None
    }

    require(
        CAPTURE_LINE in line_events,
        "line binding",
    )


def _runtime_environment() -> tuple[
    Mapping[str, Any],
    Mapping[str, str],
]:
    """Only boundary that imports torch/Transformers."""
    modules = {
        MAMBA_MODULE: importlib.import_module(MAMBA_MODULE),
        CACHE_MODULE: importlib.import_module(CACHE_MODULE),
        "transformers": importlib.import_module("transformers"),
        "torch": importlib.import_module("torch"),
    }

    versions = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": modules["torch"].__version__,
        "transformers": modules["transformers"].__version__,
    }

    return modules, versions


def _resolve_and_validate_runtime_binding() -> tuple[Any, int]:
    modules, versions = _runtime_environment()

    require(
        set(versions) == set(EXPECTED_VERSIONS)
        and dict(versions) == EXPECTED_VERSIONS,
        "runtime version",
    )

    mamba_path, mamba_bytes = _source(
        modules[MAMBA_MODULE]
    )
    cache_path, cache_bytes = _source(
        modules[CACHE_MODULE]
    )

    root_text = getattr(
        modules["transformers"],
        "__file__",
        None,
    )

    require(
        isinstance(root_text, str)
        and Path(root_text).is_absolute(),
        "malformed import root",
    )

    import_root = Path(root_text).resolve().parent

    require(
        import_root.is_dir(),
        "transformers import root",
    )

    require(
        mamba_path.is_relative_to(import_root)
        and cache_path.is_relative_to(import_root),
        "shadowed import root",
    )

    try:
        distribution_root = Path(
            importlib_metadata.distribution(
                "transformers"
            ).locate_file("transformers")
        ).resolve()
    except importlib_metadata.PackageNotFoundError as exc:
        raise ContractError(
            "transformers distribution unavailable"
        ) from exc

    require(
        distribution_root == import_root,
        "import/distribution-root mismatch",
    )

    for module_name, expected_path in (
        (MAMBA_MODULE, mamba_path),
        (CACHE_MODULE, cache_path),
    ):
        spec = importlib.util.find_spec(module_name)

        require(
            spec is None
            or Path(str(spec.origin)).resolve()
            == expected_path,
            "import/distribution-root mismatch",
        )

    require(
        len(mamba_bytes) == MAMBA_BYTES,
        "Mamba byte-size mismatch",
    )

    require(
        sha256_bytes(mamba_bytes) == MAMBA_SHA256,
        "Mamba SHA256 mismatch",
    )

    require(
        len(cache_bytes) == CACHE_BYTES,
        "cache_utils byte-size mismatch",
    )

    require(
        sha256_bytes(cache_bytes) == CACHE_SHA256,
        "cache_utils SHA256 mismatch",
    )

    mixer = getattr(
        modules[MAMBA_MODULE],
        "MambaMixer",
        None,
    )

    slow_forward = getattr(
        mixer,
        "slow_forward",
        None,
    )

    forward = getattr(
        mixer,
        "forward",
        None,
    )

    require(
        inspect.isfunction(slow_forward)
        and slow_forward.__module__ == MAMBA_MODULE
        and slow_forward.__qualname__ == CAPTURE_QUALNAME
        and slow_forward.__code__.co_filename
        == str(mamba_path),
        "slow code identity",
    )

    _validate_source_roles(
        mamba_bytes,
        slow_forward.__code__,
    )

    require(
        inspect.isfunction(forward)
        and forward.__module__ == MAMBA_MODULE,
        "forward identity",
    )

    dispatch = "".join(
        inspect.getsourcelines(forward)[0]
    )

    require(
        "slow_forward" in dispatch
        and "mamba_inner_fn" not in dispatch,
        "unsupported backend",
    )

    cache_text = cache_bytes.decode("utf-8")

    require(
        "conv_states" in cache_text
        and "ssm_states" in cache_text
        and cache_text.find("conv_states")
        != cache_text.find("ssm_states"),
        "cache/recurrent ambiguity",
    )

    return slow_forward.__code__, CAPTURE_LINE


def runtime_gate() -> None:
    """Validate future scientific runtime without exposing binding."""
    _resolve_and_validate_runtime_binding()


@dataclass(frozen=True)
class _SyntheticTraceBinding:
    function: Any
    code: Any
    capture_line: int


def _synthetic_trace_binding(
    function: Any,
    capture_line: int,
) -> _SyntheticTraceBinding:
    require(
        inspect.isfunction(function)
        and type(capture_line) is int
        and capture_line > 0,
        "synthetic binding",
    )

    return _SyntheticTraceBinding(
        function=function,
        code=function.__code__,
        capture_line=capture_line,
    )


def _tensor_is_finite(state: Any) -> bool:
    method = getattr(state, "isfinite", None)

    if callable(method):
        value = method()
        all_method = getattr(value, "all", None)

        if callable(all_method):
            value = all_method()

        item_method = getattr(value, "item", None)

        if callable(item_method):
            value = item_method()

        return bool(value)

    try:
        detached = (
            state.detach()
            if hasattr(state, "detach")
            else state
        )

        cpu = (
            detached.cpu()
            if hasattr(detached, "cpu")
            else detached
        )

        value = (
            cpu.numpy()
            if hasattr(cpu, "numpy")
            else cpu
        )

        return bool(
            np.isfinite(
                np.asarray(value)
            ).all()
        )
    except Exception:
        return False


class _TraceCollector:
    """Low-level observer; scientific policy lives separately."""

    def __init__(
        self,
        code: Any,
        capture_line: int,
        registered_layers: Mapping[
            int,
            Mapping[str, Any],
        ],
        enabled: bool = False,
    ):
        self.code = code
        self.capture_line = capture_line
        self.layers = {
            int(key): dict(value)
            for key, value
            in registered_layers.items()
        }
        self.enabled = bool(enabled)

        indexes = [
            descriptor.get("layer_index")
            for descriptor
            in self.layers.values()
        ]

        require(
            all(
                type(index) is int
                for index in indexes
            )
            and len(indexes)
            == len(set(indexes)),
            "duplicate/invalid layer identity",
        )

        self.snapshots: dict[
            tuple[int, int, int],
            Any,
        ] | None = None

        self._prior = None
        self._forward_id = 0

    def _trace(
        self,
        frame: Any,
        event: str,
        arg: Any,
    ):
        del arg

        if (
            frame.f_code is self.code
            and event == "line"
            and frame.f_lineno
            == self.capture_line
        ):
            descriptor = self.layers.get(
                id(
                    frame.f_locals.get(
                        "self"
                    )
                )
            )

            if descriptor is None:
                return self._trace

            index = frame.f_locals.get("i")

            require(
                type(index) is int
                and index >= 0,
                "ambiguous token index",
            )

            state = frame.f_locals.get(
                "ssm_state"
            )

            require(
                state is not None
                and hasattr(
                    state,
                    "detach",
                )
                and hasattr(
                    state,
                    "clone",
                ),
                "not native recurrent state",
            )

            shape = getattr(
                state,
                "shape",
                None,
            )

            require(
                shape is not None
                and len(tuple(shape)) == 3,
                "wrong recurrent-state rank",
            )

            require(
                _tensor_is_finite(state),
                "nonfinite recurrent state",
            )

            snapshot = (
                state.detach().clone()
            )

            require(
                snapshot is not state,
                "snapshot alias",
            )

            key = (
                self._forward_id,
                int(
                    descriptor[
                        "layer_index"
                    ]
                ),
                index,
            )

            require(
                self.snapshots is not None
                and key
                not in self.snapshots,
                "duplicate coordinate",
            )

            self.snapshots[key] = snapshot

        return self._trace

    @contextmanager
    def capture(self):
        if not self.enabled:
            yield self
            return

        require(
            self.snapshots is None,
            "observer reuse",
        )

        self.snapshots = {}
        self._forward_id += 1
        self._prior = sys.gettrace()

        sys.settrace(self._trace)

        try:
            yield self
        finally:
            sys.settrace(self._prior)
            self.layers.clear()


def _synthetic_collector(
    binding: _SyntheticTraceBinding,
    registered_layers: Mapping[
        int,
        Mapping[str, Any],
    ],
    enabled: bool = True,
) -> _TraceCollector:
    require(
        isinstance(
            binding,
            _SyntheticTraceBinding,
        ),
        "synthetic trace binding",
    )

    return _TraceCollector(
        binding.code,
        binding.capture_line,
        registered_layers,
        enabled,
    )


def validate_primary_layer_registration(
    registered_layers: Mapping[
        int,
        Mapping[str, Any],
    ],
) -> None:
    descriptors = [
        dict(value)
        for value
        in registered_layers.values()
    ]

    require(
        len(descriptors) == 1
        and descriptors[0].get(
            "layer_index"
        )
        == PRIMARY_LAYER_INDEX,
        "primary layer binding",
    )


class NativeStateObserver(_TraceCollector):
    """Future scientific observer with exact runtime gate."""

    def __init__(
        self,
        registered_layers: Mapping[
            int,
            Mapping[str, Any],
        ],
        enabled: bool = False,
    ):
        validate_primary_layer_registration(
            registered_layers
        )

        code, capture_line = (
            _resolve_and_validate_runtime_binding()
        )

        super().__init__(
            code,
            capture_line,
            registered_layers,
            enabled,
        )


def validate_capture_coordinates(
    snapshots: Mapping[
        tuple[int, int, int],
        Any,
    ],
    token_count: int,
    layer_index: int = PRIMARY_LAYER_INDEX,
) -> list[Any]:
    require(
        type(token_count) is int
        and token_count > 0,
        "invalid token count",
    )

    require(
        layer_index
        == PRIMARY_LAYER_INDEX,
        "primary layer binding",
    )

    require(
        isinstance(snapshots, Mapping)
        and bool(snapshots),
        "empty capture",
    )

    keys = list(snapshots)

    require(
        all(
            isinstance(key, tuple)
            and len(key) == 3
            and all(
                type(value) is int
                for value in key
            )
            for key in keys
        ),
        "capture coordinate schema",
    )

    forward_ids = {
        key[0]
        for key in keys
    }

    require(
        len(forward_ids) == 1,
        "ambiguous forward identity",
    )

    forward_id = next(
        iter(forward_ids)
    )

    expected = [
        (
            forward_id,
            PRIMARY_LAYER_INDEX,
            token_index,
        )
        for token_index
        in range(token_count)
    ]

    require(
        keys == expected,
        "complete ordered capture coordinates",
    )

    return [
        snapshots[key]
        for key in expected
    ]


def flatten_scientific_state(
    state: Any,
) -> np.ndarray:
    shape = tuple(
        getattr(
            state,
            "shape",
            (),
        )
    )

    require(
        shape == STATE_SHAPE,
        "scientific recurrent-state shape",
    )

    dtype = str(
        getattr(
            state,
            "dtype",
            "",
        )
    )

    device = str(
        getattr(
            state,
            "device",
            "",
        )
    )

    require(
        dtype in {
            "float32",
            "torch.float32",
        },
        "scientific recurrent-state dtype",
    )

    require(
        device == "cpu",
        "scientific recurrent-state device",
    )

    require(
        _tensor_is_finite(state),
        "nonfinite recurrent state",
    )

    value = state.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(
        value,
        dtype=np.float32,
    )

    require(
        array.shape == STATE_SHAPE,
        "scientific recurrent-state shape",
    )

    vector = np.ascontiguousarray(
        array[0].reshape(-1),
        dtype=np.float32,
    )

    require(
        vector.shape
        == (FLATTENED_STATE_SIZE,),
        "scientific recurrent-state flattened size",
    )

    require(
        np.isfinite(vector).all(),
        "nonfinite recurrent-state vector",
    )

    return vector.copy()


def _validated_vectors(
    states: Sequence[Any],
) -> list[np.ndarray]:
    require(
        isinstance(states, Sequence)
        and len(states) > 0,
        "empty state sequence",
    )

    vectors: list[np.ndarray] = []
    dimension = None

    for state in states:
        vector = np.asarray(
            state,
            dtype=np.float32,
        )

        require(
            vector.ndim == 1
            and vector.size > 0,
            "state vector shape",
        )

        require(
            np.isfinite(
                vector
            ).all(),
            "nonfinite state vector",
        )

        vector = np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

        if dimension is None:
            dimension = vector.size

        require(
            vector.size == dimension,
            "state vector dimension mismatch",
        )

        vectors.append(vector)

    return vectors


def _transition(
    vectors: Sequence[np.ndarray],
    token_index: int,
) -> np.ndarray:
    require(
        type(token_index) is int
        and 1 <= token_index
        < len(vectors),
        "transition coordinate",
    )

    transition = np.ascontiguousarray(
        vectors[token_index]
        - vectors[token_index - 1],
        dtype=np.float32,
    )

    require(
        np.isfinite(
            transition
        ).all(),
        "nonfinite transition",
    )

    return transition


def _norm(
    vector: np.ndarray,
) -> float:
    value = float(
        np.linalg.norm(vector)
    )

    require(
        np.isfinite(value),
        "nonfinite norm",
    )

    return value


def transition_speed(
    current_transition: Any,
) -> float:
    vector = np.asarray(
        current_transition,
        dtype=np.float32,
    )

    require(
        vector.ndim == 1
        and vector.size > 0
        and np.isfinite(
            vector
        ).all(),
        "transition vector",
    )

    return _norm(vector)


def transition_turning(
    current_transition: Any,
    previous_transition: Any,
) -> float:
    current = np.asarray(
        current_transition,
        dtype=np.float32,
    )

    previous = np.asarray(
        previous_transition,
        dtype=np.float32,
    )

    require(
        current.ndim == 1
        and previous.ndim == 1
        and current.shape
        == previous.shape
        and current.size > 0
        and np.isfinite(
            current
        ).all()
        and np.isfinite(
            previous
        ).all(),
        "transition vector",
    )

    current_norm = _norm(current)
    previous_norm = _norm(previous)

    if (
        current_norm == 0.0
        or previous_norm == 0.0
    ):
        raise ContractError(
            BLOCKED_ZERO_TRANSITION
        )

    cosine = float(
        np.dot(
            current,
            previous,
        )
        / (
            current_norm
            * previous_norm
        )
    )

    require(
        np.isfinite(cosine),
        "nonfinite cosine",
    )

    turning = 1.0 - cosine

    require(
        np.isfinite(turning),
        "nonfinite turning",
    )

    return float(turning)


def _validate_post4_window(
    vectors: Sequence[np.ndarray],
    anchor: int,
) -> None:
    terminal_index = (
        len(vectors) - 1
    )

    require(
        type(anchor) is int
        and anchor >= 1,
        "invalid anchor",
    )

    require(
        anchor + 4
        <= terminal_index - 1,
        "POST4 prefix eligibility",
    )


def post4_speed(
    states: Sequence[Any],
    anchor: int,
) -> float:
    vectors = _validated_vectors(
        states
    )

    _validate_post4_window(
        vectors,
        anchor,
    )

    values = [
        _norm(
            _transition(
                vectors,
                token_index,
            )
        )
        for token_index
        in range(
            anchor + 1,
            anchor + 5,
        )
    ]

    result = float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float32,
            )
        )
    )

    require(
        np.isfinite(result),
        "nonfinite POST4 speed",
    )

    return result


def post4_turning(
    states: Sequence[Any],
    anchor: int,
) -> float:
    vectors = _validated_vectors(
        states
    )

    _validate_post4_window(
        vectors,
        anchor,
    )

    transitions = {
        token_index: _transition(
            vectors,
            token_index,
        )
        for token_index
        in range(
            anchor,
            anchor + 5,
        )
    }

    values = [
        transition_turning(
            transitions[token_index],
            transitions[
                token_index - 1
            ],
        )
        for token_index
        in range(
            anchor + 1,
            anchor + 5,
        )
    ]

    result = float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float32,
            )
        )
    )

    require(
        np.isfinite(result),
        "nonfinite POST4 turning",
    )

    return result


def post4_path_efficiency(
    states: Sequence[Any],
    anchor: int,
) -> float:
    vectors = _validated_vectors(
        states
    )

    _validate_post4_window(
        vectors,
        anchor,
    )

    speeds = [
        _norm(
            _transition(
                vectors,
                token_index,
            )
        )
        for token_index
        in range(
            anchor + 1,
            anchor + 5,
        )
    ]

    denominator = float(
        np.sum(
            np.asarray(
                speeds,
                dtype=np.float32,
            )
        )
    )

    require(
        np.isfinite(denominator),
        "nonfinite path length",
    )

    if denominator == 0.0:
        raise ContractError(
            BLOCKED_ZERO_PATH
        )

    displacement = _norm(
        np.ascontiguousarray(
            vectors[anchor + 4]
            - vectors[anchor],
            dtype=np.float32,
        )
    )

    result = (
        displacement
        / denominator
    )

    require(
        np.isfinite(result),
        "nonfinite path efficiency",
    )

    return float(result)


def post4_kinematics(
    states: Sequence[Any],
    anchor: int,
) -> dict[str, float]:
    return {
        "POST4_SPEED":
            post4_speed(
                states,
                anchor,
            ),
        "POST4_TURNING":
            post4_turning(
                states,
                anchor,
            ),
        "POST4_PATH_EFFICIENCY":
            post4_path_efficiency(
                states,
                anchor,
            ),
    }


def scientific_extraction(
    *args: Any,
    **kwargs: Any,
) -> None:
    del args, kwargs

    raise ContractError(
        BLOCKED_SCIENTIFIC_EXECUTION
    )


def main(
    argv: Sequence[str] | None = None,
) -> None:
    del argv

    raise ContractError(
        BLOCKED_SCIENTIFIC_EXECUTION
    )


if __name__ == "__main__":
    main()
