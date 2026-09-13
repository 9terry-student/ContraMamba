"""Bounded Gen4 NAME Q1/Q3 dual-layer native-state measurement implementation."""

from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import reason_router_gen4_native_mamba_state_measurement as primary


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_AUTHORITY_COMMIT = "c395e634448a3b37c30053d89abef37c5a269afe"
PRIMARY_MEASUREMENT_PATH = ROOT / "scripts" / "reason_router_gen4_native_mamba_state_measurement.py"
PRIMARY_MEASUREMENT_SHA256 = (
    "7729424f03058b86b4f120dc0e6da573"
    "d6c996b0877858f2d6d38aa94dac268c"
)

Q1_LAYER_INDEX = 5
Q3_LAYER_INDEX = 17
ALLOWED_SECONDARY_LAYERS = frozenset((Q1_LAYER_INDEX, Q3_LAYER_INDEX))
STATE_SHAPE = primary.STATE_SHAPE
FLATTENED_STATE_SIZE = primary.FLATTENED_STATE_SIZE

ContractError = primary.ContractError
require = primary.require


def _canonical_worktree_bytes(path: Path) -> bytes:
    """Return Git-canonical LF bytes while rejecting unexplained lone CR bytes."""
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    require(b"\r" not in normalized, "primary measurement line ending ambiguity")
    return normalized


def _primary_identity() -> None:
    require(PRIMARY_MEASUREMENT_PATH.is_file(), "primary measurement missing")
    observed = hashlib.sha256(
        _canonical_worktree_bytes(PRIMARY_MEASUREMENT_PATH)
    ).hexdigest()
    require(
        observed == PRIMARY_MEASUREMENT_SHA256,
        "primary measurement byte identity",
    )


def validate_secondary_layer_registration(
    registered_layers: Mapping[int, Mapping[str, Any]],
) -> None:
    _primary_identity()
    require(isinstance(registered_layers, Mapping), "secondary registration")

    items = list(registered_layers.items())
    require(len(items) == 2, "exactly two secondary mixers")

    mixer_ids: list[int] = []
    layer_indices: list[int] = []

    for mixer_id, descriptor in items:
        require(type(mixer_id) is int, "mixer identity")
        require(isinstance(descriptor, Mapping), "layer descriptor")
        layer_index = descriptor.get("layer_index")
        require(type(layer_index) is int, "duplicate/invalid layer identity")
        mixer_ids.append(mixer_id)
        layer_indices.append(layer_index)

    require(len(set(mixer_ids)) == 2, "duplicate mixer identity")
    require(len(set(layer_indices)) == 2, "duplicate layer identity")
    require(
        set(layer_indices) == ALLOWED_SECONDARY_LAYERS,
        "exact Q1/Q3 layer set",
    )


class DualLayerNativeStateObserver:
    """One observer that captures layer 5 and layer 17 in one forward."""

    def __init__(
        self,
        registered_layers: Mapping[int, Mapping[str, Any]],
        enabled: bool = False,
        *,
        synthetic_binding: Any | None = None,
    ) -> None:
        validate_secondary_layer_registration(registered_layers)

        if synthetic_binding is None:
            code, capture_line = primary._resolve_and_validate_runtime_binding()
        else:
            require(
                isinstance(
                    synthetic_binding,
                    primary._SyntheticTraceBinding,
                ),
                "synthetic trace binding",
            )
            code = synthetic_binding.code
            capture_line = synthetic_binding.capture_line

        self._collector = primary._TraceCollector(
            code,
            capture_line,
            registered_layers,
            enabled=enabled,
        )
        self.enabled = bool(enabled)

    @property
    def snapshots(self):
        return self._collector.snapshots

    @contextmanager
    def capture(self):
        with self._collector.capture():
            yield self


def synthetic_observer(
    function: Any,
    capture_line: int,
    registered_layers: Mapping[int, Mapping[str, Any]],
    enabled: bool = True,
) -> DualLayerNativeStateObserver:
    binding = primary._synthetic_trace_binding(function, capture_line)
    return DualLayerNativeStateObserver(
        registered_layers,
        enabled=enabled,
        synthetic_binding=binding,
    )


def validate_capture_coordinates(
    snapshots: Mapping[tuple[int, int, int], Any],
    token_count: int,
) -> dict[int, list[Any]]:
    require(type(token_count) is int and token_count > 0, "invalid token count")
    require(
        isinstance(snapshots, Mapping) and bool(snapshots),
        "empty capture",
    )

    keys = list(snapshots)
    require(
        all(
            isinstance(key, tuple)
            and len(key) == 3
            and all(type(value) is int for value in key)
            for key in keys
        ),
        "capture coordinate schema",
    )

    forward_ids = {key[0] for key in keys}
    require(len(forward_ids) == 1, "ambiguous forward identity")
    forward_id = next(iter(forward_ids))

    expected = [
        (forward_id, layer_index, token_index)
        for layer_index in (Q1_LAYER_INDEX, Q3_LAYER_INDEX)
        for token_index in range(token_count)
    ]
    require(keys == expected, "complete ordered capture coordinates")

    result: dict[int, list[Any]] = {}
    for layer_index in (Q1_LAYER_INDEX, Q3_LAYER_INDEX):
        layer_keys = [
            (forward_id, layer_index, token_index)
            for token_index in range(token_count)
        ]
        values = [snapshots[key] for key in layer_keys]

        # Validate each layer independently. This proves shape/dtype/device
        # rather than inferring secondary-layer compatibility from layer 11.
        for value in values:
            flatten_scientific_state(value)

        result[layer_index] = values

    return result


def flatten_scientific_state(state: Any) -> np.ndarray:
    return primary.flatten_scientific_state(state)


def endpoint_from_same_layer(
    states: Sequence[Any],
    anchor: int,
    layer_index: int,
) -> dict[str, float]:
    require(layer_index in ALLOWED_SECONDARY_LAYERS, "Q1/Q3 layer")
    require(type(anchor) is int and anchor >= 1, "invalid anchor")
    require(
        isinstance(states, Sequence) and len(states) > anchor + 4,
        "same-layer endpoint states",
    )

    vectors = [flatten_scientific_state(state) for state in states]

    return {
        "POST4_SPEED": primary.post4_speed(vectors, anchor),
        "POST4_TURNING": primary.post4_turning(vectors, anchor),
        "POST4_PATH_EFFICIENCY": primary.post4_path_efficiency(vectors, anchor),
    }


def capture_model_row(model: Any, input_ids: Any) -> dict[int, list[Any]]:
    """Future canonical primitive: exactly one model.mamba forward, two layers."""
    import torch

    require(hasattr(model, "mamba"), "missing Mamba backbone")
    layers = getattr(model.mamba, "layers", None)
    require(layers is not None and len(layers) > Q3_LAYER_INDEX, "Mamba layer count")

    registered = {
        id(layers[Q1_LAYER_INDEX].mixer): {"layer_index": Q1_LAYER_INDEX},
        id(layers[Q3_LAYER_INDEX].mixer): {"layer_index": Q3_LAYER_INDEX},
    }

    if getattr(input_ids, "ndim", None) == 1:
        input_ids = input_ids.unsqueeze(0)

    require(tuple(input_ids.shape) == (1, 128), "future input shape")

    observer = DualLayerNativeStateObserver(registered, enabled=True)
    with observer.capture(), torch.inference_mode():
        model.mamba(input_ids=input_ids)

    require(observer.snapshots is not None, "missing dual-layer capture")
    return validate_capture_coordinates(observer.snapshots, 128)


def observer_default_enabled_value() -> bool:
    """Static helper used by validation to prove default-disabled policy."""
    default = inspect.signature(
        DualLayerNativeStateObserver.__init__
    ).parameters["enabled"].default
    require(default is False, "observer default-enabled drift")
    return bool(default)
