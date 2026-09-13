from __future__ import annotations

import builtins
import hashlib
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_native_mamba_state_measurement.py"
)


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    try:
        spec.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            name,
            None,
        )
        raise

    return module


m = load_module(
    "reason_router_gen4_native_mamba_state_measurement_test_target"
)


class Flag:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class Tensor:
    def __init__(
        self,
        value,
        *,
        dtype="float32",
        device="cpu",
    ):
        self.v = np.asarray(
            value,
            dtype=np.float32,
        ).copy()

        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        return self.v.shape

    def detach(self):
        return self

    def clone(self):
        return Tensor(
            self.v.copy(),
            dtype=self.dtype,
            device=self.device,
        )

    def cpu(self):
        return self

    def numpy(self):
        return self.v.copy()

    def isfinite(self):
        return Flag(
            np.isfinite(
                self.v
            ).all()
        )


def source_line(
    function,
    needle: str,
) -> int:
    lines, start = inspect.getsourcelines(
        function
    )

    matches = [
        start + index
        for index, line
        in enumerate(lines)
        if needle in line
    ]

    assert len(matches) == 1

    return matches[0]


def test_authority_constants():
    assert (
        m.PHASE_C_AUTHORITY_COMMIT
        == "480ff74aebf5ef942aa9f47fa78c06612a8f97a4"
    )

    assert m.EXPECTED_VERSIONS == {
        "python": "3.12.13",
        "numpy": "2.0.2",
        "torch": "2.10.0+cu128",
        "transformers": "5.0.0",
    }

    assert m.PRIMARY_LAYER_INDEX == 11
    assert m.NATIVE_MAMBA_LAYER_COUNT == 24
    assert m.DELTA_B_U_LINE == 397
    assert m.RECURRENT_UPDATE_LINE == 409
    assert m.CAPTURE_LINE == 410
    assert m.FINAL_CACHE_PERSISTENCE_LINE == 417
    assert m.FLATTENED_STATE_SIZE == 24576


def test_import_safety_rejects_torch_transformers_import(
    monkeypatch,
):
    original = builtins.__import__

    def guarded(
        name,
        *args,
        **kwargs,
    ):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
        }:
            raise AssertionError(
                "model library import at module import"
            )

        return original(
            name,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        guarded,
    )

    fresh_name = (
        "reason_router_gen4_native_mamba_state_"
        "measurement_import_safety"
    )

    loaded = load_module(
        fresh_name
    )

    assert (
        loaded.PRIMARY_LAYER_INDEX
        == 11
    )

    sys.modules.pop(
        fresh_name,
        None,
    )


def synthetic_forward(self):
    ssm_state = Tensor(np.zeros((1, 2, 2)))
    for i in range(3):
        ssm_state = Tensor(ssm_state.v + (i + 1))
        readout = ssm_state.clone()
        self.live = ssm_state
    return readout


SYNTHETIC_CAPTURE_LINE = source_line(
    synthetic_forward,
    "readout = ssm_state.clone()",
)


def test_disabled_observer_is_inert():
    model = types.SimpleNamespace()

    binding = m._synthetic_trace_binding(
        synthetic_forward,
        SYNTHETIC_CAPTURE_LINE,
    )

    observer = m._synthetic_collector(
        binding,
        {
            id(model): {
                "layer_index": 11,
            }
        },
        enabled=False,
    )

    prior = sys.gettrace()

    with observer.capture():
        result = synthetic_forward(
            model
        )

    assert isinstance(
        result,
        Tensor,
    )

    assert observer.snapshots is None
    assert sys.gettrace() is prior


def test_post_update_capture_clone_and_coordinates():
    model = types.SimpleNamespace()

    binding = m._synthetic_trace_binding(
        synthetic_forward,
        SYNTHETIC_CAPTURE_LINE,
    )

    observer = m._synthetic_collector(
        binding,
        {
            id(model): {
                "layer_index": 11,
            }
        },
        enabled=True,
    )

    with observer.capture():
        synthetic_forward(
            model
        )

    assert (
        observer.snapshots
        is not None
    )

    ordered = m.validate_capture_coordinates(
        observer.snapshots,
        token_count=3,
    )

    expected = [
        1.0,
        3.0,
        6.0,
    ]

    for snapshot, value in zip(
        ordered,
        expected,
    ):
        assert np.array_equal(
            snapshot.v,
            np.full(
                (1, 2, 2),
                value,
                dtype=np.float32,
            ),
        )

    model.live.v[:] = 999.0

    assert np.array_equal(
        ordered[-1].v,
        np.full(
            (1, 2, 2),
            6.0,
            dtype=np.float32,
        ),
    )


def duplicate_forward(self):
    ssm_state = Tensor(np.zeros((1, 2, 2)))
    for i in (0, 0):
        ssm_state = Tensor(ssm_state.v + 1)
        readout = ssm_state.clone()
    return readout


DUPLICATE_CAPTURE_LINE = source_line(
    duplicate_forward,
    "readout = ssm_state.clone()",
)


def test_duplicate_coordinate_fails_closed():
    model = types.SimpleNamespace()

    observer = m._synthetic_collector(
        m._synthetic_trace_binding(
            duplicate_forward,
            DUPLICATE_CAPTURE_LINE,
        ),
        {
            id(model): {
                "layer_index": 11,
            }
        },
        enabled=True,
    )

    with pytest.raises(
        m.ContractError,
        match="duplicate coordinate",
    ):
        with observer.capture():
            duplicate_forward(
                model
            )


def failing_forward(self):
    ssm_state = Tensor(np.zeros((1, 2, 2)))
    for i in range(1):
        ssm_state = Tensor(ssm_state.v + 1)
        readout = ssm_state.clone()
        raise ValueError("synthetic")
    return readout


FAILING_CAPTURE_LINE = source_line(
    failing_forward,
    "readout = ssm_state.clone()",
)


def test_trace_restored_after_exception():
    model = types.SimpleNamespace()

    observer = m._synthetic_collector(
        m._synthetic_trace_binding(
            failing_forward,
            FAILING_CAPTURE_LINE,
        ),
        {
            id(model): {
                "layer_index": 11,
            }
        },
        enabled=True,
    )

    prior = sys.gettrace()

    with pytest.raises(
        ValueError,
        match="synthetic",
    ):
        with observer.capture():
            failing_forward(
                model
            )

    assert sys.gettrace() is prior


def nonfinite_forward(self):
    ssm_state = Tensor(np.zeros((1, 2, 2)))
    for i in range(1):
        ssm_state = Tensor(
            np.full(
                (1, 2, 2),
                np.nan,
                dtype=np.float32,
            )
        )
        readout = ssm_state.clone()
    return readout


NONFINITE_CAPTURE_LINE = source_line(
    nonfinite_forward,
    "readout = ssm_state.clone()",
)


def test_nonfinite_capture_fails_closed():
    model = types.SimpleNamespace()

    observer = m._synthetic_collector(
        m._synthetic_trace_binding(
            nonfinite_forward,
            NONFINITE_CAPTURE_LINE,
        ),
        {
            id(model): {
                "layer_index": 11,
            }
        },
        enabled=True,
    )

    with pytest.raises(
        m.ContractError,
        match="nonfinite recurrent state",
    ):
        with observer.capture():
            nonfinite_forward(
                model
            )


def test_capture_completeness_rejects_missing_and_order():
    states = {
        (1, 11, 0): Tensor(
            np.zeros((1, 2, 2))
        ),
        (1, 11, 2): Tensor(
            np.zeros((1, 2, 2))
        ),
    }

    with pytest.raises(
        m.ContractError,
        match="complete ordered capture coordinates",
    ):
        m.validate_capture_coordinates(
            states,
            token_count=3,
        )

    reordered = {
        (1, 11, 1): Tensor(
            np.zeros((1, 2, 2))
        ),
        (1, 11, 0): Tensor(
            np.zeros((1, 2, 2))
        ),
    }

    with pytest.raises(
        m.ContractError,
        match="complete ordered capture coordinates",
    ):
        m.validate_capture_coordinates(
            reordered,
            token_count=2,
        )


def test_primary_layer_binding_and_vectorization():
    good = Tensor(
        np.zeros(
            (1, 1536, 16),
            dtype=np.float32,
        )
    )

    vector = m.flatten_scientific_state(
        good
    )

    assert vector.shape == (24576,)
    assert vector.dtype == np.float32

    with pytest.raises(
        m.ContractError,
        match="primary layer binding",
    ):
        m.validate_primary_layer_registration(
            {
                1: {
                    "layer_index": 10,
                }
            }
        )

    with pytest.raises(
        m.ContractError,
        match="scientific recurrent-state shape",
    ):
        m.flatten_scientific_state(
            Tensor(
                np.zeros(
                    (1, 2, 2)
                )
            )
        )


def linear_states():
    return [
        np.array(
            [
                float(index),
                0.0,
            ],
            dtype=np.float32,
        )
        for index in range(7)
    ]


def test_known_post4_kinematics():
    result = m.post4_kinematics(
        linear_states(),
        anchor=1,
    )

    assert result[
        "POST4_SPEED"
    ] == pytest.approx(
        1.0,
        abs=0.0,
    )

    assert result[
        "POST4_TURNING"
    ] == pytest.approx(
        0.0,
        abs=0.0,
    )

    assert result[
        "POST4_PATH_EFFICIENCY"
    ] == pytest.approx(
        1.0,
        abs=0.0,
    )


def test_zero_transition_turning_blocks():
    with pytest.raises(
        m.ContractError,
        match=m.BLOCKED_ZERO_TRANSITION,
    ):
        m.transition_turning(
            np.array(
                [0.0, 0.0],
                dtype=np.float32,
            ),
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
        )


def test_zero_path_length_blocks_without_epsilon():
    states = [
        np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        np.array(
            [2.0, 0.0],
            dtype=np.float32,
        ),
    ]

    with pytest.raises(
        m.ContractError,
        match=m.BLOCKED_ZERO_PATH,
    ):
        m.post4_path_efficiency(
            states,
            anchor=1,
        )


def test_nonfinite_state_fails_closed():
    states = linear_states()

    states[3] = np.array(
        [
            np.nan,
            0.0,
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        m.ContractError,
        match="nonfinite state vector",
    ):
        m.post4_speed(
            states,
            anchor=1,
        )


def test_post4_window_cannot_be_shortened():
    states = [
        np.array(
            [
                float(index),
                0.0,
            ],
            dtype=np.float32,
        )
        for index in range(6)
    ]

    with pytest.raises(
        m.ContractError,
        match="POST4 prefix eligibility",
    ):
        m.post4_speed(
            states,
            anchor=1,
        )


class SyntheticModel:
    def __init__(self):
        self.parameter = Tensor(
            [3.0]
        )
        self.parameter_requires_grad = False
        self.buffer = Tensor(
            [4.0]
        )
        self.cache = Tensor(
            [0.0]
        )

    def forward(self):
        ssm_state = Tensor(np.zeros((1, 2, 2)))
        for i in range(3):
            ssm_state = Tensor(ssm_state.v + (i + 1))
            readout = ssm_state.clone()
        self.cache = Tensor(self.cache.v + 1.0)
        return {
            "primary":
                readout.clone(),
            "last_hidden_state":
                readout.clone(),
            "hidden_states": [
                readout.clone()
            ],
            "metadata": {
                "shape":
                    readout.shape,
                "dtype":
                    readout.dtype,
                "device":
                    readout.device,
            },
        }


MODEL_CAPTURE_LINE = source_line(
    SyntheticModel.forward,
    "readout = ssm_state.clone()",
)


def assert_exact_output(
    left,
    right,
):
    assert set(left) == set(right)

    for key in (
        "primary",
        "last_hidden_state",
    ):
        assert np.array_equal(
            left[key].v,
            right[key].v,
        )

        assert (
            left[key].shape
            == right[key].shape
        )

        assert (
            left[key].dtype
            == right[key].dtype
        )

        assert (
            left[key].device
            == right[key].device
        )

    assert (
        len(
            left[
                "hidden_states"
            ]
        )
        == 1
    )

    assert np.array_equal(
        left[
            "hidden_states"
        ][0].v,
        right[
            "hidden_states"
        ][0].v,
    )

    assert (
        left["metadata"]
        == right["metadata"]
    )


def test_exact_synthetic_noninterference():
    baseline = SyntheticModel()
    observed = SyntheticModel()

    baseline_output = (
        baseline.forward()
    )

    collector = m._synthetic_collector(
        m._synthetic_trace_binding(
            SyntheticModel.forward,
            MODEL_CAPTURE_LINE,
        ),
        {
            id(observed): {
                "layer_index": 11,
            }
        },
        enabled=True,
    )

    prior = sys.gettrace()

    with collector.capture():
        observed_output = (
            observed.forward()
        )

    assert sys.gettrace() is prior

    assert_exact_output(
        baseline_output,
        observed_output,
    )

    assert np.array_equal(
        baseline.parameter.v,
        observed.parameter.v,
    )

    assert (
        baseline.parameter_requires_grad
        == observed.parameter_requires_grad
    )

    assert np.array_equal(
        baseline.buffer.v,
        observed.buffer.v,
    )

    assert np.array_equal(
        baseline.cache.v,
        observed.cache.v,
    )

    assert (
        collector.snapshots
        is not None
    )

    m.validate_capture_coordinates(
        collector.snapshots,
        token_count=3,
    )


def build_runtime_baseline(
    tmp_path,
    monkeypatch,
):
    package_root = (
        tmp_path
        / "transformers"
    )

    mamba_dir = (
        package_root
        / "models"
        / "mamba"
    )

    mamba_dir.mkdir(
        parents=True,
    )

    init_path = (
        package_root
        / "__init__.py"
    )

    init_path.write_text(
        "",
        encoding="utf-8",
    )

    mamba_path = (
        mamba_dir
        / "modeling_mamba.py"
    )

    cache_path = (
        package_root
        / "cache_utils.py"
    )

    lines = [""] * 393

    lines.extend(
        [
            "class MambaMixer:",
            "    def slow_forward(self, ssm_state, discrete_A, discrete_B, hidden_states, C, cache_params, dtype=None):",
            "        _sentinel = 0",
            "        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()",
            "        _a = 1",
            "        _b = 2",
            "        _c = 3",
            "        _d = 4",
            "        _e = 5",
            "        _f = 6",
            "        _g = 7",
            "        _h = 8",
            "        _i = 9",
            "        _j = 10",
            "        _k = 11",
            "        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]",
            "        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))",
            "        _l = 12",
            "        _m = 13",
            "        _n = 14",
            "        _o = 15",
            "        _p = 16",
            "        _q = 17",
            "        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)",
            "        return scan_output",
            "    def forward(self, hidden_states=None, cache_params=None, cache_position=None, attention_mask=None):",
            "        is_fast_path_available = all((selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn))",
            "        if is_fast_path_available and \"cuda\" in self.x_proj.weight.device.type and not is_torchdynamo_compiling():",
            "            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)",
            "        return self.slow_forward(hidden_states, cache_params, cache_position, attention_mask)",
        ]
    )

    source = (
        "\n".join(lines)
        + "\n"
    )

    mamba_path.write_text(
        source,
        encoding="utf-8",
        newline="\n",
    )

    cache_path.write_text(
        (
            "conv_states = []\n"
            "ssm_states = []\n"
        ),
        encoding="utf-8",
        newline="\n",
    )

    mamba_module = types.ModuleType(
        m.MAMBA_MODULE
    )

    mamba_module.__file__ = str(
        mamba_path
    )

    mamba_module.__dict__[
        "__name__"
    ] = m.MAMBA_MODULE

    exec(
        compile(
            source,
            str(mamba_path),
            "exec",
        ),
        mamba_module.__dict__,
    )

    cache_module = types.ModuleType(
        m.CACHE_MODULE
    )

    cache_module.__file__ = str(
        cache_path
    )

    transformers_module = types.ModuleType(
        "transformers"
    )

    transformers_module.__file__ = str(
        init_path
    )

    transformers_module.__version__ = (
        m.EXPECTED_VERSIONS[
            "transformers"
        ]
    )

    torch_module = types.ModuleType(
        "torch"
    )

    torch_module.__version__ = (
        m.EXPECTED_VERSIONS[
            "torch"
        ]
    )

    modules = {
        m.MAMBA_MODULE:
            mamba_module,
        m.CACHE_MODULE:
            cache_module,
        "transformers":
            transformers_module,
        "torch":
            torch_module,
    }

    versions = dict(
        m.EXPECTED_VERSIONS
    )

    monkeypatch.setattr(
        m,
        "_runtime_environment",
        lambda: (
            modules,
            versions,
        ),
    )

    class Distribution:
        def locate_file(
            self,
            value,
        ):
            assert (
                value
                == "transformers"
            )

            return package_root

    monkeypatch.setattr(
        m.importlib_metadata,
        "distribution",
        lambda name:
            Distribution(),
    )

    monkeypatch.setattr(
        m.importlib.util,
        "find_spec",
        lambda name:
            types.SimpleNamespace(
                origin=str(
                    mamba_path
                    if name
                    == m.MAMBA_MODULE
                    else cache_path
                )
            ),
    )

    mamba_bytes = (
        mamba_path.read_bytes()
    )

    cache_bytes = (
        cache_path.read_bytes()
    )

    monkeypatch.setattr(
        m,
        "MAMBA_BYTES",
        len(mamba_bytes),
    )

    monkeypatch.setattr(
        m,
        "MAMBA_SHA256",
        hashlib.sha256(
            mamba_bytes
        ).hexdigest(),
    )

    monkeypatch.setattr(
        m,
        "CACHE_BYTES",
        len(cache_bytes),
    )

    monkeypatch.setattr(
        m,
        "CACHE_SHA256",
        hashlib.sha256(
            cache_bytes
        ).hexdigest(),
    )

    return (
        modules,
        versions,
        mamba_path,
        cache_path,
    )


def _rewrite_runtime_source(
    mamba_path,
    monkeypatch,
    old: str,
    new: str,
):
    text = mamba_path.read_text(
        encoding="utf-8"
    )

    assert text.count(old) == 1

    text = text.replace(
        old,
        new,
        1,
    )

    mamba_path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    raw = mamba_path.read_bytes()

    monkeypatch.setattr(
        m,
        "MAMBA_BYTES",
        len(raw),
    )

    monkeypatch.setattr(
        m,
        "MAMBA_SHA256",
        hashlib.sha256(
            raw
        ).hexdigest(),
    )


def test_runtime_gate_synthetic_baseline(
    tmp_path,
    monkeypatch,
):
    build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    m.runtime_gate()


def test_runtime_gate_accepts_frozen_style_dispatch_with_mamba_inner_fn(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    source = mamba_path.read_text(
        encoding="utf-8"
    )

    assert (
        "mamba_inner_fn"
        in source
    )
    assert (
        '"cuda" in self.x_proj.weight.device.type'
        in source
    )
    assert (
        "return self.slow_forward("
        in source
    )

    m.runtime_gate()


def test_runtime_gate_rejects_dispatch_without_slow_fallback(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        (
            "        return self.slow_forward("
            "hidden_states, cache_params, "
            "cache_position, attention_mask)"
        ),
        "        return hidden_states",
    )

    with pytest.raises(
        m.ContractError,
        match="unsupported backend",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_fast_path_without_cuda_device_guard(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        (
            '        if is_fast_path_available and '
            '"cuda" in self.x_proj.weight.device.type '
            "and not is_torchdynamo_compiling():"
        ),
        (
            "        if is_fast_path_available "
            "and not is_torchdynamo_compiling():"
        ),
    )

    with pytest.raises(
        m.ContractError,
        match="unsupported backend",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_default_branch_fast_path(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        (
            "        return self.slow_forward("
            "hidden_states, cache_params, "
            "cache_position, attention_mask)"
        ),
        (
            "        return self.cuda_kernels_forward("
            "hidden_states, cache_params, "
            "cache_position, attention_mask)"
        ),
    )

    with pytest.raises(
        m.ContractError,
        match="unsupported backend",
    ):
        m.runtime_gate()


def test_frozen_transformers_v5_source_role_fixture(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    lines = mamba_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines[m.DELTA_B_U_LINE - 1] == (
        "        deltaB_u = discrete_B * "
        "hidden_states[:, :, :, None].float()"
    )

    assert lines[m.RECURRENT_UPDATE_LINE - 1] == (
        "        ssm_state = discrete_A[:, :, i, :] * "
        "ssm_state + deltaB_u[:, :, i, :]"
    )

    assert lines[m.CAPTURE_LINE - 1] == (
        "        scan_output = torch.matmul("
        "ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))"
    )

    assert lines[m.FINAL_CACHE_PERSISTENCE_LINE - 1] == (
        "        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)"
    )

    m.runtime_gate()


def test_runtime_gate_rejects_old_cpu_build_tag(
    tmp_path,
    monkeypatch,
):
    _, versions, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    versions["torch"] = "2.10.0+cpu"

    with pytest.raises(
        m.ContractError,
        match="runtime version",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_other_torch_build_tag(
    tmp_path,
    monkeypatch,
):
    _, versions, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    versions["torch"] = "2.10.0+cu130"

    with pytest.raises(
        m.ContractError,
        match="runtime version",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_wrong_python_version(
    tmp_path,
    monkeypatch,
):
    _, versions, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    versions["python"] = "3.12.12"

    with pytest.raises(
        m.ContractError,
        match="runtime version",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_wrong_numpy_version(
    tmp_path,
    monkeypatch,
):
    _, versions, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    versions["numpy"] = "2.0.1"

    with pytest.raises(
        m.ContractError,
        match="runtime version",
    ):
        m.runtime_gate()


def test_runtime_gate_rejects_wrong_transformers_version(
    tmp_path,
    monkeypatch,
):
    _, versions, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    versions["transformers"] = "wrong"

    with pytest.raises(
        m.ContractError,
        match="runtime version",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_source_bytes(
    tmp_path,
    monkeypatch,
):
    build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    monkeypatch.setattr(
        m,
        "MAMBA_BYTES",
        m.MAMBA_BYTES + 1,
    )

    with pytest.raises(
        m.ContractError,
        match="Mamba byte-size mismatch",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_source_hash(
    tmp_path,
    monkeypatch,
):
    build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    monkeypatch.setattr(
        m,
        "MAMBA_SHA256",
        "0" * 64,
    )

    with pytest.raises(
        m.ContractError,
        match="Mamba SHA256 mismatch",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_cache_hash(
    tmp_path,
    monkeypatch,
):
    build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    monkeypatch.setattr(
        m,
        "CACHE_SHA256",
        "0" * 64,
    )

    with pytest.raises(
        m.ContractError,
        match="cache_utils SHA256 mismatch",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_code_object(
    tmp_path,
    monkeypatch,
):
    modules, _, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    def wrong(self):
        return None

    modules[
        m.MAMBA_MODULE
    ].MambaMixer.slow_forward = wrong

    with pytest.raises(
        m.ContractError,
        match="slow code identity",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_delta_b_u_target(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()",
        "        wrong_alias = discrete_B * hidden_states[:, :, :, None].float()",
    )

    with pytest.raises(
        m.ContractError,
        match="source deltaB_u role",
    ):
        m.runtime_gate()


def test_runtime_gate_delta_b_u_missing_discrete_b(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()",
        "        deltaB_u = hidden_states[:, :, :, None].float() * hidden_states[:, :, :, None].float()",
    )

    with pytest.raises(
        m.ContractError,
        match="source deltaB_u role",
    ):
        m.runtime_gate()


def test_runtime_gate_delta_b_u_missing_hidden_states(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()",
        "        deltaB_u = discrete_B * discrete_B",
    )

    with pytest.raises(
        m.ContractError,
        match="source deltaB_u role",
    ):
        m.runtime_gate()


def test_runtime_gate_update_missing_delta_b_u(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]",
        "        ssm_state = discrete_A[:, :, i, :] * ssm_state + ssm_state",
    )

    with pytest.raises(
        m.ContractError,
        match="source update role",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_recurrent_update_target(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]",
        "        wrong_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]",
    )

    with pytest.raises(
        m.ContractError,
        match="source update role",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_readout_role(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))",
        "        scan_output = torch.matmul(deltaB_u[:, :, i, :].to(dtype), C[:, i, :].unsqueeze(-1))",
    )

    with pytest.raises(
        m.ContractError,
        match="source readout role",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_cache_persistence_role(
    tmp_path,
    monkeypatch,
):
    _, _, mamba_path, _ = build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    _rewrite_runtime_source(
        mamba_path,
        monkeypatch,
        "        cache_params.ssm_states[self.layer_idx].copy_(ssm_state)",
        "        cache_params.ssm_states[self.layer_idx].copy_(deltaB_u)",
    )

    with pytest.raises(
        m.ContractError,
        match="source recurrent cache role",
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_capture_line(
    tmp_path,
    monkeypatch,
):
    build_runtime_baseline(
        tmp_path,
        monkeypatch,
    )

    monkeypatch.setattr(
        m,
        "CAPTURE_LINE",
        999,
    )

    with pytest.raises(
        m.ContractError,
        match=(
            "source readout role"
            "|line binding"
        ),
    ):
        m.runtime_gate()


def test_runtime_gate_wrong_backend(
    tmp_path,
    monkeypatch,
):
    modules, _, _, _ = (
        build_runtime_baseline(
            tmp_path,
            monkeypatch,
        )
    )

    forward = modules[
        m.MAMBA_MODULE
    ].MambaMixer.forward

    original = (
        m.inspect.getsourcelines
    )

    def fake_getsourcelines(
        obj,
    ):
        if obj is forward:
            return (
                [
                    "def forward(self):\n",
                    "    return None\n",
                ],
                419,
            )

        return original(obj)

    monkeypatch.setattr(
        m.inspect,
        "getsourcelines",
        fake_getsourcelines,
    )

    with pytest.raises(
        m.ContractError,
        match="unsupported backend",
    ):
        m.runtime_gate()


def test_scientific_execution_is_prohibited():
    with pytest.raises(
        m.ContractError,
        match=m.BLOCKED_SCIENTIFIC_EXECUTION,
    ):
        m.scientific_extraction()

    with pytest.raises(
        m.ContractError,
        match=m.BLOCKED_SCIENTIFIC_EXECUTION,
    ):
        m.main([])
