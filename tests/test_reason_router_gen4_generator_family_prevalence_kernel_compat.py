from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence as eq
from scripts import reason_router_gen4_generator_family_prevalence_kernel_compat as compat


def _spec(*, revision: str = "a" * 40, expected_binary: bytes = b"ok"):
    return compat.KernelSpec(
        label="TEST",
        repo_id="kernels-community/test-kernel",
        revision=revision,
        package_name="test_kernel",
        binary_sha256=hashlib.sha256(expected_binary).hexdigest(),
        required_functions=("f1", "f2"),
    )


def _make_snapshot(
    cache_root: Path,
    spec: compat.KernelSpec,
    *,
    prefix: str = "kernels",
    binary: bytes = b"ok",
    direct_layout: bool = False,
) -> Path:
    owner, name = spec.repo_id.split("/")
    snapshot = (
        cache_root
        / f"{prefix}--{owner}--{name}"
        / "snapshots"
        / spec.revision
    )
    variant = snapshot / "build" / compat.BUILD_VARIANT
    if direct_layout:
        init_path = variant / "__init__.py"
    else:
        init_path = variant / spec.package_name / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text("# synthetic\n", encoding="utf-8")
    (variant / "synthetic.abi3.so").write_bytes(binary)
    return snapshot


def test_exact_frozen_backend_identities_are_reused():
    assert compat.KERNELS_VERSION == eq.backend.KERNELS_VERSION == "0.10.2"
    assert (
        compat.BUILD_VARIANT
        == eq.backend.BUILD_VARIANT
        == "torch210-cxx11-cu128-x86_64-linux"
    )
    assert compat.MAMBA_SPEC.revision == eq.backend.MAMBA_REV
    assert compat.CONV_SPEC.revision == eq.backend.CONV_REV
    assert compat.MAMBA_SPEC.binary_sha256 == eq.backend.MAMBA_BINARY_SHA256
    assert compat.CONV_SPEC.binary_sha256 == eq.backend.CONV_BINARY_SHA256


def test_network_download_uses_current_kernel_repo_type_and_no_token(
    monkeypatch, tmp_path
):
    spec = _spec()
    target = _make_snapshot(tmp_path / "downloaded", spec)
    calls = []

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: ())

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(target)

    monkeypatch.setattr(compat, "_hub_snapshot_download", fake_download)

    resolved = compat.resolve_exact_snapshot(spec)
    assert resolved == target
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == spec.repo_id
    assert call["repo_type"] == "kernel"
    assert call["revision"] == spec.revision
    assert call["token"] is False
    assert call["local_files_only"] is False
    assert call["allow_patterns"] == [
        f"build/{compat.BUILD_VARIANT}/*"
    ]


def test_exact_kernel_cache_is_used_before_network(monkeypatch, tmp_path):
    spec = _spec()
    snapshot = _make_snapshot(tmp_path, spec, prefix="kernels")

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    def forbidden_download(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(compat, "_hub_snapshot_download", forbidden_download)
    assert compat.resolve_exact_snapshot(spec) == snapshot


def test_exact_legacy_model_cache_is_accepted_locally(monkeypatch, tmp_path):
    spec = _spec()
    snapshot = _make_snapshot(tmp_path, spec, prefix="models")

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))
    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )
    assert compat.resolve_exact_snapshot(spec) == snapshot


def test_wrong_revision_cache_is_not_accepted(monkeypatch, tmp_path):
    spec = _spec()
    wrong = compat.KernelSpec(
        **{**spec.__dict__, "revision": "b" * 40}
    )
    _make_snapshot(tmp_path, wrong)

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    target = _make_snapshot(tmp_path / "net", spec)
    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: str(target),
    )
    assert compat.resolve_exact_snapshot(spec) == target


def test_binary_sha_is_checked_before_import(monkeypatch, tmp_path):
    spec = _spec(expected_binary=b"expected")
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        binary=b"wrong",
    )
    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    called = {"value": False}

    def importer(_name, _path):
        called["value"] = True
        raise AssertionError("import must not happen")

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="TEST_BINARY_SHA256",
    ):
        compat.load_exact_module(spec, importer=importer)

    assert called["value"] is False


@pytest.mark.parametrize("direct_layout", (False, True))
def test_both_v0102_and_current_module_layouts_are_supported(
    monkeypatch, tmp_path, direct_layout
):
    spec = _spec()
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        direct_layout=direct_layout,
    )
    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    init_path = compat._select_module_init(snapshot, spec)

    module = SimpleNamespace(
        __file__=str(init_path),
        f1=lambda: None,
        f2=lambda: None,
    )

    observed = compat.load_exact_module(
        spec,
        importer=lambda _name, _path: module,
    )
    assert observed is module
    assert compat._variant_dir(snapshot).name == compat.BUILD_VARIANT


def test_missing_function_surface_is_blocked(monkeypatch, tmp_path):
    spec = _spec()
    snapshot = _make_snapshot(tmp_path, spec)
    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    init_path = compat._select_module_init(snapshot, spec)
    module = SimpleNamespace(
        __file__=str(init_path),
        f1=lambda: None,
        f2=None,
    )

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="TEST_FUNCTION_SURFACE:f2",
    ):
        compat.load_exact_module(
            spec,
            importer=lambda _name, _path: module,
        )


def test_transformers_patch_surface_is_exact():
    mamba = SimpleNamespace(
        selective_scan_fn=lambda: "scan",
        selective_state_update=lambda: "update",
        mamba_inner_fn=lambda: "inner",
    )
    conv = SimpleNamespace(
        causal_conv1d_fn=lambda: "conv",
        causal_conv1d_update=lambda: "conv_update",
    )
    target = SimpleNamespace()

    functions = compat.patch_transformers_mamba(
        mamba,
        conv,
        modeling_module=target,
    )

    assert set(functions) == {
        "selective_scan_fn",
        "selective_state_update",
        "mamba_inner_fn",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    }
    for name, fn in functions.items():
        assert getattr(target, name) is fn


def test_runner_uses_prevalence_compat_loader_not_frozen_network_loader():
    source = inspect.getsource(eq)
    assert (
        "reason_router_gen4_generator_family_prevalence_kernel_compat"
        in source
    )
    assert "kernel_compat.load_exact_fast_kernels()" in source
    assert "backend.load_exact_fast_kernels()" not in source


def test_compat_loader_does_not_relax_scientific_or_backend_constants():
    source = inspect.getsource(compat)
    assert "repo_type=\"kernel\"" in source
    assert "token=False" in source
    assert "MAMBA_BINARY_SHA256" in source
    assert "CONV_BINARY_SHA256" in source
    assert "STATE_ATOL" not in source
    assert "GEOMETRY_ATOL" not in source
    assert "FORWARDS_PER_BACKEND" not in source
