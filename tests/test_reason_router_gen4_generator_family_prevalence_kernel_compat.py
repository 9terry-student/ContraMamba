from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence as eq
from scripts import reason_router_gen4_generator_family_prevalence_kernel_compat as compat


def _spec(
    *,
    scientific_revision: str = "a" * 40,
    transport_revisions: tuple[str, ...] = ("b" * 40, "c" * 40),
    expected_binary: bytes = b"ok",
):
    return compat.KernelSpec(
        label="TEST",
        repo_id="kernels-community/test-kernel",
        scientific_revision=scientific_revision,
        transport_revisions=transport_revisions,
        package_name="test_kernel",
        binary_sha256=hashlib.sha256(expected_binary).hexdigest(),
        required_functions=("f1", "f2"),
    )


def _make_snapshot(
    cache_root: Path,
    spec: compat.KernelSpec,
    revision: str,
    *,
    prefix: str,
    binary: bytes = b"ok",
    direct_layout: bool = False,
) -> Path:
    owner, name = spec.repo_id.split("/")
    snapshot = (
        cache_root
        / f"{prefix}--{owner}--{name}"
        / "snapshots"
        / revision
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


def test_frozen_scientific_backend_identities_are_unchanged():
    assert compat.KERNELS_VERSION == eq.backend.KERNELS_VERSION == "0.10.2"
    assert (
        compat.BUILD_VARIANT
        == eq.backend.BUILD_VARIANT
        == "torch210-cxx11-cu128-x86_64-linux"
    )

    assert (
        compat.MAMBA_SPEC.scientific_revision
        == eq.backend.MAMBA_REV
        == "c8ffc584c147878a6eb978ae0e8db4d116c93a8c"
    )
    assert (
        compat.CONV_SPEC.scientific_revision
        == eq.backend.CONV_REV
        == "f2651e776f66069cdcf842840db637583def1223"
    )

    assert (
        compat.MAMBA_SPEC.binary_sha256
        == eq.backend.MAMBA_BINARY_SHA256
        == "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
    )
    assert (
        compat.CONV_SPEC.binary_sha256
        == eq.backend.CONV_BINARY_SHA256
        == "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"
    )


def test_transport_locators_are_immutable_and_distinct_from_scientific_revisions():
    assert compat.MAMBA_TRANSPORT_REVISIONS == (
        "170306cb84f6fac356ed839fd6e2dc53ab68080e",
        "a8ca9c4af8613ebcd16eb22873e4896ee488c840",
        "a80a7604874b108585feb87096a0c86df2a1e5e3",
        "90a845d5a0d552dc6b7f1653adf68bcf271f5437",
    )
    assert compat.CONV_TRANSPORT_REVISIONS == (
        "2999c83c99b9ac5fa87b861af3ec6bac28b1c300",
        "02ab414d848bbee389d801f87b24fa536de60273",
        "3552fa17c03203cb43a3a76efb4de5a6e31554b5",
    )

    for revision in (
        *compat.MAMBA_TRANSPORT_REVISIONS,
        *compat.CONV_TRANSPORT_REVISIONS,
    ):
        assert len(revision) == 40
        int(revision, 16)

    assert compat.MAMBA_SPEC.scientific_revision not in (
        compat.MAMBA_TRANSPORT_REVISIONS
    )
    assert compat.CONV_SPEC.scientific_revision not in (
        compat.CONV_TRANSPORT_REVISIONS
    )


def test_exact_legacy_model_cache_is_preferred_without_network(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        spec.scientific_revision,
        prefix="models",
    )

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    def forbidden_download(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        forbidden_download,
    )

    resolved = compat.resolve_exact_snapshot(spec)
    assert resolved.path == snapshot
    assert resolved.transport_revision == spec.scientific_revision
    assert resolved.transport_repo_type == "legacy-model-local"
    assert resolved.source == "legacy_model_cache"


def test_exact_migrated_kernel_cache_is_accepted_without_network(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    transport = spec.transport_revisions[0]
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
    )

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    resolved = compat.resolve_exact_snapshot(spec)
    assert resolved.path == snapshot
    assert resolved.transport_revision == transport
    assert resolved.transport_repo_type == "kernel"
    assert resolved.source == "kernel_repo_cache"


def test_remote_transport_uses_kernel_repo_type_exact_commit_and_no_token(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    transport = spec.transport_revisions[0]
    target = _make_snapshot(
        tmp_path / "downloaded",
        spec,
        transport,
        prefix="kernels",
    )
    calls = []

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: ())

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(target)

    monkeypatch.setattr(compat, "_hub_snapshot_download", fake_download)

    resolved = compat.resolve_exact_snapshot(spec)

    assert resolved.path == target
    assert resolved.transport_revision == transport
    assert resolved.transport_repo_type == "kernel"
    assert resolved.source == "kernel_repo_download"

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == spec.repo_id
    assert call["repo_type"] == "kernel"
    assert call["revision"] == transport
    assert call["token"] is False
    assert call["local_files_only"] is False
    assert call["revision"] != spec.scientific_revision


def test_remote_transport_falls_back_only_across_whitelisted_exact_commits(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    first, second = spec.transport_revisions
    target = _make_snapshot(
        tmp_path / "downloaded",
        spec,
        second,
        prefix="kernels",
    )
    calls = []

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: ())

    def fake_download(**kwargs):
        calls.append(kwargs)
        if kwargs["revision"] == first:
            raise RuntimeError("synthetic first transport failure")
        return str(target)

    monkeypatch.setattr(compat, "_hub_snapshot_download", fake_download)

    resolved = compat.resolve_exact_snapshot(spec)
    assert resolved.transport_revision == second
    assert [call["revision"] for call in calls] == [first, second]


def test_module_surface_failure_falls_back_to_next_whitelisted_candidate(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    first, second = spec.transport_revisions

    first_snapshot = _make_snapshot(
        tmp_path,
        spec,
        first,
        prefix="kernels",
    )
    second_snapshot = _make_snapshot(
        tmp_path,
        spec,
        second,
        prefix="kernels",
    )

    monkeypatch.setattr(
        compat,
        "_kernel_package_version",
        lambda: "0.10.2",
    )
    monkeypatch.setattr(
        compat,
        "_cache_roots",
        lambda: (tmp_path,),
    )
    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError(kwargs)
        ),
    )

    calls = []

    def importer(module_name, init_path):
        calls.append((module_name, Path(init_path)))

        if first in Path(init_path).parts:
            return SimpleNamespace(
                __file__=str(init_path),
                f1=lambda: None,
                f2=None,
            )

        assert second in Path(init_path).parts
        return SimpleNamespace(
            __file__=str(init_path),
            f1=lambda: None,
            f2=lambda: None,
        )

    observed, resolved = compat.load_exact_module(
        spec,
        importer=importer,
    )

    assert callable(observed.f1)
    assert callable(observed.f2)
    assert resolved.path == second_snapshot
    assert resolved.transport_revision == second
    assert [path for _name, path in calls] == [
        compat._select_module_init(first_snapshot, spec),
        compat._select_module_init(second_snapshot, spec),
    ]
    assert calls[0][0] == spec.package_name
    assert calls[1][0] == f"{spec.package_name}_{second[:12]}"


def test_wrong_binary_is_rejected_before_import(monkeypatch, tmp_path):
    spec = _spec(expected_binary=b"expected")
    transport = spec.transport_revisions[0]
    _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
        binary=b"wrong",
    )

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    # Prevent remote fallback from manufacturing a valid candidate.
    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    called = {"value": False}

    def importer(_name, _path):
        called["value"] = True
        raise AssertionError("import must not happen")

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="KERNEL_TRANSPORT_UNAVAILABLE",
    ):
        compat.load_exact_module(spec, importer=importer)

    assert called["value"] is False


@pytest.mark.parametrize("direct_layout", (False, True))
def test_both_historical_and_migrated_module_layouts_are_supported(
    monkeypatch,
    tmp_path,
    direct_layout,
):
    spec = _spec()
    transport = spec.transport_revisions[0]
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
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

    observed, resolved = compat.load_exact_module(
        spec,
        importer=lambda _name, _path: module,
    )
    assert observed is module
    assert resolved.transport_revision == transport


def test_loaded_module_may_resolve_to_other_authorized_snapshot_wrapper(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    transport = spec.transport_revisions[0]
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
    )

    variant = compat._variant_dir(snapshot)
    direct_init = variant / "__init__.py"
    direct_init.write_text("# direct synthetic\n", encoding="utf-8")

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    selected = compat._select_module_init(snapshot, spec)
    assert selected != direct_init

    module = SimpleNamespace(
        __file__=str(direct_init),
        f1=lambda: None,
        f2=lambda: None,
    )

    observed, resolved = compat.load_exact_module(
        spec,
        importer=lambda _name, _path: module,
    )

    assert observed is module
    assert resolved.transport_revision == transport


def test_loaded_module_outside_authorized_snapshot_wrappers_is_blocked(
    monkeypatch,
    tmp_path,
):
    spec = _spec()
    transport = spec.transport_revisions[0]
    _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
    )

    foreign = tmp_path / "foreign" / "__init__.py"
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_text("# foreign\n", encoding="utf-8")

    monkeypatch.setattr(compat, "_kernel_package_version", lambda: "0.10.2")
    monkeypatch.setattr(compat, "_cache_roots", lambda: (tmp_path,))

    module = SimpleNamespace(
        __file__=str(foreign),
        f1=lambda: None,
        f2=lambda: None,
    )

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="TEST_MODULE_PATH",
    ):
        compat.load_exact_module(
            spec,
            importer=lambda _name, _path: module,
        )


def test_missing_function_surface_is_blocked(monkeypatch, tmp_path):
    spec = _spec()
    transport = spec.transport_revisions[0]
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
    )

    monkeypatch.setattr(
        compat,
        "_kernel_package_version",
        lambda: "0.10.2",
    )
    monkeypatch.setattr(
        compat,
        "_cache_roots",
        lambda: (tmp_path,),
    )
    monkeypatch.setattr(
        compat,
        "_hub_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("offline")
        ),
    )

    init_path = compat._select_module_init(snapshot, spec)

    def importer(_name, path):
        return SimpleNamespace(
            __file__=str(path),
            f1=lambda: None,
            f2=None,
        )

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="KERNEL_MODULE_UNAVAILABLE:TEST",
    ):
        compat.load_exact_module(
            spec,
            importer=importer,
        )



def test_exact_causal_conv_cuda_compat_uses_wrapper_cpp_functions(
    monkeypatch,
    tmp_path,
):
    spec = _spec(expected_binary=b"frozen-conv")
    transport = spec.transport_revisions[0]
    snapshot = _make_snapshot(
        tmp_path,
        spec,
        transport,
        prefix="kernels",
        binary=b"frozen-conv",
        direct_layout=True,
    )
    monkeypatch.setattr(compat, "CONV_SPEC", spec)

    resolved = compat.ResolvedKernelSnapshot(
        path=snapshot,
        transport_revision=transport,
        transport_repo_type="kernel",
        source="synthetic",
    )

    fwd = lambda *args: ("fwd", args)
    bwd = lambda *args: ("bwd", args)
    update = lambda *args: ("update", args)

    fn_namespace = {
        "causal_conv1d_fwd_function": fwd,
        "causal_conv1d_bwd_function": bwd,
    }
    update_namespace = {
        "causal_conv1d_update_function": update,
    }
    exec("def causal_conv1d_fn():\n    pass\n", fn_namespace)
    exec("def causal_conv1d_update():\n    pass\n", update_namespace)

    conv = SimpleNamespace(
        causal_conv1d_fn=fn_namespace["causal_conv1d_fn"],
        causal_conv1d_update=update_namespace["causal_conv1d_update"],
    )

    observed = compat.build_exact_causal_conv_cuda_compat(
        conv,
        resolved,
    )

    assert observed.causal_conv1d_fwd is fwd
    assert observed.causal_conv1d_bwd is bwd
    assert observed.causal_conv1d_update is update


def test_mamba_internal_causal_conv_binding_uses_exact_compat_surface(
    monkeypatch,
):
    namespace = {}
    exec(
        "def mamba_inner_fn():\n"
        "    return causal_conv1d_cuda\n",
        namespace,
    )
    mamba = SimpleNamespace(
        mamba_inner_fn=namespace["mamba_inner_fn"],
    )
    conv = SimpleNamespace()
    exact_compat = SimpleNamespace(
        causal_conv1d_fwd=lambda: None,
        causal_conv1d_bwd=lambda: None,
        causal_conv1d_update=lambda: None,
    )

    monkeypatch.setattr(
        compat,
        "build_exact_causal_conv_cuda_compat",
        lambda observed_conv, resolved: exact_compat,
    )

    resolved = SimpleNamespace(path=Path("synthetic"))
    observed = compat.patch_mamba_internal_causal_conv_binding(
        mamba,
        conv,
        resolved,
    )

    assert observed is exact_compat
    assert (
        mamba.mamba_inner_fn.__globals__["causal_conv1d_cuda"]
        is exact_compat
    )
    assert mamba.mamba_inner_fn() is exact_compat


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


def _synthetic_exact_kernel_bundle():
    mamba = SimpleNamespace(
        selective_scan_fn=lambda: "scan",
        selective_state_update=lambda: "update",
        mamba_inner_fn=lambda: "inner",
    )
    conv = SimpleNamespace(
        causal_conv1d_fn=lambda: "conv",
        causal_conv1d_update=lambda: "conv_update",
    )
    return {
        "mamba": mamba,
        "conv": conv,
        "selective_scan_fn": mamba.selective_scan_fn,
        "selective_state_update": mamba.selective_state_update,
        "mamba_inner_fn": mamba.mamba_inner_fn,
        "causal_conv1d_fn": conv.causal_conv1d_fn,
        "causal_conv1d_update": conv.causal_conv1d_update,
        "transport_identity_status":
            "EXACT_FROZEN_BINARY_SHA256_MATCH",
    }


def test_exact_transformers_constructor_loader_routes_without_default():
    kernels = _synthetic_exact_kernel_bundle()

    def original_loader(*args, **kwargs):
        raise AssertionError((args, kwargs))

    modeling = SimpleNamespace(
        lazy_load_kernel=original_loader,
    )

    with compat.exact_transformers_kernel_loader(
        kernels,
        modeling_module=modeling,
    ) as calls:
        assert modeling.lazy_load_kernel("causal-conv1d") is kernels["conv"]
        assert modeling.lazy_load_kernel("mamba-ssm") is kernels["mamba"]

    assert calls == ["causal-conv1d", "mamba-ssm"]
    assert modeling.lazy_load_kernel is original_loader


def test_exact_transformers_constructor_loader_blocks_unknown_and_restores():
    kernels = _synthetic_exact_kernel_bundle()

    def original_loader(name):
        return name

    modeling = SimpleNamespace(
        lazy_load_kernel=original_loader,
    )

    with pytest.raises(
        compat.KernelCompatibilityError,
        match="TRANSFORMERS_KERNEL_NAME:unknown-kernel",
    ):
        with compat.exact_transformers_kernel_loader(
            kernels,
            modeling_module=modeling,
        ):
            modeling.lazy_load_kernel("unknown-kernel")

    assert modeling.lazy_load_kernel is original_loader


def test_transformers_kernel_binding_validation_is_exact():
    kernels = _synthetic_exact_kernel_bundle()
    modeling = SimpleNamespace(
        mamba_ssm=kernels["mamba"],
        causal_conv1d=kernels["conv"],
        selective_scan_fn=kernels["selective_scan_fn"],
        selective_state_update=kernels["selective_state_update"],
        mamba_inner_fn=kernels["mamba_inner_fn"],
        causal_conv1d_fn=kernels["causal_conv1d_fn"],
        causal_conv1d_update=kernels["causal_conv1d_update"],
    )

    compat.validate_transformers_kernel_bindings(
        kernels,
        modeling_module=modeling,
    )

    modeling.causal_conv1d_fn = lambda: "foreign"
    with pytest.raises(
        compat.KernelCompatibilityError,
        match="TRANSFORMERS_KERNEL_BINDING:causal_conv1d_fn",
    ):
        compat.validate_transformers_kernel_bindings(
            kernels,
            modeling_module=modeling,
        )


def test_runner_uses_compat_loader_and_persists_transport_provenance():
    source = inspect.getsource(eq.run_one_pair)

    assert "kernel_compat.load_exact_fast_kernels()" in source
    assert "backend.load_exact_fast_kernels()" not in source

    required_report_tokens = (
        '"mamba_revision": backend.MAMBA_REV',
        '"mamba_transport_revision":',
        '"mamba_transport_repo_type":',
        '"mamba_transport_source":',
        '"causal_conv_revision": backend.CONV_REV',
        '"causal_conv_transport_revision":',
        '"causal_conv_transport_repo_type":',
        '"causal_conv_transport_source":',
        '"kernel_transport_identity_status":',
    )
    for token in required_report_tokens:
        assert token in source


def test_compat_layer_never_relaxes_scientific_backend_contract():
    source = inspect.getsource(compat)

    assert 'TRANSPORT_REPO_TYPE = "kernel"' in source
    assert "token=False" in source
    assert "MAMBA_BINARY_SHA256" in source
    assert "CONV_BINARY_SHA256" in source
    assert "scientific_revision" in source
    assert "transport_revisions" in source

    # Compatibility code must not alter equivalence tolerances or forward budget.
    assert "STATE_ATOL" not in source
    assert "STATE_RTOL" not in source
    assert "GEOMETRY_ATOL" not in source
    assert "GEOMETRY_RTOL" not in source
    assert "FORWARDS_PER_BACKEND" not in source
    assert "TOTAL_MODEL_FORWARDS" not in source

    # Mutable main and hidden transport substitution are forbidden.
    assert 'revision="main"' not in source
    assert "spec.scientific_revision," not in source.split(
        "_hub_snapshot_download(", 1
    )[1].split(")", 1)[0]
