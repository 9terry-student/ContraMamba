from __future__ import annotations

import importlib.metadata
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend


KERNELS_VERSION = backend.KERNELS_VERSION
BUILD_VARIANT = backend.BUILD_VARIANT
TRANSPORT_REPO_TYPE = "kernel"

# Historical scientific identity, frozen by the validated backend.
MAMBA_SCIENTIFIC_REVISION = backend.MAMBA_REV
CONV_SCIENTIFIC_REVISION = backend.CONV_REV

# The historical Hub repositories were model repos. They are no longer
# reachable, but the exact frozen LFS payloads remain present in the migrated
# kernel-repo histories. These immutable commits are transport locators only.
# Scientific identity remains the historical revision + build variant +
# exact loaded .so SHA256.
MAMBA_TRANSPORT_REVISIONS = (
    "170306cb84f6fac356ed839fd6e2dc53ab68080e",
    "a8ca9c4af8613ebcd16eb22873e4896ee488c840",
    "a80a7604874b108585feb87096a0c86df2a1e5e3",
    "90a845d5a0d552dc6b7f1653adf68bcf271f5437",
)
CONV_TRANSPORT_REVISIONS = (
    "2999c83c99b9ac5fa87b861af3ec6bac28b1c300",
    "02ab414d848bbee389d801f87b24fa536de60273",
    "3552fa17c03203cb43a3a76efb4de5a6e31554b5",
)


class KernelCompatibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise KernelCompatibilityError(message)


@dataclass(frozen=True)
class KernelSpec:
    label: str
    repo_id: str
    scientific_revision: str
    transport_revisions: tuple[str, ...]
    package_name: str
    binary_sha256: str
    required_functions: tuple[str, ...]

    @property
    def revision(self) -> str:
        # Compatibility alias: this remains the frozen scientific revision.
        return self.scientific_revision


@dataclass(frozen=True)
class ResolvedKernelSnapshot:
    path: Path
    transport_revision: str
    transport_repo_type: str
    source: str


MAMBA_SPEC = KernelSpec(
    label="MAMBA",
    repo_id="kernels-community/mamba-ssm",
    scientific_revision=MAMBA_SCIENTIFIC_REVISION,
    transport_revisions=MAMBA_TRANSPORT_REVISIONS,
    package_name="mamba_ssm",
    binary_sha256=backend.MAMBA_BINARY_SHA256,
    required_functions=(
        "selective_scan_fn",
        "selective_state_update",
        "mamba_inner_fn",
    ),
)

CONV_SPEC = KernelSpec(
    label="CONV",
    repo_id="kernels-community/causal-conv1d",
    scientific_revision=CONV_SCIENTIFIC_REVISION,
    transport_revisions=CONV_TRANSPORT_REVISIONS,
    package_name="causal_conv1d",
    binary_sha256=backend.CONV_BINARY_SHA256,
    required_functions=(
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ),
)


def _kernel_package_version() -> str:
    try:
        return importlib.metadata.version("kernels")
    except importlib.metadata.PackageNotFoundError as exc:
        raise KernelCompatibilityError("KERNELS_PACKAGE_MISSING") from exc


def _cache_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []

    for name in ("KERNELS_CACHE", "HF_KERNELS_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home).expanduser() / "hub")

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    out: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        key = str(raw)
        if key not in seen:
            seen.add(key)
            out.append(raw)
    return tuple(out)


def _repo_cache_component(repo_id: str, prefix: str) -> str:
    parts = repo_id.split("/")
    require(len(parts) == 2 and all(parts), f"BAD_REPO_ID:{repo_id}")
    return f"{prefix}--{parts[0]}--{parts[1]}"


def _variant_dir(snapshot: Path) -> Path:
    return snapshot / "build" / BUILD_VARIANT


def _module_init_candidates(
    snapshot: Path,
    spec: KernelSpec,
) -> tuple[Path, ...]:
    variant = _variant_dir(snapshot)
    # Historical kernels==0.10.2 layout:
    #   build/<variant>/<package>/__init__.py
    # Migrated kernel repos may also expose:
    #   build/<variant>/__init__.py
    return (
        variant / spec.package_name / "__init__.py",
        variant / "__init__.py",
    )


def _select_module_init(snapshot: Path, spec: KernelSpec) -> Path:
    present = [
        path
        for path in _module_init_candidates(snapshot, spec)
        if path.is_file()
    ]
    require(
        len(present) >= 1,
        f"{spec.label}_EXACT_BUILD_MODULE_MISSING:{_variant_dir(snapshot)}",
    )
    return present[0]


def _binary_path(snapshot: Path, spec: KernelSpec) -> Path:
    variant = _variant_dir(snapshot)
    require(
        variant.is_dir(),
        f"{spec.label}_BUILD_VARIANT_MISSING:{variant}",
    )

    binaries = sorted(variant.rglob("*.so"))
    require(
        len(binaries) == 1,
        f"{spec.label}_BINARY_COUNT:{len(binaries)}:{variant}",
    )

    observed = backend.sha256_file(binaries[0])
    require(
        observed == spec.binary_sha256,
        (
            f"{spec.label}_BINARY_SHA256:"
            f"expected={spec.binary_sha256}:observed={observed}"
        ),
    )
    return binaries[0]


def _validate_snapshot(snapshot: Path, spec: KernelSpec) -> None:
    require(snapshot.is_dir(), f"{spec.label}_SNAPSHOT_NOT_DIRECTORY:{snapshot}")
    _select_module_init(snapshot, spec)
    _binary_path(snapshot, spec)


def _legacy_local_candidates(
    spec: KernelSpec,
) -> tuple[ResolvedKernelSnapshot, ...]:
    out: list[ResolvedKernelSnapshot] = []
    for root in _cache_roots():
        out.append(
            ResolvedKernelSnapshot(
                path=(
                    root
                    / _repo_cache_component(spec.repo_id, "models")
                    / "snapshots"
                    / spec.scientific_revision
                ),
                transport_revision=spec.scientific_revision,
                transport_repo_type="legacy-model-local",
                source="legacy_model_cache",
            )
        )
    return tuple(out)


def _transport_local_candidates(
    spec: KernelSpec,
) -> tuple[ResolvedKernelSnapshot, ...]:
    out: list[ResolvedKernelSnapshot] = []
    for root in _cache_roots():
        for revision in spec.transport_revisions:
            out.append(
                ResolvedKernelSnapshot(
                    path=(
                        root
                        / _repo_cache_component(spec.repo_id, "kernels")
                        / "snapshots"
                        / revision
                    ),
                    transport_revision=revision,
                    transport_repo_type=TRANSPORT_REPO_TYPE,
                    source="kernel_repo_cache",
                )
            )
    return tuple(out)


def _hub_snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(**kwargs))


def _accept_local_candidate(
    candidate: ResolvedKernelSnapshot,
    spec: KernelSpec,
) -> ResolvedKernelSnapshot | None:
    if not candidate.path.is_dir():
        return None

    try:
        _validate_snapshot(candidate.path, spec)
    except KernelCompatibilityError:
        return None

    return candidate


def resolve_exact_snapshot(
    spec: KernelSpec,
    *,
    rejected_paths: frozenset[Path] = frozenset(),
) -> ResolvedKernelSnapshot:
    require(
        _kernel_package_version() == KERNELS_VERSION,
        f"KERNELS_VERSION:{_kernel_package_version()}",
    )

    rejected = {
        path.resolve()
        for path in rejected_paths
    }

    # If the exact historical model-repo snapshot is still cached locally,
    # use it first. This is the closest possible reproduction of the original
    # validated backend and requires no reinterpretation of the old revision.
    for candidate in _legacy_local_candidates(spec):
        accepted = _accept_local_candidate(candidate, spec)
        if (
            accepted is not None
            and accepted.path.resolve() not in rejected
        ):
            return accepted

    # Next accept only migrated kernel-repo snapshots whose build bytes match
    # the already-frozen .so SHA256.
    for candidate in _transport_local_candidates(spec):
        accepted = _accept_local_candidate(candidate, spec)
        if (
            accepted is not None
            and accepted.path.resolve() not in rejected
        ):
            return accepted

    allow_patterns = [
        f"build/{BUILD_VARIANT}/*",
        f"build/{BUILD_VARIANT}/**",
    ]
    failures: list[str] = []

    # Never query mutable main and never reinterpret the historical scientific
    # revision as a kernel-repo revision. Only immutable transport commits that
    # were independently shown to contain the exact frozen LFS payload are
    # eligible.
    for revision in spec.transport_revisions:
        try:
            downloaded = Path(
                _hub_snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type=TRANSPORT_REPO_TYPE,
                    revision=revision,
                    token=False,
                    allow_patterns=allow_patterns,
                    local_files_only=False,
                )
            )
        except Exception as exc:
            failures.append(
                f"{revision}:DOWNLOAD:{type(exc).__name__}"
            )
            continue

        if downloaded.name != revision:
            failures.append(
                f"{revision}:RESOLVED_AS:{downloaded.name}"
            )
            continue

        try:
            _validate_snapshot(downloaded, spec)
        except KernelCompatibilityError as exc:
            failures.append(f"{revision}:VALIDATION:{exc}")
            continue

        resolved = ResolvedKernelSnapshot(
            path=downloaded,
            transport_revision=revision,
            transport_repo_type=TRANSPORT_REPO_TYPE,
            source="kernel_repo_download",
        )
        if resolved.path.resolve() in rejected:
            failures.append(
                f"{revision}:REJECTED_MODULE_SURFACE"
            )
            continue
        return resolved

    raise KernelCompatibilityError(
        (
            f"KERNEL_TRANSPORT_UNAVAILABLE:{spec.label}:"
            f"scientific_revision={spec.scientific_revision}:"
            f"failures={'|'.join(failures)}"
        )
    )


def _import_from_path(module_name: str, file_path: Path) -> ModuleType:
    from kernels.utils import import_from_path

    return import_from_path(module_name, file_path)


def load_exact_module(
    spec: KernelSpec,
    *,
    importer: Callable[[str, Path], ModuleType] | None = None,
) -> tuple[ModuleType, ResolvedKernelSnapshot]:
    loader = _import_from_path if importer is None else importer
    rejected_paths: set[Path] = set()
    surface_failures: list[str] = []

    while True:
        try:
            resolved = resolve_exact_snapshot(
                spec,
                rejected_paths=frozenset(rejected_paths),
            )
        except KernelCompatibilityError as exc:
            if not surface_failures:
                raise
            raise KernelCompatibilityError(
                (
                    f"KERNEL_MODULE_UNAVAILABLE:{spec.label}:"
                    f"scientific_revision={spec.scientific_revision}:"
                    f"failures={'|'.join(surface_failures)}:"
                    f"resolution={exc}"
                )
            ) from exc

        init_path = _select_module_init(resolved.path, spec)

        # Re-authenticate the exact compiled binary immediately before import.
        _binary_path(resolved.path, spec)

        # Preserve the historical canonical import name for the first attempt.
        # A fallback candidate gets a revision-qualified name so a rejected
        # wrapper cannot contaminate the next immutable candidate via module
        # caching. The loaded binary/function identity remains the frozen one.
        module_name = (
            spec.package_name
            if not surface_failures
            else (
                f"{spec.package_name}_"
                f"{resolved.transport_revision[:12]}"
            )
        )
        module = loader(module_name, init_path)

        try:
            module_file = getattr(module, "__file__", None)
            require(
                module_file is not None,
                f"{spec.label}_MODULE_FILE_MISSING",
            )

            allowed_module_paths = tuple(
                candidate.resolve()
                for candidate in _module_init_candidates(
                    resolved.path,
                    spec,
                )
                if candidate.is_file()
            )
            require(
                len(allowed_module_paths) >= 1,
                f"{spec.label}_AUTHORIZED_MODULE_SURFACE_EMPTY",
            )

            observed_module_path = Path(module_file).resolve()
            require(
                observed_module_path in allowed_module_paths,
                (
                    f"{spec.label}_MODULE_PATH:"
                    f"allowed={','.join(str(path) for path in allowed_module_paths)}:"
                    f"observed={observed_module_path}"
                ),
            )

            missing = [
                name
                for name in spec.required_functions
                if not callable(getattr(module, name, None))
            ]
            require(
                not missing,
                (
                    f"{spec.label}_FUNCTION_SURFACE:"
                    f"{','.join(missing)}"
                ),
            )
        except KernelCompatibilityError as exc:
            rejected_paths.add(resolved.path.resolve())
            surface_failures.append(
                f"{resolved.transport_revision}:{exc}"
            )
            continue

        return module, resolved


def patch_transformers_mamba(
    mamba: ModuleType,
    conv: ModuleType,
    *,
    modeling_module: Any | None = None,
) -> dict[str, Any]:
    if modeling_module is None:
        import transformers.models.mamba.modeling_mamba as modeling_module

    functions = {
        "selective_scan_fn": getattr(mamba, "selective_scan_fn", None),
        "selective_state_update": getattr(mamba, "selective_state_update", None),
        "mamba_inner_fn": getattr(mamba, "mamba_inner_fn", None),
        "causal_conv1d_fn": getattr(conv, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(conv, "causal_conv1d_update", None),
    }
    require(
        all(callable(value) for value in functions.values()),
        "KERNEL_FUNCTION_SURFACE",
    )

    for name, fn in functions.items():
        setattr(modeling_module, name, fn)

    return functions


def _validate_exact_kernel_bundle(kernels: dict[str, Any]) -> None:
    require(
        kernels.get("transport_identity_status")
        == "EXACT_FROZEN_BINARY_SHA256_MATCH",
        "KERNEL_BUNDLE_TRANSPORT_IDENTITY",
    )

    mamba = kernels.get("mamba")
    conv = kernels.get("conv")

    require(mamba is not None, "KERNEL_BUNDLE_MAMBA_MODULE")
    require(conv is not None, "KERNEL_BUNDLE_CONV_MODULE")

    required = {
        "selective_scan_fn": getattr(mamba, "selective_scan_fn", None),
        "selective_state_update": getattr(
            mamba,
            "selective_state_update",
            None,
        ),
        "mamba_inner_fn": getattr(mamba, "mamba_inner_fn", None),
        "causal_conv1d_fn": getattr(conv, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(
            conv,
            "causal_conv1d_update",
            None,
        ),
    }
    missing = [
        name
        for name, value in required.items()
        if not callable(value)
    ]
    require(
        not missing,
        "KERNEL_BUNDLE_FUNCTION_SURFACE:" + ",".join(missing),
    )


@contextmanager
def exact_transformers_kernel_loader(
    kernels: dict[str, Any],
    *,
    modeling_module: Any | None = None,
) -> Iterator[list[str]]:
    _validate_exact_kernel_bundle(kernels)

    if modeling_module is None:
        import transformers.models.mamba.modeling_mamba as modeling_module

    original_loader = modeling_module.lazy_load_kernel
    calls: list[str] = []

    def exact_loader(
        kernel_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> ModuleType:
        require(
            not args and not kwargs,
            (
                "TRANSFORMERS_KERNEL_LOADER_ARGUMENTS:"
                f"{kernel_name}:args={args}:kwargs={sorted(kwargs)}"
            ),
        )
        calls.append(kernel_name)

        if kernel_name == "causal-conv1d":
            return kernels["conv"]
        if kernel_name == "mamba-ssm":
            return kernels["mamba"]

        raise KernelCompatibilityError(
            f"TRANSFORMERS_KERNEL_NAME:{kernel_name}"
        )

    modeling_module.lazy_load_kernel = exact_loader
    try:
        yield calls
    finally:
        modeling_module.lazy_load_kernel = original_loader


def validate_transformers_kernel_bindings(
    kernels: dict[str, Any],
    *,
    modeling_module: Any | None = None,
) -> None:
    _validate_exact_kernel_bundle(kernels)

    if modeling_module is None:
        import transformers.models.mamba.modeling_mamba as modeling_module

    exact = {
        "mamba_ssm": kernels["mamba"],
        "causal_conv1d": kernels["conv"],
        "selective_scan_fn": kernels["selective_scan_fn"],
        "selective_state_update": kernels["selective_state_update"],
        "mamba_inner_fn": kernels["mamba_inner_fn"],
        "causal_conv1d_fn": kernels["causal_conv1d_fn"],
        "causal_conv1d_update": kernels["causal_conv1d_update"],
    }

    mismatched = [
        name
        for name, expected in exact.items()
        if getattr(modeling_module, name, None) is not expected
    ]
    require(
        not mismatched,
        "TRANSFORMERS_KERNEL_BINDING:" + ",".join(mismatched),
    )


def load_exact_fast_kernels() -> dict[str, Any]:
    mamba, mamba_resolved = load_exact_module(MAMBA_SPEC)
    conv, conv_resolved = load_exact_module(CONV_SPEC)
    functions = patch_transformers_mamba(mamba, conv)

    return {
        "mamba": mamba,
        "conv": conv,
        **functions,
        "mamba_transport_revision":
            mamba_resolved.transport_revision,
        "mamba_transport_repo_type":
            mamba_resolved.transport_repo_type,
        "mamba_transport_source":
            mamba_resolved.source,
        "causal_conv_transport_revision":
            conv_resolved.transport_revision,
        "causal_conv_transport_repo_type":
            conv_resolved.transport_repo_type,
        "causal_conv_transport_source":
            conv_resolved.source,
        "transport_identity_status":
            "EXACT_FROZEN_BINARY_SHA256_MATCH",
    }
