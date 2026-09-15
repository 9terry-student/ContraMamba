from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as backend


KERNELS_VERSION = backend.KERNELS_VERSION
BUILD_VARIANT = backend.BUILD_VARIANT


class KernelCompatibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise KernelCompatibilityError(message)


@dataclass(frozen=True)
class KernelSpec:
    label: str
    repo_id: str
    revision: str
    package_name: str
    binary_sha256: str
    required_functions: tuple[str, ...]


MAMBA_SPEC = KernelSpec(
    label="MAMBA",
    repo_id="kernels-community/mamba-ssm",
    revision=backend.MAMBA_REV,
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
    revision=backend.CONV_REV,
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


def _module_init_candidates(snapshot: Path, spec: KernelSpec) -> tuple[Path, ...]:
    variant = _variant_dir(snapshot)
    # kernels==0.10.2 used build/<variant>/<package>/__init__.py.
    # Current Hub kernels commonly use build/<variant>/__init__.py.
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
    # Prefer the historical v0.10.2 package layout when both are present.
    return present[0]


def _snapshot_has_exact_build(snapshot: Path, spec: KernelSpec) -> bool:
    if not snapshot.is_dir():
        return False
    if snapshot.name != spec.revision:
        return False
    return any(path.is_file() for path in _module_init_candidates(snapshot, spec))


def _local_snapshot_candidates(spec: KernelSpec) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for root in _cache_roots():
        for prefix in ("kernels", "models"):
            candidates.append(
                root
                / _repo_cache_component(spec.repo_id, prefix)
                / "snapshots"
                / spec.revision
            )
    return tuple(candidates)


def _hub_snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(**kwargs))


def resolve_exact_snapshot(spec: KernelSpec) -> Path:
    require(
        _kernel_package_version() == KERNELS_VERSION,
        f"KERNELS_VERSION:{_kernel_package_version()}",
    )

    for candidate in _local_snapshot_candidates(spec):
        if _snapshot_has_exact_build(candidate, spec):
            return candidate

    allow_patterns = [f"build/{BUILD_VARIANT}/*"]

    try:
        downloaded = Path(
            _hub_snapshot_download(
                repo_id=spec.repo_id,
                repo_type="kernel",
                revision=spec.revision,
                token=False,
                allow_patterns=allow_patterns,
                local_files_only=False,
            )
        )
    except Exception as exc:
        raise KernelCompatibilityError(
            f"KERNEL_SNAPSHOT_UNAVAILABLE:{spec.label}:{spec.revision}:{type(exc).__name__}"
        ) from exc

    require(
        downloaded.name == spec.revision,
        (
            f"{spec.label}_REVISION_RESOLUTION:"
            f"expected={spec.revision}:observed={downloaded.name}"
        ),
    )
    require(
        _snapshot_has_exact_build(downloaded, spec),
        f"{spec.label}_EXACT_BUILD_MISSING:{downloaded}",
    )
    return downloaded


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


def _import_from_path(module_name: str, file_path: Path) -> ModuleType:
    from kernels.utils import import_from_path

    return import_from_path(module_name, file_path)


def load_exact_module(
    spec: KernelSpec,
    *,
    importer: Callable[[str, Path], ModuleType] | None = None,
) -> ModuleType:
    snapshot = resolve_exact_snapshot(spec)
    init_path = _select_module_init(snapshot, spec)

    # Authenticate the exact compiled binary before importing any kernel code.
    _binary_path(snapshot, spec)

    loader = _import_from_path if importer is None else importer
    module = loader(spec.package_name, init_path)

    module_file = getattr(module, "__file__", None)
    require(
        module_file is not None,
        f"{spec.label}_MODULE_FILE_MISSING",
    )
    require(
        Path(module_file).resolve() == init_path.resolve(),
        (
            f"{spec.label}_MODULE_PATH:"
            f"expected={init_path.resolve()}:observed={Path(module_file).resolve()}"
        ),
    )

    missing = [
        name
        for name in spec.required_functions
        if not callable(getattr(module, name, None))
    ]
    require(
        not missing,
        f"{spec.label}_FUNCTION_SURFACE:{','.join(missing)}",
    )
    return module


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


def load_exact_fast_kernels() -> dict[str, Any]:
    mamba = load_exact_module(MAMBA_SPEC)
    conv = load_exact_module(CONV_SPEC)
    functions = patch_transformers_mamba(mamba, conv)
    return {
        "mamba": mamba,
        "conv": conv,
        **functions,
    }
