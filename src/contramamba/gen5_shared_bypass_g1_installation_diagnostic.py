"""Gen5 G1 synthetic-parent installation/rollback companion.

This is an implementation candidate for *synthetic CPU checks only*. It does
not load a pretrained parent, run task inference, or authorize scientific runs.
Do not modify legacy Phase2/Phase3A installation and masking functions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from typing import Iterator, Mapping

import torch
from torch import nn

from contramamba.gen5_phase2_state_update_ownership import (
    parent_parameter_fingerprint,
    Phase2Layer22MixerWrapper,
)
from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
    DiagnosticContractError,
    FrozenSharedBypassDiagnosticMixer,
    TARGET_LAYER,
    validate_frozen_sources,
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise DiagnosticContractError(reason)


def _native_buffer_sha256(model: nn.Module) -> str:
    """Hash named parent buffers with legacy wrapper-name normalization."""
    h = hashlib.sha256()
    rows: list[tuple[str, torch.Tensor]] = []
    for name, value in model.named_buffers():
        if ".correction." in name:
            continue
        normalized = name.replace(".mixer.native_mixer.", ".mixer.")
        rows.append((normalized, value))
    for name, value in sorted(rows, key=lambda pair: pair[0]):
        cpu = value.detach().cpu().contiguous()
        for field in (name, str(tuple(cpu.shape)), str(cpu.dtype)):
            h.update(field.encode("utf-8"))
            h.update(b"\0")
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        h.update(b"\n")
    return h.hexdigest()


def _require_cpu_tree(module: nn.Module, label: str) -> None:
    _require(all(t.device.type == "cpu" for t in module.parameters()),
             f"{label}_PARAMETERS_NOT_CPU")
    _require(all(t.device.type == "cpu" for t in module.buffers()),
             f"{label}_BUFFERS_NOT_CPU")


def _target(model: nn.Module) -> tuple[nn.Module, nn.Module, nn.Module]:
    _require(hasattr(model, "mamba"), "G1_MAMBA_MISSING")
    backbone = model.mamba
    _require(hasattr(backbone, "layers") and len(backbone.layers) > TARGET_LAYER,
             "G1_LAYER22_MISSING")
    _require(hasattr(backbone, "config") and
             getattr(backbone.config, "use_cache", None) is False,
             "G1_CACHE_NOT_DISABLED")
    block = backbone.layers[TARGET_LAYER]
    _require(hasattr(block, "mixer") and isinstance(block.mixer, nn.Module),
             "G1_NATIVE_MIXER_MISSING")
    return backbone, block, block.mixer


def _layer_mixer_ids(backbone: nn.Module) -> tuple[int, ...]:
    return tuple(id(layer.mixer) for layer in backbone.layers)


@dataclass
class G1Installation:
    model: nn.Module
    original_native: nn.Module
    wrapper: FrozenSharedBypassDiagnosticMixer
    source_corrections: Mapping[int, nn.Module]
    parent_parameters_before: str
    parent_buffers_before: str
    original_layer_mixer_ids: tuple[int, ...]
    is_installed: bool = True


def install_g1_synthetic_diagnostic(
    model: nn.Module,
    *,
    source_corrections: Mapping[int, nn.Module],
    heldout_seed: int,
    arm: str,
) -> G1Installation:
    """Install exactly once on a pretrained-*shaped*, frozen, eval parent.

    The caller owns the model and must supply validated frozen correction
    modules. No automatic conversion, parameter freezing or model loading.
    """
    _require(not model.training, "G1_MODEL_NOT_EVAL")
    backbone, block, native = _target(model)
    _require(not backbone.training and not native.training, "G1_BACKBONE_NOT_EVAL")
    _require(not isinstance(native, (FrozenSharedBypassDiagnosticMixer,
                                     Phase2Layer22MixerWrapper)),
             "G1_ALREADY_WRAPPED")
    _require(not hasattr(native, "native_mixer"), "G1_NESTED_WRAPPER_BLOCKED")
    _require(all(not p.requires_grad for p in model.parameters()),
             "G1_PARENT_NOT_FROZEN")
    _require_cpu_tree(model, "G1_PARENT")
    validate_frozen_sources(source_corrections)
    for seed, source in source_corrections.items():
        _require_cpu_tree(source, f"G1_SOURCE_{seed}")
    param_before = parent_parameter_fingerprint(model)
    buffer_before = _native_buffer_sha256(model)
    layer_ids_before = _layer_mixer_ids(backbone)
    wrapper = FrozenSharedBypassDiagnosticMixer(
        native_mixer=native,
        source_corrections=source_corrections,
        heldout_seed=heldout_seed,
        arm=arm,
    )
    wrapper.eval()
    _require(not wrapper.training, "G1_WRAPPER_NOT_EVAL")
    block.mixer = wrapper
    try:
        _require(parent_parameter_fingerprint(model) == param_before,
                 "G1_INSTALL_PARENT_PARAMETER_MUTATION")
        _require(_native_buffer_sha256(model) == buffer_before,
                 "G1_INSTALL_PARENT_BUFFER_MUTATION")
        _require(all(not p.requires_grad for p in model.parameters()),
                 "G1_INSTALL_TRAINABLE_PARAMETERS")
        _require(all(
            id(layer.mixer) == layer_ids_before[i]
            for i, layer in enumerate(backbone.layers) if i != TARGET_LAYER
        ), "G1_NON_TARGET_LAYER_CHANGED")
    except BaseException:
        block.mixer = native
        raise
    return G1Installation(
        model=model, original_native=native, wrapper=wrapper,
        source_corrections=dict(source_corrections),
        parent_parameters_before=param_before,
        parent_buffers_before=buffer_before,
        original_layer_mixer_ids=layer_ids_before,
    )


def remove_g1_synthetic_diagnostic(handle: G1Installation) -> None:
    """Restore original mixer reference, then verify untouched parent identity."""
    _require(type(handle) is G1Installation and handle.is_installed,
             "G1_HANDLE_INACTIVE")
    backbone, block, installed = _target(handle.model)
    _require(installed is handle.wrapper, "G1_WRAPPER_REPLACED")
    _require(handle.wrapper._phase2_active_mask is None, "G1_ACTIVE_MASK_STILL_SET")
    # Always restore before examining the hashes, including failure cases.
    block.mixer = handle.original_native
    handle.is_installed = False
    _require(_layer_mixer_ids(backbone) == handle.original_layer_mixer_ids,
             "G1_RESTORE_LAYER_IDENTITY_CHANGED")
    _require(parent_parameter_fingerprint(handle.model) ==
             handle.parent_parameters_before, "G1_RESTORE_PARENT_PARAMETER_MUTATION")
    _require(_native_buffer_sha256(handle.model) ==
             handle.parent_buffers_before, "G1_RESTORE_PARENT_BUFFER_MUTATION")
    validate_frozen_sources(handle.source_corrections)
    for seed, source in handle.source_corrections.items():
        _require_cpu_tree(source, f"G1_SOURCE_{seed}")
    _require(not handle.model.training and not backbone.training and
             not handle.original_native.training, "G1_RESTORE_TRAINING_MODE")
    _require(all(not p.requires_grad for p in handle.model.parameters()),
             "G1_RESTORE_TRAINABLE_PARAMETERS")


@contextmanager
def installed_g1_synthetic_diagnostic(
    model: nn.Module,
    *, source_corrections: Mapping[int, nn.Module], heldout_seed: int, arm: str,
) -> Iterator[FrozenSharedBypassDiagnosticMixer]:
    handle = install_g1_synthetic_diagnostic(
        model, source_corrections=source_corrections,
        heldout_seed=heldout_seed, arm=arm,
    )
    try:
        yield handle.wrapper
    finally:
        remove_g1_synthetic_diagnostic(handle)


@contextmanager
def g1_companion_active_mask(
    model: nn.Module, attention_mask: torch.Tensor,
) -> Iterator[None]:
    """Distinct from legacy phase2_active_mask; does not change it."""
    _, _, installed = _target(model)
    _require(isinstance(installed, FrozenSharedBypassDiagnosticMixer),
             "G1_COMPANION_NOT_INSTALLED")
    with installed.active_mask(attention_mask):
        yield
