"""Gen5 G2 diagnostic-only bridge for an *already loaded* frozen parent.

This module does NOT load a checkpoint, instantiate a pretrained backbone,
perform a forward, or authorize task evaluation. Its only responsibility is
installation, device consistency, active-mask delegation and restoration.
Legacy Phase2/Phase3A functions must remain unchanged.

Preconditions: parent bytes/config and three frozen P0 source factors have
already been authenticated by a separate caller; the model is frozen/eval.
G2 must not be described as pretrained-runtime verified until the separate
strict-load and real-model numerical gates have actually executed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
from torch import nn

from contramamba.gen5_phase2_state_update_ownership import (
    Phase2Layer22MixerWrapper,
    parent_parameter_fingerprint,
)
from contramamba.gen5_shared_bypass_functional_core_diagnostic import (
    DiagnosticContractError,
    FrozenSharedBypassDiagnosticMixer,
    TARGET_LAYER,
    validate_frozen_sources,
)
from contramamba.gen5_shared_bypass_g1_installation_diagnostic import (
    _native_buffer_sha256,
    _target,
    g1_companion_active_mask,
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise DiagnosticContractError(reason)


def _single_device(module: nn.Module, reason: str) -> torch.device:
    devices = {value.device for value in module.parameters()}
    devices.update(value.device for value in module.buffers())
    _require(len(devices) == 1, f"{reason}_MULTIPLE_OR_EMPTY_DEVICES")
    device = next(iter(devices))
    _require(device.type in ("cpu", "cuda"), f"{reason}_UNSUPPORTED_DEVICE")
    return device


@dataclass
class G2LoadedParentInstallation:
    """Temporary diagnostic wrapper with exact native-mixer restoration."""

    model: nn.Module
    native: nn.Module
    wrapper: FrozenSharedBypassDiagnosticMixer
    sources: Mapping[int, nn.Module]
    parent_parameter_identity: str
    parent_buffer_identity: str
    layer_mixer_ids: tuple[int, ...]
    device: torch.device
    installed: bool = True


def install_g2_loaded_parent(
    model: nn.Module,
    *,
    source_corrections: Mapping[int, nn.Module],
    heldout_seed: int,
    arm: str,
) -> G2LoadedParentInstallation:
    """Install companion without construction/forward/device migration.

    The caller must authenticate and load the real parent independently.
    In particular this function MUST NOT itself be treated as evidence of
    pretrained checkpoint identity, strict load or real forward compatibility.
    """
    _require(not model.training, "G2_MODEL_NOT_EVAL")
    backbone, block, native = _target(model)
    _require(not backbone.training and not native.training, "G2_BACKBONE_NOT_EVAL")
    _require(not isinstance(native, (FrozenSharedBypassDiagnosticMixer,
                                     Phase2Layer22MixerWrapper)),
             "G2_ALREADY_WRAPPED")
    _require(not hasattr(native, "native_mixer"), "G2_NESTED_WRAPPER_BLOCKED")
    _require(all(not p.requires_grad for p in model.parameters()),
             "G2_PARENT_NOT_FROZEN")
    device = _single_device(model, "G2_PARENT")
    _require(_single_device(native, "G2_NATIVE") == device,
             "G2_NATIVE_DEVICE_MISMATCH")
    _require(set(source_corrections) == {6201, 6202, 6203},
             "G2_SOURCE_SEED_SET")
    # Check placement before SHA256 authentication so invalid/mixed devices
    # cannot trigger accidental large host copies or unsupported tensor reads.
    for seed, source in source_corrections.items():
        _require(_single_device(source, f"G2_SOURCE_{seed}") == device,
                 f"G2_SOURCE_{seed}_DEVICE_MISMATCH")
    validate_frozen_sources(source_corrections)

    before_parameters = parent_parameter_fingerprint(model)
    before_buffers = _native_buffer_sha256(model)
    before_layers = tuple(id(layer.mixer) for layer in backbone.layers)
    wrapper = FrozenSharedBypassDiagnosticMixer(
        native_mixer=native, source_corrections=source_corrections,
        heldout_seed=heldout_seed, arm=arm,
    ).eval()
    _require(not wrapper.training, "G2_WRAPPER_NOT_EVAL")
    block.mixer = wrapper
    try:
        _require(parent_parameter_fingerprint(model) == before_parameters,
                 "G2_INSTALL_PARENT_PARAMETER_MUTATION")
        _require(_native_buffer_sha256(model) == before_buffers,
                 "G2_INSTALL_PARENT_BUFFER_MUTATION")
        _require(all(not p.requires_grad for p in model.parameters()),
                 "G2_INSTALL_TRAINABLE_PARAMETERS")
        _require(all(id(layer.mixer) == before_layers[i] for i, layer in
                     enumerate(backbone.layers) if i != TARGET_LAYER),
                 "G2_INSTALL_NON_TARGET_LAYER_CHANGED")
    except BaseException:
        block.mixer = native
        raise
    return G2LoadedParentInstallation(
        model=model, native=native, wrapper=wrapper,
        sources=dict(source_corrections),
        parent_parameter_identity=before_parameters,
        parent_buffer_identity=before_buffers,
        layer_mixer_ids=before_layers,
        device=device,
    )


def remove_g2_loaded_parent(handle: G2LoadedParentInstallation) -> None:
    """Restore first; then fail closed on any parent/source identity violation."""
    _require(type(handle) is G2LoadedParentInstallation and handle.installed,
             "G2_HANDLE_INACTIVE")
    backbone, block, installed = _target(handle.model)
    _require(installed is handle.wrapper, "G2_WRAPPER_REPLACED")
    _require(handle.wrapper._phase2_active_mask is None,
             "G2_ACTIVE_MASK_STILL_SET")
    block.mixer = handle.native
    handle.installed = False
    _require(tuple(id(layer.mixer) for layer in backbone.layers) ==
             handle.layer_mixer_ids, "G2_RESTORE_LAYER_IDENTITY_CHANGED")
    _require(parent_parameter_fingerprint(handle.model) ==
             handle.parent_parameter_identity,
             "G2_RESTORE_PARENT_PARAMETER_MUTATION")
    _require(_native_buffer_sha256(handle.model) == handle.parent_buffer_identity,
             "G2_RESTORE_PARENT_BUFFER_MUTATION")
    _require(_single_device(handle.model, "G2_PARENT") == handle.device,
             "G2_RESTORE_PARENT_DEVICE_CHANGED")
    validate_frozen_sources(handle.sources)
    for seed, source in handle.sources.items():
        _require(_single_device(source, f"G2_SOURCE_{seed}") == handle.device,
                 f"G2_RESTORE_SOURCE_{seed}_DEVICE_CHANGED")
    _require(not handle.model.training and not backbone.training and
             not handle.native.training, "G2_RESTORE_TRAINING_MODE")
    _require(all(not p.requires_grad for p in handle.model.parameters()),
             "G2_RESTORE_TRAINABLE_PARAMETERS")


@contextmanager
def installed_g2_loaded_parent(
    model: nn.Module, *, source_corrections: Mapping[int, nn.Module],
    heldout_seed: int, arm: str,
) -> Iterator[FrozenSharedBypassDiagnosticMixer]:
    handle = install_g2_loaded_parent(
        model, source_corrections=source_corrections,
        heldout_seed=heldout_seed, arm=arm,
    )
    try:
        yield handle.wrapper
    finally:
        remove_g2_loaded_parent(handle)


# The existing G1 mask bridge has no CPU restriction; re-export exactly that
# implementation. Do not touch the historical Phase2/Phase3A mask helpers.
g2_companion_active_mask = g1_companion_active_mask
