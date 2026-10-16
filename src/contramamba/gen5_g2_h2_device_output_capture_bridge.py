"""G2-H2 read-only *device boundary* bridge; CUDA execution is NOT validated.

This companion never invokes Mamba/model forward and never mutates tensors or
parameters. It preserves the frozen H2 CPU metadata/identity validation while
reusing frozen H1's detached CUDA-to-CPU output capture primitives. Only CPU
synthetic parity is tested here. GPU use requires an independent real-model
numeric equivalence preflight and explicit scientific execution authorization.

Pass the 128-column *CPU source features* to the inherited constructor; use
those exact tokens and masks transferred to the actual model device for the
separately controlled model forward. Do not pass model-generated labels.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from contramamba.gen5_g2_h2_readonly_output_adapter import (
    H2Layer22ReadOnlyAdapter, need,
)
from contramamba.gen5_g2_h1_layer22_output_observer import (
    _binary_cpu_mask, _tensor_output,
)


class H2DeviceOutputCaptureBridge(H2Layer22ReadOnlyAdapter):
    """Capture existing native/wrapped mixer outputs to detached CPU snapshots.

    expected_output_device='cpu' is exclusively for synthetic equivalence tests.
    expected_output_device='cuda' is a *future* gated real-Mamba use and does
    not itself authorize GPU execution. CPU source metadata is always required.
    """

    def __init__(self, wrapper: torch.nn.Module, *,
                 expected_output_device: str, **cpu_metadata: Any) -> None:
        need(expected_output_device in ('cpu', 'cuda'), 'DEVICE_MODE_UNSUPPORTED')
        super().__init__(wrapper, **cpu_metadata)
        # No device migrations. Fail if wrapper, sources, or native mixer are
        # already mixed across devices; the caller owns model/source placement.
        devices = {p.device.type for p in wrapper.parameters()}
        devices.update(b.device.type for b in wrapper.buffers())
        need(devices == {expected_output_device}, 'WRAPPER_DEVICE_MISMATCH')
        self.expected_output_device = expected_output_device

    def _snapshot(self, value: Any, name: str) -> torch.Tensor:
        need(isinstance(value, torch.Tensor) and
             value.device.type == self.expected_output_device,
             name + '_DEVICE_MISMATCH')
        return _tensor_output(value, self._shape, name,
                              allow_cuda_capture=self.expected_output_device == 'cuda')

    def _native_hook(self, _module: torch.nn.Module,
                     args: tuple[Any, ...], kwargs: dict[str, Any],
                     output: Any) -> None:
        self._native_count += 1
        need(self._native_count == 1 and self._wrapped_count == 0,
             'NATIVE_FORWARD_ORDER')
        self._native_mask = self._forward_mask(args, kwargs)
        self._native_device = output.device if isinstance(output, torch.Tensor) else None
        self._native = self._snapshot(output, 'NATIVE')

    def _wrapper_hook(self, _module: torch.nn.Module,
                      args: tuple[Any, ...], kwargs: dict[str, Any],
                      output: Any) -> None:
        self._wrapped_count += 1
        need(self._native_count == 1 and self._wrapped_count == 1,
             'WRAPPER_FORWARD_ORDER')
        explicit = self._forward_mask(args, kwargs)
        need(self._native_mask is explicit, 'NATIVE_MASK_FORWARDING_CHANGED')
        need(isinstance(output, torch.Tensor) and
             output.device == self._native_device, 'NATIVE_WRAPPED_DEVICE_MISMATCH')
        effective = (explicit if explicit is not None else
                     getattr(self.wrapper, '_phase2_active_mask', None))
        need(isinstance(effective, torch.Tensor) and
             effective.device.type == self.expected_output_device,
             'EFFECTIVE_MASK_DEVICE_MISMATCH')
        cpu_effective = _binary_cpu_mask(
            effective, self._shape, 'EFFECTIVE',
            allow_cuda_capture=self.expected_output_device == 'cuda')
        need(torch.equal(cpu_effective, self.valid),
             'EFFECTIVE_MASK_IDENTITY')
        self._wrapped = self._snapshot(output, 'WRAPPED')
        need(self._native is not None and
             self._wrapped.shape == self._native.shape and
             self._wrapped.dtype == self._native.dtype,
             'NATIVE_WRAPPED_OUTPUT_MISMATCH')

    # __enter__, __exit__, record(), and metric_row() remain frozen-H2 behavior.
    # record() enforces ZERO correction exact zero. Every output snapshot is CPU.


FEATURE_KEYS = frozenset(('input_ids', 'attention_mask', 'claim_mask',
                          'evidence_mask'))


def verify_h2_runtime_transfer(cpu_features: Mapping[str, torch.Tensor],
                               model_features: Mapping[str, torch.Tensor],
                               *, expected_device: str) -> None:
    """Fail closed if actual model-bound features differ from frozen CPU bytes.

    For a future separately authorized run, construct *model_features* by
    transferring the exact CPU-validated source. Call this before any forward.
    No labels are allowed. Never allocates CUDA data or performs model calls.
    """
    need(expected_device in ('cpu', 'cuda'), 'TRANSFER_DEVICE_UNSUPPORTED')
    need(set(cpu_features) == FEATURE_KEYS and
         set(model_features) == FEATURE_KEYS, 'TRANSFER_FEATURE_KEYS')
    shape: tuple[int, int] | None = None
    for key in sorted(FEATURE_KEYS):
        src, dst = cpu_features[key], model_features[key]
        need(isinstance(src, torch.Tensor) and src.device.type == 'cpu' and
             src.ndim == 2, 'TRANSFER_SOURCE_CPU_' + key)
        need(isinstance(dst, torch.Tensor) and
             dst.device.type == expected_device and dst.ndim == 2,
             'TRANSFER_DEVICE_' + key)
        need(src.shape == dst.shape and src.dtype == dst.dtype,
             'TRANSFER_SHAPE_DTYPE_' + key)
        need(tuple(src.shape)[1] == 128 and src.shape[0] > 0,
             'TRANSFER_WIDTH_' + key)
        if shape is None:
            shape = tuple(src.shape)
        need(tuple(src.shape) == shape, 'TRANSFER_CROSS_FEATURE_SHAPE')
        if key == 'input_ids':
            need(src.dtype == torch.long, 'TRANSFER_INPUT_DTYPE')
        else:
            need(src.dtype in (torch.bool, torch.uint8, torch.int16,
                               torch.int32, torch.int64), 'TRANSFER_MASK_DTYPE')
        need(torch.equal(src, dst.detach().to(device='cpu', copy=True)),
             'TRANSFER_TOKEN_MASK_DRIFT_' + key)
