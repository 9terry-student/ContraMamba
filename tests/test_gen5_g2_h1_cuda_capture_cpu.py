"""CPU-only guards for G2-H1 optional CUDA-to-host capture implementation.

Does NOT run on CUDA or authenticate pretrained/runtime numerical behavior.
"""
from __future__ import annotations

import pytest
import torch

from contramamba.gen5_g2_h1_layer22_output_observer import (
    Layer22OutputObserver,
    ObservationContractError,
    _binary_cpu_mask,
    _tensor_output,
)
from test_gen5_g2_h1_layer22_output_observer import TinyG2Wrapper


def _fixture(*, allow_cuda_capture: bool):
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    x = torch.arange(16, dtype=torch.float64).reshape(2, 4, 2).clone()
    valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64)
    claim = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int64)
    evidence = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.int64)
    observer = Layer22OutputObserver(
        wrapper,
        attention_mask=valid,
        claim_mask=claim,
        evidence_mask=evidence,
        pair_ids=("p", "p"),
        cell_ids=("C0_SHAM", "C6_EXPLICIT_DENIAL"),
        allow_cuda_capture=allow_cuda_capture,
    )
    return wrapper, x, valid, observer


def test_default_cpu_contract_preserved_and_optin_cpu_record_matches():
    def run(optin):
        wrapper, x, mask, observer = _fixture(allow_cuda_capture=optin)
        with torch.inference_mode(), wrapper.active_mask(mask):
            reference = wrapper(x)
        with torch.inference_mode(), wrapper.active_mask(mask), observer:
            output = wrapper(x)
        record = observer.record()
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert output.device.type == "cpu"
        assert record.native.device.type == record.wrapped.device.type == "cpu"
        assert record.contribution.device.type == "cpu"
        assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks
        return reference, record

    original, plain = run(False)
    opted, opt = run(True)
    torch.testing.assert_close(original, opted, atol=0, rtol=0)
    for field in ("native", "wrapped", "contribution"):
        torch.testing.assert_close(getattr(opt, field), getattr(plain, field),
                                   atol=0, rtol=0)
    assert opt.per_row == plain.per_row
    assert opt.pair_differences == plain.pair_differences


@pytest.mark.parametrize("invalid", [1, None, "true", 0.5])
def test_optin_flag_must_be_explicit_bool(invalid):
    w, _x, valid, _observer = _fixture(allow_cuda_capture=False)
    with pytest.raises(ObservationContractError, match="CUDA_CAPTURE_FLAG_MUST_BE_BOOL"):
        Layer22OutputObserver(w, attention_mask=valid, claim_mask=valid,
                              evidence_mask=valid,
                              pair_ids=("p", "p"),
                              cell_ids=("C0_SHAM", "C6_EXPLICIT_DENIAL"),
                              allow_cuda_capture=invalid)


def test_output_snapshot_is_detached_private_host_copy():
    output = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    clone = _tensor_output(output, (2, 4), "MOCK", allow_cuda_capture=True)
    assert clone.device.type == "cpu"
    assert clone.data_ptr() != output.data_ptr()
    with torch.no_grad():
        output[0, 0, 0] = 999.0
    assert clone[0, 0, 0].item() == 0.0
    assert not clone.requires_grad


def test_mask_snapshot_is_private_and_binary_values_checked_before_cast():
    mask = torch.tensor([[1, 1, 0]], dtype=torch.uint8)
    copied = _binary_cpu_mask(mask, (1, 3), "MASK", allow_cuda_capture=True)
    assert copied.dtype == torch.bool and copied.device.type == "cpu"
    assert copied.data_ptr() != mask.data_ptr()
    mask[0, 0] = 0
    assert bool(copied[0, 0])
    mask[0, 0] = 2
    with pytest.raises(ObservationContractError, match="MASK_NON_BINARY"):
        _binary_cpu_mask(mask, (1, 3), "MASK", allow_cuda_capture=True)


def test_output_require_grad_rejected_even_in_optin_mode():
    values = torch.ones(2, 3, 4, requires_grad=True)
    with pytest.raises(ObservationContractError, match="MOCK_REQUIRES_GRAD"):
        _tensor_output(values, (2, 3), "MOCK", allow_cuda_capture=True)


def test_cpu_observation_does_not_mutate_inputs_or_masks():
    wrapper, x, mask, observer = _fixture(allow_cuda_capture=True)
    initial_x = x.clone()
    initial_mask = mask.clone()
    with torch.inference_mode(), wrapper.active_mask(mask), observer:
        _ = wrapper(x)
    _ = observer.record()
    torch.testing.assert_close(x, initial_x, atol=0, rtol=0)
    torch.testing.assert_close(mask, initial_mask, atol=0, rtol=0)
