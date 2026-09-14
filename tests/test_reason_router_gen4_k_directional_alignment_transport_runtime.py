from types import SimpleNamespace

import numpy as np
import torch

from scripts import (
    reason_router_gen4_k_directional_alignment_transport_core
    as core,
)
from scripts import (
    reason_router_gen4_k_directional_alignment_transport_runtime
    as runtime,
)


def _mask():
    mask = torch.zeros(
        core.INTERMEDIATE_SIZE,
        dtype=torch.bool,
    )
    mask[
        : core.EXPECTED_STRONG_COUNT
    ] = True
    return mask


def test_apply_inproj_intervention_changes_only_target_strong_x():
    torch.manual_seed(17)

    output = torch.randn(
        1,
        9,
        2 * core.INTERMEDIATE_SIZE,
        dtype=torch.float32,
    )
    before = output.clone()

    delta = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )

    audit = {}
    result = runtime.apply_inproj_intervention(
        output,
        token_index=4,
        strong_mask=_mask(),
        delta_h=delta,
        plus_branch=True,
        audit=audit,
    )

    mask = _mask()
    nonstrong = ~mask

    assert torch.equal(
        result[:, :, core.INTERMEDIATE_SIZE :],
        before[:, :, core.INTERMEDIATE_SIZE :],
    )

    assert torch.equal(
        result[
            :,
            :,
            : core.INTERMEDIATE_SIZE,
        ][:, :, nonstrong],
        before[
            :,
            :,
            : core.INTERMEDIATE_SIZE,
        ][:, :, nonstrong],
    )

    assert torch.equal(
        result[:, :4, :],
        before[:, :4, :],
    )
    assert torch.equal(
        result[:, 5:, :],
        before[:, 5:, :],
    )

    assert (
        audit[
            "applied_correction_max_abs_residual"
        ]
        <= runtime.RUNTIME_CAST_TOL
    )


def test_paired_intervention_preserves_midpoint_and_realizes_delta():
    torch.manual_seed(23)

    plus = torch.randn(
        1,
        7,
        2 * core.INTERMEDIATE_SIZE,
        dtype=torch.float32,
    )
    minus = torch.randn(
        1,
        7,
        2 * core.INTERMEDIATE_SIZE,
        dtype=torch.float32,
    )
    delta = torch.randn(
        core.EXPECTED_STRONG_COUNT,
        dtype=torch.float64,
    )

    plus_audit = {}
    minus_audit = {}

    runtime.apply_inproj_intervention(
        plus,
        token_index=3,
        strong_mask=_mask(),
        delta_h=delta,
        plus_branch=True,
        audit=plus_audit,
    )
    runtime.apply_inproj_intervention(
        minus,
        token_index=3,
        strong_mask=_mask(),
        delta_h=delta,
        plus_branch=False,
        audit=minus_audit,
    )

    result = runtime.paired_intervention_audit(
        plus_audit,
        minus_audit,
        delta,
    )

    assert (
        result[
            "midpoint_max_abs_residual"
        ]
        <= runtime.MIDPOINT_TOL
    )
    assert (
        result[
            "pair_delta_max_abs_residual"
        ]
        <= runtime.RUNTIME_CAST_TOL
    )


def test_flatten_layer17_snapshots_accepts_exact_coordinate_sequence():
    token_count = 3
    forward_id = 1

    snapshots = {}

    for token in range(token_count):
        state = torch.zeros(
            1,
            core.INTERMEDIATE_SIZE,
            core.STATE_SIZE,
            dtype=torch.float32,
        )
        state[0, 0, 0] = float(token)

        snapshots[
            (
                forward_id,
                core.INTERVENTION_LAYER,
                token,
            )
        ] = state

    vectors = runtime.flatten_layer17_snapshots(
        snapshots,
        token_count=token_count,
    )

    assert len(vectors) == token_count
    assert all(
        vector.shape == (
            core.INTERMEDIATE_SIZE
            * core.STATE_SIZE,
        )
        for vector in vectors
    )
    assert all(
        vector.dtype == np.float32
        for vector in vectors
    )
    assert vectors[2][0] == 2.0


def test_make_layer17_observer_uses_requested_mixer_identity(monkeypatch):
    fake_mixer = object()

    def synthetic_forward():
        return None

    monkeypatch.setattr(
        runtime.measurement,
        "_resolve_and_validate_runtime_binding",
        lambda: (
            synthetic_forward.__code__,
            synthetic_forward.__code__.co_firstlineno,
        ),
    )

    observer = runtime.make_layer17_observer(
        fake_mixer
    )

    assert observer.enabled is True
    assert (
        observer.layers[id(fake_mixer)][
            "layer_index"
        ]
        == core.INTERVENTION_LAYER
    )
