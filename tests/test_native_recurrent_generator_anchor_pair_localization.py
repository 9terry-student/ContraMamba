from __future__ import annotations

import inspect
import math
import re
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    audit_native_recurrent_generator_anchor_pair_localization as audit,
)


def _fact() -> dict[str, str]:
    return {
        "schema_version": "TEST",
        "generator_family": "TEST",
        "pair_id": "xg1_fact_test",
        "title": "Envoy",
        "name": "Talia Voss",
        "role": "archive custodian",
        "predicate": "certified",
        "object": "the Quasar registry unit 001",
        "time": "the first winter week",
        "location": "Juniper Reach",
        "alternate_title": "Warden",
        "alternate_name": "Renzo Mirek",
        "alternate_role": "field convener",
        "alternate_predicate": "reclassified",
    }


class _Encoding:
    def __init__(self, ids, offsets):
        self.ids = ids
        self.offsets = offsets


class _WhitespaceTokenizer:
    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        ids = []
        offsets = []
        for match in re.finditer(r"\S+", text):
            token = match.group(0)
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
            ids.append(self._vocab[token])
            offsets.append((match.start(), match.end()))
        return _Encoding(ids, offsets)


def _bundle_for_rows(rows, tokenizer):
    max_len = audit.p3a.MAX_LENGTH
    input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
    attention = torch.zeros((len(rows), max_len), dtype=torch.bool)
    claim_mask = torch.zeros_like(attention)
    evidence_mask = torch.zeros_like(attention)

    for index, row in enumerate(rows):
        claim = tokenizer.encode(row["claim"], add_special_tokens=False).ids[
            : audit.p3a.CLAIM_BUDGET
        ]
        evidence = tokenizer.encode(
            row["evidence"], add_special_tokens=False
        ).ids[: audit.p3a.EVIDENCE_BUDGET]
        combined = claim + [audit.p3a.EOS_TOKEN_ID] + evidence
        assert len(combined) <= max_len
        input_ids[index, : len(combined)] = torch.tensor(combined)
        attention[index, : len(combined)] = True
        claim_mask[index, : len(claim)] = True
        evidence_start = len(claim) + 1
        evidence_mask[
            index,
            evidence_start : evidence_start + len(evidence),
        ] = True

    return {
        "model_inputs": {
            "input_ids": input_ids,
            "attention_mask": attention,
            "claim_mask": claim_mask,
            "evidence_mask": evidence_mask,
            "final_labels": torch.zeros(len(rows), dtype=torch.long),
        },
        "row_ids": [row["row_id"] for row in rows],
        "pair_ids": [row["source_pair_id"] for row in rows],
        "contrast_cell_ids": [row["contrast_cell_id"] for row in rows],
        "stressor_active": torch.zeros(len(rows), dtype=torch.bool),
        "target_indices": torch.full((len(rows),), -1, dtype=torch.long),
    }


def _synthetic_masks(batch: int = 2):
    seq = audit.p3a.MAX_LENGTH
    role = {
        "CLAIM": torch.zeros((batch, seq), dtype=torch.bool),
        "EOS": torch.zeros((batch, seq), dtype=torch.bool),
        "EVIDENCE": torch.zeros((batch, seq), dtype=torch.bool),
    }
    fine = {
        side: {
            cls: torch.zeros((batch, seq), dtype=torch.bool)
            for cls in audit.FINE_CLASSES
        }
        for side in ("CLAIM", "EVIDENCE")
    }

    for b in range(batch):
        claim_stop = 6 + b
        eos = claim_stop
        evidence_stop = eos + 1 + 7
        role["CLAIM"][b, :claim_stop] = True
        role["EOS"][b, eos] = True
        role["EVIDENCE"][b, eos + 1 : evidence_stop] = True

        for index in range(claim_stop):
            cls = audit.FINE_CLASSES[index % len(audit.FINE_CLASSES)]
            fine["CLAIM"][cls][b, index] = True
        for offset, index in enumerate(range(eos + 1, evidence_stop)):
            cls = audit.FINE_CLASSES[offset % len(audit.FINE_CLASSES)]
            fine["EVIDENCE"][cls][b, index] = True

    attention = role["CLAIM"] | role["EOS"] | role["EVIDENCE"]
    return role, fine, attention


def _small_recurrence_inputs(*, large_span: bool = False):
    role, fine, attention = _synthetic_masks(batch=1)
    seq = audit.p3a.MAX_LENGTH
    shape = SimpleNamespace(
        intermediate_size=2,
        state_size=2,
        state_width=4,
    )
    torch.manual_seed(19)
    component = 0.1 * torch.randn(2, 1, seq, shape.state_width)
    component = torch.where(
        attention[None, :, :, None],
        component,
        torch.zeros_like(component),
    )
    if large_span:
        log_a = torch.full(
            (1, seq, shape.intermediate_size, shape.state_size),
            -0.1,
        )
        log_a[:, 3] = -1000.0
    else:
        log_a = -0.03 - 0.05 * torch.rand(
            1,
            seq,
            shape.intermediate_size,
            shape.state_size,
        )
    survival = audit.pairgap._backward_survival_factor(
        log_a_flat=log_a.reshape(1, seq, shape.state_width),
        attention_mask=attention,
    )
    return role, fine, attention, shape, component, log_a, survival


def test_frozen_contract_constants():
    assert audit.DESIGN_COMMIT == "55b8784aa154d766cd189f60b2919451720598de"
    assert audit.VALIDATED_EVIDENCE_COMMIT == "ee61ac1f70816d7150ab9dc5af57a4b9f0714515"
    assert audit.ATOMIC_ANCHORS == (
        "A_TITLE",
        "A_NAME",
        "A_ROLE",
        "A_PREDICATE",
    )
    assert audit.FINE_CLASSES == (
        "A_TITLE",
        "A_NAME",
        "A_ROLE",
        "A_PREDICATE",
        "RESIDUAL",
    )
    assert len(audit.FINE_PAIR_NAMES) == 25
    assert len(audit.FINE_CELL_NAMES) == 75
    assert audit.PARENT_PAIR_NAMES == (
        "CLAIM->CLAIM",
        "EVIDENCE->EVIDENCE",
        "CLAIM->EVIDENCE",
    )


def test_static_contract_replays_frozen_dependencies():
    audit.validate_static_contract()


def test_base_claim_generator_spans():
    fact = _fact()
    rendered, spans = audit._base_statement_and_spans(
        fact,
        cell_id=None,
    )
    assert rendered == audit.xg1.render_statement(fact)
    assert set(spans) == set(audit.ATOMIC_ANCHORS)
    for name, (start, stop) in spans.items():
        assert rendered[start:stop]
        assert start < stop


@pytest.mark.parametrize(
    "cell_id",
    list(audit.p3static.BASE_CELLS),
)
def test_c0_c5_evidence_generator_spans(cell_id):
    fact = _fact()
    rendered, spans = audit._base_statement_and_spans(
        fact,
        cell_id=cell_id,
    )
    _mask, substitutions = audit.xg1.cell_spec(cell_id)
    overrides = {
        axis: fact[source]
        for axis, source in substitutions
    }
    assert rendered == audit.xg1.render_statement(fact, **overrides)
    assert set(spans) == set(audit.ATOMIC_ANCHORS)


def test_c6_explicit_denial_spans():
    fact = _fact()
    rendered, spans = audit._c6_statement_and_spans(fact)
    assert rendered == audit.p3static.explicit_denial_evidence(fact)
    assert rendered[slice(*spans["A_TITLE"])] == fact["title"]
    assert rendered[slice(*spans["A_NAME"])] == fact["name"]
    assert rendered[slice(*spans["A_ROLE"])] == fact["role"]
    assert rendered[slice(*spans["A_PREDICATE"])] == fact["predicate"]


def test_token_assignment_uses_offsets_and_residual():
    spans = {
        "A_TITLE": (2, 4),
        "A_NAME": (6, 9),
        "A_ROLE": (12, 15),
        "A_PREDICATE": (18, 22),
    }
    offsets = [(0, 1), (2, 4), (5, 6), (6, 9), (10, 11), (18, 22)]
    observed = audit._assign_token_classes(
        offsets,
        spans,
        kept_count=len(offsets),
    )
    assert observed == [
        "RESIDUAL",
        "A_TITLE",
        "RESIDUAL",
        "A_NAME",
        "RESIDUAL",
        "A_PREDICATE",
    ]


def test_token_assignment_fails_on_multi_anchor_overlap():
    spans = {
        "A_TITLE": (0, 3),
        "A_NAME": (4, 7),
        "A_ROLE": (10, 12),
        "A_PREDICATE": (14, 16),
    }
    with pytest.raises(audit.GeneratorAnchorLocalizationError):
        audit._assign_token_classes(
            [(2, 5)],
            spans,
            kept_count=1,
        )


def test_anchor_partition_replays_active_encoding_and_c6(monkeypatch):
    fact = _fact()
    tokenizer = _WhitespaceTokenizer()

    claim, _ = audit._base_statement_and_spans(fact, cell_id=None)
    c0, _ = audit._base_statement_and_spans(fact, cell_id="C0_SHAM")
    c6, _ = audit._c6_statement_and_spans(fact)

    rows = [
        {
            "row_id": "row_c0",
            "source_pair_id": fact["pair_id"],
            "contrast_cell_id": "C0_SHAM",
            "claim": claim,
            "evidence": c0,
        },
        {
            "row_id": "row_c6",
            "source_pair_id": fact["pair_id"],
            "contrast_cell_id": "C6_EXPLICIT_DENIAL",
            "claim": claim,
            "evidence": c6,
        },
    ]
    bundle = _bundle_for_rows(rows, tokenizer)

    monkeypatch.setattr(audit, "DEV_ROWS", 2)
    monkeypatch.setattr(
        audit.anchor,
        "load_canonical_analysis_tokenizer",
        lambda _snapshot: (tokenizer, {"test": True}),
    )
    monkeypatch.setattr(audit.p3static, "TRAIN_PAIR_COUNT", 1)
    monkeypatch.setattr(
        audit.p3static,
        "build_population",
        lambda _first, _last: ([fact], []),
    )

    masks, provenance = audit._build_anchor_partition(
        static={"dev_rows": rows},
        encoded={"dev_bundle": bundle},
        tokenizer_snapshot=None,
        row_start=0,
        row_stop=2,
    )

    assert provenance["active_encoding_replay"] is True
    assert provenance["decoded_token_strings_inspected"] is False

    for row_index in range(2):
        claim_union = torch.zeros(audit.p3a.MAX_LENGTH, dtype=torch.bool)
        evidence_union = torch.zeros_like(claim_union)
        for cls in audit.FINE_CLASSES:
            claim_union |= masks["CLAIM"][cls][row_index]
            evidence_union |= masks["EVIDENCE"][cls][row_index]
        assert torch.equal(
            claim_union,
            bundle["model_inputs"]["claim_mask"][row_index],
        )
        assert torch.equal(
            evidence_union,
            bundle["model_inputs"]["evidence_mask"][row_index],
        )
        assert not torch.any(claim_union & evidence_union)


def test_fine_pair_opportunity_counts_exactly_reconstruct_parent():
    role, fine, _attention = _synthetic_masks(batch=2)
    coarse_counts = audit.coarse._pair_opportunity_counts(role)
    observed = audit._fine_pair_opportunity_counts(
        fine,
        coarse_counts,
    )

    for parent in audit.PARENT_PAIR_NAMES:
        reconstructed = [0] * (audit.p3a.MAX_LENGTH - 1)
        for fine_name in audit.FINE_PAIR_NAMES:
            for i, value in enumerate(observed[parent][fine_name]["by_gap"]):
                reconstructed[i] += value
        assert reconstructed == coarse_counts[parent]["by_gap"]


def test_fast_fine_reducer_reconstructs_all_three_parent_regions():
    role, fine, attention, shape, component, log_a, survival = (
        _small_recurrence_inputs(large_span=False)
    )
    stats = audit._component_fine_pair_stats(
        component=component,
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=role,
        fine_masks=fine,
        shape=shape,
    )
    assert len(stats) == component.shape[0]
    for row in stats:
        assert row["fine_parent_reconstruction_abs_max"] >= 0.0
        assert math.isfinite(row["fine_pair_fft_half_log_span_max"])


def test_large_span_fallback_reconstructs_all_three_parent_regions():
    role, fine, attention, shape, component, log_a, survival = (
        _small_recurrence_inputs(large_span=True)
    )
    stats = audit._component_fine_pair_stats(
        component=component[:1],
        log_a=log_a,
        survival=survival,
        attention_mask=attention,
        role_masks=role,
        fine_masks=fine,
        shape=shape,
    )
    assert len(stats) == 1
    assert (
        stats[0]["fine_pair_fft_half_log_span_max"]
        > audit.FAST_FLOAT32_HALF_SPAN_LIMIT
    )


def test_fine_spectral_matrix_uses_complex128_accumulation_before_reduction():
    # Same precomputed complex64 FFT coefficients on all comparisons; the
    # recurrence, classes, pair geometry, and normalization are unchanged.
    k_count, batch, seq_len, width = 2, 8, 32, 64
    generator = torch.Generator(device="cpu").manual_seed(1903)
    raw = 0.1 * torch.randn(
        k_count, batch, seq_len, width,
        generator=generator,
        dtype=torch.float32,
    )
    class_ids = {
        "CLAIM": torch.randint(0, 5, (batch, seq_len), generator=generator),
        "EVIDENCE": torch.randint(0, 5, (batch, seq_len), generator=generator),
    }
    class_stacks = {
        side: torch.stack([
            torch.where(
                (class_ids[side] == index)[None, :, :, None],
                raw,
                torch.zeros_like(raw),
            )
            for index in range(len(audit.FINE_CLASSES))
        ])
        for side in ("CLAIM", "EVIDENCE")
    }
    x_scale = torch.exp(
        torch.linspace(-0.4, 0.4, seq_len, dtype=torch.float32)
    )[None, :, None].expand(batch, seq_len, width)
    y_scale = 1.0 / x_scale
    survival = torch.ones(batch, seq_len, width, dtype=torch.float32)
    n_fft = 1 << ((2 * seq_len - 1).bit_length())
    ffts = {
        side: audit._fast_class_ffts(
            class_stack=class_stacks[side],
            x_scale=x_scale,
            y_scale=y_scale,
            survival_chunk=survival,
            n_fft=n_fft,
        )
        for side in ("CLAIM", "EVIDENCE")
    }

    for parent in audit.PARENT_PAIR_NAMES:
        left, right = parent.split("->")
        source_fft = ffts[left][0]
        target_fft = ffts[right][1]
        observed = audit._spectral_pair_matrix(
            source_fft, target_fft, seq_len=seq_len,
        )
        assert observed.dtype == torch.float64
        assert observed.shape == (
            len(audit.FINE_CLASSES), len(audit.FINE_CLASSES),
            k_count, seq_len - 1,
        )

        reference_spectrum = torch.einsum(
            "akbfw,ckbfw->ackf",
            torch.conj(source_fft).to(torch.complex128),
            target_fft.to(torch.complex128),
        )
        expected = 2.0 * torch.fft.irfft(
            reference_spectrum, n=n_fft, dim=-1,
        )[..., 1:seq_len]
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=1e-12)

        old_spectrum = torch.einsum(
            "akbfw,ckbfw->ackf", torch.conj(source_fft), target_fft,
        ).to(torch.complex128)
        old = 2.0 * torch.fft.irfft(
            old_spectrum, n=n_fft, dim=-1,
        )[..., 1:seq_len]

        # Independent float64 time-domain pair-sum for nine targeted cells
        # across the three parent regions. Assert an actual precision benefit.
        for source_index, target_index in ((0, 0), (3, 2), (4, 4)):
            source = (
                class_stacks[left][source_index] * x_scale.unsqueeze(0)
            ).to(torch.float64)
            target = (
                class_stacks[right][target_index]
                * survival.unsqueeze(0) * y_scale.unsqueeze(0)
            ).to(torch.float64)
            direct = torch.stack([
                2.0 * torch.sum(
                    source[:, :, :seq_len - gap, :]
                    * target[:, :, gap:, :],
                    dim=(1, 2, 3),
                    dtype=torch.float64,
                )
                for gap in range(1, seq_len)
            ], dim=1)
            improved_error = float(
                (observed[source_index, target_index] - direct)
                .abs().max().item()
            )
            original_error = float(
                (old[source_index, target_index] - direct)
                .abs().max().item()
            )
            assert improved_error < original_error, (
                parent, source_index, target_index,
                improved_error, original_error,
            )


def test_window_sums_are_frozen_coarse_windows():
    vector = list(range(1, audit.p3a.MAX_LENGTH))
    observed = audit.coarse._window_sums(vector)
    assert observed["gap1_8"] == sum(range(1, 9))
    assert observed["gap9_16"] == sum(range(9, 17))
    assert observed["gap17_32"] == sum(range(17, 33))
    assert observed["gap33_64"] == sum(range(33, 65))
    assert observed["gap65_127"] == sum(range(65, 128))
    assert observed["gap1_32"] == sum(range(1, 33))


def test_fine_component_output_reports_h_n_and_mean_h():
    fine_c = {
        parent: {
            fine: [1.0] * (audit.p3a.MAX_LENGTH - 1)
            for fine in audit.FINE_PAIR_NAMES
        }
        for parent in audit.PARENT_PAIR_NAMES
    }
    pair_counts = {
        parent: {
            fine: {
                "windows": {
                    key: 2
                    for key in audit.WINDOWS
                }
            }
            for fine in audit.FINE_PAIR_NAMES
        }
        for parent in audit.PARENT_PAIR_NAMES
    }
    observed = audit._fine_component_output(
        fine_c=fine_c,
        self_energy=2.0,
        pair_counts=pair_counts,
    )
    cell = observed["CLAIM->CLAIM"]["A_TITLE->A_TITLE"]
    assert cell["H_windows"]["gap1_8"] == pytest.approx(4.0)
    assert cell["pair_counts"]["gap1_8"] == 2
    assert cell["mean_h_windows"]["gap1_8"] == pytest.approx(2.0)


def _synthetic_final_row(group: str, source: str, shift: float):
    fine = {
        parent: {
            fine_name: {
                "H_windows": {
                    window: shift
                    for window in audit.WINDOWS
                }
            }
            for fine_name in audit.FINE_PAIR_NAMES
        }
        for parent in audit.PARENT_PAIR_NAMES
    }
    return {
        "group": group,
        "source": source,
        "target": "T",
        "log_Q_visible": shift,
        "log_Q_complement": shift,
        "L_interference": shift,
        "fine_pair_visible": fine,
        "fine_pair_complement": fine,
    }


def test_source_matched_aggregation_preserves_visible_and_complement():
    rows = []
    for source_index in range(9):
        source = f"S{source_index}"
        rows.extend([
            _synthetic_final_row("PRIMARY_A", source, -2.0),
            _synthetic_final_row("PRIMARY_A", source, -2.0),
            _synthetic_final_row("CONTROL_R", source, -1.0),
            _synthetic_final_row("CONTROL_R", source, -1.0),
        ])
    pair_counts = {
        parent: {
            fine: {
                "windows": {
                    window: 10
                    for window in audit.WINDOWS
                }
            }
            for fine in audit.FINE_PAIR_NAMES
        }
        for parent in audit.PARENT_PAIR_NAMES
    }
    observed = audit._source_matched(rows, pair_counts)
    assert len(observed) == 9
    first = observed[0]
    assert first["metrics"]["L_interference"]["primary_minus_control"] == -1.0
    for component in ("visible", "complement"):
        cell = (
            first["fine_pair_metrics"][component]
            ["CLAIM->EVIDENCE"]["A_TITLE->A_NAME"]["gap17_32"]
        )
        assert cell["primary_minus_control"] == -1.0
        assert cell["pair_opportunity_count"] == 10
        assert cell["primary_minus_control_mean_h"] == -0.1


def test_expected_forward_counts_unchanged():
    counts = audit._expected_counts()
    assert counts["batch_rows"] == 32
    assert counts["fine_cells_total"] == 75
    assert counts["workers"]["0"]["source_gradient_forwards"] == 117
    assert counts["workers"]["1"]["source_gradient_forwards"] == 126
    assert counts["total_source_gradient_forwards"] == 243
    assert counts["total_fine_group_decompositions"] == 486


def test_source_has_no_training_or_decoded_token_inspection():
    source = inspect.getsource(audit)
    forbidden = (
        ".backward(",
        "optimizer.step(",
        ".decode(",
        "convert_ids_to_tokens",
        "batch_decode(",
    )
    for token in forbidden:
        assert token not in source


def test_no_new_semantic_classes_beyond_frozen_anchors_and_residual():
    assert set(audit.FINE_CLASSES) == {
        "A_TITLE",
        "A_NAME",
        "A_ROLE",
        "A_PREDICATE",
        "RESIDUAL",
    }
    assert "A_IDENTITY" not in audit.FINE_CLASSES


def test_eos_is_excluded_from_fine_parent_pairs():
    assert all("EOS" not in parent for parent in audit.PARENT_PAIR_NAMES)
