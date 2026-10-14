"""CPU-only G2-H1 pair-split observation/aggregation regression tests.

Tests streaming-safe identities without pretrained Mamba, tokenizer, checkpoint,
GPU, dataset, training, evaluation, or forward-path changes.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from contramamba.gen5_g2_h1_layer22_output_observer import (
    CELLS,
    G2H1PairSummaryAccumulator,
    Layer22OutputObserver,
    ObservationContractError,
)
from test_gen5_g2_h1_layer22_output_observer import TinyG2Wrapper


def _summary(value: float) -> dict:
    return {
        src: {
            region: {"tokens": 1, "mean": value, "mean_abs": abs(value),
                     "rms": abs(value)}
            for region in ("valid", "claim", "evidence", "other")
        }
        for src in ("native", "wrapped", "contribution")
    }


def _one_row_observation(pair: str, cell: str, value: float):
    wrapper = TinyG2Wrapper().eval().requires_grad_(False)
    x = torch.tensor([[[value, -value], [2 * value, value]]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.int64)
    claim = torch.tensor([[1, 0]], dtype=torch.int64)
    evidence = torch.tensor([[0, 1]], dtype=torch.int64)
    observer = Layer22OutputObserver(
        wrapper, attention_mask=mask, claim_mask=claim, evidence_mask=evidence,
        pair_ids=[pair], cell_ids=[cell], row_ids=[pair + ':' + cell],
        allow_partial_pairs=True,
    )
    with torch.inference_mode(), wrapper.active_mask(mask), observer:
        hooked = wrapper(x)
    record = observer.record()
    assert record.pair_differences == {}
    assert torch.equal(record.wrapped, hooked)
    assert not wrapper._forward_hooks and not wrapper.native_mixer._forward_hooks
    return record


def test_split_c0_c6_forward_observation_and_join():
    accumulator = G2H1PairSummaryAccumulator()
    c6 = _one_row_observation('p1', 'C6_EXPLICIT_DENIAL', 4.)
    c0 = _one_row_observation('p1', 'C0_SHAM', 2.)
    accumulator.add_batch(c6.per_row, pair_ids=['p1'],
                          cell_ids=['C6_EXPLICIT_DENIAL'], row_ids=['r6'])
    accumulator.add_batch(c0.per_row, pair_ids=['p1'],
                          cell_ids=['C0_SHAM'], row_ids=['r0'])
    answer = accumulator.finalize(expected_pairs=1, require_all_seven_cells=False)
    expect = (c6.per_row[0]['contribution']['claim']['mean'] -
              c0.per_row[0]['contribution']['claim']['mean'])
    assert answer['p1']['contribution']['claim']['mean_C6_minus_C0'] == pytest.approx(expect)
    with pytest.raises(ObservationContractError, match='ALREADY_FINALIZED'):
        accumulator.finalize(expected_pairs=1, require_all_seven_cells=False)


def test_default_observer_still_rejects_partial_pair():
    w = TinyG2Wrapper().eval().requires_grad_(False)
    valid = torch.ones((1, 2), dtype=torch.int64)
    with pytest.raises(ObservationContractError, match='MISSING_C0_OR_C6'):
        Layer22OutputObserver(w, attention_mask=valid, claim_mask=valid,
                              evidence_mask=valid, pair_ids=['p'],
                              cell_ids=['C0_SHAM'])


def test_full_840_rows_across_32row_batches_and_worker_shards():
    order = ('C0_SHAM', 'C1_TITLE', 'C2_NAME', 'C3_ROLE', 'C4_PREDICATE',
             'C5_TITLE_NAME', 'C6_EXPLICIT_DENIAL')
    ids = [(f'p{i:03d}', cell, f'row{i:03d}_{cell}', i * 10 + j)
           for i in range(120) for j, cell in enumerate(order)]
    assert len(ids) == 840
    accumulator = G2H1PairSummaryAccumulator()
    for start, end in ((0, 416), (416, 840)):
        for lo in range(start, end, 32):
            batch = ids[lo:min(lo + 32, end)]
            accumulator.add_batch(
                [_summary(float(value)) for _, _, _, value in batch],
                pair_ids=[p for p, _, _, _ in batch],
                cell_ids=[c for _, c, _, _ in batch],
                row_ids=[r for _, _, r, _ in batch],
            )
    merged = accumulator.finalize(expected_pairs=120)
    assert len(merged) == 120
    for pair_idx in (0, 4, 59, 60, 119):
        pair = f'p{pair_idx:03d}'
        assert merged[pair]['native']['valid']['mean_C6_minus_C0'] == 6
        assert merged[pair]['contribution']['claim']['mean_C6_minus_C0'] == 6


@pytest.mark.parametrize('kind', ['duplicate_row', 'duplicate_pair_cell'])
def test_across_batch_duplicate_rejected_and_accumulator_intact(kind):
    a = G2H1PairSummaryAccumulator()
    row0 = _summary(1.0)
    a.add_batch([row0], pair_ids=['p0'], cell_ids=['C0_SHAM'], row_ids=['r0'])
    if kind == 'duplicate_row':
        new_pair, new_cell, new_row = 'p1', 'C0_SHAM', 'r0'
        message = 'DUPLICATE_ROW_ID'
    else:
        new_pair, new_cell, new_row = 'p0', 'C0_SHAM', 'r1'
        message = 'DUPLICATE_PAIR_CELL'
    with pytest.raises(ObservationContractError, match=message):
        a.add_batch([_summary(2.0)], pair_ids=[new_pair],
                    cell_ids=[new_cell], row_ids=[new_row])
    a.add_batch([_summary(2.0)], pair_ids=['p0'],
                cell_ids=['C6_EXPLICIT_DENIAL'], row_ids=['r6'])
    result = a.finalize(expected_pairs=1, require_all_seven_cells=False)
    assert result['p0']['native']['valid']['mean_C6_minus_C0'] == 1


def test_fail_closed_on_missing_cell_even_when_c0_and_c6_exist():
    a = G2H1PairSummaryAccumulator()
    a.add_batch([_summary(0.), _summary(1.)], pair_ids=['p', 'p'],
                cell_ids=['C0_SHAM', 'C6_EXPLICIT_DENIAL'], row_ids=['r0','r6'])
    with pytest.raises(ObservationContractError, match='INCOMPLETE_PAIR'):
        a.finalize(expected_pairs=1)
    assert len(a.finalize(expected_pairs=1, require_all_seven_cells=False)) == 1


def test_accumulator_snapshots_small_summaries_no_tensor_storage():
    a = G2H1PairSummaryAccumulator()
    row = _summary(1.)
    a.add_batch([row], pair_ids=['p'], cell_ids=['C0_SHAM'], row_ids=['r0'])
    row['native']['valid']['mean'] = 100000.
    a.add_batch([_summary(3.)], pair_ids=['p'],
                cell_ids=['C6_EXPLICIT_DENIAL'], row_ids=['r6'])
    assert a.finalize(expected_pairs=1, require_all_seven_cells=False)['p']['native']['valid']['mean_C6_minus_C0'] == 2


def test_nonfinite_summary_rejected():
    a = G2H1PairSummaryAccumulator()
    row = _summary(2.)
    row['contribution']['valid']['rms'] = float('nan')
    with pytest.raises(ObservationContractError, match='REGION_NONFINITE'):
        a.add_batch([row], pair_ids=['p'], cell_ids=['C0_SHAM'], row_ids=['r0'])
