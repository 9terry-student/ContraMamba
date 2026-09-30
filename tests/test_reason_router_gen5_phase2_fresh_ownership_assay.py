from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import reason_router_gen5_phase2_fresh_ownership_assay as assay


def orthogonal_full_bases():
    r22 = torch.zeros((assay.STATE_WIDTH, 2), dtype=torch.float64)
    c22 = torch.zeros((assay.STATE_WIDTH, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    return r22, c22


def test_pair_range_and_shards_are_exact():
    pairs = assay.expected_pairs()
    assert len(pairs) == 300
    assert pairs[0] == "xg1_fact_8701"
    assert pairs[-1] == "xg1_fact_9000"
    s0 = assay.shard_pairs(0)
    s1 = assay.shard_pairs(1)
    assert len(s0) == len(s1) == 150
    assert s0[0] == "xg1_fact_8701"
    assert s0[-1] == "xg1_fact_8850"
    assert s1[0] == "xg1_fact_8851"
    assert s1[-1] == "xg1_fact_9000"
    assert set(s0).isdisjoint(s1)
    assert s0 + s1 == pairs


def test_training_matrix_is_exact_three_by_three():
    assert assay.TRAINING_SEEDS == (5201, 5202, 5203)
    assert assay.TRAINING_ARMS == ("G5-C0", "G5-C1", "G5-M1")
    assert len(assay.TRAINING_MATRIX) == 9
    assert set(assay.TRAINING_MATRIX) == {
        (seed, arm)
        for seed in assay.TRAINING_SEEDS
        for arm in assay.TRAINING_ARMS
    }
    assert assay.PRIMARY_ARMS == ("G5-C1", "G5-M1")


def test_shared_capture_preserves_phase1b_forward_budget():
    assert assay.FORWARDS_PER_PAIR == 160
    assert assay.SHARD_MODEL_FORWARD_BUDGET == 24000
    assert assay.FULL_MODEL_FORWARD_BUDGET == 48000
    assert assay.CPU_SCIENTIFIC_MODEL_FORWARD_BUDGET == 0


def test_correction_payload_authentication_exact():
    a = torch.arange(2 * 768, dtype=torch.float32).reshape(2, 768) / 1000.0
    b = torch.arange(24576 * 2, dtype=torch.float32).reshape(24576, 2) / 10000.0
    payload = {
        "schema_version": "GEN5_PHASE2_FINAL_CORRECTION_V1",
        "execution_commit": assay.TRAINING_EXECUTION_COMMIT,
        "parent_checkpoint_sha256": assay.PARENT_CHECKPOINT_SHA256,
        "r22_sha256": assay.R22_SHA256,
        "c22_sha256": assay.C22_SHA256,
        "seed": 5201,
        "arm": "G5-M1",
        "state_dict": {
            "A_theta.weight": a,
            "B_theta.weight": b,
        },
        "tensor_sha256": {
            "A_theta.weight": assay.tensor_sha256(a),
            "B_theta.weight": assay.tensor_sha256(b),
        },
    }
    a_sha, b_sha = assay._validate_correction_payload(
        payload,
        seed=5201,
        arm="G5-M1",
    )
    assert a_sha == assay.tensor_sha256(a)
    assert b_sha == assay.tensor_sha256(b)

    bad = dict(payload)
    bad["seed"] = 5202
    with pytest.raises(assay.OwnershipAssayError, match="CORRECTION_SEED"):
        assay._validate_correction_payload(bad, seed=5201, arm="G5-M1")


def test_discrete_a_recovery_matches_phase2_recurrence_formula():
    delta_t = torch.tensor([[0.1, -0.3]], dtype=torch.float32)
    a_matrix = torch.tensor([[-1.0, -2.0], [-0.5, -1.5]], dtype=torch.float32)
    delta_bias = torch.tensor([0.2, -0.1], dtype=torch.float32)
    observed = assay._discrete_a_from_capture(
        delta_t,
        a_matrix,
        delta_bias,
        dtype=torch.float32,
    )
    dt = torch.nn.functional.softplus(delta_t + delta_bias.unsqueeze(0))
    expected = torch.exp(a_matrix.unsqueeze(0) * dt.unsqueeze(-1))
    assert torch.allclose(observed, expected)


def test_native_donor_coefficient_is_computed_before_correction():
    r22, c22 = orthogonal_full_bases()
    native_background = torch.zeros(assay.STATE_SHAPE, dtype=torch.float32)
    donor = torch.zeros(assay.STATE_SHAPE, dtype=torch.float32)
    correction = torch.zeros(assay.STATE_SHAPE, dtype=torch.float32)
    donor.reshape(-1)[0] = 3.0
    donor.reshape(-1)[1] = -4.0
    correction.reshape(-1)[0] = 100.0
    correction.reshape(-1)[1] = 200.0
    native_background.reshape(-1)[10] = 2.0

    rr = assay.ownership_write_plan(
        native_background,
        donor,
        correction,
        r22,
        c22,
        "RR",
    )
    rc = assay.ownership_write_plan(
        native_background,
        donor,
        correction,
        r22,
        c22,
        "RC",
    )
    expected = torch.tensor([3.0, -4.0], dtype=torch.float64)
    assert torch.equal(rr["a_native"], expected)
    assert torch.equal(rc["a_native"], expected)
    assert torch.equal(rr["a_native"], rc["a_native"])
    assert torch.equal(rr["corrected_background"], rc["corrected_background"])
    assert rr["corrected_background"][0].item() == pytest.approx(100.0)
    assert rr["corrected_background"][1].item() == pytest.approx(200.0)


def test_same_correction_is_background_for_b_rr_rc():
    r22, c22 = orthogonal_full_bases()
    native_background = torch.randn(assay.STATE_SHAPE)
    donor = torch.randn(assay.STATE_SHAPE)
    correction = torch.randn(assay.STATE_SHAPE)
    plans = {
        condition: assay.ownership_write_plan(
            native_background,
            donor,
            correction,
            r22,
            c22,
            condition,
        )
        for condition in assay.CONDITIONS
    }
    assert torch.equal(plans["B"]["corrected_background"], plans["RR"]["corrected_background"])
    assert torch.equal(plans["B"]["corrected_background"], plans["RC"]["corrected_background"])
    assert torch.equal(plans["RR"]["a_native"], plans["RC"]["a_native"])


def test_m1_and_c1_projector_semantics_remain_frozen():
    shape = assay.CorrectionShape(hidden_size=3, intermediate_size=2, state_size=2, rank=2)
    r22 = torch.zeros((4, 2), dtype=torch.float64)
    c22 = torch.zeros((4, 2), dtype=torch.float64)
    r22[0, 0] = 1.0
    r22[1, 1] = 1.0
    c22[2, 0] = 1.0
    c22[3, 1] = 1.0
    raw = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    m1 = assay.StateWriteCorrection(
        arm="G5-M1",
        r22=r22,
        c22=c22,
        seed=5201,
        shape=shape,
        strict_frozen_dimensions=False,
    )
    c1 = assay.StateWriteCorrection(
        arm="G5-C1",
        r22=r22,
        c22=c22,
        seed=5201,
        shape=shape,
        strict_frozen_dimensions=False,
    )
    m1_out = m1.project(raw)
    c1_out = c1.project(raw)
    assert torch.allclose(m1_out[..., :2], torch.zeros_like(m1_out[..., :2]))
    assert torch.allclose(m1_out[..., 2:], raw[..., 2:])
    assert torch.allclose(c1_out[..., :2], raw[..., :2])
    assert torch.allclose(c1_out[..., 2:], torch.zeros_like(c1_out[..., 2:]))


def test_endpoint_algebra_and_i_a_identity():
    row = assay.endpoint(1.0, 1.4, 1.1)
    assert row["S_R"] == pytest.approx(0.4)
    assert row["S_C"] == pytest.approx(0.1)
    assert row["D_SUF22"] == pytest.approx(0.3)
    assert row["I_A"] == pytest.approx(0.3)


def _raw_item(index: int, *, c0_i: float = 0.0, c1_i: float = 0.10, m1_i: float = 0.30):
    cells = {}
    for seed_offset, seed in enumerate(assay.TRAINING_SEEDS):
        seed_delta = seed_offset * 0.03
        for arm, i_value in (
            ("G5-C0", c0_i),
            ("G5-C1", c1_i),
            ("G5-M1", m1_i),
        ):
            q_b = 0.20 + seed_delta
            q_rc = 0.30 + seed_delta
            q_rr = q_rc + i_value
            cells[assay.cell_key(seed, arm)] = {
                "seed": seed,
                "arm": arm,
                **assay.endpoint(q_b, q_rr, q_rc),
            }
    return {
        "source_pair_id": assay.expected_pairs()[index],
        "pair_index": index,
        "cells": cells,
        "row_dropped": False,
    }


def test_seed_aggregation_occurs_before_m1_minus_c1():
    raw = [_raw_item(i) for i in range(300)]
    aggregated = assay.aggregate_items(raw)
    first = aggregated[0]
    assert first["seed_averaged"]["G5-C1"]["I_A"] == pytest.approx(0.10)
    assert first["seed_averaged"]["G5-M1"]["I_A"] == pytest.approx(0.30)
    assert first["D_OWN"] == pytest.approx(0.20)


def test_confirmatory_test_uses_exactly_300_item_level_values():
    aggregated = assay.aggregate_items([_raw_item(i) for i in range(300)])
    observed = {}

    def fake_ttest(values):
        observed["count"] = len(values)
        observed["values"] = list(values)
        return {
            "t_statistic": 9.0,
            "degrees_of_freedom": 299,
            "p_one_sided_greater": 1e-9,
        }

    decision = assay.confirmatory_decision(aggregated, ttest_fn=fake_ttest)
    assert observed["count"] == 300
    assert all(value == pytest.approx(0.20) for value in observed["values"])
    assert decision["confirmatory_test"]["sample_size"] == 300
    assert decision["confirmatory_p_value_count"] == 1
    assert decision["label"] == assay.LABEL_SUPPORTED


def test_c0_is_excluded_from_confirmatory_decision():
    base = assay.aggregate_items([_raw_item(i, c0_i=0.0) for i in range(300)])
    changed = assay.aggregate_items([_raw_item(i, c0_i=-1000.0) for i in range(300)])

    def fake_ttest(values):
        return {"t_statistic": 5.0, "degrees_of_freedom": 299, "p_one_sided_greater": 1e-6}

    d1 = assay.confirmatory_decision(base, ttest_fn=fake_ttest)
    d2 = assay.confirmatory_decision(changed, ttest_fn=fake_ttest)
    assert d1 == d2


def test_negative_result_is_terminal_not_established_label():
    aggregated = assay.aggregate_items([
        _raw_item(i, c1_i=0.30, m1_i=0.10)
        for i in range(300)
    ])

    def fake_ttest(values):
        return {"t_statistic": -8.0, "degrees_of_freedom": 299, "p_one_sided_greater": 0.999}

    decision = assay.confirmatory_decision(aggregated, ttest_fn=fake_ttest)
    assert decision["mean_D_OWN"] < 0
    assert decision["label"] == assay.LABEL_NOT_ESTABLISHED
    assert decision["confirmatory_p_value_count"] == 1


def test_merge_shards_requires_canonical_order_and_exact_shared_forward_budget():
    def rows(pairs, start):
        return [
            {
                "source_pair_id": pair,
                "pair_index": start + i,
                "scientific_model_forward_count_this_run": assay.FORWARDS_PER_PAIR,
                "row_dropped": False,
            }
            for i, pair in enumerate(pairs)
        ]

    s0 = rows(assay.shard_pairs(0), 0)
    s1 = rows(assay.shard_pairs(1), 150)
    merged = assay.merge_shards(s0, s1)
    assert len(merged) == 300
    assert sum(x["scientific_model_forward_count_this_run"] for x in merged) == 48000
    with pytest.raises(assay.OwnershipAssayError, match="SHARD0_ORDER"):
        assay.merge_shards(list(reversed(s0)), s1)


def test_output_checksums_cover_finalized_bytes_only(tmp_path):
    items = [{"source_pair_id": "synthetic", "D_OWN": 0.1}]
    summary = {
        "result": assay.RESULT_PASS,
        "execution_head": "deadbeef",
    }
    workers = [
        {"shard_id": 0, "result": "PASS"},
        {"shard_id": 1, "result": "PASS"},
    ]
    files = assay.build_output_bundle(
        items=items,
        summary=summary,
        worker_manifests=workers,
    )
    assert set(files) == {
        assay.ITEM_FILE,
        assay.SUMMARY_FILE,
        assay.WORKER0_FILE,
        assay.WORKER1_FILE,
        assay.MANIFEST_FILE,
        assay.OUTPUT_CHECKSUM_FILE,
    }
    listed = {}
    for line in files[assay.OUTPUT_CHECKSUM_FILE].decode().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    assert assay.OUTPUT_CHECKSUM_FILE not in listed
    for name, digest in listed.items():
        assert digest == hashlib.sha256(files[name]).hexdigest()


def _args(**updates):
    values = dict(
        static_verify_only=True,
        worker=False,
        run_assay=False,
        shard_id=None,
        expected_head=assay.IMPLEMENTATION_AUTHORITY_COMMIT,
        implementation_freeze_commit=None,
        model_snapshot=None,
        tokenizer_snapshot=None,
        checkpoint=None,
        training_artifact_root=Path("artifacts"),
        output_dir=None,
        shard_output=None,
        shard_meta=None,
    )
    values.update(updates)
    return argparse.Namespace(**values)


def test_static_mode_forbids_scientific_runtime_inputs():
    assay.validate_mode_args(_args())
    with pytest.raises(assay.OwnershipAssayError, match="STATIC_CHECKPOINT_FORBIDDEN"):
        assay.validate_mode_args(_args(checkpoint=Path("parent.pt")))
    with pytest.raises(assay.OwnershipAssayError, match="STATIC_OUTPUT_FORBIDDEN"):
        assay.validate_mode_args(_args(output_dir=Path("out")))


def test_scientific_mode_requires_future_execution_binding():
    args = _args(
        static_verify_only=False,
        run_assay=True,
        expected_head="future",
    )
    with pytest.raises(assay.OwnershipAssayError, match="IMPLEMENTATION_FREEZE_REQUIRED"):
        assay.validate_mode_args(args)

    args.implementation_freeze_commit = "impl"
    with pytest.raises(assay.OwnershipAssayError, match="MODEL_SNAPSHOT_REQUIRED"):
        assay.validate_mode_args(args)


def test_parser_modes_are_mutually_exclusive():
    parser = assay.build_parser()
    parsed = parser.parse_args([
        "--static-verify-only",
        "--expected-head",
        "abc",
        "--training-artifact-root",
        "artifacts",
    ])
    assert parsed.static_verify_only is True
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--static-verify-only",
            "--run-assay",
            "--expected-head",
            "abc",
            "--training-artifact-root",
            "artifacts",
        ])
