from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pp3_pp5_fresh_xg1_specificity_fast_cuda
    as run,
)


def _signed_probe(
    orientation: int,
    f_value: float,
) -> dict[str, object]:
    return {
        "orientation": orientation,
        "delta_h_l2":
            2.0 * run.EPSILON,
        "plus_path_efficiency":
            f_value,
        "minus_path_efficiency":
            0.0,
        "F": f_value,
        "midpoint_max_abs_residual":
            0.0,
        "pair_delta_max_abs_residual":
            0.0,
        "applied_correction_max_abs_residual":
            0.0,
        "runtime_correction_l2":
            2.0 * run.EPSILON,
        "model_forward_count":
            run.FORWARDS_PER_SIGNED_PROBE,
    }


def _direction_probe(
    key: str,
    j_value: float,
) -> dict[str, object]:
    f_plus = (
        j_value * run.EPSILON
    )
    f_minus = (
        -j_value * run.EPSILON
    )
    observed_j = (
        f_plus - f_minus
    ) / (
        2.0 * run.EPSILON
    )

    return {
        "schema_version":
            run.DIRECTION_PROBE_SCHEMA,
        "direction_key":
            key,
        "vector_sha256":
            run.DIRECTION_SHA256[key],
        "epsilon":
            run.EPSILON,
        "F_plus":
            f_plus,
        "F_minus":
            f_minus,
        "J":
            observed_j,
        "J_squared":
            observed_j * observed_j,
        "positive_probe":
            _signed_probe(
                1,
                f_plus,
            ),
        "negative_probe":
            _signed_probe(
                -1,
                f_minus,
            ),
        "model_forward_count":
            run.FORWARDS_PER_DIRECTION,
    }


def _synthetic_item(
    index: int,
) -> dict[str, object]:
    pair = run._expected_pairs()[
        index
    ]

    values = {
        "pp3_plus": 2.0,
        "pp3_minus": 1.0,
        "pp5_plus": 1.5,
        "pp5_minus": 1.0,
    }

    probes = {
        key: _direction_probe(
            key,
            values[key],
        )
        for key
        in run.DIRECTION_ORDER
    }

    observed = {
        key: float(probes[key]["J"])
        for key
        in run.DIRECTION_ORDER
    }

    c3 = (
        run.S3 / 5.0
    ) * (
        observed["pp3_plus"] ** 2
        - observed["pp3_minus"] ** 2
    )
    c5 = (
        run.S5 / 5.0
    ) * (
        observed["pp5_plus"] ** 2
        - observed["pp5_minus"] ** 2
    )

    return {
        "schema_version":
            run.ITEM_SCHEMA,
        "probe_seed_schema_version":
            run.PROBE_SEED_SCHEMA,
        "family_key": "xg1",
        "source_pair_id": pair,
        "pair_index": index,
        "pair_ordinal": index + 1,
        "target_plus_anchor": 10,
        "target_minus_anchor": 11,
        "reference_plus_anchor": 12,
        "reference_minus_anchor": 13,
        "implementation_authority_commit":
            run.IMPLEMENTATION_AUTHORITY_COMMIT,
        "static_preparation_freeze_commit":
            run.STATIC_PREPARATION_FREEZE_COMMIT,
        "eligibility_freeze_commit":
            run.ELIGIBILITY_FREEZE_COMMIT,
        "epsilon": run.EPSILON,
        "s3": run.S3,
        "s5": run.S5,
        "pp3_plus_sha256":
            run.PP3_PLUS_SHA256,
        "pp3_minus_sha256":
            run.PP3_MINUS_SHA256,
        "pp5_plus_sha256":
            run.PP5_PLUS_SHA256,
        "pp5_minus_sha256":
            run.PP5_MINUS_SHA256,
        "direction_order":
            list(
                run.DIRECTION_ORDER
            ),
        "direction_probes":
            probes,
        "J_PP3_PLUS":
            observed["pp3_plus"],
        "J_PP3_MINUS":
            observed["pp3_minus"],
        "J_PP5_PLUS":
            observed["pp5_plus"],
        "J_PP5_MINUS":
            observed["pp5_minus"],
        "C_PP3": c3,
        "C_PP5": c5,
        "D_SPEC": c3 - c5,
        "baseline_model_forward_count_this_run":
            0,
        "scientific_model_forward_count_this_run":
            run.FORWARDS_PER_PAIR,
    }


def test_frozen_contract_and_budget():
    assert run.SOURCE_PAIR_COUNT == 300
    assert run.AMBIENT_DIM == 395
    assert run.EPSILON == 0.025

    assert run.S3 == (
        0.98692852916688512
    )
    assert run.S5 == (
        0.99986792842854511
    )

    assert run.DIRECTION_ORDER == (
        "pp3_plus",
        "pp3_minus",
        "pp5_plus",
        "pp5_minus",
    )

    assert (
        run.FORWARDS_PER_SIGNED_PROBE
        == 2
    )
    assert (
        run.FORWARDS_PER_DIRECTION
        == 4
    )
    assert (
        run.FORWARDS_PER_PAIR
        == 16
    )
    assert (
        run.SCIENTIFIC_FORWARD_BUDGET
        == 4800
    )
    assert (
        run.BASELINE_FORWARD_BUDGET_THIS_RUN
        == 0
    )


def test_exact_fresh_pair_population():
    pairs = run._expected_pairs()

    assert len(pairs) == 300
    assert pairs[0] == (
        "xg1_fact_301"
    )
    assert pairs[-1] == (
        "xg1_fact_600"
    )

    assert pairs == tuple(
        f"xg1_fact_{index:03d}"
        for index in range(
            301,
            601,
        )
    )


def test_frozen_dependency_blobs():
    expected = {
        run.AUTHORITY_PATH:
            run.AUTHORITY_BLOB,
        run.PRIOR_RUNNER_PATH:
            run.PRIOR_RUNNER_BLOB,
        run.FRESH_ELIGIBILITY_GATE_PATH:
            run.FRESH_ELIGIBILITY_GATE_BLOB,
    }

    for path, want in expected.items():
        got = subprocess.check_output(
            [
                "git",
                "rev-parse",
                f"HEAD:{path}",
            ],
            cwd=run.ROOT,
            text=True,
        ).strip()

        assert got == want


def test_fresh_structural_and_anchor_artifacts():
    facts, rows, manifest = (
        run.eligibility
        .load_frozen_fresh_inputs()
    )

    assert len(facts) == 300
    assert len(rows) == 1800
    assert (
        facts[0]["pair_id"]
        == "xg1_fact_301"
    )
    assert (
        facts[-1]["pair_id"]
        == "xg1_fact_600"
    )

    assert manifest["result"] == (
        "PASS_XG1_FRESH_SPECIFICITY_"
        "STRUCTURAL_FREEZE"
    )

    anchors = (
        run.load_fresh_anchor_manifest()
    )

    assert len(anchors) == 1800
    assert all(
        row["post4_eligible"]
        is True
        for row in anchors
    )
    assert all(
        row["exclusion_code"]
        is None
        for row in anchors
    )


def test_pp3_pp5_vectors_are_exact_frozen_bytes():
    vectors = (
        run.load_frozen_vectors()
    )

    for key in run.DIRECTION_ORDER:
        vector = vectors[key]
        assert (
            vector.dtype
            == torch.float64
        )
        assert tuple(
            vector.shape
        ) == (395,)
        assert float(
            torch.linalg.vector_norm(
                vector
            ).item()
        ) == pytest.approx(
            1.0,
            abs=1.0e-12,
        )

    assert (
        vectors["pp3_metrics"][
            "plus_pivot"
        ]
        == 267
    )
    assert (
        vectors["pp3_metrics"][
            "minus_pivot"
        ]
        == 23
    )
    assert (
        vectors["pp5_metrics"][
            "plus_pivot"
        ]
        == 319
    )
    assert (
        vectors["pp5_metrics"][
            "minus_pivot"
        ]
        == 197
    )


def test_direction_j_uses_exact_two_orientations(
    monkeypatch,
):
    calls: list[int] = []

    def fake_signed_probe(
        seed,
        direction,
        *,
        orientation,
        **kwargs,
    ):
        calls.append(
            orientation
        )

        f_value = (
            0.2
            if orientation == 1
            else -0.1
        )

        return _signed_probe(
            orientation,
            f_value,
        )

    monkeypatch.setattr(
        run.prior,
        "_run_signed_probe",
        fake_signed_probe,
    )

    vector = torch.zeros(
        395,
        dtype=torch.float64,
    )
    vector[0] = 1.0

    result = run._run_direction_j(
        {
            "family_key": "xg1",
            "source_pair_id":
                "xg1_fact_301",
        },
        vector,
        direction_key="pp3_plus",
        vector_sha256=
            run.PP3_PLUS_SHA256,
        model=None,
        runtime_ctx={},
        trace_code=None,
        trace_line=0,
        encoded={},
        row_index={},
        events={},
        budget=None,
    )

    assert calls == [1, -1]

    expected_j = (
        0.2 - (-0.1)
    ) / (
        2.0 * run.EPSILON
    )

    assert result["J"] == (
        expected_j
    )
    assert result["J_squared"] == (
        expected_j ** 2
    )
    assert (
        result[
            "model_forward_count"
        ]
        == 4
    )


def test_pair_direction_order_and_endpoint_algebra(
    monkeypatch,
):
    observed: list[str] = []

    values = {
        "pp3_plus": 2.0,
        "pp3_minus": 1.0,
        "pp5_plus": 1.5,
        "pp5_minus": 1.0,
    }

    def fake_direction(
        seed,
        vector,
        *,
        direction_key,
        **kwargs,
    ):
        observed.append(
            direction_key
        )
        return _direction_probe(
            direction_key,
            values[
                direction_key
            ],
        )

    monkeypatch.setattr(
        run,
        "_run_direction_j",
        fake_direction,
    )

    vector = torch.zeros(
        395,
        dtype=torch.float64,
    )
    vector[0] = 1.0

    vectors = {
        key: vector
        for key
        in run.DIRECTION_ORDER
    }

    item = run._run_pair(
        {
            "schema_version":
                run.PROBE_SEED_SCHEMA,
            "family_key": "xg1",
            "source_pair_id":
                "xg1_fact_301",
            "pair_index": 0,
            "pair_ordinal": 1,
            "target_plus_anchor": 10,
            "target_minus_anchor": 11,
            "reference_plus_anchor": 12,
            "reference_minus_anchor": 13,
        },
        vectors=vectors,
        model=None,
        runtime_ctx={},
        trace_code=None,
        trace_line=0,
        encoded={},
        row_index={},
        events={},
        budget=None,
    )

    assert tuple(
        observed
    ) == run.DIRECTION_ORDER

    observed_j3p = float(
        item["J_PP3_PLUS"]
    )
    observed_j3m = float(
        item["J_PP3_MINUS"]
    )
    observed_j5p = float(
        item["J_PP5_PLUS"]
    )
    observed_j5m = float(
        item["J_PP5_MINUS"]
    )

    c3 = (
        run.S3 / 5.0
    ) * (
        observed_j3p ** 2
        - observed_j3m ** 2
    )
    c5 = (
        run.S5 / 5.0
    ) * (
        observed_j5p ** 2
        - observed_j5m ** 2
    )

    assert item["C_PP3"] == c3
    assert item["C_PP5"] == c5
    assert item["D_SPEC"] == (
        c3 - c5
    )
    assert (
        item[
            "scientific_model_forward_count_this_run"
        ]
        == 16
    )


def test_item_summary_manifest_validation(
    tmp_path: Path,
):
    items = [
        _synthetic_item(index)
        for index
        in range(300)
    ]

    run._validate_items(items)

    summary = (
        run._make_summary(
            items,
            expected_head="f" * 40,
            checkpoint_sha256=
                "1" * 64,
        )
    )

    assert (
        summary[
            "scientific_model_forward_count_this_run"
        ]
        == 4800
    )
    assert (
        summary[
            "primary_inference_executed"
        ]
        is False
    )
    assert (
        summary[
            "scientific_conclusion"
        ]
        is None
    )

    output = (
        tmp_path / "artifact"
    )

    run._write_outputs(
        output,
        items=items,
        summary=summary,
    )

    validated = (
        run.validate_artifact(
            output
        )
    )

    assert (
        validated["summary"][
            "result"
        ]
        == run.RESULT_PASS
    )
    assert len(
        validated["items"]
    ) == 300


def test_no_inference_or_adaptive_rescue_logic():
    source = inspect.getsource(
        run
    ).lower()

    for forbidden in (
        "scipy",
        "ttest",
        "pvalue",
        "p_value",
        "pp1",
        "pp2",
        "pp4",
        "epsilon_sweep",
        "checkpoint_sweep",
        "layer_sweep",
        "token_sweep",
    ):
        assert forbidden not in source

    assert (
        "primary_inference_executed"
        in source
    )
    assert (
        "scientific_conclusion"
        in source
    )
