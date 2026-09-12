from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ADAPTER_PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_six_cell_tier2_inference_adapter.py"
)

SNAPSHOT_PATH = (
    ROOT
    / "src"
    / "contramamba"
    / "modeling_v6b_minimal_gen3_grouped_snapshot.py"
)

HISTORICAL_COMMIT = "3e0e9a435068c552abf20f3a74e0c3eccca344a3"
EXPECTED_SNAPSHOT_SHA = (
    "8c365bfa857157d91f363358d5db3abaab425dec3e0d7c62683b4207a589b6a5"
)
EXPECTED_HEADS_TREE = "68d26855aa511fcd41d6f395ae5f87177a162678"


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "gen4_tier2_adapter_under_test",
        ADAPTER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


A = _load_adapter()


def test_historical_snapshot_exact_sha256():
    observed = hashlib.sha256(
        SNAPSHOT_PATH.read_bytes()
    ).hexdigest()

    assert observed == EXPECTED_SNAPSHOT_SHA


def test_historical_and_current_heads_tree_identical():
    historical = subprocess.check_output(
        [
            "git",
            "rev-parse",
            f"{HISTORICAL_COMMIT}:src/contramamba/heads",
        ],
        text=True,
    ).strip()

    current = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD:src/contramamba/heads",
        ],
        text=True,
    ).strip()

    assert historical == EXPECTED_HEADS_TREE
    assert current == EXPECTED_HEADS_TREE


def test_exact_evaluator_population():
    assert A.SEEDS == (180, 181, 182)

    assert len(A.ARMS) == 6
    assert len(A.CHECKPOINT_SHA256) == 18

    assert set(A.CHECKPOINT_SHA256) == {
        (seed, arm)
        for seed in A.SEEDS
        for arm in A.ARMS
    }


def test_unknown_seed_or_arm_rejected():
    with pytest.raises(ValueError):
        A.expected_checkpoint_sha256(
            999,
            A.ARMS[0],
        )

    with pytest.raises(ValueError):
        A.expected_checkpoint_sha256(
            180,
            "UNKNOWN-ARM",
        )

    with pytest.raises(ValueError):
        A.expected_edge_gradient_lambdas(
            "UNKNOWN-ARM",
        )


def test_historical_constructor_contract_exact():
    assert A.historical_model_constructor_kwargs() == {
        "frame_size": 128,
        "predicate_size": 128,
        "sufficiency_size": 128,
        "energy_size": 64,
        "dropout": 0.1,
        "freeze_a_log": True,
        "decision_mode": "explicit_product",
        "reason_router_epsilon": 1e-8,
        "use_temporal_comparator": False,
        "use_predicate_comparator": False,
        "alpha_temporal_init": 1.25,
        "alpha_predicate_init": 1.25,
    }

    assert A.HISTORICAL_FREEZE_ENCODER is True
    assert A.HISTORICAL_GRADIENT_OWNERSHIP_LAMBDA is None


def test_resolved_edge_gradient_registry_exact():
    expected = {
        "G3-GROUP-D-HALF": {
            "F_TO_P": 1.0,
            "F_TO_S": 1.0,
            "P_TO_S": 1.0,
            "F_TO_Q": 1.0,
            "P_TO_Q": 1.0,
            "S_TO_Q": 1.0,
            "F_TO_D": 0.5,
            "P_TO_D": 0.5,
            "S_TO_D": 0.5,
            "Q_TO_D": 0.5,
        },
        "G3-GROUP-Q-D-HALF": {
            "F_TO_P": 1.0,
            "F_TO_S": 1.0,
            "P_TO_S": 1.0,
            "F_TO_Q": 0.5,
            "P_TO_Q": 0.5,
            "S_TO_Q": 0.5,
            "F_TO_D": 0.5,
            "P_TO_D": 0.5,
            "S_TO_D": 0.5,
            "Q_TO_D": 0.5,
        },
        "G3-GROUP-Q-HALF": {
            "F_TO_P": 1.0,
            "F_TO_S": 1.0,
            "P_TO_S": 1.0,
            "F_TO_Q": 0.5,
            "P_TO_Q": 0.5,
            "S_TO_Q": 0.5,
            "F_TO_D": 1.0,
            "P_TO_D": 1.0,
            "S_TO_D": 1.0,
            "Q_TO_D": 1.0,
        },
        "G3-GROUP-U-D-HALF": {
            "F_TO_P": 0.5,
            "F_TO_S": 0.5,
            "P_TO_S": 0.5,
            "F_TO_Q": 1.0,
            "P_TO_Q": 1.0,
            "S_TO_Q": 1.0,
            "F_TO_D": 0.5,
            "P_TO_D": 0.5,
            "S_TO_D": 0.5,
            "Q_TO_D": 0.5,
        },
        "G3-GROUP-U-HALF": {
            "F_TO_P": 0.5,
            "F_TO_S": 0.5,
            "P_TO_S": 0.5,
            "F_TO_Q": 1.0,
            "P_TO_Q": 1.0,
            "S_TO_Q": 1.0,
            "F_TO_D": 1.0,
            "P_TO_D": 1.0,
            "S_TO_D": 1.0,
            "Q_TO_D": 1.0,
        },
        "G3-GROUP-U-Q-HALF": {
            "F_TO_P": 0.5,
            "F_TO_S": 0.5,
            "P_TO_S": 0.5,
            "F_TO_Q": 0.5,
            "P_TO_Q": 0.5,
            "S_TO_Q": 0.5,
            "F_TO_D": 1.0,
            "P_TO_D": 1.0,
            "S_TO_D": 1.0,
            "Q_TO_D": 1.0,
        },
    }

    assert set(A.EDGE_GRADIENT_LAMBDAS_BY_ARM) == set(A.ARMS)

    for arm in A.ARMS:
        assert A.expected_edge_gradient_lambdas(arm) == expected[arm]


def test_historical_builder_and_forward_require_frozen_arm_binding():
    import inspect

    builder = inspect.signature(
        A.build_historical_model_from_backbone
    )
    forward = inspect.signature(A.historical_forward)

    assert "arm" in builder.parameters
    assert builder.parameters["arm"].default is inspect.Parameter.empty

    assert "arm" in forward.parameters
    assert forward.parameters["arm"].default is inspect.Parameter.empty

    assert "edge_gradient_lambdas" not in forward.parameters


def test_checkpoint_sha_mismatch_rejected_before_loader(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"definitely-not-the-frozen-checkpoint")

    called = False

    def loader(_path):
        nonlocal called
        called = True
        raise AssertionError(
            "loader must not be called before SHA authentication"
        )

    with pytest.raises(
        ValueError,
        match="before deserialization",
    ):
        A.authenticated_checkpoint_load(
            path,
            seed=180,
            arm="G3-GROUP-D-HALF",
            loader=loader,
        )

    assert called is False


def _row(
    row_id="r0",
    pair_id="p0",
    cell_id="C0_SHAM",
):
    return {
        "row_id": row_id,
        "source_pair_id": pair_id,
        "contrast_cell_id": cell_id,
        "claim": "claim text",
        "evidence": "evidence text",
    }


def test_label_free_gen4_row_accepted():
    row = _row()

    validated = A.validate_gen4_rows([row])

    assert validated[0]["row_id"] == "r0"

    for forbidden_gold in (
        "final_label",
        "frame_compatible_label",
        "predicate_covered_label",
        "sufficiency_label",
        "polarity_label",
        "primary_failure_type",
    ):
        assert forbidden_gold not in row


@pytest.mark.parametrize(
    "missing",
    (
        "row_id",
        "source_pair_id",
        "contrast_cell_id",
    ),
)
def test_required_identity_field_missing_rejected(missing):
    row = _row()
    del row[missing]

    with pytest.raises(ValueError):
        A.validate_gen4_rows([row])


def test_duplicate_row_id_rejected():
    rows = [
        _row("same", "p0", "C0_SHAM"),
        _row("same", "p1", "C1_TITLE"),
    ]

    with pytest.raises(
        ValueError,
        match="duplicate row_id",
    ):
        A.validate_gen4_rows(rows)


def test_63_1_64_feature_construction():
    claim = list(range(10, 80))
    evidence = list(range(100, 170))

    features = A.construct_feature_tensors(
        claim,
        evidence,
    )

    input_ids = features["input_ids"]
    attention = features["attention_mask"]
    claim_mask = features["claim_mask"]
    evidence_mask = features["evidence_mask"]

    assert input_ids.shape == (128,)
    assert input_ids.dtype == torch.long

    assert input_ids[:63].tolist() == claim[:63]
    assert input_ids[63].item() == A.EOS_TOKEN_ID
    assert input_ids[64:].tolist() == evidence[:64]

    assert attention.dtype == torch.bool
    assert attention.all()

    assert claim_mask[:63].all()
    assert claim_mask[63].item() is False
    assert not claim_mask[64:].any()

    assert not evidence_mask[:64].any()
    assert evidence_mask[64:].all()


def test_padding_and_separator_masks():
    features = A.construct_feature_tensors(
        [11, 12],
        [21, 22, 23],
    )

    assert features["input_ids"][:6].tolist() == [
        11,
        12,
        A.EOS_TOKEN_ID,
        21,
        22,
        23,
    ]

    assert features["attention_mask"][:6].all()
    assert not features["attention_mask"][6:].any()

    assert features["claim_mask"].tolist()[:6] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]

    assert features["evidence_mask"].tolist()[:6] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    assert (
        features["input_ids"][6:]
        == A.EFFECTIVE_PAD_TOKEN_ID
    ).all()


def test_fixed_external_class_order():
    assert A.EXTERNAL_CLASS_ORDER == (
        "REFUTE",
        "NOT_ENTITLED",
        "SUPPORT",
    )


def test_synthetic_historical_output_serialization():
    metadata = [
        _row("r0", "p0", "C0_SHAM"),
        _row("r1", "p0", "C1_TITLE"),
    ]

    output = {
        "q_authorized": torch.tensor(
            [0.25, 0.75],
            dtype=torch.float32,
        ),
        "entitlement_prob": torch.tensor(
            [0.20, 0.70],
            dtype=torch.float32,
        ),
        "logits": torch.tensor(
            [
                [2.0, 1.0, 0.5],
                [-1.0, 0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        "predictions": torch.tensor(
            [0, 2],
            dtype=torch.long,
        ),
    }

    sha = A.expected_checkpoint_sha256(
        180,
        "G3-GROUP-D-HALF",
    )

    rows = A.serialize_model_outputs(
        output,
        metadata,
        seed=180,
        arm="G3-GROUP-D-HALF",
        checkpoint_sha256=sha,
    )

    assert len(rows) == 2

    assert rows[0]["prediction"] == "REFUTE"
    assert rows[1]["prediction"] == "SUPPORT"

    assert rows[0]["q_authorized"] == pytest.approx(0.25)
    assert rows[1]["entitlement_prob"] == pytest.approx(0.70)

    assert rows[0]["refute_logit"] == pytest.approx(2.0)
    assert rows[0]["ne_logit"] == pytest.approx(1.0)
    assert rows[0]["support_logit"] == pytest.approx(0.5)

    assert (
        rows[0]["support_vs_best_nonsupport_logit_margin"]
        == pytest.approx(-1.5)
    )

    assert A.output_key(rows[0]) == (
        180,
        "G3-GROUP-D-HALF",
        "r0",
    )


def test_serializer_rejects_prediction_disagreement():
    metadata = [_row()]

    output = {
        "q_authorized": torch.tensor([0.5]),
        "entitlement_prob": torch.tensor([0.5]),
        "logits": torch.tensor([[0.0, 0.0, 1.0]]),
        "predictions": torch.tensor([0]),
    }

    sha = A.expected_checkpoint_sha256(
        180,
        "G3-GROUP-D-HALF",
    )

    with pytest.raises(
        ValueError,
        match="disagrees",
    ):
        A.serialize_model_outputs(
            output,
            metadata,
            seed=180,
            arm="G3-GROUP-D-HALF",
            checkpoint_sha256=sha,
        )


def test_complete_matrix_small_fixture():
    expected_rows = [
        _row("r0", "p0", "C0_SHAM"),
        _row("r1", "p0", "C1_TITLE"),
    ]

    evaluators = (
        (180, "G3-GROUP-D-HALF"),
        (181, "G3-GROUP-Q-HALF"),
    )

    output = []

    for seed, arm in evaluators:
        sha = A.expected_checkpoint_sha256(seed, arm)

        for row in expected_rows:
            output.append({
                "seed": seed,
                "arm": arm,
                "checkpoint_sha256": sha,
                "row_id": row["row_id"],
                "source_pair_id": row["source_pair_id"],
                "contrast_cell_id": row["contrast_cell_id"],
            })

    A.validate_complete_matrix(
        output,
        expected_rows,
        evaluators,
    )

    with pytest.raises(
        ValueError,
        match="incomplete evaluator matrix",
    ):
        A.validate_complete_matrix(
            output[:-1],
            expected_rows,
            evaluators,
        )

    duplicated = output + [dict(output[0])]

    with pytest.raises(
        ValueError,
        match="duplicate evaluator-row key",
    ):
        A.validate_complete_matrix(
            duplicated,
            expected_rows,
            evaluators,
        )


def test_provenance_builder_is_deterministic_and_complete():
    sha = A.expected_checkpoint_sha256(
        182,
        "G3-GROUP-U-Q-HALF",
    )

    kwargs = dict(
        adapter_source_commit="future-adapter-commit",
        python_version="3.x",
        torch_version="x",
        transformers_version="5.0.0",
        device="cuda",
        dtype="float32",
        batch_size=64,
        seed=182,
        arm="G3-GROUP-U-Q-HALF",
        checkpoint_sha256=sha,
    )

    first = A.build_provenance_record(**kwargs)
    second = A.build_provenance_record(**kwargs)

    assert first == second

    assert first["historical_source_commit"] == (
        A.HISTORICAL_SOURCE_COMMIT
    )
    assert first["historical_snapshot_sha256"] == (
        A.HISTORICAL_MODEL_SHA256
    )
    assert first["historical_heads_tree"] == (
        A.HISTORICAL_HEADS_TREE
    )
    assert first[
        "historical_wrapper_runtime_equivalence"
    ] == "UNRESOLVED"


def test_cli_requires_explicit_operation():
    parser = A.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_adapter_binds_historical_snapshot_module_lazily():
    tree = ast.parse(
        ADAPTER_PATH.read_text(encoding="utf-8")
    )

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module)

    assert (
        "contramamba.modeling_v6b_minimal_gen3_grouped_snapshot"
        in imports
    )

    # Current model module must never be imported by the adapter.
    assert "contramamba.modeling_v6b_minimal" not in imports


def test_r3_tests_do_not_execute_model_or_checkpoint_loader():
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    def callable_name(node):
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            prefix = callable_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr

        return None

    forbidden_calls = {
        "build_historical_model_from_backbone",
        "historical_forward",
        "strict_load_state_dict",
        "torch.load",
        "MambaModel.from_pretrained",
        "MambaConfig.from_pretrained",
    }

    observed_calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in [callable_name(node.func)]
        if name is not None
    }

    violations = sorted(forbidden_calls & observed_calls)

    assert violations == []
