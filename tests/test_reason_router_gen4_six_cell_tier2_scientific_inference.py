from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

HARNESS_PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_six_cell_tier2_scientific_inference.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "gen4_r5_scientific_harness_under_test",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = _load_harness()


class FakeMamba(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def forward(self, *, input_ids):
        self.calls.append(int(input_ids.shape[0]))
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = FakeMamba()
        self.downstream_batch_sizes: list[int] = []
        self.observed_edge_maps: list[dict[str, float]] = []

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        claim_mask,
        evidence_mask,
        encoder_hidden_states,
        decision_mode,
        gradient_ownership_mode,
        edge_gradient_lambdas,
        return_q_diagnostics,
    ):
        assert encoder_hidden_states is not None
        assert tuple(encoder_hidden_states.shape[:2]) == tuple(input_ids.shape)
        assert decision_mode == H.adapter.DECISION_MODE
        assert gradient_ownership_mode == H.adapter.GRADIENT_OWNERSHIP_MODE
        assert return_q_diagnostics is True

        self.downstream_batch_sizes.append(int(input_ids.shape[0]))
        self.observed_edge_maps.append(dict(edge_gradient_lambdas))

        batch = input_ids.shape[0]
        base = encoder_hidden_states[:, 0, 0]

        logits = torch.stack(
            [
                base * 0.0 + 1.0,
                base * 0.0 + 2.0,
                base * 0.0 + 3.0,
            ],
            dim=-1,
        )

        return {
            "q_authorized": torch.full(
                (batch,),
                0.75,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "entitlement_prob": torch.full(
                (batch,),
                0.60,
                dtype=torch.float32,
                device=input_ids.device,
            ),
            "logits": logits,
            "predictions": torch.full(
                (batch,),
                2,
                dtype=torch.long,
                device=input_ids.device,
            ),
        }


def _encoded(rows: int) -> dict:
    input_ids = torch.ones(
        (rows, H.adapter.MAX_LENGTH),
        dtype=torch.long,
    )
    attention = torch.ones(
        (rows, H.adapter.MAX_LENGTH),
        dtype=torch.bool,
    )
    claim_mask = torch.zeros_like(attention)
    evidence_mask = torch.zeros_like(attention)

    claim_mask[:, :32] = True
    evidence_mask[:, 33:65] = True

    return {
        "row_id": [f"r{i}" for i in range(rows)],
        "source_pair_id": [f"p{i // 6}" for i in range(rows)],
        "contrast_cell_id": [
            H.adapter.CANONICAL_CELLS[i % 6]
            for i in range(rows)
        ],
        "input_ids": input_ids,
        "attention_mask": attention,
        "claim_mask": claim_mask,
        "evidence_mask": evidence_mask,
    }


def _metadata(rows: int) -> list[dict]:
    return [
        {
            "row_id": f"r{i}",
            "source_pair_id": f"p{i // 6}",
            "contrast_cell_id": H.adapter.CANONICAL_CELLS[i % 6],
            "claim": f"claim {i}",
            "evidence": f"evidence {i}",
        }
        for i in range(rows)
    ]


def test_authority_and_correction_binding_exact():
    assert H.R5_AUTHORITY_COMMIT == (
        "6b0ab6e62fc670191f3921d86fae8daf27d43caf"
    )
    assert H.R5_CORRECTION_COMMIT == (
        "8f335e88ea905b7e1acd7b164e01ec6750144bf4"
    )


def test_encoder_cache_batch_size_exact():
    assert H.ENCODER_CACHE_BATCH_SIZE == 8


def test_1800_rows_imply_exactly_225_encoder_chunks():
    slices = H.encoder_cache_slices(1800)

    assert len(slices) == 225
    assert slices[0] == (0, 8)
    assert slices[-1] == (1792, 1800)
    assert all(end - start == 8 for start, end in slices)


def test_downstream_batch_size_and_partition_exact():
    assert H.DOWNSTREAM_BATCH_SIZE == 720

    slices = H.downstream_slices(1800)

    assert tuple(
        end - start
        for start, end in slices
    ) == (720, 720, 360)


def test_exact_evaluator_order_and_counts():
    evaluators = H.expected_evaluators()

    assert len(evaluators) == 18
    assert evaluators == tuple(
        (seed, arm)
        for seed in (180, 181, 182)
        for arm in H.adapter.ARMS
    )

    assert H.EXPECTED_SCIENTIFIC_MAMBA_CACHE_FORWARD_CALLS == 4050
    assert H.EXPECTED_SCIENTIFIC_DOWNSTREAM_FORWARD_CALLS == 54
    assert H.EXPECTED_SCIENTIFIC_ROWS == 32400


def test_encoder_cache_mock_uses_fixed_batch_eight():
    model = FakeModel()
    encoded = _encoded(18)

    hidden, count = H.cache_encoder_hidden_states(
        model,
        encoded,
        device=torch.device("cpu"),
    )

    assert model.mamba.calls == [8, 8, 2]
    assert count == 3
    assert hidden.shape == (18, H.adapter.MAX_LENGTH, 1)
    assert hidden.dtype == torch.float32


def test_cached_downstream_passes_hidden_and_does_not_call_mamba():
    model = FakeModel()
    encoded = _encoded(4)

    hidden, _ = H.cache_encoder_hidden_states(
        model,
        encoded,
        device=torch.device("cpu"),
    )

    before = len(model.mamba.calls)

    batch = {
        key: value
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }

    output = H.cached_downstream_forward(
        model,
        batch,
        hidden,
        arm="G3-GROUP-D-HALF",
    )

    assert len(model.mamba.calls) == before
    assert output["logits"].shape == (4, 3)

    assert model.observed_edge_maps == [
        H.adapter.expected_edge_gradient_lambdas(
            "G3-GROUP-D-HALF"
        )
    ]


def test_downstream_1800_mock_partition_is_720_720_360():
    model = FakeModel()
    encoded = _encoded(1800)
    metadata = _metadata(1800)

    hidden = (
        encoded["input_ids"]
        .to(torch.float32)
        .unsqueeze(-1)
    )

    rows, count = H.run_cached_downstream_batches(
        model,
        encoded,
        metadata,
        hidden,
        device=torch.device("cpu"),
        seed=180,
        arm="G3-GROUP-D-HALF",
        checkpoint_sha256=(
            H.adapter.expected_checkpoint_sha256(
                180,
                "G3-GROUP-D-HALF",
            )
        ),
    )

    assert count == 3
    assert model.downstream_batch_sizes == [720, 720, 360]
    assert model.mamba.calls == []
    assert len(rows) == 1800


def test_scientific_rows_have_required_provenance_fields():
    raw = [{
        "seed": 180,
        "arm": "G3-GROUP-D-HALF",
        "checkpoint_sha256": H.adapter.expected_checkpoint_sha256(
            180,
            "G3-GROUP-D-HALF",
        ),
        "row_id": "r0",
        "source_pair_id": "p0",
        "contrast_cell_id": "C0_SHAM",
        "q_authorized": 0.5,
        "entitlement_prob": 0.4,
        "refute_logit": 1.0,
        "ne_logit": 2.0,
        "support_logit": 3.0,
        "final_logits": [1.0, 2.0, 3.0],
        "prediction_id": 2,
        "prediction": "SUPPORT",
        "support_vs_best_nonsupport_logit_margin": 1.0,
    }]

    row = H._decorate_rows(raw)[0]

    assert row["schema_version"] == H.SCIENTIFIC_ROW_SCHEMA
    assert row["evaluator_seed"] == 180
    assert row["evaluator_arm"] == "G3-GROUP-D-HALF"
    assert row["r5_execution_authority_commit"] == H.R5_AUTHORITY_COMMIT
    assert row["r5_batch_cache_correction_commit"] == H.R5_CORRECTION_COMMIT
    assert row["tokenizer_identity"] == H.tokenizer_identity()


@pytest.mark.parametrize(
    "field",
    (
        "q_authorized",
        "entitlement_prob",
        "refute_logit",
        "ne_logit",
        "support_logit",
        "support_vs_best_nonsupport_logit_margin",
    ),
)
def test_nonfinite_scientific_output_rejected(field):
    raw = {
        "seed": 180,
        "arm": "G3-GROUP-D-HALF",
        "checkpoint_sha256": H.adapter.expected_checkpoint_sha256(
            180,
            "G3-GROUP-D-HALF",
        ),
        "row_id": "r0",
        "source_pair_id": "p0",
        "contrast_cell_id": "C0_SHAM",
        "q_authorized": 0.5,
        "entitlement_prob": 0.4,
        "refute_logit": 1.0,
        "ne_logit": 2.0,
        "support_logit": 3.0,
        "final_logits": [1.0, 2.0, 3.0],
        "prediction_id": 2,
        "prediction": "SUPPORT",
        "support_vs_best_nonsupport_logit_margin": 1.0,
    }

    raw[field] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        H._decorate_rows([raw])


def test_jsonl_serialization_is_compact_sorted_utf8_lf():
    rows = [
        {
            "z": 1.25,
            "a": "한글",
        },
        {
            "z": -2.0,
            "a": "x",
        },
    ]

    raw = H.scientific_jsonl_bytes(rows)

    assert raw == (
        b'{"a":"\xed\x95\x9c\xea\xb8\x80","z":1.25}\n'
        b'{"a":"x","z":-2.0}\n'
    )

    assert not raw.startswith(b"\xef\xbb\xbf")


def test_parser_requires_explicit_operation():
    parser = H.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def _called_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    def name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = name(node.value)
            return (
                f"{prefix}.{node.attr}"
                if prefix
                else node.attr
            )
        return None

    return {
        value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for value in [name(node.func)]
        if value is not None
    }


def test_harness_does_not_use_adapter_historical_forward():
    calls = _called_names(HARNESS_PATH)

    assert "adapter.historical_forward" not in calls


def test_harness_uses_frozen_adapter_edge_helper():
    calls = _called_names(HARNESS_PATH)

    assert "adapter.expected_edge_gradient_lambdas" in calls


def test_no_from_pretrained_or_mutable_model_resolution():
    source = HARNESS_PATH.read_text(encoding="utf-8")

    assert ".from_pretrained(" not in source
    assert "MambaModel(config)" in source
    assert "MambaConfig.from_dict(config_data)" in source


def test_no_adaptive_or_oom_batch_fallback_contract():
    source = HARNESS_PATH.read_text(encoding="utf-8").lower()

    forbidden = (
        "batch_size //",
        "batch_size / 2",
        "out of memory",
        "cuda out of memory",
        "reduce batch",
        "adaptive_batch",
    )

    for token in forbidden:
        assert token not in source


def test_cache_is_inside_per_evaluator_execution():
    tree = ast.parse(HARNESS_PATH.read_text(encoding="utf-8"))

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    run_one = functions["run_one_evaluator"]
    run_matrix = functions["run_scientific_matrix"]

    run_one_source = ast.unparse(run_one)
    run_matrix_source = ast.unparse(run_matrix)

    assert "cache_encoder_hidden_states" in run_one_source
    assert "run_one_evaluator" in run_matrix_source
    assert "for seed, arm in expected_evaluators()" in run_matrix_source


def test_synthetic_preflight_is_non_gen4_and_two_stage():
    source = HARNESS_PATH.read_text(encoding="utf-8")

    assert "SYNTHETIC_NON_GEN4" in source
    assert "for _repeat in range(2)" in source
    assert "cached downstream unexpectedly re-executed Mamba" in source


def test_static_tests_do_not_run_real_execution_paths():
    source = Path(__file__).read_text(encoding="utf-8")
    calls = _called_names(Path(__file__))

    forbidden_calls = {
        "H.load_evaluator_model",
        "H.synthetic_forward_preflight",
        "H.run_scientific_matrix",
        "torch.load",
    }

    assert sorted(forbidden_calls & calls) == []
