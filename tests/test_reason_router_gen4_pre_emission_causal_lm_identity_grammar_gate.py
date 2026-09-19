from __future__ import annotations

import inspect
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from scripts import (
    reason_router_gen4_pre_emission_causal_lm_identity_grammar_gate
    as subject,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return FakeEncoding(self.mapping[text])


def valid_mapping():
    common = [10, 11, 12, 13, 14, 15]
    return {
        subject.COMMITMENT_SURFACES["REFUTE"]:
            common + [101, 201],
        subject.COMMITMENT_SURFACES["NOT_ENTITLED"]:
            common + [102, 202, 302],
        subject.COMMITMENT_SURFACES["SUPPORT"]:
            common + [103, 203],
    }


def test_frozen_design_and_causal_object_constants():
    assert subject.DESIGN_FREEZE_COMMIT == (
        "8879e80db019384eef37e63e1095e9f79366d60f"
    )
    assert subject.HF_REPO == "state-spaces/mamba-370m-hf"
    assert subject.HF_REVISION == (
        "589179554943157be31701edd8b4558889276674"
    )
    assert subject.SELECTED_PLANE == "P3"
    assert subject.CONTROL_PLANE == "P5"
    assert subject.EARLY_BLOCK == 35
    assert subject.LATE_BLOCK == 47
    assert subject.CLASS_ORDER == (
        "REFUTE",
        "NOT_ENTITLED",
        "SUPPORT",
    )
    assert subject.DECISIVE_CLASSES == ("REFUTE", "SUPPORT")
    assert subject.MIN_PRECOMMIT_GENERATED_TOKENS == 4


def test_commitment_surfaces_are_exact_and_response_independent():
    assert subject.COMMITMENT_SURFACES == {
        "REFUTE":
            "Based on the evidence, the verdict is REFUTE.",
        "NOT_ENTITLED":
            "Based on the evidence, the verdict is NOT_ENTITLED.",
        "SUPPORT":
            "Based on the evidence, the verdict is SUPPORT.",
    }


def test_valid_finite_grammar_has_shared_prefix_and_unique_tstar():
    tokenizer = FakeTokenizer(valid_mapping())
    manifest = subject.build_grammar_manifest(
        tokenizer,
        tokenizer_provenance={"fake": True},
    )

    assert manifest["longest_common_prefix_length"] == 6
    assert manifest["unique_commitment_token_index_zero_based"] == {
        "REFUTE": 6,
        "NOT_ENTITLED": 6,
        "SUPPORT": 6,
    }
    assert manifest["decisive_t_star_zero_based"] == {
        "REFUTE": 6,
        "SUPPORT": 6,
    }
    assert len(manifest["grammar_sha256"]) == 64


def test_grammar_rejects_too_short_common_prefix():
    mapping = valid_mapping()
    mapping[subject.COMMITMENT_SURFACES["SUPPORT"]] = [
        10, 11, 12, 999, 14, 15, 103,
    ]
    tokenizer = FakeTokenizer(mapping)

    with pytest.raises(
        subject.PrecursorGateError,
        match="COMMON_PREFIX_TOO_SHORT",
    ):
        subject.build_grammar_manifest(
            tokenizer,
            tokenizer_provenance={},
        )


def test_grammar_rejects_complete_prefix_relationship():
    common = [10, 11, 12, 13, 14]
    token_ids = {
        "REFUTE": common + [101],
        "NOT_ENTITLED": common + [102],
        "SUPPORT": common + [101, 201],
    }
    with pytest.raises(
        subject.PrecursorGateError,
        match="COMPLETE_PREFIX",
    ):
        subject.validate_token_sequences(token_ids)


def test_allowed_next_tokens_is_online_and_fail_closed():
    token_ids = {
        "REFUTE": [10, 11, 12, 13, 14, 101],
        "NOT_ENTITLED": [10, 11, 12, 13, 14, 102],
        "SUPPORT": [10, 11, 12, 13, 14, 103],
    }

    assert subject.allowed_next_tokens(
        [10, 11, 12, 13, 14],
        token_ids,
    ) == (101, 102, 103)

    assert subject.allowed_next_tokens(
        [10, 11, 12, 13, 14, 101],
        token_ids,
    ) == ()

    with pytest.raises(
        subject.PrecursorGateError,
        match="GRAMMAR_PREFIX_NOT_RECOGNIZED",
    ):
        subject.allowed_next_tokens([999], token_ids)


def test_config_contract_requires_causal_lm_architecture():
    config = {
        "architectures": ["MambaForCausalLM"],
        "hidden_size": 1024,
        "num_hidden_layers": 48,
        "state_size": 16,
        "intermediate_size": 2048,
        "vocab_size": 50280,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "transformers_version":
            subject.LEGACY_CONFIG_TRANSFORMERS_VERSION,
    }
    out = subject.validate_config_dict(config)
    assert out["architectures"] == ["MambaForCausalLM"]
    assert out["num_hidden_layers"] == 48

    bad = dict(config)
    bad["architectures"] = ["MambaModel"]
    with pytest.raises(
        subject.PrecursorGateError,
        match="CONFIG_ARCHITECTURES",
    ):
        subject.validate_config_dict(bad)



def test_legacy_v439_tied_lm_head_compat_is_exact_and_fail_closed():
    config = SimpleNamespace()
    source = {
        "transformers_version":
            subject.LEGACY_CONFIG_TRANSFORMERS_VERSION,
    }
    out = subject.apply_legacy_tied_lm_head_compat(
        config,
        snapshot_config=source,
    )
    assert config.tie_word_embeddings is True
    assert out == {
        "mode": subject.LEGACY_TIE_COMPAT_MODE,
        "snapshot_transformers_version":
            subject.LEGACY_CONFIG_TRANSFORMERS_VERSION,
        "snapshot_explicit_tie_word_embeddings": False,
        "restored_tie_word_embeddings": True,
    }

    with pytest.raises(
        subject.PrecursorGateError,
        match="LEGACY_TIE_COMPAT_EXPLICIT_SNAPSHOT_SETTING",
    ):
        subject.apply_legacy_tied_lm_head_compat(
            SimpleNamespace(),
            snapshot_config={
                "transformers_version":
                    subject.LEGACY_CONFIG_TRANSFORMERS_VERSION,
                "tie_word_embeddings": False,
            },
        )

    with pytest.raises(
        subject.PrecursorGateError,
        match="LEGACY_TIE_COMPAT_SOURCE_VERSION",
    ):
        subject.apply_legacy_tied_lm_head_compat(
            SimpleNamespace(),
            snapshot_config={"transformers_version": "5.0.0"},
        )

    with pytest.raises(
        subject.PrecursorGateError,
        match="LEGACY_TIE_COMPAT_RUNTIME_ALREADY_SPECIFIED",
    ):
        subject.apply_legacy_tied_lm_head_compat(
            SimpleNamespace(tie_word_embeddings=False),
            snapshot_config=source,
        )

def test_loading_info_is_fail_closed():
    clean = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    assert subject.validate_loading_info(clean) == clean

    for field in clean:
        bad = dict(clean)
        bad[field] = ["bad"]
        with pytest.raises(
            subject.PrecursorGateError,
            match="CAUSAL_LM_LOADING_INFO",
        ):
            subject.validate_loading_info(bad)


class FakeBackbone:
    def __init__(self, shared_weight):
        self.embeddings = SimpleNamespace(weight=shared_weight)

    def state_dict(self):
        return {
            f"k{i}": torch.tensor([float(i)], dtype=torch.float32)
            for i in range(subject.EXPECTED_BACKBONE_STATE_KEY_COUNT)
        }


class FakeModel:
    def __init__(self, *, tied=True):
        shared_weight = torch.empty(
            (
                subject.EXPECTED_VOCAB_SIZE,
                subject.EXPECTED_HIDDEN_SIZE,
            ),
            dtype=torch.float32,
            device="meta",
        )
        self.backbone = FakeBackbone(shared_weight)
        lm_weight = (
            shared_weight
            if tied
            else torch.empty(
                (
                    subject.EXPECTED_VOCAB_SIZE,
                    subject.EXPECTED_HIDDEN_SIZE,
                ),
                dtype=torch.float32,
                device="meta",
            )
        )
        self.lm_head = SimpleNamespace(weight=lm_weight)
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.training = False
        self._parameters = [
            torch.nn.Parameter(
                torch.zeros(1),
                requires_grad=False,
            )
        ]

    def parameters(self):
        return iter(self._parameters)

def test_loaded_causal_lm_requires_exact_backbone_hash(monkeypatch):
    monkeypatch.setattr(
        subject.geom,
        "canonical_state_sha256",
        lambda _state: subject.EXPECTED_BACKBONE_CANONICAL_SHA256,
    )
    monkeypatch.setattr(
        subject,
        "tensor_sha256",
        lambda _tensor: "a" * 64,
    )
    model = FakeModel()
    out = subject.validate_loaded_causal_lm(
        model,
        loading_info={
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )
    assert (
        out["backbone_canonical_sha256"]
        == subject.EXPECTED_BACKBONE_CANONICAL_SHA256
    )
    assert out["lm_head_shape"] == [50280, 1024]
    assert out["lm_head_tied_to_input_embeddings"] is True
    assert out["tie_word_embeddings"] is True
    assert out["any_parameter_requires_grad"] is False

    with pytest.raises(
        subject.PrecursorGateError,
        match="LM_HEAD_NOT_TIED_TO_INPUT_EMBEDDINGS",
    ):
        subject.validate_loaded_causal_lm(
            FakeModel(tied=False),
            loading_info={
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )

    monkeypatch.setattr(
        subject.geom,
        "canonical_state_sha256",
        lambda _state: "0" * 64,
    )
    with pytest.raises(
        subject.PrecursorGateError,
        match="BACKBONE_CANONICAL_SHA256",
    ):
        subject.validate_loaded_causal_lm(
            model,
            loading_info={
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
        )


def test_production_gate_contains_no_generation_or_forward_path():
    source = inspect.getsource(subject)
    lowered = source.lower()

    # Loading a causal LM for identity is permitted; executing it is not.
    assert ".generate(" not in lowered
    assert "generate(" not in lowered
    assert "torch.cuda" not in lowered
    assert "inference_mode" not in lowered
    assert "scientific_model_forward_count" in lowered
    assert "generation_response_inspected" in lowered


def test_manifest_semantics_are_non_scientific_and_zero_forward(monkeypatch):
    monkeypatch.setattr(
        subject,
        "validate_snapshot_and_config",
        lambda _snapshot: {"snapshot_files": {}, "config": {}},
    )
    monkeypatch.setattr(
        subject.geom,
        "load_tokenizer",
        lambda _snapshot: (
            FakeTokenizer(valid_mapping()),
            {"fake": True},
        ),
    )

    manifest = subject.build_gate_manifest(
        snapshot=subject.Path("."),
        check_causal_lm=False,
    )
    assert manifest["result"] == subject.RESULT_PASS
    assert manifest["causal_lm_identity"]["status"] == "NOT_RUN"
    assert manifest["scientific_generation_executed"] is False
    assert manifest["scientific_model_forward_count"] == 0
    assert manifest["cuda_executed"] is False
    assert manifest["training_executed"] is False
    assert manifest["evaluation_executed"] is False
    assert manifest["p_value_count_added"] == 0
    assert manifest["generation_response_inspected"] is False


def test_direct_script_entrypoint_can_import_repo_package():
    repo_root = Path(subject.__file__).resolve().parents[1]
    script = (
        repo_root
        / "scripts"
        / "reason_router_gen4_pre_emission_causal_lm_identity_grammar_gate.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--snapshot" in completed.stdout
    assert "--output" in completed.stdout
    assert "--check-causal-lm" in completed.stdout
