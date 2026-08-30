import ast
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import observe_longterm_o0a_native_mamba_state_dynamics as o0a


class ByteTokenizer:
    """Deterministic tiny tokenizer; it never accesses a model repository."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return bytes(token_ids).decode("utf-8")


def _metrics(**updates):
    values = {
        "terminal_hidden_norm": 1.0,
        "last_step_delta": 2.0,
        "terminal_consecutive_state_cosine": 0.5,
        "terminal_acceleration": 3.0,
        "evidence_region_mean_consecutive_state_delta": 4.0,
        "evidence_region_max_consecutive_state_delta": 5.0,
    }
    values.update(updates)
    return values


def _zero_state(text="Claim: c\nEvidence:", tokens=(1, 2), vectors=None):
    if vectors is None:
        vectors = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    return {
        "actual_evidence_prefix_token_count": 0,
        "serialized_input_text": text,
        "input_ids": list(tokens),
        "terminal_vectors": vectors,
    }


def test_deterministic_token_prefix_schedule_has_exact_quarters():
    expected = [
        (0.00, [0.00], 0, 0.00),
        (0.25, [0.25], 1, 0.25),
        (0.50, [0.50], 2, 0.50),
        (0.75, [0.75], 3, 0.75),
        (1.00, [1.00], 4, 1.00),
    ]
    first = o0a.build_prefix_token_schedule(4)
    second = o0a.build_prefix_token_schedule(4)
    observed = [
        (
            item["requested_prefix_fraction"],
            item["requested_prefix_fractions"],
            item["actual_evidence_prefix_token_count"],
            item["actual_fraction"],
        )
        for item in first
    ]
    assert observed == expected
    assert first == second


def test_constructs_0_25_50_75_100_prefixes_in_token_space():
    tokenizer = ByteTokenizer()
    result = o0a.construct_token_prefixes(tokenizer, "c", "abcd")
    marker_ids = list(b"Claim: c\nEvidence:")
    full_ids = list(b"Claim: c\nEvidence: abcd")
    suffix = full_ids[len(marker_ids) :]

    assert result["marker_token_ids"] == marker_ids
    assert result["full_token_ids"] == full_ids
    assert result["evidence_suffix_token_ids"] == suffix
    assert [item["actual_evidence_prefix_token_count"] for item in result["prefixes"]] == [0, 2, 3, 4, 5]
    assert result["prefixes"][0]["serialized_input_text"] == "Claim: c\nEvidence:"
    assert result["prefixes"][0]["input_ids"] == marker_ids
    assert result["prefixes"][-1]["input_ids"] == full_ids
    for prefix in result["prefixes"]:
        count = prefix["actual_evidence_prefix_token_count"]
        assert prefix["input_ids"] == marker_ids + suffix[:count]


def test_duplicate_prefix_token_counts_are_deduplicated_with_all_requests_retained():
    schedule = o0a.build_prefix_token_schedule(1)
    assert len(schedule) == 2
    assert schedule[0]["actual_evidence_prefix_token_count"] == 0
    assert schedule[0]["requested_prefix_fractions"] == [0.0]
    assert schedule[1]["actual_evidence_prefix_token_count"] == 1
    assert schedule[1]["requested_prefix_fractions"] == [0.25, 0.5, 0.75, 1.0]


def test_serialization_prefix_relationship_fails_closed():
    class BrokenTokenizer(ByteTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("Evidence:"):
                ids[-1] += 1
            return ids

    with pytest.raises(o0a.ContractError, match="serialization/token-prefix invariant"):
        o0a.construct_token_prefixes(BrokenTokenizer(), "claim", "evidence")


def test_reference_selection_is_by_pair_id_and_requires_exactly_one_none():
    rows = [
        {"id": "a_none", "pair_id": "a", "intervention_type": "none"},
        {"id": "a_swap", "pair_id": "a", "intervention_type": "entity_swap"},
        {"id": "b_none", "pair_id": "b", "intervention_type": "none"},
    ]
    references = o0a.select_none_references(rows)
    assert references["a"]["id"] == "a_none"
    assert references["b"]["id"] == "b_none"

    with pytest.raises(o0a.ContractError, match="exactly one none"):
        o0a.select_none_references(rows + [{"id": "c", "pair_id": "c", "intervention_type": "paraphrase"}])
    with pytest.raises(o0a.ContractError, match="exactly one none"):
        o0a.select_none_references(rows + [{"id": "a_none_2", "pair_id": "a", "intervention_type": "none"}])


def test_paired_distance_formulas():
    current_vector = np.asarray([0.0, 1.0])
    reference_vector = np.asarray([1.0, 0.0])
    current_metrics = _metrics(last_step_delta=5.0, terminal_hidden_norm=2.0)
    reference_metrics = _metrics(last_step_delta=2.0, terminal_hidden_norm=1.0)
    distances = o0a.compute_paired_distances(
        current_vector,
        reference_vector,
        current_metrics,
        reference_metrics,
    )
    assert distances["terminal_normalized_l2_distance"] == pytest.approx(math.sqrt(2.0))
    assert distances["terminal_cosine_distance"] == pytest.approx(1.0)
    assert distances["transition_magnitude_difference"] == pytest.approx(3.0)
    assert distances["trajectory_summary_difference"] == pytest.approx(math.sqrt(10.0))


def test_cosine_and_normalized_l2_zero_vector_handling():
    zero = np.zeros(2)
    unit = np.asarray([1.0, 0.0])
    orthogonal = np.asarray([0.0, 1.0])
    assert o0a.safe_cosine_similarity(zero, zero) == 1.0
    assert o0a.safe_cosine_similarity(zero, unit) == 0.0
    assert o0a.safe_cosine_similarity(unit, orthogonal) == 0.0
    assert o0a.normalized_l2_distance(zero, zero) == 0.0
    assert o0a.normalized_l2_distance(zero, unit) == 1.0


def test_normalized_l2_and_cosine_are_algebraically_redundant_for_unit_vectors():
    left = np.asarray([3.0, 4.0], dtype=np.float64)
    right = np.asarray([-4.0, 3.0], dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)

    d_l2 = o0a.normalized_l2_distance(left, right)
    d_cos = 1.0 - o0a.safe_cosine_similarity(left, right)

    assert d_l2**2 == pytest.approx(2.0 * d_cos, abs=1e-12)
    assert "algebraically redundant" in o0a.NORMALIZED_L2_COSINE_REDUNDANCY


def test_o0a_runtime_is_bound_to_cpu_float32():
    torch_module, dtype, device = o0a._resolve_runtime("cpu", "float32")
    assert torch_module is torch
    assert dtype == torch.float32
    assert device == "cpu"

    with pytest.raises(o0a.ContractError, match="device=cpu"):
        o0a._resolve_runtime("cuda", "float32")
    with pytest.raises(o0a.ContractError, match="dtype=float32"):
        o0a._resolve_runtime("cpu", "bfloat16")


def test_huggingface_loading_calls_bind_immutable_revision_in_source():
    source = Path(o0a.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    from_pretrained_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert len(from_pretrained_calls) == 2
    for call in from_pretrained_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "revision" in keywords
        assert isinstance(keywords["revision"], ast.Name)
        assert keywords["revision"].id in {"MODEL_REVISION", "TOKENIZER_REVISION"}

    assert o0a.MODEL_ID == "state-spaces/mamba-130m-hf"
    assert o0a.MODEL_REVISION == "5708daa364c50b880e7bd92eab456e0d34492ee9"
    assert o0a.TOKENIZER_ID == o0a.MODEL_ID
    assert o0a.TOKENIZER_REVISION == o0a.MODEL_REVISION


def test_claim_boundary_language_is_present_for_proxy_and_paraphrase_controls():
    assert "native pretrained Mamba hidden-state proxies" in o0a.HIDDEN_STATE_PROXY_BOUNDARY
    assert "not the selective SSM recurrent state" in o0a.HIDDEN_STATE_PROXY_BOUNDARY
    assert "cache_params" in o0a.HIDDEN_STATE_PROXY_BOUNDARY
    assert "whole-pair semantic-invariance" in o0a.PARAPHRASE_CONTROL_BOUNDARY
    assert "not a pure evidence-only paraphrase control" in o0a.PARAPHRASE_CONTROL_BOUNDARY
    assert "0% evidence prefix" in o0a.PARAPHRASE_CONTROL_BOUNDARY
    assert "no independent inferential weight" in o0a.TRAJECTORY_SUMMARY_BOUNDARY


def test_zero_prefix_identical_input_and_state_invariant():
    reference = _zero_state()
    candidate = _zero_state()
    o0a.assert_zero_prefix_identical(
        reference,
        candidate,
        claims_identical=True,
        intervention_type="entity_swap",
        tolerance=1e-6,
    )

    with pytest.raises(o0a.ContractError, match="input tokens differ"):
        o0a.assert_zero_prefix_identical(
            reference,
            _zero_state(tokens=(1, 3)),
            claims_identical=True,
            intervention_type="predicate_swap",
            tolerance=1e-6,
        )
    with pytest.raises(o0a.ContractError, match="hidden states differ"):
        o0a.assert_zero_prefix_identical(
            reference,
            _zero_state(vectors=np.zeros((2, 2), dtype=np.float32)),
            claims_identical=True,
            intervention_type="polarity_flip",
            tolerance=1e-6,
        )

    # Paraphrase and differing claims are expressly outside this equality gate.
    o0a.assert_zero_prefix_identical(
        reference,
        _zero_state(text="different", tokens=(9,), vectors=np.zeros((2, 2))),
        claims_identical=True,
        intervention_type="paraphrase",
        tolerance=0.0,
    )


def test_collision_and_existing_output_directory_fail_closed():
    with pytest.raises(o0a.ContractError, match="collision"):
        o0a.assert_unique_keys([("row", 0), ("row", 0)], "test key")

    class FakePath:
        def __init__(self, exists):
            self._exists = exists

        def exists(self):
            return self._exists

        def __str__(self):
            return "synthetic-output"

    with pytest.raises(FileExistsError, match="already exists"):
        o0a.ensure_output_directory_available(FakePath(True))
    o0a.ensure_output_directory_available(FakePath(False))


def test_output_schema_and_artifact_set_are_stable():
    assert o0a.OBSERVATION_KEYS == (
        "schema_version",
        "row_id",
        "pair_id",
        "intervention_type",
        "primary_failure_type",
        "final_label",
        "requested_prefix_fraction",
        "requested_prefix_fractions",
        "actual_evidence_prefix_token_count",
        "full_evidence_token_count",
        "actual_fraction",
        "serialization_template",
        "serialized_input_text",
        "serialized_input_utf8_sha256",
        "input_token_ids",
        "vector_index",
        "layers",
    )
    assert o0a.PAIRED_DISTANCE_KEYS[-4:] == (
        "terminal_normalized_l2_distance",
        "terminal_cosine_distance",
        "transition_magnitude_difference",
        "trajectory_summary_difference",
    )

    output_dir = Path(__file__).parent / "o0a_schema_test_output"
    assert not output_dir.exists()
    try:
        manifest = {
            "repository_head": "0" * 40,
            "dataset_sha256": "1" * 64,
            "model_id": o0a.MODEL_ID,
            "model_revision": o0a.MODEL_REVISION,
            "model_name": o0a.MODEL_NAME,
            "device": o0a.AUTHORIZED_DEVICE,
            "dtype": o0a.AUTHORIZED_DTYPE_NAME,
        }
        summary = {"paired_distance_groups": []}
        o0a.write_artifacts(
            output_dir,
            manifest=manifest,
            observations=[],
            vectors=np.empty((0, 0, 0), dtype=np.float32),
            paired_distances=[],
            summary=summary,
        )
        assert {path.name for path in output_dir.iterdir()} == set(o0a.REQUIRED_ARTIFACTS)
        assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == manifest
        with np.load(output_dir / "terminal_hidden_states.npz", allow_pickle=False) as artifact:
            assert artifact["schema_version"].item() == o0a.SCHEMA_VERSION
            assert artifact["terminal_hidden_states"].shape == (0, 0, 0)
        checksum_names = {
            line.split("  ", 1)[1]
            for line in (output_dir / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
        }
        assert checksum_names == set(o0a.REQUIRED_ARTIFACTS) - {"SHA256SUMS.txt"}
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_fake_native_model_exercises_observation_pairing_and_summary_pipeline():
    class FakeNativeModel:
        def __call__(self, *, input_ids, output_hidden_states, return_dict, use_cache):
            assert output_hidden_states is True
            assert return_dict is True
            assert use_cache is False
            token_values = input_ids.to(dtype=torch.float32).unsqueeze(-1)
            positions = torch.arange(input_ids.shape[1], dtype=torch.float32).reshape(1, -1, 1)
            initial = torch.cat((token_values, positions), dim=-1)
            output = initial + 0.5
            return SimpleNamespace(hidden_states=(initial, output), last_hidden_state=output)

    rows = [
        {
            "id": "p_none",
            "pair_id": "p",
            "claim": "c",
            "evidence": "abcd",
            "intervention_type": "none",
            "primary_failure_type": "none",
            "final_label": "SUPPORT",
        },
        {
            "id": "p_paraphrase",
            "pair_id": "p",
            "claim": "rewritten c",
            "evidence": "dcba",
            "intervention_type": "paraphrase",
            "primary_failure_type": "none",
            "final_label": "SUPPORT",
        },
        {
            "id": "p_entity_swap",
            "pair_id": "p",
            "claim": "c",
            "evidence": "zbcd",
            "intervention_type": "entity_swap",
            "primary_failure_type": "frame",
            "final_label": "NOT_ENTITLED",
        },
    ]
    observations, vectors, context, layer_descriptors = o0a.collect_observations(
        rows,
        ByteTokenizer(),
        torch,
        FakeNativeModel(),
        "cpu",
        0.0,
    )
    assert len(observations) == 15
    assert vectors.shape == (15, 2, 2)
    assert [descriptor["layer_role"] for descriptor in layer_descriptors] == [
        "embedding_or_initial_hidden_state",
        "output_hidden_state",
    ]

    paired = o0a.build_paired_distances(rows, context, layer_descriptors)
    assert len(paired) == 20
    entity_zero = [
        record
        for record in paired
        if record["intervention_type"] == "entity_swap"
        and record["requested_prefix_fraction"] == 0.0
    ]
    assert all(record["terminal_normalized_l2_distance"] == 0.0 for record in entity_zero)
    summary = o0a.build_summary(observations, paired)
    assert summary["status"] == "DESCRIPTIVE_SCREENING_ONLY"
    assert summary["hard_pass_threshold"] is None
    assert summary["paired_distance_groups"]


def test_import_and_source_safety_do_not_load_or_run_a_model():
    script_path = Path(o0a.__file__)
    o0a.assert_observer_source_safety(script_path)
    source = script_path.read_text(encoding="utf-8")
    assert "ContraMambaV6BMinimal" not in source
    assert "from transformers import AutoTokenizer, MambaModel" in source
