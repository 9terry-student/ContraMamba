from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "longterm_k0_rvg_post_p2_active_encoding_token_coordinate_equivalence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "active_encoding_equivalence",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture():
    candidates, mate_map, mapping, contract = (
        m._synthetic_fixture()
    )
    texts = m.reconstruct_branch_texts(
        candidates,
        mate_map,
        0,
    )
    return (
        candidates,
        mate_map,
        mapping,
        contract,
        texts,
    )


def validate(mapping=None, contract=None):
    candidates, _, default_mapping, default_contract, texts = (
        fixture()
    )
    if mapping is None:
        mapping = default_mapping
    if contract is None:
        contract = default_contract
    return m.validate_equivalence_item(
        m.FakeTokenizer(mapping),
        texts,
        contract,
        candidates[0],
        candidates[1],
    )


def test_branch_reconstruction_exact_no_normalization():
    candidates = {
        0: {
            "prefix_text": "P \n",
            "correction_text": "  C",
            "control_text": "\tD",
        },
        1: {
            "prefix_text": "Q",
            "correction_text": "  X",
            "control_text": "\tY",
        },
    }
    out = m.reconstruct_branch_texts(
        candidates,
        {0: 1, 1: 0},
        0,
    )
    assert out["matched_corr"] == "P \n  C"
    assert out["matched_ctrl"] == "P \n\tD"
    assert out["swapped_corr"] == "P \n  X"
    assert out["swapped_ctrl"] == "P \n\tY"


def test_valid_exact_contract_passes():
    out = validate()
    assert out["exact_contract_match"] is True
    assert out["matched_divergence_anchor"] == 3
    assert out["swapped_divergence_anchor"] == 3
    assert len(out["event_relative_windows"]["matched_corr"]) == 9


def test_prefix_count_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["prefix_token_count"] = 3
    with pytest.raises(
        m.ContractError,
        match="PREFIX_TOKEN_COUNT_MISMATCH",
    ):
        validate(mapping, contract)


def test_prefix_sha256_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["prefix_token_sha256"] = "0" * 64
    with pytest.raises(
        m.ContractError,
        match="PREFIX_TOKEN_SHA256_MISMATCH",
    ):
        validate(mapping, contract)


def test_single_token_prefix_drift_fails():
    _, _, mapping, contract, texts = fixture()
    mapping = {
        key: list(values)
        for key, values in mapping.items()
    }
    mapping[texts["prefix"]][1] = 999
    for role in (
        "matched_corr",
        "matched_ctrl",
        "swapped_corr",
        "swapped_ctrl",
    ):
        mapping[texts[role]][1] = 999

    with pytest.raises(
        m.ContractError,
        match="PREFIX_TOKEN_SHA256_MISMATCH",
    ):
        validate(mapping, contract)


def test_branch_count_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["matched_correction_token_count"] += 1
    with pytest.raises(
        m.ContractError,
        match="TOKEN_COUNT_MISMATCH:matched_corr",
    ):
        validate(mapping, contract)


def test_matched_te_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["matched_divergence_anchor"] = 4
    with pytest.raises(
        m.ContractError,
        match="MATCHED_TE_MISMATCH",
    ):
        validate(mapping, contract)


def test_swapped_te_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["swapped_divergence_anchor"] = 4
    with pytest.raises(
        m.ContractError,
        match="SWAPPED_TE_MISMATCH",
    ):
        validate(mapping, contract)


def test_matched_divergence_offset_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["matched_divergence_offset_from_prefix"] = 2
    with pytest.raises(
        m.ContractError,
        match="MATCHED_DIVERGENCE_OFFSET_MISMATCH",
    ):
        validate(mapping, contract)


def test_swapped_divergence_offset_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["swapped_divergence_offset_from_prefix"] = 2
    with pytest.raises(
        m.ContractError,
        match="SWAPPED_DIVERGENCE_OFFSET_MISMATCH",
    ):
        validate(mapping, contract)


def test_w8_contract_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["matched_w8_available"] = False
    with pytest.raises(
        m.ContractError,
        match="MATCHED_W8_AVAILABILITY_MISMATCH",
    ):
        validate(mapping, contract)


def test_missing_window_coordinate_fails_closed():
    _, _, mapping, contract, texts = fixture()

    mapping = {
        key: list(values)
        for key, values in mapping.items()
    }

    for role in (
        "matched_corr",
        "matched_ctrl",
    ):
        mapping[texts[role]] = mapping[texts[role]][:10]

    contract = dict(contract)
    contract["matched_correction_token_count"] = 10
    contract["matched_control_token_count"] = 10

    with pytest.raises(
        m.ContractError,
        match="MATCHED_W8_AVAILABILITY_MISMATCH",
    ):
        validate(mapping, contract)


def test_phase_mate_stable_id_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["phase_mate_stable_id"] = "wrong"
    with pytest.raises(
        m.ContractError,
        match="PHASE_MATE_STABLE_ID_MISMATCH",
    ):
        validate(mapping, contract)


def test_phase_mate_pair_id_mismatch_fails():
    _, _, mapping, contract, _ = fixture()
    contract = dict(contract)
    contract["phase_mate_pair_id"] = "wrong"
    with pytest.raises(
        m.ContractError,
        match="PHASE_MATE_PAIR_ID_MISMATCH",
    ):
        validate(mapping, contract)


def test_token_count_preserving_branch_drift_is_recorded_not_rejected():
    _, _, mapping, contract, texts = fixture()
    original = validate(mapping, contract)

    changed = {
        key: list(values)
        for key, values in mapping.items()
    }
    changed[texts["matched_corr"]][11] += 1000

    observed = validate(changed, contract)

    assert observed["exact_contract_match"] is True
    assert (
        observed["sequences"]["matched_corr"]["token_id_sha256"]
        != original["sequences"]["matched_corr"]["token_id_sha256"]
    )


def test_anchor_preserving_inwindow_id_change_is_evidence_not_classification():
    _, _, mapping, contract, texts = fixture()
    changed = {
        key: list(values)
        for key, values in mapping.items()
    }

    changed[texts["matched_corr"]][5] += 1000
    out = validate(changed, contract)

    assert out["matched_divergence_anchor"] == 3
    assert out["exact_contract_match"] is True
    assert "classification" not in out


class NondeterministicTokenizer(m.FakeTokenizer):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.calls = {}

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_attention_mask=False,
    ):
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=return_attention_mask,
        )
        count = self.calls.get(text, 0)
        self.calls[text] = count + 1
        if count % 2 == 1:
            result["input_ids"] = list(result["input_ids"])
            result["input_ids"][-1] += 1
        return result


def test_active_encoding_repeat_identity_failure():
    candidates, mate_map, mapping, contract, texts = fixture()

    with pytest.raises(
        m.ContractError,
        match="ACTIVE_ENCODING_REPEAT_IDENTITY_FAILURE",
    ):
        m.validate_equivalence_item(
            NondeterministicTokenizer(mapping),
            texts,
            contract,
            candidates[0],
            candidates[mate_map[0]],
        )


def test_canonical_repeat_identity():
    first = validate()
    second = validate()
    assert (
        m.canonical_json_bytes(first)
        == m.canonical_json_bytes(second)
    )


def test_duplicate_authority_marker_rejected():
    text = (
        "`REAL_ACTIVE_ENCODING_EQUIVALENCE_AUDIT_AUTHORIZED = YES`\n"
        "`REAL_ACTIVE_ENCODING_EQUIVALENCE_AUDIT_AUTHORIZED = NO`\n"
    )
    with pytest.raises(
        m.ContractError,
        match="DUPLICATE_AUTHORITY_MARKER",
    ):
        m.parse_authority_markers(text)


def valid_future_markers(commit):
    out = dict(m.REQUIRED_FUTURE_MARKERS)
    out[
        "ACTIVE_ENCODING_EQUIVALENCE_IMPLEMENTATION_COMMIT"
    ] = commit
    return out


def test_wrong_implementation_commit_marker_rejected():
    markers = valid_future_markers("b" * 40)
    with pytest.raises(
        m.ContractError,
        match="IMPLEMENTATION_COMMIT_MISMATCH",
    ):
        m.validate_execution_markers(
            markers,
            "a" * 40,
        )


def test_wrong_active_manifest_marker_rejected():
    markers = valid_future_markers("a" * 40)
    markers[
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256"
    ] = "0" * 64

    with pytest.raises(
        m.ContractError,
        match="ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256",
    ):
        m.validate_execution_markers(
            markers,
            "a" * 40,
        )


def make_snapshot(tmp_path):
    (tmp_path / "tokenizer.json").write_text(
        '{"version":"1"}',
        encoding="utf-8",
    )
    _, digest = m.tokenizer_snapshot_manifest(tmp_path)
    return digest


def test_manifest_mismatch_rejected_before_transformers_import(tmp_path):
    digest = make_snapshot(tmp_path)
    touched = {"importer": False}

    def importer():
        touched["importer"] = True
        raise AssertionError("must not run")

    markers = {
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256":
            "0" * 64
    }

    with pytest.raises(
        m.ContractError,
        match="EXECUTION_AUTHORITY_ACTIVE_TOKENIZER_MANIFEST_MISMATCH",
    ):
        m.load_local_tokenizer(
            tmp_path,
            markers,
            transformers_importer=importer,
            expected_manifest_sha256=digest,
        )

    assert touched["importer"] is False


def test_transformers_version_mismatch_fails(tmp_path):
    digest = make_snapshot(tmp_path)

    class DummyAuto:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("must not load")

    markers = {
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256":
            digest
    }

    with pytest.raises(
        m.ContractError,
        match="TRANSFORMERS_VERSION_MISMATCH",
    ):
        m.load_local_tokenizer(
            tmp_path,
            markers,
            transformers_importer=lambda:
                (DummyAuto, "wrong-version"),
            expected_manifest_sha256=digest,
        )


def test_nonfast_tokenizer_rejected(tmp_path):
    digest = make_snapshot(tmp_path)

    class Loaded:
        is_fast = False

    class DummyAuto:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return Loaded()

    markers = {
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256":
            digest
    }

    with pytest.raises(
        m.ContractError,
        match="TOKENIZER_MUST_BE_FAST",
    ):
        m.load_local_tokenizer(
            tmp_path,
            markers,
            transformers_importer=lambda:
                (DummyAuto, m.TRANSFORMERS_VERSION),
            expected_manifest_sha256=digest,
        )


def test_loader_forces_local_files_only_and_fast(tmp_path):
    digest = make_snapshot(tmp_path)
    captured = {}

    class Loaded:
        is_fast = True

    class DummyAuto:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = dict(kwargs)
            return Loaded()

    markers = {
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256":
            digest
    }

    tokenizer, provenance = m.load_local_tokenizer(
        tmp_path,
        markers,
        transformers_importer=lambda:
            (DummyAuto, m.TRANSFORMERS_VERSION),
        expected_manifest_sha256=digest,
    )

    assert tokenizer.is_fast is True
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["use_fast"] is True
    assert provenance["local_files_only"] is True
    assert provenance["snapshot_manifest_sha256"] == digest


def test_authority_failure_precedes_real_input_and_tokenizer_reads(tmp_path):
    touched = {
        "input": False,
        "tokenizer": False,
    }

    def deny(_root, _authority):
        raise m.ContractError("DENIED")

    def input_loader(_root):
        touched["input"] = True
        raise AssertionError("must not run")

    def tokenizer_loader(_snapshot, _markers):
        touched["tokenizer"] = True
        raise AssertionError("must not run")

    with pytest.raises(
        m.ContractError,
        match="DENIED",
    ):
        m.execute_real(
            tmp_path,
            "reports/future.md",
            tmp_path / "snapshot",
            tmp_path / "out",
            authority_authenticator=deny,
            real_input_loader=input_loader,
            tokenizer_loader=tokenizer_loader,
        )

    assert touched == {
        "input": False,
        "tokenizer": False,
    }


def test_synthetic_self_check_is_provenance_only():
    out = m.synthetic_self_check()
    assert (
        out["status"]
        == "PASS_SYNTHETIC_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE_CORE"
    )
    assert out["real_p0_scientific_input_read"] is False
    assert out["real_p2_artifact_read"] is False
    assert out["real_hf_tokenizer_loaded"] is False
    assert out["network_access"] is False
    assert out["model_constructed"] is False
    assert out["scientific_model_forward_executed"] is False
    assert (
        out["scientific_token_relation_classification_performed"]
        is False
    )