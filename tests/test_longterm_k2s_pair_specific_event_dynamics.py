import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

P = Path(__file__).parents[1] / "scripts" / "longterm_k2s_pair_specific_event_dynamics.py"
spec = importlib.util.spec_from_file_location("k2s", P)
k = importlib.util.module_from_spec(spec)
sys.modules["k2s"] = k
spec.loader.exec_module(k)


class CharTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        out = {"input_ids": [ord(ch) for ch in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def make_rows(n=300):
    rows = []
    for i in range(n):
        prefix = f"Claim: claim{i}\nEvidence: evidence{i}\nAdditional evidence:\n"
        # Shared first char makes d-p = 2 under CharTokenizer and leaves >8 tokens.
        rows.append(
            {
                "stable_item_id": f"k2w-v1:{i:064x}",
                "base_claim_sha256": f"{i + 1000:064x}",
                "pair_id": f"pair{i}",
                "prefix_text": prefix,
                "correction_text": "aCORRECTION-LONG",
                "control_text": "aCONTROL---LONG",
                "construction_status": "valid",
                "source_dataset_physical_sha256": k.SOURCE_PHYSICAL_SHA256,
                "source_dataset_semantic_sha256": k.SOURCE_SEMANTIC_SHA256,
            }
        )
    return rows


def test_constants_and_no_head_selection_contract():
    assert k.K2S_PREREG_AUTHORITY_COMMIT == "c9cb68c2a48a19c3874d059ca00f5c297d929ada"
    assert k.K2W_CLOSURE_COMMIT == "b94f81b411bcbc74e32ad0ad6564b8978016eec0"
    assert k.W == 8 and k.PRIMARY_LAYER == 23 and k.N_BLOCKS == 150
    text = P.read_text(encoding="utf-8")
    for forbidden in ("predicted_final_label", "SUPPORT vote", "confidence threshold", "margin threshold"):
        assert forbidden not in text


def test_reciprocal_mapping_is_150_disjoint_blocks():
    rows = make_rows()
    blocks = k.reciprocal_blocks(rows)
    assert len(blocks) == 150
    seen = set()
    for block in blocks:
        a = block["item_a_index"]
        b = block["item_b_index"]
        assert b == (a ^ 1) and a == (b ^ 1)
        seen |= {a, b}
    assert seen == set(range(300))


def test_input_contracts_fail_closed_if_tokenizer_changes_frozen_distribution():
    rows = make_rows()
    with pytest.raises(k.ContractError, match="K2S_FROZEN_TOKEN_FEASIBILITY_MISMATCH"):
        k.build_input_contracts(rows, CharTokenizer())


def test_first_divergence_is_strictly_after_prefix():
    p = [1, 2, 3]
    c = p + [10, 11, 12]
    n = p + [10, 99, 12]
    assert k.first_divergence_after_prefix(c, n, len(p)) == 4


def test_analyze_mamba_source_binds_post_update_readout_line():
    source = b'''\nclass MambaMixer:\n    def slow_forward(self, seq_len, discrete_A, deltaB_u, C, ssm_state, torch, dtype):\n        scan_outputs = []\n        for i in range(seq_len):\n            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]\n            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))\n            scan_outputs.append(scan_output)\n        return scan_outputs\n    def forward(self, hidden_states, cache_params=None, attention_mask=None):\n        if False:\n            return hidden_states\n        return self.slow_forward(hidden_states, cache_params, attention_mask)\n'''
    result = k.analyze_mamba_source(source)
    assert result["capture_line"] == result["recurrence_update_line"] + 1


def test_analyze_mamba_source_rejects_readout_before_update():
    source = b'''\nclass MambaMixer:\n    def slow_forward(self, seq_len, discrete_A, deltaB_u, C, ssm_state, torch, dtype):\n        for i in range(seq_len):\n            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))\n            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]\n        return scan_output\n    def forward(self, hidden_states):\n        return self.slow_forward(hidden_states)\n'''
    with pytest.raises(k.ContractError):
        k.analyze_mamba_source(source)


def test_analyze_mamba_source_rejects_wrong_recurrence_operator():
    source = b'''\nclass MambaMixer:\n    def slow_forward(self, seq_len, discrete_A, deltaB_u, C, ssm_state, torch, dtype):\n        for i in range(seq_len):\n            ssm_state = discrete_A[:, :, i, :] * ssm_state - deltaB_u[:, :, i, :]\n            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))\n        return scan_output\n    def forward(self, hidden_states):\n        return self.slow_forward(hidden_states)\n'''
    with pytest.raises(k.ContractError):
        k.analyze_mamba_source(source)


def test_encoder_fingerprint_enforces_secondary_raw_digest(monkeypatch):
    torch = pytest.importorskip("torch")
    state = {
        "mamba.alpha": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "mamba.beta": torch.tensor([3.0], dtype=torch.float32),
    }
    tensor_digests = {}
    raw_chunks = []
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().numpy().tobytes()
        tensor_digests[key] = k.sha256_bytes(raw)
        raw_chunks.append(raw)
    monkeypatch.setattr(k, "COMMON_ENCODER_CANONICAL_SHA256", k.sha256_bytes(k.canonical_json(tensor_digests)))
    monkeypatch.setattr(k, "COMMON_ENCODER_RAW_CONCAT_SHA256", "0" * 64)
    monkeypatch.setattr(k, "COMMON_ENCODER_TENSOR_COUNT", 2)
    monkeypatch.setattr(k, "COMMON_ENCODER_NUMEL", 3)
    monkeypatch.setattr(k, "COMMON_ENCODER_RAW_BYTES", sum(len(x) for x in raw_chunks))
    with pytest.raises(k.ContractError, match="COMMON_ENCODER_RAW_CONCAT_DIGEST_MISMATCH"):
        k.encoder_fingerprint(state)


def test_compute_branch_metrics_linear_path():
    torch = pytest.importorskip("torch")
    p = 2
    # 1D-equivalent tensor embedded in [1,1,1]; constant +1 velocity.
    states = {idx: torch.tensor([[[float(idx)]]], dtype=torch.float32) for idx in range(p - 1, p + 9)}
    result = k.compute_branch_metrics(states, p)
    assert result["R_mean_speed"] == pytest.approx(1.0)
    assert result["D_mean_turn"] == pytest.approx(0.0)
    assert result["P_efficiency"] == pytest.approx(1.0)
    assert result["n_valid_turns"] == 8


def test_recipient_specificity_uses_absolute_contrast_magnitude():
    base = {
        "state_p_sha256": "p",
        "state_p_minus_1_sha256": "pm1",
        "prefix_state_sequence_sha256": "prefix",
        "prefix_state_token_count": 7,
        "D_mean_turn": 0.0,
    }
    branches = {
        "matched_corr": base | {"R_mean_speed": 5.0, "P_efficiency": 0.8},
        "matched_ctrl": base | {"R_mean_speed": 2.0, "P_efficiency": 0.2},
        "swapped_corr": base | {"R_mean_speed": 4.0, "P_efficiency": 0.6},
        "swapped_ctrl": base | {"R_mean_speed": 2.0, "P_efficiency": 0.3},
    }
    result = k.recipient_layer_result(branches)
    assert result["signed_contrasts"]["R_matched"] == pytest.approx(3.0)
    assert result["signed_contrasts"]["R_swapped"] == pytest.approx(2.0)
    assert result["X_pair_specificity"]["R"] == pytest.approx(1.0)
    assert result["X_pair_specificity"]["P"] == pytest.approx(0.3)


def test_recipient_direction_undefined_if_any_branch_turn_undefined():
    base = {
        "state_p_sha256": "p",
        "state_p_minus_1_sha256": "pm1",
        "prefix_state_sequence_sha256": "prefix",
        "prefix_state_token_count": 7,
        "R_mean_speed": 1.0,
        "P_efficiency": 0.5,
    }
    branches = {name: base | {"D_mean_turn": 0.1} for name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl")}
    branches["swapped_ctrl"]["D_mean_turn"] = None
    result = k.recipient_layer_result(branches)
    assert result["X_pair_specificity"]["D"] is None


def test_recipient_requires_full_common_prefix_state_identity():
    base = {
        "state_p_sha256": "p",
        "state_p_minus_1_sha256": "pm1",
        "prefix_state_sequence_sha256": "prefix",
        "prefix_state_token_count": 7,
        "R_mean_speed": 1.0,
        "D_mean_turn": 0.1,
        "P_efficiency": 0.5,
    }
    branches = {name: dict(base) for name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl")}
    branches["swapped_ctrl"]["prefix_state_sequence_sha256"] = "different"
    with pytest.raises(k.ContractError, match="SCIENTIFIC_PREFIX_STATE_IDENTITY_FAILURE"):
        k.recipient_layer_result(branches)


def test_block_unit_averages_reciprocal_item_x_values():
    items = []
    for i in range(300):
        items.append({
            "item_index": i,
            "block_index": i // 2,
            "stable_item_id": str(i),
            "X_R": float(i % 2),
            "X_D": None if i < 2 else 1.0,
            "X_P": 2.0,
        })
    rows = k.block_rows_from_items(items)
    assert len(rows) == 150
    assert rows[0]["B_R"] == pytest.approx(0.5)
    assert rows[0]["B_D"] is None
    assert rows[1]["B_D"] == pytest.approx(1.0)
    assert rows[0]["B_P"] == pytest.approx(2.0)


def test_exact_sign_test_known_cases():
    assert k.exact_two_sided_sign_p(0, 0) == 1.0
    assert k.exact_two_sided_sign_p(10, 0) == pytest.approx(2 / 1024)
    assert k.exact_two_sided_sign_p(5, 5) == 1.0


def test_holm_tie_order_and_monotone_adjustment():
    result = k.holm_adjust({"R": 0.01, "D": 0.01, "P": 0.04})
    assert result["R"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert result["D"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert result["P"]["holm_adjusted_p"] == pytest.approx(0.04)


def block_rows(values_r, values_d, values_p):
    out = []
    for i, (r, d, p) in enumerate(zip(values_r, values_d, values_p)):
        out.append({"B_R": r, "B_D": d, "B_P": p, "block_index": i})
    return out


def test_primary_positive_null_reversed_and_mixed_verdicts():
    pos = [1.0] * 150
    neg = [-1.0] * 150
    balanced = [1.0 if i % 2 == 0 else -1.0 for i in range(150)]
    assert k.primary_statistics(block_rows(pos, balanced, balanced))["scientific_verdict"] == k.POSITIVE_VERDICT
    assert k.primary_statistics(block_rows(balanced, balanced, balanced))["scientific_verdict"] == k.NULL_VERDICT
    assert k.primary_statistics(block_rows(neg, balanced, balanced))["scientific_verdict"] == k.REVERSED_VERDICT
    assert k.primary_statistics(block_rows(pos, neg, balanced))["scientific_verdict"] == k.MIXED_VERDICT


def test_promotion_floor_forces_p_one():
    values = [1.0] * 119 + [None] * 31
    summary = k.summarize_endpoint(values)
    assert summary["n_valid"] == 119
    assert summary["promotion_floor_pass"] is False
    assert summary["raw_p"] == 1.0


def test_handoff_path_safety():
    with pytest.raises(k.ContractError):
        k._safe_member("../evil")
    with pytest.raises(k.ContractError):
        k._safe_member("C:/evil")
    assert k._safe_member("manifest.json") == "manifest.json"


def test_git_provenance_mode_split(monkeypatch):
    status = [
        "?? scripts/longterm_k2s_pair_specific_event_dynamics.py",
        "?? tests/test_longterm_k2s_pair_specific_event_dynamics.py",
    ]

    def fake_git(root, *args):
        if args[:2] == ("branch", "--show-current"):
            return k.EXPECTED_BRANCH
        if args[:2] == ("rev-parse", "HEAD"):
            return "head"
        if args[:2] == ("rev-parse", f"head:{k.A0_MODEL_REL}"):
            return "same-model"
        if args[:2] == ("rev-parse", f"{k.A0_COMMIT}:{k.A0_MODEL_REL}"):
            return "same-model"
        if args[:2] == ("rev-parse", f"head:{k.A0_HEADS_REL}"):
            return "same-heads"
        if args[:2] == ("rev-parse", f"{k.A0_COMMIT}:{k.A0_HEADS_REL}"):
            return "same-heads"
        raise AssertionError(args)

    def fake_check_output(args, cwd=None, text=None, **kwargs):
        if args[1:3] == ["status", "--porcelain=v1"]:
            return "\n".join(status) + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(k, "_git", fake_git)
    monkeypatch.setattr(k.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(k.subprocess, "call", lambda *a, **kw: 0)
    assert k.git_provenance(Path("."), instrumentation_preflight=True)["runtime_branch"] == k.EXPECTED_BRANCH
    with pytest.raises(k.ContractError):
        k.git_provenance(Path("."), instrumentation_preflight=False)


def test_static_scientific_boundaries():
    text = P.read_text(encoding="utf-8")
    assert "PAIR_SPECIFIC_EVENT_ALIGNED_NATIVE_STATE_RESPONSE_OBSERVED" in text
    assert "post_consumption_s_t" in text
    assert "cpython_line_trace_local_ssm_state_clone" in text
    assert "PRIMARY_LAYER = 23" in text
    assert "W = 8" in text
    assert '"report.md": report' in text
    assert "prefix_state_sequence_sha256" in text
