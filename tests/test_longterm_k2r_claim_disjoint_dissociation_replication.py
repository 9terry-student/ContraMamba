import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

P = Path(__file__).parents[1] / "scripts" / "longterm_k2r_claim_disjoint_dissociation_replication.py"
spec = importlib.util.spec_from_file_location("k2r", P)
k = importlib.util.module_from_spec(spec)
sys.modules["k2r"] = k
spec.loader.exec_module(k)


class CharTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}


def make_candidate_rows(n=300):
    rows = []
    for i in range(n):
        prefix = f"Claim: claim{i}\nEvidence: evidence{i}\nAdditional evidence:\n"
        rows.append(
            {
                "stable_item_id": f"k2r-v1:{i:064x}",
                "base_claim_sha256": f"{i + 1000:064x}",
                "pair_id": f"generated_fact_{i+301:03d}",
                "prefix_text": prefix,
                "correction_text": "aCORRECTION-LONG",
                "control_text": "aCONTROL---LONG",
                "correction_source_intervention": "none" if i < 150 else "polarity_flip",
            }
        )
    return rows


def branch_base(**updates):
    base = {
        "state_p_sha256": "p",
        "state_p_minus_1_sha256": "pm1",
        "prefix_state_sequence_sha256": "prefix",
        "prefix_state_token_count": 9,
        "R_mean_speed": 1.0,
        "D_mean_turn": 0.1,
        "displacement": 2.0,
        "P_efficiency": 0.5,
    }
    base.update(updates)
    return base


def test_frozen_constants_and_direction_family():
    assert k.K2R_PREREG_AUTHORITY_COMMIT == "54b1a9a2188e3e678e8378b43faa5d527aca1457"
    assert k.K2R_PREREG_SHA256 == "8cc8f10be0dac663267412e8221b87a851cb08ee9bf78fb6c8034ed0b1824dbc"
    assert k.GENERATOR_SHA256 == "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
    assert k.GENERATOR_GIT_BLOB == "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
    assert k.K2S_DEPENDENCY_SHA256 == "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
    assert k.GLOBAL_TEMPLATE_START == 300 and k.GLOBAL_TEMPLATE_STOP == 600
    assert k.W == 8 and k.PRIMARY_LAYER == 23 and k.N_BLOCKS == 150
    assert k.PRIMARY_ORDER == ("R", "D", "DISP", "P")
    assert k.EXPECTED_DIRECTION == {"R": 1, "D": 1, "DISP": -1, "P": -1}


def test_candidate_recipe_schema_is_exactly_preregistered():
    recipe = {
        "schema_version": k.CANDIDATE_SCHEMA,
        "generator_sha256": k.GENERATOR_SHA256,
        "global_template_start": 300,
        "global_template_stop": 600,
        "pair_id": "generated_fact_301",
        "truncation_source_id": "t",
        "correction_source_id": "q",
        "correction_source_intervention": "polarity_flip",
        "control_source_id": "e",
        "prefix_text": "P",
        "correction_text": "C",
        "control_text": "N",
    }
    stable = "k2r-v1:" + k.sha256_bytes(k.canonical_json(recipe))
    assert stable.startswith("k2r-v1:") and len(stable) == len("k2r-v1:") + 64


def test_reciprocal_mapping_is_150_disjoint_blocks():
    rows = make_candidate_rows()
    blocks = k.reciprocal_blocks(rows)
    assert len(blocks) == 150
    seen = set()
    for block in blocks:
        a, b = block["item_a_index"], block["item_b_index"]
        assert b == (a ^ 1) and a == (b ^ 1)
        assert block["block_id"].startswith("k2r-block-")
        seen |= {a, b}
    assert seen == set(range(300))


def test_input_contracts_fail_closed_if_tokenizer_distribution_differs_from_frozen():
    with pytest.raises(k.ContractError, match="K2R_FROZEN_TOKEN_FEASIBILITY_MISMATCH"):
        k.build_input_contracts(make_candidate_rows(), CharTokenizer())


def test_first_divergence_is_after_prefix():
    p = [1, 2, 3]
    c = p + [10, 11, 12]
    n = p + [10, 99, 12]
    assert k.first_divergence_after_prefix(c, n, len(p)) == 4


def test_recipient_layer_result_contains_disp_and_absolute_specificity():
    branches = {
        "matched_corr": branch_base(R_mean_speed=5.0, D_mean_turn=0.3, displacement=2.0, P_efficiency=0.8),
        "matched_ctrl": branch_base(R_mean_speed=2.0, D_mean_turn=0.1, displacement=1.0, P_efficiency=0.2),
        "swapped_corr": branch_base(R_mean_speed=4.0, D_mean_turn=0.2, displacement=5.0, P_efficiency=0.6),
        "swapped_ctrl": branch_base(R_mean_speed=2.0, D_mean_turn=0.1, displacement=1.0, P_efficiency=0.3),
    }
    result = k.recipient_layer_result(branches)
    assert result["signed_contrasts"]["R_matched"] == pytest.approx(3.0)
    assert result["signed_contrasts"]["DISP_matched"] == pytest.approx(1.0)
    assert result["signed_contrasts"]["DISP_swapped"] == pytest.approx(4.0)
    assert result["X_pair_specificity"]["R"] == pytest.approx(1.0)
    assert result["X_pair_specificity"]["DISP"] == pytest.approx(-3.0)
    assert result["X_pair_specificity"]["P"] == pytest.approx(0.3)


def test_recipient_direction_can_be_undefined_but_other_three_cannot():
    branches = {name: branch_base() for name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl")}
    branches["swapped_ctrl"]["D_mean_turn"] = None
    result = k.recipient_layer_result(branches)
    assert result["X_pair_specificity"]["D"] is None
    assert all(result["X_pair_specificity"][m] is not None for m in ("R", "DISP", "P"))


def test_recipient_requires_full_common_prefix_identity():
    branches = {name: branch_base() for name in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl")}
    branches["swapped_ctrl"]["prefix_state_sequence_sha256"] = "different"
    with pytest.raises(k.ContractError, match="SCIENTIFIC_PREFIX_STATE_IDENTITY_FAILURE"):
        k.recipient_layer_result(branches)


def test_block_unit_averages_all_four_endpoints():
    items = []
    for i in range(300):
        items.append(
            {
                "item_index": i,
                "block_index": i // 2,
                "stable_item_id": str(i),
                "X_R": float(i % 2),
                "X_D": None if i < 2 else 1.0,
                "X_DISP": -2.0,
                "X_P": -3.0,
            }
        )
    rows = k.block_rows_from_items(items)
    assert len(rows) == 150
    assert rows[0]["B_R"] == pytest.approx(0.5)
    assert rows[0]["B_D"] is None
    assert rows[0]["B_DISP"] == pytest.approx(-2.0)
    assert rows[0]["B_P"] == pytest.approx(-3.0)


def test_exact_sign_test_known_cases():
    assert k.exact_two_sided_sign_p(0, 0) == 1.0
    assert k.exact_two_sided_sign_p(10, 0) == pytest.approx(2 / 1024)
    assert k.exact_two_sided_sign_p(5, 5) == 1.0


def test_holm_four_endpoint_tie_order_and_monotone_adjustment():
    result = k.holm_adjust({"R": 0.01, "D": 0.01, "DISP": 0.02, "P": 0.04})
    assert result["R"]["holm_adjusted_p"] == pytest.approx(0.04)
    assert result["D"]["holm_adjusted_p"] == pytest.approx(0.04)
    assert result["DISP"]["holm_adjusted_p"] == pytest.approx(0.04)
    assert result["P"]["holm_adjusted_p"] == pytest.approx(0.04)


def rows_for(r, d, disp, p):
    return [
        {"block_index": i, "B_R": rv, "B_D": dv, "B_DISP": xv, "B_P": pv}
        for i, (rv, dv, xv, pv) in enumerate(zip(r, d, disp, p))
    ]


def test_full_replication_requires_all_four_holm_and_direction_matches():
    pos = [1.0] * 150
    neg = [-1.0] * 150
    result = k.primary_statistics(rows_for(pos, pos, neg, neg))
    assert result["scientific_verdict"] == k.REPLICATED_VERDICT
    assert result["full_replication"] is True
    assert result["direction_matched_endpoints"] == ["R", "D", "DISP", "P"]


def test_partial_directional_match_is_not_full_replication():
    pos = [1.0] * 150
    neg = [-1.0] * 150
    balanced = [1.0 if i % 2 == 0 else -1.0 for i in range(150)]
    result = k.primary_statistics(rows_for(pos, pos, neg, balanced))
    assert result["scientific_verdict"] == k.NOT_FULLY_REPLICATED_VERDICT
    assert result["full_replication"] is False
    assert result["directional_contradiction_endpoints"] == []


def test_significant_opposite_endpoint_is_directional_contradiction():
    pos = [1.0] * 150
    neg = [-1.0] * 150
    result = k.primary_statistics(rows_for(pos, pos, pos, neg))
    assert result["scientific_verdict"] == k.CONTRADICTION_VERDICT
    assert result["directional_contradiction_endpoints"] == ["DISP"]


def test_promotion_floor_forces_p_one():
    values = [1.0] * 119 + [None] * 31
    summary = k.summarize_endpoint(values)
    assert summary["n_valid"] == 119
    assert summary["promotion_floor_pass"] is False
    assert summary["raw_p"] == 1.0


def test_claim_text_extraction_is_exact_prefix_claim_segment():
    p = "Claim: Alpha\nEvidence: beta\nAdditional evidence:\n"
    assert k._claim_text_from_prefix(p) == "Claim: Alpha"
    with pytest.raises(k.ContractError):
        k._claim_text_from_prefix("Claim only")


def test_generated_output_artifact_contract_mentions_population_materializations():
    text = P.read_text(encoding="utf-8")
    assert '"generated_source.jsonl"' in text
    assert '"candidate_pool.jsonl"' in text
    assert '"report.md": report' in text
    assert "GENERATED_SOURCE_CANONICAL_SHA256" in text
    assert "CANDIDATE_POOL_CANONICAL_SHA256" in text


def test_k2s_instrumentation_is_dependency_not_reimplemented():
    text = P.read_text(encoding="utf-8")
    assert "K2S_DEPENDENCY_SHA256" in text
    assert "k2s.run_native_branch" in text
    assert "k2s.resolve_capture_binding" in text
    assert "class TraceCollector" not in text
    assert "def analyze_mamba_source" not in text


def test_static_anti_rescue_and_scope_contract():
    text = P.read_text(encoding="utf-8")
    assert "PRIMARY_ORDER = (\"R\", \"D\", \"DISP\", \"P\")" in text
    assert "PRIMARY_LAYER = 23" in text
    assert "W = 8" in text
    assert "GLOBAL_TEMPLATE_START = 300" in text
    assert "GLOBAL_TEMPLATE_STOP = 600" in text
    for forbidden in ("predicted_final_label", "confidence threshold", "margin threshold", "k=7"):
        assert forbidden not in text


def test_git_provenance_preflight_dirty_mode_allows_only_k1_plus_k2r(monkeypatch, tmp_path):
    # Exercise status policy separately from actual file/hash checks by patching those dependencies.
    prereg = tmp_path / k.K2R_PREREG_REL
    generator = tmp_path / k.GENERATOR_REL
    dep = tmp_path / k.K2S_DEPENDENCY_REL
    prereg.parent.mkdir(parents=True, exist_ok=True)
    generator.parent.mkdir(parents=True, exist_ok=True)
    dep.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_bytes(b"p")
    generator.write_bytes(b"g")
    dep.write_bytes(b"s")

    monkeypatch.setattr(k, "file_sha256", lambda path: {
        prereg.resolve(): k.K2R_PREREG_SHA256,
        generator.resolve(): k.GENERATOR_SHA256,
        dep.resolve(): k.K2S_DEPENDENCY_SHA256,
    }[path.resolve()])

    def fake_git(root, *args):
        if args == ("branch", "--show-current"):
            return k.EXPECTED_BRANCH
        if args == ("rev-parse", "HEAD"):
            return "head"
        mapping = {
            ("rev-parse", f"{k.K2R_GENERATOR_AUTHORITY_COMMIT}:{k.GENERATOR_REL}"): k.GENERATOR_GIT_BLOB,
            ("rev-parse", f"head:{k.GENERATOR_REL}"): k.GENERATOR_GIT_BLOB,
            ("rev-parse", f"head:{k.K2S_DEPENDENCY_REL}"): k.K2S_DEPENDENCY_GIT_BLOB,
            ("rev-parse", f"head:{k.A0_MODEL_REL}"): "same-model",
            ("rev-parse", f"{k.A0_COMMIT}:{k.A0_MODEL_REL}"): "same-model",
            ("rev-parse", f"head:{k.A0_HEADS_REL}"): "same-heads",
            ("rev-parse", f"{k.A0_COMMIT}:{k.A0_HEADS_REL}"): "same-heads",
        }
        if args in mapping:
            return mapping[args]
        raise AssertionError(args)

    status = [
        "?? scripts/longterm_k1_native_state_kinematics.py",
        "?? tests/test_longterm_k1_native_state_kinematics.py",
        "?? scripts/longterm_k2r_claim_disjoint_dissociation_replication.py",
        "?? tests/test_longterm_k2r_claim_disjoint_dissociation_replication.py",
    ]

    def fake_check_output(args, cwd=None, text=None, **kwargs):
        if args[1:3] == ["status", "--porcelain=v1"]:
            return "\n".join(status) + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(k, "_git", fake_git)
    monkeypatch.setattr(k.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(k.subprocess, "call", lambda *a, **kw: 0)
    assert k.git_provenance(tmp_path, instrumentation_preflight=True)["runtime_branch"] == k.EXPECTED_BRANCH
    with pytest.raises(k.ContractError, match="GIT_DIRTY_CONTRACT_MISMATCH"):
        k.git_provenance(tmp_path, instrumentation_preflight=False)
