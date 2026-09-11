from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "longterm_k3t_phase_balanced_r_transportability.py"
spec = importlib.util.spec_from_file_location("k3t_impl", MODULE_PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def block_rows(values):
    assert len(values) == m.N_BLOCKS
    return [
        {
            "schema_version": m.BLOCK_SCHEMA,
            "block_index": i,
            "block_id": f"k3t-block-{i:03d}",
            "B_R": value,
        }
        for i, value in enumerate(values)
    ]


def test_frozen_constants():
    assert m.K3T_PREREG_COMMIT == "58cbcd316c7714ddc8c041c2a2ec4376e79a4bd0"
    assert m.K3T_PREREG_SHA256 == "57b5bd2375fbacb7ef5e22260f3f1ecf71dba839f383f7726c03544e33a45cc5"
    assert m.GLOBAL_TEMPLATE_START == 900
    assert m.GLOBAL_TEMPLATE_STOP == 1236
    assert m.N_ITEMS == 336
    assert m.N_BLOCKS == 168
    assert m.PHASE_PERIOD == 168
    assert m.W == 8
    assert m.PRIMARY_LAYER == 23
    assert m.GENERATED_SOURCE_SHA256 == "fb699bcc99e00b8c437fd49c204345933d61c69215af853746bc4639545c7400"
    assert m.CANDIDATE_POOL_SHA256 == "d95d245e358ff497ea50b95e4f54d1192be64d09fec2c06538fc15f75b09ef70"
    assert m.RECIPROCAL_MAPPING_SHA256 == "1c458054540ac0d38857cb3f55595e0b286428225ef11ed22059981dfd1e28ad"


def test_dependency_bindings_are_frozen():
    assert m.GENERATOR_SHA256 == "4e9798591fbfffb6d15ea9b2f8cf5cd804a9e76713b9f6b13b354fe6ce93aa5c"
    assert m.GENERATOR_GIT_BLOB == "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
    assert m.K2S_HELPER_SHA256 == "f741780e7199452e64b7c4a3d70f50f3e55ebc84296f69f288aa317582de84b8"
    assert m.K2S_HELPER_GIT_BLOB == "3a651fb508669bdcf72441a4869b863d6eee6c1f"
    assert m.RECURRENCE_SOURCE_SHA256 == "23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41"


@pytest.mark.parametrize(
    "g,expected",
    [
        (900, (900 - 30) % 168),
        (1068, (1068 - 30) % 168),
        (1235, (1235 - 30) % 168),
    ],
)
def test_phase_class(g, expected):
    assert m.phase_class(g) == expected


def test_phase_class_rejects_seed_region():
    with pytest.raises(m.ContractError, match="GLOBAL_TEMPLATE_BEFORE_GENERATED_REGION"):
        m.phase_class(29)


def test_reciprocal_mapping_shape_and_xor():
    candidates = [
        {"stable_item_id": f"id-{i:03d}"}
        for i in range(m.N_ITEMS)
    ]
    rows = m.reciprocal_mapping(candidates)
    assert len(rows) == 168
    assert rows[0]["item_a_donor_index"] == 1
    assert rows[0]["item_b_donor_index"] == 0
    assert rows[-1]["item_a_donor_index"] == 335
    assert rows[-1]["item_b_donor_index"] == 334


def test_exact_sign_p_symmetry_and_extremes():
    assert m.exact_two_sided_sign_p(0, 0) == 1.0
    assert m.exact_two_sided_sign_p(10, 10) == 1.0
    assert m.exact_two_sided_sign_p(20, 0) == pytest.approx(2.0 / (2 ** 20))
    assert m.exact_two_sided_sign_p(12, 5) == m.exact_two_sided_sign_p(5, 12)


@pytest.mark.parametrize("a,b", [(-1, 0), (0, -1), (1.2, 2), (2, 1.2)])
def test_exact_sign_p_rejects_invalid_counts(a, b):
    with pytest.raises(m.ContractError):
        m.exact_two_sided_sign_p(a, b)


def test_positive_primary_verdict():
    values = [1.0] * 120 + [-1.0] * 40 + [0.0] * 8
    stats = m.primary_statistics(block_rows(values))
    assert stats["n_valid"] == 168
    assert stats["n_eff"] == 160
    assert stats["positive_count"] == 120
    assert stats["negative_count"] == 40
    assert stats["zero_count"] == 8
    assert stats["family_m"] == 1
    assert stats["adjusted_p"] == stats["raw_p"]
    assert stats["rank_biserial_sign_effect"] == 0.5
    assert stats["scientific_verdict"] == m.POSITIVE_VERDICT


def test_negative_primary_verdict():
    values = [1.0] * 40 + [-1.0] * 120 + [0.0] * 8
    stats = m.primary_statistics(block_rows(values))
    assert stats["rank_biserial_sign_effect"] == -0.5
    assert stats["scientific_verdict"] == m.NEGATIVE_VERDICT


def test_null_balanced_primary_verdict():
    values = [1.0] * 84 + [-1.0] * 84
    stats = m.primary_statistics(block_rows(values))
    assert stats["raw_p"] == 1.0
    assert stats["rank_biserial_sign_effect"] == 0.0
    assert stats["scientific_verdict"] == m.NULL_VERDICT


def test_floor_failure_is_null():
    values = [1.0] * 20 + [0.0] * 148
    stats = m.primary_statistics(block_rows(values))
    assert stats["n_eff"] == 20
    assert stats["promotion_floor_pass"] is False
    assert stats["raw_p"] == 1.0
    assert stats["scientific_verdict"] == m.NULL_VERDICT


def test_undefined_primary_is_integrity_failure():
    values = [1.0] * 167 + [None]
    with pytest.raises(m.ContractError, match="PRIMARY_R_UNDEFINED_INTEGRITY_FAILURE"):
        m.primary_statistics(block_rows(values))


def test_nonfinite_primary_is_integrity_failure():
    values = [1.0] * 167 + [float("nan")]
    with pytest.raises(m.ContractError, match="PRIMARY_R_NONFINITE"):
        m.primary_statistics(block_rows(values))


def test_block_rows_exact_reciprocal_mean():
    items = []
    for i in range(m.N_ITEMS):
        items.append({
            "block_index": i // 2,
            "stable_item_id": f"id-{i}",
            "X_R": float(i % 2),
            # Reference values intentionally irrelevant.
            "D": 10_000 + i,
            "P": -10_000 - i,
        })
    rows = m.block_rows_from_items(items)
    assert len(rows) == 168
    assert all(row["B_R"] == 0.5 for row in rows)


def test_reference_diagnostics_do_not_enter_primary_statistics():
    base = [1.0] * 120 + [-1.0] * 40 + [0.0] * 8
    a = m.primary_statistics(block_rows(base))
    b_rows = block_rows(base)
    for row in b_rows:
        row["D"] = -999999.0
        row["P"] = 999999.0
        row["DISP"] = -123.0
    b = m.primary_statistics(b_rows)
    assert a == b
    assert a["scientific_verdict"] == m.POSITIVE_VERDICT


def valid_authority_text(**overrides):
    values = {
        "K3T_EXECUTION_AUTHORITY_SCHEMA": m.AUTHORITY_SCHEMA,
        "K3T_SCIENTIFIC_RECURRENT_STATE_EXECUTION_AUTHORIZED": "YES",
        "K3T_ONE_SCIENTIFIC_EXECUTION": "YES",
        "K3T_EXECUTION_IMPLEMENTATION_COMMIT": "a" * 40,
        "K3T_EXECUTION_RUNNER_SHA256": "b" * 64,
        "K3T_EXECUTION_TEST_SHA256": "c" * 64,
        "K3T_PREREG_COMMIT": m.K3T_PREREG_COMMIT,
        "K3T_PREREG_SHA256": m.K3T_PREREG_SHA256,
    }
    values.update(overrides)
    return "\n".join(f"`{k}={v}`" for k, v in values.items()) + "\n"


def test_authority_marker_parser_accepts_exact_set():
    parsed = m.parse_authority_markers(valid_authority_text())
    assert parsed["K3T_EXECUTION_AUTHORITY_SCHEMA"] == m.AUTHORITY_SCHEMA
    assert parsed["K3T_PREREG_COMMIT"] == m.K3T_PREREG_COMMIT


def test_authority_marker_parser_rejects_missing_marker():
    text = valid_authority_text().replace(
        f"`K3T_ONE_SCIENTIFIC_EXECUTION=YES`\n",
        "",
    )
    with pytest.raises(m.ContractError, match="AUTHORITY_MARKER_SET_MISMATCH"):
        m.parse_authority_markers(text)


def test_authority_marker_parser_rejects_duplicate_marker():
    text = valid_authority_text() + "`K3T_ONE_SCIENTIFIC_EXECUTION=YES`\n"
    with pytest.raises(m.ContractError, match="AUTHORITY_DUPLICATE_MARKER"):
        m.parse_authority_markers(text)


def test_output_dir_must_be_outside_repo_and_new(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(m.ContractError, match="OUTPUT_DIRECTORY_MUST_BE_OUTSIDE_REPO"):
        m._output_dir(repo, repo / "k3t-r-transportability-abc")
    outside = tmp_path / "outside" / "k3t-r-transportability-abc"
    assert m._output_dir(repo, outside) == outside.resolve()
    outside.mkdir(parents=True)
    with pytest.raises(m.ContractError, match="OUTPUT_DIRECTORY_ALREADY_EXISTS"):
        m._output_dir(repo, outside)


def test_output_dir_requires_frozen_prefix(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside" / "wrong-name"
    with pytest.raises(m.ContractError, match="OUTPUT_DIRECTORY_BASENAME_INVALID"):
        m._output_dir(repo, outside)


def test_scientific_authority_missing_fails_before_other_scientific_work(tmp_path, monkeypatch):
    # Narrow unit proof: once dependency checks are mocked as already passed,
    # a missing authority file fails immediately.
    monkeypatch.setattr(m, "_require_prereg_and_dependencies", lambda root: None)
    with pytest.raises(m.ContractError, match="K3T_EXECUTION_AUTHORITY_MISSING"):
        m.require_scientific_authority(tmp_path)


def test_candidate_recipe_canonical_hash_is_deterministic():
    recipe_a = {"b": 2, "a": 1}
    recipe_b = {"a": 1, "b": 2}
    assert m.canonical_json(recipe_a) == m.canonical_json(recipe_b)
    assert m.sha256_bytes(m.canonical_json(recipe_a)) == m.sha256_bytes(m.canonical_json(recipe_b))


def test_jsonl_parser_rejects_cr_and_bom():
    good = b'{"a":1}\n'
    assert m.parse_jsonl_bytes(good) == [{"a": 1}]
    with pytest.raises(m.ContractError, match="JSONL_CR_FORBIDDEN"):
        m.parse_jsonl_bytes(b'{"a":1}\r\n')
    with pytest.raises(m.ContractError, match="JSONL_BOM_FORBIDDEN"):
        m.parse_jsonl_bytes(b"\xef\xbb\xbf" + good)


def test_claim_text_from_prefix():
    prefix = "Claim: alpha\nEvidence: beta\nAdditional evidence:\n"
    assert m.claim_text_from_prefix(prefix) == "Claim: alpha"


def test_branch_reference_metrics_excludes_inferential_aggregate():
    branch = {
        "R_mean_speed": 1.0,
        "D_mean_turn": 2.0,
        "P_efficiency": 3.0,
        "path_length": 4.0,
        "displacement": 5.0,
        "state_window_sha256": "a",
        "state_p_minus_1_sha256": "b",
        "state_p_sha256": "c",
        "prefix_state_sequence_sha256": "d",
        "prefix_state_token_count": 10,
        "some_other_field": "forbidden",
    }
    out = m._branch_reference_metrics(branch)
    assert out["R_mean_speed"] == 1.0
    assert out["D_mean_turn"] == 2.0
    assert "some_other_field" not in out


def test_cli_modes_are_mutually_exclusive():
    parser = m.build_parser()
    args = parser.parse_args(["--state-blind-preflight"])
    assert args.state_blind_preflight
    with pytest.raises(SystemExit):
        parser.parse_args(["--state-blind-preflight", "--scientific"])


def test_verdict_strings_are_exact():
    assert m.POSITIVE_VERDICT == "R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_REPLICATED"
    assert m.NEGATIVE_VERDICT == "R_PHASE_BALANCED_TRANSPORTABILITY_SIGNAL_CONTRADICTED"
    assert m.NULL_VERDICT == "R_PHASE_BALANCED_TRANSPORTABILITY_NOT_ESTABLISHED"


def test_direct_cli_repo_root_is_bound_on_sys_path():
    assert m.REPO_ROOT == MODULE_PATH.resolve().parents[1]
    assert str(m.REPO_ROOT) in m.sys.path
