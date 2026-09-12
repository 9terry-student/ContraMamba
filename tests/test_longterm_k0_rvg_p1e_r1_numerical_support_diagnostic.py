from __future__ import annotations

import importlib.util
import ast
import inspect
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_p1e_r1_numerical_support_diagnostic.py"

spec = importlib.util.spec_from_file_location("k0_rvg_p1e_r1", SCRIPT)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_frozen_authority_constants():
    assert m.R1_PROTOCOL_COMMIT == "02f184a186aa6b90fc98ece717723de0adafc89c"
    assert m.R1_INCIDENT_COMMIT == "6fe6a80f5fa0125971ac83d436e6e400cb7ab6f1"
    assert m.R1_PROTOCOL_SHA256 == "d9a10812c96d9453460356658d09f282a6f4da022b299ce9ceab99cb098f2fb0"


def test_fixed_scope_constants():
    assert m.R1_LOCAL_TEMPLATE_INDEX == 0
    assert m.R1_BRANCH_ROLE == "matched_corr"
    assert m.PRIMARY_LAYER == 23
    assert m.WINDOW == 8
    assert m.TARGET_COUNT == 9
    assert m.VELOCITY_ATOL == 1e-6
    assert m.VELOCITY_RTOL == 1e-5


def test_p1_binding_constants():
    assert m.P1_IMPLEMENTATION_COMMIT == "2e6bb106d5d3081b7ae69ec4cde652e79d36070c"
    assert m.P1_RUNNER_SHA256 == "2bd59a25a6303dc86c36c9438296b42197cc67019e9d4e9ee2fbbe1d832ec6eb"
    assert m.P1_TEST_SHA256 == "06c30c974029126fdc853a235071376738cc2769409369ccdb60f0878bac7706"


def test_observer_binding_constants():
    assert m.OBSERVER_SHA256 == "12542e32d49b368e727de782b0cb833991464503b1fab56d375f15ec58649c25"
    assert m.OBSERVER_GIT_BLOB == "f2dbdfe52661eca384897578ab272e602e36deac"


def test_p0_six_file_hashes():
    assert len(m.P0_ARTIFACT_SHA256) == 6
    assert m.P0_ARTIFACT_SHA256["candidate_pool.jsonl"] == (
        "743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246"
    )
    assert m.P0_ARTIFACT_SHA256["token_contracts.jsonl"] == (
        "6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998"
    )


def test_target_indices_exact_nine():
    assert m.target_indices(10) == (9, 10, 11, 12, 13, 14, 15, 16, 17)
    with pytest.raises(m.ContractError, match="TARGET_TE_LT_ONE"):
        m.target_indices(0)


def test_retain_primary_records_discards_full_capture_map():
    targets = m.target_indices(10)
    records = {
        (layer, token): object()
        for layer in range(m.EXPECTED_LAYER_COUNT)
        for token in targets
    }
    primary_before = {
        token: records[(m.PRIMARY_LAYER, token)]
        for token in targets
    }
    primary = m.retain_primary_records_and_discard_nonprimary(records, targets)
    assert records == {}
    assert primary == primary_before
    assert set(primary) == set(targets)


class Meta:
    shape = m.EXPECTED_STATE_SHAPE
    dtype = m.EXPECTED_DTYPE
    device = m.EXPECTED_DEVICE


class Record:
    layer_index = m.PRIMARY_LAYER
    token_index = 7
    s_prev_meta = Meta()


def make_record(s_prev, g, w):
    rec = Record()
    rec.s_prev = s_prev
    rec.g = g
    rec.w = w
    rec.s_post = g * s_prev + w
    return rec


def test_pass_fixture_exact_and_allclose():
    row = m.diagnose_record(m.fabricated_fixture(True))
    assert row["exact_recurrence_pass"] is True
    assert row["frozen_allclose_pass"] is True
    assert row["failing_element_count"] == 0


def test_failure_fixture_exact_but_allclose_fails():
    row = m.diagnose_record(m.fabricated_fixture(False))
    assert row["exact_recurrence_pass"] is True
    assert row["frozen_allclose_pass"] is False
    assert row["failing_element_count"] > 0
    assert row["max_scaled_tolerance_residual"] > 1.0


def test_exact_recurrence_failure_diagnostic_does_not_raise():
    torch = pytest.importorskip("torch")
    rec = make_record(
        torch.ones(m.EXPECTED_STATE_SHAPE, dtype=torch.float32),
        torch.ones(m.EXPECTED_STATE_SHAPE, dtype=torch.float32),
        torch.zeros(m.EXPECTED_STATE_SHAPE, dtype=torch.float32),
    )
    rec.s_post = rec.s_post.clone()
    rec.s_post.reshape(-1)[0] += 1.0
    row = m.diagnose_record(rec)
    assert row["exact_recurrence_pass"] is False


def test_elementwise_tolerance_failure_count_fraction():
    row = m.diagnose_record(m.fabricated_fixture(False))
    assert row["element_count"] == 1 * 1536 * 16
    assert row["failing_element_count"] == row["element_count"]
    assert row["failing_element_fraction"] == 1.0
    assert row["tolerance_scale_min"] > 0.0
    assert row["tolerance_scale_max"] >= row["tolerance_scale_min"]


def test_max_residual_flat_index_deterministic():
    torch = pytest.importorskip("torch")
    s_prev = torch.ones(m.EXPECTED_STATE_SHAPE, dtype=torch.float32)
    g = torch.ones_like(s_prev)
    w = torch.zeros_like(s_prev)
    rec = make_record(s_prev, g, w)
    rec.s_post = rec.s_post.clone()
    rec.s_post.reshape(-1)[123] += 1.0
    row = m.diagnose_record(rec)
    assert row["max_residual_element"]["flat_index"] == 123


def test_frobenius_and_state_velocity_diagnostics_finite():
    row = m.diagnose_record(m.fabricated_fixture(False))
    keys = (
        "residual_frobenius_norm",
        "v_raw32_frobenius_norm",
        "v_rearr32_frobenius_norm",
        "relative_frobenius_residual",
        "s_prev_frobenius_norm",
        "s_post_frobenius_norm",
        "w_frobenius_norm",
        "carry_frobenius_norm",
        "state_to_raw_velocity_norm_ratio",
    )
    assert all(math.isfinite(row[key]) for key in keys)


def test_float64_snapshot_diagnostic_present_and_finite():
    row = m.diagnose_record(m.fabricated_fixture(False))
    snap = row["float64_snapshot"]
    assert set(snap) == {
        "max_abs_residual",
        "residual_frobenius_norm",
        "v_raw64_frobenius_norm",
        "relative_frobenius_residual",
    }
    assert all(math.isfinite(v) for v in snap.values())


def test_state_hashes_deterministic():
    row1 = m.diagnose_record(m.fabricated_fixture(True))
    row2 = m.diagnose_record(m.fabricated_fixture(True))
    assert row1["state_hashes"] == row2["state_hashes"]


def fake_rows(*, exact_pass=True, failing_count=0):
    rows = []
    for i in range(9):
        rows.append({
            "token_index": 10 + i,
            "exact_recurrence_pass": exact_pass,
            "frozen_allclose_pass": i >= failing_count,
            "max_scaled_tolerance_residual": float(i + 1),
        })
    return rows


def test_classification_exact_recurrence_failure():
    rows = fake_rows(exact_pass=True)
    rows[3]["exact_recurrence_pass"] = False
    classification, profile, support = m.classify_coordinate_rows(rows)
    assert classification == "EXACT_RECURRENCE_OR_CAPTURE_FAILURE"


def test_classification_float32_tolerance_failure():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(exact_pass=True, failing_count=2)
    )
    assert classification == (
        "EXACT_RECURRENCE_INTACT_FLOAT32_REARRANGEMENT_TOLERANCE_FAILURE"
    )


def test_classification_no_reproduced_failure():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(exact_pass=True, failing_count=0)
    )
    assert classification == (
        "ORIGINAL_FAILURE_NOT_REPRODUCED_IN_FIXED_SINGLE_BRANCH_DIAGNOSTIC"
    )


def test_support_profile_single_failure():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(failing_count=1)
    )
    assert profile == "SINGLE_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH"
    assert support["failing_coordinate_count"] == 1
    assert support["first_failing_token_index"] == 10


def test_support_profile_multiple_failure():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(failing_count=3)
    )
    assert profile == "MULTI_COORDINATE_FAILURE_WITHIN_FIXED_BRANCH"
    assert support["failing_coordinate_count"] == 3


def test_support_profile_no_failure():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(failing_count=0)
    )
    assert profile == "NO_FAILURE_REPRODUCED_WITHIN_FIXED_BRANCH"
    assert support["first_failing_token_index"] is None


def test_support_median_fixed_nine():
    classification, profile, support = m.classify_coordinate_rows(
        fake_rows(failing_count=0)
    )
    assert support["median_scaled_tolerance_residual"] == 5.0
    assert support["maximum_scaled_tolerance_residual"] == 9.0


def test_duplicate_coordinate_rejected():
    rows = fake_rows()
    rows[-1]["token_index"] = rows[0]["token_index"]
    with pytest.raises(m.ContractError, match="R1_DUPLICATE_TOKEN_INDEX"):
        m.classify_coordinate_rows(rows)


def test_json_rejects_nan():
    with pytest.raises(ValueError):
        m.canonical_json_bytes({"x": float("nan")})


def test_json_none_serializes_null():
    assert m.canonical_json_bytes({"x": None}) == b'{"x":null}'


def test_cli_has_no_scientific_quantity_overrides():
    parser = m.build_parser()
    options = {opt for action in parser._actions for opt in action.option_strings}
    assert {
        "--synthetic-preflight",
        "--execute-diagnostic",
        "--seed180-handoff",
        "--execution-authority",
        "--output-dir",
    } <= options
    for forbidden in (
        "--item",
        "--branch",
        "--layer",
        "--window",
        "--atol",
        "--rtol",
        "--metric",
        "--population",
        "--threshold",
    ):
        assert forbidden not in options


def test_runner_ast_contains_no_p1_endpoint_implementation_or_calls():
    tree = ast.parse(SCRIPT.read_text("utf-8"))
    forbidden_names = {
        "compute_pair_metrics",
        "item_endpoint_metrics",
        "aggregate_blocks",
        "summarize_endpoints",
        "exact_two_sided_sign_test",
        "holm_m2",
        "overall_verdict",
    }
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert not (defined & forbidden_names)
    assert not (loaded & forbidden_names)
    assert not (attrs & forbidden_names)


def test_execution_gate_precedes_model_construction():
    source = inspect.getsource(m.execute_diagnostic)
    assert source.index("authenticate_r1_execution_authority") < source.index(
        "validate_runtime_and_build_model"
    )


def test_execution_authority_absent_rejected():
    with pytest.raises(m.ContractError, match="R1E_AUTHORITY_NOT_TRACKED"):
        m.authenticate_r1_execution_authority(ROOT, "reports/does_not_exist.md")


def test_execution_authority_absolute_path_rejected():
    with pytest.raises(m.ContractError, match="R1E_AUTHORITY_PATH_ABSOLUTE"):
        m.authenticate_r1_execution_authority(
            ROOT,
            str((ROOT / "reports" / "x.md").resolve()),
        )


def test_validate_r1_implementation_commit_mock(monkeypatch):
    commit = "1" * 40

    def fake_git(root, *args):
        if args == ("rev-parse", f"{commit}^"):
            return m.R1_PROTOCOL_COMMIT
        if args == ("diff-tree", "--no-commit-id", "--name-only", "-r", commit):
            return m.RUNNER_REL + "\n" + m.TEST_REL
        if args == ("rev-parse", f"{commit}:{m.RUNNER_REL}"):
            return "2" * 40
        if args == ("rev-parse", f"{commit}:{m.TEST_REL}"):
            return "3" * 40
        raise AssertionError(args)

    monkeypatch.setattr(m, "_git", fake_git)
    assert m.validate_r1_implementation_commit(ROOT, commit) == {
        "runner_blob": "2" * 40,
        "test_blob": "3" * 40,
    }


def test_validate_r1_implementation_wrong_parent_rejected(monkeypatch):
    commit = "1" * 40
    monkeypatch.setattr(
        m,
        "_git",
        lambda root, *args: "0" * 40,
    )
    with pytest.raises(m.ContractError, match="R1_IMPLEMENTATION_PARENT_MISMATCH"):
        m.validate_r1_implementation_commit(ROOT, commit)


def test_atomic_output_exact_one_file(tmp_path):
    out = tmp_path / "run"
    sha = m.write_diagnostic_atomic(out, {"schema_version": "x", "value": 1})
    assert (out / m.DIAGNOSTIC_FILENAME).is_file()
    assert [p.name for p in out.iterdir()] == [m.DIAGNOSTIC_FILENAME]
    assert len(sha) == 64


def test_atomic_output_existing_dir_rejected(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(m.ContractError, match="R1_OUTPUT_DIR_ALREADY_EXISTS"):
        m.write_diagnostic_atomic(out, {"x": 1})


def test_diagnostic_filename_exact():
    assert m.DIAGNOSTIC_FILENAME == "numerical_support_diagnostic.json"


def test_no_import_time_model_execution():
    source = SCRIPT.read_text("utf-8")
    assert 'if __name__ == "__main__":' in source
    assert "build_a0_model" in source


@pytest.mark.skipif(
    not (ROOT / m.P0_ARCHIVE_REL).exists(),
    reason="requires real repository archive",
)
def test_real_p0_static_item0_authentication():
    p0 = m.authenticate_p0_static(ROOT)
    assert int(p0.candidate["local_template_index"]) == 0
    assert p0.candidate["pair_id"] == "generated_fact_1237"
    assert p0.candidate["stable_item_id"] == p0.token_contract["stable_item_id"]


@pytest.mark.skipif(
    not (ROOT / m.R1_PROTOCOL_REL).exists(),
    reason="requires real repository",
)
def test_real_static_dependency_authentication():
    deps = m.authenticate_static_dependencies(ROOT)
    assert deps["r1_protocol_sha256"] == m.R1_PROTOCOL_SHA256
    assert deps["observer_sha256"] == m.OBSERVER_SHA256
