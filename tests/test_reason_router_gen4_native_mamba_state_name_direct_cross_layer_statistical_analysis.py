from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_native_mamba_state_name_direct_cross_layer_statistical_analysis.py"
)
spec = importlib.util.spec_from_file_location("direct_cross_layer_stats", MODULE_PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_phase_rows(n: int = 300) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for endpoint_index, endpoint in enumerate(m.ENDPOINTS, start=1):
        for estimand_index, estimand in enumerate(m.PHASE_F_ESTIMANDS, start=1):
            for i in range(n):
                pair_id = f"pair-{i:03d}"
                base = 0.01 * endpoint_index + 0.001 * estimand_index + 0.0001 * i
                if estimand == m.ESTIMAND:
                    base = 0.10 * endpoint_index + 0.0005 * i
                rows.append(
                    {
                        "schema_version": m.PHASE_F_INPUT_SCHEMA,
                        "source_pair_id": pair_id,
                        "endpoint": endpoint,
                        "estimand": estimand,
                        "contrast_value": repr(base),
                    }
                )
    return rows


def make_q_rows(n: int = 300) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for layer in m.SECONDARY_LAYERS:
        for endpoint_index, endpoint in enumerate(m.ENDPOINTS, start=1):
            for i in range(n):
                pair_id = f"pair-{i:03d}"
                midpoint = 0.10 * endpoint_index + 0.0005 * i
                shift = (
                    0.020 * endpoint_index + 0.0002 * i
                    if layer == 5
                    else -0.030 * endpoint_index + 0.0003 * i
                )
                rows.append(
                    {
                        "schema_version": m.Q1_Q3_INPUT_SCHEMA,
                        "source_pair_id": pair_id,
                        "layer_index": str(layer),
                        "endpoint": endpoint,
                        "estimand": m.ESTIMAND,
                        "contrast_value": repr(midpoint + shift),
                    }
                )
    return rows


def make_inputs(n: int = 300):
    return make_phase_rows(n), make_q_rows(n)


def test_constants_and_exact_contract():
    assert m.SCIENTIFIC_SPECIFICATION_COMMIT == "b3e0ade126622f244b557e1db07296c622bd7202"
    assert m.IMPLEMENTATION_AUTHORITY_COMMIT == "724c28b528b0f182bc0c79cf5ee0b3adfca76ec2"
    assert m.MIDPOINT_LAYER_INDEX == 11
    assert m.SECONDARY_LAYERS == (5, 17)
    assert m.LAYER_CONTRASTS == (("5_MINUS_11", 5), ("17_MINUS_11", 17))
    assert m.ENDPOINTS == (
        "POST4_SPEED",
        "POST4_TURNING",
        "POST4_PATH_EFFICIENCY",
    )
    assert m.ESTIMAND == "DELTA_NAME"
    assert m.HYPOTHESIS_COUNT == 6
    assert m.SOURCE_PAIR_COUNT == 300


def test_phase_f_and_q1q3_synthetic_validation_accepts_exact_matrices():
    phase, q = make_inputs(3)
    phase_lookup, phase_ids = m.validate_phase_f_rows(phase, expected_pair_count=3)
    q_lookup, q_ids = m.validate_q1_q3_rows(q, expected_pair_count=3)
    assert len(phase_lookup) == 9
    assert len(q_lookup) == 18
    assert phase_ids == q_ids == ["pair-000", "pair-001", "pair-002"]


def test_correct_subtraction_directions_all_endpoints():
    phase, q = make_inputs(3)
    rows = m.construct_pair_level_differences(phase, q, expected_pair_count=3)
    observed = {
        (row["layer_contrast"], row["endpoint"], row["source_pair_id"]): row
        for row in rows
    }
    for endpoint_index, endpoint in enumerate(m.ENDPOINTS, start=1):
        for i in range(3):
            pair_id = f"pair-{i:03d}"
            assert observed[("5_MINUS_11", endpoint, pair_id)][
                "cross_layer_difference"
            ] == pytest.approx(0.020 * endpoint_index + 0.0002 * i)
            assert observed[("17_MINUS_11", endpoint, pair_id)][
                "cross_layer_difference"
            ] == pytest.approx(-0.030 * endpoint_index + 0.0003 * i)


def test_pair_output_order_and_component_audit_values():
    phase, q = make_inputs(300)
    rows = m.construct_pair_level_differences(phase, q)
    assert len(rows) == 1800
    assert rows[0]["layer_contrast"] == "5_MINUS_11"
    assert rows[0]["endpoint"] == "POST4_SPEED"
    assert rows[0]["source_pair_id"] == "pair-000"
    assert rows[299]["source_pair_id"] == "pair-299"
    assert rows[300]["endpoint"] == "POST4_TURNING"
    assert rows[900]["layer_contrast"] == "17_MINUS_11"
    assert rows[-1]["endpoint"] == "POST4_PATH_EFFICIENCY"
    assert rows[-1]["source_pair_id"] == "pair-299"
    first = rows[0]
    assert first["cross_layer_difference"] == pytest.approx(
        first["secondary_contrast_value"] - first["midpoint_contrast_value"]
    )
    assert all(row["midpoint_layer_index"] == 11 for row in rows)
    assert all(row["estimand"] == "DELTA_NAME" for row in rows)


@pytest.mark.parametrize(
    ("which", "mutation", "message"),
    [
        ("phase", "duplicate", "duplicate Phase F structural key"),
        ("q", "duplicate", "duplicate Q1/Q3 structural key"),
        ("phase", "endpoint", "unexpected Phase F endpoint"),
        ("q", "endpoint", "unexpected Q1/Q3 endpoint"),
        ("phase", "estimand", "unexpected Phase F estimand"),
        ("q", "estimand", "unexpected Q1/Q3 estimand"),
        ("q", "layer", "unexpected Q1/Q3 layer_index"),
        ("phase", "nan", "must be finite"),
        ("q", "inf", "must be finite"),
    ],
)
def test_input_validation_fail_closed(which: str, mutation: str, message: str):
    phase, q = make_inputs(2)
    rows = phase if which == "phase" else q
    if mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "endpoint":
        rows[0]["endpoint"] = "POST4_OTHER"
    elif mutation == "estimand":
        rows[0]["estimand"] = "DELTA_OTHER"
    elif mutation == "layer":
        rows[0]["layer_index"] = "23"
    elif mutation == "nan":
        rows[0]["contrast_value"] = "nan"
    elif mutation == "inf":
        rows[0]["contrast_value"] = "inf"

    with pytest.raises(RuntimeError, match=message):
        if which == "phase":
            m.validate_phase_f_rows(rows, expected_pair_count=2)
        else:
            m.validate_q1_q3_rows(rows, expected_pair_count=2)


def test_missing_cells_and_row_count_fail_closed():
    phase, q = make_inputs(2)
    with pytest.raises(RuntimeError, match="Phase F row count mismatch"):
        m.validate_phase_f_rows(phase[:-1], expected_pair_count=2)
    with pytest.raises(RuntimeError, match="Q1/Q3 row count mismatch"):
        m.validate_q1_q3_rows(q[:-1], expected_pair_count=2)


def test_cross_input_source_pair_population_mismatch_rejected():
    phase, q = make_inputs(2)
    for row in q:
        if row["source_pair_id"] == "pair-001":
            row["source_pair_id"] = "pair-999"
    with pytest.raises(RuntimeError, match="cross-input source-pair population mismatch"):
        m.construct_pair_level_differences(phase, q, expected_pair_count=2)


def test_nonlexicographic_cell_order_rejected():
    phase, q = make_inputs(3)
    phase[0], phase[1] = phase[1], phase[0]
    with pytest.raises(RuntimeError, match="not lexicographic"):
        m.validate_phase_f_rows(phase, expected_pair_count=3)

    phase, q = make_inputs(3)
    q[0], q[1] = q[1], q[0]
    with pytest.raises(RuntimeError, match="not lexicographic"):
        m.validate_q1_q3_rows(q, expected_pair_count=3)


def test_exact_field_contracts_reject_missing_and_extra_fields():
    phase, q = make_inputs(2)
    phase[0]["extra"] = "x"
    with pytest.raises(RuntimeError, match="field contract mismatch"):
        m.validate_phase_f_rows(phase, expected_pair_count=2)

    phase, q = make_inputs(2)
    q[0].pop("estimand")
    with pytest.raises(RuntimeError, match="field contract mismatch"):
        m.validate_q1_q3_rows(q, expected_pair_count=2)


def test_descriptive_t_ci_and_dz_reference_values():
    values = [float(i) for i in range(1, 301)]
    stats = m.one_sample_t_statistics(values)
    expected_sd = math.sqrt(300 * 301 / 12)
    assert stats["n"] == 300
    assert stats["df"] == 299
    assert stats["mean"] == pytest.approx(150.5)
    assert stats["sample_sd"] == pytest.approx(expected_sd)
    assert stats["standard_error"] == pytest.approx(expected_sd / math.sqrt(300))
    assert stats["d_z"] == pytest.approx(stats["mean"] / stats["sample_sd"])
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]
    assert m.student_t_two_sided_p(2.0, 299) == pytest.approx(
        0.04640453988722304, rel=1e-11, abs=1e-13
    )
    assert m.student_t_critical(299, 0.95) == pytest.approx(
        1.9679296690656698, rel=1e-11, abs=1e-13
    )


def test_n_and_zero_variance_blockers():
    with pytest.raises(RuntimeError, match="cross-layer N mismatch"):
        m.one_sample_t_statistics([1.0] * 299)
    with pytest.raises(RuntimeError, match="zero-variance"):
        m.one_sample_t_statistics([1.0] * 300)


def test_global_holm_exact_six_and_ties():
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    result = m.holm_bonferroni(raw)
    assert [row["holm_adjusted_p_value"] for row in result] == pytest.approx(
        [0.006, 0.05, 0.08, 0.12, 0.4, 0.9]
    )
    assert [row["reject_holm_alpha_0_05"] for row in result] == [
        True, False, False, False, False, False
    ]
    with pytest.raises(RuntimeError, match="exactly 6"):
        m.holm_bonferroni(raw[:5])

    tied = m.holm_bonferroni([0.01, 0.01, 0.02, 0.5, 0.5, 0.9])
    assert tied[0]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert tied[1]["holm_adjusted_p_value"] == pytest.approx(0.06)


def test_analysis_uses_exact_one_six_member_holm_and_order(monkeypatch):
    phase, q = make_inputs(300)
    seen: list[int] = []
    original = m.holm_bonferroni

    def wrapped(values, *, alpha=m.FAMILYWISE_ALPHA):
        seen.append(len(values))
        return original(values, alpha=alpha)

    monkeypatch.setattr(m, "holm_bonferroni", wrapped)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    assert len(pair_rows) == 1800
    assert len(confirmatory) == 6
    assert seen == [6]
    assert [
        (row["layer_contrast"], row["endpoint"])
        for row in confirmatory
    ] == [
        (contrast, endpoint)
        for contrast, _ in m.LAYER_CONTRASTS
        for endpoint in m.ENDPOINTS
    ]


def test_decision_labels_and_family_semantics():
    rows = [{"reject_holm_alpha_0_05": False} for _ in range(6)]
    assert m.family_decision(rows) == m.FAMILY_NOT_ESTABLISHED
    rows[5]["reject_holm_alpha_0_05"] = True
    assert m.family_decision(rows) == m.FAMILY_SUPPORTED

    phase, q = make_inputs(300)
    _, confirmatory = m.analyze_rows(phase, q)
    for row in confirmatory:
        expected = m.SUPPORTED if row["reject_holm_alpha_0_05"] else m.NOT_ESTABLISHED
        assert row["decision"] == expected


def test_exact_output_fields_and_filename_order():
    assert m.PAIR_LEVEL_FIELDS == (
        "schema_version", "source_pair_id", "layer_contrast",
        "secondary_layer_index", "midpoint_layer_index", "endpoint", "estimand",
        "secondary_contrast_value", "midpoint_contrast_value", "cross_layer_difference",
    )
    assert m.CONFIRMATORY_FIELDS[-1] == "decision"
    assert m.OUTPUT_FILENAMES == (
        "name_direct_cross_layer_pair_level_differences.csv",
        "name_direct_cross_layer_confirmatory_results.csv",
        "name_direct_cross_layer_statistical_analysis_manifest.json",
        "name_direct_cross_layer_statistical_analysis_report_candidate.md",
    )


def test_deterministic_csv_json_report_and_lf():
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    pair_a = m.csv_bytes(pair_rows, m.PAIR_LEVEL_FIELDS)
    pair_b = m.csv_bytes(pair_rows, m.PAIR_LEVEL_FIELDS)
    conf_a = m.csv_bytes(confirmatory, m.CONFIRMATORY_FIELDS)
    conf_b = m.csv_bytes(confirmatory, m.CONFIRMATORY_FIELDS)
    assert pair_a == pair_b and conf_a == conf_b
    assert b"\r\n" not in pair_a and b"\r\n" not in conf_a
    assert m.canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'
    kwargs = dict(
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
        confirmatory=confirmatory,
    )
    report_a = m.render_report(**kwargs)
    report_b = m.render_report(**kwargs)
    assert report_a == report_b
    assert report_a.endswith(b"\n")
    assert b"\r\n" not in report_a
    assert b"Overall adaptive-program FWER" in report_a
    assert b"broad unqualified claim that NAME is depth-selective is prohibited" in report_a


def test_manifest_exact_posix_paths_peer_hashes_and_boundaries():
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        confirmatory=confirmatory,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    manifest = json.loads(
        outputs["name_direct_cross_layer_statistical_analysis_manifest.json"].decode("utf-8")
    )
    assert manifest["phase_f_input_path"] == m.CANONICAL_PHASE_F_PAIR_PATH
    assert manifest["q1_q3_input_path"] == m.CANONICAL_Q1_Q3_PAIR_PATH
    assert "\\" not in manifest["phase_f_input_path"]
    assert "\\" not in manifest["q1_q3_input_path"]
    assert manifest["layer_contrast_order"] == ["5_MINUS_11", "17_MINUS_11"]
    assert manifest["endpoint_order"] == list(m.ENDPOINTS)
    assert manifest["hypothesis_count"] == 6
    assert manifest["overall_adaptive_program_fwer_claimed"] is False
    assert manifest["direct_cross_layer_difference_test"] is True
    assert "manifest_sha256" not in manifest
    assert set(manifest["peer_artifact_sha256"]) == {
        "name_direct_cross_layer_pair_level_differences.csv",
        "name_direct_cross_layer_confirmatory_results.csv",
        "name_direct_cross_layer_statistical_analysis_report_candidate.md",
    }
    for name, digest in manifest["peer_artifact_sha256"].items():
        assert digest == m.sha256_bytes(outputs[name])


def test_report_uses_exact_posix_paths_and_specific_inference_boundary():
    phase, q = make_inputs(300)
    _, confirmatory = m.analyze_rows(phase, q)
    report = m.render_report(
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
        confirmatory=confirmatory,
    ).decode("utf-8")
    assert f"- Phase F input: `{m.CANONICAL_PHASE_F_PAIR_PATH}`" in report
    assert f"- Q1/Q3 input: `{m.CANONICAL_Q1_Q3_PAIR_PATH}`" in report
    assert "exact prespecified between-layer" in report
    assert "depth-selective is prohibited" in report


def test_write_output_exact_four_atomic_and_collision(tmp_path: Path):
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        confirmatory=confirmatory,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    out = tmp_path / "result"
    hashes = m.write_output_bytes(out, outputs)
    assert set(path.name for path in out.iterdir()) == set(m.OUTPUT_FILENAMES)
    assert set(hashes) == set(m.OUTPUT_FILENAMES)
    assert not (tmp_path / "result.staging").exists()
    with pytest.raises(RuntimeError, match="already exists"):
        m.write_output_bytes(out, outputs)


def test_preexisting_staging_collision_blocker(tmp_path: Path):
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        confirmatory=confirmatory,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    (tmp_path / "result.staging").mkdir()
    with pytest.raises(RuntimeError, match="staging directory already exists"):
        m.write_output_bytes(tmp_path / "result", outputs)


def test_exact_file_identity_validation_synthetic_only(tmp_path: Path):
    path = tmp_path / "synthetic.csv"
    payload = b"a,b\n1,2\n"
    path.write_bytes(payload)
    assert m._validate_exact_file(
        path,
        expected_sha256=m.sha256_bytes(payload),
        expected_bytes=len(payload),
    ) == path
    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        m._validate_exact_file(
            path,
            expected_sha256=m.sha256_bytes(payload),
            expected_bytes=len(payload) + 1,
        )


def test_load_pair_csv_exact_header_synthetic_only(tmp_path: Path):
    path = tmp_path / "phase.csv"
    rows = make_phase_rows(1)
    path.write_bytes(m.csv_bytes(rows, m.PHASE_F_FIELDS))
    loaded = m.load_pair_csv(path, expected_fields=m.PHASE_F_FIELDS)
    assert loaded == rows

    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="header field contract mismatch"):
        m.load_pair_csv(bad, expected_fields=m.PHASE_F_FIELDS)


def test_synthetic_suite_does_not_open_either_canonical_artifact(monkeypatch):
    canonical_suffixes = {
        m.CANONICAL_PHASE_F_PAIR_PATH,
        m.CANONICAL_Q1_Q3_PAIR_PATH,
    }
    original_open = Path.open
    original_read_bytes = Path.read_bytes

    def guard_path(path: Path):
        normalized = str(path).replace("\\", "/")
        if any(normalized.endswith(suffix) for suffix in canonical_suffixes):
            raise AssertionError("canonical pair artifact opened during synthetic tests")

    def guarded_open(self, *args, **kwargs):
        guard_path(self)
        return original_open(self, *args, **kwargs)

    def guarded_read_bytes(self, *args, **kwargs):
        guard_path(self)
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        confirmatory=confirmatory,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    assert len(pair_rows) == 1800
    assert len(confirmatory) == 6
    assert tuple(outputs) == m.OUTPUT_FILENAMES


def test_run_canonical_serializes_frozen_posix_paths_without_opening_canonical(
    monkeypatch,
    tmp_path: Path,
):
    phase, q = make_inputs(300)
    native_phase = Path(r"Z:\host-native\phase\pair.csv")
    native_q = Path(r"Z:\host-native\q1q3\pair.csv")
    captured: dict[str, object] = {}

    def fake_validate_exact_file(path, *, expected_sha256, expected_bytes):
        normalized = str(path).replace("\\", "/")
        if normalized == m.CANONICAL_PHASE_F_PAIR_PATH:
            assert expected_sha256 == m.CANONICAL_PHASE_F_PAIR_SHA256
            assert expected_bytes == m.CANONICAL_PHASE_F_PAIR_BYTES
            return native_phase
        if normalized == m.CANONICAL_Q1_Q3_PAIR_PATH:
            assert expected_sha256 == m.CANONICAL_Q1_Q3_PAIR_SHA256
            assert expected_bytes == m.CANONICAL_Q1_Q3_PAIR_BYTES
            return native_q
        raise AssertionError("unexpected path")

    def fake_load_pair_csv(path, *, expected_fields):
        if path == native_phase:
            assert tuple(expected_fields) == m.PHASE_F_FIELDS
            return phase
        if path == native_q:
            assert tuple(expected_fields) == m.Q1_Q3_FIELDS
            return q
        raise AssertionError("unexpected loaded path")

    def fake_write_output_bytes(output_dir, outputs):
        captured["outputs"] = dict(outputs)
        return {name: m.sha256_bytes(data) for name, data in outputs.items()}

    monkeypatch.setattr(m, "_validate_exact_file", fake_validate_exact_file)
    monkeypatch.setattr(m, "load_pair_csv", fake_load_pair_csv)
    monkeypatch.setattr(m, "write_output_bytes", fake_write_output_bytes)

    result = m.run_canonical(
        phase_f_pair_csv=m.CANONICAL_PHASE_F_PAIR_PATH,
        q1_q3_pair_csv=m.CANONICAL_Q1_Q3_PAIR_PATH,
        output_dir=tmp_path / "synthetic-result",
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    outputs = captured["outputs"]
    assert isinstance(outputs, dict)
    manifest = json.loads(
        outputs["name_direct_cross_layer_statistical_analysis_manifest.json"].decode("utf-8")
    )
    report = outputs[
        "name_direct_cross_layer_statistical_analysis_report_candidate.md"
    ].decode("utf-8")
    assert manifest["phase_f_input_path"] == m.CANONICAL_PHASE_F_PAIR_PATH
    assert manifest["q1_q3_input_path"] == m.CANONICAL_Q1_Q3_PAIR_PATH
    assert str(native_phase) not in report
    assert str(native_q) not in report
    assert result["hypothesis_count"] == 6
    assert result["source_pair_count"] == 300
    assert result["direct_cross_layer_difference_test"] is True


def test_run_canonical_rejects_noncanonical_logical_paths():
    with pytest.raises(RuntimeError, match="canonical logical path mismatch"):
        m.run_canonical(
            phase_f_pair_csv="wrong.csv",
            q1_q3_pair_csv=m.CANONICAL_Q1_Q3_PAIR_PATH,
            output_dir="unused",
            implementation_commit="b" * 40,
            script_sha256="c" * 64,
            execution_authority_commit="d" * 40,
        )


def test_no_17_minus_5_in_contract_or_outputs():
    assert all(name != "17_MINUS_5" for name, _ in m.LAYER_CONTRASTS)
    phase, q = make_inputs(300)
    pair_rows, confirmatory = m.analyze_rows(phase, q)
    assert {row["layer_contrast"] for row in pair_rows} == {"5_MINUS_11", "17_MINUS_11"}
    assert {row["layer_contrast"] for row in confirmatory} == {"5_MINUS_11", "17_MINUS_11"}
