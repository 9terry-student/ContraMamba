from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_native_mamba_state_name_q1_q3_statistical_analysis.py"
)
PHASE_F_PATH = (
    ROOT
    / "scripts"
    / "reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py"
)

spec = importlib.util.spec_from_file_location("q1q3_stats", MODULE_PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

phase_f_spec = importlib.util.spec_from_file_location("phase_f_stats_reference", PHASE_F_PATH)
phase_f = None
if phase_f_spec is not None and phase_f_spec.loader is not None and PHASE_F_PATH.is_file():
    phase_f = importlib.util.module_from_spec(phase_f_spec)
    phase_f_spec.loader.exec_module(phase_f)


def make_pair_rows(pair_id: str, i: int) -> list[dict]:
    rows: list[dict] = []
    scales = {
        "POST4_SPEED": 1.0,
        "POST4_TURNING": 2.0,
        "POST4_PATH_EFFICIENCY": 3.0,
    }

    for layer in m.LAYERS:
        for cell in ("C0_SHAM", "C2_NAME"):
            row = {
                "source_pair_id": pair_id,
                "row_id": f"{pair_id}-{cell}",
                "contrast_cell_id": cell,
                "semantic_anchor": "A_NAME",
                "anchor_token_index": 40,
                "layer_index": layer,
                "support_absolute_token_indices": [39, 40, 41, 42, 43, 44],
                "support_tensor_row_indices": [0, 1, 2, 3, 4, 5],
            }
            for endpoint, scale in scales.items():
                base = layer * 10.0 + 0.01 * i + 0.001 * scale
                if cell == "C0_SHAM":
                    effect = 0.0
                elif layer == 5:
                    effect = scale * (0.10 + 0.0001 * i)
                else:
                    effect = scale * (0.20 + 0.0002 * i)
                row[endpoint] = base + effect
            rows.append(row)

    assert len(rows) == 4
    return rows


def make_records(n: int = 300) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        rows.extend(make_pair_rows(f"pair-{i:03d}", i))
    return rows


def test_constants_and_exact_contract():
    assert m.LAYERS == (5, 17)
    assert m.ENDPOINTS == (
        "POST4_SPEED",
        "POST4_TURNING",
        "POST4_PATH_EFFICIENCY",
    )
    assert m.ESTIMAND == "DELTA_NAME"
    assert m.SECONDARY_HYPOTHESIS_COUNT == 6
    assert m.SOURCE_PAIR_COUNT == 300
    assert m.ROWS_PER_PAIR == 4
    assert m.CANONICAL_INPUT_ROWS == 1200
    assert m.CANONICAL_INPUT_BYTES == 513396


def test_exact_four_row_matrix_acceptance():
    rows = make_records(2)
    m.validate_endpoint_records(rows, expected_pair_count=2)


def test_cross_layer_row_id_reuse_is_valid():
    rows = make_pair_rows("pair-reuse", 7)

    for cell in ("C0_SHAM", "C2_NAME"):
        cell_rows = [
            row for row in rows
            if row["contrast_cell_id"] == cell
        ]
        assert len(cell_rows) == 2
        assert {row["layer_index"] for row in cell_rows} == {5, 17}
        assert len({row["row_id"] for row in cell_rows}) == 1

    m.validate_endpoint_records(rows, expected_pair_count=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("non_object", "non-object"),
        ("missing_field", "required fields missing"),
        ("empty_pair", "empty source_pair_id"),
        ("empty_row", "empty row_id"),
        ("duplicate_row_layer", "duplicate row/layer identity"),
        ("duplicate_identity", "duplicate pair/layer/cell"),
        ("wrong_anchor", "unexpected semantic_anchor"),
        ("extra_layer", "unexpected layer_index"),
        ("extra_cell", "unexpected contrast_cell_id"),
        ("nan", "must be finite"),
        ("inf", "must be finite"),
        ("boolean", "numeric and non-boolean"),
    ],
)
def test_row_validation_fail_closed(mutation: str, message: str):
    rows = make_records(2)

    if mutation == "non_object":
        rows[0] = 123
    elif mutation == "missing_field":
        rows[0].pop("POST4_SPEED")
    elif mutation == "empty_pair":
        rows[0]["source_pair_id"] = ""
    elif mutation == "empty_row":
        rows[0]["row_id"] = ""
    elif mutation == "duplicate_row_layer":
        rows[1]["row_id"] = rows[0]["row_id"]
    elif mutation == "duplicate_identity":
        rows[1]["layer_index"] = rows[0]["layer_index"]
        rows[1]["contrast_cell_id"] = rows[0]["contrast_cell_id"]
    elif mutation == "wrong_anchor":
        rows[0]["semantic_anchor"] = "A_TITLE"
    elif mutation == "extra_layer":
        rows[0]["layer_index"] = 11
    elif mutation == "extra_cell":
        rows[0]["contrast_cell_id"] = "C1_TITLE"
    elif mutation == "nan":
        rows[0]["POST4_SPEED"] = float("nan")
    elif mutation == "inf":
        rows[0]["POST4_TURNING"] = float("inf")
    elif mutation == "boolean":
        rows[0]["POST4_PATH_EFFICIENCY"] = True

    with pytest.raises(RuntimeError, match=message):
        m.validate_endpoint_records(rows, expected_pair_count=2)


def test_wrong_total_row_count_and_missing_pair_identity_fail():
    rows = make_records(2)
    with pytest.raises(RuntimeError, match="row count mismatch"):
        m.validate_endpoint_records(rows[:-1], expected_pair_count=2)

    rows = make_records(2)
    rows[3]["source_pair_id"] = "pair-002"
    rows[3]["row_id"] = "moved-unique-row"
    with pytest.raises(RuntimeError, match="source-pair count mismatch"):
        m.validate_endpoint_records(rows, expected_pair_count=2)


def test_all_three_endpoints_both_layers_delta_name_formula():
    rows = make_pair_rows("pair-z", 10)
    pair_rows = m.construct_pair_level_contrasts(rows, expected_pair_count=1)
    observed = {
        (row["layer_index"], row["endpoint"]): row["contrast_value"]
        for row in pair_rows
    }

    for endpoint, scale in (
        ("POST4_SPEED", 1.0),
        ("POST4_TURNING", 2.0),
        ("POST4_PATH_EFFICIENCY", 3.0),
    ):
        assert observed[(5, endpoint)] == pytest.approx(
            scale * (0.10 + 0.0001 * 10),
            abs=1e-12,
        )
        assert observed[(17, endpoint)] == pytest.approx(
            scale * (0.20 + 0.0002 * 10),
            abs=1e-12,
        )


def test_300_pair_cardinality_and_layer_endpoint_pair_order():
    rows = make_records()
    pair_rows = m.construct_pair_level_contrasts(rows)

    assert len(pair_rows) == 1800
    assert pair_rows[0]["layer_index"] == 5
    assert pair_rows[0]["endpoint"] == "POST4_SPEED"
    assert pair_rows[0]["source_pair_id"] == "pair-000"
    assert pair_rows[299]["source_pair_id"] == "pair-299"
    assert pair_rows[300]["layer_index"] == 5
    assert pair_rows[300]["endpoint"] == "POST4_TURNING"
    assert pair_rows[900]["layer_index"] == 17
    assert pair_rows[900]["endpoint"] == "POST4_SPEED"
    assert pair_rows[-1]["layer_index"] == 17
    assert pair_rows[-1]["endpoint"] == "POST4_PATH_EFFICIENCY"
    assert pair_rows[-1]["source_pair_id"] == "pair-299"
    assert all(row["estimand"] == "DELTA_NAME" for row in pair_rows)


def test_descriptive_and_one_sample_statistics():
    values = [float(i) for i in range(1, 301)]
    stats = m.one_sample_t_statistics(values)

    assert stats["n"] == 300
    assert stats["df"] == 299
    assert stats["mean"] == pytest.approx(150.5)
    assert stats["median"] == pytest.approx(150.5)
    assert stats["minimum"] == 1.0
    assert stats["maximum"] == 300.0

    expected_sd = math.sqrt(300 * 301 / 12)
    assert stats["sample_sd"] == pytest.approx(expected_sd)
    assert stats["standard_error"] == pytest.approx(expected_sd / math.sqrt(300))
    assert stats["t_statistic"] == pytest.approx(
        stats["mean"] / stats["standard_error"]
    )
    assert stats["d_z"] == pytest.approx(stats["mean"] / stats["sample_sd"])
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_student_t_reference_values_and_phase_f_agreement():
    assert m.student_t_two_sided_p(2.0, 299) == pytest.approx(
        0.04640453988722304,
        rel=1e-11,
        abs=1e-13,
    )
    assert m.student_t_critical(299, 0.95) == pytest.approx(
        1.9679296690656698,
        rel=1e-11,
        abs=1e-13,
    )

    if phase_f is not None:
        for t in (0.0, 0.5, 2.0, 8.0):
            assert m.student_t_two_sided_p(t, 299) == pytest.approx(
                phase_f.student_t_two_sided_p(t, 299),
                rel=1e-15,
                abs=1e-15,
            )
        assert m.student_t_critical(299, 0.95) == pytest.approx(
            phase_f.student_t_critical(299, 0.95),
            rel=1e-15,
            abs=1e-15,
        )


def test_n_enforcement_and_zero_variance_blocker():
    with pytest.raises(RuntimeError, match="secondary N mismatch"):
        m.one_sample_t_statistics([1.0] * 299)
    with pytest.raises(RuntimeError, match="zero-variance"):
        m.one_sample_t_statistics([1.0] * 300)


def test_global_holm_exact_six_step_down_and_threshold():
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    result = m.holm_bonferroni(raw)
    assert [row["holm_adjusted_p_value"] for row in result] == pytest.approx(
        [0.006, 0.05, 0.08, 0.12, 0.4, 0.9]
    )
    assert [row["reject_holm_alpha_0_05"] for row in result] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    with pytest.raises(RuntimeError, match="exactly 6"):
        m.holm_bonferroni(raw[:5])


def test_holm_tie_handling_and_monotonic_adjusted_values():
    raw = [0.01, 0.01, 0.02, 0.5, 0.5, 0.9]
    result = m.holm_bonferroni(raw)
    ordered = sorted(
        enumerate(result),
        key=lambda item: (item[1]["raw_p_value"], item[0]),
    )
    adjusted = [item[1]["holm_adjusted_p_value"] for item in ordered]
    assert adjusted == sorted(adjusted)
    assert result[0]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert result[1]["holm_adjusted_p_value"] == pytest.approx(0.06)


def test_analysis_uses_one_global_six_member_holm(monkeypatch):
    rows = make_records()
    seen: list[int] = []
    original = m.holm_bonferroni

    def wrapped(raw_p_values, *, alpha=m.FAMILYWISE_ALPHA):
        seen.append(len(raw_p_values))
        return original(raw_p_values, alpha=alpha)

    monkeypatch.setattr(m, "holm_bonferroni", wrapped)
    pair_rows, secondary = m.analyze_records(rows)

    assert len(pair_rows) == 1800
    assert len(secondary) == 6
    assert seen == [6]
    assert [
        (row["layer_index"], row["endpoint"])
        for row in secondary
    ] == [
        (layer, endpoint)
        for layer in m.LAYERS
        for endpoint in m.ENDPOINTS
    ]


def test_supported_and_not_established_labels_and_family_semantics():
    rows = [
        {"reject_holm_alpha_0_05": False}
        for _ in range(m.SECONDARY_HYPOTHESIS_COUNT)
    ]
    assert m.family_decision(rows) == m.FAMILY_NOT_ESTABLISHED

    rows[4]["reject_holm_alpha_0_05"] = True
    assert m.family_decision(rows) == m.FAMILY_SUPPORTED

    records = make_records()
    _, secondary = m.analyze_records(records)
    for row in secondary:
        if row["reject_holm_alpha_0_05"]:
            assert row["decision"] == m.SUPPORTED
        else:
            assert row["decision"] == m.NOT_ESTABLISHED


def test_deterministic_csv_json_report_bytes_and_line_endings():
    records = make_records()
    pair_rows, secondary = m.analyze_records(records)

    pair_a = m.csv_bytes(pair_rows, m.PAIR_FIELDS)
    pair_b = m.csv_bytes(pair_rows, m.PAIR_FIELDS)
    secondary_a = m.csv_bytes(secondary, m.SECONDARY_FIELDS)
    secondary_b = m.csv_bytes(secondary, m.SECONDARY_FIELDS)

    assert pair_a == pair_b
    assert secondary_a == secondary_b
    assert b"\r\n" not in pair_a
    assert b"\r\n" not in secondary_a

    encoded = m.canonical_json_bytes({"z": 1, "a": 2})
    assert encoded == b'{"a":2,"z":1}\n'
    assert encoded.endswith(b"\n")
    assert not encoded.endswith(b"\n\n")

    kwargs = dict(
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
        secondary=secondary,
    )
    report_a = m.render_report(**kwargs)
    report_b = m.render_report(**kwargs)
    assert report_a == report_b
    assert report_a.endswith(b"\n")
    assert b"\r\n" not in report_a
    assert b"direct layer-to-layer difference test" in report_a


def test_manifest_binds_three_peers_without_self_and_preserves_boundaries():
    records = make_records()
    pair_rows, secondary = m.analyze_records(records)

    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        secondary=secondary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=1200,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    assert tuple(outputs) == m.OUTPUT_FILENAMES

    manifest = json.loads(
        outputs["name_q1_q3_statistical_analysis_manifest.json"].decode("utf-8")
    )
    assert set(manifest["peer_artifact_sha256"]) == {
        "name_q1_q3_pair_level_contrasts.csv",
        "name_q1_q3_secondary_confirmatory_results.csv",
        "name_q1_q3_statistical_analysis_report_candidate.md",
    }
    assert "manifest_sha256" not in manifest
    assert manifest["hypothesis_count"] == 6
    assert manifest["layer_order"] == [5, 17]
    assert manifest["endpoint_order"] == list(m.ENDPOINTS)
    assert manifest["structural_estimand"] == "DELTA_NAME"
    assert manifest["overall_adaptive_program_fwer_claimed"] is False
    assert manifest["cross_layer_difference_test"] is False
    assert manifest["peer_artifact_sha256"][
        "name_q1_q3_pair_level_contrasts.csv"
    ] == m.sha256_bytes(outputs["name_q1_q3_pair_level_contrasts.csv"])


def test_write_output_bytes_exact_four_atomic_publication_and_collision(tmp_path: Path):
    records = make_records()
    pair_rows, secondary = m.analyze_records(records)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        secondary=secondary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=1200,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )

    out = tmp_path / "result"
    hashes = m.write_output_bytes(out, outputs)
    assert tuple(sorted(path.name for path in out.iterdir())) == tuple(
        sorted(m.OUTPUT_FILENAMES)
    )
    assert set(hashes) == set(m.OUTPUT_FILENAMES)
    assert not (tmp_path / "result.staging").exists()

    with pytest.raises(RuntimeError, match="already exists"):
        m.write_output_bytes(out, outputs)


def test_preexisting_staging_directory_is_hard_blocker(tmp_path: Path):
    records = make_records()
    pair_rows, secondary = m.analyze_records(records)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        secondary=secondary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=1200,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    staging = tmp_path / "result.staging"
    staging.mkdir()

    with pytest.raises(RuntimeError, match="staging directory already exists"):
        m.write_output_bytes(tmp_path / "result", outputs)


def test_exact_file_identity_validation_on_synthetic_file(tmp_path: Path):
    path = tmp_path / "synthetic.jsonl"
    payload = b'{"x":1}\n'
    path.write_bytes(payload)

    observed = m._validate_exact_file(
        path,
        expected_sha256=m.sha256_bytes(payload),
        expected_bytes=len(payload),
    )
    assert observed == path

    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        m._validate_exact_file(
            path,
            expected_sha256=m.sha256_bytes(payload),
            expected_bytes=len(payload) + 1,
        )


def test_synthetic_analysis_does_not_open_canonical_input(monkeypatch):
    canonical = m.CANONICAL_INPUT_PATH.replace("\\", "/")
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        normalized = str(self).replace("\\", "/")
        if normalized.endswith(canonical):
            raise AssertionError("canonical input opened during synthetic tests")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    records = make_records()
    pair_rows, secondary = m.analyze_records(records)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        secondary=secondary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=1200,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )
    assert len(pair_rows) == 1800
    assert len(secondary) == 6
    assert tuple(outputs) == m.OUTPUT_FILENAMES


def test_output_filename_order_and_fixed_columns():
    assert m.OUTPUT_FILENAMES == (
        "name_q1_q3_pair_level_contrasts.csv",
        "name_q1_q3_secondary_confirmatory_results.csv",
        "name_q1_q3_statistical_analysis_manifest.json",
        "name_q1_q3_statistical_analysis_report_candidate.md",
    )
    assert m.PAIR_FIELDS == (
        "schema_version",
        "source_pair_id",
        "layer_index",
        "endpoint",
        "estimand",
        "contrast_value",
    )
    assert m.SECONDARY_FIELDS[-1] == "decision"
