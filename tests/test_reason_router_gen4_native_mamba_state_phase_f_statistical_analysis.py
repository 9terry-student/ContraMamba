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
    / "reason_router_gen4_native_mamba_state_phase_f_statistical_analysis.py"
)

spec = importlib.util.spec_from_file_location("phase_f_stats", MODULE_PATH)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_pair_rows(pair_id: str, i: int) -> list[dict]:
    rows = []
    endpoint_scales = {
        "POST4_SPEED": 1.0,
        "POST4_TURNING": 2.0,
        "POST4_PATH_EFFICIENCY": 3.0,
    }

    for anchor, cells in m.ANCHOR_CELLS.items():
        anchor_offset = {
            "A_TITLE": 10.0,
            "A_NAME": 20.0,
            "A_ROLE": 30.0,
            "A_PREDICATE": 40.0,
            "A_IDENTITY": 50.0,
        }[anchor]

        for cell in cells:
            row = {
                "source_pair_id": pair_id,
                "anchor_name": anchor,
                "contrast_cell_id": cell,
                "layer_index": 11,
            }

            for endpoint, scale in endpoint_scales.items():
                base = anchor_offset + 0.01 * i + 0.001 * scale

                if anchor == "A_TITLE":
                    effect = 0.0 if cell == "C0_SHAM" else scale * (0.10 + 0.0001 * i)
                elif anchor == "A_NAME":
                    effect = 0.0 if cell == "C0_SHAM" else scale * (-0.20 + 0.0002 * i)
                elif anchor == "A_ROLE":
                    effect = 0.0 if cell == "C0_SHAM" else scale * (0.30 + 0.0003 * i)
                elif anchor == "A_PREDICATE":
                    effect = 0.0 if cell == "C0_SHAM" else scale * (-0.40 + 0.0004 * i)
                else:
                    title = scale * (0.50 + 0.0005 * i)
                    name = scale * (-0.60 + 0.0006 * i)
                    interaction = scale * (0.07 + 0.0007 * i)
                    effect = {
                        "C0_SHAM": 0.0,
                        "C1_TITLE": title,
                        "C2_NAME": name,
                        "C5_TITLE_NAME": title + name + interaction,
                    }[cell]

                row[endpoint] = base + effect

            rows.append(row)

    assert len(rows) == 12
    return rows


def make_records(n: int = 300) -> list[dict]:
    rows = []
    for i in range(n):
        rows.extend(make_pair_rows(f"pair-{i:03d}", i))
    return rows


def test_exact_matrix_acceptance_small():
    rows = make_records(2)
    m.validate_endpoint_records(rows, expected_pair_count=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "row count mismatch"),
        ("missing", "row count mismatch"),
        ("unexpected_anchor", "unexpected anchor"),
        ("unexpected_cell", "unexpected anchor/cell"),
        ("wrong_layer", "layer_index mismatch"),
        ("nan", "must be finite"),
        ("inf", "must be finite"),
        ("boolean", "numeric and non-boolean"),
    ],
)
def test_matrix_fail_closed(mutation: str, message: str):
    rows = make_records(2)

    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unexpected_anchor":
        rows[0]["anchor_name"] = "A_OTHER"
    elif mutation == "unexpected_cell":
        rows[0]["contrast_cell_id"] = "C4_PREDICATE"
    elif mutation == "wrong_layer":
        rows[0]["layer_index"] = 10
    elif mutation == "nan":
        rows[0]["POST4_SPEED"] = float("nan")
    elif mutation == "inf":
        rows[0]["POST4_TURNING"] = float("inf")
    elif mutation == "boolean":
        rows[0]["POST4_PATH_EFFICIENCY"] = True

    with pytest.raises(RuntimeError, match=message):
        m.validate_endpoint_records(rows, expected_pair_count=2)


def test_all_five_contrast_formulas_across_all_three_endpoints():
    rows = make_pair_rows("pair-z", 10)
    pair_rows = m.construct_pair_level_contrasts(rows, expected_pair_count=1)

    observed = {
        (row["endpoint"], row["estimand"]): row["contrast_value"]
        for row in pair_rows
    }

    for endpoint, scale in (
        ("POST4_SPEED", 1.0),
        ("POST4_TURNING", 2.0),
        ("POST4_PATH_EFFICIENCY", 3.0),
    ):
        expected = {
            "DELTA_TITLE": scale * (0.10 + 0.0001 * 10),
            "DELTA_NAME": scale * (-0.20 + 0.0002 * 10),
            "DELTA_ROLE": scale * (0.30 + 0.0003 * 10),
            "DELTA_PREDICATE": scale * (-0.40 + 0.0004 * 10),
            "INTERACTION_TITLE_NAME": scale * (0.07 + 0.0007 * 10),
        }
        for estimand, value in expected.items():
            assert observed[(endpoint, estimand)] == pytest.approx(value, abs=1e-12)


def test_300_pair_cardinality_and_deterministic_order():
    rows = make_records(300)
    pair_rows = m.construct_pair_level_contrasts(rows)

    assert len(pair_rows) == 4500
    assert pair_rows[0]["endpoint"] == "POST4_SPEED"
    assert pair_rows[0]["estimand"] == "DELTA_TITLE"
    assert pair_rows[0]["source_pair_id"] == "pair-000"
    assert pair_rows[299]["source_pair_id"] == "pair-299"
    assert pair_rows[300]["estimand"] == "DELTA_NAME"
    assert pair_rows[-1]["endpoint"] == "POST4_PATH_EFFICIENCY"
    assert pair_rows[-1]["estimand"] == "INTERACTION_TITLE_NAME"
    assert pair_rows[-1]["source_pair_id"] == "pair-299"


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


def test_student_t_reference_values_df299():
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


def test_primary_n_enforcement():
    with pytest.raises(RuntimeError, match="primary N mismatch"):
        m.one_sample_t_statistics([1.0] * 299)


def test_zero_variance_is_hard_blocker():
    with pytest.raises(RuntimeError, match="zero-variance"):
        m.one_sample_t_statistics([1.0] * 300)


def test_global_holm_exact_15_and_step_down():
    raw = [
        0.0001, 0.001, 0.005, 0.01, 0.02,
        0.03, 0.04, 0.05, 0.06, 0.07,
        0.08, 0.09, 0.1, 0.2, 0.9,
    ]
    result = m.holm_bonferroni(raw)

    expected_adjusted = [
        0.0015, 0.014, 0.065, 0.12, 0.22,
        0.3, 0.36, 0.4, 0.42, 0.42,
        0.42, 0.42, 0.42, 0.42, 0.9,
    ]

    assert [x["holm_adjusted_p_value"] for x in result] == pytest.approx(
        expected_adjusted
    )
    assert [x["reject_holm_alpha_0_05"] for x in result] == [
        True, True, False, False, False,
        False, False, False, False, False,
        False, False, False, False, False,
    ]

    with pytest.raises(RuntimeError, match="exactly 15"):
        m.holm_bonferroni(raw[:5])


def test_analysis_is_global_15_not_endpoint_wise(monkeypatch):
    rows = make_records(300)
    seen = {}
    original = m.holm_bonferroni

    def wrapped(raw_p_values, *, alpha=m.FAMILYWISE_ALPHA):
        seen["count"] = len(raw_p_values)
        return original(raw_p_values, alpha=alpha)

    monkeypatch.setattr(m, "holm_bonferroni", wrapped)
    pair_rows, primary = m.analyze_records(rows)

    assert len(pair_rows) == 4500
    assert len(primary) == 15
    assert seen["count"] == 15
    assert [
        (row["endpoint"], row["estimand"]) for row in primary
    ] == [
        (endpoint, estimand)
        for endpoint in m.ENDPOINTS
        for estimand in m.ESTIMANDS
    ]


def test_deterministic_csv_json_and_report_bytes():
    rows = make_records(300)
    pair_rows, primary = m.analyze_records(rows)

    pair_a = m.csv_bytes(pair_rows, m.PAIR_FIELDS)
    pair_b = m.csv_bytes(pair_rows, m.PAIR_FIELDS)
    primary_a = m.csv_bytes(primary, m.PRIMARY_FIELDS)
    primary_b = m.csv_bytes(primary, m.PRIMARY_FIELDS)

    assert pair_a == pair_b
    assert primary_a == primary_b
    assert b"\r\n" not in pair_a
    assert b"\r\n" not in primary_a

    assert m.canonical_json_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'

    kwargs = dict(
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
        primary=primary,
    )
    report_a = m.render_report(**kwargs)
    report_b = m.render_report(**kwargs)
    assert report_a == report_b
    assert report_a.endswith(b"\n")
    assert b"\r\n" not in report_a


def test_manifest_binds_three_peers_but_not_self():
    rows = make_records(300)
    pair_rows, primary = m.analyze_records(rows)

    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        primary=primary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=3600,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )

    assert tuple(outputs) == m.OUTPUT_FILENAMES
    manifest = json.loads(
        outputs["phase_f_statistical_analysis_manifest.json"].decode("utf-8")
    )

    assert set(manifest["peer_artifact_sha256"]) == {
        "phase_f_pair_level_contrasts.csv",
        "phase_f_primary_confirmatory_results.csv",
        "phase_f_statistical_analysis_report_candidate.md",
    }
    assert "manifest_sha256" not in manifest
    assert (
        manifest["peer_artifact_sha256"]["phase_f_pair_level_contrasts.csv"]
        == m.sha256_bytes(outputs["phase_f_pair_level_contrasts.csv"])
    )


def test_write_output_bytes_exact_four_files(tmp_path: Path):
    rows = make_records(300)
    pair_rows, primary = m.analyze_records(rows)
    outputs = m.build_output_bytes(
        pair_rows=pair_rows,
        primary=primary,
        input_path="synthetic.jsonl",
        input_sha256="a" * 64,
        input_bytes=123,
        input_rows=3600,
        implementation_commit="b" * 40,
        script_sha256="c" * 64,
        execution_authority_commit="d" * 40,
    )

    out = tmp_path / "result"
    hashes = m.write_output_bytes(out, outputs)

    assert sorted(p.name for p in out.iterdir()) == sorted(m.OUTPUT_FILENAMES)
    assert set(hashes) == set(m.OUTPUT_FILENAMES)


def test_synthetic_tests_do_not_open_canonical_input(monkeypatch):
    canonical = str(Path(m.CANONICAL_INPUT_PATH))
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if str(self).replace("\\", "/").endswith(canonical.replace("\\", "/")):
            raise AssertionError("canonical input opened during synthetic tests")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    rows = make_records(300)
    pair_rows, primary = m.analyze_records(rows)
    assert len(pair_rows) == 4500
    assert len(primary) == 15
