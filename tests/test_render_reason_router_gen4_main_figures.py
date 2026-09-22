"""Focused static contracts; never import scientific runners or inference code."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import render_reason_router_gen4_main_figures as subject


@pytest.fixture(scope="module")
def sources():
    return subject.Sources()


@pytest.mark.parametrize("field,value,match", [
    ("schema_version", "future-schema", "schema_version"),
    ("scientific_execution", "OPEN", "CLOSED"),
    ("new_p_values", True, "new_p_values"),
    ("new_statistical_tests", True, "new_statistical_tests"),
    ("new_model_execution", True, "new_model_execution"),
])
def test_manifest_schema_and_closed_execution(sources, field, value, match):
    manifest = copy.deepcopy(sources.manifest)
    manifest[field] = value
    with pytest.raises(subject.ContractError, match=match):
        subject.validate_manifest(manifest)


def test_manifest_rebinding_fails_closed(sources):
    manifest = copy.deepcopy(sources.manifest)
    manifest["figures"][1]["panels"][0]["sources"][0] = "alternative.jsonl"
    with pytest.raises(subject.ContractError, match="bindings"):
        subject.validate_manifest(manifest)


def test_missing_bound_source(tmp_path):
    with pytest.raises(subject.ContractError, match="Missing bound source"):
        subject.Sources(root=tmp_path)


def test_all_frozen_sources_extract_without_inference(sources):
    data = subject.prepare(sources)
    assert len(sources.raw) == 22
    assert len(data["xg1"]) == len(data["readout"]) == len(data["transport"]) == 300
    assert data["core_means"] == [3.974290010502882e-08, 9.282848764823318e-09]
    assert data["residual"][0]["ranks"] == ["P1", "P2", "P4", "P5"]
    assert data["residual"][1]["ranks"] == ["P1", "P2", "P3", "P4"]
    assert data["residual"][1]["mass"][1] == 0.5744614649420721
    assert data["specificity"]["mean D_SPEC"] == 2.3040673691981465e-08
    assert data["necessity"]["mean"] == 4.7414371121837106e-08
    assert data["restoration"]["mean"] == 4.078872356598753e-08
    assert data["angles"] == [86.44753153112424, 89.22379912052105]
    assert data["table"][2]["conclusion"] == "Negative transfer not established"
    assert data["table"][3]["frozen_p"] == "not estimable"
    assert data["table"][3]["primary_outcome"] == "corrections=0; damages=0"


def test_static_validation_leaves_all_sources_byte_identical(sources):
    before = {name: path.read_bytes() for name, path in sources.paths.items()}
    manifest_before = sources.manifest_path.read_bytes()
    subject.prepare(sources)
    for name, path in sources.paths.items():
        assert path.read_bytes() == before[name], name
    assert sources.manifest_path.read_bytes() == manifest_before
    sources.verify_unchanged()


@pytest.mark.parametrize("figure,panel,validator", [
    (5, "B", subject.validate_readout), (4, "C", subject.validate_transport),
])
@pytest.mark.parametrize("damage,match", [
    ("missing_field", "Required source field"),
    ("short", "N=300"), ("long", "N=300"),
    ("duplicate", "Duplicate|identity mismatch"),
    ("nonfinite", "Nonfinite"),
    ("wrong_identity", "unexpected|identity mismatch"),
])
def test_pair_schemas_and_frozen_population(sources, figure, panel, validator, damage, match):
    rows = sources.panel(figure, panel)
    analysis = sources.panel(figure, panel, 1)
    field = "R" if figure == 5 else "D_CAN_descriptive_only"
    if damage == "missing_field":
        del rows[0][field]
    elif damage == "short":
        rows.pop()
    elif damage == "long":
        rows.append(copy.deepcopy(rows[0]))
    elif damage == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif damage == "nonfinite":
        rows[0][field] = float("nan")
    else:
        rows[0]["source_pair_id"] = "foreign_cohort_0"
    with pytest.raises(subject.ContractError, match=match):
        validator(rows, analysis)


@pytest.mark.parametrize("field", ["Delta_L_370M", "Delta_L_1.4B", "R"])
def test_readout_frozen_mean_consistency(sources, field):
    rows = sources.panel(5, "B")
    rows[0][field] += 1e-6
    with pytest.raises(subject.ContractError, match="arithmetic mismatch"):
        subject.validate_readout(rows, sources.panel(5, "B", 1))


@pytest.mark.parametrize("field", ["mean_R", "sd_R", "t_statistic", "p_value"])
def test_readout_analysis_cannot_change_frozen_values(sources, field):
    analysis = sources.panel(5, "B", 1)
    analysis["primary_test"][field] *= 1.001
    with pytest.raises(subject.ContractError, match="arithmetic mismatch"):
        subject.validate_readout(sources.panel(5, "B"), analysis)


def test_transport_identity_and_arithmetic(sources):
    rows, analysis = sources.panel(4, "C"), sources.panel(4, "C", 1)
    rows[0]["pair_index"] = 1
    with pytest.raises(subject.ContractError, match="identity mismatch"):
        subject.validate_transport(rows, analysis)
    rows = sources.panel(4, "C")
    rows[0]["G"] += 1e-12
    with pytest.raises(subject.ContractError, match="arithmetic mismatch"):
        subject.validate_transport(rows, analysis)


def test_row_order_is_deterministic(sources):
    for figure, panel, validator in ((5, "B", subject.validate_readout), (4, "C", subject.validate_transport)):
        rows, analysis = sources.panel(figure, panel), sources.panel(figure, panel, 1)
        assert validator(list(reversed(rows)), analysis) == validator(rows, analysis)


def test_required_summary_field_missing(sources, monkeypatch):
    original = sources.panel

    def broken(figure, panel, index=0):
        data = original(figure, panel, index)
        if figure == 2 and panel == "E":
            del data["spectral_profiles"]["0.025"]["mean_plane_contributions"]["P3"]
        return data

    monkeypatch.setattr(sources, "panel", broken)
    with pytest.raises(subject.ContractError, match="Required source field"):
        subject.prepare(sources)


def test_markdown_summary_schema_fails_closed():
    with pytest.raises(subject.ContractError, match="Missing/ambiguous"):
        subject.bullet("- other: `1`", "mean D_NEC")
    with pytest.raises(subject.ContractError, match="Missing/ambiguous"):
        subject.bullet("- mean: `1`\n- mean: `2`", "mean")


def test_output_cannot_overwrite_sources_or_existing_files(sources, tmp_path):
    frozen = next(iter(sources.paths.values()))
    with pytest.raises(subject.ContractError, match="frozen source"):
        subject.validate_output_dir(frozen, sources)
    (tmp_path / "figure1.pdf").write_bytes(b"existing artifact")
    with pytest.raises(subject.ContractError, match="already exists"):
        subject.validate_output_dir(tmp_path, sources)
    assert (tmp_path / "figure1.pdf").read_bytes() == b"existing artifact"


def test_output_hardlink_cannot_mutate_source(sources, tmp_path):
    # Link only a temporary stand-in, never modify actual scientific artifacts.
    source = tmp_path / "frozen.json"
    source.write_bytes(b"immutable")
    output = tmp_path / "out"
    output.mkdir()
    try:
        (output / "figure1.pdf").hardlink_to(source)
    except OSError:
        pytest.skip("Filesystem does not support hardlinks")
    with pytest.raises(subject.ContractError, match="already exists"):
        subject.validate_output_dir(output, sources)
    assert source.read_bytes() == b"immutable"


def test_no_scientific_execution_or_new_inferential_code():
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    allowed = {"__future__", "argparse", "csv", "hashlib", "json", "math", "pathlib",
               "re", "sys", "numpy", "matplotlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] in allowed for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in allowed
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            assert not any(forbidden in name for forbidden in
                           ("ttest", "bootstrap", "polyfit", "linregress", "random", "backward", "cuda", "eval(", "exec("))
    # Frozen inferential fields are checked/copied; no new test routine exists.
    assert not any(isinstance(n, ast.FunctionDef) and "inference" in n.name for n in ast.walk(tree))


@pytest.fixture(scope="module")
def plot_objects(sources):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    data = subject.prepare(sources)
    figures = subject.build_figures(plt, data)
    yield figures, data
    for figure in figures:
        plt.close(figure)


@pytest.mark.parametrize("figure,axis,rows_key,field,unit,analysis_key,primary_key,mean_key", [
    (3, 2, "transport", "G", 1e-9, "transport_analysis", "primary", "mean_G"),
    (4, 1, "readout", "R", 1e-3, "readout_analysis", "primary_test", "mean_R"),
])
def test_histograms_show_all_stored_primary_values(
    plot_objects, figure, axis, rows_key, field, unit, analysis_key, primary_key, mean_key,
):
    figures, data = plot_objects
    ax = figures[figure].axes[axis]
    values = [row[field] / unit for row in data[rows_key]]
    counts, edges = np.histogram(values, bins=20)
    assert sum(patch.get_height() for patch in ax.patches) == 300
    np.testing.assert_array_equal([patch.get_height() for patch in ax.patches], counts)
    np.testing.assert_allclose([patch.get_x() for patch in ax.patches], edges[:-1])
    np.testing.assert_allclose([patch.get_width() for patch in ax.patches], np.diff(edges))
    # One horizontal baseline and two vertical references; no item trajectories.
    assert len(ax.lines) == 3
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0, 0])
    assert ax.lines[1].get_linestyle() == "--"
    primary = data[analysis_key][primary_key]
    np.testing.assert_array_equal(ax.lines[2].get_xdata(), [primary[mean_key] / unit] * 2)
    assert ax.lines[2].get_color() == subject.ORANGE
    assert ax.get_ylabel() == "Count"
    assert ax.get_xlabel().startswith(f"{field} = ")
    annotation = "\n".join(text.get_text() for text in ax.texts)
    assert "N=300" in annotation
    assert f"{primary[mean_key]:+.2e}" in annotation
    assert f"{primary['p_value']:.2e}" in annotation
    if field == "G":
        assert "Mean D_ADJ=-1.55e-09" in annotation
        assert "Mean D_TRANSPORT=-5.51e-10" in annotation
        assert "Positive restoration not established." in annotation
    else:
        assert f"SD(R)={primary['sd_R']:.2e}" in annotation
        assert f"t(299)={primary['t_statistic']:.2f}" in annotation


def test_item_ticks_and_stage_labels_preserve_observations(plot_objects):
    figures, data = plot_objects
    item_ax = figures[1].axes[0]
    np.testing.assert_array_equal(item_ax.collections[0].get_offsets()[:, 0], np.arange(1, 301))
    np.testing.assert_array_equal(item_ax.collections[0].get_offsets()[:, 1],
                                  [row["C_PP3"] / 1e-8 for row in data["xg1"]])
    assert item_ax.get_xticks()[0] == 1 and item_ax.get_xticks()[-1] == 300
    assert 0 not in item_ax.get_xticks()
    stage_ax = figures[4].axes[3]
    assert [text.get_text() for text in stage_ax.get_xticklabels()] == ["pre35", "35", "39", "43", "47", "norm"]
    for line, values in zip(stage_ax.lines[1:], data["stage_means"]):
        np.testing.assert_array_equal(line.get_ydata(), np.asarray(values) / 1e-3)
    figures[4].canvas.draw()
    bounds = [text.get_window_extent() for text in stage_ax.get_xticklabels()]
    assert all(left.x1 < right.x0 for left, right in zip(bounds, bounds[1:]))
    annotation = "\n".join(text.get_text() for text in stage_ax.texts)
    assert "Persistent C2 opposition: post_block_35" in annotation
    assert "Persistent pair opposition: post_block_47" in annotation


def test_renderer_smoke_expected_names_source_integrity_and_determinism(sources, tmp_path):
    before = {name: path.read_bytes() for name, path in sources.paths.items()}
    first, second = tmp_path / "first", tmp_path / "second"
    result = subject.render(subject.DEFAULT_MANIFEST, first)
    assert set(p.name for p in first.iterdir()) == set(subject.OUTPUT_NAMES)
    assert result["source_sha256"] == {k: hashlib.sha256(v).hexdigest() for k, v in before.items()}
    assert result["readout_frozen_primary_test"] == sources.panel(5, "B", 1)["primary_test"]
    assert result["transport_frozen_primary"] == sources.panel(4, "C", 1)["primary"]
    for key in ("new_model_execution", "new_statistical_tests", "new_p_values"):
        assert result[key] is False
    for i in range(1, 6):
        assert (first / f"figure{i}.pdf").read_bytes().startswith(b"%PDF-")
        assert (first / f"figure{i}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    # Second invocation exercises the actual CLI and a fresh process. No timestamps
    # or absolute output paths enter the artifacts, including PDF metadata.
    command = [sys.executable, str(Path(subject.__file__).resolve()),
               "--manifest", str(subject.DEFAULT_MANIFEST), "--output-dir", str(second)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    for name in subject.OUTPUT_NAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert json.loads((first / "render_manifest.json").read_text()) == result
    for name, path in sources.paths.items():
        assert path.read_bytes() == before[name], name
    sources.verify_unchanged()
