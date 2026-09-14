import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import reason_router_gen4_k_fast_cuda_full_compare as cmp


def _checksum_text(files):
    lines = []
    for name in sorted(files):
        lines.append(
            f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        )
    return "".join(lines).encode()


def _minimal_bundle_files():
    payload = {
        "manifest.json": b"{}\n",
        "item_metrics.jsonl": b"",
        "summary.json": b"{}\n",
    }
    payload["SHA256SUMS.txt"] = _checksum_text(payload)
    return payload


def _write_bundle_dir(root: Path):
    root.mkdir()
    files = _minimal_bundle_files()
    for name, raw in files.items():
        (root / name).write_bytes(raw)


def _write_bundle_zip(path: Path):
    files = _minimal_bundle_files()
    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in files.items():
            zf.writestr(f"bundle/{name}", raw)


def test_compare_constants_are_frozen():
    assert (
        cmp.CPU_REFERENCE_ZIP_SHA256
        == "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
    )
    assert cmp.SOURCE_PAIR_COUNT == 300
    assert cmp.MODEL_FORWARD_COUNT == 2400
    assert cmp.PE_ATOL == 1e-4
    assert cmp.GEOMETRY_ATOL == 1e-4
    assert cmp.GEOMETRY_RTOL == 1e-4


def test_directory_bundle_checksum_roundtrip(tmp_path):
    root = tmp_path / "bundle"
    _write_bundle_dir(root)
    bundle = cmp.load_bundle(root)
    assert set(bundle.files) == cmp.FILES


def test_zip_bundle_checksum_roundtrip(tmp_path):
    path = tmp_path / "bundle.zip"
    _write_bundle_zip(path)
    bundle = cmp.load_bundle(path)
    assert set(bundle.files) == cmp.FILES



def test_zip_bundle_uses_posix_member_semantics(tmp_path):
    path = tmp_path / "bundle.zip"
    files = _minimal_bundle_files()

    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in files.items():
            zf.writestr(
                f"nested/bundle/{name}",
                raw,
            )

    bundle = cmp.load_bundle(path)
    assert set(bundle.files) == cmp.FILES

def test_corrupt_checksum_fails(tmp_path):
    root = tmp_path / "bundle"
    _write_bundle_dir(root)
    (root / "manifest.json").write_text(
        '{"corrupt":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(cmp.CompareError):
        cmp.load_bundle(root)


def test_numeric_gate_accepts_and_rejects():
    assert (
        cmp._close(
            1.00005,
            1.0,
            atol=1e-4,
            label="ok",
        )
        <= 1e-4
    )

    with pytest.raises(cmp.CompareError):
        cmp._close(
            1.001,
            1.0,
            atol=1e-4,
            label="bad",
        )


def _synthetic_item():
    row = {}
    for field in cmp.EXACT_FIELDS:
        if field == "source_pair_id":
            row[field] = "pair0"
        elif field == "schema_version":
            row[field] = "schema"
        elif field.endswith("_cell") or field == "anchor_name":
            row[field] = field
        elif "path_efficiency" in field:
            row[field] = 0.5
        else:
            row[field] = 1

    for field in cmp.GEOMETRY_FIELDS:
        row[field] = 0.25
    for field in cmp.PE_FIELDS:
        row[field] = 0.1
    for field in cmp.BASELINE_RESIDUAL_FIELDS:
        row[field] = 0.0
    for field in cmp.VECTOR_RESIDUAL_FIELDS:
        row[field] = 0.0
    for field in cmp.MIDPOINT_RESIDUAL_FIELDS:
        row[field] = 0.0
    for field in cmp.CAST_RESIDUAL_FIELDS:
        row[field] = 0.0
    return row


def test_item_comparison_accepts_small_backend_delta():
    cpu = _synthetic_item()
    cuda = dict(cpu)
    cuda["R_ALIGN"] += 5e-5
    cuda["target_A"] += 5e-5
    result = cmp.compare_item(cpu, cuda)
    assert result["max_pe_abs_diff"] > 0.0
    assert result["max_geometry_abs_diff"] > 0.0


def test_item_comparison_rejects_large_backend_delta():
    cpu = _synthetic_item()
    cuda = dict(cpu)
    cuda["R_ALIGN"] += 1e-3
    with pytest.raises(cmp.CompareError):
        cmp.compare_item(cpu, cuda)


def test_summary_requires_identical_holm_decisions():
    cpu = {
        "population_size": 300,
        "baseline_delta_name_path_efficiency_mean": -0.01,
        "hypothesis_family": [
            {
                "id": "H1_R_ALIGN_GT_ZERO",
                "n": 300,
                "mean": -1e-5,
                "reject_holm_alpha_0_05": False,
            },
            {
                "id": "H2_ALIGNMENT_SPECIFICITY_GT_ZERO",
                "n": 300,
                "mean": -2e-5,
                "reject_holm_alpha_0_05": False,
            },
        ],
        "outcome": cmp.NEGATIVE_OUTCOME,
    }
    cuda = json.loads(json.dumps(cpu))
    cuda["hypothesis_family"][0][
        "reject_holm_alpha_0_05"
    ] = True

    with pytest.raises(cmp.CompareError):
        cmp.compare_summaries(cpu, cuda)
