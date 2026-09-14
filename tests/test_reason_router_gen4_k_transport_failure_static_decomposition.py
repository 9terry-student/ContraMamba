import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import (
    reason_router_gen4_k_transport_failure_static_decomposition
    as dec,
)


def _bundle_payload():
    manifest = {
        "mode": "full",
        "runtime_git_head":
            dec.CPU_REFERENCE_HEAD,
        "source_pair_count": 300,
        "model_forward_count": 2400,
        "training_executed": False,
        "backward_executed": False,
        "task_heads_executed": False,
        "logits_read": False,
        "raw_vectors_persisted": False,
    }
    summary = {
        "outcome": dec.EXPECTED_OUTCOME,
        "all_mandatory_manipulation_checks_pass":
            True,
        "hypothesis_family": [
            {
                "id": "H1_R_ALIGN_GT_ZERO",
                "mean": 0.0,
            },
            {
                "id":
                    "H2_ALIGNMENT_SPECIFICITY_GT_ZERO",
                "mean": 0.0,
            },
        ],
    }

    rows = []
    for i in range(300):
        rows.append(
            {
                "source_pair_id": f"pair{i:03d}",
                "target_A": 1.0 + i / 1000,
                "target_B": 2.0 + i / 1000,
                "target_C": -0.5 + i / 1000,
                "reference_A": 1.1 + i / 1000,
                "reference_B": 2.1 + i / 1000,
                "reference_C": -0.4 + i / 1000,
                "delta_baseline": -0.1,
                "R_ALIGN": 0.0,
                "R_MAG": 0.0,
                "ALIGNMENT_SPECIFICITY": 0.0,
            }
        )

    files = {
        "manifest.json":
            (
                json.dumps(manifest)
                + "\n"
            ).encode(),
        "item_metrics.jsonl":
            "".join(
                json.dumps(row) + "\n"
                for row in rows
            ).encode(),
        "summary.json":
            (
                json.dumps(summary)
                + "\n"
            ).encode(),
    }

    lines = []
    for name in sorted(files):
        lines.append(
            f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        )
    files["SHA256SUMS.txt"] = "".join(
        lines
    ).encode()

    return files


def _write_zip(path: Path):
    files = _bundle_payload()
    with zipfile.ZipFile(path, "w") as zf:
        for name, raw in files.items():
            zf.writestr(
                f"nested/reference/{name}",
                raw,
            )


def test_constants_lock_closed_bridge():
    assert (
        dec.CLOSURE_COMMIT
        == "ad375c88385eef507e151f95717e951919a6a3fe"
    )
    assert (
        dec.CPU_REFERENCE_ZIP_SHA256
        == "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
    )
    assert dec.SOURCE_PAIR_COUNT == 300
    assert dec.MODEL_FORWARD_COUNT == 2400


def test_average_ranks_handle_ties():
    assert dec.average_ranks(
        [10.0, 10.0, 20.0, 30.0]
    ) == [
        1.5,
        1.5,
        3.0,
        4.0,
    ]


def test_correlations_are_exact_for_monotone_data():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [2.0, 4.0, 6.0, 8.0]
    assert dec.pearson(x, y) == pytest.approx(1.0)
    assert dec.spearman(x, y) == pytest.approx(1.0)


def test_stable_quartiles_are_equal_count():
    pair_ids = [
        f"p{i:03d}"
        for i in range(300)
    ]
    values = [
        float(i // 10)
        for i in range(300)
    ]
    groups = dec.stable_quartiles(
        pair_ids,
        values,
    )
    assert [
        len(group)
        for group in groups
    ] == [75, 75, 75, 75]
    flattened = [
        index
        for group in groups
        for index in group
    ]
    assert sorted(flattened) == list(range(300))


def test_feature_derivation():
    row = {
        "target_A": 2.0,
        "target_B": 4.0,
        "target_C": 0.1,
        "reference_A": 4.0,
        "reference_B": 2.0,
        "reference_C": 0.3,
        "delta_baseline": -0.2,
    }
    f = dec.derive_features(row)
    assert f["baseline_severity_abs"] == 0.2
    assert f["alignment_shift"] == pytest.approx(0.2)
    assert f["alignment_shift_abs"] == pytest.approx(0.2)
    assert f["magnitude_shift_norm"] > 0.0


def test_summary_sign_counts():
    out = dec.summarize_values(
        [-2.0, -1.0, 0.0, 3.0]
    )
    assert out["positive_count"] == 1
    assert out["negative_count"] == 2
    assert out["zero_count"] == 1
    assert out["positive_fraction"] == 0.25


def test_nested_zip_uses_posix_paths(tmp_path):
    path = tmp_path / "reference.zip"
    _write_zip(path)

    # Test ZIP parsing independent of the production outer-hash lock.
    original = dec.CPU_REFERENCE_ZIP_SHA256
    try:
        dec.CPU_REFERENCE_ZIP_SHA256 = dec.sha256_file(path)
        files = dec.load_cpu_bundle(path)
    finally:
        dec.CPU_REFERENCE_ZIP_SHA256 = original

    assert set(files) == dec.BUNDLE_FILES


def test_validate_reference_accepts_expected_shape():
    files = _bundle_payload()
    manifest, items, summary = dec.decode_bundle(files)
    dec.validate_cpu_reference(
        manifest,
        items,
        summary,
    )


def test_zero_variance_pearson_fails():
    with pytest.raises(
        dec.StaticDecompositionError
    ):
        dec.pearson(
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        )
