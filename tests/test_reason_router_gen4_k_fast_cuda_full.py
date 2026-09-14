from pathlib import Path

from scripts import reason_router_gen4_k_fast_cuda_full as full
from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as gate


def test_full_constants_bind_passed_gate():
    assert (
        full.ONE_PAIR_GATE_COMMIT
        == "d6e1521c3f2c92f16aa88bbaf3f6b6c332a85ae6"
    )
    assert (
        full.ONE_PAIR_GATE_BLOB
        == "4a8326c442b883510ba452f049c88f3163677fc5"
    )
    assert (
        full.CPU_REFERENCE_ZIP_SHA256
        == "25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf"
    )


def test_full_budget_is_exact():
    assert full.SOURCE_PAIR_COUNT == 300
    assert full.FORWARDS_PER_PAIR == 8
    assert full.FULL_FORWARD_BUDGET == 2400


def test_full_output_surface_is_exact():
    assert full.EXPECTED_OUTPUT_FILES == {
        "manifest.json",
        "item_metrics.jsonl",
        "summary.json",
        "SHA256SUMS.txt",
    }


def test_manifest_scope_is_equivalence_only():
    manifest = full.build_cuda_manifest(
        head="abc",
        checkpoint_sha="def",
        forward_count=2400,
    )
    assert manifest["backend_equivalence_only"] is True
    assert manifest["scientific_authority"] is False
    assert manifest["device"] == "cuda:0"
    assert manifest["execution_backend"] == (
        "transformers_fast_cuda_selective_scan"
    )
    assert manifest["cpu_reference_required_for_interpretation"] is True


def test_fast_backend_identity_is_inherited_from_passed_gate():
    manifest = full.build_cuda_manifest(
        head="abc",
        checkpoint_sha="def",
        forward_count=2400,
    )
    assert manifest["mamba_revision"] == gate.MAMBA_REV
    assert manifest["mamba_binary_sha256"] == gate.MAMBA_BINARY_SHA256
    assert manifest["causal_conv_revision"] == gate.CONV_REV
    assert manifest["causal_conv_binary_sha256"] == gate.CONV_BINARY_SHA256


def test_full_runner_reuses_gate_fast_capture_source():
    source = (
        Path(full.__file__)
        .read_text(encoding="utf-8")
    )
    assert "gate._make_fast_capture" in source
    assert "gate.load_exact_fast_kernels" in source
    assert "parent.capture_branch = fast_capture" in source
