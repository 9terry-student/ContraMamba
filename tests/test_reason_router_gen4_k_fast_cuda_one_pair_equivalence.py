from scripts import reason_router_gen4_k_fast_cuda_one_pair_equivalence as eq


def test_frozen_backend_identity():
    assert eq.KERNELS_VERSION == "0.10.2"
    assert eq.MAMBA_REV == "c8ffc584c147878a6eb978ae0e8db4d116c93a8c"
    assert eq.CONV_REV == "f2651e776f66069cdcf842840db637583def1223"
    assert eq.MAMBA_BINARY_SHA256 == "dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587"
    assert eq.CONV_BINARY_SHA256 == "6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6"


def test_frozen_equivalence_thresholds():
    assert eq.STATE_ATOL == 1e-4
    assert eq.STATE_RTOL == 1e-4
    assert eq.GEOMETRY_ATOL == 1e-4
    assert eq.GEOMETRY_RTOL == 1e-4
    assert eq.PE_ATOL == 1e-4
    assert eq.TOTAL_MODEL_FORWARDS == 16


def test_branch_plan_is_exact():
    assert eq.BRANCH_LABELS == (
        "baseline_tp",
        "baseline_tm",
        "baseline_rp",
        "baseline_rm",
        "alignment_tp",
        "alignment_tm",
        "magnitude_tp",
        "magnitude_tm",
    )


def test_scalar_gate_accepts_and_rejects():
    assert eq._compare_scalar(
        1.00005,
        1.0,
        atol=1e-4,
        label="ok",
    ) <= 1e-4

    try:
        eq._compare_scalar(
            1.001,
            1.0,
            atol=1e-4,
            label="bad",
        )
    except eq.EquivalenceError:
        pass
    else:
        raise AssertionError("equivalence gate failed open")


def test_no_scientific_conclusion_constant():
    assert eq.FORWARDS_PER_BACKEND == 8
    assert eq.TOTAL_MODEL_FORWARDS == 16
