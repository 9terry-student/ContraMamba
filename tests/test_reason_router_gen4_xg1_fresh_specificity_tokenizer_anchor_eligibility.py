from __future__ import annotations

import inspect

from scripts import (
    reason_router_gen4_xg1_fresh_specificity_tokenizer_anchor_eligibility
    as gate
)


def test_fresh_gate_identity():
    assert gate.EXPECTED_BRANCH == "gen4-k-xg2-basis-holdout"
    assert gate.STRUCTURAL_FREEZE_COMMIT == (
        "ddb1404800af6dbd89982bbbcdd8262d203577f6"
    )
    assert gate.DESIGN_FREEZE_COMMIT == (
        "0bc49ab95cbb2c8735b4bc79422d660fa64e3e01"
    )
    assert gate.EXPECTED_PAIR_COUNT == 300
    assert gate.EXPECTED_ROW_COUNT == 1800
    assert gate.EXPECTED_ANCHOR_ROWS == 1800
    assert gate.PAIR_ID_FIRST == "xg1_fact_301"
    assert gate.PAIR_ID_LAST == "xg1_fact_600"


def test_frozen_fresh_inputs_validate():
    facts, rows, manifest = gate.load_frozen_fresh_inputs()

    assert len(facts) == 300
    assert len(rows) == 1800

    assert facts[0]["pair_id"] == "xg1_fact_301"
    assert facts[-1]["pair_id"] == "xg1_fact_600"

    assert manifest["result"] == (
        "PASS_XG1_FRESH_SPECIFICITY_STRUCTURAL_FREEZE"
    )


def test_gate_reuses_frozen_legacy_anchor_logic():
    assert gate.LEGACY_GATE_BLOB == (
        "6c98ce022ca134e385db28851fd364dc6daff423"
    )

    assert gate.legacy.MAX_LENGTH == 128
    assert gate.legacy.CLAIM_BUDGET == 63
    assert gate.legacy.EVIDENCE_BUDGET == 64
    assert gate.legacy.EOS_TOKEN_ID == 0

    assert gate.legacy.ANCHOR_EXPECTED_COUNTS == {
        "A_IDENTITY": 1200,
        "A_NAME": 600,
    }


def test_gate_has_no_model_execution_surface():
    source = inspect.getsource(gate)

    for token in (
        "AutoModel",
        "load_representative_model",
        "capture_branch(",
        "run_full(",
        "torch.cuda",
        ".cuda(",
    ):
        assert token not in source
