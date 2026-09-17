from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_reason_router_gen4_xg2_basis_cross_family_holdout as holdouts
from scripts import reason_router_gen4_xg2_basis_cross_family_holdout_tokenizer_anchor_eligibility as gate


class FakeTokenizer:
    def encode(
        self,
        text,
        add_special_tokens=False,
    ):
        assert add_special_tokens is False

        ids = []
        offsets = []
        cursor = 0

        for token_id, word in enumerate(
            text.split(" "),
            1,
        ):
            start = text.index(
                word,
                cursor,
            )
            end = start + len(word)

            ids.append(token_id)
            offsets.append(
                (start, end)
            )
            cursor = end

        return SimpleNamespace(
            ids=ids,
            offsets=offsets,
        )


def test_frozen_provenance_constants():
    assert (
        gate.STRUCTURAL_FREEZE_COMMIT
        == "c5679b8a22f103956a46cbc665354b4f9cd7063a"
    )
    assert (
        gate.SCOPE_FREEZE_COMMIT
        == "c5679b8a22f103956a46cbc665354b4f9cd7063a"
    )
    assert (
        gate.FRESH_BUILDER_SHA256
        == "441b7c4701a36740c0a7a97c2eac3b1007db012e99d50bf10b2f5323dccdd223"
    )
    assert (
        gate.INHERITED_GATE_BLOB
        == "29a97f343f372718ca56436c505893136cb505b6"
    )
    assert (
        gate.XG1_GATE_BLOB
        == "6c98ce022ca134e385db28851fd364dc6daff423"
    )


def test_exact_fresh_input_hashes():
    assert gate.FROZEN == {
        "xg2": {
            "source_sha256":
                "aaf0ad1f049feee7ee2f1893d86ce6410a9f3ded0ef8fb30fb7df570993d2ef2",
            "rows_sha256":
                "399fc00f7cd20ae9d46305224dfbc74cd1064e1d0276df42bf5d1db5b389e558",
            "manifest_sha256":
                "4e0451caebabbf5df86a522c8fa9ec74b539de77316ad738024c0a20253f4365",
        },
        "xg4": {
            "source_sha256":
                "e0dde08785de0ff5702ef004696eb4e218ca9887f72eff1fe6573fd82286a6d7",
            "rows_sha256":
                "4e3b31dfc0aaf008f62e0f91e2e26f52bdd5a0d39d12f247d4c9158d4dddd274",
            "manifest_sha256":
                "a434443da59705620fb7efd3251d1398c1d488f1cb62e6548c51a6697b454b0d",
        },
    }


@pytest.mark.parametrize(
    "family",
    ["xg2", "xg4"],
)
def test_frozen_fresh_inputs_are_exact(
    family,
):
    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    facts, rows, manifest = (
        gate.load_family(
            family,
            root,
        )
    )

    assert len(facts) == 300
    assert len(rows) == 1800

    assert facts[0]["pair_id"] == (
        f"{family}_fact_601"
    )
    assert facts[-1]["pair_id"] == (
        f"{family}_fact_900"
    )

    assert manifest["family_key"] == family


@pytest.mark.parametrize(
    "family",
    ["xg2", "xg4"],
)
def test_fresh_holdouts_have_zero_static_topology_failures(
    family,
):
    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    facts, _rows, _manifest = (
        gate.load_family(
            family,
            root,
        )
    )

    assert (
        gate.inherited.topology_failures(
            family,
            facts,
        )
        == []
    )


@pytest.mark.parametrize(
    "family",
    ["xg2", "xg4"],
)
def test_fake_tokenizer_preserves_target_identity_name_terminal(
    family,
):
    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    facts, frozen_rows, _manifest = (
        gate.load_family(
            family,
            root,
        )
    )

    fact = facts[0]
    pair_id = str(
        fact["pair_id"]
    )

    rows = [
        row
        for row in frozen_rows
        if str(
            row["source_pair_id"]
        )
        == pair_id
    ]

    assert len(rows) == 6

    tokenizer = FakeTokenizer()

    for cell_id in (
        "C0_SHAM",
        "C2_NAME",
    ):
        row = next(
            item
            for item in rows
            if item[
                "contrast_cell_id"
            ]
            == cell_id
        )

        events = (
            gate.inherited
            .analyze_row(
                family,
                row,
                fact,
                tokenizer,
            )
        )

        lookup = {
            event["anchor_name"]:
                event
            for event in events
        }

        assert (
            lookup[
                "A_IDENTITY"
            ][
                "absolute_anchor_token_index"
            ]
            == lookup[
                "A_NAME"
            ][
                "absolute_anchor_token_index"
            ]
        )


def test_only_xg2_xg4_are_supported():
    with pytest.raises(
        gate.EligibilityError,
        match="UNKNOWN_FAMILY:xg3",
    ):
        gate.load_family(
            "xg3"
        )


def test_source_has_no_model_cuda_or_training_dependency():
    source = inspect.getsource(
        gate
    )

    for token in (
        "import torch",
        "torch.cuda",
        "selected_checkpoint.pt",
        "load_representative_model",
        "capture_branch(",
        ".backward(",
        ".train(",
    ):
        assert token not in source


def test_schema_and_anchor_contract_are_frozen():
    assert gate.FAMILY_KEYS == (
        "xg2",
        "xg4",
    )
    assert (
        gate.EXPECTED_PAIR_COUNT
        == 300
    )
    assert (
        gate.EXPECTED_ROW_COUNT
        == 1800
    )
    assert (
        gate.EXPECTED_ANCHOR_ROWS
        == 1800
    )
    assert (
        gate.ANCHOR_EXPECTED_COUNTS
        == {
            "A_IDENTITY": 1200,
            "A_NAME": 600,
        }
    )
    assert (
        gate.TARGET_IDENTITY_NAME_CELLS
        == (
            "C0_SHAM",
            "C2_NAME",
        )
    )

def test_scope_and_new_cohort_are_pinned():
    assert gate.EXPECTED_BRANCH == "gen4-k-xg2-basis-holdout"
    assert gate.SCOPE_PATH == (
        "reports/reason_router_gen4_xg2_basis_cross_family_holdout_scope.md"
    )
    assert gate.SCOPE_BLOB == (
        "94baa8cda2c2b2ea5400609f89588ae212507553"
    )
    assert gate.COHORT_ROOT.as_posix() == (
        "data/reason_router_gen4_xg2_basis_cross_family_holdout_v1"
    )


def test_exact_601_900_manifest_contract():
    assert gate.ANCHOR_SCHEMA == (
        "GEN4_XG2_BASIS_CROSS_FAMILY_HOLDOUT_"
        "TOKENIZER_ANCHOR_ELIGIBILITY_V1"
    )
    assert gate.SUMMARY_SCHEMA == (
        "GEN4_XG2_BASIS_CROSS_FAMILY_HOLDOUT_"
        "TOKENIZER_ANCHOR_ELIGIBILITY_SUMMARY_V1"
    )
