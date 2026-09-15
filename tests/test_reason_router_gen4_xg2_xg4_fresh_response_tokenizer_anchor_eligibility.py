from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_reason_router_gen4_xg2_xg4_fresh_response_holdouts as holdouts
from scripts import reason_router_gen4_xg2_xg4_fresh_response_tokenizer_anchor_eligibility as gate


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
        == "4bda8dd4b46d56ffe9bb37cffc91f8506026ca69"
    )
    assert (
        gate.DESIGN_FREEZE_COMMIT
        == "2074f52d39bf0fca6d63248016ce47c54bff5e06"
    )
    assert (
        gate.FRESH_BUILDER_BLOB
        == "911846cf0b3304caaab0399535bd9f3de7f8249e"
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
                "c02d2fea5a7f3c8b5243598ad5505bba3f276c099be7b8c428c06eb39ffc7141",
            "rows_sha256":
                "1ca4f1c79caf5719e8970bf889c75b0e59f8711e5c78a36f7578727020e23670",
            "manifest_sha256":
                "7df94b0788a55c6a4927926f6b92ef7e90d66346c5342608f5716ddd517e47f9",
        },
        "xg4": {
            "source_sha256":
                "0666c9345505f993bce70d66b5b1f9b784edca782eecb13a750ecf827f14ba3a",
            "rows_sha256":
                "b407613cdcee15d193847130f64e6aa674e666f72d6d08a30b573005e5e9de7d",
            "manifest_sha256":
                "ee8d9bc4ae0ca135c335cf9eaa6bd9025cd5354304d435f1e3bb703a8b11e80d",
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
        f"{family}_fact_301"
    )
    assert facts[-1]["pair_id"] == (
        f"{family}_fact_600"
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