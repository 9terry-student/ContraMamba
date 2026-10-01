from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import train_reason_router_gen5_phase3a_contention as train


def base_args(**updates):
    values = {
        "static_verify_only": True,
        "cuda_preflight_only": False,
        "run_cell": False,
        "expected_head": train.IMPLEMENTATION_AUTHORITY_COMMIT,
        "allow_opening_worktree": True,
        "pressure": None,
        "seed": None,
        "tokenizer_snapshot": None,
        "model_snapshot": None,
        "checkpoint": None,
        "preflight_output": None,
        "output_root": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_static_mode_forbids_runtime_inputs():
    train.validate_mode_args(base_args())
    with pytest.raises(Exception):
        train.validate_mode_args(base_args(checkpoint=Path("x.pt")))
    with pytest.raises(Exception):
        train.validate_mode_args(base_args(pressure="PR"))
    with pytest.raises(Exception):
        train.validate_mode_args(base_args(tokenizer_snapshot=Path("tok")))


def test_runtime_modes_require_pressure_seed_checkpoint():
    args = base_args(
        static_verify_only=False,
        cuda_preflight_only=True,
        allow_opening_worktree=False,
        preflight_output=Path("preflight.json"),
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)

    args.pressure = "PR"
    args.seed = 6201
    args.checkpoint = Path("parent.pt")
    train.validate_mode_args(args)


def test_run_cell_requires_output_root():
    args = base_args(
        static_verify_only=False,
        run_cell=True,
        allow_opening_worktree=False,
        pressure="P0",
        seed=6201,
        checkpoint=Path("parent.pt"),
    )
    with pytest.raises(Exception):
        train.validate_mode_args(args)
    args.output_root = Path("outputs/p3a")
    train.validate_mode_args(args)


def test_training_contract_constants():
    assert train.TRAIN_ROWS == 3360
    assert train.DEV_ROWS == 840
    assert train.TRAIN_PAIRS == 480
    assert train.DEV_PAIRS == 120
    assert train.SPLIT_SEED == 16384
    assert train.TRAINING_SEEDS == (6201, 6202, 6203)
    assert train.TRAINING_PRESSURES == ("P0", "PR", "PC")
    assert train.ARM == "G5-C0"
    assert train.EPOCHS == 20
    assert train.TOTAL_OPTIMIZER_STEPS == 20
    assert train.LEARNING_RATE == 0.001
    assert train.WEIGHT_DECAY == 0.0001
    assert train.GRADIENT_CLIP_NORM == 5.0
    assert train.BACKBONE_STREAM_ROWS == 240
    assert train.TRAIN_ROWS % train.BACKBONE_STREAM_ROWS == 0


def test_static_file_contract_is_exact_17():
    assert len(train.STATIC_FILES) == 17
    assert len(set(train.STATIC_FILES)) == 17


def test_row_order_hash_is_order_sensitive():
    rows = [
        {
            "id": "a",
            "pair_id": "p1",
            "contrast_cell_id": "C0_SHAM",
            "final_label": "SUPPORT",
            "final_label_id": 2,
            "stressor_domain": True,
        },
        {
            "id": "b",
            "pair_id": "p2",
            "contrast_cell_id": "C6_EXPLICIT_DENIAL",
            "final_label": "REFUTE",
            "final_label_id": 0,
            "stressor_domain": False,
        },
    ]
    assert train.row_order_sha256(rows) != train.row_order_sha256(list(reversed(rows)))


class ToyEncoding:
    eos_token_id = 0
    pad_token_id = 0

    class E:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = [ord(ch) % 97 + 1 for ch in text]
        return self.E(ids)


def test_encoding_uses_frozen_target_map_and_labels():
    rows = [
        {
            "id": "a",
            "row_id": "a",
            "pair_id": "p1",
            "source_pair_id": "p1",
            "contrast_cell_id": "C0_SHAM",
            "claim": "abcd",
            "evidence": "efghijkl",
            "final_label": "SUPPORT",
            "final_label_id": 2,
            "stressor_domain": True,
        },
        {
            "id": "b",
            "row_id": "b",
            "pair_id": "p1",
            "source_pair_id": "p1",
            "contrast_cell_id": "C6_EXPLICIT_DENIAL",
            "claim": "abcd",
            "evidence": "efghijkl",
            "final_label": "REFUTE",
            "final_label_id": 0,
            "stressor_domain": False,
        },
    ]
    bundle = train._encode_rows_with_active_tokenizer(
        rows,
        ToyEncoding(),
        {("p1", "C0_SHAM"): 2},
    )
    assert bundle["model_inputs"]["input_ids"].shape == (2, 128)
    assert bundle["model_inputs"]["final_labels"].tolist() == [2, 0]
    assert bundle["stressor_active"].tolist() == [True, False]
    assert bundle["target_indices"].tolist() == [2, -1]


def test_encoded_hash_changes_with_target_coordinate():
    rows = [
        {
            "id": "a",
            "row_id": "a",
            "pair_id": "p1",
            "source_pair_id": "p1",
            "contrast_cell_id": "C0_SHAM",
            "claim": "abcdef",
            "evidence": "abcdefghij",
            "final_label": "SUPPORT",
            "final_label_id": 2,
            "stressor_domain": True,
        }
    ]
    a = train._encode_rows_with_active_tokenizer(
        rows, ToyEncoding(), {("p1", "C0_SHAM"): 2}
    )
    b = train._encode_rows_with_active_tokenizer(
        rows, ToyEncoding(), {("p1", "C0_SHAM"): 3}
    )
    assert train.encoded_bundle_sha256(a) != train.encoded_bundle_sha256(b)


def test_parser_modes_are_mutually_exclusive():
    parser = train.build_parser()
    parsed = parser.parse_args(["--static-verify-only", "--expected-head", "abc"])
    assert parsed.static_verify_only

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--static-verify-only",
            "--cuda-preflight-only",
            "--expected-head",
            "abc",
        ])
