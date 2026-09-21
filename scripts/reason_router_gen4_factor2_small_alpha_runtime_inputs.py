#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import build_reason_router_gen4_factor2_small_alpha_holdout as holdout
from scripts import reason_router_gen4_mamba370m14b_behavioral_bridge_fast_cuda as bridge


ROOT = _REPO_ROOT

SCALES = ("mamba370m", "mamba14b")
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
PAIR_FIRST = 5701
PAIR_LAST = 6000
PAIR_COUNT = 300
PAIR_IDS = tuple(f"xg1_fact_{i}" for i in range(PAIR_FIRST, PAIR_LAST + 1))
ROWS_PER_SCALE = PAIR_COUNT * len(TARGET_CELLS)

BEHAVIORAL_ALPHAS = (0.25, 0.125, 0.0625, 0.03125)

STRUCTURAL_SOURCE_SHA256 = (
    "05026973b2ec61847c85d6aab800eada130aad9e9c7edb14a4d8d88f41544c4e"
)
STRUCTURAL_ROW_SHA256 = (
    "08da7b3b1d9d92f189b6481abd0b889aeac908519eeae804b8574328ce497f51"
)
TOKEN_GATE_DIR = Path(
    "reports/"
    "reason_router_gen4_mamba370m14b_factor2_small_alpha_"
    "tokenizer_anchor_eligibility_v1"
)
TOKEN_GATE_FILE = "cross_scale_summary.json"
TOKEN_GATE_CROSS_SCALE_SHA256 = (
    "81790aa7fe8c6dc35022485cb70c9300ea28cfe299f2d8822673e947abed2d9d"
)
TOKEN_GATE_RESULT = "PASS_GEN4_FACTOR2_SMALL_ALPHA_TOKENIZER_ANCHOR_ELIGIBILITY"


class Factor2RuntimeInputError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Factor2RuntimeInputError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scale_spec(scale: str) -> Mapping[str, Any]:
    require(scale in SCALES, f"SCALE:{scale}")
    return bridge.scale_spec(scale)


def validate_token_gate() -> dict[str, Any]:
    path = ROOT / TOKEN_GATE_DIR / TOKEN_GATE_FILE
    require(path.is_file(), "TOKEN_GATE_MISSING")
    require(
        sha256_file(path) == TOKEN_GATE_CROSS_SCALE_SHA256,
        "TOKEN_GATE_SHA",
    )
    gate = json.loads(path.read_text(encoding="utf-8"))
    require(gate["result"] == TOKEN_GATE_RESULT, "TOKEN_GATE_RESULT")
    require(gate["all_scales_pass"] is True, "TOKEN_GATE_COMBINED")
    require(
        gate["scale_results"] == {
            "mamba370m": "PASS_600_OF_600",
            "mamba14b": "PASS_600_OF_600",
        },
        "TOKEN_GATE_SCALE_RESULTS",
    )
    require(gate["source_pair_count"] == 300, "TOKEN_GATE_PAIR_COUNT")
    require(
        gate["target_row_count_per_scale"] == 600,
        "TOKEN_GATE_ROW_COUNT",
    )
    require(gate["model_forward_count"] == 0, "TOKEN_GATE_FORWARD")
    require(gate["checkpoint_load_count"] == 0, "TOKEN_GATE_CHECKPOINT")
    require(gate["gpu_used"] is False, "TOKEN_GATE_GPU")
    require(
        gate["scientific_outcomes_observed"] is False,
        "TOKEN_GATE_OUTCOMES",
    )
    return gate


def load_population() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = ROOT / holdout.OUTPUT_DIR
    manifest = holdout.validate_written(root)
    require(manifest["result"] == holdout.RESULT, "HOLDOUT_RESULT")
    require(
        manifest["pair_id_first"] == "xg1_fact_5701"
        and manifest["pair_id_last"] == "xg1_fact_6000",
        "HOLDOUT_RANGE",
    )
    require(manifest["source_pair_count"] == 300, "HOLDOUT_PAIR_COUNT")
    require(manifest["row_count"] == 1800, "HOLDOUT_ROW_COUNT")
    require(manifest["target_cells"] == list(TARGET_CELLS), "HOLDOUT_CELLS")
    require(
        manifest["behavioral_alphas"] == list(BEHAVIORAL_ALPHAS),
        "HOLDOUT_ALPHAS",
    )
    require(manifest["primary_curve"] == "K(alpha)", "HOLDOUT_PRIMARY_CURVE")
    require(
        manifest["negative_alpha_arm_allowed"] is False,
        "HOLDOUT_NEGATIVE_ALPHA",
    )
    require(
        manifest["source_file_sha256"] == STRUCTURAL_SOURCE_SHA256,
        "HOLDOUT_SOURCE_SHA",
    )
    require(
        manifest["row_file_sha256"] == STRUCTURAL_ROW_SHA256,
        "HOLDOUT_ROW_SHA",
    )

    facts = holdout._read_jsonl(root / holdout.SOURCE_FILE)
    rows = holdout._read_jsonl(root / holdout.ROW_FILE)
    return facts, rows, manifest


def validate_target_rows(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    facts_by_id = {str(row["pair_id"]): row for row in facts}
    require(len(facts_by_id) == PAIR_COUNT, "FACT_ID_COUNT")

    by_key = {
        (str(row["source_pair_id"]), str(row["contrast_cell_id"])): row
        for row in rows
    }
    selected: list[dict[str, Any]] = []
    for pair in PAIR_IDS:
        for cell in TARGET_CELLS:
            require(
                (pair, cell) in by_key,
                f"TARGET_ROW_MISSING:{pair}:{cell}",
            )
            selected.append(dict(by_key[(pair, cell)]))

    require(len(selected) == ROWS_PER_SCALE, "SELECTED_ROW_COUNT")
    require(
        len({
            (str(row["source_pair_id"]), str(row["contrast_cell_id"]))
            for row in selected
        }) == ROWS_PER_SCALE,
        "SELECTED_ROW_UNIQUENESS",
    )
    return selected, facts_by_id


def build_input_state(
    *,
    spec: Mapping[str, Any],
    snapshot: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Any],
]:
    gate = validate_token_gate()
    facts, all_rows, manifest = load_population()
    selected, facts_by_id = validate_target_rows(facts, all_rows)

    geom = spec["geom"]
    tokenizer_gate = spec["tokenizer_gate"]
    adapter = spec["adapter"]

    tokenizer, tokenizer_provenance = geom.load_tokenizer(snapshot)
    encoded = adapter.encode_gen4_rows(selected, tokenizer)
    require(
        tuple(encoded["input_ids"].shape) == (ROWS_PER_SCALE, 128),
        "ENCODED_INPUT_SHAPE",
    )

    events: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        analyzed = tokenizer_gate.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair],
            tokenizer,
        )
        by_name = {
            str(event["anchor_name"]): dict(event)
            for event in analyzed
        }
        require(
            {"A_IDENTITY", "A_NAME"} <= set(by_name),
            f"ANCHOR_SET:{pair}:{cell}",
        )
        identity = by_name["A_IDENTITY"]
        name = by_name["A_NAME"]
        require(
            bool(identity["post4_eligible"]),
            f"IDENTITY_INELIGIBLE:{pair}:{cell}",
        )
        require(
            bool(name["post4_eligible"]),
            f"NAME_INELIGIBLE:{pair}:{cell}",
        )
        require(
            identity["absolute_anchor_token_index"]
            == name["absolute_anchor_token_index"],
            f"IDENTITY_NAME_INDEX_MISMATCH:{pair}:{cell}",
        )
        for anchor_name in ("A_IDENTITY", "A_NAME"):
            events[(pair, cell, anchor_name)] = by_name[anchor_name]

    require(len(events) == ROWS_PER_SCALE * 2, "ANCHOR_EVENT_COUNT")
    return selected, encoded, events, {
        "manifest": manifest,
        "tokenizer": tokenizer_provenance,
        "token_gate": gate,
        "token_gate_cross_scale_sha256": TOKEN_GATE_CROSS_SCALE_SHA256,
    }


def row_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        require(key not in out, f"DUPLICATE_ROW:{key}")
        out[key] = index
    require(len(out) == ROWS_PER_SCALE, "ROW_INDEX_COUNT")
    return out
