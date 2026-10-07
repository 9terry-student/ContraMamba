from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_native_state_kinematics_vitaminc_cluster_feasibility.py"
)

spec = importlib.util.spec_from_file_location("q1audit", SCRIPT)
assert spec is not None and spec.loader is not None
q1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q1)


def row(case_id, claim, evidence, raw_idx=0, label="SUPPORTS"):
    return {
        "raw_idx": str(raw_idx),
        "unique_id": f"{case_id}_{raw_idx}",
        "case_id": case_id,
        "label": label,
        "claim": claim,
        "evidence": evidence,
    }


def test_raw_edit_span_middle_substitution():
    out = q1.raw_edit_span(
        "alpha beta old value omega tail one two three four",
        "alpha beta new value omega tail one two three four",
    )
    assert out["common_prefix_tokens"] == 2
    assert out["postedit_suffix_tokens"] == 7


def test_raw_edit_span_terminal_change_has_no_suffix():
    out = q1.raw_edit_span("a b c x", "a b c y")
    assert out["common_prefix_tokens"] == 3
    assert out["postedit_suffix_tokens"] == 0


def test_complete_2x2_accepts_exact_cartesian_product():
    rows = [
        row("c", "q1", "e1", 0),
        row("c", "q1", "e2", 1),
        row("c", "q2", "e1", 2),
        row("c", "q2", "e2", 3),
    ]
    assert q1.is_complete_2x2(rows)


def test_complete_2x2_rejects_missing_cell():
    rows = [
        row("c", "q1", "e1", 0),
        row("c", "q1", "e2", 1),
        row("c", "q2", "e1", 2),
    ]
    assert not q1.is_complete_2x2(rows)


def test_complete_2x2_rejects_duplicate_cell():
    rows = [
        row("c", "q1", "e1", 0),
        row("c", "q1", "e2", 1),
        row("c", "q2", "e1", 2),
        row("c", "q2", "e1", 3),
    ]
    assert not q1.is_complete_2x2(rows)


def test_label_normalization():
    assert q1.normalize_label("SUPPORT") == "SUPPORTS"
    assert q1.normalize_label("REFUTE") == "REFUTES"
    assert q1.normalize_label("NOT ENOUGH INFO") == "NEI"


def test_truthy01():
    assert q1.truthy01("1") is True
    assert q1.truthy01("true") is True
    assert q1.truthy01("0") is False
    assert q1.truthy01("false") is False


def test_source_has_no_model_runtime_dependencies():
    text = SCRIPT.read_text(encoding="utf-8")
    forbidden_imports = (
        "import torch",
        "from torch",
        "import transformers",
        "from transformers",
        "import numpy",
        "import pandas",
    )
    for token in forbidden_imports:
        assert token not in text


def test_expected_contract_is_cluster_level():
    assert q1.EXPECTED["dataset_rows"] == 5000
    assert q1.EXPECTED["unique_case_ids"] == 1497
    assert q1.EXPECTED["complete_2x2_case_ids"] == 1010
    assert q1.EXPECTED["complete_2x2_postedit_suffix_ge4_case_ids"] == 681
    assert q1.EXPECTED["historical_unique_raw_error_rows"] == 93
    assert q1.EXPECTED["historical_unique_error_case_ids"] == 78


def test_historical_pair_defects_are_frozen():
    assert q1.EXPECTED["historical_pair_rows"] == 128
    assert q1.EXPECTED["historical_same_predicted_class_pairs"] == 57
    assert q1.EXPECTED["historical_wrong_control_case_id_overlap"] == 4


def test_verify_mode_requires_no_output():
    parser = q1.build_parser()
    args = parser.parse_args(
        [
            "--static-verify-only",
            "--raw-dataset",
            "a.csv",
            "--fixed-errors",
            "b.csv",
            "--pairs",
            "c.csv",
        ]
    )
    assert args.static_verify_only
    assert args.output_json is None