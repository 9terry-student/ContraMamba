from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    reason_router_gen4_factor2_small_alpha_tokenizer_anchor_eligibility
    as m,
)


class FakeTokenizer:
    pass


def _target_fixture() -> tuple[list[dict], list[dict]]:
    facts = []
    rows = []
    for i in range(m.PAIR_FIRST, m.PAIR_LAST + 1):
        pair = f"xg1_fact_{i}"
        facts.append({"pair_id": pair})
        for cell in m.TARGET_CELLS:
            rows.append({
                "source_pair_id": pair,
                "contrast_cell_id": cell,
                "row_id": f"{pair}__{cell}",
            })
    return facts, rows


def _event(name: str, index: int = 10, eligible: bool = True) -> dict:
    return {
        "anchor_name": name,
        "absolute_anchor_token_index": index,
        "post4_eligible": eligible,
        "exclusion_code": None if eligible else "POST4_PREFIX_INELIGIBLE",
    }


def test_constants_match_frozen_plan() -> None:
    assert m.PLAN_FREEZE_COMMIT == (
        "8614dd5a4ed77cb56cc6a485f4ab379434a284fa"
    )
    assert m.STRUCTURAL_FREEZE_COMMIT == (
        "8cc7aa450cb6a16f2e5a2564399d3f07dfd9231e"
    )
    assert m.PAIR_FIRST == 5701
    assert m.PAIR_LAST == 6000
    assert m.PAIR_COUNT == 300
    assert m.TARGET_CELLS == ("C0_SHAM", "C2_NAME")
    assert m.TARGET_ROW_COUNT == 600
    assert m.SCALE_CONFIG["mamba370m"]["revision"] == (
        "589179554943157be31701edd8b4558889276674"
    )
    assert m.SCALE_CONFIG["mamba14b"]["revision"] == (
        "6e46eae61c27280517feef46f536d16b91076f08"
    )


def test_scale_tokenizer_contracts_are_distinct() -> None:
    a = m.SCALE_CONFIG["mamba370m"]["tokenizer_file_sha256"]
    b = m.SCALE_CONFIG["mamba14b"]["tokenizer_file_sha256"]
    assert a["tokenizer.json"] != b["tokenizer.json"]
    assert a["tokenizer_config.json"] != b["tokenizer_config.json"]


def test_analyze_scale_passes_only_600_of_600(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, rows = _target_fixture()

    monkeypatch.setattr(
        m,
        "load_scale_tokenizer",
        lambda scale, snapshot: (
            FakeTokenizer(),
            {
                "repo": m.SCALE_CONFIG[scale]["repo"],
                "revision": m.SCALE_CONFIG[scale]["revision"],
                "tokenizers_version": "0.22.2",
                "vocab_size": 50277,
                "file_sha256":
                    m.SCALE_CONFIG[scale]["tokenizer_file_sha256"],
            },
        ),
    )
    monkeypatch.setattr(
        m.anchor_gate,
        "analyze_required_anchors_for_row",
        lambda row, fact, tokenizer: [
            _event("A_IDENTITY"),
            _event("A_NAME"),
        ],
    )

    anchor_rows, summary = m.analyze_scale(
        scale="mamba370m",
        snapshot=Path(m.SCALE_CONFIG["mamba370m"]["revision"]),
        facts=facts,
        target_rows=rows,
    )
    assert len(anchor_rows) == 1200
    assert summary["result"] == m.SCALE_PASS
    assert summary["eligible_identity_count"] == 600
    assert summary["eligible_name_count"] == 600
    assert summary["identity_name_mismatch_count"] == 0
    assert summary["exclusion_counts"] == {}
    assert summary["model_forward_count"] == 0
    assert summary["checkpoint_load_count"] == 0
    assert summary["gpu_used"] is False


def test_single_ineligible_event_blocks_whole_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, rows = _target_fixture()

    monkeypatch.setattr(
        m,
        "load_scale_tokenizer",
        lambda scale, snapshot: (
            FakeTokenizer(),
            {
                "repo": m.SCALE_CONFIG[scale]["repo"],
                "revision": m.SCALE_CONFIG[scale]["revision"],
                "tokenizers_version": "0.22.2",
                "vocab_size": 50277,
                "file_sha256":
                    m.SCALE_CONFIG[scale]["tokenizer_file_sha256"],
            },
        ),
    )

    first_key = (
        rows[0]["source_pair_id"],
        rows[0]["contrast_cell_id"],
    )

    def analyze(row, fact, tokenizer):
        key = (
            row["source_pair_id"],
            row["contrast_cell_id"],
        )
        if key == first_key:
            return [
                _event("A_IDENTITY", eligible=False),
                _event("A_NAME"),
            ]
        return [_event("A_IDENTITY"), _event("A_NAME")]

    monkeypatch.setattr(
        m.anchor_gate,
        "analyze_required_anchors_for_row",
        analyze,
    )

    _, summary = m.analyze_scale(
        scale="mamba14b",
        snapshot=Path(m.SCALE_CONFIG["mamba14b"]["revision"]),
        facts=facts,
        target_rows=rows,
    )
    assert summary["result"] == m.SCALE_BLOCKED
    assert summary["eligible_identity_count"] == 599
    assert summary["eligible_name_count"] == 600
    assert summary["exclusion_counts"] == {
        "POST4_PREFIX_INELIGIBLE": 1
    }


def test_identity_name_index_mismatch_blocks_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, rows = _target_fixture()

    monkeypatch.setattr(
        m,
        "load_scale_tokenizer",
        lambda scale, snapshot: (
            FakeTokenizer(),
            {
                "repo": m.SCALE_CONFIG[scale]["repo"],
                "revision": m.SCALE_CONFIG[scale]["revision"],
                "tokenizers_version": "0.22.2",
                "vocab_size": 50277,
                "file_sha256":
                    m.SCALE_CONFIG[scale]["tokenizer_file_sha256"],
            },
        ),
    )

    first = True
    def analyze(row, fact, tokenizer):
        nonlocal first
        if first:
            first = False
            return [_event("A_IDENTITY", 10), _event("A_NAME", 11)]
        return [_event("A_IDENTITY"), _event("A_NAME")]

    monkeypatch.setattr(
        m.anchor_gate,
        "analyze_required_anchors_for_row",
        analyze,
    )

    _, summary = m.analyze_scale(
        scale="mamba370m",
        snapshot=Path(m.SCALE_CONFIG["mamba370m"]["revision"]),
        facts=facts,
        target_rows=rows,
    )
    assert summary["result"] == m.SCALE_BLOCKED
    assert summary["identity_name_mismatch_count"] == 1


def test_cross_scale_requires_both_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts, rows = _target_fixture()
    monkeypatch.setattr(
        m,
        "validate_structural_inputs",
        lambda root=m.ROOT: (facts, rows, {}),
    )

    def analyze_scale(*, scale, snapshot, facts, target_rows):
        verdict = (
            m.SCALE_PASS
            if scale == "mamba370m"
            else m.SCALE_BLOCKED
        )
        return [], {
            "result": verdict,
        }

    monkeypatch.setattr(m, "analyze_scale", analyze_scale)

    _, summaries, cross = m.run_gate(
        mamba370m_snapshot=Path("x"),
        mamba14b_snapshot=Path("y"),
    )
    assert summaries["mamba370m"]["result"] == m.SCALE_PASS
    assert summaries["mamba14b"]["result"] == m.SCALE_BLOCKED
    assert cross["all_scales_pass"] is False
    assert cross["result"] == m.RESULT_BLOCKED


def test_write_outputs_has_exact_seven_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    scale_rows = {
        "mamba370m": [{"scale": "mamba370m", "anchor_name": "A_IDENTITY"}],
        "mamba14b": [{"scale": "mamba14b", "anchor_name": "A_IDENTITY"}],
    }
    scale_summaries = {
        "mamba370m": {"result": m.SCALE_PASS},
        "mamba14b": {"result": m.SCALE_PASS},
    }
    cross = {
        "result": m.RESULT_PASS,
        "all_scales_pass": True,
    }

    m.write_outputs(
        output_dir=out,
        scale_rows=scale_rows,
        scale_summaries=scale_summaries,
        cross=cross,
    )

    assert {
        p.name for p in out.iterdir() if p.is_file()
    } == {
        "mamba370m_anchor_manifest.jsonl",
        "mamba370m_eligibility_summary.json",
        "mamba14b_anchor_manifest.jsonl",
        "mamba14b_eligibility_summary.json",
        "cross_scale_summary.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
    }

    sums = (out / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for name in (
        "mamba370m_anchor_manifest.jsonl",
        "mamba370m_eligibility_summary.json",
        "mamba14b_anchor_manifest.jsonl",
        "mamba14b_eligibility_summary.json",
        "cross_scale_summary.json",
        "artifact_manifest.json",
    ):
        assert name in sums


def test_source_contains_no_scientific_execution_calls() -> None:
    source = Path(m.__file__).read_text(encoding="utf-8")
    forbidden = (
        ".forward(",
        ".backward(",
        "torch.cuda",
        "selected_checkpoint.pt",
        "load_representative_model",
        "correct_class_logit_margin",
        "D_BEH",
        "p_value",
    )
    for token in forbidden:
        assert token not in source


def _repository_auth_git_stub(branch: str, expected_head: str):
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return branch
        if args == ("rev-parse", "HEAD"):
            return expected_head
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", f"HEAD:{m.PLAN_PATH}"):
            return m.PLAN_BLOB
        raise AssertionError(f"unexpected git call: {args!r}")

    return fake_git


def test_authenticate_repo_allows_kaggle_detached_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_head = "test-detached-head"
    monkeypatch.setattr(
        m,
        "git",
        _repository_auth_git_stub("", expected_head),
    )
    monkeypatch.setattr(m, "git_rc", lambda *args: 0)

    m.authenticate_repo(expected_head)


def test_authenticate_repo_allows_expected_named_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_head = "test-named-head"
    monkeypatch.setattr(
        m,
        "git",
        _repository_auth_git_stub(m.EXPECTED_BRANCH, expected_head),
    )
    monkeypatch.setattr(m, "git_rc", lambda *args: 0)

    m.authenticate_repo(expected_head)


def test_authenticate_repo_rejects_other_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        m,
        "git",
        lambda *args: "unexpected-branch",
    )

    with pytest.raises(
        m.LowDisplacementEligibilityError,
        match=r"BRANCH_MISMATCH:unexpected-branch",
    ):
        m.authenticate_repo("test-head")

def test_factor2_structural_hashes_and_report_are_pinned() -> None:
    assert m.PLAN_BLOB == "90ec869dde6cad2131411ef74beef3328c42c4e7"
    assert m.EXPECTED_SOURCE_SHA256 == (
        "05026973b2ec61847c85d6aab800eada130aad9e9c7edb14a4d8d88f41544c4e"
    )
    assert m.EXPECTED_ROWS_SHA256 == (
        "08da7b3b1d9d92f189b6481abd0b889aeac908519eeae804b8574328ce497f51"
    )
    assert m.EXPECTED_STRUCTURAL_MANIFEST_SHA256 == (
        "ffa42c683f76d8efdb77cb30404f63a6d0dc3915f5b27953ae215ef1f45aeaa9"
    )
    assert m.OUTPUT_DIR.as_posix() == (
        "reports/"
        "reason_router_gen4_mamba370m14b_factor2_small_alpha_"
        "tokenizer_anchor_eligibility_v1"
    )
