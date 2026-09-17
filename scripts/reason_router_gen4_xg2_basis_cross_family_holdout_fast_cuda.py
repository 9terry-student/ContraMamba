from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts import (
    reason_router_gen4_family_subspace_sensitivity_fast_cuda
    as prior,
)
from scripts import (
    reason_router_gen4_xg2_basis_cross_family_holdout_tokenizer_anchor_eligibility
    as eligibility,
)
from scripts import (
    reason_router_gen4_xg1_tokenizer_anchor_eligibility
    as tokenizer_gate,
)
from scripts import (
    reason_router_gen4_six_cell_tier2_inference_adapter
    as adapter,
)
from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1
    as phase1,
)
from scripts import (
    reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase2
    as phase2,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg2-basis-holdout"

SCOPE_FREEZE_COMMIT = (
    "c5679b8a22f103956a46cbc665354b4f9cd7063a"
)
PREPARATION_FREEZE_COMMIT = (
    "69b830b16d5b97745957c049cbc9eca14c5e5209"
)

PRIOR_RUNNER_PATH = (
    "scripts/reason_router_gen4_family_subspace_sensitivity_fast_cuda.py"
)
PRIOR_RUNNER_BLOB = (
    "03f3bf1482913bce30bfdb665ab223a67e6e4159"
)

PREPARATION_PATHS = (
    "data/reason_router_gen4_xg2_basis_cross_family_holdout_v1",
    (
        "reports/"
        "reason_router_gen4_xg2_basis_cross_family_holdout_"
        "tokenizer_eligibility_v1"
    ),
    "scripts/build_reason_router_gen4_xg2_basis_cross_family_holdout.py",
    (
        "scripts/"
        "reason_router_gen4_xg2_basis_cross_family_holdout_"
        "tokenizer_anchor_eligibility.py"
    ),
    "scripts/validate_reason_router_gen4_xg2_basis_cross_family_holdout.py",
    "tests/test_reason_router_gen4_xg2_basis_cross_family_holdout.py",
    (
        "tests/"
        "test_reason_router_gen4_xg2_basis_cross_family_holdout_"
        "tokenizer_anchor_eligibility.py"
    ),
)

FAMILIES = ("xg2", "xg4")
SOURCE_PAIR_COUNT = 300
SUBSPACE_DIM = prior.SUBSPACE_DIM
EPSILON = prior.EPSILON

FORWARDS_PER_DIRECTION = prior.FORWARDS_PER_DIRECTION
FORWARDS_PER_PAIR = (
    2 * SUBSPACE_DIM * FORWARDS_PER_DIRECTION
)
SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY = (
    SOURCE_PAIR_COUNT * FORWARDS_PER_PAIR
)
BASELINE_FORWARD_BUDGET_THIS_RUN = 0

HOLDOUT_ROOT = Path(
    "data/reason_router_gen4_xg2_basis_cross_family_holdout_v1"
)
ELIGIBILITY_ROOT = Path(
    "reports/"
    "reason_router_gen4_xg2_basis_cross_family_holdout_"
    "tokenizer_eligibility_v1"
)

FROZEN_INPUT = {
    "xg2": {
        "source_sha256":
            "aaf0ad1f049feee7ee2f1893d86ce6410a9f3ded0ef8fb30fb7df570993d2ef2",
        "rows_sha256":
            "399fc00f7cd20ae9d46305224dfbc74cd1064e1d0276df42bf5d1db5b389e558",
        "structural_manifest_sha256":
            "4e0451caebabbf5df86a522c8fa9ec74b539de77316ad738024c0a20253f4365",
        "eligibility_summary_sha256":
            "ffe762fb7fc1ed5bd9b6ba50902f396672d3f34006b8cf8781349d5b51824662",
        "anchor_manifest_sha256":
            "7c511ed89013067fbc9af87f74336c749ebe7a3bdca7592960ac1e46a8922ef0",
    },
    "xg4": {
        "source_sha256":
            "e0dde08785de0ff5702ef004696eb4e218ca9887f72eff1fe6573fd82286a6d7",
        "rows_sha256":
            "4e3b31dfc0aaf008f62e0f91e2e26f52bdd5a0d39d12f247d4c9158d4dddd274",
        "structural_manifest_sha256":
            "a434443da59705620fb7efd3251d1398c1d488f1cb62e6548c51a6697b454b0d",
        "eligibility_summary_sha256":
            "dbf5cfc022438ff5fc08601fbb5fc04852a4bc20c5b81a3a6b9203065a87f90e",
        "anchor_manifest_sha256":
            "48c71b9ec42f52960a07f31d88835f3992880d1bf96683e5cbaef5dbc6fb9b67",
    },
}

PROBE_SEED_SCHEMA = (
    "gen4-xg2-basis-cross-family-holdout-probe-seed-v1"
)
ITEM_SCHEMA = (
    "gen4-xg2-basis-cross-family-holdout-item-v1"
)
SUMMARY_SCHEMA = (
    "gen4-xg2-basis-cross-family-holdout-summary-v1"
)
MANIFEST_SCHEMA = (
    "gen4-xg2-basis-cross-family-holdout-manifest-v1"
)
RESULT_PASS = (
    "PASS_XG2_BASIS_CROSS_FAMILY_HOLDOUT_OBSERVATION"
)

ITEM_FILE = "xg2_basis_cross_family_items.jsonl"
SUMMARY_FILE = "xg2_basis_cross_family_summary.json"
MANIFEST_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class XG2BasisCrossFamilyHoldoutError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise XG2BasisCrossFamilyHoldoutError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XG2BasisCrossFamilyHoldoutError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def _git_is_ancestor(
    ancestor: str,
    descendant: str,
) -> bool:
    return (
        subprocess.call(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(
        branch in {"", EXPECTED_BRANCH},
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        head == expected_head,
        f"HEAD_MISMATCH:{head}",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )

    for ancestor, label in (
        (SCOPE_FREEZE_COMMIT, "SCOPE_FREEZE"),
        (
            PREPARATION_FREEZE_COMMIT,
            "PREPARATION_FREEZE",
        ),
        (
            prior.FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
            "FINITE_DIFFERENCE_BASIS_CORRECTION",
        ),
        (
            prior.PHASE1_ARTIFACT_FREEZE_COMMIT,
            "PHASE1_ARTIFACT_FREEZE",
        ),
    ):
        require(
            _git_is_ancestor(
                ancestor,
                expected_head,
            ),
            f"{label}_NOT_ANCESTOR",
        )

    observed_prior_blob = git(
        "rev-parse",
        f"HEAD:{PRIOR_RUNNER_PATH}",
    )
    require(
        observed_prior_blob == PRIOR_RUNNER_BLOB,
        (
            "PRIOR_RUNNER_BLOB_DRIFT:"
            f"{observed_prior_blob}"
        ),
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            PREPARATION_FREEZE_COMMIT,
            expected_head,
            "--",
            *PREPARATION_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "PREPARATION_TREE_DRIFT",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            prior.PHASE1_ARTIFACT_FREEZE_COMMIT,
            expected_head,
            "--",
            prior.PHASE1_ARTIFACT_ROOT.as_posix(),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "FROZEN_PHASE1_ARTIFACT_TREE_DRIFT",
    )

    rc = subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            prior.FINITE_DIFFERENCE_BASIS_CORRECTION_COMMIT,
            expected_head,
            "--",
            *prior.REUSED_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        rc == 0,
        "REUSED_RUNTIME_DEPENDENCY_DRIFT",
    )


def _expected_pairs(
    family: str,
) -> tuple[str, ...]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    return tuple(
        f"{family}_fact_{index:03d}"
        for index in range(601, 901)
    )


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        path.read_text(
            encoding="utf-8-sig"
        ).splitlines(),
        1,
    ):
        if not line.strip():
            continue

        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"JSONL_OBJECT_REQUIRED:{path}:{line_no}",
        )
        rows.append(value)

    return rows


def _eligibility_paths(
    family: str,
) -> tuple[Path, Path]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    base = ELIGIBILITY_ROOT / family

    return (
        base / "tokenizer_anchor_manifest.jsonl",
        base / "summary.json",
    )


def load_frozen_anchor_manifest(
    family: str,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    cfg = FROZEN_INPUT[family]
    anchor_rel, summary_rel = (
        _eligibility_paths(family)
    )

    anchor_path = root / anchor_rel
    summary_path = root / summary_rel

    require(
        anchor_path.is_file(),
        f"ANCHOR_MANIFEST_MISSING:{family}",
    )
    require(
        summary_path.is_file(),
        f"ELIGIBILITY_SUMMARY_MISSING:{family}",
    )

    require(
        phase2.sha256_file(anchor_path)
        == cfg["anchor_manifest_sha256"],
        f"ANCHOR_MANIFEST_SHA256:{family}",
    )
    require(
        phase2.sha256_file(summary_path)
        == cfg["eligibility_summary_sha256"],
        f"ELIGIBILITY_SUMMARY_SHA256:{family}",
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("schema_version")
        == eligibility.SUMMARY_SCHEMA,
        f"ELIGIBILITY_SCHEMA:{family}",
    )
    require(
        summary.get("family_key") == family,
        f"ELIGIBILITY_FAMILY:{family}",
    )
    require(
        summary.get(
            "primary_complete_pair_prefix_feasibility"
        )
        == "PASS_300_OF_300",
        f"ELIGIBILITY_RESULT:{family}",
    )
    require(
        summary.get(
            "complete_source_pair_count"
        )
        == SOURCE_PAIR_COUNT,
        f"ELIGIBILITY_PAIR_COUNT:{family}",
    )
    require(
        summary.get(
            "required_anchor_row_count"
        )
        == 1800,
        f"ELIGIBILITY_ANCHOR_COUNT:{family}",
    )
    require(
        summary.get("model_forward_count") == 0,
        f"ELIGIBILITY_MODEL_FORWARD:{family}",
    )
    require(
        summary.get("checkpoint_load_count") == 0,
        f"ELIGIBILITY_CHECKPOINT_LOAD:{family}",
    )
    require(
        summary.get("gpu_used") is False,
        f"ELIGIBILITY_GPU:{family}",
    )
    require(
        summary.get(
            "scientific_outcomes_observed"
        )
        is False,
        f"ELIGIBILITY_SCIENCE:{family}",
    )
    require(
        summary.get("tokenizer_executed") is True,
        f"ELIGIBILITY_TOKENIZER:{family}",
    )
    require(
        summary.get(
            "anchor_manifest_sha256"
        )
        == cfg["anchor_manifest_sha256"],
        f"ELIGIBILITY_ANCHOR_IDENTITY:{family}",
    )

    provenance = summary.get(
        "provenance",
        {},
    )
    require(
        provenance.get("scope_freeze_commit")
        == SCOPE_FREEZE_COMMIT,
        f"ELIGIBILITY_SCOPE_FREEZE:{family}",
    )
    require(
        provenance.get("head")
        == SCOPE_FREEZE_COMMIT,
        f"ELIGIBILITY_EXECUTION_HEAD:{family}",
    )

    frozen = summary.get(
        "frozen_input",
        {},
    )
    require(
        frozen.get("source_facts_sha256")
        == cfg["source_sha256"],
        f"SOURCE_SHA256:{family}",
    )
    require(
        frozen.get("rows_sha256")
        == cfg["rows_sha256"],
        f"ROWS_SHA256:{family}",
    )
    require(
        frozen.get(
            "structural_manifest_sha256"
        )
        == cfg[
            "structural_manifest_sha256"
        ],
        f"STRUCTURAL_MANIFEST_SHA256:{family}",
    )
    require(
        frozen.get("pair_id_first")
        == f"{family}_fact_601",
        f"FIRST_PAIR:{family}",
    )
    require(
        frozen.get("pair_id_last")
        == f"{family}_fact_900",
        f"LAST_PAIR:{family}",
    )

    rows = _read_jsonl(anchor_path)

    require(
        len(rows) == 1800,
        f"ANCHOR_MANIFEST_ROW_COUNT:{family}",
    )
    require(
        Counter(
            row["anchor_name"]
            for row in rows
        )
        == Counter({
            "A_IDENTITY": 1200,
            "A_NAME": 600,
        }),
        f"ANCHOR_NAME_COUNTS:{family}",
    )

    lookup: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for index, row in enumerate(rows):
        require(
            row.get("schema_version")
            == eligibility.ANCHOR_SCHEMA,
            f"ANCHOR_SCHEMA:{family}:{index}",
        )
        require(
            row.get("family_key") == family,
            f"ANCHOR_FAMILY:{family}:{index}",
        )
        require(
            row.get("post4_eligible") is True,
            f"ANCHOR_NOT_ELIGIBLE:{family}:{index}",
        )
        require(
            row.get("exclusion_code") is None,
            f"ANCHOR_EXCLUSION:{family}:{index}",
        )

        key = (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        )
        require(
            key not in lookup,
            f"ANCHOR_DUPLICATE:{family}:{key}",
        )
        lookup[key] = row

    require(
        len(lookup) == 1800,
        f"ANCHOR_EVENT_KEY_COUNT:{family}",
    )

    for pair in _expected_pairs(family):
        for cell in (
            "C0_SHAM",
            "C2_NAME",
        ):
            identity = lookup[
                (
                    pair,
                    cell,
                    "A_IDENTITY",
                )
            ]
            name = lookup[
                (
                    pair,
                    cell,
                    "A_NAME",
                )
            ]
            require(
                identity[
                    "absolute_anchor_token_index"
                ]
                == name[
                    "absolute_anchor_token_index"
                ],
                (
                    "TARGET_IDENTITY_NAME_MISMATCH:"
                    f"{pair}:{cell}"
                ),
            )

    return rows


def _pair_order(
    family: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()

    for row in rows:
        pair = str(
            row["source_pair_id"]
        )
        if pair not in seen:
            seen.add(pair)
            order.append(pair)

    expected = _expected_pairs(family)

    require(
        tuple(order) == expected,
        f"PAIR_ORDER:{family}",
    )

    return tuple(order)


def validate_family_population(
    family: str,
    rows: Sequence[Mapping[str, Any]],
    encoded: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    normalized = (
        adapter.validate_gen4_rows(
            rows,
            require_canonical_shape=True,
        )
    )

    pairs = _pair_order(
        family,
        normalized,
    )

    require(
        list(encoded["row_id"])
        == [
            str(row["row_id"])
            for row in normalized
        ],
        f"ENCODED_ROW_ORDER:{family}",
    )
    require(
        list(encoded["source_pair_id"])
        == [
            str(row["source_pair_id"])
            for row in normalized
        ],
        f"ENCODED_PAIR_ORDER:{family}",
    )
    require(
        list(encoded["contrast_cell_id"])
        == [
            str(row["contrast_cell_id"])
            for row in normalized
        ],
        f"ENCODED_CELL_ORDER:{family}",
    )

    input_ids = encoded["input_ids"]
    attention = encoded[
        "attention_mask"
    ]
    claim_mask = encoded[
        "claim_mask"
    ]
    evidence_mask = encoded[
        "evidence_mask"
    ]

    require(
        torch.is_tensor(input_ids),
        f"INPUT_IDS_TENSOR:{family}",
    )
    require(
        tuple(input_ids.shape)
        == (
            1800,
            adapter.MAX_LENGTH,
        ),
        f"INPUT_IDS_SHAPE:{family}",
    )

    row_id_index = {
        str(row["row_id"]): index
        for index, row in enumerate(
            normalized
        )
    }
    require(
        len(row_id_index) == 1800,
        f"ROW_ID_CARDINALITY:{family}",
    )

    parent = (
        phase1.base.prevalence_eq.parent
    )
    events = parent.event_lookup(
        event_rows
    )
    parent.validate_transport_event_plan(
        pairs,
        events,
    )

    for event in event_rows:
        row_id = str(event["row_id"])
        require(
            row_id in row_id_index,
            f"EVENT_ROW_ID:{family}:{row_id}",
        )

        index = row_id_index[row_id]

        anchor = int(
            event[
                "absolute_anchor_token_index"
            ]
        )
        evidence_anchor = int(
            event[
                "anchor_evidence_token_index"
            ]
        )
        terminal = int(
            event["terminal_index"]
        )

        claim_count = int(
            claim_mask[index]
            .sum()
            .item()
        )
        evidence_count = int(
            evidence_mask[index]
            .sum()
            .item()
        )

        require(
            anchor
            == claim_count
            + 1
            + evidence_anchor,
            (
                "ANCHOR_COORDINATE:"
                f"{family}:{row_id}"
            ),
        )
        require(
            terminal
            == claim_count
            + evidence_count,
            (
                "TERMINAL_COORDINATE:"
                f"{family}:{row_id}"
            ),
        )
        require(
            anchor + 4
            <= terminal - 1,
            f"POST4_RULE:{family}:{row_id}",
        )
        require(
            int(
                attention[index]
                .sum()
                .item()
            )
            == terminal + 1,
            (
                "ATTENTION_TERMINAL:"
                f"{family}:{row_id}"
            ),
        )

    return pairs


def load_family_inputs(
    family: str,
    tokenizer_snapshot: str | Path | None,
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    facts, rows, _manifest = (
        eligibility.load_family(
            family,
            root,
        )
    )

    require(
        len(facts) == SOURCE_PAIR_COUNT,
        f"FACT_COUNT:{family}",
    )
    require(
        len(rows) == 1800,
        f"ROW_COUNT:{family}",
    )

    tokenizer, _provenance = (
        tokenizer_gate
        .load_canonical_analysis_tokenizer(
            tokenizer_snapshot
        )
    )

    encoded = (
        adapter.encode_gen4_rows(
            rows,
            tokenizer,
        )
    )

    event_rows = (
        load_frozen_anchor_manifest(
            family,
            root,
        )
    )

    validate_family_population(
        family,
        rows,
        encoded,
        event_rows,
    )

    return (
        rows,
        encoded,
        event_rows,
    )


def _load_frozen_bases() -> dict[str, Any]:
    loaded = (
        prior._load_all_phase1_and_bases()
    )

    require(
        set(loaded) == set(FAMILIES),
        "BASIS_FAMILY_SET",
    )

    for family in FAMILIES:
        require(
            loaded[family]["plan_sha256"]
            == prior.PHASE1_PLAN_SHA256[
                family
            ],
            f"BASIS_PLAN_SHA256:{family}",
        )

    return loaded


def _probe_seed(
    family: str,
    index: int,
    pair: str,
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    require(
        pair
        == _expected_pairs(family)[index],
        f"PROBE_SEED_PAIR:{index}",
    )

    anchors = phase1._anchors_for_pair(
        pair,
        events,
    )

    return {
        "schema_version":
            PROBE_SEED_SCHEMA,
        "family_key": family,
        "source_pair_id": pair,
        "holdout_pair_index": index,
        "target_plus_anchor":
            int(anchors["tp"]),
        "target_minus_anchor":
            int(anchors["tm"]),
        "reference_plus_anchor":
            int(anchors["rp"]),
        "reference_minus_anchor":
            int(anchors["rm"]),
    }


def _run_pair(
    family: str,
    seed: Mapping[str, Any],
    *,
    xg2_basis: torch.Tensor,
    xg4_basis: torch.Tensor,
    model: Any,
    runtime_ctx: Mapping[str, Any],
    trace_code: Any,
    trace_line: int,
    encoded: Mapping[str, Any],
    row_index: Mapping[
        tuple[str, str],
        int,
    ],
    events: Mapping[
        tuple[str, str, str],
        Mapping[str, Any],
    ],
    budget: Any,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    require(
        xg2_basis.ndim == 2
        and int(xg2_basis.shape[1])
        == SUBSPACE_DIM,
        "XG2_BASIS_SHAPE",
    )
    require(
        xg4_basis.ndim == 2
        and int(xg4_basis.shape[1])
        == SUBSPACE_DIM,
        "XG4_BASIS_SHAPE",
    )
    require(
        int(xg2_basis.shape[0])
        == int(xg4_basis.shape[0]),
        "BASIS_WIDTH_MISMATCH",
    )

    xg2_probes: list[
        dict[str, Any]
    ] = []
    xg4_probes: list[
        dict[str, Any]
    ] = []

    for basis_index in range(
        SUBSPACE_DIM
    ):
        xg2_probes.append(
            prior._run_direction_j(
                family,
                seed,
                xg2_basis[
                    :,
                    basis_index,
                ],
                basis_family="xg2",
                basis_index=basis_index,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=budget,
            )
        )

    for basis_index in range(
        SUBSPACE_DIM
    ):
        xg4_probes.append(
            prior._run_direction_j(
                family,
                seed,
                xg4_basis[
                    :,
                    basis_index,
                ],
                basis_family="xg4",
                basis_index=basis_index,
                model=model,
                runtime_ctx=runtime_ctx,
                trace_code=trace_code,
                trace_line=trace_line,
                encoded=encoded,
                row_index=row_index,
                events=events,
                budget=budget,
            )
        )

    e_xg2 = sum(
        float(row["J_squared"])
        for row in xg2_probes
    ) / float(SUBSPACE_DIM)

    e_xg4 = sum(
        float(row["J_squared"])
        for row in xg4_probes
    ) / float(SUBSPACE_DIM)

    q_value = e_xg2 - e_xg4

    require(
        all(
            math.isfinite(value)
            for value in (
                e_xg2,
                e_xg4,
                q_value,
            )
        ),
        (
            "NONFINITE_PAIR_ENDPOINT:"
            f"{seed['source_pair_id']}"
        ),
    )

    item = dict(seed)
    item[
        "probe_seed_schema_version"
    ] = item["schema_version"]
    item["schema_version"] = ITEM_SCHEMA

    item["scope_freeze_commit"] = (
        SCOPE_FREEZE_COMMIT
    )
    item[
        "preparation_freeze_commit"
    ] = PREPARATION_FREEZE_COMMIT
    item[
        "phase1_artifact_freeze_commit"
    ] = prior.PHASE1_ARTIFACT_FREEZE_COMMIT

    item["epsilon"] = EPSILON
    item["subspace_dim"] = (
        SUBSPACE_DIM
    )
    item["xg2_basis_probes"] = (
        xg2_probes
    )
    item["xg4_basis_probes"] = (
        xg4_probes
    )
    item["E_XG2"] = e_xg2
    item["E_XG4"] = e_xg4
    item["Q"] = q_value

    item[
        "baseline_model_forward_count_this_run"
    ] = 0
    item[
        "scientific_model_forward_count_this_run"
    ] = FORWARDS_PER_PAIR

    return item


def _validate_items(
    family: str,
    items: Sequence[Mapping[str, Any]],
) -> None:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )
    require(
        len(items) == SOURCE_PAIR_COUNT,
        "ITEM_COUNT",
    )

    for index, (
        expected_pair,
        raw,
    ) in enumerate(
        zip(
            _expected_pairs(family),
            items,
            strict=True,
        )
    ):
        row = dict(raw)

        require(
            row.get("schema_version")
            == ITEM_SCHEMA,
            f"ITEM_SCHEMA:{index}",
        )
        require(
            row.get(
                "probe_seed_schema_version"
            )
            == PROBE_SEED_SCHEMA,
            f"PROBE_SEED_SCHEMA:{index}",
        )
        require(
            row.get("family_key")
            == family,
            f"ITEM_FAMILY:{index}",
        )
        require(
            row.get("source_pair_id")
            == expected_pair,
            f"PAIR_ORDER:{index}",
        )
        require(
            row.get("holdout_pair_index")
            == index,
            f"HOLDOUT_INDEX:{index}",
        )
        require(
            row.get("scope_freeze_commit")
            == SCOPE_FREEZE_COMMIT,
            f"SCOPE_FREEZE:{index}",
        )
        require(
            row.get(
                "preparation_freeze_commit"
            )
            == PREPARATION_FREEZE_COMMIT,
            f"PREPARATION_FREEZE:{index}",
        )
        require(
            row.get(
                "phase1_artifact_freeze_commit"
            )
            == (
                prior
                .PHASE1_ARTIFACT_FREEZE_COMMIT
            ),
            f"PHASE1_FREEZE:{index}",
        )
        require(
            float(row["epsilon"])
            == EPSILON,
            f"EPSILON:{index}",
        )
        require(
            int(row["subspace_dim"])
            == SUBSPACE_DIM,
            f"SUBSPACE_DIM:{index}",
        )
        require(
            row[
                "baseline_model_forward_count_this_run"
            ]
            == 0,
            (
                "BASELINE_FORWARD_COUNT:"
                f"{index}"
            ),
        )
        require(
            row[
                "scientific_model_forward_count_this_run"
            ]
            == FORWARDS_PER_PAIR,
            (
                "SCIENTIFIC_FORWARD_COUNT:"
                f"{index}"
            ),
        )

        xg2_probes = row[
            "xg2_basis_probes"
        ]
        xg4_probes = row[
            "xg4_basis_probes"
        ]

        require(
            isinstance(
                xg2_probes,
                list,
            )
            and len(xg2_probes)
            == SUBSPACE_DIM,
            f"XG2_PROBE_COUNT:{index}",
        )
        require(
            isinstance(
                xg4_probes,
                list,
            )
            and len(xg4_probes)
            == SUBSPACE_DIM,
            f"XG4_PROBE_COUNT:{index}",
        )

        for basis_index, probe in enumerate(
            xg2_probes
        ):
            prior._validate_direction_probe(
                probe,
                expected_basis_family="xg2",
                expected_basis_index=(
                    basis_index
                ),
            )

        for basis_index, probe in enumerate(
            xg4_probes
        ):
            prior._validate_direction_probe(
                probe,
                expected_basis_family="xg4",
                expected_basis_index=(
                    basis_index
                ),
            )

        e_xg2 = float(row["E_XG2"])
        e_xg4 = float(row["E_XG4"])
        q_value = float(row["Q"])

        expected_xg2 = sum(
            float(probe["J_squared"])
            for probe in xg2_probes
        ) / float(SUBSPACE_DIM)

        expected_xg4 = sum(
            float(probe["J_squared"])
            for probe in xg4_probes
        ) / float(SUBSPACE_DIM)

        require(
            e_xg2 == expected_xg2,
            f"E_XG2_IDENTITY:{index}",
        )
        require(
            e_xg4 == expected_xg4,
            f"E_XG4_IDENTITY:{index}",
        )
        require(
            q_value == e_xg2 - e_xg4,
            f"Q_IDENTITY:{index}",
        )


def _basis_summary(
    loaded: Mapping[str, Any],
) -> dict[str, Any]:
    return prior._basis_summary(
        loaded
    )


def _write_outputs(
    output_dir: Path,
    *,
    family: str,
    items: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, str]:
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    _validate_items(
        family,
        items,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    payloads = {
        ITEM_FILE:
            phase2.jsonl_bytes(items),
        SUMMARY_FILE:
            phase2.canonical_json_bytes(
                summary
            ),
    }

    hashes: dict[str, str] = {}

    for name, raw in payloads.items():
        path = output_dir / name
        path.write_bytes(raw)
        hashes[name] = (
            phase2.sha256_bytes(raw)
        )

    manifest = {
        "schema_version":
            MANIFEST_SCHEMA,
        "files": {
            name: {
                "sha256": digest,
                "bytes": int(
                    (
                        output_dir
                        / name
                    )
                    .stat()
                    .st_size
                ),
            }
            for name, digest in sorted(
                hashes.items()
            )
        },
    }

    manifest_raw = (
        phase2.canonical_json_bytes(
            manifest
        )
    )
    (
        output_dir / MANIFEST_FILE
    ).write_bytes(
        manifest_raw
    )
    hashes[MANIFEST_FILE] = (
        phase2.sha256_bytes(
            manifest_raw
        )
    )

    checksum_raw = "".join(
        f"{digest}  {name}\n"
        for name, digest in sorted(
            hashes.items()
        )
    ).encode("utf-8")

    (
        output_dir / CHECKSUM_FILE
    ).write_bytes(
        checksum_raw
    )

    return hashes


def validate_artifact(
    output_dir: Path,
    family: str,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    manifest_path = (
        output_dir / MANIFEST_FILE
    )
    checksum_path = (
        output_dir / CHECKSUM_FILE
    )

    require(
        manifest_path.is_file(),
        "MANIFEST_MISSING",
    )
    require(
        checksum_path.is_file(),
        "CHECKSUM_MISSING",
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8-sig"
        )
    )
    require(
        manifest.get("schema_version")
        == MANIFEST_SCHEMA,
        "MANIFEST_SCHEMA",
    )

    files = manifest.get("files")
    require(
        isinstance(files, dict),
        "MANIFEST_FILES",
    )
    require(
        set(files)
        == {
            ITEM_FILE,
            SUMMARY_FILE,
        },
        "MANIFEST_FILE_SET",
    )

    observed_hashes: dict[
        str,
        str,
    ] = {}

    for name in (
        ITEM_FILE,
        SUMMARY_FILE,
    ):
        path = output_dir / name
        require(
            path.is_file(),
            f"ARTIFACT_MISSING:{name}",
        )

        observed_sha = (
            phase2.sha256_file(path)
        )
        observed_bytes = int(
            path.stat().st_size
        )

        require(
            observed_sha
            == files[name]["sha256"],
            f"ARTIFACT_SHA256:{name}",
        )
        require(
            observed_bytes
            == int(files[name]["bytes"]),
            f"ARTIFACT_BYTES:{name}",
        )
        observed_hashes[name] = (
            observed_sha
        )

    observed_hashes[
        MANIFEST_FILE
    ] = phase2.sha256_file(
        manifest_path
    )

    checksum_rows: dict[
        str,
        str,
    ] = {}

    for line in (
        checksum_path
        .read_text(
            encoding="utf-8-sig"
        )
        .splitlines()
    ):
        if not line.strip():
            continue

        digest, name = line.split(
            "  ",
            1,
        )
        require(
            name not in checksum_rows,
            f"CHECKSUM_DUPLICATE:{name}",
        )
        checksum_rows[name] = digest

    require(
        checksum_rows
        == {
            name: digest
            for name, digest in sorted(
                observed_hashes.items()
            )
        },
        "CHECKSUM_CONTENT",
    )

    items = _read_jsonl(
        output_dir / ITEM_FILE
    )
    _validate_items(
        family,
        items,
    )

    summary = json.loads(
        (
            output_dir
            / SUMMARY_FILE
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    require(
        summary.get("schema_version")
        == SUMMARY_SCHEMA,
        "SUMMARY_SCHEMA",
    )
    require(
        summary.get("result")
        == RESULT_PASS,
        "SUMMARY_RESULT",
    )
    require(
        summary.get("family_key")
        == family,
        "SUMMARY_FAMILY",
    )
    require(
        summary.get("source_pair_count")
        == SOURCE_PAIR_COUNT,
        "SUMMARY_PAIR_COUNT",
    )
    require(
        summary.get("pair_id_first")
        == f"{family}_fact_601",
        "SUMMARY_FIRST_PAIR",
    )
    require(
        summary.get("pair_id_last")
        == f"{family}_fact_900",
        "SUMMARY_LAST_PAIR",
    )
    require(
        summary.get(
            "scope_freeze_commit"
        )
        == SCOPE_FREEZE_COMMIT,
        "SUMMARY_SCOPE_FREEZE",
    )
    require(
        summary.get(
            "preparation_freeze_commit"
        )
        == PREPARATION_FREEZE_COMMIT,
        "SUMMARY_PREPARATION_FREEZE",
    )
    require(
        float(summary["epsilon"])
        == EPSILON,
        "SUMMARY_EPSILON",
    )
    require(
        int(summary["subspace_dim"])
        == SUBSPACE_DIM,
        "SUMMARY_SUBSPACE_DIM",
    )
    require(
        summary.get(
            "baseline_model_forward_count_this_run"
        )
        == 0,
        "SUMMARY_BASELINE_FORWARD_COUNT",
    )
    require(
        summary.get(
            "scientific_model_forward_count_this_run"
        )
        == (
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
        ),
        "SUMMARY_SCIENTIFIC_FORWARD_COUNT",
    )
    require(
        summary.get(
            "basis_reconstruction_before_model_setup"
        )
        is True,
        "SUMMARY_BASIS_BEFORE_MODEL",
    )
    require(
        summary.get(
            "response_used_for_basis_construction"
        )
        is False,
        "SUMMARY_RESPONSE_BASIS_BOUNDARY",
    )
    require(
        summary.get(
            "primary_endpoint_Q_observed"
        )
        is True,
        "SUMMARY_Q_OBSERVED",
    )
    require(
        summary.get(
            "primary_inference_executed"
        )
        is False,
        "SUMMARY_INFERENCE_BOUNDARY",
    )
    require(
        summary.get(
            "holm_correction_executed"
        )
        is False,
        "SUMMARY_HOLM_BOUNDARY",
    )
    require(
        summary.get(
            "training_executed"
        )
        is False
        and summary.get(
            "backward_executed"
        )
        is False
        and summary.get(
            "task_heads_executed"
        )
        is False
        and summary.get(
            "logits_read"
        )
        is False,
        "SUMMARY_EXECUTION_BOUNDARY",
    )
    require(
        summary.get(
            "scientific_conclusion"
        )
        is None,
        "SUMMARY_CONCLUSION_BOUNDARY",
    )

    basis = summary.get(
        "basis_reconstruction"
    )
    require(
        isinstance(basis, dict)
        and set(basis)
        == set(FAMILIES),
        "SUMMARY_BASIS_RECONSTRUCTION",
    )

    for basis_family in FAMILIES:
        row = basis[basis_family]
        require(
            row["phase1_plan_sha256"]
            == prior.PHASE1_PLAN_SHA256[
                basis_family
            ],
            (
                "SUMMARY_PLAN_SHA256:"
                f"{basis_family}"
            ),
        )

    return {
        "summary": summary,
        "items": items,
        "manifest": manifest,
    }


def run_holdout(
    *,
    family: str,
    expected_head: str,
    model_snapshot: Path,
    tokenizer_snapshot: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    require(
        family in FAMILIES,
        f"UNSUPPORTED_FAMILY:{family}",
    )

    authenticate_repo(
        expected_head
    )
    require(
        not output_dir.exists(),
        "OUTPUT_DIR_COLLISION",
    )

    # Basis construction is frozen from prior
    # 301..600 Phase-1 evidence and happens
    # before runtime/model setup.
    loaded = _load_frozen_bases()

    xg2_basis = (
        loaded["xg2"]
        ["basis"]["basis"]
    )
    xg4_basis = (
        loaded["xg4"]
        ["basis"]["basis"]
    )

    (
        phase1.base
        .prevalence_eq
        .backend
        .runtime_gate()
    )

    with (
        phase1.base
        .prevalence_eq
        .backend
        .parent_runtime_rebind()
    ):
        rows, encoded, event_rows = (
            load_family_inputs(
                family,
                tokenizer_snapshot,
            )
        )

        pairs = _pair_order(
            family,
            rows,
        )

        parent = (
            phase1.base
            .prevalence_eq
            .parent
        )
        events = parent.event_lookup(
            event_rows
        )

        parent.validate_transport_event_plan(
            pairs,
            events,
        )

        row_index = (
            parent.build_row_index(
                rows
            )
        )

        trace_code, trace_line = (
            phase1.base
            .prevalence_eq
            .measurement
            ._resolve_and_validate_runtime_binding()
        )

        kernels = (
            phase1.base
            .prevalence_eq
            .kernel_compat
            .load_exact_fast_kernels()
        )

        with (
            phase1.base
            .prevalence_eq
            .kernel_compat
            .exact_transformers_kernel_loader(
                kernels
            )
        ) as constructor_kernel_calls:
            model, checkpoint_sha = (
                parent
                .load_representative_model_external(
                    model_snapshot=(
                        model_snapshot
                    ),
                    checkpoint_path=(
                        checkpoint_path
                    ),
                )
            )

            require(
                checkpoint_sha
                == (
                    phase1.base
                    .prevalence_eq
                    .extraction
                    .REPRESENTATIVE_CHECKPOINT_SHA256
                ),
                "CHECKPOINT_IDENTITY",
            )

            runtime_ctx = (
                phase1.base
                .prevalence_eq
                .transport_runtime
                .validate_runtime_components(
                    model
                )
            )

        constructor_counts = Counter(
            constructor_kernel_calls
        )

        require(
            set(constructor_counts)
            == {
                "causal-conv1d",
                "mamba-ssm",
            },
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_NAMES:"
                f"{dict(constructor_counts)}"
            ),
        )
        require(
            constructor_counts[
                "causal-conv1d"
            ]
            > 0
            and constructor_counts[
                "causal-conv1d"
            ]
            == constructor_counts[
                "mamba-ssm"
            ],
            (
                "TRANSFORMERS_CONSTRUCTOR_"
                "KERNEL_CALL_COUNT:"
                f"{dict(constructor_counts)}"
            ),
        )

        (
            phase1.base
            .prevalence_eq
            .kernel_compat
            .validate_transformers_kernel_bindings(
                kernels
            )
        )

        model.to(
            torch.device("cuda:0")
        )
        model.eval()

        require(
            all(
                parameter.device.type
                == "cuda"
                for parameter
                in model.mamba.parameters()
            ),
            "GPU_MODEL_DEVICE",
        )

        fast_capture = (
            phase1.base
            .prevalence_eq
            .backend
            ._make_fast_capture(
                kernels
            )
        )

        original_capture = (
            parent.capture_branch
        )

        budget = (
            parent.ForwardBudget(
                SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY
            )
        )

        items: list[
            dict[str, Any]
        ] = []

        parent.capture_branch = (
            fast_capture
        )

        try:
            for index, pair in enumerate(
                pairs
            ):
                seed = _probe_seed(
                    family,
                    index,
                    pair,
                    events,
                )

                item = _run_pair(
                    family,
                    seed,
                    xg2_basis=xg2_basis,
                    xg4_basis=xg4_basis,
                    model=model,
                    runtime_ctx=runtime_ctx,
                    trace_code=trace_code,
                    trace_line=trace_line,
                    encoded=encoded,
                    row_index=row_index,
                    events=events,
                    budget=budget,
                )

                items.append(item)

            budget.assert_exact()
            torch.cuda.synchronize()

        finally:
            parent.capture_branch = (
                original_capture
            )

    summary = {
        "schema_version":
            SUMMARY_SCHEMA,
        "result":
            RESULT_PASS,
        "family_key":
            family,
        "execution_head":
            expected_head,
        "scope_freeze_commit":
            SCOPE_FREEZE_COMMIT,
        "preparation_freeze_commit":
            PREPARATION_FREEZE_COMMIT,
        "phase1_artifact_freeze_commit":
            prior.PHASE1_ARTIFACT_FREEZE_COMMIT,
        "phase1_artifact_root":
            prior.PHASE1_ARTIFACT_ROOT
            .as_posix(),
        "basis_source_index_range":
            "301_600",
        "probe_source_index_range":
            "601_900",
        "source_pair_count":
            SOURCE_PAIR_COUNT,
        "pair_id_first":
            items[0]["source_pair_id"],
        "pair_id_last":
            items[-1]["source_pair_id"],
        "epsilon":
            EPSILON,
        "subspace_dim":
            SUBSPACE_DIM,
        "basis_reconstruction":
            _basis_summary(loaded),
        (
            "basis_reconstruction_before_"
            "model_setup"
        ):
            True,
        (
            "response_used_for_basis_"
            "construction"
        ):
            False,
        (
            "baseline_model_forward_"
            "count_this_run"
        ):
            0,
        (
            "scientific_model_forward_"
            "count_this_run"
        ):
            SCIENTIFIC_FORWARD_BUDGET_PER_FAMILY,
        "model_forwards_per_pair":
            FORWARDS_PER_PAIR,
        "model_forwards_per_direction":
            FORWARDS_PER_DIRECTION,
        "E_XG2_observed":
            True,
        "E_XG4_observed":
            True,
        "primary_endpoint_Q_observed":
            True,
        "primary_endpoint_definition":
            "Q=E_XG2-E_XG4",
        "primary_inference_executed":
            False,
        "holm_correction_executed":
            False,
        "training_executed":
            False,
        "backward_executed":
            False,
        "task_heads_executed":
            False,
        "logits_read":
            False,
        "scientific_conclusion":
            None,
        "scientific_conclusion_scope":
            (
                "XG2_BASIS_CROSS_FAMILY_"
                "HOLDOUT_OBSERVATION_ONLY"
            ),
        (
            "representative_checkpoint_"
            "sha256"
        ):
            checkpoint_sha,
    }

    _write_outputs(
        output_dir,
        family=family,
        items=items,
        summary=summary,
    )

    validated = validate_artifact(
        output_dir,
        family,
    )

    require(
        validated["summary"]["result"]
        == RESULT_PASS,
        "POSTWRITE_VALIDATION_RESULT",
    )

    return summary


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective 601..900 XG2-basis "
            "cross-family holdout observation. "
            "Reconstructs frozen 301..600 XG2/XG4 "
            "top-5 bases before model setup, then "
            "measures E_XG2, E_XG4 and "
            "Q=E_XG2-E_XG4 at epsilon=0.025. "
            "No new baseline forward and no "
            "inferential test are executed."
        )
    )

    parser.add_argument(
        "--family",
        choices=FAMILIES,
        required=True,
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    args = parse_args(argv)

    summary = run_holdout(
        family=args.family,
        expected_head=args.expected_head,
        model_snapshot=args.model_snapshot,
        tokenizer_snapshot=(
            args.tokenizer_snapshot
        ),
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )

    print(
        "RESULT =",
        summary["result"],
    )
    print(
        "FAMILY =",
        summary["family_key"],
    )
    print(
        "PAIR_ID_FIRST =",
        summary["pair_id_first"],
    )
    print(
        "PAIR_ID_LAST =",
        summary["pair_id_last"],
    )
    print(
        "SCIENTIFIC_MODEL_FORWARD_COUNT =",
        summary[
            "scientific_model_forward_count_this_run"
        ],
    )
    print(
        "BASELINE_MODEL_FORWARD_COUNT =",
        summary[
            "baseline_model_forward_count_this_run"
        ],
    )
    print(
        "PRIMARY_INFERENCE_EXECUTED =",
        summary[
            "primary_inference_executed"
        ],
    )
    print(
        "SCIENTIFIC_CONCLUSION =",
        summary[
            "scientific_conclusion"
        ],
    )


if __name__ == "__main__":
    main()
