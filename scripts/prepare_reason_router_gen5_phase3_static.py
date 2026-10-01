#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from contramamba.labels import FinalLabel  # noqa: E402
from scripts import build_reason_router_gen4_mamba1_five_scale_ladder_holdouts as ladder  # noqa: E402
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as eligibility  # noqa: E402

EXPECTED_BRANCH = "gen5-causal-role-state-ownership"

PHASE3_DESIGN_COMMIT = "ae52dbf98cce21226036ae38121edcdfa6f79d7b"
LABEL_AMENDMENT_COMMIT = "14d2bd7db70fc6000b480221a17fa8475dfefbe5"
STRESSOR_DOMAIN_AMENDMENT_COMMIT = "67e97cae7f6fd20072a9422d0ac40bc6f781ed13"

FROZEN_DEPENDENCY_BLOBS = {
    "scripts/build_reason_router_gen4_mamba1_five_scale_ladder_holdouts.py":
        "4ff5ee70f463cae94d5188999b350efecbe212bb",
    "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py":
        "6c98ce022ca134e385db28851fd364dc6daff423",
}

PRIOR_LINEAGE = (
    {
        "label": "phase1b_restoration",
        "first": 8401,
        "last": 8700,
        "root": Path("data/reason_router_gen5_phase1b_xg1_restoration_confirmation_v1"),
        "source_sha256":
            "a2681a1fe7a76ffa7ba42c08bbe9809bc8e751d282e4909fb84ee65c49bb0245",
        "rows_sha256":
            "8d601c44a25f7733d3613e2ce361fe1add467ae84bd7db7ce41c511802590de2",
    },
    {
        "label": "phase2_ownership_assay",
        "first": 8701,
        "last": 9000,
        "root": Path("data/reason_router_gen5_phase2_xg1_ownership_assay_v1"),
        "source_sha256":
            "f316d7e2a187e90451ff0743829d6c615c063969b66ca51f91e15158bcc1ee06",
        "rows_sha256":
            "529027ba79d3bd9fc4e300fbf317f0d84668152ed4fa0593c55af601e7934976",
    },
)

TRAIN_FIRST = 9001
TRAIN_LAST = 9600
TRAIN_PAIR_COUNT = 600

ASSAY_FIRST = 9601
ASSAY_LAST = 9900
ASSAY_PAIR_COUNT = 300

BASE_ROWS_PER_PAIR = 6
TRAIN_ROWS_PER_PAIR = 7
BASE_TRAIN_ROW_COUNT = TRAIN_PAIR_COUNT * BASE_ROWS_PER_PAIR
LABELED_TRAIN_ROW_COUNT = TRAIN_PAIR_COUNT * TRAIN_ROWS_PER_PAIR
ASSAY_ROW_COUNT = ASSAY_PAIR_COUNT * BASE_ROWS_PER_PAIR

SPLIT_SEED = 16384
DEV_RATIO = 0.2
TRAIN_SPLIT_PAIRS = 480
DEV_SPLIT_PAIRS = 120
TRAIN_SPLIT_ROWS = TRAIN_SPLIT_PAIRS * TRAIN_ROWS_PER_PAIR
DEV_SPLIT_ROWS = DEV_SPLIT_PAIRS * TRAIN_ROWS_PER_PAIR

STRESSOR_DOMAIN_CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C5_TITLE_NAME",
)
NON_STRESSOR_CELLS = (
    "C3_ROLE",
    "C4_PREDICATE",
    "C6_EXPLICIT_DENIAL",
)
BASE_CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C3_ROLE",
    "C4_PREDICATE",
    "C5_TITLE_NAME",
)
TRAINING_CELLS = BASE_CELLS + ("C6_EXPLICIT_DENIAL",)

STRESSOR_ANCHOR_NAME = "A_IDENTITY"
STRESSOR_TARGET_OFFSET = 2
STRESSOR_LAYER = 17

LABEL_BY_CELL = {
    "C0_SHAM": "SUPPORT",
    "C1_TITLE": "NOT_ENTITLED",
    "C2_NAME": "NOT_ENTITLED",
    "C3_ROLE": "NOT_ENTITLED",
    "C4_PREDICATE": "NOT_ENTITLED",
    "C5_TITLE_NAME": "NOT_ENTITLED",
    "C6_EXPLICIT_DENIAL": "REFUTE",
}

EXPECTED_LABEL_TO_ID = {
    "REFUTE": 0,
    "NOT_ENTITLED": 1,
    "SUPPORT": 2,
}

TRAIN_OUTPUT_DIR = Path("data/reason_router_gen5_phase3_xg1_contention_training_v1")
ASSAY_OUTPUT_DIR = Path("data/reason_router_gen5_phase3_xg1_ownership_interaction_assay_v1")

SOURCE_FILE = ladder.SOURCE_FILE
BASE_ROW_FILE = ladder.ROW_FILE
LABELED_ROW_FILE = "phase3_seven_cell_labeled_training.jsonl"
SPLIT_FILE = "pair_split_manifest.json"
STRUCTURAL_MANIFEST_FILE = "structural_manifest.json"
ANCHOR_FILE = "tokenizer_anchor_manifest.jsonl"
TOKENIZER_SUMMARY_FILE = "tokenizer_eligibility_summary.json"
STRESSOR_TARGET_FILE = "stressor_target_manifest.jsonl"
CHECKSUM_FILE = "SHA256SUMS.txt"
PREP_SUMMARY_FILE = "static_preparation_summary.json"

MAX_LENGTH = eligibility.MAX_LENGTH
CLAIM_BUDGET = eligibility.CLAIM_BUDGET
EVIDENCE_BUDGET = eligibility.EVIDENCE_BUDGET

FORBIDDEN_BASE_OUTCOME_FIELDS = ladder.FORBIDDEN_OUTCOME_FIELDS


class Phase3StaticPrepError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise Phase3StaticPrepError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase3StaticPrepError("GIT_FAILURE:" + " ".join(args)) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def git_blob_bytes(path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=ROOT,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase3StaticPrepError(
            f"GIT_BLOB_FAILURE:{path.as_posix()}"
        ) from exc


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def checksum_text(files: Mapping[str, bytes]) -> str:
    return "".join(
        f"{sha256_bytes(raw)}  {name}\n"
        for name, raw in sorted(files.items())
    )


def authenticate_repository(expected_head: str) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    require(head == expected_head, f"HEAD_MISMATCH:{head}")
    require(git("status", "--porcelain") == "", "WORKTREE_NOT_CLEAN")

    for ancestor, label in (
        (PHASE3_DESIGN_COMMIT, "PHASE3_DESIGN"),
        (LABEL_AMENDMENT_COMMIT, "LABEL_AMENDMENT"),
        (STRESSOR_DOMAIN_AMENDMENT_COMMIT, "STRESSOR_DOMAIN_AMENDMENT"),
    ):
        require(
            git_rc("merge-base", "--is-ancestor", ancestor, head) == 0,
            f"{label}_NOT_ANCESTOR",
        )

    observed_blobs: dict[str, str] = {}
    for path, expected_blob in FROZEN_DEPENDENCY_BLOBS.items():
        observed = git("rev-parse", f"HEAD:{path}")
        require(
            observed == expected_blob,
            f"FROZEN_DEPENDENCY_BLOB_DRIFT:{path}:{observed}",
        )
        observed_blobs[path] = observed

    return {
        "branch": branch,
        "head": head,
        "phase3_design_commit": PHASE3_DESIGN_COMMIT,
        "label_amendment_commit": LABEL_AMENDMENT_COMMIT,
        "stressor_domain_amendment_commit": STRESSOR_DOMAIN_AMENDMENT_COMMIT,
        "frozen_dependency_blobs": observed_blobs,
    }


def verify_label_contract() -> dict[str, int]:
    observed = {label.name: int(label) for label in FinalLabel}
    require(observed == EXPECTED_LABEL_TO_ID, f"FINAL_LABEL_MAPPING:{observed}")
    return observed


def verify_prior_lineage() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for spec in PRIOR_LINEAGE:
        source_rel = Path(spec["root"]) / SOURCE_FILE
        rows_rel = Path(spec["root"]) / BASE_ROW_FILE
        source_raw = git_blob_bytes(source_rel)
        rows_raw = git_blob_bytes(rows_rel)

        require(
            sha256_bytes(source_raw) == spec["source_sha256"],
            f"PRIOR_SOURCE_SHA:{spec['label']}",
        )
        require(
            sha256_bytes(rows_raw) == spec["rows_sha256"],
            f"PRIOR_ROWS_SHA:{spec['label']}",
        )

        facts_regen, rows_regen = ladder.build_population(
            int(spec["first"]),
            int(spec["last"]),
        )
        require(
            ladder.jsonl_bytes(facts_regen) == source_raw,
            f"PRIOR_SOURCE_REGEN:{spec['label']}",
        )
        require(
            ladder.jsonl_bytes(rows_regen) == rows_raw,
            f"PRIOR_ROWS_REGEN:{spec['label']}",
        )

        results.append({
            "label": spec["label"],
            "pair_range": f"xg1_fact_{spec['first']}..xg1_fact_{spec['last']}",
            "source_sha256": spec["source_sha256"],
            "rows_sha256": spec["rows_sha256"],
            "byte_regeneration": True,
        })

    return results


def build_population(first: int, last: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts_a, rows_a = ladder.build_population(first, last)
    facts_b, rows_b = ladder.build_population(first, last)

    require(
        ladder.jsonl_bytes(facts_a) == ladder.jsonl_bytes(facts_b),
        f"SOURCE_REGEN:{first}:{last}",
    )
    require(
        ladder.jsonl_bytes(rows_a) == ladder.jsonl_bytes(rows_b),
        f"ROWS_REGEN:{first}:{last}",
    )

    expected_pairs = last - first + 1
    require(len(facts_a) == expected_pairs, f"PAIR_COUNT:{first}:{last}")
    require(
        len(rows_a) == expected_pairs * BASE_ROWS_PER_PAIR,
        f"ROW_COUNT:{first}:{last}",
    )

    for row in [*facts_a, *rows_a]:
        require(
            not (FORBIDDEN_BASE_OUTCOME_FIELDS & set(row)),
            f"BASE_OUTCOME_FIELD_PRESENT:{first}:{last}",
        )

    return facts_a, rows_a


def row_sets(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(row["pair_id"]) for row in facts},
        {str(row["claim"]) for row in rows},
        {str(row["evidence"]) for row in rows},
    )


def prove_no_overlap_with_prior_9000(
    train_facts: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    assay_facts: Sequence[Mapping[str, Any]],
    assay_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prior_facts, prior_rows = ladder.build_population(1, 9000)

    prior = row_sets(prior_facts, prior_rows)
    train = row_sets(train_facts, train_rows)
    assay = row_sets(assay_facts, assay_rows)

    require(not (prior[0] & train[0]), "TRAIN_PAIR_OVERLAP_PRIOR")
    require(not (prior[1] & train[1]), "TRAIN_CLAIM_OVERLAP_PRIOR")
    require(not (prior[2] & train[2]), "TRAIN_EVIDENCE_OVERLAP_PRIOR")

    merged_prior = tuple(a | b for a, b in zip(prior, train))
    require(not (merged_prior[0] & assay[0]), "ASSAY_PAIR_OVERLAP_PRIOR")
    require(not (merged_prior[1] & assay[1]), "ASSAY_CLAIM_OVERLAP_PRIOR")
    require(not (merged_prior[2] & assay[2]), "ASSAY_EVIDENCE_OVERLAP_PRIOR")

    require(
        prior[0] == set(ladder.expected_pair_ids(1, 9000)),
        "PRIOR_PAIR_COVERAGE_001_9000",
    )
    require(
        train[0] == set(ladder.expected_pair_ids(TRAIN_FIRST, TRAIN_LAST)),
        "TRAIN_PAIR_COVERAGE",
    )
    require(
        assay[0] == set(ladder.expected_pair_ids(ASSAY_FIRST, ASSAY_LAST)),
        "ASSAY_PAIR_COVERAGE",
    )

    return {
        "prior_pair_coverage": "xg1_fact_001..xg1_fact_9000",
        "train_pair_overlap_with_prior": 0,
        "train_claim_overlap_with_prior": 0,
        "train_evidence_overlap_with_prior": 0,
        "assay_pair_overlap_with_prior_and_train": 0,
        "assay_claim_overlap_with_prior_and_train": 0,
        "assay_evidence_overlap_with_prior_and_train": 0,
        "final_pair_coverage": "xg1_fact_001..xg1_fact_9900",
    }


def explicit_denial_evidence(fact: Mapping[str, Any]) -> str:
    return (
        f"During {fact['time']}, records from {fact['location']} identify "
        f"{fact['title']} {fact['name']} as {fact['role']}; "
        f"the record explicitly denies that this person "
        f"{fact['predicate']} {fact['object']}."
    )


def build_training_view(
    facts: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    label_to_id = verify_label_contract()

    facts_by_id = {str(row["pair_id"]): row for row in facts}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_rows:
        grouped[str(row["source_pair_id"])].append(row)

    out: list[dict[str, Any]] = []

    for pair_id in ladder.expected_pair_ids(TRAIN_FIRST, TRAIN_LAST):
        fact = facts_by_id[pair_id]
        block = grouped[pair_id]
        require(
            [str(row["contrast_cell_id"]) for row in block] == list(BASE_CELLS),
            f"BASE_CELL_ORDER:{pair_id}",
        )
        require(
            len({str(row["claim"]) for row in block}) == 1,
            f"CLAIM_INVARIANCE:{pair_id}",
        )

        claim = str(block[0]["claim"])

        for row in block:
            cell = str(row["contrast_cell_id"])
            label = LABEL_BY_CELL[cell]
            stressor = cell in STRESSOR_DOMAIN_CELLS
            out.append({
                "schema_version":
                    "GEN5_PHASE3_SEVEN_CELL_LABELED_TRAINING_V1",
                "id": f"{pair_id}__phase3__{cell.lower()}",
                "pair_id": pair_id,
                "source_pair_id": pair_id,
                "row_id": str(row["row_id"]),
                "contrast_cell_id": cell,
                "claim": claim,
                "evidence": str(row["evidence"]),
                "final_label": label,
                "final_label_id": label_to_id[label],
                "stressor_domain": stressor,
                "stressor_anchor_name":
                    STRESSOR_ANCHOR_NAME if stressor else None,
                "stressor_target_offset":
                    STRESSOR_TARGET_OFFSET if stressor else None,
                "stressor_layer":
                    STRESSOR_LAYER if stressor else None,
            })

        cell = "C6_EXPLICIT_DENIAL"
        label = LABEL_BY_CELL[cell]
        out.append({
            "schema_version":
                "GEN5_PHASE3_SEVEN_CELL_LABELED_TRAINING_V1",
            "id": f"{pair_id}__phase3__{cell.lower()}",
            "pair_id": pair_id,
            "source_pair_id": pair_id,
            "row_id": f"{pair_id}__phase3__c6_explicit_denial",
            "contrast_cell_id": cell,
            "claim": claim,
            "evidence": explicit_denial_evidence(fact),
            "final_label": label,
            "final_label_id": label_to_id[label],
            "stressor_domain": False,
            "stressor_anchor_name": None,
            "stressor_target_offset": None,
            "stressor_layer": None,
        })

    require(len(out) == LABELED_TRAIN_ROW_COUNT, "LABELED_TRAIN_ROW_COUNT")

    cell_counts = Counter(str(row["contrast_cell_id"]) for row in out)
    require(
        cell_counts == Counter({cell: TRAIN_PAIR_COUNT for cell in TRAINING_CELLS}),
        f"TRAIN_CELL_COUNTS:{dict(cell_counts)}",
    )

    label_counts = Counter(str(row["final_label"]) for row in out)
    require(
        label_counts == Counter({
            "REFUTE": 600,
            "NOT_ENTITLED": 3000,
            "SUPPORT": 600,
        }),
        f"TRAIN_LABEL_COUNTS:{dict(label_counts)}",
    )

    stressor_counts = Counter(
        str(row["contrast_cell_id"])
        for row in out
        if bool(row["stressor_domain"])
    )
    require(
        stressor_counts
        == Counter({cell: TRAIN_PAIR_COUNT for cell in STRESSOR_DOMAIN_CELLS}),
        f"STRESSOR_CELL_COUNTS:{dict(stressor_counts)}",
    )

    return out


def split_pair_ids(pair_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(str(x) for x in pair_ids)
    require(len(ordered) == TRAIN_PAIR_COUNT, "SPLIT_PAIR_COUNT")
    require(len(set(ordered)) == TRAIN_PAIR_COUNT, "SPLIT_PAIR_UNIQUENESS")

    shuffled = list(ordered)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    dev_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * DEV_RATIO)),
    )
    dev_set = set(shuffled[:dev_count])

    train = [pair for pair in ordered if pair not in dev_set]
    dev = [pair for pair in ordered if pair in dev_set]

    require(len(train) == TRAIN_SPLIT_PAIRS, "TRAIN_SPLIT_PAIR_COUNT")
    require(len(dev) == DEV_SPLIT_PAIRS, "DEV_SPLIT_PAIR_COUNT")
    require(not (set(train) & set(dev)), "PAIR_SPLIT_OVERLAP")
    require(set(train) | set(dev) == set(ordered), "PAIR_SPLIT_COVERAGE")

    return train, dev


def build_split_manifest(
    training_view: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pair_ids = sorted({str(row["pair_id"]) for row in training_view})
    train_pairs, dev_pairs = split_pair_ids(pair_ids)
    train_set, dev_set = set(train_pairs), set(dev_pairs)

    train_rows = [row for row in training_view if str(row["pair_id"]) in train_set]
    dev_rows = [row for row in training_view if str(row["pair_id"]) in dev_set]

    require(len(train_rows) == TRAIN_SPLIT_ROWS, "TRAIN_SPLIT_ROW_COUNT")
    require(len(dev_rows) == DEV_SPLIT_ROWS, "DEV_SPLIT_ROW_COUNT")

    train_labels = Counter(str(row["final_label"]) for row in train_rows)
    dev_labels = Counter(str(row["final_label"]) for row in dev_rows)

    require(
        train_labels == Counter({
            "REFUTE": 480,
            "NOT_ENTITLED": 2400,
            "SUPPORT": 480,
        }),
        f"TRAIN_SPLIT_LABEL_COUNTS:{dict(train_labels)}",
    )
    require(
        dev_labels == Counter({
            "REFUTE": 120,
            "NOT_ENTITLED": 600,
            "SUPPORT": 120,
        }),
        f"DEV_SPLIT_LABEL_COUNTS:{dict(dev_labels)}",
    )

    return {
        "schema_version": "GEN5_PHASE3_PAIR_SPLIT_V1",
        "split_seed": SPLIT_SEED,
        "dev_ratio": DEV_RATIO,
        "algorithm":
            "sorted_pair_ids_then_random.Random(seed).shuffle;"
            "dev=first_round(n*ratio);rows_preserve_canonical_order",
        "train_pair_count": len(train_pairs),
        "dev_pair_count": len(dev_pairs),
        "train_row_count": len(train_rows),
        "dev_row_count": len(dev_rows),
        "train_pair_ids": train_pairs,
        "dev_pair_ids": dev_pairs,
        "train_label_counts": dict(sorted(train_labels.items())),
        "dev_label_counts": dict(sorted(dev_labels.items())),
        "pair_overlap_count": 0,
    }


def validate_training_serialization(
    training_view: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    claim_truncation = 0
    evidence_truncation = 0
    max_serialized = 0

    for row in training_view:
        claim_ids = [
            int(x)
            for x in tokenizer.encode(
                str(row["claim"]),
                add_special_tokens=False,
            ).ids
        ]
        evidence_ids = [
            int(x)
            for x in tokenizer.encode(
                str(row["evidence"]),
                add_special_tokens=False,
            ).ids
        ]
        require(bool(claim_ids), f"EMPTY_CLAIM:{row['id']}")
        require(bool(evidence_ids), f"EMPTY_EVIDENCE:{row['id']}")

        claim_truncation += int(len(claim_ids) > CLAIM_BUDGET)
        evidence_truncation += int(len(evidence_ids) > EVIDENCE_BUDGET)

        serialized = (
            min(len(claim_ids), CLAIM_BUDGET)
            + 1
            + min(len(evidence_ids), EVIDENCE_BUDGET)
        )
        require(serialized <= MAX_LENGTH, f"SERIALIZATION_LENGTH:{row['id']}")
        max_serialized = max(max_serialized, serialized)

    return {
        "row_count": len(training_view),
        "all_rows_tokenizable": True,
        "claim_truncation_row_count": claim_truncation,
        "evidence_truncation_row_count": evidence_truncation,
        "max_serialized_length": max_serialized,
        "serialization": "claim[:63]+EOS(0)+evidence[:64]",
    }


def compute_base_anchor_eligibility(
    facts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, Any],
    *,
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts_by_id = {str(fact["pair_id"]): fact for fact in facts}
    expected_pairs = len(facts_by_id)
    require(expected_pairs > 0, f"{role}:NO_FACTS")

    anchors: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in facts_by_id, f"{role}:MISSING_FACT:{pair_id}")
        produced = eligibility.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair_id],
            tokenizer,
        )
        for item in produced:
            item = dict(item)
            item["schema_version"] = "GEN5_PHASE3_TOKENIZER_ANCHOR_V1"
            item["phase3_role"] = role
            anchors.append(item)

    expected_identity = expected_pairs * 4
    expected_name = expected_pairs * 2
    expected_anchor_rows = expected_identity + expected_name

    require(
        len(anchors) == expected_anchor_rows,
        f"{role}:ANCHOR_ROW_COUNT:{len(anchors)}",
    )

    counts = Counter(str(row["anchor_name"]) for row in anchors)
    require(
        counts == Counter({
            "A_IDENTITY": expected_identity,
            "A_NAME": expected_name,
        }),
        f"{role}:ANCHOR_COUNTS:{dict(counts)}",
    )

    exclusion = Counter(
        str(row["exclusion_code"])
        for row in anchors
        if row["exclusion_code"] is not None
    )
    eligible = sum(bool(row["post4_eligible"]) for row in anchors)
    require(
        eligible == expected_anchor_rows,
        f"{role}:ANCHOR_INELIGIBLE:{dict(exclusion)}",
    )

    lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in anchors
    }
    require(
        len(lookup) == expected_anchor_rows,
        f"{role}:ANCHOR_KEY_DUPLICATE",
    )

    mismatch = 0
    for pair_id in facts_by_id:
        for cell in eligibility.TARGET_IDENTITY_NAME_CELLS:
            a = lookup[(pair_id, cell, "A_IDENTITY")]
            b = lookup[(pair_id, cell, "A_NAME")]
            mismatch += int(
                a["absolute_anchor_token_index"]
                != b["absolute_anchor_token_index"]
            )
    require(mismatch == 0, f"{role}:IDENTITY_NAME_MISMATCH:{mismatch}")

    anchor_raw = jsonl_bytes(anchors)
    summary = {
        "schema_version": "GEN5_PHASE3_TOKENIZER_ELIGIBILITY_SUMMARY_V1",
        "result": "PASS_GEN5_PHASE3_TOKENIZER_ANCHOR_ELIGIBILITY",
        "role": role,
        "source_pair_count": expected_pairs,
        "base_row_count": len(rows),
        "required_anchor_row_count": expected_anchor_rows,
        "eligible_anchor_row_count": eligible,
        "eligible_anchor_counts": dict(sorted(counts.items())),
        "exclusion_counts": dict(sorted(exclusion.items())),
        "identity_name_coordinate_mismatch_count": mismatch,
        "anchor_manifest_sha256": sha256_bytes(anchor_raw),
        "tokenizer": dict(tokenizer_provenance),
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    return anchors, summary


def build_stressor_targets(
    training_anchors: Sequence[Mapping[str, Any]],
    training_view: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in training_anchors
    }

    targets: list[dict[str, Any]] = []
    for row in training_view:
        if not bool(row["stressor_domain"]):
            continue

        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        anchor = lookup[(pair, cell, STRESSOR_ANCHOR_NAME)]

        require(bool(anchor["post4_eligible"]), f"STRESSOR_POST4:{pair}:{cell}")
        absolute = int(anchor["absolute_anchor_token_index"])
        terminal = int(anchor["terminal_index"])
        target = absolute + STRESSOR_TARGET_OFFSET

        require(target < terminal, f"STRESSOR_TARGET_TERMINAL:{pair}:{cell}")
        require(
            absolute + 4 <= terminal - 1,
            f"STRESSOR_FROZEN_POST4:{pair}:{cell}",
        )

        targets.append({
            "schema_version": "GEN5_PHASE3_STRESSOR_TARGET_V1",
            "source_pair_id": pair,
            "contrast_cell_id": cell,
            "anchor_name": STRESSOR_ANCHOR_NAME,
            "absolute_anchor_token_index": absolute,
            "target_offset": STRESSOR_TARGET_OFFSET,
            "intervention_token_index": target,
            "intervention_layer": STRESSOR_LAYER,
            "terminal_index": terminal,
            "post4_eligible": True,
            "pressure_P0": "native",
            "pressure_PR": "pp3_neutralized",
            "pressure_PC": "pp5_matched_control",
        })

    require(
        len(targets) == TRAIN_PAIR_COUNT * len(STRESSOR_DOMAIN_CELLS),
        f"STRESSOR_TARGET_COUNT:{len(targets)}",
    )

    counts = Counter(str(row["contrast_cell_id"]) for row in targets)
    require(
        counts
        == Counter({cell: TRAIN_PAIR_COUNT for cell in STRESSOR_DOMAIN_CELLS}),
        f"STRESSOR_TARGET_CELL_COUNTS:{dict(counts)}",
    )

    return targets


def structural_manifest(
    *,
    role: str,
    first: int,
    last: int,
    pair_count: int,
    rows: Sequence[Mapping[str, Any]],
    source_raw: bytes,
    row_raw: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": "GEN5_PHASE3_XG1_STRUCTURAL_V1",
        "result": "PASS_GEN5_PHASE3_XG1_STRUCTURAL",
        "role": role,
        "phase3_design_commit": PHASE3_DESIGN_COMMIT,
        "label_amendment_commit": LABEL_AMENDMENT_COMMIT,
        "stressor_domain_amendment_commit": STRESSOR_DOMAIN_AMENDMENT_COMMIT,
        "generator_family": rows[0]["generator_family"],
        "deterministic_generator_semantics": True,
        "pair_id_first": f"xg1_fact_{first}",
        "pair_id_last": f"xg1_fact_{last}",
        "source_pair_count": pair_count,
        "row_count": len(rows),
        "rows_per_pair": BASE_ROWS_PER_PAIR,
        "source_file_sha256": sha256_bytes(source_raw),
        "row_file_sha256": sha256_bytes(row_raw),
        "labels_present": False,
        "response_fields_present": False,
        "endpoint_values_present": False,
        "model_geometry_present": False,
        "checkpoint_loaded": False,
        "model_executed": False,
        "cuda_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "p_value_count": 0,
    }


def write_file_bundle(root: Path, files: Mapping[str, bytes]) -> dict[str, Any]:
    require(not root.exists(), f"OUTPUT_COLLISION:{root.relative_to(ROOT)}")
    root.mkdir(parents=True, exist_ok=False)

    for name, raw in files.items():
        (root / name).write_bytes(raw)

    sums = checksum_text(files)
    (root / CHECKSUM_FILE).write_text(sums, encoding="utf-8", newline="\n")

    return {
        "directory": root.relative_to(ROOT).as_posix(),
        "files": {
            name: {
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
            }
            for name, raw in sorted(files.items())
        },
        "checksums_sha256": sha256_bytes(sums.encode("utf-8")),
    }


def self_test() -> None:
    require(
        EXPECTED_LABEL_TO_ID == {label.name: int(label) for label in FinalLabel},
        "SELFTEST_LABEL_ENUM",
    )
    require(tuple(LABEL_BY_CELL) == TRAINING_CELLS, "SELFTEST_CELL_ORDER")
    require(
        set(STRESSOR_DOMAIN_CELLS).isdisjoint(NON_STRESSOR_CELLS),
        "SELFTEST_DOMAIN_OVERLAP",
    )
    require(
        set(STRESSOR_DOMAIN_CELLS) | set(NON_STRESSOR_CELLS)
        == set(TRAINING_CELLS),
        "SELFTEST_DOMAIN_COVERAGE",
    )

    facts, rows = ladder.build_population(TRAIN_FIRST, TRAIN_LAST)
    view = build_training_view(facts, rows)
    split = build_split_manifest(view)

    require(len(view) == 4200, "SELFTEST_VIEW_ROWS")
    require(split["train_row_count"] == 3360, "SELFTEST_TRAIN_ROWS")
    require(split["dev_row_count"] == 840, "SELFTEST_DEV_ROWS")

    denial = explicit_denial_evidence(facts[0])
    require(
        "explicitly denies that this person" in denial,
        "SELFTEST_DENIAL_TEXT",
    )

    c6 = [
        row
        for row in view
        if row["contrast_cell_id"] == "C6_EXPLICIT_DENIAL"
    ]
    require(len(c6) == 600, "SELFTEST_C6_COUNT")
    require(all(row["final_label"] == "REFUTE" for row in c6), "SELFTEST_C6_LABEL")
    require(all(row["final_label_id"] == 0 for row in c6), "SELFTEST_C6_ID")
    require(all(not row["stressor_domain"] for row in c6), "SELFTEST_C6_DOMAIN")

    print("RESULT=PASS_GEN5_PHASE3_STATIC_PREPARER_SELF_TEST")
    print("MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("P_VALUE_COUNT=0")


def run(expected_head: str, tokenizer_snapshot: Path | None) -> dict[str, Any]:
    provenance = authenticate_repository(expected_head)
    label_mapping = verify_label_contract()
    prior_lineage = verify_prior_lineage()

    train_facts, train_base_rows = build_population(TRAIN_FIRST, TRAIN_LAST)
    assay_facts, assay_base_rows = build_population(ASSAY_FIRST, ASSAY_LAST)

    overlap = prove_no_overlap_with_prior_9000(
        train_facts,
        train_base_rows,
        assay_facts,
        assay_base_rows,
    )

    training_view = build_training_view(train_facts, train_base_rows)
    split_manifest = build_split_manifest(training_view)

    tokenizer, tokenizer_provenance = eligibility.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )

    training_serialization = validate_training_serialization(
        training_view,
        tokenizer,
    )

    train_anchors, train_token_summary = compute_base_anchor_eligibility(
        train_facts,
        train_base_rows,
        tokenizer,
        tokenizer_provenance,
        role="contention_training_base",
    )
    assay_anchors, assay_token_summary = compute_base_anchor_eligibility(
        assay_facts,
        assay_base_rows,
        tokenizer,
        tokenizer_provenance,
        role="ownership_interaction_assay",
    )

    stressor_targets = build_stressor_targets(train_anchors, training_view)

    train_source_raw = ladder.jsonl_bytes(train_facts)
    train_base_raw = ladder.jsonl_bytes(train_base_rows)
    assay_source_raw = ladder.jsonl_bytes(assay_facts)
    assay_base_raw = ladder.jsonl_bytes(assay_base_rows)

    train_struct = structural_manifest(
        role="contention_training_base",
        first=TRAIN_FIRST,
        last=TRAIN_LAST,
        pair_count=TRAIN_PAIR_COUNT,
        rows=train_base_rows,
        source_raw=train_source_raw,
        row_raw=train_base_raw,
    )
    assay_struct = structural_manifest(
        role="ownership_interaction_assay",
        first=ASSAY_FIRST,
        last=ASSAY_LAST,
        pair_count=ASSAY_PAIR_COUNT,
        rows=assay_base_rows,
        source_raw=assay_source_raw,
        row_raw=assay_base_raw,
    )

    train_token_summary = dict(train_token_summary)
    train_token_summary["seven_cell_training_serialization"] = training_serialization
    train_token_summary["stressor_domain_cells"] = list(STRESSOR_DOMAIN_CELLS)
    train_token_summary["stressor_target_count"] = len(stressor_targets)
    train_token_summary["seven_cell_complete_pair_count"] = TRAIN_PAIR_COUNT
    train_token_summary["seven_cell_complete_pair_target"] = TRAIN_PAIR_COUNT

    train_files = {
        SOURCE_FILE: train_source_raw,
        BASE_ROW_FILE: train_base_raw,
        LABELED_ROW_FILE: jsonl_bytes(training_view),
        SPLIT_FILE: pretty_json_bytes(split_manifest),
        STRUCTURAL_MANIFEST_FILE: pretty_json_bytes(train_struct),
        ANCHOR_FILE: jsonl_bytes(train_anchors),
        TOKENIZER_SUMMARY_FILE: pretty_json_bytes(train_token_summary),
        STRESSOR_TARGET_FILE: jsonl_bytes(stressor_targets),
    }

    assay_files = {
        SOURCE_FILE: assay_source_raw,
        BASE_ROW_FILE: assay_base_raw,
        STRUCTURAL_MANIFEST_FILE: pretty_json_bytes(assay_struct),
        ANCHOR_FILE: jsonl_bytes(assay_anchors),
        TOKENIZER_SUMMARY_FILE: pretty_json_bytes(assay_token_summary),
    }

    train_record = write_file_bundle(ROOT / TRAIN_OUTPUT_DIR, train_files)
    assay_record = write_file_bundle(ROOT / ASSAY_OUTPUT_DIR, assay_files)

    report_rel = Path(
        f"reports/reason_router_gen5_phase3_static_preparation_{expected_head[:7]}_v1"
    )
    report_root = ROOT / report_rel
    require(not report_root.exists(), f"OUTPUT_COLLISION:{report_rel}")

    summary = {
        "schema_version": "GEN5_PHASE3_STATIC_PREPARATION_SUMMARY_V1",
        "result": "PASS_GEN5_PHASE3_STATIC_PREPARATION",
        "execution_head": expected_head,
        "repository": provenance,
        "label_mapping": label_mapping,
        "prior_lineage": prior_lineage,
        "overlap": overlap,
        "contention_training": {
            "pair_range": f"xg1_fact_{TRAIN_FIRST}..xg1_fact_{TRAIN_LAST}",
            "source_pair_count": TRAIN_PAIR_COUNT,
            "outcome_blind_base_row_count": BASE_TRAIN_ROW_COUNT,
            "labeled_training_row_count": LABELED_TRAIN_ROW_COUNT,
            "rows_per_pair": TRAIN_ROWS_PER_PAIR,
            "label_counts": dict(sorted(Counter(
                str(row["final_label"]) for row in training_view
            ).items())),
            "stressor_domain_cells": list(STRESSOR_DOMAIN_CELLS),
            "non_stressor_cells": list(NON_STRESSOR_CELLS),
            "stressor_target_count": len(stressor_targets),
            "split_seed": SPLIT_SEED,
            "train_pair_count": TRAIN_SPLIT_PAIRS,
            "dev_pair_count": DEV_SPLIT_PAIRS,
            "train_row_count": TRAIN_SPLIT_ROWS,
            "dev_row_count": DEV_SPLIT_ROWS,
            "bundle": train_record,
        },
        "ownership_interaction_assay": {
            "pair_range": f"xg1_fact_{ASSAY_FIRST}..xg1_fact_{ASSAY_LAST}",
            "source_pair_count": ASSAY_PAIR_COUNT,
            "row_count": ASSAY_ROW_COUNT,
            "rows_per_pair": BASE_ROWS_PER_PAIR,
            "labels_present": False,
            "bundle": assay_record,
        },
        "tokenizer": tokenizer_provenance,
        "scientific_model_forward_count": 0,
        "checkpoint_load_count": 0,
        "model_instantiated": False,
        "cuda_executed": False,
        "training_executed": False,
        "backward_executed": False,
        "task_evaluation_executed": False,
        "contention_computed": False,
        "causal_role_outcome_computed": False,
        "p_value_count": 0,
        "next_stage":
            "FREEZE_PHASE3_STATIC_PREPARATION_THEN_DEFINE_TRAINING_STRESSOR_IMPLEMENTATION_AUTHORITY",
    }

    report_root.mkdir(parents=True, exist_ok=False)
    summary_raw = pretty_json_bytes(summary)
    (report_root / PREP_SUMMARY_FILE).write_bytes(summary_raw)
    sums = checksum_text({PREP_SUMMARY_FILE: summary_raw})
    (report_root / CHECKSUM_FILE).write_text(
        sums,
        encoding="utf-8",
        newline="\n",
    )

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU/static-only Gen5 Phase 3 XG1 9001..9900 preparation. "
            "No model/checkpoint/CUDA/training/scientific inference."
        )
    )
    parser.add_argument("--expected-head")
    parser.add_argument("--tokenizer-snapshot", type=Path, default=None)
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test_only:
        self_test()
        return 0

    require(bool(args.expected_head), "--expected-head is required")
    summary = run(str(args.expected_head), args.tokenizer_snapshot)

    print("RESULT=" + summary["result"])
    print(f"EXECUTION_HEAD={summary['execution_head']}")
    print("TRAIN_RANGE=" + summary["contention_training"]["pair_range"])
    print(
        "TRAIN_BASE_ROWS="
        + str(summary["contention_training"]["outcome_blind_base_row_count"])
    )
    print(
        "TRAIN_LABELED_ROWS="
        + str(summary["contention_training"]["labeled_training_row_count"])
    )
    print(
        "TRAIN_SPLIT_ROWS="
        + str(summary["contention_training"]["train_row_count"])
    )
    print(
        "DEV_SPLIT_ROWS="
        + str(summary["contention_training"]["dev_row_count"])
    )
    print(
        "STRESSOR_TARGET_COUNT="
        + str(summary["contention_training"]["stressor_target_count"])
    )
    print("ASSAY_RANGE=" + summary["ownership_interaction_assay"]["pair_range"])
    print(
        "ASSAY_ROWS="
        + str(summary["ownership_interaction_assay"]["row_count"])
    )
    print("MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("CUDA_EXECUTED=False")
    print("TRAINING_EXECUTED=False")
    print("BACKWARD_EXECUTED=False")
    print("P_VALUE_COUNT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
