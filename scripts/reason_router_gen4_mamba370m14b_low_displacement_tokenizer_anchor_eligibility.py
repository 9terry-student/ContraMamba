#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as anchor_gate


ROOT = _REPO_ROOT
EXPECTED_BRANCH = "gen4-mamba370m-core-replication"

PLAN_FREEZE_COMMIT = "7bee947899be6c7808217dc68d09f6d2ad2c5978"
PLAN_PATH = (
    "reports/"
    "reason_router_gen4_low_displacement_tokenizer_anchor_eligibility_"
    "prospective_plan.md"
)
PLAN_BLOB = "64490646390e78b9bf5dce483b86e5b4d38bebe8"

STRUCTURAL_FREEZE_COMMIT = "37e8176edceb9da9a976df4520904e31e1134ff5"
STRUCTURAL_DIR = Path(
    "data/reason_router_gen4_mamba370m14b_low_displacement_xg1_v1"
)
SOURCE_FILE = "structured_source_facts.jsonl"
ROW_FILE = "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_MANIFEST_FILE = "structural_manifest.json"
STRUCTURAL_SUMS_FILE = "SHA256SUMS.txt"

EXPECTED_SOURCE_SHA256 = (
    "eddd6a264130e6451c72aeab010758dac43de82d9521898dfd1489c86717d11a"
)
EXPECTED_ROWS_SHA256 = (
    "1d0f21ba44b4a282f42edb7e56626bb7d522ce09e25e1611bf1424ead4da2b28"
)
EXPECTED_STRUCTURAL_MANIFEST_SHA256 = (
    "684821b2b8d5d3e84ee9cec38b55cab848a50ebec75daf7892935a417183b28d"
)

PAIR_FIRST = 5401
PAIR_LAST = 5700
PAIR_COUNT = 300
ROW_COUNT = 1800
TARGET_CELLS = ("C0_SHAM", "C2_NAME")
TARGET_ROW_COUNT = PAIR_COUNT * len(TARGET_CELLS)

SCALES = ("mamba370m", "mamba14b")
SCALE_CONFIG = {
    "mamba370m": {
        "geom_module":
            "scripts.reason_router_gen4_mamba370m_geometry_prepare_fast_cuda",
        "repo": "state-spaces/mamba-370m-hf",
        "revision": "589179554943157be31701edd8b4558889276674",
        "tokenizer_file_sha256": {
            "special_tokens_map.json":
                "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
            "tokenizer.json":
                "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
            "tokenizer_config.json":
                "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
        },
    },
    "mamba14b": {
        "geom_module":
            "scripts.reason_router_gen4_mamba14b_geometry_prepare_fast_cuda",
        "repo": "state-spaces/mamba-1.4b-hf",
        "revision": "6e46eae61c27280517feef46f536d16b91076f08",
        "tokenizer_file_sha256": {
            "tokenizer.json":
                "3cf430678137c8491ca82fb7092ee49e44ad38857fffe1e4a4a5ed860139a5b8",
            "tokenizer_config.json":
                "3ba257483d22a5a84aab5465aa427e59bdaeb55f09fb14349e2d571ff67e8020",
        },
    },
}

OUTPUT_DIR = Path(
    "reports/"
    "reason_router_gen4_mamba370m14b_low_displacement_"
    "tokenizer_anchor_eligibility_v1"
)

RESULT_PASS = (
    "PASS_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_ELIGIBILITY"
)
RESULT_BLOCKED = (
    "BLOCKED_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_ELIGIBILITY"
)
SCALE_PASS = "PASS_600_OF_600"
SCALE_BLOCKED = "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"

SUMMARY_SCHEMA = (
    "GEN4_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_"
    "ELIGIBILITY_SUMMARY_V1"
)
CROSS_SCALE_SCHEMA = (
    "GEN4_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_"
    "ELIGIBILITY_CROSS_SCALE_V1"
)
ARTIFACT_SCHEMA = (
    "GEN4_MAMBA370M14B_LOW_DISPLACEMENT_TOKENIZER_ANCHOR_"
    "ELIGIBILITY_ARTIFACT_V1"
)

SCALE_OUTPUTS = {
    "mamba370m": (
        "mamba370m_anchor_manifest.jsonl",
        "mamba370m_eligibility_summary.json",
    ),
    "mamba14b": (
        "mamba14b_anchor_manifest.jsonl",
        "mamba14b_eligibility_summary.json",
    ),
}
CROSS_SCALE_FILE = "cross_scale_summary.json"
ARTIFACT_FILE = "artifact_manifest.json"
CHECKSUM_FILE = "SHA256SUMS.txt"


class LowDisplacementEligibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LowDisplacementEligibilityError(message)


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LowDisplacementEligibilityError(
            "GIT_FAILURE:" + " ".join(args)
        ) from exc


def git_rc(*args: str) -> int:
    return subprocess.call(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        require(
            isinstance(row, dict),
            f"JSONL_OBJECT:{path}:{line_no}",
        )
        rows.append(row)
    return rows


def authenticate_repo(expected_head: str) -> None:
    branch = git("branch", "--show-current")
    require(
        branch in ("", EXPECTED_BRANCH),
        f"BRANCH_MISMATCH:{branch}",
    )
    require(
        git("rev-parse", "HEAD") == expected_head,
        "HEAD_MISMATCH",
    )
    require(
        git("status", "--porcelain") == "",
        "WORKTREE_NOT_CLEAN",
    )
    for ancestor, label in (
        (STRUCTURAL_FREEZE_COMMIT, "STRUCTURAL_FREEZE"),
        (PLAN_FREEZE_COMMIT, "PLAN_FREEZE"),
    ):
        require(
            git_rc(
                "merge-base",
                "--is-ancestor",
                ancestor,
                expected_head,
            )
            == 0,
            f"{label}_NOT_ANCESTOR",
        )
    require(
        git("rev-parse", f"HEAD:{PLAN_PATH}") == PLAN_BLOB,
        "PLAN_BLOB_DRIFT",
    )


def validate_structural_inputs(
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cohort = root / STRUCTURAL_DIR
    source = cohort / SOURCE_FILE
    rows_path = cohort / ROW_FILE
    manifest_path = cohort / STRUCTURAL_MANIFEST_FILE
    sums_path = cohort / STRUCTURAL_SUMS_FILE

    for path in (source, rows_path, manifest_path, sums_path):
        require(path.is_file(), f"STRUCTURAL_INPUT_MISSING:{path}")

    observed = {
        SOURCE_FILE: sha256_file(source),
        ROW_FILE: sha256_file(rows_path),
        STRUCTURAL_MANIFEST_FILE: sha256_file(manifest_path),
    }
    expected = {
        SOURCE_FILE: EXPECTED_SOURCE_SHA256,
        ROW_FILE: EXPECTED_ROWS_SHA256,
        STRUCTURAL_MANIFEST_FILE: EXPECTED_STRUCTURAL_MANIFEST_SHA256,
    }
    require(observed == expected, "STRUCTURAL_SHA256_DRIFT")

    expected_sums = "".join(
        f"{expected[name]}  {name}\n"
        for name in sorted(expected)
    )
    require(
        sums_path.read_text(encoding="utf-8") == expected_sums,
        "STRUCTURAL_SUMS_DRIFT",
    )

    facts = read_jsonl(source)
    rows = read_jsonl(rows_path)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    require(len(facts) == PAIR_COUNT, "SOURCE_PAIR_COUNT")
    require(len(rows) == ROW_COUNT, "ROW_COUNT")
    expected_pairs = [
        f"xg1_fact_{i}"
        for i in range(PAIR_FIRST, PAIR_LAST + 1)
    ]
    require(
        [str(row["pair_id"]) for row in facts] == expected_pairs,
        "PAIR_ORDER",
    )
    require(
        manifest["result"]
        == "PASS_MAMBA370M14B_LOW_DISPLACEMENT_XG1_5401_5700_STRUCTURAL",
        "STRUCTURAL_RESULT",
    )
    require(
        manifest["pair_id_first"] == "xg1_fact_5401"
        and manifest["pair_id_last"] == "xg1_fact_5700",
        "STRUCTURAL_RANGE",
    )
    require(
        manifest["source_pair_count"] == PAIR_COUNT
        and manifest["row_count"] == ROW_COUNT,
        "STRUCTURAL_COUNTS",
    )
    require(
        manifest["target_cells"] == list(TARGET_CELLS),
        "STRUCTURAL_TARGET_CELLS",
    )
    for key in (
        "response_fields_present",
        "endpoint_values_present",
        "tokenizer_executed",
        "checkpoint_loaded",
        "model_executed",
        "cuda_executed",
    ):
        require(
            manifest[key] is False,
            f"STRUCTURAL_BOUNDARY:{key}",
        )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }
    require(len(facts_by_id) == PAIR_COUNT, "FACT_ID_UNIQUENESS")

    target_rows = [
        row
        for row in rows
        if str(row["contrast_cell_id"]) in TARGET_CELLS
    ]
    require(len(target_rows) == TARGET_ROW_COUNT, "TARGET_ROW_COUNT")
    expected_target_keys = [
        (pair, cell)
        for pair in expected_pairs
        for cell in TARGET_CELLS
    ]
    observed_target_keys = [
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
        )
        for row in target_rows
    ]
    require(
        observed_target_keys == expected_target_keys,
        "TARGET_ROW_ORDER",
    )
    require(
        len(set(observed_target_keys)) == TARGET_ROW_COUNT,
        "TARGET_ROW_DUPLICATE",
    )
    return facts, target_rows, manifest


def load_scale_tokenizer(
    scale: str,
    snapshot: Path,
) -> tuple[Any, dict[str, Any]]:
    require(scale in SCALES, f"SCALE:{scale}")
    config = SCALE_CONFIG[scale]
    require(
        snapshot.resolve().name == config["revision"],
        f"SNAPSHOT_REVISION:{scale}",
    )
    module = importlib.import_module(
        str(config["geom_module"])
    )
    tokenizer, provenance = module.load_tokenizer(
        snapshot.resolve()
    )
    require(provenance["repo"] == config["repo"], f"TOKENIZER_REPO:{scale}")
    require(
        provenance["revision"] == config["revision"],
        f"TOKENIZER_REVISION:{scale}",
    )
    require(
        provenance["tokenizers_version"] == "0.22.2",
        f"TOKENIZERS_VERSION:{scale}",
    )
    require(
        provenance["file_sha256"]
        == config["tokenizer_file_sha256"],
        f"TOKENIZER_FILE_SHA256:{scale}",
    )
    require(
        int(provenance["vocab_size"]) == 50277,
        f"TOKENIZER_VOCAB_SIZE:{scale}",
    )
    return tokenizer, dict(provenance)


def analyze_scale(
    *,
    scale: str,
    snapshot: Path,
    facts: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer, provenance = load_scale_tokenizer(
        scale,
        snapshot,
    )
    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in facts
    }

    anchor_rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    identity_eligible = 0
    name_eligible = 0
    mismatches: list[dict[str, Any]] = []

    for row in target_rows:
        pair = str(row["source_pair_id"])
        cell = str(row["contrast_cell_id"])
        require(cell in TARGET_CELLS, f"UNEXPECTED_TARGET_CELL:{cell}")
        events = anchor_gate.analyze_required_anchors_for_row(
            row,
            facts_by_id[pair],
            tokenizer,
        )
        by_name = {
            str(event["anchor_name"]): dict(event)
            for event in events
        }
        require(
            set(by_name) == {"A_IDENTITY", "A_NAME"},
            f"ANCHOR_SET:{scale}:{pair}:{cell}",
        )
        require(
            len(events) == 2,
            f"ANCHOR_DUPLICATE:{scale}:{pair}:{cell}",
        )

        identity = by_name["A_IDENTITY"]
        name = by_name["A_NAME"]

        for event in (identity, name):
            out = dict(event)
            out["scale"] = scale
            anchor_rows.append(out)
            if not bool(event["post4_eligible"]):
                exclusions[
                    str(event["exclusion_code"] or "UNKNOWN")
                ] += 1

        if bool(identity["post4_eligible"]):
            identity_eligible += 1
        if bool(name["post4_eligible"]):
            name_eligible += 1

        if (
            identity["absolute_anchor_token_index"]
            != name["absolute_anchor_token_index"]
        ):
            mismatches.append(
                {
                    "source_pair_id": pair,
                    "contrast_cell_id": cell,
                    "identity_index":
                        identity["absolute_anchor_token_index"],
                    "name_index":
                        name["absolute_anchor_token_index"],
                }
            )

    require(
        len(anchor_rows) == TARGET_ROW_COUNT * 2,
        f"ANCHOR_ROW_COUNT:{scale}",
    )

    verdict = (
        SCALE_PASS
        if (
            identity_eligible == TARGET_ROW_COUNT
            and name_eligible == TARGET_ROW_COUNT
            and not exclusions
            and not mismatches
        )
        else SCALE_BLOCKED
    )

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "scale": scale,
        "result": verdict,
        "hf_repo": provenance["repo"],
        "hf_revision": provenance["revision"],
        "tokenizers_version": provenance["tokenizers_version"],
        "tokenizer_file_sha256": provenance["file_sha256"],
        "vocab_size": int(provenance["vocab_size"]),
        "structural_source_sha256": EXPECTED_SOURCE_SHA256,
        "structural_row_sha256": EXPECTED_ROWS_SHA256,
        "source_pair_count": PAIR_COUNT,
        "target_cells": list(TARGET_CELLS),
        "target_row_count": TARGET_ROW_COUNT,
        "required_anchor_names": ["A_IDENTITY", "A_NAME"],
        "required_anchor_count_per_name": TARGET_ROW_COUNT,
        "eligible_identity_count": identity_eligible,
        "eligible_name_count": name_eligible,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "identity_name_mismatch_count": len(mismatches),
        "identity_name_mismatches": mismatches,
        "post4_rule": "a+4 <= terminal_index-1",
        "shortened_post_window_allowed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "inference_performed": False,
        "training_executed": False,
        "row_filtering_performed": False,
        "rescue_performed": False,
    }
    return anchor_rows, summary


def serialize_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    return b"".join(
        canonical_json_bytes(row)
        for row in rows
    )


def run_gate(
    *,
    mamba370m_snapshot: Path,
    mamba14b_snapshot: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    facts, target_rows, _manifest = validate_structural_inputs(
        ROOT
    )

    snapshots = {
        "mamba370m": mamba370m_snapshot,
        "mamba14b": mamba14b_snapshot,
    }
    scale_rows: dict[str, list[dict[str, Any]]] = {}
    scale_summaries: dict[str, dict[str, Any]] = {}

    for scale in SCALES:
        rows, summary = analyze_scale(
            scale=scale,
            snapshot=snapshots[scale],
            facts=facts,
            target_rows=target_rows,
        )
        scale_rows[scale] = rows
        scale_summaries[scale] = summary

    combined_pass = all(
        scale_summaries[scale]["result"] == SCALE_PASS
        for scale in SCALES
    )
    cross = {
        "schema_version": CROSS_SCALE_SCHEMA,
        "result": RESULT_PASS if combined_pass else RESULT_BLOCKED,
        "source_pair_count": PAIR_COUNT,
        "target_row_count_per_scale": TARGET_ROW_COUNT,
        "scale_results": {
            scale: scale_summaries[scale]["result"]
            for scale in SCALES
        },
        "all_scales_pass": combined_pass,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "inference_performed": False,
        "training_executed": False,
        "row_filtering_performed": False,
        "rescue_performed": False,
    }
    return scale_rows, scale_summaries, cross


def write_outputs(
    *,
    output_dir: Path,
    scale_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    scale_summaries: Mapping[str, Mapping[str, Any]],
    cross: Mapping[str, Any],
) -> None:
    require(
        not output_dir.exists(),
        f"OUTPUT_COLLISION:{output_dir}",
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    generated: list[str] = []
    for scale in SCALES:
        anchor_name, summary_name = SCALE_OUTPUTS[scale]
        (output_dir / anchor_name).write_bytes(
            serialize_anchor_rows(scale_rows[scale])
        )
        (output_dir / summary_name).write_bytes(
            pretty_json_bytes(scale_summaries[scale])
        )
        generated.extend((anchor_name, summary_name))

    (output_dir / CROSS_SCALE_FILE).write_bytes(
        pretty_json_bytes(cross)
    )
    generated.append(CROSS_SCALE_FILE)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "result": cross["result"],
        "plan_freeze_commit": PLAN_FREEZE_COMMIT,
        "structural_freeze_commit": STRUCTURAL_FREEZE_COMMIT,
        "structural_source_sha256": EXPECTED_SOURCE_SHA256,
        "structural_row_sha256": EXPECTED_ROWS_SHA256,
        "source_pair_count": PAIR_COUNT,
        "target_row_count_per_scale": TARGET_ROW_COUNT,
        "scales": list(SCALES),
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "gpu_used": False,
        "scientific_outcomes_observed": False,
        "inference_performed": False,
        "training_executed": False,
        "row_filtering_performed": False,
        "rescue_performed": False,
    }
    (output_dir / ARTIFACT_FILE).write_bytes(
        pretty_json_bytes(artifact)
    )
    generated.append(ARTIFACT_FILE)

    (output_dir / CHECKSUM_FILE).write_text(
        "".join(
            f"{sha256_file(output_dir / name)}  {name}\n"
            for name in sorted(generated)
        ),
        encoding="utf-8",
        newline="\n",
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static dual-scale tokenizer-anchor eligibility gate for "
            "Gen4 low-displacement XG1 5401..5700. "
            "No model/checkpoint/GPU/forward."
        )
    )
    parser.add_argument(
        "--expected-head",
        required=True,
    )
    parser.add_argument(
        "--mamba370m-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--mamba14b-snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)
    authenticate_repo(args.expected_head)

    scale_rows, scale_summaries, cross = run_gate(
        mamba370m_snapshot=args.mamba370m_snapshot,
        mamba14b_snapshot=args.mamba14b_snapshot,
    )
    write_outputs(
        output_dir=args.output_dir,
        scale_rows=scale_rows,
        scale_summaries=scale_summaries,
        cross=cross,
    )

    print("RESULT=" + str(cross["result"]))
    for scale in SCALES:
        s = scale_summaries[scale]
        label = scale.upper()
        print(label + "_RESULT=" + str(s["result"]))
        print(
            label + "_TARGET_ROW_COUNT="
            + str(s["target_row_count"])
        )
        print(
            label + "_IDENTITY_ELIGIBLE="
            + str(s["eligible_identity_count"])
        )
        print(
            label + "_NAME_ELIGIBLE="
            + str(s["eligible_name_count"])
        )
        print(
            label + "_MISMATCH_COUNT="
            + str(s["identity_name_mismatch_count"])
        )
        print(
            label + "_EXCLUSION_COUNTS="
            + json.dumps(
                s["exclusion_counts"],
                sort_keys=True,
            )
        )
    print("MODEL_FORWARD_COUNT=0")
    print("CHECKPOINT_LOAD_COUNT=0")
    print("GPU_USED=False")
    print("SCIENTIFIC_OUTCOMES_OBSERVED=False")
    print("INFERENCE_PERFORMED=False")
    return 0 if bool(cross["all_scales_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
