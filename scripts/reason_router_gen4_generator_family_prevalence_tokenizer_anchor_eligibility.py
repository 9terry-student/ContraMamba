
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_generator_family_prevalence_cohorts as cohorts
from scripts import reason_router_gen4_xg1_tokenizer_anchor_eligibility as xg1


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
STRUCTURAL_FREEZE_COMMIT = "341de2398e1c06fffa73ee323502f52972c2d58e"
DESIGN_FREEZE_COMMIT = "4b6f0c831ebc89f0e786e1eb526739c7c9c06413"
BUILDER_PATH = "scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py"
BUILDER_BLOB = "4505acc99db0627733592694da290a007d281421"
XG1_GATE_PATH = "scripts/reason_router_gen4_xg1_tokenizer_anchor_eligibility.py"
XG1_GATE_BLOB = "6c98ce022ca134e385db28851fd364dc6daff423"
COHORT_ROOT = Path("data/reason_router_gen4_generator_family_prevalence_v1")

FROZEN = {
    "xg2": (
        "0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e",
        "c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa",
        "29afcee82a4f0251870b6a6ef8c8b626bc65acfc1a21b98b4c019a7b20de5704",
    ),
    "xg3": (
        "ce6014efe1916fa872f97ae9893f5b373c6b8d2b4c06bf0f9f8f85d2d247d18f",
        "70153d2fecfa6dcd6bb423e35ed6a7ef0da1e75153a6c64b64deecbfe06c4747",
        "beafc6f78304ee2c14285b9df40ddc8ab1eaeac7394fdf35176d97829b2c54ce",
    ),
    "xg4": (
        "23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0",
        "9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347",
        "bc7091577242a7610f5c7c87d8c9b1b8a4bb17ccb047eed48c48bddb9ebadc2a",
    ),
}

ANCHOR_PLAN = xg1.ANCHOR_PLAN
ANCHOR_EXPECTED_COUNTS = xg1.ANCHOR_EXPECTED_COUNTS
TARGET_IDENTITY_NAME_CELLS = xg1.TARGET_IDENTITY_NAME_CELLS
EXPECTED_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = 1800
EXPECTED_ANCHOR_ROWS = 1800
ANCHOR_SCHEMA = "GEN4_PREVALENCE_TOKENIZER_ANCHOR_ELIGIBILITY_V1"
SUMMARY_SCHEMA = "GEN4_PREVALENCE_TOKENIZER_ANCHOR_ELIGIBILITY_SUMMARY_V1"

EligibilityError = xg1.EligibilityError
require = xg1.require


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EligibilityError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate(root: Path) -> dict[str, str]:
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")
    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", STRUCTURAL_FREEZE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "STRUCTURAL_FREEZE_NOT_ANCESTOR")
    observed: dict[str, str] = {}
    for path, expected, label in (
        (BUILDER_PATH, BUILDER_BLOB, "BUILDER"),
        (XG1_GATE_PATH, XG1_GATE_BLOB, "XG1_GATE"),
    ):
        blob = git(root, "rev-parse", f"HEAD:{path}")
        require(blob == expected, f"{label}_BLOB_DRIFT:{blob}")
        for args, state in (
            (("diff", "--quiet", "--", path), "WORKTREE"),
            (("diff", "--cached", "--quiet", "--", path), "INDEX"),
        ):
            rc = subprocess.call(
                ["git", *args],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            require(rc == 0, f"{label}_{state}_DRIFT")
        observed[label.lower() + "_blob"] = blob
    return {
        "branch": branch,
        "head": head,
        "structural_freeze_commit": STRUCTURAL_FREEZE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        **observed,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL_OBJECT:{path}:{n}")
            out.append(value)
    return out


def load_family(
    family: str, root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    require(family in FROZEN, f"UNKNOWN_FAMILY:{family}")
    base = root / COHORT_ROOT / family
    source = base / "structured_source_facts.jsonl"
    rows_path = base / "synthetic_reason_router_six_cell.jsonl"
    manifest_path = base / "structural_manifest.json"
    for path in (source, rows_path, manifest_path):
        require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")

    source_sha, rows_sha, manifest_sha = FROZEN[family]
    require(xg1.sha256_file(source) == source_sha, f"SOURCE_SHA256:{family}")
    require(xg1.sha256_file(rows_path) == rows_sha, f"ROWS_SHA256:{family}")
    require(xg1.sha256_file(manifest_path) == manifest_sha, f"MANIFEST_SHA256:{family}")

    facts = _load_jsonl(source)
    rows = _load_jsonl(rows_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    spec = cohorts.family_spec(family)
    cohorts.validate_source_facts(spec, facts)
    cohorts.validate_materialized_rows(spec, rows, expected_pairs=EXPECTED_PAIR_COUNT)
    require(manifest["family_key"] == family, f"MANIFEST_FAMILY:{family}")
    require(manifest["source_file_sha256"] == source_sha, f"MANIFEST_SOURCE_SHA:{family}")
    require(manifest["row_file_sha256"] == rows_sha, f"MANIFEST_ROWS_SHA:{family}")
    return facts, rows, manifest


def overrides_for_cell(fact: Mapping[str, Any], cell_id: str) -> dict[str, str]:
    _mask, substitutions = cohorts.cell_spec(cell_id)
    return {axis: str(fact[source]) for axis, source in substitutions}


def realized_statement_and_spans(
    family: str,
    fact: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> tuple[str, dict[str, tuple[int, int]]]:
    spec = cohorts.family_spec(family)
    values = {**dict(fact), **dict(overrides)}
    rendered = cohorts.render_statement(spec, fact, **dict(overrides))
    title = str(values["title"])
    name = str(values["name"])
    identity = f"{title} {name}"

    require(
        rendered.count(identity) == 1,
        f"NONCONTIGUOUS_IDENTITY_ANCHOR:{family}:{fact['pair_id']}",
    )
    identity_start = rendered.index(identity)
    identity_end = identity_start + len(identity)
    name_start = identity_start + len(title) + 1
    name_end = name_start + len(name)
    require(rendered[name_start:name_end] == name, "NAME_SPAN_MISMATCH")
    return rendered, {
        "A_IDENTITY": (identity_start, identity_end),
        "A_NAME": (name_start, name_end),
    }


def topology_failures(
    family: str, facts: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for fact in facts:
        for cell_id in ("C0_SHAM", "C1_TITLE", "C2_NAME", "C5_TITLE_NAME"):
            try:
                realized_statement_and_spans(
                    family, fact, overrides_for_cell(fact, cell_id)
                )
            except EligibilityError as exc:
                failures.append({
                    "source_pair_id": str(fact["pair_id"]),
                    "contrast_cell_id": cell_id,
                    "error": str(exc),
                })
    return failures


def analyze_row(
    family: str,
    row: Mapping[str, Any],
    fact: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    cell_id = str(row["contrast_cell_id"])
    rendered, spans = realized_statement_and_spans(
        family, fact, overrides_for_cell(fact, cell_id)
    )
    require(rendered == row["evidence"], f"RENDER_IDENTITY:{row['row_id']}")

    claim_ids, _ = xg1._encoding(tokenizer, str(row["claim"]))
    evidence_ids, evidence_offsets = xg1._encoding(tokenizer, str(row["evidence"]))
    claim_kept = claim_ids[:xg1.CLAIM_BUDGET]
    evidence_kept = evidence_ids[:xg1.EVIDENCE_BUDGET]
    evidence_start = len(claim_kept) + 1
    terminal_index = evidence_start + len(evidence_kept) - 1
    require(terminal_index < xg1.MAX_LENGTH, "SERIALIZED_LENGTH_EXCEEDED")

    out: list[dict[str, Any]] = []
    for anchor_name in ANCHOR_PLAN[cell_id]:
        start, end = spans[anchor_name]
        evidence_index, error = xg1.final_overlapping_token_index(
            evidence_offsets, start, end
        )
        absolute = None
        eligible = False
        if evidence_index is not None:
            if evidence_index >= len(evidence_kept):
                error = "ANCHOR_NOT_CONSUMED_AFTER_EVIDENCE_TRUNCATION"
            else:
                absolute = evidence_start + evidence_index
                eligible = absolute + 4 <= terminal_index - 1
                if not eligible:
                    error = "POST4_PREFIX_INELIGIBLE"

        out.append({
            "schema_version": ANCHOR_SCHEMA,
            "family_key": family,
            "source_pair_id": str(row["source_pair_id"]),
            "row_id": str(row["row_id"]),
            "contrast_cell_id": cell_id,
            "anchor_name": anchor_name,
            "realized_semantic_text": rendered[start:end],
            "generator_span_start_char": start,
            "generator_span_end_char": end,
            "anchor_evidence_token_index": evidence_index,
            "absolute_anchor_token_index": absolute,
            "terminal_index": terminal_index,
            "post4_end_index": absolute + 4 if absolute is not None else None,
            "post4_rule": "a+4 <= terminal_index-1",
            "post4_eligible": eligible,
            "exclusion_code": error,
        })
    return out


def _base_summary(
    family: str,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    source_sha, rows_sha, manifest_sha = FROZEN[family]
    base = COHORT_ROOT / family
    return {
        "schema_version": SUMMARY_SCHEMA,
        "phase": "GENERATOR_FAMILY_PREVALENCE_TOKENIZER_ANCHOR_ELIGIBILITY",
        "family_key": family,
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
        "provenance": dict(provenance),
        "frozen_input": {
            "source_facts_path": (base / "structured_source_facts.jsonl").as_posix(),
            "source_facts_sha256": source_sha,
            "rows_path": (base / "synthetic_reason_router_six_cell.jsonl").as_posix(),
            "rows_sha256": rows_sha,
            "structural_manifest_path": (base / "structural_manifest.json").as_posix(),
            "structural_manifest_sha256": manifest_sha,
            "row_count": EXPECTED_ROW_COUNT,
            "source_pair_count": EXPECTED_PAIR_COUNT,
        },
    }


def compute_eligibility(
    *,
    family: str,
    root: Path = ROOT,
    tokenizer_snapshot: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance = authenticate(root)
    facts, rows, _manifest = load_family(family, root)
    failures = topology_failures(family, facts)
    summary = _base_summary(family, provenance)

    if failures:
        empty = xg1.serialize_anchor_manifest([])
        summary.update({
            "tokenizer_executed": False,
            "structural_anchor_topology_failure_count": len(failures),
            "structural_anchor_topology_failures": failures,
            "required_anchor_row_count": EXPECTED_ANCHOR_ROWS,
            "required_anchor_counts": ANCHOR_EXPECTED_COUNTS,
            "eligible_anchor_counts": {"A_IDENTITY": 0, "A_NAME": 0},
            "complete_source_pair_count": 0,
            "complete_source_pair_target": EXPECTED_PAIR_COUNT,
            "primary_complete_pair_prefix_feasibility":
                "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY",
            "anchor_manifest_sha256": xg1.sha256_bytes(empty),
        })
        return [], summary

    tokenizer, tokenizer_provenance = xg1.load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )
    facts_by_id = {str(f["pair_id"]): f for f in facts}
    anchor_rows: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["source_pair_id"])
        anchor_rows.extend(analyze_row(family, row, facts_by_id[pair_id], tokenizer))

    require(len(anchor_rows) == EXPECTED_ANCHOR_ROWS, "ANCHOR_ROW_COUNT")
    counts = Counter(r["anchor_name"] for r in anchor_rows)
    require(dict(counts) == ANCHOR_EXPECTED_COUNTS, f"ANCHOR_COUNTS:{dict(counts)}")

    eligible_counts = Counter(
        r["anchor_name"] for r in anchor_rows if r["post4_eligible"]
    )
    exclusions = Counter(
        r["exclusion_code"] for r in anchor_rows if r["exclusion_code"] is not None
    )
    pair_ok = {pair_id: True for pair_id in facts_by_id}
    for r in anchor_rows:
        if not r["post4_eligible"]:
            pair_ok[str(r["source_pair_id"])] = False

    lookup = {
        (str(r["source_pair_id"]), str(r["contrast_cell_id"]), str(r["anchor_name"])): r
        for r in anchor_rows
    }
    mismatches: list[dict[str, Any]] = []
    for pair_id in facts_by_id:
        for cell_id in TARGET_IDENTITY_NAME_CELLS:
            identity = lookup[(pair_id, cell_id, "A_IDENTITY")]
            name = lookup[(pair_id, cell_id, "A_NAME")]
            if identity["absolute_anchor_token_index"] != name["absolute_anchor_token_index"]:
                pair_ok[pair_id] = False
                mismatches.append({
                    "source_pair_id": pair_id,
                    "contrast_cell_id": cell_id,
                })

    complete = sum(pair_ok.values())
    verdict = (
        "PASS_300_OF_300"
        if complete == EXPECTED_PAIR_COUNT
        and sum(eligible_counts.values()) == EXPECTED_ANCHOR_ROWS
        and not mismatches
        else "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"
    )
    manifest_bytes = xg1.serialize_anchor_manifest(anchor_rows)
    summary.update({
        "tokenizer_executed": True,
        "tokenizer": tokenizer_provenance,
        "structural_anchor_topology_failure_count": 0,
        "structural_anchor_topology_failures": [],
        "required_anchor_row_count": EXPECTED_ANCHOR_ROWS,
        "required_anchor_counts": ANCHOR_EXPECTED_COUNTS,
        "eligible_anchor_counts": {
            key: int(eligible_counts.get(key, 0))
            for key in ANCHOR_EXPECTED_COUNTS
        },
        "target_identity_name_mismatch_count": len(mismatches),
        "target_identity_name_mismatches": mismatches,
        "exclusion_counts": dict(sorted(exclusions.items())),
        "complete_source_pair_count": complete,
        "complete_source_pair_target": EXPECTED_PAIR_COUNT,
        "primary_complete_pair_prefix_feasibility": verdict,
        "anchor_manifest_sha256": xg1.sha256_bytes(manifest_bytes),
        "coordinate_contract": {
            "active_serialization": "claim[:63]+EOS(0)+evidence[:64]",
            "anchor_mapping":
                "frozen_contiguous_title_name_span_final_overlapping_evidence_token",
            "post4_rule": "a+4 <= terminal_index-1",
            "shortened_post_window_allowed": False,
        },
    })
    return anchor_rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=tuple(FROZEN), required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_rows, summary = compute_eligibility(
        family=args.family,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )
    xg1.write_outputs(
        anchor_rows=anchor_rows,
        summary=summary,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
    )
    print("FAMILY =", args.family.upper())
    print("RESULT =", summary["primary_complete_pair_prefix_feasibility"])
    print("TOKENIZER_EXECUTED =", summary["tokenizer_executed"])
    print(
        "STRUCTURAL_ANCHOR_TOPOLOGY_FAILURE_COUNT =",
        summary["structural_anchor_topology_failure_count"],
    )
    print("COMPLETE_SOURCE_PAIR_COUNT =", summary["complete_source_pair_count"])
    print("MODEL_FORWARD_COUNT = 0")
    print("GPU_USED = FALSE")
    return 0 if summary["primary_complete_pair_prefix_feasibility"] == "PASS_300_OF_300" else 2


if __name__ == "__main__":
    raise SystemExit(main())
