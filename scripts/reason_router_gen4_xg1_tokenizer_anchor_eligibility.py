from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import build_reason_router_gen4_xg1_cross_generator_cohort as xg1


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-xg1-cross-generator-replication"
STRUCTURAL_FREEZE_COMMIT = "d9029801fd47636c155b1c846c433fc561424c8f"
DESIGN_FREEZE_COMMIT = "a31d2bc5ab4b939f52e969c89f3783feb9c3b233"

XG1_BUILDER_PATH = "scripts/build_reason_router_gen4_xg1_cross_generator_cohort.py"
XG1_BUILDER_BLOB = "c830026935a6c9f4990c6a3315c75fd5580e7264"

COHORT_DIR = Path("data/reason_router_gen4_xg1_cross_generator_v1")
SOURCE_FACTS_PATH = COHORT_DIR / "structured_source_facts.jsonl"
ROWS_PATH = COHORT_DIR / "synthetic_reason_router_six_cell.jsonl"
STRUCTURAL_MANIFEST_PATH = COHORT_DIR / "structural_manifest.json"

EXPECTED_SOURCE_FACTS_SHA256 = (
    "fccd6821eeb97194d5b898aca4911eaba71e893df7fe27c910aa37255a5695e0"
)
EXPECTED_ROWS_SHA256 = (
    "6ea0484517e0ae7479ad7f3b0a74af4d75f7f7353d29586c597f2a9fee1e649f"
)
EXPECTED_STRUCTURAL_MANIFEST_SHA256 = (
    "f7c881dd1a4a400e600eea4b03b46e05a48bf0da07d29905462fc3543d8af822"
)

EXPECTED_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = 1800
CANONICAL_CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C3_ROLE",
    "C4_PREDICATE",
    "C5_TITLE_NAME",
)

# Exact active-encoding contract inherited unchanged from the completed
# prospective validation.
MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64
EOS_TOKEN_ID = 0
EFFECTIVE_PAD_TOKEN_ID = 0

TOKENIZER_ACTIVE_ENCODING_VALIDATION_COMMIT = (
    "17f1ddfc8286796f27c4a61716a21e14126bb836"
)
CANONICAL_ANALYSIS_TOKENIZER_REFERENCE = (
    "40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37"
)
CANONICAL_ANALYSIS_TOKENIZER_IS_CLAIMED_HISTORICAL_SNAPSHOT = False
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
TOKENIZER_FILE_SHA256 = {
    "tokenizer.json":
        "b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf",
    "tokenizer_config.json":
        "9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb",
    "special_tokens_map.json":
        "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8",
}

ANCHOR_PLAN: dict[str, tuple[str, ...]] = {
    "C0_SHAM": ("A_IDENTITY", "A_NAME"),
    "C1_TITLE": ("A_IDENTITY",),
    "C2_NAME": ("A_IDENTITY", "A_NAME"),
    "C3_ROLE": (),
    "C4_PREDICATE": (),
    "C5_TITLE_NAME": ("A_IDENTITY",),
}
ANCHOR_EXPECTED_COUNTS = {
    "A_IDENTITY": 1200,
    "A_NAME": 600,
}
EXPECTED_ANCHOR_ROWS = 1800
TARGET_IDENTITY_NAME_CELLS = ("C0_SHAM", "C2_NAME")

ANCHOR_MANIFEST_SCHEMA = "GEN4_XG1_TOKENIZER_ANCHOR_ELIGIBILITY_V1"
SUMMARY_SCHEMA = "GEN4_XG1_TOKENIZER_ANCHOR_ELIGIBILITY_SUMMARY_V1"


class EligibilityError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise EligibilityError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EligibilityError("GIT_FAILURE:" + " ".join(args)) from exc


def authenticate_repository(root: Path = ROOT) -> dict[str, str]:
    root = root.resolve()
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")

    require(branch == EXPECTED_BRANCH, f"BRANCH_MISMATCH:{branch}")

    rc = subprocess.call(
        ["git", "merge-base", "--is-ancestor", STRUCTURAL_FREEZE_COMMIT, head],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "STRUCTURAL_FREEZE_NOT_ANCESTOR")

    observed_blob = git(root, "rev-parse", f"HEAD:{XG1_BUILDER_PATH}")
    require(
        observed_blob == XG1_BUILDER_BLOB,
        f"XG1_BUILDER_BLOB_DRIFT:{observed_blob}",
    )

    for diff_args, label in (
        (("diff", "--quiet", "--", XG1_BUILDER_PATH), "WORKTREE"),
        (("diff", "--cached", "--quiet", "--", XG1_BUILDER_PATH), "INDEX"),
    ):
        rc = subprocess.call(
            ["git", *diff_args],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(rc == 0, f"XG1_BUILDER_{label}_DRIFT")

    return {
        "branch": branch,
        "head": head,
        "structural_freeze_commit": STRUCTURAL_FREEZE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "xg1_builder_blob": XG1_BUILDER_BLOB,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"JSONL_OBJECT_REQUIRED:{path}:{line_no}")
        rows.append(value)
    return rows, raw


def validate_structural_manifest(manifest: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": "GEN4_XG1_STRUCTURAL_MANIFEST_V1",
        "result": "PASS_XG1_STRUCTURAL_INDEPENDENCE",
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "design_sha256":
            "21ce2dd133721766a9a18d924774b8f02aa026364e1102809ea839c3985bf403",
        "generator_family": "xg1_independent_structured_records_v1",
        "source_pair_count": EXPECTED_PAIR_COUNT,
        "row_count": EXPECTED_ROW_COUNT,
        "rows_per_pair": 6,
        "pair_id_first": "xg1_fact_001",
        "pair_id_last": "xg1_fact_300",
        "source_file_sha256": EXPECTED_SOURCE_FACTS_SHA256,
        "row_file_sha256": EXPECTED_ROWS_SHA256,
        "exact_inventory_overlap_count": 0,
        "casefold_inventory_overlap_count": 0,
        "discovery_claim_overlap_count": 0,
        "discovery_evidence_overlap_count": 0,
        "prior_holdout_claim_overlap_count": 0,
        "prior_holdout_evidence_overlap_count": 0,
        "deterministic_byte_regeneration": True,
        "production_builder_uses_historical_generator": False,
        "labels_present": False,
        "model_geometry_present": False,
        "endpoint_values_present": False,
        "response_fields_present": False,
        "tokenizer_executed": False,
        "model_executed": False,
        "cuda_executed": False,
    }
    for key, expected in exact.items():
        require(
            manifest.get(key) == expected,
            f"STRUCTURAL_MANIFEST_DRIFT:{key}",
        )


def load_frozen_xg1_inputs(
    root: Path = ROOT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    source_path = root / SOURCE_FACTS_PATH
    rows_path = root / ROWS_PATH
    manifest_path = root / STRUCTURAL_MANIFEST_PATH

    for path in (source_path, rows_path, manifest_path):
        require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")

    require(
        sha256_file(source_path) == EXPECTED_SOURCE_FACTS_SHA256,
        "SOURCE_FACTS_SHA256_MISMATCH",
    )
    require(
        sha256_file(rows_path) == EXPECTED_ROWS_SHA256,
        "ROWS_SHA256_MISMATCH",
    )
    require(
        sha256_file(manifest_path) == EXPECTED_STRUCTURAL_MANIFEST_SHA256,
        "STRUCTURAL_MANIFEST_SHA256_MISMATCH",
    )

    manifest = _load_json(manifest_path)
    validate_structural_manifest(manifest)

    facts, _facts_raw = _load_jsonl(source_path)
    rows, _rows_raw = _load_jsonl(rows_path)

    xg1.validate_source_facts(facts)
    xg1.validate_materialized_rows(rows, expected_pairs=EXPECTED_PAIR_COUNT)

    require(len(facts) == EXPECTED_PAIR_COUNT, "SOURCE_FACT_COUNT")
    require(len(rows) == EXPECTED_ROW_COUNT, "ROW_COUNT")

    expected_pair_ids = [f"xg1_fact_{i:03d}" for i in range(1, 301)]
    observed_pair_ids = [str(fact["pair_id"]) for fact in facts]
    require(observed_pair_ids == expected_pair_ids, "PAIR_ID_ORDER")

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_pair_id"])].append(str(row["contrast_cell_id"]))
    for pair_id in expected_pair_ids:
        require(
            grouped[pair_id] == list(CANONICAL_CELLS),
            f"SIX_CELL_ORDER_DRIFT:{pair_id}",
        )

    return facts, rows, manifest


def canonical_tokenizer_snapshot_dir(
    home: str | Path | None = None,
) -> Path:
    home_path = Path(home) if home is not None else Path.home()
    return (
        home_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--state-spaces--mamba-130m-hf"
        / "snapshots"
        / CANONICAL_ANALYSIS_TOKENIZER_REFERENCE
    )


def _token_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def load_canonical_analysis_tokenizer(
    snapshot_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    if snapshot_dir is None:
        snapshot_dir = canonical_tokenizer_snapshot_dir()
    snapshot_dir = Path(snapshot_dir)

    require(
        snapshot_dir.name == CANONICAL_ANALYSIS_TOKENIZER_REFERENCE,
        "TOKENIZER_SNAPSHOT_REFERENCE_MISMATCH",
    )

    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise EligibilityError("TOKENIZERS_PACKAGE_REQUIRED") from exc

    require(
        tokenizers.__version__ == EXPECTED_TOKENIZERS_VERSION,
        "TOKENIZERS_VERSION_MISMATCH:"
        f"expected={EXPECTED_TOKENIZERS_VERSION}:observed={tokenizers.__version__}",
    )

    observed_hashes: dict[str, str] = {}
    for filename, expected in TOKENIZER_FILE_SHA256.items():
        path = snapshot_dir / filename
        require(path.is_file(), f"MISSING_TOKENIZER_FILE:{path}")
        observed = sha256_file(path)
        require(
            observed == expected,
            f"TOKENIZER_FILE_SHA256_MISMATCH:{filename}",
        )
        observed_hashes[filename] = observed

    config = _load_json(snapshot_dir / "tokenizer_config.json")
    special = _load_json(snapshot_dir / "special_tokens_map.json")

    eos_candidates = [
        _token_content(special.get("eos_token")),
        _token_content(config.get("eos_token")),
    ]
    eos_candidates = [x for x in eos_candidates if x is not None]
    require(
        bool(eos_candidates) and len(set(eos_candidates)) == 1,
        "EOS_TOKEN_STRING_NOT_UNIQUELY_RECOVERABLE",
    )

    tokenizer = Tokenizer.from_file(str(snapshot_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    eos_id = tokenizer.token_to_id(eos_candidates[0])
    require(eos_id == EOS_TOKEN_ID, f"EOS_TOKEN_ID_MISMATCH:{eos_id}")

    pad_candidates = [
        _token_content(special.get("pad_token")),
        _token_content(config.get("pad_token")),
    ]
    pad_candidates = [x for x in pad_candidates if x is not None]
    require(
        len(set(pad_candidates)) <= 1,
        "PAD_TOKEN_DECLARATIONS_DISAGREE",
    )
    pad_id = tokenizer.token_to_id(pad_candidates[0]) if pad_candidates else eos_id
    require(
        pad_id == EFFECTIVE_PAD_TOKEN_ID,
        f"EFFECTIVE_PAD_TOKEN_ID_MISMATCH:{pad_id}",
    )

    provenance = {
        "analysis_reference": CANONICAL_ANALYSIS_TOKENIZER_REFERENCE,
        "claimed_historical_snapshot":
            CANONICAL_ANALYSIS_TOKENIZER_IS_CLAIMED_HISTORICAL_SNAPSHOT,
        "tokenizers_version": tokenizers.__version__,
        "file_sha256": observed_hashes,
        "eos_token_id": eos_id,
        "effective_pad_token_id": pad_id,
        "add_special_tokens": False,
        "max_length": MAX_LENGTH,
        "claim_budget": CLAIM_BUDGET,
        "evidence_budget": EVIDENCE_BUDGET,
        "serialization": "claim[:63]+EOS(0)+evidence[:64]",
        "active_encoding_validation_commit":
            TOKENIZER_ACTIVE_ENCODING_VALIDATION_COMMIT,
    }
    return tokenizer, provenance


def realized_statement_and_spans(
    fact: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> tuple[str, dict[str, tuple[int, int]]]:
    values = {**dict(fact), **dict(overrides)}

    time = str(values["time"])
    location = str(values["location"])
    title = str(values["title"])
    name = str(values["name"])
    role = str(values["role"])
    predicate = str(values["predicate"])
    obj = str(values["object"])

    prefix = f"During {time}, records from {location} identify "
    title_start = len(prefix)
    title_end = title_start + len(title)

    name_start = title_end + 1
    name_end = name_start + len(name)

    role_start = name_end + len(" as ")
    role_end = role_start + len(role)

    predicate_start = role_end + len("; this person ")
    predicate_end = predicate_start + len(predicate)

    rendered = (
        f"During {time}, records from {location} identify "
        f"{title} {name} as {role}; this person {predicate} {obj}."
    )

    spans = {
        "A_TITLE": (title_start, title_end),
        "A_NAME": (name_start, name_end),
        "A_ROLE": (role_start, role_end),
        "A_PREDICATE": (predicate_start, predicate_end),
        "A_IDENTITY": (title_start, name_end),
    }

    for anchor, (start, end) in spans.items():
        require(0 <= start < end <= len(rendered), f"INVALID_SPAN:{anchor}")

    require(
        rendered == xg1.render_statement(fact, **dict(overrides)),
        "XG1_RENDERER_IDENTITY_MISMATCH",
    )
    return rendered, spans


def final_overlapping_token_index(
    offsets: Sequence[tuple[int, int]] | Sequence[Sequence[int]],
    span_start: int,
    span_end: int,
) -> tuple[int | None, str | None]:
    overlapping: list[int] = []
    for index, raw_offset in enumerate(offsets):
        start = int(raw_offset[0])
        end = int(raw_offset[1])
        if end <= start:
            continue
        if end > span_start and start < span_end:
            overlapping.append(index)

    if not overlapping:
        return None, "SPAN_NOT_MAPPED"

    first = overlapping[0]
    last = overlapping[-1]
    if int(offsets[first][0]) > span_start or int(offsets[last][1]) < span_end:
        return None, "SPAN_MAPPING_INCOMPLETE"

    return last, None


def _encoding(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(value) for value in encoded.ids]
    offsets = [(int(a), int(b)) for a, b in encoded.offsets]
    require(len(ids) == len(offsets), "TOKEN_ID_OFFSET_LENGTH_MISMATCH")
    require(bool(ids), "EMPTY_TOKENIZATION")
    return ids, offsets


def _overrides_for_cell(
    fact: Mapping[str, Any],
    cell_id: str,
) -> dict[str, str]:
    _mask, substitutions = xg1.cell_spec(cell_id)
    return {axis: str(fact[source]) for axis, source in substitutions}


def analyze_required_anchors_for_row(
    row: Mapping[str, Any],
    fact: Mapping[str, Any],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    cell_id = str(row["contrast_cell_id"])
    require(cell_id in ANCHOR_PLAN, f"UNKNOWN_CELL:{cell_id}")

    overrides = _overrides_for_cell(fact, cell_id)
    rendered, spans = realized_statement_and_spans(fact, overrides)
    require(
        rendered == row["evidence"],
        f"RENDER_IDENTITY_MISMATCH:{row['row_id']}",
    )

    claim_ids, _claim_offsets = _encoding(tokenizer, str(row["claim"]))
    evidence_ids, evidence_offsets = _encoding(tokenizer, str(row["evidence"]))

    claim_kept = claim_ids[:CLAIM_BUDGET]
    evidence_kept = evidence_ids[:EVIDENCE_BUDGET]
    require(bool(claim_kept), f"EMPTY_CLAIM_AFTER_ENCODING:{row['row_id']}")
    require(bool(evidence_kept), f"EMPTY_EVIDENCE_AFTER_ENCODING:{row['row_id']}")

    serialized_length = len(claim_kept) + 1 + len(evidence_kept)
    require(serialized_length <= MAX_LENGTH, "SERIALIZED_LENGTH_EXCEEDED")
    evidence_start = len(claim_kept) + 1
    terminal_index = serialized_length - 1

    results: list[dict[str, Any]] = []
    for anchor_name in ANCHOR_PLAN[cell_id]:
        span_start, span_end = spans[anchor_name]
        anchor_evidence_index, mapping_error = final_overlapping_token_index(
            evidence_offsets,
            span_start,
            span_end,
        )

        absolute_index: int | None = None
        prefix_eligible = False
        exclusion_code: str | None = mapping_error
        anchor_token_offset: list[int] | None = None

        if anchor_evidence_index is not None:
            anchor_token_offset = [
                int(evidence_offsets[anchor_evidence_index][0]),
                int(evidence_offsets[anchor_evidence_index][1]),
            ]
            if anchor_evidence_index >= len(evidence_kept):
                exclusion_code = "ANCHOR_NOT_CONSUMED_AFTER_EVIDENCE_TRUNCATION"
            else:
                absolute_index = evidence_start + anchor_evidence_index
                prefix_eligible = absolute_index + 4 <= terminal_index - 1
                if not prefix_eligible:
                    exclusion_code = "POST4_PREFIX_INELIGIBLE"

        results.append({
            "schema_version": ANCHOR_MANIFEST_SCHEMA,
            "source_pair_id": str(row["source_pair_id"]),
            "row_id": str(row["row_id"]),
            "contrast_cell_id": cell_id,
            "anchor_name": anchor_name,
            "realized_semantic_text": rendered[span_start:span_end],
            "generator_span_start_char": span_start,
            "generator_span_end_char": span_end,
            "anchor_evidence_token_index": anchor_evidence_index,
            "anchor_token_offset": anchor_token_offset,
            "absolute_anchor_token_index": absolute_index,
            "claim_raw_token_count": len(claim_ids),
            "claim_consumed_token_count": len(claim_kept),
            "evidence_raw_token_count": len(evidence_ids),
            "evidence_consumed_token_count": len(evidence_kept),
            "evidence_start": evidence_start,
            "terminal_index": terminal_index,
            "post4_end_index":
                (absolute_index + 4 if absolute_index is not None else None),
            "post4_rule": "a+4 <= terminal_index-1",
            "post4_eligible": prefix_eligible,
            "exclusion_code": exclusion_code,
        })

    return results


def serialize_anchor_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    if not rows:
        return b""
    text = "\n".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    ) + "\n"
    return text.encode("utf-8")


def compute_eligibility(
    *,
    root: Path = ROOT,
    tokenizer_snapshot: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provenance = authenticate_repository(root)
    facts, rows, _structural_manifest = load_frozen_xg1_inputs(root)
    tokenizer, tokenizer_provenance = load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )

    facts_by_id = {str(fact["pair_id"]): fact for fact in facts}
    require(len(facts_by_id) == EXPECTED_PAIR_COUNT, "FACT_ID_COUNT_MISMATCH")

    anchor_rows: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in facts_by_id, f"MISSING_FACT_FOR_PAIR:{pair_id}")
        anchor_rows.extend(
            analyze_required_anchors_for_row(
                row,
                facts_by_id[pair_id],
                tokenizer,
            )
        )

    require(len(anchor_rows) == EXPECTED_ANCHOR_ROWS, "ANCHOR_ROW_COUNT_MISMATCH")

    anchor_counts = Counter(row["anchor_name"] for row in anchor_rows)
    require(
        dict(anchor_counts) == ANCHOR_EXPECTED_COUNTS,
        f"ANCHOR_COUNT_MISMATCH:{dict(anchor_counts)}",
    )

    eligible_anchor_counts = Counter(
        row["anchor_name"]
        for row in anchor_rows
        if row["post4_eligible"]
    )
    exclusion_counts = Counter(
        row["exclusion_code"]
        for row in anchor_rows
        if row["exclusion_code"] is not None
    )

    pair_ok = {pair_id: True for pair_id in facts_by_id}
    for row in anchor_rows:
        if not row["post4_eligible"]:
            pair_ok[str(row["source_pair_id"])] = False

    event_lookup = {
        (
            str(row["source_pair_id"]),
            str(row["contrast_cell_id"]),
            str(row["anchor_name"]),
        ): row
        for row in anchor_rows
    }
    require(
        len(event_lookup) == EXPECTED_ANCHOR_ROWS,
        "DUPLICATE_ANCHOR_EVENT_KEY",
    )

    target_identity_name_mismatches: list[dict[str, Any]] = []
    for pair_id in facts_by_id:
        for cell_id in TARGET_IDENTITY_NAME_CELLS:
            identity = event_lookup[(pair_id, cell_id, "A_IDENTITY")]
            name = event_lookup[(pair_id, cell_id, "A_NAME")]
            if (
                identity["absolute_anchor_token_index"]
                != name["absolute_anchor_token_index"]
            ):
                pair_ok[pair_id] = False
                target_identity_name_mismatches.append({
                    "source_pair_id": pair_id,
                    "contrast_cell_id": cell_id,
                    "identity_absolute_anchor_token_index":
                        identity["absolute_anchor_token_index"],
                    "name_absolute_anchor_token_index":
                        name["absolute_anchor_token_index"],
                })

    complete_pair_count = sum(pair_ok.values())
    all_eligible = (
        complete_pair_count == EXPECTED_PAIR_COUNT
        and sum(eligible_anchor_counts.values()) == EXPECTED_ANCHOR_ROWS
        and not target_identity_name_mismatches
    )
    verdict = (
        "PASS_300_OF_300"
        if all_eligible
        else "BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY"
    )

    claim_truncation_count = sum(
        1 for row in rows
        if len(_encoding(tokenizer, str(row["claim"]))[0]) > CLAIM_BUDGET
    )
    evidence_truncation_count = sum(
        1 for row in rows
        if len(_encoding(tokenizer, str(row["evidence"]))[0]) > EVIDENCE_BUDGET
    )

    manifest_bytes = serialize_anchor_manifest(anchor_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "phase": "XG1_TOKENIZER_ANCHOR_ELIGIBILITY",
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
        "provenance": provenance,
        "frozen_input": {
            "source_facts_path": SOURCE_FACTS_PATH.as_posix(),
            "source_facts_sha256": EXPECTED_SOURCE_FACTS_SHA256,
            "rows_path": ROWS_PATH.as_posix(),
            "rows_sha256": EXPECTED_ROWS_SHA256,
            "structural_manifest_path": STRUCTURAL_MANIFEST_PATH.as_posix(),
            "structural_manifest_sha256":
                EXPECTED_STRUCTURAL_MANIFEST_SHA256,
            "row_count": EXPECTED_ROW_COUNT,
            "source_pair_count": EXPECTED_PAIR_COUNT,
        },
        "tokenizer": tokenizer_provenance,
        "coordinate_contract": {
            "active_serialization": "claim[:63]+EOS(0)+evidence[:64]",
            "anchor_mapping":
                "xg1_generator_declared_span_final_overlapping_evidence_token_"
                "mapped_independently_per_cell",
            "post4_rule": "a+4 <= terminal_index-1",
            "shortened_post_window_allowed": False,
        },
        "required_anchor_row_count": EXPECTED_ANCHOR_ROWS,
        "required_anchor_counts": ANCHOR_EXPECTED_COUNTS,
        "eligible_anchor_counts": {
            anchor: int(eligible_anchor_counts.get(anchor, 0))
            for anchor in ANCHOR_EXPECTED_COUNTS
        },
        "target_identity_name_cells": list(TARGET_IDENTITY_NAME_CELLS),
        "target_identity_name_mismatch_count":
            len(target_identity_name_mismatches),
        "target_identity_name_mismatches":
            target_identity_name_mismatches,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "claim_truncation_row_count": claim_truncation_count,
        "evidence_truncation_row_count": evidence_truncation_count,
        "complete_source_pair_count": complete_pair_count,
        "complete_source_pair_target": EXPECTED_PAIR_COUNT,
        "primary_complete_pair_prefix_feasibility": verdict,
        "anchor_manifest_sha256": sha256_bytes(manifest_bytes),
        "claim_boundary": [
            "This artifact establishes XG1 tokenizer/event-anchor prefix eligibility only.",
            "The canonical tokenizer is a reproducible active-encoding analysis reference, not a claimed historical snapshot.",
            "No model forward, checkpoint load, native-state extraction, kinematic endpoint computation, or scientific evaluation is performed.",
            "Eligibility PASS does not itself authorize scientific model execution.",
        ],
    }
    return anchor_rows, summary


def write_outputs(
    *,
    anchor_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_jsonl: Path,
    summary_path: Path,
) -> None:
    require(not output_jsonl.exists(), f"OUTPUT_COLLISION:{output_jsonl}")
    require(not summary_path.exists(), f"OUTPUT_COLLISION:{summary_path}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_bytes = serialize_anchor_manifest(anchor_rows)
    output_jsonl.write_bytes(manifest_bytes)

    require(
        sha256_file(output_jsonl) == summary["anchor_manifest_sha256"],
        "WRITTEN_ANCHOR_MANIFEST_SHA256_MISMATCH",
    )

    summary_path.write_text(
        json.dumps(
            dict(summary),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static tokenizer/event-anchor eligibility gate for the frozen "
            "Gen4-K XG1 cross-generator cohort. No model load/forward."
        )
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=None,
        help=(
            "Canonical analysis tokenizer snapshot. Defaults to the frozen "
            "40e5... Hugging Face cache path under the current home directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    anchor_rows, summary = compute_eligibility(
        root=ROOT,
        tokenizer_snapshot=args.tokenizer_snapshot,
    )
    write_outputs(
        anchor_rows=anchor_rows,
        summary=summary,
        output_jsonl=args.output_jsonl,
        summary_path=args.summary,
    )

    print("RESULT =", summary["primary_complete_pair_prefix_feasibility"])
    print("SOURCE_PAIR_COUNT =", summary["complete_source_pair_count"])
    print("ANCHOR_ROW_COUNT =", summary["required_anchor_row_count"])
    print("ELIGIBLE_ANCHOR_COUNTS =", summary["eligible_anchor_counts"])
    print("EXCLUSION_COUNTS =", summary["exclusion_counts"])
    print("MODEL_FORWARD_COUNT = 0")
    print("CHECKPOINT_LOAD_COUNT = 0")
    print("GPU_USED = FALSE")

    return (
        0
        if summary["primary_complete_pair_prefix_feasibility"]
        == "PASS_300_OF_300"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
