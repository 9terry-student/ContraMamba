from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BRANCH = "gen4-k-large-correction-prospective-validation"
PROSPECTIVE_HOLDOUT_FREEZE_COMMIT = "c5abe9e3fd551c49ab67f68e2cd8fb3100a7c91e"
PROSPECTIVE_DESIGN_FREEZE_COMMIT = "9c8ef022b76c7e25ecefb2568b13a534cc317720"
DISCOVERY_FREEZE_COMMIT = "41f4678d09f7c779a8edc2eec6cc08a9effd9f41"

GENERATOR_PATH = "scripts/build_controlled_v5.py"
GENERATOR_SOURCE_BLOB = "baee23a9f71333125f4a8735c2c92d20cab7eb4f"
MATERIALIZER_PATH = "scripts/materialize_reason_router_gen4_six_cell_contrast.py"
MATERIALIZER_SOURCE_BLOB = "6a0f5bf58614cdff0dad5f78c7d4bd86d507cd3b"

PROSPECTIVE_DATA_PATH = Path(
    "data/reason_router_gen4_large_correction_prospective_holdout_v1/"
    "synthetic_reason_router_six_cell.jsonl"
)
PROSPECTIVE_MANIFEST_PATH = Path(
    "reports/reason_router_gen4_large_correction_prospective_holdout_materialization_manifest.json"
)
EXPECTED_PROSPECTIVE_ROWS_SHA256 = (
    "f4289173ba3837fad217728830b8aefb8baa7dce4f45dde7c58441021b436bdc"
)
EXPECTED_VALIDATION_PAIR_IDS_SHA256 = (
    "68ae2c29fa9da23a2c1d541bba59d1597f3f2ca4959bb6986d78d7784da1c90a"
)
EXPECTED_PAIR_COUNT = 300
EXPECTED_ROW_COUNT = 1800
VALIDATION_PAIR_START = 300
VALIDATION_PAIR_STOP = 600

CANONICAL_CELLS = (
    "C0_SHAM",
    "C1_TITLE",
    "C2_NAME",
    "C3_ROLE",
    "C4_PREDICATE",
    "C5_TITLE_NAME",
)

# Exact active-encoding contract.
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

# Exact event plan inherited from the closed directional-alignment transport runner:
# identity anchor on C0/C1/C2/C5; name anchor additionally on target C0/C2,
# with A_IDENTITY == A_NAME required on those target branches.
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

ANCHOR_MANIFEST_SCHEMA = (
    "GEN4_LARGE_CORRECTION_PROSPECTIVE_TOKENIZER_ANCHOR_ELIGIBILITY_V1"
)
SUMMARY_SCHEMA = (
    "GEN4_LARGE_CORRECTION_PROSPECTIVE_TOKENIZER_ANCHOR_ELIGIBILITY_SUMMARY_V1"
)


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


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PROSPECTIVE_HOLDOUT_FREEZE_COMMIT,
            head,
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(rc == 0, "PROSPECTIVE_HOLDOUT_FREEZE_NOT_ANCESTOR")

    for rel_path, expected_blob in (
        (GENERATOR_PATH, GENERATOR_SOURCE_BLOB),
        (MATERIALIZER_PATH, MATERIALIZER_SOURCE_BLOB),
    ):
        observed_blob = git(root, "rev-parse", f"HEAD:{rel_path}")
        require(
            observed_blob == expected_blob,
            f"FROZEN_SOURCE_BLOB_DRIFT:{rel_path}:{observed_blob}",
        )
        diff_rc = subprocess.call(
            ["git", "diff", "--quiet", "--", rel_path],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(diff_rc == 0, f"FROZEN_SOURCE_WORKTREE_DRIFT:{rel_path}")

        staged_diff_rc = subprocess.call(
            ["git", "diff", "--cached", "--quiet", "--", rel_path],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        require(
            staged_diff_rc == 0,
            f"FROZEN_SOURCE_INDEX_DRIFT:{rel_path}",
        )

    return {
        "branch": branch,
        "head": head,
        "prospective_holdout_freeze_commit": PROSPECTIVE_HOLDOUT_FREEZE_COMMIT,
        "prospective_design_freeze_commit": PROSPECTIVE_DESIGN_FREEZE_COMMIT,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "generator_source_blob": GENERATOR_SOURCE_BLOB,
        "materializer_source_blob": MATERIALIZER_SOURCE_BLOB,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def validate_materialization_manifest(
    manifest: Mapping[str, Any],
) -> None:
    exact = {
        "schema_version": "GEN4_LARGE_CORRECTION_PROSPECTIVE_COHORT_V1",
        "phase": "PROSPECTIVE_HOLDOUT_MATERIALIZATION",
        "branch": EXPECTED_BRANCH,
        "discovery_freeze_commit": DISCOVERY_FREEZE_COMMIT,
        "generator_source_blob": GENERATOR_SOURCE_BLOB,
        "materializer_source_blob": MATERIALIZER_SOURCE_BLOB,
        "mechanism_id": "masked_slot_substitution_v1",
        "validation_pair_count": EXPECTED_PAIR_COUNT,
        "validation_row_count": EXPECTED_ROW_COUNT,
        "first_validation_pair_id": "generated_fact_301",
        "last_validation_pair_id": "generated_fact_600",
        "rows_sha256": EXPECTED_PROSPECTIVE_ROWS_SHA256,
        "validation_pair_ids_sha256": EXPECTED_VALIDATION_PAIR_IDS_SHA256,
        "tokenizer_invoked": False,
        "model_forward_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "scientific_outcomes_observed": False,
    }
    for key, expected in exact.items():
        require(
            manifest.get(key) == expected,
            f"MATERIALIZATION_MANIFEST_DRIFT:{key}",
        )

    require(
        manifest.get("validation_pair_index_range") == [300, 600],
        "VALIDATION_PAIR_RANGE_DRIFT",
    )
    require(
        manifest.get("discovery_pair_index_range") == [0, 300],
        "DISCOVERY_PAIR_RANGE_DRIFT",
    )
    require(
        manifest.get("contrast_cells") == list(CANONICAL_CELLS),
        "CONTRAST_CELL_ORDER_DRIFT",
    )


def load_frozen_prospective_rows(
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    data_path = root / PROSPECTIVE_DATA_PATH
    manifest_path = root / PROSPECTIVE_MANIFEST_PATH

    require(data_path.is_file(), f"MISSING_PROSPECTIVE_DATA:{data_path}")
    require(
        manifest_path.is_file(),
        f"MISSING_PROSPECTIVE_MANIFEST:{manifest_path}",
    )

    raw = data_path.read_bytes()
    require(
        sha256_bytes(raw) == EXPECTED_PROSPECTIVE_ROWS_SHA256,
        "PROSPECTIVE_ROWS_SHA256_MISMATCH",
    )

    manifest = _load_json(manifest_path)
    validate_materialization_manifest(manifest)

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(
            isinstance(value, dict),
            f"PROSPECTIVE_JSONL_OBJECT_REQUIRED:{line_no}",
        )
        rows.append(value)

    require(len(rows) == EXPECTED_ROW_COUNT, "PROSPECTIVE_ROW_COUNT_MISMATCH")
    pair_ids = [str(row["source_pair_id"]) for row in rows]
    unique_pair_ids = list(dict.fromkeys(pair_ids))
    require(len(unique_pair_ids) == EXPECTED_PAIR_COUNT, "PAIR_COUNT_MISMATCH")
    require(
        unique_pair_ids[0] == "generated_fact_301"
        and unique_pair_ids[-1] == "generated_fact_600",
        "PAIR_BOUNDARY_MISMATCH",
    )
    pair_id_hash = sha256_bytes(
        ("\n".join(unique_pair_ids) + "\n").encode("utf-8")
    )
    require(
        pair_id_hash == EXPECTED_VALIDATION_PAIR_IDS_SHA256,
        "PAIR_ID_SHA256_MISMATCH",
    )

    cell_counts = Counter(str(row["contrast_cell_id"]) for row in rows)
    require(
        cell_counts == Counter({cell: EXPECTED_PAIR_COUNT for cell in CANONICAL_CELLS}),
        "PROSPECTIVE_CELL_COUNTS_MISMATCH",
    )

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_pair_id"])].append(str(row["contrast_cell_id"]))
    for pair_id in unique_pair_ids:
        require(
            grouped[pair_id] == list(CANONICAL_CELLS),
            f"SIX_CELL_ORDER_DRIFT:{pair_id}",
        )

    return rows, raw, manifest


def _lazy_frozen_modules(root: Path) -> tuple[Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    generator = importlib.import_module("scripts.build_controlled_v5")
    materializer = importlib.import_module(
        "scripts.materialize_reason_router_gen4_six_cell_contrast"
    )
    return generator, materializer


def reconstruct_validation_facts_and_verify_bytes(
    rows_raw: bytes,
    *,
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any]:
    generator, materializer = _lazy_frozen_modules(root)
    facts = generator.fact_templates_for_count(VALIDATION_PAIR_STOP)
    require(len(facts) == VALIDATION_PAIR_STOP, "GENERATOR_FACT_COUNT_MISMATCH")

    validation_facts = [
        dict(row)
        for row in facts[VALIDATION_PAIR_START:VALIDATION_PAIR_STOP]
    ]
    require(len(validation_facts) == EXPECTED_PAIR_COUNT, "VALIDATION_FACT_COUNT")

    regenerated = materializer.materialize_facts(validation_facts)
    materializer.validate_materialized_rows(regenerated)
    regenerated_bytes = materializer.serialize_materialized_rows(
        regenerated
    ).encode("utf-8")

    require(
        regenerated_bytes == rows_raw,
        "FROZEN_RENDER_OR_MATERIALIZATION_BYTE_IDENTITY_MISMATCH",
    )

    return validation_facts, regenerated, materializer


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
    eos_candidates = [value for value in eos_candidates if value is not None]
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
    pad_candidates = [value for value in pad_candidates if value is not None]
    require(
        len(set(pad_candidates)) <= 1,
        "PAD_TOKEN_DECLARATIONS_DISAGREE",
    )
    pad_id = (
        tokenizer.token_to_id(pad_candidates[0])
        if pad_candidates
        else eos_id
    )
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
    title = str(values["title"])
    name = str(values["name"])
    role = str(values["role"])
    predicate = str(values["predicate"])
    obj = str(values["object"])
    location = str(values["location"])
    time = str(values["time"])

    title_start = 0
    title_end = len(title)

    name_start = title_end + 1
    name_end = name_start + len(name)

    role_start = name_end + len(", the ")
    role_end = role_start + len(role)

    predicate_start = role_end + len(", ")
    predicate_end = predicate_start + len(predicate)

    rendered = (
        f"{title} {name}, the {role}, {predicate} "
        f"{obj} in {location} during {time}."
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
    first_start = int(offsets[first][0])
    last_end = int(offsets[last][1])

    if first_start > span_start or last_end < span_end:
        return None, "SPAN_MAPPING_INCOMPLETE"

    return last, None


def _encoding(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = [int(value) for value in encoded.ids]
    offsets = [(int(a), int(b)) for a, b in encoded.offsets]
    require(len(ids) == len(offsets), "TOKEN_ID_OFFSET_LENGTH_MISMATCH")
    require(bool(ids), "EMPTY_TOKENIZATION")
    return ids, offsets


def analyze_required_anchors_for_row(
    row: Mapping[str, Any],
    fact: Mapping[str, Any],
    materializer: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    cell_id = str(row["contrast_cell_id"])
    require(cell_id in ANCHOR_PLAN, f"UNKNOWN_CELL:{cell_id}")

    overrides = materializer._overrides_for_cell(fact, cell_id)
    rendered, spans = realized_statement_and_spans(fact, overrides)
    require(rendered == row["evidence"], f"RENDER_IDENTITY_MISMATCH:{row['row_id']}")

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


def serialize_anchor_manifest(rows: Sequence[Mapping[str, Any]]) -> bytes:
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
    prospective_rows, prospective_raw, _manifest = load_frozen_prospective_rows(root)
    validation_facts, regenerated_rows, materializer = (
        reconstruct_validation_facts_and_verify_bytes(
            prospective_raw,
            root=root,
        )
    )
    tokenizer, tokenizer_provenance = load_canonical_analysis_tokenizer(
        tokenizer_snapshot
    )

    facts_by_id = {
        str(fact["pair_id"]): fact
        for fact in validation_facts
    }
    require(len(facts_by_id) == EXPECTED_PAIR_COUNT, "FACT_ID_COUNT_MISMATCH")

    anchor_rows: list[dict[str, Any]] = []
    for row in regenerated_rows:
        pair_id = str(row["source_pair_id"])
        require(pair_id in facts_by_id, f"MISSING_FACT_FOR_PAIR:{pair_id}")
        anchor_rows.extend(
            analyze_required_anchors_for_row(
                row,
                facts_by_id[pair_id],
                materializer,
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

    pair_ok: dict[str, bool] = {
        pair_id: True
        for pair_id in facts_by_id
    }
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
        else "BLOCKED_PENDING_NEW_AUTHORITY"
    )

    claim_truncation_count = sum(
        1
        for row in regenerated_rows
        if len(_encoding(tokenizer, str(row["claim"]))[0]) > CLAIM_BUDGET
    )
    evidence_truncation_count = sum(
        1
        for row in regenerated_rows
        if len(_encoding(tokenizer, str(row["evidence"]))[0]) > EVIDENCE_BUDGET
    )

    manifest_bytes = serialize_anchor_manifest(anchor_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "phase": "PROSPECTIVE_TOKENIZER_ANCHOR_ELIGIBILITY",
        "scientific_outcomes_observed": False,
        "model_forward_count": 0,
        "checkpoint_load_count": 0,
        "training_executed": False,
        "evaluation_executed": False,
        "gpu_used": False,
        "provenance": provenance,
        "frozen_input": {
            "path": PROSPECTIVE_DATA_PATH.as_posix(),
            "rows_sha256": EXPECTED_PROSPECTIVE_ROWS_SHA256,
            "row_count": EXPECTED_ROW_COUNT,
            "source_pair_count": EXPECTED_PAIR_COUNT,
            "validation_pair_ids_sha256":
                EXPECTED_VALIDATION_PAIR_IDS_SHA256,
            "render_and_materialization_byte_identity": "PASS",
        },
        "tokenizer": tokenizer_provenance,
        "coordinate_contract": {
            "active_serialization": "claim[:63]+EOS(0)+evidence[:64]",
            "anchor_mapping":
                "generator_declared_span_final_overlapping_evidence_token_"
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
            "This artifact establishes tokenizer/event-anchor prefix eligibility only.",
            "The canonical tokenizer is a reproducible active-encoding analysis reference, not a claimed historical snapshot.",
            "No model forward, checkpoint load, native-state extraction, kinematic endpoint computation, or scientific evaluation is performed.",
            "Eligibility PASS does not itself authorize model execution.",
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

    observed_manifest_sha = sha256_file(output_jsonl)
    require(
        observed_manifest_sha == summary["anchor_manifest_sha256"],
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
            "Gen4 large-correction prospective holdout. No model load/forward."
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
