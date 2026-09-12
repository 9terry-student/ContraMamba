from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUTHORITY_COMMIT = "95709b318bb4b4454d57edf52266860ded9e9da3"
AUTHORITY_REL = (
    "reports/longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic_"
    "authority_spec_candidate.md"
)
RUNNER_REL = "scripts/longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic.py"
TEST_REL = "tests/test_longterm_k0_rvg_p2_item_level_degeneracy_static_diagnostic.py"

P1_RESULT_DIR = (
    r"C:\Users\Home1\Desktop\ContraMamba-K0-RVG-P1-Runs"
    r"\p1-r2-scientific-db8f72b-v1"
)
P0_ARCHIVE_REL = "reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1"

P1_ARTIFACT_SHA256 = {
    "item_metrics.jsonl": "7a8ac4cb347a1a64c2dd69653a4bc575641679667534a89f42d44e257f9550e9",
    "block_metrics.jsonl": "c00197ab3c93ee2bb0d3951dbb47b452a0f0aa2ba920219c28707ab544e50530",
    "endpoint_summary.json": "5138fb7456e8626093753e188a3e334d79c91e5d9bc2a465073e4ae43823ec0d",
    "recurrence_audit.json": "f97dd5a3d934eccc91fde8430ebfea8c67018b7ef2af6a940e5adf968c4cdd5f",
    "state_hash_audit.jsonl": "8b9d68a81f245b2dce46238dd915a9cf6eb0a8032556fb44b92a9c8cfce2a274",
    "execution_manifest.json": "20bd491dbebaa3a882e7fce02d1ed039fec036bec08d7bd01287f109e6d74b26",
}

P0_ARTIFACT_SHA256 = {
    "candidate_pool.jsonl": "743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246",
    "generated_source.jsonl": "8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf",
    "phase_pair_mapping.json": "c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda",
    "token_contracts.jsonl": "6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998",
    "provisioning_manifest.json": "feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52",
    "validation_report_candidate.md": "ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a",
}

EXPECTED_P1_RUNTIME_HEAD = "db8f72bc8121a06a5dc9120611fa56b36eaec26f"
EXPECTED_P1_IMPLEMENTATION_COMMIT = "50a1daa781e47d1c0f1ba158beb445878e049a65"
EXPECTED_P1_AUTHORITY_COMMIT = "db8f72bc8121a06a5dc9120611fa56b36eaec26f"
EXPECTED_P1_VERDICT = "RAW_NATIVE_VECTOR_ORGANIZATION_NOT_ESTABLISHED"

ITEM_COUNT = 336
BLOCK_COUNT = 168
STATE_HASH_ROW_COUNT = 12096
EVENT_RELATIVE_COORDS = tuple(range(-1, 8))
BRANCH_PAIRS = (
    ("matched_corr", "swapped_corr", "corr"),
    ("matched_ctrl", "swapped_ctrl", "ctrl"),
)
HASH_FIELDS = ("S_prev_sha256", "G_sha256", "W_sha256", "S_post_sha256")
DIAGNOSTIC_FIELDS = (
    "response_norm_mean",
    "carry_response_norm_mean",
    "write_response_norm_mean",
    "write_total_cosine_mean",
    "carry_total_cosine_mean",
    "write_carry_cosine_mean",
)

TURN_COMPONENTWISE = "TURNING_COMPONENTWISE_IDENTITY"
TURN_CANCELLATION = "TURNING_EQUAL_BY_OFFSET_CANCELLATION"
TURN_DIFFERENT = "TURNING_DIFFERENT"
TURN_INVALID = "TURNING_INVALID"

COH_IDENTITY = "COHERENCE_EXACT_IDENTITY"
COH_DIFFERENT = "COHERENCE_DIFFERENT"
COH_INVALID = "COHERENCE_INVALID"

STATE_IDENTITY = "ENDPOINT_SUPPORTING_STATE_HASH_IDENTITY"
STATE_DIFFERENCE = "ENDPOINT_SUPPORTING_STATE_HASH_DIFFERENCE"

LOC_STATE_IDENTITY = "STATE_IDENTITY_EXPLAINS_ENDPOINT_IDENTITY"
LOC_FUNCTIONAL_COLLAPSE = "DISTINCT_STATE_ENDPOINT_FUNCTIONAL_COLLAPSE"
LOC_TURN_CANCELLATION = "DISTINCT_COMPONENTS_EQUAL_BY_TURNING_CANCELLATION"

TEXT_UNAVAILABLE = "TEXT_RECONSTRUCTION_NOT_AVAILABLE_FROM_AUTHORIZED_STATIC_INPUTS"
OUTPUT_SCHEMA = "k0-rvg-p2-item-level-degeneracy-static-diagnostic-v1"
OUTPUT_NAME = "p2_degeneracy_diagnostic.json"


class ContractError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any, *, final_lf: bool = False) -> bytes:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def parse_canonical_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ContractError(f"JSON_PARSE_FAILURE:{path.name}") from exc
    require(
        raw in {canonical_json_bytes(value), canonical_json_bytes(value, final_lf=True)},
        f"NONCANONICAL_JSON:{path.name}",
    )
    return value


def parse_canonical_jsonl(path: Path) -> list[Any]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"JSONL_FINAL_LF_REQUIRED:{path.name}")
    lines = raw.splitlines()
    values: list[Any] = []
    for index, line in enumerate(lines):
        require(bool(line), f"JSONL_EMPTY_LINE:{path.name}:{index}")
        try:
            value = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise ContractError(f"JSONL_PARSE_FAILURE:{path.name}:{index}") from exc
        require(
            line == canonical_json_bytes(value),
            f"NONCANONICAL_JSONL:{path.name}:{index}",
        )
        values.append(value)
    return values


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def authority_markers(text: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if " = " not in stripped:
            continue
        key, value = stripped.split(" = ", 1)
        markers[key.strip()] = value.strip()
    return markers


def authenticate_real_execution_authority(root: Path, authority_path: str) -> dict[str, str]:
    rel = Path(authority_path)
    require(not rel.is_absolute(), "P2_EXECUTION_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "P2_EXECUTION_AUTHORITY_PATH_TRAVERSAL")
    normalized = rel.as_posix()
    require(normalized.startswith("reports/"), "P2_EXECUTION_AUTHORITY_NOT_REPORT")

    try:
        _git(root, "ls-files", "--error-unmatch", "--", normalized)
    except subprocess.CalledProcessError as exc:
        raise ContractError("P2_EXECUTION_AUTHORITY_NOT_TRACKED") from exc

    worktree = root / normalized
    require(worktree.is_file(), "P2_EXECUTION_AUTHORITY_MISSING")
    head_bytes = subprocess.check_output(
        ["git", "show", f"HEAD:{normalized}"],
        cwd=root,
    )
    require(worktree.read_bytes() == head_bytes, "P2_EXECUTION_AUTHORITY_WORKTREE_DRIFT")

    markers = authority_markers(worktree.read_text(encoding="utf-8"))
    require(
        markers.get("P2_REAL_ARTIFACT_DIAGNOSTIC_EXECUTION_AUTHORIZED") == "YES",
        "P2_REAL_ARTIFACT_DIAGNOSTIC_EXECUTION_NOT_AUTHORIZED",
    )
    require(
        markers.get("SCIENTIFIC_MODEL_FORWARD_AUTHORIZED") == "NO",
        "P2_EXECUTION_AUTHORITY_MODEL_FORWARD_MARKER_MISMATCH",
    )
    require(
        markers.get("SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED") == "NO",
        "P2_EXECUTION_AUTHORITY_STATE_READ_MARKER_MISMATCH",
    )
    impl_commit = markers.get("P2_IMPLEMENTATION_COMMIT", "")
    require(len(impl_commit) == 40, "P2_EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_INVALID")
    actual_impl_commit = _git(root, "log", "-1", "--format=%H", "--", RUNNER_REL)
    require(
        impl_commit == actual_impl_commit,
        "P2_EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_MISMATCH",
    )
    require(
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", impl_commit, "HEAD"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0,
        "P2_IMPLEMENTATION_NOT_ANCESTOR",
    )
    return markers


def classify_turning(item: Mapping[str, Any]) -> str:
    if not bool(item.get("turning_valid")):
        return TURN_INVALID
    matched = item.get("matched")
    swapped = item.get("swapped")
    require(isinstance(matched, Mapping), "TURNING_MATCHED_MAPPING_REQUIRED")
    require(isinstance(swapped, Mapping), "TURNING_SWAPPED_MAPPING_REQUIRED")

    tm = matched.get("T_M")
    ts = swapped.get("T_S")
    amc = matched.get("A_corr")
    amk = matched.get("A_ctrl")
    asc = swapped.get("A_corr")
    ask = swapped.get("A_ctrl")
    if any(v is None for v in (tm, ts, amc, amk, asc, ask)):
        return TURN_INVALID
    if tm != ts:
        return TURN_DIFFERENT
    if amc == asc and amk == ask:
        return TURN_COMPONENTWISE
    return TURN_CANCELLATION


def classify_coherence(item: Mapping[str, Any]) -> str:
    if not bool(item.get("coherence_valid")):
        return COH_INVALID
    matched = item.get("matched")
    swapped = item.get("swapped")
    require(isinstance(matched, Mapping), "COHERENCE_MATCHED_MAPPING_REQUIRED")
    require(isinstance(swapped, Mapping), "COHERENCE_SWAPPED_MAPPING_REQUIRED")
    cm = matched.get("C_M")
    cs = swapped.get("C_S")
    if cm is None or cs is None:
        return COH_INVALID
    return COH_IDENTITY if cm == cs else COH_DIFFERENT


def _diagnostic_relation(a: Any, b: Any) -> str:
    if a is None and b is None:
        return "NULL_NULL_EQUAL"
    if a == b:
        return "EXACT_EQUAL"
    return "DIFFERENT"


def compare_diagnostic_summaries(item: Mapping[str, Any]) -> dict[str, str]:
    matched = item.get("matched")
    swapped = item.get("swapped")
    require(isinstance(matched, Mapping), "DIAG_MATCHED_MAPPING_REQUIRED")
    require(isinstance(swapped, Mapping), "DIAG_SWAPPED_MAPPING_REQUIRED")
    md = matched.get("diagnostics")
    sd = swapped.get("diagnostics")
    require(isinstance(md, Mapping), "DIAG_MATCHED_DIAGNOSTICS_REQUIRED")
    require(isinstance(sd, Mapping), "DIAG_SWAPPED_DIAGNOSTICS_REQUIRED")
    return {field: _diagnostic_relation(md.get(field), sd.get(field)) for field in DIAGNOSTIC_FIELDS}


def index_state_hash_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    index: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    allowed_roles = {p[0] for p in BRANCH_PAIRS} | {p[1] for p in BRANCH_PAIRS}
    for row in rows:
        local = int(row["local_template_index"])
        role = str(row["branch_role"])
        token = int(row["token_index"])
        require(role in allowed_roles, f"UNEXPECTED_BRANCH_ROLE:{role}")
        key = (local, role, token)
        require(key not in index, f"DUPLICATE_STATE_HASH_COORDINATE:{local}:{role}:{token}")
        for field in HASH_FIELDS:
            value = row.get(field)
            require(
                isinstance(value, str) and len(value) == 64,
                f"STATE_HASH_FIELD_INVALID:{field}",
            )
        index[key] = row
    return index


def compare_item_state_hashes(
    local_template_index: int,
    matched_te: int,
    swapped_te: int,
    state_index: Mapping[tuple[int, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    equal_count = 0
    different_count = 0
    first_difference: dict[str, Any] | None = None

    for matched_role, swapped_role, role_label in BRANCH_PAIRS:
        for relative in EVENT_RELATIVE_COORDS:
            m_key = (local_template_index, matched_role, matched_te + relative)
            s_key = (local_template_index, swapped_role, swapped_te + relative)
            require(m_key in state_index, f"MISSING_STATE_HASH_COORDINATE:{m_key}")
            require(s_key in state_index, f"MISSING_STATE_HASH_COORDINATE:{s_key}")
            mrow = state_index[m_key]
            srow = state_index[s_key]
            for field in HASH_FIELDS:
                if mrow[field] == srow[field]:
                    equal_count += 1
                else:
                    different_count += 1
                    if first_difference is None:
                        first_difference = {
                            "branch_role": role_label,
                            "relative_coordinate": relative,
                            "tensor_field": field,
                            "matched_token_index": matched_te + relative,
                            "swapped_token_index": swapped_te + relative,
                        }

    total = len(BRANCH_PAIRS) * len(EVENT_RELATIVE_COORDS) * len(HASH_FIELDS)
    require(equal_count + different_count == total, "STATE_HASH_COMPARISON_COUNT_MISMATCH")
    classification = STATE_IDENTITY if different_count == 0 else STATE_DIFFERENCE
    return {
        "classification": classification,
        "total_aligned_hash_comparisons": total,
        "exact_equal_count": equal_count,
        "different_count": different_count,
        "all_endpoint_supporting_snapshots_identical": different_count == 0,
        "first_difference": first_difference,
    }


def index_by_local(rows: Sequence[Mapping[str, Any]], expected_count: int = ITEM_COUNT) -> dict[int, Mapping[str, Any]]:
    by_local: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        local = int(row["local_template_index"])
        require(local not in by_local, f"DUPLICATE_LOCAL_TEMPLATE_INDEX:{local}")
        by_local[local] = row
    require(
        sorted(by_local) == list(range(expected_count)),
        "LOCAL_TEMPLATE_INDEX_SET_MISMATCH",
    )
    return by_local


def _phase_mate_candidate(
    candidate: Mapping[str, Any],
    candidates_by_stable: Mapping[str, Mapping[str, Any]],
    token_contract: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    mate_stable = token_contract.get("phase_mate_stable_id")
    if not isinstance(mate_stable, str):
        return None
    return candidates_by_stable.get(mate_stable)


def compare_construction_metadata(
    candidate: Mapping[str, Any] | None,
    mate: Mapping[str, Any] | None,
    token_contract: Mapping[str, Any],
) -> dict[str, Any]:
    matched_anchor = token_contract.get("matched_divergence_anchor")
    swapped_anchor = token_contract.get("swapped_divergence_anchor")
    matched_corr_count = token_contract.get("matched_correction_token_count")
    swapped_corr_count = token_contract.get("swapped_correction_token_count")
    matched_ctrl_count = token_contract.get("matched_control_token_count")
    swapped_ctrl_count = token_contract.get("swapped_control_token_count")
    matched_w8 = token_contract.get("matched_w8_available")
    swapped_w8 = token_contract.get("swapped_w8_available")

    result: dict[str, Any] = {
        "divergence_anchor_exact_equal": matched_anchor == swapped_anchor,
        "correction_token_count_exact_equal": matched_corr_count == swapped_corr_count,
        "control_token_count_exact_equal": matched_ctrl_count == swapped_ctrl_count,
        "w8_availability_exact_equal": matched_w8 == swapped_w8,
    }

    if candidate is None or mate is None:
        result["correction_continuation_text_relation"] = TEXT_UNAVAILABLE
        result["control_continuation_text_relation"] = TEXT_UNAVAILABLE
    else:
        corr_a = candidate.get("correction_text")
        corr_b = mate.get("correction_text")
        ctrl_a = candidate.get("control_text")
        ctrl_b = mate.get("control_text")
        if all(isinstance(v, str) for v in (corr_a, corr_b, ctrl_a, ctrl_b)):
            result["correction_continuation_text_relation"] = (
                "EXACT_EQUAL" if corr_a == corr_b else "DIFFERENT"
            )
            result["control_continuation_text_relation"] = (
                "EXACT_EQUAL" if ctrl_a == ctrl_b else "DIFFERENT"
            )
        else:
            result["correction_continuation_text_relation"] = TEXT_UNAVAILABLE
            result["control_continuation_text_relation"] = TEXT_UNAVAILABLE
    return result


def localize_zero_endpoint(
    endpoint: str,
    item: Mapping[str, Any],
    turning_class: str,
    coherence_class: str,
    state_class: str,
) -> str | None:
    require(endpoint in {"turning", "response_coherence"}, "UNKNOWN_ENDPOINT")
    value = item.get("X_turn") if endpoint == "turning" else item.get("X_coh")
    if value != 0.0:
        return None
    if state_class == STATE_IDENTITY:
        return LOC_STATE_IDENTITY
    require(state_class == STATE_DIFFERENCE, "UNKNOWN_STATE_CLASSIFICATION")
    if endpoint == "turning" and turning_class == TURN_CANCELLATION:
        return LOC_TURN_CANCELLATION
    if endpoint == "turning":
        require(
            turning_class == TURN_COMPONENTWISE,
            "ZERO_TURNING_UNEXPECTED_CLASSIFICATION",
        )
    else:
        require(
            coherence_class == COH_IDENTITY,
            "ZERO_COHERENCE_UNEXPECTED_CLASSIFICATION",
        )
    return LOC_FUNCTIONAL_COLLAPSE


def _counter_summary(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_static_diagnostic(
    item_rows: Sequence[Mapping[str, Any]],
    state_hash_rows: Sequence[Mapping[str, Any]],
    token_contract_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    items = index_by_local(item_rows)
    contracts = index_by_local(token_contract_rows)
    state_index = index_state_hash_rows(state_hash_rows)

    candidates_by_local: dict[int, Mapping[str, Any]] = {}
    candidates_by_stable: dict[str, Mapping[str, Any]] = {}
    if candidate_rows is not None:
        candidates_by_local = index_by_local(candidate_rows)
        for row in candidate_rows:
            stable = row.get("stable_item_id")
            require(isinstance(stable, str), "CANDIDATE_STABLE_ID_REQUIRED")
            require(stable not in candidates_by_stable, "DUPLICATE_CANDIDATE_STABLE_ID")
            candidates_by_stable[stable] = row

    out_items: list[dict[str, Any]] = []
    diag_field_relations: dict[str, list[str]] = {field: [] for field in DIAGNOSTIC_FIELDS}
    construction_relations: dict[str, list[str]] = {}
    turn_classes: list[str] = []
    coh_classes: list[str] = []
    state_classes: list[str] = []

    for local in range(ITEM_COUNT):
        item = items[local]
        contract = contracts[local]
        matched_te = int(item["matched_divergence_anchor"])
        swapped_te = int(item["swapped_divergence_anchor"])
        require(
            matched_te == int(contract["matched_divergence_anchor"]),
            "MATCHED_DIVERGENCE_ANCHOR_MISMATCH",
        )
        require(
            swapped_te == int(contract["swapped_divergence_anchor"]),
            "SWAPPED_DIVERGENCE_ANCHOR_MISMATCH",
        )

        turn_class = classify_turning(item)
        coh_class = classify_coherence(item)
        state_cmp = compare_item_state_hashes(local, matched_te, swapped_te, state_index)
        state_class = str(state_cmp["classification"])
        diag_cmp = compare_diagnostic_summaries(item)

        candidate = candidates_by_local.get(local) if candidates_by_local else None
        mate = (
            _phase_mate_candidate(candidate, candidates_by_stable, contract)
            if candidate is not None
            else None
        )
        construction = compare_construction_metadata(candidate, mate, contract)

        for field, relation in diag_cmp.items():
            diag_field_relations[field].append(relation)
        for field, value in construction.items():
            relation = str(value) if not isinstance(value, bool) else ("TRUE" if value else "FALSE")
            construction_relations.setdefault(field, []).append(relation)

        turn_classes.append(turn_class)
        coh_classes.append(coh_class)
        state_classes.append(state_class)

        out_items.append(
            {
                "local_template_index": local,
                "stable_item_id": item["stable_item_id"],
                "turning_classification": turn_class,
                "coherence_classification": coh_class,
                "state_hash_comparison": state_cmp,
                "diagnostic_summary_relations": diag_cmp,
                "construction_comparison": construction,
                "turning_localization": localize_zero_endpoint(
                    "turning", item, turn_class, coh_class, state_class
                ),
                "response_coherence_localization": localize_zero_endpoint(
                    "response_coherence", item, turn_class, coh_class, state_class
                ),
                "X_turn": item.get("X_turn"),
                "X_coh": item.get("X_coh"),
            }
        )

    turn_zero = sum(row.get("X_turn") == 0.0 for row in item_rows)
    coh_zero = sum(row.get("X_coh") == 0.0 for row in item_rows)

    return {
        "schema_version": OUTPUT_SCHEMA,
        "item_count": len(out_items),
        "turning_exact_zero_count": turn_zero,
        "coherence_exact_zero_count": coh_zero,
        "turning_classification_counts": _counter_summary(turn_classes),
        "coherence_classification_counts": _counter_summary(coh_classes),
        "state_hash_classification_counts": _counter_summary(state_classes),
        "diagnostic_summary_relation_counts": {
            field: _counter_summary(relations)
            for field, relations in diag_field_relations.items()
        },
        "construction_relation_counts": {
            field: _counter_summary(relations)
            for field, relations in sorted(construction_relations.items())
        },
        "items": out_items,
    }


def validate_exact_artifact_set(directory: Path, expected: Mapping[str, str]) -> None:
    require(directory.is_dir(), f"ARTIFACT_DIRECTORY_MISSING:{directory}")
    observed = {p.name for p in directory.iterdir() if p.is_file()}
    require(observed == set(expected), "ARTIFACT_SET_MISMATCH")
    for name, expected_sha in expected.items():
        require(
            file_sha256(directory / name) == expected_sha,
            f"ARTIFACT_SHA256_MISMATCH:{name}",
        )


def validate_real_inputs(repo_root: Path, p1_dir: Path) -> dict[str, Any]:
    validate_exact_artifact_set(p1_dir, P1_ARTIFACT_SHA256)

    p0_dir = repo_root / P0_ARCHIVE_REL
    require(p0_dir.is_dir(), "P0_ARCHIVE_MISSING")
    for name, expected_sha in P0_ARTIFACT_SHA256.items():
        path = p0_dir / name
        require(path.is_file(), f"P0_ARTIFACT_MISSING:{name}")
        require(file_sha256(path) == expected_sha, f"P0_ARTIFACT_SHA256_MISMATCH:{name}")

    item_rows = parse_canonical_jsonl(p1_dir / "item_metrics.jsonl")
    block_rows = parse_canonical_jsonl(p1_dir / "block_metrics.jsonl")
    state_rows = parse_canonical_jsonl(p1_dir / "state_hash_audit.jsonl")
    manifest = parse_canonical_json(p1_dir / "execution_manifest.json")
    endpoint = parse_canonical_json(p1_dir / "endpoint_summary.json")
    parse_canonical_json(p1_dir / "recurrence_audit.json")

    require(len(item_rows) == ITEM_COUNT, "P1_ITEM_COUNT_MISMATCH")
    require(len(block_rows) == BLOCK_COUNT, "P1_BLOCK_COUNT_MISMATCH")
    require(len(state_rows) == STATE_HASH_ROW_COUNT, "P1_STATE_HASH_ROW_COUNT_MISMATCH")
    require(
        manifest.get("runtime_git_head") == EXPECTED_P1_RUNTIME_HEAD,
        "P1_RUNTIME_HEAD_MISMATCH",
    )
    require(
        manifest.get("p1_implementation_commit") == EXPECTED_P1_IMPLEMENTATION_COMMIT,
        "P1_IMPLEMENTATION_COMMIT_MISMATCH",
    )
    require(
        manifest.get("scientific_execution_authority_commit") == EXPECTED_P1_AUTHORITY_COMMIT,
        "P1_SCIENTIFIC_AUTHORITY_COMMIT_MISMATCH",
    )
    require(endpoint.get("overall_verdict") == EXPECTED_P1_VERDICT, "P1_VERDICT_MISMATCH")
    require(sum(row.get("X_turn") == 0.0 for row in item_rows) == 335, "P1_TURN_ZERO_COUNT_MISMATCH")
    require(sum(row.get("X_coh") == 0.0 for row in item_rows) == 335, "P1_COH_ZERO_COUNT_MISMATCH")

    token_contracts = parse_canonical_jsonl(p0_dir / "token_contracts.jsonl")
    candidates = parse_canonical_jsonl(p0_dir / "candidate_pool.jsonl")
    require(len(token_contracts) == ITEM_COUNT, "P0_TOKEN_CONTRACT_COUNT_MISMATCH")
    require(len(candidates) == ITEM_COUNT, "P0_CANDIDATE_COUNT_MISMATCH")

    return {
        "item_rows": item_rows,
        "state_rows": state_rows,
        "token_contracts": token_contracts,
        "candidates": candidates,
        "manifest": manifest,
        "endpoint": endpoint,
    }


def write_output_atomic(output_dir: Path, diagnostic: Mapping[str, Any]) -> Path:
    require(not output_dir.exists(), "P2_OUTPUT_ALREADY_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent)
    )
    try:
        output_path = temp_dir / OUTPUT_NAME
        output_path.write_bytes(canonical_json_bytes(diagnostic, final_lf=True))
        observed = {p.name for p in temp_dir.iterdir() if p.is_file()}
        require(observed == {OUTPUT_NAME}, "P2_OUTPUT_ARTIFACT_SET_MISMATCH")
        temp_dir.rename(output_dir)
        return output_dir / OUTPUT_NAME
    except Exception:
        for child in temp_dir.iterdir():
            child.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise


def execute_real(
    repo_root: Path,
    p1_dir: Path,
    execution_authority_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    # Critical order: authority gate before any real P0/P1 artifact read.
    markers = authenticate_real_execution_authority(repo_root, execution_authority_path)
    inputs = validate_real_inputs(repo_root, p1_dir)
    diagnostic = build_static_diagnostic(
        inputs["item_rows"],
        inputs["state_rows"],
        inputs["token_contracts"],
        inputs["candidates"],
    )
    diagnostic["provenance"] = {
        "p2_execution_authority_path": execution_authority_path,
        "p2_implementation_commit": markers["P2_IMPLEMENTATION_COMMIT"],
        "p1_artifact_sha256": dict(P1_ARTIFACT_SHA256),
        "p0_artifact_sha256": dict(P0_ARTIFACT_SHA256),
        "p1_runtime_head": EXPECTED_P1_RUNTIME_HEAD,
        "p1_implementation_commit": EXPECTED_P1_IMPLEMENTATION_COMMIT,
        "p1_scientific_execution_authority_commit": EXPECTED_P1_AUTHORITY_COMMIT,
    }
    output_path = write_output_atomic(output_dir, diagnostic)
    return {
        "status": "COMPLETE_P2_STATIC_DEGENERACY_DIAGNOSTIC",
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "p1_scientific_rerun_executed": False,
    }


def synthetic_self_check() -> dict[str, Any]:
    base_item = {
        "turning_valid": True,
        "coherence_valid": True,
        "matched": {
            "A_corr": 0.2,
            "A_ctrl": 0.8,
            "T_M": 0.6,
            "C_M": 0.25,
            "diagnostics": {field: 1.0 for field in DIAGNOSTIC_FIELDS},
        },
        "swapped": {
            "A_corr": 0.2,
            "A_ctrl": 0.8,
            "T_S": 0.6,
            "C_S": 0.25,
            "diagnostics": {field: 1.0 for field in DIAGNOSTIC_FIELDS},
        },
        "X_turn": 0.0,
        "X_coh": 0.0,
    }
    require(classify_turning(base_item) == TURN_COMPONENTWISE, "SYNTH_TURN_COMPONENTWISE")
    require(classify_coherence(base_item) == COH_IDENTITY, "SYNTH_COH_IDENTITY")

    cancellation = json.loads(json.dumps(base_item))
    cancellation["matched"]["A_corr"] = 0.1
    cancellation["matched"]["A_ctrl"] = 0.7
    cancellation["matched"]["T_M"] = 0.6
    require(classify_turning(cancellation) == TURN_CANCELLATION, "SYNTH_TURN_CANCELLATION")

    different = json.loads(json.dumps(base_item))
    different["swapped"]["T_S"] = 0.5
    different["X_turn"] = 0.1
    require(classify_turning(different) == TURN_DIFFERENT, "SYNTH_TURN_DIFFERENT")

    return {
        "schema_version": "k0-rvg-p2-synthetic-self-check-v1",
        "status": "PASS_SYNTHETIC_P2_STATIC_DIAGNOSTIC_CORE",
        "real_p1_artifact_read": False,
        "model_forward": False,
        "recurrent_state_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("synthetic-self-check")

    real = sub.add_parser("execute-real")
    real.add_argument("--repo-root", type=Path, required=True)
    real.add_argument("--p1-dir", type=Path, required=True)
    real.add_argument("--execution-authority", required=True)
    real.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "synthetic-self-check":
        print(json.dumps(synthetic_self_check(), indent=2, sort_keys=True))
        return 0
    if args.command == "execute-real":
        result = execute_real(
            args.repo_root.resolve(),
            args.p1_dir.resolve(),
            args.execution_authority,
            args.output_dir.resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise ContractError("UNKNOWN_COMMAND")


if __name__ == "__main__":
    raise SystemExit(main())
