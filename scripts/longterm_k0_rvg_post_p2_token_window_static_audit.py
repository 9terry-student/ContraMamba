from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


IMPLEMENTATION_AUTHORITY_COMMIT = "e13458f32d1b08685f962bb5d8ba1351fd9a0928"
IMPLEMENTATION_AUTHORITY_REL = (
    "reports/longterm_k0_rvg_post_p2_token_window_static_audit_"
    "implementation_authority_spec_candidate.md"
)
RUNNER_REL = "scripts/longterm_k0_rvg_post_p2_token_window_static_audit.py"
TEST_REL = "tests/test_longterm_k0_rvg_post_p2_token_window_static_audit.py"

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"
HISTORICAL_K1_UNTRACKED = {
    "scripts/longterm_k1_native_state_kinematics.py",
    "tests/test_longterm_k1_native_state_kinematics.py",
}

P0_ARCHIVE_REL = "reports/longterm_k0_rvg_p0_state_blind_provisioning_421d798_v1"
P0_ARTIFACT_SHA256 = {
    "candidate_pool.jsonl": "743657411af4e143931e4d2c79f17043134bff3a370d30910588504c7f19f246",
    "generated_source.jsonl": "8137c0020a040faaf0c6be833b123e143dc5a0a09ae092e8e99530bb8671c1bf",
    "phase_pair_mapping.json": "c5e1fa2ac946d153821896d5153354fae6e17038a55be87b3d31ab43d2921eda",
    "token_contracts.jsonl": "6eb006f7deca28affa73318887421879e1279fd6897fab857b460873add9a998",
    "provisioning_manifest.json": "feab9c60e3546ace3258f068e38d5bb577fc63b803b95349379fa8b4db425e52",
    "validation_report_candidate.md": "ae2fe4d1db13e1765415eab9c95263aec618fccf9299fe93fad0c8e31141b73a",
}

P2_ARTIFACT_SHA256 = "058d00adb99cdbfad1893593a3e8917322c5ca2f186f7305561460f868ddddc3"
P2_SCHEMA = "k0-rvg-p2-item-level-degeneracy-static-diagnostic-v1"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"

ITEM_COUNT = 336
BLOCK_COUNT = 168
WINDOW_COORDS = tuple(range(-1, 8))
OUTPUT_NAME = "post_p2_token_window_audit.json"
OUTPUT_SCHEMA = "k0-rvg-post-p2-token-window-static-audit-v1"

STATE_IDENTITY = "ENDPOINT_SUPPORTING_STATE_HASH_IDENTITY"
STATE_DIFFERENCE = "ENDPOINT_SUPPORTING_STATE_HASH_DIFFERENCE"

C_IDENTICAL = "TOKEN_SEQUENCES_EXACTLY_IDENTICAL_FULL_BRANCH"
C_PREWINDOW = "TOKEN_SEQUENCES_DIFFER_BEFORE_ENDPOINT_WINDOW"
C_K_MINUS_1 = "TOKEN_SEQUENCES_FIRST_DIFFER_AT_WINDOW_K_MINUS_1"
C_INWINDOW = "TOKEN_SEQUENCES_FIRST_DIFFER_WITHIN_WINDOW_K_0_TO_PLUS_7"
C_AFTER = "TOKEN_SEQUENCES_FIRST_DIFFER_AFTER_ENDPOINT_WINDOW"
C_UNKNOWN = "TOKEN_RELATION_NOT_IDENTIFIABLE_FROM_AUTHORIZED_EVIDENCE"

X_WINDOW_IDENTITY = "TOKEN_WINDOW_IDENTITY_SUPPORTS_STATE_IDENTITY"
X_PREWINDOW = "PREWINDOW_TOKEN_DIFFERENCE_PRECEDES_STATE_IDENTITY"
X_INWINDOW = "INWINDOW_TOKEN_DIFFERENCE_WITH_STATE_IDENTITY"
X_UNRESOLVED = "TOKEN_BOUNDARY_UNRESOLVED"

REQUIRED_FUTURE_MARKERS = {
    "POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED": "YES",
    "TOKENIZER_REEXECUTION_AUTHORIZED": "YES",
    "MODEL_CONSTRUCTION_AUTHORIZED": "NO",
    "CHECKPOINT_LOADING_AUTHORIZED": "NO",
    "SCIENTIFIC_MODEL_FORWARD_AUTHORIZED": "NO",
    "SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED": "NO",
    "NETWORK_ACCESS_AUTHORIZED": "NO",
    "TOKENIZER_MODEL_ID": HF_MODEL,
    "TOKENIZER_REVISION": HF_REVISION,
    "TOKENIZER_TRANSFORMERS_VERSION": TRANSFORMERS_VERSION,
}


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
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row), final_lf=True) for row in rows)


def parse_canonical_json(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"JSON_PARSE_FAILURE:{path.name}") from exc
    require(
        raw in {canonical_json_bytes(value), canonical_json_bytes(value, final_lf=True)},
        f"NONCANONICAL_JSON:{path.name}",
    )
    return value


def parse_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"JSONL_FINAL_LF_REQUIRED:{path.name}")
    require(b"\r" not in raw, f"JSONL_CR_FORBIDDEN:{path.name}")
    require(not raw.startswith(b"\xef\xbb\xbf"), f"JSONL_BOM_FORBIDDEN:{path.name}")
    values: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw[:-1].split(b"\n"), start=1):
        require(bool(line), f"JSONL_EMPTY_LINE:{path.name}:{lineno}")
        try:
            value = json.loads(line.decode("utf-8", "strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"JSONL_PARSE_FAILURE:{path.name}:{lineno}") from exc
        require(isinstance(value, dict), f"JSONL_OBJECT_REQUIRED:{path.name}:{lineno}")
        require(line == canonical_json_bytes(value), f"NONCANONICAL_JSONL:{path.name}:{lineno}")
        values.append(value)
    return values


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"GIT_FAILURE:{' '.join(args)}") from exc


def parse_authority_markers(text: str) -> dict[str, str]:
    markers: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().strip("`")
        if " = " not in stripped:
            continue
        key, value = stripped.split(" = ", 1)
        key = key.strip()
        value = value.strip()
        require(key not in markers, f"DUPLICATE_AUTHORITY_MARKER:{key}")
        markers[key] = value
    return markers


def validate_execution_markers(markers: Mapping[str, str], actual_impl_commit: str) -> None:
    for key, expected in REQUIRED_FUTURE_MARKERS.items():
        require(markers.get(key) == expected, f"EXECUTION_AUTHORITY_MARKER_MISMATCH:{key}")
    declared = markers.get("POST_P2_TOKEN_WINDOW_IMPLEMENTATION_COMMIT", "")
    require(len(declared) == 40, "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_INVALID")
    require(all(ch in "0123456789abcdef" for ch in declared), "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_NONHEX")
    require(declared == actual_impl_commit, "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_MISMATCH")


def authenticate_execution_authority(root: Path, authority_rel: str) -> dict[str, str]:
    rel = Path(authority_rel)
    require(not rel.is_absolute(), "EXECUTION_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "EXECUTION_AUTHORITY_PATH_TRAVERSAL")
    normalized = rel.as_posix()
    require(normalized.startswith("reports/"), "EXECUTION_AUTHORITY_NOT_REPORT")

    branch = _git(root, "branch", "--show-current")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    try:
        _git(root, "ls-files", "--error-unmatch", "--", normalized)
    except ContractError as exc:
        raise ContractError("EXECUTION_AUTHORITY_NOT_TRACKED") from exc

    worktree = root / normalized
    require(worktree.is_file(), "EXECUTION_AUTHORITY_MISSING")
    head_bytes = subprocess.check_output(["git", "show", f"HEAD:{normalized}"], cwd=root)
    require(worktree.read_bytes() == head_bytes, "EXECUTION_AUTHORITY_WORKTREE_DRIFT")

    markers = parse_authority_markers(worktree.read_text(encoding="utf-8"))
    actual_impl_commit = _git(root, "log", "-1", "--format=%H", "--", RUNNER_REL)
    validate_execution_markers(markers, actual_impl_commit)

    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", actual_impl_commit, "HEAD"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    return markers


def build_phase_mate_map(mapping_obj: Mapping[str, Any], *, expected_count: int = ITEM_COUNT) -> dict[int, int]:
    blocks = mapping_obj.get("blocks")
    require(isinstance(blocks, list), "PHASE_MAPPING_BLOCKS_REQUIRED")
    mate: dict[int, int] = {}
    for block in blocks:
        require(isinstance(block, Mapping), "PHASE_MAPPING_BLOCK_NOT_OBJECT")
        a = int(block["item_a_local_index"])
        b = int(block["item_b_local_index"])
        require(a != b, "PHASE_MAPPING_SELF_PAIR")
        require(a not in mate and b not in mate, "PHASE_MAPPING_DUPLICATE_ITEM")
        mate[a] = b
        mate[b] = a
    require(len(mate) == expected_count, "PHASE_MAPPING_ITEM_COUNT_MISMATCH")
    require(sorted(mate) == list(range(expected_count)), "PHASE_MAPPING_INDEX_DOMAIN_MISMATCH")
    for i, m in mate.items():
        require(mate.get(m) == i, "PHASE_MAPPING_NOT_RECIPROCAL")
    return mate


def index_by_local(rows: Sequence[Mapping[str, Any]], *, expected_count: int = ITEM_COUNT) -> dict[int, Mapping[str, Any]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        local = int(row["local_template_index"])
        require(local not in indexed, f"DUPLICATE_LOCAL_INDEX:{local}")
        indexed[local] = row
    require(len(indexed) == expected_count, "LOCAL_INDEX_COUNT_MISMATCH")
    require(sorted(indexed) == list(range(expected_count)), "LOCAL_INDEX_DOMAIN_MISMATCH")
    return indexed


def reconstruct_branch_texts(
    candidate_by_local: Mapping[int, Mapping[str, Any]],
    mate_map: Mapping[int, int],
    local_index: int,
) -> dict[str, str]:
    require(local_index in candidate_by_local, "CANDIDATE_LOCAL_INDEX_MISSING")
    require(local_index in mate_map, "PHASE_MATE_LOCAL_INDEX_MISSING")
    item = candidate_by_local[local_index]
    mate = candidate_by_local[mate_map[local_index]]
    prefix = str(item["prefix_text"])
    return {
        "prefix": prefix,
        "matched_corr": prefix + str(item["correction_text"]),
        "matched_ctrl": prefix + str(item["control_text"]),
        "swapped_corr": prefix + str(mate["correction_text"]),
        "swapped_ctrl": prefix + str(mate["control_text"]),
    }


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    require(isinstance(ids, list), "TOKEN_IDS_NOT_LIST")
    require(all(type(value) is int for value in ids), "TOKEN_IDS_NOT_INT")
    return ids


def token_id_sha256(ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json_bytes(list(ids)))


def first_corr_ctrl_divergence(
    corr: Sequence[int],
    ctrl: Sequence[int],
    prefix_len: int,
) -> int:
    upper = min(len(corr), len(ctrl), prefix_len + 8)
    for index in range(prefix_len, upper):
        if corr[index] != ctrl[index]:
            return index
    raise ContractError("CORR_CTRL_DIVERGENCE_NOT_WITHIN_FIRST_8_CONTINUATION_TOKENS")


def first_difference_index(a: Sequence[int], b: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def compare_window(
    matched: Sequence[int],
    swapped: Sequence[int],
    matched_te: int,
    swapped_te: int,
) -> list[dict[str, Any]]:
    require(matched_te == swapped_te, "MATCHED_SWAPPED_EVENT_ANCHOR_MISMATCH")
    out: list[dict[str, Any]] = []
    for relative in WINDOW_COORDS:
        mi = matched_te + relative
        si = swapped_te + relative
        require(mi >= 0 and si >= 0, f"WINDOW_COORDINATE_NEGATIVE:{relative}")
        require(mi < len(matched), f"MATCHED_WINDOW_COORDINATE_MISSING:{relative}")
        require(si < len(swapped), f"SWAPPED_WINDOW_COORDINATE_MISSING:{relative}")
        m_id = int(matched[mi])
        s_id = int(swapped[si])
        out.append(
            {
                "relative_coordinate": relative,
                "matched_absolute_index": mi,
                "swapped_absolute_index": si,
                "matched_token_id": m_id,
                "swapped_token_id": s_id,
                "exact_equal": m_id == s_id,
            }
        )
    return out


def classify_first_difference(first_difference: int | None, common_te: int) -> str:
    if first_difference is None:
        return C_IDENTICAL
    relative = first_difference - common_te
    if relative < -1:
        return C_PREWINDOW
    if relative == -1:
        return C_K_MINUS_1
    if 0 <= relative <= 7:
        return C_INWINDOW
    return C_AFTER


def analyze_role(
    matched: Sequence[int],
    swapped: Sequence[int],
    matched_te: int,
    swapped_te: int,
) -> dict[str, Any]:
    require(matched_te == swapped_te, "MATCHED_SWAPPED_EVENT_ANCHOR_MISMATCH")
    first = first_difference_index(matched, swapped)
    window = compare_window(matched, swapped, matched_te, swapped_te)
    common_te = matched_te
    return {
        "matched_token_count": len(matched),
        "swapped_token_count": len(swapped),
        "full_sequence_exact_equal": first is None,
        "first_difference_absolute_index": first,
        "first_difference_relative_to_te": None if first is None else first - common_te,
        "difference_before_window": first is not None and first < common_te - 1,
        "window_exact_equal": all(row["exact_equal"] for row in window),
        "window": window,
        "classification": classify_first_difference(first, common_te),
    }


def cross_level_classification(
    state_hash_classification: str,
    corr_role: Mapping[str, Any],
    ctrl_role: Mapping[str, Any],
) -> str:
    if state_hash_classification != STATE_IDENTITY:
        return X_UNRESOLVED
    classes = {str(corr_role.get("classification")), str(ctrl_role.get("classification"))}
    if C_UNKNOWN in classes:
        return X_UNRESOLVED
    if classes & {C_K_MINUS_1, C_INWINDOW}:
        return X_INWINDOW
    if C_PREWINDOW in classes:
        return X_PREWINDOW
    if classes <= {C_IDENTICAL, C_AFTER}:
        return X_WINDOW_IDENTITY
    return X_UNRESOLVED


def validate_retokenized_item(
    tokenizer: Any,
    texts: Mapping[str, str],
    archived_contract: Mapping[str, Any],
) -> dict[str, Any]:
    prefix_ids = tokenizer_ids(tokenizer, texts["prefix"])
    ids = {
        role: tokenizer_ids(tokenizer, texts[role])
        for role in ("matched_corr", "matched_ctrl", "swapped_corr", "swapped_ctrl")
    }
    for role, branch_ids in ids.items():
        require(branch_ids[: len(prefix_ids)] == prefix_ids, f"PREFIX_IDENTITY_FAILURE:{role}")

    require(len(prefix_ids) == int(archived_contract["prefix_token_count"]), "PREFIX_TOKEN_COUNT_MISMATCH")
    require(token_id_sha256(prefix_ids) == archived_contract["prefix_token_sha256"], "PREFIX_TOKEN_SHA256_MISMATCH")

    matched_te = first_corr_ctrl_divergence(
        ids["matched_corr"], ids["matched_ctrl"], len(prefix_ids)
    )
    swapped_te = first_corr_ctrl_divergence(
        ids["swapped_corr"], ids["swapped_ctrl"], len(prefix_ids)
    )
    require(matched_te == int(archived_contract["matched_divergence_anchor"]), "MATCHED_TE_MISMATCH")
    require(swapped_te == int(archived_contract["swapped_divergence_anchor"]), "SWAPPED_TE_MISMATCH")
    require(matched_te == swapped_te, "MATCHED_SWAPPED_EVENT_ANCHOR_MISMATCH")

    expected_counts = {
        "matched_corr": int(archived_contract["matched_correction_token_count"]),
        "matched_ctrl": int(archived_contract["matched_control_token_count"]),
        "swapped_corr": int(archived_contract["swapped_correction_token_count"]),
        "swapped_ctrl": int(archived_contract["swapped_control_token_count"]),
    }
    for role, branch_ids in ids.items():
        require(len(branch_ids) == expected_counts[role], f"TOKEN_COUNT_MISMATCH:{role}")

    corr = analyze_role(ids["matched_corr"], ids["swapped_corr"], matched_te, swapped_te)
    ctrl = analyze_role(ids["matched_ctrl"], ids["swapped_ctrl"], matched_te, swapped_te)
    return {
        "prefix_token_count": len(prefix_ids),
        "matched_divergence_anchor": matched_te,
        "swapped_divergence_anchor": swapped_te,
        "correction_role": corr,
        "control_role": ctrl,
    }


def extract_p2_state_by_local(p2: Mapping[str, Any]) -> dict[int, str]:
    require(p2.get("schema_version") == P2_SCHEMA, "P2_SCHEMA_MISMATCH")
    require(int(p2.get("item_count", -1)) == ITEM_COUNT, "P2_ITEM_COUNT_MISMATCH")
    items = p2.get("items")
    require(isinstance(items, list), "P2_ITEMS_REQUIRED")
    state_by_local: dict[int, str] = {}
    for row in items:
        require(isinstance(row, Mapping), "P2_ITEM_NOT_OBJECT")
        local = int(row["local_template_index"])
        cmp = row.get("state_hash_comparison")
        require(isinstance(cmp, Mapping), "P2_STATE_HASH_COMPARISON_REQUIRED")
        classification = str(cmp.get("classification"))
        require(classification in {STATE_IDENTITY, STATE_DIFFERENCE}, "P2_STATE_HASH_CLASSIFICATION_INVALID")
        require(local not in state_by_local, "P2_DUPLICATE_LOCAL_INDEX")
        state_by_local[local] = classification
    require(sorted(state_by_local) == list(range(ITEM_COUNT)), "P2_LOCAL_INDEX_DOMAIN_MISMATCH")
    return state_by_local


def validate_real_inputs(repo_root: Path, p2_path: Path) -> dict[str, Any]:
    # This function is scientific-input I/O and must only be called after authority authentication.
    p0_dir = repo_root / P0_ARCHIVE_REL
    require(p0_dir.is_dir(), "P0_ARCHIVE_MISSING")
    for name, expected in P0_ARTIFACT_SHA256.items():
        path = p0_dir / name
        require(path.is_file(), f"P0_ARTIFACT_MISSING:{name}")
        require(file_sha256(path) == expected, f"P0_ARTIFACT_SHA256_MISMATCH:{name}")

    require(p2_path.is_file(), "P2_ARTIFACT_MISSING")
    require(file_sha256(p2_path) == P2_ARTIFACT_SHA256, "P2_ARTIFACT_SHA256_MISMATCH")

    candidates = parse_canonical_jsonl(p0_dir / "candidate_pool.jsonl")
    contracts = parse_canonical_jsonl(p0_dir / "token_contracts.jsonl")
    mapping = parse_canonical_json(p0_dir / "phase_pair_mapping.json")
    p2 = parse_canonical_json(p2_path)

    candidate_by_local = index_by_local(candidates)
    contract_by_local = index_by_local(contracts)
    mate_map = build_phase_mate_map(mapping)
    state_by_local = extract_p2_state_by_local(p2)

    return {
        "candidate_by_local": candidate_by_local,
        "contract_by_local": contract_by_local,
        "mate_map": mate_map,
        "state_by_local": state_by_local,
    }


def load_local_tokenizer(snapshot_path: Path) -> tuple[Any, dict[str, Any]]:
    # No network fallback: local_files_only=True and a pre-existing local path are mandatory.
    require(snapshot_path.is_dir(), "TOKENIZER_LOCAL_SNAPSHOT_MISSING")
    try:
        from transformers import AutoTokenizer, __version__ as transformers_version
    except Exception as exc:
        raise ContractError("TRANSFORMERS_IMPORT_FAILURE") from exc
    require(transformers_version == TRANSFORMERS_VERSION, "TRANSFORMERS_VERSION_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        use_fast=True,
    )
    require(getattr(tokenizer, "is_fast", False), "TOKENIZER_MUST_BE_FAST")
    files = sorted(
        p for p in snapshot_path.rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    file_hashes = {
        p.relative_to(snapshot_path).as_posix(): file_sha256(p)
        for p in files
    }
    require(bool(file_hashes), "TOKENIZER_SNAPSHOT_EMPTY")
    snapshot_digest = sha256_bytes(canonical_json_bytes(file_hashes))
    return tokenizer, {
        "model_id": HF_MODEL,
        "revision": HF_REVISION,
        "transformers_version": transformers_version,
        "local_files_only": True,
        "snapshot_file_sha256": file_hashes,
        "snapshot_manifest_sha256": snapshot_digest,
    }


def build_real_audit(
    inputs: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_by_local = inputs["candidate_by_local"]
    contract_by_local = inputs["contract_by_local"]
    mate_map = inputs["mate_map"]
    state_by_local = inputs["state_by_local"]

    items: list[dict[str, Any]] = []
    role_counts = {"correction": Counter(), "control": Counter()}
    cross_counts: Counter[str] = Counter()

    for local in range(ITEM_COUNT):
        texts = reconstruct_branch_texts(candidate_by_local, mate_map, local)
        token_rel = validate_retokenized_item(tokenizer, texts, contract_by_local[local])
        state_class = str(state_by_local[local])
        cross = cross_level_classification(
            state_class,
            token_rel["correction_role"],
            token_rel["control_role"],
        )
        role_counts["correction"][token_rel["correction_role"]["classification"]] += 1
        role_counts["control"][token_rel["control_role"]["classification"]] += 1
        cross_counts[cross] += 1
        items.append(
            {
                "local_template_index": local,
                "stable_item_id": candidate_by_local[local]["stable_item_id"],
                "matched_divergence_anchor": token_rel["matched_divergence_anchor"],
                "swapped_divergence_anchor": token_rel["swapped_divergence_anchor"],
                "correction_role": token_rel["correction_role"],
                "control_role": token_rel["control_role"],
                "p2_state_hash_classification": state_class,
                "cross_level_classification": cross,
            }
        )

    return {
        "schema_version": OUTPUT_SCHEMA,
        "item_count": ITEM_COUNT,
        "p0_artifact_sha256": dict(P0_ARTIFACT_SHA256),
        "p2_artifact_sha256": P2_ARTIFACT_SHA256,
        "tokenizer": dict(tokenizer_provenance),
        "correction_role_classification_counts": dict(sorted(role_counts["correction"].items())),
        "control_role_classification_counts": dict(sorted(role_counts["control"].items())),
        "cross_level_classification_counts": dict(sorted(cross_counts.items())),
        "items": items,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "checkpoint_loaded": False,
        "network_access": False,
    }


def write_output_atomic(output_dir: Path, audit: Mapping[str, Any]) -> Path:
    require(not output_dir.exists(), "OUTPUT_ALREADY_EXISTS")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        output = temp_dir / OUTPUT_NAME
        output.write_bytes(canonical_json_bytes(audit, final_lf=True))
        require({p.name for p in temp_dir.iterdir() if p.is_file()} == {OUTPUT_NAME}, "OUTPUT_ARTIFACT_SET_MISMATCH")
        temp_dir.rename(output_dir)
        return output_dir / OUTPUT_NAME
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def execute_real(
    repo_root: Path,
    execution_authority_rel: str,
    p2_path: Path,
    tokenizer_snapshot: Path,
    output_dir: Path,
    *,
    authority_authenticator: Callable[[Path, str], Mapping[str, str]] = authenticate_execution_authority,
    real_input_loader: Callable[[Path, Path], Mapping[str, Any]] = validate_real_inputs,
    tokenizer_loader: Callable[[Path], tuple[Any, Mapping[str, Any]]] = load_local_tokenizer,
) -> dict[str, Any]:
    # Critical invariant: authority authentication precedes every real scientific input read
    # and precedes loading the real frozen tokenizer.
    markers = authority_authenticator(repo_root, execution_authority_rel)
    require(not output_dir.exists(), "OUTPUT_ALREADY_EXISTS")
    inputs = real_input_loader(repo_root, p2_path)
    tokenizer, tokenizer_provenance = tokenizer_loader(tokenizer_snapshot)
    audit = build_real_audit(inputs, tokenizer, tokenizer_provenance)
    output_path = write_output_atomic(output_dir, audit)
    return {
        "status": "COMPLETE_POST_P2_TOKEN_WINDOW_STATIC_AUDIT",
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "authority_authenticated": bool(markers),
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "checkpoint_loaded": False,
        "network_access": False,
    }


class FakeTokenizer:
    def __init__(self, mapping: Mapping[str, Sequence[int]]):
        self.mapping = {str(k): list(v) for k, v in mapping.items()}
        self.is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, list[int]]:
        require(add_special_tokens is False, "SYNTHETIC_SPECIAL_TOKEN_POLICY_DRIFT")
        require(return_attention_mask is False, "SYNTHETIC_ATTENTION_MASK_POLICY_DRIFT")
        require(text in self.mapping, "SYNTHETIC_TEXT_NOT_MAPPED")
        return {"input_ids": list(self.mapping[text])}


def synthetic_self_check() -> dict[str, Any]:
    candidates = {
        0: {
            "local_template_index": 0,
            "stable_item_id": "synthetic-a",
            "prefix_text": "P0:",
            "correction_text": " corr-a",
            "control_text": " ctrl-a",
        },
        1: {
            "local_template_index": 1,
            "stable_item_id": "synthetic-b",
            "prefix_text": "P1:",
            "correction_text": " corr-b",
            "control_text": " ctrl-b",
        },
    }
    mate_map = {0: 1, 1: 0}
    texts = reconstruct_branch_texts(candidates, mate_map, 0)
    mapping = {
        texts["prefix"]: [10, 11],
        texts["matched_corr"]: [10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        texts["matched_ctrl"]: [10, 11, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        texts["swapped_corr"]: [10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 99],
        texts["swapped_ctrl"]: [10, 11, 20, 31, 32, 33, 34, 35, 36, 37, 38, 98],
    }
    tokenizer = FakeTokenizer(mapping)
    contract = {
        "prefix_token_count": 2,
        "prefix_token_sha256": token_id_sha256([10, 11]),
        "matched_correction_token_count": 12,
        "matched_control_token_count": 12,
        "swapped_correction_token_count": 12,
        "swapped_control_token_count": 12,
        "matched_divergence_anchor": 3,
        "swapped_divergence_anchor": 3,
    }
    token_rel = validate_retokenized_item(tokenizer, texts, contract)
    require(token_rel["correction_role"]["classification"] == C_AFTER, "SYNTHETIC_CORR_CLASSIFICATION_FAILURE")
    require(token_rel["control_role"]["classification"] == C_AFTER, "SYNTHETIC_CTRL_CLASSIFICATION_FAILURE")
    require(
        cross_level_classification(
            STATE_IDENTITY,
            token_rel["correction_role"],
            token_rel["control_role"],
        )
        == X_WINDOW_IDENTITY,
        "SYNTHETIC_CROSS_LEVEL_FAILURE",
    )

    snapshot = canonical_json_bytes(token_rel)
    repeat = validate_retokenized_item(tokenizer, texts, contract)
    require(snapshot == canonical_json_bytes(repeat), "SYNTHETIC_REPEAT_IDENTITY_FAILURE")

    return {
        "status": "PASS_SYNTHETIC_POST_P2_TOKEN_WINDOW_AUDIT_CORE",
        "checks": {
            "branch_reconstruction": "PASS",
            "fake_tokenizer_only": True,
            "window_coordinate_count_per_role": len(WINDOW_COORDS),
            "canonical_repeat_identity": "PASS",
        },
        "real_p0_artifact_read": False,
        "real_p2_artifact_read": False,
        "real_hf_tokenizer_loaded": False,
        "network_access": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-self-check", action="store_true")
    parser.add_argument("--repo-root")
    parser.add_argument("--execution-authority")
    parser.add_argument("--p2-artifact")
    parser.add_argument("--tokenizer-snapshot")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.synthetic_self_check:
        print(json.dumps(synthetic_self_check(), sort_keys=True, indent=2))
        return 0

    required = {
        "--repo-root": args.repo_root,
        "--execution-authority": args.execution_authority,
        "--p2-artifact": args.p2_artifact,
        "--tokenizer-snapshot": args.tokenizer_snapshot,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if not value]
    require(not missing, "MISSING_REAL_EXECUTION_ARGUMENTS:" + ",".join(missing))
    result = execute_real(
        Path(args.repo_root).resolve(),
        str(args.execution_authority),
        Path(args.p2_artifact).resolve(),
        Path(args.tokenizer_snapshot).resolve(),
        Path(args.output_dir).resolve(),
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
