from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


BASE_PATH = Path(__file__).with_name(
    "longterm_k0_rvg_post_p2_token_window_static_audit.py"
)
BASE_SPEC = importlib.util.spec_from_file_location(
    "frozen_post_p2_token_window_base", BASE_PATH
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("BASE_IMPLEMENTATION_IMPORT_SPEC_FAILURE")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


ContractError = base.ContractError
require = base.require
canonical_json_bytes = base.canonical_json_bytes
parse_canonical_json = base.parse_canonical_json
parse_canonical_jsonl = base.parse_canonical_jsonl
file_sha256 = base.file_sha256
token_id_sha256 = base.token_id_sha256
tokenizer_ids = base.tokenizer_ids
first_corr_ctrl_divergence = base.first_corr_ctrl_divergence
reconstruct_branch_texts = base.reconstruct_branch_texts
build_phase_mate_map = base.build_phase_mate_map
index_by_local = base.index_by_local
parse_authority_markers = base.parse_authority_markers
tokenizer_snapshot_manifest = base.tokenizer_snapshot_manifest


IMPLEMENTATION_AUTHORITY_COMMIT = (
    "8b6d56bd313948fa34d122cda0b7a06ebfa40527"
)
IMPLEMENTATION_AUTHORITY_REL = (
    "reports/"
    "longterm_k0_rvg_post_p2_active_encoding_token_coordinate_equivalence_"
    "implementation_authority_spec_candidate.md"
)
RUNNER_REL = (
    "scripts/"
    "longterm_k0_rvg_post_p2_active_encoding_token_coordinate_equivalence.py"
)

EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

P0_ARCHIVE_REL = base.P0_ARCHIVE_REL
P0_ARTIFACT_SHA256 = dict(base.P0_ARTIFACT_SHA256)

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION_LABEL = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
ACTIVE_SNAPSHOT_MANIFEST_SHA256 = (
    "f45af0fad1ae940487eb6461c65f13cf1634b6141f4e4e03b6c88bc38fa6db9c"
)

ITEM_COUNT = 336
WINDOW_COORDS = tuple(range(-1, 8))

OUTPUT_NAME = "active_encoding_token_coordinate_equivalence.json"
OUTPUT_SCHEMA = (
    "k0-rvg-post-p2-active-encoding-token-coordinate-equivalence-v1"
)

PASS_VERDICT = "PASS_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE"
BLOCKED_VERDICT = "BLOCKED_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE"

HISTORICAL_STATUS = "HISTORICAL_P1_BYTE_IDENTITY_NOT_ESTABLISHED"
HISTORICAL_RECOVERY = "NOT_RECOVERABLE_FROM_FROZEN_REPOSITORY_EVIDENCE"


REQUIRED_FUTURE_MARKERS = {
    "REAL_ACTIVE_ENCODING_EQUIVALENCE_AUDIT_AUTHORIZED": "YES",
    "REAL_P0_SCIENTIFIC_INPUT_READ_AUTHORIZED": "YES",
    "REAL_P2_ARTIFACT_READ_AUTHORIZED": "NO",
    "POST_P2_TOKEN_WINDOW_REAL_ARTIFACT_EXECUTION_AUTHORIZED": "NO",
    "SCIENTIFIC_TOKEN_RELATION_CLASSIFICATION_AUTHORIZED": "NO",
    "TOKENIZER_REEXECUTION_AUTHORIZED": "YES",
    "REAL_HF_TOKENIZER_LOAD_AUTHORIZED": "YES",
    "MODEL_CONSTRUCTION_AUTHORIZED": "NO",
    "CHECKPOINT_LOADING_AUTHORIZED": "NO",
    "SCIENTIFIC_MODEL_FORWARD_AUTHORIZED": "NO",
    "SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED": "NO",
    "LOGITS_READ_AUTHORIZED": "NO",
    "NETWORK_ACCESS_AUTHORIZED": "NO",
    "SNAPSHOT_DOWNLOAD_AUTHORIZED": "NO",
    "TRAINING_AUTHORIZED": "NO",
    "EVALUATION_AUTHORIZED": "NO",
    "KAGGLE_EXECUTION_AUTHORIZED": "NO",
    "ACTIVE_TOKENIZER_MODEL_ID": HF_MODEL,
    "ACTIVE_TOKENIZER_REVISION_LABEL": HF_REVISION_LABEL,
    "ACTIVE_TOKENIZER_TRANSFORMERS_VERSION": TRANSFORMERS_VERSION,
    "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256":
        ACTIVE_SNAPSHOT_MANIFEST_SHA256,
    "HISTORICAL_P1_BYTE_IDENTITY_STATUS": HISTORICAL_STATUS,
    "HISTORICAL_P1_BYTE_IDENTITY_RECOVERY": HISTORICAL_RECOVERY,
}


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


def validate_execution_markers(
    markers: Mapping[str, str],
    actual_impl_commit: str,
) -> None:
    for key, expected in REQUIRED_FUTURE_MARKERS.items():
        require(
            markers.get(key) == expected,
            f"EXECUTION_AUTHORITY_MARKER_MISMATCH:{key}",
        )

    declared = markers.get(
        "ACTIVE_ENCODING_EQUIVALENCE_IMPLEMENTATION_COMMIT", ""
    )
    require(
        len(declared) == 40,
        "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_INVALID",
    )
    require(
        all(ch in "0123456789abcdef" for ch in declared),
        "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_NONHEX",
    )
    require(
        declared == actual_impl_commit,
        "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_MISMATCH",
    )


def authenticate_execution_authority(
    root: Path,
    authority_rel: str,
) -> dict[str, str]:
    rel = Path(authority_rel)
    require(not rel.is_absolute(), "EXECUTION_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "EXECUTION_AUTHORITY_PATH_TRAVERSAL")

    normalized = rel.as_posix()
    require(
        normalized.startswith("reports/"),
        "EXECUTION_AUTHORITY_NOT_REPORT",
    )

    require(
        _git(root, "branch", "--show-current") == EXPECTED_BRANCH,
        "GIT_BRANCH_MISMATCH",
    )

    try:
        _git(root, "ls-files", "--error-unmatch", "--", normalized)
    except ContractError as exc:
        raise ContractError("EXECUTION_AUTHORITY_NOT_TRACKED") from exc

    worktree = root / normalized
    require(worktree.is_file(), "EXECUTION_AUTHORITY_MISSING")

    head_bytes = subprocess.check_output(
        ["git", "show", f"HEAD:{normalized}"],
        cwd=root,
    )
    require(
        worktree.read_bytes() == head_bytes,
        "EXECUTION_AUTHORITY_WORKTREE_DRIFT",
    )

    markers = parse_authority_markers(
        worktree.read_text(encoding="utf-8")
    )

    actual_impl_commit = _git(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        RUNNER_REL,
    )
    validate_execution_markers(markers, actual_impl_commit)

    impl_ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", actual_impl_commit, "HEAD"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        impl_ancestor == 0,
        "IMPLEMENTATION_COMMIT_NOT_ANCESTOR",
    )

    auth_ancestor = subprocess.call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            IMPLEMENTATION_AUTHORITY_COMMIT,
            "HEAD",
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(
        auth_ancestor == 0,
        "IMPLEMENTATION_AUTHORITY_COMMIT_NOT_ANCESTOR",
    )

    return markers


def validate_real_inputs(repo_root: Path) -> dict[str, Any]:
    p0_dir = repo_root / P0_ARCHIVE_REL
    require(p0_dir.is_dir(), "P0_ARCHIVE_MISSING")

    for name, expected_sha in P0_ARTIFACT_SHA256.items():
        path = p0_dir / name
        require(path.is_file(), f"P0_ARTIFACT_MISSING:{name}")
        require(
            file_sha256(path) == expected_sha,
            f"P0_ARTIFACT_SHA256_MISMATCH:{name}",
        )

    candidates = parse_canonical_jsonl(
        p0_dir / "candidate_pool.jsonl"
    )
    contracts = parse_canonical_jsonl(
        p0_dir / "token_contracts.jsonl"
    )
    mapping = parse_canonical_json(
        p0_dir / "phase_pair_mapping.json"
    )

    candidate_by_local = index_by_local(candidates)
    contract_by_local = index_by_local(contracts)
    mate_map = build_phase_mate_map(mapping)

    return {
        "candidate_by_local": candidate_by_local,
        "contract_by_local": contract_by_local,
        "mate_map": mate_map,
    }


def load_local_tokenizer(
    snapshot_path: Path,
    markers: Mapping[str, str],
    *,
    transformers_importer: Callable[[], tuple[Any, str]] | None = None,
    expected_manifest_sha256: str = ACTIVE_SNAPSHOT_MANIFEST_SHA256,
) -> tuple[Any, dict[str, Any]]:
    file_hashes, snapshot_digest = tokenizer_snapshot_manifest(
        snapshot_path
    )

    require(
        snapshot_digest == expected_manifest_sha256,
        "ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256_MISMATCH",
    )
    require(
        markers.get("ACTIVE_TOKENIZER_SNAPSHOT_MANIFEST_SHA256")
        == expected_manifest_sha256,
        "EXECUTION_AUTHORITY_ACTIVE_TOKENIZER_MANIFEST_MISMATCH",
    )

    if transformers_importer is None:
        try:
            from transformers import (
                AutoTokenizer,
                __version__ as transformers_version,
            )
        except Exception as exc:
            raise ContractError("TRANSFORMERS_IMPORT_FAILURE") from exc
        auto_tokenizer_cls = AutoTokenizer
    else:
        auto_tokenizer_cls, transformers_version = (
            transformers_importer()
        )

    require(
        transformers_version == TRANSFORMERS_VERSION,
        "TRANSFORMERS_VERSION_MISMATCH",
    )

    with tempfile.TemporaryDirectory(
        prefix="k0-rvg-active-encoding-"
    ) as temp_name:
        curated = Path(temp_name)

        for rel_text, expected_sha in file_hashes.items():
            src = snapshot_path / Path(rel_text)
            dst = curated / Path(rel_text)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            require(
                file_sha256(dst) == expected_sha,
                "TOKENIZER_CURATED_COPY_SHA256_MISMATCH",
            )

        tokenizer = auto_tokenizer_cls.from_pretrained(
            str(curated),
            local_files_only=True,
            use_fast=True,
        )
        require(
            getattr(tokenizer, "is_fast", False),
            "TOKENIZER_MUST_BE_FAST",
        )

    return tokenizer, {
        "model_id": HF_MODEL,
        "revision_label": HF_REVISION_LABEL,
        "transformers_version": transformers_version,
        "local_files_only": True,
        "snapshot_file_sha256": file_hashes,
        "snapshot_manifest_sha256": snapshot_digest,
        "historical_p1_byte_identity_status": HISTORICAL_STATUS,
        "historical_p1_byte_identity_recovery": HISTORICAL_RECOVERY,
    }


def encode_repeat(tokenizer: Any, text: str) -> list[int]:
    first = tokenizer_ids(tokenizer, text)
    second = tokenizer_ids(tokenizer, text)
    require(
        first == second,
        "ACTIVE_ENCODING_REPEAT_IDENTITY_FAILURE",
    )
    return first


def _contract_bool(
    contract: Mapping[str, Any],
    key: str,
) -> bool:
    require(key in contract, f"CONTRACT_FIELD_MISSING:{key}")
    value = contract[key]
    require(type(value) is bool, f"CONTRACT_BOOL_REQUIRED:{key}")
    return value


def _window_available(
    corr_ids: Sequence[int],
    ctrl_ids: Sequence[int],
    te: int,
) -> bool:
    return (
        te - 1 >= 0
        and te + 7 < len(corr_ids)
        and te + 7 < len(ctrl_ids)
    )


def _sequence_evidence(ids: Sequence[int]) -> dict[str, Any]:
    token_ids = [int(value) for value in ids]
    return {
        "token_count": len(token_ids),
        "token_id_sha256": token_id_sha256(token_ids),
        "token_ids": token_ids,
    }


def _window_evidence(
    ids: Sequence[int],
    te: int,
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []

    for relative in WINDOW_COORDS:
        absolute = te + relative
        require(
            absolute >= 0,
            f"WINDOW_COORDINATE_NEGATIVE:{relative}",
        )
        require(
            absolute < len(ids),
            f"WINDOW_COORDINATE_MISSING:{relative}",
        )
        out.append(
            {
                "relative_coordinate": relative,
                "absolute_index": absolute,
                "token_id": int(ids[absolute]),
            }
        )

    return out


def validate_equivalence_item(
    tokenizer: Any,
    texts: Mapping[str, str],
    archived_contract: Mapping[str, Any],
    item: Mapping[str, Any],
    mate_item: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        str(archived_contract.get("stable_item_id"))
        == str(item.get("stable_item_id")),
        "STABLE_ITEM_ID_MISMATCH",
    )
    require(
        str(archived_contract.get("pair_id"))
        == str(item.get("pair_id")),
        "PAIR_ID_MISMATCH",
    )
    require(
        str(archived_contract.get("phase_mate_stable_id"))
        == str(mate_item.get("stable_item_id")),
        "PHASE_MATE_STABLE_ID_MISMATCH",
    )
    require(
        str(archived_contract.get("phase_mate_pair_id"))
        == str(mate_item.get("pair_id")),
        "PHASE_MATE_PAIR_ID_MISMATCH",
    )

    prefix_ids = encode_repeat(tokenizer, texts["prefix"])

    branch_ids = {
        role: encode_repeat(tokenizer, texts[role])
        for role in (
            "matched_corr",
            "matched_ctrl",
            "swapped_corr",
            "swapped_ctrl",
        )
    }

    for role, ids in branch_ids.items():
        require(
            ids[: len(prefix_ids)] == prefix_ids,
            f"PREFIX_IDENTITY_FAILURE:{role}",
        )

    require(
        len(prefix_ids)
        == int(archived_contract["prefix_token_count"]),
        "PREFIX_TOKEN_COUNT_MISMATCH",
    )
    require(
        token_id_sha256(prefix_ids)
        == str(archived_contract["prefix_token_sha256"]),
        "PREFIX_TOKEN_SHA256_MISMATCH",
    )

    expected_counts = {
        "matched_corr":
            int(archived_contract["matched_correction_token_count"]),
        "matched_ctrl":
            int(archived_contract["matched_control_token_count"]),
        "swapped_corr":
            int(archived_contract["swapped_correction_token_count"]),
        "swapped_ctrl":
            int(archived_contract["swapped_control_token_count"]),
    }

    for role, ids in branch_ids.items():
        require(
            len(ids) == expected_counts[role],
            f"TOKEN_COUNT_MISMATCH:{role}",
        )

    matched_te = first_corr_ctrl_divergence(
        branch_ids["matched_corr"],
        branch_ids["matched_ctrl"],
        len(prefix_ids),
    )
    swapped_te = first_corr_ctrl_divergence(
        branch_ids["swapped_corr"],
        branch_ids["swapped_ctrl"],
        len(prefix_ids),
    )

    require(
        matched_te
        == int(archived_contract["matched_divergence_anchor"]),
        "MATCHED_TE_MISMATCH",
    )
    require(
        swapped_te
        == int(archived_contract["swapped_divergence_anchor"]),
        "SWAPPED_TE_MISMATCH",
    )

    if "matched_divergence_offset_from_prefix" in archived_contract:
        require(
            matched_te - len(prefix_ids)
            == int(
                archived_contract[
                    "matched_divergence_offset_from_prefix"
                ]
            ),
            "MATCHED_DIVERGENCE_OFFSET_MISMATCH",
        )

    if "swapped_divergence_offset_from_prefix" in archived_contract:
        require(
            swapped_te - len(prefix_ids)
            == int(
                archived_contract[
                    "swapped_divergence_offset_from_prefix"
                ]
            ),
            "SWAPPED_DIVERGENCE_OFFSET_MISMATCH",
        )

    matched_w8 = _window_available(
        branch_ids["matched_corr"],
        branch_ids["matched_ctrl"],
        matched_te,
    )
    swapped_w8 = _window_available(
        branch_ids["swapped_corr"],
        branch_ids["swapped_ctrl"],
        swapped_te,
    )

    if "matched_w8_available" in archived_contract:
        require(
            matched_w8
            == _contract_bool(
                archived_contract,
                "matched_w8_available",
            ),
            "MATCHED_W8_AVAILABILITY_MISMATCH",
        )

    if "swapped_w8_available" in archived_contract:
        require(
            swapped_w8
            == _contract_bool(
                archived_contract,
                "swapped_w8_available",
            ),
            "SWAPPED_W8_AVAILABILITY_MISMATCH",
        )

    require(
        matched_w8 and swapped_w8,
        "W8_REQUIRED_COORDINATES_UNAVAILABLE",
    )

    sequences = {
        "prefix": _sequence_evidence(prefix_ids),
        **{
            role: _sequence_evidence(ids)
            for role, ids in branch_ids.items()
        },
    }

    windows = {
        "matched_corr":
            _window_evidence(branch_ids["matched_corr"], matched_te),
        "matched_ctrl":
            _window_evidence(branch_ids["matched_ctrl"], matched_te),
        "swapped_corr":
            _window_evidence(branch_ids["swapped_corr"], swapped_te),
        "swapped_ctrl":
            _window_evidence(branch_ids["swapped_ctrl"], swapped_te),
    }

    return {
        "exact_contract_match": True,
        "stable_item_id": str(item["stable_item_id"]),
        "pair_id": str(item["pair_id"]),
        "phase_mate_stable_id":
            str(mate_item["stable_item_id"]),
        "phase_mate_pair_id":
            str(mate_item["pair_id"]),
        "prefix_token_count": len(prefix_ids),
        "matched_divergence_anchor": matched_te,
        "swapped_divergence_anchor": swapped_te,
        "matched_divergence_offset_from_prefix":
            matched_te - len(prefix_ids),
        "swapped_divergence_offset_from_prefix":
            swapped_te - len(prefix_ids),
        "matched_w8_available": matched_w8,
        "swapped_w8_available": swapped_w8,
        "sequences": sequences,
        "event_relative_windows": windows,
    }


def build_equivalence_audit(
    inputs: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_by_local = inputs["candidate_by_local"]
    contract_by_local = inputs["contract_by_local"]
    mate_map = inputs["mate_map"]

    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pass_count = 0

    for local in range(ITEM_COUNT):
        item = candidate_by_local[local]
        mate_local = int(mate_map[local])
        mate_item = candidate_by_local[mate_local]

        try:
            texts = reconstruct_branch_texts(
                candidate_by_local,
                mate_map,
                local,
            )
            evidence = validate_equivalence_item(
                tokenizer,
                texts,
                contract_by_local[local],
                item,
                mate_item,
            )
        except ContractError as exc:
            failure = {
                "local_template_index": local,
                "stable_item_id": str(item["stable_item_id"]),
                "error_code": str(exc),
            }
            failures.append(failure)
            items.append(
                {
                    "local_template_index": local,
                    "stable_item_id": str(item["stable_item_id"]),
                    "status": BLOCKED_VERDICT,
                    "error_code": str(exc),
                }
            )
            continue

        pass_count += 1
        items.append(
            {
                "local_template_index": local,
                "status": "EXACT_CONTRACT_MATCH",
                **evidence,
            }
        )

    failure_count = ITEM_COUNT - pass_count

    verdict = (
        PASS_VERDICT
        if pass_count == ITEM_COUNT and failure_count == 0
        else BLOCKED_VERDICT
    )

    return {
        "schema_version": OUTPUT_SCHEMA,
        "diagnostic_scope":
            "ACTIVE_ENCODING_TOKEN_COORDINATE_PROVENANCE_ONLY",
        "equivalence_verdict": verdict,
        "item_count": ITEM_COUNT,
        "exact_item_pass_count": pass_count,
        "exact_item_failure_count": failure_count,
        "p0_artifact_sha256": dict(P0_ARTIFACT_SHA256),
        "active_tokenizer": dict(tokenizer_provenance),
        "historical_p1_byte_identity_status": HISTORICAL_STATUS,
        "historical_p1_byte_identity_recovery":
            HISTORICAL_RECOVERY,
        "historical_p1_full_token_sequence_identity_established":
            False,
        "scientific_token_relation_classification_performed":
            False,
        "p2_artifact_read": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "logits_read": False,
        "network_access": False,
        "failures": failures,
        "items": items,
    }


def write_output_atomic(
    output_dir: Path,
    audit: Mapping[str, Any],
) -> Path:
    require(
        not output_dir.exists(),
        "OUTPUT_ALREADY_EXISTS",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=output_dir.name + ".partial-",
            dir=output_dir.parent,
        )
    )

    try:
        output = temp_dir / OUTPUT_NAME
        output.write_bytes(
            canonical_json_bytes(audit, final_lf=True)
        )
        require(
            {
                path.name
                for path in temp_dir.iterdir()
                if path.is_file()
            }
            == {OUTPUT_NAME},
            "OUTPUT_ARTIFACT_SET_MISMATCH",
        )
        temp_dir.rename(output_dir)
        return output_dir / OUTPUT_NAME
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def execute_real(
    repo_root: Path,
    execution_authority_rel: str,
    tokenizer_snapshot: Path,
    output_dir: Path,
    *,
    authority_authenticator:
        Callable[[Path, str], Mapping[str, str]]
        = authenticate_execution_authority,
    real_input_loader:
        Callable[[Path], Mapping[str, Any]]
        = validate_real_inputs,
    tokenizer_loader:
        Callable[
            [Path, Mapping[str, str]],
            tuple[Any, Mapping[str, Any]],
        ]
        = load_local_tokenizer,
) -> dict[str, Any]:
    # Critical order:
    # authority -> output collision -> real P0 -> real tokenizer -> audit.
    markers = authority_authenticator(
        repo_root,
        execution_authority_rel,
    )

    require(
        not output_dir.exists(),
        "OUTPUT_ALREADY_EXISTS",
    )

    inputs = real_input_loader(repo_root)
    tokenizer, tokenizer_provenance = tokenizer_loader(
        tokenizer_snapshot,
        markers,
    )

    audit = build_equivalence_audit(
        inputs,
        tokenizer,
        tokenizer_provenance,
    )

    output_path = write_output_atomic(
        output_dir,
        audit,
    )

    return {
        "status": audit["equivalence_verdict"],
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "authority_authenticated": bool(markers),
        "real_p0_scientific_input_read": True,
        "real_p2_artifact_read": False,
        "scientific_token_relation_classification_performed":
            False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "network_access": False,
    }


class FakeTokenizer:
    def __init__(self, mapping: Mapping[str, Sequence[int]]):
        self.mapping = {
            str(key): [int(value) for value in values]
            for key, values in mapping.items()
        }
        self.is_fast = True

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ) -> dict[str, list[int]]:
        require(
            add_special_tokens is False,
            "SYNTHETIC_SPECIAL_TOKEN_POLICY_DRIFT",
        )
        require(
            return_attention_mask is False,
            "SYNTHETIC_ATTENTION_MASK_POLICY_DRIFT",
        )
        require(
            text in self.mapping,
            "SYNTHETIC_TEXT_NOT_MAPPED",
        )
        return {"input_ids": list(self.mapping[text])}


def _synthetic_fixture() -> tuple[
    dict[int, dict[str, Any]],
    dict[int, int],
    dict[str, list[int]],
    dict[str, Any],
]:
    candidates = {
        0: {
            "local_template_index": 0,
            "stable_item_id": "synthetic-a",
            "pair_id": "pair-a",
            "prefix_text": "P:",
            "correction_text": " corr-a",
            "control_text": " ctrl-a",
        },
        1: {
            "local_template_index": 1,
            "stable_item_id": "synthetic-b",
            "pair_id": "pair-b",
            "prefix_text": "Q:",
            "correction_text": " corr-b",
            "control_text": " ctrl-b",
        },
    }

    mate_map = {0: 1, 1: 0}
    texts = reconstruct_branch_texts(
        candidates,
        mate_map,
        0,
    )

    mapping = {
        texts["prefix"]:
            [10, 11],
        texts["matched_corr"]:
            [10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        texts["matched_ctrl"]:
            [10, 11, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        texts["swapped_corr"]:
            [10, 11, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        texts["swapped_ctrl"]:
            [10, 11, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    }

    contract = {
        "local_template_index": 0,
        "stable_item_id": "synthetic-a",
        "pair_id": "pair-a",
        "phase_mate_stable_id": "synthetic-b",
        "phase_mate_pair_id": "pair-b",
        "prefix_token_count": 2,
        "prefix_token_sha256":
            token_id_sha256([10, 11]),
        "matched_correction_token_count": 13,
        "matched_control_token_count": 13,
        "swapped_correction_token_count": 13,
        "swapped_control_token_count": 13,
        "matched_divergence_anchor": 3,
        "swapped_divergence_anchor": 3,
        "matched_divergence_offset_from_prefix": 1,
        "swapped_divergence_offset_from_prefix": 1,
        "matched_w8_available": True,
        "swapped_w8_available": True,
    }

    return candidates, mate_map, mapping, contract


def synthetic_self_check() -> dict[str, Any]:
    candidates, mate_map, mapping, contract = (
        _synthetic_fixture()
    )
    texts = reconstruct_branch_texts(
        candidates,
        mate_map,
        0,
    )

    tokenizer = FakeTokenizer(mapping)

    first = validate_equivalence_item(
        tokenizer,
        texts,
        contract,
        candidates[0],
        candidates[1],
    )
    second = validate_equivalence_item(
        tokenizer,
        texts,
        contract,
        candidates[0],
        candidates[1],
    )

    require(
        canonical_json_bytes(first)
        == canonical_json_bytes(second),
        "SYNTHETIC_CANONICAL_REPEAT_IDENTITY_FAILURE",
    )

    return {
        "status":
            "PASS_SYNTHETIC_ACTIVE_ENCODING_TOKEN_COORDINATE_EQUIVALENCE_CORE",
        "checks": {
            "exact_branch_reconstruction": "PASS",
            "exact_prefix_contract": "PASS",
            "exact_branch_counts": "PASS",
            "exact_event_anchors": "PASS",
            "exact_divergence_offsets": "PASS",
            "exact_w8_availability": "PASS",
            "active_encoding_repeat_identity": "PASS",
            "canonical_repeat_identity": "PASS",
            "window_coordinate_count_per_branch":
                len(WINDOW_COORDS),
        },
        "real_p0_scientific_input_read": False,
        "real_p2_artifact_read": False,
        "real_hf_tokenizer_loaded": False,
        "network_access": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
        "scientific_token_relation_classification_performed":
            False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-self-check",
        action="store_true",
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--execution-authority")
    parser.add_argument("--tokenizer-snapshot")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    if args.synthetic_self_check:
        result = synthetic_self_check()
        print(
            canonical_json_bytes(
                result,
                final_lf=True,
            ).decode("utf-8"),
            end="",
        )
        return 0

    required = (
        args.repo_root,
        args.execution_authority,
        args.tokenizer_snapshot,
        args.output_dir,
    )
    require(
        all(required),
        "REAL_EXECUTION_ARGUMENTS_REQUIRED",
    )

    result = execute_real(
        Path(args.repo_root),
        str(args.execution_authority),
        Path(args.tokenizer_snapshot),
        Path(args.output_dir),
    )
    print(
        canonical_json_bytes(
            result,
            final_lf=True,
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())