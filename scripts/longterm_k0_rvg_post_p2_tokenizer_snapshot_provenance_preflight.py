from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IMPLEMENTATION_AUTHORITY_COMMIT = "c3fab42efcff287dcc96ee89026d927f4d69df63"
IMPLEMENTATION_AUTHORITY_REL = (
    "reports/longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_"
    "preflight_implementation_authority_spec_candidate.md"
)
SCRIPT_REL = "scripts/longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py"
EXPECTED_BRANCH = "longterm-k-series-native-state-kinematics"

HF_MODEL = "state-spaces/mamba-130m-hf"
HF_REVISION = "5708daa364c50b880e7bd92eab456e0d34492ee9"
TRANSFORMERS_VERSION = "5.12.1"
MODEL_CACHE_DIRNAME = "models--state-spaces--mamba-130m-hf"

TOKENIZER_SNAPSHOT_PATTERNS = (
    "config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
)

OUTPUT_NAME = "tokenizer_snapshot_provenance_preflight.json"
OUTPUT_SCHEMA = "k0-rvg-post-p2-tokenizer-snapshot-provenance-preflight-v1"

PASS = "PASS_REVISION_BOUND_LOCAL_SNAPSHOT_MANIFEST"
PARTIAL = "PARTIAL_LOCAL_SNAPSHOT_PROVENANCE"
BLOCKED = "BLOCKED_LOCAL_TOKENIZER_SNAPSHOT_PROVENANCE"
HISTORICAL_UNKNOWN = "HISTORICAL_P1_BYTE_IDENTITY_NOT_ESTABLISHED"
SYNTHETIC_PASS = "PASS_SYNTHETIC_TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_CORE"

REQUIRED_FUTURE_EXECUTION_MARKERS = {
    "TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_REAL_CACHE_EXECUTION_AUTHORIZED": "YES",
    "LOCAL_REAL_TOKENIZER_CACHE_METADATA_READ_AUTHORIZED": "YES",
    "LOCAL_REAL_TOKENIZER_FILE_BYTE_READ_AUTHORIZED": "YES",
    "REAL_TOKENIZER_FILE_SHA256_AUTHORIZED": "YES",
    "REAL_HF_TOKENIZER_LOAD_AUTHORIZED": "NO",
    "TOKENIZER_REEXECUTION_AUTHORIZED": "NO",
    "NETWORK_ACCESS_AUTHORIZED": "NO",
    "SNAPSHOT_DOWNLOAD_AUTHORIZED": "NO",
    "MODEL_CONSTRUCTION_AUTHORIZED": "NO",
    "CHECKPOINT_LOADING_AUTHORIZED": "NO",
    "SCIENTIFIC_MODEL_FORWARD_AUTHORIZED": "NO",
    "SCIENTIFIC_RECURRENT_STATE_READ_AUTHORIZED": "NO",
    "TOKENIZER_MODEL_ID": HF_MODEL,
    "TOKENIZER_REVISION": HF_REVISION,
    "TOKENIZER_TRANSFORMERS_VERSION": TRANSFORMERS_VERSION,
}

_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class ContractError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    prefix = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + raw).hexdigest()


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


def authenticate_future_execution_authority(
    repo_root: Path,
    authority_rel: str,
) -> dict[str, str]:
    rel = Path(authority_rel)
    require(not rel.is_absolute(), "EXECUTION_AUTHORITY_PATH_ABSOLUTE")
    require(".." not in rel.parts, "EXECUTION_AUTHORITY_PATH_TRAVERSAL")
    normalized = rel.as_posix()
    require(normalized.startswith("reports/"), "EXECUTION_AUTHORITY_NOT_REPORT")

    branch = _git(repo_root, "branch", "--show-current")
    require(branch == EXPECTED_BRANCH, "GIT_BRANCH_MISMATCH")

    try:
        _git(repo_root, "ls-files", "--error-unmatch", "--", normalized)
    except ContractError as exc:
        raise ContractError("EXECUTION_AUTHORITY_NOT_TRACKED") from exc

    worktree = repo_root / normalized
    require(worktree.is_file(), "EXECUTION_AUTHORITY_MISSING")
    try:
        head_bytes = subprocess.check_output(
            ["git", "show", f"HEAD:{normalized}"],
            cwd=repo_root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("EXECUTION_AUTHORITY_HEAD_READ_FAILURE") from exc
    require(
        worktree.read_bytes() == head_bytes,
        "EXECUTION_AUTHORITY_WORKTREE_DRIFT",
    )

    markers = parse_authority_markers(worktree.read_text(encoding="utf-8"))
    for key, expected in REQUIRED_FUTURE_EXECUTION_MARKERS.items():
        require(
            markers.get(key) == expected,
            f"EXECUTION_AUTHORITY_MARKER_MISMATCH:{key}",
        )

    actual_impl_commit = _git(
        repo_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        SCRIPT_REL,
    )
    require(
        len(actual_impl_commit) == 40
        and all(ch in "0123456789abcdef" for ch in actual_impl_commit),
        "IMPLEMENTATION_COMMIT_INVALID",
    )
    require(
        markers.get("TOKENIZER_SNAPSHOT_PROVENANCE_PREFLIGHT_IMPLEMENTATION_COMMIT")
        == actual_impl_commit,
        "EXECUTION_AUTHORITY_IMPLEMENTATION_COMMIT_MISMATCH",
    )
    ancestor = subprocess.call(
        ["git", "merge-base", "--is-ancestor", actual_impl_commit, "HEAD"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    require(ancestor == 0, "IMPLEMENTATION_COMMIT_NOT_ANCESTOR")
    return markers


def manifest_from_entries(entries: Iterable[tuple[str, str]]) -> tuple[dict[str, str], str]:
    file_hashes: dict[str, str] = {}
    for rel_text, digest in entries:
        require(rel_text not in file_hashes, "TOKENIZER_SNAPSHOT_DUPLICATE_RELATIVE_PATH")
        require(
            len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
            f"TOKENIZER_FILE_SHA256_INVALID:{rel_text}",
        )
        file_hashes[rel_text] = digest
    require(bool(file_hashes), "TOKENIZER_SNAPSHOT_EMPTY")
    ordered = dict(sorted(file_hashes.items()))
    return ordered, sha256_bytes(canonical_json_bytes(ordered))


def tokenizer_snapshot_manifest(snapshot_path: Path) -> tuple[dict[str, str], str]:
    # Intentionally matches the frozen post-P2 runner semantics exactly.
    require(snapshot_path.is_dir(), "TOKENIZER_LOCAL_SNAPSHOT_MISSING")
    entries: list[tuple[str, str]] = []
    for path in sorted(snapshot_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_path)
        if not any(rel.match(pattern) for pattern in TOKENIZER_SNAPSHOT_PATTERNS):
            continue
        entries.append((rel.as_posix(), file_sha256(path)))
    return manifest_from_entries(entries)


def normalized_abs(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def validate_snapshot_namespace(
    snapshot_path: Path,
    model_cache_root: Path,
) -> str:
    cache_abs = normalized_abs(model_cache_root)
    snapshot_abs = normalized_abs(snapshot_path)

    require(
        cache_abs.name == MODEL_CACHE_DIRNAME,
        "MODEL_CACHE_NAMESPACE_MISMATCH",
    )
    require(
        snapshot_abs.parent == cache_abs / "snapshots",
        "SNAPSHOT_CACHE_NAMESPACE_MISMATCH",
    )
    require(
        snapshot_abs.name == HF_REVISION,
        "SNAPSHOT_REVISION_MISMATCH",
    )
    return f"{MODEL_CACHE_DIRNAME}/snapshots/{HF_REVISION}"


def verify_digest_named_blob(raw: bytes, blob_basename: str) -> dict[str, Any]:
    if _HEX64.fullmatch(blob_basename):
        actual = sha256_bytes(raw)
        expected = blob_basename.lower()
        return {
            "kind": "RAW_SHA256",
            "identifier": expected,
            "actual_digest": actual,
            "verified": actual == expected,
            "status": "VERIFIED_SHA256_BLOB" if actual == expected else "SHA256_BLOB_MISMATCH",
        }
    if _HEX40.fullmatch(blob_basename):
        actual = git_blob_sha1(raw)
        expected = blob_basename.lower()
        return {
            "kind": "GIT_BLOB_SHA1",
            "identifier": expected,
            "actual_digest": actual,
            "verified": actual == expected,
            "status": "VERIFIED_GIT_BLOB_SHA1" if actual == expected else "GIT_BLOB_SHA1_MISMATCH",
        }
    return {
        "kind": "UNSUPPORTED",
        "identifier": blob_basename,
        "actual_digest": None,
        "verified": False,
        "status": "UNSUPPORTED_BLOB_IDENTIFIER",
    }


def _relative_if_within(path: Path, root: Path) -> str | None:
    try:
        return normalized_abs(path).relative_to(normalized_abs(root)).as_posix()
    except ValueError:
        return None


def inspect_local_snapshot(
    snapshot_path: Path,
    model_cache_root: Path,
) -> dict[str, Any]:
    snapshot_identity = validate_snapshot_namespace(snapshot_path, model_cache_root)
    file_hashes, manifest_sha256 = tokenizer_snapshot_manifest(snapshot_path)

    cache_abs = normalized_abs(model_cache_root)
    observations: list[dict[str, Any]] = []
    blocked = False
    partial = False

    for rel_text, digest in file_hashes.items():
        path = snapshot_path / Path(rel_text)
        raw = path.read_bytes()
        require(sha256_bytes(raw) == digest, f"TOKENIZER_FILE_CHANGED_DURING_PREFLIGHT:{rel_text}")

        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ContractError(f"TOKENIZER_FILE_RESOLVE_FAILURE:{rel_text}") from exc

        resolved_rel = _relative_if_within(resolved, cache_abs)
        within_namespace = resolved_rel is not None
        if not within_namespace:
            blocked = True

        blob_evidence: dict[str, Any]
        if resolved_rel is not None and Path(resolved_rel).parts[:1] == ("blobs",):
            blob_evidence = verify_digest_named_blob(raw, resolved.name)
            if blob_evidence["status"] in {"SHA256_BLOB_MISMATCH", "GIT_BLOB_SHA1_MISMATCH"}:
                blocked = True
            elif blob_evidence["status"] == "UNSUPPORTED_BLOB_IDENTIFIER":
                partial = True
        else:
            blob_evidence = {
                "kind": "NONE",
                "identifier": None,
                "actual_digest": None,
                "verified": False,
                "status": "NO_CONTENT_ADDRESSED_BLOB_TARGET",
            }
            partial = True

        observations.append(
            {
                "relative_path": rel_text,
                "raw_sha256": digest,
                "is_symlink": path.is_symlink(),
                "resolved_target_relative_to_model_cache": resolved_rel,
                "within_model_cache_namespace": within_namespace,
                "content_addressed_evidence": blob_evidence,
            }
        )

    if blocked:
        verdict = BLOCKED
    elif partial:
        verdict = PARTIAL
    else:
        verdict = PASS

    return {
        "schema_version": OUTPUT_SCHEMA,
        "tokenizer_model_id": HF_MODEL,
        "tokenizer_revision": HF_REVISION,
        "tokenizer_transformers_version": TRANSFORMERS_VERSION,
        "normalized_snapshot_identity": snapshot_identity,
        "selected_file_family_patterns": list(TOKENIZER_SNAPSHOT_PATTERNS),
        "selected_files": observations,
        "selected_file_count": len(observations),
        "tokenizer_snapshot_manifest_sha256": manifest_sha256,
        "historical_p1_byte_identity_status": HISTORICAL_UNKNOWN,
        "assertions": {
            "network_access": False,
            "snapshot_download": False,
            "real_hf_tokenizer_loaded": False,
            "scientific_text_tokenized": False,
            "model_constructed": False,
            "checkpoint_loaded": False,
            "scientific_model_forward_executed": False,
            "scientific_recurrent_state_read": False,
        },
        "provenance_verdict": verdict,
    }


def run_real_cache_preflight(
    *,
    repo_root: Path,
    authority_rel: str,
    snapshot_path: Path,
    model_cache_root: Path,
) -> dict[str, Any]:
    # Gate ordering is intentional: no snapshot/cache metadata or bytes are touched
    # before the separately frozen execution authority is authenticated.
    authenticate_future_execution_authority(repo_root, authority_rel)
    return inspect_local_snapshot(snapshot_path, model_cache_root)


def synthetic_self_check() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="k0-rvg-tokenizer-preflight-synth-") as tmp:
        root = Path(tmp)
        cache = root / MODEL_CACHE_DIRNAME
        snapshot = cache / "snapshots" / HF_REVISION
        snapshot.mkdir(parents=True)

        (snapshot / "config.json").write_bytes(b'{"model_type":"mamba"}\n')
        (snapshot / "tokenizer.json").write_bytes(b'{"version":"1.0"}\n')
        (snapshot / "README.md").write_bytes(b"excluded\n")
        nested = snapshot / "nested"
        nested.mkdir()
        (nested / "tokenizer_config.json").write_bytes(b'{"fast":true}\n')

        first_map, first_digest = tokenizer_snapshot_manifest(snapshot)
        second_map, second_digest = tokenizer_snapshot_manifest(snapshot)
        require(first_map == second_map, "SYNTHETIC_MANIFEST_MAP_NONDETERMINISTIC")
        require(first_digest == second_digest, "SYNTHETIC_MANIFEST_DIGEST_NONDETERMINISTIC")
        require("README.md" not in first_map, "SYNTHETIC_UNRELATED_FILE_INCLUDED")
        require(
            set(first_map) == {"config.json", "tokenizer.json", "nested/tokenizer_config.json"},
            "SYNTHETIC_FILE_SELECTION_MISMATCH",
        )

        raw64 = b"synthetic-sha256-blob"
        ev64 = verify_digest_named_blob(raw64, sha256_bytes(raw64))
        require(ev64["status"] == "VERIFIED_SHA256_BLOB", "SYNTHETIC_SHA256_BLOB_FAILURE")

        raw40 = b"synthetic-git-blob"
        ev40 = verify_digest_named_blob(raw40, git_blob_sha1(raw40))
        require(ev40["status"] == "VERIFIED_GIT_BLOB_SHA1", "SYNTHETIC_GIT_BLOB_FAILURE")

        identity = validate_snapshot_namespace(snapshot, cache)
        require(
            identity == f"{MODEL_CACHE_DIRNAME}/snapshots/{HF_REVISION}",
            "SYNTHETIC_SNAPSHOT_IDENTITY_MISMATCH",
        )

    return {
        "status": SYNTHETIC_PASS,
        "manifest_semantics": "FROZEN_POST_P2_EQUIVALENT_ALGORITHM",
        "real_tokenizer_cache_metadata_read": False,
        "real_tokenizer_file_byte_read": False,
        "real_hf_tokenizer_loaded": False,
        "scientific_text_tokenized": False,
        "real_p0_artifact_read": False,
        "real_p2_artifact_read": False,
        "network_access": False,
        "model_constructed": False,
        "checkpoint_loaded": False,
        "scientific_model_forward_executed": False,
        "scientific_recurrent_state_read": False,
    }


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(dict(value), final_lf=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K0-RVG post-P2 tokenizer snapshot provenance preflight"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-self-check", action="store_true")
    mode.add_argument("--real-cache-preflight", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--execution-authority")
    parser.add_argument("--snapshot-path", type=Path)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.synthetic_self_check:
        require(
            all(
                value is None
                for value in (
                    args.repo_root,
                    args.execution_authority,
                    args.snapshot_path,
                    args.model_cache_root,
                    args.output,
                )
            ),
            "SYNTHETIC_SELF_CHECK_REAL_PATH_ARGUMENT_FORBIDDEN",
        )
        print(canonical_json_bytes(synthetic_self_check()).decode("utf-8"))
        return 0

    require(args.repo_root is not None, "REAL_PREFLIGHT_REPO_ROOT_REQUIRED")
    require(bool(args.execution_authority), "REAL_PREFLIGHT_EXECUTION_AUTHORITY_REQUIRED")
    require(args.snapshot_path is not None, "REAL_PREFLIGHT_SNAPSHOT_PATH_REQUIRED")
    require(args.model_cache_root is not None, "REAL_PREFLIGHT_MODEL_CACHE_ROOT_REQUIRED")
    require(args.output is not None, "REAL_PREFLIGHT_OUTPUT_REQUIRED")
    require(args.output.name == OUTPUT_NAME, "REAL_PREFLIGHT_OUTPUT_NAME_MISMATCH")

    result = run_real_cache_preflight(
        repo_root=args.repo_root,
        authority_rel=str(args.execution_authority),
        snapshot_path=args.snapshot_path,
        model_cache_root=args.model_cache_root,
    )
    write_canonical_json(args.output, result)
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
