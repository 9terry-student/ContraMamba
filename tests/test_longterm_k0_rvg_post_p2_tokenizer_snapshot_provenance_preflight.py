from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "longterm_k0_rvg_post_p2_tokenizer_snapshot_provenance_preflight.py"
FROZEN_RUNNER = ROOT / "scripts" / "longterm_k0_rvg_post_p2_token_window_static_audit.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_module(SCRIPT, "k0_rvg_tokenizer_preflight")


class PreflightTests(unittest.TestCase):
    def make_snapshot(self, root: Path) -> tuple[Path, Path]:
        cache = root / preflight.MODEL_CACHE_DIRNAME
        snapshot = cache / "snapshots" / preflight.HF_REVISION
        snapshot.mkdir(parents=True)
        return cache, snapshot

    def test_constants_bind_frozen_authority(self):
        self.assertEqual(
            preflight.IMPLEMENTATION_AUTHORITY_COMMIT,
            "c3fab42efcff287dcc96ee89026d927f4d69df63",
        )
        self.assertEqual(preflight.HF_MODEL, "state-spaces/mamba-130m-hf")
        self.assertEqual(
            preflight.HF_REVISION,
            "5708daa364c50b880e7bd92eab456e0d34492ee9",
        )
        self.assertEqual(preflight.TRANSFORMERS_VERSION, "5.12.1")


    def test_implementation_imports_are_standard_library_only(self):
        import ast

        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        forbidden = {"transformers", "huggingface_hub", "torch", "mamba_ssm", "requests", "httpx", "urllib3"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertTrue(forbidden.isdisjoint(imported), sorted(forbidden & imported))

    def test_file_selection_recursive_and_unrelated_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot = self.make_snapshot(Path(tmp))
            (snapshot / "config.json").write_bytes(b"config")
            (snapshot / "tokenizer.json").write_bytes(b"tok")
            (snapshot / "special_tokens_map.json").write_bytes(b"special")
            (snapshot / "vocab.txt").write_bytes(b"vocab")
            (snapshot / "merges.txt").write_bytes(b"merges")
            (snapshot / "README.md").write_bytes(b"no")
            nested = snapshot / "nested"
            nested.mkdir()
            (nested / "tokenizer_config.json").write_bytes(b"nested")
            (nested / "weights.bin").write_bytes(b"no")

            mapping, _ = preflight.tokenizer_snapshot_manifest(snapshot)

            self.assertEqual(
                list(mapping),
                [
                    "config.json",
                    "merges.txt",
                    "nested/tokenizer_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "vocab.txt",
                ],
            )
            self.assertNotIn("README.md", mapping)
            self.assertNotIn("nested/weights.bin", mapping)

    def test_raw_sha256_correctness(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot = self.make_snapshot(Path(tmp))
            raw = b"abc\x00def"
            (snapshot / "tokenizer.json").write_bytes(raw)
            mapping, _ = preflight.tokenizer_snapshot_manifest(snapshot)
            self.assertEqual(mapping["tokenizer.json"], hashlib.sha256(raw).hexdigest())

    def test_manifest_is_sorted_deterministic_and_repeat_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, snapshot = self.make_snapshot(Path(tmp))
            (snapshot / "tokenizer.json").write_bytes(b"z")
            (snapshot / "config.json").write_bytes(b"a")
            first_map, first_digest = preflight.tokenizer_snapshot_manifest(snapshot)
            second_map, second_digest = preflight.tokenizer_snapshot_manifest(snapshot)
            self.assertEqual(first_map, second_map)
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(list(first_map), sorted(first_map))
            expected_raw = json.dumps(
                first_map,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            self.assertEqual(first_digest, hashlib.sha256(expected_raw).hexdigest())

    def test_exact_equivalence_with_frozen_runner_manifest_semantics(self):
        frozen = load_module(FROZEN_RUNNER, "frozen_post_p2_runner")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_bytes(b'{"x":1}')
            (snapshot / "tokenizer.json").write_bytes(b'{"y":2}')
            nested = snapshot / "nested"
            nested.mkdir()
            (nested / "tokenizer_config.json").write_bytes(b'{"z":3}')
            (snapshot / "README.md").write_bytes(b"excluded")

            expected = frozen.tokenizer_snapshot_manifest(snapshot)
            actual = preflight.tokenizer_snapshot_manifest(snapshot)
            self.assertEqual(actual, expected)

    def test_missing_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            with self.assertRaisesRegex(
                preflight.ContractError,
                "TOKENIZER_LOCAL_SNAPSHOT_MISSING",
            ):
                preflight.tokenizer_snapshot_manifest(missing)

    def test_empty_matching_set_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "README.md").write_bytes(b"excluded")
            with self.assertRaisesRegex(preflight.ContractError, "TOKENIZER_SNAPSHOT_EMPTY"):
                preflight.tokenizer_snapshot_manifest(snapshot)

    def test_duplicate_relative_path_ambiguity_fails_closed(self):
        digest = "0" * 64
        with self.assertRaisesRegex(
            preflight.ContractError,
            "TOKENIZER_SNAPSHOT_DUPLICATE_RELATIVE_PATH",
        ):
            preflight.manifest_from_entries(
                [
                    ("tokenizer.json", digest),
                    ("tokenizer.json", digest),
                ]
            )

    def test_sha256_blob_verification_and_mismatch(self):
        raw = b"sha256-content"
        good = hashlib.sha256(raw).hexdigest()
        verified = preflight.verify_digest_named_blob(raw, good)
        self.assertEqual(verified["status"], "VERIFIED_SHA256_BLOB")
        self.assertTrue(verified["verified"])

        bad = "0" * 64
        if bad == good:
            bad = "1" * 64
        mismatch = preflight.verify_digest_named_blob(raw, bad)
        self.assertEqual(mismatch["status"], "SHA256_BLOB_MISMATCH")
        self.assertFalse(mismatch["verified"])

    def test_git_blob_sha1_verification_and_mismatch(self):
        raw = b"git-blob-content"
        good = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        verified = preflight.verify_digest_named_blob(raw, good)
        self.assertEqual(verified["status"], "VERIFIED_GIT_BLOB_SHA1")
        self.assertTrue(verified["verified"])

        bad = "0" * 40
        if bad == good:
            bad = "1" * 40
        mismatch = preflight.verify_digest_named_blob(raw, bad)
        self.assertEqual(mismatch["status"], "GIT_BLOB_SHA1_MISMATCH")
        self.assertFalse(mismatch["verified"])

    def test_unsupported_blob_identifier_is_partial_evidence(self):
        result = preflight.verify_digest_named_blob(b"x", "not-a-supported-digest")
        self.assertEqual(result["status"], "UNSUPPORTED_BLOB_IDENTIFIER")
        self.assertFalse(result["verified"])

    def test_snapshot_revision_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / preflight.MODEL_CACHE_DIRNAME
            snapshot = cache / "snapshots" / "wrong-revision"
            snapshot.mkdir(parents=True)
            with self.assertRaisesRegex(preflight.ContractError, "SNAPSHOT_REVISION_MISMATCH"):
                preflight.validate_snapshot_namespace(snapshot, cache)

    def test_model_cache_namespace_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "models--wrong--repo"
            snapshot = cache / "snapshots" / preflight.HF_REVISION
            snapshot.mkdir(parents=True)
            with self.assertRaisesRegex(preflight.ContractError, "MODEL_CACHE_NAMESPACE_MISMATCH"):
                preflight.validate_snapshot_namespace(snapshot, cache)

    def test_missing_execution_authority_fails_before_snapshot_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, snapshot = self.make_snapshot(root)
            (snapshot / "tokenizer.json").write_bytes(b"synthetic")
            with mock.patch.object(
                preflight,
                "authenticate_future_execution_authority",
                side_effect=preflight.ContractError("EXECUTION_AUTHORITY_MISSING"),
            ) as auth_mock, mock.patch.object(
                preflight,
                "inspect_local_snapshot",
            ) as inspect_mock:
                with self.assertRaisesRegex(
                    preflight.ContractError,
                    "EXECUTION_AUTHORITY_MISSING",
                ):
                    preflight.run_real_cache_preflight(
                        repo_root=root,
                        authority_rel="reports/future.md",
                        snapshot_path=snapshot,
                        model_cache_root=cache,
                    )
                auth_mock.assert_called_once()
                inspect_mock.assert_not_called()

    def test_duplicate_authority_markers_rejected(self):
        text = "`A = YES`\n`A = NO`\n"
        with self.assertRaisesRegex(
            preflight.ContractError,
            "DUPLICATE_AUTHORITY_MARKER:A",
        ):
            preflight.parse_authority_markers(text)

    def test_canonical_json_repeat_identity(self):
        value = {"z": [3, 2, 1], "a": {"β": True}}
        first = preflight.canonical_json_bytes(value)
        second = preflight.canonical_json_bytes(value)
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
        )

    def test_synthetic_self_check_has_all_forbidden_real_actions_false(self):
        result = preflight.synthetic_self_check()
        self.assertEqual(result["status"], preflight.SYNTHETIC_PASS)
        keys = [
            "real_tokenizer_cache_metadata_read",
            "real_tokenizer_file_byte_read",
            "real_hf_tokenizer_loaded",
            "scientific_text_tokenized",
            "real_p0_artifact_read",
            "real_p2_artifact_read",
            "network_access",
            "model_constructed",
            "checkpoint_loaded",
            "scientific_model_forward_executed",
            "scientific_recurrent_state_read",
        ]
        for key in keys:
            self.assertIs(result[key], False, key)

    def test_synthetic_cli_rejects_real_path_arguments(self):
        with self.assertRaisesRegex(
            preflight.ContractError,
            "SYNTHETIC_SELF_CHECK_REAL_PATH_ARGUMENT_FORBIDDEN",
        ):
            preflight.main(
                [
                    "--synthetic-self-check",
                    "--snapshot-path",
                    "C:/real/cache",
                ]
            )

    def test_real_cli_requires_execution_authority_before_use(self):
        with self.assertRaisesRegex(
            preflight.ContractError,
            "REAL_PREFLIGHT_EXECUTION_AUTHORITY_REQUIRED",
        ):
            preflight.main(
                [
                    "--real-cache-preflight",
                    "--repo-root",
                    str(ROOT),
                ]
            )


if __name__ == "__main__":
    unittest.main()
