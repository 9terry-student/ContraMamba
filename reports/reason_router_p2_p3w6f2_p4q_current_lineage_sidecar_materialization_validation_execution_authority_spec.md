# P3-W6-F2-P4-Q Current-Lineage Sidecar Materialization Validation Execution Authority Specification

Authority/version:

`P3W6F2P4Q_CURRENT_LINEAGE_SIDECAR_MATERIALIZATION_VALIDATION_EXECUTION_AUTHORITY_V1`

This document is a candidate execution-authority specification only. It becomes
canonical only after independent static verification PASS and immutable freeze.
Candidate creation does not authorize materialization, validation execution,
Kaggle, CPU execution, GPU execution, trainer rebind, manifest modification,
parameter adoption, A0, training, evaluation, staging, commit, or push.

After freeze, the workflow controller must supply exactly one runtime
parameter:

`P4Q_AUTHORITY_FREEZE`

`P4Q_AUTHORITY_FREEZE` is the actual immutable P4-Q authority commit after the
candidate is frozen. The candidate cannot know that future commit SHA in
advance. Fetching or proving availability of that authority commit does not
authorize checking out the P4-Q authority commit for Gate 1. Gate 1 and Gate 2
execute only from detached exact implementation HEAD
`2f9e6076791358922e3ebd70e89533d9cb83b458`. The executable wrapper body is
frozen in this authority; the only post-freeze substitution is the runtime
value of `P4Q_AUTHORITY_FREEZE`. No mutable-main authority semantics are
permitted.

P4-Q authorizes exactly one future CPU-only Kaggle execution after freeze:

1. Gate 1: canonical P4-L sidecar and provenance materialization using the
   validated implementation at exact detached commit
   `2f9e6076791358922e3ebd70e89533d9cb83b458`.
2. Gate 2: immediate read-only validation of the just-materialized canonical
   artifacts against the frozen P4-L contract.

No other execution is authorized by this specification.

## A. Candidate Creation State

Candidate creation requires:

- HEAD exactly
  `016274d707fb335c623fc654bf4e3de022d8b705`.
- Tracked worktree clean.
- Index clean.
- Candidate path absent before creation:
  `reports/reason_router_p2_p3w6f2_p4q_current_lineage_sidecar_materialization_validation_execution_authority_spec.md`.
- Pre-existing unrelated untracked files must remain untouched.
- Exactly one new untracked authority specification may be created.

Current candidate creation evidence:

- `git rev-parse HEAD` returned
  `016274d707fb335c623fc654bf4e3de022d8b705`.
- `git diff --quiet` exited `0`.
- `git diff --cached --quiet` exited `0`.
- `git status --short --branch` reported branch `main...origin/main`, no
  tracked modifications, and pre-existing unrelated untracked files.
- `Test-Path -LiteralPath '<candidate path>'` returned `False`.
- `git ls-files -- '<candidate path>'` returned no tracked path.

Warnings from inaccessible local pytest/cache directories during broad git
status are environmental read warnings only. They are not PASS evidence for
any execution gate and do not authorize ignoring tracked or index dirt.

## B. Namespace And History Search

Repository content and Git history were searched for:

- `P4-Q`
- `P4Q`
- `current-lineage sidecar materialization`
- `canonical integrity sidecar materialization`
- `materialization/validation authority`
- `P3W6F2P4Q_CURRENT_LINEAGE_SIDECAR_MATERIALIZATION_VALIDATION_EXECUTION_AUTHORITY`

Current tracked content found no existing P4-Q/P4Q canonical namespace and no
existing applicable materialization/validation authority for this exact
current-lineage sidecar execution.

Git history search found no applicable collision.

P4-Q is BLOCKED if independent verification finds a namespace collision,
existing applicable authority, or authority conflict.

## C. Authority Chain

P4-Q consumes, without superseding except where explicitly stated:

- Current workflow-controller instruction.
- Current HEAD and frozen P4-P PASS evidence:
  `016274d707fb335c623fc654bf4e3de022d8b705`.
- Frozen P4-P authority:
  `178ebee692bb11cc397e3000e25d09b0680e8814`.
- Validated P4-O builder implementation:
  `2f9e6076791358922e3ebd70e89533d9cb83b458`.
- Builder source SHA256:
  `b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`.
- Frozen P4-L artifact contract:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`.
- Frozen P4-L contract freeze:
  `80cb034792f03226cf6e22c196c1229ed4e6dd62`.

P4-P established code correctness PASS and runtime publication readiness
ESTABLISHED, including actual Linux `renameat2(RENAME_NOREPLACE)` PASS.
P4-P did not establish canonical artifact/provenance validity or scientific
conclusion.

P4-Q is an execution authority for canonical materialization plus validation
only. It does not authorize trainer rebind, A0, training, evaluation, parameter
adoption, manifest modification, or GPU use.

## D. Exact Implementation Identity

Gate 1 and Gate 2 must run after checkout to exact detached implementation
commit:

`2f9e6076791358922e3ebd70e89533d9cb83b458`

Gate 1 must not run from:

- `016274d707fb335c623fc654bf4e3de022d8b705`
- mutable `main`
- P4-Q authority freeze HEAD
- any descendant while relabeling builder provenance

Before Gate 1, require:

- `P4Q_AUTHORITY_FREEZE` is set.
- `P4Q_AUTHORITY_FREEZE` matches exactly 40 lowercase hex characters.
- `git cat-file -e "${P4Q_AUTHORITY_FREEZE}^{commit}"` succeeds.
- that commit contains the frozen P4-Q authority path:
  `reports/reason_router_p2_p3w6f2_p4q_current_lineage_sidecar_materialization_validation_execution_authority_spec.md`.
- proving authority availability does not check out or execute from the
  authority freeze commit.
- `git rev-parse HEAD == 2f9e6076791358922e3ebd70e89533d9cb83b458`
- builder source path exists exactly at
  `scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- builder source SHA256 exactly
  `b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`

The canonical output namespace and provenance `builder_source_commit` must bind
directly to the validated implementation commit. No provenance relabeling is
permitted.

## E. Canonical Output Contract

The exact canonical output directory is:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458`

The exact expected files are:

1. `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
2. `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

The canonical output directory must be absent before Gate 1. Existing file,
directory, symlink, or broken symlink at that path is BLOCKED.

No delete, reset, clean, overwrite, backup, move-aside, or path replacement is
authorized.

After successful Gate 1, the canonical output directory must be a normal
directory, not a symlink, and must contain exactly the two expected files. No
sibling `.p4l-staging-*` directory may remain.

## F. Gate 1 Exact Command

Gate 1 materialization command is frozen exactly:

```bash
python scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py --repo-root . --materialize
```

Do not pass `--builder-commit`.

Because execution HEAD is exact
`2f9e6076791358922e3ebd70e89533d9cb83b458`, the builder must resolve
`builder_source_commit` from actual HEAD.

Do not pass `--created-at`. Observed `created_at` is runtime provenance and
must be recorded, not predicted.

No other builder CLI invocation is authorized.

## G. Gate 1 PASS/FAIL Contract

Gate 1 must capture exact stdout JSON and exit code.

Gate 1 PASS requires all of:

- exit code `0`
- stdout is a JSON object
- `status == "PASS"`
- `row_count == 3600`
- `artifact_materialization_requested == true`
- `publish_status == "PUBLISHED"`
- `training_authorized == false`
- `evaluation_authorized == false`
- canonical directory now exists
- exactly the two canonical files exist
- no leftover sibling `.p4l-staging-*` directory
- tracked git state remains clean

Gate 1 must record, without prediction:

- `sidecar_physical_sha256`
- `sidecar_semantic_sha256`
- `provenance_physical_sha256`

If the builder blocks or exits nonzero, P4-Q is FAIL unless the cause is an
external or environmental inability unrelated to implementation/artifact
semantics, in which case P4-Q is BLOCKED.

## H. Exact Gate 1 + Gate 2 Execution Wrapper

After freeze, execution must use this exact Bash wrapper body. It covers
preconditions, Gate 1 exactly once, Gate 2 exactly once only if Gate 1 succeeds,
and postconditions.

The wrapper uses `set -euo pipefail` except where temporarily disabled solely
to capture the Gate 1 exit status. Gate 1 stdout and stderr are captured into
separate files inside an attempt-owned system temporary directory, not under
`reports/`, so exact stdout bytes are preserved. Gate 1 stdout is parsed from
that captured file. The temp directory may be removed by shell trap only
because it is attempt-owned scratch outside canonical paths.

Gate 2 is read-only with respect to repository artifacts. It must never unlink,
rename, replace, mkdir, write, chmod, recursively remove, or open repository
artifacts in write/append/update mode. It may read repository artifacts and
print validation observations/results only.

```bash
#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="2f9e6076791358922e3ebd70e89533d9cb83b458"
EXPECTED_BUILDER_SHA="b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d"
P4Q_AUTHORITY_PATH="reports/reason_router_p2_p3w6f2_p4q_current_lineage_sidecar_materialization_validation_execution_authority_spec.md"
OUTPUT_DIR="reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_${EXPECTED_HEAD}"
SIDECAR_NAME="p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME="p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"

fail() {
  printf 'P4Q_FAIL:%s\n' "$1" >&2
  exit 1
}

blocked() {
  printf 'P4Q_BLOCKED:%s\n' "$1" >&2
  exit 2
}

case "${P4Q_AUTHORITY_FREEZE:-}" in
  "")
    blocked "P4Q_AUTHORITY_FREEZE_UNSET"
    ;;
  *[!0123456789abcdef]*)
    blocked "P4Q_AUTHORITY_FREEZE_NOT_LOWERCASE_HEX"
    ;;
esac
if [ "${#P4Q_AUTHORITY_FREEZE}" -ne 40 ]; then
  blocked "P4Q_AUTHORITY_FREEZE_NOT_40_HEX"
fi
git cat-file -e "${P4Q_AUTHORITY_FREEZE}^{commit}" || blocked "P4Q_AUTHORITY_FREEZE_COMMIT_UNAVAILABLE"
git cat-file -e "${P4Q_AUTHORITY_FREEZE}:${P4Q_AUTHORITY_PATH}" || blocked "P4Q_AUTHORITY_PATH_NOT_IN_FREEZE"

test "$(git rev-parse HEAD)" = "${EXPECTED_HEAD}" || blocked "EXECUTION_HEAD_MISMATCH"
git diff --quiet || blocked "TRACKED_WORKTREE_DIRTY_BEFORE_GATE1"
git diff --cached --quiet || blocked "INDEX_DIRTY_BEFORE_GATE1"

builder_path="scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py"
test -f "${builder_path}" || blocked "BUILDER_SOURCE_MISSING"
observed_builder_sha="$(sha256sum "${builder_path}" | awk '{print $1}')"
test "${observed_builder_sha}" = "${EXPECTED_BUILDER_SHA}" || blocked "BUILDER_SHA_MISMATCH"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|-1)
    ;;
  *)
    blocked "GPU_NOT_DISABLED_BEFORE_GATE1"
    ;;
esac

if [ -e "${OUTPUT_DIR}" ] || [ -L "${OUTPUT_DIR}" ]; then
  blocked "CANONICAL_OUTPUT_PREEXISTS"
fi

repo_root="$(pwd -P)"
reports_root="${repo_root}/reports"
attempt_tmp="$(mktemp -d "/tmp/p4q-gate1.XXXXXX")" || blocked "ATTEMPT_TMP_MKTEMP_UNAVAILABLE"
attempt_tmp_real="$(cd "${attempt_tmp}" && pwd -P)" || blocked "ATTEMPT_TMP_REALPATH_UNAVAILABLE"
case "${attempt_tmp_real}" in
  "${repo_root}"|"${repo_root}"/*)
    echo "P4Q_PRECONDITION=BLOCKED_ATTEMPT_TMP_INSIDE_REPO" >&2
    exit 2
    ;;
esac
case "${attempt_tmp_real}" in
  "${reports_root}"|"${reports_root}"/*)
    echo "P4Q_PRECONDITION=BLOCKED_ATTEMPT_TMP_INSIDE_REPORTS" >&2
    exit 2
    ;;
esac
trap 'rm -rf "${attempt_tmp}"' EXIT
gate1_stdout="${attempt_tmp}/gate1.stdout.json"
gate1_stderr="${attempt_tmp}/gate1.stderr.txt"

set +e
python scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py --repo-root . --materialize >"${gate1_stdout}" 2>"${gate1_stderr}"
gate1_exit="$?"
set -e

printf '%s\n' '---P4Q_GATE1_STDOUT_BEGIN---'
cat "${gate1_stdout}"
printf '%s\n' '---P4Q_GATE1_STDOUT_END---'
printf '%s\n' '---P4Q_GATE1_STDERR_BEGIN---' >&2
cat "${gate1_stderr}" >&2
printf '%s\n' '---P4Q_GATE1_STDERR_END---' >&2
printf 'P4Q_GATE1_EXIT=%s\n' "${gate1_exit}"

if [ "${gate1_exit}" -ne 0 ]; then
  printf 'P4Q_GATE1_FAILURE_NO_GATE2_NO_RETRY_NO_CANONICAL_CLEANUP\n' >&2
  exit "${gate1_exit}"
fi

P4Q_GATE1_STDOUT_PATH="${gate1_stdout}"
export P4Q_GATE1_STDOUT_PATH

gate1_exports="${attempt_tmp}/gate1.exports"
python - <<'PY' >"${gate1_exports}"
import json
import os
import pathlib
import shlex
import sys

def require(condition, message):
    if not condition:
        raise SystemExit(message)

raw_bytes = pathlib.Path(os.environ["P4Q_GATE1_STDOUT_PATH"]).read_bytes()
raw = raw_bytes.decode("utf-8")
decoder = json.JSONDecoder(parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"NON_FINITE:{token}")))
obj, end = decoder.raw_decode(raw)
require(raw[end:].strip() == "", "GATE1_STDOUT_TRAILING_DATA")
require(isinstance(obj, dict), "GATE1_STDOUT_NOT_OBJECT")
require(obj.get("status") == "PASS", "GATE1_STATUS_NOT_PASS")
require(obj.get("row_count") == 3600, "GATE1_ROW_COUNT_MISMATCH")
require(obj.get("artifact_materialization_requested") is True, "GATE1_MATERIALIZE_NOT_TRUE")
require(obj.get("publish_status") == "PUBLISHED", "GATE1_NOT_PUBLISHED")
require(obj.get("training_authorized") is False, "GATE1_TRAINING_AUTHORIZED")
require(obj.get("evaluation_authorized") is False, "GATE1_EVALUATION_AUTHORIZED")
for key in ("sidecar_physical_sha256", "sidecar_semantic_sha256", "provenance_physical_sha256"):
    value = obj.get(key)
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"GATE1_HASH_INVALID:{key}")
print("P4Q_GATE1_SIDECAR_PHYSICAL_SHA256=" + shlex.quote(obj["sidecar_physical_sha256"]))
print("P4Q_GATE1_SIDECAR_SEMANTIC_SHA256=" + shlex.quote(obj["sidecar_semantic_sha256"]))
print("P4Q_GATE1_PROVENANCE_PHYSICAL_SHA256=" + shlex.quote(obj["provenance_physical_sha256"]))
PY
. "${gate1_exports}"
export P4Q_GATE1_SIDECAR_PHYSICAL_SHA256
export P4Q_GATE1_SIDECAR_SEMANTIC_SHA256
export P4Q_GATE1_PROVENANCE_PHYSICAL_SHA256

test -d "${OUTPUT_DIR}" || fail "CANONICAL_OUTPUT_DIR_MISSING_AFTER_GATE1"
test ! -L "${OUTPUT_DIR}" || fail "CANONICAL_OUTPUT_DIR_SYMLINK_AFTER_GATE1"
test -f "${OUTPUT_DIR}/${SIDECAR_NAME}" || fail "CANONICAL_SIDECAR_MISSING_AFTER_GATE1"
test ! -L "${OUTPUT_DIR}/${SIDECAR_NAME}" || fail "CANONICAL_SIDECAR_SYMLINK_AFTER_GATE1"
test -f "${OUTPUT_DIR}/${PROVENANCE_NAME}" || fail "CANONICAL_PROVENANCE_MISSING_AFTER_GATE1"
test ! -L "${OUTPUT_DIR}/${PROVENANCE_NAME}" || fail "CANONICAL_PROVENANCE_SYMLINK_AFTER_GATE1"
test "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" = "${PROVENANCE_NAME} ${SIDECAR_NAME} " || fail "CANONICAL_OUTPUT_ENTRIES_MISMATCH_AFTER_GATE1"
test -z "$(find "$(dirname "${OUTPUT_DIR}")" -maxdepth 1 -name '.p4l-staging-*' -print -quit)" || fail "LEFTOVER_P4L_STAGING_SIBLING_AFTER_GATE1"
test -z "$(find "$(dirname "${OUTPUT_DIR}")" -maxdepth 1 \( -name '*p4q*backup*' -o -name '*p4q*move-aside*' -o -name '*p4q*moved*' \) -print -quit)" || fail "P4Q_BACKUP_OR_MOVE_ASIDE_ARTIFACT_AFTER_GATE1"
git diff --quiet || fail "TRACKED_WORKTREE_DIRTY_AFTER_GATE1"
git diff --cached --quiet || fail "INDEX_DIRTY_AFTER_GATE1"

python - <<'PY'
import collections
import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys

PASS_TOKEN = "P3W6F2P4Q_CURRENT_LINEAGE_SIDECAR_ARTIFACT_VALIDATION_PASS"
EXPECTED_HEAD = "2f9e6076791358922e3ebd70e89533d9cb83b458"
EXPECTED_BUILDER_SHA = "b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d"
EXPECTED_ROW_COUNT = 3600
SOURCE_DATASET_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl"
SOURCE_PHYSICAL_SHA = "eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3"
SOURCE_SEMANTIC_SHA = "3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b"
HISTORICAL_SOURCE_SHA = "f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640"
OUTPUT_DIR_REL = "reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458"
SIDECAR_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl"
PROVENANCE_NAME = "p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json"
BUILDER_REL = "scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py"
P4B_ROWS_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_rows.jsonl"
P4B_SUMMARY_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_summary.json"
P4B_PROVENANCE_REL = "reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/p3w6f2_p4b_r1_stage185_predicate_realization_compatibility_provenance_manifest.json"
P4B_ROWS_SHA = "59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f"
P4B_SUMMARY_SHA = "ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8"
P4B_PROVENANCE_SHA = "09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6"
SOURCE_SEMANTIC_FIELDS = (
    "id",
    "pair_id",
    "claim",
    "evidence",
    "final_label",
    "frame_compatible_label",
    "predicate_covered_label",
    "sufficiency_label",
    "polarity_label",
    "primary_failure_type",
    "intervention_type",
)
P2_REQUIRED_FIELDS = {
    "row_id",
    "split",
    "pair_id",
    "canonical_row_id",
    "canonical_status",
    "intervention_contract_status",
    "integrity_status",
    "schema_status",
    "dataset_source_status",
    "grammar_status",
    "polarity_contamination_status",
    "time_swap_status",
    "reason_codes",
    "source_dataset_path",
    "source_dataset_sha256",
    "frame_compatible_label",
}
STAGE185_FIELDS = {
    "intervention_type",
    "eligible_for_positive_margin",
    "family_contract_id",
    "rule_version",
    "generator_source_sha256",
    "integrity_builder_sha256",
    "created_at",
}

def require(condition, message):
    if not condition:
        raise AssertionError(message)

repo_root = pathlib.Path.cwd().resolve(strict=True)

def repo_path(relative, *, expect_dir=False):
    candidate = pathlib.PurePosixPath(relative)
    require(not candidate.is_absolute(), f"ABSOLUTE_REPO_RELATIVE_PATH:{relative}")
    require(".." not in candidate.parts, f"PARENT_TRAVERSAL_REPO_PATH:{relative}")
    current = repo_root
    for part in candidate.parts:
        current = current / part
        require(current.exists() or current.is_symlink(), f"PATH_COMPONENT_MISSING:{relative}:{part}")
        require(not current.is_symlink(), f"SYMLINK_COMPONENT_FORBIDDEN:{relative}:{part}")
    raw = current
    resolved = raw.resolve(strict=True)
    require(resolved == repo_root or repo_root in resolved.parents, f"PATH_OUTSIDE_REPO:{relative}")
    if expect_dir:
        require(resolved.is_dir(), f"REQUIRED_DIRECTORY_MISSING:{relative}")
    else:
        require(resolved.is_file(), f"REQUIRED_FILE_MISSING:{relative}")
    require(not resolved.is_symlink(), f"RESOLVED_SYMLINK_FORBIDDEN:{relative}")
    return resolved

def require_no_p4l_staging_sibling(output_dir):
    for sibling in output_dir.parent.iterdir():
        if sibling.name.startswith(".p4l-staging-"):
            raise AssertionError(f"LEFTOVER_STAGING_DIRECTORY:{sibling.name}")

def require_no_attempt_backup_sibling(output_dir):
    lowered_needles = ("p4q", "backup", "move-aside", "moved")
    for sibling in output_dir.parent.iterdir():
        name = sibling.name.lower()
        if "p4q" in name and any(needle in name for needle in lowered_needles[1:]):
            raise AssertionError(f"P4Q_BACKUP_OR_MOVE_ASIDE_ARTIFACT:{sibling.name}")

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def reject_constant(token):
    raise ValueError(f"NON_FINITE_JSON_CONSTANT:{token}")

def decode_utf8_no_bom_lf_final_newline(path, *, jsonl):
    data = path.read_bytes()
    require(not data.startswith(b"\xef\xbb\xbf"), f"{path}: UTF-8 BOM forbidden")
    require(b"\r\n" not in data and b"\r" not in data, f"{path}: CR/CRLF forbidden")
    require(data.endswith(b"\n"), f"{path}: final newline required")
    text = data.decode("utf-8")
    if jsonl:
        require(text.strip() != "", f"{path}: empty JSONL forbidden")
    return text, data

def reject_non_json_numbers(value, where):
    if isinstance(value, float):
        require(value == value and value not in (float("inf"), float("-inf")), where)
    elif isinstance(value, dict):
        for key, item in value.items():
            reject_non_json_numbers(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            reject_non_json_numbers(item, f"{where}[{i}]")

def read_jsonl_objects(path):
    text, _ = decode_utf8_no_bom_lf_final_newline(path, jsonl=True)
    rows = []
    for index, line in enumerate(text.splitlines(), start=1):
        require(line != "", f"{path}: empty line {index}")
        obj = json.loads(line, parse_constant=reject_constant)
        require(isinstance(obj, dict), f"{path}: line {index} not object")
        reject_non_json_numbers(obj, f"{path}:{index}")
        rows.append(obj)
    return rows

def read_json_object(path):
    text, data = decode_utf8_no_bom_lf_final_newline(path, jsonl=False)
    decoder = json.JSONDecoder(parse_constant=reject_constant)
    obj, end = decoder.raw_decode(text)
    require(text[end:].strip() == "", f"{path}: trailing data after JSON object")
    require(isinstance(obj, dict), f"{path}: JSON root not object")
    reject_non_json_numbers(obj, str(path))
    reserialized = (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    require(reserialized == data, f"{path}: deterministic provenance serialization mismatch")
    return obj, data

def canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def source_semantic_sha(rows):
    payload = [{field: row[field] for field in SOURCE_SEMANTIC_FIELDS} for row in rows]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def sidecar_semantic_sha(rows):
    payload = []
    for row in rows:
        payload.append({key: row[key] for key in sorted(row) if key != "created_at"})
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def git_stdout(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def git_clean():
    subprocess.check_call(["git", "diff", "--quiet"])
    subprocess.check_call(["git", "diff", "--cached", "--quiet"])

head = git_stdout("rev-parse", "HEAD")
require(head == EXPECTED_HEAD, "HEAD_MISMATCH")
git_clean()
OUTPUT_DIR = repo_path(OUTPUT_DIR_REL, expect_dir=True)
SIDECAR_PATH = repo_path(f"{OUTPUT_DIR_REL}/{SIDECAR_NAME}")
PROVENANCE_PATH = repo_path(f"{OUTPUT_DIR_REL}/{PROVENANCE_NAME}")
BUILDER_PATH = repo_path(BUILDER_REL)
SOURCE_DATASET_PATH = repo_path(SOURCE_DATASET_REL)
P4B_ROWS_PATH = repo_path(P4B_ROWS_REL)
P4B_SUMMARY_PATH = repo_path(P4B_SUMMARY_REL)
P4B_PROVENANCE_PATH = repo_path(P4B_PROVENANCE_REL)
require(sha256_file(BUILDER_PATH) == EXPECTED_BUILDER_SHA, "BUILDER_SHA_MISMATCH")
require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"), "GPU_NOT_DISABLED")

gate1_stdout_path = os.environ.get("P4Q_GATE1_STDOUT_PATH", "")
require(gate1_stdout_path.strip(), "MISSING_GATE1_STDOUT_PATH")
gate1_raw = pathlib.Path(gate1_stdout_path).read_bytes().decode("utf-8")
require(gate1_raw.strip(), "EMPTY_GATE1_STDOUT")
decoder = json.JSONDecoder(parse_constant=reject_constant)
gate1, gate1_end = decoder.raw_decode(gate1_raw)
require(gate1_raw[gate1_end:].strip() == "", "GATE1_STDOUT_TRAILING_DATA")
require(isinstance(gate1, dict), "GATE1_STDOUT_NOT_OBJECT")
require(gate1.get("status") == "PASS", "GATE1_STATUS_NOT_PASS")
require(gate1.get("row_count") == EXPECTED_ROW_COUNT, "GATE1_ROW_COUNT_MISMATCH")
require(gate1.get("artifact_materialization_requested") is True, "GATE1_MATERIALIZE_NOT_TRUE")
require(gate1.get("publish_status") == "PUBLISHED", "GATE1_NOT_PUBLISHED")
require(gate1.get("training_authorized") is False, "GATE1_TRAINING_AUTHORIZED")
require(gate1.get("evaluation_authorized") is False, "GATE1_EVALUATION_AUTHORIZED")
for key in ("sidecar_physical_sha256", "sidecar_semantic_sha256", "provenance_physical_sha256"):
    value = gate1.get(key)
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"GATE1_HASH_INVALID:{key}")

entries = sorted(p.name for p in OUTPUT_DIR.iterdir())
require(entries == [PROVENANCE_NAME, SIDECAR_NAME], "OUTPUT_DIR_ENTRIES_MISMATCH")
require_no_p4l_staging_sibling(OUTPUT_DIR)
require_no_attempt_backup_sibling(OUTPUT_DIR)

source_physical = sha256_file(SOURCE_DATASET_PATH)
require(source_physical == SOURCE_PHYSICAL_SHA, "SOURCE_PHYSICAL_SHA_MISMATCH")
source_rows = read_jsonl_objects(SOURCE_DATASET_PATH)
require(len(source_rows) == EXPECTED_ROW_COUNT, "SOURCE_ROW_COUNT_MISMATCH")
require(source_semantic_sha(source_rows) == SOURCE_SEMANTIC_SHA, "SOURCE_SEMANTIC_SHA_MISMATCH")

sidecar_rows = read_jsonl_objects(SIDECAR_PATH)
require(len(sidecar_rows) == EXPECTED_ROW_COUNT, "SIDECAR_ROW_COUNT_MISMATCH")
provenance, provenance_bytes = read_json_object(PROVENANCE_PATH)

source_ids = [row["id"] for row in source_rows]
sidecar_ids = [row.get("row_id") for row in sidecar_rows]
require(sidecar_ids == source_ids, "SIDECAR_SOURCE_ORDER_MISMATCH")
require(all(isinstance(row_id, str) and row_id for row_id in sidecar_ids), "ROW_ID_EMPTY_OR_NONSTRING")
require(len(set(sidecar_ids)) == EXPECTED_ROW_COUNT, "ROW_ID_DUPLICATE")
source_by_id = {row["id"]: row for row in source_rows}
sidecar_by_id = {row["row_id"]: row for row in sidecar_rows}

# Exact replay is intentionally bound to builder/P4-L semantics, including
# Python round() behavior in the dev_count expression.
pairs = sorted({str(row["pair_id"]) for row in source_rows})
require(pairs, "NO_PAIR_IDS")
shuffled = list(pairs)
random.Random(174).shuffle(shuffled)
dev_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * 0.2)))
dev_pairs = set(shuffled[:dev_count])
expected_split_by_pair = {pair_id: ("dev" if pair_id in dev_pairs else "train") for pair_id in pairs}

canonical_by_pair = {}
for row in source_rows:
    pair_id = str(row["pair_id"])
    if str(row["intervention_type"]) == "none":
        require(pair_id not in canonical_by_pair, f"MULTIPLE_CANONICAL_SOURCE_ROWS:{pair_id}")
        canonical_by_pair[pair_id] = str(row["id"])
require(set(canonical_by_pair) == set(pairs), "CANONICAL_PAIR_SET_MISMATCH")

integrity_counts = collections.Counter()
eligible_counts = collections.Counter()
reason_supervision_counts = collections.Counter()
unresolved = []

for index, row in enumerate(sidecar_rows):
    source = source_rows[index]
    missing = (P2_REQUIRED_FIELDS | STAGE185_FIELDS) - set(row)
    require(not missing, f"SIDECAR_REQUIRED_FIELDS_MISSING:{row.get('row_id')}:{sorted(missing)}")
    if "source_order_index" in row:
        require(row["source_order_index"] == index, f"SOURCE_ORDER_INDEX_MISMATCH:{row['row_id']}")
    require(row["source_dataset_path"] == SOURCE_DATASET_REL, f"SOURCE_PATH_MISMATCH:{row['row_id']}")
    require(row["source_dataset_sha256"] == SOURCE_PHYSICAL_SHA, f"SOURCE_SHA_MISMATCH:{row['row_id']}")
    require(row.get("source_dataset_semantic_sha256", SOURCE_SEMANTIC_SHA) == SOURCE_SEMANTIC_SHA, f"SOURCE_SEMANTIC_SHA_MISMATCH:{row['row_id']}")
    require(row.get("historical_stage185_used_as_current_source_identity") is not True, f"HISTORICAL_CURRENT_IDENTITY:{row['row_id']}")
    require(row["source_dataset_sha256"] != HISTORICAL_SOURCE_SHA, f"HISTORICAL_SHA_AS_CURRENT:{row['row_id']}")
    require(type(row["frame_compatible_label"]) is int and row["frame_compatible_label"] in (0, 1), f"FRAME_LABEL_TYPE:{row['row_id']}")
    require(isinstance(row["reason_codes"], list), f"REASON_CODES_NOT_LIST:{row['row_id']}")
    require(row["reason_codes"] == sorted(row["reason_codes"]) and len(row["reason_codes"]) == len(set(row["reason_codes"])), f"REASON_CODES_NOT_SORTED_UNIQUE:{row['row_id']}")
    require(type(row["eligible_for_positive_margin"]) is bool, f"ELIGIBLE_TYPE:{row['row_id']}")
    expected_split = expected_split_by_pair[str(source["pair_id"])]
    require(row["split"] == expected_split, f"SPLIT_MISMATCH:{row['row_id']}")
    expected_canonical = canonical_by_pair[str(source["pair_id"])]
    require(row["canonical_row_id"] == expected_canonical, f"CANONICAL_ID_MISMATCH:{row['row_id']}")
    target = sidecar_by_id.get(expected_canonical)
    require(target is not None, f"CANONICAL_TARGET_MISSING:{row['row_id']}")
    require(target["pair_id"] == row["pair_id"], f"CANONICAL_PAIR_MISMATCH:{row['row_id']}")
    require(target["split"] == row["split"], f"CANONICAL_SPLIT_MISMATCH:{row['row_id']}")
    require(target["row_id"] == target["canonical_row_id"], f"CANONICAL_NOT_SELF_ANCHORED:{target['row_id']}")
    if source["frame_compatible_label"] == 0:
        expected_reason = "FRAME"
    elif source["predicate_covered_label"] == 0:
        expected_reason = "PREDICATE"
    elif source["sufficiency_label"] == 0:
        expected_reason = "SUFFICIENCY"
    else:
        expected_reason = "AUTHORIZED"
    if "p2_primary_reason" in row:
        require(row["p2_primary_reason"] == expected_reason, f"PRIMARY_REASON_MISMATCH:{row['row_id']}")
    expected_margin = (
        row["integrity_status"] == "ELIGIBLE"
        and row["split"] == "train"
        and row["frame_compatible_label"] == 1
        and row["time_swap_status"] == "PASS"
        and row["dataset_source_status"] == "PASS"
    )
    require(row["eligible_for_positive_margin"] is expected_margin, f"POSITIVE_MARGIN_MISMATCH:{row['row_id']}")
    integrity_counts[row["integrity_status"]] += 1
    eligible_counts[str(row["eligible_for_positive_margin"]).lower()] += 1
    if "p2_reason_supervision_eligible" in row:
        reason_supervision_counts[str(row["p2_reason_supervision_eligible"]).lower()] += 1
    if row["integrity_status"] == "UNRESOLVED" or any(row.get(field) == "UNRESOLVED" for field in ("canonical_status", "intervention_contract_status", "schema_status", "dataset_source_status", "grammar_status", "polarity_contamination_status", "time_swap_status")):
        unresolved.append(row)

require(sha256_file(P4B_ROWS_PATH) == P4B_ROWS_SHA, "P4B_ROWS_SHA_MISMATCH")
require(sha256_file(P4B_SUMMARY_PATH) == P4B_SUMMARY_SHA, "P4B_SUMMARY_SHA_MISMATCH")
require(sha256_file(P4B_PROVENANCE_PATH) == P4B_PROVENANCE_SHA, "P4B_PROVENANCE_SHA_MISMATCH")
p4b_rows = read_jsonl_objects(P4B_ROWS_PATH)
p4b_pairs = {row["pair_id"] for row in p4b_rows}
p4b_members = {row["member_id"] for row in p4b_rows if "member_id" in row}
if not p4b_members:
    p4b_members = {row["row_id"] for row in p4b_rows if "row_id" in row}
require(len(p4b_pairs) == 119, "P4B_PAIR_SCOPE_MISMATCH")
require(len(p4b_members) == 357, "P4B_MEMBER_SCOPE_MISMATCH")

sidecar_physical = sha256_file(SIDECAR_PATH)
sidecar_semantic = sidecar_semantic_sha(sidecar_rows)
provenance_physical = hashlib.sha256(provenance_bytes).hexdigest()
require(sidecar_physical == gate1["sidecar_physical_sha256"], "GATE1_SIDECAR_PHYSICAL_MISMATCH")
require(sidecar_semantic == gate1["sidecar_semantic_sha256"], "GATE1_SIDECAR_SEMANTIC_MISMATCH")
require(provenance_physical == gate1["provenance_physical_sha256"], "GATE1_PROVENANCE_PHYSICAL_MISMATCH")
require(provenance.get("sidecar_physical_sha256") == sidecar_physical, "PROVENANCE_SIDECAR_PHYSICAL_MISMATCH")
require(provenance.get("sidecar_semantic_sha256") == sidecar_semantic, "PROVENANCE_SIDECAR_SEMANTIC_MISMATCH")

required_provenance = {
    "schema_version": "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1",
    "authority_version": "P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1",
    "p4l_authority_commit": "80cb034792f03226cf6e22c196c1229ed4e6dd62",
    "builder_source_path": BUILDER_REL,
    "builder_source_commit": EXPECTED_HEAD,
    "builder_source_sha256": EXPECTED_BUILDER_SHA,
    "source_dataset_path": SOURCE_DATASET_REL,
    "source_dataset_sha256": SOURCE_PHYSICAL_SHA,
    "source_dataset_semantic_sha256": SOURCE_SEMANTIC_SHA,
    "sidecar_path": f"{OUTPUT_DIR_REL}/{SIDECAR_NAME}",
    "row_count": EXPECTED_ROW_COUNT,
    "one_to_one_row_coverage": True,
    "unique_row_id": True,
    "p4b_compatibility_rows_path": P4B_ROWS_REL,
    "p4b_compatibility_rows_sha256": P4B_ROWS_SHA,
    "p4b_compatibility_summary_path": P4B_SUMMARY_REL,
    "p4b_compatibility_summary_sha256": P4B_SUMMARY_SHA,
    "p4b_compatibility_provenance_path": P4B_PROVENANCE_REL,
    "p4b_compatibility_provenance_sha256": P4B_PROVENANCE_SHA,
    "historical_stage185_used_as_current_source_identity": False,
    "training_admission_released": False,
    "a0_execution_authorized": False,
    "training_authorized": False,
    "evaluation_authorized": False,
    "gpu_authorized": False,
}
for key, expected in required_provenance.items():
    require(provenance.get(key) == expected, f"PROVENANCE_FIELD_MISMATCH:{key}")
require(provenance.get("blockers") == [], "PROVENANCE_BLOCKERS_NOT_EMPTY")
require(provenance.get("failure_reasons") == [], "PROVENANCE_FAILURE_REASONS_NOT_EMPTY")
scope = provenance.get("p4b_compatibility_authorized_scope", {})
require(scope.get("pair_count") == 119 and scope.get("member_count") == 357, "PROVENANCE_P4B_SCOPE_MISMATCH")
if "artifact_materialization_authorized_by_p4l" in provenance:
    require(provenance["artifact_materialization_authorized_by_p4l"] is False, "P4L_MATERIALIZATION_FLAG_UNEXPECTED")
require(provenance.get("provenance_physical_sha256_self_certified") is False, "PROVENANCE_SELF_CERTIFICATION_FLAG_NOT_FALSE")

post_head = git_stdout("rev-parse", "HEAD")
require(post_head == EXPECTED_HEAD, "POST_HEAD_MISMATCH")
git_clean()
require_no_p4l_staging_sibling(OUTPUT_DIR)
require_no_attempt_backup_sibling(OUTPUT_DIR)
require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"), "GPU_NOT_DISABLED_POST_GATE2")

observations = {
    "python_version": sys.version,
    "sys_platform": sys.platform,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "sidecar_physical_sha256": sidecar_physical,
    "sidecar_semantic_sha256": sidecar_semantic,
    "provenance_physical_sha256": provenance_physical,
    "integrity_status_counts": dict(sorted(integrity_counts.items())),
    "eligible_for_positive_margin_counts": dict(sorted(eligible_counts.items())),
    "p2_reason_supervision_eligible_counts": dict(sorted(reason_supervision_counts.items())),
    "unresolved_row_count": len(unresolved),
    "unresolved_status_summary": dict(sorted(collections.Counter(row["integrity_status"] for row in unresolved).items())),
}
print(json.dumps(observations, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
print(PASS_TOKEN)
PY

test "$(git rev-parse HEAD)" = "${EXPECTED_HEAD}" || fail "POST_GATE2_HEAD_MISMATCH"
git diff --quiet || fail "TRACKED_WORKTREE_DIRTY_AFTER_GATE2"
git diff --cached --quiet || fail "INDEX_DIRTY_AFTER_GATE2"
test -d "${OUTPUT_DIR}" || fail "CANONICAL_OUTPUT_DIR_MISSING_AFTER_GATE2"
test ! -L "${OUTPUT_DIR}" || fail "CANONICAL_OUTPUT_DIR_SYMLINK_AFTER_GATE2"
test -f "${OUTPUT_DIR}/${SIDECAR_NAME}" || fail "CANONICAL_SIDECAR_MISSING_AFTER_GATE2"
test ! -L "${OUTPUT_DIR}/${SIDECAR_NAME}" || fail "CANONICAL_SIDECAR_SYMLINK_AFTER_GATE2"
test -f "${OUTPUT_DIR}/${PROVENANCE_NAME}" || fail "CANONICAL_PROVENANCE_MISSING_AFTER_GATE2"
test ! -L "${OUTPUT_DIR}/${PROVENANCE_NAME}" || fail "CANONICAL_PROVENANCE_SYMLINK_AFTER_GATE2"
test "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | tr '\n' ' ')" = "${PROVENANCE_NAME} ${SIDECAR_NAME} " || fail "CANONICAL_OUTPUT_ENTRIES_MISMATCH_AFTER_GATE2"
test -z "$(find "$(dirname "${OUTPUT_DIR}")" -maxdepth 1 -name '.p4l-staging-*' -print -quit)" || fail "LEFTOVER_P4L_STAGING_SIBLING_AFTER_GATE2"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|-1)
    ;;
  *)
    fail "GPU_NOT_DISABLED_AFTER_GATE2"
    ;;
esac
```

No retry, cleanup, deletion, replacement, or second builder invocation is
authorized. If Gate 1 exits nonzero, Gate 2 must not run, canonical output must
not be deleted or altered, and the wrapper stops with Gate-1 failure status.

Gate 2 PASS requires the final printed line to be exactly:

`P3W6F2P4Q_CURRENT_LINEAGE_SIDECAR_ARTIFACT_VALIDATION_PASS`

## I. Serialization Checks

Gate 2 validates:

- canonical directory is a normal directory and not a symlink
- canonical directory contains exactly the two expected entries
- no staging, backup, or temp artifact remains in the canonical directory
- no sibling `.p4l-staging-*` remains
- sidecar is a regular file and not a symlink
- provenance is a regular file and not a symlink
- sidecar and provenance are UTF-8, no BOM, LF-only, no CRLF, and have final
  newlines
- sidecar has exactly 3600 non-empty JSONL objects
- every JSONL line parses to a JSON object
- provenance parses to one JSON object
- no NaN, Infinity, or non-JSON value appears
- provenance deterministic JSON form is checked where frozen by P4-L and this
  authority
- provenance exact serialized bytes must equal
  `json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False,
  allow_nan=False) + "\n"` encoded as UTF-8
- Python `json.loads` default NaN/Infinity acceptance is not relied upon;
  parsing uses explicit non-finite rejection and recursively rejects
  non-finite floats after parsing

## J. Source Physical And Semantic Identity Checks

Gate 2 recomputes source dataset physical SHA256 and requires:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Gate 2 recomputes source semantic SHA256 using ordered source rows, fields
exactly:

- `id`
- `pair_id`
- `claim`
- `evidence`
- `final_label`
- `frame_compatible_label`
- `predicate_covered_label`
- `sufficiency_label`
- `polarity_label`
- `primary_failure_type`
- `intervention_type`

with JSON `sort_keys = true`, `separators = (",", ":")`, and
`ensure_ascii = false`, and requires:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

Source row count must be exactly `3600`.

## K. Row Universe, Order, And Schema Checks

Gate 2 requires:

- sidecar row count equals source row count equals `3600`
- `[sidecar.row_id]` exactly equals `[source.id]` in physical source order
- unique, non-empty `row_id`
- no missing, extra, or duplicate rows
- `source_order_index`, where emitted, equals `0..3599` exactly
- every sidecar row contains all frozen P2 required fields
- every sidecar row contains all frozen Stage185-compatible fields

P2 required fields:

- `row_id`
- `split`
- `pair_id`
- `canonical_row_id`
- `canonical_status`
- `intervention_contract_status`
- `integrity_status`
- `schema_status`
- `dataset_source_status`
- `grammar_status`
- `polarity_contamination_status`
- `time_swap_status`
- `reason_codes`
- `source_dataset_path`
- `source_dataset_sha256`
- `frame_compatible_label`

Stage185-compatible fields:

- `intervention_type`
- `eligible_for_positive_margin`
- `family_contract_id`
- `rule_version`
- `generator_source_sha256`
- `integrity_builder_sha256`
- `created_at`

## L. Per-Row Identity And Type Checks

For every sidecar row, Gate 2 requires:

- `source_dataset_path` equals
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
- `source_dataset_sha256` equals
  `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
- `source_dataset_semantic_sha256`, when emitted, equals
  `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- `historical_stage185_used_as_current_source_identity` is not true, and when
  emitted is false
- historical dataset SHA
  `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
  never appears as current `source_dataset_sha256`
- `frame_compatible_label` is exact integer `0` or `1`; bool is rejected
- `reason_codes` is a sorted unique JSON array
- `eligible_for_positive_margin` is exact JSON boolean

## M. Split And Canonical Replay Checks

Gate 2 independently recomputes pair-level split:

- sorted pair IDs
- `random.Random(174).shuffle`
- dev ratio `0.2`
- exact dev-count rule:
  `dev_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * 0.2)))`
- dev IDs are exactly `set(shuffled[:dev_count])`
- each sorted pair ID maps to `"dev"` if it is in that set, otherwise
  `"train"`
- pair-level assignment

Exact replay is intentionally bound to builder/P4-L semantics, including
Python `round()` behavior. Floor/int behavior and numpy are not authorized.

Every sidecar `split` must match.

Gate 2 independently derives the canonical row per pair as the same-pair
source row with `intervention_type == "none"`, and requires exactly one such
row per pair.

Every sidecar `canonical_row_id` must match that row.

Canonical target requirements:

- target exists
- target has same pair
- target has same split
- target self-anchors with `row_id == canonical_row_id`

## N. Reason And Eligibility Checks

Gate 2 independently derives primary reason from source axes:

- `frame_compatible_label == 0 -> FRAME`
- else `predicate_covered_label == 0 -> PREDICATE`
- else `sufficiency_label == 0 -> SUFFICIENCY`
- else `AUTHORIZED`

If emitted, `p2_primary_reason` must match.

Positive-margin eligibility must equal exactly:

- `integrity_status == "ELIGIBLE"`
- and `split == "train"`
- and `frame_compatible_label == 1`
- and `time_swap_status == "PASS"`
- and `dataset_source_status == "PASS"`

Gate 2 reports counts for:

- `integrity_status` values
- `eligible_for_positive_margin` true/false
- `p2_reason_supervision_eligible` true/false where present
- unresolved rows and associated reason/status summaries

P4-Q does not require unresolved count `0` unless frozen P4-L is later
superseded to require zero.

## O. P4-B Scoped Compatibility Checks

Gate 2 verifies frozen P4-B compatibility input SHAs:

- rows:
  `59e4367d29e3c49152049e4a1b46e8783d5b81d1ebaa931ca9fd4ae0ac967b9f`
- summary:
  `ce618d214dc4d660706d927a0d91ec5945d3a0edbffbf615eab8b9c9ff585aa8`
- provenance:
  `09a56f1dca325e0749a0e4b0f822d68dad5d85fd90fa427e0efd6d617d41b2d6`

Gate 2 requires compatibility scope remains `119` pairs / `357` members.

P4-Q must not infer full-3600 compatibility from those scoped artifacts.

## P. Sidecar Hash Checks

Gate 2 recomputes sidecar physical SHA256 from exact bytes and requires it
equals:

- Gate 1 reported `sidecar_physical_sha256`
- provenance `sidecar_physical_sha256`

Gate 2 recomputes semantic SHA256 exactly per P4-L:

- ordered sidecar rows
- remove `created_at` only
- remove no other field
- JSON `sort_keys = true`
- `separators = (",", ":")`
- `ensure_ascii = false`
- hash canonical UTF-8 bytes

It requires this semantic hash equals:

- Gate 1 reported `sidecar_semantic_sha256`
- provenance `sidecar_semantic_sha256`

No expected sidecar hash is predicted before execution.

## Q. Provenance Manifest Checks

Gate 2 requires exact provenance values:

- `schema_version =
  P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_SIDECAR_PROVENANCE_V1`
- `authority_version =
  P3W6F2P4L_CURRENT_LINEAGE_INTEGRITY_ARTIFACT_CONTRACT_AUTHORITY_V1`
- `p4l_authority_commit =
  80cb034792f03226cf6e22c196c1229ed4e6dd62`
- `builder_source_path =
  scripts/build_reason_router_p3w6f2_p4l_current_lineage_integrity_sidecar.py`
- `builder_source_commit =
  2f9e6076791358922e3ebd70e89533d9cb83b458`
- `builder_source_sha256 =
  b7fad3672428c8199347c278be0521e8355a8ca1c1491b5d3036e6cdc45f3f6d`
- source dataset path, physical hash, and semantic hash exact
- sidecar path exact canonical path
- `row_count == 3600`
- `one_to_one_row_coverage == true`
- `unique_row_id == true`
- P4-B paths and SHAs exact
- P4-B authorized scope is `119` pairs / `357` members
- `historical_stage185_used_as_current_source_identity == false`
- `blockers == []`
- `failure_reasons == []`
- `training_admission_released == false`
- `a0_execution_authorized == false`
- `training_authorized == false`
- `evaluation_authorized == false`
- `gpu_authorized == false`

Do not treat `artifact_materialization_authorized_by_p4l == false` as a
contradiction. P4-L itself did not authorize materialization; P4-Q is the later
execution authority.

Gate 2 recomputes provenance physical SHA256 externally as:

`external_provenance_physical_sha256 = SHA256(exact provenance file bytes)`

It requires:

`external_provenance_physical_sha256 == Gate1.stdout["provenance_physical_sha256"]`

The manifest must contain:

`provenance_physical_sha256_self_certified == false`

The manifest is not required to contain `provenance_physical_sha256`; absence
of a provenance self-hash is not failure. This external comparison is the
complete P4-Q provenance physical-hash check. P4-Q must not insert a physical
SHA into provenance or reconstruct provenance with a recursive self-hash.

## R. Builder Output Identity

Because Gate 1 runs exact implementation HEAD with no `--builder-commit`,
P4-Q requires:

`provenance.builder_source_commit == git rev-parse HEAD == 2f9e6076791358922e3ebd70e89533d9cb83b458`

No relabeling is permitted.

## S. PASS/BLOCKED/FAIL Mapping

P4-Q PASS requires both:

- Gate 1 canonical materialization PASS
- Gate 2 artifact/provenance validation PASS

P4-Q is BLOCKED for:

- exact state or identity mismatch
- output path pre-existing before authorized run
- tooling/environment inability without evidence of implementation/artifact
  defect
- inability to perform required validation safely
- missing or unusable Gate 1 stdout capture before Gate 2 when it prevents
  safe comparison and no artifact defect is evidenced

P4-Q is FAIL for:

- builder semantic/build failure
- publication failure attributable to implementation
- artifact content, hash, schema, or provenance mismatch
- source identity/order mismatch
- overwrite/no-clobber violation
- validator assertion failure indicating artifact or implementation defect

## T. Execution And Global Postconditions

After both gates, require:

- `git rev-parse HEAD ==
  2f9e6076791358922e3ebd70e89533d9cb83b458`
- tracked worktree clean
- index clean
- canonical output directory exists
- canonical output directory contains exactly the two expected files
- no `.p4l-staging-*` sibling remains
- GPU off before and after

Execution evidence must record:

- Python version
- `sys.platform`
- `CUDA_VISIBLE_DEVICES` observation
- Gate 1 stdout
- Gate 1 exit code
- Gate 2 validation output/token
- observed sidecar physical SHA
- observed sidecar semantic SHA
- observed provenance physical SHA
- status, eligibility, and unresolved counts

## U. Interpretation Boundary

On full P4-Q PASS:

1. code correctness: PASS from prior independent static verification
2. runtime publication: PASS from P4-P
3. canonical artifact materialization: PASS
4. canonical artifact/provenance integrity: ESTABLISHED for frozen P4-L
   contract
5. trainer rebind readiness: NOT YET AUTHORIZED
6. A0 execution: NOT AUTHORIZED
7. scientific conclusion: NOT ESTABLISHED

Do not claim scientific efficacy from artifact validity.

## V. Authority Flags

Candidate-time flags are all false:

- `artifact_materialization_authorized = false`
- `artifact_validation_execution_authorized = false`
- `kaggle_authorized = false`
- `cpu_authorized = false`
- `gpu_authorized = false`
- `trainer_rebind_authorized = false`
- `a0_execution_authorized = false`
- `training_admission_released = false`
- `training_authorized = false`
- `evaluation_authorized = false`

After independent verification plus P4-Q freeze, only the exact P4-Q execution
may have:

- `artifact_materialization_authorized = true`
- `artifact_validation_execution_authorized = true`
- `kaggle_authorized = true`
- `cpu_authorized = true`
- `gpu_authorized = false`

All trainer-rebind, A0, training, and evaluation flags remain false.

## W. Collection And Evidence Boundary

The future canonical artifact must ultimately be transferred back from Kaggle
with provenance preserved.

After successful run, authorized transport/collection may package the
canonical directory and execution evidence without modifying canonical artifact
contents.

Transport does not establish a new scientific claim.

Do not create a handoff or ZIP now. Do not invent a run ID now. Do not predict
artifact hashes now.

Future deterministic execution evidence namespace:

`reports/reason_router_p2_p3w6f2_p4q_current_lineage_sidecar_materialization_validation_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/`

This namespace must not be created during candidate-authority creation.

Future P4-Q execution evidence must record:

- P4-Q authority freeze commit
- exact implementation commit
- builder SHA
- exact Gate 1 command
- Gate 1 stdout/exit
- exact Gate 2 validator command
- Gate 2 output/token
- environment versions/platform/GPU observations
- pre/post HEAD and tracked/index state
- pre-output absence/post-output presence
- all three observed artifact hashes
- row/status/eligibility/unresolved summaries
- blockers/failure reasons

## X. Scientific Boundary

P4-Q makes no change to:

- Conditional First-Blocker Reason Router
- Reason-Specific Supervision
- Explicit Gradient Ownership
- `FRAME > PREDICATE > SUFFICIENCY > AUTHORIZED`
- secondary reasons diagnostic-only
- router-only final 3-way CE
- detached F/P/S and polarity CE path
- EMA observer/baseline-only
- A0-A3
- E0

No trainer, model, config, checkpoint, dataset, split, label, manifest, or
promotion-criteria edit is authorized.

## Y. Independent Verification Requirement

Because P4-Q permits future canonical provenance-bearing artifact creation
after freeze, independent static verification of this candidate is required
before freeze.

The verifier must specifically inspect the embedded Gate 2 validator for:

- path safety
- hash algorithms
- source-order algorithm
- split/canonical replay
- semantic-hash algorithm
- provenance interpretation
- no accidental mutation/materialization beyond Gate 1
- no trainer/A0/training widening

Validation PASS may not be reported unless the future Gate 2 command actually
runs successfully under frozen P4-Q authority.

## Z. Stop Conditions

Candidate creation is BLOCKED if any of the following holds:

- namespace collision
- authority conflict
- current state mismatch
- exact validator cannot be specified from frozen contracts
- validation would require modifying builder/trainer/source artifacts
- execution would need GPU
- artifact path/provenance identity cannot be bound to exact
  `2f9e6076791358922e3ebd70e89533d9cb83b458`

## AA. Candidate Path

Candidate path:

`reports/reason_router_p2_p3w6f2_p4q_current_lineage_sidecar_materialization_validation_execution_authority_spec.md`

Candidate SHA256 is computed after file creation and is not predicted inside
this specification body.

Final candidate readiness token:

`P3W6F2P4Q_CURRENT_LINEAGE_SIDECAR_MATERIALIZATION_VALIDATION_EXECUTION_AUTHORITY_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
