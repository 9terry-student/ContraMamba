# P3-W7 Seed8192 A0 Seed180 R2 Provenance Recovery-Handoff Command Freeze Candidate

## Status and authority

**Execution status: NOT YET PERFORMED.**  This report freezes the exact later
CPU-only Kaggle recovery command under activated authority commit
`48d258fc5e8b09e163e9252f33c86936244ef872`.  It does not authorize or perform
recovery execution, Kaggle execution, real import, training, evaluation,
checkpoint loading, model/trainer import, commit, or push.

The command uses only Python's standard library.  It neither reads nor mutates
`start.marker`; makes no mtime-, glob-, or recursive-artifact-based selection;
does not reuse a collector or ZIP; and copies neither source evidence nor
wrapper evidence except into a newly created staging directory after the
specified SHA256 checks pass.

## Frozen command

The payload below is serialized as UTF-8 **without BOM** with exactly one final
LF.  Its byte count and SHA256 are recorded after independent recomputation in
the verification record below.

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
python3 - <<'PY'
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

RUN_NAME = "p3w7-seed8192-a0-seed180-r2"
EXECUTION_COMMIT = "abd85a088c274678004432160625d42208112848"
COMMAND_SHA256 = "82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4"
RUN_LOG_SHA256 = "a0d4b015cc77e8060184a5035333fe46e101f52d931af3101950382e24407f4e"
RUN_META_SHA256 = "23989109951bdf2b3fdfc17f9a7739c34e2875131bc76c0ccb1e313858fcdc27"
STARTED_UTC = "2026-09-08T22:38:29Z"
FINISHED_UTC = "2026-09-08T22:41:44Z"
EXIT_CODE = 0
REL_SOURCE_DIR = Path("reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0")
ARTIFACTS = (
    ("training_report.json", "146b7330f6cf479bd339b2eb0af886d5eda3eed72589f57c0879bb5f6d9f5d9c"),
    ("training_report_predictions.jsonl", "80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334"),
    ("clean_dev_predictions.json", "5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d"),
    ("run_provenance.json", "057237823c0b56a907b051b1d4018eb9317aacd860f986083aaa2277307ffbbf"),
    ("selected_checkpoint.pt", "0724f5a2e537c932f6692dd74713d57fc70f8182ee2719c33665b09114bd944a"),
)
OUTPUT = Path("/kaggle/working/contramamba_handoffs/p3w7-seed8192-a0-seed180-r2_abd85a088c274.zip")
LOG_ROOT = Path("/kaggle/working/contramamba_run_logs")
WRAPPERS = (
    (LOG_ROOT / (RUN_NAME + "_abd85a0.command.sh"), "command.sh", COMMAND_SHA256),
    (LOG_ROOT / (RUN_NAME + "_abd85a0.log"), "run.log", RUN_LOG_SHA256),
    (LOG_ROOT / (RUN_NAME + "_abd85a0.meta"), "run.meta", RUN_META_SHA256),
)

def stop(message):
    raise SystemExit("FAIL_CLOSED: " + message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def regular_file(path, label):
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        stop(label + " is missing: " + str(path))
    if not stat.S_ISREG(mode):
        stop(label + " is not a regular file: " + str(path))

def require_sha256(path, expected, label):
    regular_file(path, label)
    actual = sha256(path)
    if actual != expected:
        stop(label + " SHA256 mismatch: " + str(path))

def ancestors(path):
    path = path.resolve()
    yield path
    yield from path.parents

def candidate_roots():
    candidates = set()
    configured = os.environ.get("CONTRAMAMBA_REPO_ROOT")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            stop("CONTRAMAMBA_REPO_ROOT is not absolute")
        candidates.add(candidate.resolve())
    for base in (Path.cwd(), Path("/kaggle/working")):
        if base.exists():
            candidates.update(ancestors(base))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        for first in input_root.iterdir():
            if first.is_dir():
                candidates.add(first.resolve())
                for second in first.iterdir():
                    if second.is_dir():
                        candidates.add(second.resolve())
    return candidates

def resolve_repo_root():
    roots = []
    for root in candidate_roots():
        source_dir = root / REL_SOURCE_DIR
        if source_dir.is_dir():
            roots.append(root)
    if len(roots) != 1:
        stop("expected exactly one evidence root; found " + str(len(roots)))
    return roots[0]

def checked_out_commit(root):
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            stop("malformed .git file")
        git_dir = (root / text[len("gitdir: "):]).resolve()
    elif dot_git.is_dir():
        git_dir = dot_git
    else:
        stop("repository .git metadata is missing")
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: "):]
        if not ref.startswith("refs/") or ".." in Path(ref).parts:
            stop("unsafe git HEAD ref")
        ref_path = git_dir / ref
        if ref_path.is_file():
            head = ref_path.read_text(encoding="utf-8").strip()
        else:
            packed = git_dir / "packed-refs"
            if not packed.is_file():
                stop("unresolvable symbolic git HEAD")
            matches = [line.split(" ", 1)[0] for line in packed.read_text(encoding="utf-8").splitlines()
                       if line and not line.startswith(("#", "^")) and line.endswith(" " + ref)]
            if len(matches) != 1:
                stop("ambiguous packed symbolic git HEAD")
            head = matches[0]
    if head != EXECUTION_COMMIT:
        stop("repository HEAD does not match execution commit")

def parse_meta(meta_path):
    fields = {}
    required = {
        "RUN_NAME": RUN_NAME,
        "EXPECTED_COMMIT": EXECUTION_COMMIT,
        "ACTUAL_COMMIT": EXECUTION_COMMIT,
        "COMMAND_SHA256": COMMAND_SHA256,
        "STARTED_UTC": STARTED_UTC,
        "FINISHED_UTC": FINISHED_UTC,
        "EXIT_CODE": str(EXIT_CODE),
    }
    try:
        lines = meta_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        stop("run.meta is not strict UTF-8")
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in required:
            if key in fields:
                stop("duplicate run.meta field: " + key)
            fields[key] = value
    if fields != required:
        stop("run.meta semantic binding mismatch")

repo_root = resolve_repo_root()
source_dir = repo_root / REL_SOURCE_DIR
checked_out_commit(repo_root)

for wrapper_source, staged_name, expected in WRAPPERS:
    require_sha256(wrapper_source, expected, "authentic " + staged_name)
parse_meta(WRAPPERS[2][0])

verified_sources = []
for filename, expected in ARTIFACTS:
    source = source_dir / filename
    require_sha256(source, expected, "authorized source artifact")
    size = source.stat().st_size
    verified_sources.append((source, filename, expected, size))

if OUTPUT.exists():
    stop("refusing to overwrite existing output ZIP: " + str(OUTPUT))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
stage_parent = Path(tempfile.mkdtemp(prefix="r2-recovery-stage-", dir=str(OUTPUT.parent)))
stage = stage_parent / "package"
try:
    stage.mkdir()
    staged_records = []
    for wrapper_source, staged_name, expected in WRAPPERS:
        destination = stage / staged_name
        shutil.copyfile(wrapper_source, destination)
        require_sha256(destination, expected, "staged " + staged_name)
        if destination.stat().st_size != wrapper_source.stat().st_size:
            stop("staged wrapper size mismatch: " + staged_name)

    files_root = stage / "files" / REL_SOURCE_DIR
    files_root.mkdir(parents=True)
    for source, filename, expected, size in verified_sources:
        destination = files_root / filename
        shutil.copyfile(source, destination)
        require_sha256(destination, expected, "staged artifact")
        if destination.stat().st_size != size:
            stop("staged artifact size mismatch: " + filename)
        staged_records.append({
            "path": (Path("files") / REL_SOURCE_DIR / filename).as_posix(),
            "size_bytes": size,
            "sha256": expected,
        })

    manifest = {
        "schema": "contramamba-handoff-v3",
        "run_name": RUN_NAME,
        "expected_commit": EXECUTION_COMMIT,
        "actual_commit": EXECUTION_COMMIT,
        "command_file": "command.sh",
        "command_sha256": COMMAND_SHA256,
        "run_log_sha256": RUN_LOG_SHA256,
        "run_meta_sha256": RUN_META_SHA256,
        "started_utc": STARTED_UTC,
        "finished_utc": FINISHED_UTC,
        "exit_code": EXIT_CODE,
        "file_count": 5,
        "artifact_discovery": "fixed_authorized_path_sha256_recovery",
        "collected_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": staged_records,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")

    expected_members = [
        "manifest.json", "run.log", "run.meta", "command.sh",
        *[record["path"] for record in staged_records],
    ]
    if len(expected_members) != 9 or len(set(expected_members)) != 9:
        stop("internal expected ZIP namespace is invalid")
    with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for member in expected_members:
            zf.write(stage / member, member)

    with zipfile.ZipFile(OUTPUT, "r") as zf:
        members = zf.namelist()
        if members != expected_members or len(members) != len(set(members)):
            stop("ZIP member namespace mismatch, duplicate, extra, or missing member")
        for member in members:
            pure = Path(member)
            if member.startswith("/") or "\\" in member or ".." in pure.parts or pure.as_posix() != member:
                stop("unsafe ZIP member path: " + member)
        packaged_manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        for member in members[1:]:
            data = zf.read(member)
            if member == "command.sh":
                expected = COMMAND_SHA256
            elif member == "run.log":
                expected = RUN_LOG_SHA256
            elif member == "run.meta":
                expected = RUN_META_SHA256
            else:
                expected = next(r["sha256"] for r in staged_records if r["path"] == member)
            if hashlib.sha256(data).hexdigest() != expected:
                stop("ZIP member SHA256 mismatch: " + member)
        required_manifest = {
            "schema": "contramamba-handoff-v3", "run_name": RUN_NAME,
            "expected_commit": EXECUTION_COMMIT, "actual_commit": EXECUTION_COMMIT,
            "command_file": "command.sh", "command_sha256": COMMAND_SHA256,
            "run_log_sha256": RUN_LOG_SHA256, "run_meta_sha256": RUN_META_SHA256,
            "started_utc": STARTED_UTC, "finished_utc": FINISHED_UTC,
            "exit_code": EXIT_CODE, "file_count": 5,
            "artifact_discovery": "fixed_authorized_path_sha256_recovery",
        }
        if any(packaged_manifest.get(k) != v for k, v in required_manifest.items()):
            stop("manifest immutable identity mismatch")
        if not isinstance(packaged_manifest.get("collected_utc"), str):
            stop("manifest collected_utc is invalid")
        if packaged_manifest.get("files") != staged_records:
            stop("manifest file records mismatch")
        if len(packaged_manifest["files"]) != packaged_manifest["file_count"]:
            stop("manifest file_count mismatch")
        for record in packaged_manifest["files"]:
            member = record["path"]
            if member not in members or len(zf.read(member)) != record["size_bytes"]:
                stop("manifest artifact size/path mismatch: " + member)
finally:
    shutil.rmtree(stage_parent)

print("RECOVERY_HANDOFF_READY=" + str(OUTPUT))
PY
```

## Frozen identities

| Binding | Frozen value |
| --- | --- |
| Run name | `p3w7-seed8192-a0-seed180-r2` |
| Execution / expected / actual commit | `abd85a088c274678004432160625d42208112848` |
| command.sh SHA256 | `82be0c377e305228609e8ce9f75a6b3e8b8a6f83f9999a1543914f9fc517c1f4` |
| run.log SHA256 | `a0d4b015cc77e8060184a5035333fe46e101f52d931af3101950382e24407f4e` |
| run.meta SHA256 | `23989109951bdf2b3fdfc17f9a7739c34e2875131bc76c0ccb1e313858fcdc27` |
| STARTED_UTC | `2026-09-08T22:38:29Z` |
| FINISHED_UTC | `2026-09-08T22:41:44Z` |
| EXIT_CODE | `0` |
| Source directory | `reports/reason_router_p3w7_seed8192_revised_split_a0_runs/seed180/A0` |
| ZIP output | `/kaggle/working/contramamba_handoffs/p3w7-seed8192-a0-seed180-r2_abd85a088c274.zip` |

| Fixed source relative path | SHA256 |
| --- | --- |
| `training_report.json` | `146b7330f6cf479bd339b2eb0af886d5eda3eed72589f57c0879bb5f6d9f5d9c` |
| `training_report_predictions.jsonl` | `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334` |
| `clean_dev_predictions.json` | `5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d` |
| `run_provenance.json` | `057237823c0b56a907b051b1d4018eb9317aacd860f986083aaa2277307ffbbf` |
| `selected_checkpoint.pt` | `0724f5a2e537c932f6692dd74713d57fc70f8182ee2719c33665b09114bd944a` |

## Verification record

The exact fenced payload is `11386` UTF-8 bytes with SHA256
`e4e7f9e8a15082b7b0a93faefaf5d60245674a65360432c204f2131b498a309c`.
It has one final LF and no UTF-8 BOM.  Python compilation of the embedded
stdlib program passed.  The local Windows Bash bridge could not start its WSL
instance (`E_ACCESSDENIED`), so Bash syntax validation is not run in this
authoring environment; the command was otherwise statically inspected.
Final report/blob identity is verified without executing the recovery command
after this authored file is complete.  Recovery/Kaggle/import/training/
evaluation remain **NOT YET PERFORMED**.
