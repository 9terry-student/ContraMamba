# Longterm O0b CPU Package-Version Preflight Authority Spec Candidate

## Status

PASS_READY_FOR_INDEPENDENT_VERIFICATION

This document is a static authority/specification candidate only. It resolves only the remaining package-version blocker recorded by the active O0b exact-command recovery authority. It does not establish, execute, validate, or interpret any O0b scientific result.

Authoring this document does not authorize package preflight execution, Kaggle execution, tokenizer execution, model loading, model weights, model forward, hidden-state scientific forward, dataset loading or regeneration, training, evaluation, staging, commit, or push.

## 1. Authority Chain

Authority precedence for this candidate is:

1. Current controller instruction.
2. Independent verifier FAIL report over repaired v2 candidate bytes: SHA256 `43cc049e1fcf78d9ec3c2cb581ded5ae6d98e3ebf17091ec0aa00f19270a0037`, byte size `22133`.
3. Rejected v2 exact-command SHA256: `72ae10ee45ffc7d4d5b70e320d991f4ffd3edbe4fa9fb15e47a9e12bed9d46d0`.
4. Active O0b exact-command recovery authority candidate: `reports/longterm_o0b_execution_provenance_preflight_exact_command_recovery_authority_spec_candidate.md`, freeze identity `67cc985963aa44df952978fd98b1ed18dfc9e13c`.
5. Final exact-command repaired observer implementation: `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.
6. Committed observer Git-object identity for `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`: SHA256 `7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375`, byte size `45255`.
7. O0b scientific design: `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
8. O0b boundary recovery: `2ed4439e511f7534186cbd5df9110e45fdc1d66c`.
9. Repaired matched-control implementation: `7ce4e0cd05d87118c29526a53ab5178dc722db27`.
10. O0b observer implementation authority: `65881cf398d26b136e4984686b14f7d40b939c3e`.
11. Runtime-package provenance recovery authority: `27515b7cde33e02f992b093c70fec08d92e1b721`.
12. Repository `AGENTS.md`.

O0a authorities and P3-W7/P4-L authorities were inspected only as workflow precedent for activation, exact command hashing, cm registration, collection/import provenance, fail-closed guards, and non-claim language. They do not authorize O0b execution.

## 2. Purpose

This authority candidate defines the exact future CPU-only Kaggle package-version preflight needed before final O0b scientific execution authority can be authored.

The future preflight may establish immutable exact intended runtime values for:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

Successful package preflight is environment/provenance evidence only. It is not a scientific result, not an observer semantic result, not an entitlement result, and not evidence that hidden-state dynamics were measured correctly.

## 3. Final Implementation Binding

Future preflight repository HEAD must be exactly:

```text
9a249c071b76fbf693f63b36ba8ec1036c69b2ba
```

Observer path:

```text
scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py
```

Committed observer Git-object SHA256:

```text
7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375
```

Committed observer Git-object byte size:

```text
45255
```

This authority binds the committed Git-object bytes that a clean Kaggle checkout receives. It explicitly does not bind or reuse the former Windows working-copy SHA:

```text
c8c023d12a36112fdbfe8060f7c4b374f339e365b4cd7dc1d70e4c2b861f03fb
```

If the future Kaggle checkout or collection/import evidence cannot prove the committed observer Git-object SHA and byte size above, the preflight fails closed.

## 4. CPU/GPU And Execution Boundary

Future preflight is:

- Kaggle only.
- CPU only.
- GPU OFF for the package-preflight runtime.
- Package/runtime identity only.

GPU OFF is defined operationally as: no NVIDIA GPU is exposed to the package-preflight runtime at either the framework level or the OS/device level.

The exact command must observe the Kaggle runtime as supplied. It must not set or overwrite `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, or an equivalent GPU-masking environment variable. `CUDA_VISIBLE_DEVICES` and related environment values may be observed diagnostically, but their values alone never prove GPU OFF. An empty `CUDA_VISIBLE_DEVICES` value must not substitute for the required PyTorch, OS/device, proc, and `nvidia-smi` checks.

The preflight must fail closed unless all substantive GPU-OFF checks pass:

- `torch.cuda.is_available()` is `False`;
- `torch.cuda.device_count()` equals `0`;
- no NVIDIA GPU/control device nodes are exposed under `/dev/nvidia*`, including any discovered NVIDIA character-device path relevant to an attached GPU;
- `/proc/driver/nvidia/gpus` does not expose any GPU entry;
- `nvidia-smi -L`, used only as a hardware/provenance probe and not as a workload, does not successfully enumerate one or more GPU devices.

For `nvidia-smi -L`, the exact command must capture return code, stdout, and stderr. It must inspect combined stdout plus stderr before using return code to classify the probe. Any line in combined output matching `^\s*GPU\s+\d+\s*:` in multiline mode is substantive GPU enumeration and must fail immediately regardless of return code. Executable-not-found may be treated as no `nvidia-smi` evidence, and a nonzero return code with no GPU enumeration may be treated as no positive `nvidia-smi` evidence, but GPU OFF may pass only when every other independent check also passes. Timeout or any unexpected subprocess/runtime exception other than executable-not-found fails closed.

The preflight must not perform a CUDA computation, allocate CUDA tensors, initialize tokenizer/model code, or treat CUDA-capable package-build metadata as device exposure. In particular, `torch.version.cuda` or equivalent package-build metadata may contain a CUDA version without causing failure. This preflight measures exposed hardware/accelerator access, not whether the installed PyTorch wheel was compiled with CUDA support.

Allowed Python imports are limited to the minimum required to verify provenance, validate strings, and emit a deterministic runtime-version artifact:

- `hashlib`
- `json`
- `re`
- `stat`
- `subprocess`
- `sys`
- `pathlib.Path`
- `numpy`
- `torch`
- `transformers`

Importing `numpy`, `torch`, and `transformers` solely to read exact version strings is allowed.

Forbidden during preflight:

- tokenizer invocation;
- tokenizer download or use;
- model loading;
- model weights;
- model forward;
- hidden-state forward;
- dataset loading or regeneration;
- scientific artifact generation;
- training;
- evaluation;
- scientific interpretation.

The preflight command must not import the O0b observer module. It may inspect the observer only through Git-object bytes.

## 5. Exact Output Contract

The future preflight must create exactly one deterministic machine-readable runtime artifact under the repository:

```text
reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v3.json
```

The v1 and v2 preflight identities are historical rejected candidates only. They are not active authority, must not be reused, and must not be interpreted as authorizing any future run or artifact.

The artifact is environment/provenance evidence only. It must be JSON encoded as UTF-8 with LF line ending, `indent=2`, `sort_keys=True`, and no extra fields beyond:

- `numpy_version`
- `observer_implementation_commit`
- `observer_script_sha256`
- `python_version`
- `torch_version`
- `transformers_version`

The exact required JSON key set is:

```text
observer_implementation_commit
observer_script_sha256
python_version
numpy_version
torch_version
transformers_version
```

`observer_implementation_commit` must equal `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.

`observer_script_sha256` must equal `7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375`.

The four package/runtime values must be concrete, non-empty strings captured from the runtime:

- `python_version = sys.version.split()[0]`
- `numpy_version = numpy.__version__`
- `torch_version = torch.__version__`
- `transformers_version = transformers.__version__`

Reject/forbid:

- missing values;
- `None`;
- empty strings;
- whitespace-only strings;
- leading or trailing whitespace;
- `unknown` / `UNKNOWN`;
- `n/a` / `N/A`;
- `none` / `None`;
- inferred values;
- guessed values.

The stdout success token and printed artifact SHA are supporting evidence only. Process exit code alone is not sufficient evidence.

## 6. Frozen Run Name

The exact future cm run name is:

```text
longterm-o0b-cpu-package-version-preflight-9a249c0-v3
```

This fresh v3 run name supersedes the independently rejected v1 and v2 preflight candidate identities. It must not be reused for another commit, another observer identity, a scientific run, a tokenizer/model run, or a retry with changed command bytes. Any provenance-compatible retry requires a separately authorized run name and authority.

## 7. Exact Package-Preflight Command

The future package preflight command is the exact command between the `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND` markers below. It is specified here but must not be executed by this authoring task.

BEGIN_EXACT_COMMAND
```bash
python - <<'PY'
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy
import torch
import transformers

EXPECTED_HEAD = "9a249c071b76fbf693f63b36ba8ec1036c69b2ba"
OBSERVER_PATH = "scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py"
EXPECTED_OBSERVER_SHA256 = "7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375"
EXPECTED_OBSERVER_SIZE = 45255
OUTPUT = Path("reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v3.json")
FORBIDDEN_VERSION_VALUES = {"", "unknown", "n/a", "none"}

def fail(code):
    raise SystemExit("O0B_CPU_PACKAGE_PREFLIGHT_BLOCKED:" + code)

def require(condition, code):
    if not condition:
        fail(code)

def git_text(*args):
    result = subprocess.run(["git", *args], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, "GIT_FAILED_" + "_".join(args))
    return result.stdout

def git_bytes(*args):
    result = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, "GIT_FAILED_" + "_".join(args))
    return result.stdout

def valid_version(value):
    return type(value) is str and value == value.strip() and value.strip().lower() not in FORBIDDEN_VERSION_VALUES

def exposed_nvidia_device_paths():
    paths = []
    dev = Path("/dev")
    if dev.exists():
        paths.extend(str(path) for path in dev.glob("nvidia*"))
    dev_char = Path("/dev/char")
    if dev_char.exists():
        for path in dev_char.iterdir():
            try:
                resolved = path.resolve(strict=False)
                mode = path.stat().st_mode
            except OSError:
                continue
            if stat.S_ISCHR(mode) and "nvidia" in str(resolved).lower():
                paths.append(str(path) + "->" + str(resolved))
    return sorted(set(paths))

def proc_nvidia_gpu_entries():
    proc_gpus = Path("/proc/driver/nvidia/gpus")
    if not proc_gpus.exists():
        return []
    try:
        return sorted(path.name for path in proc_gpus.iterdir() if path.name not in {"", ".", ".."})
    except OSError as exc:
        fail("PROC_NVIDIA_GPUS_UNREADABLE_" + exc.__class__.__name__)

def nvidia_smi_enumerates_gpu():
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        fail("NVIDIA_SMI_TIMEOUT")
    except Exception as exc:
        fail("NVIDIA_SMI_SUBPROCESS_FAILED_" + exc.__class__.__name__)
    return_code = result.returncode
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined_output = stdout + "\n" + stderr
    if re.search(r"^\s*GPU\s+\d+\s*:", combined_output, flags=re.MULTILINE) is not None:
        return True
    if return_code != 0:
        return False
    return False

require(torch.cuda.is_available() is False, "TORCH_CUDA_AVAILABLE")
require(torch.cuda.device_count() == 0, "TORCH_CUDA_DEVICE_COUNT_NONZERO")
require(exposed_nvidia_device_paths() == [], "NVIDIA_DEVICES_EXPOSED")
require(proc_nvidia_gpu_entries() == [], "NVIDIA_PROC_GPUS_EXPOSED")
require(not nvidia_smi_enumerates_gpu(), "NVIDIA_SMI_ENUMERATES_GPU")
require(git_text("rev-parse", "HEAD").strip() == EXPECTED_HEAD, "HEAD_MISMATCH")
require(git_text("status", "--short", "--untracked-files=no") == "", "TRACKED_WORKTREE_DIRTY")
require(git_text("diff", "--cached", "--name-status") == "", "INDEX_DIRTY")
require(re.fullmatch(r"[0-9a-f]{40}", EXPECTED_HEAD) is not None, "EXPECTED_HEAD_NOT_LOWERCASE_40_HEX")
require(git_text("cat-file", "-t", EXPECTED_HEAD).strip() == "commit", "EXPECTED_HEAD_NOT_COMMIT")

observer_bytes = git_bytes("show", EXPECTED_HEAD + ":" + OBSERVER_PATH)
observer_sha256 = hashlib.sha256(observer_bytes).hexdigest()
observer_size = len(observer_bytes)
require(observer_sha256 == EXPECTED_OBSERVER_SHA256, "OBSERVER_GIT_OBJECT_SHA_MISMATCH")
require(observer_size == EXPECTED_OBSERVER_SIZE, "OBSERVER_GIT_OBJECT_SIZE_MISMATCH")

versions = {
    "python_version": sys.version.split()[0],
    "numpy_version": numpy.__version__,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}
for key, value in versions.items():
    require(valid_version(value), "INVALID_" + key.upper())

payload = {
    "observer_implementation_commit": EXPECTED_HEAD,
    "observer_script_sha256": observer_sha256,
    **versions,
}
require(set(payload) == {
    "observer_implementation_commit",
    "observer_script_sha256",
    "python_version",
    "numpy_version",
    "torch_version",
    "transformers_version",
}, "OUTPUT_KEY_SET_MISMATCH")
require(not OUTPUT.exists(), "OUTPUT_COLLISION")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
OUTPUT.write_text(text, encoding="utf-8", newline="\n")
roundtrip = json.loads(OUTPUT.read_text(encoding="utf-8"))
require(roundtrip == payload, "OUTPUT_ROUNDTRIP_MISMATCH")
artifact_bytes = OUTPUT.read_bytes()
print("O0B_CPU_PACKAGE_PREFLIGHT_PASS")
print("artifact_path=" + OUTPUT.as_posix())
print("artifact_sha256=" + hashlib.sha256(artifact_bytes).hexdigest())
print("artifact_byte_size=" + str(len(artifact_bytes)))
PY
```
END_EXACT_COMMAND

No argument or byte may be edited in the future Kaggle cell generated by `cm run`.

## 8. Exact Command-Byte/SHA Contract

Repository precedent establishes two relevant command hash conventions:

- `cm run save <name>` stores the command string and computes its SHA256 over the exact stored UTF-8 command bytes with no added final LF.
- Some execution authorities additionally define display-level wrapper command hashes with an added final LF.

For this preflight, the governing registration hash is the current `cm run save` hash. The exact-byte command identity is computed from the command between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND` after removing only the Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF.

Exact `cm` registry command SHA256:

```text
868a190855f56489fa0a9d998e6978aa2e280667bf235d72048b9e6bbcfdb4e5
```

Exact command byte statistics under that convention:

```text
byte_length=5398
CR_count=0
LF_count=135
first_byte=112
final_byte=89
```

The hash excludes:

- `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`;
- Markdown fence bytes;
- display wrapping;
- any leading blank line;
- any trailing blank line;
- any added final LF byte after the final `PY` line.

If `cm run save` produces any different command SHA for the copied command, the preflight is not authorized.

## 9. cm Registration / Run / Collect / Import Design

After this candidate becomes active, future execution must use the existing cm provenance workflow:

```text
cm run save longterm-o0b-cpu-package-version-preflight-9a249c0-v3
cm run longterm-o0b-cpu-package-version-preflight-9a249c0-v3
cm collect longterm-o0b-cpu-package-version-preflight-9a249c0-v3
cm import <handoff.zip>
```

The exact command above must be the command saved by `cm run save`. `cm run` must transport that command unchanged through the registered command SHA mechanism. `cm collect` must package wrapper evidence and the generated JSON artifact through the standard handoff manifest. `cm import` must validate the local registry, expected and actual commits, command hash, wrapper metadata/log hashes, artifact paths, sizes, and artifact SHA256 before importing.

This design remains compatible with the current live `C:\Users\Home1\.contramamba\cm.ps1` identity inspected for this repair task: SHA256 `b15d70832e7c76c05fea6a9955bd199edcf9fb633fe0fe34266c44788260f570`. It relies on the independently verified default `cm run save` semantics: `Get-Clipboard -Raw`, `.Trim()`, Markdown fences excluded, leading `%%bash` stripped if present, UTF-8, multiline LF, and no final LF. Because this repaired command is multiline, it must not use `CONTRAMAMBA_RUN_COMMAND_BYTE_MODE=utf8-final-lf-v1`.

Required future provenance evidence:

- local cm run-registry entry for `longterm-o0b-cpu-package-version-preflight-9a249c0-v3`;
- registry HEAD exactly `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- registry command exactly equal to the command in this authority;
- registry command SHA exactly equal to the SHA in this authority;
- Kaggle wrapper `command.sh`;
- Kaggle wrapper `run.meta`;
- Kaggle wrapper `run.log`;
- Kaggle wrapper `start.marker`;
- handoff `manifest.json` with schema `contramamba-handoff-v3`;
- manifest run name equal to the frozen run name;
- manifest expected and actual commits both equal `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- manifest command hash equal to the registry and authority command hash;
- collected artifact entry for `reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v3.json`;
- collected artifact SHA256 and byte size;
- imported JSON content matching the exact output contract.

Do not treat process exit code alone, stdout alone, or an unimported local Kaggle file as sufficient evidence.

## 10. Fail-Closed Conditions

Future preflight must fail closed if:

- Kaggle HEAD differs from `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- committed observer Git-object SHA differs from `7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375`;
- committed observer Git-object byte size differs from `45255`;
- tracked worktree or index is dirty under cm requirements;
- GPU OFF is not proven by all required PyTorch, OS/device, proc, and `nvidia-smi` checks;
- any required package import fails;
- any required package version is missing, `None`, empty, whitespace-only, trimmed-different, `unknown`, `n/a`, `none`, inferred, or guessed;
- the output artifact already exists before the attempt;
- the output artifact has extra or missing keys;
- the output artifact cannot be round-tripped exactly as JSON;
- `cm run save`, `cm run`, `cm collect`, or `cm import` cannot bind the evidence to the registered run name, registered command SHA, and registered commit;
- collected/imported evidence cannot be bound to the registered run and commit.

No cleanup, retry, rerun, model execution, or scientific interpretation is authorized by a preflight failure.

## 11. Relation To Exact-Command Repair

This package preflight preserves the two-layer provenance distinction from the active exact-command recovery authority:

Layer 1:

```text
external cm run / exact-byte shell command identity
```

Layer 2:

```text
observer manifest canonical argv identity
```

This package preflight is environment evidence. It does not reopen, redefine, weaken, or replace either layer.

No observer semantic change is authorized. No observer test change is authorized. The future scientific observer manifest must still obey the repaired canonical argv identity independently of this package preflight.

## 12. Future Result Use

After a future authorized preflight is run and its evidence is validated/imported, the controller may use the exact four recorded values to author the final O0b scientific execution authority.

The final scientific execution authority must freeze those exact values:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

The eventual scientific manifest must match all four values exactly. A package-version mismatch later means provenance invalidity even if the scientific observer itself exits successfully, emits a complete artifact set, or produces scientifically interesting-looking measurements.

## 13. Activation Rule

This candidate is NOT active while uncommitted.

It becomes active only after:

1. independent verifier PASS over exact candidate bytes;
2. candidate committed and pushed unchanged;
3. controller records the full authority freeze commit.

Only that active freeze may authorize the package-only preflight. Authoring this document does NOT authorize execution.

No post-freeze textual edit is required for activation.

## 14. Protected State

This candidate authoring task must not modify, stage, delete, clean, reset, stash, or consume as substantive task inputs:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/stage180a_pass2_annotations_completed.csv`
- historical `reason_router_*.patch` files
- unrelated URP/P3-W7/reason-router state.

Permission warnings from protected or unrelated local temporary directories are nonblocking unless they prevent required static validation.

## 15. Validation Contract For This Candidate

For this static authoring task, run at minimum:

```text
git rev-parse HEAD
git diff --check -- reports/longterm_o0b_cpu_package_version_preflight_authority_spec_candidate.md
git diff --name-status
git diff --stat
git diff --cached --name-status
git status --short
```

Confirm:

- HEAD remains exactly `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- exactly one intended untracked candidate file for this task;
- no existing tracked file modified;
- nothing staged.

Compute and report:

- candidate SHA256;
- candidate byte size.

No pytest is required for this authority-document authoring task.

## 16. Explicit Non-Execution Attestation

NO PREFLIGHT EXECUTION

NO CM RUN SAVE

NO CM RUN

NO CM COLLECT

NO CM IMPORT

NO KAGGLE EXECUTION

NO TOKENIZER EXECUTION

NO MODEL LOADING

NO MODEL WEIGHTS

NO MODEL FORWARD

NO HIDDEN-STATE SCIENTIFIC FORWARD

NO DATASET EXECUTION

NO TRAINING

NO EVALUATION

NO COMMIT

NO PUSH

## 17. Next Authorized Action

The exact next authorized action is independent verifier review of these exact candidate bytes. If and only if independent verification returns PASS, the candidate may be committed and pushed unchanged for controller freeze-recording. Only after that activation may the package-only preflight be registered and run through cm under this authority.
