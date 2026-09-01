# Longterm O0b CPU Package-Version Preflight Invalid Torch-Version Recovery Authority Spec Candidate

## Status

PASS_READY_FOR_INDEPENDENT_VERIFICATION

This document is a static authority/specification candidate only. It resolves only the consumed O0b CPU package-version preflight v3 failure token:

```text
O0B_CPU_PACKAGE_PREFLIGHT_BLOCKED:INVALID_TORCH_VERSION
```

It does not authorize implementation, Kaggle execution, package/runtime probing, tokenizer execution, model loading, model weights, model forward, dataset loading, training, evaluation, scientific interpretation, commit, or push.

## 1. Authority Chain

Authority precedence for this recovery candidate is:

1. Current controller instruction.
2. Frozen O0b CPU package-version preflight authority at commit `a0ee0a260369b99db160a117bef842ba6c0e945c`, file `reports/longterm_o0b_cpu_package_version_preflight_authority_spec_candidate.md`.
3. Final observer implementation commit `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`.
4. Active upstream exact-command recovery authority commit `67cc985963aa44df952978fd98b1ed18dfc9e13c`.
5. Imported v3 failure evidence supplied by the controller.
6. Repository `AGENTS.md` and applicable operational documentation.

If this candidate conflicts with a higher authority above, this candidate is invalid.

## 2. Purpose And Scope

This candidate defines a narrow future v4 package-version preflight authority for the same final O0b observer implementation binding as v3.

The only validation-contract defect being recovered is the v3 exact command's stricter-than-authority runtime-version string predicate:

```python
type(value) is str
```

The replacement rule is:

```python
isinstance(value, str)
```

All other version-value checks remain unchanged:

- `value == value.strip()`
- `value.strip().lower() not in FORBIDDEN_VERSION_VALUES`
- missing, `None`, empty, whitespace-only, `unknown`, `n/a`, and `none` remain rejected.

This candidate does not permit `str(value)` coercion, repr-based acceptance, fallback or inferred package versions, package metadata lookup as substitute, pip/importlib metadata substitution, or guessed version values.

The future v4 command must still read package/runtime values directly from the imported runtime objects:

- `python_version = sys.version.split()[0]`
- `numpy_version = numpy.__version__`
- `torch_version = torch.__version__`
- `transformers_version = transformers.__version__`

This candidate explicitly does not assert that the live Kaggle `torch.__version__` value was a specific subclass, built-in type, or raw value. The v3 command did not print the raw type or raw value, so that fact is unobserved.

## 3. Consumed v3 Failure Provenance

The v3 run is classified as a consumed failed preflight attempt. Its run name and exact command identity are non-reusable.

Frozen v3 run name:

```text
longterm-o0b-cpu-package-version-preflight-9a249c0-v3
```

Frozen v3 execution HEAD:

```text
9a249c071b76fbf693f63b36ba8ec1036c69b2ba
```

Frozen v3 exact command SHA256:

```text
868a190855f56489fa0a9d998e6978aa2e280667bf235d72048b9e6bbcfdb4e5
```

Frozen v3 exact command byte contract:

```text
bytes=5398
CR=0
LF=135
first_byte=0x70
final_byte=0x59
final_LF_absent=true
```

Observer binding:

```text
path=scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py
sha256=7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375
bytes=45255
```

v3 execution evidence:

```text
EXPECTED_COMMIT=9a249c071b76fbf693f63b36ba8ec1036c69b2ba
ACTUAL_COMMIT=9a249c071b76fbf693f63b36ba8ec1036c69b2ba
wrapper_command_sha256=868a190855f56489fa0a9d998e6978aa2e280667bf235d72048b9e6bbcfdb4e5
python_exact_command_reached=true
failure_token=O0B_CPU_PACKAGE_PREFLIGHT_BLOCKED:INVALID_TORCH_VERSION
EXIT_CODE=1
STARTED_UTC=2026-09-01T13:52:44Z
FINISHED_UTC=2026-09-01T13:53:13Z
RUN_LOG_SHA256=84fcd0a7226dd3cc19b23da5c9e327b991cc1945885af29cab68d2c2231e1bd8
RUN_META_SHA256=e021b04f9a970bf2e94a001a6f0b7b73d1c87b63da4f8e439cf8092f67994c04
```

v3 collection/import evidence:

```text
COLLECT=PASS
FILES_COLLECTED=0
package_version_json_created=false
imported_zip_sha256=4daa6f15b85f7f2d1a93e46b3e26af2e2da47e0befc5876795b8760d792ddf2e
import_EXIT_CODE=1
VALIDATED=0
COPIED=0
IDENTICAL=0
IMPORT=PASS
```

`FILES_COLLECTED=0` is expected for this failure class because the exact command failed during version validation before the package-version output JSON was written. The absence of the JSON artifact is therefore an artifact outcome of a fail-closed preflight, not evidence of collector malfunction.

## 4. Classification

The v3 evidence separates as follows:

- Command/provenance correctness: PASS. Expected and actual commits matched, the wrapper command SHA matched the frozen v3 SHA, and execution reached the Python exact command.
- Execution outcome: FAIL-CLOSED PREFLIGHT. The command exited with `O0B_CPU_PACKAGE_PREFLIGHT_BLOCKED:INVALID_TORCH_VERSION`.
- Artifact outcome: NO PACKAGE-VERSION JSON. This is expected because validation failed before the output artifact write.
- Scientific conclusion: NONE. This was package/runtime provenance preflight only. It did not run tokenizer, model load, model weights, model forward, dataset execution, training, evaluation, or hidden-state scientific observation.

The defect is narrowly classified as a package-version validation-contract issue. The frozen authority prose requires concrete, non-empty strings with no surrounding whitespace and no forbidden placeholder value. The v3 exact command additionally required exact built-in string type via `type(value) is str`, which is stricter than the authority's string contract and may reject legitimate string subclasses. This classification does not claim PyTorch, CUDA, the observer, or the Kaggle runtime is invalid.

## 5. Supersession Boundary

This candidate supersedes only the exact-type predicate in v3 runtime-version validation:

```text
type(value) is str -> isinstance(value, str)
```

It preserves unchanged:

- v3 GPU-OFF safeguards;
- execution HEAD `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- observer Git-object SHA256 `7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375`;
- observer Git-object byte size `45255`;
- allowed-import boundary;
- direct package/runtime version sources;
- tokenizer/model/weights/forward/dataset/training/evaluation prohibitions;
- output key set;
- JSON encoding contract;
- command success evidence;
- fail-closed behavior.

## 6. v4 Future Run Identity

The consumed v3 run name must not be reused.

Fresh future v4 run name:

```text
longterm-o0b-cpu-package-version-preflight-9a249c0-v4
```

Fresh future v4 artifact path:

```text
reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v4.json
```

The v4 run name must not be reused for another commit, another observer identity, a scientific run, a tokenizer/model run, or a retry with changed command bytes. Any further provenance-compatible retry requires a separately authorized run name and authority.

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
OUTPUT = Path("reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v4.json")
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
    return isinstance(value, str) and value == value.strip() and value.strip().lower() not in FORBIDDEN_VERSION_VALUES

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

## 8. v4 Exact Command-Byte/SHA Contract

The governing registration hash is the current `cm run save` hash. The exact-byte command identity is computed from the command between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND` after removing only the Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF.

Future v4 exact command SHA256:

```text
68d4b5b77b87bb2d531a67dee009b3383cd582c44f3e2e29328823dbac90e08b
```

Future v4 exact command byte statistics:

```text
byte_length=5402
CR_count=0
LF_count=135
first_byte=0x70
final_byte=0x59
final_LF_absent=true
```

The hash excludes:

- `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`;
- Markdown fence bytes;
- display wrapping;
- any leading blank line;
- any trailing blank line;
- any added final LF byte after the final `PY` line.

If `cm run save` produces any different command SHA for the copied command, the v4 preflight is not authorized.

## 9. v3 To v4 Semantic Delta

The v4 exact command is derived from the frozen v3 command with only these semantic differences:

```text
OUTPUT = Path("reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v3.json")
->
OUTPUT = Path("reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v4.json")
```

```text
return type(value) is str and value == value.strip() and value.strip().lower() not in FORBIDDEN_VERSION_VALUES
->
return isinstance(value, str) and value == value.strip() and value.strip().lower() not in FORBIDDEN_VERSION_VALUES
```

There is no semantic relaxation beyond accepting legitimate string subclasses under the authority's string contract. The command still rejects missing, `None`, empty, whitespace-only, trimmed-different, `unknown`, `n/a`, and `none` values. It still does not coerce, infer, guess, or substitute package versions.

## 10. Future cm Workflow Design

After this candidate becomes active, future execution must use the existing cm provenance workflow:

```text
cm run save longterm-o0b-cpu-package-version-preflight-9a249c0-v4
cm run longterm-o0b-cpu-package-version-preflight-9a249c0-v4
cm collect longterm-o0b-cpu-package-version-preflight-9a249c0-v4
cm import <handoff.zip>
```

Required future provenance evidence:

- local cm run-registry entry for `longterm-o0b-cpu-package-version-preflight-9a249c0-v4`;
- registry HEAD exactly `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- registry command exactly equal to the command in this authority;
- registry command SHA exactly equal to `68d4b5b77b87bb2d531a67dee009b3383cd582c44f3e2e29328823dbac90e08b`;
- Kaggle wrapper `command.sh`;
- Kaggle wrapper `run.meta`;
- Kaggle wrapper `run.log`;
- Kaggle wrapper `start.marker`;
- handoff `manifest.json` with the active schema required by cm;
- manifest run name equal to `longterm-o0b-cpu-package-version-preflight-9a249c0-v4`;
- manifest expected and actual commits both equal `9a249c071b76fbf693f63b36ba8ec1036c69b2ba`;
- manifest command hash equal to the registry and authority command hash;
- collected artifact entry for `reports/longterm_o0b_cpu_package_version_preflight_9a249c0_v4.json`;
- collected artifact SHA256 and byte size;
- imported JSON content matching the exact output contract.

Do not treat process exit code alone, stdout alone, or an unimported local Kaggle file as sufficient evidence.

## 11. Fail-Closed Conditions

Future v4 preflight must fail closed if:

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
- `cm run save`, `cm run`, `cm collect`, or `cm import` cannot bind evidence to the registered v4 run name, registered v4 command SHA, and registered commit;
- collected/imported evidence cannot be bound to the registered v4 run and commit.

No cleanup, retry, rerun, model execution, or scientific interpretation is authorized by a preflight failure.

## 12. Activation Rule

This candidate is not executable merely because it exists.

Activation requires:

1. independent verifier PASS on exact candidate bytes;
2. exact candidate freeze commit/push;
3. controller records the full freeze commit and committed Git-object identity;
4. exact command byte/hash verification;
5. explicit controller transition to execution.

Only that active freeze may authorize the package-only v4 preflight. No post-freeze textual edit is required for activation.

## 13. Protected State

This candidate authoring task must not modify, stage, delete, clean, reset, stash, or consume as substantive task inputs:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`
- `reports/stage180a_pass2_annotations_completed.csv`
- pre-existing root patch files
- unrelated untracked state.

Permission warnings from protected or unrelated local temporary directories are nonblocking unless they prevent required static validation.

## 14. Validation Contract For This Candidate

For this static authoring task, run at minimum:

```text
git rev-parse HEAD
git diff --check
git diff --name-status
git status --short
```

Confirm:

- HEAD remains exactly `a0ee0a260369b99db160a117bef842ba6c0e945c` during authoring;
- exactly one intended new candidate file for this task;
- no existing tracked file modified;
- nothing staged;
- v4 exact command can be extracted from this candidate;
- v3 and v4 exact commands have no unexplained semantic differences.

Compute and report:

- candidate SHA256;
- candidate byte size;
- v4 exact command byte size;
- v4 exact command CR count;
- v4 exact command LF count;
- v4 exact command first byte;
- v4 exact command final byte;
- whether final LF is absent or present;
- v4 exact command SHA256.

No pytest, package import, runtime probing, Kaggle execution, training, or evaluation is authorized by this candidate-authoring task.

## 15. Explicit Non-Execution Attestation

NO KAGGLE

NO PACKAGE PROBE

NO TOKENIZER

NO MODEL LOAD

NO MODEL WEIGHTS

NO FORWARD

NO DATASET EXECUTION

NO TRAINING

NO EVALUATION

NO COMMIT

NO PUSH

## 16. Next Authorized Action

The exact next authorized action is independent verifier review of these exact candidate bytes. If and only if independent verification returns PASS, the candidate may be committed and pushed unchanged for controller freeze-recording. Only after controller activation may the package-only v4 preflight be registered and run through cm under this authority.
