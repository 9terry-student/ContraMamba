# ContraMamba O0c Exact-Runtime Source-Provenance Preflight Execution Authority Spec Candidate

## 1. Overall Verdict And Phase

Overall verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

Phase:

`STATIC O0c EXACT-RUNTIME SOURCE-PROVENANCE PREFLIGHT EXECUTION AUTHORITY AUTHORING ONLY`

This candidate authorizes exactly one future bounded CPU-only exact-runtime preflight run after independent verification and controller activation. This authoring task does not run the preflight and does not authorize model/tokenizer execution, training, evaluation, Kaggle execution, package mutation, implementation changes, staging, committing, or pushing.

The future run is infrastructure/provenance evidence only. It exists only to establish exact installed runtime/source provenance and static O0c compatibility/backend classification.

It is not O0c recurrent-state capture, model execution, tokenizer execution, training/evaluation, scientific O0c evidence, O1, or semantic ownership analysis.

## 2. Authority Chain

Authority order used:

1. Current controller instruction for this task card.
2. Frozen O0c runtime-source provenance preflight implementation: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.
3. Frozen O0c runtime-source provenance preflight implementation authority: `811ae9c843564e8cddb5fc373761afb618cb7cfd`.
4. Frozen O0c runtime-source provenance preflight authority: `8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`.
5. Frozen O0c native-state instrumentation authority: `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`.
6. Repository `AGENTS.md`.

Canonical repository:

`C:\Users\Home1\Desktop\ContraMamba`

Canonical HEAD verified before authoring:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

## 3. Frozen Implementation Identity

The future execution is bound to exact commit:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

Exact implementation script:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

Expected script SHA256:

`73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc`

Expected script bytes:

`32034`

Frozen test identity for provenance context:

`tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

Expected test SHA256:

`ddb5e04bc6399453237f787d66ce8a81363a069f247bdb225cfbdf6157645491`

Expected test bytes:

`27894`

Execution must fail closed on commit mismatch, dirty/widened repository state, script SHA256 mismatch, script byte-count mismatch, command mismatch, run-name mismatch, or artifact mismatch.

## 4. Target Runtime

Exact expected runtime strings:

```text
Python: 3.12.13
NumPy: 2.0.2
torch: 2.10.0+cpu
Transformers: 5.0.0
```

No package installation, removal, upgrade, or downgrade is authorized. If the execution environment differs, the frozen implementation must return:

`BLOCKED_RUNTIME_VERSION_MISMATCH`

The environment must not be repaired inside the run.

## 5. Execution Environment

Preferred environment:

`Kaggle Notebook / kernel with CPU only`

Required environment constraints:

- Accelerator: `None`.
- GPU: OFF.
- Internet: preflight must not require network access; prefer Internet OFF if compatible with repository provisioning.
- No model, tokenizer, dataset, or pretrained-weight provisioning.
- No CUDA.
- No optional kernel installation or compilation.

## 6. Canonical Output Artifact

The exact canonical preflight artifact path is:

`reports/longterm_o0c_runtime_source_provenance_preflight.json`

This path is repository-relative, deterministic, under `reports/`, not timestamped, and does not conflict with the existing O0c preflight authority/implementation naming convention. It is distinct from the authority candidates:

- `reports/longterm_o0c_runtime_source_provenance_preflight_authority_spec_candidate.md`
- `reports/longterm_o0c_runtime_source_provenance_preflight_implementation_authority_spec_candidate.md`
- `reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`

The output must be produced only by the frozen implementation script. It must not be synthesized manually.

Expected `schema_version`:

`o0c_runtime_source_provenance_preflight_v1`

Collision behavior:

- if the output path exists before execution, the command and implementation must fail closed;
- the implementation status for output collision is `BLOCKED_OUTPUT_COLLISION`;
- overwrite, append, timestamped fallback, random fallback, and manual deletion are not authorized.

On success, artifact SHA256 and byte count must be captured and printed immediately after validation.

Deterministic JSON requirements:

- UTF-8 JSON;
- sorted keys;
- two-space indentation;
- final LF;
- `allow_nan=False` semantics;
- deterministic list ordering;
- no timestamp, hostname, username, UUID, mutable branch name, random filename, training/evaluation metric, model output, tokenized data, or scientific O0c signal.

## 7. Exact CLI

The exact implemented CLI to be invoked by the frozen command is:

```text
python scripts/preflight_longterm_o0c_runtime_source_provenance.py \
  --output reports/longterm_o0c_runtime_source_provenance_preflight.json \
  --expected-python 3.12.13 \
  --expected-numpy 2.0.2 \
  --expected-torch 2.10.0+cpu \
  --expected-transformers 5.0.0
```

Kaggle shell syntax may adapt only path quoting/line-continuation mechanics required by the shell wrapper. No implementation-unauthorized flags may be added.

## 8. Run Name

Exact future run name:

`longterm-o0c-runtime-source-provenance-preflight-de874a2-v1`

This descriptive name follows the existing longterm O0 run naming style, includes the short frozen commit identity, and must not be reused for any other attempt.

## 9. Exact Future Kaggle Shell Command

The exact future Kaggle shell command is the command between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`. This authoring task records it but does not execute it.

BEGIN_EXACT_COMMAND
```bash
set -euo pipefail

EXPECTED_COMMIT="de874a22df4f60adbdc5efbcf294961c7b3a48a5"
SCRIPT_PATH="scripts/preflight_longterm_o0c_runtime_source_provenance.py"
OUTPUT_PATH="reports/longterm_o0c_runtime_source_provenance_preflight.json"
EXPECTED_SCRIPT_SHA256="73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc"
EXPECTED_SCRIPT_BYTES="32034"
EXPECTED_SCHEMA_VERSION="o0c_runtime_source_provenance_preflight_v1"
EXPECTED_PASS_STATUS="PASS_SOURCE_IDENTITY_FROZEN"

actual_commit="$(git rev-parse HEAD)"
if [ "${actual_commit}" != "${EXPECTED_COMMIT}" ]; then
  echo "BLOCKED_COMMIT_MISMATCH expected=${EXPECTED_COMMIT} actual=${actual_commit}"
  exit 1
fi

status_porcelain="$(git status --porcelain)"
if [ -n "${status_porcelain}" ]; then
  echo "BLOCKED_REPOSITORY_STATE_DIRTY"
  git status --short
  exit 1
fi

if [ -e "${OUTPUT_PATH}" ]; then
  echo "preflight_status=BLOCKED_OUTPUT_COLLISION"
  echo "output=${OUTPUT_PATH}"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import hashlib
import sys

path = Path("scripts/preflight_longterm_o0c_runtime_source_provenance.py")
data = path.read_bytes()
sha = hashlib.sha256(data).hexdigest()
size = len(data)
print(f"script_sha256={sha}")
print(f"script_bytes={size}")
if sha != "73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc":
    print("BLOCKED_SCRIPT_SHA256_MISMATCH", file=sys.stderr)
    raise SystemExit(1)
if size != 32034:
    print("BLOCKED_SCRIPT_BYTES_MISMATCH", file=sys.stderr)
    raise SystemExit(1)
PY

python "${SCRIPT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --expected-python 3.12.13 \
  --expected-numpy 2.0.2 \
  --expected-torch 2.10.0+cpu \
  --expected-transformers 5.0.0

if [ ! -f "${OUTPUT_PATH}" ]; then
  echo "BLOCKED_PASS_ARTIFACT_MISSING output=${OUTPUT_PATH}"
  exit 1
fi

python - <<'PY'
from pathlib import Path
import hashlib
import json
import sys

path = Path("reports/longterm_o0c_runtime_source_provenance_preflight.json")
data = path.read_bytes()
if not data.endswith(b"\n"):
    print("BLOCKED_ARTIFACT_FINAL_LF_MISSING", file=sys.stderr)
    raise SystemExit(1)
artifact = json.loads(data.decode("utf-8"))
if artifact.get("schema_version") != "o0c_runtime_source_provenance_preflight_v1":
    print("BLOCKED_ARTIFACT_SCHEMA_VERSION_MISMATCH", file=sys.stderr)
    raise SystemExit(1)
if artifact.get("preflight_status") != "PASS_SOURCE_IDENTITY_FROZEN":
    print("BLOCKED_ARTIFACT_STATUS_NOT_PASS", file=sys.stderr)
    raise SystemExit(1)
runtime = artifact.get("runtime")
if runtime != {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "torch": "2.10.0+cpu",
    "transformers": "5.0.0",
}:
    print("BLOCKED_ARTIFACT_RUNTIME_MISMATCH", file=sys.stderr)
    raise SystemExit(1)
for key in ("timestamp", "hostname", "username", "uuid", "branch"):
    if key in data.decode("utf-8").lower():
        print(f"BLOCKED_ARTIFACT_FORBIDDEN_FIELD key={key}", file=sys.stderr)
        raise SystemExit(1)
required = [
    "backend_static_classification",
    "cache_source",
    "expected_runtime",
    "mamba_source",
    "o0c_full_sequence_capture_feasibility",
    "o0c_state_indexing_compatibility",
    "source_resolution",
    "symbol_locations",
]
missing = [key for key in required if key not in artifact or artifact[key] in (None, {}, "")]
if missing:
    print(f"BLOCKED_ARTIFACT_REQUIRED_FIELDS_MISSING fields={','.join(missing)}", file=sys.stderr)
    raise SystemExit(1)
print(f"artifact_sha256={hashlib.sha256(data).hexdigest()}")
print(f"artifact_bytes={len(data)}")
PY
```
END_EXACT_COMMAND

Normative command identity for `cm run save` is the exact command text inside the fenced block above after removing only the Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF, consistent with the repository's established `BEGIN_EXACT_COMMAND`/`END_EXACT_COMMAND` convention. If `cm run save` records any different command bytes, command SHA, run name, or commit, the future preflight is not authorized.

## 10. Command And Provenance Binding

The frozen command enforces or records:

- exact repository commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- clean repository state before execution;
- exact script SHA256 `73ffff2681928389933867caa9586b3f54260b2f64bdb1a44c50b77e4accdbfc`;
- exact script byte count `32034`;
- output nonexistence before execution;
- exact CLI expected runtime versions;
- nonzero exit propagation;
- output existence only after successful `PASS_SOURCE_IDENTITY_FROZEN`;
- artifact SHA256 and byte count printed after success.

The command contains no `pip`, `conda`, package installation/removal/upgrade/downgrade, package repair, model/tokenizer loading, dataset loading, CUDA enablement, optional-kernel installation, or hidden package mutation.

## 11. Forbidden Model/Tokenizer Boundary

The future execution authority explicitly prohibits:

- `AutoTokenizer.from_pretrained`;
- `MambaModel.from_pretrained`;
- `AutoModel*.from_pretrained`;
- pretrained weights;
- tokenizer invocation;
- model forward;
- generation;
- dataset reading.

This is installed source-provenance inspection only.

## 12. Forbidden Package/Environment Mutation Boundary

The future execution authority explicitly prohibits:

- `pip install`;
- `pip uninstall`;
- `conda`;
- package upgrade/downgrade;
- extension compilation;
- optional Mamba kernel installation;
- CUDA initialization;
- editing `site-packages`;
- environment manipulation to force favorable backend selection.

Observed runtime is evidence. Do not repair it.

## 13. PASS Interpretation

`PASS_SOURCE_IDENTITY_FROZEN` means only:

- exact expected runtime versions matched;
- exact installed Transformers source identity was frozen;
- required source files/hashes were resolved;
- source-root provenance reconciled;
- required static symbol/source semantics were sufficient;
- backend/path static requirements were satisfied according to frozen logic.

It does not mean:

- O0c instrumentation implemented;
- instrumentation non-interference proven;
- recurrent-state capture succeeded;
- scientific precursor finding replicated;
- O0c scientific PASS.

## 14. BLOCKED Interpretation

Any `BLOCKED_*` status is a valid preflight result and must stop progression. It must not be converted into an automatic rerun requirement.

Frozen statuses to preserve include at minimum:

- `BLOCKED_RUNTIME_VERSION_MISMATCH`
- `BLOCKED_SOURCE_FILE_UNRESOLVED`
- `BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED`
- `BLOCKED_SOURCE_HASH_UNAVAILABLE`
- `BLOCKED_TRANSFORMERS_SOURCE_SHADOWING`
- `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`
- `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`
- `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`
- `BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE`
- `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`
- `BLOCKED_BACKEND_PATH_UNRESOLVED`
- `BLOCKED_O0C_INDEXING_INCOMPATIBLE`
- `BLOCKED_OUTPUT_COLLISION`

The implementation also preserves:

- `BLOCKED_RUNTIME_VERSION_UNAVAILABLE`
- `BLOCKED_ARTIFACT_SERIALIZATION_NONDETERMINISTIC`
- `BLOCKED_FORBIDDEN_MODEL_TOKENIZER_INVOCATION`
- `BLOCKED_FORBIDDEN_PACKAGE_MUTATION`
- `BLOCKED_IMPLEMENTATION_SCOPE_WOULD_WIDEN`

The exact Transformers `5.0.0` source has not yet been observed through this canonical preflight. It is therefore scientifically and provenance-correct for the future run to return `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` or `BLOCKED_BACKEND_PATH_UNRESOLVED` if the exact installed source structure cannot satisfy the frozen static proof. The implementation and command must not be weakened or rerun with altered rules merely to obtain PASS.

## 15. Blocked-Artifact Behavior

Source inspection confirms blocked paths do not publish the canonical JSON artifact.

The frozen implementation constructs and publishes the JSON only after runtime checks, source resolution, source hashing/parsing, recurrent semantics classification, backend classification, and symbol binding all succeed. `main()` catches `PreflightBlocked`, prints:

```text
preflight_status=<BLOCKED_STATUS>
output=<output path>
blocker=<blocker note>
```

and returns exit code `2` before artifact publication.

Therefore:

- BLOCKED exit means no canonical PASS artifact is expected;
- stdout/stderr and `cm` wrapper logs must be preserved as execution evidence;
- no JSON artifact may be manually synthesized;
- collection/reporting must distinguish BLOCKED-with-no-artifact from command transport failure, notebook/session failure, Python crash, or other infrastructure crash.

## 16. Success Artifact Validation Contract

If `PASS_SOURCE_IDENTITY_FROZEN` occurs, immediately validate:

- artifact exists exactly at `reports/longterm_o0c_runtime_source_provenance_preflight.json`;
- `schema_version` exactly equals `o0c_runtime_source_provenance_preflight_v1`;
- `preflight_status` exactly equals `PASS_SOURCE_IDENTITY_FROZEN`;
- artifact SHA256 is recorded;
- artifact byte count is recorded;
- final LF exists;
- JSON parses deterministically as UTF-8;
- no forbidden nondeterministic fields exist;
- recorded runtime exactly equals Python `3.12.13`, NumPy `2.0.2`, torch `2.10.0+cpu`, Transformers `5.0.0`;
- recorded source paths/hashes are populated;
- source compatibility classifications are populated;
- required symbol locations are populated.

No scientific interpretation beyond provenance is authorized.

## 17. Rerun Policy

Do not automatically rerun a BLOCKED preflight.

A rerun is permitted only for:

- clearly transient execution-infrastructure failure;
- command transport failure;
- notebook/session failure before script semantics executed.

A semantic/provenance `BLOCKED_*` status requires controller review and potentially a separate recovery authority. Packages must not be mutated between attempts.

## 18. cm Run/Save Workflow

The later controller may instruct the user, in this exact order:

```text
cm kaggle
```

Then copy exactly one frozen shell command from Section 9 into Kaggle.

After successful command execution:

```text
cm run save longterm-o0c-runtime-source-provenance-preflight-de874a2-v1
cm run longterm-o0c-runtime-source-provenance-preflight-de874a2-v1
```

This order follows the repository's established `cm kaggle`, exact command, `cm run save`, `cm run` pattern. This authoring task does not run any of these commands.

## 19. Collection And Import Workflow

After execution evidence is saved under the frozen run name, a later controller may separately authorize:

```text
cm collect longterm-o0c-runtime-source-provenance-preflight-de874a2-v1
```

Then the user runs the collector in Kaggle, downloads the handoff ZIP, and performs:

```text
cm import <handoff.zip>
```

This authoring task does not collect or import.

Fail-closed provenance checks must block import or interpretation on:

- commit mismatch;
- script mismatch;
- command mismatch;
- run-name mismatch;
- artifact mismatch.

For a semantic/provenance BLOCKED result with no canonical JSON artifact, collection/import handling must preserve wrapper evidence and logs and must not impersonate a PASS artifact.

## 20. Post-Import Authority Boundary

After validated import:

- if `PASS_SOURCE_IDENTITY_FROZEN`, the next possible stage is a separate O0c recurrent-state instrumentation implementation authority informed by the exact source artifact;
- if `BLOCKED_*`, the next possible stage is a narrowly scoped recovery authority addressing the exact blocker.

This candidate does not pre-authorize either branch.

## 21. No Scientific Claim

This preflight execution is provenance/infrastructure evidence only.

It must not be counted as:

- O0c experiment;
- O0b replication;
- scientific sample;
- model evaluation;
- evidence of hallucination mechanism;
- evidence of semantic ownership.

## 22. Protected State And Non-Authorizations

This candidate authorizes exactly one new report file:

`reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md`

It does not authorize modifying:

- frozen implementation;
- tests;
- prior authorities;
- O0b/O0c reports;
- `cm` tooling;
- Kaggle notebooks;
- reason-router/URP state;
- protected temporary directories;
- root patch files;
- stage180 artifacts.

Protected unrelated state includes:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`
- `reports/stage180a_pass2_annotations_completed.csv`
- unrelated root patch / URP / reason-router state

## 23. Candidate Validation Contract

Required validation after authoring:

```powershell
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
```

Also compute and report:

- candidate SHA256 and byte count;
- frozen script SHA256 and byte count;
- exact task-attributable delta;
- staged state.

Expected task-attributable delta:

```text
A reports/longterm_o0c_runtime_source_provenance_preflight_execution_authority_spec_candidate.md
```

Nothing may be staged, committed, or pushed.

## 24. Explicit Non-Execution Attestation

NO PREFLIGHT EXECUTION.

NO MODEL TOKENIZER LOADING.

NO TOKENIZER INVOCATION.

NO PRETRAINED MODEL LOADING.

NO MODEL FORWARD.

NO GENERATION.

NO TRAINING.

NO EVALUATION.

NO KAGGLE.

NO `cm run`.

NO PACKAGE INSTALL.

NO PACKAGE UNINSTALL.

NO PACKAGE UPGRADE OR DOWNGRADE.

NO OPTIONAL KERNEL ENABLEMENT.

NO ENVIRONMENT MUTATION.

NO IMPLEMENTATION.

NO COMMIT.

NO PUSH.

## 25. Next Authorized Action

The exact next authorized action is independent verification of this candidate's exact bytes and authority sufficiency.

Only after independent verification and controller activation may the future run be registered/executed under the frozen run name and exact command.
