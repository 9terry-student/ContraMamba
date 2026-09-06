# Longterm O0c Transformers Distribution-Root Ambiguity Diagnostic Execution Authority Spec Candidate

## 1. Verdict

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This is a narrow report-only diagnostic execution authority candidate for the ContraMamba O0c Transformers distribution-root ambiguity blocker.

It authorizes exactly one future read-only, CPU-only Kaggle diagnostic after independent verification and controller activation.

This authoring task does not run Kaggle, register a run, execute a diagnostic, rerun the preflight, load a model/tokenizer, load a dataset, mutate packages, mutate the repository, stage, commit, or push.

## 2. Authority Chain

Authority order used:

1. Current workflow-controller instruction for this task.
2. Current canonical repository: `C:\Users\Home1\Desktop\ContraMamba`.
3. Expected canonical HEAD: `9397696a185344364bdb5f2dcc781e505ab5389b`.
4. Frozen O0c preflight implementation: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`.
5. Frozen preflight execution authority: `4726f2ab1540f6fe6148e89d11200ac0469f286f`.
6. Frozen command-transport recovery authority: `a9f588c00cad90050dd0e38a3b52a2b03fab98ae`.
7. Activated controller-level PowerShell hash compatibility correction committed candidate `9397696`, independently verified.
8. Imported v2 provenance for `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`.
9. Current controller static inspection finding recorded in this candidate.
10. Repository `AGENTS.md` and runbooks.

Canonical repository HEAD verified before authoring:

`9397696a185344364bdb5f2dcc781e505ab5389b`

## 3. Frozen Failure Evidence

Imported v2 provenance is preserved as a consumed semantic preflight attempt:

- run name: `longterm-o0c-runtime-source-provenance-preflight-de874a2-v2`;
- expected HEAD: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- actual HEAD: `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- command SHA256: `bd72dcc900083aea209b2e61d78145bb4c84fd9b7044a07f217578e7f8e7ef92`;
- exit code: `2`;
- preflight status: `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`;
- blocker: `distribution roots`.

The absence of a failure JSON artifact is expected frozen implementation behavior for blocked paths and is not a collector defect.

Do not rerun v2. Do not create preflight v3 under this candidate. Imported v2 provenance remains valid.

## 4. Static Hypothesis

The frozen implementation function `_distribution_root()` contains the following root derivation:

```python
root = located.parent if located.name == "__init__.py" else located.parents[len(parts) - 2]
```

This appears capable of treating nested package `__init__.py` files as distinct distribution roots when metadata entries include paths such as `transformers/models/__init__.py` or deeper package initializers.

This is only a static hypothesis. It is not a validator-defect finding. The future diagnostic must enumerate actual Kaggle-installed Transformers distribution metadata before making any root-cause classification.

## 5. Future Diagnostic Scope

The future diagnostic is authorized only as:

- read-only installed Transformers distribution metadata inspection;
- CPU-only / GPU OFF;
- Python stdlib only;
- `importlib.metadata`;
- `importlib.util` only for non-instantiating source-origin observations;
- `pathlib`;
- `hashlib` / `json` if needed;
- filesystem metadata reads.

The diagnostic must not:

- rerun the frozen preflight;
- execute O0c scientific instrumentation;
- load or instantiate any model;
- load or instantiate any tokenizer;
- load any dataset;
- perform training or evaluation;
- use network access;
- install, uninstall, upgrade, downgrade, or repair packages;
- mutate packages or `site-packages`;
- mutate the repository;
- create files inside the repository;
- alter the frozen algorithm.

The diagnostic may compare the frozen algorithm's outputs against a uniformly depth-derived package-root candidate. It must not change the frozen algorithm or prescribe a fix.

## 6. Required Diagnostic Evidence

The future diagnostic must print deterministic JSON to stdout containing at least:

1. Runtime identity:
   - Python version;
   - `transformers` distribution version;
   - distribution metadata / dist-info location if obtainable;
   - `sys.executable`.
2. Distribution entries:
   - every `importlib.metadata.distribution("transformers").files` entry whose first path component is exactly `transformers`;
   - relative distribution path;
   - path parts;
   - `distribution.locate_file(entry)` result;
   - canonical resolved path;
   - whether filename equals `__init__.py`;
   - frozen-derived root using the frozen implementation algorithm.
3. Exact distinct frozen roots:
   - number of qualifying files;
   - number of distinct frozen-derived roots;
   - each distinct root sorted deterministically;
   - count of entries producing each root;
   - first deterministic sample entry for each root.
4. Nested `__init__.py` test:
   - whether any qualifying entry has filename `__init__.py` and `len(parts) > 2`;
   - for each such entry, relative path, frozen-derived root, and uniform package-root candidate `located.parents[len(parts) - 2]`.
5. Top-level package root:
   - canonical path from `distribution.locate_file("transformers")` when obtainable;
   - `importlib.util.find_spec("transformers")` origin and parent when obtainable.
6. Source module origins, if obtainable without model/tokenizer loading:
   - `transformers`;
   - `transformers.models.mamba.modeling_mamba`;
   - `transformers.cache_utils`.

Source module origins are observation only and must not be interpreted as model/tokenizer execution.

## 7. Root-Cause Classification Criteria

The future diagnostic report must support exactly one classification.

`VALIDATOR_DISTRIBUTION_ROOT_DERIVATION_FALSE_AMBIGUITY`

Use only if actual evidence shows that multiple frozen-derived roots arise solely or materially because nested `__init__.py` entries are assigned `located.parent` instead of the common Transformers package root, while their resolved files remain within one reconciled installed Transformers package tree.

`ACTUAL_TRANSFORMERS_DISTRIBUTION_ROOT_AMBIGUITY`

Use if actual installed metadata/files genuinely resolve into multiple independent package roots even after accounting for nested `__init__.py` path depth.

`BLOCKED_ROOT_CAUSE_UNRESOLVED`

Use if evidence is insufficient, unavailable, contradictory, or mixed.

No implementation correction is authorized by this candidate.

## 8. Run Identity

Exact future diagnostic run name:

`longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`

Exact execution commit:

`de874a22df4f60adbdc5efbcf294961c7b3a48a5`

The diagnostic must be registered/executed through normal `cm run save` / `cm run` workflow so that command SHA256, expected/actual commit, run logs, metadata, command text, and stdout evidence are collectible and importable.

## 9. Exact Future Bash Command

The exact future Kaggle bash command is the command between `BEGIN_EXACT_COMMAND` and `END_EXACT_COMMAND`. This authoring task records it but does not execute it.

BEGIN_EXACT_COMMAND
```bash
set -euo pipefail

EXPECTED_COMMIT="de874a22df4f60adbdc5efbcf294961c7b3a48a5"

actual_commit="$(git rev-parse HEAD)"
if [ "${actual_commit}" != "${EXPECTED_COMMIT}" ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_COMMIT_MISMATCH\",\"expected\":\"${EXPECTED_COMMIT}\",\"actual\":\"${actual_commit}\"}"
  exit 1
fi

status_porcelain="$(GIT_OPTIONAL_LOCKS=0 git status --porcelain)"
if [ -n "${status_porcelain}" ]; then
  printf '%s\n' "{\"status\":\"BLOCKED_REPOSITORY_STATE_DIRTY\",\"details\":$(python -B -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines(), sort_keys=True))' <<< "${status_porcelain}")}"
  exit 1
fi

python -B - <<'PY'
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path


def text_path(value):
    return None if value is None else str(value)


def canonical(path):
    return str(Path(path).resolve(strict=True))


def spec_observation(name):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return {"module": name, "status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    if spec is None:
        return {"module": name, "status": "UNAVAILABLE", "error": "spec is None"}
    locations = getattr(spec, "submodule_search_locations", None)
    return {
        "module": name,
        "status": "OBSERVED",
        "origin": text_path(getattr(spec, "origin", None)),
        "submodule_search_locations": sorted(str(item) for item in locations or []),
    }


dist = importlib.metadata.distribution("transformers")
files = list(dist.files or [])

entries = []
root_counts = Counter()
samples_by_root = {}
nested_inits = []
uniform_root_counts = Counter()
qualifying = []

for file in sorted(files, key=lambda item: str(item).replace("\\", "/")):
    rel = str(file).replace("\\", "/")
    parts = tuple(getattr(file, "parts", Path(str(file)).parts))
    if not parts or parts[0] != "transformers":
        continue
    qualifying.append(rel)
    located_raw = Path(dist.locate_file(file))
    located = Path(canonical(located_raw))
    is_init = located.name == "__init__.py"
    frozen_root = located.parent if is_init else located.parents[len(parts) - 2]
    uniform_root = located.parents[len(parts) - 2]
    frozen_root_text = str(frozen_root)
    uniform_root_text = str(uniform_root)
    root_counts[frozen_root_text] += 1
    uniform_root_counts[uniform_root_text] += 1
    samples_by_root.setdefault(frozen_root_text, rel)
    entry = {
        "relative_distribution_path": rel,
        "parts": list(parts),
        "locate_file": str(located_raw),
        "resolved_path": str(located),
        "filename_is_init_py": is_init,
        "frozen_derived_root": frozen_root_text,
    }
    entries.append(entry)
    if is_init and len(parts) > 2:
        nested_inits.append({
            "relative_distribution_path": rel,
            "frozen_derived_root": frozen_root_text,
            "uniform_package_root_candidate": uniform_root_text,
        })

roots = sorted(root_counts)
uniform_roots = sorted(uniform_root_counts)

top_level_locate = None
try:
    top_level_locate = canonical(dist.locate_file("transformers"))
except Exception as exc:
    top_level_locate = f"UNAVAILABLE: {type(exc).__name__}: {exc}"

dist_info_location = None
for attr in ("_path", "path"):
    value = getattr(dist, attr, None)
    if value is not None:
        dist_info_location = str(value)
        break

payload = {
    "diagnostic_name": "longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1",
    "execution_commit": "de874a22df4f60adbdc5efbcf294961c7b3a48a5",
    "status": "DIAGNOSTIC_OBSERVATION_ONLY",
    "runtime_identity": {
        "python_version": platform.python_version(),
        "sys_executable": sys.executable,
        "transformers_distribution_version": dist.version,
        "transformers_distribution_metadata_location": dist_info_location,
    },
    "qualifying_file_count": len(qualifying),
    "distinct_frozen_root_count": len(roots),
    "distinct_frozen_roots": [
        {
            "root": root,
            "entry_count": root_counts[root],
            "first_sample_entry": samples_by_root[root],
        }
        for root in roots
    ],
    "distinct_uniform_package_root_count": len(uniform_roots),
    "distinct_uniform_package_roots": [
        {
            "root": root,
            "entry_count": uniform_root_counts[root],
        }
        for root in uniform_roots
    ],
    "nested_init_py_test": {
        "present": bool(nested_inits),
        "count": len(nested_inits),
        "entries": sorted(nested_inits, key=lambda item: item["relative_distribution_path"]),
    },
    "top_level_package_root": {
        "distribution_locate_file_transformers_resolved": top_level_locate,
        "find_spec_transformers": spec_observation("transformers"),
    },
    "source_module_origins": [
        spec_observation("transformers"),
        spec_observation("transformers.models.mamba.modeling_mamba"),
        spec_observation("transformers.cache_utils"),
    ],
    "entries": entries,
    "classification_candidates": [
        "VALIDATOR_DISTRIBUTION_ROOT_DERIVATION_FALSE_AMBIGUITY",
        "ACTUAL_TRANSFORMERS_DISTRIBUTION_ROOT_AMBIGUITY",
        "BLOCKED_ROOT_CAUSE_UNRESOLVED",
    ],
    "classification_note": "This diagnostic prints evidence only; classification must be assigned by the future diagnostic report under the authority criteria.",
}

print(json.dumps(payload, sort_keys=True, indent=2))
PY
```
END_EXACT_COMMAND

Normative command identity for `cm run save` is the exact command text inside the fenced block above after removing only the Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF, consistent with the repository's established `BEGIN_EXACT_COMMAND` / `END_EXACT_COMMAND` convention.

The command must block if independent extraction cannot reproduce the command identity recorded in Section 10.

## 10. Command Identity

Exact command SHA256:

`5d4fd534e1d0dd96ea02a1393a43aa4f0379219301e550632d549040b39e2fc1`

Exact command bytes:

`5602`

The command identity must be computed over the Section 9 command bytes after removing only the opening and closing Markdown fence lines, encoded as UTF-8 with LF line endings and no added final LF.

## 11. Provenance Workflow

After independent verification and controller activation, the normal workflow is:

```text
cm run save longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1
cm run longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1
```

The future saved run must preserve:

- run name `longterm-o0c-transformers-distribution-root-diagnostic-de874a2-v1`;
- execution commit `de874a22df4f60adbdc5efbcf294961c7b3a48a5`;
- command SHA256 from Section 10;
- run log/meta/command evidence;
- stdout diagnostic JSON.

The candidate does not authorize `cm collect` or import by itself, but the evidence is intentionally shaped so normal collection/import can preserve the diagnostic record after separate controller instruction.

## 12. V2 Disposition Preserved

The v2 preflight attempt is consumed and semantically valid as a blocked preflight attempt:

- do not rerun v2;
- do not create preflight v3 yet;
- imported provenance remains valid;
- failure JSON absence is expected implementation behavior and not a collector defect;
- the diagnostic exists only to determine why the v2 blocker `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS("distribution roots")` was emitted.

## 13. Non-Authorization Boundary

This candidate does not authorize:

- implementation correction;
- test modification;
- existing report modification;
- `cm.ps1` modification;
- run registry mutation during authoring;
- package or environment mutation;
- Kaggle execution during authoring;
- preflight rerun;
- scientific O0c execution;
- model/tokenizer loading;
- dataset loading;
- training or evaluation;
- commit or push.

## 14. Expected Delta

This task creates exactly one task-attributable file:

`reports/longterm_o0c_transformers_distribution_root_ambiguity_diagnostic_execution_authority_spec_candidate.md`

No other task-attributable repository change is authorized.

## 15. Validation Contract

After authoring, verify and report:

- candidate SHA256;
- candidate byte count;
- LF/CR/final-LF facts;
- `git diff --check`;
- `git diff --name-status`;
- `git diff --cached --name-status`;
- `git status --short`;
- exactly one task-attributable new file.

Training/evaluation run:

`NO`

Kaggle run:

`NO`

Diagnostic execution:

`NO`

Repository/package mutation beyond this single candidate file:

`NO`

## 16. Required Independent Verification Report Fields

An independent verifier should report:

1. Verdict: `PASS_READY_FOR_INDEPENDENT_VERIFICATION` or `BLOCKED`.
2. HEAD and repository state.
3. Candidate path and identity.
4. Frozen failure evidence.
5. Static hypothesis recorded only as hypothesis.
6. Exact future diagnostic scope.
7. Exact run name and execution commit.
8. Exact command SHA256 and bytes.
9. Classification criteria A/B/C.
10. Explicit no-execution / no-Kaggle / no-mutation confirmation.
11. Final git state.

## 17. Next Authorized Action

The next authorized action is independent verification of this candidate.

Only after independent verification and controller activation may the future diagnostic be registered/executed under the frozen run name and exact command.
