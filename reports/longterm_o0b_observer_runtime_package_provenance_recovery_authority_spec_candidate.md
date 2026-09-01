# Longterm O0b Observer Runtime-Package Provenance Recovery Authority Spec Candidate

## Overall Status

PASS_READY_FOR_INDEPENDENT_PROVENANCE_RECOVERY_AUTHORITY_VERIFICATION

This document is a narrow static provenance-recovery authority candidate for one discovered production provenance defect in the frozen O0b observer. It does not reopen O0b science, measurements, coordinates, metrics, serialization, publication semantics, or any previously verified behavior.

## Authority

Authority precedence for this document:

1. Current controller instruction.
2. Frozen O0b scientific-design authority: `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
3. Frozen O0b boundary-recovery authority: `2ed4439e511f7534186cbd5df9110e45fdc1d66c`.
4. Frozen repaired input implementation: `7ce4e0cd05d87118c29526a53ab5178dc722db27`.
5. Frozen O0b observer implementation authority: `65881cf398d26b136e4984686b14f7d40b939c3e`.
6. Current observer implementation freeze: `0a3bdd80dfda84a4e1bac09e3f614e0c31362f04`.
7. Current observer SHA256: `334d6148b93e3d0fe43de94f042ffe8b6ac8e69c2bd6f29ee00b043660e1b02d`.
8. Current test SHA256: `c45a8f0c62c232beed29060484f50fd05ad97899cec67c443ab5e960ae5f7ab7`.

Phase: STATIC PROVENANCE-RECOVERY AUTHORITY AUTHORING ONLY.

## Created Path

Created exactly one new file:

`reports/longterm_o0b_observer_runtime_package_provenance_recovery_authority_spec_candidate.md`

No existing file is authorized to be modified by this authority-authoring task.

## Exact Defect Statement

The frozen production observer implementation identity is:

`0a3bdd80dfda84a4e1bac09e3f614e0c31362f04`

The current production code semantically does:

```python
versions = d.get(
    "runtime_versions",
    {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    },
)
```

then:

```python
torch_version = versions.get("torch_version", "unknown")
transformers_version = versions.get("transformers_version", "unknown")
```

Therefore ordinary CLI execution without test dependency injection publishes literal `"unknown"` for `torch_version` and `transformers_version`.

This was not caught by the completed implementation test suite. All previously verified measurement, anchor, metric, artifact, and publication behavior remains preserved. No scientific execution has occurred under O0b. No O0b scientific artifact needs invalidation because none exists yet. Implementation freeze `0a3bdd80dfda84a4e1bac09e3f614e0c31362f04` remains historical and valid as a code correctness milestone, but is not the implementation identity to authorize for scientific execution.

## Future Implementation Whitelist

A future repair is authorized to modify exactly these two files:

1. `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`
2. `tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

No third file is authorized.

The repair may change only runtime-package provenance capture/validation and tests directly required for that repair.

The repair must not change:

- scientific question;
- dataset;
- validation artifact;
- model/tokenizer IDs or revision;
- `trust_remote_code` policy;
- CPU/float32;
- 12-forward semantics;
- layer extraction;
- comparison/anchor ownership;
- pre-divergence invariant;
- metric path;
- vector indexing;
- observation schema;
- distance schema;
- summary schema;
- canonical JSON/JSONL/report;
- NPY/NPZ/ZIP semantics;
- checksums;
- publication gate;
- run-name semantics;
- exact-command semantics;
- runtime HEAD/script-SHA verification.

## Production Four-Version Capture Contract

The repaired ordinary production path must obtain actual runtime versions for all four mandatory manifest fields:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

No scientific CLI execution may depend on `_test_dependencies` or another test-only injection to populate those fields.

This authority authorizes a dedicated production helper equivalent in semantics to:

```python
capture_runtime_versions() -> {
    "python_version": <actual running Python version>,
    "numpy_version": <actual imported NumPy version>,
    "torch_version": <actual imported PyTorch version>,
    "transformers_version": <actual imported Transformers version>,
}
```

Preferred exact sources:

- `python_version`: `sys.version.split()[0]`
- `numpy_version`: `np.__version__`
- `torch_version`: `torch.__version__`
- `transformers_version`: `transformers.__version__`

Alternative APIs are permitted only if tests prove the emitted strings equal the actual runtime package versions that the execution environment exposes.

No timestamp, platform hostname, user, cwd, or machine-specific identity may be added.

## Placeholder And Unknown Fail-Closed Contract

For a publishable COMPLETE bundle, each of the four runtime-version fields must be:

- a Python `str`;
- nonempty;
- strip-stable;
- actual runtime package/version identity.

The repair must explicitly reject:

- `"unknown"`
- `"UNKNOWN"`
- `"n/a"`
- `"N/A"`
- `"none"`
- `"None"`
- empty string
- whitespace-only string
- null / `None`

The repair must not introduce a fallback placeholder. Failure to determine any required version must fail closed before valid scientific publication.

## Production And Test Separation

Tests may inject controlled runtime-version values only for synthetic verification.

Required separation:

- `main()` ordinary production CLI must use actual runtime capture.
- No CLI flag may permit overriding package versions.
- No CLI flag may enable test dependencies.
- Scientific execution authority must never use a Python import/test-injection bypass to populate provenance.
- Default production behavior must itself be complete.

## Model-Load Boundary

This correction does not authorize a new scientific model call.

Runtime-package capture itself is not scientific execution.

During future implementation verification:

- no tokenizer execution;
- no model weights;
- no Mamba forward;
- no network.

Tests may monkeypatch/fake `torch` and `transformers` module/version sources.

Production runtime capture may occur at a point where the runtime packages are already imported/available, but must not itself trigger an additional model load or forward.

## Manifest-Validation Contract

Future implementation must strengthen manifest validation so a publishable manifest fails if any of these fields is missing, non-string, empty, whitespace-only, or a forbidden placeholder:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

A fully valid controlled synthetic manifest with four concrete version strings must pass.

No other manifest key or schema changes are authorized.

## Required Test Matrix

Future tests must substantively prove at least:

1. Default production runtime-version capture returns all four exact keys.
2. Python value comes from actual/faked runtime Python source.
3. NumPy value comes from actual/faked NumPy version source.
4. Torch value comes from actual/faked torch version source.
5. Transformers value comes from actual/faked transformers version source.
6. Ordinary `run_observer` production path without `runtime_versions` injection uses the production capture helper.
7. No production CLI override for package versions exists.
8. Missing any one of four version fields fails manifest validation.
9. `None` for any one fails.
10. Empty string for any one fails.
11. Whitespace-only string for any one fails.
12. Literal `"unknown"` for `torch_version` fails.
13. Literal `"unknown"` for `transformers_version` fails.
14. Representative case/placeholder variants above fail.
15. Concrete controlled synthetic versions pass, including values equivalent to:
    - `python_version="3.x.y"`
    - `numpy_version="x.y.z"`
    - `torch_version="x.y.z+cpu"`
    - `transformers_version="x.y.z"`
16. Existing 44-test authority suite remains passing or is updated only where necessary without weakening any of the 27 prior obligations.
17. Observer import/static verification still performs no real tokenizer/model execution.

## Recovery Identity Rule

After the repair is implemented, independently verified, committed, and pushed:

- the new repair commit becomes the only O0b observer implementation identity eligible for future scientific execution authority;
- observer script SHA256 must be recomputed and frozen;
- test SHA256 must be recomputed and recorded;
- old implementation freeze `0a3bdd80dfda84a4e1bac09e3f614e0c31362f04` remains historical provenance only;
- history must not be rewritten and the old commit must not be amended.

The later O0b scientific execution authority must bind the new implementation commit/SHA, not `0a3bdd80dfda84a4e1bac09e3f614e0c31362f04`.

## Execution Remains Forbidden

This recovery authority does not authorize:

- tokenizer execution;
- model download;
- model loading;
- hidden-state forward;
- O0b artifact creation;
- Kaggle;
- scientific interpretation.

A separate scientific execution authority remains mandatory after the corrected implementation is frozen.

## Protected State

This authority does not authorize modifying or consuming unrelated URP/reason-router state, including:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/stage180a_pass2_annotations_completed.csv`
- the 75 historical root patch files.

## Static Validation To Run

Only these commands are authorized for this authority-authoring task:

```powershell
git diff --check -- reports/longterm_o0b_observer_runtime_package_provenance_recovery_authority_spec_candidate.md
git status --short
```

No execution is authorized.

## Explicit Non-Execution Attestation

NO EXISTING FILE MODIFIED

NO TOKENIZER EXECUTION

NO MODEL LOADING

NO MODEL WEIGHTS

NO HIDDEN-STATE FORWARD

NO TRAINING

NO EVALUATION

NO KAGGLE

NO COMMIT

NO PUSH

## Pass Token

LONGTERM_O0B_RUNTIME_PACKAGE_PROVENANCE_RECOVERY_AUTHORITY_PASS_READY_FOR_INDEPENDENT_VERIFICATION
