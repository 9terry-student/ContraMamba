# Longterm O0b Execution Provenance Preflight Exact-Command Recovery Authority Spec Candidate

## Status

PASS_READY_FOR_INDEPENDENT_RECOVERY_AUTHORITY_VERIFICATION

This document is a narrow static recovery-authority candidate. It resolves, at authority/specification level only, the two blockers that prevented O0b scientific execution-authority authoring against the corrected observer implementation freeze:

```text
f309d7101ff356974ac3cbb3978f4cfc23c35cf3
```

It creates no O0b scientific execution authority. It authorizes no implementation change while uncommitted, no Kaggle operation, no package preflight, no tokenizer execution, no model loading, no model weights, no hidden-state forward, no training, and no evaluation.

## 1. Authority Chain

Authority precedence for this candidate is:

1. Current controller instruction.
2. O0b scientific design: `df461469cb087f7f5db1e41a2b08e65ea517ad8a`.
3. O0b boundary recovery: `2ed4439e511f7534186cbd5df9110e45fdc1d66c`.
4. Repaired matched-control implementation: `7ce4e0cd05d87118c29526a53ab5178dc722db27`.
5. O0b observer implementation authority: `65881cf398d26b136e4984686b14f7d40b939c3e`.
6. Runtime-package provenance recovery authority: `27515b7cde33e02f992b093c70fec08d92e1b721`.
7. Corrected observer implementation freeze: `f309d7101ff356974ac3cbb3978f4cfc23c35cf3`.
8. Repository `AGENTS.md`.

O0a recovery and execution authorities may be read only as workflow precedent. They do not authorize O0b execution.

## 2. Blocker Record

Attempted O0b scientific execution-authority authoring against:

```text
f309d7101ff356974ac3cbb3978f4cfc23c35cf3
```

was correctly BLOCKED for exactly two reasons.

### A. Missing frozen package-version evidence

No immutable repository evidence currently freezes the exact intended CPU scientific-execution values for:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

The corrected observer implementation captures and rejects placeholder runtime-package provenance, but the exact intended Kaggle CPU values for a scientific execution authority have not yet been established or frozen.

### B. Ambiguous/self-referential exact-command semantics

Existing `--exact-command` semantics are insufficient for scientific provenance because:

- `verify_runtime_provenance` computes `actual_command(sys.argv)`;
- `run_observer` later stores `args.exact_command`;
- production does not compare those values;
- a naive complete-argv interpretation is self-referential because argv contains `--exact-command` and its own payload.

Therefore no O0b scientific execution authority currently exists.

## 3. Exact-Command Recovery Semantics

This candidate freezes the exact semantic definition below.

There are TWO distinct command provenance layers.

### Layer 1 - external shell command identity

Layer 1 is owned by the existing `cm` run registry / exact-byte command SHA mechanism.

It includes shell/interpreter invocation such as `python`, `-u`, wrappers, quoting, and other command-transport bytes. It remains unchanged by this recovery. It is NOT replaced by manifest `exact_command`.

### Layer 2 - observer argv identity

Layer 2 is stored in the O0b manifest field `exact_command`.

It represents the actual observer `sys.argv` after removing exactly one `--exact-command <payload>` argument pair. It is serialized canonically as a compact JSON array of strings.

The canonical operation is semantically equivalent to:

```python
def canonical_observer_argv(argv):
    require argv is a sequence of strings
    locate occurrences of "--exact-command"
    require exactly one occurrence
    require exactly one following payload value
    remove only:
        "--exact-command"
        <its following payload>
    preserve every other argv element exactly and in original order
    return json.dumps(
        remaining_argv,
        ensure_ascii=False,
        separators=(",", ":")
    )
```

The first element remains the actual `sys.argv[0]` observed by Python. The observer MUST NOT invent or prepend `python` or `-u`.

Forbidden transformations:

- no path normalization;
- no case normalization;
- no argument sorting;
- no shell reconstruction;
- no whitespace normalization inside argument values.

The supplied `--exact-command` payload MUST equal the canonical serialization computed from the actual runtime `sys.argv`. Comparison MUST be exact string equality.

The verified canonical value, not the unchecked CLI input, becomes manifest `exact_command`.

The implementation must fail closed on:

- missing `--exact-command`;
- duplicate `--exact-command`;
- missing payload;
- payload mismatch;
- malformed/non-string argv inputs where applicable;
- any attempt to infer or repair the payload.

This removes self-reference because the pair carrying the expected value is excluded from the value being represented.

## 4. Future Implementation Whitelist

After this recovery authority is independently verified and frozen, a later implementation task may modify exactly:

1. `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`
2. `tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

No third repository file is authorized.

Authorized semantic delta ONLY:

- canonical observer argv construction;
- validation of `args.exact_command` against actual runtime argv;
- propagation of the verified canonical observer argv into manifest;
- tests directly required for this behavior.

Do NOT reopen or alter the already-correct four-package runtime capture/validation repair except where plumbing is strictly necessary to preserve the manifest construction path.

If exact-command repair cannot be implemented within this exact two-file whitelist, the later implementation task must BLOCK.

## 5. Required Exact-Command Test Matrix

Future implementation tests must prove at least:

1. canonicalization preserves ordinary argv element order exactly.
2. exactly one `--exact-command` pair is removed.
3. `argv[0]` is preserved exactly.
4. no `python` or `-u` token is invented.
5. supplied payload equal to canonical observer argv passes.
6. one-character payload mismatch fails.
7. missing `--exact-command` fails.
8. duplicate `--exact-command` fails.
9. `--exact-command` without following value fails.
10. arguments after the removed pair remain in exact order.
11. arguments before the removed pair remain in exact order.
12. argument values containing spaces remain byte/text exact.
13. JSON serialization uses the exact compact canonical form.
14. `run_observer` manifest `exact_command` equals the independently computed verified canonical observer argv.
15. an arbitrary test payload such as `["python","observer.py"]` no longer passes merely because it was supplied.
16. no CLI option other than the existing `--exact-command` mechanism is added.
17. existing runtime-package provenance tests remain passing.
18. all pre-existing observer tests remain present and are not weakened.
19. import/static testing performs no real tokenizer/model/network execution.

The implementation must not solve this by deleting `exact_command` provenance or by trusting the CLI payload.

## 6. Package-Version Preflight Design

The package-version blocker is NOT resolved by guessing versions and NOT resolved by recording whatever happens during the scientific run.

The scientific execution authority must later freeze exact values for:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

Those values must be established by a dedicated CPU-only environment preflight against the intended Kaggle scientific runtime.

THIS RECOVERY AUTHORITY DOES NOT YET AUTHORIZE THAT PREFLIGHT TO RUN.

The required sequence is:

1. freeze this recovery authority;
2. implement exact-command repair;
3. independently verify repair;
4. commit/push the repaired observer;
5. controller records the new full implementation commit and committed observer SHA256;
6. ONLY THEN author an exact CPU package-version preflight authority bound to that final implementation identity;
7. freeze that preflight authority;
8. perform the authorized package-only preflight;
9. use validated preflight evidence to author the final O0b scientific execution authority.

This ordering prevents environment evidence from being attached to a superseded observer implementation identity.

## 7. Future Package Preflight Boundary

This section describes, but does not activate, the later preflight contract.

The later preflight must be:

- Kaggle CPU only;
- GPU OFF;
- no tokenizer invocation;
- no model loading;
- no model weights;
- no model forward;
- no dataset regeneration;
- no scientific artifact generation;
- no training/evaluation.

It may import only what is necessary to obtain exact version identity, including:

- `sys`
- `numpy`
- `torch`
- `transformers`

It must record exact concrete strings for all four packages/runtime values. No `unknown`, placeholder, empty, `None`, or inferred value is permitted.

The later preflight authority must bind:

- the then-current final O0b observer implementation commit;
- the exact committed observer SHA256;
- an exact package-preflight command;
- an exact preflight run name;
- CPU/GPU policy;
- provenance/collection requirements.

This candidate does not predict the future repair commit or its observer SHA.

## 8. Future Scientific Authority Requirement

After package preflight evidence exists, the later O0b scientific execution authority must freeze the exact four values:

- `python_version`
- `numpy_version`
- `torch_version`
- `transformers_version`

The runtime-produced manifest must match all four frozen values exactly before scientific interpretation is valid.

A mismatch is provenance invalidity even if:

- the observer exits successfully;
- all seven files exist;
- metrics look scientifically interesting.

The future execution workflow should preferably check package-version equality before model loading/run where the final execution authority can do so without another observer semantic change. At minimum, imported manifest equality against all four authority-frozen values is mandatory and fail-closed.

Do not weaken the existing observer's runtime capture or placeholder rejection.

## 9. Preserved Scientific Contract

This recovery changes no O0b scientific semantics.

Preserve:

- exact scientific question;
- dataset identity/SHA;
- validation artifact identity/SHA;
- model/tokenizer ID and immutable revision;
- CPU/float32;
- `trust_remote_code=false`;
- tokenizer `use_fast=true`;
- `add_special_tokens=false`;
- exact 12-forward semantics;
- coordinates/anchors;
- layer/state semantics;
- pre-divergence invariant;
- metrics/tolerances;
- deterministic artifacts;
- schemas;
- checksums/publication;
- interpretation boundary.

No training, evaluation, tokenizer execution, model loading, model weights, hidden-state forward, Kaggle, package preflight, or scientific execution is authorized.

## 10. Recovery Identity And Supersession

Current implementation commit:

```text
f309d7101ff356974ac3cbb3978f4cfc23c35cf3
```

remains historical once a later exact-command repair commit is frozen.

The future repaired commit becomes the only observer implementation identity eligible for subsequent O0b package preflight and scientific execution authority.

History must not be amended or rewritten.

## 11. Activation Rule

This candidate is NOT active while uncommitted.

It becomes ACTIVE recovery authority only after:

1. independent verifier PASS over these exact bytes;
2. exact candidate committed and pushed unchanged;
3. controller records the resulting full SHA as the recovery-authority freeze identity.

No post-freeze textual edit is required.

Activation authorizes only the later bounded exact-command implementation task.

Activation DOES NOT authorize:

- Kaggle;
- package preflight;
- tokenizer/model execution;
- scientific execution.

## 12. Protected State

Do not modify, delete, stage, clean, reset, or consume as task inputs:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/stage180a_pass2_annotations_completed.csv`
- 75 historical root patch files
- unrelated URP/reason-router state

This candidate-authoring task modifies no protected state.

## 13. Candidate Validation Contract

For this authoring task only, required validation is:

```text
git diff --check -- reports/longterm_o0b_execution_provenance_preflight_exact_command_recovery_authority_spec_candidate.md
git status --short
```

Confirm:

- exactly one new intended file;
- no existing tracked file modified.

Compute and report:

- candidate byte size;
- candidate SHA256.

No pytest is needed for documentation authoring.

## 14. Explicit Non-Execution Attestation

NO EXISTING FILE MODIFIED

NO PACKAGE PREFLIGHT

NO TOKENIZER EXECUTION

NO MODEL LOADING

NO MODEL WEIGHTS

NO HIDDEN-STATE FORWARD

NO TRAINING

NO EVALUATION

NO KAGGLE

NO COMMIT

NO PUSH

## 15. Next Authorized Action

The exact next authorized action is independent verifier review of these exact candidate bytes. If and only if independent verification returns PASS, the candidate may be committed and pushed unchanged for controller freeze-recording.

That future freeze would authorize only the bounded two-file exact-command implementation task described above.
