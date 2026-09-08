# ContraMamba O0c cache-guard role-selection corrected-preflight execution authority candidate

## 1. Status and purpose

`PASS_READY_FOR_FORMAL_FREEZE_CACHE_GUARD_ROLE_SELECTION_CORRECTED_PREFLIGHT_EXECUTION_AUTHORITY`

This is a report-only execution-authority candidate under the current controller instruction.

It does not itself register or execute a run. It authorizes, only after formal freeze and remote verification, exactly one future CPU-only runtime-source-provenance preflight using the cache-guard role-selection corrected validator implementation frozen at the exact commit below.

The authority is result-neutral. It does not predict PASS and does not establish any scientific conclusion.

No training, evaluation, model execution, Kaggle execution, run registration, collection, import, staging, commit, or push is authorized by this authoring step.

`SCIENTIFIC_CONCLUSION: NONE`.

## 2. Frozen authority and evidence chain

| Authority / evidence | Frozen identity |
| --- | --- |
| Cache-guard role-selection corrected implementation | `0063254795aa21011364833c95d25cbce262c0bf` |
| Cache-guard role-selection correction implementation authority | `1e7630c70a3a7cd85caa64128d89c241ed8b0960` |
| Cache-guard ambiguity root-cause interpretation | `6fe942ae7d872314d4fd4da2c68ca221e7f45b0e` |
| Linkage-ambiguity diagnostic execution authority | `b20857b78af86d814b48db9c9846a3c8bb049d61` |
| Earlier corrected-preflight execution authority | `755987eea6230bb0ad6f73400e46ae680434e6f6` |
| Earlier corrected convolution-cache validator implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Earlier validator correction implementation authority | `3f03e5dec1faf2edb443dff10387f02350b02b6f` |
| Earlier convolution-cache root-cause interpretation | `c4ec40fc8e4df82243c2facb810146513ec97b55` |

The sole future execution commit authorized by this candidate is:

`0063254795aa21011364833c95d25cbce262c0bf`

No parent, descendant, dirty variant, local-only variant, or alternate commit may substitute for this execution commit.

## 3. Frozen implementation identity

At commit `0063254795aa21011364833c95d25cbce262c0bf`:

| Role | Path | Git blob | Canonical LF SHA256 | Bytes | LF | CR | Final LF | Blank line at EOF |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Production | `scripts/preflight_longterm_o0c_runtime_source_provenance.py` | `86f2cf3e942f1429c2a60c3c1358e188e09bf82f` | `34ed1663fb7b6081f42e4e4fe2a3c4c42791aaeff8910f133a3218ed14f6d62a` | 42734 | 1098 | 0 | true | false |
| Tests | `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` | `c936009843603b9bf5cf179439f04ccf55c8da4e` | `10bc240f17c07405c5d01bc5d7064ee15e2689f20cb01272933aea8514277525` | 55692 | 1342 | 0 | true | false |

Independent implementation verdict:

`PASS_SAFE_TO_FREEZE_CACHE_GUARD_ROLE_SELECTION_IMPLEMENTATION`

Frozen targeted local-suite evidence:

`87 passed, 3 skipped`

The first independent verification attempt was blocked only by Windows pytest temporary-directory ACL failure before affected test bodies ran. An explicit external `--basetemp` recovery then produced the passing result above without modifying repository files or semantic content.

`git diff --check` passed.

These facts establish code correctness only. They are not corrected-preflight execution evidence.

## 4. Frozen correction semantics

The latest frozen infrastructure root cause is:

Primary:

`VALIDATOR_CONVOLUTION_CACHE_CACHE_PRESENT_BRANCH_PREDICATE_OVERAPPROXIMATION`

Secondary:

`VALIDATOR_CONVOLUTION_CACHE_BRANCH_ROLE_DISAMBIGUATION_MISSING`

The earlier corrected validator at `6f394792763abb168f49c1cb1957a326d16eed2b` enumerated every same-lexical-scope `cache_params is not None` branch and required broad cache-present branch uniqueness before applying convolution-cache role evidence.

On the validated Transformers-5.0.0 Mamba source, `MambaMixer.slow_forward` contains:

- one cache-present branch owning the convolution-cache prefill/decode behavior;
- one unrelated later cache-present guard that persists recurrent `ssm_state` into `cache_params.ssm_states`.

That broad predicate count was two and produced the false:

`BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`

with note:

`convolution_cache_initialization_update`.

The frozen implementation at `0063254795aa21011364833c95d25cbce262c0bf` moves uniqueness to the complete convolution-cache semantic-proof level.

For the semantic-family path it:

- enumerates same-lexical-scope cache-present branches;
- inspects prefill/decode splits within each branch;
- treats a branch/split pair as complete only when the split supplies:
  - prefill convolution-state construction;
  - at least one prefill `cache_params.update_conv_state` call;
  - at least one decode `cache_params.update_conv_state` call;
- returns unresolved for zero complete proofs;
- returns ambiguous for more than one complete proof;
- after exactly one complete proof, preserves the pre-existing uniquely linked mutation-proven `update_conv_state` method requirement;
- preserves canonical `MambaMixer.slow_forward` symbol-location anchoring;
- preserves the legacy direct-single-assignment compatibility path.

No source-position heuristic, hard-coded line number, first-branch choice, recurrent-guard special case, schema widening, public blocker addition, runtime execution, or model-based branch resolution is part of the frozen correction.

## 5. Consumed historical runs

The following exact run identities are permanently consumed historical evidence and must not be rerun, reused, overwritten, or aliased:

- `longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1`
- `longterm-o0c-convolution-cache-initialization-update-diagnostic-eebf4da-v1`
- `longterm-o0c-runtime-source-provenance-preflight-6f39479-v1`
- `longterm-o0c-convolution-cache-linkage-ambiguity-diagnostic-6f39479-v1`

The consumed corrected-preflight run at `6f394792...` completed with:

- exit code `2`;
- `preflight_status=BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`;
- blocker `convolution_cache_initialization_update`.

Its execution, collection, handoff, and local import were provenance-valid.

The consumed linkage-ambiguity diagnostic then established:

- historical Mamba source identity unchanged;
- broad cache-present branch count `2`;
- prefill/decode split count `1`;
- raw linked update-method candidate count `1`;
- semantic-deduplicated linked candidate count `1`;
- frozen validator replay still `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`.

Formal interpretation froze the cache-present branch-role selection defect subsequently corrected at `0063254795aa21011364833c95d25cbce262c0bf`.

None of these historical runs is the future execution authorized here.

## 6. Sole reserved future run name

This candidate reserves exactly one new run name:

`longterm-o0c-runtime-source-provenance-preflight-0063254-v1`

No fallback or alternate name is authorized by this candidate.

Repository search at authoring time found no exact occurrence of this name.

Before registration/execution, collision checks must be repeated against all available relevant surfaces, including:

- repository text/history;
- local `cm` run registry/state;
- local imports;
- downloaded/imported handoffs;
- accessible run metadata.

Any exact-name collision blocks registration and execution. Do not silently select a replacement name.

## 7. Exact future execution guard

Immediately before registration/execution require all of:

- local execution HEAD exactly `0063254795aa21011364833c95d25cbce262c0bf`;
- remote `main` exactly `0063254795aa21011364833c95d25cbce262c0bf`, unless a later authority explicitly freezes another ref relationship;
- clean execution worktree;
- no staged path;
- no tracked modification;
- no untracked execution-affecting path;
- exact production/test Git blobs and canonical LF identities from section 3;
- no reserved-name collision;
- no output-path collision;
- no runner/provenance/hash guard failure.

Any mismatch is a blocker.

The earlier execution worktree at `6f394792...` is historical only and must not be reused for this run.

## 8. Result-neutral allowed outcomes

The future run exists only to observe what the formally frozen corrected validator reports against the actually resolved runtime/source environment.

Valid result classes include, without prediction:

1. Exit `0` with `PASS_SOURCE_IDENTITY_FROZEN` and a produced preflight artifact.
2. A fail-closed existing validator blocker.
3. A newly exposed later blocker after convolution-cache binding resolves.
4. Runtime/environment mismatch.
5. Runtime source identity drift.
6. Command transport, Kaggle, runner, collector, handoff, import, or provenance failure.

No outcome authorizes automatic code repair, ad hoc rerun, alternate run-name selection, scientific execution, or scientific interpretation.

## 9. Frozen CLI and runtime boundary

The future execution may invoke only:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

from exact execution commit:

`0063254795aa21011364833c95d25cbce262c0bf`

Use only the existing CLI arguments:

```text
--output
--expected-python 3.12.13
--expected-numpy 2.0.2
--expected-torch 2.10.0+cpu
--expected-transformers 5.0.0
```

Required runtime boundary:

- Python `3.12.13`
- NumPy `2.0.2`
- torch `2.10.0+cpu`
- Transformers `5.0.0`
- CUDA available `False`
- CUDA device count `0`
- Kaggle Accelerator `None`
- GPU `OFF`

Runtime mismatch must fail through the frozen validator or execution guard. GPU use is not authorized.

The exact executable shell command and command SHA256 are deliberately not frozen in this report. They may be generated/frozen only after this authority is formally frozen, pushed, remote-verified, and `cm kaggle` is run for the exact implementation commit.

## 10. Historical runtime-source identity and drift handling

Historical validated Mamba source identity:

| Field | Historical value |
| --- | --- |
| Path | `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py` |
| SHA256 | `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83` |
| Bytes | `39500` |
| LF | `860` |
| CR | `0` |
| Final LF | `true` |
| Distribution/import root | `/usr/local/lib/python3.12/dist-packages/transformers` |

Historical validated cache-utils source identity from the imported diagnostic:

| Field | Historical value |
| --- | --- |
| Path | `/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py` |
| SHA256 | `6c123bbe3d23500462f0b617119a8231aa054b6d5475b295443a45f34466e6bc` |
| Bytes | `60432` |
| LF | `1295` |
| CR | `0` |
| Final LF | `true` |

The frozen CLI must independently resolve and record actual runtime/source identities.

This authority does not assume that source identity must remain unchanged. However, if the imported execution materially differs from the historically diagnosed source environment, controller interpretation must record:

`RUNTIME_SOURCE_IDENTITY_DRIFT`

and STOP before claiming that this role-selection correction was exercised against the same source previously diagnosed.

Source drift does not authorize repair or rerun.

## 11. No-model / no-science boundary

This corrected preflight remains provenance/static-source infrastructure only.

It must not:

- load a model;
- load a tokenizer;
- load a dataset;
- perform tensor/model forward;
- perform generation;
- train;
- evaluate;
- execute optional kernels as a scientific test;
- mutate installed packages.

If the exact frozen CLI would cross this boundary, execution is blocked.

Even a fully provenance-valid `PASS_SOURCE_IDENTITY_FROZEN` establishes corrected-preflight infrastructure status only.

It does not authorize O0c scientific execution, model forward, training, evaluation, or scientific interpretation.

`SCIENTIFIC_CONCLUSION: NONE`.

## 12. Kaggle and command-transport boundary

This report authoring step does not authorize Kaggle execution.

Only after this authority is:

1. materialized;
2. statically reviewed;
3. formally frozen;
4. committed/pushed;
5. independently remote-verified;

may the controller move to:

`cm kaggle`

for the exact execution commit `0063254795aa21011364833c95d25cbce262c0bf`.

The Kaggle session must remain CPU-only with Accelerator None and GPU OFF.

After `cm kaggle` supplies the exact executable command:

- freeze exactly one shell command;
- freeze its exact command SHA256 using the `cm` workflow's prescribed byte convention;
- copy only that command;
- preserve exact clipboard command identity through `cm run save`;
- fail closed on command/hash/HEAD/dirty mismatch.

Then, and only then, the authorized registration/execution sequence is:

```text
cm run save longterm-o0c-runtime-source-provenance-preflight-0063254-v1
cm run longterm-o0c-runtime-source-provenance-preflight-0063254-v1
```

The run must bind exact run name, execution commit, command SHA, timestamps, exit code, log/meta paths, and output path.

Existing output collision is a blocker. Never delete or overwrite an output merely to force execution.

## 13. Collection and import requirement

After the one authorized run completes, collect with:

```text
cm collect longterm-o0c-runtime-source-provenance-preflight-0063254-v1
```

Run the generated collector command in the same relevant Kaggle session.

Download the handoff ZIP.

Then locally run:

`cm import <handoff.zip>`

The result is not validated evidence until import returns PASS.

Require provenance verification of at least:

- run name;
- execution commit;
- command SHA;
- start/finish timestamps;
- exit code;
- run-log SHA;
- run-meta SHA;
- handoff ZIP SHA;
- collected file set/count;
- copied/identical status;
- output artifact identity if produced;
- actual runtime versions;
- actual source identities;
- backend classification;
- recurrent/O0c classification;
- convolution-cache symbol binding outcome;
- preflight status/blocker.

Any provenance mismatch blocks interpretation.

## 14. Conditional PASS semantics

PASS is not predicted.

If, and only if, the validated imported result reports:

`preflight_status: PASS_SOURCE_IDENTITY_FROZEN`

then verify that the artifact also supports all existing frozen requirements, including as applicable:

- `backend_static_classification: BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`;
- recurrent/O0c convention `SOURCE_SUPPORTS_O0C_CONVENTION`;
- valid source-resolution/provenance record;
- valid `convolution_cache_initialization_update` symbol location;
- unchanged artifact/location schema.

For the corrected semantic-family path on the historically diagnosed Transformers-5.0.0 source, the convolution-cache family location is expected to anchor to:

`qualname=MambaMixer.slow_forward`

with:

`source_file_key=mamba`

and the actually resolved Mamba source identity recorded by the validator.

This is infrastructure validation only.

## 15. Failure and recovery

For any failure involving:

- command transport;
- registration;
- environment/runtime;
- source drift;
- output collision;
- Kaggle infrastructure;
- runner metadata;
- collector/handoff;
- import;
- provenance mismatch;

STOP.

Do not automatically rerun the consumed name.

Do not modify code or create an ad hoc workaround.

Use the frozen failure-recovery workflow and require a separate recovery authority/new run identity where applicable.

A validator blocker likewise does not automatically authorize repair or rerun.

## 16. Evidence-layer separation

Maintain separate states:

| Layer | State at authority authoring |
| --- | --- |
| A. Cache-guard role-selection corrected implementation correctness | PASS / frozen at `0063254795aa21011364833c95d25cbce262c0bf` |
| B. New corrected-preflight execution | `NOT_YET_EXECUTED` |
| C. Artifact/provenance validity | `NOT_YET_ESTABLISHED` |
| D. Resulting corrected-preflight status | `UNKNOWN` |
| E. Scientific conclusion | `NONE` |

A successful process exit alone does not collapse B, C, D, and E.

## 17. Formal-freeze and execution boundary

Before this candidate can authorize `cm kaggle`, it must be formally frozen as a repository commit and pushed/remote-verified.

During candidate materialization/freeze preparation:

- no preflight execution;
- no `cm kaggle`;
- no run registration;
- no `cm run`;
- no collection/import;
- no model execution;
- no training/evaluation.

After formal freeze, the controller may prepare the exact Kaggle command for the one reserved run, subject to all guards in this authority.

## 18. Exact next action

The immediate next action after materializing this candidate is static verification and formal-freeze preparation only.

No execution is authorized until the candidate itself is frozen and remote-verified.
