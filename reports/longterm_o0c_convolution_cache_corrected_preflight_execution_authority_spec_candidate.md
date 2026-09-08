# ContraMamba O0c convolution-cache corrected-preflight execution authority candidate

## 1. Status and purpose

This is a report-only execution-authority candidate under the current controller instruction.

It does not itself register or execute a run. It authorizes, only after formal freeze and remote verification, exactly one future CPU-only runtime-source-provenance preflight using the corrected validator implementation frozen at the exact commit below.

The authority is result-neutral. It does not predict PASS and does not establish any scientific conclusion.

No training, evaluation, model execution, Kaggle execution, run registration, collection, import, staging, commit, or push is authorized by this authoring step.

## 2. Frozen authority chain

| Authority / evidence | Frozen identity |
| --- | --- |
| Convolution-cache validator corrected implementation | `6f394792763abb168f49c1cb1957a326d16eed2b` |
| Convolution-cache validator correction implementation authority | `3f03e5dec1faf2edb443dff10387f02350b02b6f` |
| Convolution-cache diagnostic interpretation / root-cause freeze | `c4ec40fc8e4df82243c2facb810146513ec97b55` |
| Convolution-cache diagnostic execution authority | `e7f1d8a0c38c13c8a10bbad8b489cd3012ff66e8` |
| Historical implementation under diagnosis | `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc` |

The sole future execution commit authorized by this candidate is:

`6f394792763abb168f49c1cb1957a326d16eed2b`

Older commits are authority/evidence lineage only and must not be substituted for the new execution commit.

## 3. Frozen corrected implementation identity

At commit `6f394792763abb168f49c1cb1957a326d16eed2b`:

| Role | Path | Git blob | Raw SHA256 | Bytes | LF | CR | Final LF | Blank line at EOF |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Production | `scripts/preflight_longterm_o0c_runtime_source_provenance.py` | `e9d9a7f21ae1cb27e559caed0bbe1502344f6bea` | `b498308b8d518e715be0fa64579c411b772d99e4faec5421bc052575881ba8cb` | 43071 | 1102 | 0 | true | false |
| Tests | `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` | `7d8613aa86c790954ce1ff145d7afa75708bea60` | `1501b6822646e08fd9552a2eb1ed5df4dfef8ef574931d72a404b8d6fc7f84bf` | 51854 | 1254 | 0 | true | false |

Independent implementation verdict:

`PASS_SAFE_TO_FREEZE_CONVOLUTION_CACHE_VALIDATOR_CORRECTION_IMPLEMENTATION`

Frozen targeted local-suite evidence:

`78 passed, 3 skipped`

That test result establishes code correctness only. It is not corrected-preflight execution evidence.

## 4. Frozen correction semantics

The established infrastructure root cause remains:

Primary:

`VALIDATOR_CONVOLUTION_CACHE_SYMBOL_FAMILY_PATTERN_FALSE_NEGATIVE`

Secondary:

`VALIDATOR_CONVOLUTION_CACHE_DIRECT_BODY_SCOPE_FALSE_NEGATIVE`

The frozen corrected implementation replaces the inadequate direct-only / assignment-only convolution-cache symbol-family binding with a branch-aware static AST proof while preserving the existing artifact schema.

For the semantic-family path it requires, as applicable:

- unique `MambaMixer.slow_forward`;
- structural cache-present branch;
- structural same-`If` prefill/decode split;
- prefill local convolution-state construction;
- proven `cache_params.update_conv_state` persistence in prefill;
- proven `cache_params.update_conv_state` persistence in decode;
- unique statically linked mutation-proven relevant `update_conv_state` method;
- fail-closed unresolved/ambiguous linkage;
- canonical `MambaMixer.slow_forward` symbol-location anchor.

The legacy narrow direct-single-assignment path remains supported.

Nested lexical functions/classes cannot supply assignment, method-linkage, call, or persistent-mutation proof.

No schema, runtime, source-resolution, recurrent-state, backend, CLI, serialization, or output-publication semantics were authorized to change.

`SCIENTIFIC_CONCLUSION: NONE`.

## 5. Consumed historical runs

The following run identities are permanently consumed historical evidence and must never be rerun, reused, overwritten, or aliased:

- `longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1`
- `longterm-o0c-convolution-cache-initialization-update-diagnostic-eebf4da-v1`

The first ended in the observed blocker:

`BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`

with note:

`convolution_cache_initialization_update`

The second established the frozen infrastructure root cause later corrected at `6f394792763abb168f49c1cb1957a326d16eed2b`.

Neither historical run is the new corrected execution.

## 6. Sole reserved future run name

This candidate reserves exactly one new run name:

`longterm-o0c-runtime-source-provenance-preflight-6f39479-v1`

No fallback or alternate name is authorized by this candidate.

Repository search at authoring time found no exact occurrence of this name.

Before registration/execution, collision checks must be repeated against all available relevant surfaces, including:

- repository text/history;
- local `cm` run registry/state;
- local imports;
- downloaded/imported handoffs;
- accessible run metadata.

Any exact-name collision blocks registration and execution. Do not silently choose a replacement name.

## 7. Exact future execution guard

Immediately before registration/execution require:

- local execution HEAD exactly `6f394792763abb168f49c1cb1957a326d16eed2b`;
- remote `main` exactly the same commit unless a later authority explicitly freezes another ref relationship;
- clean execution worktree: no tracked modification, no staged path, no untracked execution-affecting path;
- exact production/test raw identities and Git blobs from section 3;
- no reserved-name collision;
- no provenance/hash guard failure.

A parent, descendant, dirty variant, local-only variant, or different commit is not authorized.

## 8. Result-neutral allowed outcomes

This future run exists only to observe what the formally frozen corrected validator reports against the actual resolved runtime/source environment.

Valid result classes include, without prediction:

1. Exit `0` with `PASS_SOURCE_IDENTITY_FROZEN` and a produced preflight artifact.
2. A fail-closed existing validator blocker.
3. A newly exposed later blocker after convolution-cache binding resolves.
4. Runtime/environment mismatch.
5. Runtime source identity drift.
6. Command transport, Kaggle, runner, collector, handoff, import, or provenance failure.

Outcomes other than a validated imported result do not authorize an ad hoc rerun, repair, or scientific interpretation.

## 9. Frozen CLI and runtime boundary

The future execution may invoke only:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

from exact execution commit `6f394792763abb168f49c1cb1957a326d16eed2b`.

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

The exact executable shell command and its command SHA256 are deliberately not frozen in this report. They may be generated/frozen only after this authority is formally frozen, pushed, remote-verified, and `cm kaggle` is run for the exact implementation commit.

## 10. Historical runtime-source identity and drift handling

Historical validated Mamba source identity:

| Field | Historical value |
| --- | --- |
| Path | `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py` |
| SHA256 | `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83` |
| Bytes | `39500` |
| LF / CR | `860` / `0` |
| Final LF | `true` |
| Distribution/import root | `/usr/local/lib/python3.12/dist-packages/transformers` |

The frozen CLI must independently resolve and record the actual source identity.

This authority does not assume that the source must remain identical. However, if the imported execution reports materially different source identity, controller interpretation must record:

`RUNTIME_SOURCE_IDENTITY_DRIFT`

and STOP before claiming that the corrected validator has been exercised against the same historically diagnosed source.

Source drift does not authorize code modification, rerun, or scientific execution.

## 11. No-model / no-science boundary

The corrected preflight is provenance/static-source infrastructure only.

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

1. verified;
2. formally frozen;
3. committed/pushed;
4. independently remote-verified;

may the controller move to:

`cm kaggle`

The Kaggle session must remain CPU-only with Accelerator None and GPU OFF.

After `cm kaggle` supplies the exact executable command:

- freeze exactly one shell command;
- freeze its exact command SHA256 using the `cm` workflow's prescribed byte/terminal-LF convention;
- copy only that command;
- do not register a wrapper or unrelated clipboard content;
- preserve clipboard identity between command copy and `cm run save`;
- fail closed on command/hash/HEAD/dirty mismatch.

Then, and only then, the authorized registration/execution sequence is:

```text
cm run save longterm-o0c-runtime-source-provenance-preflight-6f39479-v1
cm run longterm-o0c-runtime-source-provenance-preflight-6f39479-v1
```

The run must bind exact run name, execution commit, command SHA, timestamps, exit code, log/meta paths, and output path.

Existing output collision is a blocker. Never delete or overwrite an output merely to force execution.

## 13. Collection and import requirement

After the one authorized run completes, collect with:

```text
cm collect longterm-o0c-runtime-source-provenance-preflight-6f39479-v1
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
- actual source identity;
- backend classification;
- recurrent/O0c classification;
- convolution-cache symbol binding outcome;
- preflight status/blocker.

Any provenance mismatch blocks interpretation.

## 14. Conditional PASS semantics

PASS is not predicted.

If, and only if, the validated imported result reports:

`preflight_status: PASS_SOURCE_IDENTITY_FROZEN`

then verify that the artifact also supports the existing frozen requirements, including as applicable:

- `backend_static_classification: BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`;
- recurrent/O0c convention `SOURCE_SUPPORTS_O0C_CONVENTION`;
- valid source-resolution/provenance record;
- valid `convolution_cache_initialization_update` symbol location;
- unchanged artifact/location schema.

For the corrected semantic-family path on the historical Transformers-5-like source, the convolution-cache family location is expected to anchor to:

`qualname=MambaMixer.slow_forward`

with `source_file_key=mamba`.

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
| A. Corrected implementation correctness | PASS / frozen at `6f394792763abb168f49c1cb1957a326d16eed2b` |
| B. Corrected-preflight execution | `NOT_YET_EXECUTED` |
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
