# ContraMamba O0c backend-path corrected-preflight execution authority candidate

## 1. Status and scope

This is a report-only candidate, authored under the current controller instruction. It is not itself a freeze, execution authorization, registration, command, or result. Independent read-only verification and formal freeze are mandatory before it can authorize exactly one future run.

**Candidate purpose:** authorize, only after that freeze sequence, one CPU-only, read-only runtime-source-provenance preflight using the corrected validator at the exact implementation commit below. The candidate is result-neutral: it neither predicts nor implies PASS.

No training, evaluation, Kaggle activity, run registration, execution, collection, import, package mutation, staging, commit, or push is authorized by this authoring task.

## 2. Frozen authority chain

| Authority | Frozen commit |
| --- | --- |
| `BACKEND_PATH_CORRECTED_IMPLEMENTATION` | `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc` |
| `BACKEND_PATH_CORRECTION_IMPLEMENTATION_AUTHORITY` | `7ad49dda29c55a8fefaf7b1d46a1cec351abd9ce` |
| `BACKEND_PATH_INTERPRETATION_ROOT_CAUSE` | `65ebe123832a4a15c2f5e11f81daff878c13469f` |
| `BACKEND_PATH_DIAGNOSTIC_TRANSPORT_RECOVERY_AUTHORITY` | `c67122e0b2f629d82ba1025f01dad3fd204f9b6f` |
| `BACKEND_PATH_DIAGNOSTIC_EXECUTION_AUTHORITY` | `4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c` |
| `VALIDATED_BACKEND_DIAGNOSTIC_EXECUTION_BASELINE` | `686b7457ed220e7d74ecdf41eb7ec500d24bb23f` |

The current corrected implementation is the sole future execution commit. Every older commit in this table is evidence or authority lineage only and must not be executed by the new corrected run.

## 3. Frozen implementation identity and correctness evidence

At `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc`, the independently measured implementation identities are:

| Role | Path | Git blob | Raw SHA256 | Bytes | LF | CR | Final LF |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Production | `scripts/preflight_longterm_o0c_runtime_source_provenance.py` | `96c4ab0331364ca579c53af917f265083dc9294d` | `7886ac4318e4bb0f16b1d27aa8efa599eefd23f30e5d7899457ea36b861377aa` | 36044 | 935 | 0 | true |
| Test | `tests/test_preflight_longterm_o0c_runtime_source_provenance.py` | `6cd19269a7cefe55857a8fd1e4e0a7856cb20fc8` | `e6fa8a223ab56ad7e987ae574c3130511343a7ac2ed077e2b705c7f5426ddc3a` | 43750 | 1106 | 0 | true |

Frozen independent implementation verdict: `PASS_SAFE_TO_FREEZE_BACKEND_PATH_VALIDATOR_CORRECTION_IMPLEMENTATION`.

Frozen targeted local-suite evidence: `68 passed, 3 skipped`. This is code-correctness evidence only; it is not corrected-preflight execution evidence.

## 4. Frozen correction semantics

The frozen implementation corrects only these established infrastructure defects:

- Primary: `VALIDATOR_OPTIONAL_BACKEND_REACHABILITY_OVERAPPROXIMATION`.
- Secondary: `VALIDATOR_BACKEND_PATH_PATTERN_FALSE_NEGATIVE`.

The correction scopes optional-backend analysis to CPU/non-CUDA forward reachability; recognizes a positive CUDA-gated fast-return followed by direct `slow_forward` fallthrough; preserves the legacy CPU-positive proof; preserves fail-closed optional and ambiguous paths; preserves `BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN`; and does not change schema or classification names.

`SCIENTIFIC_CONCLUSION: NONE`. No broader implementation claim is made here.

## 5. Consumed historical runs

The following runs are permanently consumed and historical only:

- `longterm-o0c-runtime-source-provenance-preflight-686b745-v5`, executed at `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`, ended `BLOCKED_BACKEND_PATH_UNRESOLVED` with `BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE`.
- `longterm-o0c-backend-path-diagnostic-686b745-v2`, whose established infrastructure diagnosis is primary `VALIDATOR_OPTIONAL_BACKEND_REACHABILITY_OVERAPPROXIMATION` and secondary `VALIDATOR_BACKEND_PATH_PATTERN_FALSE_NEGATIVE`.

Neither may be rerun, reused, overwritten, aliased, or treated as the corrected execution. Their validated provenance remains historical evidence only. The diagnostic's scientific conclusion remains `NONE`.

## 6. Sole reservation and collision state

The only name reserved by this candidate is:

`longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1`

Read-only authoring collision checks found no exact reservation-name occurrence in repository text, reachable repository history, the locally accessible `C:\Users\Home1\.contramamba\run-registry.json`, locally accessible imports, or Downloads. Relevant local `handoff` and `handoffs` directories were absent. No run was registered. This is a point-in-time authoring result, not a waiver: all relevant collision surfaces, including cm registry/state, local imports, downloaded/imported handoffs, repository records, and accessible local run metadata, must be checked again immediately before registration/execution. An exact-name collision blocks; no replacement name may be selected without a new controller decision and authority candidate.

## 7. Exact future execution guard

The future execution commit must be exactly `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc`: no parent, descendant, detached alternative, dirty worktree, or local-only variant is authorized.

Before registration or execution, require all of the following without bypass:

- exact local HEAD and exact remote commit;
- no tracked modification, no staged path, and no untracked execution-affecting path;
- exact frozen production and test identities from section 3;
- no reserved-name collision.

## 8. Result-neutral purpose and allowed outcomes

The sole purpose of the future run is to observe what the formally frozen corrected validator reports against the actual resolved runtime/source environment. It may establish only the corrected-preflight execution outcome, frozen-validator classifications or blockers, captured source/runtime identity, and a produced preflight artifact if published. It does not establish a scientific O0c result.

Without predeclaring which occurs, valid future outcomes are:

1. Exit 0 with `PASS_SOURCE_IDENTITY_FROZEN` and the frozen-validator artifact.
2. A fail-closed existing preflight blocker, including backend, recurrent, source-resolution, runtime, symbol, serialization, or output blockers.
3. A newly exposed later fail-closed blocker after backend selection resolves.
4. Runtime/environment incompatibility.
5. Resolved source/provenance differing from the historical diagnostic source.
6. Infrastructure, command-transport, Kaggle, collector, or import failure.

Outcomes 2–6 do not authorize rerun, repair, or scientific interpretation under this authority.

## 9. Frozen CLI and runtime boundary

The future execution may run only `scripts/preflight_longterm_o0c_runtime_source_provenance.py`, from the exact commit in section 7, through its existing CLI. It requires exactly these existing arguments; no argument may be invented or added:

```text
--output
--expected-python 3.12.13
--expected-numpy 2.0.2
--expected-torch 2.10.0+cpu
--expected-transformers 5.0.0
```

The exact shell command and its command SHA256 are deliberately not frozen here. They may be generated and frozen only after this authority candidate is independently verified, formally frozen, committed/pushed, remote-verified, and `cm kaggle` has run.

The required environment is CPU-only: Python `3.12.13`; NumPy `2.0.2`; torch `2.10.0+cpu`; Transformers `5.0.0`; CUDA available `False`; CUDA device count `0`; Kaggle Accelerator `None`; GPU `OFF`. Runtime-version mismatch must fail through the frozen validator. GPU use is not authorized.

## 10. Historical source identity and drift treatment

Historical Mamba source identity observed by the validated diagnostic:

| Field | Historical value |
| --- | --- |
| Path | `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py` |
| SHA256 | `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83` |
| Bytes / LF / CR / final LF | `39500` / `860` / `0` / `true` |
| Distribution/import root | `/usr/local/lib/python3.12/dist-packages/transformers` |

The frozen CLI must independently resolve and record the actual source identity. This candidate does not claim that the CLI hard-codes or rejects a different source SHA. Before treating an imported result as execution against the same analyzed source, compare its source identity with this historical diagnostic identity. If materially different, record `RUNTIME_SOURCE_IDENTITY_DRIFT` at the controller/provenance interpretation layer and STOP. Do not infer that the backend correction was validated against the same source. Drift requires separate interpretation/authority before scientific or implementation action. Additional source identities may be recorded only when independently verified.

## 11. No-model and scientific-execution boundary

The frozen preflight is provenance/static-source infrastructure. It must not load a model, tokenizer, or dataset; perform model forward or generation; train; evaluate; import or execute optional kernels as a scientific test; or mutate installed packages. If the exact frozen CLI would cross this boundary, STOP rather than widen authority.

`SCIENTIFIC_CONCLUSION: NONE`. Even a provenance-valid preflight PASS authorizes neither model forward nor O0c scientific execution, training, evaluation, or scientific interpretation. Any later O0c execution needs its own authority.

## 12. Kaggle, command transport, and output boundary

This authoring task does not authorize Kaggle. Only after independent verification, formal freeze commit, push, and independent remote verification may the controller move to `cm kaggle`; that session remains CPU-only with Accelerator None and GPU OFF.

After `cm kaggle` establishes the executable command, freeze one exact command line and its exact SHA256 using cm's prescribed encoding and terminal-LF convention. Copy only that command; do not register a later wrapper; do not permit clipboard replacement between copy and `cm run save`; and perform a pre-registration command-identity guard. Only then may the workflow use:

```text
cm run save longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1
cm run longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1
```

Require runner command-SHA verification and metadata binding the exact run name, commit, command identity, timestamps, exit code, and paths. The consumed diagnostic's command-transport failure teaches that a PowerShell wrapper or any other clipboard content must never be registered as the intended command.

The eventual exact command must use the output path supplied/approved by standard `cm kaggle` workflow or explicitly bound by the frozen command. Existing output collision is a blocker. Never delete or overwrite an output to force execution.

## 13. Collection, import, and post-import provenance

After a valid future execution and collection, locally run:

```text
cm collect longterm-o0c-runtime-source-provenance-preflight-eebf4da-v1
```

Then run the generated collector command in the same relevant Kaggle session, download the resulting handoff ZIP, and locally run `cm import <handoff.zip>`. The result cannot be treated as validated evidence until `cm import` returns PASS and provenance identities are checked.

Require at least: run name; execution commit; command SHA; start/finish timestamps; exit code; run-log SHA; run-meta SHA; handoff-ZIP SHA; collected file set/count; copied/identical status as applicable; preflight artifact raw identity if produced; actual runtime values; actual Mamba source identity; backend classification; recurrent classification; and preflight status/blocker. Any mismatch blocks interpretation.

## 14. Conditional PASS semantics and recovery

PASS is not predicted. If, and only if, a validated imported artifact reports `preflight_status: PASS_SOURCE_IDENTITY_FROZEN`, verify that it also supports, as applicable, `backend_static_classification: BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`, recurrent/O0c source convention `SOURCE_SUPPORTS_O0C_CONVENTION`, and a valid source-resolution/provenance record. This establishes only successful corrected-preflight infrastructure status.

For a failure involving command transport, registration, environment/runtime, source drift, output collision, Kaggle infrastructure, runner metadata, collector, handoff, import, or provenance mismatch: STOP. Do not automatically rerun with the same name, modify code, or create an ad hoc workaround. Apply the frozen failure-recovery rules and require a separate recovery authority and new run identity where needed.

## 15. Evidence-layer separation

Maintain these distinct states:

| Layer | State at authoring |
| --- | --- |
| A. Corrected implementation correctness | PASS and frozen at `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc` |
| B. Corrected-preflight execution | `NOT_YET_EXECUTED` |
| C. Artifact/provenance validity | `NOT_YET_ESTABLISHED` |
| D. Resulting corrected-preflight status | `UNKNOWN` |
| E. Scientific conclusion | `NONE` |

A successful run alone must not collapse B, C, D, and E.

## 16. Authoring validation, final-state requirements, and next action

During authoring, only read-only/static Git checks and read-only reservation checks are permitted. Do not run the preflight, `cm kaggle`, `cm run save`, `cm run`, `cm collect`, `cm import`, tests, package changes, staging, commit, or push.

The required final state is exact HEAD `eebf4da0207f00d993d7a6ae213e0cc17b4b1dfc`, no tracked modifications, no staged paths, and exactly this one untracked candidate path, with no temporary files. Its raw identity must be measured from final raw bytes and reported before freeze, preferably `CR=0`, final LF `true`, and blank-line-at-EOF `false`.

The exact next authorized action is independent read-only verification of this candidate. The verifier must independently confirm the authority chain, implementation identity and PASS evidence, consumed-v5/v2 preservation, v1 reservation/collision state, result-neutrality, CPU/runtime/CLI boundaries, source-drift handling, command transport, collection/import, recovery, evidence separation, no scientific authority, raw candidate identity, and exact final Git state. No execution follows from this candidate until formal freeze and the stated subsequent gates complete.
