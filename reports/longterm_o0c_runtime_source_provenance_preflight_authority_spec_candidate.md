# ContraMamba O0c Exact-Runtime Source-Provenance Preflight Authority Spec Candidate

## 1. Overall Verdict And Phase

Overall verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This document is a single candidate authority/specification for a future O0c exact-runtime source-provenance preflight.

Phase:

`STATIC O0c RUNTIME-SOURCE PROVENANCE PREFLIGHT AUTHORITY AUTHORING ONLY`

This candidate does not run the preflight. It does not authorize O0c instrumentation implementation, tokenizer invocation, model loading, pretrained weight loading, model forward passes, generation, training, evaluation, Kaggle execution, package installation, package removal, package upgrade/downgrade, optional-kernel enabling, environment mutation, staging, committing, or pushing.

The future preflight's sole purpose is to freeze the exact installed source identity and backend semantics required before any O0c instrumentation implementation authority may be authored.

## 2. Authority Chain

Authority order used:

1. Current controller instruction for this task card.
2. Frozen O0c native-state instrumentation authority: `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`.
3. Frozen O0b scientific interpretation: `f1dc559d546d20611d66b27684bbfa0f02afa696`.
4. Frozen O0b runtime/execution provenance lineage referenced by O0c.
5. Repository `AGENTS.md`.

Canonical authoring HEAD required and verified before authoring:

`242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`

## 3. Motivation

Frozen O0c found a feasible sequential recurrent-state instrumentation surface by inspecting local Hugging Face Transformers `5.12.1` source.

Frozen O0b scientific execution used this runtime lineage:

- Python `3.12.13`
- NumPy `2.0.2`
- torch `2.10.0+cpu`
- transformers `5.0.0`

O0c explicitly forbids inferring exact Transformers `5.0.0` implementation semantics from local Transformers `5.12.1` source. Therefore the next gate must bind the exact installed runtime source that a future O0c implementation/execution would actually use.

The preferred O0c compatibility target is the validated O0b runtime lineage above. This candidate does not silently authorize a newer Transformers version. Any departure from Transformers `5.0.0` requires a documented technical incompatibility and a separate authority decision.

## 4. Exact-Runtime Source Binding Rule

Upstream GitHub tags, release notes, source distributions, or package-version strings are useful context only.

They are not substitutes for hashing the exact source files installed in the future execution runtime.

The authoritative future identity is:

```text
exact installed runtime versions
+ exact import-resolved installed source-file bytes/hashes
+ exact static symbol/backend semantic classification
```

Package version string alone is insufficient. Filename alone is insufficient. A source path without source bytes and SHA256 is insufficient.

## 5. Future Preflight Runtime Scope

The future preflight must record concrete runtime identity without loading any pretrained model or tokenizer:

- Python version
- NumPy version
- torch version
- transformers version

Preferred intended values:

```json
{
  "python_version": "3.12.13",
  "numpy_version": "2.0.2",
  "torch_version": "2.10.0+cpu",
  "transformers_version": "5.0.0"
}
```

The preflight must fail closed with `BLOCKED_RUNTIME_VERSION_MISMATCH` if the concrete runtime versions differ from the authority-frozen expected values, unless a later authority explicitly supersedes the expected runtime lineage before execution.

## 6. Future Source Identity Scope

Without loading a model or tokenizer, the future preflight must identify and bind:

- `transformers` package root.
- exact installed distribution metadata/version.
- exact path of the relevant Mamba implementation source.
- exact path of the relevant cache implementation source.

At minimum, it must inspect source identities corresponding to:

- `transformers.models.mamba.modeling_mamba`
- `transformers.cache_utils`

If Transformers `5.0.0` uses different module/file organization, the preflight must discover and report the actual import-resolved paths rather than guessing. If either authoritative source file cannot be resolved, the preflight must fail closed with `BLOCKED_SOURCE_FILE_UNRESOLVED`.

## 7. Exact Source-Byte Facts

For every authoritative source file, the future preflight must record:

- absolute import-resolved path;
- byte count;
- SHA256;
- LF count;
- CR count;
- final-LF presence;
- deterministic Git/package metadata if available without environment mutation.

Primary identity is exact source bytes. If bytes cannot be read or hashed, the preflight must fail with `BLOCKED_SOURCE_HASH_UNAVAILABLE`.

Line-ending facts are scientific provenance fields because source-byte identity can differ even when visible text appears equivalent.

## 8. Required Static Symbol Families

The future preflight must statically locate and report the exact implementation symbols or source spans responsible for:

- Mamba mixer forward dispatch;
- CPU/sequential/slow forward path;
- recurrent SSM state initialization;
- recurrent SSM state update;
- convolution/cache state initialization/update;
- cache object recurrent-state storage, if present;
- output hidden-state production;
- backend/kernel availability decision.

The preflight must use static source inspection and import metadata only. It must not execute a model forward to discover these symbols.

Each symbol report must include enough deterministic source-bound location detail to guide a later implementation authority, such as qualified symbol name, file path, and AST line range or equivalent source-span facts. If a required symbol cannot be uniquely resolved, the preflight must fail closed with a specific blocker rather than inventing an instrumentation location.

## 9. Recurrent-State Semantics To Establish

The source-bound report must determine whether the exact runtime source supports the O0c convention:

```text
s_t = post-consumption recurrent SSM state after token x_t
```

It must also determine whether the sequential full-sequence path exposes or can be instrumented to copy every per-token post-update recurrent state without requiring token-by-token model replay.

Allowed classification values:

- `SOURCE_SUPPORTS_O0C_CONVENTION`
- `SOURCE_REQUIRES_IMPLEMENTATION_REVIEW`
- `SOURCE_INCOMPATIBLE_WITH_FROZEN_O0C_DESIGN`

The preflight must not claim scientific feasibility merely from a package version. If update timing, tensor shape, state lifecycle, or full-sequence capture semantics cannot be proven statically from the exact source bytes, the artifact must record `SOURCE_REQUIRES_IMPLEMENTATION_REVIEW` and set `preflight_status` to `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` unless a later authority defines an additional tiny execution probe.

If the exact source contradicts the post-consumption indexing convention or requires token-by-token replay where O0c forbids it, the preflight must fail with `BLOCKED_O0C_INDEXING_INCOMPATIBLE` or another more specific incompatibility status.

## 10. Backend Determination Scope

The future preflight must define a fail-closed static/backend result determining:

- whether optimized Mamba kernels are available in the installed runtime;
- whether CPU execution would select a sequential/non-fast path;
- whether associative scan or another backend could alter the exact code path;
- which exact function/path a future O0c observer would need to instrument.

No package installation, compilation, kernel enabling, or environment mutation is permitted.

The preflight must distinguish:

- source contains an inspectable sequential path;
- that exact path is guaranteed to be selected under future O0c runtime.

If path selection cannot be proven without a tiny execution probe that does not load a model/tokenizer or run a forward pass, this candidate requires the preflight to record that as a separate later question. It must not guess. In that case `preflight_status` must be `BLOCKED_BACKEND_PATH_UNRESOLVED` unless a later authority authorizes the probe.

## 11. No Model/Tokenizer Boundary

The future preflight must not:

- call `AutoTokenizer.from_pretrained`;
- call `MambaModel.from_pretrained`;
- download model files;
- instantiate or load pretrained weights;
- tokenize O0b data;
- run model forward passes;
- generate text;
- train or evaluate.

Importing Python modules and reading installed source files is allowed only if it does not instantiate/download the pretrained model or mutate packages.

## 12. No Environment Mutation

The future preflight must not:

- `pip install`;
- `pip uninstall`;
- upgrade or downgrade packages;
- enable optional kernels;
- compile extensions;
- alter environment variables to force a favorable backend unless the exact variable is already part of a separately frozen runtime contract;
- modify installed package files.

Observed optional-kernel availability must be recorded as runtime evidence, not adjusted to satisfy O0c.

## 13. Deterministic Provenance Artifact

The future preflight must produce exactly one deterministic JSON source-provenance artifact as its scientific artifact. A preferred path may be defined by the later implementation authority, but it must be deterministic and repository-relative under `reports/`.

Required JSON fields:

```json
{
  "schema_version": "o0c_runtime_source_provenance_preflight_v1",
  "preflight_status": "<status>",
  "python_version": "<string>",
  "numpy_version": "<string>",
  "torch_version": "<string>",
  "transformers_version": "<string>",
  "transformers_distribution_location": "<absolute path or deterministic package metadata value>",
  "mamba_source_path": "<absolute path>",
  "mamba_source_sha256": "<64 hex chars>",
  "mamba_source_bytes": 0,
  "mamba_source_lf_count": 0,
  "mamba_source_cr_count": 0,
  "mamba_source_final_lf": true,
  "cache_source_path": "<absolute path>",
  "cache_source_sha256": "<64 hex chars>",
  "cache_source_bytes": 0,
  "cache_source_lf_count": 0,
  "cache_source_cr_count": 0,
  "cache_source_final_lf": true,
  "mamba_forward_symbol": "<qualified symbol/span>",
  "sequential_forward_symbol": "<qualified symbol/span>",
  "recurrent_state_initialization_site": "<qualified symbol/span>",
  "recurrent_state_update_site": "<qualified symbol/span>",
  "cache_state_site": "<qualified symbol/span>",
  "hidden_state_output_site": "<qualified symbol/span>",
  "backend_selection_site": "<qualified symbol/span>",
  "optimized_kernel_availability": "<deterministic static status>",
  "cpu_sequential_path_static_status": "<deterministic static status>",
  "o0c_state_indexing_compatibility": "<classification>",
  "o0c_full_sequence_capture_feasibility": "<classification>",
  "notes": []
}
```

The implementation may add deterministic fields only if a later implementation authority freezes the expanded schema before execution.

Forbidden scientific artifact fields include timestamp, hostname, username, random UUID, mutable branch name, working directory if it embeds a username, or any non-deterministic run-local identifier.

JSON serialization must be deterministic: UTF-8, sorted keys, two-space indentation, final LF, stable list ordering, and no NaN/Infinity.

## 14. Fail-Closed Status Taxonomy

The future artifact must use specific fail-closed statuses. At minimum:

- `PASS_SOURCE_IDENTITY_FROZEN`
- `BLOCKED_RUNTIME_VERSION_MISMATCH`
- `BLOCKED_SOURCE_FILE_UNRESOLVED`
- `BLOCKED_SOURCE_HASH_UNAVAILABLE`
- `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`
- `BLOCKED_BACKEND_PATH_UNRESOLVED`
- `BLOCKED_O0C_INDEXING_INCOMPATIBLE`

Recommended additional implementation statuses:

- `BLOCKED_FORBIDDEN_MODEL_TOKENIZER_INVOCATION`
- `BLOCKED_FORBIDDEN_PACKAGE_MUTATION`
- `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`
- `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`
- `BLOCKED_ARTIFACT_SERIALIZATION_NONDETERMINISTIC`
- `BLOCKED_DIRTY_OR_WRONG_HEAD`
- `BLOCKED_COMMAND_PROVENANCE_UNBOUND`

Do not collapse these into generic `PASS`/`FAIL`.

`PASS_SOURCE_IDENTITY_FROZEN` is allowed only if all required runtime versions, source files, hashes, line-ending facts, static symbol families, backend-path status, and O0c state-indexing/full-sequence capture classifications are complete and compatible with later implementation-authority authoring.

## 15. Relationship To O0c Implementation

A future O0c implementation authority may be authored only after this source preflight is executed successfully and establishes sufficient source semantics to define:

- exact instrumentation location;
- exact recurrent-state shape/update semantics;
- backend path;
- non-interference equivalence tests.

This preflight does not authorize implementation.

If the future preflight blocks, no O0c observer implementation may proceed until a separate recovery authority resolves the specific blocker.

## 16. Relationship To Scientific Execution

This preflight is infrastructure/provenance evidence only.

It is not scientific evidence. It does not authorize O0c model execution, O0c data tokenization, O0c recurrent-state capture, O0b reinterpretation, learned probes, threshold tuning, candidate selection, promotion, or any scientific conclusion.

## 17. Preferred Execution Environment

Because the target is exact compatibility with the previously validated O0b runtime, the eventual preflight may be proposed for Kaggle CPU with `Accelerator=None`, provided:

- runtime version guard passes;
- no model/tokenizer is loaded;
- no package mutation occurs;
- no optional kernel is enabled;
- execution receives separate authority after this candidate is frozen.

This authoring task does not execute Kaggle and does not authorize Kaggle.

## 18. Future Command Discipline

A later exact command must be repository commit-bound and must fail closed on dirty or wrong HEAD where applicable.

The later command must have:

- deterministic output path;
- no hidden environment mutation;
- no package installation/removal;
- no model/tokenizer/model-forward entry point;
- exact command hash captured by `cm run`;
- exact implementation script SHA256/byte count captured before execution;
- explicit runtime-version guard before source parsing;
- exact source-provenance JSON path captured in the command or implementation contract.

This candidate does not generate the final execution command because the required preflight script does not yet exist.

## 19. Future Script Requirement

A dedicated repository script will be required later.

Proposed path:

`scripts/preflight_longterm_o0c_runtime_source_provenance.py`

Exact responsibility:

- capture Python/NumPy/torch/Transformers versions without model/tokenizer loading;
- resolve installed `transformers` package distribution metadata and source paths;
- compute source byte, SHA256, LF/CR/final-LF facts;
- statically parse and bind required Mamba/cache symbol families;
- classify recurrent-state indexing compatibility and full-sequence capture feasibility;
- classify backend/kernel availability and CPU sequential path status;
- emit exactly one deterministic JSON artifact;
- fail closed with the taxonomy in this authority.

The script is not implemented by this candidate. Separate implementation authority is required before creating or editing it.

## 20. Required Later Implementation Tests

A later preflight implementation must have tests covering at minimum:

- wrong Transformers version;
- missing Mamba source;
- missing cache source;
- source hash and byte calculation;
- CR/LF/final-LF calculation;
- unresolved required symbol;
- ambiguous recurrent-state update site;
- forbidden model/tokenizer invocation;
- forbidden package mutation;
- deterministic JSON serialization;
- rerun byte identity under the same source tree.

Tests must remain CPU-only and must not load pretrained model/tokenizer assets or mutate packages.

## 21. Protected State And Non-Authorizations

This candidate authorizes exactly one new report file:

`reports/longterm_o0c_runtime_source_provenance_preflight_authority_spec_candidate.md`

It does not authorize modifying:

- frozen O0c authority;
- O0b artifacts, reports, scripts, or tests;
- current docs;
- any URP/reason-router state;
- stage180 files;
- protected temp directories;
- root patch files;
- `cm` tooling.

Protected unrelated state includes:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`
- `reports/stage180a_pass2_annotations_completed.csv`

## 22. Candidate Validation Contract

Required validation for this authoring task:

```powershell
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
```

Also compute and report this candidate file's SHA256 and byte count.

Expected task-attributable delta:

```text
A reports/longterm_o0c_runtime_source_provenance_preflight_authority_spec_candidate.md
```

Nothing may be staged, committed, or pushed.

## 23. Explicit Non-Execution Attestation

NO MODEL TOKENIZER LOADING.

NO TOKENIZER INVOCATION.

NO PRETRAINED MODEL LOADING.

NO MODEL FORWARD.

NO GENERATION.

NO TRAINING.

NO EVALUATION.

NO KAGGLE.

NO PACKAGE INSTALL.

NO PACKAGE UNINSTALL.

NO PACKAGE UPGRADE OR DOWNGRADE.

NO OPTIONAL KERNEL ENABLEMENT.

NO ENVIRONMENT MUTATION.

NO IMPLEMENTATION.

NO COMMIT.

NO PUSH.

## 24. Next Authorized Action

The next authorized action is independent verification of this candidate's exact bytes.

Only after independent verification, unchanged commit/push, and controller freeze-recording may a separate implementation-authority task consider creating the dedicated preflight script.
