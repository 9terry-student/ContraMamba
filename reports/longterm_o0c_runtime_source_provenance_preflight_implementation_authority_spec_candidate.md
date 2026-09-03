# ContraMamba O0c Runtime-Source Provenance Preflight Implementation Authority Spec Candidate

## 1. Overall Verdict And Phase

Overall verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

This document is a single candidate implementation authority/specification for a later bounded implementation of the deterministic O0c runtime-source provenance preflight.

Phase:

`STATIC O0c RUNTIME-SOURCE PREFLIGHT IMPLEMENTATION AUTHORITY AUTHORING ONLY`

This candidate does not authorize implementation yet. It does not authorize preflight execution, canonical artifact production, tokenizer invocation, model loading, pretrained weight loading, model forward passes, generation, training, evaluation, Kaggle execution, package installation, package removal, package upgrade/downgrade, optional-kernel enabling, environment mutation, staging, committing, or pushing.

## 2. Authority Chain

Authority order used:

1. Current controller instruction for this task card.
2. Current candidate before path-canonicalization correction: `reports/longterm_o0c_runtime_source_provenance_preflight_implementation_authority_spec_candidate.md`.
3. Independent verifier result: `BLOCKED_SOURCE_SHADOWING_PATH_CANONICALIZATION_UNFROZEN`.
4. Frozen O0c runtime-source provenance preflight authority: `8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`.
5. Frozen O0c native-state instrumentation authority: `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`.
6. Frozen O0b scientific interpretation: `f1dc559d546d20611d66b27684bbfa0f02afa696`.
7. Repository `AGENTS.md`.

Canonical authoring HEAD required and verified before authoring:

`8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`

This candidate supersedes no frozen scientific interpretation. It only narrows the frozen O0c preflight authority into a future implementation contract.

## 3. Future Implementation Scope

After this candidate is independently verified and frozen, the bounded implementation task may create or modify only:

- `scripts/preflight_longterm_o0c_runtime_source_provenance.py`
- `tests/test_preflight_longterm_o0c_runtime_source_provenance.py`

No other production or test files are authorized by default.

If implementation appears to require any additional fixture/helper file, even a tiny one, the implementation task must stop and report `BLOCKED_IMPLEMENTATION_SCOPE_WOULD_WIDEN`. It must not silently widen scope.

## 4. Implementation Purpose

The future script exists only to produce deterministic infrastructure/provenance evidence establishing whether the exact runtime source is sufficiently frozen and understood for later O0c recurrent-state instrumentation authority.

It must not:

- instrument Mamba;
- patch Transformers;
- capture recurrent states;
- load pretrained weights;
- tokenize O0b data;
- run model forwards;
- generate;
- train or evaluate;
- produce scientific O0c signal;
- produce the canonical target runtime provenance artifact without separate execution authority.

## 5. Frozen CLI Contract

The future CLI must be exactly:

```text
python scripts/preflight_longterm_o0c_runtime_source_provenance.py \
  --output <deterministic-json-path> \
  --expected-python 3.12.13 \
  --expected-numpy 2.0.2 \
  --expected-torch 2.10.0+cpu \
  --expected-transformers 5.0.0
```

Required flags:

- `--output`
- `--expected-python`
- `--expected-numpy`
- `--expected-torch`
- `--expected-transformers`

No default output path is allowed. No model ID, tokenizer ID, dataset path, Kaggle path, package index, cache directory, random seed, branch name, or hidden network-control flag is allowed.

The script must fail closed with `BLOCKED_OUTPUT_COLLISION` if the output path already exists. No overwrite flag is authorized.

The script must perform no implicit network action and must not require network access. Stdout/stderr must contain a deterministic summary of the final status, output path, and blocker category. A blocked status must return a meaningful nonzero process exit. `PASS_SOURCE_IDENTITY_FROZEN` must return zero only after every mandatory field is resolved.

This authority does not freeze a final Kaggle command and does not authorize Kaggle execution.

## 6. Runtime Version Acquisition

The future implementation may obtain versions by safe package/module metadata only:

- Python: `platform.python_version()` or an equivalent deterministic standard-library value.
- NumPy: `importlib.metadata.version("numpy")`, with direct import only if needed and safe.
- torch: `importlib.metadata.version("torch")`, with direct import only if needed and safe.
- Transformers distribution: `importlib.metadata.version("transformers")`.

Version comparison must be exact string equality against the CLI expected values. No normalization is authorized.

PyTorch versions may be represented by a `str` subclass in some contexts. The implementation must not use `type(value) is str`. It must use semantics compatible with `isinstance(value, str)` or explicitly deterministic string conversion with `str(value)` before comparison. Tests must include a torch-version `str` subclass regression.

If any actual version cannot be obtained deterministically, the status must be `BLOCKED_RUNTIME_VERSION_UNAVAILABLE`. If any actual version differs from the expected CLI value, the status must be `BLOCKED_RUNTIME_VERSION_MISMATCH`.

## 7. Import And Distribution Resolution Contract

The future script must independently reconcile:

1. installed Transformers distribution metadata/root;
2. import-resolved `transformers` package root;
3. import-resolved Mamba source path;
4. import-resolved cache-utils source path.

The implementation must use `importlib.metadata.distribution("transformers")`, `importlib.util.find_spec("transformers")`, `importlib.util.find_spec("transformers.models.mamba.modeling_mamba")`, and `importlib.util.find_spec("transformers.cache_utils")`, or equivalently safe source/spec discovery that does not instantiate model/tokenizer objects.

Distribution root means the canonical resolved parent root implied by the installed distribution metadata and files. Import-resolved package root means the canonical resolved directory containing the imported `transformers` package's `__init__.py` or namespace package location. Source files for Mamba and cache-utils must resolve inside the reconciled Transformers distribution root and under the import-resolved `transformers` package root.

The script must fail closed with `BLOCKED_TRANSFORMERS_SOURCE_SHADOWING` if any of these conditions hold:

- a repo-local `transformers` package would be imported;
- a `PYTHONPATH`/sys.path location outside the reconciled distribution/package root supplies `transformers`;
- the import-resolved `transformers` package root is outside the recorded distribution root;
- Mamba or cache-utils source resolves outside the recorded Transformers distribution root;
- distribution metadata cannot be reconciled with the imported package location.

The script must fail closed with `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS` when multiple plausible `transformers` roots or distribution locations are discovered and cannot be uniquely reconciled.

The script must not rely only on printing paths. Shadowing detection is a required machine-enforced gate.

## 8. Canonical Path Reconciliation Contract

Every path participating in Transformers provenance reconciliation must be converted to a deterministic canonical filesystem path before comparison. This applies at minimum to:

- installed Transformers distribution root/location;
- import-resolved `transformers` package root;
- import-resolved `transformers.models.mamba.modeling_mamba` source;
- import-resolved `transformers.cache_utils` source;
- repository root;
- any candidate repo-local, `PYTHONPATH`, sys.path, editable-install, or other out-of-distribution source root considered by shadowing checks.

Canonicalization must be filesystem-aware, not raw textual normalization. The future implementation must conceptually:

1. convert each path to `pathlib.Path` or an equivalent path-object representation;
2. require or derive an absolute path;
3. normalize `.` and `..` path segments;
4. resolve symlinks, junctions, and other reparse/indirection targets where the host filesystem and Python API allow deterministic resolution;
5. compare ancestry by filesystem path components, not string prefix;
6. account for platform case semantics.

The preferred implementation contract is `Path(...).resolve(strict=True)` for authoritative provenance paths, followed by component-aware comparison. This authority freezes `strict=True` for source files, package roots, distribution roots, and repository root because authoritative imported source files and installed package/distribution roots must already exist. Missing paths, permission failures, broken links, resolution loops, or otherwise unresolvable canonical targets must block; the implementation must not synthesize a normalized path and continue.

Canonicalization failure for any authoritative source, package root, distribution root, or repository root must produce `BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED`. A module source that cannot be found at all may use `BLOCKED_SOURCE_FILE_UNRESOLVED`; if a path is found but cannot be canonically resolved against the filesystem, use `BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED`.

Path ancestry is true only when the canonical source path is equal to the canonical root or is a descendant by path components. Acceptable conceptual checks are:

```python
source_canonical == root_canonical
source_canonical.is_relative_to(root_canonical)
```

or a compatibility implementation with identical component-aware semantics when `Path.is_relative_to` is unavailable.

Forbidden ancestry checks include raw `startswith`, substring checks, common string prefix checks, naive separator-appending tricks, and case-sensitive textual equality on case-insensitive platforms. The root `C:\env\site-packages\transformers` must not classify the source `C:\env\site-packages\transformers_evil\models\mamba\modeling_mamba.py` as a descendant.

On Windows, path reconciliation must not classify the same filesystem location as different merely because path strings use different case. For example, `C:\Python\Lib\site-packages\transformers` and `c:\python\lib\SITE-PACKAGES\Transformers` must reconcile according to Windows filesystem/path semantics. Filesystem/path APIs are preferred. If secondary string normalization is unavoidable, it may use `os.path.normcase` or an equivalent platform-aware normalization, but component-aware ancestry remains mandatory. This authority does not prescribe POSIX case-folding; on POSIX, case remains significant.

Shadowing checks must operate on resolved canonical filesystem targets rather than only lexical paths:

- if a lexical source path is outside the distribution/package root but its resolved canonical target is inside, it may be provenance-compatible only if all other distribution/import checks agree;
- if a lexical source path appears inside the distribution/package root but its resolved canonical target is outside, the script must block as `BLOCKED_TRANSFORMERS_SOURCE_SHADOWING` or, when the mismatch is specifically between reconciled roots, `BLOCKED_TRANSFORMERS_SOURCE_ROOT_AMBIGUOUS`;
- if source/root resolution fails or loops, the script must block with `BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED`.

PASS requires all authoritative imported sources to belong to one uniquely reconciled Transformers installation according to component-safe canonical ancestry:

1. import-resolved `transformers` package root is compatible with the installed Transformers distribution root;
2. Mamba source lies under the reconciled imported Transformers package root;
3. cache-utils source lies under the reconciled imported Transformers package root;
4. neither source resolves into repo-local or unrelated `PYTHONPATH`/out-of-distribution shadow trees;
5. ambiguous roots block.

The future implementation must know the canonical repository root and must block with `BLOCKED_TRANSFORMERS_SOURCE_SHADOWING` if the imported `transformers` package root, Mamba source, or cache-utils source resolves inside the repository tree. Current O0c does not authorize vendored Transformers. This repo-local check must use the same component-safe canonical ancestry rule and must not use raw prefix matching.

Any import-resolved Transformers module/source outside the reconciled installed distribution/package root is a blocker, regardless of whether it arrived through `PYTHONPATH`, sys.path ordering, an editable install, a repo-local module, or another mechanism. The script does not need to infer the human cause perfectly; it must prove provenance consistency or block.

## 9. Source Resolution Without Model Loading

The future script must resolve source corresponding to:

- `transformers.models.mamba.modeling_mamba`
- `transformers.cache_utils`

Discovery must not instantiate models or tokenizers. Prefer module specs and source paths over importing modules. If a module must be imported only to access source metadata, the implementation must document why the import is side-effect safe in code comments and tests must still prove no model/tokenizer/download/forward path is invoked.

The no-model/no-tokenizer boundary is testable: production code must have no reference path that calls `AutoTokenizer.from_pretrained`, `MambaModel.from_pretrained`, any `AutoModel*.from_pretrained`, model `forward`, or Hugging Face Hub download helpers.

## 10. Raw Source-Byte Identity

For each source file, the future script must read raw bytes before any text decoding and compute exactly:

- canonical resolved absolute path used in source/distribution reconciliation;
- byte count;
- SHA256 over raw bytes;
- LF byte count, counting byte `0x0A`;
- CR byte count, counting byte `0x0D`;
- final-LF boolean, true iff the final byte is `0x0A`.

No newline normalization may occur before hashing or counting. Text decoding for AST analysis must occur separately after raw byte facts are captured. UTF-8 decoding is required for AST parsing. If byte reading or hashing fails, use `BLOCKED_SOURCE_HASH_UNAVAILABLE`. If decoding or parsing fails, use `BLOCKED_SOURCE_DECODE_OR_PARSE_FAILURE`.

## 11. Static Source Parsing Contract

The future implementation must use Python AST for symbol/source-span binding wherever technically possible.

It must statically identify deterministic source locations for:

- mixer forward dispatch;
- sequential/slow path;
- recurrent state initialization;
- recurrent state update;
- convolution/cache initialization/update;
- cache recurrent-state storage if present;
- hidden-state output path;
- backend/kernel selection.

Raw substring search may be used only for diagnostics and candidate discovery. It must not produce `PASS_SOURCE_IDENTITY_FROZEN` if required symbol semantics remain unresolved or ambiguous.

For every required symbol family, the implementation must find exactly one compatible site unless the schema explicitly allows an optional absent field. Missing required sites must produce `BLOCKED_REQUIRED_SYMBOL_UNRESOLVED`. Multiple compatible sites without a frozen deterministic disambiguation rule must produce `BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`.

## 12. Symbol Location Format

Source-bound locations must be structured JSON objects:

```json
{
  "module": "transformers.models.mamba.modeling_mamba",
  "qualname": "MambaMixer.slow_forward",
  "source_file_key": "mamba",
  "source_sha256": "<64 hex chars>",
  "start_line": 1,
  "end_line": 1
}
```

Rules:

- `module` is the import module name, not an absolute path.
- `qualname` is the AST-derived qualified class/function name or a frozen symbolic label for a non-function assignment site.
- `source_file_key` is one of `mamba` or `cache`.
- `source_sha256` must equal the recorded raw source SHA256 for that file.
- `start_line` and `end_line` are one-based inclusive line numbers from AST nodes.
- absolute temporary working directories must not be embedded in the symbol identity object.
- the separately recorded source path is the only place absolute installed source paths appear.

Locations must be deterministic, source-hash-bound, unambiguous, and reconstructable by tests.

## 13. Recurrent-State Semantic Classification

Allowed recurrent-state classification values:

- `SOURCE_SUPPORTS_O0C_CONVENTION`
- `SOURCE_REQUIRES_IMPLEMENTATION_REVIEW`
- `SOURCE_INCOMPATIBLE_WITH_FROZEN_O0C_DESIGN`

Conservative rules:

- Use `SOURCE_SUPPORTS_O0C_CONVENTION` only when static AST-bound source proves that the recurrent SSM state is initialized for a fresh sequence, updated after consuming each token, and exposed to the hidden-state/output path in a way consistent with `s_t = post-consumption recurrent SSM state after token x_t`.
- Use `SOURCE_REQUIRES_IMPLEMENTATION_REVIEW` when source indicates plausible recurrent state support but update timing, tensor shape, full-sequence capture feasibility, cache lifecycle, or backend interaction cannot be fully proven statically.
- Use `SOURCE_INCOMPATIBLE_WITH_FROZEN_O0C_DESIGN` when source contradicts post-consumption indexing, lacks a recurrent SSM state surface, requires token-by-token replay where O0c forbids it, or would require model behavior changes to observe the state.

The script must not infer runtime non-interference. It must not state that capturing `ssm_state_t` is safe merely because a source assignment exists.

If recurrent-state semantics are unresolved, final status must be `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`. If indexing contradicts O0c, final status must be `BLOCKED_O0C_INDEXING_INCOMPATIBLE`.

## 14. Backend Static Classification

The script may inspect:

- optional kernel import/spec availability;
- relevant static flags/constants;
- static dispatch conditions;
- CPU branch semantics;
- sequential/slow path existence;
- associative-scan conditions if applicable.

Optional kernel availability checks must be observation-only. Checking `importlib.util.find_spec` is permitted. Importing optional compiled kernels is prohibited unless a later authority proves the import is side-effect safe and necessary. No installation, compilation, CUDA initialization, or environment manipulation is allowed.

Backend path classification values:

- `BACKEND_CPU_SEQUENTIAL_STATICALLY_PROVEN`
- `BACKEND_SEQUENTIAL_PRESENT_SELECTION_UNPROVEN`
- `BACKEND_ASSOCIATIVE_OR_KERNEL_PATH_MAY_INTERVENE`
- `BACKEND_STATICALLY_UNRESOLVED`
- `BACKEND_INCOMPATIBLE_WITH_O0C`

`PASS_SOURCE_IDENTITY_FROZEN` is allowed only when CPU sequential path selection is statically proven under the expected runtime and no optional/associative path can alter the intended future instrumentation surface. If static inspection cannot prove exact CPU path selection, final status must be `BLOCKED_BACKEND_PATH_UNRESOLVED`.

## 15. Deterministic Artifact Schema

The future implementation must emit exactly one primary JSON artifact with:

- UTF-8 encoding;
- sorted keys;
- `indent=2`;
- final LF;
- `allow_nan=False` or equivalent;
- deterministic stable list ordering;
- atomic write using a same-directory temporary path followed by atomic replace only when the destination did not previously exist.

The exact `schema_version` is:

`o0c_runtime_source_provenance_preflight_v1`

Required top-level key set:

```text
backend_static_classification
cache_source
expected_runtime
mamba_source
notes
o0c_full_sequence_capture_feasibility
o0c_state_indexing_compatibility
optimized_kernel_availability
preflight_status
runtime
schema_version
source_resolution
symbol_locations
```

The artifact must not contain timestamp, hostname, username, random UUID, mutable branch name, nondeterministic temp paths, training/evaluation metrics, model outputs, tokenized data, or scientific O0c signal.

Absolute installed package/source paths are allowed because runtime source resolution is required, but they must not be the sole identity. Every source path must be paired with raw byte identity and SHA256. The `transformers_distribution_root`, `transformers_import_root`, `mamba_source.path`, and `cache_source.path` fields must represent the canonical resolved paths used in reconciliation, not unresolved lexical paths. Canonicalization is a validation mechanism; do not add nondeterministic temporary fixture paths or additional canonicalization trace paths to production artifacts.

## 16. Required Schema Details

`expected_runtime` and `runtime` must each contain exactly:

```text
python
numpy
torch
transformers
```

`source_resolution` must contain exactly:

```text
transformers_distribution_root
transformers_import_root
transformers_distribution_version
shadowing_status
```

`mamba_source` and `cache_source` must each contain exactly:

```text
module
path
sha256
bytes
lf_count
cr_count
final_lf
```

`symbol_locations` must contain exactly:

```text
mixer_forward_dispatch
sequential_slow_path
recurrent_state_initialization
recurrent_state_update
convolution_cache_initialization_update
cache_recurrent_state_storage
hidden_state_output_path
backend_kernel_selection
```

Each `symbol_locations` value must be either a valid structured location object or `null` only where the implementation authority explicitly marks the site optional. A `null` required site blocks PASS.

`notes` must be a deterministic list of strings sorted lexicographically. Notes are diagnostic only and cannot convert a blocker into PASS.

## 17. Status Taxonomy

Allowed `preflight_status` values:

- `PASS_SOURCE_IDENTITY_FROZEN`
- `BLOCKED_RUNTIME_VERSION_UNAVAILABLE`
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
- `BLOCKED_ARTIFACT_SERIALIZATION_NONDETERMINISTIC`
- `BLOCKED_OUTPUT_COLLISION`
- `BLOCKED_FORBIDDEN_MODEL_TOKENIZER_INVOCATION`
- `BLOCKED_FORBIDDEN_PACKAGE_MUTATION`
- `BLOCKED_IMPLEMENTATION_SCOPE_WOULD_WIDEN`

Do not use generic `PASS`, `FAIL`, `ERROR`, or `UNKNOWN` as final status. Do not use `PASS_SOURCE_IDENTITY_FROZEN` when any mandatory field is unresolved.

## 18. Forbidden Invocation Guard

Production code must be architected so it never imports or calls model/tokenizer constructors or pretrained-loading APIs.

Tests must be capable of failing if production code attempts to call:

- `AutoTokenizer.from_pretrained`;
- `MambaModel.from_pretrained`;
- any `AutoModel*.from_pretrained`;
- model `forward`;
- Hugging Face Hub download helpers if introduced.

Tests may monkeypatch sentinel APIs/modules. Production code must not add runtime monkeypatching only to satisfy tests.

## 19. Forbidden Package Mutation Guard

Production code must contain no:

- subprocess pip invocation;
- package manager invocation;
- install/uninstall command;
- upgrade/downgrade command;
- extension compilation command;
- site-packages write.

Tests must detect obvious prohibited subprocess/package mutation paths, including static scans for `pip`, `conda`, `mamba install`, `uv pip`, `python -m pip`, compiler/build invocations, and write attempts into installed package roots.

## 20. Required Test Matrix

The future test file must cover at least:

Runtime/version:

- exact expected runtime PASS fixture;
- wrong Python;
- wrong NumPy;
- wrong torch;
- wrong Transformers;
- torch-version `str` subclass regression.

Source resolution:

- Mamba source missing;
- cache source missing;
- distribution root mismatch;
- repo-local Transformers shadowing;
- `PYTHONPATH`-style/out-of-distribution source shadowing;
- ambiguous source root;
- valid canonical package root plus true descendant source reconciles;
- prefix trap where root is `.../transformers` and source is `.../transformers_evil/...` blocks;
- lexical `.` and `..` segments resolving to a valid descendant reconcile;
- lexical source path appearing under root but symlink/junction/reparse target resolving outside blocks;
- lexical source path outside root but resolved target inside follows the documented resolved-target policy and reconciles only when all other distribution/import checks agree;
- canonical imported package/source under repository root blocks as repo-local shadowing;
- canonical source outside installed distribution/package root blocks as out-of-distribution shadowing;
- Windows same-location case variants reconcile on Windows rather than false blocking;
- POSIX case-distinct paths remain distinct on POSIX;
- missing, broken, permission-denied, looping, or otherwise unresolvable source/root canonicalization blocks with `BLOCKED_SOURCE_PATH_CANONICALIZATION_FAILED`.

Raw source identity:

- SHA256 exact;
- byte count exact;
- LF count;
- CR count;
- final LF true and false;
- no newline normalization.

Static parsing:

- required symbol resolved;
- required symbol missing;
- duplicate/ambiguous candidate sites;
- parse/decode failure;
- stable source-span representation.

Semantics/backend:

- supports-O0c classification fixture;
- requires-review fixture;
- incompatible-indexing fixture;
- backend unresolved fixture;
- sequential path present but selection not provable.

Safety:

- no model/tokenizer invocation;
- no download;
- no package mutation;
- no model forward.

Artifact:

- exact required schema;
- deterministic serialization;
- no NaN/Infinity;
- sorted/stable lists;
- final LF;
- same inputs produce byte-identical JSON;
- output collision behavior;
- no timestamp/hostname/username/UUID.

## 21. Test Environment Boundary

Most implementation tests must use deterministic synthetic source/package fixtures so local development environment version does not control results.

Tests must not require installed Transformers `5.0.0` and must not require package downgrade locally.

Canonical path and shadowing tests must use deterministic temporary synthetic directory trees where possible. Platform-specific cases may use pytest skip/conditional behavior for Windows-only case semantics or POSIX-only case distinctions. Tests must not fake Windows case-insensitive filesystem behavior and call that authoritative when the host cannot exercise it; pure helper-level platform-normalization tests may supplement, but not replace, filesystem-level tests. The test design must remain valid for the intended future Kaggle/Linux runtime while preventing unsafe development behavior on Windows.

A narrow integration-style source-discovery test may inspect the current local environment only if it:

- does not expect PASS against Transformers `5.0.0`;
- does not load a model or tokenizer;
- does not run a model forward;
- is clearly separated from deterministic unit tests.

## 22. Execution Boundary

Later implementation validation may execute the script only against synthetic fixtures or explicitly safe local-source fixtures if necessary for tests.

It must not execute the real scientific/preflight run intended to create canonical runtime provenance evidence.

Prohibited during implementation validation:

- Kaggle;
- `cm run`;
- canonical `PASS_SOURCE_IDENTITY_FROZEN` artifact from the target Kaggle runtime;
- model/tokenizer loading;
- model forward;
- training/evaluation;
- package/environment mutation.

Canonical runtime provenance execution requires separate execution authority after implementation is frozen.

## 23. Future Implementation Validation

After implementation, the bounded implementation task must run:

- targeted test file: `python -m pytest tests/test_preflight_longterm_o0c_runtime_source_provenance.py`;
- relevant existing tests only if dependency boundaries touch shared utilities;
- full repository test suite only if separately justified and registered as reasonably scoped;
- `git diff --check`;
- exact changed-file audit.

The implementation task must not report PASS for commands not actually run. It must distinguish pre-existing/environmental failures from patch failures.

## 24. Independent Verification Requirement

Because this code controls provenance and authority gating, implementation is high-risk enough to require independent verification before freeze.

The independent verifier must inspect:

- fail-closed behavior;
- source-shadowing detection;
- canonical path resolution, component-safe ancestry, platform case semantics, symlink/junction/reparse behavior, repo-local shadow detection, out-of-distribution shadow detection, and ambiguous-root blocking;
- exact byte/hash calculations;
- AST/symbol ambiguity behavior;
- deterministic artifact serialization;
- forbidden model/tokenizer/download/package paths;
- absence of accidental widening into instrumentation;
- exact changed-file set.

No implementation may be treated as frozen until this independent verification completes.

## 25. Commit And Future Authority Boundary

This implementation authority candidate does not authorize implementation yet.

After this candidate is frozen, a bounded implementation task may modify only the authorized script/test files and must use `Commit/Push: NO`.

After implementation verification, a separate freeze commit may be considered by a later task. Preflight execution still requires separate execution authority.

## 26. Protected State And Non-Authorizations

This candidate authorizes exactly one new report file:

`reports/longterm_o0c_runtime_source_provenance_preflight_implementation_authority_spec_candidate.md`

It does not authorize modifying:

- `scripts/`;
- `tests/`;
- frozen O0c authorities;
- O0b artifacts/reports/code;
- docs;
- URP/reason-router state;
- stage180 files;
- `cm` tooling;
- root patches;
- protected temp directories.

Protected unrelated state includes:

- `p3w7_a0_final_verify_focus_tmp/`
- `p3w7_a0_final_verify_full_rs_tmp/`
- `p3w7_a0_final_verify_full_tmp/`
- `reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`
- `reports/stage180a_pass2_annotations_completed.csv`

## 27. Candidate Validation Contract

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
A reports/longterm_o0c_runtime_source_provenance_preflight_implementation_authority_spec_candidate.md
```

Nothing may be staged, committed, or pushed.

## 28. Explicit Non-Execution Attestation

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

## 29. Next Authorized Action

The next authorized action is independent verification of this candidate's exact bytes and authority sufficiency.

Only after independent verification and controller freeze-recording may a separate implementation task create or modify the dedicated preflight script and its dedicated test file.
