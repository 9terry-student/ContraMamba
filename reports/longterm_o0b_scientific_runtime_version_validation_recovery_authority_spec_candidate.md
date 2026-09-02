# O0b Runtime-Version Validation Recovery Authority Candidate

## 1. Verdict and Boundary

Verdict: PASS_READY_FOR_INDEPENDENT_VERIFICATION.

This document is a narrow static recovery-authority candidate for the consumed O0b v1 scientific failure. It does not implement a repair, does not authorize execution, and does not authorize reuse of any output or in-memory result from the failed process.

This candidate may authorize only a later implementation repair for the runtime-version validation type predicate that rejected the concrete torch runtime version after the scientific run had already started.

## 2. Authority Chain

Authority precedence for this candidate:

1. Current controller instruction.
2. Frozen O0b scientific execution authority: 49d3361aa96cd1aea958bd0e85f462811b92540c.
3. Consumed failed scientific attempt: longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1.
4. Failed execution implementation: 9a249c071b76fbf693f63b36ba8ec1036c69b2ba.
5. Frozen observer Git-object SHA256: 7d401c429db0962d6691610d6249549db93c10046d0eb805c6505de9a4b56375.
6. Validated failure provenance supplied by the controller.
7. Upstream O0b scientific-design, boundary, input, observer, and runtime-provenance authority chain already bound by the frozen execution authority.
8. Repository AGENTS.md.

Phase: STATIC RECOVERY-AUTHORITY AUTHORING ONLY.

Training/evaluation authorization: none.

Tokenizer/model authorization: none.

Kaggle authorization: none.

cm run/save/collect/import authorization: none.

Commit/push authorization: none.

## 3. Frozen Failure Provenance

The following failure provenance is frozen exactly for this recovery candidate:

| Field | Value |
|---|---|
| Run | longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1 |
| Execution HEAD | 9a249c071b76fbf693f63b36ba8ec1036c69b2ba |
| Layer-1 command SHA256 | 49a4e753d074bf353ffbff1677605d0078bd9a802182342e72cd512ae66c026b |
| Started UTC | 2026-09-01T18:03:09Z |
| Finished UTC | 2026-09-01T18:04:23Z |
| Exit code | 1 |
| Runtime precheck | PASS |
| python | 3.12.13 |
| numpy | 2.0.2 |
| torch | 2.10.0+cpu |
| transformers | 5.0.0 |
| Observed exception | ContractError: runtime version torch_version |
| Failure stack location | build_artifact_bundle -> build_manifest -> validate_runtime_versions -> require(...) |
| Collector | PASS |
| FILES_COLLECTED | 0 |
| Run log SHA256 | e803e0328359f843f99fa98ca28ae56b2d762f268e7fb9b1b746f4e8b0ad7933 |
| Run meta SHA256 | c99b9d671206f6b6038ad5a9dfc6193d9713ec47dad56bbd2bd864e44a5ee38c |
| Failure handoff ZIP SHA256 | c8c4741f73fb7ced84ba1118ac8ceb9c06a6feb4786ed8b01c402c981ddb72c6 |
| cm import | IMPORT PASS |
| Import VALIDATED | 0 |
| Import COPIED | 0 |
| Import IDENTICAL | 0 |
| Import audit | C:\Users\Home1.contramamba\imports\longterm-o0b-token-aligned-native-mamba-state-dynamics-9a249c0-v1_9a249c071b76_20260902_030613 |

VALIDATED=0/COPIED=0 is consistent with FILES_COLLECTED=0.

No complete scientific artifact bundle exists.

This run provides failure/provenance evidence only.

It provides no scientific conclusion.

Any in-memory forward results from the failed process are unusable and must not be recovered or reused.

## 4. Root-Cause Contract

Static source inspection of the committed observer at 9a249c071b76fbf693f63b36ba8ec1036c69b2ba verifies:

- `capture_runtime_versions()` obtains `torch_version` from `torch_module.__version__`.
- `validate_runtime_versions()` iterates over `RUNTIME_VERSION_KEYS`.
- For each runtime-version field, it currently requires `type(value) is str` before also requiring non-empty content, no leading/trailing whitespace, and rejection of forbidden placeholders.

The concrete Kaggle runtime can supply a string-compatible torch version object whose displayed value is exactly `2.10.0+cpu` while failing the exact-builtin-type predicate `type(value) is str`.

Classification: RUNTIME VERSION VALIDATION-CONTRACT FAILURE.

This failure is not classified as:

- scientific signal failure;
- model failure;
- tokenizer failure;
- runtime-version mismatch;
- provenance mismatch;
- artifact corruption.

The scientific computation must not be claimed to have succeeded merely because forward/model execution occurred before manifest validation failed.

## 5. Only Authorized Future Implementation Delta

A later implementation phase may modify only:

- `scripts/observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`
- `tests/test_observe_longterm_o0b_token_aligned_native_mamba_state_dynamics.py`

The only authorized observer semantic change is narrowly equivalent to replacing the exact builtin string requirement:

```python
type(value) is str
```

with non-coercive string-interface acceptance, such as:

```python
isinstance(value, str)
```

All remaining validation requirements must be preserved:

- value must be non-empty;
- no leading/trailing whitespace;
- forbidden placeholders remain rejected;
- non-string numeric/list/dict/object values remain rejected;
- no `str(value)` coercion;
- no runtime-version override;
- no fallback/default version;
- no normalization of the concrete version value.

The future implementation must add a regression test proving that a concrete subclass of `str` is accepted as a runtime-version value while all existing rejection tests remain valid. The test should prefer a synthetic `str` subclass and must not depend on the installed torch version.

No other observer semantic change is authorized.

## 6. Scientific Invariants Preserved

The recovery path must preserve unchanged:

- scientific design;
- dataset/input bytes;
- tokenizer/model identity and immutable revision;
- `add_special_tokens=False`;
- `use_fast=True` / fast tokenizer mode;
- `trust_remote_code=False` for tokenizer and model;
- CPU;
- float32;
- `model.eval()`;
- frozen parameters;
- `torch.inference_mode()`;
- `use_cache=False`;
- `output_hidden_states=True`;
- `return_dict=True`;
- exact 12 full-sequence forwards;
- one sequence per forward;
- no padding;
- no generation;
- no optimizer/backward;
- no cache_params instrumentation;
- no A/B/C/Delta instrumentation;
- exact pair/condition/divergence/anchor semantics;
- native pretrained Mamba hidden-state proxy definition;
- distance/metric definitions;
- pre-divergence invariant;
- summary/report semantics;
- deterministic artifact serialization;
- exact seven-artifact bundle contract;
- no best-layer selection;
- no best-anchor selection;
- no hard scientific threshold;
- no significance/generalization claim.

All listed invariants are frozen and unchanged by the permitted future runtime-version validator repair.

No outputs or numerical results from consumed v1 may be reused.

Any future recovery run must execute the complete observer from scratch.

## 7. Provenance and Future Run Rules

1. The consumed v1 run name MUST NOT be reused.
2. Its old Layer-1 command MUST NOT be reused because a repaired observer will have a new implementation commit and Git-object SHA.
3. A new observer implementation must first be independently verified and frozen at a new commit.
4. Only after that freeze may a separate scientific recovery execution authority define:
   - a new run name;
   - a new output directory;
   - a new Layer-1 exact command;
   - a new Layer-2 canonical argv;
   - the repaired observer commit/SHA.
5. No Kaggle execution is authorized by this recovery-authority candidate.

This candidate does not assign or freeze a future scientific run name. Run identity belongs to the later execution-authority stage after the repaired implementation identity exists.

## 8. Required Later Implementation Validation

The future implementation phase must require:

- exact intended two-file delta only;
- `git diff --check`;
- targeted observer contract tests;
- existing runtime-version rejection tests;
- new `str`-subclass acceptance regression test;
- full observer test file if feasible;
- no model/tokenizer invocation for tests;
- no training/evaluation;
- independent verifier because this touches provenance/runtime validation;
- explicit new script Git-object SHA256 after freeze.

No scientific execution is authorized during implementation validation.

## 9. Candidate Activation Boundary

This candidate does not become implementation authority merely by existing.

Activation requires:

1. exact candidate byte identity reported;
2. independent verifier PASS;
3. candidate committed/pushed unchanged;
4. controller records the committed recovery-authority identity;
5. controller explicitly transitions to implementation.

Until all activation conditions are satisfied, this document is only a candidate recovery authority.

## 10. Explicit Non-Authorization Ledger

NO IMPLEMENTATION.

NO TOKENIZER.

NO MODEL.

NO FORWARD.

NO TRAINING.

NO EVALUATION.

NO KAGGLE.

NO CM RUN SAVE.

NO CM RUN.

NO CM COLLECT.

NO CM IMPORT.

NO COMMIT.

NO PUSH.
