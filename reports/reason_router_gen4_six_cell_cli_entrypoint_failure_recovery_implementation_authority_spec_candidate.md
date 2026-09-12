# ContraMamba Gen4 Six-Cell CLI Entrypoint Failure Recovery Implementation Authority - Candidate

## 1. Status

- Status: CANDIDATE
- Phase: GEN4_SIX_CELL_CLI_ENTRYPOINT_FAILURE_RECOVERY_IMPLEMENTATION_AUTHORITY
- Failed execution authority: 3c3e0dfba7455a09e36d4fbeabaa6785b79582f2
- Frozen pre-failure implementation commit: 1e5ee9794c2b849d3b3ca2fbfa36baf63e55a312
- Frozen implementation authority: 8e3aa94c69360c870fe6f6f808a611916401fa03
- Frozen implementation specification: 40d735d83018c7fc9b086226e95d1e46b141b0ab
- Frozen generator semantic authority: 91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea
- Training/Evaluation allowed: NO
- Canonical materialization allowed: NO
- Model execution allowed: NO
- Tokenizer execution allowed: NO
- Kaggle allowed: NO
- Commit/Push: NO

This authority exists solely to correct the direct CLI entrypoint import defect
revealed by the first authorized canonical materialization attempt.

## 2. Observed failure

The exact frozen execution command began with:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py ...

and failed before materialization with:

ModuleNotFoundError: No module named 'scripts'

The failure occurred while importing:

from scripts.build_controlled_v5 import ...

No canonical artifact was created.

Therefore:

FAILED_CANONICAL_EXECUTION_ATTEMPT = 1

FAILED_ATTEMPT_EXITED_BEFORE_MATERIALIZATION = YES

CANONICAL_ARTIFACT_CREATED_BY_FAILED_ATTEMPT = NO

## 3. Failure classification

This is classified as:

FAILURE_CLASS = CLI_ENTRYPOINT_IMPORT_PATH_DEFECT

It is not:

- a structured-fact failure;
- a generator semantic failure;
- an artifact cardinality failure;
- a provenance-content failure;
- a deterministic-serialization failure;
- a model failure;
- a tokenizer failure;
- a scientific outcome failure.

ROOT_CAUSE = DIRECT_FILE_EXECUTION_DOES_NOT_PLACE_REPOSITORY_ROOT_ON_SYS_PATH

The existing focused pytest suite did not catch this because pytest execution
provided repository-root importability that direct file execution does not.

## 4. Failed execution authorization state

The one-time canonical materialization authority at:

3c3e0dfba7455a09e36d4fbeabaa6785b79582f2

has been exercised and failed.

It must not be silently retried.

Therefore:

FAILED_EXECUTION_AUTHORITY_REUSE = FORBIDDEN

COMMAND_SUBSTITUTION_WITHOUT_NEW_AUTHORITY = FORBIDDEN

In particular, changing the command to:

python -m scripts.materialize_reason_router_gen4_six_cell_contrast ...

is not an authorized workaround under the failed authority.

A replacement execution authority will be required after the corrected
implementation is frozen.

## 5. Exact authorized correction delta

Exactly two files may be modified:

scripts/materialize_reason_router_gen4_six_cell_contrast.py

tests/test_materialize_reason_router_gen4_six_cell_contrast.py

Therefore:

AUTHORIZED_CORRECTION_FILE_COUNT = 2

No third file is authorized.

## 6. Required implementation correction

The materializer must support the already-frozen direct file invocation form:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py ...

when launched from the ContraMamba repository root.

The correction must establish repository-root importability before:

from scripts.build_controlled_v5 import ...

is evaluated.

A minimal acceptable design is:

- determine repository root from __file__;
- ensure that repository root is present on sys.path;
- only then import scripts.build_controlled_v5.

The correction must not alter any Gen4 structural semantics.

Therefore:

DIRECT_FILE_ENTRYPOINT_IMPORT_SUPPORT = REQUIRED

REPOSITORY_ROOT_BOOTSTRAP_BEFORE_SCRIPTS_IMPORT = REQUIRED

## 7. Required regression test

The focused test suite must add a regression test that launches the materializer
as a direct file from repository root.

The regression test must not generate canonical data.

A permitted non-materializing invocation is:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py --help

The test must assert successful process exit.

Therefore:

DIRECT_FILE_HELP_REGRESSION_TEST = REQUIRED

DIRECT_FILE_HELP_EXPECTED_EXIT_CODE = 0

CANONICAL_DATA_GENERATION_IN_REGRESSION_TEST = FORBIDDEN

## 8. Semantic invariance requirement

The correction must not change:

- schema_version;
- mechanism_id;
- canonical cell IDs;
- canonical cell ordering;
- axis masks;
- intended_changed_axes;
- generator_source_fields;
- row-id construction;
- output field order;
- rendering semantics;
- source validation;
- pair-block validation;
- serialization;
- provenance constants;
- canonical source semantics.

Therefore:

SCIENTIFIC_SEMANTICS_DELTA = NONE

STRUCTURAL_MATERIALIZATION_SEMANTICS_DELTA = NONE

The only intended behavioral delta is direct CLI importability.

## 9. Protected source

The following must remain unchanged:

scripts/build_controlled_v5.py

Frozen generator Git blob:

baee23a9f71333125f4a8735c2c92d20cab7eb4f

Therefore:

MODIFY_BUILD_CONTROLLED_V5 = FORBIDDEN

No model, loss, tokenizer, dataset, sidecar, or existing report may be modified.

## 10. Authorized validation

After correction, run only local CPU validation:

python -m py_compile scripts/materialize_reason_router_gen4_six_cell_contrast.py tests/test_materialize_reason_router_gen4_six_cell_contrast.py

python -m pytest -q tests/test_materialize_reason_router_gen4_six_cell_contrast.py

and the direct non-materializing smoke check:

python scripts/materialize_reason_router_gen4_six_cell_contrast.py --help

No --num-pairs canonical execution is authorized.

No artifact output path may be supplied.

## 11. Validation interpretation

Successful correction validation may establish only:

CLI_ENTRYPOINT_CORRECTION_CANDIDATE = PASS

IMPLEMENTATION_CORRECTNESS_CANDIDATE = PASS

It does not establish canonical execution success.

It does not restore or extend the failed execution authority.

## 12. Canonical execution boundary

During correction implementation and validation:

CANONICAL_MATERIALIZATION = NOT_AUTHORIZED

DETERMINISTIC_RERUN = NOT_AUTHORIZED

The previously failed authority must not be reused.

After the corrected implementation is manually reviewed, committed, and pushed,
a new replacement execution authority must bind to:

- the new implementation commit;
- the new implementation SHA256;
- the new test SHA256;
- the exact direct-file command.

## 13. No outcome-bearing execution

The correction phase does not authorize:

- model execution;
- tokenizer execution;
- training;
- evaluation;
- representation extraction;
- label derivation;
- statistical testing;
- Kaggle.

## 14. Working-tree boundary

Do not clean or reset unrelated files.

Do not stage unrelated files.

Do not use:

git add .

Implementation work must stop after the exact two-file correction and authorized
validation.

Commit/push requires later manual review.

## 15. Stop conditions

Stop immediately if:

- HEAD differs from the frozen recovery-authority start commit;
- a third file would need modification;
- scripts/build_controlled_v5.py would need modification;
- direct CLI support cannot be fixed without semantic changes;
- canonical materialization is required to validate the correction;
- any outcome-bearing execution appears necessary.

## 16. Required correction evidence

After correction and validation, report:

- exact starting HEAD;
- exact changed-file list;
- implementation SHA256;
- test SHA256;
- py_compile result;
- focused pytest result and pass count;
- direct-file --help exit result;
- confirmation generator file remained unchanged;
- confirmation no canonical artifact was generated;
- confirmation no model/tokenizer/training/evaluation/Kaggle execution occurred.

## 17. Current decision

FAILURE_CLASS = CLI_ENTRYPOINT_IMPORT_PATH_DEFECT

FAILED_CANONICAL_EXECUTION_ATTEMPT = 1

CANONICAL_ARTIFACT_CREATED_BY_FAILED_ATTEMPT = NO

FAILED_EXECUTION_AUTHORITY_REUSE = FORBIDDEN

AUTHORIZED_CORRECTION_FILE_COUNT = 2

DIRECT_FILE_ENTRYPOINT_IMPORT_SUPPORT = REQUIRED

REPOSITORY_ROOT_BOOTSTRAP_BEFORE_SCRIPTS_IMPORT = REQUIRED

DIRECT_FILE_HELP_REGRESSION_TEST = REQUIRED

SCIENTIFIC_SEMANTICS_DELTA = NONE

STRUCTURAL_MATERIALIZATION_SEMANTICS_DELTA = NONE

CANONICAL_MATERIALIZATION = NOT_AUTHORIZED

DETERMINISTIC_RERUN = NOT_AUTHORIZED

TRAINING_EVALUATION = NOT_AUTHORIZED

MODEL_EXECUTION = NOT_AUTHORIZED

TOKENIZER_EXECUTION = NOT_AUTHORIZED

KAGGLE_EXECUTION = NOT_AUTHORIZED

## 18. Next boundary

If this recovery authority is reviewed and frozen:

CLI_ENTRYPOINT_CORRECTION_IMPLEMENTATION = AUTHORIZED_FOR_EXACT_TWO_FILES

After corrected implementation validation and freeze, the next execution object
must be a replacement authority:

GEN4_SIX_CELL_CANONICAL_MATERIALIZATION_EXECUTION_AUTHORITY_V2

## 19. Stop condition for this authority phase

Stop after this candidate is created and reviewed.

Do not modify the implementation yet.

Do not run canonical materialization.

Do not run deterministic rerun.

Do not reuse the failed execution authority.

Do not train.

Do not evaluate.

Do not run model inference.

Do not execute tokenizers.

Do not use Kaggle.