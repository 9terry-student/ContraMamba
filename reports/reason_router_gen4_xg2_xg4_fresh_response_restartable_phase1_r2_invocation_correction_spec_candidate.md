# XG2/XG4 Restartable Phase-1 r2 Invocation Correction

Parent execution freeze:

`a24b08c6f2143a335ae83dbde42c6d1ac8b469b5`

Implementation:

`d237953cae76a801c99ff2be8adf7bc1d9300d6d`

## Observed r1 failure

Run:

`g4k-xg2-xg4-restartable-phase1-d237953-r1`

failed before runner import completed with:

`ModuleNotFoundError: No module named 'scripts'`

The failing invocation was:

`python scripts/reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1.py ...`

No XG2 or XG4 scientific model forward occurred.

The r1 run identity is failure provenance and must not be reused.

## Correction

No Python source file, scientific definition, input, checkpoint, tokenizer,
threshold, cohort, budget, artifact schema, or acceptance criterion changes.

The only command correction is to invoke the existing package module as:

`python -m scripts.reason_router_gen4_xg2_xg4_fresh_response_fast_cuda_restartable_phase1 ...`

This preserves the repository root on the Python import path so package imports
under `scripts` resolve correctly.

## Retry identity

New run name:

`g4k-xg2-xg4-restartable-phase1-d237953-r2`

The exact retry command must use the full commit produced by freezing this
correction as both bootstrap identity and `--expected-head`.

All scientific constraints from the parent execution freeze remain unchanged:

- XG2 only and XG4 only;
- 300 pairs per family;
- 1200 baseline forwards per family;
- 2400 total scientific forwards maximum;
- zero alignment/Phase-2 forwards;
- zero R_ALIGN/H1/H2;
- zero training/backward/task-head execution;
- XG2 expected LARGE/SMALL = 92/208;
- XG4 expected LARGE/SMALL = 57/243;
- output artifacts remain outside the repository.

The retry must fail closed if the r1 output root already contains any partial
scientific artifact. No partial output may be silently overwritten or reused.
