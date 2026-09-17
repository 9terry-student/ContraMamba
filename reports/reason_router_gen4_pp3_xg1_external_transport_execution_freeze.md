# Gen4-K PP3 XG1 External Transport Execution Freeze

## Status

EXECUTION_READY

This is the sole minimal execution freeze for the prospectively frozen PP3 external-transport experiment on the independent XG1 generator family.

It does not alter the frozen scientific question, population, PP3 construction, epsilon, endpoint, confirmatory rule, forward budget, or failure policy. Those remain defined by the scope freeze.

## Frozen identity

- Branch: `gen4-k-xg2-basis-holdout`
- Scope freeze: `02e4c897d65f8f6e90b855054793594866545bf2`
- Scope path: `reports/reason_router_gen4_pp3_xg1_external_transport_scope.md`
- Scope Git blob: `7f5d297f3f02225d7935c372db9c36caa6ad60ab`
- Preparation freeze: `30a1dcbb1be7dc9b3a834b0b539b5d29016e55ed`
- Preparation root: `reports/reason_router_gen4_pp3_xg1_external_transport_preparation_02e4c89`
- Runner implementation commit: `96bcfc214358361ad836044faff1b1fa92f6676d`
- Scientific runner: `scripts/reason_router_gen4_pp3_xg1_external_transport_fast_cuda.py`
- Scientific runner Git blob: `1b81deacc330beb9a7cf1d09b520ec55aaf2cc0d`
- Unit/static test: `tests/test_reason_router_gen4_pp3_xg1_external_transport_fast_cuda.py`
- Test Git blob: `ff405d348a69400103a3724b884b3f0a24ff672a`
- Local validation before runner freeze: `18 passed`
- PP3+ SHA256: `66ad0cd0f931b0aff88bfc8afc4e9ffa9e355d32f054d376acebb97b469a3cff`
- PP3- SHA256: `ea2c997de7dbaea4edad8b1f30cdac31b4a0c76db0f0df2983900f28a2529ce7`
- XG1 source-pair count: `300`
- XG1 pair order: `xg1_fact_001` through `xg1_fact_300`

The scientific runner, frozen preparation artifacts, and reused runtime dependencies must retain the exact identities enforced by the runner.

The authorized execution HEAD is the clean, pushed repository HEAD that contains this freeze while preserving the frozen identities above. That exact full commit SHA must be passed to the runner as `--expected-head` and recorded in run provenance.

No dirty-tree, alternate-runner, hash-mismatched, or preparation-drift execution is authorized.

## Authorized scientific observation

Run exactly the frozen PP3 observation defined by the scope:

- `epsilon = 0.025`
- direction order: `PP3+`, then `PP3-`
- exactly 4 scientific model forwards per direction per pair
- exactly 8 scientific model forwards per pair
- exactly 2400 scientific model forwards over 300 pairs
- exactly 0 new baseline model forwards
- retain every finite negative or positive observed endpoint without sign reversal or rescue

For each pair, the runner may observe and persist:

- `J_i(PP3+)`
- `J_i(PP3-)`
- `C_PP3_i = (s_3 / 5) * [J_i(PP3+)^2 - J_i(PP3-)^2]`

with frozen `s_3 = 0.98692852916688512`.

The runner must perform raw scientific observation only. During this GPU run it must not execute the confirmatory t-test or assign a scientific transport label.

The artifact summary must therefore preserve:

- `scientific_model_forward_count_this_run = 2400`
- `baseline_model_forward_count_this_run = 0`
- `primary_inference_executed = false`
- `multiplicity_correction_executed = false`
- `training_executed = false`
- `backward_executed = false`
- `task_heads_executed = false`
- `logits_read = false`
- `scientific_conclusion = null`

## Post-import inference boundary

Only after the execution artifact is collected, imported, provenance-validated, complete for all 300 pairs, and its forward budget is validated may the already-frozen primary inference be applied:

- one-sample Student t-test on the 300 `C_PP3_i` values
- one-sided alternative: `mean(C_PP3) > 0`
- alpha: `0.05`
- exactly one primary hypothesis
- no multiplicity correction

The positive label remains:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_SUPPORTED_ON_XG1_EXTERNAL_GENERATOR`

only if all provenance/completeness checks pass, `mean(C_PP3) > 0`, and the frozen one-sided test rejects at alpha `0.05`.

Otherwise the frozen label is:

`PP3_PROJECTOR_CONTRAST_TRANSPORT_NOT_ESTABLISHED_ON_XG1_EXTERNAL_GENERATOR`

No rescue hypothesis or alternate direction is authorized.

## Failure policy

Any commit, blob, preparation hash, XG1 population, anchor, tokenizer, runtime, checkpoint, finite-value, forward-budget, output-collision, or artifact-validation failure blocks the run.

Do not replace pairs, change epsilon, rotate/re-rank PP3, test another principal pair, change checkpoint/layer/intervention semantics, or use a rerun from a different commit to rescue a scientific result.

## Authorization

Scientific GPU observation is authorized only from the clean, pushed HEAD containing this freeze.

The exact pushed execution HEAD must be supplied to the runner as `--expected-head`.

This freeze does not authorize execution from an uncommitted or dirty worktree and does not authorize post-hoc scientific analysis beyond the already frozen post-import rule above.
