# Longterm O0c Recurrent-State Initialization Diagnostic Interpretation Candidate

## 1. Overall interpretation verdict

Verdict:

`PASS_READY_FOR_INDEPENDENT_VERIFICATION`

Phase:

`REPORT_ONLY_RECURRENT_STATE_INITIALIZATION_DIAGNOSTIC_INTERPRETATION`

This report assigns exactly one frozen-authority recurrent-state root-cause classification:

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`

This is an infrastructure/preflight-validator interpretation only. It is not an O0c scientific result and does not authorize implementation, tests, Kaggle, another diagnostic, preflight rerun, model/tokenizer/dataset loading, training, evaluation, staging, committing, or pushing.

## 2. Starting repo/HEAD/state

Starting-state validation passed before authoring:

- repository root: `C:\o0c-preflight-auth-c551747`
- expected HEAD: `ca112032841fe316e5e2e7335dc95d89aeedb450`
- observed HEAD: `ca112032841fe316e5e2e7335dc95d89aeedb450`
- tracked modifications: none
- staged changes: none
- pre-existing candidate collision at `reports/longterm_o0c_recurrent_state_initialization_diagnostic_interpretation_candidate.md`: none

No mismatch was observed, so report authoring was permitted.

## 3. Authority chain actually used

Authority order used:

1. Current controller instruction for this report-only interpretation task.
2. Frozen recurrent-state-initialization diagnostic execution authority: commit `ca112032841fe316e5e2e7335dc95d89aeedb450`, `reports/longterm_o0c_recurrent_state_initialization_diagnostic_execution_authority_spec_candidate.md`.
3. Parent frozen corrected-preflight execution authority: commit `52bae21fc5f5ee4e3ba3b5b15a0d5682f86daf0f`, `reports/longterm_o0c_runtime_source_provenance_corrected_preflight_execution_authority_spec_candidate.md`.
4. Frozen corrected preflight implementation / diagnostic execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, especially `scripts/preflight_longterm_o0c_runtime_source_provenance.py`.
5. Frozen O0c runtime-source preflight implementation authority: commit `811ae9c843564e8cddb5fc373761afb618cb7cfd`, `reports/longterm_o0c_runtime_source_provenance_preflight_implementation_authority_spec_candidate.md`.
6. Frozen O0c runtime-source provenance preflight authority: commit `8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328`, `reports/longterm_o0c_runtime_source_provenance_preflight_authority_spec_candidate.md`.
7. Frozen O0c native-state instrumentation authority: commit `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3`, `reports/longterm_o0c_selective_ssm_native_state_instrumentation_authority_spec_candidate.md`.
8. Supporting O0c distribution-root correction authority: commit `387f0d2dda27ae0448fc7c0e66533306d9743e21`, `reports/longterm_o0c_transformers_distribution_root_derivation_correction_implementation_authority_spec_candidate.md`, for prior corrected-preflight lineage only.
9. Imported diagnostic audit: `C:\Users\Home1\.contramamba\imports\longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1_c551747180ce_20260907_095816`.
10. Imported prior corrected-preflight audit: `C:\Users\Home1\.contramamba\imports\longterm-o0c-runtime-source-provenance-preflight-c551747-v3_c551747180ce_20260907_093530`.
11. Repository `AGENTS.md`.

The prompt-rendered import paths containing a separator before `_c551747180ce` did not exist literally. The actual on-disk import directories use the repository/runbook naming convention with the suffix joined in the directory name, as listed above. This discrepancy is non-blocking because the run names, commits, hashes, timestamps, and import metadata in those actual directories match the required evidence.

## 4. Diagnostic v1 provenance verification

Direct imported diagnostic files verified:

- `run.meta`
- `manifest.json`
- `import.json`
- `command.sh`
- `run.log`

Required facts verified from those files:

- run: `longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1`
- execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`
- command SHA256: `0795668fa8e7770bed5dabde1991d0447f5c42d30d6062f499ff0033ca016247`
- started UTC: `2026-09-07T00:56:38Z`
- finished UTC: `2026-09-07T00:56:39Z`
- exit code: `0`
- run log SHA256: `abe1e486379001e9d079fdabe72f13f9c5464f79c5378edf41ee7f307f357935`
- run meta SHA256: `8189f302995151564dfbad736102908285c6b25d670e961b969f045064b732b7`
- handoff ZIP SHA256: `32c157f2586ffc7cffd5cc08cae3e99394371070c941592f0017cb7c88ca3e9a`
- collector: `PASS`
- `FILES_COLLECTED`: `0`
- import: `PASS`
- imported execution commit: `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`

`command.sh` raw identity was recomputed as SHA256 `0795668fa8e7770bed5dabde1991d0447f5c42d30d6062f499ff0033ca016247`, bytes `15564`, LF `364`, CR `0`, final LF `false`.

`FILES_COLLECTED=0` is expected and non-invalidating for this diagnostic because the frozen diagnostic authority authorized stdout-only evidence. The log, meta, command, handoff ZIP, collector, and import provenance remain valid. Diagnostic v1 is consumed and must never be rerun, reused, or overwritten.

## 5. Exact runtime/source identity

The imported diagnostic `run.log` verifies:

- Python: `3.12.13`
- NumPy: `2.0.2`
- torch: `2.10.0+cpu`
- Transformers: `5.0.0`
- `sys.executable`: `/usr/bin/python3`
- Transformers distribution root: `/usr/local/lib/python3.12/dist-packages/transformers`
- `modeling_mamba.py`: `/usr/local/lib/python3.12/dist-packages/transformers/models/mamba/modeling_mamba.py`
- `modeling_mamba.py` SHA256: `4c972b30f3c2cca977824fcc6891f956cd4387b6383aa7336848fbc5f2db1d83`
- bytes: `39500`
- LF: `860`
- CR: `0`
- final LF: `true`

No mismatch was observed.

## 6. Frozen validator predicate

At `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, `scripts/preflight_longterm_o0c_runtime_source_provenance.py` implements recurrent-state proof in `_recurrent_proof_nodes`.

The frozen recurrent-state initialization proof searches only statements in the direct body of `MambaMixer.slow_forward` and requires exactly one direct-body statement that:

- is `Assign` or `AnnAssign`;
- assigns bare `ssm_state`;
- contains a call ending in `new_zeros`, `zeros`, or `empty`.

Zero matches raise:

`BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED`

with blocker:

`recurrent_state_initialization`

More than one match raises:

`BLOCKED_REQUIRED_SYMBOL_AMBIGUOUS`

with blocker:

`recurrent_state_initialization`

This is the frozen validator contract only; it is not by itself a source-semantic conclusion.

## 7. Exact source evidence

The diagnostic `run.log` verifies `MambaMixer.slow_forward` spans lines `342-421`.

Cache branch evidence:

- line `353`: `if cache_params is not None:`
- line `354`: `ssm_state = cache_params.ssm_states[self.layer_idx].clone()`
- line `355`: `ssm_state = ssm_state.to(hidden_states.device)`

Non-cache branch evidence:

- line `374`: `else:`
- lines `375-378`: `ssm_state = torch.zeros((batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype)`

Alternative training path evidence:

- line `400`: `if self.use_mambapy and self.training and cache_params is None:`
- line `401`: `hs = pscan(...)`

Sequential path evidence:

- line `406`: `else:`
- line `408`: `for i in range(seq_len):`
- line `409`: `ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]`
- line `410`: `scan_output = torch.matmul(ssm_state.to(dtype), ...)`

Cache persistence evidence:

- line `416`: `if cache_params is not None:`
- line `417`: `cache_params.ssm_states[self.layer_idx].copy_(ssm_state)`

The frozen-validator replay in `run.log` reports:

- `candidate_count = 0`
- `matching_statement_spans = []`

The general AST occurrence analysis in the same `run.log` also reports the nested lines `375-378` assignment as an `Assign` to bare `ssm_state` with call path `torch.zeros` and `satisfies_frozen_recurrent_initialization_predicate: true`. The distinction is precise: the nested assignment satisfies the frozen statement-level predicate, but it is inside the body of the cache/non-cache `If` whose direct-body span is lines `353-379`; the frozen validator iterates only over direct statements in `slow_forward.body` and therefore sees the enclosing `If`, not the nested `Assign`.

## 8. Applicable frozen O0c recurrent-state semantics

The frozen O0c native-state instrumentation authority at `242ad9ed70fc995ebda560911a7d0dfd2f18f9b3` defines the relevant state as the selective-SSM recurrent state variable updated by the Mamba mixer recurrence, not exposed hidden states, convolution cache state, selective-scan temporaries, or SSM-A/B/C/Delta diagnostics.

The same authority defines the O0c indexing convention:

```text
s_{-1} = zero initial recurrent SSM state for that sequence/member/layer
s_t = recurrent SSM state after consuming token x_t
```

It further requires that the convention apply independently for each layer and each full-sequence forward, with no recurrent state reused across matched members.

The authority also states that the preferred O0c surface is the sequential full-sequence recurrence, because it directly exposes the recurrence loop; `cache_params` can store final recurrent and convolutional state, not the full per-token recurrent trajectory by default. Future O0c implementation must prefer the sequential backend, preserve 12 full-sequence forwards, and avoid token-by-token replay unless separately authorized.

The frozen O0c runtime-source preflight authority at `8c6a0ccf2a8583b9b7accbdb5ab757d722b6e328` requires the preflight to determine whether exact runtime source supports `s_t = post-consumption recurrent SSM state after token x_t`, whether sequential full-sequence path exposes or can be instrumented to copy every per-token post-update recurrent state, and whether exact static symbol/backend semantics are sufficiently bound before later implementation authority.

The frozen implementation authority at `811ae9c843564e8cddb5fc373761afb618cb7cfd` conservatively allows `SOURCE_SUPPORTS_O0C_CONVENTION` only when static AST-bound source proves the recurrent SSM state is initialized for a fresh sequence, updated after consuming each token, and exposed to hidden-state/output path consistently with post-consumption indexing.

## 9. Fresh/non-cache initialization analysis

For a fresh/non-cache sequence, installed Transformers `5.0.0` source sets `ssm_state` at lines `375-378` to `torch.zeros` with shape:

`(batch_size, self.intermediate_size, self.ssm_state_size)`

and with `device=hidden_states.device, dtype=dtype`.

This exactly matches the frozen O0c design requirement that a fresh full-sequence forward begins from deterministic zero recurrent SSM state `s_{-1}` for that sequence/member/layer. The source evidence is not merely the presence of `torch.zeros`; it is that the zero state is assigned to `ssm_state` in the non-cache branch before the recurrence-selection block and before the sequential update loop.

## 10. Continuation/cache-state analysis

When a prior recurrent state is supplied through `cache_params`, lines `353-355` load `cache_params.ssm_states[self.layer_idx]`, clone it, and move it to `hidden_states.device`. Lines `416-417` copy the final `ssm_state` back into `cache_params.ssm_states[self.layer_idx]` after the sequential recurrence path.

This is compatible with continuation/cache semantics: supplied prior recurrent state is used as the recurrence's starting state, and the post-recurrence state is persisted. The applicable O0c authority does not require this diagnostic interpretation report to prove the cache object's original allocation/zeroing in order to resolve the specific `slow_forward` recurrent-initialization blocker, because the O0c primary measurement design uses independent full-sequence forwards with no recurrent-state reuse across matched members. For the specific fresh-sequence question, `cache_params is None` leads to the deterministic zero `ssm_state` at lines `375-378`. For supplied-continuation semantics, the exact source shows `slow_forward` consumes the supplied recurrent state rather than inventing an incompatible state.

This report does not claim that future O0c implementation may ignore cache provenance generally. The existing O0c authorities still require future implementation/execution authorities to freeze source identity and non-interference contracts for any cache behavior they rely on.

## 11. mambapy/training-path relevance

The mambapy path is guarded by line `400`:

`self.use_mambapy and self.training and cache_params is None`

The frozen O0c native-state authority requires CPU, float32, `eval()`, frozen parameters, `torch.inference_mode()`, no dropout/training path, no optional optimized kernels, and the sequential backend as the preferred direct recurrence surface. Therefore the line `400` mambapy training-only branch is outside the authorized O0c execution regime.

For authorized O0c eval/full-sequence recurrence, the relevant branch is the line `406` `else`, containing the line `408` loop and line `409` post-token recurrent update.

## 12. Initialization-to-sequential-update dataflow

For the non-cache/fresh sequence branch, dataflow is established by exact source order:

1. Lines `353-379` select either supplied cache state or fresh zero state.
2. Lines `386-397` compute SSM parameters, `discrete_A`, `discrete_B`, and `deltaB_u`.
3. Under the authorized eval/non-mambapy regime, line `406` selects the sequential path.
4. Line `408` iterates over `seq_len`.
5. Line `409` updates `ssm_state` from the existing `ssm_state` and current-token terms.
6. Line `410` reads the just-updated `ssm_state` into `scan_output`.

Thus, for every authorized fresh/non-cache execution branch that matters to frozen O0c design, the deterministic zero `ssm_state` reaches the first sequential update. For supplied continuation/cache execution, the cloned supplied cache state reaches the same update path. The only branch that bypasses the line `408-410` loop is the training-only mambapy branch, which is outside frozen O0c execution semantics.

## 13. Interpretation of NOT_ESTABLISHED_BY_FROZEN_PATTERN

The diagnostic emitted:

`initial_state_reaches_first_update_static_observation = "NOT_ESTABLISHED_BY_FROZEN_PATTERN"`

This is not source-semantic uncertainty after reading the full diagnostic evidence and applicable O0c authority. It means the frozen direct-body syntactic proof did not establish a direct-body initialization before the first loop. The same `run.log` independently records that a general AST/source occurrence analysis found the nested lines `375-378` `ssm_state = torch.zeros(...)` assignment satisfying the predicate's statement-level conditions, and the exact source excerpt shows that this state reaches the line `409` recurrence under the authorized O0c execution regime.

The unresolved field therefore describes failure of the frozen validator pattern, not a remaining dataflow ambiguity in the exact installed source for the relevant O0c branches.

## 14. Exactly one final frozen-authority classification

Final classification:

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`

This classification is justified because all frozen criteria are established:

- applicable frozen O0c semantics are known from formal O0c authorities;
- exact installed source provides deterministic fresh-state initialization through nested lines `375-378`;
- continuation/cache behavior in `slow_forward` is compatible with supplied recurrent-state continuation;
- the initialized or supplied state reaches the relevant recurrence update at line `409`;
- the frozen preflight blocker arises solely because the qualifying initialization is nested under the cache/non-cache `If` and outside the validator's direct-body-only scan;
- no other recurrent-initialization semantic incompatibility remains for the authorized O0c execution regime.

## 15. Narrow root cause

The narrow root cause is:

The frozen validator's direct-body-only AST scan fails to descend into the cache/non-cache `If`, so it misses a semantically relevant nested `ssm_state = torch.zeros(...)` initialization at lines `375-378`.

This report does not specify or implement a code patch.

## 16. Evidence-layer separation

Code correctness:

Existing corrected-preflight code test status remains prior code-correctness evidence only. It is not reinterpreted as runtime semantic correctness.

Prior v3 preflight execution:

Validated blocked result: `BLOCKED_RECURRENT_STATE_SEMANTICS_UNRESOLVED` with blocker `recurrent_state_initialization`.

Prior v3 provenance:

Valid. Imported prior audit verifies run `longterm-o0c-runtime-source-provenance-preflight-c551747-v3`, execution commit `c551747180ce1e8fe4eed5f7aa5ab6294cd89948`, command SHA256 `369bd49fda005c35d22894e0dca0a105472f72daa6eab55a19821c7b697c9db0`, started UTC `2026-09-07T00:33:07Z`, finished UTC `2026-09-07T00:33:30Z`, exit code `2`, run log SHA256 `f475b6b1f025756d2abc8eceb813b65704f0b2a1d173c57bc6a8c085d5f371eb`, run meta SHA256 `112edde6121303cab2da65b62e679e03c990c9d4e7475378893b3eee66076b98`, collector/import PASS, and `FILES_COLLECTED=0`.

Diagnostic v1 execution:

PASS, exit code `0`.

Diagnostic v1 provenance:

Valid after collector PASS and import PASS.

Recurrent-state root-cause classification:

`VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`.

Scientific O0c conclusion:

`NONE`.

## 17. Scientific conclusion

Scientific conclusion:

`NONE`

No training, evaluation, model execution, recurrent-state capture, matched-control measurement, layer/anchor result, or model scientific claim is authorized or made by this report.

## 18. Exact next authorized phase recommendation

Because the selected classification is `VALIDATOR_RECURRENT_STATE_INITIALIZATION_PATTERN_FALSE_NEGATIVE`, the only recommended next phase is:

Author a narrow recurrent-initialization validator-correction implementation authority/spec.

This report does not authorize editing the validator, editing tests, rerunning v1, rerunning v3, executing Kaggle, registering a run, or producing a corrected preflight result.

## 19. Git validation results

Validation is to be run after this candidate is written:

```text
git diff --check
git diff --name-status
git diff --cached --name-status
git status --short
git rev-parse HEAD
```

The final observed results and candidate raw identity are reported outside the self-hashing body of this candidate to avoid recursive hash churn.

## 20. Candidate path and final raw identity

Candidate path:

`reports/longterm_o0c_recurrent_state_initialization_diagnostic_interpretation_candidate.md`

Final raw identity is to be reported after final validation:

- SHA256: recompute after final edits
- bytes: recompute after final edits
- LF: recompute after final edits
- CR: recompute after final edits
- final LF: recompute after final edits

## 21. Explicit no-code-change/no-execution boundary

Confirmed for this report task:

- NO implementation modification
- NO test modification
- NO previous report modification
- NO `AGENTS.md` modification
- NO `cm.ps1` modification
- NO run registry modification
- NO imported audit file modification
- NO package modification
- NO data modification
- NO Kaggle
- NO diagnostic v1 rerun
- NO preflight v3 rerun
- NO run registration
- NO model loading
- NO tokenizer loading
- NO dataset loading
- NO forward pass
- NO generation
- NO training
- NO evaluation
- NO staging
- NO commit
- NO push

## 22. Discrepancies/blockers

No blocking discrepancy was found.

Non-blocking discrepancy recorded: the prompt-rendered imported audit paths included a directory separator before `_c551747180ce`, while the actual import directories on disk join the suffix into the run directory name:

- `C:\Users\Home1\.contramamba\imports\longterm-o0c-recurrent-state-initialization-diagnostic-c551747-v1_c551747180ce_20260907_095816`
- `C:\Users\Home1\.contramamba\imports\longterm-o0c-runtime-source-provenance-preflight-c551747-v3_c551747180ce_20260907_093530`

The actual imported files verify all required run/provenance facts, so this path-rendering discrepancy does not block the interpretation.
