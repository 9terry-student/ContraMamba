# ContraMamba Gen5 Phase 2 Fresh Ownership Assay
## Validated Scientific Interpretation Report Candidate

### Status

`VALIDATED_TERMINAL_NEGATIVE`

Scientific conclusion:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

This report records the validated interpretation of the single frozen
confirmatory Gen5 Phase 2 fresh ownership assay. It does not authorize
additional training, evaluation, rescue sweeps, implementation changes,
or repeat execution.

---

## 1. Frozen execution identity

- execution HEAD:
  `22ac4099526893fdd40e7525efe599bcc5194b0d`
- implementation freeze:
  `6f202464b7c933611728ac861c6e7153510c7270`
- run name:
  `gen5-phase2-fresh-ownership-assay-22ac409-r1`
- run command SHA256:
  `1abe31cb491b3fc0261fa1328713895fc00709c753a48b88b4e13772f2207e82`
- imported handoff ZIP SHA256:
  `466dfcc1cf6c2fb936fa079e95bc5e57baead5f181866ad9164c7f6010bf7e5b`
- run start:
  `2026-09-30T12:44:24Z`
- run finish:
  `2026-09-30T13:45:42Z`
- execution exit code:
  `0`

The imported artifact bundle contained exactly six files.

---

## 2. Artifact and provenance validation

The imported evidence passed all post-import validation gates:

- exact six-file artifact boundary: PASS
- output SHA256 checksum validation: PASS
- artifact manifest validation: PASS
- worker provenance validation: PASS
- worker 0 scientific forwards: `24000`
- worker 1 scientific forwards: `24000`
- global CUDA scientific forwards: `48000`
- CPU scientific forwards: `0`
- item evidence validation: PASS
- confirmatory item count: `300`
- summary aggregate recomputation: PASS
- artifact/provenance validity: PASS
- training executed: `False`
- backward executed: `False`
- task evaluation executed: `False`
- row dropping executed: `False`

Therefore execution success, artifact validity, and scientific
interpretability are all established separately.

---

## 3. Confirmatory estimand

The primary comparison was frozen as:

`G5-M1_MINUS_G5-C1`

For each fresh XG1 item, the three training seeds were averaged within
item before confirmatory testing.

The confirmatory endpoint was:

`D_OWN_i = Ibar_M1_i - Ibar_C1_i`

The confirmatory sample size was exactly:

`n = 300`

The only confirmatory test was the pre-specified one-sided one-sample
Student t-test on `D_OWN` with alternative mean greater than zero.

No seed expansion into pseudo-replicates was used.

---

## 4. Validated scientific result

Validated aggregate quantities:

- `mean_Ibar_M1 = 1.3327600706416034e-07`
- `mean_Q_RR_M1 = 2.3840579959434754e-07`
- `mean_S_R_M1 = 1.6427901626504933e-07`
- `mean_D_OWN = 9.5939370052090783e-12`
- `p_one_sided_greater = 0.36700787067950813`

The pre-specified positive conclusion required all of:

1. `mean_Ibar_M1 > 0`
2. `mean_Q_RR_M1 > 0`
3. `mean_S_R_M1 > 0`
4. `mean_D_OWN > 0`
5. one-sided confirmatory `p < 0.05`

Conditions 1-4 were satisfied.

Condition 5 was not satisfied.

Therefore the frozen decision rule yields:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

---

## 5. Scientific interpretation

The assay does not support the claim that constraining the learned
state-update correction to R22 causal-role ownership produces a
confirmatory advantage over the matched C22 control.

The M1 arm retained positive absolute R22-related quantities:
`mean_Ibar_M1`, `mean_Q_RR_M1`, and `mean_S_R_M1` were all positive.

However, the pre-specified M1-minus-C1 ownership contrast was extremely
small in the aggregate:

`mean_D_OWN = 9.5939370052090783e-12`

and its one-sided confirmatory test was not significant:

`p = 0.36700787067950813`.

Accordingly, the experiment distinguishes two statements:

- evidence for positive absolute M1 restoration-related structure was
  observed;
- evidence that M1 ownership is superior to the matched C1 control was
  not established.

This result does not establish statistical equivalence between M1 and
C1, and it does not establish absence of causal-role information in the
native R22 state geometry.

It does reject the stronger frozen Phase 2 claim that R22-aligned
state-update ownership yields a confirmatory advantage over the matched
C22 control under this assay.

---

## 6. Terminal-negative boundary

This is a valid negative terminal outcome under the frozen Phase 2
design.

No rescue search is authorized within this Phase 2 result, including:

- alternate rank,
- alternate layer,
- alternate projector,
- alternate correction width,
- alternate objective,
- additional seeds,
- replacement cohort,
- post-hoc filtering,
- repeated confirmatory execution.

The completed result must be retained as-is.

Any future experiment addressing a materially different scientific
question must be scoped as a new stage rather than as a rescue of this
confirmatory assay.

---

## 7. Final disposition

Code correctness: `PASS`

Execution success: `PASS`

Artifact/provenance validity: `PASS`

Frozen confirmatory claim:

`NOT_ESTABLISHED`

Final scientific label:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`
