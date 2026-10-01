# ContraMamba Gen5 Phase 3A
## Contention Qualification Validated Scientific Interpretation Report Candidate

### Status

`VALIDATED_TERMINAL_NEGATIVE_MANIPULATION_QUALIFICATION`

Scientific conclusion:

`GEN5_PHASE3_CAUSAL_ROLE_CONTENTION_NOT_ESTABLISHED`

Phase 3B eligibility:

`BLOCKED`

This report records the validated interpretation of the single frozen
Gen5 Phase 3A contention-qualification execution.

It does not authorize Phase 3B, additional training, stressor sweeps,
alternate PP planes, layer/token search, or confirmatory inference.

---

## 1. Frozen execution identity

- training execution authority HEAD:
  `d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e`
- run:
  `gen5-phase3a-contention-qualification-9cell-d58e894-retry3`
- registered command SHA256:
  `b74bbc76b5addddc5d1647db1da3af3dfec4a708467bf494a767b07bc36ef156`
- imported handoff ZIP SHA256:
  `f87af66ed246c4c3cc052bd5bec3ea376011c7407b87706228f8b7ff15fce243`
- run start:
  `2026-10-01T03:46:32Z`
- run finish:
  `2026-10-01T06:34:53Z`
- exit code:
  `0`
- imported artifact count:
  `38`

---

## 2. Execution and provenance validation

Post-import validation established:

- exact imported file count: `38`
- collection source byte identity: `PASS`
- nine-cell execution manifest: `PASS`
- nine-cell imported artifact validation: `PASS`
- arm: `G5-C0`
- seeds: `6201,6202,6203`
- pressures: `P0,PR,PC`
- cells: `9`
- optimizer steps per cell: `20`
- total optimizer steps: `180`
- dual-GPU topology:
  `TWO_INDEPENDENT_SINGLE_GPU_WORKERS_NO_DDP`
- confirmatory assay loaded: `False`
- scientific p-value count: `0`

The execution therefore establishes valid Phase 3A trained correction
artifacts and valid manipulation-qualification inputs.

---

## 3. Frozen prospective qualification rule

For every seed, PR was required to satisfy all seven conditions:

1. `F_R(PR) >= 0.005`
2. `F_R(PR) >= 10 * F_R(P0)`
3. `F_R(PR) >= 4 * F_R(PC)`
4. `F_R(PR) >= 4 * F_C(PR)`
5. `post_step20_matched_rng_loss < step0_loss`
6. correction parameters finite
7. parent identity preserved

No averaging across seeds may rescue a failed seed.

No p-value is part of this qualification rule.

---

## 4. Validated per-seed results

### Seed 6201

- `F_R(P0) = 0.00023826065374962614`
- `F_R(PR) = 0.00023592660252971006`
- `F_R(PC) = 0.00024792909899578301`
- `F_C(PR) = 0.00080916470058622046`
- `F_R(PR) / F_R(P0) = 0.99020379075107889`
- `F_R(PR) / F_R(PC) = 0.95158899655309481`
- `F_R(PR) / F_C(PR) = 0.29156808540805956`
- step-0 loss:
  `1.2651928663253784`
- post-step20 matched-RNG loss:
  `0.84397387504577637`

Qualification conditions:

- G1 absolute R22 floor: `FAIL`
- G2 PR versus P0 ratio: `FAIL`
- G3 PR versus PC ratio: `FAIL`
- G4 R22 versus C22 under PR: `FAIL`
- G5 loss decrease: `PASS`
- G6 finite correction: `PASS`
- G7 parent identity: `PASS`

Seed result:

`FAIL`

### Seed 6202

- `F_R(P0) = 0.00013934667212399358`
- `F_R(PR) = 0.0001376137678244099`
- `F_R(PC) = 0.00014631138115233841`
- `F_C(PR) = 0.00079074518088241097`
- `F_R(PR) / F_R(P0) = 0.98756407832946524`
- `F_R(PR) / F_R(PC) = 0.94055408909801341`
- `F_R(PR) / F_C(PR) = 0.17403048561212031`
- step-0 loss:
  `1.258916974067688`
- post-step20 matched-RNG loss:
  `0.8397899866104126`

Qualification conditions:

- G1 absolute R22 floor: `FAIL`
- G2 PR versus P0 ratio: `FAIL`
- G3 PR versus PC ratio: `FAIL`
- G4 R22 versus C22 under PR: `FAIL`
- G5 loss decrease: `PASS`
- G6 finite correction: `PASS`
- G7 parent identity: `PASS`

Seed result:

`FAIL`

### Seed 6203

- `F_R(P0) = 0.00023961367801937799`
- `F_R(PR) = 0.00023838288516485359`
- `F_R(PC) = 0.00024441715090972474`
- `F_C(PR) = 0.00057798077681896611`
- `F_R(PR) / F_R(P0) = 0.99486342822873053`
- `F_R(PR) / F_R(PC) = 0.97531161081613338`
- `F_R(PR) / F_C(PR) = 0.412440853962033`
- step-0 loss:
  `1.2649465799331665`
- post-step20 matched-RNG loss:
  `0.84300601482391357`

Qualification conditions:

- G1 absolute R22 floor: `FAIL`
- G2 PR versus P0 ratio: `FAIL`
- G3 PR versus PC ratio: `FAIL`
- G4 R22 versus C22 under PR: `FAIL`
- G5 loss decrease: `PASS`
- G6 finite correction: `PASS`
- G7 parent identity: `PASS`

Seed result:

`FAIL`

---

## 5. Scientific interpretation

The optimization procedure itself behaved normally.

For every PR seed:

- final matched-RNG loss decreased substantially from step 0;
- correction parameters remained finite;
- parent identity remained preserved.

Therefore the qualification failure is not attributable to a failed
optimizer, non-finite correction, or parent mutation.

However, the frozen PP3 training-time stressor did not induce the
prospectively required R22 contention.

Across all three seeds:

- `F_R(PR)` remained near the corresponding P0 value;
- `F_R(PR)` remained below the corresponding PC value;
- the absolute `F_R(PR)` values remained far below `0.005`;
- R22 occupancy under PR remained below matched C22 occupancy.

The PR/P0 ratios were approximately:

- `0.9902`
- `0.9876`
- `0.9949`

rather than the prospectively required ratio of at least `10`.

The PR/PC ratios were approximately:

- `0.9516`
- `0.9406`
- `0.9753`

rather than the required ratio of at least `4`.

Thus the upstream causal-role stressor did not create R22-specific
optimization pressure under this frozen Phase 3A design.

---

## 6. What this result does and does not establish

This result establishes:

`THE_PROPOSED_FROZEN_PP3_TRAINING_STRESSOR_DID_NOT_ESTABLISH_R22_CONTENTION`

It does not establish:

- that causal-role contention is impossible;
- that R22 lacks causal importance;
- that ownership has no effect when genuine contention exists;
- that Phase 3B C1/M1 would be negative;
- statistical equivalence of P0, PR, and PC;
- universal absence of optimization-time contention.

The Phase 3B ownership-by-contention hypothesis was not tested because
its prospective manipulation prerequisite failed.

This distinction is mandatory.

---

## 7. Relation to Phase 2

Phase 2 remains independently frozen as:

`GEN5_R22_STATE_UPDATE_OWNERSHIP_ADVANTAGE_NOT_ESTABLISHED`

Phase 3A attempted to establish the missing independent variable:

`TRAINING_TIME_CAUSAL_ROLE_CONTENTION`

The proposed PP3 manipulation did not establish that variable.

The result is therefore consistent with, but stronger and more specific
than, merely observing low natural R22 occupation in Phase 2:

even under the frozen upstream PP3 training stressor, the learned
unrestricted correction did not become preferentially R22-occupying.

This does not convert Phase 2 into a positive or negative contention
experiment retroactively.

---

## 8. Terminal boundary

The frozen Phase 3 design requires Phase 3A PASS before Phase 3B.

Because all three seeds fail multiple required geometry conditions:

`PHASE3A_CONTENTION_GATE = FAIL`

Therefore:

`GEN5_PHASE3_CAUSAL_ROLE_CONTENTION_NOT_ESTABLISHED`

and:

`PHASE3B_ELIGIBILITY = BLOCKED`

No rescue action is authorized within this Phase 3 design, including:

- stressor-strength sweep;
- alternate PP plane;
- alternate layer;
- alternate token;
- alternate owner rank;
- correction-rank search;
- extra seeds;
- threshold modification;
- response-conditioned cohort selection;
- Phase 3B execution.

Any future attempt to create contention would require a genuinely new
scientific stage and a prospectively different manipulation hypothesis,
not a rescue of this result.

---

## 9. Final disposition

Code/runtime correctness: `PASS`

Nine-cell execution success: `PASS`

Artifact/provenance validity: `PASS`

Training optimization success: `PASS`

Prospective contention manipulation qualification: `FAIL`

Scientific p-value count: `0`

Phase 3B: `BLOCKED`

Final scientific label:

`GEN5_PHASE3_CAUSAL_ROLE_CONTENTION_NOT_ESTABLISHED`
