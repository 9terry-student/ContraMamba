# Gen5 G2-P0 — Within-pair decision-contrast static follow-up

**Status:** VALIDATED_EXPLORATORY_STATIC_FINDING; NOT_CONFIRMATORY

**Parent frozen findings commit:** `8b70dd105afc7e5638f2e12fee389d864326be3c`

**Original GPU science commit:** `75ad31e42e1baed7d99afbe9281e36276dfd1992`

**Run:** `gen5-g2-p0-full-75ad31e-r1`

**Data:** Frozen Phase3A P0 dev, 120 distinct factual pairs × 7 contrast cells = 840 rows; 3 fixed seeds × 6 arms. No new GPU execution, model invocation, training, or confirmatory data access was required for this follow-up.

## 1. Evidence and validation boundary

The five original imported artifacts and initial scientific interpretation were frozen at `8b70dd1`. The original result files were not changed. The original run had a successful 2-T4 execution, validated collection/import, and independent row-level CE/logit aggregate replay. This follow-up computes additional within-pair contrasts from the same frozen logits. It is **exploratory**, even though the specified static checks have been independently reproduced on the local imported artifacts.

Worker source file identities (SHA256):

- `worker0_science.json`: `d0432108dfba731bc0eaf78854faa043842b5d9cebfa6183c867b6d39fd330de`
- `worker1_science.json`: `dd79c803dee3b5fe96822ec9d8074e24abbf6624e35bb93988571e7d55fcabc4`

Source path: `reports/reason_router_gen5_g2_functional_science_runs/gen5-g2-p0-full-75ad31e-r1/`.

For every one of the 120 pairs the frozen cell-to-label identity is: `C0_SHAM → SUPPORT` (label 2), `C1_TITLE/C2_NAME/C3_ROLE/C4_PREDICATE/C5_TITLE_NAME → NOT_ENTITLED` (label 1), and `C6_EXPLICIT_DENIAL → REFUTE` (label 0). The pair identity and class mapping are essential to the contrast; rows must not be treated as independent or re-paired using position alone.

**Static independent verification reported by the local operator:** `G2_WITHIN_PAIR_DECISION_CONTRAST_AUDIT_PASS`, with exact parent HEAD, both worker hashes, 120 complete unique seven-cell pairs, 840 distinct row IDs, and all nine FULL/SHARED_ONLY/SHARED_NORMMATCH pair contrasts checked. This follow-up is *not* an independent GPU re-execution.

## 2. Primary within-pair observable

For logits `z_R`, `z_N`, and `z_S`, let `R–S` mean `z_R - z_S`. Define for each factual pair `p`:

`D_RS(p, arm) = (z_R - z_S)[p, C6_EXPLICIT_DENIAL, arm] - (z_R - z_S)[p, C0_SHAM, arm]`.

The preferred sign is positive for the contrast between the REFUTE and SUPPORT cases. A shared, input-independent additive class-logit bias cancels exactly in `D_RS`. Thus the observed sign reversal cannot be explained by a *constant* class-logit offset alone. This does not establish which internal model layer produces the input-dependent effect.

| Condition | Seed 6201: mean D_RS | Seed 6202 | Seed 6203 | Pair count with D_RS > 0, per seed |
|---|---:|---:|---:|---:|
| ZERO | +2.759805 | +2.759805 | +2.759805 | 120/120 |
| FULL | −0.058027 | −0.029036 | −0.048808 | 0/120 |
| SHARED_ONLY | −0.216966 | −0.226244 | −0.233549 | 0/120 |
| SHARED_NORMMATCH | −0.018696 | −0.026183 | −0.022516 | 0/120 |

The FULL and SHARED_NORMMATCH means are close to zero as well as negative, so a global conclusion of large-magnitude semantic reversal is not warranted. The exact sign in this frozen population is negative for every pair; that is distinct from claiming the same sign would persist under perturbation, quantization, distribution shift, or a different split.

## 3. Attribution to contrast cells at the final-logit level

Define `Delta_RS(c, arm) = (z_R - z_S)[c, arm] - (z_R - z_S)[c, ZERO]`. All values below are 120-pair means, not a new latent-state measurement.

| Arm | Seed | Mean Delta_RS at C6 (REFUTE) | Mean Delta_RS at C0 (SUPPORT) | C6 negative pairs |
|---|---:|---:|---:|---:|
| FULL | 6201 | −2.423609 | +0.394223 | 120/120 |
| FULL | 6202 | −2.371397 | +0.417443 | 120/120 |
| FULL | 6203 | −2.408015 | +0.400598 | 120/120 |
| SHARED_ONLY | 6201 | −2.663392 | +0.313378 | 120/120 |
| SHARED_ONLY | 6202 | −2.679544 | +0.306505 | 120/120 |
| SHARED_ONLY | 6203 | −2.689695 | +0.303659 | 120/120 |
| SHARED_NORMMATCH | 6201 | −2.354898 | +0.423603 | 120/120 |
| SHARED_NORMMATCH | 6202 | −2.370794 | +0.415195 | 120/120 |
| SHARED_NORMMATCH | 6203 | −2.361914 | +0.420407 | 120/120 |

The principal negative final-logit shift is associated with `C6_EXPLICIT_DENIAL`, while the mean change in `C0_SHAM` is positive. This is a **condition-dependent final-logit observation**, not yet a verified state-space mechanism.

## 4. Distinguish absolute decision collapse from surviving pair ranks

Define the REFUTE-vs-NE score `s_RN = z_R - z_N`. For each pair, evaluate whether `s_RN(C6)` is strictly greater than `s_RN(C1..C5)` for all five NOT_ENTITLED cells. This is a *relative rank diagnostic*; it does not count as correct class prediction.

| Arm | Seed 6201 | Seed 6202 | Seed 6203 |
|---|---:|---:|---:|
| ZERO | 120/120 | 120/120 | 120/120 |
| FULL | 115/120 | 112/120 | 113/120 |
| SHARED_ONLY | 116/120 | 116/120 | 116/120 |
| SHARED_NORMMATCH | 110/120 | 112/120 | 112/120 |

Thus the final logits simultaneously show strong preservation of one *within-pair ranking* and total majority-class prediction collapse in FULL/SHARED_ONLY. There is no contradiction: absolute margins, pairwise rankings, and 3-way argmax answer different questions. The REFUTE–SUPPORT *pair contrast* is overturned even where REFUTE–NE *relative ranking* survives.

## 5. Hypothesis for the next, separately authorized stage

**G2-H1 (mechanism candidate):** On frozen explicit-denial inputs the layer-22 correction write disproportionately suppresses decision-relevant REFUTE–SUPPORT distinctions relative to paired sham inputs. The final-logit contrast inversion has been measured; the internal layer-22 origin and necessity of this mechanism have **not** been established.

A minimal proposed observation for a future state-kinematics stage is the correction-branch write tensor at layer 22, before downstream layers, from the existing frozen wrapper. The currently frozen implementation calls `accelerated_correction_mixer_contribution(...)` with the controlled `effective_mask`, then adds it to the original native mixer output. The observation must preserve those native-call and masking semantics. Its initial read-only diagnostics should be pair-matched `C6 − C0` contrasts at valid, predeclared token locations, alongside the downstream `D_RS` score, without constructing a new latent basis or tuning a coefficient.

Predefine failure conditions before execution: (i) the proposed within-pair state contrast has no reproducible relationship to the final-logit inversion; (ii) the same variation is equally present in unrelated contrast cells; (iii) purported selectivity disappears under class-prior-only controls or token-mask integrity checks; or (iv) the observable can be explained only after post-hoc selection of layer/token/direction. An association from observational instrumentation alone cannot prove causal necessity or sufficiency; any intervention to test those claims requires a distinct authorization and numerical-equivalence checks.

The next phase is **static specification/inspection only** unless explicitly authorized. No Phase3B promotion, new GPU inference, model training, optimizer, checkpoint changes, or scientific confirmation is implied by this report.

## 6. Scientific interpretation and limitations

- The original G2-P0 majority-class collapse finding remains frozen and valid. High aggregate CE retention is not evidence of preserved three-way discrimination.
- Newly validated: `D_RS` is positive in all 120 ZERO pairs and negative in all 120 pairs for each of FULL, SHARED_ONLY, and SHARED_NORMMATCH across the three fixed seeds.
- Newly validated: mean C6 REFUTE–SUPPORT logit change is strongly negative for those arms, whereas mean C0 change is positive.
- Newly validated: most pairs retain correct *relative* REFUTE–NE ranking, showing that distinct decision-contrast quantities can behave differently under the same intervention.
- **Not established:** an interpretable causal latent, an identifiable layer-22 state mediator, out-of-sample generalization, statistical independence of 840 rows or of the three seeds, or a corrected task-functional shared core.

**Disposition:** Record as a bounded, exploratory, independently replayed within-pair static follow-up to the closed G2-P0 negative scientific result. Maintain `PHASE3B=BLOCKED_UNCHANGED`.
