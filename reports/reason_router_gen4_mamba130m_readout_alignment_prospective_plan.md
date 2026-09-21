# ContraMamba Gen4 — Mamba-130M Readout-Alignment Prospective Plan

Status: PROSPECTIVE_PLAN_ONLY
Execution authorized by this file: NO
Training authorized: NO

## 1. Scientific purpose

This study asks whether the frozen response-blind P3-versus-P5 causal geometry at
Mamba-130M is locally aligned with the downstream correct-class task-margin readout
on the same matched population used by the completed seed181 behavioral bridge.

The completed 130M behavioral bridge is already known to have a positive
restored-versus-control downstream margin effect. This readout study is therefore
a mechanistic follow-up, not an independent replication of that behavioral result.

No result from the completed 370M/1.4B readout study or the later low-displacement
study may alter the 130M checkpoint, population, plane identities, intervention
coordinate, endpoint, or inferential test defined here.

## 2. Frozen model identity

Use exactly:

- scale: `mamba130m`
- Hugging Face repository: `state-spaces/mamba-130m-hf`
- revision: `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`
- seed: `181`
- arm: `G3-GROUP-D-HALF`
- selected epoch: `19`
- checkpoint SHA256:
  `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`
- checkpoint bytes: `518270455`

The checkpoint must be authenticated against the existing seed181 registry and may
not be replaced by the historical representative checkpoint used by the separate
small-epsilon internal-Q robustness line.

## 3. Frozen geometry

Reuse exactly the completed seed181 checkpoint-replication geometry:

- homolog / selected plane: `P3`
- response-blind matched control plane: `P5`
- geometry checkpoint SHA256:
  `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f`
- geometry artifact:
  `reports/reason_router_gen4_seed181_checkpoint_replication_runs/`
  `g4k-seed181-checkpoint-replication-8e96fd1-retry1/`
  `seed181_principal_geometry.json`
- tensor artifact:
  same directory, `seed181_principal_geometry.pt`

No plane reselection, coefficient fitting, geometry refit, or response-conditioned
selection is permitted.

## 4. Frozen population

Use exactly the existing seed181 behavioral-bridge population:

- family: `XG1`
- pair range: `xg1_fact_2701..xg1_fact_3000`
- N = `300`
- cells per pair:
  - `C0_SHAM`
  - `C2_NAME`

Frozen correct labels:

- `C0_SHAM -> SUPPORT -> class 2`
- `C2_NAME -> NOT_ENTITLED -> class 1`

This population is reused because the scientific question is whether local readout
alignment explains the already-frozen 130M behavioral bridge on the same matched
pairs. No fresh cohort is introduced for this mechanistic follow-up.

## 5. Frozen intervention/readout coordinate

Reuse the existing 130M seed181 behavioral intervention coordinate:

- intervention layer: `17`
- anchor: `A_IDENTITY`
- target offset: `+2`
- tensor coordinate:
  content half of `mixer.in_proj` at the frozen target token
- frozen strong-coordinate mask / dimension inherited from the seed181 geometry

No layer search, token search, anchor substitution, or coordinate change is allowed.

## 6. Native task margin and local gradient

For row `r` with correct class `y`, define:

`m_r = z_y - max_{c != y} z_c`

Evaluate the gradient only at the unmodified native forward state.

Let `x_r` be the layer-17 target-token content-half activation at the frozen
`mixer.in_proj` coordinate.

Define:

`g_r = d m_r / d x_r`

Implementation semantics:

- all model parameters remain frozen;
- no optimizer or training step exists;
- the upstream graph is cut at the intervention tensor;
- the intervention tensor is treated as a local leaf;
- the downstream graph remains intact;
- no P3/P5 intervention is applied during gradient measurement;
- exact wrong-class argmax is recorded;
- exact wrong-class ties are technical blockers and do not permit row deletion.

## 7. Frozen selected/control components

In frozen strong coordinates let:

`U_3 = [u_3,+, u_3,-]`
`U_5 = [u_5,+, u_5,-]`

For native strong activation `h`:

`a = h^T u_3,+`
`b = h^T u_3,-`

Define:

`C_3(h) = a u_3,+ + b u_3,-`

`C_5(h) = a u_5,+ + b u_5,-`

The coefficient-matched local selected-versus-control displacement is:

`Delta_h = C_3(h) - C_5(h)`

No coefficient fitting is performed.

## 8. Per-row readout-alignment quantity

For every row compute:

`L_3 = g^T C_3(h)`

`L_5 = g^T C_5(h)`

`Delta_L_row = L_3 - L_5 = g^T Delta_h`

Also save descriptively:

- `||g||_2`
- `||Pi_3 g||_2`
- `||Pi_5 g||_2`
- selected/control projection fractions relative to `||g||_2`
- `U_3^T g`
- `U_5^T g`
- `||C_3(h)||_2`
- `||C_5(h)||_2`
- selected/control cosine values when denominators are nonzero
- correct-class ID
- active wrong-class argmax ID
- finite/shape/provenance audit fields

`Delta_L_row` is a local first-order task-margin readout quantity. It is not a
behavioral intervention result.

## 9. Pair aggregation

For source pair `q`:

`Delta_L_q = 0.5 * (Delta_L_q,C0_SHAM + Delta_L_q,C2_NAME)`

The inferential sample is the 300 pair-level values.

No row may be filtered by gradient norm, prediction correctness, margin, sign,
or `Delta_L`.

## 10. Primary hypothesis

Exactly one primary inferential test is permitted.

`H0: E[Delta_L_130M] <= 0`

`H1: E[Delta_L_130M] > 0`

Test:

- one-sided one-sample Student t-test
- N = `300`
- df = `299`
- alpha = `0.05`
- exactly one primary p-value
- no multiplicity correction

Support requires both:

1. `mean(Delta_L_130M) > 0`
2. one-sided `p < 0.05`

If either fails, positive 130M readout alignment is not established.

## 11. Secondary descriptive outputs

With zero additional inferential p-values, report:

- mean, sample SD, population SD, median, quartiles, min/max of pair `Delta_L`
- fraction `Delta_L > 0`
- gradient norm distribution
- P3/P5 gradient projection fractions
- directional-coordinate summaries
- selected/control cosine summaries
- zero-gradient count
- wrong-class tie count
- nonfinite count

After raw evidence is frozen, a CPU-only static merge may read the already-frozen
130M behavioral bridge `D_BEH` values from `xg1_fact_2701..3000` and report:

- Pearson correlation between `Delta_L` and `D_BEH`
- Spearman correlation between `Delta_L` and `D_BEH`
- sign agreement
- residual `D_BEH - Delta_L` distribution

These merge quantities are descriptive only and add no p-value.

## 12. Three-scale contextual synthesis boundary

After the 130M result is frozen, it may be placed descriptively beside the already
frozen 370M and 1.4B readout-alignment results.

Allowed descriptive synthesis:

- mean `Delta_L` at 130M, 370M, and 1.4B
- fraction-positive at each scale
- checkpoint/plane identity table

Not allowed in this stage:

- new three-scale trend p-value
- monotonicity test
- interpolated zero-crossing estimate
- scale-threshold optimization
- new plane selection
- new cohort introduced in response to the 130M result

## 13. Technical gate

Before scientific execution, a bounded technical gate must verify:

- exact model/checkpoint/runtime identity
- exact frozen P3/P5 geometry identity
- exact XG1 2701..3000 structural population identity
- exact A_IDENTITY anchor and `+2` target coordinate
- local activation leaf construction
- finite task margin
- gradient shape and finiteness
- no parameter gradient creation
- no optimizer/training state

The gate must retain only capability booleans. It must not retain numeric
`Delta_L`, population summaries, p-values, or a scientific conclusion.

## 14. Preferred execution shape

Use the two available T4 GPUs as execution shards for the single 130M scale:

- GPU0: `xg1_fact_2701..2850`
- GPU1: `xg1_fact_2851..3000`

Each pair contributes exactly two native rows.

Per shard:

- 150 pairs
- 300 native full-model forwards
- 300 local backward evaluations

Total scientific execution:

- 300 pairs
- 600 native full-model forwards
- 600 local backward evaluations
- zero intervention-condition forwards
- zero training steps
- zero parameter updates

GPU assignment is execution-only; the statistical sample is the merged 300 pairs.

## 15. Raw artifact contract

Freeze exactly:

1. `readout_alignment_items.jsonl`
2. `raw_readout_alignment_summary.json`
3. `artifact_manifest.json`
4. `SHA256SUMS.txt`

Raw GPU execution computes:

- no p-value
- no primary decision
- no behavioral merge
- no three-scale synthesis

## 16. Static analysis after raw freeze

Only after the four raw artifacts are frozen:

1. verify exact pair/cell coverage;
2. aggregate row `Delta_L` to pair `Delta_L`;
3. execute the single prespecified one-sided t-test;
4. apply the positive mean sign gate;
5. freeze the result;
6. optionally perform the descriptive-only merge with frozen 130M `D_BEH`;
7. optionally produce the descriptive three-scale context table.

## 17. No-rescue rules

After any 130M gradient/readout result is observed, prohibit:

- row filtering
- alternative cohort
- layer/token/anchor changes
- plane/control changes
- gradient normalization replacing raw `Delta_L`
- alternative primary endpoint
- alternative inferential test
- second primary p-value
- checkpoint substitution
- epsilon/intervention tuning
- subgroup rescue

## 18. Interpretation boundary

If the primary test and sign gate pass, the supported claim is limited to:

> On the frozen seed181 Mamba-130M checkpoint and the matched XG1
> 2701..3000 behavioral-bridge population, the local downstream task-margin
> readout is positively aligned with the frozen response-blind P3-versus-P5
> causal displacement.

This does not by itself establish:

- monotonic scaling across 130M, 370M, and 1.4B;
- a universal scale threshold;
- causal mediation of the full behavioral effect;
- exact calibration of the first-order approximation;
- universal P3 identity across scales.
