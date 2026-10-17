# Gen5 G2-B — P0→H2 pair-excluded decision remapping residual audit

**Classification:** Exploratory, CPU-only static replay of previously frozen logits; **not** an independent holdout, confirmatory validation, architecture/evaluation authorization, causal mediation result, or Phase3B promotion.

**Status:** `G2_B_FROZEN_P0_H2_STATIC_REPLAY_PASS` reported on 2026-10-17; per-run hash, pair, shard and schema checks PASS in supplied stdout. The present report is a faithful interpretation of that stdout and the exact replay script; it does **not** assert that the GPU model was rerun or that a separate reviewer independently reran this CPU calculation.

## 1. Exact inputs and executed analysis

- Repository / branch: `9terry-student/ContraMamba`, `gen5-causal-role-state-ownership`.
- Replay working-tree HEAD: `444f504683c1f2284f9cbc52e59de1d4426cc8cd`; status CLEAN; no file writes.
- P0 execution HEAD: `75ad31e42e1baed7d99afbe9281e36276dfd1992`; frozen 840-row, 120-pair, 7-cell P0 dev logits, 2 workers.
- Worker 0 SHA256: `d0432108dfba731bc0eaf78854faa043842b5d9cebfa6183c867b6d39fd330de`; rows `[0,416)`.
- Worker 1 SHA256: `dd79c803dee3b5fe96822ec9d8074e24abbf6624e35bb93988571e7d55fcabc4`; rows `[416,840)`.
- H2 execution HEAD: `3c9fc396acb2621c001432ab5b6a2bc748a25a94`; H2 frozen-archive HEAD: `9db4914de6751dd69aeeb5f6da174398623f848a`.
- H2 120-pair / 360-row summary SHA256: `6aa730522416e688d3105dbdaccd9d04695726bf78ecfebe766e68671ebe8491`; all six seed-arm JSON files verified against its listed SHA256.
- **Exact replay source:** `provenance/g2_b_input_dependent_residual_cpu_444f504_v2.py`, SHA256 `bf91236f895fd9e3bfd599b7fe9f01849c13095c1794e5aa74c5751e812230f6`. Its `HEAD` check intentionally pins the **original analysis checkout**; reproducibility requires a clean checkout/worktree at that immutable HEAD, not running this archived file unmodified at a later report-commit HEAD.
- The first local script revision stopped before results because it incorrectly assumed 420/420 worker shards. The exact v2 script changes the two shard/row-count checks to the actual frozen 416/424 split. This correction changes no regression family, fold, predictor, or science hypothesis.

No source/tokenizer refresh, model forward, gradient, checkpoint modification, H2 label fitting, GPU work, or scientific evaluation was performed in the replay.

## 2. Analysis design and residual definition

Pairs were sorted lexicographically by ID, partitioned into five 24-pair folds; each fold's 24 pairs contributes all seven P0 cells to the held-out set. For each fold, train on the remaining 96 pairs / 672 P0 rows. Apply only that fold's P0-trained coefficients to the 24 excluded pairs' P0/H2 logits. **No H2 refitting.** The split and quadratic comparator were specified before seeing this replay output, but after earlier H2 results; all conclusions remain exploratory. The three heldout donor seeds share the same 120 pairs and are not statistically independent replication cohorts.

Map the two decision coordinates `u=(z_REFUTE-z_SUPPORT, z_REFUTE-z_NOT_ENTITLED)` from ZERO to FULL. M0 is a constant intercept, M3 an axis-wise linear map, M4 a general affine map, M5 a fixed full quadratic basis `(1,u1,u2,u1²,u1*u2,u2²)` with prespecified normalized ridge `0.01*672` on non-intercept coefficients. Ordinary fixed least-squares comparator fits use minimal diagonal stabilization.

For a contrast `AB`, the signed RS residual is:

`epsilon_AB = (u_FULL_RS(A)-u_FULL_RS(B)) - (g_P0(u_ZERO(A))_RS-g_P0(u_ZERO(B))_RS)`.

NC=`DENIES-CONFIRMS`; NA=`DENIES-ASSERTS`; CA=`CONFIRMS-ASSERTS`. Exact relation `epsilon_NA=epsilon_NC+epsilon_CA` means these are **not three independent results**. 2D contrast RMSE uses both RS/RN coordinate errors, whereas the signed residual table below uses RS only. A negative residual is underprediction of the magnitude of a negative FULL contrast; it is not evidence of a semantic write.

NEG DENIES matches frozen P0 C6 within the source's numerical tolerance. In RS/RN coordinates, maximum reported absolute differences ZERO `9.4473362e-06`, FULL `2.41445377e-07` to `6.09084964e-07` across seeds. These are coordinate differences (differences of logits), not an independent stochastic error floor.

## 3. P0 fit and H2 contrast transfer

### P0 overall OOF 2D RMSE

| Model | Seed 6201 | Seed 6202 | Seed 6203 |
|---|---:|---:|---:|
| M0 constant | 0.023237 | 0.013434 | 0.020017 |
| M3 diagonal | 0.015470 | 0.009928 | 0.013935 |
| M4 affine | 0.014826 | 0.009485 | 0.013367 |
| M5 quadratic | **0.014452** | **0.009411** | **0.013165** |

### H2 NC/NA/CA 2D contrast RMSE after P0-only transfer

| Model | Seed | NC | NA | CA |
|---|---:|---:|---:|---:|
| M3 diagonal | 6201 | 0.019658 | 0.020350 | 0.001623 |
| M4 affine | 6201 | 0.021201 | 0.021909 | 0.001483 |
| M5 quadratic | 6201 | **0.016834** | **0.017456** | **0.001359** |
| M3 diagonal | 6202 | 0.010512 | 0.010886 | 0.000939 |
| M4 affine | 6202 | 0.011652 | 0.012026 | 0.000843 |
| M5 quadratic | 6202 | **0.010230** | **0.010581** | **0.000808** |
| M3 diagonal | 6203 | 0.017464 | 0.017961 | 0.001353 |
| M4 affine | 6203 | 0.018978 | 0.019484 | 0.001224 |
| M5 quadratic | 6203 | **0.015653** | **0.016105** | **0.001140** |

M5 has the lowest aggregate 2D NC/NA/CA contrast errors among the tested M3/M4/M5 maps for each seed. This is a **partial predictive improvement for this fixed family**, not proof that all nonlinear decision maps fail or that the correction depends on semantic annotations.

### Signed RS residual (M5; H2 contrast minus P0-trained prediction)

| Seed | NC mean | NC negative pairs | NC RMS / FULL NC RMS | NA mean | NA negative pairs | CA mean |
|---|---:|---:|---:|---:|---:|---:|
| 6201 | −0.017377 | 114/120 | 0.49717 | −0.017913 | 111/120 | −0.000536 |
| 6202 | −0.010957 | 115/120 | 0.57243 | −0.011213 | 113/120 | −0.000256 |
| 6203 | −0.017389 | 118/120 | 0.52823 | −0.017778 | 116/120 | −0.000389 |

M5 NC RMS residual / ZERO NC RMS is 1.786%, 1.089%, 1.663%, respectively. It remains nonzero, but is relatively small against the original input-scale contrast. The larger relative error against FULL is partly due to the very small FULL contrast; **report both denominators**. CA is not error-free (its M5 residual RMSE / FULL CA RMS is 64.5%, 72.2%, 66.5%).

### Pre-existing C6 error and row-level error composition

| Seed | M3 P0 C6 OOF 2D RMSE | M5 P0 C6 OOF 2D RMSE | M5 H2 NEG 2D RMSE |
|---|---:|---:|---:|
| 6201 | 0.022497 | 0.022085 | 0.022085 |
| 6202 | 0.015220 | 0.014986 | 0.014986 |
| 6203 | 0.020238 | 0.020159 | 0.020159 |

The M5 row RS residual means are NEG/C6 `−0.008240, −0.004875, −0.008019`, versus CONFIRMS `+0.009137, +0.006082, +0.009369`, and ASSERTS `+0.009672, +0.006338, +0.009758` across seeds 6201/6202/6203. Thus negative NC/NA residuals combine already-existing NEG/C6 approximation error with opposite-signed POS errors. The residual cannot be assigned uniquely to newly constructed POS inputs.

## 4. ZERO decision-space support limitation

The support calculations use **ZERO logits only**; donor seed has no effect on these inputs, so the same coverage numbers are repeated for all three seeds. The P0-heldout nearest-neighbor radius is calibrated at its 95th percentile in each fold, rather than fitted to H2.

| H2 condition | Within fold-calibrated P0 NN95 radius | In common N/C/A axis-aligned rectangle |
|---|---:|---:|
| NEG_DENIES | 114/120 | 94/120 |
| POS_CONFIRMS | 103/120 | 57/120 |
| POS_ASSERTS | 93/120 | 53/120 |

- Condition cross-NN coverage relative to the query condition's own within-condition 95th-percentile NN radius: `N→C=92/120`, `C→N=55/120`, `N→A=85/120`, `A→N=53/120`, `C→A=118/120`, `A→C=114/120`.
- Shared rectangular coordinate bounds: RS `[1.1861461,2.7694927]`, RN `[1.6916302,2.9350047]`. Rectangle inclusion is **only a weak necessary geometric screen, not sufficient evidence of density overlap**. The directed nearest-neighbor asymmetry likewise does not prove zero overlap.
- In particular, C→N and A→N each cover fewer than half the 120 POS points under this descriptive rule. The current data provide inadequate common-support evidence to uniquely separate condition-associated residual from an input-location or extrapolation effect. No post-hoc hard significance/selection threshold is asserted.

## 5. Final interpretation and stop rule

1. **Confirmed within this exploratory CPU replay:** source SHA/schema/pair identity; P0→H2 NEG numerical correspondence; predeclared M5 outperforms M3/M4 on aggregate H2 contrasts; residual remains of systematic sign; P0 C6 approximation error already exists; ZERO-space overlap is asymmetric/partial.
2. **Not shown:** condition-specific computation not reducible to general ZERO-space nonlinearity, semantic polarity understanding/revision, output-based accuracy recovery, independent holdout generalization, causal effect of a particular layer-22 temporal gate, native recurrent state write mediation, or invalidity of all nonlinear remappings.
3. **Disposition:** `G2_B_EXPLORATORY_REMAP_PARTIAL_SUPPORT_LIMITED` / `SEMANTIC_SPECIFICITY_NOT_IDENTIFIABLE_FROM_CURRENT_STATIC_EVIDENCE`. This is a methodological stopping point for G2 mechanism expansion, not proof that no selective effect exists. Do not search additional flexible maps or thresholds on the same 120 pairs to force a conclusion.
4. **Next research priority:** Return to the separate native recurrent-transport / local causal-intervention research stream only after identifying its current applicable stage authority and artifacts. The H3 CPU-only temporal gate can remain frozen as an instrument; **do not authorize an H3 CUDA run from this report**.

`SCIENCE_SCOPE=EXPLORATORY_STATIC_ONLY; TRAINING=NO; CUDA_RUN=NO; CAUSAL_MEDIATION=NOT_ESTABLISHED; PHASE3B=BLOCKED_UNCHANGED`.
