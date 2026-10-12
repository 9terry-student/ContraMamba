# Gen5 v10 full hybrid — imported evidence and static replay summary

**Evidence status:** `ARTIFACT_INTEGRITY_VERIFIED / FROZEN_STATIC_REPLAY_PASS_REPORTED / SCIENTIFIC_E0_E3_NOT_EVALUATED`  
**Execution HEAD:** `9135810af6c607cc61a7c6b4197e4cebf2eaa059`  
**Runner:** `GEN5_V10_HYBRID_96_CHUNK_GEMM_WITH_FROZEN_TAIL8_CANDIDATE`  
**Run:** `gen5-v10-full-hybrid-9135810-r1`  
**Import:** `VERIFIED_DIRECT_COLLECT_NOT_CM_IMPORT` (not an official v4 `cm import` handoff)  
**Original ZIP SHA256:** `e9aa78842f0fa1822fa8c2a826bbbabe60b636ecd8bfea911c2f8983e9ae6038`

## 1. Evidence checked and boundaries

The original collect ZIP was independently read without training or GPU execution. ZIP CRC, manifest-declared byte counts and SHA256 for all seven payload files, both worker result sidecars, execution HEAD, engine attribution, group dispatch, row coverage, and training-disabled flags were checked against the payloads. All these **artifact checks passed**.

The operator independently ran the repository's *original frozen* static reducers/merger and returned `GEN5_V10_FULL_INDEPENDENT_STATIC_REPLAY_PASS`, with `FROZEN_SCALAR_REPLAY=PASS` and `FROZEN_PARENT_VECTOR_REPLAY=PASS`. This report records that **operator-executed** validation, not a second independently executed replay by this report author.

No scientific E0–E3 promotion, full-cohort FFT↔v10 paired numerical equivalence, independent run attestation, or formal hypothesis test is established by these checks. The original imported `direct_import_audit.json` retains `PENDING` for science validation.

## 2. Execution completeness and observed cost

| Metric | Worker 0 | Worker 1 | Total |
|---|---:|---:|---:|
| Dev rows | 416 | 424 | 840 |
| Group calls | 234 | 252 | 486 |
| B32 calls | 234 | 234 | 468 |
| B8 frozen-tail calls | 0 | 18 | 18 |
| GEMM chunks | 22,464 | 22,464 | 44,928 |
| Fast FFT chunks at B32 | 0 | 0 | 0 |
| Reducer time (seconds) | 3420.294 | 4185.073 | 7605.367 (GPU-seconds, not wall-clock) |
| Worker elapsed (seconds) | 3731.581 | 4502.693 | Parallel wall-time at least 4502.693 sec plus launcher overhead |

Valid token coverage: **60,094**. Each worker produced 36 orientation accumulators. Model/optimizer training was not executed according to both frozen worker payloads.

For the earlier single-group r2, the supplied measured windows were frozen FFT **166.363 s** versus v10 Hybrid **14.689 s** (~**11.33×** faster for that measured group). This is **not** an end-to-end whole-cohort speedup ratio, since full-worker comparison against frozen FFT was not executed here.

## 3. Original frozen scalar and parent replay (operator-executed)

The provided local replay output reported:

| Check | Observation |
|---|---:|
| Merged orientations | 36 |
| Source-matched cells | 9 |
| Max raw reconstruction absolute error | `7.450580596923828e-09` |
| Max fine-parent reconstruction absolute error | `5.966460525641537e-08` |
| Max fine-parent reconstruction relative error | `2441.446932524268` |
| Frozen scalar replay | PASS |
| Frozen parent-vector replay | PASS |

The frozen scalar/vector replay accepts each element only if its **absolute OR relative** error criterion is satisfied. The large reported maximum relative error is therefore not, in isolation, evidence of a failed replay; it remains a flagged diagnostic rather than being silently discarded.

## 4. Descriptive primary minus control comparison

Using merged worker accumulator `E` and `S` values, define `L_interference = log[(E_complement/S_complement)/(E_visible/S_visible)]`. Compute each source's mean across its two `PRIMARY_A` orientations, subtract the mean across its two `CONTROL_R` orientations (identical to the frozen source-matched scalar method, before any inferential testing).

| Source | ΔL (PRIMARY_A − CONTROL_R) |
|---|---:|
| `A6201-R6201` | -1.349906 |
| `A6201-R6202` | -1.194651 |
| `A6201-R6203` | -1.225752 |
| `A6202-R6201` | -0.807187 |
| `A6202-R6202` | -0.788124 |
| `A6202-R6203` | -0.789404 |
| `A6203-R6201` | -1.011708 |
| `A6203-R6202` | -1.037444 |
| `A6203-R6203` | -1.070608 |

Across 9 sources, ΔL was negative in **9/9**; mean **-1.030532**, median **-1.037444**, range **[-1.349906, -0.788124]**. These are **descriptive frozen-dev-set summaries only**, not p-values, replication evidence, proof of causal localization, or an E0–E3 scientific verdict.

## 5. Status by evidence layer

1. **Code and static contract:** 11/11 full worker CPU tests and static source contract passed in the operator's local checkout before execution.
2. **GPU execution:** two T4 workers completed all 486 full-group calls at the pinned HEAD, with distinct v10 engine attribution.
3. **Artifact/provenance:** collect ZIP and local direct-import integrity passed; this was not registered through the official `cm` handoff contract.
4. **Numerical and scientific conclusion:** frozen scalar/parent replay passed in the operator's independent local static run. Full-cohort FFT↔v10 paired equivalence, E0–E3, and causal claims remain **pending / not evaluated**.

**Ship classification:** Preserve as *validated execution artifacts + operator static replay evidence*, not a conclusive scientific or equivalence PASS. No re-training, optimizer step, GPU rerun, or mutation of the frozen implementation is needed for this archival action.
