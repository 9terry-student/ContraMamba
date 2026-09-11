# K3C Native Recurrence Contribution Decomposition Scientific Report

Scientific verdict: `INCONCLUSIVE_DUE_TO_BASELINE_REPLICATION_FAILURE`

BASE replication gate: `NOT_ESTABLISHED`

Full K3C support: `False`

Execution authority commit: `db75edfbf34bb6efacf94982b18406fd4284de56`

## Integrity

- natural H+W identity: `PASS_EXACT`
- natural structural replay: `PASS_EXACT`
- sham replay: `PASS_EXACT`
- W-seed d-step semantics: `PASS_EXACT`
- WH equalization pair collapse: `PASS_EXACT`

## BASE replication gate

- R: +78 / -68 / 0=4 / U=0; effect=0.0684931506849315; Holm p=0.45648223496512574; match=False
- D: +94 / -52 / 0=4 / U=0; effect=0.2876712328767123; Holm p=0.0012796461326663538; match=True
- DISP: +28 / -118 / 0=4 / U=0; effect=-0.6164383561643836; Holm p=7.20741808493248e-14; match=True
- P: +15 / -131 / 0=4 / U=0; effect=-0.7945205479452054; Holm p=1.0719079124469416e-23; match=True

## Confirmatory mechanism tests

- R_DOM: +77 / -69 / 0=4 / U=0; effect=0.0547945205479452; Holm p=0.5625209423048506; match=False; contradiction=False
- R_CARRY: +43 / -103 / 0=4 / U=0; effect=-0.410958904109589; Holm p=3.698470102279042e-06; match=False; contradiction=True
- D_DOM: +95 / -51 / 0=4 / U=0; effect=0.3013698630136986; Holm p=0.0013577937200149987; match=True; contradiction=False
- D_CARRY: +63 / -83 / 0=4 / U=0; effect=-0.136986301369863; Holm p=0.23107942050385333; match=False; contradiction=False
- DISP_DOM: +118 / -28 / 0=4 / U=0; effect=0.6164383561643836; Holm p=1.6817308864842456e-13; match=True; contradiction=False
- DISP_CARRY: +40 / -106 / 0=4 / U=0; effect=-0.4520547945205479; Holm p=2.6688328759130383e-07; match=False; contradiction=True
- P_DOM: +131 / -15 / 0=4 / U=0; effect=0.7945205479452054; Holm p=2.143815824893883e-23; match=True; contradiction=False
- P_CARRY: +94 / -52 / 0=4 / U=0; effect=0.2876712328767123; Holm p=0.0019194691989995306; match=True; contradiction=False

## Claim boundary

This execution tests only the preregistered layer-23 W-vs-H contribution decomposition and retained-carry hypothesis on the prospectively frozen K3C controlled population.

It does not establish task-decision causality, authorization causality, external-distribution generalization, or K4 claims.
