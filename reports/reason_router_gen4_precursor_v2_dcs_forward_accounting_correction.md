# ContraMamba Gen4 — Precursor v2 Dynamic Causal Susceptibility Forward-Accounting Correction

## Status

`STATIC_PRECURSOR_V2_DCS_FORWARD_ACCOUNTING_CORRECTION`

This artifact corrects one arithmetic error in:

`reports/reason_router_gen4_precursor_v2_dynamic_causal_susceptibility_stage_a_design.md`

Frozen design commit:

`412b8a5af45b4f3c10712bc184ab7f983c2cc5bc`

The error was detected after the response-free N=800 cohort was frozen and **before**:

- Precursor-v2 susceptibility runner implementation;
- CPU/CUDA equivalence execution;
- any Precursor-v2 model forward;
- any Precursor-v2 generation response;
- any Precursor-v2 p-value.

No scientific response was inspected in detecting this error.

---

## 1. Error

The frozen design correctly defines, at each of four offsets:

- P3 plus basis direction at `+epsilon` and `-epsilon`;
- P3 minus basis direction at `+epsilon` and `-epsilon`;
- P5 plus basis direction at `+epsilon` and `-epsilon`;
- P5 minus basis direction at `+epsilon` and `-epsilon`.

That is:

`2 planes × 2 basis directions × 2 epsilon signs = 8 probe forwards / offset`

The design then incorrectly stated:

`16 full-prefix probe forwards / row`

This omitted a factor of two across the four offsets.

---

## 2. Correct accounting

Frozen temporal offsets:

`4`

Probe forwards per offset:

`8`

Therefore:

`4 × 8 = 32 susceptibility probe forwards / row`

Native forced-decisive generation remains:

`12 full-prefix forwards / row`

Therefore the corrected total is:

`12 + 32 = 44 scientific full-model forwards / row`

For frozen Stage A cohort size:

`N = 800`

the corrected total scientific forward budget is:

`800 × 44 = 35200`

Therefore the authoritative forward accounting is:

- native generation forwards / row: `12`;
- susceptibility probe forwards / offset: `8`;
- susceptibility probe forwards / row: `32`;
- total scientific forwards / row: `44`;
- Stage A scientific forward budget: `35200`.

The design phrases:

- `16 full-prefix probe forwards / row`;
- `28`;
- `22400`;
- `Before the 22,400-forward scientific execution`

are superseded only with respect to forward accounting.

---

## 3. Scientific design unchanged

This correction changes **none** of the following:

- scientific question;
- Mamba-370M model identity;
- block-35 intervention site;
- block-47 readout;
- frozen P3 selected plane;
- frozen P5 response-blind control plane;
- frozen P3/P5 basis vectors;
- strong-channel mask;
- `epsilon = 0.025`;
- offsets `t*-4, t*-3, t*-2, t*-1`;
- gold-aligned decisive readout `F_t`;
- central-difference derivative definition;
- `chi_P3,t`;
- `chi_P5,t`;
- `D_t = chi_P3,t - chi_P5,t`;
- equal-weight four-offset endpoint `Z_i`;
- response-free N=800 cohort identity;
- Refuted 400 / Supported 400 balance;
- future supported/unsupported outcome definition;
- two-sided Welch primary test;
- alpha `0.05`;
- primary p-value count `1`;
- minimum group size `30`;
- success/failure labels;
- descriptive-only diagnostics;
- prohibition on feature/offset/layer/epsilon selection;
- prohibition on rescue;
- requirement for bounded historical-row CPU-slow/CUDA-fast equivalence before scientific execution.

---

## 4. Prospectivity

The correction is purely arithmetic.

At correction time:

- Precursor-v2 scientific model forward count: `0`;
- Precursor-v2 generation response observed: `False`;
- Precursor-v2 susceptibility response observed: `False`;
- Precursor-v2 inference executed: `False`;
- Precursor-v2 p-value count: `0`;
- feature selection performed: `False`;
- offset selection performed: `False`;
- epsilon selection performed: `False`;
- cohort selection reopened: `False`.

The frozen N=800 cohort remains unchanged.

---

## 5. Correct next phase

After this correction is frozen, the next phase remains:

`PRECURSOR_V2 DCS RUNNER IMPLEMENTATION + BOUNDED CPU/CUDA EQUIVALENCE`

Scientific Stage A execution remains unauthorized until the runner is frozen and the required equivalence gate passes.

Correct planned Stage A scientific forward budget:

`35200`
