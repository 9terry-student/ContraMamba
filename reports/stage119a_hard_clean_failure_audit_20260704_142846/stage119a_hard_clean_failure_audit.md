# Stage119-A Hard-Clean Failure Audit

Decision: `STAGE119A_HARD_CLEAN_FAILURE_AUDIT_LOCKED`

Status: repository provenance audit created after the requested Stage119-A add patterns matched no files.

## Audit Trigger

The requested commit target was:

| pattern | result before audit |
|---|---|
| `reports/stage119a_hard_clean_failure_audit_*/stage119a_hard_clean_failure_audit.json` | no match |
| `reports/stage119a_hard_clean_failure_audit_*/stage119a_hard_clean_failure_audit.md` | no match |
| `reports/stage119a_hard_clean_failure_audit_*/*.csv` | no match |

No commit or push was created before this audit because nothing was staged and the working tree was clean.

## Local Evidence

| item | finding |
|---|---|
| branch state | `main` tracking `origin/main` |
| latest commit | `4136120 Add Stage118 generic diagnostic eval path` |
| Stage117-B scaffold | `scripts/build_stage117_hard_clean_diagnostic.py` exists |
| Stage118 evaluator path | `scripts/train_controlled_v6b_minimal.py` contains Stage118 generic diagnostic arguments and evaluation helpers |
| Stage117 clean source | `data/controlled_v5_v3_without_time_swap.jsonl` exists |
| Stage119-A artifacts before audit | absent |
| local vNext checkpoint or run export | not found in `checkpoints`, `outputs`, or `models` |

## Interpretation

Stage117-A locked the external collapse interpretation and recommended a controlled hard-clean diagnostic. Stage117-B added the hard-clean diagnostic builder. Stage118 added a generic diagnostic evaluation path.

The current checkout does not contain Stage119-A hard-clean failure audit outputs, nor does it contain local vNext checkpoint or Stage118 prediction-summary artifacts needed to make metric-backed claims about hard-clean failure.

## What This Audit Does And Does Not Claim

This audit does claim:

- the requested Stage119-A report files were absent before this audit;
- the repository contains the Stage117-B builder and Stage118 evaluation hook;
- local metric verification is blocked by missing vNext evaluation artifacts.

This audit does not claim:

- that hard-clean reproduces the Stage112 external NOT_ENTITLED collapse;
- any Stage118 hard-clean accuracy, macro-F1, recall, or prediction-count metric;
- any external generalization result.

## Recommendation

Run Stage117-B hard-clean generation and Stage118 diagnostic evaluation in the environment that has the vNext checkpoint/run artifacts. Then replace or extend this provenance audit with metric-backed Stage119-A results.

Do not tune thresholds on VitaminC, train on external data, or claim external generalization from this audit.
