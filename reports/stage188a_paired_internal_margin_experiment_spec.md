# Stage188-A paired internal compatible-positive margin experiment specification

**Decision:** `STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_SPEC_READY`

**Authorized next stage:** `STAGE188B_PAIRED_INTERNAL_MARGIN_TRAINING`, only after the Stage188-A manifest is materialized and validated in Kaggle.

## Scope and purpose

Stage188-B is one paired, single-seed, internal-clean diagnostic. It validates the actual trainer path, measures the first-order mechanism direction, and screens for clean-performance regression. It cannot establish robustness, final scientific benefit, an external-evaluation claim, or a publication claim. A positive result authorizes only Stage189 three-seed replication.

No training, model import, Torch import, forward pass, checkpoint load, external evaluation, or source-data modification is part of Stage188-A.

## Authoritative inputs

| Input | Authoritative identity |
|---|---|
| Main dataset | `data/controlled_v5_v3_without_time_swap.jsonl` |
| Dataset SHA-256 | `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640` |
| Stage185 sidecar | `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/stage185a_controlled_train_integrity_sidecar.jsonl` |
| Sidecar semantic SHA-256 | `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc` |
| Stage186-A | `reports/stage186a_compatible_positive_margin_fixed_spec_20260715_150006` |
| Stage187-B | `reports/stage187b_default_off_implementation_runtime_validation_20260715_152204` |

Stage187-B closed with 14/14 checks passed and topology 3,600 / 605 / 121 / 5.

## Authoritative Stage174-D1 baseline resolution

The repository contains a Stage174-D1F closure but no self-contained Stage174-D1 run directory. The closure fixes these facts: architecture `v6b_minimal`; Mamba backbone; seed `174`; 20 epochs; main data `data/controlled_v5_v3_without_time_swap.jsonl`; no time-swap main training; internal clean-dev checkpoint selection; final CE from `output["logits"]`; and no external evaluation. Stage175-C and Stage177-D independently corroborate model `state-spaces/mamba-130m-hf`, CUDA, and disabled Stage174/175 baseline interventions. Stage177-D records its own Stage177 intervention and therefore is not itself the clean baseline.

The exact Stage174-D1 command, parsed arguments, resolved optimizer/batch settings, output policy, and commit are not present in this checkout. The manifest builder must therefore resolve them from the user-supplied `--stage174d1-dir`, preferably its `run_provenance.json`. If artifacts disagree or any required value is absent, it must emit a blocked report and no executable manifests. It must never choose a candidate heuristically or fill a missing value from current trainer defaults.

## Paired common configuration

Both runs must have identical resolved values for:

- Git commit and trainer source identity.
- Dataset path and SHA, split, dev ratio, and exact seed `174`.
- Architecture `v6b_minimal`, Mamba backbone, model name `state-spaces/mamba-130m-hf`, tokenizer, and CUDA device.
- Epochs `20`, optimizer, learning rates, weight decay, scheduler, batch/full-batch semantics, gradient accumulation, and deterministic flags.
- Checkpoint selection on clean-dev `final_macro_f1` and final CE from `output["logits"]`.
- Every native auxiliary weight, intervention flag, bridge flag, external-evaluation flag, and output/export setting.
- Stage174-C support-pairwise intervention off; Stage175 support-anchor intervention off; Stage177 frame-pairwise intervention off; no time-swap, external train, external calibration, external selection, or unrelated bridge/auxiliary intervention unless the authoritative Stage174-D1 provenance explicitly records it.

Run directory and descriptive run name may differ. No seed, weight, margin, or selection sweep is permitted.

## Exact allowed differences

| Field | Baseline | Intervention |
|---|---|---|
| `compatible_positive_margin_weight` | `0.0` | `0.05` |
| `compatible_positive_margin_logit` | `0.0` | `0.0` |
| `controlled_integrity_sidecar_path` | `null` | authoritative Stage185 sidecar |
| `expected_integrity_sidecar_semantic_sha256` | `null` | `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc` |
| run directory / descriptive run name | baseline-specific | intervention-specific |

The margin logit is listed in both arms to require explicit equality; it is not a differing value.

## Forbidden differences

Every other CLI, parsed-argument, or resolved-configuration difference is forbidden. In particular: commit, data, SHA, seed, architecture, backbone, model/tokenizer, device, epochs, optimizer, learning rates, weight decay, scheduler, batch semantics, split, dev ratio, selection metric, native auxiliary weights, intervention flags, bridge flags, deterministic flags, and export semantics may not differ.

Any forbidden difference, identity mismatch, row-ID mismatch, mask/logit length mismatch, external-data access, baseline sidecar access, or missing required evidence blocks Stage188-B.

## Manifest builder contract

`scripts/build_stage188a_paired_internal_margin_manifest.py` accepts required `--repo-root`, `--stage174d1-dir`, `--stage186a-dir`, `--stage187b-dir`, `--trainer-source`, and `--output-dir`, plus optional expected dataset/sidecar SHA arguments. Weight, margin, seed, epochs, and selection metric are intentionally not CLI-selectable.

It statically validates authoritative identities and decisions, requires an unambiguous Stage174-D1 provenance/argv/config, freezes the common configuration, produces explicit baseline and intervention argv arrays and previews, audits allowed and forbidden differences, and writes all twelve required Stage188-A artifacts. It never invokes a subprocess and never runs training.

The builder is successful only when `stage188a_stage188b_gate.csv` passes every row and both manifests are materialized. On failure it writes the report/audit artifacts with decision `STAGE188A_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED` and does not write runnable arm manifests.

## Stage188-B full-path validation

The intervention run must record and verify: sidecar enabled; exact dataset and semantic sidecar SHA; exact row-ID join; aligned train rows; 605 aligned eligible rows; bridge/external rows forced ineligible; frame-logit/mask length equality; observations accumulated across epochs; zero-eligible batch count; raw/weighted loss; active count/rate; mean eligible frame logit; score source `output["frame_logit"]`; eligible-row-mean normalization; and unchanged checkpoint selection.

The baseline must prove the margin is disabled and the sidecar was not accessed. Both runs must prove successful completion and absence of external evaluation, training, calibration, and selection data.

## Stage188-B analyzer contract

`scripts/analyze_stage188b_paired_internal_margin_training.py` accepts the seven required directories from the experiment contract and optional expected identity arguments. It loads reports, provenance, predictions, scalars, and Stage182/185 evidence only. It never trains, imports a model, or loads a checkpoint.

It validates exact pairing and allowed differences; compares selected clean-dev metrics, prediction counts, confusion matrices, and error families; emits row-level prediction transitions; reports native frame diagnostics without substituting final-classifier logits; evaluates the exact Stage182-B 13 compatible-FN rows, one incompatible-FP row, matched controls, and clean-model-failure cohort; summarizes intervention mechanism diagnostics; applies every precommitted gate; and emits a JSON/Markdown report plus CSV audits.

Stage182-B is a prior-evidence-selected diagnostic cohort, not independent evaluation. The analyzer must say so.

## Precommitted guardrails

All identity/runtime gates must pass. Clean guardrails are:

- Intervention macro-F1 >= baseline macro-F1 - 0.01.
- Intervention accuracy >= baseline accuracy - 0.01.
- Intervention SUPPORT recall >= baseline SUPPORT recall - 0.02.
- Polarity errors do not increase.
- False entitlement does not increase by more than 2.
- No new prediction-count collapse.

Mechanism-direction gates are:

- Eligible-train mean frame logit is higher than the comparable baseline-compatible reference.
- Compatible-positive active rate decreases across training or at the selected epoch versus its initial/available reference.
- Stage182-B compatible-FN median frame-logit delta is positive.
- At least 9 of 13 compatible-FN rows have positive frame-logit delta.
- The incompatible-FP count does not increase.

Training-eligible diagnostics are mechanism checks, not generalization evidence.

## Decision taxonomy

- `STAGE188B_PAIRED_INTERNAL_MARGIN_EXPERIMENT_BLOCKED`
- `STAGE188B_SINGLE_SEED_MARGIN_NEGATIVE_OR_REGRESSIVE`
- `STAGE188B_SINGLE_SEED_MARGIN_MIXED_NO_REPLICATION_YET`
- `STAGE188B_SINGLE_SEED_MARGIN_POSITIVE_THREE_SEED_REPLICATION_CANDIDATE`

Only the positive diagnostic decision authorizes `STAGE189_THREE_SEED_COMPATIBLE_POSITIVE_MARGIN_REPLICATION`. It does not authorize external evaluation.

## Unresolved static risks

1. This checkout lacks the authoritative Stage174-D1 run directory, exact raw argv, parsed arguments, resolved configuration, and commit. The Kaggle manifest build must supply and validate them.
2. The current prediction exporter visibly exports native `frame_prob` but not row-level native `frame_logit`. The analyzer requires direct native frame-logit evidence for the Stage182-B logit-delta gate and fails closed rather than substituting final-classifier logits. Stage188-B must confirm an allowed existing export contains that field; otherwise it is blocked.
3. Stage187-B did not perform a model forward, so full mask/logit alignment remains unvalidated until Stage188-B.
4. Stage187-B did not execute weight-zero full-training numerical equivalence.
