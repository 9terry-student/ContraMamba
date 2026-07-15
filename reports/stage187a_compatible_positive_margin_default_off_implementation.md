# Stage187-A compatible-positive margin default-off implementation

## 1. Decision

`STAGE187A_DEFAULT_OFF_COMPATIBLE_POSITIVE_MARGIN_IMPLEMENTED_PENDING_RUNTIME_VALIDATION`

Authorized next: `STAGE187B_DEFAULT_OFF_IMPLEMENTATION_RUNTIME_VALIDATION`.

This is an implementation record based on static inspection only. No trainer invocation, compile, test, forward pass, or training was performed.

## 2. Scope

The change is trainer-local. It adds an opt-in compatible-positive absolute-margin hinge without modifying the model, native frame head, delegated auxiliary losses, controlled dataset, Stage185 sidecar, or checkpoint-selection semantics.

## 3. CLI contract

- `--compatible-positive-margin-weight`: default `0.0`; exact allowed values `0.0` and `0.05`.
- `--compatible-positive-margin-logit`: default and only allowed value `0.0`.
- `--controlled-integrity-sidecar-path`: default `None`; required only at weight `0.05`.
- `--expected-integrity-sidecar-semantic-sha256`: default `None`; required only at weight `0.05` and must equal the authoritative Stage185 semantic SHA.

Smoke, train truncation, and loss-sweep paths are rejected when the intervention is enabled.

## 4. Exact default-off behavior

At weight `0.0`, activation validation returns before requiring or opening a sidecar. No dataset SHA, sidecar SHA, row join, eligibility mask, hinge tensor, or objective term is constructed for this feature. The existing total objective and checkpoint selector are unchanged.

## 5. Dataset identity gate

Enabled mode requires the exact resolved authoritative path `data/controlled_v5_v3_without_time_swap.jsonl`, SHA-256 `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`, exactly 3600 unique source IDs, and the frozen Stage185 train/dev split.

## 6. Sidecar semantic identity

Enabled mode requires semantic SHA-256 `5bc03caa2a29f9b9176ab4eb0201db57ebad516352797546db1a18e6ec3373fc`. The calculation matches the Stage185 builder: preserve source row order, remove only `created_at`, sort keys, encode compact UTF-8 JSON, and hash the canonical list.

## 7. Row join contract

The sidecar must have exactly 3600 nonempty JSON-object rows, unique nonempty `row_id` values, the exact source-ID set, and exact source order. Missing, extra, duplicated, reordered, or ambiguous rows fail closed; eligibility is never inferred from row position.

## 8. Eligibility defense in depth

`eligible_for_positive_margin` must be a strict JSON boolean. Every `true` row must also have `split=train`, `frame_compatible_label=1`, `integrity_status=ELIGIBLE`, `time_swap_status=PASS`, and `dataset_source_status=PASS`.

The frozen topology is checked exactly: 605 rows, 121 pairs, five families, five eligible rows per pair, and 121 eligible rows per family.

## 9. Split and batch alignment

The trainer-reconstructed train/dev ID sets must exactly match the Stage185 sidecar split. A boolean mask is constructed in final `train_records` order. Clean-main rows join by ID; bridge or external rows are explicitly false. The aligned mask must contain exactly 605 eligible rows.

## 10. Score and loss

The loss uses `output["frame_logit"].reshape(-1)` without detaching it. For eligible mask `E`, it computes `mean(relu(0.0 - z[E]))`, normalized only by eligible row count. It does not target `output["logits"]` or a diagnostic head.

## 11. Zero-eligible behavior

An empty eligible mask returns `frame_logit.reshape(-1).sum() * 0.0`, retaining a graph-connected zero. No division by zero and no detached scalar substitution is used.

## 12. Objective wiring

Only enabled mode appends `0.05 * compatible_positive_margin_loss` to the existing trainer total objective. Native model-forward frame BCE and all pre-existing auxiliary objectives remain intact. No model-side loss is replaced or recomputed.

## 13. Boundary semantics

The implementation uses native `torch.nn.functional.relu`. The active region is `frame_logit < 0`; the inactive region is `frame_logit > 0`; behavior at zero is native PyTorch autograd behavior. No custom subgradient is introduced.

## 14. Logging and aggregation

Per epoch, the report records raw and weighted loss, eligible count, active count/rate, eligible-logit sum and mean, and zero-eligible status. Run aggregation uses summed hinge/logit numerators and summed eligible denominators. The 605-row cohort count is reported separately from epoch-level eligible observation count.

## 15. Configuration and provenance

Resolved runtime configuration and provenance record activation state, fixed weight/margin, score source, eligible-row normalization, dataset and semantic sidecar identities, row-join audit, frozen topology, and runtime validation status. The final run report contains the same loss and cohort diagnostics.

## 16. Checkpoint-selection invariance

Final CE remains sourced from `output["logits"]`. The new loss is not a selection metric or tie-break. Existing clean-dev selection and final checkpoint logic are unchanged.

## 17. Safety constraints

The implementation performs no annotation, LLM labeling, text classification, learned parsing/probing, threshold fitting, calibration, external evaluation, time-swap use, multi-seed search, hyperparameter sweep, model import change, checkpoint load, or dataset/sidecar rewrite.

## 18. Remaining validation gate

Runtime behavior remains unvalidated by design. Stage187-B must verify exact default-off equivalence, enabled identity/SHA gates, the 605-row mask, loss numerics including zero eligible behavior, gradient flow to the native frame channel, aggregate logging, and checkpoint-selection invariance before any training authorization. Kaggle runtime validation is still required; it was not performed in Stage187-A.