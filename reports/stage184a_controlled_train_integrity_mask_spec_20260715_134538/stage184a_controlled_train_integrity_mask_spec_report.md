# Stage184-A controlled-train integrity mask specification report

## Decision

`STAGE184A_DETERMINISTIC_FAIL_CLOSED_INTEGRITY_SIDECAR_SPEC_READY`

Authorized next: `STAGE185_CONTROLLED_TRAIN_INTEGRITY_SIDECAR_BUILDER`. Stage185 is limited to deterministic sidecar generation and static audit. Loss implementation and training remain unauthorized.

## Stage183-A closure

Stage183-A required an integrity mask before the contingent compatible-positive absolute-margin hinge. Train frame labels are balanced at 1,440/1,440, so positive reweighting has no imbalance basis. The hinge remains distinct from Stage175's final SUPPORT anchor and Stage177's pair ordering, but its margin and nonzero weight remain unset.

## Authoritative dataset identity

- Path: `/kaggle/working/ContraMamba/data/controlled_v5_v3_without_time_swap.jsonl`
- SHA-256: `f5525866860c2c153c63296e28cac27321f4e140c56c37400844cb0baefbb640`
- Rows/pairs: 3600 / 300
- Families: 12
- Train/dev rows: 2880 / 720
- Train compatible/incompatible: 1440 / 1440
- Time-swap rows: 0

Topology is cross-checked against the Stage183-A closure. Any mismatch blocks the audit.

## Complete mask

Complete means every source row receives exactly one of `ELIGIBLE`, `INELIGIBLE`, or `UNRESOLVED`; it does not mean every row is clean. `UNRESOLVED` is fail-closed and never enters the positive-margin loss. Dev, frame-incompatible, time-swap, external, and Stage34/35 rows are excluded independently.

## Derivability and generator cleanliness boundary

The generator retains structured original/alternate fact arguments and exact intervention branches. Stage182-A supplies deterministic same-pair canonical, structured-axis, polarity-leak, and known morphology rules. This is enough to build a reproducible fail-closed sidecar without rewriting the JSONL. General grammar or missing provenance becomes `UNRESOLVED`; no model, LLM, annotation, text classifier, or family-name shortcut is allowed.

Exact generator equality proves provenance only. A buggy generator can exactly reproduce `did not` plus an inflected predicate or unintended non-polarity polarity changes, so grammar, contract, and polarity remain independent gates.

## Family and pair contracts

All families are enumerated from the actual JSONL, then joined to generator branch evidence. Each contract specifies changed/preserved axes, expected labels, canonical counterpart, evidence relation, and fail-closed reason codes. Unknown families remain unresolved.

Missing/duplicate canonical rows or invalid canonical linkage make the whole pair unresolved. A single deterministic grammar, polarity, schema, or intervention-contract failure makes the affected row ineligible. Duplicate row identity, SHA mismatch, or an impossible one-to-one join blocks the artifact.

## Sidecar and join contract

The sidecar is one-to-one on `row_id`, uses enum criterion statuses, records sorted stable reason codes, canonical/family/rule identity, source/generator/builder hashes, and the frozen pair split. Missing/extra/duplicate sidecar rows or dataset SHA mismatch block use. Generator SHA mismatch also blocks rebuilding or use under a different provenance contract.

## Coverage feasibility

- Decision coverage: `all rows classifiable as ELIGIBLE/INELIGIBLE/UNRESOLVED at Stage185`
- Positive eligible coverage: `unavailable before builder`
- Expected exact eligible count: `None`
- Scientific usability: `unknown until family/reason-code coverage is materialized`

Stage184-A does not estimate semantic cleanliness or fit a coverage threshold. Stage185 must compute exact counts by split, frame label, family, and reason code; family concentration or very low coverage remains a separate scientific risk.

## Safety

Static specification/feasibility audit only. No dataset/JSONL/generator modification, model or Torch import, checkpoint load, forward, loss implementation, training, smoke, annotation, LLM labeling, text classifier, learned parser/probe, threshold fitting, calibration, external evaluation, time-swap use, multi-seed run, or hyperparameter sweep is authorized.
