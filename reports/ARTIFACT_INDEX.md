# ContraMamba Artifact Index

This is a curated human index for important anchors only. It is not a full
repository inventory and does not supersede frozen authority artifacts.

## Current / Canonical

- `README.md`
  - Category: current roadmap.
  - Role: P4-AA 15-week research-credit roadmap and current status summary.
  - Status: current.
  - Required for future A0: context yes; execution authority no.

- `docs/RESEARCH_OPERATIONS.md`
  - Category: operations handbook.
  - Role: stable operating layer and evidence-level boundary.
  - Status: current process guidance.
  - Required for future A0: process context yes.

- `docs/research_credit/README.md`
  - Category: research-credit narrative scaffold.
  - Role: Week 0 through Week 15 reporting calendar, with reports retained as
    provenance/source-of-truth.
  - Status: current narrative scaffold.
  - Required for future A0: context no; execution authority no.

- `scripts/train_controlled_v6b_minimal.py`
  - Category: current trainer.
  - Role: P4-X/P4-Y-bound trainer implementation.
  - Status: current.
  - Required for future A0: yes, unless a later frozen authority supersedes it.
  - Authority:
    `reports/reason_router_p2_p3w6f2_p4x_trainer_rebind_authority_spec_candidate.md`.

- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
  - Category: controlled dataset.
  - Role: current canonical controlled source dataset for the P4-L/P4-X line.
  - Status: current, tracked, present.
  - Required for future A0: yes.
  - Physical SHA256:
    `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.
  - Semantic SHA256:
    `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
  - Category: canonical P4-L sidecar.
  - Role: current canonical P4-L sidecar for the P4-L/P4-X line.
  - Status: current, tracked, present, provisioned, validated.
  - Required for future A0: yes.
  - Physical SHA256:
    `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`.
  - Semantic SHA256:
    `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.
  - Row count: `3600`.
  - Canonical artifact freeze:
    `a93c291b79974f4aaa0b51f4578c807e8a5d6301`.
  - Validation authority freeze:
    `026216aedb3fa3290dfef65bb81f164580992918`.

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
  - Category: canonical P4-L provenance.
  - Role: current canonical P4-L provenance for the P4-L/P4-X line.
  - Status: current, tracked, present, provisioned, validated.
  - Required for future A0: yes.
  - Physical SHA256:
    `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`.
  - Canonical artifact freeze:
    `a93c291b79974f4aaa0b51f4578c807e8a5d6301`.
  - Validation authority freeze:
    `026216aedb3fa3290dfef65bb81f164580992918`.

## P4-L Provisioning / Validation Authorities And Runs

- `reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_corrected_execution_authority_spec_candidate.md`
  - Category: P4-L provisioning authority.
  - Role: final corrected exact-byte provisioning authority.
  - Status: historical authority consumed by retry1.
  - Required for future A0: provenance context yes; execution authority no.

- `reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_freeze_binding_correction_execution_authority_spec_candidate.md`
  - Category: P4-L provisioning authority correction.
  - Role: freeze-binding correction authority before retry1.
  - Status: historical authority consumed by retry1.
  - Required for future A0: provenance context yes; execution authority no.

- `reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_environment_binding_retry1_execution_authority_spec_candidate.md`
  - Category: P4-L provisioning retry1 authority.
  - Role: command-local freeze binding and exact-byte retry1 provisioning
    authority.
  - Status: consumed by successful provisioning run
    `p3w6f2-p4l-exact-byte-provision-retry1-23233cc`.
  - Provisioning authority freeze:
    `23233cce48262979d2c39444724668cc86fadcc7`.
  - Provisioning command SHA256:
    `bd65af1ff45c8ec36958ca51012b2ab1351b3e47c2c02ebd178a5028bb64cf2e`.
  - Provisioning run log SHA256:
    `b43f985116350b93a082eae945dff55198178721a8741a1600aed46a15d803c8`.
  - Provisioning run meta SHA256:
    `be1c469f55701fcd7e59509334e39a3af89904963286ee3ae9ec9d8033beeadb`.
  - Provisioning import ZIP SHA256:
    `2dff07af8bdc0d15497eda6bc692e40229da5058fe89ac9991441ccb41b1b7c5`.
  - Import: `PASS`.

- `reports/reason_router_p2_p3w6f2_p4l_current_provisioning_result_validation_execution_authority_spec_candidate.md`
  - Category: P4-L read-only validation authority.
  - Role: exact source/canonical/runtime-HEAD byte validation authority.
  - Status: consumed by successful read-only validation run
    `p3w6f2-p4l-provisioning-result-validate-026216a`.
  - Validation authority freeze:
    `026216aedb3fa3290dfef65bb81f164580992918`.
  - Validation command SHA256:
    `01661746e878162d002a951b3a924a0a34b0421b19007631bf77545cbc7866dd`.
  - Success token:
    `P3W6F2P4L_CURRENT_PROVISIONING_RESULT_VALIDATION_PASS`.
  - Validation run log SHA256:
    `335d9fbfda17e08e9d9ca61b974288137a8accc8d9475c8b8baf5e93c5f33b96`.
  - Validation run meta SHA256:
    `2850bcce398cce93f30112e8ffcd1441051fc864270f0eda12803a8c1f85a89c`.
  - Validation import ZIP SHA256:
    `c81094a04580599fc967de72f055effbb7a29e05c15a3fcfcf9c3bee9f7fa079`.
  - Collected files: `0`.
  - Import: `PASS`.

- `p3w6f2-p4l-exact-byte-provision-67d9dcc`
  - Category: consumed failed provisioning attempt.
  - Role: first environment-binding failure provenance.
  - Status: historical failure; pre-mutation; collected/imported; never rerun.
  - Failure: `FREEZE_NOT_LOWERCASE_40_HEX`.
  - Imported failure provenance retained.

## Historical Authority

- `AGENTS.md`
  - Category: repository-wide agent contract.
  - Role: stable workflow and research-integrity rules.
  - Status: current repository-wide guidance.
  - Required for future A0: yes as process context.

- `reports/PRE_URP_HANDOFF.md`
  - Category: historical handoff.
  - Role: pre-provisioning handoff and formal-start boundary.
  - Status: historical; P4-L absence blocker language is superseded by the
    validated P4-L provisioning/validation evidence above.
  - Required for future A0: historical context yes; current P4-L blocker no.
  - Notes: do not rewrite this historical evidence.

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
  - Category: P4-L authority.
  - Role: current-lineage sidecar/provenance contract.
  - Status: current authority anchor.
  - Required for future A0: yes.

- `reports/reason_router_p2_p3w6f2_p4w_canonical_p4l_exact_byte_restore_execution_authority_spec.md`
  - Category: P4-W restore authority.
  - Role: historical exact-byte restore/provisioning constraints.
  - Status: historical only; must not be resurrected unless a later frozen
    authority explicitly reopens it.
  - Required for future A0: no current execution role.

## Implementation / Validation

- `reports/reason_router_p2_p3w6f2_p4x_trainer_rebind_authority_spec_candidate.md`
  - Category: implementation authority.
  - Role: P4-X trainer rebind scope and fail-closed contract.
  - Status: current trainer authority anchor.
  - Required for future A0: yes.

- `reports/reason_router_p2_p3w6f2_p4y_trainer_rebind_validation_8f6defacc1995f263c97000fe43f6034b1ce9324/p3w6f2_p4y_trainer_rebind_validation_result_candidate.json`
  - Category: validation evidence.
  - Role: bounded P4-X/P4-Y trainer-rebind validation result.
  - Status: current validation anchor.
  - Required for future A0: process/evidence context yes.

## Raw / Validated Evidence

- `reports/reason_router_p2_p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_execution_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4v_canonical_sidecar_path_provenance_schema_correction_result.json`
  - Category: validated evidence.
  - Role: P4-V closure evidence for canonical P4-L identities.
  - Status: current provenance/evidence anchor.
  - Required for future A0: yes.
  - Establishes sidecar physical SHA256
    `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`,
    provenance physical SHA256
    `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`,
    and sidecar semantic SHA256
    `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.

## Infrastructure Incidents

- `cm kaggle fresh` CWD deletion incident
  - Category: known non-blocking infrastructure incident.
  - Role: fresh-bootstrap failure/recovery chronology.
  - Status: manual clean recovery succeeded before validation; validation
    attempt was not consumed.
  - Notes: this index records the incident but does not claim it is fixed.

## Superseded But Potentially Confusing

- `reports/stage185a_controlled_train_integrity_sidecar_20260715_141914/`
  - Category: historical sidecar-era material.
  - Role: Stage185 historical integrity sidecar material.
  - Status: historical/superseded for the current P4-X trainer binding.
  - Required for future A0: no, unless a future authority explicitly reopens
    Stage185 lineage.

- `data/controlled_v5_v3_without_time_swap.jsonl`
  - Category: historical controlled dataset path.
  - Role: earlier lineage dataset path.
  - Status: historical for current P4-L/P4-X binding.
  - Required for future A0: no for the current planned A0 line.

## Untracked Review Material

- Root review patch group, 75 files matching `reason_router_*.patch`
  - Category: untracked review material.
  - Role: review-cache/archive material, not execution dependencies.
  - Status: untracked, left untouched.
  - Required for future A0: no.
  - Notes: most are review-cache/archive candidates. Root P2/revision patches
    and P3-W6-F1 fixture/contract groups require more caution before cleanup.
    No cleanup has been performed.

## Ambiguous / Special Cases

- `reports/stage180a_pass2_annotations_completed.csv`
  - Category: untracked duplicate/special case.
  - Role: root-path copy historically referenced by frozen P4 documents.
  - Status: untracked, present at latest audit, left untouched.
  - Required for future A0: no direct dependency identified.
  - SHA256:
    `1f431e557ec3c63cf1d027a89485fbb086943966c9c0d683562ea0440e6f7f27`.

- `reports/stage180_manual_annotations/stage180a_pass2_annotations_completed_chatgpt.csv`
  - Category: tracked Stage180A pass2 annotation artifact.
  - Role: tracked byte-identical copy of the root-path Stage180A pass2 CSV.
  - Status: historical, tracked, present.
  - Required for future A0: no direct dependency identified.
  - SHA256:
    `1f431e557ec3c63cf1d027a89485fbb086943966c9c0d683562ea0440e6f7f27`.
  - Notes: the untracked root-path CSV contains no unique annotation content.

## Formal Research Boundary

- P4-L closure is infrastructure/provenance evidence, not scientific evidence.
- Formal P3-W7-A0 authority remains `NOT_CREATED` / `NOT_AUTHORIZED`.
- No formal training, evaluation, A0, A1, A2, A3, promotion, or scientific
  execution has occurred.
- The next formal research step is independent P3-W7-A0 authority creation.
