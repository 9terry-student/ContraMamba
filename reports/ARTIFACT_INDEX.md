# ContraMamba Artifact Index

This is a curated human index for important anchors only. It is not a full
repository inventory and does not supersede frozen authority artifacts.

## Current / Canonical

- `README.md`
  - Category: current roadmap.
  - Role: P4-AA 15-week research-credit roadmap and current status summary.
  - Status: current.
  - Required for future A0: context yes; execution authority no.

- `scripts/train_controlled_v6b_minimal.py`
  - Category: current trainer.
  - Role: P4-X/P4-Y-bound trainer implementation.
  - Status: current.
  - Required for future A0: yes, unless a later frozen authority supersedes it.
  - Authority: `reports/reason_router_p2_p3w6f2_p4x_trainer_rebind_authority_spec_candidate.md`.

- `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
  - Category: controlled dataset.
  - Role: current canonical controlled source dataset for the P4-L/P4-X line.
  - Status: current, tracked, present.
  - Required for future A0: yes.
  - Physical SHA256:
    `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`.
  - Semantic SHA256:
    `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`.

## Historical Authority

- `AGENTS.md`
  - Category: repository-wide agent contract.
  - Role: stable workflow and research-integrity rules.
  - Status: current repository-wide guidance.
  - Required for future A0: yes as process context.

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_artifact_contract_authority_spec.md`
  - Category: P4-L authority.
  - Role: current-lineage sidecar/provenance contract.
  - Status: current authority anchor.
  - Required for future A0: yes.

- `reports/reason_router_p2_p3w6f2_p4w_canonical_p4l_exact_byte_restore_execution_authority_spec.md`
  - Category: P4-W restore authority.
  - Role: exact-byte restore/provisioning constraints for absent P4-L bytes.
  - Status: historical authority relevant to future provisioning.
  - Required for future A0: likely yes if P4-L bytes are provisioned through
    the same exact-byte route.

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

## Future Required But Currently Absent

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
  - Category: canonical P4-L sidecar.
  - Role: future A0 dependency for the current trainer binding.
  - Status: current identity, absent locally.
  - Required for future A0: yes.
  - Physical SHA256:
    `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`.
  - Semantic SHA256:
    `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`.
  - Notes: exact external source path unresolved. Exact-byte provisioning only;
    no silent reconstruction.

- `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
  - Category: canonical P4-L provenance.
  - Role: future A0 dependency for the current trainer binding.
  - Status: current identity, absent locally.
  - Required for future A0: yes.
  - Physical SHA256:
    `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`.
  - Notes: exact external source path unresolved. Exact-byte provisioning only;
    no silent reconstruction.
