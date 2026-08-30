# ContraMamba Research State

## Current Head

`026216aedb3fa3290dfef65bb81f164580992918`

## Current Phase

- Pre-URP infrastructure/provenance closure.
- P4-L formal-start blocker is `CLOSED`.
- P3-W7-A0 formal authority is `NOT_CREATED` / `NOT_AUTHORIZED`.

## Active Research Topic

ContraMamba Reason-Preserving Authorization Router.

## Established

- Current controlled dataset identity is established for the P4-X/P4-L-bound
  trainer line.
- P4-L canonical bytes are present, tracked, and validated.
- P4-L artifact/provenance identity contract is established by frozen authority,
  successful exact-byte provisioning, canonical artifact freeze, and read-only
  validation.
- Bounded P4-X/P4-Y trainer-rebind code correctness is established.
- The research-credit reporting scaffold records Week 0 separately from the
  15 official research weeks.

## Not Established

- Formal P3-W7-A0 execution authority is not created.
- A0 execution has not occurred.
- No formal training, evaluation, or scientific execution has occurred.
- Trainer runtime success for formal A0 is not established.
- Scientific effectiveness is not established.
- A1/A2/A3 results are not established.

## Current Canonical Dependencies

- Controlled dataset:
  `reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`
  - Physical SHA256:
    `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`
  - Semantic SHA256:
    `3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`
- Current trainer:
  `scripts/train_controlled_v6b_minimal.py`
- Research operations:
  `docs/RESEARCH_OPERATIONS.md`
- Research-credit narrative scaffold:
  `docs/research_credit/README.md`
- P4-L sidecar identity:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
  - Status: current canonical, present, tracked, validated.
  - Physical SHA256:
    `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
  - Semantic SHA256:
    `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
  - Row count: `3600`
- P4-L provenance identity:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
  - Status: current canonical, present, tracked, validated.
  - Physical SHA256:
    `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

## P4-L Closure Evidence

- Provisioning retry1 run:
  `p3w6f2-p4l-exact-byte-provision-retry1-23233cc`
  - HEAD: `23233cce48262979d2c39444724668cc86fadcc7`
  - Command SHA256:
    `bd65af1ff45c8ec36958ca51012b2ab1351b3e47c2c02ebd178a5028bb64cf2e`
  - Exit: `0`
  - Run log SHA256:
    `b43f985116350b93a082eae945dff55198178721a8741a1600aed46a15d803c8`
  - Run meta SHA256:
    `be1c469f55701fcd7e59509334e39a3af89904963286ee3ae9ec9d8033beeadb`
  - Import ZIP SHA256:
    `2dff07af8bdc0d15497eda6bc692e40229da5058fe89ac9991441ccb41b1b7c5`
  - Import: `PASS`
- Canonical artifact freeze:
  `a93c291b79974f4aaa0b51f4578c807e8a5d6301`
- Read-only validation:
  - Authority freeze: `026216aedb3fa3290dfef65bb81f164580992918`
  - Run: `p3w6f2-p4l-provisioning-result-validate-026216a`
  - Command SHA256:
    `01661746e878162d002a951b3a924a0a34b0421b19007631bf77545cbc7866dd`
  - Success token:
    `P3W6F2P4L_CURRENT_PROVISIONING_RESULT_VALIDATION_PASS`
  - Exit: `0`
  - Status before/after: empty
  - Run log SHA256:
    `335d9fbfda17e08e9d9ca61b974288137a8accc8d9475c8b8baf5e93c5f33b96`
  - Run meta SHA256:
    `2850bcce398cce93f30112e8ffcd1441051fc864270f0eda12803a8c1f85a89c`
  - Collected files: `0` as expected for read-only validation
  - Import ZIP SHA256:
    `c81094a04580599fc967de72f055effbb7a29e05c15a3fcfcf9c3bee9f7fa079`
  - Import: `PASS`

This closure is infrastructure/provenance evidence only. It is not scientific
evidence and does not establish A0, training, evaluation, promotion, or
reason-router effectiveness.

## Known Non-Blocking Infrastructure Incident

- During fresh bootstrap work, `cm kaggle fresh` deleted its active current
  working directory and clone failed.
- Manual clean recovery succeeded before validation.
- The validation attempt was not consumed by the incident.
- This incident is recorded as known infrastructure behavior; it is not claimed
  fixed by this state sync.

## Historical Failed Attempt

- Earlier consumed failure:
  `p3w6f2-p4l-exact-byte-provision-67d9dcc`
  - Failure: `FREEZE_NOT_LOWERCASE_40_HEX`
  - Status: pre-mutation; collected/imported; never rerun.

## Current Blockers

- Formal A0 authority is intentionally not created and not authorized.
- P3-W7-A0 cannot begin until an independent P3-W7-A0 authority is created,
  independently verified, and frozen.

## Local Hygiene Notes

- Latest validation basis used HEAD
  `026216aedb3fa3290dfef65bb81f164580992918`.
- 75 untracked review patch files remain in place.
- `reports/stage180a_pass2_annotations_completed.csv` is untracked and
  byte-identical to the tracked
  `reports/stage180_manual_annotations/stage180a_pass2_annotations_completed_chatgpt.csv`
  copy.

## Next Formal Research Step

Create, independently verify, and freeze independent P3-W7-A0 execution
authority.

No speculative future scientific results are established by this document.
