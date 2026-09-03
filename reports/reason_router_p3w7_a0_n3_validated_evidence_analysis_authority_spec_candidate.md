# P3-W7 A0 N=3 Validated-Evidence Analysis Authority Spec Candidate

Status: READY candidate.

This candidate authorizes a future static P3-W7 A0 N=3 validated-evidence analysis authority and nothing else. It does not itself perform the N=3 analysis.

Authority/version:

`P3W7_A0_N3_VALIDATED_EVIDENCE_ANALYSIS_AUTHORITY_V1_CANDIDATE`

## Authority Basis

Current controller instruction is the highest authority for this candidate.

Formal P3-W7 A0 execution authority:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Previous validated-evidence analysis authority:

`759124743a9441a4c1811912770c9389fe7432f6`

Previous N=2 validated-evidence analysis:

`34a5df306a00a28492b08f666bf3f7ac06c26944`

Seed180 corrected runtime recovery authority:

`c34a80e676b75f18889ec48cabc623d7a59d490c`

Seed180 independently verified normal-reference small-file freeze:

`b32d73dfa49b6b9dfabf3093802904323cf679cd`

Seed181 `REPLACEMENT_R1` validated result:

`fb4f0e2c2a8382a642f1272b66f29552adaecb0e`

Seed182 validated result:

`82739bdfc8eee184de10ed8f55434f203a6d59a5`

Repository `AGENTS.md` applies.

## Candidate State

This candidate is authored in static authority-writing phase only.

Expected materialization state:

- HEAD: `34a5df306a00a28492b08f666bf3f7ac06c26944`
- branch: `p3w7-a0-n3-validated-evidence-analysis-authority`
- new file: `reports/reason_router_p3w7_a0_n3_validated_evidence_analysis_authority_spec_candidate.md`

No tracked modification, staging, commit, push, merge, cherry-pick, result-artifact materialization, recovery execution, training, evaluation, inference, checkpoint load, or Kaggle execution is authorized by this candidate.

## Predecessor Preservation

The previous N=2 report at commit `34a5df306a00a28492b08f666bf3f7ac06c26944` remains immutable historical analysis.

It is preserved exactly as:

- N = 2;
- membership = `seed181 REPLACEMENT_R1` plus `seed182`;
- seed180 excluded from that historical report.

This candidate must not be used to rewrite, reinterpret, or retroactively convert that N=2 report into N=3. This candidate is a successor authority for a new future N=3 analysis.

## Exact Primary Membership

The future N=3 A0 validated-evidence analysis primary membership is exactly:

| Member label | Result/evidence commit | Namespace | Admission |
|---|---|---|---|
| `seed180` | `b32d73dfa49b6b9dfabf3093802904323cf679cd` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0` | `PRIMARY_ADMISSIBLE_AS_NORMAL_RECOVERED_A0_REFERENCE_FOR_THIS_AUTHORIZED_N3_ANALYSIS` |
| `seed181 REPLACEMENT_R1` | `fb4f0e2c2a8382a642f1272b66f29552adaecb0e` | `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0` | `PRIMARY_ADMISSIBLE_AS_SEED181_REPLACEMENT_R1` |
| `seed182` | `82739bdfc8eee184de10ed8f55434f203a6d59a5` | `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0` | `PRIMARY_ADMISSIBLE` |

Primary N is exactly `3`.

No alternate seed, silent N reduction, or substitution of the consumed original seed181 attempt is authorized.

## Seed180 Admission

The previous analysis authority classified seed180 as:

`CAVEATED_ADMISSIBLE_ONLY_IF_EXPLICITLY_LABELED`

because seed180 then lacked repository result-import/freeze artifacts and a repository recovery audit-output artifact.

Those missing conditions are now resolved by the independently verified normal-reference small-file freeze:

`b32d73dfa49b6b9dfabf3093802904323cf679cd`

For the authorized purpose of this future N=3 A0 aggregate only, this candidate admits seed180 into the primary membership as a normal recovered A0 reference.

Required seed180 source artifacts at `b32d73dfa49b6b9dfabf3093802904323cf679cd`:

| Artifact path | Size | SHA256 |
|---|---:|---|
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0/A0_REFERENCE_AUDIT.json` | 3167 | `37db2a99f55346f0b91bd8f5a1e8b3b9134922b4c51e87babab29d6c73e52d7b` |

The seed180 audit binds:

| Audit field | Required value |
|---|---|
| `status` | `PASS` |
| `recovery_reference_status` | `RECOVERY_REFERENCE_AUDIT_PASS` |
| `standard_cm_wrapper_provenance` | `INCOMPLETE` |
| `provenance_disposition` | `RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE` |
| `source_execution_commit` | `2737c3c6116ae3766b469801f990e2c45ba9a55e` |
| `dataset_sha256_expected` | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |
| `sidecar_semantic_sha256_expected` | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |

Historical standard-CM wrapper provenance remains `INCOMPLETE`. This candidate does not relabel, complete, repair, or normalize historical wrapper provenance.

Seed180 selected checkpoint remains a non-Git artifact. No checkpoint loading is allowed. The checkpoint identity may be cited only from audit/provenance binding:

- SHA256: `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da`
- size: `518269815`

## Seed181 Membership

The only admissible seed181 member is:

`seed181 REPLACEMENT_R1`

Result commit:

`fb4f0e2c2a8382a642f1272b66f29552adaecb0e`

Namespace:

`reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0`

This is a separately authorized replacement execution. It is not a recovery of the consumed original seed181 attempt.

The original seed181 attempt remains `INADMISSIBLE` and must not be included, relabeled as recovered, substituted for `REPLACEMENT_R1`, or used as metric evidence.

Seed181 `REPLACEMENT_R1` committed artifacts:

| Artifact path | Size | SHA256 |
|---|---:|---|
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report.json` | 305374 | `6eddb2f101bd513b91befe1b6edefcd078cc61e5626db6d13876d9b85b198ff3` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/clean_dev_predictions.json` | 4840808 | `893534cd27d8df7bf5eb6a7fa888b17b4ee803bb1d2a8233d40c15f8cf62ae12` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report_predictions.jsonl` | 3936706 | `b186af00684279095a2257ef46f826be281a92bb2da9b0b9ee8f157f5bdbc13c` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/run_provenance.json` | 68927 | `11885151854401107b82b05c49da009bf47d9f465d348bc13965d60facad4e10` |

Seed181 `REPLACEMENT_R1` selected checkpoint remains a non-Git artifact. The checkpoint identity may be cited only from validated provenance:

- SHA256: `3dbd7c32cc2d60b2de13da3a72cff05eaa080520f7cac076225c5a55870721ca`
- size: `518269943`

## Seed182 Membership

Seed182 remains the original validated result with existing `PRIMARY_ADMISSIBLE` status.

Result commit:

`82739bdfc8eee184de10ed8f55434f203a6d59a5`

Namespace:

`reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0`

Seed182 committed artifacts:

| Artifact path | Size | SHA256 |
|---|---:|---|
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report.json` | 305735 | `319e13bbda07363a334d0b6615b2c4074dfcf5d30d0c43e1f0735f269c2b5e3e` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/clean_dev_predictions.json` | 4842166 | `80205044dceed9b2131cd3caf06524f7869b1651577504dca1503605d0471036` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report_predictions.jsonl` | 3938064 | `95aa57b9f14ff7119b19f0ec8e412bf5b4494ae325e73d4c0ef0df5b24e050e5` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/run_provenance.json` | 68388 | `8357efe9ff609b8a99f580aa47ecbe4d018b5a0d24668755369e6e65a6cab421` |

Seed182 selected checkpoint remains a non-Git artifact. The checkpoint identity may be cited only from validated provenance:

- SHA256: `212873153bc6cecf107e79a4ea86385033c7944a9af222d4984192b232803946`
- size: `518269815`

## Shared A0 Scientific Envelope

All three primary members must be independently verified against the formal A0 envelope before any future analysis output is accepted.

The required common envelope is:

| Property | Required value |
|---|---|
| architecture | `v6b_minimal` |
| backbone | `mamba` |
| model | `state-spaces/mamba-130m-hf` |
| encoder | frozen |
| split seed | `174` |
| training seeds | `180`, `181`, `182` respectively |
| arm | `A0` |
| router mode | `explicit_product` |
| gradient ownership | `joint` |
| effective reason loss | `0.0` |
| epochs | `20` |
| max length | `128` |
| dev ratio | `0.2` |
| learning rate | `0.001` |
| checkpoint selection | `final_macro_f1` |
| class weighting | `none` |
| controlled sidecar semantic SHA256 | `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08` |
| dataset SHA256 | `eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3` |

Ranking/objective auxiliaries must remain neutralized according to the formal A0 contract:

- weighted label loss: `false`;
- intervention loss: `false`;
- ranking weight absent or effectively `0.0`;
- compatible positive margin disabled and weight `0.0`;
- Stage174C clean pairwise mode `off` and weight `0.0`;
- Stage175B support anchor mode `off` and weight `0.0`;
- Stage177C frame pairwise mode `off` and weight `0.0`;
- auxiliary/bridge/external training rows not used.

No A0 reference prediction was consumed during any of these A0 executions.

If any material envelope incompatibility exists, the future N=3 analysis must return `BLOCKED`.

## Commit-Addressed Evidence Model

The future N=3 analysis may read evidence directly from the three immutable result commits. It is not necessary or authorized to merge, cherry-pick, import, or materialize the seed180, seed181, or seed182 result commits into a single tree.

Every analysis input must be addressed by all available immutable coordinates:

- immutable commit SHA;
- exact repository path;
- exact artifact SHA256 and byte size where frozen;
- checkpoint identity only from validated provenance/audit SHA256 and size.

No mutable working-copy result file may override commit-addressed evidence.

## Future Static Analysis Scope

This candidate authorizes only future static parsing and arithmetic over already validated stored artifacts, after this candidate is independently verified, frozen, and separately activated by an explicit analysis instruction.

The future N=3 analysis should reproduce the previous N=2 report structure where applicable, now for N=3:

- exact membership and envelope verification;
- observed per-seed selected metrics;
- best/selected epoch;
- final macro F1;
- final accuracy;
- frame accuracy;
- predicate accuracy;
- sufficiency accuracy;
- polarity accuracy entitled;
- per-label precision, recall, and F1;
- selected checkpoint identity from provenance/audit only;
- prediction distributions;
- pairwise/intervention diagnostics when all three artifacts support the same metric;
- descriptive aggregate: mean, sample SD using denominator `N - 1`, min, max.

For any metric unavailable or structurally non-comparable in one member, the future analysis must not silently reduce N. It must report that metric as non-comparable or unavailable for the N=3 aggregate and explain why.

No silent N=2 fallback is authorized.

## Interpretation Boundary

The future N=3 report is descriptive A0 baseline characterization only.

It may establish:

- the complete planned A0 baseline replicate set N=3;
- descriptive central tendency and dispersion for the A0 baseline;
- A0 diagnostic behavior.

It must not establish:

- P2 mechanism superiority;
- A1/A2/A3 effects;
- factorial effects;
- causal contribution of router, supervision, or gradient ownership;
- statistical significance unless separately authorized;
- promotion or release of A1/A2/A3.

## Factorial Boundary

This authority does not authorize factorial analysis or A1/A2/A3 execution. Its purpose is to close the A0 N=3 baseline evidence prerequisite.

After a future N=3 analysis is independently verified and frozen, the controller may separately reassess the previously blocked factorial authority.

No factorial conclusion may be drawn under this candidate.

## Non-Authorization

This candidate does not authorize:

- training;
- evaluation;
- inference;
- A1/A2/A3 execution;
- model loading;
- checkpoint loading or deserialization;
- Kaggle execution;
- result artifact materialization or recovery;
- result artifact modification;
- dataset or sidecar mutation;
- merge or cherry-pick of result commits;
- staging;
- commit;
- push.

## Stop Conditions

The future N=3 analysis must return `BLOCKED` if:

- `b32d73dfa49b6b9dfabf3093802904323cf679cd` is not independently verifiable;
- seed180 audit is not `PASS`;
- seed180 historical wrapper caveat is missing or weakened;
- seed181 `REPLACEMENT_R1` is mislabeled;
- original seed181 is included;
- seed182 result identity differs from `82739bdfc8eee184de10ed8f55434f203a6d59a5`;
- the A0 envelope differs materially across members;
- the analysis candidate allows silent N reduction;
- the analysis requires result-commit merging or cherry-picking;
- the analysis crosses into factorial or A1/A2/A3 interpretation;
- more than one file would be changed during authority materialization.

## Verification Notes For This Candidate

Candidate-authoring verification was read-only until this file was created.

Verified before authoring:

- dedicated worktree HEAD was `34a5df306a00a28492b08f666bf3f7ac06c26944`;
- dedicated worktree branch was `p3w7-a0-n3-validated-evidence-analysis-authority`;
- dedicated worktree status was clean;
- all named commits resolved to Git commit objects;
- seed180 required source paths existed at `b32d73dfa49b6b9dfabf3093802904323cf679cd`;
- seed181 `REPLACEMENT_R1` source paths existed at `fb4f0e2c2a8382a642f1272b66f29552adaecb0e`;
- seed182 source paths existed at `82739bdfc8eee184de10ed8f55434f203a6d59a5`;
- requested artifact byte sizes and SHA256 values matched for seed180, seed181 `REPLACEMENT_R1`, and seed182;
- seed180 `A0_REFERENCE_AUDIT.json` status was `PASS`;
- seed180 `A0_REFERENCE_AUDIT.json` preserved `standard_cm_wrapper_provenance = INCOMPLETE`;
- seed180 `A0_REFERENCE_AUDIT.json` preserved `provenance_disposition = RECOVERY_BRIDGE_WITH_HISTORICAL_STANDARD_CM_WRAPPER_PROVENANCE_INCOMPLETE`;
- parsed artifact/provenance fields confirmed common A0 envelope compatibility for architecture, backbone, model, frozen encoder, split seed, training seed, A0 arm, router mode, gradient ownership, reason loss, epochs, max length, dev ratio, learning rate, selection metric, class weighting, neutralized objectives, dataset identity, sidecar identity, and absence of A0 reference prediction consumption.

Final file hash, size, newline counts, `git diff --check`, and final git status are intentionally reported outside this file to avoid self-referential candidate content.
