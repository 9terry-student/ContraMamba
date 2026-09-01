# P3-W7-A0 Validated Evidence Analysis Authority Specification Candidate

Authority/version:

`P3W7_A0_VALIDATED_EVIDENCE_ANALYSIS_AUTHORITY_V1_CANDIDATE`

## Status

Candidate only.

This candidate defines whether and how already executed, collected, imported,
or otherwise immutably anchored P3-W7-A0 evidence may later be analyzed. It
does not authorize analysis by existence alone.

This candidate authorizes no trainer execution, model loading, checkpoint
loading, prediction generation, dataset evaluation, A1/A2/A3 execution,
A1/A2/A3 interpretation, artifact mutation, dataset regeneration, commit, or
push.

Future analysis requires all of:

1. independent verification of this exact candidate;
2. immutable Git freeze commit containing this exact candidate;
3. explicit subsequent analysis authorization.

## Authority Basis

Current controller instruction is the highest authority for this candidate.

Frozen formal P3-W7-A0 execution authority:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Path:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Frozen seed181 runtime-loss recovery authority:

`74defa2c679ca2244d69b6ee950dd4a6a7a643b4`

Path:

`reports/reason_router_p3w7_a0_seed181_runtime_loss_recovery_execution_authority_spec_candidate.md`

Frozen seed181 replacement execution authority:

`592844f0806e96a37d01c4445cd96b990cf0dae4`

Path:

`reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_execution_authority_spec_candidate.md`

Validated/imported seed181 replacement result commit:

`fb4f0e2c2a8382a642f1272b66f29552adaecb0e`

Validated/imported seed182 result commit:

`82739bdfc8eee184de10ed8f55434f203a6d59a5`

Resolved seed180 recovery chain:

- `233ed0be080e1d30dd47de2e66136475ec2ede76` - seed180 A0 provenance recovery authority.
- `cdd71ea4f556392eab594ebb5df8258355610e01` - seed180 recovery tooling implementation authority.
- `1b6516d16596d1169ff2fa4fd8d8c8f8adb80450` - external cm drift reconciliation authority.
- `15387eb6fd2af9b1171b8b988a64cfcf4417c1cd` - seed180 provenance recovery tooling implementation.
- `8752646b106eb5b11d2de5241fce874edae75087` - prediction row-count reconciliation authority.
- `9a9b11a3212fb0073d3f3678875bc2a3ae003501` - row-count remediation implementation.
- `3de16c2215fe50e6f17aabe5ae33da3eab3f8540` - dataset semantic SHA forensic attestation.
- `a5ffa107882947842be3d04993d3a534c6909490` - dataset semantic SHA reconciliation authority.
- `6189be22715e435ddc3247271e4966bb3d3b526d` - semantic SHA recovery remediation implementation.

Repository `AGENTS.md` applies.

## Shared A0 Scientific Envelope

The formal A0 authority fixes the A0 baseline/control arm:

- seed set: `180`, `181`, `182`;
- split seed: `174`;
- architecture: `v6b_minimal`;
- backbone/model: `mamba`, `state-spaces/mamba-130m-hf`;
- frozen encoder;
- epochs: `20`;
- max length: `128`;
- dev ratio: `0.2`;
- learning rate: `0.001`;
- flag source: `controlled_heuristic`;
- selection metric: `final_macro_f1`;
- class weighting: `none`;
- reason-router arm: `A0`;
- router mode: `explicit_product`;
- gradient ownership: `joint`;
- effective reason loss: `0.0`;
- all ranking, Stage174C, Stage175B, Stage177C, and compatible-margin objective terms neutralized;
- no A0 reference predictions consumed.

Any admissible A0 analysis must preserve this as a baseline/control envelope.
A0 evidence alone does not establish effectiveness or superiority of the P2
mechanism and does not establish any A1/A2/A3 conclusion.

## Evidence Matrix

| Evidence unit | Code/scientific envelope correctness | Execution success | Artifact/provenance validity | Scientific admissibility |
| --- | --- | --- | --- | --- |
| seed180 original A0 run plus recovery chain | Envelope matches formal A0 authority by original command SHA `dde0f92b09bc0b6c3a5334c0f7519de113e9405f5291bc583d81209b1706c21e` and recovered trainer provenance anchors. | `OBSERVED` completed; attempt `CONSUMED`. | Historical standard cm wrapper provenance remains `INCOMPLETE / MISSING`. Artifact bytes are immutably anchored by SHA256/size in recovery authorities and the stage174a_v1 provenance schema was later reconciled/remediated, but no repository result-import commit or recovery audit-output artifact analogous to seed181 R1/seed182 was found. | `CAVEATED_ADMISSIBLE_ONLY_IF_EXPLICITLY_LABELED`. Not equivalent to a normal wrapper-collected/imported run. May enter a primary A0 aggregate only if the future analysis authority explicitly accepts the seed180 recovery chain as sufficient and labels the caveat. |
| original seed181 formal attempt | Formal seed181 command existed under the A0 envelope; command SHA `3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`. | `OBSERVED` runtime completion, but attempt permanently `CONSUMED`. | Original artifacts and standard wrapper provenance are `DESTROYED_OR_UNAVAILABLE`; no original artifact SHA256/size is validated. | `INADMISSIBLE`. Must never be represented as recovered or used as the seed181 evidence unit. |
| seed181 `REPLACEMENT_R1` | Replacement authority verifies execution-relevant formal-vs-recovery blobs are byte-identical and that exactly three output-destination occurrences changed to the replacement namespace. Same seed `181`, split seed `174`, A0 arm, model, flags, neutralizations, and omission of `--reason-loss-weight`. | Validated/imported replacement result exists at commit `fb4f0e2c2a8382a642f1272b66f29552adaecb0e`; trainer provenance status `completed`. | Result artifacts are immutable Git objects at the replacement namespace with validated SHA256/size below. Selected checkpoint is intentionally not committed; its SHA256/size are bound through validated trainer provenance only. | `PRIMARY_ADMISSIBLE_AS_SEED181_REPLACEMENT_R1`. May serve as the seed181 replicate only when labeled `seed181 REPLACEMENT_R1`, with caveat that it is a separately authorized replacement, not the consumed original. |
| seed182 original A0 result | Formal A0 seed182 envelope; command SHA `5c0a7609069f8c6e4a5ae4c27bda7c9cbd1be6f3cbb35a0b42d18acfd7dd1fac`. Trainer provenance records seed `182`, A0 arm, `explicit_product`, and `joint`. | Validated/imported result exists at commit `82739bdfc8eee184de10ed8f55434f203a6d59a5`; trainer provenance status `completed`. | Result artifacts are immutable Git objects at the original current-lineage namespace with validated SHA256/size below. Selected checkpoint is intentionally not committed; its SHA256/size are bound through validated trainer provenance only. | `PRIMARY_ADMISSIBLE`. |

## Seed180 Resolution

Seed180 must not be silently normalized into a standard wrapper-provenance run.

What is recovered or validated:

- the original trainer attempt is `CONSUMED`;
- execution success is `OBSERVED`;
- `run_provenance.json` source Git commit is
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- `run_provenance.json` source dirty state is `false`;
- the five source artifacts are anchored by exact SHA256 and size;
- selected checkpoint SHA256 is anchored;
- clean dev prediction cardinality `720` was reconciled through prediction
  artifacts and genuine stage174a_v1 dev-row fields;
- dataset semantic SHA was reconciled through genuine stage174a_v1 semantic
  paths after forensic attestation;
- recovery tooling was remediated through `6189be22715e435ddc3247271e4966bb3d3b526d`.

What is not recovered:

- historical standard cm `run.meta`;
- historical standard cm `run.log`;
- historical standard cm `command.sh`;
- historical standard cm `start.marker`;
- a repository result-import commit for seed180 artifacts;
- a repository recovery audit-output artifact proving a completed recovery import.

Seed180 verdict:

`CAVEATED_ADMISSIBLE_ONLY_IF_EXPLICITLY_LABELED`

Seed180 may be used in a future primary A0 multi-seed aggregate only if the
future frozen analysis authority explicitly accepts the recovered-artifact and
trainer-provenance chain as sufficient for that aggregate and reports the
standard-wrapper-provenance caveat. Without that explicit caveated admission,
seed180 is not eligible for the planned primary 3-seed aggregate.

Seed180 artifact anchors:

| Path under `reports/reason_router_p3w7_a0_current_lineage_runs/seed180/A0` | Size | SHA256 |
| --- | ---: | --- |
| `training_report.json` | 306114 | `71423b654722f055d20876bb9e7d4029c8e06e57d878359b1452232516915508` |
| `clean_dev_predictions.json` | 4838225 | `92c5f1da0d7fe8c3b51ade4cf323bf660d509f500143d15f475060065c254aa2` |
| `training_report_predictions.jsonl` | 3934123 | `e4fc95992dcd9dc3ea35da7527d948fdd9e419f8764aea6ef87b8b14fe6ac9ef` |
| `selected_checkpoint.pt` | 518269815 | `dbf663f32c7780ab6ebe949dbe79cc205576cd3e6b7686591e1623fb039282da` |
| `run_provenance.json` | 68429 | `4fa8d362000e0a085368306328c1562b09a2ffea7bc7932cf5edeaa037ea9c3b` |

These are recovery-chain anchors, not Git blob identities in the current
repository tree.

## Seed181 Resolution

Original seed181:

- original run name: `p3w7-a0-seed181`;
- formal A0 authority: `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- original command SHA256:
  `3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`;
- original disposition: `CONSUMED`;
- original execution success: `OBSERVED`;
- original artifact/provenance status: `DESTROYED_OR_UNAVAILABLE`;
- original scientific admissibility: `INADMISSIBLE`.

The original seed181 artifacts must never be represented as recovered.

Replacement R1:

- recovery authority freeze:
  `74defa2c679ca2244d69b6ee950dd4a6a7a643b4`;
- replacement execution authority freeze:
  `592844f0806e96a37d01c4445cd96b990cf0dae4`;
- replacement result commit:
  `fb4f0e2c2a8382a642f1272b66f29552adaecb0e`;
- replacement label: `REPLACEMENT_R1`;
- replacement run name:
  `p3w7-a0-seed181-runtime-loss-replacement-r1-74defa2c679c`;
- replacement command SHA256:
  `2b4722e3442580eae21b676d5a4a82f1b5aebbb776f159ace68ebe1571a42d0d`;
- replacement namespace:
  `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0`.

Seed181 `REPLACEMENT_R1` may serve as the seed181 replicate in a future A0
aggregate only under label `seed181 REPLACEMENT_R1`, with explicit caveat that
it is a distinct separately authorized replacement execution and not a recovery
of the consumed original attempt.

Seed181 `REPLACEMENT_R1` committed artifacts:

| Path | Size | SHA256 |
| --- | ---: | --- |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report.json` | 305374 | `6eddb2f101bd513b91befe1b6edefcd078cc61e5626db6d13876d9b85b198ff3` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/clean_dev_predictions.json` | 4840808 | `893534cd27d8df7bf5eb6a7fa888b17b4ee803bb1d2a8233d40c15f8cf62ae12` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/training_report_predictions.jsonl` | 3936706 | `b186af00684279095a2257ef46f826be281a92bb2da9b0b9ee8f157f5bdbc13c` |
| `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/run_provenance.json` | 68927 | `11885151854401107b82b05c49da009bf47d9f465d348bc13965d60facad4e10` |

Seed181 `REPLACEMENT_R1` selected checkpoint identity from
`run_provenance.json`:

- path:
  `reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_74defa2c679c/seed181/A0/selected_checkpoint.pt`;
- size: `518269943`;
- SHA256:
  `3dbd7c32cc2d60b2de13da3a72cff05eaa080520f7cac076225c5a55870721ca`.

No Git blob identity is claimed for the checkpoint.

## Seed182 Resolution

Seed182 result identity:

- formal A0 authority:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- seed182 command SHA256:
  `5c0a7609069f8c6e4a5ae4c27bda7c9cbd1be6f3cbb35a0b42d18acfd7dd1fac`;
- validated/imported result commit:
  `82739bdfc8eee184de10ed8f55434f203a6d59a5`;
- namespace:
  `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0`;
- trainer provenance source Git commit:
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- trainer provenance dirty state: `false`;
- trainer provenance status: `completed`.

Seed182 committed artifacts:

| Path | Size | SHA256 |
| --- | ---: | --- |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report.json` | 305735 | `319e13bbda07363a334d0b6615b2c4074dfcf5d30d0c43e1f0735f269c2b5e3e` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/clean_dev_predictions.json` | 4842166 | `80205044dceed9b2131cd3caf06524f7869b1651577504dca1503605d0471036` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/training_report_predictions.jsonl` | 3938064 | `95aa57b9f14ff7119b19f0ec8e412bf5b4494ae325e73d4c0ef0df5b24e050e5` |
| `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/run_provenance.json` | 68388 | `8357efe9ff609b8a99f580aa47ecbe4d018b5a0d24668755369e6e65a6cab421` |

Seed182 selected checkpoint identity from `run_provenance.json`:

- path:
  `reports/reason_router_p3w7_a0_current_lineage_runs/seed182/A0/selected_checkpoint.pt`;
- size: `518269815`;
- SHA256:
  `212873153bc6cecf107e79a4ea86385033c7944a9af222d4984192b232803946`.

No Git blob identity is claimed for the checkpoint.

## Common Dataset And Sidecar Identities

Canonical dataset path:

`reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl`

Dataset physical SHA256:

`eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3`

Dataset semantic SHA256:

`3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b`

P4-L sidecar path:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`

P4-L sidecar physical SHA256:

`2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`

P4-L sidecar semantic SHA256:

`0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`

P4-L provenance path:

`reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`

P4-L provenance physical SHA256:

`9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`

## Permitted Future Analysis

If this candidate is independently verified, frozen, and later explicitly
activated by a subsequent analysis authorization, analysis is limited to
read-only static parsing of already validated A0 result artifacts.

Allowed input files are only:

- admissible or explicitly caveated-admissible `training_report.json`;
- admissible or explicitly caveated-admissible `run_provenance.json`;
- admissible or explicitly caveated-admissible `clean_dev_predictions.json`;
- admissible or explicitly caveated-admissible
  `training_report_predictions.jsonl`;
- immutable recovery-chain authority/attestation files needed only to label
  seed180 caveats.

Allowed per admissible seed extraction:

- selected/best epoch, from `training_report.json` key `best_epoch` or a
  semantically equivalent explicitly sourced selected-checkpoint field;
- `best_dev_macro_f1`;
- `best_dev_acc`, and `final_accuracy` only where explicitly present and
  semantically identical to the selected clean-dev accuracy source;
- selected checkpoint identity, from validated committed provenance or
  recovery-chain checkpoint SHA256/size anchors;
- per-label precision/recall/F1, only from existing report metric structures;
- frame, predicate, sufficiency, and polarity diagnostics already present in
  the report/provenance artifacts;
- existing intervention/pairwise diagnostics already present in the training
  report, including `best_dev_interventions` and `best_dev_pairwise_checks`
  when present.

Forbidden:

- trainer execution;
- model loading;
- checkpoint loading;
- new forward passes;
- new prediction generation;
- dataset evaluation;
- threshold tuning;
- candidate selection from external diagnostics;
- A1/A2/A3 execution or interpretation;
- changing any artifact, report, sidecar, dataset, or checkpoint.

## Aggregation Policy

Aggregation policy is fixed before any aggregate conclusion may be computed.

Primary A0 aggregate membership under this candidate:

- seed180 original A0: `CAVEATED_ADMISSIBLE_ONLY_IF_EXPLICITLY_LABELED`;
- seed181 original A0: excluded, `INADMISSIBLE`;
- seed181 `REPLACEMENT_R1`: included as `seed181 REPLACEMENT_R1`;
- seed182 original A0: included as `seed182`.

Therefore the default primary-admissible membership without extra seed180
caveated acceptance is:

`{seed181 REPLACEMENT_R1, seed182}`

Default primary N:

`2`

This N=2 set must not be presented as the planned 3-seed A0 aggregate.

If a later frozen analysis authority explicitly accepts seed180 as caveated
primary evidence, the permitted primary membership becomes:

`{seed180 CAVEATED_RECOVERY, seed181 REPLACEMENT_R1, seed182}`

Caveated primary N:

`3`

If three primary-admissible or explicitly caveated-primary-admissible
replicates exist, permitted descriptive statistics are:

- per-seed values;
- arithmetic mean;
- sample standard deviation with denominator `N - 1`;
- min/max where useful.

If fewer than three primary-admissible replicates exist, every analysis output
must report the exact N and must prohibit presenting the result as the planned
3-seed A0 aggregate.

No two-seed aggregate may be silently substituted for the planned 3-seed A0
aggregate.

## Scientific Boundary

A0 is a baseline/control arm.

A0 results alone do not establish:

- effectiveness of conditional first-blocker routing;
- effectiveness of reason-specific supervision;
- effectiveness of explicit gradient ownership;
- superiority of the P2 mechanism;
- any A1/A2/A3 conclusion.

No A1/A2/A3 execution or interpretation is authorized by this candidate.

## Activation Boundary

Candidate existence alone authorizes no analysis.

Before any analysis occurs, an independent verifier must confirm:

- all referenced commits resolve to immutable Git commit objects;
- all referenced committed result paths exist at the named commits;
- all committed result SHA256/size records in this candidate match the named
  Git objects;
- seed180 is labeled according to its actual recovery-chain status and not
  represented as standard wrapper-collected provenance;
- original seed181 is permanently excluded and not represented as recovered;
- seed181 `REPLACEMENT_R1` is labeled as a replacement, not an original;
- seed182 result identity is tied to
  `82739bdfc8eee184de10ed8f55434f203a6d59a5`;
- no unsupported P2/A1/A2/A3 scientific claim is introduced.

Only after independent verification, immutable freeze commit, and explicit
subsequent analysis authorization may read-only/static A0 analysis proceed.
