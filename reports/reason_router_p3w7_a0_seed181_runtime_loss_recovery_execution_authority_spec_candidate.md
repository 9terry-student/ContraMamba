# P3-W7-A0 Seed181 Runtime-Loss Recovery Execution Authority Specification Candidate

Authority/version:

`P3W7_A0_SEED181_RUNTIME_LOSS_RECOVERY_EXECUTION_AUTHORITY_V1`

## Status

CANDIDATE ONLY.

This candidate authorizes nothing by existence alone. It is a recovery
execution-authority candidate for the P3-W7-A0 seed181 case where the originally
authorized seed181 attempt was observed to complete successfully, but the Kaggle
runtime was reset before successful collection/import and all original artifact
bytes and standard wrapper provenance were lost.

This candidate does not authorize training, evaluation, Kaggle execution, GPU
use, checkpoint regeneration, artifact mutation, cleanup, overwrite, resume,
retry, restoration, A1, A2, A3, result promotion, or scientific interpretation.

Future replacement execution requires all of:

1. independent verification PASS for this exact candidate;
2. this exact candidate frozen in a new immutable Git commit;
3. exact replacement command construction from the frozen formal seed181 A0
   command source of truth and the future recovery authority freeze;
4. exact replacement command SHA256 freeze and verification;
5. exact clean checkout of the recovery authority freeze;
6. standard post-freeze local gates and per-run preflight;
7. explicit subsequent execution authorization.

## Authority Basis

Current recovery-authority drafting base:

`9a249c071b76fbf693f63b36ba8ec1036c69b2ba`

Formal P3-W7-A0 execution authority freeze:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Formal authority path:

`reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`

Prior analogous seed180 recovery authority:

`reports/reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md`

Prior analogous seed180 recovery authority freeze:

`233ed0be080e1d30dd47de2e66136475ec2ede76`

Seed182 validated-result freeze, used here only as sibling context that a
separate seed result freeze exists:

`82739bdfc8eee184de10ed8f55434f203a6d59a5`

The seed182 freeze is not the base of this candidate and does not authorize,
repair, replace, or validate seed181.

Operational material inspected for this candidate:

- `AGENTS.md`
- `docs/RESEARCH_OPERATIONS.md`
- `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`
- `reports/reason_router_p3w7_a0_seed180_provenance_recovery_execution_authority_spec_candidate.md`
- repository references to P3-W7-A0 seed181 command/output naming

## Formal A0 Attempt-Consumption Contract

The formal A0 authority states:

- exactly one authorized A0 trainer attempt exists for each of seeds 180, 181,
  and 182;
- wrapper rejection before trainer launch does not consume the seed attempt;
- trainer process launch consumes the seed attempt regardless of later PASS or
  FAIL;
- any retry requires a separately created, independently verified, frozen
  recovery authority;
- no automatic retry, resume, cleanup, overwrite, output reuse, alternate seed,
  or scope change is authorized.

This candidate preserves that contract. It does not reopen the original seed181
attempt and does not transform a future replacement into the original attempt.

## Supplied Incident Evidence

Original run name:

`p3w7-a0-seed181`

Formal A0 authority HEAD:

`2737c3c6116ae3766b469801f990e2c45ba9a55e`

Authorized seed181 wrapper command SHA256:

`3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`

Original wrapper `STARTED_UTC`:

`2026-09-01T02:45:25Z`

Original wrapper `FINISHED_UTC`:

`2026-09-01T02:48:25Z`

Original wrapper `EXIT_CODE`:

`0`

Operational observations supplied for this candidate:

- the trainer completed epochs 1 through 20;
- runtime output showed `CONTRAMAMBA RUN PASS`;
- before successful collection/import, the Kaggle runtime/filesystem was reset;
- a subsequent read-only diagnostic showed all named original repo/runtime,
  wrapper, trainer-output, checkpoint, prediction, and provenance files missing
  from the surviving Kaggle runtime;
- no displaced matching files were found under `/kaggle/working`.

These observations are recorded only as supplied operational evidence. This
candidate does not infer artifact hashes, artifact sizes, selected checkpoint
identity, prediction bytes, run-provenance bytes, or scientific results from
console text.

## Original Attempt Disposition

Seed:

`181`

Original attempt disposition:

`CONSUMED`

Execution success status:

`OBSERVED`

Artifact/provenance status:

`DESTROYED_OR_UNAVAILABLE`

Scientific conclusion:

`NOT_ESTABLISHED`

The original seed181 trainer attempt is permanently consumed. It must never be
rerun, resumed, continued, restored, or represented as unconsumed under the
formal A0 authority.

The observed successful runtime completion is not enough to establish valid
local artifacts, standard `cm` handoff provenance, A0 reference predictions, or
any scientific conclusion.

## Lost Original Artifact Boundary

The following original seed181 bytes are unavailable and cannot be recovered
from the surviving Kaggle runtime according to the supplied diagnostic evidence:

- original `run.log`;
- original `run.meta`;
- original `command.sh`;
- original `start.marker`;
- original `training_report.json`;
- original `clean_dev_predictions.json`;
- original `training_report_predictions.jsonl`;
- original `selected_checkpoint.pt`;
- original `run_provenance.json`;
- any other original files under
  `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0`;
- any matching displaced files under `/kaggle/working`.

No SHA256 or size for the destroyed seed181 artifacts is available as a
validated immutable artifact anchor. No future document, importer, wrapper, or
analysis may claim the destroyed original seed181 artifact identity is known
unless a separate authenticated immutable store is later discovered and
separately authorized for validation.

Do not infer artifact hashes from console text. Do not fabricate historical
`run.meta`, `run.log`, `command.sh`, `start.marker`, checkpoint bytes,
prediction exports, `training_report.json`, or `run_provenance.json`.

## Seed180 Analogy Boundary

The seed180 recovery authority is useful only as a fail-closed provenance
precedent. Its artifact-recovery mechanism is not applicable to seed181.

Seed180 recovery depended on surviving immutable source artifact bytes with
known SHA256 and size anchors. Seed181 has no surviving original artifact bytes
and no validated artifact hash/size anchors. Therefore seed181 cannot use a
seed180-style recovery-only capture/import package to recover the original run.

Any seed181 path that pretends newly produced bytes are recovered original bytes
is forbidden.

## Recovery Model Decision

Chosen recovery model:

`NEW_SEPARATELY_AUTHORIZED_REPLACEMENT_EXECUTION`

Reason:

The original seed181 attempt was consumed and its artifact bytes are
unavailable. The narrow truthful path is not artifact recovery. It is a future
replacement execution, separately authorized, separately named, separately
collected/imported, and separately interpreted.

The replacement execution is not:

- the original attempt;
- a continuation;
- a resume;
- a restoration;
- a byte-identical recovery;
- a repair of historical wrapper provenance;
- evidence that the destroyed original artifacts had any particular SHA256,
  size, selected checkpoint, predictions, or metrics.

The replacement execution may only create a new seed181 A0 replacement-result
candidate under the same scientific envelope, with explicit provenance linking
it back to the consumed original attempt and to the recovery authority that
authorized the replacement.

## Required Scientific Envelope Preservation

Any future replacement execution must preserve the original A0 scientific
envelope exactly:

- seed `181`;
- split seed `174`;
- architecture `v6b_minimal`;
- backbone `mamba`;
- model `state-spaces/mamba-130m-hf`;
- frozen encoder;
- frame downstream mode `joint`;
- epochs `20`;
- maximum sequence length `128`;
- dev ratio `0.2`;
- learning rate `0.001`;
- device `cuda`;
- flag source `controlled_heuristic`;
- selection metric `final_macro_f1`;
- no class weighting;
- reason-router arm `A0`;
- router mode `explicit_product`;
- gradient ownership `joint`;
- reason-loss effective `0.0` via omission of `--reason-loss-weight`;
- ranking weight `0.0`;
- all Stage174C, Stage175B, Stage177C, and compatible-margin neutralizations
  exactly as in the frozen formal A0 seed181 command;
- no A0 reference predictions;
- no A1, A2, or A3;
- no hyperparameter changes.

The frozen formal seed181 command in the A0 authority is the semantic source of
truth. A future replacement authority must derive the replacement command from
that command rather than manually inventing, normalizing, reordering for taste,
or upgrading trainer flags.

## Replacement Naming and Output Namespace

The original run name `p3w7-a0-seed181` and original output namespace
`reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0` are consumed
historical identifiers. A replacement execution must use distinct identifiers
that cannot impersonate, overwrite, or collide with them.

Recommended convention for the future exact replacement run name:

`p3w7-a0-seed181-runtime-loss-replacement-r1-<recovery-shortcommit>`

Recommended convention for the future exact replacement output root:

`/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_seed181_runtime_loss_replacement_runs/replacement_r1_<recovery-shortcommit>/seed181/A0`

`<recovery-shortcommit>` must be replaced in the later frozen execution
authority by the unambiguous short prefix of that authority's immutable freeze
commit. It must not remain a runtime placeholder in an executable command.

The later authority may choose a longer commit prefix if required to avoid
ambiguity. It must not choose the original run name, the original A0 output
directory, or any seed180/seed182 namespace.

## Future Command Freeze Decision

The recovery command itself must not be frozen in this candidate.

Fail-closed reason:

This file is a candidate and does not yet have its own immutable recovery
authority freeze SHA. The future replacement command must bind to the later
frozen recovery authority commit, must resolve the distinct replacement run name
and output namespace using that freeze identity, and must have a new exact
command SHA256 computed over the final one-line command bytes. Freezing a
command now would either predict a future commit identity or omit the recovery
authority binding that this candidate requires.

A later authority, after this candidate is independently verified and frozen,
must freeze:

- the exact replacement run name;
- the exact replacement output namespace;
- the exact one-line replacement wrapper command;
- the exact replacement command SHA256;
- the exact recovery authority freeze SHA;
- the exact clean checkout and command-hash verification gates.

Placeholder command hashes are prohibited.

## Standard `cm` Handoff Requirement

If the future replacement execution is performed through the current standard
workflow, recovery collection/import must use standard `cm` wrapper provenance.

That means the replacement must preserve standard `cm run save`, `cm run`,
`cm collect`, local ZIP acquisition, and `cm import` validation, including
wrapper-created `run.log`, `run.meta`, `command.sh`, `start.marker`, command
SHA verification, expected/actual commit equality, file hashes, file sizes,
manifest validation, and import audit.

Standard `cm` provenance must not be weakened, bypassed, or retrofitted.
Historical seed181 wrapper provenance remains lost and is not repaired by a
replacement run's valid wrapper provenance.

Collection and local ZIP acquisition must complete before accelerator changes,
session changes, notebook resets, or runtime/filesystem changes that could reset
`/kaggle/working`.

## Required Replacement Provenance Fields

Any valid replacement result must retain explicit provenance fields linking it
to:

- original formal A0 authority SHA
  `2737c3c6116ae3766b469801f990e2c45ba9a55e`;
- original seed181 command SHA
  `3794fbdcb9e347a13aef02a258bab2a7a597d49acee12686d363cb178e5ae1ea`;
- original consumed-attempt disposition `CONSUMED`;
- observed original execution success `OBSERVED`;
- original artifact/provenance status `DESTROYED_OR_UNAVAILABLE`;
- recovery authority freeze SHA;
- replacement command SHA;
- replacement run name;
- replacement output namespace;
- replacement artifact SHA256 values and sizes;
- standard `cm` handoff/import audit identity and status.

The replacement result must label itself as a replacement result. It must never
be represented as byte-identical recovery of the destroyed original artifacts.

Scientific interpretation remains separate from execution success, artifact
validity, wrapper provenance validity, and import success.

## Explicit Non-Authorizations

This candidate does not authorize:

- immediate execution of any command;
- direct or indirect trainer launch;
- Kaggle execution;
- seed181 original-run continuation, resume, restoration, or overwrite;
- reuse of run name `p3w7-a0-seed181`;
- reuse of output directory
  `/kaggle/working/ContraMamba/reports/reason_router_p3w7_a0_current_lineage_runs/seed181/A0`;
- mutation of scripts, tests, datasets, P4-L sidecar/provenance, formal A0
  authority, seed180 recovery artifacts/specs, seed182 result artifacts,
  `AGENTS.md`, `cm.ps1`, canonical main/O0b work, or existing untracked/user
  files;
- manual invention or normalization of trainer flags;
- any A1, A2, or A3 work;
- class weighting, reason-loss-weight addition, ranking-weight change, seed
  change, split change, model change, device change, epoch change, output reuse,
  or hyperparameter search;
- fabrication of historical wrapper or trainer artifacts;
- scientific conclusion or result promotion.

## Failure Semantics

Any evidence that original seed181 artifacts still exist in the repository or
an authenticated immutable store is a stop condition. In that case the
replacement-execution model must be halted and a separate artifact-recovery
authority question must be drafted.

Any conflict with the formal A0 attempt-consumption contract is a stop
condition.

Any proposed path that requires weakening standard `cm` provenance validation,
pretending a replacement is the original attempt, or crossing from candidate
drafting into training/evaluation is forbidden.

Any future replacement run that launches the trainer is a new consumed recovery
attempt under its own authority. If it fails after trainer launch, it must not be
silently retried under the same authority unless that future authority
explicitly permits a bounded retry model.

## Next Required Stage

The next authorized action is independent verification of this candidate.

If verification passes, the candidate may be frozen in a new immutable Git
commit only under explicit commit authorization. A later, separate execution
authority may then freeze the exact replacement command and command SHA.

No training/evaluation is authorized before that later explicit execution
authorization.
