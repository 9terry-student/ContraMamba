# Long-Term O0a Native Mamba State Dynamics Execution Authority Spec Candidate

Status:
PRE-FREEZE EXACT EXECUTION AUTHORITY CANDIDATE / NOT YET ACTIVE

## Activation Rule

This exact document becomes O0a execution authority only after:

1. independent verifier PASS over these exact bytes;
2. this exact file is committed and pushed unchanged;
3. the controller identifies that resulting commit as the authority-freeze commit.

No post-freeze textual edit is required for activation.

This file is NOT active execution authority while uncommitted.

Required sequence:

1. independent verifier reviews these exact bytes;
2. verifier returns PASS;
3. cm ship;
4. stage ONLY this authority candidate;
5. commit/push unchanged;
6. controller records that commit as O0a authority-freeze identity.

After step 6, the document becomes ACTIVE execution authority without another textual modification.

The authority-freeze commit is the authority-document identity.

The separate execution implementation identity remains:

```text
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401
```

## Authority And Identity Separation

AUTHORITY STORAGE IDENTITY
!=
EXECUTION IMPLEMENTATION IDENTITY

The authority document may be stored/frozen in a commit later than the implementation commit.

The authorized scientific execution MUST run against:

```text
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401
```

not against the later authority-storage commit.

The execution checkout must therefore be an isolated detached worktree pinned to the implementation commit.

This separation is intentional and solves the otherwise self-referential problem where committing the authority document would change HEAD.

The main working tree must not be switched away from main for O0a execution.

## 1. Scientific Question

Authorized O0a question:

Do layer hidden-state proxy trajectories of frozen native pretrained Mamba show early entitlement-sensitive separation under the controlled toy intervention families?

Measurement target:

Hugging Face MambaModel layer hidden states.

Preferred terminology:

native pretrained Mamba hidden-state proxies

Explicitly NOT:

- selective-SSM recurrent state itself
- cache_params state
- A/B/C/Delta dynamics
- a hallucination detector
- a generated-answer experiment

O0a is mechanistic screening only.

## 2. Exact Implementation Binding

Execution repository HEAD:

```text
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401
```

Observer path:

```text
scripts/observe_longterm_o0a_native_mamba_state_dynamics.py
```

Exact authorized runtime observer SHA256:

```text
2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f
```

Require runtime script to receive:

```text
--authority-repository-head
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401

--authority-script-sha256
2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f
```

Repository HEAD establishes canonical code lineage.

Runtime script SHA binds exact observer bytes and prevents unstaged/local observer modification.

If the isolated execution worktree produces a different script SHA because of line-ending or checkout-byte differences:
STOP.
Do not normalize/rewrite/patch the script.
Do not bypass the check.
Return the observed SHA for recovery review.

## 3. Dataset Binding

Dataset path:

```text
data/toy_interventions_v5.jsonl
```

Exact SHA256:

```text
17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc
```

Expected:

- 18 rows
- 3 pair_id families
- exactly one none per pair
- intervention families:
  - none
  - paraphrase
  - entity_swap
  - predicate_swap
  - evidence_deletion
  - polarity_flip

Interpretation boundary:

none:
entitlement-preserving reference

paraphrase:
whole-pair semantic-invariance / surface-form control

entity_swap:
Frame-related non-entitlement intervention

predicate_swap:
Predicate-related non-entitlement intervention

evidence_deletion:
Sufficiency-related non-entitlement intervention

polarity_flip:
authorized REFUTE semantic-sensitivity control

Do not classify polarity_flip as non-entitlement or hallucination.

## 4. Model / Tokenizer Binding

Model ID:

```text
state-spaces/mamba-130m-hf
```

Model revision:

```text
5708daa364c50b880e7bd92eab456e0d34492ee9
```

Tokenizer ID:

```text
state-spaces/mamba-130m-hf
```

Tokenizer revision:

```text
5708daa364c50b880e7bd92eab456e0d34492ee9
```

Mutable main is forbidden.

No alternative model/revision is authorized.

## 5. Numerical / Execution Binding

device:
cpu

dtype:
float32

deterministic tolerance:
1e-6

No GPU is authorized or needed.

If Kaggle accelerator is enabled, user must turn GPU OFF before this run.

Forbidden:

- CUDA
- float16
- bfloat16
- torch.compile
- optimizer
- backward
- training
- generation
- checkpoint writing
- model modification
- ContraMamba semantic heads
- reason-router code

Allowed:

- tokenizer/model download at exact immutable revision
- frozen MambaModel inference
- output_hidden_states
- deterministic descriptive artifact generation

## 6. Execution Environment Isolation

The authorized execution environment is:

Kaggle CPU execution from a separate isolated Git detached worktree whose HEAD is exactly:

```text
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401
```

The main ContraMamba working tree may continue advancing due to the separate URP track.

That does NOT invalidate O0a provided the isolated execution worktree remains exactly bound to the implementation commit and exact observer SHA.

Before Kaggle bootstrap, the controller/user will create an isolated local worktree from the main repo.

Conceptual local path:

```text
C:\Users\Home1\Desktop\ContraMamba-o0a-9f595e6
```

The exact filesystem path is operational, not scientific identity.

Required isolated-worktree state before bootstrap:

```text
git rev-parse HEAD
=
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401

git status --short
=
empty

runtime observer SHA256
=
2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f

dataset SHA256
=
17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc
```

If any differ:
STOP before cm kaggle.

## 7. Run Identity

Authorized run name:

```text
longterm-o0a-native-hidden-proxy-screen-v1
```

Do not reuse this run name for:

- a different commit
- a different command
- a retry with changed semantics
- a repaired implementation

If a provenance-compatible infrastructure retry becomes necessary, it must be separately authorized and use a new retry name.

## 8. Exact Authorized Command

The exact scientific execution command is:

```text
python -u scripts/observe_longterm_o0a_native_mamba_state_dynamics.py --dataset data/toy_interventions_v5.jsonl --output-dir reports/longterm_o0a_native_mamba_state_dynamics_execution_9f595e6_v1 --authority-repository-head 9f595e64f8a6aaec5bb1975521a6ee86e2ab1401 --authority-script-sha256 2937223733750f72344f18d30f3686d4ffc96cde5b397524560443a9e885489f --authority-dataset-sha256 17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc --device cpu --dtype float32 --deterministic-tolerance 1e-6
```

No argument may be edited in the Kaggle cell.

Run command identity must additionally be pinned by the existing cm run registry/command-SHA mechanism.

## 9. Output Contract

Output directory:

```text
reports/longterm_o0a_native_mamba_state_dynamics_execution_9f595e6_v1
```

It must not exist before execution.

Required exact artifacts:

- manifest.json
- observations.jsonl
- terminal_hidden_states.npz
- paired_distances.jsonl
- summary.json
- report.md
- SHA256SUMS.txt

No incomplete output set may be interpreted scientifically.

## 10. Prefix / Measurement Contract

Serialization:

```text
Claim: <claim>
Evidence: <evidence-prefix>
```

Prefix schedule:

- 0.00
- 0.25
- 0.50
- 0.75
- 1.00

Token-space prefixing only.

Nonzero token count:

```text
ceil(fraction * N)
```

Repeated actual token counts may share one forward while requested aliases are retained.

0% contains no evidence-content token.

Claim-identical:

- entity_swap
- predicate_swap
- evidence_deletion
- polarity_flip

must satisfy the exact 0%-identity invariant against none.

Paraphrase is excluded from that equality invariant.

## 11. Observations / Descriptive Metrics

Authorized descriptive measurements include:

- terminal hidden norm
- last-step hidden delta
- terminal consecutive-state cosine
- terminal acceleration
- evidence-region mean delta
- evidence-region max delta
- normalized terminal L2 to none
- terminal cosine distance to none
- transition-magnitude difference
- descriptive trajectory-summary difference
- paraphrase ranking diagnostic

Explicitly preserve:

```text
D_l2^2 = 2 * D_cos
```

for unit-normalized nonzero vectors.

Normalized L2 and cosine distance are algebraically redundant and must NOT be treated as independent corroboration.

trajectory_summary_difference has no independent inferential weight.

## 12. No Learned Inference

O0a does NOT authorize:

- classifier fitting
- linear probes
- threshold fitting
- best-layer selection for confirmatory claims
- p-value/significance testing
- population inference
- generalization estimates

No hard PASS threshold exists.

## 13. Screening Interpretation Boundary

Potential descriptive clue:

Frame/Predicate/Sufficiency interventions diverge from none:

- before 100% evidence,
- after the causally differing evidence arrives,
- more consistently than whole-pair paraphrase distance,
- across multiple pair families,
- without unexplained 0%-prefix divergence,
- and not solely in one isolated layer.

This is only a clue pattern.

It does NOT establish:

- hallucination prediction
- pre-hallucination detector validity
- causal semantic ownership
- population generalization

Negative/surface-form-dominated results are scientifically valid outcomes.

## 14. Execution Success / Evidence Separation

Explicitly preserve four separate judgments:

1. code correctness
2. execution success
3. artifact/provenance validity
4. scientific interpretation

Successful command exit is not enough.

Complete artifacts are not enough.

A scientifically interesting-looking report is not enough if provenance fails.

No scientific conclusion may be made until artifacts are collected/imported and independently validated.

## 15. Kaggle Workflow

Once this authority is ACTIVE:

From the isolated worktree only:

```text
cm kaggle
```

GPU must remain OFF.

After bootstrap PASS and exact SHA confirmation, the controller provides the exact command above.

Then:

```text
cm run save longterm-o0a-native-hidden-proxy-screen-v1
cm run longterm-o0a-native-hidden-proxy-screen-v1
```

The generated Kaggle cell must not be manually edited.

After run completion, success or failure:

```text
cm collect longterm-o0a-native-hidden-proxy-screen-v1
```

Run the generated collector cell in Kaggle.

Download the handoff ZIP.

Then import from the appropriate local repository/controller context using:

```text
cm import <handoff.zip>
```

Do not interpret scientific results before IMPORT PASS and provenance review.

GPU/session note:
GPU should stay OFF for the entire O0a run.
Terminate the Kaggle session when collection is complete.

## 16. Fail-Closed Conditions

STOP without scientific interpretation on:

- execution worktree HEAD mismatch
- runtime observer SHA mismatch
- dirty execution worktree before bootstrap
- dataset SHA mismatch
- output directory collision
- model/tokenizer revision mismatch
- mutable model revision
- device/dtype drift
- missing intervention family
- duplicate IDs/references
- token-prefix invariant failure
- 0%-identity invariant failure
- API hidden-layer layout inconsistency
- non-finite tensor/metric
- incomplete artifact set
- artifact checksum failure
- cm command provenance mismatch
- imported handoff provenance mismatch

Do not bypass or edit around blockers.

## 17. URP Separation

The O0a authority:

- consumes no URP A0-A3 authority
- consumes no reason-router checkpoint
- consumes no URP attempt
- does not change URP run counts
- does not interpret URP scientific evidence
- does not modify URP files

The separate main-tree URP activity may proceed independently.

## 18. Authorized Claim Maximum

After valid execution/import, the maximum immediate claim is descriptive:

"Under this fixed 18-row / 3-family mechanistic screening set, the frozen Mamba-130m hidden-state proxy trajectories exhibited [observed descriptive pattern] under controlled evidence-prefix interventions."

Do not yet say:

- predicts hallucinations
- detects hallucinations before output
- generalizes
- establishes native recurrent-state mechanism
- validates ContraMamba semantic ownership

Those require later experiments.

## 19. Current Task Boundary

This Codex task only creates the authority candidate.

NO:

- worktree creation
- cm kaggle
- model download
- model forward
- run save
- run
- collect
- import
- training
- generation
- staging
- commit
- push

Expected delta:
ONLY

```text
?? reports/longterm_o0a_native_mamba_state_dynamics_execution_authority_spec_candidate.md
```

Do not change:

- the existing O0a observer
- O0a tests
- existing O0a implementation candidate
- hypothesis map
- vision
- dataset
- src/
- any URP/reason-router file
- reports/stage180a_pass2_annotations_completed.csv
- root .patch files
- any other file

## 20. Provenance Handling

At task start inspect current HEAD.

The authority must ALWAYS bind execution implementation commit:

```text
9f595e64f8a6aaec5bb1975521a6ee86e2ab1401
```

even if current main HEAD has advanced due to unrelated URP commits.

If current main has advanced:

- verify 9f595e6 is an ancestor;
- inspect intervening paths;
- continue only if drift does not alter:
  - O0a observer/tests,
  - O0a hypothesis map,
  - O0a dataset,
  - AGENTS.md,
  - relevant cm execution/provenance tooling.

Unrelated URP-only drift may coexist.

## 21. Validation Contract For This Candidate

Validation required for this documentation-only task:

- inspect complete new document
- verify every SHA is full and exact
- verify exact command appears once as the canonical command
- verify run name is exact
- verify implementation identity and authority-storage identity are explicitly separated
- verify Kaggle CPU / GPU OFF
- verify execution is still NOT authorized before freeze
- git diff --check / no-index equivalent
- git status --short

Training/Evaluation allowed:
NO.

Commit/Push:
NO.
