# Gen4 Six-Cell Native Mamba State Bridge Feasibility Audit Authority Specification — Candidate

STATUS = CANDIDATE

AUTHORITY_ID =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_FEASIBILITY_AUDIT_AUTHORITY

PARENT_COMMIT =
a2617aa037d1a9834003535b62ac81770a5b96aa

PARENT_SPECIFICATION =
reports/reason_router_gen4_six_cell_native_mamba_state_mechanistic_bridge_spec_candidate.md

AUTHORITY_CLASS =
OUTCOME_BLIND_STATIC_AND_BOUNDED_PROVENANCE_FEASIBILITY_AUDIT

PHASE =
GEN4_NATIVE_MAMBA_STATE_BRIDGE_PHASE_A_B

CODEX =
PROHIBITED

KAGGLE =
NOT_AUTHORIZED

GPU =
NOT_AUTHORIZED

MODEL_FORWARD =
NOT_AUTHORIZED

NATIVE_STATE_EXTRACTION =
NOT_AUTHORIZED

PRIMARY_KINEMATIC_ENDPOINT_COMPUTATION =
NOT_AUTHORIZED

PRIMARY_STATISTICAL_TESTING =
NOT_AUTHORIZED

TRAINING =
NOT_AUTHORIZED

EVALUATION =
NOT_AUTHORIZED


# 1. PURPOSE

This authority permits exactly the outcome-blind feasibility work required
before implementation of the frozen Gen4 six-cell native-Mamba-state bridge.

It combines only:

PHASE A =
STATIC_NATIVE_BACKBONE_AND_INSTRUMENTATION_PROVENANCE

and

PHASE B =
GEN4_EVENT_ANCHOR_AND_PREFIX_FEASIBILITY_WITHOUT_NATIVE_STATES

into one bounded authority.

The purpose is to determine whether the frozen bridge measurement can later be
implemented without changing its scientific design.

This authority does not establish a native-state effect.
It does not authorize extraction of native states.
It does not authorize any of the 15 primary hypothesis tests.


# 2. FROZEN SCIENTIFIC PARENT

The controlling scientific specification is the exact file:

reports/reason_router_gen4_six_cell_native_mamba_state_mechanistic_bridge_spec_candidate.md

frozen at:

a2617aa037d1a9834003535b62ac81770a5b96aa

All layer, state, anchor, geometry, window, estimand, admission, and
multiplicity decisions frozen there remain unchanged.

This audit may determine whether those objects are technically identifiable
and feasible.

It may not optimize, tune, replace, or reinterpret them.


# 3. OUTCOME-BLINDNESS

No q_authorized value, R6 effect magnitude, R6 p-value, R6 effect direction,
or later native-state outcome may be used to choose:

- tensor source;
- layer;
- token coordinate;
- semantic anchor;
- POST4 window;
- state representation;
- state geometry;
- pair admission;
- checkpoint subset;
- endpoint definition;
- instrumentation path.

Previously frozen R6 behavioral findings remain scientific context only.

They are not audit decision variables.


# 4. AUTHORIZED READ SCOPE

Read-only inspection is authorized for the minimum repository material needed
to answer this audit, including:

1. the frozen parent bridge specification;

2. the frozen Gen4 six-cell structural artifact and its generator/provenance
   material;

3. the frozen R5/R6 provenance needed solely to resolve the prespecified
   18 evaluator checkpoint identities, checkpoint paths, hashes, tokenizer
   identity, and model/config identity;

4. existing native-state/O0c source, tests, reports, and provenance needed
   to determine whether the exact frozen native recurrent-state object can
   be instrumented;

5. model source/configuration needed to establish native-Mamba backbone
   boundaries, layer count, layer numbering, recurrent-state update order,
   and state_dict namespace;

6. tokenizer source/configuration and frozen active-encoding semantics needed
   to map generator-declared Gen4 semantic spans into token coordinates.

Repository search commands such as git grep, GitHub search, directory listing,
and exact-file reads are allowed only for locating these objects.

Unrelated untracked files are not evidence and must not be modified.

An untracked local checkpoint payload may be read only after its exact identity
has first been resolved from frozen provenance and its expected file identity
has been verified.


# 5. PHASE A — EXACT NATIVE STATE SOURCE BINDING

The audit must identify the exact implementation object corresponding to:

s_t^(l) =
the vectorized native selective-SSM recurrent state of layer l
after token t has been consumed.

The audit must establish, without model forward:

- exact module/class/function;
- exact tensor or cache object;
- state update timing relative to consumed token t;
- layer indexing convention;
- state tensor dimensions and vectorization semantics;
- whether the state is genuinely the native selective-SSM recurrent state.

The following are not acceptable substitutes:

- ordinary downstream hidden state;
- cached encoder output;
- router activation;
- q_authorized;
- task logits;
- arbitrary post-block representation.

If more than one technically plausible native recurrent-state tensor remains
and the frozen scientific definition does not uniquely select one:

NATIVE_TENSOR_SOURCE =
BLOCKED_AMBIGUOUS

and the audit must stop before implementation.


# 6. PRIMARY LAYER BINDING

The architecture layer count L and exact zero-based/one-based source mapping
must be established statically.

The scientific primary layer remains:

L_PRIMARY = floor((L - 1) / 2)

over the frozen bridge's zero-based layer coordinate.

No best-layer scan is permitted.

No Q1/Q3 or other robustness layer is introduced by this authority.


# 7. EXISTING O0c INSTRUMENTATION REUSE FEASIBILITY

Existing O0c/native-state instrumentation may be inspected only as a
methodological implementation source.

The audit must classify reuse as one of:

REUSE_AS_IS

BOUNDED_PHASE_C_INSTRUMENTATION_DELTA_REQUIRED

NOT_FEASIBLE_WITH_CURRENT_SOURCE

A bounded later implementation delta may be described, but no production
source or test file may be modified under this authority.

Earlier confident-error native-state results and O0c scientific results are
not Gen4 evidence and must not be reused as Gen4 measurements.


# 8. PRESPECIFIED 18-CHECKPOINT NATIVE-BACKBONE IDENTITY

The exact 18 evaluator checkpoint identities must first be recovered from
frozen Gen4 evaluator provenance.

Directory enumeration alone does not define the checkpoint population.

For each prespecified checkpoint, the audit must verify the expected payload
identity before any tensor-level inspection.

If whole-checkpoint hashes differ because downstream/evaluator heads differ,
the audit may inspect checkpoint tensor payloads on CPU solely to determine
native-Mamba backbone identity.

Permitted checkpoint payload access is restricted to non-executing,
read-only tensor/state_dict inspection.

Preferred loading semantics:

torch.load(..., map_location="cpu", weights_only=True)

No fallback to unsafe arbitrary pickle/object execution is authorized.

No model class may be instantiated from a checkpoint.
No model forward may be executed.

If safe tensor-only access cannot read the required payload:

CHECKPOINT_TENSOR_INSPECTION =
BLOCKED_UNSAFE_OR_UNSUPPORTED

and the audit stops for that question.


# 9. NATIVE-MAMBA BACKBONE TENSOR SET

The audit must statically define the exact checkpoint key namespace belonging
to the native-Mamba encoder/backbone.

Evaluator/router/downstream-head parameters must be excluded from the
native-backbone identity decision.

For the resolved native-Mamba backbone tensor set, exact identity across all
18 checkpoints requires equality of:

- key set;
- tensor dtype;
- tensor shape;
- raw tensor value bytes.

Container metadata or downstream-head differences do not count as
native-backbone differences.

If every native-Mamba backbone tensor is exactly identical:

NATIVE_MAMBA_BACKBONE_IDENTITY =
IDENTICAL

NATIVE_STATE_MODEL_REPLICATION_COUNT =
1

If any native-Mamba backbone tensor differs:

NATIVE_MAMBA_BACKBONE_IDENTITY =
MISMATCH

EXECUTION =
BLOCKED

No checkpoint subset may be selected post hoc.
No 18-checkpoint multi-backbone analysis is authorized here.


# 10. TOKENIZER COORDINATE BINDING

The audit must bind the exact tokenizer/serialization semantics corresponding
to the frozen native backbone and Gen4 rendered input.

Tokenizer-only deterministic preprocessing is authorized in Phase B after
tokenizer identity is proven.

It may produce only quantities required for coordinate feasibility, such as:

- token IDs;
- token boundaries or equivalent deterministic span mapping;
- encoded length;
- semantic-anchor token indices;
- terminal token index;
- POST4 prefix eligibility.

Tokenizer execution is not model execution.

No hidden state, logit, q_authorized value, recurrent state, or kinematic
quantity may be produced.

If the 18 prespecified evaluator checkpoints do not admit one exact common
active encoding coordinate system for this bridge:

TOKENIZER_COORDINATE_BINDING =
BLOCKED

and scientific execution remains unauthorized.


# 11. GEN4 SEMANTIC EVENT ANCHORS

Only generator-declared semantic spans may define anchors.

The frozen anchors remain:

A_TITLE =
last consumed evidence token of the complete realized title span

A_NAME =
last consumed evidence token of the complete realized name span

A_ROLE =
last consumed evidence token of the complete realized role span

A_PREDICATE =
last consumed evidence token of the complete realized predicate span

A_IDENTITY =
last consumed token after the complete realized title + name identity block

Rendered-text similarity, state trajectories, logits, q_authorized values,
manual semantic scanning, and outcome-dependent token selection may not define
or alter an anchor.

Each cell is mapped independently.

Equal absolute token indices across cells must not be assumed.

If generator-declared span information cannot uniquely determine an anchor
under the frozen tokenizer coordinate:

ANCHOR_MAPPING =
BLOCKED

No rendered-text reconstruction may silently replace generator-declared
identity.


# 12. REQUIRED PRIMARY CELL-ANCHOR FEASIBILITY SET

For every one of the 300 source pairs, Phase B must establish coordinate and
prefix feasibility for exactly the cell-anchor combinations required by the
five frozen primary structural estimands:

A_TITLE:
C0_SHAM
C1_TITLE

A_NAME:
C0_SHAM
C2_NAME

A_ROLE:
C0_SHAM
C3_ROLE

A_PREDICATE:
C0_SHAM
C4_PREDICATE

A_IDENTITY:
C0_SHAM
C1_TITLE
C2_NAME
C5_TITLE_NAME

No additional cell-anchor combination is promoted to a primary requirement
by this authority.


# 13. POST4 PREFIX FEASIBILITY

For every required cell-anchor combination:

a + 4 <= terminal_index - 1

must hold under the frozen token coordinate.

The window must not be shortened.

SHORTENED_POST_WINDOW =
PROHIBITED

No source pair may be silently dropped.

No smaller complete-case sample may be substituted.

The complete primary feasibility gate is:

300 / 300 source pairs

with every required primary cell-anchor combination successfully mapped and
POST4 prefix eligible.

If the count is below 300 / 300:

PRIMARY_COMPLETE_PAIR_PREFIX_FEASIBILITY =
FAIL

EXECUTION =
BLOCKED

The audit records the failure without redesigning the experiment.


# 14. PERMITTED AUDIT OPERATIONS

Under this authority, the following local CPU-only operations are permitted:

- cryptographic hashing of frozen artifacts/checkpoints;
- exact metadata/schema inspection;
- static source inspection;
- safe tensor-only checkpoint state_dict inspection as restricted above;
- exact native-backbone tensor identity comparison;
- tokenizer-only deterministic encoding for Phase B;
- generator-span to token-coordinate mapping;
- terminal-index and POST4 eligibility calculation;
- deterministic manifest/report generation for this audit.

These operations are provenance/feasibility checks, not native-state
scientific execution.


# 15. PERMITTED AUDIT ARTIFACTS

The audit may write only bounded feasibility/provenance artifacts under:

reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_a2617aa/

Expected artifacts are:

native_backbone_identity_manifest_candidate.jsonl

event_anchor_prefix_manifest_candidate.jsonl

feasibility_summary_candidate.json

feasibility_audit_report_candidate.md

artifact_sha256_manifest.json

The audit report must record:

- authority commit actually used;
- parent scientific specification commit;
- every repository file actually inspected;
- every local checkpoint payload actually read;
- expected and observed checkpoint identities;
- exact native-backbone key-set definition;
- exact tensor-source binding;
- layer/time semantics;
- tokenizer identity and mapping semantics;
- 300-pair anchor/prefix counts;
- any blocker;
- whether a bounded Phase C implementation delta is required.

No scientific state-effect result belongs in these artifacts.


# 16. EXPLICITLY PROHIBITED

This authority does not permit:

- model forward;
- model inference;
- native recurrent-state extraction;
- POST4_SPEED computation;
- POST4_TURNING computation;
- POST4_PATH_EFFICIENCY computation;
- any of the 15 primary native-state hypothesis tests;
- new statistical testing;
- training;
- evaluator execution;
- Kaggle execution;
- GPU use;
- source-code implementation;
- source-code refactoring;
- test implementation;
- learned probes;
- PCA/UMAP/t-SNE outcome selection;
- best-layer scan;
- token scan;
- window scan;
- checkpoint subset selection;
- threshold relaxation;
- pair dropping;
- shortened POST4 windows;
- changing state geometry;
- changing the primary layer;
- changing the five structural estimands;
- adding state-level title-minus-name as a primary estimand;
- treating 18 downstream heads as 18 native-state replications;
- causal interpretation.


# 17. STOP CONDITIONS

Stop fail-closed if any of the following occurs:

- parent commit or required frozen artifact identity mismatch;
- prespecified 18-checkpoint population cannot be uniquely resolved;
- required checkpoint payload is missing or hash-mismatched;
- safe tensor-only checkpoint inspection is insufficient;
- exact native recurrent-state tensor source remains ambiguous;
- native-Mamba backbone tensors differ across checkpoints;
- layer/time indexing cannot be uniquely bound;
- tokenizer identity/serialization coordinate cannot be uniquely bound;
- generator-declared semantic span cannot be uniquely mapped;
- required primary POST4 feasibility is below 300 / 300;
- answering the audit requires model forward or native-state extraction;
- answering the audit requires production-code modification;
- answering the audit requires changing the frozen scientific design.

A blocker is an audit result.
It must not be bypassed by redesign inside this authority.


# 18. AUDIT DECISION CONTRACT

The strongest successful audit decision is:

PASS_READY_FOR_SEPARATE_PHASE_C_AUTHORITY

This decision requires all of the following:

- exact native recurrent-state tensor source is bound;
- layer/time semantics are bound;
- L_PRIMARY is mechanically determined from the frozen rule;
- instrumentation is either reusable as-is or requires only a bounded,
  explicitly described Phase C implementation delta;
- all 18 prespecified checkpoints have exactly identical native-Mamba
  backbone tensors;
- one exact active tokenizer coordinate is bound;
- every required Gen4 semantic anchor is deterministically constructible;
- all 300 / 300 source pairs satisfy the complete primary POST4 gate;
- no prohibited scientific execution occurred.

PASS does not authorize Phase C automatically.

PASS does not authorize native-state extraction.

PASS does not authorize the 15-test statistical analysis.

The next boundary, if PASS is frozen, is a separate minimal Phase C
measurement-implementation-and-synthetic-validation authority.


# 19. SCIENTIFIC INTERPRETATION LIMIT

This audit can establish technical identifiability and feasibility only.

It cannot establish:

NATIVE_MAMBA_STATE_KINEMATIC_RESPONSE

NATIVE_MAMBA_STATE_CAUSALITY

OUTPUT_STATE_MEDIATION

TRAINING_BENEFIT

TASK_PERFORMANCE_IMPROVEMENT

ARBITRARY_MODEL_GENERALIZATION

ARBITRARY_DATASET_GENERALIZATION

A successful audit is infrastructure/provenance evidence, not a scientific
native-state result.
