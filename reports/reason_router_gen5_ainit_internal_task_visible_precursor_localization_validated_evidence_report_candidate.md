# Gen5 A-init Internal Task-Visible Precursor Localization Validated Evidence Report

## Evidence identity

- execution authority commit:
  `e7eba19b102016131e4990f724c825cbec49ec5c`
- authority correction commit:
  `e2c563188e9534e898c6ea944c5a05f4056b2fe3`
- source confirmatory evidence freeze:
  `1468938af9753fa9f4a511d4e7f740dea0110bba`
- source forward-Jacobian recovery evidence freeze:
  `a4e78ee84bbfcc64d859ccc6af3efc0f987ee07e`
- source residual-localization evidence freeze:
  `3c0a3d8a67e9910f91de2354ba29a5c4b3b28942`
- run:
  `gen5-ainit-internal-precursor-e2c5631-r1`
- executed command SHA256:
  `aabfd9f652638503c8fcde23c4f0bf273b730dd52376ca0b14e6ef9ba54f7ef1`
- imported handoff ZIP SHA256:
  `4a742cbc63faa85d0d8185736a225e3700827b4413fd1cf6b9b45269e8896950`
- run log SHA256:
  `5edadaf81f3722103910904f1f7c8c107d8339a5aa1851f13d9f8eb55d52df44`
- run meta SHA256:
  `5060e5ede315b5c907d0451cdc457465bc5a2f4dde6fcec3cb6fe16f63a6143e`

Imported artifact identities:

- `ainit_internal_precursor_localization_summary.json`
  SHA256 `25c3e86d8d4ebbd19f7945755b5ca8033d5b05b0a6dd2a6804709b61c34293d2`
- `internal_stage_task_visible_metrics.pt`
  SHA256 `d00aedc95873ab47716d96773fdb81ee9a3b81c80479b06de38959829e9afa9d`
- `run_provenance.json`
  SHA256 `a1408d07b6e8e3ed4dddfcab38466acfc2a915a2bb32195806122f102ca7c8e5`

## Execution validity

The internal precursor audit completed successfully under the corrected frozen
authority contract.

Observed execution:

- exact HEAD matched;
- two Tesla T4 workers were used;
- deterministic source-cell parity sharding was used;
- GPU0 processed source cells 0,2,4,6,8 and 20 complete orientations;
- GPU1 processed source cells 1,3,5,7 and 16 complete orientations;
- all 840 frozen Phase3A P0 dev rows were processed by both workers;
- no pair orientation was split across GPUs;
- confirmatory `9601..9900` was not loaded.

No training, optimizer construction, parameter-gradient accumulation, `.backward()`,
checkpoint mutation, or confirmatory-population access occurred.

Analysis autograd was used only on detached internal stage leaves under the
forward-equivalent joint-gradient semantics authorized by the recovered
forward-Jacobian evidence.

## Authentication

All semantic guardrails passed.

Observed maximum discrepancies:

- frozen functional fingerprint max abs:
  `7.89761543274e-07`
- edge-specific versus joint forward logits:
  `0`
- full stage replay versus real endpoint:
  `7.89761543274e-07`
- reference versus frozen streaming correction semantics:
  `2.09808349609e-05`

At `layer22_out_proj`, the corrected recovery authentication reproduced:

- directional gain:
  observed `0.000305109176286`
  versus frozen `0.0003051091215898591`
- local task-row-space squared-energy fraction:
  observed `0.00415862672484`
  versus frozen `0.004158599422848653`

The recovered forward-Jacobian semantics therefore remained intact.

## Preregistered localization result

Scientific result:

`RAW_WRITE_PRECURSOR`

Earliest passing stage:

`raw_write`

All five ordered stages satisfied the frozen finite task-visible decomposition
criterion, but `raw_write` is the first stage by construction and therefore the
localized internal precursor.

## Stagewise results

### 1. raw_write

Same-training-RNG / different-A residual:

`0.45922708704`

Local task-row-space squared-energy fraction:

`0.000323695863574`

Mean signed-permutation control fraction:

`0.0000263172556629`

Actual/control enrichment:

`12.2997575325`

Centered-logit finite intervention:

- `R_visible = 0.89483541376`
- `R_complement = 0.00438080799597`
- `R_interaction = 0.00840346338907`

Two-margin finite intervention:

- `R_visible = 0.894958000002`
- `R_complement = 0.00440660190005`
- `R_interaction = 0.00843951843385`

Preregistered gate:

`PASS`

Thus approximately `0.0324%` of raw-write residual squared energy lies in the
source-local two-margin Jacobian row space, yet that tiny component reproduces
about `89.5%` of the measured endpoint functional-effect energy.

The much larger orthogonal complement contributes under `0.45%` of the endpoint
effect under the finite intervention.

### 2. recurrent_state

- residual: `0.235291923486`
- task-row energy fraction: `0.000731257881536`
- control fraction: `0.00000571972625045`
- enrichment: `127.848405591`
- centered `R_visible/R_complement/R_interaction`:
  `0.888550731319 / 0.00398127026635 / 0.00744435729382`
- margins `R_visible/R_complement/R_interaction`:
  `0.888783520104 / 0.00399334924973 / 0.00742988211672`
- gate: `PASS`

### 3. c_readout_pre_gate

- residual: `0.201189572384`
- task-row energy fraction: `0.00503912297853`
- control fraction: `0.0000787036200197`
- enrichment: `64.0265718053`
- centered `R_visible/R_complement/R_interaction`:
  `0.835742078262 / 0.0115340938224 / 0.019890663813`
- margins `R_visible/R_complement/R_interaction`:
  `0.83598721418 / 0.0115526234768 / 0.0198481098766`
- gate: `PASS`

### 4. gated_scan

- residual: `0.139630056208`
- task-row energy fraction: `0.00360756863498`
- control fraction: `0.0000419685789097`
- enrichment: `85.9587989086`
- centered `R_visible/R_complement/R_interaction`:
  `0.871823600104 / 0.00629319575612 / 0.0155368797096`
- margins `R_visible/R_complement/R_interaction`:
  `0.872004738067 / 0.00630973376831 / 0.015500962163`
- gate: `PASS`

### 5. layer22_out_proj

- residual: `0.155009134597`
- task-row energy fraction: `0.00415862672484`
- control fraction: `0.000111689247832`
- enrichment: `37.2339039395`
- centered `R_visible/R_complement/R_interaction`:
  `0.875034786322 / 0.00785712400933 / 0.0250881266692`
- margins `R_visible/R_complement/R_interaction`:
  `0.875129348819 / 0.0078796288773 / 0.0250476870596`
- gate: `PASS`

## Mechanistic interpretation

The data reject the hypothesis that recurrence, C readout, gating, or
out-projection first creates the task-visible A-init component.

Under the frozen local intervention contract, the task-visible component is
already present in the learned correction write:

`raw_write = B_theta A_theta x`

The later Mamba operations preserve and reshape this decomposition rather than
creating task visibility de novo.

The validated bounded mechanism statement is:

`A_INIT_SPECIFIC_REPRESENTATIONAL_NON_IDENTIFIABILITY_IS_ALREADY_PRESENT_AT_THE_LEARNED_STATE_WRITE_BOUNDARY_WHERE_A_TINY_TASK_VISIBLE_COMPONENT_CARRIES_MOST_MEASURABLE_FUNCTIONAL_DIFFERENCE_AND_A_MUCH_LARGER_COMPONENT_IS_ALREADY_DOWNSTREAM_LOW_GAIN`

This strengthens the endpoint result by localizing the phenomenon to the
earliest tested A-init-dependent computation.

## Relation to prior residual localization

The earlier residual-propagation audit established that raw A-init residual
magnitude decreases through recurrence/readout/gating and collapses strongly
before final logits.

The present result adds a different fact:

the functional visible/complement separation is already well formed at the
raw-write boundary.

Therefore the recurrence should not be described as creating the null/visible
decomposition. It may suppress, reshape, or concentrate an already-existing
decomposition.

## Scientific boundary

This evidence supports a bounded causal mechanism claim inside the frozen
layer-22 correction pathway.

It does not establish:

- exact mathematical gauge symmetry;
- a universal Mamba law;
- arbitrary perturbation invariance;
- a global null manifold;
- an optimization-trajectory mechanism;
- that the same decomposition appears in an unconstrained fully trained Mamba;
- that the same result holds on natural-language tasks or other domains;
- Transformer inferiority or Mamba superiority.

The experiment localizes the earliest tested A-init-dependent boundary, not an
upstream precursor in frozen parent layers 0..21.

## Next scientific requirement

Before making a priority or "first" claim in a manuscript, conduct a formal
literature novelty audit covering:

- Mamba/SSM mechanistic interpretability;
- activation subspace bottlenecks;
- random-initialization representation equivalence;
- state-space null-space methods;
- learned write/read dynamics;
- representational gauge/non-identifiability literature.

The scientific result itself should be frozen before any new mechanism
extension.
