# Gen4 PP3-Excluded Residual Individual-Plane Necessity Localization — Confirmatory Result

## Status

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_NECESSITY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`

This report records the completed prospective confirmatory individual residual-plane
necessity localization experiment on the fresh XG1 holdout.

The result is positive under the frozen design, with exactly four one-sided
Student t-tests across the pre-registered family `{P1, P2, P4, P5}` and Holm
step-down control of familywise error at `alpha = 0.05`.

The unordered supported set is:

`{P1, P2, P4, P5}`

No plane ranking is inferred.

## Scientific question

Within the already established PP3-excluded residual subspace

`R = P1 ⊕ P2 ⊕ P4 ⊕ P5`,

which, if any, of the four pre-registered residual principal planes has an
individually detectable local causal necessity contribution to the frozen
layer-17 / target-token native-Mamba susceptibility endpoint, relative to an
equal-norm orthogonal quarter-turn matched control within the same plane?

All four planes were prospectively included in the confirmatory family.

No outcome-guided plane selection or post-hoc promotion was permitted.

## Frozen design

Design commit:

`d046a8e03e7522a72dfbd08cc9129b769cd5686a`

Static preparation commit:

`ab9a6ebbc95bec20e8682b9538365f53167f108e`

Implementation authority commit:

`ac13d93785719345fc0362298c9988bf040e694d`

Implementation commit:

`87acea31459d5b08e2af02e840c098c3a3e188b4`

Original execution freeze:

`f6cd2dca4b76e54ba603877d5730001bd078ea2e`

Retry1 execution freeze:

`fceffc87571116f0007dd6a7ef0346fd6f4d6883`

Retry2 execution freeze / accepted execution HEAD:

`aeffcdf9d18c81ba07df8d245c880ece074778c2`

## Population

Prospective fresh XG1 holdout:

`xg1_fact_1801..xg1_fact_2100`

Pair count:

`N = 300`

The holdout was frozen before scientific execution.

## Conditions

Exactly nine conditions were run in frozen order:

1. `native`
2. `p1_neutralized`
3. `p1_quarter_turn_control`
4. `p2_neutralized`
5. `p2_quarter_turn_control`
6. `p4_neutralized`
7. `p4_quarter_turn_control`
8. `p5_neutralized`
9. `p5_quarter_turn_control`

For each plane `Pk`, the branch-local native coefficient pair was:

`a_k = <h, p_k+>`

`b_k = <h, p_k->`

with native plane component:

`c_k = a_k p_k+ + b_k p_k-`

The treatment neutralized that exact native plane component:

`delta_N,k = -c_k`

The matched control used the frozen within-plane quarter-turn:

`r_k = -b_k p_k+ + a_k p_k-`

`delta_C,k = -r_k`

The matched control therefore preserved intervention norm while rotating the
native coefficient pair by 90 degrees within the same plane.

PP3 was preserved, and the other residual planes were required to remain
unchanged under each individual-plane intervention.

## Endpoint

Shared native endpoint:

`Q0 = Q(native)`

For each plane `Pk`:

`QN,k = Q(Pk neutralized)`

`QC,k = Q(Pk quarter-turn matched control)`

Native-plane attenuation:

`A_N,k = Q0 - QN,k`

Matched-control attenuation:

`A_C,k = Q0 - QC,k`

Primary confirmatory endpoint:

`D_k = A_N,k - A_C,k = QC,k - QN,k`

The canonical recorded form was revalidated item-by-item before confirmatory
inference:

`D_k = QC,k - QN,k`

for every item and every one of P1, P2, P4, and P5.

## Execution

Accepted run name:

`g4k-residual-individual-plane-necessity-xg1-1801-2100-aeffcdf-retry2`

Execution HEAD:

`aeffcdf9d18c81ba07df8d245c880ece074778c2`

Pinned run command SHA256:

`f326e3d0bce85c7dbf374264cb87076a212b01baae8ed2873da7f82a09c3c68a`

Run log SHA256:

`d29b6c16a0d6081bdb088348ddbb47448a5b9d382567b3798650b83a11864da2`

Run meta SHA256:

`c6c6d27f264bdc567488b6f6234d26452686a30d34d89fd02067eaffd13ce2f9`

Imported handoff ZIP SHA256:

`de46bbca52dff104b8849ed66b1db4f503cbfe58ea985cefc1bfde4f869e3d34`

Execution result:

`PASS_PP3_EXCLUDED_RESIDUAL_INDIVIDUAL_PLANE_NECESSITY_LOCALIZATION_RAW_OBSERVATION`

GPU count:

`2`

Shard 0:

`xg1_fact_1801..xg1_fact_1950`

Scientific forwards:

`54000`

Shard 1:

`xg1_fact_1951..xg1_fact_2100`

Scientific forwards:

`54000`

Total scientific model forwards:

`108000`

Baseline model forwards:

`0`

Raw primary inference:

`False`

Raw multiplicity correction:

`False`

Raw scientific conclusion:

`None`

## Runtime identity

Retry2 zero-forward preflight passed with:

`kernels == 0.10.2`

Frozen Mamba binary SHA256:

`dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587`

Frozen causal-conv1d binary SHA256:

`6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6`

Transport identity status:

`EXACT_FROZEN_BINARY_SHA256_MATCH`

Preflight scientific model forwards:

`0`

Preflight checkpoint loads:

`0`

## Raw artifact validation

Imported raw files:

1. `SHA256SUMS.txt`
2. `artifact_manifest.json`
3. `pp3_excluded_residual_individual_plane_necessity_items.jsonl`
4. `pp3_excluded_residual_individual_plane_necessity_summary.json`

Import validation:

`PASS`

Validated imported files:

`4`

Frozen runner artifact validation:

`PASS`

Item count:

`300`

Endpoint revalidation:

`PASS`

Raw scientific model forwards:

`108000`

Raw baseline model forwards:

`0`

No raw artifact contained primary inference, multiplicity correction, or a
scientific conclusion.

## Confirmatory family

Frozen family:

`{P1, P2, P4, P5}`

For each plane `Pk`:

`H0,k: mean(D_k) <= 0`

`H1,k: mean(D_k) > 0`

Test:

one-sample Student t-test, one-sided greater

Sample size:

`N = 300`

Degrees of freedom:

`df = 299`

Raw confirmatory p-value count:

`4`

Additional p-value count:

`0`

Multiplicity method:

Holm step-down across exactly P1, P2, P4, and P5

Familywise alpha:

`0.05`

No fifth p-value, subgroup test, alternative-tail test, rescue analysis,
response-guided plane selection, or plane-ranking test was executed.

## Shared positive gate

Mean native endpoint:

`mean(Q0) = 1.8920353590615954e-07`

Shared gate:

`mean(Q0) > 0`

Result:

`PASS`

## P1 result

Mean neutralization attenuation:

`mean(A_N,P1) = 2.8320710131173835e-08`

Mean matched-control-adjusted endpoint:

`mean(D_P1) = 1.6805447554809328e-08`

Sample standard deviation:

`sd(D_P1) = 3.2734570942260901e-08`

Student t statistic:

`t(299) = 8.8920942511224794`

One-sided raw p-value:

`p = 2.9047373440553166e-17`

Holm step-down threshold at its ordered step:

`0.016666666666666666`

Holm rejection:

`PASS`

Frozen plane gates:

- `mean(Q0) > 0`: `PASS`
- `mean(A_N,P1) > 0`: `PASS`
- `mean(D_P1) > 0`: `PASS`
- Holm rejection: `PASS`

P1 individual local necessity support:

`SUPPORTED`

## P2 result

Mean neutralization attenuation:

`mean(A_N,P2) = 2.8646069304788239e-08`

Mean matched-control-adjusted endpoint:

`mean(D_P2) = 1.8197843675927426e-08`

Sample standard deviation:

`sd(D_P2) = 1.5875721508561748e-08`

Student t statistic:

`t(299) = 19.853957388899666`

One-sided raw p-value:

`p = 7.7731454652991615e-57`

Holm step-down threshold at its ordered step:

`0.012500000000000001`

Holm rejection:

`PASS`

Frozen plane gates:

- `mean(Q0) > 0`: `PASS`
- `mean(A_N,P2) > 0`: `PASS`
- `mean(D_P2) > 0`: `PASS`
- Holm rejection: `PASS`

P2 individual local necessity support:

`SUPPORTED`

## P4 result

Mean neutralization attenuation:

`mean(A_N,P4) = 1.8540349335461023e-09`

Mean matched-control-adjusted endpoint:

`mean(D_P4) = 3.1782303407902329e-09`

Sample standard deviation:

`sd(D_P4) = 2.6170423399192339e-08`

Student t statistic:

`t(299) = 2.1034647947558693`

One-sided raw p-value:

`p = 0.018130193204377327`

Holm step-down threshold at its ordered step:

`0.050000000000000003`

Holm rejection:

`PASS`

Frozen plane gates:

- `mean(Q0) > 0`: `PASS`
- `mean(A_N,P4) > 0`: `PASS`
- `mean(D_P4) > 0`: `PASS`
- Holm rejection: `PASS`

P4 individual local necessity support:

`SUPPORTED`

## P5 result

Mean neutralization attenuation:

`mean(A_N,P5) = 1.2380632555142974e-08`

Mean matched-control-adjusted endpoint:

`mean(D_P5) = 1.4818512341760204e-08`

Sample standard deviation:

`sd(D_P5) = 3.4307305756407402e-08`

Student t statistic:

`t(299) = 7.4813267036341236`

One-sided raw p-value:

`p = 4.1264762552556459e-13`

Holm step-down threshold at its ordered step:

`0.025000000000000001`

Holm rejection:

`PASS`

Frozen plane gates:

- `mean(Q0) > 0`: `PASS`
- `mean(A_N,P5) > 0`: `PASS`
- `mean(D_P5) > 0`: `PASS`
- Holm rejection: `PASS`

P5 individual local necessity support:

`SUPPORTED`

## Holm family conclusion

Holm-rejected planes:

`{P1, P2, P4, P5}`

Frozen supported set:

`{P1, P2, P4, P5}`

The set is intentionally unordered.

All four pre-registered residual planes satisfy their individual local
necessity criteria under familywise error control.

## Scientific conclusion

The fresh XG1 holdout supports the claim that each of the four pre-registered
PP3-excluded residual principal planes — P1, P2, P4, and P5 — has an
individually detectable local causal necessity contribution to the frozen
layer-17 / target-token native-Mamba susceptibility endpoint relative to its
pre-specified equal-norm orthogonal within-plane quarter-turn matched control.

This localizes the previously established aggregate residual necessity result:
the aggregate causal contribution is not supported only as an undifferentiated
whole; each of the four prospectively tested constituent residual planes passes
its own matched-control necessity criterion under Holm familywise correction.

This does not imply that the four contributions are additive, independent,
exchangeable, or equal in scientific importance.

## Relation to prior residual evidence

Before this experiment, two properties of the PP3-excluded residual were
already established on separate fresh XG1 populations:

1. the aggregate residual orientation transported as XG2-like relative to the
   frozen XG4 residual template;
2. the complete residual subspace `R = P1 ⊕ P2 ⊕ P4 ⊕ P5` had aggregate local
   necessity relative to a within-R quarter-turn matched control.

The present result adds a new causal localization statement:

`each of P1, P2, P4, and P5 is individually locally necessary under its frozen matched-control test`

The XG2-like template orientation remains a geometric transport result. This
experiment does not establish that the XG2-like template direction itself is a
causal object.

## Interpretation boundary

This result does **not** establish:

- sufficiency of P1, P2, P4, or P5;
- dominance or ranking among P1, P2, P4, and P5;
- that the plane with the smallest p-value has the largest causal effect;
- an additive decomposition of aggregate residual necessity into the four
  individual results;
- absence of interactions among residual planes;
- equality or independence of plane contributions;
- causal necessity or sufficiency of the XG2-like residual-template
  orientation itself;
- aggregate residual sufficiency;
- global identity of XG1 and XG2;
- behavioral or downstream task necessity;
- benchmark improvement;
- universality across checkpoints, layers, token positions, generators,
  datasets, architectures, or Mamba models generally.

The individual-plane results are bounded local-intervention statements under the
frozen model, checkpoint, layer, token, geometry, intervention, and endpoint.

## Failed-attempt provenance

### Original pinned-run command orchestration failure

The original run identity

`g4k-residual-individual-plane-necessity-xg1-1801-2100-f6cd2dc`

failed before scientific preflight because PowerShell orchestration bytes were
mistakenly stored as the Kaggle Bash command.

Scientific model forwards:

`0`

Scientific conclusion:

`None`

### Retry1 runtime package failure

Retry1

`g4k-residual-individual-plane-necessity-xg1-1801-2100-fceffc8-retry1`

failed during zero-forward runtime preflight because the Kaggle environment was
missing the required Python package metadata for:

`kernels==0.10.2`

Scientific model forwards:

`0`

Scientific conclusion:

`None`

### Retry2 manual repair diagnostic failure

The subsequent manual retry2 environment-repair diagnostic successfully
installed:

`kernels==0.10.2`

but then failed because `validate_transformers_kernel_bindings()` was invoked
before the model-construction lazy kernel-loader path had established the
`mamba_ssm` and `causal_conv1d` module bindings.

This was a diagnostic/preflight ordering error, not a scientific execution
failure.

Scientific model forwards:

`0`

Scientific conclusion:

`None`

The accepted retry2 pinned run removed that premature diagnostic condition,
authenticated the exact frozen binaries in zero-forward preflight, and then
allowed the frozen runner's existing model-construction path to perform its own
kernel-binding validation in the intended order.


## Repository archival form for oversized raw item artifact

The imported raw item artifact

`pp3_excluded_residual_individual_plane_necessity_items.jsonl`

has original byte size:

`155282457`

and original SHA256:

`b9dbcebdae9bbebb256a9e0fc84a6db886b3619c1c9236a9353e45172b0610cf`

Because this single raw file exceeds GitHub's 100 MiB object limit, the repository
stores a deterministic gzip archival copy instead of the uncompressed JSONL:

`pp3_excluded_residual_individual_plane_necessity_items.jsonl.gz`

Archived gzip SHA256:

`af033fdafa2751eb0f158a41131039b2d5cf9989ce1adadfc8cd24cb1c3dedbd`

Archived gzip byte size:

`5518743`

The gzip was created with deterministic metadata (`mtime = 0`, empty embedded
filename, compression level 9).

Decompressing the archived gzip reproduces the exact original raw JSONL SHA256:

`b9dbcebdae9bbebb256a9e0fc84a6db886b3619c1c9236a9353e45172b0610cf`

The scientific raw artifact manifest and `SHA256SUMS.txt` remain unchanged and
continue to describe the original uncompressed raw artifact set. They are not
rewritten to reinterpret the scientific artifact.

The uncompressed JSONL remains preserved in the local working tree during this
commit operation but is intentionally excluded from Git history solely because
of the remote object-size limit.

## Final status

Raw execution:

`PASS`

Artifact/provenance validation:

`PASS`

Endpoint revalidation:

`PASS`

Confirmatory inference:

`PASS`

Raw confirmatory p-values:

`4`

Additional p-values:

`0`

Holm familywise correction:

`PASS`

Supported planes, unordered:

`{P1, P2, P4, P5}`

Scientific conclusion:

`PP3_EXCLUDED_INDIVIDUAL_RESIDUAL_PLANE_NECESSITY_LOCALIZATION_SUPPORTED_ON_FRESH_XG1_HOLDOUT`
