# ContraMamba Gen4 Mamba-1 Five-Scale Delta-L Mechanistic Autopsy

## Status

`STATIC_DELTA_L_MECHANISTIC_AUTOPSY_SYNTHESIS`

This synthesis introduces no model execution, training, backward pass, new
p-value, threshold estimation, rescue, re-selection, or row filtering.

## 1. Exact row-level identity

For every frozen readout row:

`Delta_L = L_selected - L_control`.

Because the selected and matched-control components have equal norm to numerical
precision, the readout reduces to:

`Delta_L_i = w_i * h_i`

where:

`w_i = ||g_i|| * ||component_i|| > 0`

and:

`h_i = cos(g_i, selected_i) - cos(g_i, control_i)`.

Across all five scales:

- maximum selected/control component-norm relative mismatch is approximately
  `1e-15`;
- rowwise reconstruction error is approximately `1e-18..1e-17`;
- `sign(Delta_L_i) = sign(h_i)` for every row tested.

Therefore the scale-dependent Delta-L sign cannot be attributed to mismatched
selected/control component norm.

## 2. Exact mean-moment decomposition

The frozen mean endpoint admits the exact decomposition:

`E[g n h] =
 mu_g * mu_n * mu_h
 + mu_g * Cov(n,h)
 + mu_n * Cov(g,h)
 + mu_h * Cov(g,n)
 + E[(g-mu_g)(n-mu_n)(h-mu_h)]`.

The five-scale values are:

| scale | mean h | base | norm-gap | grad-gap | grad-norm | triple | mean Delta_L |
|---|---:|---:|---:|---:|---:|---:|---:|
| 130M | +0.008881965010493697 | +0.001278782557357266 | +0.000133190457109135 | -0.000519576976183540 | -0.000012825062806655 | +0.000240795061055922 | +0.001120366036532126 |
| 370M | -0.000802156327488121 | -0.000094095358107031 | +0.000042452455047036 | +0.000402747883547661 | -0.000001430912906175 | +0.000159282284303926 | +0.000508956351885417 |
| 790M | +0.007793196921091330 | +0.001040616136353141 | -0.000068706620106335 | +0.001426961516464109 | -0.000040009026410042 | -0.000082150556444377 | +0.002276711449856494 |
| 1.4B | -0.009643894952004040 | -0.000877367363000991 | -0.000176563578561815 | -0.000101775327135446 | -0.000010052654126476 | -0.000029688583061496 | -0.001195447505886224 |
| 2.8B | +0.004473173637057720 | +0.000329550929066379 | -0.000171543926043767 | -0.000162660052525846 | -0.000009378023827281 | -0.000060143880171834 | -0.000074174953502349 |

## 3. Scale-dependent mechanisms

The same endpoint sign is produced by different joint-distribution mechanisms.

### 130M

The mean angular gap is positive.

The positive angular base is partially attenuated by negative gradient-gap
dependence but remains positive.

### 370M

The mean angular gap is slightly negative.

Nevertheless, positive gradient-gap dependence and positive higher-order
interaction outweigh the negative angular base.

Thus the positive 370M Delta-L is a covariance-rescued positive mean.

### 790M

The mean angular gap is strongly positive.

Positive gradient-gap dependence further amplifies the endpoint.

This scale combines positive average angular advantage with favorable
magnitude weighting.

### 1.4B

The mean angular gap is strongly negative.

Approximately 73.4 percent of the final negative magnitude is already present
in the negative angular base. The remaining dependence terms are also negative:

- norm-gap contribution: approximately 14.8 percent;
- gradient-gap contribution: approximately 8.5 percent;
- gradient-norm contribution: approximately 0.8 percent;
- centered three-way contribution: approximately 2.5 percent.

Both C0_SHAM and C2_NAME are negative.

Therefore the 1.4B reversal is a broad angular-coupling reversal reinforced by
magnitude dependence.

### 2.8B

The mean angular gap is positive:

`+0.004473173637057720`.

Its unweighted mean-angular base is therefore also positive:

`+0.000329550929066379`.

However, the total dependence correction is:

`-0.000403725882568728`.

This adverse dependence is approximately 122.5 percent of the positive base and
therefore reverses its sign.

Within that adverse correction:

- norm-gap coupling contributes approximately 42.5 percent;
- gradient-gap coupling contributes approximately 40.3 percent;
- centered three-way interaction contributes approximately 14.9 percent;
- the gradient-norm covariance term contributes approximately 2.3 percent.

Thus the 2.8B negative mean is not an average angular reversal.

It is a magnitude-weighted reversal: rows carrying larger gradient/component
weight are preferentially associated with worse selected-minus-control angular
gap.

## 4. 2.8B cell localization

At 2.8B:

C0_SHAM:

- mean Delta_L = `-0.000587831901597167`;
- angular base = `-0.000150157556464464`;
- norm-gap = `-0.000290551219095538`;
- gradient-gap = `-0.000084030187316978`;
- triple = `-0.000062999917564546`.

C2_NAME:

- mean Delta_L = `+0.000439481994592469`;
- angular base = `+0.000608442189854029`;
- norm-gap = `-0.000078818310190814`;
- gradient-gap = `-0.000043818830806836`.

C0 therefore contains both an average angular disadvantage and strong adverse
magnitude-gap coupling, while C2 retains a substantial positive angular
advantage.

The aggregate 2.8B endpoint becomes slightly negative because the C0 negative
contribution exceeds the surviving C2 positive contribution.

## 5. Projection magnitude is not the direct explanation

The selected-plane gradient projection fraction does not track the Delta-L sign
across scales.

For example:

- at 130M, selected projection fraction is below control while Delta-L is
  positive;
- at 2.8B, selected projection fraction is above control while Delta-L is
  negative.

Therefore the sign is not explained by how much gradient norm lies in the
selected plane alone.

The relevant quantity is signed angular alignment to the matched native
component, together with its sample-dependent gradient and component magnitude.

## 6. Coordinate bookkeeping

Where frozen item schemas retain the native selected coordinates, the paired
basis decomposition also reconstructs Delta-L exactly.

At 2.8B:

- plus-coordinate mean contribution:
  `+0.000051023572701968`;
- minus-coordinate mean contribution:
  `-0.000125198526204317`.

The negative minus-coordinate contribution dominates the smaller positive
plus-coordinate contribution.

These basis coordinates are frozen within-backbone bookkeeping axes and are not
asserted to be homologous semantic channels across model scales.

## 7. Mechanistic conclusion

The five-scale Delta-L sign pattern is not explained by a single monotonic
change in gradient magnitude, state magnitude, projection fraction, or average
angular alignment.

Instead, scale changes reorganize the joint distribution of:

1. task-gradient magnitude;
2. matched native-component magnitude;
3. selected-versus-control angular alignment.

The two negative-mean scales have distinct mechanisms:

- Mamba-1.4B: broad angular-coupling reversal, reinforced by adverse weighting;
- Mamba-2.8B: positive mean angular advantage survives, but adverse
  gradient-gap and component-gap dependence overwhelms it, with the negative
  contribution localized primarily to C0_SHAM.

The 370M result provides the complementary case: its average angular gap is
slightly negative, yet favorable magnitude-gap dependence produces a positive
Delta-L mean.

Thus the sign of the aggregate readout is a property of the joint
magnitude-alignment distribution rather than of mean geometry alone.

## 8. Scientific boundary

This is an exact algebraic and descriptive decomposition of frozen evidence.

It does not establish:

- a new inferential result;
- a new p-value;
- a universal scaling law;
- an exact parameter-count threshold;
- semantic equivalence of plane labels across backbones;
- that C0/C2 are independent causal mechanisms;
- any conclusion for D_CORE.

Previous signed-mass synthesis freeze:

`6017896f7a4b42282f8b9e096595a56f28c54413`

`DELTA_L_MECHANISTIC_AUTOPSY_SYNTHESIS = FROZEN_CANDIDATE`
