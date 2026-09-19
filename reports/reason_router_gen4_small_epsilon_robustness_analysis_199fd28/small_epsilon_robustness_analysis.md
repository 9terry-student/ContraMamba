# Gen4 small-epsilon robustness analysis

## Result

`SMALL_EPSILON_P3_SPECTRAL_DOMINANCE_PRESERVED`

This analysis is descriptive only: 0 inferential tests, 0 p-values, 0 multiplicity corrections, and 0 epsilon selections.

## Frozen inputs

- analysis HEAD: `199fd287f7071ec73817c4dd4cad3590e1927874`
- raw freeze: `552236c171f3467cd6eb12f02aa89a3eefee2f45`
- reference epsilon: `0.025`
- new epsilons: `[0.0125, 0.00625]`
- population N: `300`

## Mean signed spectral profiles

### epsilon = 0.025

- P1: `2.8557904254582005e-08`
- P2: `3.4693191890669348e-08`
- P3: `7.2858080938013261e-08`
- P4: `2.8744010983894689e-09`
- P5: `4.8767819587928714e-08`
- rank: `['P3', 'P5', 'P2', 'P1', 'P4']`
- unique spectral dominant candidate: `P3`
- P3 - P5 mean contribution: `2.4090261350084547e-08`

### epsilon = 0.0125

- P1: `2.9031340346826891e-08`
- P2: `3.4310646666594007e-08`
- P3: `7.3606255487119651e-08`
- P4: `3.0066516109215877e-09`
- P5: `4.835850382853564e-08`
- rank: `['P3', 'P5', 'P2', 'P1', 'P4']`
- unique spectral dominant candidate: `P3`
- P3 - P5 mean contribution: `2.524775165858401e-08`

### epsilon = 0.00625

- P1: `2.8418413437143893e-08`
- P2: `3.4665927346485558e-08`
- P3: `7.2258285690748869e-08`
- P4: `3.2958067966287045e-09`
- P5: `4.8018230707962314e-08`
- rank: `['P3', 'P5', 'P2', 'P1', 'P4']`
- unique spectral dominant candidate: `P3`
- P3 - P5 mean contribution: `2.4240054982786555e-08`

## Core qualitative robustness gates

### epsilon = 0.0125

- P3 unique argmax: `True`
- P3-P5 positive: `True`
- gate pass: `True`

### epsilon = 0.00625

- P3 unique argmax: `True`
- P3-P5 positive: `True`
- gate pass: `True`

## Normalized profile similarity

- cosine_0.025_vs_0.0125: `0.9999495217216572`
- cosine_0.025_vs_0.00625: `0.9999793916161879`
- cosine_0.0125_vs_0.00625: `0.9999457044844027`

## Finite-difference magnitude convergence

```json
{
  "0.00625_minus_0.0125": {
    "RMS_difference": 3.40428287340867e-05,
    "mean_absolute_difference": 1.4542002641891501e-05,
    "median_absolute_difference": 1.0287843111855821e-05
  },
  "0.0125_minus_0.025": {
    "RMS_difference": 2.408289804421932e-05,
    "mean_absolute_difference": 8.477577717103853e-06,
    "median_absolute_difference": 5.407902381460161e-06
  },
  "RMS_J_by_epsilon": {
    "0.00625": 0.0004089507240923341,
    "0.0125": 0.00041055806289093025,
    "0.025": 0.0004096705580336831
  },
  "adjacent_RMS_difference_ratio_second_over_first": 1.413568610869824
}
```

## Reconstruction stability

```json
{
  "0.00625": {
    "Q0_RMS": 2.211157648913954e-07,
    "absolute_relative_residual_quantiles": {
      "q50": 0.030446988858442908,
      "q90": 0.13619188157917356,
      "q95": 0.3566759925284935,
      "q99": 1.1896735470063324
    },
    "mean_Q0": 1.8756264438819785e-07,
    "mean_Q_principal": 1.866566639789693e-07,
    "mean_absolute_residual": 9.141498209683283e-09,
    "mean_residual": 9.059804092285247e-10,
    "normalized_MAE_over_mean_abs_Q0": 0.04873837346184537,
    "normalized_RMSE_over_RMS_Q0": 0.1436897413856051,
    "pearson_Q0_Q_principal": 0.9653809979746518,
    "residual_RMSE": 3.177206707352487e-08,
    "sign_agreement": 0.99
  },
  "0.0125": {
    "Q0_RMS": 2.211157648913954e-07,
    "absolute_relative_residual_quantiles": {
      "q50": 0.018923253975181138,
      "q90": 0.10125405690134323,
      "q95": 0.1911547602703345,
      "q99": 0.851280559366119
    },
    "mean_Q0": 1.8756264438819785e-07,
    "mean_Q_principal": 1.883133979399978e-07,
    "mean_absolute_residual": 4.672973812655518e-09,
    "mean_residual": -7.507535517999309e-10,
    "normalized_MAE_over_mean_abs_Q0": 0.024914203080779126,
    "normalized_RMSE_over_RMS_Q0": 0.04963471004306311,
    "pearson_Q0_Q_principal": 0.9960070622805086,
    "residual_RMSE": 1.0975016876334524e-08,
    "sign_agreement": 1.0
  },
  "0.025": {
    "Q0_RMS": 2.211157648913954e-07,
    "absolute_relative_residual_quantiles": {
      "q50": 0.010794005732471159,
      "q90": 0.09156247567780464,
      "q95": 0.14795219050155795,
      "q99": 0.4785292699496198
    },
    "mean_Q0": 1.8756264438819785e-07,
    "mean_Q_principal": 1.877513977695828e-07,
    "mean_absolute_residual": 2.8531396523418933e-09,
    "mean_residual": -1.8875338138496053e-10,
    "normalized_MAE_over_mean_abs_Q0": 0.015211662544257792,
    "normalized_RMSE_over_RMS_Q0": 0.031101116832661122,
    "pearson_Q0_Q_principal": 0.9982775514985435,
    "residual_RMSE": 6.8769472374305165e-09,
    "sign_agreement": 1.0
  }
}
```

## Numerical degeneracy diagnostics

```json
{
  "0.00625": {
    "J_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 0.0017545328437051921,
      "median": 0.00025447982720194773,
      "min": 2.6850162093694507e-07,
      "nonfinite_count": 0,
      "q05": 2.57606841618241e-05,
      "q95": 0.0007870348031693282
    },
    "central_difference_numerator_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 2.1931660546314902e-05,
      "median": 3.1809978400243466e-06,
      "min": 3.3562702617118134e-09,
      "nonfinite_count": 0,
      "q05": 3.220085520228012e-07,
      "q95": 9.837935039616602e-06
    }
  },
  "0.0125": {
    "J_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 0.0016491578312094468,
      "median": 0.00025673223873035056,
      "min": 9.469381723192782e-08,
      "nonfinite_count": 0,
      "q05": 2.4155075187382028e-05,
      "q95": 0.0007920300326934578
    },
    "central_difference_numerator_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 4.122894578023617e-05,
      "median": 6.418305968258764e-06,
      "min": 2.3673454307981956e-09,
      "nonfinite_count": 0,
      "q05": 6.038768796845506e-07,
      "q95": 1.9800750817336448e-05
    }
  },
  "0.025": {
    "J_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 0.0013890838283625584,
      "median": 0.00025782456576084467,
      "min": 1.5027211031082288e-06,
      "nonfinite_count": 0,
      "q05": 2.4373437579217068e-05,
      "q95": 0.0007896590231312172
    },
    "central_difference_numerator_absolute_distribution": {
      "exact_zero_count": 0,
      "max": 6.945419141812792e-05,
      "median": 1.2891228288042234e-05,
      "min": 7.513605515541144e-08,
      "nonfinite_count": 0,
      "q05": 1.2186718789608533e-06,
      "q95": 3.948295115656086e-05
    }
  }
}
```

## Interpretation boundary

Finite-epsilon spectral contribution magnitude is descriptive and must not be interpreted as a new causal-plane ranking.

This result does not establish causal additivity, plane independence, an exact finite-epsilon identity, an optimal epsilon, improved steering, AVeriTeC utility, or a new causal rank discovery.
