# Gen5 v10 two-locus E3 paired-gap diagnostic (static)

Run: `gen5-v10-two-locus-7f74a84-r1`  
Commit: `7f74a84ed19e49727e3adf69c016af9e4339fb86`  
Scope: two B32 K4 samples; 8 paired components. No GPU, model, or tokenizer execution.

## Frozen full-record and provisional E2

Both samples use `FAST=0, GEMM=96`. The prior independently imported auditor reported zero E2-tolerance payload failures, and 8 relative-reconstruction diagnostic divergences. `LEGACY_FULL_RECORD=FAIL` remains unchanged.

## Matched-gap E3 localization

| Locus | k | Frozen rel-max | v10 rel-max | Frozen worst gap | v10 worst gap | max coarse engine delta | max fine-sum engine delta |
|---|---:|---:|---:|---|---|---:|---:|
| gpu0 | 0 | 41.358475 | 1 | EVIDENCE->EVIDENCE gap 66 | CLAIM->CLAIM gap 39 | 0 | 3.3629e-08 |
| gpu0 | 1 | 103.90377 | 1 | CLAIM->EVIDENCE gap 103 | CLAIM->CLAIM gap 39 | 0 | 4.0513e-08 |
| gpu0 | 2 | 2.82811 | 1 | EVIDENCE->EVIDENCE gap 47 | CLAIM->CLAIM gap 39 | 0 | 2.77e-09 |
| gpu0 | 3 | 22.96441 | 1 | EVIDENCE->EVIDENCE gap 47 | CLAIM->CLAIM gap 39 | 0 | 8.4862e-09 |
| gpu1 | 0 | 95.8081 | 1 | EVIDENCE->EVIDENCE gap 82 | CLAIM->CLAIM gap 35 | 0 | 2.8154e-08 |
| gpu1 | 1 | 99.342991 | 1 | EVIDENCE->EVIDENCE gap 82 | CLAIM->CLAIM gap 35 | 0 | 2.8735e-08 |
| gpu1 | 2 | 6.3058707 | 1 | EVIDENCE->EVIDENCE gap 47 | CLAIM->CLAIM gap 35 | 0 | 3.1525e-09 |
| gpu1 | 3 | 4.1035936 | 1 | EVIDENCE->EVIDENCE gap 125 | CLAIM->CLAIM gap 35 | 0 | 3.8662e-09 |

## Paired-gap details

### gpu0 / k=0
- frozen worst at EVIDENCE->EVIDENCE gap 66: own coarse=-9.27296e-13, own fine sum=3.742425e-11, own |residual|=3.835155e-11, own relative=41.35847; other engine at same parent/gap: coarse=-9.27296e-13, fine sum=0, |residual|=9.27296e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 39: own coarse=1.853643e-10, own fine sum=0, own |residual|=1.853643e-10, own relative=1; other engine at same parent/gap: coarse=1.853643e-10, fine sum=1.823367e-10, |residual|=3.027612e-12, relative=0.01633331
- Max engine delta: coarse 0 at None gap None; fine sum 3.362889e-08 at EVIDENCE->EVIDENCE gap 11

### gpu0 / k=1
- frozen worst at CLAIM->EVIDENCE gap 103: own coarse=-3.365145e-14, own fine sum=-3.530164e-12, own |residual|=3.496513e-12, own relative=103.9038; other engine at same parent/gap: coarse=-3.365145e-14, fine sum=0, |residual|=3.365145e-14, relative=1
- v10 worst at CLAIM->CLAIM gap 39: own coarse=5.634208e-11, own fine sum=0, own |residual|=5.634208e-11, own relative=1; other engine at same parent/gap: coarse=5.634208e-11, fine sum=5.206117e-11, |residual|=4.280914e-12, relative=0.07598075
- Max engine delta: coarse 0 at None gap None; fine sum 4.0513e-08 at EVIDENCE->EVIDENCE gap 31

### gpu0 / k=2
- frozen worst at EVIDENCE->EVIDENCE gap 47: own coarse=-1.949452e-12, own fine sum=3.563812e-12, own |residual|=5.513264e-12, own relative=2.82811; other engine at same parent/gap: coarse=-1.949452e-12, fine sum=0, |residual|=1.949452e-12, relative=1
- v10 worst at CLAIM->CLAIM gap 39: own coarse=-1.350484e-10, own fine sum=0, own |residual|=1.350484e-10, own relative=1; other engine at same parent/gap: coarse=-1.350484e-10, fine sum=-1.360713e-10, |residual|=1.022915e-12, relative=0.007574429
- Max engine delta: coarse 0 at None gap None; fine sum 2.769997e-09 at EVIDENCE->EVIDENCE gap 3

### gpu0 / k=3
- frozen worst at EVIDENCE->EVIDENCE gap 47: own coarse=8.401697e-13, own fine sum=2.013417e-11, own |residual|=1.9294e-11, own relative=22.96441; other engine at same parent/gap: coarse=8.401697e-13, fine sum=0, |residual|=8.401697e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 39: own coarse=-3.102097e-10, own fine sum=0, own |residual|=3.102097e-10, own relative=1; other engine at same parent/gap: coarse=-3.102097e-10, fine sum=-3.154025e-10, |residual|=5.192786e-12, relative=0.0167396
- Max engine delta: coarse 0 at None gap None; fine sum 8.486224e-09 at CLAIM->CLAIM gap 31

### gpu1 / k=0
- frozen worst at EVIDENCE->EVIDENCE gap 82: own coarse=4.850293e-13, own fine sum=4.695477e-11, own |residual|=4.646974e-11, own relative=95.8081; other engine at same parent/gap: coarse=4.850293e-13, fine sum=0, |residual|=4.850293e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 35: own coarse=8.036421e-11, own fine sum=0, own |residual|=8.036421e-11, own relative=1; other engine at same parent/gap: coarse=8.036421e-11, fine sum=9.139693e-11, |residual|=1.103272e-11, relative=0.137284
- Max engine delta: coarse 0 at None gap None; fine sum 2.815419e-08 at CLAIM->CLAIM gap 31

### gpu1 / k=1
- frozen worst at EVIDENCE->EVIDENCE gap 82: own coarse=9.133382e-13, own fine sum=9.164709e-11, own |residual|=9.073375e-11, own relative=99.34299; other engine at same parent/gap: coarse=9.133382e-13, fine sum=0, |residual|=9.133382e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 35: own coarse=7.743604e-11, own fine sum=0, own |residual|=7.743604e-11, own relative=1; other engine at same parent/gap: coarse=7.743604e-11, fine sum=8.283683e-11, |residual|=5.400786e-12, relative=0.06974511
- Max engine delta: coarse 0 at None gap None; fine sum 2.873546e-08 at CLAIM->CLAIM gap 31

### gpu1 / k=2
- frozen worst at EVIDENCE->EVIDENCE gap 47: own coarse=-3.455497e-13, own fine sum=1.833442e-12, own |residual|=2.178991e-12, own relative=6.305871; other engine at same parent/gap: coarse=-3.455497e-13, fine sum=0, |residual|=3.455497e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 35: own coarse=-7.95053e-11, own fine sum=0, own |residual|=7.95053e-11, own relative=1; other engine at same parent/gap: coarse=-7.95053e-11, fine sum=-8.057703e-11, |residual|=1.07173e-12, relative=0.01347999
- Max engine delta: coarse 0 at None gap None; fine sum 3.152475e-09 at EVIDENCE->EVIDENCE gap 3

### gpu1 / k=3
- frozen worst at EVIDENCE->EVIDENCE gap 125: own coarse=5.717947e-13, own fine sum=-1.774618e-12, own |residual|=2.346413e-12, own relative=4.103594; other engine at same parent/gap: coarse=5.717947e-13, fine sum=0, |residual|=5.717947e-13, relative=1
- v10 worst at CLAIM->CLAIM gap 35: own coarse=-8.104567e-11, own fine sum=0, own |residual|=8.104567e-11, own relative=1; other engine at same parent/gap: coarse=-8.104567e-11, fine sum=-8.645031e-11, |residual|=5.404641e-12, relative=0.06668636
- Max engine delta: coarse 0 at None gap None; fine sum 3.866169e-09 at CLAIM->CLAIM gap 3

## Scientific boundary

- The original absolute-OR-relative `2e-5` replay/field criterion is retained and not recalibrated.
- Near-zero coarse denominators can amplify relative maxima. This is a supported *numerical mechanism hypothesis*, not proof of pure floating-point accumulation-order causation.
- Fine sums equal to zero do not establish zero combinatorial opportunity; opportunity counts are not in these records.
- E0/E1 third-party authentication and E3 NONCOMPARABLE status are not established by this static report.
- No mixed fast+GEMM case or full 840-row equivalence is established.

`LEGACY_FULL_RECORD=FAIL`  
`V10_E2_BOUNDED_NUMERIC=PASS_PROVISIONAL`  
`V10_E3_BOUNDED_RAW=DIVERGENCE`  
`FORMAL_E0_E3=NOT_PROMOTED`
