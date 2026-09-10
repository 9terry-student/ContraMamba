# Seed8192 A0/A1/A2/A3 Factorial Validated-Evidence Analysis Candidate

## Verdict and strict boundary

```text
VERDICT = PASS
EVIDENCE_VALIDATION = PASS
READY_FOR_SEPARATE_FACTORIAL_INTERPRETATION_AUTHORITY_AUTHORING = YES
READY_FOR_INDEPENDENT_REVERIFICATION = YES
TRAINING_OR_EVALUATION_AUTHORIZED = NO
FACTORIAL_SCIENTIFIC_OR_CAUSAL_CONCLUSION = NONE
```

This is a report-only, descriptive validation of imported evidence. It makes no
promotion, causal, significance, production-readiness, or new-execution claim.
Authority used: commit `3a76c6cd3f6bd8b011317f37938677822ce9191d` and
`reports/reason_router_p3w7_seed8192_a1_a2_a3_factorial_execution_authority_spec_candidate.md`.
The A0 N=3 report at commit `dd183f59f4040405c178da193fe99c7c7f3ef57f` was used only as structural precedent.

## Membership, provenance, and comparability

The factorial root contains exactly `seed180`, `seed181`, and `seed182`, each with exactly `A1`, `A2`, and `A3`; no missing or extra scientific cell was found. Each factorial cell has the five required artifacts. Seed180 A1 alone carries an A0-reference runtime sidecar; seed180 A3 has no such sidecar. The sidecar is not an extra scientific run, and its absence for seed180 A3 does not invalidate that cell because the governing imported provenance and contract evidence otherwise reconcile. Checkpoints were byte-hashed only and were never deserialized or executed.

All twelve cells report 720 exported dev rows. Within each seed, A0/A1/A2/A3 have identical ordered stable IDs, identical gold labels, and no duplicate IDs. Reported numeric losses/metrics and parsed prediction values are finite. The JSON recomputation used `gold_final_label`, `pred_final_label`, `is_correct`, and exported probability/diagnostic fields; recomputed final accuracy, macro-F1, class F1, distributions, and the report's selected-epoch fields reconcile with `best_dev_metrics` (float serialization differences only). Axis and pairwise values are reconciled against their explicit report exports.

Every factorial provenance binds to execution commit `3a76c6cd3f6bd8b011317f37938677822ce9191d`, split seed 8192, 0.2 dev ratio, 2,880 train and 720 dev rows, frozen encoder, Mamba, `v6b_minimal`, the frozen dataset/sidecar identities in the execution authority, no time-swap, and clean-dev checkpoint selection. The exact arm matrix reconciles: A1=`conditional_first_blocker`/`joint`/0.6273209029272248; A2=`explicit_product`/`explicit_local`/0.0; A3=`conditional_first_blocker`/`explicit_local`/0.6273209029272248; A0=`explicit_product`/`joint`/0.0.

A0 binding is same-seed only. The admitted prediction files exactly match: seed180 replacement R1 `3937018` / `80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334`; seed181 `3935282` / `d7a2d79091e2b076610d58b6e3539a347c29706b796c9364b6998c5995b42472`; seed182 `3938383` / `ae1a1dfd9a844437e032a506716abb68544b16ea1fcdff7eaba14a040a48e650`. Historical seed180 r2 is excluded.

## Independently recomputed cell metrics

Values are `epoch | accuracy | macro-F1 | NE/REFUTE/SUPPORT F1 | predicted NE/REFUTE/SUPPORT`.

| Seed | A0 | A1 | A2 | A3 |
|---:|---|---|---|---|
| 180 | 20 \| .909722 \| .804826 \| .942932/1.000000/.471545 \| 599/89/32 | 18 \| .898611 \| .783188 \| .935796/.994413/.419355 \| 597/90/33 | 16 \| .754167 \| .610488 \| .842718/.685714/.303030 \| 490/156/74 | 20 \| .805556 \| .618559 \| .910394/.615385/.329897 \| 576/41/103 |
| 181 | 17 \| .898611 \| .792734 \| .935455/1.000000/.442748 \| 591/89/40 | 19 \| .909722 \| .810307 \| .942731/1.000000/.488189 \| 595/89/36 | 20 \| .804167 \| .686835 \| .884507/.802260/.373737 \| 525/88/107 | 20 \| .838889 \| .635050 \| .932751/.666667/.305732 \| 605/49/66 |
| 182 | 20 \| .901389 \| .813349 \| .936550/1.000000/.503497 \| 579/89/52 | 19 \| .909722 \| .806536 \| .944688/.982857/.492063 \| 599/86/35 | 19 \| .813889 \| .707261 \| .885338/.863388/.373057 \| 524/94/102 | 20 \| .777778 \| .593109 \| .883888/.623188/.272251 \| 571/49/100 |

Frame/predicate/sufficiency/polarity and relevant pairwise diagnostics were independently read from the exported selected-dev evidence and reconciled with `best_dev_metrics`/`best_dev_pairwise_checks`; all pairwise families have 60 active groups. Their full values remain in the immutable imported JSON evidence, rather than being promoted to an interpretation here.

## Matched descriptive factorial contrasts

Values are seed180 / seed181 / seed182; no inferential test was run.

| Contrast | Macro-F1 | Accuracy | NE F1 | REFUTE F1 | SUPPORT F1 |
|---|---:|---:|---:|---:|---:|
| A1−A0 | -.021638 / .017572 / -.006813 | -.011111 / .011111 / .008333 | -.007136 / .007276 / .008138 | -.005587 / .000000 / -.017143 | -.052190 / .045441 / -.011433 |
| A2−A0 | -.194338 / -.105900 / -.106088 | -.155556 / -.094444 / -.087500 | -.100214 / -.050948 / -.051212 | -.314286 / -.197740 / -.136612 | -.168514 / -.069011 / -.130440 |
| A3−A2 | .008071 / -.051785 / -.114152 | .051389 / .034722 / -.036111 | .067676 / .048244 / -.001450 | -.070330 / -.135593 / -.240200 | .026867 / -.068005 / -.100806 |
| A3−A1 | -.164629 / -.175257 / -.213427 | -.093056 / -.070833 / -.131944 | -.025402 / -.009980 / -.060800 | -.379029 / -.333333 / -.359669 | -.089458 / -.182456 / -.219812 |
| interaction | .029709 / -.069357 / -.107339 | .062500 / .023611 / -.044444 | .074812 / .040968 / -.009588 | -.064743 / -.135593 / -.223057 | .079056 / -.113446 / -.089373 |

## N=3 descriptive aggregates

Format is mean / population-SD / sample-SD / min / max / range.

| Measure | Macro-F1 | Accuracy | NE F1 | REFUTE F1 | SUPPORT F1 |
|---|---|---|---|---|---|
| A0 | .803636/.008458/.010359/.792734/.813349/.020615 | .903241/.004721/.005782/.898611/.909722/.011111 | .938313/.003297/.004038/.935455/.942932/.007477 | 1/0/0/1/1/0 | .472596/.024812/.030388/.442748/.503497/.060748 |
| A1 | .800010/.011994/.014690/.783188/.810307/.027119 | .906019/.005238/.006415/.898611/.909722/.011111 | .941072/.003815/.004673/.935796/.944688/.008892 | .992424/.007139/.008743/.982857/1/.017143 | .466536/.033399/.040906/.419355/.492063/.072709 |
| A2 | .668195/.041648/.051009/.610488/.707261/.096773 | .790741/.026165/.032045/.754167/.813889/.059722 | .870855/.019898/.024370/.842718/.885338/.042620 | .783787/.073702/.090266/.685714/.863388/.177674 | .349942/.033172/.040628/.303030/.373737/.070707 |
| A3 | .615573/.017252/.021129/.593109/.635050/.041941 | .807407/.024983/.030598/.777778/.838889/.061111 | .909011/.019972/.024461/.883888/.932751/.048863 | .635080/.022561/.027632/.615385/.666667/.051282 | .302627/.023636/.028948/.272251/.329897/.057646 |
| interaction | -.048996/.057772/.070756/-.107339/.029709/.137048 | .013889/.044198/.054131/-.044444/.062500/.106944 | .035398/.034681/.042475/-.009588/.074812/.084400 | -.141131/.064750/.079302/-.223057/-.064743/.158314 | -.041254/.085638/.104885/-.113446/.079056/.192502 |

N=3 aggregates are descriptive only.

## Required-evidence identity ledger (all 60 consumed files)

For each cell, ordered identities are `bytes / SHA256` for `clean_dev_predictions.json; run_provenance.json; selected_checkpoint.pt; training_report.json; training_report_predictions.jsonl`. This is the complete 60-artifact byte-count-plus-SHA256 ledger; it does not represent any sidecar as a scientific cell or invent a seed180 A3 sidecar.

```text
180 A0 4840320 / 5c9722ac0f75c411b3d744a29beec0c35d2f2809331f257a5e1d5ea81e6cf75d; 69444 / a758538e93e6e52ca261cb593285c298344808a3626eed7d9b9664e29a6c1a3d; 518269943 / 4f7ad019bddb988a534c477b58b36bdabe2775d6c9748331e8311653c07c864c; 306325 / 2cdf0925e3a0ef1b925f6b00ac4b2095d18a113896a437ded77622f5134b2013; 3937018 / 80fef1e7fa1df6b99c797ef61dcc79bd552a65f79126f231dce47d5971ecd334
180 A1 5494788 / febf9fd731407d4d91369d97a426be23a88106ede67eea1af37b96c6fa7599d7; 70101 / 241dd834148531b4ebb6f7ce05b61471a2d47a0d8b9ca2ac554b3dc2790924cf; 522528964 / f9e267baade22856e0319c1ab57784edbaaa4ff4fce32c5640a2d4f02fab48d2; 307395 / adb223ad4d96cf87739e68d38b4b9e52f4c7b612ef21a94fc94fed0f40289c40; 4395646 / 21211e9e5bab21367cb8b50bba21ded25311feb5fb4c6bc714133407e1a4f736
180 A2 4904848 / 1470c576f9c625c03cc0b1f80fe0639378adbd8eae4a68a69c792191f4cbbd81; 70075 / 141f2d95a381f87fd8caf2f05155d3c97aef25766295c7f0909266fc8ae43dd0; 522528631 / cb37bb809ed381b675fd4fd5f6187e7a023a1466b0327f0d796e1c948f5be400; 307941 / bbdb3bb28e81e8ce3c4b2a1a87462d4632311c4b8f42be7ee42f1fe317184596; 4001546 / 86a5afee632c14271b875d2c75ff53cede44f2f3605930e1f50dd9f6ad136b28
180 A3 5496195 / 9c54d41a766b22c854eec6c6180b93bcaa7f7f65b401a34d325bb928063c2d1a; 70188 / 3554b62c0b7d5cb7cf18ffb80d8277cd6a48ae329b351f63156ee4079cca3f23; 522528964 / 688b52c25a4e137bfa65044a33a1f244b6aa5a961aa8c37650c7e26d1836cc93; 306886 / c4589f995b00cb501464baec0834370e62ddcd4e6971c3f518ee13792eece525; 4397053 / 7fa20c9b4cd15ee3893c8e5298212c0ef9c5a6193de9536039df607be6126051
181 A0 4838584 / 789d02f9092ce6b051d0ca435272c9e93a3962183dbb0a4d4dbb20cebf2ac3fe; 69133 / 82f6a511f9b8228c91d3419cc8872f7371c0785a01aa730904d43a8f04f6a98b; 518269879 / e9af8ca6e5c1ed62b8ec3014401d79ca0f8ab869a88db4ee7f18685a46f78e00; 306101 / 0068aec52a9afb4bd8e79d711ac666a5257186ab142c8c72da890fe64c2c45e8; 3935282 / d7a2d79091e2b076610d58b6e3539a347c29706b796c9364b6998c5995b42472
181 A1 5476359 / 69e8ba5915be14f8861e97d0e8cbaffb51c5eca80667ffe4662c9a357850ad64; 69978 / cb46f048e9385ec101636bb708804cfa9b5b477fb1a6e954b2572e843eba31cc; 522528708 / cd1109494108807fababfd7b4920cb2cf9557da5fb4be30d5135ad80d40df6bc; 307484 / 660dbc65dd7c353da809216838d1e42b387b8fba11e885d7f730fb2bbf013931; 4377217 / a0572bb55d154ffa045e245f03455ebe5cf64883c920e51fa74487a1f383506f
181 A2 4890154 / cc8acad244fb825d95382e9423ffae812eac9766488ecb50df4ba06ffeaa2dc9; 69965 / 9a4287acd58ae1f26cb957782a6274ece39051c4af061c41a9aeea861f020718; 522528375 / 290466fe3d2e91b8a9f841ef7a24655e9d417e07dfdc4c5c57fcfe962c6fc9bb; 307889 / cfd58527b3bbed7a38ff07f9709541c110c5e0c05d295bbee476027e09c54724; 3986852 / 248632ebff668d543eae30af349278d8d6a9cb2f7335299d5484731151e5c75c
181 A3 5482896 / 9a7ff8668e27894fdca09ff66032ba7340c3191a257c0ba533d2b35b1ea91aa0; 70085 / a9cdaaa986a73f143f8ab37fc1dc956df5cbce53048305bf10e6d0e15befaa06; 522528708 / ffe01dc2b82909e136112390753908d5d31cef099eea3f29a5d3fe62a13619e9; 306886 / b2b17a01be78783a486f8272953aa0eb7e570733f8e5a4b0065add74fda0022d; 4383754 / eb95e92a18ca3d137ff4c22da7f10eb40ebc29e1a67eef41bf6a65d168421a79
182 A0 4841685 / 029ec6ae31df2f5ca9526d1e631496f7ee272967a6f5e08684f29aa09ad490d4; 69143 / 934acab332773b4127ffa5c68b09a8bded168ab96ac5b18c29018e3e78b77c66; 518269879 / a8c1b03095ec75d5e28eca9d8dc1dd83af309c93828ff4f12453fcdf36c61b63; 306108 / ef60c457a28be8ed91e57ef8d9be4d89150b0e93fe90a045eb12a6ea78262a22; 3938383 / ae1a1dfd9a844437e032a506716abb68544b16ea1fcdff7eaba14a040a48e650
182 A1 5479534 / 7348a75bb575255e0eec31a343daaa9bb9be7257f39d46806031d37d4ca72a8b; 70024 / 51a65c651fb52ffb219a58eda459f5dceeb03021a18ecbca12c3f1afaca820c6; 522528324 / 2a1a92512aa6619a431389da3690317cb05cab817684ba7b7d030f2cec9b9188; 307447 / 570f1ab6b64b4ea0c1d6ca1958dda58f6b3a8fa52b0fc672b220743807a5985c; 4380392 / e17e1ec15277fa2beab5a6aa82d54fb7cb68db9cbc9e7a137356da2bea315aca
182 A2 4891241 / 0fb3001a40419787b6ea2a315e8a0c1c90d4373ee5d7cac8ff340e8db55667ed; 69964 / 1f06e36c3244555097ca40facdc3c4ec244604448b9b8ecbf4353a97b47aa05c; 522527991 / a9ca5fff240cd573f618d31a590c50439a6229c9e5efda068d8438695196d8de; 307839 / 9d577e71920d6ecfff19070ddd2a8ed67dcb0a491992709e19756eeb05685e28; 3987939 / 22765b9e1f7624dd16e4e02a2829716f2b60106d07599d272e71ef6e6eb0a36a
182 A3 5479742 / eba1d5523fee714ccd0ef9a09e9c445c9d078f797ff8b88d142c6cbf14dadd1a; 70071 / 284005cb9506be7d9a13620ac00e7ed40fd31ba103c02b68656b4dbedbe22ca5; 522528324 / 07f9fa523f152da56ce0ba2391af626f5a318bb7d0be8dd81fe236ad214d3336; 307391 / ce967912508328ea0672b8d07462b71db6b590a857e976bee372f837310f2dc9; 4380600 / 4130f7f3c890ff340fd9626d798612c34bfb446e3df1e0122b9cf7295ef19bab
```

## Seed180 A3 retry-name caveat and remaining boundary

```text
SEED180_A3_RETRY1_REPO_EVIDENCE_VERIFICATION = NOT_ESTABLISHED
SEED180_A3_RETRY1_CONTROLLER_HISTORY = RECORDED_BUT_NOT_INDEPENDENTLY_VERIFIED_HERE
```

The imported seed180/A3 artifact set is scientifically and provenance-valid under the validated five imported artifacts and the execution contract. Controller-supplied research-session history records that the valid execution used a run name ending in `-retry1`, while an earlier attempt blocked before scientific execution. That retry-name distinction is not independently recoverable from the currently permitted imported seed180/A3 artifacts or the active execution-authority file. Accordingly, this report does not use the retry suffix as a premise for any metric, factorial contrast, evidence membership, or scientific conclusion, and it infers no tenth scientific run from current repository evidence.

No blocker remains for authoring a separate factorial interpretation authority. That later authority, not this report, must decide whether any descriptive pattern warrants a scientific interpretation.
