# ContraMamba K0-RVG Frozen-Strong Alignment Sign / Co-Contribution
## Static Design Provenance Correction Amendment Candidate

## 1. Status

**Phase:** provenance-only correction amendment to the frozen static design.

**Original static design freeze:**

`fe73f67e84176f5b7d468287d8084a8836e9a240`

**Original static design file:**

`reports/longterm_k0_rvg_frozen_strong_alignment_sign_contribution_static_design_candidate.md`

**Immediate parent evidence freeze:**

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

This amendment corrects exactly one malformed SHA256 literal in the frozen static design.

It does not change:

- the scientific question;
- the population;
- the frozen strong-240 partition;
- the parent evidence commit;
- the decomposition;
- the outcome classes;
- any numerical tolerance;
- any execution boundary;
- any artifact schema;
- any scientific interpretation.

No model execution, tokenizer work, training, evaluation, intervention, learned geometry, PCA/SVD, channel search, item search, or K1 work is authorized.

---

## 2. Defect

The original frozen static design recorded the parent execution-manifest SHA256 as:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fe`

That string is malformed because it contains only **63 hexadecimal characters**.

A SHA256 digest must contain exactly 64 hexadecimal characters.

The defect is therefore a provenance-literal transcription error.

---

## 3. Executed evidence resolving the defect

At repository HEAD:

`fe73f67e84176f5b7d468287d8084a8836e9a240`

the frozen parent artifacts were read directly from Git object bytes with:

`git show 431e8faa6e5c82a20d87f532b4ab960fcf641ec2:<path>`

and hashed in Python using:

`hashlib.sha256(raw).hexdigest()`.

The resulting frozen Git-byte identities were:

### Parent item metrics

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_item_metrics.jsonl`

SHA256:

`e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`

Bytes:

`683435`

### Parent strong-channel validation

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/layer20_y20_strong_interaction_geometry_channel_validation.jsonl`

SHA256:

`111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`

Bytes:

`143535`

### Parent summary

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/summary.json`

SHA256:

`35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`

Bytes:

`6428`

### Parent execution manifest

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_0b11681_v1/execution_manifest.json`

Correct SHA256:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`

Bytes:

`4297`

### Parent validated evidence report

Path:

`reports/longterm_k0_rvg_layer20_y20_strong_interaction_geometry_validated_evidence_analysis_report_candidate.md`

SHA256:

`6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163`

Bytes:

`13632`

All identities except the manifest already matched the original design exactly.

---

## 4. Exact correction

Replace only the malformed parent execution-manifest SHA literal:

### Incorrect frozen literal

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fe`

### Correct frozen Git-byte SHA256

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`

The correction appends the missing final hexadecimal digit:

`e`

No other provenance identity changes.

---

## 5. Corrected parent artifact identity set

The authoritative parent identities for the static analyzer are therefore:

- item:
  `e7002e03bd170c05ea068e70eb32cc1f7a8b0d4f569ac44e531a42a8a4e1ebfe`;
- channel:
  `111bf87720c5755ef974c4a98a8a12e62c6728952c72b0a76e436e385e6e9942`;
- summary:
  `35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`;
- execution manifest:
  `1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`;
- validated report:
  `6e33951e0fe02ba6e82da6f59f786034ddbda869d2f0a2f592b9a51467477163`.

The immediate parent evidence freeze remains:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`.

---

## 6. Scientific design remains unchanged

The primary exact role-level decomposition remains:

`I_b = P_b + N_b`.

The paired decomposition remains:

`ΔI = ΔP + ΔN`.

Definitions remain:

`ΔP = P_corr - P_ctrl`

and:

`ΔN = N_corr - N_ctrl`.

A positive `ΔN` continues to mean reduced negative cancellation.

No mathematical definition is changed by this amendment.

---

## 7. Population remains unchanged

The frozen population remains:

- common DDSSSSS cohort:
  `330`;
- current token only;
- relative coordinate:
  `k=2`;
- frozen strong partition:
  `240` channels;
- source block:
  `20`;
- target residual layer:
  `21`;
- downstream parent map:
  layer `22`.

No subset is added or removed.

---

## 8. Parent numerical targets remain unchanged

The parent strong interaction population mean remains:

`+0.760594723140676`.

The parent role-level aggregate targets remain:

### Positive interaction mass

corr:

`2.267583155684786`

ctrl:

`1.7361014562993915`

### Negative interaction mass

corr:

`-0.5449066994977081`

ctrl:

`-0.7740197232529892`

### Same-sign channel count mean

corr:

`140.5818181818182`

ctrl:

`136.03030303030303`

### Opposite-sign channel count mean

corr:

`99.41818181818182`

ctrl:

`103.96969696969697`

No scientific number changes.

---

## 9. Outcome classes remain unchanged

The preregistered outcome classes remain:

### Outcome A

positive co-contribution dominant.

### Outcome B

cancellation-relief dominant.

### Outcome C

mixed.

No threshold is added or changed.

---

## 10. Tolerances remain unchanged

Use:

- parent scalar relative tolerance:
  `1e-13`;
- parent scalar absolute tolerance:
  `1e-13`;
- identity absolute tolerance:
  `5e-12`;
- channel bridge absolute tolerance:
  `5e-12`.

No tolerance weakening is authorized.

---

## 11. Static-only boundary remains unchanged

The corrected analyzer must remain model-free.

It must not:

- import model code;
- load checkpoint weights;
- open the handoff ZIP;
- import Transformers;
- invoke a tokenizer;
- execute model forwards;
- read logits;
- execute task heads;
- train;
- intervene;
- use PCA/SVD;
- fit a learned projection;
- conduct post-hoc channel or item search.

---

## 12. Analyzer authority after amendment freeze

After this amendment is frozen, the static analyzer must authenticate:

1. the original scientific static design freeze:
   `fe73f67e84176f5b7d468287d8084a8836e9a240`;
2. this correction-amendment freeze;
3. the parent evidence freeze:
   `431e8faa6e5c82a20d87f532b4ab960fcf641ec2`;
4. the corrected 64-character manifest SHA256:
   `1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`.

The original design remains historical authority for all scientific semantics.

This amendment overrides only the malformed manifest SHA literal.

---

## 13. Failure-recovery interpretation

The preflight blocker:

`PARENT_MANIFEST_SHA256_MISMATCH`

is classified as:

**provenance-literal transcription defect in the new static design/analyzer**

and not as:

- parent artifact drift;
- scientific evidence corruption;
- runtime nondeterminism;
- model change;
- line-ending mutation of the frozen Git object;
- decomposition failure.

The four other frozen artifact identities matched exactly.

---

## 14. No need to reopen parent evidence

The parent evidence freeze:

`431e8faa6e5c82a20d87f532b4ab960fcf641ec2`

remains valid.

Its manifest content is not being modified.

The correction concerns only the SHA literal copied into the new static-design authority.

Therefore parent evidence does not need to be rerun, revalidated, or recommitted.

---

## 15. Required implementation correction

The analyzer implementation must update exactly:

`PARENT_MANIFEST_SHA256`

from the malformed 63-character value to:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`.

It must additionally authenticate this correction amendment after its freeze.

No scientific calculation should otherwise change.

---

## 16. Validation after correction

After the correction-amendment freeze and analyzer patch:

1. `py_compile` must pass;
2. static preflight must authenticate all five parent artifact SHA256 values;
3. 330 frozen item rows must validate;
4. 240 frozen strong-channel rows must validate;
5. exact sign decomposition must close;
6. no scientific output should be persisted during preflight;
7. model forward count must remain zero.

Only then may the corrected analyzer be implementation-frozen.

---

## 17. Amendment conclusion

The frozen scientific design at:

`fe73f67e84176f5b7d468287d8084a8836e9a240`

remains scientifically valid.

Exactly one provenance literal is corrected:

**parent execution-manifest SHA256**

from malformed 63-character:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fe`

to exact frozen Git-byte SHA256:

`1cf47b4d3b7c1a4a4ea6dbc212721fca51d7370273acdafe4c78e98d3cd62fee`.

No scientific semantics, data, population, equation, outcome class, tolerance, or execution authorization changes.
