# O0a — Native Mamba Entitlement-Sensitive State Dynamics Screening

Status: **PRE-EXECUTION CANDIDATE / NOT YET EXECUTION AUTHORITY**

This document is an implementation-review candidate. It does not authorize a
model download, data forward, scientific execution, Kaggle run, result claim,
promotion decision, training run, or generation run.

## 1. Authority and provenance boundary

The implementation preparation began from repository base commit:

```text
56bf9e7dca92d1d7e61ab153038a68aeb21c4017
```

That commit is the pre-code base identity only. It is **not** the commit that a
future O0a execution may use. Before O0a may execute, an independent verifier
must inspect the implementation and tests, require any repairs, and establish a
later implementation-freeze commit. A later execution authority must bind the
exact full 40-character SHA of that implementation-freeze commit and the exact
script SHA256 at that commit.

Candidate implementation:

```text
scripts/observe_longterm_o0a_native_mamba_state_dynamics.py
candidate-repair SHA256: 760841ba6b4e71732c511142ac1003fb03205b2fe7789936946514a791d9f9f6
```

This hash reflects the repaired candidate observer bytes at pre-execution
repair time. It is not yet a future execution-bound hash. A later freeze
authority must bind the exact committed script bytes and full implementation
freeze commit after independent verification.

The candidate script requires both `--authority-repository-head` and
`--authority-dataset-sha256`. It checks both before loading the model. The
repository-HEAD argument must eventually be the later freeze commit, not the
pre-code base above.

The canonical hypothesis context is:

```text
docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md
O0 status there: FUTURE EXPERIMENT CONCEPT / NOT AUTHORIZED
```

The current instruction authorizes implementation and this candidate only. No
scientific model/data execution occurred under this preparation authority.

## 2. Scientific question and claim boundary

Question: do layer hidden-state proxy trajectories of frozen native Mamba
exhibit early entitlement-sensitive separation under controlled interventions?

O0a measures Hugging Face `MambaModel` layer hidden states. These are **native
pretrained Mamba hidden-state proxies**. They are **not** claimed to be the
selective SSM recurrent state itself, internal selective-scan state matrices,
or direct A/B/C/Delta dynamics. Hugging Face `cache_params` and deeper
instrumentation of actual selective-SSM recurrent state are **out of scope for
O0a** and may become O0b/O1 only under separate authority.

O0a is not a hallucination-detector experiment. It does not observe an actual
unauthorized generated commitment. It screens a prerequisite: whether native
backbone dynamics contain early risk-relevant semantic separability before a
specialized ContraMamba head exists.

**Execution success != scientific evidence of a hallucination precursor.** A
successful process and complete artifacts establish only that the observer ran
under its contract. Scientific interpretation requires separately authorized
review of the descriptive results.

No hard PASS threshold is defined. No layer may be selected using final labels
and then presented as confirmatory evidence. A negative or surface-form-
dominated result provides little support for a useful early precursor in this
formulation, but does not falsify the overall ContraMamba program.

## 3. Model and tokenizer binding

Exact model repository:

```text
state-spaces/mamba-130m-hf
```

Exact model revision:

```text
5708daa364c50b880e7bd92eab456e0d34492ee9
```

Exact tokenizer repository:

```text
state-spaces/mamba-130m-hf
```

Exact tokenizer revision:

```text
5708daa364c50b880e7bd92eab456e0d34492ee9
```

The only permitted loading path is:

```python
from transformers import AutoTokenizer, MambaModel

tokenizer = AutoTokenizer.from_pretrained(
    "state-spaces/mamba-130m-hf",
    revision="5708daa364c50b880e7bd92eab456e0d34492ee9",
)
model = MambaModel.from_pretrained(
    "state-spaces/mamba-130m-hf",
    revision="5708daa364c50b880e7bd92eab456e0d34492ee9",
    torch_dtype=authorized_dtype,
)
```

Mutable `main` loading is not permitted. O0a has no silent CLI model or
revision override. If such an override is introduced later, it must fail closed
unless it exactly equals the authority-bound constants above.

The first O0a execution candidate is bound to:

```text
device = CPU
dtype = float32
```

The script must fail closed before model loading if another device or dtype is
requested. `torch.compile` is not permitted. The model is moved to CPU, placed
in `eval()` mode, and frozen with `requires_grad_(False)`. Forward calls occur
only inside `torch.inference_mode()` with `output_hidden_states=True`,
`return_dict=True`, and `use_cache=False`.

Forbidden paths include optimizers, backward, parameter-value mutation,
fine-tuning, checkpoint writing, generation, language-generation heads,
downstream semantic heads, `ContraMambaV6BMinimal`, and every reason-router
component.

## 4. Dataset binding

Exact repository-relative path:

```text
data/toy_interventions_v5.jsonl
```

Exact SHA256 of the dataset bytes inspected during preparation:

```text
17e07f4ca9d3a11511c67493b7b9e9ff9ef6b1ebdca4c9d779705ad72e7c04dc
```

Observed preparation-time row count: `18`.

Observed preparation-time unique `pair_id` family count: `3`.

Required intervention families are exactly required to include:

- `none`: entitlement-preserving native reference;
- `paraphrase`: whole-pair semantic-invariance / surface-form control;
- `entity_swap`: Frame-related non-entitlement intervention;
- `predicate_swap`: Predicate-related non-entitlement intervention;
- `evidence_deletion`: Sufficiency-related non-entitlement intervention;
- `polarity_flip`: authorized REFUTE semantic-sensitivity control.

`polarity_flip` must not be described as a hallucination or non-entitlement
case. Dataset IDs must be unique. Each `pair_id` must have exactly one `none`
reference, and a pair may not contain duplicate intervention types.

The toy `paraphrase` rows may alter both claim wording and evidence wording.
They are not pure evidence-only paraphrase controls. The paired ranking
`distance(intervention, none) > distance(paraphrase, none)` remains a
descriptive screening diagnostic, but it must not be described as perfectly
isolating evidence semantics from claim surface form. At the 0% evidence prefix,
paraphrase distance can legitimately be nonzero because the claim itself may be
paraphrased.

## 5. Serialization and token-prefix contract

The exact recorded template is:

```text
Claim: <claim>
Evidence: <evidence-prefix>
```

Tokenization uses `add_special_tokens=False`. For each row, the implementation
tokenizes both:

```text
Claim: <claim>
Evidence:
```

and the full serialization containing one literal space after `Evidence:`. The
fixed marker token IDs must be an exact prefix of the full token IDs. The
remaining suffix is the evidence region, including whatever leading-space
ownership the matching tokenizer assigns. Any prefix relationship failure is a
serialization error.

Requested schedule:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

`0.00` uses the complete claim and `Evidence:` marker with zero evidence suffix
tokens. For nonzero fraction `f` and evidence suffix length `N`, the token count
is `ceil(f * N)`, capped at `N`. Repeated counts for short evidence are executed
once; every requested fraction alias remains in observation metadata and is
expanded again for paired comparisons and summaries.

## 6. Native observations and compact state artifact

For every row and unique evidence-prefix token count, the observer requests all
layer hidden states exposed by the native Transformers API. It retains each
`hidden_states[i]` and also retains `last_hidden_state` if it is not already
identical to the last exposed hidden state. Roles identify the initial/
embedding-level, intermediate, and output-level states without asserting more
than the API exposes.

Each `observations.jsonl` record contains:

- row ID, pair ID, intervention type, primary failure type, and final label;
- primary and complete requested-fraction metadata;
- actual and full evidence token counts plus actual fraction;
- exact serialization template, serialized input text and its UTF-8 SHA256;
- exact input token IDs and terminal-vector NPZ index;
- per-layer identity and terminal hidden norm;
- last-step transition magnitude;
- terminal consecutive-state cosine;
- terminal acceleration when at least three tokens are present;
- evidence-region mean and maximum consecutive-state delta where evidence is
  present.

`terminal_hidden_states.npz` stores float32 terminal vectors with shape
`[unique_row_prefix_records, exposed_layers, hidden_size]` plus row, pair,
intervention, actual-token-count, and requested-fraction index metadata. It does
not dump token-by-layer trajectories.

Every emitted numerical value must be finite. Unavailable conditional metrics
are JSON `null`, not NaN.

## 7. Paired comparison contract

For each non-`none` row, requested fraction, and exposed layer, the native
reference is the unique `none` row with the same `pair_id`. Row-specific actual
token counts may differ because pairing is by requested fraction.

The four descriptive comparisons are:

1. `D_l2`: Euclidean distance between unit-normalized terminal vectors. Two
   zero vectors have distance zero; one zero vector is represented by the zero
   unit vector.
2. `D_cos`: `1 - safe_cosine_similarity(terminal, reference)`. Two zero vectors
   have cosine one; exactly one zero vector has cosine zero.
3. Transition-magnitude difference: absolute difference between the two
   last-step delta norms.
4. Trajectory-summary difference: Euclidean distance between jointly available
   scalar summaries: terminal norm, last-step delta, terminal cosine,
   acceleration, evidence-region mean delta, and evidence-region maximum delta.

For unit-normalized nonzero vectors:

```text
D_l2^2 = 2 * D_cos
```

Therefore terminal normalized-L2 distance and terminal cosine distance are
algebraically redundant coordinates. They may both remain in artifacts for
auditing and convenience, but agreement between them must not be counted as two
independent pieces of scientific evidence or described as independent
corroboration.

The trajectory-summary difference is retained only as an unweighted descriptive
convenience diagnostic over heterogeneous scalar summaries. Its component
scales are not made comparable by O0a, so the composite has no independent
inferential weight and is not a primary scientific score. Screening conclusions
must rely on the atomic component metrics and the explicitly descriptive paired
diagnostic, not on the composite as a tuned or normalized score.

The summary groups observations and paired distances by intervention type,
requested fraction, and layer. For every non-`none` intervention it also counts
pair IDs satisfying:

```text
D_l2(intervention, none) > D_l2(paraphrase, none)
```

This is a descriptive ranking diagnostic. It is not a classifier, threshold,
layer selector, or promotion rule.

## 8. Critical 0% sanity invariant

For `entity_swap`, `predicate_swap`, `evidence_deletion`, and `polarity_flip`
rows whose claim text exactly matches their `none` reference, the 0% serialized
UTF-8 bytes, token IDs, terminal-state shapes, and terminal states at every
exposed layer must match the reference. State equality uses zero relative
tolerance and an explicitly recorded deterministic absolute tolerance (default
`1e-6`). Any failure invalidates the run as an implementation/serialization
error.

No 0% equality is required for `paraphrase`, because its claim may differ. This
does not weaken the 0% identity invariant for `entity_swap`, `predicate_swap`,
`evidence_deletion`, or `polarity_flip` when their claim text is identical to
`none`.

## 9. Required future-run artifacts

The output directory must be supplied explicitly and must not already exist.
The exact required artifact set is:

```text
manifest.json
observations.jsonl
terminal_hidden_states.npz
paired_distances.jsonl
summary.json
report.md
SHA256SUMS.txt
```

The manifest records repository HEAD, authority HEAD, script path and SHA256,
dataset path and SHA256, exact model ID, exact model revision, exact tokenizer
ID, exact tokenizer revision, Transformers and Torch versions, CPU device,
float32 dtype, relevant exposed Mamba config flags, prefix schedule and
rounding/deduplication rules, serialization and tokenization settings,
row/unique-pair/observation/distance counts, exposed layer descriptors,
deterministic settings and equality tolerance, execution timestamp, artifact
set, scientific claim boundary, and explicit absence of training, generation,
Kaggle, and URP relationships.

If a resolved Hugging Face commit hash is exposed at future runtime, the
manifest may record it too. It must not rely on a private/internal field as the
sole provenance mechanism; the exact model/tokenizer IDs and immutable
revisions above are the authority binding.

`SHA256SUMS.txt` covers every other required artifact.

## 10. Fail-closed conditions

A future run must fail before model loading on an existing output directory,
repository-freeze mismatch, dataset hash mismatch, invalid/missing dataset, a
missing required family, duplicate row ID, missing/multiple `none` reference,
pair/intervention collision, non-CPU device, non-float32 dtype, or any mutable
model/tokenizer revision attempt. It must also fail on token-prefix violations,
output-key collisions, changing API layer layouts, non-finite tensors or
metrics, 0% invariant failure, prohibited optimizer/backward/generation/compile
calls, or an incomplete artifact set.

## 11. URP and execution-environment separation

O0a has no authority or provenance relationship to the URP/reason-router track.
It must not use, modify, consume, or interpret URP A0–A3 authority, URP attempts,
reason-router checkpoints, URP Kaggle artifacts, or URP scientific conclusions.
No URP checkpoint is needed.

Kaggle is not required or authorized by this candidate. A later authority may
choose an execution environment without changing the scientific boundary, but
that choice must be explicit.

## 12. Promotion blockers

O0a execution remains blocked until all of the following occur:

1. independent inspection of the exact three-file implementation delta;
2. successful authorized synthetic/unit/static validation;
3. repair and re-verification of any identified defect;
4. creation of an exact implementation-freeze commit without rewriting the
   dataset bytes;
5. a later execution-authority document binding that full freeze commit, exact
   script SHA256, exact dataset SHA256, model/tokenizer identities, and run
   contract;
6. explicit authorization for the scientific data/model forward.

Until then this document remains a PRE-EXECUTION CANDIDATE and no O0a output may
be described as scientific evidence.

## 13. Small-N and claim boundary

The bound toy dataset has `18` rows across `3` unique `pair_id` families. O0a is
a mechanistic screening experiment only. This candidate does not authorize, and
future toy-screening output alone must not produce, a population estimate,
significance claim, generalization claim, hallucination-prediction claim, or
claim that execution success is hallucination-precursor evidence.
