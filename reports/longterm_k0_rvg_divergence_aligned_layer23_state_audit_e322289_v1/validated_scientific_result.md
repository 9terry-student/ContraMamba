# K0-RVG Divergence-Aligned Layer-23 State Audit

## Status

VALIDATED SCIENTIFIC EVIDENCE

This report freezes the interpretation of the divergence-aligned exact-state
audit executed from commit:

`e32228951a96ae595ec29c51b808033612e4b534`

Frozen active-token evidence was read from:

`75f34389faf6b83de4fb87758be0beeec8f7f2b1`

No tokenizer execution, training, logits read, task-head evaluation, causal
intervention, geometry analysis, or hyperparameter sweep was performed.

## Execution protocol

Population:

- 336 frozen K0-RVG items
- corr and ctrl roles independently
- matched versus swapped within each role
- 672 role-level comparisons total

Observation:

- primary Mamba layer: 23
- divergence anchor: first matched-versus-swapped token-value difference
- coordinates: k=-1 through k=+6
- state fields: S_prev, G, W, S_post

Matched and swapped branches were truncated to the same effective sequence
length through k=+6 before each forward:

`EQUAL_LENGTH_PREFIX_TRUNCATED_THROUGH_K_PLUS_6`

This removes differing future suffix length as an explanation for exact-byte
state differences inside the audited window.

Total scientific model forwards:

`1344`

## Artifact authentication

Execution manifest SHA256:

`8fef70560bc78cf4e0546b89c3d347b1e6262f4243071d7f275e291e5488f288`

Summary SHA256:

`9c9cc8fad1699a4d298d224863760790c5afe839a4648fc5dbfa9a8d8bdafe5a`

Full pair_summaries.jsonl remains in the local validated run artifact and is
not duplicated into the repository. Its SHA256 is:

`8fb8fc78cf893f4fb6cca34f4c5e08b6f01df8671e5d2e22b13f7036725c6281`

The execution-manifest-declared hashes of state_hash_comparisons.jsonl,
pair_summaries.jsonl, and summary.json were independently revalidated before
this freeze.

## Validated result

All 672 role-level comparisons showed the same transition-localization
signature.

At k=-1:

- matched and swapped token IDs were identical: 672/672
- S_prev, G, W, and S_post hashes were all identical: 672/672

At k=0, the first divergent token:

- token IDs differed: 672/672
- S_prev remained identical: 672/672
- G differed: 672/672
- W differed: 672/672
- S_post differed: 672/672

The first state-hash difference therefore occurred at k=0 for:

- corr: 336/336
- ctrl: 336/336

The first-difference field signature was universally:

`G, W, S_post`

From k=1 through k=+6, S_prev, G, W, and S_post all differed in all
672 role-level comparisons.

No role-level matched/swapped pair remained state-identical through k=+6.
No item remained state-identical in both roles through k=+6.

## Scientific interpretation

The previously observed P1/P2 matched-versus-swapped state identity does not
support a claim that distinct matched/swapped inputs collapsed to the same
layer-23 recurrent state.

The frozen active-token evidence showed that the historical P1 observation
window ended before matched-versus-swapped token divergence was exposed.
When the recurrence observation was realigned to the actual first
matched-versus-swapped token difference, all 336 items in both corr and ctrl
roles diverged immediately at that token.

The transition signature is consistent with the native recurrence ordering:

`S_post(t) = G(t) * S_prev(t) + W(t)`

At the first divergent token, the incoming state S_prev is still identical,
while token-conditioned transition quantities G and W differ and the
resulting S_post differs. At the next coordinate, the differing S_post has
become the differing S_prev.

This establishes transition localization, not semantic magnitude or direction.
Exact hashes establish byte-level identity/difference only. They do not
establish geometric distance, state-space direction, decision relevance, or
causal effect on model output.

## Item 163

The historical item-163 anomaly does not persist under the equal-length,
divergence-aligned protocol.

For both corr and ctrl, item 163 has:

- first state-hash difference at k=0
- identical S_prev at k=0
- differing G, W, and S_post at k=0

This supports interpreting the earlier pre-token-divergence byte difference
as an execution-path/numerical artifact associated with the unequal
full-sequence protocol rather than as evidence of anticipatory native-state
dependence.

## Next scientific boundary

Static transition localization is complete.

A subsequent experiment may measure the magnitude and trajectory of the
divergence-aligned raw layer-23 state vectors, but that requires a new
state-vector capture execution. Hash evidence alone does not answer that
question.