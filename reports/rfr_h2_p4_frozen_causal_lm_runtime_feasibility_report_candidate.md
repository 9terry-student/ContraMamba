# ContraMamba RFR-H2-P4 — Frozen Causal LM Runtime and Input-Shape Feasibility Report (Candidate)

- Date: 2026-10-24 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Inspected baseline HEAD: `b75c0f66844457d61bc3d021064b7f62a35c9782`
- Phase: **P4 technical preflight evidence consolidation / report-only**.
- Status: `H2_P4_CPU_151_TOKEN_FORWARD_SHAPE_PASS_ONLY / NO_RFR_BEHAVIOR_EVALUATION / NO_H3_CAUSAL_CLAIM`.
- Provenance boundary: the local Windows inventory, checkpoint verification, model-load checks and CPU smoke output below were **provided as operator console output in this research conversation**. This candidate does not contain an independently archived run log or a new reproducible run artifact. GitHub was read at the baseline HEAD to confirm the previously frozen P0/P2 and P3 definitions.
- Authority boundary: this report does not authorize new model evaluation, data/root generation, prompt/verbalizer tuning, native-state interventions, training, GPU/Kaggle, code changes, or Phase3B promotion.

## 1. Frozen prerequisites

1. P0 `reports/rfr_h2_p0_temporal_serialization_readout_identifiability_design_candidate.md`: prospective single-stream H2 design with PRE/POST/DELAYED and A/B/U, separate from H3 causal mediation.
2. P1 `scripts/rfr_h2_p1_temporal_serialization_contract_cpu.py` and tests: static deterministic serialization/label agreement; prior operator report **42/42 P1 tests PASS**.
3. P2 `reports/rfr_h2_p2_tokenizer_lm_readout_static_feasibility_decision_candidate.md`: same-revision Mamba causal LM candidate, historical Gen4 router excluded.
4. P3 `scripts/rfr_h2_p3_tokenizer_continuation_conformance_cpu.py` and tests, frozen at inspected HEAD: **216 hypothetical contexts, 864 token-anchor checks and 648 single-token continuation checks** PASS under fixed O0a tokenizer. These are related logical variants, not independent cases. Verbalizers use the boundary-reassigned `Answer:` context followed by leading-space continuations; the completed visible text is unchanged.
5. P3 fixed output IDs: `REFUTE=16708` (` False`); `NOT_ENTITLED=32471` (` Unknown`); `SUPPORT=11793` (` True`). Proposed, **not executed**, readout: one next-token logit at the final query position, no EOS or normalization. P1's original scaffold remains frozen and P3 defines a readout seam without editing P1.

## 2. Pinned identities from operator-reported checks

| Object | Frozen identity / observed result |
|---|---|
| Base model and tokenizer | `state-spaces/mamba-130m-hf` @ `5708daa364c50b880e7bd92eab456e0d34492ee9` |
| Model weights | `model.safetensors`; **516,567,560 bytes**; SHA256 `1a5ed29c492ef4d485df3b7c2c8109771696589855b2162ad1ba618b6067cbea`; PASS |
| Model `config.json` | SHA256 `784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a`; PASS |
| Tokenizer file identities | P3's exact O0a snapshot hashes authenticated; Gen4 tokenizer files are different bytes and were **not** mixed |
| Installed runtime | Windows; Python `3.13.2`; PyTorch `2.12.1`; Transformers `4.45.0`; tokenizers `0.20.3`; safetensors `0.8.0` |
| Installed Mamba source | `transformers/models/mamba/modeling_mamba.py`; SHA256 `0fb3d343b2016aa4abe0c493ae143dc2193e1148bdd8792e9486ad04f5e30d71`; **37,982 bytes** |
| Model architecture | `MambaForCausalLM`, 24 layers, hidden size 768, vocabulary 50,280, selective state size 16 |
| Weight storage | 242 tensors; embedding `backbone.embeddings.weight` of `[50280,768]`; no distinct `lm_head.weight` in safetensors |
| Git status after preflights | Branch correct; HEAD unchanged; no staged/unstaged/untracked changes reported |

These identities describe **P4 local runtime**, not the previously frozen Gen4 native-state instrumentation runtime (`torch 2.10.0+cpu`, `transformers 5.0.0`) or its raw-state source hashes. Do not transfer a Gen4 observer gate to this environment.

## 3. Model-loading and source-contract evidence

- Installed `MambaForCausalLM` declares `_tied_weights_keys = ["lm_head.weight"]`, constructs `nn.Linear(hidden_size, vocab_size, bias=False)`, and exposes that projection as its output embedding.
- Installed `PreTrainedModel.tie_weights()` links the output and input embeddings when `config.tie_word_embeddings` is true. The **actual CPU-loaded** model reported `config.tie_word_embeddings=True`, both embeddings shaped `(50280, 768)`, and `input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr()` **True**.
- `from_pretrained` used the exact local pinned snapshot, float32 CPU and `local_files_only=True`; reported `missing_keys=[]`, `unexpected_keys=[]`, `mismatched_keys=[]`, `error_msgs=[]`; model layer count was 24.
- Source AST inspection located `MambaModel.forward`, `MambaMixer.forward`, `MambaMixer.slow_forward` and `MambaForCausalLM.forward`. With `use_cache=False`, model code selects a no-cache full-sequence branch; the local system warned that GPU fast paths were unavailable and selected the sequential Mamba implementation. No cross-runtime or CUDA equivalence claim follows.
- The causal LM `forward` produces `logits = lm_head(hidden_states).float()`. If `labels` is omitted, no supervised CE-loss evaluation occurs. The code inspection alone is not evidence of an executed RFR next-token readout.

## 4. Bounded shape-only CPU smoke evidence

The operator ran a **single** local CPU forward after authenticating model/config and Mamba source hashes. Explicit inputs/settings:

```text
input_ids = constant token ID 42 repeated 151 times; shape=(1,151)
device=cpu; dtype=float32; model.eval()
use_cache=False; labels=None; output_hidden_states=False; return_dict=True
no RFR rule, evidence, query, gold labels, candidate IDs or answer scores
```

Observed console output:

```text
CHECKPOINT_IDENTITY=PASS
RUNTIME_SOURCE_IDENTITY=PASS
INPUT_SHAPE=(1,151)
LOGITS_SHAPE=(1,151,50280)
LOGITS_DTYPE=torch.float32
ALL_FINITE=True
CACHE_RETURNED=False
RESULT=H2_P4_CPU_151_TOKEN_FORWARD_SHAPE_PASS
RFR_EVALUATION=NO
PHASE3B=BLOCKED_UNCHANGED
```

**Technical decision:** `H2_P4_CPU_151_TOKEN_FORWARD_SHAPE_PASS_ONLY`. The verified 151-token **synthetic shape** covers P3's reported maximum completed sequence length of 151; this does not certify that every P3 semantic prompt has been run, that every arbitrary 151-token sequence succeeds, that cache-based streaming is equivalent, that attention masking works, or that a 151-token model context limit has been formally established. No decoding, gold-label comparison, accuracy, or causal-state intervention occurred.

## 5. Claim ladder: explicit PASS versus unknown

| Question | P4 status |
|---|---|
| Exact source/weight/config authentication | `PASS_OPERATOR_REPORTED` |
| Local model load and no missing/unexpected weight keys | `PASS_OPERATOR_REPORTED` |
| Input/output LM head weight sharing | `PASS_OPERATOR_REPORTED` |
| CPU single 151-token synthetic forward; finite expected-shape logits; cache disabled | `PASS_OPERATOR_REPORTED` |
| Tokenizer readout seam with distinct context-stable one-token candidates | `PASS_P3_OPERATOR_REPORTED` |
| RFR prompt forward, or readout candidate logits | `NOT_EVALUATED` |
| Class-prior/bias diagnostic and shortlist of valid scientific controls | `NOT_ESTABLISHED` |
| Success on authenticated A-revision, conditional B, preservation U, delayed contrasts | `NOT_EVALUATED` |
| `NOT_ENTITLED` gold coverage in eight strict-Boolean worlds | `ABSENT` |
| Independent-root diversity and confirmatory population | `NOT_ESTABLISHED` |
| H3 raw selective-state causal necessity/sufficiency | `NOT_ESTABLISHED` |

## 6. Next bounded action: RFR-H2-P5 **design-only**, not evaluation

P4 has removed the *technical* blockers that made it premature to propose a concrete H2 evaluation. It does **not** grant scientific execution authority. The next task is to **freeze a falsifiable P5 pre-evaluation contrast/metric and baseline specification without running the LM**:

1. Predeclare the frozen readout `argmax(logits[16708], logits[32471], logits[11793])`, label mapping `REFUTE=0`, `NOT_ENTITLED=1`, `SUPPORT=2`, score tie handling, reporting of **all three raw candidate logits and normalized within-candidate probabilities**, and a no-post-hoc-threshold rule. Candidate-only normalization is descriptive and must never be described as calibrated 3-way epistemic probability.
2. Define matched full-case contrasts for `VALID_FLIP`, `VALID_SHAM`, `INVALID_FLIP`, at PRE/POST/DELAYED and A/B/U queries: A changes only under authenticated flip; B changes iff C permits and A's value switches; U must be retained. Report paired **joint case success**, not just correlated row accuracy. PRE arm copies count only once for independence.
3. Prospectively specify constant-label, latest-assertion, authenticated/unauthenticated-marker-only, query-only, C-omission/rule-agnostic, source-alias/template, and candidate-token-prior diagnostics. P3's fixed source markers `REV_A`/`UNVERIFIED_A` are a potential shortcut and must not be used to justify source invariance.
4. Predeclare whether a separately authorized future run on the eight related Boolean worlds would be labeled **exploratory smoke only**. Existing D3 M4–M8 does not establish independently sampled task-bearing-diverse roots, train/dev/test isolation, or a confirmatory sample; no generalization confidence interval is licensed. Strict-Boolean worlds cannot validate NOT_ENTITLED sensitivity. Any missing-premise extension needs separate pre-specified contract, not silent expansion.
5. Define fail-closed gates for tokenizer revision drift, prompt/answer seam drift, checkpoint or Mamba source drift, class-token ID drift, short readout, invalid gold mapping, family leakage, shortcut-dominated patterns, and conflation of H2 behavior with H3 mediation.

**P5 authorization boundary:** P5 may produce a static prospective protocol/report only; no new corpus or independent roots, tokenizer rerun, LM forward/score/evaluation, training, checkpoint modification, Kaggle/GPU, or H3 intervention unless separately and explicitly authorized. No P4 technical PASS alone authorizes an evaluation.

Final: `H2_P4=TECHNICAL_CPU_SMOKE_PASS / BEHAVIOR=NOT_EVALUATED / EVALUATION_AUTHORITY=NO / P5=PROSPECTIVE_DESIGN_NEXT / PHASE3B=BLOCKED_UNCHANGED / GPU=OFF`.
