# ContraMamba RFR-H2-P2 — Tokenizer / Causal-LM Readout Static Feasibility Decision

- Date: 2026-10-24 (Asia/Seoul)
- Branch: `gen5-causal-role-state-ownership`
- Inspected baseline HEAD: `f0c1d7c99a449adc7f3446b0822b86b296c0193a`
- Status: `H2_P2_STATIC_ARCHITECTURE_FEASIBLE / TOKENIZED_READOUT_NOT_VERIFIED / NO_EXECUTION`
- Scope: read-only examination of frozen RFR-H2-P0/P1 and historical Mamba model/tokenizer references, plus publicly available model/config/API metadata. No Python module import, tokenizer load/encode, model/checkpoint load, logits, generated inputs, dataset, new root, seed, split, GPU, or native-state experiment.

## 1. Exact question and decision

Does a *plausible frozen, training-free causal LM readout* exist for the future RFR-H2 sequential evidence task, without treating the historical ContraMamba three-way classifier as an RFR decoder?

**Decision:** `CAUSAL_LM_ARCHITECTURE_CANDIDATE_EXISTS`; `ACTUAL_TOKEN_ID_CONFORMANCE=NOT_EVALUATED`; `COMPARABLE_ANSWER_SCORES=NOT_ESTABLISHED`; `MODEL_EVALUATION=BLOCKED`. The design can proceed to a separately bounded, CPU-only tokenizer/continuation conformance preflight, but P2 itself does not enable it.

## 2. Checked source identities

- H2-P0: `reports/rfr_h2_p0_temporal_serialization_readout_identifiability_design_candidate.md` defines one causal stream, PRE physical t1 exclusion, query branching after identical pre-query prefixes, untrained causal terminal continuation scoring and H2/H3 separation.
- H2-P1: `scripts/rfr_h2_p1_temporal_serialization_contract_cpu.py` uses fixed hypothetical English rule, t0 A/C/U, t1 A, neutral delay, and `Query: Is {A|B|U} true under the current verified record?\nAnswer: `. Its `READOUT_SPEC` deliberately leaves candidate continuations, score aggregation, tokenizer and head identity unset. The reported integration result preceding commit was 220/220 tests PASS, **not** a tokenizer or model run.
- Historical O0a `scripts/observe_longterm_o0a_native_mamba_state_dynamics.py` names the base repo `state-spaces/mamba-130m-hf` and immutable model *and* tokenizer revision `5708daa364c50b880e7bd92eab456e0d34492ee9`, but loads `MambaModel` without an LM head. Its hidden states are proxies, not recurrent raw states.
- Historical Gen4 adapter `scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py` also names `state-spaces/mamba-130m-hf`, but pins tokenizer revision reference `40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37`, specific tokenizer file hashes, and a fixed **128-token 63/1/64 claim/evidence** allocation. Different revision identifiers are **not evidence of different tokenizer bytes**; equality has not been established here. The historical allocation cannot be reused as an RFR length limit.
- The public Hugging Face repository at **the exact O0a revision `5708daa...`** lists `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` (file presence only). Its config declares `architectures=["MambaForCausalLM"]`, `num_hidden_layers=24`, `hidden_size=768`, `state_size=16`, `vocab_size=50280`, with BOS/EOS/PAD model IDs declared as 0. This is metadata evidence, not an authenticated weight load or scoring test.
- The public **Transformers 5.0.0 Mamba** documentation defines `MambaForCausalLM` with an LM head tied to embeddings, next-token logits, and causal forward. It documents teacher-forced labels, `cache_params`, and `logits_to_keep` functionality. These API descriptions do **not** verify the frozen repo runtime's actual implementation bytes, loaded head weights, or stable token boundaries.

Public references (accessed for this static review):

1. `https://huggingface.co/state-spaces/mamba-130m-hf/tree/5708daa364c50b880e7bd92eab456e0d34492ee9`
2. `https://huggingface.co/state-spaces/mamba-130m-hf/blob/5708daa364c50b880e7bd92eab456e0d34492ee9/config.json`
3. `https://huggingface.co/docs/transformers/v5.0.0/en/model_doc/mamba`
4. `https://huggingface.co/state-spaces/mamba-130m-hf/blob/5708daa364c50b880e7bd92eab456e0d34492ee9/tokenizer_config.json`

## 3. Critical unresolved tokenizer/readout contracts

1. **One pinned bundle:** Prospective preferred identity is model + tokenizer at the **same existing O0a exact revision `5708daa...`**. Do not silently pair an unverified Gen4 tokenizer revision with a different model revision. Independently check exact files/hashes and runtime loader behavior in a later preflight.
2. **Actual token-boundaries:** P1's UTF-8 byte anchors and `pre_query_sha256` do not establish token anchors. The entire `pre_query + query + answer` token sequence must be audited with actual pinned tokenizer rules, including automatic BOS/EOS, special-token insertion, normalization, whitespace and leading-space behavior. Identical standalone prefix encodings need not remain identical when a suffix is appended due to subword boundary effects.
3. **Query isolation:** For the same `(world, arm, readout)`, the tokenizer-visible token IDs up to the query anchor must match across A/B/U branches; PRE token inputs must be arm-identical; POST and DELAYED must share identical t0/t1 bytes and tokens before the added neutral delay. The **scored conditional input is `pre_query + query`**, not `pre_query` alone; its query changes as intended.
4. **Candidate continuation scheme:** Predeclare exactly one answer continuation/verbalizer per `REFUTE`, `NOT_ENTITLED`, `SUPPORT`, with an exact appended delimiter policy, special-token/EOS policy, and a proof that each produces a nonempty, distinguishable candidate token sequence without altering/scoring prompt tokens as answer. Do not guess token IDs, same-token-count, or calibration from word appearance.
5. **Comparable scoring:** A future selected rule must compute a teacher-forced conditional candidate log-probability from the same fully specified query context. Determine **before outcomes** whether to use summed log likelihood, a justified normalization, or a controlled token-equated scheme. Multi-token and unequal-length verbalizers create length/prior confounds. Check causal shift, mask, candidate-only score indices, model vocabulary bounds, no generation or answer-derived selection. No post-hoc best verbalizer, prompt, layer or temperature search.
6. **Input budget:** The tokenizer's declared giant `model_max_length` or historical 128-token Gen4 limit does not establish RFR safety. Check actual whole sequence and continuation lengths, event/query anchor survival and model/runtime admissibility. Reject truncation that loses rule, t0, t1 authority, delay or query. Delay length remains unfrozen.
7. **Semantic shortcuts:** P1 literals use `authenticated=yes/no` and source names `REV_A` vs `UNVERIFIED_A`, which make validity explicitly readable but may also support an unintended fixed alias cue. A valid task must present authentication evidence, but eventual matched source/alias, polarity, duration, class-prior, and query controls must be frozen prospectively. P2 cannot certify shortcut resistance.
8. **Architecture boundary:** The documented LM class exposes `backbone` + `lm_head`; the historical Gen4 instrumentation calls `model.mamba` and is runtime/source/shape pinned. No direct plug-in compatibility or H3 raw-state measurement is established by the LM-head existence. The original Gen4 claim/evidence router's final 3-way logits remain **invalid** as an RFR-specific readout.
9. **Epistemic support:** D1's eight strict-Boolean primary strata only yield `REFUTE`/`SUPPORT` gold. A three-answer scorer does not prove `NOT_ENTITLED` competence without a separately specified missing-premise control; none is added here.

## 4. Claim ledger and stop conditions

| Claim | P2 status |
|---|---|
| Public pinned repo contains LM-designated model/config and tokenizer files | `STATIC_FILE_PRESENCE_CONFIRMED` |
| Documented Mamba causal LM interface exists | `STATIC_API_CONFIRMED` |
| Loaded checkpoint and head source/weight identity verified | `NOT_EVALUATED` |
| Actual tokenizer / continuation token IDs | `NOT_EVALUATED` |
| Byte-to-token anchor preservation, full-prefix continuity | `NOT_EVALUATED` |
| Teacher-forced answer-score comparability | `NOT_ESTABLISHED` |
| RFR learned or zero-shot behavior | `NOT_EVALUATED` |
| Independent case population | `NOT_ESTABLISHED` |
| Native recurrent-state causal mediation | `NOT_ESTABLISHED` |

Fail closed with `BLOCKED_TOKENIZER_BOUNDARY_OR_TRUNCATION` for mismatched prefix anchors or lost events, `BLOCKED_LM_HEAD_IDENTITY` for unverified model/head bundle, `BLOCKED_UNCOMPARABLE_CONTINUATIONS` for non-distinct or improperly scored answer strings, and `BLOCKED_SHORTCUT_OR_H3_PROMOTION` for evidence leakage or unsupported science claims. The P1 tests' PASS status cannot override any of these failures.

## 5. Next bounded action (not execution authority)

After this report-only decision is frozen, the next scientific feasibility step is **RFR-H2-P3: CPU-only tokenizer/answer-continuation conformance preflight design and independently testable implementation contract**, pinned prospectively to an exact model/tokenizer bundle and explicit input/continuation strings. Its *actual tokenizer loading or tokenization* requires a separate, scoped permission and validation; **do not assume P2 authorizes it**. Do not load the LM, compute logits, run H2/H3 evaluation, draw roots, create splits or train. If the exact compatible bundle, token anchors or comparable readout cannot be established, stop instead of escalating to GPU or a learned probe.

Final: `H2_P2=STATIC_ARCHITECTURE_FEASIBLE / TOKENIZER_CONFORMANCE=NOT_EVALUATED / LM_READOUT=NOT_ESTABLISHED / MODEL_EVALUATION=NO / H3_CAUSAL_MEDIATION=NOT_ESTABLISHED / PHASE3B=BLOCKED_UNCHANGED / GPU=OFF`.
