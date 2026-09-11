# K2 Exact-Prefix Corrective-Evidence Execution Preregistration Candidate

**Status:** STATIC DESIGN / IMPLEMENTATION-READY PREREGISTRATION CANDIDATE ONLY.

**Verdict:** PASS_READY_FOR_INDEPENDENT_REVIEW_AND_FREEZE.

This is the single prospective K2 construction and execution preregistration. It is not implementation or execution authority merely by being created. Once independently reviewed and committed, it is the complete scientific/design authority for the bounded implementation, Phase A eligibility screen, preflight, and conditional Phase B K2 execution below. No additional scientific execution-authority/specification document is needed for that frozen work.

## 1. Authority, audit, and boundary

Authority order: current controller instruction; frozen K2 theory at e8227945401d83b952e35b056343e3e1cb19e18f; frozen K0 at 7bd1cf824cd53c7f6cf6215346b42cabf351b70a; K1 closure at e8227945401d83b952e35b056343e3e1cb19e18f (INCONCLUSIVE_DUE_TO_PRESTATE_MATCHING_SUPPORT_FAILURE); frozen A0 source/evidence and authenticated common encoder; and O0c native selective-SSM semantics at ff2fb076f6e66a34a632515bb8502d8b1c90ad7f.

Audited materialization:

    pwd    = C:\Users\Home1\Desktop\ContraMamba-K
    branch = longterm-k-series-native-state-kinematics
    HEAD   = e8227945401d83b952e35b056343e3e1cb19e18f

Read-only audit covered frozen K2/K0/K1 reports; controlled-data source/schema/generator; the A0 lineage source and prediction/provenance authorities; v6b_minimal model/head interface; and O0c instrumentation/preflight authority. K1 scaffolding is neither inspected as authority nor reused.

K2 changes identification from observational prestate matching to an exactly shared pre-event sequence and a predefined continuation intervention. It does not repair or reopen K1.

## 2. Fixed source population and sole construction family

### 2.1 Source population

The only source population is the completed A0-lineage controlled dataset:

    reports/reason_router_p2_p3w6f2_p4b_r1_regeneration_execution_4122078ab7962042e3d6bf89f8b4eb5cec463458/controlled_v5_v3_without_time_swap_p3w6f2_r1_regenerated.jsonl
    physical SHA256 = eb1e0614939cda1421052702223f0fda91f098564692141b085b95b18558c0d3
    semantic SHA256 = 3797c174294f6d4f4efbe3afd05530b39c891f1e986dc05fbace59345d6e9c3b

The source schema is exactly: id, pair_id, claim, evidence, final_label, frame_compatible_label, predicate_covered_label, sufficiency_label, polarity_label, primary_failure_type, intervention_type. Its generator supplies same-pair controlled rows including evidence_truncation, entity_swap, none, and polarity_flip.

### 2.2 Selection: truncation-prefix / same-pair refute / same-pair entity control

For each source pair, derive exactly one candidate recipe:

    T = evidence_truncation row
    E = entity_swap row
    Q = polarity_flip if its final_label is REFUTE;
        otherwise none if its final_label is REFUTE.

The sole literal UTF-8 serialization is:

    P_i = "Claim: " + T.claim + "\nEvidence: " + T.evidence + "\nAdditional evidence:"
    C_i = " This is false: " + Q.evidence
    N_i = " A separate event: " + E.evidence
    X_i^corr = P_i + C_i
    X_i^ctrl = P_i + N_i

No Unicode normalization, escaping rewrite, added special tokens, or padding is permitted.

The controlled source makes semantic validity mechanical: T is NOT_ENTITLED with sufficiency zero; Q is REFUTE for the byte-identical claim; E is NOT_ENTITLED, frame-mismatched, polarity NONE, and does not refute the claim. Thus P is misleading/insufficient, C is decisive REFUTE evidence, and N is matched non-corrective evidence. The continuations are evidence-bearing sentences, not padding.

This is selected because it is first under the required hierarchy: source labels make semantics auditable; it minimally transforms existing controlled examples; exact common-prefix identity is literal construction; correction/control matching uses no native state; and every usable source pair is considered. There are no alternate construction arms.

## 3. Candidate pool frozen before head screening

Read the source JSONL in physical line order and group by pair_id. A pair is construction-invalid unless it has exactly one T, E, and Q as above; all same-pair claim bytes agree; T has final_label NOT_ENTITLED, primary_failure_type sufficiency, sufficiency_label 0; E has final_label NOT_ENTITLED, primary_failure_type frame, polarity_label NONE, and evidence bytes distinct from T; and Q has final_label REFUTE, polarity_label REFUTE, and the T claim bytes.

Tokenize the two complete branches under section 4. Construction validity additionally requires continuation lengths L_i in [8,24] canonical tokenizer tokens, exact branch-length equality, and a divergent first continuation token. An item failing this gets INVALID_CONTINUATION_TOKEN_CONTRACT. It is never regenerated, synonym-swapped, shortened, or artificially padded.

Each candidate representation is canonical JSON (UTF-8, sorted keys, compact separators, final LF) with exactly:

    schema_version, stable_item_id, base_claim_sha256, pair_id,
    truncation_source_id, refute_source_id, control_source_id,
    prefix_text, correction_text, control_text,
    source_dataset_physical_sha256, source_dataset_semantic_sha256,
    construction_status, construction_failure_label

base_claim_sha256 is SHA256 of UTF-8 claim bytes. stable_item_id is "k2ep-v1:" plus lowercase SHA256 of canonical JSON containing schema_version, pair_id, the three source IDs, and the three text fields, before stable_item_id is inserted.

For duplicate base_claim_sha256 values, retain only the lexicographically smallest (pair_id, stable_item_id) as construction-valid; label the others DUPLICATE_BASE_CLAIM_EXCLUDED. Do not substitute another variant after screening. Order all attempts by ascending stable_item_id and emit all, including failures and duplicates, as canonical JSONL. CANDIDATE_POOL_SHA256 is SHA256 of these exact bytes.

No native-state value, head prediction, confidence, margin, recovery, inertia, or endpoint may affect generation, semantic validation, duplicate resolution, validity, or order.

## 4. Token and exact-prefix contract

The K2 reproducibility identity is state-spaces/mamba-130m-hf at model/config/tokenizer revision 5708daa364c50b880e7bd92eab456e0d34492ee9. This is not historical A0 tokenizer identity. Before Phase A, record SHA256 and byte count of every resolved model configuration and tokenizer file, plus resolved paths, tokenizer implementation identity, add_special_tokens=false, and trust_remote_code=false. Mismatch fails closed.

Tokenize branch strings separately and unpadded with add_special_tokens=false. Define tau_i = len(tokenize(P_i).input_ids)-1 and require tau_i >= 1. Require:

    input_ids_corr[:tau_i+1] == input_ids_ctrl[:tau_i+1]
    attention_mask_corr[:tau_i+1] == attention_mask_ctrl[:tau_i+1]
    position_ids_corr[:tau_i+1] == position_ids_ctrl[:tau_i+1]

Unpadded single-sequence masks are all one and positions are consecutive zero-based values. Require equal post-tau token count L_i, 8 <= L_i <= 24, and first post-tau token divergence. No batch-padding token can be captured, enter a state row, or enter an endpoint. Violations are INVALID_EXACT_PREFIX or INVALID_CONTINUATION_TOKEN_CONTRACT.

tau_i is the final common-prefix post-consumption state; tau_i+1 is the first divergent continuation token. The fixed primary window is W = {1,2,3,4,5,6,7,8}. There is no event, token, or window search.

## 5. Phase A eligibility and final N

Phase A may construct the pool, tokenize, execute prefix-only fixed decision heads, classify eligibility, and produce hashes/manifests. It must not capture native state, instantiate/enable a state observer, inspect native state, calculate R/D/P, or inspect K2 outcomes.

For every construction-valid nonduplicate candidate, run literal P_i through all frozen A0 decision observers. Eligibility is exactly:

    gold(prefix) = NOT_ENTITLED
    seed180 = SUPPORT
    seed181 = SUPPORT
    seed182 = SUPPORT

No confidence/margin cutoff exists. Store probability, confidence, and margin only as diagnostics. The full screening artifact must retain every pool row, construction status/failure, all three outputs, and eligible/ineligible; a successes-only output is forbidden.

Before any native-state capture, bind these SHA256 values:

    CANDIDATE_POOL_SHA256
    SCREENING_ARTIFACT_SHA256
    ELIGIBLE_ID_LIST_SHA256
    FINAL_CONFIRMATORY_ID_LIST_SHA256

The eligible ID list is canonical JSONL of ascending stable IDs. If fewer than 30 eligible distinct base items exist:

    K2_CONSTRUCTION_VERDICT = INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE

Stop; no native-state scientific capture occurs. Do not weaken unanimity or add threshold tuning. If 30--64 are eligible, use all. If more than 64 are eligible, select exactly 64 by ascending SHA256 of canonical stable item representation, tie-break ascending stable ID. The final list is canonical JSONL in that selection order.

The N>=30 floor is a single-endpoint design-sensitivity calculation: approximately 30 independent paired items gives about 80% power for a two-sided exact sign test at alpha 0.05 under strong 75% directional imbalance versus the 50% null. It is not a fitted effect estimate, and does not claim 80% power for the full three-endpoint Holm family, arbitrary effect sizes, arbitrary zero fractions, or the complete K2 experiment.

## 6. Canonical model and native observer

Native capture uses one full canonical A0 realization, seed180, only after authenticating its full checkpoint and the common-encoder identity for each of seed180, seed181, and seed182. The normative K2 common-encoder value fingerprint has this exact algorithm: select all checkpoint/model-state entries whose names begin exactly `mamba.`; order names lexicographically ascending; for each selected tensor compute `tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()` and `per_tensor_sha256 = lowercase SHA256(tensor_bytes)`; construct the mapping `{tensor_name: per_tensor_sha256, ...}`; serialize that mapping as canonical JSON encoded UTF-8 with sorted keys, compact separators, and no trailing newline; then compute `K2_COMMON_ENCODER_CANONICAL_VALUE_SHA256 = SHA256(canonical_json_bytes)`. It is required independently for each of seed180, seed181, and seed182:

    K2_COMMON_ENCODER_CANONICAL_VALUE_SHA256 = 48a7e9ac9dfa6c8c292090ee0fcb606bd4c85d13706bfc8a3e371af77c440597

The following structural fingerprint is mandatory for each of the same three seeds, and is a companion check rather than a substitute for the canonical digest:

    tensor_count    = 242
    total_numel     = 129135360
    total_raw_bytes = 516541440
    dtype           = float32 for all 242 tensors

As an informative independent, non-normative cross-check, `RAW_CONCAT_NAME_SORTED_SHA256` is SHA256 of the concatenation of `tensor.detach().cpu().contiguous().numpy().tobytes()` for all `mamba.*` tensors in ascending tensor-name order. It reproduces as follows for each of seed180, seed181, and seed182:

    RAW_CONCAT_NAME_SORTED_SHA256 = 968c12c095a6aab883db5984f4c02ad5e893a5ff140781ffbda41b97970401ae

This raw-concatenation value is a secondary cross-check only. It does not replace the normative canonical map digest and is not a second execution gate when the normative digest and structural checks pass.

Provenance correction: the historical literal `67bfc8cb253fef88b2b8936d442468b9ddcbffa8b79582ba3e2432cb271a937b` is SUPERSEDED for K2 encoder validation because its derivation cannot be reproduced from the authenticated A0 checkpoints or strict-loaded encoder state; historical implementation deriving that literal was not recovered. This is a provenance defect, not evidence of encoder inequality or corruption. Forensic recovery established that all 242 float32 encoder tensors are byte-identical across seed180/181/182, preserving the intended common-encoder premise.

The seed180/181/182 heads remain fixed decision observers for eligibility/recovery only. They are not three native-state replicates. Every branch/item native trajectory is one fresh seed180 common-encoder forward with no cache, state, snapshot, or padding reuse.

Capture only native selective-SSM recurrent S_t^(l), post-consumption after token t, under validated O0c semantics: frame-local ssm_state cloned after the recurrent update and before readout. Hidden states, convolution cache, final cache, scan workspaces, and reconstructed proxies are forbidden. Require exactly 24 layer states, indexed 0--23; trace is disabled by default.

Before scientific capture, require trace-off/trace-on exact noninterference; prefix prestate equality in every layer through tau_i; all-layer presence; complete valid-token state rows; finite float32 states; source/runtime/checkpoint provenance identity; and fresh-state isolation. Any failure stops under section 10.

## 7. Primary layer and frozen endpoints

PRIMARY_LAYER is layer 23, zero-indexed final native Mamba layer. It is final recurrent state feeding downstream representation and prevents a 24-layer multiplicity search. Capture every layer; layers 0--22 are secondary descriptive profiles only and cannot promote K2.

Set epsilon = 1e-12. For b in {corr, ctrl}, k=1,...,8:

    V_b,tau+k = S_b,tau+k - S_b,tau+k-1

The only three primary paired endpoints at layer 23 are defined as follows. For branch b in {corr, ctrl}:

    mean_speed_i^b = (1/8) * sum_{k=1..8} ||V_{i,tau_i+k}^{b,(23)}||_F
    R_i = mean_speed_i^corr - mean_speed_i^ctrl

R is a paired arithmetic difference. Positive R means greater post-event movement under the designed corrective continuation relative to its predefined control; it is not automatically better or recovery.

    turn_{i,tau_i+k}^{b,(23)} = 1 - cosine(
        V_{i,tau_i+k}^{b,(23)}, V_{i,tau_i+k-1}^{b,(23)})

for k=1,...,8. For k=1, the previous velocity is the final common-prefix velocity V_{i,tau_i}^{(23)}. An individual turn is invalid whenever either adjacent velocity norm is <= epsilon. For each branch, mean_turn_i^b is the arithmetic mean over its valid turns among k=1,...,8. If either branch has zero valid turns, D_i = UNDEFINED; it is never replaced with zero. Otherwise:

    D_i = mean_turn_i^corr - mean_turn_i^ctrl

D is a paired arithmetic difference.

    L_i^b = sum_{k=1..8} ||V_{i,tau_i+k}^{b,(23)}||_F
    eta_i^b = ||S_{i,tau_i+8}^{b,(23)} - S_{i,tau_i}^{(23)}||_F / (L_i^b + epsilon)
    P_i = eta_i^corr - eta_i^ctrl

For R/P, any missing or invalid required state is the capture/endpoint failure contract rather than a manufactured endpoint value. Such endpoint values are reported with reason and excluded only from that endpoint's sign count; they are never imputed or used to alter final IDs. DeltaS trajectories and layers 0--22 are descriptive only. Path length is not a fourth endpoint because in this fixed window it is algebraically redundant with mean speed.

## 8. Exact inference and decision diagnostics

The scientific unit is one base item, never a branch or layer. For every endpoint E in {R,D,P}, the scientific report must disclose N_CONFIRMATORY, n_valid_E, n_positive_E, n_negative_E, n_zero_E, n_undefined_E, n_eff_E, zero_fraction_E, and undefined_fraction_E, and check the identity:

    n_valid_E = n_positive_E + n_negative_E + n_zero_E
    n_eff_E = n_positive_E + n_negative_E
    N_CONFIRMATORY = n_valid_E + n_undefined_E
    zero_fraction_E = n_zero_E / n_valid_E when n_valid_E > 0; otherwise UNDEFINED
    undefined_fraction_E = n_undefined_E / N_CONFIRMATORY

For valid E_i, E_i > 0 is positive, E_i < 0 is negative, and E_i == 0 exactly is a zero/tie. Zeros are excluded from the sign-test sample, so n_eff_E = n_positive_E + n_negative_E. For each endpoint run a two-sided exact binomial sign test against H0: P(positive | nonzero) = 0.5, with no normal approximation. If n_eff_E > 0, compute the exact two-sided binomial sign-test p-value. If n_eff_E == 0, RAW_P_E = 1.0; this fail-safe no-evidence result preserves the fixed three-test primary family and cannot count as significance or endpoint support.

For each endpoint:

    r_rb_E = (n_positive_E - n_negative_E) / (n_positive_E + n_negative_E)

when n_eff_E > 0; when n_eff_E == 0, r_rb_E = UNDEFINED. Report r_rb whenever defined, regardless of significance. No magnitude threshold for promotion and no new minimum n_eff threshold are introduced. Large zero or undefined fractions are visible and cannot be hidden by reporting only n_eff.

The primary family always contains exactly R, D, and P (m = 3). Report RAW_P, HOLM_ADJUSTED_P, and HOLM_REJECT for all three. Apply standard Holm step-down FWER alpha=0.05. Sort raw p-values by ascending RAW_P; for exactly equal RAW_P, use fixed endpoint order R < D < P. For sorted endpoints j=1,2,3, use ordinary Holm multipliers (3,2,1) and monotone adjusted p-values:

    q_(1) = min(1, 3*p_(1))
    q_(2) = max(q_(1), min(1, 2*p_(2)))
    q_(3) = max(q_(2), min(1, 1*p_(3)))

Map q values back to R/D/P. HOLM_REJECT must match the standard Holm step-down rule at alpha=0.05. No layer, trajectory summary, recovery stratum, or other metric joins this family; the family size is not changed when an endpoint has RAW_P = 1.0 or many ties.

Level-1 promotion requires valid construction; valid instrumentation/provenance; N_ELIGIBLE >=30; and at least one of the three layer-23 primary endpoints rejected under the frozen Holm family. The scientific report must display for all R/D/P: effect direction, r_rb, n_positive, n_negative, n_zero, n_undefined, n_eff, zero_fraction, RAW_P, HOLM_ADJUSTED_P, and HOLM_REJECT. Layers 0--22 cannot rescue primary failure; recovery/inertia descriptions cannot rescue primary failure; and statistical significance does not establish recurrent-state necessity, sufficiency, or mechanism. If none pass, H_NULL remains viable. Opposite directions do not confirm directional H_INERTIA or H_RECOVERY.

After full C_i, classify heads as RECOVERY if all REFUTE, INERTIA if all SUPPORT, otherwise DECISION_DISAGREEMENT. These labels are descriptive only: they cannot select items, create another inferential family, or support Level-3 promotion in this first K2 run.

## 9. Hard Phase-A / Phase-B barrier and future authority

Phase B may begin only after all four Phase-A hashes are bound and support floor passes. It may capture both branches for final IDs only, compute frozen endpoints, perform the three-test Holm analysis, and issue provenance/scientific artifacts. It may not alter candidates, source rows, controls, eligibility, final IDs, tau, window, layer, endpoint, or use recovery/inertia for selection.

The Phase-B manifest must bind this document commit, implementation commit, all source/tokenizer/config/checkpoint/head/O0c runtime hashes, Phase-A hashes, selected/source IDs, exact inputs/token arrays/tau, integrity results, reconstructable endpoint inputs, and all output SHA256s. Any identity mismatch fails closed.

Once this document is independently verified and committed, implementation is independently verified and committed, and exact commit/provenance gates pass, K2 Phase A and conditional Phase B may run under this same preregistration. Kaggle is allowed only when needed at a specific committed implementation. CPU construction/preflight must not consume GPU.

## 10. Causal/generalization boundary and stop conditions

K2 inference initially applies only to this frozen prospectively constructed population/family. It does not automatically generalize to arbitrary natural-language correction, all factual errors, or other Mamba checkpoints; held-out construction families are required for broad generalization.

The only supported intervention is designed corrective continuation relative to predefined matched non-corrective continuation. C_i and N_i necessarily differ semantically. Exact prefix does not isolate a microscopic correction feature or support an abstract correction claim independent of continuation semantics. Native-state necessity/sufficiency remains K3.

Fail closed for INVALID_CONSTRUCTION, INVALID_EXACT_PREFIX, INVALID_PRESTATE, INVALID_CORRECTION, INVALID_CONTROL, INVALID_CAPTURE, and PROVENANCE_MISMATCH. Fewer than 30 eligible items is INCONCLUSIVE_DUE_TO_WRONG_COMMITMENT_SUPPORT_FAILURE. Infrastructure failure is never H_NULL.

Before Phase A, focused tests plus at most three state-blind sentinel items must prove exact-prefix token equality, prestate equality, 24-layer capture, post-consumption semantics, trace noninterference, tau/W indexing, padding exclusion, endpoint formulas, exact sign-test/Holm implementation, and provenance/hash fail-closed behavior. Sentinels are infrastructure only, never scientific inference.

Creation of this candidate authorizes no model/checkpoint execution, tokenization, native-state capture, training, evaluation, Kaggle, staging, commit, or push.
