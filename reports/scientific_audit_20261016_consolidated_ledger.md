# ContraMamba — Consolidated Scientific Audit Ledger

**Ledger ID:** `CM-SCI-AUDIT-20261016-V1`
**작성일:** 2026-10-16 (Asia/Seoul)
**성격:** 회고적 독립 감사 결과 통합 기록 / STATIC-REPORT-ONLY
**범위:** 초기 Stage/v1–v4, P2/P3-W7, Gen3, O-series, K-series, Gen4, Five-scale, Gen5
**최종 4개 감사의 기록상 참조 HEAD:** `ff806bd5bd8efc62fcf6420cf8b097a63ed70571` (detached)
**작성 출처:** 사용자가 이 연구 대화에 전달한 독립 Codex 감사 결과들과, 원격 저장소의 명시적 원본 frozen 보고서. 이 문서는 새로운 실행/재검증을 수행한 결과가 아니다.
**보존 상태:** 이 문서의 생성만으로 Git stage/commit/push가 되지 않는다.
**범위 유의:** 저장소 내 모든 파일/모든 가설을 전수 검증했다는 뜻이 아니다. 논문 관련 주요 과학적 계보의 회고 감사 종합이다.

---

## 0. 원칙과 판정의 의미

1. **숫자 재현**, **원본 실행·provenance 인증**, **통계 추론의 유효성**, **과학적·인과적 해석**은 서로 다르다.
2. 기존 보고서의 `PASS`나 `VALIDATED` 명칭은 감사자의 독립 인증을 대체하지 않는다.
3. `VALIDATED`는 명시한 계산/판정 범위에 한정한다. `VALID_WITH_LIMITATIONS`는 제한된 결과가 유효하다는 뜻이며 보편적인 과학적 승격이 아니다.
4. `PARTIAL`은 필요한 원본 tensor, checkpoint, 로그, runtime, 보호 holdout 또는 통계 명세의 인증이 부족함을 나타낸다. 자동으로 실패/반증을 의미하지 않는다.
5. `CLAIM_OVERSTATED`는 숫자가 옳더라도 그로부터 나온 더 강한 과학적 주장이 성립하지 않는다는 뜻이다.
6. `SCIENTIFIC_DEFECT_VERIFIED`는 코드·수학·레이블·provenance 등의 구체적 확인된 결함에 한정한다.
7. `CONTRADICTED_BY_EVIDENCE`는 해당 긍정적 주장과 저장된 증거가 모순됨을 뜻한다. 비유의성만으로 효과 0을 증명하지 않는다.
8. 본 문서는 사용자 제공 감사 보고서들을 **충실히 통합**한다. 각 보고서의 독립 재계산 자체를 본 문서 작성 과정에서 반복 실행하지 않았다.
9. 원본 원시 데이터나 감사 원문이 현재 문서 안에 재첨부되지 않은 부분은 `CHAT-REPORT-ONLY` 또는 `SOURCE-UNAVAILABLE`로 기록한다. 참고한 과거의 “29-claim audit”은 검증 가능한 로컬 경로가 없으므로 증거로 사용하지 않는다.

## 1. 전체 감사 상태

| 감사 ID | 연구 계보 | 독립 감사 종합 판정 | 완료 수준 | 핵심 경계 |
|---|---|---|---|---|
| H01 | 초기 v1–v4 / Stage31·26·51·61·71·73 | VALID_WITH_LIMITATIONS / 일부 CLAIM_OVERSTATED | PARTIAL | 평가 규칙, Stage71 flag·계수, 외부 일반화 |
| H02 | P2 / P3-W7 A0–A3 | VALID_WITH_LIMITATIONS / CLAIM_OVERSTATED | PARTIAL | router만의 독립 효과 식별 불가 |
| H03 | Gen3 grouped gradient/interaction | CLAIM_OVERSTATED / PARTIAL | PARTIAL | group 효과, ineligible loss, 동일 dev 모집단 |
| H04 | O0a/O0b/O0c 관측 | CLAIM_OVERSTATED; O0c 인증 결함 | PARTIAL | terminal evidence와 early precursor 구분 |
| H05 | K0–K3T | VALID_WITH_LIMITATIONS / null 보존 | PARTIAL | K0 측정 퇴화; K3T transport 미입증 |
| H06 | K0 strong-alignment 개입 | CLAIM_OVERSTATED | PARTIAL | correction-specific response; checksum 불일치 |
| H07 | Gen4 R5/R6·Phase F·NAME | VALID_WITH_LIMITATIONS; factor-2 correction | PARTIAL | score sensitivity ≠ entitlement correctness; mediation 미입증 |
| H08 | Gen4 PP3 transport·specificity·necessity·restoration | VALID_WITH_LIMITATIONS | PARTIAL | PP5와 비교한 국소 상대 효과 |
| H09 | Gen4 PP3-excluded residual four-stage | VALID_WITH_LIMITATIONS | PARTIAL | template orientation causality 아님, P4 약한 효과 |
| H10 | Five-scale / core-stable / AVeriTeC | VALID_WITH_LIMITATIONS | PARTIAL | scale law·정답률 향상 미입증 |
| F01 | Gen4 × K directional alignment transport | VALID_WITH_LIMITATIONS | **COMPLETE for permitted static/item audit** | frozen 양의 전이 2개 가설 모두 미충족 |
| F02 | Gen5 A-init/RNG/temporal | VALID_WITH_LIMITATIONS + SCIENTIFIC_DEFECT_VERIFIED | PARTIAL | REFUTE/SUPPORT 이름 반전, class collapse |
| F03 | Gen5 Phase1–3 / Stage E ownership | VALID_WITH_LIMITATIONS | PARTIAL | positive local geometry ≠ learned ownership advantage |
| F04 | Gen5 bypass / Jacobian / quotient | VALID_WITH_LIMITATIONS | PARTIAL | learned CE gain bypass, all-NE, partial image convergence |

**감사 수행 상태:** 요청된 최종 F01–F04 네 영역은 모두 독립 감사를 마쳤다. 이 통합 기록 이전의 H01–H10은 순차적으로 전달된 각 독립 감사 보고서를 바탕으로 한다. `PARTIAL`에 해당하는 원본 실행 재인증은 종료되지 않았다.

---

## 2. 발견·확인된 결함 등록부

| 결함 ID | 증거 계보 | 결함 내용 | 영향받는 원래 주장 | 상태 / 취급 |
|---|---|---|---|---|
| DEF-G5-LABEL-01 | F02 / Gen5 A-init behavioral onset | 실제 numeric order는 `REFUTE=0, NOT_ENTITLED=1, SUPPORT=2`인데 behavioral export의 class_order가 0/2를 반대로 명명 | `SUPPORT → REFUTE → NOT_ENTITLED` vulnerability corridor, class-specific temporal synthesis | **신규 SCIENTIFIC_DEFECT_VERIFIED.** 원래 숫자 기록 수정 없이 논문·해석에서 올바르게 표기. 이름 기반 해석 차단 |
| DEF-GRAD-OWNED-02 | H07/H10 / Gen4 D-half | `partial_grad(x, 0.5)`는 forward 값을 보존하면서 backward Jacobian만 1/2로 축소. 과거 저장된 readout을 ordinary forward derivative로 표현하면 두 배 차이 | unqualified ΔL magnitude, mysterious factor-2/backend defect | **SCIENTIFIC_DEFECT_VERIFIED, 이후 의미 교정됨.** `ΔL_forward=2ΔL_owned` |
| DEF-O0C-NONINTERFERENCE-03 | H04 / O0c | runner가 baseline ordinary forward와 instrumented output을 직접 비교하지 않았는데 production `PASS_EXACT_EQUIVALENCE_NONINTERFERENCE` 인증을 남김 | full noninterference certification | **SCIENTIFIC_DEFECT_VERIFIED for certification.** 측정 tensor 조작/오염 자체는 입증되지 않음 |
| DEF-K0-GEOMETRY-HASH-04 | H06 / K0 strong alignment | frozen geometry-summary의 선언된 SHA256과 실제 artifact SHA256 불일치; runner에서 선언된 상수와 산출 hash 비교 미집행 | 모든 literal provenance gate PASS | **VERIFIED provenance defect.** pinned Git object의 데이터 대체 증거는 없음 |
| DEF-G5-TEMP-MASK-05 | F02 / Gen5 temporal recovery | valid-token masking 문제와 원본 absolute task-row-energy 인증 실패 | t1/t17 earliest local-projector claims | **문서상 알려진 결함, 해당 주장 BLOCKED.** 새로운 감사가 최초 발견한 결함으로 취급하지 않음 |
| DEF-G5-Q22-ORDER-06 | F03 / Phase1B | 이전 layer-17 endpoint는 layer-22 개입의 인과적 downstream endpoint가 아님 | 과거 Q22 잘못된 읽기 | **기존 수정 사항 확인.** corrective commit이 accepted 실행 이전임 |
| DEF-PP3-IEEE-07 | H09 / residual aggregate necessity | cancellation-prone exact-float assert로 앞선 retry 실패 | 실패 retry 정상 실행 취급 여부 | **기존 수정됨.** canonical `QC-QR`로 수정, 실패 시 과학적 산출물 없음 |
| DEF-G5-PROV-08 | F03 / Phase2 training | stale `matrix_provenance.json` checksum 1개 | unqualified manifest PASS | **기존 문서화된 예외**; 최종 provenance digest와 9개 checkpoint hash 일치 |

### DEF-G5-LABEL-01: 필수 정정 내용

- **정식 numeric-to-name mapping:** `0=REFUTE`, `1=NOT_ENTITLED`, `2=SUPPORT`.
- **원본 120개 취약 행:** 전부 numeric gold `0`, `C6_EXPLICIT_DENIAL`.
- **1,080개 행×셀 에피소드:** 예측 숫자 corridor `(0,2,1)`.
- **정정된 이름:** `REFUTE correct → SUPPORT transient decisive error → NOT_ENTITLED`.
- **원래 보고서의 잘못된 이름:** `SUPPORT correct → REFUTE transient error → NOT_ENTITLED`.
- **유지할 값:** 취약 행 120개, 총 1,080개 에피소드, 발생 시점·재수렴, A-init 영향, 순열 불변 geometry와 class-blind wrongness.
- **문서 영향:** `reports/reason_router_gen5_ainit_shared_vulnerability_discrete_evidence_report_candidate.md`, `reports/reason_router_gen5_ainit_temporal_mechanism_static_synthesis_report_candidate.md` 및 behavioral-onset export의 class-specific prose. 원본 frozen artifact는 조용히 수정하지 않는다.
- **승격 차단:** 위 서술이 정확해질 때까지 해당 class-specific narrative를 논문 본문에 사용 금지. 수학적 t/onset 결과 전부를 무효화할 필요는 없다.

---

## 3. 초기 연구 계보 H01–H06

### H01 — 초기 v1–v4 및 Stage lineage

- v1 logit 구조 `[e_neg, (e_pos+e_neg)/2, e_pos]`는 real-valued strict middle winner를 허용하지 않음. 단 float32 tie/argmax에서 NE가 나타날 수 있으므로 “NE가 수학적으로 전혀 불가능”하다는 기술은 부정확. `predict()`와 실제 평가의 argmax 구현 경로 구분 필요.
- v4 `spread_loss = -tanh((between/within).detach())`는 detach 때문에 해당 항에서 gradient 없음. Stage26-H1은 변화 요소와 훈련 길이가 함께 바뀌어 multiplication의 독립 효과 입증 불가.
- Stage71 predicate_swap의 flag는 출력 logit 구성에 직접 개입했으며, 외부 gold-label 직접 leak이라는 표현은 부정확. Frozen coefficient `1.3277653455734253`로 720 NE logits 수치 재현; 제거 시 predicate_swap의 NE 60개 중 2개가 SUPPORT로 전환.
- Stage51/61 dev 720은 동일 집합; Stage71 144 overlap, 나머지 576은 앞선 train에 포함. 따라서 Stage71 internal 702/720, accuracy 0.975, macro-F1 0.964는 독립적인 unseen generalization으로 기술 금지.
- Stage73 VitaminC external 353/1000, accuracy 0.353, macro-F1 0.326. 해당 cohort의 all-SUPPORT baseline accuracy 0.5보다 낮음. **외부 일반화·유용성 주장 불가.**
- v3 no-time-swap negative generator에는 negation이 사라진 900개 confounded rows, 212개 malformed negative rows가 포함. 이를 깨끗한 single-factor 부정 실험으로 승격 금지.

### H02 — P2 / P3-W7

- 12개 cell과 전체 3,600 rows의 수치 재현: train 2,880/dev 720, pair overlap 0. 같은 dev를 세 seed가 반복 사용.
- A1/A3 conditional-router bundle에는 router 구조뿐 아니라 supervision eligibility, reason-specific CE, polarity representation, loss가 함께 달라짐. A0/A2 대비 독립적인 router-only 효과를 식별하지 못함.
- A2 versus A0, A3 versus A1의 gradient-isolation 비교도 전체 ownership bundle 차이를 비교한 것이므로 단일 edge의 순수 효과로 기술 불가.
- Masked arms eligibility: frame 1,409 / predicate 695 / sufficiency 576 / polarity 338. Legacy에는 ineligible train 1,471개에 local BCE가 적용됨. Source generator confound 동반.
- explicit_local은 해당 frozen experiment의 matched-family 성능에 악영향; conditional-router bundle의 A0 대비 macro-F1 이득 입증 실패.

### H03 — Gen3 grouped gradient / interaction

- 90개 prediction export×720 rows의 원래 수치에 대해 별도 독립 감사 수행됨. Frozen 42개 export는 hash-bound; 다른 48개 pair/group export는 재구성 가능하나 개별 binding 일부 불완전.
- A0 accuracy 0.903241 / macro-F1 0.803636; D1 global-half accuracy 0.863889 / macro-F1 0.778960. 111 breaks, 26 repairs.
- `partial_grad(z,λ)=detach(z)+λ(z-detach(z))`: forward-preserving/gradient-scaling. 코드 alias 16개 중 실제 소비되는 14개라는 감사 관찰을 단순히 16개 active gradient edges로 기술하지 않는다.
- Frozen51 새 break 51개 중 일부 confounded, 일부 grammatical·eligible. categorical interaction과 continuous interaction은 동일한 과학적 효과가 아님. Grouped U+D recurrent new case에는 confounded/ineligible example이 존재.
- 세 seed가 동일 dev population에서 평가됨. **Interaction synergy, independent out-of-distribution generalization 과장 금지.**

### H04 — O0a / O0b / O0c

- O0a 18 rows; endpoint deletion effect는 있지만 deletion이 literal “details unavailable”로 대체하고, paraphrase도 claim 변화를 동반. Postblock0 descriptor를 embedding으로 명명한 오류 유의.
- O0b 3 pairs×25 layers, 225 pre-divergence anchors의 동일성이 재현되었으나 신호는 관찰 가능한 토큰 분기 이후에 형성됨. **미래 증거 부족을 선행 예측하는 일반 precursor라는 주장 불가.**
- O0c는 actual selective-SSM `ssm_state` (post recurrence, pre-readout) 관측. Broad early precursor 미입증, terminal-localized separation만 관측적 유효. 24 mixers / `[1,1536,16]` state.
- O0c의 `PASS_EXACT_EQUIVALENCE_NONINTERFERENCE` production 인증은 baseline noninstrumented output 비교가 없는 상태에서 기록되었음. 별도 실행 결함과 관측값 오염을 혼동 금지.

### H05 — K0–K3T

- K0-P1/P2: matched/swapped 336개 중 335개에서 상태/지표 동일; token-history 측정 퇴화가 유력. 원본 tokenizer byte가 미인증이므로 최초 원인까지 확정 불가. Dynamics가 본질적으로 존재하지 않는다는 반증은 아님.
- K1: prestate matching 유효 graph edge 0; 과학적 궤적 가설은 테스트되지 않음.
- K2/K2W: 조건 적격 구성 실패로 궤적 평가 미실행.
- K2S: R/D 양방향, P 음방향 혼합 결과.
- K2R: 네 사전 정의 endpoint의 같은-generator/claim-disjoint 복제 확인. 새로운 독립 자연 분포 일반화는 아님.
- K3 coefficient-specialization 사전 가설과 반대 결과. K3C 필수 BASE R gate 실패; 메커니즘 승격 금지.
- K3T: 336 items / 168 reciprocal blocks, R signs 81 positive / 72 negative / 15 zero; two-sided sign p=0.51792103. **positive transport 미입증**, negative effect 확정도 아님.

### H06 — K0 strong-side alignment 개입

- Population common-330, k=2, layer-22 x-branch strong-240 intervention.
- Independent stored-item reconstruction: `R_SA=0.1281180444690374`, `R_SM=0.0984773578406091`, `R_WA=0.1289115368396328`, `R_WM=0.0993457325816154`, `F_SA=0.0234938241318631`.
- Alignment intervention 270/330 state decreases; magnitude 194/330. Alignment correction mean L2 0.310612; magnitude control 0.236707, 약 31.2% 더 큼. Alignment mean gain 우위가 unique alignment causality라는 결론은 식별되지 않음.
- Original Outcome-A inequalities 숫자는 충족. **상태·write가 해당 representation-level correction에 반응했다는 좁은 결과만 조건부 유지.**
- Frozen geometry-summary declared SHA `35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9a51467477163` vs actual `35db1428f5eab2aac50a1cdf26f2ccdc88502434ad718b7a2b9ea9ef66a80798`. 언급된 상수 비교 강제하지 않음. 데이터 변조 단정 금지.

---

## 4. Gen4 H07–H10

### H07 — R5/R6 semantic axes, Phase F, NAME

- R5 원본 `300 pairs × 6 cells × 18 evaluators = 32,400` evaluator-row observations. 18 evaluators는 공유 backbone/훈련 조건 때문에 독립 model-replication 18개가 아님.
- R6 Holm-6: NAME mean −0.01967010 (p_adj≈8.50e−24), ROLE +0.01051709 (0.0414693), PREDICATE +0.04420943 (≈1.01e−13), TITLE×NAME +0.00919858 (≈8.01e−5), TITLE−NAME +0.01505127 (≈0.000425732); TITLE −0.00461883 (0.239219) 비유의. 5/6 supported.
- 문자열 field는 정확히 치환됐으나 TITLE/ROLE/NAME/PREDICATE가 서로 논리적으로 독립적인 entitlement axis라는 증거는 아님. `q_authorized=frame_prob×predicate_coverage_prob×sufficiency_prob` score sensitivity ≠ correctness.
- Phase F 원래 15 hypotheses 중 5 Holm-supported; native-state speed/turning/path-efficiency는 같은 trajectory로부터 유도되어 수학적으로 결합됨. 원본 `support_states.npy` 부재로 raw vector → endpoint 재계산은 불가.
- Q1/Q3: layer 17 NAME path efficiency mean −0.013998957, Holm-6 p≈2.85351e−7. Direct layer17-vs-11 NAME path-efficiency p_adj≈1.02106e−7. **해당 직접 차이만 입증**; layer17 보편적 최고/유일 효과는 아님.
- NAME 후속 선택은 Phase F 뒤의 adaptive follow-up. 해당 후속 검정의 layer 5/17 규칙은 이후 endpoint 보기 전에 동결됨. 프로그램 전체 adaptive family FWER 별도 제어 없음.
- D-half factor-2는 설계상 gradient scaling. Ordinary forward derivative로 기술하려면 `2×owned`; t/sign/rank 불변.

### H08 — PP3 causal comparisons (XG1 001–1200)

| Study (300 pairs each) | Estimand | Mean | Original one-sided p | 지지되는 범위 |
|---|---|---:|---:|---|
| Transport 001–300 | `C_PP3` | 7.120276193e−8 | 1.519410e−53 | frozen PP3 susceptibility contrast XG1 유지 |
| Specificity 301–600 | `C_PP3−C_PP5` | 2.304067369e−8 | 1.336688e−13 | 사전 지정 PP5 대비 우위 |
| Necessity 601–900 | `Q5−Q3` | 4.741437112e−8 | 6.224276e−72 | PP3 제거가 equal-norm PP5 control보다 더 감쇠 |
| Restoration 901–1200 | `Q_R3−Q_R5` | 4.078872357e−8 | 9.010902e−58 | native R3가 PP5 replacement보다 큼 |

- `B=h−c3; R3=B+c3=h`는 항등식. 실증 정보는 PP5 대체 대비 차이.
- Necessity PP3 제거 후 `Q3≈1.0218e−7`로 현상 남음; 약 45.3% 감쇠. 따라서 strict necessity 불가.
- 대조군은 prescribed correction norm을 같게 하지만 native-state 거리·방향·gain까지 matching하지는 않음.
- 네 XG1 cohort는 pair/claim/evidence exact overlap 0이나 공통 generator 144 lexical configuration 반복. 실험 네 개가 독립 분포 네 개는 아님. Sequential follow-up은 각 단계에서 prespecified, program-wide FWER 미입증.

### H09 — PP3-excluded P1/P2/P4/P5 residual

| Stage | Cohort | Reproduced result |
|---|---|---|
| A template transport | 1201–1500 | `mean D_TEMPLATE=0.585312579336`, p≈2.2607e−90; XG2-like cosine profile > XG4-like |
| B aggregate necessity | 1501–1800 | `mean(QC−QR)=4.92201944615e−8`, p≈6.4276e−69; `mean QR≈1.20848e−7>0` |
| C individual necessity | 1801–2100 | P1,P2,P4,P5 all Holm-4 reject under frozen comparison |
| D individual restoration | 2101–2400 | P1,P2,P4,P5 all Holm-4 reject under frozen comparison |

- P4 individual necessity: mean control-adjusted `3.17823034e−9`, Holm p=0.0181302; native attenuation `1.85403493e−9` = original Q의 약 0.98%. 작은 조건부 효과로 표기.
- P4 restoration mean contrast `2.81379683e−9`, Holm p≈8.05203e−5.
- Quarter-turn controls는 intervention norm/orthogonality를 맞춘 상대 비교. 원래 상태와의 거리 또는 후속 nonlinear gain이 동일하다는 보장은 없음.
- Restoration `B_k=h−c_k`, `R_k=h`, `C_k=h−c_k+r_k`; original 상태 복원은 항등식, empirical contrast는 native versus within-plane quarter-turn replacement.
- Template transport만으로 XG2-like orientation 자체의 causality 입증 금지. 네 plane 효과를 additive sum으로 추정 금지.
- C/D 원본 compressed JSONL이 독립 decompress + SHA 인증됨; checkpoint/runner logs/raw recurrent tensors의 역사적 실행 full proof는 불충분.

### H10 — Five-scale / external-transfer

| Scale | Selected/control | `mean ΔL_owned` | `mean ΔL_forward` | positive pairs |
|---|---|---:|---:|---:|
| 130M | P3/P5 | +0.00112036604 | +0.00224073207 | 53.67% |
| 370M | P3/P5 | +0.000508956352 | +0.00101791270 | 57.67% |
| 790M | P2/P5 | +0.00227671145 | +0.00455342290 | 73.67% |
| 1.4B | P5/P4 | −0.00119544751 | −0.00239089501 | 35.33% |
| 2.8B | P3/P5 | −0.0000741749535 | −0.000148349907 | 55.33% |

- Positive mean sign vector `+,+,+,−,−`는 기술적으로 재현됨; 2.8B median과 다수 부호는 positive. Continuous threshold, homogeneous negative >1B regime, monotonic law 미입증.
- Core discovery/confirmation 370M, 790M, 1.4B, 2.8B에서 scale-local winner confirmed. 130M은 response-blind geometric homolog로 발견 절차가 동일한 다섯 번째 반복은 아님. 370M의 original **joint core+positive residual criterion 실패**는 반드시 유지.
- Trained-router readout의 부호 vector와 Vanilla LM functional-control 부호 vector가 반대(`−,−,−,+,+`). 특정 objective를 제거한 본질적 Mamba 부호반전이라는 주장은 불성립.
- Behavioral bridge: 370M positive mean `D_BEH=+0.00112068570` (Holm p≈1.49572e−8), 1.4B negative mean `−0.00201288531` (positive test 비유의). 양쪽에서 argmax 개선은 매우 희소.
- AVeriTeC 462 official dev source, 305 Refuted / 122 Supported / 35 NE; 38 conflicting-evidence cases 제외. Evidence 339/462 (73.38%)가 64-token budget에서 truncation. Gold label과 잘린 evidence의 실제 entitlement를 동일시하면 안 됨.
- 130M/370M frozen margin contrast positive Holm-supported, 1.4B negative-sign test p≈0.075667로 실패. Native/control prediction flips **0/462 at all three**; native correct 83/462,45/462,60/462. 따라서 “AVeriTeC 정확도 개선” 주장과 혼동 금지.
- 다섯 크기의 selected/control plane 및 XG1 cohort가 동일하지 않으므로 simple causal scaling-law 불가.

---

## 5. 최종 독립 감사 F01 — Gen4 × K negative transport

**원본 핵심 기록:** `reports/reason_router_gen4_k_directional_alignment_transport_backend_invariant_closure_report_candidate.md`; CPU execution `8496ece911e0d461f0abbdf1a0fa619f8a2f22ab`; CUDA `60f7485d37a7afb848b11b31a30242f5b89db534`; closure `ad375c88385eef507e151f95717e951919a6a3fe`.

- `C2_NAME−C0_SHAM`, reference `C5_TITLE_NAME−C1_TITLE`; layers `15→16→17`; intervention A_IDENTITY+2, layer17 x-branch before convolution, strong-395.
- Two registered positive tests, Holm-2:
  - `R_ALIGN`: mean `−7.34395589648736e−6`, t(299) `−1.3822135`, one-sided p `0.91603110137`, adjusted p=1.
  - `R_ALIGN−R_MAG`: mean `−2.888124477688825e−5`, t(299) `−2.9997861`, one-sided p `0.99853550555`, adjusted p=1.
- Baseline mean `−0.013998957453394283`, saved itemwise baseline residual zero. CPU and CUDA endpoints differ at most `1.6212262369252883e−6` < frozen `1e−4` tolerance.
- ZIP original-byte hashes: CPU `25a9e6a862f7c1ad7c272d85cb5000ba8542cca2c8976b785021e5e4159ebcaf`; CUDA `816cc586f8c6a1eed8e730937f30032befbcb6b40b59bd56b4d763a0cc2177c5`.
- **판정:** registered positive transport **NOT ESTABLISHED**. CPU/CUDA는 동일 population의 backend 비교이지 science sample 두 개가 아님. 인과 alignment 효과가 존재할 수 있는 모든 조건을 반증하지 않음.
- 역사적인 manual Kaggle 기록에는 canonical `cm run→collect→import` 이력 부재. Original raw vectors 부재; 현재 감사 COMPLETE는 허용된 static/item 범위에 한정.

## 6. 최종 독립 감사 F02 — Gen5 A-init / RNG / temporal

**증거 식별자:** 아래 §10 출처 인덱스의 I1–I11.

- 3×3 A-init × training RNG seeds `{6201,6202,6203}`; parent frozen, B0=0; 3360 train/840 dev, fixed 20 steps, dev-based checkpoint selection 없음.
- `row(A)` final same-A/different-RNG affinity `0.999963824427`, same-RNG/different-A `0.144047932169`. Dominant-right cos `0.999984480062` versus `0.535662952798`. **finite factorial 내 initialization-driven geometry**로 해석.
- 9개 cell 모두 dev `REFUTE/NE/SUPPORT=120/600/120`, prediction `0/840/0`, accuracy `600/840=0.714285714286`. Parent 대비 835/840 prediction changes. Same-A functional cosine `0.999999957562`, same-RNG/different-A `0.999917998276`; 거의 같은 붕괴된 예측을 올바른 추론의 등가성으로 승격 금지.
- t0 B=0, BA=0, grad_A=0; 첫 AdamW update의 B가 초기 nonzero operator 설명. Operator sq distance different-A same-RNG t1 `0.03201217973478823`, same-A diff-RNG `0.00025299180616308176`. A에 따른 discrete divergence t1, RNG divergence t2, decisive-error first t4, all-cell prediction reconvergence t17–t20.
- **확정 DEF-G5-LABEL-01:** 120 `C6_EXPLICIT_DENIAL`, numeric gold 0, all 1080 episode `(0,2,1)`. Correct names `REFUTE→SUPPORT→NE`.
- Earliest tested correction endpoint에는 task-visible readout이 존재하나 frozen upstream 0–21 layers의 prebehavior precursor는 확립되지 않음. Partial recovery의 task-row-energy와 masking 문제로 t1/t17 projector origin 주장은 BLOCKED.
- Visible/complement pooled k=8 ratios: visible `1.012775249547`, complement `0.001068684348`, interaction `0.000284078445`. Frozen finite dev chord 결과로만 해석, global exact null/gauge 아님.
- M7 finite-grid recipient-A / donor-A-history / donor-RNG logit effect magnitudes `0.0428053 / 0.0185676 / 0.000154306`, 15,120 hybrid rows는 독립 15,120 sampling units 아님.
- M8A GL(2) coordinate-law의 ambient subspace 설명은 false under test; 모든 일반 representational equivalence가 반증된 것은 아님.
- One-shot protected holdout의 원시 row/tensor는 본 감사에서 열지 않음; 관련 독립 실행 authentication `UNVERIFIED`.

## 7. 최종 독립 감사 F03 — Gen5 Phase1–3 / ownership / Stage E

**증거 식별자:** 아래 §10의 O0–O8.

- R22/C22: R22는 upstream-role native-write difference로 만든 rank-2 construction; C22는 R22를 제외한 centered variation의 response-blind control. 둘 다 내재적 의미론적 소유권자가 아님.
- Population construction `7801–8100`; necessity `8101–8400`; restoration `8401–8700`; ownership assay `8701–9000`, 각각 300 pairs.
- Phase1B: `D_NEC22` mean `5.687471435811073e−8`, t(299) `13.48736`, one-sided p `5.19901917e−33`; `D_SUF22` mean `1.3295631928032442e−7`, t(299) `16.82532`, p `1.83816071e−45`. Local C22-matched comparison으로 유지.
- Phase2 correction arms: C0 unrestricted, C1 correction-C22 write disabled, M1 correction-R22 write disabled. Frozen branch 그대로; trainable parameters A/B만.
- Assay frozen 3-seed item averages: `mean_Ibar_M1=1.3327600706416034e−7`, `mean_D_OWN=9.593937005209078e−12`, t(299) `0.34010568955`, one-sided p `0.3670078706795085`. **Ownership advantage NOT ESTABLISHED**; M1과 C1의 equivalence도 증명하지 않음.
- Phase3A 3 seeds × P0/PR/PC: every seed PR G1–G4 fails, G5–G7 pass; Phase3B **NOT_EXECUTED/BLOCKED**. 학습 손실 감소를 contention 성공으로 해석 금지.
- Stage E mean recovery fractions E-R22 `0.002139345066`, E-C22 `0.001860633474`, E-BFREE `0.071425088723`, BFREE-AINIT `0.133576230985`, AFIX-MONLY `0.062988442340`, AFIX-MONLY-SCALEMATCH `0.990747485451`. SCALEMATCH는 이미 학습된 unrestricted solution에서 조건화된 subspace를 사용하므로 scratch learning의 증거가 아님.
- Construction native vectors unavailable, historical GPU 실행 full provenance reexecution 금지로 complete 인증은 PARTIAL.

## 8. 최종 독립 감사 F04 — Gen5 bypass / downstream function

**증거 식별자:** 아래 §10의 B1–B7.

- Algebraic `BA=(P_R B)A+((I-P_R)B)A`는 성립하나 nonlinear task logit 가산성까지 함의하지 않음.
- Stage B 30,240 saved rows, 9 cells×4 conditions×840 dev: FULL mean CE gain `0.5026657`; R22_ONLY gain `0.00004422`; R22_REMOVED gain-retention `0.9998664`, range `0.9998471–0.9998946`. **CE 개선 약 99.9866%가 R22 없이 유지.** FULL 및 R22_REMOVED는 모든 cell에서 ALL-NOT_ENTITLED 예측.
- Stage C step0 mean gradient-energy R22 fraction `7.99510772e−5`, C22 fraction `1.07416577e−4`; gradient/R22 affinity `0.000591579`, gradient/final B `0.0316953`. 초기 gradient의 low R22 alignment 관측적 유효; 전체 optimization trajectory 증명 아님.
- Stage D mean downstream plane affinity: final hidden `0.1602187283`, task representation `0.3161430536`, decision primitives `0.7101025327`; principal-angle pairs 각각 약 `58.07°/81.06°`, `41.35°/76.88°`, `1.83°/50.88°`. **부분적 anisotropic convergence**, full rank-two equivalence 아님.
- Forward-Jacobian recovery: edge/joint gradient norm 약 0.5, cosine≈1, fitting residual≈`1.93e−8`. 옛 edge readout magnitude는 보정 필요; 복구한 범위의 normalized geometry는 유지.
- Finite task-reachable quotient: Gram participation ratio dimension `3.585322797137`, ambient residual `1.04816095366`, task-restricted residual `0.459227092986`, suppression ratio `0.438154063771`. **incomplete** suppression, exact operator quotient equality 아님.
- Downstream residual chain `0.459226→0.235283→0.201189→0.139629→0.155009→0.011953`는 원래 saved cross-Gram 수치의 기술적 재현. Semantics의 유실이나 reconvergence 원인 입증 금지.
- Primitive Stage D image matrices, 일부 Jacobian 원자료 미보존: 숫자와 source consistency는 확인 가능하지만 모든 원시 변환 재생성 인증은 불가.

---

## 9. 논문 주장 disposition: KEEP / CORRECT / DROP / APPENDIX

| Claim group | 처리 | 허용 문장·제한 |
|---|---|---|
| Gen4 PP3 control-relative susceptibility | KEEP (bounded) | 사전 동결된 PP5 대비 internal layer17/target-token sensitivity/attenuation/restoration |
| PP3-excluded residual four-plane effects | KEEP (bounded) | 지정한 quarter-turn control과 frozen Q에서 P1/P2/P4/P5의 작은/큰 상대 차이, causal template claim 제외 |
| R6 semantic-axis q_authorized | KEEP (bounded) | field substitution의 output-score sensitivity; entitlement correctness/semantic exclusion claim 제외 |
| Phase F / NAME Q3 | KEEP (bounded) | local state kinematics 및 정확히 검정된 17-vs-11 contrast; mediation 제외 |
| Gen4×K positive transport | DROP | 두 confirmatory positive gate 실패. 이 부정적 결과는 결과 섹션에 KEEP |
| Five-scale readout signs | CORRECT | owned-gradient semantics와 `×2` forward conversion 표시; sampled-grid only |
| Core-stable / residual-plastic | KEEP (qualified) | scale-local procedure role과 기술적 residual reorganization; original 370M joint failure 보존 |
| 370M behavioral positive / 1.4B negative | KEEP (bounded) | margin-level scale-specific evidence. Uniform transfer 금지 |
| AVeriTeC 130/370 margin effects | KEEP (bounded/appendix) | zero control/native prediction flips, high truncation, no accuracy improvement |
| Gen5 A-init geometry & temporal update | KEEP (bounded) | finite seed/RNG grid의 operator geometry, B1 first-update birth |
| Gen5 shared vulnerability named corridor | **CORRECT / quarantine old naming** | `REFUTE→SUPPORT→NE`, 120 C6 explicit-denial items |
| Gen5 learned R22 ownership superiority | DROP | Phase2 p≈0.367, Phase3A unqualified, Phase3B not run |
| Gen5 functionally equivalent correct reasoning | DROP | all-NE collapse invalidates “correct reasoning” reading |
| Gen5 learned correction bypass / downstream image | KEEP (bounded) | CE gain bypass, partial downstream image convergence; general mechanism equivalence 금지 |
| O0 broad early hallucination precursor | DROP | small terminal/path-conditional state observations만 유지 |
| Strong alignment uniquely causes K0 mechanism | CORRECT | specific activation correction response, unmatched magnitude and checksum qualification |
| Universal scaling law / 1B threshold | DROP | five sampled scales, different planes/populations/objectives |
| Random null/global gauge/semantic owner | DROP | task-restricted finite effects not exact identity or universal semantic role |

**Paper-facing stop rule:** 통합 감사 원장의 `KEEP`도 독립적 모델 실행 인증이 아닌 한 본문에서 “fully verified end-to-end”로 기술하지 않는다. 강한 “causes correctness”, “hallucination detector”, “universal mechanism”, “learned semantic owner” 같은 표현은 현재 증거로 승인되지 않는다.

## 10. 최종 네 감사 원본 출처/claim 인덱스

아래 alias는 마지막 네 감사가 실제 참조한 원본 commit + report/artifact 경로다. 위치는 역사적 Windows worktree가 아니라 **repository-relative path**로 정리했다. 데이터의 SHA256 및 실행-bound hash는 원본 보고서/manifest를 우선한다.

### F01 source

| Alias | Full commit | Repository-relative source |
|---|---|---|
| G4 CPU | `8496ece911e0d461f0abbdf1a0fa619f8a2f22ab` | `scripts/reason_router_gen4_k_directional_alignment_transport_runner.py` |
| G4 CUDA | `60f7485d37a7afb848b11b31a30242f5b89db534` | `scripts/reason_router_gen4_k_directional_alignment_transport_core.py` / CUDA handoff |
| G4 closure | `ad375c88385eef507e151f95717e951919a6a3fe` | `reports/reason_router_gen4_k_directional_alignment_transport_backend_invariant_closure_report_candidate.md` |

### F02 source

| Alias | Full commit | Repository-relative source |
|---|---|---|
| I1 | `87f82551c721f953f710cd5dc102aca23161c4e7` | `scripts/train_reason_router_gen5_ainit_rng_causal_intervention.py`; `reports/reason_router_gen5_ainit_rng_causal_geometry_validated_evidence_report_candidate.md` |
| I2 | `17a783c61e310d6298d4105233ea3f1b71ec8add` | `reports/reason_router_gen5_ainit_rng_functional_fingerprint_runs/gen5-ainit-rng-functional-fingerprint-17a783c-r1/cell_fingerprints.pt` |
| I3 | `d940e19d497f85315155e52818c4a78a1c1c38e0` | `reports/reason_router_gen5_ainit_temporal_birth_replay_runs/gen5-ainit-temporal-birth-phase-a-numerical-auth-d940e19-r1/temporal_birth_trajectory.pt` |
| I4 | `1633d67146a7f049c6ecf28b333f53fdf79ad00e` | `reports/reason_router_gen5_ainit_behavioral_onset_runs/gen5-ainit-behavioral-onset-1633d67-r1/behavioral_onset_cell_metrics.json` |
| I5a | `07cf87b90fc94bd090692e414995eb9923d8b9ab` | `reports/reason_router_gen5_ainit_shared_vulnerability_discrete_evidence_report_candidate.md` |
| I5b | `4841ed1d106f625e690c9cd1ae021ab6040f7afc` | `reports/reason_router_gen5_ainit_temporal_mechanism_static_synthesis_report_candidate.md` |
| I6 | `e2c563188e9534e898c6ea944c5a05f4056b2fe3` | `reports/reason_router_gen5_ainit_internal_task_visible_precursor_localization_validated_evidence_report_candidate.md` |
| I7a | `21d7b514f7b88617cff3b679fb97ecaf6c5411ec` | `reports/reason_router_gen5_ainit_temporal_birth_recovered_partial_runs/gen5-ainit-temporal-birth-phase-b-21d7b51-r1-recovered-partial-v1/RECOVERED_PARTIAL_EVIDENCE.md` |
| I7b | `d9b790b62f8db9f875cfa13d7ba155ac75137a2b` | `reports/reason_router_gen5_ainit_temporal_mechanism_partial_recovery_validated_evidence_report_candidate.md` |
| I8a | `cc2cd80990f4ef8ae8e2a0350ebaafe3195b93b2` | `reports/reason_router_gen5_ainit_task_visible_projection_clustering_null_complement_validated_evidence_report_candidate.md` |
| I8b | `4a973b0725b8fe9eebc8c07950457e58869e10b3` | `reports/reason_router_gen5_ainit_visible_complement_causal_intervention_validated_evidence_report_candidate.md` |
| I9 | `9cced33e32c4a5f84a1b5c6d3fa38ea85d45360f` | `reports/reason_router_gen5_ainit_visible_complement_one_shot_confirmatory_validated_evidence_report_candidate.md` |
| I10 | `c95194f1169ae75db242bdb751bd98649486c95f` | `reports/reason_router_gen5_ainit_t1_factor_swap_runs/gen5-m7-t1-factor-swap-c95194f-r1/factor_swap_logits.pt` |
| I11 | `b0853fd5fef64dafffc55ad3e7b6a5067bf54b4d` | `reports/reason_router_gen5_m8a_latent_alignment_quotient_validated_evidence_report_candidate.md`; `scripts/audit_reason_router_gen5_m8a_latent_alignment_quotient.py` |

### F03 source

| Alias | Full commit | Repository-relative source |
|---|---|---|
| O0 | `25a4206ad64ae64cc942bacdac92b9a300d4f8c3` | `reports/reason_router_gen5_causal_role_grounded_state_ownership_phase0_scientific_spec_candidate.md` |
| O1 | `c9eca380dba9974c02bf272f30228befa7285090` | `reports/reason_router_gen5_phase1b_r22_c22_construction_c9eca38_v1/construction_summary.json` |
| O2 | `5075d862c69ca406b12c30f9ca438c3203aef2a2` | `reports/reason_router_gen5_phase1b_q22_cuda_equivalence_5075d86_r3/q22_cuda_equivalence_summary.json` |
| O3 | `437187c495d031d7d6a9dd6ecc7f9d95ceffd076` | `reports/reason_router_gen5_phase1b_r22_local_necessity_full_cuda_437187c_retry2/r22_local_necessity_items.jsonl` |
| O4 | `7551a14bffe4e1534b9e64c7ad20fff45dd054e7` | `reports/reason_router_gen5_phase1b_r22_restoration_full_cuda_7551a14/r22_restoration_items.jsonl` |
| O5 training | `acdf2ee8940070a6e60190a1c2fcbede8933e419` | `scripts/train_reason_router_gen5_phase2_state_update_ownership.py` |
| O5 assay | `22ac4099526893fdd40e7525efe599bcc5194b0d` | `reports/reason_router_gen5_phase2_fresh_ownership_assay_22ac409_r1_validated_scientific_interpretation_report_candidate.md` |
| O6 | `d58e89477fe43d0e5fa6aaaa7cec31d8c78cda4e` | `reports/reason_router_gen5_phase3a_contention_qualification_retry3_validated_scientific_interpretation_report_candidate.md` |
| O7 fixed | `fdbecf39c8ccf757c4e3afbd5df62c22c74e7f1c` | `reports/reason_router_gen5_stage_e_learned_b_plane_afix_monly_validated_evidence_report_candidate.md` |
| O7 BFREE | `467148d074f7cbd91a5e107a520df21ed4fe508c` | `reports/reason_router_gen5_stage_e_learned_b_plane_positive_control_validated_evidence_report_candidate.md` |
| O8 AINIT | `51bfc4d1d2e1d3fae970e7c09a75cf3aa72cf488` | `reports/reason_router_gen5_stage_e_learned_b_plane_ainit_validated_evidence_report_candidate.md` |
| O8 AFIX | `094fecbcf63c9dfece06df3c6c7a818abe1f9172` | `reports/reason_router_gen5_stage_e_learned_b_plane_afix_monly_validated_evidence_report_candidate.md` |
| O8 SCALEMATCH | `a894416889c4333c591efa987239915ac8160335` | `reports/reason_router_gen5_stage_e_afix_monly_scalematch_validated_evidence_report_candidate.md` |

### F04 source

| Alias | Full commit | Repository-relative source |
|---|---|---|
| B1 | `ae61d962443cb374086420c62b33dd86441b4d8f` | `reports/reason_router_gen5_optimization_path_bypass_stage_a_static_geometry_interpretation_report_candidate.md` |
| B2 | `f70c0b265b2bba8c2a6728f3f96f8c831c805346` | `reports/reason_router_gen5_optimization_path_bypass_stage_b_runs/gen5-stageb-functional-decomposition-f70c0b2-r2/functional_decomposition_rows.jsonl` |
| B3 | `b4439f05a31e6f554e0a99b990ce9222d50afc61` | `reports/reason_router_gen5_optimization_path_bypass_stage_c_runs/gen5-stagec-step0-gradient-geometry-b4439f0-r1/step0_gradient_B_tensors.pt` |
| B4 | `76bef94f642e9c69355bcf0d336a79fff6372266` | `reports/reason_router_gen5_optimization_path_bypass_stage_d_runs/gen5-staged-downstream-image-equivalence-76bef94-r1/downstream_image_rows.jsonl` |
| B5 | `1c9d650c4cc06169c221c1525e93610ba495bdcb` | `reports/reason_router_gen5_forward_jacobian_recovery_runs/gen5-forward-jacobian-recovery-1c9d650-r4/recovered_task_sensitive_subspace.pt` |
| B6 | `c731270221c4e0e131fb68173f40bf0ad8a2bfdd` | `reports/reason_router_gen5_task_reachable_operator_quotient_runs/gen5-task-reachable-operator-quotient-c731270-r1/task_state_gram.pt` |
| B7 | `69473ac76629ef1e7c78b45c8c802a89e1e9695b` | `reports/reason_router_gen5_correction_residual_localization_runs/gen5-correction-residual-localization-69473ac-r1/stage_inner_product_grams.pt` |

## 11. 마지막 네 감사의 claim register

| ID | Source | Scientific estimand | Verdict | 보존할 정확한 결론 / 미해결 범위 |
|---|---|---|---|---|
| G4-01 | G4 CPU | mean positive `R_ALIGN` | VALIDATED | positive transport criterion fails; not global null |
| G4-02 | G4 CPU | mean `R_ALIGN−R_MAG` | VALIDATED | specificity criterion fails under frozen controls |
| G4-03 | G4 CPU/CUDA | backend item agreement | VALID_WITH_LIMITATIONS | negative conclusion matches within tolerance, runtime not rerun |
| I-01 | I1 | A-init/RNG geometry factorial | VALID_WITH_LIMITATIONS | finite-grid A-init geometric representative dependence |
| I-02 | I2 | nine-cell output fingerprints | VALIDATED | nearly identical ALL-NE outputs, not correct reasoning |
| I-03 | I3 | B1 first-update operator birth | VALIDATED | t1 first nonzero effective correction explained |
| I-04 | I4 | temporal separation/reconvergence | VALIDATED | A t1 divergence, t17+ argmax reconvergence |
| I-05 | I4/I5 | named vulnerability corridor | CONTRADICTED_BY_EVIDENCE | `REFUTE→SUPPORT→NE` correct names; old prose wrong |
| I-06 | I6/I7 | earliest prior task-visible precursor | UNVERIFIED | endpoint exists, prebehavior claim blocked |
| I-07 | I8 | finite visible/complement perturbations | VALID_WITH_LIMITATIONS | dev finite-chord task visibility, not global gauge |
| I-08 | I9 | one-shot holdout replication | UNVERIFIED | protocol inspected, protected primitive rows not audited |
| I-09 | I10 | recipient-A/donor-A/RNG grid | VALID_WITH_LIMITATIONS | finite grid factor effects, repeated dependencies |
| I-10 | I11 | GL(2) coordinate-law account | VALIDATED negative | proposed ambient alignment law fails |
| O-01 | O0/O1 | R22/C22 constructed roles | VALID_WITH_LIMITATIONS | authenticated rank-2 bases, original construction vectors absent |
| O-02 | O2 | Q22 backend equivalence | VALID_WITH_LIMITATIONS | frozen gate-level equivalence only |
| O-03 | O3 | relative R22 necessity | VALID_WITH_LIMITATIONS | matched C22 local endpoint attenuation |
| O-04 | O4 | relative R22 restoration | VALID_WITH_LIMITATIONS | matched additive donor comparison |
| O-05 | O5 | M1−C1 advantage | VALIDATED negative | p≈0.367, advantage NOT ESTABLISHED |
| O-06 | O6 | Phase3A contention qualification | VALIDATED negative | every seed fails gates 1–4 |
| O-07 | O6 | Phase3B qualified contention outcome | NOT_EXECUTED | eligibility blocked, no Phase3B outcome |
| O-08 | O7/O8 | Stage E recovery/optimization accessibility | VALID_WITH_LIMITATIONS | SCALEMATCH oracle-conditioned, no scratch ownership |
| B-01 | B1/O6 | seed-pressure geometry | VALID_WITH_LIMITATIONS | A-init dominance, weak pressure redirection |
| B-02 | B2 | full vs R22-removed CE gain | VALIDATED | ~99.9866% gain retained, all-NE |
| B-03 | B3 | step0 gradient R22 alignment | VALID_WITH_LIMITATIONS | low initial projected gradient energy |
| B-04 | B4 | downstream image affinity/angles | VALID_WITH_LIMITATIONS | anisotropic partial convergence only |
| B-05 | B5 | edge gradient vs forward Jacobian | VALID_WITH_LIMITATIONS | absolute scale correction, subset geometry stable |
| B-06 | B6 | task-restricted operator discrepancy | VALIDATED | strong but incomplete suppression |
| B-07 | B7 | propagation chain cross-Grams | VALID_WITH_LIMITATIONS | downstream suppression descriptive, no semantic cause |

## 12. 미해결 증거와 적절한 종료 경계

### 12.1 인증 부족으로 `PARTIAL` 유지

- 원본 O0c noninstrumented/instrumented production output pair 미보존.
- K0 최초 tokenizer exact bytes 및 frozen pinned runtime 일부 부재.
- Gen4 일부 raw selective-SSM state tensor 및 evaluator checkpoint / original dependency files 부재.
- PP3·residual 계보 일부 raw tensors, execution logs, checkpoint actual-loading traces 부재.
- AVeriTeC upstream pinned raw-byte checksum 전체 재계산 실패 및 다수 evidence truncation.
- Gen5 R22/C22 최초 native construction matrix와 Stage D raw downstream image matrices 부재.
- Protected one-shot holdout의 primitive rows/tensors는 감사 제한에 따라 접근하지 않음.
- 이 원장에 요약된 H01–H10 개별 원본 Codex 채팅/첨부 파일 모두에 대한 permanent SHA catalog는 아직 확보되지 않음. **따라서 원장 기록을 원본 1차 측정 전체의 대체물로 쓰지 않는다.**

### 12.2 새 실행 없이 종료 가능한 부정적 결과

- Stage73 external generalization 취약.
- Hard ownership explicit_local 성능 악화.
- K1/K2 population support 실패, K3C gate 실패, K3T positive transport 미입증.
- Gen4×K positive alignment transport 미입증.
- 370M original joint core+positive-residual criterion 미충족.
- 1.4B positive behavioral bridge 미입증.
- 1.4B AVeriTeC negative-sign transfer 미입증.
- Gen5 M1 ownership advantage 미입증.
- Gen5 Phase3A contention qualification 실패, Phase3B NOT_EXECUTED.
- Gen5 latent coordinate-law explanation 실패.
- Gen5 functional classification collapse (all-NE), learned CE gain largely outside R22.

### 12.3 남은 과학적 작업과 혼동 금지

- 원본 checkpoint/raw tensor의 **부재를 메우는 새로운 학습/평가를 자동 승인하지 않는다**.
- 이미 실패한 frozen hypothesis를 새 subgroup/sweep/post-hoc gate로 구제하지 않는다.
- 감사 원장과 증거 기반 코드/논문 수정은 별도 authorization을 요구한다. 이 문서는 frozen study authority를 변경하지 않는다.
- 새 벤치마크, Kaggle, GPU 실행 필요 없음. 현재 단계는 감사 기록 보존과 제한된 claim 정정의 검토다.

## 13. 감사 원장의 보존·검토 계약

**권장 repo-relative destination:** `reports/scientific_audit_20261016_consolidated_ledger.md`.

- 다른 frozen 보고서나 원본 `.jsonl/.pt/.zip` 파일은 변경하지 않는다.
- 이 보고서를 repository에 복사하기 전 현재 `cm context`와 target path 존재 여부를 확인한다.
- 미추적 `audits/`와 다른 local changes를 건드리지 않는다.
- 텍스트/일부 path 표기가 현재 브랜치와 다르면 사전 수정 아닌 **검증 가능한 차이**로 제시한다.
- `cm ship` 및 명시적 파일 staging 이후에만 사용자가 수동 commit/push 여부를 최종 결정한다. Detached HEAD에서 무조건 push하지 않는다.
- Ledger 검증은 최소 UTF-8, Markdown 링크/파싱, full commit SHA 길이, 원본 증거 path 존재, 수치·용어 consistency, 미해결 인증 표시를 확인한다.
- **본 문서는 증거를 정리한 고정 시점의 연구 기록이지 새 과학적 PASS/승격 결정이 아니다.**
