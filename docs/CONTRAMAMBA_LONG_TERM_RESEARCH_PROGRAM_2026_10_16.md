# ContraMamba — Long-Term Intelligence Architecture Research Program

> **지능의 계산 원리를 발견하고, 검증하고, 재설계하여 새로운 지능 아키텍처를 발명한다.**

| Field | Value |
|---|---|
| Document date | 2026-10-16 (Asia/Seoul) |
| Status | **LONG-TERM RESEARCH INTENT / MATHEMATICAL HYPOTHESIS REGISTER / NON-AUTHORITY** |
| Repository | `9terry-student/ContraMamba` |
| Reviewed branch | `gen5-causal-role-state-ownership` |
| Remote reference at review | `f2c931a03b9103b3a5f4a7c6ed3eb75bcbb4378e` |
| Intended repository path | `docs/CONTRAMAMBA_LONG_TERM_RESEARCH_PROGRAM_2026_10_16.md` |
| Program-level interpretation | **This is the governing long-term program statement once adopted**; older Vision/Map/Kinematics documents remain preserved as historically situated hypotheses and evidence pointers, not equal-priority definitions of the final goal |
| Relation to existing documents | Integrates and updates long-horizon interpretation of `docs/CONTRAMAMBA_RESEARCH_VISION.md`, `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md`, and `docs/CONTRAMAMBA_NATIVE_STATE_KINEMATICS_HYPOTHESIS.md` without rewriting their scientific history |
| Revision scope | Research-only revision of the five-stage program; adds neuromodulation and falsifiable strategic hypotheses about language-model reliability and Transformer-family architectures; **not** proof of local Git staging or remote publication |
| Authority | **None.** Does not approve implementation, training, inference, evaluation, Kaggle, paper promotion, artifact mutation, or commit/push |

---

## 0. 이 문서가 보존하는 것과 보존하지 않는 것

이 문서는 ContraMamba의 단기 작업 계획이 아니라 **장기적인 연구 목적, 수학적 구상, 인과적 검증 철학, 설계 선택 원칙의 날짜가 있는 스냅샷**이다. 환각 전구체·Mamba 해석·semantic ownership·최소 충분 계산·뇌과학·AGI라는 표현이 대화마다 등장해도, 그중 마지막에 강조된 하나를 새로운 최종 목표로 오인해서는 안 된다. 이들은 하나의 누적적인 연구 프로그램에서 서로 다른 질문이다. **본 문서에는 연구의 목적·가설·방법·실증적 경계만 담는다. 연구 외적인 개인적 계획이나 작업 환경은 포함하지 않는다.**

**주장의 출처를 반드시 구분한다.**

- **[O] Original/transcribed concept:** 연구자가 제공한 과거 논의의 전사·복원. 이 대화에서 전체 과거 원본 및 모든 당시 커밋을 독립 감사한 것은 아니다.
- **[M] Mathematical reconstruction:** 초기 발상을 표현하기 위해 지금 정리한 수학적 모델 또는 도식. 원문에 그대로 있었다고 주장하지 않는다.
- **[H] Hypothesis:** 아직 실험으로 확립되지 않은 설계·인지·기능 가설.
- **[E] Executed evidence:** 실제 저장소에 존재하는 범위가 한정된 실험 보고서의 결과. 원 보고서가 주장하는 것보다 강하게 해석하지 않는다.
- **[P] Program principle:** 과학적 방법론, 장기 목적, 조건부 설계 선호. 사실 확인이나 실행 권한과 구별한다.

### 0.1 장기 연구 철학의 해석 우선순위

이 문서가 Git에 확정된 이후, **장기 목적·철학·미래 아키텍처의 해석에서 이 2026-10-16 프로그램이 기존 Research Vision과 Hypothesis Map보다 우선한다.** 다만 이는 문서 간 *개념적 계승*이며, 실험 권한의 승격도, 이전 stage authority의 무효화도, 이미 검증된 보고서의 결론 변경도 아니다.

- **장기 목표:** 이 문서의 North Star와 설계·반증 원칙을 따른다.
- **세부 가설 및 역사:** 기존 Vision/Map/Kinematics를 그 작성 시점의 구체적 후보·대조군·기술 규약으로 보존한다. 본 문서에 없는 구체 항목은 충돌하지 않는 한 유지한다.
- **실제 결과:** 각 stage의 frozen experiment report, artifact, provenance가 우선한다. 이 문서의 요약이 오류라면 원 보고서를 따른다.
- **현재 허용 행동:** 현재 사용자 지시, 현 stage-specific authority/spec/manifest, repository policy에 따르며 본 문서가 권한을 부여하지 않는다.

구 Research Vision의 `Generation 1–6` 서술은 **당시의 전망적 예시**다. 실제로 존재하는 `Gen5` 연구의 단계 번호·증거 상태와 일대일로 대응시키지 않는다. 원래 문서의 'current controlled generation', Hypothesis Map의 'O0 future' 및 Kinematics의 'DO NOT EXECUTE YET' 같은 표현 역시 **작성 당시의 시점 설명**으로 읽어야 하며 2026-10-16 현재의 실행 상태를 추정하는 근거로 삼지 않는다.

본 문서는 **실행 authority가 아니다.** 적용 가능한 사용자 지시, stage-specific authority/spec/manifest/report, frozen artifacts와 provenance, `AGENTS.md`의 범위를 변경하지 않는다. `ACTIVE_AUTHORITY=NOT_INFERRED`만으로 새로운 권위 문서를 생성할 필요도 없다. 미래의 중요한 방향 변경은 이 파일을 소급 수정하는 대신 후속 날짜의 문서와 변경 이유로 남긴다. 당일 커밋 전에 오탈자·사실 오류를 바로잡는 것은 이 스냅샷의 최초 확정 작업에 해당한다.

## 1. North Star — 무엇을 최종적으로 만들고 싶은가

> 인간 뇌와 기존 인공지능의 내부 계산을 깊이 이해하고, 실제 지능 기능에 기여하는 계산 원리를 찾아 인과적으로 검증·통제한다. 유효한 원리는 생물학적 구조나 현재 모델의 구현 방식에 얽매이지 않고 추상화·재조합·확장한다. 기억, 학습, 추론, 판단, 계획, 적응과 새로운 능력을 구현하는 더 신뢰성 있고 강력한 인공지능 아키텍처를 발명한다. 장기적으로는 AGI를, 궁극적으로는 인간 지능의 한계를 넘어서는 시스템(ASI)을 지향한다.

**최종 목표는 '최소 크기'가 아니라 '더 유능한 지능 시스템의 발견과 개발'이다.** 최소 충분 계산은 능력과 인과적 기능이 요구 조건을 충족할 때 적용하는 **조건부 설계 원칙**이다. 필요한 복잡성을 억제하지 않는다.

이 프로그램은 다음 중 하나로 축소되지 않는다.

- 환각 탐지기, abstention classifier, steering vector, 특정 benchmark;
- Mamba의 영구 채택이나 Transformer의 선험적 폐기;
- 인간 뉴런·수상돌기의 해부학적 복제;
- 의미별 latent state나 ownership module의 필수 채택;
- 가장 작은 모델이나 파라미터 수 자체;
- 한 세대의 실험, ICLR 논문, 하나의 연구 성패.

AGI/ASI는 **장기 지향점이지 이미 확보된 결과나 특정 아키텍처의 성능 보증이 아니다.**

## 2. 뇌과학·기존 AI·수학 사이의 관계 [P]

### 2.1 뇌는 연구 대상이자 영감의 원천이지 설계 청사진은 아니다

뇌는 recurrence, 시간에 따른 기억 유지, 선택적 갱신, 국소 계산, 가소성, **neuromodulation에 따른 상태·학습 규칙의 맥락적 조절**, 다중 시간 척도, 기능적 분화와 협력 같은 **검증할 가치가 있는 계산적 가설**을 제시한다. 그러나 해부학적 닮음은 인과적 필요성의 증거가 아니다. 세포의 생물학적 한계를 그대로 구현할 필요도 없다. 수상돌기적 계산을 추상화할 수 있다면 그 기능을 인공적인 상태 전이·통신·비선형 연산으로 나타낼 수 있다. 이는 현재 모델에 해당 기능이 필요하다는 주장과도 별개다.

### 2.2 기존 AI는 실패하는 시스템이자 성공한 계산의 표본이다

Mamba/SSM, Transformer, 기타 모델의 내부 동역학과 오류 형성 과정을 관찰한다. 오류 전구체를 찾는 것과 정상적인 판단에 필요한 계산을 찾는 것은 서로 연결돼 있다. 학습·추론에서의 잠재 상태 개입은 단순한 성능 조절 수단을 넘어 **계산 원리의 원인과 경계를 식별하는 실험**이다.

### 2.3 인공적인 확장을 적극적으로 고려한다

정확한 외부 기억과 검색, 검증 가능한 상태, 병렬 처리, 장기적인 적응, 도구 사용, 명시적 추론 검증, 자원 배분 등은 생물학적 구현과 달라도 된다. 이들은 미래 **능력 요구 또는 설계 후보**이지 당장 추가해야 할 모듈의 확정 목록이 아니다.

### 2.4 연구 발전의 다섯 단계 — 하나의 누적적인 과학적 질문 [P]

**이 다섯 단계가 장기 프로그램의 기준 순서다.** 환각 전구체 탐색을 끝내고 별개의 AGI 아키텍처 연구를 시작하는 것이 아니다. **기존 지능의 내부를 이해하는 동안 발견한 계산 원리를, 인과적 검증과 조작을 통해 다음 세대 아키텍처의 설계 지식으로 발전시키는 한 연구**다.

```text
뇌과학 ↔ 기존 인공지능의 내부 관찰 ↔ 계산 이론
                    │  (전 과정에서 가설을 제안하고 반증)
                    ▼
01. INTERNAL COMPUTATION / MECHANISM DISCOVERY
    Recurrent state, 저장·전이·변환, 증거 통합,
    환각/잘못된 판단 전구체의 의심 지점 탐색
                    ▼
02. CAUSAL UNDERSTANDING
    어떤 상태·전이·정보 경로가 어떤 기능 또는 오류를 유발하며,
    무엇이 필요하고 무엇으로 대체 가능한가?
                    ▼
03. LATENT CONTROL
    추론 시 상태 개입·제거·억제·복원,
    학습 중 상태 형성·갱신·정보 흐름의 통제 가능성
                    ▼
04. COMPUTATIONAL PRINCIPLE EXTRACTION
    개입이 왜 작동했는지 설명하고, 특정 좌표·모듈에서 벗어나
    기능·인과적 효과를 보존하는 필요한 계산을 추출
                    ▼
05. INTELLIGENCE ARCHITECTURE INVENTION
    발견한 원리로 기억·학습·추론·판단·계획을 통합하고,
    기능적으로 충분한 구조를 설계·대조·확장
                    └── 검증 실패·새 능력·예상 밖의 동역학 ──↺ 01~04
```

다섯 단계의 핵심 질문은 다음처럼 연결된다.

\[
\text{What is represented and how does it evolve?}
\;\Rightarrow\;
\text{What does it causally do?}
\;\Rightarrow\;
\text{Can we control it?}
\;\Rightarrow\;
\text{What computation is actually necessary?}
\]

**특히 03 → 04의 전환이 장기 연구의 핵심이다.** 특정 latent intervention이 오류를 줄였다는 사실에 머물지 않고, *왜 효과가 있었는지*를 경쟁 설명과 반증 가능한 대조로 밝힌 뒤, 그 효과를 더 일반적인 상태 갱신·정보 선택·보존·통신·학습 원리로 추상화한다. 다음 세대 구조는 그 원리를 구현하고 재검증하기 위한 후속 실험이다.

이는 엄격한 일정표가 아니다. 전구체 가설의 성공을 후속 아키텍처 연구의 필수 조건으로 두지 않는다. 01~03은 **하나의 연속적인 메커니즘 연구**이며, 04~05 역시 앞 단계와 피드백으로 왕복한다. 뇌과학은 특정 단계에 늦게 도입되는 것이 아니라 모든 단계에서 검증 가능한 가설을 공급한다.

### 2.5 역사적 ContraMamba 질문을 전체 목표 아래에 배치한다 [P]

기존 Vision은 `Frame → Predicate → Sufficiency → Polarity → Authorization`이라는 해석 가능한 판단 구조, 그리고 정보·gradient 소유권의 분리를 중심에 두었다. Hypothesis Map은 native Mamba에서 *먼저 관찰하고*, 충분한 증거가 없을 때에만 구성적 semantic-state 가설로 넘어가도록 제한했다. Kinematics는 최종 표현보다 **token-time recurrent-state trajectory**를 연구 단위로 확장했다.

이 세 문서는 모두 남겨야 할 기여가 있다. 그러나 **semantic ownership, 특정 reasoning ontology, three-way classifier가 범용 지능의 필수 설계 원리라고 가정하는 것은 장기 목표가 아니다.** 연구에서는 이들을 *경쟁 가능한 구조 가설*로 검증한다. 어떤 기능이 공유 계산에서 자연스럽게 출현한다면 별도 owner를 강요하지 않는다.

최초의 환각 전구체 가설이 실패해도 장기 프로그램은 종료되지 않는다. 어떤 정보가 올바른 판단에 필요한지와 무엇이 불필요한지는 실패한 가설을 통해서도 배울 수 있다.

### 2.6 01~05의 구체적인 관찰·개입·검증 내용 [P/H]

**아래 세부 작업은 2.4절의 다섯 단계를 풀어 쓴 것이다.** 별도의 여섯 단계 프로그램이 아니며, 관찰·전구체 국소화·인과 개입·학습 중 제어는 같은 과학적 질문을 정교하게 만드는 수단이다. 각 항목은 현 단계의 실험 허가가 아니라 후속 연구에서 채택하거나 반증할 후보와 기준이다.

#### 01 — 내부 계산 원리 발견: native state의 해부와 전구체 후보 탐색

모델을 새로 설계하기 전에 실제로 잘 작동하는 계산과 실패하는 계산을 관찰한다. Mamba에서는 가능한 한 **native selective-SSM recurrent state**와 입력 의존적 업데이트, layer별 mixing, 최종 hidden/readout을 구분한다. `S_t^(ell)`의 상태 궤적, 상태 변화, 증거 위치, 모델 출력의 시간적 순서를 연결한다. Mamba 외 다른 아키텍처에서도 기능적으로 대응되는 계산을 비교할 수 있지만 native-state 분석을 일반 hidden-state 분석으로 바꿔치기하지 않는다.

- 관찰 대상 후보: read/write/retain/transform, 상태 이동, 토큰별 증거 통합, layer 간 정보 전달, 정보 그래프 `G_I`, gradient 그래프 `G_G`, 결정 그래프 `G_D`.
- 반드시 구별할 것: **raw native recurrent state**, 실제 native `write` 또는 correction write, mixer/wrapper **output**, 보조 probe/readout, 최종 logits. 한 지점의 관찰값을 다른 지점의 측정인 것처럼 보고하지 않는다.
- 초기 원칙: 구조를 우선 추가하지 않는다. **Measure first → explain second → modify third.**

**이 내부 해부의 연속선상에서 전구체 후보를 탐색한다.** 독립된 별도 프로젝트가 아니다.

단순히 출력이 틀렸다는 사실에서 시작하는 것이 아니라, 최종 판단보다 **먼저** 어떤 정보 처리의 이상이 발생하는지를 찾는다. 처음부터 모든 생성형 환각을 아우르는 보편 신호를 가정하지 않는다. 초기 검증 목표는 통제된 **confident-correct 대 confident-wrong**, 이후에 범위를 좁혀 **unsupported commitment**와 generative hallucination으로 확장할 수 있다.

초기 후보들은 결론이 아니라 **반증 가능한 관찰 가설**이다.

1. **Evidence non-incorporation:** 새 증거가 도착했는데 적절한 상태 갱신이 일어나지 않는가?
2. **Premature commitment / lock-in:** 판단 자격이 확보되기 전에 특정 결론 방향으로 고착하는가?
3. **Failed correction / commitment inertia:** 반대 증거 후에도 기존 결론의 영향이 부적절하게 유지되는가?
4. **Wrong-attractor-like / unstable trajectories:** 잘못된 판단과 연관된 궤적의 방향 전환, 급격한 이동, 불필요한 진동이나 안정화가 있는가? 'attractor'는 역학적 조건이 검증되기 전까지 비유적 후보어다.
5. **Cross-level mismatch:** native recurrent update, semantic decision coordinates 및 output confidence가 서로 다르게 움직이는가?

이 후보들에서 velocity, acceleration, turning statistic, path length, evidence-relative timing 및 `authorization × signed polarity`는 가능성 있는 진단량이지 **자동 환각 점수**가 아니다. 정보가 '유지되었다/버려졌다'는 해석도 정확한 state/write와 과제 맥락의 증거를 필요로 한다.

#### 02 — 인과적 메커니즘 규명: 전구체 검증과 실제 계산 경로 국소화

먼저 **예측 신호**를 판별하고, 그 뒤에 **인과적 매개**를 검증한다. 후보가 특정 층, 토큰, state dimension, update path, evidence moment에서 발견돼도 그것만으로 원인이거나 필수 메커니즘은 아니다.

- **Precursor gate:** 미래 결과를 사용하지 않는 prefix-only 측정, 최초 발생 시점, gold/predicted class 및 confidence matching, 길이·표면적 lexical cue·증거 도착 시간 통제, held-out 및 seed/perturbation 재현성.
- **Localization gate:** layer/token/site와 state-to-output 경로를 사전 규정하고, 마스크·좌표계·토크나이저·checkpoint·시점의 정체성을 고정한다. post-hoc 최적 지점 찾기를 확증적 발견으로 승격하지 않는다.
- **Causal gate:** 보존하고 싶은 기능에 대한 matched intervention, sham/negative controls, output-only bias 대조, 제거와 복원, 경로 매개 확인. 상관관계·probe 정확도·activation steering의 즉각적 출력 변화만으로 necessity 또는 sufficiency를 선언하지 않는다.
- **Decision semantics:** `REFUTE`, `NOT_ENTITLED`, `SUPPORT`를 섞지 않는다. 근거 부족 신호를 제거하면 오히려 잘못된 확신이 증가할 수 있다.

#### 03 — 잠재공간 제어 가능성 검증: 개입·복원·억제·학습 중 제어

이 단계는 **발견한 후보 메커니즘을 관찰하는 데서 실제로 작동을 바꿔보는 단계**다. 잠재공간 제어 자체가 최종 목적이 아니라 **계산 원리의 기능적 원인을 드러내는 실험**이라는 원래 의도를 보존한다.

- **추론 시 개입:** 국소 activation/state patching, targeted ablation, selective suppression, state or correction-write replacement, causal restoration, evidence-consistent update, 오류 경로 차단. intervention site와 참조 상태가 실제로 같은 tensor/시간 의미를 갖는지 검증한다.
- **인과적 비교:** 동일 입력의 통제된 counterfactual, 정상 판단과 오류 판단에 대한 대칭 검사, 필요한 REFUTE/SUPPORT 구별 보존, NOT_ENTITLED의 calibration/entitlement 보존, 원치 않는 downstream side effect 측정.
- **학습 중 제어(후속 후보):** gradient ownership, local-vs-joint optimization, update gating, state/channel constraints, correction learning, memory/plasticity 규칙을 이용해 상태 형성 자체가 통제 가능한지 조사한다. 단순히 loss가 감소했다고 '의미적 상태 통제 성공'이라고 하지 않는다.
- **잠재기하의 한계:** 특정 방향의 조작 효과가 다른 seed·좌표 변환·데이터에서도 같은 causal role을 갖는지 구분한다. geometry, readout sensitivity, actual mediated task function은 서로 다른 estimand다.

개입으로 환각이나 잘못된 판단이 줄더라도, **왜** 줄었는지 판별해야 한다. 잘못된 activation 억제, 유용한 증거의 유지, 상충 정보 통합, 조기 결론 방지, 단순 class-prior 변화는 서로 다른 설명이다. 정답·확신·안전한 보류·장기 인지 능력을 같이 측정해야 한다.

#### 04 — 계산 원리 추출: 관찰된 효과의 원인을 분해하고 필요한 계산을 식별

원인을 식별한 후 다음을 묻는다. **어떤 정보가 반드시 기억돼야 하는가, 어떤 계산이 상황에 따라 선택돼야 하는가, 어떤 통신·가소성·구획이 실제로 필요한가?** 현 모델의 좌표나 특정 모듈 이름은 추출 대상이 아니다.

- 상태 정보 보존/변환/폐기의 기능적 기준, evidence-conditioned recurrence, corrective update와 안정적 기억, causal-role-preserving state compression.
- 명시적 semantic ownership과 공유 상태의 자발적 기능 분화, `G_I/G_G/G_D`의 필요한 분리, 독립 상태 사이의 정보 교환 필요성.
- neuron-wise, compartmental, multi-timescale, dynamic-plastic recurrence가 수행하는 **추가적인 기능**이 matched simpler alternatives와 구별되는지 검증.

원리를 주장하려면 단일 개입의 성능 증가가 아니라 **필요성, 복원, 충분성, 조건부 대체 가능성, 미래 상황에서의 일반화** 중 해당 주장에 필요한 근거가 있어야 한다.

##### 핵심 연결 관문 — "개입이 효과가 있었다"에서 "왜 효과가 있었는가"로

예를 들어 Mamba의 특정 recurrent-state 개입이 **통제된 조건에서** unsupported commitment를 줄였다고 가정하자. 이는 가상의 예시이지 지금까지의 Gen5 실증 결과가 아니다. 하나의 유효한 개입에 대해 최소한 다음의 *서로 경쟁하는 설명*을 구분해야 한다.

| 경쟁 설명 [H] | 구별할 증거 | 그 설명이 지지될 때의 설계 시사점 |
|---|---|---|
| A. 단순 activation/logit 방향 억제 | class-prior-only·output-bias·norm-matched 대조에서도 같은 효과가 나는가 | 별도 recurrent 구조 없이 읽기/보정 수준의 최소 수단으로 충분할 수 있음 |
| B. 부적절한 증거의 상태 기록 차단 | 증거 유형·도착 시점에 따른 실제 write 변화, matched evidence intervention 및 복원 | selective evidence-conditioned update 또는 쓰기 자격 제어가 후보 |
| C. 필요한 정보의 장기 보존 | 지연 후 증거 회수·기억 간섭·state restoration에서 기능적 이점이 남는가 | 안정적 기억 또는 다중 시간 척도가 후보지만 별도 물리적 분리는 미확정 |
| D. 상충 정보의 결합·재평가 개선 | 모순 증거 순서 교란·갱신 경로 차단·복원에서 특정 효과가 재현되는가 | corrective recurrence, 정보 교환 또는 내부 구획 계산이 후보 |
| E. downstream 보상·비인과적 동반 변화 | 개입 위치를 고정한 인과적 매개/차단 실험에서 해당 경로 없이도 효과가 나타나는가 | 의심 state를 본질적 원리로 승격하지 않고 설명을 축소/기각 |

실패 원인과 정상 기능 보존을 **함께** 측정한다. 예를 들어 REFUTE/SUPPORT의 판별이 붕괴한 상태에서 NOT_ENTITLED 비율만 증가했다면 신뢰성 향상을 주장할 수 없다. 출력 효과의 동일성은 내부 계산의 동일성을 자동으로 증명하지 않으며, 특정 좌표계의 방향을 아키텍처에 필수적인 모듈로 고정하지 않는다.

**04단계의 산출물은 특정 steering vector나 intervention recipe 자체가 아니다.** 먼저 반증을 견딘 기능적 진술, 예컨대 "어떤 조건에서 어떤 정보가 유지되어야 하는가", "어떤 업데이트가 어떤 근거에 의해 제한되어야 하는가", "어떤 정보 교환이 필수인가"를 명시한다. 그 다음 shared/structured/compartmental/plastic/externally assisted 구현을 대조하여 계산의 필요성과 대체 가능성을 판단한다.

#### 05 — 새로운 지능 아키텍처 설계: 원리의 구현·확장·재검증

발견한 원리들을 새로운 recurrent unit, 내부 구획, 학습 규칙, 기억/계획/검증 구조로 **필요할 때** 구현한다. 과거의 Mamba·Transformer·뇌 해부학 또는 ENTHROPIC의 임시 이름에 묶이지 않는다. 목표는 기존 능력의 보존과 더 높은 능력의 개발이며, 동등한 기능·인과적 반응을 재현하는 후보 중에서 불필요한 복잡성을 제거한다.

새 시스템에서 얻은 성공·실패는 다시 **01 내부 해부**로 되돌아와 원래 메커니즘 가설을 검증하거나 반증한다. 뇌과학은 01~05 전체에서 새로운 가설을 제안한다. 환각 전구체가 존재하지 않거나 국소적이라는 부정적 결과도, 어떤 계산이 정상 판단에 필수적이며 어떤 기능 분화가 불필요한지에 관한 설계 지식으로 축적할 수 있다.

**연결의 해석:** `01 → 02 → 03 → 04 → 05`는 주장을 점차 강화하기 위한 기본적인 과학적 논리다. 01~03을 종료하고 전혀 별개의 AGI 연구로 전환한다는 뜻이 아니다. 전구체 탐색에 실패해도 정상적인 판단을 지탱하는 계산 원리를 탐구할 수 있고, 제어 효과가 설명되지 않으면 02~03을 다시 검토한다. 05에서 설계한 시스템 역시 01~04로 되돌려 검증한다. 어떤 단계도 본 문서만으로 구현·학습·평가·GPU·Kaggle 실행을 승인하지 않는다.

## 3. 수학적 출발점 — 일반 recurrence와 선택적 Mamba SSM

### 3.1 일반 recurrent state [O]

초기 논의에서 출발점으로 제공된 일반식:

\[
\boxed{s_{t+1}=f_\theta(s_t,x_t),\qquad y_t=g_\theta(s_t)}.
\]

이는 **일반적인 상태공간 표현**이며 Mamba에 고유하지 않다. 상태는 단순한 저장 벡터가 아니라 시간에 따라 변하는 계산의 충분한 내부 요약일 수 있다. 관찰 출력과 갱신 시점을 어떻게 인덱싱하는지는 구체적 구현에서 명시해야 한다.

### 3.2 Selective SSM과의 비교 [M]

\[
\boxed{s_t=\bar A_t s_{t-1}+\bar B_t x_t,\qquad y_t=C_t s_t}.
\]

이 식은 앞선 일반식을 Mamba 계열의 선택적 상태공간 관점에서 **비교하기 위해 추가한 표준적인 도식**이다. Mamba의 입력 의존적 선택성은 전형적으로 입력으로부터 생성되는 \(\Delta_t\), \(B_t\), \(C_t\) 등을 통해 상태의 시간 척도·쓰기·읽기에 작용한다. \(\bar A_t\), \(\bar B_t\)는 이산화된 연산이며, 모든 계수가 임의로 독립 변화한다는 뜻은 아니다. 생략된 skip/direct term 등은 실제 구현·논문의 정확한 정의에 따른다.

**핵심 질문:** 어떤 정보를 오래 보존하고, 새로 기록하고, 변환하여 미래 판단에 사용해야 하는가? 이 질문은 단순한 `forget/write` gate의 이름만 바꾼 것으로 끝나서는 안 된다.

### 3.3 상태 변화·정보 선택·출력 읽기는 별개의 연산 [M/H]

선택적 SSM의 **상태 유지/감쇠, 쓰기, 읽기**를 하나의 `retain/throw` 이분법으로 환원하지 않는다. 이전 정보를 보존하면서 새 정보를 기록할 수도 있고, 기존 상태를 **변환하거나 결합**하여 이후 판단에 필요한 의미만 유지할 수도 있다. 새로운 계산 단위의 필요성을 판단하려면 그 단위가 기존 `\bar A_t, \bar B_t, C_t` 및 기존 gating으로 구현 가능한 연산인지 대조해야 한다.

추론 시 상태 전이 `s_t → s_{t+1}`와 학습 중 파라미터 최적화 `\theta → \theta'`는 다른 인과 경로다. 이후 연결 가소성 `W_{ij}(t)`를 논할 때에도 추론 시 fast-weights와 persistent learned weights를 혼동하지 않는다.

## 4. 핵심 장기 아키텍처 후보 — 뉴런·내부 구획·연결 [O/M/H]

이 절은 **확정 아키텍처가 아니라 경쟁 가능한 수학적 가설군**이다. 여기 수식을 통째로 '이미 검증된 ContraMamba 모델'로 인용해서는 안 된다.

### 4.1 Neuron-wise vector recurrence [O]

각 계산 단위 \(i\)가 개별적인 벡터 상태를 가진다는 원래 발상:

\[
h_i(t)\in\mathbb R^d,
\]

\[
\boxed{
 h_i(t+1)=f_\theta\!\left(h_i(t),\sum_{j\in\mathcal N(i)}W_{ij}(t)\,g_\theta(h_j(t))\right)
}.
\]

각 단위의 상태 유지와 **단위 간 상호작용**을 구분한다. \(W_{ij}\)가 스칼라인지 선형사상인지, 연결이 dense/sparse인지, 그래프 구조가 학습되는지는 열린 문제다. 생물학적 뉴런 하나를 반드시 인공 벡터 하나에 대응시키겠다는 의무도 없다.

**검증할 가설:** 동일한 상태량·파라미터·연산량에서 완전 공유 상태보다 분산된 상호작용이 특정 기억·조합·오류 복원 기능에 유리한가? 개선이 단지 용량 증가나 더 긴 계산 경로 때문은 아닌가?

### 4.2 연결 가소성: activation update와 다른 층위 [O]

\[
\boxed{W_{ij}(t+1)=W_{ij}(t)+\Delta W_{ij}(t)}.
\]

여기서 \(\Delta W_{ij}(t)\)의 규칙은 **확정하지 않았다**. 단기 활동 상태 \(h_i(t)\)의 변화와 경험으로 바뀌는 연결 \(W_{ij}(t)\)의 변화는 개념적으로 다르다. 앞으로 이 변화를 추론 시 일시적인 fast weights로 둘지, 장기 매개변수 학습으로 둘지, 두 수준의 결합으로 둘지 결정해야 한다.

**필수 반증 대조:** 같은 기능을 고정 연결+증가된 상태 용량으로 구현할 수 있는가? 가소성은 필요한 정보를 축적하는가, 아니면 간섭·망각·불안정성을 악화하는가? credit assignment, update stability, online evaluation leakage에 대한 별도 규칙이 필요한가?

### 4.2.1 Neuromodulation과 상황 의존적 학습 규칙 [M/H]

기본 가소성 식은 **무엇이 변하는가**를 나타내지만, **언제·어떤 조건에서·어떤 규칙으로 변하는가**는 지정하지 않는다. 뇌과학에서 영감을 얻은 새로운 연구 후보는 내부 조절 상태 \(m_t\)가 활성 상태 전이뿐 아니라 가소성의 강도 또는 학습 규칙을 조건부로 조절할 수 있는지 묻는다. 아래 수식은 **이번에 추가한 개념적 재정식화**로, 과거에 확정된 Mamba 방정식이나 실행된 구현이 아니다.

\[
\boxed{
\begin{aligned}
h_{t+1}&=f_\theta(h_t,x_t;m_t),\\
W_{t+1}&=W_t+\Delta W_t,\\
\Delta W_t&=\eta(m_t)\,\Phi(h_t,x_t).
\end{aligned}}
\]

여기서 \(m_t\)는 맥락적 조절 신호, \(\eta(m_t)\)는 업데이트 크기, \(\Phi\)는 상태와 입력으로부터 국소적 연결 변화를 정의하는 **아직 지정되지 않은** 규칙이다. \(m_t\)는 novelty, uncertainty, feedback, task context 등의 정보를 활용하는 후보일 수 있으나, 특정 생물학적 neuromodulator와 일대일 대응하거나 그러한 신호가 이미 계산된다고 전제하지 않는다.

위 식은 \(\eta\)만 변할 때 **학습 강도의 조절**을 기술한다. 학습 **규칙의 형태 자체**가 변하는 더 강한 가설은 별도로 나타낼 수 있다.

\[
\Delta W_t=\eta(m_t)\,\Phi_{\psi(m_t)}(h_t,x_t).
\]

이는 \(m_t\)에 의해 가소성 규칙의 매개변수/형태가 달라질 수 있다는 *추가 가설*이지 필수 구성요소가 아니다. \(m_t\)의 형성, 느린/빠른 상태와의 상호작용, 피드백의 도착 시점, 추론 중 가중치 수정과 훈련 중 최적화의 구별 역시 열린 문제다. **연결 갱신 \(\Delta W_t\)**를 Gen5의 low-rank **state-write correction \(\delta w_t^{\mathrm{write}}\)**과 동일시하지 않는다.

**경쟁 가설과 반증 기준:** 조절이 없는 고정 학습 규칙, 입력 조건부 게이팅만 있는 표준 recurrent/SSM, 동일한 용량·연산량의 fast-weight/외부 메모리, 단순 학습률 스케줄을 대조한다. 조절 신호를 제거·고정·교란했을 때 학습 속도, 장기 유지, 전이, 간섭 및 복원에 고유한 인과적 차이가 없으면 이 표현에서 neuromodulation의 별도 구조적 필요성은 지지되지 않는다. 신호가 추가적인 라벨·미래 피드백을 몰래 사용하거나 단순 용량 증가에 의존해서는 안 된다.

### 4.3 Neuron-internal compartmental dynamics [O]

초기 구상의 핵심: 하나의 계산 단위가 하나의 균일한 상태 변수가 아닐 수 있다.

\[
\boxed{h_i(t)=\begin{bmatrix}h_{i,1}(t)\\h_{i,2}(t)\\\vdots\\h_{i,K_i}(t)\end{bmatrix}},
\]

\[
\boxed{h_{i,c}(t+1)=f_{i,c}\!\left(h_{i,c}(t),u_{i,c}(t)\right)},
\]

\[
\boxed{y_i(t)=g_i\!\left(h_{i,1}(t),\ldots,h_{i,K_i}(t)\right)}.
\]

**구조 가설:** 내부 구획이 서로 다른 비선형 계산, 입력 통합, 억제·증폭, 상태 보존 시간, 국소적 갱신 규칙을 가질 수 있다. 하위 구획 사이에 통신이 필요한 경우 그 규칙을 \(u_{i,c}(t)\)에 명시해야 한다. 단순히 벡터를 나누는 것만으로 새로운 계산 원리라고 주장하지 않는다.

**필수 비교:** 동일한 총 차원과 FLOPs를 가진 단일 상태, multi-head gated state, 무작위/학습된 compartment 구성, 구획 간 통신 제거, 서로 다른 비선형성과 동질 비선형성. 실제 생물학적 수상돌기와의 기능적 대응은 별도의 실증을 요구한다.

### 4.4 기능적으로 분화된 다중 상태 [M/H]

\[
s_t=\begin{bmatrix}s_t^{(1)}\\s_t^{(2)}\\\vdots\\s_t^{(K)}\end{bmatrix},
\]

\[
\boxed{s_{t+1}^{(k)}=f_k\!\left(s_t^{(k)},x_t,\sum_{\ell\ne k}M_{k\ell}(s_t^{(\ell)})\right)}.
\]

작업 기억, 안정적 지식, 불확실성, 증거, 판단, 제어 등을 **가능한 기능**으로 구분해 보되, 독립 모듈이 필요하다고 가정하지 않는다. **명시적인 specialization**과 **공유 상태 안의 자발적 specialization**을 경쟁 가설로 놓는다. 연구에서 정의한 semantic labels는 모델 내부의 자연법칙이 아니다.

### 4.5 선택적 state-update ownership [M/H]

\[
\boxed{s_{t+1}=s_t+\sum_{k=1}^{K}\gamma_{k,t}\,\Delta s_t^{(k)}}.
\]

\(\gamma_{k,t}\)는 업데이트의 반영 강도를 나타내는 개념적 제어 변수다. 이것이 반드시 독립된 '판단 모듈'이거나 생물학적 gate여야 하는 것은 아니다. 공유 파라미터·암묵적 학습·기존 SSM 갱신의 재매개변수화와 **수학적 또는 기능적 동치**가 가능한지 먼저 확인해야 한다. \(\gamma\)가 존재한다고 해서 ownership 이점이 입증된 것도 아니다.

### 4.6 다중 시간 척도 [M/H]

\[
\begin{aligned}
s_{t+1}^{\mathrm{fast}}&=f_{\mathrm{fast}}(s_t^{\mathrm{fast}},x_t,s_t^{\mathrm{slow}}),\\
s_{t+1}^{\mathrm{slow}}&=f_{\mathrm{slow}}(s_t^{\mathrm{slow}},s_{t+1}^{\mathrm{fast}}).
\end{aligned}
\]

빠른 환경 변화와 오래 보존할 지식이 다른 속도로 바뀌는 구조를 시험할 수 있다. 하지만 slow/fast 분리가 동일 용량의 single-state 또는 연속 스펙트럼의 time constants보다 낫다는 것은 가설이다. slow state의 변경 기준과 간섭 방지도 검증 대상이다.

### 4.7 개념적 보조어: ENTHROPIC / End–Throw–Pick [M/H]

과거 대화에서 제안된 재미있는 **연구 별칭 후보**로서, 무엇을 폐기(Throw)하고, 무엇을 선택·기록(Pick)하며, 언제 판단을 종료·보류(End)하는지 묻는다. 한 가지 가능한 도식은

\[
\begin{aligned}
s_{t+1}&=(1-\tau_t)\odot s_t+p_t\odot\phi(x_t,s_t),\\
e_t&=\pi_{\mathrm{end}}(s_{t+1},\mathcal E_{\le t}).
\end{aligned}
\]

단, 이것은 **원래의 neuron-wise/compartmental 모델을 대체하는 완성식이 아니다**. \(\tau_t,p_t\)의 독립성, 기존 gated recurrence와의 동치 가능성, 상태의 **변환·재구성·전달**은 별도로 다뤄야 한다. `entropy`라는 연상만으로 Shannon entropy의 최적화 목표나 정보 이론적 새로움이 성립하지 않는다. `ENTHROPIC`은 확정 연구명·논문 제목도, 특정 기업에 관한 검증된 주장도 아니다.

### 4.8 분할 자체가 아닌 필요한 계산의 고유 기여를 증명한다 [H]

미래의 각 아키텍처 후보에는 언제나 다음 경쟁 설명을 함께 둔다.

| 제안 | 대조해야 할 가장 가까운 경쟁 설명 |
|---|---|
| Neuron-wise interaction | 동등 용량의 dense recurrence 또는 단순 message passing |
| Compartmental recurrence | 동일한 총 차원·비선형성·time constants를 갖는 균일 상태 |
| Dynamic plasticity | 고정 연결 + 충분한 activation memory / 외부 메모리 |
| Neuromodulated update / learning | 동일 용량의 고정 규칙, 입력 조건부 gate, 학습률 schedule, fast weights |
| Explicit semantic ownership | 공유 상태의 자발적 specialization / 복제·무작위 분기 |
| Separate time scales | 균일 구조가 학습한 연속 decay spectrum |
| End–Throw–Pick | 기존 gated RNN/SSM의 재매개변수화 |

핵심은 새로운 명칭·모듈의 존재가 아니라 **수행 기능, 인과 반응, 지연 기억, 학습 안정성, 일반화에 대한 고유한 기여**다. 발견이 부정적이면 후보를 보류·수정·기각할 수 있어야 한다.

## 5. 현재 ContraMamba에서 실제로 시험한 계열: 무엇이 다른가 [E/M]

### 5.1 구조 설계 가설과 개입 장치는 구별한다

4절의 neuron-wise/compartmental/plastic architecture는 **미래 설계 후보**다. 반면 Gen5의 low-rank correction은 이미 존재하는 모델의 특정 native write 또는 출력을 통제·관찰하여 인과적 역할을 식별하려는 **실험 장치**다. 둘을 같은 아키텍처의 구현이라고 부르지 않는다.

### 5.2 Gen5 layer-22 low-rank write correction [O/M/E]

원문 전사에서 사용된 보정 후보의 개념식:

\[
\boxed{\delta w_t^{\mathrm{write}}=B_\theta A_\theta x_t,\qquad \operatorname{rank}(B_\theta A_\theta)\le r}.
\]

원문에는 \(\Delta W_t=B_\theta A_\theta x_t\)로 쓰인 부분이 있으나 여기서는 **연결 가소성의 \(\Delta W_{ij}\)와 혼동하지 않도록** \(\delta w_t^{\mathrm{write}}\)로 바꾸어 표기한다. 예시로 \(r=2\)인 layer-22 correction과 frozen \(R_{22}\), \(C_{22}\) subspace를 사용하는 비교가 논의되었다. 정사영 \(P_R,P_C\)를 사용한 다음 식은 **실험군 의도의 도식**이지 전체 실제 코드의 exact operator를 증명한 식이 아니다.

\[
\begin{array}{ll}
\mathrm{C0}:&\delta w_t,\\
\mathrm{C1}:&(I-P_{C_{22}})\delta w_t,\\
\mathrm{M1}:&(I-P_{R_{22}})\delta w_t.
\end{array}
\]

실제 실험군의 정확한 정의·마스크·시점·스케일·손실·결과는 **해당 frozen stage spec, 코드, 보고서**를 따른다. 위 도식은 실행 절차가 아니고, 독립적인 새 결과가 아니다. 원보고서에 기록된 Phase2 M1–C1 ownership advantage는 확립되지 않았다.

### 5.3 State compression과 기능적 충분성 [O/M/H]

\[
\boxed{Z_t=E_\phi(S_t)}.
\]

문제는 차원 감소 자체가 아니라 **압축된 상태가 미래의 적절한 판단·기억·학습·제어 기능을 보존하는가**다. 복원 오차, 다음 토큰 예측, 평균 CE만으로 충분성을 선언하지 않는다. 정보 손실의 중요도는 사용 맥락, 시간 지연, 인과적 개입, 불확실성, 분포 변화에 좌우된다.

#### 5.3.1 두 종류의 충분성을 구별한다

- **Behavioral sufficiency:** 정의한 과제·입력·환경에서 목표 능력과 신뢰성을 유지한다.
- **Causal/interventional sufficiency:** 지정된 개입들의 효과와 필요한 기능적 반응을 허용오차 안에서 재현한다.

잘못된 판단을 그대로 재현하는 것을 '지능 기능 보존'으로 착각해서는 안 된다. 원 모델 자체가 실패하는 경우에는 목표 기능 기준 및 비교 기준을 명시한다.

### 5.4 기존 Vision/Map의 세 그래프를 보존한다 [P/H]

ContraMamba의 중요한 원래 기여는 `state ownership`이라는 이름보다, *어떤 연산이 무엇을 읽을 수 있고, 어떤 목적함수가 무엇을 바꿀 수 있으며, 최종 판단이 어떤 논리로 허용되는지*를 분리했다는 데 있다.

- **정보 그래프 `G_I`**: 표현·state·module 간의 forward read/communication 권한.
- **gradient 그래프 `G_G`**: 목적함수·loss가 어떤 표현과 파라미터를 수정할 수 있는지의 backward optimization 권한.
- **판단 그래프 `G_D`**: Frame, Predicate, Evidence Sufficiency, Polarity에서 최종 factual authorization으로 가는 논리적 결합 및 차단 순서.

`G_I`, `G_G`, `G_D`는 같을 필요가 없다. 특히 `Frame > Predicate > Sufficiency`라는 first-blocker 우선순위는 **최종 결정의 논리적 순서**이지 `Frame owner`가 매 토큰마다 `Predicate owner`에 정보를 보내야 한다는 뜻이 아니다. 정보는 공유 표현으로부터 독립적으로 읽고 판단 시점에만 결합할 수 있다.

과거 Hypothesis Map의 **`NO CROSS-OWNER COMMUNICATION`**은 *초기 대조군이자 과도한 혼합을 막는 출발 가설*로 보존한다. 전체 연구 프로그램의 영구적인 통신 금지는 아니다. 실제 결핍이 확인되면 `no edge → narrow read-only message → small adapter → partial gradient sharing`을 단계별 대조할 수 있다. 반대로 unrestricted concatenation과 raw-hidden bypass가 semantic ownership의 인과적 역할을 흐릴 수 있다는 기존 경고는 유지한다.

개념적인 reasoning ontology는 다음과 같다.

```text
F: frame / entity-event context compatibility
P: predicate / relation coverage
S: evidence sufficiency
Q: support-refute polarity, conditional on evidence
D: evidence-entitlement and final decision
```

이 역할은 **연구를 위한 해석적 변수**이며 필연적인 뇌 구획이나 영구적인 neural owner가 아니다. `NOT_ENTITLED`는 세 번째 polarity가 아니라, 충분한 근거로 decisive output을 낼 자격이 없다는 의미일 수 있다. 이 차이는 더 단순한 3-class alternative와 비교·검증해야 한다.

### 5.5 구조화 판단과 소유권 가설의 대조 [M/H]

기존 router의 제품형 authorization 진단은 예컨대 `A = F P S`로 표시할 수 있고, first-blocker mass를

\[
q_F=1-F,\quad q_P=F(1-P),\quad q_S=FP(1-S),\quad q_A=FPS
\]

와 같이 표현할 수 있다. 이 masses는 `q_A=A`와 곱 분해의 **대수적 재표현**이므로, 합산만으로 기존 product composer보다 새로운 인과 메커니즘이 성립했다고 주장할 수 없다. 기존의 reject된 equivalence를 되살리지 않는다. 실제 비교 대상은 reason-specific supervision, gradient pathways, 정보 전달 정책, state-update dynamics의 **검증 가능한 차이**다.

장래의 semantic-state 연구는 다음과 같은 matched controls를 유지한다.

```text
O0  native frozen state precursor observation
M0  shared-state control
M1  replicated-state null (동일 갱신의 복제, novelty null)
M2  arbitrarily diversified-state control
M3  semantic ownership + local supervision
M4  M3 forward 동일, joint/downstream gradient 허용
```

이는 기존 Hypothesis Map의 **후보 대조군**이며 지금 실험을 허가하는 실행 매트릭스가 아니다. `M0 vs M1`은 상태 복제만의 효과, `M1 vs M2`는 임의적 trajectory diversity, `M2 vs M3`는 semantic ownership, `M3 vs M4`는 gradient ownership을 구분하려는 설계다. 용량·연산량·state memory·최적화 경로의 대조가 동반되어야 한다.

## 6. 복잡도, 능력, 인과적 충실성의 조건부 최적화 [O/M/P]

원문에서 제안된 탐색적 목적함수:

\[
J=\mathcal L_{\mathrm{task}}+\lambda C(\mathcal A)+\mu C_{\mathrm{compute}}.
\]

그러나 이것만으로 장기 목표를 정의하지 않는다. 능력·인과·일반화를 우선 명시한 뒤, 이를 만족하는 후보 중 효율과 설명력을 비교한다.

\[
\begin{aligned}
\min_{\mathcal A,\theta}\quad&C(\mathcal A,\theta)\\
\text{subject to}\quad&D_{\mathrm{behavior}}(\mathcal A,\theta)\le\varepsilon_b,\\
&D_{\mathrm{causal}}(\mathcal A,\theta)\le\varepsilon_c,\\
&D_{\mathrm{generalization}}(\mathcal A,\theta)\le\varepsilon_g,\\
&Q_j(\mathcal A,\theta)\ge q_j\quad\text{for defined capabilities }j.
\end{aligned}
\]

여기서 \(D\)의 reference, 개입 집합, 측정 분포, 허용오차, \(Q_j\)는 **미래 개별 연구에서 사전 지정**해야 한다. \(C\)도 상태량, 파라미터, FLOPs, latency, memory bandwidth, 통신, 데이터 효율, 적응 비용 등 여러 축의 **Pareto cost**일 수 있다. 단일 숫자나 절대적인 전역 최소해를 전제하지 않는다.

**설계 규칙:** *동일한 요구 능력과 관련 인과적 효과를 만족한다면 더 단순하고 신뢰할 수 있는 구조를 선호한다. 새로운 능력에 복잡성이 필요하면 도입하고 검증한다.*

## 7. 검증 난점과 반증 조건 — 아이디어에서 원리로 넘어가는 관문 [P/H]

### 7.1 Credit assignment와 representation coordination

계산 단위·구획·시간 척도가 늘어나면 어느 상태 갱신이 어떤 결과에 기여했는지 학습·식별하기가 더 어려워질 수 있다. 서로 분리된 표현은 호환되지 않거나 정보 교환 비용을 증가시킬 수 있다. **이 문제는 모듈 개수를 늘리거나 GPU를 추가하는 것으로 해결됐다고 가정하지 않는다.** 학습 안정성, gradient pathways, state/read/write attribution, 정보 공유, 경로 중복 및 교란을 직접 검증한다.

### 7.2 최소 증거 사다리

1. **Measurement:** 현상이 재현되고 정확한 state/readout provenance를 가진다.
2. **Prediction:** 타당한 holdout과 시간적 선후관계에서 실제 결과를 예측한다.
3. **Causal effect:** 통제된 개입이 지정된 출력을 변화시킨다.
4. **Necessity / restoration:** 제거·복원이 제한된 기능에 대해 인과적 기여를 확인한다.
5. **Sufficiency / equivalence:** 더 단순하거나 다른 구조가 목표 기능과 개입 반응을 충족한다.
6. **Transfer:** seed·scale·task·architecture·시간 조건을 넘어 사전 지정된 범위에서 일반화한다.
7. **Architecture benefit:** 능력, 안전성·신뢰성, 비용, 적응 등에 대해 정당화 가능한 설계 이점이 있다.

단계는 상황에 따라 병렬·반복될 수 있지만, **한 단계의 결과를 증거 없이 윗단계의 결론으로 승격하지 않는다.**

### 7.3 반드시 실패 가능해야 하는 가설

| 가설 / 후보 | 반증 또는 강한 약화 조건 | 다음 해석 |
|---|---|---|
| 범용 환각 전구체 | 타당한 미래 구간·held-out 조건에서 예측 및 인과적 효과 부재 | 범용성 주장 철회; 국소 기능/정상 판단 분석으로 전환 |
| 명시적 기능 분화 필요 | 동등한 공유 상태가 같은 기능·개입 반응·비용 조건 충족 | 별도 모듈의 필수성 주장 철회 |
| Neuron-wise recurrence 이점 | 용량·연산·시간 경로 통제 후 우위 사라짐 | 연결 구조의 필요성 미확립 |
| Compartmental dynamics 이점 | 동일 용량의 균일 계산과 동등, 구획 개입의 고유 인과 효과 없음 | 생물학적 유사성만으로 유지 불가 |
| 동적 가소성 이점 | 고정 연결과 상태만으로 동일 기능 유지; 가소성은 불안정성만 증가 | 해당 범위의 가소성 필요성 기각 |
| Neuromodulation / 학습 규칙 적응 이점 | 고정 규칙 또는 입력 조건부 gate와 matched 비교에서 고유 효과 없음, 조절 신호 개입이 학습·기억에 영향 없음 | 별도 조절 구조의 필요성 미확립 |
| Slow/fast 분화 이점 | 동등한 single-state baseline에서 우위 없음 | 시간척도 물리적 분리 불필요 가능 |
| Causal-role ownership 이점 | matched control에서 우위나 필요성 미확립 | ownership을 원리로 승격하지 않음 |
| 압축 상태 충분성 | 장기 지연·개입·OOD에서 핵심 기능 소실 | 압축 설정/목표/필요 정보 수정 |
| ENTHROPIC 고유 메커니즘 | 기존 gated recurrence와 algebraic/functional equivalence | 이름 변경만으로 novelty 주장 금지 |

### 7.4 공통 실험 대조

- **Matched capacity/compute/data:** 늘어난 파라미터나 FLOPs를 인과적 구조 효과로 오인하지 않는다.
- **Correct task semantics:** 정확도뿐 아니라 클래스별 실패, 증거 자격, calibration, 잘못된 확신을 분리한다.
- **Controlled interventions:** 단순 output bias·norm·prior shift·mask 오류·데이터 누출과 genuine state-path mediation을 구분한다. Neuromodulation 후보에서는 조절 신호를 고정·삭제·치환하는 개입과 추가된 파라미터/연산을 일치시킨 비교를 포함한다.
- **Temporal validity:** '전구체'는 결과 이전에 이용 가능한 정보로만 측정한다.
- **Reproducibility:** seed/split/checkpoint/model/tokenizer/kernel/command/artifact identities를 보존한다.
- **Generalization:** 재구성 기하학, 인과적 동일성, 과제 성능, cross-seed transfer를 같은 값으로 다루지 않는다.
- **Architecture alternatives:** 기능이 같다면 shared, divided, compressed, locally plastic, externally assisted 설계 모두 비교 후보가 될 수 있다.

### 7.5 실패 형태의 이름까지 검증 규약으로 남긴다 [H/P]

기존 연구가 이미 경계한 `Latent Soup`(owner들이 사실상 동일 상태), `Dead Owner`(갱신되지 않는 owner), `Always-Open Gate`(무차별적인 write), `Readout Illusion`(head만 분화하고 recurrent state는 비분화), `Over-Isolation`(구별은 좋아지지만 능력 붕괴), `Arbitrary Diversification`(의미적 owner와 무작위 분기의 차이 없음), `Shortcut Owner`(어휘적 단축경로)는 후속 아키텍처 비교에 유용하다. 새 compartment/fast–slow 설계에서도 같은 실패 유형을 별도로 찾는다.

Probes나 분류기 정확도만으로 상태의 인과적 의미를 증명하지 않는다. 서로 다른 의미를 주장하는 상태가 결국 같은 계산을 하고 있다면 독립 구조의 필수성을 주장할 수 없다. 반대로 shared state에서 필요한 기능이 확인되면 구조를 분리하지 않는 것이 합리적일 수 있다.

## 8. 현재까지의 실증 결과 — 제한된 증거를 그대로 보존 [E]

여기 적은 것은 **2026-10-16 검토한 원격 브랜치의 연구 보고서 요약**이다. 과거 모든 결과를 전수 감사한 종합 메타분석이 아니다. 정확한 결론, 실행 권한, 표본, 통계, 확인 수준은 **각 frozen report가 우선**한다.

| 계열 | 확인 가능한 범위 | 넘어서는 안 되는 결론 |
|---|---|---|
| K-series / Gen4 [user retrospective] | 일부 native Mamba state/write의 제한된 인과 기여를 탐구; 범용 hallucination precursor 미확립 | 범용 선행 신호 또는 universal state-law 입증 아님 |
| Five-scale [user retrospective] | 인과적 역할과 특정 기하학적 구현의 대응이 단순한 보편 법칙으로 확인되지 않음 | 모든 scale에서 동일 역할이 증명된 것 아님 |
| Gen5 Phase2 [verified report] | M1–C1 confirmatory ownership advantage **NOT ESTABLISHED**, \(p=0.36700787067950813\) | 모든 ownership 가설의 불가능성도, 우위 입증도 아님 |
| Gen5 Phase3A [verified report] | **Causal-role contention NOT ESTABLISHED**; Phase3B **BLOCKED** | 정적 gate 성공이 실험 허가가 아님 |
| Cross-seed operator [verified report] | 3 seed의 two-donor actual operator-span mean coverage 약 **28.09%** | ex-post 기하학적 표현 가능성을 행동적 동치·전이로 해석 불가 |
| G2-P0 [verified report] | FULL, SHARED_NORMMATCH에서 3 seeds 각각 840/840개 `NOT_ENTITLED` 예측; 정확도 **71.43%**, balanced accuracy **33.33%** | CE/accuracy의 평균 향상을 정상적인 3-way 판단 향상으로 승격 금지 |
| G2 within-pair [verified report] | C6 explicit-denial vs C0 sham의 REFUTE–SUPPORT logit contrast가 특정 correction arms에서 음으로 뒤집힘 | 단순 상수 class-logit bias만으로 설명 불가; raw recurrent-write 매개 증명 아님 |
| G2-H1 [verified code + commit] | Layer-22 output observer와 toy CUDA tensor micropreflight | output observer를 native **raw write** 관찰 또는 pretrained task evaluation으로 부르지 않음 |

이 중 G2 결과가 장기 프로그램과 연결되는 이유는 중요하다. **평균 손실이 좋아도 핵심 판단 능력이 붕괴할 수 있다.** 따라서 미래 아키텍처는 목표 기능의 선택적 보존을 실험적으로 보여야 하며, '상태를 바꿨다'는 이유만으로 새 계산 원리를 발견했다고 해서는 안 된다.

### 8.1 해석상의 중요 구분 [E/P]

- **Phase1B R22:** 인증된 native site의 제한된 필요성/복원 증거를 전역적 의미 기능이나 전구체로 일반화하지 않는다.
- **Gen5 Phase2:** `p=0.367...`은 해당 사전 지정 비교에서 우위가 *확립되지 않았음*을 뜻한다. 영가설의 참이나 모든 ownership의 실패를 확증하지 않는다.
- **Gen5 Phase3A:** causal-role contention 미확립이며 Phase3B의 차단 상태를 변경하지 않는다.
- **Cross-seed:** `28.09%`는 frozen operator Frobenius energy의 ex-post span coverage이지, 28.09% 기능 보존도 아니고 범용 인과 계산 핵심의 발견도 아니다.
- **G2-P0:** 일부 correction 조건의 `71.43%` accuracy는 원 모집단 다수 클래스의 비율과 같고 balanced accuracy는 `33.33%`다. 이는 지능 기능 개선을 주장하기에 부적절하다.
- **G2 within-pair:** C6 대비 C0의 `REFUTE−SUPPORT` logit contrast sign inversion은 *상수 class-logit offset만으로는 설명할 수 없지만*, 내부 매개 원인까지 식별하지는 않는다.
- **G2-H1:** frozen wrapper/mixer output observer는 raw recurrent write, true native state, causal necessity, pretrained task evaluation과 동의어가 아니다. 최근 CUDA micropreflight는 tensor-level validation이지 새 task evidence가 아니다.

**Code correctness, execution success, artifact/provenance validity, scientific conclusion**은 항상 별개다.

## 9. 지금의 연구에서 미래 설계 지식으로 건너가는 방법 [P]

**현재의 좁은 질문:** G2-P0의 수정이 왜 SUPPORT/REFUTE 구별을 무너뜨렸는가? output bias, input-dependent logit distortion, 보존 정보 손실, layer-22 내부 전이, 다른 층의 보상 작용은 구별되어야 한다.

**관찰 한계:** 현 G2-H1 observer는 mixer/wrapper **output**을 볼 뿐 내부 **raw recurrent write** 자체를 읽는 것이 아니다. 따라서 그 결과로 native write mediation·필요성을 주장하지 않는다.

**장기적 연결:** 정상적인 증거 판단에 필요한 계산이 식별되고, 이를 제어·복원하며, 다른 단순하거나 더 능력 있는 구조에서도 해당 기능이 구현되는 것을 확인하면, 그때 비로소 아키텍처 설계 지식으로 축적한다.

**반대로 가설 실패도 유의미하다.** 전구체가 발견되지 않아도 어떤 상태가 올바른 판단을 지지하는지, 어떤 개입이 무관한지, 어떤 모듈이 불필요한지를 확인할 수 있다. 연구 목표가 '환각 전구체를 반드시 발견한다'는 성공 조건에 종속되어서는 안 된다.

현재 stage에서 새로운 GPU 실험·학습·검증·모델 수정은 이 문서만으로 허용되지 않는다. 구체적인 다음 행동은 로컬 `cm context`, 실제 stage authority, code/artifact 및 상태를 기준으로 별도로 결정한다.

### 9.1 Native-state kinematics: 현재 연구와 미래 설계의 연결 고리 [H/M]

기존 `CONTRAMAMBA_NATIVE_STATE_KINEMATICS_HYPOTHESIS.md`의 중요한 통찰은 최종 hidden representation만 보지 않고 **원래 selective-SSM recurrent state**의 시간에 따른 궤적을 연구한다는 것이다. layer `\ell`에서 `s_t^{(\ell)}=vec(S_t^{(\ell)})`라 놓고, 토큰 인덱스의 차분으로

\[
\begin{aligned}
v_t^{(\ell)} &= s_t^{(\ell)}-s_{t-1}^{(\ell)},\\
a_t^{(\ell)} &= v_t^{(\ell)}-v_{t-1}^{(\ell)},\\
\kappa_t^{(\ell)} &= 1-\cos(v_t^{(\ell)},v_{t-1}^{(\ell)})
\end{aligned}
\]

를 정의할 수 있다. `\kappa`는 방향 전환의 간단한 진단량이지 물리적 곡률이나 인식론적 진실의 측정치가 아니다. path length, displacement, path efficiency 역시 기준축·scale·layer의 영향을 받는다. token/model time을 물리적 시간으로 해석하지 않는다.

강한 연구 질문은 **잘못된 최종 답변이 아니라, 잘못된 답변 이전부터 다른 recurrent-state trajectory가 발생하는가**다. 첫 검증 대상은 일반적인 generative hallucination 전체가 아닌, final confidence와 gold/predicted class 등을 통제한 `confident-correct vs confident-wrong`이어야 한다. 미래 출력을 미리 보고 선택한 층·시간·threshold를 다시 전구체 성능으로 보고해서는 안 된다.

판단 공간은 native state 좌표와 분리한다. flat `REFUTE / NOT_ENTITLED / SUPPORT` simplex와 `Authorization × Signed Polarity` factorization은 경쟁적인 **후보 기하학**이다. `A_t`가 근거에 기반한 authorization 강도, `Q_t`가 signed polarity의 후보라고 할 때

\[
\tau_Q<\tau_A
\]

는 `commitment-before-entitlement`의 **가능한 시간적 형태**지만, 올바른 판단도 이 패턴을 보일 수 있으므로 이것만으로 오류라고 선언할 수 없다. 주어진 원본 기록은 이를 실증된 결론이 아닌 가설로 명시한다.

'전구체'를 주장하려면 prefix-only 시점, 발생 시점, matched confident error, class/length/lexical controls, held-out evidence, temporal locality, seed와 input intervention 전반의 재현성이 필요하다. 상관 신호가 관찰되어도 실제 causal role의 필요성·충분성은 별도 개입으로 검증해야 한다. 원문의 `NO SUPPORT / REFORMULATE / SUPPORT` 반증 구분을 보존한다.

### 9.2 다음 아키텍처 가설의 도출 논리 [P/H]

```text
Raw native dynamics에서 이른·재현 가능한 정보 관찰
    ├─ 공유 state가 필요한 기능을 충분히 구현: 작은 readout / control 우선
    ├─ 기능 존재하지만 강하게 얽힘: local states/communication의 제한된 대조
    ├─ 명시 구조가 matched baseline보다 우수: 새로운 계산 단위 검증
    └─ 후보 부정·모순: 기하학/측정/인과 가설 재정식화 또는 폐기
```

여기서 `neuron-wise`, `compartmental`, `plasticity`, `multi-timescale`은 별개 경쟁 기계장치이므로 자동으로 하나의 거대 모델로 합치지 않는다. 한 메커니즘이 기능적으로 필요하다는 증거가 쌓였을 때 그 메커니즘을 위한 가장 간단한 구현을 먼저 시험하고, 더 높은 능력에 복잡성이 필요하다는 근거가 있으면 확장한다.

### 9.3 현 단계와 장기 실험 순서의 경계 [P/E/H]

이 장기 경로를 **현재 Git 상태에서 곧바로 실행한다는 뜻은 아니다**. 현재 frozen G2-P0의 현상은 3-way class collapse 및 within-pair contrast distortion이며, G2-H1의 확보된 관찰 장치는 layer-22 **output observer**다. 따라서 다음을 혼동하지 않는다.

| 관찰 또는 조치 | 허용 가능한 가장 강한 해석 | 아직 필요한 근거 |
|---|---|---|
| Native trajectory와 실패의 동반 변화 | 특정 population에서 연관된 후보 현상 | prefix-only 전구체 검증, 통제된 재현 |
| Layer-22 mixer/output 차이 | 해당 출력 경로의 관찰 | raw recurrent-state/write 정체성, 인과 매개 |
| Correction 후 class collapse | bounded input/arm/task에서의 실제 기능 저하 | 표준 bias, norm, class-prior, masking과 내부 메커니즘 분리 |
| Ablation/patch 후 출력 변화 | intervention-specific causal effect | 정확한 site, sham, rescue, 다른 기능 보존 |
| 특정 개입으로 오류 감소 | 정의한 표본에서의 제어 가능성 | 오개입 부작용, 정답 보존, 다양한 오류·정상 분포 |
| 새 recurrence가 더 높은 성능 획득 | 비교 과제에서의 능력 차이 | matched compute/capacity, mechanism necessity, 일반화 |

**현재 단계에서 권고하는 논리:** 이미 frozen된 관측을 정확하게 설명하고 → 내부 원인에 관한 서로 배타적인 반증 가설을 좁힌 뒤 → 별도 authority가 있는 경우에만 최소 개입·복원 → *왜 효과가 생겼는지* 확인하는 계산 원리 식별 → 충분성/대체 실험으로 나아간다. 반대로 잘못된 판단의 전구체가 성립하지 않으면 가설을 폐기하고 정상 계산/실패 메커니즘 분석을 계속한다. 본 문서만으로 새 모델 호출, 학습, 평가, Kaggle 실행, Phase3B 전환을 승인하지 않는다.

## 10. 논문·세대·미래 아키텍처의 관계 [P]

| 단위 | 역할 | 금지되는 혼동 |
|---|---|---|
| 개별 논문(예: 계획된 ICLR 연구) | 좁고 재현 가능한 주장 | 논문 하나가 AGI 연구 철학 전체를 입증한다는 주장 |
| ContraMamba K/Gen4/Gen5 | 실패와 인과적 계산을 배우는 실험적 계보 | 현 모델 구조를 영구적인 정답으로 고정 |
| 뇌과학적 계산 가설 | 새 상태/학습/통신 구조에 대한 아이디어 공급 | 뇌의 구조적 유사성을 필요성 증거로 간주 |
| 향후 architecture | 발견한 원리를 구현·반증하고 능력을 확장 | 개념적 도식만으로 새 지능 능력을 달성했다고 선언 |

### 10.1 연구자의 전략적 신념과 반증 가능한 아키텍처 가설 [P/H]

이 절은 **실험으로 입증된 결론이 아니라, 왜 현재 연구 문제와 순서를 선택했는지에 관한 강한 전략적 입장**을 기록한다. 반증 가능한 가설로 유지하며 부정적인 결과를 삭제하거나 우회하지 않는다.

**언어 기반 추론과 신뢰성을 우선한다.** 현재의 핵심 연구 전략은 LLM의 근거 없는 사실적 확신, 증거 미반영, 환각 전구체 후보, 장기 문맥에서의 기억·추론 실패를 먼저 해부하고 제어하는 것이다. 아직 기초적인 판단 신뢰성을 설명하지 못한 구조에 복잡한 멀티모달 인지 또는 현실 세계의 행동을 결합하면 실패 경로가 더 복잡해지고 결과적 오류가 전파될 수 있다는 **위험 가설**이 동기다. 이는 멀티모달·행동 연구가 항상 부차적이거나 불필요하다는 주장도, 언어 모델 문제를 모두 '완결'한 뒤에야 확장할 수 있다는 고정된 선후 규칙도 아니다. 기본 신뢰성 메커니즘을 충분히 파악한 뒤 필요에 따라 순서를 갱신한다.

**Transformer에 대한 강한 아키텍처 가설을 명시한다.** 연구의 현재 전략적 입장은 **전통적인 attention 중심 Transformer만으로는** 초장기 기억, 지속적 상태·학습, 계산·메모리 비용 및 장기 추론의 확장성 문제 때문에 **궁극적인 ASI 아키텍처로 적합하기 어려울 것이라는 강한 예상**이다. 이는 실증적으로 반증할 수 있는 아키텍처 가설이지 입증된 불가능성 정리가 아니다. 이 믿음은 alternative recurrent/SSM, 하이브리드, 외부 기억, 가소적 상태 구조를 연구할 충분한 동기다. 그러나 **Transformer가 AGI/ASI를 원리적으로 달성할 수 없다고 증명된 것은 아니며**, Mamba나 다른 recurrent 계열의 우월성 또한 확립된 사실이 아니다. 기존 architecture는 비교 대상이지 미리 배제할 상대가 아니다.

**검증 가능한 예측으로 내린다.** 긴 문맥과 지연 의존성에 대한 기능 보존, 메모리 점유·대역폭·연산량·지연시간의 scaling, 온라인 갱신의 안정성·간섭·망각, 인과적 증거 통합·수정 가능성, 장기 계획의 일관성을 사전에 정의해 평가한다. Transformer뿐 아니라 long-context 개선, 검색/외부 메모리, recurrent hybrid를 포함한 **강한 최선의 대조군**과 동일한 과제·학습 데이터·능력 수준·적절히 정렬된 자원 예산에서 비교한다. Transformer/attention 계열이 해당 조건에서 기능·비용·일반화를 충족하면 '원리적으로 부적합하다'는 강한 가설은 수정하거나 기각한다. Recurrent 구조가 실패하면 그 구조를 보호하지 않는다.

**이것은 동일한 5단계 연구 프로그램 안의 문제 선택 원칙이다.** 언어 모델에서 내부 계산 → 전구체 후보 → 인과적 국소화 → 잠재공간 제어 → 계산 원리 추출을 수행하며, 실증된 원리를 이후 멀티모달·행동·일반지능으로 확장하거나 그 한계를 반증한다. 별개의 '환각 프로젝트가 종료된 뒤 시작하는 AGI 프로젝트'로 나누지 않는다.

### 10.2 장기 검증 대상의 확장 경로는 기능 중심으로 설계한다 [H/P]

미래 연구의 궁극적인 능력은 `memory`, `learning`, `reasoning`, `judgment`, `planning`, `adaptation`의 통합이다. 이를 단순히 독립 모듈 6개로 대응시키지 않는다. 예를 들어 장기 기억은 가소성 있는 연결, 압축된 recurrent state, 외부 검색 또는 그 결합으로 가능하며, 계획은 state-based rollout 또는 다른 계산으로 구현될 수 있다. 어느 경로가 충분한지 적절한 delayed-dependency, distribution shift, online-interference, compositional task, prospective intervention에서 비교해야 한다.

뇌과학적 동기는 연구 전 과정의 가설 공급원이지만, 해부학적 유사성이 곧바로 성능·일반화·안전성에 대한 증거가 되지는 않는다. 기존 Transformer/attention의 한계 및 Mamba의 이점도 절대명제가 아닌 **조건별 대조 가설**로 다룬다.

## 11. 장기 연구를 이어갈 때 지킬 운영 원칙 [P]

1. **내부 해부에서 시작하는 발견 순서 보존.** 2.4~2.6절의 01~05(내부 계산 원리 및 전구체 후보 발견 → 인과적 규명 → 잠재공간 제어 → 계산 원리 추출 → 아키텍처 설계)를 장기 연구의 기본 논리로 유지한다. 특정 전구체가 부정되더라도 전체 프로그램을 폐기하지 않는다.
2. **연구 목표 고정, 가설 수정 허용.** 최근 유행하는 한 질문이 전체 목적을 대체하지 않는다.
3. **수식 보존.** neuron-wise recurrence, compartmental dynamics, plasticity, **neuromodulated state/learning rules**, specialized states, low-rank intervention, compression, conditional complexity는 철학적 한 문장으로 축약해 지우지 않는다.
4. **가설 상태 라벨링.** 전사 원문 [O], 수학적 정식화 [M], 예측 [H], 실제 증거 [E], 방법론 [P]를 구별한다.
5. **실패를 편집하지 않는다.** frozen result와 부정적 관측은 뒤집어 쓰지 않는다.
6. **명시 구조와 자발 분화를 공정하게 비교한다.** 역할 분리가 필요하다는 답을 먼저 정하지 않는다.
7. **함수를 먼저 보고 구현을 나중에 고른다.** 다만 탐색적 구현으로 새로운 인과 가설을 생성할 수 있다.
8. **기능이 더 중요하나 기하학적 분석을 무시하지 않는다.** representation geometry는 가설 발굴 도구이지 곧바로 causal functional equivalence가 아니다.
9. **기술 난점들을 기록한다.** credit assignment, representation coordination, state interference, catastrophic forgetting, computation/communication cost, provenance를 회피하지 않는다.
10. **권위와 실행 경계를 유지한다.** 별도 허가 없는 Codex, 구현, training/evaluation, Kaggle, stage promotion, 강제 커밋/푸시는 없다.
11. **연구 전략과 입증 사실을 구별한다.** 언어 기반 추론 신뢰성 우선과 Transformer에 대한 강한 비판은 연구 가설·문제 선택의 근거이지 완성된 과학적 결론이 아니다.
12. **사실이 바뀌면 연구 결론도 수정한다.** 하지만 장기 지향점을 바꿀 때는 근거와 변경 범위를 문서로 남긴다.

### 11.1 미래 연구 제어기에 대한 기억 규칙 [P]

대화의 최근 강조점만 보고 목표를 재설정하지 않는다. 해당 시점의 실제 stage authority와 frozen 결과를 먼저 읽고, 이 문서는 장기 목적과 실험을 넘어선 설계 철학의 기준으로 삼는다. 연구 수행 중 사고의 변화가 생기면 **무엇이 새 사실이고, 무엇이 새로운 해석이며, 기존의 어떤 가설이 기각되는지**를 구분하여 후속 날짜 문서에 남긴다.

```text
research program (이 문서) : enduring why / architectural destination
older Vision / Map / Kinematics : historical mechanisms and competing hypotheses
stage authorities / manifests : execution scope at that stage
frozen reports / artifacts : what was actually observed
cm context / local git state : what the current worktree is
```

**문서의 우선순위를 높이는 것은 연구의 검증 기준을 낮추는 것과 반대다.** 이 문서는 최종 목적을 정의하지만 실험이 참인지 판단하는 권한은 주지 않는다.

## 12. 근거 문서와 계보 [Source map]

### 12.1 Repository sources independently located (reviewed ref: `f2c931a03b9103b3a5f4a7c6ed3eb75bcbb4378e`)

- `AGENTS.md` — repository research integrity and authority boundaries.
- `docs/CONTRAMAMBA_RESEARCH_VISION.md` — pre-existing long-term vision; **NON-AUTHORITY**.
- `docs/CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md` — native precursor first; `G_I/G_G/G_D`; independent-owner and null controls; hypothesis/stop rules; **NON-AUTHORITY**.
- `docs/CONTRAMAMBA_NATIVE_STATE_KINEMATICS_HYPOTHESIS.md` — native-state speed/acceleration/turning, temporality, decision-space factorization, confident-error vs hallucination; **NON-AUTHORITY**.
- `reports/gen5_cross_seed_bypass_loso_donor_operator_scientific_interpretation_b7212f8.md` — cross-seed geometry and bounded Phase2/3A conclusions.
- `reports/reason_router_gen5_g2_p0_class_collapse_scientific_findings_75ad31e.md` — frozen P0 class collapse.
- `reports/gen5_g2_p0_within_pair_decision_contrast_followup_8b70dd1.md` — exploratory, frozen within-pair logit contrast.
- `src/contramamba/gen5_g2_h1_layer22_output_observer.py` — read-only output observer; not a raw-write measurement.

Stable browser base for these reviewed files:
`https://github.com/9terry-student/ContraMamba/blob/f2c931a03b9103b3a5f4a7c6ed3eb75bcbb4378e/`

### 12.2 User-supplied retrospective sources [O]

The present conversation provides a lengthy reconstruction described as based on two past text files, including original mathematical expressions and the distinction between original ideas, later mathematical reconstructions, and executed ContraMamba science. Those two historical raw text files were **not independently available in this drafting context**. Accordingly this document identifies those equations as [O] per supplied transcription, not as independently verified verbatim archival quotations.

Earlier dated themes (May–October 2026) are retained as **researcher-supplied retrospective chronology**, not independently authenticated commit history. Use the exact earlier source if historical attribution becomes part of a scholarly claim.

### 12.3 Versioning policy

- This document is the **2026-10-16 research-only five-stage revised proposed snapshot** aligned to the three historical Git documents; neuromodulation and strategic architecture hypotheses were added before initial Git freeze. It does not attest that Git staging, commit, push, or upstream publication has occurred.
- Once Git-frozen, this document is **program-level conceptual successor** to older Vision/Map/Kinematics; the older documents are retained as earlier hypotheses, not erased. Old generation labels and stage statuses remain historically dated.
- Earlier chat-generated `2026_10_15` drafts were working drafts; do not silently replace an already-staged local copy without checking its Git status.
- If the research philosophy materially changes later, create a date-stamped successor and record *what changed and why*. Frozen experiment reports remain unchanged.

---

## 13. 변경 내용과 legacy 문서의 정확한 정렬 [P]

| Existing file | 유지할 실질적 내용 | 후속 해석으로 교정할 점 | 권장 조치 |
|---|---|---|---|
| `CONTRAMAMBA_RESEARCH_VISION.md` | F/P/S/Q 구조, G_I/G_G, A0–A3 causal controls, falsification, long-term fidelity | semantic ownership을 지능의 유일한 최종 아키텍처 철학으로 읽지 않음; 구 `Generation 1–6`은 실제 Gen5 연대기가 아님 | 파일 상단에 본 문서의 계승 사실과 역사적 시점 배너만 추가 |
| `CONTRAMAMBA_RESEARCH_HYPOTHESIS_MAP.md` | Q1 native precursor first, Q2 constructed precursor, G_I/G_G/G_D, no-cross-owner baseline, M0–M4 비교, 실패 taxonomy | candidate `NO CROSS-OWNER COMMUNICATION`을 영구적인 범용 지능 원리로 해석하지 않음 | 상단 배너로 새 long-term precedence를 명시하고 후보의 역사성 보존 |
| `CONTRAMAMBA_NATIVE_STATE_KINEMATICS_HYPOTHESIS.md` | raw native SSM trajectory, token time, decision-space factorization, confident-error controls, falsification | 작성 당시 `DO NOT EXECUTE YET`는 2026-09-11의 stage 설명이며 현재 활동 상태가 아님 | 상단 배너만 추가, 수식·가설·falsification을 변경하지 않음 |

이러한 배너의 목적은 기존 문서를 삭제·평탄화하는 것이 아니라, 후속 연구자가 이전 가설에 *현재의 최종 목표*라는 지위를 잘못 부여하지 않도록 하는 데 있다. 세 문서의 실제 수식, 기존 실험 규칙, 이미 실행한 결과는 이 변경으로 재해석·변형되지 않는다.

## Closing statement

> **내부 계산과 오류 전구체를 발견하고, 인과적 역할을 규명하며, 잠재 상태에 개입하여 제어 가능한지 검증하고, 왜 효과가 발생했는지 설명하여 필요한 계산 원리를 추출한 뒤, 그 원리로 더 나은 지능 아키텍처를 발명한다.** 뇌와 현재 인공지능은 출발점이지 족쇄가 아니다. 필요한 복잡성은 도입하고, 불필요한 복잡성은 제거한다. 실험의 부정적 결과까지 축적하여 기억·학습·추론·판단·계획을 더 신뢰성 있게 수행하는 일반 지능 아키텍처를 설계하고, 궁극적으로 인간 지능의 한계를 넘어서는 지능을 지향한다.
