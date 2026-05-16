# ContraMamba

> 환각 감소를 위한 명시적 불확실성 출력 기반 Mamba 분류 모델

## 동기

기존 LLM은 틀린 답도 확신을 갖고 생성하는 환각 문제가 있음.
ContraMamba는 모델이 불확실성을 명시적으로 표현하도록 강제하는
**3값 출력 헤드**를 도입해 이 문제를 완화하고자 함.

## 핵심 아이디어

### 1. 3값 출력 {False, Unknown, True}
소프트맥스 확률 대신 세 가지 상태 중 하나를 출력:
- `True`    : 맞음
- `False`   : 틀림
- `Unknown` : 모름 (역질문 트리거 가능)

### 2. Frequency Decomposition 기반 표현
Mamba hidden state `h`를 smooth component와 residual로 분해:
```
h = h_low + h_high
h_low  = DCT_low(h) + W_low @ h   (smooth semantic structure)
h_high = h - h_low                 (residual: exception / anomaly)
```
- `h_low`  : 안정적이고 일반적인 패턴 — semantic consensus
- `h_high` : DCT로 설명되지 않는 날카로운 성분 — contradiction / ambiguity
- 둘은 독립 공간이 아니라 **같은 공간의 decomposition**

### 3. Geometric Energy Space
각 component에서 e_pos / e_neg를 추출하고,
(e_pos, e_neg) 좌표계에서 기하학적으로 분류:
```
ratio = e_pos / (e_pos + e_neg)   ← truth polarity
r     = sqrt(e_pos² + e_neg²)     ← evidence strength
```
- true region      : ratio > 0.5 + δ
- false region     : ratio < 0.5 - δ
- ignorance        : r < r_min
- ambiguity        : r_min ≤ r < r_max, ratio ≈ 0.5
- contradiction    : r ≥ r_max, ratio ≈ 0.5

### 4. 왜 Mamba인가?
- 추론 시간 O(n) — Transformer의 O(n²) 대비 효율적
- SSM 구조가 신호 및 시스템의 선형 필터와 동일한 수학적 구조
- HiPPO 초기화가 자연스럽게 직교 기저로 시퀀스를 투영

## 이론적 배경

### Parseval 정리와 메타인지 출력
DCT 기반 직교 분해에서 Parseval 정리가 성립:
입력의 총 에너지 = low-freq 에너지 + high-freq 에너지

이를 통해 3값 출력을 에너지 관점에서 정의:
- `True`    : h_low의 S+ subspace에 에너지 집중
- `False`   : h_low의 S- subspace에 에너지 집중
- `Unknown` : r이 작거나 ratio ≈ 0.5 → evidence 부족 or 충돌

### DCT + Learned Projection (Hybrid)
- DCT: 안정적인 직교 기저 보장 (고정)
- W_low: semantic flexibility 추가 (학습)
- W_low norm을 regularization으로 제한 → DCT 기여 압도 방지

### 에너지 해석
e_pos / e_neg는 투영 강도(projection strength):
- 클수록 해당 방향에 강하게 속함
- h_low 기반 e: semantic consensus 강도
- h_high 기반 e: contradiction / anomaly 강도

### Dual GAT — Semantic Energy Routing
그래프를 단순 smoothing이 아닌 semantic energy routing으로 사용:
```
low-GAT  : h_low 기반 dense graph   → consensus propagation
high-GAT : h_high 기반 sparse graph → discontinuity modeling
```
- low graph  엣지: cosine sim ≥ 0.7 (stable neighbor)
- high graph 엣지: cosine sim ≥ 0.5 (uncertain neighbor)
- node embedding: concat(e_pos, e_neg, r) — 공통 base 위에서 분리

## 실험

- **모델**: Mamba-130m (파인튜닝)
- **데이터셋**: BoolQ
- **태스크**: 이진 QA → 3값 분류
- **레이블**: True → 2, False → 0, Unknown → 1 (추론 시 geometric threshold)

## 결과

| 모델 | Val Accuracy | 비고 |
|---|---|---|
| Mamba-130m 베이스라인 | 0.6217 | 질문만 입력 |
| ContraMamba v2 | 0.6780 | 질문+지문, classifier |
| ContraMamba v3 (threshold=8.5) | 0.7005 | Unknown 제외 정확도 |
| ContraMamba v4 (geometric) | 0.7209 | Known만 선별, geometric classifier |

### 삼진논리 출력 분포 (v4, threshold r_min=1.0)
- Known 정확도: 0.7209
- Unknown 비율: 30.2%
- Unknown 중 실제 True: 58.9% / False: 41.1%
- 해석: 불확실한 샘플을 Unknown으로 유보 → 메타인지 작동 증거

### 에너지 분리 결과 (v2)
| 샘플 | S+ 에너지 | S- 에너지 | 갭 |
|---|---|---|---|
| True 샘플 | 7.41 | 3.83 | +3.58 |
| False 샘플 | 5.19 | 6.03 | +0.84 |

## 아키텍처 (Phase 1)

```
Mamba-130m
    ↓
AttentionPooling → h ∈ R^768
    ↓
HybridFrequencyDecomposition
    ├── h_low  = IDCT(DCT(h)[:576]) + W_low @ h   (75%)
    └── h_high = h - h_low                         (25%)
    ↓
EnergyHead (각각)
    ├── e_pos, e_neg = (W @ h_sub)²
    └── ratio, r 계산
    ↓
DualGraphBuilder (FAISS)
    ├── low-GAT  : dense,  k=15, sim ≥ 0.7
    └── high-GAT : sparse, k=8,  sim ≥ 0.5
    ↓
Geometric Classifier
    → True / False / ignorance / ambiguity / contradiction
```

## 로드맵

- [x] S+/S- 직교 subspace 설계
- [x] DCT 초기화 + Attention Pooling
- [x] 비대칭 Gap Loss (v1)
- [x] 개별 Subspace Freeze (v2)
- [x] 3값 출력 헤드
- [x] Threshold 기반 Unknown 판단
- [x] Threshold Loss로 학습
- [x] Unknown 샘플 분석 (v3)
- [x] Geometric Energy Space 기반 Unknown State Modeling (v4)
- [x] Frequency decomposition (h = h_low + h_high)
- [x] Hybrid DCT + Learned projection
- [x] Dual GAT (low / high 분리)
- [ ] Dual graph inference 연동 (v5)
- [ ] Phase 2: concept clustering / prototype routing (v6)
- [ ] 동적 해빙 (v7)
- [ ] 생성 태스크 확장 (v8)

## 한계 및 향후 연구

### Frequency Decomposition의 트레이드오프
- DCT는 signal domain에서 정의된 연산 — semantic space에서의 "frequency"는 근사적 개념
- W_low가 커지면 DCT 기여를 압도할 수 있음 → W_low norm 모니터링 필수
- h_high = h - h_low 정의상 직교하지만, energy head가 실제로 exception/anomaly를 학습하는지는 실험으로 확인 필요

### Phase 2: Concept Formation
- h = h_low + h_high는 decomposition이고, 여기서 concept이 자동으로 나오지 않음
- concept emergence는 별도의 clustering / routing 메커니즘 필요
- 계획: z = concat(h_low, h_high) 위에서 prototype learning 또는 graph community detection

### 동적 해빙 (Dynamic Unfreezing)
- low component는 slow adaptation (낮은 lr), high component는 fast adaptation
- 강한 반증 감지 시 low 방향도 미세 조정하는 동적 메커니즘 연구 예정
- 관련 분야: Continual Learning, Catastrophic Forgetting 방지

### 향후 연구 방향
- [ ] Phase 2: concept prototype routing (mixture of experts)
- [ ] alternating training (backbone ↔ decomposition 교대 학습)
- [ ] 복소수 SSM으로 확장 (S4 계열)
- [ ] 생성 태스크 확장 및 역질문 모듈 구현
