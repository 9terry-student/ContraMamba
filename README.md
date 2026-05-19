# ContraMamba

> 환각 감소를 위한 명시적 불확실성 출력 기반 Mamba 분류 모델

## 동기

기존 LLM은 틀린 답도 확신을 갖고 생성하는 환각 문제가 있음.
ContraMamba는 모델이 불확실성을 명시적으로 표현하도록 강제하는
**3값 출력 헤드**를 도입해 이 문제를 완화하고자 함.

---

## 핵심 아이디어

### 1. 3값 출력 {False, Unknown, True}

소프트맥스 확률 대신 세 가지 상태 중 하나를 출력:

| 출력 | 의미 |
|---|---|
| `True` | 맞음 |
| `False` | 틀림 |
| `Unknown` | 모름 (역질문 트리거 가능) |

### 2. Frequency Decomposition 기반 표현

Mamba hidden state `h`를 smooth component와 residual로 분해:

```
h = h_low + h_high
h_low  = IDCT(DCT(h)[:k]) + W_low @ h   (smooth semantic structure)
h_high = h - h_low                        (residual: exception / anomaly)
```

- `h_low` : 안정적이고 일반적인 패턴 — semantic consensus
- `h_high` : DCT로 설명되지 않는 날카로운 성분 — contradiction / ambiguity

### 3. Shared Prototype Manifold

공유 프로토타입 행렬 `P_shared ∈ ℝ^(m×d)`로부터 두 개의 뷰를 투영:

```
P_low  = Norm(P_shared @ W_low^T)
P_high = Norm(P_shared @ W_high^T)

q_low  = softmax(ĥ_low  @ P_low^T  / τ)
q_high = softmax(ĥ_high @ P_high^T / τ)

q_fused = α·q_low + (1-α)·q_high       (α는 학습된 gate)
```

- 두 주파수 뷰가 같은 프로토타입 공간을 공유 → concept 일관성 유지
- EMA 기반 usage tracking으로 prototype collapse 방지

### 4. Prototype Co-Activation GAT

배치 내 샘플들의 공동 활성화로 prototype을 업데이트:

```
A_ij   = q_fused^(i)^T · q_fused^(j)
P_new  = softmax(A) @ P_shared
```

attention-weighted prototype propagation으로 배치 내 semantic 구조 반영.

### 5. Geometric Energy Space

```
ratio = e_pos / (e_pos + e_neg)   ← truth polarity
r     = 1 - H(q_fused) / log(m)  ← entropy-based confidence
```

| 영역 | 조건 |
|---|---|
| True | `r > r_min` & `ratio > 0.5 + δ` |
| False | `r > r_min` & `ratio < 0.5 - δ` |
| ignorance | `r ≤ r_min` |
| ambiguity | `r_min < r < r_max` & `ratio ≈ 0.5` |
| contradiction | `r ≥ r_max` & `ratio ≈ 0.5` |

### 6. Graph Memory + Local GAT Refinement

FAISS index로 훈련 셋에서 top-K cosine neighbor를 검색하고, Local GAT로 기하 표현을 정제:

```
h̃_i = h_i + λ · GAT(G, h_i)
```

엣지 가중치는 `q_fused` cosine similarity 기반 (proto-aware routing). label gating으로 동일 레이블 이웃에 더 높은 가중치 부여.

### 7. 왜 Mamba인가?

- 추론 시간 O(n) — Transformer의 O(n²) 대비 효율적
- SSM 구조가 선형 시불변 필터와 동일한 수학적 구조
- HiPPO 초기화가 자연스럽게 직교 기저로 시퀀스를 투영

---

## 아키텍처 (v5)

```
                 ┌─────────────────────┐
                 │   Raw Semantic h    │
                 │ (Mamba hidden state)│
                 └─────────┬───────────┘
                           │
                Frequency Decomposition
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
 ┌─────────────────┐              ┌─────────────────┐
 │      h_low      │              │     h_high      │
 │ stable semantics│              │ residual/detail │
 └────────┬────────┘              └────────┬────────┘
          │                                │
          │ Prototype Routing              │ Prototype Routing
          ▼                                ▼
 ┌─────────────────┐              ┌─────────────────┐
 │      q_low      │              │     q_high      │
 │ prototype probs │              │ prototype probs │
 └────────┬────────┘              └────────┬────────┘
          │                                │
          ▼                                ▼
 ┌─────────────────┐              ┌─────────────────┐
 │      z_low      │              │     z_high      │
 │ q_low @ P_low   │              │ q_high @ P_high │
 └────────┬────────┘              └────────┬────────┘
          │                                │
          │ Reconstruction Residual        │
          ▼                                ▼
 ┌─────────────────┐              ┌─────────────────┐
 │ r_low=h_low-z_l │              │r_high=h_high-z_h│
 │ unexplained low │              │ unexplained high│
 └────────┬────────┘              └────────┬────────┘
          │                                │
          └──────────────┬─────────────────┘
                         │
               Memory / Graph Refinement
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 ┌─────────────────┐          ┌─────────────────┐
 │ refined_low     │          │ refined_high    │
 │ semantic anchor │          │ uncertainty cue │
 └────────┬────────┘          └────────┬────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
               ┌──────────────────┐
               │ Energy Geometry  │
               │ ratio / radius   │
               │ uncertainty type │
               └────────┬─────────┘
                        ▼
                 Final Prediction
```

---

## 이론적 배경

### Parseval 정리와 메타인지 출력

DCT 기반 직교 분해에서 Parseval 정리가 성립:

> 입력의 총 에너지 = low-freq 에너지 + high-freq 에너지

이를 통해 3값 출력을 에너지 관점에서 정의:

- `True` : h_low의 S⁺ subspace에 에너지 집중
- `False` : h_low의 S⁻ subspace에 에너지 집중
- `Unknown` : confidence가 낮거나 ratio ≈ 0.5 → evidence 부족 또는 충돌

### DCT + Learned Projection (Hybrid)

- DCT: 안정적인 직교 기저 보장 (고정)
- `W_low`: semantic flexibility 추가 (학습)
- Frobenius norm regularization으로 W_low가 DCT를 압도하는 것을 방지

### Prototype Loss 구성

```
L_proto = 0.5 · (VQ commitment + VQ codebook)
        + 0.05 · uniformity loss      (prototype spread)
        + 0.10 · diversity loss       (inter-prototype orthogonality)
        + 0.01 · entropy loss         (usage balance)
        + 0.10 · W_low ⊥ W_high       (view independence)
```

---

## 학습 스케줄

| Epoch | 활성화 모듈 |
|---|---|
| 0–1 | Backbone + Decomp only |
| ≥ 2 | + Local GAT (graph-assisted training) |
| ≥ 3 | + Prototype Manifold (proto-aware routing) |
| ≥ 5 | GAT loss → representation alignment (cosine) |

`warmup_epochs` 파라미터로 전환 시점을 조정할 수 있음.

---

## 실험

- **모델**: `state-spaces/mamba-130m-hf`
- **데이터셋**: SNLI (train 5000 / val 1000)
- **태스크**: 3값 분류 (entailment / neutral / contradiction)
- **레이블 매핑**: entailment → 2 (True), neutral → 1 (Unknown), contradiction → 0 (False)

---

## 결과 (이전 버전, BoolQ 기준)

| 모델 | Val Accuracy | 비고 |
|---|---|---|
| Mamba-130m 베이스라인 | 0.6217 | 질문만 입력 |
| ContraMamba v2 | 0.6780 | 질문+지문, classifier |
| ContraMamba v3 (threshold=8.5) | 0.7005 | Unknown 제외 정확도 |
| ContraMamba v4 (geometric) | 0.7209 | Known만 선별, geometric classifier |

### 삼진논리 출력 분포 (v4, r_min=1.0)

- Known 정확도: 0.7209
- Unknown 비율: 30.2%
- Unknown 중 실제 True: 58.9% / False: 41.1%

### 에너지 분리 결과 (v2)

| 샘플 | S⁺ 에너지 | S⁻ 에너지 | 갭 |
|---|---|---|---|
| True 샘플 | 7.41 | 3.83 | +3.58 |
| False 샘플 | 5.19 | 6.03 | +0.84 |

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_proto` | 16 | prototype 개수 |
| `low_ratio` | 0.75 | DCT 저주파 보존 비율 |
| `proto_tau` | 0.1 | routing softmax 온도 |
| `warmup_epochs` | 3 | prototype 활성화 시작 epoch |
| `k` (FAISS) | 15 | graph memory neighbor 수 |
| `delta` | 0.1 | geometric classification margin |
| `r_min` | 0.5 | confidence 최소 threshold |

---

## 의존성

```
torch
transformers        # MambaModel
torch-geometric     # GATConv
faiss-cpu / faiss-gpu
datasets            # HuggingFace
numpy
matplotlib
tqdm
```

---

## 로드맵

- [x] S⁺/S⁻ 직교 subspace 설계
- [x] DCT 초기화 + Attention Pooling
- [x] 3값 출력 헤드 + Geometric Energy Space
- [x] Hybrid DCT + Learned projection
- [x] Frequency decomposition (h = h_low + h_high)
- [x] Shared Prototype Manifold + soft routing
- [x] Co-Activation GAT (prototype propagation)
- [x] Graph Memory (FAISS) + Local GAT refinement
- [x] EMA usage tracking + prototype diversity loss
- [x] Label-aware edge gating
- [x] Warmup-scheduled proto / GAT 활성화
- [ ] Dual graph inference 완전 연동 (v5 진행 중)
- [ ] 동적 해빙 (low-rate slow / high-rate fast adaptation)
- [ ] 복소수 SSM 확장 (S4 계열)
- [ ] 생성 태스크 확장 및 역질문 모듈

---

## 한계 및 향후 연구

**Frequency decomposition의 근사성**
DCT는 signal domain에서 정의된 연산이므로, semantic space에서의 "주파수"는 근사적 개념임. `W_low` norm이 커지면 DCT 기여를 압도할 수 있어 모니터링 필요.

**Prototype collapse**
soft routing + EMA usage tracking으로 완화하고 있으나, 초기 학습 단계에서 특정 prototype으로 쏠리는 현상이 발생할 수 있음. `visualize_usage()`로 per-epoch 확인 권장.

**Energy head 방향성 검증**
`ratio_low True mean > ratio_low False mean` gap이 epoch 진행에 따라 벌어지지 않으면 `polarity` 파라미터 초기화 방향을 재검토해야 함.

**동적 해빙**
현재 `A_log` 파라미터만 freeze. low component는 slow adaptation, high component는 fast adaptation하는 동적 메커니즘은 향후 연구 과제.
