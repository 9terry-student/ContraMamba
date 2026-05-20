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
| `True` (2) | 맞음 — entailment |
| `Unknown` (1) | 모름 — neutral |
| `False` (0) | 틀림 — contradiction |

---

### 2. Single Shared Semantic Manifold + Residual Contradiction

v7의 핵심 구조 변경. 기존 SSM frequency decomposition(A_log 기반 채널 분리) 대신,
**prototype routing 결과를 기준으로 두 branch를 의미론적으로 분리**:

```
h  (Mamba pooled hidden)
│
├─ prototype routing → z_semantic
│
├─ low branch:  h_low  = z_semantic + 0.1 · h    ← semantic refinement
│
└─ high branch: h_high = h - stopgrad(z_semantic) ← residual contradiction
```

- `h_low` : prototype manifold 위의 성분 → entailment / contradiction 방향
- `h_high` : prototype으로 설명되지 않는 잔차 → 모순 / 노이즈 / 미지 신호
- `h_low + h_high ≈ h` → 정보 손실 없음 (stopgrad로 gradient 분리)

**SSMFrequencyDecomposition 제거** — W_refine, W_gate, A_log mask 불필요.

---

### 3. Shared Prototype Manifold

공유 프로토타입 행렬 `P_shared ∈ ℝ^(n_proto × hidden)`에서 단일 뷰를 투영:

```
P_low  = F.normalize(LayerNorm(P_shared @ W_low^T),  dim=1)
P_high = F.normalize(LayerNorm(P_shared @ W_high^T), dim=1)

q_low  = softmax(normalize(h_low)  @ P_low^T  / τ)
q_fused = q_low   (v7: high bank 통합 전 단일 routing)

z_semantic = q_fused @ P_norm
```

- context-adaptive: `ctx_encoder`로 배치별 P_shared 미세 조정
- EMA usage tracking으로 prototype collapse 방지
- dead prototype rescue: 10 step 이상 미사용 prototype 자동 재초기화

---

### 4. PrototypeController (FiLM + Sample-wise EMA)

routing 결과를 기반으로 **low branch만** FiLM 교정:

```
z_cond  = q_ema_i @ P_norm             ← sample-wise EMA q로 conditioning
γ, β    = FiLM_net(z_cond)             ← scale + shift 분리
gate    = conf(q) · focus(q)           ← entropy/confidence 곱 게이트
h_low_hat = h_low + gate · (γ·LN(h_low) + β - h_low)
```

- **Sample-wise EMA** (slot_size=512): 배치 평균 대신 샘플별 독립 EMA
- **Entropy/Confidence gate**: routing이 불확실할 때 교정 억제
- **FiLM modulation**: residual 덧셈 대신 scale + shift 분리로 방향·크기 독립 조정
- Mamba layer 중간 삽입 없음 — routing 이후에만 동작

---

### 5. Prototype Co-Activation GAT

배치 내 샘플들의 공동 활성화로 prototype 전파:

```
A_ij   = softmax(q_fused^(i)^T · q_fused^(j) / √B)
P_new  = W(A @ P_shared)
q_fused_refined = softmax(normalize(h_enriched) @ normalize(P_new)^T / τ)
```

---

### 6. Geometric Energy Space

```
e_pos, e_neg = EnergyHead(q_fused, h_enriched)
ratio = e_pos / (e_pos + e_neg)     ← truth polarity
r     = sqrt(e_pos² + e_neg²)       ← confidence radius

h_high_norm  = ‖h_high‖             ← contradiction intensity
ratio_high   = sigmoid(h_high_norm - 1.0)
```

| 영역 | 조건 |
|---|---|
| True | `r > r_min` & `ratio > 0.5 + δ` |
| False | `r > r_min` & `ratio < 0.5 - δ` |
| ignorance | `r ≤ r_min` |
| ambiguity | `r_min < r` & `ratio ≈ 0.5` & `r_high < r_max` |
| contradiction | `r_min < r` & `ratio ≈ 0.5` & `r_high ≥ r_max` |

**h_high의 역할 변화**: v7부터 `h_high_norm`이 prototype 밖 에너지를 직접 측정.  
레이블별 타깃:
- entailment → `h_high_norm` 억제 (prototype에 잘 투영됨)
- contradiction → `h_high_norm` 증가 (manifold 밖에 위치)
- neutral → `h_high_norm ≈ 1.0` (중간)

---

### 7. Prototype Memory Bank

episodic + semantic cache 이중 메모리:

```
write: q_fused > threshold 인 prototype 슬롯에 z_input 기록 (circular buffer)
read:  top-3 prototype 기준으로 episodic + semantic 혼합 검색
       h_enriched = h + fusion_gate(anchor ‖ ep_summary ‖ cache)
```

---

### 8. Graph Memory + Local GAT Refinement

FAISS index로 훈련 셋에서 top-K cosine neighbor 검색 후 Local GAT로 정제:

```
h̃_i = clamp(h_i + λ · GAT(G, h_i), min=0)
```

- Low GAT: q_fused cosine similarity 기반 proto-aware edge
- High GAT: h_high_norm 기반 contradiction-aware edge
- label gating으로 동일 레이블 이웃에 더 높은 가중치

---

### 9. 왜 Mamba인가?

- 추론 시간 O(n) — Transformer의 O(n²) 대비 효율적
- SSM 구조가 선형 시불변 필터와 동일한 수학적 구조
- HiPPO 초기화가 자연스럽게 직교 기저로 시퀀스를 투영

---

## 아키텍처 (v7)

```
                 ┌─────────────────────┐
                 │   Raw Semantic h    │
                 │ (Mamba + AttnPool)  │
                 └─────────┬───────────┘
                           │
               Shared Prototype Manifold
               (P_shared, ctx_encoder)
                           │
                    z_semantic = q @ P
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
 ┌─────────────────┐              ┌──────────────────────┐
 │     h_low       │              │       h_high         │
 │ z_sem + 0.1·h   │              │  h - stopgrad(z_sem) │
 │ semantic anchor │              │  residual / contrast │
 └────────┬────────┘              └────────┬─────────────┘
          │                                │
          ▼                                │
 ┌─────────────────┐                       │
 │PrototypeCtrl    │                       │
 │ FiLM + EMA gate │                       │
 │ (low only)      │                       │
 └────────┬────────┘                       │
          │                                │
          └──────────────┬─────────────────┘
                         │
              Memory Bank + CoAttn GAT
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 ┌─────────────────┐          ┌─────────────────┐
 │ h_enriched      │          │  h_high_norm    │
 │ semantic anchor │          │ contradiction   │
 │ (memory 보강)   │          │    intensity    │
 └────────┬────────┘          └────────┬────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
               ┌──────────────────┐
               │  Energy Geometry │
               │  ratio / radius  │
               │  uncertainty type│
               └────────┬─────────┘
                        ▼
                 Final Prediction
            {True / Unknown / False}
```

---

## Loss 구성

```
L_total = L_direction       (pos/neg margin loss)
        + L_radius          (confidence floor)
        + L_spread          (pos-neg center 분리)
        + L_unknown         (neutral → ratio ≈ 0.5 유도)
        + L_high_supervision(h_high_norm label-aware 타깃)
        + L_proto           (VQ commit + uniformity + diversity + entropy)
        + L_velocity        (prototype 이동 속도 제한)
        + L_routing_consist (두 view KL consistency)
        + L_load_balance    (prototype 사용 균등화)
        + L_anticollapse    (low-high cosine 억제)
        + L_orth            (W_low ⊥ W_high)
        + L_align           (energy head - prototype 정렬)
        + L_gat             (graph neighbor contrastive)
```

---

## 학습 스케줄

| Epoch | 활성화 모듈 |
|---|---|
| 0–1 | Backbone (Mamba) + AttentionPooling |
| ≥ 2 | + Local GAT (graph-assisted training) |
| ≥ 3 | + Prototype Manifold + PrototypeController (FiLM EMA) |
| ≥ 5 | + High GAT / GAT loss → representation alignment |

---

## 실험 설정

- **모델**: `state-spaces/mamba-130m-hf`
- **데이터셋**: SNLI (train 5,000 / val 1,000)
- **태스크**: 3값 분류 (entailment / neutral / contradiction)
- **레이블 매핑**: entailment → 2, neutral → 1, contradiction → 0
- **배치 크기**: 16
- **시퀀스 길이**: 64 (max_length)

---

## 결과 (이전 버전, BoolQ 기준)

| 모델 | Val Accuracy | 비고 |
|---|---|---|
| Mamba-130m 베이스라인 | 0.6217 | 질문만 입력 |
| ContraMamba v2 | 0.6780 | 질문+지문, classifier |
| ContraMamba v3 (threshold=8.5) | 0.7005 | Unknown 제외 정확도 |
| ContraMamba v4 (geometric) | 0.7209 | Known만 선별, geometric classifier |

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `n_proto` | 16 | prototype 개수 |
| `proto_tau` | 0.1 | routing softmax 온도 |
| `warmup_epochs` | 3 | prototype / controller 활성화 시점 |
| `split_epoch` | 5 | GAT representation alignment 전환 시점 |
| `ctrl_ema_decay` | 0.95 | sample-wise EMA decay |
| `ctrl_slot_size` | 512 | EMA 순환 슬롯 수 |
| `k` (FAISS) | 15 | graph memory neighbor 수 |
| `delta` | 0.1 | geometric classification margin |
| `r_min` | 0.5 | confidence 최소 threshold |
| `mem_size` | 50 | episodic memory 슬롯 수 (proto별) |
| `gat_alpha` | 0.3→0.1 | GAT residual 강도 (epoch 따라 감소) |

---

## 의존성

```
torch
transformers        # MambaModel, MambaConfig
torch-geometric     # GATConv, add_self_loops
faiss-cpu / faiss-gpu
datasets            # HuggingFace SNLI
numpy
matplotlib
tqdm
```

---

## 버전 히스토리

| 버전 | 주요 변경 |
|---|---|
| v2 | 3값 출력 헤드 + classifier |
| v3 | Unknown threshold 기반 선별 |
| v4 | Geometric Energy Space 도입 |
| v5 | Shared Prototype Manifold + Co-Activation GAT + Graph Memory |
| v6 | PrototypeController (FiLM + EMA gating) + prototype normalize 안정화 |
| v7 | **SSM decomp 제거** / h_high = h − stopgrad(z_semantic) 으로 재정의 / Sample-wise EMA Controller / High branch residual contradiction 구조화 |
| v8 | High prototype bank (P_high_bank) / disagreement loss / contrastive separation loss |

---

## 로드맵

- [x] S⁺/S⁻ 직교 subspace 설계
- [x] Attention Pooling
- [x] 3값 출력 헤드 + Geometric Energy Space
- [x] Shared Prototype Manifold + soft routing
- [x] Co-Activation GAT (prototype propagation)
- [x] Graph Memory (FAISS) + Local GAT refinement
- [x] EMA usage tracking + prototype diversity loss
- [x] Label-aware edge gating
- [x] PrototypeController — FiLM + sample-wise EMA gate
- [x] Single semantic manifold + residual contradiction (v7)
- [x] High branch prototype bank (독립 P_high_bank)
- [x] Low-High disagreement loss
- [x] Contrastive separation loss (h_low ⊥ h_high)
- [x] Dual graph inference 완전 연동
- [ ] 동적 해빙 (low slow / high fast adaptation)
- [ ] 복소수 SSM 확장 (S4 계열)
- [ ] 생성 태스크 확장 및 역질문 모듈

---

## 한계 및 향후 연구

**Residual contradiction의 노이즈 혼재**  
`h_high = h - stopgrad(z_semantic)`는 prototype으로 설명되지 않는 모든 성분을 담으므로
순수 contradiction 신호와 학습 초기 노이즈가 섞일 수 있음.
`high_supervision_loss`의 lambda 스케줄로 점진적 분리 유도.

**Prototype collapse**  
soft routing + EMA usage tracking으로 완화하고 있으나, 초기 학습 단계에서
특정 prototype으로 쏠리는 현상이 발생할 수 있음. `visualize_usage()`로 per-epoch 확인 권장.

**Energy head 방향성 검증**  
`ratio_low True mean > ratio_low False mean` gap이 epoch 진행에 따라
벌어지지 않으면 EnergyHead polarity 초기화 방향 재검토 필요.

**Controller gate 포화**  
`ctrl_scale`이 너무 커지면 FiLM이 h_low를 과도하게 교정.
학습 중 `ctrl_gate` / `ctrl_scale` 모니터링 권장.
