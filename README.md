# ContraMamba v10

> **Mamba 기반 명시적 불확실성 분류 모델**  
> Semantic manifold / Topology manifold 이중 공간 + episodic memory + FAISS geometric retrieval

-----

## 동기

기존 LLM은 틀린 답도 확신을 갖고 생성하는 **환각(hallucination)** 문제가 있다.  
ContraMamba는 모델이 불확실성을 **명시적으로 표현**하도록 강제하는 3값 출력 헤드와,  
의미 공간과 위상 공간을 기하학적으로 분리하는 dual manifold 구조를 통해 이 문제를 완화한다.

-----

## 핵심 철학 (v10)

```
LOW  = semantic attractor space   → "무엇을 의미하는가"
HIGH = orthogonal topology space  → "얼마나 불확실한가"
FAISS = long-term geometric memory → "과거 기하 압축"
Episodic = event compression buffer → "충돌·새로움 기록"
Controller = LOW-space dynamic modulation → "slot EMA로 안정화"
```

**핵심 한 줄:**  
LOW는 *의미 결정*, HIGH는 *존재 불확실성 기하*, MEMORY는 *충돌 기록*, FAISS는 *과거 기하 압축*

-----

## 3값 출력

소프트맥스 확률 대신 세 가지 상태를 기하학적으로 결정:

|출력       |레이블|의미                |
|---------|---|------------------|
|`True`   |2  |Entailment — 맞음   |
|`Unknown`|1  |Neutral — 모름      |
|`False`  |0  |Contradiction — 틀림|

-----

## 아키텍처 (v10)

```
INPUT: input_ids
        │
        ▼
┌─────────────────────────────┐
│  Mamba Backbone (130M)      │
│  + Attention Pooling        │
│  h ∈ ℝ^hidden              │
└─────────────┬───────────────┘
              │
     ┌────────┴─────────┐
     ▼                  ▼
┌──────────────────┐  ┌───────────────────────────────┐
│  LOW: Semantic   │  │  HIGH: Topology               │
│  Attractor Space │  │  Orthogonal Residual Space    │
│                  │  │                               │
│  P_low routing   │  │  semantic_residual            │
│  z_semantic      │  │  = h - proj(z_low)            │
│  h_low = z_sem   │  │                               │
│       + 0.1·h    │  │  Gram-Schmidt orthogonalize   │
│                  │  │  h_high_base ⊥ z_low          │
│  Controller      │  │                               │
│  (FiLM + EMA)    │  │  P_high routing               │
│                  │  │  topology_key                 │
│  Energy Head     │  │  = h_high_base - z_high       │
│  ratio / r_low   │  │                               │
└──────────────────┘  │  FAISS retrieval              │
                      │  r_ctx → HIGH biasing only    │
                      │                               │
                      │  epistemic decomposition      │
                      │  novelty / contradiction /    │
                      │  ambiguity / ignorance        │
                      └───────────────┬───────────────┘
                                      │
                              ┌───────┴────────┐
                              ▼                ▼
                         WRITE (novelty    READ (proto +
                         or contradiction  FAISS neighbor
                         triggered)        weighted fusion)
                              └───────┬────────┘
                                      ▼
                           h_high_enriched
                           = 0.7·h_high + 0.3·f(episodic)
                                      │
                                      ▼
                           h_final = h_low
                           (HIGH → epistemic gate +
                            uncertainty modulation +
                            memory write trigger)
                                      │
                                      ▼
                              CLASSIFICATION
                           ratio_low, r_low → {True / Unknown / False}
```

-----

## 모듈 설명

### AttentionPooling

시퀀스 hidden state를 가중 평균으로 단일 벡터로 압축.  
`w = softmax(Linear(x))`, `h = (x * w).sum(dim=1)`

-----

### SharedPrototypeManifold

두 개의 완전 독립 prototype 공간을 관리:

```
P_low  ∈ ℝ^(n_proto × hidden)  → semantic attractor manifold
P_high ∈ ℝ^(n_proto × hidden)  → residual topology manifold (완전 독립)
```

- **context-adaptive**: `ctx_encoder`로 배치별 P_low 미세 조정 (P_high는 고정)
- **온도 스케줄**: `τ = max(τ_min, τ_base × decay^progress)` — warmup 이후 점진적 hard routing
- **EMA usage tracking**: prototype collapse 방지
- **dead prototype rescue**: 10 step 이상 미사용 prototype 자동 재초기화
- **velocity loss**: prototype 이동 속도 제한 (안정성 보장)

-----

### Gram-Schmidt Orthogonalization

HIGH path 입력을 LOW semantic manifold와 완전히 분리:

```python
h_high_base = h - proj(h, z_low)  # semantic 성분 제거
h_high_base = gram_schmidt_orthogonalize(semantic_residual, z_low_proj)
```

이후 LayerNorm으로 정규화. **H_high ⊥ z_low 를 수학적으로 보장.**

-----

### PrototypeController

LOW branch만을 대상으로 FiLM 기반 조정:

```
q_ema (slot EMA)  →  z = q_ema @ P_low_norm
FiLM: γ, β = film_net(z)
gate = conf(q) · focus(q)           ← entropy/confidence 곱 게이트
h_low_hat = h_low + gate · (γ·LN(h_low) + β - h_low)
```

- **slot_size=512**: 샘플별 독립 EMA (배치 평균 아님)
- **drift gate**: z와 h_low의 cosine similarity 기반 추가 게이팅
- r_ctx_high가 있으면 LOW에 HIGH context 역투영 차감 (분리 강제)

-----

### EpisodicMemory

prototype별 circular buffer (mem_size=50):

- **WRITE 조건**: `contradiction_score > threshold` OR `novelty_score > threshold`
- **저장 대상**: topology_key + h_high + residual_cache (EMA)
- **READ**: top-3 prototype 가중 앵커 + episode cosine 검색 + cache 혼합

```
h_high_enriched = 0.7·h_high + 0.3·fusion_gate([anchor ‖ ep_summary ‖ cache])
```

-----

### FAISSMemory

```
update: topology_key 버퍼에 누적 (max 2000)
rebuild_index: L2 normalize → IndexFlatIP
retrieve: top-k cosine neighbor → center + 0.3·spread
```

**HIGH branch에만 주입** — LOW에는 절대 직접 연결 안 함.

-----

### DualEnergyHead

```
e_pos = softplus(W_pos · h_low)
e_neg = softplus(W_neg · h_low)
ratio = e_pos / (e_pos + e_neg)   ← truth polarity  [0, 1]
r     = sqrt(e_pos² + e_neg²)     ← confidence radius
```

-----

### Epistemic Score 분해

`q_high`의 entropy와 `h_high`의 norm을 조합해 4가지 불확실성 유형 구분:

|유형               |조건              |의미                  |
|-----------------|----------------|--------------------|
|**ignorance**    |norm↓ · entropy↓|정보 자체가 없음           |
|**ambiguity**    |norm↓ · entropy↑|여러 해석 가능, 확신 낮음     |
|**contradiction**|norm↑ · entropy↓|강하게 특정 prototype에 쏠림|
|**novelty**      |norm↑ · entropy↑|고에너지 미지 입력          |

-----

## 기하학적 분류

```
r_low > r_min  AND  ratio > 0.5 + δ  →  True  (2)
r_low > r_min  AND  ratio < 0.5 - δ  →  False (0)
r_low ≤ r_min                        →  ignorance → Unknown (1)
그 외                                →  epistemic score 최댓값 기반 Unknown 세부 유형
```

-----

## Loss 구성

```
L_total = L_direction          (pos/neg margin: ratio 방향)
        + L_radius             (confidence floor: r > r_min)
        + L_spread             (True/False center 분리)
        + L_collapse           (ratio ≈ 0.5 붕괴 방지)
        + L_unknown            (Unknown → ratio ≈ 0.5, r ≤ r_min)
        + L_anticollapse_cos   (h_low ⊥ h_high cosine 억제)
        + L_anticollapse_r     (r_high / r_low 비율 유지)
        + L_high_supervision   (h_high norm label-aware 타깃)
        + L_prototype          (VQ commit + uniformity + diversity + usage entropy)
        + L_velocity           (prototype 이동 속도 제한)
        + L_routing_consist    (두 noisy view KL consistency)
        + L_separation         (epoch≥split: label-aware pull + low↔high push)
```

-----

## 학습 스케줄

|Epoch            |활성화                                        |
|-----------------|-------------------------------------------|
|0 – warmup_epochs|Mamba backbone + AttentionPooling만         |
|≥ warmup_epochs  |+ Prototype routing + Controller (FiLM EMA)|
|≥ warmup_epochs+2|+ EpisodicMemory write                     |
|≥ split_epoch    |+ separation loss (label-aware contrastive)|

-----

## 실험 설정

|항목    |값                                                 |
|------|--------------------------------------------------|
|모델    |`state-spaces/mamba-130m-hf`                      |
|데이터셋  |SNLI (train 5,000 / val 1,000)                    |
|태스크   |3값 분류 (entailment=2 / neutral=1 / contradiction=0)|
|배치 크기 |16                                                |
|시퀀스 길이|64                                                |

-----

## 주요 하이퍼파라미터

|파라미터            |기본값 |설명                           |
|----------------|----|-----------------------------|
|`n_proto`       |16  |prototype 개수                 |
|`proto_tau`     |0.1 |routing softmax 초기 온도        |
|`warmup_epochs` |3   |prototype / controller 활성화 시점|
|`split_epoch`   |5   |separation loss 활성화 시점       |
|`ctrl_ema_decay`|0.95|slot EMA decay               |
|`ctrl_slot_size`|512 |EMA 순환 슬롯 수                  |
|`faiss_k`       |15  |FAISS top-k neighbor         |
|`faiss_max_size`|2000|FAISS 버퍼 최대 크기               |
|`mem_size`      |50  |prototype별 episodic 슬롯 수     |
|`delta`         |0.1 |기하 분류 margin                 |
|`r_min`         |0.5 |confidence 최소 threshold      |

-----

## 의존성

```
torch
transformers       # MambaModel, MambaConfig
faiss-cpu          # (GPU 환경: faiss-gpu)
datasets           # HuggingFace SNLI
numpy
matplotlib
tqdm
```

-----

## 모니터링 지표

학습 중 확인해야 할 핵심 지표:

|지표                                |정상 범위           |의미                   |
|----------------------------------|----------------|---------------------|
|`ratio_low True mean - False mean`|> 0.3 (warmup 후)|LOW branch 방향성       |
|`routing_agree_rate`              |> 0.7           |prototype 안정성        |
|`r_high / r_low`                  |≠ 1.0           |HIGH-LOW 분리          |
|`ctrl_gate`                       |0.1 ~ 0.7       |FiLM 교정 강도           |
|`ctrl_scale`                      |< 0.5           |FiLM 포화 여부           |
|`q_entropy`                       |> 0             |prototype collapse 여부|

-----

## 버전 히스토리

|버전 |주요 변경                                                                                                                      |
|---|---------------------------------------------------------------------------------------------------------------------------|
|v2 |3값 출력 헤드 + classifier                                                                                                      |
|v3 |Unknown threshold 기반 선별                                                                                                    |
|v4 |Geometric Energy Space 도입                                                                                                  |
|v5 |Shared Prototype Manifold + Co-Activation GAT + Graph Memory                                                               |
|v6 |PrototypeController (FiLM + EMA gating)                                                                                    |
|v7 |SSM decomp 제거 / h_high = h − stopgrad(z_semantic)                                                                          |
|v8 |High prototype bank (P_high_bank) / disagreement loss / contrastive separation                                             |
|v10|**Gram-Schmidt 직교화** / P_low·P_high 완전 독립 / FAISS HIGH-only injection / epistemic 4분해 / velocity loss / slot EMA drift gate|

-----

## 로드맵

- [x] Geometric Energy Space (ratio / radius)
- [x] Shared Prototype Manifold + soft routing
- [x] PrototypeController (FiLM + sample-wise EMA gate)
- [x] Dual prototype space (P_low ⊥ P_high)
- [x] Gram-Schmidt orthogonalization (HIGH ⊥ LOW 수학적 보장)
- [x] FAISS long-term geometric memory (HIGH only)
- [x] EpisodicMemory (novelty/contradiction triggered write)
- [x] Epistemic 4분해 (ignorance / ambiguity / contradiction / novelty)
- [x] Velocity loss + routing consistency loss
- [x] Separation loss (label-aware contrastive)
- [ ] 동적 해빙 (LOW slow / HIGH fast adaptation)
- [ ] ablation: HIGH 제거 시 성능 하락 측정
- [ ] 생성 태스크 확장 (역질문 모듈)

-----

## 한계

**Gram-Schmidt 직교화의 완전성**  
수치적으로 완전한 직교가 보장되지 않을 수 있음. `orthogonality_loss`로 학습 중 보완.

**Prototype collapse**  
초기 학습에서 특정 prototype으로 쏠릴 수 있음.  
`visualize_usage()`로 per-epoch 확인 권장. `dead_counter > 10` 시 자동 rescue.

**h_final = h_low (alpha gate 미연결)**  
현재 HIGH enrichment가 decision에 직접 반영되지 않음.  
`alpha gate` 연결이 다음 우선 과제.

**FAISS warm-up 지연**  
초기 epoch에서 FAISS index가 비어 있어 HIGH retrieval 없이 동작.  
`topology_key is None` 체크 필요 (훈련 루프).