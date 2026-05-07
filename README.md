# ContraMamba
# 🚀 ContraMamba
### Contradiction-Aware State Space Model with Corrective Dynamics

ContraMamba는 Mamba 기반의 contradiction-aware SSM architecture로,

기존 selective scan이 수행하던:
- retain (기억)
- forget (망각)

을 넘어,

- reinforce
- abstain
- corrective suppression

까지 가능한 ternary corrective dynamics를 제안한다.

핵심 목표는:

> "Selective Memory"  
→ "Corrective Memory Dynamics"

로의 확장이다.

---

# ✨ Core Idea

기존 Mamba gate:

[
g_t \in [0,1]
]

는:
- 얼마나 기억할지
만 결정 가능하다.

ContraMamba는 이를:

[
g_t \in {-1,0,1}
]

로 확장한다.

| Gate | 의미 | 역할 |
|---|---|---|
| +1 | Reinforce | 기존 정보 강화 |
| 0 | Abstain | 불확실하면 업데이트 보류 |
| -1 | Corrective Suppression | 충돌 성분 억제 |

즉:
- 단순 selective retention이 아니라,
- contradiction-aware state transition을 수행한다.

---

# 🧠 Architecture

## 1. Expanded Orthogonal Projection

입력을 고차원 overcomplete latent space로 projection하고,
orthogonal regularization을 적용한다.

### 목적
- latent interference 감소
- decorrelated semantic subspaces 형성
- contradiction-related component localization 강화

직관적으로는:
- entangled representation을 줄이고,
- 충돌 성분(conflict component)을 더 분리된 방향으로 표현하려는 시도다.

---

## 2. Ternary Targeted Gating

기존 sigmoid selective gate를:

[
g_t \in {-1,0,1}
]

기반 ternary polarity gate로 확장한다.

---

### ✅ (g_t = +1) — Reinforcement

기존 상태와 일치하는 정보 강화.

- evidence accumulation
- memory reinforcement

---

### ⚪ (g_t = 0) — Abstention

불확실하거나 관련 없는 정보는 skip.

- no-op transition
- uncertainty-aware update skipping
- sparse transition dynamics

즉:
> "모르면 업데이트하지 않는다"

를 state dynamics 수준에서 구현.

---

### ❌ (g_t = -1) — Corrective Suppression

충돌 가능성이 높은 입력에 대해 reverse update 수행.

- contradiction-aware transition
- corrective residual dynamics
- latent conflict suppression

기존 Mamba가:
- selective retention

에 집중했다면,

ContraMamba는:
- corrective memory dynamics

를 수행한다.

---

# ⚡ Destructive Corrective Update

기존 selective scan:

[
h_t = A h_{t-1} + B_t x_t
]

ContraMamba:

[
h_t = A h_{t-1} + g_t(B_t x_t)
]

특히:

[
g_t = -1
]

일 때,
입력 업데이트 방향을 반전시켜 contradiction-related latent component를 억제한다.

이는:
- sign inversion 자체라기보다,
- vector subtraction 기반 corrective update

로 해석된다.

---

## Geometric Interpretation

ContraMamba의 corrective suppression은:

> "Vector Subtraction as Error Correction"

으로 해석 가능하다.

즉:
- hidden state 내부의 conflict-related activation을
- 반대 방향 residual로 상쇄(cancellation)하는 방식.

이는:
- destructive interference
- latent energy suppression
- corrective state stabilization

관점으로 연결 가능하다.

---

# 🛡️ Abstention-Aware Safety

[
g_t = 0
]

gate는:
- sparse computation
뿐 아니라,

- uncertainty preservation
- overconfident generation suppression

역할도 수행한다.

즉:
> "모르는 것을 모른다고 할 수 있는 모델"

을 목표로 한다.

이는:
- hallucination mitigation
- trustworthy generation
- AI Safety

와 연결된다.

---

# ⚙️ Efficient Linear Scaling

ContraMamba는 기존 SSM selective scan 구조를 유지하므로:

[
O(Ld)
]

linear scaling characteristic을 유지한다.

또한 ternary gating은:
- lightweight element-wise operation
- sign-aware routing
- branch-free implementation

형태로 구현 가능하다.

따라서:
- Triton
- CUDA
- fused scan kernel

기반 hardware-aware optimization과도 호환 가능하다.

---

# 🔬 Training Strategy

## Stage 1 — Soft Continuous Gating

초기 학습:

[
g_t = \tanh(W_g x_t)
]

사용.

목적:
- stable optimization
- smooth gradient flow

---

## Stage 2 — Hard Ternary Discretization

후반부:

[
g_t \rightarrow {-1,0,1}
]

using STE (Straight-Through Estimator).

목표:
- discrete corrective routing
- stable training dynamics

---

# 📊 Evaluation Plan

## 1. Synthetic Contradiction Task

예시:

- "Paris is capital of France."
- 이후:
  "Paris is capital of Germany."

입력 시:
- latent trajectory
- corrective suppression
- contamination recovery

분석.

---

## 2. TruthfulQA

TruthfulQA

측정:
- hallucination reduction
- abstention behavior
- uncertainty-aware generation

---

## 3. Conflict-RAG Evaluation

충돌 retrieval 상황에서:
- contradiction robustness
- false acceptance rate
- conflict-sensitive suppression

평가.

---

# 🎯 Core Research Claim

ContraMamba는:

> contradiction-aware ternary state transitions

를 통해,

- corrective suppression
- abstention-aware updates
- conflict-sensitive memory dynamics

를 linear-time state space model 내부에 도입한다.

---

# 📌 Current Status

- [x] Research concept formulation
- [ ] Mamba baseline reproduction
- [ ] Ternary gate implementation
- [ ] Synthetic contradiction benchmark
- [ ] TruthfulQA evaluation
- [ ] Conflict-RAG benchmark
- [ ] Hardware-aware optimizatio
