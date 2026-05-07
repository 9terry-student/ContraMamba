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

까지 가능한 corrective memory dynamics를 제안한다.

핵심 목표는:

> Selective Memory  
→ Corrective Memory Dynamics

로의 확장이다.

---

# ✨ Core Idea

기존 Mamba gate는:
- 얼마나 기억할지
만 결정 가능했다.

ContraMamba는 이를 ternary polarity mechanism으로 확장한다.

| Gate | Meaning | 역할 |
|---|---|---|
| +1 | Reinforce | 기존 정보 강화 |
| 0 | Abstain | 불확실하면 업데이트 skip |
| -1 | Corrective Suppression | contradiction-related update 억제 |

즉:
- 단순 selective retention이 아니라,
- contradiction-aware state transition을 수행한다.

---

# 🧠 Architecture Overview

## 1. Expanded Orthogonal Projection

입력을 higher-dimensional latent space로 projection하고,
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

ContraMamba는 기존 sigmoid selective routing을 ternary polarity gating으로 확장한다.

---

### +1 — Reinforcement

consistent information strengthens memory accumulation.

- evidence accumulation
- memory reinforcement

---

### 0 — Abstention

uncertain or irrelevant updates are skipped.

이는:
- uncertainty-aware transitions
- sparse updates
- hallucination mitigation through abstention

을 가능하게 한다.

즉:
> "모르면 업데이트하지 않는다"

를 state dynamics 수준에서 구현한다.

---

### -1 — Corrective Suppression

contradiction-sensitive updates are reversed before entering the hidden state transition.

이는:
- corrective residual dynamics
- latent conflict suppression
- contradiction-aware memory correction

을 수행한다.

기존 Mamba가 selective retention에 집중했다면,

ContraMamba는:
> corrective memory dynamics

를 수행한다.

---

# ⚡ Destructive Corrective Update

Original selective scan:

text id="7h4o2m" h_t = A h_(t-1) + B_t x_t 

ContraMamba:

text id="8v1k9c" h_t = A h_(t-1) + g_t(B_t x_t) 

When the gate becomes -1,
the update direction is reversed to suppress contradiction-related latent components.

이는 단순 sign inversion이라기보다:

- vector subtraction as error correction
- destructive interference in latent space
- corrective memory stabilization

으로 해석된다.

중요하게도,
ContraMamba는 symbolic logical reasoning 자체를 주장하지 않는다.

대신:
> contradiction-aware corrective dynamics in representation space

를 제안한다.

---

# 🛡️ Abstention-Aware Safety

ContraMamba는 uncertainty-aware abstention dynamics를 도입한다.

0 gate를 통해:
- uncertain updates를 skip하고,
- ambiguous state를 유지할 수 있다.

Potential benefits:
- reduced hallucination
- safer generation
- trustworthy responses
- uncertainty-aware reasoning

즉:
> "모르는 것을 모른다고 할 수 있는 모델"

을 목표로 한다.

---

# ⚡ Efficient Linear Scaling

ContraMamba는 기존 SSM selective scan 구조를 유지하므로:
- linear-time sequence scaling
- no quadratic attention matrix

특성을 유지한다.

또한 ternary gating은:
- lightweight element-wise operation
- sign-aware routing
- hardware-friendly implementation

형태로 구현 가능하다.

Compatible with:
- Triton kernels
- CUDA fused scan
- parallel scan algorithms

---

# 🔬 Planned Evaluation

## Synthetic Contradiction Tasks

예시:
- "Paris is capital of France"
- followed by:
- "Paris is capital of Germany"

분석:
- latent trajectory recovery
- corrective suppression behavior
- state contamination dynamics

---

## TruthfulQA Benchmark

TruthfulQA

Evaluate:
- hallucination reduction
- abstention behavior
- uncertainty-aware generation

---

## Conflict-RAG Evaluation

Measure:
- contradiction robustness
- false acceptance rate
- retrieval conflict handling

---

# 🎯 Research Goal

ContraMamba는 sequence model이 단순 selective retention을 넘어:

- corrective memory dynamics
- contradiction-sensitive transitions
- abstention-aware generation

을 수행할 수 있는지 탐구한다.

동시에:
- efficient linear-time computation
- hardware-aware implementation

을 유지하는 것을 목표로 한다.

---

# 📌 Current Status

- [x] Research concept formulation
- [ ] Mamba baseline reproduction
- [ ] Ternary gate implementation
- [ ] Synthetic contradiction experiments
- [ ] TruthfulQA evaluation
- [ ] Conflict-RAG benchmark
- [ ] Hardware-aware optimization
