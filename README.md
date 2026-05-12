# ContraMamba

> 환각 감소를 위한 명시적 불확실성 출력 기반 Mamba 분류 모델

## 동기

기존 LLM은 틀린 답도 확신을 갖고 생성하는 환각 문제가 있음.
ContraMamba는 모델이 불확실성을 명시적으로 표현하도록 강제하는
**3값 출력 헤드**를 도입해 이 문제를 완화하고자 함.

## 핵심 아이디어

### 1. 3값 출력 {-1, 0, 1}
소프트맥스 확률 대신 세 가지 상태 중 하나를 출력:
- `1` : 맞음
- `-1` : 틀림
- `0` : 모름 (역질문 트리거 가능)

### 2. 구조화된 직교 표현
푸리에 변환의 직교 기저와 SSM의 HiPPO 행렬에서 착안,
은닉 상태를 의미론적 카테고리별 직교 부분공간으로 제약.
계층 구조는 사람이 정의하지 않고 학습으로 자동 형성.

### 3. 왜 Mamba인가?
- 추론 시간 O(n) — Transformer의 O(n²) 대비 효율적
- SSM 구조가 신호 및 시스템의 선형 필터와 동일한 수학적 구조
- HiPPO 초기화가 자연스럽게 직교 기저로 시퀀스를 투영

## 이론적 배경

### Parseval 정리와 메타인지 출력
DCT 기반 직교 표현에서 Parseval 정리가 성립: 입력 시퀀스의 총 에너지 = 각 직교 subspace 에너지의 합
이를 통해 {-1, 0, 1} 출력을 에너지 관점에서 정의:
- `1` (맞음): S+ subspace에 에너지 집중
- `-1` (틀림): S- subspace에 에너지 집중
- `0` (모름): 어떤 subspace에도 에너지 없음 → 역질문 트리거

S+, S-, 나머지 공간은 서로 직교 (Kleene 삼진논리의 기하학적 구현)

기존 confidence threshold 기반 접근과 달리,  
에너지 임계값 ε 이하일 때 "모름"으로 판단하는 방식이  
수학적으로 정당화됨.

### 에너지 해석 주의사항
ContraMamba의 e_pos/e_neg는 물리적 에너지와 역의 관계.
투영 강도(projection strength) 개념으로,
SSM의 상태 에너지가 낮을수록 안정적인 것과 달리
e_pos/e_neg는 클수록 해당 공간에 강하게 속함을 의미.

## 실험 (베이스라인)

- **모델**: Mamba-130m (파인튜닝)
- **데이터셋**: BoolQ
- **태스크**: 이진 QA → 3값 분류
- **레이블**: True → 1, False → -1, Unknown → 0 (추론 시 threshold)

## 결과

| 모델 | Val Accuracy | 비고 |
|---|---|---|
| Mamba-130m 베이스라인 | 0.6217 | 질문만 입력 |
| ContraMamba v2 | 0.6780 | 질문+지문, classifier |
| ContraMamba v3 (threshold=8.5) | 0.7005 | Unknown 제외 정확도 |

### 삼진논리 출력 분포 (threshold=8.5)
- Known 정확도: 0.7005
- Unknown 비율: 30.37%

### 에너지 분리 결과 (v2)
| 샘플 | S+ 에너지 | S- 에너지 | 갭 |
|---|---|---|---|
| True 샘플 | 7.41 | 3.83 | +3.58 |
| False 샘플 | 5.19 | 6.03 | +0.84 |

### Unknown 샘플 분석 (threshold=8.5)
- Unknown 총 개수: 988 (30.2%)
- Unknown 중 실제 True: 58.9% / False: 41.1%
- **Known 정확도: 0.7209**
- 해석: 모델이 확신할 수 있는 샘플만 선별해 72% 정확도 달성
  불확실한 샘플은 Unknown으로 판단 유보 → 메타인지 작동 증거

## 로드맵

- [x] 3값 출력 헤드
- [x] S+/S- 직교 subspace 설계
- [x] DCT 초기화 + Attention Pooling
- [x] 비대칭 Gap Loss
- [x] 개별 Subspace Freeze
- [x] Threshold 기반 Unknown 판단
- [x] Threshold Loss로 학습
- [ ] Unknown 샘플 분석
- [ ] 계층적 직교 subspace
- [ ] 그래프 추론 연동
- [ ] 동적 해빙
- [ ] 생성 태스크 확장

## 한계 및 향후 연구

### 직교 제약의 딜레마
- Subspace freeze 시 Mamba backbone이 고정된 공간에 맞게 
  표현을 조정하지 못해 성능 저하
- Backbone까지 freeze하면 학습 자체가 불가능
- 완전한 직교성 보장과 학습 자유도 사이의 트레이드오프 존재

### 동적 해빙 (Dynamic Unfreezing)
- 개념: Frozen된 직교 기저를 강한 반증 증거 감지 시 자동으로 해빙
- 트리거: inference 중 에너지 갭이 강한 음수 (gap < -strong_threshold) 감지
- 업데이트: 극소량 lr (1e-6)로 미세 조정 후 재동결
- 철학: 단단한 공리(frozen)도 충분한 반증 앞에서 수정 가능한 말랑말랑한 뇌
- 관련 분야: Continual Learning, Catastrophic Forgetting 방지
- 난이도: 높음 (언제/얼마나 unfreeze할지 기준 설계 필요)

### 향후 연구 방향
- [ ] Backbone과 Subspace를 교대로 학습하는 방식 (alternating training)
- [ ] 복소수 SSM으로 확장 (S4 계열)
- [ ] 생성 태스크로 확장 및 역질문 모듈 구현
- [ ] 생성 태스크로 확장
- [ ] 삼진논리 기반 S+/S- 직교 subspace 설계
