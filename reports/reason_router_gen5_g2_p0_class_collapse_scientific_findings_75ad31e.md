# ContraMamba Gen5 G2 — Frozen P0 Shared/Residual Intervention 최종 과학 분석

**상태:** VALIDATED_NEGATIVE_TASK_FUNCTIONAL_FINDING (frozen P0 기술적 증거, 비확증적)
**작성일:** 2026-10-14
**실행 커밋:** `75ad31e42e1baed7d99afbe9281e36276dfd1992`
**실행 이름:** `gen5-g2-p0-full-75ad31e-r1`
**실행 명령 SHA256:** `216611ce9ddb26652f08f19926b7fc8cf61c3bd077f0db2a788a7afb5facc378`
**Import ZIP SHA256:** `6df6ebcdb05daf09ac1db6826235f7ac713a52906ad113a7448a35153676a7ca`
**실험 결과 경로:** `reports/reason_router_gen5_g2_functional_science_runs/gen5-g2-p0-full-75ad31e-r1/`

## 1. 범위 및 증거의 효력

이 보고서는 **frozen Phase3A P0 dev에 대한 정적 분석**이며 Phase3B 확증 결과가 아니다. 대상은 **120개 7-cell pair로 구성된 840행**이다. 정답 라벨 분포는 `REFUTE=0`(120행), `NOT_ENTITLED=1`(600행), `SUPPORT=2`(120행)이다. 동일한 dev 모집단에서 3개 seed(`6201`, `6202`, `6203`)와 6개 arm(`ZERO`, `FULL`, `SHARED_ONLY`, `RESIDUAL_ONLY`, `SHARED_NORMMATCH`, `RESIDUAL_NORMMATCH`)의 **18조건**을 비교했다.

독립된 Tesla T4 두 개가 각각 행 `[0,416)`, `[416,840)`을 처리했다(No DDP). 총 **594 backbone forward**(286+308)를 수행했으며 실제 실행시간은 **27.9957분**이다. 모든 배치에서 ZERO/FULL reference numerical gate를 통과했다. GPU별 peak CUDA allocated memory는 약 0.753 GiB였다.

실행 결과는 worker JSON 2개, worker log 2개, summary JSON 1개 총 5개다. `cm collect`, `cm import` 모두 PASS이고 import 단계에서 5개 파일의 provenance/hash가 검증됐다. 독립 CPU static audit에서는 row/pair/cell/label 정렬, frozen arm coefficient, worker hash, shard identity 및 집계 지표 독립 재계산이 모두 PASS였다. 보고된 행별 CE 재계산 최대 절댓값 차이는 `2.19691185e-07`, 허용오차는 `2e-05`였다. **이는 artifact와 수치 재계산의 유효성을 뒷받침하지만 CUDA를 독립적으로 재실행했다는 의미는 아니다.**

## 2. 주요 관측: aggregate CE gain 보존

`CE gain = mean CE(ZERO) - mean CE(arm)`으로 정의한다. 양수이면 전체 CE 감소를 뜻한다.

| Seed | FULL CE gain | SHARED_ONLY CE gain | SHARED_ONLY / FULL 유지율 | RESIDUAL_ONLY / FULL 유지율 | SHARED_NORMMATCH / FULL 유지율 |
|---|---:|---:|---:|---:|---:|
| 6201 | +0.498486 | +0.482144 | 96.72% | 79.23% | 100.87% |
| 6202 | +0.502010 | +0.480336 | 95.68% | 92.29% | 99.93% |
| 6203 | +0.499462 | +0.479919 | 96.09% | 82.80% | 100.59% |

**성립하는 좁은 주장:** Frozen shared 방향은 P0 전체 평균 CE에 대해 FULL과 유사한 변화를 만든다. Norm-matched shared 조건도 이 관측을 유지한다.

**성립하지 않는 확대 해석:** 이는 3-class 판별 능력의 보존이나 task-functional shared core의 증명이 아니다. FULL과 SHARED_ONLY의 높은 overall accuracy는 퇴화된 예측에서 나온다.

## 3. Class-collapse 반증

| 지표 | ZERO | FULL | SHARED_ONLY | SHARED_NORMMATCH |
|---|---:|---:|---:|---:|
| Accuracy | 26.90% | 71.43% | 71.43% | 71.43% |
| Balanced accuracy (macro recall) | 61.89% | 33.33% | 33.33% | 33.33% |
| REFUTE recall | 100% | 0% | 0% | 0% |
| NOT_ENTITLED recall | 0.67% | 100% | 100% | 100% |
| SUPPORT recall | 85% | 0% | 0% | 0% |

Frozen dev의 NOT_ENTITLED 비율은 `600/840=71.43%`이다. **세 seed 모두에서 FULL 및 SHARED_NORMMATCH는 840개 행 전부를 NOT_ENTITLED로 예측한다.** SHARED_ONLY는 seed 순서대로 839, 839, 837개 행을 NOT_ENTITLED로 예측하며, REFUTE/SUPPORT 정답 240개를 하나도 맞히지 못한다. 따라서 overall accuracy의 상승은 3-class reasoning 회복의 증거가 아니다.

주의할 점은 ZERO 역시 타당한 전체 예측기가 아니라는 것이다. ZERO의 예측 클래스 분포는 `[REFUTE 348, NOT_ENTITLED 5, SUPPORT 487]`로 소수 클래스 쪽에 크게 편향되어 있다. 즉, 본 결과는 ZERO의 올바른 기능을 FULL이 망가뜨렸다는 단순한 이분법이 아니라 **서로 반대되는 클래스 편향과 aggregate CE의 불일치**를 보여준다.

Seed 6201에서 정답 클래스별 ZERO 대비 CE gain:

| 정답 클래스 | 데이터 비율 | FULL CE gain | SHARED_ONLY CE gain |
|---|---:|---:|---:|
| REFUTE | 120/840 | −1.390166 | −1.390291 |
| NOT_ENTITLED | 600/840 | +1.115509 | +1.076649 |
| SUPPORT | 120/840 | −0.697978 | −0.617942 |

Aggregate CE 개선은 다수 클래스의 개선이 두 소수 클래스의 악화를 상쇄해서 발생한다. Seed 6202, 6203에서도 동일한 부호 패턴이 확인됐다.

**현재 모집단에 대한 반증 결론:** SHARED_ONLY가 *과제의 판별 기능*을 보존한다는 해석은 이 frozen P0 결과와 양립하지 않는다. 그러나 aggregate CE gain 보존이라는 수치적 관측 자체는 유효하다.

## 4. NE logit margin 및 pair 내부 선택성

NE margin과 intervention 효과를 다음처럼 정의한다.

\[
m_{\mathrm{NE}}(x)=z_{\mathrm{NE}}(x)-\frac{z_{\mathrm{REFUTE}}(x)+z_{\mathrm{SUPPORT}}(x)}{2},\qquad
\Delta m_{\mathrm{NE}}(x)=m_{\mathrm{arm}}(x)-m_{\mathrm{ZERO}}(x).
\]

**모든 seed × 5개 non-ZERO arm에 대해, 840행의 `Δm_NE`가 전부 양수**였다. 총 15개 condition-population 검사에서 12,600개 row-condition 값이 양수라는 뜻이다. 그러나 이 측정값들은 동일한 row/pair 및 correction을 공유하므로 12,600개의 독립 표본이 아니다.

| Seed | FULL 평균 ΔNE margin | SHARED_ONLY 평균 ΔNE margin | SHARED_NORMMATCH 평균 ΔNE margin | SHARED_ONLY pair 선택성 | FULL pair 선택성 |
|---|---:|---:|---:|---:|---:|
| 6201 | +1.710533 | +1.605336 | +1.742777 | −0.181792 | −0.259323 |
| 6202 | +1.735823 | +1.598346 | +1.734164 | −0.180312 | −0.271636 |
| 6203 | +1.717595 | +1.592522 | +1.739397 | −0.172181 | −0.263373 |

Pair 선택성은 7-cell pair마다 `정답 NE 5행의 평균 ΔNE margin − 정답 REFUTE/SUPPORT 2행의 평균 ΔNE margin`을 계산한 뒤 120개 pair에 걸쳐 평균한 값이다. **음수는 같은 pair 안에서도 소수 클래스 정답 행에서 오히려 NE margin이 더 많이 상승한다는 뜻**으로, 정답에 맞춘 선택적 correction과 반대 방향이다. SHARED_NORMMATCH 역시 더 큰 음수 선택성(`−0.276361`, `−0.273183`, `−0.274659`)을 보였다. RESIDUAL_ONLY 등 다른 non-ZERO arm도 모든 행에서 NE margin이 증가했으므로 이런 방향성은 shared에만 존재하는 고유 현상이 아니다.

다만 각 행의 `Δm_NE`는 일정하지 않다(SHARED_ONLY의 행별 표준편차 약 0.424–0.432). 따라서 **입력과 무관한 단일 상수 bias만 작동한다는 강한 메커니즘은 증명되지 않았다.** 지금 확인된 것은 광범위한 NE-directed logit shift 및 음의 pair-conditional 선택성이다.

## 5. 최종 판정 및 연구 경계

1. **코드·실행·artifact·정적 재계산:** 각각 유효성 확인. 독립 CUDA 재실행은 하지 않음.
2. **기술적 관측:** SHARED_ONLY는 전체 평균 CE의 FULL 개선분 약 95.7–96.7%를 유지하며 norm-matching 후에도 유사한 방향을 보임.
3. **과학적 반증:** 이 aggregate 지표 보존은 multi-class task discrimination의 보존을 뜻하지 않는다. FULL/SHARED는 majority-class collapse를 보인다.
4. **설명 메커니즘:** NE 방향의 전행 logit 이동이 확인되지만, 정확히 일정한 logit intercept 하나만이 원인이라는 결론은 보류.
5. **통계적 범위:** 840행은 120개의 7-cell pair에 묶여 있고 3개 seed도 동일 데이터를 반복 사용한다. 이를 독립 반복으로 취급해 유의성 또는 일반화 결론을 만들지 않는다.
6. **Phase3B:** `BLOCKED_UNCHANGED`. 현재 결과를 Phase3B나 task-functional core의 확증적 근거로 승격하지 않는다.

**최종 상태: `G2_P0_CLOSED_AS_VALID_NEGATIVE_TASK_FUNCTIONAL_RESULT`.** Shared 방향이 전체 CE 개선을 재현하는 현상은 보존하되, task-functional core 주장은 현재 evidence에서 지지되지 않는다. 이후 가설은 class-balanced/within-pair 민감도, class-prior-only 대조, minority-class 보존을 사전에 명시해야 하며, 향후 GPU 실험은 별도의 승인과 commit-bound 실행 계약이 필요하다.

## 6. 원본 보존 및 권한 비승격

이 보고서는 **결과 해석 파일**이지 새 학습·평가·Phase3B 실행 권위 문서가 아니다. Import된 JSON/log 원본을 수정하지 않는다. 두 worker JSON의 실행 로그에 기록된 SHA256은 다음과 같다.

- `worker0_science.json`: `d0432108dfba731bc0eaf78854faa043842b5d9cebfa6183c867b6d39fd330de`
- `worker1_science.json`: `dd79c803dee3b5fe96822ec9d8074e24abbf6624e35bb93988571e7d55fcabc4`
