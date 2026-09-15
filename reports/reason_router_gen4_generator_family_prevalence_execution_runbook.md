# Gen4 × K Generator-Family Prevalence Execution Runbook

## 0. 문서 목적

이 문서는 **Gen4 × K Generator-Family Prevalence Transportability** stage를 새 채팅으로 옮겨도 즉시 이어갈 수 있도록 만든 실행 handoff/runbook이다.

이 문서는 현재 stage의:

- authority/freeze chain
- scientific question
- XG2/XG3/XG4 상태
- tokenizer eligibility
- runtime assets
- CUDA backend identity
- Hub migration provenance
- current code HEAD
- failed execution history
- exact next execution boundary

를 한곳에 모은다.

새 채팅에서는 이 문서를 먼저 읽고 `cm context`를 대조한다.

---

# 1. 현재 branch / code-ready state

Local active worktree used during this stage:

```text
C:\g4k-cudaeq
```

Branch:

```text
gen4-k-xg1-cross-generator-replication
```

Latest code/provenance fix HEAD before this runbook update:

```text
b539ec16cd1406457000e52648604c583ef34d5d
```

Commit:

```text
Fix migrated CUDA kernel transport provenance
```

Validation before freeze:

```text
51 passed in 3.15s
```

Files changed in `b539ec1`:

```text
scripts/reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py
scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py
tests/test_reason_router_gen4_generator_family_prevalence_kernel_compat.py
```

Important:

이 runbook 자체를 commit하면 HEAD는 `b539ec1`보다 앞으로 이동한다.

따라서 실제 Kaggle run의 `--expected-head`와 `cm run` provenance는 **runbook commit 이후의 exact current HEAD**를 사용해야 한다.

단, execution code identity가 바뀌지 않았음을 확인하기 위해 필요하면:

```powershell
git diff --quiet b539ec16cd1406457000e52648604c583ef34d5d HEAD -- `
  scripts/reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py `
  scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py `
  tests/test_reason_router_gen4_generator_family_prevalence_kernel_compat.py
```

를 사용한다.

docs-only commit이면 execution logic은 `b539ec1`과 byte-identical이어야 한다.

---

# 2. Scientific stage

Stage:

```text
Gen4 × K Generator-Family Prevalence Transportability
```

목적:

고정된 discriminator가 independent synthetic generator families에서도 inherited 300-pair response-replication design에 필요한 LARGE/SMALL prevalence를 충분히 만드는지 확인한다.

이 stage는 intervention response study가 아니다.

---

# 3. Frozen design

Design freeze:

```text
4b6f0c831ebc89f0e786e1eb526739c7c9c06413
```

Frozen discriminator:

```text
alignment_shift_abs = abs(reference_C - target_C)
```

Frozen threshold:

```text
T = 0.11228626366380845
```

Classification:

```text
LARGE iff alignment_shift_abs >= T
SMALL iff alignment_shift_abs < T
```

Target pair:

```text
C2_NAME - C0_SHAM
```

Reference pair:

```text
C5_TITLE_NAME - C1_TITLE
```

Layer roles:

```text
15 → 16 → 17
```

Minimum viable group size:

```text
30
```

Per family:

```text
300 source pairs
6 cells per pair
1800 rows
```

Scientific baseline cells only:

```text
C0
C1
C2
C5
```

Scientific full baseline budget per eligible family:

```text
300 × 4 = 1200 model forwards
```

---

# 4. Forbidden within this stage

XG2/XG4에서도 다음은 금지:

- alignment intervention
- magnitude intervention
- `R_ALIGN`
- response endpoint
- task head
- logits
- training
- backward
- threshold re-estimation
- threshold sweep
- pair replacement
- renderer repair after tokenizer/model observation
- LARGE enrichment
- family merge to satisfy minimum group size

이 stage는 baseline discriminator prevalence까지만 본다.

---

# 5. Generator families

Exactly:

```text
XG2
XG3
XG4
```

IDs:

```text
xg2_fact_001 ... xg2_fact_300
xg3_fact_001 ... xg3_fact_300
xg4_fact_001 ... xg4_fact_300
```

---

# 6. Structural cohort freeze

Structural cohort freeze:

```text
341de2398e1c06fffa73ee323502f52972c2d58e
```

Builder:

```text
scripts/build_reason_router_gen4_generator_family_prevalence_cohorts.py
```

Frozen builder Git blob:

```text
4505acc99db0627733592694da290a007d281421
```

Validator:

```text
scripts/validate_reason_router_gen4_generator_family_prevalence_cohorts.py
```

Frozen validator blob:

```text
b82bd783456ecd748959bddb781cef0fa85620e6
```

Structural validation result:

```text
34 passed
```

No tokenizer/model/CUDA/intervention executed during structural construction.

---

# 7. Frozen structural hashes

## XG2

```text
source SHA256
0d24fe924a9e30ba7341b515e6cd2bf1164eff441462451317ca5ec3beb33c5e

rows SHA256
c96ce74bb89dbd374386c27f3a7daa10b5c6630c22711074e0b9a44d59ab5cfa

manifest SHA256
29afcee82a4f0251870b6a6ef8c8b626bc65acfc1a21b98b4c019a7b20de5704

checksum SHA256
30c1ba26f8cf5c3c80e402ab9460ed162feb9681049d49315029bfbc9772e582
```

## XG3

```text
source SHA256
ce6014efe1916fa872f97ae9893f5b373c6b8d2b4c06bf0f9f8f85d2d247d18f

rows SHA256
70153d2fecfa6dcd6bb423e35ed6a7ef0da1e75153a6c64b64deecbfe06c4747

manifest SHA256
beafc6f78304ee2c14285b9df40ddc8ab1eaeac7394fdf35176d97829b2c54ce

checksum SHA256
b94501455f7c97b9612ce0ddab474e56d8aaae4450016880cb30da41128aeedd
```

## XG4

```text
source SHA256
23f81cb93a5deb7a18c98a6a2f4718b34f4bfd1713bd1fb294e037a714f71fb0

rows SHA256
9146404eefdbcf26e25b81c65496cbcc459d2eb85c90b5da44db35307de9a347

manifest SHA256
bc7091577242a7610f5c7c87d8c9b1b8a4bb17ccb047eed48c48bddb9ebadc2a

checksum SHA256
00a268007b5a1cf637daee319c54fa8f1610b3d780929d7b12c0f998b07c7b71
```

Cross-family manifest:

```text
c521ee38fa62213ab904790201e651e44071c80e92ea5241aa1681f6c94405c4
```

Root checksum:

```text
c3f877b40dde63c5aebdcb1097237c88382ef07ea08b850405deedf14f749af6
```

---

# 8. Structural topology result

Before tokenizer/model/CUDA:

## XG2

```text
1200 anchor-relevant rows
0 contiguous identity failures
```

Renderer topology:

```text
title/name contiguous
title-before-name
```

## XG3

```text
1200 anchor-relevant rows
1200 failures
```

all:

```text
C0/C1/C2/C5
```

first failure:

```text
xg3_fact_001
```

Renderer topology:

```text
name first
title later/separated
```

## XG4

```text
1200 anchor-relevant rows
0 failures
```

Renderer topology:

```text
title/name contiguous
title-before-name
```

Interpretation:

XG3는 frozen contiguous:

```text
A_IDENTITY = title → name
```

topology와 구조적으로 incompatible.

따라서 XG3는 repair/replacement하지 않는다.

---

# 9. Tokenizer / anchor eligibility freeze

Eligibility implementation freeze:

```text
8df2a5e2cb72f8f4eed33c4996dfba594a1f9687
```

Eligibility artifact freeze:

```text
0af44566eaadc324a6aaf5cec19f3972c8e371ef
```

Gate path:

```text
scripts/reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility.py
```

Frozen blob:

```text
29a97f343f372718ca56436c505893136cb505b6
```

Artifact directory:

```text
reports/reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility_8df2a5e_r1/
```

---

# 10. Tokenizer eligibility results

## XG2

```text
PASS_300_OF_300
tokenizer = true
topology failures = 0
complete pairs = 300
```

Anchor artifact SHA:

```text
c1177819713bb850cb6cfa93fb76ff06b9feddbec9cac622fc015f969844135a
```

Summary SHA:

```text
ddc6158339345a9f37ba4a0edbff0b8040e51d39bdffe92c3a409a286ec08843
```

## XG3

```text
BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY
tokenizer = false
topology failures = 1200
complete pairs = 0
```

Anchor artifact:

```text
empty file SHA256
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Summary SHA:

```text
0a0ba4a02108a4f2ee3fbae8c44dd43886a90ddbf3a85114373647d5536b08a4
```

**No XG3 model forward is allowed.**

## XG4

```text
PASS_300_OF_300
tokenizer = true
topology failures = 0
complete pairs = 300
```

Anchor SHA:

```text
5d1d2c1921c464d82229e13189ff92dbec31c06d86f1948bd4fdba6d4dad06ef
```

Summary SHA:

```text
b7d9679bc68e79892dd4e6e6b4eb69af570aba6584493ff40b16a43db1312a55
```

Cross-family primary decision is already structurally:

```text
PREVALENCE_TRANSPORTABILITY_INCOMPLETE
```

because XG3 cannot produce valid baseline geometry under the frozen contract.

그러나 XG2/XG4 family-level prevalence는 여전히 authorized/useful.

---

# 11. Frozen active encoding

```text
MAX_LENGTH = 128
CLAIM_BUDGET = 63
EVIDENCE_BUDGET = 64
EOS token ID = 0
effective PAD = 0
tokenizers = 0.22.2
```

Tokenizer revision:

```text
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37
```

Tokenizer files:

```text
tokenizer.json
b074ad869d4f45d1265ca5c9814f78604f3d7e187acc063b15dd232b27585fcf

tokenizer_config.json
9d7016c33747c6309346e59bd7bf63bfc33c9d9366ecb7e514b3b84dc6b46acb

special_tokens_map.json
57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8
```

Model `config.json` SHA:

```text
784825b6b6cdde47a1602278db0e66d6764169af4a8e662404701bc636a2686a
```

---

# 12. Representative checkpoint

Checkpoint:

```text
seed180
G3-GROUP-D-HALF
```

SHA256:

```text
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f
```

Observed Kaggle Input path:

```text
/kaggle/input/datasets/terryterry9/contramamba-seed180-g3-group-d-half-checkpoint/selected_checkpoint.pt
```

Path는 session/input mount에 따라 달라질 수 있으므로 실행 시 SHA256를 authoritative identity로 검증한다.

---

# 13. Observed runtime snapshot path

Observed in prior Kaggle session:

```text
/kaggle/working/contramamba_runtime_assets/gen4-k-xg1-full-e551484-r6/40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37
```

Required files:

```text
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

이 path는 working storage에 남아 있을 수 있지만 session reset 후 항상 존재한다고 가정하지 않는다.

실행 전 hash로 확인한다.

---

# 14. Bounded backend gate for this stage

Eligible families only:

```text
XG2
XG4
```

Fixed outcome-blind pairs:

```text
XG2 → xg2_fact_001
XG4 → xg4_fact_001
```

Baseline labels only:

```text
baseline_tp
baseline_tm
baseline_rp
baseline_rm
```

Per backend:

```text
4 model forwards
```

Per family total:

```text
CPU slow 4
CUDA fast 4
total 8
```

Scientific budget:

```text
0
```

No intervention, no response.

---

# 15. Bounded equivalence implementation

Runner:

```text
scripts/reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py
```

Compatibility loader:

```text
scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py
```

Tests:

```text
tests/test_reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py
tests/test_reason_router_gen4_generator_family_prevalence_kernel_compat.py
tests/test_reason_router_gen4_xg1_fast_cuda_one_pair_equivalence.py
tests/test_reason_router_gen4_generator_family_prevalence_tokenizer_anchor_eligibility.py
```

Current relevant regression:

```text
51 passed in 3.15s
```

---

# 16. Frozen backend tolerances

Reuse XG1 frozen tolerances:

```text
state atol    = 1e-4
state rtol    = 1e-4
geometry atol = 1e-4
geometry rtol = 1e-4
PE atol       = 1e-4
```

This prevalence gate persists state/geometry difference maxima only.

It does not persist scientific geometry/discriminator values.

---

# 17. Known-good CUDA runtime

```text
Python       3.12.13
NumPy        2.0.2
PyTorch      2.10.0+cu128
Transformers 5.0.0
CUDA runtime 12.8
GPU          Tesla T4
Capability   7.5
kernels      0.10.2
```

---

# 18. Frozen kernel scientific identity

## Mamba

Historical scientific revision:

```text
c8ffc584c147878a6eb978ae0e8db4d116c93a8c
```

Build:

```text
torch210-cxx11-cu128-x86_64-linux
```

Frozen `.so` SHA:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

## causal-conv1d

Historical scientific revision:

```text
f2651e776f66069cdcf842840db637583def1223
```

Frozen `.so` SHA:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

These scientific revisions belonged to historical legacy model-repo semantics.

Do not send them directly as current kernel-repo revisions.

---

# 19. Hub migration transport provenance

Legacy model endpoints are no longer accessible.

Exact frozen bytes were recovered in migrated kernel repo history.

## Mamba transport whitelist

```text
170306cb84f6fac356ed839fd6e2dc53ab68080e
a8ca9c4af8613ebcd16eb22873e4896ee488c840
a80a7604874b108585feb87096a0c86df2a1e5e3
90a845d5a0d552dc6b7f1653adf68bcf271f5437
```

Exact LFS SHA:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

## causal-conv transport whitelist

```text
2999c83c99b9ac5fa87b861af3ec6bac28b1c300
02ab414d848bbee389d801f87b24fa536de60273
3552fa17c03203cb43a3a76efb4de5a6e31554b5
```

Exact LFS SHA:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

Current fix commit:

```text
b539ec16cd1406457000e52648604c583ef34d5d
```

---

# 20. Current compatibility resolution semantics

Order:

1. exact historical legacy-model local cache if present and SHA matches
2. whitelisted migrated kernel local cache if SHA matches
3. whitelisted migrated immutable kernel repo commit download
4. build variant validation
5. exact single `.so` SHA validation
6. import
7. callable surface validation
8. inject into Transformers

Never:

- mutable `main`
- scientific revision reinterpretation
- binary substitution
- silent latest kernel

---

# 21. Report provenance fields after `b539ec1`

Scientific fields remain:

```text
mamba_revision
mamba_binary_sha256
causal_conv_revision
causal_conv_binary_sha256
build_variant
```

New transport fields:

```text
mamba_transport_revision
mamba_transport_repo_type
mamba_transport_source
causal_conv_transport_revision
causal_conv_transport_repo_type
causal_conv_transport_source
kernel_transport_identity_status
```

Expected status:

```text
EXACT_FROZEN_BINARY_SHA256_MATCH
```

---

# 22. Prior failed execution history

이 실패들은 scientific evidence가 아니다.

## Failure A — stale bootstrap HEAD

`cm kaggle`이 예상 branch worktree가 아니라 default/main worktree를 잡아:

```text
75b56b5...
```

를 bootstrap한 적이 있다.

해결:

```powershell
$env:CONTRAMAMBA_REPO_ROOT = "C:\g4k-cudaeq"
$cm = "$HOME\.contramamba\cm.ps1"
& $cm context
```

로 local root를 명시.

---

## Failure B — checkout blocked by untracked XG1 artifacts

Kaggle 기존 repo에 XG1 r6 artifacts가 untracked로 남아 newer checkout을 막았다.

artifact를 즉시 삭제하지 않고 detached worktree로 diagnostic했다.

이후 해당 artifacts가 Git에 보존된 상태임을 확인한 뒤 fresh bootstrap을 안전하게 사용할 수 있게 됐다.

---

## Failure C — shell pasted into Python cell

일반 Kaggle Python cell에 shell command를 붙여 `SyntaxError`.

manual diagnostic은 `%%bash`.

정식 run은 `cm run` generated pinned cell.

---

## Failure D — `kernels` package absent

Runtime provisioning failure.

Model forward/scientific execution 아님.

---

## Failure E — Hub auth/API mismatch

Runtime transport failure.

Implicit credentials에 의존하지 않는 방향으로 교정.

---

## Failure F — legacy scientific revision을 current kernel repo revision으로 잘못 해석

`repo_type="kernel"`에:

```text
c8ffc...
f2651...
```

를 직접 넣어 404.

원인:

historical scientific revisions는 legacy model-repo revisions였음.

해결:

`b539ec1`에서 scientific revision과 transport revision 분리.

---

# 23. Failed attempts are not reusable runs

이전 retry identities는 failure provenance로 보존.

새 command/HEAD에서는 새 run name 사용.

Failed run을 success artifact로 overwrite하지 않는다.

---

# 24. 현재 즉시 다음 단계

**현재 `b539ec1` code fix 이후 bounded XG2/XG4 CUDA equivalence는 아직 성공 재실행되지 않았다.**

따라서 다음 authorized execution은:

```text
XG2 xg2_fact_001 CPU-slow vs CUDA-fast baseline-only gate
+
XG4 xg4_fact_001 CPU-slow vs CUDA-fast baseline-only gate
```

이다.

아직 하지 말 것:

```text
1200-forward XG2 full baseline
1200-forward XG4 full baseline
```

bounded equivalence PASS + valid collect/import/freeze 전에는 full scientific baseline을 시작하지 않는다.

---

# 25. New-chat exact first action

새 채팅 시작 시:

```powershell
Set-Location C:\g4k-cudaeq

$env:CONTRAMAMBA_REPO_ROOT = "C:\g4k-cudaeq"
$cm = "$HOME\.contramamba\cm.ps1"

& $cm context
```

출력을 붙인다.

ChatGPT는 다음을 확인한다.

1. branch가 `gen4-k-xg1-cross-generator-replication`인지
2. HEAD가 이 runbook 이후 expected current commit인지
3. worktree clean인지
4. `b539ec1` compatibility code가 ancestor인지
5. newer execution artifact가 이미 생겼는지
6. bounded equivalence가 이미 완료됐는지

bounded gate가 아직이면 곧바로 그 실행 command를 만든다.

---

# 26. Recommended next run naming

이 runbook 작성 시점에 다음 성공 retry는 아직 실행되지 않았다.

새 HEAD에 맞는 descriptive name 예:

```text
g4k-prev-xg2-xg4-baseline-cudaeq-<shortsha>-r3
```

`r3`는 앞선 infrastructure retries와 구분하기 위한 suffix다.

이미 registry에 같은 이름이 있으면 재사용하지 말고 다음 monotonic suffix를 사용한다.

---

# 27. Run command delivery 방식

ChatGPT가 긴 shell command를 `.txt`로 생성하는 방식을 권장한다.

사용자:

1. txt 다운로드
2. SHA256 확인
3. exact bytes를 clipboard
4. `cm run save`
5. `cm run`
6. generated pinned cell 실행

PowerShell template:

```powershell
$env:CONTRAMAMBA_REPO_ROOT = "C:\g4k-cudaeq"
$cm = "$HOME\.contramamba\cm.ps1"

$cmdFile = Join-Path $HOME "Downloads\<approved-command>.txt"
$expectedSha = "<sha256>"

$sha = (Get-FileHash -LiteralPath $cmdFile -Algorithm SHA256).Hash.ToLower()

if ($sha -ne $expectedSha) {
    throw "BLOCKED: command SHA mismatch"
}

[IO.File]::ReadAllText(
    $cmdFile,
    [Text.Encoding]::UTF8
) | Set-Clipboard

& $cm run save <run-name>
& $cm run <run-name>
```

---

# 28. Kaggle state for next bounded run

Before pinned run:

- safe/fresh bootstrap as appropriate
- exact new HEAD
- clean `/kaggle/working/ContraMamba`
- checkpoint available
- model/tokenizer snapshot hashes verified
- Internet ON if migrated kernels not cached
- T4 GPU ON only immediately before CUDA run

Execution command should provision/check:

```text
kernels==0.10.2
```

and then let the repo compatibility loader resolve exact frozen bytes.

---

# 29. Expected bounded gate contract

Per family report must show:

```text
result = PASS_GENERATOR_FAMILY_PREVALENCE_BASELINE_FAST_CUDA_ONE_PAIR_EQUIVALENCE

cpu_model_forward_count = 4
gpu_model_forward_count = 4
total_model_forward_count = 8
scientific_budget_forward_count = 0

baseline_only = true
alignment_intervention_executed = false
magnitude_intervention_executed = false
response_endpoints_computed = false

training_executed = false
backward_executed = false
task_heads_executed = false
logits_read = false

scientific_conclusion = null
```

그리고 transport status:

```text
kernel_transport_identity_status
= EXACT_FROZEN_BINARY_SHA256_MATCH
```

---

# 30. Successful bounded gate 이후

성공만으로 full scientific execution을 시작하지 않는다.

먼저:

```powershell
& $cm collect <run-name>
```

Kaggle collector 실행 → ZIP download.

그 다음:

```powershell
& $cm import <handoff.zip>
```

`IMPORT PASS`.

그 뒤 artifact hashes/provenance 검토.

필요하면 result artifacts를 Git에 freeze.

그 후에만 XG2/XG4 full baseline scientific execution authorization으로 이동.

---

# 31. Scientific execution after backend gate

Eligible families:

```text
XG2
XG4
```

Per family:

```text
300 pairs × 4 baseline cells = 1200 scientific forwards
```

No XG3 model forward ever.

Output endpoints:

```text
n_LARGE
n_SMALL
p_LARGE
VIABLE
```

plus frozen descriptive summaries.

---

# 32. Final cross-family decision

Because XG3 is blocked before valid baseline geometry:

```text
PREVALENCE_TRANSPORTABILITY_INCOMPLETE
```

is already the global primary decision class for this study.

XG2/XG4 results remain family-level evidence.

Do not reinterpret two-family success as three-family ROBUST/MIXED/SYSTEMATICALLY_LOW under the frozen rule.

---

# 33. Historical references only

Frozen historical values:

Same-family prospective:

```text
n_LARGE = 81
n_SMALL = 219
```

XG1:

```text
n_LARGE = 19
n_SMALL = 281
```

context only.

Do not pool or refit threshold.

---

# 34. Interpretation boundary

항상 분리:

1. code correctness
2. execution success
3. artifact/provenance validity
4. scientific conclusion

예:

bounded CUDA equivalence PASS

는:

```text
backend acceptable for authorized workload
```

이지:

```text
prevalence hypothesis confirmed
```

가 아니다.

---

# 35. Current code correctness status

As of `b539ec1`:

```text
STATIC_TRANSPORT_PROVENANCE_AUDIT = PASS
51 tests = PASS
commit/push = PASS
worktree = clean
```

아직 남은 것은 runtime bounded gate다.

---

# 36. New-chat minimal handoff summary

새 채팅에 이 부분만 보여줘도 된다.

```text
Stage:
Gen4 × K Generator-Family Prevalence Transportability

Branch:
gen4-k-xg1-cross-generator-replication

Code/provenance fix baseline:
b539ec16cd1406457000e52648604c583ef34d5d

XG2:
eligible, backend gate pending

XG3:
BLOCKED_TOKENIZER_ANCHOR_INELIGIBILITY
NO MODEL FORWARD

XG4:
eligible, backend gate pending

Next:
run bounded baseline-only CPU-slow vs CUDA-fast equivalence
for xg2_fact_001 and xg4_fact_001 through cm run provenance chain.

Do not:
run full 1200-forward baseline yet.
Do not execute XG3.
Do not execute interventions/responses.

Checkpoint:
1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f

Tokenizer rev:
40e5d2bd7452abb3ca8fadbafe9131ee0e2c2f37

CUDA:
T4 / torch 2.10.0+cu128 / CUDA 12.8 / kernels 0.10.2

Kernel scientific revisions:
Mamba c8ffc584...
Conv  f2651e77...

Important:
those are legacy scientific revisions, NOT current kernel-repo transport revisions.

Transport commits are whitelisted in:
scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py

Exact frozen .so SHA must match before import/model forward.
```
