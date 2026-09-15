# Gen4 Native-State Fast CUDA Runbook

## 0. 이 문서의 역할

이 문서는 ContraMamba Gen4 native-Mamba state workload에서 재사용 가능한 **fast-CUDA backend identity, equivalence contract, runtime contract, kernel provenance, migration recovery 규칙**을 기록한다.

중요:

- 이 문서는 scientific hypothesis를 새로 승인하지 않는다.
- closed historical experiments를 다시 열지 않는다.
- CPU slow와 CUDA fast의 의미적 동등성을 임의로 가정하지 않는다.
- 새 workload는 task-appropriate bounded equivalence gate를 통과한 뒤에만 CUDA backend를 scientific execution에 사용한다.
- Hub migration이나 package ecosystem 변화가 있어도 **scientific kernel identity를 mutable locator와 혼동하지 않는다.**

---

# 1. Historical validated backend equivalence

검증된 bridge:

- CPU reference HEAD:
  `8496ece911e0d461f0abbdf1a0fa619f8a2f22ab`
- fast-CUDA one-pair gate HEAD:
  `d6e1521c3f2c92f16aa88bbaf3f6b6c332a85ae6`
- fast-CUDA full backend-equivalence HEAD:
  `60f7485d37a7afb848b11b31a30242f5b89db534`
- bridge closure HEAD:
  `ad375c88385eef507e151f95717e951919a6a3fe`

300-pair CPU↔CUDA comparator:

- max item geometry abs diff:
  `3.166055559500336e-06`
- max item PE abs diff:
  `1.6212262369252883e-06`
- baseline mean abs diff:
  `9.497736354335817e-09`
- max hypothesis-mean abs diff:
  `1.4363876632191572e-08`
- Holm decisions:
  identical
- final scientific outcome:
  identical

Frozen tolerances:

```text
geometry atol = 1e-4
geometry rtol = 1e-4
PE atol       = 1e-4
```

이 equivalence는 해당 Gen4 × K bridge가 실제로 읽고 개입한 quantities에 대한 것이다.

---

# 2. Known-good GPU runtime

Validated runtime:

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

silent substitution 금지.

특히:

- 다른 Torch CUDA build
- 다른 GPU architecture
- 다른 `kernels`
- mutable kernel `main`

을 자동 대체하지 않는다.

historical frozen CPU environment는 별도 identity다.

---

# 3. 가장 중요한 개념: scientific identity와 transport identity를 분리

2026-09 Hub migration 이후 반드시 두 identity를 분리한다.

## 3.1 Scientific kernel identity

Scientific identity는 historical validation에서 동결된 다음 조합이다.

```text
historical repository identity
+ historical immutable revision
+ build variant
+ exact loaded scientific .so SHA256
+ callable surface
```

이 값은 Hub repository migration 때문에 바꾸지 않는다.

---

## 3.2 Transport identity

Transport identity는 **현재 시점에 동일한 frozen bytes를 어디서 가져오는가**를 나타낸다.

예:

```text
repo type = kernel
transport commit = <current immutable commit>
```

Transport commit은 scientific revision을 대체하지 않는다.

동일 frozen `.so` SHA256가 검증될 때만 acceptable transport locator다.

---

# 4. Frozen Mamba scientific identity

Repository label:

```text
kernels-community/mamba-ssm
```

Historical scientific revision:

```text
c8ffc584c147878a6eb978ae0e8db4d116c93a8c
```

Build variant:

```text
torch210-cxx11-cu128-x86_64-linux
```

Frozen scientific `.so` SHA256:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

Observed LFS object size:

```text
610662256 bytes
```

Required functions:

```text
selective_scan_fn
selective_state_update
mamba_inner_fn
```

---

# 5. Frozen causal-conv1d scientific identity

Repository label:

```text
kernels-community/causal-conv1d
```

Historical scientific revision:

```text
f2651e776f66069cdcf842840db637583def1223
```

Build variant:

```text
torch210-cxx11-cu128-x86_64-linux
```

Frozen scientific `.so` SHA256:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

Observed LFS object size:

```text
107169840 bytes
```

Required functions:

```text
causal_conv1d_fn
causal_conv1d_update
```

---

# 6. Historical loader semantics

`kernels==0.10.2`의 historical `get_kernel()`은 내부적으로:

```python
snapshot_download(
    repo_id,
    revision=revision,
    ...
)
```

을 사용했고 `repo_type="kernel"`을 지정하지 않았다.

따라서 historical revisions:

```text
c8ffc584...
f2651e77...
```

은 당시 **legacy model-repo revision**으로 해석된 값이다.

이 사실이 매우 중요하다.

절대 이 historical scientific revision을 현재 migrated kernel-repo commit이라고 재해석하지 않는다.

---

# 7. Hub migration forensic result

2026-09 forensic scan에서 다음이 확인됐다.

## 7.1 Legacy model endpoints

다음 legacy endpoints는 현재 접근 불가:

```text
https://huggingface.co/kernels-community/mamba-ssm
https://huggingface.co/kernels-community/causal-conv1d
```

따라서 historical model-repo commit 자체를 network에서 다시 가져오는 경로는 현재 사용할 수 없다.

---

## 7.2 Migrated kernel repos

현재 reachable:

```text
https://huggingface.co/kernels/kernels-community/mamba-ssm
https://huggingface.co/kernels/kernels-community/causal-conv1d
```

historical scientific revision SHA는 이 migrated repo history에 존재하지 않는다.

그러나 **exact frozen LFS payload는 reachable history에 존재한다.**

---

# 8. Mamba migrated transport commits

Frozen `.so` SHA:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

exact payload를 포함하는 확인된 immutable transport commits:

```text
170306cb84f6fac356ed839fd6e2dc53ab68080e
a8ca9c4af8613ebcd16eb22873e4896ee488c840
a80a7604874b108585feb87096a0c86df2a1e5e3
90a845d5a0d552dc6b7f1653adf68bcf271f5437
```

Observed path:

```text
build/torch210-cxx11-cu128-x86_64-linux/_mamba_ssm_cuda_785446d.abi3.so
```

Current compatibility loader tries whitelisted immutable transport commits only.

---

# 9. causal-conv1d migrated transport commits

Frozen `.so` SHA:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

exact payload를 포함하는 확인된 immutable transport commits:

```text
2999c83c99b9ac5fa87b861af3ec6bac28b1c300
02ab414d848bbee389d801f87b24fa536de60273
3552fa17c03203cb43a3a76efb4de5a6e31554b5
```

Observed path:

```text
build/torch210-cxx11-cu128-x86_64-linux/_causal_conv1d_cuda_63d1a23.abi3.so
```

---

# 10. Current accepted resolution order

Current compatibility path must resolve kernel bytes in this order.

## 10.1 Historical local cache first

If exact legacy model-repo snapshot is already locally cached under historical scientific revision:

```text
models--<owner>--<repo>/snapshots/<scientific_revision>
```

and exact build + exact `.so` SHA matches, use it.

이 경로가 historical validation과 가장 가깝다.

---

## 10.2 Migrated local kernel cache second

If one of the whitelisted transport commits is locally cached under:

```text
kernels--<owner>--<repo>/snapshots/<transport_revision>
```

validate:

- build variant
- module surface
- exactly one relevant `.so`
- exact frozen SHA256

then use it.

---

## 10.3 Network transport last

If local cache is absent:

```text
repo_type = kernel
revision = one of frozen transport whitelist
token = false
```

로 immutable transport commit을 download한다.

Mutable `main`은 사용하지 않는다.

historical scientific revision을 kernel repo revision으로 보내지 않는다.

---

# 11. Exact byte authentication is mandatory

Transport commit만 맞는 것으로 충분하지 않다.

import 전에:

```text
build variant
module init path
single .so surface
exact .so SHA256
```

을 확인한다.

Mamba expected:

```text
dc4d76a6323b510e77cfb66b5aa7bb0086c8f5cba238002b9c20bc31ea706587
```

causal-conv expected:

```text
6b013d7b9a033bb9b0a2a714b26470e1aaba4af9bf1b3ec7442c2a53afb6b7b6
```

다르면 import/model forward 전에 BLOCK.

---

# 11.1 Wrapper-path authentication after Hub migration

`kernels==0.10.2` uses a path-derived unique module name and
`importlib.util.spec_from_file_location(...)` when importing kernel wrappers.

Migrated kernel snapshots can expose both:

```text
build/<variant>/<package>/__init__.py
build/<variant>/__init__.py
```

The preferred pre-import wrapper path is not itself a frozen scientific
identity. After import, `module.__file__` must resolve to **one of the wrapper
files that actually exists in the exact authenticated snapshot**.

Do not require `module.__file__` to equal only the single preferred wrapper
candidate selected before import. That over-constrains migrated snapshots and
previously caused a false `MAMBA_MODULE_PATH` blocker after the exact frozen
`.so` had already been authenticated.

Still fail closed if `module.__file__` resolves outside the authorized wrapper
surface of the exact snapshot.

The scientific identity remains:

```text
historical scientific revision
+ build variant
+ exact frozen .so SHA256
+ required callable surface
```

Wrapper-path acceptance does not relax any of those identities.

---

# 11.2 Transformers 5.0.0 constructor-time kernel routing

Transformers 5.0.0 resolves Mamba kernels during every `MambaMixer`
constructor, before any forward and regardless of whether the model is still
on CPU.

Therefore a successful exact-kernel preflight in a separate Python process is
not enough. The actual runner process must:

```text
1. authenticate/load exact Mamba and causal-conv modules
2. temporarily route modeling_mamba.lazy_load_kernel
   - causal-conv1d -> authenticated conv module
   - mamba-ssm    -> authenticated Mamba module
3. construct both CPU and GPU model instances
4. restore the original lazy_load_kernel callable
5. validate the resulting modeling_mamba global module/function bindings
6. run CPU slow and CUDA fast paths
```

Unknown kernel names or unexpected loader arguments are blockers. The exact
constructor router never falls through to the Transformers default Hub loader.

CPU execution remains slow because Transformers chooses its CUDA fast forward
only when the mixer weights are on CUDA. This routing changes dependency
resolution only; it does not alter weights, forward budget, geometry,
recurrent-state reconstruction, tolerances, or scientific endpoints.

---

# 12. Current compatibility implementation

Reference:

```text
scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py
```

Migration/provenance fix commit:

```text
b539ec16cd1406457000e52648604c583ef34d5d
```

이 implementation은:

- scientific revision 보존
- transport revisions 별도 보존
- local legacy cache 우선
- migrated local cache second
- migrated kernel repo network fallback
- `token=False`
- mutable main 금지
- exact binary SHA pre-import validation
- actual transport provenance return

을 수행한다.

---

# 13. Report에 반드시 남길 transport provenance

새 equivalence report는 scientific identity와 transport identity를 둘 다 남겨야 한다.

Scientific fields:

```text
mamba_revision
mamba_binary_sha256
causal_conv_revision
causal_conv_binary_sha256
build_variant
```

Transport fields:

```text
mamba_transport_revision
mamba_transport_repo_type
mamba_transport_source
causal_conv_transport_revision
causal_conv_transport_repo_type
causal_conv_transport_source
kernel_transport_identity_status
```

Expected transport identity status:

```text
EXACT_FROZEN_BINARY_SHA256_MATCH
```

Transport revision이 달라도 exact frozen binary SHA가 같고 whitelist에 포함된 경우에만 허용한다.

---

# 14. 왜 Transformers default loader를 쓰지 않는가

Transformers 5.0.0 historical default Mamba revision:

```text
v0.0.4
```

validated build:

```text
torch210-cxx11-cu128-x86_64-linux
```

에서 required build가 없었다.

따라서 default loader warning/behavior는 scientific fast backend proof가 아니다.

Validated harness는 exact kernel functions를 명시적으로 주입하고 fast scan path를 검증한다.

---

# 15. Fast-path warning 해석

Model construction 시 다음 warning이 나올 수 있다.

```text
The fast path is not available...
Falling back to the sequential implementation...
```

이 warning 자체는 이후 scientific forward가 어떤 backend를 썼는지 증명하지 않는다.

검증 기준:

1. exact kernel bytes authenticated
2. callable surface authenticated
3. functions injected
4. GPU model on `cuda:0`
5. layer-17 fast scan intercepted
6. bounded CPU↔CUDA equivalence passed

---

# 16. Frozen recurrent-state reconstruction

CUDA에서 chunk internal buffer를 tokenwise full recurrent-state sequence로 오해하지 않는다.

Validated reconstruction:

1. capture layer-17 fast scan inputs
2. `selective_scan_fn(..., return_last_state=True)` through anchor `a`
3. returned last state = `s[a]`
4. sequential `selective_state_update` for:
   - `a+1`
   - `a+2`
   - `a+3`
   - `a+4`
5. reconstruct:
   - `s[a]`
   - `s[a+1]`
   - `s[a+2]`
   - `s[a+3]`
   - `s[a+4]`

Frozen semantic timing:

```text
post recurrent update
before C readout
```

---

# 17. Equivalence gate 원칙

새 comparable workload는 full run부터 시작하지 않는다.

순서:

1. runtime/kernel provenance gate
2. bounded one-pair CPU slow vs CUDA fast gate
3. full scientific execution only if bounded gate PASS
4. full backend comparison when applicable

Equivalence failure 시 tolerance를 늘려 통과시키지 않는다.

---

# 18. Migration 관련 failure history

향후 같은 debugging을 반복하지 않기 위해 기록한다.

## 18.1 `kernels` package missing

의미:

- Kaggle runtime provisioning missing
- scientific failure 아님

조치:

```bash
python -m pip install --no-deps "kernels==0.10.2"
```

또는 approved run command가 exact version을 provision하도록 한다.

---

## 18.2 Hub 401/authentication mismatch

의미:

- runtime transport/auth infrastructure 문제
- model scientific result 아님

조치:

- implicit stale token에 의존하지 않음
- compatibility transport는 `token=False`
- exact public immutable transport locator 사용

---

## 18.3 historical scientific revision을 `repo_type="kernel"`로 직접 조회하여 404

이 방식은 잘못됐다.

이유:

- historical revision은 legacy model repo의 revision
- migrated kernel repo의 revision이 아님

해결:

- scientific revision 보존
- separate transport revision 사용
- exact frozen `.so` SHA로 identity bridge

---

## 18.4 legacy model endpoint inaccessible

현재 정상적인 migration condition이다.

그 자체로 backend scientific identity가 invalid해진 것은 아니다.

exact frozen bytes가 migrated repo에서 검증되면 transport만 바뀐다.

---

# 19. 절대 하지 말 것

- historical scientific revision을 current kernel repo revision으로 재해석
- mutable `main` 사용
- 현재 최신 kernel binary로 silent upgrade
- binary SHA 검증 생략
- tolerance relaxation
- package roulette
- failed run 덮어쓰기
- equivalence gate 없이 full scientific CUDA execution
- Hub cache 전체를 scientific identity로 기록

---

# 20. Reusable code

Historical backend:

```text
scripts/reason_router_gen4_k_fast_cuda_one_pair_equivalence.py
scripts/reason_router_gen4_k_fast_cuda_full.py
scripts/reason_router_gen4_k_fast_cuda_full_compare.py
```

Current migrated-transport compatibility:

```text
scripts/reason_router_gen4_generator_family_prevalence_kernel_compat.py
```

Current generator-family bounded equivalence:

```text
scripts/reason_router_gen4_generator_family_prevalence_fast_cuda_one_pair_equivalence.py
```

---

# 21. New-chat rule

새 채팅에서 fast CUDA 문제가 다시 나오면:

1. 이 runbook을 먼저 읽는다.
2. `cm context`로 현재 HEAD 확인.
3. scientific revision과 transport revision을 구분.
4. exact binary SHA가 이미 frozen identity임을 확인.
5. migration forensic을 처음부터 다시 하지 않는다.
6. 현재 workload의 bounded equivalence gate 상태부터 확인.
7. gate PASS 전에는 full scientific execution을 시작하지 않는다.
