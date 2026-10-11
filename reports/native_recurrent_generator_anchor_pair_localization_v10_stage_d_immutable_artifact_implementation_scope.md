# Gen5 v10 — Stage D1 Immutable Bounded Artifact Implementation Scope

**Status:** `STAGE_D1_CPU_STATIC_ARTIFACT_CANDIDATE / NO_GPU_GATE / NO_SCIENTIFIC_EXECUTION`

**Base (frozen Stage C):** `869b92b1f3b17f97f4d844a20a7c1388f24cb10f` on `gen5-causal-role-state-ownership`, working tree `C:\g5-state`. The user reported clean, pushed status; GitHub commit and its five-file scope were separately inspected.

## Purpose and phase boundary

The Stage C callable bridge and V3 verifier are frozen CPU/static candidates. They do not constitute a CUDA worker/experiment, and V3's `RUNTIME_CLAIM_UNAUTHENTICATED` mode examines self-asserted identities without independently authenticating their origins. The next necessary implementation is a **write-once external artifact directory and fail-closed offline read-back**. This D1 patch does not create a GPU runner, enable Kaggle, authenticate model/data inputs or execute scientific evaluation. It does not change numerical tolerance or E0–E3 verdict semantics.

## Whitelist: three new files only

1. `scripts/gen5_v10_stage_d_immutable_artifact_candidate.py`: CPU-only package writer/reader and static CLI.
2. `tests/test_gen5_v10_stage_d_immutable_artifact_candidate.py`: CPU fixture, integrity and failure-path tests.
3. `reports/native_recurrent_generator_anchor_pair_localization_v10_stage_d_immutable_artifact_implementation_scope.md`: this scope.

No changes to original FFT audit/tests, Hybrid/GEMM, Stage A provenance, Stage B worker/V2, Stage C bridge/V3, original execution gate, data, model, checkpoint or science reports. Do not create any artifacts inside the repo during testing. Do not stage, commit or push until separately instructed after local verification. No Codex.

Frozen references at the base HEAD include:

- Original FFT audit blob `f2e6feeee8cef2d70b1ba39d05d2b0eb5f291635`.
- Hybrid reducer blob `56cbcf3b2b4c47eeb807a14778a93ec8d60ffbf0`.
- Stage A canonical tensor manifest blob `703e0f6faf15ee499ec6a97f1aa8edc8d2a93333`.
- Stage B bounded worker static helper blob `cbfc6004c6ab177eea8e7def1cc001049ec4bc4e`.
- Stage C runtime bridge blob `7c134614d0e9b4f35e0af70e058a7d184c6332ca`.
- Stage C V3 offline verifier blob `c506c45eaea394c6ef7aa047aed40e6c0d5fd4f0`.

## Accepted artifact modes

`GEN5_V10_BOUNDED_ARTIFACT_DIRECTORY_D1` has **two distinct modes**:

- `V3_UNAUTHENTICATED`: accepts an already-complete `GEN5_V10_BOUNDED_RUNTIME_EXPORT_V3` bundle, runs the unmodified V3 validation and stores exact canonical JSON for both original bundle and recomputed V3 verdict. The writer does **not** create, complete, seal, or repair any input bundle; it never infers provenance from JSON claims. The original legacy full-record failure remains visible.
- `FAILURE_ONLY`: stores `GEN5_V10_BOUNDED_FAILURE_ENVELOPE_D1` for an explicit failing phase/reason. It must never synthesize missing K4 results, a completed V3 bundle, an E1 success or any E0/E2/E3 scientific PASS. A failure before selector resolution is supported only at `PRESELECT`; subsequent phases require the precise frozen selected locus.

Artifacts use exact canonical JSON UTF-8 and physical SHA256 sidecar files. Each artifact includes `artifact_manifest.d1.json` and its SHA256 sidecar, plus mode-specific payloads and sidecars. The output is placed in a **new, external, non-overwriting directory**: one sibling staging directory, exclusive reservation, fsynced files, full read-back validation before final directory rename. Incomplete staging directories are never published as complete artifacts. Collision, symlinked target/parent, repo-internal output, missing/extra files, noncanonical bytes, duplicate keys, digest drift, forged report, wrong selector, and invalid failure status block. Individual JSON files may not exceed 64 MiB; require at least 64 MiB free disk at creation. No automatic deletion of pre-existing user files.

SHA256 and recomputed offline V3 verdict establish **local artifact integrity, not independent source authenticity**. A malicious actor able to replace every file and its sidecars can fabricate another self-consistent directory. For any future authenticated promotion, require an external trusted bundle digest, exact code HEAD, tensor lineage and separately authorized execution evidence. An artifact writer is not a substitute for those proofs.

## Fixed bounded science locus and unresolved blockers

The only planned real-input group remains worker 1, 13th one-based batch, dev rows `[800,832)`, first source `(6201,6201)` and first targets `(6201,6202)`, `(6201,6203)`, K4/B32/S128/W24576. Stage D1 does not execute it. Full-width three float32 logical tensors total approximately 2.25 GiB; three sequential capture phases would additionally transfer substantial host-device data. GPU scratch budget, allocation profiles, peak memory headroom, actual original pre-group guard attribution and physical T4 provenance remain **unmeasured**. Do not infer runtime safety from D1's CPU package tests. The 840-row run, B8 tail and mixed-route CUDA claim remain out of scope.

A later independently reviewed implementation may add a separately versioned bounded CUDA worker and gated artifact producer; do not retrofit one into this static CLI. An independently approved GPU preflight/experiment authority and a new full execution code identity will be mandatory. Code freeze alone is not that authority.

## CPU validation contract

From `C:\g5-state` with GPU visibility disabled, run only:

- `python -B -m unittest discover -s tests -p test_gen5_v10_stage_d_immutable_artifact_candidate.py -v`
- `python -B -m scripts.gen5_v10_stage_d_immutable_artifact_candidate --static-contract-check`
- `git -C C:\g5-state diff --cached --check` (if this scope is staged), and `git -C C:\g5-state status --short`.

Static tests check the two artifact kinds, unmodified original V3 verdict, overwrite and worktree-write protection, symlinks, duplicate/noncanonical JSON, sidecar/manifest tampering, altered report, wrong selector, failure-only nonpromotion and self-asserted runtime metadata. Any test skipped or missing is a blocker for D1 freeze. The tests use fresh external `tempfile` directories only, with no CUDA/model load.

**Stop** if a non-whitelisted file changes, source HEAD/blob drifts, output would overwrite evidence, the V3 verdict gains a false PASS, an incomplete or invalid package becomes readable as complete, or tests unexpectedly use GPU/model code. `CODEX=NO; GPU=OFF; TRAINING=NO; FULL_840_RUN=BLOCKED; SCIENTIFIC_EXECUTION=NOT_AUTHORIZED; COMMIT_PUSH=NO`.
