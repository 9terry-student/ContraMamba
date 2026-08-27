# P3-W6-F2-P4-L Freeze-Binding Correction Execution-Authority Candidate

Authority/version:

`P3W6F2P4L_CURRENT_EXACT_BYTE_PROVISIONING_FREEZE_BINDING_CORRECTION_CANDIDATE_V1`

## 1. Disposition and narrow supersession

This is a candidate authority only. Candidate creation does not provision
P4-L, invoke Kaggle, run Python for provisioning, run a builder or P4-Q, run
training or evaluation, mutate data/checkpoints/manifests, validate a
provisioned artifact, perform Git identity transitions, commit, push, or use a
GPU. It becomes executable only after independent static verification and an
explicit immutable freeze.

This candidate supersedes the frozen authority
`reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_execution_authority_spec.md`
only for its execution/freeze identity binding. Every source, destination,
byte-copy, preflight, publication, failure-retention, status-delta, and
downstream boundary rule in that authority remains in force unless this
document states the correction explicitly. No other semantic or operational
change is authorized.

The frozen authority was created/frozen at historical HEAD
`b818f523c95331058680128b5943dd94cf7aca4b`, and its pre-freeze content SHA256
was `c74768efa14b9f220f8317684aa98c83697d81da855a087493f70f01b2fd4462`.
Those values are provenance of the defect, not the future execution HEAD.

## 2. Corrected execution identity contract

The workflow must supply exactly one execution identity variable:

```text
P4L_PROVISION_AUTHORITY_FREEZE
```

It is valid only when all of the following hold, before any destination
mutation:

1. The value matches exactly `[0-9a-f]{40}` (lowercase ASCII hexadecimal).
2. `git cat-file -e "$P4L_PROVISION_AUTHORITY_FREEZE^{commit}"` succeeds,
   and `git cat-file -t "$P4L_PROVISION_AUTHORITY_FREEZE"` is `commit`.
3. `git ls-tree -r --name-only "$P4L_PROVISION_AUTHORITY_FREEZE" --
   reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_freeze_binding_correction_execution_authority_spec_candidate.md`
   returns exactly that final canonical correction-authority path.
4. `git rev-parse HEAD` equals the supplied value exactly.

The supplied value is not predicted, embedded, or hard-coded in this
candidate. The eventual freeze commit therefore supplies its own value after
the candidate has been frozen; the command reads that value and verifies that
the checked-out commit contains this exact authority file. Candidate-creation
HEAD `b818f523c95331058680128b5943dd94cf7aca4b` is historical context only.

No `git switch`, `checkout`, `reset`, `clean`, `pull`, `merge`, `rebase`, or
`cherry-pick` is authorized or needed. Kaggle bootstrap for the later,
separately authorized execution must check out/pin this exact correction
freeze commit before invoking the command.

## 3. Unchanged P4-L provisioning contract

The following values and rules are carried forward byte-for-byte in meaning
from the frozen authority and the current P4-L lineage-integrity artifact
contract:

- Sidecar source:
  `/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl`
- Provenance source:
  `/kaggle/input/datasets/terryterry9/proside/p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`
- Sidecar physical SHA256:
  `2b8cffdf71d68a8abeb3b6eb3534eeb664bd012483bcebd9716c7a6645a487f1`
- Sidecar semantic SHA256:
  `0e652c80ccae796bc2fded883ed099e0af71084a83e4a2fd4dd3524899d81b08`
- Sidecar row count: `3600`.
- Provenance physical SHA256:
  `9d248df09ae8ba471966c468a1e06278ad046908cfe53da623ecc95d8da4cdf2`
- Canonical destination:
  `reports/reason_router_p2_p3w6f2_p4l_current_lineage_integrity_sidecar_2f9e6076791358922e3ebd70e89533d9cb83b458/`
- Exact children:
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar.jsonl` and
  `p3w6f2_p4l_current_lineage_effective_integrity_sidecar_provenance.json`.

The destination and all `.p4l-staging-*` siblings must be absent before the
attempt. Both sources must be distinct regular non-symlink files with exact
basenames, outside the destination, and unchanged. The operation is Linux
only, CPU-only (`CUDA_VISIBLE_DEVICES` exactly empty), and fail-closed.
Preflight must include repository-root/path-containment checks, all source
physical and semantic identity checks, row count, clean tracked worktree and
index, and exact source/destination absence checks. Copying is raw byte copying
only. It must use a unique same-parent staging directory, retain staging on
failure, perform no automatic cleanup, and publish only by dynamic libc
`renameat2` with `RENAME_NOREPLACE`. Any unsupported or failed atomic operation
blocks publication.

After publication, the command must verify exact destination entry set,
regular non-symlink children, physical and semantic hashes, row count, byte
equality to both sources, source unchanged status, absence of staging
siblings, exact Git-status delta, unchanged clean tracked/index state, and
unchanged HEAD equal to `P4L_PROVISION_AUTHORITY_FREEZE`. The only permitted
new untracked paths are the canonical destination directory and its two exact
children. No builder/P4-Q, trainer or dataset mutation, training, evaluation,
or promotion is authorized. A failure may retain only attempt-created staging
evidence and may not clean it automatically.

## 4. Corrected future command (not run here)

The following is the required command shape for the later authorized
execution. Its `provision()` body must be the frozen authority's complete
provisioning body, unchanged except for the identity checks identified below;
all prior checks listed in Section 3 are mandatory and may not be omitted.

```bash
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import os, re, subprocess, sys

AUTHORITY_RELATIVE = "reports/reason_router_p2_p3w6f2_p4l_current_exact_byte_provisioning_freeze_binding_correction_execution_authority_spec_candidate.md"
FREEZE = os.environ.get("P4L_PROVISION_AUTHORITY_FREEZE")

def blocked(code):
    raise SystemExit("P4L_PROVISION_BLOCKED:" + code)

def git(*args):
    p = subprocess.run(["git", *args], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        blocked("GIT_COMMAND_FAILED_" + "_".join(args))
    return p.stdout

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    blocked("GPU_NOT_OFF")
if FREEZE is None or re.fullmatch(r"[0-9a-f]{40}", FREEZE) is None:
    blocked("FREEZE_NOT_LOWERCASE_40_HEX")
if git("cat-file", "-t", FREEZE).strip() != "commit":
    blocked("FREEZE_NOT_COMMIT")
if git("cat-file", "-e", FREEZE + "^{commit}").strip() != "":
    blocked("FREEZE_COMMIT_UNRESOLVED")
tree_paths = git("ls-tree", "-r", "--name-only", FREEZE, "--", AUTHORITY_RELATIVE).splitlines()
if tree_paths != [AUTHORITY_RELATIVE]:
    blocked("FREEZE_DOES_NOT_CONTAIN_FINAL_CORRECTION_AUTHORITY")
if git("rev-parse", "HEAD").strip() != FREEZE:
    blocked("HEAD_NOT_EXACT_FREEZE")
if git("status", "--short", "--untracked-files=no") != "":
    blocked("TRACKED_WORKTREE_DIRTY")
if git("diff", "--cached", "--name-status") != "":
    blocked("INDEX_DIRTY")

# Execute the frozen authority's complete raw-byte provisioning body here,
# with its EXPECTED_HEAD binding replaced everywhere by FREEZE. The body must
# retain every source, destination, staging, renameat2, hash, byte-equality,
# status-delta, and postflight check from Section 3.
provision(expected_head=FREEZE)

if git("rev-parse", "HEAD").strip() != FREEZE:
    blocked("POST_HEAD_NOT_EXACT_FREEZE")
print("P3W6F2P4L_CURRENT_EXACT_BYTE_PROVISION_FREEZE_BOUND_PASS")
PY
```

The `provision` placeholder is a specification marker, not an invitation to
implement a different mechanism: the later execution artifact must inline the
already verified frozen body, with only its hard-coded `EXPECTED_HEAD`
replaced by the workflow-supplied `FREEZE`, and with the additional checks in
this section executed before mutation. This candidate itself does not run the
command and does not authorize that later inlining/provisioning step.

## 5. Proof obligations and negative cases

There is no self-reference impossibility. The candidate can be committed at
an unknown future SHA because it contains no future SHA. Once that commit is
created, the workflow supplies its actual 40-character SHA; the command proves
that the SHA is a commit, that its tree contains the final canonical authority
path, and that current HEAD equals it. Thus the same commit is both freeze
authority identity and execution HEAD.

Supplying `b818f523c95331058680128b5943dd94cf7aca4b` after a later correction
freeze fails unless that exact historical commit contains the final correction
authority path and is current HEAD; it cannot satisfy the corrected contract.
Supplying `a1b614d0e659d2b34889cb55aef94e1824df2fd1` fails either the authority
tree-presence check or the exact-current-HEAD check. An arbitrary commit that
does not contain the correction file fails the `git ls-tree` check even if it
is a valid commit. No Git transition is needed in any case.

## 6. Preserved boundaries and later validation

This correction does not alter source identity, provenance, destination,
filenames, containment, Linux-only behavior, dynamic `renameat2`,
`RENAME_NOREPLACE`, staging retention, raw copying, source validation, status
delta, scientific semantics, or clean-vs-external evaluation separation. It
does not authorize a builder, P4-Q, trainer/data mutation, training,
evaluation, Kaggle execution, or GPU use. A new current-HEAD read-only
provisioning-result validation authority remains required after any future
provisioning; this candidate does not create or execute that validator.

## 7. Creation result and stop conditions

Candidate creation is blocked if current HEAD is not
`b818f523c95331058680128b5943dd94cf7aca4b`, if tracked or index dirt exists,
if the candidate path already exists, if the frozen authority would need
editing, if any invariant above would be weakened, or if commit/tree presence
cannot be checked fail-closed. Existing untracked files are preserved and are
not part of this candidate's delta.

Required readiness token:

`P3W6F2P4L_CURRENT_EXACT_BYTE_PROVISIONING_FREEZE_BINDING_CORRECTION_CANDIDATE_READY_FOR_INDEPENDENT_VERIFICATION`
