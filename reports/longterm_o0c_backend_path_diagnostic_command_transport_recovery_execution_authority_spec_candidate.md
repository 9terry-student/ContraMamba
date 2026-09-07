# ContraMamba O0c backend-path diagnostic command-transport recovery execution authority — candidate

## Status, authority, and starting-state guard

This is a report-only recovery authority candidate under the current controller instruction. During authoring it authorizes no diagnostic execution, Kaggle use, run registration, implementation/test/package change, staging, commit, or push. Training and evaluation are `NO`.

Before authoring, require exactly: root `C:\\o0c-preflight-exec-auth-686b745`; HEAD `4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c`; no tracked modifications; nothing staged; no untracked files; this candidate path absent; and no task temporary files. Any mismatch blocks without changes. The observed authoring starting state met that guard.

The frozen parent diagnostic authority is commit `4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c`. It validly authorized one CPU-only, read-only/static backend-path diagnostic at exact execution commit `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`. Its intended scientific/infrastructure question and static-only boundaries remain valid. This candidate changes only the consumed run identity and command-transport safeguards after the v1 transport failure; it does not broaden the question.

## Validated v1 provenance and narrow failure classification

Freeze the following imported provenance exactly:

```text
RUN=longterm-o0c-backend-path-diagnostic-686b745-v1
EXECUTION_COMMIT=686b7457ed220e7d74ecdf41eb7ec500d24bb23f
REGISTERED_COMMAND_SHA256=72aa28bf4505e8937053cce0faf21e5d341ac57d482f6866ddffc138848547af
STARTED_UTC=2026-09-07T12:35:48Z
FINISHED_UTC=2026-09-07T12:35:48Z
EXIT_CODE=2
RUN_LOG_SHA256=315e459e57873d4c6ce08d660b10a183e1c590a22ae8a30ffbd1645da3edbdf9
RUN_META_SHA256=a37df1e8c34a256194a3217fb76ab3c8afa3dfa384b537a500d44e17d934adc9
HANDOFF_ZIP_SHA256=894ef0861dfc8f6646d084c64930f09e3e9795f6eda9ee0537464c4c57c37849
COLLECT=PASS
FILES_COLLECTED=0
IMPORT=PASS
```

The import audit is `C:\\Users\\Home1\\.contramamba\\imports\\longterm-o0c-backend-path-diagnostic-686b745-v1_686b7457ed22_20260907_213845`.

The registered v1 command payload was not the authorized Python backend-path diagnostic command. Its decoded command began with PowerShell control/registration content, including `$oldRoot = $env:CONTRAMAMBA_REPO_ROOT`, `try {`, `cm run save ...`, and `cm run ...`; the pinned Kaggle runner executed it as a Bash command file. The log consequently includes shell errors `=: command not found`, `try: command not found`, `cm: command not found`, and `syntax error near unexpected token`. The intended Python diagnostic payload did not execute.

Its sole classification is `VALIDATED_COMMAND_TRANSPORT_REGISTRATION_CONTENT_ERROR`. This is not backend-path diagnostic evidence, validator false-positive/negative evidence, runtime-source evidence, or scientific evidence. `DIAGNOSTIC_RESULT=NONE` and `SCIENTIFIC_CONCLUSION=NONE`.

`longterm-o0c-backend-path-diagnostic-686b745-v1` is permanently consumed: no rerun, overwrite, alias, re-registration, reuse of its command SHA, or interpretation as a backend diagnostic.

## Recovery scope and v2 reservation

The only recovery purpose is one replacement execution of the same previously authorized static backend-path diagnostic question, with a fresh identity and stronger command-transport guards. It authorizes no implementation/test/package change, no diagnostic-semantic change, no model loading or forward, and no training/evaluation.

Reserve exactly `longterm-o0c-backend-path-diagnostic-686b745-v2`; no other replacement name is authorized. At authoring, read-only collision checks found no incompatible v2 identity in repository text/state/history records; `C:\\Users\\Home1\\.contramamba\\run-registry.json`; `C:\\Users\\Home1\\.contramamba\\imports`; present relevant `C:\\Users\\Home1\\.contramamba\\handoff` / `handoffs` locations; or relevant Downloads/ContraMamba handoff ZIP filenames. References to v2 inside this candidate are not collisions. Repeat those checks immediately before v2 registration/execution. Any incompatible pre-existing v2 identity blocks.

The v2 execution commit remains exactly `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`; an authority-report commit cannot substitute. A HEAD mismatch or dirty execution worktree blocks.

## Preserved parent diagnostic contract

All substantive requirements of frozen parent authority `4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c` remain mandatory: exact CPU-only runtime guard; Python `3.12.13`; NumPy `2.0.2`; torch `2.10.0+cpu`; Transformers `5.0.0`; CUDA `False`; CUDA device count `0`; GPU OFF / Accelerator None; no package mutation; fresh runtime-source identity measurement; `MambaMixer.forward` AST/control-flow inspection; frozen helper replay of `_backend_proof_if`, `_has_optional_backend_path`, and `classify_backend`; static CPU reachability analysis; optional-symbol lexical/reachability analysis; and result-neutral classifications.

The diagnostic must not construct or load a model, run a model forward or synthetic-tensor dispatch experiment, use a tokenizer/dataset/generation, or perform training/evaluation. It must preserve `SCIENTIFIC_CONCLUSION=NONE`.

## Pre-registration command-transport guard

The controller must provide exactly one shell command line for the diagnostic. Before `cm run save`, verify the local clipboard payload read-only and require every condition below:

1. The clipboard contains exactly one non-empty line, with no carriage return or embedded newline.
2. The command is the controller-approved Python/shell diagnostic command.
3. It contains none of the PowerShell control syntax `$env:`, `$oldRoot`, `try {`, `finally {`, or `Remove-Item`.
4. It contains none of `cm run`, `cm run save`, `cm collect`, `cm import`, or `CONTRAMAMBA_REPO_ROOT`.
5. Locally compute command SHA256 before registration; it must equal the controller-declared expected command SHA256.
6. If the command embeds a compressed/encoded Python payload, compute its diagnostic payload SHA256 and require equality with the controller-declared expected payload SHA256.

Failure of any guard blocks registration. Do not silently trim, normalize, CRLF-convert, or semantically edit the command after hash verification.

## Registration, execution, and result guards

Only after command-guard `PASS`, run:

```text
cm run save longterm-o0c-backend-path-diagnostic-686b745-v2
```

Inspect saved output and require HEAD `686b7457ed220e7d74ecdf41eb7ec500d24bb23f` and saved HASH exactly equal to the locally measured pre-registration command SHA256. Only then may the controller run `cm run longterm-o0c-backend-path-diagnostic-686b745-v2`. Require the generated pinned handoff to report that same HEAD and command SHA. Do not execute its generated Kaggle cell until those identities have been reviewed and explicitly authorized by the controller.

Before pinned v2 execution, require safe bootstrap `PASS`; Kaggle HEAD exactly `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`; clean Kaggle worktree; Accelerator None / GPU OFF; a fresh run-name collision result of `NO`; runner expected/actual HEAD equality; runner decoded command SHA equal to the registered SHA; and payload verification that the command is the approved diagnostic command rather than a control block. Any mismatch stops.

No outcome is predetermined. Only the parent-authorized result-neutral outcomes remain possible: `VALIDATOR_BACKEND_PATH_PATTERN_FALSE_NEGATIVE`, `VALIDATOR_OPTIONAL_BACKEND_REACHABILITY_OVERAPPROXIMATION`, `BACKEND_CPU_SELECTION_GENUINELY_UNRESOLVED`, `RUNTIME_SOURCE_IDENTITY_DRIFT`, or `DIAGNOSTIC_INCONCLUSIVE`.

## Consumption, collection/import, and failure recovery

After its first actual v2 execution attempt, v2 is consumed. Then run `cm collect longterm-o0c-backend-path-diagnostic-686b745-v2` in that same Kaggle session, download the ZIP, and locally run `cm import <handoff.zip>` at exact execution commit `686b7457ed220e7d74ecdf41eb7ec500d24bb23f`. No diagnostic interpretation/root-cause report may occur before `IMPORT=PASS`.

If v2 registration or execution has another command identity/content failure, stop; do not reuse v2 and do not invent v3 without another authority. Preserve logs and collect/import if an execution attempt began and final metadata exists.

Keep evidence layers separate: parent diagnostic authority validity is `FROZEN`; v1 execution is `EXECUTED TRANSPORT FAILURE`; v1 provenance is `VALID` after COLLECT/IMPORT PASS; v1 diagnostic result is `NONE`; v2 recovery authority is `NOT YET FROZEN` during authoring; backend root cause is `NOT YET ESTABLISHED`; scientific conclusion is `NONE`.

## Independent verification and authoring validation

Independent verification is required before freeze because this changes authority, provenance, and run identity. The verifier must check: exact parent authority; exact v1 imported provenance; v1 consumed status; transport-failure interpretation from logs; diagnostic result `NONE`; v2 collision status; exact v2 execution commit; complete preservation of parent semantics/boundaries; clipboard/command-hash guard; prohibition of PowerShell/cm-control content in registered command; registration review before Kaggle execution; result neutrality; collection/import; failure recovery; `SCIENTIFIC_CONCLUSION=NONE`; raw candidate identity; and final Git state.

After authoring, run only `git diff --check`, `git diff --name-status`, `git diff --cached --name-status`, `git status --short`, and `git rev-parse HEAD`. The required final state is HEAD `4ced96fa8f8f2d71492fba3c8a5a2dec1a7c6c0c`, no tracked modifications, nothing staged, exactly this untracked candidate, and no task temporary files. Compute and report candidate SHA256, bytes, LF, CR, and final-LF from raw bytes. Do not stage. The next authorized action is independent verification of this candidate; diagnostic execution, registration, Kaggle use, interpretation, and implementation remain unauthorized.
