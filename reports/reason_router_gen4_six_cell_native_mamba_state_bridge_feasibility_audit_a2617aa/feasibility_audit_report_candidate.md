# Gen4 Six-Cell Native Mamba State Bridge Feasibility Audit Report ? Candidate

## Formal decision

`PASS_READY_FOR_SEPARATE_PHASE_C_AUTHORITY`

- Authority commit: `7337efa05f7230a6e0c700a8ee99579c337c4243`
- Parent scientific specification: `a2617aa037d1a9834003535b62ac81770a5b96aa`
- Scientific conclusion: `NONE`
- Model forward performed: `NO`
- Native-state extraction performed: `NO`
- Kinematic endpoint computation performed: `NO`
- Statistical testing performed: `NO`

## Phase A ? native state source binding

- `NATIVE_TENSOR_SOURCE = BOUND`
- Module: `transformers.models.mamba.modeling_mamba`
- Class/function: `MambaMixer.slow_forward`
- Tensor object: local `ssm_state`
- Local static source update/readout: `350 -> 351`
- Timing: post-consumption `s_t`, after recurrence update and before C readout.
- Native state shape: `[batch, 1536, 16]`
- Per-example flattened dimension: `24576`
- Vectorization: contiguous per-example `ssm_state.reshape(-1)`.
- Layer count: `24`, zero-based `0..23`
- `L_PRIMARY = 11`

### Runtime/source provenance

- Frozen R5 Transformers runtime: `5.0.0`
- Frozen R5 backend: `transformers_5.0.0_no_mamba_ssm`
- Current local static inspection Transformers: `5.12.1`
- Current local Mamba source SHA256: `23c7b410e204b5da01732566de10c94b70a8418ecb608e409754b00332eb2a41`
- Current local cache source SHA256: `2783b2702910dbb6e70aa05a093f71e1c51c110d7fa73c904fc38d12e5882545`
- The current local source is static audit evidence, not automatically the future Phase C execution source.
- Phase C must exact-pin its runtime source hash before implementation validation/execution.

### 18-checkpoint native-backbone identity

- Prespecified checkpoints authenticated: `18/18`
- Safe load: `torch.load(..., map_location="cpu", weights_only=True)`
- Native-backbone namespace: `mamba.*`
- Native-backbone tensor keys: `242`
- Key-set SHA256: `ca60db8c4379a0d84a2218023966107b9ba8a7816128aab8bd5813a20aa2e153`
- Native-backbone signature SHA256: `81cd368d8a94932561e0ccd50f45a7db1f27941c00b3a08c8b816badaf25f415`
- `NATIVE_MAMBA_BACKBONE_IDENTITY = IDENTICAL`
- `NATIVE_STATE_MODEL_REPLICATION_COUNT = 1`

### O0c reuse disposition

- `BOUNDED_PHASE_C_INSTRUMENTATION_DELTA_REQUIRED`
- The native recurrent-state ontology and post-update/pre-readout capture concept are reusable.
- Gen4 requires a separately authorized runtime-source binding, Gen4 integration, and synthetic non-interference validation.

## Phase B ? event-anchor and prefix feasibility

- Generator render identity: `PASS_1800_OF_1800`
- Tokenizer byte identity: `PASS`
- Source pairs: `300`
- Required cell-anchor rows: `3600`
- A_TITLE: `600/600`
- A_NAME: `600/600`
- A_ROLE: `600/600`
- A_PREDICATE: `600/600`
- A_IDENTITY: `1200/1200`
- Complete source pairs: `300/300`
- `PRIMARY_COMPLETE_PAIR_PREFIX_FEASIBILITY = PASS_300_OF_300`
- POST4 window was not shortened.

## Inspected repository files

- `7337efa05f7230a6e0c700a8ee99579c337c4243:reports/reason_router_gen4_six_cell_native_mamba_state_bridge_feasibility_audit_authority_spec_candidate.md` ? `f9c997a65eebefa286f967783d4d6d2d41109394a959b9bf6bc0f65f96b6ebae`
- `a2617aa037d1a9834003535b62ac81770a5b96aa:reports/reason_router_gen4_six_cell_native_mamba_state_mechanistic_bridge_spec_candidate.md` ? `27a42e9e19c47dbeb6c52d3b0ddc3b74e63be74cd52f35a89876b0d940031997`
- `7337efa05f7230a6e0c700a8ee99579c337c4243:reports/reason_router_gen4_six_cell_tier2_checkpoint_loadability_83f32cbb8bfbed7b2b88d0cf864499422e650954/r4_checkpoint_loadability_summary.json` ? `ecd9e939e8103fbb80bd7c77114e35e53ed6352a514405676572bae8408632bd`
- `7337efa05f7230a6e0c700a8ee99579c337c4243:reports/reason_router_gen4_r5_execution_cf0826174c2ab1b2203f68afbdeed9da3ff64aa2_retry1/r5_scientific_inference_summary.json` ? `e6887d86c1ac2242ce5c2d74c379912fd81ac441d71b23a13d0db37412d2905b`
- `cf0826174c2ab1b2203f68afbdeed9da3ff64aa2:scripts/reason_router_gen4_six_cell_tier2_scientific_inference.py` ? `468a758a7d20d048c75a0ca7e298b73a65f538527df55d3ecad3c7ff1760cf4d`
- `7337efa05f7230a6e0c700a8ee99579c337c4243:scripts/reason_router_gen4_six_cell_tier2_inference_adapter.py` ? `83177c351f82a781586c63bd8d4ef1b40e759b5a94858dc1837d65502cbff6e5`
- `7337efa05f7230a6e0c700a8ee99579c337c4243:reports/reason_router_gen4_six_cell_masked_slot_substitution_materialization_fbc780ce12cbfbcf4e20a3cb9d1f099553045fc0/gen4_six_cell_masked_slot_substitution.jsonl` ? `b9c54604863ed15c237fa17c7890a20f3f5ec062a7429638b39e8b468be050a7`
- `91c3dcd7abadcd6cf0d6d2f1c299f3d6fa6e28ea:scripts/build_controlled_v5.py` ? `9fbd94a151c4d83a5e824412d7c0837062fedd20628f4f198116b2d08b679410`

## Checkpoint payloads

- seed `180`, arm `G3-GROUP-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-D-HALF/selected_checkpoint.pt` ? `1ff3fcf2ebd754ab6f9483d6a9982b9b04b9a4eb3357f9f8cdbe2b30399e7d2f` ? `PASS`
- seed `180`, arm `G3-GROUP-Q-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-Q-D-HALF/selected_checkpoint.pt` ? `2e51f64702a3ebf21d5d8e8aa84745b62b3faa01112b5b8f10525ba6435dbc8c` ? `PASS`
- seed `180`, arm `G3-GROUP-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-Q-HALF/selected_checkpoint.pt` ? `eb349aefca6d992df42f6239e7cf642d560755c0b1819397dba1d746b33bd8e3` ? `PASS`
- seed `180`, arm `G3-GROUP-U-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-U-D-HALF/selected_checkpoint.pt` ? `08654abb9c1ec67d42fa1b3464f19298f21ff79b866fb0cf8b7a97d59a45ff86` ? `PASS`
- seed `180`, arm `G3-GROUP-U-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-U-HALF/selected_checkpoint.pt` ? `a8cd296136816f806394ca98d6433bfa560f5691ab37e661347c2db838966708` ? `PASS`
- seed `180`, arm `G3-GROUP-U-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed180/G3-GROUP-U-Q-HALF/selected_checkpoint.pt` ? `0701ce934ae3ef34cd9f9d229c9321599b4ca150db8dabc3c8a740668b8f0aad` ? `PASS`
- seed `181`, arm `G3-GROUP-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-D-HALF/selected_checkpoint.pt` ? `afc55ef0bf6a250dadc16dfa85ae2350505dd1289e781e109519c6bc8009422f` ? `PASS`
- seed `181`, arm `G3-GROUP-Q-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-Q-D-HALF/selected_checkpoint.pt` ? `390b4fe3266d8eddebe74d9732321d1f96e2a7095ecae67b6155a2d535b655ba` ? `PASS`
- seed `181`, arm `G3-GROUP-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-Q-HALF/selected_checkpoint.pt` ? `3b5044fddb7f542c9e06a318a5a81a731d94475f7f67b7e5c5a7787ab3af0ba6` ? `PASS`
- seed `181`, arm `G3-GROUP-U-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-U-D-HALF/selected_checkpoint.pt` ? `7adffc577e00b9a9150bca28ed83b35eb5574458f71d5bc276ebd8f557b00e4d` ? `PASS`
- seed `181`, arm `G3-GROUP-U-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-U-HALF/selected_checkpoint.pt` ? `e2a9fd1ca6e50856b2349fc5bc915c54e6a71848aaa8c59aaa1f8c8647e89699` ? `PASS`
- seed `181`, arm `G3-GROUP-U-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed181/G3-GROUP-U-Q-HALF/selected_checkpoint.pt` ? `1be3be2ddd13762d36c69ef16ccbdd0ee4bd5ad732eff46e7a66cab703c2db50` ? `PASS`
- seed `182`, arm `G3-GROUP-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-D-HALF/selected_checkpoint.pt` ? `f9db48a3b3b9fdc6df4e2bb2086d11fd80fd595e6096c0095d1992f6c7d777f2` ? `PASS`
- seed `182`, arm `G3-GROUP-Q-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-Q-D-HALF/selected_checkpoint.pt` ? `cb1f4812d11643089bb87064c436b2e890554435254c961e5ed3f766b61b412b` ? `PASS`
- seed `182`, arm `G3-GROUP-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-Q-HALF/selected_checkpoint.pt` ? `67d0cbf855b24a291f55ce87425dcd4d77b5f7a59fb119c57c261c6378a4342e` ? `PASS`
- seed `182`, arm `G3-GROUP-U-D-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-U-D-HALF/selected_checkpoint.pt` ? `f1d84bab31f9080f0f3cfc6d0ee49cdc2743ad7c8c3a620bee7f576ca32ebef1` ? `PASS`
- seed `182`, arm `G3-GROUP-U-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-U-HALF/selected_checkpoint.pt` ? `47b43899119a0a450e0b5cf8134ade521d8cea9ca568110b32223de6109ef5a4` ? `PASS`
- seed `182`, arm `G3-GROUP-U-Q-HALF` ? `reports/reason_router_gen3_grouped_factorial_runs/seed182/G3-GROUP-U-Q-HALF/selected_checkpoint.pt` ? `129be6e930b5f7e6737ad671a9646c867d150e8751907bb1abfd3fe64570669f` ? `PASS`

## Scientific boundary

This audit establishes technical identifiability and feasibility only.

It does **not** establish:

- native-state kinematic response;
- native-state causality;
- output-state mediation;
- training benefit;
- task-performance improvement;
- arbitrary model or dataset generalization.

A successful feasibility audit does not authorize Phase C automatically.
It does not authorize native-state extraction or the 15 primary statistical tests.

## Final disposition

`PASS_READY_FOR_SEPARATE_PHASE_C_AUTHORITY`
