# ContraMamba Gen4 Related-Work Citation Spine v1

## Status

- Status: STATIC CURRENT-LITERATURE AUDIT
- Audit date: 2026-09-22
- Repository HEAD: `e1b3cf48db498f9f0162e266a10257a171ba6f18`
- Scientific execution: CLOSED
- New scientific claims: NONE
- Priority claim authorized: NO

This artifact records manuscript-facing primary literature verified during the
current Related Work audit. `NOVEL_CANDIDATE` remains a positioning label, not
a first-ever priority claim.

## Primary citation set

### `gu2023mamba`

- Albert Gu and Tri Dao.
- *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
- arXiv:2312.00752.
- Role: architecture background only.

### `sharma2024locating`

- Arnab Sen Sharma, David Atkinson, David Bau.
- *Locating and Editing Factual Associations in Mamba*.
- COLM 2024.
- arXiv:2404.03646.
- Role: closest prior for factual causal tracing, localization, information-flow
  intervention, and model editing in Mamba.
- ContraMamba must not claim first causal localization or first editing in Mamba.

### `jafari2024mambalrp`

- Farnoush Rezaei Jafari, Grégoire Montavon, Klaus-Robert Müller, Oliver Eberle.
- *MambaLRP: Explaining Selective State Space Sequence Models*.
- NeurIPS 2024.
- arXiv:2406.07592.
- DOI: 10.52202/079017-3764.
- Role: prior Mamba-specific attribution and explainability.
- ContraMamba must not claim first interpretation/explanation of Mamba.

### `wang2025universality`

- Junxuan Wang, Xuyang Ge, Wentao Shu, Qiong Tang, Yunhua Zhou, Zhengfu He,
  Xipeng Qiu.
- *Towards Universality: Studying Mechanistic Similarity Across Language Model
  Architectures*.
- ICLR 2025.
- Role: closest prior for Transformer-Mamba mechanistic similarity, feature
  correspondence, and analogous induction circuits.
- Boundary: cross-architecture similarity is not the same question as whether
  recurrence of a causally validated role across checkpoints preserves its
  geometry or downstream task alignment.

### `endy2025knockout`

- Nir Endy, Idan Daniel Grosbard, Yuval Ran-Milo, Yonatan Slutzky,
  Itay Tshuva, Raja Giryes.
- *Mamba Knockout for Unraveling Factual Information Flow*.
- ACL 2025 Long Papers.
- arXiv:2505.24244.
- DOI: 10.18653/v1/2025.acl-long.1143.
- Role: token- and layer-level factual information-flow localization in Mamba-1
  and Mamba-2.

### `arora2025mechanistic`

- Aryaman Arora, Neil Rathi, Nikil Roashan Selvam, Róbert Csórdas,
  Dan Jurafsky, Christopher Potts.
- *Mechanistic evaluation of Transformers and state space models*.
- arXiv:2505.15105.
- Role: causal mechanistic comparison showing that similar behavioral
  performance can arise from different internal retrieval mechanisms.
- Boundary: does not directly test checkpoint-local causal-role recurrence,
  geometric preservation, or matched downstream readout orientation.

### `mohan2026subspace`

- Vamshi Sunku Mohan, Kaustubh Gupta, Aneesha Das, Chandan Singh.
- *Interpreting and Steering State-Space Models via Activation Subspace
  Bottlenecks*.
- ICML 2026.
- arXiv:2602.22719.
- Role: closest prior for activation-subspace interpretation and test-time
  steering in Mamba-family SSMs.
- ContraMamba must not claim subspace intervention or steering itself as novel.

### `jiang2026circuit`

- Yuhang Jiang and Bowen Zhang.
- *A Circuit, Not The Circuit: Non-Unique Causal Localisation of the Mamba-2
  State Sink*.
- arXiv:2606.00930.
- Status: preprint.
- Role: closest conceptual prior for causal-localization non-uniqueness,
  representation/function dissociation, and intervention-surface dependence.
- Critical novelty boundary: `representation != function` cannot be the
  standalone ContraMamba novelty claim.

### `koren2026recall`

- Yuval Koren, Assaf Ben-Kish, Raja Giryes, Lior Wolf, Itamar Zimerman.
- *On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via
  Hashing*.
- arXiv:2609.07681.
- Status: preprint; submitted September 2026.
- Role: current closest prior for mechanistic scaling questions in Mamba.
- Boundary: studies associative-recall capacity/circuit scaling, not the
  preservation implications of a recurrent causal role across fixed pretrained
  checkpoints.

## Paper-facing gap after current audit

The Related Work should explicitly concede all of the following:

1. Mamba mechanisms have already been causally localized and edited.
2. Mamba information flow has already been studied with causal interventions.
3. Mamba-specific attribution/explainability methods already exist.
4. Transformer-Mamba mechanistic similarity has already been studied.
5. Activation subspaces in SSMs have already been interpreted and steered.
6. Causal localization in Mamba need not be unique and can depend on
   intervention surface.
7. Mamba scaling has already been studied mechanistically in associative recall.

The remaining paper gap is narrower:

> When a causally validated internal role recurs across model checkpoints, does
> that recurrence require preservation of its geometric realization or its
> downstream task-readout orientation?

The frozen ContraMamba evidence addresses this relation within one common
causal program.

## Prohibited literature positioning

Do not write:

- first mechanistic interpretation of Mamba;
- first causal intervention in Mamba;
- first causal subspace in an SSM;
- first steering of Mamba;
- first demonstration that representation and causal function differ;
- first mechanistic comparison involving Mamba;
- first Mamba scaling analysis;
- universal theorem that recurrence never preserves geometry;
- architecture-independent universality;
- any first-ever priority statement based on this targeted audit.

`RELATED_WORK_CITATION_SPINE_V1_READY = YES`
