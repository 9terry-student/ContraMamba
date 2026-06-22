# ContraMamba-CAR Related Work Notes

This file tracks citation categories that require bibliographic verification before the paper is converted to a submission format. It intentionally contains no BibTeX entries or unverified author, venue, or year information.

## Fact verification and NLI-style verification

- **FEVER / fact verification benchmark:** establish the standard claim-evidence verification framing and label space. [TODO: cite FEVER / fact verification benchmark]
- **VitaminC:** consider only if the final paper discusses contrastive claim-evidence training or evaluation. [TODO: cite VitaminC if used]
- **NLI-style claim-evidence verification:** identify representative work connecting natural language inference to evidence verification. [TODO: cite NLI-style claim-evidence verification]

## Factuality and hallucination detection

- **SelfCheckGPT:** verify the exact paper and use it only to position sampling or self-consistency-based hallucination detection. [TODO: cite SelfCheckGPT]
- **FActScore:** verify the exact paper and use it only to position factual precision evaluation. [TODO: cite FActScore]
- **Factuality evaluation overview:** identify a suitable verified survey or overview if broad field context is needed. [TODO: cite factuality evaluation survey]

ContraMamba-CAR should remain explicitly distinguished from deployed hallucination detection: its evidence and claims concern controlled claim-evidence verification.

## Uncertainty, abstention, and selective prediction

- Identify foundational and recent selective-prediction work relevant to rejection or abstention. [TODO: cite uncertainty / selective prediction]
- Identify verified NLP-specific abstention work if needed. [TODO: cite abstention in NLP]

The related-work comparison should focus on the distinction between scalar confidence rejection and structured entitlement auditing. `NOT_ENTITLED` denotes an evidence relation, not merely low confidence.

## Faithfulness and controlled evaluation

- Identify verified contrast-set or counterfactual evaluation work that tests targeted behavioral changes. [TODO: cite counterfactual or contrast-set evaluation]
- Identify work on faithfulness under controlled perturbations, especially when both outputs and internal signals are evaluated. [TODO: cite faithfulness evaluation under perturbations]

Avoid implying that the controlled dataset represents all naturally occurring evidence failures.

## State-space sequence models

- Verify the canonical Mamba citation and the exact version relevant to the backbone used here. [TODO: cite Mamba]
- Add broader state-space sequence-model context only if it materially supports the method description. [TODO: cite state-space sequence models]

The paper should not present Mamba as the main novelty. The methodological focus is classifier-auditor entitlement routing.

## Citation verification checklist

* Verify exact FEVER citation.
* Verify exact VitaminC citation if used.
* Verify exact SelfCheckGPT citation.
* Verify exact FActScore citation.
* Verify exact Mamba citation.
* Verify selective prediction / abstention citations.
* Verify counterfactual evaluation / contrast sets citations.
