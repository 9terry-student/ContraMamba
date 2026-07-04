\*\*1. Files Inspected\*\*

\- `src/contramamba/modeling\_v7\_hierarchical.py`

\- `src/contramamba/modeling\_v6b\_minimal.py`

\- `src/contramamba/heads/\*.py`

\- `src/contramamba/labels.py`

\- `src/contramamba/losses.py`

\- `scripts/train\_controlled\_v6b\_minimal.py`

\- `scripts/train\_controlled\_v5.py`

\- Relevant `data/` and `results/` filenames/provenance conventions



\*\*2. Existing Relevant Components\*\*

\- `modeling\_v7\_hierarchical.py` already contains an entitlement-first architecture in spirit:

&#x20; - frame/predicate/sufficiency/temporal signals feed `EntitlementGateV7`

&#x20; - polarity is post-entitlement

&#x20; - final order is explicitly `REFUTE=0, NOT\_ENTITLED=1, SUPPORT=2`

&#x20; - `output\["logits"]` is final CE/prediction logits

\- Shared heads are usable:

&#x20; - `FrameGate`

&#x20; - `PredicateCoverageHead`

&#x20; - `SufficiencyGate`

&#x20; - `PolarityEnergyHead`

&#x20; - `FinalEntitlementDecisionHead`

\- Runner already supports `--architecture v7\_hierarchical`, but only as the old v7 path.

\- Prediction export is fairly strong:

&#x20; - `prediction\_records\_v6b()` exports `final\_logits`, `final\_probs`, per-class logits/probs, gold/pred labels, and v7 scalars when present.

&#x20; - External/VitaminC prediction exports use the same record path and include label-space metadata.

\- Clean/external separation is explicitly guarded in several places:

&#x20; - temporal diagnostic/safety/mismatch data cannot equal main `--data`

&#x20; - external eval is marked eval-only and not used for checkpoint selection/calibration/training.



\*\*3. Salvageability Decision\*\*

\*\*C. HYBRID\_SALVAGE\_HEADS\_NEW\_ROUTER\*\*



The existing v7 is conceptually close enough to salvage its heads, output-contract ideas, label order, and export compatibility. But the file is too entangled with Stage26-H1, Stage27-H2, Stage28 location caps, Stage30 temporal safety/mismatch/preservation caps, and later shadow/export-owner machinery to be the clean vNext implementation surface.



Stage109 should create a dedicated vNext model file and reuse stable head modules rather than patching `modeling\_v7\_hierarchical.py` into yet another branch of historical behavior.



\*\*4. Recommended Stage109 Patch Scope\*\*

\- Add a new minimal vNext model file with a clean entitlement-first router.

\- Keep canonical final logits order: `\[REFUTE, NOT\_ENTITLED, SUPPORT]`.

\- Reuse stable heads from `src/contramamba/heads/`.

\- Keep output keys compatible with current runner/export code:

&#x20; - `logits`

&#x20; - `base\_logits`

&#x20; - `predictions`

&#x20; - `frame\_prob`

&#x20; - `predicate\_coverage\_prob`

&#x20; - `sufficiency\_prob`

&#x20; - `entitlement\_prob`

&#x20; - `polarity\_margin`

&#x20; - `positive\_energy`

&#x20; - `negative\_energy`

\- Add a new runner architecture option such as `vnext\_minimal`.

\- Ensure prediction export gets logits/probs unchanged through `prediction\_records\_v6b()`.

\- Keep external/VitaminC diagnostic-only hooks unchanged.



\*\*5. Do-Not-Touch List\*\*

\- Do not change clean main data path policy:

&#x20; - `data/controlled\_v5\_v3\_without\_time\_swap.jsonl`

\- Do not use `time\_swap` as final classifier training data.

\- Do not convert VitaminC/external data into training, selection, or calibration input.

\- Do not alter canonical label mapping:

&#x20; - `REFUTE=0`

&#x20; - `NOT\_ENTITLED=1`

&#x20; - `SUPPORT=2`

\- Do not modify Stage71/Stage73/Stage99/Stage100 result artifacts.

\- Do not change existing v6B or v7 behavior by default.



\*\*6. Risks\*\*

\- \*\*Label order:\*\* v7 and labels are correct, but old v6B comments around index `0`/`2` are misleading. Stage109 should assert/log label order in the new file.

\- \*\*Final logits order:\*\* must remain `\[refute, ne, support]`; export assumes this.

\- \*\*Runner naming:\*\* current architecture choices only include `v6b\_minimal` and `v7\_hierarchical`.

\- \*\*Loss keys:\*\* runner and pairwise losses expect legacy keys like `entitlement\_prob`, `positive\_energy`, `negative\_energy`, and `polarity\_margin`.

\- \*\*Prediction export:\*\* reliable if new vNext returns existing output keys; brittle if renamed.

\- \*\*Checkpoint/report naming:\*\* metadata currently maps model version from `args.architecture`; new architecture needs explicit metadata.

\- \*\*Clean/external separation:\*\* mostly strong already, but vNext must not add any shortcut that uses external metrics for thresholding or selection.

\- \*\*v7 complexity:\*\* modifying `modeling\_v7\_hierarchical.py` risks accidentally activating or preserving old H1/cap assumptions.



\*\*7. Exact Files Likely To Be Modified In Stage109\*\*

\- `src/contramamba/modeling\_vnext\_minimal.py` new

\- `scripts/train\_controlled\_v6b\_minimal.py`

\- Possibly `src/contramamba/\_\_init\_\_.py`

\- Possibly a focused static/plumbing test file if Stage109 allows tests later, but not needed for this Stage108 audit.

