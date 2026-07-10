# Stage149-B Route/Order False-Positive Forensic Synthesis

## Summary decision

Decision: `STAGE149B_ROUTE_ORDER_FALSE_POSITIVE_PATCHABLE_SELF_LOOP_TITLE_ERROR`

Stage149-A confirmed that the single broader false positive from Stage148-A was `Side to Side`, a song title. The analyzer extracted a degenerate self-loop route `side -> side` and treated it as a route/order signal.

The failure is patchable. Final integration remains blocked.

## Stage148-A failure recap

Stage148-A applied `route_order_reversal_v1` to broader existing prediction exports and did not pass as a broader/general shadow diagnostic.

- Rows: 33,000
- Changed predictions: 1
- Delta false SUPPORT: 0
- Delta false NOT-ENTITLED: +1
- Feature false SUPPORT TP: 0
- Feature correct SUPPORT FP: 1

The only broader route/order trigger was harmful, so route/order broader use remains blocked until v2 patch and revalidation.

## Forensic row

- Gold label: `SUPPORT`
- Original prediction: `SUPPORT`
- Shadow prediction: `NOT_ENTITLED`
- False positive: true
- True positive: false

Claim:

`Side to Side was released after June 2 , 2016 .`

Evidence:

`'' Side to Side '' is a song recorded by American singer Ariana Grande , released on August 30 , 2016 , as the third single from her studio album , Dangerous Woman ( 2016 ) .`

Extracted canonical claim route:

`side -> side`

Extracted canonical evidence route:

`side -> side`

## Failure cause

This is not a real route/order reversal. `Side to Side` is a song title, not a route. The extractor incorrectly interpreted the title phrase as a directional route expression and produced a degenerate self-loop in both claim and evidence.

A route with source equal to destination should never trigger reversal.

## Why this is patchable

The failure is local to route extraction and policy guarding. A conservative v2 patch can reject any directional route whose canonical source equals canonical destination, preventing self-loop routes such as `side -> side` from triggering.

The same patch can add a title/context guard for quoted or title-like phrases around `to` patterns while preserving the existing shadow-only design, non-SUPPORT safety, organization endpoint blocking, and alias canonicalization.

## Boundary after Stage149

Stage147-B targeted pass remains valid: targeted synthetic route/order reversal cases were caught without correct-SUPPORT false positives.

Stage148-A broader failure is now explained: the only broader trigger was a patchable extractor error. Even so, `route_order_reversal_v1` broader use remains blocked until v2 patch and revalidation. It is not broader-shadow ready and not final integration ready.

## Stage150 recommendation

Stage150-A should implement `route_order_reversal_v2` with conservative self-loop and title/context guards, then re-run Stage147-B plus Stage148-A.

Required patch behavior:

- Reject any directional route whose canonical source equals canonical destination.
- Do not trigger on self-loop routes such as `side -> side`.
- Add conservative title/context guard for quoted or title-like phrases around to-patterns.
- Keep non-SUPPORT prediction safety.
- Keep organization endpoint blocking.
- Keep alias canonicalization.
- Keep analyzer shadow-only.
- Do not modify source predictions or final logits.

Avoid automatic final prediction override, training loss integration, checkpoint selection using route/order outputs, and broad route/order deployment before v2 revalidation.

## Safety policy

The route/order analyzer remains shadow-only and diagnostic-only. It does not mutate source predictions, final logits, final predictions, training, checkpoint selection, Stage128 guard behavior, Stage15 behavior, external training data, or threshold/model-selection behavior.

The policy uses claim text, evidence text, the original prediction, deterministic route rules, deterministic alias rules, and deterministic organization-like rules. It does not use gold labels for policy behavior, intervention type, diagnostic family labels, file path heuristics, or row id heuristics.

Remaining risks:

- Other title-like to-phrases may still cause false positives unless guarded.
- Route extraction remains deterministic and pattern-based.
- Complex route language and implicit directionality remain untested.
- Broader route/order use must be re-run after Stage150 patch.
