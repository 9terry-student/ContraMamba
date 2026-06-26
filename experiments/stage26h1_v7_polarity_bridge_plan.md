\# Stage26-H1: v7 Polarity Bridge Plan



\## Current finding



Dummy/toy results are not model evidence. They are allowed only as plumbing or hypothesis seeds.



However, v6B PolarityEnergy is real-Mamba validated:

\- Stage23/24 clean predictions show SUPPORT/REFUTE recall near 1.0.

\- Polarity margin is strongly positive for SUPPORT and strongly negative for REFUTE.



Current v7 PolarityChannel is real-Mamba tested but failed:

\- Stage26F/G/G2/G3/G4 show SUPPORT recall = 0.0.



\## Main hypothesis



The failure is not the polarity idea itself.



The likely failure point is the v6B -> v7 semantic bridge:



v6B:

\- positive\_energy = softplus(positive\_head)

\- negative\_energy = softplus(negative\_head)

\- support\_logit = entitlement\_prob \* positive\_energy

\- refute\_logit = entitlement\_prob \* negative\_energy

\- ne\_logit = bias + alpha \* (1 - entitlement\_prob)



v7:

\- support\_score = entitlement\_logit + raw\_support\_logit

\- refute\_score = entitlement\_logit + raw\_refute\_logit

\- ne\_score = -entitlement\_logit + ne\_bias



Therefore Stage26-H1 should test whether restoring v6B-style softplus energy and multiplicative decision inside v7 recovers SUPPORT.



\## Proposed patch



Add optional flags, default off:



\- --v7-use-softplus-polarity-energy

\- --v7-use-v6b-style-polarity-composition



When enabled:



positive\_energy = softplus(v7\_polarity\_support\_logit)

negative\_energy = softplus(v7\_polarity\_refute\_logit)

entitlement\_for\_decision = v7\_entitlement\_prob



support\_score = entitlement\_for\_decision \* positive\_energy

refute\_score = entitlement\_for\_decision \* negative\_energy

ne\_score = ne\_bias + alpha \* (1.0 - entitlement\_for\_decision)



final\_logits = \[refute\_score, ne\_score, support\_score]



\## Invariants



\- Default v7 behavior must remain unchanged.

\- Final label order remains \[REFUTE=0, NOT\_ENTITLED=1, SUPPORT=2].

\- output\["logits"] remains final logits and CE uses output\["logits"].

\- v6B must not be changed.

\- Stage15/OOD/time\_swap must not be used for training, tuning, selection, thresholding, or calibration.

\- Dummy results remain plumbing-only and are not performance evidence.



\## Expected diagnostic



If H1 recovers SUPPORT:

\- v7 failure was final polarity composition / raw-logit semantics.



If H1 still has SUPPORT recall = 0:

\- v7 failure is likely polarity input representation or channel training dynamics.

