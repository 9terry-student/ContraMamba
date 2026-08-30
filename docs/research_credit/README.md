# Research-Credit Reporting

This directory is a narrative scaffold for the research-credit period. The
source of truth for provenance, authority identity, exact hashes, command
identity, run identity, and validation evidence remains `reports/`.

## Calendar Boundary

- Week boundary: Monday 00:00 Asia/Seoul.
- Week 0 is kickoff/formal-start preparation outside the 15 official weeks.
- Weeks 1 through 15 are the official research-credit weeks.

## Week Calendar

| Week | Dates (Asia/Seoul) | Role |
| --- | --- | --- |
| Week 0 | 2026-08-24 through 2026-08-30 | Kickoff/formal-start preparation; outside official 15 weeks |
| Week 1 | 2026-08-31 through 2026-09-06 | Official week |
| Week 2 | 2026-09-07 through 2026-09-13 | Official week |
| Week 3 | 2026-09-14 through 2026-09-20 | Official week |
| Week 4 | 2026-09-21 through 2026-09-27 | Official week |
| Week 5 | 2026-09-28 through 2026-10-04 | Official week |
| Week 6 | 2026-10-05 through 2026-10-11 | Official week |
| Week 7 | 2026-10-12 through 2026-10-18 | Official week |
| Week 8 | 2026-10-19 through 2026-10-25 | Official week |
| Week 9 | 2026-10-26 through 2026-11-01 | Official week |
| Week 10 | 2026-11-02 through 2026-11-08 | Official week |
| Week 11 | 2026-11-09 through 2026-11-15 | Official week |
| Week 12 | 2026-11-16 through 2026-11-22 | Official week |
| Week 13 | 2026-11-23 through 2026-11-29 | Official week |
| Week 14 | 2026-11-30 through 2026-12-06 | Official week |
| Week 15 | 2026-12-07 through 2026-12-13 | Official week |

## Reporting Boundary

- P4-L closure is infrastructure/provenance evidence only.
- Long-term research vision is recorded at
  `docs/CONTRAMAMBA_RESEARCH_VISION.md`, frozen at
  `bca6db6de2e1bb5d1b81188b61b2023be20eadd3`, with status
  `LONG-TERM RESEARCH VISION / NON-AUTHORITY`; it does not authorize
  implementation, training, evaluation, promotion, Kaggle execution, or
  scientific claims.
- A pre-start frozen P3-W7-A0 authority candidate exists at
  `reports/reason_router_p3w7_a0_current_lineage_execution_authority_spec_candidate.md`.
  It was materialized from basis
  `bca6db6de2e1bb5d1b81188b61b2023be20eadd3`, independently verified `PASS`,
  and frozen at `ecda9707cc054ec26428b3f0937be8829f754f1b` during Week 0
  before the intended formal Week 1 boundary.
- The pre-start frozen candidate is retained as immutable provenance and a
  verified draft basis only; it must not be consumed for A0 execution.
- Formal Week 1 consumable P3-W7-A0 execution authority is
  `NOT_ESTABLISHED`.
- A0 execution is `NOT_STARTED`.
- No formal training, evaluation, A0 run, promotion, or scientific execution has
  occurred.
- No post-freeze local gate, per-seed preflight, Kaggle A0 execution, or trainer
  launch occurred. Seeds `180`, `181`, and `182` have consumed `ZERO`
  authorized trainer attempts. A1/A2/A3 remain unauthorized.
- The next formal Week 1 session must start with `cm context`, use current
  repository state and applicable authority ordering, treat
  `ecda9707cc054ec26428b3f0937be8829f754f1b` as pre-start
  provenance/verified draft basis only, and create, independently verify, and
  freeze a new formal-start P3-W7-A0 execution authority at the then-current
  HEAD. Only after that new authority and its required gates/preflights may A0
  execution be considered.
- P4-L remains `CLOSED` unless an actual new failure is observed.
