# Stage191-A deterministic replay required closure

Decision: `STAGE191A_DETERMINISTIC_REPLAY_REQUIRED`.

The authoritative input is exactly `stage191a_trajectory_feasibility_report.json`. Stage191-A has no identity, closure, artifact-validity, or general blocking reasons, and its identity-and-closure gates passed. Across all six Stage189 runs, the authoritative 20-epoch compatible-positive margin history is complete. The 20-epoch clean macro-F1 history and the sparse historical prediction distributions, normalized to `REFUTE`, `NOT_ENTITLED`, and `SUPPORT`, are also complete.

The only absent clean trajectory metrics are `support_recall`, `false_entitlement_total`, `polarity_error_total`, and `clean_dev_ce`. Epoch-addressable model state is absent for the full trajectory for epochs 1 through 20, so the six histories cannot support epoch-by-epoch state analysis without deterministic replay.

A Stage191-B replay specification is required and authorized. This closure does not authorize replay execution, training, or model advancement; all three remain false.
