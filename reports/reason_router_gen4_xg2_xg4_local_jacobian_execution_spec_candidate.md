# Gen4-K XG2/XG4 Unit-Direction Local Jacobian — Execution Freeze

Implementation HEAD:

`363eb95967a827ae611e074866abbb0b5b7aaaab`

Implementation scope:

`3a9dd04b3de398069eed3844dfed719e2ba36ca7`

## Authorized execution

This authority permits exactly one scientific local-Jacobian execution over each frozen family:

- XG2 pairs 301..600
- XG4 pairs 301..600

Use the frozen Phase-1 `alignment_delta_h` plans and normalize each pair direction to unit L2 norm.

Fixed branch-wise radii:

- epsilon = 0.025
- epsilon = 0.05

For each pair and epsilon execute the forward and reverse symmetric probes defined by the frozen implementation scope.

Scientific forward budget:

- 4 forwards per pair per epsilon
- 8 forwards per pair total
- 2400 new scientific forwards per family
- 4800 new scientific forwards across XG2 and XG4
- 0 new baseline model forwards

The frozen Phase-1 `delta_baseline` is the only allowed F(0) value.

## Execution outputs

The execution may persist only the runner-defined raw local-Jacobian artifacts, including:

- F(+epsilon)
- F(-epsilon)
- J_epsilon
- K_epsilon
- intervention audit/provenance
- manifest/checksum/summary fields required by the implementation

The execution summary must retain `scientific_conclusion = null` or the runner-equivalent no-conclusion boundary.

## Explicitly prohibited during execution

- H_local hypothesis tests
- p-values
- Holm correction
- final replication decision
- threshold optimization
- subgroup or tail rescue
- PCA-based selection
- epsilon values other than 0.025 and 0.05
- new baseline model forwards
- training or backward
- task heads or logits
- layer, endpoint, offset, channel, checkpoint, or direction search

## Post-execution boundary

Successful execution establishes only that the fixed local-Jacobian measurements were obtained.

After both families complete:

1. collect/import the exact artifacts;
2. validate provenance and frozen identities;
3. freeze the imported artifacts;
4. only then perform the pre-specified read-only confirmatory analysis of J_0.025 with Holm correction across XG2 and XG4 and the descriptive epsilon=0.05 scale-consistency audit.

No scientific conclusion is authorized before that sequence is complete.
