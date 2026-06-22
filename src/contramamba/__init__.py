"""ContraMamba-v5 package."""

from .modeling_v5 import ContraMambaV5
from .labels import FinalLabel, InterventionType, PolarityLabel, PrimaryFailureType
from .losses import intervention_pairwise_losses
from .heads import (
    FinalEntitlementDecisionHead,
    FrameGate,
    PolarityEnergyHead,
    PredicateCoverageHead,
    SufficiencyGate,
)

__all__ = [
    "ContraMambaV5",
    "FrameGate",
    "PredicateCoverageHead",
    "SufficiencyGate",
    "PolarityEnergyHead",
    "FinalEntitlementDecisionHead",
    "FinalLabel",
    "PolarityLabel",
    "InterventionType",
    "PrimaryFailureType",
    "intervention_pairwise_losses",
]
