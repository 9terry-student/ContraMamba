"""Architecture-aware scalar export schema helpers for Stage167-A."""

from __future__ import annotations

COMMON_NATIVE_SCALAR_FIELDS = (
    "frame_prob",
    "predicate_coverage_prob",
    "sufficiency_prob",
    "entitlement_prob",
    "polarity_margin",
    "positive_energy",
    "negative_energy",
)

VNEXT_ONLY_SCALAR_FIELDS = (
    "compositional_entitlement_prob",
    "learned_entitlement_prob",
    "learned_entitlement_logit",
)

REQUIRED_SCALARS_BY_ARCHITECTURE = {
    "v6b_minimal": COMMON_NATIVE_SCALAR_FIELDS,
    "vnext_minimal": (
        "frame_prob",
        "predicate_coverage_prob",
        "sufficiency_prob",
        "entitlement_prob",
        "compositional_entitlement_prob",
        "learned_entitlement_prob",
        "learned_entitlement_logit",
        "polarity_margin",
        "positive_energy",
        "negative_energy",
    ),
}

ALL_KNOWN_SCALAR_FIELDS = tuple(
    dict.fromkeys(
        field
        for fields in REQUIRED_SCALARS_BY_ARCHITECTURE.values()
        for field in fields
    )
)

SCALAR_SCHEMA_NAME = "architecture_aware_native_scalar_schema_v1"


def required_scalars_for_architecture(architecture: str | None) -> tuple[str, ...]:
    return tuple(REQUIRED_SCALARS_BY_ARCHITECTURE.get(architecture or "", ALL_KNOWN_SCALAR_FIELDS))


def unsupported_scalars_for_architecture(architecture: str | None) -> tuple[str, ...]:
    if architecture == "v6b_minimal":
        return VNEXT_ONLY_SCALAR_FIELDS
    return ()


def optional_scalars_for_architecture(architecture: str | None) -> tuple[str, ...]:
    required = set(required_scalars_for_architecture(architecture))
    unsupported = set(unsupported_scalars_for_architecture(architecture))
    return tuple(
        field
        for field in ALL_KNOWN_SCALAR_FIELDS
        if field not in required and field not in unsupported
    )