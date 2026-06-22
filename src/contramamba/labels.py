"""Stable labels and intervention vocabulary for v5 datasets."""

from enum import Enum, IntEnum


class FinalLabel(IntEnum):
    REFUTE = 0
    NOT_ENTITLED = 1
    SUPPORT = 2


class PolarityLabel(IntEnum):
    NONE = 0
    REFUTE = 1
    SUPPORT = 2


class InterventionType(str, Enum):
    NONE = "none"
    PARAPHRASE = "paraphrase"
    ENTITY_SWAP = "entity_swap"
    EVENT_SWAP = "event_swap"
    TIME_SWAP = "time_swap"
    LOCATION_SWAP = "location_swap"
    ROLE_SWAP = "role_swap"
    TITLE_NAME_SWAP = "title_name_swap"
    PREDICATE_SWAP = "predicate_swap"
    EVIDENCE_DELETION = "evidence_deletion"
    EVIDENCE_TRUNCATION = "evidence_truncation"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"
    POLARITY_FLIP = "polarity_flip"


class PrimaryFailureType(str, Enum):
    NONE = "none"
    FRAME = "frame"
    PREDICATE = "predicate"
    SUFFICIENCY = "sufficiency"
    POLARITY = "polarity"

