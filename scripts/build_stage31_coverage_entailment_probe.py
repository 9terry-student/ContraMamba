"""Stage31-A: Coverage/Entailment Diagnostic Probe Generator.

Generates a deterministic 200-row diagnostic probe dataset that isolates
directional entailment and coverage failures while keeping frame/entity/
relation mostly stable.

This is NOT a training dataset. It is a diagnostic probe for architecture
analysis of the Coverage/Entailment axis in the target architecture:
    Mamba Encoder -> Hard Core Validity -> Coverage/Entailment -> ...

Usage:
    python scripts/build_stage31_coverage_entailment_probe.py

Outputs:
    data/stage31_coverage_entailment_probe.jsonl
    reports/stage31_coverage_entailment_probe_report.json
    reports/stage31_coverage_entailment_probe_report.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Label convention (matches project-wide mapping)
# ---------------------------------------------------------------------------
LABEL_TO_GOLD: dict[str, int] = {
    "REFUTE": 0,
    "NOT_ENTITLED": 1,
    "SUPPORT": 2,
}

GOLD_TO_LABEL = {v: k for k, v in LABEL_TO_GOLD.items()}

POLARITY_MAP: dict[str, str] = {
    "SUPPORT": "support",
    "NOT_ENTITLED": "none",
    "REFUTE": "refute",
}

COMPAT_POLARITY_LABEL: dict[str, str] = {
    "SUPPORT": "SUPPORT",
    "NOT_ENTITLED": "NONE",
    "REFUTE": "REFUTE",
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "stage31_coverage_entailment_probe.jsonl"
REPORT_JSON_PATH = ROOT / "reports" / "stage31_coverage_entailment_probe_report.json"
REPORT_MD_PATH = ROOT / "reports" / "stage31_coverage_entailment_probe_report.md"

# ---------------------------------------------------------------------------
# Example builders — each returns a list of dicts (no "id" yet)
# ---------------------------------------------------------------------------

def _row(
    group: str,
    claim: str,
    evidence: str,
    label: str,
    coverage_relation: str,
    hard_core_should_pass: bool,
    notes: str,
) -> dict[str, Any]:
    gold = LABEL_TO_GOLD[label]
    return {
        "claim": claim,
        "evidence": evidence,
        "label": label,
        "gold": gold,
        "label_id": gold,
        "final_label": label,
        "group": group,
        "coverage_relation": coverage_relation,
        "expected_owner": "coverage_entailment",
        "hard_core_should_pass": hard_core_should_pass,
        "polarity_should_be": POLARITY_MAP[label],
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Group 1: all_to_some_support  (SUPPORT)
# Evidence: ALL X → Claim: SOME X
# ---------------------------------------------------------------------------
def build_all_to_some_support() -> list[dict]:
    G = "all_to_some_support"
    CR = "all_entails_some"
    NOTE = "All entails some; claim is weaker than evidence."
    rows = []

    templates = [
        ("All participants in the workshop received a certificate.",
         "Some participants in the workshop received a certificate."),
        ("All delegates at the summit signed the declaration.",
         "Some delegates at the summit signed the declaration."),
        ("All patients in the trial showed improvement.",
         "Some patients in the trial showed improvement."),
        ("All students in the programme passed the final exam.",
         "Some students in the programme passed the final exam."),
        ("All sensors in the array detected the signal.",
         "Some sensors in the array detected the signal."),
        ("All team members completed the safety training.",
         "Some team members completed the safety training."),
        ("All branches of the bank adopted the new software.",
         "Some branches of the bank adopted the new software."),
        ("All respondents in the survey reported satisfaction.",
         "Some respondents in the survey reported satisfaction."),
        ("All devices in the lab were calibrated before the test.",
         "Some devices in the lab were calibrated before the test."),
        ("All articles in the journal were peer-reviewed.",
         "Some articles in the journal were peer-reviewed."),
        ("All employees at the facility wore protective gear.",
         "Some employees at the facility wore protective gear."),
        ("All villages in the district received aid packages.",
         "Some villages in the district received aid packages."),
        ("All flights from the terminal were delayed.",
         "Some flights from the terminal were delayed."),
        ("All servers in the cluster were updated overnight.",
         "Some servers in the cluster were updated overnight."),
        ("All speakers at the conference were introduced by the chair.",
         "Some speakers at the conference were introduced by the chair."),
        ("All children at the event received a gift.",
         "Some children at the event received a gift."),
        ("All proposals submitted before noon were reviewed.",
         "Some proposals submitted before noon were reviewed."),
        ("All volunteers at the site wore identification badges.",
         "Some volunteers at the site wore identification badges."),
        ("All units in the complex passed the fire inspection.",
         "Some units in the complex passed the fire inspection."),
        ("All coaches in the league attended the annual meeting.",
         "Some coaches in the league attended the annual meeting."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "SUPPORT", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 2: some_to_all_not_entitled  (NOT_ENTITLED)
# Evidence: SOME X → Claim: ALL X
# ---------------------------------------------------------------------------
def build_some_to_all_not_entitled() -> list[dict]:
    G = "some_to_all_not_entitled"
    CR = "some_does_not_entail_all"
    NOTE = "Some does not entail all; claim overgeneralises."
    rows = []

    templates = [
        ("Some participants in the workshop received a certificate.",
         "All participants in the workshop received a certificate."),
        ("Some delegates at the summit signed the declaration.",
         "All delegates at the summit signed the declaration."),
        ("Some patients in the trial showed improvement.",
         "All patients in the trial showed improvement."),
        ("Some students in the programme passed the final exam.",
         "All students in the programme passed the final exam."),
        ("Some sensors in the array detected the signal.",
         "All sensors in the array detected the signal."),
        ("Some team members completed the safety training.",
         "All team members completed the safety training."),
        ("Some branches of the bank adopted the new software.",
         "All branches of the bank adopted the new software."),
        ("Some respondents in the survey reported satisfaction.",
         "All respondents in the survey reported satisfaction."),
        ("Some devices in the lab were calibrated before the test.",
         "All devices in the lab were calibrated before the test."),
        ("Some articles in the journal were peer-reviewed.",
         "All articles in the journal were peer-reviewed."),
        ("Some employees at the facility wore protective gear.",
         "All employees at the facility wore protective gear."),
        ("Some villages in the district received aid packages.",
         "All villages in the district received aid packages."),
        ("Some flights from the terminal were delayed.",
         "All flights from the terminal were delayed."),
        ("Some servers in the cluster were updated overnight.",
         "All servers in the cluster were updated overnight."),
        ("Some speakers at the conference were introduced by the chair.",
         "All speakers at the conference were introduced by the chair."),
        ("Some children at the event received a gift.",
         "All children at the event received a gift."),
        ("Some proposals submitted before noon were reviewed.",
         "All proposals submitted before noon were reviewed."),
        ("Some volunteers at the site wore identification badges.",
         "All volunteers at the site wore identification badges."),
        ("Some units in the complex passed the fire inspection.",
         "All units in the complex passed the fire inspection."),
        ("Some coaches in the league attended the annual meeting.",
         "All coaches in the league attended the annual meeting."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "NOT_ENTITLED", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 3: specific_to_general_support  (SUPPORT)
# Evidence: more specific fact → Claim: general fact (weaker claim)
# ---------------------------------------------------------------------------
def build_specific_to_general_support() -> list[dict]:
    G = "specific_to_general_support"
    CR = "specific_entails_general"
    NOTE = "Specific evidence entails the weaker general claim."
    rows = []

    templates = [
        ("The Vega team won the championship final.",
         "The Vega team won a match."),
        ("Dr Lena Park published a paper in a peer-reviewed journal.",
         "Dr Lena Park published a paper."),
        ("The committee approved the budget proposal unanimously.",
         "The committee approved a proposal."),
        ("The software update fixed a critical security vulnerability.",
         "The software update fixed an issue."),
        ("Clara Osei scored the highest mark in the advanced module.",
         "Clara Osei scored a mark in the module."),
        ("The Raven bridge was completed ahead of the original deadline.",
         "The Raven bridge was completed."),
        ("The investigation concluded with a formal written report.",
         "The investigation concluded."),
        ("Agent Torres recovered the stolen documents from the vault.",
         "Agent Torres recovered the documents."),
        ("The merger was finalised after eighteen months of negotiation.",
         "The merger was finalised."),
        ("The reactor reached full operational capacity on the third day.",
         "The reactor reached operational capacity."),
        ("The athlete broke the national record in the 400-metre sprint.",
         "The athlete broke a record."),
        ("The vaccine was approved for use in children under five.",
         "The vaccine was approved for use."),
        ("The satellite successfully transmitted data to the ground station.",
         "The satellite transmitted data."),
        ("The expedition reached the summit via the northern ridge route.",
         "The expedition reached the summit."),
        ("The contract was signed by the director of operations.",
         "The contract was signed."),
        ("The quarterly report showed a twenty percent increase in revenue.",
         "The quarterly report showed an increase in revenue."),
        ("The prototype was tested under extreme temperature conditions.",
         "The prototype was tested."),
        ("The policy was extended to cover permanent part-time workers.",
         "The policy was extended."),
        ("The museum acquired the painting through a private auction.",
         "The museum acquired the painting."),
        ("The team resolved the server outage within two hours.",
         "The team resolved the server outage."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "SUPPORT", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 4: general_to_specific_not_entitled  (NOT_ENTITLED)
# Evidence: general fact → Claim: specific fact
# ---------------------------------------------------------------------------
def build_general_to_specific_not_entitled() -> list[dict]:
    G = "general_to_specific_not_entitled"
    CR = "general_does_not_entail_specific"
    NOTE = "General evidence does not entail a more specific claim."
    rows = []

    templates = [
        ("The Vega team won a match.",
         "The Vega team won the championship final."),
        ("Dr Lena Park published a paper.",
         "Dr Lena Park published a paper in a peer-reviewed journal."),
        ("The committee approved a proposal.",
         "The committee approved the budget proposal unanimously."),
        ("The software update fixed an issue.",
         "The software update fixed a critical security vulnerability."),
        ("Clara Osei scored a mark in the module.",
         "Clara Osei scored the highest mark in the advanced module."),
        ("The Raven bridge was completed.",
         "The Raven bridge was completed ahead of the original deadline."),
        ("The investigation concluded.",
         "The investigation concluded with a formal written report."),
        ("Agent Torres recovered the documents.",
         "Agent Torres recovered the stolen documents from the vault."),
        ("The merger was finalised.",
         "The merger was finalised after eighteen months of negotiation."),
        ("The reactor reached operational capacity.",
         "The reactor reached full operational capacity on the third day."),
        ("The athlete broke a record.",
         "The athlete broke the national record in the 400-metre sprint."),
        ("The vaccine was approved for use.",
         "The vaccine was approved for use in children under five."),
        ("The satellite transmitted data.",
         "The satellite successfully transmitted data to the ground station."),
        ("The expedition reached the summit.",
         "The expedition reached the summit via the northern ridge route."),
        ("The contract was signed.",
         "The contract was signed by the director of operations."),
        ("The quarterly report showed an increase in revenue.",
         "The quarterly report showed a twenty percent increase in revenue."),
        ("The prototype was tested.",
         "The prototype was tested under extreme temperature conditions."),
        ("The policy was extended.",
         "The policy was extended to cover permanent part-time workers."),
        ("The museum acquired the painting.",
         "The museum acquired the painting through a private auction."),
        ("The team resolved the server outage.",
         "The team resolved the server outage within two hours."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "NOT_ENTITLED", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 5: only_to_base_support  (SUPPORT)
# Evidence: X was the only winner → Claim: X was a winner
# ---------------------------------------------------------------------------
def build_only_to_base_support() -> list[dict]:
    G = "only_to_base_support"
    CR = "only_entails_base_membership"
    NOTE = "Being the only member of a set entails being a member."
    rows = []

    templates = [
        ("Mia Chen was the only finalist selected by the panel.",
         "Mia Chen was a finalist selected by the panel."),
        ("Team Orion was the only group to complete all three stages.",
         "Team Orion was a group that completed all three stages."),
        ("Professor Hart was the only keynote speaker at the symposium.",
         "Professor Hart was a keynote speaker at the symposium."),
        ("The Altair project was the only initiative approved that year.",
         "The Altair project was an initiative approved that year."),
        ("Dara Walsh was the only candidate endorsed by the board.",
         "Dara Walsh was a candidate endorsed by the board."),
        ("Station Seven was the only facility to pass the inspection.",
         "Station Seven was a facility that passed the inspection."),
        ("The Helix model was the only design submitted on time.",
         "The Helix model was a design submitted on time."),
        ("Agent Solis was the only officer present at the handover.",
         "Agent Solis was an officer present at the handover."),
        ("The Northern route was the only path cleared for transit.",
         "The Northern route was a path cleared for transit."),
        ("Dr Ama Osei was the only researcher to replicate the result.",
         "Dr Ama Osei was a researcher who replicated the result."),
        ("Unit Four was the only unit to exceed the production target.",
         "Unit Four was a unit that exceeded the production target."),
        ("The Crestview bid was the only offer accepted by the seller.",
         "The Crestview bid was an offer accepted by the seller."),
        ("River Lane was the only street included in the pilot programme.",
         "River Lane was a street included in the pilot programme."),
        ("Nina Torres was the only author credited on the final report.",
         "Nina Torres was an author credited on the final report."),
        ("Block C was the only building evacuated during the drill.",
         "Block C was a building evacuated during the drill."),
        ("Sensor Grid Alpha was the only array to record the anomaly.",
         "Sensor Grid Alpha was an array that recorded the anomaly."),
        ("The Falcon module was the only component replaced in the upgrade.",
         "The Falcon module was a component replaced in the upgrade."),
        ("Chief Vance was the only official to attend both sessions.",
         "Chief Vance was an official who attended both sessions."),
        ("The third proposal was the only one that met all criteria.",
         "The third proposal was one that met all criteria."),
        ("Lab B was the only lab certified for the procedure.",
         "Lab B was a lab certified for the procedure."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "SUPPORT", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 6: also_to_only_not_entitled  (NOT_ENTITLED)
# Evidence: X was also a winner → Claim: X was the only winner
# ---------------------------------------------------------------------------
def build_also_to_only_not_entitled() -> list[dict]:
    G = "also_to_only_not_entitled"
    CR = "also_does_not_entail_only"
    NOTE = "'Also a member' does not entail 'only member'."
    rows = []

    templates = [
        ("Mia Chen was also a finalist selected by the panel.",
         "Mia Chen was the only finalist selected by the panel."),
        ("Team Orion was also a group that completed all three stages.",
         "Team Orion was the only group to complete all three stages."),
        ("Professor Hart was also a keynote speaker at the symposium.",
         "Professor Hart was the only keynote speaker at the symposium."),
        ("The Altair project was also an initiative approved that year.",
         "The Altair project was the only initiative approved that year."),
        ("Dara Walsh was also a candidate endorsed by the board.",
         "Dara Walsh was the only candidate endorsed by the board."),
        ("Station Seven was also a facility that passed the inspection.",
         "Station Seven was the only facility to pass the inspection."),
        ("The Helix model was also a design submitted on time.",
         "The Helix model was the only design submitted on time."),
        ("Agent Solis was also an officer present at the handover.",
         "Agent Solis was the only officer present at the handover."),
        ("The Northern route was also a path cleared for transit.",
         "The Northern route was the only path cleared for transit."),
        ("Dr Ama Osei was also a researcher who replicated the result.",
         "Dr Ama Osei was the only researcher to replicate the result."),
        ("Unit Four was also a unit that exceeded the production target.",
         "Unit Four was the only unit to exceed the production target."),
        ("The Crestview bid was also an offer accepted by the seller.",
         "The Crestview bid was the only offer accepted by the seller."),
        ("River Lane was also a street included in the pilot programme.",
         "River Lane was the only street included in the pilot programme."),
        ("Nina Torres was also an author credited on the final report.",
         "Nina Torres was the only author credited on the final report."),
        ("Block C was also a building evacuated during the drill.",
         "Block C was the only building evacuated during the drill."),
        ("Sensor Grid Alpha was also an array that recorded the anomaly.",
         "Sensor Grid Alpha was the only array to record the anomaly."),
        ("The Falcon module was also a component replaced in the upgrade.",
         "The Falcon module was the only component replaced in the upgrade."),
        ("Chief Vance was also an official who attended both sessions.",
         "Chief Vance was the only official to attend both sessions."),
        ("The third proposal was also one that met all criteria.",
         "The third proposal was the only one that met all criteria."),
        ("Lab B was also a lab certified for the procedure.",
         "Lab B was the only lab certified for the procedure."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "NOT_ENTITLED", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 7: whole_to_part_support  (SUPPORT)
# Evidence: whole set → Claim: known included subset
# ---------------------------------------------------------------------------
def build_whole_to_part_support() -> list[dict]:
    G = "whole_to_part_support"
    CR = "whole_entails_included_part"
    NOTE = "Property of whole set entails property of its known included subset."
    rows = []

    templates = [
        ("The new policy affected all employees at the company.",
         "The new policy affected the engineers at the company.",
         "engineers are employees"),
        ("The grant covered all research projects in the department.",
         "The grant covered the biology projects in the department.",
         "biology projects are research projects"),
        ("The outage disrupted all services on the platform.",
         "The outage disrupted the payment service on the platform.",
         "payment service is a platform service"),
        ("The regulation applied to all vehicles on the road.",
         "The regulation applied to trucks on the road.",
         "trucks are vehicles"),
        ("The discount was offered to all members of the club.",
         "The discount was offered to senior members of the club.",
         "senior members are club members"),
        ("The recall affected all models in the product line.",
         "The recall affected the Mark II model in the product line.",
         "Mark II is a model in the line"),
        ("The inspection covered all floors of the building.",
         "The inspection covered the third floor of the building.",
         "third floor is a floor of the building"),
        ("The festival hosted all performing arts groups in the region.",
         "The festival hosted the regional dance troupe.",
         "dance troupe is a performing arts group"),
        ("The audit reviewed all financial records for the fiscal year.",
         "The audit reviewed the expense reports for the fiscal year.",
         "expense reports are financial records"),
        ("The directive required all staff to complete the declaration.",
         "The directive required contract staff to complete the declaration.",
         "contract staff are staff"),
        ("The closure affected all access roads to the site.",
         "The closure affected the eastern access road to the site.",
         "eastern road is an access road"),
        ("The training programme was mandatory for all new hires.",
         "The training programme was mandatory for new hires in operations.",
         "operations hires are new hires"),
        ("The prize was awarded to all top-ranked competitors.",
         "The prize was awarded to the top-ranked competitor from Zone A.",
         "Zone A competitor is a top-ranked competitor"),
        ("The announcement reached all subscribers to the newsletter.",
         "The announcement reached premium subscribers to the newsletter.",
         "premium subscribers are newsletter subscribers"),
        ("The upgrade was deployed to all nodes in the network.",
         "The upgrade was deployed to the gateway nodes in the network.",
         "gateway nodes are network nodes"),
        ("The clause applied to all contracts signed after the merger.",
         "The clause applied to service contracts signed after the merger.",
         "service contracts are contracts"),
        ("The benefit extended to all registered participants.",
         "The benefit extended to registered participants from abroad.",
         "participants from abroad are registered participants"),
        ("The warning was issued to all residents in the zone.",
         "The warning was issued to residents in the northern sector of the zone.",
         "northern sector residents are zone residents"),
        ("The moratorium covered all construction projects in the area.",
         "The moratorium covered residential construction projects in the area.",
         "residential projects are construction projects"),
        ("The curfew applied to all persons under the travel restriction.",
         "The curfew applied to foreign nationals under the travel restriction.",
         "foreign nationals are persons under restriction"),
    ]
    for ev, cl, why in templates:
        rows.append(_row(G, cl, ev, "SUPPORT", CR, True,
                         f"Subset inclusion: {why}. {NOTE}"))
    return rows


# ---------------------------------------------------------------------------
# Group 8: part_to_whole_not_entitled  (NOT_ENTITLED)
# Evidence: subset → Claim: whole set
# ---------------------------------------------------------------------------
def build_part_to_whole_not_entitled() -> list[dict]:
    G = "part_to_whole_not_entitled"
    CR = "part_does_not_entail_whole"
    NOTE = "Property of subset does not entail property of the whole set."
    rows = []

    templates = [
        ("The new policy affected the engineers at the company.",
         "The new policy affected all employees at the company."),
        ("The grant covered the biology projects in the department.",
         "The grant covered all research projects in the department."),
        ("The outage disrupted the payment service on the platform.",
         "The outage disrupted all services on the platform."),
        ("The regulation applied to trucks on the road.",
         "The regulation applied to all vehicles on the road."),
        ("The discount was offered to senior members of the club.",
         "The discount was offered to all members of the club."),
        ("The recall affected the Mark II model in the product line.",
         "The recall affected all models in the product line."),
        ("The inspection covered the third floor of the building.",
         "The inspection covered all floors of the building."),
        ("The festival hosted the regional dance troupe.",
         "The festival hosted all performing arts groups in the region."),
        ("The audit reviewed the expense reports for the fiscal year.",
         "The audit reviewed all financial records for the fiscal year."),
        ("The directive required contract staff to complete the declaration.",
         "The directive required all staff to complete the declaration."),
        ("The closure affected the eastern access road to the site.",
         "The closure affected all access roads to the site."),
        ("The training programme was mandatory for new hires in operations.",
         "The training programme was mandatory for all new hires."),
        ("The prize was awarded to the top-ranked competitor from Zone A.",
         "The prize was awarded to all top-ranked competitors."),
        ("The announcement reached premium subscribers to the newsletter.",
         "The announcement reached all subscribers to the newsletter."),
        ("The upgrade was deployed to the gateway nodes in the network.",
         "The upgrade was deployed to all nodes in the network."),
        ("The clause applied to service contracts signed after the merger.",
         "The clause applied to all contracts signed after the merger."),
        ("The benefit extended to registered participants from abroad.",
         "The benefit extended to all registered participants."),
        ("The warning was issued to residents in the northern sector of the zone.",
         "The warning was issued to all residents in the zone."),
        ("The moratorium covered residential construction projects in the area.",
         "The moratorium covered all construction projects in the area."),
        ("The curfew applied to foreign nationals under the travel restriction.",
         "The curfew applied to all persons under the travel restriction."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "NOT_ENTITLED", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 9: none_to_some_refute  (REFUTE)
# Evidence: NO X → Claim: SOME X  → contradiction
# ---------------------------------------------------------------------------
def build_none_to_some_refute() -> list[dict]:
    G = "none_to_some_refute"
    CR = "none_contradicts_some"
    NOTE = "'No X' directly contradicts 'some X'."
    rows = []

    templates = [
        ("No participants in the workshop received a certificate.",
         "Some participants in the workshop received a certificate."),
        ("No delegates at the summit signed the declaration.",
         "Some delegates at the summit signed the declaration."),
        ("No patients in the trial showed improvement.",
         "Some patients in the trial showed improvement."),
        ("No students in the programme passed the final exam.",
         "Some students in the programme passed the final exam."),
        ("No sensors in the array detected the signal.",
         "Some sensors in the array detected the signal."),
        ("No team members completed the safety training.",
         "Some team members completed the safety training."),
        ("No branches of the bank adopted the new software.",
         "Some branches of the bank adopted the new software."),
        ("No respondents in the survey reported satisfaction.",
         "Some respondents in the survey reported satisfaction."),
        ("No devices in the lab were calibrated before the test.",
         "Some devices in the lab were calibrated before the test."),
        ("No articles in the journal were peer-reviewed.",
         "Some articles in the journal were peer-reviewed."),
        ("No employees at the facility wore protective gear.",
         "Some employees at the facility wore protective gear."),
        ("No villages in the district received aid packages.",
         "Some villages in the district received aid packages."),
        ("No flights from the terminal were delayed.",
         "Some flights from the terminal were delayed."),
        ("No servers in the cluster were updated overnight.",
         "Some servers in the cluster were updated overnight."),
        ("No speakers at the conference were introduced by the chair.",
         "Some speakers at the conference were introduced by the chair."),
        ("No children at the event received a gift.",
         "Some children at the event received a gift."),
        ("No proposals submitted before noon were reviewed.",
         "Some proposals submitted before noon were reviewed."),
        ("No volunteers at the site wore identification badges.",
         "Some volunteers at the site wore identification badges."),
        ("No units in the complex passed the fire inspection.",
         "Some units in the complex passed the fire inspection."),
        ("No coaches in the league attended the annual meeting.",
         "Some coaches in the league attended the annual meeting."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "REFUTE", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Group 10: some_to_none_refute  (REFUTE)
# Evidence: SOME X → Claim: NO X  → contradiction
# ---------------------------------------------------------------------------
def build_some_to_none_refute() -> list[dict]:
    G = "some_to_none_refute"
    CR = "some_contradicts_none"
    NOTE = "'Some X' directly contradicts 'no X'."
    rows = []

    templates = [
        ("Some participants in the workshop received a certificate.",
         "No participants in the workshop received a certificate."),
        ("Some delegates at the summit signed the declaration.",
         "No delegates at the summit signed the declaration."),
        ("Some patients in the trial showed improvement.",
         "No patients in the trial showed improvement."),
        ("Some students in the programme passed the final exam.",
         "No students in the programme passed the final exam."),
        ("Some sensors in the array detected the signal.",
         "No sensors in the array detected the signal."),
        ("Some team members completed the safety training.",
         "No team members completed the safety training."),
        ("Some branches of the bank adopted the new software.",
         "No branches of the bank adopted the new software."),
        ("Some respondents in the survey reported satisfaction.",
         "No respondents in the survey reported satisfaction."),
        ("Some devices in the lab were calibrated before the test.",
         "No devices in the lab were calibrated before the test."),
        ("Some articles in the journal were peer-reviewed.",
         "No articles in the journal were peer-reviewed."),
        ("Some employees at the facility wore protective gear.",
         "No employees at the facility wore protective gear."),
        ("Some villages in the district received aid packages.",
         "No villages in the district received aid packages."),
        ("Some flights from the terminal were delayed.",
         "No flights from the terminal were delayed."),
        ("Some servers in the cluster were updated overnight.",
         "No servers in the cluster were updated overnight."),
        ("Some speakers at the conference were introduced by the chair.",
         "No speakers at the conference were introduced by the chair."),
        ("Some children at the event received a gift.",
         "No children at the event received a gift."),
        ("Some proposals submitted before noon were reviewed.",
         "No proposals submitted before noon were reviewed."),
        ("Some volunteers at the site wore identification badges.",
         "No volunteers at the site wore identification badges."),
        ("Some units in the complex passed the fire inspection.",
         "No units in the complex passed the fire inspection."),
        ("Some coaches in the league attended the annual meeting.",
         "No coaches in the league attended the annual meeting."),
    ]
    for ev, cl in templates:
        rows.append(_row(G, cl, ev, "REFUTE", CR, True, NOTE))
    return rows


# ---------------------------------------------------------------------------
# Assemble dataset
# ---------------------------------------------------------------------------

GROUP_BUILDERS = [
    build_all_to_some_support,
    build_some_to_all_not_entitled,
    build_specific_to_general_support,
    build_general_to_specific_not_entitled,
    build_only_to_base_support,
    build_also_to_only_not_entitled,
    build_whole_to_part_support,
    build_part_to_whole_not_entitled,
    build_none_to_some_refute,
    build_some_to_none_refute,
]

EXPECTED_GROUPS = [
    "all_to_some_support",
    "some_to_all_not_entitled",
    "specific_to_general_support",
    "general_to_specific_not_entitled",
    "only_to_base_support",
    "also_to_only_not_entitled",
    "whole_to_part_support",
    "part_to_whole_not_entitled",
    "none_to_some_refute",
    "some_to_none_refute",
]

EXAMPLES_PER_GROUP = 20


def build_dataset() -> list[dict]:
    rows: list[dict] = []
    for builder in GROUP_BUILDERS:
        group_rows = builder()
        group_name = group_rows[0]["group"]
        for idx, row in enumerate(group_rows):
            row_id = f"stage31_{group_name}_{idx:02d}"
            row["id"] = row_id
            row["pair_id"] = row_id
            label = row["label"]
            is_sufficient = 0 if label == "NOT_ENTITLED" else 1
            row["frame_compatible_label"] = 1
            row["predicate_covered_label"] = 1
            row["sufficiency_label"] = is_sufficient
            row["evidence_sufficient_label"] = is_sufficient
            row["polarity_label"] = COMPAT_POLARITY_LABEL[label]
            row["intervention_type"] = row["group"]
            row["probe_type"] = row["group"]
        rows.extend(group_rows)
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "id", "pair_id", "claim", "evidence", "label", "gold", "label_id",
    "final_label", "group",
    "coverage_relation", "expected_owner", "hard_core_should_pass",
    "polarity_should_be", "notes", "frame_compatible_label",
    "predicate_covered_label", "sufficiency_label",
    "evidence_sufficient_label", "polarity_label", "intervention_type",
    "probe_type",
}


def validate(rows: list[dict]) -> list[str]:
    errors: list[str] = []

    # Unique IDs
    ids = [r["id"] for r in rows]
    duplicates = [id_ for id_, cnt in Counter(ids).items() if cnt > 1]
    if duplicates:
        errors.append(f"Duplicate IDs: {duplicates}")

    pair_ids = [r["pair_id"] for r in rows]
    duplicate_pair_ids = [id_ for id_, cnt in Counter(pair_ids).items() if cnt > 1]
    if duplicate_pair_ids:
        errors.append(f"Duplicate pair_ids: {duplicate_pair_ids}")
    for row in rows:
        if row.get("pair_id") != row.get("id"):
            errors.append(
                f"Row {row.get('id', '?')} pair_id must match id for Stage31-A2."
            )

    # Required fields
    for row in rows:
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            errors.append(f"Row {row.get('id', '?')} missing fields: {missing}")

    # Label/gold consistency
    for row in rows:
        expected_gold = LABEL_TO_GOLD.get(row["label"])
        if expected_gold is None:
            errors.append(f"Row {row['id']} unknown label: {row['label']}")
        elif row["gold"] != expected_gold:
            errors.append(
                f"Row {row['id']} label/gold mismatch: "
                f"label={row['label']} gold={row['gold']} expected_gold={expected_gold}"
            )
        if row.get("label_id") != expected_gold:
            errors.append(
                f"Row {row['id']} label/label_id mismatch: "
                f"label={row['label']} label_id={row.get('label_id')} expected={expected_gold}"
            )
        if row.get("final_label") != row["label"]:
            errors.append(
                f"Row {row['id']} final_label mismatch: "
                f"label={row['label']} final_label={row.get('final_label')}"
            )

    # Polarity consistency
    for row in rows:
        expected_pol = POLARITY_MAP.get(row["label"])
        if expected_pol and row["polarity_should_be"] != expected_pol:
            errors.append(
                f"Row {row['id']} polarity mismatch: "
                f"label={row['label']} polarity_should_be={row['polarity_should_be']}"
            )
        expected_compat_pol = COMPAT_POLARITY_LABEL.get(row["label"])
        if expected_compat_pol and row.get("polarity_label") != expected_compat_pol:
            errors.append(
                f"Row {row['id']} compatibility polarity mismatch: "
                f"label={row['label']} polarity_label={row.get('polarity_label')}"
            )

    # Controlled-style compatibility defaults
    for row in rows:
        label = row["label"]
        expected_sufficient = 0 if label == "NOT_ENTITLED" else 1
        for field in ("sufficiency_label", "evidence_sufficient_label"):
            if row.get(field) != expected_sufficient:
                errors.append(
                    f"Row {row['id']} {field}={row.get(field)} "
                    f"expected {expected_sufficient}"
                )
        if row.get("frame_compatible_label") != 1:
            errors.append(f"Row {row['id']} frame_compatible_label must be 1")
        if row.get("predicate_covered_label") != 1:
            errors.append(f"Row {row['id']} predicate_covered_label must be 1")
        if row.get("intervention_type") != row["group"]:
            errors.append(f"Row {row['id']} intervention_type must equal group")
        if row.get("probe_type") != row["group"]:
            errors.append(f"Row {row['id']} probe_type must equal group")

    # Group counts
    group_counts = Counter(r["group"] for r in rows)
    for g in EXPECTED_GROUPS:
        if g not in group_counts:
            errors.append(f"Group missing: {g}")
        elif group_counts[g] != EXAMPLES_PER_GROUP:
            errors.append(
                f"Group {g} has {group_counts[g]} rows, expected {EXAMPLES_PER_GROUP}"
            )

    # expected_owner
    for row in rows:
        if row["expected_owner"] != "coverage_entailment":
            errors.append(
                f"Row {row['id']} wrong expected_owner: {row['expected_owner']}"
            )

    # Total
    if len(rows) != len(EXPECTED_GROUPS) * EXAMPLES_PER_GROUP:
        errors.append(
            f"Total rows {len(rows)} != "
            f"{len(EXPECTED_GROUPS) * EXAMPLES_PER_GROUP}"
        )

    return errors


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_json_report(rows: list[dict], errors: list[str]) -> dict:
    group_counts = Counter(r["group"] for r in rows)
    label_dist = Counter(r["label"] for r in rows)
    gold_dist = Counter(r["gold"] for r in rows)

    return {
        "stage": "Stage31-A",
        "probe_name": "Coverage/Entailment Diagnostic Probe",
        "version": "1.0",
        "date": "2026-06-27",
        "total_rows": len(rows),
        "examples_per_group": EXAMPLES_PER_GROUP,
        "groups": EXPECTED_GROUPS,
        "group_counts": dict(group_counts),
        "label_distribution": dict(label_dist),
        "gold_distribution": {str(k): v for k, v in gold_dist.items()},
        "label_mapping": {"REFUTE": 0, "NOT_ENTITLED": 1, "SUPPORT": 2},
        "schema_version": "Stage31-A2",
        "required_fields": sorted(REQUIRED_FIELDS),
        "compatibility_fields": [
            "pair_id",
            "label_id",
            "final_label",
            "frame_compatible_label",
            "predicate_covered_label",
            "sufficiency_label",
            "evidence_sufficient_label",
            "polarity_label",
            "intervention_type",
            "probe_type",
        ],
        "external_eval_compatibility": (
            "Rows include controlled-style id/pair_id/final_label/label_id fields "
            "for v5/v6 external-eval encoder compatibility."
        ),
        "expected_owner": "coverage_entailment",
        "diagnostic_only": True,
        "leakage_policy": (
            "This dataset must NOT be used for main training, calibration, "
            "threshold selection, checkpoint selection, or train/dev split "
            "construction. It is a diagnostic probe only."
        ),
        "validation_errors": errors,
        "validation_passed": len(errors) == 0,
        "known_limitations": [
            "Synthetic template sentences; does not reflect natural language diversity.",
            "Narrow coverage phenomena; no compositional nesting yet.",
            "No world-knowledge verification; entities are synthetic.",
            "Claim/evidence length is short and controlled; OOD robustness untested.",
            "Quantifier scope limited to all/some/no/only/also patterns.",
        ],
    }


def build_md_report(rows: list[dict], report: dict) -> str:
    gc = report["group_counts"]
    ld = report["label_distribution"]
    lines = [
        "# Stage31-A Coverage/Entailment Diagnostic Probe Report",
        "",
        "## Purpose",
        "",
        "Stage31-A introduces the **Coverage/Entailment** diagnostic track for the",
        "ContraMamba architecture. The target architecture hierarchy is:",
        "",
        "```",
        "Mamba Encoder",
        "→ Hard Core Validity",
        "→ Coverage / Entailment      ← Stage31 focus",
        "→ Residual Adjudication",
        "→ ANI-style Epistemic Diagnosis",
        "→ Polarity",
        "→ Final Label",
        "```",
        "",
        "This probe dataset is **diagnostic only**. It must not be used for",
        "training, calibration, threshold selection, checkpoint selection,",
        "or train/dev split construction.",
        "",
        "---",
        "",
        "## Why Coverage/Entailment is Distinct from Frame",
        "",
        "Frame mismatch asks: *Do claim and evidence involve the same",
        "event/entity/predicate frame?*",
        "",
        "Coverage/Entailment asks: *Does the evidence cover or entail the claim's",
        "required scope, strength, or specificity?*",
        "",
        "A pair can pass Hard Core Validity (same frame, entity, predicate) while",
        "failing Coverage/Entailment because the evidence is weaker in scope",
        "(e.g. `some` vs `all`), weaker in specificity (general vs specific), or",
        "weaker in exclusivity (`also` vs `only`).",
        "",
        "---",
        "",
        "## Directional Entailment Rules",
        "",
        "| Direction | Label |",
        "|-----------|-------|",
        "| All → Some | SUPPORT |",
        "| Some → All | NOT_ENTITLED |",
        "| Specific → General | SUPPORT |",
        "| General → Specific | NOT_ENTITLED |",
        "| Only → Base membership | SUPPORT |",
        "| Also → Only | NOT_ENTITLED |",
        "| Whole → Included part | SUPPORT |",
        "| Part → Whole | NOT_ENTITLED |",
        "| None → Some | REFUTE |",
        "| Some → None | REFUTE |",
        "",
        "---",
        "",
        "## Group Definitions and Counts",
        "",
        "| Group | Description | Label | Count |",
        "|-------|-------------|-------|-------|",
        f"| all_to_some_support | Evidence: ALL X; Claim: SOME X | SUPPORT | {gc.get('all_to_some_support', 0)} |",
        f"| some_to_all_not_entitled | Evidence: SOME X; Claim: ALL X | NOT_ENTITLED | {gc.get('some_to_all_not_entitled', 0)} |",
        f"| specific_to_general_support | Evidence: specific; Claim: general | SUPPORT | {gc.get('specific_to_general_support', 0)} |",
        f"| general_to_specific_not_entitled | Evidence: general; Claim: specific | NOT_ENTITLED | {gc.get('general_to_specific_not_entitled', 0)} |",
        f"| only_to_base_support | Evidence: only member; Claim: a member | SUPPORT | {gc.get('only_to_base_support', 0)} |",
        f"| also_to_only_not_entitled | Evidence: also a member; Claim: only member | NOT_ENTITLED | {gc.get('also_to_only_not_entitled', 0)} |",
        f"| whole_to_part_support | Evidence: whole set; Claim: included subset | SUPPORT | {gc.get('whole_to_part_support', 0)} |",
        f"| part_to_whole_not_entitled | Evidence: subset; Claim: whole set | NOT_ENTITLED | {gc.get('part_to_whole_not_entitled', 0)} |",
        f"| none_to_some_refute | Evidence: NO X; Claim: SOME X | REFUTE | {gc.get('none_to_some_refute', 0)} |",
        f"| some_to_none_refute | Evidence: SOME X; Claim: NO X | REFUTE | {gc.get('some_to_none_refute', 0)} |",
        "",
        "---",
        "",
        "## Label Distribution",
        "",
        "| Label | Gold | Count |",
        "|-------|------|-------|",
        f"| SUPPORT | 2 | {ld.get('SUPPORT', 0)} |",
        f"| NOT_ENTITLED | 1 | {ld.get('NOT_ENTITLED', 0)} |",
        f"| REFUTE | 0 | {ld.get('REFUTE', 0)} |",
        f"| **Total** | — | **{report['total_rows']}** |",
        "",
        "---",
        "",
        "## Stage31-A2 Schema Compatibility",
        "",
        "Each row includes both Stage31 diagnostic fields and controlled-style",
        "compatibility fields for v5/v6 external-eval encoders.",
        "",
        "Required identity and label fields:",
        "",
        "- `id` and stable unique `pair_id` (same value in this probe)",
        "- `claim` and `evidence`",
        "- `label` and `final_label` as string labels",
        "- `gold` and `label_id` as numeric labels",
        "- `group`, `coverage_relation`, `expected_owner`,",
        "  `hard_core_should_pass`, `polarity_should_be`, and `notes`",
        "",
        "Controlled-style auxiliary fields:",
        "",
        "- `frame_compatible_label = 1`",
        "- `predicate_covered_label = 1`",
        "- `sufficiency_label` and `evidence_sufficient_label` are `1` for",
        "  SUPPORT/REFUTE and `0` for NOT_ENTITLED",
        "- `polarity_label` is SUPPORT, REFUTE, or NONE",
        "- `intervention_type = group` and `probe_type = group`",
        "",
        "Numeric mapping remains `REFUTE=0`, `NOT_ENTITLED=1`, `SUPPORT=2`.",
        "",
        "These compatibility fields exist only so external prediction export can",
        "read the probe without ad hoc Kaggle-only schema rewrites. They do not",
        "change Stage31 probe semantics.",
        "",
        "---",
        "",
        "## Owner Rule",
        "",
        "All rows in this dataset have `expected_owner = coverage_entailment`.",
        "",
        "This means the intended decision axis for each pair is the",
        "Coverage/Entailment component of the target architecture, not Frame,",
        "Residual Adjudication, Polarity, or the Final Composer.",
        "",
        "---",
        "",
        "## Leakage Policy",
        "",
        "> **This dataset is diagnostic-only.**",
        ">",
        "> It must NOT be used for:",
        "> - Main classification training or fine-tuning",
        "> - Calibration",
        "> - Threshold selection",
        "> - Checkpoint, model, or hyperparameter selection",
        "> - Train/dev split construction",
        "> - OOD evaluation benchmarks",
        "",
        "Its sole purpose is to probe whether the Coverage/Entailment component",
        "of the target architecture correctly handles directional entailment.",
        "",
        "---",
        "",
        "## Known Limitations",
        "",
        "1. **Synthetic templates** — sentences use controlled vocabulary and",
        "   do not reflect natural language diversity.",
        "2. **Narrow coverage phenomena** — only all/some/no, only/also,",
        "   specific/general, and whole/part patterns are covered.",
        "   Compositional nesting (e.g. `all of some`) is not yet included.",
        "3. **No world-knowledge verification** — all entities are synthetic.",
        "4. **Short and controlled length** — OOD robustness on longer,",
        "   noisier text is untested.",
        "5. **No cross-axis interactions** — each pair keeps frame/entity/",
        "   relation stable to isolate coverage failures, so interactions",
        "   between Frame and Coverage axes are not exercised.",
        "",
        "---",
        "",
        f"*Generated by `scripts/build_stage31_coverage_entailment_probe.py` — "
        f"Stage31-A, 2026-06-27.*",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = build_dataset()
    errors = validate(rows)

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Write JSONL
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build and write JSON report
    report = build_json_report(rows, errors)
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Build and write MD report
    md = build_md_report(rows, report)
    with REPORT_MD_PATH.open("w", encoding="utf-8") as fh:
        fh.write(md)

    # Summary
    gc = Counter(r["group"] for r in rows)
    ld = Counter(r["label"] for r in rows)
    print(f"Stage31-A probe generation complete.")
    print(f"  Total rows : {len(rows)}")
    print(f"  Groups     : {len(gc)}")
    for g in EXPECTED_GROUPS:
        print(f"    {g}: {gc[g]}")
    print(f"  Labels     : SUPPORT={ld['SUPPORT']}  NOT_ENTITLED={ld['NOT_ENTITLED']}  REFUTE={ld['REFUTE']}")
    print(f"  JSONL      : {DATA_PATH}")
    print(f"  Report JSON: {REPORT_JSON_PATH}")
    print(f"  Report MD  : {REPORT_MD_PATH}")
    print(f"  Validation : PASSED")


if __name__ == "__main__":
    main()
