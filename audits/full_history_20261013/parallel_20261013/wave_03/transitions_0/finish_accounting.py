"""Finalize saved historical audit evidence without executing project/Git code.

Only stdlib parsing, hashing and five named output writes within this directory.
Existing helpers are parsed/read, never imported or invoked. No source inspection,
scientific computation, authority change, or authentication rerun is performed.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
QUEUE = ROOT / 'audits/full_history_20261013/parallel_20261013/wave_02/integration_02/remaining_census_work_queue.jsonl'
OUTPUT_NAMES = {'transition_interpretations.jsonl', 'historical_scientific_changes.jsonl',
                'remaining_transition_gaps.jsonl', 'coverage_validation.json', 'report.md'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write(name, value):
    assert name in OUTPUT_NAMES
    target = OUT / name
    assert target.resolve().parent == OUT.resolve()
    target.write_text(value, encoding='utf-8', newline='\n')

def jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)

# These decisions preserve source readings already performed before the resume.
# No missing binary/PDF/bytecode review is promoted to completion.
CONTINUED = {
80: ('S', 'README synchronizes the frontier to the five-scale paper parent b4d20e3fe61ec97af35a699b4bd454a5bd0eecb7 and independent K parent 5f5f4d6a80085ad8baf43445475d1c3049535c22. It closes standalone Gen4 NAME, parks unsupported broad K precursor, retains bounded 130M directional-alignment evidence, and defines Gen5 as causal-role-grounded state ownership. Five-scale recurrence does not establish fixed-plane, exact-geometry, universal-behavior or steering invariance. Phase0 is frozen but implementation/training/evaluation remain unauthorized in this historical README; causal-role integrity precedes task score. Prospective generation names are replaced with evidence-aware lineage. Earlier failures remain context.', 'Pinned five-scale/K syntheses and Phase0 authority, exact checkpoints and cohorts are needed for final study identity. A README branch synchronization alone does not establish a Git branch merge or linear succession between sibling README records.'),
82: ('P', 'Repairs manual-review infrastructure: require_no_validation_errors raises only for a nonempty errors list, avoiding eager errors[0] evaluation on a valid empty list at four call sites. Removes a tautological pair-list equality check; substantive membership/schema validators remain. No scientific label or cohort definition changes.', 'Historical review invocation/error logs would identify the repaired blocked attempt.'),
83: ('S', 'Introduces eval-only OOD NE-shift diagnostics: clone logits and subtract each configured shift from NOT_ENTITLED on rows with both temporal and predicate flags zero, then recompute three-class predictions and group metrics. False-entitlement rates use all group rows; false-NE rates use gold-entitled rows. One full-flag OOD forward feeds the sweep, mutually exclusive with flag ablation. The sweep changes available diagnostic comparisons, not training loss or checkpoint selection.', 'Original sweep exports, launch/checkpoint identities and clean/external calibration protocol are needed for measured results or adoption; OOD-tuned comparisons are diagnostic upper bounds.'),
84: ('P', 'Registers the dynamically loaded observer in sys.modules under spec.name before exec_module. This repairs module resolution for dataclass/import processing; the observer endpoint and intervention formulas do not change in this edge.', 'Original import failure and successful invocation logs are needed for attempt attribution.'),
86: ('S', 'Adds default-off TD-constrained checkpoint selection: paraphrase preservation >=0.80, diagnostic mismatch recall >=0.50 and control acceptance >=0.80 determine eligibility; absent/NaN diagnostic metrics fail via zero. The existing selection score chooses among eligible epochs, with unconstrained fallback when none qualify and explicit selected/fallback metadata. CLI describes clean final macro-F1, but code uses existing score. Reported eligible_epoch_count is a 0/1 indicator here rather than a count of every eligible epoch. Stage15/OOD is excluded.', 'Actual select_metric, clean/diagnostic dev data, epoch ledger and selected checkpoint are required for run identity and whether fallback occurred.'),
87: ('S', 'Introduces a reporting-only run audit ledger with loss accumulation, enabled losses/modifiers/selection rules, checkpoint fallback warnings and clean-versus-Stage15 attestations. Ranking is active when structured intervention loss is disabled. It retains historical shared temporal-gradient collapse warnings, untrained-adapter penalty warnings, and OOD-tuned shifts as nonpromotable diagnostic bounds. Individual auxiliary accumulators are not a fully weighted contribution ledger at this point; total-minus-CE mixes weighted total with raw component reporting, motivating later correction at ordinal22.', 'Original selected epochs, raw/weighted losses and launch manifests are required; static provenance booleans are reported claims, not proof of run behavior.'),
88: ('S', 'Adds Stage29-B external probe evaluation after checkpoint selection/best-state loading, with final-epoch fallback recorded. Probes are separately loaded, encoded with training tokenizer/vocabulary and padded/truncated to training max length; optional fields receive tensor-construction defaults. Three-class macro-F1 includes absent classes as zero, and false-SUPPORT is broken down by intervention/axis. Exports assert no external training/selection/calibration. The tolerant loader is not complete schema/label validation.', 'External source hashes, compatible-label schema, best-dev checkpoint and actual export/launch identity are needed; source wiring does not validate external observations.'),
89: ('S', 'Raw-recurrence observer changes velocity-rearrangement tolerance exceedance from a blocking require to DIAGNOSTIC_TOLERANCE_EXCEEDED with explicit allclose boolean and pass/exceedance accounting. Other record gates remain. Authority pointer changes from 41dbeae22a3831300f0de2f1f44a4b49733d70a5 to d8d717a09516b8562f64f7c503d4bbdcc9c34c5d, and allowed dirty-state contract now distinguishes historical untracked files from four tracked modified R2 implementation files. This is a historical execution/interpretation criterion revision, not a numerical PASS.', 'Original correction authority, failed capture/preflight and later diagnostic counts are needed for final attempt identity.'),
91: ('S', 'Adds O0C static backend-proof tests for a CUDA-only guarded fast return followed by a unique direct slow fallthrough. Missing/inverted/or guards, CPU-reachable optional kernels, unresolved conditions and duplicate dispatch fail or remain ambiguous. Optional names outside the reachable CPU path do not automatically block proof. Source location binding of the guard is asserted; static source proof is distinct from runtime numerical validity.', 'Installed Mamba source and historical preflight/dispatch evidence are needed for attempt eligibility.'),
92: ('S', 'Adds raw-write trajectory contract requiring visible and complement projected components to be added to the raw source state. Packed seven-arm order is source, two raw targets, two source-plus-visible and two source-plus-complement states. This distinguishes replacement trajectories from isolated projected displacements; the test is source evidence, not an executed transport result.', 'Companion transport implementation, original worker manifests and exact source/target tensors are needed for study attribution.'),
94: ('P', 'Adds runner/publication integration regression for str subclasses (including torch version) normalized to exact built-in strings accepted by the unchanged observer. Required provenance strings still reject empty/unknown values; seven-artifact publication and staging cleanup are checked in a synthetic fixture. No real Mamba execution or scientific conclusion is evidenced by the fixture.', 'Historical runner manifest and publication logs would show whether the type-related execution blocker was repaired.'),
95: ('S', 'Introduces optional Stage22-A preservation boundary auxiliary head/loss: none/paraphrase positive, five structural frame/slot swaps negative; sufficiency, polarity and predicate interventions are masked. BCE defaults weight0, pos_weight2.5 and exports diagnostic probability/accuracy/counts. The head does not directly rewrite logits, but an enabled auxiliary loss joins total loss and can train shared representations; unchanged predictions after training are not implied by the readout-only wording.', 'Companion model representation/detach definition, controlled cohort hash and enabled weight/checkpoint are needed for adoption/results.'),
96: ('S', 'README replaces the Stage5 mainline account with Stage26-H1 hierarchical Mamba and softplus polarity-energy gating: SUPPORT/REFUTE equal entitlement times positive energies, NE equals bias plus alpha times one-minus-entitlement. It claims additive geometry caused SUPPORT collapse and H1 recovered three-way behavior, reports three-seed macro-F1 0.900 +/-0.062, and introduces the clean no-time-swap 3600-row/300-pair cohort. It retains location/role false-SUPPORT concentrations, post-hoc min-gate results as diagnostic rather than trained H2, historical Stage7 routing, and controlled-data/generalization limits. Later ordinal9 explicitly supersedes H1 mainline; these historical claims are not independently certified.', 'Original H1/ablation seed reports, checkpoint selection and no-time-swap dataset/split hashes are needed for final study identity and causal-strength assessment.'),
97: ('P', 'Adds D1 reporting regression for resumed/no-P2-epoch-snapshot execution. It extracts the nested run_training_v6b function into a synthetic fixture with next_epoch2 and epochs1, so no training epoch runs, and requires empty neutral loss summary/history rather than accessing an A0-A3-only snapshot. This is report-path robustness, not a new D1 loss or gradient finding.', 'Original resumed-run report/error logs are needed to identify any actual blocked reporting attempt.'),
98: ('P', 'Generalizes vanilla-LM checkpoint key enumeration from one model.safetensors file to all pinned HF_FILES safetensor shards. Missing files, empty weight-file lists and duplicate keys across shards fail; observed weight filenames/count enter provenance. No objective/readout formula changes in this edge.', 'Pinned HF shard hashes, model configuration and runtime load log are needed for checkpoint identity.'),
100: ('S', 'Corrects in-projection reconstruction gating while retaining failed implementation cc539ff5a3b9d692fa946c0d8e470685ea8ac6e3. Authenticates correction authority 3df1cfaac5af7a5551307c33380b5dc1f33fecbd and its blob/hash. Replaces cancellation-sensitive differenced relative-residual <=1e-6 gate with frozen parent branch-level <=1e-6 gate plus an absolute error-difference identity check. Matched/swapped branch vectors are differenced in float64; difference reconstruction residual becomes diagnostic. Row-gain/alignment and energy closure gates persist. No causal effect is newly established.', 'Correction-design source, failed-run logs, parent capture provenance and corrected residual exports are required for study/attempt identity.'),
102: ('S', 'Adds Stage33-E whole/part-v2 shadow diagnostics and decision criteria: any overclaim/refute-to-SUPPORT or macro delta <=-0.05 is unsafe. Promising requires whole-to-part SUPPORT count >1, part-to-whole NE count >0, zero safety errors and macro delta >-0.02; weaker detection/recovery is diagnostic-only. Candidate/allowed/block reasons enter reports. These are historical source criteria, not observed promotion.', 'Paired shadow/baseline exports, group metadata and frozen Stage33-E protocol are needed for run identity.'),
103: ('S', 'Repairs P4-D integrity adversaries: one fresh pair ID now produces exactly PAIR_COUNT+1 while preserving row count, isolating pair-count drift instead of conflating duplicate-intervention errors. Non-F2 and polarity adversaries restore an authorized evidence change to historical evidence while perturbing targeted rows, isolating unauthorized deltas within authorized pairs. The authorized delta budget/labels are not changed by these fixture repairs.', 'Original gate authority, historical/corrected sidecars and actual integrity output are needed for attempt identity.'),
104: ('S', 'Adds Gen5 Phase2 streamed-backbone feasibility-recovery contracts pinned to amendment 360670cc657dbb7d2ee3ee894d9109846631a838. Backbone stream rows240 feed one full-batch downstream call and one loss/backward/optimizer step per epoch. Synthetic tests restore RNG and compare logits/loss/A-B gradients with monolithic execution, require frozen noncorrection parameters, and prohibit per-chunk downstream calls. Streamed CUDA preflight may backward but must attest no optimizer step/training/fresh XG1/scientific p-values. These are unexecuted test requirements.', 'Original amendment, bounded equivalence/preflight attestation, worker outputs and frozen checkpoint/data hashes are needed for actual feasibility or study outcomes.'),
105: ('S', 'Corrects O0C cache provenance to MambaCache in the same modeling_mamba source rather than a separate cache_utils identity. Adds semantic contracts for distinct convolution/SSM lists, correct constructor sizes, family-linked append/store and exact persistent indexed returns. Aliasing/crossed families, wrong indices/nonpersistent returns and ambiguous source shapes fail. This improves eligibility/source binding without proving observed dynamics.', 'Installed modeling_mamba hash/runtime registration and historical preflight manifest are needed for attempt attribution.'),
107: ('S', 'Introduces equal-length causal replay prefix-truncated through anchor+k6 for matched/swapped branches, retaining all k=-1..+6 tokens and requiring identical preanchor prefix. This removes full-sequence length/future-suffix execution-path differences; manifests explicitly state future_suffix_executed False. It changes execution conditions for divergence-aligned state audit, not the audited horizon or a demonstrated effect.', 'Original prefix token manifest, capture source/runtime and before/after diagnostic exports are needed to determine historical confound impact.'),
}

# Partial findings already recovered from saved/previously read material. These
# have no completion disposition because relevant changes remain unadjudicated.
PARTIAL = {
14: 'Companion main.tex/appendix.tex are unchanged. Figures README describes promotion of validated vector redesign without rerendering under an incompatible Python/NumPy ABI. PDF text extraction still has 53 changed hunks; publication-only status for the full PDF has not been established.',
38: 'Companion main.tex weakens test-and-reject exact invariance to exact CKA-level invariance does not hold on frozen cohorts, explicitly denies a calibrated-null similarity claim, and explains XG2/XG4 as controlled mechanistic generator contrast rather than latent semantic dichotomy. AI disclosure scope changes. Complete PDF text/source correspondence remains unadjudicated.',
47: 'Companion main.tex explicates projector-contrast eigenvalue lambda-plus=sin(theta), P5 as global maximizer and non-P3 control, and separates independently frozen 130M control from later discovery-conditioned control. Reproducibility archive and AI result-interpretation disclosures expand. Complete PDF correspondence remains unadjudicated.',
48: 'Read companion manuscript changes qualify opposing five-scale signs as point estimates; 370M pretrained-LM and 2.8B task resampling intervals cross zero. Partial CKA conservation, non-rowwise inversion, separate 130M completeness chronology and external 1.4B failure remain. Full multi-artifact/PDF reconciliation was not completed.',
64: 'Companion appendix replaces printed commit locators with byte-identified supplementary snapshot/SHA-256 provenance. Main.tex unchanged. Full 16-hunk PDF reconciliation was not completed; no new scientific effect is inferred from this packaging edit.',
77: 'Previous review identified addition of final-decision-aware temporal constrained selection; exact thresholds, denominator changes and full remaining hunks were not saved as a completed adjudication.',
78: 'Previous review identified A-init temporal-birth tolerance/replay contract changes. Source diffs exist, but full authority/endpoint/trace comparison was not saved as a completed adjudication.',
79: 'Previous review identified temporal-channel supervision additions. Source diffs exist, but full label, detach, objective and cohort semantics were not saved as a completed adjudication.',
85: 'Read Stage30-D trainer adds mismatch-positive1 inverse of safety labels, F/P/S multihead BCE/data wiring and optional hard/soft cap with fusion choices, defaults off, separate clean train-pair filtering. Companion model transition68 is adjudicated. Remaining full trainer hunks/exports were not completely reconciled.',
90: 'Companion manuscript/figure sources include replacement of scalar-only geometry narrative by frozen coordinate-insensitive XG2/XG4 CKA/RSM comparison. Complete 48-hunk PDF and companion-source reconciliation remains unfinished.',
93: 'Authenticated original source identity is tree 303f2863736d4ecb723314dab8a7616c43e49207; the same blob occurs at mapped after commit68bd1884510805f66a35317739d75ed7c444d799. This establishes byte identity, not the capture date or owning commit of the original tree snapshot. Image interpretation is unfinished.',
99: 'Read Stage31-C adds four-class directional diagnostic head, optional detach, separate auxiliary train/dev CE, explicit rejection of named evaluation probe and no direct composer edits. Detach defaults off so readout-only does not mean all representation gradients are isolated. Remaining complete trainer hunk/report reconciliation was not finished.',
}

MAP = {'S': 'SCIENTIFIC_REVISION_RECONSTRUCTED', 'P': 'EDITORIAL_OR_PROCEDURAL_ONLY',
       'D': 'DEPENDENT_ON_OTHER_SOURCE', 'I': 'INTERPRETATION_COMPLETE',
       'B': 'BOUNDED_LIMITATION', 'U': 'UNRESOLVED'}

def main():
    helpers = {name: OUT / name for name in ('reconstruct.py', 'adjudications.py', 'companions.py')}
    for name, path in helpers.items():
        # Parsing checks syntax without running imports, marshal or subprocess.
        ast.parse(path.read_text(encoding='utf-8'), filename=name)
    tree = ast.parse(helpers['adjudications.py'].read_text(encoding='utf-8'))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'NOTES' for t in node.targets))
    notes = ast.literal_eval(assignment.value)
    assert not set(notes).intersection(CONTINUED)
    notes.update(CONTINUED)
    assert len(notes) == 85

    manifest_path = OUT / 'target_manifest.jsonl'
    manifest = read_jsonl(manifest_path)
    queue = read_jsonl(QUEUE)
    population = sorted((r for r in queue if r['category'] == 'HISTORICAL_TRANSITION_INTERPRETATION'),
                        key=lambda r: r['original_queue_id'])
    assert len(population) == 216
    assert len({r['original_queue_id'] for r in population}) == 216
    assert all(r['id'] == r['original_queue_id'] == r['original_queue_record']['id'] for r in population)
    selected = population[::2]
    selected_ids = [r['original_queue_id'] for r in selected]
    assert len(manifest) == len(selected) == 108
    assert [m['original_queue_id'] for m in manifest] == selected_ids
    assert [m['population_index'] for m in manifest] == list(range(0, 216, 2))
    assert all(m['shard'] == 0 and m['integrated_queue_record'] == r for m, r in zip(manifest, selected))
    prior_validation = json.loads((OUT / 'partition_validation.json').read_text(encoding='utf-8'))
    assert prior_validation['queue_sha256'] == digest(QUEUE)
    assert prior_validation['sorted_population_ids'] == [r['original_queue_id'] for r in population]
    assert prior_validation['selected_ids'] == selected_ids

    input_paths = [QUEUE, manifest_path, OUT / 'partition_validation.json', *helpers.values()]
    interpretations, changes, gaps = [], [], []
    object_counts = collections.Counter()
    for k, m in enumerate(manifest):
        path = OUT / 'evidence' / f'{k:03}.json'
        e = json.loads(path.read_text(encoding='utf-8'))
        input_paths.append(path)
        assert e['shard_ordinal'] == k and e['original_queue_id'] == m['original_queue_id']
        assert e['original_source_ids'] == m['original_source_ids']
        original = e['original_source_identity']
        assert {key: original[key] for key in ('commit', 'path', 'blob')} == m['original_source_identities'][0]
        assert original['blob'] == e['after']['blob']
        assert e['relationship'] == 'ANCESTOR_TO_DESCENDANT_VERIFIED'
        for side in ('before', 'after'):
            identity = e[side]
            assert len(identity['commit']) == len(identity['blob']) == 40
            assert len(identity['sha256']) == 64 and identity['bytes'] > 0
            projection = OUT / 'evidence' / f'{k:03}_{side}.txt'
            assert projection.is_file()
            input_paths.append(projection)
            object_counts[identity['object_type']] += 1
        companion_path = OUT / 'evidence' / f'{k:03}_companions.json'
        companions = []
        if companion_path.exists():
            input_paths.append(companion_path)
            for c in json.loads(companion_path.read_text(encoding='utf-8')):
                companions.append({'path': c['path'],
                                   'before': c.get('before', {}).get('identity'),
                                   'after': c.get('after', {}).get('identity')})
        code, interpretation, dependency = notes.get(k, ('U', PARTIAL.get(k,
            'Before/after identities and full projections/diffs are recovered, but a completed scientific/editorial adjudication is not saved. No conclusion is inferred from filename, commit message, extraction or byte change alone.'), ''))
        disposition = MAP[code]
        complete = code in ('S', 'P', 'I')
        if code == 'U':
            suffix = Path(e['after']['path']).suffix
            if suffix == '.pyc':
                missing = 'Complete semantic adjudication of saved canonical bytecode changes and companion source; distinguish assertion rewrite, compilation metadata and actual objective/label/cohort changes. Compilation/source correspondence and exact pytest environment must not be inferred from filename.'
            elif suffix in ('.pdf', '.png'):
                missing = 'Complete interpretation/reconciliation of saved before/after publication evidence and authenticated companion sources; identify all substantive claim/endpoint/provenance changes versus render-only differences. Existing recovery is not completed semantic review.'
            else:
                missing = 'Complete adjudication of the remaining saved historical source hunks against companion implementation/stage authority and previous corrections; retain partial findings without declaring full interpretation closure.'
            dependency = missing
        r = {'schema_version': 'W03_TRANSITION_INTERPRETATION_V1',
             'original_queue_id': m['original_queue_id'], 'original_source_ids': m['original_source_ids'],
             'population_index': m['population_index'], 'shard': 0, 'shard_ordinal': k,
             'disposition': disposition, 'interpretation_complete': complete,
             'original_source_identities': m['original_source_identities'],
             'original_identity_attestation': original, 'before': e['before'], 'after': e['after'],
             'chronology': {'selected_edge': e['relationship'], 'immediate_parent_has_before_blob': e['immediate_parent_has_before_blob'],
                            'after_parents': e['after_parents'],
                            'limitation': 'Selected authenticated edge need not be an immediate commit parent. Separate branches/duplicate blobs do not by themselves establish supersession. Original tree identity, where present, is not a dated owning commit.'},
             'source_backed_interpretation': interpretation,
             'evidence': {'saved_record': f'evidence/{k:03}.json', 'saved_record_sha256': digest(path),
                          'before_projection': f'evidence/{k:03}_before.txt', 'after_projection': f'evidence/{k:03}_after.txt',
                          'comparison_projection': e['comparison_projection'], 'changed_hunk_count': len(e['changes']),
                          'changed_source_spans': [{key: h[key] for key in ('operation', 'before_start', 'before_end_exclusive', 'after_start', 'after_end_exclusive')} for h in e['changes']],
                          'canonical_bytecode_changed_functions': [c['code_path'] for c in e.get('canonical_bytecode_changes', [])],
                          'prior_audit_evidence': e['prior_evidence'], 'companion_identities': companions,
                          'authenticity_basis': 'Saved successful original/before/after Git object authentication; reused at user direction without re-running Git.'},
             'previous_correction_context': e.get('prior_interpretation'),
             'remaining_source_interpretation_gap': None if complete else dependency,
             'evidence_needed_for_final_study_identity': dependency if complete or code == 'D' else
                'Finish source interpretation first; then associate exact authority, cohort/split, invocation, checkpoint/runtime and original result artifacts where the historical source supports an executed study.',
             'scientific_correctness_validated': False, 'scientific_execution_performed': False,
             'bounded_limitation': None}
        if original['object_type'] == 'tree':
            r['chronology']['original_snapshot_owning_commit_or_capture_date'] = 'UNRESOLVED; byte identity to mapped after commit does not determine snapshot chronology'
        interpretations.append(r)
        # Partial substantive discoveries are explicitly separated from closed revisions.
        if code == 'S' or k in (38, 47, 48, 85, 99):
            changes.append({'schema_version': 'W03_HISTORICAL_SCIENTIFIC_CHANGE_V1',
                            'original_queue_id': r['original_queue_id'], 'original_source_ids': r['original_source_ids'],
                            'population_index': r['population_index'], 'shard_ordinal': k,
                            'disposition': disposition, 'reconstruction_complete': code == 'S',
                            'before': e['before'], 'after': e['after'],
                            'original_source_identities': m['original_source_identities'],
                            'historical_revision': interpretation,
                            'evidence_record': f'evidence/{k:03}.json',
                            'companion_identities': companions,
                            'remaining_question': None if complete else dependency,
                            'research_claim_boundary': 'Historical source meaning only; no independent numerical, causal, empirical or whole-census validation.'})
        if not complete:
            gaps.append({'schema_version': 'W03_REMAINING_TRANSITION_GAP_V1',
                         'original_queue_id': r['original_queue_id'], 'original_source_ids': r['original_source_ids'],
                         'population_index': r['population_index'], 'shard_ordinal': k,
                         'disposition': disposition, 'before': e['before'], 'after': e['after'],
                         'original_source_identities': m['original_source_identities'],
                         'established_so_far': interpretation, 'exact_missing_information': dependency,
                         'evidence_already_available': r['evidence'],
                         'blocks_source_interpretation_closure': True,
                         'additional_scientific_execution_authorized': False,
                         'required_next_action': 'Read/reconcile existing historical evidence or recover original source artifacts under a separately authorized continuation; do not rerun scientific models.'})

    counts = collections.Counter(r['disposition'] for r in interpretations)
    assert len(interpretations) == 108 and len({r['original_queue_id'] for r in interpretations}) == 108
    assert [r['original_queue_id'] for r in interpretations] == selected_ids
    assert all(r['disposition'] in MAP.values() for r in interpretations)
    assert len(gaps) == sum(not r['interpretation_complete'] for r in interpretations)
    assert {r['original_queue_id'] for r in gaps} == {r['original_queue_id'] for r in interpretations if not r['interpretation_complete']}
    assert sum(r['interpretation_complete'] for r in interpretations) == 84
    assert len(gaps) == 24 and len(changes) == 81
    # Preserve the inherited manifest/helpers/evidence before and after writing.
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(input_paths))}
    write('transition_interpretations.jsonl', jsonl(interpretations))
    write('historical_scientific_changes.jsonl', jsonl(changes))
    write('remaining_transition_gaps.jsonl', jsonl(gaps))

    report = [
        '# Wave 03A — Historical transition reconstruction, Shard 0', '',
        '**Status: accounting complete; historical interpretation incomplete.**', '',
        'All 108 assigned original IDs have exactly one disposition and source-evidence references. '
        '84 interpretations are complete: 76 scientific revisions reconstructed and 8 editorial/procedural changes. '
        '24 remain open: 23 UNRESOLVED and 1 DEPENDENT_ON_OTHER_SOURCE. No BOUNDED_LIMITATION was used. '
        'The scientific-changes file contains 76 completed revision records and 5 explicitly partial findings; '
        'these are target-level records, not independent studies or a count of individual propositions.', '',
        '## Authority and partition', '',
        'Authority is the current user-directed full-history census and the resume instruction. Phase is historical evidence reconstruction only. '
        'AGENTS.md supplies repository integrity rules. Historical stage documents describe prior authority; none is adopted as new execution authority.', '',
        f'The integrated queue has {len(queue)} records, of which exactly 216 have category '
        '`HISTORICAL_TRANSITION_INTERPRETATION`. The stable original_queue_id is unique and equals both id '
        'and original_queue_record.id. Ascending string order and even zero-based indices select 108 IDs. '
        'The inherited target_manifest.jsonl is unchanged and matches that exact selection. No old unresolved list substitutes for the integrated queue.', '',
        f'Queue SHA-256: `{digest(QUEUE)}`. Saved authentication HEAD: `{prior_validation["head"]}`. '
        'This resume does not reauthenticate objects or inspect current Git state; it checks cached identity/partition evidence and records the distinction.', '',
        '## Recovered revisions and preserved negatives', '',
        '- Gradient ownership: first-blocker/D1/Gen3 edge-specific and grouped-gradient tests distinguish forward identity from recipient gradient ownership. These are historical contracts, not newly executed gradient results.',
        '- Calibration/cohorts: split 174 is superseded by 8192; ordered identity and label-aware hashes differ. Calibration validity does not release class-degenerate polarity training. Stage189 training-compatible topology is 1440 within a 3600-row sidecar.',
        '- H1/temporal/coverage: README mainline and objective geometry change, optional diagnostic/supervision paths, abstention and shadow criteria are distinguished from trained final routing. OOD-tuned shifts remain diagnostic upper bounds. Readout-only heads may still affect shared representations through auxiliary gradients.',
        '- Historical failures retained: v7 SUPPORT collapse, selected-output identity mismatch, temporal shared-gradient collapse warnings, untrained adapter penalties, Stage99–106 no promotion, harmful/mixed pre-D ownership results, unsupported broad precursor and absence of universal invariance/steering conclusions.',
        '- Runtime and capture: nested trace capture fixes, persistent cache/source-role recognition, pinned environments, streaming worker/full-batch/RNG contracts and prefix truncation revise historical execution eligibility. Source tests and synthetic fixture assertions are not reported as runtime PASS.',
        '- Blocking criteria: velocity rearrangement exceedance becomes diagnostic; in-projection reconstruction preserves branch-level gate and checks error-difference identity. The earlier failed implementation is retained rather than erased.',
        '- Publication: paired primary endpoint display and frozen bootstrap interval provenance are reconstructed in source records. Partially reviewed manuscript PDFs retain explicit open reconciliation; no missing visual/bytecode adjudication is labeled editorial merely because source recovery exists.', '',
        '## Remaining gaps and final study identity', '',
        'The 23 unresolved records comprise 13 publication PDF/image targets, 5 bytecode targets and 5 source transitions '
        '(ordinals 77, 78, 79, 85, 99). Existing raw objects, projections, diffs and authenticated companion sources are retained. '
        'They require completed semantic reconciliation, not renewed Git authentication. The dependent notebook target '
        'ordinal7 requires the epoch10 checkpoint, launch history and complete threshold-sweep evidence; stored output alone does not recover missing results.', '',
        'The original figure snapshot at ordinal93 is a tree object, not a dated commit. Its blob matches the mapped '
        'after commit, which establishes byte identity without inventing an owning commit/capture date. This chronology gap remains.', '',
        'For completed source interpretations, evidence_needed_for_final_study_identity lists remaining historical '
        'run attribution needs: exact authority/cohort and pair-group split hashes, launch arguments, selected versus final '
        'checkpoint, installed code/runtime identities, export schemas and original result logs. These downstream attribution '
        'needs do not turn a fully reconstructed source delta into an observed scientific result.', '',
        '## Validation and changes', '',
        'Safe local stdlib accounting validated the original 216 population, exact even-index selection, 108 unique '
        'manifest IDs, cached original/before/after identity linkage, full disposition coverage and exact open-gap mapping. '
        'Saved authentication covers 108 original identities and 216 before/after endpoints. Companion identities '
        'are preserved where recovered. No Git object authentication, PDF/image inspection, project tests or scientific '
        'computation was repeated during this resume.', '',
        'The three helpers were reviewed as untrusted source: reconstruct.py uses read-only Git subprocesses and '
        'writes recovered evidence beside itself; companions.py adds recovered companion objects and disassembly; '
        'adjudications.py is a literal decision dictionary. Their prepare paths were not executed. The dictionary '
        'was parsed with ast.literal_eval, with no helper import, marshal load or historical code execution.', '',
        'Resume commands actually executed: PowerShell Get-Content of the three helpers; stdlib Python via stdin '
        'to inventory saved notes/schema/identity evidence; `python -X utf8 .../finish_accounting.py`; and the separate '
        'read-only accounting verification recorded in coverage_validation.json. The initial reconstruction previously '
        'executed read-only Git cat-file/rev-parse/show/merge-base and source extraction/disassembly, as documented '
        'in the saved scripts. An environmental Python launcher-location warning did not prevent successful accounting.', '',
        'Only the five requested deliverables and the local deterministic finish_accounting.py helper are added in '
        'this resume. Inherited manifest, helper decisions, prior audit files and evidence stay unchanged. '
        'No tracked-code diff/stat command or Git state mutation was run in this resume. Training, inference, '
        'evaluation, GPU work, commit and push: **not run**. No canonical ledger or other parallel output was read '
        'or modified during accounting; the prior saved evidence references completed Wave02 work, not concurrent Wave03 Shard1.', '',
        'Next action requires a separately authorized continuation for the 24 listed gaps and historical study '
        'attribution. This report does not claim scientific correctness or whole-census completion.', '',
        '## Assigned target inventory', '',
        '| Shard ordinal | Population index | Original queue ID | Disposition |',
        '|---:|---:|---|---|',
    ]
    report += [f'| {r["shard_ordinal"]} | {r["population_index"]} | {r["original_queue_id"]} | {r["disposition"]} |' for r in interpretations]
    write('report.md', '\n'.join(report) + '\n')
    assert all(digest(ROOT / name) == sha for name, sha in inputs.items())
    validation = {'schema_version': 'W03_TRANSITION_COVERAGE_V1',
        'status': 'PASS_ACCOUNTING_WITH_OPEN_INTERPRETATIONS', 'scientific_validation_status': 'NOT_RUN',
        'authority': 'Current user-directed full-history census plus explicit resume instruction',
        'queue_path': str(QUEUE.relative_to(ROOT)), 'queue_sha256': digest(QUEUE),
        'queue_record_count': len(queue), 'category': 'HISTORICAL_TRANSITION_INTERPRETATION',
        'population_count': 216, 'stable_id_field': 'original_queue_id',
        'stable_id_unique_and_original_identity_consistent': True,
        'sorting': 'ascending string original_queue_id', 'shard_selection': 'zero-based even population indices',
        'selected_count': 108, 'assigned_ids': selected_ids,
        'population_indices': list(range(0, 216, 2)), 'disposition_counts': dict(counts),
        'completed_interpretations': 84, 'remaining_open_interpretations': 24,
        'unresolved': 23, 'dependent_on_other_source': 1, 'bounded_limitations': 0,
        'completed_consequential_revision_records': 76, 'partial_consequential_revision_records': 5,
        'scientific_change_rows': 81, 'gap_rows': 24, 'full_one_disposition_per_assigned_id': True,
        'exact_gap_coverage': True, 'original_manifest_preserved': True,
        'git_identity_validation': {'mode': 'REUSED_SAVED_AUTHENTICATION_NO_GIT_RERUN',
            'saved_head': prior_validation['head'], 'original_identity_count': 108,
            'before_after_identity_count': 216, 'selected_endpoint_object_types': dict(object_counts),
            'full_commit_path_blob_references_preserved': True,
            'manifest_to_evidence_identity_checks': 108, 'authenticated_ancestor_edges_reused': 108,
            'tree_snapshot_chronology_closed': False},
        'helper_review': {'prepare_paths_run_on_resume': False, 'imports_executed': False,
                         'adjudications_parser': 'ast.literal_eval',
                         'effect_boundary': 'Read-only stdlib accounting; output allowlist under this directory'},
        'input_sha256': inputs,
        'input_files_unchanged_after_accounting': True,
        'output_sha256': {name: digest(OUT / name) for name in sorted(OUTPUT_NAMES - {'coverage_validation.json'})},
        'outputs_written': sorted(OUTPUT_NAMES), 'current_git_state_rechecked_on_resume': False,
        'training_evaluation_inference_gpu_executed': False, 'git_mutation_commit_push_executed': False,
        'canonical_ledgers_modified': False, 'concurrent_wave03_shard1_read': False,
        'independent_verification': 'PENDING_SEPARATE_READ_ONLY_ACCOUNTING_CHECK',
        'whole_census_complete': False, 'scientific_correctness_claimed': False}
    write('coverage_validation.json', json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'assigned': 108, 'completed': 84, 'open': 24, 'dispositions': dict(counts),
                      'scientific_revisions_complete': 76, 'scientific_revisions_partial': 5}, sort_keys=True))

if __name__ == '__main__':
    main()
