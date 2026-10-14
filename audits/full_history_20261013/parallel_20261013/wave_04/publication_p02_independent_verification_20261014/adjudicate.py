from verify import *

LIMITS = [
 'Historical P02 FAIL_STOP_REQUIRED and raw Git ref/worktree FAILs remain immutable; movement qualification is a separate scoped conclusion.',
 'Transient observations and intervals between captures remain uncertified; ref-moving actors, causes and exact movement times are unknown.',
 'Historical 370M failed joint residual-recurrence criterion remains unsuccessful; later positive tests do not rescue it.',
 'Stored CKA, means, intervals, association values and support flags are documentary evidence; no scientific validity, original successful execution or independently replicated inference is certified.',
 'No global paper adoption, cross-scale semantic homology, model/run equality, arbitrary-linear representation equivalence or fitted scaling law is inferred.',
 'Original source-specific root requirements are empty for this documentary obligation. Transition instructions about subsequent execution-identity work are retained, not declared completed.',
 'Zero new closure credit; no ledger or prior closure overlay is changed.'
]

def finish(owner_id, findings, op_texts, hunk_texts, visuals):
    cp=read(OUT/'checkpoint.json');expected=list(OWNERS);assert owner_id==expected[len(cp['completed_owners'])]
    e=read(OUT/(owner_id+'_evidence.json'));assert len(op_texts)==len(e['manifest_operations']);assert len(hunk_texts)==len(e['README_hunks'])
    e['manifest_operation_review']=[{'operation_number':i+1,'pointer':x['pointer'],'independent_interpretation':t,'review_status':'REVIEWED'} for i,(x,t) in enumerate(zip(e['manifest_operations'],op_texts))]
    e['README_hunk_review']=[{'hunk_number':i+1,'before_lines':[x['before_line_start'],x['before_line_end']],'after_lines':[x['after_line_start'],x['after_line_end']],'independent_interpretation':t,'review_status':'REVIEWED'} for i,(x,t) in enumerate(zip(e['README_hunks'],hunk_texts))]
    e['visual_inspection']={'performed_by':'current independent verifier through view_image; complete original PNGs and new complete PDF-page renders','inspection_utc':utc(),'visuals':visuals,'complete':True,'all_panels_and_footnotes_inspected':True,'pdf_png_scientific_content_agrees':True}
    e['independent_substantive_findings']=findings;e['semantic_review_status']='COMPLETE';e['completed_utc']=utc();write(owner_id+'_evidence.json',e)
    evidence_name=owner_id+'_evidence.json';h=meta(OUT/evidence_name)['sha256']
    loc=lambda p:{'file':str((OUT/evidence_name).relative_to(ROOT)),'file_sha256':h,'json_pointer':p}
    clauses=[
      ('complete_before_after_interpretation',['/complete_pdf_content','/visual_inspection','/independent_substantive_findings']),
      ('authenticated_companion_sources',['/source_bindings','/numeric_verification/source_declaration_and_output_checks']),
      ('all_substantive_claim_changes',['/independent_substantive_findings','/manifest_operation_review','/README_hunk_review']),
      ('all_endpoint_changes',['/numeric_verification','/all_existing_extracted_values','/old_plot_data_checks']),
      ('all_provenance_changes',['/chronology','/source_bindings','/numeric_verification/source_declaration_and_output_checks','/manifest_operation_review']),
      ('render_only_versus_substantive_differences',['/visual_inspection','/manifest_operation_review','/README_hunk_review','/independent_substantive_findings'])]
    row={'schema':'P02_INDEPENDENT_OWNER_VERDICT_V1','phase':'CENSUS_RECOVERY_ONLY','source_id':owner_id,'debt_id':OWNERS[owner_id],'verdict':'VERIFIED_FULL_DOCUMENTARY_SATISFACTION','original_obligation':e['original_records']['original_locator']['payload'],'original_debt_locator':e['original_records']['original_locator']['locator'],'original_root_locator':e['original_records']['root_locator']['locator'],'original_transition_locator':e['original_records']['transition_locator']['locator'],'literal_original_requirement_coverage':[{'literal_clause':c,'status':'DOCUMENTARILY_SATISFIED','decomposition':list(k for k,p in clauses)} for c in e['original_records']['original_locator']['payload']['exact_missing_evidence']],'clause_verdicts':[{'clause':k,'status':'SATISFIED','evidence_locators':[loc(p) for p in ps]} for k,ps in clauses],'additional_source_specific_root_clauses':[],'unresolved_original_documentary_clauses':[],'substantive_findings':findings,'evidence_file':loc(''),'numeric_checks':loc('/numeric_verification'),'visual_checks':loc('/visual_inspection'),'companion_operation_count':len(e['manifest_operations']),'README_hunk_count':len(e['README_hunks']),'retained_limitations':LIMITS,'new_applied_credit':0,'existing_accounting':[222,20,192],'frozen_accounting':[222,17,195],'scientific_certification':False,'global_paper_adoption_inferred':False,'training_evaluation_performed':False,'completed_utc':utc()}
    dest=OUT/'owner_verdicts.jsonl';old=dest.read_bytes();line=json.dumps(row,ensure_ascii=True,sort_keys=True).encode();dest.write_bytes(old+line+b'\n');assert dest.read_bytes().startswith(old)
    for x in cp['completed_owners']:assert sha(dest.read_bytes().splitlines()[x['line']-1])==x['line_sha256_excluding_lf']
    cp['completed_owners'].append({'source_id':owner_id,'debt_id':OWNERS[owner_id],'verdict':row['verdict'],'line':len(cp['completed_owners'])+1,'line_sha256_excluding_lf':sha(line),'evidence_file':evidence_name,'evidence_sha256':h,'completed_utc':row['completed_utc']});cp['owner_verdicts_sha256']=sha(dest.read_bytes());write('checkpoint.json',cp)
    em=read(OUT/'evidence_manifest.json');em['owners'][owner_id].update({'evidence_sha256':h,'verdict':row['verdict'],'owner_verdict_line':len(cp['completed_owners'])});write('evidence_manifest.json',em)
    print(owner_id+': VERIFIED_FULL_DOCUMENTARY_SATISFACTION persisted; zero credit.')

def hv114():
    e=read(OUT/'HV000114_evidence.json')
    texts=[]
    for op in e['manifest_operations']:
        p=op['pointer']
        if p.startswith('/extracted_values/') and p.endswith('/figure'):t='Reassigns existing scalar or geometry-execution metadata from fig3 to supporting; value, source and locator are unchanged. Documentary membership changes, not a new measurement.'
        elif p.startswith('/extracted_values/chain.'):t='Adds a source-reported supported Boolean for the 130M Figure 2 causal chain. Existing endpoint means are unchanged; this is shared companion bookkeeping, not new Figure 3 numerical evidence or independently validated causal support.'
        elif p.startswith('/extracted_values/xg'):t='Adds the frozen centered-linear CKA or secondary cosine-RSM matrix with its exact JSON pointer; independently compared to authenticated result bytes. New main geometry comparison, not recomputation.'
        elif p=='/notes':t='Replaces four source/design caveats: CKA transcription, scalar relocation, historical inferential-family retention and later 130M completeness chronology. Manuscript qualification and failed criterion are independently read.'
        elif p.startswith('/outputs/'):t='Changes a declared regenerated output SHA256. Authenticated against commit:path:blob bytes; only this owner original Figure 3 PDF/PNG visuals are interpreted here. Sibling content/adoption is not inferred from its hash.'
        elif p=='/plot_data/chain_supported':t='Adds four frozen support indicators to shared Figure 2 plot data; no existing numeric value changes.'
        elif p=='/plot_data/coordinate_free_geometry':t='Adds both frozen 5x5 CKA and RSM matrices. All CKA display cells and raw matrices are independently source-checked; historical geometry and objectives remain.'
        elif p=='/schema_version':t='Manifest schema changes from 1 to 2 to distinguish source-specific authorities; an artifact schema/provenance change is reported, not silently normalized.'
        elif p=='/scientific_authorities':t='Adds explicit base-evidence and coordinate-free-snapshot commit identities while the existing base scientific_head remains unchanged.'
        elif p.startswith('/sources/') and p.endswith('/authority_commit'):t='Adds the existing base-evidence commit as an explicit per-source authority. Blob/hash/source byte identity remains unchanged and independently authenticated.'
        elif p.startswith('/sources/') and p.endswith('/figures/0'):t='Reassigns a historical scalar/source dependency from fig3 to supporting. Historical source bytes and scientific claims are retained.'
        elif p.startswith('/sources/'):t='Adds a frozen coordinate-free result/provenance/snapshot source with exact commit/blob/SHA256 and figure membership. Complete documentary provenance payload is retained; underlying scientific execution is not replicated.'
        elif p.startswith('/tools/'):t='Changes declared publication tool/render provenance: removes Poppler declaration, adds 400 dpi and PyMuPDF 1.28.2, updates ReportLab 4.4.9 to 5.0.1. This can change pixels and hashes but does not alone establish a scientific change.'
        else:raise ValueError(p)
        texts.append(t)
    htexts=[
      'Splits the former single scientific authority into base-evidence commit and frozen geometry snapshot commit.',
      'Removes the prior ReportLab/Poppler environment requirement from the reproduction introduction.',
      'Deletes git diff --check from the example command block; no command in that block is executed by this verifier.',
      'Replaces the detailed rebuild, worktree-drift, deterministic-render and environment claims with pinned-Git-only/no-execution wording. Documentary reproduction detail is reduced; scientific source identities are checked directly.',
      'Changes the scientific-boundaries heading to Main-figure design.',
      'Rewrites all figure-design bullets: new causal-chain and CKA narrative, scalar geometry moved to appendix, chronology referred to appendix. Deleted visible failed-criterion wording is not treated as a successful criterion; after appendix explicitly preserves failure.',
      'Replaces Exact extracted values heading with Frozen coordinate-free geometry.',
      'Replaces Figure 2 section heading with XG2 CKA heading.',
      'Replaces Figure 2 means table with full XG2 CKA matrix; old means remain unchanged in manifest and source inventory.',
      'Replaces the Figure 2 association bullets with XG4 CKA heading; existing correlation values remain in the manifest.',
      'Replaces Figure 3 heading with full XG4 CKA matrix.',
      'Replaces old combined rank/causal/scalar table with secondary RSM heading; old causal and scalar values remain unchanged in manifest, with scalars repeated below.',
      'Replaces Figure 4 heading with 20 upper-triangle secondary cosine-RSM entries, two families with ten unique scale pairs each.',
      'Rewrites objective/association table as supporting 10-scalar geometry table and 10 objective means. Associations disappear from README table but persist unchanged in manifest; no numerical correction is inferred.',
      'Adds explicit base commit pins to overview references without changing source bytes.',
      'Adds explicit base commit pins to 130M causal-foundation references without changing source bytes.',
      'Revises Figure 3 reference inventory: base causal/confirmation references retained and pinned; direct scalar-source listing replaced by frozen coordinate-free result. Scalar sources remain authenticated as supporting manifest dependencies.',
      'Adds explicit base commit pins to objective references; original source bytes and objective means remain unchanged.'
    ]
    pdfs=e['complete_pdf_content'];pngs=e['png_metadata'];visuals=[]
    for x in pdfs:
        rp=OUT/('HV000114_'+x['side']+'_pdf-1.png');visuals.append({'kind':'complete_rendered_PDF_page','side':x['side'],'source':x['saved_file'],'source_sha256':x['sha256'],'render_file':rp.relative_to(ROOT).as_posix(),**meta(rp),'panels':'A table, B and C full plots/matrices, complete footnotes; all content inspected'})
    for x in pngs:visuals.append({'kind':'complete_original_companion_PNG',**x,'panels':'A table, B and C full plots/matrices, complete footnotes; all content inspected'})
    findings=[
      {'id':'114-CLAIMS','finding':'Panel A title changes from Scale-local causal roles across all five Mamba-1 scales to The causal role recurs across all five Mamba-1 scales. Result becomes Causal role; five source-reported Supported entries and all five selected/control pairs remain. D_SUF/D_DOM/D_CORE row and visible protocol/failure caveat are removed from figure, with historical details moved to Appendix A.4. Equal rank numbers explicitly do not imply semantic homology.'},
      {'id':'114-GEOMETRY','finding':'Panel B kernel mean-squared weights and Panel C leading-plane eigenvalues are replaced with two 5x5 centered-linear CKA heatmaps. New caption says same 300 response-blind pairs per family, identical centered sample geometry at CKA=1, partial/non-invariant cross-scale geometry, secondary RSM in Appendix A.2. All 50 displayed two-decimal CKA cells agree with authenticated source and manifest; all 50 raw CKA and 50 RSM cells transcribe frozen matrices. No CKA computation or representation-equivalence proof is performed.'},
      {'id':'114-SCALARS','finding':'All ten displaced mu_k2/lambda1 scalars match original source locators, retain exact raw values in manifest and README supporting table. Original 390 extracted values and every pre-existing plot-data field remain unchanged. Appendix lines 76-81 retain exact 130M scalars; lines 232-241 retain rounded four-scale series and distinct provenance. README no longer displays every old causal/association scalar, but the manifest preserves them.'},
      {'id':'114-FAILURE','finding':'Authenticated after appendix lines 206-228 explicitly says 370M historical joint residual criterion remains failed/not established and later core confirmations do not retroactively rescue it. The frozen cross-scale synthesis also retains NOT ESTABLISHED and NO rescue. All-scale positive support in Panel A is a scale-local endpoint report, not historical joint-criterion success.'},
      {'id':'114-PROVENANCE','finding':'All 84 manifest operations and 18 README hunks are individually reviewed. Added frozen geometry result/provenance/snapshot sources use snapshot 2e65dd887ba4159d4024d85b8996306cfadec023; base evidence remains 3bd8164bdf4b4f433d85a181bcfe8f6c81b41e69. Existing sources gain explicit authority pins; scalar memberships move to supporting. Shared chain support/output digest additions do not substitute for Figure 3 owner evidence. Tool versions and manifest schema change are reported.'},
      {'id':'114-RENDER','finding':'Full original PDF page and PNG on both sides inspected. PDF mediabox changes from 504x376 to 504x390 points; PNG from 2800x2089 to 2800x2167. Table geometry, fonts, white space and color treatment change, including PDF/PNG text-color differences. Ten scalar markers/axes disappear and 50 matrix cells/labels appear: substantive comparison and narrative changes, not render-only. PDF/PNG agree in scientific content. No global paper adoption or scientific validity is inferred.'}
    ]
    finish('HV000114',findings,texts,htexts,visuals)
def objective_ops(owner_id):
    e=read(OUT/(owner_id+'_evidence.json'));texts=[]
    for op in e['manifest_operations']:
        p=op['pointer']
        if p.startswith('/extracted_values/'):
            field=p.rsplit('.',1)[1]
            t=('Adds a duplicate frozen bootstrap point mean, equal to the existing objective mean; no point estimate changes.' if field=='point_mean' else 'Adds a frozen percentile interval endpoint, a substantive uncertainty field from exact CSV row/column; no bootstrap inference is rerun.')
            if owner_id=='HV000117':t+=' This shared companion field belongs to Figure 4 and is not a Figure 3 CKA change.'
        elif p=='/notes':t='Adds the frozen Figure 4 percentile-interval and no-builder-bootstrap note; all four existing geometry/inferential-family/chronology notes survive unchanged.'
        elif p.startswith('/outputs/'):t='Changes a declared publication output SHA256; exact commit/path/blob bytes authenticate. This owner original visuals are inspected; sibling hashes alone do not establish their visual semantics or global adoption.'
        elif p=='/plot_data/objective_intervals':t='Adds all ten point-mean/lower/upper records from frozen CSV: ten duplicated unchanged means and twenty new endpoints. This is shared Figure 4 content, separate from Figure 3 CKA.'
        elif p=='/scientific_authorities/pair_resampling_robustness':t='Adds exact pair-resampling documentary authority commit 8423ef9d1325c007ecd31849a668131197e2214d; base and coordinate-free authority identities remain unchanged.'
        elif p.startswith('/sources/'):t='Adds objective_bootstrap.csv at exact commit, blob 956f01c764871c8d27ab34dab04195dba773432e, SHA256 833d3f670afe4aae5a481226c49d6fc11bf710c7e521178d3d83911da0e42cd8, Figure 4 membership. Stored data authentication is separate from independent bootstrap replication.'
        else:raise ValueError(p)
        texts.append(t)
    return texts

def owner_visuals(owner_id):
    e=read(OUT/(owner_id+'_evidence.json'));out=[]
    for x in e['complete_pdf_content']:
        rp=OUT/(owner_id+'_'+x['side']+'_pdf-1.png');out.append({'kind':'complete_rendered_PDF_page','side':x['side'],'source':x['saved_file'],'source_sha256':x['sha256'],'render_file':rp.relative_to(ROOT).as_posix(),**meta(rp),'panels':'A, B, C and all complete labels/captions/footnotes visually inspected'})
    for x in e['png_metadata']:out.append({'kind':'complete_original_companion_PNG',**x,'panels':'A, B, C and all complete labels/captions/footnotes visually inspected'})
    return out

OBJECTIVE_HUNKS=[
 'Adds the exact objective-bootstrap CSV source path and full frozen robustness commit in the README authority header.',
 'Replaces the Figure 4 descriptive-readout design bullet with point means plus frozen percentile 95% pair intervals, retaining prospective chronology in appendix. A substantive uncertainty-display description, not a Figure 3 numerical edit.',
 'Adds the exact CSV/full-commit reference to the Figure 4 source inventory. Other README content remains byte-identical.'
]

def hv117():
    caveat='Selected/control ranks are local identities; equal rank numbers do not imply semantic homology.'
    e=read(OUT/'HV000117_evidence.json');before=next(x for x in e['complete_pdf_content'] if x['side']=='before')['pages'][0]['text'];after=next(x for x in e['complete_pdf_content'] if x['side']=='after')['pages'][0]['text']
    assert caveat in before and caveat not in after
    main=next(x for x in e['manuscripts'] if x['path'].endswith('main.tex'))
    assert 'These are scale-local identities rather than claims of same-numbered-plane homology.' in main['numbered_text'][116]
    findings=[
      {'id':'117-DELETION','exact_deleted_text':caveat,'finding':'The complete before/after Figure 3 PDF and PNG remove exactly this explicit local-rank/non-homology sentence. This is a substantive loss of visible qualification, not solely spacing/rendering. Panel A title, five rank pairs, five Supported labels, common contrast and appendix protocol referral remain.'},
      {'id':'117-QUALIFICATION','finding':'At authenticated after commit 048fb78258bbe0dac29f0dfbe0310b9023ecc672, main.tex line 117 retains: These are scale-local identities rather than claims of same-numbered-plane homology. Lines 118-119 retain common contrast and separate failed 370M residual criterion referral. Appendix lines 205-227 preserves unsuccessful joint criterion and no retroactive rescue; line 240 rejects same-numbered-plane homology.'},
      {'id':'117-NUMBERS','finding':'All 50 CKA cells on before and after PDFs independently match rounded authenticated matrices; raw CKA and RSM matrices stay exactly equal in manifest and source. All 398 pre-existing extracted values and all prior plot-data fields remain equal. No Figure 3 numerical correction accompanies the removed caveat. Ten supporting scalars also remain source-identical.'},
      {'id':'117-COMPANIONS','finding':'All 40 manifest operations and 3 README hunks are independently recomputed and individually reviewed for this owner. Thirty bootstrap fields are ten duplicate point means plus twenty interval endpoints, all checked against frozen CSV rows 2-11; objective_intervals, robustness authority, source and notes are shared Figure 4 additions. Six output hashes change; each authenticates against actual Git blobs without assuming unseen sibling visuals. Tools, base/geometry source bindings and all existing numerical fields remain.'},
      {'id':'117-RENDER','finding':'Both complete one-page PDFs and both original PNGs inspected. PDF height decreases 390 to 378 points at 504-point width; PNG height 2167 to 2100 at 2800-pixel width. Table/heatmap placement and white space shift upward after caveat deletion; fonts and pixels also differ. Matrix labels, values, ranks, support statuses, centered-geometry and secondary-RSM caption agree across PDF/PNG. Figure omission does not establish semantic homology, causal proof or global paper adoption.'}
    ]
    finish('HV000117',findings,objective_ops('HV000117'),OBJECTIVE_HUNKS,owner_visuals('HV000117'))
def hv121():
    e=read(OUT/'HV000121_evidence.json');n=e['numeric_verification'];assert len(n['actual_PDF_vector_checks'])==2
    after=next(x for x in n['actual_PDF_vector_checks'] if x['side']=='after');assert len(after['interval_endpoint_coordinates'])==20
    findings=[
      {'id':'121-UNCERTAINTY','finding':'Figure 4 adds percentile 95% pair-resampling whiskers to all ten bars. Caption changes from frozen means/explicit signs to bars and signs are point means, whiskers are percentile 95% pair intervals. Point-sign sequences +,+,+,-,- and -,-,-,+,+ remain descriptive. Panel C heading changes from Opposite aggregate signs do not imply rowwise inversion to Pair-level associations remain mixed. Main.tex lines 161-168 qualifies two cells as lacking individual sign stability and rejects simple rowwise inversion.'},
      {'id':'121-ENDPOINTS','finding':'All ten point means remain byte-equivalent numerical fields before/after; stored CSV point_mean, not bootstrap_mean, supplies each bar. All twenty added lower/upper bounds match frozen CSV rows 2-11 and manifest plot_data/objective_intervals plus extracted-value locators. Independently parsed actual PDF vector rectangles match all ten means on each side; twenty whisker endpoints and caps match source values within 0.001 PDF point serialization tolerance. No new statistical calculation is substituted.'},
      {'id':'121-ZERO','finding':'Exactly two stored and visibly drawn intervals contain zero: 370M LM [-0.006551474274649998, 0.0025359292834950076] around -0.0019753191100731616; 2.8B task [-0.0004860163949396766, 0.00018914839266922622] around -0.0001483499070046979. Their drawn vertical segments cross their actual zero grid line. The other eight intervals remain wholly one-sided. Containing zero does not establish true-zero effects or scientific coverage.'},
      {'id':'121-ASSOCIATIONS','finding':'All fifteen Pearson/Spearman/pair-sign-agreement associations remain numerically identical. Both complete PDF tables match the raw manifest values at four-decimal display precision, and both original PNG tables agree visually. All 398 pre-existing extracted values and all prior plot-data fields remain equal.'},
      {'id':'121-COMPANIONS','finding':'All 40 manifest operations and 3 README hunks are independently recomputed and individually interpreted for HV000121. Thirty extracted additions contain ten duplicate point means and twenty new bounds; objective_intervals, new frozen authority/CSV source, interval note and six output digest updates complete the forty. Shared Figure 3 regeneration/deleted caveat is not a Figure 4 mean change and another owner verdict is not used as satisfaction evidence.'},
      {'id':'121-BOOTSTRAP','finding':'Exact CSV authority is 8423ef9d1325c007ecd31849a668131197e2214d; blob 956f01c764871c8d27ab34dab04195dba773432e; SHA256 833d3f670afe4aae5a481226c49d6fc11bf710c7e521178d3d83911da0e42cd8. Complete stored CSV has ten records, N_pairs=300 each; manifest contra maps to CSV task and manifest vanilla maps to CSV lm. Appendix lines 342-347 reports B=10000, percentile 95% intervals, both zero crossings and no new prospective family. This verifies stored documentary correspondence, not original invocation, resampling seed, successful execution, bootstrap replication or inferential validity.'},
      {'id':'121-RENDER','finding':'Both full original PDF pages and companion PNGs inspected across all panels/captions. Dimensions remain 504x398 PDF points and 2800x2212 PNG pixels. Whiskers are new substantive uncertainty content; task upper tick expands to 0.006 and plot coordinates remap. Separate task/LM raw scales, bar colors, point signs, objective definitions, numerical association table and later-130M chronology remain. Font/placement changes are render differences; added intervals and changed caption/Panel C wording are substantive documentary changes. No causal or global paper adoption inference is made.'}
    ]
    finish('HV000121',findings,objective_ops('HV000121'),OBJECTIVE_HUNKS,owner_visuals('HV000121'))
if __name__=='__main__':{'HV000114':hv114,'HV000117':hv117,'HV000121':hv121}[sys.argv[1] if len(sys.argv)>1 else 'HV000114']()
