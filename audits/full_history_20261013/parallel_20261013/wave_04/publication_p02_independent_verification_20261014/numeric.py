import ast, json, pathlib, re, sys
from verify import *

def numerical(owner_id):
    e=read(OUT/(owner_id+'_evidence.json'))
    src=e['source_bindings']; checks=[]; data={}
    for side in ['before','after']:
        manifest=next(x for x in src if x.get('side')==side and x['path'].endswith('figure_data_manifest.json'))
        _,mb=binding(manifest);m=json.loads(mb);data[side]=m
        for p,d in m['sources'].items():
            matches=[x for x in src if x.get('side')==side and x['path']==p]
            assert len(matches)==1,(side,p)
            x=matches[0]
            assert x['blob']==d['git_blob'] and x['sha256']==d['frozen_bytes_sha256']
            authority=d.get('authority_commit',m.get('scientific_head'))
            assert x['commit']==authority,(x,authority)
            checks.append({'side':side,'source':p,'authority':authority,'blob':x['blob'],'sha256':x['sha256'],'status':'AUTHENTICATED_DECLARATION_AND_BYTES'})
        # Authenticate all sibling digest changes without making a visual claim about siblings.
        commit=next(x['commit'] for x in src if x.get('side')==side and x.get('role')=='publication')
        for name,h in m['outputs'].items():
            p='paper/iclr2027/figures/'+name;bid=git('rev-parse',commit+':'+p).decode().strip();b=git('cat-file','blob',bid)
            assert sha(b)==h;checks.append({'side':side,'path':p,'commit':commit,'blob':bid,'sha256':h,'status':'SIBLING_DIGEST_ONLY_NO_UNSEEN_VISUAL_INFERENCE'})
    a,b=data['before'],data['after']; result={'source_declaration_and_output_checks':checks,'reused_numeric_computations':False,'existing_value_count':len(e['all_existing_extracted_values'])}
    if owner_id in ['HV000114','HV000117']:
        cka=[];rsm=[];scalars=[];rd=next(x for x in src if x.get('side')=='after' and x['path'].endswith('README.md'));_,rb=binding(rd)
        result_source=next(x for x in src if x.get('side')=='after' and x['path'].endswith('coordinate_free_geometry_result.json'));_,gb=binding(result_source);g=json.loads(gb)
        after_pdf=next(x for x in e['complete_pdf_content'] if x['side']=='after');text=after_pdf['pages'][0]['text']
        lines=[line for line in text.splitlines() if re.match(r'^\s*(130M|370M|790M|1\.4B|2\.8B)\s+\d\.\d\d\s+',line)]
        assert len(lines)==5,('PDF_MATRIX_ROWS',lines)
        displayed=[[re.findall(r'(?<![\d.])\d\.\d{2}(?!\d)',line) for line in lines]]
        cells=[re.findall(r'(?<![\d.])\d\.\d{2}(?!\d)',line) for line in lines]
        assert all(len(x)==10 for x in cells),cells
        before_cells=None
        if owner_id=='HV000117':
            bt=next(x for x in e['complete_pdf_content'] if x['side']=='before')['pages'][0]['text']
            bl=[line for line in bt.splitlines() if re.match(r'^\s*(130M|370M|790M|1\.4B|2\.8B)\s+\d\.\d\d\s+',line)]
            before_cells=[re.findall(r'(?<![\d.])\d\.\d{2}(?!\d)',line) for line in bl]
            assert len(before_cells)==5 and all(len(x)==10 for x in before_cells)
        for fi,family in enumerate(['xg2','xg4']):
            for key,sourcekey in [('cka_matrix','cka_matrix'),('rsm_matrix','cosine_rsm_pearson_matrix')]:
                value=g['families'][family][sourcekey]
                assert b['extracted_values'][family+'.'+key]['value']==value
                assert b['plot_data']['coordinate_free_geometry'][family][key]==value
                if owner_id=='HV000117':assert a['plot_data']['coordinate_free_geometry'][family][key]==value
                for i,row in enumerate(value):
                    for k,v in enumerate(row):
                        rec={'family':family,'row_scale':g['scale_order'][i],'column_scale':g['scale_order'][k],'raw_value':v,'source':{**result_source,'json_pointer':'/families/'+family+'/'+sourcekey+'/'+str(i)+'/'+str(k)}}
                        if key=='cka_matrix':
                            rounded=f'{v:.2f}';assert cells[i][fi*5+k]==rounded
                            rec.update({'PDF_display':cells[i][fi*5+k],'expected_display':rounded,'README_6dp_present':f'{v:.6f}' in rb.decode(),'unchanged':owner_id=='HV000117'})
                            if before_cells is not None:
                                assert before_cells[i][fi*5+k]==rounded;rec['before_PDF_display']=before_cells[i][fi*5+k]
                            assert rec['README_6dp_present'];cka.append(rec)
                        else:rsm.append(rec)
        for scale in b['plot_data']['scale_order']:
            for metric in ['mu_k2','lambda1']:
                k=scale+'.'+metric;ex=b['extracted_values'][k];assert a['extracted_values'][k]['value']==ex['value']
                source=next(x for x in src if x.get('side')=='after' and x['path']==ex['source']);_,sb=binding(source)
                if ex['locator'].startswith('/'):value=ptr(json.loads(sb),ex['locator'])
                else:
                    tree=ast.parse(sb.decode());found=[]
                    for node in ast.walk(tree):
                        if isinstance(node,(ast.Assign,ast.AnnAssign)):
                            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                            if any(isinstance(t,ast.Name) and t.id==ex['locator'] for t in targets):found.append(ast.literal_eval(node.value))
                    assert len(found)==1;value=found[0]
                assert value==ex['value'];assert str(value) in rb.decode()
                scalars.append({'scale':scale,'metric':metric,'before':a['extracted_values'][k]['value'],'after':value,'source':source,'locator':ex['locator'],'unchanged':True,'after_figure_membership':ex['figure']})
        result.update({'cka_display_checks':cka,'rsm_source_checks':rsm,'displaced_scalar_checks':scalars,'coordinate_result_identity':result_source,'coordinate_result_metadata':{k:v for k,v in g.items() if k!='families'}})
        # Read complete frozen documentary provenance companions, without loading matrices.
        result['frozen_geometry_provenance']=[]
        for x in src:
            if x.get('side')=='after' and '/coordinate_free_geometry_v1/' in x['path'] and not x['path'].endswith('coordinate_free_geometry_result.json'):
                _,gb=binding(x);result['frozen_geometry_provenance'].append({'binding':x,'complete_payload':json.loads(gb)})
    else:
        csv_source=next(x for x in src if x.get('side')=='after' and x['path'].endswith('objective_bootstrap.csv'));_,cb=binding(csv_source);cr=list(csv.DictReader(io.StringIO(cb.decode())))
        intervals=[];means=[];associations=[]
        for scale in b['plot_data']['scale_order']:
            for objective in ['contra','vanilla']:
                av=a['plot_data']['objectives'][scale][objective];bv=b['plot_data']['objectives'][scale][objective]
                assert av==bv;means.append({'scale':scale,'objective':objective,'before':av,'after':bv,'unchanged':True})
                iv=b['plot_data']['objective_intervals'][scale][objective]
                rr=next(x for x in cr if x['scale']==scale and x['objective']=={'contra':'task','vanilla':'lm'}[objective])
                for k in ['point_mean','ci_low','ci_high']:assert iv[k]==float(rr[k])
                assert iv['point_mean']==bv
                z=iv['ci_low']<=0<=iv['ci_high'];assert rr['ci_contains_zero'].lower()==str(z).lower()
                intervals.append({'scale':scale,'objective':objective,**iv,'contains_zero':z,'CSV_row':cr.index(rr)+2,'N_pairs':int(rr['N_pairs'])})
            for metric in ['pearson','spearman','pair_sign_agreement_fraction']:
                av=a['plot_data']['objectives'][scale][metric];bv=b['plot_data']['objectives'][scale][metric];assert av==bv
                associations.append({'scale':scale,'metric':metric,'before':av,'after':bv,'unchanged':True})
        assert len(means)==10 and len(intervals)==10 and len(associations)==15
        assert [(x['scale'],x['objective']) for x in intervals if x['contains_zero']]==[('370M','vanilla'),('2.8B','contra')]
        result.update({'point_mean_checks':means,'interval_checks':intervals,'association_checks':associations,'bootstrap_csv_binding':csv_source,'complete_bootstrap_csv':cr,'added_interval_endpoints':20,'bootstrap_replicated':False})
    if owner_id in ['HV000117','HV000121']:
        source=next(x for x in src if x.get('side')=='after' and x['path'].endswith('objective_bootstrap.csv'));_,cb=binding(source);csv_rows=list(csv.DictReader(io.StringIO(cb.decode())))
        added=[]
        for op in e['manifest_operations']:
            if op['pointer'].startswith('/extracted_values/') and '.bootstrap.' in op['pointer']:
                ex=op['after'];match=re.fullmatch(r'CSV row (\d+); (\w+)',ex['locator']);assert match
                row_no=int(match[1]);field=match[2];rr=csv_rows[row_no-2]
                assert ex['source']==source['path'] and ex['value']==float(rr[field])
                added.append({'manifest_pointer':op['pointer'],'CSV_row':row_no,'CSV_field':field,'value':ex['value'],'status':'EXACT_STORED_SOURCE_CORRESPONDENCE'})
        assert len(added)==30
        result['all_30_bootstrap_manifest_additions']=added;result['bootstrap_source_identity']=source
        result['shared_Figure4_additions_are_Figure3_numeric_changes']=False
    e['numeric_verification']=result;write(owner_id+'_evidence.json',e)
    em=read(OUT/'evidence_manifest.json');em['commands'].extend(COMMANDS);write('evidence_manifest.json',em)
    print(owner_id+': source numeric correspondence checked independently; no scientific statistic recomputed.')
if __name__=='__main__':numerical(sys.argv[1])
