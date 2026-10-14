from verify import *
from pypdf.generic import ContentStream
import re

e=read(OUT/'HV000121_evidence.json');result=[]
for x in e['complete_pdf_content']:
    reader=PdfReader(ROOT/x['saved_file']);ops=ContentStream(reader.pages[0].get_contents(),reader).operations
    assert [list(map(float,a)) for a,o in ops if o==b'cm']==[[1,0,0,1,0,0]],'Unexpected PDF transform'
    rectangles=[];segments=[];ticks=[];m=None;tm=None
    for a,o in ops:
        if o==b're':rectangles.append(list(map(float,a)))
        if o==b'm':m=list(map(float,a))
        if o==b'l':segments.append([m,list(map(float,a))]);m=list(map(float,a))
        if o==b'Tm':tm=list(map(float,a))
        if o==b'Tj' and tm[5]>200 and re.fullmatch(r'-?0\.\d+',str(a[0])):ticks.append({'value':float(str(a[0])),'text_x':tm[4],'baseline_y':tm[5]})
    bars=[r for r in rectangles if r[2]==24.0 and r[0]>50 and r[1]>200];assert len(bars)==10
    axes={}
    for objective,left,right in [('contra',51,234),('vanilla',303,486)]:
        tt=[t for t in ticks if (t['text_x']<100)==(objective=='contra')]
        assert len(tt)>=4
        horizontal=[s for s in segments if s[0][0]==left and s[1][0]==right and s[0][1]==s[1][1]]
        grid=sorted(set(s[0][1] for s in horizontal));assert len(grid)==len(tt)
        # All tick baselines sit 3 PDF points below the horizontal grid line.
        paired=[]
        for t in tt:
            gy=min(grid,key=lambda y:abs(y-(t['baseline_y']+3)));assert abs(gy-t['baseline_y']-3)<0.001
            paired.append((t['value'],gy))
        z=next(y for v,y in paired if v==0);v,y=next((v,y) for v,y in paired if v>0);slope=(y-z)/v
        assert all(abs(y-(z+v*slope))<0.001 for v,y in paired)
        axes[objective]={'zero_y':z,'points_per_raw_unit':slope,'tick_grid_pairs':paired}
    drawn_means=[];drawn_intervals=[]
    for fi,objective in enumerate(['contra','vanilla']):
        ax=axes[objective];z=ax['zero_y'];slope=ax['points_per_raw_unit']
        for i,scale in enumerate(['130M','370M','790M','1.4B','2.8B']):
            r=bars[fi*5+i];raw=next(t['after'] for t in e['numeric_verification']['point_mean_checks'] if t['scale']==scale and t['objective']==objective)
            observed_y=r[1]+r[3] if raw>0 else r[1];expected_y=z+raw*slope;error=abs(observed_y-expected_y);assert error<0.001
            assert abs((r[1] if raw>0 else r[1]+r[3])-z)<0.001
            drawn_means.append({'scale':scale,'objective':objective,'raw_mean':raw,'rectangle':r,'observed_mean_y':observed_y,'expected_mean_y':expected_y,'absolute_error_points':error,'tolerance_points':0.001})
            vertical=[s for s in segments if s[0][0]==r[0]+r[2]/2==s[1][0]]
            if x['side']=='after':
                assert len(vertical)==1
                lo,hi=sorted([vertical[0][0][1],vertical[0][1][1]])
                source=next(t for t in e['numeric_verification']['interval_checks'] if t['scale']==scale and t['objective']==objective)
                for field,observed in [('ci_low',lo),('ci_high',hi)]:
                    expected_y=z+source[field]*slope;error=abs(expected_y-observed);assert error<0.001
                    caps=[s for s in segments if abs(s[0][1]-observed)<0.001 and abs(s[1][1]-observed)<0.001 and s[0][0]==r[0]+r[2]/2-5 and s[1][0]==r[0]+r[2]/2+5];assert len(caps)==1
                    drawn_intervals.append({'scale':scale,'objective':objective,'endpoint':field,'raw_value':source[field],'observed_endpoint_y':observed,'expected_endpoint_y':expected_y,'absolute_error_points':error,'tolerance_points':0.001,'cap_present':True})
                assert (lo<=z<=hi)==source['contains_zero']
            else:assert len(vertical)==0
    text=x['pages'][0]['text'];display_associations=[]
    for t in e['numeric_verification']['association_checks']:
        rendered=f'{t["after"]:+.4f}' if t['metric']!='pair_sign_agreement_fraction' else f'{t["after"]:.4f}'
        assert rendered in text;display_associations.append({**t,'display':rendered,'present_in_complete_PDF_text':True})
    result.append({'side':x['side'],'source_file':x['saved_file'],'source_sha256':x['sha256'],'axes_inferred_from_actual_ticks_and_grids':axes,'bar_mean_coordinates':drawn_means,'interval_endpoint_coordinates':drawn_intervals,'association_display_checks':display_associations,'whole_page_vector_content_parsed':True,'tolerance_explanation':'0.001 PDF point bounds decimal serialization error, not statistical tolerance.'})
e['numeric_verification']['actual_PDF_vector_checks']=result;write('HV000121_evidence.json',e)
print('HV000121: 20 before/after bar coordinates, 20 added whisker endpoints/caps, both zero crossings and 30 association displays checked.')
