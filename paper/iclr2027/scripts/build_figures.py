"""Physical-size vector artwork from pinned canonical evidence; no model/analysis.

Run with the bundled Python runtime. The optional isolated Matplotlib installation
is paper/iclr2027/tmp/final_figure_render_deps. Figma files are never read here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parents[1]
OUT = PAPER / "figures"
_VERSIONED_DEPS = PAPER / f"tmp/final_figure_render_deps_py{sys.version_info.major}{sys.version_info.minor}"
if _VERSIONED_DEPS.is_dir():
    sys.path.insert(0, str(_VERSIONED_DEPS))
sys.path.insert(0, str(PAPER / "scripts"))
os.environ["MPLCONFIGDIR"] = str(PAPER / "tmp/final_figure_mpl_config")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex
from matplotlib.patches import Rectangle, Circle
from matplotlib import font_manager
import pdfplumber
from pypdf import PdfReader
import figure_data_source as frozen

TEXT, SECONDARY, RULE, LIGHT_BG = "#1F2937", "#667381", "#D5DCE2", "#F5F7F9"
TASK_BLUE, TASK_LIGHT, LM_ORANGE, LM_LIGHT = "#0072B2", "#DCECF5", "#D55E00", "#F7E8DE"
W, MIN_FONT, PAD = 518.4, 11.4, .03
SCALES = frozen.SCALES
STEMS = ["fig1_study_overview", "fig2_130m_causal_foundation",
         "fig3_five_scale_recurrence_reorganization", "fig4_objective_conditioned_functionalization"]
STAGES = ["Controlled contrasts", "Native-state localization", "Geometry specificity",
          "Matched necessity", "Restoration sufficiency", "Task-margin consequence"]
CHAINS = ["Transport", "Specificity", "Necessity", "Restoration sufficiency"]
NORM = Normalize(vmin=.30, vmax=1.00, clip=True)
CMAP = LinearSegmentedColormap.from_list("causal_blue", [LIGHT_BG, TASK_BLUE], N=65536)
for filename in ["arial.ttf", "arialbd.ttf"]:
    path = Path("C:/Windows/Fonts") / filename
    assert path.is_file(), f"Required locally verified font missing: {path}"
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({"font.family": "Arial", "font.size": MIN_FONT,
    "svg.fonttype": "none", "svg.hashsalt": "canonical-publication-figures",
    "pdf.fonttype": 42, "pdf.compression": 9, "axes.unicode_minus": True})


def luminance(rgb):
    def linear(c):
        return c/12.92 if c <= .04045 else ((c+.055)/1.055)**2.4
    return sum(w*linear(c) for w,c in zip([.2126,.7152,.0722],rgb[:3]))


def color(value):
    return to_hex(CMAP(NORM(value)))


class Figure:
    def __init__(self, stem, height):
        self.stem, self.height = stem, height
        self.fig = plt.figure(figsize=(7.2,height/72),dpi=144,facecolor="white")
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.set(xlim=(0,W),ylim=(height,0))
        self.ax.set_axis_off()
        self.texts, self.table_rules, self.payload, self.gid_payload = [],[],{},{}

    def text(self,x,y,text,size=MIN_FONT,bold=False,color=TEXT,ha="left",va="baseline",rotation=0,gid=None):
        assert size >= MIN_FONT
        a=self.ax.text(x,y,str(text),fontsize=size,fontweight="bold" if bold else "normal",
            color=color,ha=ha,va=va,rotation=rotation,clip_on=False,gid=gid,zorder=5)
        self.texts.append(a)
        return a

    def lines(self,x,y,lines,gap=14,**kw):
        for i,line in enumerate(lines): self.text(x,y+i*gap,line,**kw)

    def line(self,x1,y1,x2,y2,color=RULE,width=.5,gid=None):
        a,=self.ax.plot([x1,x2],[y1,y2],color=color,lw=width,solid_capstyle="butt",zorder=2,gid=gid)
        return a

    def rect(self,x,y,w,h,fill=LIGHT_BG,edge=None,lw=.5,gid=None):
        a=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge or "none",linewidth=lw,gid=gid,zorder=1)
        self.ax.add_patch(a)
        return a

    def dot(self,x,y,r=1.1,fill=TASK_BLUE,alpha=1,edge=None,lw=.6,gid=None):
        a=Circle((x,y),r,facecolor=fill,edgecolor=edge or "none",linewidth=lw,alpha=alpha,gid=gid,zorder=3)
        self.ax.add_patch(a)
        return a

    def arrow(self,x1,y,x2,color=TASK_BLUE):
        self.line(x1,y,x2,y,color,.7)
        self.line(x2-3,y-2,x2,y,color,.7)
        self.line(x2-3,y+2,x2,y,color,.7)

    def panel(self,letter,title,y,x=4):
        self.text(x,y,letter,13,True,TASK_BLUE,gid=f"panel-{letter}")
        self.text(x+18,y,title,12.5,True,gid=f"title-{letter}")

    def save(self):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        boxes=[]
        for a in self.texts:
            bb=a.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            x0,x1=sorted([bb.x0,bb.x1]); y0,y1=sorted([bb.y0,bb.y1])
            assert x0>=0 and x1<=W and y0>=0 and y1<=self.height,(self.stem,a.get_text(),(x0,y0,x1,y1))
            boxes.append((x0,y0,x1,y1,a.get_text()))
        collisions=[]
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                if min(a[2],b[2])-max(a[0],b[0])>.05 and min(a[3],b[3])-max(a[1],b[1])>.05:
                    collisions.append([a[4],b[4]])
        assert not collisions,(self.stem,"text overlap",collisions)
        clearances=[]
        for x0,y,x1 in self.table_rules:
            for a in boxes:
                if a[2]>x0 and a[0]<x1:
                    gap=max(a[1]-y,y-a[3])
                    assert gap>=4.5,("Table rule/text clearance",y,a,gap)
                    clearances.append(gap)
        # Tight export, source-sized fonts, 0.03-inch padding. No giant design canvas.
        for ext in ["svg","pdf","png"]:
            metadata={"Creator":"Deterministic scientific figure renderer"}
            if ext=="svg": metadata["Date"]=None
            if ext=="pdf": metadata.update({"CreationDate":None,"ModDate":None,"Author":""})
            self.fig.savefig(OUT/(self.stem+"."+ext),format=ext,dpi=400,
                bbox_inches="tight",pad_inches=PAD,facecolor="white",metadata=metadata)
        # Raw frozen numbers are attached to their editable vector groups, not rounded.
        svg=OUT/(self.stem+".svg")
        ET.register_namespace("","http://www.w3.org/2000/svg")
        tree=ET.parse(svg); root=tree.getroot()
        for gid,attrs in self.gid_payload.items():
            node=root.find(f".//*[@id='{gid}']")
            assert node is not None,gid
            for key,value in attrs.items(): node.set("data-"+key,str(value))
        meta=ET.SubElement(root,"{http://www.w3.org/2000/svg}metadata",id="frozen-scientific-values")
        meta.text=json.dumps(self.payload,ensure_ascii=False,separators=(",",":"),allow_nan=False)
        tree.write(svg,encoding="utf-8",xml_declaration=True)
        page=PdfReader(OUT/(self.stem+".pdf")).pages[0]
        source_width=float(page.mediabox.width)
        with pdfplumber.open(OUT/(self.stem+".pdf")) as pdf:
            # pdfplumber calls a rotated glyph's vertical advance its size. Use
            # upright text here; every rotated artist is independently font-audited.
            min_actual=min(c["size"] for c in pdf.pages[0].chars if c["upright"])
        assert abs(min_actual-MIN_FONT)<1e-5
        effective=min_actual*396/source_width
        assert effective>=8.5
        assert not page.images
        audit={"source_canvas_inches":[7.2,self.height/72],
               "export_inches":[source_width/72,float(page.mediabox.height)/72],
               "minimum_source_font_pt":min_actual,"minimum_effective_font_pt":effective,
               "source_panel_letter_pt":13,"source_panel_title_pt":12.5,
               "text_bbox_intersections":len(collisions),"table_rule_intersections":0,
               "minimum_table_rule_clearance_source_pt":min(clearances) if clearances else None,
               "crop_padding_inches":PAD,"pdf_raster_image_count":0}
        plt.close(self.fig)
        return audit


def figure1(d):
    f=Figure(STEMS[0],348)
    f.panel("A","Deep causal establishment at Mamba-130M",15)
    f.text(208,37,"Native susceptibility mechanism",color=SECONDARY,ha="center")
    f.text(514,37,"Task-functional bridge",color=TASK_BLUE,ha="right")
    f.line(4,44,422,44); f.line(436,44,514,44,TASK_BLUE)
    xs=[42,128,214,300,386,472]
    for x,stage in zip(xs,STAGES):
        a,b=stage.rsplit(" ",1)
        f.text(x,62,a,11.4,True,ha="center")
        f.text(x,76,b,ha="center")
    f.arrow(xs[0],86,514)
    for x in xs: f.dot(x,86,1.55,"white",edge=TASK_BLUE,lw=.8)
    f.line(4,103,514,103)
    f.panel("B","Five scales: causal role recurs; geometry is only partially conserved",122)
    for i,scale in enumerate(SCALES):
        x=46+106*i
        f.text(x,148,scale,13,True,ha="center")
        f.text(x,165," / ".join(d["planes"][scale]),color=SECONDARY,ha="center",gid=f"ranks-{i}")
        f.dot(x-29,179.3,1.05)
        f.text(x-24,183,"Supported",11.4,True,TASK_BLUE)
    f.text(4,204,"Selected / control are scale-local ranks; equal rank numbers do not imply semantic homology.",color=SECONDARY)
    f.line(4,218,514,218)
    f.panel("C","Objective-conditioned functional readout",237)
    f.text(4,263,"Within each scale",color=SECONDARY)
    f.lines(4,279,["same frozen native","Mamba substrate"],gap=15,bold=True)
    f.line(117,278,141,278,SECONDARY,.7)
    f.line(141,253,141,297,SECONDARY,.7)
    for key,y,label,c in [("contra",258,"Structured downstream task objective",TASK_BLUE),
                           ("vanilla",301,"Vanilla Mamba pretrained next-token LM objective",LM_ORANGE)]:
        f.arrow(141,y-5,158,c)
        f.line(163,y-12,163,y+23,c,1)
        f.text(171,y,label,11.4,True,c)
        f.text(171,y+20,"Point-estimate signs:",color=SECONDARY)
        for i,scale in enumerate(SCALES):
            f.text(329+42*i,y+21,"+" if d["objectives"][scale][key]>0 else "−",13,True,c,ha="center",gid=f"{key}-sign-{i}")
    f.line(4,331,514,331)
    f.text(W/2,344,"Causal role | Native geometry | Objective-conditioned readout",11.4,True,ha="center")
    f.payload={"stages":STAGES,"scales":SCALES,"planes":d["planes"],"supported":d["causal_supported"],
               "signs":{k:[1 if d["objectives"][s][k]>0 else -1 for s in SCALES] for k in ["contra","vanilla"]}}
    return f.save()


def figure2(d):
    f=Figure(STEMS[1],372)
    f.panel("A","Prospective evidence chain at 130M",15)
    f.text(22,34,"Four independent holdouts test distinct causal requirements.",color=SECONDARY)
    for i,(title,n) in enumerate(zip(CHAINS,d["chain_n"])):
        x=4+130*i
        f.line(x,44,min(x+120,514),44)
        f.lines(x,60,title.split() if i==3 else [title],bold=True,gap=14)
        f.text(x,90,"Supported",11.4,True,TASK_BLUE)
        f.text(x,107,f"N = {n}",color=SECONDARY)
    f.line(4,117,514,117)
    f.panel("B","Behavioral restoration",139)
    f.lines(4,168,["selected restoration","− matched control"],gap=16)
    f.text(4,210,"Mean task-margin shift")
    f.text(4,247,f"{d['behavior']:+.5f}",31,True,TASK_BLUE,gid="behavior-mean")
    f.text(4,271,f"N = {d['behavior_n']} paired items",color=SECONDARY)
    f.lines(4,298,["Continuous margin","consequence; not an","accuracy-improvement claim."],gap=15,color=SECONDARY)
    f.panel("C","Local readout tracks behavioral effect",139,x=174)
    for x,name in [(181,"Pearson"),(285,"Spearman"),(400,"Sign agreement")]:
        f.text(x,164,name,color=SECONDARY)
        f.text(x,181,f"{d['metrics'][name]:.3f}",12,True,gid="stat-"+name.replace(" ","-"))
    x,y,w,h=218,199,294,113
    xmin,xmax,ymin,ymax=-.014,.022,-.03,.08
    xp=lambda v:x+(v-xmin)/(xmax-xmin)*w
    yp=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
    f.line(x,y,x,y+h); f.line(x,y+h,x+w,y+h)
    f.line(xp(0),y,xp(0),y+h,width=.7)
    f.line(x,yp(0),x+w,yp(0),width=.7)
    for v in [-.01,0,.01,.02]: f.text(xp(v),330,f"{v:g}".replace("-","−"),color=SECONDARY,ha="center")
    for v in [-.02,0,.04,.08]: f.text(x-6,yp(v),f"{v:g}".replace("-","−"),color=SECONDARY,ha="right",va="center")
    for i,row in enumerate(d["pairs"]):
        assert xmin<=row["Delta_L"]<=xmax and ymin<=row["D_BEH"]<=ymax
        f.dot(xp(row["Delta_L"]),yp(row["D_BEH"]),1.05,alpha=.70,gid=f"scatter-pair-{i:03}")
        f.gid_payload[f"scatter-pair-{i:03}"]={"x":repr(row["Delta_L"]),"y":repr(row["D_BEH"]),
            "cx":repr(xp(row["Delta_L"])),"cy":repr(yp(row["D_BEH"]))}
    f.text(x+w/2,347,"Local task readout",ha="center",gid="scatter-x-label")
    f.text(172,y+h/2,"Behavioral margin shift",ha="center",va="center",rotation=90,gid="scatter-y-label")
    f.text(4,367,"All 300 pairs retained; associations are frozen descriptive results.",color=SECONDARY)
    f.payload={"stages":CHAINS,"holdout_n":d["chain_n"],"mean":d["behavior"],"n":d["behavior_n"],
               "statistics":d["metrics"],"pairs":d["pairs"],"axis":{"x":x,"y":y,"w":w,"h":h,"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax}}
    return f.save()


def figure3(d):
    f=Figure(STEMS[2],334)
    f.panel("A","The causal role recurs across all five Mamba-1 scales",15)
    xs=[167,247,327,407,487]
    f.rect(4,25,510,21)
    for x,s in zip(xs,SCALES): f.text(x,40,s,11.4,True,ha="center")
    for y,label in [(60,"Selected / control"),(80,"Causal role")]: f.text(8,y,label,bold=True)
    for i,(x,s) in enumerate(zip(xs,SCALES)):
        f.text(x,60,"/".join(d["planes"][s]),ha="center",gid=f"ranks-{i}")
        f.text(x,80,"Supported",11.4,True,TASK_BLUE,ha="center")
    f.line(4,91,514,91)
    f.text(4,109,"Common scale-local contrast: selected restoration minus coefficient-matched control.",color=SECONDARY)
    f.text(4,123,"Historical endpoint labels and inferential-family details are retained in Appendix A.4.",color=SECONDARY)
    for letter,family,left in [("B","xg2",4),("C","xg4",269)]:
        f.panel(letter,family.upper()+" centered linear CKA",146,x=left)
        x,y,cw,ch=left+54,176,36,21
        for i,s in enumerate(SCALES):
            f.text(x+(i+.5)*cw,167,s,ha="center",color=SECONDARY)
            f.text(x-7,y+(i+.5)*ch,s,ha="right",va="center",color=SECONDARY)
        for i,row in enumerate(d["coordinate_free_geometry"][family]["cka_matrix"]):
            for j,value in enumerate(row):
                rgba=CMAP(NORM(value)); fill=color(value)
                gid=f"{family}-cell-{i}-{j}"
                f.rect(x+j*cw,y+i*ch,cw,ch,fill,edge="white",lw=.7,gid=gid)
                f.gid_payload[gid]={"value":repr(value),"norm-min":.30,"norm-max":1.00,"fill":fill}
                f.text(x+(j+.5)*cw,y+(i+.5)*ch,f"{value:.2f}",color="white" if luminance(rgba)<.34 else TEXT,
                       ha="center",va="center",gid=f"{family}-value-{i}-{j}")
    f.lines(4,301,["Same 300 response-blind pairs per family. CKA = 1 denotes identical centered sample",
        "geometry; off-diagonal values show partial, not invariant, cross-scale geometry.",
        "Cosine-RSM: Appendix A.2."],gap=14,color=SECONDARY)
    f.payload={"scales":SCALES,"planes":d["planes"],"supported":d["causal_supported"],
               "matrices":{k:d["coordinate_free_geometry"][k]["cka_matrix"] for k in ["xg2","xg4"]},
               "normalization":{"vmin":.30,"vmax":1.00,"shared":True}}
    return f.save()


def figure4(d):
    f=Figure(STEMS[3],422)
    for letter,key,left,title,subtitle,c,lo,hi,ticks in [
        ("A","contra",4,"Downstream task objective","Mean ΔL (forward-equivalent)",TASK_BLUE,-.0035,.0065,[-.002,0,.002,.004,.006]),
        ("B","vanilla",269,"Vanilla next-token LM objective","Mean TASK_MATCHED readout",LM_ORANGE,-.075,.03,[-.06,-.04,-.02,0,.02])]:
        f.panel(letter,title,15,x=left)
        f.text(left+18,35,subtitle,color=SECONDARY)
        x,y,w,h=left+41,57,197,117
        yp=lambda v:y+h-(v-lo)/(hi-lo)*h
        for tick in ticks:
            f.line(x,yp(tick),x+w,yp(tick),RULE,.45)
            lab="0" if tick==0 else (f"{tick:.3f}" if key=="contra" else f"{tick:.2f}")
            f.text(x-6,yp(tick),lab.replace("-","−"),color=SECONDARY,ha="right",va="center")
        f.line(x,yp(0),x+w,yp(0),SECONDARY,.95)
        for i,s in enumerate(SCALES):
            cx=x+(i+.5)*w/5
            values=d["objective_intervals"][s][key]
            mean,low,high=[values[a] for a in ["point_mean","ci_low","ci_high"]]
            f.rect(cx-.25*w/5,min(yp(mean),yp(0)),.5*w/5,abs(yp(mean)-yp(0)),c,gid=f"{key}-bar-{i}")
            f.line(cx,yp(low),cx,yp(high),TEXT,1.1,gid=f"{key}-interval-{i}")
            for end in [low,high]: f.line(cx-3,yp(end),cx+3,yp(end),TEXT,1.1)
            f.text(cx,193,s,ha="center")
            f.text(cx,231,"+" if mean>0 else "−",13,True,c,ha="center",gid=f"{key}-sign-{i}")
            f.gid_payload[f"{key}-bar-{i}"]={"mean":repr(mean),"ci-low":repr(low),"ci-high":repr(high),
                "y":repr(min(yp(mean),yp(0))),"height":repr(abs(yp(mean)-yp(0)))}
            f.gid_payload[f"{key}-interval-{i}"]={"low-y":repr(yp(low)),"high-y":repr(yp(high))}
        f.text(x+w/2,212,"Point-estimate sign",color=SECONDARY,ha="center")
    f.text(4,253,"Separate y-scales; bars and signs are point means, whiskers are percentile 95% pair intervals.",color=SECONDARY)
    f.line(4,267,514,267)
    f.panel("C","Pair-level associations remain mixed",287)
    xs=[182,259,336,413,490]
    f.rect(4,296,510,19)
    for x,s in zip(xs,SCALES): f.text(x,310,s,11.4,True,ha="center")
    for y,label,key in [(334,"Pearson","pearson"),(358,"Spearman","spearman"),(382,"Pair sign agreement","pair_sign_agreement_fraction")]:
        f.text(8,y,label,bold=True)
        for x,s in zip(xs,SCALES):
            value=d["objectives"][s][key]
            lab=(f"{value:.4f}" if key.startswith("pair") else f"{value:+.4f}").replace("-","−")
            f.text(x,y,lab,ha="center",gid=f"association-{key}-{s}")
            f.gid_payload[f"association-{key}-{s}"]={"value":repr(value)}
    for y in [318,342,366,390]:
        f.line(4,y,514,y,width=.45)
        f.table_rules.append((4,y,514))
    f.lines(4,408,["All five scales are shown descriptively; the 130M LM cell is the later completeness",
                  "extension (Appendix A.6)."],gap=14,color=SECONDARY)
    f.height=428
    f.fig.set_size_inches(7.2,428/72)
    f.ax.set_ylim(428,0)
    f.payload={"scales":SCALES,"intervals":d["objective_intervals"],"objectives":d["objectives"],
               "axes":{"contra":{"x":45,"y":57,"w":197,"h":117,"lo":-.0035,"hi":.0065},
                       "vanilla":{"x":310,"y":57,"w":197,"h":117,"lo":-.075,"hi":.03}}}
    return f.save()


def node(root,gid):
    found=root.find(f".//*[@id='{gid}']")
    assert found is not None,gid
    return found


def text_of(root,gid):
    return "".join(node(root,gid).itertext()).strip()


def validate_outputs(d,audits):
    roots=[ET.parse(OUT/(stem+".svg")).getroot() for stem in STEMS]
    for i,root in enumerate(roots):
        assert root.findall(".//{http://www.w3.org/2000/svg}text")
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
        assert not re.search(r"(?i)\b[0-9a-f]{40}\b",(OUT/(STEMS[i]+".svg")).read_text(encoding="utf-8"))
    p1=json.loads(node(roots[0],"frozen-scientific-values").text)
    assert p1["stages"]==STAGES and p1["planes"]==d["planes"]
    assert p1["signs"]=={"contra":[1,1,1,-1,-1],"vanilla":[-1,-1,-1,1,1]}
    for fignum in [0,2]:
        for i,s in enumerate(SCALES): assert re.sub(r"\s+","",text_of(roots[fignum],f"ranks-{i}"))=="/".join(d["planes"][s])
        with pdfplumber.open(OUT/(STEMS[fignum]+".pdf")) as pdf:
            assert pdf.pages[0].extract_text().count("Supported")==5
    marks=[e for e in roots[1].iter() if e.get("id","").startswith("scatter-pair-")]
    assert len(marks)==300
    for i,(e,row) in enumerate(zip(marks,d["pairs"])):
        assert e.get("id")==f"scatter-pair-{i:03}"
        assert float(e.get("data-x"))==row["Delta_L"] and float(e.get("data-y"))==row["D_BEH"]
        assert float(e.get("data-cx"))==218+(row["Delta_L"]+.014)/.036*294
        assert abs(float(e.get("data-cy"))-(312-(row["D_BEH"]+.03)/.11*113))<1e-10
    assert text_of(roots[1],"behavior-mean")==f"{d['behavior']:+.5f}"
    for name,v in d["metrics"].items(): assert text_of(roots[1],"stat-"+name.replace(" ","-"))==f"{v:.3f}"
    checked=0
    for family in ["xg2","xg4"]:
        for i,row in enumerate(d["coordinate_free_geometry"][family]["cka_matrix"]):
            for j,value in enumerate(row):
                rect=node(roots[2],f"{family}-cell-{i}-{j}")
                assert float(rect.get("data-value"))==value
                assert rect.get("data-fill")==color(value)
                assert text_of(roots[2],f"{family}-value-{i}-{j}")==f"{value:.2f}"
                assert float(rect.get("data-norm-min"))==.30 and float(rect.get("data-norm-max"))==1.
                checked+=1
    levels=[.30+i*.70/4096 for i in range(4097)]
    light=[luminance(matplotlib.colors.to_rgb(color(v))) for v in levels]
    assert all(a>=b for a,b in zip(light,light[1:]))
    assert luminance(CMAP(NORM(.74)))<luminance(CMAP(NORM(.44)))
    assert luminance(CMAP(NORM(.66)))<luminance(CMAP(NORM(.39)))
    crossings=[]
    for key in ["contra","vanilla"]:
        for i,s in enumerate(SCALES):
            expected=d["objective_intervals"][s][key]
            bar=node(roots[3],f"{key}-bar-{i}")
            assert [float(bar.get("data-"+k)) for k in ["mean","ci-low","ci-high"]]==[expected[k] for k in ["point_mean","ci_low","ci_high"]]
            if expected["ci_low"]<=0<=expected["ci_high"]: crossings.append([s,key])
            assert text_of(roots[3],f"{key}-sign-{i}")==("+" if expected["point_mean"]>0 else "−")
            assert text_of(roots[0],f"{key}-sign-{i}")==text_of(roots[3],f"{key}-sign-{i}")
    assert crossings==[["2.8B","contra"],["370M","vanilla"]]
    for s in SCALES:
        for key in ["pearson","spearman","pair_sign_agreement_fraction"]:
            e=node(roots[3],f"association-{key}-{s}")
            value=d["objectives"][s][key]
            assert float(e.get("data-value"))==value
            expected=(f"{value:.4f}" if key.startswith("pair") else f"{value:+.4f}").replace("-","−")
            assert "".join(e.itertext()).strip()==expected
    return {"status":"PASS","scatter_mark_count":300,"scatter_coordinates_exact":300,
            "figure3_rank_cells_exact":5,"figure3_printed_cka_cells_exact":checked,
            "figure3_full_precision_cka_cells_exact":checked,"shared_normalization":[.30,1.],
            "monotonic_colormap_samples":4097,"monotonic_colormap":"PASS",
            "figure4_means_exact":10,"figure4_ci_endpoints_exact":20,"figure4_association_cells_exact":15,
            "figure4_zero_crossing_cells":crossings,"figure4_table_rule_intersections":0,
            "figure_audits":dict(zip(STEMS,audits)),"scientific_analysis_executed":False,
            "model_tokenizer_training_evaluation_gpu_execution":False}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, d = frozen.load_data()
    audits = [fn(d) for fn in [figure1, figure2, figure3, figure4]]
    result = validate_outputs(d, audits)
    result["production_figures_unchanged"] = False
    result["outputs"] = {
        stem + ext: hashlib.sha256((OUT / (stem + ext)).read_bytes()).hexdigest()
        for stem in STEMS for ext in [".svg", ".pdf", ".png"]
    }
    (OUT / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
