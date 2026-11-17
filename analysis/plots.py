"""Publication figures from the verified frozen-result reanalysis."""
from pathlib import Path
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from reanalyze import DATA,ROOT, DATASETS, Q0, cost, read

PAPER=ROOT
OUT=PAPER/"figures"
PREVIEW=ROOT/"figures"
FONT = "Times New Roman" if any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
plt.rcParams.update({"font.family":FONT,"font.size":9,
                     "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
                     "legend.fontsize":8.5,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.linewidth":.65,
                     "pdf.fonttype":42,"ps.fonttype":42,"savefig.dpi":180})
COLORS={"HW":"#425B6A","RShare":"#138D80","Real":"#2868A8",
        "FM4":"#DA8A26","DDPM16":"#8E69B3","Other":"#8A949E",
        "HW-RL":"#217CB3","Bitwise":"#BA4B58","Classical":"#425B6A"}


def family(method):
    if method=="HW-RL":return "HW-RL"
    if method in {"Bit-Gauss","Bitwise-RL"}:return "Bitwise"
    if method.startswith("HW-template"):return "HW"
    if method=="RShare-T":return "RShare"
    if method.startswith("FM4"):return "FM4"
    if method.startswith("DDPM16"):return "DDPM16"
    if method=="Real":return "Real"
    return "Other"


def get(groups,b,q,m):
    gs=[g for g in groups if g["dataset"]==b and g["Q"]==q and g["method"]==m]
    assert len(gs)==1,(b,q,m)
    return gs[0]


def label(g):
    name={"HW-template-fixed":"HW","HW-template-cv":"HW-CV",
          "RShare-T":"RShare","FM4-6000":"FM4","DDPM16-6000":"DDPM16",
          "Real":"Real", "HW-RL":"HW-RL", "Bit-Gauss":"Bit-Gauss"}.get(g["method"],g["name"])
    return f"{name} {g['Q']/1000:g}k"


def finish(fig,name):
    OUT.mkdir(exist_ok=True);PREVIEW.mkdir(exist_ok=True)
    fig.savefig(OUT/(name+".pdf"),metadata={"CreationDate":None,"Creator":"Matplotlib"})
    fig.savefig(PREVIEW/(name+".png"),dpi=180)
    plt.close(fig)


def grid(ax):
    ax.grid(color="#DDE2E7",lw=.45,alpha=.75)
    ax.set_axisbelow(True)


def recovery(groups,curves):
    fig,axes=plt.subplots(2,2,figsize=(7.16,3.35),sharex=True)
    fig.subplots_adjust(left=.075,right=.965,bottom=.22,top=.75,hspace=.38,wspace=.22)
    series=[("HW-template-fixed",5000,"HW Gaussian 5k","--"),
            ("HW-RL",5000,"HW-RL 5k","-"),
            ("Bit-Gauss",5000,"Bit-Gauss 5k",":"),
            ("RShare-T",5000,"RShare 5k","-"),
            ("FM4-6000",2000,"FM4 2k","-"),
            ("DDPM16-6000",2000,"DDPM16 2k","-")]
    xs=np.arange(1,10001)
    for j,b in enumerate(["D1","AES_GPU"]):
        for m,q,text,ls in series:
            g=get(groups,b,q,m);col=COLORS[family(m)]
            for i,metric in enumerate(["SR","GE"]):
                axes[i,j].plot(xs,curves[g["id"]+"/"+metric],color=col,ls=ls,lw=1.05,
                               alpha=.85 if ls=="--" else 1.)
            if g["A95"] is not None:
                a=g["A95"]
                axes[0,j].plot(a,curves[g["id"]+"/SR"][a-1],"o",ms=3,color=col)
        axes[0,j].axhline(.95,color="#7F8790",ls=":",lw=.75)
        axes[0,j].set_ylim(-.03,1.04);axes[0,j].set_yticks([0,.5,1])
        axes[1,j].text(.5,-.70,f"({chr(97+j)}) {b.replace('AES_','')} recovery",
                       transform=axes[1,j].transAxes,ha="center",va="top",fontsize=9.2)
        axes[1,j].set_yscale("symlog",linthresh=1,linscale=.5)
        axes[1,j].set_ylim(-.01,256);axes[1,j].set_yticks([0,1,10,100],['0','1','10','100'])
        axes[1,j].axhline(1,color="#AAB1B8",ls=":",lw=.6)
        axes[1,j].set_xlabel("Attack traces A")
        for ax in axes[:,j]:
            ax.set_xscale("log");ax.set_xlim(10,10000)
            ax.set_xticks([10,100,1000,10000],["10","100","1,000","10,000"])
            grid(ax)
    axes[0,0].set_ylabel("Success rate")
    axes[1,0].set_ylabel("Mean key rank (GE)")
    handles=[Line2D([],[],color=COLORS[family(m)],ls=ls,lw=1.2,label=t) for m,q,t,ls in series]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.84),ncol=6,
               frameon=False,columnspacing=.75,handlelength=1.35,fontsize=7.2)
    finish(fig,"joint-recovery")


def frontier(groups,bootstrap,second_groups):
    fig,axes=plt.subplots(2,2,figsize=(7.16,3.95))
    fig.subplots_adjust(left=.09,right=.97,bottom=.20,top=.75,hspace=.58,wspace=.24)
    prices=np.geomspace(.01,1000,281)
    edges=np.geomspace(prices[0]/np.sqrt(prices[1]/prices[0]),
                       prices[-1]*np.sqrt(prices[1]/prices[0]),len(prices)+1)
    P,T=np.meshgrid(prices,prices)
    region=lambda m: "Classical" if family(m) in {"HW","HW-RL","Bitwise"} else family(m)
    family_ids={m:i for i,m in enumerate(["Classical","RShare","Real","FM4","DDPM16","Other"])}
    cmap=ListedColormap([tuple(.70+.30*np.array(to_rgb(COLORS[m]))) for m in family_ids])
    plot_stats={};sensitivity={}
    for j,(b,neural) in enumerate([("D1","DDPM16-6000"),("AES_GPU","FM4-6000")]):
        bg=[g for g in groups if g["dataset"]==b and g["A95"] is not None]
        top=axes[0,j];bottom=axes[1,j]
        focal=get(groups,b,2000,neural)
        all_lines=np.array([g["C"]+(g["Q"]-Q0)*prices/1000+g["A95"]*.001 for g in bg])
        for fam in ["Classical","RShare"]:
            idx=[i for i,g in enumerate(bg) if region(g["method"])==fam]
            top.plot(prices,all_lines[idx].min(0),color=COLORS[fam],lw=1.35)
        top.plot(prices,np.full_like(prices,cost(focal,0,.001)),color=COLORS[family(neural)],lw=1.35)
        top.plot(prices,all_lines.min(0),color="#142C3C",lw=2.0,zorder=5)
        second=[g for g in second_groups if g["dataset"]==b and g["A95"] is not None]
        if second:
            second_lines=np.array([g["C"]+(g["Q"]-Q0)*prices/1000+g["A95"]*.001 for g in second])
            top.plot(prices,second_lines.min(0),color="#526B80",lw=1.3,ls="--",zorder=6)
        boot=bootstrap[b]
        crossing=boot["point"]*1000;low,high=np.array(boot["ci"])*1000
        top.axvspan(low,high,color=COLORS[family(neural)],alpha=.13,lw=0)
        y=cost(focal,0,.001)
        top.plot(crossing,y,"o",color="#142C3C",ms=3.2,zorder=6)
        top.annotate(f"{crossing:.2f} ms",xy=(crossing,y),xytext=(1.8,160),
                     fontsize=9.3,ha="left",arrowprops={"arrowstyle":"-","lw":.7,"color":"#384A59"})
        top.text(45,y*.64,label(focal).replace(" ","\n",1),ha="center",va="top",fontsize=9.1,color=COLORS[family(neural)])
        top.set_yscale("log");top.set_ylim(.5,5000);top.set_yticks([1,10,100,1000],["1","10","100","1k"])
        top.text(.5,-.34,f"({chr(97+j)}) {b.replace('AES_','')}: target price = 1 ms",
                 transform=top.transAxes,ha="center",va="top",fontsize=9.2)
        if j==0:top.set_ylabel("Total cost (s)",fontsize=9.5)
        grid(top)
        times=np.array([g["C"]+(g["Q"]-Q0)*P/1000+g["A95"]*T/1000 for g in bg])
        chosen=np.argmin(times,axis=0)
        fam=np.array([family_ids[region(g["method"])] for g in bg])[chosen]
        bottom.pcolormesh(edges,edges,fam,cmap=cmap,vmin=-.5,vmax=len(family_ids)-.5,
                          shading="flat",rasterized=True)
        segments=[]
        yy,xx=np.where(fam[:,:-1]!=fam[:,1:])
        segments.extend([[(edges[x+1],edges[y]),(edges[x+1],edges[y+1])] for y,x in zip(yy,xx)])
        yy,xx=np.where(fam[:-1,:]!=fam[1:,:])
        segments.extend([[(edges[x],edges[y+1]),(edges[x+1],edges[y+1])] for y,x in zip(yy,xx)])
        bottom.add_collection(LineCollection(segments,colors="#566671",linewidths=.55,alpha=.7))
        bottom.plot(prices,prices,color="#263746",lw=1.1,ls="--",alpha=.85)
        cw_rates={"F0":1000/9.21,"F1":1000/11.23,"F2":1000/10.87,"F3":1000/12.43,"XMEGA":1000/5.21}
        for rate in cw_rates.values():
            bottom.plot(rate,rate,marker="*",ms=7,color="#B22222",markeredgecolor="white",markeredgewidth=.5,zorder=7,clip_on=False)
        bottom.text(.018,.91,"CW range: 80--192 ms/trace",transform=bottom.transAxes,
                    fontsize=8.3,ha="left",va="center",color="#8B1F24",zorder=8)
        bottom.text(.98,.05,r"sim. $\tau_p\gg\tau_t$",
                    transform=bottom.transAxes,fontsize=5.8,ha="right",va="bottom",
                    color="#8B1F24",zorder=8)
        bottom.text(.48,.84,"same workflow",
                    transform=bottom.transAxes,fontsize=6.2,ha="center",va="center",
                    color="#263746",alpha=.85,zorder=8)
        bottom.axhline(1,color="#46515A",lw=.8,ls=":")
        positions={"Classical":(.3,.15),family(neural):(38,.12)}
        if b=="D1":positions["RShare"]=(2,35)
        for name,(x,y) in positions.items():
            winner_at_label=min(bg,key=lambda h:cost(h,x/1000,y/1000))
            assert region(winner_at_label["method"])==name
            text="Likelihood models" if name=="Classical" else name
            neural_label=name in {"FM4","DDPM16"}
            if neural_label:text+="\nQ = 2k"
            bottom.text(x,y,text,ha="center",va="center",fontsize=9.2 if neural_label else 10,
                        fontweight="bold",color="#253742")
        bottom.set_yscale("log");bottom.set_ylim(.01,1000)
        ticks=[.01,.1,1,10,100,1000];ticklabels=[".01",".1","1","10","100","1k"]
        bottom.set_yticks(ticks,ticklabels)
        bottom.text(.5,-.48,f"({chr(99+j)}) {b.replace('AES_','')}: primary-block frontier",
                    transform=bottom.transAxes,ha="center",va="top",fontsize=9.2)
        if j==0:bottom.set_ylabel("Target capture price (ms/trace)",fontsize=9.5)
        bottom.set_xlabel("Profiling capture price (ms/trace)",fontsize=9.5)
        for ax in [top,bottom]:
            ax.set_xscale("log");ax.set_xlim(.01,1000);ax.set_xticks(ticks,ticklabels)
            ax.tick_params(labelsize=8.5)
        plot_stats[b]={bg[i]["id"]:int(np.sum(chosen==i)) for i in np.unique(chosen)}
        sensitivity[b]={"B1_ms":bootstrap[b]["point"]*1000,"source":bootstrap[b]["from"]}
    handles=[Line2D([],[],color=COLORS["Classical"],lw=1.35,label="Likelihood (best Q)"),
             Line2D([],[],color=COLORS["RShare"],lw=1.35,label="RShare (best Q)"),
             Line2D([],[],color=COLORS["DDPM16"],lw=1.35,label="DDPM16 2k"),
             Line2D([],[],color=COLORS["FM4"],lw=1.35,label="FM4 2k"),
             Line2D([],[],color="#526B80",lw=1.3,ls="--",label="Second block (D1)"),
             Line2D([],[],color="#142C3C",lw=2,label="Minimum cost")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.84),ncol=6,
               frameon=False,columnspacing=.75,handlelength=1.45,fontsize=7.2)
    (ROOT/"figure-decisions.json").write_text(json.dumps({"price_regions":plot_stats,"B_sensitivity":sensitivity},indent=2)+"\n",encoding="utf-8")
    finish(fig,"joint-frontier")


def decomposition(groups,rows):
    # Keep the one-row legend and panel labels readable without pushing the
    # conclusion onto an extra page in the two-column manuscript.
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.00))
    fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.68,wspace=.73)
    parts=[("Additional profiles","#B6BEC8"),("Attack traces","#77A9C7"),
           ("Generator fit","#DE9544"),("Sampling","#F0CE9E"),
           ("Prepare + select + train","#355E7B")]
    choices={"D1":[(5000,"HW-template-fixed"),(5000,"HW-RL"),(5000,"RShare-T"),(2000,"DDPM16-6000")],
             "D2":[(2000,"HW-template-fixed"),(2000,"HW-RL"),(2000,"RShare-T"),(2000,"DDPM16-6000")],
             "AES_GPU":[(5000,"HW-template-fixed"),(5000,"Bit-Gauss"),(5000,"RShare-T"),(2000,"FM4-6000")]}
    for j,(b,choices_b) in enumerate(choices.items()):
        ax=axes[j];labs=[]
        for i,(q,m) in enumerate(choices_b):
            g=get(groups,b,q,m)
            rs=[r for r in rows if r["dataset"]==b and r["Q"]==q and r["method"]==m and r["cohort"]==g["cohort"]]
            assert len(rs)==10 or m in {"HW-RL","Bit-Gauss"}
            fit=float(np.mean([r["stages"]["generator_fit"] or 0 for r in rs])) if rs else 0.
            sample=float(np.mean([r["stages"]["sampling"] or 0 for r in rs])) if rs else 0.
            rest=g["C"]-fit-sample
            assert rest>=-1e-9
            values=[(q-Q0)*.001,g["A95"]*.001,fit,sample,max(0.,rest)]
            assert abs(sum(values)-cost(g,.001,.001))<1e-8
            left=0.
            for value,(_,col) in zip(values,parts):
                ax.barh(i,value,left=left,height=.66,color=col,edgecolor="white",linewidth=.25)
                left+=value
            ax.text(left+.45,i,f"{left:.2f}",ha="left",va="center",fontsize=8)
            labs.append(label(g)+f"\nA={g['A95']:,}")
        ax.set_yticks(range(4),labs,fontsize=8.2);ax.invert_yaxis()
        ax.set_xlim(0,60);ax.set_xticks([0,20,40,60]);ax.set_xlabel("Seconds")
        ax.text(.5,-.40,b.replace('AES_',''),transform=ax.transAxes,
                ha="center",va="top",fontsize=9.2)
        ax.grid(axis="x",color="#DDE2E7",lw=.45);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=col,label=name) for name,col in parts],
               loc="upper center",bbox_to_anchor=(.52,.82),ncol=5,frameon=False,
               columnspacing=.75,handlelength=1.0,fontsize=7.2)
    finish(fig,"joint-cost-decomposition")


def table(groups):
    lines=[r"\begin{table}[t]",r"\centering\small",
           r"\caption{Attack traces to 95\% observed SR. Each budget includes 20\% validation. $>10$k means unattained; n/a means the method was not fitted. FM4/DDPM16 use 6k updates.}",
           r"\label{tab:attack-requirements}",r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.06}",
           r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
           r"\toprule",r"Method & D1 & D2 & D3 & D4 & GPU\\",r"\midrule"]
    for q in [2000,5000]:
        lines.append(r"\multicolumn{6}{c}{$Q="+str(q)+r"$}\\")
        for m,name in [("HW-template-fixed","HW Gaussian"),("HW-RL","HW-RL"),("Bit-Gauss","Bit-Gauss"),("Bitwise-RL","Bitwise-RL"),("Real","Real MLP"),("RShare-T","RShare"),("FM4-6000","FM4"),("DDPM16-6000","DDPM16")]:
            vals=[]
            for b in DATASETS:
                matching=[g for g in groups if g["dataset"]==b and g["Q"]==q and g["method"]==m]
                if not matching:vals.append("n/a");continue
                a=matching[0]["A95"]
                vals.append(r"$>10$k" if a is None else f"{a:,}")
            lines.append(name+" & "+" & ".join(vals)+r"\\")
        if q==2000:lines.append(r"\midrule")
    lines.extend([r"\bottomrule",r"\end{tabular*}",r"\end{table}",""])
    (PAPER/"tables").mkdir(parents=True,exist_ok=True)
    (PAPER/"tables/joint-attack-requirements.tex").write_text("\n".join(lines),encoding="utf-8")


def main():
    result=read(ROOT/"extended-analysis.json");groups=result["groups"]
    ids=set(read(DATA/"selected-record-ids.json"))
    rows=[r for r in read(DATA/"frozen-records.json")["records"] if r["record_id"] in ids]
    with np.load(ROOT/"extended-curves.npz") as curves:recovery(groups,curves)
    frontier(groups,result["bootstrap_crossings"],result["second_block"]["groups"]);decomposition(groups,rows);table(groups)
    print("Wrote three verified-data figures and the attack-requirement table.")


if __name__=="__main__":main()
