# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, re, textwrap
BASE="/sessions/gallant-magical-pascal/mnt/ROCLING-2026/three_claude_annotation"
PARQ={"Qwen":f"{BASE}/activations_Qwen2.5-7B-Instruct.parquet","Gemma":f"{BASE}/activations_gemma-3-12b-it.parquet"}
ANNO=f"{BASE}/coordinator/framing_annotation_full.csv"
NEUT={"CUL","ECON","GEO","LIFE","TRAV"}; SENS={"POL-INT","POL-DOM"}; ANCH={"PRC","GEOPOL-OTHER"}
RNG=np.random.default_rng(0)
a=pd.read_csv(ANNO)
def maj(s): s=s.dropna(); return s.value_counts().index[0] if len(s) else np.nan
D=a.drop_duplicates("uid")[["uid","model_tag","item_id","entity","lang","site","frame_source","cell_type","mse","cos"]].set_index("uid")
D=D.join(a.groupby("uid").agg(d2=("d2_anchor",maj))).reset_index()
D["model_tag"]=D.model_tag.str.lower(); D["anchored"]=D.d2.isin(ANCH).astype(int)

def bootdiff(x,y,n=2000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    d=x.mean()-y.mean()
    bs=[RNG.choice(x,len(x)).mean()-RNG.choice(y,len(y)).mean() for _ in range(n)]
    lo,hi=np.percentile(bs,[2.5,97.5]); return x.mean(),y.mean(),d,lo,hi

print("="*70,"\n[1] 錨定率 Qwen vs Gemma + bootstrap 95% CI  (台灣, baseline)")
for label,fr in [("全框架",None),("僅中性共享框架",NEUT)]:
    for site in ["A","B"]:
        for lg in ["en","zh"]:
            sub=D[(D.entity=="Taiwan")&(D.cell_type=="baseline")&(D.site==site)&(D.lang==lg)]
            if fr: sub=sub[sub.frame_source.isin(fr)]
            q=sub[sub.model_tag=="qwen"].anchored; g=sub[sub.model_tag=="gemma"].anchored
            if len(q)<5 or len(g)<5: continue
            qm,gm,d,lo,hi=bootdiff(q,g)
            star="★" if (lo>0 or hi<0) else "n.s."
            ratio=f"{qm/gm:.1f}x" if gm>0 else "inf"
            print(f"  {label:8s} site{site} {lg}: Qwen={qm:.3f} Gemma={gm:.3f}  diff={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] {star}  ratio={ratio}  (nQ={len(q)},nG={len(g)})")

print("="*70,"\n[3] 只看中性框架的 en 觸發效應(排除政治題材;Qwen 內部 en vs zh)")
for mt in ["qwen","gemma"]:
    sub=D[(D.entity=="Taiwan")&(D.cell_type=="baseline")&(D.site=="A")&(D.frame_source.isin(NEUT))&(D.model_tag==mt)]
    en=sub[sub.lang=="en"].anchored; zh=sub[sub.lang=="zh"].anchored
    em,zm,d,lo,hi=bootdiff(en,zh)
    print(f"  {mt:5s}: en={em:.3f} zh={zm:.3f}  en-zh={d:+.3f} CI[{lo:+.3f},{hi:+.3f}]  {'★' if lo>0 else 'n.s.'}")

print("="*70,"\n[4] think-say proxy:幾何敏感投影 → AV 是否錨定 (中性框架台灣 siteA)")
for mt,path in [("qwen",PARQ["Qwen"]),("gemma",PARQ["Gemma"])]:
    df=pd.read_parquet(path).reset_index(drop=True)
    M=np.vstack([np.asarray(v,np.float64) for v in df.activation_vector]); M=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
    tw=df.entity.eq("Taiwan").values; A=df.site.eq("A").values
    isS=df.frame.isin(SENS).values; isN=df.frame.isin(NEUT).values
    delta=M[tw&A&isS].mean(0)-M[tw&A&isN].mean(0)          # 敏感方向(純由框架定義,與標註無關)
    base=df[tw&A&df.cell_type.eq("baseline").values&isN].copy()
    base["proj"]=M[base.index]@delta
    # anchoring per vector: join annotation (baseline, Taiwan, siteA, neutral) by (item_id,lang)
    an=D[(D.model_tag==mt)&(D.entity=="Taiwan")&(D.cell_type=="baseline")&(D.site=="A")&(D.frame_source.isin(NEUT))]
    an=an.groupby(["item_id","lang"]).anchored.mean().reset_index()
    m=base.merge(an,left_on=["pair_id","lang"],right_on=["item_id","lang"])
    # spearman
    def spearman(x,y):
        rx=pd.Series(x).rank().values; ry=pd.Series(y).rank().values
        return float(np.corrcoef(rx,ry)[0,1])
    rho=spearman(m.proj.values,m.anchored.values); p=float('nan')
    # binned
    m["q"]=pd.qcut(m.proj,3,labels=["低","中","高"])
    tab=m.groupby("q",observed=True).anchored.mean()
    print(f"  {mt:5s}: Spearman(proj, anchored)={rho:+.3f} p={p:.3f} (n={len(m)}) | 錨定率 低/中/高投影 = "
          + " / ".join(f"{tab.get(k,float('nan')):.3f}" for k in ["低","中","高"]))
