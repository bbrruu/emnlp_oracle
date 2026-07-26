#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ2 preliminary analysis on existing RQ1 data (zero GPU).
PART A: representation geometry (activation parquet, direction-based, scale-free scalars)
PART B: AV verbalization + framing annotation (cross-model: Qwen vs Gemma)
Outputs: prints, results.json, and PNG figures.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE="/sessions/gallant-magical-pascal/mnt/ROCLING-2026/three_claude_annotation"
OUT="/sessions/gallant-magical-pascal/mnt/outputs"
PARQ={"Qwen2.5-7B":f"{BASE}/activations_Qwen2.5-7B-Instruct.parquet",
      "Gemma-3-12B":f"{BASE}/activations_gemma-3-12b-it.parquet"}
ANNO=f"{BASE}/coordinator/framing_annotation_full.csv"
SENS={"POL-INT","POL-DOM"}; NEUT={"CUL","ECON","GEO","LIFE","TRAV"}
RNG=np.random.default_rng(0)
R={}

def l2(M): return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)
def auc(s,y):
    y=np.asarray(y); o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    p=y.sum(); n=len(y)-p
    return np.nan if p==0 or n==0 else (r[y==1].sum()-p*(p+1)/2)/(p*n)
def cv_auc(X,y,k=5):
    y=np.asarray(y); idx=np.arange(len(y)); RNG.shuffle(idx); f=np.array_split(idx,k); a=[]
    for i in range(k):
        te=f[i]; tr=np.concatenate([f[j] for j in range(k) if j!=i])
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        d=X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0)
        a.append(auc(X[te]@d,y[te]))
    return float(np.nanmean(a)) if a else np.nan

print("#"*72,"\n# PART A — Representation geometry (direction-based)\n","#"*72)
R["geometry"]={}
for name,path in PARQ.items():
    df=pd.read_parquet(path).reset_index(drop=True)
    M=l2(np.vstack([np.asarray(v,np.float64) for v in df["activation_vector"].values]))
    tw=df["entity"].eq("Taiwan").values
    isS=df["frame"].isin(SENS).values; isN=df["frame"].isin(NEUT).values
    g={"dim":int(M.shape[1])}
    print(f"\n=== {name} ({M.shape[0]} vecs, dim {M.shape[1]}) ===")
    # A1/A2 separability sensitive-vs-neutral, per site + direction drift A vs B
    dirs={}
    for site in ["A","B"]:
        m=tw&df["site"].eq(site).values&(isS|isN)
        X=M[m]; y=isS[m].astype(int)
        a=cv_auc(X,y); dirs[site]=X[y==1].mean(0)-X[y==0].mean(0)
        g[f"sep_AUC_site{site}"]=round(float(a),3)
        print(f"  site {site}: sensitive-vs-neutral Δ-probe AUC={a:.3f}  (n_sens={int(y.sum())}, n_neut={int((1-y).sum())})")
    cosAB=float(dirs["A"]@dirs["B"]/((np.linalg.norm(dirs['A'])*np.linalg.norm(dirs['B']))+1e-9))
    g["cos_dir_siteA_siteB"]=round(cosAB,3)
    print(f"  cos(Δ_siteA, Δ_siteB)={cosAB:.3f}  (低=概念 token 與句尾方向不同=位置框架漂移)")
    # A3 cross-lingual consistency of sensitive direction (Taiwan, per site)
    for site in ["A","B"]:
        d={}
        for lg in ["en","zh"]:
            m=tw&df["site"].eq(site).values&df["lang"].eq(lg).values
            mS=m&isS; mN=m&isN
            d[lg]=M[mS].mean(0)-M[mN].mean(0)
        c=float(d["en"]@d["zh"]/((np.linalg.norm(d['en'])*np.linalg.norm(d['zh']))+1e-9))
        g[f"crosslang_cos_site{site}"]=round(c,3)
        print(f"  site {site}: cross-lingual sensitive-dir cos(Δ_en,Δ_zh)={c:.3f}  (高=語言不變)")
    # A4 per-entity language-displacement direction consistency (neutral shared frames)
    g["lang_disp_consistency"]={}
    for ent in ["Taiwan","Japan","Iceland"]:
        cons=[]
        for site in ["A","B"]:
            base=df["entity"].eq(ent).values&df["cell_type"].eq("baseline").values&isN&df["site"].eq(site).values
            sub=df[base]
            disp=[]
            for pid,grp in sub.groupby("pair_id"):
                if set(grp["lang"])>={"en","zh"}:
                    ie=grp.index[grp["lang"].eq("en")][0]; iz=grp.index[grp["lang"].eq("zh")][0]
                    disp.append(M[ie]-M[iz])
            if len(disp)>=3:
                D=np.vstack(disp); cons.append(np.linalg.norm(D.mean(0))/ (np.linalg.norm(D,axis=1).mean()+1e-9))
        if cons: g["lang_disp_consistency"][ent]=round(float(np.mean(cons)),3)
    print("  per-entity language-shift direction consistency (低=方向雜亂):",g["lang_disp_consistency"])
    R["geometry"][name]=g

print("\n"+"#"*72,"\n# PART B — AV verbalization + framing annotation (cross-model)\n","#"*72)
a=pd.read_csv(ANNO)
# dedupe to one row per description (desc-level fields), majority vote for annotation dims
desc_cols=["uid","model_tag","entity","lang","site","frame_source","cell_type","mse","cos","gate","desc_lang"]
base=a.drop_duplicates("uid")[desc_cols].set_index("uid")
def majority(s):
    s=s.dropna();  return s.value_counts().index[0] if len(s) else np.nan
maj=a.groupby("uid").agg(d2=("d2_anchor",majority), d5=("d5_drift",majority))
D=base.join(maj).reset_index()
D["model_tag"]=D["model_tag"].str.lower()
print("desc-level rows:",len(D),"| models:",D.model_tag.value_counts().to_dict())
print("d2 values:",D.d2.value_counts().to_dict())
print("desc_lang values:",D.desc_lang.value_counts().to_dict())
ANCH={"PRC","GEOPOL-OTHER"}
D["anchored"]=D.d2.isin(ANCH).astype(int)
D["drifted"]=D.d5.isin({"DRIFTED","HALLUCINATED"}).astype(int)
D["lang_drift"]=(D.desc_lang.str.lower()!=D.lang.str.lower()) & (D.desc_lang.str.lower()!="mixed")

R["annotation"]={}
for mt,lab in [("qwen","Qwen2.5-7B"),("gemma","Gemma-3-12B")]:
    sub=D[D.model_tag==mt]; b={}
    # B1 NLA faithfulness
    b["median_mse"]=round(float(sub.mse.median()),4); b["median_cos"]=round(float(sub["cos"].median()),4)
    b["gate_fail_rate"]=round(float((sub.gate==1).mean()),3)
    # B4 drift
    b["av_drift_rate"]=round(float(sub.drifted.mean()),3)
    # B2 language drift
    b["lang_drift_overall"]=round(float(sub.lang_drift.mean()),3)
    b["lang_drift_by_ctx"]={lg:round(float(sub[sub.lang==lg].lang_drift.mean()),3) for lg in ["en","zh"]}
    # B3 Taiwan anchoring by lang x site
    tw=sub[sub.entity=="Taiwan"]; b["anchor_taiwan"]={}
    for site in ["A","B"]:
        for lg in ["en","zh"]:
            s2=tw[(tw.site==site)&(tw.lang==lg)]
            b["anchor_taiwan"][f"site{site}_{lg}"]=round(float(s2.anchored.mean()),3) if len(s2) else None
    R["annotation"][lab]=b
    print(f"\n=== {lab} ===")
    print(f"  NLA faithfulness: median mse={b['median_mse']}, median cos={b['median_cos']}, gate-fail rate={b['gate_fail_rate']}")
    print(f"  AV semantic drift rate (d5)={b['av_drift_rate']}")
    print(f"  AV output-language drift: overall={b['lang_drift_overall']}, by ctx={b['lang_drift_by_ctx']}")
    print(f"  Taiwan China-anchoring rate (d2) by site×lang: {b['anchor_taiwan']}")

json.dump(R, open(f"{OUT}/rq2_results.json","w"), ensure_ascii=False, indent=2)
print("\nsaved results.json")

# ---------------- FIGURES ----------------
plt.rcParams.update({"figure.dpi":120,"font.size":9})
models=list(PARQ.keys()); colors={"Qwen2.5-7B":"#c0504d","Gemma-3-12B":"#4f81bd"}

# Fig1 separability AUC by model x site
fig,ax=plt.subplots(figsize=(5,3))
x=np.arange(len(models)); w=0.35
for i,site in enumerate(["A","B"]):
    ax.bar(x+(i-0.5)*w,[R["geometry"][m][f"sep_AUC_site{site}"] for m in models],w,label=f"site {site}")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.5,1.02); ax.axhline(0.5,ls=":",c="grey")
ax.set_ylabel("Δ-probe AUC"); ax.set_title("A1. Sensitive vs neutral separability"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_separability.png"); plt.close()

# Fig2 cross-lingual consistency by model x site
fig,ax=plt.subplots(figsize=(5,3))
for i,site in enumerate(["A","B"]):
    ax.bar(x+(i-0.5)*w,[R["geometry"][m][f"crosslang_cos_site{site}"] for m in models],w,label=f"site {site}")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("cos(Δ_en, Δ_zh)")
ax.set_title("A3. Cross-lingual consistency of sensitive direction"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_crosslang.png"); plt.close()

# Fig3 Taiwan anchoring by model (en vs zh, site A)
fig,ax=plt.subplots(figsize=(5,3))
for i,lg in enumerate(["en","zh"]):
    ax.bar(x+(i-0.5)*w,[R["annotation"][m]["anchor_taiwan"][f"siteA_{lg}"] for m in models],w,label=f"{lg} ctx")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("China-anchoring rate (site A)")
ax.set_title("B3. Taiwan→China anchoring (AV), by context language"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_anchoring.png"); plt.close()

# Fig4 NLA faithfulness + lang drift by model
fig,axs=plt.subplots(1,2,figsize=(7,3))
axs[0].bar(models,[R["annotation"][m]["median_cos"] for m in models],color=[colors[m] for m in models])
axs[0].set_ylim(0.9,1.0); axs[0].set_title("B1. NLA round-trip cos (median)"); axs[0].set_ylabel("cos")
axs[1].bar(models,[R["annotation"][m]["lang_drift_overall"] for m in models],color=[colors[m] for m in models])
axs[1].set_title("B2. AV output-language drift rate"); axs[1].set_ylabel("rate")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_faithfulness.png"); plt.close()
print("saved 4 figures")
