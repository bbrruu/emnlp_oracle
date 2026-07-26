# -*- coding: utf-8 -*-
"""RQ2 §4.2 對等性規範檢查器。WS-D 潤稿後可重跑。
用法: python3 check_norms.py rq2_stimuli_360.csv
檢查: norm-5(概念卡專名) / norm-3 近似長度平衡 / 空白 mention / 命題對數完整。
"""
import sys, re, pandas as pd
f=sys.argv[1] if len(sys.argv)>1 else "rq2_stimuli_360.csv"
d=pd.read_csv(f)

# ---- norm-5: 概念不得出現在專名/固定複合詞中 ----
BLOCK={  # concept -> 禁用專名（子字串命中即違規）
 "自由":["自由時報","自由廣場","自由女神","自由之家"],
 "民主":["民主進步黨","民進黨","民主大道","民主紀念"],
 "人權":["人權觀察","人權律師","世界人權宣言","人權促進會"],
 "言論自由":[],"法治":["法治時報"],"集會自由":[],
 "freedom":["Freedom House","Freedom Tower","Statue of Liberty","Freedom of Information","freedom fighter"],
 "democracy":["Democratic Progressive","Democracy Index"],
 "human rights":["Human Rights Watch","Universal Declaration of Human Rights"],
 "freedom of speech":[],"the rule of law":["Rule of Law Index"],"freedom of assembly":[],
}
viol=[]
for _,r in d.iterrows():
    t=str(r.text)
    for name in BLOCK.get(str(r.mention),[]):
        if name.lower() in t.lower(): viol.append((r.pair_id,r.lang,name,t))
print(f"[norm-5] 概念卡專名違規: {len(viol)} 句")
for v in viol[:20]: print("   ",v)

# ---- norm-3: 近似長度平衡（估算，真正 ±20% 需用 Qwen/Gemma tokenizer）----
def est_tokens(s,lang):
    s=str(s)
    if lang=="zh": return sum(1 for ch in s if '一'<=ch<='鿿')  # CJK 字數 ≈ token
    return round(len(re.findall(r"[A-Za-z']+", s))*1.3)                 # 英文詞數×1.3
pairs=d.pivot_table(index="pair_id",columns="lang",values="text",aggfunc="first")
bad=[]
for pid,row in pairs.iterrows():
    if pd.isna(row.get("zh")) or pd.isna(row.get("en")): continue
    z=est_tokens(row["zh"],"zh"); e=est_tokens(row["en"],"en")
    if min(z,e)==0: continue
    ratio=max(z,e)/min(z,e)
    if ratio>1.5: bad.append((pid,z,e,round(ratio,2),row["en"][:50]))
print(f"\n[norm-3 近似] 中英長度估算比 >1.5 的句對: {len(bad)} 對 (估算，需以真 tokenizer 覆核)")
for b in sorted(bad,key=lambda x:-x[3])[:12]: print("   ",b)

# ---- 結構完整性 ----
print(f"\n[結構] 空白 mention: {int(d.mention.isna().sum())} | 每 pair_id 應恰 2 語言:",
      "OK" if (d.groupby('pair_id').lang.nunique()==2).all() else "有異常")
print("[結構] part:",d.part.value_counts().to_dict(),"| 概念詞面一致(mention 唯一數/概念):",
      d.groupby('concept_en').mention.nunique().to_dict())
