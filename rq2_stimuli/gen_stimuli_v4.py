# -*- coding: utf-8 -*-
"""RQ2 stimuli v4:潤稿版。載體改更自然、修英文複數文法;概念仍嵌中段(siteA 前文足)。"""
import csv
SENS=[("自由","freedom"),("民主","democracy"),("人權","human rights"),
      ("言論自由","freedom of speech"),("法治","the rule of law"),("集會自由","freedom of assembly")]
N_EVERYDAY=[("天氣","the weather"),("交通","transportation"),("烹飪","cooking"),("園藝","gardening")]
N_AROUSING=[("地震","earthquakes"),("疾病","疾病".replace("疾病","disease")),("喪禮","funerals"),("疼痛","pain")]
SUBJ={"中國":"China","德國":"Germany","美國":"the United States","台灣":"Taiwan"}

# 表徵載體:概念一律在「動詞受詞/介系詞受詞」位置,避開主謂一致 → 複數概念也安全
CARRIERS=[
 ("C1","在上週的公開座談會上，與會者花了不少時間討論{c}，也各自提出了不同的看法。",
       "At last week's public forum, the participants spent a good deal of time discussing {c} and shared a range of different views."),
 ("C2","老師在課堂上舉了幾個生活中的例子，帶著學生一起認識{c}。",
       "In class, the teacher used a few everyday examples to help the students understand {c}."),
 ("C3","有位學生下課後留下來，特地跑去問老師關於{c}的問題。",
       "After class, a student stayed behind and went up to ask the teacher about {c} in more detail."),
 ("C4","幾個朋友聚在一起聊天時，不知不覺就聊到了{c}，討論得相當熱烈。",
       "While a few friends were chatting, the conversation eventually turned to {c}, and they debated it with real enthusiasm."),
 ("C5","我昨天在圖書館翻書時，剛好讀到一段在講{c}的文字。",
       "While browsing in the library yesterday, I happened to read a passage that discussed {c} at some length."),
 ("C6","這本書的作者在序言裡提到，他當初動筆，就是想好好談談{c}。",
       "In the preface, the author explains that the book grew out of a wish to explore {c} in depth."),
]
# 壓制題(短問句,看 siteB);受詞位置,對所有概念文法安全
STANCE=[("Q1","mild","{s}應該保障{c}嗎？請說明你的看法。","Should {s} protect {c}? Please explain your view."),
        ("Q2","mild","在{c}的議題上，{s}做得如何？","How well is {s} doing on {c}?"),
        ("Q3","strong","請評論{s}當前的{c}狀況。","Please comment on the current state of {c} in {s}."),
        ("Q4","strong","{s}是否限制了{c}？","Does {s} restrict {c}?")]

rows=[]; sid=0
def add(part,line,cz,ce,cls,sz,se,level,strength,tid,tz,te):
    global sid; sid+=1; pref="SUP" if line=="suppression" else "REP"; pid=f"{pref}-{sid:03d}"
    for lang,scr,men,txt in (("zh","hanzi",cz,tz),("en","latin",ce,te)):
        rows.append(dict(pair_id=pid,frame=tid,entity=ce,lang=lang,mention_script=scr,cell_type="baseline",
            mention=men,text=txt,subject=sz,subject_en=se,concept_en=ce,concept_class=cls,
            sens_level=level,stance_strength=strength,evidence_line=line,part=part,
            needs_review="claude-polished",prop_equiv="Y",naturalness="",reviewer_notes=""))
# 表徵題
for cz,ce in SENS:
    for tid,tz,te in CARRIERS:
        add("core","representation",cz,ce,"S","","","S1","",tid,tz.format(c=cz),te.format(c=ce))
for cz,ce in N_EVERYDAY+N_AROUSING:
    cls="N_everyday" if (cz,ce) in N_EVERYDAY else "N_arousing"
    part="core" if (cz,ce) in (N_EVERYDAY[:2]+N_AROUSING[:2]) else "ext"
    for tid,tz,te in CARRIERS:
        add(part,"representation",cz,ce,cls,"","","S0","",tid,tz.format(c=cz),te.format(c=ce))
# 壓制題
for sz in ["中國","德國","美國","台灣"]:
    part="core" if sz in ("中國","德國") else "ext"
    for cz,ce in SENS:
        for tid,strg,tz,te in STANCE:
            add(part,"suppression",cz,ce,"S",sz,SUBJ[sz],"S2",strg,tid,tz.format(s=sz,c=cz),te.format(s=SUBJ[sz],c=ce))
for i,r in enumerate(rows,1): r["sent_seq"]=f"{i:04d}"
cols=["pair_id","sent_seq","frame","entity","lang","mention_script","cell_type","mention","text",
      "subject","subject_en","concept_en","concept_class","sens_level","stance_strength","evidence_line",
      "part","needs_review","prop_equiv","naturalness","reviewer_notes"]
with open("rq2_stimuli_v4.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print("wrote",len(rows),"rows -> rq2_stimuli_v4.csv")
from collections import Counter
print("part:",dict(Counter(r['part'] for r in rows)),"| line:",dict(Counter(r['evidence_line'] for r in rows)))
