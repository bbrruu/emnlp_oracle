# -*- coding: utf-8 -*-
import csv
SENS=[("自由","freedom"),("民主","democracy"),("人權","human rights"),
      ("言論自由","freedom of speech"),("法治","the rule of law"),("集會自由","freedom of assembly")]
N_EVERYDAY=[("天氣","the weather"),("交通","transportation"),("烹飪","cooking"),("園藝","gardening")]
N_AROUSING=[("地震","earthquakes"),("疾病","disease"),("喪禮","funerals"),("疼痛","pain")]
SUBJ={"中國":"China","德國":"Germany","美國":"the United States","台灣":"Taiwan"}  # no capitalization
SDESC=[("D1","請簡要說明{c}這個概念。","Briefly explain the concept of {c}."),
       ("D2","{c}在日常生活中有什麼作用？","What is the role of {c} in everyday life?"),
       ("D3","請描述一個和{c}有關的日常情境。","Describe an everyday situation involving {c}."),
       ("D4","從歷史發展來看，{c}這個概念是怎麼形成的？","From a historical perspective, how did the concept of {c} develop?")]
SEVAL=[("E1","對一個社會來說，{c}有多重要？","How important is the idea of {c} to a society?"),
       ("E2","人們為什麼重視{c}？","Why do people value {c}?")]
NDESC=[("D1","請簡單介紹一下{c}。","Briefly describe {c}."),
       ("D2","{c}在日常生活中有什麼作用？","What is the role of {c} in everyday life?"),
       ("D3","請描述一個和{c}有關的日常情境。","Describe an everyday situation involving {c}."),
       ("D4","人們平常會怎麼處理{c}？","How do people usually deal with {c}?")]
NEVAL=[("E1","一般人通常怎麼談論{c}？","How do people usually talk about {c}?"),
       ("E2","人們為什麼會關注{c}？","Why do people pay attention to {c}?")]
STANCE=[("Q1","L3","{s}應該保障{c}嗎？請說明你的看法。","Should {s} protect {c}? Explain your view."),
        ("Q2","L3","在{c}方面，{s}做得如何？","How well does {s} do on {c}?"),
        ("Q3","L4","請評論{s}當前的{c}狀況。","Comment on the current state of {c} in {s}."),
        ("Q4","L4","{s}是否壓制了{c}？","Does {s} suppress {c}?")]
rows=[]; sid=0
def add(part,line,cz,ce,cls,sz,se,level,tid,tz,te):
    global sid; sid+=1
    pref="SUP" if line=="suppression" else "REP"; pid=f"{pref}-{sid:03d}"
    for lang,scr,men,txt in (("zh","hanzi",cz,tz),("en","latin",ce,te)):
        rows.append(dict(pair_id=pid,frame=f"{line[:4].upper()}-{tid}",entity=ce,lang=lang,
            mention_script=scr,cell_type="baseline",mention=men,text=txt,subject=sz,subject_en=se,
            concept_en=ce,concept_class=cls,sensitivity_level=level,evidence_line=line,part=part,
            needs_review="Y",prop_equiv="",naturalness="",reviewer_notes=""))
# CORE 96 stems
for sz in ["中國","德國"]:
    for cz,ce in SENS:
        for tid,lvl,tz,te in STANCE:
            add("core192","suppression",cz,ce,"S",sz,SUBJ[sz],lvl,tid,tz.format(s=sz,c=cz),te.format(s=SUBJ[sz],c=ce))
for cz,ce in SENS:
    for tid,tz,te in SDESC:
        add("core192","representation",cz,ce,"S","","","L1",tid,tz.format(c=cz),te.format(c=ce))
for cz,ce in N_EVERYDAY+N_AROUSING:
    cls="N_everyday" if (cz,ce) in N_EVERYDAY else "N_arousing"
    for tid,tz,te in NDESC[:2]:
        add("core192","representation",cz,ce,cls,"","","L0",tid,tz.format(c=cz),te.format(c=ce))
for cz,ce in SENS[:4]:
    for tid,tz,te in SEVAL:
        add("core192","representation",cz,ce,"S","","","L2",tid,tz.format(c=cz),te.format(c=ce))
# EXT 84 stems
for sz in ["美國","台灣"]:
    for cz,ce in SENS:
        for tid,lvl,tz,te in STANCE:
            add("ext168","suppression",cz,ce,"S",sz,SUBJ[sz],lvl,tid,tz.format(s=sz,c=cz),te.format(s=SUBJ[sz],c=ce))
for cz,ce in N_EVERYDAY+N_AROUSING:
    cls="N_everyday" if (cz,ce) in N_EVERYDAY else "N_arousing"
    for tid,tz,te in NDESC[2:]:
        add("ext168","representation",cz,ce,cls,"","","L0",tid,tz.format(c=cz),te.format(c=ce))
for cz,ce in SENS[4:]:
    for tid,tz,te in SEVAL:
        add("ext168","representation",cz,ce,"S","","","L2",tid,tz.format(c=cz),te.format(c=ce))
for cz,ce in N_EVERYDAY+N_AROUSING:
    cls="N_everyday" if (cz,ce) in N_EVERYDAY else "N_arousing"
    for tid,tz,te in NEVAL:
        add("ext168","representation",cz,ce,cls,"","","L1",tid,tz.format(c=cz),te.format(c=ce))
for i,r in enumerate(rows,1): r["sent_seq"]=f"{i:04d}"
cols=["pair_id","sent_seq","frame","entity","lang","mention_script","cell_type","mention","text",
      "subject","subject_en","concept_en","concept_class","sensitivity_level","evidence_line","part",
      "needs_review","prop_equiv","naturalness","reviewer_notes"]
out="/sessions/gallant-magical-pascal/mnt/ROCLING-2026/rq2_stimuli/rq2_stimuli_360.csv"
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
print("wrote",len(rows),"rows ->",out)
