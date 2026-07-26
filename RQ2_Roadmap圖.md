
## 主問題

> **Qwen vs Gemma 在敏感議題(自由/民主/人權)上,內部機制有何不同?**
> **尤其:內部表徵「想的」與實際輸出「說的」是否有 think–say gap?**

---

## (A) 子問題 ── 用什麼回答 ── 走到哪

```
①  敏感概念怎麼被表徵?
    └─ 用:activation 幾何(Δ 可分性、siteA/B)      ✅ 已驗 AUC≈0.99
②  Qwen 和 Gemma 差在哪?（主軸）
    └─ 用:NLA 把內部狀態說成英文 + 錨定率比較      ✅ proxy 已驗 38.7×
③  這些差異是「因果」還是相關?
    └─ 用:steering(加/消融敏感方向)               ⬜ 選配,時間夠再做
④  差異是「文化」還是「RLHF」?
    └─ 用:base vs instruct 行為比較                ⬜ 或直接寫 discussion
⑤  有沒有 think–say gap?（賣點）
    ├─ 壓制存在嗎?  → 壓制 pilot                    ✅ Gate 過(軟壓制)
    └─ 想≠說?       → siteB 的 NLA vs 實際輸出       ⏳ pilot 過,主實驗未跑
```

**一句話**:主軸 = ②(Qwen vs Gemma),賣點 = ⑤(think–say gap);③④是加分/discussion;**不用 crosscoder**。

---

## (B) 關鍵路徑（正確順序）

> ⚠ 注意順序:**潤稿在前、judge 在後**。judge 評的是「模型的回答」,回答要先生成出來才存在,所以 judge 在生成之後才「跑」。

```
✅ 初步分析(RQ1 proxy,38.7×)
✅ 壓制 pilot + judge 試評(24 則,Gate 過)
✅ ① 潤稿 + 驗 token ──► 定版 360(兩 tokenizer 都過)
        │
        ▼
🔄 ② 餵進 Qwen/Gemma(×2 模型)  ← 你在這
     ├─ 生成回答(say)：Qwen ✅ / Gemma 🔄 跑中(Colab 4-bit,備份;正式版用國網 bf16)
     └─ 抽 activation(think)：⏳ 等國網(bf16;每句 siteA+siteB = 720 向量/模型)
        │
        ▼
⬜ ③ NLA 語言化(360 都要:siteA=表徵、siteB=開口前狀態 → 說成文字)
        │
        ▼
⬜ ④ 分析(這裡分岔成兩條 RQ)
     ├─ RQ1 幾何：Δ / 可分性 AUC / 跨語言 cos / bootstrap CI   → 純程式,不用人
     └─ 語意 judge(RQ1 描述框架 + RQ2 想 vs 說)               → LLM judge 全標 + 人抽驗~15%
        │
        ▼
⬜ ⑤ Qwen vs Gemma 比較 + think–say gap 量化 → 🏁 寫作投稿(8/1)
```

**進度**:`█████████░░░░░`  約 55% ── 定版 360 完成、say 收了一半;剩抽 activation → NLA → 分析。
論文 Intro / Related Work / bib 平行推進中。

---

## Qwen vs Gemma 藏在哪?（`×2` 的意思）

`×2` = **兩個模型**,每個步驟各跑一次,最後「比兩邊」就是主結果。它不是單獨一步,是所有步驟都做兩遍:

```
        定版 360 stimuli
             │
    ┌────────┴─────────┐
    ▼                  ▼
  Qwen:              Gemma:
  抽取→NLA→生成→judge  抽取→NLA→生成→judge
    └────────┬─────────┘
             ▼
   比較兩邊 = 「Qwen vs Gemma 差異」(主軸②)
            + 各自的 think–say gap(賣點⑤)
```

補充:主軸的 **proxy 版已經先驗過**(初步分析的 38.7× 就是 Qwen vs Gemma),正式版是在乾淨的 RQ2 stimuli 上重做一次。

---

## 每個模型 × 每一題,收「兩種資料」

這兩種正是 think–say 的兩邊:

```
(a) activation(內部「想的」)
      ├── siteA = 概念 token  → 表徵分析(概念怎麼被編碼)
      └── siteB = 句末 token  → think–say(開口前的內部狀態)→ 經 NLA 語言化
(b) 生成回答(嘴上「說的」)
      └── 模型對 prompt 實際輸出的文字 → 給 judge 評 directness/限制

  think–say gap = (a 的 siteB 經 NLA)「想的」  vs  (b)「說的」  之間的落差
```

完整就是:**2 個模型 × 每題(siteA activation + siteB activation + 生成回答)**。

---

## think–say gap 怎麼標、誰標、標多少

think–say 有**兩個標的**,用**兩套不同的問法(rubric)**,但**都可以由同一個 LLM judge 做**(不是人機分工):

| 標的 | rubric(問什麼) | 誰標 |
|---|---|---|
| **say(回答)** | directness / 政治性限制合理化 | LLM judge |
| **think(AV 描述)** | 這段內部描述有沒有承載敏感/批判內容 | LLM judge（另一套問法） |

**gap = 兩邊比**:想的(AV)有敏感內容、說的(回答)卻消毒 → gap 成立。

**機器全標 + 人抽驗（沿用 RQ1 效度做法）:**
- **LLM judge 標全部**(RQ1 是 3 個 Claude 標 7200 則)。
- **人類只驗抽樣 ~15%**,算「LLM vs 人」一致度確認可信(RQ1 是人工 240 則)。人不用碰全部。

**量的概念:**

```
360 stimuli × 2 模型 = 720
think–say 聚焦 S2(壓制題)= 192 句 × 2 模型 = 384
每個 S2 輸出有 2 個標的:回答(say) + AV 描述(think)
  → 約 384 × 2 ≈ 768 項,全部 LLM judge 打分
  → 人類抽驗 ~15%(約 100 多則)確認一致度
（表徵題 S0/S1 沒有「壓制」可比,不進 think–say,走幾何/NLA 那條）
```

---

## 名詞速查

- **WS-D 潤稿**:資料組對 360 草稿句做人工品管(改順句子、評 naturalness、確認中英命題等價、S2 保持嗆)。
- **驗 token**:跑 `verify_tokenization.py`,用真 Qwen/Gemma tokenizer 檢查 leadin(前文夠)、中英 ±20%、siteA/B 位置。→ 這一步的產出 = **定版的 core192 + ext168(360)**。
- **judge 接 API 放大**:pilot 是手動評 24 則;放大 = 用中立 API judge(Gemini/GPT-4o)自動評全部回答 + 抽 15–20% 人工校驗。
- **siteA / siteB**:siteA = 句中概念 token;siteB = 句末 token。

---

## 保底線

就算 ⑤ 的 think–say 沒跑成,**①②(Qwen vs Gemma 表徵差異,NLA+幾何)已有數據,單獨就是一篇合格的 workshop paper**。這條路有地板,摔不重。
