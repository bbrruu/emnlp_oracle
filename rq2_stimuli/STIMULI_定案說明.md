# RQ2 Stimuli 定案說明

**定案檔:`rq2_stimuli_FINAL.csv`(360 句)**
**狀態:內容 / 結構 / 規範已鎖定;剩兩步——① naturalness 人工評分 ② 筆電跑 `verify_tokenization`(±20% 長度)。過了即完全定案。**

---

## 組成(360 列 = 180 句對 × 中英)

| 維度 | 內容 |
|---|---|
| evidence_line | 壓制題(S2)192 / 表徵題(S0+S1)168 |
| sens_level | **S0 中性** 96、**S1 敏感描述** 72、**S2 壓制** 192 |
| part | **core** 216(敏感表徵全 + 半中性 + 中/德壓制)、**ext** 144(美/台壓制 + 另半中性) |
| 敏感概念(6) | 自由、民主、人權、言論自由、法治、集會自由 |
| 中性-日常(4) | 天氣、交通、烹飪、園藝 |
| 中性-敏感非政治(4) | 地震、疾病、喪禮、疼痛 |
| 壓制主語 | 中國 / 德國 / 美國 / 台灣 各 48 |
| stance_strength | mild(Q1/Q2)96 / strong(Q3/Q4)96 |
| 載體 | 表徵 C1–C6、壓制 Q1–Q4 |
| 語言 | 中 180 / 英 180 |

三種題型:**S0 中性**=Δ 減項(對照);**S1 敏感描述**=表徵探針(siteA);**S2 壓制**=think–say(siteB + 生成)。Q4 已由「壓制」改中性「限制」。

---

## 欄位速查

**設計標籤(別改)**:`pair_id, frame, entity, lang, mention_script, cell_type, mention, subject, concept_en, concept_class, sens_level, stance_strength, evidence_line, part`

**可改 / 待填**:
- `text` — 句子本身,潤稿時可改(守則見下)
- `naturalness` — **待人工填**(1–5;中文母語者評,<4 改 text;英文輕掃)
- `prop_equiv` — 中英是否命題等價(Y/partial/N)
- `needs_review` — 看完改成 `done`
- `reviewer_notes` — 備註

---

## 潤稿守則(改 text 時守住)

1. 概念詞留著、**只出現一次**(siteA 靠它定位)。
2. **表徵題(S0/S1)概念別移到句首**,前面要有鋪陳(leadin);壓制題(S2)看句末,不限。
3. 概念**別卡進專名**(自由≠自由時報、民主≠民進黨、人權≠人權觀察)。
4. 中英要對得上;改中文順手看英文那列要不要跟著動。
5. **中性和敏感的句型要一致**(配對設計,Δ 才乾淨);中性某句略generic可微調用詞,但別換成不同句型。
6. 用 Excel/Numbers 存檔要存回 **UTF-8 CSV**(別存 xlsx/Big5,中文會亂碼)。

---

## 最後一步:跑 verify_tokenization(你筆電,不需國網)

```bash
pip install -U "transformers>=4.50" tokenizers huggingface_hub pandas
huggingface-cli login          # Gemma-3 為 gated,需先在 HF 網頁同意授權;Qwen2.5 免登入

python verify_tokenization.py \
    --models Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it \
    --pairs-csv rq2_pairs.csv \
    --min-leadin 8 \
    --outdir tokenizer_report
```

判準:**表徵題(S0/S1)`leadin_tokens ≥ 8` + 句對長度差 ≤ ±20%**;不過關的改 text 再跑一次。壓制題(S2)看 siteB,leadin 不設限。

> `rq2_pairs.csv` 已重生自 FINAL;改了 `rq2_stimuli_FINAL.csv` 的 text 後,記得重生一次 pairs(`d[['pair_id','frame','lang','mention','text']].to_csv('rq2_pairs.csv')`)再跑。

---

## 檔案清單

| 檔案 | 用途 |
|---|---|
| **`rq2_stimuli_FINAL.csv`** | ✅ 定案語料(要動就動這個) |
| `rq2_pairs.csv` | 給 verify_tokenization 的精簡欄位 |
| `verify_tokenization.py` | tokenizer 驗證(leadin/±20%) |
| `check_norms.py` | norm-5 + 結構檢查(已過) |
| `RQ2_對等性規範_4.2.md` | §4.2 規範 |
| `RUN_verify_tokenization.md` | 驗證指令 |
| `gen_stimuli_v4.py` | 生成器(可重現 FINAL) |
| `rq2_stimuli_v4_backup.csv` | C3/C5 修正前的備份 |
| `_archive/` | 舊版(v2/v3) |
