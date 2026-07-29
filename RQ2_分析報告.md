# RQ2 全量分析報告

分析日期：2026-07-29
分析腳本：`pipeline/rq2_analysis_skeleton.py`
資料：`results/`（Qwen2.5-7B-Instruct、gemma-3-12b-it，各 720 activation 向量 / 192 則 S2 回答）

---

## 一分鐘總結

**跑完了什麼**：幾何 + 四段 judge（say / think / frame / stability），**兩個獨立 judge 各跑一遍**
（`claude-opus-5` 與 `gpt-4o-mini`），共 2,788 次評分，失敗 0、拒答 0。

### 三個成立的發現

**1. 表徵層找到了，而且擋得住最嚴格的對照組。**
句末（siteB）表徵能區分敏感概念 vs 中性概念——**即使對照組換成情緒喚起詞**（地震/疾病/喪禮/疼痛）也成立。
兩模型 **12/12 個 cell 方向一致、p = 0.001**，中文英文各自都顯著。

> ⚠️ 這個結論**只有配對檢定看得到**。原本的 Δ-probe AUC 因為檢定力不足，Gemma 的 CI 蓋到 0.5、英文條件兩模型都蓋到 0.5，會讓人誤判成「沒效果」。刺激是嚴格最小對，統計單位要用 cell 不是句子。

**2. Qwen 的中德不對稱複現了 pilot，而且換 judge 還在。**

| Qwen say directness | 中國 | 德國 |
|---|---|---|
| 本次（Claude judge）| **0.56** | **1.62** |
| pilot（gpt-4o-mini）| 0.58 | 1.92 |

**3. think–say 消毒率的中德不對稱，兩個 judge 都顯著**（restriction 版 +0.417 / +0.375）。

### 三個不成立的（別寫進論文）

| | 為什麼 |
|---|---|
| **Gemma 的 say 側全部結論** | 回答 **95–100% 被截斷**（德國 100%），量到的是「誰被切得多」而不是壓制行為 |
| **主軸② Qwen vs Gemma 描述框架差異** | 兩個 judge 都 n.s.（proxy 版的 38.7× 錨定率沒複現）。改述為「兩模型都有框架效應，但彼此無差異」 |
| **`think_official_frame` 這一軸** | 跨 judge 加權 κ = **0.097**（近乎隨機），且兩個 judge 結論相反 |

另外 **siteA 的 AUC = 1.000 不要報**——那只是在讀詞彙身分（`freedom` vs `the weather` 本來就是不同 token），非設計缺陷但無法補救。

### 這批分析最重要的方法學訊息

> **絕對分數會隨 judge 大幅飄動，方向與顯著性不會。**
> 例：Qwen 的 `internal_content_rate` 在 Claude judge 下是 0.297，在 GPT judge 下是 0.802（**差 2.7 倍**）。
> 但「中國 − 德國」的差在兩個 judge 下同號同顯著（4 條軸中的 3 條）。
> **論文只能主張不對稱，不能主張絕對水準。**

### 現在該做的兩件事

1. **重跑 S2 回答**（`--max-new-tokens 1024`，寫到新目錄）。這一件事同時卡住 Gemma 的 say 側和整個跨模型行為比較。
2. **人工抽驗**：`results/analysis*/human_sample_*.csv` 各約 60 筆已分層抽好，填完 `human_*` 欄跑 `--stage agreement`。這是唯一能為 `internal_content_rate` 那個 2.7 倍落差定錨的方法。

---

## 摘要

| 結論 | 可用性 |
|---|---|
| siteB 句末表徵區分敏感 vs 中性（含情緒喚起嚴格對照），兩模型、兩語言均顯著 | ✅ 可寫 |
| Qwen 對中國比對德國更迴避（directness），複現 pilot | ✅ 可寫 |
| Qwen 對中國更常把限制正當化（restriction），跨 judge 穩健 | ✅ 可寫 |
| Qwen 的 think–say 消毒率中德不對稱（restriction 版） | ✅ 可寫 |
| siteA 的 Δ-probe AUC = 1.000 | ❌ 詞彙身分混淆，不可用 |
| Gemma 的 say 側任何結論 | ❌ 回答 95–100% 截斷 |
| `think_official_frame` 任何結論 | ❌ 跨 judge κ=0.097，結論分歧 |
| 主軸②「Qwen vs Gemma 描述框架差異」 | ❌ 兩個 judge 都不顯著 |

**最高優先的後續工作：S2 回答用 `max_new_tokens ≥ 1024` 重跑。** 它同時卡住 Gemma 的 say 側與整個跨模型行為比較。

---

## 1. 資料與設計

180 個 pair_id × 2 語言 × 2 site = 720 向量/模型，layer 20，bf16。

| 條件 | n（每 site） | 說明 |
|---|---|---|
| S0 中性 | 192 | 再分 `N_everyday` 96（天氣/烹飪/園藝/交通）與 `N_arousing` 96（地震/疾病/喪禮/疼痛）|
| S1 敏感 | 144 | freedom, democracy, human rights, freedom of speech, freedom of assembly, the rule of law |
| S2 壓制 | 384 | 提問句，走行為線 |

**刺激是嚴格最小對**：同一 (frame, lang) cell 內，S1 與 S0 的句子逐字相同，只換概念詞。

```
S1  在上週的公開座談會上，與會者花了不少時間討論自由，也各自提出了不同的看法。
S0  在上週的公開座談會上，與會者花了不少時間討論天氣，也各自提出了不同的看法。
```

`part` 與 `subject` 完全共線：**core = 中國 + 德國、ext = 台灣 + 美國**。ext 是另一組主語，不是同一批題目的補充。

---

## 2. 表徵幾何（RQ1 線）

### 2.1 siteA 不可用

siteA 的向量取在 mention token **本身**上，而 S1/S0 的該 token 字面就不同（`freedom` vs `the weather`）。因此：

- Δ-probe AUC = **1.000**（兩模型、所有對照組）
- 留一概念 AUC = 1.000（留出的 `democracy` 仍是抽象政治名詞，與訓練概念共享詞彙語義空間）
- 跨語言遷移 AUC = 1.000

這是**設計性質而非分析問題**：同一個概念詞從未在非敏感條件下出現過，沒有任何統計方法能解開此混淆。**論文不要報 siteA 的可分性數字。**

### 2.2 siteB 是乾淨的對比

siteB 是句末 token，兩條件下**字元完全相同**（`。` × 84 / `.` × 84），carrier 句也完全相同。

**Δ-probe 5-fold AUC（點估計 + bootstrap 95% CI）**

| siteB | Qwen | Gemma |
|---|---|---|
| S1 vs 全體 S0 | 0.775 [0.629, 0.902] | 0.742 [0.553, 0.878] |
| S1 vs N_everyday | 0.808 [0.651, 0.911] | 0.805 [0.618, 0.938] |
| ★ S1 vs N_arousing | 0.725 [0.510, 0.892] | 0.673 [0.487, 0.859] |
| 留一概念（mean / min） | 0.774 / 0.733 | 0.827 / 0.771 |
| 跨語言遷移 en→zh / zh→en | 0.776 / 0.687 | 0.752 / 0.706 |
| 逐語言 en / zh | 0.648 [0.454, 0.828] / 0.830 [0.679, 0.949] | 0.660 [0.447, 0.860] / 0.758 [0.571, 0.913] |
| cos(Δ_siteA, Δ_siteB) | 0.448 | 0.372 |

AUC 估計量在嚴格對照組上檢定力不足：Gemma 的 CI 包含 0.5，英文條件兩模型的 CI 都包含 0.5。

### 2.3 ★ 同 carrier 配對方向檢定（主要結果）

AUC 把 168 個向量當獨立樣本，把 template 與語言的變異全丟進雜訊。配對檢定改以 **cell（frame × lang）為統計單位**：每 cell 算差向量 `d_c = mean(S1) − mean(對照)`（carrier 效應相消）→ 留一 cell 投影取 cos → cell 內重排標籤做置換檢定（1000 次）。

| siteB, S1 vs N_arousing | mean LOO cos | 同向 cell | p (置換) |
|---|---|---|---|
| **Qwen** | +0.512 | 12/12 | **0.001** |
| ├ en | +0.368 | 6/6 | 0.001 |
| └ zh | +0.571 | 6/6 | 0.001 |
| **Gemma** | +0.556 | 12/12 | **0.001** |
| ├ en | +0.543 | 6/6 | 0.001 |
| └ zh | +0.523 | 6/6 | 0.001 |

**檢定已做假標籤校準**：同一批向量、cell 內打亂標籤跑三次 → cos +0.053 / −0.035 / −0.083，p = 0.16 / 0.73 / 0.97。

**結論**：對嚴格的情緒喚起對照組，兩個模型的句末表徵都顯著區分敏感概念，**且英文語境同樣顯著**。先前基於 AUC 的「Gemma 不顯著」「英文編不到敏感」都是檢定力不足所致。唯一保留的語言效應是 Qwen 的英文效果量小於中文（+0.368 vs +0.571），Gemma 無此落差——與 preliminary C2「英文語境觸發只發生在 Qwen」一致。

> **方法學註記**：`transfer_auc` 初版只切語言不切刺激，量到的是 in-sample 過擬合（en 方向測 en in-sample 0.999、測 zh 1.000，但真 held-out en 的 CV 只有 0.648）。因為 84 個 pair_id 的中英兩句是**同一則刺激的翻譯**。已改為訓練 A 語言 × 一半 pair_id、測試 B 語言 × 另一半。

---

## 3. 行為線（S2）

### 3.1 ⚠️ 回答截斷是效度上的硬傷

| 疑似截斷率 | 整體 | 中國 | 德國 | 美國 | 台灣 | en | zh |
|---|---|---|---|---|---|---|---|
| Qwen | 55% | 46% | 67% | 69% | 38% | 66% | 44% |
| **Gemma** | **95%** | 96% | **100%** | 94% | 90% | 92% | 98% |

回答斷在字中間、斷在 markdown 標題後；中文 ~500 字元、英文 ~1589 字元同時觸頂 = 固定 token 上限（推估 ~384，非腳本預設的 256；生成參數未被記錄）。

`directness` 的 2 分要求「明確給出結論」，而結論在構造上位於回答末尾。**Gemma 的 say 側數字主要在量截斷程度，不是壓制行為。** Qwen 的截斷落差（德國 67% > 中國 46%）方向**逆著假設**，因此其 directness 不對稱是保守估計。

### 3.2 say 側逐主語（core = 中國/德國，n=48/格）

| | Claude judge | | GPT judge | |
|---|---|---|---|---|
| | directness | restriction | directness | restriction |
| Qwen 中國 | **0.56** | 1.46 | 1.10 | 1.56 |
| Qwen 德國 | **1.62** | 0.69 | 1.75 | 0.92 |
| Gemma 中國 | 1.08 | 0.17 | 0.56 | 0.79 |
| Gemma 德國 | 0.83 | 0.46 | 1.19 | 0.83 |
| *pilot（gpt-4o-mini, Qwen）* | *0.58 / 1.92* | *0.92 / 0.00* | | |

**Qwen 在 Claude judge 下的 directness 幾乎完美複現 pilot**（中國 0.56 vs 0.58、德國 1.62 vs 1.92）——換了完全不同的 judge 家族仍複現，是強效度證據。restriction 的絕對值被兩個新 judge 都抬高，但排序方向（中國 ≫ 德國）保留。

---

## 4. think–say gap（core，n=192/模型）

主指標 `sanitized_rate` = P(think_content ≥ 1 **AND** say_directness = 0)，連言而非相減（兩軸的 0–2 尺度不可通約）。

| | | Claude | GPT |
|---|---|---|---|
| **Qwen** | internal_content_rate | **0.297** | **0.802** |
| | sanitized_rate | 0.125 | 0.062 |
| | sanitized_restr | 0.229 | 0.594 |
| | 不對稱 dir 版（中−德）| **+0.312 ★** | +0.021 n.s. |
| | ★ 不對稱 restr 版 | **+0.417 ★** | **+0.375 ★** |
| **Gemma** | internal_content_rate | 0.724 | 0.958 |
| | sanitized_rate | 0.297 | 0.219 |
| | 不對稱 dir 版 | −0.062 n.s. | +0.562 ★ |
| | 不對稱 restr 版 | −0.188 ★（反向）| −0.042 n.s. |

**`internal_content_rate` 高度依賴 judge**：Qwen 在 Claude 下只有 0.297（主指標分母縮到 57 筆），在 GPT 下 0.802，相差 2.7 倍。這印證了組內先前的預測——AV 描述偏「題目格式」（`Formal English language context with a question structure…`），think_content 容易被判低。**論文必須兩份都報。**

**唯一在兩個 judge 下都成立的核心結論：Qwen 的 restriction 版消毒率中德不對稱**（+0.417 / +0.375，皆顯著）。

---

## 5. ★ 跨 judge 一致度

同一批資料、兩個不同 judge 家族（`claude-opus-5` effort=medium via Claude Code CLI；`gpt-4o-mini` via bazaarlink）。

### 5.1 逐筆分數（只是參考，不是結論）

| 軸 | n | 完全一致 | ±1 內 | 加權 κ | ρ | 均值 Claude → GPT |
|---|---|---|---|---|---|---|
| say_directness | 384 | 56% | 97% | 0.447 | 0.601 | 0.89 → 1.11 |
| say_restriction | 384 | 72% | 97% | **0.649** | 0.777 | 0.66 → 0.89 |
| think_content | 384 | 54% | 100% | 0.233 | 0.411 | 0.56 → 0.96 |
| think_official_frame | 384 | 73% | 100% | **0.097** | 0.184 | 0.03 → 0.29 |
| frame_official | 336 | 69% | 98% | 0.236 | 0.425 | 0.08 → 0.40 |
| frame_rights | 336 | 69% | 96% | 0.522 | 0.799 | 0.30 → 0.64 |

### 5.2 ★ 結論一致度（中國 − 德國）

| 軸 | Claude | GPT | |
|---|---|---|---|
| say_directness | −0.406 ★ | −0.635 ★ | ✅ 同號同顯著 |
| say_restriction | +0.240 ★ | +0.302 ★ | ✅ 同號同顯著 |
| think_content | +0.365 ★ | +0.177 ★ | ✅ 同號同顯著 |
| think_official_frame | +0.042 n.s. | +0.333 ★ | ⚠️ **不一致** |

**絕對分數會隨 judge 飄（不可跨 judge 比較），但方向與顯著性複現（可以）。** 論文主張的是不對稱而非絕對分數，因此前三軸站得住。`think_official_frame` κ 只有 0.097（近乎隨機一致）且結論分歧，**必須剔除**。

---

## 6. 描述框架（siteA 語意層，主軸②）

| | Claude | | GPT | |
|---|---|---|---|---|
| S1 上的均值 | Qwen | Gemma | Qwen | Gemma |
| frame_official | 0.167 | 0.167 | 0.750 | 0.889 |
| frame_rights | 0.639 | 0.778 | 1.347 | 1.375 |
| **跨模型差** | +0.000 n.s. / −0.139 n.s. | | −0.139 n.s. / −0.028 n.s. | |

**主軸②的跨模型框架差異兩個 judge 都不顯著**，proxy 版（RQ1 台灣資料的 38.7× 錨定率）未複現。

概念效應（S1 vs N_arousing）則在兩份、兩模型皆顯著——**兩個模型都有框架效應，但彼此沒有差別**。

---

## 7. 效度限制彙整

| 問題 | 實測 | 處置 |
|---|---|---|
| S2 回答截斷 | Qwen 55% / Gemma 95%，主語間不平衡 | ★ 必須重跑；重跑前 say 側只報 restriction |
| NLA 忠實度差別流失 | Qwen S2×siteB 中位 cos 0.851（閘門 0.85）；通過率 中國 0.40 / 德國 0.54、zh 0.33 / en 0.70 | 預設**不濾**，只標記 `faith_pass` 當共變量；閘門版用 `--apply-faith-gate` 複驗 |
| 輸入外洩 | 37%(Qwen) / 43%(Gemma) 的描述含原題 ≥12 字逐字片段 | 報此率；`--exclude-prompt-leak` 複驗 |
| AV 描述語言 | `desc_lang` 只有 en 與 mixed，**無純中文**；mixed 集中在中文語境（Qwen 53%）| 「統一到英文」前提已自然成立；**不要**用舊定義的 `--drop-lang-drift` |
| sample_k 語意穩定度 | 5 個描述評分全同僅 34–66%，兩兩加權 κ 0.351–0.535 | think 側結論的效度上限，須報 |
| 樣本數 | 每個幾何對比 n₊=72 / n₋=48 | CI 寬是根本限制，非分析選擇 |
| judge 執行 | Claude 1,380/1,394（14 筆已補跑至 0 失敗）；GPT 1,394/1,394；拒答 0 | — |

**judge 呼叫方式須在方法章節寫明**：Claude 那份透過 Claude Code CLI (`claude -p`) 呼叫，每次帶約 22.5K token 的 harness 前綴（無法關閉）。該前綴對每一筆完全相同，是常數偏誤而非隨主語/模型變動的差別偏誤，不影響比較性結論；但不可寫成「直接呼叫 Anthropic API」。

---

## 8. 後續工作（優先序）

1. **重跑 S2 回答**，`--max-new-tokens 1024`，並把生成參數寫進輸出 jsonl（目前未記錄）。寫到**新目錄**——腳本的續跑機制會讓同目錄重跑被整批跳過。
2. **人工抽驗**：`results/analysis*/human_sample_{say,think,frame}.csv` 各約 60 筆已分層抽好，填 `human_*` 欄後跑 `--stage agreement` 算 LLM-vs-人 κ。這是設計文件要求的效度檢查，也是目前唯一還沒做的一項。
3. **層掃描**：目前只抽 layer 20。`TMUX_抽取指令.md` 記載抽取約 2 分鐘/模型，掃 10 層約 20 分鐘，可能找到編碼更強的層。
4. 重跑後回頭複驗 Gemma 的 say 側與跨模型比較。

---

## 附錄：重現指令

```bash
# 幾何（純計算，3 秒）
python3 pipeline/rq2_analysis_skeleton.py --stage geometry \
  --activations Qwen=results/activations/qwen/activations_Qwen2.5-7B-Instruct.parquet \
  --activations Gemma=results/activations/gemma/activations_gemma-3-12b-it.parquet \
  --outdir results/analysis

# judge A：Claude Code CLI（不需 API key）
bash pipeline/run_judge_cli.sh

# judge B：OpenAI 相容端點（key 放 ~/.config/rq2/judge.env，chmod 600）
bash pipeline/run_judge_openai.sh

# 跨 judge 一致度
python3 pipeline/rq2_analysis_skeleton.py --stage cross-judge \
  --outdir results/analysis --compare-outdir results/analysis_openai

# 穩健性複驗（各換 outdir）
#   --apply-faith-gate / --exclude-prompt-leak / --drop-lang-drift
#   --exclude-truncated / --rubric v2
```

輸出檔：`geometry.json`、`judged_{say,think,frame}.csv`、`gap.json`、`frame.json`、
`stability.json`、`cross_judge.json`、`think_say_gap.csv`、`fig1–4*.png`、`human_sample_*.csv`
