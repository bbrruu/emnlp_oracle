# RQ2 進度總覽（給組員）

**投稿目標**：EMNLP 2026 · ORACLE Workshop（Open Reasoning Across Cultures and Languages）
**Deadline**：2026/08/01
**更新日期**：2026-07-24
**一句話**：初步證據 + pilot **都通過**，**綠燈可以做下去**；主故事已驗證存在且中國特異，pilot 顯示壓制是「官方框架式軟迴避」而非硬拒答。

---

## 1. 研究問題（對外單一 Headline）

> **在自由/民主/人權等敏感議題上，開源中文模型（Qwen）與英語模型（Gemma）的內部表徵機制有何不同——尤其，兩者在「內部表徵（想）」與「實際輸出（說）」之間是否出現不同程度的 think–say gap？**

老師 instruction 的三個成分如何收進這一句：

| 成分 | 角色 |
|---|---|
| Qwen vs Gemma 內部機制差異 | **主軸**（比較框架） |
| NLA 作 tertium comparationis | **方法**（跨模型比較基準） |
| think–say gap | **核心貢獻／賣點** |
| 文化 vs RLHF | discussion／歸因（不過度宣稱） |
| crosscoder | 砍到 future work（跨家族+算力，不可行） |
| steering | 選配（有餘力補因果） |

---

## 2. 方法與關鍵決策

- **模型**：`Qwen2.5-7B-Instruct`（層 20）＋`Gemma-3-12B-IT`（層 32）。此配對正好是官方 **NLA checkpoint**（`kitft/nla-*`）支援的兩個模型與層，**零額外訓練**。
- **NLA 作跨模型橋樑**：兩模型幾何不可比，故各自把 activation「說成英文」再比文字。忠實度用 AR 的 `mse_nrm`（=2(1−cos)）閘門。
- **crosscoder 不做**：Qwen/Gemma 不同 tokenizer、不同 d_model，無法逐 token 配對，是前沿題＋要算力；NLA 已解決同一需求。
- **抽取位置（沿用 RQ1）**：
  - `siteA` = 概念 token（句中，需前文足）→ **表徵**探針
  - `siteB` = 句末 token（開口前狀態）→ **壓制／think–say** 探針
- **base vs instruct**：非必要。RQ1 activation 與官方 NLA 皆僅 instruct；文化 vs RLHF 先放 discussion，要做就做「行為層」輕量版（不需重訓 NLA）。

---

## 3. 刺激材料（Stimuli）

- **360 句 = core192（建議，主結果）+ ext168（補充，robustness）**，檔案 `rq2_stimuli/rq2_stimuli_v3.csv`。
- **層級砍成 3 檔**（原 L0–L4 太細、且與層數 L/假設 H/標註 D 撞名）：

| 級 | 意思 | 服務 |
|---|---|---|
| **S0** 中性對照 | 中性概念 | Δ 減項（表徵） |
| **S1** 敏感描述 | 敏感概念、陳述句（概念嵌中段、前文足） | siteA 表徵探針 |
| **S2** 敏感表態 | 敏感概念 × 主語、逼問（+ `stance_strength` mild/strong） | siteB 壓制／think–say |

- **表徵題已改 RQ1 式長句**（概念嵌中後段），實測 leadin 中位：中文 21 字／英文 14 詞，**無一句前文不足**。
- **對等性規範 §4.2**（`rq2_stimuli/RQ2_對等性規範_4.2.md`）：命題對等＋回譯、自然度、±20% 長度（兩 tokenizer）、固定概念詞面、**概念不得卡專名（siteA 才乾淨）**、主語替換對等、多 subtoken 抽最後一個。
- 工具：`check_norms.py`（norm 掃描，已過 0 違規）、`verify_tokenization.py`＋`rq2_pairs.csv`（leadin/±20% 覆核，需在國網跑，Gemma gated 要 HF 登入）。

---

## 4. 結果一：初步分析（用現有 RQ1 資料，零 GPU）

> RQ1 資料以「台灣」為敏感 proxy（對 Qwen 政治敏感），描述句。已有 activation（Qwen L20／Gemma L32，siteA/B）＋**已跑完的 7200 則 AV 語言化**＋framing 標註。詳見 `rq2_preliminary/RQ2_初步分析報告.md`。

**表徵幾何（方向法，跨模型可比純量）**
- 敏感 vs 中性 **近乎完美可分**：Qwen AUC A 0.99／B 0.98，Gemma A 1.00／B 0.99 → 訊號真實存在。
- **位置框架漂移大**：`cos(Δ_siteA,Δ_siteB)` ≈ 0.52（Qwen）/0.47（Gemma）→ siteA/B 抓不同東西，**兩者分開報**。

**跨模型錨定（AV 語言化，主證雛形）**
- **只看非政治的中性框架**、英文語境，台灣→中國「錨定率」：**Qwen 0.322 vs Gemma 0.008 = 38.7×**，bootstrap 95% CI [+0.267, +0.364]（顯著）。排除政治題材後差異**不減反增** → 擋掉 confound。
- 英文語境觸發（中性框架）：**只發生在 Qwen**（en−zh +0.281，顯著）；Gemma 無（n.s.）。
- 質化：同一句台灣文化題，Qwen 一律往「Chinese-language／泛中國」靠，Gemma 保持「Taiwan」為獨立主體。

**NLA 忠實度（可行性 + 風險）**
- Gemma round-trip cos **0.996**（極佳）；Qwen **0.904**（明顯較差，MSE 高 20 倍）→ **Qwen 描述要用更嚴閘門、逐模型分報、謹慎解讀**。

**AV 輸出語言漂移**：英文語境 0；中文語境會大量漂成非中文（Qwen 0.48／Gemma 0.63）→ 比較前統一翻譯。

**think–say proxy**：Qwen 的 AV 錨定率隨內部敏感投影**單調上升**（0.15/0.17/0.23）；Gemma 全平。

---

## 5. 結果二：壓制 Pilot（Qwen2.5-7B-Instruct，24 題）

> 3 概念 × 中/德 × mild/strong × 中英。詳見 `pilot/PILOT_判讀.md`。

**核心發現：Qwen 不硬拒答（硬拒答率 0%），而是「官方框架式軟迴避」，且中國特異。**

用**中立 judge**（以 directness + 政治性限制合理化兩軸）評分，結果乾淨：

**directness（0=迴避 ~ 2=明確表態）× 強度**

| | mild | strong |
|---|---|---|
| 中國 | 1.0 | **0.17** |
| 德國 | 2.0 | **1.83** |

**兩軸 × 主語**

| | directness | restriction（政治性） |
|---|---|---|
| 中國 | 0.58 | **0.92** |
| 德國 | **1.92** | **0.00** |

**judge 選擇很重要（三個 judge 對照，directness 主語平均）：**

| judge | 中國 | 德國 | gap |
|---|---|---|---|
| 關鍵字（粗略） | 0.58 | 1.42 | 0.84 |
| 本地 Qwen（**自評**） | 1.08 | 1.50 | 0.42 ← 被壓扁 |
| **中立 judge（乾淨）** | 0.58 | **1.92** | **1.34** |

**判讀（回應「德國也有官方腔怎麼分」）：**
- **收窄後的 restriction 軸完美分離**：中國 0.92 / 德國 **0.00**。德國確實也合理化限制（仇恨言論法），但那是**權利保護型**（dignity/hate speech）→ 給 0；中國是**政治管控型**（國家安全/顛覆政權/穩定）→ 給高分。差別在**「用什麼理由合理化限制」**，不是「有沒有」。
- **不能用 Qwen 自評**：Qwen 判自己時捨不得給中國低分、也低估德國（gap 只剩 0.42）；換中立 judge 後 gap 回到 1.34。正式跑要用中立 API judge（Gemini/GPT-4o）。
- **「批評意願」不是判別器**（兩邊相同），已從主張中拿掉。
- 三個獨立方法**方向全一致** → 收斂效度。

**測量方式的教訓**：二元「拒答/照答」測不到（全 0%）；改用 **directness + 政治性限制合理化** 兩軸（中立 LLM-judge + 人工校，沿用 RQ1 標註機制）。

---

## 6. 重要效度立場（論文必寫）

**「用官方框架」≠「壓制」**——這是 proxy，不是定義，且與「訓練語料/文化」混淆：
- (a) 壓制/RLHF：內部有敏感內容、輸出消毒 → **想≠說（有 gap）**。
- (b) 語料/文化：內部也只有官方版（真心這麼想）→ **想=說（無 gap）**。

**只測輸出永遠分不開 (a)/(b)。** 把「官方框架」升級成「壓制」需三者疊加：中國 vs 德國不對稱（已有）＋ base/instruct（歸因）＋ **think–say gap（決定性）**。論文措辭：先報可觀測行為（官方框架/低直接度），再用 think–say 論證「輸出端選擇性消毒，而非內部認知缺乏」。

---

## 7. 目前進度

| 項目 | 狀態 |
|---|---|
| 研究問題與方法對齊 | ✅ 完成 |
| 模型/層/NLA checkpoint 確認 | ✅（Qwen L20 / Gemma L32，官方 NLA 支援） |
| 360 stimuli 設計（v3, leadin 修正, S0–S2） | ✅ 草稿完成，待 WS-D 潤稿+QC |
| §4.2 對等性規範 + norm 掃描 | ✅（norm-5 零違規） |
| tokenizer 驗證（leadin/±20%） | ⏳ 腳本備好，待國網跑真 tokenizer |
| 初步分析（幾何+AV，RQ1 proxy） | ✅ 完成，結果強 |
| 壓制 pilot（Qwen） | ✅ 完成，**Gate 過**（軟壓制、中國特異） |
| LLM-judge（directness+政治性限制合理化） | ✅ 腳本完成 + 中立 judge 已評（`pilot/pilot_judged_claude.csv`）；正式版待接 API judge |
| 全量抽取 / NLA / 生成（360） | ⬜ 未開始（需國網/Colab） |
| think–say 主實驗（siteB → NLA vs 輸出） | ⬜ 未開始 |
| 論文撰寫 | ⬜ 未開始 |

---

## 8. 下一步（建議順序）

1. **LLM-judge 接 API**：pilot 已用中立 judge 拿到乾淨數字；正式跑接 Gemini/GPT-4o API judge，並抽樣人工校驗（算一致度）。
2. **WS-D 潤稿 core192**：尤其 S2 嗆題；跑 `verify_tokenization.py` 過 leadin/±20%。
3. **放大**：core192 全量跑生成 → 壓制率 + dose-response；抽 activation → NLA。
4. **think–say 主實驗**：siteB 的 NLA 描述 vs 實際輸出，量落差，比 Qwen/Gemma。
5. **寫作**：headline 單一問題，A/C 支撐、E/D 主線、B 進 future work。

**去風險化的地板**：就算 think–say 沒做出來，「Qwen vs Gemma 表徵差異（NLA+幾何）」本身已是合格 workshop paper，且已有數據支撐。

---

## 9. 檔案索引（都在 `LOPE/EMNLP/`）

```
EMNLP/
├── RQ2_進度總覽.md              ← 本檔
├── RQ2_研究設計.md              ← 完整研究設計（headline/證據線/方法/混淆控制）
├── rq2_preliminary/            ← 初步分析
│   ├── RQ2_初步分析報告.md       ← 幾何+AV 完整結果
│   ├── RQ2_實驗Roadmap與分工.md  ← 8 天 roadmap + 四工作流
│   ├── RQ2_blueprint.svg        ← 實驗流程藍圖
│   ├── fig1–4.png / rq2_results.json
│   └── rq2_full_analysis.py / rq2_deepen.py / derisk_representation.py
├── rq2_stimuli/                ← 刺激材料
│   ├── rq2_stimuli_v3.csv       ← 360 句（最新）
│   ├── RQ2_對等性規範_4.2.md
│   ├── check_norms.py / verify_tokenization.py / rq2_pairs.csv
│   └── RUN_verify_tokenization.md
└── pilot/                      ← 壓制 pilot
    ├── RQ2_pilot_colab.ipynb    ← 生成 notebook（跑 Qwen 拿回答）
    ├── RQ2_judge_colab.ipynb    ← LLM-judge notebook（API/本地打分）
    ├── pilot_results.csv        ← 24 題原始輸出
    ├── pilot_judged_claude.csv  ← 中立 judge 評分（乾淨版）
    └── PILOT_判讀.md            ← pilot 判讀
```
