# RQ2 專案接手文件（給後續的人 / LLM）

> 目的：讓接手者不看聊天記錄也能完整承接。投稿 EMNLP 2026 · ORACLE Workshop，deadline **2026/08/01**。
> 更新：2026-07-27。進度 ~55%。

---

## 1. 研究主軸
開源**中文模型（Qwen2.5-7B-Instruct）**與**英語模型（Gemma-3-12B-IT）**在敏感議題（自由 / 民主 / 人權）上，**內部表徵機制有何不同**；尤其 Qwen 是否存在「內部表徵 ≠ 實際輸出」的 **think–say gap**，以及此落差是 **RLHF 對齊**還是**預訓練文化**的產物。
方法核心：兩模型 `d_model`／層數／幾何不可比，故用 **Natural Language Autoencoder（NLA）把各自內部狀態說成自然語言**，以語言為 **tertium comparationis** 做跨模型比較。

## 2. 研究問題
- **RQ1**：中文與英文開源 LLM，對敏感概念的**內部表徵**是否不同？
- **RQ2**：Qwen 的**內部表徵（think）**與**生成輸出（say）**之間是否存在可觀測落差？

## 3. Data 與方法論

### 模型
- **Qwen2.5-7B-Instruct**（抽 residual stream **第 20 層**）
- **Gemma-3-12B-IT**（第 **32 層**）
- 兩者正好是官方 NLA checkpoint（`kitft/nla-*`）支援的模型與層；bf16 抽取。

### Stimuli：`rq2_stimuli/rq2_stimuli_FINAL.csv`（360 句）
- **S2 壓制題 192**（問句）：敏感概念 × 主語（中/德/美/台）× {mild, strong} × {zh, en}。→ 生成回答 + siteB → think–say。
- **S0 中性 96 + S1 敏感 72 = 表徵題 168**（陳述句，概念嵌中段）：→ siteA → 表徵分析。
- 規範（§4.2）：命題等價（回譯）、naturalness ≥4、±20% 長度（兩 tokenizer）、固定概念詞面、概念不卡專名。已過 `verify_tokenization`（leadin/±20%）與 `check_norms`（norm-5 零違規）。

### 抽取位置
- **siteA** = 概念最後一個 subtoken（表徵；需前文足）。
- **siteB** = 句末 token（開口前狀態；think–say）。

### 管線（腳本都在 `pipeline/`）
1. `rq2_extract_activations.py` → activation parquet（720 向量/模型，含 RQ2 metadata）。**bf16、國網 GPU。**
2. `verbalize.py`（RQ1 的，NLA AV/AR，需 SGLang server）→ 描述 + `mse_nrm` 忠實度。**360 都要（siteA+siteB）。**
3. `rq2_generate_responses.py`（**bf16**）→ 回答（say），只跑 192 壓制題。含續跑（JSONL）。
   - `pipeline/rq2_generate_colab.ipynb` = Colab 4-bit 版（國網掛掉時的備份；已寫 Drive 續跑）。
4. **Judge**：directness（0–2）+ 政治性限制合理化（0–2），中立 LLM judge（Gemini 免費 / Claude）+ 人工抽驗 ~15%。

### 分析分岔（步驟 ④）
- **RQ1 幾何 → 純程式，不用人**：Δ = mean(S1 敏感) − mean(S0 中性)；可分性 AUC；跨語言 cos(Δ_en, Δ_zh)；bootstrap CI。（參考 `rq2_preliminary/rq2_full_analysis.py`、`rq2_deepen.py`。）
- **語意 judge → LLM judge + 人抽驗**：RQ1 的「描述框架」（Qwen vs Gemma 怎麼描述概念）+ RQ2 的「想 vs 說」。

### 目前證據（proxy + pilot，非全量）
- **Proxy（RQ1 台灣資料）**：可分性 AUC ≈ 0.99；錨定 Qwen 0.322 vs Gemma 0.008 = **38.7×**（CI [+0.267,+0.364]）；NLA 忠實度 Qwen cos 0.90 / Gemma 1.00。
- **Pilot（Qwen 24 題）**：硬拒答 **0%**；**軟壓制**；directness 中 0.58 / 德 1.92；政治性限制 中 0.92 / 德 0.00。

### 關鍵立場與決策
- **官方框架 ≠ 壓制**：output 混淆 alignment 與 culture，需 **think–say gap + 國家替換**才能歸因。measurement 用 **directness / 政治性限制**，非二元拒答。
- **不用 Qwen 當 judge**（自評偏誤）→ 用中立 judge。
- **砍 crosscoder / SAE**（跨家族不可行 + 現成 SAE 版本對不上 Qwen2.5/Gemma-3）→ future work。
- **base-vs-instruct 降選配 / discussion**（NLA 只有 instruct 版）。
- **精度**：think–say 最終配對用 **bf16 activation 配 bf16 response**；Colab 4-bit 回答當備份 / 行為分析。

## 4. Roadmap（~55%）
```
✅ proxy 初步分析 / pilot（Gate 過）/ 定版 360（驗 token 過）
✅ Qwen 回答（Colab）; 論文 Intro / Related Work / bib 草稿
🔄 Gemma 回答（Colab）; 抽 activation（等國網 bf16）
⬜ ③ NLA 語言化（360：siteA + siteB）
⬜ ④ 分析：RQ1 幾何（純程式）+ 語意 judge（RQ1 描述 + RQ2 想vs說，judge+人抽驗）
⬜ ⑤ Qwen vs Gemma 比較 + think–say 量化 → 填 Results 三處 [TODO] → 投稿
```

### 接手者「下一步」清單
1. **抽 activation**（國網 tmux）：`rq2_extract_activations.py`，Qwen + Gemma，各 720 向量。
2. **bf16 回答**（國網）：`rq2_generate_responses.py`（正式版，取代 Colab 4-bit 做 think–say）。
3. **NLA 語言化**：掛 SGLang + AV/AR checkpoint，跑 `verbalize.py`（360）。
4. **Judge**：對回答（say）+ AV 描述（think）評分；抽 15% 人工驗。
5. **分析**：RQ1 幾何腳本 + judge 結果 → 填論文 Results §4.1/§4.2/§4.3 的 `[TODO]`。

## 5. 檔案索引（`Desktop/LOPE/EMNLP/`）
```
EMNLP/
├── HANDOFF_專案接手.md          ← 本檔
├── RQ2_進度總覽.md              結果 + 進度（含 pilot 乾淨數字）
├── RQ2_研究設計.md              完整研究設計（已對齊現況）
├── RQ2_Roadmap圖.md             視覺路線圖 + 分析分工
├── rq2_stimuli/
│   ├── rq2_stimuli_FINAL.csv    ★ 定版語料 360
│   ├── rq2_pairs.csv            給 verify_tokenization
│   ├── RQ2_對等性規範_4.2.md、check_norms.py、verify_tokenization.py、STIMULI_定案說明.md
├── pipeline/
│   ├── rq2_extract_activations.py  抽 activation（bf16）
│   ├── rq2_generate_responses.py   生成回答（bf16）
│   ├── rq2_generate_colab.ipynb    Colab 4-bit 生成（備份，Drive 續跑）
│   ├── rq2_run_slurm.py、TMUX_抽取指令.md、requirements.txt
├── pilot/
│   ├── pilot_results.csv、pilot_judged_claude.csv、PILOT_判讀.md
│   ├── RQ2_pilot_colab.ipynb、RQ2_judge_colab.ipynb
├── rq2_preliminary/
│   ├── rq2_full_analysis.py、rq2_deepen.py、derisk_representation.py
│   ├── RQ2_初步分析報告.md、fig1–4.png、rq2_results.json
└── paper/
    ├── main.tex、references.bib、README_Overleaf.md
```

## 6. 已知坑 / 注意事項
- **國網（nano5.nchc.org.tw ＝ 登入後主機 cbi-lgn01，2×H100 80GB）時好時壞**；連不上多為節點維護，換 nano1–4 或等。抽取只需 ~2 分鐘/模型，抓穩定空檔即可；tmux 保險。
- **共用 Tyler 帳號**：別 `hf auth login/logout`（會蓋他的 token），Gemma 用 `export HF_TOKEN=` 或已 cache 的離線讀（`HF_HUB_OFFLINE=1`）。
- **Colab**：runtime 重置會清 `/content` → 輸出寫 Drive；筆電別睡。
- **bib**：`[NEW]` 6 筆要核對、`chinesecensorship` 是 placeholder 要換；內文 `\citep` key 要跟 bib 一致（見 references.bib 圖例）。
- **論文 Results 三處 `[TODO]`**：§4.1 proxy 要換正式 stimuli、§4.2 pilot 放大全量、§4.3 think–say 待補；**別把 proxy/pilot 當全量結果呈現**。
