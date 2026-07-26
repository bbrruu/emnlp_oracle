# RQ2 實驗 Roadmap · 分工 · 資料需求

**投稿**：EMNLP 2026 · ORACLE Workshop ｜ **Deadline**：2026/08/01（約 8 天）
**現況**：已有強初步證據（RQ1 proxy，見 `RQ2_初步分析報告.md`）；管線、NLA、層數皆就緒。

---

## 0. 對外 Headline（論文只答這一句）

> **Qwen 對敏感議題（自由/民主/人權）的「內部表徵」與「實際輸出」是否存在系統性落差（think–say gap）？此落差主要源於 RLHF 對齊，還是文化？**（以 Gemma 為跨模型基準。）

---

## 1. 八個實驗單元（Experiment Units）與所需資料

| ID | 實驗 | 服務的證據線 | 需要的資料 | 需 GPU? |
|---|---|---|---|---|
| **E1** | 表徵可分性 + siteA/B 幾何 | A（方法基礎） | 360 stimuli 的 activation（Qwen L20 / Gemma L32, siteA/B） | 抽取需 |
| **E2** | 跨模型錨定/框架（AV 語言化） | 主線 | E1 的向量 → NLA AV 描述 + `mse_nrm` | 需（NLA 推論） |
| **E3** | 語言觸發效應（en vs zh） | 主線 | 雙語 stimuli（已內建於 360） | 併入 E1/E2 |
| **E4** | 壓制 / 拒答 dose–response（L0–L4） | E 主線① | L0–L4 stance prompt 的**模型生成輸出** + 拒答標籤 | 生成需 |
| **E5** | **think–say gap**（賣點） | E 主線② | 同題的〔AV 描述（想）〕vs〔實際生成（說）〕配對 | 需（E2+E4 產物） |
| **E6** | 文化 vs RLHF 歸因 | D（歸因） | base vs instruct 的**行為輸出 + 原始 Δ**（不需重訓 NLA） | 生成+抽取需 |
| **E7** | steering 因果驗證（選配） | C（因果） | 敏感方向向量 + steering 生成 | 需 |
| **E8** | NLA 忠實度閘門（方法） | 全線品質 | 每則描述的 `mse_nrm`（E2 附帶） | 併入 E2 |

> 關鍵：**E1/E3 的「表徵幾何」在 RQ1 proxy 上已驗證可行**（AUC≈0.99、跨模型差 38.7×）。RQ2 只是把 stimuli 換成抽象概念。**E4/E5 是全新、也是最有價值、最需要 GPU 生成的部分。**

---

## 2. 四條工作流（分工）

| 工作流 | 負責人 | 需 GPU? | 負責的實驗 / 產物 |
|---|---|---|---|
| **WS-D 資料** | 1–2 人 | ❌ | 潤稿+QC 360 句、拒答標註、AV 描述標註（E4 標籤、E2/E5 標註） |
| **WS-G 算力**（國網帳號） | 1 人 | ✅ | 抽取(E1)、NLA 推論(E2/E8)、生成(E4/E6)、steering(E7) |
| **WS-A 分析** | 1–2 人 | ❌ | E1–E6 分析碼、圖表、CI、think-say 計算（`rq2_*.py` 已備底） |
| **WS-W 寫作/PM** | 1 人 | ❌ | 論文骨架、related work、範圍控制、整合結果 |

**核心觀念**：WS-G 是「跑幾次短 GPU 批次」的角色,不是瓶頸。WS-D / WS-A / WS-W **第一天就能零 GPU 全速啟動**。沒有國網帳號的人扛下大半專案。

---

## 3. Roadmap（8 天，含兩個 de-risk 關卡）

### Phase 0 · Day 1（7/24）— 啟動
- 敲定 headline + 收斂到 E1–E6（E7 steering 列選配）。
- WS-D 開始潤稿 core192；WS-A 凍結分析碼；WS-W 起 related work。

### Phase 1 · Day 2–3（7/25–26）— **兩個 de-risk 關卡**
- 🚦**Gate①（WS-D，可用免費 Colab/API）**：跑 Qwen 生成 L0–L4，量拒答率。**確認 L3/L4 踩到壓制門檻**。沒觸發 → 調 prompt，別往下。
- 🚦**Gate②（WS-G）**：拿**現有 RQ1 activation** 跑一次 NLA 推論 smoke test，確認 AV/AR 串通、`mse_nrm` 合理。
- WS-D 完成 360 QC（naturalness / 命題等價）。

### Phase 2 · Day 3–4（7/27）— 全量產資料（GPU）
- **E1 抽取**：360 stimuli → Qwen/Gemma × siteA/B × L20/L32 activation.parquet（沿用你們現有 `extract_activations.py`）。
- **E4/E6 生成**：跑 instruct（+base）在 stance prompt 上的輸出，存拒答標記。

### Phase 3 · Day 4–5（7/28）— 語言化 + 標註
- **E2/E8 NLA**：AV/AR 跑 360×2 → 描述 + `mse_nrm` 閘門（Qwen 用更嚴門檻）。
- WS-D 標註新描述的錨定/框架（可先自動、抽樣人工校）。

### Phase 4 · Day 5–6（7/29）— 分析
- **E1/E3**：可分性、siteA/B、跨語言一致性（套 `rq2_full_analysis.py`）。
- **E2**：Qwen vs Gemma 概念錨定率 + bootstrap CI。
- **E5 think–say gap**：同題〔AV 描述〕vs〔實際生成〕落差量化。
- **E6**：base vs instruct 行為 + Δ 幾何比較。

### Phase 5 · Day 6–7（7/30）— 收尾分析 + 寫作
- 補 CI / z-scored 對照 / limitations；E7 steering（若有餘裕）。
- WS-W 組裝 results + 圖，寫 discussion（文化 vs RLHF 誠實框定）。

### Phase 6 · Day 7–8（7/31–8/1）— 定稿投稿
- 全文潤飾、abstract、reproducibility 附錄、submit。**Day 8 留 buffer。**

---

## 4. 依賴關係（關鍵路徑）

```
QC stimuli(WS-D) ──▶ 抽取 E1(WS-G) ──▶ NLA E2(WS-G) ──┐
        └──▶ 生成 E4/E6(WS-G) ──────────────────────┤
                                                     ▼
                             分析 E1–E3 / E5 think-say / E6(WS-A)
                                                     ▼
                                        寫作整合(WS-W) ──▶ 投稿
```
兩個 Gate（拒答 pilot、NLA smoke）**必須在全量產資料前通過**，否則卡住整條路。

---

## 5. 風險與對策（延續初步報告）

| 風險 | 對策 |
|---|---|
| Qwen NLA 忠實度較低（cos 0.90 vs Gemma 1.00） | Qwen 用更嚴 `mse_nrm` 閘門、逐模型分報、描述謹慎解讀 |
| L3/L4 沒觸發壓制 → think-say 崩 | Gate① 先驗；備妥主語替換 + 更嗆變體 |
| 位置框架漂移大（cos_A,B≈0.5） | siteA/B 一律分開報 |
| 中文語境 AV 語言漂移高 | 比較前統一翻譯；漂移率當附帶結果 |
| 只有一個 GPU 帳號 | GPU 工作切成短批次；其餘全部平行零 GPU |
| 8 天太趕 | E7 steering 可移 future work；先鎖 E1–E6 |
