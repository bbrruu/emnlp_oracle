# 在 Overleaf 開這份稿子

## 方法一(推薦):用官方 ACL 範本
1. Overleaf → **New Project → Templates**,搜尋 **"ACL"**,選官方 ACL 樣式範本(含 `acl.sty`、`acl_natbib.bst`)開一個新專案。
2. 把本資料夾的 **`main.tex`** 內容整個貼進專案的 `main.tex`(覆蓋)。
3. 把 **`references.bib`** 上傳到專案(覆蓋範本原本的 .bib)。
4. Recompile。

## 方法二:自己上傳樣式檔
1. 去 **github.com/acl-org/acl-style-files** 下載 `acl.sty` 和 `acl_natbib.bst`。
2. Overleaf 開空白專案,上傳 `acl.sty`、`acl_natbib.bst`、`main.tex`、`references.bib`。
3. Recompile。

## 注意事項
- 投稿(匿名)版用 `\usepackage[]{acl}`;**最終版(camera-ready)改成 `\usepackage[final]{acl}`** 並填作者。
- **Limitations 段是必需的**,已放在 Conclusion 之前(ACL 規定)。
- 長論文 8 頁、短論文 4 頁上限(References 不計)。
- `references.bib` 每一筆的作者/年份/出處**要核對後再定稿**(草稿先放合理值)。
- 文中 `% [TODO]` 標記處 = 叢集恢復、跑完全量後要補的內容:
  - §Results 4.1:用正式 RQ2 stimuli(S0/S1)重跑,取代 Taiwan proxy。
  - §Results 4.3:think–say gap 全量結果 + Qwen vs Gemma 比較。
  - Abstract 末:補一句量化結論。

## 現在稿子已含
- 完整 Intro / Related Work / Method / Discussion / Limitations / Conclusion。
- Results 4.1(proxy 幾何 + 錨定 38.7×)、4.2(pilot directness/restriction 表)已填真數字。
- 效度論證(官方框架 ≠ 壓制,靠 think–say 拆)已寫進 Discussion。
