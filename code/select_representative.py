#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_representative.py — RQ2 ③c：每向量取「min-MSE 代表」(k=5 → 1)
=====================================================================
把 verbalize.py 的 k=5 描述表 join 上 score_roundtrip.py 的忠實度分數，
對**每個向量**在它自己的 k 則裡挑 mse 最低那 1 則當代表，輸出「每向量 1 列」的
代表表，供下游 judge / 分析使用。

政策（與老師確認）：
  * **不做忠實度門檻、不丟任何向量**（τ 門檻會砍掉約 2/3 資料，不採用）。
  * mse 只用來「在 k 則裡挑最低」，並**保留為報告欄**（descriptive，不拿來刪列）。
  * 原始 k=5 表（3,600 則/模型 = 7,200 則）**原封保留當 backup**，本腳本不動它。

輸入 / 輸出（每模型各一份）：
  verbalizations/qwen_av.parquet  (3,600 列)  ┐
  roundtrip/qwen_roundtrip.csv    (3,600 列)  ┴─► representatives/qwen_rep.parquet (720 列)

720 = 360 句 × 2 site(siteA + siteB)。siteA/siteB 各 360。

用法：
  python3 select_representative.py \
      --verbalizations verbalizations/qwen_av.parquet \
      --roundtrip roundtrip/qwen_roundtrip.csv \
      --out representatives/qwen_rep.parquet

  # Gemma 同理，換路徑即可。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbalizations", type=Path, required=True,
                    help="verbalize.py 的 k=5 描述 parquet（backup 來源，不會被改動）")
    ap.add_argument("--roundtrip", type=Path, required=True,
                    help="score_roundtrip.py 的忠實度 csv（提供 mse/cos）")
    ap.add_argument("--out", type=Path, required=True,
                    help="輸出：每向量 1 列的代表 parquet")
    args = ap.parse_args()

    # 1) 讀 k=5 描述（帶完整 metadata + description + desc_id/vector_id/sample_k）
    vb = pq.read_table(args.verbalizations).to_pandas()
    # 2) 讀忠實度分數，只取需要的欄，用 desc_id join
    rt = pd.read_csv(args.roundtrip, encoding="utf-8-sig")
    score = rt[["desc_id", "mse", "cos"]].drop_duplicates("desc_id")
    df = vb.merge(score, on="desc_id", how="left")

    n_desc = len(df)
    n_vec = df["vector_id"].nunique()
    n_missing_score = int(df["mse"].isna().sum())

    # 3) 每向量挑 min-mse 代表：整列排序後保留第一列（NaN mse 排最後，退而用 sample_k=0）
    #    用 drop_duplicates(keep="first") 而非 groupby().first()，
    #    後者會逐欄跳過 NaN → 可能混到不同列，是常見陷阱。
    df["_mse_sort"] = df["mse"].fillna(float("inf"))
    df = df.sort_values(["vector_id", "_mse_sort", "sample_k"], kind="mergesort")
    rep = df.drop_duplicates(subset="vector_id", keep="first").copy()
    rep = rep.drop(columns=["_mse_sort"])
    rep = rep.rename(columns={"sample_k": "won_k"})   # 哪一個 k 勝出

    # 4) 寫出
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(rep, preserve_index=False), args.out)

    # 5) 摘要
    n_rep = len(rep)
    site_counts = rep["site"].value_counts().to_dict() if "site" in rep.columns else {}
    scored = rep["mse"].dropna()
    print(f"===== select_representative =====")
    print(f"  輸入描述：{n_desc} 則；向量數：{n_vec}")
    if n_missing_score:
        print(f"  [注意] {n_missing_score} 則描述在 roundtrip 找不到分數（join 後為 NaN）；"
              f"該向量若整組皆缺分數，退用 k=0 當代表。")
    print(f"  代表輸出：{n_rep} 列（每向量 1 則）  site 分布：{site_counts}")
    if len(scored):
        print(f"  代表 mse：mean={scored.mean():.4f}  median={scored.median():.4f}  "
              f"min={scored.min():.4f}  max={scored.max():.4f}（僅報告，未用於篩選）")
    n_rep_missing = int(rep["mse"].isna().sum())
    if n_rep_missing:
        print(f"  [注意] {n_rep_missing} 個代表無 mse（該向量整組缺分數）。")
    print(f"  -> {args.out}")
    print(f"  （原始 k=5 表 {args.verbalizations} 未更動，續作 backup。）")


if __name__ == "__main__":
    main()
