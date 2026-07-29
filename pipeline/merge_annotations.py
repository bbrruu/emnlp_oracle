#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_annotations.py — 合併三人的標註，算人 vs 人一致度，並回填成
rq2_analysis_skeleton.py --stage agreement 吃得下的格式。

用法：
    python3 pipeline/merge_annotations.py
    python3 pipeline/merge_annotations.py --annot-dir results/annotation

流程：
  1. 讀 results/annotation/{side}_標註_{A,B,C}.csv
  2. 重疊題（三人都標）→ 算兩兩加權 κ = 【人 vs 人】一致度
     這一步不能省：若人與人之間都對不起來，那 LLM vs 人的數字沒有意義。
  3. 重疊題取多數決（平手取中位數）、獨有題直接採用
  4. 從 _key/ 回填 LLM 分數，寫回 human_sample_{side}.csv
     → 之後跑 --stage agreement 就會算出【LLM vs 人】
"""
from __future__ import annotations
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rq2_analysis_skeleton import weighted_kappa  # noqa: E402

SIDES = {
    "say":   (["human_directness", "human_restriction"],
              {"human_directness": "llm__say_directness",
               "human_restriction": "llm__say_restriction"}),
    "think": (["human_think_content"],
              {"human_think_content": "llm__think_content"}),
    "frame": (["human_frame_official", "human_frame_rights"],
              {"human_frame_official": "llm__frame_official",
               "human_frame_rights": "llm__frame_rights"}),
}
PEOPLE = ["A", "B", "C"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot-dir", type=Path, default=Path("results/annotation"))
    ap.add_argument("--outdirs", nargs="*", type=Path,
                    default=[Path("results/analysis"), Path("results/analysis_openai")])
    a = ap.parse_args()

    for side, (axes, key_map) in SIDES.items():
        frames = {}
        for p in PEOPLE:
            f = a.annot_dir / f"{side}_標註_{p}.csv"
            if not f.exists():
                print(f"[跳過] {side}: 缺 {f.name}")
                break
            frames[p] = pd.read_csv(f).set_index("item_id")
        if len(frames) < len(PEOPLE):
            continue

        print(f"\n=== {side} ===")
        # --- 人 vs 人（只在重疊題上）---
        for ax in axes:
            for p, q in combinations(PEOPLE, 2):
                s1 = pd.to_numeric(frames[p].get(ax), errors="coerce")
                s2 = pd.to_numeric(frames[q].get(ax), errors="coerce")
                both = pd.concat([s1, s2], axis=1, join="inner").dropna()
                if len(both) < 5:
                    continue
                x, y = both.iloc[:, 0].values, both.iloc[:, 1].values
                print(f"  人vs人 {ax:<22} {p}-{q}  n={len(both):<3} "
                      f"完全一致 {(x == y).mean():.0%}  加權κ {weighted_kappa(x, y):.3f}")

        # --- 合併：重疊題多數決，其餘直接採用 ---
        merged = {}
        for ax in axes:
            cols = pd.concat({p: pd.to_numeric(frames[p].get(ax), errors="coerce")
                              for p in PEOPLE}, axis=1)
            n_rated = cols.notna().sum(axis=1)
            # 多數決：眾數；平手（3 人 3 種答案）退回中位數並四捨五入
            def _agg(r):
                v = r.dropna()
                if v.empty:
                    return np.nan
                m = v.mode()
                return float(m.iloc[0]) if len(m) == 1 else float(np.round(v.median()))
            merged[ax] = cols.apply(_agg, axis=1)
            tie = (cols.notna().sum(axis=1) == 3) & (cols.nunique(axis=1) == 3)
            if tie.any():
                print(f"  [note] {ax}: {int(tie.sum())} 題三人全異，已取中位數")
            print(f"  合併 {ax:<22} 已標 {int((n_rated > 0).sum())}/{len(cols)} 題")
        M = pd.DataFrame(merged)

        # --- 回填 LLM 分數，寫回各 outdir ---
        for od in a.outdirs:
            src = od / f"human_sample_{side}.csv"
            if not src.exists():
                continue
            base = pd.read_csv(src)
            base["item_id"] = [f"{side.upper()}-{i:03d}" for i in range(len(base))]
            for ax in axes:
                base[ax] = base["item_id"].map(M[ax])
            base.drop(columns=["item_id"]).to_csv(src, index=False)
            print(f"  → 已回填 {src}")

    print("\n接著跑：")
    for od in a.outdirs:
        print(f"  python3 pipeline/rq2_analysis_skeleton.py --stage agreement --outdir {od}")


if __name__ == "__main__":
    main()
