#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_annotations.py - merge the three annotators, report human-vs-human
agreement, and backfill the scores into the format
rq2_analysis_skeleton.py --stage agreement expects.

Usage:
    python3 code/merge_annotations.py
    python3 code/merge_annotations.py --annot-dir annotation

Steps:
  1. Read annotation/human/{side}_annotator{A,B,C}.csv
  2. On the overlap items (scored by all three), compute pairwise weighted
     kappa: the HUMAN-VS-HUMAN agreement. This step is not optional - if the
     annotators do not agree with each other, an LLM-vs-human number is
     meaningless. The means of these pairwise kappas are the human column of
     the paper's Table 3.
  3. Merge: majority vote on overlap items (median if all three differ),
     single-rater items taken as given.
  4. Backfill LLM scores from annotation/_key/ into each outdir's
     human_sample_{side}.csv, so that --stage agreement can then compute the
     LLM-VS-HUMAN agreement. Those templates are produced by the judge stages;
     if you have not re-run them, step 4 is skipped and steps 1-3 still report.
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
    ap.add_argument("--annot-dir", type=Path, default=Path("annotation"))
    ap.add_argument("--outdirs", nargs="*", type=Path,
                    default=[Path("results/analysis_r2"),
                             Path("results/analysis_r2_openai")])
    a = ap.parse_args()

    for side, (axes, key_map) in SIDES.items():
        frames = {}
        for p in PEOPLE:
            f = a.annot_dir / "human" / f"{side}_annotator{p}.csv"
            if not f.exists():
                print(f"[skip] {side}: missing {f.name}")
                break
            frames[p] = pd.read_csv(f).set_index("item_id")
        if len(frames) < len(PEOPLE):
            continue

        print(f"\n=== {side} ===")
        # --- human vs human, on the overlap items only ---
        for ax in axes:
            for p, q in combinations(PEOPLE, 2):
                s1 = pd.to_numeric(frames[p].get(ax), errors="coerce")
                s2 = pd.to_numeric(frames[q].get(ax), errors="coerce")
                both = pd.concat([s1, s2], axis=1, join="inner").dropna()
                if len(both) < 5:
                    continue
                x, y = both.iloc[:, 0].values, both.iloc[:, 1].values
                print(f"  human-human {ax:<22} {p}-{q}  n={len(both):<3} "
                      f"exact {(x == y).mean():.0%}  weighted kappa {weighted_kappa(x, y):.3f}")

        # --- merge: majority vote on overlap items, otherwise take as given ---
        merged = {}
        for ax in axes:
            cols = pd.concat({p: pd.to_numeric(frames[p].get(ax), errors="coerce")
                              for p in PEOPLE}, axis=1)
            n_rated = cols.notna().sum(axis=1)
            # majority = mode; if all three differ, fall back to the rounded median
            def _agg(r):
                v = r.dropna()
                if v.empty:
                    return np.nan
                m = v.mode()
                return float(m.iloc[0]) if len(m) == 1 else float(np.round(v.median()))
            merged[ax] = cols.apply(_agg, axis=1)
            tie = (cols.notna().sum(axis=1) == 3) & (cols.nunique(axis=1) == 3)
            if tie.any():
                print(f"  [note] {ax}: {int(tie.sum())} items where all three differ; median used")
            print(f"  merged {ax:<22} scored {int((n_rated > 0).sum())}/{len(cols)} items")
        M = pd.DataFrame(merged)

        # --- truncation control (say only): score the SAYT-* batch separately ---
        if side == "say" and any(str(i).startswith("SAYT-") for i in M.index):
            key = pd.read_csv(a.annot_dir / "_key" / "say_llm_key.csv").set_index("item_id")
            t = M[[str(i).startswith("SAYT-") for i in M.index]]
            print(f"  -- truncation control, {len(t)} items: LLM vs human "
                  "(does the judge underestimate truncated answers?)")
            for ax, lc in [("human_directness", "llm__say_directness"),
                           ("human_restriction", "llm__say_restriction")]:
                if ax not in t.columns or lc not in key.columns:
                    continue
                d = pd.concat([t[ax], key[lc]], axis=1, join="inner").apply(
                    pd.to_numeric, errors="coerce").dropna()
                if len(d) < 3:
                    continue
                h, l = d[ax].values, d[lc].values
                print(f"     {ax:<20} human={h.mean():.2f}  LLM={l.mean():.2f}  "
                      f"diff={h.mean() - l.mean():+.2f}  weighted kappa {weighted_kappa(h, l):.3f}"
                      + ("   -> humans score higher than the LLM, supporting "
                         "'the judge underestimates truncated answers'"
                         if h.mean() - l.mean() > 0.3 else ""))

        # --- backfill LLM scores into each outdir ---
        # The new say batch (SAY-*) corresponds to analysis_r2*; the old batch
        # (SAYT-*) is not backfilled, since it only serves the comparison above
        # and does not enter the main --stage agreement flow.
        for od in a.outdirs:
            src = od / f"human_sample_{side}.csv"
            if not src.exists():
                continue
            base = pd.read_csv(src)
            base["item_id"] = [f"{side.upper()}-{i:03d}" for i in range(len(base))]
            for ax in axes:
                base[ax] = base["item_id"].map(M[ax])
            n = int(base[axes[0]].notna().sum())
            base.drop(columns=["item_id"]).to_csv(src, index=False)
            print(f"  -> backfilled {src} ({n}/{len(base)} items annotated)")

    print("\nNext, for LLM-vs-human agreement:")
    for od in a.outdirs:
        print(f"  python3 code/rq2_analysis_skeleton.py --stage agreement --outdir {od}")


if __name__ == "__main__":
    main()
