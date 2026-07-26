#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rq2_extract_activations.py — RQ2 residual-stream activation 抽取（獨立版）
==========================================================================
RQ2 專用、自成一體（不 import RQ1 的檔）。輸入定案語料 rq2_stimuli_FINAL.csv，
每句 × 2 抽取位置（siteA=概念末 subtoken、siteB=句末非 special token）→ 向量，
寫成 NLA 推論所需 parquet（含 activation_vector float32 [N, d_model] + 完整
RQ2 metadata）。

與設計對齊：
  * 抽 Qwen2.5-7B-Instruct L20 / Gemma-3-12B-IT L32（層取自 --nla-meta，缺省用內建）。
  * bf16、output_hidden_states；hidden_states[L] = 第 L 個 block 後之殘差流。
  * siteA = mention 最後一個 subtoken；siteB = 句末（最後一個非 special）token。
  * tokenizer 呼叫 add_special_tokens=True + offset_mapping，與 verify_tokenization 一致
    （mention 定位邏輯已內嵌，與該腳本同源）。
  * 預設「不套 chat template」（NLA 以純文字 activation 訓練）→ 表徵題(S0/S1) 正確。
    壓制題(S2)若要抓「即將回答/迴避」的內部狀態，可加 --chat-template
    （⚠ 對 NLA 為 OOD，忠實度會掉，需嚴格 gate；建議另存一批、與老師確認方向）。

用法：
  # 1) dry-run 先驗 site 索引（CPU、秒級，不載模型）
  python rq2_extract_activations.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model Qwen/Qwen2.5-7B-Instruct --dry-run

  # 2) 正式抽（GPU, bf16）
  python rq2_extract_activations.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model Qwen/Qwen2.5-7B-Instruct \
      --nla-meta /path/to/nla-qwen2.5-7b-L20-av/nla_meta.yaml \
      --outdir activations/qwen --keep-all

  # Gemma（gated：先 huggingface-cli login 並於網頁同意授權）
  python rq2_extract_activations.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model google/gemma-3-12b-it \
      --nla-meta /path/to/nla-gemma3-12b-L32-av/nla_meta.yaml \
      --outdir activations/gemma --keep-all
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

# ── 要保留進 parquet 的 metadata（含 RQ2 設計欄位，後續分析靠這些 group）──
META_COLS = [
    "sent_id", "pair_id", "sent_seq", "frame", "entity", "lang",
    "mention_script", "cell_type", "mention", "text",
    "subject", "subject_en", "concept_en", "concept_class",
    "sens_level", "stance_strength", "evidence_line", "part",
]

# 缺 --nla-meta 且無 --layer 時的退回層位
DEFAULT_LAYER = {
    "Qwen/Qwen2.5-7B-Instruct": 20,
    "google/gemma-3-12b-it": 32,
    "unsloth/gemma-3-12b-it": 32,
}


# ======================================================================
# 純邏輯層（與 torch 無關）—— mention 定位與 verify_tokenization 同源（內嵌）
# ======================================================================
def find_mention_char_spans(text: str, mention: str) -> list[tuple[int, int]]:
    """回傳 mention 在 text 中所有出現位置之字元區間 [start, end)。"""
    spans, start = [], 0
    while True:
        i = text.find(mention, start)
        if i < 0:
            break
        spans.append((i, i + len(mention)))
        start = i + 1
    return spans


def locate_token_span(offsets, char_span):
    """以區間重疊（half-open）找出覆蓋 mention 字元區間的 token span。找不到 → (-1,-1)。"""
    cs, ce = char_span
    hit = [i for i, (ts, te) in enumerate(offsets) if ts < ce and te > cs and ts != te]
    if not hit:
        return -1, -1
    return hit[0], hit[-1] + 1


@dataclass
class SitePlan:
    sent_id: str
    lang: str
    mention: str
    n_tokens: int
    site_a_idx: int          # 概念末 subtoken（含 special 之絕對索引）
    site_b_idx: int          # 句末非 special token
    mention_n_subtokens: int
    warnings: str = ""


def prep_text(tok, raw_text: str, chat_template: bool) -> str:
    """回傳實際要 tokenize 的字串。預設純文字；--chat-template 時套 user turn。"""
    if not chat_template:
        return raw_text
    return tok.apply_chat_template(
        [{"role": "user", "content": raw_text}],
        tokenize=False, add_generation_prompt=True,
    )


def plan_sites(tok, text_to_tok: str, mention: str, sent_id: str, lang: str) -> SitePlan:
    enc = tok(text_to_tok, return_offsets_mapping=True, add_special_tokens=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    special = set(getattr(tok, "all_special_ids", []) or [])
    warns: list[str] = []

    spans = find_mention_char_spans(text_to_tok, mention)
    if len(spans) != 1:
        warns.append(f"mention 出現 {len(spans)} 次（規範要求恰為 1）")
    char_span = spans[0] if spans else (0, 0)
    t0, t1 = locate_token_span(offsets, char_span)
    if t0 < 0:
        warns.append("offset 對齊失敗：找不到覆蓋 mention 的 token span")

    non_special = [i for i, x in enumerate(ids) if x not in special]
    site_b = non_special[-1] if non_special else -1
    return SitePlan(
        sent_id=sent_id, lang=lang, mention=mention,
        n_tokens=len(non_special),
        site_a_idx=(t1 - 1) if t0 >= 0 else -1,
        site_b_idx=site_b,
        mention_n_subtokens=max(t1 - t0, 0),
        warnings="; ".join(warns),
    )


def assign_sent_id(row: dict, idx: int) -> str:
    """唯一句 id。優先用 sent_id/sent_seq；否則 pair_id|lang(+script) + 序號。"""
    if row.get("sent_id"):
        return row["sent_id"].strip()
    seq = (row.get("sent_seq") or "").strip()
    parts = [row.get(k, "").strip() for k in ("pair_id", "lang", "mention_script")]
    key = "|".join(p for p in parts if p)
    if seq:
        return f"{key}#{seq}" if key else f"seq{seq}"
    return f"{key}#{idx:04d}" if key else f"row{idx:04d}"


# ======================================================================
# I/O
# ======================================================================
def load_pairs_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for k in ("text", "mention"):
        if rows and k not in rows[0]:
            raise SystemExit(f"[錯誤] CSV 缺必要欄位 {k!r}（實際欄位：{list(rows[0])}）")
    return rows


def row_passes_review(row: dict, min_naturalness: float) -> bool:
    """naturalness < 門檻視為需改寫；缺分數（空值）不擋。"""
    v = (row.get("naturalness") or "").strip()
    if not v:
        return True
    try:
        return float(v) >= min_naturalness
    except ValueError:
        return True


def resolve_layer(model: str, nla_meta: Path | None, layer_arg: int | None) -> int:
    if layer_arg is not None:
        return layer_arg
    if nla_meta and nla_meta.exists():
        import yaml
        meta = yaml.safe_load(nla_meta.read_text(encoding="utf-8")) or {}
        for path in (("extraction", "layer"), ("layer",), ("model", "layer")):
            node = meta
            for k in path:
                node = node.get(k) if isinstance(node, dict) else None
            if isinstance(node, int):
                print(f"  層位取自 nla_meta.yaml：L{node}")
                return node
        print("  [警告] nla_meta.yaml 未見層位鍵，退回預設。")
    if model in DEFAULT_LAYER:
        return DEFAULT_LAYER[model]
    raise SystemExit(f"[錯誤] 無法決定層位：請給 --layer 或含層位的 --nla-meta（model={model}）")


def write_plan_csv(path: Path, plans: list[SitePlan]) -> None:
    if not plans:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(plans[0]).keys()))
        w.writeheader()
        w.writerows(asdict(p) for p in plans)


def write_parquet(path: Path, records: list[dict], d_model: int) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not records:
        raise SystemExit("[錯誤] 無任何向量可寫出。")
    meta_cols = [k for k in records[0] if k != "activation_vector"]
    arrays = {c: pa.array([r[c] for r in records]) for c in meta_cols}
    mat = np.stack([r["activation_vector"] for r in records]).astype(np.float32)
    assert mat.shape[1] == d_model, (mat.shape, d_model)
    flat = pa.array(mat.reshape(-1), type=pa.float32())
    arrays["activation_vector"] = pa.FixedSizeListArray.from_arrays(flat, d_model)
    pq.write_table(pa.table(arrays), path)


# ======================================================================
# 主抽取流程
# ======================================================================
def run_extract(rows, model, layer, outdir, dry_run, chat_template):
    import numpy as np
    from transformers import AutoTokenizer
    tag = model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(model)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("[錯誤] 需 fast tokenizer 才有 offset_mapping。")
    if chat_template:
        print("  [chat-template] 已開啟：activations 對 NLA 為 OOD，僅建議用於 S2/think-say，"
              "並嚴格 gate 忠實度。")

    # (1) site 規劃
    plans, keep = [], []
    for idx, row in enumerate(rows):
        sid = assign_sent_id(row, idx)
        ttext = prep_text(tok, row["text"], chat_template)
        p = plan_sites(tok, ttext, row["mention"], sid, row.get("lang", ""))
        plans.append(p)
        keep.append(row)
        if p.warnings:
            print(f"  [警告] {sid}: {p.warnings}")
    outdir.mkdir(parents=True, exist_ok=True)
    write_plan_csv(outdir / f"site_plan_{tag}.csv", plans)
    if len({p.sent_id for p in plans}) != len(plans):
        print("  [警告] sent_id 不唯一，請確認 CSV 的 pair_id/lang/sent_seq。")
    print(f"  site 規劃 -> {outdir}/site_plan_{tag}.csv（{len(plans)} 句）")
    if dry_run:
        print("  [dry-run] 未載入模型、未抽向量。")
        return

    # (2) 載模型抽向量
    import torch
    from transformers import AutoModelForCausalLM
    print(f"  載入 {model}（bf16, device_map=auto）…")
    net = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    net.eval()
    dev = next(net.parameters()).device

    records, n_bad = [], 0
    with torch.no_grad():
        for row, p in zip(keep, plans):
            ttext = prep_text(tok, row["text"], chat_template)
            enc = tok(ttext, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            hs = net(**enc).hidden_states[layer][0].float().cpu().numpy()   # [seq, d]
            for site, idx in (("A", p.site_a_idx), ("B", p.site_b_idx)):
                if idx < 0 or idx >= hs.shape[0]:
                    n_bad += 1
                    continue
                vec = hs[idx].astype(np.float32)
                rec = {k: (p.sent_id if k == "sent_id" else row.get(k, "")) for k in META_COLS}
                rec.update(
                    vector_id=f"{p.sent_id}#site{site}",
                    site=site, site_idx=int(idx), layer=int(layer), model=model,
                    n_tokens=int(p.n_tokens), mention_n_subtokens=int(p.mention_n_subtokens),
                    chat_template=bool(chat_template),
                    vec_norm=float(np.linalg.norm(vec)),
                    activation_vector=vec,
                )
                records.append(rec)

    cfg = net.config
    hidden = (getattr(cfg, "hidden_size", None)
              or getattr(getattr(cfg, "text_config", None), "hidden_size", None))
    if hidden is None:
        raise AttributeError(f"no hidden_size on {type(cfg).__name__}")
    d_model = int(hidden)
    suffix = "_chat" if chat_template else ""
    out = outdir / f"activations_{tag}{suffix}.parquet"
    write_parquet(out, records, d_model)
    print(f"  抽取完成：{len(records)} 向量（跳過 {n_bad}） -> {out}  d_model={d_model}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-csv", type=Path, required=True, help="rq2_stimuli_FINAL.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--nla-meta", type=Path, default=None, help="AV checkpoint 的 nla_meta.yaml（讀層位）")
    ap.add_argument("--layer", type=int, default=None, help="明確指定抽取層（覆蓋 meta）")
    ap.add_argument("--outdir", type=Path, default=Path("activations"))
    ap.add_argument("--min-naturalness", type=float, default=4.0)
    ap.add_argument("--keep-all", action="store_true", help="不套自然度門檻，抽全部")
    ap.add_argument("--chat-template", action="store_true",
                    help="套 chat template 抽（僅 S2/think-say 用；對 NLA 為 OOD）")
    ap.add_argument("--dry-run", action="store_true", help="只算 site 索引、不載模型")
    args = ap.parse_args()

    rows = load_pairs_csv(args.pairs_csv)
    if not args.keep_all:
        before = len(rows)
        rows = [r for r in rows if row_passes_review(r, args.min_naturalness)]
        if len(rows) != before:
            print(f"  自然度門檻：{before} → {len(rows)} 句（排除 {before - len(rows)}）")
    layer = resolve_layer(args.model, args.nla_meta, args.layer)
    print(f"===== 抽取 {args.model} @ L{layer}"
          f"{'（chat-template）' if args.chat_template else ''} =====")
    run_extract(rows, args.model, layer, args.outdir, args.dry_run, args.chat_template)


if __name__ == "__main__":
    main()
