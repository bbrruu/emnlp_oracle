#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rq2_generate_responses.py — RQ2 生成模型回答(think–say 的「說」那半，獨立版)
==========================================================================
對壓制題(S2)用 instruct/chat 模式跑模型生成,存下回答,供 judge 評 directness /
政治性限制,並與 site~B 的 NLA 描述(「想」)比對出 think–say gap。

與 rq2_extract_activations.py 分工:
  * extract → activation(想的原料);verbalize → NLA 描述(想);
  * 本檔 → 實際生成回答(說)。

特色:
  * 套各模型 chat template(instruct 模式,壓制才會出現)。
  * 預設 greedy(do_sample=False),可重現。
  * JSONL 增量寫 + 續跑:中途斷線重跑會自動略過已完成的(對叢集不穩很重要)。
  * 預設只跑壓制題(evidence_line=suppression);表徵題是描述句、非問句,不生成。

用法:
  python rq2_generate_responses.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model Qwen/Qwen2.5-7B-Instruct --outdir responses/qwen --keep-all
  # Gemma(gated:先 export HF_TOKEN 或於同帳號登入)
  python rq2_generate_responses.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model google/gemma-3-12b-it --outdir responses/gemma --keep-all

  # 只驗流程不載模型(CPU,秒級):--dry-run
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from pathlib import Path

META_COLS = [
    "pair_id", "sent_seq", "frame", "entity", "lang", "mention", "text",
    "subject", "subject_en", "concept_en", "concept_class",
    "sens_level", "stance_strength", "evidence_line", "part",
]


def load_pairs_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if rows and "text" not in rows[0]:
        raise SystemExit(f"[錯誤] CSV 缺 text 欄（實際欄位：{list(rows[0])}）")
    return rows


def row_passes_review(row: dict, min_naturalness: float) -> bool:
    v = (row.get("naturalness") or "").strip()
    if not v:
        return True
    try:
        return float(v) >= min_naturalness
    except ValueError:
        return True


def load_done(out_path: Path) -> set[str]:
    """讀既有 JSONL,回傳已完成的 sent_seq(用於續跑)。"""
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sent_seq"])
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-csv", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--outdir", type=Path, default=Path("responses"))
    ap.add_argument("--which", choices=["suppression", "representation", "all"],
                    default="suppression", help="跑哪類題(預設只壓制題)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0=greedy(可重現);>0 才 sample")
    ap.add_argument("--min-naturalness", type=float, default=4.0)
    ap.add_argument("--keep-all", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 題(除錯)")
    ap.add_argument("--dry-run", action="store_true", help="不載模型,只印會跑幾題")
    args = ap.parse_args()

    rows = load_pairs_csv(args.pairs_csv)
    if args.which != "all":
        rows = [r for r in rows if r.get("evidence_line") == args.which]
    if not args.keep_all:
        rows = [r for r in rows if row_passes_review(r, args.min_naturalness)]
    if args.limit:
        rows = rows[:args.limit]

    tag = args.model.split("/")[-1]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"responses_{tag}.jsonl"
    done = load_done(out_path)
    todo = [r for r in rows if r.get("sent_seq") not in done]
    print(f"===== 生成 {args.model}｜{args.which} 題 =====")
    print(f"  共 {len(rows)} 題;已完成 {len(done)};本輪要跑 {len(todo)}（續跑）")
    if args.dry_run:
        print("  [dry-run] 不載模型。")
        return
    if not todo:
        print("  沒有新題要跑(可能全部已完成)。")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  載入 {args.model}（bf16, device_map=auto）…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    dev = next(model.parameters()).device

    def generate(prompt: str) -> str:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(dev)
        gen = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        if args.temperature and args.temperature > 0:
            gen.update(do_sample=True, temperature=args.temperature)
        else:
            gen.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**inputs, **gen)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            resp = generate(row["text"])
            rec = {k: row.get(k, "") for k in META_COLS}
            rec.update(model=args.model, response=resp,
                       gen_greedy=(args.temperature == 0.0))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"完成 -> {out_path}（本輪 {len(todo)} 題）")


if __name__ == "__main__":
    main()
