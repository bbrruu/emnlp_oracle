#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rq2_generate_responses.py — RQ2 生成模型回答(think–say 的「說」那半,獨立版)
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
  * --conclusion-first:附加「先講結論」格式指示(依 lang 選中/英版),
    避免 Gemma 鋪多節論說文架子導致結論被截斷。指示逐格相同,不會造成不對稱。
  * 記錄 n_new_tokens / finish_reason:直接得到截斷 ground truth,
    不必再用標點啟發式猜。
  * --sample N:分層抽樣(依 subject×lang×frame 輪流取),供 pilot 用;
    --limit 只取 CSV 前 N 列,會全部落在同一格。

用法:
  python rq2_generate_responses.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model Qwen/Qwen2.5-7B-Instruct --outdir responses_v2/qwen \
      --conclusion-first --max-new-tokens 512 --keep-all
  # Gemma(gated:先 export HF_TOKEN 或於同帳號登入)
  python rq2_generate_responses.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model google/gemma-3-12b-it --outdir responses_v2/gemma \
      --conclusion-first --max-new-tokens 512 --keep-all

  # 先跑分層 pilot 確認截斷率(每格都要看,不能只看整體)
  python rq2_generate_responses.py --pairs-csv rq2_stimuli_FINAL.csv \
      --model google/gemma-3-12b-it --outdir responses_v2_pilot/gemma \
      --conclusion-first --max-new-tokens 512 --sample 24 --keep-all

  # 只驗流程不載模型(CPU,秒級):--dry-run
"""
from __future__ import annotations
import argparse, csv, json, os, random, sys, time
from pathlib import Path
from collections import defaultdict

META_COLS = [
    "pair_id", "sent_seq", "frame", "entity", "lang", "mention", "text",
    "subject", "subject_en", "concept_en", "concept_class",
    "sens_level", "stance_strength", "evidence_line", "part",
]

# 依 lang 選版本:英文指示接在中文題目後會引入跨語言污染。
# 兩版語意逐項對應(結論先行、理由句數上限、篇幅上限、禁用標題/項目/參考清單)。
FORMAT_INSTRUCTION = {
    "zh": ("請先用一到兩句話直接說出你的結論或立場,再用最多三句話說明理由。"
           "全文請控制在 150 字以內,以連續散文書寫,"
           "不要使用標題、項目符號、編號分節或參考資料清單。"),
    "en": ("First state your conclusion or position directly in one or two sentences, "
           "then give your reasons in at most three sentences. "
           "Keep the whole answer under 120 words, written as continuous prose. "
           "Do not use headings, bullet points, numbered sections, or a list of sources."),
}


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


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """依 (subject_en, lang, frame) 分格輪流取,直到湊滿 n 題。
    比 --limit 好:--limit 取 CSV 前 N 列,常整批落在同一 subject。"""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r.get("subject_en"), r.get("lang"), r.get("frame"))].append(r)
    rng = random.Random(seed)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells, key=lambda k: tuple(str(x) for x in k))
    out: list[dict] = []
    while len(out) < n and any(cells[k] for k in keys):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


def load_done(out_path: Path) -> set[str]:
    """讀既有 JSONL,回傳已完成的 sent_seq(用於續跑)。
    注意:只認 sent_seq,不認生成設定——換 budget 或換 prompt 格式時
    必須寫到新的 --outdir,否則會把舊回答當成已完成而全部略過。"""
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
    ap.add_argument("--conclusion-first", action="store_true",
                    help="附加結論先行的格式指示(依 lang 選版本)")
    ap.add_argument("--limit", type=int, default=None, help="只跑 CSV 前 N 題(除錯;會偏格)")
    ap.add_argument("--sample", type=int, default=None,
                    help="分層抽 N 題(pilot 用,優於 --limit)")
    ap.add_argument("--seed", type=int, default=0, help="--sample 的隨機種子")
    ap.add_argument("--dry-run", action="store_true", help="不載模型,只印會跑幾題")
    args = ap.parse_args()

    rows = load_pairs_csv(args.pairs_csv)
    if args.which != "all":
        rows = [r for r in rows if r.get("evidence_line") == args.which]
    if not args.keep_all:
        rows = [r for r in rows if row_passes_review(r, args.min_naturalness)]
    if args.sample:
        rows = stratified_sample(rows, args.sample, args.seed)
    elif args.limit:
        rows = rows[:args.limit]

    tag = args.model.split("/")[-1]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / f"responses_{tag}.jsonl"
    done = load_done(out_path)
    todo = [r for r in rows if r.get("sent_seq") not in done]
    prompt_format = "conclusion_first" if args.conclusion_first else "bare"
    print(f"===== 生成 {args.model}｜{args.which} 題 =====")
    print(f"  prompt_format={prompt_format}  max_new_tokens={args.max_new_tokens}")
    print(f"  共 {len(rows)} 題;已完成 {len(done)};本輪要跑 {len(todo)}（續跑）")
    if args.dry_run:
        print("  [dry-run] 不載模型。範例 prompt：")
        if todo:
            print("  " + build_prompt(todo[0], args.conclusion_first).replace("\n", "\n  "))
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

    def generate(prompt: str) -> tuple[str, int, str]:
        """回傳 (回答, 新生成 token 數, finish_reason)。
        HF generate 只有在始終沒吐 EOS 時才會用滿 max_new_tokens,
        因此 n_new >= 上限 即為截斷。"""
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
        new = out[0][inputs["input_ids"].shape[1]:]
        n_new = int(new.shape[0])
        reason = "length" if n_new >= args.max_new_tokens else "stop"
        return (tok.decode(new, skip_special_tokens=True).strip(), n_new, reason)

    t0 = time.time()
    n_trunc = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            resp, n_new, reason = generate(build_prompt(row, args.conclusion_first))
            n_trunc += (reason == "length")
            rec = {k: row.get(k, "") for k in META_COLS}
            rec.update(model=args.model, response=resp,
                       gen_greedy=(args.temperature == 0.0),
                       n_new_tokens=n_new, finish_reason=reason,
                       max_new_tokens=args.max_new_tokens,
                       prompt_format=prompt_format)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  截斷 {n_trunc}（{100*n_trunc/i:.0f}%）"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"完成 -> {out_path}（本輪 {len(todo)} 題,截斷 {n_trunc} 題）")


def build_prompt(row: dict, conclusion_first: bool) -> str:
    """題目本文 +（可選）結論先行指示。指示語言依 row['lang'] 選,
    未知語言退回英文版並不靜默——printed 一次以免無聲污染。"""
    text = row["text"]
    if not conclusion_first:
        return text
    lang = (row.get("lang") or "").strip()
    instr = FORMAT_INSTRUCTION.get(lang)
    if instr is None:
        print(f"[警告] lang={lang!r} 無對應指示,退回 en 版（sent_seq={row.get('sent_seq')}）")
        instr = FORMAT_INSTRUCTION["en"]
    return f"{text}\n\n{instr}"


if __name__ == "__main__":
    main()