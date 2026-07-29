#!/usr/bin/env bash
# ===========================================================================
# run_judge_cli.sh — 用本機 Claude Code CLI 當 judge 跑完 RQ2 全部分析
#
#   不需要 API key：judge 走 `claude -p`，用的是 Claude Code 自己的登入。
#
# 用法：
#   bash pipeline/run_judge_cli.sh                 # 全量
#   SMOKE=1 bash pipeline/run_judge_cli.sh         # 只跑 say 前 8 筆試水溫
#   EFFORT=low bash pipeline/run_judge_cli.sh      # 換 effort
#   OUTDIR=results/analysis_x bash ...             # 換輸出目錄
#
# 成本（實測 2026-07-29）：
#   每次 `claude -p` 會帶約 22.5K token 的 Claude Code harness 前綴。
#   該前綴會被 prompt cache 跨行程重用，所以只有第一筆付寫入費：
#       第 1 筆 $0.14  →  之後每筆約 $0.008
#   四段 judge 合計約 1,394 筆 → 約 US$12、4 條並行約 25–40 分鐘。
#   腳本會先序列跑 1 筆暖機，不要把 WORKERS 調高後跳過暖機。
#
# ⚠ 方法學：這不是乾淨的裸模型呼叫，judge 是在 Claude Code 的 agent harness 裡評分
#   （那 22.5K 前綴關不掉）。前綴對每一筆都相同 → 是常數偏誤，不是隨主語/模型變動的
#   差別偏誤，所以不影響「中國 vs 德國」「Qwen vs Gemma」這類比較。
#   但論文方法章節要寫「透過 Claude Code CLI 呼叫 claude-opus-5」，
#   不能寫成「呼叫 Anthropic API」。要完全乾淨請改 --judge-backend anthropic。
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-results/analysis}"
MODEL="${MODEL:-claude-opus-5}"
EFFORT="${EFFORT:-medium}"
# cli 後端並行別開太高：實測 8 條會撞到速率上限，出現一整段連續 exit 1。
WORKERS="${WORKERS:-4}"
PY="${PY:-python3}"
S=pipeline/rq2_analysis_skeleton.py

command -v claude >/dev/null || { echo "[錯誤] 找不到 claude CLI"; exit 1; }
echo "judge = $MODEL (effort=$EFFORT, cli) → $OUTDIR"
claude --version

COMMON=(--judge-backend cli --judge-model "$MODEL" --judge-effort "$EFFORT"
        --workers "$WORKERS" --outdir "$OUTDIR")

RESP=(--responses  Qwen=results/responses/qwen/responses_Qwen2.5-7B-Instruct.jsonl
      --responses  Gemma=results/responses/gemma/responses_gemma-3-12b-it.jsonl)
# ★ 一定要餵 representatives（720，每向量一筆、含 cos），不要餵 av（3600，5 個 sample_k）
VERB=(--verbalize  Qwen=results/representatives/qwen_rep.parquet
      --verbalize  Gemma=results/representatives/gemma_rep.parquet
      --roundtrip  Qwen=results/roundtrip/qwen_roundtrip.csv
      --roundtrip  Gemma=results/roundtrip/gemma_roundtrip.csv)
ACTS=(--activations Qwen=results/activations/qwen/activations_Qwen2.5-7B-Instruct.parquet
      --activations Gemma=results/activations/gemma/activations_gemma-3-12b-it.parquet)
AV=(--av Qwen=results/verbalizations/qwen_av.parquet
    --av Gemma=results/verbalizations/gemma_av.parquet)

if [[ "${SMOKE:-0}" == "1" ]]; then
  # 試水溫：只評 8 筆，看 reason 欄合不合理再放全量。
  # 快取同名，所以這 8 筆在正式跑時會直接命中、不會重複付費。
  echo "=== SMOKE：say 側前 8 筆 ==="
  $PY - "$OUTDIR" "$MODEL" "$EFFORT" <<'PYEOF'
import json, subprocess, sys, importlib.util
from pathlib import Path
outdir, model, effort = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("m", "pipeline/rq2_analysis_skeleton.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
Path(outdir).mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in
        open("results/responses/qwen/responses_Qwen2.5-7B-Instruct.jsonl")][:8]
j = m.Judge(model=model, base_url="", api_key=None,
            cache_path=Path(outdir) / "cache_smoke.jsonl",
            backend="cli", workers=4, effort=effort)
items = [{"_key": f"smoke|{r['pair_id']}|{r['lang']}", "prompt": r["text"],
          "response": r["response"]} for r in rows]
out = j.run(items, lambda x: m.RUBRIC_SAY_PILOT
            .replace("@@PROMPT@@", str(x["prompt"]))
            .replace("@@RESPONSE@@", str(x["response"])[:1500]))
for r, s in zip(rows, out):
    print(f"  {r['subject']}/{r['lang']}  dir={s.get('directness')} "
          f"restr={s.get('restriction_justified')}  {str(s.get('reason'))[:60]}")
bad = [s for s in out if "_error" in s or "_parse_error" in s]
print(f"\n失敗 {len(bad)}/{len(out)}")
PYEOF
  echo; echo "看過上面的 reason 覺得合理，再把 SMOKE 拿掉跑全量。"
  exit 0
fi

echo "=== 1/5 geometry（不打 judge，3 秒）==="
$PY "$S" --stage geometry "${ACTS[@]}" --outdir "$OUTDIR"

echo "=== 2/5 judge-say（384 筆）==="
$PY "$S" --stage judge-say "${COMMON[@]}" "${RESP[@]}"

echo "=== 3/5 judge-think（384 筆）==="
$PY "$S" --stage judge-think "${COMMON[@]}" "${VERB[@]}"

echo "=== 4/5 judge-frame（336 筆）==="
$PY "$S" --stage judge-frame "${COMMON[@]}" "${VERB[@]}"

echo "=== 5/5 stability（290 筆）+ gap + 圖 ==="
$PY "$S" --stage stability "${COMMON[@]}" "${AV[@]}"
$PY "$S" --stage gap    --outdir "$OUTDIR"
$PY "$S" --stage figures --outdir "$OUTDIR"

echo
echo "完成 → $OUTDIR"
echo "下一步：填 $OUTDIR/human_sample_*.csv 的 human_* 欄，再跑"
echo "  $PY $S --stage agreement --outdir $OUTDIR"
