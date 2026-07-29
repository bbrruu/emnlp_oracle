#!/usr/bin/env bash
# ===========================================================================
# run_judge_openai.sh — 用 OpenAI 相容端點（如 bazaarlink）當 judge
#
# 這是「第二 judge」路線，和 run_judge_cli.sh 互不衝突：
#   * 兩者輸出到【不同 outdir】，快取檔名也帶 backend+model，不會互相覆蓋
#   * gpt-4o-mini 那份才對得上 pilot 的基準數字（中國 dir 0.58 / restr 0.92）
#   * 兩份的一致度本身就是效度證據
#
# ── key 怎麼放（不要寫進 repo、不要留在 shell history）──────────────────
# 建議一：放在 repo 外的檔案，只有你讀得到
#     mkdir -p ~/.config/rq2 && touch ~/.config/rq2/judge.env
#     chmod 600 ~/.config/rq2/judge.env
#     # 用編輯器打開，填入這兩行（不要用 echo，會進 history）：
#     #   RQ2_JUDGE_API_KEY=你的key
#     #   RQ2_JUDGE_BASE_URL=https://<bazaarlink 的端點>/v1
#     bash pipeline/run_judge_openai.sh
#
# 建議二：當場輸入，不落地也不進 history（-s 不回顯）
#     read -rs -p "key: " RQ2_JUDGE_API_KEY && export RQ2_JUDGE_API_KEY
#     export RQ2_JUDGE_BASE_URL=https://<bazaarlink 的端點>/v1
#     bash pipeline/run_judge_openai.sh
#
# ⚠ 不要 `export RQ2_JUDGE_API_KEY=sk-...` 直接打在指令列 —— 那會寫進
#   ~/.zsh_history，而且 `ps` 看得到。
# ⚠ 本腳本不會印出 key，也不會把它寫進任何輸出檔或 log。
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${RQ2_ENV_FILE:-$HOME/.config/rq2/judge.env}"
if [[ -z "${RQ2_JUDGE_API_KEY:-}" && -f "$ENV_FILE" ]]; then
  # set -a 讓檔案裡的變數自動 export；用 set +x 確保不會被 trace 印出來
  set +x; set -a; . "$ENV_FILE"; set +a
fi

if [[ -z "${RQ2_JUDGE_API_KEY:-}" ]]; then
  echo "[錯誤] 沒有 RQ2_JUDGE_API_KEY。看本檔開頭的兩種放法。" >&2; exit 1
fi
if [[ -z "${RQ2_JUDGE_BASE_URL:-}" ]]; then
  echo "[錯誤] 沒有 RQ2_JUDGE_BASE_URL。bazaarlink 的端點要自己填，我不猜。" >&2
  echo "       OpenAI 官方是 https://api.openai.com/v1" >&2; exit 1
fi

OUTDIR="${OUTDIR:-results/analysis_openai}"
MODEL="${MODEL:-gpt-4o-mini}"
WORKERS="${WORKERS:-8}"
PY="${PY:-python3}"
S=pipeline/rq2_analysis_skeleton.py

$PY -c "import openai" 2>/dev/null || { echo "[錯誤] pip install openai"; exit 1; }

# 只印遮罩後的樣子，確認讀到了但不外洩
echo "judge = $MODEL @ $RQ2_JUDGE_BASE_URL → $OUTDIR"
echo "key   = ${RQ2_JUDGE_API_KEY:0:3}…${RQ2_JUDGE_API_KEY: -2} (len=${#RQ2_JUDGE_API_KEY})"

# key 只透過環境變數傳給 python（腳本會讀 RQ2_JUDGE_API_KEY），
# 不放在指令列參數 → 不會出現在 ps 或 log。
COMMON=(--judge-backend api --judge-model "$MODEL"
        --base-url "$RQ2_JUDGE_BASE_URL" --workers "$WORKERS" --outdir "$OUTDIR")

RESP=(--responses  Qwen=results/responses/qwen/responses_Qwen2.5-7B-Instruct.jsonl
      --responses  Gemma=results/responses/gemma/responses_gemma-3-12b-it.jsonl)
VERB=(--verbalize  Qwen=results/representatives/qwen_rep.parquet
      --verbalize  Gemma=results/representatives/gemma_rep.parquet
      --roundtrip  Qwen=results/roundtrip/qwen_roundtrip.csv
      --roundtrip  Gemma=results/roundtrip/gemma_roundtrip.csv)
ACTS=(--activations Qwen=results/activations/qwen/activations_Qwen2.5-7B-Instruct.parquet
      --activations Gemma=results/activations/gemma/activations_gemma-3-12b-it.parquet)
AV=(--av Qwen=results/verbalizations/qwen_av.parquet
    --av Gemma=results/verbalizations/gemma_av.parquet)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "=== SMOKE：先只跑 say，看數字對不對得上 pilot ==="
  $PY "$S" --stage judge-say "${COMMON[@]}" "${RESP[@]}"
  exit 0
fi

$PY "$S" --stage geometry   "${ACTS[@]}" --outdir "$OUTDIR"
$PY "$S" --stage judge-say  "${COMMON[@]}" "${RESP[@]}"
$PY "$S" --stage judge-think "${COMMON[@]}" "${VERB[@]}"
$PY "$S" --stage judge-frame "${COMMON[@]}" "${VERB[@]}"
$PY "$S" --stage stability  "${COMMON[@]}" "${AV[@]}"
$PY "$S" --stage gap        --outdir "$OUTDIR"
$PY "$S" --stage figures    --outdir "$OUTDIR"

echo
echo "完成 → ${OUTDIR}（Claude 那份在 results/analysis）"
echo "兩份 judged_say.csv 併起來算加權 κ，就是 judge 間信度。"
