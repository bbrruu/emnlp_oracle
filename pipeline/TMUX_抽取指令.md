# RQ2 activation 抽取 — tmux 指令備忘

在實驗室伺服器(~/EMNLP/,與 rq2_extract_activations.py + rq2_stimuli_FINAL.csv 同層)照順序跑。

## ① 先確認 Qwen 是否已跑完
```bash
cd ~/EMNLP
ls -la activations/qwen/
# 有 activations_Qwen2.5-7B-Instruct.parquet = 已完成(只補 Gemma);沒有 = 兩個都跑
```

## ② 開 tmux
```bash
tmux new -s rq2
```

## ③ 在 tmux 裡跑(Qwen → Gemma 串好)
```bash
cd ~/EMNLP
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python3 rq2_extract_activations.py --pairs-csv rq2_stimuli_FINAL.csv --model Qwen/Qwen2.5-7B-Instruct --outdir activations/qwen --keep-all && CUDA_VISIBLE_DEVICES=0 python3 rq2_extract_activations.py --pairs-csv rq2_stimuli_FINAL.csv --model google/gemma-3-12b-it --outdir activations/gemma --keep-all
```

## ④ 離開 tmux(job 繼續跑)
按 `Ctrl-b` 放開，再按 `d`。→ 這時可關 VS Code / 關筆電。

## ⑤ 晚點回來看
```bash
tmux attach -t rq2        # 連回去看進度/完成沒
# 看完想再離開:再 Ctrl-b d;job 完成後可 exit 關掉 session
```

## ⑥ 驗證輸出
```bash
ls -la activations/qwen/ activations/gemma/
# 兩個 activations_*.parquet + 各印過「抽取完成:720 向量」= 完成
```

## 備註
- 這支腳本只寫進 activations/{qwen,gemma}/，不動任何其他檔案，安全。
- 沒有中途續跑：被中斷就整個重跑(約 2 分鐘/模型)。用 tmux 就不怕斷線。
- 模型已在 cache → HF_HUB_OFFLINE=1 離線讀,不連網、不需 token(Gemma 也不用)。
- GPU 資源:那台有 2×H100 80GB,綁一張(CUDA_VISIBLE_DEVICES=0)就夠。
```
