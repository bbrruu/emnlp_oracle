# 在國網(TWCC)驗證 tokenization

```bash
pip install -U "transformers>=4.50" tokenizers huggingface_hub pandas
export HF_HOME=/work/$USER/hf_cache
huggingface-cli login          # Gemma 為 gated，需先在 HF 網頁同意授權

python verify_tokenization.py \
    --models Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it \
    --pairs-csv rq2_pairs.csv \
    --min-leadin 8 \
    --outdir tokenizer_report
```

輸出 `tokenizer_report/`:
- `report_{model}.csv` 每句診斷(mention subtoken 切分、siteA/B 索引、leadin_tokens、span 解碼是否含 mention)
- `pairs_{model}.csv` 中英 ±20% 長度匹配診斷
- `summary.md` 跨模型摘要

**判準**:表徵題(S1/S0)`leadin_tokens ≥ 8`、句對長度差 ≤ ±20%;不過關者改寫。
壓制題(S2)看 siteB,leadin 不設限。
