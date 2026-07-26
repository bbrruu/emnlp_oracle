#!/usr/bin/env python3
#SBATCH --job-name=rq2_extract
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=rq2_extract_%j.log
# =============================================================================
# rq2_run_slurm.py — 用 sbatch 提交的「Python job 檔」(不用寫 bash)
#   提交：sbatch rq2_run_slurm.py
#   它會在 GPU 節點依序抽 Qwen 與 Gemma 的 activation。
#
# 需與 rq2_extract_activations.py、rq2_stimuli_FINAL.csv 放在「同一個資料夾」。
# 關電腦也會繼續跑(SLURM job 在節點上跑,與你的登入無關)。
# 進度看 log：tail -f rq2_extract_<jobid>.log
#
# ⚠ GPU 資源寫法各叢集不同：若 --gres=gpu:1 不吃，改成你們叢集的寫法
#    (例如 --gpus=1 或 --gres=gpu:<型號>:1)。分區可用 dev(2h)或 normal(2天)。
# =============================================================================
import os, sys, subprocess

# 模型已在 cache(RQ1 用過同兩個模型)→ 離線讀,不連網、不需 token,最安全。
# 若 `hf cache scan` 沒看到模型 → 先在 login node 下載，或把下一行註解掉(讓它連網下載)。
os.environ["HF_HUB_OFFLINE"] = "1"
# 若一定要連網抓 Gemma 且要用自己的 token，取消下一行並填入(或提交前先 export HF_TOKEN)：
# os.environ["HF_TOKEN"] = "hf_xxx"

HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "rq2_extract_activations.py")
PAIRS  = os.path.join(HERE, "rq2_stimuli_FINAL.csv")

# (model, 輸出資料夾)；層位用腳本內建預設(Qwen L20 / Gemma L32)
JOBS = [
    ("Qwen/Qwen2.5-7B-Instruct", os.path.join(HERE, "activations", "qwen")),
    ("google/gemma-3-12b-it",    os.path.join(HERE, "activations", "gemma")),
]

def main() -> None:
    for f in (SCRIPT, PAIRS):
        if not os.path.exists(f):
            sys.exit(f"[錯誤] 找不到 {f}；請把三個檔放同一資料夾。")
    for model, outdir in JOBS:
        print(f"\n===== 抽取 {model} =====", flush=True)
        r = subprocess.run(
            [sys.executable, SCRIPT,
             "--pairs-csv", PAIRS, "--model", model,
             "--outdir", outdir, "--keep-all"],
            cwd=HERE,
        )
        if r.returncode != 0:
            sys.exit(f"[錯誤] {model} 抽取失敗(exit {r.returncode})，中止。")
    print("\n全部抽取完成 ✅  輸出在 activations/qwen 與 activations/gemma", flush=True)

if __name__ == "__main__":
    main()
