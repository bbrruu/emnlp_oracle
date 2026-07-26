"""
RQ2 免費早期 de-risk:表徵幾何驗證(零 GPU,只需現有 activation parquet)

目的:在跑任何 NLA/GPU 之前,先確認
  (1) 殘差流在 NLA 目標層(Qwen L20 / Gemma L32)確實編碼了「敏感 vs 中性」的可分方向
  (2) 差分向量(Δ = mean_sens - mean_neut)pipeline 可運作
  (3) siteA(概念 token)vs siteB(句尾 token)的「框架訊號」差異可量測

資料:activations_Qwen2.5-7B-Instruct.parquet, activations_gemma-3-12b-it.parquet
      (用 RQ1 的「台灣」句子:固定 entity=Taiwan,以 POL 框架當 sensitive、其餘框架當 neutral)
依賴:fastparquet, numpy  (皆 CPU,無需 sklearn/GPU)
"""
import fastparquet, numpy as np
np.random.seed(0)
SENS_FRAMES = {"POL-INT", "POL-DOM"}   # RQ2 正式版改成 concept_class=='S' / sensitivity_level in {L1..L4}

def load(path):
    df = fastparquet.ParquetFile(path).to_pandas()
    df = df[df.entity == "Taiwan"].copy()                    # 固定實體、只變框架
    df["label"] = df.frame.isin(SENS_FRAMES).astype(int)     # 1=敏感 0=中性
    X = np.stack([np.asarray(v, dtype=np.float32) for v in df.activation_vector])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)  # L2 normalize(與 NLA 一致)
    return df.reset_index(drop=True), X

def auc(scores, y):                                          # Mann-Whitney AUC,免 sklearn
    y = np.asarray(y); order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def cv_delta_auc(X, y, k=5):                                 # 用 Δ 方向做 k-fold 探針
    y = np.asarray(y); idx = np.arange(len(y)); rng = np.random.RandomState(0); rng.shuffle(idx)
    folds = np.array_split(idx, k); aucs = []
    for i in range(k):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)   # 訓練集算 Δ
        if len(np.unique(y[te])) < 2: continue
        aucs.append(auc(X[te] @ d, y[te]))                          # 測試集投影
    return float(np.mean(aucs))

for f in ["activations_Qwen2.5-7B-Instruct.parquet", "activations_gemma-3-12b-it.parquet"]:
    df, X = load(f)
    print("=" * 58, "\n", f.split("_", 1)[1].replace(".parquet", ""))
    for site in ["A", "B"]:
        m = (df.site == site).values; Xs, ys = X[m], df.label.values[m]
        mu1, mu0 = Xs[ys == 1].mean(0), Xs[ys == 0].mean(0)
        cosmn = float(mu1 @ mu0 / ((np.linalg.norm(mu1) * np.linalg.norm(mu0)) + 1e-8))
        print(f"  site {site}  n_sens={int(ys.sum()):3d} n_neut={int((1-ys).sum()):3d} | "
              f"cos(mean_sens,mean_neut)={cosmn:.3f} | 5-fold Δ-probe AUC={cv_delta_auc(Xs, ys):.3f}")
