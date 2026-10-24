"""
规模实验: 颗粒细胞数量从 1e3 拉到 1e6, 性能在哪儿饱和?
====================================================================
上一轮的结论是"小脑 = 规模 x 稀疏 x 单层学习", 这里实测规模那一项:
  - 颗粒层换成稀疏实现 (每细胞仍只接 4 根苔藓纤维, CSR 稀疏矩阵),
    否则 1e6 细胞的稠密权重矩阵要 1.6GB, 存不下 —— 这本身就是
    "稀疏连接是规模的前提"的实证。
  - 读出仍是单层 (岭回归, 用对偶形式在样本空间求解, 600x600,
    与特征维数无关) —— 学的只有读出, 扩张层不学, 和小脑一致。
任务:
  A) 函数逼近 (与消融实验同一题): 测试 RMSE vs 颗粒数
  B) 模式分离 (加难版: 20 类、强重叠): 类均值相关性 & 分类准确率
运行: python3 granule_scaling.py (约几分钟, 生成 exp6 图)
"""

import time
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cerebellar_net import MossyLayer

SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
SPARSITY = 0.04
FAN_IN = 4


class SparseGranule:
    """稀疏颗粒层: 每列 (颗粒细胞) 恰好 FAN_IN 个非零权重 (CSR)"""
    def __init__(self, n_mossy, n_granule, seed=0):
        rng = np.random.default_rng(seed)
        cols = np.repeat(np.arange(n_granule), FAN_IN)
        rows = np.concatenate([rng.choice(n_mossy, FAN_IN, replace=False)
                               for _ in range(n_granule)]) \
            if n_granule <= 20000 else \
            rng.integers(0, n_mossy, n_granule * FAN_IN)
        vals = rng.uniform(-1, 1, len(cols)).astype(np.float32) \
            / np.sqrt(FAN_IN)
        self.W = sp.csr_matrix((vals, (rows, cols)),
                               shape=(n_mossy, n_granule))
        self.n_g = n_granule
        self.k = max(1, int(SPARSITY * n_granule))

    def encode(self, mossy):
        """苔藓活动 (n x n_mossy) -> 稀疏平行纤维编码 (CSR)"""
        outs = []
        for s in range(0, len(mossy), 32):
            X = mossy[s:s + 32].astype(np.float32)
            drive = np.asarray(X @ self.W)          # (chunk x n_g)
            np.maximum(drive, 0, out=drive)         # ReLU
            # k-WTA: 每行只保留最大的 k 个
            idx = np.argpartition(drive, -self.k, axis=1)[:, -self.k:]
            mask = np.zeros_like(drive, dtype=bool)
            np.put_along_axis(mask, idx, True, axis=1)
            outs.append(sp.csr_matrix(np.where(mask, drive, 0)))
        return sp.vstack(outs).tocsr()


def ridge_dual(Phi_tr, Y, Phi_te, lam=1.0, center=False):
    """对偶形式岭回归: 解在样本空间, 维度与特征数无关。
    center=True 时做 Gram 中心化 —— 等价于原网络 BasketLayer 的
    共模抑制 (减去群体均值), 稀疏表示下在核层面实现, 不破坏稀疏性"""
    K = (Phi_tr @ Phi_tr.T).toarray()
    Kte = (Phi_te @ Phi_tr.T).toarray()
    if center:
        Kte = Kte - Kte.mean(1, keepdims=True) - K.mean(0, keepdims=True) \
            + K.mean()
        K = K - K.mean(1, keepdims=True) - K.mean(0, keepdims=True) + K.mean()
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Y)
    return Kte @ alpha


def class_mean_corr(Phi, y, n_cls):
    """PF 空间中类均值向量的平均两两 |相关| (越低分离越好)"""
    means = []
    Phi = Phi.tocsr()
    grand = np.asarray(Phi.mean(axis=0)).ravel()   # 共模 (篮状细胞去除的成分)
    for c in range(n_cls):
        m = np.asarray(Phi[y == c].mean(axis=0)).ravel() - grand
        means.append(m / (np.linalg.norm(m) + 1e-9))
    M = np.stack(means)
    C = np.abs(M @ M.T)
    iu = np.triu_indices(n_cls, 1)
    return float(C[iu].mean())


def main():
    rng = np.random.default_rng(7)
    # ---- 任务 A 数据 (与消融实验同一生成器) ----
    f = lambda X: (np.sin(X[:, 0] * 2) * np.cos(X[:, 1] * 2)
                   + 0.5 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    Xtr = rng.uniform(-1.5, 1.5, (600, 2)); Ytr = f(Xtr)
    Xte = rng.uniform(-1.5, 1.5, (300, 2)); Yte = f(Xte)
    mossyA = MossyLayer(200, seed=1).set_centers(-1.5, 1.5, 2)
    Mtr_A, Mte_A = mossyA.forward(Xtr), mossyA.forward(Xte)

    # ---- 任务 B 数据: 20 类, 6 维, 强重叠 ----
    rngB = np.random.default_rng(3)
    centers = rngB.normal(0, 1, (20, 6))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 1.1
    def gen(n_per):
        Xs, ys = [], []
        for c in range(20):
            Xs.append(centers[c] + rngB.normal(0, 0.38, (n_per, 6)))
            ys += [c] * n_per
        return np.vstack(Xs), np.array(ys)
    XtrB, ytrB = gen(40); XteB, yteB = gen(20)
    mossyB = MossyLayer(200, seed=2).set_centers(-2.5, 2.5, 6)
    Mtr_B, Mte_B = mossyB.forward(XtrB), mossyB.forward(XteB)
    Yoh = np.eye(20)[ytrB]
    in_corr = class_mean_corr(sp.csr_matrix(XtrB), ytrB, 20)
    print(f"输入空间类均值相关: {in_corr:.3f} (任务 B 基准)")

    rows = []
    for n_g in SIZES:
        t0 = time.time()
        g = SparseGranule(200, n_g, seed=0)
        # 任务 A
        Ptr, Pte = g.encode(Mtr_A), g.encode(Mte_A)
        pred = ridge_dual(Ptr, Ytr, Pte, lam=0.5)
        rmse = float(np.sqrt(np.mean((pred - Yte) ** 2)))
        # 任务 B
        Qtr, Qte = g.encode(Mtr_B), g.encode(Mte_B)
        corr = class_mean_corr(Qtr, ytrB, 20)
        predB = ridge_dual(Qtr, Yoh, Qte, lam=1.0, center=True)
        acc = float((predB.argmax(1) == yteB).mean())
        dt = time.time() - t0
        rows.append((n_g, rmse, corr, acc, dt))
        print(f"n_granule={n_g:>8,}: 函数逼近 RMSE {rmse:.4f} | "
              f"类均值相关 {corr:.3f} | 分类 {acc:.3f} | {dt:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ns = [r[0] for r in rows]
    axes[0].plot(ns, [r[1] for r in rows], "o-")
    axes[0].set_xscale("log"); axes[0].set_xlabel("granule cells")
    axes[0].set_ylabel("test RMSE")
    axes[0].set_title("Function approximation vs scale")
    ax2 = axes[1]
    ax2.plot(ns, [r[2] for r in rows], "o-", color="tab:red",
             label="class-mean |corr| (lower=better)")
    ax2.plot(ns, [r[3] for r in rows], "s-", color="tab:green",
             label="classification accuracy")
    ax2.axhline(in_corr, color="tab:red", ls=":", label="input-space corr")
    ax2.set_xscale("log"); ax2.set_xlabel("granule cells")
    ax2.legend(); ax2.set_title("Pattern separation vs scale (20 classes)")
    fig.tight_layout(); fig.savefig("exp6_granule_scaling.png", dpi=110)
    print("图已保存: exp6_granule_scaling.png")


if __name__ == "__main__":
    main()
