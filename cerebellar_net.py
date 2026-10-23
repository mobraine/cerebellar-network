"""
CerebellarNet —— 按小脑皮层神经元结构搭建的神经网络 (NumPy 实现)

生物结构 -> 代码模块
====================================================================
脑桥核团等预处理   MossyLayer: 苔藓纤维携带的是经预处理的信号, 这里用
                   高斯感受野编码 (每个苔藓纤维元对输入空间一小块区域响应)
颗粒细胞           GranuleLayer:
 (Granule cells)     - 数量远多于输入 (expansion recoding, 人脑约 500 亿)
                     - 每个颗粒细胞只接收约 4 条苔藓纤维 (稀疏随机连接)
                     - 高尔基细胞 (Golgi) 反馈抑制 -> k-WTA 稀疏化,
                       任一时刻只有约 1-5% 颗粒细胞活跃
平行纤维           颗粒细胞轴突 = GranuleLayer 的输出
篮状/星状细胞      BasketLayer: 前馈侧抑制 (减去群体平均活动, 锐化表征)
浦肯野细胞         PurkinjeLayer: 皮层唯一输出, 抑制性 (GABA);
 (Purkinje cells)    平行纤维突触可塑 —— 与攀缘纤维共激活时发生 LTD
攀缘纤维           ClimbingLayer 的角色由误差信号承担: 每个输出单元的
 (Climbing fibers)   误差 1:1 门控对应浦肯野细胞的突触更新 (下橄榄核教师信号)
小脑深部核团       DCNReadout 并入 PurkinjeLayer: DCN 有紧张性基线活动,
 (Deep nuclei)       被浦肯野输出持续抑制, 最终输出 = 基线 - 抑制 (去抑制)
====================================================================
学习规则 (Marr-Albus-Ito):
  fit_local —— 攀缘纤维门控的局部 LTD/LTP 规则, 无反向传播:
      err_j * pf_i 共激活 -> 该突触 LTD; 另加弱 LTP 回复防单向漂移
  fit_ridge —— 对照: 固定颗粒层, 读出用岭回归闭式解
运行: python3 cerebellar_net.py
"""

import numpy as np


# ----------------------------------------------------------------------
# 0. 苔藓纤维层: 高斯感受野编码
# ----------------------------------------------------------------------
class MossyLayer:
    def __init__(self, n_mossy=300, width=0.8, seed=2):
        self.n_mossy, self.width = n_mossy, width
        self.rng = np.random.default_rng(seed)
        self.centers = None

    def set_centers(self, lo, hi, n_in):
        self.centers = self.rng.uniform(lo, hi, (self.n_mossy, n_in))
        return self

    def forward(self, x):
        d2 = ((x[:, None, :] - self.centers[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.width ** 2))


# ----------------------------------------------------------------------
# 1. 颗粒细胞层: 扩张 + 稀疏连接 + 高尔基式 k-WTA 抑制
# ----------------------------------------------------------------------
class GranuleLayer:
    def __init__(self, n_in, n_granule=3000, inputs_per_cell=4,
                 sparsity=0.04, seed=0):
        rng = np.random.default_rng(seed)
        self.n_g = n_granule
        self.k = max(1, int(sparsity * n_granule))
        self.W = np.zeros((n_granule, n_in))
        for j in range(n_granule):
            idx = rng.choice(n_in, size=min(inputs_per_cell, n_in),
                             replace=False)
            # 兴奋/抑制混合权重 (颗粒细胞本身只兴奋, 符号混合是对
            # 上游通路净效应的简化; 也可全正, 见 README 说明)
            self.W[j, idx] = rng.choice([-1.0, 1.0], size=len(idx)) \
                * rng.uniform(0.5, 1.5, size=len(idx))

    def forward(self, x):
        drive = np.maximum(x @ self.W.T, 0.0)
        if self.k < self.n_g:
            idx = np.argpartition(-drive, self.k, axis=1)[:, :self.k]
            out = np.zeros_like(drive)
            np.put_along_axis(out, idx,
                              np.take_along_axis(drive, idx, axis=1), axis=1)
            return out
        return drive


# ----------------------------------------------------------------------
# 2. 篮状/星状细胞: 前馈侧抑制
# ----------------------------------------------------------------------
class BasketLayer:
    def __init__(self, strength=0.5):
        self.strength = strength

    def forward(self, pf):
        return pf - self.strength * pf.mean(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# 3. 浦肯野细胞层 (+ DCN 去抑制读出) 与攀缘纤维 LTD/LTP 局部学习
# ----------------------------------------------------------------------
class PurkinjeLayer:
    def __init__(self, n_pf, n_out, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.05, size=(n_pf, n_out))
        self.b = np.zeros(n_out)          # DCN 紧张性基线

    def forward(self, pf):
        return self.b - pf @ self.W       # 去抑制读出

    def ltd_step(self, pf, error, lr=3e-3, lr_ltp=1e-5):
        """攀缘纤维误差门控的局部更新 (批量平均):
        y = b - pf@W, error = target - y;
        error>0 需减小 pf@W -> 活跃突触 LTD: W -= lr * pf^T err / N"""
        grad = pf.T @ error / pf.shape[0]
        self.W -= lr * grad
        self.b += lr * error.mean(axis=0) * 10
        self.W -= lr_ltp * self.W         # 弱 LTP 回复, 防单向漂移


# ----------------------------------------------------------------------
# 组装: CerebellarNet
# ----------------------------------------------------------------------
class CerebellarNet:
    def __init__(self, n_in, n_out, x_range=(-1.5, 1.5), n_mossy=300,
                 n_granule=3000, sparsity=0.04, seed=0, use_mossy=True):
        # use_mossy: 低维连续输入用高斯感受野苔藓编码; 高维输入时苔藓
        # 纤维本身已是混合表征, 颗粒层可直接接收 (use_mossy=False)
        self.use_mossy = use_mossy
        if use_mossy:
            self.mossy = MossyLayer(n_mossy, seed=seed + 2)
            self.mossy.set_centers(x_range[0], x_range[1], n_in)
            g_in = n_mossy
        else:
            g_in = n_in
        self.granule = GranuleLayer(g_in, n_granule, sparsity=sparsity,
                                    seed=seed)
        self.basket = BasketLayer(0.5)
        self.purkinje = PurkinjeLayer(n_granule, n_out, seed=seed + 1)

    def _pf(self, x):
        if self.use_mossy:
            x = self.mossy.forward(x)
        return self.basket.forward(self.granule.forward(x))

    def predict(self, x):
        return self.purkinje.forward(self._pf(x))

    def fit_local(self, X, Y, epochs=400, lr=2e-3, verbose=True):
        pf = self._pf(X)
        for ep in range(epochs):
            err = Y - self.purkinje.forward(pf)   # 攀缘纤维误差信号
            self.purkinje.ltd_step(pf, err, lr=lr)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  train RMSE "
                      f"{np.sqrt(np.mean(err ** 2)):.4f}")
        return self

    def fit_ridge(self, X, Y, lam=0.5):
        pf = self._pf(X)
        A = np.concatenate([-pf, np.ones((pf.shape[0], 1))], axis=1)
        theta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
        self.purkinje.W, self.purkinje.b = theta[:-1], theta[-1]
        return self


# ----------------------------------------------------------------------
# 对照基线 1: 原始输入上的线性读出 (无扩张编码)
# ----------------------------------------------------------------------
def linear_readout(Xtr, Ytr, Xte, lam=1.0):
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    th = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    return np.concatenate([Xte, np.ones((len(Xte), 1))], axis=1) @ th


# ----------------------------------------------------------------------
# 对照基线 2: 普通 MLP (反向传播, 一层 tanh 隐层)
# ----------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, (n_in, n_hidden)) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 1, (n_hidden, n_out)) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_out)

    def predict(self, X):
        return np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2

    def fit(self, X, Y, epochs=3000, lr=0.03):
        n = len(X)
        for _ in range(epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            err = h @ self.W2 + self.b2 - Y
            dh = (err @ self.W2.T) * (1 - h ** 2)
            self.W2 -= lr * h.T @ err / n * 2
            self.b2 -= lr * err.mean(0) * 2
            self.W1 -= lr * X.T @ dh / n * 2
            self.b1 -= lr * dh.mean(0) * 2
        return self


# ----------------------------------------------------------------------
# 演示 1: 模式分离 —— 重叠输入经颗粒层扩张后, 表征重叠大幅下降
# ----------------------------------------------------------------------
def demo_pattern_separation():
    print("=" * 64)
    print("演示 1: 模式分离 (pattern separation)")
    rng = np.random.default_rng(42)
    n_in, n_cls, per = 30, 5, 60
    shared = rng.normal(0, 1, n_in)                      # 五类共享的大底
    deltas = 0.35 * rng.normal(0, 1, (n_cls, n_in))      # 类别差异很小
    protos = shared + deltas
    X = np.vstack([protos[c] + 0.25 * rng.normal(0, 1, (per, n_in))
                   for c in range(n_cls)])
    y = np.repeat(np.arange(n_cls), per)
    Y = np.eye(n_cls)[y]
    tr = np.arange(len(X)) % 5 != 0
    Xtr, Xte, Ytr, Yte = X[tr], X[~tr], Y[tr], Y[~tr]

    def mean_corr(reps):
        means = np.stack([reps[y == c].mean(0) for c in range(n_cls)])
        cs = [np.corrcoef(means[i], means[j])[0, 1]
              for i in range(n_cls) for j in range(i + 1, n_cls)]
        return float(np.mean(cs))

    net = CerebellarNet(n_in, n_cls, n_granule=3000, sparsity=0.04,
                        use_mossy=False)
    pf_all = net._pf(X)
    net.fit_ridge(Xtr, Ytr, lam=1.0)
    acc_c = (net.predict(Xte).argmax(1) == Yte.argmax(1)).mean()
    print(f"  类别均值表征相关性 — 输入空间:   {mean_corr(X):+.3f}")
    print(f"  类别均值表征相关性 — 平行纤维层: {mean_corr(pf_all):+.3f}"
          "  (扩张+稀疏化降低了表征重叠)")
    print(f"  测试准确率 — 小脑网络 (岭读出): {acc_c:.3f}")


# ----------------------------------------------------------------------
# 演示 2: 非线性函数逼近 —— 局部 LTD 规则 vs 岭读出 vs MLP
# ----------------------------------------------------------------------
def demo_function_approx():
    print("=" * 64)
    print("演示 2: 函数逼近 (攀缘纤维 LTD 局部学习, 无反向传播)")
    rng = np.random.default_rng(7)
    f = lambda X: (np.sin(X[:, 0] * 2) * np.cos(X[:, 1] * 2)
                   + 0.5 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    Xtr = rng.uniform(-1.5, 1.5, (600, 2)); Ytr = f(Xtr)
    Xte = rng.uniform(-1.5, 1.5, (300, 2)); Yte = f(Xte)
    rmse = lambda P: float(np.sqrt(np.mean((P - Yte) ** 2)))

    net = CerebellarNet(2, 1)
    net.fit_local(Xtr, Ytr, epochs=401, lr=2e-3)
    r_local = rmse(net.predict(Xte))

    net2 = CerebellarNet(2, 1)
    net2.fit_ridge(Xtr, Ytr, lam=0.5)
    r_ridge = rmse(net2.predict(Xte))

    mlp = MLP(2, 32, 1).fit(Xtr, Ytr)
    print(f"  测试 RMSE — 小脑(局部LTD): {r_local:.4f} | "
          f"小脑(岭读出): {r_ridge:.4f} | MLP(反向传播): {rmse(mlp.predict(Xte)):.4f}")


if __name__ == "__main__":
    demo_pattern_separation()
    demo_function_approx()
    print("=" * 64)
    print("要点: 小脑的算力来自结构 —— 高维扩张 + 极稀疏编码让重叠模式")
    print("      变得可分; 浦肯野突触的学习由误差信号 (攀缘纤维) 局部")
    print("      门控, 不需要全局反向传播。局部规则收敛较慢, 与生物学习")
    print("      需要大量重复试次的特点一致。")
