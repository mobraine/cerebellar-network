"""
小脑学习实验模拟 —— 刺激 (stimulus) 与反馈 (feedback) 范式
====================================================================
实验 1: 延迟型瞬目条件反射 (delay eyeblink conditioning)
        小脑最经典的学习范式 (Thompson / Mauk)。
        CS = 声音 (苔藓纤维输入, 持续到 US 出现)
        US = 气吹眼睛 (攀缘纤维误差信号)
        CR = 在 US 到来之前就眨眼 (浦肯野->DCN->运动输出)
        测: 习得曲线、CR 峰值时刻 (是否对准 US)、消退、节省效应

实验 2: 前庭眼反射 (VOR) 增益适应
        头动 (前庭苔藓输入) 时眼睛应反向等量转动; 戴放大/缩小镜后
        视网膜滑动 (retinal slip) 作为攀缘纤维误差, 驱动增益上调/下调。
        测: 增益学习曲线; 攀缘纤维"损毁"后是否还能学

实验 3: 结构消融 (ablation)
        在函数逼近任务上逐个关掉小脑结构元件 (扩张/稀疏/侧抑制),
        看每个元件贡献多少 —— 相当于给架构做"反馈测试"。

时间模型 (实验 1/2 共用): 颗粒细胞带时间调谐 (spectral timing,
Mauk & Buonomano): 每个颗粒细胞在 CS 内的某个时刻最活跃, 群体构成
时间基 (temporal basis); 攀缘纤维误差只更新在该时刻活跃的突触,
于是学到的反应天然带时间定位 (timed response)。

运行: python3 cerebellar_experiments.py  (生成 PNG 图到同目录)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cerebellar_net import CerebellarNet, MLP

rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------------
# 时序型小脑模型: 颗粒细胞带时间调谐基
# ----------------------------------------------------------------------
class TemporalCerebellum:
    def __init__(self, n_granule=500, t_max=60, seed=0):
        rng = np.random.default_rng(seed)
        self.n_g, self.t_max = n_granule, t_max
        # 每个颗粒细胞的时间调谐中心与宽度 (谱时间模型)
        self.mu = rng.uniform(0, t_max, n_granule)
        self.sigma = rng.uniform(2.0, 6.0, n_granule)
        self.w = np.zeros(n_granule)      # 平行纤维->浦肯野权重

    def granule(self, t, cs_on):
        """时刻 t 的颗粒细胞活动: CS 门控 x 时间调谐"""
        if not cs_on:
            return np.zeros(self.n_g)
        return np.exp(-((t - self.mu) ** 2) / (2 * self.sigma ** 2))

    def response(self, t, cs_on):
        """浦肯野->DCN 读出的条件反应强度 (0~1 量级)"""
        return float(self.granule(t, cs_on) @ self.w)

    def trial_trace(self, cs_start, cs_end):
        g = np.stack([self.granule(t, cs_start <= t < cs_end)
                      for t in range(self.t_max)])
        return g, g @ self.w              # (时间 x 细胞) 活动 与 CR 波形


# ----------------------------------------------------------------------
# 实验 1: 延迟型瞬目条件反射
#   每 trial: CS 在 t=5 开始, US 在 t=50 出现 (ISI=450ms, 步长 10ms)
#   攀缘纤维误差 = US - CR(t_US): 预测不足 -> LTD 增强权重;
#   消退期 US=0 -> 误差为负 -> 权重回落 (Mauk 模型中的 LTP 反向过程)
# ----------------------------------------------------------------------
def eyeblink(seed=0, n_acq=120, n_ext=40, n_reacq=40, lr=0.02):
    T, CS0, US_T = 60, 5, 50
    net = TemporalCerebellum(t_max=T, seed=seed)

    def run_phase(n_trials, us_present):
        cr_amp, cr_peak_t = [], []
        for _ in range(n_trials):
            g, cr = net.trial_trace(CS0, T)
            # CR 幅度: US 之前 100ms 窗内的峰值 (不含 US 时刻本身)
            window = cr[US_T - 10:US_T]
            cr_amp.append(window.max())
            cr_peak_t.append(US_T - 10 + int(window.argmax()))
            us = 1.0 if us_present else 0.0
            err = us - cr[US_T]                    # 攀缘纤维误差
            net.w += lr * err * g[US_T]            # 只更新 US 时刻活跃的突触
            net.w = np.maximum(net.w, 0)           # 权重非负 (简化)
        return np.array(cr_amp), np.array(cr_peak_t)

    amp1, peak1 = run_phase(n_acq, True)           # 习得
    _, cr_after = net.trial_trace(CS0, T)
    amp2, _ = run_phase(n_ext, False)              # 消退
    amp3, _ = run_phase(n_reacq, True)             # 再习得 (节省效应)

    def blocks(a, k=10):
        return [a[i:i + k].mean() for i in range(0, len(a), k)]

    print("实验 1: 瞬目条件反射 (delay eyeblink conditioning)")
    print(f"  习得期 CR 幅度 (每10试次): "
          f"{[round(float(x), 2) for x in blocks(amp1)]}")
    print(f"  习得末 CR 峰值时刻: t={peak1[-10:].mean():.0f} 步 "
          f"(US 在 t={US_T}) -> 反应对准在 US 之前片刻")
    print(f"  消退期 CR 幅度: {[round(float(x), 2) for x in blocks(amp2)]}")
    print(f"  再习得前 10 试次 CR: {amp3[:10].mean():.2f} "
          f"(对比初次习得前 10 试次 {amp1[:10].mean():.2f}) "
          f"-> 节省效应: {'有' if amp3[:10].mean() > amp1[:10].mean() + 0.05 else '不明显'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(blocks(amp1), "o-", label="acquisition (CS+US)")
    axes[0].plot(range(len(blocks(amp1)), len(blocks(amp1)) + len(blocks(amp2))),
                 blocks(amp2), "s-", label="extinction (CS only)")
    axes[0].set_xlabel("trial block (x10)"); axes[0].set_ylabel("CR amplitude")
    axes[0].legend(); axes[0].set_title("Eyeblink conditioning: learning curve")
    t_axis = np.arange(T) * 10
    axes[1].plot(t_axis, cr_after, label="CR waveform after training")
    axes[1].axvline(US_T * 10, color="r", ls="--", label="US onset")
    axes[1].axvline(CS0 * 10, color="g", ls="--", label="CS onset")
    axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("CR amplitude")
    axes[1].legend(); axes[1].set_title("Timed CR peaks just before the US")
    fig.tight_layout(); fig.savefig("exp1_eyeblink.png", dpi=110)
    return amp1, amp2, amp3


# ----------------------------------------------------------------------
# 实验 2: VOR 增益适应
#   头速 v(t)=sin; 眼速 = gain * v(t) 的反向补偿; 目标增益 g* 由"眼镜"
#   决定: 放大镜 g*=1.5 (增益上调), 缩小镜 g*=0.5 (下调)。
#   攀缘纤维误差 (视网膜滑动) = (g* - g) * v(t), 沿周期积分更新权重。
#   颗粒基: 相位调谐细胞; 权重读出即时增益 g(t) = 1 + w . basis(t)/v(t)
#   简化实现: 用标量增益 g 与其误差规则展示学习动力学, 并设损毁对照。
# ----------------------------------------------------------------------
def vor(seed=0, cycles=60, lr=0.05):
    rng = np.random.default_rng(seed)
    print("实验 2: VOR 增益适应 (vestibular-ocular reflex)")

    def run(g_star, climbing_ok=True):
        g, hist = 1.0, []
        for c in range(cycles):
            slip = g_star - g                       # 视网膜滑动 (误差)
            hist.append(g)
            err = slip if climbing_ok else 0.0      # 损毁: 切断攀缘纤维
            g += lr * err + rng.normal(0, 0.004)    # 噪声模拟生理抖动
        return np.array(hist)

    up = run(1.5); down = run(0.5); les = run(1.5, climbing_ok=False)
    print(f"  增益上调 (目标1.5): 末值 {up[-1]:.2f}")
    print(f"  增益下调 (目标0.5): 末值 {down[-1]:.2f}")
    print(f"  攀缘纤维损毁 + 目标1.5: 末值 {les[-1]:.2f} "
          f"-> 无误差信号则不适应 (与动物损毁实验一致)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(up, label="gain-up (x1.5 spectacles)")
    ax.plot(down, label="gain-down (x0.5 spectacles)")
    ax.plot(les, "--", label="climbing-fiber lesion")
    ax.set_xlabel("training cycle"); ax.set_ylabel("VOR gain")
    ax.legend(); ax.set_title("VOR adaptation & lesion control")
    fig.tight_layout(); fig.savefig("exp2_vor.png", dpi=110)
    return up, down, les


# ----------------------------------------------------------------------
# 实验 3: 结构消融 —— 逐个关掉小脑元件, 看各自的贡献
# ----------------------------------------------------------------------
def ablation():
    print("实验 3: 结构消融 (函数逼近任务, 岭读出, 测试 RMSE 越低越好)")
    rng = np.random.default_rng(7)
    f = lambda X: (np.sin(X[:, 0] * 2) * np.cos(X[:, 1] * 2)
                   + 0.5 * X[:, 0] * X[:, 1]).reshape(-1, 1)
    Xtr = rng.uniform(-1.5, 1.5, (600, 2)); Ytr = f(Xtr)
    Xte = rng.uniform(-1.5, 1.5, (300, 2)); Yte = f(Xte)
    rmse = lambda P: float(np.sqrt(np.mean((P - Yte) ** 2)))

    configs = [
        ("完整小脑结构", dict()),
        ("无稀疏化 (sparsity=1.0)", dict(sparsity=1.0)),
        ("弱稀疏 (sparsity=0.30)", dict(sparsity=0.30)),
        ("颗粒细胞减半 (1500)", dict(n_granule=1500)),
        ("颗粒细胞 x3 (9000)", dict(n_granule=9000)),
        ("无苔藓编码 (直连颗粒层)", dict(use_mossy=False)),
    ]
    results = {}
    for name, kw in configs:
        net = CerebellarNet(2, 1, **kw).fit_ridge(Xtr, Ytr, lam=0.5)
        results[name] = rmse(net.predict(Xte))
        print(f"  {name:28s} RMSE {results[name]:.4f}")
    return results


if __name__ == "__main__":
    eyeblink()
    print()
    vor()
    print()
    ablation()
    print("\n图已保存: exp1_eyeblink.png, exp2_vor.png")
