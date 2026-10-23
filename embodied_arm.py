"""
具身智能模拟: 双连杆机械臂够取 + 小脑逆模型 (Kawato 反馈误差学习)
====================================================================
设置:
  - 被控对象: 二连杆机械臂 (真实动力学: 惯性/科氏/重力)
  - 基线控制器: 脑干/脊髓式 PD 反馈 + 名义模型前馈, 但名义模型
    不知道末端负载 -> 前馈不准, 留下持续误差
  - 小脑模块: CerebellarNet 结构 (苔藓编码->颗粒稀疏层->浦肯野读出),
    学习逆模型前馈力矩; 教师信号 = 反馈控制器被迫输出的纠正力矩
    (feedback-error learning: 反馈力矩就是攀缘纤维误差)
  - 范式: 基线(无负载) -> 适应(末端加负载) -> 洗脱(撤负载),
    观察 适应 / 后遗效应 (aftereffect); 另设小脑冻结的损毁对照
运行: python3 embodied_arm.py (生成 PNG)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cerebellar_net import GranuleLayer, BasketLayer

G = 9.81
DT = 0.005
T_TRIAL = 1.4                      # 每试次 1.4s
N_STEP = int(T_TRIAL / DT)


# ----------------------------------------------------------------------
# 二连杆机械臂动力学 (标准 Spong 模型)
# ----------------------------------------------------------------------
class Arm:
    def __init__(self, m1=1.0, m2=0.8, l1=1.0, l2=0.8):
        self.m1, self.m2, self.l1, self.l2 = m1, m2, l1, l2

    def dynamics(self, q, qd, tau, payload=0.0):
        m1, m2, l1, l2 = self.m1, self.m2 + payload, self.l1, self.l2
        lc1, lc2 = l1 / 2, l2 / 2
        I1 = m1 * l1 ** 2 / 12; I2 = m2 * l2 ** 2 / 12
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        M = np.array([
            [m1 * lc1**2 + I1 + m2 * (l1**2 + lc2**2 + 2*l1*lc2*c2) + I2,
             m2 * (lc2**2 + l1*lc2*c2) + I2],
            [m2 * (lc2**2 + l1*lc2*c2) + I2, m2 * lc2**2 + I2]])
        h = -m2 * l1 * lc2 * s2
        C = np.array([[h * qd[1], h * (qd[0] + qd[1])],
                      [-h * qd[0], 0.0]])
        Gv = np.array([
            (m1 * lc1 + m2 * l1) * G * np.cos(q[0])
            + m2 * lc2 * G * np.cos(q[0] + q[1]),
            m2 * lc2 * G * np.cos(q[0] + q[1])])
        qdd = np.linalg.solve(M, tau - C @ qd - Gv)
        return qdd, Gv

    def fk(self, q):
        x1, y1 = self.l1 * np.sin(q[0]), -self.l1 * np.cos(q[0])
        x2 = x1 + self.l2 * np.sin(q[0] + q[1])
        y2 = y1 - self.l2 * np.cos(q[0] + q[1])
        return np.array([x2, y2])


def min_jerk(q0, q1, t, T=1.0):
    """最小急动轨迹 (0..T 内完成, 之后保持) 及其导数"""
    s = np.clip(t / T, 0, 1)
    h = 10 * s**3 - 15 * s**4 + 6 * s**5
    hd = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    hdd = (60 * s - 180 * s**2 + 120 * s**3) / T**2
    return q0 + (q1 - q0) * h, (q1 - q0) * hd, (q1 - q0) * hdd


# ----------------------------------------------------------------------
# 小脑逆模型: 输入 = 期望状态 (q_d, qd_d, qdd_d) 的苔藓编码
# 输出 = 前馈力矩修正; 用反馈力矩作为误差, 逐试次批量更新读出
# ----------------------------------------------------------------------
class CerebellarInverseModel:
    def __init__(self, traj_states, n_granule=1500, sparsity=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.n_g = n_granule
        # 苔藓感受野中心取自期望轨迹上的状态点 (+抖动), 6 维状态空间
        idx = rng.choice(len(traj_states), size=250, replace=True)
        self.centers = traj_states[idx] + rng.normal(0, 0.08,
                                                     (250, traj_states.shape[1]))
        self.width = 0.35
        self.granule = GranuleLayer(250, n_granule, sparsity=sparsity,
                                    seed=seed)
        self.basket = BasketLayer(0.5)
        self.W = np.zeros((n_granule, 2))

    def _pf(self, state):
        d2 = ((self.centers - state) ** 2).sum(-1)
        mossy = np.exp(-d2 / (2 * self.width ** 2))
        return self.basket.forward(self.granule.forward(mossy[None]))[0]

    def torque(self, state):
        return self._pf(state) @ self.W

    def learn_from_trial(self, states, tau_fb, lr=0.01):
        """反馈误差学习 (Kawato 1987): 逐时刻以反馈力矩为误差,
        ΔW = lr * mean_t( pf_t ⊗ tau_fb_t ) —— 小脑逐步接管反馈的工作"""
        PF = np.stack([self._pf(s) for s in states])
        self.W += lr * PF.T @ tau_fb / len(states)
        return PF


# ----------------------------------------------------------------------
# 单个试次: PD 反馈 (低增益, 脑干式) + 名义重力补偿 + 小脑前馈
# ----------------------------------------------------------------------
def run_trial(arm, cereb, q_start, q_target, payload, learn=True,
              lesion=False):
    q, qd = q_start.copy(), np.zeros(2)
    states, tau_fbs, hand = [], [], []
    Kp, Kd = np.diag([60.0, 48.0]), np.diag([12.0, 9.6])
    for i in range(N_STEP):
        t = i * DT
        qd_d, qv_d, qa_d = min_jerk(q_start, q_target, t)
        state = np.concatenate([qd_d, qv_d, qa_d])
        _, Gv_nominal = arm.dynamics(q, qd, np.zeros(2), payload=0.0)
        tau_ff = Gv_nominal                       # 名义模型只补重力(无负载)
        tau_cb = np.zeros(2) if lesion else np.clip(cereb.torque(state), -25, 25)
        tau_fb = Kp @ (qd_d - q) + Kd @ (qv_d - qd)
        tau = tau_ff + tau_cb + tau_fb
        tau = np.clip(tau, -80, 80)
        qdd, _ = arm.dynamics(q, qd, tau, payload=payload)
        qd = qd + qdd * DT
        q = q + qd * DT
        states.append(state); tau_fbs.append(tau_fb)
        hand.append(arm.fk(q))
    if learn and not lesion:
        cereb.learn_from_trial(np.array(states), np.array(tau_fbs))
    hand = np.array(hand)
    target_xy = arm.fk(q_target)
    end_err = float(np.linalg.norm(hand[-30:].mean(0) - target_xy))
    path_err = float(np.mean(np.linalg.norm(
        hand - np.array([arm.fk(min_jerk(q_start, q_target, i * DT)[0])
                         for i in range(N_STEP)]), axis=1)))
    return end_err, path_err, hand


def run_condition(lesion=False, seed=0):
    arm = Arm()
    q_start = np.array([0.5, 0.4])
    q_target = np.array([1.15, 0.75])
    # 期望轨迹状态采样 (给小脑苔藓中心用)
    traj = np.array([np.concatenate(min_jerk(q_start, q_target, i * DT))
                     for i in range(N_STEP)])
    cereb = CerebellarInverseModel(traj, seed=seed)
    phases = [("baseline", 0.0, 8), ("adapt (+0.8kg payload)", 0.8, 30),
              ("washout", 0.0, 12)]
    errs, hands, labels = [], {}, []
    trial = 0
    for name, payload, n in phases:
        for k in range(n):
            e, p, hand = run_trial(arm, cereb, q_start, q_target,
                                   payload, lesion=lesion)
            errs.append(e); labels.append(name)
            if (name == "baseline" and k == 0) or \
               (name.startswith("adapt") and k in (0, 29)) or \
               (name == "washout" and k == 0):
                hands[f"{name.split()[0]}#{trial}"] = hand
            trial += 1
    return np.array(errs), labels, hands, arm, q_target


if __name__ == "__main__":
    errs, labels, hands, arm, q_target = run_condition(lesion=False)
    errs_l, _, _, _, _ = run_condition(lesion=True)
    b_end = 8; a_end = 8 + 30
    print("具身实验: 机械臂够取 + 小脑逆模型 (反馈误差学习)")
    print(f"  基线终点误差 (首试次): {errs[0]:.3f} m")
    print(f"  加负载首试次误差:       {errs[b_end]:.3f} m (扰动冲击)")
    print(f"  适应末 (30 试次后):     {errs[a_end - 1]:.3f} m")
    print(f"  洗脱首试次 (后遗效应):  {errs[a_end]:.3f} m "
          f"(撤负载后朝反方向偏 = aftereffect)")
    print(f"  洗脱末:                 {errs[-1]:.3f} m")
    print(f"  损毁对照 适应末误差:    {errs_l[a_end - 1]:.3f} m "
          f"(小脑冻结 -> 误差消不掉)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(errs, "o-", ms=3, label="with cerebellum")
    axes[0].plot(errs_l, "s--", ms=3, label="cerebellar lesion (frozen)")
    for x in (b_end - 0.5, a_end - 0.5):
        axes[0].axvline(x, color="gray", ls=":")
    axes[0].set_xlabel("trial"); axes[0].set_ylabel("endpoint error (m)")
    axes[0].legend(); axes[0].set_title("Adaptation, aftereffect & lesion")
    tgt = arm.fk(q_target)
    for k, h in hands.items():
        axes[1].plot(h[:, 0], h[:, 1], label=k)
    axes[1].plot(*tgt, "r*", ms=16, label="target")
    axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
    axes[1].legend(fontsize=7); axes[1].set_title("Hand trajectories")
    axes[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig("exp4_arm_reaching.png", dpi=110)
    print("图已保存: exp4_arm_reaching.png")
