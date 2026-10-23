"""
具身实验 2: 人形机器人站立平衡 + 小脑预期性姿势调整 (APA)
====================================================================
平台: MuJoCo 平面 (矢状面) 人形 —— 双腿 髋/膝/踝 x2, 躯干/头/臂。
全 3D Gym 人形在朴素 PD 下会直接倒地 (已实测), 平面人形是双足
平衡研究的标准简化 (与 Walker2d/Hopper 同一思路)。

范式 (仿人类推扰实验):
  每试次 6s, t=2.0s 时躯干受 150N x 0.15s 前推。
  基线控制器 (关节 PD + COM 踝策略) 在 150N 能恢复但晃动明显,
  190N 必然摔倒 —— 适应后迁移测试看小脑能否抬高这个阈值。
  小脑模块 (苔藓编码->颗粒稀疏层->读出) 输出 6 关节前馈力矩,
  学习规则用"预测式"反馈误差学习: 当前状态配对 +120ms 后的
  反馈力矩 (补偿感觉运动延迟, Smith 预测器思想), 并按 |FB|
  重要性加权。直接模仿当前反馈力矩会造成双重纠正 (实测反而
  变差), 资格迹版本则因高维苔藓激活塌陷而学不动 —— 都试过。
阶段: 基线(无学习) -> 适应 40 试次 -> 捕获试次(撤推力, 测后遗)
     对照: 小脑冻结的损毁组走同一流程。
运行: .venv/bin/python humanoid_balance.py
"""

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cerebellar_net import GranuleLayer, BasketLayer

PUSH_F, PUSH_T, PUSH_DUR = 150.0, 2.0, 0.15
TRIAL_T = 6.0
CTRL_EVERY = 5                      # 小脑 100Hz, 物理 500Hz


class HumanoidBalance:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path("humanoid2d.xml")
        self.d = mujoco.MjData(self.m)
        self.gear = self.m.actuator_gear[:, 0]
        self.Kp = np.full(6, 500.0); self.Kd = np.full(6, 40.0)
        self.Kpc, self.Kdc = 3500.0, 250.0

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.z0 = self.d.body(1).xpos[2]
        self.prev_com = None; self.com_v = 0.0

    def state(self, t):
        d = self.d
        q, qd = d.qpos, d.qvel
        com = d.subtree_com[1]
        feet_x = 0.5 * (d.body(4).xpos[0] + d.body(7).xpos[0]) + 0.05
        bump1 = np.exp(-((t - PUSH_T) / 0.25) ** 2)   # 时间基 (谱时间)
        bump2 = np.exp(-((t - PUSH_T) / 0.60) ** 2)
        return np.concatenate([q[3:], qd[3:], [q[2], qd[2],
                                com[0] - feet_x, self.com_v,
                                t, bump1, bump2]])

    def baseline_torque(self):
        d = self.d
        q, qd = d.qpos, d.qvel
        tau = -self.Kp * q[3:] - self.Kd * qd[3:]
        com = d.subtree_com[1]
        if self.prev_com is not None:
            self.com_v = (com[0] - self.prev_com) / self.m.opt.timestep
        self.prev_com = com[0]
        feet_x = 0.5 * (d.body(4).xpos[0] + d.body(7).xpos[0]) + 0.05
        tau_a = self.Kpc * (com[0] - feet_x) + self.Kdc * self.com_v
        tau[2] += tau_a / 2; tau[5] += tau_a / 2
        tau[0] += tau_a * 0.25; tau[3] += tau_a * 0.25
        return tau

    @property
    def fallen(self):
        return (self.d.body(1).xpos[2] < self.z0 - 0.35
                or abs(self.d.qpos[2]) > 0.7)


class CerebellarPosture:
    """小脑预期模块: 状态 -> 6 关节前馈力矩; FEL + 资格迹学习"""
    def __init__(self, sample_states, n_granule=1500, seed=0):
        rng = np.random.default_rng(seed)
        self.mu = sample_states.mean(0)
        self.sd = sample_states.std(0) + 1e-6
        idx = rng.choice(len(sample_states), size=250, replace=True)
        Z = (sample_states - self.mu) / self.sd
        self.centers = Z[idx] + rng.normal(0, 0.15, (250, Z.shape[1]))
        self.width = 0.9
        self.granule = GranuleLayer(250, n_granule, sparsity=0.05, seed=seed)
        self.basket = BasketLayer(0.5)
        self.W = np.zeros((n_granule, 6))
        self.buf_pf = []
        self.buf_fb = []

    def _pf(self, state):
        z = (state - self.mu) / self.sd
        d2 = ((self.centers - z) ** 2).sum(-1)
        mossy = np.exp(-d2 / (2 * self.width ** 2))
        return self.basket.forward(self.granule.forward(mossy[None]))[0]

    def torque(self, state):
        return np.clip(self._pf(state) @ self.W, -50, 50)

    def accumulate(self, state, tau_fb, dt_ctrl, tau_e=0.4):
        """记录本试次的 (pf, tau_fb) 序列, 供 apply 做预测式配对"""
        self.buf_pf.append(self._pf(state))
        self.buf_fb.append(np.asarray(tau_fb, dtype=float))

    def apply(self, n_steps, lr=0.6, shift=12):
        """预测式反馈误差学习: 用当前状态预测 +120ms 后的反馈力矩
        (shift=12 个控制步 x 10ms)。小脑由此学会在误差出现之前
        发出纠正力矩 —— 补偿感觉运动延迟 (Smith 预测器思想),
        而不是亦步亦趋地模仿当前反馈 (那会造成双重纠正)。"""
        if len(self.buf_pf) > shift:
            PF = np.stack(self.buf_pf)
            FB = np.stack(self.buf_fb)
            # 按 |FB| 重要性加权: 集中学推力响应段, 不被姿势噪声稀释
            wgt = np.abs(FB[shift:]).sum(-1)
            wgt = wgt / (wgt.sum() + 1e-9)
            self.W += lr * (PF[:-shift] * wgt[:, None]).T @ FB[shift:]
            self.W = np.clip(self.W, -20.0, 20.0)
        self.buf_pf.clear(); self.buf_fb.clear()


def run_trial(env, cereb=None, push=True, learn=False):
    env.reset()
    m, d = env.m, env.d
    dt = m.opt.timestep
    n = int(TRIAL_T / dt)
    max_pitch = 0.0; max_com = 0.0; fell = False
    pre_lean = []                      # 推前 1.6-2.0s 的 COM 偏移 (预期)
    n_ctrl = 0
    for i in range(n):
        t = i * dt
        tau_fb = env.baseline_torque()
        tau_cb = np.zeros(6)
        if cereb is not None and i % CTRL_EVERY == 0:
            s = env.state(t)
            tau_cb = cereb.torque(s)
            if learn and 1.7 <= t <= 3.2:
                cereb.accumulate(s, tau_fb, dt * CTRL_EVERY)
                n_ctrl += 1
        if push and PUSH_T <= t <= PUSH_T + PUSH_DUR:
            d.xfrc_applied[1] = [PUSH_F, 0, 0, 0, 0, 0]
        else:
            d.xfrc_applied[1] = 0
        d.ctrl[:] = np.clip((tau_fb + tau_cb) / env.gear, -1, 1)
        mujoco.mj_step(m, d)
        if push and t > PUSH_T:
            max_pitch = max(max_pitch, abs(d.qpos[2]))
            com = d.subtree_com[1]
            feet_x = 0.5 * (d.body(4).xpos[0] + d.body(7).xpos[0]) + 0.05
            max_com = max(max_com, abs(com[0] - feet_x))
        if 1.6 <= t < PUSH_T:
            com = d.subtree_com[1]
            feet_x = 0.5 * (d.body(4).xpos[0] + d.body(7).xpos[0]) + 0.05
            pre_lean.append(com[0] - feet_x)
        if env.fallen:
            fell = True
            break
    if cereb is not None and learn:
        cereb.apply(n_ctrl)
    return dict(fell=fell, max_pitch=float(max_pitch),
                max_com=float(max_com),
                pre_lean=float(np.mean(pre_lean)) if pre_lean else 0.0)


def run_protocol(learn=True, seed=0, n_adapt=40):
    env = HumanoidBalance()
    # 1) 基线采样: 无小脑跑几试次, 收集状态分布给苔藓中心
    samples = []
    base = []
    env.reset()
    for tr in range(4):
        env.reset()
        for i in range(int(TRIAL_T / env.m.opt.timestep)):
            t = i * env.m.opt.timestep
            tau = env.baseline_torque()
            if tr >= 2 and i % 25 == 0:
                samples.append(env.state(t))
            if PUSH_T <= t <= PUSH_T + PUSH_DUR:
                env.d.xfrc_applied[1] = [PUSH_F, 0, 0, 0, 0, 0]
            else:
                env.d.xfrc_applied[1] = 0
            env.d.ctrl[:] = np.clip(tau / env.gear, -1, 1)
            mujoco.mj_step(env.m, env.d)
            if env.fallen:
                break
    cereb = CerebellarPosture(np.array(samples), seed=seed)
    for _ in range(6):
        base.append(run_trial(env, cereb=None, push=True))
    adapt = [run_trial(env, cereb=cereb, push=True, learn=learn)
             for _ in range(n_adapt)]
    catch = [run_trial(env, cereb=cereb, push=False) for _ in range(6)]
    return base, adapt, catch, cereb


if __name__ == "__main__":
    base, adapt, catch, cereb = run_protocol(learn=True, n_adapt=40)
    _, adapt_l, _, _ = run_protocol(learn=False, n_adapt=40)

    print("人形平衡实验: 150N 推力适应 (基线可恢复但晃动大)")
    print(f"  基线最大俯仰 {np.mean([r['max_pitch'] for r in base]):.3f} rad, "
          f"COM 最大偏移 {np.mean([r['max_com'] for r in base]):.3f} m")
    for blk in range(4):
        seg = adapt[blk * 10:(blk + 1) * 10]
        print(f"  适应 {blk * 10 + 1:2d}-{blk * 10 + 10:2d}: "
              f"俯仰 {np.mean([r['max_pitch'] for r in seg]):.3f}, "
              f"COM 偏移 {np.mean([r['max_com'] for r in seg]):.3f} m, "
              f"推前预倾 {np.mean([r['pre_lean'] for r in seg]):+.3f} m")
    print(f"  捕获试次 (撤推力) 预倾: "
          f"{np.mean([r['pre_lean'] for r in catch]):+.3f} m "
          f"(基线约 -0.007; 更负 = 预期性后遗)")

    # 迁移测试: 训练后面对更大推力, 与无小脑对照
    env = HumanoidBalance()
    print("  迁移测试 (训练后, 每力 3 试次, 摔倒数):")
    for F in (170.0, 190.0, 210.0):
        PUSH_F_OLD = PUSH_F
        globals()["PUSH_F"] = F
        tr = [run_trial(env, cereb=cereb, push=True) for _ in range(3)]
        ls = [run_trial(env, cereb=None, push=True) for _ in range(3)]
        globals()["PUSH_F"] = PUSH_F_OLD
        print(f"    {F:.0f}N: 有小脑 {sum(r['fell'] for r in tr)}/3 倒 "
              f"(俯仰 {np.mean([r['max_pitch'] for r in tr]):.2f}) | "
              f"无小脑 {sum(r['fell'] for r in ls)}/3 倒")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(1, len(adapt) + 1)
    axes[0].plot(x, [r["max_com"] * 100 for r in adapt], "o-", ms=3,
                 label="with cerebellum")
    axes[0].plot(x, [r["max_com"] * 100 for r in adapt_l], "s--", ms=3,
                 label="lesion (frozen)")
    axes[0].set_xlabel("adaptation trial")
    axes[0].set_ylabel("max COM excursion (cm)")
    axes[0].legend(); axes[0].set_title("Push recovery: COM excursion @150N")
    axes[1].plot(x, [r["max_pitch"] for r in adapt], "o-", ms=3,
                 label="pitch, cerebellum")
    axes[1].plot(x, [r["max_pitch"] for r in adapt_l], "s--", ms=3,
                 label="pitch, lesion")
    axes[1].set_xlabel("adaptation trial")
    axes[1].set_ylabel("max pitch (rad)")
    axes[1].legend(); axes[1].set_title("Max pitch excursion over trials")
    fig.tight_layout(); fig.savefig("exp5_humanoid_balance.png", dpi=110)
    print("图已保存: exp5_humanoid_balance.png")
