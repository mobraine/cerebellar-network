"""
在你自己的电脑上打开 MuJoCo 实时窗口看人形机器人平衡实验。

用法 (需要图形界面):
    pip install mujoco numpy
    python watch_humanoid.py            # 若已有 cerebellum_weights.npz 直接回放
    python watch_humanoid.py --train    # 重新训练 (约 1 分钟) 后回放

回放顺序 (循环播放):
    1) 150N 无小脑 —— 基线, 能站稳但晃得厉害
    2) 190N 无小脑 —— 摔倒
    3) 190N 有小脑 —— 站稳 (小脑只在 150N 训练过, 这是迁移)
关掉窗口即退出。
"""
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

import humanoid_balance as H
from humanoid_balance import HumanoidBalance, CerebellarPosture


def load_cerebellum():
    z = np.load("cerebellum_weights.npz")
    from cerebellar_net import GranuleLayer, BasketLayer
    cb = CerebellarPosture.__new__(CerebellarPosture)
    cb.mu, cb.sd, cb.centers = z["mu"], z["sd"], z["centers"]
    cb.width = 0.9
    cb.granule = GranuleLayer(250, 1500, sparsity=0.05, seed=0)
    cb.basket = BasketLayer(0.5)
    cb.W = z["W"]
    cb.buf_pf, cb.buf_fb = [], []
    return cb


def play(env, viewer, cereb, force, label, realtime=1.0):
    print(label)
    H.PUSH_F = force
    env.reset()
    m, d = env.m, env.d
    tau_cb = np.zeros(6)
    n = int(6.0 / m.opt.timestep)
    for i in range(n):
        t0 = time.time()
        t = i * m.opt.timestep
        tau_fb = env.baseline_torque()
        if cereb is not None and i % H.CTRL_EVERY == 0:
            tau_cb = cereb.torque(env.state(t))
        if H.PUSH_T <= t <= H.PUSH_T + H.PUSH_DUR:
            d.xfrc_applied[1] = [force, 0, 0, 0, 0, 0]
        else:
            d.xfrc_applied[1] = 0
        d.ctrl[:] = np.clip((tau_fb + tau_cb) / env.gear, -1, 1)
        mujoco.mj_step(m, d)
        viewer.sync()
        if env.fallen and t > H.PUSH_T + 1.0:
            time.sleep(0.8)
            return
        dt = m.opt.timestep / realtime - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def main():
    if "--train" in sys.argv or not _exists():
        print("训练小脑中 (约 1 分钟)...")
        _, _, _, cereb = H.run_protocol(learn=True, n_adapt=40)
        np.savez("cerebellum_weights.npz", W=cereb.W,
                 centers=cereb.centers, mu=cereb.mu, sd=cereb.sd)
        print("训练完成, 权重已保存。")
    env = HumanoidBalance()
    cb = load_cerebellum()
    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        viewer.cam.lookat[:] = [0.1, 0, 0.85]
        viewer.cam.distance = 2.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -6
        while viewer.is_running():
            play(env, viewer, None, 150.0, "== 150N, 无小脑 (基线) ==")
            play(env, viewer, None, 190.0, "== 190N, 无小脑: 摔倒 ==")
            play(env, viewer, cb, 190.0, "== 190N, 有小脑: 站稳 ==")


def _exists():
    import os
    return os.path.exists("cerebellum_weights.npz")


if __name__ == "__main__":
    main()
