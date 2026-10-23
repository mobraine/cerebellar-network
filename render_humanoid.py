"""
把人形平衡实验录成 GIF (侧视骨架渲染)。
说明: 本机无 GPU/GL 环境, MuJoCo 自带 3D 渲染器跑不起来;
模型本身是平面 (矢状面) 的, 故用 MuJoCo 的刚体位置数据
画侧视骨架动画 —— 动力学与数据都来自 MuJoCo 本体。
在自己电脑上看真 3D 窗口请用 watch_humanoid.py。

场景: A) 150N 无小脑  B) 190N 无小脑 (倒)  C) 190N 有小脑 (稳)
运行: .venv/bin/python render_humanoid.py
产物: humanoid_demo.gif + cerebellum_weights.npz
"""
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

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


def body_xz(d, bid):
    p = d.body(bid).xpos
    return np.array([p[0], p[2]])


def local_to_world(d, bid, local):
    b = d.body(bid)
    return b.xpos[[0, 2]] + (b.xmat.reshape(3, 3) @ local)[[0, 2]]


def skeleton(d):
    """返回各关节点的 (x,z): 躯干基点/顶, 左右腿 髋/膝/踝/脚尖/脚跟"""
    torso_base = body_xz(d, 1)
    torso_top = local_to_world(d, 1, np.array([0, 0, 0.55]))
    head = local_to_world(d, 1, np.array([0, 0, 0.70]))
    legs = {}
    for side, (th, sh, ft) in {"l": (2, 3, 4), "r": (5, 6, 7)}.items():
        hip = body_xz(d, th); knee = body_xz(d, sh); ankle = body_xz(d, ft)
        toe = local_to_world(d, ft, np.array([0.19, 0, -0.045]))
        heel = local_to_world(d, ft, np.array([-0.09, 0, -0.045]))
        legs[side] = (hip, knee, ankle, toe, heel)
    com = d.subtree_com[1]
    return torso_base, torso_top, head, legs, np.array([com[0], com[2]])


def draw_frame(d, label, t, pushing):
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=100)
    tb, tt, head, legs, com = skeleton(d)
    ax.axhline(0, color="0.4", lw=2)
    for side, c, lw in (("r", "0.65", 3), ("l", "#2050a0", 4)):
        hip, knee, ankle, toe, heel = legs[side]
        pts = np.array([hip, knee, ankle])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=lw,
                solid_capstyle="round")
        ax.plot([heel[0], toe[0]], [heel[1], toe[1]], "-", color=c, lw=lw)
    ax.plot([tb[0], tt[0]], [tb[1], tt[1]], "-", color="#2050a0", lw=7,
            solid_capstyle="round")
    ax.add_patch(plt.Circle(head, 0.085, color="#e8c890", zorder=5))
    ax.plot(com[0], com[1], "o", color="red", ms=7, zorder=6)
    if pushing:
        ax.annotate("", xy=(tb[0] - 0.05, 1.35), xytext=(tb[0] - 0.75, 1.35),
                    arrowprops=dict(arrowstyle="->", color="red", lw=3))
        ax.text(tb[0] - 0.72, 1.45, "PUSH", color="red", fontsize=11,
                weight="bold")
    ax.set_xlim(-0.9, 1.1); ax.set_ylim(-0.05, 1.85)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{label}   t={t:.1f}s", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(arr)


def record(env, cereb, force, label, fps=25):
    H.PUSH_F = force
    env.reset()
    m, d = env.m, env.d
    frames = []
    skip = int(1 / fps / m.opt.timestep)
    tau_cb = np.zeros(6)
    for i in range(int(5.2 / m.opt.timestep)):
        t = i * m.opt.timestep
        tau_fb = env.baseline_torque()
        if cereb is not None and i % H.CTRL_EVERY == 0:
            tau_cb = cereb.torque(env.state(t))
        pushing = H.PUSH_T <= t <= H.PUSH_T + H.PUSH_DUR
        d.xfrc_applied[1] = [force, 0, 0, 0, 0, 0] if pushing else 0
        d.ctrl[:] = np.clip((tau_fb + tau_cb) / env.gear, -1, 1)
        mujoco.mj_step(m, d)
        if t >= 1.4 and i % skip == 0:
            frames.append(draw_frame(d, label, t, pushing))
        if env.fallen and t > H.PUSH_T + 1.2:
            break
    return frames


if __name__ == "__main__":
    _, adapt, _, cereb = H.run_protocol(learn=True, n_adapt=40)
    np.savez("cerebellum_weights.npz", W=cereb.W, centers=cereb.centers,
             mu=cereb.mu, sd=cereb.sd)
    print("trained; max_com first/last:",
          round(adapt[0]["max_com"], 3), round(adapt[-1]["max_com"], 3))
    env = HumanoidBalance()
    cb = load_cerebellum()
    frames = []
    frames += record(env, None, 150.0, "150N - no cerebellum")
    frames += record(env, None, 190.0, "190N - no cerebellum: FALLS")
    frames += record(env, cb, 190.0, "190N - with cerebellum: STANDS")
    frames[0].save("humanoid_demo.gif", save_all=True,
                   append_images=frames[1:], duration=40, loop=0)
    print(f"saved humanoid_demo.gif ({len(frames)} frames)")
