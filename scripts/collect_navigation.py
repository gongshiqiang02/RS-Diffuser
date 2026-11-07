"""采集 Risky PointMass 离线数据集。

行为策略是三种脚本控制器的混合，加入动作噪声：
  - 直线：径直走向目标，会穿过危险区域（回报高但有风险）
  - 左绕 / 右绕：先经过危险区域侧方的路点再走向目标（更安全但更慢）
数据中同时包含冒险与保守两类轨迹，离线算法需要从中学会在不同风险偏好下取舍。

    python scripts/collect_navigation.py --task risky-pointmass --episodes 2000 --out data/risky-pointmass.hdf5

Risky Ant 需要一个能完成目标导航的行为策略（如用 SAC 训练）来采集数据，
本仓库未附带该策略；采好的数据按同样的 D4RL 格式保存为 hdf5 即可用于训练。
"""

import argparse
from pathlib import Path

import numpy as np

from rs_diffuser.data import save_hdf5
from rs_diffuser.navigation import RiskyPointMassEnv


def scripted_action(pos, goal, waypoint, noise, rng):
    target = waypoint if waypoint is not None else goal
    direction = target - pos
    action = direction / (np.linalg.norm(direction) + 1e-6)
    return np.clip(action + rng.normal(0, noise, 2), -1, 1).astype(np.float32)


def collect_pointmass(n_episodes: int, noise: float, seed: int):
    env = RiskyPointMassEnv(seed=seed)
    rng = np.random.default_rng(seed)
    side = env.hazard_radius + 0.3
    keys = ("observations", "actions", "rewards", "next_observations", "terminals", "timeouts", "risky")
    buf = {k: [] for k in keys}
    for _ in range(n_episodes):
        mode = rng.choice(["direct", "left", "right"], p=[0.4, 0.3, 0.3])
        waypoint = None if mode == "direct" else env.hazard_center + np.array([-side if mode == "left" else side, 0.0])
        obs, done = env.reset(), False
        while not done:
            if waypoint is not None and np.linalg.norm(obs - waypoint) < 0.15:
                waypoint = None  # 到达路点后转向目标
            act = scripted_action(obs, env.goal, waypoint, noise, rng)
            next_obs, rew, done, info = env.step(act)
            for k, v in zip(keys, (obs, act, rew, next_obs, info["reached"], info["TimeLimit.truncated"], info["risky"])):
                buf[k].append(v)
            obs = next_obs
    data = {k: np.asarray(v) for k, v in buf.items()}
    data["observations"] = data["observations"].astype(np.float32)
    data["next_observations"] = data["next_observations"].astype(np.float32)
    data["actions"] = data["actions"].astype(np.float32)
    data["rewards"] = data["rewards"].astype(np.float32)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="risky-pointmass", choices=["risky-pointmass"])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = collect_pointmass(args.episodes, args.noise, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(data, args.out)
    n_ep = int(data["terminals"].sum() + data["timeouts"].sum())
    print(f"{n_ep} 条轨迹，{len(data['rewards'])} 步；处于危险区域的步数占 {data['risky'].mean():.2%} -> {args.out}")


if __name__ == "__main__":
    main()
