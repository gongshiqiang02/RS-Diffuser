"""评测：按论文 5.1 节的指标统计 episode 回报与违规步数。"""

from typing import Callable, Dict, List

import numpy as np


def cvar_of_returns(returns: np.ndarray, alpha: float = 0.1) -> float:
    """最差 alpha 比例 episode 的平均回报，例如 100 个 episode 时取最差 10 个。"""
    returns = np.sort(np.asarray(returns, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * len(returns))))
    return float(returns[:k].mean())


def summarize(returns: List[float], violations: List[int]) -> Dict[str, float]:
    returns = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": int(len(returns)),
        "mean": float(returns.mean()),
        "median": float(np.median(returns)),
        "cvar_0.1": cvar_of_returns(returns, 0.1),
        "violations": float(np.mean(violations)),
    }


def run_episodes(
    make_env: Callable[[int], "gym.Env"],
    policy: Callable[[np.ndarray], np.ndarray],
    n_episodes: int = 100,
    n_parallel: int = 10,
    max_steps: int = 500,
    seed: int = 0,
) -> Dict[str, list]:
    """并行跑 ``n_episodes`` 个 episode，返回每个 episode 的回报与违规步数。

    ``make_env(seed)`` 返回单个环境；环境需在 info 中给出 ``risky``。
    多个环境的观测拼成一个 batch 一次性规划，以摊薄扩散采样的开销。
    """
    returns, violations = [], []
    episode_seed = seed * 100_000
    while len(returns) < n_episodes:
        n = min(n_parallel, n_episodes - len(returns))
        envs = []
        for _ in range(n):
            env = make_env(episode_seed)
            env.seed(episode_seed)
            envs.append(env)
            episode_seed += 1
        obs = np.stack([env.reset() for env in envs])
        ep_ret, ep_vio = np.zeros(n), np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=bool)
        for _ in range(max_steps):
            if not alive.any():
                break
            act = policy(obs)
            for i, env in enumerate(envs):
                if not alive[i]:
                    continue
                o, r, d, info = env.step(act[i])
                obs[i] = o
                ep_ret[i] += r
                ep_vio[i] += int(info.get("risky", False))
                alive[i] = not d
        returns += ep_ret.tolist()
        violations += ep_vio.tolist()
        for env in envs:
            env.close()
    return {"returns": returns, "violations": violations}
