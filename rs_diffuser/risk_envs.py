"""风险敏感 MuJoCo 任务（Risk-sensitive D4RL）。

在标准 D4RL 运动任务上加入“罕见但严重”的随机惩罚：当智能体进入危险状态时，
以概率 p 获得一个大的负奖励。危险状态的定义：

- HalfCheetah：前向速度超过阈值 v̄（medium / medium-replay 为 4，含 expert 的数据集为 10）
- Walker2D / Hopper：躯干俯仰角绝对值超过阈值（姿态不稳）

默认参数参照 ORAAC（Urpí et al., ICLR 2021）与 CODAC（Ma et al., NeurIPS 2021）
使用的设定，可在 ``get_risk_config`` 中修改。离线数据与在线评测使用同一套规则：
离线数据通过 ``relabel_dataset`` 重新打标签，在线评测通过 ``RiskyMuJoCoWrapper``。
"""

from dataclasses import dataclass
from typing import Dict, Optional

import gym
import numpy as np


@dataclass
class RiskConfig:
    obs_index: int        # 观测中用于判断危险状态的维度
    threshold: float      # 危险阈值
    penalty: float        # 惩罚（负数）
    prob: float = 0.1     # 进入危险状态后触发惩罚的概率
    use_abs: bool = False  # 是否对该维度取绝对值后比较（俯仰角用）

    def is_risky(self, obs: np.ndarray) -> np.ndarray:
        value = obs[..., self.obs_index]
        value = np.abs(value) if self.use_abs else value
        return value > self.threshold


def get_risk_config(env_name: str) -> RiskConfig:
    """按 D4RL 任务名返回风险规则。"""
    if env_name.startswith("halfcheetah"):
        # obs[8] = qvel[0]，即前向速度
        threshold = 10.0 if "expert" in env_name else 4.0
        return RiskConfig(obs_index=8, threshold=threshold, penalty=-70.0, prob=0.1)
    if env_name.startswith("walker2d"):
        # obs[1] = qpos[2]，躯干俯仰角
        return RiskConfig(obs_index=1, threshold=0.5, penalty=-30.0, prob=0.1, use_abs=True)
    if env_name.startswith("hopper"):
        return RiskConfig(obs_index=1, threshold=0.1, penalty=-50.0, prob=0.1, use_abs=True)
    raise NotImplementedError(f"{env_name} 没有定义风险规则")


def relabel_dataset(dataset: Dict[str, np.ndarray], cfg: RiskConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    """对 D4RL 离线数据集按风险规则重新计算奖励，返回新字典（不修改原数据）。

    危险判断基于转移后的状态 s_{t+1}，与在线评测时 ``step`` 返回的观测一致。
    额外写入 ``risky``（是否处于危险状态）和 ``penalized``（是否实际触发惩罚）。
    """
    rng = np.random.default_rng(seed)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dataset.items()}
    next_obs = dataset.get("next_observations")
    if next_obs is None:
        next_obs = np.concatenate([dataset["observations"][1:], dataset["observations"][-1:]], 0)
    risky = cfg.is_risky(next_obs)
    penalized = risky & (rng.random(risky.shape[0]) < cfg.prob)
    out["rewards"] = dataset["rewards"].astype(np.float32) + cfg.penalty * penalized.astype(np.float32)
    out["risky"] = risky
    out["penalized"] = penalized
    return out


class RiskyMuJoCoWrapper(gym.Wrapper):
    """在线评测用：按同一规则对奖励加随机惩罚，并在 info 中记录是否违规。"""

    def __init__(self, env: gym.Env, cfg: RiskConfig, seed: Optional[int] = None):
        super().__init__(env)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        risky = bool(self.cfg.is_risky(obs))
        penalized = risky and self.rng.random() < self.cfg.prob
        if penalized:
            reward += self.cfg.penalty
        info = dict(info, risky=risky, penalized=penalized)
        return obs, reward, done, info
