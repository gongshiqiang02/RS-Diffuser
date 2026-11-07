"""任务注册：环境构造、episode 长度与数据集处理方式。"""

from dataclasses import dataclass
from typing import Optional

import gym


@dataclass
class TaskSpec:
    name: str
    max_episode_steps: int        # 评测时的最大步数
    max_path_length: int          # 离线数据中单条轨迹的最大长度
    terminal_penalty: Optional[float]  # 提前终止（摔倒）的额外惩罚；导航任务到达目标也会终止，不能惩罚


def get_task(name: str) -> TaskSpec:
    if name.startswith(("halfcheetah", "walker2d", "hopper")):
        # 论文 5.2 节：风险敏感 D4RL 的最大 episode 长度为 500
        return TaskSpec(name, max_episode_steps=500, max_path_length=1000, terminal_penalty=-100.0)
    if name == "risky-pointmass":
        return TaskSpec(name, max_episode_steps=100, max_path_length=100, terminal_penalty=None)
    if name == "risky-ant":
        return TaskSpec(name, max_episode_steps=500, max_path_length=500, terminal_penalty=None)
    raise NotImplementedError(name)


def make_env(name: str, seed: Optional[int] = None) -> gym.Env:
    """构造带风险规则的评测环境，info 中包含 ``risky`` 字段。"""
    if name.startswith(("halfcheetah", "walker2d", "hopper")):
        import d4rl  # noqa: F401

        from rs_diffuser.risk_envs import RiskyMuJoCoWrapper, get_risk_config

        return RiskyMuJoCoWrapper(gym.make(name), get_risk_config(name), seed=seed)
    from rs_diffuser.navigation import make_navigation_env

    return make_navigation_env(name, seed=seed)
