"""离线数据：读取 / 保存数据集，构造规划器与逆动力学的训练样本。"""

from typing import Dict, Optional

import h5py
import numpy as np
import torch
from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoDataset

DATA_KEYS = ("observations", "actions", "rewards", "terminals", "timeouts")


def save_hdf5(dataset: Dict[str, np.ndarray], path: str):
    with h5py.File(path, "w") as f:
        for k, v in dataset.items():
            if isinstance(v, np.ndarray):
                f.create_dataset(k, data=v, compression="gzip")


def load_hdf5(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in f.keys()}


def load_offline_dataset(task: str, dataset_path: Optional[str] = None, seed: int = 0) -> Dict[str, np.ndarray]:
    """返回 D4RL 格式的离线数据字典。

    - 指定 ``dataset_path`` 时直接读取 hdf5（由 ``scripts/make_risky_d4rl.py`` 或
      ``scripts/collect_navigation.py`` 生成）。
    - 否则对 D4RL MuJoCo 任务现场下载原始数据并按风险规则重新打标签。
    """
    if dataset_path is not None:
        return load_hdf5(dataset_path)
    if task.startswith(("halfcheetah", "walker2d", "hopper")):
        import d4rl  # noqa: F401
        import gym

        from rs_diffuser.risk_envs import get_risk_config, relabel_dataset

        raw = gym.make(task).get_dataset()
        return relabel_dataset(raw, get_risk_config(task), seed=seed)
    raise ValueError(f"{task} 需要通过 --dataset 指定离线数据文件")


class PlanDataset(torch.utils.data.Dataset):
    """规划器 + critic 的训练样本（论文 Algorithm 1 第 4 行）。

    每个样本是一段长度 H+1 的归一化状态序列 (s_t, s_{t+1}, ..., s_{t+H})，
    其中 s_t 在训练与采样时都通过 fix_mask 固定，扩散模型只生成未来 H 步，
    即论文中的 x0 = (s_{t+1}, ..., s_{t+H}) | s_t。
    ``condition_cg`` 是从 t 开始的 Monte Carlo 折扣回报 R_t（Eq. 5），除以
    ``return_scale`` 归一化，作为 critic 的回归目标。
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        horizon: int = 32,
        discount: float = 0.99,
        max_path_length: int = 1000,
        terminal_penalty: Optional[float] = -100.0,
        return_scale: Optional[float] = None,
    ):
        self.seq = D4RLMuJoCoDataset(
            {k: data[k] for k in DATA_KEYS},
            terminal_penalty=terminal_penalty,
            horizon=horizon + 1,
            max_path_length=max_path_length,
            discount=discount,
        )
        self.obs_dim, self.act_dim = self.seq.obs_dim, self.seq.act_dim
        self.normalizer = self.seq.get_normalizer()
        idx = np.asarray(self.seq.indices)
        starts = self.seq.seq_val[idx[:, 0], idx[:, 1], 0]
        # 默认用起点回报绝对值的最大值归一化，使 critic 目标落在 [-1, 1]
        self.return_scale = float(return_scale or max(np.abs(starts).max(), 1e-6))

    def __len__(self):
        return len(self.seq.indices)

    def __getitem__(self, i):
        path, start, end = self.seq.indices[i]
        return {
            "x0": self.seq.seq_obs[path, start:end],
            "condition_cg": self.seq.seq_val[path, start] / self.return_scale,
        }


class TransitionDataset(torch.utils.data.Dataset):
    """逆动力学的训练样本 (s_t, a_t, s_{t+1})，状态已归一化（Eq. 4）。"""

    def __init__(self, plan_dataset: PlanDataset):
        seq = plan_dataset.seq
        obs, act, next_obs = [], [], []
        for p, length in enumerate(seq.path_lengths):
            length = min(int(length), seq.seq_obs.shape[1])
            if length < 2:
                continue
            obs.append(seq.seq_obs[p, : length - 1])
            act.append(seq.seq_act[p, : length - 1])
            next_obs.append(seq.seq_obs[p, 1:length])
        self.obs = np.concatenate(obs)
        self.act = np.concatenate(act)
        self.next_obs = np.concatenate(next_obs)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return {"obs": self.obs[i], "act": self.act[i], "next_obs": self.next_obs[i]}
