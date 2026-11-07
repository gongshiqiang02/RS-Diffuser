"""RS-Diffuser 模型组装与风险引导规划（论文 4.2–4.5 节，Algorithm 1）。"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_classifier import HalfDiT1d
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import GaussianNormalizer

from rs_diffuser.critic import QuantileCritic
from rs_diffuser.risk import RiskMeasure


@dataclass
class RSDiffuserConfig:
    """默认值对应论文 5.1 节的模型配置。"""

    horizon: int = 32                 # 规划步数 H
    d_model: int = 256                # Transformer 隐层维度
    n_heads: int = 8
    planner_depth: int = 2            # 2-block Transformer
    critic_depth: int = 2
    emb_dim: int = 128
    n_quantiles: int = 128            # 分位数个数 N
    kappa: float = 1.0                # 分位数 Huber 阈值
    invdyn_hidden: int = 256          # 逆动力学 MLP 隐层
    lr: float = 3e-4
    ema_rate: float = 0.999
    next_state_loss_weight: float = 10.0  # 加大 s_{t+1} 的去噪损失权重，它直接决定执行的动作
    # 推理
    solver: str = "ddim"
    sample_steps: int = 20            # DDIM 20 步
    w_cg: float = 0.3                 # 风险引导强度 w
    num_candidates: int = 32          # 每个状态并行采样的候选规划数，按风险值选最优


@dataclass
class RSDiffuserMeta:
    """与权重一起保存的元信息，推理时据此重建模型与归一化。"""

    task: str
    obs_dim: int
    act_dim: int
    return_scale: float
    obs_mean: list
    obs_std: list
    config: dict = field(default_factory=dict)


def build_planner(obs_dim: int, cfg: RSDiffuserConfig) -> ContinuousDiffusionSDE:
    """状态规划扩散模型 + 分位数 critic（作为 classifier 插件）。"""
    seq_len = cfg.horizon + 1  # 第 0 帧为当前状态 s_t
    nn_diffusion = DiT1d(
        x_dim=obs_dim, x_seq_len=seq_len, emb_dim=cfg.emb_dim, d_model=cfg.d_model,
        n_heads=cfg.n_heads, depth=cfg.planner_depth, timestep_emb_type="untrainable_fourier",
    )
    nn_critic = HalfDiT1d(
        x_dim=obs_dim, out_dim=cfg.n_quantiles, x_seq_len=seq_len, emb_dim=cfg.emb_dim,
        d_model=cfg.d_model, n_heads=cfg.n_heads, depth=cfg.critic_depth,
        timestep_emb_type="untrainable_fourier",
    )
    critic = QuantileCritic(nn_critic, n_quantiles=cfg.n_quantiles, kappa=cfg.kappa, ema_rate=cfg.ema_rate)

    fix_mask = torch.zeros((seq_len, obs_dim))
    fix_mask[0] = 1.0  # 固定当前状态，只生成未来 H 步
    loss_weight = torch.ones((seq_len, obs_dim))
    loss_weight[1] = cfg.next_state_loss_weight

    return ContinuousDiffusionSDE(
        nn_diffusion, None, fix_mask, loss_weight, classifier=critic, ema_rate=cfg.ema_rate,
        optimizer_params={
            "diffusion": {"optimizer": "adam", "lr": cfg.lr},
            "classifier": {"optimizer": "adam", "lr": cfg.lr},
        },
    )


def build_invdyn(obs_dim: int, act_dim: int, cfg: RSDiffuserConfig) -> MlpInvDynamic:
    """确定性逆动力学 a_t = f(s_t, s_{t+1})，MSE 训练（Eq. 4）。"""
    return MlpInvDynamic(obs_dim, act_dim, hidden_dim=cfg.invdyn_hidden, tanh_out_activation=True)


class RSDiffuserAgent:
    """推理：风险引导采样未来状态规划，再由逆动力学解码出动作。"""

    def __init__(self, planner, invdyn, meta: RSDiffuserMeta, cfg: RSDiffuserConfig, device="cpu"):
        self.planner = planner.to(device).eval()
        self.invdyn = invdyn.to(device).eval()
        self.meta = meta
        self.cfg = cfg
        self.device = device
        self.normalizer = GaussianNormalizer(x_mean=np.asarray(meta.obs_mean, np.float32),
                                             x_std=np.asarray(meta.obs_std, np.float32))

    @property
    def critic(self) -> QuantileCritic:
        return self.planner.classifier

    def set_risk(self, name: str, alpha: float = 0.1):
        """推理时切换风险偏好，无需重新训练（论文 4.5 节 Remarks）。"""
        self.critic.set_risk(name, alpha)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (B, obs_dim) 原始观测；返回 (B, act_dim) 动作。"""
        cfg, n_envs, c = self.cfg, obs.shape[0], self.cfg.num_candidates
        norm_obs = torch.as_tensor(self.normalizer.normalize(obs), dtype=torch.float32, device=self.device)
        prior = torch.zeros((n_envs * c, cfg.horizon + 1, self.meta.obs_dim), device=self.device)
        prior[:, 0] = norm_obs.repeat_interleave(c, dim=0)

        with torch.enable_grad():  # 引导梯度需要对 x_k 求导
            traj, log = self.planner.sample(
                prior, solver=cfg.solver, sample_steps=cfg.sample_steps,
                w_cg=cfg.w_cg, condition_cg=None, use_ema=True,
            )

        if c > 1:
            if "log_p" in log:
                score = log["log_p"].view(n_envs, c)
            else:  # w_cg = 0 时采样器不计算风险值，这里补算一次用于候选选择
                t0 = torch.zeros((n_envs * c,), device=self.device)
                score = self.critic.logp(traj, t0).view(n_envs, c)
            best = score.argmax(-1)
            traj = traj.view(n_envs, c, *traj.shape[1:])[torch.arange(n_envs), best]

        act = self.invdyn(traj[:, 0], traj[:, 1])
        return act.clamp(-1.0, 1.0).cpu().numpy()


def save_agent(path: Path, planner, invdyn, meta: RSDiffuserMeta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(planner.state_dict(), path / "planner.pt")
    torch.save(invdyn.state_dict(), path / "invdyn.pt")
    (path / "meta.json").write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False))


def load_agent(path: Path, device: str = "cpu", overrides: Optional[dict] = None) -> RSDiffuserAgent:
    path = Path(path)
    meta = RSDiffuserMeta(**json.loads((path / "meta.json").read_text()))
    cfg = RSDiffuserConfig(**{**meta.config, **(overrides or {})})
    planner = build_planner(meta.obs_dim, cfg)
    planner.load_state_dict(torch.load(path / "planner.pt", map_location=device))
    invdyn = build_invdyn(meta.obs_dim, meta.act_dim, cfg)
    invdyn.load_state_dict(torch.load(path / "invdyn.pt", map_location=device))
    return RSDiffuserAgent(planner, invdyn, meta, cfg, device)


def default_risk() -> RiskMeasure:
    return RiskMeasure("cvar_lower", 0.1)
