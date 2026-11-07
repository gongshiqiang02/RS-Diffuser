"""Monte Carlo 分布式价值 critic（论文 4.4 节）。

critic 以带噪规划 x_k、扩散时间 k 为输入（当前状态 s_t 固定在规划序列第 0 帧，
因此 critic 同样以 s_t 为条件），输出 N 个回报分位数，用分位数 Huber 损失
拟合数据集中的 Monte Carlo 折扣回报 R_t（Eq. 5–7）。

推理时 ``logp`` 返回所选风险泛函的值，CleanDiffuser 的 classifier guidance
会对它关于 x_k 求梯度，并在每个去噪步修正噪声预测（Eq. 10 / Eq. 2）。
"""

import torch
import torch.nn.functional as F
from cleandiffuser.classifier.base import BaseClassifier
from cleandiffuser.nn_classifier import BaseNNClassifier

from rs_diffuser.risk import RiskMeasure, quantile_taus


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """分位数 Huber 损失（Eq. 7）。

    Args:
        pred: (B, N) 预测分位数 z_i
        target: (B, 1) Monte Carlo 回报样本 R_t
        taus: (N,) 分位点
        kappa: Huber 阈值
    """
    u = target - pred  # (B, N)
    huber = F.huber_loss(pred, target.expand_as(pred), reduction="none", delta=kappa)
    weight = torch.abs(taus.view(1, -1) - (u.detach() < 0).float())
    return (weight * huber / kappa).mean()


class QuantileCritic(BaseClassifier):
    """分位数回归 critic，作为 CleanDiffuser 扩散模型的 classifier 插件使用。

    Args:
        nn_classifier: 输出维度为 ``n_quantiles`` 的骨干网络（如 HalfDiT1d）。
        n_quantiles: 分位数个数 N。
        kappa: 分位数 Huber 损失的阈值。
        risk: 推理时使用的风险泛函，可通过 ``set_risk`` 随时切换。
        ema_rate: EMA 更新率，推理时使用 EMA 权重。
    """

    def __init__(
        self,
        nn_classifier: BaseNNClassifier,
        n_quantiles: int = 128,
        kappa: float = 1.0,
        risk: RiskMeasure = RiskMeasure("cvar_lower", 0.1),
        ema_rate: float = 0.995,
    ):
        super().__init__(nn_classifier, ema_rate)
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        self.risk = risk
        self.register_buffer("taus", quantile_taus(n_quantiles))

    def set_risk(self, name: str, alpha: float = 0.1):
        self.risk = RiskMeasure(name, alpha)
        return self

    def quantiles(self, x: torch.Tensor, t: torch.Tensor, use_ema: bool = True) -> torch.Tensor:
        model = self.model_ema if use_ema else self.model
        return model(x, t, None)

    def loss(self, x: torch.Tensor, t: torch.Tensor, R: torch.Tensor):
        """x: 带噪规划 (B, H+1, obs_dim)；t: 扩散时间 (B,)；R: 归一化后的回报 (B, 1)。"""
        pred = self.model(x, t, None)
        return quantile_huber_loss(pred, R, self.taus, self.kappa)

    def logp(self, x: torch.Tensor, t: torch.Tensor, c=None):
        """返回风险泛函 rho_alpha(x_k, s_t, k)，形状 (B, 1)，供引导梯度使用。"""
        return self.risk(self.model_ema(x, t, None))
