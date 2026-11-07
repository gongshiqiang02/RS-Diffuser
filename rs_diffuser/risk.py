"""从分位数估计中计算风险泛函（论文 Eq. 8–9）。

critic 输出 N 个分位数 z_1..z_N，对应分位点 tau_i = (i - 0.5) / N。
不同的风险偏好只是对这组分位数取不同的泛函，因此同一个 critic
可以在推理时切换风险规避 / 风险中性 / 风险偏好，而无需重新训练。
"""

from dataclasses import dataclass

import torch

SUPPORTED_MEASURES = ("cvar_lower", "cvar_upper", "mean", "var")


def quantile_taus(n_quantiles: int, device=None) -> torch.Tensor:
    """均匀分位点 tau_i = (i - 0.5) / N，严格落在 (0, 1) 内。"""
    return (torch.arange(n_quantiles, device=device, dtype=torch.float32) + 0.5) / n_quantiles


@dataclass
class RiskMeasure:
    """风险泛函配置。

    name:
        - ``cvar_lower``: 下尾 CVaR_alpha，风险规避（Eq. 8）
        - ``cvar_upper``: 上尾均值，风险偏好（Eq. 9）
        - ``mean``: 分位数均值，即期望回报，风险中性
        - ``var``: VaR_alpha，即 alpha 分位点处的回报
    alpha: 尾部比例，例如 0.1 表示最差 / 最好的 10%。
    """

    name: str = "cvar_lower"
    alpha: float = 0.1

    def __post_init__(self):
        if self.name not in SUPPORTED_MEASURES:
            raise ValueError(f"未知风险度量 {self.name}，可选 {SUPPORTED_MEASURES}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha 需要在 (0, 1] 内")

    def __call__(self, quantiles: torch.Tensor) -> torch.Tensor:
        return risk_functional(quantiles, self.name, self.alpha)

    def __str__(self):
        return self.name if self.name == "mean" else f"{self.name}@{self.alpha:g}"


def risk_functional(quantiles: torch.Tensor, name: str = "cvar_lower", alpha: float = 0.1) -> torch.Tensor:
    """对 (B, N) 的分位数计算风险泛函，返回 (B, 1)。

    先对分位数排序再按 tau 截取尾部。网络输出的分位数不保证单调
    （quantile crossing），排序后才能保证“前 k 个就是 tau <= alpha 的部分”，
    这与 Eq. 8 中 I_alpha = {i : tau_i <= alpha} 的定义一致。
    """
    n = quantiles.shape[-1]
    sorted_q, _ = torch.sort(quantiles, dim=-1)
    taus = quantile_taus(n, quantiles.device)

    if name == "mean":
        return sorted_q.mean(-1, keepdim=True)
    if name == "cvar_lower":
        k = max(1, int((taus <= alpha).sum().item()))
        return sorted_q[..., :k].mean(-1, keepdim=True)
    if name == "cvar_upper":
        k = max(1, int((taus >= 1.0 - alpha).sum().item()))
        return sorted_q[..., -k:].mean(-1, keepdim=True)
    if name == "var":
        idx = min(n - 1, max(0, int(torch.searchsorted(taus, torch.tensor(alpha, device=taus.device)).item())))
        return sorted_q[..., idx : idx + 1]
    raise ValueError(f"未知风险度量 {name}")
