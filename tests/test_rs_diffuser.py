"""CPU 冒烟测试：风险泛函、分位数回归、模型组装与风险引导采样。

    pytest -q tests
"""

import numpy as np
import pytest
import torch

from rs_diffuser.agent import RSDiffuserAgent, RSDiffuserConfig, RSDiffuserMeta, build_invdyn, build_planner
from rs_diffuser.critic import quantile_huber_loss
from rs_diffuser.data import PlanDataset, TransitionDataset
from rs_diffuser.evaluation import cvar_of_returns
from rs_diffuser.risk import RiskMeasure, quantile_taus, risk_functional
from rs_diffuser.risk_envs import RiskyMuJoCoWrapper, get_risk_config, relabel_dataset

TINY = RSDiffuserConfig(horizon=4, d_model=32, n_heads=2, planner_depth=1, critic_depth=1, emb_dim=16,
                        n_quantiles=16, invdyn_hidden=32, sample_steps=3, num_candidates=4)


def test_risk_functionals_on_known_quantiles():
    q = torch.arange(10, dtype=torch.float32).flip(0)[None]  # 故意乱序，检验排序
    assert risk_functional(q, "mean").item() == pytest.approx(4.5)
    assert risk_functional(q, "cvar_lower", 0.2).item() == pytest.approx(0.5)   # 最差 2 个: 0,1
    assert risk_functional(q, "cvar_upper", 0.2).item() == pytest.approx(8.5)   # 最好 2 个: 8,9
    assert risk_functional(q, "var", 0.5).item() == pytest.approx(5.0)
    assert quantile_taus(4).tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])
    with pytest.raises(ValueError):
        RiskMeasure("unknown")


def test_quantile_regression_recovers_distribution():
    """直接优化 N 个分位数去拟合双峰分布，学到的分位数应接近经验分位数。"""
    torch.manual_seed(0)
    n = 32
    taus = quantile_taus(n)
    samples = torch.cat([torch.randn(4000) * 0.1 - 1.0, torch.randn(1000) * 0.1 + 2.0])  # 80% 在 -1，20% 在 2
    z = torch.zeros(1, n, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(600):
        batch = samples[torch.randint(len(samples), (256,))][:, None]
        loss = quantile_huber_loss(z.expand(256, n), batch, taus, kappa=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    empirical = torch.quantile(samples, taus)
    assert torch.allclose(z.detach()[0], empirical, atol=0.25)
    # 下尾 CVaR 落在 -1 附近，上尾落在 2 附近：同一组分位数给出截然不同的风险评价
    assert risk_functional(z.detach(), "cvar_lower", 0.1).item() < -0.8
    assert risk_functional(z.detach(), "cvar_upper", 0.1).item() > 1.5


def test_relabel_and_wrapper_follow_same_rule():
    cfg = get_risk_config("halfcheetah-medium-v2")
    obs = np.zeros((1000, 17), np.float32)
    obs[:500, 8] = 5.0  # 前一半超速
    data = {"observations": obs, "next_observations": obs, "rewards": np.ones(1000, np.float32)}
    out = relabel_dataset(data, cfg, seed=0)
    assert out["risky"][:500].all() and not out["risky"][500:].any()
    assert not out["penalized"][500:].any()
    assert 0.05 < out["penalized"][:500].mean() < 0.15          # 约 10% 触发
    assert np.allclose(out["rewards"][out["penalized"]], 1.0 - 70.0)
    assert np.allclose(data["rewards"], 1.0)                    # 不修改原数据


def _toy_dataset(n_paths=6, length=40, obs_dim=3, act_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n_paths * length, obs_dim)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(n_paths * length, act_dim)).astype(np.float32)
    timeouts = np.zeros(n_paths * length, bool)
    timeouts[length - 1 :: length] = True
    return {"observations": obs, "actions": act, "rewards": rng.normal(size=n_paths * length).astype(np.float32),
            "terminals": np.zeros(n_paths * length, bool), "timeouts": timeouts}


def test_datasets_shapes():
    ds = PlanDataset(_toy_dataset(), horizon=TINY.horizon, max_path_length=40)
    item = ds[0]
    assert item["x0"].shape == (TINY.horizon + 1, 3)
    assert item["condition_cg"].shape == (1,) and abs(item["condition_cg"].item()) <= 1.0 + 1e-6
    tr = TransitionDataset(ds)
    assert len(tr) == 6 * 39 and tr[0]["obs"].shape == (3,)


def test_planner_training_step_and_guided_sampling():
    torch.manual_seed(0)
    obs_dim, act_dim = 3, 2
    ds = PlanDataset(_toy_dataset(), horizon=TINY.horizon, max_path_length=40)
    planner = build_planner(obs_dim, TINY)
    batch = torch.utils.data.default_collate([ds[i] for i in range(8)])

    # 一次规划器损失 + 一次 critic 损失（与 Lightning training_step 中的两步一致）
    loss_plan = planner.loss(batch["x0"])
    xt, t, _ = planner.add_noise(batch["x0"])
    loss_critic = planner.classifier.loss(xt, t, batch["condition_cg"])
    (loss_plan + loss_critic).backward()
    assert torch.isfinite(loss_plan) and torch.isfinite(loss_critic)

    meta = RSDiffuserMeta("toy", obs_dim, act_dim, ds.return_scale, ds.normalizer.mean.tolist(),
                          ds.normalizer.std.tolist())
    agent = RSDiffuserAgent(planner, build_invdyn(obs_dim, act_dim, TINY), meta, TINY)
    obs = np.random.default_rng(1).normal(size=(5, obs_dim)).astype(np.float32)
    for name in ("cvar_lower", "mean", "cvar_upper", "var"):
        agent.set_risk(name, 0.1)
        act = agent.act(obs)
        assert act.shape == (5, act_dim) and np.all(np.abs(act) <= 1.0)


def test_risk_guidance_changes_gradient_direction():
    """下尾 CVaR 与上尾目标对同一批规划的梯度应不同，说明推理时切换风险偏好确实改变引导信号。"""
    torch.manual_seed(0)
    planner = build_planner(3, TINY)
    critic = planner.classifier
    x = torch.randn(4, TINY.horizon + 1, 3)
    t = torch.full((4,), 0.5)
    # DiT 输出层为零初始化，未训练的 critic 输出与输入无关；先拟合一个依赖输入的回报分布
    opt = torch.optim.Adam(critic.model.parameters(), lr=1e-2)
    for _ in range(50):
        xb = torch.randn(64, TINY.horizon + 1, 3)
        R = xb[:, 1:, 0].mean(1, keepdim=True) + 0.5 * torch.randn(64, 1) * xb[:, 1:, 1].mean(1, keepdim=True).sign()
        loss = critic.loss(xb, torch.rand(64), R)
        opt.zero_grad(); loss.backward(); opt.step()
    critic.model_ema.load_state_dict(critic.model.state_dict())
    critic.set_risk("cvar_lower", 0.1)
    _, g_low = critic.gradients(x.clone(), t, None)
    critic.set_risk("cvar_upper", 0.1)
    _, g_up = critic.gradients(x.clone(), t, None)
    assert g_low.shape == x.shape and not torch.allclose(g_low, g_up)


def test_cvar_of_returns():
    returns = np.arange(100, dtype=float)
    assert cvar_of_returns(returns, 0.1) == pytest.approx(4.5)  # 最差 10 个: 0..9


def test_lightning_fit_updates_planner_and_critic():
    """Lightning 训练循环中，规划器与 critic 两个优化器都应真正更新各自的参数。"""
    import pytorch_lightning as L

    torch.manual_seed(0)
    ds = PlanDataset(_toy_dataset(), horizon=TINY.horizon, max_path_length=40)
    planner = build_planner(3, TINY)
    before_diff = [p.detach().clone() for p in planner.model["diffusion"].parameters()]
    before_critic = [p.detach().clone() for p in planner.classifier.model.parameters()]
    L.Trainer(accelerator="cpu", max_steps=5, logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False).fit(
        planner, torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True))
    changed = lambda before, params: any(not torch.equal(b, p) for b, p in zip(before, params))
    assert changed(before_diff, planner.model["diffusion"].parameters())
    assert changed(before_critic, planner.classifier.model.parameters())
