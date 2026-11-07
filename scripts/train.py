"""训练 RS-Diffuser：状态规划扩散模型 + 分位数 critic + 逆动力学（Algorithm 1 第 2–8 行）。

规划器与 critic 在同一个训练步中各自更新（两个独立的优化器）；逆动力学与二者互不依赖，
单独训练，效果等价于 Algorithm 1 中逐轮依次更新三者。

    python scripts/train.py --task halfcheetah-medium-v2 --dataset data/halfcheetah-medium-v2-risky.hdf5 \
        --out runs/halfcheetah-medium-v2/seed0 --seed 0
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from rs_diffuser.agent import RSDiffuserConfig, RSDiffuserMeta, build_invdyn, build_planner, save_agent
from rs_diffuser.data import PlanDataset, TransitionDataset, load_offline_dataset
from rs_diffuser.tasks import get_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--dataset", default=None, help="hdf5 数据集；D4RL 任务不指定时现场下载并打风险标签")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000, help="规划器与 critic 的训练步数")
    p.add_argument("--invdyn_steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--n_quantiles", type=int, default=128)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=100_000)
    args = p.parse_args()

    L.seed_everything(args.seed)
    out = Path(args.out)
    task = get_task(args.task)
    cfg = RSDiffuserConfig(horizon=args.horizon, n_quantiles=args.n_quantiles)

    data = load_offline_dataset(args.task, args.dataset, seed=args.seed)
    plan_ds = PlanDataset(data, horizon=cfg.horizon, discount=args.discount,
                          max_path_length=task.max_path_length, terminal_penalty=task.terminal_penalty)
    print(f"[data] {len(plan_ds)} 个规划片段 | obs_dim={plan_ds.obs_dim} act_dim={plan_ds.act_dim} "
          f"| return_scale={plan_ds.return_scale:.2f}")

    loader_kw = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                     persistent_workers=args.num_workers > 0, drop_last=True)

    # ---- 规划器 + 分位数 critic ----
    planner = build_planner(plan_ds.obs_dim, cfg)
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=args.devices, max_steps=args.steps,
        default_root_dir=out / "lightning",
        callbacks=[ModelCheckpoint(dirpath=out / "checkpoints", filename="planner-{step}",
                                   every_n_train_steps=args.save_every, save_top_k=-1)],
        log_every_n_steps=100,
    )
    trainer.fit(planner, DataLoader(plan_ds, **loader_kw))

    # ---- 逆动力学 ----
    invdyn = build_invdyn(plan_ds.obs_dim, plan_ds.act_dim, cfg)
    L.Trainer(accelerator=args.accelerator, devices=args.devices, max_steps=args.invdyn_steps,
              default_root_dir=out / "lightning_invdyn", log_every_n_steps=100,
              enable_checkpointing=False).fit(invdyn, DataLoader(TransitionDataset(plan_ds), **loader_kw))

    norm = plan_ds.normalizer
    meta = RSDiffuserMeta(
        task=args.task, obs_dim=plan_ds.obs_dim, act_dim=plan_ds.act_dim, return_scale=plan_ds.return_scale,
        obs_mean=norm.mean.tolist(), obs_std=norm.std.tolist(), config=asdict(cfg),
    )
    save_agent(out, planner.cpu(), invdyn.cpu(), meta)
    print(f"[done] 模型已保存到 {out}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
