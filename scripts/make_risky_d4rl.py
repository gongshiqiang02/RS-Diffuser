"""生成风险敏感 D4RL 离线数据集。

下载标准 D4RL 数据，按 ``rs_diffuser/risk_envs.py`` 中的规则对奖励加随机惩罚，保存为 hdf5。

    python scripts/make_risky_d4rl.py --task halfcheetah-medium-v2 --out data/halfcheetah-medium-v2-risky.hdf5
"""

import argparse
from pathlib import Path

import d4rl  # noqa: F401
import gym
import numpy as np

from rs_diffuser.data import save_hdf5
from rs_diffuser.risk_envs import get_risk_config, relabel_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="D4RL 任务名，如 halfcheetah-medium-v2")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = get_risk_config(args.task)
    data = relabel_dataset(gym.make(args.task).get_dataset(), cfg, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(data, args.out)

    n = len(data["rewards"])
    print(f"{args.task}: {n} transitions | 规则 {cfg}")
    print(f"处于危险状态 {data['risky'].mean():.2%}，实际触发惩罚 {data['penalized'].sum()} 次 ({data['penalized'].mean():.2%})")
    print(f"平均奖励 {np.mean(data['rewards']):.3f} -> 已保存 {args.out}")


if __name__ == "__main__":
    main()
