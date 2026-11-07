"""评测 RS-Diffuser，并在推理时切换风险偏好（无需重新训练）。

    # 默认：风险规避，CVaR_0.1
    python scripts/evaluate.py --run runs/halfcheetah-medium-v2/seed0

    # 同一个模型，换成风险中性 / 风险偏好 / VaR
    python scripts/evaluate.py --run runs/halfcheetah-medium-v2/seed0 --risk mean
    python scripts/evaluate.py --run runs/halfcheetah-medium-v2/seed0 --risk cvar_upper --alpha 0.1
    python scripts/evaluate.py --run runs/halfcheetah-medium-v2/seed0 --risk var --alpha 0.1

    # 汇总多个随机种子（论文：100 个 episode × 5 个种子，报告 mean ± std）
    python scripts/evaluate.py --run runs/halfcheetah-medium-v2/seed{0,1,2,3,4}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rs_diffuser.agent import load_agent
from rs_diffuser.evaluation import run_episodes, summarize
from rs_diffuser.tasks import get_task, make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="+", required=True, help="一个或多个训练输出目录（每个对应一个随机种子）")
    p.add_argument("--risk", default="cvar_lower", choices=["cvar_lower", "cvar_upper", "mean", "var"])
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--w_cg", type=float, default=None, help="风险引导强度，不指定则用训练时的配置")
    p.add_argument("--num_candidates", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--parallel", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    overrides = {k: v for k, v in (("w_cg", args.w_cg), ("num_candidates", args.num_candidates),
                                   ("sample_steps", args.sample_steps)) if v is not None}
    per_seed = []
    for seed, run in enumerate(args.run):
        agent = load_agent(run, device=args.device, overrides=overrides)
        agent.set_risk(args.risk, args.alpha)
        task = get_task(agent.meta.task)
        result = run_episodes(lambda s: make_env(task.name, seed=s), agent.act, n_episodes=args.episodes,
                              n_parallel=args.parallel, max_steps=task.max_episode_steps, seed=seed)
        summary = summarize(result["returns"], result["violations"])
        per_seed.append(summary)
        tag = f"{args.risk}" if args.risk == "mean" else f"{args.risk}_{args.alpha:g}"
        Path(run, f"eval_{tag}.json").write_text(json.dumps({**summary, **result}, indent=2))
        print(f"[{run}] {agent.critic.risk} | " + " ".join(f"{k}={v:.2f}" for k, v in summary.items() if k != "episodes"))

    if len(per_seed) > 1:
        print(f"\n{len(per_seed)} 个种子汇总（mean ± std）:")
        for k in ("mean", "median", "cvar_0.1", "violations"):
            vals = np.array([s[k] for s in per_seed])
            print(f"  {k:<10} {vals.mean():9.2f} ± {vals.std():.2f}")


if __name__ == "__main__":
    main()
