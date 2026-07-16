"""
v1: score every one of Tahoe's 1138 perturbations once against the fixed
SW480 DMSO-neutral starting state, and rank by apoptosis score.

STATE's forward pass is deterministic at eval() (dropout disabled) and
env.reset() always returns the same starting state, so there is no reward
uncertainty to manage here. Every arm's true value is observable in a single
query, which means this reduces to an exhaustive sweep rather than a
multi-armed bandit -- there's nothing left for UCB/Thompson sampling to buy
you when you can just ask every arm directly.
"""

import csv
from pathlib import Path

from relearn.envs.small_molecules import RelearnChemicalEnv


def rank_all_perturbations(env: RelearnChemicalEnv):
    results = []
    for action in range(env.action_space.n):
        env.reset()
        _, reward, terminated, _, _ = env.step(action)
        results.append({
            "action": action,
            "drug": env.drug_list[action],
            "reward": reward,
            "terminated": terminated,
        })
    results.sort(key=lambda r: r["reward"], reverse=True)
    return results


if __name__ == "__main__":
    env = RelearnChemicalEnv()
    results = rank_all_perturbations(env)

    out_path = Path(__file__).parent.parent.parent / "experiments" / "perturbation_ranking.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "drug", "reward", "terminated"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} ranked perturbations to {out_path}")
    print("\nTop 10:")
    for r in results[:10]:
        print(f"  {r['drug']}: score={r['reward']:.4f} terminated={r['terminated']}")
