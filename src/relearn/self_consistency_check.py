"""
Self-consistency check: does f(output_A, DMSO) ~ output_A?

A null second dose (the DMSO_TF vehicle control) should leave the cell state
roughly where drug A alone left it. Compares against the existing single-step
sweep in experiments/perturbation_ranking.csv (env=sw480, horizon=1) rather
than recomputing drug A's effect from scratch -- step 1 here is identical
regardless of horizon, so score_A should match that file almost exactly
(reported as score_A_baseline_mismatch, a sanity check on the comparison
itself, not a real experimental finding).
"""

import csv
from pathlib import Path

import numpy as np

from relearn.config import EnvConfig
from relearn.envs.single_step import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent
BASELINE_PATH = REPO_ROOT / "experiments" / "perturbation_ranking.csv"
OUT_PATH = REPO_ROOT / "experiments" / "self_consistency_ranking.csv"


def load_baseline_scores(path: Path) -> dict:
    with open(path) as f:
        return {row["drug"]: float(row["reward"]) for row in csv.DictReader(f)}


def run_self_consistency_sweep(env: RelearnChemicalEnv, baseline_scores: dict) -> list:
    dmso_action = env.drug_list.index(env.cfg.dmso_control_pert)
    results = []
    for action in range(env.action_space.n):
        env.reset()
        obs_a, score_a, _, _, _ = env.step(action)
        obs_a_dmso, score_a_dmso, _, _, _ = env.step(dmso_action)

        cosine_sim = float(
            np.dot(obs_a, obs_a_dmso) / (np.linalg.norm(obs_a) * np.linalg.norm(obs_a_dmso))
        )
        l2_dist = float(np.linalg.norm(obs_a - obs_a_dmso))

        drug = env.drug_list[action]
        baseline = baseline_scores.get(str(drug))
        results.append({
            "action": action,
            "drug": drug,
            "score_A": score_a,
            "score_A_then_dmso": score_a_dmso,
            "score_delta": score_a_dmso - score_a,
            "cosine_sim": cosine_sim,
            "l2_dist": l2_dist,
            "score_A_baseline_mismatch": None if baseline is None else abs(baseline - score_a),
        })
    results.sort(key=lambda r: r["cosine_sim"])
    return results


if __name__ == "__main__":
    baseline_scores = load_baseline_scores(BASELINE_PATH)

    env = RelearnChemicalEnv(EnvConfig(horizon=2))
    results = run_self_consistency_sweep(env, baseline_scores)

    fieldnames = [
        "action", "drug", "score_A", "score_A_then_dmso", "score_delta",
        "cosine_sim", "l2_dist", "score_A_baseline_mismatch",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    cosine_sims = [r["cosine_sim"] for r in results]
    l2_dists = [r["l2_dist"] for r in results]
    mismatches = [r["score_A_baseline_mismatch"] for r in results if r["score_A_baseline_mismatch"] is not None]

    print(f"Wrote {len(results)} results to {OUT_PATH}")
    print(f"cosine_sim: mean={np.mean(cosine_sims):.4f} min={np.min(cosine_sims):.4f} max={np.max(cosine_sims):.4f}")
    print(f"l2_dist:    mean={np.mean(l2_dists):.4f}")
    if mismatches:
        print(f"sanity check -- max |score_A - baseline.csv| mismatch: {max(mismatches):.6f} (should be ~0)")

    print("\nWorst self-consistency (lowest cosine similarity vs. drug A alone):")
    for r in results[:10]:
        print(f"  {r['drug']}: cosine={r['cosine_sim']:.4f} l2={r['l2_dist']:.4f}")

    print("\nBest self-consistency (highest cosine similarity vs. drug A alone):")
    for r in results[-10:][::-1]:
        print(f"  {r['drug']}: cosine={r['cosine_sim']:.4f} l2={r['l2_dist']:.4f}")
