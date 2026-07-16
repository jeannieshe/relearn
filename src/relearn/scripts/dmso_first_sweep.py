"""
Deterministic sweep: DMSO fixed as the first dose, then every one of the
1138 drugs tried as the second, unconstrained dose. Answers "does
pre-treating with the vehicle control change which drug looks best?"

This is a single free step (the second dose) with the first held fixed, so
it's fully enumerable in one pass -- no RL needed. STATE's forward pass is
deterministic and env.reset() always returns the same starting state, so the
state after the fixed DMSO dose is one exact, reproducible value; every
second-dose "arm" has an exact, directly-queryable score. See
enumerate_perturbations.py's docstring for the same reasoning applied to the
single-step case, and EXPERIMENTS.md for why epsilon-greedy DQN training
isn't the right tool here.

Compares against experiments/perturbation_ranking.csv (no DMSO pretreatment)
to see whether pretreatment shifts the ranking, not just the scores.
"""

import csv
from pathlib import Path

import numpy as np

from relearn.config import EnvConfig
from relearn.envs.small_molecules import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent
BASELINE_PATH = REPO_ROOT / "experiments" / "perturbation_ranking.csv"
OUT_PATH = REPO_ROOT / "experiments" / "dmso_first_ranking.csv"


def load_baseline_scores(path: Path) -> dict:
    with open(path) as f:
        return {row["drug"]: float(row["reward"]) for row in csv.DictReader(f)}


def run_dmso_first_sweep(env: RelearnChemicalEnv, baseline_scores: dict) -> list:
    dmso_action = env.drug_list.index(env.cfg.dmso_control_pert)

    # step 1 is fixed and identical every time -- compute it exactly once
    env.reset()
    post_dmso_state, post_dmso_score, _, _, _ = env.step(dmso_action)

    results = []
    for action in range(env.action_space.n):
        # bypass env.step()'s internal state mutation/step-counter -- we're
        # sweeping the *same* fixed starting state 1138 times, not advancing
        # an episode, so call the low-level transition directly
        next_state = env._state_stepper_helper(post_dmso_state, action)
        score = env.apoptosis_predictor(next_state, gene_names=env.hvg_gene_names, signature_genes=env.sig_genes)

        drug = env.drug_list[action]
        baseline = baseline_scores.get(str(drug))
        results.append({
            "action": action,
            "drug": drug,
            "score_dmso_then_drug": float(score),
            "score_drug_alone": baseline,
            "score_delta": None if baseline is None else float(score) - baseline,
        })
    results.sort(key=lambda r: r["score_dmso_then_drug"], reverse=True)
    return results, post_dmso_score


if __name__ == "__main__":
    baseline_scores = load_baseline_scores(BASELINE_PATH)

    env = RelearnChemicalEnv(EnvConfig(horizon=2))
    results, post_dmso_score = run_dmso_first_sweep(env, baseline_scores)

    fieldnames = ["action", "drug", "score_dmso_then_drug", "score_drug_alone", "score_delta"]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    scores = [r["score_dmso_then_drug"] for r in results]
    baseline_ranked = sorted(baseline_scores.items(), key=lambda kv: kv[1], reverse=True)
    top10_baseline_drugs = {d for d, _ in baseline_ranked[:10]}
    top10_dmso_first_drugs = {r["drug"] for r in results[:10]}
    overlap = len(top10_baseline_drugs & top10_dmso_first_drugs)

    print(f"Wrote {len(results)} results to {OUT_PATH}")
    print(f"score after DMSO alone (step 1): {post_dmso_score:.4f}")
    print(f"score_dmso_then_drug: mean={np.mean(scores):.4f} min={np.min(scores):.4f} max={np.max(scores):.4f}")
    print(f"top-10 overlap with no-pretreatment ranking: {overlap}/10 drugs in common")

    print("\nTop 10 (DMSO pretreatment, then best second drug):")
    for r in results[:10]:
        delta = f"{r['score_delta']:+.4f}" if r["score_delta"] is not None else "n/a"
        print(f"  {r['drug']}: score={r['score_dmso_then_drug']:.4f} (vs. alone: {delta})")
