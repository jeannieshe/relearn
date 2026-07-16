"""
Deterministic sweep: every one of the 1138 drugs tried first, DMSO fixed as
the second, wash-out dose. One sweep, two questions:

  1. Self-consistency: does f(drug, DMSO) ~ f(drug) alone? (cosine_sim, l2_dist
     between the state after the drug alone and after drug+DMSO)
  2. Best final score: if you always follow up with a DMSO wash-out, which
     first drug gives the best final apoptosis score? (score_delta vs. the
     no-washout baseline) -- the mirror image of dmso_first_sweep.py (DMSO
     first, drug second).

The free slot here is the *first* step, so unlike dmso_first_sweep.py there's
no shortcut -- every one of the 1138 candidates needs its own forward pass
through both steps (~90s).
"""

import csv
from pathlib import Path

import numpy as np

from relearn.config import EnvConfig
from relearn.envs.small_molecules import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent.parent
BASELINE_PATH = REPO_ROOT / "experiments" / "perturbation_ranking.csv"
OUT_PATH = REPO_ROOT / "experiments" / "dmso_second_ranking.csv"


def load_baseline_scores(path: Path) -> dict:
    with open(path) as f:
        return {row["drug"]: float(row["reward"]) for row in csv.DictReader(f)}


def run_dmso_second_sweep(env: RelearnChemicalEnv, baseline_scores: dict) -> list:
    dmso_action = env.drug_list.index(env.cfg.dmso_control_pert)
    results = []
    for action in range(env.action_space.n):
        env.reset()
        obs_drug, score_drug_alone, _, _, _ = env.step(action)
        obs_drug_dmso, score_drug_then_dmso, _, _, _ = env.step(dmso_action)

        cosine_sim = float(
            np.dot(obs_drug, obs_drug_dmso) / (np.linalg.norm(obs_drug) * np.linalg.norm(obs_drug_dmso))
        )
        l2_dist = float(np.linalg.norm(obs_drug - obs_drug_dmso))

        drug = env.drug_list[action]
        baseline = baseline_scores.get(str(drug))
        results.append({
            "action": action,
            "drug": drug,
            "score_drug_alone": score_drug_alone,
            "score_drug_then_dmso": score_drug_then_dmso,
            "score_delta": score_drug_then_dmso - score_drug_alone,
            "cosine_sim": cosine_sim,
            "l2_dist": l2_dist,
            "score_drug_alone_baseline_mismatch": None if baseline is None else abs(baseline - score_drug_alone),
        })
    results.sort(key=lambda r: r["score_drug_then_dmso"], reverse=True)
    return results


if __name__ == "__main__":
    baseline_scores = load_baseline_scores(BASELINE_PATH)

    env = RelearnChemicalEnv(EnvConfig(horizon=2))
    results = run_dmso_second_sweep(env, baseline_scores)

    fieldnames = [
        "action", "drug", "score_drug_alone", "score_drug_then_dmso", "score_delta",
        "cosine_sim", "l2_dist", "score_drug_alone_baseline_mismatch",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    scores = [r["score_drug_then_dmso"] for r in results]
    cosine_sims = [r["cosine_sim"] for r in results]
    l2_dists = [r["l2_dist"] for r in results]
    mismatches = [r["score_drug_alone_baseline_mismatch"] for r in results if r["score_drug_alone_baseline_mismatch"] is not None]

    baseline_ranked = sorted(baseline_scores.items(), key=lambda kv: kv[1], reverse=True)
    top10_baseline_drugs = {d for d, _ in baseline_ranked[:10]}
    top10_dmso_second_drugs = {r["drug"] for r in results[:10]}
    overlap = len(top10_baseline_drugs & top10_dmso_second_drugs)

    print(f"Wrote {len(results)} results to {OUT_PATH}")
    print(f"score_drug_then_dmso: mean={np.mean(scores):.4f} min={np.min(scores):.4f} max={np.max(scores):.4f}")
    print(f"cosine_sim (self-consistency): mean={np.mean(cosine_sims):.4f} min={np.min(cosine_sims):.4f} max={np.max(cosine_sims):.4f}")
    print(f"l2_dist:    mean={np.mean(l2_dists):.4f}")
    if mismatches:
        print(f"sanity check -- max |score_drug_alone - baseline.csv| mismatch: {max(mismatches):.6f} (should be ~0)")
    print(f"top-10 overlap with no-washout ranking: {overlap}/10 drugs in common")

    print("\nTop 10 (best first drug, DMSO wash-out second):")
    for r in results[:10]:
        print(f"  {r['drug']}: score={r['score_drug_then_dmso']:.4f} (vs. alone: {r['score_delta']:+.4f}) cosine={r['cosine_sim']:.4f}")

    print("\nWorst self-consistency (lowest cosine similarity vs. drug alone):")
    for r in sorted(results, key=lambda r: r["cosine_sim"])[:10]:
        print(f"  {r['drug']}: cosine={r['cosine_sim']:.4f} l2={r['l2_dist']:.4f}")
