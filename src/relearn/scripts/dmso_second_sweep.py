"""
Deterministic sweep: every one of the 1138 drugs tried first, DMSO fixed as
the second, wash-out dose. Answers "if you always follow up with a DMSO
wash-out, which first drug gives the best final apoptosis score?" -- the
mirror image of dmso_first_sweep.py (DMSO first, drug second).

Unlike dmso_first_sweep.py, the free slot here is the *first* step, so there
is no shortcut -- every one of the 1138 candidates needs its own forward pass
through both steps. That sweep was already run for a different question
(self-consistency: does f(drug, DMSO) ~ f(drug) alone?) in
self_consistency_check.py, whose output CSV already contains score_A_then_dmso
for every drug. Re-rank that existing data by final score instead of paying
for the ~90s of forward passes twice.
"""

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
SELF_CONSISTENCY_PATH = REPO_ROOT / "experiments" / "self_consistency_ranking.csv"
BASELINE_PATH = REPO_ROOT / "experiments" / "perturbation_ranking.csv"
OUT_PATH = REPO_ROOT / "experiments" / "dmso_second_ranking.csv"


def load_baseline_scores(path: Path) -> dict:
    with open(path) as f:
        return {row["drug"]: float(row["reward"]) for row in csv.DictReader(f)}


def load_dmso_second_results(path: Path, baseline_scores: dict) -> list:
    with open(path) as f:
        rows = list(csv.DictReader(f))

    results = []
    for r in rows:
        drug = r["drug"]
        score_drug_then_dmso = float(r["score_A_then_dmso"])
        baseline = baseline_scores.get(drug)
        results.append({
            "action": int(r["action"]),
            "drug": drug,
            "score_drug_then_dmso": score_drug_then_dmso,
            "score_drug_alone": baseline,
            "score_delta": None if baseline is None else score_drug_then_dmso - baseline,
        })
    results.sort(key=lambda r: r["score_drug_then_dmso"], reverse=True)
    return results


if __name__ == "__main__":
    baseline_scores = load_baseline_scores(BASELINE_PATH)
    results = load_dmso_second_results(SELF_CONSISTENCY_PATH, baseline_scores)

    fieldnames = ["action", "drug", "score_drug_then_dmso", "score_drug_alone", "score_delta"]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    scores = [r["score_drug_then_dmso"] for r in results]
    baseline_ranked = sorted(baseline_scores.items(), key=lambda kv: kv[1], reverse=True)
    top10_baseline_drugs = {d for d, _ in baseline_ranked[:10]}
    top10_dmso_second_drugs = {r["drug"] for r in results[:10]}
    overlap = len(top10_baseline_drugs & top10_dmso_second_drugs)

    print(f"Wrote {len(results)} results to {OUT_PATH}")
    print(f"score_drug_then_dmso: mean={np.mean(scores):.4f} min={np.min(scores):.4f} max={np.max(scores):.4f}")
    print(f"top-10 overlap with no-washout ranking: {overlap}/10 drugs in common")

    print("\nTop 10 (best first drug, DMSO wash-out second):")
    for r in results[:10]:
        delta = f"{r['score_delta']:+.4f}" if r["score_delta"] is not None else "n/a"
        print(f"  {r['drug']}: score={r['score_drug_then_dmso']:.4f} (vs. alone: {delta})")
