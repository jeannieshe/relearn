"""
First-drug x second-drug grid sweep for one-two-punch candidate discovery.

Batching warning: verified empirically (2026-07-24) that STATE's transformer
attends across the S dimension of a batch -- batching multiple different
(ctrl_state, pert) pairs together in one forward call produces DIFFERENT
results than running them individually (max abs diff ~1.5-2.4 in log1p
expression space, not numerical noise). So every (first, second) pair here is
scored via two sequential S=1 forward calls, exactly matching what
env.step() does -- no batching shortcut, and no cross-candidate leakage.

Scope: NOT the full 1138x1138 grid (1.3M S=1 calls, ~9h at ~25ms/call observed
elsewhere in this codebase). This sweeps the top N_FIRST first-drug candidates
(by single-agent baseline score, from artifacts/perturbation_ranking.csv)
against all 1138 second-drug candidates -- a tractable first pass, not a claim
that these N_FIRST are the best possible primers. Extend N_FIRST (or flip to a
full 1138x1138 run as a genuine multi-hour background job) once this first
pass is validated. See memory one-two-punch-reward-design.md.

For each pair, reports `beats_best_single_agent` = score_pair - max(single-agent
score of first drug, single-agent score of second drug) -- the synergy metric
borrowed from the SequenTx recon (see memory sequentx-prior-work.md) as an
interpretable ranking criterion for eventual candidate prioritization.
"""

import csv
import time
from pathlib import Path

from relearn.config import EnvConfig
from relearn.envs.small_molecules import RelearnChemicalEnv

REPO_ROOT = Path(__file__).parent.parent.parent.parent
BASELINE_PATH = REPO_ROOT / "artifacts" / "perturbation_ranking.csv"
OUT_PATH = REPO_ROOT / "artifacts" / "one_two_punch_grid.csv"

N_FIRST = 100  # number of first-drug ("prime") candidates to sweep -- see scope note above


def load_baseline_scores(path: Path) -> dict:
    with open(path) as f:
        return {row["drug"]: float(row["reward"]) for row in csv.DictReader(f)}


def run_grid(env: RelearnChemicalEnv, baseline_scores: dict, n_first: int) -> list:
    drug_str_to_action = {str(d): i for i, d in enumerate(env.drug_list)}
    ranked_drugs = sorted(baseline_scores.items(), key=lambda kv: kv[1], reverse=True)
    first_candidates = [drug_str_to_action[d] for d, _ in ranked_drugs if d in drug_str_to_action][:n_first]

    results = []
    t0 = time.time()
    for i, first_action in enumerate(first_candidates):
        env.reset()
        state_after_first, _, _, _, info_first = env.step(first_action)
        score_first_alone = info_first["score"]
        first_drug = env.drug_list[first_action]
        first_baseline = baseline_scores.get(str(first_drug), float(score_first_alone))

        for second_action in range(env.action_space.n):
            next_state = env._transition_model.step(state_after_first, second_action)
            score_pair = float(env._score(next_state))

            second_drug = env.drug_list[second_action]
            second_baseline = baseline_scores.get(str(second_drug))
            best_single_agent = (
                max(first_baseline, second_baseline) if second_baseline is not None else first_baseline
            )
            beats_best_single_agent = score_pair - best_single_agent

            results.append({
                "first_action": first_action,
                "first_drug": first_drug,
                "second_action": second_action,
                "second_drug": second_drug,
                "score_first_alone": float(score_first_alone),
                "score_pair": score_pair,
                "best_single_agent": best_single_agent,
                "beats_best_single_agent": beats_best_single_agent,
            })

        elapsed = time.time() - t0
        print(f"[{i + 1}/{len(first_candidates)}] first={first_drug} done ({elapsed:.1f}s elapsed)", flush=True)

    results.sort(key=lambda r: r["beats_best_single_agent"], reverse=True)
    return results


if __name__ == "__main__":
    baseline_scores = load_baseline_scores(BASELINE_PATH)
    env = RelearnChemicalEnv(EnvConfig(horizon=2))
    results = run_grid(env, baseline_scores, N_FIRST)

    fieldnames = [
        "first_action", "first_drug", "second_action", "second_drug",
        "score_first_alone", "score_pair", "best_single_agent", "beats_best_single_agent",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} pairs to {OUT_PATH}")
    print(
        f"Scope: top {N_FIRST} first-drug candidates (by single-agent score) x all "
        f"{env.action_space.n} second-drug candidates -- NOT the full "
        f"{env.action_space.n}x{env.action_space.n} grid."
    )
    print("\nTop 10 by beats_best_single_agent:")
    for r in results[:10]:
        print(
            f"  {r['first_drug']} -> {r['second_drug']}: pair={r['score_pair']:.4f} "
            f"best_single={r['best_single_agent']:.4f} beats={r['beats_best_single_agent']:+.4f}"
        )
