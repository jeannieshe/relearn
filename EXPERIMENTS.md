# Experiments log

The "why" behind runs — hypothesis, result, interpretation, next step. See
README.md's "The config system" section for how to actually run things with
Hydra, and "Config system vs. EXPERIMENTS.md" for how this file relates to
`configs/agent/`, `configs/env/`, and wandb.

Each entry names the exact Hydra invocation (e.g. `agent=eps_fix env=sw480`)
and/or the wandb run id — that's the join key back to the full resolved
config, whether you're reading this file or looking at a wandb run page.

## 2026-07-15 — oracle sweep shows single-drug reward ceiling is far below termination

Run: `env=sw480` (no training run — static sweep via
`src/relearn/enumerate_perturbations.py`, which uses `EnvConfig()` defaults directly)
| Data: `experiments/perturbation_ranking.csv`

Hypothesis: before tuning the DQN further, check whether the apoptosis
reward landscape has enough structure/range for single-drug, single-step
actions to ever hit `termination_epsilon=0.1` (score >= 0.9).

Result: scored all 1138 single drugs once from the fixed SW480 DMSO-neutral
state. Max score 0.661 (Bicalutamide), mean 0.617, stdev 0.018. Zero drugs
terminate. The 2026-07-14 DQN run (wandb `kny202d4`) topped out at 0.66 too —
matches the oracle ceiling almost exactly.

Next: don't scale up single-drug DQN training — the ceiling is an
environment/reward-design limit, not a training problem. Test whether
multi-step/combination dosing (feeding a drug's resulting state back in as
`ctrl_cell_emb` for a second drug) breaks past 0.66 before investing further.
See memory `reward-landscape-flat-single-drug` for full detail.

## 2026-07-16 — num_episodes in A001 is too low to explore all action space

Run: `python src/relearn/agents/dqn.py experiment=A run_id=A019 description='"one_drug, horizon=1, HVG, ST, episodes=2500"'` | wandb: `fpvl6wny`

Hypothesis: kny202d4 (300 episodes) does not explore all the action space even in the multi-armed bandit setting, since there are a total of 1138 chemical perturbations in the environment 'small_molecules.py.' Now I've updated the 'baseline.yaml' config in agent/ to be specifically num_episodes=2500 and eps_decay=150 so there is a chance for full exploration and a little exploitation.

Result: The model explored a lot until around episode 500, then did not really learn to optimize the reward even though it had around 2000 more steps to do that. Maybe I should not have that steep of epsilon decay.

Next: Want to fix epsilon decay to be around 500.
