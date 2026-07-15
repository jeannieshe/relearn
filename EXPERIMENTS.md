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

## 2026-07-15 — eps-decay/step-budget mismatch in kny202d4

Run: `agent=eps_fix env=sw480` (fix, not yet run) | wandb: `kny202d4` (the run being diagnosed, predates the config system)

Hypothesis: kny202d4 (300 episodes) showed flat episode_reward from episode 0
(mean 0.616) to episode 299 (mean 0.617) — investigate whether the DQN ever
got a chance to learn anything, independent of the reward-ceiling issue above.

Result: `EPS_DECAY=2500` assumes thousands of steps, but the env truncates
every episode after 1 step, so `num_episodes` IS the total step count.
Epsilon only decayed to ~0.80 by the final step — the agent was ~80%
random-acting for the entire run and never got to meaningfully exploit a
learned policy.

Next: re-run with `agent=eps_fix` (`eps_decay: 150`, `num_episodes: 2000`)
once the environment supports a horizon worth exploiting (see the sweep
entry above) — no point re-running this fix against a 1-step env whose
ceiling is already known.
