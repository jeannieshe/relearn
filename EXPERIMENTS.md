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

## 2026-07-25 — two-step sweeps are uninterpretable: the mean-DMSO basal is off-distribution

Run: `python src/relearn/experiments/order_additivity.py --drug-a palbociclib --drug-b venetoclax --dose 0.5`
| Data: `artifacts/order_additivity_*_palbociclib_venetoclax_0.5uM.*`

Hypothesis: measure geometry, not reward. Compare displacement vectors
`v = expr(after) - expr(baseline)` in 2000-HVG space across forward (A→B),
reverse (B→A), and simultaneous co-dose, against the additive null `v_A + v_B`.
Wash-out arms that should be no-ops (`A→DMSO` vs `A`) give a drift floor.

Result: gate check failed. Drift floor was `rel_resid = 1.02` — a no-op DMSO
wash-out moved the state as much as the drug did. The order effect (0.81) came in
below that floor, so nothing was measurable. Cause is the starting state: DMSO
applied to the cached DMSO-neutral state moves it by ‖v‖ = 4.54 and raises UCell
0.5585 → 0.6229, so the vehicle control outscores palbociclib (0.6138).
`_load_dmso_neutral_state` averages a control population, but the mean of sparse
count vectors is a dense vector unlike any real cell. This also explains
2026-07-15: the ~0.62 every drug converges to is a generic "went through STATE
once" value, and the 0.018 stdev was the entire drug-specific signal.

Note for older entries: `env.step()` now returns shaped `delta`, not the absolute
score, but `enumerate_perturbations.py` still writes it as `reward` and both DMSO
sweeps read it back as absolute. Ranking order is unaffected, values are not.
`dmso_first_sweep.py` also calls `env._state_stepper_helper`, which no longer
exists — AttributeError on rerun.

Next: replace the synthetic basal with Tahoe's real measured cells.

## 2026-07-25 — STATE is basal-invariant: sequential dosing is not expressible

Run: `python src/relearn/experiments/real_basal_order.py --n-draws 20` (and `--sentence-len 1`)
| Data: `artifacts/real_basal_*_palbociclib_venetoclax_0.5uM_S{256,1}.*`

Hypothesis: the drift above is a plumbing artifact. Ground step 1 in measurement —
use Tahoe's real palbociclib-treated SW480 cells (`c22.h5ad`, 6012 @ 0.5 µM) as
basal for drug B. Also lets us run at the trained `cell_sentence_len=256` instead
of S=1, and gives a real noise floor: two disjoint 256-cell draws from one pool.

Result: plumbing fixes worked and exposed the real problem. Fixed — the fixed
point holds (`ctrl→DMSO` vs real DMSO = 0.78× floor), and pass-1 accuracy is now
visible and good (`ctrl→A` vs real A cells, cos 0.877 / 0.80× floor): STATE's
single-drug predictions sit within measurement noise of real cells.

Not fixable by plumbing — the model ignores the basal:

    realA→B vs ctrl→B    cos 0.967   0.33× floor
    realB→A vs ctrl→A    cos 0.976   0.33× floor
    realA→B vs real_B    cos 0.835   0.79× floor   (vs ctrl→B vs real_B at 0.77×)

Predicting B from real A-treated cells equals predicting B from untreated
controls — closer than two halves of one real population. "A then B" predicts
real B-alone cells as well as "B alone" does. `realA→DMSO` collapses to ‖v‖ = 0.33,
on top of `ctrl→DMSO` (0.30), while real A cells sit at 1.58. The apparent order
effect (cos 0.387) is just cos(real_A, real_B) = 0.327.

Robust to metric choice: per-gene standardization over the 795 genes detected in
>1% of DMSO cells leaves basal-invariance at 0.92–0.96, pass-1 at 0.80–0.85. The
S=1 rerun is uninformative, not contradictory — single cells are 96% zeros, the
floor jumps to 1.44, everything falls under it.

Scope mismatch, not a broken checkpoint: STATE is trained as a control → perturbed
map and `state tx infer` only samples basal from the DMSO control pool, so it
never had reason to learn basal-dependence.

Next: multi-step RL has no substrate — `env.step()` cannot compose, so horizon > 1
is meaningless regardless of agent tuning. Either (a) single-step with
combinations as one blended perturbation vector (`co_mean` is architecturally
coherent, but `pert_encoder` is a single Linear so it only interpolates the two
embeddings, and Tahoe has no combination data to validate against), or (b) a model
trained on sequential/combination data. Caveat: one drug pair — see the 30-pair
sweep below, which generalizes it.

## 2026-07-25 — basal-invariance holds for 30/30 random drug pairs

Run: `python src/relearn/experiments/basal_invariance_sweep.py --n-pairs 30 --n-draws 5`
| Data: `artifacts/basal_invariance_sweep_30pairs_0.5uM_S256.csv`

Hypothesis: the entry above rests on palbociclib/venetoclax alone. Run the reduced
test — `realA→B` vs `ctrl→B`, against the DMSO split-half floor — over 30 random
pairs drawn from the 374 drugs with ≥600 real SW480 cells at 0.5 µM. A spread with
some pairs clearing the floor would mean the basal sometimes carries through.

Result: no spread. `cos(realA→B, ctrl→B)` median 0.971, IQR [0.964, 0.976], range
0.931–0.991; as a multiple of the noise floor, median 0.31× and max 0.35× — never
close to 1.0. 30/30 pairs indistinguishable from "B alone". The direct form agrees:
`cos(realA→B, real_B)` median 0.828 vs `cos(ctrl→B, real_B)` median 0.830, i.e. the
prior treatment changes prediction quality by nothing. Basal-invariance is
architectural, not a quirk of one pair. Runtime 2m38s.

Side observation: pass-1 accuracy varies a lot by drug — `cos(ctrl→B, real_B)`
spans 0.657–0.945. Worth knowing which drugs STATE predicts well before relying
on any single-agent ranking.

Next: the sequential framing is closed. Decide between single-step-with-blended-
perturbations, or sourcing combination data and a model trained on it.
