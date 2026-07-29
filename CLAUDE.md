# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An RL agent that uses the STATE virtual cell model as a simulated environment to discover
drug perturbations (and drug sequences/combinations) that drive SW480 cancer cells toward
apoptosis. STATE predicts the next cell state given a current state + a drug perturbation;
the RL agent (DQN) learns a policy over that simulated transition function, with reward
shaped by an UCell apoptosis-signature score.

## Commands

Get the STATE checkpoint (not committed to git, ~1 GB) before the first run:

```bash
python scripts/download_ST-HVG-Tahoe.py   # for env=sw480 (default, raw 2000-HVG panel)
python scripts/download_ST-SE-Tahoe.py    # for env=sw480_se (2058-dim SE-600M embedding)
```

Run training (Hydra entrypoint, requires `hydra-core`):

```bash
python src/relearn/agents/dqn.py
python src/relearn/agents/dqn.py agent=eps_fix env=sw480
python src/relearn/agents/dqn.py agent=eps_fix env=sw480 agent.lr=0.001
python src/relearn/agents/dqn.py experiment=B run_id=B003 description="testing eps_fix at 2-step horizon" agent=eps_fix env=sw480_2step
```

Always set `experiment` / `run_id` / `description` on a real run — they're the wandb
group/name/notes AND name `outputs/<experiment>/<run_id>/` on disk, so pick the next
unused ID the way you'd claim a row in a shared spreadsheet (two people run experiments
against this repo).

Useful Hydra tricks:
- `python src/relearn/agents/dqn.py --cfg job` — print the fully resolved config without running.
- `python src/relearn/agents/dqn.py --help` — list every overridable field.
- Group names have no `.yaml`: `agent=eps_fix`, not `agent=eps_fix.yaml`.
- A typo in an override fails loudly (`Key 'eps_decy' not in 'DQNConfig'`) — that's schema
  validation working, not a bug to route around.

There is no formal test suite or linter configured at the repo root; one-off analyses live
as scripts under `src/relearn/experiments/` (run directly with `python`, not via Hydra/pytest).

## Architecture

**`src/relearn/transitions/`** — the state-transition function is pluggable behind the
`TransitionModel` protocol (`transitions/base.py`): `step(cell_state, action) -> next_state`,
plus a `drug_list` of `(name, concentration, units)` tuples that `action` indexes into.
`build_transition_model(cfg)` (`transitions/__init__.py`) picks the implementation from
`cfg.transition_model` ("state" → `StateTransitionModel`, "rhaister" →
`RhaisterTransitionModel`). `StateTransitionModel` wraps STATE's
`StateTransitionPerturbationModel`, whose `forward(batch, padded=False)` accepts a variable-length
*set* ("cell sentence") of `S` cells that self-attend over each other — `S` is capped by the
checkpoint's trained `cell_set_len` (its GPT2 backbone has learned positional embeddings only
up to that length; the default fewshot checkpoint under `data/models/ST-HVG-Tahoe` was trained
with `cell_set_len=256`). `step_with_pert_vector` is the escape hatch for feeding an arbitrary
(non-one-hot) perturbation vector instead of a `drug_list` index, e.g. for combination/co-perturbation
experiments.

**`src/relearn/envs/small_molecules.py`** — `RelearnChemicalEnv(gym.Env)`. Observation/state is
a cell embedding in whichever representation `cfg.embed_key` names (`"X_hvg"`: 2000-dim raw HVGs,
directly scorable; `"X_state"`: 2058-dim SE-600M latent, must be decoded back to the 2000-HVG panel
via the transition model's `gene_decoder` before scoring — see `_to_gene_expression`). The episode
starts from real DMSO control cells for `cfg.cell_type_name`, read from the Tahoe-100M STATE-preprocessed
h5ad files under `cfg.tahoe_se_dir` and cached to `.npy` under `cfg.tahoe_dataset_dir` on first use
(cache keyed by `embed_key` so HVG/SE runs never collide). Reward is potential-based shaping —
`reward_t = apoptosis_score(s_t) - apoptosis_score(s_{t-1})` via UCell scoring
(`utils.ucell_score`) against `cfg.msigdb_gene_set` (default `HALLMARK_APOPTOSIS`) — which is
policy-invariant and gives a gradient across an episode instead of a single terminal 0/1 signal.
Termination is `|1 - score| <= termination_epsilon`; truncation is `cfg.horizon` steps.
`envs/` and `agents/` are packages (not single files) because more than one implementation is
expected on each side (e.g. a multi-step/combination env, a non-DQN agent) — add new files named
for what they *are* (`envs/multi_step_combo.py`), not a version number.

**`src/relearn/config.py`** — `DQNConfig`/`EnvConfig` dataclasses are the single source of truth
for every tunable; both are registered with Hydra's `ConfigStore`. `configs/config.yaml` composes
one named file from `configs/agent/*.yaml` and `configs/env/*.yaml` (each only states the fields
it overrides, falling back to the dataclass defaults) plus top-level `experiment`/`run_id`/`description`
bookkeeping fields. Hydra validates every override against the dataclass field types, so typos and
the classic PyYAML bare-scientific-notation gotcha (`3e-4` parsing as a string) fail immediately.

**Config system vs. `EXPERIMENTS.md`** — these are deliberately two different records and neither
replaces the other. Hydra + wandb capture *what* ran (exact resolved config, metrics, git commit) —
mechanical and complete but silent on *why*. `EXPERIMENTS.md` is the hypothesis/result/interpretation
layer, keyed by `run_id` so either record can be traced to the other. When starting a new experiment
family, add a config file under `configs/agent/` or `configs/env/` *and* an `EXPERIMENTS.md` entry.

**`src/relearn/experiments/`** — standalone analysis scripts (drug-order/additivity checks, DMSO
self-consistency sweeps, HVG/apoptosis gene-panel overlap, etc.), run directly rather than through
the Hydra/DQN training entrypoint.

## Libraries

Prefer [pertpy](https://pertpy.readthedocs.io/en/latest/) — specifically
`pertpy.tools.Distance` ([distances tutorial](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/distances.html))
— over hand-rolled distribution-comparison metrics when comparing sets/populations of cell
states (e.g. real vs. predicted, control vs. perturbed). It implements e-distance, Wasserstein,
MMD, mean-pairwise, and pseudobulk-Euclidean distances with a consistent `pairwise()` API,
instead of reimplementing them ad hoc per script (see the manual `_mean_pairwise_dist`/
`energy_distance`/`cos` helpers in `experiments/real_basal_order.py` for an existing candidate
to migrate). Not currently an installed dependency — add it (`pertpy`) before relying on it.
