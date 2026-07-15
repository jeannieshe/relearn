# relearn

RL agent using the STATE virtual cell model as a simulated environment to
discover drug perturbations that drive SW480 cells toward apoptosis.

## Layout

```
src/relearn/
  envs/single_step.py   # RelearnChemicalEnv — one-shot single-drug environment
  agents/dqn.py         # DQN agent + training loop (the Hydra entrypoint)
  config.py             # DQNConfig / EnvConfig dataclasses, registered with Hydra
  utils.py              # UCell scoring, .gmt signature loading
configs/
  config.yaml           # top-level Hydra composition (defaults: agent + env)
  agent/*.yaml          # named DQN hyperparameter variants
  env/*.yaml            # named environment variants (cell line, state rep, checkpoint, ...)
```

`envs/` and `agents/` are packages (not single files) because more than one
implementation is expected on each side — e.g. a multi-step/combination
environment alongside `single_step.py`, or a non-DQN agent alongside `dqn.py`.
Add a new file under the matching package, named for what it *is*
(`envs/multi_step_combo.py`, `agents/ppo.py`), not a version number.

## Running a training run

Requires `hydra-core` (`pip install hydra-core`) in addition to the frozen
environment/requirements files.

```
python src/relearn/agents/dqn.py
python src/relearn/agents/dqn.py agent=eps_fix
python src/relearn/agents/dqn.py agent=eps_fix env=sw480 agent.lr=0.001
```

### Hydra cheatsheet

If you haven't used Hydra before, the whole system is: pick which named
config to use per group, optionally override individual fields, run.

| You want to... | Do this |
|---|---|
| Run with all defaults | `python src/relearn/agents/dqn.py` |
| Use a named agent config | `python src/relearn/agents/dqn.py agent=eps_fix` |
| Use a named env config | `python src/relearn/agents/dqn.py env=sw480` |
| Override one field, no new file | `python src/relearn/agents/dqn.py agent.lr=0.001` |
| Override a field in a specific group | `python src/relearn/agents/dqn.py env.termination_epsilon=0.05` |
| Combine a named config *and* a one-off override | `python src/relearn/agents/dqn.py agent=eps_fix agent.seed=7` (the named config applies first, then the override) |
| See the fully resolved config without running | `python src/relearn/agents/dqn.py --cfg job` |
| List everything available to override | `python src/relearn/agents/dqn.py --help` |

Two things that trip people up the first time:
- **Group names have no `.yaml`** — `agent=eps_fix`, not `agent=eps_fix.yaml`
  (Hydra looks for `configs/agent/eps_fix.yaml` itself).
- **A typo in a field name fails loudly**, e.g. `agent.eps_decy=1` raises
  `Key 'eps_decy' not in 'DQNConfig'` — that's the schema validation working
  as intended, not a bug to work around.

## The config system

Uses [Hydra](https://hydra.cc/) to compose config from two independent axes:

- **`agent`** — DQN hyperparameters (`batch_size`, `gamma`, `eps_start/end/decay`,
  `tau`, `lr`, `seed`, `num_episodes`, `replay_capacity`), defaults in
  `DQNConfig` (`src/relearn/config.py`).
- **`env`** — what the environment *is* (cell line, state representation,
  which STATE checkpoint predicts transitions, reward signature, termination
  criterion), defaults in `EnvConfig` (`src/relearn/config.py`).

`configs/config.yaml` is the top-level file that picks one config from each
group (see its `defaults:` list). Named YAML files under `configs/agent/` and
`configs/env/` only need to state the fields a given run overrides — the rest
fall back to the dataclass defaults:

```yaml
# configs/agent/eps_fix.yaml
eps_decay: 150
num_episodes: 2000
```

Hydra validates every override against `DQNConfig`/`EnvConfig`'s field types —
typos raise immediately (`Key 'eps_decy' not in 'DQNConfig'`), and values are
coerced to the right type, including the classic PyYAML gotcha where bare
scientific notation (`3e-4`) parses as a *string* unless written `3.0e-4`.

You aren't limited to named files — anything can be overridden straight from
the CLI without creating a new YAML at all: `agent.lr=0.001 env.termination_epsilon=0.05`.

### Starting a new experiment

1. For a new **agent** variant: copy `configs/agent/baseline.yaml` (fully
   spelled out) or a smaller delta file like `configs/agent/eps_fix.yaml` to
   a name describing the hypothesis, e.g. `configs/agent/two_step_combo.yaml`.
2. For a new **environment** variant (new cell line, different STATE
   checkpoint, different reward signature): same pattern under `configs/env/`,
   e.g. `configs/env/hela.yaml` overriding just `cell_type_name` and
   `cell_type_accession_number`.
3. Only write the fields you're changing, plus a one-line comment on *why*.
4. Run `python src/relearn/agents/dqn.py agent=your_agent_config env=your_env_config`.
5. Commit the config file(s). They're small and text, so `git log configs/`
   becomes a readable history of what's been tried and why, independent of wandb.

### What ends up where

- **Hydra** writes the fully-resolved config and the exact CLI overrides for
  every run to `outputs/<date>/<time>/.hydra/` automatically (gitignored —
  it's per-run provenance, not something to commit).
- **wandb** gets that same fully-resolved config (every field, not just your
  overrides) plus system info, the git commit, and — via `wandb.save` — a
  copy of Hydra's `.hydra/*.yaml` files. So any wandb run is self-describing:
  you can always answer "what config produced this curve" from the run page
  alone.
- **git** gets the named YAML files under `configs/agent/` and `configs/env/`,
  so config changes are reviewable and diffable the normal way, and survive
  independently of wandb.

## Config system vs. EXPERIMENTS.md

These solve different problems and neither replaces the other:

- **The config system (Hydra configs + wandb) is the record of *what* ran** —
  exact hyperparameters, metrics, git commit, system info. It's mechanical
  and complete, but it doesn't know why you ran it or what you concluded.
- **`EXPERIMENTS.md` is the record of *why* and *so what*** — the hypothesis
  behind a run or batch of runs, the interpretation of the result, dead ends,
  and the decision that came out of it. This is exactly the layer that's easy
  to lose track of between two people, since it's not something any metrics
  tool captures automatically.

Convention: when you kick off a run worth remembering, add an `EXPERIMENTS.md`
entry that names the exact Hydra invocation and the wandb run, e.g.:

```markdown
## 2026-07-15 — fix eps-decay/step-budget mismatch
Run: agent=eps_fix env=sw480 | wandb: <run-url>
Hypothesis: kny202d4's EPS_DECAY=2500 barely decayed epsilon over its 300
total steps (1-step-per-episode env), so the agent never got to exploit.
Result: ...
Next: ...
```

The Hydra invocation is the join key between the two records — anyone reading
`EXPERIMENTS.md` can find the exact hyperparameters via the named configs (or
by reading the wandb run's attached `.hydra/config.yaml`), and anyone staring
at a wandb run can trace it back to the entry that explains it.
