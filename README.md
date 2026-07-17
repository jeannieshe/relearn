# relearn

RL agent using the STATE virtual cell model as a simulated environment to
discover drug perturbations that drive SW480 cells toward apoptosis.

## Layout

```
src/relearn/
  envs/small_molecules.py   # RelearnChemicalEnv — one-shot single-drug environment
  agents/dqn.py         # DQN agent + training loop (the Hydra entrypoint)
  config.py             # DQNConfig / EnvConfig dataclasses, registered with Hydra
  utils.py              # UCell scoring, .gmt signature loading
configs/
  config.yaml           # top-level Hydra composition (defaults: agent + env)
  agent/*.yaml          # named DQN hyperparameter variants
  env/*.yaml            # named environment variants (cell line, state rep, checkpoint, ...)
scripts/
  download_<MODEL>.py   # one per model — fetch its checkpoint from Hugging Face into data/models/
```

`envs/` and `agents/` are packages (not single files) because more than one
implementation is expected on each side — e.g. a multi-step/combination
environment alongside `small_molecules.py`, or a non-DQN agent alongside `dqn.py`.
Add a new file under the matching package, named for what it *is*
(`envs/multi_step_combo.py`, `agents/ppo.py`), not a version number.

## Getting the model

The environment's state-transition function is a STATE checkpoint that is **not**
committed to git — the checkpoint alone is ~1 GB. Fetch it once from Hugging Face
before your first run:

```
python scripts/download_ST-HVG-Tahoe.py
```

This downloads only what the agent loads (`best.ckpt` + `pert_onehot_map.pt`,
plus ~2 MB of run metadata, ~1.08 GB total) into `data/models/ST-HVG-Tahoe/`,
which is the default `env.tahoe_dataset_dir`. It deliberately skips the repo's
multi-GB evaluation artifacts. One downloader per model lives in `scripts/`
(see `scripts/README.md` for the naming convention and how to add another).

The neutral start-state cache (`SW480_dmso_neutral_hvg.npy`) is not on Hugging
Face; it is generated automatically on the first env init from `env.tahoe_se_dir`
(requires cluster access), then cached.

## Running a training run

Requires `hydra-core` (`pip install hydra-core`) in addition to the frozen
environment/requirements files.

```
python src/relearn/agents/dqn.py
python src/relearn/agents/dqn.py agent=eps_fix
python src/relearn/agents/dqn.py agent=eps_fix env=sw480 agent.lr=0.001
python src/relearn/agents/dqn.py experiment=B run_id=B003 description="testing eps_fix at 2-step horizon" agent=eps_fix env=sw480_2step
```

### Experiment identity: `experiment` / `run_id` / `description`

Every run also takes three top-level fields (siblings of `agent`/`env`, not
nested under either) that exist purely for bookkeeping across two people
running many experiments:

| Field | Default | Goes to | Purpose |
|---|---|---|---|
| `experiment` | `"A"` | wandb **group** | The experiment family — runs that share a hypothesis get grouped together in the wandb UI |
| `run_id` | `"A001"` | wandb **run name** + output dir | A compact, unique label — also names `outputs/<experiment>/<run_id>/` on disk, so the wandb run and the Hydra output dir always match |
| `description` | `""` | wandb **notes** | A one-line free-text summary of what this specific run tests |

Set all three from the CLI on every real run:

```
python src/relearn/agents/dqn.py experiment=B run_id=B003 description="testing eps_fix at 2-step horizon"
```

Assign `experiment`/`run_id` the same way you'd assign a row in a shared
experiment-tracking spreadsheet — pick the next unused ID so runs between the
two of you never collide, and the wandb group/name stay meaningful at a
glance instead of Hydra's default timestamp-named runs.

`description` is a *quick* note, not a replacement for a full `EXPERIMENTS.md`
entry — it's what shows up as the wandb run's notes field so you can tell
what a run was for without leaving the wandb UI, but the hypothesis/result/
interpretation still belongs in `EXPERIMENTS.md` (see below).

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
| Tag a real run's identity for wandb + the output dir | `python src/relearn/agents/dqn.py experiment=B run_id=B003 description="..."` |

Two things that trip people up the first time:
- **Group names have no `.yaml`** — `agent=eps_fix`, not `agent=eps_fix.yaml`
  (Hydra looks for `configs/agent/eps_fix.yaml` itself).
- **A typo in a field name fails loudly**, e.g. `agent.eps_decy=1` raises
  `Key 'eps_decy' not in 'DQNConfig'` — that's the schema validation working
  as intended, not a bug to work around.

## The config system

Uses [Hydra](https://hydra.cc/) to compose config from two named-file axes,
plus the run-identity fields set directly from the CLI (see above):

- **`agent`** — DQN hyperparameters (`batch_size`, `gamma`, `eps_start/end/decay`,
  `tau`, `lr`, `seed`, `num_episodes`, `replay_capacity`, `forced_second_action`),
  defaults in `DQNConfig` (`src/relearn/config.py`).
- **`env`** — what the environment *is* (cell line, state representation,
  which STATE checkpoint predicts transitions, reward signature, termination
  criterion, episode `horizon`), defaults in `EnvConfig` (`src/relearn/config.py`).

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
4. Run it with a real identity: `python src/relearn/agents/dqn.py agent=your_agent_config env=your_env_config experiment=B run_id=B004 description="..."`.
5. Commit the config file(s). They're small and text, so `git log configs/`
   becomes a readable history of what's been tried and why, independent of wandb.

### What ends up where

- **Hydra** writes the fully-resolved config and the exact CLI overrides for
  every run to `outputs/<experiment>/<run_id>/.hydra/` automatically
  (gitignored — it's per-run provenance, not something to commit). Naming the
  dir after `experiment`/`run_id` instead of a timestamp means the folder on
  disk, the wandb run, and the row in your tracking spreadsheet all share one
  identity.
- **wandb** gets that same fully-resolved config (every field, not just your
  overrides) plus system info, the git commit, and — via `wandb.save` — a
  copy of Hydra's `.hydra/*.yaml` files. The run is grouped by `experiment`,
  named by `run_id`, and annotated with `description` as its notes. So any
  wandb run is self-describing: you can always answer "what config produced
  this curve, and what was it testing" from the run page alone.
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
entry that names the `run_id` and the Hydra invocation, e.g.:

```markdown
## 2026-07-15 — fix eps-decay/step-budget mismatch
Run: B003 (agent=eps_fix env=sw480) | wandb: <run-url>
Hypothesis: kny202d4's EPS_DECAY=2500 barely decayed epsilon over its 300
total steps (1-step-per-episode env), so the agent never got to exploit.
Result: ...
Next: ...
```

`run_id` is the join key between the two records — it's the wandb run name
*and* the `outputs/<experiment>/<run_id>/` directory name, so anyone reading
`EXPERIMENTS.md` can find the exact hyperparameters (via the named configs or
the wandb run's attached `.hydra/config.yaml`), and anyone staring at a wandb
run can trace it back to the entry that explains it.
