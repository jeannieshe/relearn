"""
Hydra structured configs for relearn.

DQNConfig and EnvConfig each hold every tunable default for their half of the
experiment (agent vs. environment). configs/agent/*.yaml and configs/env/*.yaml
only need to state the fields a given run overrides -- Hydra composes them
onto these dataclass defaults via the `defaults:` list in configs/config.yaml,
and validates every override against the dataclass's field types (catches
typos, and coerces things like the classic PyYAML gotcha where bare
scientific notation such as `3e-4` parses as a string, not a float).
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DQNConfig:
    batch_size: int = 128       # transitions sampled from the replay buffer per optimize_model() call
    gamma: float = 0.99         # discount factor on future reward
    eps_start: float = 0.9      # exploration/exploitation starting epsilon
    eps_end: float = 0.01       # epsilon floor
    eps_decay: float = 2500     # decay rate in steps; must be scaled to the run's actual step budget
    tau: float = 0.005          # soft-update rate for the target network
    lr: float = 3.0e-4          # AdamW learning rate
    seed: int = 42
    num_episodes: int = 300     # if the env employs a 1-step horizon, this is also the total step count
    replay_capacity: int = 10000
    forced_first_action: str | None = None   # e.g. "dmso" -- force step 0, agent chooses freely after
    forced_second_action: str | None = None  # e.g. "dmso" -- force every step after the first


@dataclass
class EnvConfig:
    # cell line -- name and accession are coupled, must refer to the same line
    cell_type_name: str = "SW480"
    cell_type_accession_number: str = "CVCL_0546"

    # state representation: which STATE embedding the observation is expressed in.
    # embed_key is the single switch between representations -- it must match the
    # obsm key the chosen STATE checkpoint was trained on ("X_hvg" or "X_state"),
    # and it selects both the neutral-state source (obsm[embed_key]) and how the
    # reward is scored (see RelearnChemicalEnv). "X_hvg" is 2000-dim raw HVGs the
    # apoptosis signature scores directly; "X_state" is the 2058-dim SE-600M
    # embedding, decoded back to the 2000-HVG panel via the model's gene_decoder
    # before scoring. The env yaml (env=sw480 vs env=sw480_se) bundles embed_key
    # with the coupled fields below so a run only needs to flip the one tag.
    embed_key: str = "X_hvg"
    cell_representation_dim: int = 2000
    hvg_gene_names_path: str = "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy"

    # state transition function: which virtual cell model predicts next states
    transition_model: str = "state"  # "state" or "rhaister"

    # STATE fewshot run/checkpoint (used when transition_model == "state")
    tahoe_dataset_dir: str = "data/models/ST-HVG-Tahoe"
    state_run_dir: str = "fewshot/state_generalization_X_hvg"
    checkpoint_name: str = "checkpoints/best.ckpt"

    # Rhaister fewshot run (used when transition_model == "rhaister")
    rhaister_dataset_dir: str = "data/models/Rhaister"
    rhaister_experiment_name: str = "tahoe_fewshot"

    # reward signature
    msigdb_gene_set: str = "HALLMARK_APOPTOSIS"
    gmt_path: str = "data/HALLMARK_APOPTOSIS.v2026.1.Hs.gmt"

    # reward function: which scoring strategy computes the potential-based
    # reward signal (see src/relearn/rewards.py). "ucell" is the original
    # per-cell UCell-vs-signature scorer (decoded gene space); "edistance_from_control"
    # scores distributional distance from a fresh draw of env's real DMSO
    # control pool, in embed_key (latent) space -- swappable without
    # touching RelearnChemicalEnv.
    reward_fn: str = "ucell"
    reward_reference_n_cells: int = 256  # only used by edistance_from_control
    reward_seed: int | None = None  # only used by edistance_from_control, see rewards.py

    # termination criterion (interpreted by the configured reward_fn's
    # goal_reached() -- e.g. UCellApoptosisReward treats this as a tolerance
    # window around its [0, 1] scale's goal of 1.0; other reward functions
    # may ignore it entirely, see rewards.py)
    termination_epsilon: float = 0.1

    # episode horizon: max number of perturbations the agent may apply before
    # the episode is truncated. 1 recovers the original single-step behavior.
    horizon: int = 1

    # rarely-varying dataset/machine paths
    tahoe_se_dir: str = "/large_storage/ctc/ML/transcriptomics_filtered/tahoe_se"
    dmso_control_pert: str = "[('DMSO_TF', 0.0, 'uM')]"
    num_cells: int = 1


@dataclass
class Config:
    # experiment bookkeeping -- these drive the wandb run name/group so a run's
    # spreadsheet row (e.g. experiment "A", run_id "A001") is traceable in wandb.
    experiment: str = "A"       # experiment family -> wandb group
    run_id: str = "A001"        # unique run label -> wandb run name (compact, sorts with spreadsheet)
    description: str = ""      # free-text summary of what this run tests -> wandb notes

    agent: DQNConfig = field(default_factory=DQNConfig)
    env: EnvConfig = field(default_factory=EnvConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
