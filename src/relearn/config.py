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


@dataclass
class EnvConfig:
    # cell line -- name and accession are coupled, must refer to the same line
    cell_type_name: str = "SW480"
    cell_type_accession_number: str = "CVCL_0546"

    # state representation: which STATE gene panel the observation is expressed in
    cell_representation_dim: int = 2000
    hvg_gene_names_path: str = "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy"

    # state transition function: which STATE fewshot run/checkpoint predicts next states
    tahoe_dataset_dir: str = "notebooks/jeannie/ST-HVG-Tahoe"
    state_run_dir: str = "fewshot/state_generalization_X_hvg"
    checkpoint_name: str = "checkpoints/best.ckpt"

    # reward signature
    msigdb_gene_set: str = "HALLMARK_APOPTOSIS"
    gmt_path: str = "data/HALLMARK_APOPTOSIS.v2026.1.Hs.gmt"

    # termination criterion
    termination_epsilon: float = 0.1

    # rarely-varying dataset/machine paths
    tahoe_se_dir: str = "/large_storage/ctc/ML/transcriptomics_filtered/tahoe_se"
    dmso_control_pert: str = "[('DMSO_TF', 0.0, 'uM')]"
    num_cells: int = 1


@dataclass
class Config:
    agent: DQNConfig = field(default_factory=DQNConfig)
    env: EnvConfig = field(default_factory=EnvConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
