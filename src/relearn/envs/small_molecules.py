"""
This environment is specifically designed to model a reinforcement learning environment where the virtual cell model STATE is the environment. The actions are limited to the 1138 small molecule perturbations in Tahoe-100M.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import h5py
from pathlib import Path
from relearn.config import EnvConfig
from relearn.transitions import build_transition_model
from relearn.utils import ucell_score, _load_gmt_signature

class RelearnChemicalEnv(gym.Env):
    def __init__(self, cfg: Optional[EnvConfig] = None):
        cfg = cfg if cfg is not None else EnvConfig()
        self.cfg = cfg

        # globally used vars
        self.tahoe_dataset_dir = Path(cfg.tahoe_dataset_dir)
        self.dmso_control_pert = cfg.dmso_control_pert
        self.step_counter = 0

        # cluster paths for the STATE-preprocessed Tahoe data (X_hvg + 2000-HVG panel
        # this checkpoint was trained on), separate from the fewshot bundle above
        self.tahoe_se_dir = Path(cfg.tahoe_se_dir)
        self.hvg_gene_names_path = Path(cfg.hvg_gene_names_path)

        # experiment vars
        self.cell_type_name = cfg.cell_type_name
        self.cell_type_accession_number = cfg.cell_type_accession_number
        self.num_cells = cfg.num_cells
        self.cell_representation_dim = cfg.cell_representation_dim
        self.termination_epsilon = cfg.termination_epsilon
        self.horizon = cfg.horizon
        self.msigdb_gene_set = cfg.msigdb_gene_set

        # which STATE embedding the agent observes and the model transitions in.
        # "X_hvg" is already gene-expression space (2000 HVGs) that the apoptosis
        # reward can score directly; any other embedding (e.g. "X_state", the
        # 2058-dim SE-600M representation) is a latent that must be decoded back
        # to the 2000-HVG panel via the model's gene_decoder before scoring.
        # This mirrors STATE's own convention (state_transition.py) that
        # embed_key in {"X_hvg", None} => output is gene space.
        self.embed_key = cfg.embed_key
        self._output_is_gene_space = self.embed_key in ("X_hvg", None)

        # the state-transition function: which virtual cell model predicts next
        # states, selected via cfg.transition_model ("state" or "rhaister")
        self._transition_model = build_transition_model(cfg)
        self.drug_list = self._transition_model.drug_list # actions are (name, concentration, units)
        self.action_space = gym.spaces.Discrete(len(self.drug_list))

        # define what the agent can observe
        # pass in the cell state, expressed in the cfg.embed_key representation
        # (2000-dim for X_hvg, 2058-dim for the X_state SE embedding)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.cell_representation_dim,),
            dtype=np.float32
        )

        # STATE's 2000-HVG gene panel, in the exact column order of obsm/X_hvg
        self.hvg_gene_names = np.load(self.hvg_gene_names_path, allow_pickle=True).astype(str)

        # define the apoptosis classifier
        self.sig_genes = _load_gmt_signature(cfg.gmt_path, self.msigdb_gene_set)
        self.apoptosis_predictor = ucell_score

        # when the observation is an embedding (not raw HVGs), the reward is scored
        # on gene expression decoded from that embedding, so the transition model
        # must carry a gene_decoder. Fail fast here rather than at the first step().
        if not self._output_is_gene_space and getattr(self._transition_model, "gene_decoder", None) is None:
            raise ValueError(
                f"embed_key={self.embed_key!r} is an embedding space, so the reward "
                "needs a gene_decoder to map it back to the 2000-HVG panel, but "
                f"transition_model={cfg.transition_model!r} has no gene_decoder. Use a "
                "checkpoint trained with a decoder (e.g. ST-SE-Tahoe) or set embed_key=X_hvg."
            )

        # begin with a neutral cell state: mean STATE profile (in the embed_key
        # representation) across all cells of this line treated with DMSO_TF
        self.initial_cell_state = self._load_dmso_neutral_state()
        self._cell_state = self.initial_cell_state
        self._step_count = 0

    def _load_dmso_neutral_state(self) -> np.ndarray:
        """
        Neutral initial state for self.cell_type_name: the mean STATE profile in
        the self.embed_key representation (obsm[embed_key] -- 2000-HVG X_hvg or
        the 2058-dim X_state SE embedding) across every cell in Tahoe-100M
        treated with the DMSO_TF vehicle control. A single control cell is too
        sparse (dropout leaves only ~50/2000 genes nonzero) to be a stable
        starting point, so we average over the full control population for this
        cell line. Cached to disk after the first (multi-GB h5ad) read.

        The cache is keyed by embed_key (e.g. SW480_dmso_neutral_hvg.npy vs
        SW480_dmso_neutral_state.npy) so HVG and SE runs never reuse each
        other's neutral state.
        """
        # "X_hvg" -> "hvg", "X_state" -> "state": short, back-compatible suffix
        key_suffix = self.embed_key[2:] if self.embed_key.startswith("X_") else self.embed_key
        cache_path = self.tahoe_dataset_dir / f"{self.cell_type_name}_dmso_neutral_{key_suffix}.npy"
        if cache_path.exists():
            return np.load(cache_path).astype(np.float32)

        h5ad_path = None
        for candidate in sorted(self.tahoe_se_dir.glob("c*.h5ad")):
            with h5py.File(candidate, "r") as f:
                cell_line = f["obs"]["cell_line"]["categories"][0]
                cell_line = cell_line.decode() if isinstance(cell_line, bytes) else cell_line
                if cell_line == self.cell_type_accession_number:
                    h5ad_path = candidate
                    break
        if h5ad_path is None:
            raise FileNotFoundError(
                f"No Tahoe-SE h5ad under {self.tahoe_se_dir} matches cell line {self.cell_type_accession_number}"
            )

        with h5py.File(h5ad_path, "r") as f:
            if self.embed_key not in f["obsm"]:
                raise KeyError(
                    f"obsm['{self.embed_key}'] not found in {h5ad_path} "
                    f"(available: {list(f['obsm'].keys())})"
                )
            pert_cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["drugname_drugconc"]["categories"][:]]
            control_idx = pert_cats.index(self.dmso_control_pert)
            pert_codes = f["obs"]["drugname_drugconc"]["codes"][:]
            control_rows = np.where(pert_codes == control_idx)[0]
            neutral_state = f["obsm"][self.embed_key][control_rows, :].mean(axis=0)

        neutral_state = neutral_state.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, neutral_state)
        return neutral_state

    def _get_obs(self):
        return self._cell_state

    def _to_gene_expression(self, cell_state: np.ndarray) -> np.ndarray:
        """
        Map a cell state in the model's embed_key representation to the 2000-HVG
        gene-expression vector the apoptosis signature is scored on.

        For X_hvg the state already *is* gene expression, so this is the identity.
        For an embedding representation (e.g. X_state) the STATE model's
        gene_decoder maps the 2058-dim latent back to the 2000-HVG counts panel
        (the same decode the `state tx infer` CLI writes into obsm['X_hvg']).
        """
        if self._output_is_gene_space:
            return cell_state
        return self._transition_model.decode_to_genes(cell_state)  # type: ignore[attr-defined]

    def _score_apoptosis(self, cell_state: np.ndarray) -> float:
        """Apoptosis reward for a cell state: decode to gene space if needed, then
        UCell-score against the signature on STATE's 2000-HVG panel order."""
        expr = self._to_gene_expression(cell_state)
        return self.apoptosis_predictor(expr, gene_names=self.hvg_gene_names, signature_genes=self.sig_genes)

    def _get_info(self):
        return {
            "apoptosis score": self._score_apoptosis(self._cell_state),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # seed the rng
        super().reset(seed=seed)

        # reset the cell state
        self._cell_state = self.initial_cell_state
        self._step_count = 0
        self._current_score = self._score_apoptosis(self._cell_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # begin with an uninformed agent, take a random action
        # given an action, apply it to the state
        next_state = self._transition_model.step(self._cell_state, action)

        # score the state -- decodes the embedding to the 2000-HVG panel first
        # when the observation isn't already raw HVGs (see _to_gene_expression)
        new_score = self._score_apoptosis(next_state)

        # make progress the signal instead of raw reward
        # potential based shaping looks like reward_t = score(s_t) - score(s_{t_1})
        # this is policy invariant (does not change the optimal policy of the agent)
        old_score = self._current_score
        delta = new_score - old_score

        # update the state
        self._cell_state = next_state
        self._current_score = new_score
        self._step_count += 1

        # check termination, truncation criteria
        # terminated: reached the apoptosis goal (score ~= 1) -- a real end state,
        #   so the agent bootstraps no future value past it.
        # truncated: hit the horizon without reaching the goal -- an artificial
        #   cutoff, so the agent should still bootstrap the next state's value.
        terminated = abs(1 - new_score) <= self.termination_epsilon
        truncated = self._step_count >= self.horizon

        # calculate reward
        reward = delta

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

        