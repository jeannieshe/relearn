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
from relearn.rewards import build_reward_fn
from relearn.utils import ucell_score, _load_gmt_signature

class RelearnChemicalEnv(gym.Env):
    def __init__(self, cfg: Optional[EnvConfig] = None, seed: Optional[int] = None):
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

        # define what the agent can observe: a set of num_cells cell states,
        # each expressed in the cfg.embed_key representation (2000-dim for
        # X_hvg, 2058-dim for the X_state SE embedding) -- see _get_obs().
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_cells, self.cell_representation_dim),
            dtype=np.float32
        )

        # STATE's 2000-HVG gene panel, in the exact column order of obsm/X_hvg
        self.hvg_gene_names = np.load(self.hvg_gene_names_path, allow_pickle=True).astype(str)

        # define the apoptosis classifier (only used by reward_fn="ucell")
        self.sig_genes = _load_gmt_signature(cfg.gmt_path, self.msigdb_gene_set)
        self.apoptosis_predictor = ucell_score

        # the reward function: which scoring strategy computes the shaped
        # reward, selected via cfg.reward_fn ("ucell" or
        # "edistance_from_control") -- see rewards.py
        self._reward_fn = build_reward_fn(cfg)

        # only reward_fn="ucell" needs decoded gene space (it scores against
        # a gene-symbol signature); "edistance_from_control" compares clouds
        # directly in embed_key/latent space, no decode involved. So the
        # gene_decoder requirement below only applies to "ucell" -- fail fast
        # here rather than at the first step().
        if (
            cfg.reward_fn == "ucell"
            and not self._output_is_gene_space
            and getattr(self._transition_model, "gene_decoder", None) is None
        ):
            raise ValueError(
                f"embed_key={self.embed_key!r} is an embedding space, so the reward "
                "needs a gene_decoder to map it back to the 2000-HVG panel, but "
                f"transition_model={cfg.transition_model!r} has no gene_decoder. Use a "
                "checkpoint trained with a decoder (e.g. ST-SE-Tahoe) or set embed_key=X_hvg."
            )

        # seed the rng that reset() draws fresh control-cell sets from. Passing
        # the same seed here reproduces the whole run's draw sequence end to
        # end (mirrors gym's own reset(seed=...) contract); reset() itself is
        # never reseeded unless the caller explicitly passes reset(seed=...)
        # again, so consecutive episodes get *different* draws by default.
        self.rng = np.random.default_rng(seed=seed)

        # real DMSO control pool for this cell line, in the embed_key
        # representation. reset() defaults to replaying the same num_cells-sized
        # starting set every episode (drawn once here); pass
        # reset(options={"resample": True}) to draw a fresh set instead, which
        # then becomes the new default until resampled again.
        self._control_pool = self._load_dmso_control_pool()
        self._initial_cell_set = self._draw_cell_set()
        self._cell_state = None
        self._step_count = 0

    def _draw_cell_set(self) -> np.ndarray:
        """Sample self.num_cells rows from self._control_pool using self.rng."""
        idx = self.rng.choice(
            len(self._control_pool), size=self.num_cells, replace=len(self._control_pool) < self.num_cells
        )
        return self._control_pool[idx]

    def _load_dmso_control_pool(self) -> np.ndarray:
        """
        Every real cell of self.cell_type_name treated with the DMSO_TF vehicle
        control, in the self.embed_key representation (obsm[embed_key] --
        2000-HVG X_hvg or the 2058-dim X_state SE embedding). reset() samples
        self.num_cells rows from this pool fresh each episode -- a single
        control cell is too sparse (dropout leaves only ~50/2000 genes nonzero)
        to be a stable starting point, but a set of num_cells doesn't need
        pre-averaging since the transition model attends over the set itself.
        Cached to disk after the first (multi-GB h5ad) read.

        The cache is keyed by embed_key (e.g. SW480_dmso_pool_hvg.npy vs
        SW480_dmso_pool_state.npy) so HVG and SE runs never reuse each other's
        pool.
        """
        # "X_hvg" -> "hvg", "X_state" -> "state": short, back-compatible suffix
        key_suffix = self.embed_key[2:] if self.embed_key.startswith("X_") else self.embed_key
        cache_path = self.tahoe_dataset_dir / f"{self.cell_type_name}_dmso_pool_{key_suffix}.npy"
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
            control_rows = np.sort(np.where(pert_codes == control_idx)[0])  # h5py fancy indexing needs sorted rows
            pool = f["obsm"][self.embed_key][control_rows, :]

        pool = pool.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pool)
        return pool

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

    def _score(self, cell_state: np.ndarray) -> float:
        """Score a set of cell states with whatever reward function cfg.reward_fn
        selects (see rewards.py) -- the single scalar step()/reset() use for
        reward shaping and termination."""
        return self._reward_fn(self, cell_state)

    def _get_info(self):
        return {
            "score": self._score(self._cell_state),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # reset(seed=X) reseeds self.rng and redraws the fixed starting set
        # under the new seed (so a given seed always reproduces the same set).
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            self._initial_cell_set = self._draw_cell_set()
        # reset(options={"resample": True}) explicitly requests a fresh draw
        # without needing a new seed; it becomes the new default going forward.
        elif options and options.get("resample"):
            self._initial_cell_set = self._draw_cell_set()

        # default: replay the same starting set every episode
        self._cell_state = self._initial_cell_set
        self._step_count = 0
        self._current_score = self._score(self._cell_state)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _advance(self, next_state: np.ndarray):
        """
        Shared bookkeeping for step()/step_vector(): score the new state,
        potential-based reward shaping, advance self._cell_state, check
        termination/truncation. Factored out so both a single-drug action and
        an arbitrary (e.g. multi-hot combination) perturbation vector go
        through identical accounting.
        """
        # score the state with whatever reward function is configured
        new_score = self._score(next_state)

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
        # terminated: the configured reward function's own goal_reached() --
        #   a real end state, so the agent bootstraps no future value past it.
        #   (e.g. UCellApoptosisReward: score within termination_epsilon of 1.0;
        #   EDistanceFromControlReward: never -- see rewards.py)
        # truncated: hit the horizon without reaching the goal -- an artificial
        #   cutoff, so the agent should still bootstrap the next state's value.
        terminated = self._reward_fn.goal_reached(new_score)
        truncated = self._step_count >= self.horizon

        # calculate reward
        reward = delta

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def step(self, action):
        # begin with an uninformed agent, take a random action
        # given an action, apply it to the state
        next_state = self._transition_model.step(self._cell_state, action)
        return self._advance(next_state)

    def step_vector(self, pert_vec):
        """
        Same as step(), but takes an arbitrary perturbation vector instead of
        a single drug_list index -- e.g. a multi-hot combination of several
        drugs applied simultaneously. Build one with self.multi_hot(), or hand
        -craft any weighted combination (see StateTransitionModel.step_with_pert_vector
        for the extrapolation caveats of a non-one-hot pert_emb).

        Not part of action_space / Discrete -- this bypasses the discrete
        action contract agents/dqn.py and agents/dqn_set.py rely on, so it's
        meant for standalone combinatorial-perturbation experiment scripts,
        not for driving the env inside a training loop.
        """
        if not hasattr(self._transition_model, "step_with_pert_vector"):
            raise NotImplementedError(
                f"transition_model={self.cfg.transition_model!r} has no step_with_pert_vector "
                "-- step_vector() is only supported for transition_model='state'"
            )
        next_state = self._transition_model.step_with_pert_vector(self._cell_state, pert_vec)
        return self._advance(next_state)

    def multi_hot(self, drug_indices):
        """
        Build a multi-hot perturbation vector selecting multiple drugs from
        self.drug_list at once -- e.g.
        env.step_vector(env.multi_hot([12, 47])) applies drugs 12 and 47
        simultaneously (their one-hot pert vectors summed). Feed the result to
        step_vector(), not step() (which only accepts a single int index).
        """
        pert_matrix = getattr(self._transition_model, "pert_matrix", None)
        if pert_matrix is None:
            raise NotImplementedError(
                f"transition_model={self.cfg.transition_model!r} has no pert_matrix -- "
                "multi_hot() is only supported for transition_model='state'"
            )
        return pert_matrix[list(drug_indices)].sum(dim=0)
