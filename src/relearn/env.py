"""
This environment is specifically designed to model a reinforcement learning environment where the virtual cell model STATE is the environment. The actions are limited to the 1138 small molecule perturbations in Tahoe-100M.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import torch
import pickle
import h5py
from state.tx.models.state_transition import StateTransitionPerturbationModel
from pathlib import Path
from relearn.utils import ucell_score, _load_gmt_signature

class RelearnChemicalEnv(gym.Env):
    def __init__(self):
        # globally used vars
        self.tahoe_dataset_dir = Path("notebooks/jeannie/ST-HVG-Tahoe")
        self.dmso_control_pert = "[('DMSO_TF', 0.0, 'uM')]"

        # cluster paths for the STATE-preprocessed Tahoe data (X_hvg + 2000-HVG panel
        # this checkpoint was trained on), separate from the fewshot bundle above
        self.tahoe_se_dir = Path("/large_storage/ctc/ML/transcriptomics_filtered/tahoe_se")
        self.hvg_gene_names_path = Path("/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy")

        # experiment vars
        self.cell_type_name = "SW480"
        self.cell_type_accession_number = "CVCL_0546"
        self.num_cells = 1
        self.cell_representation_dim = 2000 # 2000 HVG
        self.termination_epsilon = 0.1
        self.msigdb_gene_set = "HALLMARK_APOPTOSIS"

        # define action space
        self.pert_map = torch.load(Path(self.tahoe_dataset_dir / "fewshot/state_generalization_X_hvg/pert_onehot_map.pt"), weights_only=False)
        self.drug_list = list(self.pert_map.keys()) # actions are (name, concentration, units)
        self.pert_matrix = torch.stack(list(self.pert_map.values())) # shape: (1138, 1138)
        self.action_space = gym.spaces.Discrete(len(self.drug_list))

        # define what the agent can observe
        # pass in the cell state
        # here, starting with 2000 HVG raw representation
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=np.inf,
            shape=(self.cell_representation_dim,),
            dtype=np.float32
        )

        # STATE's 2000-HVG gene panel, in the exact column order of obsm/X_hvg
        self.hvg_gene_names = np.load(self.hvg_gene_names_path, allow_pickle=True).astype(str)

        # begin with a neutral cell state: mean STATE X_hvg profile across all
        # SW480 cells treated with the DMSO_TF vehicle control
        self.initial_cell_state = self._load_dmso_neutral_state()
        self._cell_state = self.initial_cell_state

        # define the apoptosis classifier
        self.sig_genes = _load_gmt_signature("data/HALLMARK_APOPTOSIS.v2026.1.Hs.gmt", self.msigdb_gene_set)
        self.apoptosis_predictor = ucell_score
    
        # define the state applier
        # load the STATE model
        checkpoint = Path(self.tahoe_dataset_dir / "fewshot/state_generalization_X_hvg/checkpoints/best.ckpt")
        self._state_stepper = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint)
        self._state_stepper.eval()
        self._device = next(self._state_stepper.parameters()).device
    
    def _load_dmso_neutral_state(self) -> np.ndarray:
        """
        Neutral initial state for self.cell_type_name: the mean STATE X_hvg
        profile (library-normalized + log1p, 2000-HVG) across every cell in
        Tahoe-100M treated with the DMSO_TF vehicle control. A single control
        cell is too sparse (dropout leaves only ~50/2000 genes nonzero) to be
        a stable starting point, so we average over the full control population
        for this cell line. Cached to disk after the first (multi-GB h5ad) read.
        """
        cache_path = self.tahoe_dataset_dir / f"{self.cell_type_name}_dmso_neutral_hvg.npy"
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
            pert_cats = [c.decode() if isinstance(c, bytes) else c for c in f["obs"]["drugname_drugconc"]["categories"][:]]
            control_idx = pert_cats.index(self.dmso_control_pert)
            pert_codes = f["obs"]["drugname_drugconc"]["codes"][:]
            control_rows = np.where(pert_codes == control_idx)[0]
            neutral_state = f["obsm"]["X_hvg"][control_rows, :].mean(axis=0)

        neutral_state = neutral_state.astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, neutral_state)
        return neutral_state

    def _get_obs(self):
        return self._cell_state

    def _get_info(self):
        return {
            "apoptosis score": self.apoptosis_predictor(self._cell_state, gene_names=self.hvg_gene_names, signature_genes=self.sig_genes),
        }

    def _state_stepper_helper(self, cell_state: np.ndarray, action: int) -> np.ndarray:
        """
        Helper for cell stepper to match the expected STATE input. StateTransitionPerturbationModel takes in a batch dict and returns predicted cell states.
        model.forward(batch, padded=False) where batch must have ctrl_cell_emb, pert_emb, and pert_name

        ctrl_cell_emb has shape [S, E_in] being the control cell embeddings
        pert_emb has shape [S, pert_dim] and represents the perturbation one-hot vector, repeated S times
        pert_name (type: list[str]) has length S and is the drug name string, repeated S times

        With padded=False, S can be any length.
        """
        drug_name = self.drug_list[action]
        pert_vec = self.pert_map[drug_name].float() # shape: (1138,)

        # build batch
        batch = {
            "ctrl_cell_emb": torch.tensor(cell_state, dtype=torch.float32, device=self._device).unsqueeze(0), # [self.num_cells, E_in]
            "pert_emb": pert_vec.unsqueeze(0).to(self._device), # [self.num_cells, pert_dim]
            "pert_name": [str(drug_name)],
        }

        with torch.no_grad():
            pred = self._state_stepper.forward(batch, padded=False) # [self.num_cells, self.cell_representation_dim]
        
        return pred.squeeze(0).cpu().numpy() # [self.cell_representation_dim]


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # seed the rng
        super().reset(seed=seed)

        # reset the cell state
        self._cell_state = self.initial_cell_state

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # begin with an uninformed agent, take a random action
        # given an action, apply it to the state
        next_state = self._state_stepper_helper(self._cell_state, action)

        # score the state
        # single cell: expr is (2000,), gene_names is STATE's HVG panel order
        new_score = self.apoptosis_predictor(next_state, gene_names=self.hvg_gene_names, signature_genes=self.sig_genes)

        # update the state
        self._cell_state = next_state

        # check termination, truncation criteria
        terminated = abs(1 - new_score) <= self.termination_epsilon
        truncated = True # TODO: add a step count, set truncation to be true once horizon is reached. currently one-step horizon

        # calculate reward
        reward = new_score

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

        