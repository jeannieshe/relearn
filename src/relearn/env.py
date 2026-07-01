"""
This environment is specifically designed to model a reinforcement learning environment where the virtual cell model STATE is the environment. The actions are limited to the 1138 small molecule, chemical perturbations in Tahoe-100M.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import torch
import pickle
from state.tx.models.state_transition import StateTransitionPerturbationModel
from pathlib import Path

class RelearnChemicalEnv(gym.Env):
    def __init__(self):
        # globally used vars
        self.tahoe_dataset_dir = Path("../../notebooks/jeannie/ST-HVG-Tahoe/")
        self.dmso_control_pert = "[('DMSO_TF', 0.0, 'uM')]"

        # experiment vars
        self.cell_type_name = "SW480"
        self.cell_type_accession_number = "CVCL_0546"
        self.num_cells = 1
        self.cell_representation_dim = 2000 # 2000 HVG
        self.termination_epsilon = 0.1

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

        # begin with a neutral cell state
        # TODO: this should be biologically meaningful instead of just zeroes
        self.initial_cell_state = np.array(np.zeros(shape=(2000,), dtype=np.float32))
        self._cell_state = self.initial_cell_state

        # define the apoptosis classifier
        self.apoptosis_predictor = pass
    
        # define the state applier
        # load the STATE model
        checkpoint = Path(self.tahoe_dataset_dir / "checkpoints/best.ckpt")
        self._state_stepper = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint)
        self._state_stepper.eval()
        self._device = next(self._state_stepper.parameters()).device
    
    def _get_obs(self):
        return {
            "cell_state": self._cell_state,
        }

    def _get_info(self):
        return {
            "apoptosis score": self.apoptosis_predictor(self._cell_state),
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
        new_score = self.apoptosis_predictor(next_state)

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

        