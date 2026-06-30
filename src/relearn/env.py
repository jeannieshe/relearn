"""
This environment is specifically designed to model a reinforcement learning environment where the virtual cell model STATE is the environment. The actions are limited to the 1138 small molecule, chemical perturbations in Tahoe-100M.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import torch
import pickle

class RelearnChemicalEnv(gym.Env):
    def __init__(self):
        # define action space
        pert_map = torch.load("../../notebooks/jeannie/ST-HVG-Tahoe/fewshot/state_generalization_X_hvg/pert_onehot_map.pt", weights_only=False)
        self.drug_list = list(pert_map.keys()) # actions are (name, concentration, units)
        self.pert_matrix = torch.stack(list(pert_map.values())) # shape: (1138, 1138)
        self.action_space = gym.spaces.Discrete(len(self.drug_list))

        # define what the agent can observe
        # choose one fixed cell type first

        # pass in the cell state
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=np.inf,
            shape=(2000,),
            dtype=np.float32
        )
        # begin with a neutral cell state
        # TODO: this should be biologically meaningful instead of just zeroes
        self.initial_cell_state = np.array(np.zeros(shape=(2000,), dtype=np.float32))
        self._cell_state = self.initial_cell_state

        # define the apoptosis classifier
        self.apoptosis_predictor = pass
    
    def _get_obs(self):
        return {
            "cell_state": self._cell_state,
        }

    def _get_info(self):
        return {
            "apoptosis score": self.apoptosis_predictor(self._cell_state),
        }

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
        