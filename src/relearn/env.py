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

        # # define what the agent can observe
        # with open("../../notebooks/jeannie/ST-HVG-Tahoe/fewshot/state_generalization_X_hvg/cell_type_onehot_map.pkl", 'rb') as file:
        #     self.cell_type_map = pickle.load(file)

        # # pass in the cell state and the cell type
        # # concatenate and flatten into one vector
        # self.observation_space = gym.spaces.Box(
        #     low=0, 
        #     high=np.inf,
        #     shape=(2000 + len(self.cell_type_map),),
        #     dtype=np.float32
        # )

    def _get_obs(self):
        pass