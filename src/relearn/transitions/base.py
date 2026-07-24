"""
TransitionModel is the interface a virtual cell model implements to serve as
RelearnChemicalEnv's state-transition function: given the current cell state
and a discrete drug action, predict the next cell state. StateTransitionModel
(STATE) and RhaisterTransitionModel (Rhaister) are the two implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TransitionModel(Protocol):
    # Fixed action-space list of (name, concentration, units) tuples, in the
    # exact order `action` (an int index into this list) refers to.
    drug_list: list

    def step(self, cell_state: np.ndarray, action: int) -> np.ndarray:
        """Predict the next cell state given the current state and a drug action."""
        ...
