"""STATE-backed TransitionModel: wraps StateTransitionPerturbationModel as
used by RelearnChemicalEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

from relearn.config import EnvConfig


class StateTransitionModel:
    def __init__(self, cfg: EnvConfig):
        tahoe_dataset_dir = Path(cfg.tahoe_dataset_dir)
        state_run_dir = tahoe_dataset_dir / cfg.state_run_dir

        # define action space
        self.pert_map = torch.load(Path(state_run_dir / "pert_onehot_map.pt"), weights_only=False)
        self.drug_list = list(self.pert_map.keys())  # actions are (name, concentration, units)
        self.pert_matrix = torch.stack(list(self.pert_map.values()))  # shape: (1138, 1138)

        checkpoint = Path(state_run_dir / cfg.checkpoint_name)
        self._model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint)
        self._model.eval()
        self._device = next(self._model.parameters()).device

        # exposed for RelearnChemicalEnv's embed_key != X_hvg decode path (see
        # _to_gene_expression); None if this checkpoint wasn't trained with one
        self.gene_decoder = getattr(self._model, "gene_decoder", None)

    def step(self, cell_state: np.ndarray, action: int) -> np.ndarray:
        """
        StateTransitionPerturbationModel takes in a batch dict and returns predicted
        cell states. model.forward(batch, padded=False) where batch must have
        ctrl_cell_emb, pert_emb, and pert_name.

        ctrl_cell_emb has shape [S, E_in] being the control cell embeddings
        pert_emb has shape [S, pert_dim] and represents the perturbation one-hot
        vector, repeated S times
        pert_name (type: list[str]) has length S and is the drug name string,
        repeated S times

        With padded=False, S can be any length.
        """
        drug_name = self.drug_list[action]
        pert_vec = self.pert_map[drug_name].float()  # shape: (1138,)

        batch = {
            "ctrl_cell_emb": torch.tensor(cell_state, dtype=torch.float32, device=self._device).unsqueeze(0),
            "pert_emb": pert_vec.unsqueeze(0).to(self._device),
            "pert_name": [str(drug_name)],
        }

        with torch.no_grad():
            pred = self._model.forward(batch, padded=False)  # [1, cell_representation_dim]

        return pred.squeeze(0).cpu().numpy()  # [cell_representation_dim]

    def step_with_pert_vector(self, cell_state: np.ndarray, pert_vec: torch.Tensor) -> np.ndarray:
        """
        Same as step(), but takes an arbitrary perturbation vector instead of an
        action index -- the escape hatch for co-perturbation, where the input is
        a two-hot (or weighted) combination rather than a single drug's one-hot.

        forward() never reads pert_name (only pert_emb reaches the network), so
        an off-one-hot pert_emb is a well-formed input. It is still extrapolation:
        pert_encoder is a single Linear(1138, 768) trained exclusively on one-hot
        inputs, so encode(a + b) == encode(a) + encode(b) - bias by construction.
        The model has no learned representation of drug combinations; whatever
        non-additivity shows up downstream comes from the transformer and decoder.
        """
        batch = {
            "ctrl_cell_emb": torch.tensor(cell_state, dtype=torch.float32, device=self._device).unsqueeze(0),
            "pert_emb": pert_vec.float().unsqueeze(0).to(self._device),
            "pert_name": ["<combination>"],
        }
        with torch.no_grad():
            pred = self._model.forward(batch, padded=False)
        return pred.squeeze(0).cpu().numpy()

    def decode_to_genes(self, cell_state: np.ndarray) -> np.ndarray:
        """Decode an embed_key latent (e.g. X_state) back to the 2000-HVG gene
        panel via this checkpoint's gene_decoder (see EnvConfig.embed_key)."""
        if self.gene_decoder is None:
            raise ValueError("this checkpoint has no gene_decoder to decode with")
        with torch.no_grad():
            latent = torch.as_tensor(cell_state, dtype=torch.float32, device=self._device)
            genes = self.gene_decoder(latent.unsqueeze(0))  # [1, 2000]
        return genes.squeeze(0).cpu().numpy()
