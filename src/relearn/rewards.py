"""
RewardFunction is the interface a scoring strategy implements to serve as
RelearnChemicalEnv's reward signal: given the current cell-state set (shape
[S, D], in cfg.embed_key space) it returns one scalar score, used both for
potential-based reward shaping (reward_t = score(s_t) - score(s_{t-1})) and,
via goal_reached(), whether the episode should terminate early.
UCellApoptosisReward (the original per-cell apoptosis-signature scorer) and
EDistanceFromControlReward (a distributional distance scorer) are the two
implementations; build_reward_fn(cfg) selects one via cfg.reward_fn, the
same pattern transitions/base.py's build_transition_model uses for
cfg.transition_model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from relearn.config import EnvConfig


@runtime_checkable
class RewardFunction(Protocol):
    def __call__(self, env, cell_state: np.ndarray) -> float:
        """Score the current cell-state set. `env` is the RelearnChemicalEnv
        instance, so a reward function can read whatever it needs from it
        (env._control_pool, env.hvg_gene_names, env.sig_genes, env.rng, ...)."""
        ...

    def goal_reached(self, score: float) -> bool:
        """Whether this score means the episode has reached its goal (a real
        terminal state, distinct from hitting the horizon truncation)."""
        ...


class UCellApoptosisReward:
    """
    Mean per-cell UCell score against env.sig_genes (HALLMARK_APOPTOSIS by
    default), in decoded gene-expression space -- env._to_gene_expression
    handles the decode when cfg.embed_key is a latent representation like
    X_state. The original/default reward.

    UCell scores are bounded in [0, 1], with 1.0 meaning the signature genes
    are maximally rank-dominant in that cell -- so "goal reached" is a
    tolerance window around 1.0.
    """

    def __init__(self, termination_epsilon: float):
        self.termination_epsilon = termination_epsilon

    def __call__(self, env, cell_state: np.ndarray) -> float:
        expr = env._to_gene_expression(cell_state)
        scores = env.apoptosis_predictor(expr, gene_names=env.hvg_gene_names, signature_genes=env.sig_genes)
        return float(np.mean(scores))

    def goal_reached(self, score: float) -> bool:
        return abs(1.0 - score) <= self.termination_epsilon


def _mean_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> float:
    d2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None, :] - 2.0 * (X @ Y.T)
    return float(np.sqrt(np.maximum(d2, 0)).mean())


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    2*E||x-y|| - E||x-x'|| - E||y-y'||. Zero iff the two populations share a
    distribution, so unlike a mean-vs-mean comparison it also sees differences
    in spread -- which matters because STATE's predictions are known to be
    smoother than real cells (see experiments/real_basal_order.py, which
    imports this rather than keeping its own copy).
    """
    return float(2 * _mean_pairwise_dist(X, Y) - _mean_pairwise_dist(X, X) - _mean_pairwise_dist(Y, Y))


class EDistanceFromControlReward:
    """
    E-distance between the current cell-state cloud and a reference cloud
    drawn from env's real DMSO control pool (env._control_pool -- the same
    pool RelearnChemicalEnv._load_dmso_control_pool() loads its own starting
    cell sets from), both in cfg.embed_key (latent) space -- unlike
    UCellApoptosisReward, no gene decode is needed, so this works identically
    whether embed_key is X_hvg or a latent like X_state. Higher score =
    predicted cells sit farther from the untreated baseline distribution;
    0 = statistically indistinguishable from control.

    The reference cloud is drawn ONCE, lazily on first call, from its own
    RNG seeded by `seed` -- not redrawn every call, and not tied to
    env.rng (which keeps advancing for the env's own starting-set draws).
    A given seed always picks the same reference cells from the pool, and
    that same reference is then reused for every score for this reward
    function instance's whole lifetime, so score deltas reflect the
    perturbation, not which reference cells happened to be sampled that call.
    """

    def __init__(self, n_ref_cells: int = 256, seed: int | None = None):
        self.n_ref_cells = n_ref_cells
        self._rng = np.random.default_rng(seed)
        self._reference: np.ndarray | None = None

    def __call__(self, env, cell_state: np.ndarray) -> float:
        reference = self._reference
        if reference is None:
            pool = env._control_pool
            n = min(self.n_ref_cells, len(pool))
            idx = self._rng.choice(len(pool), size=n, replace=False)
            reference = pool[idx].astype(np.float64)
            self._reference = reference
        return energy_distance(np.asarray(cell_state, dtype=np.float64), reference)

    def goal_reached(self, score: float) -> bool:
        return False


def build_reward_fn(cfg: EnvConfig) -> RewardFunction:
    if cfg.reward_fn == "ucell":
        return UCellApoptosisReward(cfg.termination_epsilon)
    elif cfg.reward_fn == "edistance_from_control":
        return EDistanceFromControlReward(cfg.reward_reference_n_cells, seed=cfg.reward_seed)
    else:
        raise ValueError(f"Unknown reward_fn: {cfg.reward_fn!r}")
