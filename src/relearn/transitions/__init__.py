from relearn.config import EnvConfig
from relearn.transitions.base import TransitionModel


def build_transition_model(cfg: EnvConfig) -> TransitionModel:
    if cfg.transition_model == "state":
        from relearn.transitions.state_model import StateTransitionModel

        return StateTransitionModel(cfg)
    elif cfg.transition_model == "rhaister":
        from relearn.transitions.rhaister_model import RhaisterTransitionModel

        return RhaisterTransitionModel(cfg)
    else:
        raise ValueError(f"Unknown transition_model: {cfg.transition_model!r}")
