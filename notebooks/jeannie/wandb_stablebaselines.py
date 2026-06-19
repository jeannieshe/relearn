from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # log training progress on each RL step natively
        wandb.log(
            {"train/reward": self.locals["rewards"][0]},
            step = self.num_timesteps
            )
        
# initialize wandb
run = wandb.init(
    entity="goodarzilab",
    project="relearn",
)

# train agent
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=1000, callback=WandbCallback())