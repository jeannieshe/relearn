from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
import wandb

# class WandbCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(WandbCallback, self).__init__(verbose)

    # def _on_step(self) -> bool:
    #     # log training progress on each RL step natively
    #     wandb.log(
    #         {"train/reward": self.locals["rewards"][0]},
    #         step = self.num_timesteps
    #         )
    #     return True

# initialize wandb
run = wandb.init(
    entity="goodarzilab",
    project="relearn",
    sync_tensorboard=True
)

# train agent
model = PPO("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=20000, callback=WandbCallback())