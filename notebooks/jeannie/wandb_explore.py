# this script is a dry-run of using the wandb api to track your expt progress!
# code and setup from: https://docs.wandb.ai/models/quickstart

import wandb
import random

wandb.login()

run = wandb.init(
    entity="goodarzilab",
    project="relearn",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs":10,
    },
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # log metrics to wandb
    run.log({
        "acc": acc,
        "loss": loss
    })

# finish the run
run.finish()