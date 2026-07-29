"""
DQN agent variant that acts on RelearnChemicalEnv when its observation is a
*set* of num_cells cell states (shape [S, D]) rather than a single vector --
see envs/small_molecules.py's _load_dmso_control_pool/_draw_cell_set. Same
training loop as agents/dqn.py; the only difference is the network, which
must consume an exchangeable set instead of a flat vector.

Action space: 1138 Tahoe-100M small molecule perturbations.

Run with: python src/relearn/agents/dqn_set.py [agent=<name>] [env=<name>] [key=value ...]
See configs/config.yaml and README.md for the config system.
"""

import math
import random
from collections import namedtuple, deque
from itertools import count

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from relearn.envs.small_molecules import RelearnChemicalEnv

import wandb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNSet(nn.Module):
    """
    Deep-Sets-style Q-network: a per-cell encoder with weights shared across
    every cell in the set (the cells are exchangeable, so the network must
    not depend on their order), mean-pooled into one summary vector, then the
    same shape of head as the single-vector DQN.
    """

    def __init__(self, n_features, n_actions, hidden_dim=128):
        super(DQNSet, self).__init__()

        # per-cell encoder -- applied identically to every one of the S cells
        self.cell_layer1 = nn.Linear(n_features, hidden_dim)
        self.cell_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Q-head over the pooled set summary
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        """
        x: (batch, S, n_features) -- a set of S cells per batch element.
        Returns tensor of shape (batch, n_actions), one Q-value per drug.
        """
        x = F.relu(self.cell_layer1(x))
        x = F.relu(self.cell_layer2(x))  # (batch, S, hidden_dim)
        x = x.mean(dim=1)                # mean-pool over the set -- permutation invariant, any S
        return self.layer3(x)


episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    agent_cfg = cfg.agent

    env = RelearnChemicalEnv(cfg.env)

    # setup device, since no gpu at this moment let's use cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # set random seeds for reproducibility
    random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    env.reset(seed=agent_cfg.seed)
    env.action_space.seed(agent_cfg.seed)
    env.observation_space.seed(agent_cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(agent_cfg.seed)

    print(f"{device=}")
    print(OmegaConf.to_yaml(cfg))

    resolved_config = dict(OmegaConf.to_container(cfg, resolve=True))
    wandb.init(
        project="relearn-dqn",
        group=cfg.experiment,
        name=cfg.run_id,
        notes=cfg.description,
        config={**resolved_config, "device": str(device)},
    )
    hydra_dir = HydraConfig.get().runtime.output_dir
    wandb.save(f"{hydra_dir}/.hydra/*.yaml", base_path=hydra_dir)

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get number of per-cell features -- observation is (num_cells, n_features),
    # so len(state) would give num_cells, not the feature dim
    state, info = env.reset()
    n_features = state.shape[-1]

    # instantiate our model
    policy_net = DQNSet(n_features, n_actions).to(device)
    target_net = DQNSet(n_features, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=agent_cfg.lr, amsgrad=True)
    memory = ReplayMemory(agent_cfg.replay_capacity)

    steps_done = 0

    dmso_action = env.drug_list.index(env.cfg.dmso_control_pert)

    def select_action(state):
        """Epsilon-greedy over the pooled Q-network. state: (1, S, n_features)."""
        nonlocal steps_done

        sample = random.random()
        eps_threshold = agent_cfg.eps_end + (agent_cfg.eps_start - agent_cfg.eps_end) * math.exp(-1. * steps_done / agent_cfg.eps_decay)

        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                pred = policy_net(state)
                max_elem = pred.max(1)
                return max_elem.indices.view(1, 1)
        else:
            return torch.tensor([
                [env.action_space.sample()]], device=device, dtype=torch.long
            )

    def optimize_model():
        if len(memory) < agent_cfg.batch_size:
            return

        transitions = memory.sample(agent_cfg.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)      # (batch, S, n_features)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(agent_cfg.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * agent_cfg.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        return loss.item()

    # training loop
    # each RelearnChemicalEnv episode is exactly one transition (truncated=True every step,
    # see env.py's one-step-horizon TODO), so agent_cfg.num_episodes here is really a transition count.
    # it must clear agent_cfg.batch_size or optimize_model() never runs.

    for i_episode in range(agent_cfg.num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, S, n_features)
        episode_reward = 0.0
        for t in count():
            forced = agent_cfg.forced_first_action if t == 0 else agent_cfg.forced_second_action
            if forced is None:
                action = select_action(state)
            else:
                action = torch.tensor([[dmso_action]], device=device, dtype=torch.long)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)

            state = next_state

            loss = optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent_cfg.tau + target_net_state_dict[key]*(1-agent_cfg.tau)
            target_net.load_state_dict(target_net_state_dict)

            eps_threshold = agent_cfg.eps_end + (agent_cfg.eps_start - agent_cfg.eps_end) * math.exp(-1. * steps_done / agent_cfg.eps_decay)
            wandb.log({
                "loss": loss,
                "epsilon": eps_threshold,
            }, step=steps_done)

            if done:
                episode_durations.append(t+1)
                plot_durations()
                wandb.log({
                    "episode": i_episode,
                    "episode_duration": t + 1,
                    "episode_reward": episode_reward,
                }, step=steps_done)
                break

    print('Complete')
    plot_durations(show_result=True)
    wandb.finish()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
