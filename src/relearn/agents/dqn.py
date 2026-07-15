"""
Building a DQN agent that acts on the RelearnChemicalEnv.
Action space: 1138 Tahoe-100M small molecule perturbations.

Run with: python src/relearn/agents/dqn.py [agent=<name>] [env=<name>] [key=value ...]
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

from relearn.envs.single_step import RelearnChemicalEnv

import wandb

# follow the pseudo code

"""
initialize replay memory D to capacity N # replay memory helps with sample efficiency
initialize action-value function Q with random weights $\theta$
initialize target action-value function Qhat with weights theta_minus = theta

for episode in range(1, M):
	initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)
	for t in range(1, T):
		# sampling phase
		with probability epsilon select a random action a_t # exploration / exploitation argument here
		otherwise select a_t = argmax_a of Q(phi(s_t), a; theta)
		execute action a_t in emulator and observe reward r_t and image x_(t+1) # actually take the step here
		set s_(t+1) = s_t, a_t, x_(t+1) and preprocess phi_(t+1) = phi(s_(t+1)) # move the state forward
		store transition (phi_t, a_t, r_t, phi_(t+1)) in D # store in the replay memory

		# training phase
		sample a random minibatch of transitions (phi_j, a_j, r_j, phi_(j+1)) from D
		set y_j (the q-target) to be r_j if the episode terminates at step j+1 # terminal step; no further reward
		otherwise, set y_j to be the td-target # you should learn from the estimate of the optimal q-value of the next state
		perform a GD step on the MSE (td error)**2 wrt to the network parameters theta.
		every C steps reset Qhat to Q

"""

# let's set up the replay memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque allows us to quickly push and pop from both ends. useful for cyclic buffer

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# let's now define our model

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        # define a simple, 3 layer NN with ReLU nonlinearities
        # goal: take in the n_observations, and try to learn about them to produce the next action
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        """This method is called either with one element to determine the next action, or a batch
        during optimization. Returns tensor of shape (batch, n_actions) with one Q-value per drug."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
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

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())


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
        config={**resolved_config, "device": str(device)},
    )
    # Hydra already wrote the fully-resolved config and exact CLI overrides for
    # this run to its own output dir -- attach them so the wandb run is
    # self-describing without duplicating that bookkeeping here.
    hydra_dir = HydraConfig.get().runtime.output_dir
    wandb.save(f"{hydra_dir}/.hydra/*.yaml", base_path=hydra_dir)

    # get number of actions from gym action space
    n_actions = env.action_space.n

    # get number of state observations
    state, info = env.reset()
    n_observations = len(state)

    # instantiate our model
    # use a double dqn setup to reduce the correlation between the chosen next action and the calculation of that action's estimated q value
    policy_net = DQN(n_observations, n_actions).to(device) # this begins with completely randomized weights theta
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict()) # initialize the target action-value func Qhat with weights theta_minus = theta

    optimizer = optim.AdamW(policy_net.parameters(), lr=agent_cfg.lr, amsgrad=True)
    # initialize the replay memory
    memory = ReplayMemory(agent_cfg.replay_capacity)

    steps_done = 0

    def select_action(state):
        """Here you can implement any action selection algorithm of your choice. For us, we're sticking with epsilon greedy for now."""
        nonlocal steps_done

        # determine if we should be exploiting or exploring here
        sample = random.random()
        eps_threshold = agent_cfg.eps_end + (agent_cfg.eps_start - agent_cfg.eps_end) * math.exp(-1. * steps_done / agent_cfg.eps_decay)

        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row
                # second column on max result is index of where max element was found
                # pick the action with the larger expected reward
                # return policy_net(state).max(1).indices.view(1, 1)

                # apply the policy network on the state
                pred = policy_net(state)
                # grab the largest column values of each row (each of the biggest) ??? # TODO what do the rows and columns mean for the state?
                max_elem = pred.max(1)
                # grab the indices and reshape
                return max_elem.indices.view(1, 1)

        else:
            return torch.tensor([
                [env.action_space.sample()]], device=device, dtype=torch.long
            )

    # define an optimize_model func that performs a single step of the optimization. it samples a batch, concatenates all the tensors into a single one, computes Q(s, a) and V(s_(t+1)) = max_a Q(s_(t+1), a) and combines them into the loss.
    # by definition, if s is a terminal state, we set V(s) == 0.
    # we use a target network to compute V(s_(t+1)) for added stability

    # the target network is updated at every step with a soft update controlled by agent_cfg.tau.
    def optimize_model():
        if len(memory) < agent_cfg.batch_size:
            return # only optimize the model (update the target parameters) if you know that there has been enough experience "collected"; i.e. until the replay buffer has enough samples to form a full batch

        # pull a random minibatch of Transition tuples. this is the point of a replay buffer, to decorrelate sequential experiences
        transitions = memory.sample(agent_cfg.batch_size)

        batch = Transition(*zip(*transitions))

        # compute a mask of non-final states. concatenate with the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute Q(s_t, a) the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(agent_cfg.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

        # compute the expected Q values
        expected_state_action_values = (next_state_values * agent_cfg.gamma) + reward_batch

        # compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        optimizer.zero_grad()
        loss.backward()

        # in-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        return loss.item()

    # training loop
    # each RelearnChemicalEnv episode is exactly one transition (truncated=True every step,
    # see env.py's one-step-horizon TODO), so agent_cfg.num_episodes here is really a transition count.
    # it must clear agent_cfg.batch_size or optimize_model() never runs.

    for i_episode in range(agent_cfg.num_episodes):
        # initialize env, get the state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            reward = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state
            state = next_state

            # perform one step of the optimization (on the policy network)
            loss = optimize_model()

            # soft update of the target network's weights
            # theta prime = tau * theta + (1 - tau) * theta prime
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
