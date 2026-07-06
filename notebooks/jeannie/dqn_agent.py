"""
Practice building a DQN agent that acts on the CartPole-v1 environment.
Action space: left and right.
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1") # use the basic cartpole env to mock the dqn

# setup device, since no gpu at this moment let's use cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# set random seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(f"{device=}")

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
        during optimization. Returns tensor([
        [left0exp, right0exp]...
        ])"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# let's define our hyperparameters and utilities

BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor as mentioned in the previous section -- this discounts the future reward
EPS_START = 0.9 # the starting value of epsilon for the exploration/exploitation
EPS_END = 0.01 # the end value of epsilon
EPS_DECAY = 2500 # controls the rate of exponential decay of epsilon. higher means a slower decay
TAU = 0.005 # the update rate of the target network; how often is the target network updated?
LR = 3e-4 # the learning rate of the AdamW optimizer

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

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# initialize the replay memory
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    """Here you can implement any action selection algorithm of your choice. For us, we're sticking with epsilon greedy for now."""
    pass