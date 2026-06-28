from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime

# my goal here is to write 3 functions to train a Q-learning agent to learn how to play Blackjack.
# funcs required: 1) choosing actions, 2) learning from experience, 3) managing exploration

# we want to create the Blackjack environment first

class BlackjackAgent:
    def __init__(
            self,
            env: gym.Env, # needs to know what is the experience environment
            learning_rate: float, # how much will the agent learn from each step
            initial_epsilon: float, # defines exploration:exploitation ratio
            epsilon_decay: float, # defines how the epsilon greedy strat will change over steps
            final_epsilon: float, # final goal for epsilon greedy strat
            discount_factor: float = 0.95 # how much we will allow future rewards to affect the policy
    ):
        """
        Initialize a Q-learning agent.

        Args:
            env: Training env
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # define the Q-table which maps (state, action) to expected reward
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) # for any state, there would be a 2d entry that stores the expected reward if either action were taken

        # bellman equation parameters
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # track learning progress 
        self.training_error = []

    def get_action(
            self,
            obs: tuple[int, int, bool]
    ) -> int:
        """
        Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # with probability epsilon, choose to explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability 1-epsilon, choose the greedy option
        else:
            # look into the q-table and extract the best action out of the two stored values
            return int(np.argmax(self.q_values[obs]))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool]
    ):
        """
        Update the Q-value based on experience.

        Heart of Q_learning, learn from (state, action, reward, next_state)
        """
        # what's the best we could do from the next state?
        # if terminated, then reward is 0
        # optimism baked into q-learning, assuming that all other steps from here will be acted optimally.
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # what should the q-value be? calc using the bellman eqn
        # this is the better informed estimate, what the agent sees after what happened
        # discount factor makes the future reward worth slightly less than the current reward
        target = reward + self.discount_factor * future_q_value

        # how wrong was our current estimate, compared to the better informed target?
        # this is the heart of q-learning. the gap between two estimates of the same thing, made at two different moments of time. this is the error signal the agent learns from.
        # positive TD -> the outcome was better than what i expected -> bump q-val up.
        # negative TD -> the outcome was worse than what i expected -> bump q-val down.
        # almost 0 -> my estimate was good -> barely change.
        # this is what makes q-learning sample efficient -- you are learning after every step, not just updating once at the end of the trajectory.
        temporal_difference = target - self.q_values[obs][action]

        # update our estimates in the direction of the error
        # learning rate controls how big the steps we take are
        # move the old estimate a fraction (lr) of the way toward the target.
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # track learning progress, expect this to decrease over steps if learning occurs
        self.training_error.append(temporal_difference)

    def decay_epsilon(
        self
    ):
        """
        Reduce exploration rate after each episode.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# helper for visualization
def get_moving_avgs(arr, window, convolution_mode):
    """
    Compute moving average to smooth noisy data.
    """
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def test_agent(agent, env, num_episodes=1000):
    """
    Test agent performance without learning or exploration.
    """
    total_rewards=[]

    # temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 # purely exploit from what's been learned during training

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)

    # restore old epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


# now that we have created the necessary steps for the agent, let's train one.
# process:
# reset the env to start a new episode
# play one complete hand (episode), choosing actions and learning from each step
# update the exploration rate (reduce epsilon)
# repeat for many episodes

def main():
    # set training hyperparameters
    learning_rate = 0.01 # how fast to learn (higher = faster but less stable)
    n_episodes = 100_000 # number of hands to practice
    start_epsilon = 1.0 # start with complete exploration, random actions
    epsilon_decay = start_epsilon / (n_episodes / 2) # reduce exploration over time
    final_epsilon = 0.1 # always keep some exploration

    # create environment and agent
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # training loop
    for episode in tqdm(range(n_episodes)):
        # start a new hand
        obs, info = env.reset()
        done = False

        # play a complete hand
        while not done:
            # agent chooses an action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # take action, observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # move to next state
            done = terminated or truncated
            obs = next_obs
        
        # reduce exploration rate (agent become less random over time)
        agent.decay_epsilon()

    # visualize outcome

    # smooth over a 500-epsiode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    now = datetime.now()
    plt.savefig(f"outputs/blackjack_qlearning_{now:%m%d-%H:%M}")

    # test agent behavior
    test_agent(agent, env)

if __name__ == "__main__":
    main()