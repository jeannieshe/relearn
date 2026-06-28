import gymnasium as gym

# create the training environment
env = gym.make("CartPole-v1") #render_mode is how the environment is visualized, which can be human, rgb_array, or none (fastest for training)
print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

# reset the environment to start a new episode
# this is like starting a new game or episode
observation, info = env.reset()
# the observation is what the agent can "see": cart position, velocity, pole angle
# the info is extra debugging info (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# outputs: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # choose an action: 0 = left, 1 = right
    action = env.action_space.sample() # random action for now

    # take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: true if pole falls too far (agent failed) or if the agent succeeded
    # truncated: true if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()