import gymnasium as gym

# Create a simple environment
env = gym.make("CartPole-v1")

# Reset to get initial observation
obs, info = env.reset()

print("Initial observation:", obs)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Take a random action
action = env.action_space.sample()
print("Random action:", action)

# Step the environment
next_obs, reward, terminated, truncated, info = env.step(action)
print("Next observation:", next_obs)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)

env.close()