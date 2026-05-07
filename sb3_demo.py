import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Create PPO agent (On-Policy)
model = PPO("MlpPolicy", env, verbose=1)

# Train for a short time
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Test the trained agent
env = gym.make("CartPole-v1")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()