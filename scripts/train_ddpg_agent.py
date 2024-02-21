import gym

from src.train import train_agent

config = {
    "eval_freq": 2000,
    "eval_episodes": 10,
    "policy_learning_rate": 5e-3,
    "critic_learning_rate": 5e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "tau": 0.01,
    "batch_size": 128,
    "buffer_capacity": int(1e6),
    "episode_length": 200,
    "max_timesteps": 100000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "car_latest.pt",
}

env = gym.make("WindyCar-v0")
_ = train_agent(env, config)
