import os
from typing import List, Tuple

import gym

from src.ddpg_agent import DDPGAgent
from src.train import play_episode

config = {
    "eval_freq": 1000,
    "eval_episodes": 10,
    "policy_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "tau": 0.01,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "target_return": 0,
    "episode_length": 200,
    "max_timesteps": 100000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "car_latest.pt",
}

env = gym.make("WindyCar-v0", render_mode="rgb_array")
agent = DDPGAgent(
    action_space=env.action_space, observation_space=env.observation_space, **config
)
try:
    agent.restore(os.path.join("../checkpoints/best.pt"))
except:
    raise ValueError("Could not find model to load")

eval_returns_all = []
eval_times_all = []

eval_returns = 0
for ep_idx in range(config["eval_episodes"]):
    _, episode_return, _ = play_episode(
        env,
        agent,
        0,
        train=False,
        explore=False,
        render=False,
        record_fp=f"../recordings/{ep_idx}.mp4",
        max_steps=config["episode_length"],
        batch_size=config["batch_size"],
    )
    eval_returns += episode_return / config["eval_episodes"]

print(eval_returns)
env.close()
