import os
from typing import List, Tuple

import gym

from src.constants import DDPG_AGENT_DEFAULT_CONFIG
from src.ddpg_agent import DDPGAgent
from src.train import play_episode

# merge default config with custom config
config = {**DDPG_AGENT_DEFAULT_CONFIG, "eval_episodes": 20}

env = gym.make("WindyCar-v0", render_mode="rgb_array")
agent = DDPGAgent(
    action_space=env.action_space, observation_space=env.observation_space, **config
)
try:
    agent.restore(os.path.join("../checkpoints/best_success.pt"))
except:
    raise ValueError("Could not find model to load")

eval_returns_all = []
eval_times_all = []

eval_returns, success_rate = 0, 0
for ep_idx in range(config["eval_episodes"]):
    _, episode_return, _, success = play_episode(
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
    success_rate += int(success) / config["eval_episodes"]

print(f"Mean return {eval_returns}, success rate {round(100 * success_rate)}%")
env.close()
