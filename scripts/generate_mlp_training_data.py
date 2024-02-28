import os
from typing import List, Tuple

import gym
import h5py
from tqdm import tqdm

from src.constants import DDPG_AGENT_DEFAULT_CONFIG
from src.ddpg_agent import DDPGAgent
from src.replay import ReplayBuffer, load_saved_replay_data
from src.train import play_episode

# merge default config with custom config
config = {**DDPG_AGENT_DEFAULT_CONFIG, "buffer_capacity": int(1e6)}

# create environment and load agent checkpoint from file
env = gym.make("WindyCar-v0", render_mode="rgb_array")
agent = DDPGAgent(
    action_space=env.action_space, observation_space=env.observation_space, **config
)
try:
    agent.restore(os.path.join("../checkpoints/best_success.pt"))
except:
    raise ValueError("Could not find model to load")

# create replay buffer
replay_buffer = ReplayBuffer(config["buffer_capacity"])

# sample episodes and save to replay buffer
for ep_idx in tqdm(range(1000)):
    _, episode_return, _, success = play_episode(
        env,
        agent,
        replay_buffer,
        train=False,
        save_to_buffer=True,
        explore=False,
        render=False,
        max_steps=config["episode_length"],
    )
env.close()

# save replay buffer to file
replay_buffer.save("../replay_buffer.hdf5")
