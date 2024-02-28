import gym

from src.constants import DDPG_AGENT_DEFAULT_CONFIG
from src.train import train_agent

env = gym.make("WindyCar-v0")
_ = train_agent(env, DDPG_AGENT_DEFAULT_CONFIG)
