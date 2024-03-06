from gym.envs.registration import register

register(
    id="WindyCar-v0",
    entry_point="src.environments.basic_env:BasicEnvironment",
    max_episode_steps=1000,
)
