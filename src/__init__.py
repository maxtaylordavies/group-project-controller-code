from gym.envs.registration import register

register(
    id="WindyCar-v0",
    entry_point="src.environment:Environment",
    max_episode_steps=1000,
)
