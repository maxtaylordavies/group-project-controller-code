import numpy as np

# STATE/ACTION CONSTANTS
# (all values relating to rates of change are in units of smth per second)
MAX_VELOCITY = 2.0
MAX_ACCEL = 0.5
MAX_ANGULAR_V = np.pi / 8
MAX_WHEEL_ANGLE = np.pi / 2
MAX_WIND = 0.1
MAX_WIND_CHANGE = 0.05
OBSTACLE_WIDTH = 1.0
OBSTACLE_HEIGHT = 1.0

# REWARD CONSTANTS
SUCCESS_REWARD = 1000.0
FAILURE_PENALTY = 1000.0
STATIONARY_STEPS_THRESHOLD = 5
GOAL_DISTANCE_COEFF = 1.0
WALL_DISTANCE_COEFF = 1.0
OBSTACLE_THRESHOLD = 0.75

DDPG_AGENT_DEFAULT_CONFIG = {
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
}
