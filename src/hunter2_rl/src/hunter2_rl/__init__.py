from gym.envs.registration import register

register(
        id='Hunter2Maze-v0',
        entry_point='hunter2_rl.task_envs.hunter2_maze:Hunter2MazeEnv'
)