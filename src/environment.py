import gym
from gym import spaces
import numpy as np

from .constants import MAX_VELOCITY, MAX_WIND, MAX_ACCEL, MAX_ANGULAR_V


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=6, length=20):
        self.width, self.length = width, length

        """
        define observation space as consisting of the car's current (continuous)
        position and velocity, the obstacle's position, and the current wind speed.
        """
        self.observation_space = spaces.Dict(
            {
                "car_p": spaces.Box(
                    low=np.array([0, -width / 2]),
                    high=np.array([length, width / 2]),
                    shape=(2,),
                ),
                "car_v": spaces.Box(
                    low=np.array([0, -MAX_VELOCITY]),  # car can't move backwards
                    high=np.array([MAX_VELOCITY, MAX_VELOCITY]),
                    shape=(2,),
                ),
                "wind": spaces.Box(low=-MAX_WIND, high=MAX_WIND, shape=(1,)),
            }
        )

        """
        define action space as consisting of continuous forward acceleration
        and change in steering angle (i.e. angular velocity)
        """
        self.action_space = spaces.Box(
            low=np.array([0, -MAX_ANGULAR_V]),
            high=np.array([MAX_ACCEL, MAX_ANGULAR_V]),
            shape=(2,),
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        if human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. they will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
