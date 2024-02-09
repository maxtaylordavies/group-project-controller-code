import gym
from gym import spaces
import numpy as np
import pygame

from src.constants import (
    MAX_VELOCITY,
    MAX_ACCEL,
    MAX_ANGULAR_V,
    MAX_WHEEL_ANGLE,
    MAX_WIND,
    MAX_WIND_CHANGE,
)


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=6, length=24):
        self.width, self.length = width, length
        self.window_size = (
            512,
            int(512 * (width / length)),
        )

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
                "wheel_angle": spaces.Box(
                    low=-MAX_WHEEL_ANGLE, high=MAX_WHEEL_ANGLE, shape=(1,)
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

    def _get_obs(self):
        return {
            "car_p": self._car_pos,
            "car_v": self._car_vel,
            "wheel_angle": self._wheel_angle,
            "wind": self._wind,
        }

    def reset(self, seed=None, options=None):
        # seed self.np_random
        super().reset(seed=seed)

        # initialise car position and velocity at track start
        self._car_pos = np.array([0, 0])
        self._car_vel = np.array([0, 0])
        self._wheel_angle = 0

        # sample random initial wind speed
        self._wind = self.np_random.uniform(-MAX_WIND, MAX_WIND)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _update_state(self, action):
        # first, update p_t based on v_t and wind_t
        delta_p = self.dt * (self._car_vel + np.array([0, self._wind]))
        self._car_pos += delta_p

        # then, update v_t based on acceleration and current wheel angle
        delta_v = self.dt * np.array([action[0], np.sin(self._wheel_angle)])
        self._car_vel += delta_v

        # update wheel angle based on angular velocity
        self._wheel_angle += self.dt * action[1]

        # finally, update wind speed based on random perturbation
        self._wind += self.np_random.uniform(-MAX_WIND_CHANGE, MAX_WIND_CHANGE)

    def _compute_reward_and_terminated(self):
        # reward is negative distance to goal
        r = -np.linalg.norm(self._car_pos - np.array([self.length, 0]))

        # terminate if car goes off track or if it reaches the goal
        terminated = (
            self._car_pos[0] < 0
            or self._car_pos[0] > self.length
            or abs(self._car_pos[1]) > self.width / 2
        )

        return r, terminated

    def step(self, action):
        self._update_state(action)

        observation = self._get_obs()
        info = self._get_info()

        reward, terminated = self._compute_reward_and_terminated()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_size = self.window_size[0] / self.length

        # draw track
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                0,
                self.window_size[1] / 2 - self.width / 2,
                self.window_size[0],
                self.width,
            ),
        )

        # draw car
        car_x = int(self._car_pos[0] * pix_size)
        car_y = int(self.window_size[1] / 2 - self._car_pos[1] * pix_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
