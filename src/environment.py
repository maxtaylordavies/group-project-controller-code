import cv2
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
    SUCCESS_REWARD,
    FAILURE_PENALTY,
    STATIONARY_STEPS_THRESHOLD,
    GOAL_DISTANCE_COEFF,
    WALL_DISTANCE_COEFF,
    OBSTACLE_WIDTH,
    OBSTACLE_HEIGHT,
    OBSTACLE_THRESHOLD,
)


class Environment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=6, length=30, dt=0.2):
        self.width, self.length, self.dt = width, length, dt
        self.window_size = (
            512,
            int(512 * ((width * 2) / length)),
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
                    dtype=np.float32,
                ),
                "car_v": spaces.Box(
                    low=np.array([0, -MAX_VELOCITY]),  # car can't move backwards
                    high=np.array([MAX_VELOCITY, MAX_VELOCITY]),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "wheel_angle": spaces.Box(
                    low=-MAX_WHEEL_ANGLE,
                    high=MAX_WHEEL_ANGLE,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "wind": spaces.Box(
                    low=-MAX_WIND, high=MAX_WIND, shape=(1,), dtype=np.float32
                ),
                "obstacle": spaces.Box(
                    low=np.array([0, -width / 2]),
                    high=np.array([length, width / 2]),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        """
        define action space as consisting of continuous forward acceleration
        and change in steering angle (i.e. angular velocity)
        """
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.recording = False

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
            "wheel_angle": np.array([self._wheel_angle], dtype=np.float32),
            "wind": np.array([self._wind], dtype=np.float32),
            "obstacle": np.array(self._obstacles[0], dtype=np.float32),
        }

    def _get_info(self):
        return {}

    def _scale_action(self, action):
        # return action * np.array([MAX_ACCEL, MAX_ANGULAR_V], dtype=np.float32)
        accel = ((action[0] + 1.0) / 2.0) * MAX_ACCEL  # scale [-1,1] to [0, MAX_ACCEL]
        angular_v = (
            action[1] * MAX_ANGULAR_V
        )  # scale [-1,1] to [-MAX_ANGULAR_V, MAX_ANGULAR_V]
        return np.array([accel, angular_v], dtype=np.float32)

    def reset(self, seed=None, options=None):
        # seed self.np_random
        super().reset(seed=seed)

        # initialise car position and velocity at track start
        self._car_pos = np.array([0.0, 0.0], dtype=np.float32)
        self._car_vel = np.array([0.0, 0.0], dtype=np.float32)
        self._wheel_angle = 0.0

        # place obstacles
        obstacle_x = self.np_random.choice([0.25, 0.5, 0.75]) * self.length
        obstacle_y = self.np_random.choice([-0.2, 0.0, 0.2]) * self.width
        self._obstacles = np.array([[obstacle_x, obstacle_y]], dtype=np.float32)

        # sample random initial wind speed
        self._wind = self.np_random.uniform(-MAX_WIND, MAX_WIND)

        # to track how many steps the car has been stationary
        self._prev_x_pos, self._steps_stationary = 0.0, 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _update_state(self, action):
        # scale action to physical units
        action = self._scale_action(action)

        # first, update p_t based on v_t and wind_t
        delta_p = self.dt * (self._car_vel + np.array([0, self._wind]))
        self._car_pos = np.clip(
            self._car_pos + delta_p, [0, -self.width / 2], [self.length, self.width / 2]
        )

        # update stationary steps tracker
        if np.abs(self._car_pos[0] - self._prev_x_pos) < 1e-2:
            self._steps_stationary += 1
        else:
            self._steps_stationary = 0
        self._prev_x_pos = self._car_pos[0]

        # then, update v_t based on acceleration and current wheel angle
        delta_v = self.dt * np.array([action[0], np.sin(self._wheel_angle)])
        self._car_vel = np.clip(
            self._car_vel + delta_v, [0, -MAX_VELOCITY], [MAX_VELOCITY, MAX_VELOCITY]
        )

        # update wheel angle based on angular velocity
        delta_wheel = self.dt * action[1]
        self._wheel_angle = np.clip(
            self._wheel_angle + delta_wheel, -MAX_WHEEL_ANGLE, MAX_WHEEL_ANGLE
        )

        # finally, update wind speed based on random perturbation
        delta_wind = self.dt * self.np_random.uniform(-MAX_WIND_CHANGE, MAX_WIND_CHANGE)
        self._wind = np.clip(self._wind + delta_wind, -MAX_WIND, MAX_WIND)

    def _distance_to_goal(self):
        return np.linalg.norm(self._car_pos - np.array([self.length, 0]))

    def _detect_collision(self):
        for o in self._obstacles:
            x_bounds = o[0] + np.array([-OBSTACLE_WIDTH / 2, OBSTACLE_WIDTH / 2])
            y_bounds = o[1] + np.array([-OBSTACLE_HEIGHT / 2, OBSTACLE_HEIGHT / 2])
            if np.all(
                [
                    x_bounds[0] - self._car_pos[0] <= OBSTACLE_THRESHOLD,
                    self._car_pos[0] - x_bounds[1] <= OBSTACLE_THRESHOLD,
                    y_bounds[0] - self._car_pos[1] <= OBSTACLE_THRESHOLD,
                    self._car_pos[1] - y_bounds[1] <= OBSTACLE_THRESHOLD,
                ]
            ):
                return True

    def _compute_reward_and_terminated(self):
        # terminate if car reaches the goal, goes off track, collides with an obstacle or is stationary for too long
        if self._car_pos[0] >= self.length:
            return SUCCESS_REWARD, True
        elif abs(self._car_pos[1]) >= self.width / 2:
            return -FAILURE_PENALTY, True
        elif self._detect_collision():
            return -FAILURE_PENALTY, True
        elif self._steps_stationary >= STATIONARY_STEPS_THRESHOLD:
            return -FAILURE_PENALTY, True

        # base reward is some multiple of negative distance to goal
        r = GOAL_DISTANCE_COEFF * -self._distance_to_goal() / self.length

        # penalise distance from track centre
        r -= WALL_DISTANCE_COEFF * np.abs(self._car_pos[1])

        return r, False

    def step(self, action):
        self._update_state(action)

        observation = self._get_obs()
        info = self._get_info()
        reward, terminated = self._compute_reward_and_terminated()

        if self.render_mode == "human":
            self._render_frame()
        if self.recording:
            self._record_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_size = self.window_size[0] / self.length

        # draw track
        track_width = pix_size * (self.width + 2)
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                0,
                self.window_size[1] / 2 - track_width / 2,
                self.window_size[0],
                track_width,
            ),
        )

        # draw obstacles
        for o in self._obstacles:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    int(o[0] * pix_size),
                    int(self.window_size[1] / 2 - o[1] * pix_size),
                    int(pix_size * OBSTACLE_WIDTH),
                    int(pix_size * OBSTACLE_HEIGHT),
                ),
            )

        # render car from png file
        car_x = int(self._car_pos[0] * pix_size)
        car_y = int(self.window_size[1] / 2 - self._car_pos[1] * pix_size)
        car_img = pygame.image.load("../car.png")
        car_img = pygame.transform.scale(car_img, (2 * int(pix_size), int(pix_size)))
        car_img = pygame.transform.rotate(car_img, np.degrees(self._wheel_angle))
        canvas.blit(car_img, (car_x, car_y))

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

    def init_recording(self, filepath):
        size = self.render().shape[:2][::-1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(filepath, fourcc, 30, size)
        self.recording = True

    def _record_frame(self):
        if not self.recording:
            return
        frame = self.render()
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def finish_recording(self):
        self._writer.release()
        self.recording = False

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
