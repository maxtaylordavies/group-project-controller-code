import gym
from gym import spaces
import numpy as np
import rospy

from src.gazebo import GazeboConnection, ControllersConnection, RLExperimentInfo


class GazeboEnvironment(gym.Env):
    def __init__(
        self,
        robot_name_space,
        controllers_list,
        reset_controls,
        start_init_physics_parameters=True,
        reset_world_or_sim="SIMULATION",
    ):
        rospy.logdebug("START init RobotGazeboEnv")

        # initialise gazebo and controllers
        self.gazebo = GazeboConnection(
            start_init_physics_parameters, reset_world_or_sim
        )
        self.controllers = ControllersConnection(
            namespace=robot_name_space, controllers_list=controllers_list
        )
        self.reset_controls = reset_controls
        self.seed()

        self.episode_num = 0
        self.episode_reward = 0
        self.reward_pub = rospy.Publisher(
            "/openai/reward", RLExperimentInfo, queue_size=1
        )
        rospy.logdebug("END init RobotGazeboEnv")

    def reset(self):
        self._reset_sim()

    def step(self, action):
        rospy.logdebug("START STEP OpenAIROS")

        # take action in environment and update state
        self.gazebo.unpauseSim()
        self._apply_action_and_update_state(action)
        self.gazebo.pauseSim()

        # get next observation, reward, and termination status
        obs = self._get_obs()
        info = self._get_info()
        reward, terminated = self._compute_reward_and_terminated()

        self.episode_reward += reward

        rospy.logdebug("END STEP OpenAIROS")
        return obs, reward, terminated, False, info

    def close(self):
        raise NotImplementedError

    def _reset_sim(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _scale_action(self, action):
        raise NotImplementedError

    def _apply_action_and_update_state(self, action):
        raise NotImplementedError

    def _compute_reward_and_terminated(self):
        raise NotImplementedError
