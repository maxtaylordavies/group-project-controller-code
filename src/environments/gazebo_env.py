import gazebo
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import rospy

from src.gazebo import GazeboConnection, ControllersConnection, RLExperimentInfo


class GazeboEnvironment(gym.Env):
    def __init__(
        self,
        robot_name_space,
        controllers_list,
        reset_controls,
        width=6,
        length=30,
        start_init_physics_parameters=True,
        reset_world_or_sim="SIMULATION",
    ):
        rospy.logdebug("START INIT")

        self.robot_name_space = robot_name_space
        self.controllers_list = controllers_list
        self.reset_controls = reset_controls
        self.episode_num = 0
        self.episode_reward = 0

        # define observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "car_p": spaces.Box(
                    low=np.array([0, -width / 2]),
                    high=np.array([length, width / 2]),
                    shape=(2,),
                    dtype=np.float32,
                ),
            #     "car_v": spaces.Box(
            #         low=np.array([0, -MAX_VELOCITY]),  # car can't move backwards
            #         high=np.array([MAX_VELOCITY, MAX_VELOCITY]),
            #         shape=(2,),
            #         dtype=np.float32,
            #     ),
            #     "wheel_angle": spaces.Box(
            #         low=-MAX_WHEEL_ANGLE,
            #         high=MAX_WHEEL_ANGLE,
            #         shape=(1,),
            #         dtype=np.float32,
            #     ),
            #     "wind": spaces.Box(
            #         low=-MAX_WIND, high=MAX_WIND, shape=(1,), dtype=np.float32
            #     ),
            #     "obstacle": spaces.Box(
            #         low=np.array([0, -width / 2]),
            #         high=np.array([length, width / 2]),
            #         shape=(2,),
            #         dtype=np.float32,
            #     ),
            # }
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
        )

        # initialise gazebo and controllers
        self.gazebo = GazeboConnection(
            start_init_physics_parameters, reset_world_or_sim
        )
        self.controllers = ControllersConnection(
            namespace=robot_name_space, controllers_list=controllers_list
        )

        # set seed
        self.seed()

        # wait for sensors to come online
        self.gazebo.unpauseSim()
        self._wait_for_sensors_ready()

        # set up all ROS subscribers and publishers
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/imu", Imu, self._imu_callback)
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._reward_pub = rospy.Publisher(
            "/openai/reward", RLExperimentInfo, queue_size=1
        )

        # wait for publishers to connect
        self._wait_for_publishers_ready()
        self.gazebo.pauseSim()

        rospy.logdebug("END INIT")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        rospy.logdebug("START RESET")
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        rospy.logdebug("END RESET")
        return obs

    def step(self, action):
        rospy.logdebug("START STEP")

        # take action in environment and update state
        self.gazebo.unpauseSim()
        self._take_action(action)
        self.gazebo.pauseSim()

        # get next observation, reward, and termination status
        obs, info = self._get_obs(), self._get_info()
        reward, terminated = self._compute_reward_and_terminated()

        self.episode_reward += reward

        rospy.logdebug("END STEP")
        return obs, reward, terminated, False, info

    def close(self):
        raise NotImplementedError

    def _reset_sim(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _scale_action(self, action):
        # return action * np.array([MAX_ACCEL, MAX_ANGULAR_V], dtype=np.float32)
        accel = ((action[0] + 1.0) / 2.0) * MAX_ACCEL  # scale [-1,1] to [0, MAX_ACCEL]
        angular_v = (
            action[1] * MAX_ANGULAR_V
        )  # scale [-1,1] to [-MAX_ANGULAR_V, MAX_ANGULAR_V]
        return np.array([accel, angular_v], dtype=np.float32)

    def _take_action(self, action, tolerance=0.05, update_rate=10):
        # scale action
        action = self._scale_action(action)

        # convert action values to Twist message
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]

        # publish command and wait for it to be executed
        self._cmd_vel_pub.publish(cmd)
        rate, start = rospy.Rate(update_rate), rospy.get_rostime().to_sec()
        while not rospy.is_shutdown():
            odom = self._check_odom_ready()
            linear_delta = np.abs(odom.twist.twist.linear.x - cmd.linear.x)
            angular_delta = np.abs(odom.twist.twist.angular.z - cmd.angular.z)
            if linear_delta < tolerance and angular_delta < tolerance:
                break
            rate.sleep()
        rospy.logdebug(f"Wait Time={rospy.get_rostime().to_sec() - start} seconds")

    def _compute_reward_and_terminated(self):
        raise NotImplementedError

    def _wait_for_sensors_ready(self):
        self._check_odom_ready()
        self._check_imu_ready()
        self._check_laser_scan_ready()

    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /odom to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
                rospy.logdebug("Current /odom READY=>")
            except:
                rospy.logerr("Current /odom not ready yet, retrying for getting odom")
        return self.odom

    def _check_imu_ready(self):
        self.imu = None
        rospy.logdebug("Waiting for /imu to be READY...")
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu", Imu, timeout=5.0)
                rospy.logdebug("Current /imu READY=>")
            except:
                rospy.logerr("Current /imu not ready yet, retrying for getting imu")
        return self.imu

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(
                    "/scan", LaserScan, timeout=5.0
                )
                rospy.logdebug("Current /scan READY=>")
            except:
                rospy.logerr(
                    "Current /scan not ready yet, retrying for getting laser_scan"
                )
        return self.laser_scan


    def _wait_for_publishers_ready(self):
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")

    def _odom_callback(self, data):
        self.odom = data

    def _imu_callback(self, data):
        self.imu = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data
