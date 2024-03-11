import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class Hunter2Env(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments."""

    def __init__(self):
        """
        Initializes a new Hunter2Env environment.
        Hunter2 doesnt use controller_manager, therefore we wont reset the
        controllers in the standard fashion. For the moment we wont reset them.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered usefull for AI learning.

        Sensor Topic List:
        * /odom : Odometry readings of the Base of the Robot
        * /camera/depth/image_raw: 2d Depth image of the depth sensor.
        * /camera/depth/points: Pointcloud sensor readings
        * /camera/rgb/image_raw: RGB camera
        * /kobuki/laser/scan: Laser Readings

        Actuators Topic List: /cmd_vel,

        Args:
        """
        rospy.logdebug("Start Hunter2Env INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(Hunter2Env, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=False,
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD",
        )

        self.gazebo.unpauseSim()
        # self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        # rospy.Subscriber("/camera/depth/image_raw", Image, self._camera_depth_image_raw_callback)
        # rospy.Subscriber("/camera/depth/points", PointCloud2, self._camera_depth_points_callback)
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self._camera_rgb_image_raw_callback)
        rospy.Subscriber("/camera/scan", LaserScan, self._laser_scan_callback)

        self._cmd_vel_pub = rospy.Publisher(
            "/ackermann_steering_controller/cmd_vel", Twist, queue_size=1
        )

        self._check_publishers_connection()

        self.gazebo.pauseSim()

        rospy.logdebug("Finished Hunter2Env INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True

    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        self._check_laser_scan_ready()
        rospy.logdebug("ALL SENSORS READY")

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

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /camera/scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message(
                    "/camera/scan", LaserScan, timeout=5.0
                )
                rospy.logdebug("Current /camera/scan READY=>")
            except:
                rospy.logerr(
                    "Current /camera/scan not ready yet, retrying for getting laser_scan"
                )
        return self.laser_scan

    def _odom_callback(self, data):
        self.odom = data

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._cmd_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _cmd_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_cmd_vel_pub Publisher Connected")
        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose"""
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given."""
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(
        self,
        linear_speed,
        angular_speed,
        epsilon=0.05,
        update_rate=10,
        min_laser_distance=-1,
    ):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_speed: Speed in the X axis of the robot base frame
        :param angular_speed: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed
        rospy.logdebug("Hunter2 Base Twist Cmd>>" + str(cmd_vel_value))
        self._check_publishers_connection()
        self._cmd_vel_pub.publish(cmd_vel_value)
        time.sleep(0.2)
        # time.sleep(0.02)
        """
        self.wait_until_twist_achieved(cmd_vel_value,
                                        epsilon,
                                        update_rate,
                                        min_laser_distance)
        """

    def wait_until_twist_achieved(
        self, cmd, epsilon, update_rate, min_laser_distance=-1
    ):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        rate, start = rospy.Rate(update_rate), rospy.get_rostime().to_sec()
        rospy.logwarn("START wait_until_twist_achieved...")
        rospy.logdebug("Desired Twist Cmd>>" + str(cmd))
        rospy.logdebug("epsilon>>" + str(epsilon))

        while not rospy.is_shutdown():
            if self.has_crashed(min_laser_distance):
                rospy.logerr("Hunter2 has crashed, stopping movement!")
                break

            odom = self._check_odom_ready()
            linear_delta = np.abs(odom.twist.twist.linear.x - cmd.linear.x)
            angular_delta = np.abs(odom.twist.twist.angular.z - cmd.angular.z)
            rospy.logdebug(
                f"linear_delta: {linear_delta}, angular_delta: {angular_delta}"
            )

            if linear_delta < epsilon and angular_delta < epsilon:
                rospy.logwarn("Reached Velocity!")
                break

            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()

        delta_time = rospy.get_rostime().to_sec() - start
        rospy.logdebug("[Wait Time=" + str(delta_time) + "]")
        rospy.logwarn("END wait_until_twist_achieved...")

    def has_crashed(self, min_laser_distance):
        """
        It states based on the laser scan if the robot has crashed or not.
        Crashed means that the minimum laser reading is lower than the
        min_laser_distance value given.
        If min_laser_distance == -1, it returns always false, because its the way
        to deactivate this check.
        """
        if min_laser_distance == -1:
            return False

        laser_data = self.get_laser_scan()
        for i, item in enumerate(laser_data.ranges):
            isinf = item == float("Inf") or np.isinf(item)
            if not (isinf or np.isnan(item)) and item < min_laser_distance:
                rospy.logerr(
                    "Hunter2 HAS CRASHED >>> item="
                    + str(item)
                    + "< "
                    + str(min_laser_distance)
                )
                return True

        return False

    def get_odom(self):
        return self.odom

    def get_laser_scan(self):
        return self.laser_scan

    def reinit_sensors(self):
        """
        This method is for the tasks so that when reseting the episode
        the sensors values are forced to be updated with the real data and

        """
