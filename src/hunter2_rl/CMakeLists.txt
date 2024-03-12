cmake_minimum_required(VERSION 2.8.3)
project(hunter2_rl)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

catkin_python_setup()


add_message_files(
   FILES
   RLExperimentInfo.msg
 )


generate_messages(
   DEPENDENCIES
   std_msgs
 )

catkin_package(
  DEPENDS message_runtime rospy gazebo_msgs std_msgs geometry_msgs controller_manager_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
  src/start_ddpg_wall.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)