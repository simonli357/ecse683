cmake_minimum_required(VERSION 3.0.2)
project(planning)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  roscpp
  rospy
  cv_bridge
  sensor_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
  grid_map_cv
  grid_map_costmap_2d
  grid_map_sdf
  grid_map_pcl
  grid_map_octomap
  mapping
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS rospy roscpp cv_bridge sensor_msgs geometry_msgs std_msgs grid_map_core grid_map_ros grid_map_msgs
  DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(astar 
  src/Planner.cpp
  src/AStarPlanner.cpp
  src/a_star_planner_node.cpp
)
target_link_libraries(astar ${catkin_LIBRARIES})

add_executable(controller 
  src/Planner.cpp
  src/AStarPlanner.cpp
  src/controller_node.cpp
)
target_link_libraries(planner ${catkin_LIBRARIES})