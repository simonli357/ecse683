cmake_minimum_required(VERSION 3.0.2)
project(bev)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)