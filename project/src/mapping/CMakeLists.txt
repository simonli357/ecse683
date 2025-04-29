cmake_minimum_required(VERSION 3.0.2)
project(mapping)

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
)

set(LIB_SOURCES
  src/TerrainMap.cpp
)

add_library(${PROJECT_NAME} ${LIB_SOURCES})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rospy roscpp cv_bridge sensor_msgs geometry_msgs std_msgs grid_map_core grid_map_ros grid_map_msgs
  DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(TerrainMap src/TerrainMapOld.cpp)
target_link_libraries(TerrainMap ${catkin_LIBRARIES})

add_executable(ImageToCost src/ImageToCost_node.cpp)
target_link_libraries(ImageToCost ${catkin_LIBRARIES})

add_executable(main 
  src/TerrainMap.cpp
  src/main.cpp
)
target_link_libraries(main ${catkin_LIBRARIES})

########################################
## Installation
########################################

# Install the library:
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Install only the public headers under `include/my_package`
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)