#include "Planner.hpp"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/GridMapRosConverter.hpp>

std::vector<grid_map::Index> Planner::planFromPosition(const grid_map::GridMap& gridMap,
                                                       const std::string& layerName,
                                                       const grid_map::Position& startPos,
                                                       const grid_map::Position& goalPos)
{
  grid_map::Index startIndex, goalIndex;

  // Convert world positions to grid indices
  bool gotStart = gridMap.getIndex(startPos, startIndex);
  bool gotGoal  = gridMap.getIndex(goalPos, goalIndex);

  if (!gotStart) {
    ROS_WARN("Planner::planFromPosition() - Could not convert startPos to an index!");
    return {};
  }
  if (!gotGoal) {
    ROS_WARN("Planner::planFromPosition() - Could not convert goalPos to an index!");
    return {};
  }

  return plan(gridMap, layerName, startIndex, goalIndex);
}
