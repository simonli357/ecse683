#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/TypeDefs.hpp>
#include <nav_msgs/Path.h>
#include <grid_map_msgs/GridMap.h>
#include <Eigen/Core>

namespace Utility {
    /**
   * @brief Converts a sequence of grid_map::Index to a nav_msgs::Path.
   * @param pathIndices  The path as a list of grid_map::Index.
   * @param gridMap      The GridMap needed to map grid indices to world coordinates.
   * @param frameId      The frame to use in the Path header (default = "map").
   * @return             The corresponding nav_msgs::Path message.
   */
  nav_msgs::Path toPathMsg(const std::vector<grid_map::Index>& pathIndices,
                                  const grid_map::GridMap& gridMap,
                                  const std::string& frameId)
  {
    nav_msgs::Path pathMsg;
    pathMsg.header.stamp = ros::Time::now();
    pathMsg.header.frame_id = frameId;

    for (const auto& idx : pathIndices) {
      grid_map::Position pos;
      // Convert from grid cell index to world coordinates
      if (!gridMap.getPosition(idx, pos)) {
        // If for some reason the index is invalid, skip
        continue;
      }

      geometry_msgs::PoseStamped pose;
      pose.header = pathMsg.header;
      pose.pose.position.x = pos.x();
      pose.pose.position.y = pos.y();
      // Keep orientation neutral
      pose.pose.orientation.w = 1.0;

      pathMsg.poses.push_back(pose);
    }

    return pathMsg;
  }

  /**
   * @brief Converts the provided grid_map::GridMap into a grid_map_msgs::GridMap message.
   * @param gridMap  The GridMap to convert.
   * @return         A grid_map_msgs::GridMap message.
   */
  grid_map_msgs::GridMap toGridMapMsg(const grid_map::GridMap& gridMap)
  {
    grid_map_msgs::GridMap msg;
    grid_map::GridMapRosConverter::toMessage(gridMap, msg);
    msg.info.header.stamp = ros::Time::now();
    msg.info.header.frame_id = "map";
    return msg;
  }

  Eigen::Vector2d globalToLocal(const grid_map::Position& globalPos,
    double robot_x, double robot_y, double robot_yaw)
  {
    // Shift by robot position.
    double dx = globalPos.x() - robot_x;
    double dy = globalPos.y() - robot_y;
    // Rotate by -robot_yaw.
    double local_x = dx * std::cos(robot_yaw) + dy * std::sin(robot_yaw);
    double local_y = -dx * std::sin(robot_yaw) + dy * std::cos(robot_yaw);
    return Eigen::Vector2d(local_x, local_y);
  }

  grid_map::Position localToGlobal(const Eigen::Vector2d& localPos,
    double robot_x, double robot_y, double robot_yaw)
  {
    double global_x = robot_x + localPos.x() * std::cos(robot_yaw) - localPos.y() * std::sin(robot_yaw);
    double global_y = robot_y + localPos.x() * std::sin(robot_yaw) + localPos.y() * std::cos(robot_yaw);
    return grid_map::Position(global_x, global_y);
  }
} // namespace Utility