#include "AStarPlanner.hpp"
#include <ros/ros.h>
#include <limits>
#include <chrono>
#include <Eigen/Dense>

constexpr double AStarPlanner::INF;

std::vector<grid_map::Index> AStarPlanner::plan(const grid_map::GridMap& gridMap,
                                                const std::string& layerName,
                                                const grid_map::Index& startIndex,
                                                const grid_map::Index& goalIndex)
{
  auto start = std::chrono::high_resolution_clock::now();

  // Fetch grid dimensions.
  const auto& mapSize = gridMap.getSize();
  int rows = mapSize(0);
  int cols = mapSize(1);

  // Prepare / reset all internal data for new search.
  resetData(rows, cols);

  // If a global guidance path is set, convert it to grid indices.
  if (useGlobalPath_ && !globalPath_.poses.empty()) {
    globalPathIndices_ = convertGlobalPathToIndices(globalPath_, gridMap);
  } else {
    globalPathIndices_.clear();
  }

  // Clear the priority queue (open set).
  openSet_ = std::priority_queue<AStarNode, std::vector<AStarNode>, CompareF>();

  // Initialize the start node and push to open set.
  const int startId = to1D(startIndex(0), startIndex(1));
  AStarNode startNode;
  startNode.index = startIndex;
  startNode.gCost = 0.0;
  startNode.fCost = heuristic(startIndex, goalIndex);
  openSet_.push(startNode);

  gScore_[startId] = 0.0;
  visited_[startId] = false;

  // A* main loop.
  while (!openSet_.empty())
  {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    if (elapsed.count() > 2.0) {
      ROS_WARN_STREAM("[AStarPlanner] Planning timed out after 2 seconds.");
      return {};
    }
    AStarNode current = openSet_.top();
    openSet_.pop();

    const auto& curIdx = current.index;
    const int curId = to1D(curIdx(0), curIdx(1));

    // Skip if already visited or if we already found a better path.
    if (visited_[curId]) {
      continue;
    }
    if (current.gCost > gScore_[curId]) {
      continue; // Outdated entry.
    }

    // Mark current node as visited.
    visited_[curId] = true;

    // Goal reached.
    if (current.index(0) == goalIndex(0) && current.index(1) == goalIndex(1)) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      ROS_INFO_STREAM("[AStarPlanner] Path found in " << elapsed.count() << " seconds.");
      return reconstructPath(goalIndex);
    }

    // Explore neighbors.
    for (const auto& dir : directions_) {
      grid_map::Index neighbor(curIdx(0) + dir.first,
                               curIdx(1) + dir.second);

      // Check bounds.
      if (!isInBounds(neighbor)) {
        continue;
      }
      const int neighborId = to1D(neighbor(0), neighbor(1));

      // Skip if already visited.
      if (visited_[neighborId]) {
        continue;
      }

      // Retrieve cell cost.
      float cellCost = gridMap.at(layerName, neighbor);

      // Treat high cost as an obstacle.
      if (cellCost >= 200.0f) {
        continue;
      }

      // Compute guidance penalty if using a global path.
      double guidancePenalty = 0.0;
      if (useGlobalPath_ && !globalPathIndices_.empty()) {
        double minDist = INF;
        for (const auto& gIdx : globalPathIndices_) {
          double d = heuristic(neighbor, gIdx);
          if (d < minDist) {
            minDist = d;
          }
        }
        guidancePenalty = guidanceWeight_ * minDist;
      }

      // Tentative cost: current cost + cell cost + guidance penalty.
      double tentativeG = gScore_[curId] + static_cast<double>(cellCost) + guidancePenalty;

      // If a better path is found, update the score and record the parent.
      if (tentativeG < gScore_[neighborId]) {
        gScore_[neighborId] = tentativeG;
        cameFrom_[neighborId] = curIdx;

        AStarNode neighborNode;
        neighborNode.index = neighbor;
        neighborNode.gCost = tentativeG;
        neighborNode.fCost = tentativeG + heuristic(neighbor, goalIndex);

        openSet_.push(neighborNode);
      }
    }
  }

  // No path found.
  ROS_WARN_STREAM("[AStarPlanner] No path found from start to goal!");
  return {};
}

double AStarPlanner::heuristic(const grid_map::Index& a, const grid_map::Index& b) const
{
  // Euclidean distance.
  double dr = static_cast<double>(a(0) - b(0));
  double dc = static_cast<double>(a(1) - b(1));
  return std::sqrt(dr * dr + dc * dc);
}

bool AStarPlanner::isInBounds(const grid_map::Index& index) const
{
  return (index(0) >= 0 && index(0) < nRows_ &&
          index(1) >= 0 && index(1) < nCols_);
}

std::vector<grid_map::Index> AStarPlanner::reconstructPath(const grid_map::Index& goalIndex) const
{
  std::vector<grid_map::Index> path;
  grid_map::Index current = goalIndex;

  while (true)
  {
    path.push_back(current);
    const int curId = to1D(current(0), current(1));

    // If no valid parent is stored, we have reached the start.
    if (cameFrom_[curId](0) < 0) {
      break;
    }
    current = cameFrom_[curId];
  }

  std::reverse(path.begin(), path.end());
  return path;
}

void AStarPlanner::resetData(int rows, int cols)
{
  if (rows != nRows_ || cols != nCols_) {
    nRows_ = rows;
    nCols_ = cols;
    visited_.resize(rows * cols, false);
    gScore_.resize(rows * cols, INF);
    cameFrom_.resize(rows * cols, grid_map::Index(-1, -1));
  } else {
    std::fill(visited_.begin(), visited_.end(), false);
    std::fill(gScore_.begin(), gScore_.end(), INF);
    std::fill(cameFrom_.begin(), cameFrom_.end(), grid_map::Index(-1, -1));
  }
}

std::vector<grid_map::Index> AStarPlanner::convertGlobalPathToIndices(const nav_msgs::Path& globalPath,
                                                                      const grid_map::GridMap& gridMap) const
{
  std::vector<grid_map::Index> indices;
  // Obtain grid dimensions for bounds checking.
  const auto& mapSize = gridMap.getSize();
  int rows = mapSize(0);
  int cols = mapSize(1);

  for (const auto& poseStamped : globalPath.poses) {
    grid_map::Index idx;
    // Convert the ROS geometry point to an Eigen::Vector2d.
    Eigen::Vector2d position(poseStamped.pose.position.x, poseStamped.pose.position.y);
    if (gridMap.getIndex(position, idx)) {
      // Only add indices that lie within the local grid.
      if (idx(0) >= 0 && idx(0) < rows && idx(1) >= 0 && idx(1) < cols) {
        indices.push_back(idx);
      }
    }
  }
  return indices;
}

void AStarPlanner::setGlobalPath(const nav_msgs::Path& globalPath)
{
  globalPath_ = globalPath;
  useGlobalPath_ = true;
}

void AStarPlanner::clearGlobalPath()
{
  useGlobalPath_ = false;
  globalPath_.poses.clear();
  globalPathIndices_.clear();
}
