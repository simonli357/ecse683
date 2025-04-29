#pragma once

#include "Planner.hpp"
#include <queue>
#include <limits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <nav_msgs/Path.h>
#include <unordered_set>
#include <Eigen/Core>

/**
 * @brief Class that implements an A* path planner for a single-layer GridMap.
 *        Inherits from Planner.
 */
class AStarPlanner : public Planner
{
public:
  AStarPlanner() = default;
  virtual ~AStarPlanner() = default;

  /**
   * @brief Finds a path from start to goal using A* search.
   * @param gridMap    The GridMap on which to plan.
   * @param layerName  The name of the layer in the grid map to use for cost.
   * @param startIndex The start cell index (row, col) in the grid.
   * @param goalIndex  The goal cell index (row, col) in the grid.
   * @return A vector of grid_map::Index forming the path from start to goal.
   *         If no path is found, returns an empty vector.
   */
  std::vector<grid_map::Index> plan(const grid_map::GridMap& gridMap,
                                    const std::string& layerName,
                                    const grid_map::Index& startIndex,
                                    const grid_map::Index& goalIndex) override;

  /**
   * @brief Set a global guidance path that biases the local planner.
   * @param globalPath The global path as a nav_msgs::Path.
   */
  void setGlobalPath(const nav_msgs::Path& globalPath);

  /**
   * @brief Clear any set global guidance path.
   */
  void clearGlobalPath();

private:
  /// Internal struct representing a node in the A* open set.
  struct AStarNode
  {
    grid_map::Index index;
    double gCost;  ///< Cost from the start node
    double fCost;  ///< gCost + heuristic
  };

  /// Comparison functor for the priority queue (min-heap by fCost).
  struct CompareF
  {
    bool operator()(const AStarNode& a, const AStarNode& b) const
    {
      return a.fCost > b.fCost;
    }
  };

  // Priority queue (open set) for A*, sorted by fCost (lowest first).
  std::priority_queue<AStarNode, std::vector<AStarNode>, CompareF> openSet_;

  // Pre-allocated buffers to avoid creating these containers on every call:
  std::vector<bool> visited_;             ///< Visited marker.
  std::vector<double> gScore_;            ///< gScore array: cost so far from start to each cell.
  std::vector<grid_map::Index> cameFrom_;   ///< Parent for path reconstruction.

  // Keep track of current map dimensions for indexing.
  int nRows_ = 0;
  int nCols_ = 0;

  /// Movement directions (8-connected). Remove diagonals for 4-connected.
  const std::vector<std::pair<int, int>> directions_ = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
  };

  // Global path guidance members.
  bool useGlobalPath_ = false;
  double guidanceWeight_ = 1.0; // Weight factor for guidance penalty.
  nav_msgs::Path globalPath_;
  std::vector<grid_map::Index> globalPathIndices_;

private:
  /// Converts (row, col) to a unique 1D index for accessing arrays.
  inline int to1D(int row, int col) const { return row * nCols_ + col; }

  /// Heuristic function (Euclidean distance).
  double heuristic(const grid_map::Index& a, const grid_map::Index& b) const;

  /// Checks if a given cell index is within the bounds of the grid map.
  bool isInBounds(const grid_map::Index& index) const;

  /// Reconstructs the path once the goal is reached.
  std::vector<grid_map::Index> reconstructPath(const grid_map::Index& goalIndex) const;

  /// Reset / resize internal buffers to match grid dimensions.
  void resetData(int rows, int cols);

  /**
   * @brief Converts a global nav_msgs::Path to grid indices within the current grid map.
   * @param globalPath The provided global path.
   * @param gridMap    The current grid map.
   * @return A vector of grid_map::Index corresponding to valid poses on the grid.
   */
  std::vector<grid_map::Index> convertGlobalPathToIndices(const nav_msgs::Path& globalPath,
                                                          const grid_map::GridMap& gridMap) const;

  /// Large cost to represent "infinity" in floating point.
  static constexpr double INF = std::numeric_limits<double>::infinity();
};
