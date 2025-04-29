#pragma once

#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/TypeDefs.hpp>
#include <nav_msgs/Path.h>
#include <grid_map_msgs/GridMap.h>

/**
 * @brief Abstract base class for path planners operating on a single-layer GridMap.
 */
class Planner
{
public:
  /**
   * @brief Virtual destructor.
   */
  virtual ~Planner() = default;

  /**
   * @brief Pure virtual method for planning a path from start to goal,
   *        given a GridMap and a cost layer name.
   * @param gridMap     The GridMap on which to plan.
   * @param layerName   The name of the cost layer (e.g., "cost").
   * @param startIndex  The start cell index in the grid.
   * @param goalIndex   The goal cell index in the grid.
   * @return            A vector of grid_map::Index from start to goal
   *                    or an empty vector if no path is found.
   */
  virtual std::vector<grid_map::Index> plan(const grid_map::GridMap& gridMap,
                                            const std::string& layerName,
                                            const grid_map::Index& startIndex,
                                            const grid_map::Index& goalIndex) = 0;

  /**
   * @brief Convenience method to plan from a Position (world coordinates).
   *        This default implementation converts Positions to Indices, then calls plan(...).
   * @param gridMap     The GridMap on which to plan.
   * @param layerName   The name of the cost layer (e.g., "cost").
   * @param startPos    The start position in world coordinates (x,y).
   * @param goalPos     The goal position in world coordinates (x,y).
   * @return            A vector of grid_map::Index forming the path in grid cells.
   *                    Returns an empty vector if no path is found or if conversion fails.
   */
  virtual std::vector<grid_map::Index> planFromPosition(const grid_map::GridMap& gridMap,
                                                        const std::string& layerName,
                                                        const grid_map::Position& startPos,
                                                        const grid_map::Position& goalPos);
};

