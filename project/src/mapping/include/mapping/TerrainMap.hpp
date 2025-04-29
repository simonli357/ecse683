#pragma once

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/iterators/GridMapIterator.hpp>
#include <grid_map_core/iterators/PolygonIterator.hpp>
#include <grid_map_core/Polygon.hpp>

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

class TerrainMap
{
public:
    TerrainMap();

    /**
     * @brief Updates the global map "terrainCost" layer with a local cost image.
     * @param localMap  Grayscale CV_8UC1 image (0 to 255).
     * @param x         Robot's global X (m).
     * @param y         Robot's global Y (m).
     * @param yaw       Robot's heading (radians).
     */
    void updateGlobalMap(const cv::Mat& localMap, double x, double y, double yaw);

    /**
     * @return reference to the underlying grid map
     */
    grid_map::GridMap& getMap() { return globalMap_; }

private:
    grid_map::GridMap globalMap_;

    // Utility to build a polygon in the global frame that represents
    // the corners of the local map footprint at (x,y,yaw).
    grid_map::Polygon buildGlobalFootprintPolygon(
        double x, double y, double yaw,
        double widthMeters, double heightMeters);

    // For coordinate transforms
    inline double deg2rad(double deg) const { return deg * M_PI / 180.0; }
};
