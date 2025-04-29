#include "mapping/TerrainMap.hpp"
#include "mapping/Constants.h"
#include <chrono>

using namespace grid_map;

TerrainMap::TerrainMap()
{
    double width  = Constants::GLOBAL_MAP_WIDTH;
    double height = Constants::GLOBAL_MAP_HEIGHT;
    double resolution = Constants::METERS_PER_CELL;

    // Create a GridMap with one layer: "terrainCost"
    std::vector<std::string> layers = {"terrainCost"};
    globalMap_.setFrameId("map");
    // The center is at (0,0) in the global frame.
    globalMap_.setGeometry(Length(width, height), resolution, Position(0, 0));

    // Initialize the "terrainCost" layer to 127 = unknown / mid-cost
    globalMap_.add("terrainCost", 75.0f);
}

void TerrainMap::updateGlobalMap(const cv::Mat& localMap,
                                       double x, double y, double yaw)
{
    auto start = std::chrono::high_resolution_clock::now();
    if (localMap.empty() || localMap.type() != CV_8UC1) {
        std::cerr << "[TerrainMap] localMap is invalid (empty or not CV_8UC1). Abort.\n";
        return;
    }

    // 1) Build a polygon in the global frame for the local map footprint
    //    centered at (x,y) with orientation yaw.
    //    The local map width/height in meters is derived from the image size * METERS_PER_PIXEL.
    double localWidthMeters  = Constants::IMG_WIDTH_PX  * Constants::METERS_PER_PIXEL;
    double localHeightMeters = Constants::IMG_HEIGHT_PX * Constants::METERS_PER_PIXEL;

    grid_map::Polygon footprintPolygon = buildGlobalFootprintPolygon(
        x, y, yaw, localWidthMeters, localHeightMeters);

    // 2) Use a PolygonIterator to iterate over cells in the global map
    //    that lie within this footprint.
    auto start2 = std::chrono::high_resolution_clock::now();
    for (PolygonIterator polyIt(globalMap_, footprintPolygon);
         !polyIt.isPastEnd(); ++polyIt)
    {
        // The *polyIt is an index into the grid (row,col).
        const Index index(*polyIt);

        // Get the center position of this grid cell in global coords
        Position cellPos;
        globalMap_.getPosition(index, cellPos);

        // cellPos.x() and cellPos.y() are the global coordinates of the cell center.

        // 3) Transform from global coords back to local coords relative to the robot.
        double dx = cellPos.x() - x;
        double dy = cellPos.y() - y;

        // Rotate by -yaw to go into robot's local frame
        double cosYaw = std::cos(-yaw);
        double sinYaw = std::sin(-yaw);
        double xLocal = dx * cosYaw - dy * sinYaw;
        double yLocal = dx * sinYaw + dy * cosYaw;

        // 4) Convert local meters to local image pixels
        //    The robot is at pixel (cx, cy) in the local image.
        double cx = 0.5 * localMap.cols; // center in pixel
        double cy = 0.5 * localMap.rows;
        double u  = (xLocal / Constants::METERS_PER_PIXEL) + cx;
        double v  = -(yLocal / Constants::METERS_PER_PIXEL) + cy;

        // Round to nearest pixel index
        int uInt = static_cast<int>(std::round(u));
        int vInt = static_cast<int>(std::round(v));

        // Check if pixel is within the local image
        if (uInt < 0 || uInt >= localMap.cols || vInt < 0 || vInt >= localMap.rows) {
            continue; // outside the local cost image
        }

        // 5) Read the cost from localMap
        uint8_t localCost = localMap.at<uint8_t>(vInt, uInt);

        // 6) Update the global map "terrainCost" layer
        //    Overwrite or use max? For now just overwrite.
        globalMap_.at("terrainCost", index) = static_cast<float>(localCost);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "[TerrainMap] Updated global map in " << elapsed.count() << " seconds." << std::endl;

    std::chrono::duration<double> elapsed2 = end - start2;
    std::cout << "[TerrainMap] Iterated in " << elapsed2.count() << " seconds." << std::endl;
}

grid_map::Polygon TerrainMap::buildGlobalFootprintPolygon(
    double x, double y, double yaw,
    double widthMeters, double heightMeters)
{
    // approximate the local map as a rectangle centered at (x,y) in global coords.
    // define the corners in local coords (meters):
    double halfW = 0.5 * widthMeters;
    double halfH = 0.5 * heightMeters;

    // corners in local frame:
    // top-left:    (-halfW, +halfH)
    // top-right:   (+halfW, +halfH)
    // bottom-right(+halfW, -halfH)
    // bottom-left (-halfW, -halfH)

    // rotate and translate them into the global frame.
    double cosYaw = std::cos(yaw);
    double sinYaw = std::sin(yaw);

    auto transformToGlobal = [&](double lx, double ly) {
        double gx = x + lx * cosYaw - ly * sinYaw;
        double gy = y + lx * sinYaw + ly * cosYaw;
        return Position(gx, gy);
    };

    // Construct polygon (in CCW or CW order)
    grid_map::Polygon polygon;
    polygon.addVertex(transformToGlobal(-halfW,  halfH)); // top-left
    polygon.addVertex(transformToGlobal( halfW,  halfH)); // top-right
    polygon.addVertex(transformToGlobal( halfW, -halfH)); // bottom-right
    polygon.addVertex(transformToGlobal(-halfW, -halfH)); // bottom-left

    return polygon;
}
