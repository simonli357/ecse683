#include <ros/ros.h>
#include "mapping/TerrainMap.hpp"
#include "mapping/Constants.h"
#include "mapping/ImageToCost.hpp"
#include <grid_map_cv/grid_map_cv.hpp>

int main(int argc, char** argv)
{
    // initialize ROS
    // ros::init(argc, argv, "grid_map_example");
    // ros::NodeHandle nh;
    TerrainMap myGlobalMap;

    cv::Mat colorImage = cv::imread("/home/slsecret/PreferentialTerrainNavigation/src/mapping/data/bev_image.png");
    if (colorImage.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    ImageToCost converter;
    cv::Mat localMap = converter.convert(colorImage);

    // 3) The robot's global pose
    double robotX = 50.0;     // center of global map
    double robotY = 50.0;     
    double robotYaw = M_PI/4; // 45 deg

    // 4) Update the global map with the local cost image
    myGlobalMap.updateGlobalMap(localMap, robotX, robotY, robotYaw);


    // 5) Optionally, retrieve the map and inspect the "terrainCost" layer
    auto& gridMap = myGlobalMap.getMap();
    // print the map size
    std::cout << "Global map size: " << gridMap.getSize().transpose() << std::endl;

    //print position of index 0,0
    grid_map::Position pos;
    gridMap.getPosition(grid_map::Index(0,0), pos);
    std::cout << "Position at index (0,0): " << pos.transpose() << std::endl;
    gridMap.getPosition(grid_map::Index(499,399), pos);
    std::cout << "Position at index (499,399): " << pos.transpose() << std::endl;
    gridMap.getPosition(grid_map::Index(499,0), pos);
    std::cout << "Position at index (499,0): " << pos.transpose() << std::endl;
    gridMap.getPosition(grid_map::Index(0,399), pos);
    std::cout << "Position at index (0,399): " << pos.transpose() << std::endl;
    gridMap.getPosition(grid_map::Index(0,200), pos);
    std::cout << "Position at index (0,200): " << pos.transpose() << std::endl;
    gridMap.getPosition(grid_map::Index(250,0), pos);
    std::cout << "Position at index (250,0): " << pos.transpose() << std::endl;
    std::cout << "Length: " << gridMap.getLength().transpose() << std::endl;

    // For demonstration, let's convert the "terrainCost" layer to a cv::Mat so we can visualize it.
    cv::Mat globalCostImage(gridMap.getSize().y(), gridMap.getSize().x(), CV_8UC1, cv::Scalar(0));
    std::cout << "globalCostImage size: " << globalCostImage.size() << std::endl;

    for (grid_map::GridMapIterator it(gridMap); !it.isPastEnd(); ++it) {
        const grid_map::Index index(*it);
        float costVal = gridMap.at("terrainCost", index);
        globalCostImage.at<uint8_t>(index(1), index(0)) = static_cast<uint8_t>(costVal);
    }

    std::cout << "globalCostImage size2: " << globalCostImage.size() << std::endl;
    cv::imshow("Global Cost (grid_map)", globalCostImage);
    cv::waitKey(0);

    // image is 16-bit, convert to 8-bit for visualization
    cv::Mat globalCostImage8;
    double minVal, maxVal;
    cv::minMaxLoc(globalCostImage, &minVal, &maxVal);
    std::cout << "minVal: " << minVal << ", maxVal: " << maxVal << std::endl;
    globalCostImage.convertTo(globalCostImage8, CV_8UC1, 255.0/(maxVal));

    std::cout << "globalCostImage8 size: " << globalCostImage8.size() << std::endl;

    cv::Mat displayMap;
    // rotate so that x points to the right and y points up
    cv::rotate(globalCostImage8, displayMap, cv::ROTATE_90_CLOCKWISE); 

    std::cout << "displayMap size: " << displayMap.size() << std::endl;

    cv::Mat colorImage2 = converter.convertToColor(displayMap);
    std::cout << "colorImage2 size: " << colorImage2.size() << std::endl;

    cv::namedWindow("Global Cost (grid_map)", cv::WINDOW_NORMAL);
    cv::imshow("Global Cost colored (grid_map)", colorImage2);
    cv::imshow("Global Cost (grid_map)", displayMap);
    cv::waitKey(0);

    return 0;
}
