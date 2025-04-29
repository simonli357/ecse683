#include <ros/ros.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>

#include "AStarPlanner.hpp"

#include "mapping/TerrainMap.hpp"
#include "mapping/Constants.h"
#include "mapping/ImageToCost.hpp"
#include "Utility.h"

class Controller {
    
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "grid_map_a_star_publisher");
    ros::NodeHandle nh;

    ros::Publisher mapPub = nh.advertise<grid_map_msgs::GridMap>("terrain_map", 1);
    ros::Publisher pathPub = nh.advertise<nav_msgs::Path>("planned_path", 1);

    // grid_map::GridMap map({"terrainCost"});
    // map.setGeometry(grid_map::Length(10.0, 10.0), 0.1, grid_map::Position(0.0, 0.0));
    // std::cout << "Created map with size: " << map.getSize().transpose()
    //         << " (rows x cols)" << std::endl;
    // for (grid_map::GridMapIterator it(map); !it.isPastEnd(); ++it) {
    //     map.at("terrainCost", *it) = 1.0;
    // }
    // grid_map::Polygon starPolygon;
    // starPolygon.addVertex(grid_map::Position(0.0,  2.0));   // top
    // starPolygon.addVertex(grid_map::Position(0.5,  0.7));
    // starPolygon.addVertex(grid_map::Position(2.0,  0.7));
    // starPolygon.addVertex(grid_map::Position(0.8, -0.2));
    // starPolygon.addVertex(grid_map::Position(1.2, -1.8));  
    // starPolygon.addVertex(grid_map::Position(0.0, -0.8));  // bottom center
    // starPolygon.addVertex(grid_map::Position(-1.2, -1.8));
    // starPolygon.addVertex(grid_map::Position(-0.8, -0.2));
    // starPolygon.addVertex(grid_map::Position(-2.0,  0.7));
    // starPolygon.addVertex(grid_map::Position(-0.5,  0.7));

    // for (grid_map::PolygonIterator polyIt(map, starPolygon); !polyIt.isPastEnd(); ++polyIt) {
    //     // *polyIt is a grid_map::Index
    //     map.at("terrainCost", *polyIt) = 50.0;
    // }

    cv::Mat colorImage = cv::imread("/home/slsecret/PreferentialTerrainNavigation/src/mapping/data/bev_image.png");
    if (colorImage.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    ImageToCost converter;
    cv::Mat localMap = converter.convert(colorImage);

    // 3) The robot's global pose
    double robotX = 50.0;     // center of global map
    double robotY = 70.0;     
    double robotYaw = M_PI/4; // 45 deg
    TerrainMap globalMap;
    globalMap.updateGlobalMap(localMap, robotX, robotY, robotYaw);
    auto& map = globalMap.getMap();

    AStarPlanner planner;

    // 3. Define start and goal in index space
    grid_map::Index startIndex(0, 0);
    // grid_map::Index goalIndex(92, 80);
    grid_map::Index goalIndex(460, 415);

    // 4. Plan
    auto pathIndices = planner.plan(map, "terrainCost", startIndex, goalIndex);

    // 5. Convert to ROS messages
    auto pathMsg = Utility::toPathMsg(pathIndices, map, "map"); // or any frame you'd like
    auto gridMapMsg = Utility::toGridMapMsg(map);
    
    // Publish in a loop
    ros::Rate rate(1.0); // 1 Hz
    while (ros::ok()) {
        gridMapMsg.info.header.stamp = ros::Time::now();
        pathMsg.header.stamp = ros::Time::now();

        mapPub.publish(gridMapMsg);
        pathPub.publish(pathMsg);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
