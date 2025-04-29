#include <ros/ros.h>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/iterators/GridMapIterator.hpp>
#include <grid_map_core/TypeDefs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

static const std::vector<std::array<int, 3>> COLOR_MAP = {
    {0,0,0}, {128,64,128}, {244,35,232}, {70,70,70},  {102,102,156},
    {190,153,153}, {153,153,153}, {250,170,30}, {220,220,0}, {107,142,35},
    {152,251,152}, {70,130,180}, {220,20,60}, {255,0,0}, {0,0,142},
    {0,0,70}, {0,60,100}, {0,80,100}, {0,0,230}, {119,11,32},
    {110, 190, 160}, {170, 120, 50}, {55, 90, 80}, {45, 60, 150}, {157, 234, 50},
    {81, 0, 81}, {150, 100, 100}, {230, 150, 140}, {180, 165, 180}
};

static const std::vector<std::string> CLASSES = {
    "unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person",
    "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "Static", "Dynamic", "Other", "Water", "Roadline", "Ground", "Bridge", "Railtrack", "Guardrail"
};

static const std::map<std::string, float> CLASS_COSTS = {
    {"unlabeled", 100.0f},
    {"road", 10.0f},
    {"sidewalk", 0.0f},
    {"building", 255.0f},
    {"wall", 255.0f},
    {"fence", 255.0f},
    {"pole", 50.0f},
    {"traffic light", 125.0f},
    {"traffic sign", 125.0f},
    {"vegetation", 30.0f},
    {"terrain", 20.0f},
    {"sky", 255.0f},
    {"person", 210.0f},
    {"rider", 200.0f},
    {"car", 255.0f},
    {"truck", 255.0f},
    {"bus", 255.0f},
    {"train", 255.0f},
    {"motorcycle", 200.0f},
    {"bicycle", 200.0f},
    {"Static", 250.0f},
    {"Dynamic", 249.0f},
    {"Other", 201.0f},
    {"Water", 202.0f},
    {"Roadline", 203.0f},
    {"Ground", 204.0f},
    {"Bridge", 205.0f},
    {"Railtrack", 206.0f},
    {"Guardrail", 207.0f} 
};

int getClassIdFromColor(const cv::Vec3b& colorBGR)
{
    // NOTE: OpenCV stores images in BGR, while your COLOR_MAP is likely in RGB.
    //       We must swap channels or store the COLOR_MAP in BGR if the image is in BGR.
    // e.g., colorBGR = (blue, green, red)
    // Compare with (r, g, b) if your color map is RGB.

    int blue  = static_cast<int>(colorBGR[0]);
    int green = static_cast<int>(colorBGR[1]);
    int red   = static_cast<int>(colorBGR[2]);

    for (size_t i = 0; i < COLOR_MAP.size(); ++i) {
        // If COLOR_MAP is in (r,g,b), then compare accordingly:
        if ((COLOR_MAP[i][0] == red) &&
            (COLOR_MAP[i][1] == green) &&
            (COLOR_MAP[i][2] == blue))
        {
            return i;  // returns the index 0..19
        }
    }
    // If no match:
    return 0; // default to "unlabeled" or handle error
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "terrain_map_publisher");
    ros::NodeHandle nh;

    ros::Publisher gridMapPub = nh.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);

    // 1. Load the segmented BEV image (BGR or RGB).
    cv::Mat bevImage = cv::imread("/home/slsecret/PreferentialTerrainNavigation/src/bev/scripts/output/bev_image.png", cv::IMREAD_COLOR);
    if (bevImage.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Dimensions in pixels
    int imgWidth  = bevImage.cols;  // 964
    int imgHeight = bevImage.rows;  // 604

    // 2. Calculate the map geometry
    double pixelsPerMeter = 13.6;
    double resolution     = 1.0 / pixelsPerMeter;       // ~0.0735294 [m/pixel]
    double mapWidth       = imgWidth  * resolution;     // ~70.88  [m]
    double mapHeight      = imgHeight * resolution;     // ~44.41  [m]

    // 3. Create the grid map
    grid_map::GridMap terrainMap({"terrain"});  // A single layer called "terrain"
    terrainMap.setFrameId("map");
    terrainMap.setGeometry(
        grid_map::Length(mapWidth, mapHeight),  // size in meters
        resolution,                             // cell size (m/cell)
        grid_map::Position(0.0, 0.0)            // center of map is at (0,0)
    );

    // Optionally set a default cost:
    terrainMap["terrain"].setConstant(100.0);

    // 4. Populate the grid map from the image
    //    We'll loop over each pixel and compute its (x,y) in map coordinates.
    for (int row = 0; row < imgHeight; ++row) {
        for (int col = 0; col < imgWidth; ++col) {
            // a) Get the color in BGR
            cv::Vec3b colorBGR = bevImage.at<cv::Vec3b>(row, col);

            // b) Find which class it corresponds to:
            int classId = getClassIdFromColor(colorBGR);
            if (classId < 0 || classId >= (int)CLASSES.size()) {
                continue; // skip or default
            }
            std::string className = CLASSES[classId];

            // c) Lookup cost
            float cost = 100.0f; // default
            auto it = CLASS_COSTS.find(className);
            if (it != CLASS_COSTS.end()) {
              cost = it->second;
            }

            // d) Convert (row, col) -> (x, y) in [m], with robot at center.
            //    row=0 is top of image -> typically positive Y if we want Y up.
            //    col=0 is left of image -> negative X if we want X to the left.
            double x =  (col - imgWidth  * 0.5) * resolution;   // center is col=imgWidth/2 -> x=0
            double y = -(row - imgHeight * 0.5) * resolution;   // center is row=imgHeight/2 -> y=0
            // (Minus sign if you want row=0 to be at positive Y. Adjust as needed.)

            grid_map::Position position(x, y);

            // e) Update the grid map cell if inside
            if (terrainMap.isInside(position)) {
                // "terrain" layer at that (x,y)
                terrainMap.atPosition("terrain", position) = cost;
            }
        }
    }

    // Now you have a grid map with a "terrain" layer,
    // where each cell's value is based on its semantic label.

    // 5. (Optional) You can now save, publish, or visualize the terrain map.
    // For example, if using ROS, you can do GridMapRosConverter::toMessage(...)
    // or store it as an image, etc.

    // Done!
    std::cout << "Created terrain map of size: "
              << terrainMap.getLength().transpose()
              << " meters, resolution "
              << terrainMap.getResolution() << " m/cell." << std::endl;

    ros::Rate rate(2.0);
    while (ros::ok())
    {
        // Update the timestamp for each publish cycle
        terrainMap.setTimestamp(ros::Time::now().toNSec());

        // Convert to ROS message and publish
        grid_map_msgs::GridMap msg;
        grid_map::GridMapRosConverter::toMessage(terrainMap, msg);
        gridMapPub.publish(msg);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
