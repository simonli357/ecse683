#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>

namespace Constants {
    static constexpr int IMG_WIDTH_PX = 964;
    static constexpr int IMG_HEIGHT_PX = 604;
    static constexpr double PIXELS_PER_METER = 13.6;
    static constexpr double METERS_PER_PIXEL = 1.0 / PIXELS_PER_METER;
    static constexpr double LOCAL_MAP_WIDTH = IMG_WIDTH_PX * METERS_PER_PIXEL;
    static constexpr double LOCAL_MAP_HEIGHT = IMG_HEIGHT_PX * METERS_PER_PIXEL;
    static constexpr double GLOBAL_MAP_WIDTH = 320.0;
    static constexpr double GLOBAL_MAP_HEIGHT = 300.0;
    static constexpr double METERS_PER_CELL = 0.50;
    static constexpr double ROBOT_WIDTH = 2.0;
    static constexpr double ROBOT_LENGTH = 4.0;

    enum class TERRAIN_TYPE {
        UNLABELED, ROAD, SIDEWALK, BUILDING, WALL, FENCE, POLE,
        TRAFFIC_LIGHT, TRAFFIC_SIGN, VEGETATION, TERRAIN, SKY, PERSON,
        RIDER, CAR, TRUCK, BUS, TRAIN, MOTORCYCLE, BICYCLE,
        STATIC, DYNAMIC, OTHER, WATER, ROADLINE, GROUND, BRIDGE, RAILTRACK, GUARDRAIL
    };

    static const std::vector<std::string> TERRAIN_NAMES = {
        "unlabeled", "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person",
        "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        "Static", "Dynamic", "Other", "Water", "Roadline", "Ground", "Bridge", "Railtrack", "Guardrail"
    };

    static const std::vector<cv::Vec3b> COLOR_MAP = {
        {0, 0, 0}, {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
        {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35},
        {152, 251, 152}, {70, 130, 180}, {220, 20, 60}, {255, 0, 0}, {0, 0, 142},
        {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32},
        {110, 190, 160}, {170, 120, 50}, {55, 90, 80}, {45, 60, 150}, {157, 234, 50},
        {81, 0, 81}, {150, 100, 100}, {230, 150, 140}, {180, 165, 180}
    };

    static const std::vector<float> TERRAIN_COSTS = {
        100.0f, 0.0f, 10.0f, 255.0f, 254.0f,
        253.0f, 50.0f, 125.0f, 124.0f, 30.0f,
        20.0f, 252.0f, 210.0f, 200.0f, 251.0f,
        248.0f, 250.0f, 249.0f, 201.0f, 202.0f,
        201.0f, 202.0f, 203.0f, 75.0f, 1.0f,
        25.0f, 57.0f, 89.0f, 90.0f
    };
}

#endif