#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <string>
#include "mapping/Constants.h"

using namespace Constants;
class ImageToCost {
public:
    ImageToCost(): costImage(cv::Mat())
    {
        initializeColorMap();
    }

    const cv::Mat& convert(const cv::Mat& colorImage) {
        if (colorImage.empty() || colorImage.channels() != 3) {
            throw std::invalid_argument("Input image must be a non-empty 3-channel image.");
        }

        costImage = cv::Mat(colorImage.rows, colorImage.cols, CV_8UC1);

        int cx = colorImage.cols / 2;
        int cy = colorImage.rows / 2;
        int halfWidth = ROBOT_LENGTH * PIXELS_PER_METER / 2;
        int halfHeight = ROBOT_WIDTH * PIXELS_PER_METER / 2;
        for (int y = 0; y < colorImage.rows; ++y) {
            for (int x = 0; x < colorImage.cols; ++x) {
                if (x >= cx - halfWidth && x < cx + halfWidth &&
                    y >= cy - halfHeight && y < cy + halfHeight) {
                    costImage.at<uchar>(y, x) = 0;
                } else {
                    cv::Vec3b color = colorImage.at<cv::Vec3b>(y, x);
                    costImage.at<uchar>(y, x) = getCost(color);
                }
            }
        }

        return costImage;
    }

    const cv::Mat& convertToColor(const cv::Mat& costImage) {
        if (costImage.empty() || costImage.channels() != 1) {
            throw std::invalid_argument("Input image must be a non-empty single-channel image.");
        }

        colorImage = cv::Mat(costImage.rows, costImage.cols, CV_8UC3);

        for (int y = 0; y < costImage.rows; ++y) {
            for (int x = 0; x < costImage.cols; ++x) {
                uchar cost = costImage.at<uchar>(y, x);
                colorImage.at<cv::Vec3b>(y, x) = getColor(cost);
            }
        }

        return colorImage;
    }

private:
    struct Vec3bHash {
        size_t operator()(const cv::Vec3b& color) const {
            return std::hash<int>()(color[0]) ^ (std::hash<int>()(color[1]) << 1) ^ (std::hash<int>()(color[2]) << 2);
        }
    };

    static std::vector<std::string> CLASSES;
    std::unordered_map<cv::Vec3b, float, Vec3bHash> colorCostMap;
    std::unordered_map<float, cv::Vec3b> costColorMap;
    std::set<std::array<uchar, 3>> unlabelledColors;
    cv::Mat costImage;
    cv::Mat colorImage;

    void initializeColorMap() {
        for (size_t i = 0; i < COLOR_MAP.size(); ++i) {
            colorCostMap[COLOR_MAP[i]] = TERRAIN_COSTS[i];
            costColorMap[TERRAIN_COSTS[i]] = COLOR_MAP[i];
        }
    }

    uchar getCost(const cv::Vec3b& color) {
        auto it = colorCostMap.find(color);
        if (it != colorCostMap.end()) {
            return static_cast<uchar>(it->second);
        }
        std::array<uchar, 3> color_key = { color[0], color[1], color[2] };
        unlabelledColors.insert(color_key);
        // print all unlabelled colors
        for (const auto& c : unlabelledColors) {
            std::cout << "Unlabelled color: {" << (int)c[0] << ", " << (int)c[1] << ", " << (int)c[2] << "}" << std::endl;
        }
        return 255; // Default cost for unmatched colors
    }

    cv::Vec3b getColor(float cost) const {
        auto it = costColorMap.find(cost);
        if (it != costColorMap.end()) {
            return it->second;
        }
        return {0, 0, 0}; // Default color for unmatched costs
    }
};
