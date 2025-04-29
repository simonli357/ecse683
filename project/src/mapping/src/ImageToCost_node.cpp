#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "mapping/ImageToCost.hpp"

class ImageProcessor {
public:
    ImageProcessor() {
        sub_ = nh_.subscribe("/bev_image", 1, &ImageProcessor::imageCallback, this);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv::Mat colorImage = cv_bridge::toCvShare(msg, "rgb8")->image;
            if (colorImage.empty()) {
                ROS_ERROR("Received empty image!");
                return;
            }

            cv::imshow("Color Image", colorImage);
            ImageToCost converter;
            cv::Mat grayImage = converter.convert(colorImage);

            cv::imshow("Converted Grayscale Cost Image", grayImage);
            cv::waitKey(1);

            cv::Mat colorImage2 = converter.convertToColor(grayImage);
            cv::imshow("Converted Color Image", colorImage2);
            cv::waitKey(1);

        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void spin() {
        ros::spin();
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_to_cost_node");
    ImageProcessor processor;
    processor.spin();
    return 0;
}
