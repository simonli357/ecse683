#include <ros/ros.h>
#include <tf/tf.h>

#include <grid_map_core/GridMap.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>
#include <grid_map_msgs/GridMap.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

#include "AStarPlanner.hpp"
#include "mapping/TerrainMap.hpp"
#include "mapping/Constants.h"
#include "mapping/ImageToCost.hpp"
#include "Utility.h"

class Controller
{
public:
    Controller(ros::NodeHandle &nh) : nh_(nh)
    {
        map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("terrain_map", 1);
        path_pub_ = nh_.advertise<nav_msgs::Path>("planned_path", 1);

        bev_image_sub_ = nh_.subscribe("bev_image", 1, &Controller::bevImageCallback, this);
        odometry_sub_ = nh_.subscribe("/carla/ego_vehicle/odometry", 1, &Controller::odometryCallback, this);
        global_path_sub_ = nh_.subscribe("/global_planner/ego_vehicle/waypoints", 1, &Controller::globalPathCallback, this);

        ROS_INFO("Controller initialized: waiting for BEV images, odometry, and global waypoints.");
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher map_pub_;
    ros::Publisher path_pub_;
    ros::Subscriber bev_image_sub_;
    ros::Subscriber odometry_sub_;
    ros::Subscriber global_path_sub_;

    ImageToCost image_to_cost_converter_;
    TerrainMap global_map_;
    AStarPlanner planner_;

    double robot_x_ = 50.0;
    double robot_y_ = 30.0;
    double robot_yaw_ = M_PI / 4;
    nav_msgs::Path global_path_;

    cv::Mat bev_image_;

    void odometryCallback(const nav_msgs::Odometry::ConstPtr &msg)
    {
        robot_x_ = msg->pose.pose.position.x;
        robot_y_ = msg->pose.pose.position.y;
        robot_yaw_ = tf::getYaw(msg->pose.pose.orientation);
    }

    void globalPathCallback(const nav_msgs::Path::ConstPtr &msg)
    {
        planner_.setGlobalPath(*msg);
        global_path_ = *msg;
        std::cout << "Received global path with " << msg->poses.size() << " waypoints." << std::endl;
    }

    // Callback to process the BEV image, update the map, and plan a path
    void bevImageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        std::cout << "received bev image" << std::endl;
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        bev_image_ = cv_ptr->image;
        if (bev_image_.empty())
        {
            ROS_ERROR("Received an empty image!");
            return;
        }

        // Convert the BEV image to a local cost map
        auto &local_map = image_to_cost_converter_.convert(bev_image_);

        // Update the global map using the current robot position and orientation
        global_map_.updateGlobalMap(local_map, robot_x_, robot_y_, robot_yaw_);
        auto &map_ = global_map_.getMap();

        // Set the start position as the current robot location
        auto start_pos = grid_map::Position(robot_x_, robot_y_);

        grid_map::Position goal_pos;
        if (!global_path_.poses.empty())
        {
            // goal_pos = grid_map::Position(
            //     global_path_.poses.back().pose.position.x,
            //     global_path_.poses.back().pose.position.y);
            auto start = std::chrono::high_resolution_clock::now();
            goal_pos = computeLocalGoalFromGlobalPath(global_path_, robot_x_, robot_y_, robot_yaw_);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Computed local goal in " << elapsed.count() << " seconds." << std::endl;
        }
        else
        {
            // Default goal position is the robot's current position, going straight ahead for 20 meters in the direction of the robot's orientation
            goal_pos = grid_map::Position(
                robot_x_ + 20.0 * std::cos(robot_yaw_),
                robot_y_ + 20.0 * std::sin(robot_yaw_));
        }

        // Plan the path from the current robot position to the goal position
        auto path_indices = planner_.planFromPosition(map_, "terrainCost", start_pos, goal_pos);

        if (path_indices.empty() && !global_path_.poses.empty())
        {
            std::cout << "Planner returned empty path. Falling back to a portion of the global path." << std::endl;

            // Use computeLocalGoalFromGlobalPath to find a local goal
            grid_map::Position fallback_goal = computeLocalGoalFromGlobalPath(global_path_, robot_x_, robot_y_, robot_yaw_);

            // Build a partial path from the global path starting near the robot position
            nav_msgs::Path partial_path;
            partial_path.header = global_path_.header;

            const double max_distance = 10.0; // meters — max length of fallback path
            double accumulated_distance = 0.0;
            geometry_msgs::PoseStamped prev_pose;
            bool first = true;

            for (const auto& pose : global_path_.poses)
            {
                double dx = pose.pose.position.x - robot_x_;
                double dy = pose.pose.position.y - robot_y_;
                double distance_to_robot = std::sqrt(dx * dx + dy * dy);

                if (distance_to_robot < 1.0 || !first) // close enough to start or already started collecting
                {
                    if (!first)
                    {
                        double dx_seg = pose.pose.position.x - prev_pose.pose.position.x;
                        double dy_seg = pose.pose.position.y - prev_pose.pose.position.y;
                        accumulated_distance += std::sqrt(dx_seg * dx_seg + dy_seg * dy_seg);
                        if (accumulated_distance > max_distance)
                            break;
                    }
                    partial_path.poses.push_back(pose);
                    prev_pose = pose;
                    first = false;
                }
            }

            // Convert partial global path into grid_map::Index format (optional, if needed for consistency)
            path_indices.clear();
            for (const auto& pose : partial_path.poses)
            {
                grid_map::Index idx;
                Eigen::Vector2d pos(pose.pose.position.x, pose.pose.position.y);
                if (map_.getIndex(pos, idx))
                {
                    path_indices.push_back(idx);
                }
            }
        }
        
        // Convert the planned path and the map to ROS messages and publish them
        auto path_msg = Utility::toPathMsg(path_indices, map_, "map");
        auto grid_map_msg = Utility::toGridMapMsg(map_);

        map_pub_.publish(grid_map_msg);
        path_pub_.publish(path_msg);
    }

    /**
     * @brief Computes the local goal by finding the point along the global path where it exits the local map rectangle.
     *
     * The local map rectangle is centered at (robot_x, robot_y) with orientation robot_yaw and dimensions
     * localWidthMeters x localHeightMeters.
     *
     * @param globalPath         The global path.
     * @param robot_x, robot_y   The robot’s position.
     * @param robot_yaw          The robot’s heading.
     * @return grid_map::Position The computed local goal in global coordinates.
     */
    grid_map::Position computeLocalGoalFromGlobalPath(const nav_msgs::Path &globalPath,
                                                      double robot_x, double robot_y, double robot_yaw)
    {
        // Define half extents of the rectangle.
        double localWidthMeters  = Constants::IMG_WIDTH_PX  * Constants::METERS_PER_PIXEL;
        double localHeightMeters = Constants::IMG_HEIGHT_PX * Constants::METERS_PER_PIXEL;
        double halfWidth = localWidthMeters / 2.0;
        double halfHeight = localHeightMeters / 2.0;

        // Require at least two waypoints to compute an intersection.
        if (globalPath.poses.size() < 2)
        {
            if (!globalPath.poses.empty())
            {
                return grid_map::Position(globalPath.poses.back().pose.position.x,
                                          globalPath.poses.back().pose.position.y);
            }
            return grid_map::Position(robot_x, robot_y);
        }

        bool foundIntersection = false;
        Eigen::Vector2d local_goal; // In robot–centric coordinates.

        // Iterate over consecutive segments of the global path.
        for (size_t i = 0; i < globalPath.poses.size() - 1; ++i)
        {
            grid_map::Position p1(globalPath.poses[i].pose.position.x,
                                  globalPath.poses[i].pose.position.y);
            grid_map::Position p2(globalPath.poses[i + 1].pose.position.x,
                                  globalPath.poses[i + 1].pose.position.y);

            // Convert both endpoints to the robot–centric frame.
            Eigen::Vector2d l1 = Utility::globalToLocal(p1, robot_x, robot_y, robot_yaw);
            Eigen::Vector2d l2 = Utility::globalToLocal(p2, robot_x, robot_y, robot_yaw);

            // Check: does the segment go from inside to outside the rectangle?
            bool p1Inside = (std::abs(l1.x()) <= halfWidth && std::abs(l1.y()) <= halfHeight);
            bool p2Inside = (std::abs(l2.x()) <= halfWidth && std::abs(l2.y()) <= halfHeight);

            if (p1Inside && !p2Inside)
            {
                // We now compute the intersection between the segment and the rectangle boundary.
                double t_min = 1.0;
                bool validIntersection = false;
                Eigen::Vector2d intersection;

                // Lambda for checking intersection with a vertical or horizontal boundary.
                auto tryIntersection = [&](double boundary, bool vertical)
                {
                    double t;
                    if (vertical)
                    {
                        double dx = l2.x() - l1.x();
                        if (std::abs(dx) < 1e-6)
                            return;
                        t = (boundary - l1.x()) / dx;
                        if (t < 0.0 || t > 1.0)
                            return;
                        double y_int = l1.y() + t * (l2.y() - l1.y());
                        if (std::abs(y_int) <= halfHeight)
                        {
                            if (t < t_min)
                            {
                                t_min = t;
                                intersection = Eigen::Vector2d(boundary, y_int);
                                validIntersection = true;
                            }
                        }
                    }
                    else
                    {
                        double dy = l2.y() - l1.y();
                        if (std::abs(dy) < 1e-6)
                            return;
                        t = (boundary - l1.y()) / dy;
                        if (t < 0.0 || t > 1.0)
                            return;
                        double x_int = l1.x() + t * (l2.x() - l1.x());
                        if (std::abs(x_int) <= halfWidth)
                        {
                            if (t < t_min)
                            {
                                t_min = t;
                                intersection = Eigen::Vector2d(x_int, boundary);
                                validIntersection = true;
                            }
                        }
                    }
                };

                // Check all four boundaries.
                tryIntersection(halfWidth, true);    // Right boundary.
                tryIntersection(-halfWidth, true);   // Left boundary.
                tryIntersection(halfHeight, false);  // Top boundary.
                tryIntersection(-halfHeight, false); // Bottom boundary.

                if (validIntersection)
                {
                    local_goal = intersection;
                    foundIntersection = true;
                    break;
                }
            }
        }

        // If no segment crossed out of the rectangle, try to use the last waypoint that is inside.
        if (!foundIntersection)
        {
            for (int i = globalPath.poses.size() - 1; i >= 0; --i)
            {
                grid_map::Position p(globalPath.poses[i].pose.position.x,
                                     globalPath.poses[i].pose.position.y);
                Eigen::Vector2d l = Utility::globalToLocal(p, robot_x, robot_y, robot_yaw);
                if (std::abs(l.x()) <= halfWidth && std::abs(l.y()) <= halfHeight)
                {
                    local_goal = l;
                    foundIntersection = true;
                    break;
                }
            }
            // If still not found, default to a point on the forward edge.
            if (!foundIntersection)
            {
                local_goal = Eigen::Vector2d(halfWidth, 0.0); // For example, right at the forward edge.
            }
        }

        // Convert the local goal back to global coordinates.
        return Utility::localToGlobal(local_goal, robot_x, robot_y, robot_yaw);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "grid_map_a_star_publisher");
    ros::NodeHandle nh;
    Controller controller(nh);
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
