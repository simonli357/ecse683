# Preferential Terrain Navigation

## Overview
Preferential Terrain Navigation is a research project focused on developing an autonomous navigation system for robots capable of traversing semi-structured, off-road environments. Unlike many existing navigation systems that prioritize the shortest path, this system seeks to emulate human-like decision-making by considering terrain preferences. For example, a human would typically choose a sidewalk over a snowy path, even if the sidewalk is longer.

The system integrates terrain recognition with optimal control to achieve this goal. The approach consists of two key steps:

1. **Terrain Recognition and Mapping**: Using computer vision to analyze the environment and generate a terrain map with cost values that incorporates human-like terrain preferences and obstacles.
2. **Path Planning and Refinement**: Using the terrain map to plan a global path through trajectory optimization and refining the route using model predictive control (MPC).

This repository contains the implementation of the system using the CARLA simulator and the Robot Operating System (ROS).

---

## Features
- **Terrain Recognition**: Leveraging computer vision and segmentation techniques to classify terrain types and create a preference-based terrain map. Key steps consist of segmenting the camera images mounted on the robot, applying inverse perspective mapping to get the BEV image, and then convert that into a grid map with cost values associated with obstacles and terrain preferences.
- **Path Planning**: Utilizing path planning algorithms and trajectory optimization to compute global paths that align with human-like terrain preferences.
- **Path Refinement**: Applying model predictive control (MPC) to adjust the planned route dynamically in response to real-time conditions.
- **Simulation Environment**: Using CARLA to simulate unstructured environments and test navigation algorithms.

---

## Prerequisites

### **CARLA Simulator**: 
[Download CARLA](https://carla.org/) version 0.9.13.

### **CARLA ROS Bridge**:
[Follow the installation instructions](https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/) to install CARLA ROS Bridge from source.
Then create a symbolic link:
```bash
cd ~/ECSE683/project/src
ln -s ~/CARLA_0.9.13/Unreal/CarlaUE4/Content/Carla/Blueprints/ROSBridge/ carla_ros_bridge/ros-bridge
```
Then copy objects.json in the repository to the CARLA directory:
```bash
cp ~/ECSE683/project/src/objects.json ~/ECSE683/project/src/ros-bridge/carla_spawn_objects/config/
```

### **Robot Operating System (ROS)**: 
[Install ROS](http://wiki.ros.org/ROS/Installation) (tested with ROS Noetic).
### **Eigen**:
- Installation:
```bash
sudo apt install libeigen3-dev
```
### **Python**: 
Python 3.8 or higher.
### **Python Packages**:
- Installation:
```bash
pip install -r requirements.txt
```
### **Grid_Map Library**:
- Installation:
```bash
sudo apt install ros-<your-ros-distro>-grid-map*
```
### **PIDNet**: 
clone the PIDNet repository into the `src/segmentation` directory:
```bash
git clone https://github.com/XuJiacong/PIDNet.git
```

### Additional dependencies are managed through the included `CMakeLists.txt` and ROS packages.

---

## Getting Started

### How to use

#### Start CARLA Simulator
```bash
cd ~/CARLA_0.9.13
./CarlaUE4.sh -prefernvidia
```

#### Start ROS Bridge
```bash
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch
```

#### Start Ackermann Message Converter
```bash
roslaunch carla_ackermann_control carla_ackermann_control.launch
```

#### Start Segmentation Node
```bash
rosrun segmentation segmentation_node.py
```

#### Start Bird’s Eye View (BEV) Node
```bash
rosrun bev ipm_node.py
```

#### Start Global Path Planning Node
```bash
rosrun bev ipm_node.py
```
---

#### Start Path Planning Node
```bash
rosrun planning planner
```

#### Start Control Node
```bash
roslaunch control control.launch
```
---

### Useful Topics

#### Sensor Topics
- **IMU**: `/carla/ego_vehicle/imu` (Message Type: `sensor_msgs/Imu`)
- **GNSS**: `/carla/ego_vehicle/gnss` (Message Type: `sensor_msgs/NavSatFix`)
- **Odometry**: `/carla/ego_vehicle/odometry` (Message Type: `nav_msgs/Odometry`)
- **Front RGB Camera**: `/carla/ego_vehicle/camera/rgb/front/image_color` (Message Type: `sensor_msgs/Image`)
- **Speedometer**: `/carla/ego_vehicle/speedometer` (Message Type: `std_msgs/Float64`)
- **Front Radar**: `/carla/ego_vehicle/radar_front` (Message Type: `carla_msgs/CarlaRadar`)
- **LiDAR**: `/carla/ego_vehicle/lidar` (Message Type: `sensor_msgs/PointCloud2`)
- **Laser Scan**: `/scan` (Message Type: `sensor_msgs/LaserScan`) – added using the `pointcloud_to_laserscan` package.

---

## Repository Structure
```
PreferentialTerrainNavigation
├── src
│   ├── bev                  # Bird’s Eye View (BEV) generation
│   ├── segmentation         # Terrain segmentation
│   ├── mapping              # Grid Map Creation from Segmented BEV Image
│   ├── planning             # Path planning modules
│   ├── control              # MPC modules
│   ├── ros-bridge           # Integration with ROS and CARLA
│   ├── CMakeLists.txt       # Build configuration
```

---

## How to Contribute
1. Fork the repository and create a new branch for your feature or bug fix.
2. Ensure your code adheres to the existing style and structure.
3. Submit a pull request with a detailed description of your changes.

---

## Acknowledgments
- [CARLA Simulator](https://carla.org/)
- [ROS](http://wiki.ros.org/)

---

## Contact
For questions or collaboration, please contact Simon Li at xi.yang.li@mcgill.ca.

