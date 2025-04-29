#!/usr/bin/env python3

import math
import sys
import threading

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

import carla_common.transforms as trans
import ros_compatibility as roscomp
from ros_compatibility.exceptions import *
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_msgs.msg import CarlaWorldInfo
from carla_waypoint_types.srv import GetWaypoint, GetActorWaypoint
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np


import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ExtractBorders:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.spectator = world.get_spectator()
        self.vehicle_pose = vehicle.get_transform()
        self.map = world.get_map()
        self.actors_to_destroy = []

    def draw_waypoints(self, waypoints):
        for waypoint in waypoints:
            self.world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(155, 10, 20))

   

    def get_global_route(self):
        # get the global route of type Waypoint
        x_random, y_random = np.random.rand(), np.random.rand()
        gr = GlobalRoutePlanner(self.map, 2.0)
        start_waypoint = self.map.get_waypoint(self.vehicle_pose.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        end_waypoint = self.map.get_waypoint(carla.Location(x=x_random, y=y_random, z=0),project_to_road=True, lane_type=(carla.LaneType.Driving))
        route = gr.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        # global_wp = [x[0] for x in route]    ## LIST OF WAYPOINTS

        # return global_wp
        return route
        
    def set_specator(self):

        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        # print('spectator set at:', self.spectator.get_transform())

    # get the center waypoints of the road ahead and behind the vehicle by a certain distance
    def get_center_waypoints(self):
        vehicle_pose = self.vehicle.get_transform()
        current_waypoint = self.map.get_waypoint(vehicle_pose.location,project_to_road=True, lane_type=(carla.LaneType.Driving))
        # print("current s: ", current_waypoint.s)
        prev_waypoint_list = []
        next_waypoint_list = []
        for i in range(1, 20):
            prev_waypoint = current_waypoint.previous(i)
            next_waypoint = current_waypoint.next(i)
            if prev_waypoint:
                prev_waypoint_list.append(prev_waypoint[0])
            if next_waypoint:
                next_waypoint_list.append(next_waypoint[0])

        waypoint_list = list(reversed(prev_waypoint_list)) + [current_waypoint] + next_waypoint_list

        for waypoint in waypoint_list:

            self.world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(255, 0, 0), life_time=0.05)


        return waypoint_list

    # get the border waypoints of type Location
    def get_border_waypoints(self, waypoint_list):     

        left_border_list    = []
        right_border_list   = []
        for waypoint in waypoint_list:
            
            left_border_location = carla.Location(
                x=waypoint.transform.location.x + waypoint.lane_width * np.sin(np.deg2rad(waypoint.transform.rotation.yaw))* 0.5,
                y=waypoint.transform.location.y - waypoint.lane_width * np.cos(np.deg2rad(waypoint.transform.rotation.yaw))* 0.5,
                z=waypoint.transform.location.z
            )
            left_border_list.append(left_border_location)
            # self.world.debug.draw_point(left_border_location, size=0.15, color=carla.Color(0, 255, 0))

            right_border_location = carla.Location(
                x=waypoint.transform.location.x - waypoint.lane_width * np.sin(np.deg2rad(waypoint.transform.rotation.yaw)) * 0.5,
                y=waypoint.transform.location.y + waypoint.lane_width * np.cos(np.deg2rad(waypoint.transform.rotation.yaw)) * 0.5,
                z=waypoint.transform.location.z 
            )
            right_border_list.append(right_border_location)
            # self.world.debug.draw_point(right_border_location, size=0.15, color=carla.Color(0, 255, 0))

        return left_border_list, right_border_list

class CarlaToRosWaypointConverter(CompatibleNode):

    """
    This class generates a plan of waypoints in cartesian and frenet to follow.

    The calculation is done whenever:
    - the hero vehicle appears
    - a new goal is set
    """
    WAYPOINT_DISTANCE = 100.0

    def __init__(self):
        """
        Constructor
        """
        super(CarlaToRosWaypointConverter, self).__init__('global_planner')
        self.connect_to_carla()
        self.map = self.world.get_map()
        self.ego_vehicle = None
        self.ego_vehicle_location = None
        self.on_tick = None
        self.role_name = self.get_param("role_name", 'ego_vehicle')
        self.waypoint_publisher = self.new_publisher(
            Path,
            '/global_planner/{}/waypoints'.format(self.role_name),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.left_border_publisher = self.new_publisher(
            Path,
            '/global_planner/{}/left_border'.format(self.role_name),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        self.right_border_publisher = self.new_publisher(
            Path,
            '/global_planner/{}/right_border'.format(self.role_name),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        
        
        # initialize ros services
        self.get_waypoint_service = self.new_service(
            GetWaypoint,
            '/carla_waypoint_publisher/{}/get_waypoint'.format(self.role_name),
            self.get_waypoint)
        self.get_actor_waypoint_service = self.new_service(
            GetActorWaypoint,
            '/carla_waypoint_publisher/{}/get_actor_waypoint'.format(self.role_name),
            self.get_actor_waypoint)
        

        # set initial goal
        self.goal = self.world.get_map().get_spawn_points()[0]
        # speedway goal
        # speedway_goal = carla.Location(x=-64.6, y=24.4, z=0.59)
        # self.goal.location = speedway_goal
        print("Goal:", self.goal)
        self.current_route = None
        self.goal_subscriber = self.new_subscription(
            PoseStamped,
            # "/carla/{}/goal".format(self.role_name),
            "/move_base_simple/goal",
            self.on_goal,
            qos_profile=10)

        # use callback to wait for ego vehicle
        self.loginfo("Waiting for ego vehicle...")
        self.on_tick = self.world.on_tick(self.find_ego_vehicle_actor)

    def destroy(self):
        """
        Destructor
        """
        self.ego_vehicle = None
        if self.on_tick:
            self.world.remove_on_tick(self.on_tick)

    def get_waypoint(self, req, response=None):
        """
        Get the waypoint for a location
        """
        carla_position = carla.Location()
        carla_position.x = req.location.x
        carla_position.y = -req.location.y
        carla_position.z = req.location.z

        carla_waypoint = self.map.get_waypoint(carla_position)

        response = roscomp.get_service_response(GetWaypoint)
        response.waypoint.pose = trans.carla_transform_to_ros_pose(carla_waypoint.transform)
        response.waypoint.is_junction = carla_waypoint.is_junction
        response.waypoint.road_id = carla_waypoint.road_id
        response.waypoint.section_id = carla_waypoint.section_id
        response.waypoint.lane_id = carla_waypoint.lane_id
        return response

    def get_actor_waypoint(self, req, response=None):
        """
        Convenience method to get the waypoint for an actor
        """
        # self.loginfo("get_actor_waypoint(): Get waypoint of actor {}".format(req.id))
        actor = self.world.get_actors().find(req.id)

        response = roscomp.get_service_response(GetActorWaypoint)
        if actor:
            carla_waypoint = self.map.get_waypoint(actor.get_location())
            response.waypoint.pose = trans.carla_transform_to_ros_pose(carla_waypoint.transform)
            response.waypoint.is_junction = carla_waypoint.is_junction
            response.waypoint.road_id = carla_waypoint.road_id
            response.waypoint.section_id = carla_waypoint.section_id
            response.waypoint.lane_id = carla_waypoint.lane_id
        else:
            self.logwarn("get_actor_waypoint(): Actor {} not valid.".format(req.id))
        return response

    def on_goal(self, goal):
        """
        Callback for /move_base_simple/goal

        Receiving a goal (e.g. from RVIZ '2D Nav Goal') triggers a new route calculation.

        :return:
        """
        self.loginfo("Received goal, trigger rerouting...")
        carla_goal = trans.ros_pose_to_carla_transform(goal.pose)
        self.goal = carla_goal
        self.reroute()

    def reroute(self):
        """
        Triggers a rerouting
        """
        if self.ego_vehicle is None or self.goal is None:
            # no ego vehicle, remove route if published
            self.current_route = None
            self.publish_waypoints()
        else:
            self.current_route = self.calculate_route(self.goal)
            ## ----- EXTRACT BORDERS ----- ##


        self.publish_waypoints()

    def extract_borders(self):

        extractBorders = ExtractBorders(self.world, self.ego_vehicle)
        global_wp = [x[0] for x in self.current_route]
        left_border_wp, right_border_wp = extractBorders.get_border_waypoints(global_wp)
        # self.loginfo("Extracted borders.")
        # print('left_border_wp:', left_border_wp[0])
        # print('right_border_wp:', right_border_wp[0])
        return left_border_wp, right_border_wp
        # self.publish_borders(left_border_wp, right_border_wp)

    def publish_borders(self):
        """
        Publish two ROS messages containing the left and right border waypoints
        """
        left_msg = Path()
        left_msg.header.frame_id = "map"
        left_msg.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

        right_msg = Path()
        right_msg.header.frame_id = "map"
        right_msg.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)

        if self.current_route is not None:
            left_border_wp, right_border_wp = self.extract_borders()

            for wp in left_border_wp:
                pose = PoseStamped()
                # print('wp:', wp)
                # print("wp type:", type(wp))
                pose.pose = trans.carla_transform_to_ros_pose(carla.Transform(wp, carla.Rotation(0, 0, 0)))
                left_msg.poses.append(pose)
            
            self.left_border_publisher.publish(left_msg)
            self.loginfo("Published {} left border waypoints.".format(len(left_msg.poses)))

            for wp in right_border_wp:
                pose = PoseStamped()
                pose.pose = trans.carla_transform_to_ros_pose(carla.Transform(wp, carla.Rotation(0, 0, 0)))
                right_msg.poses.append(pose)

            self.right_border_publisher.publish(right_msg)
            self.loginfo("Published {} right border waypoints.".format(len(right_msg.poses)))




    def find_ego_vehicle_actor(self, _):
        """
        Look for an carla actor with name 'ego_vehicle'
        """
        hero = None
        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == self.role_name:
                hero = actor
                break

        ego_vehicle_changed = False
        if hero is None and self.ego_vehicle is not None:
            ego_vehicle_changed = True

        if not ego_vehicle_changed and hero is not None and self.ego_vehicle is None:
            ego_vehicle_changed = True

        if not ego_vehicle_changed and hero is not None and \
                self.ego_vehicle is not None and hero.id != self.ego_vehicle.id:
            ego_vehicle_changed = True

        if ego_vehicle_changed:
            self.loginfo("Ego vehicle changed.")
            self.ego_vehicle = hero
            self.reroute()
        elif self.ego_vehicle:
            current_location = self.ego_vehicle.get_location()
            if self.ego_vehicle_location:
                dx = self.ego_vehicle_location.x - current_location.x
                dy = self.ego_vehicle_location.y - current_location.y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance > self.WAYPOINT_DISTANCE:
                    self.loginfo("Ego vehicle was repositioned.")
                    self.loginfo("distance: {}".format(distance))   
                    self.reroute()
            self.ego_vehicle_location = current_location

    def calculate_route(self, goal):
        """
        Calculate a route from the current location to 'goal'
        """
        self.loginfo("Calculating route to x={}, y={}, z={}".format(
            goal.location.x,
            goal.location.y,
            goal.location.z))

        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=0.5)
        route = grp.trace_route(self.ego_vehicle.get_location(),
                                carla.Location(goal.location.x,
                                               goal.location.y,
                                               goal.location.z))

        return route

    def publish_waypoints(self):
        """
        Publish the ROS message containing the waypoints
        """
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = roscomp.ros_timestamp(self.get_time(), from_sec=True)
        if self.current_route is not None:
            for wp in self.current_route:
                # print("center wp:", wp[0].transform)

                pose = PoseStamped()
                pose.pose = trans.carla_transform_to_ros_pose(wp[0].transform)
                msg.poses.append(pose)

        self.waypoint_publisher.publish(msg)
        self.loginfo("Published {} waypoints.".format(len(msg.poses)))

        self.publish_borders()

    def connect_to_carla(self):

        self.loginfo("Waiting for CARLA world (topic: /carla/world_info)...")
        try:
            self.wait_for_message(
                "/carla/world_info",
                CarlaWorldInfo,
                qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                timeout=15.0)
        except ROSException as e:
            self.logerr("Error while waiting for world info: {}".format(e))
            raise e

        host = self.get_param("host", "127.0.0.1")
        port = self.get_param("port", 2000)
        timeout = self.get_param("timeout", 10)
        self.loginfo("CARLA world available. Trying to connect to {host}:{port}".format(
            host=host, port=port))

        carla_client = carla.Client(host=host, port=port)
        carla_client.set_timeout(timeout)

        try:
            self.world = carla_client.get_world()
        except RuntimeError as e:
            self.logerr("Error while connecting to Carla: {}".format(e))
            raise e

        self.loginfo("Connected to Carla.")


def main(args=None):
    """
    main function
    """
    roscomp.init('global_path_publisher', args)

    waypoint_converter = None
    try:
        waypoint_converter = CarlaToRosWaypointConverter()
        waypoint_converter.spin()
    except (RuntimeError, ROSException):
        pass
    except KeyboardInterrupt:
        roscomp.loginfo("User requested shut down.")
    finally:
        roscomp.loginfo("Shutting down.")
        if waypoint_converter:
            waypoint_converter.destroy()
        roscomp.shutdown()


if __name__ == "__main__":
    main()
