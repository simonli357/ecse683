#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import math
import numpy as np
import carla
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from path_manager import PathManager
from mpc_acados import Optimizer
import matplotlib.pyplot as plt
from agents.navigation.global_route_planner import GlobalRoutePlanner

def transform_point(pt, frame1, frame2):
    '''
    Transforms a point from frame1 to frame2.
    '''
    x1, y1, theta1 = frame1
    x2, y2, theta2 = frame2
    x, y, psi = pt
    x -= x1
    y -= y1
    rotation_angle = theta2 - theta1
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    rotated_xy = np.dot(np.array([x, y]), rotation_matrix.T)
    rotated_psi = (psi + rotation_angle) % (2 * np.pi)
    transformed_xy = rotated_xy + np.array([x2, y2])
    while rotated_psi < -np.pi:
        rotated_psi += 2*np.pi
    while rotated_psi > np.pi:
        rotated_psi -= 2*np.pi
    return np.array([transformed_xy[0], transformed_xy[1], rotated_psi]) 

class CarlaBridge:
    def __init__(self):
        rospy.init_node('mpc_ackermann_controller')

        # Connect to CARLA
        host = rospy.get_param("~carla_host", "localhost")
        port = rospy.get_param("~carla_port", 2000)
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        # Parameters
        self.max_steering_angle = rospy.get_param("~max_steering_angle", 0.418)  # ~24 degrees
        self.max_speed = rospy.get_param("~max_speed", 10.0)  # m/s

        # Vehicle state
        self.ego_pose_received = False
        self.current_state = np.zeros(3)  # [x, y, yaw]
        self.v = 0
        self.mpc_frame = np.array([0, 0, 0])
        self.odom_frame = None
        self.path_frame = None
        
        # Subscribers
        rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self.odom_callback)
        rospy.Subscriber("/global_planner/ego_vehicle/waypoints", Path, self.path_callback)
        
        # Publisher
        self.cmd_pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size=10)
        self.path_pub = rospy.Publisher("/mpc/path", Path, queue_size=1)
        
        self.path_msg = None
        rospy.wait_for_message("/carla/ego_vehicle/odometry", Odometry)
        rospy.wait_for_message("/global_planner/ego_vehicle/waypoints", Path)
        rospy.loginfo("Carla Bridge Started.")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_state = np.array([x, y, yaw])
        # print("odom_callback(): self.current_state = ", self.current_state)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v = math.sqrt(vx**2 + vy**2)
        if not self.ego_pose_received:
            self.odom_frame = self.current_state
        self.ego_pose_received = True

    def path_callback(self, msg):
        self.path_msg = msg
        if not self.ego_pose_received:
            return
        
    def send_ackermann_cmd(self, speed, steering_angle, clip=True):
        # speed: m/s, steering_angle: rad
        if clip:
            speed = max(min(speed, self.max_speed), -self.max_speed)
            steering_angle = max(min(steering_angle, self.max_steering_angle), -self.max_steering_angle)
        ack_msg = AckermannDrive()
        ack_msg.speed = speed
        ack_msg.steering_angle = steering_angle
        self.cmd_pub.publish(ack_msg)
        
    def send_twist_cmd(self, msg):
        # speed: m/s, steering_angle: rad
        desired_speed = msg.linear.x
        desired_steering = msg.angular.z
        self.send_ackermann_cmd(desired_speed, desired_steering)
        rospy.loginfo_throttle(1.0, f"[CMD] Speed: {desired_speed:.2f} m/s | Steer: {math.degrees(desired_steering):.2f} deg")
    
    def spin(self):
        rospy.spin()
    
    def init_display(self):
        # Enable interactive mode for matplotlib.
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Path and Car Pose')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)

        # Plot path if available.
        if self.path_msg is not None and len(self.path_msg.poses) > 0:
            xs = [pose.pose.position.x for pose in self.path_msg.poses]
            ys = [pose.pose.position.y for pose in self.path_msg.poses]
            self.ax.plot(xs, ys, '-b', label='Path')
            self.ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')  # Green for start
            self.ax.plot(xs[-1], ys[-1], 'ro', markersize=10, label='Goal')  # Red for goal

        # Car pose marker (position)
        self.car_pose_marker, = self.ax.plot([], [], 'kx', markersize=12, label='Car Pose')

        # Orientation arrow (initially None)
        self.car_arrow = None

        self.ax.legend()
        plt.show()


    def update_display(self):
        if not hasattr(self, 'ax'):
            return

        # Transform current car pose
        if self.odom_frame is not None:
            # car_pose = transform_point(self.current_state, self.odom_frame, self.path_frame)
            car_pose = self.current_state
        else:
            car_pose = self.current_state

        x, y, psi = car_pose  # psi is orientation (heading)

        # Update position marker
        self.car_pose_marker.set_data(x, y)

        # Remove old arrow if it exists
        if self.car_arrow is not None:
            self.car_arrow.remove()

        # Draw new arrow for orientation
        arrow_length = 1.0
        dx = arrow_length * math.cos(psi)
        dy = arrow_length * math.sin(psi)
        self.car_arrow = self.ax.arrow(
            x, y, dx, dy,
            head_width=0.3, head_length=0.3,
            fc='k', ec='k'
        )

        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    try:
        bridge = CarlaBridge()
        bridge.init_display()
        path = PathManager()
        path.create_path(bridge.path_msg)
        print("stateref:")
        for i in range(30):
            print(path.state_refs[i])
        mpc = Optimizer(path)
        mpc.target_waypoint_index = 0
        maxiter = 4000
        rate = rospy.Rate(1/mpc.T)
        iter = 0
        print("start mpc loop")
        while not rospy.is_shutdown() and mpc.target_waypoint_index < len(path.waypoints_x):
            # state_mpc_frame = transform_point(bridge.current_state, bridge.odom_frame, bridge.mpc_frame)
            # state_mpc_frame = transform_point(bridge.current_state, bridge.odom_frame, bridge.path_frame)
            state_mpc_frame = bridge.current_state
            mpc.update(state_mpc_frame, bridge.v)
            u_res = mpc.update_and_solve()
            bridge.send_ackermann_cmd(u_res[0], u_res[1])
            bridge.update_display()
            iter += 1
            if iter >= maxiter:
                break
            error_norm = np.linalg.norm(state_mpc_frame[:2] - np.array([path.waypoints_x[-1], path.waypoints_y[-1]]))
            if error_norm <3:
                print("goal reached: current: ", state_mpc_frame[:2], "goal: ", np.array([path.waypoints_x[-1], path.waypoints_y[-1]]))
                break
            rate.sleep()
        for hsy in range(10):
            bridge.send_ackermann_cmd(0, 0)
            bridge.update_display()
            rate.sleep()
        print("done")
    except rospy.ROSInterruptException:
        pass
