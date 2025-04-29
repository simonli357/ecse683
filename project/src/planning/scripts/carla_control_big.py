#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDrive
import tf
import numpy as np

from optimizer import TrajectoryOptimizer, iLQR

def transform_point(pt, frame1, frame2):
    x1, y1, theta1 = frame1
    x2, y2, theta2 = frame2
    x, y, psi = pt
    # Translate to the origin of frame1
    x -= x1
    y -= y1
    # Rotate to align frame1 with frame2
    rotation_angle = theta2 - theta1
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    rotated_xy = np.dot(np.array([x, y]), rotation_matrix.T)
    rotated_psi = (psi + rotation_angle) % (2 * np.pi)
    # Translate to the origin of frame2
    transformed_xy = rotated_xy + np.array([x2, y2])
    return np.array([transformed_xy[0], transformed_xy[1], rotated_psi])

class CarlaController(object):
    def __init__(self):
        # Initialize the ROS node (if not already done in your main script)
        rospy.init_node('carla_controller', anonymous=True)

        # Subscribe to the vehicle odometry to get the current pose and speed
        self.odom_sub = rospy.Subscriber(
            "/carla/ego_vehicle/odometry", Odometry, self.odom_callback
        )
        self.imu_sub = rospy.Subscriber(
            "/carla/ego_vehicle/imu", Imu, self.imu_callback
        )
        # Publisher for vehicle control commands
        self.acker_pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDrive, queue_size=1)
        self.acker_msg = AckermannDrive()

        # Variables to hold the current pose and speed
        self.current_pose = None
        self.current_state2 = np.zeros(3) # current state in frame 2
        self.current_state1 = np.zeros(3) # current state in frame 1
        self.current_speed = 0.0
        self.yaw = None
        self.initialized = False
        self.frame1 = None
        self.frame2 = np.array([0, 0, 0])
        self.goal = np.array([20, 0, 0]) # in frame 2
        self.obstacle = np.array([5.0, 0.1, 1.0]) # third element is the radius

    def imu_callback(self, msg):
        """
        Callback for the IMU topic.
        Stores the current orientation.
        """
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.yaw = tf.transformations.euler_from_quaternion(orientation_list)
        self.current_state1[2] = self.yaw
        
    def odom_callback(self, msg):
        """
        Callback for the odometry topic.
        Stores the current pose and speed.
        """
        self.current_pose = msg.pose.pose
        self.current_state1[0] = self.current_pose.position.x
        self.current_state1[1] = self.current_pose.position.y
        self.current_speed = msg.twist.twist.linear.x

    def get_pose(self):
        """
        Returns the current pose of the car (geometry_msgs/Pose).
        """
        return self.current_pose

    def initialize(self):
        if self.current_pose is None or self.yaw is None:
            return
        if self.frame1 is None:
            self.frame1 = self.current_state1.copy()
        traj_opt = TrajectoryOptimizer(self.frame2, self.goal, self.obstacle)
        self.X_opt, self.U_opt = traj_opt.solve()
        
        # Simulation params
        sim_time = 20.0
        dt_sim = 0.1
        N_sim = int(sim_time / dt_sim)
        horizon = 10

        # Initialize iLQR controller
        self.ilqr = iLQR(dt=dt_sim, L=2.5, horizon=horizon)
        self.t_idx = 0
        self.initialized = True
        print("initialized")
        print("frame1: ", self.frame1)
        print("X_opt: ", self.X_opt)
    def send_command(self):
        """
        Sends a control command to the vehicle.
        speed: desired speed in m/s
        steer: desired steering angle in rad
        """
        if not self.initialized:
            self.initialize()
            return
        speed = 0
        steer = 0
        if self.t_idx < len(self.X_opt[0]) - self.ilqr.horizon:
            self.current_state2 = transform_point(self.current_state1, self.frame1, self.frame2)
            print("current state: ", self.current_state2)
            ref_horizon = self.X_opt[:, self.t_idx:self.t_idx+self.ilqr.horizon+1]
            u_seq, x_seq = self.ilqr.solve(self.current_state2, ref_horizon)
            if u_seq.size == 0: 
                return
            u_apply = u_seq[:, 0]
            speed = u_apply[0]
            steer = u_apply[1]
            print("u: ", u_apply)
            self.t_idx += 1
        self.acker_msg.speed = speed
        self.acker_msg.steering_angle = steer
        self.acker_pub.publish(self.acker_msg)        
    
if __name__ == '__main__':
    try:
        controller = CarlaController()
        rate = rospy.Rate(10)  # 10 Hz control loop

        while not rospy.is_shutdown():
            # Example command: go at 10 m/s and a slight steering angle of 0.1 rad.
            controller.send_command()

            # pose = controller.get_pose()
            # if pose is not None:
            #     rospy.loginfo("Car position: x=%.2f, y=%.2f, yaw=%.2f", pose.position.x, pose.position.y, controller.yaw)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
