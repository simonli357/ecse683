#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import tf
import numpy as np
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import String

from ilqr import iLQR
from direct_collocation import TrajectoryOptimizer

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

class CarlaController(object):
    def __init__(self):
        rospy.init_node('carla_controller', anonymous=True)

        self.imu_sub = rospy.Subscriber(
            "/car1/imu", Imu, self.imu_callback
        )
        self.cmd_vel_pub = rospy.Publisher("/car1/command", String, queue_size=3)
        self.msg = String()
        self.msg2 = String()
        self.model = ModelStates()
        self.model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_callback, queue_size=3)

        self.current_state2 = np.zeros(3) # current state in frame 2
        self.current_state1 = np.zeros(3) # current state in frame 1
        self.current_speed = 0.0
        self.current_pose = None
        self.yaw = None
        self.initialized = False
        self.frame1 = None
        self.frame2 = np.array([0, 0, 0])
        self.goal = np.array([2, 0, 0]) # in frame 2
        self.obstacle = np.array([0.9, 0.01, 0.2]) # third element is the radius
        self.i = None
        self.maxspeed = 1.0

    def model_callback(self, model):
        if self.i is None:
            try:
                self.i = model.name.index("car1") # index of the car
                print("car1 found in model states")
            except ValueError:
                print("automobile not found in model states")
                return
        self.car_inertial = model.twist[self.i]
        self.x_speed = self.car_inertial.linear.x
        self.y_speed = self.car_inertial.linear.y
        self.gps_x = model.pose[self.i].position.x
        self.gps_y = 15+model.pose[self.i].position.y
        self.current_pose = np.array([self.gps_x, self.gps_y, 0])
        self.current_state1[0] = self.gps_x
        self.current_state1[1] = self.gps_y
            
    def imu_callback(self, msg):
        """
        Callback for the IMU topic.
        Stores the current orientation.
        """
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, self.yaw = tf.transformations.euler_from_quaternion(orientation_list)
        while self.yaw < -np.pi:
            self.yaw += 2*np.pi
        while self.yaw > np.pi:
            self.yaw -= 2*np.pi
        self.current_state1[2] = self.yaw

    def initialize(self):
        if self.current_pose is None or self.yaw is None:
            return
        if self.frame1 is None:
            self.frame1 = self.current_state1.copy()
        
        # Create direct collocation trajectory optimizer
        traj_opt = TrajectoryOptimizer(self.frame2, self.goal, self.obstacle)
        self.X_opt, self.U_opt = traj_opt.solve()
        
        # Simulation params
        sim_time = 20.0
        dt_sim = 0.1
        N_sim = int(sim_time / dt_sim)
        horizon = 10

        # create iLQR controller
        self.ilqr = iLQR(dt=dt_sim, L=0.258, horizon=horizon)
        self.t_idx = 0
        self.initialized = True
        print("initialized")
        print("frame1: ", self.frame1)
    def publish_cmd_vel(self, steering_angle, velocity = None, clip = True):
        """
        steering_angle: in degrees
        velocity: in m/s
        """
        if velocity is None:
            velocity = self.maxspeed
        if clip:
            steering_angle = np.clip(steering_angle, -21, 21)
        self.msg.data = '{"action":"1","speed":'+str(velocity)+'}'
        self.msg2.data = '{"action":"2","steerAngle":'+str(float(-steering_angle))+'}'

        self.cmd_vel_pub.publish(self.msg)
        self.cmd_vel_pub.publish(self.msg2)
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
            goal_error = np.linalg.norm(self.current_state2[:2] - self.goal[:2])
            if goal_error < 0.2:
                print("goal reached")
                self.publish_cmd_vel(0, 0)
                exit()
            print("current state: ", self.current_state2)
            ref_horizon = self.X_opt[:, self.t_idx:self.t_idx+self.ilqr.horizon+1]
            u_seq, x_seq = self.ilqr.solve(self.current_state2, ref_horizon)
            if u_seq.size == 0: 
                return
            u_apply = u_seq[:, 0]
            speed = u_apply[0]
            steer = u_apply[1]
            print(self.t_idx,") u: ", u_apply)
            self.t_idx += 1
        else: 
            print("done")
            self.publish_cmd_vel(0, 0)
            exit()
        self.publish_cmd_vel(steer * 180/np.pi, speed)     
    
if __name__ == '__main__':
    try:
        controller = CarlaController()
        rate = rospy.Rate(10)  # 10 Hz control loop

        while not rospy.is_shutdown():
            controller.send_command()

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
