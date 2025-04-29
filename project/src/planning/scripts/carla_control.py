#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl  
from ackermann_msgs.msg import AckermannDriveStamped
class CarlaController(object):
    def __init__(self):
        # Initialize the ROS node (if not already done in your main script)
        rospy.init_node('carla_controller', anonymous=True)

        # Subscribe to the vehicle odometry to get the current pose and speed
        self.odom_sub = rospy.Subscriber(
            "/carla/ego_vehicle/odometry", Odometry, self.odom_callback
        )
        # Publisher for vehicle control commands
        self.acker_pub = rospy.Publisher("/carla/ego_vehicle/ackermann_cmd", AckermannDriveStamped, queue_size=1)
        self.acker_msg = AckermannDriveStamped()

        # Variables to hold the current pose and speed
        self.current_pose = None
        self.current_speed = 0.0

    def odom_callback(self, msg):
        """
        Callback for the odometry topic.
        Stores the current pose and speed.
        """
        self.current_pose = msg.pose.pose
        self.current_speed = msg.twist.twist.linear.x

    def get_pose(self):
        """
        Returns the current pose of the car (geometry_msgs/Pose).
        """
        return self.current_pose

    def send_command(self, speed, steer):
        """
        Sends a control command to the vehicle.
        speed: desired speed in m/s
        steer: desired steering angle in rad
        """
        self.acker_msg.drive.speed = speed
        self.acker_msg.drive.steering_angle = steer
        self.acker_msg.header.stamp = rospy.Time.now()
        self.acker_pub.publish(self.acker_msg)        
    
if __name__ == '__main__':
    try:
        controller = CarlaController()
        rate = rospy.Rate(10)  # 10 Hz control loop

        while not rospy.is_shutdown():
            # Example command: go at 10 m/s and a slight steering angle of 0.1 rad.
            controller.send_command(10.0, 0.1)

            # Optionally, you can get and log the current pose:
            pose = controller.get_pose()
            if pose is not None:
                rospy.loginfo("Car position: x=%.2f, y=%.2f", pose.position.x, pose.position.y)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
