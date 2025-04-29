#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import argparse

def image_publisher(image_path, show):
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('bev_image', Image, queue_size=1)

    if not os.path.exists(image_path):
        rospy.logerr(f"Image file does not exist at path: {image_path}")
        return

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        rospy.logerr("Failed to load image! Make sure the file exists and is a valid image.")
        return

    bridge = CvBridge()
    rate = rospy.Rate(1)  # Publish at 1 Hz

    rospy.loginfo("Publishing image on topic 'bev_image_topic'...")

    while not rospy.is_shutdown():
        try:
            # Convert OpenCV image to ROS Image message
            image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")

            # Publish the image
            image_pub.publish(image_msg)

            if show:
                cv2.imshow("Published Image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    rospy.loginfo("Display window closed by user.")
                    break

            rate.sleep()
        except rospy.ROSInterruptException:
            rospy.loginfo("Image publishing interrupted.")
            break

    if show:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Publish an image to a ROS topic.")
    parser.add_argument("--image", type=str, default="/home/slsecret/PreferentialTerrainNavigation/src/mapping/data/bev_image.png",
                        help="Path to the image file to publish.")
    parser.add_argument("--show", action="store_true",
                        help="Display the image in a window while publishing.")
    args = parser.parse_args()

    try:
        image_publisher(args.image, args.show)
    except rospy.ROSInterruptException:
        pass