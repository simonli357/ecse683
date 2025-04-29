#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import yaml
import numpy as np
import cv2
import argparse

class Camera:
    def __init__(self, config):
        self.K = np.zeros([3, 3])  # Camera intrinsic matrix
        self.R = np.zeros([3, 3])  # Rotation matrix
        self.t = np.zeros([3, 1])  # Translation vector
        self.P = np.zeros([3, 4])  # Projection matrix
        self._initialize(config)

    def _initialize(self, config):
        self.setK(config["fx"], config["fy"], config["px"], config["py"])
        self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
        self.setT(config["XCam"], config["YCam"], config["ZCam"])
        self.updateP()

    def setK(self, fx, fy, px, py):
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = px
        self.K[1, 2] = py
        self.K[2, 2] = 1.0

    def setR(self, yaw, pitch, roll):
        # Rotation matrices around each axis
        Rz = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
                       [np.sin(-yaw),  np.cos(-yaw), 0.0],
                       [0.0,           0.0,          1.0]])
        Ry = np.array([[np.cos(-pitch), 0.0, np.sin(-pitch)],
                       [0.0,            1.0, 0.0],
                       [-np.sin(-pitch), 0.0, np.cos(-pitch)]])
        Rx = np.array([[1.0,    0.0,           0.0],
                       [0.0,    np.cos(-roll), -np.sin(-roll)],
                       [0.0,    np.sin(-roll),  np.cos(-roll)]])
        # Switch axes (x = -y, y = -z, z = x)
        Rs = np.array([[0.0, -1.0, 0.0],
                       [0.0,  0.0, -1.0],
                       [1.0,  0.0, 0.0]])

        self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

    def setT(self, XCam, YCam, ZCam):
        X = np.array([XCam, YCam, ZCam])
        self.t = -self.R.dot(X)

    def updateP(self):
        Rt = np.zeros([3, 4])
        Rt[0:3, 0:3] = self.R
        Rt[0:3, 3] = self.t.ravel()
        self.P = self.K.dot(Rt)

class IPMNode:
    def __init__(self, gt):
        rospy.init_node('ipm_node', anonymous=True)
        self.bridge = CvBridge()
        if gt:
            self.image_subs = [
                rospy.Subscriber('/carla/ego_vehicle/semantic_camera_front/image', Image, self.callback_front),
                rospy.Subscriber('/carla/ego_vehicle/semantic_camera_left/image', Image, self.callback_left),
                rospy.Subscriber('/carla/ego_vehicle/semantic_camera_rear/image', Image, self.callback_rear),
                rospy.Subscriber('/carla/ego_vehicle/semantic_camera_right/image', Image, self.callback_right)
            ]
        else:
            self.image_subs = [
                rospy.Subscriber('/carla/ego_vehicle/segmentation_front', Image, self.callback_front),
                rospy.Subscriber('/carla/ego_vehicle/segmentation_left', Image, self.callback_left),
                rospy.Subscriber('/carla/ego_vehicle/segmentation_rear', Image, self.callback_rear),
                rospy.Subscriber('/carla/ego_vehicle/segmentation_right', Image, self.callback_right)
            ]
        self.image_pub = rospy.Publisher('/bev_image', Image, queue_size=10)
        self.images = [None, None, None, None]

    def callback_front(self, msg):
        self.images[0] = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback_left(self, msg):
        self.images[2] = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback_rear(self, msg):
        self.images[1] = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def callback_right(self, msg):
        self.images[3] = self.bridge.imgmsg_to_cv2(msg, "bgr8")


class IPMProcessor:
    def __init__(self, args):
        self.args = args
        self.node = IPMNode(args.gt)
        self.current_dir = os.path.dirname(os.path.realpath(__file__))

        # Configuration paths
        self.camera_config_paths = [
            os.path.join(self.current_dir, "camera_configs/1_FRLR/front.yaml"),
            os.path.join(self.current_dir, "camera_configs/1_FRLR/rear.yaml"),
            os.path.join(self.current_dir, "camera_configs/1_FRLR/left.yaml"),
            os.path.join(self.current_dir, "camera_configs/1_FRLR/right.yaml")
        ]

        self.drone_config_path = os.path.join(self.current_dir, "camera_configs/1_FRLR/drone.yaml")

        # IPM parameters
        self.width_m = 20
        self.height_m = 40
        self.resolution = 20
        self.cc = True
        self.batch = True
        self.toDrone = True

        self.droneConfig = None
        if self.toDrone:
            with open(os.path.abspath(self.drone_config_path)) as stream:
                self.droneConfig = yaml.safe_load(stream)

        self.cameraConfigs= self.load_camera_configs(self.camera_config_paths)

        # Create Camera objects
        self.cams = [Camera(config) for config in self.cameraConfigs]
        if self.toDrone:
            self.drone = Camera(self.droneConfig)

        # Compute output resolution and pixel-to-meter conversion
        if not self.toDrone:
            self.pxPerM = (self.resolution, self.resolution)
            self.outputRes = (int(self.width_m * self.pxPerM[0]), int(self.height_m * self.pxPerM[1]))
        else:
            self.outputRes = (int(2 * self.droneConfig["py"]), int(2 * self.droneConfig["px"]))
            dx = self.outputRes[1] / self.droneConfig["fx"] * self.droneConfig["ZCam"]
            dy = self.outputRes[0] / self.droneConfig["fy"] * self.droneConfig["ZCam"]
            self.pxPerM = (self.outputRes[0] / dy, self.outputRes[1] / dx)
            print("pxPerM: ", self.pxPerM)
            print("outputRes: ", self.outputRes)

        # Compute shift and transformation matrix M
        shift = (self.outputRes[0] / 2.0, self.outputRes[1] / 2.0)
        if self.toDrone:
            shift = (
                shift[0] + self.droneConfig["YCam"] * self.pxPerM[0],
                shift[1] - self.droneConfig["XCam"] * self.pxPerM[1]
            )
        self.shift = shift

        self.M = np.array([
            [1.0 / self.pxPerM[1], 0.0, -shift[1] / self.pxPerM[1]],
            [0.0, -1.0 / self.pxPerM[0], shift[0] / self.pxPerM[0]],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Compute inverse perspective matrices for each camera
        self.IPMs = [np.linalg.inv(cam.P.dot(self.M)) for cam in self.cams]
        self.masks = self.compute_masks(self.cameraConfigs, self.outputRes, self.droneConfig, self.pxPerM)
        self.interpMode = cv2.INTER_NEAREST if self.cc else cv2.INTER_LINEAR

    def load_camera_configs(self, camera_config_paths):
        cameraConfigs = []
        for config_path in camera_config_paths:
            with open(os.path.abspath(config_path)) as stream:
                cameraConfigs.append(yaml.safe_load(stream))

        return cameraConfigs

    def compute_masks(self, cameraConfigs, outputRes, droneConfig, pxPerM):
        masks = []
        h, w = outputRes
        j_coords, i_coords = np.indices((h, w))
        offset_y = outputRes[0] / 2.0 - droneConfig["YCam"] * pxPerM[0]
        offset_x = -outputRes[1] / 2.0 + droneConfig["XCam"] * pxPerM[1]
        theta_radians = np.arctan2(-(j_coords - offset_y), (i_coords + offset_x))
        theta_degrees = np.degrees(theta_radians)
        for config in cameraConfigs:
            yaw = config["yaw"]
            angle_diff = np.abs(theta_degrees - yaw)
            invalid = (angle_diff > 90) & (angle_diff < 270)
            mask = np.repeat(invalid[:, :, np.newaxis], 3, axis=2)
            masks.append(mask)
        return masks

    def process_images(self):
        hsy = 1
        while not rospy.is_shutdown():
            images = self.node.images
            if any(image is None for image in images):
                rospy.loginfo("Waiting for images... %d", hsy)
                hsy += 1
                rospy.sleep(0.1)  # 10 Hz
                continue

            # Warp images using the precomputed IPM matrices
            warpedImages = [
                cv2.warpPerspective(img, IPM, (self.outputRes[1], self.outputRes[0]), flags=self.interpMode)
                for img, IPM in zip(images, self.IPMs)
            ]

            # Apply masks to remove invalid regions
            i = 0
            for warpedImg, mask in zip(warpedImages, self.masks):
                cv2.imwrite(os.path.join(self.current_dir, f"output/warped_image_{i}.png"), warpedImg)
                warpedImg[mask] = 0
                cv2.imwrite(os.path.join(self.current_dir, f"output/mask_{i}.png"), warpedImg)
                i += 1

            # Stitch warped images together to form the bird's eye view
            birdsEyeView = np.zeros(warpedImages[0].shape, dtype=np.uint8)
            for wImg in warpedImages:
                mask = np.any(wImg != (0, 0, 0), axis=-1)
                birdsEyeView[mask] = wImg[mask]

            # Publish the generated bird's eye view image
            self.node.image_pub.publish(self.node.bridge.cv2_to_imgmsg(birdsEyeView, "bgr8"))

            if self.args.show:
                self.show_images(images, birdsEyeView)

    def show_images(self, images, birdsEyeView):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)
        thickness = 2
        line_type = cv2.LINE_AA
        base_shape = images[0].shape if len(images) > 0 else (480, 640, 3)
        blank_image = np.zeros(base_shape, dtype=np.uint8)

        # Label camera images
        cam_names = ["front", "rear", "left", "right"]
        labeled_cams = []
        for i in range(4):
            cam_img = images[i] if i < len(images) else blank_image.copy()
            cv2.putText(cam_img, cam_names[i], (10, 30), font, font_scale, font_color, thickness, line_type)
            labeled_cams.append(cam_img)

        # Create a 2x2 grid of camera images
        top_row = cv2.hconcat([labeled_cams[0], labeled_cams[1]])
        bottom_row = cv2.hconcat([labeled_cams[2], labeled_cams[3]]) if len(labeled_cams) > 2 else blank_image
        camera_grid = cv2.vconcat([top_row, bottom_row])

        bev_resized = birdsEyeView
        if camera_grid.shape[1] != bev_resized.shape[1]:
            bev_resized = cv2.resize(bev_resized, (camera_grid.shape[1], bev_resized.shape[0]))
        cv2.putText(bev_resized, "Bird's Eye View", (10, 30), font, font_scale, font_color, thickness, line_type)

        # Combine the camera grid and the bird's eye view
        display_image = cv2.vconcat([camera_grid, bev_resized])
        scale_factor = 0.5
        new_width = int(display_image.shape[1] * scale_factor)
        new_height = int(display_image.shape[0] * scale_factor)
        display_image = cv2.resize(display_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.namedWindow("Bird's Eye View", cv2.WINDOW_NORMAL)
        cv2.imshow("Bird's Eye View", display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            output_dir = os.path.join(self.current_dir, "output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_path = os.path.join(output_dir, "bev_image.png")
            cv2.imwrite(save_path, birdsEyeView)
            rospy.loginfo("Bird's Eye View image saved to %s", save_path)

    def run(self):
        try:
            self.process_images()
        except rospy.ROSInterruptException:
            pass
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Generate Bird's Eye View from multiple camera images")
    parser.add_argument("--gt", action="store_true", help="Use ground truth images")
    parser.add_argument("--show", action="store_true", help="Show images in a window")
    args = parser.parse_args()
    processor = IPMProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()
