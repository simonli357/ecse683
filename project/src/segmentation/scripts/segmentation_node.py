#!/usr/bin/env python3
import os
import cv2
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import time

# Configuration Section (change these variables as needed)
MODEL_TYPE = 'pidnet-l'  # Options: 'pidnet-s', 'pidnet-m', 'pidnet-l'
USE_CITYSCAPES = True    # True for Cityscapes pretrained model
# PRETRAINED_MODEL_PATH = './pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt'
cur_dir = os.path.dirname(os.path.realpath(__file__))
PRETRAINED_MODEL_PATH = os.path.join(cur_dir, '../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt')
# SAVE_DIR = './saved_images/'  # Directory to save images
SAVE_DIR = os.path.join(cur_dir, '../saved_images/')

# Normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# cityscapes
COLOR_MAP = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32)
]
CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

# carla
# COLOR_MAP = [
#     (0, 0, 0),        # Unlabeled
#     (128, 64, 128),   # Roads
#     (244, 35, 232),   # SideWalks
#     (70, 70, 70),     # Building
#     (102, 102, 156),  # Wall
#     (190, 153, 153),  # Fence
#     (153, 153, 153),  # Pole
#     (250, 170, 30),   # TrafficLight
#     (220, 220, 0),    # TrafficSign
#     (107, 142, 35),   # Vegetation
#     (152, 251, 152),  # Terrain
#     (70, 130, 180),   # Sky
#     (220, 20, 60),    # Pedestrian
#     (255, 0, 0),      # Rider
#     (0, 0, 142),      # Car
#     (0, 0, 70),       # Truck
#     (0, 60, 100),     # Bus
#     (0, 80, 100),     # Train
#     (0, 0, 230),      # Motorcycle
#     (119, 11, 32),    # Bicycle
#     (110, 190, 160),  # Static
#     (170, 120, 50),   # Dynamic
#     (55, 90, 80),     # Other
#     (45, 60, 150),    # Water
#     (157, 234, 50),   # RoadLine
#     (81, 0, 81),      # Ground
#     (150, 100, 100),  # Bridge
#     (230, 150, 140),  # RailTrack
#     (180, 165, 180)   # GuardRail
# ]
# CLASSES = [
#     'unlabeled', 'roads', 'sidewalks', 'building', 'wall', 'fence', 'pole',
#     'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
#     'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
#     'bicycle', 'static', 'dynamic', 'other', 'water', 'road line', 'ground',
#     'bridge', 'rail track', 'guard rail'
# ]

# ROS Topics
TOPIC_INPUTS = {
    "front": "/carla/ego_vehicle/camera_front/image",
    "left": "/carla/ego_vehicle/camera_left/image",
    "rear": "/carla/ego_vehicle/camera_rear/image",
    "right": "/carla/ego_vehicle/camera_right/image"
}
TOPIC_OUTPUTS = {
    "front": "/carla/ego_vehicle/segmentation_front",
    "left": "/carla/ego_vehicle/segmentation_left",
    "rear": "/carla/ego_vehicle/segmentation_rear",
    "right": "/carla/ego_vehicle/segmentation_right"
}

class SegmentationNode:
    def __init__(self):
        rospy.init_node('segmentation_node')
        self.bridge = CvBridge()

        if torch.cuda.is_available():
            print("GPU is available. Using GPU.")
            self.device = torch.device('cuda')
        else:
            print("GPU is not available. Using CPU.")
            self.device = torch.device('cpu')
        # Load model
        self.model = models.pidnet.get_pred_model(MODEL_TYPE, 19 if USE_CITYSCAPES else 11)
        self.model = self.load_pretrained(self.model, PRETRAINED_MODEL_PATH).to(self.device)
        self.model.eval()

        # Subscribers and publishers
        self.subscribers = {}
        self.publishers = {}
        for key, topic in TOPIC_INPUTS.items():
            self.subscribers[key] = rospy.Subscriber(topic, ROSImage, self.callback, callback_args=key)
            self.publishers[key] = rospy.Publisher(TOPIC_OUTPUTS[key], ROSImage, queue_size=1)

        self.images = {"front": None, "left": None, "rear": None, "right": None}
        self.frame_times = []

    def load_pretrained(self, model, pretrained):
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        print(f"Loaded {len(pretrained_dict)} parameters!")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        return model

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= MEAN
        image /= STD
        return image

    def callback(self, msg, camera):
        # start_time = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        transformed_image = self.segment_image(cv_image)
        self.images[camera] = transformed_image
        self.publish_segmented_image(transformed_image, camera)
        # frame_time = time.time() - start_time
        # self.frame_times.append(frame_time)
        # fps = 1.0 / frame_time
        # print(f"Frame Time: {frame_time:.3f}s | FPS: {fps:.2f}")

    def segment_image(self, img):
        resized = False
        old_shape = img.shape
        if img.shape[:2] != (1024, 2048):
            img = cv2.resize(img, (2048, 1024), interpolation=cv2.INTER_LINEAR)
            resized = True

        img_input = self.input_transform(img).transpose((2, 0, 1))
        img_input = torch.from_numpy(img_input).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_input)
            pred = F.interpolate(pred, size=img.shape[:2], mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        sv_img = np.zeros_like(img).astype(np.uint8)
        for i, color in enumerate(COLOR_MAP):
            for j in range(3):
                sv_img[:, :, j][pred == i] = color[j]

        if resized:
            sv_img = cv2.resize(sv_img, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_LINEAR)
        return sv_img

    def publish_segmented_image(self, img, camera):
        msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        self.publishers[camera].publish(msg)

    def save_images(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_folder = os.path.join(SAVE_DIR, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        for cam, img in self.images.items():
            if img is not None:
                save_path = os.path.join(save_folder, f"{cam}.png")
                cv2.imwrite(save_path, img)
                print(f"Saved {cam} image to {save_path}")
                
    def display_images(self):
        while not rospy.is_shutdown():
            if all(img is not None for img in self.images.values()):
                # Resize individual images to 512x256 for display purposes
                resized_images = {key: cv2.resize(img, (512, 256)) for key, img in self.images.items()}
                combined_image = np.hstack((
                    np.vstack((resized_images["front"], resized_images["rear"])),
                    np.vstack((resized_images["left"], resized_images["right"]))
                ))
                cv2.imshow("Segmented Images", combined_image)
                # if len(self.frame_times) > 0:
                #     avg_fps = len(self.frame_times) / sum(self.frame_times)
                #     print(f"Average FPS: {avg_fps:.2f}")
                key = cv2.waitKey(1)
                if key == ord('s'):
                    self.save_images()
                    print("Images saved!")

if __name__ == '__main__':
    node = SegmentationNode()
    node.display_images()
    rospy.spin()
