#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
import math

def smooth_yaw_angles(yaw_angles):
    diffs = np.diff(yaw_angles)

    diffs[diffs > np.pi * 0.8] -= 2 * np.pi
    diffs[diffs < -np.pi * 0.8] += 2 * np.pi
    # Compute the smoothed yaw angles
    smooth_yaw = np.concatenate(([yaw_angles[0]], yaw_angles[0] + np.cumsum(diffs)))
    return smooth_yaw

class PathManager:
    def __init__(self, ref_speed=2.0):
        """
        PathManager processes a nav_msgs/Path and extracts all the data needed for MPC.
        """
        self.ref_speed = ref_speed

        self.waypoints_x = None
        self.waypoints_y = None
        self.num_waypoints = 0
        self.wp_normals = None
        self.kappa = None
        self.density = None
        self.state_refs = None     # Nx3 [x, y, yaw]
        self.input_refs = None     # Nx2 [v_ref, delta_ref]

        self._path_received = False

    def create_path(self, path_msg):
        poses = path_msg.poses
        if len(poses) < 2:
            rospy.logwarn("PathManager: Received path is too short.")
            return

        # Extract positions
        x = []
        y = []
        for pose_stamped in poses:
            pos = pose_stamped.pose.position
            x.append(pos.x)
            y.append(pos.y)
        x = np.array(x)
        y = np.array(y)

        # Compute yaw from dx, dy
        dx = np.gradient(x)
        dy = np.gradient(y)
        yaw = np.arctan2(dy, dx)

        # Save main references
        self.waypoints_x = x
        self.waypoints_y = y
        self.state_refs = np.stack([x, y, yaw], axis=1)
        self.state_refs[:, 2] = smooth_yaw_angles(self.state_refs[:, 2])
        self.num_waypoints = len(x)

        # Curvature
        self.kappa = self.compute_curvature(x, y)

        # Normal vectors (2D perpendicular to direction of travel)
        norm_x = -np.sin(yaw)
        norm_y = np.cos(yaw)
        self.wp_normals = np.stack([norm_x, norm_y], axis=1)

        # Reference speed + dummy steering (zeros)
        self.input_refs = np.stack([
            np.full_like(x, self.ref_speed),
            np.zeros_like(x)
        ], axis=1)

        # Density: number of waypoints per meter (approximate)
        dists = np.linalg.norm(np.diff(self.state_refs[:, :2], axis=0), axis=1)
        total_length = np.sum(dists)
        self.density = (self.num_waypoints - 1) / total_length if total_length > 0 else 1.0

        self._path_received = True
        rospy.loginfo(f"[PathManager] Path processed. {self.num_waypoints} waypoints, length ≈ {total_length:.2f} m, density ≈ {self.density:.2f} pts/m")

    def compute_curvature(self, x, y):
        """
        Compute discrete curvature using finite differences.
        κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        """
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        numerator = dx * ddy - dy * ddx
        denominator = (dx ** 2 + dy ** 2) ** 1.5 + 1e-8  # avoid division by zero
        curvature = numerator / denominator
        return curvature

    def is_ready(self):
        return self._path_received

    def get_path_as_dict(self):
        """
        Returns all relevant attributes in a dictionary (optional helper)
        """
        return {
            'waypoints_x': self.waypoints_x,
            'waypoints_y': self.waypoints_y,
            'num_waypoints': self.num_waypoints,
            'wp_normals': self.wp_normals,
            'kappa': self.kappa,
            'density': self.density,
            'state_refs': self.state_refs,
            'input_refs': self.input_refs
        }
