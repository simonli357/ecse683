#!/usr/bin/env python3
import rospy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import numpy as np

class GridMapVisualizer:
    def __init__(self):
        # Initialize node
        rospy.init_node("grid_map_visualizer", anonymous=True)

        # Instance variables to store the latest data
        self.latest_grid = None
        self.latest_grid_info = None
        self.latest_path = None

        # Termination flag
        self.terminate = False

        # Subscribers
        rospy.Subscriber("terrain_map", GridMap, self.grid_map_callback, queue_size=1)
        rospy.Subscriber("planned_path", Path, self.path_callback, queue_size=1)

        # Initialize interactive plot
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        plt.ion()  # Interactive mode

    def on_close(self, event):
        """Handles the Matplotlib window close event."""
        rospy.loginfo("Matplotlib window closed. Exiting...")
        self.terminate = True

    def grid_map_callback(self, msg):
        # Extract grid dimensions
        resolution = msg.info.resolution
        length_x = msg.info.length_x
        length_y = msg.info.length_y

        size_x = int(round(length_x / resolution))
        size_y = int(round(length_y / resolution))

        # Locate the "terrainCost" layer
        if "terrainCost" not in msg.layers:
            rospy.logwarn("No 'cost' layer found in GridMap message.")
            return
        layer_index = msg.layers.index("terrainCost")

        # Extract and reshape the cost data
        cost_data = msg.data[layer_index].data
        if len(cost_data) != size_x * size_y:
            rospy.logwarn("Cost data size does not match computed grid size!")
            return

        cost_array = np.array(cost_data, dtype=np.float32).reshape(size_y, size_x)

        # Store the grid data and info
        self.latest_grid = cost_array
        self.latest_grid_info = msg.info

    def path_callback(self, msg):
        # Convert path poses into a list of (x, y)
        pts = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.latest_path = pts

    def visualize(self):
        """
        Plots the latest grid and path if available, using matplotlib in interactive mode.
        """
        if self.latest_grid is None:
            return  # No grid data yet

        plt.clf()  # Clear the figure

        flipped_grid = np.flipud(np.fliplr(self.latest_grid))
        # Display the grid as an image
        plt.imshow(flipped_grid, cmap='gray', origin='lower')

        # Overlay the path if available
        if self.latest_path is not None and len(self.latest_path) > 0:
            resolution = self.latest_grid_info.resolution
            origin_x = self.latest_grid_info.pose.position.x - 0.5 * self.latest_grid_info.length_x
            origin_y = self.latest_grid_info.pose.position.y - 0.5 * self.latest_grid_info.length_y

            path_pixels_x = []
            path_pixels_y = []
            for (wx, wy) in self.latest_path:
                px = (wx - origin_x) / resolution
                py = (wy - origin_y) / resolution
                path_pixels_x.append(px)
                path_pixels_y.append(py)

            plt.plot(path_pixels_x, path_pixels_y, 'r-')  # Red line for path

        plt.title("GridMap (cost) + A* Path")
        plt.pause(0.001)  # Allow matplotlib to update the figure

    def run(self):
        """Main loop to periodically visualize data."""
        rate = rospy.Rate(2.0)  # 2 Hz
        while not rospy.is_shutdown() and not self.terminate:
            self.visualize()
            rate.sleep()

        # Show the final plot upon exit
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    visualizer = GridMapVisualizer()
    visualizer.run()
