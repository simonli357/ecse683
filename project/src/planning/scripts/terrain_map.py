import numpy as np
import cv2

COLOR_MAP = [
    (0, 0, 0),         # unlabeled
    (128, 64, 128),    # road
    (244, 35, 232),    # sidewalk
    (70, 70, 70),      # building
    (102, 102, 156),   # wall
    (190, 153, 153),   # fence
    (153, 153, 153),   # pole
    (250, 170, 30),    # traffic light
    (220, 220, 0),     # traffic sign
    (107, 142, 35),    # vegetation
    (152, 251, 152),   # terrain
    (70, 130, 180),    # sky
    (220, 20, 60),     # person
    (255, 0, 0),       # rider
    (0, 0, 142),       # car
    (0, 0, 70),        # truck
    (0, 60, 100),      # bus
    (0, 80, 100),      # train
    (0, 0, 230),       # motorcycle
    (119, 11, 32)      # bicycle
]
CLASSES = [
    'unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def color_to_class_id(semantic_bev: np.ndarray) -> np.ndarray:
    """
    Convert a color-coded semantic BEV image (H, W, 3) to a 2D array (H, W)
    of class IDs using the COLOR_MAP. This uses a simple per-pixel matching.
    
    Args:
        semantic_bev: np.ndarray of shape (H, W, 3) with color-coded semantic labels.
    
    Returns:
        A 2D np.ndarray of shape (H, W) where each element is an integer class ID.
    """
    # Convert input to the same dtype to avoid type mismatches
    semantic_bev = semantic_bev.astype(np.uint8)

    # Prepare an output array
    H, W, _ = semantic_bev.shape
    class_id_map = np.zeros((H, W), dtype=np.uint8)

    # Build a lookup from color -> class_id for fast matching
    # (R, G, B) => class_id
    color_to_id = {}
    for idx, color in enumerate(COLOR_MAP):
        color_to_id[color] = idx

    # Reshape for easier processing
    # shape: (H*W, 3)
    flat = semantic_bev.reshape(-1, 3)
    
    # For each pixel, find which class color it matches
    # Naive approach: for each pixel, see if its color is in color_to_id.
    # In practice, you might do something more sophisticated if needed.
    for i in range(flat.shape[0]):
        pixel_color = tuple(flat[i])
        if pixel_color in color_to_id:
            flat[i] = color_to_id[pixel_color]
        else:
            # If color not found in map, treat as 'unlabeled' (class_id=0)
            flat[i] = 0

    # Reshape back to (H, W)
    class_id_map = flat[:, 0].reshape(H, W)
    return class_id_map

def create_local_terrain_grid_map(
    class_id_map: np.ndarray,
    bev_px_per_meter: float = 13.6,
    desired_resolution_m: float = 0.5,
    grid_width_m: float = 40.0,
    grid_height_m: float = 40.0
) -> np.ndarray:
    """
    Create a local terrain (class ID) grid map around the ego-robot using 
    a downsample/aggregation from the raw BEV class ID map.
    
    The center of class_id_map is assumed to be the robot's position.
    
    Args:
        class_id_map: 2D np.ndarray of shape (H, W) with class IDs.
        bev_px_per_meter: float, how many pixels in the BEV correspond to 1 meter.
        desired_resolution_m: float, the desired resolution in the output grid (meters per cell).
        grid_width_m: width of the local grid in meters.
        grid_height_m: height of the local grid in meters.
        
    Returns:
        A 2D np.ndarray (grid_height_cells, grid_width_cells) with class IDs.
        - The center of this grid corresponds to the ego-robot (0,0).
        - Index 0,0 in the output grid is top-left (i.e. "front-left" in typical BEV).
    """
    H, W = class_id_map.shape
    center_y = H // 2
    center_x = W // 2

    # How many cells we want in the final grid
    num_cells_x = int(np.round(grid_width_m / desired_resolution_m))
    num_cells_y = int(np.round(grid_height_m / desired_resolution_m))
    
    # Prepare the output local grid
    local_grid = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)
    
    # Number of original BEV pixels per desired grid cell
    px_per_cell = bev_px_per_meter * desired_resolution_m
    
    # For each cell in the local grid, figure out which region of the original
    # class_id_map it corresponds to. We'll do a simple "nearest pixel" or 
    # "average"/"majority" approach. Below is a simple nearest approach.
    
    for gy in range(num_cells_y):
        for gx in range(num_cells_x):
            # Real-world offset from ego in meters
            # Top-left in the local grid is negative in x, positive in y if 'front' is up.
            # But let's define the coordinate frame so that 
            #  gy=0 => front of the robot (negative in BEV row index), 
            #  gx=0 => left of the robot (negative in BEV column index).

            # Convert (gx, gy) to offsets in meters relative to the center
            #   offset_x_m = (gx - num_cells_x/2) * desired_resolution_m
            #   offset_y_m = (gy - num_cells_y/2) * desired_resolution_m (but up is negative row)
            
            offset_x_m = (gx - num_cells_x / 2.0) * desired_resolution_m
            offset_y_m = (gy - num_cells_y / 2.0) * desired_resolution_m

            # Convert those offsets to pixel coordinates in the original BEV
            # We assume:
            #  +X in real world is to the right in BEV => pixel_x = center_x + offset_x_m * px_per_m
            #  +Y in real world is forward => pixel_y = center_y - offset_y_m * px_per_m
            #    (since row index decreases as we go "up" in the image)
            
            pixel_x = center_x + offset_x_m * bev_px_per_meter
            pixel_y = center_y - offset_y_m * bev_px_per_meter
            
            # Round to nearest pixel
            px_x = int(round(pixel_x))
            px_y = int(round(pixel_y))
            
            # Check bounds
            if 0 <= px_x < W and 0 <= px_y < H:
                local_grid[gy, gx] = class_id_map[px_y, px_x]
            else:
                # Out of original map => you might set to unlabeled or something else
                local_grid[gy, gx] = 0  # unlabeled

    return local_grid

if __name__ == "__main__":
    # Example usage:
    # 1) Suppose we have a semantic BEV image of shape (604, 964, 3).
    # 2) Convert it to a class_id_map.
    # 3) Create a local grid map around the ego with some desired resolution.

    # Fake example BEV image - random colors, just for demonstration
    # In practice, you'll load your actual semantic BEV image from disk or elsewhere.
    H, W = 604, 964
    fake_bev_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Let's say we paint some pixels like 'road' color near the center
    fake_bev_image[300:310, 480:488] = (128, 64, 128)  # road color in COLOR_MAP

    # Convert the color-coded semantic image to class IDs
    class_id_map = color_to_class_id(fake_bev_image)

    # Create a local terrain grid map with 0.5 m resolution,
    # covering 40m x 40m around the ego.
    local_grid = create_local_terrain_grid_map(
        class_id_map,
        bev_px_per_meter=13.6,
        desired_resolution_m=0.5,
        grid_width_m=40.0,
        grid_height_m=40.0
    )

    print("Local grid shape:", local_grid.shape)
    print("Sample of local grid (center region):")
    print("Resolutions: %.2f m/cell" % 0.5)
    cy, cx = local_grid.shape[0]//2, local_grid.shape[1]//2
    print(local_grid[cy-5:cy+5, cx-5:cx+5])
