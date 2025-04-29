import math
import heapq

class Node:
    def __init__(self, x, y, theta, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # Orientation (in radians)
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h  # Total cost
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    """Euclidean distance heuristic."""
    return math.sqrt((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2)

def get_successors(node, grid, step_size=1.0, turning_angles=[-0.5, 0, 0.5], wheelbase=2.0):
    """Generate successors based on the vehicle kinematic model."""
    successors = []
    for angle in turning_angles:
        # Apply the kinematic model to get the new state
        new_theta = node.theta + angle
        new_x = node.x + step_size * math.cos(new_theta)
        new_y = node.y + step_size * math.sin(new_theta)

        # Check if the new position is within bounds and not an obstacle
        if 0 <= int(new_x) < len(grid) and 0 <= int(new_y) < len(grid[0]) and grid[int(new_x)][int(new_y)] == 0:
            successor = Node(new_x, new_y, new_theta, parent=node)
            successors.append(successor)
    return successors

def reconstruct_path(node):
    """Reconstruct the path from goal to start."""
    path = []
    while node:
        path.append((node.x, node.y, node.theta))
        node = node.parent
    return path[::-1]

def hybrid_a_star(grid, start, goal):
    """Hybrid A* algorithm for continuous space pathfinding."""
    open_list = []
    closed_set = set()

    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    start_node.h = heuristic(start_node, goal_node)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # Check if we reached the goal
        if heuristic(current_node, goal_node) < 1.0:
            return reconstruct_path(current_node)

        closed_set.add((int(current_node.x), int(current_node.y), round(current_node.theta, 1)))

        # Generate successors
        for neighbor in get_successors(current_node, grid):
            if (int(neighbor.x), int(neighbor.y), round(neighbor.theta, 1)) in closed_set:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h

            heapq.heappush(open_list, neighbor)

    return None

# Example usage
if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0]
    ]
    start = (0, 0, 0)  # (x, y, orientation in radians)
    goal = (4, 5, 0)
    
    path = hybrid_a_star(grid, start, goal)
    if path:
        print("Path found:", path)
    else:
        print("No path found")
