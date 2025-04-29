import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = float('inf')  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to goal
        self.f = float('inf')  # Total cost (f = g + h)

    def __lt__(self, other):
        return self.f < other.f

def heuristic(current, goal):
    """Calculate the Manhattan distance heuristic."""
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def get_neighbors(node, grid):
    """Get all valid neighbors of the current node."""
    neighbors = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
    for d in directions:
        new_pos = (node.position[0] + d[0], node.position[1] + d[1])
        if 0 <= new_pos[0] < len(grid) and 0 <= new_pos[1] < len(grid[0]):
            neighbors.append(Node(new_pos, node))
    return neighbors

def reconstruct_path(node):
    """Reconstruct the path from start to goal."""
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def a_star(grid, start, goal):
    """Perform A* search with varying grid costs."""
    open_list = []
    closed_set = set()

    start_node = Node(start)
    start_node.g = 0
    start_node.f = heuristic(start, goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # If we reached the goal, reconstruct the path
        if current_node.position == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_node.position)

        # Explore neighbors
        for neighbor in get_neighbors(current_node, grid):
            if neighbor.position in closed_set:
                continue

            # Calculate the tentative g value (cost to reach this neighbor)
            terrain_cost = grid[neighbor.position[0]][neighbor.position[1]]
            tentative_g = current_node.g + terrain_cost

            # If this path to neighbor is better, record it
            if tentative_g < neighbor.g:
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor.position, goal)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node

                # Add neighbor to open list if not already there
                heapq.heappush(open_list, neighbor)

    return None  # No path found

# Example usage
if __name__ == "__main__":
    # Example grid with varying costs (0 = obstacle, higher numbers = higher cost)
    grid = [
        [1, 1, 1, 1, 1, 1],
        [1, 5, 1, 10, 10, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 10, 10, 10, 10, 1],
        [1, 1, 1, 1, 1, 1]
    ]
    start = (0, 0)
    goal = (4, 5)
    
    path = a_star(grid, start, goal)
    if path:
        print("Path found:", path)
    else:
        print("No path found")
