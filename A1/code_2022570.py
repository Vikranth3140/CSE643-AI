import numpy as np
import pickle
from tqdm import tqdm
import heapq
import math

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def is_reachable_bfs(adj_matrix, start_node, goal_node):
    visited = set()
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node == goal_node:
            return True
        visited.add(node)
        
        for neighbor in range(len(adj_matrix[node])):
            if adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                queue.append(neighbor)
    return False


def depth_limited_search(adj_matrix, start_node, goal_node, limit, visited=None):
    if visited is None:
        visited = set()
    
    if start_node == goal_node:
        return [start_node]
    
    if limit <= 0:
        return None
    
    visited.add(start_node)

    for neighbor in range(len(adj_matrix[start_node])):
        if adj_matrix[start_node][neighbor] > 0 and neighbor not in visited:
            path = depth_limited_search(adj_matrix, neighbor, goal_node, limit - 1, visited)
            if path:
                return [start_node] + path
    
    visited.remove(start_node)
    
    return None


def get_ids_path(adj_matrix, start_node, goal_node, max_depth=float('inf')):
    if not is_reachable_bfs(adj_matrix, start_node, goal_node):
        return None
    
    max_depth = int(max_depth) if max_depth != float('inf') else 1000
    for depth in tqdm(range(max_depth + 1), desc="Searching with depth limits"):
        visited = set()
        path = depth_limited_search(adj_matrix, start_node, goal_node, depth, visited)
        if path:
            return path
    return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def push(queue, node, cost):
    heapq.heappush(queue, (cost, node))

def pop(queue):
    return heapq.heappop(queue)[1]

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    n = len(adj_matrix)

    if start_node == goal_node:
        return [start_node]

    frontierF = [(0, start_node)]
    frontierB = [(0, goal_node)]

    reachedF = {start_node: 0}
    reachedB = {goal_node: 0}

    src_parent = [-1] * n
    dest_parent = [-1] * n

    def expand(frontier, reached, parent):
        node = pop(frontier)
        for neighbor, is_connected in enumerate(adj_matrix[node]):
            if is_connected:
                new_cost = reached[node] + 1
                if neighbor not in reached or new_cost < reached[neighbor]:
                    reached[neighbor] = new_cost
                    push(frontier, neighbor, new_cost)
                    parent[neighbor] = node

    def is_intersecting():
        for node in reachedF:
            if node in reachedB:
                return node
        return -1

    def construct_path(intersecting_node):
        path = []

        current = intersecting_node
        while current != -1:
            path.append(current)
            current = src_parent[current]
        path.reverse()

        current = dest_parent[intersecting_node]
        while current != -1:
            path.append(current)
            current = dest_parent[current]

        return path

    while frontierF and frontierB:
        if frontierF[0][0] <= frontierB[0][0]:
            expand(frontierF, reachedF, src_parent)
        else:
            expand(frontierB, reachedB, dest_parent)

        intersecting_node = is_intersecting()

        if intersecting_node != -1:
            return construct_path(intersecting_node)

    return None


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

# Heuristic function: Euclidean distance between two nodes based on their coordinates
def heuristic(node1, node2, node_attributes):
    node1 = int(node1)
    node2 = int(node2)
    
    # Access the 'x' and 'y' coordinates from the node_attributes dictionary
    x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
    x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
    
    # Calculate and return the Euclidean distance
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Example usage in your A* algorithm
def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)  # Number of nodes in the graph

    # Priority queue for A* search
    open_list = []
    heapq.heappush(open_list, (0, start_node))  # (f_cost, node)
    
    # To store the best known cost to reach each node
    g_costs = {i: float('inf') for i in range(n)}
    g_costs[start_node] = 0
    
    # To track the path
    came_from = {start_node: None}
    
    # To store f-cost for each node
    f_costs = {i: float('inf') for i in range(n)}
    f_costs[start_node] = heuristic(start_node, goal_node, node_attributes)
    
    while open_list:
        # Get the node with the lowest f_cost
        current_f, current_node = heapq.heappop(open_list)
        
        # If the goal is reached, reconstruct and return the path
        if current_node == goal_node:
            return reconstruct_path(came_from, current_node)
        
        # Explore neighbors
        for neighbor, is_connected in enumerate(adj_matrix[current_node]):
            if is_connected:  # Only consider connected neighbors
                tentative_g_cost = g_costs[current_node] + 1  # Assuming uniform cost
                
                # If we find a better path, update it
                if tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current_node
                    g_costs[neighbor] = tentative_g_cost
                    f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal_node, node_attributes)
                    heapq.heappush(open_list, (f_costs[neighbor], neighbor))
    
    return None  # Return None if no path is found

# Function to reconstruct the path from start to goal
def reconstruct_path(came_from, current_node):
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()
    return path



# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):

  return []



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):

  return []


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')