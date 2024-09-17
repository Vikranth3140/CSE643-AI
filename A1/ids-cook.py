import numpy as np
import pickle
from tqdm import tqdm
import heapq
import math
import psutil
import time
import matplotlib.pyplot as plt
from collections import deque

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
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        if node == goal_node:
            return True
        visited.add(node)

        for neighbor in range(len(adj_matrix[node])):
            if adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)  # Mark the node as visited here to prevent duplicate checks
                queue.append(neighbor)
    return False


def depth_limited_search(adj_matrix, start_node, goal_node, limit):
    stack = [(start_node, [start_node], limit)]  # Stack for the iterative DFS
    visited = set()
    
    while stack:
        node, path, depth = stack.pop()

        if node == goal_node:
            return path

        if depth > 0:
            for neighbor in range(len(adj_matrix[node])):
                if adj_matrix[node][neighbor] > 0 and neighbor not in path:  # Only proceed if not already in path
                    stack.append((neighbor, path + [neighbor], depth - 1))
    
    return None


def get_ids_path(adj_matrix, start_node, goal_node, max_depth=float('inf')):
    if not is_reachable_bfs(adj_matrix, start_node, goal_node):
        print(f"No path exists between {start_node} and {goal_node}. Skipping IDS.")
        return None

    max_depth = int(max_depth) if max_depth != float('inf') else 1000

    for depth in tqdm(range(max_depth + 1), desc="Searching with depth limits"):
        path = depth_limited_search(adj_matrix, start_node, goal_node, depth)
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

def heuristic(node1, node2, node_attributes):
    node1 = int(node1)
    node2 = int(node2)
    
    x1, y1 = node_attributes[node1]['x'], node_attributes[node1]['y']
    x2, y2 = node_attributes[node2]['x'], node_attributes[node2]['y']
    
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def reconstruct_path(came_from, current_node):
    path = []
    while current_node is not None:
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()
    return path

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    n = len(adj_matrix)

    open_list = []
    heapq.heappush(open_list, (0, start_node))
    
    g_costs = {i: float('inf') for i in range(n)}
    g_costs[start_node] = 0
    
    came_from = {start_node: None}
    
    f_costs = {i: float('inf') for i in range(n)}
    f_costs[start_node] = heuristic(start_node, goal_node, node_attributes)
    
    while open_list:
        temp, current_node = heapq.heappop(open_list)
        
        if current_node == goal_node:
            return reconstruct_path(came_from, current_node)
        
        for neighbor, edge_weight in enumerate(adj_matrix[current_node]):
            if edge_weight > 0:
                tentative_g_cost = g_costs[current_node] + edge_weight
                
                if tentative_g_cost < g_costs[neighbor]:
                    came_from[neighbor] = current_node
                    g_costs[neighbor] = tentative_g_cost
                    
                    h_cost = heuristic(neighbor, goal_node, node_attributes)
                    
                    f_costs[neighbor] = tentative_g_cost + h_cost
                    
                    heapq.heappush(open_list, (f_costs[neighbor], neighbor))
    
    return None


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
    if start_node == goal_node:
        return [start_node]
    
    def a_star_step(frontier, cost_so_far, other_cost, came_from, direction):
        current_f, current = heapq.heappop(frontier)
        if current in other_cost:
            return current
        
        for neighbor, cost in enumerate(adj_matrix[current]):
            if cost > 0:
                new_cost = cost_so_far[current] + cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal_node, node_attributes)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        return None

    frontier_fwd = [(0, start_node)]
    frontier_bwd = [(0, goal_node)]
    
    came_from_fwd = {start_node: None}
    came_from_bwd = {goal_node: None}
    
    cost_fwd = {start_node: 0}
    cost_bwd = {goal_node: 0}
    
    meeting_node = None

    while frontier_fwd and frontier_bwd:
        meeting_node = a_star_step(frontier_fwd, cost_fwd, cost_bwd, came_from_fwd, "forward")
        if meeting_node:
            break
        
        meeting_node = a_star_step(frontier_bwd, cost_bwd, cost_fwd, came_from_bwd, "backward")
        if meeting_node:
            break
    
    if meeting_node is None:
        return None
    
    def reconstruct_path(meeting_node):
        path_fwd = []
        current = meeting_node
        while current is not None:
            path_fwd.append(current)
            current = came_from_fwd[current]
        path_fwd.reverse()
        
        path_bwd = []
        current = came_from_bwd[meeting_node]
        while current is not None:
            path_bwd.append(current)
            current = came_from_bwd[current]
        
        return path_fwd + path_bwd
    
    return reconstruct_path(meeting_node)



# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def dfs(adj_matrix, node, parent, visited, disc, low, bridges, time):
    visited[node] = True
    disc[node] = low[node] = time[0]
    time[0] += 1

    for neighbor, connected in enumerate(adj_matrix[node]):
        if connected:
            if not visited[neighbor]:
                dfs(adj_matrix, neighbor, node, visited, disc, low, bridges, time)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    bridges.append((node, neighbor))
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])

def bonus_problem(adj_matrix):
    n = len(adj_matrix)
    visited = [False] * n
    disc = [float('inf')] * n
    low = [float('inf')] * n
    bridges = []
    time = [0]

    for i in range(n):
        if not visited[i]:
            dfs(adj_matrix, i, -1, visited, disc, low, bridges, time)
    
    return bridges

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





def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss // 1024


def performance_test(adj_matrix, node_attributes):
    num_nodes = len(node_attributes)

    ids_paths = []
    bds_paths = []
    astar_paths = []
    bhds_paths = []

    # IDS Performance Test
    start_memory_ids = get_memory_usage()
    start_time_ids = time.time()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Run IDS only if a path exists
            path = get_ids_path(adj_matrix, i, j)
            ids_paths.append((i, j, path))

    end_time_ids = time.time()
    end_memory_ids = get_memory_usage()

    print(f"Memory used for IDS: {end_memory_ids - start_memory_ids} KB")
    print(f"Time taken for IDS: {end_time_ids - start_time_ids} seconds")

    # BDS Performance Test
    start_memory_bds = get_memory_usage()
    start_time_bds = time.time()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            path = get_bidirectional_search_path(adj_matrix, i, j)
            bds_paths.append((i, j, path))

    end_time_bds = time.time()
    end_memory_bds = get_memory_usage()

    print(f"Memory used for BDS: {end_memory_bds - start_memory_bds} KB")
    print(f"Time taken for BDS: {end_time_bds - start_time_bds} seconds")

    # A* Performance Test
    start_memory_astar = get_memory_usage()
    start_time_astar = time.time()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            path = get_astar_search_path(adj_matrix, node_attributes, i, j)
            astar_paths.append((i, j, path))

    end_time_astar = time.time()
    end_memory_astar = get_memory_usage()

    print(f"Memory used for A*: {end_memory_astar - start_memory_astar} KB")
    print(f"Time taken for A*: {end_time_astar - start_time_astar} seconds")

    # BHDS Performance Test
    start_memory_bhds = get_memory_usage()
    start_time_bhds = time.time()

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            path = get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, i, j)
            bhds_paths.append((i, j, path))

    end_time_bhds = time.time()
    end_memory_bhds = get_memory_usage()

    print(f"Memory used for BHDS: {end_memory_bhds - start_memory_bhds} KB")
    print(f"Time taken for BHDS: {end_time_bhds - start_time_bhds} seconds")

    # Return the paths and performance data
    return {
        'IDS': {
            'paths': ids_paths,
            'memory_used': end_memory_ids - start_memory_ids,
            'time_taken': end_time_ids - start_time_ids
        },
        'BDS': {
            'paths': bds_paths,
            'memory_used': end_memory_bds - start_memory_bds,
            'time_taken': end_time_bds - start_time_bds
        },
        'A*': {
            'paths': astar_paths,
            'memory_used': end_memory_astar - start_memory_astar,
            'time_taken': end_time_astar - start_time_astar
        },
        'BHDS': {
            'paths': bhds_paths,
            'memory_used': end_memory_bhds - start_memory_bhds,
            'time_taken': end_time_bhds - start_time_bhds
        }
    }


results = performance_test(adj_matrix, node_attributes)

for algorithm, data in results.items():
    print(f"\n--- {algorithm} Paths ---")
    for path_info in data['paths']:
        start, goal, path = path_info
        print(f"Path from {start} to {goal}: {path if path else 'No path found'}")

for algorithm, data in results.items():
    print(f"\n--- {algorithm} Performance ---")
    print(f"Memory used: {data['memory_used']} KB")
    print(f"Time taken: {data['time_taken']} seconds")




# # Extracting data for plotting
# algorithms = list(results_data.keys())
# memory_usage = [results_data[alg]['memory_used'] for alg in algorithms]
# time_taken = [results_data[alg]['time_taken'] for alg in algorithms]
# path_lengths = [results_data[alg]['path_length'] for alg in algorithms]

# # Scatter Plot 1: Efficiency in terms of Time and Memory Usage
# plt.figure(figsize=(8, 6))
# plt.scatter(memory_usage, time_taken, c=['r', 'g', 'b'], s=100, label=algorithms)
# for i, alg in enumerate(algorithms):
#     plt.text(memory_usage[i], time_taken[i], alg, fontsize=12, ha='right')
# plt.title('Time vs. Memory Usage for Search Algorithms')
# plt.xlabel('Memory Used (KB)')
# plt.ylabel('Time Taken (seconds)')
# plt.grid(True)
# plt.show()

# # Scatter Plot 2: Efficiency in terms of Optimality (Path Length) and Time
# plt.figure(figsize=(8, 6))
# plt.scatter(path_lengths, time_taken, c=['r', 'g', 'b'], s=100, label=algorithms)
# for i, alg in enumerate(algorithms):
#     plt.text(path_lengths[i], time_taken[i], alg, fontsize=12, ha='right')
# plt.title('Path Length (Optimality) vs. Time for Search Algorithms')
# plt.xlabel('Path Length (Number of Nodes)')
# plt.ylabel('Time Taken (seconds)')
# plt.grid(True)
# plt.show()