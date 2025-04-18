# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = None                # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    # Create trip_id to route_id mapping
    for tmp, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    for tmp, row in df_stop_times.iterrows():
        route_id = trip_to_route.get(row['trip_id'])
        if route_id:
            if route_id not in route_to_stops:
                route_to_stops[route_id] = []
            route_to_stops[route_id].append((row['stop_sequence'], row['stop_id']))
            
            # Count trips per stop
            stop_trip_count[row['stop_id']] += 1

    # Ensure each route only has unique stops
    for route_id, stops in route_to_stops.items():
        stops = [stop for stop in stops if isinstance(stop, tuple) and len(stop) == 2]
        unique_stops = sorted(set(stops), key=lambda x: x[0])
        route_to_stops[route_id] = [stop_id for _, stop_id in unique_stops]

    # Create fare rules for routes
    fare_rules = df_fare_rules.set_index('route_id').T.to_dict()

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='inner')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    route_trip_count = defaultdict(int)

    for trip_id, route_id in trip_to_route.items():
        route_trip_count[route_id] += 1

    busiest_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return busiest_routes

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    most_frequent_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return most_frequent_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_to_routes = defaultdict(set)

    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_to_routes[stop_id].add(route_id)

    stop_route_count = {stop_id: len(routes) for stop_id, routes in stop_to_routes.items()}

    top_5_busiest_stops = sorted(stop_route_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return top_5_busiest_stops

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    stop_pairs = defaultdict(lambda: defaultdict(int))

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            pair = (stops[i], stops[i + 1])
            stop_pairs[pair][route_id] += 1
    single_route_pairs = [(pair, route_id) for pair, routes in stop_pairs.items() if len(routes) == 1 for route_id in routes]
    sorted_pairs = sorted(single_route_pairs, key=lambda x: stop_trip_count[x[0][0]] + stop_trip_count[x[0][1]], reverse=True)[:5]
    return sorted_pairs

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    G = nx.Graph()

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route_id)

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_text.append(f"Route: {edge[2]['route']}")

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Stop ID: {node}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='#00bfff',
            line_width=2),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Stop-Route Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    # Save as HTML for viewing in a browser
    fig.write_html("stop_route_graph.html")
    print("Plot saved as 'stop_route_graph.html'. Open this file in a browser to view the interactive plot.")
    
    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    direct_routes = []

    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            direct_routes.append(route_id)

    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, Action, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(R, X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y))

    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog
    
# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            +RouteHasStop(route_id, stop_id)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    query_result = pyDatalog.ask("DirectRoute(R, {}, {})".format(start, end))
    if query_result is None:
        return []

    route_ids = [answer[0] for answer in query_result.answers]

    return sorted(route_ids)

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    paths = []

    OptimalRoute(R1, R2, stop_id_to_include) <= (
        DirectRoute(R1, start_stop_id, stop_id_to_include) & 
        DirectRoute(R2, stop_id_to_include, end_stop_id)
    )

    valid_paths = pyDatalog.ask(f"OptimalRoute(R1, R2, {stop_id_to_include})")

    if valid_paths:
        for answer in valid_paths.answers:
            route1, route2 = answer[0], answer[1]
            paths.append((route1, stop_id_to_include, route2))

    if paths:
        return paths
    else:
        return []

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    paths = []

    OptimalRoute(X, Y, Z) <= (
        DirectRoute(X, start_stop_id, Z) & 
        DirectRoute(Y, Z, end_stop_id)
    )

    valid_paths = OptimalRoute(X, Y, stop_id_to_include)

    paths.append((valid_paths[0][1], stop_id_to_include, valid_paths[0][0]))

    for tmp in range(1, len(valid_paths)):
        route1, route2 = valid_paths[tmp][0], valid_paths[tmp][1]

        if route1 == route2:
            continue
        else:
            paths.append((route2, stop_id_to_include, route1))

    if paths:
        return paths
    else:
        return []

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id1 (int): The ID of the first route.
              - stop_id (int): The ID of the intermediate stop.
              - route_id2 (int): The ID of the second route.
    """
    Action('board_route', R, X) <= RouteHasStop(R, X)
    Action('transfer_route', R1, R2, Z) <= (
        DirectRoute(R1, start_stop_id, Z) & 
        DirectRoute(R2, Z, end_stop_id)
    )

    result = Action('board_route', X, start_stop_id) & \
             Action('transfer_route', X, Y, stop_id_to_include) & \
             Action('board_route', Y, end_stop_id)

    paths = []
    
    for action_set in result:
        route1, route2 = action_set[0], action_set[1]
        paths.append((route1, stop_id_to_include, route2))

    if paths:
        return paths
    else:
        return []

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pruned_df = merged_fare_df[merged_fare_df['price'] <= initial_fare]
    return pruned_df

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    route_summary = {}

    for route_id, group in pruned_df.groupby('route_id'):
        min_price = group['price'].min()
        stops = set(group['origin_id']).union(set(group['destination_id']))
        
        route_summary[route_id] = {
            'min_price': min_price,
            'stops': stops
        }

    return route_summary

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    queue = deque([(start_stop_id, initial_fare, [], 0)])
    visited = set()
    
    while queue:
        current_stop, remaining_fare, path, transfers = queue.popleft()

        if current_stop == end_stop_id:
            return path

        if transfers > max_transfers:
            continue

        for route_id, route_info in route_summary.items():
            if current_stop in route_info['stops'] and route_info['min_price'] <= remaining_fare:
                for stop in route_info['stops']:
                    if stop != current_stop:
                        new_path = path + [(route_id, stop)]
                        new_fare = remaining_fare - route_info['min_price']
                        new_transfers = transfers + 1 if route_id not in [r[0] for r in path] else transfers

                        if (stop, new_fare) not in visited:
                            visited.add((stop, new_fare))
                            queue.append((stop, new_fare, new_path, new_transfers))

    return []