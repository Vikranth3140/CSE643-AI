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

    # 1. Convert IDs to strings (e.g., stop_id, trip_id, route_id)
    df_stops['stop_id'] = df_stops['stop_id'].astype(str)
    df_stop_times['trip_id'] = df_stop_times['trip_id'].astype(str)
    df_stop_times['stop_id'] = df_stop_times['stop_id'].astype(str)
    df_routes['route_id'] = df_routes['route_id'].astype(str)
    df_trips['trip_id'] = df_trips['trip_id'].astype(str)
    df_trips['route_id'] = df_trips['route_id'].astype(str)
    df_fare_rules['fare_id'] = df_fare_rules['fare_id'].astype(str)
    df_fare_rules['route_id'] = df_fare_rules['route_id'].astype(str)

    # 2. Convert time columns (arrival_time, departure_time) in stop_times to datetime
    df_stop_times['arrival_time'] = pd.to_datetime(df_stop_times['arrival_time'], format='%H:%M:%S', errors='coerce')
    df_stop_times['departure_time'] = pd.to_datetime(df_stop_times['departure_time'], format='%H:%M:%S', errors='coerce')

    # Create trip_id to route_id mapping
    for _, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    # Map route_id to a list of stops in order of their sequence
    stop_sequences = df_stop_times.groupby('trip_id').apply(lambda x: x.sort_values(by='stop_sequence')).reset_index(drop=True)

    for trip_id, stop_sequence in stop_sequences.groupby('trip_id'):
        route_id = trip_to_route[trip_id]  # Find route ID from trip_id
        stops_in_route = stop_sequence['stop_id'].tolist()
        route_to_stops[route_id].extend(stops_in_route)

    # Ensure each route only has unique stops
    for route_id, stops in route_to_stops.items():
        route_to_stops[route_id] = list(dict.fromkeys(stops))  # Removes duplicates while preserving order
    
    # Count trips per stop
    for stop_id in df_stop_times['stop_id']:
        stop_trip_count[stop_id] += 1

    # Create fare rules for routes
    for _, row in df_fare_rules.iterrows():
        route_id = row['route_id']
        fare_rules[route_id] = {
            'origin_id': row['origin_id'],
            'destination_id': row['destination_id'],
            'fare_id': row['fare_id']
        }
    
    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='left')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (str): The ID of the route.
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
              - stop_id (str): The ID of the stop.
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
              - stop_id (str): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    stop_to_routes = defaultdict(set)

    for route_id, stops in route_to_stops.items():
        for stop_id in stops:
            stop_to_routes[stop_id].add(route_id)

    stop_route_count = {stop_id: len(routes) for stop_id, routes in stop_to_routes.items()}

    top_5_busiest_stops = sorted(stop_route_count.items(), key=lambda x: x[1], reverse=True)[:5]

    return top_5_busiest_stops

# Function to find pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify stop pairs that are connected by exactly one direct route.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple containing two stop IDs (stop_1, stop_2).
              - route_id (str): The ID of the route connecting the two stops.
    """
    stop_pair_to_route = defaultdict(list)

    for route_id, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            stop_pair = (stops[i], stops[i + 1])
            reverse_pair = (stops[i + 1], stops[i])
            stop_pair_to_route[stop_pair].append(route_id)
            stop_pair_to_route[reverse_pair].append(route_id)

    result = []
    for stop_pair, routes in stop_pair_to_route.items():
        if len(routes) == 1:
            result.append((stop_pair, routes[0]))

    return result

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

    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (str): The ID of the starting stop.
        end_stop (str): The ID of the ending stop.

    Returns:
        list: A list of route IDs (str) that connect the two stops directly.
    """
    direct_routes = []

    for route_id, stops in route_to_stops.items():
        if start_stop in stops and end_stop in stops:
            if stops.index(start_stop) < stops.index(end_stop):
                direct_routes.append(route_id)

    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')  
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute")  # Confirmation print

    # Define Datalog predicates
    DirectRoute(X, Y) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X._index < Y._index))

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
        start (str): The ID of the starting stop.
        end (str): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    result = DirectRoute(X, Y) & (X == start) & (Y == end)

    route_ids = [route[R] for route in result]

    return sorted(route_ids)

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
    pass  # Implementation here

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
    pass  # Implementation here

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        stop_id_to_include (str): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (str): The ID of the route.
              - stop_id (str): The ID of the stop.
    """
    pass  # Implementation here

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
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (str): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (str): The starting stop ID.
        end_stop_id (str): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (str), stop_id (str)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here
