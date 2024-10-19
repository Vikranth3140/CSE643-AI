# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx

from pyDatalog import pyDatalog

from collections import defaultdict, deque
from itertools import combinations
from datetime import datetime

import os

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing


route_to_stops = defaultdict(list)  # Maps route_id to an ordered list of stop_ids
trip_to_route = {}  # Maps trip_id to route_id
stop_trip_count = defaultdict(int)  # Maps stop_id to count of trips stopping there
fare_rules = {}  # Maps route_id to fare information


## Q1: Data Loading and Knowledge Base Creation
# Function to load the OTD static data
def load_static_data():
    """
    Purpose: 
        Load the provided OTD static data and store it in Python data types.

    Expected Input:
        - None

    Expected Output:
        - Dictionary containing the loaded data for routes, trips, stops, stop times, and fare rules.

    """
    
    # Loading the data into pandas DataFrames
    data = {}
    
    # Load each dataset using pandas
    data['routes'] = pd.read_csv('GTFS/routes.txt', dtype={'route_id': str, 'agency_id': str, 'route_type': int})
    data['trips'] = pd.read_csv('GTFS/trips.txt', dtype={'route_id': str, 'trip_id': str, 'service_id': str})
    data['stops'] = pd.read_csv('GTFS/stops.txt', dtype={'stop_id': str, 'stop_code': str, 'stop_lat': float, 'stop_lon': float, 'stop_name': str, 'zone_id': str})

    # For stop_times.txt, we need to convert time fields to handle "24+" hour format
    stop_times = pd.read_csv('GTFS/stop_times.txt', dtype={'trip_id': str, 'stop_id': str, 'stop_sequence': int})
    
    # Convert time strings to "normalized" format (handling times like '24:xx:xx')
    stop_times['arrival_time'] = stop_times['arrival_time'].apply(convert_gtfs_time)
    stop_times['departure_time'] = stop_times['departure_time'].apply(convert_gtfs_time)
    
    data['stop_times'] = stop_times
    
    data['fare_rules'] = pd.read_csv('GTFS/fare_rules.txt', dtype={'fare_id': str, 'route_id': str, 'origin_id': str, 'destination_id': str})
    data['fare_attributes'] = pd.read_csv('GTFS/fare_attributes.txt', dtype={'fare_id': str, 'price': float, 'currency_type': str})
    
    return data

def convert_gtfs_time(time_str):
    """
    Function to convert GTFS time format to handle values like "24:xx:xx".
    GTFS time may exceed 24 hours if the service runs past midnight.
    """
    # Split the time into hours, minutes, and seconds
    h, m, s = map(int, time_str.split(':'))
    
    # If the hour is 24 or more, convert to standard time without using timedelta
    if h >= 24:
        h = h - 24  # Normalize the hour by subtracting 24
    
    # Format the time back to 'HH:MM:SS' string
    return f'{h:02}:{m:02}:{s:02}'

# Function to create the Knowledge Base (KB)
def create_knowledge_base():
    """
    Purpose: 
        Set up the knowledge base (KB) for reasoning and planning tasks.

    Expected Input:
        - None

    Expected Output:
        - Dictionary mapping route to stops, trip to route, and stop trip count.
    """
    # Load the static data
    data = load_static_data()  # Call to the data loading function
    
    # Initializing the dictionaries
    route_to_stops = defaultdict(list)  # Maps route_id to a list of stop_ids
    trip_to_route = {}  # Maps trip_id to route_id
    stop_trip_count = defaultdict(int)  # Maps stop_id to the count of trips stopping there
    
    # Step 1: Build trip_to_route using 'trips' DataFrame
    trips_df = data['trips']
    
    for _, row in trips_df.iterrows():
        trip_id = row['trip_id']
        route_id = row['route_id']
        trip_to_route[trip_id] = route_id
    
    # Step 2: Build route_to_stops and stop_trip_count using 'stop_times' DataFrame
    stop_times_df = data['stop_times']
    
    for _, row in stop_times_df.iterrows():
        trip_id = row['trip_id']
        stop_id = row['stop_id']
        
        # Get the route_id for the current trip_id
        route_id = trip_to_route[trip_id]
        
        # Append stop_id to the list of stops for the route
        route_to_stops[route_id].append(stop_id)
        
        # Increment the stop_trip_count for this stop_id
        stop_trip_count[stop_id] += 1
    
    return {
        'route_to_stops': dict(route_to_stops),
        'trip_to_route': trip_to_route,
        'stop_trip_count': dict(stop_trip_count)
    }

# Function to find the busiest routes based on the number of trips
def get_busiest_routes():
    """
    Purpose: 
        Identify the busiest routes based on the number of trips.

    Expected Input:
        - None

    Expected Output:
        - List of route IDs sorted by the number of trips in descending order.
    """
    # Step 1: Create a count of trips per route using trip_to_route dictionary
    # Assuming the KB has been set up and we have access to `trip_to_route`
    
    # We need to create trip_to_route first, assuming it's available in the KB
    knowledge_base = create_knowledge_base()  # Load the knowledge base
    trip_to_route = knowledge_base['trip_to_route']
    
    # Create a dictionary to count the number of trips per route
    route_trip_count = defaultdict(int)
    
    for trip_id, route_id in trip_to_route.items():
        route_trip_count[route_id] += 1
    
    # Step 2: Sort the routes by the number of trips in descending order
    sorted_routes = sorted(route_trip_count.items(), key=lambda x: x[1], reverse=True)
    
    # Step 3: Return the sorted list of route IDs
    # We only need the route_id, not the count, so we'll return just the route IDs
    busiest_routes = [route for route, count in sorted_routes]
    
    return busiest_routes

# Function to find the stops with the most frequent trips
def get_most_frequent_stops():
    """
    Purpose: 
        Find the stops with the most frequent trips.

    Expected Input:
        - None

    Expected Output:
        - List of stop IDs sorted by the frequency of trips in descending order.
    """
    # Step 1: Access the `stop_trip_count` from the Knowledge Base (KB)
    # Assuming the KB has been set up and we have access to `stop_trip_count`
    
    knowledge_base = create_knowledge_base()  # Load the knowledge base
    stop_trip_count = knowledge_base['stop_trip_count']
    
    # Step 2: Sort the stops by the number of trips in descending order
    sorted_stops = sorted(stop_trip_count.items(), key=lambda x: x[1], reverse=True)
    
    # Step 3: Return the sorted list of stop IDs
    # We only need the stop_id, not the count, so we'll return just the stop IDs
    most_frequent_stops = [stop for stop, count in sorted_stops]
    
    return most_frequent_stops

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Purpose: 
        Identify the top 5 busiest stops based on the number of routes passing through them.

    Expected Input:
        - None

    Expected Output:
        - List of the top 5 stop IDs sorted by the number of routes passing through them.
    """
    pass

# Function to find pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Purpose: 
        Find pairs of stops (start and end) that have only one direct route between them.

    Expected Input:
        - None

    Expected Output:
        - List of tuples representing pairs of stop IDs with one direct route between them.
    """
    pass

# Function to create a graph representation using Plotly
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Purpose: 
        Create a graph representation of the knowledge base using the route to stops mapping.

    Expected Input:
        - route_to_stops: mapped route to stop ids

    Expected Output:
        - Interactive Graph representation using Plotly.
    """
    pass


# Q.2: Reasoning
# Brute-Force Approach for DirectRoute function
def direct_route_brute_force(start_stop, end_stop, kb):
    """
    Purpose: 
        Find all direct routes between two stops using a brute-force approach.

    Expected Input:
        - start_stop: ID of the start stop.
        - end_stop: ID of the end stop.

    Expected Output:
        - List of route IDs connecting the start stop to the end stop directly (no interchanges).
    """
    pass

# Create terms
# define predicates

# adding facts to Knowledge Base
def add_route_data(route_to_stops):
    """
    Purpose: 
        Add route to stop mappings to knowledge base.

    Expected Input:
        - route_to_stops: mapping created, which maps route id to stop ids.

    Expected Output:
        - None
    """
    pass


# defining query functions
def query_direct_routes(start, end):
    """
    Purpose: 
        Find all direct routes between two stops using the PyDatalog library.

    Expected Input:
        - start_stop: ID of the start stop.
        - end_stop: ID of the end stop.

    Expected Output:
        - List of route IDs connecting the start stop to the end stop directly (no interchanges).
    """

    # Test cases: 

    # 1. 
    # I/p - (2573, 1177) 
    # O/p - [10001, 1117, 1407]

    # 2. 
    # I/p - (2001, 2005)
    # O/p - [10001, 1151]

    pass

# Planning: Forward Chaining for Optimal Route

# Create terms
# Define predicates
# Add facts to knowledge base

def forward_chaining(start_stop, end_stop, via_stop, max_transfers):
    """
    Purpose: 
        Plan an optimal route using Forward Chaining.

    Expected Input:
        - start_stop: ID of the start stop.
        - end_stop: ID of the end stop.
        - via_stop: ID of the intermediate stop.
        - max_transfers: Maximum number of route interchanges allowed.

    Expected Output:
        - List of optimal route IDs with respect to the constraints.
        - output format: list of (route_id1, via_stop_id, route_id2) 
    """
    pass

# Planning: Backward Chaining for Optimal Route
def backward_chaining_planning(start_stop, end_stop, via_stop, max_transfers, kb):
    """
    Purpose: 
        Plan an optimal route using Backward Chaining.

    Expected Input:
        - start_stop: ID of the start stop.
        - end_stop: ID of the end stop.
        - via_stop: ID of the intermediate stop.
        - max_transfers: Maximum number of route interchanges allowed.

    Expected Output:
        - List of optimal route IDs with respect to the constraints.
        - output format: list of (route_id1, via_stop_id, route_id2) 
    """
    pass


# Create terms
# Define predicates for routes and states
# Define initial and goal state
# Add facts to knowledge base


# Planning using PDLL (Planning Domain Definition Language)
def pdll_planning(start_stop, end_stop, via_stop, max_transfers):
    """
    Purpose: 
        Plan an optimal route using PDLL.

    Expected Input:
        - start_stop: ID of the start stop.
        - end_stop: ID of the end stop.
        - via_stop: ID of the intermediate stop.
        - max_transfers: Maximum number of route interchanges allowed.

    Expected Output:
        - List of optimal route IDs with respect to the constraints.
        - output format: list of (route_id1, via_stop_id, route_id2)
        - print the state information at each step
        - example:   
            Step 1: 
                Action: Board Route 10153 at Stop 22540
                Current State: At Stop 22540 on Route 10153
                Current Path: [(10153, 22540)]
    """
    pass

# Public test cases for all three parts: 
# [start_id, stop_id, intermediate_stop_id, max_transfers]

# 1. 
# I/p - [22540, 2573, 4686, 1]
# O/p - [(10153, 4686, 1407)]

# 2. 
# I/p - [951, 340, 300, 1]
# O/p - [(294, 300, 712),
#  (10453, 300, 712),
#  (1211, 300, 712),
#  (1158, 300, 712),
#  (37, 300, 712),
#  (1571, 300, 712),
#  (49, 300, 712),
#  (387, 300, 712),
#  (1206, 300, 712),
#  (1038, 300, 712),
#  (10433, 300, 712),
#  (121, 300, 712)]


# Bonus: Extend Planning by Considering Fare Constraints


# Data Pruning
def prune_data(merged_fare_df, initial_fare):
    # Filter routes that have minimum fare less than or equal to initial fare
    """
    Purpose: 
        Use merged fare dataframes and prune the data to filter out routes.

    Expected Input:
        - merged_fare_df: merging fare rules df and fare attributes df
        - initial_fare: some initial fare value to be passed as a parameter

    Expected Output:
        - pruned_df: pruned merged_fare_df 
    """
    pass


# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Purpose: 
        Pre-compute a summary of each route, including the minimum price and the set of stops for each route.

    Expected Input:
        - pruned_df: A DataFrame with at least the following columns:
            - 'route_id': The ID of the route.
            - 'origin_id': The ID of the stop.
            - 'price': The price associated with the route and stop.

    Expected Output:
        - route_summary: A dictionary where:
            - Keys are route IDs.
            - Values are dictionaries containing:
                - 'min_price': The minimum price found for the route.
                - 'stops': A set of all unique stop IDs for the route.
    """
    pass


def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Purpose: 
        Perform a breadth-first search (BFS) to find an optimized route from a start stop to an end stop 
        considering the fare and transfer limits.

    Expected Input:
        - start_stop_id: The ID of the starting stop.
        - end_stop_id: The ID of the destination stop.
        - initial_fare: The total fare available for the journey.
        - route_summary: A dictionary with route summaries containing:
            - 'stops': A set of stops for each route.
            - 'min_price': The minimum fare for the route.
        - max_transfers: The maximum number of transfers allowed (default is 3).

    Expected Output:
        - result: A list representing the optimal path taken, or None if no valid route is found.

    Note:
        The function prints detailed steps of the search process, including actions taken and current state.
        Output format: [(route_id1, intermediate_stop_id1), (route_id2, intermediate_stop_id2), …, (route_idn, end_stop_id)]
        Example: 
            Step 1:
                Action: Move to 1562 on Route 10004
                Current State: At Stop 22540 on Route None
                Current Path: [(10004, 1562)]
                Remaining Fare: 5.0
    """

    # test case: 
    # I/p - [22540, 2573, 10, 3]
    # O/p - [(10153, 4686), (1407, 2573)]
    pass
