pyDatalog.create_terms('RouteHasStop, DirectRoute, CanReachWithTransfer, X, Y, Z, R, R1, R2, Stops')

# Sample Data - You would replace this with your actual data
# Updated route_to_stops with additional data for testing
route_to_stops = {
    10153: [22540, 4686, 2573],
    1407: [4686, 2573],
    294: [951, 300, 340],
    712: [300, 340],
    1211: [951, 300, 340],        # Add additional routes for new test cases
    10453: [951, 300, 340],
    387: [951, 300, 340],
    49: [951, 300, 340],
    1571: [951, 300, 340],
    37: [951, 300, 340],
    1038: [951, 300, 340],
    10433: [951, 300, 340],
    121: [951, 300, 340]
}

# Add the data to PyDatalog globally
for route, stops in route_to_stops.items():
    for stop in stops:
        +RouteHasStop(route, stop)

# Define a direct route rule for stops on the same route
DirectRoute(X, Y, R) <= (RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y))

# Define a rule for reaching an endpoint with a transfer specifically at a via stop
CanReachWithTransfer(X, Y, Z, R1, R2) <= (
    DirectRoute(X, Z, R1) & DirectRoute(Z, Y, R2) & (R1 != R2)
)

# Updated forward chaining function for optimal route planning with stricter transfer logic
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering a single specific transfer.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    paths = []

    # Find possible paths with a transfer at the specified via stop
    valid_paths = pyDatalog.ask(f"CanReachWithTransfer({start_stop_id}, {end_stop_id}, {stop_id_to_include}, R1, R2)")

    # Process results with stricter transfer requirements
    if valid_paths is not None:
        for answer in valid_paths.answers:
            route1, route2 = answer[0], answer[1]
            if route1 != route2:  # Ensure there is a transfer
                paths.append((route1, stop_id_to_include, route2))

    return paths