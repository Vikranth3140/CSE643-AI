# initialize_datalog()
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

    # Define OptimalRoute using DirectRoute with start, intermediate (transfer), and end stops
    OptimalRoute(R1, R2, stop_id_to_include) <= (
        DirectRoute(R1, start_stop_id, stop_id_to_include) & 
        DirectRoute(R2, stop_id_to_include, end_stop_id)
    )

    # Query for optimal routes that satisfy the transfer condition
    valid_paths = pyDatalog.ask(f"OptimalRoute(R1, R2, {stop_id_to_include})")

    # Process the results if valid paths are found
    if valid_paths is not None:
        for answer in valid_paths.answers:
            route1, route2 = answer[0], answer[1]
            paths.append((route1, stop_id_to_include, route2))

    return paths


test = [ ((22540, 2573, 4686, 1), [(10153, 4686, 1407)]), ((951, 340, 300, 1), [(294, 300, 712), (10453, 300, 712), (1211, 300, 712), (1158, 300, 712), (37, 300, 712), (1571, 300, 712), (49, 300, 712), (387, 300, 712), (1206, 300, 712), (1038, 300, 712), (10433, 300, 712), (121, 300, 712)])]
# print('out : ',forward_chaining(951, 340, 300, 1))
test_inputs = {
    "forward_chaining": [
        ((22540, 2573, 4686, 1), [(10153, 4686, 1407)]),
        ((951, 340, 300, 1), [(1211, 300, 712), (10453, 300, 712), (387, 300, 712), (49, 300, 712), 
                              (1571, 300, 712), (37, 300, 712), (1038, 300, 712), (10433, 300, 712), 
                              (121, 300, 712)])
    ]
}

def check_output(expected, actual):
    """Function to compare expected and actual outputs."""
    if isinstance(expected, list) and isinstance(actual, list):
        return sorted(expected) == sorted(actual)  # Ensures order-independent comparison
    return expected == actual  # For non-list types

def test_forward_chaining():
    for (start_stop, end_stop, via_stop, max_transfers), expected_output in test_inputs["forward_chaining"]:
        actual_output = forward_chaining(start_stop, end_stop, via_stop, max_transfers)
        print(f"Test forward_chaining ({start_stop}, {end_stop}, {via_stop}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (\n\n\nExpected: {expected_output},\n\n Got: {actual_output})")

test_forward_chaining()