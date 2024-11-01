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
    # pass  # Implementation here
    paths = []
    direct_routes_to_transfer = DirectRoute(R1, start_stop_id, stop_id_to_include)
    direct_routes_from_transfer = DirectRoute(R2, stop_id_to_include, end_stop_id)
    
    print("Direct routes to transfer stop:", direct_routes_to_transfer)
    print("Direct routes from transfer stop:", direct_routes_from_transfer)

    routes_to_transfer = [route[0] for route in direct_routes_to_transfer]  
    routes_from_transfer = [route[0] for route in direct_routes_from_transfer] 
    for route_to_transfer in routes_to_transfer:
        for route_from_transfer in routes_from_transfer:
            # Append as (route_id1, stop_id_to_include, route_id2)
            paths.append((route_to_transfer, stop_id_to_include, route_from_transfer))

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
def test_forward_chaining():
    for (start_stop, end_stop, via_stop, max_transfers), expected_output in test_inputs["forward_chaining"]:
        actual_output = forward_chaining(start_stop, end_stop, via_stop, max_transfers)
        print(f"Test forward_chaining ({start_stop}, {end_stop}, {via_stop}, {max_transfers}): ", 
              "Pass" if check_output(expected_output, actual_output) else f"Fail (\n\n\nExpected: {expected_output},\n\n Got: {actual_output})")

test_forward_chaining()