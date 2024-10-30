# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.
    
    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df

    for tmp, row in df_trips.iterrows():
        trip_to_route[row['trip_id']] = row['route_id']

    for tmp, row in df_stop_times.iterrows():
        route_id = trip_to_route.get(row['trip_id'])
        if route_id:
            if route_id not in route_to_stops:
                route_to_stops[route_id] = []
            stop_entry = (row['stop_sequence'], row['stop_id'])
            if stop_entry not in route_to_stops[route_id]:
                route_to_stops[route_id].append(stop_entry)
            stop_trip_count[row['stop_id']] += 1

    for route_id, stops in route_to_stops.items():
        if all(isinstance(stop, tuple) and len(stop) == 2 for stop in stops):
            route_to_stops[route_id] = [stop_id for _, stop_id in stops]
        else:
            print(f"Unexpected structure in stops for route_id {route_id}: {stops}")

    # Create fare_rules as a dictionary with route_id as key
    fare_rules = df_fare_rules.set_index('route_id').T.to_dict()

    merged_fare_df = pd.merge(df_fare_rules, df_fare_attributes, on='fare_id', how='inner')