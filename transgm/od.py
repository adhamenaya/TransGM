import numpy as np
from importlib.resources import files
import urbimg.functions.mapper as mp
import urbimg.utils.geo_utils as gutils
import pandas as pd
from itertools import product

dataset_dir = 'od'

x_kde_edge = None
y_kde_edge = None


def set_kde_edges(x_column, y_column):
    x_kde_edge = x_column
    y_kde_edge = y_column


def get_data_input_filepath():
    return files('urbimg.input')


def read_raw(city_name):
    df_data = pd.read_csv(get_data_input_filepath().joinpath(dataset_dir).joinpath(f'{city_name}_od.csv'))
    df_data[['from_x', 'from_y']] = df_data['geometry_x'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    df_data[['to_x', 'to_y']] = df_data['geometry_y'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)

    return df_data


def process_od_data(df, x_column, y_column, size=10):
    # Compute the KDE edges based on the destination coordinates
    if x_kde_edge is not None or y_kde_edge is not None:
        x_edge = x_kde_edge
        y_edge = y_kde_edge
    else:
        x_edge, y_edge = mp.compute_kde_edges(df, x_column=x_column, y_column=y_column, size=size)

    # Assign grid cells for the origin and destination coordinates
    df['from_cell'] = mp.find_grid_cell(df[['from_x', 'from_y']].values, x_edge, y_edge)
    df['to_cell'] = mp.find_grid_cell(df[['to_x', 'to_y']].values, x_edge, y_edge)

    # Calculate the distance row by row
    df['distance'] = df.apply(
        lambda row: gutils.calculate_haversine_distance(row['from_x'], row['from_y'], row['to_x'], row['to_y']), axis=1
    )

    return df


def count_trips(df, from_cell_col='from_cell', to_cell_col='to_cell'):
    # Group by 'from_cell' and 'to_cell', then count occurrences
    trips = df.groupby([from_cell_col, to_cell_col]).size().reset_index(name='trips')

    # Set the index to ('from_cell', 'to_cell')
    trips = trips.set_index([from_cell_col, to_cell_col])

    return trips


def count_origins(df, from_cell_col='from_cell'):
    # Group by 'from_cell' and count occurrences
    origins = df.groupby(from_cell_col).size().reset_index(name='origins')

    # Set the index to 'from_cell'
    origins = origins.set_index(from_cell_col)

    return origins


def calculate_mean_distance(df, from_cell_col='from_cell', to_cell_col='to_cell', distance_col='distance'):
    # Group by 'from_cell' and 'to_cell', calculate mean distance
    mean_distance = (
        df.groupby([from_cell_col, to_cell_col])
        .agg({distance_col: 'mean'})
        .round(2)  # Round the numeric columns like 'distance'
        .reset_index()
        .set_index([from_cell_col, to_cell_col])
    )

    return mean_distance


def calculate_probability(df_od):
    # Step 1: Create a list of unique coordinates (both from_cell and to_cell)
    unique_from_cells = set(df_od['from_cell'])
    unique_to_cells = set(df_od['to_cell'])

    # Combine both from_cell and to_cell into one set of unique coordinate pairs
    all_unique_cells = unique_from_cells.union(unique_to_cells)

    # Step 2: Create a mapping of coordinates to indices (starting from 1)
    cell_to_index = {cell: idx + 1 for idx, cell in enumerate(sorted(all_unique_cells))}

    # Step 3: Replace the coordinates with their corresponding indices in the original dataframe
    df_od['from_index'] = df_od['from_cell'].map(cell_to_index)
    df_od['to_index'] = df_od['to_cell'].map(cell_to_index)
    df_od[['from_index', 'to_index']]

    # Step 1: Count the occurrences of each unique (from_index, to_index) pair
    df_grouped = df_od.groupby(['from_index', 'to_index']).size().reset_index(name='count')

    # Display the resulting DataFrame
    df_grouped['probability'] = df_grouped['count'] / df_grouped['count'].sum()

    return df_od['from_index'], df_od['to_index'], df_grouped['probability']


def process_flow_data(city_name, od_module, poi_module, size=10):
    # Read and process OD data
    df_od = od_module.read_raw(city_name)
    df_od = od_module.process_od_data(df_od, x_column='to_x', y_column='to_y', size=size)

    # Analyze OD data
    trips = od_module.count_trips(df_od)
    origins = od_module.count_origins(df_od)
    mean_distance = od_module.calculate_mean_distance(df_od)

    # Read and process POI data
    df_poi = poi_module.process_poi_data(city_name, size=size)
    pois_grouped = df_poi[['in_cell', 'category', 'x']].groupby(['in_cell', 'category']).count()

    # Compile results
    return {
        'od_data': df_od,
        'trips': trips,
        'origins': origins,
        'mean_distance': mean_distance,
        'poi_data': df_poi,
        'pois_grouped': pois_grouped
    }


def generate_flow_dataset0(city_name, od_module, poi_module, size, empty_value=1):
    # Initialize cells and base dataset
    cells = list(product(range(size), repeat=2))
    data = [(from_cell, to_cell) for from_cell in cells for to_cell in cells]
    columns = ['from_cell', 'to_cell']

    city_dataset = pd.DataFrame(data, columns=columns)

    # Add initial columns
    city_dataset['trips'] = empty_value
    city_dataset['origins'] = empty_value

    results = process_flow_data(city_name, od_module, poi_module, size)

    # Add destination columns based on POI categories
    for category in results['pois_grouped'].index.get_level_values('category').unique():
        city_dataset[category] = empty_value

    # Add distance column
    city_dataset['distance'] = empty_value

    city_dataset.set_index(['from_cell', 'to_cell'], inplace=True)

    # Function to update specific columns in the dataset
    def update_column(base_df, update_df, column_name):
        if not isinstance(base_df.index, pd.MultiIndex) or not isinstance(update_df.index, pd.MultiIndex):
            raise ValueError(f"Both DataFrames must have a MultiIndex to update '{column_name}'.")

        if not base_df.index.isin(update_df.index).any():
            print(f"No overlapping indices found for '{column_name}'.")
            return

        base_df.loc[update_df.index, column_name] = update_df[column_name]

    # Update columns from results
    update_column(city_dataset, results['mean_distance'], 'distance')
    update_column(city_dataset, results['trips'], 'trips')

    # Extract 'origins' column without index and ensure it's a float
    origins_map = dict(
        zip(results['origins'].index.get_level_values('from_cell'), results['origins'].values.astype(float)))

    # Update 'origins' column in city_dataset
    city_dataset['origins'] = city_dataset.index.get_level_values('from_cell').map(origins_map).astype(np.float32)

    # Update destination columns with POI data
    for category in results['poi_data']['category'].unique():
        filtered_poi = results['pois_grouped'][results['pois_grouped'].index.get_level_values(1) == category]
        city_dataset[category] = city_dataset.index.get_level_values('to_cell').map(
            dict(zip(filtered_poi.index.get_level_values('in_cell'), filtered_poi['x']))
        )

    # Fill NaN values with 0
    city_dataset.fillna(empty_value, inplace=True)

    return city_dataset
def generate_flow_dataset(city_name, od_module, poi_module, size, empty_value=1):
    # Initialize cells and base dataset
    cells = list(product(range(size), repeat=2))
    data = [(from_cell, to_cell) for from_cell in cells for to_cell in cells]
    columns = ['from_cell', 'to_cell']

    city_dataset = pd.DataFrame(data, columns=columns)

    # Add initial columns
    city_dataset['trips'] = empty_value
    city_dataset['origins'] = empty_value

    results = process_flow_data(city_name, od_module, poi_module, size)

    # Add destination columns based on POI categories
    for category in results['pois_grouped'].index.get_level_values('category').unique():
        city_dataset[category] = empty_value

    # Add distance column
    city_dataset['distance'] = empty_value

    city_dataset.set_index(['from_cell', 'to_cell'], inplace=True)

    # Function to update specific columns in the dataset
    def update_column(base_df, update_df, column_name):
        if not isinstance(base_df.index, pd.MultiIndex) or not isinstance(update_df.index, pd.MultiIndex):
            raise ValueError(f"Both DataFrames must have a MultiIndex to update '{column_name}'.")

        if not base_df.index.isin(update_df.index).any():
            print(f"No overlapping indices found for '{column_name}'.")
            return

        base_df.loc[update_df.index, column_name] = update_df[column_name]

    # Update columns from results
    update_column(city_dataset, results['mean_distance'], 'distance')
    update_column(city_dataset, results['trips'], 'trips')

    # Extract 'origins' column without index and ensure it's a float
    origins_map = dict(
        zip(results['origins'].index.get_level_values('from_cell'), results['origins'].values.astype(float)))

    # Update 'origins' column in city_dataset
    city_dataset['origins'] = city_dataset.index.get_level_values('from_cell').map(origins_map).astype(np.float32)

    # Update destination columns with POI data
    for category in results['poi_data']['category'].unique():
        filtered_poi = results['pois_grouped'][results['pois_grouped'].index.get_level_values(1) == category]
        city_dataset[category] = city_dataset.index.get_level_values('to_cell').map(
            dict(zip(filtered_poi.index.get_level_values('in_cell'), filtered_poi['x']))
        )

    # Fill NaN values with 0
    city_dataset.fillna(empty_value, inplace=True)

    return city_dataset
