import numpy as np
import pandas as pd
import urbimg.functions.normalizer as n
from importlib.resources import files
from urbimg.functions import mapper as mp

dataset_dir = 'poi'

def get_data_input_filepath():
    return files('urbimg.input')


def read_raw(city_name):
    df_data = pd.read_csv(get_data_input_filepath().joinpath(dataset_dir).joinpath(f'improved_poi_{city_name}.csv'))
    df_data[['x', 'y']] = df_data['centroid'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)

    return df_data


def read_filtered_raw(city_name, column_name, column_value):
    # Read the raw data for the city
    df_data = read_raw(city_name)

    # Filter the DataFrame where the column matches the value
    return df_data[df_data[column_name] == column_value]


def read_categories(*city_names):
    categories = []

    # Loop through all the city names
    for cn in city_names:
        # Assuming read_raw() returns a DataFrame with a column 'category'
        data = read_raw(cn)

        # Ensure the 'category' column exists in the data
        if 'category' in data.columns:
            categories.extend(data["category"].values)
        else:
            print(f"Warning: 'category' column missing in data for city '{cn}'")

    # Return unique categories as a pandas Series
    return pd.Series(categories).unique()


def read_normalized(city_name=None, method='kde', verbose=False, df_data=None, size=10, auto_bandwidth=False,
                    actual_xy=True):
    # Ensure that at least one of city_name or df_data is provided
    if city_name is None and df_data is None:
        raise ValueError("Both 'city_name' and 'df_data' are None. You need to provide at least one of them!")

    # If df_data is not provided, load it using the city_name
    if df_data is None:
        df_data = read_raw(city_name)

    # Ensure that the method is valid, if required
    if method != 'kde':
        raise ValueError(f"Unsupported method: {method}. Currently, only 'kde' is supported.")

    # Assuming 'n.kde' is a valid function
    x, y, _ = n.kde(df_data, verbose=verbose, size=size, auto_size=auto_bandwidth, actual_xy=actual_xy)

    return x, y


def process_poi_data(city_name, size=10):
    """
    Process the POI data for a given city by reading the dataset and assigning grid cells.
    """
    # Read the raw POI data for the given city
    df_poi = read_raw(city_name)

    # Compute the KDE edges for the given POI data
    x_edge, y_edge = mp.compute_kde_edges(df_poi, x_column='x', y_column='y', size=size)

    # Assign the grid cell based on 'x' and 'y' coordinates
    df_poi['in_cell'] = mp.find_grid_cell(df_poi[['x', 'y']].values, x_edge, y_edge)

    return df_poi
