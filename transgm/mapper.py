from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule
import numpy as np


def find_grid_cell(df_points, x_edge, y_edge):
    """
    Finds the grid cell indices for a set of points based on the given x and y edges.

    Args:
    - df_points (np.ndarray): Array of points with shape (n_samples, 2), where each point has x and y coordinates.
    - x_edge (np.ndarray): Array representing the x-edge grid values.
    - y_edge (np.ndarray): Array representing the y-edge grid values.

    Returns:
    - list: List of tuples containing the (x, y) grid indices for each point.
    """
    # Find the grid indices using np.digitize and subtract 1 to get the correct index
    x_cell = np.digitize(df_points[:, 0], x_edge) - 1
    y_cell = np.digitize(df_points[:, 1], y_edge) - 1

    # Ensure that the indices are within valid bounds for the grid
    x_cell = np.clip(x_cell, 0, len(x_edge) - 2)  # Valid grid range for x
    y_cell = np.clip(y_cell, 0, len(y_edge) - 2)  # Valid grid range for y

    # Return the grid cell indices as a list of tuples
    return list(zip(x_cell, y_cell))


def compute_bandwidth(data):
    """
    Compute the bandwidth using Silverman's rule of thumb.

    Args:
    - data (np.ndarray): Input data array with a single dimension (n_samples, 1).

    Returns:
    - float: Computed bandwidth for the given data.
    """
    return silvermans_rule(data)


def scale_data(data, bw1, bw2):
    """
    Scale the data using the computed bandwidths.

    Args:
    - data (np.ndarray): Input data array to be scaled.
    - bw1 (float): Bandwidth for the first dimension (x).
    - bw2 (float): Bandwidth for the second dimension (y).

    Returns:
    - np.ndarray: Scaled data.
    """
    return data / np.array([bw1, bw2])


def fit_kde(data_scaled, size=10):
    """
    Fit the KDE model and evaluate it on a grid of the specified size.

    Args:
    - data_scaled (np.ndarray): Scaled data to be fitted.
    - size (int): The size of the grid to evaluate the KDE.

    Returns:
    - np.ndarray: Scaled x and y values from the KDE evaluation.
    """
    kde = FFTKDE(bw=1).fit(data_scaled)
    return kde.evaluate((size, size))  # Evaluates on a grid of (size, size)


def compute_bandwidth(data):
    """
    Compute the bandwidth using Silverman's rule of thumb.

    Args:
    - data (np.ndarray): Input data array with a single dimension (n_samples, 1).

    Returns:
    - float: Computed bandwidth for the given data.
    """
    return silvermans_rule(data)


def scale_data(data, bw1, bw2):
    """
    Scale the data using the computed bandwidths.

    Args:
    - data (np.ndarray): Input data array to be scaled.
    - bw1 (float): Bandwidth for the first dimension (x).
    - bw2 (float): Bandwidth for the second dimension (y).

    Returns:
    - np.ndarray: Scaled data.
    """
    return data / np.array([bw1, bw2])


def fit_kde(data_scaled, size=10):
    """
    Fit the KDE model and evaluate it on a grid of the specified size.

    Args:
    - data_scaled (np.ndarray): Scaled data to be fitted.
    - size (int): The size of the grid to evaluate the KDE.

    Returns:
    - np.ndarray: Scaled x and y values from the KDE evaluation.
    """
    kde = FFTKDE(bw=1).fit(data_scaled)
    return kde.evaluate((size, size))  # Evaluates on a grid of (size, size)


def compute_kde_edges(df_od, x_column, y_column, size=10):
    """
    Computes the KDE edges based on origin coordinates (or any pair of columns) in the given DataFrame.

    Args:
    - df_od (pd.DataFrame): DataFrame with columns corresponding to origin and destination coordinates.
    - x_column (str): The name of the column for the x-coordinate (e.g., 'from_x').
    - y_column (str): The name of the column for the y-coordinate (e.g., 'from_y').
    - size (int): The size of the grid to evaluate the KDE.

    Returns:
    - tuple: x_edge and y_edge arrays containing unique edges after scaling.
    """
    # Convert the selected columns to a NumPy array
    data = df_od[[x_column, y_column]].to_numpy()

    # Compute bandwidths using Silverman's rule
    bw1 = compute_bandwidth(data[:, [0]])  # Bandwidth for x-dimension
    bw2 = compute_bandwidth(data[:, [1]])  # Bandwidth for y-dimension

    # Scale the data using the computed bandwidths
    data_scaled = scale_data(data, bw1, bw2)

    # Fit the KDE and evaluate on a grid
    x_scaled, y_scaled = fit_kde(data_scaled, size)

    # Scale back the x and y values to the original domain
    x = x_scaled * np.array([bw1, bw2])
    y = y_scaled / (bw1 * bw2)

    # Extract unique edges from the scaled x and y values
    x_edge = np.unique(x[:, 0])
    y_edge = np.unique(x[:, 1])

    return x_edge, y_edge
