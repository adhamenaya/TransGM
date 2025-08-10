import numpy as np

def get_window(grid_values, i, j, window_size=3):
    """
    (i, j) are x,y/ col,row indices. j starts from the bottom (j=0 is the last row).
    Extract a 3x3 window (or smaller if near edges) around position (i, j).
    """
    rows, cols = grid_values.shape

    # Adjust row index 'j' to be 0-based from the top
    row_index_top_origin = j

    # Compute window boundaries for rows (top-origin indexing)
    row_start = max(row_index_top_origin - 1, 0)
    row_end = min(row_index_top_origin + 1, rows - 1)

    # Compute window boundaries for columns
    col_start = max(i - 1, 0)
    col_end = min(i + 1, cols - 1)

    # Extract neighborhood
    neighborhood = grid_values[row_start:row_end + 1, col_start:col_end + 1]
    return neighborhood

def lisa_window(grid_values, i, j, weights=None, standardize=True):
    neighborhood = get_window(grid_values, i, j)
    x = neighborhood.flatten()
    n = len(x)
    mean_x = np.mean(grid_values)  # global mean
    std_x = np.std(grid_values)  # global std

    if std_x == 0:
        return 0.0  # handle division by zero

    z = np.abs(x - mean_x) / std_x  # standardize

    # mean_x = np.mean(grid_values)  # global mean
    # std_x = np.std(grid_values)  # global std
    #
    # if std_x == 0:
    #     return 0.0  # handle division by zero
    #
    # z = np.abs(x - mean_x) / std_x  # standardize

    lm = 0
    for m in range(n):
        for k in range(n):
            if m == k:
                continue
            row_m, col_m = m // neighborhood.shape[1], m % neighborhood.shape[1]
            row_k, col_k = k // neighborhood.shape[1], k % neighborhood.shape[1]
            distance = np.sqrt((row_m - row_k) ** 2 + (col_m - col_k) ** 2)
            lm += (z[m] * z[k]) / distance

    return lm / n  # Normalize by number of cells

def get_window(grid_values, i, j, window_size=3):
    rows, cols = grid_values.shape

    # Calculate window radius from the desired dimensions
    # For a window_dimensions of 3, radius is 1
    # For a window_dimensions of 5, radius is 2, etc.
    window_radius = window_size // 2

    # Compute window boundaries for rows (top-origin indexing)
    row_start = max(j - window_radius, 0)
    row_end = min(j + window_radius, rows - 1)

    # Compute window boundaries for columns
    col_start = max(i - window_radius, 0)
    col_end = min(i + window_radius, cols - 1)

    # Extract neighborhood
    neighborhood = grid_values[row_start:row_end + 1, col_start:col_end + 1]
    return neighborhood
# =======
#
#             # lm += (z[m] * z[k]) / distance #emphasize the deviation from the global mean
#             # Inverse absolute difference similarity
#             similarity = (1.0 / (1.0 + abs(x[m] - x[k]))) / distance
#             lm += similarity
#
#     return lm / n  # Normalize by number of cells
# def get_window(grid_values, i, j, window_size=3):
#     rows, cols = grid_values.shape
#
#     window_radius = window_size // 2
#
#     # Compute window boundaries (i: row, j: col)
#     row_start = max(0, i - window_radius)
#     row_end = min(rows, i + window_radius + 1)
#
#     col_start = max(0, j - window_radius)
#     col_end = min(cols, j + window_radius + 1)
#
#     # Extract neighborhood
#     neighborhood = grid_values[row_start:row_end, col_start:col_end]
#     return neighborhood

def get_weight(window, window_size=3):
    weights = np.ones_like(window, dtype=float)
    center_row = window.shape[0] // 2
    center_col = window.shape[1] // 2
    if weights.shape[0] > 0 and weights.shape[1] > 0:
        weights[center_row, center_col] = 0  # Exclude center for rook/queen contiguity
        if weights.sum() > 0:
            weights /= weights.sum()  # Normalize to sum to 1
    return weights
def lisa_window4(grid, window_size):
    mean = np.mean(grid)
    std = np.std(grid)
    z = (grid - mean) / std

    local_i = np.zeros_like(grid, dtype=float)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighbors = get_window(z, i, j, window_size)
            weights = get_weight(neighbors, window_size)
            local_i[i, j] = z[i, j] * np.sum(weights * neighbors)
    return local_i



def get_window_padded(grid_values, i, j, window_size=3):
    half_w = window_size // 2
    padded = np.pad(grid_values, half_w, mode='constant', constant_values=0)
    return padded[i:i + window_size, j:j + window_size]

def lisa_window1(grid_values, i, j, weights, standardize=True):
    rows, cols = grid_values.shape
    neighborhood = get_window(grid_values, i, j)
    x = neighborhood.flatten()
    n = len(x)
    mean_x = np.mean(grid_values)  # global mean
    std_x = np.std(grid_values)  # global std

    if std_x == 0:
        return 0.0  # handle division by zero

    if standardize:
        z = (x - mean_x) / std_x  # standardize
    else:
        z = x

    # Get window dimensions
    window_rows, window_cols = neighborhood.shape

    # Calculate horizontal and vertical components separately
    lm_horizontal = 0
    lm_vertical = 0

    for m in range(n):
        row_m, col_m = m // window_cols, m % window_cols
        for k in range(n):
            if m == k:
                continue
            row_k, col_k = k // window_cols, k % window_cols

            # For horizontal aggregation (same row)
            if row_m == row_k:
                lm_horizontal += z[m] * z[k] * weights[row_m, col_m]

            # For vertical aggregation (same column)
            if col_m == col_k:
                lm_vertical += z[m] * z[k] * weights[row_m, col_m]

    # Normalize by sum of weights
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return 0  # handle division by zero

    # Return combined horizontal and vertical components
    return (lm_horizontal + lm_vertical) / (2 * weight_sum)  # Average of the two components

def lisa_window0(grid_values, i, j, weights, standardize=True):
    rows, cols = grid_values.shape
    neighborhood = get_window_padded(grid_values, i, j)
    w = weights
    x = grid_values

    # Global mean and std
    mean_x = np.mean(grid_values)  # global mean
    std_x = np.std(grid_values)  # global std

    if std_x == 0:
        return 0.0

    z_grid = (x - mean_x) / std_x  # standardize
    z_neighborhood = (neighborhood - mean_x) / std_x  # standardize neighborhood
    z_i = z_grid[i, j]  # standardized value at (i, j)

    I_i = z_i * np.sum(w * z_neighborhood)  # LISA value at (i, j)

    return I_i


def lisa00(grid_values, window=3, standardize=True):
    lisa_g = lisa_window(grid_values, window)

    return np.round(lisa_g, 3)

def lisa(grid_values, window=3, standardize=True):
    weights = np.ones((window, window))
    center = window // 2
    weights[center, center] = 0 # rook/queen weights

    rows, cols = grid_values.shape
    lisa_g = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            lisa_g[i, j] = lisa_window(grid_values, i, j, weights, standardize)

    return np.round(lisa_g, 3)

def lisa0(grid_values, p_threshold=0.05, window=3):
    # Calculate LISA values
    lisa_values = lisa0(grid_values, window)

    # Calculate Z-scores for each cell
    mean_x = np.mean(grid_values)
    std_x = np.std(grid_values)
    if std_x == 0:
        return np.zeros_like(grid_values)  # Return all zeros if no variation

    z_scores = (grid_values - mean_x) / std_x

    # Calculate spatial lag (weighted average of neighboring values)
    rows, cols = grid_values.shape
    weights = np.ones((window, window))
    weights[window//2, window//2] = 0  # Exclude self from neighbors

    # Normalize weights
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum

    # Calculate spatial lag for each cell
    spatial_lag = np.zeros_like(grid_values)
    padded = np.pad(z_scores, window//2, mode='constant')

    for i in range(rows):
        for j in range(cols):
            neighborhood = padded[i:i+window, j:j+window]
            # Apply weights to get spatial lag
            spatial_lag[i, j] = np.sum(neighborhood * weights)

    # Create the mask
    mask = np.zeros_like(grid_values)

    # HH clusters: high value surrounded by high values (positive Z-score, positive lag)
    # LL clusters: low value surrounded by low values (negative Z-score, negative lag)
    for i in range(rows):
        for j in range(cols):
            # Check if HH or LL and if statistically significant
            if (z_scores[i, j] > 0 and spatial_lag[i, j] > 0) or (z_scores[i, j] < 0 and spatial_lag[i, j] < 0):
                if abs(lisa_values[i, j]) > p_threshold:  # Using LISA value as a proxy for significance
                    mask[i, j] = 1

    return mask