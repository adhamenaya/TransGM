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

def get_window_padded(grid_values, i, j, window_size=3):
    half_w = window_size // 2
    padded = np.pad(grid_values, half_w, mode='constant', constant_values=0)
    return padded[i:i + window_size, j:j + window_size]
