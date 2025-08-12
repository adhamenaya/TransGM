import geopandas as gpd
import numpy as np
from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule
from shapely import Point, Polygon


class SpatialGrid:

    def __init__(self, shp_path, spacing_km=2, optimal_bws=None):
        self.spacing_km = spacing_km
        self.shp_path = shp_path
        self.centers = None  # added centers attribute
        self.optimal_bws = optimal_bws  # added optimal bandwidths attribute


    def optimal_kde(self, data, new_size=None):
        # Compute bandwidth using Silverman's rule (keeping original code)
        init_bw1 = silvermans_rule(data[:, [0]])  # Bandwidth for x
        init_bw2 = silvermans_rule(data[:, [1]])  # Bandwidth for y

        # Define a range of bandwidths to test
        bandwidth_range_x = np.linspace(0.1 * init_bw1, 2 * init_bw1, 20)
        bandwidth_range_y = np.linspace(0.1 * init_bw2, 2 * init_bw2, 20)
        cv_scores = {}
        for bw1 in bandwidth_range_x:
            for bw2 in bandwidth_range_y:
                log_likelihoods = []
                n = data.shape[0]

                for i in range(data.shape[0]):
                    # Scale data for KDE computation
                    data_scaled = data[:, :2] / np.array([bw1, bw2])
                    wts = data[:, 2]  # Extract weights

                    weights = np.ones(n)
                    weights[i] = 0  # Leave-one-out by zeroing this weight
                    weights /= weights.sum()  # Renormalize weights

                    # Fit Kernel Density Estimator using FFT-based KDE
                    kde = FFTKDE(bw=1).fit(data_scaled, weights=wts)

                    # Evaluate KDE on a (size x size) grid
                    xy_scl, z_scl = kde.evaluate((new_size[0], new_size[1]))

                    # Rescale coordinates back to original space
                    xy = xy_scl * np.array([bw1, bw2])
                    z = z_scl / (bw1 * bw2)  # Adjust for scaling
                    log_likelihoods.append(np.log(z))
                cv_scores[(bw1, bw2)] = np.mean(log_likelihoods)
        best_bandwidth = max(cv_scores, key=cv_scores.get)
        print(f"Best bandwidth: {best_bandwidth}")

        (optimal_bw1, optimal_bw2) = best_bandwidth
        # Transform data using the best bandwidth
        data_scaled = data[:, :2] / np.array([optimal_bw1, optimal_bw2])
        wts = data[:, 2]  # Extract weights
        kde = FFTKDE(bw=1).fit(data_scaled, weights=wts)

        # Evaluate KDE on a (size x size) grid
        xy_scl, z_scl = kde.evaluate((new_size[0], new_size[1]))

        # Rescale coordinates back to original space
        xy = xy_scl * np.array([optimal_bw1, optimal_bw2])
        z = z_scl / (optimal_bw1 * optimal_bw2)  # Adjust for scaling

        # Create the meshgrid
        x_unique = np.unique(xy[:, 1])
        y_unique = np.unique(xy[:, 0])
        X, Y = np.meshgrid(x_unique, y_unique)

        # Reshape z to match the grid dimensions
        z_reshaped = z.reshape(len(y_unique), len(x_unique)).T
        # Extract sorted unique x and y values
        return X, Y, z_reshaped, new_size

    def resample_kde(self, data, target_dims=None, normalize=False):
        """Resamples KDE output to target dimensions."""
        x,y,z, _ = self.optimal_kde(data, new_size=target_dims)
        return x,y,z

    def gen_grid(self, city_name, grid_spacing_km=3, center=None, bbox_buffer_km=1, bottom_left_buffer_km=2):
        """
        Generates a grid of specified cell size over a shapefile, centered on a given point,
        extending the grid to an enlarged shapefile's bounding box and creates grid cells, ensuring full coverage.
        """
        try:
            gdf = gpd.read_file(self.shp_path)
        except Exception as e:
            print(f"Error reading shapefile: {e}")
            return None, None, None

        gdf = gdf.to_crs(epsg=4326)

        min_lon, min_lat, max_lon, max_lat = gdf.total_bounds

        # Enlarge the bounding box by a buffer
        lat_buffer = bbox_buffer_km / 111.0
        lon_buffer = bbox_buffer_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))

        min_lon -= lon_buffer
        max_lon += lon_buffer
        min_lat -= lat_buffer
        max_lat += lat_buffer

        # Add extra buffer to the bottom and left
        bottom_left_lat_buffer = bottom_left_buffer_km / 111.0
        bottom_left_lon_buffer = bottom_left_buffer_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))

        min_lon -= bottom_left_lon_buffer
        min_lat -= bottom_left_lat_buffer

        lat_step = grid_spacing_km / 111.0
        lon_step = grid_spacing_km / (111.0 * np.cos(np.radians((min_lat + max_lat) / 2)))

        if center:
            center_point = Point(center)

            # Calculate the central cell's boundaries
            central_lon_min = center_point.x - lon_step / 2
            central_lon_max = center_point.x + lon_step / 2
            central_lat_min = center_point.y - lat_step / 2
            central_lat_max = center_point.y + lat_step / 2

            # Extend grid to the left and right, up and down, ensuring complete coverage
            lon_grid_left = np.arange(central_lon_min - lon_step, min_lon - lon_step, -lon_step)[::-1]
            lon_grid_right = np.arange(central_lon_max + lon_step, max_lon + lon_step, lon_step)

            # Extend until the grid covers the bounding box
            if len(lon_grid_left) == 0 or lon_grid_left[-1] > min_lon:
                lon_grid_left = np.arange(central_lon_min - lon_step, min_lon - 0.0001, -lon_step)[::-1]
            if len(lon_grid_right) == 0 or lon_grid_right[-1] < max_lon:
                lon_grid_right = np.arange(central_lon_max + lon_step, max_lon + 0.0001, lon_step)

            lon_grid = np.concatenate((lon_grid_left, [central_lon_min, central_lon_max], lon_grid_right))

            lat_grid_down = np.arange(central_lat_min - lat_step, min_lat - lat_step, -lat_step)[::-1]
            lat_grid_up = np.arange(central_lat_max + lat_step, max_lat + lat_step, lat_step)

            # Extend until the grid covers the bounding box
            if len(lat_grid_down) == 0 or lat_grid_down[-1] > min_lat:
                lat_grid_down = np.arange(central_lat_min - lat_step, min_lat - 0.001, -lat_step)[::-1]
            if len(lat_grid_up) == 0 or lat_grid_up[-1] < max_lat:
                lat_grid_up = np.arange(central_lat_max + lat_step, max_lat + 0.001, lat_step)

            lat_grid = np.concatenate((lat_grid_down, [central_lat_min, central_lat_max], lat_grid_up))

            # Check and extend if necessary to cover the bottom and left
            if lon_grid[0] > min_lon:
                lon_grid = np.concatenate(
                    (np.arange(lon_grid[0] - lon_step, min_lon - 0.001, -lon_step)[::-1], lon_grid))
            if lat_grid[0] > min_lat:
                lat_grid = np.concatenate(
                    (np.arange(lat_grid[0] - lat_step, min_lat - 0.001, -lat_step)[::-1], lat_grid))

        else:
            # Default behavior: create grid based on the bounding box
            lon_grid = np.arange(min_lon, max_lon + lon_step, lon_step)
            lat_grid = np.arange(min_lat, max_lat + lat_step, lat_step)

        # Create grid cells
        grid_cells = []
        for i in range(len(lon_grid) - 1):
            for j in range(len(lat_grid) - 1):
                cell_polygon = Polygon([
                    (lon_grid[i], lat_grid[j]),
                    (lon_grid[i + 1], lat_grid[j]),
                    (lon_grid[i + 1], lat_grid[j + 1]),
                    (lon_grid[i], lat_grid[j + 1]),
                    (lon_grid[i], lat_grid[j])
                ])
                grid_cells.append(cell_polygon)

        grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

        return lon_grid, lat_grid, grid_gdf

    def assign_grid(self, lon_grid, lat_grid, x_pts, y_pts):
        """Assigns points to grid cells based on grid generated by generate_grid."""

        # Initialize grid
        grid = np.zeros((len(lat_grid) - 1, len(lon_grid) - 1))
        ctrs, ctr_wts = [], []

        # Compute centroids
        for i, lat in enumerate(lat_grid[:-1]):
            for j, lon in enumerate(lon_grid[:-1]):
                ctr_lon = lon + (lon_grid[1] - lon_grid[0]) / 2
                ctr_lat = lat + (lat_grid[1] - lat_grid[0]) / 2
                ctrs.append((ctr_lon, ctr_lat))

        # Assign points to grid cells
        for x, y in zip(x_pts, y_pts):
            x_idx = np.digitize(x, lon_grid) - 1
            y_idx = np.digitize(y, lat_grid) - 1
            if 0 <= x_idx < len(lon_grid) - 1 and 0 <= y_idx < len(lat_grid) - 1:
                grid[y_idx, x_idx] += 1

        # Normalize grid counts
        total = np.sum(grid)
        norm_grid = grid / total if total > 0 else np.zeros_like(grid)

        # Store centroid weights
        for i in range(norm_grid.shape[0]):
            for j in range(norm_grid.shape[1]):
                ctr_wts.append(norm_grid[i, j])

        self.centers = np.array(ctrs)  # Store centers
        return len(lon_grid) - 1, len(lat_grid) - 1, grid, norm_grid, ctrs, ctr_wts


    def get_window(self, grid_values, i, j, window_size=3):
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

    def get_weight(self, window, window_size=3):
        weights = np.ones_like(window, dtype=float)
        center_row = window.shape[0] // 2
        center_col = window.shape[1] // 2
        if weights.shape[0] > 0 and weights.shape[1] > 0:
            weights[center_row, center_col] = 0  # Exclude center for rook/queen contiguity
            if weights.sum() > 0:
                weights /= weights.sum()  # Normalize to sum to 1
        return weights

    def get_window_padded(self, grid_values, i, j, window_size=3):
        half_w = window_size // 2
        padded = np.pad(grid_values, half_w, mode='constant', constant_values=0)
        return padded[i:i + window_size, j:j + window_size]