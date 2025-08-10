import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from geopy.distance import geodesic
from TranSim.data.spatialgrid import SpatialGrid
from collections import defaultdict


class ODData:
    def __init__(self, spacing_km=1, file_path=None, shp_path=None):
        """Initialize ODData with an optional dataset file path."""
        self.data = None
        self.spacing_km = spacing_km
        self.shp_path = shp_path
        if file_path:
            self.load_data(file_path)

        self.origin_grid = SpatialGrid(shp_path, spacing_km)
        self.destination_grid = SpatialGrid(shp_path, spacing_km)

    def load_data(self, file_path):
        """Load OD dataset from a CSV or similar file."""
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        """Apply necessary preprocessing to the OD dataset."""
        self.data[['from_x', 'from_y']] = self.data['geometry_x'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(
            float)
        self.data[['to_x', 'to_y']] = self.data['geometry_y'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(
            float)
        # Calculate Euclidean distance correctly
        self.data['distance'] = self.data.apply(
            lambda row: euclidean((row['from_x'], row['from_y']), (row['to_x'], row['to_y'])), axis=1
        )

    def gen_origin_grid(self, x_scale, y_scale, size):
        """Compute the spatial grid for origin points."""
        odata = self.data[['from_x', 'from_y']].to_numpy()
        # Separate latitude and longitude
        latitudes = sorted(set(x_scale))
        longitudes = sorted(set(y_scale))

        # Determine the range of latitudes and longitudes
        lat_min, lat_max = min(latitudes), max(latitudes)
        lon_min, lon_max = min(longitudes), max(longitudes)

        # Define the number of desired grid cells
        num_lat_cells = size[0]
        num_lon_cells = size[1]  # Corrected to size[1] for longitude grid cells

        # Calculate the step size based on the desired number of grid cells
        lat_step = (lat_max - lat_min) / num_lat_cells
        lon_step = (lon_max - lon_min) / num_lon_cells

        # Generate grid edges
        lat_edges = np.arange(lat_min, lat_max + lat_step, lat_step)
        lon_edges = np.arange(lon_min, lon_max + lon_step, lon_step)

        # Initialize a 2D grid to count points
        grid_counts = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))

        # Count points in the grid
        for lat, lon in odata:
            lat_idx = np.digitize(lat, lat_edges) - 1  # Corrected index to start from 0
            lon_idx = np.digitize(lon, lon_edges) - 1  # Corrected index to start from 0
            if 0 <= lat_idx < grid_counts.shape[0] and 0 <= lon_idx < grid_counts.shape[1]:
                grid_counts[lat_idx, lon_idx] += 1

        # Print the grid count
        return grid_counts

    def gen_destination_grid(self, x_scale, y_scale, size):
        """Compute the spatial grid for origin points."""
        odata = self.data[['to_x', 'to_y']].to_numpy()
        # Separate latitude and longitude
        latitudes = sorted(set(x_scale))
        longitudes = sorted(set(y_scale))

        # Determine the range of latitudes and longitudes
        lat_min, lat_max = min(latitudes), max(latitudes)
        lon_min, lon_max = min(longitudes), max(longitudes)

        # Define the number of desired grid cells
        num_lat_cells = size[0]
        num_lon_cells = size[1]  # Corrected to size[1] for longitude grid cells

        # Calculate the step size based on the desired number of grid cells
        lat_step = (lat_max - lat_min) / num_lat_cells
        lon_step = (lon_max - lon_min) / num_lon_cells

        # Generate grid edges
        lat_edges = np.arange(lat_min, lat_max + lat_step, lat_step)
        lon_edges = np.arange(lon_min, lon_max + lon_step, lon_step)

        # Initialize a 2D grid to count points
        grid_counts = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1))

        # Count points in the grid
        for lat, lon in odata:
            lat_idx = np.digitize(lat, lat_edges) - 1  # Corrected index to start from 0
            lon_idx = np.digitize(lon, lon_edges) - 1  # Corrected index to start from 0
            if 0 <= lat_idx < grid_counts.shape[0] and 0 <= lon_idx < grid_counts.shape[1]:
                grid_counts[lat_idx, lon_idx] += 1

        # Print the grid count
        return grid_counts

    def gen_od(self, grid):
        def find_grid_cell(p, x, y):
            """
            Finds the closest grid cell index for given points.

            Args:
                p (np.array): Nx2 array of (longitude, latitude) points.
                x (np.array): Grid x-coordinates (longitude).
                y (np.array): Grid y-coordinates (latitude).

            Returns:
                list: List of (lat_index, lon_index) tuples.
            """
            y = np.sort(y)
            x = np.sort(x)
            x_cell = np.digitize(p[:, 0], x)  # Longitude index
            y_cell = np.digitize(p[:, 1], y)  # Latitude index

            # Ensure indices are within valid bounds
            x_cell = np.clip(x_cell, 0, len(x))
            y_cell = np.clip(y_cell, 0, len(y))

            return list(zip(y_cell, x_cell))  # (lat_index, lon_index)

        od_mat = np.ones((grid.shap[0] ** 2,))

    def gen_dist(self, resample, resample_size):
        # Generate origin and destination grids
        xo, yo, _ = self.gen_origin_grid(resample=resample, resample_size=resample_size)
        xd, yd, _ = self.gen_destination_grid(resample=resample, resample_size=resample_size)

        # Calculate pairwise distances between each origin and destination pair
        dist_dict = {}
        for i in range(len(xo)):
            for j in range(len(yo)):
                for k in range(len(xd)):
                    for l in range(len(yd)):
                        # Calculate geodesic distance between origin (xo[i], yo[j]) and destination (xd[k], yd[l])
                        distance = geodesic((yo[j], xo[i]), (yd[l], xd[k])).kilometers
                        dist_dict[((i, j), (k, l))] = distance
                        print(f"dist {((i, j), (k, l))}")
        print("finished gen_dist")
        return dist_dict

    def validate(self):
        """Validate the consistency and quality of the OD dataset."""
        pass
