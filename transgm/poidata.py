import sys

import numpy as np
import pandas as pd

from TranSim.data.spatialgrid import SpatialGrid
from models import utils

sys.path.append('./models')


class POIData:
    """
    A class for handling Origin-Destination (OD) datasets, including loading, preprocessing, and validation.

    Attributes:
      - data: Pandas DataFrame containing OD flow data.
    """

    def __init__(self, spacing_km=1, file_path=None, shp_path=None):
        """Initialize ODData with an optional dataset file path."""
        self.data = None
        if file_path:
            self.load_data(file_path)
        self.shp_path = shp_path
        self.features_matrix = None
        self.spacing_km = spacing_km

    def load_data(self, file_path):
        """Load OD dataset from a CSV or similar file."""
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        """Apply necessary preprocessing to the OD dataset."""
        self.data[['x', 'y']] = self.data['centroid'].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)

    def gen_features_grid(self, resample=False, resample_size=None, features_list = None):
        """Compute the spatial grid for origin points."""
        if features_list is None:
            raise Exception("You should provide the list of features!")
        dest_tensor = np.zeros((resample_size[0], resample_size[1], len(features_list)))
        i = 0
        for c in features_list:
            temp = self.data[self.data["category"] == c]  # Fixed comparison syntax
            temp = temp[['x', 'y']].to_numpy()

            feature_grid = SpatialGrid(self.shp_path, self.spacing_km)
            lon_grid, lat_grid, grid_gdf = feature_grid.gen_grid("", grid_spacing_km=self.spacing_km, center=None,
                                                                 bottom_left_buffer_km=2)

            _, _, _, norm_grid, ctrs, ctr_wts = feature_grid.assign_grid(lon_grid, lat_grid, x_pts=temp[:, 0], y_pts=temp[:, 1])
            data = np.hstack([ctrs, np.array(ctr_wts).reshape(-1, 1)])
            print("Data shape:", data.shape)
            print("dest_tensor shape:", dest_tensor.shape)


            # Resize the grid
            x,y, data = feature_grid.resample_kde(data, resample_size, True)

            dest_tensor[:, :, i] = data
            i += 1
        return dest_tensor, x, y

    def validate(self):
        """Validate the consistency and quality of the OD dataset."""
        pass
