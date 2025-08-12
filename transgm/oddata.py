import sys
import numpy as np
import pandas as pd
from collections import Counter
from spatialgrid import SpatialGrid
from poidata import POIData


class ODData:
    """
    A class for handling Origin-Destination (OD) datasets, including loading,
    preprocessing, and validation.
    """

    def __init__(self, spacing_km=1, size=(8, 8), file_path=None, shp_path=None):
        """Initialize ODData with an optional dataset file path."""
        self.data = None
        self.city_data = None
        self.from_data = None
        self.to_data = None
        self.shp_path = shp_path
        self.features_matrix = None
        self.spacing_km = spacing_km
        self.size = size
        self.categories = set()  # Use a set to collect unique categories
        self.destinations = None  # destination POIs

        if file_path:
            self.load_data(file_path)

    def load_data(self, city):
        """Load OD dataset and destination POIs from CSV files."""
        # 1. Load OD dataset from a CSV file
        self.shp_path = f'input/geo/{city}/WD_DEC_2022_UK_BGC.shp'
        self.city_data = pd.read_csv(f'input/od/{city}_od.csv')

        # 2. Load destination POIs dataset
        for cn in ['coventry', 'birmingham']:
            data = pd.read_csv(f'input/poi/improved_poi_{cn}.csv')
            if 'category' in data.columns:
                filtered_categories = data.loc[~data['category'].isin(
                    ['trips', 'distance', 'origins', 'other', 'residential', 'public']), 'category'].dropna().values
                self.categories.update(filtered_categories)

        self.categories = sorted(list(self.categories))

        poi = POIData(file_path=f'./input/poi/improved_poi_{city}.csv',
                      shp_path=f'./input/geo/{city}/WD_DEC_2022_UK_BGC.shp',
                      spacing_km=self.spacing_km)
        poi.preprocess()
        self.destinations, _, _ = poi.gen_features_grid(True, (8, 8), features_list=self.categories)

    def preprocess(self):
        """Apply necessary preprocessing to the OD dataset."""
        if self.city_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self.from_data = self._extract_coords(self.city_data['geometry_x'])
        self.to_data = self._extract_coords(self.city_data['geometry_y'])

    def create_od_matrix(self):
        """Create origin-destination matrix from processed data."""
        grid1, points1, cells1 = self.od_to_grid(self.from_data)
        grid2, points2, cells2 = self.od_to_grid(self.to_data)

        # Zip and count pairs
        pairs = list(zip(cells1, cells2))
        pair_counts = Counter(pairs)

        data = []

        for (from_cell, to_cell), count in pair_counts.items():
            dist = np.linalg.norm(np.array(to_cell) - np.array(from_cell))

            row = {
                "from_cell": from_cell,
                "to_cell": to_cell,
                "trips": count,
                "distance": dist,
                "origin": grid1[from_cell],
            }

            i = 0
            for cat in self.categories:
                row[cat] = self.destinations[to_cell[0], to_cell[1], i]
                i += 1

            data.append(row)

        return pd.DataFrame(data)

    def _extract_coords(self, geometry_series):
        """Extract coordinates from POINT geometry strings."""
        coords = geometry_series.str.extract(
            r"POINT \(([-\d.]+) ([-\d.]+)\)"
        ).astype(float)
        return coords.to_numpy()

    def od_to_grid(self, city_data):
        """Transform OD points to grid representation."""
        def _find_pos(point, x_scale, y_scale):
            """Find grid position for a given point."""
            x_scale = sorted(np.unique(x_scale))
            y_scale = sorted(np.unique(y_scale))

            found_i, found_j = -1, -1

            for i in range(1, len(x_scale)):
                if x_scale[i-1] <= point[0] < x_scale[i]:
                    found_i = i - 1
                    break
            if point[0] >= x_scale[-1]:
                found_i = len(x_scale) - 1

            for j in range(1, len(y_scale)):
                if y_scale[j-1] <= point[1] < y_scale[j]:
                    found_j = j - 1
                    break
            if point[1] >= y_scale[-1]:
                found_j = len(y_scale) - 1

            return (found_i, found_j)

        def _scale_pt(point, origin_x, new_x, origin_y, new_y):
            """Scale point coordinates to new grid dimensions."""
            # Calculate the proportion for x-coordinate
            x_proportion = (point[0] - np.min(origin_x)) / (np.max(origin_x) - np.min(origin_x))
            # Map the proportion to the new x range
            new_point_x = np.min(new_x) + x_proportion * (np.max(new_x) - np.min(new_x))

            # Calculate the proportion for y-coordinate
            y_proportion = (point[1] - np.min(origin_y)) / (np.max(origin_y) - np.min(origin_y))
            # Map the proportion to the new y range
            new_point_y = np.min(new_y) + y_proportion * (np.max(new_y) - np.min(new_y))

            return new_point_x, new_point_y

        # 1. Generate the grid for the city shapefile using specified cell size
        od_grid = SpatialGrid(self.shp_path, self.spacing_km)
        lon_grid, lat_grid, grid_gdf = od_grid.gen_grid("",
                                                        grid_spacing_km=self.spacing_km,
                                                        center=None,
                                                        bottom_left_buffer_km=2)

        # 2. Assign points to the grid
        lat_size, lon_size, grid, norm_grid, ctrs, ctr_wts = od_grid.assign_grid(
            lon_grid, lat_grid,
            x_pts=city_data[:, 0],
            y_pts=city_data[:, 1]
        )
        data = np.hstack([ctrs, np.array(ctr_wts).reshape(-1, 1)])

        # 3. Resize the grid to the new size
        x_grid, y_grid, new_data = od_grid.resample_kde(
            data,
            (self.size[0] + 1, self.size[1] + 1)
        )

        # 4. Define the new grid dimensions
        origin_x = np.unique(data[:, 0])
        origin_y = np.unique(data[:, 1])

        new_y = np.unique(x_grid[0, :])
        new_x = np.unique(y_grid[:, 0])

        # 5. Transform points to the new grid size
        scaled_points = np.array([
            _scale_pt(p, origin_x, new_x, origin_y, new_y)
            for p in city_data
        ])

        point_cells = [
            _find_pos(p, new_x, new_y)
            for p in scaled_points
        ]

        # 6. Assign the new points to the new grid
        lat_size2, lon_size2, grid2, norm_grid2, ctrs2, ctr_wts2 = od_grid.assign_grid(
            new_x, new_y,
            x_pts=scaled_points[:, 0],
            y_pts=scaled_points[:, 1]
        )
        return grid2, scaled_points, point_cells