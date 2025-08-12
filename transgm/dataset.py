import ast
import os
from typing import Tuple, Optional, List, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utils
EXCLUDE_CATEGORIES = {'trips', 'distance', 'origins', 'other', 'residential', 'public'}

class DataSet:
    """
    A class for processing origin-destination trip data and converting it
    to matrix representations suitable for machine learning models.

    Attributes:
        size (tuple): Grid dimensions (default: 8x8)
        categories (set): Set of POI categories found in the data
        city (str): Name of the city being processed
    """

    def __init__(self, city, size =(8, 8)) :
        """
        Initialize the DataSet with city name and grid size.

        Args:
            city (str): Name of the city to process
            size (tuple): Grid dimensions as (rows, columns)
        """
        self.size = size
        self.categories: Set[str] = set()
        self.city = city
        self._scaler = MinMaxScaler()

        self._load_categories()

    def _load_categories(self) -> list[str]:
        """
        Load POI categories from city data files.

        Searches for improved POI files for Coventry and Birmingham,
        extracts categories and stores them in self.categories.
        """
        cities = ['coventry', 'birmingham']
        categories_set = set()

        for city in cities:
            file_path = f'input/poi/improved_poi_{city}.csv'

            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue

            try:
                data = pd.read_csv(file_path)
                if 'category' in data.columns:
                    filtered = (data['category']
                    .dropna()
                    .loc[lambda x: ~x.isin(EXCLUDE_CATEGORIES)])
                    categories_set.update(filtered.unique())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        self.categories = sorted(categories_set)
        return self.categories


    def _get_unique_pairs(self):
        """
        Generate all unique coordinate pairs for the grid.

        Returns:
            List of (i, j) coordinate pairs
        """
        return [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]

    def _get_pair_to_index_mapping(self):
        """
        Create mapping from coordinate pairs to matrix indices.

        Returns:
            Dictionary mapping (i, j) pairs to indices
        """
        unique_pairs = self._get_unique_pairs()
        return {pair: idx for idx, pair in enumerate(unique_pairs)}

    def _parse_coordinate(self, coord):
        """
        Parse coordinate from string or return as-is if already parsed.

        Args:
            coord: Coordinate as string or tuple

        Returns:
            Coordinate as tuple of integers
        """
        if isinstance(coord, str):
            return ast.literal_eval(coord)
        return coord

    def create_trip_matrix(self, data: pd.DataFrame):
        """
        Convert the 'trips' column from dataset into matrix representation.

        Args:
            data (pd.DataFrame): Dataset containing trips information

        Returns:
            np.ndarray: Square matrix of trip counts between grid cells
        """
        trips_data = data[["trips"]].copy()
        trips_data.reset_index(inplace=True)
        trips_values = trips_data.values

        pair_to_index = self._get_pair_to_index_mapping()
        matrix_size = self.size[0] * self.size[1]
        matrix = np.zeros((matrix_size, matrix_size))

        for trip in trips_values:
            origin = self._parse_coordinate(trip[0])
            destination = self._parse_coordinate(trip[1])
            trip_count = trip[2]

            origin_idx = pair_to_index[origin]
            dest_idx = pair_to_index[destination]
            matrix[origin_idx, dest_idx] = trip_count

        return matrix

    def create_origin_matrix(self, data: pd.DataFrame):
        """
        Create matrix from origin column data.

        Args:
            data (pd.DataFrame): Dataset containing origin information

        Returns:
            np.ndarray: Square matrix of origin-destination relationships
        """
        origin_data = data[["origin"]].copy()
        origin_data.reset_index(inplace=True)
        origin_values = origin_data.values

        pair_to_index = self._get_pair_to_index_mapping()
        matrix_size = len(pair_to_index)
        matrix = np.zeros((matrix_size, matrix_size))

        for origin_record in origin_values:
            origin = self._parse_coordinate(origin_record[0])
            destination = self._parse_coordinate(origin_record[1])
            value = origin_record[2]

            origin_idx = pair_to_index[origin]
            dest_idx = pair_to_index[destination]
            matrix[origin_idx, dest_idx] = value

        return matrix

    def create_dest_feat_matrix(self, data):
        """
        Create 3D matrix to store multiple feature values based on destination cell pairs.

        Args:
            data (pd.DataFrame): Dataset containing destination features

        Returns:
            np.ndarray: 3D matrix (origins x destinations x features)
        """
        missing = [cat for cat in self.categories if cat not in data.columns]
        if missing:
            raise ValueError(f"The following categories are missing from the data columns: {missing}")

        feature_data = data[self.categories].copy()
        feature_data.reset_index(inplace=True)

        unique_pairs = self._get_unique_pairs()
        pair_to_index = self._get_pair_to_index_mapping()

        # Calculate number of features (excluding index columns)
        num_features = feature_data.shape[1] - 2
        matrix_3d = np.zeros((len(unique_pairs), len(unique_pairs), num_features))

        for feature_record in feature_data.values:
            origin = self._parse_coordinate(feature_record[0])
            destination = self._parse_coordinate(feature_record[1])
            features = feature_record[2:]  # All columns after origin and destination

            origin_idx = pair_to_index[origin]
            dest_idx = pair_to_index[destination]
            matrix_3d[origin_idx, dest_idx, :] = features

        return matrix_3d

    def create_dist_matrix(self, data: pd.DataFrame):
        """
        Create distance matrix using Euclidean distance between grid cells.

        Args:
            data (pd.DataFrame): Dataset (used for size reference)

        Returns:
            np.ndarray: Square matrix of Euclidean distances
        """
        unique_pairs = self._get_unique_pairs()
        matrix_size = len(unique_pairs)
        matrix = np.zeros((matrix_size, matrix_size))

        for i in range(matrix_size):
            for j in range(matrix_size):
                x1, y1 = unique_pairs[i]
                x2, y2 = unique_pairs[j]
                distance = self._calculate_euclidean_distance(x1, y1, x2, y2)
                matrix[i, j] = distance

        return matrix

    @staticmethod
    def _calculate_euclidean_distance(x1, y1, x2, y2):
        """
        Calculate Euclidean distance between two points.

        Args:
            x1, y1 (float): Coordinates of first point
            x2, y2 (float): Coordinates of second point

        Returns:
            float: Euclidean distance
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def od_data(self, sample_ratio = 1.0, random_seed = 42):
        """
        Load and process origin-destination data with optional sampling.

        Args:
            sample_ratio (float): Fraction of data to sample (0.0 to 1.0)
            random_seed (Optional[int]): Random seed for reproducible sampling

        Returns:
            Tuple of (features DataFrame, target array)

        Raises:
            ValueError: If sample_ratio is not between 0.0 and 1.0
        """
        if not (0.0 <= sample_ratio <= 1.0):
            raise ValueError("sample_ratio must be between 0.0 and 1.0")

        # Load dataset
        dataset_path = f'input/od/{self.city}_od_grid.csv'
        dataset = pd.read_csv(dataset_path)
        print(f"Original dataset shape: {dataset.shape}")

        # Extract target variable
        y = dataset['trips'].values
        print(f"Original y shape: {y.shape}")

        # Extract features
        X = dataset.drop(columns=['trips'])
        print(f"Original X shape: {X.shape}")

        # Handle sampling
        if sample_ratio == 0.0:
            print("Returning empty dataset based on sample_ratio = 0.0")
            return pd.DataFrame(), np.array([])

        elif sample_ratio == 1.0:
            print("Using the entire dataset (sample_ratio = 1.0)")
            return X, y

        else:
            # Sample subset of data
            if random_seed is not None:
                np.random.seed(random_seed)

            sample_size = int(len(dataset) * sample_ratio)
            sample_indices = np.random.choice(
                dataset.index,
                size=sample_size,
                replace=False
            )

            subset_dataset = dataset.loc[sample_indices].copy()
            y_subset = subset_dataset['trips'].values
            X_subset = subset_dataset.drop(columns=['trips'])

            print(f"\nCreating subset with sample_ratio = {sample_ratio}")
            print(f"Subset dataset shape: {subset_dataset.shape}")
            print(f"Subset y shape: {y_subset.shape}")
            print(f"Subset X shape: {X_subset.shape}")

            return X_subset, y_subset

    def transform_to_matrices(self, dataset: pd.DataFrame):
        """
        Transform dataset to scaled matrices for machine learning.

        Args:
            dataset (pd.DataFrame): Input dataset

        Returns:
            Tuple of (origin_matrix, destination_features, distance_matrix, target_matrix)
        """
        # Create matrices
        y = self.create_trip_matrix(dataset)
        X0 = self.create_origin_matrix(dataset)
        X2 = self.create_dist_matrix(dataset)
        X1 = self.create_dest_feat_matrix(dataset)

        # Apply scaling with small epsilon to avoid zeros
        epsilon = 1e-6

        # Scale 3D destination features matrix
        original_shape = X1.shape
        X1_reshaped = X1.reshape(-1, original_shape[2])
        X1_scaled = self._scaler.fit_transform(X1_reshaped) + epsilon
        X1 = X1_scaled.reshape(original_shape)

        # Scale distance matrix
        X2 = self._scaler.fit_transform(X2) + epsilon

        # Scale origin matrix
        X0 = self._scaler.fit_transform(X0) + epsilon

        # Scale target matrix (with log transformation)
        y_log = utils.log(y + epsilon)
        y = self._scaler.fit_transform(y_log) + epsilon

        return X0, X1, X2, y