import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import MinMaxScaler
import ast
import json
import utils

scaler = MinMaxScaler()

class DataSet:
    def __init__(self, city, size=(8,8)):
        self.size = size
        self.categories = set()

        # Load categories from city data
        for city in ['coventry', 'birmingham']:
            file_path = f'input/poi/improved_poi_{city}.csv'

            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                if 'category' in data.columns:
                    categories = data['category'].dropna().replace('entertainment', 'leisure')
                    self.categories.update(categories.values)
            else:
                print(f"Warning: File not found - {file_path}")

        self.categories = sorted(list(self.categories))

        #TODO: calculate it ------>
        # Load divergence configurations
        df_divs = pd.read_csv('div_results.csv')
        self.divs = []
        self.divs_name = []

        for category in self.categories:
            matched = df_divs[df_divs['feature'] == category]['observed_divergence'].values
            if matched.size > 0 and pd.notna(matched[0]) and matched[0] > 0:
                self.divs.append(float(matched[0]))
                self.divs_name.append(category)

        self.divs = np.array(self.divs)

    # Matrix creation functions
    def create_trip_matrix(self, dataset):
        """Converts the 'trips' column from dataset into matrix representation"""
        trips = dataset[["trips"]]
        trips.reset_index(inplace=True)
        trips = trips.values

        unique_pairs = [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]
        pair_to_index = {pair: i for i, pair in enumerate(unique_pairs)}

        matrix = np.zeros((self.size[0]**2, self.size[1]**2))

        for t in trips:
            origin = ast.literal_eval(t[0]) if isinstance(t[0], str) else t[0]
            destination = ast.literal_eval(t[1]) if isinstance(t[1], str) else t[1]
            matrix[pair_to_index[origin], pair_to_index[destination]] = t[2]

        return matrix

    def create_origin_matrix(self, dataset):
        """Create matrix from origin column"""
        X0 = dataset[["origin"]]
        X0.reset_index(inplace=True)
        X0 = X0.values

        unique_pairs = [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]
        pair_to_index = {pair: i for i, pair in enumerate(unique_pairs)}

        matrix = np.zeros((len(pair_to_index), len(pair_to_index)))

        for x0 in X0:
            origin = ast.literal_eval(x0[0]) if isinstance(x0[0], str) else x0[0]
            destination = ast.literal_eval(x0[1]) if isinstance(x0[1], str) else x0[1]
            matrix[pair_to_index[origin], pair_to_index[destination]] = x0[2]

        return matrix

    def create_dest_feat_matrix(self, dataset):
        """Creates 3D matrix to store multiple feature values based on 'to_cell' pairs"""
        X1 = dataset.drop(columns=['trips', 'distance', 'origin', 'other', 'residential', 'public'])
        X1.reset_index(inplace=True)

        unique_pairs = [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]
        pair_to_index = {pair: i for i, pair in enumerate(unique_pairs)}

        num_features = X1.shape[1] - 2
        matrix3d = np.zeros((len(unique_pairs), len(unique_pairs), num_features))

        for x1 in X1.values:
            origin = ast.literal_eval(x1[0]) if isinstance(x1[0], str) else x1[0]
            destination = ast.literal_eval(x1[1]) if isinstance(x1[1], str) else x1[1]
            matrix3d[pair_to_index[origin], pair_to_index[destination], :] = x1[2:]

        return matrix3d

    def create_dist_matrix(self, dataset):
        """Create distance matrix"""
        X2 = dataset[["distance"]]
        X2.reset_index(inplace=True)
        X2 = X2.values

        unique_pairs = [(i, j) for i in range(self.size[0]) for j in range(self.size[1])]
        pair_to_index = {pair: i for i, pair in enumerate(unique_pairs)}

        def calc_euclidean_dist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        matrix = np.zeros((len(unique_pairs), len(unique_pairs)))

        for x22 in range(matrix.shape[0]):
            for y22 in range(matrix.shape[1]):
                x1, y1 = unique_pairs[x22]
                x2, y2 = unique_pairs[y22]
                matrix[x22, y22] = calc_euclidean_dist(x1, y1, x2, y2)

        return matrix

    def get_city_data(self, city_name, sample_ratio=1.0, random_seed=None):
        """Get and process city data with optional sampling"""
        dataset = pd.read_csv(f'input/od/{city_name}_od_grid.csv')
        print(f"Original dataset shape: {dataset.shape}")

        y = dataset['trips'].values
        print(f"Original y shape: {y.shape}")

        X = dataset.drop(columns=['trips'])
        print(f"Original X shape: {X.shape}")

        # Handle sampling
        if 0.0 < sample_ratio < 1.0:
            if random_seed is not None:
                np.random.seed(random_seed)

            sample_indices = np.random.choice(
                dataset.index,
                size=int(len(dataset) * sample_ratio),
                replace=False
            )
            subset_dataset = dataset.loc[sample_indices].copy()

            y = subset_dataset['trips'].values
            X = subset_dataset.drop(columns=['trips'])

            print(f"\nCreating subset with sample_ratio = {sample_ratio}")
            print(f"Subset dataset shape: {subset_dataset.shape}")
            print(f"Subset y shape: {y.shape}")
            print(f"Subset X shape: {X.shape}")

        elif sample_ratio < 0.0 or sample_ratio > 1.0:
            raise ValueError("sample_ratio must be between 0.0 and 1.0")

        elif sample_ratio == 0.0:
            print("Returning an empty dataset based on sample_ratio = 0.0")
            return pd.DataFrame(), np.array([])

        else:
            print("Using the entire dataset (sample_ratio = 1.0)")

        return X, y

    def transform_to_matrices(self, dataset):
        """Transform dataset to matrices with scaling"""
        # Dependent variable
        y = self.create_trip_matrix(dataset)

        # Independent variables
        X0 = self.create_origin_matrix(dataset)
        X2 = self.create_dist_matrix(dataset)
        X1 = self.create_dest_feat_matrix(dataset)

        # Transform features
        scaled_matrix = scaler.fit_transform(X1.reshape(-1, 11)) + 1e-6
        X1 = scaled_matrix.reshape(X1.shape)
        X2 = scaler.fit_transform(X2) + 1e-6
        X0 = scaler.fit_transform(X0) + 1e-6
        y = scaler.fit_transform(utils.log(y + 1e-6)) + 1e-6

        return X0, X1, X2, y