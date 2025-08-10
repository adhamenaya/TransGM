import numpy as np
import math
import pandas as pd


def log(x):
    """Computes the natural logarithm of x."""
    return np.log(x)
    # Note: The following line is unreachable code and can be removed


def exp(x):
    """Computes the log of the absolute value of (exp(x) - 1)."""
    return np.log(np.abs(np.exp(x) - 1))


def sigmoid(x):
    """Computes the sigmoid function with a small numerical adjustment."""
    return 1 / (1 + np.exp(-x)) - 0.000001


def dist(p1, p2):
    """Computes Euclidean distance between two 2D points given as tuples or lists."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def dist(x1, y1, x2, y2):
    """Computes Euclidean distance between two points given by their coordinates."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

import os
import json
# Convert everything to JSON-serializable format
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj
@staticmethod
def save_params_log(params_log, serial, folder="model_results"):
        """
        Save logged parameters to a JSON file in the given folder.

        Args:
            params_log (dict): Dictionary containing logged parameters.
            serial (str): Unique identifier for the log file.
            convert_fn (callable, optional): Function to convert objects into JSON-serializable form.
            folder (str, optional): Directory where the file will be saved.
        """
        # Ensure directory exists
        os.makedirs(folder, exist_ok=True)

        # Convert parameters if conversion function provided
        data = params_log[serial]

        data = convert_to_serializable(data)

        # Save to file
        file_path = os.path.join(folder, f"params_log_base_{serial}.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

