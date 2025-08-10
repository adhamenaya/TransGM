"""
Domain Adaptation Penalty Functions
"""
import numpy as np
from typing import Callable, List


class AdaptFunc:
    """Collection of penalty functions for domain adaptation."""

    @staticmethod
    def inv_sig(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Inverse sigmoid: k / (1 + exp(-divs))"""
        return k / (1 + np.exp(np.clip(-divs, -500, 500)))

    @staticmethod
    def sig(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Sigmoid: k / (1 + exp(divs))"""
        return k / (1 + np.exp(np.clip(divs, -500, 500)))

    @staticmethod
    def sqrt(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Square root: k / sqrt(divs + eps)"""
        return k / np.sqrt(divs + 1e-6)

    @staticmethod
    def log(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Logarithmic: k / log(divs + 1.1)"""
        return k / np.log(divs + 1.1)

    @staticmethod
    def exp(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Exponential: exp(divs + 1.1)"""
        return np.exp(np.clip(divs + 1.1, -500, 50))

    @staticmethod
    def cons(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Conservative: k * (1 - norm_divs)"""
        penalty = k * (1 - divs / (divs.max() + 1e-6))
        return np.clip(penalty, 0.01, 1.0)

    @staticmethod
    def none(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """No penalty: constant weights"""
        return np.ones_like(divs) * k

    @staticmethod
    def lin(divs: np.ndarray, k: float = 1.0) -> np.ndarray:
        """Linear: k * (1 - norm_divs)"""
        min_div, max_div = divs.min(), divs.max()
        if max_div - min_div > 1e-10:
            norm_divs = (divs - min_div) / (max_div - min_div)
        else:
            norm_divs = np.zeros_like(divs)
        return k * (1 - norm_divs)

    @classmethod
    def get_all(cls) -> List[Callable]:
        """Get all penalty functions."""
        return [cls.sig, cls.inv_sig, cls.sqrt, cls.log, cls.exp, cls.cons, cls.none, cls.lin]

    @classmethod
    def get_names(cls) -> List[str]:
        """Get function names."""
        return [func.__name__ for func in cls.get_all()]

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get function by name."""
        funcs = {f.__name__: f for f in cls.get_all()}
        if name not in funcs:
            raise ValueError(f"Unknown function: {name}. Available: {list(funcs.keys())}")
        return funcs[name]