"""
Domain Adaptation Penalty Functions
"""
import numpy as np
from typing import Callable, List


class AdaptFunc:
    def inverse_sigmoid_penalty(divs, k=1.0):
        """1. Sigmoid transformation"""
        return k / (1 + np.exp(-divs))

    def sigmoid_penalty(divs, k=1.0):
        """1. Sigmoid transformation"""
        return k / (1 + np.exp(divs))

    def sqrt_penalty(divs, k=1.0):
        """2. Square root transformation"""
        return k / np.sqrt(divs + 1e-6)

    def log_penalty(divs, k=1.0):
        """3. Logarithmic transformation"""
        return k / np.log(divs + 1.1)

    def exp_penalty(divs, k=1.0):
        """Alternative: Exponential growth penalty (rarely used)"""
        return np.exp(divs + 1.1)

    def conservative_penalty(divs, k=1.0):
        """4. Conservative penalty (small divs → less penalty)"""
        penalty = k * (1 - divs / (divs.max() + 1e-6))
        return np.clip(penalty, 0.01, 1.0)

    def no_penalty(divs, k=1.0):
        """5. Constant penalty (baseline comparison)"""
        return np.ones_like(divs)

    def linear_penalty(divs, k=1.0):
        """6. Linear penalty (inverse relation with divergence)"""
        return k * (np.ones_like(divs) - divs)