#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module6_knn-regr.py
The program asks the user for input N (positive integer) and reads it.
Then the program asks the user for input k (positive integer) and reads it.
Then the program asks the user to provide N (x, y) points (one by one) and reads all of them: first: x value, then: y value for every point one by one. X and Y are the real numbers.
In the end, the program asks the user for input X and outputs: the result (Y) of k-NN Regression if k <= N, or any error message otherwise.
The basic functionality of data processing (data initialization, data insertion, data calculation) should be done using Numpy library as much as possible (note: you can combine with OOP from the previous task).

"""

from __future__ import annotations
import sys
from typing import Tuple
import numpy as np


def read_positive_int(prompt: str) -> int:
    """Read a positive integer from stdin with the given prompt."""
    while True:
        try:
            value_str = input(prompt).strip()
            value = int(value_str)
            if value <= 0:
                print("Please enter a positive integer (> 0).")
                continue
            return value
        except ValueError:
            print("Invalid integer. Please try again.")


def read_float(prompt: str) -> float:
    """Read a float from stdin with the given prompt."""
    while True:
        try:
            value_str = input(prompt).strip()
            return float(value_str)
        except ValueError:
            print("Invalid number. Please try again.")


class KNNRegressor1D:
    """Simple k-NN regressor for 1D input (x) that predicts y via mean of k nearest."""
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.xs: np.ndarray | None = None
        self.ys: np.ndarray | None = None

    def fit(self, xs: np.ndarray, ys: np.ndarray) -> None:
        if xs.ndim != 1 or ys.ndim != 1:
            raise ValueError("xs and ys must be 1D arrays.")
        if xs.shape[0] != ys.shape[0]:
            raise ValueError("xs and ys must have the same length.")
        if xs.shape[0] == 0:
            raise ValueError("At least one training sample is required.")
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)

    def predict_one(self, X: float) -> float:
        if self.xs is None or self.ys is None:
            raise RuntimeError("Model is not fitted.")
        n = self.xs.shape[0]
        if self.k > n:
            raise ValueError(f"k ({self.k}) cannot be larger than number of samples N ({n}).")
        # Distances in 1D: |x - X|
        dists = np.abs(self.xs - X)
        # Indices of k smallest distances (stable: full sort then slice)
        idx = np.argsort(dists)[: self.k]
        return float(np.mean(self.ys[idx]))


def main() -> int:
    print("=== k-NN Regression (1D) ===")
    N = read_positive_int("Enter N (number of points): ")
    k = read_positive_int("Enter k (number of neighbors): ")

    xs = np.empty(N, dtype=float)
    ys = np.empty(N, dtype=float)

    print(f"Now enter {N} points (x, y).")
    for i in range(N):
        xi = read_float(f"  Point {i+1} - x: ")
        yi = read_float(f"  Point {i+1} - y: ")
        xs[i] = xi
        ys[i] = yi

    # Initialize and fit the model
    try:
        model = KNNRegressor1D(k=k)
        model.fit(xs, ys)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    Xq = read_float("Enter query X: ")

    try:
        y_pred = model.predict_one(Xq)
        print(f"Predicted Y (k={k} NN Regression) at X={Xq}: {y_pred}")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
