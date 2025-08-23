#!/usr/bin/env python3
"""
module7_knn-regr-scikit.py
"""

from __future__ import annotations

import sys
from typing import Tuple
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            value_str = input(prompt).strip()
            value = int(value_str)
            if value <= 0:
                print("Please enter a positive integer.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def read_float(prompt: str) -> float:
    while True:
        try:
            value_str = input(prompt).strip()
            return float(value_str)
        except ValueError:
            print("Invalid input. Please enter a real number (e.g., 3, -1.5, 2.0).")


def read_points(n: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    print(f"Please enter {n} points (x, y). You will be prompted for x then y for each point.")
    for i in range(n):
        x_i = read_float(f"Point {i+1} - x: ")
        y_i = read_float(f"Point {i+1} - y: ")
        xs[i] = x_i
        ys[i] = y_i
    return xs, ys


def main() -> None:
    print("=== k-NN Regression (scikit-learn) ===")
    N = read_positive_int("Enter N (number of points, positive integer): ")
    k = read_positive_int("Enter k (number of neighbors, positive integer): ")

    xs, ys = read_points(N)

    # Compute variance of labels (population variance by default, ddof=0)
    y_variance = float(np.var(ys, ddof=0))
    print(f"\nVariance of labels (y) in training data: {y_variance:.6f}")

    if k > N:
        print(f"Error: k ({k}) must be <= N ({N}).")
        sys.exit(1)

    # Reshape features for scikit-learn (N, 1)
    X_train = xs.reshape(-1, 1)
    y_train = ys

    # Create and fit k-NN regressor
    # You can choose weights='uniform' (default) or 'distance'.
    model = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    model.fit(X_train, y_train)

    # Read query X and predict Y
    X_query = read_float("\nEnter X (a real number) to predict Y with k-NN regression: ")
    y_pred = model.predict(np.array([[X_query]], dtype=np.float64))

    print(f"Predicted Y for X={X_query}: {float(y_pred[0]):.6f}")


if __name__ == "__main__":
    main()
