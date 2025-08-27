#!/usr/bin/env python3
"""
module8_metrics-scikit.py

"""
from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score


def _read_positive_int(prompt: str) -> int:
    """Read a positive integer from stdin."""
    while True:
        try:
            value_str = input(prompt).strip()
            value = int(value_str)
            if value <= 0:
                print("Please enter a positive integer (> 0).")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def _read_binary_label(prompt: str) -> int:
    """Read a binary label 0 or 1 from stdin."""
    while True:
        s = input(prompt).strip()
        if s in {"0", "1"}:
            return int(s)
        print("Invalid input. Please enter 0 or 1.")


def _collect_pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect n pairs (x, y) where x is ground truth and y is prediction.
    Returns (y_true, y_pred) as NumPy arrays of shape (n,).
    """
    y_true = np.empty(n, dtype=np.int64)
    y_pred = np.empty(n, dtype=np.int64)

    print("\nEnter N pairs of labels. For each pair:")
    for i in range(n):
        x = _read_binary_label(f"  Pair {i+1}/{n} – enter x (ground truth, 0 or 1): ")
        y = _read_binary_label(f"  Pair {i+1}/{n} – enter y (prediction, 0 or 1): ")
        y_true[i] = x
        y_pred[i] = y
    return y_true, y_pred


def main() -> int:
    print("=== Binary Classification Metrics (Precision & Recall) ===")
    n = _read_positive_int("Enter N (number of (x, y) pairs): ")

    y_true, y_pred = _collect_pairs(n)

    # Compute metrics using scikit-learn
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)

    # Display results
    print("\n--- Results ---")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
