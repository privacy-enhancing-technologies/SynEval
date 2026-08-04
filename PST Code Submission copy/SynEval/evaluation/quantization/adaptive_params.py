"""Adaptive parameter selection for Semantic Quantization.

Selects K (text clusters) and B (tabular bins) based on dataset size
to ensure contingency tables are not too sparse.
"""
import numpy as np


def compute_adaptive_k(n_samples: int) -> int:
    """
    Compute adaptive K (number of text clusters) based on dataset size.

    Args:
        n_samples: Number of samples in dataset

    Returns:
        K (number of clusters) in range [5, 50]

    Rules:
        - Small (<1K): K = max(5, n // 100)
        - Medium (1K-10K): K = min(20, n // 500)
        - Large (>10K): K = min(50, n // 500)
    """
    if n_samples < 1000:
        return max(5, n_samples // 100)
    elif n_samples < 10000:
        return min(20, n_samples // 500)
    else:
        return min(50, n_samples // 500)


def compute_adaptive_bins(n_samples: int) -> int:
    """
    Compute adaptive B (number of tabular bins) based on dataset size.

    Args:
        n_samples: Number of samples in dataset

    Returns:
        B (number of bins) in range [3, 20]

    Rules:
        - Small (<1K): B = max(3, n // 200)
        - Medium (1K-10K): B = min(10, n // 1000)
        - Large (>10K): B = min(20, n // 1000)
    """
    if n_samples < 1000:
        return max(3, n_samples // 200)
    elif n_samples < 10000:
        return min(10, n_samples // 1000)
    else:
        return min(20, n_samples // 1000)


def select_adaptive_params(n_samples: int) -> tuple:
    """
    Select K and B ensuring contingency table not too sparse.

    Ensures at least 5 samples per cell on average: n / (K * B) >= 5

    Args:
        n_samples: Number of samples in dataset

    Returns:
        (K, B) tuple
    """
    k = compute_adaptive_k(n_samples)
    b = compute_adaptive_bins(n_samples)

    # Check sparsity
    total_cells = k * b
    avg_samples_per_cell = n_samples / total_cells

    # If too sparse, reduce K and B proportionally
    if avg_samples_per_cell < 5:
        scale_factor = np.sqrt(5 * total_cells / n_samples)
        k = max(3, int(k / scale_factor))
        b = max(3, int(b / scale_factor))

    return k, b
