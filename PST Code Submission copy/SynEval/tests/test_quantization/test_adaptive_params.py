import pytest
from evaluation.quantization.adaptive_params import compute_adaptive_k, compute_adaptive_bins, select_adaptive_params


def test_compute_adaptive_k_small_dataset():
    """Small dataset (<1000) should get K proportional to size"""
    n_samples = 500
    k = compute_adaptive_k(n_samples)
    assert k == max(5, n_samples // 100)
    assert k == 5


def test_compute_adaptive_k_medium_dataset():
    """Medium dataset (1K-10K) should get K up to 20"""
    n_samples = 5000
    k = compute_adaptive_k(n_samples)
    assert k == min(20, n_samples // 500)
    assert k == 10


def test_compute_adaptive_k_large_dataset():
    """Large dataset (>10K) should get K up to 50"""
    n_samples = 50000
    k = compute_adaptive_k(n_samples)
    assert k == min(50, n_samples // 500)
    assert k == 50


def test_compute_adaptive_bins_small_dataset():
    """Small dataset should get fewer bins"""
    n_samples = 500
    bins = compute_adaptive_bins(n_samples)
    assert bins == max(3, n_samples // 200)
    assert bins == 3


def test_compute_adaptive_bins_medium_dataset():
    """Medium dataset should get moderate bins"""
    n_samples = 5000
    bins = compute_adaptive_bins(n_samples)
    assert bins == min(10, n_samples // 1000)
    assert bins == 5


def test_compute_adaptive_bins_large_dataset():
    """Large dataset should get more bins"""
    n_samples = 50000
    bins = compute_adaptive_bins(n_samples)
    assert bins == min(20, n_samples // 1000)
    assert bins == 20


def test_select_adaptive_params_prevents_sparse_contingency():
    """Should reduce K and B if contingency table would be too sparse"""
    n_samples = 100  # Very small dataset
    k, b = select_adaptive_params(n_samples)

    # Ensure at least 5 samples per cell on average
    total_cells = k * b
    avg_samples_per_cell = n_samples / total_cells
    assert avg_samples_per_cell >= 5


def test_select_adaptive_params_normal_case():
    """Normal dataset should use standard adaptive formulas"""
    n_samples = 5000
    k, b = select_adaptive_params(n_samples)
    assert k == 10  # 5000 // 500
    assert b == 5   # 5000 // 1000
