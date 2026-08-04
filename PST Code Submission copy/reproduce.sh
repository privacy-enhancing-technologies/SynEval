#!/usr/bin/env bash
# reproduce.sh — Reproduce all paper results from pre-generated synthetic data.
#
# No LLM API key required. Expected runtime: 30–90 minutes.
# To generate synthetic data from scratch instead, see README.md Step 0.
#
# Usage:
#   bash reproduce.sh            # from the project root
#   bash /workspace/reproduce.sh # from inside the Docker container

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNGEN_DIR="$SCRIPT_DIR/SynGen"

cd "$SYNGEN_DIR"

RESULTS_DIR="experiments/baselines_filtered_20260428_195011"

echo "========================================================"
echo " SynGen + SynEval — Artifact Reproduction"
echo "========================================================"
echo ""
echo "Working directory: $SYNGEN_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# ── Step 1: Semantic quantization ────────────────────────────────────────────
echo "[Step 1/3] Semantic quantization (maps datasets to (C_X, C_T) grid)"
echo "   Output: $RESULTS_DIR/syneval/quantized_data/"
echo ""
python syneval_quantization.py
echo ""

# ── Step 2: Four-dimension evaluation ────────────────────────────────────────
echo "[Step 2/3] Four-dimension evaluation  →  Table 2 + Figure 3"
echo "   Output: $RESULTS_DIR/syneval/four_dimensions/four_dimensions_results.csv"
echo ""
python syneval_four_dimensions.py
echo ""

# ── Step 3: Traditional isolated metrics ─────────────────────────────────────
echo "[Step 3/3] Traditional isolated metrics  →  Table 1"
echo "   Output: $RESULTS_DIR/traditional_metrics/traditional_metrics_results.csv"
echo ""
python run_traditional_metrics.py
echo ""

echo "========================================================"
echo " Reproduction complete."
echo ""
echo " Paper results are in:"
echo "   Table 1:        $SYNGEN_DIR/$RESULTS_DIR/traditional_metrics/traditional_metrics_results.csv"
echo "   Table 2 / Fig 3: $SYNGEN_DIR/$RESULTS_DIR/syneval/four_dimensions/four_dimensions_results.csv"
echo "========================================================"
