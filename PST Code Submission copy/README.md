# Beyond the Stitching Assumption: A Unified Framework for Multimodal Synthetic Data Evaluation via Semantic Quantization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21156486.svg)](https://doi.org/10.5281/zenodo.21156486)

Artifact for the paper submission. Contains:

- **SynGen** — four multimodal synthetic-data generation methods and the full evaluation pipeline.
- **SynEval** — the semantic quantization library used to implement the four evaluation axes.
- **Data** — pre-generated synthetic datasets for three benchmarks (Amazon Reviews, Kiva Loans, Fake Job Postings).

No LLM API key is required to reproduce the paper results; all pre-generated datasets are included.

---

## Repository layout

```
.
├── README.md              ← you are here
├── reproduce.sh           ← end-to-end reproduction script (Steps 1–3)
├── Dockerfile             ← Docker image definition
├── SynGen/                ← generation methods + evaluation scripts
│   ├── syneval_quantization.py     # Step 1: semantic quantization
│   ├── syneval_four_dimensions.py  # Step 2: four-dimension evaluation (Table 2 / Figure 3)
│   ├── run_traditional_metrics.py  # Step 3: traditional isolated metrics (Table 1)
│   ├── generate_final_report.py    # Optional: consolidated report
│   ├── generators/                 # CTGAN+LLM, Prompt-LLM, Diffusion, Tilted
│   ├── experiments/
│   │   └── baselines_filtered_20260428_195011/
│   │       ├── synthetic_data/     ← pre-generated datasets (input)
│   │       ├── traditional_metrics/
│   │       │   └── traditional_metrics_results.csv   ← Table 1 output
│   │       └── syneval/
│   │           └── four_dimensions/
│   │               └── four_dimensions_results.csv   ← Table 2 / Figure 3 output
│   └── requirements.txt
└── SynEval/               ← semantic quantization library
    └── requirements.txt
```

---

## System requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| GPU | not required | CUDA (speeds up SBERT encoding) |
| Internet | not required for reproduction | required for Step 0 only |

---

## Quickstart — Docker (recommended)

### 1. Build the image

```bash
docker build -t syngen-artifact .
```

### 2. Run the reproduction script

```bash
docker run --rm syngen-artifact bash /workspace/reproduce.sh
```

The script runs all three evaluation steps automatically and prints the location of both result CSVs when it finishes. Expected runtime: **30–90 minutes** on a modern CPU.

### 3. Copy results out of the container (optional)

```bash
docker run --rm \
  -v "$(pwd)/results_out:/out" \
  syngen-artifact \
  bash -c "bash /workspace/reproduce.sh && \
           cp -r /workspace/SynGen/experiments/baselines_filtered_20260428_195011/traditional_metrics /out/ && \
           cp -r /workspace/SynGen/experiments/baselines_filtered_20260428_195011/syneval/four_dimensions /out/"
```

---

## Quickstart — local Python environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# Install both packages
pip install -r SynEval/requirements.txt
pip install -e SynEval/
pip install -r SynGen/requirements.txt
pip install -e SynGen/

# Run the full reproduction pipeline
bash reproduce.sh
```

---

## Reproduction steps (manual)

`reproduce.sh` executes the following three steps from inside `SynGen/`:

### Step 1 — Semantic quantization

Maps all synthetic datasets to a discrete (C_X, C_T) grid.

```bash
cd SynGen
python syneval_quantization.py
```

Output: `experiments/.../syneval/quantized_data/`

### Step 2 — Four-dimension evaluation (Table 2 / Figure 3)

Computes Fidelity, Utility, Diversity, and Privacy axes.

```bash
python syneval_four_dimensions.py
```

Output: `experiments/.../syneval/four_dimensions/four_dimensions_results.csv`

### Step 3 — Traditional isolated metrics (Table 1)

Demonstrates the Stitching Fallacy using SDV (KS / TV Complement) and BERTScore.

```bash
python run_traditional_metrics.py
```

Output: `experiments/.../traditional_metrics/traditional_metrics_results.csv`

### Optional — Consolidated report

```bash
python generate_final_report.py
```

Output: `experiments/.../FINAL_REPORT.md` (Markdown summary of both tables)

---

## Step 0 — Regenerate synthetic data from scratch (optional)

Pre-generated data is already included in `SynGen/experiments/`. To regenerate from scratch using LLM APIs:

```bash
export OPENAI_API_KEY="sk-..."
cd SynGen
python run_full_experiment.py
```

Expected runtime: 2–4 hours. Requires an OpenAI or Anthropic API key.

---

## Paper results

| File | Paper reference |
|---|---|
| `traditional_metrics/traditional_metrics_results.csv` | Table 1 |
| `syneval/four_dimensions/four_dimensions_results.csv` | Table 2, Figure 3 |

Both files are produced automatically by `reproduce.sh`.
