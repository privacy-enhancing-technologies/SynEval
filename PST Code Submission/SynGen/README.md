# SynGen: Multimodal Synthetic Data Generation Framework

SynGen is the artifact accompanying the paper submission. It provides four methods for generating multimodal synthetic datasets (paired tabular + text) and a complete evaluation pipeline based on the SynEval four-dimension framework.

---

## Artifact Contents

```
SynGen/
├── generators/                  # Core generation methods
│   ├── base.py                  # Abstract BaseGenerator interface
│   ├── ctgan_llm_stitcher.py    # Baseline: independent CTGAN + LLM (Stitching Fallacy)
│   ├── prompt_llm.py            # Baseline: tabular-conditioned LLM generation
│   ├── multimodal_diffusion.py  # Proposed: joint diffusion in shared latent space
│   └── tilted.py                # Adversarial baseline: shuffled text
├── cli/                         # Command-line interface
│   ├── generate.py              # Entry point (4 subcommands)
│   └── utils.py                 # Shared CLI utilities
├── config/                      # Configuration management
│   ├── defaults.py              # Default configs per method
│   ├── loader.py                # YAML/JSON config loader
│   └── examples/                # Example YAML configs
├── utils/                       # Shared utilities
├── tests/                       # Unit and integration tests (113 passing)
├── examples/                    # Demo scripts
├── experiments/                 # Experimental outputs (data, metrics, reports)
├── syneval_quantization.py      # Step 1: semantic quantization preprocessing
├── syneval_four_dimensions.py   # Step 2: four-axis evaluation
├── run_traditional_metrics.py   # Comparison: traditional isolated metrics
├── create_tilted_data.py        # Creates adversarial tilted datasets
├── check_category_coverage.py   # Category coverage analysis
├── run_full_experiment.py       # End-to-end pipeline runner
├── generate_final_report.py     # Report generation
├── setup.py                     # Package installation
└── requirements.txt             # Python dependencies
```

---

## System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| **Python** | 3.9 | 3.10 |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB | 10 GB |
| **GPU** | None (CPU works) | CUDA GPU (5–10× faster for Diffusion) |
| **Internet** | Required for LLM APIs | — |

> **LLM API keys** (OpenAI or Anthropic) are required for the CTGAN+LLM Stitcher and Prompt-Conditioned LLM baselines. The Multimodal Diffusion and Tilted baselines run fully offline.

---

## Installation

### Option A — Docker (recommended for reviewers)

Build the image from the project root (the `Dockerfile` is one level above `SynGen/`):

```bash
# From the artifact root directory (contains both SynGen/ and SynEval/)
docker build -t syngen-artifact .

# Run interactively (no API key needed for Steps 1–4)
docker run -it --rm syngen-artifact bash

# Run interactively with an API key for fresh generation (Step 0)
docker run -it --rm \
  -e OPENAI_API_KEY=<your-key> \
  syngen-artifact bash
```

Inside the container the working directory is `/workspace/SynGen`. All reproduction scripts can be run directly with `python <script>.py`.

### Option B — Local Python environment

Install both packages from the artifact root:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 2. Install SynEval (evaluation dependency)
pip install -r ../SynEval/requirements.txt
pip install -e ../SynEval/

# 3. Install SynGen
pip install -r requirements.txt
pip install -e .

# 4. Set API keys (only needed for LLM-based methods in Step 0)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Quick Start

### Generate synthetic data with each method

```bash
# No API key needed — adversarial baseline
syngen tilted \
  --input data/kiva_loans.csv \
  --output synthetic/kiva_tilted.csv \
  --text-column use \
  --tabular-column sector \
  --n-samples 1000

# Requires OPENAI_API_KEY
syngen prompt-llm \
  --input data/kiva_loans.csv \
  --output synthetic/kiva_prompt_llm.csv \
  --text-column use \
  --tabular-column sector \
  --n-samples 1000 \
  --model gpt-4o-mini

# No API key needed — joint diffusion model
syngen diffusion \
  --input data/kiva_loans.csv \
  --output synthetic/kiva_diffusion.csv \
  --text-column use \
  --tabular-column sector \
  --n-samples 1000 \
  --epochs 100
```

See `CLI_README.md` for complete CLI documentation and all options.

---

## Reproducing Paper Results

**Quick path (recommended for reviewers):** Pre-generated synthetic datasets are bundled in
`experiments/baselines_filtered_20260428_195011/synthetic_data/`. Skip Step 0 entirely and
run Steps 1–4. From the artifact root you can also run the all-in-one script:

```bash
bash reproduce.sh      # ~30–90 min, no API key required
```

The full pipeline runs in four steps. Each step reads from the previous step's output in `experiments/`.

### Step 0 — Generate baselines (optional — requires API key + 2–4 hours)

Pre-generated synthetic datasets are included — Step 0 is **not required** to reproduce
Tables 1–2 and Figure 3. To regenerate from scratch (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY="sk-..."
python run_full_experiment.py
```

> Expected runtime: 2–4 hours (3 datasets × 3 LLM baselines at 1,000 samples each).  
> GPU recommended for the Diffusion baseline (5–10× faster training).

### Step 1 — Semantic quantization

Maps all datasets onto a discrete (C_X, C_T) grid for evaluation:

```bash
python syneval_quantization.py
```

Output: `experiments/.../syneval/quantized_data/*.csv` and fitted quantizer models.

### Step 2 — Four-dimension evaluation (SynEval)

Computes the four evaluation axes (Fidelity, Utility, Diversity, Privacy) on the Fake Job
Postings dataset with three tabular configurations: binary fraud detection (`fraudulent`),
binary logo presence (`has_company_logo`), and joint multiclass (`fraudulent × has_company_logo`).

```bash
python syneval_four_dimensions.py
```

Output: `experiments/.../syneval/four_dimensions/four_dimensions_results.csv`
This file reproduces **Table 2** and **Figure 3** in the paper.

### Step 3 — Traditional metrics comparison

Demonstrates the Stitching Fallacy with industry-standard isolated metrics:

```bash
python run_traditional_metrics.py
```

Output: `experiments/.../traditional_metrics/traditional_metrics_results.csv`
This file reproduces **Table 1** in the paper.

### Step 4 — Generate report

```bash
python generate_final_report.py
```

Produces a consolidated summary of all results.

---

## Running Tests

```bash
pytest tests/ -v
```

113 tests cover all four generators, CLI integration, and end-to-end behaviour.

---

## Generation Methods

| Method | Description | Preserves Correlation | API Required | Train Required |
|---|---|---|---|---|
| **CTGAN+LLM Stitcher** | Independent CTGAN + LLM, random pairing | No (Stitching Fallacy) | Yes | CTGAN only |
| **Prompt-Conditioned LLM** | LLM text conditioned on tabular row | Yes | Yes | No |
| **Multimodal Diffusion** | Joint denoising in shared latent space | Yes | No | Yes (~5 min) |
| **Tilted Data** | Row-shuffled text, intact tabular | No (adversarial) | No | No |

---

## Evaluation Framework (SynEval)

| Axis | Metric | Description |
|---|---|---|
| **I — Fidelity** | Conditional JSD + MMD | Cross-modal distribution divergence |
| **II — Utility** | Bidirectional TSTR F1 | Text→Attribute and Attribute→Text prediction |
| **III — Diversity** | Joint Shannon Entropy | Coverage of the (C_X, C_T) space |
| **IV — Privacy** | Mean DCR | Distance to closest real record |

---

## Datasets

The experiments use three publicly available datasets:

| Dataset | Source | Task | Text Column |
|---|---|---|---|
| Amazon Reviews | Hugging Face | Rating prediction | `text` |
| Kiva Loans | Kaggle | Sector classification | `use` |
| Fake Job Postings | Kaggle | Fraud detection | `description` |

Raw and processed datasets are included in `experiments/.../synthetic_data/`.
