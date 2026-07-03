# SynGen CLI Documentation

## Installation

```bash
cd SynGen
pip install -e .
```

This will install the `syngen` command globally.

## Usage

### Basic Command Structure

```bash
syngen <method> --input <data.csv> --output <synthetic.csv> \
  --text-column <col> --tabular-column <col> --n-samples <N> [options]
```

### Available Methods

1. **CTGAN + LLM Stitcher** (`ctgan-llm`) - Separate generation + random pairing
2. **Prompt-Conditioned LLM** (`prompt-llm`) - LLM conditioned on tabular data  
3. **Multimodal Diffusion** (`diffusion`) - Joint diffusion model
4. **Tilted** (`tilted`) - Adversarial baseline (random shuffle)

---

## Examples

### 1. Tilted Generator (Adversarial Baseline)

Simplest method - randomly shuffles text-tabular pairs:

```bash
syngen tilted \
  --input data/kiva_loans.csv \
  --output synthetic/tilted_1000.csv \
  --text-column use \
  --tabular-column sector \
  --tabular-column loan_amount \
  --n-samples 1000
```

**Options:**
- `--shuffle-strategy`: `random` (default), `stratified`, or `adversarial`
- `--random-seed`: Random seed for reproducibility (default: 42)

---

### 2. CTGAN + LLM Stitcher

Uses CTGAN for tabular, LLM for text, then randomly pairs them:

```bash
export OPENAI_API_KEY="your-key"

syngen ctgan-llm \
  --input data/kiva_loans.csv \
  --output synthetic/ctgan_llm_1000.csv \
  --text-column use \
  --tabular-column sector \
  --tabular-column loan_amount \
  --n-samples 1000 \
  --provider openai \
  --model gpt-4o-mini \
  --n-few-shot 3
```

**Options:**
- `--provider`: `openai` (default) or `anthropic`
- `--model`: Model name (e.g., `gpt-4o-mini`, `claude-3-5-sonnet-latest`)
- `--n-few-shot`: Number of examples for LLM (default: 3)
- `--random-seed`: Random seed

---

### 3. Prompt-Conditioned LLM

LLM generates text conditioned on tabular values:

```bash
export OPENAI_API_KEY="your-key"

syngen prompt-llm \
  --input data/kiva_loans.csv \
  --output synthetic/prompt_llm_1000.csv \
  --text-column use \
  --tabular-column sector \
  --tabular-column loan_amount \
  --n-samples 1000 \
  --provider openai \
  --model gpt-4o-mini \
  --temperature 0.8
```

**Options:**
- `--provider`: `openai` or `anthropic`
- `--model`: Model name
- `--temperature`: Sampling temperature (default: 0.8)
- `--batch-size`: Batch size for generation (default: 10)
- `--max-retries`: Max API retries (default: 3)

---

### 4. Multimodal Diffusion

Joint diffusion model for text and tabular:

```bash
syngen diffusion \
  --input data/kiva_loans.csv \
  --output synthetic/diffusion_1000.csv \
  --text-column use \
  --tabular-column sector \
  --tabular-column loan_amount \
  --n-samples 1000 \
  --epochs 100 \
  --latent-dim 128
```

**Options:**
- `--text-encoder`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--latent-dim`: Latent dimension (default: 128)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--n-diffusion-steps`: Number of diffusion steps (default: 50)
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)

---

## Using Configuration Files

Instead of command-line arguments, you can use YAML or JSON config files:

```bash
syngen tilted \
  --input data/kiva_loans.csv \
  --output synthetic/output.csv \
  --text-column use \
  --tabular-column sector \
  --n-samples 1000 \
  --config config/examples/tilted.yaml
```

Example config file (`config/examples/tilted.yaml`):

```yaml
method: tilted
shuffle_strategy: random
random_state: 42
```

See `config/examples/` for more configuration examples.

---

## Verbose Logging

Add `--verbose` or `-v` to any command for detailed logging:

```bash
syngen tilted --input data.csv --output synthetic.csv \
  --text-column use --tabular-column sector \
  --n-samples 1000 --verbose
```

---

## Output Statistics

After generation, SynGen prints comparison statistics:

```
============================================================
GENERATION COMPLETE
============================================================
Real samples: 10000
Synthetic samples: 1000
Text columns: ['use']
Tabular columns: ['sector', 'loan_amount']

Tabular Column Statistics:
------------------------------------------------------------

sector:
  Real unique values: 15
  Synthetic unique values: 14

loan_amount:
  Real - Mean: 542.12, Std: 623.45
  Synthetic - Mean: 538.67, Std: 610.23

Text Column Statistics:
------------------------------------------------------------

use:
  Real - Mean length: 54.2, Std: 32.1
  Synthetic - Mean length: 52.8, Std: 30.4

============================================================
```

---

## Multiple Text/Tabular Columns

You can specify multiple columns:

```bash
syngen tilted \
  --input data.csv \
  --output synthetic.csv \
  --text-column description \
  --text-column requirements \
  --tabular-column sector \
  --tabular-column amount \
  --tabular-column duration \
  --n-samples 1000
```

---

## Complete Workflow Example

```bash
# 1. Generate with all 4 methods
syngen tilted --input real.csv --output syn_tilted.csv \
  --text-column use --tabular-column sector -n 1000

syngen ctgan-llm --input real.csv --output syn_ctgan.csv \
  --text-column use --tabular-column sector -n 1000

syngen prompt-llm --input real.csv --output syn_prompt.csv \
  --text-column use --tabular-column sector -n 1000

syngen diffusion --input real.csv --output syn_diffusion.csv \
  --text-column use --tabular-column sector -n 1000

# 2. Evaluate with SynEval (in SynEval directory)
cd ../SynEval
python -m syneval \
  --original ../Data/real.csv \
  --synthetic ../SynGen/syn_tilted.csv \
  --text-columns use \
  --tabular-columns sector \
  --dimensions fidelity utility diversity privacy
```

---

## Troubleshooting

**API Key Issues:**
```bash
# Set your API key as environment variable
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

**Module Import Errors:**
```bash
# Reinstall in development mode
pip install -e .
```

**CUDA/PyTorch Issues:**
```bash
# Diffusion runs on CPU by default (no CUDA needed)
# For GPU support, install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Testing

Run integration tests:

```bash
pytest tests/integration/ -v
```

Run all tests (unit + integration):

```bash
pytest tests/ -v
```
