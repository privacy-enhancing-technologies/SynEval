# NER Analysis Utility (`use_cases/NER/ner_analysis.py`)

This utility runs a comprehensive Named Entity Recognition (NER) analysis on large text datasets (e.g., the Amazon Fashion reviews bundled with SynEval). It produces detailed reports, cached intermediate results, and optional visualisations to help you audit entity density and potential privacy risks.

## Features

- **Large Dataset Processing**: Handles tens of thousands of records with device-aware batching (CPU/GPU).
- **Entity Density Analysis**: Computes entity counts, per-text density, and dataset-wide statistics.
- **Comprehensive Reporting**: Generates multiple timestamped text reports describing the corpus.
- **Top Entity Texts**: Highlights the texts with the highest entity density (top 2000 by default).
- **Visualisations**: Saves summary charts (entity type distribution, density histogram).
- **Caching**: Uses a JSON cache (`./cache/amazon_fashion_ner_fast_results.json`) to avoid recomputation.
- **Progress Tracking**: Verbose logging and progress bars for long-running operations.

## Requirements

Ensure your environment satisfies the main SynEval requirements (`pip install -r requirements.txt`), then download the required NLTK datasets and install Flair if needed:

```bash
python -m nltk.downloader punkt punkt_tab stopwords
pip install flair  # Only if Flair did not install cleanly with requirements.txt
```

## Quick Start

By default the script analyses the bundled reviews dataset (`data/real_10k.csv`) using the `text` column:

```bash
python use_cases/NER/ner_analysis.py
```

All reports are written to `./reports`, cache to `./cache`, and progress printed to the console.

## Configuration

The entry-point of `use_cases/NER/ner_analysis.py` defines simple parameters you can tweak:

```python
# Configuration (within main())
csv_file = "data/real_10k.csv"
text_column = "text"
top_n = 2000  # Number of texts to keep in the "top entity" report
```

Adjust these values to target different datasets, text columns, or to reduce output size while testing.

## Output

When the run completes you will find the following artefacts inside `./reports` (timestamp in filenames omitted for brevity):

1. **Main Analysis Report** – `amazon_fashion_ner_fast_report_YYYYMMDD_HHMMSS.txt`  
   Dataset statistics, entity counts by type, density metrics, and representative entities.
2. **Top 2000 High Entity Texts** – `top_2000_high_entity_texts_fast_YYYYMMDD_HHMMSS.txt`  
   Highest-density texts with the entities that caused the spikes.
3. **Entity Density Analysis** – `entity_density_analysis_fast_YYYYMMDD_HHMMSS.txt`  
   Distribution statistics, percentile breakdowns, and top 50 density records.
4. **Visualisations** – `entity_distribution_fast_*.png`, `entity_density_histogram_fast_*.png`  
   Ready-to-share plots summarising entity frequency and density.

The cache directory stores JSON summaries for subsequent runs. Delete `./cache` if you wish to recompute from scratch.

## Custom Usage

You can import the analyser in your own scripts for custom pipelines:

```python
from use_cases.NER.ner_analysis import FastAmazonFashionNERAnalyzer

analyzer = FastAmazonFashionNERAnalyzer()
results = analyzer.analyze_dataset(
    csv_file="your_dataset.csv",
    text_column="review_text",
    top_n=500,
)
analyzer.generate_report(results, output_dir="./custom_reports")
```

## Performance Tips

| Scenario                     | Recommendation                                     |
|------------------------------|-----------------------------------------------------|
| Limited RAM / CPU            | Lower `top_n`, reduce batch size (see `_process_entities_batch_optimized`). |
| GPU available                | The script auto-detects CUDA and increases batch size. |
| Very large datasets          | Process in chunks and cache intermediate results.   |
| First run is slow            | Flair model download and cache creation happen once. |

Example chunked processing pattern:

```python
chunk_size = 100_000
for start in range(0, total_records, chunk_size):
    end = min(start + chunk_size, total_records)
    subset = df.iloc[start:end]
    # call analyzer on subset and persist intermediate outputs
```

## Troubleshooting

- **`Text column 'text' not found`** – verify the `text_column` argument matches your CSV headers.  
- **Flair model download errors** – ensure internet access or install the model from a local wheel.  
- **`LookupError: Resource punkt not found`** – run the `python -m nltk.downloader ...` command above.  
- **Memory errors / long runtimes** – decrease batch size, analyse a subset, or run on a machine with more RAM/GPU.

For further questions, open an issue or consult the main SynEval README.
