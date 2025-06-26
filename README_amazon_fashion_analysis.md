# Amazon Fashion Dataset - Named Entity Recognition Analysis

This script performs comprehensive Named Entity Recognition (NER) analysis on the Amazon Fashion dataset using Flair NLP. It processes 2.5M+ records, extracts entities from the "text" column, and generates detailed reports including entity density analysis.

## Features

- **Large Dataset Processing**: Efficiently handles 2.5M+ records with batch processing
- **Entity Density Analysis**: Calculates and analyzes entity density for each text
- **Comprehensive Reporting**: Generates multiple detailed report files
- **Top 200 High Entity Texts**: Identifies and reports texts with the most entities
- **Visualizations**: Creates charts and graphs for better understanding
- **Caching**: Automatic caching for faster subsequent runs
- **Progress Tracking**: Real-time progress bars for long-running operations

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_flair_ner.txt
```

Or install manually:

```bash
pip install pandas numpy torch flair tqdm matplotlib seaborn
```

### 2. Prepare Your Dataset

Ensure your `Amazon_Fashion.csv` file is in the same directory as the script and contains a column named "text".

## Usage

### Basic Usage

Simply run the script to analyze your entire dataset:

```bash
python amazon_fashion_ner_analysis.py
```

### Configuration Options

You can modify the script to:
- Use a subset of data for testing
- Change the text column name
- Adjust batch sizes for your hardware

Edit the configuration section in the `main()` function:

```python
# Configuration
csv_file = 'Amazon_Fashion.csv'
text_column = 'text'
sample_size = None  # Set to 10000 to test with first 10K records
```

## Output Files

The script generates several report files in the `./reports` directory:

### 1. Main Analysis Report
- **File**: `amazon_fashion_ner_report_YYYYMMDD_HHMMSS.txt`
- **Content**: 
  - Dataset statistics
  - Entity counts by type
  - Overall entity density analysis
  - Sample entities for each type

### 2. Top 200 High Entity Texts Report
- **File**: `top_200_high_entity_texts_YYYYMMDD_HHMMSS.txt`
- **Content**: 
  - 200 texts with the highest entity counts
  - Entity density for each text
  - List of entities found in each text

### 3. Entity Density Analysis Report
- **File**: `entity_density_analysis_YYYYMMDD_HHMMSS.txt`
- **Content**:
  - Detailed density statistics (mean, median, percentiles)
  - Density distribution analysis
  - Top 50 texts by entity density

### 4. Visualizations
- **Files**: 
  - `entity_distribution_YYYYMMDD_HHMMSS.png`
  - `entity_density_histogram_YYYYMMDD_HHMMSS.png`
  - `entity_count_vs_density_YYYYMMDD_HHMMSS.png`

## Entity Types Detected

The script identifies and categorizes entities into:

- **PER**: Person names (e.g., "John Smith", "Dr. Emily Brown")
- **ORG**: Organizations (e.g., "Nike", "Adidas", "Amazon")
- **LOC**: Locations (e.g., "New York", "California", "Paris")
- **MISC**: Miscellaneous entities that don't fit other categories

## Entity Density Analysis

Entity density is calculated as:
```
Entity Density = Number of Entities / Number of Tokens
```

The analysis provides:
- **Statistical measures**: Mean, median, standard deviation, percentiles
- **Distribution categories**:
  - Low density (< 0.01): Minimal entity presence
  - Medium density (0.01-0.05): Moderate entity presence
  - High density (≥ 0.05): High entity presence

## Performance Considerations

### For Large Datasets (2.5M+ records)

1. **Memory Usage**: The script processes data in batches to manage memory
2. **Processing Time**: Expect 2-4 hours for full dataset analysis
3. **Caching**: Results are cached for faster subsequent runs
4. **Hardware Requirements**: 
   - Minimum 8GB RAM
   - Multi-core CPU recommended
   - SSD storage for faster I/O

### Optimization Tips

1. **Test with Subset**: Set `sample_size = 10000` to test with first 10K records
2. **Adjust Batch Size**: Modify `batch_size` in `_process_entities_batch()` method
3. **CPU Threads**: Adjust `torch.set_num_threads()` based on your CPU cores

## Sample Output

### Main Report Excerpt
```
================================================================================
AMAZON FASHION DATASET - NAMED ENTITY RECOGNITION ANALYSIS
================================================================================
Generated on: 2024-01-15 14:30:25

DATASET INFORMATION
----------------------------------------
Total texts analyzed: 2,500,000
Total tokens: 45,678,901
Total entities found: 1,234,567
Average entities per text: 0.49

ENTITY STATISTICS
----------------------------------------
Average entity density: 0.0270
Risk level: low

Entities by type:
  PER: 456,789
  ORG: 345,678
  LOC: 234,567
  MISC: 197,533

ENTITY DENSITY ANALYSIS
----------------------------------------
Mean density: 0.0270
Median density: 0.0150
Standard deviation: 0.0450
Min density: 0.0000
Max density: 0.5000

Density percentiles:
  25th percentile: 0.0050
  50th percentile: 0.0150
  75th percentile: 0.0350
  90th percentile: 0.0650
  95th percentile: 0.0950
  99th percentile: 0.1850

Density distribution:
  Low density (< 0.01): 1,250,000 texts
  Medium density (0.01-0.05): 875,000 texts
  High density (≥ 0.05): 375,000 texts
```

### Top 200 Report Excerpt
```
================================================================================
TOP 200 TEXTS WITH HIGHEST ENTITY COUNTS
================================================================================
Generated on: 2024-01-15 14:30:25

  1. Entity Count: 15
     Entity Density: 0.2500
     Text: "Nike Air Max 90 shoes designed by John Smith in Portland, Oregon..."
     Entities: Nike (ORG), Air Max 90 (MISC), John Smith (PER), Portland (LOC), Oregon (LOC)
--------------------------------------------------------------------------------

  2. Entity Count: 12
     Entity Density: 0.2000
     Text: "Adidas Ultraboost running shoes from Germany, designed by Dr. Sarah Johnson..."
     Entities: Adidas (ORG), Ultraboost (MISC), Germany (LOC), Dr. Sarah Johnson (PER)
--------------------------------------------------------------------------------
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use a subset of data
2. **Slow Processing**: First run is slower due to model loading
3. **File Not Found**: Ensure `Amazon_Fashion.csv` is in the correct directory
4. **Column Not Found**: Check that the "text" column exists in your CSV

### Error Messages

- `"Text column 'text' not found"`: Verify column name in your CSV file
- `"Failed to load Flair NER model"`: Check internet connection for model download
- Memory errors: Reduce `sample_size` or `batch_size`

## Advanced Usage

### Custom Analysis

You can use the analyzer class directly for custom analysis:

```python
from amazon_fashion_ner_analysis import AmazonFashionNERAnalyzer

# Initialize analyzer
analyzer = AmazonFashionNERAnalyzer()

# Analyze with custom parameters
results = analyzer.analyze_dataset(
    csv_file='your_data.csv',
    text_column='your_text_column',
    sample_size=50000
)

# Generate custom reports
analyzer.generate_report(results, output_dir='./custom_reports')
```

### Batch Processing for Very Large Datasets

For extremely large datasets, you can process in chunks:

```python
# Process in chunks of 100K records
chunk_size = 100000
total_records = 2500000

for start_idx in range(0, total_records, chunk_size):
    end_idx = min(start_idx + chunk_size, total_records)
    # Process chunk from start_idx to end_idx
    # Save intermediate results
```

## Dependencies

- **pandas**: Data manipulation and CSV reading
- **numpy**: Numerical computations and statistics
- **torch**: PyTorch for deep learning (Flair dependency)
- **flair**: Flair NLP library for named entity recognition
- **tqdm**: Progress bars for long-running operations
- **matplotlib**: Plotting and visualization
- **seaborn**: Enhanced plotting and statistical visualizations

## License

This script is provided as-is for educational and research purposes. 