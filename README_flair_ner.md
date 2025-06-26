# Flair Named Entity Recognition (NER) Processor

This standalone script extracts and analyzes named entities from text data using the Flair NLP library. It provides both batch processing capabilities and single text analysis with caching for improved performance.

## Features

- **Batch Processing**: Efficiently process large datasets with progress tracking
- **Entity Filtering**: Intelligent filtering to remove noise and irrelevant entities
- **Caching**: Automatic caching of results to avoid reprocessing
- **Multiple Entity Types**: Detects Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC) entities
- **Statistics**: Provides comprehensive statistics about entity density and distribution
- **Single Text Processing**: Extract entities from individual text strings

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements_flair_ner.txt
```

Or install manually:

```bash
pip install pandas numpy torch flair tqdm
```

### 2. Verify Installation

The script will automatically download the Flair NER model on first run. This may take a few minutes depending on your internet connection.

## Usage

### Running the Script

Simply run the script to process the included sample dataset:

```bash
python flair_ner.py
```

### Expected Output

The script will:
1. Load the Flair NER model (first time only)
2. Process the sample dataset
3. Display comprehensive results including:
   - Total entities found
   - Entity counts by type
   - Sample entities for each type
   - Single text processing example
4. Save results to `ner_results.json`

### Sample Output

```
==================================================
NAMED ENTITY RECOGNITION RESULTS
==================================================
Total entities found: 25
Total tokens: 89
Average entity density: 0.2809
Risk level: high

Entities by type:
  PER: 10
  ORG: 8
  LOC: 7

Sample entities by type:
  PER: ['John Smith', 'Sarah Johnson', 'Tim Cook', 'Emily Brown', 'Mark Zuckerberg']
  ORG: ['Microsoft', 'Stanford University', 'Apple', 'Johns Hopkins Hospital', 'Facebook']
  LOC: ['Seattle', 'Washington', 'California', 'Baltimore', 'Cambridge']

==================================================
SINGLE TEXT PROCESSING EXAMPLE
==================================================
Text: John Smith works at Microsoft in Seattle, Washington.
Entities found:
  - John Smith (PER)
  - Microsoft (ORG)
  - Seattle (LOC)
  - Washington (LOC)
```

## Using Your Own Data

### Method 1: Modify the Script

Edit the `create_sample_dataset()` function in `flair_ner.py`:

```python
def create_sample_dataset():
    """
    Create your own dataset for testing the NER functionality.
    """
    your_texts = [
        "Your first text here.",
        "Your second text here.",
        # Add more texts...
    ]
    
    return pd.DataFrame({
        'id': range(1, len(your_texts) + 1),
        'text': your_texts
    })
```

### Method 2: Load from CSV

Replace the dataset creation in the `main()` function:

```python
# Load your own dataset
df = pd.read_csv('your_data.csv')
texts = df['text_column'].tolist()  # Replace 'text_column' with your column name
```

### Method 3: Use the Class Directly

```python
from flair_ner import FlairNERProcessor

# Initialize processor
ner_processor = FlairNERProcessor()

# Process your texts
your_texts = ["Text 1", "Text 2", "Text 3"]
results = ner_processor.analyze_named_entities(your_texts, "your_dataset_name")

# Process single text
entities = ner_processor.extract_entities_from_text("John Smith works at Microsoft.")
```

## Configuration

### Model Selection

You can use different Flair models by changing the model name:

```python
# Use a smaller, faster model
ner_processor = FlairNERProcessor('flair/ner-english-fast')

# Use a larger, more accurate model (default)
ner_processor = FlairNERProcessor('flair/ner-english-large')
```

### Batch Size

Adjust the batch size in the `_process_entities_batch` method for your hardware:

```python
batch_size = 32  # Increase for faster GPUs, decrease for slower CPUs
```

### CPU Threads

Modify the number of CPU threads in the `get_flair_model` function:

```python
torch.set_num_threads(8)  # Adjust based on your CPU cores
```

## Output Files

- `ner_results.json`: Complete analysis results in JSON format
- `cache/flair_ner_results.json`: Cached results for faster subsequent runs

## Entity Types

The script detects and categorizes entities into:

- **PER**: Person names (e.g., "John Smith", "Dr. Emily Brown")
- **ORG**: Organizations (e.g., "Microsoft", "Stanford University")
- **LOC**: Locations (e.g., "Seattle", "California")
- **MISC**: Miscellaneous entities that don't fit other categories

## Filtering Rules

The script applies several filtering rules to improve entity quality:

- Removes entities shorter than 2 characters or longer than 50 characters
- Excludes purely numeric entities
- Filters out URLs and email addresses
- Removes entities with incomplete words (ending with '-' or '&')
- Excludes all-uppercase entities (likely labels or categories)

## Performance Tips

1. **Use Caching**: The script automatically caches results. Subsequent runs with the same dataset will be much faster.

2. **Batch Processing**: For large datasets, the script processes texts in batches to optimize memory usage.

3. **Model Loading**: The Flair model is loaded once and cached in memory for subsequent uses.

4. **CPU Optimization**: The script is configured for CPU usage. For GPU acceleration, modify the device settings in the code.

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Ensure you have a stable internet connection for the first run.

2. **Memory Issues**: Reduce the batch size if you encounter memory errors.

3. **Slow Performance**: The first run will be slower due to model loading. Subsequent runs will be faster.

4. **Import Errors**: Make sure all dependencies are installed correctly using the requirements file.

### Error Messages

- `"Failed to load Flair NER model"`: Check your internet connection and try again
- `"Cache file is corrupted"`: The script will automatically create a new cache file
- Import errors: Install missing dependencies using `pip install -r requirements_flair_ner.txt`

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- torch: PyTorch for deep learning
- flair: Flair NLP library for named entity recognition
- tqdm: Progress bars for batch processing
- pathlib: Path manipulation (included in Python 3.4+)

## License

This script is provided as-is for educational and research purposes. 