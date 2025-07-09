import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import torch
import threading
from functools import lru_cache
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
import json
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings('ignore')

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

def get_flair_model(model_name: str = 'flair/ner-english-fast') -> SequenceTagger:
    """
    Get or load Flair model with caching - using faster model.
    """
    with _model_lock:
        if model_name not in _model_cache:
            # Set number of threads for CPU
            torch.set_num_threads(mp.cpu_count())  # Use all CPU cores
            _model_cache[model_name] = SequenceTagger.load(model_name)
            # Disable gradient computation for inference
            _model_cache[model_name].eval()
            # Use CPU
            _model_cache[model_name] = _model_cache[model_name].to('cpu')
        return _model_cache[model_name]

class FastAmazonFashionNERAnalyzer:
    def __init__(self, model_name: str = 'flair/ner-english-fast'):
        """
        Initialize the fast Amazon Fashion NER analyzer.
        
        Args:
            model_name: Name of the Flair model to use (faster model by default)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Initialize cache directory and file
        self.cache_dir = Path('./cache')
        self.cache_file = self.cache_dir / 'amazon_fashion_ner_fast_results.json'
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load cache if exists
        self.cache = self._load_cache()
        
        # Initialize Flair NER model with caching
        try:
            self.logger.info("Loading Flair NER model (this may take a few minutes)...")
            self.ner_tagger = get_flair_model(model_name)
            self.logger.info("Loaded Flair NER model successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Flair NER model: {str(e)}")
            raise

    def _load_cache(self) -> Dict:
        """Load the detection cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.logger.warning("Cache file is corrupted. Creating new cache.")
                return {}
        return {}

    def _save_cache(self):
        """Save the current cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _process_text_batch(self, texts_batch: List[str]) -> List[Tuple]:
        """
        Process a batch of texts to extract named entities - optimized version.
        """
        results = []
        
        # Create sentences
        sentences = [Sentence(text) for text in texts_batch]
        
        # Run NER on the batch
        self.ner_tagger.predict(sentences)
        
        # Process results
        for sentence in sentences:
            valid_entities = []
            for entity in sentence.get_spans('ner'):
                # Simplified filtering for speed
                if len(entity.text) < 2 or len(entity.text) > 50:
                    continue
                    
                if entity.text.isdigit():
                    continue
                    
                # Map Flair labels to our categories
                label = entity.tag
                if label.startswith('B-') or label.startswith('I-'):
                    label = label[2:]
                
                if label in {'PER', 'ORG', 'LOC', 'MISC'}:
                    valid_entities.append((entity.text, label))
            
            # Count tokens (simple word count)
            num_tokens = len(sentence.text.split())
            
            results.append((valid_entities, len(valid_entities), num_tokens))
        
        return results

    def _process_entities_batch_optimized(self, texts: List[str]) -> List[Tuple]:
        """
        Process texts in larger batches with better optimization.
        """
        results = []
        batch_size = 64  # Larger batch size for better performance
        
        # Process texts in batches with progress bar
        with tqdm(total=len(texts), desc="Processing entities", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = self._process_text_batch(batch_texts)
                results.extend(batch_results)
                pbar.update(len(batch_texts))
        
        return results

    def analyze_dataset(self, csv_file: str, text_column: str = 'text', sample_size: Optional[int] = None) -> Dict:
        """
        Analyze named entities in the Amazon Fashion dataset - optimized version.
        """
        self.logger.info(f"Loading dataset from {csv_file}...")
        
        # Load dataset
        if sample_size:
            df = pd.read_csv(csv_file, nrows=sample_size)
            self.logger.info(f"Loaded {len(df)} samples from dataset")
        else:
            df = pd.read_csv(csv_file)
            self.logger.info(f"Loaded {len(df)} samples from dataset")
        
        # Check if text column exists
        if text_column not in df.columns:
            available_columns = list(df.columns)
            raise ValueError(f"Text column '{text_column}' not found. Available columns: {available_columns}")
        
        # Clean and prepare texts
        texts = df[text_column].astype(str).fillna('').tolist()
        
        # Remove empty texts and very short texts
        texts = [text for text in texts if len(text.strip()) > 10]  # Only texts with >10 chars
        self.logger.info(f"Processing {len(texts)} non-empty texts")
        
        # Check cache first
        cache_key = f"amazon_fashion_entities_fast_{len(texts)}"
        if cache_key in self.cache:
            self.logger.info("Using cached results...")
            return self.cache[cache_key]
        
        # Process entities
        self.logger.info("Processing entities...")
        results = self._process_entities_batch_optimized(texts)
        
        # Create detailed analysis
        analysis_results = self._create_detailed_analysis(results, texts, df)
        
        # Save to cache
        self.cache[cache_key] = analysis_results
        self._save_cache()
        
        return analysis_results

    def _create_detailed_analysis(self, results: List[Tuple], texts: List[str], df: pd.DataFrame) -> Dict:
        """
        Create detailed analysis from processing results.
        """
        # Extract all entities and create per-text analysis
        all_entities = set()
        text_analyses = []
        
        for i, (entities, num_entities, num_tokens) in enumerate(results):
            # Calculate entity density for this text
            entity_density = num_entities / num_tokens if num_tokens > 0 else 0
            
            text_analysis = {
                'text_index': i,
                'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                'full_text': texts[i],
                'entities': entities,
                'num_entities': num_entities,
                'num_tokens': num_tokens,
                'entity_density': entity_density,
                'entities_by_type': self._group_entities_by_type(entities)
            }
            text_analyses.append(text_analysis)
            
            # Collect all entities
            all_entities.update(entities)
        
        # Group entities by type
        entities_by_type = {}
        for entity, type_ in all_entities:
            if type_ not in entities_by_type:
                entities_by_type[type_] = []
            entities_by_type[type_].append(entity)
        
        # Calculate overall statistics
        total_entities = len(all_entities)
        total_tokens = sum(num_tokens for _, _, num_tokens in results)
        avg_entity_density = total_entities / total_tokens if total_tokens > 0 else 0
        
        # Entity density statistics
        entity_densities = [analysis['entity_density'] for analysis in text_analyses]
        density_stats = {
            'mean': np.mean(entity_densities),
            'median': np.median(entity_densities),
            'std': np.std(entity_densities),
            'min': np.min(entity_densities),
            'max': np.max(entity_densities),
            'percentiles': {
                '25': np.percentile(entity_densities, 25),
                '50': np.percentile(entity_densities, 50),
                '75': np.percentile(entity_densities, 75),
                '90': np.percentile(entity_densities, 90),
                '95': np.percentile(entity_densities, 95),
                '99': np.percentile(entity_densities, 99)
            }
        }
        
        # Count entities by type
        entity_counts = {}
        for entity_type, entities in entities_by_type.items():
            entity_counts[entity_type] = len(entities)
        
        # Find texts with highest entity counts
        text_analyses_sorted = sorted(text_analyses, key=lambda x: x['num_entities'], reverse=True)
        top_200_high_entity_texts = text_analyses_sorted[:200]
        
        return {
            'dataset_info': {
                'total_texts': len(texts),
                'total_tokens': total_tokens,
                'total_entities': total_entities,
                'avg_entities_per_text': total_entities / len(texts) if texts else 0
            },
            'entity_statistics': {
                'total_entities': total_entities,
                'avg_entity_density': avg_entity_density,
                'entity_counts_by_type': entity_counts,
                'risk_level': 'high' if avg_entity_density > 0.1 else 'low'
            },
            'entity_density_analysis': {
                'statistics': density_stats,
                'distribution': {
                    'low_density': len([d for d in entity_densities if d < 0.01]),
                    'medium_density': len([d for d in entity_densities if 0.01 <= d < 0.05]),
                    'high_density': len([d for d in entity_densities if d >= 0.05])
                }
            },
            'entities_by_type': entities_by_type,
            'top_200_high_entity_texts': top_200_high_entity_texts,
            'text_analyses': text_analyses
        }

    def _group_entities_by_type(self, entities: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Group entities by their type for a single text."""
        grouped = {}
        for entity, type_ in entities:
            if type_ not in grouped:
                grouped[type_] = []
            grouped[type_].append(entity)
        return grouped

    def generate_report(self, results: Dict, output_dir: str = './reports'):
        """
        Generate comprehensive report files.
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Main Analysis Report
        self._generate_main_report(results, output_dir, timestamp)
        
        # 2. Top 200 High Entity Texts Report
        self._generate_top_200_report(results, output_dir, timestamp)
        
        # 3. Entity Density Analysis Report
        self._generate_density_report(results, output_dir, timestamp)
        
        # 4. Visualizations
        self._generate_visualizations(results, output_dir, timestamp)
        
        self.logger.info(f"Reports generated in {output_dir}")

    def _generate_main_report(self, results: Dict, output_dir: str, timestamp: str):
        """Generate main analysis report."""
        report_file = Path(output_dir) / f"amazon_fashion_ner_fast_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AMAZON FASHION DATASET - FAST NER ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 40 + "\n")
            dataset_info = results['dataset_info']
            f.write(f"Total texts analyzed: {dataset_info['total_texts']:,}\n")
            f.write(f"Total tokens: {dataset_info['total_tokens']:,}\n")
            f.write(f"Total entities found: {dataset_info['total_entities']:,}\n")
            f.write(f"Average entities per text: {dataset_info['avg_entities_per_text']:.2f}\n\n")
            
            # Entity Statistics
            f.write("ENTITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            entity_stats = results['entity_statistics']
            f.write(f"Average entity density: {entity_stats['avg_entity_density']:.4f}\n")
            f.write(f"Risk level: {entity_stats['risk_level']}\n\n")
            
            f.write("Entities by type:\n")
            for entity_type, count in entity_stats['entity_counts_by_type'].items():
                f.write(f"  {entity_type}: {count:,}\n")
            f.write("\n")
            
            # Entity Density Analysis
            f.write("ENTITY DENSITY ANALYSIS\n")
            f.write("-" * 40 + "\n")
            density_analysis = results['entity_density_analysis']
            stats = density_analysis['statistics']
            f.write(f"Mean density: {stats['mean']:.4f}\n")
            f.write(f"Median density: {stats['median']:.4f}\n")
            f.write(f"Standard deviation: {stats['std']:.4f}\n")
            f.write(f"Min density: {stats['min']:.4f}\n")
            f.write(f"Max density: {stats['max']:.4f}\n\n")
            
            f.write("Density percentiles:\n")
            for percentile, value in stats['percentiles'].items():
                f.write(f"  {percentile}th percentile: {value:.4f}\n")
            f.write("\n")
            
            f.write("Density distribution:\n")
            dist = density_analysis['distribution']
            f.write(f"  Low density (< 0.01): {dist['low_density']:,} texts\n")
            f.write(f"  Medium density (0.01-0.05): {dist['medium_density']:,} texts\n")
            f.write(f"  High density (≥ 0.05): {dist['high_density']:,} texts\n\n")
            
            # Sample entities by type
            f.write("SAMPLE ENTITIES BY TYPE\n")
            f.write("-" * 40 + "\n")
            for entity_type, entities in results['entities_by_type'].items():
                f.write(f"{entity_type} (showing first 20):\n")
                sample_entities = entities[:20]
                for entity in sample_entities:
                    f.write(f"  - {entity}\n")
                f.write("\n")

    def _generate_top_200_report(self, results: Dict, output_dir: str, timestamp: str):
        """Generate report for top 200 texts with highest entity counts."""
        report_file = Path(output_dir) / f"top_200_high_entity_texts_fast_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TOP 200 TEXTS WITH HIGHEST ENTITY COUNTS (FAST VERSION)\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, text_analysis in enumerate(results['top_200_high_entity_texts'], 1):
                f.write(f"{i:3d}. Entity Count: {text_analysis['num_entities']}\n")
                f.write(f"    Entity Density: {text_analysis['entity_density']:.4f}\n")
                f.write(f"    Complete Text: {text_analysis['full_text']}\n")
                f.write(f"    Entities: {', '.join([f'{e[0]} ({e[1]})' for e in text_analysis['entities']])}\n")
                f.write("-" * 80 + "\n\n")

    def _generate_density_report(self, results: Dict, output_dir: str, timestamp: str):
        """Generate detailed entity density analysis report."""
        report_file = Path(output_dir) / f"entity_density_analysis_fast_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ENTITY DENSITY ANALYSIS REPORT (FAST VERSION)\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            density_analysis = results['entity_density_analysis']
            stats = density_analysis['statistics']
            
            f.write("DENSITY STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean density: {stats['mean']:.6f}\n")
            f.write(f"Median density: {stats['median']:.6f}\n")
            f.write(f"Standard deviation: {stats['std']:.6f}\n")
            f.write(f"Minimum density: {stats['min']:.6f}\n")
            f.write(f"Maximum density: {stats['max']:.6f}\n\n")
            
            f.write("PERCENTILE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for percentile, value in stats['percentiles'].items():
                f.write(f"{percentile}th percentile: {value:.6f}\n")
            f.write("\n")
            
            f.write("DENSITY DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            dist = density_analysis['distribution']
            total_texts = sum(dist.values())
            f.write(f"Low density (< 0.01): {dist['low_density']:,} texts ({dist['low_density']/total_texts*100:.1f}%)\n")
            f.write(f"Medium density (0.01-0.05): {dist['medium_density']:,} texts ({dist['medium_density']/total_texts*100:.1f}%)\n")
            f.write(f"High density (≥ 0.05): {dist['high_density']:,} texts ({dist['high_density']/total_texts*100:.1f}%)\n\n")
            
            # Find texts with highest density
            text_analyses = results['text_analyses']
            sorted_by_density = sorted(text_analyses, key=lambda x: x['entity_density'], reverse=True)
            
            f.write("TOP 50 TEXTS BY ENTITY DENSITY\n")
            f.write("-" * 40 + "\n")
            for i, text_analysis in enumerate(sorted_by_density[:50], 1):
                f.write(f"{i:2d}. Density: {text_analysis['entity_density']:.4f} | ")
                f.write(f"Entities: {text_analysis['num_entities']} | ")
                f.write(f"Text: {text_analysis['text'][:100]}...\n")

    def _generate_visualizations(self, results: Dict, output_dir: str, timestamp: str):
        """Generate visualization plots."""
        try:
            # 1. Entity Type Distribution
            plt.figure(figsize=(10, 6))
            entity_counts = results['entity_statistics']['entity_counts_by_type']
            plt.bar(entity_counts.keys(), entity_counts.values())
            plt.title('Entity Distribution by Type (Fast Version)')
            plt.ylabel('Number of Entities')
            plt.xlabel('Entity Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'entity_distribution_fast_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Entity Density Histogram
            plt.figure(figsize=(12, 6))
            densities = [analysis['entity_density'] for analysis in results['text_analyses']]
            plt.hist(densities, bins=50, alpha=0.7, edgecolor='black')
            plt.title('Entity Density Distribution (Fast Version)')
            plt.xlabel('Entity Density')
            plt.ylabel('Number of Texts')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'entity_density_histogram_fast_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")

def main():
    """
    Main function to analyze Amazon Fashion dataset with optimized performance.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Configuration
        csv_file = 'real_10k.csv'
        text_column = 'text'
        sample_size = 10000  # Start with 10K for testing
        
        # Initialize analyzer
        logger.info("Initializing Fast Amazon Fashion NER Analyzer...")
        analyzer = FastAmazonFashionNERAnalyzer()
        
        # Analyze dataset
        logger.info("Starting analysis...")
        results = analyzer.analyze_dataset(csv_file, text_column, sample_size)
        
        # Generate reports
        logger.info("Generating reports...")
        analyzer.generate_report(results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 