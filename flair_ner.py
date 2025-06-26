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

# Suppress warnings
warnings.filterwarnings('ignore')

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()

def get_flair_model(model_name: str = 'flair/ner-english-large') -> SequenceTagger:
    """
    Get or load Flair model with caching.
    """
    with _model_lock:
        if model_name not in _model_cache:
            # Set number of threads for CPU
            torch.set_num_threads(4)  # Adjust based on your CPU
            _model_cache[model_name] = SequenceTagger.load(model_name)
            # Disable gradient computation for inference
            _model_cache[model_name].eval()
            # Use CPU
            _model_cache[model_name] = _model_cache[model_name].to('cpu')
        return _model_cache[model_name]

class FlairNERProcessor:
    def __init__(self, model_name: str = 'flair/ner-english-large'):
        """
        Initialize the Flair NER processor.
        
        Args:
            model_name: Name of the Flair model to use
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Initialize cache directory and file
        self.cache_dir = Path('./cache')
        self.cache_file = self.cache_dir / 'flair_ner_results.json'
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

    def _process_entities_batch(self, texts: List[str]) -> List[Tuple]:
        """
        Process a batch of texts to extract named entities using Flair with optimized batch processing.
        """
        results = []
        batch_size = 16  # Smaller batch size for CPU
        
        # Process texts in batches with progress bar
        with tqdm(total=len(texts), desc="Processing entities", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                sentences = [Sentence(text) for text in batch_texts]
                
                # Run NER on the batch
                self.ner_tagger.predict(sentences)
                
                # Process results
                for sentence in sentences:
                    valid_entities = []
                    for entity in sentence.get_spans('ner'):
                        # Skip if entity is too short or too long
                        if len(entity.text) < 2 or len(entity.text) > 50:
                            continue
                            
                        # Skip if entity is just a number or special character
                        if entity.text.isdigit() or not any(c.isalnum() for c in entity.text):
                            continue
                            
                        # Skip if entity contains URLs or email addresses
                        if 'http' in entity.text.lower() or '@' in entity.text or '.' in entity.text.split()[-1]:
                            continue
                        
                        # Skip if entity contains incomplete words or phrases
                        if any(word.endswith('-') or word.endswith('&') for word in entity.text.split()):
                            continue
                        
                        # Skip if entity is all uppercase (likely a label or category)
                        if entity.text.isupper() and len(entity.text) > 1:
                            continue
                        
                        # Map Flair labels to our categories
                        label = entity.tag
                        if label.startswith('B-') or label.startswith('I-'):
                            label = label[2:]  # Remove B- or I- prefix
                        
                        if label in {'PER', 'ORG', 'LOC', 'MISC'}:
                            valid_entities.append((entity.text, label))
                    
                    # Count tokens (simple word count)
                    num_tokens = len(sentence.text.split())
                    
                    results.append((valid_entities, len(valid_entities), num_tokens))
                
                pbar.update(len(batch_texts))
        
        return results

    def analyze_named_entities(self, texts: List[str], dataset_name: str = "default") -> Dict:
        """
        Analyze named entities in text data.
        
        Args:
            texts: List of text strings to analyze
            dataset_name: Name of the dataset for caching purposes
            
        Returns:
            Dictionary containing entity analysis results
        """
        # Check cache first
        cache_key = f"entities_{dataset_name}"
        if cache_key in self.cache:
            self.logger.info(f"Using cached results for dataset: {dataset_name}")
            return self.cache[cache_key]
        
        self.logger.info(f"Processing entities for dataset: {dataset_name}...")
        
        # Process entities
        results = self._process_entities_batch(texts)
        
        # Extract all entities
        all_entities = set()
        for entities, _, _ in results:
            all_entities.update(entities)
        
        # Group entities by type
        entities_by_type = {}
        for entity, type_ in all_entities:
            if type_ not in entities_by_type:
                entities_by_type[type_] = []
            entities_by_type[type_].append(entity)
        
        # Calculate statistics
        total_entities = len(all_entities)
        total_tokens = sum(num_tokens for _, _, num_tokens in results)
        avg_entity_density = total_entities / total_tokens if total_tokens > 0 else 0
        
        # Count entities by type
        entity_counts = {}
        for entity_type, entities in entities_by_type.items():
            entity_counts[entity_type] = len(entities)
        
        analysis_results = {
            'total_entities': total_entities,
            'total_tokens': total_tokens,
            'avg_entity_density': avg_entity_density,
            'entity_counts_by_type': entity_counts,
            'entities_by_type': entities_by_type,
            'risk_level': 'high' if avg_entity_density > 0.1 else 'low'
        }
        
        # Save to cache
        self.cache[cache_key] = analysis_results
        self._save_cache()
        
        return analysis_results

    def extract_entities_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from a single text.
        
        Args:
            text: Input text string
            
        Returns:
            List of tuples (entity_text, entity_type)
        """
        sentence = Sentence(text)
        self.ner_tagger.predict(sentence)
        
        entities = []
        for entity in sentence.get_spans('ner'):
            # Apply same filtering as batch processing
            if len(entity.text) < 2 or len(entity.text) > 50:
                continue
                
            if entity.text.isdigit() or not any(c.isalnum() for c in entity.text):
                continue
                
            if 'http' in entity.text.lower() or '@' in entity.text or '.' in entity.text.split()[-1]:
                continue
            
            if any(word.endswith('-') or word.endswith('&') for word in entity.text.split()):
                continue
            
            if entity.text.isupper() and len(entity.text) > 1:
                continue
            
            label = entity.tag
            if label.startswith('B-') or label.startswith('I-'):
                label = label[2:]
            
            if label in {'PER', 'ORG', 'LOC', 'MISC'}:
                entities.append((entity.text, label))
        
        return entities

def create_sample_dataset():
    """
    Create a sample dataset for testing the NER functionality.
    """
    sample_texts = [
        "John Smith works at Microsoft in Seattle, Washington.",
        "Sarah Johnson is a professor at Stanford University in California.",
        "The CEO of Apple, Tim Cook, announced new products yesterday.",
        "Dr. Emily Brown practices medicine at Johns Hopkins Hospital in Baltimore.",
        "Mark Zuckerberg founded Facebook in Cambridge, Massachusetts.",
        "The president of the United States, Joe Biden, gave a speech in Washington D.C.",
        "Professor David Wilson teaches at MIT in Boston.",
        "Lisa Anderson is a lawyer at Goldman Sachs in New York.",
        "The mayor of Chicago, Lori Lightfoot, attended the conference.",
        "Dr. Michael Chen works at Google headquarters in Mountain View, California."
    ]
    
    return pd.DataFrame({
        'id': range(1, len(sample_texts) + 1),
        'text': sample_texts
    })

def main():
    """
    Main function to demonstrate Flair NER usage.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Create sample dataset
        logger.info("Creating sample dataset...")
        df = create_sample_dataset()
        logger.info(f"Created dataset with {len(df)} samples")
        
        # Initialize NER processor
        logger.info("Initializing Flair NER processor...")
        ner_processor = FlairNERProcessor()
        
        # Analyze named entities
        logger.info("Analyzing named entities...")
        texts = df['text'].tolist()
        results = ner_processor.analyze_named_entities(texts, "sample_dataset")
        
        # Print results
        print("\n" + "="*50)
        print("NAMED ENTITY RECOGNITION RESULTS")
        print("="*50)
        print(f"Total entities found: {results['total_entities']}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"Average entity density: {results['avg_entity_density']:.4f}")
        print(f"Risk level: {results['risk_level']}")
        
        print("\nEntities by type:")
        for entity_type, count in results['entity_counts_by_type'].items():
            print(f"  {entity_type}: {count}")
        
        print("\nSample entities by type:")
        for entity_type, entities in results['entities_by_type'].items():
            print(f"  {entity_type}: {entities[:5]}")  # Show first 5 entities
        
        # Demonstrate single text processing
        print("\n" + "="*50)
        print("SINGLE TEXT PROCESSING EXAMPLE")
        print("="*50)
        sample_text = "John Smith works at Microsoft in Seattle, Washington."
        entities = ner_processor.extract_entities_from_text(sample_text)
        print(f"Text: {sample_text}")
        print("Entities found:")
        for entity_text, entity_type in entities:
            print(f"  - {entity_text} ({entity_type})")
        
        # Save results to file
        output_file = "ner_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error running NER analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 