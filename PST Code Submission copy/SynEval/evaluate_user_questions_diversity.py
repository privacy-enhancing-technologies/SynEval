#!/usr/bin/env python3
"""
Evaluate diversity of USER QUESTION content in train_v3.jsonl
"""

import json
import re
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd
from evaluation.diversity import DiversityEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_user_questions(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract USER QUESTION content and completion labels from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        Tuple of (questions list, completions list)
    """
    questions = []
    completions = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                
                # Extract the USER QUESTION part using regex
                match = re.search(r'USER QUESTION:\s*(.+?)\s*ALLOWED\?:', prompt, re.DOTALL)
                if match:
                    question = match.group(1).strip()
                    questions.append(question)
                    completions.append(completion)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line: {e}")
                continue
    
    return questions, completions


def find_exact_duplicates(questions: List[str]) -> Dict:
    """
    Find exact duplicate questions.
    
    Args:
        questions: List of questions
        
    Returns:
        Dictionary with duplicate analysis
    """
    question_counts = Counter(questions)
    duplicates = {q: count for q, count in question_counts.items() if count > 1}
    
    return {
        'total_questions': len(questions),
        'unique_questions': len(question_counts),
        'duplicate_count': len(duplicates),
        'duplicates': duplicates,
        'uniqueness_ratio': len(question_counts) / len(questions) if questions else 0
    }


def analyze_question_patterns(questions: List[str], completions: List[str]) -> Dict:
    """
    Analyze patterns in questions and their relationship to completions.
    
    Args:
        questions: List of questions
        completions: List of completion labels
        
    Returns:
        Dictionary with pattern analysis
    """
    # Create dataframe for analysis
    df = pd.DataFrame({
        'question': questions,
        'completion': completions
    })
    
    # Analyze by completion type
    permitted = df[df['completion'] == 'permitted']
    restricted = df[df['completion'] == 'restricted']
    
    # Character length statistics
    df['length'] = df['question'].str.len()
    
    # Word count statistics
    df['word_count'] = df['question'].str.split().str.len()
    
    # Check for questions that differ only by punctuation or case
    df['normalized'] = df['question'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    normalized_dupes = df.groupby('normalized').size()
    near_duplicates = normalized_dupes[normalized_dupes > 1]
    
    return {
        'by_completion': {
            'permitted': len(permitted),
            'restricted': len(restricted),
            'permitted_unique': permitted['question'].nunique(),
            'restricted_unique': restricted['question'].nunique()
        },
        'length_stats': {
            'mean': df['length'].mean(),
            'median': df['length'].median(),
            'min': df['length'].min(),
            'max': df['length'].max(),
            'std': df['length'].std()
        },
        'word_count_stats': {
            'mean': df['word_count'].mean(),
            'median': df['word_count'].median(),
            'min': df['word_count'].min(),
            'max': df['word_count'].max(),
            'std': df['word_count'].std()
        },
        'near_duplicates': {
            'count': len(near_duplicates),
            'examples': near_duplicates.head(10).to_dict()
        }
    }


def create_metadata_for_diversity(questions: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Create synthetic and original dataframes with proper metadata for DiversityEvaluator.
    
    Args:
        questions: List of questions
        
    Returns:
        Tuple of (synthetic_df, original_df, metadata)
    """
    # For diversity analysis, we'll treat all questions as "original" 
    # and create a synthetic version by shuffling to compare
    df = pd.DataFrame({
        'question_text': questions,
        'question_id': range(len(questions))
    })
    
    # Create metadata structure expected by DiversityEvaluator
    metadata = {
        'columns': {
            'question_text': {
                'sdtype': 'text'
            },
            'question_id': {
                'sdtype': 'numerical'
            }
        }
    }
    
    # For this analysis, we'll use the same data as both original and synthetic
    # to focus on the diversity metrics of the questions themselves
    return df, df, metadata


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("USER QUESTION Diversity Evaluation")
    print("=" * 80)
    print()
    
    # Load data
    jsonl_path = 'train_v3.jsonl'
    print(f"Loading data from {jsonl_path}...")
    questions, completions = extract_user_questions(jsonl_path)
    print(f"Extracted {len(questions)} questions")
    print()
    
    # Find exact duplicates
    print("=" * 80)
    print("EXACT DUPLICATE ANALYSIS")
    print("=" * 80)
    duplicate_analysis = find_exact_duplicates(questions)
    print(f"Total questions: {duplicate_analysis['total_questions']}")
    print(f"Unique questions: {duplicate_analysis['unique_questions']}")
    print(f"Duplicate questions: {duplicate_analysis['duplicate_count']}")
    print(f"Uniqueness ratio: {duplicate_analysis['uniqueness_ratio']:.2%}")
    print()
    
    if duplicate_analysis['duplicates']:
        print("Top 20 duplicated questions:")
        sorted_dupes = sorted(duplicate_analysis['duplicates'].items(), 
                            key=lambda x: x[1], reverse=True)[:20]
        for question, count in sorted_dupes:
            print(f"  [{count}x] {question[:100]}{'...' if len(question) > 100 else ''}")
        print()
    
    # Pattern analysis
    print("=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    pattern_analysis = analyze_question_patterns(questions, completions)
    
    print("\nDistribution by completion type:")
    print(f"  Permitted: {pattern_analysis['by_completion']['permitted']} "
          f"({pattern_analysis['by_completion']['permitted_unique']} unique)")
    print(f"  Restricted: {pattern_analysis['by_completion']['restricted']} "
          f"({pattern_analysis['by_completion']['restricted_unique']} unique)")
    
    print("\nQuestion length statistics (characters):")
    for stat, value in pattern_analysis['length_stats'].items():
        print(f"  {stat.capitalize()}: {value:.1f}")
    
    print("\nQuestion word count statistics:")
    for stat, value in pattern_analysis['word_count_stats'].items():
        print(f"  {stat.capitalize()}: {value:.1f}")
    
    print(f"\nNear-duplicates (differ only in punctuation/case): "
          f"{pattern_analysis['near_duplicates']['count']}")
    if pattern_analysis['near_duplicates']['examples']:
        print("  Examples:")
        for norm, count in list(pattern_analysis['near_duplicates']['examples'].items())[:5]:
            print(f"    [{count}x] {norm[:80]}...")
    print()
    
    # Advanced diversity metrics using existing framework
    print("=" * 80)
    print("TEXT DIVERSITY METRICS")
    print("=" * 80)
    print("Preparing data for diversity analysis...")
    
    synthetic_df, original_df, metadata = create_metadata_for_diversity(questions)
    
    try:
        print("Initializing DiversityEvaluator...")
        evaluator = DiversityEvaluator(
            synthetic_data=synthetic_df,
            original_data=original_df,
            metadata=metadata,
            selected_metrics=['text_diversity']
        )
        
        print("Running diversity evaluation (this may take a while)...")
        results = evaluator.evaluate()
        
        if 'text_diversity' in results:
            text_div = results['text_diversity']
            
            # Display results for the question_text column
            if 'synthetic' in text_div and 'question_text' in text_div['synthetic']:
                syn_metrics = text_div['synthetic']['question_text']
                
                print("\nLexical Diversity (n-gram analysis):")
                if 'lexical_diversity' in syn_metrics:
                    for ngram_type, metrics in syn_metrics['lexical_diversity'].items():
                        if isinstance(metrics, dict) and 'unique_ratio' in metrics:
                            print(f"  {ngram_type}:")
                            print(f"    Total: {metrics.get('total', 0)}")
                            print(f"    Unique: {metrics.get('unique', 0)}")
                            print(f"    Unique ratio: {metrics.get('unique_ratio', 0):.4f}")
                            print(f"    Entropy: {metrics.get('entropy', 0):.4f}")
                            print(f"    Normalized entropy: {metrics.get('normalized_entropy', 0):.4f}")
                
                print("\nSemantic Diversity (word embedding analysis):")
                if 'semantic_diversity' in syn_metrics:
                    sem_metrics = syn_metrics['semantic_diversity']
                    print(f"  Total MST weight: {sem_metrics.get('total_mst_weight', 0):.4f}")
                    print(f"  Average edge weight: {sem_metrics.get('average_edge_weight', 0):.4f}")
                    print(f"  Distinct nodes: {sem_metrics.get('distinct_nodes', 0)}")
                    print(f"  Distinct ratio: {sem_metrics.get('distinct_ratio', 0):.4f}")
                    print(f"  Sample size: {sem_metrics.get('sample_size', 0)}")
                
                print("\nSentiment Diversity:")
                if 'sentiment_diversity' in syn_metrics:
                    sent_metrics = syn_metrics['sentiment_diversity']
                    if 'sentiment_distribution' in sent_metrics:
                        print("  Sentiment distribution:")
                        for sentiment, ratio in sent_metrics['sentiment_distribution'].items():
                            print(f"    {sentiment}: {ratio:.2%}")
                    print(f"  Sample size: {sent_metrics.get('sample_size', 0)}")
            
            # Display coverage metrics
            if 'coverage' in text_div and 'question_text' in text_div['coverage']:
                print(f"\nVocabulary coverage: {text_div['coverage']['question_text']:.2f}%")
                
    except Exception as e:
        print(f"Error running diversity evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("DIVERSITY SUMMARY")
    print("=" * 80)
    print(f"Overall uniqueness: {duplicate_analysis['uniqueness_ratio']:.2%}")
    print(f"Exact duplicates found: {duplicate_analysis['duplicate_count']}")
    print(f"Near-duplicates found: {pattern_analysis['near_duplicates']['count']}")
    
    # Diversity assessment
    uniqueness = duplicate_analysis['uniqueness_ratio']
    if uniqueness >= 0.95:
        assessment = "EXCELLENT - Very high diversity"
    elif uniqueness >= 0.85:
        assessment = "GOOD - High diversity with some duplicates"
    elif uniqueness >= 0.70:
        assessment = "MODERATE - Noticeable duplicates present"
    else:
        assessment = "LOW - Significant duplicate content"
    
    print(f"\nDiversity Assessment: {assessment}")
    print("=" * 80)


if __name__ == "__main__":
    main()
