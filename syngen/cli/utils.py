"""Utility functions for CLI."""
import logging
import sys
from typing import List, Optional
import pandas as pd


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_columns(df: pd.DataFrame, text_columns: List[str], tabular_columns: List[str]) -> None:
    """
    Validate that specified columns exist in the DataFrame.

    Args:
        df: Input DataFrame
        text_columns: List of text column names
        tabular_columns: List of tabular column names

    Raises:
        ValueError: If any column is not found in the DataFrame
    """
    all_columns = text_columns + tabular_columns
    missing = [col for col in all_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Columns not found in input data: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Check for duplicates
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("Duplicate columns found in text_columns and tabular_columns")


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        input_path: Path to input CSV file

    Returns:
        DataFrame containing the data

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is not supported
    """
    if not input_path.endswith('.csv'):
        raise ValueError("Only CSV files are currently supported")

    try:
        df = pd.read_csv(input_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path to output CSV file
    """
    df.to_csv(output_path, index=False)


def print_statistics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                    text_columns: List[str], tabular_columns: List[str]) -> None:
    """
    Print statistics about generated data.

    Args:
        real_df: Original DataFrame
        synthetic_df: Generated DataFrame
        text_columns: List of text column names
        tabular_columns: List of tabular column names
    """
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Real samples: {len(real_df)}")
    print(f"Synthetic samples: {len(synthetic_df)}")
    print(f"Text columns: {text_columns}")
    print(f"Tabular columns: {tabular_columns}")

    # Print sample statistics for tabular columns
    if tabular_columns:
        print("\nTabular Column Statistics:")
        print("-" * 60)
        for col in tabular_columns:
            if col in synthetic_df.columns:
                print(f"\n{col}:")
                if pd.api.types.is_numeric_dtype(synthetic_df[col]):
                    print(f"  Real - Mean: {real_df[col].mean():.2f}, Std: {real_df[col].std():.2f}")
                    print(f"  Synthetic - Mean: {synthetic_df[col].mean():.2f}, Std: {synthetic_df[col].std():.2f}")
                else:
                    print(f"  Real unique values: {real_df[col].nunique()}")
                    print(f"  Synthetic unique values: {synthetic_df[col].nunique()}")

    # Print sample text lengths
    if text_columns:
        print("\nText Column Statistics:")
        print("-" * 60)
        for col in text_columns:
            if col in synthetic_df.columns:
                real_lengths = real_df[col].astype(str).str.len()
                synth_lengths = synthetic_df[col].astype(str).str.len()
                print(f"\n{col}:")
                print(f"  Real - Mean length: {real_lengths.mean():.1f}, Std: {real_lengths.std():.1f}")
                print(f"  Synthetic - Mean length: {synth_lengths.mean():.1f}, Std: {synth_lengths.std():.1f}")

    print("\n" + "=" * 60)
