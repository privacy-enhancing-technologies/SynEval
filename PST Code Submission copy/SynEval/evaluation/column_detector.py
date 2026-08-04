"""Auto-detection of text vs tabular columns."""
import pandas as pd
from typing import Dict, List


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Auto-detect text vs tabular columns.

    Text heuristics:
        - dtype = object/string
        - Average length > 40 chars
        - High cardinality (>50% unique values)

    Tabular heuristics:
        - Numeric dtypes
        - Low-cardinality categorical (<20 unique values)

    Args:
        df: DataFrame to analyze

    Returns:
        Dict with 'text' and 'tabular' column lists
    """
    text_columns = []
    tabular_columns = []

    # Skip ID columns
    id_keywords = ['id', 'index', 'key', 'uuid']

    for col in df.columns:
        # Skip ID columns
        if any(keyword in col.lower() for keyword in id_keywords):
            continue

        # Text detection
        if df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            cardinality_ratio = df[col].nunique() / len(df)

            if avg_length > 40 and cardinality_ratio > 0.5:
                text_columns.append(col)
            elif df[col].nunique() < 20:
                # Low-cardinality categorical
                tabular_columns.append(col)

        # Numeric columns are tabular
        elif df[col].dtype in ['int64', 'float64']:
            tabular_columns.append(col)

    return {
        "text": text_columns,
        "tabular": tabular_columns
    }
