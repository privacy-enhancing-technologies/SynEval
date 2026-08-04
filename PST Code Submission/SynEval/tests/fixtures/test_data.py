"""Test data fixtures for quantization tests."""
import pandas as pd


def get_sample_texts():
    """Sample text data with clear clusters."""
    return [
        # Cluster 1: Luxury/expensive
        "Luxurious penthouse with stunning panoramic city views and marble finishes",
        "Premium upscale apartment in exclusive gated neighborhood with concierge",
        "High-end luxury suite with top-tier amenities and designer furnishings",
        # Cluster 2: Budget/affordable
        "Affordable cozy room perfect for students on tight budget",
        "Budget-friendly basic shared accommodation hostel dorm",
        "Simple inexpensive cheap lodging for backpackers and travelers",
        # Cluster 3: Family/spacious
        "Spacious family home with large backyard playground for children",
        "Big comfortable house with multiple bedrooms suitable for large groups",
        "Roomy family-friendly apartment perfect for families with kids",
    ]


def get_sample_tabular_data():
    """Sample tabular data correlated with text clusters."""
    return pd.DataFrame({
        'price': [500, 450, 480, 50, 60, 55, 200, 210, 195],
        'accommodates': [2, 2, 2, 1, 1, 1, 6, 8, 6],
        'property_type': ['Apartment', 'Apartment', 'Apartment',
                         'Room', 'Room', 'Room',
                         'House', 'House', 'House']
    })


def get_sample_multimodal_df():
    """Complete multimodal dataset."""
    texts = get_sample_texts()
    df = get_sample_tabular_data()
    df['description'] = texts
    return df
