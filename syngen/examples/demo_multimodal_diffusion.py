"""
Demo script for MultimodalDiffusionGenerator.

This script demonstrates how to use the Multimodal Diffusion Generator
to create synthetic multimodal data that preserves cross-modal correlations
through joint latent space modeling and diffusion processes.
"""
import pandas as pd
import numpy as np
from generators.multimodal_diffusion import MultimodalDiffusionGenerator

# Sample microloan data
sample_data = pd.DataFrame({
    'loan_purpose': [
        'To buy seeds and fertilizer for planting season',
        'Purchase livestock for dairy business',
        'Expand inventory for retail shop',
        'Buy sewing machine for tailoring business',
        'Invest in fishing equipment and nets',
        'Renovate small restaurant kitchen',
        'Purchase materials for construction business',
        'Buy tools for carpentry workshop'
    ],
    'sector': ['Agriculture', 'Agriculture', 'Retail', 'Services', 'Agriculture', 'Food', 'Construction', 'Services'],
    'loan_amount': [500, 1200, 800, 350, 900, 1500, 2000, 600],
    'term_in_months': [12, 24, 18, 6, 15, 36, 48, 12],
    'interest_rate': [12.5, 10.0, 11.5, 15.0, 12.0, 9.5, 8.5, 13.0]
})

print("Sample Data:")
print(sample_data)
print("\n" + "="*80 + "\n")

# Initialize generator
print("Initializing Multimodal Diffusion Generator...")
generator = MultimodalDiffusionGenerator(
    text_encoder_model='all-MiniLM-L6-v2',  # Fast sentence transformer
    latent_dim=64,                           # Joint latent space dimension
    hidden_dim=128,                          # Hidden layer size
    n_diffusion_steps=20,                    # Number of diffusion steps (reduced for demo)
    n_epochs=50,                             # Training epochs (reduced for demo)
    batch_size=4,                            # Batch size
    learning_rate=1e-3,                      # Learning rate
    random_seed=42                           # For reproducibility
)
print("Generator initialized.\n")

# Fit on real data
print("Fitting generator on sample data...")
print("This trains:")
print("  1. Text encoder (sentence-transformers)")
print("  2. Tabular encoder (MLP)")
print("  3. Diffusion denoising network")
print("  4. Text/tabular decoders")
print()

generator.fit(
    sample_data,
    text_columns=['loan_purpose'],
    tabular_columns=['sector', 'loan_amount', 'term_in_months', 'interest_rate']
)
print("Fitting complete.\n")

# Generate synthetic data
print("Generating 5 synthetic samples...")
print("Process:")
print("  1. Start with random noise in joint latent space")
print("  2. Iteratively denoise using trained diffusion model")
print("  3. Decode to text and tabular features")
print("  4. Cross-modal correlations preserved through joint modeling")
print()

synthetic_data = generator.generate(n_samples=5)

print("Synthetic Data:")
print(synthetic_data)
print("\n" + "="*80 + "\n")

# Demonstrate correlation preservation
print("Correlation Preservation Analysis:")
print()
print("Original Data - Sector Distribution:")
print(sample_data['sector'].value_counts())
print()
print("Synthetic Data - Sector Distribution:")
print(synthetic_data['sector'].value_counts())
print()

print("Original Data - Average Loan Amount by Sector:")
sector_avg_original = sample_data.groupby('sector')['loan_amount'].mean()
print(sector_avg_original)
print()

print("Synthetic Data - Average Loan Amount by Sector:")
sector_avg_synthetic = synthetic_data.groupby('sector')['loan_amount'].mean()
print(sector_avg_synthetic)
print()

# Show how text correlates with tabular features
print("Example Text-Tabular Correlations:")
print()
for idx, row in synthetic_data.iterrows():
    print(f"Sample {idx + 1}:")
    print(f"  Sector: {row['sector']}")
    print(f"  Loan Amount: ${row['loan_amount']:.2f}")
    print(f"  Purpose: {row['loan_purpose']}")
    print()

print("="*80)
print()

# Key advantages over other methods
print("Key Advantages of Multimodal Diffusion:")
print()
print("1. Joint Modeling:")
print("   - Text and tabular data are encoded into a shared latent space")
print("   - Diffusion process operates on joint representation")
print("   - Correlations are naturally preserved")
print()
print("2. Flexible Generation:")
print("   - Can control generation through diffusion process")
print("   - Gradual denoising allows for quality control")
print("   - Can potentially interpolate between samples")
print()
print("3. No LLM API Needed:")
print("   - Uses pre-trained sentence transformers for encoding")
print("   - Text decoding via nearest neighbor (no API calls)")
print("   - Faster and more cost-effective than LLM-based methods")
print()
print("4. Better Correlation Preservation:")
print("   - Compared to Stitching: No random pairing")
print("   - Compared to Prompt LLM: Joint training ensures consistency")
print("   - Diffusion captures complex multi-modal dependencies")
print()

print("="*80)
print()

# Save and load demonstration
print("Save/Load Demonstration:")
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
    temp_path = f.name

try:
    print(f"Saving model to {temp_path}...")
    generator.save(temp_path)
    print("Model saved successfully.")
    print()

    print("Loading model from file...")
    loaded_generator = MultimodalDiffusionGenerator()
    loaded_generator.load(temp_path)
    print("Model loaded successfully.")
    print()

    print("Generating with loaded model...")
    loaded_synthetic = loaded_generator.generate(n_samples=2)
    print("Generated data:")
    print(loaded_synthetic)
    print()

finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"Cleaned up temporary file: {temp_path}")

print()
print("="*80)
print("Demo complete!")
