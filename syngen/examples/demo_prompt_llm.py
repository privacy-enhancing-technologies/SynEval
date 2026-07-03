"""
Demo script for PromptLLMGenerator.

This script demonstrates how to use the Prompt-Conditioned LLM Generator
to create synthetic multimodal data that preserves cross-modal correlations.
"""
import pandas as pd
from generators.prompt_llm import PromptLLMGenerator

# Sample microloan data
sample_data = pd.DataFrame({
    'loan_purpose': [
        'To buy seeds and fertilizer for planting season',
        'Purchase livestock for dairy business',
        'Expand inventory for retail shop',
        'Buy sewing machine for tailoring business',
        'Invest in fishing equipment and nets'
    ],
    'sector': ['Agriculture', 'Agriculture', 'Retail', 'Services', 'Agriculture'],
    'loan_amount': [500, 1200, 800, 350, 900],
    'term_in_months': [12, 24, 18, 6, 15]
})

print("Sample Data:")
print(sample_data)
print("\n" + "="*80 + "\n")

# Initialize generator
# Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
generator = PromptLLMGenerator(
    provider='openai',  # or 'anthropic'
    model='gpt-4o-mini',
    temperature=0.8,
    batch_size=10
)

# Fit on real data
print("Fitting generator on sample data...")
generator.fit(
    sample_data,
    text_columns=['loan_purpose'],
    tabular_columns=['sector', 'loan_amount', 'term_in_months']
)
print("Fitting complete.\n")

# Generate synthetic data
print("Generating 5 synthetic samples...")
print("Note: Text generation is conditioned on tabular values")
print("This preserves cross-modal correlations\n")

synthetic_data = generator.generate(n_samples=5)

print("Synthetic Data:")
print(synthetic_data)
print("\n" + "="*80 + "\n")

# Show how the prompt conditioning works
print("Example prompt construction:")
example_row = pd.Series({
    'sector': 'Agriculture',
    'loan_amount': 500,
    'term_in_months': 14
})
example_prompt = generator._build_prompt(example_row, 'loan_purpose')
print(example_prompt)
print("\n" + "="*80 + "\n")

# Key difference from Stitching Fallacy
print("Key Features:")
print("1. Text is generated AFTER tabular data")
print("2. Each text generation is conditioned on its corresponding tabular row")
print("3. This maintains cross-modal correlations (unlike random stitching)")
print("4. For Agriculture + $500 + 12 months -> generates farming-related purpose")
print("5. For Retail + $1000 + 24 months -> generates retail-related purpose")
