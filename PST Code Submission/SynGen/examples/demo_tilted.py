"""
Demo: Tilted Data Generator - Adversarial Baseline

This script demonstrates the Tilted generator, which intentionally destroys
cross-modal correlations to serve as a negative control in experiments.

Key Concepts:
1. The Tilted generator samples real text and real tabular data
2. It randomly shuffles the pairings to break correlations
3. This creates a "worst-case" baseline for evaluation
4. Proper generators must outperform this baseline

Use Cases:
- Validate that evaluation metrics detect broken correlations
- Establish lower bound for synthetic data quality
- Negative control in comparative experiments
"""
import pandas as pd
import numpy as np
from generators.tilted import TiltedGenerator


def create_medical_dataset():
    """Create a sample medical dataset with clear correlations."""
    np.random.seed(42)

    specialties = ['Cardiology', 'Neurology', 'Orthopedics', 'Dermatology', 'Psychiatry']
    texts = [
        'Patient presents with chest pain and elevated cardiac enzymes',
        'MRI scan shows abnormal brain activity in frontal lobe',
        'X-ray reveals fracture in left femur, requires surgery',
        'Skin biopsy confirms melanoma diagnosis',
        'Patient reports severe anxiety and depression symptoms'
    ]

    n_samples = 50
    data = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(specialties))
        data.append({
            'transcription': texts[idx],
            'specialty': specialties[idx],
            'age': np.random.randint(25, 80),
            'severity': np.random.choice(['low', 'medium', 'high']),
            'insurance': np.random.choice(['private', 'medicare', 'medicaid'])
        })

    return pd.DataFrame(data)


def create_kiva_dataset():
    """Create a sample Kiva loans dataset with clear correlations."""
    np.random.seed(42)

    sectors = ['Agriculture', 'Retail', 'Education', 'Health', 'Transportation']
    uses = [
        'buy seeds and fertilizer for next harvest',
        'purchase inventory for my store',
        'pay school fees for my children',
        'buy medical supplies for clinic',
        'repair motorcycle for delivery business'
    ]

    n_samples = 50
    data = []
    for _ in range(n_samples):
        idx = np.random.randint(0, len(sectors))
        data.append({
            'use': uses[idx],
            'sector': sectors[idx],
            'loan_amount': np.random.randint(100, 5000),
            'country': np.random.choice(['Kenya', 'Philippines', 'Peru', 'Cambodia']),
            'gender': np.random.choice(['F', 'M'])
        })

    return pd.DataFrame(data)


def analyze_correlations(df, text_col, tabular_col):
    """
    Analyze text-tabular correlations by checking co-occurrence.

    Returns the proportion of samples where text and tabular values
    have the expected correlation (same index in our demo data).
    """
    # In our demo data, specialty[i] matches transcription[i]
    # We'll check this by seeing if keywords appear
    specialty_keywords = {
        'Cardiology': ['chest', 'cardiac', 'heart'],
        'Neurology': ['brain', 'MRI', 'neurological'],
        'Orthopedics': ['fracture', 'bone', 'surgery'],
        'Dermatology': ['skin', 'biopsy', 'melanoma'],
        'Psychiatry': ['anxiety', 'depression', 'mental']
    }

    sector_keywords = {
        'Agriculture': ['seeds', 'fertilizer', 'harvest'],
        'Retail': ['inventory', 'store', 'shop'],
        'Education': ['school', 'fees', 'children'],
        'Health': ['medical', 'clinic', 'supplies'],
        'Transportation': ['motorcycle', 'delivery', 'transport']
    }

    keywords_map = specialty_keywords if 'specialty' in df.columns else sector_keywords

    matches = 0
    for _, row in df.iterrows():
        category = row[tabular_col]
        text = str(row[text_col]).lower()

        # Check if any keyword for this category appears in text
        if category in keywords_map:
            if any(keyword.lower() in text for keyword in keywords_map[category]):
                matches += 1

    correlation = matches / len(df) if len(df) > 0 else 0
    return correlation


def demo_medical_data():
    """Demonstrate Tilted generator on medical data."""
    print("=" * 80)
    print("DEMO 1: Medical Dataset")
    print("=" * 80)

    # Create dataset
    real_df = create_medical_dataset()
    print(f"\nCreated medical dataset with {len(real_df)} samples")
    print("\nSample real data (showing strong correlations):")
    print(real_df[['transcription', 'specialty', 'age']].head(3))

    # Analyze real data correlations
    real_corr = analyze_correlations(real_df, 'transcription', 'specialty')
    print(f"\nReal data correlation: {real_corr:.2%}")
    print("(Proportion of samples where specialty matches transcription content)")

    # Create Tilted generator with random shuffling
    print("\n" + "-" * 80)
    print("Testing RANDOM shuffle strategy")
    print("-" * 80)

    gen_random = TiltedGenerator(shuffle_strategy="random", random_state=42)
    gen_random.fit(
        real_df,
        text_columns=['transcription'],
        tabular_columns=['specialty', 'age', 'severity', 'insurance']
    )

    synthetic_random = gen_random.generate(50)
    print("\nGenerated synthetic data (random shuffle):")
    print(synthetic_random[['transcription', 'specialty', 'age']].head(3))

    tilted_corr = analyze_correlations(synthetic_random, 'transcription', 'specialty')
    print(f"\nTilted data correlation: {tilted_corr:.2%}")
    print(f"Correlation drop: {(real_corr - tilted_corr):.2%}")

    # Show specific examples of broken correlations
    print("\n" + "-" * 80)
    print("Examples of BROKEN correlations in Tilted data:")
    print("-" * 80)
    for idx in range(min(5, len(synthetic_random))):
        row = synthetic_random.iloc[idx]
        print(f"\n{idx+1}. Specialty: {row['specialty']}")
        print(f"   Transcription: {row['transcription'][:70]}...")


def demo_kiva_data():
    """Demonstrate Tilted generator on Kiva loans data."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Kiva Loans Dataset")
    print("=" * 80)

    # Create dataset
    real_df = create_kiva_dataset()
    print(f"\nCreated Kiva dataset with {len(real_df)} samples")
    print("\nSample real data (showing strong correlations):")
    print(real_df[['use', 'sector', 'loan_amount']].head(3))

    # Analyze real data correlations
    real_corr = analyze_correlations(real_df, 'use', 'sector')
    print(f"\nReal data correlation: {real_corr:.2%}")

    # Test different shuffling strategies
    strategies = ['random', 'stratified', 'adversarial']

    for strategy in strategies:
        print("\n" + "-" * 80)
        print(f"Testing {strategy.upper()} shuffle strategy")
        print("-" * 80)

        if strategy in ['stratified', 'adversarial']:
            # Add binary label for stratified/adversarial
            real_df['label'] = (real_df['loan_amount'] > 2500).astype(int)

        gen = TiltedGenerator(shuffle_strategy=strategy, random_state=42)

        if strategy == 'random':
            gen.fit(
                real_df,
                text_columns=['use'],
                tabular_columns=['sector', 'loan_amount', 'country', 'gender']
            )
        else:
            gen.fit(
                real_df,
                text_columns=['use'],
                tabular_columns=['sector', 'loan_amount', 'country', 'gender'],
                target_column='label'
            )

        synthetic = gen.generate(50)
        print(f"\nGenerated {len(synthetic)} synthetic samples")
        print("\nSample tilted data:")
        print(synthetic[['use', 'sector', 'loan_amount']].head(3))

        tilted_corr = analyze_correlations(synthetic, 'use', 'sector')
        print(f"\nTilted correlation: {tilted_corr:.2%}")
        print(f"Correlation drop: {(real_corr - tilted_corr):.2%}")


def demo_comparison_with_real():
    """Compare real vs tilted data side-by-side."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Side-by-Side Comparison - Real vs Tilted")
    print("=" * 80)

    real_df = create_medical_dataset()

    gen = TiltedGenerator(shuffle_strategy="random", random_state=42)
    gen.fit(
        real_df,
        text_columns=['transcription'],
        tabular_columns=['specialty', 'age']
    )
    synthetic_df = gen.generate(10)

    print("\n" + "-" * 80)
    print("REAL DATA (Correlated)")
    print("-" * 80)
    for idx in range(3):
        row = real_df.iloc[idx]
        print(f"\n{idx+1}. Specialty: {row['specialty']}")
        print(f"   Text: {row['transcription'][:60]}...")
        print(f"   ✅ Correlation: PRESERVED")

    print("\n" + "-" * 80)
    print("TILTED DATA (Broken Correlations)")
    print("-" * 80)
    for idx in range(3):
        row = synthetic_df.iloc[idx]
        print(f"\n{idx+1}. Specialty: {row['specialty']}")
        print(f"   Text: {row['transcription'][:60]}...")
        print(f"   ❌ Correlation: BROKEN")


def demo_use_as_baseline():
    """Demonstrate using Tilted as a baseline in experiments."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Using Tilted Generator as Experimental Baseline")
    print("=" * 80)

    real_df = create_medical_dataset()

    print("\nExperimental Setup:")
    print("- Real data: 50 samples with strong text-tabular correlations")
    print("- Tilted baseline: Same data with shuffled pairings")
    print("- Evaluation: Correlation analysis")

    print("\n" + "-" * 80)
    print("Baseline Performance (what metrics should detect as 'bad'):")
    print("-" * 80)

    gen = TiltedGenerator(shuffle_strategy="random", random_state=42)
    gen.fit(
        real_df,
        text_columns=['transcription'],
        tabular_columns=['specialty', 'age', 'severity']
    )

    synthetic = gen.generate(100)
    corr = analyze_correlations(synthetic, 'transcription', 'specialty')

    print(f"\nTilted Generator Correlation: {corr:.2%}")
    print(f"Expected Random Chance: ~{1/5:.2%} (1/{len(synthetic['specialty'].unique())} specialties)")
    print("\n✅ Any proper generative model should significantly outperform this baseline!")

    print("\n" + "-" * 80)
    print("Key Takeaway:")
    print("-" * 80)
    print("The Tilted generator establishes a lower bound for quality.")
    print("If your model doesn't beat this baseline, it's not preserving correlations!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TILTED DATA GENERATOR - ADVERSARIAL BASELINE DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows how the Tilted generator intentionally breaks")
    print("cross-modal correlations to create a 'worst-case' baseline.")
    print("\n" + "=" * 80)

    # Run all demos
    demo_medical_data()
    demo_kiva_data()
    demo_comparison_with_real()
    demo_use_as_baseline()

    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe Tilted generator is a crucial tool for:")
    print("1. ✅ Validating that evaluation metrics detect broken correlations")
    print("2. ✅ Establishing a lower bound for synthetic data quality")
    print("3. ✅ Serving as negative control in experiments")
    print("4. ✅ Proving that proper generators preserve meaningful correlations")
    print("\n" + "=" * 80)
