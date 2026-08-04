#!/usr/bin/env python3
"""
SynEval Four-Dimension Evaluation Framework.

Computes four evaluation axes in the semantic quantization space (C_X, C_T):

  Axis I   — Fidelity:  Joint Spectral Divergence (conditional JSD + MMD)
  Axis II  — Utility:   Bidirectional TSTR (Text->Attribute and Attribute->Text)
  Axis III — Diversity: Joint Shannon Entropy
  Axis IV  — Privacy:   Distance to Closest Record (DCR) in joint representation

Expected results:
  - Tilted Data:  low Fidelity (high JSD — joint distribution destroyed)
  - Baseline 1:   lower Utility (independent generation, no cross-modal constraint)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


print_flush("Importing libraries...")
print_flush("  pandas, numpy, scipy")
print_flush("  scikit-learn (GradientBoosting, LogisticRegression)")
print_flush("  SentenceTransformer")

BASE_DIR = Path(__file__).parent
EXPERIMENT_DIR = BASE_DIR / "experiments" / "baselines_filtered_20260428_195011"
SYNEVAL_DIR = EXPERIMENT_DIR / "syneval"
QUANTIZED_DATA_DIR = SYNEVAL_DIR / "quantized_data"
SYNTHETIC_DATA_DIR = EXPERIMENT_DIR / "synthetic_data"
OUTPUT_DIR = SYNEVAL_DIR / "four_dimensions"
OUTPUT_DIR.mkdir(exist_ok=True)

SBERT_MODEL = 'all-MiniLM-L6-v2'
TEXT_CLUSTER_K = 20  # K-Means clusters used for Attribute->Text direction

# One-to-Many evaluation configuration (3 tasks on Fake Jobs)
DATASETS = {
    'fake_jobs': {
        'text_col': 'description',
        'label_col': 'fraudulent',
        'label_type': 'binary',
        'original': 'fake_jobs_original',
        'baselines': [
            'fake_jobs_baseline1_independent',
            'fake_jobs_baseline2_sequential',
            'fake_jobs_baseline3_joint_llm',
            'fake_jobs_baseline4_tabsyn',
            'fake_jobs_tilted'
        ]
    },
    'fake_jobs_logo': {
        'text_col': 'description',
        'label_col': 'has_company_logo',
        'label_type': 'binary',
        'original': 'fake_jobs_logo_original',
        'baselines': [
            'fake_jobs_logo_baseline1_independent',
            'fake_jobs_logo_baseline2_sequential',
            'fake_jobs_logo_baseline3_joint_llm',
            'fake_jobs_logo_baseline4_tabsyn',
            'fake_jobs_logo_tilted'
        ]
    },
    'fake_jobs_joint': {
        'text_col': 'description',
        'label_col': 'fraudulent_logo',
        'label_type': 'multiclass',  # 4 combinations of (fraudulent, has_company_logo)
        'original': 'fake_jobs_joint_original',
        'baselines': [
            'fake_jobs_joint_baseline1_independent',
            'fake_jobs_joint_baseline2_sequential',
            'fake_jobs_joint_baseline3_joint_llm',
            'fake_jobs_joint_baseline4_tabsyn',
            'fake_jobs_joint_tilted'
        ]
    }
}


# ================================================================================
# Axis I: Fidelity — Joint Spectral Divergence (JSD) + MMD
# ================================================================================

def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Maximum Mean Discrepancy (MMD) in embedding space.

    Directly compares distributions of real and synthetic embeddings.
    Detects semantic-level corruption (e.g., Tilted Data).

    Args:
        X: real data embeddings (n, d)
        Y: synthetic data embeddings (m, d)
        kernel: 'rbf' or 'linear'
        gamma: RBF bandwidth (None = median heuristic)

    Returns:
        mmd: MMD distance (lower is better; 0 means identical distributions)
    """
    max_samples = 1000
    if len(X) > max_samples:
        np.random.seed(42)
        X = X[np.random.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        np.random.seed(42)
        Y = Y[np.random.choice(len(Y), max_samples, replace=False)]

    n, m = len(X), len(Y)

    if kernel == 'rbf':
        if gamma is None:
            pairwise_dists = np.sum((X[:100] - X[:100, None])**2, axis=2)
            gamma = 1.0 / np.median(pairwise_dists[pairwise_dists > 0])

        def rbf_kernel(A, B):
            sq_norms_A = np.sum(A**2, axis=1, keepdims=True)
            sq_norms_B = np.sum(B**2, axis=1, keepdims=True)
            dists_sq = sq_norms_A + sq_norms_B.T - 2 * A @ B.T
            return np.exp(-gamma * dists_sq)

        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)
    else:
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T

    mmd_sq = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
           - 2 * K_XY.sum() / (n * m) \
           + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))

    return np.sqrt(max(0, mmd_sq))


def compute_joint_distribution(quantized_df):
    """
    Compute the joint probability distribution P(C_X, C_T) from quantized data.

    Returns:
        joint_prob: normalized counts over (C_X, C_T) pairs
    """
    counts = quantized_df.groupby(['C_X', 'C_T']).size()
    total = counts.sum()
    return counts / total


def compute_jsd_fidelity(real_quantized_df, synth_quantized_df):
    """
    Joint Spectral Divergence: measures corruption of the conditional distribution.

    Computes JSD(P(C_T|C_X) || P_real(C_T|C_X)), which detects Tilted Data.
    Also returns the unconditional joint JSD for reference.

    Args:
        real_quantized_df:  quantized real data
        synth_quantized_df: quantized synthetic data

    Returns:
        jsd_conditional: weighted average of per-C_X conditional JSD (lower is better)
        jsd_joint:       joint distribution JSD (reference metric)
        coverage:        fraction of real grid cells covered by synthetic data
    """
    real_dist = compute_joint_distribution(real_quantized_df)
    synth_dist = compute_joint_distribution(synth_quantized_df)

    all_keys = set(real_dist.index) | set(synth_dist.index)
    real_prob = np.array([real_dist.get(key, 0) for key in all_keys])
    synth_prob = np.array([synth_dist.get(key, 0) for key in all_keys])
    jsd_joint = jensenshannon(real_prob, synth_prob, base=2)

    all_cx = set(real_quantized_df['C_X'].unique()) | set(synth_quantized_df['C_X'].unique())
    conditional_jsds = []
    weights = []

    for cx in all_cx:
        real_cx_data = real_quantized_df[real_quantized_df['C_X'] == cx]
        synth_cx_data = synth_quantized_df[synth_quantized_df['C_X'] == cx]

        if len(real_cx_data) == 0:
            continue

        real_ct_counts = real_cx_data['C_T'].value_counts()
        real_ct_prob = (real_ct_counts / real_ct_counts.sum()).to_dict()

        if len(synth_cx_data) == 0:
            conditional_jsds.append(1.0)
            weights.append(len(real_cx_data))
            continue

        synth_ct_counts = synth_cx_data['C_T'].value_counts()
        synth_ct_prob = (synth_ct_counts / synth_ct_counts.sum()).to_dict()

        all_ct = set(real_ct_prob.keys()) | set(synth_ct_prob.keys())
        real_ct_vec = np.array([real_ct_prob.get(ct, 0) for ct in all_ct])
        synth_ct_vec = np.array([synth_ct_prob.get(ct, 0) for ct in all_ct])

        cond_jsd = jensenshannon(real_ct_vec, synth_ct_vec, base=2) if len(all_ct) > 1 else 0.0
        conditional_jsds.append(cond_jsd)
        weights.append(len(real_cx_data))

    jsd_conditional = np.average(conditional_jsds, weights=weights) if conditional_jsds else 1.0

    real_covered = set(real_dist.index)
    synth_covered = set(synth_dist.index)
    coverage = len(synth_covered & real_covered) / len(real_covered) if real_covered else 0

    return jsd_conditional, jsd_joint, coverage


def compute_nmi_gap(real_quantized_df, synth_quantized_df):
    """
    Normalized Mutual Information Gap: measures whether synthetic data preserves
    the statistical association between C_X and C_T.

    Expected:
      - Real data: high NMI (label and text semantics are correlated)
      - Baseline 1: NMI ~ 0 (independent generation)
      - Tilted Data: NMI ~ 0 (shuffling destroyed the association)

    Returns:
        nmi_real:  NMI(C_X, C_T) on real data
        nmi_synth: NMI(C_X, C_T) on synthetic data
        nmi_gap:   |nmi_real - nmi_synth| (lower is better)
    """
    nmi_real = normalized_mutual_info_score(
        real_quantized_df['C_X'],
        real_quantized_df['C_T'],
        average_method='arithmetic'
    )
    nmi_synth = normalized_mutual_info_score(
        synth_quantized_df['C_X'],
        synth_quantized_df['C_T'],
        average_method='arithmetic'
    )
    return nmi_real, nmi_synth, abs(nmi_real - nmi_synth)


# ================================================================================
# Axis II: Utility — Train on Synthetic, Test on Real (TSTR)
# ================================================================================

def prepare_tstr_data(dataset_name, config):
    """
    Load and encode all data needed for bidirectional TSTR evaluation.

    Returns:
        real_train_X, real_train_y:          real training set (embeddings + labels)
        real_test_X, real_test_y:            real test set
        real_test_text_clusters:             K-Means cluster IDs for real test text
        embeddings_dict:                     {baseline_name: (synth_X, synth_y, synth_text_clusters)}
    """
    text_col = config['text_col']
    label_col = config['label_col']

    real_file = SYNTHETIC_DATA_DIR / f"{config['original']}.csv"
    real_df = pd.read_csv(real_file)

    print_flush(f"  Loading SBERT model...")
    sbert = SentenceTransformer(SBERT_MODEL)

    print_flush(f"  Encoding real data text...")
    real_texts = [str(t) for t in real_df[text_col] if pd.notna(t)]
    real_embeddings = sbert.encode(real_texts, show_progress_bar=False, batch_size=32)
    real_labels = real_df[label_col].values[:len(real_texts)]

    if real_labels.dtype == object or isinstance(real_labels[0], str):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        real_labels_encoded = label_encoder.fit_transform(real_labels)
    else:
        label_encoder = None
        real_labels_encoded = real_labels

    print_flush(f"  Clustering real text (K={TEXT_CLUSTER_K})...")
    kmeans = KMeans(n_clusters=TEXT_CLUSTER_K, random_state=42, n_init=10)
    real_embeddings_float64 = real_embeddings.astype(np.float64)
    real_text_clusters = kmeans.fit_predict(real_embeddings_float64)

    unique, counts = np.unique(real_labels_encoded, return_counts=True)
    min_count = counts.min()

    if min_count >= 2:
        real_train_X, real_test_X, real_train_y, real_test_y, _, real_test_text_clusters = train_test_split(
            real_embeddings_float64, real_labels_encoded, real_text_clusters,
            test_size=0.2, random_state=42, stratify=real_labels_encoded
        )
    else:
        print_flush(f"    Warning: {(counts == 1).sum()} class(es) with a single sample — using non-stratified split")
        real_train_X, real_test_X, real_train_y, real_test_y, _, real_test_text_clusters = train_test_split(
            real_embeddings_float64, real_labels_encoded, real_text_clusters,
            test_size=0.2, random_state=42
        )

    print_flush(f"  Real data split: train={len(real_train_X)}, test={len(real_test_X)}")

    embeddings_dict = {}

    for baseline_name in config['baselines']:
        synth_file = SYNTHETIC_DATA_DIR / f"{baseline_name}.csv"
        if not synth_file.exists():
            continue

        synth_df = pd.read_csv(synth_file)
        is_baseline4 = 'baseline4' in baseline_name
        sbert_cols = [c for c in synth_df.columns if c.startswith('sbert_')]

        if is_baseline4 and sbert_cols:
            synth_embeddings = synth_df[sbert_cols].values.astype(np.float64)
        else:
            synth_texts = [str(t) for t in synth_df[text_col] if pd.notna(t)]
            synth_embeddings = sbert.encode(synth_texts, show_progress_bar=False, batch_size=32)

        synth_labels = synth_df[label_col].values[:len(synth_embeddings)]

        if is_baseline4 and config['label_type'] in ['multiclass', 'binary']:
            if synth_labels.dtype in [np.float32, np.float64]:
                synth_labels = np.round(synth_labels).astype(int)
            elif synth_labels.dtype != object:
                synth_labels = synth_labels.astype(int)

        if label_encoder is not None:
            if synth_labels.dtype == object or (len(synth_labels) > 0 and isinstance(synth_labels[0], str)):
                synth_labels = label_encoder.transform(synth_labels)
            elif synth_labels.dtype != real_labels_encoded.dtype:
                synth_labels = synth_labels.astype(real_labels_encoded.dtype)

        synth_embeddings_float64 = synth_embeddings.astype(np.float64)
        synth_text_clusters = kmeans.predict(synth_embeddings_float64)
        embeddings_dict[baseline_name] = (synth_embeddings_float64, synth_labels, synth_text_clusters)

    return real_train_X, real_train_y, real_test_X, real_test_y, real_test_text_clusters, embeddings_dict


def compute_tstr_utility(synth_X, synth_y, real_test_X, real_test_y, label_type):
    """
    Text->Attribute TSTR: train on synthetic embeddings, test on real embeddings.

    Returns:
        (F1, accuracy) for classification or (RMSE, MAE) for regression
    """
    if label_type == 'regression':
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        regressor = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        regressor.fit(synth_X, synth_y)
        y_pred = regressor.predict(real_test_X)
        return np.sqrt(mean_squared_error(real_test_y, y_pred)), mean_absolute_error(real_test_y, y_pred)

    else:
        if label_type == 'binary':
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)

        clf.fit(synth_X, synth_y)
        y_pred = clf.predict(real_test_X)

        if label_type == 'binary':
            f1 = f1_score(real_test_y, y_pred, average='binary')
        else:
            f1 = f1_score(real_test_y, y_pred, average='macro')

        return f1, accuracy_score(real_test_y, y_pred)


def compute_reverse_tstr_utility(synth_y, synth_text_clusters, real_test_y, real_test_text_clusters):
    """
    Attribute->Text TSTR: train on synthetic (tabular label -> text cluster),
    test on real data. Measures whether synthetic data preserves the relationship
    "tabular attributes predict text semantics".

    Returns:
        f1: macro F1 score
        accuracy: accuracy
    """
    synth_y_2d = synth_y.reshape(-1, 1)
    real_test_y_2d = real_test_y.reshape(-1, 1)

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(synth_y_2d, synth_text_clusters)
    y_pred = clf.predict(real_test_y_2d)

    return (
        f1_score(real_test_text_clusters, y_pred, average='macro'),
        accuracy_score(real_test_text_clusters, y_pred)
    )


# ================================================================================
# Axis III: Diversity — Joint Shannon Entropy
# ================================================================================

def compute_joint_entropy(quantized_df):
    """
    Joint Shannon entropy H(C_X, C_T) of the synthetic data.

    Returns:
        joint_entropy:      entropy in bits (higher = more diverse)
        normalized_entropy: entropy / max_possible_entropy
    """
    joint_dist = compute_joint_distribution(quantized_df)
    joint_ent = shannon_entropy(joint_dist.values, base=2)

    n_unique_states = len(joint_dist)
    max_entropy = np.log2(n_unique_states) if n_unique_states > 0 else 0
    normalized_ent = joint_ent / max_entropy if max_entropy > 0 else 0

    return joint_ent, normalized_ent


# ================================================================================
# Axis IV: Privacy — Distance to Closest Record (DCR)
# ================================================================================

def compute_dcr_privacy(real_tabular, real_embeddings, synth_tabular, synth_embeddings, sample_size=500):
    """
    DCR in the joint (tabular + text) representation space.

    Concatenates L2-normalized tabular features and scaled SBERT embeddings
    to form a unified representation, then measures nearest-neighbor distances.

    Args:
        real_tabular:    real tabular features (N,)
        real_embeddings: real SBERT embeddings (N, 384)
        synth_tabular:   synthetic tabular features (M,)
        synth_embeddings:synthetic SBERT embeddings (M, 384)
        sample_size:     subsample size to keep runtime manageable

    Returns:
        mean_dcr: mean distance to closest real record (higher = safer)
        min_dcr:  minimum DCR (most exposed synthetic record)
        dcr_std:  standard deviation of DCR scores
    """
    real_X = real_tabular.reshape(-1, 1).astype(np.float64)
    synth_X = synth_tabular.reshape(-1, 1).astype(np.float64)

    real_X_norm = real_X / (np.linalg.norm(real_X, axis=1, keepdims=True) + 1e-8)
    synth_X_norm = synth_X / (np.linalg.norm(synth_X, axis=1, keepdims=True) + 1e-8)

    real_E_norm = real_embeddings / (np.linalg.norm(real_embeddings, axis=1, keepdims=True) + 1e-8)
    synth_E_norm = synth_embeddings / (np.linalg.norm(synth_embeddings, axis=1, keepdims=True) + 1e-8)

    var_X = np.var(real_X_norm)
    var_E = np.var(real_E_norm)
    lambda_scale = np.sqrt(var_X / var_E) if var_E > 0 else 1.0

    real_Z = np.concatenate([real_X_norm, lambda_scale * real_E_norm], axis=1)
    synth_Z = np.concatenate([synth_X_norm, lambda_scale * synth_E_norm], axis=1)

    if len(synth_Z) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(synth_Z), sample_size, replace=False)
        synth_sample = synth_Z[indices]
    else:
        synth_sample = synth_Z

    dcr_list = [np.linalg.norm(real_Z - synth_vec, axis=1).min() for synth_vec in synth_sample]
    dcr_array = np.array(dcr_list)

    return dcr_array.mean(), dcr_array.min(), dcr_array.std()


# ================================================================================
# Main evaluation loop
# ================================================================================

def evaluate_dataset(dataset_name, config):
    """Evaluate all baselines for a single dataset across all four axes."""

    print_flush(f"\n{'='*80}")
    print_flush(f"Dataset: {dataset_name.replace('_', ' ').title()}")
    print_flush(f"{'='*80}")

    results = []

    real_quantized_file = QUANTIZED_DATA_DIR / f"{config['original']}_quantized.csv"
    real_quantized_df = pd.read_csv(real_quantized_file)
    print_flush(f"  Loaded quantized real data: {len(real_quantized_df)} rows")

    print_flush(f"\n[Preparing TSTR data]")
    (real_train_X, real_train_y, real_test_X, real_test_y,
     real_test_text_clusters, embeddings_dict) = prepare_tstr_data(dataset_name, config)

    for baseline_name in config['baselines']:
        print_flush(f"\n{'─'*80}")
        print_flush(f"Baseline: {baseline_name.replace(config['original'] + '_', '')}")
        print_flush(f"{'─'*80}")

        result = {
            'dataset': dataset_name,
            'baseline': baseline_name.replace(config['original'] + '_', '')
        }

        synth_quantized_file = QUANTIZED_DATA_DIR / f"{baseline_name}_quantized.csv"
        if not synth_quantized_file.exists():
            print_flush(f"  Quantized file not found, skipping.")
            continue

        synth_quantized_df = pd.read_csv(synth_quantized_file)

        # Axis I: Fidelity
        print_flush(f"  [Axis I] Fidelity...")

        jsd_conditional, jsd_joint, coverage = compute_jsd_fidelity(real_quantized_df, synth_quantized_df)
        result['fidelity_jsd_conditional'] = jsd_conditional
        result['fidelity_jsd_joint'] = jsd_joint
        result['fidelity_coverage'] = coverage
        print_flush(f"    [Discrete space] JSD_conditional: {jsd_conditional:.4f}, JSD_joint: {jsd_joint:.4f}")

        nmi_real, nmi_synth, nmi_gap = compute_nmi_gap(real_quantized_df, synth_quantized_df)
        result['fidelity_nmi_real'] = nmi_real
        result['fidelity_nmi_synth'] = nmi_synth
        result['fidelity_nmi_gap'] = nmi_gap
        print_flush(f"    [Association]    NMI_real: {nmi_real:.4f}, NMI_synth: {nmi_synth:.4f}, Gap: {nmi_gap:.4f}")

        if baseline_name in embeddings_dict:
            synth_X, _, _ = embeddings_dict[baseline_name]
            mmd = compute_mmd(real_train_X, synth_X, kernel='rbf')
            result['fidelity_mmd'] = mmd
            print_flush(f"    [Embedding space] MMD: {mmd:.4f}")
        else:
            result['fidelity_mmd'] = None
            print_flush(f"    [Embedding space] MMD: cannot compute")

        # Axis III: Diversity
        print_flush(f"  [Axis III] Diversity (Entropy)...")
        joint_ent, normalized_ent = compute_joint_entropy(synth_quantized_df)
        result['diversity_entropy'] = joint_ent
        result['diversity_normalized'] = normalized_ent
        print_flush(f"    Joint Entropy: {joint_ent:.4f} bits")
        print_flush(f"    Normalized:    {normalized_ent:.2%}")

        # Axis II: Utility (bidirectional TSTR)
        print_flush(f"  [Axis II] Utility (Bidirectional TSTR)...")
        if baseline_name in embeddings_dict:
            synth_X, synth_y, synth_text_clusters = embeddings_dict[baseline_name]

            metric1_t2a, metric2_t2a = compute_tstr_utility(
                synth_X, synth_y, real_test_X, real_test_y, config['label_type']
            )

            if config['label_type'] == 'regression':
                result['utility_t2a_rmse'] = metric1_t2a
                result['utility_t2a_mae'] = metric2_t2a
                print_flush(f"    [T2A] RMSE: {metric1_t2a:.4f}, MAE: {metric2_t2a:.4f}")
            else:
                result['utility_t2a_f1'] = metric1_t2a
                result['utility_t2a_accuracy'] = metric2_t2a
                print_flush(f"    [T2A] F1: {metric1_t2a:.4f}, Accuracy: {metric2_t2a:.4f}")

            f1_a2t, accuracy_a2t = compute_reverse_tstr_utility(
                synth_y, synth_text_clusters, real_test_y, real_test_text_clusters
            )
            result['utility_a2t_f1'] = f1_a2t
            result['utility_a2t_accuracy'] = accuracy_a2t
            print_flush(f"    [A2T] F1: {f1_a2t:.4f}, Accuracy: {accuracy_a2t:.4f}")

            if config['label_type'] == 'regression':
                result['utility_f1'] = f1_a2t
                result['utility_rmse'] = metric1_t2a
            else:
                result['utility_f1'] = (metric1_t2a + f1_a2t) / 2
                result['utility_accuracy'] = (metric2_t2a + accuracy_a2t) / 2
                print_flush(f"    [Average] F1: {result['utility_f1']:.4f}, Accuracy: {result['utility_accuracy']:.4f}")
        else:
            for key in ['utility_t2a_f1', 'utility_t2a_accuracy', 'utility_t2a_rmse',
                        'utility_t2a_mae', 'utility_a2t_f1', 'utility_a2t_accuracy',
                        'utility_f1', 'utility_accuracy']:
                result[key] = None
            print_flush(f"    Cannot compute TSTR (no embeddings)")

        # Axis IV: Privacy (DCR)
        print_flush(f"  [Axis IV] Privacy (DCR — Joint)...")
        if baseline_name in embeddings_dict:
            synth_X, synth_y, _ = embeddings_dict[baseline_name]
            mean_dcr, min_dcr, dcr_std = compute_dcr_privacy(
                real_train_y, real_train_X,
                synth_y, synth_X
            )
            result['privacy_mean_dcr'] = mean_dcr
            result['privacy_min_dcr'] = min_dcr
            result['privacy_std_dcr'] = dcr_std
            print_flush(f"    Mean DCR: {mean_dcr:.4f}")
            print_flush(f"    Min DCR:  {min_dcr:.4f} (most exposed record)")
            print_flush(f"    Std DCR:  {dcr_std:.4f}")
        else:
            result['privacy_mean_dcr'] = None
            result['privacy_min_dcr'] = None
            result['privacy_std_dcr'] = None
            print_flush(f"    Cannot compute DCR")

        results.append(result)

    return results


def main():
    print_flush("="*80)
    print_flush("SynEval — Four-Dimension Evaluation Framework")
    print_flush("="*80)
    print_flush("\nAxes:")
    print_flush("  Axis I   — Fidelity:  Joint Spectral Divergence")
    print_flush("  Axis II  — Utility:   Train on Synthetic, Test on Real (bidirectional)")
    print_flush("  Axis III — Diversity: Joint Shannon Entropy")
    print_flush("  Axis IV  — Privacy:   Distance to Closest Record")
    print_flush("\nExpected outcomes:")
    print_flush("  Tilted Data: high JSD (conditional distribution destroyed)")
    print_flush("  Baseline 1:  lower utility F1 (no cross-modal conditioning)")

    all_results = []

    for dataset_name, config in DATASETS.items():
        dataset_results = evaluate_dataset(dataset_name, config)
        all_results.extend(dataset_results)

    results_df = pd.DataFrame(all_results)

    csv_file = OUTPUT_DIR / "four_dimensions_results.csv"
    results_df.to_csv(csv_file, index=False)
    print_flush(f"\n  Saved CSV: {csv_file}")

    json_file = OUTPUT_DIR / "four_dimensions_results.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print_flush(f"  Saved JSON: {json_file}")

    print_flush(f"\n{'='*80}")
    print_flush("Results Summary")
    print_flush(f"{'='*80}")

    for dataset_name in DATASETS.keys():
        dataset_results = results_df[results_df['dataset'] == dataset_name]
        if len(dataset_results) == 0:
            continue

        print_flush(f"\n{dataset_name.replace('_', ' ').title()}:")
        print_flush(f"{'Baseline':<25} {'JSD_cond':<12} {'JSD_joint':<12} {'F1':<10} {'DCR':<10}")
        print_flush("-" * 75)

        for _, row in dataset_results.iterrows():
            baseline = row['baseline']
            jsd_cond = f"{row['fidelity_jsd_conditional']:.4f}" if pd.notna(row.get('fidelity_jsd_conditional')) else "N/A"
            jsd_joint = f"{row['fidelity_jsd_joint']:.4f}" if pd.notna(row.get('fidelity_jsd_joint')) else "N/A"
            f1 = f"{row['utility_f1']:.4f}" if pd.notna(row['utility_f1']) else "N/A"
            dcr = f"{row['privacy_mean_dcr']:.4f}" if pd.notna(row['privacy_mean_dcr']) else "N/A"
            marker = " *" if 'tilted' in baseline else ""
            print_flush(f"{baseline:<25} {jsd_cond:<12} {jsd_joint:<12} {f1:<10} {dcr:<10}{marker}")

    print_flush(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
