"""
Quick validation script to test adaptive thresholding across datasets
with different poisoning levels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import time

from detection.statistical import StatisticalDetector
from detection.spectral import SpectralDetector
from detection.clustering import ClusteringDetector
from detection.art_integration import ARTDetector
from detection.influence import InfluenceFunctionDetector
from detection.backdoor import BackdoorTriggerDetector
from domain_profiles import get_profile

DATASETS = [
    ("demo_upi_fraud_clean.csv", "CLEAN (0% poison)"),
    ("demo_upi_fraud_25pct_mixed.csv", "25% POISONED"),
    ("demo_upi_fraud_40pct_poison.csv", "40% POISONED"),
]

profile = get_profile("upi_fraud")

for filename, label in DATASETS:
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(filepath):
        print(f"\n⚠️  {filename} not found, skipping")
        continue

    df = pd.read_csv(filepath)

    # Auto-detect label column
    label_col = None
    for candidate in ["target", "label", "class", "y", "is_fraud", "fraud"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col:
        y = df[label_col].values
        df_features = df.drop(columns=[label_col])
    else:
        y = None
        df_features = df.copy()

    df_numeric = df_features.select_dtypes(include=[np.number]).fillna(0)
    X = df_numeric.to_numpy()

    if y is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    n_samples = X.shape[0]

    print(f"\n{'='*60}")
    print(f"  {label} — {filename} ({n_samples} samples)")
    print(f"{'='*60}")

    t0 = time.time()

    # Layer 1: Statistical
    stat = StatisticalDetector(
        z_threshold=profile["z_threshold"],
        iqr_multiplier=profile["iqr_multiplier"],
        min_anomaly_features=profile["min_anomaly_features"],
    )
    stat_r = stat.detect(X, y)

    # Layer 2: Spectral
    spec = SpectralDetector(
        top_k_singular=profile["spectral_top_k"],
        percentile_threshold=profile["spectral_percentile"],
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    spec_r = spec.detect(X, y)

    # Layer 3: Clustering
    clust = ClusteringDetector(
        n_clusters=profile["n_clusters"],
        purity_threshold=profile["purity_threshold"],
        umap_n_neighbors=profile["umap_n_neighbors"],
        umap_min_dist=profile["umap_min_dist"],
        auto_k=True,
        max_cluster_frac=profile.get("max_cluster_frac", 0.25),
        min_impurity_margin=profile.get("min_impurity_margin", 0.10),
    )
    clust_r = clust.detect(X, y)

    # Layer 4: ART (fallback)
    art = ARTDetector()
    art_r = art.detect(X, y)

    # Layer 5: Influence
    inf = InfluenceFunctionDetector(
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    inf_r = inf.detect(X, y)

    # Layer 6: Backdoor
    bd = BackdoorTriggerDetector(
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    bd_r = bd.detect(X, y)

    elapsed = time.time() - t0

    layers = [
        ("L1 Statistical", stat_r),
        ("L2 Spectral", spec_r),
        ("L3 Clustering", clust_r),
        ("L4 ART/IsoForest", art_r),
        ("L5 Influence", inf_r),
        ("L6 Backdoor", bd_r),
    ]

    print(f"  {'Layer':<20} {'Flagged':>8} {'Ratio':>8}  Method")
    print(f"  {'-'*55}")
    for name, result in layers:
        n_flagged = result.get("n_flagged", 0)
        ratio = result.get("flagged_ratio", 0)
        method = result.get("threshold_method", result.get("clustering_method", "-"))
        print(f"  {name:<20} {n_flagged:>8} {ratio:>7.1%}  {method}")

    # Ensemble
    from collections import Counter
    vote_counts = Counter()
    for _, result in layers:
        for idx in result.get("flagged_indices", []):
            vote_counts[idx] += 1

    ensemble_flagged = sum(1 for v in vote_counts.values() if v >= 2)
    warnings = sum(1 for v in vote_counts.values() if v == 1)
    print(f"\n  ENSEMBLE (≥2 votes): {ensemble_flagged} flagged, {warnings} warnings")
    print(f"  Time: {elapsed:.1f}s")

print(f"\n{'='*60}")
print("  VALIDATION COMPLETE")
print(f"{'='*60}")
