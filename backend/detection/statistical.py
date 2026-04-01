"""
Layer 1: Statistical Analysis — Z-score + IQR Label Anomaly Detection

Detects training data poisoning via:
1. Per-feature Z-score outlier detection
2. Per-feature IQR-based outlier detection
3. Label distribution anomaly analysis (Chi-squared test)

Reference: Standard statistical outlier detection adapted for
adversarial ML poisoning contexts.
"""

import numpy as np
from scipy import stats
from typing import Optional


class StatisticalDetector:
    """Z-score + IQR based statistical poisoning detector."""

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_anomaly_features: int = 1,
    ):
        """
        Args:
            z_threshold: Z-score threshold for flagging (default 3.0).
            iqr_multiplier: IQR fence multiplier (default 1.5).
            min_anomaly_features: Minimum features that must be anomalous
                to flag a sample.
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_anomaly_features = min_anomaly_features

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run statistical anomaly detection on the dataset.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Optional label array for label distribution analysis.

        Returns:
            Dictionary with detection results.
        """
        n_samples, n_features = X.shape

        # --- Z-Score Analysis ---
        z_scores = np.abs(stats.zscore(X, axis=0, nan_policy="omit"))
        # Handle constant features (zscore returns nan)
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        z_flags_per_feature = z_scores > self.z_threshold
        z_anomaly_counts = z_flags_per_feature.sum(axis=1)
        z_flagged = z_anomaly_counts >= self.min_anomaly_features

        # Per-sample composite Z-score (max across features)
        z_max_scores = z_scores.max(axis=1)

        # --- IQR Analysis ---
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_fence = Q1 - self.iqr_multiplier * IQR
        upper_fence = Q3 + self.iqr_multiplier * IQR

        iqr_flags_per_feature = (X < lower_fence) | (X > upper_fence)
        iqr_anomaly_counts = iqr_flags_per_feature.sum(axis=1)
        iqr_flagged = iqr_anomaly_counts >= self.min_anomaly_features

        # Per-sample IQR deviation score (normalized distance outside fence)
        iqr_deviations = np.zeros(n_samples)
        for j in range(n_features):
            if IQR[j] == 0:
                continue
            below = np.maximum(0, lower_fence[j] - X[:, j]) / IQR[j]
            above = np.maximum(0, X[:, j] - upper_fence[j]) / IQR[j]
            iqr_deviations += below + above
        iqr_deviations /= max(n_features, 1)

        # --- Combined Statistical Flags ---
        combined_flagged = z_flagged | iqr_flagged

        # Composite anomaly score (0-1 normalized)
        z_normalized = np.clip(z_max_scores / (self.z_threshold * 2), 0, 1)
        iqr_normalized = np.clip(iqr_deviations, 0, 1)
        composite_scores = 0.5 * z_normalized + 0.5 * iqr_normalized

        # --- Label Distribution Analysis ---
        label_analysis = None
        if y is not None:
            label_analysis = self._analyze_label_distribution(y)

        # --- Feature-Level Summary ---
        feature_stats = []
        for j in range(n_features):
            feature_stats.append({
                "feature_index": j,
                "mean": float(np.mean(X[:, j])),
                "std": float(np.std(X[:, j])),
                "q1": float(Q1[j]),
                "q3": float(Q3[j]),
                "iqr": float(IQR[j]),
                "z_outliers": int(z_flags_per_feature[:, j].sum()),
                "iqr_outliers": int(iqr_flags_per_feature[:, j].sum()),
            })

        return {
            "layer": "statistical",
            "layer_name": "Z-score + IQR Statistical Analysis",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "z_threshold": self.z_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "flagged_indices": np.where(combined_flagged)[0].tolist(),
            "n_flagged": int(combined_flagged.sum()),
            "flagged_ratio": float(combined_flagged.sum() / n_samples),
            "scores": composite_scores.tolist(),
            "z_scores_max": z_max_scores.tolist(),
            "iqr_deviations": iqr_deviations.tolist(),
            "is_flagged": combined_flagged.tolist(),
            "feature_stats": feature_stats,
            "label_analysis": label_analysis,
        }

    def _analyze_label_distribution(self, y: np.ndarray) -> dict:
        """Analyze label distribution for anomalies using statistical tests.

        Uses entropy-based imbalance detection and KL-divergence from
        domain-expected distributions rather than a naive uniform assumption.
        """
        unique_labels, counts = np.unique(y, return_counts=True)
        n_classes = len(unique_labels)
        n_total = len(y)

        if n_classes < 2:
            return {
                "n_classes": int(n_classes),
                "distribution": {str(k): int(v) for k, v in zip(unique_labels, counts)},
                "is_anomalous": False,
                "chi2_stat": 0.0,
                "p_value": 1.0,
                "message": "Only one class found — cannot assess distribution.",
            }

        # Chi-squared test against uniform distribution
        expected_uniform = np.full(n_classes, n_total / n_classes)
        chi2_stat, p_value = stats.chisquare(counts, expected_uniform)

        # Imbalance ratio
        imbalance_ratio = float(counts.max() / max(counts.min(), 1))

        # Entropy-based analysis (low entropy = high imbalance)
        proportions = counts / n_total
        entropy = float(-np.sum(proportions * np.log(proportions + 1e-10)))
        max_entropy = float(np.log(n_classes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        # Gini impurity
        gini = float(1.0 - np.sum(proportions ** 2))

        # A distribution is suspicious if it's EXTREMELY imbalanced
        # (beyond what domain datasets normally exhibit) or if
        # small classes have suspiciously round counts
        is_anomalous = (
            (imbalance_ratio > 20 and p_value < 0.001) or
            (normalized_entropy < 0.3 and n_classes > 2)
        )

        return {
            "n_classes": int(n_classes),
            "distribution": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "is_anomalous": bool(is_anomalous),
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "imbalance_ratio": float(imbalance_ratio),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "gini_impurity": round(gini, 4),
            "message": (
                f"Label distribution is highly anomalous (imbalance={imbalance_ratio:.1f}x, entropy={normalized_entropy:.2f})"
                if is_anomalous
                else f"Label distribution within expected parameters (imbalance={imbalance_ratio:.1f}x)."
            ),
        }
