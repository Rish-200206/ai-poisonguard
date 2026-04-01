"""
Layer 4: IBM Adversarial Robustness Toolbox (ART) Integration

Provides a validation layer using IBM ART's defences for:
- Activation-based clustering defence
- Provenance/spectral validation

This serves as a secondary validation on top of our custom
detection layers, leveraging ART's tested implementations.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ART availability is checked lazily at runtime
def _check_art():
    try:
        from art.defences.detector.poison import ActivationDefence
        from art.estimators.classification import SklearnClassifier
        return True
    except ImportError:
        return False


class ARTDetector:
    """IBM ART-based poisoning detection validation layer."""

    def __init__(
        self,
        nb_clusters: int = 2,
        nb_dims: int = 10,
        clustering_method: str = "KMeans",
    ):
        self.nb_clusters = nb_clusters
        self.nb_dims = nb_dims
        self.clustering_method = clustering_method

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        model=None,
    ) -> dict:
        """
        Run ART-based poisoning detection.

        Args:
            X: Feature matrix.
            y: Labels array.
            model: Optional sklearn classifier. If None, trains one.

        Returns:
            Dictionary with ART detection results.
        """
        if not _check_art():
            return self._fallback_detection(X, y)

        try:
            return self._art_activation_defence(X, y, model)
        except Exception as e:
            logger.error(f"ART detection failed: {e}")
            return self._fallback_detection(X, y)

    def _art_activation_defence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model=None,
    ) -> dict:
        """Run ART's ActivationDefence."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from art.defences.detector.poison import ActivationDefence
        from art.estimators.classification import SklearnClassifier

        # Encode labels to contiguous integers for safe one-hot encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(int))
        n_classes = len(le.classes_)

        if model is None:
            model = RandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=-1
            )
            model.fit(X, y_encoded)

        # Wrap in ART classifier
        art_classifier = SklearnClassifier(model=model)

        # One-hot encode labels safely using contiguous encoded labels
        y_onehot = np.eye(n_classes)[y_encoded].astype(np.float32)

        # Run activation defence
        defence = ActivationDefence(
            classifier=art_classifier,
            x_train=X.astype(np.float32),
            y_train=y_onehot,
        )

        # Detect poison — pass cluster config to detect_poison()
        report, is_clean_lst = defence.detect_poison(
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce="PCA",
            clustering_method=self.clustering_method,
        )

        # Parse results
        n_samples = len(X)
        is_flagged = [not clean for clean in is_clean_lst]

        # Compute scores from the report
        scores = np.zeros(n_samples)
        flagged_indices = [i for i, f in enumerate(is_flagged) if f]
        scores[flagged_indices] = 1.0

        return {
            "layer": "art",
            "layer_name": "IBM ART Activation Defence",
            "art_available": True,
            "n_samples": int(n_samples),
            "flagged_indices": flagged_indices,
            "n_flagged": len(flagged_indices),
            "flagged_ratio": float(len(flagged_indices) / n_samples),
            "scores": scores.tolist(),
            "is_flagged": is_flagged,
            "report": report if isinstance(report, dict) else str(report),
        }

    def _fallback_detection(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> dict:
        """
        Fallback when ART is not installed.
        Uses a simplified statistical approach as a proxy.
        """
        from sklearn.ensemble import IsolationForest

        n_samples = X.shape[0]

        clf = IsolationForest(
            contamination=0.05, random_state=42, n_jobs=-1
        )
        predictions = clf.fit_predict(X)
        scores_raw = -clf.score_samples(X)  # Higher = more anomalous

        # Normalize
        s_min, s_max = scores_raw.min(), scores_raw.max()
        if s_max > s_min:
            scores = (scores_raw - s_min) / (s_max - s_min)
        else:
            scores = np.zeros(n_samples)

        flagged = predictions == -1
        flagged_indices = np.where(flagged)[0].tolist()

        return {
            "layer": "art",
            "layer_name": "IBM ART (Fallback — Isolation Forest)",
            "art_available": False,
            "n_samples": int(n_samples),
            "flagged_indices": flagged_indices,
            "n_flagged": int(flagged.sum()),
            "flagged_ratio": float(flagged.sum() / n_samples),
            "scores": scores.tolist(),
            "is_flagged": flagged.tolist(),
            "report": "ART not installed. Using Isolation Forest fallback.",
        }
