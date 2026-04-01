"""
Layer 6: Backdoor Trigger Scanning

Detects potential backdoor trigger patterns in training data by:
1. Feature-space trigger detection — identifying fixed/constant feature patterns
   injected across samples (characteristic of backdoor attacks).
2. Neural Cleanse-inspired analysis — detecting input perturbation patterns that
   cause misclassification to a target label.
3. Entropy-based anomaly detection — low-entropy feature subsets suggest triggers.

Reference:
  Wang et al. (2019), "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
  in Neural Networks", IEEE S&P.
  Tran et al. (2018), "Spectral Signatures in Backdoor Attacks", NeurIPS.
"""

import numpy as np
import logging
from typing import Optional
from scipy import stats

logger = logging.getLogger(__name__)


class BackdoorTriggerDetector:
    """Backdoor trigger pattern scanner for training data poisoning detection."""

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        pattern_min_count: int = 5,
        trigger_percentile: float = 95.0,
    ):
        """
        Args:
            entropy_threshold: Normalized entropy below which a feature is
                flagged as potentially containing a trigger (low entropy = fixed values).
            pattern_min_count: Minimum samples sharing a pattern to flag it.
            trigger_percentile: Percentile for trigger score flagging.
        """
        self.entropy_threshold = entropy_threshold
        self.pattern_min_count = pattern_min_count
        self.trigger_percentile = trigger_percentile

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        model=None,
    ) -> dict:
        """
        Run backdoor trigger scanning.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Optional label array.
            model: Optional sklearn classifier.

        Returns:
            Dictionary with backdoor trigger detection results.
        """
        n_samples, n_features = X.shape

        try:
            # --- Component 1: Feature-space trigger pattern detection ---
            trigger_patterns = self._detect_feature_triggers(X, y)

            # --- Component 2: Neural Cleanse-inspired perturbation analysis ---
            perturbation_result = self._neural_cleanse_approximation(X, y, model)

            # --- Component 3: Entropy-based anomaly detection ---
            entropy_result = self._entropy_analysis(X, y)

            # --- Combine all components into composite scores ---
            scores = np.zeros(n_samples)

            # Weight from trigger patterns
            pattern_scores = trigger_patterns["sample_scores"]
            if np.max(pattern_scores) > 0:
                scores += 0.4 * (pattern_scores / max(np.max(pattern_scores), 1e-10))

            # Weight from perturbation analysis
            perturb_scores = perturbation_result["sample_scores"]
            if np.max(perturb_scores) > 0:
                scores += 0.35 * (perturb_scores / max(np.max(perturb_scores), 1e-10))

            # Weight from entropy analysis
            entropy_scores = entropy_result["sample_scores"]
            if np.max(entropy_scores) > 0:
                scores += 0.25 * (entropy_scores / max(np.max(entropy_scores), 1e-10))

            # Normalize to 0-1
            if scores.max() > 0:
                scores = scores / scores.max()

            threshold = np.percentile(scores, self.trigger_percentile)
            flagged = scores > threshold
            flagged_indices = np.where(flagged)[0].tolist()

            # Detected trigger features
            suspicious_features = trigger_patterns.get("suspicious_features", [])

            return {
                "layer": "backdoor",
                "layer_name": "Backdoor Trigger Scanner",
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "threshold": float(threshold),
                "flagged_indices": flagged_indices,
                "n_flagged": int(flagged.sum()),
                "flagged_ratio": float(flagged.sum() / n_samples),
                "scores": scores.tolist(),
                "is_flagged": flagged.tolist(),
                "trigger_patterns": trigger_patterns["patterns"],
                "suspicious_features": suspicious_features,
                "entropy_analysis": entropy_result["feature_entropies"],
                "perturbation_analysis": perturbation_result["summary"],
                "n_trigger_patterns_found": len(trigger_patterns["patterns"]),
            }

        except Exception as e:
            logger.error(f"Backdoor trigger scanning failed: {e}")
            return {
                "layer": "backdoor",
                "layer_name": "Backdoor Trigger Scanner",
                "n_samples": int(n_samples),
                "flagged_indices": [],
                "n_flagged": 0,
                "flagged_ratio": 0.0,
                "scores": [0.0] * n_samples,
                "is_flagged": [False] * n_samples,
                "trigger_patterns": [],
                "suspicious_features": [],
                "message": f"Analysis failed: {e}",
            }

    def _detect_feature_triggers(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> dict:
        """
        Detect fixed feature-value patterns (triggers) injected into a subset of samples.

        Backdoor triggers typically inject specific constant or near-constant
        values into certain features. We look for:
        - Features with suspiciously low variance in small sample subsets
        - Fixed value patterns co-occurring across samples with the same label
        """
        n_samples, n_features = X.shape
        patterns = []
        sample_scores = np.zeros(n_samples)
        suspicious_features = []

        for j in range(n_features):
            col = X[:, j]

            # Find values that appear with suspicious frequency
            # (but not so common as to be the majority)
            try:
                unique_vals, counts = np.unique(np.round(col, 2), return_counts=True)
            except Exception:
                continue

            for val, count in zip(unique_vals, counts):
                freq = count / n_samples
                # Suspicious: appears in 1-15% of samples (trigger range)
                if self.pattern_min_count <= count <= n_samples * 0.15:
                    matching_mask = np.abs(col - val) < 0.01
                    matching_indices = np.where(matching_mask)[0]

                    # Check if these samples disproportionately share a label
                    label_purity = 0.0
                    target_label = None
                    if y is not None and len(matching_indices) >= self.pattern_min_count:
                        matching_labels = y[matching_indices]
                        unique_labels, label_counts = np.unique(
                            matching_labels, return_counts=True
                        )
                        dominant_idx = label_counts.argmax()
                        label_purity = label_counts[dominant_idx] / len(matching_labels)
                        target_label = int(unique_labels[dominant_idx])

                    # High purity on a fixed value = potential backdoor trigger
                    if label_purity >= 0.8:
                        suspicion_score = label_purity * (1 - freq)
                        patterns.append({
                            "feature_index": int(j),
                            "trigger_value": float(val),
                            "n_samples": int(count),
                            "frequency": round(float(freq), 4),
                            "label_purity": round(float(label_purity), 4),
                            "target_label": target_label,
                            "suspicion_score": round(float(suspicion_score), 4),
                        })
                        sample_scores[matching_indices] += suspicion_score
                        if j not in suspicious_features:
                            suspicious_features.append(int(j))

        return {
            "patterns": patterns,
            "sample_scores": sample_scores,
            "suspicious_features": suspicious_features,
        }

    def _neural_cleanse_approximation(
        self, X: np.ndarray, y: Optional[np.ndarray], model=None,
    ) -> dict:
        """
        Neural Cleanse-inspired backdoor detection approximation.

        For each class, compute the minimum perturbation needed to flip
        predictions to that class. Classes with anomalously small perturbations
        are potential backdoor targets (the backdoor creates a shortcut).

        Uses a simplified, fast approximation suitable for tabular data.
        """
        n_samples, n_features = X.shape
        sample_scores = np.zeros(n_samples)
        summary = {"method": "neural_cleanse_approximation"}

        if y is None or model is None:
            # Without a model, use feature-deviation approach
            return self._feature_deviation_analysis(X, y)

        try:
            classes = np.unique(y)
            perturbation_norms = {}

            for target_class in classes:
                # Find samples NOT of this class
                other_mask = y != target_class
                other_X = X[other_mask]

                if len(other_X) < 10:
                    continue

                # Compute mean perturbation needed to flip to target class
                # Use model's decision function or probabilities
                proba = model.predict_proba(other_X)
                target_idx = list(classes).index(target_class)

                if target_idx < proba.shape[1]:
                    # Low confidence gap = easy to flip = suspicious
                    confidence_in_target = proba[:, target_idx]
                    mean_confidence = float(np.mean(confidence_in_target))
                    perturbation_norms[int(target_class)] = mean_confidence

                    # Samples already close to flipping are suspicious
                    close_to_flip = confidence_in_target > 0.3  # Unusually high confidence in wrong class
                    other_indices = np.where(other_mask)[0]
                    sample_scores[other_indices[close_to_flip]] += confidence_in_target[close_to_flip]

            # Detect anomalous class (one class much easier to flip to)
            if len(perturbation_norms) >= 2:
                norms = np.array(list(perturbation_norms.values()))
                anomaly_index = float(np.max(norms) / (np.median(norms) + 1e-10))
                summary["perturbation_norms"] = perturbation_norms
                summary["anomaly_index"] = round(anomaly_index, 4)
                summary["is_suspicious"] = anomaly_index > 2.0
            else:
                summary["is_suspicious"] = False

        except Exception as e:
            logger.warning(f"Neural Cleanse approximation failed: {e}")
            summary["error"] = str(e)
            summary["is_suspicious"] = False

        return {"sample_scores": sample_scores, "summary": summary}

    def _feature_deviation_analysis(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> dict:
        """Fallback: detect backdoors via feature deviation patterns when no model."""
        n_samples, n_features = X.shape
        sample_scores = np.zeros(n_samples)
        summary = {"method": "feature_deviation_fallback"}

        if y is None:
            return {"sample_scores": sample_scores, "summary": summary}

        classes = np.unique(y)
        for cls in classes:
            mask = y == cls
            cls_X = X[mask]
            cls_mean = cls_X.mean(axis=0)
            cls_std = cls_X.std(axis=0) + 1e-10

            # For each sample in this class, compute Mahalanobis-like distance
            deviations = np.abs(cls_X - cls_mean) / cls_std
            max_devs = deviations.max(axis=1)

            # Samples with extreme deviation in specific features may have triggers
            indices = np.where(mask)[0]
            sample_scores[indices] = max_devs

        summary["is_suspicious"] = bool(np.any(sample_scores > 5.0))
        return {"sample_scores": sample_scores, "summary": summary}

    def _entropy_analysis(
        self, X: np.ndarray, y: Optional[np.ndarray]
    ) -> dict:
        """
        Analyze per-feature entropy within class subsets.

        Backdoor triggers inject fixed values, causing abnormally low entropy
        in specific features for specific label subsets.
        """
        n_samples, n_features = X.shape
        sample_scores = np.zeros(n_samples)
        feature_entropies = []

        for j in range(n_features):
            col = X[:, j]

            # Discretize feature into bins for entropy computation
            try:
                n_bins = min(20, max(5, int(np.sqrt(n_samples))))
                hist, _ = np.histogram(col, bins=n_bins)
                hist = hist / hist.sum()
                hist = hist[hist > 0]

                feature_entropy = float(-np.sum(hist * np.log2(hist)))
                max_entropy = np.log2(n_bins)
                normalized_entropy = feature_entropy / max_entropy if max_entropy > 0 else 1.0
            except Exception:
                normalized_entropy = 1.0
                feature_entropy = 0.0

            is_suspicious = normalized_entropy < self.entropy_threshold

            # Per-class entropy analysis
            per_class_entropies = {}
            if y is not None:
                for cls in np.unique(y):
                    cls_col = col[y == cls]
                    if len(cls_col) < 5:
                        continue
                    try:
                        hist_c, _ = np.histogram(cls_col, bins=min(10, len(cls_col) // 2))
                        hist_c = hist_c / max(hist_c.sum(), 1)
                        hist_c = hist_c[hist_c > 0]
                        class_entropy = float(-np.sum(hist_c * np.log2(hist_c)))
                        max_class_entropy = np.log2(len(hist_c)) if len(hist_c) > 1 else 1.0
                        norm_class_entropy = class_entropy / max_class_entropy if max_class_entropy > 0 else 1.0

                        per_class_entropies[int(cls)] = round(norm_class_entropy, 4)

                        # If a specific class has very low entropy in this feature → trigger
                        if norm_class_entropy < self.entropy_threshold and not is_suspicious:
                            is_suspicious = True
                            # Score samples of this class
                            cls_mask = y == cls
                            sample_scores[cls_mask] += (1.0 - norm_class_entropy)
                    except Exception:
                        continue

            feature_entropies.append({
                "feature_index": int(j),
                "entropy": round(feature_entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "is_suspicious": bool(is_suspicious),
                "per_class_entropies": per_class_entropies,
            })

        return {
            "feature_entropies": feature_entropies,
            "sample_scores": sample_scores,
        }
