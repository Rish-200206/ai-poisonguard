"""
Layer 2: Spectral Signature Detection — SVD-based Analysis

Detects training data poisoning by analyzing spectral signatures
in the feature representation. Poisoned samples tend to leave a
detectable trace in the top singular vectors of the data matrix.

Reference: Tran, B., Li, J., & Madry, A. (2018).
"Spectral Signatures in Backdoor Attacks."
NeurIPS 2018.
"""

import numpy as np
from scipy import stats
from typing import Optional


class SpectralDetector:
    """SVD-based spectral signature poisoning detector."""

    def __init__(
        self,
        top_k_singular: int = 3,
        percentile_threshold: float = 95.0,
        score_method: str = "correlation",
    ):
        """
        Args:
            top_k_singular: Number of top singular vectors to analyze.
            percentile_threshold: Percentile threshold for outlier flagging.
            score_method: Scoring method — 'correlation' or 'projection'.
        """
        self.top_k_singular = top_k_singular
        self.percentile_threshold = percentile_threshold
        self.score_method = score_method

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run SVD-based spectral signature detection.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Optional label array (used for per-class analysis).

        Returns:
            Dictionary with spectral detection results.
        """
        n_samples, n_features = X.shape
        k = min(self.top_k_singular, n_features, n_samples)

        # Step 1: Center the data matrix
        X_centered = X - X.mean(axis=0)

        # Step 2: Compute SVD (truncated for large datasets, full for small)
        if n_samples > 5000 and k < min(n_samples, n_features) - 1:
            try:
                from scipy.sparse.linalg import svds
                U, S, Vt = svds(X_centered.astype(np.float64), k=k)
                # svds returns in ascending order — reverse to descending
                idx = np.argsort(S)[::-1]
                S, U, Vt = S[idx], U[:, idx], Vt[idx, :]
            except Exception:
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        else:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Compute spectral scores
        if self.score_method == "correlation":
            scores = self._correlation_scores(X_centered, Vt, k)
        else:
            scores = self._projection_scores(X_centered, Vt, k)

        # Step 4: Flag outliers based on percentile threshold
        threshold = np.percentile(scores, self.percentile_threshold)
        flagged = scores > threshold

        # Step 5: Per-class analysis if labels are provided
        per_class_results = None
        if y is not None:
            per_class_results = self._per_class_analysis(
                X_centered, y, Vt, k
            )

        # Singular value analysis (poisoning often inflates top singular values)
        sv_analysis = self._analyze_singular_values(S)

        # Normalize scores to 0-1
        score_min, score_max = scores.min(), scores.max()
        if score_max > score_min:
            normalized_scores = (scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = np.zeros(n_samples)

        return {
            "layer": "spectral",
            "layer_name": "SVD Spectral Signature Analysis",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "top_k_singular": int(k),
            "threshold": float(threshold),
            "flagged_indices": np.where(flagged)[0].tolist(),
            "n_flagged": int(flagged.sum()),
            "flagged_ratio": float(flagged.sum() / n_samples),
            "scores": normalized_scores.tolist(),
            "raw_scores": scores.tolist(),
            "is_flagged": flagged.tolist(),
            "singular_values": S[:min(10, len(S))].tolist(),
            "singular_value_analysis": sv_analysis,
            "explained_variance_ratio": self._explained_variance(S),
            "per_class_results": per_class_results,
        }

    def _correlation_scores(
        self, X_centered: np.ndarray, Vt: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Compute spectral signature scores using correlation with
        top-k right singular vectors.

        Each sample's score = sum of squared correlations with top-k
        singular vectors, weighted by singular value magnitude.
        """
        n_samples = X_centered.shape[0]
        scores = np.zeros(n_samples)

        for i in range(k):
            v = Vt[i]  # i-th right singular vector
            # Project each sample onto this singular vector
            projections = X_centered @ v
            # Squared projection captures the spectral signature
            scores += projections ** 2

        return scores

    def _projection_scores(
        self, X_centered: np.ndarray, Vt: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Compute scores using residual after removing top-k components.
        Poisoned samples have higher residual in the spectral subspace.
        """
        # Project onto top-k subspace
        V_k = Vt[:k].T  # (n_features, k)
        projections = X_centered @ V_k  # (n_samples, k)
        reconstructed = projections @ V_k.T  # (n_samples, n_features)

        # Spectral signature = norm of projection onto top-k subspace
        scores = np.linalg.norm(projections, axis=1)
        return scores

    def _per_class_analysis(
        self,
        X_centered: np.ndarray,
        y: np.ndarray,
        Vt: np.ndarray,
        k: int,
    ) -> list:
        """Analyze spectral signatures per class to detect class-specific poisoning."""
        results = []
        unique_labels = np.unique(y)

        for label in unique_labels:
            mask = y == label
            class_X = X_centered[mask]

            if len(class_X) < 2:
                continue

            scores = self._correlation_scores(class_X, Vt, k)
            threshold = np.percentile(scores, self.percentile_threshold)

            results.append({
                "label": int(label) if isinstance(label, (int, np.integer)) else str(label),
                "n_samples": int(mask.sum()),
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std()),
                "n_flagged": int((scores > threshold).sum()),
                "flagged_ratio": float((scores > threshold).sum() / mask.sum()),
            })

        return results

    def _analyze_singular_values(self, S: np.ndarray) -> dict:
        """
        Analyze singular value distribution.
        Poisoning attacks often inflate the top singular values.
        """
        if len(S) < 2:
            return {"is_suspicious": False, "message": "Too few dimensions."}

        # Ratio of top SV to second
        sv_ratio = float(S[0] / S[1]) if S[1] > 0 else float("inf")

        # Energy concentration in top-1
        total_energy = (S ** 2).sum()
        top1_energy = float(S[0] ** 2 / total_energy) if total_energy > 0 else 0

        # Suspicious if top SV dominates disproportionately
        is_suspicious = sv_ratio > 5.0 or top1_energy > 0.5

        return {
            "sv_ratio_1_2": round(sv_ratio, 4),
            "top1_energy_fraction": round(top1_energy, 4),
            "is_suspicious": bool(is_suspicious),
            "message": (
                f"Top singular value is {sv_ratio:.1f}x the second — "
                f"possible spectral poisoning signature detected."
                if is_suspicious
                else "Singular value distribution appears normal."
            ),
        }

    def _explained_variance(self, S: np.ndarray) -> list:
        """Compute explained variance ratio for top singular values."""
        total = (S ** 2).sum()
        if total == 0:
            return []
        ratios = (S ** 2 / total).tolist()
        return ratios[:min(10, len(ratios))]
