"""
Layer 5: Influence Function Analysis

Estimates the influence of each training sample on the model's predictions
using a simplified Leave-One-Out (LOO) and gradient-based approximation.

Disproportionately influential samples are likely poisoned — a small change
in a single training point should not drastically shift decision boundaries.

Reference:
  Koh & Liang (2017), "Understanding Black-box Predictions via Influence Functions", ICML.
  Pruthi et al. (2020), "Estimating Training Data Influence by Tracing Gradient Descent", NeurIPS (TracIn).
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InfluenceFunctionDetector:
    """Influence function-based poisoning detector."""

    def __init__(
        self,
        n_loo_samples: int = 200,
        influence_percentile: float = 95.0,
        use_tracin: bool = True,
    ):
        """
        Args:
            n_loo_samples: Max samples for LOO estimation (subsampled for speed).
            influence_percentile: Percentile above which samples are flagged.
            use_tracin: If True and PyTorch available, use TracIn-style gradient tracing.
        """
        self.n_loo_samples = n_loo_samples
        self.influence_percentile = influence_percentile
        self.use_tracin = use_tracin

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        model=None,
    ) -> dict:
        """
        Run influence function analysis.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label array (required for influence estimation).
            model: Optional sklearn classifier.

        Returns:
            Dictionary with influence function detection results.
        """
        n_samples = X.shape[0]

        if y is None:
            return self._empty_result(n_samples, "No labels provided for influence analysis.")

        if len(np.unique(y)) < 2:
            return self._empty_result(n_samples, "Need at least 2 classes for influence analysis.")

        try:
            # Try TracIn-style gradient influence first (if PyTorch available)
            if self.use_tracin:
                try:
                    return self._tracin_influence(X, y)
                except Exception as e:
                    logger.warning(f"TracIn failed ({e}), falling back to LOO.")

            # Fallback to LOO-based influence estimation
            return self._loo_influence(X, y, model)

        except Exception as e:
            logger.error(f"Influence function analysis failed: {e}")
            return self._empty_result(n_samples, f"Analysis failed: {e}")

    def _tracin_influence(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        TracIn-style gradient-based influence estimation using PyTorch.

        Trains a small model, records per-sample gradient norms at checkpoints,
        and computes a self-influence score = sum of ||grad_i||^2 across checkpoints.
        Samples with abnormally high self-influence are potentially poisoned.
        """
        import torch
        import torch.nn as nn

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Select device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Simple linear model for fast gradient computation
        class InfluenceModel(nn.Module):
            def __init__(self, n_in, n_out):
                super().__init__()
                self.fc1 = nn.Linear(n_in, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, n_out)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = InfluenceModel(n_features, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.LongTensor(y.astype(int)).to(device)

        # Track per-sample gradient norms across checkpoints
        n_checkpoints = 5
        n_epochs = 20
        checkpoint_epochs = np.linspace(0, n_epochs - 1, n_checkpoints, dtype=int)
        self_influence = np.zeros(n_samples)

        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X_t)
            losses = criterion(outputs, y_t)
            total_loss = losses.mean()
            total_loss.backward()
            optimizer.step()

            if epoch in checkpoint_epochs:
                # Compute per-sample gradient norms at this checkpoint
                for i in range(n_samples):
                    model.zero_grad()
                    out_i = model(X_t[i:i+1])
                    loss_i = criterion(out_i, y_t[i:i+1]).squeeze()
                    loss_i.backward()

                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    self_influence[i] += grad_norm

        # Normalize
        si_min, si_max = self_influence.min(), self_influence.max()
        if si_max > si_min:
            scores = (self_influence - si_min) / (si_max - si_min)
        else:
            scores = np.zeros(n_samples)

        threshold = np.percentile(scores, self.influence_percentile)
        flagged = scores > threshold
        flagged_indices = np.where(flagged)[0].tolist()

        # Compute influence ranking
        influence_ranking = np.argsort(scores)[::-1][:20].tolist()

        return {
            "layer": "influence",
            "layer_name": "TracIn Gradient Influence Analysis",
            "method": "tracin",
            "n_samples": int(n_samples),
            "n_checkpoints": n_checkpoints,
            "threshold": float(threshold),
            "flagged_indices": flagged_indices,
            "n_flagged": int(flagged.sum()),
            "flagged_ratio": float(flagged.sum() / n_samples),
            "scores": scores.tolist(),
            "is_flagged": flagged.tolist(),
            "top_influential": influence_ranking,
            "self_influence_raw": self_influence.tolist(),
        }

    def _loo_influence(
        self, X: np.ndarray, y: np.ndarray, model=None
    ) -> dict:
        """
        Leave-One-Out (LOO) influence estimation.

        For each sample i, approximate how much the model's loss changes
        when sample i is removed. Uses a fast approximation: train once
        on all data, then measure prediction confidence shift for each sample.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict

        n_samples = X.shape[0]

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1, oob_score=True,
            )
            model.fit(X, y)

        # Get prediction probabilities (OOB if available)
        try:
            if hasattr(model, 'oob_decision_function_'):
                proba = model.oob_decision_function_
            else:
                proba = model.predict_proba(X)
        except Exception:
            proba = model.predict_proba(X)

        # Influence score = 1 - P(correct label)
        # High influence = model is uncertain about this sample = potentially crafted
        influence_scores = np.zeros(n_samples)
        for i in range(n_samples):
            correct_class = int(y[i])
            if correct_class < proba.shape[1]:
                influence_scores[i] = 1.0 - proba[i, correct_class]
            else:
                influence_scores[i] = 1.0

        # Subsample for more expensive LOO on most suspicious candidates
        n_loo = min(self.n_loo_samples, n_samples)
        candidate_indices = np.argsort(influence_scores)[::-1][:n_loo]

        loo_scores = np.zeros(n_samples)
        loo_scores[:] = influence_scores  # Start with proxy scores

        for idx in candidate_indices[:min(50, n_loo)]:
            # Quick LOO: retrain without this sample and measure accuracy shift
            mask = np.ones(n_samples, dtype=bool)
            mask[idx] = False
            try:
                loo_model = RandomForestClassifier(
                    n_estimators=50, random_state=42, n_jobs=-1,
                )
                loo_model.fit(X[mask], y[mask])
                pred_with = model.predict(X[idx:idx+1])[0]
                pred_without = loo_model.predict(X[idx:idx+1])[0]

                # If removing this sample changes the prediction, it's influential
                if pred_with != pred_without:
                    loo_scores[idx] = max(loo_scores[idx], 0.8)

                # Also check probability shift
                prob_without = loo_model.predict_proba(X[idx:idx+1])[0]
                correct_class = int(y[idx])
                if correct_class < prob_without.shape[0]:
                    prob_shift = abs(
                        proba[idx, correct_class] - prob_without[correct_class]
                    )
                    loo_scores[idx] = max(loo_scores[idx], min(1.0, prob_shift * 3))
            except Exception:
                pass

        # Normalize
        s_min, s_max = loo_scores.min(), loo_scores.max()
        if s_max > s_min:
            scores = (loo_scores - s_min) / (s_max - s_min)
        else:
            scores = np.zeros(n_samples)

        threshold = np.percentile(scores, self.influence_percentile)
        flagged = scores > threshold
        flagged_indices = np.where(flagged)[0].tolist()

        influence_ranking = np.argsort(scores)[::-1][:20].tolist()

        return {
            "layer": "influence",
            "layer_name": "Leave-One-Out Influence Analysis",
            "method": "loo",
            "n_samples": int(n_samples),
            "n_loo_evaluated": int(min(50, n_loo)),
            "threshold": float(threshold),
            "flagged_indices": flagged_indices,
            "n_flagged": int(flagged.sum()),
            "flagged_ratio": float(flagged.sum() / n_samples),
            "scores": scores.tolist(),
            "is_flagged": flagged.tolist(),
            "top_influential": influence_ranking,
        }

    def _empty_result(self, n_samples: int, message: str) -> dict:
        return {
            "layer": "influence",
            "layer_name": "Influence Function Analysis",
            "method": "none",
            "n_samples": n_samples,
            "flagged_indices": [],
            "n_flagged": 0,
            "flagged_ratio": 0.0,
            "scores": [0.0] * n_samples,
            "is_flagged": [False] * n_samples,
            "top_influential": [],
            "message": message,
        }
