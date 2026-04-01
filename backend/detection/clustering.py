"""
Layer 3: Activation Clustering — PyTorch Shadow Model + UMAP + KMeans

Detects training data poisoning via:
1. Training a shadow model (PyTorch MLP or sklearn fallback) on the data.
2. Extracting hidden-layer activations (penultimate layer outputs).
3. Reducing to 2D with UMAP for visualisation.
4. Clustering with KMeans and analyzing cluster-label purity.
5. Flagging low-purity clusters as potentially poisoned.

Reference: Chen et al. "Detecting Backdoor Attacks on Deep Neural
Networks by Activation Clustering." (2019)
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Optional

logger = logging.getLogger(__name__)


def _check_umap():
    try:
        import umap
        return True
    except ImportError:
        return False


def _check_hdbscan():
    try:
        import hdbscan
        return True
    except ImportError:
        return False


def _check_torch():
    try:
        import torch
        import torch.nn as nn
        return True
    except ImportError:
        return False


class ClusteringDetector:
    """PyTorch MLP + UMAP + KMeans activation clustering poisoning detector."""

    def __init__(
        self,
        n_clusters: int = 5,
        purity_threshold: float = 0.85,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        shadow_n_estimators: int = 100,
        use_hdbscan: bool = True,
    ):
        """
        Args:
            n_clusters: Number of clusters for KMeans.
            purity_threshold: Cluster label purity threshold.
                Clusters below this are flagged suspicious.
            umap_n_neighbors: UMAP n_neighbors parameter.
            umap_min_dist: UMAP min_dist parameter.
            shadow_n_estimators: Number of trees in fallback RandomForest.
            use_hdbscan: If True and available, use HDBSCAN; else KMeans.
        """
        self.n_clusters = n_clusters
        self.purity_threshold = purity_threshold
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.shadow_n_estimators = shadow_n_estimators
        self.use_hdbscan = use_hdbscan

    def detect(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Run activation clustering detection.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label array (required for shadow model + purity analysis).

        Returns:
            Dictionary with clustering detection results.
        """
        n_samples, n_features = X.shape

        # Step 1: Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 2: Train shadow model and extract activations
        # Prefer PyTorch MLP (MVP requirement), fallback to RandomForest
        shadow_model_type = "none"
        if y is not None and len(np.unique(y)) >= 2:
            if _check_torch():
                try:
                    activations = self._extract_pytorch_activations(X_scaled, y)
                    shadow_model_type = "PyTorch MLP"
                    logger.info("Using PyTorch MLP shadow model for activation extraction")
                except Exception as e:
                    logger.warning(f"PyTorch shadow model failed ({e}), falling back to RandomForest")
                    activations = self._extract_rf_activations(X_scaled, y)
                    shadow_model_type = "RandomForest (fallback)"
            else:
                activations = self._extract_rf_activations(X_scaled, y)
                shadow_model_type = "RandomForest (torch unavailable)"
                logger.info("PyTorch not available, using RandomForest shadow model")
        else:
            activations = X_scaled

        # Step 3: UMAP dimensionality reduction to 2D
        coords_2d = self._reduce_dimensions(activations)

        # Step 4: Clustering — use HDBSCAN if available, else KMeans
        used_hdbscan = False
        cluster_centers = None

        if self.use_hdbscan and _check_hdbscan() and n_samples > 50:
            import hdbscan as hdb
            clusterer = hdb.HDBSCAN(
                min_cluster_size=max(5, n_samples // 50),
                min_samples=3,
                metric="euclidean",
            )
            cluster_labels = clusterer.fit_predict(coords_2d)
            # HDBSCAN labels: -1 = noise. Remap to valid cluster IDs.
            unique_clusters = sorted(set(cluster_labels))
            actual_k = len([c for c in unique_clusters if c >= 0])
            if actual_k < 2:
                # HDBSCAN found too few clusters — fall back to KMeans
                actual_k = min(self.n_clusters, n_samples)
                kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(coords_2d)
                cluster_centers = kmeans.cluster_centers_
            else:
                used_hdbscan = True
                # Compute cluster centers manually for HDBSCAN
                all_clusters = sorted(set(cluster_labels))
                cluster_centers = np.array([
                    coords_2d[cluster_labels == c].mean(axis=0)
                    if c >= 0 else coords_2d.mean(axis=0)
                    for c in all_clusters
                ])
                actual_k = len(all_clusters)
        else:
            actual_k = min(self.n_clusters, n_samples)
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(coords_2d)
            cluster_centers = kmeans.cluster_centers_

        # Step 5: Compute silhouette score
        sil_score = -1.0
        if actual_k >= 2 and n_samples > actual_k:
            try:
                sil_score = float(silhouette_score(coords_2d, cluster_labels))
            except Exception:
                sil_score = -1.0

        # Step 6: Analyze cluster purity and flag poisoned clusters
        cluster_analysis = []
        flagged_indices = []
        scores = np.zeros(n_samples)

        # Use actual unique cluster labels (HDBSCAN can produce -1, 0, 1, …)
        unique_cluster_ids = sorted(set(cluster_labels.tolist()))

        for ci, c in enumerate(unique_cluster_ids):
            mask = cluster_labels == c
            cluster_size = int(mask.sum())
            cluster_indices = np.where(mask)[0].tolist()

            # Skip empty clusters
            if cluster_size == 0:
                continue

            purity = 1.0
            dominant_label = None
            label_dist = {}

            if y is not None and cluster_size > 0:
                cluster_labels_y = y[mask]
                unique, counts = np.unique(cluster_labels_y, return_counts=True)
                dominant_idx = counts.argmax()
                dominant_label = int(unique[dominant_idx]) if isinstance(
                    unique[dominant_idx], (int, np.integer)
                ) else str(unique[dominant_idx])
                purity = float(counts[dominant_idx] / cluster_size)
                label_dist = {
                    str(k): int(v) for k, v in zip(unique, counts)
                }

            # Noise cluster (HDBSCAN label -1) is always suspicious
            is_suspicious = purity < self.purity_threshold or c == -1

            # Score samples in suspicious clusters higher
            if is_suspicious:
                impurity_score = 1.0 - purity
                scores[mask] = impurity_score
                flagged_indices.extend(cluster_indices)

            # Distance from cluster center as secondary score
            if cluster_centers is not None and ci < len(cluster_centers):
                center = cluster_centers[ci]
            else:
                center = coords_2d[mask].mean(axis=0)

            distances = np.linalg.norm(coords_2d[mask] - center, axis=1)
            if len(distances) > 0 and distances.max() > 0:
                dist_scores = distances / distances.max()
                if is_suspicious:
                    scores[mask] = 0.6 * scores[mask] + 0.4 * dist_scores

            cluster_analysis.append({
                "cluster_id": int(c),
                "size": cluster_size,
                "purity": round(purity, 4),
                "dominant_label": dominant_label,
                "label_distribution": label_dist,
                "is_suspicious": bool(is_suspicious),
                "center": [float(x) for x in center],
                "indices": cluster_indices,
            })

        # Normalize scores
        if scores.max() > 0:
            scores = scores / scores.max()

        is_flagged = np.array(
            [i in flagged_indices for i in range(n_samples)]
        )

        clustering_method = "HDBSCAN" if used_hdbscan else "KMeans"
        dim_method = "UMAP" if _check_umap() else "PCA"

        return {
            "layer": "clustering",
            "layer_name": f"{dim_method} + {clustering_method} Activation Clustering",
            "n_samples": int(n_samples),
            "n_clusters": int(actual_k),
            "purity_threshold": self.purity_threshold,
            "silhouette_score": round(sil_score, 4),
            "flagged_indices": sorted(set(flagged_indices)),
            "n_flagged": int(is_flagged.sum()),
            "flagged_ratio": float(is_flagged.sum() / n_samples),
            "scores": scores.tolist(),
            "is_flagged": is_flagged.tolist(),
            "umap_coords": coords_2d.tolist(),
            "cluster_labels": cluster_labels.tolist(),
            "cluster_analysis": cluster_analysis,
            "used_umap": _check_umap(),
            "used_hdbscan": used_hdbscan,
            "clustering_method": clustering_method,
            "shadow_model": shadow_model_type,
        }

    def _extract_pytorch_activations(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Train a PyTorch MLP shadow model and extract penultimate-layer
        activations for activation clustering analysis.

        Architecture: Input → Linear(64) → ReLU → Linear(32) → ReLU → Output
        Activations are extracted from the 32-dim penultimate layer.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Select device: MPS (Apple Silicon), CUDA, or CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        logger.info(f"PyTorch shadow model using device: {device}")

        # Define shadow MLP
        class ShadowMLP(nn.Module):
            def __init__(self, in_features, n_classes):
                super().__init__()
                self.layer1 = nn.Linear(in_features, 64)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(0.2)
                self.layer2 = nn.Linear(64, 32)
                self.relu2 = nn.ReLU()
                self.output = nn.Linear(32, n_classes)

            def forward(self, x):
                x = self.dropout1(self.relu1(self.layer1(x)))
                x = self.relu2(self.layer2(x))
                return self.output(x)

            def get_activations(self, x):
                """Extract penultimate layer (32-dim) activations."""
                x = self.dropout1(self.relu1(self.layer1(x)))
                x = self.relu2(self.layer2(x))
                return x

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y.astype(int)).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(256, n_samples), shuffle=True)

        # Train shadow model
        model = ShadowMLP(n_features, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        n_epochs = min(30, max(10, 5000 // n_samples))  # Adaptive epochs
        for epoch in range(n_epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Extract activations
        model.eval()
        with torch.no_grad():
            activations = model.get_activations(X_tensor).cpu().numpy()

        return activations

    def _extract_rf_activations(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Fallback: Train a shadow RandomForest and extract leaf-index activations.

        Uses leaf node indices as the activation representation —
        this captures the model's learned decision structure,
        analogous to hidden-layer activations in deep networks.
        """
        rf = RandomForestClassifier(
            n_estimators=self.shadow_n_estimators,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
        )
        rf.fit(X, y)

        # Extract leaf indices as activation representation
        leaf_indices = rf.apply(X)  # (n_samples, n_estimators)

        return leaf_indices.astype(np.float64)

    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """Reduce to 2D using UMAP (preferred) or PCA (fallback)."""
        n_samples = X.shape[0]

        if _check_umap() and n_samples > 10:
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(self.umap_n_neighbors, n_samples - 1),
                    min_dist=self.umap_min_dist,
                    random_state=42,
                    metric="euclidean",
                )
                return reducer.fit_transform(X)
            except Exception:
                pass  # Fall through to PCA

        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, X.shape[1]))
        coords = pca.fit_transform(X)
        if coords.shape[1] == 1:
            coords = np.column_stack((coords, np.zeros(n_samples)))
        return coords
