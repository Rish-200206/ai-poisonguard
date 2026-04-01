"""
AI PoisonGuard — Multi-Layer Poisoning Detection Engine

Layer 1: Statistical Analysis (Z-score + IQR)
Layer 2: Spectral Signature Analysis (SVD-based)
Layer 3: Activation Clustering (UMAP + KMeans/HDBSCAN shadow model)
Layer 4: IBM ART Integration (validation layer)
Layer 5: Influence Function Analysis (TracIn / LOO)
Layer 6: Backdoor Trigger Scanning (Neural Cleanse approximation)
"""

from .statistical import StatisticalDetector
from .spectral import SpectralDetector
from .clustering import ClusteringDetector, _check_hdbscan
from .influence import InfluenceFunctionDetector
from .backdoor import BackdoorTriggerDetector

__all__ = [
    "StatisticalDetector",
    "SpectralDetector",
    "ClusteringDetector",
    "InfluenceFunctionDetector",
    "BackdoorTriggerDetector",
    "_check_hdbscan",
]
