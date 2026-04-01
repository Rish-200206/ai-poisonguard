"""
Pydantic Models — Request/Response Schemas for AI PoisonGuard API
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


class AnalysisResponse(BaseModel):
    """Full analysis response from the detection pipeline."""
    status: str = Field(description="Overall status: clean, suspicious, compromised")
    risk_score: float = Field(description="Composite risk score 0-100")
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    domain_profile: str = Field(description="Domain profile used for analysis")
    n_samples: int = Field(description="Total samples analysed")
    n_flagged: int = Field(description="Total unique flagged samples (ensemble threshold met)")
    n_warnings: int = Field(default=0, description="Samples with single-layer flags (below ensemble threshold)")
    flagged_ratio: float = Field(description="Fraction of samples flagged")
    ensemble_threshold: int = Field(default=2, description="Minimum layers required to confirm a sample as poisoned")
    summary: str = Field(description="Human-readable summary")

    # Per-layer results (6 layers)
    statistical: Optional[dict] = Field(default=None, description="Layer 1 results")
    spectral: Optional[dict] = Field(default=None, description="Layer 2 results")
    clustering: Optional[dict] = Field(default=None, description="Layer 3 results")
    art: Optional[dict] = Field(default=None, description="Layer 4 ART results")
    influence: Optional[dict] = Field(default=None, description="Layer 5 Influence Function results")
    backdoor: Optional[dict] = Field(default=None, description="Layer 6 Backdoor Trigger results")

    # Aggregated data for visualisation
    scatter_data: Optional[list] = Field(default=None, description="2D coords for scatter plot")
    flagged_samples: Optional[list] = Field(default=None, description="Detailed flagged sample list")
    influence_scores: Optional[list] = Field(default=None, description="Top-N influence scores")
    label_heatmap_data: Optional[dict] = Field(default=None, description="Heatmap data")

    # Metadata
    profile_info: Optional[dict] = Field(default=None, description="Domain profile details")
    model_info: Optional[dict] = Field(default=None, description="Uploaded model info if any")
    analysis_time: Optional[float] = Field(default=None, description="Analysis time in seconds")


class ProfileResponse(BaseModel):
    """Domain profile listing."""
    profiles: list = Field(description="Available domain profiles")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    art_available: bool = False
    umap_available: bool = False
    hdbscan_available: bool = False
